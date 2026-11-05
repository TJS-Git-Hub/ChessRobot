"""
下棋机器人中枢主程序 (main.py)
==============================
架构：视觉感知(独立线程) → move_queue → 逻辑决策(主线程) → 硬件控制(主线程)

坐标系约定（黑方第一视角，机械臂在棋盘正后方）：
  - a 列（索引 0）在最右侧，i 列（索引 8）在最左侧
  - row 0 最近（黑方底线），row 9 最远（红方底线）
  - 视觉输出 "a0" 格式，游戏内部 (x,y) 纯数字，硬件接收 "a0" 格式

数据流：
  最新相机帧 → ONNX分类/外观复核 → 稳态判定 → 走棋提议 → 游戏确认 → 动作后新帧复核
"""

import os
import sys
import time
import threading
import queue

import cv2

# ---- 三层模块导入 ----
from localization.onnx_vision_adapter import ONNXVisionAdapter
from game.board import ChessBoard


# ============================================================
# 1. 统一坐标适配层
# ============================================================
class Coord:
    """三套坐标体系之间的无损双向转换

    视觉格式  "a0":       列字母(a-i) + 行数字(0-9)
    游戏格式  (x, y):     整数元组, x=0..8 列索引(0=a), y=0..9 行索引
    游戏走法  "x1y1x2y2": 4位纯数字 — board.parse_move() 的输入格式
    硬件格式  "a0":       与视觉相同 — control.board_to_physical() 消费此格式
    """
    COL_CHARS = "abcdefghi"
    _col_to_idx = {c: i for i, c in enumerate(COL_CHARS)}

    @classmethod
    def vision_to_game(cls, coord: str):
        """ "a0" → (0, 0) """
        from localization.vision_contract import parse_coordinate
        return parse_coordinate(coord)

    @classmethod
    def game_to_vision(cls, x: int, y: int):
        """ (0, 0) → "a0" """
        from localization.vision_contract import coordinate
        return coordinate(x,y)

    @classmethod
    def vision_move_to_game_move(cls, from_vis: str, to_vis: str):
        """ ("e6", "e4") → "4644" """
        x1, y1 = cls.vision_to_game(from_vis)
        x2, y2 = cls.vision_to_game(to_vis)
        return f"{x1}{y1}{x2}{y2}"

    @classmethod
    def game_move_to_vision_move(cls, move_str: str):
        """ "4644" → ("e6", "e4") """
        if not isinstance(move_str,str) or len(move_str)!=4 or any(ch not in '0123456789' for ch in move_str):
            raise ValueError('Game move must have four ASCII coordinate digits')
        x1, y1, x2, y2 = (int(c) for c in move_str)
        return cls.game_to_vision(x1, y1), cls.game_to_vision(x2, y2)


# ============================================================
# 2. 视觉适配器 — 将独立视觉模块改造成可被主线程调度的组件
# ============================================================
# ============================================================
# 3. 中枢调度器
# ============================================================
class ChessRobotOrchestrator:
    """Production orchestration; injectable camera/vision/robot for offline tests.

    Default is observation only. Hardware creation is lazy and requires an
    explicit flag, verified vision calibration and a fresh pickup transaction.
    """
    def __init__(self,model_path=None,*,arm=None,camera=None,vision=None,board=None,ai=None,
                 enable_hardware=False,observation_only=True,calibration=None,hand_model=None,
                 camera_index=2,headless=False,initialize_from_vision=False,red_turn=True,
                 pickup_timeout=5.,verification_timeout=8.):
        from localization.live_vision import LatestCamera
        from localization.offset_measurement import VisionCalibration
        self.stop_event=threading.Event();self.move_queue=queue.Queue();self.vision_ready=threading.Event()
        self.board=board if board is not None else ChessBoard();self.ai=ai
        self.arm=arm;self.enable_hardware=enable_hardware;self.observation_only=observation_only
        self.calibration=calibration or VisionCalibration.from_config()
        self.initialize_from_vision=initialize_from_vision;self.initial_red_turn=red_turn
        self.pickup_timeout=pickup_timeout;self.verification_timeout=verification_timeout
        self.headless=headless;self.vision_thread=None;self._shutdown=False
        self.last_verification=None;self.error=None
        if enable_hardware and (not self.calibration.calibrated or not self.calibration.compatible_with_robot()):
            raise ValueError('Hardware mode needs a verified calibration matching the existing robot grid')
        if not observation_only and arm is None and not enable_hardware:
            raise ValueError('Autoplay needs an explicit fake robot or --hardware')
        if vision is not None:
            self.vision=vision;self.move_queue=vision.move_queue;self.vision_ready=vision.ready_event
        elif model_path and model_path.endswith('.pt'):
            from localization.legacy_vision_adapter import VisionAdapter
            if not os.path.isfile(model_path):raise ValueError('Legacy YOLO weight file missing')
            if not observation_only:raise ValueError('Legacy YOLO observation remains available; automatic control requires Gen4 verification')
            self.vision=VisionAdapter(model_path,self.move_queue,self.vision_ready)
        else:
            self.vision=ONNXVisionAdapter(move_queue=self.move_queue,ready_event=self.vision_ready,
                calibration=self.calibration,hand_model=hand_model,require_hand_check=True,headless=headless,allow_control=enable_hardware)
        legacy=bool(model_path and model_path.endswith('.pt'))
        if legacy:
            from localization.legacy_vision_adapter import VideoStream
        self.vs=camera if camera is not None else (VideoStream(src=camera_index) if legacy else LatestCamera(camera_index,1920,1080))
        self.piece_count=32

    def _vision_thread_func(self):
        try:self.vision.run(self.vs)
        except Exception as exc:self.error=str(exc);print('Vision failed:',exc)
        finally:
            self.stop_event.set()
            if self.arm is not None:
                try:self.arm.request_stop()
                except Exception as exc:self.error='Vision stopped; robot stop request failed: '+str(exc)

    def start_vision(self):
        if self.vision_thread is not None:return
        self.vs.start()
        self.vision_thread=threading.Thread(target=self._vision_thread_func,daemon=True,name='VisionThread')
        self.vision_thread.start()

    def _wait_for_human_move(self):
        while not self.stop_event.is_set():
            try:event=self.move_queue.get(timeout=.1)
            except queue.Empty:continue
            try:
                self.vision.session.commit_human_move(event.event_id,self.board)
                self.piece_count=sum(p is not None for row in self.board.snapshot() for p in row)
                return True
            except (ValueError,AttributeError) as exc:
                if hasattr(event,'event_id'):self.vision.session.reject_move(event.event_id,str(exc))
                print('Move rejected; restore/reconcile physical board:',exc)
                return False
        return False

    def _ensure_arm(self):
        if self.arm is None:
            if not self.enable_hardware:raise RuntimeError('Real hardware not enabled')
            from control import ChessArmController
            self.arm=ChessArmController(auto_home=False)
        return self.arm

    def _robot_move(self):
        from localization.vision_session import VerificationStatus
        if self.observation_only:raise RuntimeError('Observation-only mode cannot command a robot')
        if self.ai is None:
            from game.ai import ChessAI
            self.ai=ChessAI(self.board)
        move=self.ai.get_best_move()
        if not move:return False
        if not self.board.is_move_valid(move):
            self.error='AI move is illegal';self.stop_event.set();return False
        source,target=Coord.game_move_to_vision_move(move)
        x1,y1,x2,y2=map(int,move)
        capture=self.board.snapshot()[y2][x2] is not None
        try:
            if self.stop_event.is_set():raise RuntimeError('System stopped before pickup preparation')
            plan=self.vision.prepare_action(source,target,self.board,timeout=self.pickup_timeout)
            if self.stop_event.is_set():raise RuntimeError('System stopped before robot initialization')
            arm=self._ensure_arm()
            if self.stop_event.is_set():raise RuntimeError('System stopped before motor command')
            if self.vision.clock()>plan.get('expires_at',float('inf')):
                raise RuntimeError('Pickup measurement expired before motor command')
            start_offset=plan['offsets'][source]
            if capture:
                success=arm.capture(source,target,start_offset=start_offset,target_offset=plan['offsets'][target])
            else:
                success=arm.pick_and_place(source,target,start_offset=start_offset,end_offset=(0.,0.))
            if success is not True:raise RuntimeError('Robot did not report successful command completion')
            packet=self.vs.read()
            barrier=packet.frame_id if packet is not None and hasattr(packet,'frame_id') else None
            self.vision.finish_action(plan['token'],barrier)
            self.last_verification=self.vision.post_move_verification(token=plan['token'],
                timeout=self.verification_timeout,stop_event=self.stop_event)
            if self.last_verification['status']!=VerificationStatus.SUCCESS.value:
                raise RuntimeError('Post-action verification: '+str(self.last_verification))
            self.vision.session.commit_robot_move(plan['token'],self.board,move)
            self.piece_count=sum(p is not None for row in self.board.snapshot() for p in row)
            return True
        except Exception as exc:
            self.error=str(exc);self.stop_event.set();self.vision.session.invalidate('Action failed: '+str(exc))
            if self.arm is not None:
                try:self.arm.request_stop()
                except Exception as stop_error:self.error+='; stop request failed: '+str(stop_error)
            print('Stopped; physical board needs reconciliation:',exc)
            return False

    def run(self):
        try:
            self.start_vision()
            deadline=time.monotonic()+180
            while not self.vision_ready.wait(.1):
                if self.stop_event.is_set() or time.monotonic()>=deadline:
                    self.error='Vision initialization failed or timed out';return
            if self.observation_only:
                print('VISION OBSERVATION ONLY: no robot actions. Q/Esc or UI stop to exit.')
                while not self.stop_event.wait(.1):pass
                return
            self.vision.session.synchronize_game(self.board,initialize=self.initialize_from_vision,
                                                red_turn=self.initial_red_turn)
            while not self.stop_event.is_set():
                if self.board.check_game_over():break
                if self.board.red_turn:
                    self._wait_for_human_move()
                elif not self._robot_move():break
        except Exception as exc:
            self.error=str(exc);self.stop_event.set();print('System stopped:',exc)
        finally:self.shutdown()

    def shutdown(self):
        if self._shutdown:return
        self._shutdown=True;self.stop_event.set()
        cleanup=[]
        try:
            if hasattr(self.vision,'close'):self.vision.close()
            else:self.vision.stopped=True
        except Exception as exc:cleanup.append('vision: '+str(exc))
        if self.arm is not None:
            for action in (self.arm.request_stop,self.arm.shutdown):
                try:action()
                except Exception as exc:cleanup.append('robot: '+str(exc))
        try:
            if hasattr(self.vs,'close'):self.vs.close()
            elif hasattr(self.vs,'stop'):self.vs.stop()
        except Exception as exc:cleanup.append('camera: '+str(exc))
        if self.vision_thread is not None and self.vision_thread is not threading.current_thread():
            self.vision_thread.join(timeout=2)
        if not self.headless:cv2.destroyAllWindows()
        if cleanup:self.error=(self.error+'; ' if self.error else '')+'; '.join(cleanup)


# ============================================================
# 4. 入口
# ============================================================
def __getattr__(name):
    if name in ('VisionAdapter','VideoStream'):
        from localization import legacy_vision_adapter
        return getattr(legacy_vision_adapter,name)
    raise AttributeError(name)


def main():
    import argparse
    from localization.offset_measurement import VisionCalibration
    parser=argparse.ArgumentParser(description='Gen4: observation by default; hardware requires explicit opt-in')
    parser.add_argument('--mode',choices=['observe','simulate','hardware'],default='observe')
    parser.add_argument('--camera',type=int,default=2)
    parser.add_argument('--hand-model')
    parser.add_argument('--calibration')
    parser.add_argument('--legacy-pt',action='store_true')
    parser.add_argument('--initialize-from-vision',action='store_true')
    parser.add_argument('--turn',choices=['red','black'],default='red')
    args=parser.parse_args()
    if args.mode=='simulate':
        from localization.simulation import run_smoke
        result=run_smoke()
        print(result)
        return 0 if result['passed'] else 1
    if not args.legacy_pt and not args.hand_model:parser.error('--hand-model is required for a live camera')
    calibration=VisionCalibration.load(args.calibration) if args.calibration else VisionCalibration.from_config()
    model=os.path.join(os.path.dirname(__file__),'recognition','models','best.pt') if args.legacy_pt else None
    app=ChessRobotOrchestrator(model,enable_hardware=args.mode=='hardware',observation_only=args.mode=='observe',
        calibration=calibration,hand_model=args.hand_model,camera_index=args.camera,
        initialize_from_vision=args.initialize_from_vision,red_turn=args.turn=='red')
    try:app.run()
    except KeyboardInterrupt:pass
    finally:app.shutdown()
    return 1 if app.error else 0


if __name__=='__main__':
    raise SystemExit(main())
