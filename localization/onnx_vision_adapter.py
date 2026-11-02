"""
基于第四代 ONNX 双阶段识别管线与毫米级偏置补偿的高精视觉适配器
(ONNXVisionAdapter)
====================================================================
特性：
  1. 自动化角点定位 (RTMPose) + 90 交叉点 100% 精准分类 (SwinV2 + 实体守恒仲裁)；
  2. 亚像素物理圆心拟合与毫米级偏置矢量计算 (robot_offset_mm)；
  3. 多帧投票防抖动稳态判定机制；
  4. 线程安全队列推送与机械臂运动期间的视觉抑制。
"""

import time
import threading
import queue
from typing import Dict, Optional, Tuple, List
import cv2
import numpy as np

from recognition.onnx_pipeline.chess_recognizer import ChessRecognizer
from localization.find_chess_new import VideoStream


class ONNXVisionAdapter:
    def __init__(self,
                 recognizer: Optional[ChessRecognizer] = None,
                 move_queue: Optional[queue.Queue] = None,
                 ready_event: Optional[threading.Event] = None):
        self.recognizer = recognizer if recognizer is not None else ChessRecognizer(enable_physics_refine=True)
        self.move_queue = move_queue if move_queue is not None else queue.Queue()
        self.ready_event = ready_event if ready_event is not None else threading.Event()

        self._lock = threading.Lock()
        self.suppress_detection = False
        self.stopped = False

        self.last_stable_grid: Optional[List[List[str]]] = None
        self.latest_offsets: Dict[str, Dict] = {}
        self.expected_piece_count = 32

        # 稳态判定窗口 (连续 3 帧识别一致视为稳态盘面)
        self.recent_grids: List[List[List[str]]] = []
        self.stable_window_size = 3

    @staticmethod
    def _grids_equal(g1: List[List[str]], g2: List[List[str]]) -> bool:
        if not g1 or not g2:
            return False
        for r in range(10):
            for c in range(9):
                if g1[r][c] != g2[r][c]:
                    return False
        return True

    def get_piece_offset(self, pos_str: str) -> Tuple[float, float]:
        """获取指定格点 ('a0'~'i9') 上棋子的机械臂毫米偏置 (robot_dx, robot_dy)"""
        with self._lock:
            info = self.latest_offsets.get(pos_str.lower())
            if info and "robot_offset_mm" in info:
                return info["robot_offset_mm"]
            return (0.0, 0.0)

    def reset_baseline(self, new_expected_count: Optional[int] = None):
        """机械臂动作完成后调用，重置稳态基准并丢弃运动过渡帧"""
        with self._lock:
            if new_expected_count is not None:
                self.expected_piece_count = new_expected_count
            self.last_stable_grid = None
            self.recent_grids.clear()
            self.ready_event.clear()
        print(f"🔄 [视觉] 基准已重置，预期盘面棋子数: {self.expected_piece_count}")

    def process_frame(self, frame: np.ndarray) -> Optional[Tuple[str, str]]:
        """处理单帧图像，维护稳态并解析走法"""
        with self._lock:
            if self.suppress_detection:
                return None

        # 推理全盘
        res = self.recognizer.recognize_full(frame, use_smoothing=True)
        current_grid = res["grid_chars"]
        current_offsets = res["piece_offsets"]

        with self._lock:
            self.latest_offsets = current_offsets

            # 写入平滑稳态窗口
            self.recent_grids.append(current_grid)
            if len(self.recent_grids) > self.stable_window_size:
                self.recent_grids.pop(0)

            # 校验稳态：窗口内所有帧盘面必须完全一致
            if len(self.recent_grids) < self.stable_window_size:
                return None
            for g in self.recent_grids[:-1]:
                if not self._grids_equal(g, current_grid):
                    return None  # 处于玩家手部移动或棋子扰动过渡期

            # 锁定初始盘面
            if self.last_stable_grid is None:
                self.last_stable_grid = [row.copy() for row in current_grid]
                print("🟢 [视觉] 基准盘面已锁定，等待人类走棋...")
                self.ready_event.set()
                return None

            # 差分对比：寻找人类(红方/大写)移动轨迹
            col_chars = "abcdefghi"
            lost = []
            gain = []

            for r in range(10):
                for c in range(9):
                    old_ch = self.last_stable_grid[r][c]
                    new_ch = current_grid[r][c]
                    pos_str = f"{col_chars[c]}{r}"

                    if old_ch != new_ch:
                        # 原位置有子，现变为空位 -> 起点
                        if old_ch not in ['.', 'x'] and new_ch in ['.', 'x']:
                            lost.append((pos_str, old_ch))
                        # 现位置为新子，或吃掉了对方黑子 -> 终点
                        elif new_ch not in ['.', 'x'] and (old_ch in ['.', 'x'] or old_ch.islower()):
                            gain.append((pos_str, new_ch))

            if len(lost) == 1 and len(gain) == 1:
                from_pos = lost[0][0]
                to_pos = gain[0][0]
                piece_moved = gain[0][1]
                print(f"\n🚀 [视觉] 精准捕获走法: {piece_moved} {from_pos} → {to_pos}")

                # 偏置上报
                from_off = current_offsets.get(from_pos, {}).get("robot_offset_mm", (0.0, 0.0))
                to_off = current_offsets.get(to_pos, {}).get("robot_offset_mm", (0.0, 0.0))
                print(f"   起点偏置: {from_off} mm, 终点落点偏置: {to_off} mm")

                self.last_stable_grid = [row.copy() for row in current_grid]
                self.move_queue.put((from_pos, to_pos))
                return from_pos, to_pos

        return None

    def run(self, vs: VideoStream):
        """视觉线程主循环"""
        print("🎯 [视觉] 第四代 ONNX 视觉感知线程已启动")
        cv2.namedWindow("ChessRobot Vision Monitor", cv2.WINDOW_NORMAL)

        while not self.stopped:
            if vs.stopped:
                print("🛑 收到底层硬件停止信号，视觉线程退出。")
                break

            frame = vs.read()
            if frame is None:
                time.sleep(0.01)
                continue

            try:
                self.process_frame(frame)
            except Exception as e:
                print(f"⚠️ [视觉处理异常]: {e}")

            # 绘制实时监控视窗
            display = frame.copy()
            cv2.putText(display, f"Pieces: {self.expected_piece_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("ChessRobot Vision Monitor", display)

            key = cv2.waitKey(20) & 0xFF
            if key == 27 or key == ord('q'):
                break

        cv2.destroyAllWindows()
