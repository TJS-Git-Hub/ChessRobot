"""Explicit software doubles. No camera, control, driver or motor imports."""
import copy
import threading
import time
import cv2
import numpy as np
from localization.live_vision import CameraFrame
from localization.vision_contract import board_to_grid, moved_grid, validate_grid


def draw_board(grid,offsets=None,code=0):
    offsets=offsets or {}
    image=np.full((500,450,3),(150,175,190),np.uint8)
    for x in np.linspace(50,400,9):cv2.line(image,(round(x),50),(round(x),450),(55,55,55),1)
    for y in np.linspace(50,450,10):cv2.line(image,(50,round(y)),(400,round(y)),(55,55,55),1)
    for r,row in enumerate(grid):
        for c,ch in enumerate(row):
            if ch=='.':continue
            delta=offsets.get(f'{chr(97+c)}{r}',(0,0))
            center=(round(50+c*350/8+delta[0]),round(50+r*400/9+delta[1]))
            cv2.circle(image,center,16,(232,235,235),-1,cv2.LINE_AA)
            cv2.circle(image,center,16,(55,55,55),1,cv2.LINE_AA)
            cv2.putText(image,ch.upper(),(center[0]-7,center[1]+6),0,.55,
                        (25,25,180) if ch.isupper() else (25,25,25),1,cv2.LINE_AA)
    image[0,0]=[code%256,(code//256)%256,0]
    return image


class FakeCamera:
    def __init__(self,grid,offsets=None,fps=30):
        self.lock=threading.RLock();self.stopped=False;self.packet=None;self.seq=0
        self.scenes={};self.code=0;self.fps=fps;self.freeze=False;self.disconnected=False
        self.actual_size=(450,500);self.capture_fps=fps;self.preview_fps=fps;self.error=None
        self._stop=threading.Event();self.thread=threading.Thread(target=self._run,daemon=True)
        self.set_board(grid,offsets)

    def set_board(self,grid,offsets=None):
        with self.lock:
            self.grid=validate_grid(grid);self.offsets=copy.deepcopy(offsets or {})
            self.code+=1;self.scenes[self.code]=copy.deepcopy(self.grid)
            self.image=draw_board(self.grid,self.offsets,self.code)

    def start(self):
        if self.thread.ident is None:self.thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            with self.lock:
                if self.disconnected:self.packet=None
                elif not self.freeze:
                    self.seq+=1;self.packet=CameraFrame(self.seq,time.monotonic(),self.image.copy())
            self._stop.wait(1/self.fps)

    def read(self):
        with self.lock:return self.packet

    def close(self):
        self.stopped=True;self._stop.set()
        if self.thread.ident is not None:self.thread.join(1)


class FakeRecognizer:
    def __init__(self,camera):self.camera=camera;self.calls=0
    def predict_corners(self,image):return np.float32([[50,50],[400,50],[50,450],[400,450]]),np.full(4,.3)
    def warp_chessboard(self,image,corners):return image.copy(),np.eye(3)
    def classify_board(self,image):
        self.calls+=1;code=int(image[0,0,0])+int(image[0,0,1])*256
        with self.camera.lock:grid=copy.deepcopy(self.camera.scenes[code])
        return grid,np.full((10,9),.95).tolist(),'\n'.join(map(''.join,grid))
    def get_allowed_classes(self,*args):return list('KABNRCPkabnrcp')


class FakeRobot:
    def __init__(self,camera,*,failure=False,wrong_board=False,residual=(.2,-.2)):
        self.camera=camera;self.commands=[];self.failure=failure;self.wrong_board=wrong_board
        self.residual=residual;self.stopped=False;self.closed=False
    def _move(self,source,target,**kwargs):
        self.commands.append(dict(source=source,target=target,**kwargs))
        if self.failure:return False
        if not self.wrong_board:
            self.camera.set_board(moved_grid(self.camera.grid,source,target),{target:self.residual})
        return True
    def pick_and_place(self,source,target,**kwargs):return self._move(source,target,**kwargs)
    def capture(self,source,target,**kwargs):return self._move(source,target,**kwargs)
    def request_stop(self):self.stopped=True
    def shutdown(self):self.closed=True


class FirstLegalAI:
    def __init__(self,board):self.board=board
    def get_best_move(self):
        moves=self.board.get_all_legal_moves(self.board.red_turn)
        return moves[0] if moves else None


def wait_until(predicate,timeout=8.):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate():return True
        threading.Event().wait(.01)
    return False


def run_smoke():
    from main import ChessRobotOrchestrator
    from game.board import ChessBoard
    from localization.onnx_vision_adapter import ONNXVisionAdapter
    from localization.offset_measurement import VisionCalibration
    board=ChessBoard();camera=FakeCamera(board_to_grid(board.snapshot()))
    cal=VisionCalibration(calibrated=True,calibration_id='SIMULATION-ONLY-NOT-PHYSICAL')
    vision=ONNXVisionAdapter(FakeRecognizer(camera),calibration=cal,headless=True,allow_control=True)
    robot=FakeRobot(camera);app=ChessRobotOrchestrator(arm=robot,camera=camera,vision=vision,
        board=board,ai=FirstLegalAI(board),observation_only=False,headless=True,
        calibration=cal,verification_timeout=5.)
    result=dict(passed=False,simulated=True,hardware_accessed=False)
    try:
        app.start_vision()
        if not vision.ready_event.wait(5):raise RuntimeError('Initial visual board not confirmed')
        vision.session.synchronize_game(board)
        camera.set_board(moved_grid(camera.grid,'a6','a5'))
        if not wait_until(lambda:not app.move_queue.empty(),5):raise RuntimeError('Human move not observed')
        if not app._wait_for_human_move():raise RuntimeError('Human move not committed')
        if not app._robot_move():raise RuntimeError(app.error)
        result.update(passed=True,commands=robot.commands,verification=app.last_verification,
                      board_matches_camera=board_to_grid(board.snapshot())==camera.grid)
    except Exception as exc:result['error']=str(exc)
    finally:app.shutdown()
    return result
