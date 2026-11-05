"""Entrypoints and unknowns: no physical camera, driver or GUI display needed."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
import cv2
import numpy as np
from recognition.onnx_pipeline.chess_recognizer import ChessRecognizer,DICT_CATE_NAMES
from localization.simulation import draw_board

ROOT=Path(__file__).resolve().parents[1]


class TestUnknownContracts(unittest.TestCase):
    def test_unknown_is_not_encoded_as_empty_in_fen(self):
        recognizer=ChessRecognizer.__new__(ChessRecognizer)
        grid=[['.']*9 for _ in range(10)];grid[4][2]='x'
        with self.assertRaises(ValueError):recognizer.format_fen(grid)

    def test_invalid_corner_geometry_never_runs_classifier(self):
        rec=ChessRecognizer.__new__(ChessRecognizer)
        rec.predict_corners=Mock(return_value=(np.zeros((4,2)),np.ones(4)))
        rec.classify_board=Mock(side_effect=AssertionError('Cannot classify invalid geometry'))
        result=rec.recognize_full(np.full((500,450,3),150,np.uint8))
        self.assertTrue(result['rejection']);self.assertFalse(result['control_ready'])
        self.assertEqual(result['fen'],'');self.assertEqual(result['piece_offsets'],{})
        rec.classify_board.assert_not_called()

    def test_probability_smoothing_cannot_erase_latest_occlusion(self):
        rec=ChessRecognizer.__new__(ChessRecognizer)
        rec.classes_labels=list(DICT_CATE_NAMES);rec.label_2_short=DICT_CATE_NAMES;rec.enable_physics_refine=False
        old=np.zeros((90,16),np.float32);old[:,0]=1;old[0,0]=0;old[0,rec.classes_labels.index('red_pawn')]=1
        current=np.zeros_like(old);current[:,0]=1;current[0,0]=0;current[0,rec.classes_labels.index('other')]=1
        rec.recent_logits_buffer=[old.copy(),old.copy()]
        def classify(image):
            rec.recent_logits_buffer.append(current)
            grid=[['.']*9 for _ in range(10)];grid[0][0]='x'
            return grid,np.full((10,9),.95).tolist(),''
        rec.classify_board=classify
        grid,confidence,_=rec.classify_board_smoothed(np.zeros((500,450,3),np.uint8))
        self.assertEqual(grid[0][0],'x');self.assertEqual(confidence[0][0],0)

    def test_web_single_profile_function_without_gradio_server(self):
        import tools.web_demo as web
        grid=[['.']*9 for _ in range(10)];grid[5][2]='P'
        rec=ChessRecognizer.__new__(ChessRecognizer)
        rec.reset_buffer=Mock();rec.predict_corners=Mock(return_value=(np.float32([[50,50],[400,50],[50,450],[400,450]]),np.full(4,.3)))
        rec.warp_chessboard=lambda image,corners:(image,np.eye(3))
        rec.classify_board=lambda image:(grid,np.full((10,9),.95).tolist(),'')
        rec.get_allowed_classes=lambda *args:list('KABNRCPkabnrcp')
        with patch.object(web,'_recognizer',rec):
            result=web.process_image(cv2.cvtColor(draw_board(grid),cv2.COLOR_BGR2RGB))
        self.assertEqual(len(result),6);self.assertIn('P',result[2]);self.assertIn('unverified',result[4])


class TestEntrypointIsolation(unittest.TestCase):
    def test_hardware_gui_bootstrap_loads_inference_before_qt_without_devices(self):
        code="""
import sys,cv2
sys.argv=['ui_main.py','--mode','hardware']
cv2.VideoCapture=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No camera'))
import ui_main
assert 'hardware' not in sys.modules
assert 'control' not in sys.modules
from PyQt5.QtWidgets import QApplication
app=QApplication([])
assert 'torch' in sys.modules
print('HARDWARE_GUI_IMPORT_ONLY_OK')
"""
        env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1')
        run=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,text=True,capture_output=True,timeout=20)
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)

    def test_import_and_simulation_never_import_hardware_or_open_camera(self):
        code="""
import sys,importlib.abc,cv2
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname in ('control','hardware'):raise AssertionError('Hardware import forbidden: '+fullname)
sys.meta_path.insert(0,Guard())
cv2.VideoCapture=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Physical camera forbidden'))
from localization.simulation import run_smoke
r=run_smoke()
assert r['passed'],r
print('ISOLATED_SIMULATION_OK')
"""
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        run=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,text=True,capture_output=True,timeout=30)
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        self.assertIn('ISOLATED_SIMULATION_OK',run.stdout)

    def test_qt_gen4_click_keys_and_stop_do_not_require_legacy_fields(self):
        code="""
import sys,types,cv2
cv2.VideoCapture=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No camera'))
from ui_main import ChessRobotUI
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtCore import QEvent,Qt
app=QApplication([]);window=ChessRobotUI()
vision=types.SimpleNamespace(stopped=False,close=lambda:None)
source=types.SimpleNamespace(close=lambda:None)
import threading
window.orchestrator=types.SimpleNamespace(vision=vision,arm=None,vs=source,stop_event=threading.Event())
window.on_video_clicked(10,20)
window.keyPressEvent(QKeyEvent(QEvent.KeyPress,Qt.Key_1,Qt.NoModifier))
window.keyPressEvent(QKeyEvent(QEvent.KeyPress,Qt.Key_Space,Qt.NoModifier))
window.close()
assert vision.stopped
assert window.engine_thread.ident is None
# Stopping while the model/controller constructor is still loading must cancel
# startup; it must not start the camera after the window has been closed.
import ui_main
cancelled=ChessRobotUI();calls=[]
class DelayedConstruction:
 def __init__(self,*a,**k):
  self.arm=None;self.vision=vision;self.vs=source;self.stop_event=threading.Event();self.error=None
  cancelled.emergency_stop()
 def shutdown(self):calls.append('shutdown')
ui_main.ChessRobotOrchestrator=DelayedConstruction
cancelled.run_engine()
assert calls and all(x=='shutdown' for x in calls),calls
cancelled.close()
print('OFFSCREEN_GUI_OK')
"""
        env=dict(os.environ,QT_QPA_PLATFORM='offscreen',PYTHONDONTWRITEBYTECODE='1')
        run=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,text=True,capture_output=True,timeout=20)
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        self.assertIn('OFFSCREEN_GUI_OK',run.stdout)

    def test_image_and_video_entrypoints_use_files_only_and_refuse_control(self):
        from tools.vision_debug import main
        image=ROOT/'test photos/IMG_20261102_171009.jpg'
        self.assertTrue(image.is_file())
        with tempfile.TemporaryDirectory(prefix='chessrobot-offline-entry-') as temporary:
            directory=Path(temporary)
            with patch('cv2.VideoCapture',side_effect=AssertionError('Image must not open a camera')):
                code=main(['--image',str(image),'--headless','--output-dir',str(directory/'image')])
            self.assertIn(code,(0,2))
            import json
            payload=json.loads((directory/'image/image.json').read_text(encoding='utf8'))
            self.assertFalse(payload['stable']);self.assertFalse(payload['control_ready'])
            raw=cv2.imread(str(image));height=round(raw.shape[0]*640/raw.shape[1]/2)*2
            raw=cv2.resize(raw,(640,height))
            video=directory/'scene.avi'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(640,height))
            self.assertTrue(writer.isOpened())
            for _ in range(20):writer.write(raw)
            writer.release()
            original=cv2.VideoCapture
            def open_file(source,*args,**kwargs):
                self.assertIsInstance(source,str);self.assertEqual(source,str(video))
                return original(source,*args,**kwargs)
            with patch('cv2.VideoCapture',side_effect=open_file):
                self.assertEqual(main(['--video',str(video),'--headless','--output-dir',str(directory/'video')]),0)
            payload=json.loads((directory/'video/latest.json').read_text(encoding='utf8'))
            self.assertEqual(payload['source_kind'],'video');self.assertFalse(payload['control_ready'])


if __name__=='__main__':unittest.main()
