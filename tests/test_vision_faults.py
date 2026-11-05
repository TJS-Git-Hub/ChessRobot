"""Fault injection against production adapter and orchestrator; no devices."""
import copy
from concurrent.futures import Future
import threading
import time
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from game.board import ChessBoard
from localization.live_vision import CameraFrame,HandObservation
from localization.vision_contract import board_to_grid,moved_grid,grid_to_tensor
from localization.offset_measurement import VisionCalibration
from localization.camera_geometry import ImageGeometry
from localization.onnx_vision_adapter import ONNXVisionAdapter
from localization.simulation import FakeCamera,FakeRecognizer,FakeRobot,FirstLegalAI,wait_until


class Clock:
    def __init__(self):self.value=10.
    def __call__(self):return self.value


class TestAsynchronousSafety(unittest.TestCase):
    def blocked(self):
        self.camera=FakeCamera(board_to_grid(ChessBoard().snapshot()));self.rec=FakeRecognizer(self.camera)
        self.clock=Clock();self.entered=threading.Event();self.release=threading.Event()
        original=self.rec.classify_board
        def classify(image):
            self.entered.set();self.release.wait(3);return original(image)
        self.rec.classify_board=classify
        self.adapter=ONNXVisionAdapter(self.rec,clock=self.clock,headless=True)
        self.adapter.gate.settle_seconds=0
        self.addCleanup(self.adapter.close);self.addCleanup(self.release.set)
        self.adapter.process_frame(CameraFrame(1,10,self.camera.image.copy()),motion_reason='')
        self.assertTrue(self.entered.wait(1))

    def test_repeated_physical_frame_cannot_finish_a_stability_window(self):
        self.blocked()
        for _ in range(20):self.adapter.process_frame(CameraFrame(1,10,self.camera.image.copy()),motion_reason='')
        self.assertIsNone(self.adapter.get_snapshot());self.assertFalse(self.adapter.ready_event.is_set())

    def test_inflight_result_cannot_reactivate_after_motion_epoch(self):
        self.blocked();self.clock.value=10.1
        self.adapter.process_frame(CameraFrame(2,10.1,self.camera.image.copy()),motion_reason='hand')
        self.release.set();self.assertTrue(wait_until(lambda:self.adapter._pending[0].done(),1))
        self.clock.value=10.2
        self.adapter.process_frame(CameraFrame(3,10.2,self.camera.image.copy()),motion_reason='')
        self.assertIsNone(self.adapter.get_snapshot());self.assertTrue(self.adapter.move_queue.empty())
        self.assertIsNone(self.adapter.last_stable_grid)

    def test_timeout_clears_observations_and_reports_failure(self):
        self.blocked();self.clock.value=16
        self.adapter.process_frame(CameraFrame(2,16,self.camera.image.copy()),motion_reason='')
        self.assertEqual(self.adapter.error,'Classifier timed out')
        self.assertIsNone(self.adapter.get_snapshot())

    def test_inference_exception_does_not_publish_a_board(self):
        self.blocked();self.release.set()
        self.assertTrue(wait_until(lambda:self.adapter._pending[0].done(),1))
        failed=Future();failed.set_exception(RuntimeError('bad ONNX output'))
        old=self.adapter._pending
        self.adapter._pending=(failed,*old[1:]);self.clock.value=10.2
        self.adapter.process_frame(CameraFrame(2,10.2,self.camera.image.copy()),motion_reason='')
        self.assertIn('bad ONNX',self.adapter.error);self.assertIsNone(self.adapter.get_snapshot())

    def test_static_hand_check_expiry_and_disconnect_clear_freshness(self):
        camera=FakeCamera(board_to_grid(ChessBoard().snapshot()));clock=Clock()
        adapter=ONNXVisionAdapter(FakeRecognizer(camera),clock=clock,headless=True,require_hand_check=True)
        self.addCleanup(adapter.close)
        packet=CameraFrame(1,10,camera.image.copy())
        adapter.process_frame(packet,hand_record=HandObservation(1,9,9,0,0),motion_reason='')
        self.assertFalse(adapter.ready_event.is_set())
        adapter.process_packet(None)
        self.assertIsNone(adapter.get_snapshot())


class TestActionFailures(unittest.TestCase):
    def test_stop_after_pickup_plan_prevents_motor_command(self):
        app,camera,robot=self.setup_app()
        prepare=app.vision.prepare_action
        def stopped_plan(*args,**kwargs):
            plan=prepare(*args,**kwargs);app.stop_event.set();return plan
        app.vision.prepare_action=stopped_plan
        self.assertFalse(app._robot_move());self.assertEqual(robot.commands,[])
        self.assertTrue(robot.stopped)

    def test_camera_termination_requests_stop_during_inflight_action(self):
        app,camera,robot=self.setup_app()
        entered,release=threading.Event(),threading.Event()
        def blocking(*args,**kwargs):
            robot.commands.append({'inflight':True});entered.set();release.wait(3)
            return False
        def stop():robot.stopped=True;release.set()
        robot.pick_and_place=blocking;robot.request_stop=stop
        thread=threading.Thread(target=app._robot_move);thread.start()
        self.assertTrue(entered.wait(2))
        before=app.board.snapshot();camera.close()
        thread.join(2)
        self.assertFalse(thread.is_alive());self.assertTrue(robot.stopped)
        self.assertEqual(app.board.snapshot(),before);self.assertEqual(len(robot.commands),1)

    def setup_app(self,verified=True):
        from main import ChessRobotOrchestrator
        board=ChessBoard();board.red_turn=False
        camera=FakeCamera(board_to_grid(board.snapshot()))
        cal=VisionCalibration(calibrated=verified,calibration_id='SIMULATION-ONLY')
        vision=ONNXVisionAdapter(FakeRecognizer(camera),calibration=cal,headless=True,allow_control=True)
        robot=FakeRobot(camera)
        app=ChessRobotOrchestrator(arm=robot,camera=camera,vision=vision,board=board,ai=FirstLegalAI(board),
            calibration=cal,headless=True,observation_only=False,pickup_timeout=.6,verification_timeout=.6)
        self.addCleanup(app.shutdown);app.start_vision();self.assertTrue(vision.ready_event.wait(3))
        vision.session.synchronize_game(board)
        return app,camera,robot

    def test_camera_freeze_prevents_any_robot_command(self):
        app,camera,robot=self.setup_app();camera.freeze=True
        self.assertFalse(app._robot_move());self.assertEqual(robot.commands,[])
        self.assertTrue(app.stop_event.is_set())

    def test_unverified_physical_calibration_prevents_any_robot_command(self):
        app,camera,robot=self.setup_app(verified=False)
        self.assertFalse(app._robot_move());self.assertEqual(robot.commands,[])

    def test_illegal_ai_result_never_issues_a_command(self):
        app,camera,robot=self.setup_app()
        app.ai.get_best_move=lambda:'0009'
        self.assertFalse(app._robot_move());self.assertEqual(robot.commands,[])

    def test_stopping_vision_unblocks_pickup_and_prevents_actions(self):
        app,camera,robot=self.setup_app();camera.freeze=True
        thread=threading.Thread(target=app._robot_move);thread.start()
        self.assertTrue(wait_until(lambda:bool(app.vision.session.offset_request),.5))
        app.vision.close();thread.join(1)
        self.assertFalse(thread.is_alive());self.assertEqual(robot.commands,[])

    def test_capture_measures_both_pieces_and_commits_expected_full_board(self):
        from main import ChessRobotOrchestrator
        board=ChessBoard();grid=[['.']*9 for _ in range(10)]
        grid[0][4]='k';grid[9][3]='K';grid[3][0]='r';grid[5][0]='P'
        board.load_vision_grid(grid,False)
        camera=FakeCamera(grid,{'a3':(3,-2),'a5':(-2,2)})
        cal=VisionCalibration(calibrated=True,calibration_id='SIMULATION-ONLY')
        vision=ONNXVisionAdapter(FakeRecognizer(camera),calibration=cal,headless=True,allow_control=True)
        robot=FakeRobot(camera)
        ai=type('AI',(),{'get_best_move':lambda self:'0305'})()
        app=ChessRobotOrchestrator(arm=robot,camera=camera,vision=vision,board=board,ai=ai,calibration=cal,
                                  observation_only=False,headless=True,pickup_timeout=3,verification_timeout=4)
        self.addCleanup(app.shutdown);app.start_vision();self.assertTrue(vision.ready_event.wait(3))
        vision.session.synchronize_game(board)
        self.assertTrue(app._robot_move(),app.error)
        self.assertEqual(len(robot.commands),1)
        self.assertIn('target_offset',robot.commands[0]);self.assertGreater(robot.commands[0]['target_offset'][0],1.)
        self.assertEqual(sum(ch!='.' for row in camera.grid for ch in row),3)
        self.assertEqual(board_to_grid(board.snapshot()),camera.grid)


class TestCameraGeometry(unittest.TestCase):
    def test_piece_height_parallax_inverse_is_position_dependent(self):
        cal=VisionCalibration(camera_height_mm=800.,piece_height_mm=12.,camera_center_board_mm=(138.,153.5))
        nominal=cal.nominal(2,6)
        physical=(nominal-[50,50])*[276/350,307/400]+[2.,-3.]
        camera=np.array([138.,153.5]);projected=camera+(physical-camera)/(1-12/800)
        detected=projected*[350/276,400/307]+[50,50]
        board,robot=cal.offset_from_detection(nominal,detected)
        np.testing.assert_allclose(board,[2.,-3.],atol=1e-8)
        np.testing.assert_allclose(robot,[-2.,-3.],atol=1e-8)
        zero,_=cal.offset_from_detection(nominal,cal.surface_nominal(2,6))
        np.testing.assert_allclose(zero,[0.,0.],atol=1e-8)

    def test_partial_height_calibration_is_not_silently_assumed(self):
        with self.assertRaises(ValueError):VisionCalibration(camera_height_mm=800.)

    def calibration(self,distortion):
        return VisionCalibration(camera_matrix=((400.,0,225.),(0,400.,250.),(0,0,1)),
                                 distortion=distortion,image_size=(450,500))

    def test_zero_distortion_preserves_image_and_normalized_hand_regions(self):
        cal=self.calibration((0,0,0,0,0));geometry=ImageGeometry(cal)
        image=np.random.default_rng(3).integers(0,255,(500,450,3),dtype=np.uint8)
        np.testing.assert_array_equal(geometry.apply(image),image)
        record=HandObservation(1,10,10,1,0,(((.1,.2),(.3,.4),(.5,.6)),))
        transformed=geometry.hands(record,image.shape)
        np.testing.assert_allclose(transformed.regions,record.regions,atol=1e-7)
        self.assertEqual(transformed.frame_id,record.frame_id)

    def test_distorted_landmarks_map_to_correct_undistorted_points(self):
        cal=self.calibration((.1,-.03,0,0,0));k=np.array(cal.camera_matrix);d=np.array(cal.distortion)
        points=np.float32([[-.3,-.2,1],[.2,.25,1],[0,0,1]])
        projected,_=cv2.projectPoints(points,np.zeros(3),np.zeros(3),k,d)
        regions=tuple(map(tuple,projected.reshape(-1,2)/[450,500]))
        record=HandObservation(1,10,10,1,0,(regions,))
        corrected=ImageGeometry(cal).hands(record,(500,450,3))
        expected=(points[:,:2]*[400,400]+[225,250])/[450,500]
        np.testing.assert_allclose(corrected.regions[0],expected,atol=1e-5)

    def test_wrong_resolution_and_partial_intrinsics_fail_closed(self):
        geometry=ImageGeometry(self.calibration((0,0,0,0,0)))
        with self.assertRaises(ValueError):geometry.apply(np.zeros((100,100,3),np.uint8))
        with self.assertRaises(ValueError):VisionCalibration(camera_matrix=((1,0,0),(0,1,0),(0,0,1)))


if __name__=='__main__':unittest.main()
