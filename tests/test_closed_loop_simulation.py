"""Production state machine and complete simulated rounds; no ideal accuracy formula."""
import copy
from concurrent.futures import Future
import time
import unittest
import numpy as np
from game.board import ChessBoard
from localization.vision_contract import board_to_grid,moved_grid,grid_to_tensor
from localization.live_vision import CameraFrame,HandObservation
from localization.onnx_vision_adapter import ONNXVisionAdapter,VisionState,BoardObservation
from localization.offset_measurement import VisionCalibration
from localization.simulation import FakeCamera,FakeRecognizer,FakeRobot,FirstLegalAI,wait_until


class ImmediateExecutor:
    def submit(self,fn,*args):
        future=Future()
        try:future.set_result(fn(*args))
        except Exception as exc:future.set_exception(exc)
        return future


class Clock:
    def __init__(self):self.at=10.
    def __call__(self):return self.at


class TestClosedLoopSimulation(unittest.TestCase):
    def make_adapter(self):
        self.clock=Clock();self.camera=FakeCamera(board_to_grid(ChessBoard().snapshot()))
        self.adapter=ONNXVisionAdapter(FakeRecognizer(self.camera),stable_window_size=3,
            executor=ImmediateExecutor(),clock=self.clock,headless=True)
        self.adapter.gate.settle_seconds=0
        self.addCleanup(self.adapter.close);self.seq=0
        return self.adapter

    def feed(self,hands=0):
        self.seq+=1;self.clock.at+=.25
        record=HandObservation(self.seq,self.clock.at,self.clock.at,hands,0)
        return self.adapter.process_frame(CameraFrame(self.seq,self.clock.at,self.camera.image.copy()),hand_record=record)

    def test_board_observation_and_rl_tensor(self):
        board=ChessBoard();grid=board_to_grid(board.snapshot())
        tensor=grid_to_tensor(grid)
        np.testing.assert_array_equal(tensor,board.get_state_tensor())
        obs=BoardObservation('a6a5','a6','a5','P',False,grid,'',tensor,{})
        self.assertEqual(tuple(obs),('a6','a5'));self.assertEqual(tensor.sum(),32)

    def test_hand_occlusion_and_motion_suppression(self):
        adapter=self.make_adapter()
        for _ in range(4):self.feed()
        self.assertTrue(adapter.ready_event.is_set())
        for _ in range(3):
            self.assertIsNone(self.feed(hands=1))
            self.assertEqual(adapter.state,VisionState.OCCLUDED_OR_MOVING)
            self.assertIsNone(adapter.get_snapshot());self.assertTrue(adapter.move_queue.empty())
        for _ in range(5):self.feed()
        self.assertTrue(adapter.ready_event.is_set());self.assertEqual(adapter.state,VisionState.CONFIRMED_STABLE)

    def test_three_stage_stability_state_machine(self):
        adapter=self.make_adapter();self.feed(hands=1)
        self.assertEqual(adapter.state,VisionState.OCCLUDED_OR_MOVING)
        self.feed();self.assertEqual(adapter.state,VisionState.SETTLING)
        self.feed();self.assertEqual(adapter.state,VisionState.SETTLING)
        self.feed();self.assertEqual(adapter.state,VisionState.SETTLING)
        self.feed();self.assertEqual(adapter.state,VisionState.CONFIRMED_STABLE)
        previous=adapter.get_snapshot()['frame_id']
        # Repeated calls on one source frame cannot add observations.
        packet=CameraFrame(self.seq,self.clock.at,self.camera.image.copy())
        for _ in range(10):adapter.process_frame(packet)
        self.assertEqual(adapter.get_snapshot()['frame_id'],previous)

    def test_cumulative_error_elimination_simulation(self):
        from main import ChessRobotOrchestrator
        board=ChessBoard();camera=FakeCamera(board_to_grid(board.snapshot()),{'a0':(3,-2)})
        cal=VisionCalibration(calibrated=True,calibration_id='SIMULATION-ONLY')
        vision=ONNXVisionAdapter(FakeRecognizer(camera),calibration=cal,headless=True,allow_control=True)
        robot=FakeRobot(camera)
        app=ChessRobotOrchestrator(arm=robot,camera=camera,vision=vision,board=board,ai=FirstLegalAI(board),
            calibration=cal,headless=True,observation_only=False,pickup_timeout=3,verification_timeout=4)
        self.addCleanup(app.shutdown);app.start_vision();self.assertTrue(vision.ready_event.wait(4))
        vision.session.synchronize_game(board)
        records=[]
        for source,target in [('a6','a5'),('a5','a4'),('a4','b4'),('b4','b3'),('b3','b2')]:
            camera.set_board(moved_grid(camera.grid,source,target),camera.offsets)
            self.assertTrue(wait_until(lambda:not app.move_queue.empty(),4),vision.session.reason)
            self.assertTrue(app._wait_for_human_move())
            self.assertTrue(app._robot_move(),app.error)
            self.assertEqual(board_to_grid(board.snapshot()),camera.grid)
            self.assertGreater(app.last_verification['captured_at'],app.last_verification['finished_at'])
            self.assertGreater(app.last_verification['frame_id'],app.last_verification['barrier_frame'])
            self.assertLessEqual(app.last_verification['offset']['offset_dist_mm'],1.2)
            records.append(app.last_verification)
        self.assertEqual(len(robot.commands),5)
        self.assertGreater(abs(robot.commands[0]['start_offset'][0]),1.)
        self.assertTrue(all(cmd.get('end_offset')==(0.,0.) for cmd in robot.commands))
        self.assertTrue(all(records[i]['frame_id']<records[i+1]['frame_id'] for i in range(4)))
