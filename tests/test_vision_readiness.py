"""Software acceptance: actual contracts, measurements and transaction code."""
import copy
from concurrent.futures import Future
import threading
import time
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from game.board import ChessBoard
from localization.vision_contract import (CHANNEL_ORDER,coordinate,parse_coordinate,board_to_grid,
    grid_to_board,grid_to_fen,fen_to_grid,grid_to_tensor,moved_grid,infer_move)
from localization.vision_session import VisionSession,VerificationStatus
from localization.offset_measurement import VisionCalibration,CenterMeasurer,require_offset,OffsetUnavailable
from localization.simulation import draw_board,FakeCamera,FakeRecognizer,FakeRobot,FirstLegalAI,wait_until,run_smoke
from localization.live_vision import CameraFrame
from localization.onnx_vision_adapter import ONNXVisionAdapter


class Clock:
    def __init__(self):self.value=10.
    def __call__(self):return self.value


def result(grid,seq,at):
    return dict(grid_chars=copy.deepcopy(grid),conf_grid=np.full((10,9),.95).tolist(),
                frame_id=seq,captured_at=at,rejection='',raw_unknown=[])


class TestBoardContract(unittest.TestCase):
    def test_all_90_coordinate_roundtrips_and_rejection(self):
        from main import Coord
        for y in range(10):
            for x in range(9):
                self.assertEqual(parse_coordinate(coordinate(x,y)),(x,y))
                self.assertEqual(Coord.vision_to_game(Coord.game_to_vision(x,y)),(x,y))
        for value in ('a10','j0','a-1','a0x','x',''):
            with self.assertRaises(ValueError):parse_coordinate(value)
            with self.assertRaises(ValueError):Coord.vision_to_game(value)

    def test_all_fourteen_pieces_tensor_matches_game_channels(self):
        grid=[['.']*9 for _ in range(10)]
        for i,ch in enumerate(CHANNEL_ORDER):grid[i//9][i%9]=ch
        board=ChessBoard();board.board=grid_to_board(grid)
        np.testing.assert_array_equal(grid_to_tensor(grid),board.get_state_tensor())
        self.assertEqual(board_to_grid(board.snapshot()),grid)
        self.assertEqual(grid_to_tensor(grid).sum(),14)

    def test_asymmetric_fen_both_sides_and_empty(self):
        grid=[['.']*9 for _ in range(10)];grid[0][3]='k';grid[9][5]='K';grid[4][1]='R';grid[7][8]='p'
        for turn in (False,True):self.assertEqual(fen_to_grid(grid_to_fen(grid,turn)),(grid,turn))
        self.assertEqual(grid_to_fen([['.']*9 for _ in range(10)]),'9/9/9/9/9/9/9/9/9/9 w - - 0 1')

    def test_sparse_game_initialization_and_illegal_visual_layout_separate(self):
        grid=[['.']*9 for _ in range(10)];grid[0][4]='k';grid[9][3]='K';grid[5][2]='P'
        board=ChessBoard();board.load_vision_grid(grid,red_turn=False)
        self.assertEqual(board_to_grid(board.snapshot()),grid);self.assertFalse(board.red_turn)
        grid[2][2]='B'
        before=board.snapshot()
        with self.assertRaises(ValueError):board.load_vision_grid(grid)
        self.assertEqual(board.snapshot(),before)

    def test_normal_capture_and_multi_change_inference(self):
        grid=board_to_grid(ChessBoard().snapshot());next_grid=moved_grid(grid,'a6','a5')
        self.assertEqual(infer_move(grid,next_grid),('a6','a5','P',False))
        sparse=[['.']*9 for _ in range(10)];sparse[5][0]='R';sparse[3][0]='p'
        capture=moved_grid(sparse,'a5','a3')
        self.assertEqual(infer_move(sparse,capture),('a5','a3','R',True))
        capture[1][4]='C';self.assertIsNone(infer_move(sparse,capture))


class TestCenters(unittest.TestCase):
    def setUp(self):
        self.grid=[['.']*9 for _ in range(10)];self.grid[4][4]='P'
        self.cal=VisionCalibration(calibrated=True,calibration_id='SIMULATION-ONLY')

    def test_measured_center_directions_and_subpixel_synthetic_error(self):
        measurer=CenterMeasurer(self.cal)
        for delta in ((0,0),(4,-3),(-5,2),(2,5)):
            image=draw_board(self.grid,{'e4':delta})
            info=measurer.measure(image,self.grid,frame_id=2,captured_at=10)['e4']
            self.assertTrue(info['valid'],info)
            actual=np.array([round(225+delta[0]),round(50+4*400/9+delta[1])])
            self.assertLess(np.linalg.norm(np.array(info['detected_px'])-actual),.6)
            board,robot=self.cal.displacement(np.array(info['delta_px']))
            np.testing.assert_allclose(info['board_offset_mm'],board,atol=1e-10)
            np.testing.assert_allclose(info['robot_offset_mm'],robot,atol=1e-10)
            self.assertEqual(info['frame_id'],2)

    def test_blank_glyph_only_and_partial_rim_are_not_valid_circles(self):
        measurer=CenterMeasurer(self.cal)
        blank=np.full((500,450,3),180,np.uint8)
        glyph=blank.copy();cv2.putText(glyph,'K',(215,235),0,.7,(30,30,30),2)
        partial=draw_board(self.grid);partial[206:251,225:248]=(100,100,100)
        for image in (blank,glyph,partial):
            self.assertFalse(measurer.measure(image,self.grid)['e4']['valid'])

    def test_noise_and_soft_shading_preserve_center(self):
        image=draw_board(self.grid,{'e4':(3,-2)});y,x=np.mgrid[:500,:450]
        gain=1-.2*np.exp(-((x-225)**2+(y-228)**2)/(2*60**2))
        noise=np.random.default_rng(4).normal(0,1,image.shape)
        image=np.clip(image*gain[...,None]+noise,0,255).astype(np.uint8)
        info=CenterMeasurer(self.cal).measure(image,self.grid)['e4']
        self.assertTrue(info['valid'],info);self.assertLess(abs(info['detected_px'][0]-228),.7)

    def test_measurement_missing_expired_unverified_or_wrong_units_refused(self):
        info=CenterMeasurer().measure(draw_board(self.grid),self.grid,frame_id=1,captured_at=10)['e4']
        self.assertTrue(info['valid']);self.assertFalse(info['control_valid'])
        with self.assertRaises(OffsetUnavailable):require_offset(info,10.1)
        info['control_valid']=True
        with self.assertRaises(OffsetUnavailable):require_offset(info,12)
        with self.assertRaises(OffsetUnavailable):require_offset(None,10)
        self.assertEqual(len(require_offset(info,10.1)),2)

    def test_nonuniform_grid_and_rotation_displacement_are_explicit(self):
        cal=VisionCalibration(rows_mm=(0,33,66,99,132,175,208,241,274,307),robot_from_board=((0,1),(-1,0)))
        np.testing.assert_allclose(cal.nominal(5,4),[225,50+400*175/307])
        b,r=cal.displacement((2,-3));np.testing.assert_allclose(r,[b[1],-b[0]])
        self.assertFalse(cal.compatible_with_robot())
        with self.assertRaises(ValueError):VisionCalibration(calibrated='false')


class TestTransactions(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();self.board=ChessBoard();self.grid=board_to_grid(self.board.snapshot())
        self.session=VisionSession(clock=self.clock)
        self.session.publish(result(self.grid,1,10),{},0)

    def test_human_baseline_waits_for_ack_and_matches_whole_software_board(self):
        after=moved_grid(self.grid,'a6','a5');self.clock.value=10.2
        event=self.session.publish(result(after,2,10.2),{},0)
        self.assertEqual(self.session.baseline,self.grid)
        self.assertEqual(board_to_grid(self.board.snapshot()),self.grid)
        self.session.commit_human_move(event.event_id,self.board)
        self.assertEqual(self.session.baseline,after)
        self.assertEqual(board_to_grid(self.board.snapshot()),after)
        self.assertTrue(self.session.latest['fen'].split()[1]=='b')

    def test_illegal_move_keeps_both_committed_baselines(self):
        after=moved_grid(self.grid,'a6','b6');self.clock.value=10.2
        event=self.session.publish(result(after,2,10.2),{},0)
        with self.assertRaises(ValueError):self.session.commit_human_move(event.event_id,self.board)
        self.assertEqual(self.session.baseline,self.grid)
        self.assertEqual(board_to_grid(self.board.snapshot()),self.grid)
        self.clock.value=10.3
        self.assertIsNone(self.session.publish(result(after,3,10.3),{},0))
        self.clock.value=10.4;self.session.publish(result(self.grid,4,10.4),{},0)
        self.assertEqual(self.session.mode,'OBSERVING')

    def test_stale_repeated_and_invalid_class_observations_cannot_submit(self):
        self.clock.value=12
        self.assertIsNone(self.session.publish(result(self.grid,2,10),{},0))
        self.assertIsNone(self.session.snapshot())
        self.clock.value=10.2
        self.assertIsNone(self.session.publish(result(self.grid,1,10.2),{},0))
        bad=result(self.grid,3,10.2);bad['raw_unknown']=[[3,4]]
        self.assertIsNone(self.session.publish(bad,{},0))

    def test_motion_invalidates_pending_event_and_inflight_generation(self):
        after=moved_grid(self.grid,'a6','a5');self.clock.value=10.2
        event=self.session.publish(result(after,2,10.2),{},0)
        self.session.invalidate('hand')
        self.assertTrue(self.session.move_queue.empty())
        self.assertIsNone(self.session.publish(result(after,3,10.3),{},0))
        with self.assertRaises(ValueError):self.session.commit_human_move(event.event_id,self.board)
        self.assertEqual(self.session.baseline,self.grid)

    def test_missing_offsets_never_silently_mean_zero(self):
        adapter=ONNXVisionAdapter(FakeRecognizer(FakeCamera(self.grid)),headless=True)
        try:
            with self.assertRaises(OffsetUnavailable):adapter.get_piece_offset('a0')
            with self.assertRaises(ValueError):adapter.process_frame(draw_board(self.grid))
            with self.assertRaises(ValueError):adapter.post_move_verification('a3','r')
        finally:adapter.close()

    def prepare_verification(self):
        board=self.board;board.red_turn=False;self.session.red_turn=False
        outcome={}
        worker=threading.Thread(target=lambda:outcome.update(plan=self.session.request_pick('a3','a4',board,timeout=1)))
        worker.start()
        self.assertTrue(wait_until(lambda:bool(self.session.offset_request),.5))
        self.clock.value=10.2
        info=dict(valid=True,control_valid=True,robot_offset_mm=(1.,-2.),frame_id=2,captured_at=10.2)
        self.session.publish(result(self.grid,2,10.2),{'a3':info},0)
        worker.join(1);self.assertIn('plan',outcome)
        plan=outcome['plan'];self.clock.value=10.5;self.session.finish_action(plan['token'],3)
        return plan

    def test_post_action_rejects_pre_action_cached_and_wrong_board(self):
        plan=self.prepare_verification();generation=self.session.generation
        expected=plan['expected'];info=dict(valid=True,control_valid=True,robot_offset_mm=(0.,0.),offset_dist_mm=0.,frame_id=3,captured_at=10.5)
        self.clock.value=10.6
        self.session.publish(result(expected,3,10.5),{'a4':info},generation)
        self.assertIsNone(self.session.verification)
        self.session.publish(result(self.grid,4,10.6),{},generation)
        self.assertEqual(self.session.verification['status'],'STATE_CONFLICT')
        self.assertEqual(self.session.baseline,self.grid)

    def test_post_action_fresh_match_commits_only_after_success(self):
        plan=self.prepare_verification();self.clock.value=10.7
        info=dict(valid=True,control_valid=True,robot_offset_mm=(.2,.3),offset_dist_mm=.36,frame_id=4,captured_at=10.7)
        self.session.publish(result(plan['expected'],4,10.7),{'a4':info},self.session.generation)
        self.assertEqual(self.session.wait_verification(plan['token'],.01)['status'],'SUCCESS')
        self.assertEqual(board_to_grid(self.board.snapshot()),self.grid)
        with self.assertRaises(ValueError):self.session.commit_robot_move(plan['token'],self.board,'2324')
        self.assertEqual(board_to_grid(self.board.snapshot()),self.grid)
        self.session.commit_robot_move(plan['token'],self.board,'0304')
        self.assertEqual(board_to_grid(self.board.snapshot()),plan['expected'])
        self.assertEqual(self.session.baseline,plan['expected'])

    def test_post_action_large_offset_timeout_and_cancel_are_not_success(self):
        plan=self.prepare_verification()
        self.assertEqual(self.session.wait_verification(plan['token'],.01)['status'],'TIMEOUT')
        self.clock.value=10.7
        info=dict(valid=True,control_valid=True,robot_offset_mm=(3.,0.),offset_dist_mm=3.,frame_id=4,captured_at=10.7)
        self.session.publish(result(plan['expected'],4,10.7),{'a4':info},self.session.generation)
        self.assertEqual(self.session.verification['status'],'ERROR_EXCEEDED')
        with self.assertRaises(ValueError):self.session.commit_robot_move(plan['token'],self.board,'0304')
        self.session.close()
        self.assertEqual(self.session.wait_verification(plan['token'],.01)['status'],'CANCELLED')


class TestProductionIntegration(unittest.TestCase):
    def test_complete_round_with_actual_orchestrator_and_no_camera_access(self):
        with patch('cv2.VideoCapture',side_effect=AssertionError('Physical camera forbidden')):
            data=run_smoke()
        self.assertTrue(data['passed'],data);self.assertTrue(data['board_matches_camera'])
        self.assertFalse(data['hardware_accessed']);self.assertEqual(len(data['commands']),1)

    def test_robot_failure_and_post_move_conflict_block_software_commit(self):
        from main import ChessRobotOrchestrator
        for options in ({'failure':True},{'wrong_board':True},{'residual':(4.,0.)}):
            with self.subTest(options=options):
                board=ChessBoard();board.red_turn=False;grid=board_to_grid(board.snapshot())
                camera=FakeCamera(grid);cal=VisionCalibration(calibrated=True,calibration_id='SIMULATION-ONLY')
                vision=ONNXVisionAdapter(FakeRecognizer(camera),calibration=cal,headless=True,allow_control=True)
                robot=FakeRobot(camera,**options)
                app=ChessRobotOrchestrator(arm=robot,camera=camera,vision=vision,board=board,ai=FirstLegalAI(board),
                    calibration=cal,observation_only=False,headless=True,pickup_timeout=2,verification_timeout=2)
                try:
                    app.start_vision();self.assertTrue(vision.ready_event.wait(4));vision.session.synchronize_game(board)
                    self.assertFalse(app._robot_move())
                    self.assertEqual(board_to_grid(board.snapshot()),grid)
                    self.assertTrue(app.stop_event.is_set());self.assertEqual(len(robot.commands),1)
                    self.assertTrue(robot.stopped)
                finally:app.shutdown()

    def test_no_valid_calibration_prevents_any_hardware_creation(self):
        from main import ChessRobotOrchestrator
        with patch('cv2.VideoCapture',side_effect=AssertionError('Camera forbidden')):
            with self.assertRaises(ValueError):ChessRobotOrchestrator(enable_hardware=True,observation_only=False)


if __name__=='__main__':unittest.main()
