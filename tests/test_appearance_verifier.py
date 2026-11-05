"""Production exemplar verifier and live-gate integration; no real robot."""
import copy
import unittest

import cv2
import numpy as np

from localization.appearance_verifier import BoardAppearance, AppearanceTracker
from localization.live_vision import CameraFrame, LiveGate


def image(red=True):
    board = np.full((500, 450, 3), (150, 175, 190), np.uint8)
    for x in np.linspace(50, 400, 9):
        cv2.line(board, (round(x),50), (round(x),450), (45,45,45), 1)
    for y in np.linspace(50, 450, 10):
        cv2.line(board, (50,round(y)), (400,round(y)), (45,45,45), 1)
    for (x,y), glyph in [((50,361),"R"), ((138,317),"P"), ((225,50),"K")]:
        cv2.circle(board, (x,y), 18, (225,230,230), -1)
        cv2.circle(board, (x,y), 18, (35,35,35), 1)
        color = (35,35,180) if red else (35,35,35)
        cv2.putText(board,glyph,(x-9,y+7),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2,cv2.LINE_AA)
    return board


def observation(seq, at):
    grid = [["."]*9 for _ in range(10)]
    grid[0][4], grid[7][0], grid[6][2] = "K", "R", "P"
    return dict(frame_id=seq,captured_at=at,grid_chars=grid,
                conf_grid=np.full((10,9),.9).tolist(),matrix=np.eye(3).tolist(),
                corners=[[50,50],[400,50],[50,450],[400,450]],corner_scores=[.3]*4,
                raw_unknown=[],rejection="",inference_ms=100,view_transform=np.eye(3).tolist())


class TestBoardAppearance(unittest.TestCase):
    def setUp(self):
        self.reference = image()
        self.verifier = BoardAppearance(self.reference)

    def test_same_image_and_moderate_lighting_and_noise_match(self):
        rng = np.random.default_rng(42)
        variants = [self.reference.copy(),np.clip(self.reference*.8,0,255).astype(np.uint8),
                    np.clip(self.reference*1.1,0,255).astype(np.uint8),
                    np.clip(self.reference*np.array([.9,1,1.1]),0,255).astype(np.uint8),
                    np.clip(self.reference.astype(float)+rng.normal(0,2,self.reference.shape),0,255).astype(np.uint8)]
        for index, variant in enumerate(variants):
            with self.subTest(variant=index):
                self.assertTrue(self.verifier.match(variant)[0])

    def test_removed_piece_is_rejected(self):
        changed = self.reference.copy()
        changed[335:389,24:78] = (150,175,190)
        self.assertFalse(self.verifier.match(changed)[0])

    def test_extra_piece_or_local_occluder_is_rejected(self):
        changed = self.reference.copy()
        cv2.circle(changed,(300,250),18,(230,230,230),-1)
        self.assertFalse(self.verifier.match(changed)[0])

    def test_same_shape_with_red_to_black_ink_is_rejected(self):
        self.assertFalse(self.verifier.match(image(red=False))[0])

    def test_glyph_change_without_moving_circle_is_rejected(self):
        changed=self.reference.copy()
        cv2.circle(changed,(50,361),6,(0,0,0),-1)
        self.assertFalse(self.verifier.match(changed)[0])

    def test_subpixel_global_jitter_can_register(self):
        changed=cv2.warpAffine(self.reference,np.float32([[1,0,.6],[0,1,-.4]]),(450,500),
                               borderMode=cv2.BORDER_REFLECT)
        self.assertTrue(self.verifier.match(changed)[0])

    def test_reference_does_not_adapt_to_unconfirmed_changes(self):
        initial=self.verifier.reference.copy()
        changed=self.reference.copy()
        changed[330:390,:80]=0
        for _ in range(4):self.assertFalse(self.verifier.match(changed)[0])
        np.testing.assert_array_equal(initial,self.verifier.reference)
        self.assertTrue(self.verifier.match(self.reference)[0])

    def test_low_texture_empty_grid_can_match_but_a_new_piece_cannot(self):
        empty=np.full((500,450,3),190,np.uint8)
        for x in np.linspace(50,400,9):cv2.line(empty,(round(x),50),(round(x),450),(85,85,85),1)
        for y in np.linspace(50,450,10):cv2.line(empty,(50,round(y)),(400,round(y)),(85,85,85),1)
        verifier=BoardAppearance(empty)
        self.assertTrue(verifier.match(empty.copy())[0])
        changed=empty.copy();cv2.circle(changed,(225,250),18,(230,230,230),-1)
        cv2.putText(changed,'P',(216,257),cv2.FONT_HERSHEY_SIMPLEX,.65,(30,30,30),2)
        self.assertFalse(verifier.match(changed)[0])

    def test_featureless_image_is_never_a_valid_board_reference_match(self):
        flat=np.full((500,450,3),180,np.uint8)
        self.assertFalse(BoardAppearance(flat).match(flat.copy())[0])

    def test_background_beyond_piece_margin_does_not_change_board(self):
        changed = self.reference.copy()
        changed[480:] = 0
        changed[:, :15] = 255
        self.assertTrue(self.verifier.match(changed)[0])

    def test_outer_intersection_piece_changes_remain_visible(self):
        changed = self.reference.copy()
        cv2.circle(changed, (50,361), 14, (25,25,25), -1)
        self.assertFalse(self.verifier.match(changed)[0])


class TestTrackedLiveGate(unittest.TestCase):
    def prepared(self):
        board=image()
        gate=LiveGate()
        gate.observe(CameraFrame(1,10,board),10)
        for seq,at in ((2,10.4),(3,10.6)):
            packet=CameraFrame(seq,at,board.copy())
            gate.observe(packet,at)
            raw=observation(seq,at)
            gate.accept((gate.epoch,seq),raw,at+.05)
        tracker=AppearanceTracker()
        tracker.seed(gate.output,packet,gate.epoch,raw)
        return board,gate,tracker

    def test_no_single_frame_or_mismatched_image_can_seed(self):
        tracker=AppearanceTracker()
        raw=observation(1,10)
        with self.assertRaises(ValueError):tracker.seed(raw,CameraFrame(1,10,image()),1,raw)
        raw["temporal_evidence"]={"frame_ids":[0,1]}
        with self.assertRaises(ValueError):tracker.seed(raw,CameraFrame(2,10,image()),1,raw)
        other=copy.deepcopy(raw);other["grid_chars"][7][0]="C"
        with self.assertRaises(ValueError):tracker.seed(raw,CameraFrame(1,10,image()),1,other)

    def test_fresh_pixels_keep_confirmed_board_despite_rejected_neural_guess(self):
        board,gate,tracker=self.prepared()
        bad=observation(4,10.8)
        bad["grid_chars"][7][0]="C"
        bad["rejection"]="class conflict"
        gate.observe_neural_while_tracking((gate.epoch,4),bad,10.9,tracker)
        packet=CameraFrame(5,11,board.copy());gate.observe(packet,11)
        self.assertTrue(gate.accept_tracking(tracker,packet,np.eye(3),11.01))
        self.assertEqual(gate.output["grid_chars"][7][0],"R")
        self.assertEqual(gate.last_result["grid_chars"][7][0],"C")
        self.assertEqual(gate.output["source"],"appearance_verified")
        self.assertEqual(gate.output["captured_at"],11)
        self.assertEqual(gate.output["classification_reference_time"],10.6)
        self.assertEqual(gate.output["conf_grid"],tracker.anchor["conf_grid"])

    def test_raw_neural_occlusion_or_trusted_conflict_stops_tracking(self):
        for raw_unknown in (True,False):
            board,gate,tracker=self.prepared();raw=observation(4,10.8)
            if raw_unknown:raw["raw_unknown"]=[[7,0]];raw["rejection"]="unknown"
            else:raw["grid_chars"][7][0]="C"
            gate.observe_neural_while_tracking((gate.epoch,4),raw,10.9,tracker)
            if not raw_unknown:
                self.assertIsNotNone(gate.output)
                packet=CameraFrame(5,11,board.copy());gate.observe(packet,11)
                self.assertTrue(gate.accept_tracking(tracker,packet,np.eye(3),11.01))
                raw["frame_id"],raw["captured_at"]=6,11.1
                gate.observe_neural_while_tracking((gate.epoch,6),raw,11.2,tracker)
            self.assertIsNone(gate.output)
            self.assertFalse(tracker.ready(gate.epoch))

    def test_weak_neural_swap_cannot_overrule_current_matching_pixels(self):
        board,gate,tracker=self.prepared()
        packet=CameraFrame(4,10.8,board.copy());gate.observe(packet,10.8)
        self.assertTrue(gate.accept_tracking(tracker,packet,np.eye(3),10.81))
        raw=observation(5,10.9)
        raw["grid_chars"][7][0],raw["grid_chars"][6][2]="N","R"
        raw["conf_grid"][7][0],raw["conf_grid"][6][2]=.58,.65
        gate.observe_neural_while_tracking((gate.epoch,5),raw,10.95,tracker)
        self.assertIsNotNone(gate.output)
        self.assertEqual(gate.output["grid_chars"][7][0],"R")
        self.assertTrue(tracker.ready(gate.epoch))

    def test_no_fresh_pixel_evidence_cannot_defer_a_model_conflict(self):
        board,gate,tracker=self.prepared();raw=observation(4,10.8)
        raw["grid_chars"][7][0]="C"
        gate.output=None
        gate.observe_neural_while_tracking((gate.epoch,4),raw,10.9,tracker)
        self.assertIsNone(gate.output)
        self.assertFalse(tracker.ready(gate.epoch))

    def test_changed_pixels_clear_output_and_old_reference(self):
        board,gate,tracker=self.prepared();board[330:390,:80]=0
        packet=CameraFrame(4,10.8,board);gate.observe(packet,10.8)
        self.assertFalse(gate.accept_tracking(tracker,packet,np.eye(3),10.9))
        self.assertIsNone(gate.output)
        self.assertIsNone(tracker.anchor)

    def test_motion_epoch_and_late_neural_result_cannot_revive_reference(self):
        board,gate,tracker=self.prepared();epoch=gate.epoch
        gate.observe(CameraFrame(4,10.8,board),10.8,"hand")
        raw=observation(3,10.6)
        self.assertFalse(gate.observe_neural_while_tracking((epoch,3),raw,10.9,tracker))
        self.assertFalse(gate.accept_tracking(tracker,CameraFrame(5,11,board),np.eye(3),11))
        self.assertIsNone(gate.output)

    def test_duplicate_and_stale_frames_never_become_new_measurements(self):
        board,gate,tracker=self.prepared()
        packet=CameraFrame(4,10.8,board);gate.observe(packet,10.8)
        self.assertTrue(gate.accept_tracking(tracker,packet,np.eye(3),10.81))
        old=gate.output
        self.assertFalse(gate.accept_tracking(tracker,packet,np.eye(3),10.82))
        self.assertIs(gate.output,old)
        self.assertEqual(gate.appearance_publish_count,1)
        self.assertIsNone(tracker.verify(CameraFrame(5,10.8,board),np.eye(3),gate.epoch,12))

    def test_tracking_never_reuses_millimeter_offsets(self):
        board,gate,tracker=self.prepared();tracker.anchor["piece_offsets"]={"a7":[1,2]}
        packet=CameraFrame(4,10.8,board);gate.observe(packet,10.8)
        self.assertTrue(gate.accept_tracking(tracker,packet,np.eye(3),10.9))
        self.assertNotIn("piece_offsets",gate.output)
        self.assertFalse(gate.output["offsets_valid"])

    def test_tracking_output_expires_without_new_pixel_verification(self):
        board,gate,tracker=self.prepared();packet=CameraFrame(4,10.8,board);gate.observe(packet,10.8)
        gate.accept_tracking(tracker,packet,np.eye(3),10.81)
        gate.observe(CameraFrame(5,11.2,board),11.2)
        self.assertIsNone(gate.output)

    def test_camera_geometry_outside_image_is_rejected(self):
        board,gate,tracker=self.prepared();transform=np.eye(3);transform[0,2]=1000
        self.assertIsNone(tracker.verify(CameraFrame(4,10.8,board),transform,gate.epoch,10.9))

    def test_transport_rebases_flow_drift_without_adapting_visual_reference(self):
        board,gate,tracker=self.prepared()
        original=tracker.appearance.reference.copy()
        for step in range(1,21):
            transform=np.eye(3);transform[0,2]=step*.4
            at=10.6+step*.2;packet=CameraFrame(3+step,at,board.copy())
            gate.observe(packet,at)
            self.assertTrue(gate.accept_tracking(tracker,packet,transform,at+.01),tracker.reason)
        np.testing.assert_array_equal(original,tracker.appearance.reference)
        self.assertEqual(tracker.anchor["frame_id"],3)

    def test_reference_registration_recovers_projective_camera_change(self):
        board,gate,tracker=self.prepared()
        motion=np.array([[1.,.001,.4],[-.001,1.,.3],[.000003,-.000002,1.]])
        current=cv2.warpPerspective(board,motion,(450,500),borderMode=cv2.BORDER_REFLECT)
        at=10.8;packet=CameraFrame(4,at,current);gate.observe(packet,at)
        self.assertTrue(gate.accept_tracking(tracker,packet,np.eye(3),at+.01),tracker.reason)
        points=np.float32([[[50,50],[400,50],[50,450],[400,450]]])
        recovered=cv2.perspectiveTransform(cv2.perspectiveTransform(points,motion),np.array(gate.output['matrix']))
        np.testing.assert_allclose(recovered,points,atol=.6)

    def test_anchored_registration_failure_cannot_reuse_last_geometry(self):
        board,gate,tracker=self.prepared()
        tracker.reference_points=None
        packet=CameraFrame(4,10.8,board);gate.observe(packet,10.8)
        self.assertFalse(gate.accept_tracking(tracker,packet,np.eye(3),10.81))
        self.assertIsNone(gate.output)
        self.assertIsNone(tracker.anchor)


if __name__ == "__main__":
    unittest.main()
