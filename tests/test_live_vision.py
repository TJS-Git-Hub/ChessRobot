"""Production live-gate contracts; no camera, hardware or model loading."""
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np

from localization.live_vision import (
    CameraFrame, LatestCamera, LiveGate, MotionCheck, GeometryStabilizer, geometry_reason, grid_reason,
)
from recognition.onnx_pipeline.chess_recognizer import ChessRecognizer


def empty():
    return [["."]*9 for _ in range(10)]


class TestLiveGate(unittest.TestCase):
    def packet(self, seq, at):
        return CameraFrame(seq, at, np.zeros((20, 20, 3), dtype=np.uint8))

    def result(self, seq, at):
        return {"frame_id": seq, "captured_at": at, "grid_chars": empty(),
                "conf_grid": np.full((10, 9), .9).tolist(), "rejection": ""}

    def prepared(self):
        gate = LiveGate()
        gate.observe(self.packet(1, 10), 10)
        gate.observe(self.packet(2, 10.4), 10.4)
        return gate

    def test_two_fresh_classifications_confirm(self):
        gate = self.prepared()
        self.assertTrue(gate.can_infer(10.4))
        self.assertFalse(gate.accept((gate.epoch, 2), self.result(2, 10.4), 10.5))
        gate.observe(self.packet(3, 10.6), 10.6)
        self.assertTrue(gate.accept((gate.epoch, 3), self.result(3, 10.6), 10.7))

    def test_augmentation_abstention_can_use_fresh_temporal_evidence(self):
        gate = self.prepared()
        for seq, at in ((2, 10.4), (3, 10.6)):
            result = self.result(seq, at)
            gate.observe(self.packet(seq, at), at)
            gate.accept((gate.epoch, seq), result, at+.1)
        result = self.result(4, 10.8)
        result["grid_chars"][0][0] = "x"
        result["conf_grid"][0][0] = 0
        result["raw_unknown"] = []
        gate.observe(self.packet(4, 10.8), 10.8)
        self.assertTrue(gate.accept((gate.epoch, 4), result, 10.9))
        self.assertEqual(gate.output["grid_chars"][0][0], ".")
        # A current raw occlusion must invalidate that same temporal evidence.
        result["raw_unknown"] = [[0, 0]]
        result["frame_id"] = 5
        result["captured_at"] = 11
        gate.observe(self.packet(5, 11), 11)
        self.assertFalse(gate.accept((gate.epoch, 5), result, 11.1))
        self.assertEqual(gate.display_output["grid_chars"][0][0], "x")

    def test_repeated_frame_does_not_vote(self):
        gate = self.prepared()
        token = (gate.epoch, 2)
        gate.accept(token, self.result(2, 10.4), 10.5)
        self.assertFalse(gate.observe(self.packet(2, 10.4), 10.6))
        self.assertFalse(gate.accept(token, self.result(2, 10.4), 10.6))
        self.assertEqual(gate.votes, 1)

    def test_display_retention_is_not_fresh_output_and_motion_clears_both(self):
        gate = LiveGate(max_age=1, display_max_age=2)
        gate.observe(self.packet(1, 10), 10)
        for seq, at in ((2, 10.4), (3, 10.6)):
            gate.observe(self.packet(seq, at), at)
            gate.accept((gate.epoch, seq), self.result(seq, at), at+.1)
        gate.observe(self.packet(4, 11.7), 11.7)
        self.assertIsNone(gate.output)
        self.assertIsNotNone(gate.display_output)
        self.assertEqual(gate.state, "CONFIRMED_STABLE")
        self.assertTrue(gate.scene_is_stable(11.7))
        gate.observe(self.packet(5, 11.8), 11.8, "hand")
        self.assertIsNone(gate.output)
        self.assertIsNone(gate.display_output)
        self.assertEqual(len(gate.result_history), 0)
        self.assertFalse(gate.scene_is_stable(11.8))

    def test_hand_discards_inflight_result(self):
        gate = self.prepared()
        token = (gate.epoch, 2)
        gate.observe(self.packet(3, 10.5), 10.5, "hand")
        self.assertFalse(gate.accept(token, self.result(2, 10.4), 10.6))
        self.assertIsNone(gate.output)
        self.assertFalse(gate.can_infer(20))

    def test_recovery_requires_new_consensus(self):
        gate = self.prepared()
        gate.invalidate("hand")
        gate.observe(self.packet(3, 10.5), 10.5)
        self.assertFalse(gate.can_infer(10.6))
        self.assertEqual(gate.votes, 0)
        self.assertTrue(gate.can_infer(10.9))

    def test_stale_and_missing_camera_clear_result(self):
        gate = self.prepared()
        gate.output = self.result(1, 10)
        gate.observe(self.packet(2, 10.1), 11)
        self.assertIsNone(gate.output)
        gate.observe(None, 12)
        self.assertEqual(gate.state, "OCCLUDED_OR_MOVING")

    def test_late_worker_result_is_rejected(self):
        gate = self.prepared()
        self.assertFalse(gate.accept((gate.epoch, 2), self.result(2, 10.4), 12))

    def test_model_rejection_clears_overlay(self):
        gate = self.prepared()
        bad = self.result(2, 10.4)
        bad["rejection"] = "incomplete"
        gate.output = {"old": True}
        self.assertFalse(gate.accept((gate.epoch, 2), bad, 10.5))
        self.assertIsNone(gate.output)

    def test_changed_board_leaves_stable(self):
        gate = self.prepared()
        gate.output = self.result(1, 10)
        gate.state = "CONFIRMED_STABLE"
        gate.candidate = empty()
        result = self.result(3, 10.6)
        result["grid_chars"][3][0] = "p"
        self.assertFalse(gate.accept((gate.epoch, 3), result, 10.7))
        self.assertEqual(gate.state, "SETTLING")
        self.assertIsNone(gate.output)

    def test_two_of_three_recovers_from_single_misclassification(self):
        gate = self.prepared()
        good = self.result(2, 10.4)
        gate.accept((gate.epoch, 2), good, 10.45)
        bad = self.result(3, 10.5)
        bad["grid_chars"][3][0] = "p"
        gate.accept((gate.epoch, 3), bad, 10.55)
        self.assertTrue(gate.accept((gate.epoch, 4), self.result(4, 10.6), 10.65))

    def test_old_vote_cannot_confirm_one_new_result_after_a_gap(self):
        gate = self.prepared()
        gate.accept((gate.epoch, 2), self.result(2, 10.4), 10.45)
        gate.observe(self.packet(3, 13), 13)
        self.assertFalse(gate.accept((gate.epoch, 3), self.result(3, 13), 13.05))
        self.assertIsNone(gate.output)
        self.assertEqual(gate.votes, 1)

    def test_confirmation_window_allows_two_slow_but_fresh_results(self):
        gate = self.prepared()
        first = self.result(2, 10.4)
        self.assertFalse(gate.accept((gate.epoch, 2), first, 11.0))
        gate.observe(self.packet(3, 11.1), 11.1)
        self.assertTrue(gate.accept((gate.epoch, 3), self.result(3, 11.1), 11.7))
        self.assertEqual(gate.output["captured_at"], 11.1)

    def test_consistent_board_updates_without_reentering_settling(self):
        gate = self.prepared()
        for seq in range(2, 12):
            t = 10.4+(seq-2)*.1
            gate.observe(self.packet(seq, t), t)
            gate.accept((gate.epoch, seq), self.result(seq, t), t+.02)
            if seq >= 3:
                self.assertEqual(gate.state, "CONFIRMED_STABLE")
                self.assertIsNotNone(gate.output)

    def test_stable_unknown_is_displayed_but_never_published(self):
        gate = self.prepared()
        for seq, t in ((2, 10.4), (3, 10.6)):
            result = self.result(seq, t)
            result["grid_chars"][3][0] = "x"
            result["rejection"] = "unknown"
            self.assertFalse(gate.accept((gate.epoch, seq), result, t+.03))
        self.assertEqual(gate.state, "CONFIRMED_STABLE")
        self.assertIsNotNone(gate.display_output)
        self.assertIsNone(gate.output)
        gate.observe(self.packet(4, 10.8), 10.8, "hand")
        self.assertIsNone(gate.display_output)

    def test_partial_display_expires_and_geometry_failure_still_blocks(self):
        gate = self.prepared()
        for seq, t in ((2, 10.4), (3, 10.6)):
            result = self.result(seq, t)
            result["rejection"] = "low confidence"
            result["conf_grid"] = np.full((10, 9), .2).tolist()
            gate.accept((gate.epoch, seq), result, t+.02)
        gate.observe(self.packet(4, 12), 12)
        self.assertIsNone(gate.display_output)
        result = self.result(5, 12.1)
        result.update(matrix=None, rejection="bad geometry")
        gate.accept((gate.epoch, 5), result, 12.2)
        self.assertEqual(gate.state, "OCCLUDED_OR_MOVING")

    def test_individual_cell_noise_does_not_reset_the_entire_stable_board(self):
        gate = self.prepared()
        for seq, column in ((2, 0), (3, 1), (4, 2)):
            t = 10.4+(seq-2)*.1
            result = self.result(seq, t)
            result["grid_chars"][3][column] = "p"
            gate.accept((gate.epoch, seq), result, t+.02)
        self.assertEqual(gate.state, "CONFIRMED_STABLE")
        self.assertEqual(gate.output["grid_chars"], empty())
        self.assertEqual(gate.output["temporal_evidence"]["frame_ids"], [2, 3, 4])


class TestCameraAndMotion(unittest.TestCase):
    def test_geometry_filter_rejects_pose_jitter_without_lagging_tracked_camera(self):
        geometry = GeometryStabilizer()
        corners = np.float32([[100, 100], [500, 100], [100, 600], [500, 600]])
        geometry.update(corners, np.eye(3), 1)
        moved = np.eye(3)
        moved[:2, 2] = [20, -10]
        expected = corners+[20, -10]
        np.testing.assert_allclose(geometry.update(expected, moved, 1), expected, atol=.001)
        noisy = expected.copy()
        noisy[0] += [2, -2]
        filtered = geometry.update(noisy, moved, 1)
        self.assertLess(np.linalg.norm(filtered[0]-expected[0]), .5)

    def test_motion_epoch_resets_geometry_history(self):
        geometry = GeometryStabilizer()
        corners = np.float32([[100, 100], [500, 100], [100, 600], [500, 600]])
        geometry.update(corners, np.eye(3), 1)
        shifted = corners+[8, 6]
        np.testing.assert_allclose(geometry.update(shifted, np.eye(3), 2), shifted, atol=.001)

    def test_capture_negotiates_mjpeg_after_fps(self):
        camera = LatestCamera()
        cap = Mock()
        cap.isOpened.return_value = True
        def read():
            camera._stop.set()
            return True, np.full((480, 640, 3), 100, np.uint8)
        cap.read.side_effect = read
        with patch("localization.live_vision.cv2.VideoCapture", return_value=cap):
            camera._read()
        calls = cap.set.call_args_list
        self.assertEqual(calls[-1].args[0], cv2.CAP_PROP_FOURCC)
        self.assertEqual(calls[-2].args[0], cv2.CAP_PROP_FPS)
        self.assertIsNotNone(camera.read())
        cap.release.assert_called_once()

    def test_failed_reads_clear_last_camera_packet(self):
        camera = LatestCamera()
        camera._packet = CameraFrame(1, 0, np.zeros((10, 10, 3), np.uint8))
        cap = Mock()
        cap.isOpened.return_value = True
        cap.read.return_value = (False, None)
        with patch("localization.live_vision.cv2.VideoCapture", return_value=cap):
            camera._read()
        self.assertIsNone(camera.read())
        self.assertTrue(camera.error)

    def texture(self):
        rng = np.random.default_rng(33)
        gray = rng.integers(35, 210, (180, 320), dtype=np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def test_small_shake_and_brightness_do_not_block(self):
        image = self.texture()
        motion = MotionCheck()
        self.assertEqual(motion.update(image, 10), "")
        affine = np.float32([[1, 0, 1.5], [0, 1, -1.0]])
        shifted = cv2.warpAffine(image, affine, (320, 180), borderMode=cv2.BORDER_REFLECT)
        shifted = cv2.convertScaleAbs(shifted, alpha=1.02, beta=8)
        self.assertEqual(motion.update(shifted, 10.1), "")
        self.assertAlmostEqual(motion.transform[0, 2], 1.5, delta=.3)
        self.assertAlmostEqual(motion.transform[1, 2], -1.0, delta=.3)

    def test_large_motion_and_local_occluder_still_block(self):
        image = self.texture()
        motion = MotionCheck()
        motion.update(image, 10)
        shifted = cv2.warpAffine(image, np.float32([[1, 0, 13], [0, 1, 0]]),
                                 (320, 180), borderMode=cv2.BORDER_REFLECT)
        self.assertTrue(motion.update(shifted, 10.1))
        motion = MotionCheck()
        motion.update(image, 10)
        occluded = image.copy()
        occluded[30:150, 95:225] = (35, 75, 150)
        self.assertTrue(motion.update(occluded, 10.1))


class TestGeometryAndClassifier(unittest.TestCase):
    def setUp(self):
        self.rec = ChessRecognizer.__new__(ChessRecognizer)
        self.rec.piece_session = None

    def test_geometry_asymmetric_valid(self):
        points = [[130, 90], [500, 110], [110, 510], [520, 500]]
        self.assertEqual(geometry_reason(points, [.3]*4, (650, 700, 3)), "")

    def test_geometry_bad_or_cropped(self):
        good = [[130, 90], [500, 110], [110, 510], [520, 500]]
        cases = [np.zeros((4, 2)), np.full((4, 2), np.nan),
                 [[3, 3], [500, 3], [3, 510], [500, 510]],
                 np.asarray(good)[[1, 0, 2, 3]]]
        for points in cases:
            with self.subTest(points=points):
                self.assertTrue(geometry_reason(points, [.3]*4, (650, 700, 3)))
        self.assertTrue(geometry_reason(good, [.3, .01, .3, .3], (650, 700, 3)))

    def test_elephant_and_advisor_reachable_squares(self):
        self.assertIn("B", self.rec.get_allowed_classes(True, 7, 4))
        self.assertIn("b", self.rec.get_allowed_classes(False, 2, 4))
        self.assertNotIn("b", self.rec.get_allowed_classes(False, 0, 0))
        self.assertNotIn("a", self.rec.get_allowed_classes(False, 0, 4))
        self.assertNotIn("p", self.rec.get_allowed_classes(False, 1, 4))
        self.assertIn("P", self.rec.get_allowed_classes(True, 0, 8))

    def test_no_32_piece_assumption(self):
        grid = empty()
        grid[0][4], grid[9][4], grid[6][0] = "k", "K", "P"
        self.assertEqual(grid_reason(grid, np.ones((10, 9)), self.rec), "")
        self.assertEqual(grid_reason(empty(), np.ones((10, 9)), self.rec), "")

    def test_unknown_low_confidence_and_excess_rejected(self):
        grid = empty()
        grid[0][0] = "x"
        self.assertTrue(grid_reason(grid, np.ones((10, 9)), self.rec))
        self.assertTrue(grid_reason(empty(), np.zeros((10, 9)), self.rec))
        grid = empty()
        grid[0][:3] = ["r"]*3
        self.assertTrue(grid_reason(grid, np.ones((10, 9)), self.rec))

    def test_missing_auxiliary_returns_unknown(self):
        self.assertEqual(self.rec.query_mobilenet_for_cell(np.zeros((564, 514, 3), np.uint8), 0, 0, ["r"]), ("x", 0.0))

    def test_refinement_preserves_red_unknown_and_legal_elephant(self):
        grid = empty()
        grid[9][4], grid[7][4], grid[3][0], grid[8][0] = "K", "B", "p", "x"
        refined, _, _ = self.rec.refine_with_physics_and_rules(
            np.full((500, 450, 3), 127, np.uint8), grid, np.ones((10, 9)).tolist())
        self.assertEqual(refined, grid)

    def test_visual_labels_do_not_obey_game_position_priors(self):
        grid = empty()
        grid[0][0] = "A"
        refined, _, _ = self.rec.refine_with_physics_and_rules(
            np.zeros((500, 450, 3), np.uint8), grid, np.ones((10, 9)).tolist())
        self.assertEqual(refined[0][0], "A")
        self.assertNotIn("A", self.rec.get_allowed_classes(True, 0, 0))

    def test_strong_elephants_across_river_are_preserved(self):
        grid = empty()
        grid[2][1], grid[6][6] = "B", "b"
        refined, _, _ = self.rec.refine_with_physics_and_rules(
            np.zeros((500, 450, 3), np.uint8), grid, np.full((10, 9), .95).tolist())
        self.assertEqual(refined, grid)

    def test_rotation_consensus_can_correct_weak_type_without_position_priors(self):
        grid = empty()
        grid[7][6] = "B"
        grid[2][1] = grid[2][6] = "B"
        self.rec.piece_input_name = "input"
        self.rec.piece_session = Mock()
        probabilities = np.full(14, .1/13, np.float32)
        probabilities[11] = .9  # red knight N
        self.rec.piece_session.run.return_value = [np.log(probabilities)[None]]
        confidence = np.full((10, 9), .95)
        confidence[7][6] = .75
        refined, probability, _ = self.rec.refine_with_physics_and_rules(
            np.full((500, 450, 3), 127, np.uint8), grid, confidence.tolist())
        self.assertEqual(refined[7][6], "N")
        self.assertLessEqual(probability[7][6], .75)
        self.assertEqual(self.rec.piece_session.run.call_count, 4)

    def test_auxiliary_does_not_override_weak_but_consistent_primary_type(self):
        grid = empty()
        grid[3][8] = "p"
        self.rec.piece_session = Mock()
        confidence = np.full((10, 9), .95)
        confidence[3][8] = .75
        refined, _, _ = self.rec.refine_with_physics_and_rules(
            np.full((500, 450, 3), 127, np.uint8), grid, confidence.tolist())
        self.assertEqual(refined[3][8], "p")
        self.rec.piece_session.run.assert_not_called()

    def test_resolved_count_conflict_does_not_reclassify_stronger_pieces(self):
        grid = empty()
        grid[2][1] = grid[2][6] = grid[7][6] = "B"
        confidence = np.full((10, 9), .95)
        confidence[7][6] = .75
        self.rec.query_rotated_piece = Mock(return_value=("N", .9))
        refined, _, _ = self.rec.refine_with_physics_and_rules(
            np.full((500, 450, 3), 127, np.uint8), grid, confidence.tolist())
        self.assertEqual(refined[7][6], "N")
        self.assertEqual(refined[2][1], "B")
        self.assertEqual(refined[2][6], "B")
        self.rec.query_rotated_piece.assert_called_once()

    def test_strong_conflicting_primary_requires_stronger_auxiliary_evidence(self):
        grid = empty()
        grid[2][1] = grid[2][6] = grid[7][6] = "B"
        confidence = np.full((10, 9), .99)
        confidence[7][6] = .94
        self.rec.query_rotated_piece = Mock(return_value=("N", .84))
        refined, _, _ = self.rec.refine_with_physics_and_rules(
            np.full((500, 450, 3), 127, np.uint8), grid, confidence.tolist())
        self.assertEqual(refined[7][6], "B")
        self.rec.query_rotated_piece = Mock(return_value=("N", .9))
        refined, probabilities, _ = self.rec.refine_with_physics_and_rules(
            np.full((500, 450, 3), 127, np.uint8), grid, confidence.tolist())
        self.assertEqual(refined[7][6], "N")
        self.assertLessEqual(probabilities[7][6], .9)

    def test_neutral_ink_is_not_proof_of_black(self):
        patch = np.full((36, 36, 3), 200, np.uint8)
        patch[13:23, 13:23] = (40, 40, 40)
        self.assertIsNone(self.rec.ink_color_evidence(patch))
        patch[13:23, 13:23] = (40, 40, 120)
        self.assertIs(self.rec.ink_color_evidence(patch), True)

    def test_auxiliary_winner_must_be_global_and_confidence_not_inflated(self):
        probabilities = np.full(14, .09/12, np.float32)
        probabilities[13], probabilities[7] = .49, .42  # R, A
        self.rec.piece_input_name = "input"
        self.rec.piece_session = Mock()
        self.rec.piece_session.run.return_value = [np.log(probabilities)[None]]
        padded = np.zeros((564, 514, 3), np.uint8)
        ch, confidence = self.rec.query_mobilenet_for_cell(padded, 9, 0, ["N", "R", "C"])
        self.assertEqual(ch, "R")
        self.assertAlmostEqual(confidence, .49, places=5)
        probabilities[13], probabilities[7] = .42, .49
        self.rec.piece_session.run.return_value = [np.log(probabilities)[None]]
        self.assertEqual(self.rec.query_mobilenet_for_cell(padded, 9, 0, ["N", "R", "C"]), ("x", 0.0))

    def test_probability_contract(self):
        probs = np.full((90, 16), 1/16, np.float32)
        np.testing.assert_allclose(self.rec.class_probabilities(probs), probs)
        np.testing.assert_allclose(self.rec.class_probabilities(np.zeros((90, 16))), probs)
        with self.assertRaises(ValueError):
            self.rec.class_probabilities(np.full((90, 16), np.nan))


if __name__ == "__main__":
    unittest.main()
