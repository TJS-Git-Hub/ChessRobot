"""Recovery must add visual evidence without erasing unknowns or frame safety."""
import copy
import unittest
from unittest.mock import patch
from unittest.mock import Mock
import contextlib
import io
import numpy as np

from localization.live_vision import (
    CameraFrame, LiveGate, observe_board, consensus_grid, exposure_view,
    grid_edge_corners, guarded_recovery,
)
from recognition.onnx_pipeline.chess_recognizer import ChessRecognizer
from tools.evaluate_vision_lighting import compare_grid, summarize


def view(piece="n", confidence=.9, rejection=""):
    grid = [["."]*9 for _ in range(10)]
    grid[2][6] = piece
    probabilities = [[.9]*9 for _ in range(10)]
    probabilities[2][6] = confidence
    return {"grid_chars": grid, "conf_grid": probabilities, "rejection": rejection,
            "corners": [[50, 50], [400, 50], [50, 450], [400, 450]],
            "corner_scores": [.3]*4, "matrix": np.eye(3).tolist()}


class TestLightingRecovery(unittest.TestCase):
    def setUp(self):
        self.rec = ChessRecognizer.__new__(ChessRecognizer)
        self.image = np.full((500, 450, 3), 128, np.uint8)

    def test_valid_board_has_no_extra_inference(self):
        with patch("localization.live_vision._single_board", return_value=view()) as classify:
            result = observe_board(self.rec, self.image)
        self.assertEqual(classify.call_count, 1)
        self.assertFalse(result["recovery"]["attempted"])

    def test_verified_geometry_stabilizes_crop_but_still_checks_fresh_pose(self):
        corners=np.float32([[50,50],[400,50],[50,450],[400,450]])
        self.rec.predict_corners=Mock(return_value=(corners+2,np.full(4,.3)))
        self.rec.warp_chessboard=Mock(return_value=(self.image,np.eye(3)))
        sample=view()
        self.rec.classify_board=Mock(return_value=(sample['grid_chars'],sample['conf_grid'],''))
        result=observe_board(self.rec,self.image,verified_geometry=corners,allow_recovery=False)
        self.rec.predict_corners.assert_called_once()
        np.testing.assert_allclose(self.rec.warp_chessboard.call_args.args[1],corners)
        self.assertEqual(result['geometry_source'],'verified_tracking_with_fresh_pose_check')

    def test_missing_or_conflicting_fresh_pose_cannot_use_tracked_crop(self):
        corners=np.float32([[50,50],[400,50],[50,450],[400,450]])
        for predicted,scores in [(corners,np.zeros(4)),
                                  (np.float32([[80,80],[380,80],[80,420],[380,420]]),np.full(4,.3))]:
            self.rec.predict_corners=Mock(return_value=(predicted,scores))
            self.rec.classify_board=Mock()
            result=observe_board(self.rec,self.image,verified_geometry=corners,allow_recovery=False)
            self.assertIsNone(result['matrix'])
            self.assertTrue(result['rejection'])
            self.rec.classify_board.assert_not_called()

    def test_two_strong_views_can_recover_ambiguous_board(self):
        with patch("localization.live_vision._single_board", side_effect=[
                view("c", rejection="count conflict"), view(), view()]) as classify:
            result = observe_board(self.rec, self.image)
        self.assertEqual(result["grid_chars"][2][6], "n")
        self.assertFalse(result["rejection"])
        self.assertTrue(result["recovery"]["used"])
        self.assertTrue(all(call.args[0] is self.rec for call in classify.call_args_list))

    def test_unknown_base_is_not_erased_by_photometric_voting(self):
        with patch("localization.live_vision._single_board",
                   return_value=view("x", 0, "unknown")) as classify:
            result = observe_board(self.rec, self.image)
        self.assertEqual(classify.call_count, 1)
        self.assertEqual(result["grid_chars"][2][6], "x")
        self.assertTrue(result["rejection"])

    def test_one_weak_supporting_vote_cannot_recover(self):
        with patch("localization.live_vision._single_board", side_effect=[
                view("c", rejection="count conflict"), view(confidence=.49), view()]):
            result = observe_board(self.rec, self.image)
        self.assertFalse(result["recovery"]["used"])
        self.assertTrue(result["rejection"])

    def test_different_corner_geometry_does_not_vote(self):
        alternative = view()
        alternative["corners"] = (np.asarray(alternative["corners"])+100).tolist()
        with patch("localization.live_vision._single_board", side_effect=[
                view("c", rejection="count conflict"), alternative]):
            result = observe_board(self.rec, self.image)
        self.assertFalse(result["recovery"]["used"])

    def test_photometric_views_are_not_independent_live_frames(self):
        gate = LiveGate()
        gate.observe(CameraFrame(1, 10, self.image), 10)
        gate.observe(CameraFrame(2, 10.4, self.image), 10.4)
        result = view()
        result.update(captured_at=10.4, frame_id=2, recovery={"used": True, "views": 3})
        self.assertFalse(gate.accept((gate.epoch, 2), result, 10.5))
        self.assertIsNone(gate.output)

    def test_consensus_confidence_is_not_inflated(self):
        grid, confidence = consensus_grid([view(confidence=.95), view(confidence=.8), view(confidence=.6)])
        self.assertEqual(grid[2][6], "n")
        self.assertAlmostEqual(confidence[2][6], .8)

    def test_exposure_identity_and_source_unchanged(self):
        source = np.arange(256, dtype=np.uint8).reshape((16, 16, 1)).repeat(3, axis=2)
        original = source.copy()
        np.testing.assert_array_equal(exposure_view(source, 0), source)
        exposure_view(source, .5)
        np.testing.assert_array_equal(source, original)

    def test_guarded_recovery_preserves_strong_primary_and_original_unknown(self):
        base = view("n", .95)
        others = [view("c", .99) for _ in range(5)]
        grid, confidence = guarded_recovery([base]+others)
        self.assertEqual(grid[2][6], "n")
        self.assertEqual(confidence[2][6], .95)
        grid, _ = guarded_recovery([view("x", 0)]+others)
        self.assertEqual(grid[2][6], "x")

    def test_guarded_recovery_rejects_tie_and_requires_three_votes(self):
        grid, _ = guarded_recovery([view("c", .6)]*3+[view("n", .8)]*3)
        self.assertEqual(grid[2][6], "x")
        grid, confidence = guarded_recovery([view("c", .6)]+[view("n", .8)]*3)
        self.assertEqual(grid[2][6], "n")
        self.assertEqual(confidence[2][6], .8)

    def test_boundary_refinement_abstains_without_four_boundaries(self):
        self.assertIsNone(grid_edge_corners(self.image))
        image = np.full((500, 450, 3), 220, np.uint8)
        import cv2
        cv2.line(image, (40, 50), (30, 450), (0, 0, 0), 2)
        self.assertIsNone(grid_edge_corners(image))

    def test_count_conflict_needs_three_strong_geometric_classifications(self):
        base = view("B", .95)
        base["grid_chars"][2][1] = base["grid_chars"][2][2] = "B"
        geometric = [view("N", .8) for _ in range(3)]
        grid, confidence = guarded_recovery([base]+geometric)
        self.assertEqual(grid[2][6], "N")
        self.assertEqual(confidence[2][6], .8)
        geometric[0]["conf_grid"][2][6] = .64
        grid, _ = guarded_recovery([base]+geometric)
        self.assertEqual(grid[2][6], "B")

    def test_boundary_refinement_recovers_displaced_grid_edges(self):
        import cv2
        image = np.full((500, 450, 3), 220, np.uint8)
        for y in np.linspace(50, 450, 10):
            cv2.line(image, (40, round(y)), (395, round(y)), (0, 0, 0), 2)
        for x in np.linspace(40, 395, 9):
            cv2.line(image, (round(x), 50), (round(x), 450), (0, 0, 0), 2)
        corners = grid_edge_corners(image)
        self.assertIsNotNone(corners)
        np.testing.assert_allclose(corners, [[40, 50], [395, 50], [40, 450], [395, 450]], atol=3)


class TestEvaluationMetrics(unittest.TestCase):
    def test_wrong_side_miss_unknown_and_false_piece_count_separately(self):
        truth = view()["grid_chars"]
        truth[0][0], truth[0][1] = "r", "n"
        prediction = copy.deepcopy(truth)
        prediction[2][6] = "N"
        prediction[0][0], prediction[0][1], prediction[1][1] = ".", "x", "p"
        counts, _, errors = compare_grid(truth, prediction)
        self.assertEqual(counts["pieces"], 3)
        self.assertEqual(counts["wrong_side"], 1)
        self.assertEqual(counts["misses"], 1)
        self.assertEqual(counts["uncertain_pieces"], 1)
        self.assertEqual(counts["false_pieces"], 1)
        self.assertEqual(len(errors), 4)

    def test_empty_boards_do_not_get_a_piece_accuracy_of_100(self):
        grid = [["."]*9 for _ in range(10)]
        counts, classes, errors = compare_grid(grid, grid)
        result = summarize([{"counts": counts, "classes": classes, "errors": errors,
                             "position_id": "empty", "rejection": "", "inference_ms": 1}])
        self.assertIsNone(result["piece_accuracy_pct"])


class TestImageEntrypointFailures(unittest.TestCase):
    def test_no_input_or_sample_does_not_fabricate_dummy(self):
        from tools import test_offline_onnx as entry
        rec = Mock()
        with patch("sys.argv", ["test"]), patch.object(
                entry, "ChessRecognizer", return_value=rec), patch.object(
                entry.os, "listdir", return_value=[]):
            with self.assertRaises(ValueError):
                entry.main()
        rec.recognize_full.assert_not_called()

    def test_missing_explicit_image_fails_before_model_load(self):
        from tools import test_offline_onnx as entry
        with (
            patch("sys.argv", ["test", "--image", "__missing_chess_image_478234.png"]),
            patch.object(entry, "ChessRecognizer") as constructor,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit):
                entry.main()
        constructor.assert_not_called()

    def test_camera_failure_never_falls_back_to_sample(self):
        from tools import test_offline_onnx as entry
        for opened in (False, True):
            with self.subTest(opened=opened):
                cap, rec = Mock(), Mock()
                cap.isOpened.return_value = opened
                cap.read.return_value = (False, None)
                with patch("sys.argv", ["test", "--cam"]), patch.object(
                        entry, "ChessRecognizer", return_value=rec), patch.object(
                        entry.cv2, "VideoCapture", return_value=cap):
                    with self.assertRaises(RuntimeError):
                        entry.main()
                cap.release.assert_called_once()
                rec.recognize_full.assert_not_called()


if __name__ == "__main__":
    unittest.main()
