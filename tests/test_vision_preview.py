"""Preview annotations must not cover real piece faces or alter observations."""
import unittest
import numpy as np
from tools.vision_live import PreviewAnchors, PreviewLabels, coordinate_box, coordinate_leader, piece_face_circles, publish_snapshot, render
import copy
from unittest.mock import Mock, patch


class TestPreviewLayout(unittest.TestCase):
    def test_leader_identifies_piece_rim_without_crossing_glyph(self):
        start,end=coordinate_leader((100,100),20,(140,90,160,110))
        np.testing.assert_allclose(start,[140,100])
        np.testing.assert_allclose(end,[121,100])

    def test_leader_does_not_cross_neighbouring_piece(self):
        circles=[(100,100,20),(145,100,18)]
        box=coordinate_box((100,100),20,(20,16),circles,[],(0,0,300,300),preferred=(180,92))
        self.assertIsNotNone(box)
        self.assertNotEqual(box,(180,92,200,108))

    def test_coordinate_avoids_piece_faces(self):
        circles = [(100, 100, 20), (145, 100, 20), (100, 145, 20)]
        box = coordinate_box((100, 100), 20, (20, 16), circles, [], (0, 0, 300, 300))
        self.assertIsNotNone(box)
        for x, y, radius in circles:
            nearest = (np.clip(x, box[0], box[2]), np.clip(y, box[1], box[3]))
            self.assertGreaterEqual(np.linalg.norm(np.asarray(nearest)-[x, y]), radius+2)

    def test_coordinate_omitted_when_free_space_unavailable(self):
        self.assertIsNone(coordinate_box((20, 20), 20, (20, 16), [(20, 20, 100)],
                                         [], (0, 0, 40, 40)))

    def test_small_display_jitter_held_but_real_move_not_hidden(self):
        anchors = PreviewAnchors()
        np.testing.assert_allclose(anchors.point("a0", [100, 100]), [100, 100])
        np.testing.assert_allclose(anchors.point("a0", [100.5, 99.5]), [100, 100])
        np.testing.assert_allclose(anchors.point("a0", [120, 100]), [120, 100])
        anchors.clear()
        self.assertEqual(anchors.points, {})

    def test_coordinate_keeps_previous_clear_location(self):
        anchors = PreviewAnchors()
        circles = [(100, 100, 20)]
        box = anchors.box("a7", (100, 100), 20, (20, 16), circles, [], (0, 0, 300, 300))
        # A previously allocated left-side box should not jump to the default
        # right-side candidate simply because both are free on the next frame.
        anchors.box_offsets["a7"] = (-60, -8)
        preferred = anchors.box("a7", (100, 100), 20, (20, 16), circles, [], (0, 0, 300, 300))
        self.assertEqual(preferred, (40, 92, 60, 108))
        blocked = anchors.box("a7", (100, 100), 20, (20, 16), circles+[(50, 100, 10)],
                              [], (0, 0, 300, 300))
        self.assertNotEqual(blocked, preferred)
        self.assertIsNotNone(box)
        anchors.clear()
        self.assertEqual(anchors.box_offsets, {})

    def test_coordinate_follows_slow_camera_translation(self):
        anchors = PreviewAnchors()
        first = anchors.box("a7", (100, 100), 20, (20, 16), [(100, 100, 20)], [], (0, 0, 300, 300))
        for step in range(1, 31):
            dx = step*.2
            last = anchors.box("a7", (100+dx, 100), 20, (20, 16),
                               [(100+dx, 100, 20)], [], (0, 0, 300, 300))
        self.assertEqual(last[0]-first[0], 6)

    def test_actual_round_piece_rim_adds_display_obstacle(self):
        import cv2
        frame = np.full((160, 240, 3), 100, np.uint8)
        cv2.circle(frame, (100, 80), 22, (220, 220, 220), -1)
        circles = piece_face_circles(frame)
        self.assertTrue(any(abs(x-100)<2 and abs(y-80)<2 and radius>22 for x,y,radius in circles))

    def test_busy_diagnostic_reader_does_not_crash_capture(self):
        pending = Mock()
        pending.replace.side_effect = PermissionError("reader denies delete sharing")
        self.assertFalse(publish_snapshot(pending, "snapshot.jpg"))
        pending.replace.side_effect = OSError("unrelated storage failure")
        with self.assertRaises(OSError):
            publish_snapshot(pending, "snapshot.jpg")

    def test_rim_obstacles_survive_short_edge_dropout_and_do_not_shrink_with_noise(self):
        anchors = PreviewAnchors()
        with patch("tools.vision_live.piece_face_circles", side_effect=[
                [(100,100,20)], [(100.4,100,22)], [], []]):
            anchors.obstacles(None, 1)
            circles = anchors.obstacles(None, 1.1)
            self.assertEqual(circles[0][2], 22)
            self.assertTrue(anchors.obstacles(None, 1.3))
            self.assertFalse(anchors.obstacles(None, 1.7))
        anchors.clear()
        self.assertEqual(anchors.faces, [])


def observation(frame, piece="r", confidence=.9):
    grid = [["."]*9 for _ in range(10)]
    probabilities = np.full((10, 9), .9).tolist()
    grid[7][0], probabilities[7][0] = piece, confidence
    return dict(frame_id=frame, grid_chars=grid, conf_grid=probabilities)


class TestPreviewMemory(unittest.TestCase):
    def test_ambiguity_holds_glyph_without_changing_observation(self):
        labels = PreviewLabels()
        labels.update(observation(1), 1)
        raw = observation(2, "x", 0)
        original = copy.deepcopy(raw)
        shown, held = labels.update(raw, 1)
        self.assertEqual(shown["a7"], "r")
        self.assertIn("a7", held)
        self.assertEqual(raw, original)

    def test_three_distinct_good_results_clear_marker(self):
        labels = PreviewLabels()
        labels.update(observation(1), 1)
        labels.update(observation(2, "x", 0), 1)
        for _ in range(20):
            shown, held = labels.update(observation(3), 1)
        self.assertIn("a7", held)
        labels.update(observation(4), 1)
        _, held = labels.update(observation(5), 1)
        self.assertNotIn("a7", held)

    def test_changed_class_needs_two_results_and_empty_is_not_held_as_piece(self):
        labels = PreviewLabels()
        labels.update(observation(1), 1)
        shown, held = labels.update(observation(2, "c"), 1)
        self.assertEqual(shown["a7"], "r")
        self.assertIn("a7", held)
        shown, held = labels.update(observation(3, "c"), 1)
        self.assertEqual(shown["a7"], "c")
        self.assertNotIn("a7", held)
        labels.update(observation(4, "."), 1)
        labels.update(observation(5, "."), 1)
        shown, held = labels.update(observation(6, "x", 0), 1)
        self.assertEqual(shown["a7"], "x")

    def test_motion_epoch_and_expiry_discard_old_glyph(self):
        labels = PreviewLabels()
        labels.update(observation(1), 1)
        shown, held = labels.update(observation(2, "x", 0), 2)
        self.assertEqual(shown["a7"], "x")
        labels.update(observation(3), 2)
        self.assertEqual(labels.update(None, 2), ({}, set()))
        shown, held = labels.update(observation(4, "x", 0), 2)
        self.assertEqual(shown["a7"], "x")

    def test_unconfirmed_fresh_candidate_cannot_change_display_identity(self):
        labels = PreviewLabels()
        labels.update(observation(1), 1)
        shown, held = labels.update(observation(2, "c"), 1, confirmed=False)
        self.assertEqual(shown["a7"], "r")
        self.assertIn("a7", held)
        labels.update(None, 1)
        shown, held = labels.update(observation(3, "c"), 1, confirmed=False)
        self.assertNotIn("a7", shown)

    def test_rendering_fresh_candidate_cannot_publish_old_display_identity(self):
        from PIL import ImageFont
        from localization.live_vision import LiveGate
        import time
        from types import SimpleNamespace
        gate = LiveGate()
        gate.epoch = 1
        gate.quiet_since = time.monotonic()-1
        raw = observation(2, "c")
        raw.update(matrix=np.eye(3).tolist(), captured_at=time.monotonic(), inference_ms=100)
        original = copy.deepcopy(raw)
        gate.last_result = raw
        labels = PreviewLabels()
        labels.update(observation(1), 1)
        camera = SimpleNamespace(actual_size=(1280,720), capture_fps=20, preview_fps=20)
        font = ImageFont.load_default()
        diagnostics = {}
        render(np.full((720,1280,3),100,np.uint8), gate, camera, (font,font,font), "", False, 0,
               show_coordinates=False, labels=labels, diagnostics=diagnostics)
        self.assertEqual(diagnostics["labels"]["a7"], "r")
        self.assertIn("a7", diagnostics["held"])
        self.assertTrue(diagnostics["unconfirmed_candidate"])
        self.assertIsNone(gate.output)
        self.assertEqual(gate.last_result, original)
        raw["captured_at"] -= 3
        render(np.full((720,1280,3),100,np.uint8), gate, camera, (font,font,font), "", False, 0,
               show_coordinates=False, labels=labels, diagnostics=diagnostics)
        self.assertEqual(diagnostics["labels"], {})


if __name__ == "__main__":
    unittest.main()
