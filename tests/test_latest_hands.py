"""The hand worker must continue without UI polling and never queue old frames."""
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from localization.live_vision import CameraFrame, LatestHands


def packet(seq):
    return CameraFrame(seq,float(seq),np.full((2,2,3),seq,np.uint8))


def wait_for(predicate):
    deadline=time.monotonic()+2
    while time.monotonic()<deadline:
        if predicate():return True
        threading.Event().wait(.005)
    return False


class TestLatestHands(unittest.TestCase):
    def test_continues_without_ui_polling_and_ignores_duplicate_camera_frame(self):
        camera=SimpleNamespace(packet=packet(1))
        camera.read=lambda:camera.packet
        first,second=threading.Event(),threading.Event()
        detector=Mock()
        def detect(image,at):
            (first if at==1 else second).set()
            return 0
        detector.detect.side_effect=detect
        worker=LatestHands(camera,detector,.005).start()
        try:
            self.assertTrue(first.wait(2))
            camera.packet=packet(2)
            # No calls to worker.read() schedule this work.
            self.assertTrue(second.wait(2))
            self.assertTrue(wait_for(lambda:worker.read() is not None and worker.read().frame_id==2))
            threading.Event().wait(.03)
            self.assertEqual(detector.detect.call_count,2)
            self.assertEqual(worker.read().captured_at,2)
        finally:worker.close()
        detector.close.assert_called_once()

    def test_busy_detector_skips_intermediate_frames_instead_of_queuing(self):
        camera=SimpleNamespace(packet=packet(1));camera.read=lambda:camera.packet
        entered,release=threading.Event(),threading.Event();calls=[];detector=Mock()
        def detect(image,at):
            calls.append(at)
            if at==1:entered.set();release.wait(2)
            return 0
        detector.detect.side_effect=detect
        worker=LatestHands(camera,detector,0).start()
        try:
            self.assertTrue(entered.wait(2))
            camera.packet=packet(2);camera.packet=packet(3);release.set()
            self.assertTrue(wait_for(lambda:worker.read() is not None and worker.read().frame_id==3))
            self.assertEqual(calls,[1,3])
        finally:release.set();worker.close()

    def test_detection_failure_clears_previous_negative_result(self):
        camera=SimpleNamespace(packet=packet(1));camera.read=lambda:camera.packet
        detector=Mock()
        def detect(image,at):
            if at==2:raise RuntimeError('hand inference failed')
            return 0
        detector.detect.side_effect=detect
        worker=LatestHands(camera,detector,.005).start()
        try:
            self.assertTrue(wait_for(lambda:worker.read() is not None))
            camera.packet=packet(2)
            self.assertTrue(wait_for(lambda:worker.error is not None))
            self.assertIsNone(worker.read())
            self.assertIn('hand inference failed',worker.error)
        finally:worker.close()
        detector.close.assert_called_once()

    def test_closing_before_start_is_safe_and_idempotent(self):
        detector=Mock();camera=Mock()
        worker=LatestHands(camera,detector)
        worker.close();worker.close()
        detector.close.assert_called_once()


if __name__=='__main__':unittest.main()
