import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import patch
import numpy as np
import cv2

import config
from kinematics import KinematicsSolver
from control import ChessArmController
from recognition.onnx_pipeline.chess_recognizer import ChessRecognizer


class FakeHardware:
    def __init__(self):
        self.moves = []
        self.magnets = []
        self.closed = 0

    def set_pwm_us(self, channel, pulse):
        self.moves.append((channel, pulse))

    def set_magnet(self, enabled):
        self.magnets.append(enabled)

    def close(self):
        self.closed += 1


class TestMillimeterCompensation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = ChessRecognizer(enable_physics_refine=True)
        test_img_path = Path(__file__).resolve().parents[1] / "test photos" / "IMG_20261102_171009.jpg"
        cls.test_img = cv2.imread(str(test_img_path))

    def test_full_board_piece_offsets_extraction(self):
        """验证全盘 32 颗棋子的亚像素圆心与毫米偏置提取能力"""
        self.assertIsNotNone(self.test_img, "Test image not found!")
        res = self.rec.recognize_full(self.test_img)

        # 1. 验证输出结构
        self.assertIn("piece_offsets", res)
        piece_offsets = res["piece_offsets"]
        self.assertEqual(len(piece_offsets), 32, f"Expected 32 piece offsets, got {len(piece_offsets)}")

        # 2. 检查关键字段与数值范围
        for pos_str, info in piece_offsets.items():
            self.assertIn("nominal_px", info)
            self.assertIn('valid',info)
            self.assertFalse(info['control_valid'])  # Image is not a physical calibration.
            if not info['valid']:
                self.assertTrue(info['reason'])
                self.assertIsNone(info['robot_offset_mm'])
                continue
            self.assertIn("detected_px", info)
            self.assertIn("delta_px", info)
            self.assertIn("board_offset_mm", info)
            self.assertIn("robot_offset_mm", info)
            self.assertIn("offset_dist_mm", info)

            # 机械臂偏置与棋盘偏置坐标系转换规律: robot_dx = -board_dx, robot_dy = +board_dy
            b_dx, b_dy = info["board_offset_mm"]
            r_dx, r_dy = info["robot_offset_mm"]
            self.assertAlmostEqual(r_dx, -b_dx, places=2)
            self.assertAlmostEqual(r_dy, b_dy, places=2)

            # 偏移距离应在合理棋子物理半径范围内 (< 15 mm)
            self.assertLess(info["offset_dist_mm"], 15.0, f"Offset too large at {pos_str}: {info['offset_dist_mm']}mm")
            # 拟合半径应接近棋子实物像素半径 (15~18 px)
            self.assertGreaterEqual(info["radius_px"], 12.0)
            self.assertLessEqual(info["radius_px"], 21.0)
        self.assertGreater(sum(info['valid'] for info in piece_offsets.values()),0)

    def test_kinematics_reachability_with_perturbations(self):
        """验证全盘 90 格点及 ±10mm 极端物理偏移下的 100% 逆解可达性"""
        perturbations = [
            (0.0, 0.0),
            (+10.0, 0.0), (-10.0, 0.0),
            (0.0, +10.0), (0.0, -10.0),
            (+10.0, +10.0), (-10.0, +10.0),
            (+10.0, -10.0), (-10.0, -10.0)
        ]

        unreachable = []
        for c in range(9):
            col_char = "abcdefghi"[c]
            for r in range(10):
                pos_str = f"{col_char}{r}"
                for dx, dy in perturbations:
                    xyz = ChessArmController.board_to_physical(pos_str, offset_x=dx, offset_y=dy)
                    angles = KinematicsSolver.solve_ik(*xyz)
                    if angles is None:
                        unreachable.append((pos_str, dx, dy, xyz))

        self.assertEqual(len(unreachable), 0, f"Unreachable targets found with perturbations: {unreachable[:5]}")

    @patch('time.sleep', return_value=None)
    def test_control_pick_and_place_with_offsets(self, _sleep):
        """验证带毫米偏置的抓取与落子全流程"""
        hw = FakeHardware()
        arm = ChessArmController(hardware=hw, auto_home=False)
        arm._stop.wait = lambda timeout=None: False

        # 移动 a0 (偏置 +2.5mm, -1.8mm) 至 a3 (目标名义格点)
        start_off = (+2.5, -1.8)
        end_off = (0.0, 0.0)
        success = arm.pick_and_place("a0", "a3", start_offset=start_off, end_offset=end_off, return_to_wait=False)
        self.assertTrue(success)
        self.assertGreater(len(hw.moves), 0)
        self.assertIn(True, hw.magnets)   # 电磁铁吸附
        self.assertIn(False, hw.magnets)  # 电磁铁释放

    @patch('time.sleep', return_value=None)
    def test_control_capture_with_offsets(self, _sleep):
        """验证带毫米偏置的吃子动作全流程"""
        hw = FakeHardware()
        arm = ChessArmController(hardware=hw, auto_home=False)
        arm._stop.wait = lambda timeout=None: False

        target_off = (-3.2, +1.5)
        start_off = (+1.0, -0.5)
        success = arm.capture("c3", "c6", target_offset=target_off, start_offset=start_off, return_to_wait=False)
        self.assertTrue(success)
        self.assertGreater(len(hw.moves), 0)
        self.assertGreaterEqual(hw.magnets.count(True), 2)   # 两次吸取 (吃子+己方移位)
        self.assertGreaterEqual(hw.magnets.count(False), 2)  # 两次释放 (墓地+目标格点)


if __name__ == '__main__':
    unittest.main()
