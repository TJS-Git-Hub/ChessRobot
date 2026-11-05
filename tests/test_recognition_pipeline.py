"""
中国象棋盘面识别系统全自动回归与集成测试套件 (Test Recognition Pipeline)
验证项：
1. 模型加载与会话就绪 (RTMPose + SwinV2 + MobileNetV3)
2. 人工复核过的非对称开发样本：90格与FEN回归，非独立盲测
3. 本地最多五张空盘照片上的棋子误检回归（不是独立泛化评估）
"""

import os
import sys
import unittest
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recognition.onnx_pipeline.chess_recognizer import ChessRecognizer


class TestRecognitionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer = ChessRecognizer()

    def test_model_sessions_loaded(self):
        """验证所有必要 ONNX 模型会话正常加载"""
        self.assertIsNotNone(self.recognizer.pose_session)
        self.assertIsNotNone(self.recognizer.classifier_session)
        self.assertIsNotNone(self.recognizer.piece_session)

    def test_asymmetric_real_board_exact(self):
        """Existing manually reviewed development photo: all 90 cells, not just counts."""
        import json,hashlib
        fixture_path=os.path.join(PROJECT_ROOT,'tests','fixtures','asymmetric_board.json')
        with open(fixture_path,encoding='utf8') as stream:fixture=json.load(stream)
        image_path=os.path.join(PROJECT_ROOT,fixture['image'])
        if not os.path.isfile(image_path):self.skipTest('Local development photograph is not distributed with weights/source')
        with open(image_path,'rb') as stream:self.assertEqual(hashlib.sha256(stream.read()).hexdigest(),fixture['sha256'])
        image=cv2.imread(image_path)
        corners,_=self.recognizer.predict_corners(image)
        warped,_=self.recognizer.warp_chessboard(image,corners)
        grid,_,_=self.recognizer.classify_board(warped)
        truth=[list(row) for row in fixture['grid']]
        self.assertEqual(grid,truth)
        from localization.vision_contract import grid_to_fen
        self.assertEqual(self.recognizer.format_fen(grid),grid_to_fen(truth))

    def test_empty_board_cleanliness(self):
        """验证纯空棋盘 0 误检"""
        empty_dir = os.path.join(PROJECT_ROOT, "recognition", "data", "raw_dataset", "empty_board")
        if not os.path.isdir(empty_dir):self.skipTest('Empty-board photographs missing')
        if os.path.exists(empty_dir):
            files = [f for f in os.listdir(empty_dir) if f.lower().endswith(('.jpg', '.png'))][:5]
            if not files:self.skipTest('No empty-board images available')
            for fname in files:
                p = os.path.join(empty_dir, fname)
                img = cv2.imread(p)
                corners, _ = self.recognizer.predict_corners(img)
                warped, _ = self.recognizer.warp_chessboard(img, corners)
                grid, _, _ = self.recognizer.classify_board(warped)
                pieces = [grid[r][c] for r in range(10) for c in range(9) if grid[r][c] not in ['.', 'x']]
                self.assertEqual(len(pieces), 0, f"空盘 {fname} 不应误检出任何棋子，实际误检: {pieces}")


if __name__ == '__main__':
    unittest.main()

