"""
ChessRecognizer: 基于 ONNX 的中国象棋盘面两阶段识别器与几何中心测量器
两阶段架构：
  Stage 1: RTMPose ONNX 预测棋盘 4 角点 (A0, A8, J0, J8) -> 单应性/透视变换拉平棋盘
  Stage 2: Swin Transformer V2 ONNX 对 10×9 个交叉点直接输出 16 类分类 (90, 16)
  Geometry: 针对有子网格进行局部高分辨率几何轮廓/霍夫圆拟合，计算物理毫米偏移 (dx, dy)
"""

import os
import sys
import time
from typing import Tuple, List, Dict, Optional, Union
import collections
import numpy as np
import cv2
import onnxruntime as ort

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


# 16 类标签定义
DICT_CATE_NAMES = {
    'point': '.',
    'other': 'x',
    'red_king': 'K',
    'red_advisor': 'A',
    'red_bishop': 'B',
    'red_knight': 'N',
    'red_rook': 'R',
    'red_cannon': 'C',
    'red_pawn': 'P',
    'black_king': 'k',
    'black_advisor': 'a',
    'black_bishop': 'b',
    'black_knight': 'n',
    'black_rook': 'r',
    'black_cannon': 'c',
    'black_pawn': 'p',
}

BONE_NAMES = ["A0", "A8", "J0", "J8"]


class ChessRecognizer:
    def __init__(self,
                 pose_model_path: Optional[str] = None,
                 classifier_model_path: Optional[str] = None,
                 enable_physics_refine: bool = True):
        """
        初始化识别器与 ONNX Runtime 推理会话
        enable_physics_refine: 是否启用物理墨迹色度与方差规则双重纠偏引擎 (默认 True)
        """
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(root_dir, "recognition", "models", "onnx")

        self.pose_model_path = pose_model_path or os.path.join(models_dir, "pose_rtmpose_4kpt.onnx")
        # 默认优先采用泛化能力更稳健、无位置偏见的原始官方基线模型 layout_swinv2_nano.onnx
        nano_path = os.path.join(models_dir, "layout_swinv2_nano.onnx")
        self.classifier_model_path = classifier_model_path or nano_path
        self.piece_model_path = os.path.join(models_dir, "piece_classifier_mobilenetv3.onnx")
        self.enable_physics_refine = enable_physics_refine

        self.pose_session: Optional[ort.InferenceSession] = None
        self.classifier_session: Optional[ort.InferenceSession] = None
        self.piece_session: Optional[ort.InferenceSession] = None

        self.pose_input_name = ""
        self.classifier_input_name = ""
        self.piece_input_name = ""

        self.classes_labels = list(DICT_CATE_NAMES.keys())
        self.label_2_short = DICT_CATE_NAMES
        self.recent_logits_buffer = []  # 多帧平滑缓冲区

        self._load_sessions()

    def reset_buffer(self):
        """清空多帧平滑历史缓冲区"""
        self.recent_logits_buffer.clear()

    def _load_sessions(self):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Leave CPU capacity for capture, tracking, hand detection and drawing.
        # On the current device four threads beat one/two/eight/default in the
        # same-image timing check; this does not change weights or preprocessing.
        opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
        opts.inter_op_num_threads = 1

        if os.path.exists(self.pose_model_path):
            self.pose_session = ort.InferenceSession(self.pose_model_path, opts, providers=['CPUExecutionProvider'])
            self.pose_input_name = self.pose_session.get_inputs()[0].name
            print(f"✅ [ChessRecognizer] RTMPose 姿态模型就绪: {self.pose_model_path}")
        else:
            print(f"⚠️ [ChessRecognizer] 未找到角点模型文件: {self.pose_model_path}")

        if os.path.exists(self.classifier_model_path):
            self.classifier_session = ort.InferenceSession(self.classifier_model_path, opts, providers=['CPUExecutionProvider'])
            self.classifier_input_name = self.classifier_session.get_inputs()[0].name
            print(f"✅ [ChessRecognizer] SwinV2 盘面分类模型就绪: {self.classifier_model_path}")
        else:
            print(f"⚠️ [ChessRecognizer] 未找到分类模型文件: {self.classifier_model_path}")

        if os.path.exists(self.piece_model_path):
            self.piece_session = ort.InferenceSession(self.piece_model_path, opts, providers=['CPUExecutionProvider'])
            self.piece_input_name = self.piece_session.get_inputs()[0].name
            print(f"✅ [ChessRecognizer] MobileNetV3 64x64 单兵高精分类模型就绪: {self.piece_model_path}")
        else:
            print(f"⚠️ [ChessRecognizer] 未找到单兵模型文件: {self.piece_model_path}")

    # =========================================================================
    # 阶段 1：四角点检测 (RTMPose)
    # =========================================================================
    def predict_corners(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测 4 个棋盘最外侧交点坐标
        Returns:
            keypoints: shape (4, 2) [[x, y], ...] 顺序为 A0, A8, J0, J8
            scores: shape (4,) 置信度
        """
        if self.pose_session is None:
            raise RuntimeError("角点检测 ONNX 会话未加载，请检查模型文件路径。")

        h, w = image_bgr.shape[:2]
        input_size = (256, 256)

        # 仿射变换预处理
        bbox = [0, 0, w, h]
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        scale = np.array([w, h], dtype=np.float32) * 1.25

        # 保持比例计算 warp matrix
        scale_w, scale_h = scale
        aspect_ratio = input_size[0] / input_size[1]
        if scale_w > scale_h * aspect_ratio:
            scale_adj = np.array([scale_w, scale_w / aspect_ratio], dtype=np.float32)
        else:
            scale_adj = np.array([scale_h * aspect_ratio, scale_h], dtype=np.float32)

        src_w, src_h = scale_adj
        dst_w, dst_h = input_size

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center
        src[1, :] = center + np.array([src_w * -0.5, 0.], dtype=np.float32)
        src[2, :] = src[1, :] + np.array([0., src_h * -0.5], dtype=np.float32)

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = [0., dst_h * 0.5]
        dst[2, :] = [0., 0.]

        warp_mat = cv2.getAffineTransform(src, dst)
        inv_warp_mat = cv2.getAffineTransform(dst, src)

        img_crop = cv2.warpAffine(image_bgr, warp_mat, input_size, flags=cv2.INTER_LINEAR)

        # 归一化 (BGR -> RGB, mean/std)
        img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        img_norm = (img_rgb - mean) / std
        input_tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]  # (1, 3, 256, 256)

        outputs = self.pose_session.run(None, {self.pose_input_name: input_tensor})
        simcc_x, simcc_y = outputs[0], outputs[1]

        # SimCC 解码
        x_indices = np.argmax(simcc_x[0], axis=1)  # (4,)
        y_indices = np.argmax(simcc_y[0], axis=1)  # (4,)

        x_coords = x_indices / (input_size[0] * 2.0) * input_size[0]
        y_coords = y_indices / (input_size[1] * 2.0) * input_size[1]
        scores = np.max(simcc_x[0], axis=1) * np.max(simcc_y[0], axis=1)

        crop_kpts = np.stack([x_coords, y_coords], axis=1)  # (4, 2)

        # 逆仿射变换映射回原图坐标
        kpts_homo = np.hstack([crop_kpts, np.ones((4, 1), dtype=np.float32)])
        orig_kpts = (inv_warp_mat @ kpts_homo.T).T

        return orig_kpts, scores

    # =========================================================================
    # 阶段 2：透视变换与 90 交点分类 (SwinV2)
    # =========================================================================
    def warp_chessboard(self, image_bgr: np.ndarray, corners: np.ndarray, dst_size=(450, 500)) -> Tuple[np.ndarray, np.ndarray]:
        """
        透视校正棋盘图像
        corners: 4个角点 [A0(黑左底), A8(黑右底), J0(红左底), J8(红右底)]
        """
        padding = 50
        dst_corners = np.float32([
            [padding, padding],                       # A0 左上
            [dst_size[0] - padding, padding],          # A8 右上
            [padding, dst_size[1] - padding],          # J0 左下
            [dst_size[0] - padding, dst_size[1] - padding]  # J8 右下
        ])

        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners)
        warped = cv2.warpPerspective(image_bgr, M, dst_size)
        return warped, M

    def classify_board(self, warped_bgr: np.ndarray) -> Tuple[List[List[str]], List[List[float]], str]:
        """
        对校正后的棋盘图像进行 90 格分类
        Returns:
            grid_chars: 10x9 字符数组 (K, A, B, N, R, C, P, k, a, b, n, r, c, p, ., x)
            confidences: 10x9 置信度浮点数
            layout_str: 10行字符格式盘面字符串
        """
        if self.classifier_session is None:
            raise RuntimeError("分类器 ONNX 会话未加载，请检查模型文件路径。")

        # 中心裁剪 (400, 450)
        h, w = warped_bgr.shape[:2]
        crop_w, crop_h = 400, 450
        start_x = max(0, int((w - crop_w) // 2))
        start_y = max(0, int((h - crop_h) // 2))
        cropped = warped_bgr[start_y:start_y + crop_h, start_x:start_x + crop_w]

        # 判断模型所需输入尺寸 (微调版 Tiny 为 256x256，原始 Nano 为 280x315)
        in_shape = self.classifier_session.get_inputs()[0].shape
        if len(in_shape) == 4 and in_shape[2] == 256 and in_shape[3] == 256:
            input_size = (256, 256)
        elif "finetuned" in self.classifier_model_path:
            input_size = (256, 256)
        else:
            input_size = (280, 315)

        resized = cv2.resize(cropped, input_size)

        # 归一化 (BGR -> RGB)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        norm = (rgb - mean) / std
        tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)

        outputs = self.classifier_session.run(None, {self.classifier_input_name: tensor})
        logits = self.class_probabilities(outputs[0][0])

        # 多帧平滑支持
        self.recent_logits_buffer.append(logits)
        if len(self.recent_logits_buffer) > 5:
            self.recent_logits_buffer.pop(0)

        best_idx = np.argmax(logits, axis=-1)  # (90,)
        conf = logits[np.arange(90), best_idx]

        short_chars = [self.label_2_short[self.classes_labels[idx]] for idx in best_idx]

        grid_chars = [short_chars[i * 9:(i + 1) * 9] for i in range(10)]
        conf_grid = [conf[i * 9:(i + 1) * 9].tolist() for i in range(10)]

        if self.enable_physics_refine:
            grid_chars, conf_grid, layout_str = self.refine_with_physics_and_rules(warped_bgr, grid_chars, conf_grid)
        else:
            layout_str = "\n".join(["".join(row) for row in grid_chars])

        return grid_chars, conf_grid, layout_str

    @staticmethod
    def class_probabilities(values: np.ndarray) -> np.ndarray:
        """Nano exports probabilities; Tiny exports logits. Validate both."""
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (90, 16) or not np.isfinite(values).all():
            raise ValueError("Expected finite classifier output (90, 16)")
        if (values >= 0).all() and (values <= 1).all() and np.allclose(values.sum(-1), 1, atol=1e-3):
            return values
        exp = np.exp(values - values.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)

    def get_allowed_classes(self, is_red: bool, r: int, c: int) -> List[str]:
        """Reachable squares, not the legal moves of the current position."""
        if not (0 <= r < 10 and 0 <= c < 9):
            return []
        allowed = ['N', 'R', 'C']
        rr = 9 - r if is_red else r
        if rr <= 2 and 3 <= c <= 5:
            allowed.append('K')
        if (rr, c) in {(0, 3), (0, 5), (1, 4), (2, 3), (2, 5)}:
            allowed.append('A')
        if (rr, c) in {(0, 2), (0, 6), (2, 0), (2, 4), (2, 8), (4, 2), (4, 6)}:
            allowed.append('B')
        if rr >= 5 or (rr in (3, 4) and c % 2 == 0):
            allowed.append('P')
        return allowed if is_red else [ch.lower() for ch in allowed]

    def query_mobilenet_for_cell(self, padded: np.ndarray, r: int, c: int,
                                 allowed_candidates: List[str],
                                 pad=50, w_step=350.0/8.0, h_step=400.0/9.0, half=32) -> Tuple[str, float]:
        """Return independent model evidence, never a fabricated label."""
        if self.piece_session is None or not allowed_candidates:
            return 'x', 0.0
        chars = ['a', 'b', 'c', 'k', 'n', 'p', 'r', 'A', 'B', 'C', 'K', 'N', 'P', 'R']
        cx = int(round(pad + c * w_step)) + half
        cy = int(round(pad + r * h_step)) + half
        patch = padded[cy - 32:cy + 32, cx - 32:cx + 32]
        if patch.shape[:2] != (64, 64):
            return 'x', 0.0
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = ((rgb - [123.675, 116.28, 103.53]) /
                  [58.395, 57.12, 57.375]).astype(np.float32).transpose(2, 0, 1)[None]
        logits = self.piece_session.run(None, {self.piece_input_name: tensor})[0][0]
        if logits.shape != (14,) or not np.isfinite(logits).all():
            return 'x', 0.0
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        best = int(np.argmax(probs))
        competitors = [float(probs[i]) for i, ch in enumerate(chars)
                       if ch in allowed_candidates and i != best]
        margin = float(probs[best]) - max(competitors, default=0.0)
        if chars[best] not in allowed_candidates or margin < 0.15:
            return 'x', 0.0
        return chars[best], float(probs[best])

    @staticmethod
    def ink_color_evidence(patch: np.ndarray) -> Optional[bool]:
        """White-normalized ink chroma; ambiguous or weak marks have no vote.

        A low absolute R-G is not evidence for black. Normalize ink against the
        bright face of this same piece, making a washed-out red piece abstain.
        """
        h, w = patch.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        disk = (xx-w//2)**2 + (yy-h//2)**2 <= (min(h, w)/3)**2
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        face = float(np.median(gray[disk]))
        ink = disk & (gray < face-12)
        background = disk & (gray >= face-5)
        if ink.sum() < 30 or background.sum() < 20 or face < 30:
            return None
        bgr = patch[ink].astype(float)
        bg = patch[background].astype(float).mean(axis=0)
        normalized = bgr / (bg+1)
        score = float(np.median((normalized[:, 2]-normalized[:, 1]) /
                               (normalized[:, 2]+normalized[:, 1]+1e-3)))
        if score > 0.20:
            return True
        if score < -0.10:
            return False
        return None

    def query_rotated_piece(self, patch: np.ndarray) -> Tuple[str, float]:
        """Require orientation-consistent auxiliary evidence, not legal-square priors."""
        if self.piece_session is None or patch.shape[:2] != (64, 64):
            return 'x', 0.0
        chars = ['a', 'b', 'c', 'k', 'n', 'p', 'r', 'A', 'B', 'C', 'K', 'N', 'P', 'R']
        probabilities = []
        for turns in range(4):
            rgb = cv2.cvtColor(np.rot90(patch, turns).copy(), cv2.COLOR_BGR2RGB).astype(np.float32)
            tensor = ((rgb - [123.675, 116.28, 103.53]) /
                      [58.395, 57.12, 57.375]).astype(np.float32).transpose(2, 0, 1)[None]
            logits = self.piece_session.run(None, {self.piece_input_name: tensor})[0][0]
            if logits.shape != (14,) or not np.isfinite(logits).all():
                return 'x', 0.0
            probs = np.exp(logits - logits.max())
            probabilities.append(probs / probs.sum())
        probabilities = np.asarray(probabilities)
        average = probabilities.mean(axis=0)
        winner = int(average.argmax())
        supporting = probabilities.argmax(axis=1) == winner
        supporting &= probabilities[:, winner] >= .55
        margin = float(average[winner] - np.partition(average, -2)[-2])
        confidence = float(np.median(probabilities[:, winner]))
        if supporting.sum() < 3 or confidence < .65 or margin < .20:
            return 'x', 0.0
        return chars[winner], confidence

    def refine_with_physics_and_rules(self, warped_bgr: np.ndarray,
                                      grid_chars: List[List[str]],
                                      conf_grid: List[List[float]]) -> Tuple[List[List[str]], List[List[float]], str]:
        """Recognize physical pieces without rewriting them to obey game rules.

        Reachable-square checks belong to game validation. A real elephant on
        the other side of the river must remain an elephant in the observation.
        Primary predictions are preserved unless independent evidence resolves
        a class-count conflict. Weak predictions may use local ink evidence;
        stronger conflicting predictions require stronger auxiliary evidence.
        Original occlusion x and empty points are never replaced here.
        """
        refined = [row.copy() for row in grid_chars]
        confidence = [row.copy() for row in conf_grid]
        self.last_refinement_edits = []
        padded = cv2.copyMakeBorder(warped_bgr, 32, 32, 32, 32, cv2.BORDER_REFLECT)
        # First assess side independently. Type counts only trigger extra
        # measurement; they never select the replacement class.
        for r in range(10):
            for c in range(9):
                original = grid_chars[r][c]
                if original in ('.', 'x') or conf_grid[r][c] >= .90:
                    continue
                ch = original
                cx, cy = round(50+c*350/8)+32, round(50+r*400/9)+32
                side = self.ink_color_evidence(padded[cy-18:cy+18, cx-18:cx+18])
                if side is not None:
                    ch = ch.upper() if side else ch.lower()
                refined[r][c] = ch
        limits = dict(zip("KABNRCP", (1, 2, 2, 2, 2, 2, 5)))
        limits.update({ch.lower(): count for ch, count in list(limits.items())})
        counts = collections.Counter(refined[r][c] for r in range(10) for c in range(9)
                                     if confidence[r][c] >= .50)
        # Resolve weakest conflicting cells first and update counts immediately.
        # Once a conflict is resolved, stronger pieces are not reclassified.
        cells = sorted(((r, c) for r in range(10) for c in range(9)),
                       key=lambda cell: confidence[cell[0]][cell[1]])
        for r, c in cells:
            original = grid_chars[r][c]
            ch = refined[r][c]
            if ch not in limits:
                continue
            cx, cy = round(50+c*350/8)+32, round(50+r*400/9)+32
            best, probability = 'x', 0.0
            if counts[ch] > limits[ch]:
                best, probability = self.query_rotated_piece(padded[cy-32:cy+32, cx-32:cx+32])
            required = .85 if conf_grid[r][c] >= .90 else .65
            if (best != 'x' and probability >= required and best.isupper() == ch.isupper()
                    and best.lower() != ch.lower()):
                if confidence[r][c] >= .50:
                    counts[ch] -= 1
                ch = best
                confidence[r][c] = min(float(conf_grid[r][c]), probability)
                if confidence[r][c] >= .50:
                    counts[ch] += 1
            refined[r][c] = ch
            if ch != original:
                self.last_refinement_edits.append({
                    "position": f"{chr(97+c)}{r}", "before": original,
                    "after": ch, "auxiliary_confidence": probability,
                })
        return refined, confidence, "\n".join("".join(row) for row in refined)

    def format_fen(self, grid_chars: List[List[str]], active_color: str = 'w') -> str:
        """
        将 10x9 盘面转换为标准中国象棋 FEN 记谱格式
        例如: rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1
        """
        from localization.vision_contract import grid_to_fen
        if active_color not in ('w','b'):raise ValueError('FEN side must be w or b')
        return grid_to_fen(grid_chars,active_color=='w')

    @staticmethod
    def _create_annular_kernel(radius: float = 16.5, thickness: float = 2.0, size: int = 41) -> np.ndarray:
        """创建环形滤波器核（用于亚像素级快速圆形凹槽与边缘互相关）"""
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        ring = np.exp(-0.5 * ((dist - radius) / (thickness / 2.0))**2)
        ring[dist < radius - thickness * 2] = 0
        ring[dist > radius + thickness * 2] = 0
        norm_ring = ring / (np.sum(ring) + 1e-6)
        return norm_ring

    def detect_piece_offsets(self, warped_bgr, grid_chars, board_width_mm=None,
                             board_height_mm=None, *, calibration=None, frame_id=-1,
                             captured_at=None, positions=None):
        """Quality-gated pixel centers; physical readiness is explicit per entry."""
        from dataclasses import replace
        from localization.offset_measurement import CenterMeasurer, VisionCalibration
        from localization.vision_contract import validate_grid
        grid=validate_grid(grid_chars,allow_unknown=True)
        uncertain=any(ch=='x' for row in grid for ch in row)
        measurable=[[('.' if ch=='x' else ch) for ch in row] for row in grid]
        cal=calibration or VisionCalibration.from_config()
        if board_width_mm is not None:cal=replace(cal,width_mm=board_width_mm,columns_mm=())
        if board_height_mm is not None:cal=replace(cal,height_mm=board_height_mm,rows_mm=())
        result=CenterMeasurer(cal).measure(warped_bgr,measurable,frame_id=frame_id,
                                         captured_at=captured_at,positions=positions)
        if uncertain:
            for info in result.values():
                info.update(valid=False,control_valid=False,reason='Uncertain board classification')
        return result

    def visualize_board(self, warped_bgr: np.ndarray, grid_chars: List[List[str]],
                        conf_grid: List[List[float]],
                        piece_offsets: Optional[Dict[str, Dict]] = None) -> np.ndarray:
        """
        在拉平后的棋盘图像上绘制各个格子的识别结果、置信度以及毫米级偏置瞄准十字线
        """
        vis = warped_bgr.copy()
        padding = 50.0
        cell_w = (450.0 - padding * 2.0) / 8.0
        cell_h = (500.0 - padding * 2.0) / 9.0
        col_chars = "abcdefghi"

        for r in range(10):
            for c in range(9):
                char = grid_chars[r][c]
                conf = conf_grid[r][c]
                if char in ['.', 'x']:
                    continue

                cx = int(round(padding + c * cell_w))
                cy = int(round(padding + r * cell_h))
                pos_str = f"{col_chars[c]}{r}"

                # 基础字符与阵营标示圈
                is_red = char.isupper()
                color = (0, 0, 220) if is_red else (0, 180, 0)
                text = f"{char}"

                if piece_offsets and pos_str in piece_offsets and piece_offsets[pos_str].get("valid", False):
                    info = piece_offsets[pos_str]
                    u_det, v_det = info["detected_px"]
                    r_px = int(round(info["radius_px"]))
                    u_i, v_i = int(round(u_det)), int(round(v_det))

                    # 1. 理论交叉中心：蓝色细十字准星
                    cv2.drawMarker(vis, (cx, cy), (255, 120, 0), cv2.MARKER_CROSS, 10, 1)

                    # 2. 实测物理圆心：高亮黄色圆环与黄色实心中心点
                    cv2.circle(vis, (u_i, v_i), r_px, (0, 240, 255), 1, cv2.LINE_AA)
                    cv2.circle(vis, (u_i, v_i), 2, (0, 240, 255), -1)

                    # 3. 偏置连线 (红色矢量，指示机械臂微调方向)
                    if info["offset_dist_mm"] >= 0.5:
                        cv2.line(vis, (cx, cy), (u_i, v_i), (0, 0, 255), 1, cv2.LINE_AA)

                    # 4. 棋子文字标记在实测圆心位置
                    cv2.putText(vis, text, (u_i - 5, v_i + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

                    # 5. 标注物理偏移毫米值 (例如 "+1.2,-0.5")
                    rob_dx, rob_dy = info["robot_offset_mm"]
                    offset_text = f"{rob_dx:+.1f},{rob_dy:+.1f}"
                    cv2.putText(vis, offset_text, (u_i - 16, v_i + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(vis, (cx, cy), 18, color, 2)
                    cv2.putText(vis, text, (cx - 6, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        return vis

    def classify_board_smoothed(self, warped_bgr: np.ndarray, window_size: int = 3) -> Tuple[List[List[str]], List[List[float]], str]:
        """
        使用多帧滑动窗口平滑的分类器接口 (仅平滑概率，不提供物理帧新鲜度保证)
        """
        # 先执行单帧推理（会自动写入 recent_logits_buffer）
        raw_grid, _, _ = self.classify_board(warped_bgr)

        # 获取最近 N 帧平均 logits
        recent = self.recent_logits_buffer[-window_size:]
        avg_logits = np.mean(recent, axis=0)

        best_idx = np.argmax(avg_logits, axis=-1)
        conf = avg_logits[np.arange(90), best_idx]

        short_chars = [self.label_2_short[self.classes_labels[idx]] for idx in best_idx]
        grid_chars = [short_chars[i * 9:(i + 1) * 9] for i in range(10)]
        conf_grid = [conf[i * 9:(i + 1) * 9].tolist() for i in range(10)]

        for r in range(10):
            for c in range(9):
                if raw_grid[r][c]=='x':grid_chars[r][c]='x';conf_grid[r][c]=0.

        if self.enable_physics_refine:
            grid_chars, conf_grid, layout_str = self.refine_with_physics_and_rules(warped_bgr, grid_chars, conf_grid)
        else:
            layout_str = "\n".join(["".join(row) for row in grid_chars])

        return grid_chars, conf_grid, layout_str

    # =========================================================================
    # 端到端总入口：识别盘面 + 可视化生成
    # =========================================================================
    def recognize_full(self, image_bgr: np.ndarray, use_smoothing: bool = False) -> Dict:
        """
        端到端全流程测试接口
        use_smoothing: 是否开启多帧时序平滑 (默认 False，视频流连续处理时建议开启)
        """
        t0 = time.time()
        corners, corner_scores = self.predict_corners(image_bgr)
        t_corner = time.time() - t0
        from localization.live_vision import geometry_reason, grid_reason
        geometry_rejection=geometry_reason(corners,corner_scores,image_bgr.shape)
        if geometry_rejection:
            return dict(corners=np.asarray(corners).tolist(),corner_scores=np.asarray(corner_scores).tolist(),
                grid_chars=[['x']*9 for _ in range(10)],conf_grid=[[0.]*9 for _ in range(10)],
                layout_str='\n'.join(['x'*9]*10),fen='',piece_offsets={},warped_image=None,
                visualized_board=image_bgr.copy(),debug_visualization=image_bgr.copy(),
                rejection=geometry_rejection,stable=False,control_ready=False,
                timing=dict(corner_ms=t_corner*1000,classify_ms=0.,offset_ms=0.,total_ms=(time.time()-t0)*1000))

        t1 = time.time()
        warped, M = self.warp_chessboard(image_bgr, corners)
        if use_smoothing and len(self.recent_logits_buffer) >= 2:
            grid_chars, conf_grid, layout_str = self.classify_board_smoothed(warped, window_size=3)
        else:
            grid_chars, conf_grid, layout_str = self.classify_board(warped)
        t_classify = time.time() - t1

        t_offset_start = time.time()
        piece_offsets = self.detect_piece_offsets(warped, grid_chars)
        t_offset = time.time() - t_offset_start

        rejection=grid_reason(grid_chars,conf_grid,None)
        fen = self.format_fen(grid_chars) if not rejection else ""
        vis_board = self.visualize_board(warped, grid_chars, conf_grid, piece_offsets=piece_offsets)
        total_time = time.time() - t0

        # 生成可视化调试图 (原图视角叠加四角点、网格与实测圆心偏置准星)
        debug_vis = image_bgr.copy()
        for idx, (pt, name) in enumerate(zip(corners, BONE_NAMES)):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(debug_vis, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(debug_vis, f"{name}:{corner_scores[idx]:.2f}", (x + 8, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # 绘制四边框
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug_vis, [pts[[0, 1, 3, 2]]], True, (0, 255, 0), 2)

        # 将校正图上的亚像素圆心与理论网格逆透视变换回原图相机坐标系
        try:
            M_inv = np.linalg.inv(M)
            for pos_str, info in piece_offsets.items():
                if not info.get("valid"):continue
                u_d, v_d = info["detected_px"]
                cx_n, cy_n = info["nominal_px"]

                pts_warp = np.array([[[u_d, v_d]], [[cx_n, cy_n]]], dtype=np.float32)
                pts_orig = cv2.perspectiveTransform(pts_warp, M_inv)

                cam_u, cam_v = int(round(pts_orig[0][0][0])), int(round(pts_orig[0][0][1]))
                cam_cx, cam_cy = int(round(pts_orig[1][0][0])), int(round(pts_orig[1][0][1]))

                # 原图画出微调瞄准线与圆心
                cv2.circle(debug_vis, (cam_u, cam_v), 3, (0, 240, 255), -1)
                if info["offset_dist_mm"] >= 1.0:
                    cv2.line(debug_vis, (cam_cx, cam_cy), (cam_u, cam_v), (0, 0, 255), 1, cv2.LINE_AA)
        except Exception:
            pass

        return {
            "rejection": rejection,
            "stable": False,
            "control_ready": False,
            "corners": corners.tolist(),
            "corner_scores": corner_scores.tolist(),
            "grid_chars": grid_chars,
            "conf_grid": conf_grid,
            "layout_str": layout_str,
            "fen": fen,
            "piece_offsets": piece_offsets,
            "warped_image": warped,
            "visualized_board": vis_board,
            "debug_visualization": debug_vis,
            "timing": {
                "corner_ms": round(t_corner * 1000, 1),
                "classify_ms": round(t_classify * 1000, 1),
                "offset_ms": round(t_offset * 1000, 1),
                "total_ms": round(total_time * 1000, 1)
            }
        }
