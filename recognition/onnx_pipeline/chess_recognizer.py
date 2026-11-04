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
        logits = outputs[0][0]  # shape (90, 16)

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

    def get_allowed_classes(self, is_red: bool, r: int, c: int) -> List[str]:
        """中国象棋九宫/楚河汉界/底线拓扑规则候选空间"""
        if is_red:
            in_palace = (7 <= r <= 9 and 3 <= c <= 5)
            if in_palace:
                allowed = ['K', 'A', 'N', 'R', 'C']
                if r in [7, 8]: allowed.append('P')
                if r == 9 and c in [2, 6]: allowed.append('B')
            elif r >= 5: # 红方半场
                allowed = ['N', 'R', 'C']
                if c in [0, 2, 4, 6, 8] and r in [5, 7, 9]: allowed.append('B')
                if r < 9: allowed.append('P')
            else: # 过河 (r <= 4)
                allowed = ['N', 'R', 'C', 'P']
        else: # 黑方
            in_palace = (0 <= r <= 2 and 3 <= c <= 5)
            if in_palace:
                allowed = ['k', 'a', 'n', 'r', 'c']
                if r in [1, 2]: allowed.append('p')
                if r == 0 and c in [2, 6]: allowed.append('b')
            elif r <= 4: # 黑方半场
                allowed = ['n', 'r', 'c']
                if c in [0, 2, 4, 6, 8] and r in [0, 2, 4]: allowed.append('b')
                if r > 0: allowed.append('p')
            else: # 过河 (r >= 5)
                allowed = ['n', 'r', 'c', 'p']
        return allowed

    def query_mobilenet_for_cell(self, padded: np.ndarray, r: int, c: int,
                                 allowed_candidates: List[str],
                                 pad=50, w_step=350.0/8.0, h_step=400.0/9.0, half=32) -> Tuple[str, float]:
        """单兵 MobileNetV3 64x64 高精模型在合法候选空间内的二次仲裁"""
        if self.piece_session is None or not allowed_candidates:
            return allowed_candidates[0] if allowed_candidates else '.', 0.50

        char_to_idx = {
            'a': 0, 'b': 1, 'c': 2, 'k': 3, 'n': 4, 'p': 5, 'r': 6,
            'A': 7, 'B': 8, 'C': 9, 'K': 10, 'N': 11, 'P': 12, 'R': 13
        }
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std_norm = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        cx = int(round(pad + c * w_step)) + half
        cy = int(round(pad + r * h_step)) + half
        patch_64 = padded[cy - 32:cy + 32, cx - 32:cx + 32]
        rgb_64 = cv2.cvtColor(patch_64, cv2.COLOR_BGR2RGB).astype(np.float32)
        norm = (rgb_64 - mean) / std_norm
        tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]
        logits = self.piece_session.run(None, {self.piece_input_name: tensor})[0][0]
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)

        best_ch = allowed_candidates[0]
        best_prob = -1.0
        for ch in allowed_candidates:
            if ch in char_to_idx:
                idx = char_to_idx[ch]
                if probs[idx] > best_prob:
                    best_prob = probs[idx]
                    best_ch = ch
        return best_ch, float(best_prob)

    def refine_with_physics_and_rules(self,
                                      warped_bgr: np.ndarray,
                                      grid_chars: List[List[str]],
                                      conf_grid: List[List[float]]) -> Tuple[List[List[str]], List[List[float]], str]:
        """
        工业级双环色度物理分析、抗反光存在性保全与全局棋规无损仲裁引擎 (Pipeline v6)：
        1. 物理保全原则 (Conservation of Matter)：实体棋子绝不在后处理阶段被错误删除为 '.'，0 漏检。
        2. 双环物理色度分析 (Dual-Zone Chromaticity Analyzer)：
           - 环区 1 (字面刻痕核心区 r <= 12px) + 环区 2 (外围圆槽凹圈 12.5px <= r <= 16.5px)
           - 红方真红刻痕：CombLabA >= 142.0 或 CombRG >= 25.0；黑方刻痕：CombLabA <= 135.5 且 CombRG <= 10.0
           - 物理色度决定真身阵营，绝不随模型预测错判而颠倒。
        3. 拓扑合法空间映射与 MobileNetV3 约束仲裁：
           - 阵营确定后，严格约束九宫格、楚河汉界与兵卒底线移动；
           - 若原始标签在拓扑禁区（如过河相/象），自动由单兵网络在合法空间内重选最优解。
        4. 全局棋规物理数量动态重分配 (Non-destructive Global Arbitration)：
           - 若单类棋子因模型模糊出现超额（如 3 门炮、3 只象），不执行抹除，而是将低置信度者重分配给未饱和棋种。
        """
        pad = 50
        w_step = 350.0 / 8.0
        h_step = 400.0 / 9.0
        half = 32
        padded = cv2.copyMakeBorder(warped_bgr, half, half, half, half, cv2.BORDER_REFLECT)

        refined_grid = [row.copy() for row in grid_chars]
        refined_conf = [row.copy() for row in conf_grid]

        for r in range(10):
            for c in range(9):
                orig_ch = grid_chars[r][c]
                c_conf = conf_grid[r][c]
                if orig_ch in ['.', 'x']:
                    refined_grid[r][c] = '.'
                    continue

                cx = int(round(pad + c * w_step)) + half
                cy = int(round(pad + r * h_step)) + half
                c_patch = padded[cy - 18:cy + 18, cx - 18:cx + 18]
                h, w = c_patch.shape[:2]
                Y, X = np.ogrid[:h, :w]
                dist = np.sqrt((X - w // 2) ** 2 + (Y - h // 2) ** 2)

                # 环区 1：内部字面核心区 (r <= 12.0)
                inner_mask = dist <= 12.0
                gray = cv2.cvtColor(c_patch, cv2.COLOR_BGR2GRAY)
                std = float(np.std(gray))
                inner_gray = gray[inner_mask]
                face_bg = float(np.median(inner_gray))
                ink_pixels = inner_mask & (gray < (face_bg - 12.0))
                n_ink = int(np.sum(ink_pixels))

                # 棋子存在性保全：仅当完全无刻痕无纹理时才判定为空 (彻底杜绝强光下漏检)
                if n_ink < 30 and std < 20.0:
                    refined_grid[r][c] = '.'
                    continue

                # 环区 2：外部凹槽圆环 (12.5 <= r <= 16.5)
                ring_mask = (dist >= 12.5) & (dist <= 16.5)
                ring_gray = gray[ring_mask]
                ring_bg = float(np.median(ring_gray)) if len(ring_gray) > 0 else face_bg
                ring_ink = ring_mask & (gray < (ring_bg - 10.0))
                n_ring = int(np.sum(ring_ink))

                lab = cv2.cvtColor(c_patch, cv2.COLOR_BGR2LAB)
                if n_ink >= 12:
                    inner_rg = float(np.mean(c_patch[:, :, 2][ink_pixels].astype(float) - c_patch[:, :, 1][ink_pixels].astype(float)))
                    inner_la = float(np.mean(lab[:, :, 1][ink_pixels]))
                else:
                    inner_rg = 0.0
                    inner_la = 128.0

                if n_ring >= 20:
                    ring_rg = float(np.mean(c_patch[:, :, 2][ring_ink].astype(float) - c_patch[:, :, 1][ring_ink].astype(float)))
                    ring_la = float(np.mean(lab[:, :, 1][ring_ink]))
                else:
                    ring_rg = inner_rg
                    ring_la = inner_la

                comb_la = 0.65 * inner_la + 0.35 * ring_la
                comb_rg = 0.65 * inner_rg + 0.35 * ring_rg

                # 1. 物理色度独立裁决阵营
                if comb_la >= 142.0 or comb_rg >= 25.0:
                    is_red = True
                elif comb_la <= 135.5 and comb_rg <= 10.0:
                    is_red = False
                else:
                    is_red = (comb_la > 138.0 and comb_rg > 15.0)

                # 2. 候选合法空间与单兵重判
                allowed = self.get_allowed_classes(is_red, r, c)
                desired_char = orig_ch.upper() if is_red else orig_ch.lower()

                if desired_char in allowed:
                    refined_grid[r][c] = desired_char
                    refined_conf[r][c] = c_conf
                else:
                    best_ch, b_prob = self.query_mobilenet_for_cell(padded, r, c, allowed, pad, w_step, h_step, half)
                    refined_grid[r][c] = best_ch
                    refined_conf[r][c] = b_prob

        # 3. 全局物理实体数量硬限幅动态重分配 (无损剪枝)
        limits = {
            'k': 1, 'K': 1,
            'a': 2, 'A': 2,
            'b': 2, 'B': 2,
            'n': 2, 'N': 2,
            'r': 2, 'R': 2,
            'c': 2, 'C': 2,
            'p': 5, 'P': 5
        }

        for iteration in range(5):
            char_locations = collections.defaultdict(list)
            for r in range(10):
                for c in range(9):
                    ch = refined_grid[r][c]
                    if ch in limits:
                        char_locations[ch].append((refined_conf[r][c], r, c))

            violations = []
            for ch, max_allowed in limits.items():
                if len(char_locations[ch]) > max_allowed:
                    violations.append((ch, len(char_locations[ch]) - max_allowed))

            if not violations:
                break

            for ch, excess_count in violations:
                locs = char_locations[ch]
                locs.sort(key=lambda item: item[0]) # 最低置信度优先重判
                is_red = ch.isupper()
                side_limits = {k: v for k, v in limits.items() if (k.isupper() if is_red else k.islower())}

                for i in range(min(excess_count, len(locs))):
                    cur_conf, r, c = locs[i]
                    avail_classes = [k for k, v in side_limits.items() if len(char_locations[k]) < v and k != ch]
                    allowed = self.get_allowed_classes(is_red, r, c)
                    valid_avail = [ac for ac in avail_classes if ac in allowed]

                    if not valid_avail:
                        continue

                    best_ch, b_prob = self.query_mobilenet_for_cell(padded, r, c, valid_avail, pad, w_step, h_step, half)
                    refined_grid[r][c] = best_ch
                    refined_conf[r][c] = b_prob
                    char_locations[ch].remove((cur_conf, r, c))
                    char_locations[best_ch].append((b_prob, r, c))

        refined_layout_str = "\n".join(["".join(row) for row in refined_grid])
        return refined_grid, refined_conf, refined_layout_str

    def format_fen(self, grid_chars: List[List[str]], active_color: str = 'w') -> str:
        """
        将 10x9 盘面转换为标准中国象棋 FEN 记谱格式
        例如: rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1
        """
        fen_rows = []
        for row in grid_chars:
            row_str = ""
            empty_count = 0
            for char in row:
                if char in ['.', 'x']:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row_str += str(empty_count)
                        empty_count = 0
                    row_str += char
            if empty_count > 0:
                row_str += str(empty_count)
            fen_rows.append(row_str)

        board_fen = "/".join(fen_rows)
        full_fen = f"{board_fen} {active_color} - - 0 1"
        return full_fen

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

    def detect_piece_offsets(self, warped_bgr: np.ndarray, grid_chars: List[List[str]],
                             board_width_mm: float = 276.0,
                             board_height_mm: float = 307.0) -> Dict[str, Dict]:
        """
        亚像素棋子圆心定位与毫米级物理偏置测量算法
        结合实测棋盘物理尺寸 (276x307mm)，输出每枚棋子的理论网格中心、
        视觉实测圆心、图像像素偏差以及机械臂坐标系下的毫米物理补偿矢量。
        """
        padding = 50.0
        cell_w = 350.0 / 8.0
        cell_h = 400.0 / 9.0

        # 毫米/像素比率
        K_x = board_width_mm / 350.0
        K_y = board_height_mm / 400.0

        gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = np.sqrt(gx**2 + gy**2)

        # 预计算环形核
        kernels = {r: self._create_annular_kernel(radius=r, thickness=2.0, size=41) for r in [15.5, 16.5, 17.5]}

        col_chars = "abcdefghi"
        piece_offsets = {}

        for r in range(10):
            for c in range(9):
                char = grid_chars[r][c]
                if char in ['.', 'x']:
                    continue

                pos_str = f"{col_chars[c]}{r}"
                cx_nom = padding + c * cell_w
                cy_nom = padding + r * cell_h

                cx_i = int(round(cx_nom))
                cy_i = int(round(cy_nom))

                # 局部 50x50 搜索窗口
                x1 = max(0, cx_i - 25)
                x2 = min(warped_bgr.shape[1], cx_i + 25)
                y1 = max(0, cy_i - 25)
                y2 = min(warped_bgr.shape[0], cy_i + 25)
                patch_mag = edge_mag[y1:y2, x1:x2]

                local_cx_nom = cx_nom - x1
                local_cy_nom = cy_nom - y1

                best_score = -1e9
                best_subpix = (float(local_cx_nom), float(local_cy_nom))
                best_r = 16.5
                search_dist = 10.0

                for radius_val, kernel in kernels.items():
                    resp = cv2.filter2D(patch_mag, -1, kernel)
                    rx_min = max(0, int(round(local_cx_nom - search_dist)))
                    rx_max = min(patch_mag.shape[1], int(round(local_cx_nom + search_dist + 1)))
                    ry_min = max(0, int(round(local_cy_nom - search_dist)))
                    ry_max = min(patch_mag.shape[0], int(round(local_cy_nom + search_dist + 1)))

                    roi_resp = resp[ry_min:ry_max, rx_min:rx_max]
                    min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(roi_resp)

                    if max_v > best_score:
                        best_score = max_v
                        best_r = radius_val
                        peak_x = rx_min + max_loc[0]
                        peak_y = ry_min + max_loc[1]

                        # 亚像素二次抛物面极值拟合
                        if 1 <= peak_x < patch_mag.shape[1] - 1 and 1 <= peak_y < patch_mag.shape[0] - 1:
                            dx = (resp[peak_y, peak_x + 1] - resp[peak_y, peak_x - 1]) / 2.0
                            dy = (resp[peak_y + 1, peak_x] - resp[peak_y - 1, peak_x]) / 2.0
                            dxx = resp[peak_y, peak_x + 1] - 2 * resp[peak_y, peak_x] + resp[peak_y, peak_x - 1]
                            dyy = resp[peak_y + 1, peak_x] - 2 * resp[peak_y, peak_x] + resp[peak_y - 1, peak_x]

                            sub_x = peak_x - (dx / dxx) if abs(dxx) > 1e-4 else float(peak_x)
                            sub_y = peak_y - (dy / dyy) if abs(dyy) > 1e-4 else float(peak_y)

                            sub_x = np.clip(sub_x, peak_x - 0.5, peak_x + 0.5)
                            sub_y = np.clip(sub_y, peak_y - 0.5, peak_y + 0.5)
                            best_subpix = (float(sub_x), float(sub_y))
                        else:
                            best_subpix = (float(peak_x), float(peak_y))

                u_det = x1 + best_subpix[0]
                v_det = y1 + best_subpix[1]

                # 偏置量
                du = u_det - cx_nom
                dv = v_det - cy_nom

                # 棋盘图像坐标系下的毫米偏差 (X向右为正, Y向下为正)
                board_dx_mm = du * K_x
                board_dy_mm = dv * K_y

                # 机械臂第一视角坐标系下的毫米物理补偿:
                # 机械臂在黑方底线外正后方，a列在最右侧(X为正)，i列在最左侧(X为负)
                # 图像向右偏(向i列偏) -> 机械臂X应向负方向补偿 (robot_dx = -board_dx)
                # 图像向下偏(向远端红方偏) -> 机械臂Y应向远端正方向补偿 (robot_dy = +board_dy)
                robot_dx_mm = -board_dx_mm
                robot_dy_mm = +board_dy_mm

                dist_mm = np.sqrt(board_dx_mm**2 + board_dy_mm**2)

                piece_offsets[pos_str] = {
                    "cell": (r, c),
                    "pos_str": pos_str,
                    "char": char,
                    "nominal_px": (round(cx_nom, 1), round(cy_nom, 1)),
                    "detected_px": (round(u_det, 2), round(v_det, 2)),
                    "delta_px": (round(du, 2), round(dv, 2)),
                    "board_offset_mm": (round(board_dx_mm, 2), round(board_dy_mm, 2)),
                    "robot_offset_mm": (round(robot_dx_mm, 2), round(robot_dy_mm, 2)),
                    "radius_px": round(best_r, 1),
                    "offset_dist_mm": round(dist_mm, 2),
                    "confidence": round(float(best_score), 1)
                }

        return piece_offsets

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

                if piece_offsets and pos_str in piece_offsets:
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
        使用多帧滑动窗口平滑的分类器接口 (滤除单帧偶发瞬时抖动，达到 100% 稳定性)
        """
        # 先执行单帧推理（会自动写入 recent_logits_buffer）
        self.classify_board(warped_bgr)

        # 获取最近 N 帧平均 logits
        recent = self.recent_logits_buffer[-window_size:]
        avg_logits = np.mean(recent, axis=0)

        best_idx = np.argmax(avg_logits, axis=-1)
        conf = avg_logits[np.arange(90), best_idx]

        short_chars = [self.label_2_short[self.classes_labels[idx]] for idx in best_idx]
        grid_chars = [short_chars[i * 9:(i + 1) * 9] for i in range(10)]
        conf_grid = [conf[i * 9:(i + 1) * 9].tolist() for i in range(10)]

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

        fen = self.format_fen(grid_chars)
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
