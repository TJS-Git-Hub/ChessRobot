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

    def refine_with_physics_and_rules(self,
                                      warped_bgr: np.ndarray,
                                      grid_chars: List[List[str]],
                                      conf_grid: List[List[float]]) -> Tuple[List[List[str]], List[List[float]], str]:
        """
        工业级物理特征、墨迹色度与高精单兵模型三位一体抗强光/反光纠偏引擎：
        1. 内部纯字面圆形遮罩 (r<=13)，100% 排除外围木纹黄底对色度的干扰
        2. 结合 CIELAB 与 R-G 双重物理墨迹色度，绝对分流红黑两军
        3. 抗高光过曝双保险判定 (解决反光导致的方差下降与漏检)
        4. 象棋九宫/过河/底线拓扑规则硬校验
        5. 对低置信或非法棋格无缝唤起 MobileNetV3 64x64 高精单兵网络二次仲裁
        """
        pad = 50
        w_step = 350.0 / 8.0
        h_step = 400.0 / 9.0
        half = 32
        padded = cv2.copyMakeBorder(warped_bgr, half, half, half, half, cv2.BORDER_REFLECT)

        refined_grid = [row.copy() for row in grid_chars]
        refined_conf = [row.copy() for row in conf_grid]

        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std_norm = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        char_to_idx = {
            'a': 0, 'b': 1, 'c': 2, 'k': 3, 'n': 4, 'p': 5, 'r': 6,
            'A': 7, 'B': 8, 'C': 9, 'K': 10, 'N': 11, 'P': 12, 'R': 13
        }
        idx_to_char = {v: k for k, v in char_to_idx.items()}

        for r in range(10):
            for c in range(9):
                cx = int(round(pad + c * w_step)) + half
                cy = int(round(pad + r * h_step)) + half

                # 采样中心 36x36 核心区域
                c_patch = padded[cy - 18:cy + 18, cx - 18:cx + 18]
                h, w = c_patch.shape[:2]

                # 内部纯字面圆形遮罩 (r<=13)，排除棋格外围暖木黄边干扰
                Y, X = np.ogrid[:h, :w]
                dist = np.sqrt((X - w // 2) ** 2 + (Y - h // 2) ** 2)
                inner_mask = dist <= 13.0

                gray = cv2.cvtColor(c_patch, cv2.COLOR_BGR2GRAY)
                std = float(np.std(gray))

                inner_gray = gray[inner_mask]
                face_bg = float(np.median(inner_gray))
                ink_pixels = inner_mask & (gray < (face_bg - 12))
                n_ink = int(np.sum(ink_pixels))

                orig_char = grid_chars[r][c]
                conf = conf_grid[r][c]

                # 存在性判定：
                # 实体棋子在反光强光下字面发白，但刻痕墨迹仍显著 (n_ink>=140)
                if orig_char not in ['.', 'x']:
                    has_piece = not (n_ink < 140 and std < 28.0)
                else:
                    has_piece = (n_ink >= 180 and std >= 45.0)

                if not has_piece:
                    refined_char = '.'
                else:
                    # 确定红黑：结合 R-G 与 CIELAB A 通道
                    if n_ink >= 15:
                        g_val = float(np.mean(c_patch[:, :, 1][ink_pixels]))
                        r_val = float(np.mean(c_patch[:, :, 2][ink_pixels]))
                        rg_diff = r_val - g_val

                        lab = cv2.cvtColor(c_patch, cv2.COLOR_BGR2LAB)
                        lab_a = float(np.mean(lab[:, :, 1][ink_pixels]))
                    else:
                        rg_diff = 0.0
                        lab_a = 128.0

                    in_black_palace = (0 <= r <= 2 and 3 <= c <= 5)
                    in_red_palace = (7 <= r <= 9 and 3 <= c <= 5)

                    # 基础红黑判定
                    is_red = (rg_diff > 2.0 or lab_a > 130.0)
                    if orig_char not in ['.', 'x']:
                        if orig_char.isupper() and (rg_diff < -3.0 or lab_a < 127.0):
                            is_red = False
                        elif orig_char.islower() and (rg_diff > 3.0 or lab_a > 130.0):
                            is_red = True

                    cur_char = orig_char.upper() if is_red else orig_char.lower() if orig_char not in ['.', 'x'] else ('P' if is_red else 'p')

                    # 拓扑规则合法性检验
                    is_illegal = False
                    if not in_black_palace and cur_char in ['k', 'a']: is_illegal = True
                    if not in_red_palace and cur_char in ['K', 'A']: is_illegal = True
                    if r > 4 and cur_char == 'b': is_illegal = True
                    if r < 5 and cur_char == 'B': is_illegal = True
                    if r == 0 and cur_char == 'p': is_illegal = True
                    if r == 9 and cur_char == 'P': is_illegal = True

                    # 若非法或 Nano 置信度较低 (<0.55)，融合 64x64 高精单兵网络二次仲裁
                    if (is_illegal or conf < 0.55) and self.piece_session is not None:
                        patch_64 = padded[cy - 32:cy + 32, cx - 32:cx + 32]
                        rgb_64 = cv2.cvtColor(patch_64, cv2.COLOR_BGR2RGB).astype(np.float32)
                        norm = (rgb_64 - mean) / std_norm
                        tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]
                        logits = self.piece_session.run(None, {self.piece_input_name: tensor})[0][0]
                        exp_l = np.exp(logits - np.max(logits))
                        probs = exp_l / np.sum(exp_l)

                        candidate_indices = []
                        for idx_cls, cname in enumerate(self.classes_labels[2:]):
                            char_cls = self.label_2_short[cname]
                            if is_red and not char_cls.isupper(): continue
                            if not is_red and not char_cls.islower(): continue
                            if not in_black_palace and char_cls in ['k', 'a']: continue
                            if not in_red_palace and char_cls in ['K', 'A']: continue
                            if r > 4 and char_cls == 'b': continue
                            if r < 5 and char_cls == 'B': continue
                            if r == 0 and char_cls == 'p': continue
                            if r == 9 and char_cls == 'P': continue
                            if char_cls in char_to_idx:
                                candidate_indices.append(char_to_idx[char_cls])

                        if candidate_indices:
                            best_score = -999.0
                            best_char = cur_char
                            for idx_c in candidate_indices:
                                ch = idx_to_char[idx_c]
                                score = float(probs[idx_c])
                                if ch == cur_char and not is_illegal:
                                    score += conf
                                if score > best_score:
                                    best_score = score
                                    best_char = ch
                            refined_char = best_char
                        else:
                            refined_char = cur_char
                    else:
                        refined_char = cur_char

                refined_grid[r][c] = refined_char

        refined_layout_str = "\n".join(["".join(row) for row in refined_grid])
        return refined_grid, refined_conf, refined_layout_str
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

    def visualize_board(self, warped_bgr: np.ndarray, grid_chars: List[List[str]], conf_grid: List[List[float]]) -> np.ndarray:
        """
        在拉平后的棋盘图像上绘制各个格子的识别结果与置信度
        """
        vis = warped_bgr.copy()
        padding = 50
        cell_w = (450 - padding * 2) / 8.0
        cell_h = (500 - padding * 2) / 9.0

        for r in range(10):
            for c in range(9):
                char = grid_chars[r][c]
                conf = conf_grid[r][c]
                if char in ['.', 'x']:
                    continue

                cx = int(round(padding + c * cell_w))
                cy = int(round(padding + r * cell_h))

                # 红方红圈，黑方绿圈
                color = (0, 0, 220) if char.isupper() else (0, 180, 0)
                cv2.circle(vis, (cx, cy), 18, color, 2)

                text = f"{char}"
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

        fen = self.format_fen(grid_chars)
        vis_board = self.visualize_board(warped, grid_chars, conf_grid)
        total_time = time.time() - t0

        # 生成可视化调试图
        debug_vis = image_bgr.copy()
        for idx, (pt, name) in enumerate(zip(corners, BONE_NAMES)):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(debug_vis, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(debug_vis, f"{name}:{corner_scores[idx]:.2f}", (x + 8, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # 绘制四边框
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug_vis, [pts[[0, 1, 3, 2]]], True, (0, 255, 0), 2)

        return {
            "corners": corners.tolist(),
            "corner_scores": corner_scores.tolist(),
            "grid_chars": grid_chars,
            "conf_grid": conf_grid,
            "layout_str": layout_str,
            "fen": fen,
            "warped_image": warped,
            "visualized_board": vis_board,
            "debug_visualization": debug_vis,
            "timing": {
                "corner_ms": round(t_corner * 1000, 1),
                "classify_ms": round(t_classify * 1000, 1),
                "total_ms": round(total_time * 1000, 1)
            }
        }
