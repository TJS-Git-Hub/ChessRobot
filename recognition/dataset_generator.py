"""
dataset_generator.py
动态棋盘数据合成引擎与 PyTorch 数据集：
- 基于 40 张预校正的真实空白棋盘底图 (450, 400)
- 基于 192 张真实提取的 14 类透明棋子切片 (RGBA)
- 动态合成 4 种对局盘面：标准开局 (30%)、动态中局 (40%)、残局 (20%)、随机棋局 (10%)
- 棋子 0~360° 旋转、微缩放、毫米级交点偏心抖动
- 全套高鲁棒性光照增强 (室外强光反光、室内暗光噪点、色温漂移、阴影遮挡)
- 输出 256x256 标准输入张量与 90 格类别标签 (0~15)
"""

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from heavy_augmentations import (
    apply_outdoor_glare,
    apply_indoor_dark_and_noise,
    apply_color_temperature_shift,
    apply_cast_shadows,
    rotate_and_jitter_piece,
    apply_full_heavy_augmentation
)

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

CLASSES_LIST = list(DICT_CATE_NAMES.keys())
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES_LIST)}
CHAR_TO_IDX = {DICT_CATE_NAMES[c]: i for i, c in enumerate(CLASSES_LIST)}
CHAR_TO_NAME = {
    'k': 'black_king', 'a': 'black_advisor', 'b': 'black_bishop', 'n': 'black_knight',
    'r': 'black_rook', 'c': 'black_cannon', 'p': 'black_pawn',
    'K': 'red_king', 'A': 'red_advisor', 'B': 'red_bishop', 'N': 'red_knight',
    'R': 'red_rook', 'C': 'red_cannon', 'P': 'red_pawn'
}

STANDARD_OPENING = [
    ['r', 'n', 'b', 'a', 'k', 'a', 'b', 'n', 'r'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', 'c', '.', '.', '.', '.', '.', 'c', '.'],
    ['p', '.', 'p', '.', 'p', '.', 'p', '.', 'p'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['P', '.', 'P', '.', 'P', '.', 'P', '.', 'P'],
    ['.', 'C', '.', '.', '.', '.', '.', 'C', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['R', 'N', 'B', 'A', 'K', 'A', 'B', 'N', 'R'],
]

class SyntheticChessDataset(Dataset):
    def __init__(self,
                 empty_boards_dir: str,
                 cutouts_dir: str,
                 samples_per_epoch: int = 1600,
                 target_size: tuple = (256, 256),
                 is_training: bool = True):
        super().__init__()
        self.samples_per_epoch = samples_per_epoch
        self.target_size = target_size
        self.is_training = is_training

        # 加载所有空白底图到内存
        self.empty_boards = []
        if os.path.exists(empty_boards_dir):
            for f in sorted(os.listdir(empty_boards_dir)):
                if f.lower().endswith(('.jpg', '.png')):
                    img = cv2.imread(os.path.join(empty_boards_dir, f))
                    if img is not None:
                        self.empty_boards.append(img)
        if len(self.empty_boards) == 0:
            raise RuntimeError(f"未找到空白棋盘底图: {empty_boards_dir}")

        # 加载所有棋子透明切片
        self.cutouts = {c: [] for c in CHAR_TO_NAME.values()}
        for cname in self.cutouts.keys():
            cdir = os.path.join(cutouts_dir, cname)
            if os.path.exists(cdir):
                for f in os.listdir(cdir):
                    if f.lower().endswith('.png'):
                        p = cv2.imread(os.path.join(cdir, f), cv2.IMREAD_UNCHANGED)
                        if p is not None and p.ndim == 3 and p.shape[2] == 4:
                            self.cutouts[cname].append(p)
            if len(self.cutouts[cname]) == 0:
                print(f"⚠️ 警告: 类别 {cname} 未找到切片 PNG")

        # 归一化参数 (标准 ImageNet 或 OpenMMLab CChess)
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def __len__(self):
        return self.samples_per_epoch

    def _generate_layout(self) -> list:
        """生成棋盘字符布局 (10x9)"""
        mode = random.random()
        if mode < 0.35:
            # 模式 1: 标准开局
            return [row.copy() for row in STANDARD_OPENING]
        elif mode < 0.75:
            # 模式 2: 动态中局 (从开局随机移动或吃子)
            layout = [row.copy() for row in STANDARD_OPENING]
            # 随机吃掉 4~14 个子
            pieces_pos = [(r, c) for r in range(10) for c in range(9) if layout[r][c] != '.']
            num_remove = random.randint(4, 14)
            to_remove = random.sample(pieces_pos, min(num_remove, len(pieces_pos)))
            for r, c in to_remove:
                layout[r][c] = '.'
            # 随机将 2~6 个子移动到附近的空白格
            remaining_pos = [(r, c) for r in range(10) for c in range(9) if layout[r][c] != '.']
            empty_pos = [(r, c) for r in range(10) for c in range(9) if layout[r][c] == '.']
            num_move = min(random.randint(2, 6), len(remaining_pos), len(empty_pos))
            srcs = random.sample(remaining_pos, num_move)
            dsts = random.sample(empty_pos, num_move)
            for src, dst in zip(srcs, dsts):
                layout[dst[0]][dst[1]] = layout[src[0]][src[1]]
                layout[src[0]][src[1]] = '.'
            return layout
        elif mode < 0.90:
            # 模式 3: 残局 (4~12 个棋子)
            layout = [['.' for _ in range(9)] for _ in range(10)]
            # 必定有双方将帅
            layout[random.randint(0, 2)][random.randint(3, 5)] = 'k'
            layout[random.randint(7, 9)][random.randint(3, 5)] = 'K'
            # 随机添加其他 2~8 个棋子
            avail_chars = ['r', 'n', 'b', 'a', 'c', 'p', 'R', 'N', 'B', 'A', 'C', 'P']
            other_pos = [(r, c) for r in range(10) for c in range(9) if layout[r][c] == '.']
            num_other = random.randint(2, 8)
            chosen_pos = random.sample(other_pos, num_other)
            for r, c in chosen_pos:
                layout[r][c] = random.choice(avail_chars)
            return layout
        else:
            # 模式 4: 随机布局 (测试泛化与抗干扰)
            layout = [['.' for _ in range(9)] for _ in range(10)]
            all_chars = list(CHAR_TO_NAME.keys())
            all_pos = [(r, c) for r in range(10) for c in range(9)]
            num_pieces = random.randint(6, 26)
            chosen_pos = random.sample(all_pos, num_pieces)
            for r, c in chosen_pos:
                layout[r][c] = random.choice(all_chars)
            return layout

    def __getitem__(self, idx: int):
        # 1. 随机选一张底图
        board = random.choice(self.empty_boards).copy()
        H, W = board.shape[:2] # (450, 400)

        # 2. 生成布局
        layout = self._generate_layout()

        # 3. 标签数组 (90,)
        labels = np.zeros(90, dtype=np.int64)

        # 4. 贴子
        for r in range(10):
            for c in range(9):
                char = layout[r][c]
                flat_idx = r * 9 + c
                labels[flat_idx] = CHAR_TO_IDX[char]

                if char == '.':
                    continue

                cname = CHAR_TO_NAME[char]
                cutout_list = self.cutouts.get(cname, [])
                if not cutout_list:
                    continue

                cutout = random.choice(cutout_list).copy()

                if self.is_training:
                    # 棋子 0~360° 旋转与微缩放
                    cutout = rotate_and_jitter_piece(cutout)

                    # 棋子大小抖动 (38 ~ 46 像素)
                    psize = random.randint(38, 46)
                    # 交点微偏心抖动 (+/- 4 像素)
                    jitter_x = random.randint(-4, 4)
                    jitter_y = random.randint(-4, 4)
                else:
                    psize = 42
                    jitter_x = 0
                    jitter_y = 0

                cutout = cv2.resize(cutout, (psize, psize), interpolation=cv2.INTER_LINEAR)

                # 计算交点像素中心
                xc = int(round(25.0 + c * (350.0 / 8.0))) + jitter_x
                yc = int(round(25.0 + r * (400.0 / 9.0))) + jitter_y

                x0 = xc - psize // 2
                y0 = yc - psize // 2
                x1 = x0 + psize
                y1 = y0 + psize

                # 边界保护
                bx0 = max(0, x0)
                by0 = max(0, y0)
                bx1 = min(W, x1)
                by1 = min(H, y1)

                cx0 = bx0 - x0
                cy0 = by0 - y0
                cx1 = cx0 + (bx1 - bx0)
                cy1 = cy0 + (by1 - by0)

                if bx1 > bx0 and by1 > by0 and cx1 > cx0 and cy1 > cy0:
                    chip = cutout[cy0:cy1, cx0:cx1]
                    alpha = chip[:, :, 3:4].astype(np.float32) / 255.0
                    bg = board[by0:by1, bx0:bx1].astype(np.float32)
                    fg = chip[:, :, :3].astype(np.float32)
                    blended = bg * (1.0 - alpha) + fg * alpha
                    board[by0:by1, bx0:bx1] = np.clip(blended, 0, 255).astype(np.uint8)

        # 5. 全套高强度光照与阴影数据增强
        if self.is_training:
            board = apply_full_heavy_augmentation(board)

        # 6. 缩放到模型输入尺寸 (256, 256)
        resized = cv2.resize(board, self.target_size, interpolation=cv2.INTER_LINEAR)

        # 7. 归一化并转为 PyTorch Tensor (3, 256, 256)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        norm = (rgb - self.mean) / self.std
        tensor = torch.from_numpy(np.transpose(norm, (2, 0, 1))).float()
        labels_tensor = torch.from_numpy(labels).long()

        return tensor, labels_tensor


class RealValDataset(Dataset):
    """用于在验证集上评估真实棋盘表现的 Dataset"""
    def __init__(self, val_dir: str, target_size: tuple = (256, 256)):
        super().__init__()
        self.target_size = target_size
        self.image_paths = []
        if os.path.exists(val_dir):
            for f in sorted(os.listdir(val_dir)):
                if f.lower().endswith(('.jpg', '.png')):
                    self.image_paths.append(os.path.join(val_dir, f))

        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        # 标准开局 90 格标签
        labels = []
        for r in range(10):
            for c in range(9):
                char = STANDARD_OPENING[r][c]
                labels.append(CHAR_TO_IDX[char])
        self.gt_labels = np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        if img.shape[:2] != self.target_size:
            img = cv2.resize(img, self.target_size)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        norm = (rgb - self.mean) / self.std
        tensor = torch.from_numpy(np.transpose(norm, (2, 0, 1))).float()
        labels_tensor = torch.from_numpy(self.gt_labels).long()

        return tensor, labels_tensor

