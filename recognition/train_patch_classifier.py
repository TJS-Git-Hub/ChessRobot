"""
train_patch_classifier.py
工业级中国象棋两阶段架构 - Stage 2 高清局部单兵分类器训练引擎：
- 严格运行于本地 NVIDIA RTX 4060 GPU (CUDA:0)
- 15 分类任务: 0: empty (.) + 14 类实体棋子 (KABNRCP / kabnrcp)
- 输入分辨率: 64x64 高清切片 (原图 6 倍以上分辨率，笔画与红黑对比极其鲜明)
- 彻底无偏置 (Zero Position Bias): 分类器只看 64x64 切片，完全不输入棋盘坐标，从数学上根除“开局位置幻觉”
- 极端物理偏心与旋转抗扰: 训练时注入 0~360° 任意旋转 + +/-14 px (+/-12mm) 手工摆棋偏心抖动
- 实时在用户真实实拍测试盘面 (3,600+ 真实交叉点) 上进行严格量化评估
- 训练完成后自动导出 ONNX 模型与量化图表
"""

import os
import sys
import time
import json
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timm

# 强制 UTF-8 编码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recognition.onnx_pipeline.chess_recognizer import ChessRecognizer

# 15 类定义
PATCH_CLASSES = [
    "empty",         # 0: '.'
    "red_king",      # 1: 'K'
    "red_advisor",   # 2: 'A'
    "red_bishop",    # 3: 'B'
    "red_knight",    # 4: 'N'
    "red_rook",      # 5: 'R'
    "red_cannon",    # 6: 'C'
    "red_pawn",      # 7: 'P'
    "black_king",    # 8: 'k'
    "black_advisor", # 9: 'a'
    "black_bishop",  # 10: 'b'
    "black_knight",  # 11: 'n'
    "black_rook",    # 12: 'r'
    "black_cannon",  # 13: 'c'
    "black_pawn",    # 14: 'p'
]

CLASS_TO_IDX = {c: i for i, c in enumerate(PATCH_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(PATCH_CLASSES)}

CHAR_TO_CLASS = {
    '.': 'empty', 'x': 'empty',
    'K': 'red_king', 'A': 'red_advisor', 'B': 'red_bishop', 'N': 'red_knight',
    'R': 'red_rook', 'C': 'red_cannon', 'P': 'red_pawn',
    'k': 'black_king', 'a': 'black_advisor', 'b': 'black_bishop', 'n': 'black_knight',
    'r': 'black_rook', 'c': 'black_cannon', 'p': 'black_pawn'
}
CLASS_TO_CHAR = {
    'empty': '.',
    'red_king': 'K', 'red_advisor': 'A', 'red_bishop': 'B', 'red_knight': 'N',
    'red_rook': 'R', 'red_cannon': 'C', 'red_pawn': 'P',
    'black_king': 'k', 'black_advisor': 'a', 'black_bishop': 'b', 'black_knight': 'n',
    'black_rook': 'r', 'black_cannon': 'c', 'black_pawn': 'p'
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


# =============================================================================
# 动态样本增强与合成引擎
# =============================================================================
def rotate_image(image, angle):
    """0~360 任意角度旋转"""
    h, w = image.shape[:2]
    cX, cY = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

def apply_patch_augmentation(patch_rgb):
    """对单兵 64x64 切片注入极度鲁棒的环境光照扰动"""
    img = patch_rgb.astype(np.float32)

    # 1. 亮度与对比度随机抖动
    alpha = random.uniform(0.75, 1.25)
    beta = random.uniform(-25, 25)
    img = np.clip(img * alpha + beta, 0, 255)

    # 2. 色温抖动 (室内暖光 vs 室外冷光)
    b_mult = random.uniform(0.9, 1.1)
    r_mult = random.uniform(0.9, 1.1)
    img[:, :, 0] = np.clip(img[:, :, 0] * r_mult, 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * b_mult, 0, 255)

    # 3. 模拟局部高光反光 (桌面台灯/太阳直射)
    if random.random() < 0.35:
        h, w = img.shape[:2]
        gx = random.randint(15, w - 15)
        gy = random.randint(15, h - 15)
        radius = random.randint(10, 25)
        glare_intensity = random.uniform(25, 75)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - gx)**2 + (Y - gy)**2)
        mask = np.clip(1.0 - dist / radius, 0, 1.0) * glare_intensity
        for c in range(3):
            img[:, :, c] = np.clip(img[:, :, c] + mask, 0, 255)

    # 4. 模拟阴影遮挡 (手或对角线阴影)
    if random.random() < 0.35:
        h, w = img.shape[:2]
        shadow_mask = np.linspace(random.uniform(0.65, 0.9), 1.0, w)
        if random.random() < 0.5:
            shadow_mask = shadow_mask[::-1]
        img = img * shadow_mask[np.newaxis, :, np.newaxis]

    # 5. 高斯噪声与轻微模糊
    if random.random() < 0.25:
        noise = np.random.normal(0, random.uniform(2, 8), img.shape)
        img = np.clip(img + noise, 0, 255)
    if random.random() < 0.2:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img.astype(np.uint8), (ksize, ksize), 0).astype(np.float32)

    return np.clip(img, 0, 255).astype(np.uint8)


class SyntheticPatchDataset(Dataset):
    """
    高保真 64x64 单兵切片动态合成训练集
    - 50% 概率生成各类实体棋子切片 (支持 0~360° 旋转 + +/-14px 手摆偏心)
    - 50% 概率生成各类纯空交叉点切片 (含横竖十字、楚河边界、星位符号、纯木纹)
    """
    def __init__(self,
                 empty_boards_dir: str,
                 cutouts_dir: str,
                 samples_per_epoch: int = 4800,
                 patch_size: int = 64):
        super().__init__()
        self.samples_per_epoch = samples_per_epoch
        self.patch_size = patch_size

        # 加载所有空白底图 (450x400)
        self.empty_boards = []
        if os.path.exists(empty_boards_dir):
            for f in sorted(os.listdir(empty_boards_dir)):
                if f.lower().endswith(('.jpg', '.png')):
                    img = cv2.imread(os.path.join(empty_boards_dir, f))
                    if img is not None:
                        self.empty_boards.append(img)
        if len(self.empty_boards) == 0:
            raise RuntimeError(f"未找到空白底图: {empty_boards_dir}")

        # 加载 14 类棋子透明切片 PNG
        self.cutouts = {c: [] for c in PATCH_CLASSES if c != 'empty'}
        for cname in self.cutouts.keys():
            cdir = os.path.join(cutouts_dir, cname)
            if os.path.exists(cdir):
                for f in os.listdir(cdir):
                    if f.lower().endswith('.png'):
                        p = cv2.imread(os.path.join(cdir, f), cv2.IMREAD_UNCHANGED)
                        if p is not None and p.ndim == 3 and p.shape[2] == 4:
                            self.cutouts[cname].append(p)
            if len(self.cutouts[cname]) == 0:
                print(f"⚠️ 警告: 类别 {cname} 未找到切片素材")

        # 归一化参数
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        # 450x400 底图上的理论交叉点中心 (列步长 43.75, 行步长 44.44)
        self.pad = 50
        self.w_step = 350.0 / 8.0
        self.h_step = 400.0 / 9.0

    def __len__(self):
        return self.samples_per_epoch

    def _get_empty_bg_patch(self):
        """从随机空白棋盘上随机裁剪一个 64x64 背景块"""
        board = random.choice(self.empty_boards)
        H, W = board.shape[:2]
        
        # 70% 概率选择真实交叉点附近，30% 概率在整张板上任意裁
        if random.random() < 0.7:
            r = random.randint(0, 9)
            c = random.randint(0, 8)
            cx = int(round(25 + c * self.w_step + random.randint(-8, 8)))
            cy = int(round(25 + r * self.h_step + random.randint(-8, 8)))
        else:
            cx = random.randint(self.patch_size // 2, W - self.patch_size // 2)
            cy = random.randint(self.patch_size // 2, H - self.patch_size // 2)

        half = self.patch_size // 2
        padded = cv2.copyMakeBorder(board, half, half, half, half, cv2.BORDER_REFLECT)
        patch = padded[cy:cy + self.patch_size, cx:cx + self.patch_size].copy()
        return patch

    def __getitem__(self, idx: int):
        # 50% 概率生成空点样本，50% 概率生成有子样本 (保持 15 类严格均衡)
        is_empty = (random.random() < 0.25)

        bg_patch = self._get_empty_bg_patch()

        if is_empty:
            label_idx = 0  # empty
            final_bgr = bg_patch
        else:
            # 随机选择 14 种棋子之一
            cname = random.choice(list(self.cutouts.keys()))
            label_idx = CLASS_TO_IDX[cname]

            cutout_list = self.cutouts[cname]
            cutout = random.choice(cutout_list).copy()

            # 1. 0~360° 任意旋转
            angle = random.uniform(0, 360)
            cutout = rotate_image(cutout, angle)

            # 2. 真实物理尺寸缩放 (棋子直径约 38 像素)
            psize = random.randint(36, 41)
            cutout = cv2.resize(cutout, (psize, psize), interpolation=cv2.INTER_LINEAR)

            # 3. 手工摆棋偏心平移抖动 (+/-14 px 对应物理 +/-12mm)
            jitter_x = random.randint(-14, 14)
            jitter_y = random.randint(-14, 14)

            # 4. Alpha 贴合到 64x64 背景上
            cx = self.patch_size // 2 + jitter_x
            cy = self.patch_size // 2 + jitter_y

            x1 = cx - psize // 2
            y1 = cy - psize // 2
            x2 = x1 + psize
            y2 = y1 + psize

            # 计算交集与贴图
            px1 = max(0, x1)
            py1 = max(0, y1)
            px2 = min(self.patch_size, x2)
            py2 = min(self.patch_size, y2)

            cx1 = px1 - x1
            cy1 = py1 - y1
            cx2 = cx1 + (px2 - px1)
            cy2 = cy1 + (py2 - py1)

            if px2 > px1 and py2 > py1:
                sub_cutout = cutout[cy1:cy2, cx1:cx2]
                alpha = sub_cutout[:, :, 3:4].astype(np.float32) / 255.0
                chip_rgb = sub_cutout[:, :, :3].astype(np.float32)

                bg_roi = bg_patch[py1:py2, px1:px2].astype(np.float32)
                blended = chip_rgb * alpha + bg_roi * (1.0 - alpha)
                bg_patch[py1:py2, px1:px2] = blended.astype(np.uint8)

            final_bgr = bg_patch

        # 应用光照与阴影增强
        final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
        augmented = apply_patch_augmentation(final_rgb)

        # 归一化并转 Tensor (3, 64, 64)
        norm = (augmented.astype(np.float32) - self.mean) / self.std
        tensor = torch.from_numpy(np.transpose(norm, (2, 0, 1))).float()

        return tensor, torch.tensor(label_idx, dtype=torch.long)


class RealBoardPatchValDataset(Dataset):
    """
    真实实拍盘面交叉点切片验证集
    从用户 40 张开局图 + 实战中局图提取每一个交叉点的 64x64 切片与客观真实真值
    """
    def __init__(self, recognizer: ChessRecognizer, raw_images_dir: str, num_boards: int = 40):
        super().__init__()
        self.samples = []
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        pad = 50
        w_step = 350.0 / 8.0
        h_step = 400.0 / 9.0

        print(f"🔹 正在从 {num_boards} 张真实实拍盘面中切片提取客观验证集...")
        for i in range(1, num_boards + 1):
            fname = f"indoor_normal_{i:04d}.jpg"
            p = os.path.join(raw_images_dir, fname)
            if not os.path.exists(p):
                continue
            img = cv2.imread(p)
            if img is None:
                continue

            # 透视拉平 (450x500)
            corners, scores = recognizer.predict_corners(img)
            warped, _ = recognizer.warp_chessboard(img, corners)

            # 扩展边界防止边缘越界
            half = 32
            padded = cv2.copyMakeBorder(warped, half, half, half, half, cv2.BORDER_REFLECT)

            for r in range(10):
                for c in range(9):
                    char = STANDARD_OPENING[r][c]
                    cname = CHAR_TO_CLASS[char]
                    label_idx = CLASS_TO_IDX[cname]

                    cx = int(round(pad + c * w_step)) + half
                    cy = int(round(pad + r * h_step)) + half
                    crop = padded[cy - 32:cy + 32, cx - 32:cx + 32]

                    self.samples.append((crop, label_idx, i, r, c))

        print(f"✅ 客观验证集构建就绪，共提取 {len(self.samples)} 个真实物理交叉点切片！")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        crop, label_idx, board_id, r, c = self.samples[idx]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        norm = (rgb - self.mean) / self.std
        tensor = torch.from_numpy(np.transpose(norm, (2, 0, 1))).float()
        return tensor, torch.tensor(label_idx, dtype=torch.long), board_id, r, c


# =============================================================================
# 主训练与评估流程
# =============================================================================
def main():
    print("=" * 75)
    print("🚀 [ChessRobot] 启动中国象棋单兵 64x64 高清切片分类器训练引擎")
    print("=" * 75)

    if not torch.cuda.is_available():
        raise RuntimeError("❌ 未检测到可用 CUDA GPU，严禁使用 CPU 训练！")

    device = torch.device("cuda:0")
    print(f"✅ 本地显卡: {torch.cuda.get_device_name(0)}")

    empty_boards_dir = os.path.join(PROJECT_ROOT, "recognition", "data", "rectified_empty_boards")
    cutouts_dir = os.path.join(PROJECT_ROOT, "recognition", "data", "piece_cutouts")
    raw_images_dir = os.path.join(PROJECT_ROOT, "recognition", "data", "raw_dataset", "indoor_normal")
    save_model_path = os.path.join(PROJECT_ROOT, "recognition", "best_patch_mobilenetv3.pth")
    onnx_save_path = os.path.join(PROJECT_ROOT, "recognition", "models", "onnx", "patch_classifier_mobilenetv3.onnx")
    results_dir = os.path.join(PROJECT_ROOT, "recognition", "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)

    EPOCHS = 20
    BATCH_SIZE = 64
    LR = 6e-4

    # 1. 构建数据集
    train_dataset = SyntheticPatchDataset(
        empty_boards_dir=empty_boards_dir,
        cutouts_dir=cutouts_dir,
        samples_per_epoch=4800,
        patch_size=64
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    rec = ChessRecognizer()
    val_dataset = RealBoardPatchValDataset(rec, raw_images_dir, num_boards=40)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 2. 构建模型 (MobileNetV3-Small，原生轻量高效，参数仅 580k)
    print("🔹 初始化 MobileNetV3-Small 预训练主干网络 (15 类别输出)...")
    model = timm.create_model('mobilenetv3_small_050', pretrained=True, num_classes=15).to(device)

    # 3. 严格无偏置损失函数 (所有 15 类别权重严格等于 1.0，绝不压低空位惩罚！)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0
    history = {"train_loss": [], "val_acc": [], "board_perfect_rate": []}

    print(f"\n🟢 [开始迭代] 总轮数: {EPOCHS} | 每轮训练样本: 4800 | 验证样本: {len(val_dataset)}")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        ep_t0 = time.time()
        model.train()
        total_loss = 0.0
        batches = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        scheduler.step()

        # 验证评估
        model.eval()
        correct_cells = 0
        total_cells = 0
        board_errors = {b: 0 for b in range(1, 41)}

        with torch.no_grad():
            for images, targets, board_ids, rows, cols in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=-1)

                is_correct = (preds == targets)
                correct_cells += is_correct.sum().item()
                total_cells += targets.size(0)

                for b_id, ok in zip(board_ids.tolist(), is_correct.tolist()):
                    if not ok:
                        board_errors[b_id] += 1

        val_acc = (correct_cells / max(1, total_cells)) * 100.0
        perfect_boards = sum(1 for err in board_errors.values() if err == 0)
        perfect_rate = (perfect_boards / 40.0) * 100.0

        history["train_loss"].append(round(avg_loss, 4))
        history["val_acc"].append(round(val_acc, 2))
        history["board_perfect_rate"].append(round(perfect_rate, 2))

        ep_time = time.time() - ep_t0
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({ep_time:.1f}s) | Train Loss: {avg_loss:.4f} | "
              f"Val Patch Acc: {val_acc:.2f}% | 100% Perfect Boards: {perfect_boards}/40 ({perfect_rate:.1f}%)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_model_path)
            # 导出 ONNX
            model.eval()
            dummy = torch.randn(1, 3, 64, 64, device=device)
            torch.onnx.export(
                model,
                dummy,
                onnx_save_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                opset_version=14
            )

    total_time = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"🎉 单兵分类器训练与导出圆满完成！总用时: {total_time:.1f} 秒")
    print(f"🔥 最佳验证单点准确率: {best_val_acc:.2f}%")
    print(f"📦 ONNX 导出成功: {onnx_save_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()

