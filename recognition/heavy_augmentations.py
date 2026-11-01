"""
heavy_augmentations.py
专为中国象棋全光照场景打造的极度鲁棒性数据增强算子库：
1. 室外强光与镜面反光模拟 (Outdoor High Glare / Specular Highlights)
2. 室内暗光与高感噪声模拟 (Indoor Low Light / Poisson Sensor Noise)
3. 阴天冷色调 vs 白炽灯暖色调色温漂移 (Color Temperature Shifting)
4. 人手/窗户对角线斜切阴影模拟 (Cast Shadows & Gradient Occlusions)
5. 棋子 360° 连续角度旋转与毫米级交点偏心抖动 (Piece Rotation & Off-Center Jitter)
"""

import numpy as np
import cv2
import random

def apply_outdoor_glare(image: np.ndarray, prob: float = 0.5) -> np.ndarray:
    """室外强光与局部反光增强"""
    if random.random() > prob:
        return image
    img = image.astype(np.float32)
    # 随机提升亮度和对比度
    alpha = random.uniform(1.15, 1.45) # 对比度
    beta = random.uniform(25, 65)       # 亮度
    img = img * alpha + beta

    # 随机模拟局部高光反光斑 (太阳直射或台灯眩光)
    if random.random() < 0.6:
        h, w = image.shape[:2]
        cx, cy = random.randint(50, w - 50), random.randint(50, h - 50)
        radius = random.randint(60, 180)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (71, 71), 25)
        # 高光增益
        gain = random.uniform(40, 100)
        img = img + mask[:, :, np.newaxis] * gain

    return np.clip(img, 0, 255).astype(np.uint8)

def apply_indoor_dark_and_noise(image: np.ndarray, prob: float = 0.5) -> np.ndarray:
    """室内弱光、暗调与摄像头高ISO噪点模拟"""
    if random.random() > prob:
        return image
    img = image.astype(np.float32)

    # 非线性伽马变换降低暗部
    gamma = random.uniform(1.3, 2.2)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    img = cv2.LUT(img.astype(np.uint8), table).astype(np.float32)

    # 叠加暗光传感器高斯/泊松随机噪声
    noise_sigma = random.uniform(6, 18)
    noise = np.random.normal(0, noise_sigma, img.shape)
    img = img + noise

    return np.clip(img, 0, 255).astype(np.uint8)

def apply_color_temperature_shift(image: np.ndarray, prob: float = 0.6) -> np.ndarray:
    """色温漂移：阴天冷蓝光 vs 室内暖黄光模拟"""
    if random.random() > prob:
        return image
    img = image.astype(np.float32)
    # 模式选择：0 为阴天冷光，1 为室内黄灯
    mode = random.choice(["cool", "warm"])
    if mode == "cool":
        # 增强蓝通道，衰减红通道
        b_mult = random.uniform(1.08, 1.25)
        r_mult = random.uniform(0.78, 0.95)
        img[:, :, 0] *= b_mult
        img[:, :, 2] *= r_mult
    else:
        # 增强红黄通道，衰减蓝通道
        r_mult = random.uniform(1.10, 1.30)
        g_mult = random.uniform(1.02, 1.15)
        b_mult = random.uniform(0.70, 0.88)
        img[:, :, 2] *= r_mult
        img[:, :, 1] *= g_mult
        img[:, :, 0] *= b_mult

    return np.clip(img, 0, 255).astype(np.uint8)

def apply_cast_shadows(image: np.ndarray, prob: float = 0.5) -> np.ndarray:
    """真实投影模拟 (人手悬空、窗棱对角线投影)"""
    if random.random() > prob:
        return image
    h, w = image.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)

    # 随机生成 1~2 道倾斜阴影带
    num_shadows = random.randint(1, 2)
    for _ in range(num_shadows):
        poly_pts = [
            [random.randint(-50, w // 2), random.randint(-50, h // 2)],
            [random.randint(w // 2, w + 50), random.randint(-50, h // 2)],
            [random.randint(w // 2, w + 50), random.randint(h // 2, h + 50)],
            [random.randint(-50, w // 2), random.randint(h // 2, h + 50)]
        ]
        shadow_intensity = random.uniform(0.40, 0.75) # 阴影透光率
        sub_mask = np.ones((h, w), dtype=np.float32)
        cv2.fillPoly(sub_mask, [np.array(poly_pts, dtype=np.int32)], shadow_intensity)
        # 边缘羽化
        blur_k = random.choice([35, 55, 75])
        sub_mask = cv2.GaussianBlur(sub_mask, (blur_k, blur_k), 15)
        mask = np.minimum(mask, sub_mask)

    img = image.astype(np.float32) * mask[:, :, np.newaxis]
    return np.clip(img, 0, 255).astype(np.uint8)

def rotate_and_jitter_piece(chip_rgba: np.ndarray) -> np.ndarray:
    """对单个棋子透明切片进行 0~360° 旋转与微仿射变换"""
    angle = random.uniform(0, 360)
    scale = random.uniform(0.88, 1.12)
    h, w = chip_rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    rotated = cv2.warpAffine(chip_rgba, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated

def apply_full_heavy_augmentation(image: np.ndarray) -> np.ndarray:
    """组合应用全套高强度光照与阴影增强流水线"""
    aug = image.copy()
    # 随机组合光线模拟
    scene_dice = random.random()
    if scene_dice < 0.35:
        aug = apply_outdoor_glare(aug, prob=1.0)
    elif scene_dice < 0.65:
        aug = apply_indoor_dark_and_noise(aug, prob=1.0)
    else:
        aug = apply_color_temperature_shift(aug, prob=1.0)

    # 随机叠加投影与手影
    aug = apply_cast_shadows(aug, prob=0.5)

    # 随机轻微虚焦模糊
    if random.random() < 0.25:
        k = random.choice([3, 5])
        aug = cv2.GaussianBlur(aug, (k, k), 0)

    return aug

