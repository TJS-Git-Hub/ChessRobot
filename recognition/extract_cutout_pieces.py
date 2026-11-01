"""
extract_cutout_pieces.py
双源棋子切片自动提取引擎：
  来源 1: 用户采集的 14 类单兵特写照片 (single_pieces) -> 中心圆检测并提取 RGBA 透明切片
  来源 2: 100 张真实对局照片 (indoor_normal) -> 基于 10x9 盘面投影自动抠出各网格上的棋子切片
将双源融合保存至: recognition/data/piece_cutouts/<class_name>/
"""

import os
import sys
import cv2
import numpy as np

project_root = r"c:\workspace\ChessRobot"
sys.path.insert(0, project_root)
from recognition.onnx_pipeline import ChessRecognizer

save_cutouts_dir = os.path.join(project_root, "recognition", "data", "piece_cutouts")
single_pieces_dir = os.path.join(project_root, "recognition", "data", "single_pieces")
indoor_normal_dir = os.path.join(project_root, "recognition", "data", "raw_dataset", "indoor_normal")

# 14 类映射
CLASSES = [
    "red_king", "red_advisor", "red_bishop", "red_knight", "red_rook", "red_cannon", "red_pawn",
    "black_king", "black_advisor", "black_bishop", "black_knight", "black_rook", "black_cannon", "black_pawn"
]

def extract_from_single_pieces():
    """从用户采集的单兵特写提取"""
    print("🔹 [1/2] 正在从单兵特写照片库 (single_pieces) 提取高质量切片...")
    extracted_counts = {c: 0 for c in CLASSES}

    for c in CLASSES:
        c_dir = os.path.join(single_pieces_dir, c)
        if not os.path.exists(c_dir):
            continue
        out_dir = os.path.join(save_cutouts_dir, c)
        os.makedirs(out_dir, exist_ok=True)

        for f in os.listdir(c_dir):
            if not f.lower().endswith(('.jpg', '.png')):
                continue
            img_path = os.path.join(c_dir, f)
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            # 棋子通常位于画面中央附近 (cx, cy)
            cx, cy = w // 2, h // 2
            roi_size = 180
            x1 = max(0, cx - roi_size)
            y1 = max(0, cy - roi_size)
            x2 = min(w, cx + roi_size)
            y2 = min(h, cy + roi_size)
            roi = img[y1:y2, x1:x2]

            # 转换为灰度检测棋子外圆
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                param1=100, param2=35, minRadius=40, maxRadius=120
            )

            if circles is not None:
                circles = np.uint16(np.around(circles))
                best_c = circles[0][0]
                rcx, rcy, r = int(best_c[0]), int(best_c[1]), int(best_c[2])
            else:
                rcx, rcy, r = roi_size, roi_size, 75

            # 抠出圆形并生成 RGBA 透明通道
            pad = int(r * 1.15)
            rx1 = max(0, rcx - pad)
            ry1 = max(0, rcy - pad)
            rx2 = min(roi.shape[1], rcx + pad)
            ry2 = min(roi.shape[0], rcy + pad)
            chip = roi[ry1:ry2, rx1:rx2]

            # 创建透明 Alpha 遮罩
            mask = np.zeros((chip.shape[0], chip.shape[1]), dtype=np.uint8)
            cv2.circle(mask, (chip.shape[1] // 2, chip.shape[0] // 2), r, 255, -1)
            # 边缘羽化
            mask = cv2.GaussianBlur(mask, (5, 5), 1.5)

            rgba = cv2.cvtColor(chip, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = mask

            # 统一切片输出尺寸为 90x90
            rgba = cv2.resize(rgba, (90, 90), interpolation=cv2.INTER_AREA)

            out_name = f"single_{f.rsplit('.', 1)[0]}.png"
            cv2.imwrite(os.path.join(out_dir, out_name), rgba)
            extracted_counts[c] += 1

    return extracted_counts

def extract_from_indoor_normal(recognizer: ChessRecognizer, max_boards=25):
    """从 100 张实战对局照片中通过网格定位自动提取切片"""
    print(f"🔹 [2/2] 正在从真实对战棋盘照片中自动分割实战切片 (采样 {max_boards} 张)...")
    if not os.path.exists(indoor_normal_dir):
        return {}

    files = sorted([f for f in os.listdir(indoor_normal_dir) if f.lower().endswith(('.jpg', '.png'))])[:max_boards]
    auto_counts = {c: 0 for c in CLASSES}

    short_to_class = {
        'K': 'red_king', 'A': 'red_advisor', 'B': 'red_bishop', 'N': 'red_knight',
        'R': 'red_rook', 'C': 'red_cannon', 'P': 'red_pawn',
        'k': 'black_king', 'a': 'black_advisor', 'b': 'black_bishop', 'n': 'black_knight',
        'r': 'black_rook', 'c': 'black_cannon', 'p': 'black_pawn'
    }

    for f_idx, f in enumerate(files):
        img_path = os.path.join(indoor_normal_dir, f)
        img = cv2.imread(img_path)
        if img is None:
            continue

        res = recognizer.recognize_full(img)
        warped = res['warped_image']  # 450x500 的标准透视拉平图像 (w=450, h=500)
        grid_chars = res['grid_chars']
        conf_grid = res['conf_grid']

        # 450x500 棋盘内 90 个交叉点的中心坐标 (网格间距 w_step=43.75, h_step=44.44, padding=50)
        pad = 50
        w_step = (450 - 2 * pad) / 8.0   # 350 / 8 = 43.75
        h_step = (500 - 2 * pad) / 9.0   # 400 / 9 = 44.44

        for r in range(10):
            for col in range(9):
                char = grid_chars[r][col]
                conf = conf_grid[r][col]
                if char in short_to_class and conf > 0.85:
                    c_name = short_to_class[char]
                    if auto_counts[c_name] >= 8:  # 每类从实战中至多提取 8 个高质量样本
                        continue

                    # 交叉点中心
                    gx = int(pad + col * w_step)
                    gy = int(pad + r * h_step)
                    half_sz = 22
                    x1 = max(0, gx - half_sz)
                    y1 = max(0, gy - half_sz)
                    x2 = min(warped.shape[1], gx + half_sz)
                    y2 = min(warped.shape[0], gy + half_sz)

                    chip = warped[y1:y2, x1:x2]
                    if chip.shape[0] < 20 or chip.shape[1] < 20:
                        continue

                    # 建立圆形遮罩
                    mask = np.zeros((chip.shape[0], chip.shape[1]), dtype=np.uint8)
                    cv2.circle(mask, (chip.shape[1] // 2, chip.shape[0] // 2), int(min(chip.shape[:2]) * 0.46), 255, -1)
                    rgba = cv2.cvtColor(chip, cv2.COLOR_BGR2BGRA)
                    rgba[:, :, 3] = mask
                    rgba = cv2.resize(rgba, (90, 90), interpolation=cv2.INTER_AREA)

                    out_dir = os.path.join(save_cutouts_dir, c_name)
                    os.makedirs(out_dir, exist_ok=True)
                    out_name = f"auto_b{f_idx:02d}_{r}_{col}.png"
                    cv2.imwrite(os.path.join(out_dir, out_name), rgba)
                    auto_counts[c_name] += 1

    return auto_counts

def main():
    print("=" * 60)
    print("🚀 启动双源棋子切片提取 (手工采集 + 算法自动实战分割)...")
    print("=" * 60)
    rec = ChessRecognizer()

    s_counts = extract_from_single_pieces()
    a_counts = extract_from_indoor_normal(rec, max_boards=30)

    print("\n" + "=" * 60)
    print("📊 双源切片提取汇总结果：")
    print("=" * 60)
    print(f"{'类别名称':20s} | {'单兵特写切片':12s} | {'实战自动分割':12s} | {'合计':6s}")
    print("-" * 60)
    total = 0
    for c in CLASSES:
        sc = s_counts.get(c, 0)
        ac = a_counts.get(c, 0)
        tot = sc + ac
        total += tot
        print(f"{c:20s} | {sc:12d} | {ac:12d} | {tot:6d}")
    print("-" * 60)
    print(f"🎉 全部 14 类棋子切片提取完毕，共获得 {total} 份高质量透明素材！")
    print(f"📁 保存目录: {save_cutouts_dir}")

if __name__ == "__main__":
    main()

