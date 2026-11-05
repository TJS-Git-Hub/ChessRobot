"""
test_offline_onnx.py: 测试本地 ONNX 两阶段象棋盘面识别流水线
用法：
  python tools/test_offline_onnx.py --image path/to/chessboard.jpg
  python tools/test_offline_onnx.py --dummy   # 使用随机测试图验证模型前向推理
  python tools/test_offline_onnx.py --cam     # 抓取单帧摄像头实时测试
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# 加入项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


from recognition.onnx_pipeline import ChessRecognizer

def print_ascii_board(grid_chars):
    print("\n" + "=" * 35)
    print("      识别到的 10×9 棋盘盘面")
    print("=" * 35)
    col_header = "    " + " ".join([f"{i}" for i in range(9)])
    print(col_header)
    print("   +" + "-" * 19 + "+")
    for r_idx, row in enumerate(grid_chars):
        row_str = " ".join(row)
        print(f" {r_idx:2d}| {row_str} |")
    print("   +" + "-" * 19 + "+")
    print("图例: 红方大写(K A B N R C P), 黑方小写(k a b n r c p), '.'空位, 'x'遮挡")
    print("=" * 35 + "\n")

def main():
    parser = argparse.ArgumentParser(description="ChessRobot 离线 ONNX 两阶段识别测试")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--image", type=str, default="", help="待测试的棋盘图片路径")
    modes.add_argument("--dummy", action="store_true", help="仅验证合成图前向计算，不计算准确率")
    modes.add_argument("--cam", action="store_true", help="调用摄像头拍照测试")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path.home()/".codex/cache/tasks/chessrobot-image"/datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if args.image and not Path(args.image).is_file():
        parser.error(f"图片不存在: {args.image}")
    if args.output_dir.resolve().is_relative_to(Path(project_root)):
        parser.error("输出目录应在项目外")

    print("🚀 [1/3] 正在加载 ChessRecognizer ONNX 模型...")
    recognizer = ChessRecognizer()

    if recognizer.pose_session is None or recognizer.classifier_session is None:
        print("❌ 模型加载失败，请检查 recognition/models/onnx 目录下是否包含 pose 与 layout 模型。")
        sys.exit(1)

    img = None
    source_name = ""

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise ValueError(f"图片无法解码: {args.image}")
        source_name = os.path.basename(args.image)
        print(f"📷 [2/3] 读取指定测试图像: {args.image} ({img.shape[1]}x{img.shape[0]})")
    elif args.cam:
        print("📷 [2/3] 正在调用摄像头拍摄单帧...")
        cap = cv2.VideoCapture(args.camera_index)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"摄像头{args.camera_index}无法打开；停止，不回退样本")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ret, frame = cap.read()
        finally:
            cap.release()
        if ret and frame is not None:
            img = frame
            source_name = "camera_snapshot"
            print(f"✅ 摄像头拍摄成功 ({img.shape[1]}x{img.shape[0]})")
        else:
            raise RuntimeError("摄像头没有返回画面；停止，不回退样本或假图")

    if img is None and not args.dummy:
        # 尝试查找已采集的样本图片
        sample_dir = os.path.join(project_root, "recognition", "data", "raw_dataset", "indoor_normal")
        if os.path.exists(sample_dir):
            files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.png'))]
            if files:
                sample_path = os.path.join(sample_dir, files[0])
                img = cv2.imread(sample_path)
                source_name = files[0]
                print(f"📷 [2/3] 自动发现已采集样本: {sample_path}")

    if img is None and not args.dummy:
        raise ValueError("未提供有效图片且没有可用样本；请指定 --image，或显式使用 --dummy 仅验证计算")

    if img is None:
        print("ℹ️ 生成明确标记的虚拟图片，仅验证前向计算，不评估识别准确率...")
        img = np.full((720, 1280, 3), 200, dtype=np.uint8)
        # 画一个简单的深色方框模拟棋盘
        cv2.rectangle(img, (300, 100), (980, 620), (50, 50, 50), 3)
        source_name = "synthetic_dummy"

    print("🧠 [3/3] 正在执行端到端两阶段推理 (RTMPose 角点定位 + SwinV2 盘面分类)...")
    res = recognizer.recognize_full(img)

    print("\n" + "-" * 50)
    print("🎯 推理性能耗时报告:")
    print(f"  - 姿态角点提取用时: {res['timing']['corner_ms']} ms")
    print(f"  - 棋盘拉伸与分类用时: {res['timing']['classify_ms']} ms")
    print(f"  - 总端到端用时:     {res['timing']['total_ms']} ms ({1000.0/max(res['timing']['total_ms'], 1.0):.1f} FPS)")
    print("-" * 50)

    print("📌 棋盘 4 角点坐标预测 (A0, A8, J0, J8):")
    for name, pt, score in zip(["A0", "A8", "J0", "J8"], res['corners'], res['corner_scores']):
        print(f"  - {name}: ({pt[0]:.1f}, {pt[1]:.1f}) [置信度: {score:.3f}]")

    print_ascii_board(res['grid_chars'])

    # 保存可视化调试图像
    out_dir = str(args.output_dir.resolve())
    os.makedirs(out_dir, exist_ok=True)
    out_vis_path = os.path.join(out_dir, f"test_vis_{source_name}.jpg")
    cv2.imwrite(out_vis_path, res['debug_visualization'])
    print(f"💾 可视化角点与检测结果已保存至: {out_vis_path}")
    print("✅ 本次前向推理完成。正确率需要人工真值核对。")

if __name__ == "__main__":
    main()
