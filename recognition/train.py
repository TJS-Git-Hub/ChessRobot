import os
from ultralytics import YOLO

# 1. 路径配置（请确保 data.yaml 中的 train/val 路径正确）
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(_PROJ_ROOT, "recognition", "data", "data.yaml")
# 关键：加载预训练权重 best.pt，在其基础上继续微调
BASE_MODEL = os.path.join(_PROJ_ROOT, "recognition", "models", "best.pt")


def start_v6_evolution():
    if not os.path.exists(DATA_YAML):
        print("X 错误：找不到 data.yaml，请检查路径。")
        return

    # 加载模型
    model = YOLO(BASE_MODEL)

    print("RTX 4060 [V6 进化模式]：1280px 高清、极限增强训练开始...")

    # 开始微调训练
    model.train(
        data=DATA_YAML,
        epochs=200,
        imgsz=1280,
        batch=4,
        device=0,
        lr0=0.0003,
        patience=50,

        # --- 极限增强参数 ---
        degrees=45.0,
        translate=0.15,
        scale=0.6,
        shear=15.0,
        perspective=0.001,
        fliplr=0.5,
        flipud=0.5,

        # --- 核心分类与遮挡增强 ---
        mosaic=1.0,
        mixup=0.2,
        erasing=0.4,
        copy_paste=0.2,

        # --- 颜色与光效（严禁 grayscale） ---
        hsv_h=0.0,
        hsv_s=0.5,
        hsv_v=0.4,

        # --- 优化策略 ---
        cls=2.5,
        label_smoothing=0.1,
        name="Chess_V6_Pro_1280",
        exist_ok=True,
        plots=True
    )


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    start_v6_evolution()
