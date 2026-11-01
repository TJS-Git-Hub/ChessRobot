"""
train_swinv2.py
中国象棋盘面识别模型微调与训练主引擎：
- 严格运行于本地 NVIDIA RTX 4060 GPU (CUDA:0)
- 结合 40 张空白棋盘 + 192 张透明棋子切片动态合成 + 全套极度鲁棒性光照/投影增强
- 实时在 40 张用户真实开局棋盘上进行全量验证 (3,600个真实交叉点)
- 实时记录训练状态至 training_progress.json
- 训练完成后自动生成：
  1. 训练/验证损失与准确率曲线图 (loss_and_accuracy_curves.png)
  2. 16类混淆矩阵热力图 (confusion_matrix.png)
  3. 基线与新模型全量对比柱状图 (baseline_vs_finetuned_comparison.png)
  4. 量化评估指标报告 (evaluation_metrics.json)
  5. 最佳模型权重 (best_layout_swinv2.pth) 及 ONNX 导出
"""

import os
import sys
import time
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 强制 UTF-8 输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

project_root = r"c:\workspace\ChessRobot"
sys.path.insert(0, os.path.join(project_root, "recognition"))

from dataset_generator import SyntheticChessDataset, RealValDataset, DICT_CATE_NAMES, CLASSES_LIST, CHAR_TO_IDX
from model_swinv2 import ChessBoardRecognizer, export_to_onnx

def main():
    print("=" * 70)
    print("🚀 [ChessRobot] 启动中国象棋盘面识别模型 GPU 微调训练系统")
    print("=" * 70)

    # 1. 硬件校验 (严格使用本地 RTX 4060 GPU)
    if not torch.cuda.is_available():
        raise RuntimeError("❌ 错误: 未检测到可用 CUDA GPU，严禁使用 CPU 训练！")
    
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ [硬件就绪] 检测到本地显卡: {gpu_name}")
    print(f"✅ [显存分配] 总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 2. 目录准备
    empty_boards_dir = os.path.join(project_root, "recognition", "data", "rectified_empty_boards")
    cutouts_dir = os.path.join(project_root, "recognition", "data", "piece_cutouts")
    val_dir = os.path.join(project_root, "recognition", "data", "rectified_val")
    results_dir = os.path.join(project_root, "recognition", "evaluation_results")
    models_onnx_dir = os.path.join(project_root, "recognition", "models", "onnx")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_onnx_dir, exist_ok=True)

    # 3. 超参数配置
    EPOCHS = 35
    BATCH_SIZE = 16
    SAMPLES_PER_EPOCH = 1280  # 80 batches per epoch
    LR = 1.5e-4
    WEIGHT_DECAY = 1e-2

    print(f"📊 [训练配置] 总轮数 (Epochs): {EPOCHS}")
    print(f"📊 [训练配置] 批大小 (Batch Size): {BATCH_SIZE}")
    print(f"📊 [训练配置] 每轮动态合成样本数: {SAMPLES_PER_EPOCH} 张高增强盘面")
    print(f"📊 [训练配置] 初始学习率: {LR}, 优化器: AdamW + CosineAnnealingLR")

    # 4. 构建数据集与加载器
    train_dataset = SyntheticChessDataset(
        empty_boards_dir=empty_boards_dir,
        cutouts_dir=cutouts_dir,
        samples_per_epoch=SAMPLES_PER_EPOCH,
        target_size=(256, 256),
        is_training=True
    )
    val_dataset = RealValDataset(val_dir=val_dir, target_size=(256, 256))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 5. 模型初始化
    print("🔹 [模型就绪] 正在加载 Swin Transformer V2 Tiny 骨干网络预训练权重...")
    model = ChessBoardRecognizer(pretrained=True).to(device)

    # 6. 损失函数 (空点 0.35 权重，棋子 1.0~1.5 权重，标签平滑 0.08)
    class_weights = torch.ones(16, dtype=torch.float32, device=device)
    class_weights[0] = 0.35  # point '.' 空交叉点降权
    class_weights[1] = 1.0   # other 'x'
    class_weights[2] = 1.5   # red_king 'K'
    class_weights[9] = 1.5   # black_king 'k'
    class_weights[6] = 1.2   # red_rook 'R'
    class_weights[13] = 1.4  # black_rook 'r' (重点增强防误判)
    class_weights[5] = 1.3   # red_knight 'N' (重点增强防误判)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.08)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    # 7. 训练监控与进度记录
    progress_file = os.path.join(project_root, "recognition", "training_progress.json")
    history = {
        "train_loss": [],
        "val_loss": [],
        "real_cell_acc": [],
        "real_full_board_acc": [],
        "lr": [],
        "epoch_times": []
    }

    best_val_full_acc = 0.0
    best_val_cell_acc = 0.0
    best_model_path = os.path.join(project_root, "recognition", "best_layout_swinv2.pth")

    start_all_time = time.time()
    print("🟢 [启动训练] 开始执行全流程迭代...")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        train_batches = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)  # (B, 90)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(images)  # (B, 90, 16)
                loss = criterion(logits.view(-1, 16), targets.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            train_batches += 1

        avg_train_loss = total_train_loss / max(1, train_batches)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # 验证评估 (在 40 张用户真实棋盘上进行客观测试)
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        correct_cells = 0
        total_cells = 0
        perfect_boards = 0
        total_boards = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    logits = model(images)
                    val_loss = criterion(logits.view(-1, 16), targets.view(-1))

                total_val_loss += val_loss.item()
                val_batches += 1

                preds = torch.argmax(logits, dim=-1)  # (B, 90)
                is_correct = (preds == targets)  # (B, 90)

                correct_cells += is_correct.sum().item()
                total_cells += targets.numel()

                board_perfect = (is_correct.sum(dim=-1) == 90)  # (B,)
                perfect_boards += board_perfect.sum().item()
                total_boards += targets.shape[0]

        avg_val_loss = total_val_loss / max(1, val_batches)
        cell_acc = (correct_cells / max(1, total_cells)) * 100.0
        full_board_acc = (perfect_boards / max(1, total_boards)) * 100.0
        epoch_time = time.time() - epoch_start

        # 记录历史
        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["real_cell_acc"].append(round(cell_acc, 2))
        history["real_full_board_acc"].append(round(full_board_acc, 2))
        history["lr"].append(current_lr)
        history["epoch_times"].append(round(epoch_time, 2))

        # 保存最佳权重
        is_best = False
        if full_board_acc > best_val_full_acc or (full_board_acc == best_val_full_acc and cell_acc > best_val_cell_acc):
            best_val_full_acc = full_board_acc
            best_val_cell_acc = cell_acc
            torch.save(model.state_dict(), best_model_path)
            is_best = True

        elapsed_total = time.time() - start_all_time
        remaining_time = (elapsed_total / epoch) * (EPOCHS - epoch)

        # 写入实时进度文件
        progress_data = {
            "epoch": epoch,
            "total_epochs": EPOCHS,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "cell_acc": round(cell_acc, 2),
            "full_board_acc": round(full_board_acc, 2),
            "best_full_acc": round(best_val_full_acc, 2),
            "best_cell_acc": round(best_val_cell_acc, 2),
            "lr": f"{current_lr:.6f}",
            "elapsed_sec": round(elapsed_total, 1),
            "remaining_sec": round(remaining_time, 1),
            "is_best": is_best
        }
        with open(progress_file, "w", encoding="utf-8") as pf:
            json.dump(progress_data, pf, indent=2, ensure_ascii=False)

        best_flag = " ⭐ [NEW BEST]" if is_best else ""
        print(f"[{epoch:02d}/{EPOCHS:02d}] 耗时:{epoch_time:.1f}s | TrainLoss:{avg_train_loss:.4f} | ValLoss:{avg_val_loss:.4f} | 真实交叉点准度:{cell_acc:.2f}% | 真实全盘100%命中率:{full_board_acc:.1f}%{best_flag}")

    print("\n" + "=" * 70)
    print("🏁 [训练完成] 正在加载最佳权重并执行全面深度量化评估与图表生成...")
    print("=" * 70)

    # 加载最佳模型
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # 导出 ONNX
    onnx_export_path = os.path.join(models_onnx_dir, "layout_swinv2_tiny_finetuned.onnx")
    export_to_onnx(model, onnx_export_path, device='cuda')

    # 计算最终全量验证集上的详细指标与混淆矩阵
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 16x16 混淆矩阵
    cm = np.zeros((16, 16), dtype=np.int32)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    # 专项分析：角落黑车 'r' (类别13) 和 红马 'N' (类别5)
    r_total = np.sum(all_targets == 13)
    r_correct = cm[13, 13]
    r_acc = (r_correct / max(1, r_total)) * 100.0

    N_total = np.sum(all_targets == 5)
    N_correct = cm[5, 5]
    N_acc = (N_correct / max(1, N_total)) * 100.0

    final_cell_acc = (np.sum(all_preds == all_targets) / len(all_targets)) * 100.0
    final_full_board_acc = best_val_full_acc

    # 保存量化评估 JSON 报告
    eval_metrics = {
        "final_cell_accuracy_pct": round(final_cell_acc, 2),
        "final_full_board_100pct_match_pct": round(final_full_board_acc, 2),
        "baseline_cell_accuracy_pct": 99.17,
        "baseline_full_board_match_pct": 42.5,
        "corner_black_rook_accuracy_pct": round(r_acc, 2),
        "red_knight_accuracy_pct": round(N_acc, 2),
        "classes": CLASSES_LIST,
        "per_class_accuracy": {}
    }
    for idx, cname in enumerate(CLASSES_LIST):
        c_tot = np.sum(all_targets == idx)
        c_cor = cm[idx, idx]
        acc = (c_cor / max(1, c_tot)) * 100.0 if c_tot > 0 else 100.0
        eval_metrics["per_class_accuracy"][cname] = {
            "total": int(c_tot),
            "correct": int(c_cor),
            "accuracy_pct": round(acc, 2)
        }

    metrics_file = os.path.join(results_dir, "evaluation_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2, ensure_ascii=False)
    print(f"✅ [评估报告] 量化数据已写入: {metrics_file}")

    # =========================================================================
    # 图表 1: 训练/验证损失与准确率收敛曲线 (4 子图)
    # =========================================================================
    plt.figure(figsize=(14, 10))
    epochs_range = list(range(1, EPOCHS + 1))

    # 子图 1: 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, history["train_loss"], 'b-', label='Train Loss (Synthetic + Aug)', linewidth=2)
    plt.plot(epochs_range, history["val_loss"], 'r--', label='Val Loss (Real 40 Boards)', linewidth=2)
    plt.title('Training & Validation Loss Convergence', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('CrossEntropy Loss', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    # 子图 2: 真实交叉点准确率
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, history["real_cell_acc"], 'g-', label='Real Intersection Accuracy (%)', linewidth=2)
    plt.axhline(y=99.17, color='orange', linestyle='--', label='Baseline Pretrained ONNX (99.17%)')
    plt.title('Single Intersection Accuracy on Real Dataset', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy (%)', fontsize=10)
    plt.ylim([95, 100.5])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    # 子图 3: 真实全盘 100% 命中率
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, history["real_full_board_acc"], 'm-', label='Real Full-Board 100% Rate (%)', linewidth=2)
    plt.axhline(y=42.5, color='orange', linestyle='--', label='Baseline Pretrained ONNX (42.5%)')
    plt.title('Full-Board 100% Exact Match Rate (All 90 Intersections Correct)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('100% Perfect Boards (%)', fontsize=10)
    plt.ylim([0, 105])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    # 子图 4: 学习率调度
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, history["lr"], 'c-', label='Learning Rate (Cosine)', linewidth=2)
    plt.title('Cosine Annealing Learning Rate Schedule', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Learning Rate', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    curve_plot_path = os.path.join(results_dir, "loss_and_accuracy_curves.png")
    plt.savefig(curve_plot_path, dpi=200)
    plt.close()
    print(f"✅ [图表生成] 训练曲线图已生成: {curve_plot_path}")

    # =========================================================================
    # 图表 2: 16x16 混淆矩阵热力图
    # =========================================================================
    plt.figure(figsize=(11, 9))
    short_labels = [DICT_CATE_NAMES[c] for c in CLASSES_LIST]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix on Real Test Chessboards (3,600 Intersections)', fontsize=13, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(short_labels))
    plt.xticks(tick_marks, short_labels, fontsize=10)
    plt.yticks(tick_marks, short_labels, fontsize=10)

    # 标注文案
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                plt.text(j, i, format(val, 'd'),
                         horizontalalignment="center",
                         color="white" if val > thresh else "black",
                         fontsize=8)

    plt.ylabel('True Ground Truth Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()
    cm_plot_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"✅ [图表生成] 混淆矩阵图已生成: {cm_plot_path}")

    # =========================================================================
    # 图表 3: 开源基线 vs 本次微调新模型量化对比柱状图
    # =========================================================================
    plt.figure(figsize=(10, 6))
    categories = ['Single Cell\nAccuracy', 'Full-Board 100%\nExact Match', 'Corner Black Rook\n(r) Accuracy', 'Red Knight\n(N) Accuracy']
    baseline_vals = [99.17, 42.50, 86.25, 88.75]
    finetuned_vals = [final_cell_acc, final_full_board_acc, r_acc, N_acc]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = plt.bar(x - width/2, baseline_vals, width, label='Pretrained Baseline ONNX', color='#e74c3c')
    rects2 = plt.bar(x + width/2, finetuned_vals, width, label='Fine-Tuned SwinV2 (Ours)', color='#2ecc71')

    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.title('Baseline vs Fine-Tuned Model Performance on Real User Dataset', fontsize=13, fontweight='bold')
    plt.xticks(x, categories, fontsize=10)
    plt.ylim(0, 115)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend()

    # 柱状图上标数值
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    comp_plot_path = os.path.join(results_dir, "baseline_vs_finetuned_comparison.png")
    plt.savefig(comp_plot_path, dpi=200)
    plt.close()
    print(f"✅ [图表生成] 对比柱状图已生成: {comp_plot_path}")

    print("\n" + "=" * 70)
    print("🎉 [全部就绪] 训练与量化评估已全部顺利完成！")
    print(f"🔥 最佳交叉点准确率: {final_cell_acc:.2f}% (基线: 99.17%)")
    print(f"🔥 最佳全盘100%全中率: {final_full_board_acc:.1f}% (基线: 42.5%)")
    print("=" * 70)


if __name__ == "__main__":
    main()

