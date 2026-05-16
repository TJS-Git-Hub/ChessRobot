# 中国象棋对弈机器人 | Chinese Chess Robot

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLO-Ultralytics-00a8ff.svg)](https://ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一套完整的**人机对弈象棋机器人系统**。使用摄像头实时感知人类走棋，CNN 深度神经网络决策 AI 走法，四自由度机械臂自动执子。

A complete **human-vs-robot Chinese Chess system**: real-time camera perception, CNN-based AI decision-making, and a 4-DOF robotic arm for autonomous piece movement.

---

## 系统架构 | System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  视觉感知层   │ ──▶ │  逻辑决策层   │ ──▶ │  硬件执行层  │
│  (Vision)    │     │  (Strategy)  │     │  (Actuation) │
├─────────────┤     ├──────────────┤     ├─────────────┤
│ YOLO 棋子检测 │     │ 象棋规则引擎   │     │ 逆运动学求解  │
│ ROI 透视矫正  │     │ CNN 走法预测   │     │ 线性插补规划  │
│ 多帧稳态滤波  │     │ 合法走法掩码   │     │ PWM 舵机控制  │
│ 移动检测追踪  │     │ 棋盘状态张量   │     │ 电磁铁拾取    │
└─────────────┘     └──────────────┘     └─────────────┘
       ▲                                        │
       │              move_queue                │
       └────────────────────────────────────────┘
```

### 数据流

1. **摄像头** 1280x720 实时采集 → **YOLO** 棋子检测 + ID 追踪
2. 多帧稳态判定 → 检测人类走棋 → 推入线程安全队列
3. 主线程消费走棋 → **规则引擎**校验 → 更新棋盘状态
4. **CNN** 推理最佳应招 → **逆运动学**求解关节角 → 机械臂执行

---

## 技术栈 | Tech Stack

| 层级 | 技术 | 说明 |
|------|------|------|
| 视觉定位 | YOLO (Ultralytics) + OpenCV | 棋子检测、ROI 标定、EMA 平滑、多帧稳态 |
| 游戏逻辑 | 纯 Python 规则引擎 | 9 种棋子逐一校验，跑合法走法 |
| AI 决策 | PyTorch CNN (自研 ChessNet) | 14 通道 × 10×9 棋盘 → 8100 维走法概率 |
| 运动控制 | 逆运动学 (余弦定理) | 4 自由度连杆，前馈补偿 + 死区滤波 |
| 硬件驱动 | CH347 I2C + PCA9685 | PWM 舵机 (50Hz) + GPIO 电磁铁 |
| GUI | PyQt5 | 实时视频、虚拟棋盘、日志控制台 |

---

## 目录结构 | Project Structure

```
PythonProject2/
├── main.py                  # 中枢调度器 (多线程编排)
├── config.py                # 硬件参数 (连杆长度、舵机标定、棋盘尺寸)
├── kinematics.py            # 逆运动学求解器
├── control.py               # 机械臂控制器 (微距保护、平滑插补)
├── hardware.py              # PCA9685 舵机驱动 + CH347 I2C 通信
├── ui_main.py               # PyQt5 图形界面
├── requirements.txt         # Python 依赖
│
├── game/                    # 游戏逻辑
│   ├── board.py             # 象棋规则引擎 (完整走法校验)
│   ├── ai.py                # CNN 网络定义 + 推理接口
│   └── constants.py         # 棋盘初始状态
│
├── localization/            # 视觉定位
│   ├── find_chess_new.py    # YOLO 棋子检测 + 追踪 (V15.4)
│   ├── find_chess_simple.py # 简化版视觉方案
│   └── find_chess_simple_plus.py
│
├── recognition/             # 模型训练
│   ├── train.py             # YOLO 微调训练脚本
│   ├── models/              # 训练好的权重 (best.pt)
│   └── data/                # 训练数据集 + data.yaml
│
└── tools/                   # 调试与标定工具
    ├── calibrate_servo.py   # 舵机物理零点标定
    ├── test_electromagnet.py# 电磁铁功能测试
    └── calc_reach.py        # 机械臂可达空间计算
```

---

## 快速开始 | Quick Start

### 环境要求

- Python 3.10+
- CUDA 兼容 GPU (可选，CPU 也能跑)
- Windows (硬件驱动依赖 CH347 DLL)
- 4-DOF 舵机机械臂 + PCA9685 舵机驱动板
- USB 摄像头 (1280x720)

### 外部资源

以下文件因体积原因不随仓库分发，需单独下载：

| 文件 | 大小 | 说明 | 下载后放入 |
|------|------|------|-----------|
| `aaa.pth` | 79MB | CNN 走法预测权重 (不装则 AI 降级为随机走法) | `game/` |
| `CH347DLLA64.DLL` | ~1MB | 沁恒 CH347 USB-I2C 驱动 | 项目根目录 |

- **模型下载**：[百度网盘](https://pan.baidu.com/s/1cOmNqE26eN-mdrOmyEXlAQ?pwd=kmg2) (提取码: kmg2)
- **DLL 下载**：[沁恒官网](http://www.wch.cn/downloads/CH347DLL_ZIP.html)

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/<your-username>/ChineseChessRobot.git
cd ChineseChessRobot

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载上述外部资源放入对应位置

# 4. (可选) 舵机标定
python tools/calibrate_servo.py
```

### 运行

```bash
# 命令行模式
python main.py

# GUI 模式 (推荐)
python ui_main.py
```

---

## 核心设计 | Design Highlights

### 1. 象棋规则引擎 (`game/board.py`)

完整实现了中国象棋全部 9 种棋子的走法校验：
- 车/炮：直线行走 + 炮架判定
- 马：蹩脚检测
- 象/相：塞眼检测 + 不越河
- 士/仕：九宫约束
- 将/帅：九宫单步 + 飞将检测
- 兵/卒：未过河/已过河方向切换

支持的功能：合法走法遍历、局面状态张量生成、胜负判定。

### 2. 视觉稳态判定 (`localization/find_chess_new.py`)

- **多帧共识投票**：采用 35 帧滑动窗口，只有连续多帧一致的盘面才被接受为稳态
- **EMA 平滑**：指数移动平均滤波消除摄像头抖动
- **运动能量检测**：通过棋子位移量计算"运动能量"，自动判定人类是否正在走棋
- **ROI 透视矫正**：四点标定自动计算透视变换矩阵

### 3. CNN 走法预测 (`game/ai.py`)

```
输入: [1, 14, 10, 9]  棋盘状态张量
  │
  ▼  Conv2d(14→64, k=3) + ReLU
  ▼  Conv2d(64→128, k=3) + ReLU
  ▼  Conv2d(128→128, k=3) + ReLU
  ▼  Flatten → FC(128*90 → 1024) + ReLU
  ▼  FC(1024 → 8100)  Logits
  │
  ▼  Legal-Move Mask (只保留合法走法)
  ▼  argmax → 4 位数字走法 "x1y1x2y2"
```

### 4. 机械臂微距保护 (`control.py`)

在常规 pick-and-place 动作中引入**微距垂直阶段**：
- 下降前先悬停在棋子上方 8mm
- 再用 0.5 秒极慢速垂直切入
- 吸取后立即强制抬升至微距高度
- 彻底杜绝机械臂刮蹭相邻棋子

---

## License

MIT License. 详见 [LICENSE](LICENSE) 文件。

---

## 致谢 | Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — 棋子检测模型
- [PyTorch](https://pytorch.org/) — 深度学习框架
- CH347 USB-I2C 桥接芯片驱动
