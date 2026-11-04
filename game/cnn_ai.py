import os
import torch
import numpy as np
import random
from game.board import ChessBoard

# 我们需要复用 train.py 中的网络结构定义
# 实际工程中应该提取到一个单独的 network.py 中，这里为保持文件结构简洁直接复制定义
import torch.nn as nn


class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 10 * 9, 1024)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 8100)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x


class ChessAI:
    def __init__(self, board: ChessBoard):
        self.board = board
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNet().to(self.device)
        self.model_loaded = False

        # ==========================================
        # [核心修复]：动态构建绝对路径，消除转义与CWD漂移
        # ==========================================
        # 获取当前 ai.py 文件所在的物理目录绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 将目录与文件名安全拼接
        model_path = os.path.join(current_dir, "aaa.pth")

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()  # 切换至推理模式
            self.model_loaded = True
            print("🟢 成功加载深度学习模型权重。")
        else:
            print("🟡 未找到 aaa.pth，AI将降级为随机走法。请先运行 train.py 训练模型。")

    def index_to_move(self, index: int) -> str:
        """将 0-8099 的网络输出索引还原为 4位走法字符串"""
        src = index // 90
        dst = index % 90
        y1, x1 = divmod(src, 9)
        y2, x2 = divmod(dst, 9)
        return f"{x1}{y1}{x2}{y2}"

    def get_best_move(self) -> str:
        """获取AI最佳走法"""
        # 获取当前黑方所有合法的候选走法
        valid_moves = self.board.get_all_legal_moves(is_red=False)
        if not valid_moves:
            return ""

        # 如果没有模型，直接随机退化处理
        if not self.model_loaded:
            return random.choice(valid_moves)

        # 1. 提取当前棋盘特征
        state_tensor = self.board.get_state_tensor()
        # 增加 batch 维度: [1, 14, 10, 9]
        state_tensor = torch.tensor(state_tensor).unsqueeze(0).to(self.device)

        # 2. 神经网络前向传播，输出 8100 维的原始 Logits
        with torch.no_grad():
            logits = self.model(state_tensor).squeeze(0)  # 尺寸: [8100]

        # 3. 掩码过滤：只看合法的走步
        best_move = None
        max_logit = -float('inf')

        for move in valid_moves:
            # 计算该合法走法对应的张量索引
            x1, y1, x2, y2 = [int(c) for c in move]
            idx = (y1 * 9 + x1) * 90 + (y2 * 9 + x2)

            # 获取神经网络给这个动作打的分
            score = logits[idx].item()

            # 记录最高分的合法动作
            if score > max_logit:
                max_logit = score
                best_move = move

        print(f"🤖 深度学习模型决策: {best_move} (预测置信度得分: {max_logit:.2f})")
        return best_move