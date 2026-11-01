"""
model_swinv2.py
纯 PyTorch 实现的 Swin Transformer V2 中国象棋盘面识别模型：
- Backbone: Swin Transformer V2 Tiny (window_size=16, img_size=256, 768维特征)
- Neck: PureCChess10_9Neck (FPN + 多尺度空洞卷积 + 双线性插值映射至 10x9 物理网格)
- Head: 90 格 16 类别直接分类输出 (B, 90, 16)
- 零 mmcv/mmpretrain 依赖，原生支持 ONNX 导出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class PureCChess10_9Neck(nn.Module):
    """
    10x9 棋盘自适应特征金字塔 Neck
    输入: Backbone 最后一层特征图 (B, 768, 8, 8)
    输出: 10x9 网格特征图 (B, 16, 10, 9)
    """
    def __init__(self, in_channels: int = 768, mid_channels: list = [256, 128], num_classes: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_classes = num_classes

        # 初始 1x1 卷积，通道降维
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels[0]),
            nn.ReLU(inplace=True)
        )

        # 常规 3x3 卷积与空洞卷积双分支
        self.main_conv = nn.Sequential(
            nn.Conv2d(mid_channels[0], mid_channels[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels[1]),
            nn.ReLU(inplace=True)
        )
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(mid_channels[0], mid_channels[1], kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_channels[1]),
            nn.ReLU(inplace=True)
        )

        # FPN 自顶向下融合
        self.fpn_conv = nn.Sequential(
            nn.Conv2d(mid_channels[1], mid_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels[0]),
            nn.ReLU(inplace=True)
        )

        # 特征拼接与多尺度融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(sum(mid_channels), mid_channels[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels[-1]),
            nn.ReLU(inplace=True)
        )

        # 最终分类器
        self.final_conv = nn.Conv2d(mid_channels[-1], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 768, 8, 8)
        f0 = self.init_conv(x)  # (B, 256, 8, 8)
        f1 = self.main_conv(f0) + self.dilated_conv(f0)  # (B, 128, 8, 8)

        # FPN 顶层语义指导底层
        top_down = self.fpn_conv(f1)
        f0 = f0 + top_down

        # 双线性插值直接映射到 10x9 物理棋盘网格 (完全兼容 ONNX 导出)
        p0 = F.interpolate(f0, size=(10, 9), mode='bilinear', align_corners=True)  # (B, 256, 10, 9)
        p1 = F.interpolate(f1, size=(10, 9), mode='bilinear', align_corners=True)  # (B, 128, 10, 9)

        # 多尺度融合
        fused = self.fusion_conv(torch.cat([p0, p1], dim=1))  # (B, 128, 10, 9)
        out = self.final_conv(fused)  # (B, 16, 10, 9)
        return out


class ChessBoardRecognizer(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # 加载 timm 官方 SwinTransformer V2 tiny
        self.backbone = timm.create_model(
            'swinv2_tiny_window16_256',
            pretrained=pretrained,
            num_classes=0
        )
        self.neck = PureCChess10_9Neck(in_channels=768, mid_channels=[256, 128], num_classes=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入: (B, 3, 256, 256)
        feats = self.backbone.forward_features(x)  # (B, 8, 8, 768)
        feats = feats.permute(0, 3, 1, 2).contiguous()  # (B, 768, 8, 8)

        out = self.neck(feats)  # (B, 16, 10, 9)
        # 转换为 (B, 90, 16)
        logits = out.permute(0, 2, 3, 1).contiguous().view(-1, 90, 16)
        return logits


def export_to_onnx(model: nn.Module, save_path: str, device: str = 'cuda'):
    """导出模型为 ONNX 格式"""
    model.eval()
    model.to(device)
    dummy_input = torch.randn(1, 3, 256, 256, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},
            'output': {0: 'batch'}
        }
    )
    print(f"✅ [ONNX Export] 模型已成功导出至: {save_path}")
