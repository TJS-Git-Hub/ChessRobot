# 本机版本记录

版本：0.1.0-local-pikafish-onnx（2026-11-04）。这是本机设备的独立版本，不替代上游 v2.2.0。

## 与 GitHub main 的关系

本地基线 e11e038；对比的上游 main 为 c147c030cb4db1c5d782137130b545eebfbf044a，上游多出 40 个提交。本版本从本地基线创建独立分支，不将主分支重置或用旧文件覆盖它。

- 当前采用 Pikafish 替换 CNN 决策，保留 ChessAI(board).get_best_move()、ChessBoard 与机械臂调用格式；对 FEN、UCI 坐标及合法性做适配。
- ONNX 识别器与视觉适配器取自上游 c147c03，增加独立照片/摄像头预览。实体对局入口仍使用旧 YOLO 视觉，ONNX 尚未接入硬件闭环。
- 本机 config.py、control.py、hardware.py 和 kinematics.py 保持本机版本；与上游的几何参数、错误处理、急停和毫米补偿实现不同。不能直接套用上游控制标定。
- 上游含额外的训练、仿真测试、可视化工具和多语言文档，本机工作区尚未合入。本分支用于保存本机可运行状态，后续合并必须逐项处理上述差异。

## 使用与外部文件

解释器：conda 环境 Chess_Robot。终端激活后从项目根目录运行：

```text
python -m game.play
python vision_preview.py --camera 1
python -m unittest discover -s tests -p test_pikafish.py -v
```

模型和 Pikafish 可执行文件/NNUE 不放入 Git。所需相对路径、大小和 SHA-256 记录在 runtime-assets.json；克隆本分支后需要单独补齐这些文件。Pikafish 来自本机 zgxq 包，官方源码：https://github.com/official-pikafish/Pikafish 。模型由团队提供，不能仅凭文件名重建同样权重。

备份目录、原始照片、IDE 配置、DLL 和环境文件不纳入本次提交。

## 验证及限制

已完成 Pikafish 适配测试和命令行一步人机对局；三个 ONNX 会话及照片推理均能运行。照片识别曾出现重复黑将，未验证识别准确率、摄像头坐标和机械臂实物动作。复杂重复局面裁定及旧棋盘对人类走法的检查仍有局限。

无引擎文件时测试会跳过实际引擎测试，转换和非法着法拦截测试仍可执行。
