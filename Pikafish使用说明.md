# Pikafish 象棋决策

项目沿用现有棋盘、视觉和机械臂接口，由 `game/ai.py` 调用 `game/pikafish/` 中的引擎与权重，默认每步思考 3000 毫秒。`ChessAI(board, movetime_ms=...)` 可调整思考时间。原 CNN 决策代码保存在 `game/cnn_ai.py`，权重 `game/aaa.pth` 保留但默认不加载。

本机原有 conda 环境名为 `Chess_Robot`。在项目根目录运行 `python -m game.play` 可进行不连接机械臂的命令行对局；输入四位坐标（例如红兵 `0605`），输入 q 退出。运行 `python -m unittest discover -s tests -p test_pikafish.py -v` 验证适配器及实际引擎。

每次调用使用当前局面 FEN，不包含历史走法，因此不完整支持长将、长捉及复杂重复局面裁定。旧棋盘对人类走法的规则校验也不完整；本次对 AI 返回的走法额外检查了不自陷将军。真实摄像头、棋盘方向和机械臂动作仍须现场验证。

引擎使用本机 `zgxq` 目录提供的 Windows 可执行文件与 NNUE 权重，权重许可随文件保留。Pikafish 是第三方 GPL 项目，分发引擎时应核对其许可和源码义务。
