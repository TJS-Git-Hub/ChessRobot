# ONNX Models Directory

This directory stores the ONNX inference weights for the ChessRobot computer vision recognition pipeline.

### Model Weights Summary:
1. `layout_swinv2_tiny_finetuned.onnx` (114.76 MB):
   - **Primary Model**: Swin Transformer V2 Tiny + CChess10_9Neck fine-tuned on custom chessboard and physical piece set.
   - **Accuracy**: 99.81% single-cell accuracy, 92.5% single-frame full-board match rate, 100.0% 3-frame voting match rate.
   - **Input**: `(batch_size, 3, 256, 256)` RGB float32 tensor normalized by mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375].
   - **Output**: `(batch_size, 90, 16)` classification logits across the 90 intersections on the 10x9 board.
   - Generated via: `python recognition/train_swinv2.py`

2. `pose_rtmpose_4kpt.onnx` (10.70 MB):
   - **Stage 1 Corner Detection**: RTMPose model detecting the 4 outer corners (`A0`, `A8`, `J0`, `J8`) of the Chinese chessboard for perspective rectification.

3. `layout_swinv2_nano.onnx` (31.10 MB):
   - Original baseline open-source layout recognition model.

*Note: Binary `.onnx` model files exceeding 100MB are excluded from GitHub git tracking via `.gitignore`. They are generated locally by `python recognition/train_swinv2.py` or stored in local model storage.*

