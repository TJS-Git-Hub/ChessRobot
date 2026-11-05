# ONNX model inventory

The live recognizer has one default profile: RTMPose + original Nano +
conservative postprocessing, with MobileNet as a limited auxiliary classifier.

| File | Role | Runtime input / output |
|---|---|---|
| `pose_rtmpose_4kpt.onnx` | Active corner detector | 1x3x256x256 / two 1x4x512 SimCC arrays |
| `layout_swinv2_nano.onnx` | **Active default**, original open-source board model | 1x3x315x280 / 1x90x16 probabilities |
| `piece_classifier_mobilenetv3.onnx` | Locally trained auxiliary model | 1x3x64x64 / 1x14 logits |
| `layout_swinv2_tiny_finetuned.onnx` | Experimental Tiny; **not the default** | 1x3x256x256 / 1x90x16 logits |
| `patch_classifier_mobilenetv3.onnx` | Experimental patch model; not used by default | 1x3x64x64 / 1x15 logits |

RGB input uses mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375].
The runtime distinguishes probabilities from logits. Board-class order is
point, other, red K/A/B/N/R/C/P, black k/a/b/n/r/c/p.
The auxiliary order is black a/b/c/k/n/p/r, then red A/B/C/K/N/P/R.

The original project is [chinese-chess-recognition](https://github.com/TheOne1006/chinese-chess-recognition).
Commit [d7758d4](https://github.com/DongFengPo1412/ChessRobot/commit/d7758d4)
restored Nano as the default after the Tiny fine-tuning experiment.

Historical opening-validation scores do not establish independent midgame or
lighting-generalization accuracy. They must not be presented as universal 100%.
Use `tools/evaluate_vision_lighting.py` with human-reviewed grids to measure
cell accuracy, whole-board exact match, unknowns and rejection coverage separately.

The current `.gitignore` excludes all ONNX files, regardless of size.
A model's presence in this directory does not mean it is active. Runtime paths
and SHA-256 in an evaluation record identify the tested weights. No automatic
switch to Tiny or fabricated classification is permitted when a model is missing.

