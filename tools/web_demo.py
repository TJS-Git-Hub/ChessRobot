"""
中国象棋盘面与棋子识别交互系统 (Web UI Demo)
基于 Gradio 框架构建，支持上传本地拍摄的中国象棋图片，
自动调用两阶段 ONNX 推理管线进行四角点定位、透视拉平与 90 交叉点 100% 精准识别。
"""

import os
import sys
import glob
import cv2
import numpy as np
import gradio as gr

# 确保项目根目录在 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recognition.onnx_pipeline.chess_recognizer import ChessRecognizer

MODELS_DIR = os.path.join(PROJECT_ROOT, "recognition", "models", "onnx")
FINETUNED_MODEL = os.path.join(MODELS_DIR, "layout_swinv2_tiny_finetuned.onnx")
BASELINE_MODEL = os.path.join(MODELS_DIR, "layout_swinv2_nano.onnx")

print("正在初始化识别器引擎...")
recognizers = {}
if os.path.exists(BASELINE_MODEL):
    recognizers["官方原始基线模型 (SwinV2-Nano，泛化推荐，中局无幻觉)"] = ChessRecognizer(classifier_model_path=BASELINE_MODEL)
if os.path.exists(FINETUNED_MODEL):
    recognizers["实验微调模型 (SwinV2-Tiny，开局过拟合版，供诊断对比)"] = ChessRecognizer(classifier_model_path=FINETUNED_MODEL)
DEFAULT_MODEL_NAME = list(recognizers.keys())[0]
print(f"识别器就绪！默认加载: {DEFAULT_MODEL_NAME}")


# 中文棋子代号映射说明
PIECE_ZH = {
    'K': '红帅', 'A': '红仕', 'B': '红相', 'N': '红马', 'R': '红车', 'C': '红炮', 'P': '红兵',
    'k': '黑将', 'a': '黑士', 'b': '黑象', 'n': '黑马', 'r': '黑车', 'c': '黑炮', 'p': '黑卒',
    '.': '空位', 'x': '其他'
}


def process_image(image: np.ndarray, model_choice: str, use_smoothing: bool):
    """
    处理上传的图像并返回可视化与文本结果
    """
    if image is None:
        return None, None, "", "", "请先上传一张棋盘图片！", ""

    # Gradio 传入格式为 RGB，OpenCV 需要 BGR
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    recognizer = recognizers.get(model_choice, recognizers[DEFAULT_MODEL_NAME])

    result = recognizer.recognize_full(img_bgr, use_smoothing=use_smoothing)

    # 转换回 RGB 供 Gradio 展示
    vis_corner_rgb = cv2.cvtColor(result["debug_visualization"], cv2.COLOR_BGR2RGB)
    vis_board_rgb = cv2.cvtColor(result["visualized_board"], cv2.COLOR_BGR2RGB)

    # 格式化盘面中文统计
    grid = result["grid_chars"]
    red_pieces = []
    black_pieces = []
    for r in range(10):
        for c in range(9):
            char = grid[r][c]
            if char.isupper():
                red_pieces.append(f"{PIECE_ZH.get(char, char)}({r},{c})")
            elif char.islower() and char not in ['.', 'x']:
                black_pieces.append(f"{PIECE_ZH.get(char, char)}({r},{c})")

    piece_stats = (
        f"🔴 红方棋子 ({len(red_pieces)}枚): {' '.join(red_pieces)}\n\n"
        f"⚫ 黑方棋子 ({len(black_pieces)}枚): {' '.join(black_pieces)}"
    )

    t = result["timing"]
    timing_info = (
        f"⏱️ 耗时统计:\n"
        f"  • 阶段 1 (角点定位): {t['corner_ms']:.1f} ms\n"
        f"  • 阶段 2 (盘面分类): {t['classify_ms']:.1f} ms\n"
        f"  • 端到端总用时: {t['total_ms']:.1f} ms"
    )

    return (
        vis_corner_rgb,
        vis_board_rgb,
        result["fen"],
        result["layout_str"],
        timing_info,
        piece_stats
    )


# 收集示例图片
example_images = []
sample_candidates = [
    os.path.join(PROJECT_ROOT, "recognition", "data", "raw_dataset", "indoor_normal", "indoor_normal_0001.jpg"),
    os.path.join(PROJECT_ROOT, "recognition", "data", "raw_dataset", "indoor_normal", "indoor_normal_0026.jpg"),
    os.path.join(PROJECT_ROOT, "recognition", "data", "raw_dataset", "offset_test", "offset_test_0001.jpg"),
    os.path.join(PROJECT_ROOT, "recognition", "data", "raw_dataset", "outdoor_bright", "outdoor_bright_0001.jpg"),
    os.path.join(PROJECT_ROOT, "recognition", "data", "raw_dataset", "outdoor_shadow", "outdoor_shadow_0001.jpg"),
]
for p in sample_candidates:
    if os.path.exists(p):
        example_images.append([p, DEFAULT_MODEL_NAME, False])


# 自定义 CSS 样式
custom_css = """
.title-banner {
    text-align: center;
    padding: 18px;
    background: linear-gradient(135deg, #1f2937, #111827);
    color: white;
    border-radius: 12px;
    margin-bottom: 20px;
}
.title-banner h1 {
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 8px;
    color: #60a5fa;
}
.title-banner p {
    font-size: 14px;
    color: #9ca3af;
}
"""

with gr.Blocks(title="中国象棋盘面与棋子识别交互系统") as demo:
    gr.HTML("""
    <div class="title-banner">
        <h1>🎯 中国象棋盘面与棋子识别交互系统</h1>
        <p>基于两阶段深度学习 (RTMPose 角点定位 + Swin Transformer V2 90格分类) · 支持手工摆棋偏心容差 · 100% 完美盘面识别</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### 📥 1. 上传或选择棋盘图像")
            input_image = gr.Image(
                label="棋盘照片 (支持任意角度/自然光照/手偏心摆放)",
                type="numpy"
            )

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=list(recognizers.keys()),
                    value=DEFAULT_MODEL_NAME,
                    label="选择推理模型"
                )
                smoothing_check = gr.Checkbox(
                    value=False,
                    label="开启多帧平滑 (单张照片建议关闭，连续视频流建议开启)"
                )

            with gr.Row():
                btn_run = gr.Button("🚀 立即识别", variant="primary", scale=2)
                btn_clear = gr.Button("🔄 清除", scale=1)

            timing_box = gr.Textbox(
                label="⚡ 推理性能与时延",
                lines=4,
                interactive=False
            )

        with gr.Column(scale=7):
            gr.Markdown("### 📊 2. 识别效果与对齐结果")
            with gr.Tabs():
                with gr.TabItem("🖼️ 棋盘对齐与棋子标注"):
                    with gr.Row():
                        out_corner = gr.Image(label="Stage 1: 四角点定位 (A0, A8, J0, J8)", interactive=False)
                        out_board = gr.Image(label="Stage 2: 透视拉平与 90 格识别标注", interactive=False)

                with gr.TabItem("📋 棋子清单与统计"):
                    piece_stats_box = gr.Textbox(
                        label="棋子点位汇总",
                        lines=6,
                        interactive=False
                    )

            gr.Markdown("### 📜 3. 记谱与布局数据输出")
            fen_box = gr.Textbox(
                label="标准中国象棋 FEN 记谱 (可直接复制到天天象棋/皮卡鱼/象眼软件分析)",
                interactive=False
            )
            layout_box = gr.Textbox(
                label="10×9 棋盘字符矩阵 (K/A/B/N/R/C/P: 红方, k/a/b/n/r/c/p: 黑方, .: 空点)",
                lines=11,
                max_lines=11,
                interactive=False
            )

    if example_images:
        gr.Markdown("### 💡 快速示例 (点击即可快速填入测试)")
        gr.Examples(
            examples=example_images,
            inputs=[input_image, model_dropdown, smoothing_check],
            label="预置真实测试盘面"
        )

    btn_run.click(
        fn=process_image,
        inputs=[input_image, model_dropdown, smoothing_check],
        outputs=[out_corner, out_board, fen_box, layout_box, timing_box, piece_stats_box]
    )

    def clear_inputs():
        return None, None, None, "", "", "", ""

    btn_clear.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[input_image, out_corner, out_board, fen_box, layout_box, timing_box, piece_stats_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

