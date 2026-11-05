"""Local single-image diagnostic UI. No robot, video stability or accuracy guarantee."""
import sys
from pathlib import Path
import threading
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from recognition.onnx_pipeline.chess_recognizer import ChessRecognizer
from localization.live_vision import observe_board
from localization.vision_contract import grid_to_fen
from localization.offset_measurement import CenterMeasurer

_lock=threading.Lock()
_recognizer=None


def process_image(image,model_choice=None,use_smoothing=False):
    global _recognizer
    if image is None:return None,None,'','','Please choose an image.',''
    if use_smoothing:raise ValueError('Independent uploaded images cannot provide temporal stability')
    with _lock:
        if _recognizer is None:_recognizer=ChessRecognizer()
        recognizer=_recognizer
        recognizer.reset_buffer()
        raw=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        result=observe_board(recognizer,raw)
        grid=result['grid_chars'];canonical=None;offsets={};fen=''
        if result['matrix'] is not None:
            canonical=cv2.warpPerspective(raw,np.asarray(result['matrix']),(450,500))
            if not result['rejection']:
                offsets=CenterMeasurer().measure(canonical,grid)
                fen=grid_to_fen(grid)
            preview=recognizer.visualize_board(canonical,grid,result['conf_grid'],offsets)
        else:preview=raw
        lines=[]
        for pos,info in sorted(offsets.items()):
            if not info['valid']:lines.append(f"{pos}: center unavailable ({info['reason']})")
            else:
                dx,dy=info['delta_px'];lines.append(f"{pos}: pixel offset ({dx:+.2f}, {dy:+.2f}); physical calibration unverified")
        status=('Rejected: '+result['rejection']) if result['rejection'] else 'Single-image classification; temporal stability and physical accuracy unverified'
        return cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),cv2.cvtColor(preview,cv2.COLOR_BGR2RGB),fen,'\n'.join(map(''.join,grid)),status,'\n'.join(lines)


def build_demo():
    import gradio as gr
    with gr.Blocks(title='ChessRobot visual diagnostics') as demo:
        gr.Markdown('# 中国象棋视觉诊断\nRTMPose + 原始Nano。单张图不能确认时序稳态，毫米精度需实物标定。')
        image=gr.Image(type='numpy',label='棋盘图片')
        run=gr.Button('识别')
        with gr.Row():
            raw=gr.Image(label='原图')
            board=gr.Image(label='校正图与圆心诊断')
        fen=gr.Textbox(label='FEN（仅可信分类可用）')
        layout=gr.Textbox(label='10×9分类结果',lines=10)
        status=gr.Textbox(label='可信性说明')
        offsets=gr.Textbox(label='像素偏移与拒绝原因',lines=8)
        run.click(process_image,inputs=[image],outputs=[raw,board,fen,layout,status,offsets])
    return demo


if __name__=='__main__':build_demo().launch(server_name='127.0.0.1',server_port=7860,share=False)
