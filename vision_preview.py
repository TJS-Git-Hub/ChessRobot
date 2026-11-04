"""新版 ONNX 视觉预览；不导入机械臂控制或硬件驱动。"""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=1, help='摄像头编号，默认 1')
    parser.add_argument('--image', type=Path, help='仅测试一张照片')
    parser.add_argument('--output', type=Path, help='照片识别结果保存位置')
    args = parser.parse_args()
    model_dir = Path(__file__).resolve().parent / 'recognition' / 'models' / 'onnx'
    names = ['pose_rtmpose_4kpt.onnx', 'layout_swinv2_nano.onnx']
    missing = [str(model_dir / name) for name in names if not (model_dir / name).is_file()]
    if missing:
        sys.exit('缺少新版视觉模型：\n' + '\n'.join(missing))
    try:
        from recognition.onnx_pipeline.chess_recognizer import ChessRecognizer
    except ModuleNotFoundError as exc:
        sys.exit(f'缺少依赖 {exc.name}，请使用配置好的 Chess_Robot 环境。')
    import cv2
    recognizer = ChessRecognizer()
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            sys.exit(f'无法读取照片: {args.image}')
        result = recognizer.recognize_full(frame)
        print('FEN:', result['fen'])
        print('推理耗时:', result['timing'])
        if args.output:
            if not cv2.imwrite(str(args.output), result['visualized_board']):
                sys.exit(f'保存失败: {args.output}')
        return

    from localization.onnx_vision_adapter import ONNXVisionAdapter
    vision = ONNXVisionAdapter(recognizer=recognizer)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap.release()
        sys.exit(f'无法打开摄像头 {args.camera}；可尝试 --camera 0')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print('仅视觉预览。按 Q 或 Esc 退出；坐标方向和棋子身份需要与实物核对。')
    previous = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('摄像头读取失败')
            vision.process_frame(frame)
            if vision.last_stable_grid is not None:
                fen = recognizer.format_fen(vision.last_stable_grid)
                if fen != previous:
                    print('稳态盘面:', fen)
                    previous = fen
            cv2.putText(frame, vision.state.value, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
            cv2.imshow('ONNX Vision Preview', frame)
            key = cv2.waitKey(1) & 255
            if key in (27, ord('q')) or cv2.getWindowProperty('ONNX Vision Preview', cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
