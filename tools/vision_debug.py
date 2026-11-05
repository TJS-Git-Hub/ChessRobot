"""One hardware-free visual entry for image, video or camera diagnostics.

Never constructs a robot. Images are single observations, not repeated-frame
stability evidence. Videos and cameras use the same production Gen4 adapter.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
from localization.live_vision import CameraFrame,LatestCamera,HandCheck,LatestHands,observe_board
from localization.onnx_vision_adapter import ONNXVisionAdapter
from localization.offset_measurement import VisionCalibration,CenterMeasurer
from localization.camera_geometry import ImageGeometry
from recognition.onnx_pipeline.chess_recognizer import ChessRecognizer
from tools.vision_live import render,load_font,PreviewAnchors,PreviewLabels


def serializable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def save(directory,prefix,raw,result,canonical=None,overlay=None):
    directory.mkdir(parents=True,exist_ok=True)
    (directory/(prefix+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2,default=serializable),encoding='utf8')
    for suffix,pixels in (('raw',raw),('canonical',canonical),('preview',overlay)):
        if pixels is not None and not cv2.imwrite(str(directory/(prefix+'_'+suffix+'.png')),pixels):
            raise OSError('Diagnostic image could not be saved')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--image',type=Path)
    modes.add_argument('--video',type=Path)
    modes.add_argument('--camera',type=int,default=None)
    parser.add_argument('--hand-model',type=Path)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path.home()/'.codex/cache/tasks/chessrobot-debug'/datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--max-seconds',type=float,default=0)
    args=parser.parse_args(argv)
    output=args.output_dir.resolve()
    if output.is_relative_to(ROOT):parser.error('Output directory must be outside the project')
    if args.camera is None and args.video is None and args.image is None:args.camera=2
    if args.camera is not None and args.hand_model is None:parser.error('Live camera requires --hand-model')
    if args.image is not None and not args.image.is_file():parser.error('Image missing')
    if args.video is not None and not args.video.is_file():parser.error('Video missing')
    cv2.setNumThreads(4)
    calibration=VisionCalibration.load(args.calibration) if args.calibration else VisionCalibration.from_config()
    recognizer=ChessRecognizer()
    if recognizer.pose_session is None or recognizer.classifier_session is None:raise RuntimeError('Required ONNX model missing')
    if args.image:
        raw=cv2.imread(str(args.image))
        if raw is None:raise ValueError('Image decode failed')
        raw=ImageGeometry(calibration).apply(raw)
        result=observe_board(recognizer,raw)
        result.update(source_kind='single_image',stable=False,control_ready=False,
                      note='Single image does not establish temporal stability',frame_id=1,captured_at=None)
        canonical=None
        if result['matrix'] is not None:
            canonical=cv2.warpPerspective(raw,np.asarray(result['matrix']),(450,500))
            if not result['rejection']:
                result['piece_offsets']=CenterMeasurer(calibration).measure(canonical,result['grid_chars'])
        save(output,'image',raw,result,canonical)
        print(json.dumps(dict(output=str(output),rejection=result['rejection'],stable=False,control_ready=False)))
        return 2 if result['rejection'] else 0
    adapter=ONNXVisionAdapter(recognizer,calibration=calibration,headless=True,
                             measure_all=True,require_hand_check=args.camera is not None,allow_control=False)
    camera=hands=video=None;fonts=None;anchors=PreviewAnchors();labels=PreviewLabels();show_offsets=True
    output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();last_save=-1.;frame_id=0;last_frame=-1;last_event=None
    if not args.headless:fonts=(load_font(22),load_font(17),load_font(14))
    try:
        if args.camera is not None:
            camera=LatestCamera(args.camera,1920,1080).start()
            hands=LatestHands(camera,HandCheck(args.hand_model)).start()
        else:
            video=cv2.VideoCapture(str(args.video))
            if not video.isOpened():raise RuntimeError('Video cannot open')
        with (output/'events.jsonl').open('w',encoding='utf8') as log:
            while not args.max_seconds or time.monotonic()-start<args.max_seconds:
                if camera:
                    if camera.error:raise RuntimeError(camera.error)
                    if hands.error:raise RuntimeError(hands.error)
                    packet=camera.read()
                    if packet is None:time.sleep(.01);continue
                    view=camera
                else:
                    ok,raw=video.read()
                    if not ok:break
                    frame_id+=1;packet=CameraFrame(frame_id,time.monotonic(),raw)
                    view=SimpleNamespace(actual_size=(raw.shape[1],raw.shape[0]),capture_fps=video.get(cv2.CAP_PROP_FPS),preview_fps=0)
                if packet.frame_id==last_frame:time.sleep(.005);continue
                last_frame=packet.frame_id
                event=adapter.process_packet(packet,hand_record=hands.read() if hands else None)
                if adapter.error:raise RuntimeError(adapter.error)
                if event is not None:
                    last_event=event.move_str
                    log.write(json.dumps(dict(kind='move_proposal',event_id=event.event_id,move=last_event),ensure_ascii=False)+'\n');log.flush()
                snapshot=adapter.get_snapshot()
                canonical=None;overlay=None
                if snapshot is not None:
                    snapshot['source_kind']='camera' if camera else 'video'
                    snapshot['control_ready']=False
                    exact=adapter.last_published_packet
                    if exact is not None and exact.frame_id==snapshot['frame_id']:
                        canonical=cv2.warpPerspective(exact.image,np.asarray(snapshot['matrix']),(450,500))
                if not args.headless:
                    frame=adapter.image_geometry.apply(packet.image)
                    overlay=render(frame,adapter.gate,view,fonts,last_event or '仅视觉观察，无动作',show_offsets,
                        hands.read().count if hands and hands.read() else 0,adapter.motion.transform,anchors,True,labels,{})
                    cv2.imshow('ChessRobot Visual Diagnostics - NO ROBOT',overlay)
                    key=cv2.waitKey(1)&255
                    if key in (27,ord('q')):break
                    if key==ord('r'):adapter.reset_baseline();anchors.clear();labels.clear()
                    if key==ord('o'):show_offsets=not show_offsets
                    if key==ord('s'):
                        save(output,datetime.now().strftime('%Y%m%d_%H%M%S_%f'),packet.image,
                             dict(state=adapter.gate.state,reason=adapter.gate.reason,snapshot=snapshot),canonical,overlay)
                if time.monotonic()-last_save>=1:
                    save(output,'latest',packet.image,dict(state=adapter.gate.state,reason=adapter.gate.reason,
                         source_kind='camera' if camera else 'video',control_ready=False,snapshot=snapshot),canonical,overlay)
                    last_save=time.monotonic()
                if video:
                    fps=video.get(cv2.CAP_PROP_FPS)
                    time.sleep(1/min(max(fps,1),60))
    finally:
        adapter.close()
        if hands:hands.close()
        if camera:camera.close()
        if video:video.release()
        if not args.headless:cv2.destroyAllWindows()
    return 0


if __name__=='__main__':raise SystemExit(main())
