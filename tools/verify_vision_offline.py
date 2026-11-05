"""Reproducible software acceptance with physical camera/CH347 access blocked."""
import argparse
from contextlib import redirect_stdout,redirect_stderr,ExitStack
from datetime import datetime,timezone,timedelta
import gc
import hashlib
from importlib.metadata import version,PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True


def inventory():
    records=[]
    for name in ('pose_rtmpose_4kpt.onnx','layout_swinv2_nano.onnx','layout_swinv2_tiny_finetuned.onnx',
                 'piece_classifier_mobilenetv3.onnx','patch_classifier_mobilenetv3.onnx'):
        path=ROOT/'recognition/models/onnx'/name
        record=dict(file=name,exists=path.is_file(),required=name in ('pose_rtmpose_4kpt.onnx','layout_swinv2_nano.onnx'))
        if path.is_file():
            try:
                record['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                options=ort.SessionOptions();options.intra_op_num_threads=4
                session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
                record['inputs']=[dict(name=x.name,shape=x.shape,type=x.type) for x in session.get_inputs()]
                record['outputs']=[dict(name=x.name,shape=x.shape,type=x.type) for x in session.get_outputs()]
                feed={x.name:np.zeros([n if isinstance(n,int) and n>0 else 1 for n in x.shape],np.float32) for x in session.get_inputs()}
                outputs=session.run(None,feed);times=[]
                for _ in range(3):
                    start=time.perf_counter();outputs=session.run(None,feed);times.append((time.perf_counter()-start)*1000)
                record.update(loadable=True,finite=all(np.isfinite(x).all() for x in outputs),
                    actual_outputs=[list(x.shape) for x in outputs],p50_ms=float(np.percentile(times,50)),
                    p95_ms=float(np.percentile(times,95)),purpose='Shape/finite output check with zero input, NOT recognition accuracy')
                del session;gc.collect()
            except Exception as exc:record.update(loadable=False,error=str(exc))
        records.append(record)
    return records


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path.home()/'.codex/cache/tasks/chessrobot-offline'/datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'))
    args=parser.parse_args(argv);out=args.output_dir.resolve()
    if out.is_relative_to(ROOT):parser.error('Output must be outside source project')
    out.mkdir(parents=True,exist_ok=True);temporary=out/'temporary';temporary.mkdir(exist_ok=True)
    tempfile.tempdir=str(temporary)
    os.environ['TMP']=os.environ['TEMP']=str(temporary)
    os.environ['YOLO_CONFIG_DIR']=str(temporary/'ultralytics')
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    cv2.setNumThreads(4)
    models=inventory();attempts=[]
    original_capture=cv2.VideoCapture
    def capture(source,*a,**k):
        if isinstance(source,(int,np.integer)) or (isinstance(source,str) and source.isdigit()):
            attempts.append('physical_camera');raise AssertionError('Physical camera forbidden in offline acceptance')
        return original_capture(source,*a,**k)
    with ExitStack() as stack:
        stack.enter_context(patch('cv2.VideoCapture',side_effect=capture))
        import ctypes
        if hasattr(ctypes,'windll'):
            original_load=ctypes.windll.LoadLibrary
            def load(name):
                if 'ch347' in str(name).lower():
                    attempts.append('CH347');raise AssertionError('CH347 forbidden in offline acceptance')
                return original_load(name)
            stack.enter_context(patch.object(ctypes.windll,'LoadLibrary',side_effect=load))
        with (out/'tests.log').open('w',encoding='utf8') as log,redirect_stdout(log),redirect_stderr(log):
            suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
            result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    packages={}
    for name in ('opencv-python','numpy','onnxruntime','mediapipe','PyQt5','gradio'):
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    passed=result.wasSuccessful() and not attempts and all(not m['required'] or (m.get('loadable') and m.get('finite')) for m in models)
    report=dict(passed=passed,created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        python=sys.version,platform=platform.platform(),packages=packages,models=models,
        tests=dict(run=result.testsRun,passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
            failures=len(result.failures),errors=len(result.errors),skips=[dict(test=str(t),reason=r) for t,r in result.skipped]),
        forbidden_device_attempts=attempts,real_camera_tested=False,real_arm_tested=False,
        failures=[dict(test=str(t),trace=trace) for t,trace in result.failures+result.errors])
    (out/'acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(passed=passed,tests=report['tests'],report=str(out/'acceptance.json')),ensure_ascii=True))
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
