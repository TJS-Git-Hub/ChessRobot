"""Gen4 adapter sharing the live observer's recognition, motion and tracking.

Trusted snapshots, move proposals and physical center measurements are separate.
Game acknowledgement commits a human move; new post-action frames commit a robot
move. This module never imports hardware or constructs a camera on import.
"""
import copy
import queue
import threading
import time
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from recognition.onnx_pipeline.chess_recognizer import ChessRecognizer
from localization.live_vision import (CameraFrame, LiveGate, MotionCheck, GeometryStabilizer,
    LatestHands, HandCheck, board_hand_count, observe_board)
from localization.appearance_verifier import AppearanceTracker
from localization.offset_measurement import VisionCalibration, CenterMeasurer, require_offset
from localization.vision_session import BoardObservation, VisionSession, VerificationStatus
from localization.vision_contract import grid_to_tensor, CHANNEL_ORDER
from localization.camera_geometry import ImageGeometry


class VisionState(Enum):
    OCCLUDED_OR_MOVING='OCCLUDED_OR_MOVING'
    SETTLING='SETTLING'
    CONFIRMED_STABLE='CONFIRMED_STABLE'


grid_to_rl_tensor=grid_to_tensor


class ONNXVisionAdapter:
    def __init__(self,recognizer=None,move_queue=None,ready_event=None,motion_threshold=6.5,
                 stable_window_size=2,*,calibration=None,clock=time.monotonic,executor=None,
                 hand_model=None,require_hand_check=False,headless=False,measure_all=False,
                 enable_tracking=True,allow_control=False):
        self.recognizer=recognizer if recognizer is not None else ChessRecognizer()
        if recognizer is None and (self.recognizer.pose_session is None or self.recognizer.classifier_session is None):
            raise RuntimeError('Required ONNX model missing')
        self.move_queue=move_queue if move_queue is not None else queue.Queue()
        self.ready_event=ready_event if ready_event is not None else threading.Event()
        self.session=VisionSession(self.move_queue,self.ready_event,clock=clock)
        self.clock=clock;self.calibration=calibration or VisionCalibration.from_config()
        self.measurer=CenterMeasurer(self.calibration)
        self.image_geometry=ImageGeometry(self.calibration)
        self.gate=LiveGate(confirmations=max(2,stable_window_size),max_age=2.,confirmation_age=4.)
        self.motion=MotionCheck();self.geometry=GeometryStabilizer();self.tracker=AppearanceTracker()
        self.executor=executor if executor is not None else ThreadPoolExecutor(max_workers=1,thread_name_prefix='vision-classifier')
        self._owns_executor=executor is None;self._pending=None;self._process_lock=threading.Lock()
        self._seen_frame=-1;self._generation=self.session.generation;self._last_infer=-1.;self._last_verify=-1.
        self._last_published=-1;self._last_capture_id=-1;self.stopped=False;self.error=None
        self.hand_model=hand_model;self.require_hand_check=require_hand_check;self.headless=headless
        self.measure_all=measure_all;self.enable_tracking=enable_tracking;self.ui_callback=None
        self.allow_control=allow_control
        self.last_result=None;self.last_observation=None;self.motion_threshold=motion_threshold
        self.last_published_packet=None

    @property
    def state(self):return VisionState(self.gate.state)
    @property
    def suppress_detection(self):return self.session.mode=='MOVING'
    @suppress_detection.setter
    def suppress_detection(self,value):
        if value:
            self.session.mode='MOVING';self.session.invalidate('Externally suppressed')
        elif self.session.action is not None:
            raise RuntimeError('Complete robot action with finish_action(), not blind baseline reset')
        else:
            self.session.mode='OBSERVING';self.session.invalidate('External suppression ended')
    @property
    def last_stable_grid(self):return copy.deepcopy(self.session.baseline)
    @property
    def latest_offsets(self):
        s=self.session.snapshot()
        return s['piece_offsets'] if s else {}

    def reset_baseline(self,new_expected_count=None):
        # Preserve the committed baseline. A failed move may not redefine truth.
        self.session.invalidate('Reconfirming committed board')

    def get_snapshot(self):return self.session.snapshot()

    def get_piece_offset(self,pos_str):
        return require_offset(self.latest_offsets.get(pos_str.lower()),self.clock(),self.session.max_age)

    def prepare_action(self,source,target,board,timeout=5.,tolerance_mm=1.2):
        if not self.allow_control:raise RuntimeError('This visual source is observation-only')
        return self.session.request_pick(source,target,board,timeout,tolerance_mm)

    def finish_action(self,token,barrier_frame=None):
        self.session.finish_action(token,self._last_capture_id if barrier_frame is None else barrier_frame)

    def post_move_verification(self,placed_pos_str=None,expected_char=None,*,token=None,timeout=8.,stop_event=None):
        if token is None:
            raise ValueError('Post-action verification requires an action token and a fresh-frame fence')
        return self.session.wait_verification(token,timeout,stop_event)

    def _infer(self,packet,epoch,view,verified):
        result=observe_board(self.recognizer,packet.image,geometry=self.geometry,
            view_transform=view,epoch=epoch,verified_geometry=verified,
            allow_recovery=verified is None)
        result.update(frame_id=packet.frame_id,captured_at=packet.captured_at,view_transform=view.tolist())
        return result

    def process_frame(self,frame,*,frame_id=None,captured_at=None,hand_record=None,motion_reason=None):
        if isinstance(frame,CameraFrame):packet=frame
        else:
            if frame_id is None or captured_at is None:
                raise ValueError('Physical frame_id and acquisition captured_at are required; calls are not frames')
            packet=CameraFrame(frame_id,captured_at,frame)
        return self.process_packet(packet,hand_record=hand_record,motion_reason=motion_reason)

    def process_packet(self,packet,*,hand_record=None,motion_reason=None):
        with self._process_lock:
            now=self.clock();generation=self.session.generation
            if packet is None:
                self.session.invalidate('Camera disconnected');self.gate.invalidate('Camera disconnected');self.tracker.clear();return None
            self._last_capture_id=max(self._last_capture_id,packet.frame_id)
            if self.stopped or self.session.closed:return None
            if generation!=self._generation:
                self.gate.invalidate(self.session.reason);self.tracker.clear();self._generation=generation
            if self.session.mode=='MOVING':return None
            if packet.frame_id<=self._seen_frame:
                if now-packet.captured_at>.60:self.session.invalidate('Camera frame expired')
                return None
            self._seen_frame=packet.frame_id
            if not 0<=now-packet.captured_at<=.60:
                self.session.invalidate('Camera acquisition timestamp expired');self.gate.invalidate('Camera frame expired');self.tracker.clear();return None
            hand_record=self.image_geometry.hands(hand_record,packet.image.shape)
            packet=CameraFrame(packet.frame_id,packet.captured_at,self.image_geometry.apply(packet.image))
            polygon=self.tracker.board_polygon(self.motion.transform) if self.tracker.ready(self.gate.epoch) else None
            reason=self.motion.update(packet.image,now,polygon) if motion_reason is None else motion_reason
            if hand_record is not None:
                if not 0<=now-hand_record.captured_at<=.5:reason='Hand check expired'
                elif board_hand_count(hand_record,polygon,packet.image.shape):reason='Hand overlaps board'
            elif self.require_hand_check:reason='Waiting for independent hand check'
            self.gate.observe(packet,now,reason)
            if reason:
                self.session.invalidate(reason);self.tracker.clear();return None
            if self.tracker.epoch is not None and not self.tracker.ready(self.gate.epoch):self.tracker.clear()
            source=None
            if self.enable_tracking and self.tracker.ready(self.gate.epoch) and self.gate.scene_is_stable(now) and now-self._last_verify>=.15:
                if self.gate.accept_tracking(self.tracker,packet,self.motion.transform.copy(),now):source=packet
                else:
                    self.session.invalidate(self.gate.reason);return None
                self._last_verify=now
            if self._pending is not None:
                future,token,worker_generation,original=self._pending
                if not future.done() and now-original.captured_at>5:
                    self.session.invalidate('Classifier timed out');self.gate.invalidate('Classifier timed out');self.tracker.clear()
                    self.error='Classifier timed out';return None
                if future.done():
                    self._pending=None
                    try:result=future.result()
                    except Exception as exc:
                        self.error=str(exc);self.session.invalidate('Inference failed: '+str(exc));self.gate.invalidate(self.error);self.tracker.clear();return None
                    if worker_generation==self.session.generation:
                        before=self.gate.epoch
                        if self.tracker.ready(self.gate.epoch):
                            self.gate.observe_neural_while_tracking(token,result,self.clock(),self.tracker)
                        else:
                            accepted=self.gate.accept(token,result,self.clock())
                            if accepted:
                                source=original
                                if (self.enable_tracking and not result.get('raw_unknown') and not result['rejection']
                                        and result['grid_chars']==self.gate.output['grid_chars']):
                                    self.tracker.seed(self.gate.output,original,self.gate.epoch,result)
                        self.last_result=result
                        if self.gate.epoch!=before:
                            self.session.invalidate(self.gate.reason);return None
            now=self.clock()
            if generation!=self.session.generation:return None
            interval=1. if self.tracker.ready(self.gate.epoch) else .20
            if self._pending is None and self.gate.can_infer(now) and now-self._last_infer>=interval:
                verified=None
                if self.tracker.ready(self.gate.epoch):
                    mat=self.tracker.transport_matrix@self.tracker.transport_view@np.linalg.inv(self.motion.transform)
                    verified=cv2.perspectiveTransform(np.float32([[[50,50],[400,50],[50,450],[400,450]]]),np.linalg.inv(mat))[0]
                token=(self.gate.epoch,packet.frame_id)
                self._pending=(self.executor.submit(self._infer,packet,self.gate.epoch,self.motion.transform.copy(),verified),token,generation,packet)
                self._last_infer=now
            result=self.gate.output
            if result is None:
                if self.session.latest is not None:self.session.invalidate(self.gate.reason)
                return None
            if source is None or result['frame_id']!=source.frame_id or result['frame_id']<=self._last_published:return None
            offsets={}
            with self.session.condition:positions=sorted(self.session.offset_request)
            if self.measure_all or positions:
                warped=cv2.warpPerspective(source.image,np.asarray(result['matrix']),(450,500))
                offsets=self.measurer.measure(warped,result['grid_chars'],frame_id=source.frame_id,
                    captured_at=source.captured_at,positions=None if self.measure_all else positions)
                if not self.allow_control:
                    for info in offsets.values():info['control_valid']=False
            self._last_published=result['frame_id']
            result['piece_offsets']=copy.deepcopy(offsets)
            result['image_space']=self.image_geometry.image_space
            result['offsets_valid']=bool(offsets) and all(v.get('valid') for v in offsets.values())
            event=self.session.publish(result,offsets,generation)
            if self.session.latest is not None and self.session.latest['frame_id']==source.frame_id:
                self.last_published_packet=source
            if event is not None:self.last_observation=event
            return event

    def run(self,stream):
        hands=None
        missing_since=None;seen_packet=False
        last_display=-1
        try:
            if self.require_hand_check:
                if self.hand_model is None:raise RuntimeError('Live integration requires --hand-model')
                hands=LatestHands(stream,HandCheck(self.hand_model)).start()
            while not self.stopped and not self.session.closed:
                if getattr(stream,'stopped',False):break
                packet=stream.read()
                if packet is None:
                    if missing_since is None:missing_since=self.clock()
                    worker=getattr(stream,'thread',None)
                    if ((getattr(stream,'error',None) and worker is not None and not worker.is_alive())
                            or self.clock()-missing_since>(2 if seen_packet else 10)):
                        raise RuntimeError(getattr(stream,'error',None) or 'Camera stream lost')
                else:
                    missing_since=None;seen_packet=True
                if packet is not None and not isinstance(packet,CameraFrame):
                    raise TypeError('Gen4 needs a metadata-bearing camera stream')
                if hands is not None and hands.error:raise RuntimeError(hands.error)
                self.process_packet(packet,hand_record=hands.read() if hands else None)
                if self.error:raise RuntimeError(self.error)
                if packet is not None and packet.frame_id!=last_display and self.ui_callback is not None:
                    self.ui_callback(self.image_geometry.apply(packet.image).copy())
                    last_display=packet.frame_id
                elif packet is not None and packet.frame_id!=last_display and not self.headless:
                    display=packet.image.copy()
                    cv2.putText(display,self.gate.state+' / '+self.session.mode,(15,30),0,.7,(0,200,200),2)
                    cv2.imshow('ChessRobot Vision Monitor',display)
                    last_display=packet.frame_id
                    if cv2.waitKey(1)&255 in (27,ord('q')):break
                time.sleep(.005)
        except Exception as exc:
            self.error=str(exc);self.session.invalidate('Vision stopped: '+str(exc));raise
        finally:
            if hands:hands.close()
            self.close()

    def close(self):
        self.stopped=True;self.session.close()
        if self._owns_executor:self.executor.shutdown(wait=False,cancel_futures=True)
