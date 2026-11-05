"""Transactional vision/game synchronization and post-action freshness fences."""
import copy
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
import numpy as np
from localization.vision_contract import (CHANNEL_ORDER, validate_grid, grid_to_tensor,
    grid_to_fen, infer_move, moved_grid, board_to_grid, parse_coordinate)
from localization.offset_measurement import require_offset, OffsetUnavailable


class VerificationStatus(str,Enum):
    SUCCESS='SUCCESS'
    ERROR_EXCEEDED='ERROR_EXCEEDED'
    STATE_CONFLICT='STATE_CONFLICT'
    UNCONFIRMED='UNCONFIRMED'
    TIMEOUT='TIMEOUT'
    CANCELLED='CANCELLED'


@dataclass
class BoardObservation:
    move_str: str
    from_pos: str
    to_pos: str
    piece: str
    is_capture: bool
    grid_chars: list
    fen: str
    tensor_14x10x9: np.ndarray
    piece_offsets: dict
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda:uuid4().hex)
    frame_id: int = -1
    captured_at: float = -1.
    generation: int = -1
    tensor_channel_order: str = CHANNEL_ORDER
    schema_version: int = 2
    stable: bool = False
    confidence: float = 0.
    conf_grid: list = field(default_factory=list)
    confidence_kind: str = 'minimum_cell_probability_not_whole_board_accuracy'
    source: str = 'unspecified'

    def __iter__(self):return iter((self.from_pos,self.to_pos))


class VisionSession:
    def __init__(self,move_queue=None,ready_event=None,*,clock=time.monotonic,max_age=1.,require_calibration=True):
        self.clock=clock;self.max_age=max_age;self.require_calibration=require_calibration
        self.move_queue=move_queue if move_queue is not None else queue.Queue()
        self.ready_event=ready_event if ready_event is not None else threading.Event()
        self.condition=threading.Condition(threading.RLock())
        self.generation=0;self.latest=None;self.baseline=None;self.pending=None
        self.mode='OBSERVING';self.reason='Waiting for board';self.red_turn=True
        self.last_frame_id=-1;self.rejected_grid=None;self.offset_request=set()
        self.action=None;self.verification=None;self.closed=False

    def _drain(self):
        while True:
            try:self.move_queue.get_nowait()
            except queue.Empty:return

    def invalidate(self,reason):
        with self.condition:
            self.generation+=1;self.latest=None;self.pending=None;self.ready_event.clear();self.reason=reason
            self._drain();self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            if self.latest is None or not 0<=self.clock()-self.latest['captured_at']<=self.max_age:
                self.ready_event.clear()
                return None
            return copy.deepcopy(self.latest)

    def publish(self,result,offsets,generation):
        with self.condition:
            now=self.clock();capture=result['captured_at'];fid=result['frame_id']
            if (self.closed or generation!=self.generation or self.mode=='MOVING'
                    or not 0<=now-capture<=self.max_age or fid<=self.last_frame_id):return None
            grid=validate_grid(result['grid_chars'])
            from localization.live_vision import grid_reason
            if result.get('rejection') or result.get('raw_unknown') or grid_reason(grid,result.get('conf_grid'),None):
                return None
            self.last_frame_id=fid
            snapshot=copy.deepcopy(result)
            snapshot.update(piece_offsets=copy.deepcopy(offsets),generation=generation,stable=True,
                            fen=grid_to_fen(grid,self.red_turn),tensor_channel_order=CHANNEL_ORDER,
                            tensor_14x10x9=grid_to_tensor(grid),schema_version=2,
                            confidence=float(np.min(result['conf_grid'])),
                            confidence_kind='minimum_cell_probability_not_whole_board_accuracy')
            self.latest=snapshot;self.ready_event.set()
            if self.mode=='VERIFYING':
                action=self.action
                if fid<=action['barrier_frame'] or capture<=action['finished_at']:
                    self.latest=None;self.ready_event.clear();return None
                if grid!=action['expected']:
                    self.verification=dict(status=VerificationStatus.STATE_CONFLICT.value,
                                           reason='Observed board differs from commanded board',frame_id=fid)
                else:
                    info=offsets.get(action['target'])
                    if not info or not info.get('valid') or info.get('frame_id')!=fid or info.get('captured_at')!=capture:
                        self.verification=dict(status=VerificationStatus.UNCONFIRMED.value,reason='Fresh target center unavailable',frame_id=fid)
                    else:
                        try:
                            require_offset(info,now,self.max_age,self.require_calibration)
                            status=(VerificationStatus.SUCCESS if info['offset_dist_mm']<=action['tolerance_mm']
                                    else VerificationStatus.ERROR_EXCEEDED)
                            self.verification=dict(status=status.value,frame_id=fid,captured_at=capture,
                                                   piece=grid[parse_coordinate(action['target'])[1]][parse_coordinate(action['target'])[0]],
                                                   offset=copy.deepcopy(info),reason='')
                        except OffsetUnavailable as exc:
                            self.verification=dict(status=VerificationStatus.UNCONFIRMED.value,reason=str(exc),frame_id=fid)
                self.verification.update(action_token=action['token'],barrier_frame=action['barrier_frame'],
                                         finished_at=action['finished_at'])
                self.condition.notify_all();return None
            if self.mode not in ('OBSERVING','CONFLICT'):return None
            if self.baseline is None:
                self.baseline=copy.deepcopy(grid);self.reason='Initial board confirmed'
                self.condition.notify_all();return None
            if grid==self.baseline:
                self.rejected_grid=None;self.reason='Board unchanged';self.mode='OBSERVING'
                self.condition.notify_all();return None
            key=tuple(map(tuple,grid))
            move=infer_move(self.baseline,grid)
            if move is None:
                self.mode='CONFLICT';self.reason='Multiple or ambiguous board changes'
            elif self.pending is None and key!=self.rejected_grid:
                source,target,piece,capture_flag=move
                event=BoardObservation(source+target,source,target,piece,capture_flag,copy.deepcopy(grid),
                    grid_to_fen(grid,not self.red_turn),grid_to_tensor(grid),copy.deepcopy(offsets),
                    frame_id=fid,captured_at=capture,generation=generation,stable=True,
                    confidence=snapshot['confidence'],conf_grid=copy.deepcopy(result['conf_grid']),
                    source=result.get('source','neural'))
                self.pending=event;self.move_queue.put(copy.deepcopy(event));self.reason='Move awaiting game acknowledgement'
                self.condition.notify_all();return copy.deepcopy(event)
            self.condition.notify_all();return None

    def commit_human_move(self,event_id,board):
        with self.condition:
            event=self.pending;snapshot=self.snapshot()
            if (event is None or event.event_id!=event_id or event.generation!=self.generation
                    or snapshot is None or snapshot['grid_chars']!=event.grid_chars):
                raise ValueError('Move event expired or no longer matches the camera')
            if board_to_grid(board.snapshot())!=self.baseline:raise ValueError('Software and visual baseline differ')
            x1,y1=parse_coordinate(event.from_pos);x2,y2=parse_coordinate(event.to_pos)
            move=f'{x1}{y1}{x2}{y2}'
            if not board.is_move_valid(move):
                self.reject_move(event_id,'Illegal game move')
                raise ValueError('Illegal game move')
            expected=moved_grid(self.baseline,event.from_pos,event.to_pos)
            if expected!=snapshot['grid_chars']:raise ValueError('Full board mismatch')
            board.make_move(move)
            self.baseline=copy.deepcopy(expected);self.red_turn=board.red_turn;self.pending=None
            self.latest['fen']=grid_to_fen(expected,self.red_turn)
            self.mode='OBSERVING';self.reason='Move committed';self._drain()
            return event

    def reject_move(self,event_id,reason='Move rejected'):
        with self.condition:
            if self.pending is not None and self.pending.event_id==event_id:
                self.rejected_grid=tuple(map(tuple,self.pending.grid_chars));self.pending=None
                self.mode='CONFLICT';self.reason=reason;self._drain()

    def synchronize_game(self,board,*,initialize=False,red_turn=True):
        with self.condition:
            s=self.snapshot()
            if s is None:raise ValueError('No fresh trusted board for initialization')
            if initialize:board.load_vision_grid(s['grid_chars'],red_turn)
            elif board_to_grid(board.snapshot())!=s['grid_chars']:raise ValueError('Observed board is not software starting position')
            self.baseline=copy.deepcopy(s['grid_chars']);self.red_turn=board.red_turn
            self.latest['fen']=grid_to_fen(self.baseline,self.red_turn)
            self.pending=None;self._drain();self.mode='OBSERVING'

    def request_pick(self,source,target,board,timeout=5.,tolerance_mm=1.2):
        """Wait for newly measured source/captured target, then atomically fence vision."""
        x1,y1=parse_coordinate(source);x2,y2=parse_coordinate(target)
        before=board_to_grid(board.snapshot());expected=moved_grid(before,source,target)
        if not board.is_move_valid(f'{x1}{y1}{x2}{y2}'):raise OffsetUnavailable('Robot move is not legal in software board')
        required={source.lower()}
        if before[y2][x2]!='.':required.add(target.lower())
        start=self.clock();deadline=time.monotonic()+timeout
        with self.condition:
            if self.mode not in ('OBSERVING',) or self.pending is not None or self.baseline!=before:
                raise OffsetUnavailable('Vision/game not synchronized for robot move')
            self.offset_request=required
            try:
                while not self.closed:
                    s=self.snapshot()
                    if self.mode!='OBSERVING' or self.baseline!=before or board_to_grid(board.snapshot())!=before:
                        raise OffsetUnavailable('Board changed while preparing pickup')
                    if s and s['grid_chars']==before and s['captured_at']>start:
                        offsets={}
                        try:
                            for pos in required:
                                info=s['piece_offsets'].get(pos)
                                if (info or {}).get('frame_id')!=s['frame_id'] or (info or {}).get('captured_at')!=s['captured_at']:
                                    raise OffsetUnavailable('Center belongs to another frame')
                                offsets[pos]=require_offset(info,self.clock(),self.max_age,self.require_calibration)
                        except OffsetUnavailable as exc:self.reason=str(exc)
                        else:
                            token=uuid4().hex
                            self.action=dict(token=token,source=source.lower(),target=target.lower(),before=before,expected=expected,
                                             tolerance_mm=tolerance_mm,offsets=offsets,source_frame=s['frame_id'],
                                             measured_at=s['captured_at'],expires_at=s['captured_at']+self.max_age)
                            self.mode='MOVING';self.invalidate('Robot motion suppresses vision')
                            return copy.deepcopy(self.action)
                    remaining=deadline-time.monotonic()
                    if remaining<=0:raise OffsetUnavailable('Timed out waiting for fresh valid pickup centers: '+self.reason)
                    self.condition.wait(min(.1,remaining))
                raise OffsetUnavailable('Vision stopped')
            finally:self.offset_request=set()

    def finish_action(self,token,barrier_frame):
        with self.condition:
            if not self.action or self.action['token']!=token or self.mode!='MOVING':raise ValueError('Unknown robot action')
            self.action.update(finished_at=self.clock(),barrier_frame=barrier_frame)
            self.verification=None;self.mode='VERIFYING';self.offset_request={self.action['target']}
            self.invalidate('Waiting for genuinely new post-action frames')

    def wait_verification(self,token,timeout=8.,stop_event=None):
        deadline=time.monotonic()+timeout
        with self.condition:
            while True:
                if self.closed or (stop_event is not None and stop_event.is_set()):
                    return dict(status=VerificationStatus.CANCELLED.value,reason='Stopped')
                if not self.action or self.action['token']!=token:raise ValueError('Unknown action')
                if self.verification is not None:return copy.deepcopy(self.verification)
                remaining=deadline-time.monotonic()
                if remaining<=0:return dict(status=VerificationStatus.TIMEOUT.value,reason='No fresh verified post-action board')
                self.condition.wait(min(.1,remaining))

    def commit_robot_move(self,token,board,move):
        with self.condition:
            if (not self.action or self.action['token']!=token or not self.verification
                    or self.verification['status']!=VerificationStatus.SUCCESS.value):raise ValueError('Robot move not visually verified')
            s=self.snapshot()
            if s is None or s['grid_chars']!=self.action['expected']:raise ValueError('Verification expired')
            if board_to_grid(board.snapshot())!=self.action['before']:raise ValueError('Software changed during robot action')
            x1,y1=parse_coordinate(self.action['source']);x2,y2=parse_coordinate(self.action['target'])
            if move!=f'{x1}{y1}{x2}{y2}' or not board.is_move_valid(move):
                raise ValueError('Committed move differs from visually verified action')
            board.make_move(move)
            self.baseline=copy.deepcopy(self.action['expected']);self.red_turn=board.red_turn
            self.latest['fen']=grid_to_fen(self.baseline,self.red_turn)
            self.action=None;self.verification=None;self.offset_request=set();self.mode='OBSERVING';self.reason='Robot move committed'

    def close(self):
        with self.condition:
            self.closed=True;self.invalidate('Vision closed');self.condition.notify_all()
