"""Single-profile Gen4 live camera observer. Never constructs a robot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, deque
from datetime import datetime
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from localization.live_vision import (
    LatestCamera, LiveGate, MotionCheck, HandCheck, LatestHands, GeometryStabilizer, observe_board, position_warnings,
    board_hand_count,
)
from recognition.onnx_pipeline.chess_recognizer import ChessRecognizer
from localization.appearance_verifier import AppearanceTracker
from localization.offset_measurement import VisionCalibration, CenterMeasurer
from localization.camera_geometry import ImageGeometry

NAMES = dict(zip("KABNRCPkabnrcp", ["帅", "仕", "相", "马", "车", "炮", "兵",
                                  "将", "士", "象", "马", "车", "炮", "卒"]))
WINDOW = "ChessRobot - LIVE VISION ONLY"


def publish_snapshot(pending, destination):
    """Diagnostic readers on Windows may deny replacement while holding a file.

    Keep the last snapshot and try on the next tick. Optional monitoring must
    never terminate capture; unrelated filesystem errors remain visible.
    """
    try:
        pending.replace(destination)
        return True
    except PermissionError:
        return False


def infer(recognizer, packet, measure_offsets=False, view_transform=None, geometry=None, epoch=None,
          allow_recovery=True, verified_geometry=None):
    start = time.monotonic()
    result = observe_board(recognizer, packet.image, geometry=geometry,
                           view_transform=view_transform, epoch=epoch, allow_recovery=allow_recovery,
                           verified_geometry=verified_geometry)
    result.update(captured_at=packet.captured_at, frame_id=packet.frame_id,
                  view_transform=(np.eye(3) if view_transform is None else view_transform).tolist())
    if not result["rejection"] and measure_offsets:
        warped = cv2.warpPerspective(packet.image, np.asarray(result["matrix"]), (450, 500))
        result["piece_offsets"] = recognizer.detect_piece_offsets(warped, result["grid_chars"])
    result["inference_ms"] = (time.monotonic()-start)*1000
    return result


def describe_change(old, new):
    if old is None:
        return "初始可信盘面已建立"
    changes = [(r, c) for r in range(10) for c in range(9) if old[r][c] != new[r][c]]
    if not changes:
        return ""
    lost = [(r, c) for r, c in changes if old[r][c] != "." and new[r][c] == "."]
    if len(changes) == 2 and len(lost) == 1:
        r0, c0 = lost[0]
        r1, c1 = next(pos for pos in changes if pos != lost[0])
        piece = old[r0][c0]
        previous = old[r1][c1]
        if new[r1][c1] == piece and (previous == "." or previous.isupper() != piece.isupper()):
            capture = "（吃子）" if previous != "." else ""
            # Observation only: full game legality is intentionally not claimed.
            return f"观察到 {NAMES[piece]} {chr(97+c0)}{r0} → {chr(97+c1)}{r1}{capture}，未提交游戏"
    return f"盘面有{len(changes)}处变化，仅更新观察，不提交游戏"


def load_font(size):
    candidates = [os.environ.get("CHESS_FONT", ""), r"C:\Windows\Fonts\msyh.ttc",
                  "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]
    for path in candidates:
        if path and Path(path).is_file():
            return ImageFont.truetype(path, size)
    raise RuntimeError("Chinese font missing; set CHESS_FONT to a Chinese font file")


@lru_cache(maxsize=256)
def piece_badge(label, color, font):
    bounds = font.getbbox(label)
    badge = Image.new("RGB", (bounds[2]+6, bounds[3]+4), "#fff8d8")
    draw = ImageDraw.Draw(badge)
    draw.rectangle((0, 0, badge.width-1, badge.height-1), outline=color)
    draw.text((3, -1), label, font=font, fill=color)
    return badge


@lru_cache(maxsize=512)
def text_sprite(text, font, color):
    bounds = font.getbbox(text)
    tile = Image.new("RGBA", (max(1,bounds[2]+2),max(1,bounds[3]+2)))
    ImageDraw.Draw(tile).text((0,0),text,font=font,fill=color)
    return tile


class PreviewAnchors:
    """Display-only deadband; never modifies observed or compensation coordinates."""
    def __init__(self):
        self.points = {}
        self.box_offsets = {}
        self.radii = {}
        self.faces = []
        self.last_obstacles_at = -float('inf')

    def clear(self):
        self.points.clear()
        self.box_offsets.clear()
        self.radii.clear()
        self.faces.clear()
        self.last_obstacles_at = -float('inf')

    def point(self, key, xy):
        xy = np.asarray(xy, dtype=float)
        previous = self.points.get(key)
        if previous is None or np.linalg.norm(xy-previous) > 12:
            value = xy
        elif np.linalg.norm(xy-previous) <= 1.2:
            value = previous
        else:
            value = previous*.65 + xy*.35
        self.points[key] = value
        return value

    def radius(self, key, radius):
        old = self.radii.get(key, radius)
        value = old if abs(radius-old) < .6 else old*.7+radius*.3
        self.radii[key] = value
        return value

    def box(self, key, center, radius, size, circles, placed, bounds):
        offset = self.box_offsets.get(key)
        preferred = None if offset is None else (round(center[0]+offset[0]),
                                                 round(center[1]+offset[1]))
        box = coordinate_box(center, radius, size, circles, placed, bounds, preferred)
        if box is not None and (preferred is None or box[:2] != preferred):
            self.box_offsets[key] = (box[0]-center[0], box[1]-center[1])
        return box

    def obstacles(self, frame, now=None, minimum_interval=0):
        now = time.monotonic() if now is None else now
        previous = [face for face in self.faces if now-face[3] <= .5]
        if now-self.last_obstacles_at < minimum_interval:
            return [face[:3] for face in previous]
        self.last_obstacles_at = now
        used, current = set(), []
        for x, y, radius in piece_face_circles(frame):
            nearest = min((i for i in range(len(previous)) if i not in used),
                          key=lambda i: (previous[i][0]-x)**2+(previous[i][1]-y)**2,
                          default=None)
            if nearest is not None and (previous[nearest][0]-x)**2+(previous[nearest][1]-y)**2 < 100:
                ox, oy, old_radius, _ = previous[nearest]
                current.append((ox*.7+x*.3, oy*.7+y*.3, max(old_radius, radius), now))
                used.add(nearest)
            else:
                current.append((x, y, radius, now))
        current.extend(face for i, face in enumerate(previous) if i not in used)
        self.faces = current
        return [face[:3] for face in current]


class PreviewLabels:
    """Display memory only. Never supplies classifications to the live gate.

    An uncertain cell keeps its last confirmed glyph, explicitly marked as
    historical. Three new matching results clear the marker. A contradictory
    class needs two new results; repeated rendering is not extra evidence.
    Motion epochs and missing/expired displays discard the entire memory.
    """
    def __init__(self):
        self.clear()

    def clear(self):
        self.epoch = None
        self.frame_id = None
        self.cells = {}

    def update(self, result, epoch, confirmed=True):
        if result is None:
            self.clear()
            return {}, set()
        if epoch != self.epoch:
            self.clear()
            self.epoch = epoch
        key = (result["frame_id"], confirmed)
        if key != self.frame_id:
            self.frame_id = key
            if not confirmed:
                # A new candidate keeps the scene observable while the gate
                # rebuilds its consensus. It cannot teach display identities.
                for cell in self.cells.values():
                    if cell["label"] not in (None, "."):
                        cell.update(uncertain=True, pending=None, hits=0, recovery=0)
                return self.snapshot()
            unknown = set(uncertain_positions(result))
            for r in range(10):
                for c in range(9):
                    pos, ch = f"{chr(97+c)}{r}", result["grid_chars"][r][c]
                    cell = self.cells.setdefault(pos, dict(label=None, pending=None, hits=0,
                                                          uncertain=False, recovery=0))
                    if pos in unknown:
                        cell.update(uncertain=True, pending=None, hits=0, recovery=0)
                    elif cell["label"] is None:
                        cell.update(label=ch, uncertain=False)
                    elif ch != cell["label"]:
                        cell["hits"] = cell["hits"]+1 if cell["pending"] == ch else 1
                        cell.update(pending=ch, uncertain=True, recovery=0)
                        if cell["hits"] >= 2:
                            cell.update(label=ch, pending=None, hits=0, uncertain=False)
                    else:
                        cell.update(pending=None, hits=0)
                        cell["recovery"] += 1
                        if cell["recovery"] >= 3:
                            cell["uncertain"] = False
        return self.snapshot()

    def snapshot(self):
        labels = {pos: (cell["label"] if cell["label"] not in (None, ".") else "x")
                  if cell["uncertain"] else cell["label"] or "x"
                  for pos, cell in self.cells.items()}
        return labels, {pos for pos, cell in self.cells.items() if cell["uncertain"]}


def uncertain_positions(result):
    if result is None:
        return []
    counts = Counter(result["grid_chars"][r][c] for r in range(10) for c in range(9)
                     if result["conf_grid"][r][c] >= .50)
    limits = dict(zip("KABNRCP", (1, 2, 2, 2, 2, 2, 5)))
    limits.update({ch.lower(): count for ch, count in list(limits.items())})
    excess = {ch for ch, limit in limits.items() if counts[ch] > limit}
    return [f"{chr(97+c)}{r}" for r in range(10) for c in range(9)
            if result["grid_chars"][r][c] == "x" or
            result["grid_chars"][r][c] in excess or
            result["conf_grid"][r][c] < (.45 if result["grid_chars"][r][c] == "." else .50)]


def coordinate_leader(center, radius, box):
    cx,cy = float(center[0]),float(center[1])
    sx,sy = max(box[0],min(cx,box[2])),max(box[1],min(cy,box[3]))
    dx,dy = sx-cx,sy-cy
    scale = (radius+1)/max(math.hypot(dx,dy),1e-6)
    return (sx,sy),(cx+dx*scale,cy+dy*scale)


def coordinate_box(center, radius, size, circles, placed, bounds, preferred=None):
    """Find free space outside piece faces; omit a coordinate if none is free."""
    cx, cy = map(float,center)
    width, height = size
    left, top, right, bottom = bounds
    def free(box, clearance=2):
        x, y, x2, y2 = box
        if x < left or y < top or x2 > right or y2 > bottom:
            return False
        if any(not (x2 < b[0] or x > b[2] or y2 < b[1] or y > b[3]) for b in placed):
            return False
        if not all((max(x, min(ox, x2))-ox)**2+(max(y, min(oy, y2))-oy)**2 >= (rr+clearance)**2
                   for ox, oy, rr in circles):
            return False
        start,end = coordinate_leader(center,radius,box)
        dx,dy = end[0]-start[0],end[1]-start[1]
        length2 = max(dx*dx+dy*dy,1e-6)
        for ox,oy,rr in circles:
            if (ox-cx)**2+(oy-cy)**2 <= (radius*.7)**2:
                continue
            t = max(0.,min(1.,((ox-start[0])*dx+(oy-start[1])*dy)/length2))
            if (start[0]+t*dx-ox)**2+(start[1]+t*dy-oy)**2 < (rr+2)**2:
                return False
        return True

    if preferred is not None:
        x, y = preferred
        box = (x, y, x+width, y+height)
        if free(box):
            return box
    for extra in (7, 15, 24):
        for dx, dy in ((.7071, .7071), (-.7071, .7071), (.7071, -.7071),
                       (-.7071, -.7071), (1, 0), (-1, 0), (0, 1), (0, -1)):
            distance = radius + max(width, height)/2 + extra
            x = round(cx+dx*distance-width/2)
            y = round(cy+dy*distance-height/2)
            box = (x, y, x+width, y+height)
            if free(box, 8):
                return box
    return None


def piece_face_circles(frame):
    """Additional display obstacles from actual round rims; no class/mm output."""
    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 45, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if 12 < radius < 40 and cv2.contourArea(contour)/(np.pi*radius*radius) > .67:
            candidates.append((x, y, radius))
    circles = []
    for x, y, radius in sorted(candidates, key=lambda circle: -circle[2]):
        if all((x-ox)**2+(y-oy)**2 > 100 for ox, oy, _ in circles):
            circles.append((x, y, radius+2))
    return circles


@lru_cache(maxsize=128)
def sidebar_cell(text, color, font, pending=False):
    tile = Image.new("RGB", (31, 30), "white")
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, 30, 29), outline="#d5dce5")
    draw.text((6, 1), text, font=font, fill=color)
    if pending:
        draw.ellipse((25, 3, 28, 6), fill="#b16a00")
    return tile


def render(frame, gate, camera, fonts, event, show_offsets, hand_count,
           view_transform=None, anchors=None, show_coordinates=True, labels=None, diagnostics=None):
    scale = min(1.0, 1280/frame.shape[1], 720/frame.shape[0])
    view_w, view_h = round(frame.shape[1]*scale), round(frame.shape[0]*scale)
    header, footer, panel_w = 60, 34, 332
    image = Image.new("RGB", (view_w+panel_w, view_h+header+footer), "#111927")
    view_bgr = cv2.resize(frame, (view_w, view_h))
    image.paste(Image.fromarray(cv2.cvtColor(view_bgr, cv2.COLOR_BGR2RGB)), (0, header))
    draw = ImageDraw.Draw(image)
    def draw_text(position,content,font,fill):
        tile = text_sprite(content,font,fill)
        image.paste(tile,position,tile)
    width, height = image.size
    visible = getattr(gate, "display_output", None) or gate.output
    unconfirmed = False
    if visible is None and gate.scene_is_stable(time.monotonic()):
        candidate = gate.last_result
        if (candidate is not None and candidate.get("matrix") is not None
                and time.monotonic()-candidate["captured_at"] <= gate.max_age):
            visible, unconfirmed = candidate, True
    active = gate.output is not None and gate.state == "CONFIRMED_STABLE"
    anchors = anchors if anchors is not None else PreviewAnchors()
    labels = labels if labels is not None else PreviewLabels()
    if unconfirmed and not labels.cells:
        visible, unconfirmed = None, False
    display_labels, held = labels.update(visible, gate.epoch, confirmed=not unconfirmed)
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(labels=display_labels, held=sorted(held), boxes={}, leaders={}, epoch=gate.epoch,
                           unconfirmed_candidate=unconfirmed)
    unknown = set(uncertain_positions(visible))
    draw.rectangle((view_w, header, width, height-footer), fill="#f4f6fa")
    panel_x = view_w+12
    draw_text((panel_x, header+10), "识别棋盘  ·  黑方0线在上", font=fonts[1], fill="#1d2939")
    for c in range(9):
        draw_text((panel_x+27+c*32, header+40), chr(97+c), font=fonts[2], fill="#667085")
    for r in range(10):
        draw_text((panel_x+1, header+67+r*31), str(r), font=fonts[2], fill="#667085")
        for c in range(9):
            x, y = panel_x+20+c*32, header+64+r*31
            if visible is None:
                image.paste(sidebar_cell("", "#a0a9b5", fonts[1]), (x, y))
                continue
            pos = f"{chr(97+c)}{r}"
            ch = display_labels.get(pos, "x")
            text = "?" if ch == "x" else NAMES.get(ch, "·")
            color = ("#79818d" if pos in held else "#c52222" if ch.isupper()
                     else "#a0a9b5" if ch == "." else "#202630")
            image.paste(sidebar_cell(text, color, fonts[1], pos in held), (x, y))
    if visible is not None:
        positions = []
        inverse = np.linalg.inv(np.asarray(visible["matrix"]))
        if view_transform is not None:
            inverse = view_transform @ np.linalg.inv(np.asarray(
                visible.get("view_transform", np.eye(3)))) @ inverse
        for r in range(10):
            for c in range(9):
                ch = display_labels.get(f"{chr(97+c)}{r}", "x")
                if ch not in NAMES:
                    continue
                pos = f"{chr(97+c)}{r}"
                pts = np.float32([[[50+c*350/8, 50+r*400/9],
                                  [68+c*350/8, 50+r*400/9]]])
                mapped = cv2.perspectiveTransform(pts, inverse)[0]*scale
                center = anchors.point(pos, mapped[0]+[0, header])
                radius = anchors.radius(pos, float(np.clip(np.linalg.norm(mapped[1]-mapped[0]), 9, 30)))
                positions.append((pos, ch, center, radius))
                if show_offsets:
                    info=visible.get('piece_offsets',{}).get(pos)
                    if info is not None:
                        nominal=cv2.perspectiveTransform(np.float32([[info['nominal_px']]]),inverse)[0,0]*scale+[0,header]
                        x,y=nominal
                        draw.line((x-3,y,x+3,y),fill='#3999ff',width=1)
                        draw.line((x,y-3,x,y+3),fill='#3999ff',width=1)
                        if info.get('valid'):
                            detected=cv2.perspectiveTransform(np.float32([[info['detected_px']]]),inverse)[0,0]*scale+[0,header]
                            draw.line((tuple(nominal),tuple(detected)),fill='#ffb000',width=2)
                            dx,dy=detected
                            draw.ellipse((dx-2,dy-2,dx+2,dy+2),outline='#ffb000',width=1)
        if show_coordinates:
            circles = [(float(xy[0]), float(xy[1]), rr) for _, _, xy, rr in positions]
            circles += [(x, y+header, radius) for x, y, radius in anchors.obstacles(view_bgr,minimum_interval=.15)]
            placed = []
            badges = []
            for pos, ch, center, radius in positions:
                # A coordinate identifies an intersection, not a class. Keep
                # its appearance independent of momentary class confidence.
                badge = piece_badge(pos, "#182230", fonts[2])
                box = anchors.box(pos, center, radius, badge.size, circles, placed,
                                  (2, header+2, view_w-2, header+view_h-2))
                if box is not None:
                    placed.append(box)
                    start,end = coordinate_leader(center,radius,box)
                    # A halo keeps the one-to-one pointer visible on both
                    # black grid lines and bright piece faces. Stop at rim.
                    draw.line((start,end),fill="#ffffff",width=4)
                    draw.line((start,end),fill="#007c91",width=2)
                    cx,cy = center
                    draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline="#007c91",width=1)
                    draw.ellipse((end[0]-2,end[1]-2,end[0]+2,end[1]+2),fill="#007c91")
                    badges.append((badge,box))
                    if diagnostics is not None:
                        diagnostics["boxes"][pos] = list(box)
                        diagnostics["leaders"][pos] = dict(start=list(start),end=list(end),
                                                           center=center.tolist(),radius=radius)
            for badge,box in badges:
                image.paste(badge,box[:2])
        red = sum(visible["grid_chars"][r][c].isupper() for r in range(10) for c in range(9)
                  if visible["grid_chars"][r][c] in NAMES and f"{chr(97+c)}{r}" not in unknown)
        black = sum(visible["grid_chars"][r][c].islower() for r in range(10) for c in range(9)
                    if visible["grid_chars"][r][c] in NAMES and f"{chr(97+c)}{r}" not in unknown)
        lines = [f"棋子 {red+black} 枚  红{red} / 黑{black}",
                 f"待确认位置 {len(unknown)} 处",
                 "全盘识别已确认" if active else
                 "画面稳定，识别仍有疑问" if unknown else "上次确认盘面，正在更新"]
        if unknown:
            lines += [" ".join(sorted(unknown)[:8])]
        warnings = visible.get("rule_warnings", [])
        if warnings:
            lines += [f"布局提醒 {len(warnings)} 处：",
                      " ".join(warnings[:9]), "不符合棋规也保留实际识别"]
        if visible.get("rejection"):
            text = visible["rejection"]
            lines += [text[i:i+18] for i in range(0, len(text), 18)]
        lines += [f"显示待复核 {len(held)} 格", "坐标引线 → 对应棋子外圈", "灰字●：上次棋名，待复核",
                  "L：隐藏 / 显示原图小坐标",
                  "F：全屏   S：保存原图和结果"]
        if show_offsets:
            offsets=visible.get('piece_offsets',{})
            lines=lines[:5]+[f"圆心有效 {sum(bool(v.get('valid')) for v in offsets.values())}/{len(offsets)}",
                '蓝十字:交点  黄点:实测圆心','毫米数值需物理标定确认','O：关闭圆心测量视图']
        for i, line in enumerate(lines):
            draw_text((panel_x+2, header+390+i*25), line, font=fonts[2], fill="#344054")
    else:
        anchors.clear()
        draw_text((panel_x+5, header+390), "等待新的稳定盘面", font=fonts[1], fill="#667085")
    title = ("画面稳定 · 全盘已确认" if active else
             "画面稳定 · 部分待确认" if visible is not None and unknown else
             "画面稳定 · 结果更新中" if visible is not None else
             "画面稳定 · 等待识别" if gate.scene_is_stable(time.monotonic()) else "视觉暂停 / 正在确认")
    color = "#70eca1" if active else "#ffd166"
    draw_text((12, 2), f"{title} | {gate.reason}", font=fonts[0], fill=color)
    size = camera.actual_size or ("?", "?")
    ms = (f'外观复核 {visible["verification_ms"]:.0f}ms' if visible and
          visible.get("source") == "appearance_verified" else
          f'识别 {visible["inference_ms"]:.0f}ms' if visible else "识别 --")
    age = f'{max(0., time.monotonic()-visible["captured_at"]):.1f}s' if visible else "--"
    draw_text((12, 33), f"采集 {size[0]}×{size[1]} {camera.capture_fps:.1f}fps"
              f" | 预览 {getattr(camera, 'preview_fps', 0):.1f}fps | {ms}"
              f" | 观测年龄 {age} | 手 {hand_count}", font=fonts[2], fill="#d5dce5")
    draw_text((12, height-footer+5),
              f"Q退出  R重建观察  L小坐标  F全屏  S保存  | {event or '仅视觉观察，未连接机械臂'}",
              font=fonts[2], fill="#d5dce5")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def main():
    # Bound OpenCV's default (20 on this device) alongside ORT and MediaPipe.
    # Four had the best P95 in the local CV workload check; leave scheduling
    # capacity for the independent hand/camera workers.
    cv2.setNumThreads(min(4, os.cpu_count() or 1))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--hand-model", type=Path, required=True,
                        help="MediaPipe hand_landmarker.task (no automatic download/fallback)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path.home()/".codex/cache/tasks/chessrobot-live"/datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--max-seconds", type=float, default=0, help="Optional bounded observation run")
    parser.add_argument('--calibration',type=Path,help='Optional measured geometry JSON; no automatic calibration claim')
    args = parser.parse_args()
    if args.width < 320 or args.height < 240:
        parser.error("Capture resolution too small")
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.is_relative_to(ROOT):
        parser.error("Output directory must be outside the project")
    fonts = (load_font(22), load_font(17), load_font(14))
    anchors = PreviewAnchors()
    calibration=VisionCalibration.load(args.calibration) if args.calibration else VisionCalibration.from_config()
    measurer=CenterMeasurer(calibration)
    image_geometry=ImageGeometry(calibration)
    hands = HandCheck(args.hand_model)
    camera = None
    pool = None
    hand_worker = None
    try:
        recognizer = ChessRecognizer()   # One canonical profile, no model selector.
        if recognizer.pose_session is None or recognizer.classifier_session is None:
            raise RuntimeError("Required ONNX model missing; stopping")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Recovery can require six CPU forwards. A bounded confirmation
        # window must permit two such observations without extending camera
        # staleness or accepting results from an invalidated motion epoch.
        gate, motion = LiveGate(max_age=2.0, confirmation_age=4.0, display_max_age=4.0), MotionCheck()
        geometry = GeometryStabilizer()
        tracker = AppearanceTracker()
        preview_labels = PreviewLabels()
        preview_state = {}
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision-worker")
        future = token = None
        inference_packet = None
        inference_timed_out = False
        last_verify_at = -1.0
        hand_confirmed_at = -1.0
        last_hand_frame = -1
        hand_record = None
        hand_latencies = deque(maxlen=120)
        camera = LatestCamera(args.camera, args.width, args.height).start()
        hand_worker = LatestHands(camera, hands).start()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1600, 850)
        last_gate_frame = -1
        last_infer_at = -1.0
        last_status_at = -1.0
        snapshot_busy_count = 0
        display_times = deque(maxlen=60)
        last_display_frame = -1
        last_display_state = None
        display = None
        hand_count = 0
        show_offsets = False
        show_coordinates = True
        fullscreen = False
        previous_grid = None
        event = "仅视觉观察，无机械臂连接"
        last_logged_state = None
        start = time.monotonic()
        with (args.output_dir/"events.jsonl").open("a", encoding="utf-8") as log:
            def emit(kind, **data):
                log.write(json.dumps({"time": datetime.now().isoformat(), "kind": kind, **data},
                                     ensure_ascii=False)+"\n")
                log.flush()
            emit("start", camera=args.camera, requested=[args.width, args.height],
                 classifier=recognizer.classifier_model_path, pose=recognizer.pose_model_path,
                 hand_model=str(args.hand_model.resolve()), robot_connected=False,
                 opencv_threads=cv2.getNumThreads())
            while not args.max_seconds or time.monotonic()-start < args.max_seconds:
                now = time.monotonic()
                packet = camera.read()
                if packet is not None:
                    from localization.live_vision import CameraFrame
                    packet=CameraFrame(packet.frame_id,packet.captured_at,image_geometry.apply(packet.image))
                new_camera_frame = False
                if hand_worker.error:
                    gate.invalidate("手部检测异常")
                    raise RuntimeError(hand_worker.error)
                hand_record = hand_worker.read()
                if packet is not None:hand_record=image_geometry.hands(hand_record,packet.image.shape)
                if hand_record is not None:
                    hand_count, hand_confirmed_at = hand_record.count, hand_record.captured_at
                    if hand_record.frame_id != last_hand_frame:
                        hand_latencies.append((hand_record.completed_at-hand_record.captured_at)*1000)
                        last_hand_frame = hand_record.frame_id
                else:
                    hand_count, hand_confirmed_at = 0, -1.0
                if packet is None:
                    gate.invalidate(camera.error or "等待相机新帧")
                    frame = np.zeros((720, 1280, 3), np.uint8)
                    if now-start > 5 and camera.error:
                        raise RuntimeError(camera.error)
                else:
                    frame = packet.image
                    if packet.frame_id != last_gate_frame:
                        new_camera_frame = True
                        board_polygon = tracker.board_polygon(motion.transform) if tracker.ready(gate.epoch) else None
                        reason = motion.update(frame, now, board_polygon)
                        hand_count = board_hand_count(hand_record,board_polygon,frame.shape)
                        if hand_count:
                            reason = "检测到棋盘内手部，停止识别输出"
                        elif now-hand_confirmed_at > 0.50:
                            reason = "等待新的手部遮挡检查"
                        gate.observe(packet, time.monotonic(), reason)
                        last_gate_frame = packet.frame_id
                    elif now-packet.captured_at > 0.60:
                        gate.invalidate("相机画面过期")
                now = time.monotonic()
                if tracker.epoch is not None and tracker.epoch != gate.epoch:
                    tracker.clear()
                if (future is not None and not future.done() and not inference_timed_out
                        and now-inference_packet.captured_at > 5):
                    gate.invalidate("分类器处理超时，暂停确认")
                    tracker.clear()
                    inference_timed_out = True
                    emit("inference_timeout", frame_id=inference_packet.frame_id)
                # Verify the actual current frame before arbitrating an older
                # neural result; a busy UI must not make fresh pixel evidence
                # appear absent solely because of callback ordering.
                if (new_camera_frame and tracker.ready(gate.epoch) and gate.scene_is_stable(now)
                        and now-last_verify_at >= .15):
                    try:
                        if not gate.accept_tracking(tracker, packet, motion.transform.copy(), now):
                            if gate.state == "OCCLUDED_OR_MOVING":
                                emit("appearance_rejected", check=gate.last_appearance_check,
                                     reason=gate.reason)
                                cv2.imwrite(str(args.output_dir/f"appearance_rejected_{packet.frame_id}_raw.jpg"),
                                            packet.image)
                    except Exception as exc:
                        gate.invalidate("外观验证异常，暂停输出")
                        tracker.clear()
                        emit("verification_error", message=str(exc))
                    last_verify_at = time.monotonic()
                now = time.monotonic()
                if future is not None and future.done():
                    try:
                        result = future.result()
                        if tracker.ready(gate.epoch):
                            before_epoch = gate.epoch
                            accepted = gate.observe_neural_while_tracking(token, result, now, tracker)
                            if gate.epoch != before_epoch:
                                emit("neural_tracking_rejected", observation=result, reason=gate.reason)
                                cv2.imwrite(str(args.output_dir/f"neural_rejected_{inference_packet.frame_id}_raw.jpg"),
                                            inference_packet.image)
                        else:
                            accepted = gate.accept(token, result, now)
                        if gate.display_output is not None:
                            gate.display_output["rule_warnings"] = position_warnings(
                                gate.display_output["grid_chars"], recognizer)
                        if accepted:
                            if (not result["rejection"] and not result.get("raw_unknown")
                                    and result["grid_chars"] == gate.output["grid_chars"]):
                                tracker.seed(gate.output, inference_packet, gate.epoch, result)
                                reference_prefix = f"reference_{gate.epoch}_{inference_packet.frame_id}"
                                cv2.imwrite(str(args.output_dir/f"{reference_prefix}_raw.png"),
                                            inference_packet.image)
                                cv2.imwrite(str(args.output_dir/f"{reference_prefix}_canonical.png"),
                                            tracker.appearance.reference)
                                emit("appearance_reference", epoch=gate.epoch,
                                     frame_id=inference_packet.frame_id, prefix=reference_prefix,
                                     observation=gate.output)
                            result = gate.output
                            change = describe_change(previous_grid, result["grid_chars"])
                            if change:
                                event = change
                                emit("stable_board", **result, change=change)
                            previous_grid = [row[:] for row in result["grid_chars"]]
                    except Exception as exc:
                        gate.invalidate("推理异常，暂停输出")
                        tracker.clear()
                        emit("inference_error", message=str(exc))
                    future = token = None
                    inference_packet = None
                interval = 1.0 if tracker.ready(gate.epoch) else .20
                if packet is not None and future is None and gate.can_infer(now) and now-last_infer_at >= interval:
                    token = (gate.epoch, packet.frame_id)
                    inference_packet = packet
                    inference_timed_out = False
                    verified_geometry = None
                    if tracker.ready(gate.epoch):
                        current_matrix = tracker.transport_matrix @ tracker.transport_view @ np.linalg.inv(motion.transform)
                        verified_geometry = cv2.perspectiveTransform(
                            np.float32([[[50,50],[400,50],[50,450],[400,450]]]),
                            np.linalg.inv(current_matrix))[0]
                    future = pool.submit(infer, recognizer, packet, False, motion.transform.copy(),
                                         geometry, gate.epoch, not tracker.ready(gate.epoch), verified_geometry)
                    last_infer_at = now
                if (show_offsets and packet is not None and gate.output is not None
                        and gate.output['frame_id']==packet.frame_id):
                    measured=measurer.measure(cv2.warpPerspective(packet.image,np.asarray(gate.output['matrix']),(450,500)),
                        gate.output['grid_chars'],frame_id=packet.frame_id,captured_at=packet.captured_at)
                    for info in measured.values():info['control_valid']=False
                    gate.output['piece_offsets']=measured
                    gate.output['offsets_valid']=all(info['valid'] for info in measured.values())
                current_id = packet.frame_id if packet else -1
                current_state = (gate.state, gate.reason)
                if display is None or current_id != last_display_frame or current_state != last_display_state:
                    display = render(frame, gate, camera, fonts, event, show_offsets, hand_count,
                                     motion.transform, anchors, show_coordinates,
                                     preview_labels, preview_state)
                    cv2.imshow(WINDOW, display)
                    if current_id != last_display_frame:
                        display_times.append(time.monotonic())
                        if len(display_times) > 1:
                            camera.preview_fps = ((len(display_times)-1) /
                                                  (display_times[-1]-display_times[0]))
                    last_display_frame, last_display_state = current_id, current_state
                if now-last_status_at >= 0.5:
                    status = {"state": gate.state, "reason": gate.reason, "active": gate.output is not None,
                              "frame_id": packet.frame_id if packet else None,
                              "actual_resolution": camera.actual_size, "capture_fps": camera.capture_fps,
                              "preview_fps": getattr(camera, "preview_fps", 0),
                              "motion_metrics": motion.metrics,
                              "hands": hand_count, "event": event,
                              "hands_total": hand_record.count if hand_record else None,
                              "scene_stable": gate.scene_is_stable(now),
                              "display_result": gate.display_output,
                              "latest_candidate": gate.last_result,
                              "result": gate.output, "updated_at": time.time()}
                    status["appearance_reference_frame"] = (tracker.anchor["frame_id"]
                                                             if tracker.ready(gate.epoch) else None)
                    status["appearance_check"] = gate.last_appearance_check
                    status["appearance_publish_count"] = gate.appearance_publish_count
                    status["pending_neural_conflict"] = gate.neural_conflict
                    status["hand_result_age_ms"] = (now-hand_confirmed_at)*1000
                    status["hand_inference_ms"] = hand_record.inference_ms if hand_record else None
                    status["hand_frame_id"] = hand_record.frame_id if hand_record else None
                    status["hand_latency_p95_ms"] = (float(np.percentile(hand_latencies, 95))
                                                      if hand_latencies else None)
                    status["preview"] = preview_state
                    status["snapshot_busy_count"] = snapshot_busy_count
                    # Temporary file + replace prevents readers seeing half a JSON.
                    temporary = args.output_dir/"status.pending"
                    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
                    if not publish_snapshot(temporary, args.output_dir/"status.json"):
                        snapshot_busy_count += 1
                    for name, pixels in (("monitor_latest", display), ("monitor_raw_latest", frame)):
                        pending_image = args.output_dir/f"{name}.pending.jpg"
                        if not cv2.imwrite(str(pending_image), pixels):
                            raise RuntimeError(f"Cannot encode diagnostic image: {pending_image}")
                        if not publish_snapshot(pending_image, args.output_dir/f"{name}.jpg"):
                            snapshot_busy_count += 1
                    last_status_at = now
                state_key = (gate.state, gate.reason)
                if state_key != last_logged_state:
                    emit("state", state=gate.state, reason=gate.reason,
                         hand_result_age_ms=(time.monotonic()-hand_confirmed_at)*1000,
                         hands=hand_count)
                    last_logged_state = state_key
                key = cv2.waitKey(1) & 255
                if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("r"):
                    gate.invalidate("手动重建观察基准")
                    tracker.clear()
                    recognizer.reset_buffer()
                    anchors.clear()
                    preview_labels.clear()
                    previous_grid = None
                    event = "观察基准已重置"
                if key == ord("o"):
                    show_offsets = not show_offsets
                if key == ord("l"):
                    show_coordinates = not show_coordinates
                    last_display_state = None
                if key == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                    if not fullscreen:
                        cv2.resizeWindow(WINDOW, 1600, 850)
                if key == ord("s") and packet is not None:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cv2.imwrite(str(args.output_dir/f"{stamp}_raw.png"), frame)
                    cv2.imwrite(str(args.output_dir/f"{stamp}_overlay.png"), display)
                    (args.output_dir/f"{stamp}.json").write_text(
                        json.dumps({"state": gate.state, "result": gate.output,
                                    "display_result": gate.display_output,
                                    "preview": preview_state}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
                    emit("saved", prefix=stamp)
            emit("stop")
    finally:
        if hand_worker is not None:
            hand_worker.close()
        else:
            hands.close()
        if camera is not None:
            camera.close()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
