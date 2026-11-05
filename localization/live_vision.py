"""Camera-only live observation primitives. This module never imports hardware."""
from dataclasses import dataclass
from collections import Counter, deque
from pathlib import Path
from functools import lru_cache
import threading
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    frame_id: int
    captured_at: float                 # monotonic, successful read time
    image: np.ndarray


class LatestCamera:
    """Drain the camera continuously; inference never queues old frames."""
    def __init__(self, index=0, width=1920, height=1080):
        self.index, self.width, self.height = index, width, height
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._packet = None
        self.error = None
        self.actual_size = None
        self.capture_fps = 0.0
        self.thread = threading.Thread(target=self._read, name="vision-camera", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _read(self):
        cap = None
        try:
            backend = cv2.CAP_DSHOW if __import__("os").name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.index, backend)
            if not cap.isOpened():
                raise RuntimeError(f"Camera {self.index} cannot open")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # AX/DirectShow resets the media subtype when FPS is set. Set MJPG
            # LAST: measured 1080p YUY2 ~5fps versus MJPG ~29fps on this device.
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            seq, failures = 0, 0
            times = deque(maxlen=30)
            while not self._stop.is_set():
                ok, image = cap.read()
                if not ok or image is None:
                    failures += 1
                    with self._lock:
                        self._packet = None
                        self.error = "Camera read failed"
                    if failures >= 3:
                        break
                    continue
                failures = 0
                seq += 1
                now = time.monotonic()
                times.append(now)
                with self._lock:
                    self.actual_size = (image.shape[1], image.shape[0])
                    self.capture_fps = ((len(times)-1)/(times[-1]-times[0])
                                        if len(times) > 1 and times[-1] > times[0] else 0)
                    self._packet = CameraFrame(seq, now, image)
                    self.error = None
        except Exception as exc:
            with self._lock:
                self.error = str(exc)
                self._packet = None
        finally:
            if cap is not None:
                cap.release()

    def read(self):
        with self._lock:
            return self._packet

    def close(self):
        self._stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=3)


def geometry_reason(corners, scores, image_shape):
    """Validate all four outer intersections and a surrounding piece margin."""
    p = np.asarray(corners, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    if p.shape != (4, 2) or s.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(s).all():
        return "角点数据异常"
    if float(s.min()) < 0.10:
        return "角点置信度不足"
    polygon = p[[0, 1, 3, 2]]
    h, w = image_shape[:2]
    if not cv2.isContourConvex(polygon) or cv2.contourArea(polygon, oriented=True) <= 0:
        return "角点顺序或几何异常"
    if cv2.contourArea(polygon) < w*h*0.06:
        return "棋盘太小，请靠近"
    dst = np.float32([[50, 50], [400, 50], [50, 450], [400, 450]])
    inverse = cv2.getPerspectiveTransform(dst, p)
    outer = cv2.perspectiveTransform(
        np.float32([[[25, 25], [425, 25], [425, 475], [25, 475]]]), inverse)[0]
    if not np.isfinite(outer).all() or (outer[:, 0] < 3).any() or (outer[:, 0] > w-4).any() or (
            outer[:, 1] < 3).any() or (outer[:, 1] > h-4).any():
        return "棋盘不完整，请让四边及棋子完整入镜"
    return ""


def grid_reason(grid, confidence, recognizer, minimum=0.50):
    g = np.asarray(grid)
    p = np.asarray(confidence, dtype=float)
    if g.shape != (10, 9) or p.shape != (10, 9):
        return "盘面尺寸异常"
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        return "分类概率异常"
    if any(ch not in ".xKABNRCPkabnrcp" for ch in g.flat):
        return "未知类别"
    if "x" in g:
        return "遮挡或未知格"
    thresholds = np.where(g == ".", min(minimum, .45), minimum)
    if (p < thresholds).any():
        return "存在低置信度位置，暂不确认"
    counts = Counter(g.flat)
    limits = {"K": 1, "A": 2, "B": 2, "N": 2, "R": 2, "C": 2, "P": 5}
    for ch, maximum in list(limits.items()):
        limits[ch.lower()] = maximum
    if any(counts[ch] > maximum for ch, maximum in limits.items()):
        return "棋子类别数量冲突，等待重识别"
    return ""


def position_warnings(grid, recognizer):
    """Advisory game-rule information; never a visual label or motion veto."""
    return [f"{chr(97+c)}{r}" for r in range(10) for c in range(9)
            if grid[r][c] not in (".", "x") and grid[r][c] not in
            recognizer.get_allowed_classes(grid[r][c].isupper(), r, c)]


@lru_cache(maxsize=4)
def _exposure_lut(stops):
    values = np.arange(256, dtype=np.float32)/255
    linear = np.where(values <= .04045, values/12.92, ((values+.055)/1.055)**2.4)
    linear = np.clip(linear*2**stops, 0, 1)
    encoded = np.where(linear <= .0031308, linear*12.92, 1.055*linear**(1/2.4)-.055)
    return np.clip(np.round(encoded*255), 0, 255).astype(np.uint8)


def exposure_view(image, stops):
    return cv2.LUT(image, _exposure_lut(stops))


class GeometryStabilizer:
    """Filter pose quantization in a common tracked view; reset after motion."""
    def __init__(self):
        self.epoch = None
        self.history = deque(maxlen=5)
        self.filtered = None

    def update(self, corners, view_transform, epoch):
        transform = np.eye(3) if view_transform is None else np.asarray(view_transform)
        common = cv2.perspectiveTransform(np.float32([corners]), np.linalg.inv(transform))[0]
        if epoch != self.epoch:
            self.history.clear()
            self.filtered = None
            self.epoch = epoch
        if self.filtered is not None:
            edge = min(np.linalg.norm(common[1]-common[0]), np.linalg.norm(common[2]-common[0]))
            if np.max(np.linalg.norm(common-self.filtered, axis=1)) > edge*.06:
                self.history.clear()
                self.filtered = None
        self.history.append(common)
        median = np.median(self.history, axis=0)
        self.filtered = median if self.filtered is None else self.filtered*.7+median*.3
        return cv2.perspectiveTransform(np.float32([self.filtered]), transform)[0]


def grid_edge_corners(warped):
    """Small pose correction from four long grid boundaries, never piece labels.

    Work in the canonical view. Require repeated, nearly parallel evidence for
    every boundary; otherwise abstain. Paper edges outside the grid corridor
    cannot supply a correction.
    """
    edges = cv2.Canny(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), 45, 130)
    segments = cv2.HoughLinesP(edges, 1, np.pi/720, 50, minLineLength=160, maxLineGap=30)
    if segments is None:
        return None
    groups = [[], [], [], []]
    for x1, y1, x2, y2 in segments[:, 0]:
        dx, dy = float(x2-x1), float(y2-y1)
        if abs(dy) >= 160 and abs(dx) < abs(dy)*.12:
            slope = dx/dy
            intercept = x1-slope*y1
            for index, target in ((0, 50), (1, 400)):
                if abs(slope*250+intercept-target) < 35:
                    groups[index].append((slope, intercept, abs(dy)))
        if abs(dx) >= 160 and abs(dy) < abs(dx)*.12:
            slope = dy/dx
            intercept = y1-slope*x1
            for index, target in ((2, 50), (3, 450)):
                if abs(slope*225+intercept-target) < 30:
                    groups[index].append((slope, intercept, abs(dx)))
    fits = []
    for index, group in enumerate(groups):
        if len(group) < 2:
            return None
        values = np.asarray(group)
        middle = values[:, 0]*(250 if index < 2 else 225)+values[:, 1]
        outer = middle.min() if index in (0, 2) else middle.max()
        values = values[np.abs(middle-outer) < 8]
        if len(values) < 2 or values[:, 2].max() < 240:
            return None
        fits.append(np.average(values[:, :2], axis=0, weights=values[:, 2]))
    corners = []
    for vertical, horizontal in ((fits[0], fits[2]), (fits[1], fits[2]),
                                 (fits[0], fits[3]), (fits[1], fits[3])):
        y = (horizontal[0]*vertical[1]+horizontal[1])/(1-horizontal[0]*vertical[0])
        corners.append((vertical[0]*y+vertical[1], y))
    corners = np.float32(corners)
    nominal = np.float32([[50, 50], [400, 50], [50, 450], [400, 450]])
    if np.max(np.linalg.norm(corners-nominal, axis=1)) > 40:
        return None
    return corners


def guarded_recovery(views):
    """Preserve strong primary evidence; require three votes for weak cells."""
    grid = [row[:] for row in views[0]["grid_chars"]]
    confidence = [row[:] for row in views[0]["conf_grid"]]
    counts = Counter(ch for row in grid for ch in row)
    limits = dict(zip("KABNRCPkabnrcp", (1, 2, 2, 2, 2, 2, 5)*2))
    for r in range(10):
        for c in range(9):
            # When the primary result is globally inconsistent, three strong
            # boundary-aligned views can resolve a shared pose error that all
            # three photometric views retained. Counts trigger verification;
            # the actual alternative identity still comes from image evidence.
            if counts[grid[r][c]] > limits.get(grid[r][c], 90) and len(views) in (4, 6):
                evidence = [view["grid_chars"][r][c] for view in views[-3:]]
                probabilities = [view["conf_grid"][r][c] for view in views[-3:]]
                if (len(set(evidence)) == 1 and evidence[0] not in ("x", ".")
                        and min(probabilities) >= .65):
                    grid[r][c], confidence[r][c] = evidence[0], min(probabilities)
                    continue
            if grid[r][c] == "x" or confidence[r][c] >= .90:
                continue
            support = {}
            for view in views:
                ch, probability = view["grid_chars"][r][c], view["conf_grid"][r][c]
                if ch != "x" and probability >= (.45 if ch == "." else .50):
                    support.setdefault(ch, []).append(probability)
            candidates = sorted(((len(probs), ch, sorted(probs, reverse=True)[2])
                                 for ch, probs in support.items() if len(probs) >= 3), reverse=True)
            if candidates and (len(candidates) == 1 or candidates[0][0] > candidates[1][0]):
                _, grid[r][c], confidence[r][c] = candidates[0]
            else:
                grid[r][c], confidence[r][c] = "x", 0.0
    return grid, confidence


def _single_board(recognizer, image, warp_preprocess=None, geometry=None,
                  view_transform=None, epoch=None, fixed_geometry=None):
    corners, scores = (recognizer.predict_corners(image) if fixed_geometry is None else fixed_geometry)
    reason = geometry_reason(corners, scores, image.shape)
    if not reason and geometry is not None:
        corners = geometry.update(corners, view_transform, epoch)
        reason = geometry_reason(corners, scores, image.shape)
    result = {"corners": np.asarray(corners).tolist(),
              "corner_scores": np.asarray(scores).tolist(), "rejection": reason,
              "grid_chars": [["x"]*9 for _ in range(10)],
              "conf_grid": [[0.0]*9 for _ in range(10)], "matrix": None}
    if reason:
        return result
    warped, matrix = recognizer.warp_chessboard(image, corners)
    if warp_preprocess is not None:
        warped = warp_preprocess(warped)
    grid, confidence, _ = recognizer.classify_board(warped)
    result.update(grid_chars=grid, conf_grid=confidence, matrix=matrix.tolist(),
                  raw_unknown=[[r, c] for r in range(10) for c in range(9) if grid[r][c] == "x"],
                  rule_warnings=position_warnings(grid, recognizer),
                  refinement_edits=list(getattr(recognizer, "last_refinement_edits", [])),
                  rejection=grid_reason(grid, confidence, recognizer))
    return result


def consensus_grid(views, minimum=0.50):
    """Two strong votes are required per cell; never renormalize confidence."""
    grid, confidence = [], []
    for r in range(10):
        row, probabilities = [], []
        for c in range(9):
            support = {}
            for view in views:
                ch, probability = view["grid_chars"][r][c], float(view["conf_grid"][r][c])
                threshold = min(minimum, .45) if ch == "." else minimum
                if ch != "x" and np.isfinite(probability) and threshold <= probability <= 1:
                    support.setdefault(ch, []).append(probability)
            winners = [(ch, sorted(probs, reverse=True)[1])
                       for ch, probs in support.items() if len(probs) >= 2]
            if len(winners) == 1:
                ch, probability = winners[0]
            else:
                ch, probability = "x", 0.0
            row.append(ch)
            probabilities.append(probability)
        grid.append(row)
        confidence.append(probabilities)
    return grid, confidence


def observe_board(recognizer, image, allow_recovery=True, warp_preprocess=None,
                  geometry=None, view_transform=None, epoch=None, verified_geometry=None):
    """The shared live/offline profile. Recovery is not a second physical frame.

    Use only the same Nano weights. Photometric votes may help an ambiguous
    observation, but they cannot bypass the live gate's independent-frame vote
    or its freshness/hand/geometry checks.
    """
    start = time.perf_counter()
    if verified_geometry is None:
        base = _single_board(recognizer, image, warp_preprocess, geometry, view_transform, epoch)
    else:
        predicted, scores = recognizer.predict_corners(image)
        reason = geometry_reason(predicted, scores, image.shape)
        tracked = np.asarray(verified_geometry, dtype=np.float32)
        if not reason:
            reason = geometry_reason(tracked, scores, image.shape)
        if not reason:
            edge = min(np.linalg.norm(tracked[1]-tracked[0]), np.linalg.norm(tracked[2]-tracked[0]))
            if np.max(np.linalg.norm(np.asarray(predicted)-tracked, axis=1)) > edge*.08:
                reason = "自动角点与已验证几何严重不一致"
        if reason:
            base = dict(corners=np.asarray(predicted).tolist(), corner_scores=np.asarray(scores).tolist(),
                        rejection=reason, matrix=None, grid_chars=[["x"]*9 for _ in range(10)],
                        conf_grid=[[0.0]*9 for _ in range(10)], raw_unknown=[])
        else:
            base = _single_board(recognizer, image, warp_preprocess,
                                 fixed_geometry=(tracked, scores))
        base["geometry_source"] = "verified_tracking_with_fresh_pose_check"
        base["fresh_pose_corners"] = np.asarray(predicted).tolist()
    base["recovery"] = {"attempted": False, "used": False, "views": 1}
    if (allow_recovery and base["rejection"] and base["matrix"] is not None
            and not any(ch == "x" for row in base["grid_chars"] for ch in row)):
        geometry_views = []
        warped, matrix = recognizer.warp_chessboard(image, np.float32(base["corners"]))
        corrected = grid_edge_corners(warped)
        if corrected is not None:
            native = cv2.perspectiveTransform(corrected[None], np.linalg.inv(matrix))[0]
            if not geometry_reason(native, base["corner_scores"], image.shape):
                aligned, _ = recognizer.warp_chessboard(image, native)
                if warp_preprocess is not None:
                    aligned = warp_preprocess(aligned)
                for dx in (-3, 0, 3):
                    shifted = cv2.warpAffine(aligned, np.float32([[1, 0, dx], [0, 1, 0]]),
                                             (450, 500), borderMode=cv2.BORDER_REFLECT)
                    g, p, _ = recognizer.classify_board(shifted)
                    geometry_views.append({"grid_chars": g, "conf_grid": p})
                grid, confidence = guarded_recovery([base]+geometry_views)
                if not grid_reason(grid, confidence, recognizer):
                    base.update(grid_chars=grid, conf_grid=confidence, rejection="",
                                rule_warnings=position_warnings(grid, recognizer))
                    base["recovery"] = dict(attempted=True, used=True, views=4, geometry_evidence=True)
                    base["inference_ms"] = (time.perf_counter()-start)*1000
                    return base
        base["recovery"] = {"attempted": True, "used": False, "views": 3}
        views = [base]
        origin = np.asarray(base["corners"])
        edge = min(np.linalg.norm(origin[1]-origin[0]), np.linalg.norm(origin[2]-origin[0]))
        for stops in (-.5, .5):
            fixed = (np.asarray(base["corners"]), np.asarray(base["corner_scores"])) if geometry else None
            alternative = _single_board(recognizer, exposure_view(image, stops), warp_preprocess,
                                        fixed_geometry=fixed)
            if alternative["matrix"] is None or np.max(np.linalg.norm(
                    np.asarray(alternative["corners"])-origin, axis=1)) > edge*.06:
                break
            views.append(alternative)
        if len(views) == 3:
            grid, confidence = consensus_grid(views)
            if not grid_reason(grid, confidence, recognizer):
                base.update(grid_chars=grid, conf_grid=confidence, rejection="")
                base["recovery"]["used"] = True
                base["rule_warnings"] = position_warnings(grid, recognizer)
            elif geometry_views:
                grid, confidence = guarded_recovery(views+geometry_views)
                base.update(grid_chars=grid, conf_grid=confidence,
                            rejection=grid_reason(grid, confidence, recognizer),
                            rule_warnings=position_warnings(grid, recognizer))
                base["recovery"].update(used=True, views=6, geometry_evidence=True)
    base["inference_ms"] = (time.perf_counter()-start)*1000
    return base


class MotionCheck:
    """Register small camera shake, then compare exposure-normalized residuals."""
    def __init__(self):
        self.previous = None
        self.previous_color = None
        self.transform = np.eye(3)
        self.metrics = {}

    def update(self, frame, now, board_polygon=None):
        height, width = frame.shape[:2]
        size = (320, max(1, round(height*320/width)))
        color = cv2.GaussianBlur(cv2.resize(frame, size), (5,5), 0)
        old_color = self.previous_color
        self.previous_color = color
        gray = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if gray.mean() < 12 or gray.mean() > 245:
            self.previous = gray
            return "曝光过暗或过亮"
        old = self.previous
        self.previous = gray
        if old is None or old.shape != gray.shape:
            return ""
        region = None
        if board_polygon is not None:
            region = np.zeros_like(gray)
            polygon = np.asarray(board_polygon, dtype=np.float32)*[size[0]/width,size[1]/height]
            cv2.fillConvexPoly(region, np.round(polygon).astype(np.int32), 255)
            if cv2.countNonZero(region) < 100:
                return "可见棋盘区域不足"
        points = cv2.goodFeaturesToTrack(old, 120, .02, 7, mask=region)
        if points is None or len(points) < 8:
            return "画面纹理不足，等待清晰棋盘"
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            old, gray, points, None, winSize=(21, 21), maxLevel=2)
        if tracked is None or status is None:
            return "画面跟踪失败"
        valid = status.ravel().astype(bool)
        if valid.sum() < 8:
            return "画面跟踪失败"
        affine, inliers = cv2.estimateAffinePartial2D(
            points[valid], tracked[valid], method=cv2.RANSAC, ransacReprojThreshold=1.2)
        if affine is None or inliers is None or inliers.mean() < .55:
            return "相机或棋子正在明显移动"
        step = np.vstack([affine, [0., 0., 1.]])
        pixels = np.diag([width/size[0], height/size[1], 1.])
        self.transform = pixels @ step @ np.linalg.inv(pixels) @ self.transform
        shift = float(np.linalg.norm(affine[:, 2]))
        scale = float(np.hypot(affine[0, 0], affine[0, 1]))
        angle = abs(float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0]))))
        if shift > 7 or abs(scale-1) > .05 or angle > 3:
            return "相机正在明显移动"
        aligned = cv2.warpAffine(old, affine, size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(np.ones_like(old), affine, size, flags=cv2.INTER_NEAREST)
        mask = cv2.erode(mask, np.ones((7, 7), np.uint8)).astype(bool)
        if region is not None:
            mask &= region.astype(bool)
        aligned_color = cv2.warpAffine(old_color,affine,size,borderMode=cv2.BORDER_REFLECT)
        raw_difference = np.abs(aligned_color.astype(np.float32)-color.astype(np.float32)).mean(axis=2)[mask]
        if raw_difference.size >= 100 and raw_difference.mean() < 3 and np.mean(raw_difference > 12) < .005:
            self.metrics = dict(shift=shift,rotation=angle,gain=1.,offset=0.,
                                changed_fraction=0.,residual_mean=float(raw_difference.mean()),
                                illumination_gain_min=1.,illumination_gain_max=1.,
                                color_changed_fraction=0.,board_scoped=region is not None,
                                photometric_fast_path=True)
            return ""
        from localization.illumination import compensate_shading
        corrected, lighting = compensate_shading(aligned, gray)
        before, after = aligned[mask].astype(float), corrected[mask].astype(float)
        if before.size < 100:
            return "可跟踪区域不足"
        lo1, hi1 = np.percentile(before, [20, 80])
        lo2, hi2 = np.percentile(after, [20, 80])
        # Nearly uniform paper cannot estimate an exposure slope reliably.
        gain = float(np.clip((hi2-lo2)/(hi1-lo1), .7, 1.4)) if hi1-lo1 >= 10 else 1.0
        offset = float(np.median(after-gain*before))
        residual = np.abs(after-(gain*before+offset))
        changed = float(np.mean(residual > 24))
        old_rgb = aligned_color[mask].astype(float)+8
        new_rgb = color[mask].astype(float)+8
        color_delta = new_rgb/new_rgb.sum(axis=1)[:,None]-old_rgb/old_rgb.sum(axis=1)[:,None]
        color_delta -= np.median(color_delta,axis=0)
        color_changed = float(np.mean(np.abs(color_delta).sum(axis=1) > .20))
        self.metrics = {"shift": shift, "rotation": angle, "gain": gain,
                        "offset": offset, "changed_fraction": changed,
                        "residual_mean": float(residual.mean()), **lighting,
                        "color_changed_fraction": color_changed,
                        "board_scoped": region is not None}
        if (abs(offset) > 35 or gain < .75 or gain > 1.3
                or lighting['illumination_gain_min'] < .45 or lighting['illumination_gain_max'] > 1.8):
            return "光照发生明显突变"
        if changed > .06 or residual.mean() > 9 or color_changed > .06:
            return "检测到局部运动或遮挡"
        return ""


class LiveGate:
    """Single-threaded coordinator; worker results carry an invalidation epoch."""
    def __init__(self, settle_seconds=0.30, confirmations=2, max_age=1.0, confirmation_age=2.0,
                 display_max_age=None):
        self.settle_seconds = settle_seconds
        self.confirmations = confirmations
        self.max_age = max_age
        self.display_max_age = max_age if display_max_age is None else max(max_age, display_max_age)
        self.confirmation_age = max(max_age, confirmation_age)
        self.epoch = 0
        self.last_frame_id = -1
        self.quiet_since = None
        self.candidate = None
        self.votes = 0
        self.output = None
        self.display_output = None
        self.last_result = None
        self.state = "SETTLING"
        self.reason = "等待新画面"
        self.last_result_frame = -1
        self.result_history = deque(maxlen=3)
        self.last_appearance_check = None
        self.appearance_publish_count = 0
        self.neural_conflict = None

    def invalidate(self, reason):
        self.epoch += 1
        self.quiet_since = None
        self.output = self.candidate = None
        self.display_output = None
        self.last_result = None
        self.votes = 0
        self.result_history.clear()
        self.neural_conflict = None
        self.state = "OCCLUDED_OR_MOVING"
        self.reason = reason

    def observe(self, packet, now, reason=""):
        if packet is None or now-packet.captured_at > 0.60:
            self.invalidate("摄像头断流或画面过期")
            return False
        if packet.frame_id <= self.last_frame_id:
            return False
        self.last_frame_id = packet.frame_id
        if reason:
            self.invalidate(reason)
            return True
        if self.quiet_since is None:
            self.quiet_since = now
            self.state = "SETTLING"
            self.reason = "画面恢复，等待连续稳定结果"
        output_age = .35 if self.output and self.output.get("source") == "appearance_verified" else self.max_age
        if self.output and now-self.output["captured_at"] > output_age:
            self.output = None
            self.state = "CONFIRMED_STABLE" if self.display_output else "SETTLING"
            self.reason = "画面稳定，正在更新识别" if self.display_output else "等待最新识别"
        if self.display_output and now-self.display_output["captured_at"] > self.display_max_age:
            self.display_output = None
            self.state = "SETTLING"
            self.reason = "等待最新识别"
        return True

    def can_infer(self, now):
        return self.quiet_since is not None and now-self.quiet_since >= self.settle_seconds

    def scene_is_stable(self, now):
        """Physical quietness is independent of classifier certainty/freshness."""
        return self.can_infer(now)

    def observe_neural_while_tracking(self, token, result, now, tracker):
        """Keep diagnostics without replacing pixel-verified identities by noise."""
        epoch, frame_id = token
        if (epoch != self.epoch or not tracker.ready(epoch) or self.quiet_since is None
                or now-result["captured_at"] > self.max_age or frame_id <= self.last_result_frame):
            return False
        self.last_result_frame, self.last_result = frame_id, result
        must_reconfirm = result.get("matrix") is None or bool(result.get("raw_unknown"))
        if not must_reconfirm and not result["rejection"] and result["grid_chars"] != tracker.anchor["grid_chars"]:
            # A single neural hypothesis is not a confirmed board. Current
            # pixels are independent evidence supporting the existing identity.
            # Low-confidence swaps cannot overrule that evidence; repeated
            # strong alternatives still force conservative reacquisition.
            fresh_reference = (self.output is not None
                               and now-self.output["captured_at"] <= .35
                               and self.output["grid_chars"] == tracker.anchor["grid_chars"])
            changed = [(r,c) for r in range(10) for c in range(9)
                       if result["grid_chars"][r][c] != tracker.anchor["grid_chars"][r][c]]
            strong = all(result["conf_grid"][r][c] >= .80 for r,c in changed)
            if not fresh_reference:
                must_reconfirm = True
            elif strong:
                key = tuple(tuple(row) for row in result["grid_chars"])
                old = self.neural_conflict
                count = old["count"]+1 if (old and old["grid"] == key and
                                          now-old["at"] <= 3.0) else 1
                self.neural_conflict = dict(grid=key,count=count,at=now,frame_id=frame_id)
                must_reconfirm = count >= 2
            else:
                self.neural_conflict = None
        elif not must_reconfirm:
            self.neural_conflict = None
        if must_reconfirm:
            self.invalidate("新模型观测与已确认盘面冲突或存在原始遮挡")
            tracker.clear()
        return False

    def accept_tracking(self, tracker, packet, view_transform, now):
        """Publish only a different, fresh physical frame that passed matching."""
        if (not tracker.ready(self.epoch) or self.quiet_since is None
                or packet is None or packet.frame_id != self.last_frame_id):
            return False
        started = time.perf_counter()
        observation = tracker.verify(packet, view_transform, self.epoch, now)
        if tracker.reason != "重复相机帧":
            self.last_appearance_check = dict(frame_id=packet.frame_id,
                                             captured_at=packet.captured_at,
                                             valid=observation is not None,
                                             reason=tracker.reason, metrics=tracker.last_metrics,
                                             matrix=tracker.last_compare_matrix)
        if observation is None:
            if tracker.reason != "重复相机帧":
                self.invalidate(tracker.reason)
                tracker.clear()
            return False
        if now+time.perf_counter()-started-observation["captured_at"] > .35:
            self.last_appearance_check.update(valid=False, reason="外观复核结果过期")
            self.invalidate("外观复核结果过期")
            tracker.clear()
            return False
        self.output = self.display_output = observation
        self.appearance_publish_count += 1
        self.state = "CONFIRMED_STABLE"
        self.reason = "可信稳态，当前新画面外观复核通过"
        return True

    def accept(self, token, result, now):
        epoch, frame_id = token
        if epoch != self.epoch or self.quiet_since is None or now-result["captured_at"] > self.max_age:
            return False
        if frame_id <= self.last_result_frame:
            return False
        self.last_result_frame = frame_id
        self.last_result = result
        if "matrix" in result and result["matrix"] is None:
            self.invalidate(result["rejection"])
            return False
        grid = result["grid_chars"]
        if np.asarray(result.get("conf_grid")).shape != (10, 9):
            self.invalidate("缺少逐格置信度")
            return False
        while self.result_history and now-self.result_history[0][2] > self.confirmation_age:
            self.result_history.popleft()
        self.result_history.append((frame_id, grid, result["captured_at"], result))
        self.votes = len(self.result_history)
        if len(self.result_history) < self.confirmations:
            self.output = self.display_output = None
            self.state = "SETTLING"
            self.reason = f"稳定确认 {len(self.result_history)}/{self.confirmations}"
            return False

        # Physical stillness is already checked by observe(). Fuse evidence per
        # intersection, so one flickering classifier cell cannot reset 89 good
        # cells. This is a vote across distinct, fresh camera observations.
        fused_grid, fused_confidence = consensus_grid([row[3] for row in self.result_history])
        # Missing evidence from an augmentation disagreement is not a detected
        # occlusion. Fresh temporal votes may resolve it; original x never may.
        unknown = result.get("raw_unknown", [[r, c] for r in range(10) for c in range(9)
                                             if grid[r][c] == "x"])
        for r, c in unknown:
            fused_grid[r][c], fused_confidence[r][c] = "x", 0.0
        fused = dict(result)
        fused.update(
            grid_chars=fused_grid, conf_grid=fused_confidence,
            rejection=grid_reason(fused_grid, fused_confidence, None),
            temporal_evidence={"frame_ids": [row[0] for row in self.result_history],
                               "oldest_capture": self.result_history[0][2]})
        self.display_output = fused
        self.candidate = fused_grid
        self.votes = len(self.result_history)
        self.state = "CONFIRMED_STABLE"
        if fused["rejection"]:
            self.output = None
            self.reason = "画面稳定，部分识别待确认"
            return False
        self.output = fused
        self.reason = "可信稳态，持续更新"
        return True


class HandCheck:
    """Still hands remain blocking. Missing assets are startup errors."""
    def __init__(self, model_path):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Hand model missing: {model_path}")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
        self.mp = mp
        self.detector = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.45, min_hand_presence_confidence=0.45))
        self.last_timestamp = -1

    def detect(self, frame, timestamp):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (min(w, 640), max(1, int(h*min(w, 640)/w))))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        timestamp_ms = max(self.last_timestamp+1, int(timestamp*1000))
        result = self.detector.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        self.last_timestamp = timestamp_ms
        self.last_regions = tuple(tuple((float(point.x),float(point.y)) for point in hand)
                                  for hand in result.hand_landmarks)
        return len(result.hand_landmarks)

    def close(self):
        self.detector.close()


@dataclass(frozen=True)
class HandObservation:
    frame_id: int
    captured_at: float
    completed_at: float
    count: int
    inference_ms: float
    regions: tuple = None


def board_hand_count(observation, polygon, frame_shape):
    """Count hands intersecting the board; unknown geometry remains fail closed."""
    if observation is None:
        return 0
    if polygon is None or observation.regions is None:
        return observation.count
    board = cv2.convexHull(np.asarray(polygon, np.float32))
    height,width = frame_shape[:2]
    count = 0
    for points in observation.regions:
        points = np.asarray(points, np.float32)*[width,height]
        # Landmarks lie inside the skin silhouette; expand conservatively.
        center = points.mean(axis=0)
        hull = cv2.convexHull((center+(points-center)*1.25).astype(np.float32))
        area,_ = cv2.intersectConvexConvex(board,hull)
        count += area > 0
    return int(count)


class LatestHands:
    """Keep checking latest camera frames even when the preview thread is busy."""
    def __init__(self, camera, detector, minimum_interval=.08):
        self.camera, self.detector = camera, detector
        self.minimum_interval = minimum_interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._observation = None
        self._error = None
        self._closed = False
        self.thread = threading.Thread(target=self._run, name="vision-hands", daemon=True)

    def start(self):
        self.thread.start()
        return self

    @property
    def error(self):
        with self._lock:
            return self._error

    def read(self):
        with self._lock:
            return self._observation

    def _close_model(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.detector.close()

    def _run(self):
        last_frame, last_started = -1, -1.0
        try:
            while not self._stop.is_set():
                packet = self.camera.read()
                now = time.monotonic()
                if packet is None:
                    with self._lock:
                        self._observation = None
                elif packet.frame_id > last_frame and now-last_started >= self.minimum_interval:
                    last_frame, last_started = packet.frame_id, now
                    count = int(self.detector.detect(packet.image, packet.captured_at))
                    regions = getattr(self.detector, "last_regions", None)
                    if not isinstance(regions, tuple) or len(regions) != count:
                        regions = None
                    completed = time.monotonic()
                    observation = HandObservation(packet.frame_id, packet.captured_at, completed,
                                                  count, (completed-now)*1000, regions)
                    with self._lock:
                        self._observation = observation
                self._stop.wait(.005)
        except Exception as exc:
            with self._lock:
                self._observation = None
                self._error = str(exc)
        finally:
            try:
                self._close_model()
            except Exception as exc:
                with self._lock:
                    self._error = self._error or str(exc)

    def close(self):
        self._stop.set()
        if self.thread.ident is None:
            self._close_model()
        else:
            # The worker owns graph shutdown, even if an inference is delayed.
            # Never close MediaPipe concurrently from the UI thread.
            self.thread.join(timeout=2)
