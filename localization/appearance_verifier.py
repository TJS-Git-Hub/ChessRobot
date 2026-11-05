"""Fresh visual verification of an already classified board, without hardware.

This is exemplar matching, not a new neural classification or an offset sensor.
The reference is immutable for one observation epoch. A change requires a new
neural board confirmation; references never adapt toward an unconfirmed board.
"""
import copy
import time

import cv2
import numpy as np
from localization.illumination import compensate_shading


class BoardAppearance:
    def __init__(self, canonical):
        self.reference = canonical.copy()
        self.features = self.describe(canonical)

    @staticmethod
    def describe(image):
        if image.shape != (500, 450, 3) or image.dtype != np.uint8:
            raise ValueError("Expected uint8 canonical BGR image (500,450,3)")
        # Low-pass before local correlation: subpixel registration resamples
        # narrow printed lines; their interpolation phase is not a board move.
        smooth = cv2.GaussianBlur(image, (5, 5), 1.2).astype(np.float32)
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        tiles = gray.reshape(10, 50, 9, 50).transpose(0, 2, 1, 3).reshape(90, 2500)
        # The warp includes padding beyond the outer intersections. Only the
        # board plus a half-cell piece margin is evidence of board contents;
        # desk shadows outside it must not veto an unchanged position.
        support = np.zeros((500, 450), np.float32)
        support[25:475, 25:425] = 1
        weights = support.reshape(10,50,9,50).transpose(0,2,1,3).reshape(90,2500)
        count = weights.sum(axis=1)[:, None]
        mean = (tiles*weights).sum(axis=1)[:, None]/count
        centered = (tiles-mean)*weights
        std = np.sqrt((centered**2).sum(axis=1)/count[:,0])
        normalized = centered/np.maximum(std[:, None], 7)
        # Estimate the local white level sparsely; use full-resolution ink for
        # side changes. Per-channel white normalization tolerates white balance
        # but cannot turn red ink into black ink without leaving a residual.
        rgb = smooth.reshape(10, 50, 9, 50, 3).transpose(0, 2, 1, 3, 4)
        white = np.percentile(rgb[:, :, ::2, ::2].reshape(90, -1, 3), 85, axis=1).astype(np.float32)
        rgb = rgb.reshape(90, 2500, 3)/np.maximum(white[:, None], 30)
        chroma = np.log((rgb[:, :, 2]+.04)/(rgb[:, :, 1]+.04))
        chroma = chroma.reshape(10, 9, 50, 50).transpose(0, 2, 1, 3).reshape(500, 450)
        chroma = cv2.GaussianBlur(chroma.astype(np.float32), (5, 5), 0)
        chroma *= support
        return normalized, std, chroma

    def score(self, image):
        current, std, color = self.describe(image)
        reference, ref_std, ref_color = self.features
        norm = np.linalg.norm(reference, axis=1)*np.linalg.norm(current, axis=1)
        correlation = (reference*current).sum(axis=1)/np.maximum(norm, 1e-6)
        # Correlation is ill-conditioned in nearly blank paper and thin-line
        # tails. Keep the strict NCC test for textured regions; below that,
        # bounded absolute residuals still detect new/removed local content.
        textured = np.maximum(std, ref_std) > 30
        has_structure = np.count_nonzero(ref_std > 5) >= 10
        minimum = float(correlation[textured].min()) if textured.any() else 1.0
        worst = int(np.argmin(np.where(textured, correlation, 2)))
        reference_local = reference*np.maximum(ref_std, 7)[:, None]/np.maximum(ref_std, 25)[:, None]
        current_local = current*np.maximum(std, 7)[:, None]/np.maximum(std, 25)[:, None]
        residual = np.abs(reference_local-current_local)
        local = residual.reshape(10,9,50,50).transpose(0,2,1,3).reshape(500,450)
        metrics = dict(min_tile_correlation=minimum,
                       worst_tile=worst, worst_tile_std=float(max(std[worst], ref_std[worst])),
                       max_tile_residual=float(residual.mean(axis=1).max()),
                       max_local_residual=float(cv2.GaussianBlur(local,(5,5),0).max()),
                       max_ink_color_residual=float(np.abs(ref_color-color).max()))
        valid = (has_structure and minimum >= .97 and metrics["max_tile_residual"] <= .30
                 and metrics["max_local_residual"] <= 1.5
                 and metrics["max_ink_color_residual"] <= .45)
        return valid, metrics

    def match(self, canonical):
        valid, metrics = self.score(canonical)
        alignment = np.eye(3, dtype=np.float64)
        metrics["registered"] = False
        if valid:
            return True, metrics, alignment
        corrected, lighting = compensate_shading(self.reference, canonical)
        valid, metrics = self.score(corrected)
        metrics.update(lighting, illumination_corrected=True, registered=False)
        if valid:
            return True, metrics, alignment
        # Correct only a small global registration residual. A piece moving a
        # full intersection or changing identity must survive this global fit
        # and still be rejected by the local tile/ink tests.
        ref = cv2.resize(cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY), (225, 250))
        cur = cv2.resize(cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY), (225, 250))
        try:
            _, affine = cv2.findTransformECC(
                ref, cur, np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .0005), None, 3)
        except cv2.error:
            return False, metrics, alignment
        linear = affine[:, :2]
        singular = np.linalg.svd(linear, compute_uv=False)
        angle = abs(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
        if (np.max(np.abs(singular-1)) > .015 or angle > 1.0
                or np.linalg.norm(affine[:, 2]*2) > 3):
            return False, metrics, alignment
        affine[:, 2] *= 2
        aligned = cv2.warpAffine(corrected, affine, (450, 500),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REFLECT)
        valid, metrics = self.score(aligned)
        metrics["registered"] = True
        metrics.update(lighting, illumination_corrected=True)
        alignment[:2] = affine
        return valid, metrics, np.linalg.inv(alignment)


class AppearanceTracker:
    def __init__(self):
        self.clear()

    def clear(self):
        self.epoch = None
        self.anchor = None
        self.appearance = None
        self.last_frame_id = -1
        self.transport_matrix = None
        self.transport_view = None
        self.last_metrics = None
        self.last_compare_matrix = None
        self.reference_gray = self.reference_points = self.reference_scale = None
        self.reason = "尚无已确认盘面"

    def ready(self, epoch):
        return self.anchor is not None and self.epoch == epoch

    def board_polygon(self, view_transform):
        if self.anchor is None:
            return None
        matrix = self.transport_matrix @ self.transport_view @ np.linalg.inv(view_transform)
        return cv2.perspectiveTransform(np.float32([[[25,25],[425,25],[425,475],[25,475]]]),
                                        np.linalg.inv(matrix))[0]

    def seed(self, observation, packet, epoch, neural_observation):
        from localization.live_vision import grid_reason
        evidence = observation.get("temporal_evidence", {}).get("frame_ids", [])
        if (observation.get("rejection") or observation.get("matrix") is None
                or len(set(evidence)) < 2
                or observation["frame_id"] != packet.frame_id
                or observation["captured_at"] != packet.captured_at
                or grid_reason(observation["grid_chars"], observation["conf_grid"], None)
                or neural_observation.get("rejection")
                or neural_observation.get("raw_unknown")
                or neural_observation["grid_chars"] != observation["grid_chars"]
                or neural_observation["frame_id"] != packet.frame_id
                or neural_observation["captured_at"] != packet.captured_at):
            raise ValueError("Appearance reference needs a confirmed board and its exact camera frame")
        self.anchor = copy.deepcopy(observation)
        self.anchor.pop("piece_offsets", None)
        self.epoch = epoch
        canonical = cv2.warpPerspective(packet.image, np.asarray(observation["matrix"]), (450, 500),
                                        borderMode=cv2.BORDER_REFLECT)
        self.appearance = BoardAppearance(canonical)
        self.last_frame_id = packet.frame_id
        self.transport_matrix = np.asarray(observation["matrix"]).copy()
        self.transport_view = np.asarray(observation.get("view_transform", np.eye(3))).copy()
        height, width = packet.image.shape[:2]
        small_width = min(640, width)
        size = (small_width, round(height*small_width/width))
        self.reference_scale = np.diag([size[0]/width, size[1]/height, 1.])
        self.reference_gray = cv2.cvtColor(cv2.resize(packet.image, size), cv2.COLOR_BGR2GRAY)
        support = cv2.perspectiveTransform(
            np.float32([[[25,25],[425,25],[425,475],[25,475]]]),
            self.reference_scale @ np.linalg.inv(np.asarray(observation["matrix"])))[0]
        mask = np.zeros(self.reference_gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.round(support).astype(np.int32), 255)
        self.reference_points = cv2.goodFeaturesToTrack(self.reference_gray, 300, .015, 5, mask=mask)
        self.reason = "参考盘面已建立"

    def anchored_matrix(self, frame):
        """Register directly to immutable reference, never integrate flow steps."""
        if self.reference_points is None or len(self.reference_points) < 16:
            return None
        current = cv2.cvtColor(cv2.resize(frame, self.reference_gray.shape[::-1]), cv2.COLOR_BGR2GRAY)
        moved, forward, _ = cv2.calcOpticalFlowPyrLK(
            self.reference_gray, current, self.reference_points, None,
            winSize=(21,21), maxLevel=3)
        if moved is None or forward is None or not np.isfinite(moved).all():
            return None
        back, backward, _ = cv2.calcOpticalFlowPyrLK(
            current, self.reference_gray, moved, None, winSize=(21,21), maxLevel=3)
        if back is None or backward is None:
            return None
        valid = (forward.ravel().astype(bool) & backward.ravel().astype(bool)
                 & (np.linalg.norm(back-self.reference_points, axis=2).ravel() < .7))
        if valid.sum() < 16:
            return None
        homography, inliers = cv2.findHomography(self.reference_points[valid], moved[valid],
                                                cv2.RANSAC, .7)
        if homography is None or inliers is None or inliers.mean() < .70:
            return None
        source = self.reference_points[valid][inliers.ravel().astype(bool),0]
        # A tiny surviving cluster cannot establish full-board geometry.
        if np.any(np.ptp(source,axis=0) < np.ptp(self.reference_points[:,0],axis=0)*.5):
            return None
        if not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-8:
            return None
        raw_motion = np.linalg.inv(self.reference_scale) @ homography @ self.reference_scale
        return np.asarray(self.anchor["matrix"]) @ np.linalg.inv(raw_motion)

    def verify(self, packet, view_transform, epoch, now):
        from localization.live_vision import geometry_reason
        start = time.perf_counter()
        if not self.ready(epoch):
            self.reason = "观察代际已变化"
            return None
        if packet is None or now-packet.captured_at > .60 or now < packet.captured_at:
            self.reason = "相机帧过期或时间异常"
            return None
        if packet.frame_id <= self.last_frame_id:
            self.reason = "重复相机帧"
            return None
        current_view = np.asarray(view_transform)
        if (current_view.shape != (3, 3) or not np.isfinite(current_view).all()
                or abs(np.linalg.det(current_view)) < 1e-8):
            self.reason = "跟踪坐标异常"
            return None
        # Rebase camera transport after each successful registration. This
        # corrects accumulated flow drift without changing the original image
        # exemplar or its classified identities.
        matrix = self.transport_matrix @ self.transport_view @ np.linalg.inv(current_view)
        nominal = np.float32([[[50, 50], [400, 50], [50, 450], [400, 450]]])
        corners = cv2.perspectiveTransform(nominal, np.linalg.inv(matrix))[0]
        reason = geometry_reason(corners, self.anchor["corner_scores"], packet.image.shape)
        if reason:
            self.reason = reason
            return None
        matrix = self.anchored_matrix(packet.image)
        if matrix is None:
            self.reason = "参考画面配准证据不足"
            return None
        corners = cv2.perspectiveTransform(nominal, np.linalg.inv(matrix))[0]
        reason = geometry_reason(corners, self.anchor["corner_scores"], packet.image.shape)
        if reason:
            self.reason = reason
            return None
        canonical = cv2.warpPerspective(packet.image, matrix, (450, 500),
                                        borderMode=cv2.BORDER_REFLECT)
        self.last_compare_matrix = matrix.tolist()
        valid, metrics, registration = self.appearance.match(canonical)
        self.last_metrics = metrics
        if not valid:
            self.reason = "盘面外观变化，重新确认"
            return None
        matrix = registration @ matrix
        corners = cv2.perspectiveTransform(nominal, np.linalg.inv(matrix))[0]
        reason = geometry_reason(corners, self.anchor["corner_scores"], packet.image.shape)
        if reason:
            self.reason = reason
            return None
        self.last_frame_id = packet.frame_id
        self.transport_matrix = matrix.copy()
        self.transport_view = current_view.copy()
        result = copy.deepcopy(self.anchor)
        for key in ("piece_offsets", "recovery", "refinement_edits", "temporal_evidence"):
            result.pop(key, None)
        result.update(
            frame_id=packet.frame_id, captured_at=packet.captured_at,
            corners=corners.tolist(), matrix=matrix.tolist(), view_transform=current_view.tolist(),
            source="appearance_verified", confidence_source="reference_classification",
            classification_reference_frame_id=self.anchor["frame_id"],
            classification_reference_time=self.anchor["captured_at"],
            offsets_valid=False, verification=metrics, raw_unknown=[], rejection="",
            verification_ms=(time.perf_counter()-start)*1000)
        self.reason = "当前新帧外观复核通过"
        return result
