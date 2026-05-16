import os
import sys
import cv2
import numpy as np
import time
import threading
from collections import Counter, deque
from ultralytics import YOLO

# --- 0. 环境静默 ---
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;*.warning=false"


# --- 1. 异步视频流 (直接整合，防止报错) ---
class VideoStream:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            print("❌ 无法打开摄像头，请检查 USB 连接！")
            sys.exit()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


# --- 2. 核心视觉类 ---
class ChessVisionV15_4:
    def __init__(self, model_path):
        print(f"⏳ 正在启动 V15.4 [终极整合版] 加载模型...")
        self.model = YOLO(model_path)
        self.model.to('cuda')

        self.H = None
        self.roi_corners = []
        self.expanded_roi = []
        self.orientation_confirmed = False

        # 稳态变量
        self.expected_count = 32
        self.last_stable_board = {}
        self.board_window = deque(maxlen=35)
        self.is_ready = False

        # 滤波与运动
        self.smooth_centers = {}
        self.alpha = 0.28
        self.piece_energy = {}
        self.piece_history = {}

        # --- 极限灵敏度阈值 ---
        self.detect_conf = 0.35
        self.keep_conf = 0.20
        self.roi_buffer_px = 60  # 约 2-3 cm 缓冲区
        self.motion_threshold = 4.5
        self.clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))

    def on_capture_event(self):
        """外部调用：吃子后总数减 1"""
        self.expected_count -= 1
        print(f"⚠️ [系统] 吃子动作确认，当前预期总数: {self.expected_count}")

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.roi_corners) < 4:
            self.roi_corners.append([x, y])

    def calibrate(self, vs):
        print("📍 点击顺序：1.左上 -> 2.右上 -> 3.右下 -> 4.左下")
        self.roi_corners = []
        cv2.namedWindow("Main_Vision")
        cv2.setMouseCallback("Main_Vision", self._on_mouse)
        while True:
            frame = vs.read()
            display = frame.copy()
            for i, pt in enumerate(self.roi_corners):
                cv2.circle(display, tuple(pt), 10, (0, 255, 0), -1)
                cv2.putText(display, f"Pt{i + 1}", (pt[0] + 10, pt[1]), 1, 1.5, (0, 255, 0), 2)
            cv2.imshow("Main_Vision", display)
            if len(self.roi_corners) == 4: break
            cv2.waitKey(1)

        pts = np.array(self.roi_corners, dtype=np.float32)
        center = np.mean(pts, axis=0)
        self.expanded_roi = [(pt + (pt - center) / np.linalg.norm(pt - center) * self.roi_buffer_px).astype(int) for pt
                             in pts]
        print("✅ ROI 锁定。正在自动判定黑方 a0 位置...")

    def is_black_piece(self, frame, box):
        """利用 L 通道精准识别黑棋 (RGB 000)"""
        x1, y1, x2, y2 = map(int, box)
        roi = frame[max(0, y1 + 15):y2 - 15, max(0, x1 + 15):x2 - 15]
        if roi.size == 0: return False
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        # 黑色亮度极低，L < 45
        return (np.sum(lab[:, :, 0] < 45) / roi.size) > 0.5

    def set_black_view(self, mode):
        """黑方视角 a0 映射：黑方最右边的车为 a0"""
        src_pts = np.array(self.roi_corners, dtype=np.float32)
        if mode == 1:  # 黑方在上方 (远端)，其右手车在 Pt1 附近
            print("🚩 视角锁定：黑方在远端，Pt1 -> a0")
            dst_pts = np.array([[0, 0], [8, 0], [8, 9], [0, 9]], dtype=np.float32)
        else:  # 黑方在下方 (近端)，其右手车在 Pt3 附近
            print("🚩 视角锁定：黑方在近端，Pt3 -> a0")
            dst_pts = np.array([[8, 9], [0, 9], [0, 0], [8, 0]], dtype=np.float32)

        self.H, _ = cv2.findHomography(src_pts, dst_pts)
        self.orientation_confirmed = True

        # 1. 新增：彻底清除历史状态，并接收新的预期总数
        def reset_baseline(self, new_expected_count=None):
            """机械臂走完后调用，强制清除历史状态，忽略期间的物理干扰"""
            if new_expected_count is not None:
                self.expected_count = new_expected_count

            self.last_stable_board = {}
            self.board_window.clear()
            self.is_ready = False
            self.smooth_centers.clear()
            self.piece_energy.clear()
            self.piece_history.clear()
            print(f"🔄 视觉追踪基准已重置，当前预期棋子总数: {self.expected_count}")

    def run(self, vs, on_move_callback=None):
        self.calibrate(vs)
        while True:
            frame = vs.read()
            if frame is None: break

            # --- 1. 双路处理：仅在 ROI 内推理，保持显示明亮 ---
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(self.expanded_roi)], 255)
            masked_img = cv2.bitwise_and(frame, frame, mask=mask)

            lab = cv2.cvtColor(masked_img, cv2.COLOR_BGR2LAB)
            l, a, b_chan = cv2.split(lab)
            l = self.clahe.apply(l)
            ai_input = cv2.cvtColor(cv2.merge((l, a, b_chan)), cv2.COLOR_LAB2BGR)

            results = self.model.track(ai_input, imgsz=1280, persist=True, verbose=False, conf=self.keep_conf)

            display_frame = frame.copy()
            ids_this_frame = set()
            active_pieces_now = {}
            global_motion = False
            black_count_debug = 0

            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                # 自动判别视角
                if not self.orientation_confirmed:
                    black_y = [(b[1] + b[3]) / 2 for b in boxes if self.is_black_piece(frame, b)]
                    black_count_debug = len(black_y)
                    if black_count_debug >= 8:
                        self.set_black_view(1 if np.mean(black_y) < 360 else 2)

                for tid, box, conf in zip(ids, boxes, confs):
                    # 双门槛逻辑
                    if tid not in self.piece_energy and conf < self.detect_conf: continue

                    raw_cx, raw_cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

                    # 运动检测与平滑
                    if tid in self.smooth_centers:
                        dist_sq = (raw_cx - self.smooth_centers[tid][0]) ** 2 + (
                                    raw_cy - self.smooth_centers[tid][1]) ** 2
                        if dist_sq > 2.25:  # 1.5 像素死区
                            self.smooth_centers[tid][0] = self.smooth_centers[tid][0] * (
                                        1 - self.alpha) + raw_cx * self.alpha
                            self.smooth_centers[tid][1] = self.smooth_centers[tid][1] * (
                                        1 - self.alpha) + raw_cy * self.alpha
                            if dist_sq > 25: global_motion = True
                    else:
                        self.smooth_centers[tid] = [raw_cx, raw_cy]

                    scx, scy = int(self.smooth_centers[tid][0]), int(self.smooth_centers[tid][1])

                    # 始终绘制框和点
                    cv2.rectangle(display_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
                    cv2.circle(display_frame, (scx, scy), 6, (0, 0, 255), -1)

                    if self.orientation_confirmed:
                        pt = np.array([[[scx, scy]]], dtype=np.float32)
                        res = cv2.perspectiveTransform(pt, self.H)[0][0]
                        f, r = int(round(res[0])), int(round(res[1]))
                        coord = f"{chr(ord('a') + max(0, min(8, f)))}{max(0, min(9, r))}"

                        ids_this_frame.add(tid)
                        self.piece_energy[tid] = min(100, self.piece_energy.get(tid, 0) + 40)
                        if tid not in self.piece_history: self.piece_history[tid] = deque(maxlen=12)
                        self.piece_history[tid].append(coord)

                        # 巨型坐标
                        cv2.putText(display_frame, coord, (scx - 30, scy - 55), 1, 2.5, (0, 0, 0), 7)
                        cv2.putText(display_frame, coord, (scx - 30, scy - 55), 1, 2.5, (0, 255, 255), 3)

            # 汇总稳态
            for tid in list(self.piece_energy.keys()):
                if tid not in ids_this_frame: self.piece_energy[tid] -= 15
                if self.piece_energy[tid] > 50:
                    bc = Counter(self.piece_history[tid]).most_common(1)[0][0]
                    active_pieces_now[bc] = tid
                if self.piece_energy[tid] <= 0:
                    self.piece_energy.pop(tid, None)
                    self.smooth_centers.pop(tid, None)

            # --- 3. 严格稳定逻辑 ---
            current_count = len(active_pieces_now)
            # 逻辑放宽：允许棋盘总数等于预期值（平移），或者等于预期值减一（玩家吃子）
            count_ok = (current_count == self.expected_count) or (current_count == self.expected_count - 1)

            if count_ok and not global_motion:
                self.board_window.append(active_pieces_now)
            else:
                self.board_window.clear()

            if len(self.board_window) == self.board_window.maxlen and all(
                    b == self.board_window[0] for b in self.board_window):
                if not self.is_ready:
                    # 接收起点和终点
                    moved_from, moved_to = self.report_and_compare(active_pieces_now)
                    self.is_ready = True

                    # [核心阻塞逻辑]：触发主程序回调
                    if moved_from and moved_to and on_move_callback:
                        on_move_callback(moved_from, moved_to)
                        # 注意：此处不再由视觉系统自己 reset_baseline，
                        # 而是由 main.py 在机械臂动作完成后主动调用 vision.reset_baseline(new_count)
            else:
                self.is_ready = False

            # UI
            color = (0, 255, 0) if self.is_ready else (0, 0, 255)
            cv2.rectangle(display_frame, (0, 0), (650, 110), color, -1)
            status_txt = "READY: STABLE" if self.is_ready else f"SYNCING... ({len(self.board_window)}/35)"
            cv2.putText(display_frame, status_txt, (20, 45), 1, 2.0, (255, 255, 255), 2)
            cv2.putText(display_frame, f"PIECES: {len(active_pieces_now)} / {self.expected_count}", (20, 90), 1, 1.8,
                        (255, 255, 255), 2)

            if not self.orientation_confirmed:
                cv2.putText(display_frame, f"Detecting Black: {black_count_debug}/8 (Press 1/2)", (700, 45), 1, 1.5,
                            (0, 255, 255), 2)

            cv2.imshow("Main_Vision", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.set_black_view(1)
            elif key == ord('2'):
                self.set_black_view(2)
            elif key == ord('c'):
                self.on_capture_event()

        vs.stop();
        cv2.destroyAllWindows()

    def report_and_compare(self, current_board):
        if not self.last_stable_board:
            self.last_stable_board = current_board.copy()
            print("🟢 初始盘面锁定，开始监控移动...")
            return None, None

        lost = [p for p in self.last_stable_board if p not in current_board]
        gain = [p for p, tid in current_board.items() if
                p not in self.last_stable_board or self.last_stable_board[p] != tid]

        if lost and gain:
            print(f"\n🚀 [视觉上报] 运动轨迹确认: {lost[0]} -> {gain[0]}")
            self.last_stable_board = current_board.copy()
            return lost[0], gain[0]

        return None, None


if __name__ == "__main__":
    PATH = r"C:\Users\Liu\PycharmProjects\PythonProject2\recognition\models\best.pt"
    ChessVisionV15_4(PATH).run(VideoStream().start())