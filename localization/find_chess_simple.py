import cv2
import numpy as np
import os
import time
import threading
from ultralytics import YOLO


# --- 1. 异步视频流：压榨 RTX 4060 性能 ---
class VideoStream:
    def __init__(self, src=1, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
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


class ChessVisionV7:
    def __init__(self):
        # 1. 路径与模型加载
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(root, "recognition", "models", "best.pt")
        self.calib_path = os.path.join(root, "recognition", "models", "H_matrix_v7.npy")
        self.corners_path = os.path.join(root, "recognition", "models", "ROI_corners_v7.npy")

        print("⏳ 正在启动 V7.2 [智能极性对齐版]...")
        self.model = YOLO(self.model_path)
        self.model.to('cuda')

        # 2. 状态变量
        self.H = None
        self.roi_corners = []
        self.ema_boxes = {}
        self.ema_alpha = 0.35

        self.orientation_confirmed = False
        self.last_stable_board = {}
        self.last_frame_board = {}
        self.stability_counter = 0
        self.is_ready = False
        self.roi_margin = 40

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.roi_corners) < 4:
            self.roi_corners.append([x, y])

    def calibrate(self, vs):
        print("📍 请严格按顺序点击：1.左上 -> 2.右上 -> 3.右下 -> 4.左下")
        self.roi_corners = []
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self._on_mouse)
        while True:
            frame = vs.read()
            display = frame.copy()
            for i, pt in enumerate(self.roi_corners):
                cv2.circle(display, tuple(pt), 8, (0, 255, 0), -1)
                cv2.putText(display, f"Pt{i + 1}", (pt[0] + 10, pt[1]), 1, 1.5, (0, 255, 0), 2)
            cv2.imshow("Calibration", display)
            if cv2.waitKey(1) == ord('r'): self.roi_corners = []
            if len(self.roi_corners) == 4: break
        cv2.destroyWindow("Calibration")

    def get_board_pos(self, cx, cy):
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self.H)[0][0]
        f, r = int(round(res[0])), int(round(res[1]))
        f, r = max(0, min(8, f)), max(0, min(9, r))
        return f"{chr(ord('a') + f)}{r}"

    def reset_baseline(self):
        """机械臂走完后调用，忽略期间的物理干扰"""
        self.last_stable_board = {}
        self.last_frame_board = {}
        self.stability_counter = 0
        self.is_ready = False
        self.ema_boxes.clear()
        print("🔄 视觉追踪基准已重置，等待下一轮对局...")

    def run(self, on_move_callback=None):
        vs = VideoStream().start()
        time.sleep(2.0)
        self.calibrate(vs)
        print("🚀 正在通过棋子颜色分布确定 a0 极性...")
        prev_time = time.time()

        while True:
            frame = vs.read()
            if frame is None: break
            display_frame = frame.copy()

            # YOLO 追踪 (启用 RTX 4060 Half 推理)
            results = self.model.track(frame, imgsz=1280, persist=True, tracker="bytetrack.yaml", verbose=False,
                                       half=True)

            current_frame_board = {}

            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy().astype(int)

                # --- 极性自动校准逻辑 ---
                if not self.orientation_confirmed:
                    red_y_list = []
                    for box, cls_idx in zip(boxes, clss):
                        if results[0].names[cls_idx].startswith('r'):  # 检测到红子
                            red_y_list.append((box[1] + box[3]) / 2)

                    if len(red_y_list) >= 8:
                        avg_red_y = np.mean(red_y_list)
                        src_pts = np.array(self.roi_corners, dtype=np.float32)

                        # 核心映射算法：
                        # 标定点顺序为 [TL, TR, BR, BL]
                        if avg_red_y > 360:  # 红方在下方
                            print("🚩 判定：红方在摄像头【近端】，a0位于左下角")
                            # a0(0,0)对应BL, i0(8,0)对应BR, i9(8,9)对应TR, a9(0,9)对应TL
                            dst_pts = np.array([[0, 9], [8, 9], [8, 0], [0, 0]], dtype=np.float32)
                        else:  # 红方在上方
                            print("🚩 判定：红方在摄像头【远端】，a0位于右上角")
                            # a0(0,0)对应TR, i0(8,0)对应TL, i9(8,9)对应BL, a9(0,9)对应BR
                            # 也就是棋盘在摄像头视角里是旋转180度的
                            dst_pts = np.array([[8, 0], [0, 0], [0, 9], [8, 9]], dtype=np.float32)

                        self.H, _ = cv2.findHomography(src_pts, dst_pts)
                        self.orientation_confirmed = True

                # --- 坐标追踪主逻辑 ---
                if self.orientation_confirmed:
                    for tid, box in zip(ids, boxes):
                        self.ema_boxes[tid] = self.ema_alpha * box + (1 - self.ema_alpha) * self.ema_boxes.get(tid, box)
                        s_box = self.ema_boxes[tid]
                        cx, cy = int((s_box[0] + s_box[2]) / 2), int((s_box[1] + s_box[3]) / 2)

                        # ROI 检查
                        pts = np.array(self.roi_corners, dtype=np.int32)
                        if cv2.pointPolygonTest(pts, (cx, cy), True) < -self.roi_margin: continue

                        logic_pos = self.get_board_pos(cx, cy)
                        current_frame_board[logic_pos] = tid

                        # 极简 UI
                        cv2.rectangle(display_frame, (int(s_box[0]), int(s_box[1])), (int(s_box[2]), int(s_box[3])),
                                      (0, 255, 0), 2)
                        cv2.putText(display_frame, logic_pos, (int(s_box[0]), int(s_box[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # --- 稳定性判定与坐标上报 ---
            if self.orientation_confirmed:
                if current_frame_board == self.last_frame_board and len(current_frame_board) > 0:
                    self.stability_counter += 1
                else:
                    self.stability_counter = 0
                    self.is_ready = False

                self.last_frame_board = current_frame_board.copy()

                if self.stability_counter > 15 and not self.is_ready:
                    # 接收坐标返回值
                    moved_from, moved_to = self.process_move(current_frame_board)
                    self.is_ready = True

                    # [核心逻辑]：触发回调，交出控制权。阻塞视觉循环。
                    if moved_from and moved_to and on_move_callback:
                        on_move_callback(moved_from, moved_to)
                        self.reset_baseline()  # 夺回控制权后，立刻重置基准

            # 状态栏渲染
            status_color = (0, 255, 0) if self.is_ready else (0, 0, 255)
            cv2.rectangle(display_frame, (0, 0), (300, 50), status_color, -1)
            msg = "READY" if self.is_ready else "WAITING/SCANNING"
            cv2.putText(display_frame, f"STATUS: {msg}", (10, 35), 1, 1.5, (255, 255, 255), 2)

            # FPS 显示
            fps = 1 / (time.time() - prev_time)
            prev_time = time.time()
            cv2.putText(display_frame, f"FPS: {int(fps)}", (1150, 40), 1, 1.5, (0, 255, 255), 2)

            cv2.imshow("UESTC Chess Robot V7.2 - Pure Coordinate Mode", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        vs.stop()
        cv2.destroyAllWindows()

    def process_move(self, current_board):
        if not self.last_stable_board:
            self.last_stable_board = current_board.copy()
            print("🆕 初始局面已同步，实时追踪开启")
            return None, None

        moved_from, moved_to = "", ""
        is_capture = False

        for pos in self.last_stable_board:
            if pos not in current_board: moved_from = pos; break

        for pos, tid in current_board.items():
            if pos not in self.last_stable_board:
                moved_to = pos
            elif self.last_stable_board[pos] != tid:
                moved_to = pos
                is_capture = True

        if moved_from and moved_to:
            cap_flag = " [CAPTURE]" if is_capture else ""
            print(f"\n🚀 [视觉上报] 坐标移动: {moved_from} -> {moved_to}{cap_flag}")
            self.last_stable_board = current_board.copy()
            return moved_from, moved_to
        return None, None

if __name__ == "__main__":
    ChessVisionV7().run()