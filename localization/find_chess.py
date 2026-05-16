import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont


class ChessVisionEngineV6:
    def __init__(self):
        # 1. 路径与模型初始化
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(root, "recognition", "models", "best.pt")
        self.font_path = os.path.join(root, "recognition", "fonts", "msyhbd.ttc")
        self.calib_path = os.path.join(root, "recognition", "models", "H_matrix.npy")
        self.corners_path = os.path.join(root, "recognition", "models", "ROI_corners.npy")

        print("⏳ 正在启动 V6.8 鲁棒版 [双重逻辑锁]...")
        self.model = YOLO(self.model_path)
        try:
            self.font = ImageFont.truetype(self.font_path, 24)
        except:
            self.font = None

        # 2. 算法插件：背景差分器
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=120, detectShadows=True)

        # 3. 象棋标签映射
        self.label_cn = {
            'b_che': '黑車', 'b_jiang': '黑將', 'b_ma': '黑馬', 'b_pao': '黑砲',
            'b_shi': '黑士', 'b_xiang': '黑象', 'b_zu': '黑卒',
            'r_che': '红車', 'r_shuai': '红帥', 'r_ma': '红馬', 'r_pao': '红砲',
            'r_shi': '红仕', 'r_xiang': '红相', 'r_bing': '红兵'
        }

        # 4. 状态变量
        self.initial_layout = self._set_initial_layout()
        self.H = None
        self.roi_corners = []
        self.id_to_piece = {}
        self.is_occluded = False
        self.stability_counter = 0
        self.last_stable_count = 0  # 记录上次稳定时的棋子总数
        self.last_board_snapshot = {}

    def _set_initial_layout(self):
        """ 标准开局位置映射 """
        return {
            'a0': 'r_che', 'b0': 'r_ma', 'c0': 'r_xiang', 'd0': 'r_shi', 'e0': 'r_shuai',
            'f0': 'r_shi', 'g0': 'r_xiang', 'h0': 'r_ma', 'i0': 'r_che',
            'b2': 'r_pao', 'h2': 'r_pao', 'a3': 'r_bing', 'c3': 'r_bing',
            'e3': 'r_bing', 'g3': 'r_bing', 'i3': 'r_bing',
            'a9': 'b_che', 'b9': 'b_ma', 'c9': 'b_xiang', 'd9': 'b_shi', 'e9': 'b_jiang',
            'f9': 'b_shi', 'g9': 'b_xiang', 'h9': 'b_ma', 'i9': 'b_che',
            'b7': 'b_pao', 'h7': 'b_pao', 'a6': 'b_zu', 'c6': 'b_zu',
            'e6': 'b_zu', 'g6': 'b_zu', 'i6': 'b_zu'
        }

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.roi_corners) < 4:
            self.roi_corners.append([x, y])

    def interactive_calibrate(self, cap):
        """ y/n 交互标定逻辑 """
        if os.path.exists(self.calib_path) and os.path.exists(self.corners_path):
            choice = input("检测到已有标定数据，是否重新标定？(y/n): ").lower()
            if choice != 'y':
                self.H = np.load(self.calib_path)
                self.roi_corners = np.load(self.corners_path).tolist()
                return

        print("📍 进入标定模式：左上 -> 右上 -> 右下 -> 左下")
        self.roi_corners = []
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self._on_mouse)
        while True:
            ret, frame = cap.read()
            display = frame.copy()
            for i, pt in enumerate(self.roi_corners):
                cv2.circle(display, (pt[0], pt[1]), 8, (0, 255, 0), -1)
            cv2.imshow("Calibration", display)
            if cv2.waitKey(1) == ord('r'): self.roi_corners = []
            if len(self.roi_corners) == 4: break

        src_pts = np.array(self.roi_corners, dtype=np.float32)
        dst_pts = np.array([[0, 9], [8, 9], [8, 0], [0, 0]], dtype=np.float32)
        self.H, _ = cv2.findHomography(src_pts, dst_pts)
        np.save(self.calib_path, self.H)
        np.save(self.corners_path, src_pts)
        cv2.destroyWindow("Calibration")

    def check_mog_occlusion(self, frame):
        """ 物理层：背景差分检测 """
        small_h, small_w = 360, 640
        scale_x, scale_y = small_w / 1280, small_h / 720
        small_frame = cv2.resize(frame, (small_w, small_h))
        fg_mask = self.back_sub.apply(small_frame)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        if len(self.roi_corners) == 4:
            roi_mask = np.zeros((small_h, small_w), dtype=np.uint8)
            scaled_roi = (np.array(self.roi_corners) * [scale_x, scale_y]).astype(np.int32)
            cv2.fillPoly(roi_mask, [scaled_roi], 255)
            fg_mask = cv2.bitwise_and(fg_mask, roi_mask)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = max([cv2.contourArea(cnt) for cnt in contours]) if contours else 0
        return max_area > 2000

    def run(self):
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(30): cap.read()
        self.interactive_calibrate(cap)

        print("🚀 视觉系统启动...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 1. 物理层初步判定
            mog_occluded = self.check_mog_occlusion(frame)
            display_frame = frame.copy()

            # 2. 逻辑层：执行追踪识别
            results = self.model.track(frame, imgsz=1280, persist=True, tracker="bytetrack.yaml", verbose=False)

            current_frame_board = {}
            current_count = 0

            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()
                current_count = len(ids)

                for tid, box, cid in zip(ids, boxes, clss):
                    cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                    logic_pos = self.get_board_pos(cx, cy)
                    yolo_name = results[0].names[int(cid)]

                    if tid not in self.id_to_piece:
                        self.id_to_piece[tid] = self.initial_layout.get(logic_pos, yolo_name)

                    piece = self.id_to_piece[tid]
                    current_frame_board[logic_pos] = piece

                    # --- 可视化：干净的中心十字架 ---
                    color = (0, 0, 255) if piece.startswith('r') else (0, 0, 0)
                    cv2.line(display_frame, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 2)
                    cv2.line(display_frame, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 2)

                    label = f"{self.label_cn.get(piece, piece)}@{logic_pos}"
                    display_frame = self.draw_styled_label(display_frame, label, (int(box[0]), int(box[1]) - 35), color)

            # 3. 【核心】双重遮挡判定逻辑
            # 如果 MOG 检测到异物，或者当前看到的棋子数少于上一次稳定的数量（说明被手挡住了）
            if mog_occluded or (self.last_stable_count > 0 and current_count < self.last_stable_count):
                self.is_occluded = True
            else:
                self.is_occluded = False

            if self.is_occluded:
                # 绘制警告，锁定不上传
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (1280, 70), (0, 0, 150), -1)
                cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
                cv2.putText(display_frame, "VISUAL LOCKED: PIECE HIDDEN OR HAND DETECTED", (300, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                self.stability_counter = 0
            else:
                # 4. 稳定性决策
                if current_frame_board == self.last_board_snapshot and len(current_frame_board) > 0:
                    self.stability_counter += 1
                else:
                    self.stability_counter = 0
                self.last_board_snapshot = current_frame_board

                if self.stability_counter > 15:
                    # 只有在这里，我们才认为棋局已更新，并同步最新的棋子总数
                    self.last_stable_count = len(current_frame_board)
                    cv2.putText(display_frame, "BOARD READY", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            cv2.imshow("UESTC Chess V6.8 - Clean Logic", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

    def get_board_pos(self, cx, cy):
        if self.H is None: return "??"
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self.H)[0][0]
        f, r = int(round(res[0])), int(round(res[1]))
        f, r = max(0, min(8, f)), max(0, min(9, r))
        return f"{chr(ord('a') + f)}{r}"

    def draw_styled_label(self, img, text, pos, color_rgb):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        for dx, dy in [(-1, -1), (1, 1), (-1, 1), (1, -1)]:
            draw.text((pos[0] + dx, pos[1] + dy), text, font=self.font, fill=(255, 255, 255))
        draw.text(pos, text, font=self.font, fill=color_rgb)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    ChessVisionEngineV6().run()