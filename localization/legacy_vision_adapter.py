"""Unchanged Gen3 YOLO fallback, loaded only when explicitly selected."""
import queue
import threading
from localization.find_chess_new import ChessVisionV15_4, VideoStream

class VisionAdapter(ChessVisionV15_4):
    """继承 V15.4，覆写关键方法以支持外部编排

    改造点：
      1. report_and_compare() — 检测到移动时推送到线程安全队列
      2. reset_baseline()    — 修复原代码的缩进 bug，加入线程锁保护
      3. 新增 suppress_detection 标志 — 机械臂运动期间抑制视觉上报
    """

    def __init__(self, model_path, move_queue: queue.Queue, ready_event: threading.Event):
        super().__init__(model_path)
        self.move_queue = move_queue
        self.ready_event = ready_event
        self._lock = threading.Lock()
        self.suppress_detection = False

    # ── 覆写：移动检测 → 推送到队列 ──────────────────────
    def report_and_compare(self, current_board):
        with self._lock:
            if self.suppress_detection:
                return None, None

            if not self.last_stable_board:
                self.last_stable_board = current_board.copy()
                print("🟢 [视觉] 基准盘面已锁定，等待人类走棋...")
                self.ready_event.set()
                return None, None

            lost = [p for p in self.last_stable_board if p not in current_board]
            gain = [p for p, tid in current_board.items() if
                    p not in self.last_stable_board or self.last_stable_board.get(p) != tid]

            if lost and gain:
                from_pos, to_pos = lost[0], gain[0]
                print(f"\n🚀 [视觉] 检测到移动: {from_pos} → {to_pos}")
                self.last_stable_board = current_board.copy()
                self.move_queue.put((from_pos, to_pos))
                return from_pos, to_pos
            else:
                # 【核心修复】：增加 else 分支，彻底消除静默无反应的假象
                print(f"\n⚠️ [视觉警告] 数量校验通过，但无法解析空间轨迹！(Lost: {lost}, Gain: {gain})")
                print("👉 物理纠正：吃子时产生了 ID 粘连。请将己方棋子拿起，等待 1 秒让系统确认该位置为空，再重新放下并按空格。")
                return None, None

    # ── 覆写：线程安全重置基准 ──────────────────────────
    def reset_baseline(self, new_expected_count: int = None):
        """机械臂动作完成后调用，丢弃机械臂运动期间的干扰帧"""
        with self._lock:
            if new_expected_count is not None:
                self.expected_count = new_expected_count
            self.last_stable_board = {}
            self.board_window.clear()
            self.is_ready = False
            self.ready_event.clear()
            self.smooth_centers.clear()
            self.piece_energy.clear()
            self.piece_history.clear()
        print(f"🔄 [视觉] 基准已重置，预期棋子总数: {self.expected_count}")
