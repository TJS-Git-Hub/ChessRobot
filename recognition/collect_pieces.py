import cv2
import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# 定义项目目录与保存根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_base_dir = os.path.join(project_root, "recognition", "data", "single_pieces")

# 14 类棋子定义 (中文显示名, 英文类别目录名, 单字符编码)
PIECE_LIST = [
    ("红帅 (帅)", "red_king", "K"),
    ("红仕 (仕)", "red_advisor", "A"),
    ("红相 (相)", "red_bishop", "B"),
    ("红马 (马)", "red_knight", "N"),
    ("红车 (车)", "red_rook", "R"),
    ("红炮 (炮)", "red_cannon", "C"),
    ("红兵 (兵)", "red_pawn", "P"),
    ("黑将 (将)", "black_king", "k"),
    ("黑士 (士)", "black_advisor", "a"),
    ("黑象 (象)", "black_bishop", "b"),
    ("黑马 (马)", "black_knight", "n"),
    ("黑车 (车)", "black_rook", "r"),
    ("黑炮 (炮)", "black_cannon", "c"),
    ("黑卒 (卒)", "black_pawn", "p"),
]

def collect_pieces():
    os.makedirs(save_base_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_idx = 0
    flash_frames = 0

    print("=" * 65)
    print("🎯 中国象棋 14 类棋子单兵特写采集导航系统")
    print("=" * 65)
    print("系统将引导你依次完成 14 种棋子的拍摄：")
    print("  [空格键]  - 拍照并保存纯净无水印原图 (建议每颗拍 2~3 张不同角度)")
    print("  [N]或[回车]- 下一颗棋子 (Next)")
    print("  [B]       - 上一颗棋子 (Back)")
    print("  [Q]       - 退出系统")
    print("=" * 65)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取摄像头画面，请检查连接。")
            break

        piece_cn, piece_en, piece_code = PIECE_LIST[current_idx]
        piece_dir = os.path.join(save_base_dir, piece_en)
        os.makedirs(piece_dir, exist_ok=True)

        existing_count = len([f for f in os.listdir(piece_dir) if f.lower().endswith(('.jpg', '.png'))])

        # 创建显示副本（绝不修改保存的原图 frame）
        display_frame = frame.copy()

        # 白屏闪烁效果
        if flash_frames > 0:
            display_frame[:] = (255, 255, 255)
            flash_frames -= 1
        else:
            # 顶部信息横幅
            cv2.rectangle(display_frame, (0, 0), (1280, 85), (30, 30, 30), -1)

            step_text = f"[{current_idx + 1}/14] Target: {piece_en.upper()} ({piece_code})"
            cv2.putText(display_frame, step_text, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

            guide_text = f"Place '{piece_cn}' on board | Saved: {existing_count} pics"
            cv2.putText(display_frame, guide_text, (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # 操作快捷键提示
            cv2.putText(display_frame, "SPACE: Capture | N: Next Piece | B: Prev Piece | Q: Quit",
                        (650, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            # 画面中央辅助参考虚线框（提示把棋子放在中间区域）
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            cv2.rectangle(display_frame, (cx - 150, cy - 150), (cx + 150, cy + 150), (0, 200, 255), 1)
            cv2.putText(display_frame, "Center Zone", (cx - 50, cy - 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        cv2.imshow("ChessRobot Single Piece Collector", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

        # 下一颗棋子
        if key == ord('n') or key == ord('N') or key == 13:  # 13 为 Enter 键
            if current_idx < len(PIECE_LIST) - 1:
                current_idx += 1
                next_cn, next_en, _ = PIECE_LIST[current_idx]
                print(f"👉 切换到下一颗: [{current_idx + 1}/14] {next_cn} ({next_en})")
            else:
                print("🎉 全部 14 种棋子已经遍历完毕！如需退出可按 Q 键。")

        # 上一颗棋子
        elif key == ord('b') or key == ord('B'):
            if current_idx > 0:
                current_idx -= 1
                prev_cn, prev_en, _ = PIECE_LIST[current_idx]
                print(f"👈 回退到上一颗: [{current_idx + 1}/14] {prev_cn} ({prev_en})")

        # 拍照保存
        elif key == ord(' '):
            timestamp = int(time.time() * 1000)
            filename = f"{piece_en}_{timestamp}_{existing_count + 1:03d}.jpg"
            save_path = os.path.join(piece_dir, filename)

            # 保存完全纯净无任何水印的原图
            cv2.imwrite(save_path, frame)
            flash_frames = 3
            print(f"✅ 已保存: {piece_en} -> {filename} (当前已存 {existing_count + 1} 张)")

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 棋子单兵采集结束。保存根目录为:")
    print(save_base_dir)

if __name__ == "__main__":
    collect_pieces()

