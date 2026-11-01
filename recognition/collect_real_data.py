import cv2
import os
import time

# 定义项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_base_dir = os.path.join(project_root, "recognition", "data", "raw_dataset")

# 预设的采集场景类别
SCENARIOS = {
    "1": "indoor_normal",     # 室内正常光照
    "2": "indoor_dark",       # 室内偏暗/弱光
    "3": "indoor_bright",     # 室内强光/反光
    "4": "outdoor_shadow",    # 室外阴影/自然漫射光
    "5": "outdoor_bright",    # 室外强光/直射光
    "6": "empty_board",       # 空棋盘背景素材
    "7": "calibration",       # 黑白标定板照片
    "8": "offset_test"        # 已知物理偏移测试
}

def collect_full_images():
    # 摄像头初始化（默认索引 0，若有外接俯视相机可根据设备管理器微调）
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_key = "1"
    current_scenario = SCENARIOS[current_key]

    print("=" * 60)
    print("📸 ChessRobot 视觉数据采集系统 (高保真无水印版)")
    print("=" * 60)
    print("快捷键说明：")
    print("  [空格]  - 拍照并保存纯净无水印原图")
    print("  [1~8]   - 切换采集子目录/光照场景")
    print("            1: 室内正常光   2: 室内较暗   3: 室内强光")
    print("            4: 室外阴影     5: 室外强光   6: 空棋盘背景")
    print("            7: 相机标定板   8: 已知偏移验证")
    print("  [Q]     - 退出采集系统")
    print("=" * 60)

    flash_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法从摄像头读取画面，请检查连接。")
            break

        # 当前场景保存目录
        current_dir = os.path.join(data_base_dir, current_scenario)
        os.makedirs(current_dir, exist_ok=True)
        count = len([f for f in os.listdir(current_dir) if f.lower().endswith(('.jpg', '.png'))])

        # 【关键修复】创建专门用于屏幕显示的副本，绝不修改原始 frame！
        display_frame = frame.copy()

        # 白屏闪烁效果（拍摄反馈）
        if flash_frames > 0:
            display_frame[:] = (255, 255, 255)
            flash_frames -= 1
        else:
            # 绘制采集信息在显示副本上
            header_text = f"[{current_key}] {current_scenario} | Count: {count}"
            cv2.putText(display_frame, header_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, "Press SPACE to Save | 1-8 to Switch Tag | Q to Quit",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)

        cv2.imshow("ChessRobot Data Collector (Clean Raw)", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

        # 切换场景标签
        char_key = chr(key) if 0 <= key <= 255 else ''
        if char_key in SCENARIOS:
            current_key = char_key
            current_scenario = SCENARIOS[current_key]
            print(f"🔄 切换到场景目录: [{current_key}] {current_scenario}")

        # 空格拍照
        if key == ord(' '):
            timestamp = int(time.time() * 1000)
            filename = f"{current_scenario}_{timestamp}_{count + 1:04d}.jpg"
            full_path = os.path.join(current_dir, filename)

            # 【关键保证】保存未经任何文字污染的原始图像
            cv2.imwrite(full_path, frame)
            print(f"✅ [已保存原图] {filename} (分辨率: {frame.shape[1]}x{frame.shape[0]})")
            flash_frames = 3  # 闪烁反馈 3 帧

    cap.release()
    cv2.destroyAllWindows()
    print("👋 数据采集已安全退出。")

if __name__ == "__main__":
    collect_full_images()