import cv2
import os
import time

# 定义项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_path = os.path.join(project_root, "recognition/data/real_raw_full")  # 存原图


def collect_full_images():
    if not os.path.exists(save_path): os.makedirs(save_path)

    # 强制设置分辨率到 1280x720
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("--- 📸 真实高空数据采集系统 (YOLO版)启动 ---")
    print(f"请将摄像头架在对弈时的真实高度(40cm)，观察整个棋盘。")
    print("操作方式：")
    print("  [空格] - 拍摄并保存当前整幅画面")
    print("  [Q]    - 退出采集")

    count = len(os.listdir(save_path))

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 实时显示，供你调整棋子位置
        cv2.putText(frame, f"Captured: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capture - Align Piece and Press Space", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

        if key == ord(' '):  # 按空格拍照
            # 使用时间戳防止文件名重复
            timestamp = int(time.time())
            filename = os.path.join(save_path, f"chess_{timestamp}_{count}.jpg")

            # 保存原图
            cv2.imwrite(filename, frame)
            print(f"✅ 已保存整幅画面 ({1280}x{720}) 到 {filename}")
            count += 1

            # 可选：闪烁一下提示拍照成功
            frame[:] = (255, 255, 255)
            cv2.imshow("Capture - Align Piece and Press Space", frame)
            cv2.waitKey(100)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    collect_full_images()