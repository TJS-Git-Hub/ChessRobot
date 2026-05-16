"""
舵机物理零点标定工具
====================
逐通道微调 ref_us，使每个关节在"竖直向上 90°"时物理对准。

标定参考姿态（四舵机同时归 ref_us 时）：
  - Base (ch 0):  从正上方俯视，大臂指向棋盘 e 列正后方（正前方）
  - Shoulder (ch 4): 从侧面看，大臂竖直指天
  - Elbow (ch 8):    从侧面看，小臂竖直指天，与大臂成一条直线
  - Wrist (ch 12):   从小臂方向看，吸盘垂直向下（相对小臂 0°）

操作说明：
  输入 + / -     粗调 +-10 us
  输入 ++ / --   微调 +-2 us
  输入数字       直接设为目标脉宽
  输入 s/回车    确认当前值，进入下一通道
  输入 b         返回上一通道
  输入 q         退出并输出当前结果
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from hardware import PCA9685Driver
import config


CHANNELS = [0, 4, 8, 12]

CHANNEL_ALIGN_GUIDE = {
    0:  "Base    - 俯视：大臂应指向棋盘 e 列正后方（正前方）",
    4:  "Shoulder - 侧视：大臂应严格竖直指天",
    8:  "Elbow   - 侧视：小臂应严格竖直指天，与大臂成直线",
    12: "Wrist   - 侧视：吸盘应垂直向下（按住小臂不动时检查）",
}


def calibrate():
    hw = PCA9685Driver()

    # 从 config 加载当前值作为起点
    new_config = {}
    for ch in CHANNELS:
        new_config[ch] = config.SERVO_CONFIG[ch]['ref_us']

    print("=" * 60)
    print("         舵机物理零点标定工具")
    print("=" * 60)
    print()
    print("*  标定前请确认：所有舵机已安装，机械臂可自由运动")
    print()

    # ── 第一步：全通道归位检查 ──────────────────────────
    print("-" * 60)
    print("[步骤 0] 全通道归位 - 检查当前整体姿态")
    print("-" * 60)
    for ch in CHANNELS:
        cfg = config.SERVO_CONFIG[ch]
        hw.set_pwm_us(ch, cfg['ref_us'])
        print(f"  通道 {ch} ({cfg['name']}) -> {cfg['ref_us']} us")

    input("\n按回车开始逐通道标定...")

    # ── 第二步：逐通道微调 ──────────────────────────────
    idx = 0
    while idx < len(CHANNELS):
        ch = CHANNELS[idx]
        cfg = config.SERVO_CONFIG[ch]
        current_us = new_config[ch]

        print()
        print("=" * 60)
        print(f"[通道 {idx+1}/4]  通道 {ch} - {cfg['name']}")
        print(f"当前 ref_us: {current_us}  |  安全范围: {cfg['min']} ~ {cfg['max']}")
        print(f"标定目标: {CHANNEL_ALIGN_GUIDE[ch]}")
        print("=" * 60)

        hw.set_pwm_us(ch, current_us)
        time.sleep(0.3)

        print("  操作: +/-(粗调)  ++/--(微调)  数字(直接设)  s/回车(确认)  b(返回)  q(退出)")
        user_input = input("  >> ").strip()

        if user_input == "" or user_input.lower() == "s":
            new_config[ch] = current_us
            print(f"  * 通道 {ch} ({cfg['name']}) ref_us 已记录: {current_us}")
            idx += 1
            continue

        if user_input.lower() == "b":
            if idx > 0:
                idx -= 1
                hw.set_pwm_us(CHANNELS[idx], new_config[CHANNELS[idx]])
            else:
                print("  已在第一个通道")
            continue

        if user_input.lower() == "q":
            print("\n* 标定中断")
            break

        if user_input == "++":
            current_us = min(cfg['max'], current_us + 2)
        elif user_input == "--":
            current_us = max(cfg['min'], current_us - 2)
        elif user_input == "+":
            current_us = min(cfg['max'], current_us + 10)
        elif user_input == "-":
            current_us = max(cfg['min'], current_us - 10)
        else:
            try:
                val = int(user_input)
                if cfg['min'] <= val <= cfg['max']:
                    current_us = val
                else:
                    print(f"  * 超出安全范围 ({cfg['min']}~{cfg['max']})，已忽略")
                    continue
            except ValueError:
                print("  * 无效输入")
                continue

        new_config[ch] = current_us
        hw.set_pwm_us(ch, current_us)

    # ── 第三步：输出结果 ─────────────────────────────────
    print()
    print("=" * 60)
    print("         标定完成 - 新 SERVO_CONFIG")
    print("=" * 60)
    print()
    print("将以下内容更新到 config.py 的 SERVO_CONFIG 中：")
    print()
    print("SERVO_CONFIG = {")
    for ch in CHANNELS:
        cfg = config.SERVO_CONFIG[ch]
        old_us = cfg['ref_us']
        new_us = new_config[ch]
        delta = new_us - old_us
        sign = "+" if delta >= 0 else ""
        print(f"    {ch}:  {{'ref_us': {new_us}, 'min': {cfg['min']}, 'max': {cfg['max']}, 'name': '{cfg['name']}'}},   # 原 {old_us} -> {new_us} ({sign}{delta})")
    print("}")

    hw.close()
    print("\n* 标定程序结束。")


if __name__ == "__main__":
    try:
        calibrate()
    except Exception as e:
        print(f"\nX 标定失败: {e}")
        import traceback
        traceback.print_exc()
