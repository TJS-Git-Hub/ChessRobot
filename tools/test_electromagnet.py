import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ctypes
import time

# ── 加载 DLL ──────────────────────────────────────────────
dll_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CH347DLLA64.DLL")
ch347 = ctypes.WinDLL(dll_path)

# ── 函数原型声明 ───────────────────────────────────────────
ch347.CH347OpenDevice.restype  = ctypes.c_void_p
ch347.CH347OpenDevice.argtypes = [ctypes.c_ulong]

ch347.CH347CloseDevice.restype  = ctypes.c_bool
ch347.CH347CloseDevice.argtypes = [ctypes.c_ulong]

ch347.CH347GPIO_Set.restype  = ctypes.c_bool
ch347.CH347GPIO_Set.argtypes = [
    ctypes.c_ulong,
    ctypes.c_ubyte,
    ctypes.c_ubyte,
    ctypes.c_ubyte,
]

ch347.CH347GPIO_Get.restype  = ctypes.c_bool
ch347.CH347GPIO_Get.argtypes = [
    ctypes.c_ulong,
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.POINTER(ctypes.c_ubyte),
]

# ── 常量 ──────────────────────────────────────────────────
DEV_INDEX  = 0
GPIO2_MASK = 0x04       # DTR0 = GPIO2 = bit2


class Electromagnet:
    """通过 CH347F GPIO2 (DTR0) 控制继电器/电磁铁"""

    def __init__(self, dev_index: int = DEV_INDEX):
        self.dev = dev_index
        handle = ch347.CH347OpenDevice(self.dev)
        if not handle:
            raise RuntimeError(f"无法打开 CH347 设备 {self.dev}，请检查驱动和连接")
        print(f"[OK] CH347 设备已打开，句柄: {handle}")
        self._gpio_set(level=0)

    def _gpio_set(self, level: int):
        data = GPIO2_MASK if level else 0x00
        ok = ch347.CH347GPIO_Set(
            self.dev,
            GPIO2_MASK,
            GPIO2_MASK,
            data,
        )
        if not ok:
            raise RuntimeError("CH347GPIO_Set 调用失败")

    def on(self):
        self._gpio_set(1)
        print("[电磁铁] ON  - 继电器吸合")

    def off(self):
        self._gpio_set(0)
        print("[电磁铁] OFF - 继电器断开")

    def pulse(self, on_sec: float = 1.0, off_sec: float = 0.5):
        self.on()
        time.sleep(on_sec)
        self.off()
        time.sleep(off_sec)

    def close(self):
        self._gpio_set(0)
        ch347.CH347CloseDevice(self.dev)
        print("[OK] CH347 设备已关闭")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


if __name__ == "__main__":
    with Electromagnet() as em:
        print("=== 测试：持续吸合 2 秒 ===")
        em.on()
        time.sleep(2)
        em.off()
