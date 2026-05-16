import math

L1, L2, L3, L4 = 73.0, 231.0, 231.0, 66.0
K_z = 0.06
D_MAX = L2 + L3  # 462 mm
SQ_W, SQ_H, RIVER = 34.2, 34.2, 34.0
ORIGIN_Y = 107.0
BOARD_Z = -15.0
COL = {c: i for i, c in enumerate("abcdefghi")}

def phys(pos):
    c, r = COL[pos[0]], int(pos[1:])
    x = (4 - c) * SQ_W
    y = ORIGIN_Y
    if r <= 4:
        y += r * SQ_H
    else:
        y += 4 * SQ_H + RIVER + (r - 5) * SQ_H
    return x, y, BOARD_Z

def check(pos):
    x, y, z = phys(pos)
    R = math.sqrt(x**2 + y**2)
    cz = z + R * K_z
    dZ = (cz + L4) - L1
    D = math.sqrt(R**2 + dZ**2)
    ok = abs(L2 - L3) <= D <= D_MAX
    return x, y, R, D, D_MAX - D, ok

print("=" * 72)
print(f"D_max = L2+L3 = {D_MAX} mm")
print(f"ORIGIN_Y = {ORIGIN_Y}  格子={SQ_W}x{SQ_H}  河宽={RIVER}")
print("=" * 72)
print(f"{'点':>4}  {'X mm':>8}  {'Y mm':>8}  {'R mm':>8}  {'D mm':>8}  {'余量':>8}  可达")
print("-" * 72)
for p in ["e0","a0","i0","e4","a4","i4","e9","a9","i9"]:
    x, y, R, D, margin, ok = check(p)
    print(f"{p:>4}  {x:>8.1f}  {y:>8.1f}  {R:>8.1f}  {D:>8.1f}  {margin:>7.1f}  {'OK' if ok else 'FAIL'}")

# 扫全盘最远点
print()
print("--- 全盘最远点 ---")
worst = ("", 0, 0)
for col_c in "abcdefghi":
    for row in range(10):
        p = f"{col_c}{row}"
        _, _, R, D, _, _ = check(p)
        if D > worst[2]:
            worst = (p, R, D)
print(f"最远: {worst[0]}  R={worst[1]:.1f}  D={worst[2]:.1f}  余量={D_MAX-worst[2]:.1f}")

# BOARD_ORIGIN_Y 上限
print()
print("--- BOARD_ORIGIN_Y 安全上限 ---")
y_extra = 4 * SQ_H + RIVER + 4 * SQ_H  # row0 -> row9
lo, hi = 60, 200
for _ in range(30):
    mid = (lo + hi) / 2
    ya9 = mid + y_extra
    R = math.sqrt(136.8**2 + ya9**2)
    dZ = (BOARD_Z + R * K_z + L4) - L1
    D = math.sqrt(R**2 + dZ**2)
    if D <= D_MAX:
        lo = mid
    else:
        hi = mid
print(f"上限: {lo:.1f} mm  (当前 {ORIGIN_Y}, 可再推远 +{lo - ORIGIN_Y:.0f} mm)")

# 不同 ORIGIN_Y 下的余量
print()
print("--- 不同 ORIGIN_Y 下 a9 的余量 ---")
for oy in range(80, 165, 10):
    ya9 = oy + y_extra
    R = math.sqrt(136.8**2 + ya9**2)
    dZ = (BOARD_Z + R * K_z + L4) - L1
    D = math.sqrt(R**2 + dZ**2)
    m = D_MAX - D
    bar = "#" * max(0, int(m / 2))
    flag = "OK" if m > 0 else "FAIL"
    print(f"  {oy:>4} mm  ->  D={D:.1f}  余量={m:5.1f}  {flag}  {bar}")
