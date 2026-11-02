# 连杆长度 (mm)
L1 = 73.0
L2 = 231.0
L3 = 231.0
L4 = 66.0

# 舵机通道与标定基准 (三连杆全部竖直向上 90° 时的脉宽)
SERVO_CONFIG = {
    0:  {'ref_us': 1670, 'min': 500,  'max': 2500, 'name': 'Base'},
    4:  {'ref_us': 1550, 'min': 1200, 'max': 2200, 'name': 'Shoulder'}, # 1595
    8:  {'ref_us': 1675, 'min': 680,  'max': 2500, 'name': 'Elbow'},    # 1675
    12: {'ref_us': 1495, 'min': 500,  'max': 2500, 'name': 'Wrist'},
}

# 舵机特性
DEG_TO_US = 2000.0 / 270.0

# 棋盘几何物理参数 (基于用户实测标定)
BOARD_WIDTH_MM = 276.0     # 棋盘横向总跨度 (mm, a列至i列共8个格距)
BOARD_HEIGHT_MM = 307.0    # 棋盘纵向总跨度 (mm, 0行至9行共9个格距含楚河汉界)
SQUARE_WIDTH = round(BOARD_WIDTH_MM / 8.0, 2)     # 左右格间距 = 34.50 mm
SQUARE_HEIGHT = round(BOARD_HEIGHT_MM / 9.0, 2)   # 前后格间距 = 34.11 mm
RIVER_HEIGHT = SQUARE_HEIGHT                      # 楚河汉界高度 = 34.11 mm

# 机械臂物理原点标定 (机械臂底座旋转轴心距离黑方底线 e0 格点的纵向距离)
# 实测值约为 110~115 mm，设置为 115.0 mm 时全盘 90 格点及 ±10mm 偏移 100% 处于可达工作空间
BOARD_ORIGIN_Y = 115.0
BOARD_Z = -33.0            # 棋子表面高度
SAFE_Z = 70.0             # 平移时的安全高度

# 棋子与末端执行器物理尺寸
PIECE_DIAMETER_MM = 25.0         # 棋子外径约为 25mm (半径 12.5mm)
ELECTROMAGNET_RADIUS_MM = 8.0    # 电磁铁吸盘有效吸附半径约 8mm (偏置需控制在吸盘半径内)
