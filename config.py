# 连杆长度 (mm)
L1 = 73.0
L2 = 231.0
L3 = 231.0
L4 = 66.0

# 舵机通道与标定基准 (三连杆全部竖直向上 90° 时的脉宽)
SERVO_CONFIG = {
    0:  {'ref_us': 1670, 'min': 500,  'max': 2500, 'name': 'Base'},
    4:  {'ref_us': 1550, 'min': 1200, 'max': 2200, 'name': 'Shoulder'},#1595
    8:  {'ref_us': 1675, 'min': 680,  'max': 2500, 'name': 'Elbow'},#1675
    12: {'ref_us': 1495, 'min': 500,  'max': 2500, 'name': 'Wrist'},
    #15: {'ref_us': 1500, 'min': 500,  'max': 2500, 'name': 'Magnet'} # 假设电磁铁在15
}

# 舵机特性
DEG_TO_US = 2000.0 / 270.0

# 棋盘几何参数 (可随时修改)
SQUARE_WIDTH = 34.2      # 左右间隔
SQUARE_HEIGHT = 34.2    # 前后间隔
RIVER_HEIGHT = 34.0      # 楚河汉界高度
#BOARD_ORIGIN_Y = 107.0-8   # e0 格点中心距离底座的距离  115-8mm+4mm
BOARD_ORIGIN_Y = 200.0
BOARD_Z = -33.0            # 棋子表面高度-15
SAFE_Z = 70.0           # 平移时的安全高度