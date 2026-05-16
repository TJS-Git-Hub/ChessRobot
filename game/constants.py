# 棋盘坐标规则：X轴0-8（左→右），Y轴0-9（上→下）
X_RANGE = 8  # x最大坐标
Y_RANGE = 9  # y最大坐标

# 初始化棋盘 10行9列
INIT_BOARD = [
    [("black", "ju"), ("black", "ma"), ("black", "xiang"), ("black", "shi"), ("black", "jiang"), ("black", "shi"), ("black", "xiang"), ("black", "ma"), ("black", "ju")],
    [None] * 9,
    [None, ("black", "pao"), None, None, None, None, None, ("black", "pao"), None],
    [("black", "zu"), None, ("black", "zu"), None, ("black", "zu"), None, ("black", "zu"), None, ("black", "zu")],
    [None] * 9,
    [None] * 9,
    [("red", "bing"), None, ("red", "bing"), None, ("red", "bing"), None, ("red", "bing"), None, ("red", "bing")],
    [None, ("red", "pao"), None, None, None, None, None, ("red", "pao"), None],
    [None] * 9,
    [("red", "ju"), ("red", "ma"), ("red", "xiang"), ("red", "shi"), ("red", "shuai"), ("red", "shi"), ("red", "xiang"), ("red", "ma"), ("red", "ju")]
]