# import math
# import config
#
#
# class KinematicsSolver:
#     @staticmethod
#     def solve_ik(x, y, z):
#         """逆运动学：笛卡尔坐标 -> 关节角度"""
#         theta1 = math.atan2(x, y)  # 注意：x为左右偏移，y为正前方距离
#         R = math.sqrt(x ** 2 + y ** 2)
#
#         # 腕部末端修正：吸盘中心到腕部关节距离为L4
#         # 目标是吸盘在 (x,y,z)，则腕部关节在 (x,y, z+L4)
#         delta_Z = (z + config.L4) - config.L1
#         D = math.sqrt(R ** 2 + delta_Z ** 2)
#
#         # 空间范围校验
#         if D > (config.L2 + config.L3) or D < abs(config.L2 - config.L3):
#             return None
#
#         # 余弦定理计算
#         cos_alpha = (config.L2 ** 2 + D ** 2 - config.L3 ** 2) / (2 * config.L2 * D)
#         cos_gamma = (config.L2 ** 2 + config.L3 ** 2 - D ** 2) / (2 * config.L2 * config.L3)
#
#         alpha = math.acos(max(-1, min(1, cos_alpha)))
#         gamma = math.acos(max(-1, min(1, cos_gamma)))
#         beta = math.atan2(delta_Z, R)
#
#         t1_deg = math.degrees(theta1) + 90.0  # 转换到基准90度
#         t2_deg = math.degrees(beta + alpha)  # 大臂绝对角度
#         t3_deg = t2_deg - (180.0 - math.degrees(gamma))  # 小臂绝对角度
#
#         # 腕部绝对角度强制为 -90 (垂直向下)
#         # 腕部相对于小臂的角度 = -90 - t3_deg
#         t4_relative = -90.0 - t3_deg
#
#         return t1_deg, t2_deg, t3_deg, t4_relative

import math
import config

class KinematicsSolver:
    @staticmethod
    def solve_ik(x, y, z):
        # ==========================================
        # [新增] X轴独立前馈补偿
        # 放大目标 X 坐标，强制算法同步增加旋转角和伸展半径，维持 Y 不变
        # 调参基准：假设目标到 d0，实际偏内侧 10%，则 K_x 设为 1.1
        # ==========================================
        K_x = 1.125  # 建议先从 1.15 开始调参
        comp_x = x * K_x

        theta1 = math.atan2(comp_x, y)
        R_target = math.sqrt(comp_x ** 2 + y ** 2)

        # ------------------------------------------
        # 以下保留上一轮的重力和虚位补偿逻辑
        # ------------------------------------------
        K_r = 0.06
        R_comp = R_target * (1 - K_r)

        K_z = 0.12  
        comp_z = z + (R_target * K_z)

        K_a = 0.0  
        wrist_comp_angle = R_target * K_a

        # 腕部末端修正
        delta_Z = (comp_z + config.L4) - config.L1
        D = math.sqrt(R_comp ** 2 + delta_Z ** 2)

        if D > (config.L2 + config.L3) or D < abs(config.L2 - config.L3):
            return None

        # 余弦定理计算
        cos_alpha = (config.L2 ** 2 + D ** 2 - config.L3 ** 2) / (2 * config.L2 * D)
        cos_gamma = (config.L2 ** 2 + config.L3 ** 2 - D ** 2) / (2 * config.L2 * config.L3)

        alpha = math.acos(max(-1, min(1, cos_alpha)))
        gamma = math.acos(max(-1, min(1, cos_gamma)))
        beta = math.atan2(delta_Z, R_comp)

        t1_deg = math.degrees(theta1) + 90.0
        t2_deg = math.degrees(beta + alpha)
        t3_deg = t2_deg - (180.0 - math.degrees(gamma))

        t4_relative = -90.0 - t3_deg + wrist_comp_angle

        return t1_deg, t2_deg, t3_deg, t4_relative

    @staticmethod
    def angle_to_pwm(t1, t2, t3, t4_rel):
        """将角度映射到标定的脉宽"""
        # ==========================================
        # [新增] 底座舵机行程比例增益 (Base Gain)
        # ==========================================
        # 补偿物理舵机满量程非 270度 导致的转向角偏小问题。
        # 调参逻辑：如果在边缘列够不到，请增大此值。
        # 假设该舵机实际为 180 度，理论值应约为 270/180 = 1.5。
        base_gain = 1.0  # 起始测试值，需根据实际偏离量微调
        
        # 底座：单独应用 base_gain 进行脉宽变化量的放大
        us0 = config.SERVO_CONFIG[0]['ref_us'] - (t1 - 90.0) * config.DEG_TO_US * base_gain
        
        # 其他连杆保持原样
        us4 = config.SERVO_CONFIG[4]['ref_us'] - (t2 - 90.0) * config.DEG_TO_US
        us8 = config.SERVO_CONFIG[8]['ref_us'] + (t3 - t2) * config.DEG_TO_US
        us12 = config.SERVO_CONFIG[12]['ref_us'] + (t4_rel - 0.0) * config.DEG_TO_US

        return us0, us4, us8, us12