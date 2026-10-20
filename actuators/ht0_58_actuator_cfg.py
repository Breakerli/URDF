"""HT-0_58 专用关节电机与 T-N 参数，训练、Play 和 MuJoCo 共用。

沿用 actuator_cfg.py 的 CustomDelayedPDActuatorCfg：X1/X2 为平台拐点/零力矩速度
（rad/s），Y1/Y2 为同向/反向力矩（N·m）。上限取投产配套表，曲线取测试表；
膝同髋 pitch，踝授权借用 7216 数据，肘为规格近似。PD、armature 保留原仿真设置。
Isaac 资产正常导入；离线脚本通过 runpy.run_path 读取，避免加载执行器包的 Isaac 依赖。
用法：python -c "import runpy; print(runpy.run_path('legged_lab/actuators/ht0_58_actuator_cfg.py')['EFFORT_LIMIT'])"
"""

# --- Imports ---
import math

# --- Constants & Config ---
# 前缀对应双侧同类关节；waist 对应唯一腰关节。表格未提供 PD、armature 标定。
ARMATURE = {'hip_yaw': 0.048865,
 'hip_roll': 0.40318,
 'hip_pitch': 0.40318,
 'knee_pitch': 0.40318,
 'ankle_pitch': 0.21448,
 'shoulder_pitch': 0.041125,
 'elbow_pitch': 0.041125,
 'waist': 0.048865}
EFFORT_LIMIT = {'hip_yaw': 200,
 'hip_roll': 320,
 'hip_pitch': 400,
 'knee_pitch': 400,
 'ankle_pitch': 160,
 'shoulder_pitch': 50,
 'elbow_pitch': 12,
 'waist': 200}
VELOCITY_LIMIT = {'hip_yaw': 26.07521902479528,
 'hip_roll': 24.609142453120047,
 'hip_pitch': 24.609142453120047,
 'knee_pitch': 24.609142453120047,
 'ankle_pitch': 27.64601535159018,
 'shoulder_pitch': 35.29055747532534,
 'elbow_pitch': 23.038346126325152,
 'waist': 26.07521902479528}
STIFFNESS = {'hip_yaw': 500.0,
 'hip_roll': 700.0,
 'hip_pitch': 700.0,
 'knee_pitch': 700.0,
 'ankle_pitch': 100.0,
 'shoulder_pitch': 50.0,
 'elbow_pitch': 50.0,
 'waist': 400.0}
DAMPING = {'hip_yaw': 10.0,
 'hip_roll': 20.0,
 'hip_pitch': 10.0,
 'knee_pitch': 10.0,
 'ankle_pitch': 6.0,
 'shoulder_pitch': 3.0,
 'elbow_pitch': 3.0,
 'waist': 8.0}

# 实测 T-N 与配套表的速度上限分别保存；10825 Run02 为 80V 测试，不做电压缩放。
TORQUE_CURVE_X1 = {'hip_yaw': 17.15146236514016,
 'hip_roll': 13.444816887849663,
 'hip_pitch': 13.647542605148752,
 'knee_pitch': 13.647542605148752,
 'ankle_pitch': 16.267845134861613,
 'shoulder_pitch': 15.075852939617878,
 'elbow_pitch': 17.59291886010284,
 'waist': 17.15146236514016}
TORQUE_CURVE_X2 = {'hip_yaw': 28.98433382201943,
 'hip_roll': 26.0710302345905,
 'hip_pitch': 27.358036025011117,
 'knee_pitch': 27.358036025011117,
 'ankle_pitch': 26.56844907140888,
 'shoulder_pitch': 34.3805428033355,
 'elbow_pitch': 23.038346126325152,
 'waist': 28.98433382201943}
TORQUE_CURVE_Y1 = {'hip_yaw': 194.0,
 'hip_roll': 310.7,
 'hip_pitch': 400,
 'knee_pitch': 400,
 'ankle_pitch': 151.5,
 'shoulder_pitch': 47.0,
 'elbow_pitch': 12.0,
 'waist': 194.0}
# 没有再生制动测试，采用同一条对称包络，不额外提高反向力矩。
TORQUE_CURVE_Y2 = dict(TORQUE_CURVE_Y1)
MIN_ACTUATOR_DELAY = 0
MAX_ACTUATOR_DELAY = 4

# 配套型号和上限来自投产配套表；实测曲线取各 sheet 的 A17:B末行。
# 固定 Y1=min(实测峰值, 配套峰值)、X2=实测最高转速；X1 在 [0,X2) 等距搜索 20001 点，
# 以所有实测点的力矩均方误差最小为准；肘采用下面明确标注的规格近似。
# A7225 页的“不纳入”标记已由用户授权覆盖，仅用于本次近似仿真配置。
MOTOR_DATA = {'hip_yaw': {'model': 'EC-A7225-P2-22H-A',
             'source': '190-EC-A7225-P2-22H（汇总）.xlsx',
             'sheet': 'T-N曲线',
             'X1_rpm': 163.784401,
             'X2_rpm': 276.78,
             'fit_rmse_nm': 7.323376},
 'hip_roll': {'model': 'EC-A9025-P2-23.625H-B',
              'source': '310-EC-A9025-P2-23.625H（汇总）.xlsx',
              'sheet': 'T-N曲线',
              'X1_rpm': 128.388544,
              'X2_rpm': 248.96,
              'fit_rmse_nm': 9.983288},
 'hip_pitch': {'model': 'EC-A10825-P2-24H-V1.5-72V-水冷',
               'source': 'EC-10825_P2-24水冷.xlsx',
               'sheet': 'T-N曲线Run02',
               'X1_rpm': 130.324432,
               'X2_rpm': 261.25,
               'fit_rmse_nm': 12.018284},
 'shoulder_pitch': {'model': 'EC-A5025-P2-17H-B',
                    'source': 'EC-A5025-P2-17H(2).xlsx',
                    'sheet': 'T-N曲线Run02',
                    'X1_rpm': 143.963791,
                    'X2_rpm': 328.31,
                    'fit_rmse_nm': 1.099127},
 'waist': {'model': 'EC-A7225-P2-22H-A',
           'source': '190-EC-A7225-P2-22H（汇总）.xlsx',
           'sheet': 'T-N曲线',
           'X1_rpm': 163.784401,
           'X2_rpm': 276.78,
           'fit_rmse_nm': 7.323376},
 'knee_pitch': {'model': 'EC-A10825-P2-24H-V1.5-72V-水冷',
                'source': 'EC-10825_P2-24水冷.xlsx',
                'sheet': 'T-N曲线Run02',
                'X1_rpm': 130.324432,
                'X2_rpm': 261.25,
                'fit_rmse_nm': 12.018284,
                'note': '用户确认与髋 pitch 同型号'},
 'ankle_pitch': {'model': 'EC-A7220-P2-22H-B',
                 'source': '150-EC-7216-P2-22H(汇总).xlsx',
                 'sheet': 'T-N曲线',
                 'X1_rpm': 155.346478,
                 'X2_rpm': 253.71,
                 'fit_rmse_nm': 12.986615,
                 'note': '用户授权以 7216 测试曲线近似 A7220；上限取 A7220 配套规格'},
 'elbow_pitch': {'model': 'EC-A2806-P2-36-72v',
                 'source': '第三套机器人产品配套表2-关节-20261016—投产定型 -因克斯.xlsx',
                 'sheet': '硬件清单',
                 'X1_rpm': 168.0,
                 'X2_rpm': 220.0,
                 'fit_rmse_nm': None,
                 'note': '规格近似，非实测；峰值 12Nm，直线通过额定点 (207rpm,3Nm) 和 (220rpm,0Nm)，故 X1=168rpm'}}

# --- Validation ---
# 导入时检查每类关节完整覆盖、单位换算后的正值与曲线定义域。
for _table in (ARMATURE, VELOCITY_LIMIT, STIFFNESS, DAMPING, TORQUE_CURVE_X1, TORQUE_CURVE_X2,
               TORQUE_CURVE_Y1, TORQUE_CURVE_Y2, MOTOR_DATA):
    if set(_table) != set(EFFORT_LIMIT):
        raise ValueError("HT-0_58 关节电机参数表的键不一致")
for _kind in EFFORT_LIMIT:
    _values = [table[_kind] for table in (EFFORT_LIMIT, VELOCITY_LIMIT, STIFFNESS, DAMPING,
               TORQUE_CURVE_X2, TORQUE_CURVE_Y1, TORQUE_CURVE_Y2)]
    if not all(math.isfinite(value) and value > 0 for value in _values):
        raise ValueError(f"HT-0_58 {_kind} 的力矩/速度/PD 参数必须是正有限数")
    if not (math.isfinite(ARMATURE[_kind]) and ARMATURE[_kind] >= 0
            and math.isfinite(TORQUE_CURVE_X1[_kind]) and 0 <= TORQUE_CURVE_X1[_kind] < TORQUE_CURVE_X2[_kind]
            and max(TORQUE_CURVE_Y1[_kind], TORQUE_CURVE_Y2[_kind]) <= EFFORT_LIMIT[_kind]):
        raise ValueError(f"HT-0_58 {_kind} 的 armature 或 T-N 曲线参数无效")
