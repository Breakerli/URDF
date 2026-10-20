"""HT-0_58 的 15 自由度 Isaac Lab 资产配置。

以 ht-0_58.urdf 为模型基准，保留拓扑、惯量、限位和碰撞体。
电机上限与 T-N 曲线读取 ht0_58_actuator_cfg.py；PD、armature 沿用原仿真设置。
默认屈膝姿态由目标 URDF 轴向换算；base_link 位于上身，腰关节连接下身。
运行：python legged_lab/scripts/train.py --task ht0_58_run --headless
"""
import math

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.actuators import CustomDelayedPDActuatorCfg
from legged_lab.actuators.ht0_58_actuator_cfg import (
    ARMATURE, DAMPING, EFFORT_LIMIT, MAX_ACTUATOR_DELAY, MIN_ACTUATOR_DELAY,
    STIFFNESS, TORQUE_CURVE_X1, TORQUE_CURVE_X2, TORQUE_CURVE_Y1, TORQUE_CURVE_Y2, VELOCITY_LIMIT,
)
from legged_lab.assets import ISAAC_ASSET_DIR

__all__ = [
    "ROBOT_HT0_58_CFG",
    "ROBOT_HT0_58_DEFAULT_BASE_HEIGHT",
    "ROBOT_HT0_58_DEFAULT_JOINT_POS",
    "ROBOT_HT0_58_JOINT_NAMES",
    "HT0_58_AMP_LOADER_KWARGS",
]

# --- Policy Joint Order ---

# 策略/导出/实机统一顺序：左腿 5、右腿 5、左臂 2、右臂 2、腰 1。
ROBOT_HT0_58_JOINT_NAMES = (
    "hip_roll_l_joint",
    "hip_yaw_l_joint",
    "hip_pitch_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "hip_roll_r_joint",
    "hip_yaw_r_joint",
    "hip_pitch_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "shoulder_pitch_l_joint",
    "elbow_pitch_l_joint",
    "shoulder_pitch_r_joint",
    "elbow_pitch_r_joint",
    "waist_joint",
)

# --- AMP Layout ---

# AMP：右臂 2、左臂 2、右腿 5、左腿 5、腰 1；42 维。
HT0_58_AMP_LOADER_KWARGS = {
    "arm_joint_size": 2,
    "leg_joint_size": 5,
    "waist_joint_size": 1,
    "leg_mirror_flip_indices": (0, 1),
    "arm_mirror_flip_indices": (),
}

# --- Default Pose ---

# 双脚碰撞体底面 FK 距上身根节点约 1.4495 m，保留 5 mm 初始间隙。
ROBOT_HT0_58_DEFAULT_BASE_HEIGHT = 1.455

ROBOT_HT0_58_DEFAULT_JOINT_POS = {
    "hip_roll_l_joint": 0.0,
    "hip_yaw_l_joint": 0.0,
    "hip_pitch_l_joint": math.radians(-15),
    "knee_pitch_l_joint": math.radians(30),
    "ankle_pitch_l_joint": math.radians(-15),
    "hip_roll_r_joint": 0.0,
    "hip_yaw_r_joint": 0.0,
    "hip_pitch_r_joint": math.radians(-15),
    "knee_pitch_r_joint": math.radians(30),
    "ankle_pitch_r_joint": math.radians(-15),
    "shoulder_pitch_l_joint": 0.0,
    "elbow_pitch_l_joint": 0.0,
    "shoulder_pitch_r_joint": 0.0,
    "elbow_pitch_r_joint": 0.0,
    "waist_joint": 0.0,
}

# --- Actuator Params ---


def _keys(prefixes: tuple[str, ...]) -> list[str]:
    """把关节前缀展开成执行器的正则键 ``["hip_roll_.*", ...]``。"""
    return [f"{prefix}_.*" for prefix in prefixes]


def _by_prefix(prefixes: tuple[str, ...], table: dict[str, float]) -> dict[str, float]:
    """把 ``{"hip_roll": 700}`` 展开成正则键 ``{"hip_roll_.*": 700}``。"""
    return {f"{prefix}_.*": table[prefix] for prefix in prefixes}


# --- Articulation ---

ROBOT_HT0_58_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/HT-0_58/usd/ht0_58.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, ROBOT_HT0_58_DEFAULT_BASE_HEIGHT),
        joint_pos=ROBOT_HT0_58_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": CustomDelayedPDActuatorCfg(
            joint_names_expr=_keys(("hip_roll", "hip_yaw", "hip_pitch", "knee_pitch")),
            effort_limit_sim=_by_prefix(("hip_roll", "hip_yaw", "hip_pitch", "knee_pitch"), EFFORT_LIMIT),
            velocity_limit_sim=_by_prefix(("hip_roll", "hip_yaw", "hip_pitch", "knee_pitch"), VELOCITY_LIMIT),
            stiffness=_by_prefix(("hip_roll", "hip_yaw", "hip_pitch", "knee_pitch"), STIFFNESS),
            damping=_by_prefix(("hip_roll", "hip_yaw", "hip_pitch", "knee_pitch"), DAMPING),
            armature=_by_prefix(("hip_roll", "hip_yaw", "hip_pitch", "knee_pitch"), ARMATURE),
            min_delay=MIN_ACTUATOR_DELAY,
            max_delay=MAX_ACTUATOR_DELAY,
            X1=_by_prefix(("hip_roll", "hip_yaw", "hip_pitch", "knee_pitch"), TORQUE_CURVE_X1),
            X2=_by_prefix(("hip_roll", "hip_yaw", "hip_pitch", "knee_pitch"), TORQUE_CURVE_X2),
            Y1=_by_prefix(("hip_roll", "hip_yaw", "hip_pitch", "knee_pitch"), TORQUE_CURVE_Y1),
            Y2=_by_prefix(("hip_roll", "hip_yaw", "hip_pitch", "knee_pitch"), TORQUE_CURVE_Y2),
        ),
        "feet": CustomDelayedPDActuatorCfg(
            joint_names_expr=_keys(("ankle_pitch",)),
            effort_limit_sim=_by_prefix(("ankle_pitch",), EFFORT_LIMIT),
            velocity_limit_sim=_by_prefix(("ankle_pitch",), VELOCITY_LIMIT),
            stiffness=_by_prefix(("ankle_pitch",), STIFFNESS),
            damping=_by_prefix(("ankle_pitch",), DAMPING),
            armature=_by_prefix(("ankle_pitch",), ARMATURE),
            min_delay=MIN_ACTUATOR_DELAY,
            max_delay=MAX_ACTUATOR_DELAY,
            X1=_by_prefix(("ankle_pitch",), TORQUE_CURVE_X1),
            X2=_by_prefix(("ankle_pitch",), TORQUE_CURVE_X2),
            Y1=_by_prefix(("ankle_pitch",), TORQUE_CURVE_Y1),
            Y2=_by_prefix(("ankle_pitch",), TORQUE_CURVE_Y2),
        ),
        "arms": CustomDelayedPDActuatorCfg(
            joint_names_expr=_keys(("shoulder_pitch", "elbow_pitch")),
            effort_limit_sim=_by_prefix(("shoulder_pitch", "elbow_pitch"), EFFORT_LIMIT),
            velocity_limit_sim=_by_prefix(("shoulder_pitch", "elbow_pitch"), VELOCITY_LIMIT),
            stiffness=_by_prefix(("shoulder_pitch", "elbow_pitch"), STIFFNESS),
            damping=_by_prefix(("shoulder_pitch", "elbow_pitch"), DAMPING),
            armature=_by_prefix(("shoulder_pitch", "elbow_pitch"), ARMATURE),
            min_delay=MIN_ACTUATOR_DELAY,
            max_delay=MAX_ACTUATOR_DELAY,
            X1=_by_prefix(("shoulder_pitch", "elbow_pitch"), TORQUE_CURVE_X1),
            X2=_by_prefix(("shoulder_pitch", "elbow_pitch"), TORQUE_CURVE_X2),
            Y1=_by_prefix(("shoulder_pitch", "elbow_pitch"), TORQUE_CURVE_Y1),
            Y2=_by_prefix(("shoulder_pitch", "elbow_pitch"), TORQUE_CURVE_Y2),
        ),
        "waist": CustomDelayedPDActuatorCfg(
            joint_names_expr=_keys(("waist",)),
            effort_limit_sim=_by_prefix(("waist",), EFFORT_LIMIT),
            velocity_limit_sim=_by_prefix(("waist",), VELOCITY_LIMIT),
            stiffness=_by_prefix(("waist",), STIFFNESS),
            damping=_by_prefix(("waist",), DAMPING),
            armature=_by_prefix(("waist",), ARMATURE),
            min_delay=MIN_ACTUATOR_DELAY,
            max_delay=MAX_ACTUATOR_DELAY,
            X1=_by_prefix(("waist",), TORQUE_CURVE_X1),
            X2=_by_prefix(("waist",), TORQUE_CURVE_X2),
            Y1=_by_prefix(("waist",), TORQUE_CURVE_Y1),
            Y2=_by_prefix(("waist",), TORQUE_CURVE_Y2),
        ),
    },
)
