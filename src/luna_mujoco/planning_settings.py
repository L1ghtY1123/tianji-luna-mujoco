"""One validated settings snapshot for the UI, both IK solvers, planning and execution."""
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator

from .observation_motion import observation_joint_limits, observation_poses
from .vla_configuration import default_vla_input_size

CUROBO_DEFAULTS = {
    'ik_num_seeds': 32, 'ik_seed_solver_num_seeds': 32,
    'ik_position_threshold_m': .005, 'ik_rotation_threshold_rad': .05,
    'random_seed': 123, 'self_collision_check': True,
    'trajectory_collision_clearance_m': 0., 'optimizer_collision_activation_distance_m': .01,
    'trajopt_num_seeds': 4, 'plan_max_attempts': 20, 'finetune_attempts': 1,
    'finetune_dt_scale': .6, 'freq_hz': 40,
    'approach_axis': 'z', 'approach_offset_m': -.15, 'approach_in_tool_frame': True,
    'place_approach_axis': 'z', 'place_approach_offset_m': -.15, 'place_approach_in_tool_frame': True,
    'place_retreat_axis': 'z', 'place_retreat_offset_m': -.15, 'place_retreat_in_tool_frame': True,
}

GRASP_LIFT_POLICY = {'lift_axis': 'z', 'lift_offset_m': .15, 'lift_in_tool_frame': False}

RELEASE_ENDPOINT_POSITION_TOLERANCE_M = .005


def setting(default, title, description, group, *, unit="", scale=1, step=1, **limits):
    return Field(default=default, title=title, description=description,
                 json_schema_extra={"group": group, "unit": unit, "display_scale": scale, "step": step},
                 **limits)


class PointcloudFilterSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    enabled: bool = setting(False, "启用可见点云过滤", "默认关闭；只过滤实测点，不保证识别密集浮岛。", "filter")
    method: Literal["statistical", "radius"] = setting("statistical", "点云过滤方法", "统计离群点过滤或半径邻居过滤，择一执行。", "filter")
    nb_neighbors: int = setting(20, "统计邻居数", "统计过滤使用的邻居数量。", "filter", ge=2, le=1000, strict=True)
    std_ratio: float = setting(2., "统计标准差倍率", "较小数值会剔除更多实测点；需检查薄壁及稀疏表面。", "filter", step=.1, gt=0, le=20)
    radius_m: float = setting(.01, "半径过滤邻域", "半径法使用的测量点邻域，须按实际点间距验证。", "filter", unit="mm", scale=1000, step=1, gt=0, le=.5)
    min_neighbors: int = setting(4, "半径过滤最少邻居", "半径法保留点所需的邻居数量。", "filter", ge=1, le=1000, strict=True)
    dbscan_preview: bool = setting(False, "DBSCAN 点簇诊断", "只提供点簇预览，不按点簇或桌面连通性删除障碍。", "filter")
    dbscan_eps_m: float = setting(.02, "DBSCAN 邻域", "仅用于点簇诊断。", "filter", unit="mm", scale=1000, step=1, gt=0, le=.5)
    dbscan_min_points: int = setting(10, "DBSCAN 最少点数", "仅用于点簇诊断。", "filter", ge=1, le=1000, strict=True)


class AttachmentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    minimum_partition_size_m: float = setting(.01, "持物球最小分组尺寸", "空间分组停止尺度，不是TSDF/ESDF体素；实际地图统一20mm。沿各点组最大方差方向分组并比较有限球心，达到分组尺度或球预算即停止；每组覆盖原点与不确定度。", "attachment", unit="mm", scale=1000, step=1, ge=.001, le=.05)
    margin_m: float = setting(.001, "持物球附加余量", "附加到覆盖实测点的球半径；不表示遮挡面或完整物体形状。", "attachment", unit="mm", scale=1000, step=1, ge=0, le=.03)
    sphere_budget: int = setting(24, "持物球数量上限", "允许1–31个球，默认24。分组覆盖全部实测点并保留测量不确定度和附加余量；球数少时包络更粗，同手接触组件允许与附件重叠，上游臂和世界仍检查。", "attachment", ge=1, le=31, strict=True)

    @model_validator(mode="before")
    @classmethod
    def migrate_partition_name(cls, values):
        if isinstance(values, dict) and "voxel_size_m" in values:
            values = dict(values)
            legacy = values.pop("voxel_size_m")
            if "minimum_partition_size_m" in values and values["minimum_partition_size_m"] != legacy:
                raise ValueError("conflicting attachment partition sizes")
            values["minimum_partition_size_m"] = legacy
        return values


JointPositions = tuple[StrictFloat, StrictFloat, StrictFloat, StrictFloat, StrictFloat, StrictFloat, StrictFloat]


def observation_joint_field(arm):
    fields = [{"type": "number", "title": f"{arm} 臂关节 {index}",
               "description": "双臂观测的关节目标；直接规划到该关节配置，保留碰撞检查。",
               "minimum": lower, "maximum": upper, "group": "observation",
               "unit": "°", "display_scale": 180 / math.pi, "step": .1}
              for index, (lower, upper) in enumerate(observation_joint_limits()[arm], start=1)]
    return Field(default_factory=lambda: tuple(observation_poses()[arm]), json_schema_extra={"prefixItems": fields})


class ObservationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    A: JointPositions = observation_joint_field("A")
    B: JointPositions = observation_joint_field("B")

    @model_validator(mode="after")
    def validate_robot_limits(self):
        observation_poses({"observation": {"A": self.A, "B": self.B}})
        return self


def vla_input_field(axis):
    from .vla_configuration import default_vla_input_xy
    return Field(default_factory=lambda: default_vla_input_xy()[axis],
        title=f"VLA 输入位置 {'X' if axis == 0 else 'Y'}",
        description="世界坐标，单位米。指定 VLA 桌面正方形区域的中心；区域内自主选点，保留既有姿态。",
        json_schema_extra={'group': 'vla', 'unit': 'm', 'display_scale': 1, 'step': .001})


class VLAUnpackSettings(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)
    input_x_m: float = vla_input_field(0)
    input_y_m: float = vla_input_field(1)
    input_size_m: float = Field(default_factory=default_vla_input_size, ge=.02, le=.5,
        title='VLA 输入区域边长', description='无可见标记的桌面正方形边长；与 X/Y 共同定义数值放置边界。',
        json_schema_extra={'group': 'vla', 'unit': 'm', 'display_scale': 1, 'step': .01})


class PlanningSettings(BaseModel):
    experimental_semantic_only_mapping: bool = setting(True, "实验场景仅保留已识别物体",
        "仅在双臂观测位姿分析完成后，保留SAM3物体/容器及桌面；删除未分类背景。初始运动地图不使用此规则。",
        "filter")
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    holding_position_tolerance_fraction: float = setting(.02, "夹爪稳定位置容差", "执行稳定窗口内允许的位置变化，占有效行程的比例。", "verification", ge=.001, le=.1, step=.005)
    holding_stable_window_s: float = setting(.2, "夹爪稳定窗口", "用于独立夹爪执行命令的稳定性诊断。", "verification", unit="s", ge=.05, le=1., step=.05)
    gripper_stable_velocity_rad_s: float = setting(.015, "夹爪窗口稳定速度上限", "完整稳定窗口内驱动关节速度的时间加权均方根不超过此值，且位置变化满足容差才返回；超时失败。", "verification", unit="rad/s", step=.005, gt=0, le=.2)
    gripper_timeout_s: float = setting(3., "夹爪稳定超时", "独立夹爪命令的仿真等待上限。视觉节点负责夹持结果。", "verification", unit="s", step=.1, gt=0, le=15)
    motion_position_tolerance_rad: float = setting(.005, "运动到位误差", "轨迹结束后关节位置误差上限；不改变沿途跟踪保护。", "execution", unit="°", scale=180 / math.pi, step=.1, gt=0, le=.02)
    motion_stable_velocity_rad_s: float = setting(.03, "运动到位速度上限", "末端关节速度连续满足要求才返回。", "execution", unit="rad/s", step=.01, gt=0, le=.2)
    motion_stable_window_s: float = setting(.08, "运动到位稳定窗口", "仿真时间；短暂到位后再次偏离会重新计时。", "execution", unit="s", step=.02, ge=.02, le=1)
    motion_settle_timeout_s: float = setting(3., "运动到位超时", "轨迹结束后的最大等待时间；超时失败。", "execution", unit="s", step=.1, gt=0, le=15)
    grasp_goalset_size: int = setting(32, "抓取候选批大小", "每次 goalset 请求的候选上限，通过完整数值检查的首条轨迹结束搜索。", "grasp", ge=1, le=128, strict=True)
    cad_grasp_insertion_m: float = setting(0., "CAD 抓取伸入深度", "CAD 离线库已设定抓取深度，默认额外伸入 0mm。工作台显式调整后，沿执行臂末端局部 +Z 在 IK 前应用一次。仅工作台配置可修改，Responses 无覆盖权限。", "grasp", unit="mm", scale=1000, step=1, ge=-.02, le=.03)
    grasp_maximum_tilt_deg: float = setting(180., "抓姿最大倾角", "相对世界向下轴；180°不按倾角排除，最终由数值碰撞与 IK 判断。", "grasp", unit="°", step=5, ge=0, le=180)
    grasp_require_away_from_robot: bool = setting(False, "只保留背离机身的抓取方向", "可选方向约束；默认关闭，不能代替碰撞检查。", "grasp")
    grasp_maximum_lateral_angle_deg: float = setting(30., "抓姿最大左右偏角", "GraspGen 末端 +Z 限于世界 +X 前方至 −Z 下方扇区，此项限制偏离世界 XZ 平面的角度。CAD 离线抓姿使用已设定方向。", "grasp", unit="°", step=1, ge=0, le=45)

    ik_num_seeds: int = setting(32, "IK 优化 seeds", "官方 plan_grasp 内部 goalset IK 的优化起点数。",
                               "shared", ge=1, le=64, strict=True)
    ik_seed_solver_num_seeds: int = setting(32, "IK 初始解 seeds（LM）", "LM 初始解搜索数；官方默认 32。项目要求 ≥ IK seeds，避免 SDK 自动扩大搜索数。",
                                           "shared", ge=1, le=128, strict=True)
    ik_position_threshold_m: float = setting(.005, "IK 位置容差", "TCP 位置误差的成功阈值，越小越严格。",
                                             "shared", unit="mm", scale=1000, step=.1, ge=.0001, le=.02)
    ik_rotation_threshold_rad: float = setting(.05, "IK 方向容差", "TCP 方向误差的成功阈值；API 使用 rad。",
                                               "shared", unit="°", scale=180 / math.pi, step=.1, ge=.001, le=.35)
    random_seed: int = setting(123, "random seed", "两处 IK 和 TrajOpt 使用相同随机数种子；GPU 计算不保证逐位复现。",
                              "shared", ge=0, le=2147483647, strict=True)
    self_collision_check: bool = setting(True, "自碰撞检查", "启用机器人自身碰撞检查；关闭会降低规划约束。", "shared")
    trajectory_collision_clearance_m: float = setting(0., "碰撞间距（hard clearance）",
        "普通环境及非接触链接的硬净距。指腹目标接触及已测支撑接触单独处理；不是全局禁止接触距离。",
        "shared", unit="mm", scale=1000, ge=0, le=.5)

    optimizer_collision_activation_distance_m: float = setting(.01, "碰撞代价激活距离（soft activation）",
        "TrajOpt 在该距离内产生避障代价，不等于 hard clearance；必须不小于 hard clearance。",
        "planner", unit="mm", scale=1000, ge=0, le=.5)
    trajopt_num_seeds: int = setting(4, "TrajOpt seeds", "每目标轨迹优化起点数；本项目采用4，以提高包含抓取目标时的路径求解覆盖。锁定官方batch默认2。要求 ≤ IK seeds，避免 SDK 自动上调。",
                                    "planner", ge=1, le=8, strict=True)
    plan_max_attempts: int = setting(20, "最大规划 attempts", "普通plan_pose/plan_cspace的公开尝试预算。plan_grasp内部阶段采用官方默认，公共接口未暴露此参数。",
                                     "planner", ge=1, le=100, strict=True)
    finetune_attempts: int = setting(1, "finetune attempts", "仅批量普通plan_pose公开支持；单臂plan_pose及plan_grasp内部阶段不覆盖。",
                                    "planner", ge=1, le=10, strict=True)
    finetune_dt_scale: float = setting(.6, "finetune dt scale", "仅批量普通plan_pose公开支持；plan_grasp内部用官方默认。独立于执行time_scale。",
                                       "planner", step=.05, gt=0, le=1)
    freq_hz: int = setting(40, "SDK 轨迹采样频率", "实际映射 cuRobo interpolation_dt=1/frequency；不是 MuJoCo 物理步频。",
                           "planner", unit="Hz", ge=20, le=100, strict=True)

    approach_axis: Literal["x", "y", "z"] = setting("z", "Approach 轴", "生成接近位姿的坐标轴。", "grasp")
    approach_offset_m: float = setting(-.15, "Approach 偏移", "沿指定坐标轴相对抓取位姿偏移；不能为零。",
                                       "grasp", unit="mm", scale=1000, ge=-.5, le=.5)
    approach_in_tool_frame: bool = setting(True, "Approach 使用 tool frame", "开启为 tool frame；关闭为 world frame。", "grasp")
    lift_axis: Literal["x", "y", "z"] = setting("z", "Lift 轴", "抓取提升固定使用世界 Z 轴。", "grasp")
    lift_offset_m: float = setting(.15, "Lift 偏移", "抓取后沿世界 +Z 提升 150 mm。",
                                   "grasp", unit="mm", scale=1000, gt=0, le=.5)
    lift_in_tool_frame: bool = setting(False, "Lift 使用 tool frame", "抓取提升使用世界坐标系。", "grasp")

    place_approach_axis: Literal['x', 'y', 'z'] = setting('z', '放置接近轴', '官方默认', 'place')
    place_approach_offset_m: float = setting(-.15, '放置接近偏移', '官方默认', 'place', unit='mm', scale=1000)
    place_approach_in_tool_frame: bool = setting(True, '接近使用末端坐标', '官方默认', 'place')
    place_retreat_axis: Literal['x', 'y', 'z'] = setting('z', '撤离轴', '官方默认', 'place')
    place_retreat_offset_m: float = setting(-.15, '撤离偏移', '官方默认', 'place', unit='mm', scale=1000)
    place_retreat_in_tool_frame: bool = setting(True, '撤离使用末端坐标', '官方默认', 'place')

    container_release_bottom_clearance_m: float = setting(.02, "容器释放高度余量", "末端 Z = 实测盒口最高点 + 物体高度 + 余量；默认20mm，含为保守附着包络增加的10mm。", "place", unit="mm", scale=1000, step=1, ge=0, le=1)
    container_release_fixed_xy: bool = setting(True, "容器使用配置释放区域", "container_release 使用下方配置的固定 XY，目标盒必须包含此区域。", "place")
    container_release_x_m: float = setting(.430, "容器释放 X", "机器人基坐标系中的固定物体参考点 X。", "place", unit="mm", scale=1000, step=1, ge=-2, le=2)
    container_release_y_m: float = setting(-.560, "容器释放 Y", "机器人基坐标系中的固定物体参考点 Y。", "place", unit="mm", scale=1000, step=1, ge=-2, le=2)
    container_release_region_size_m: float = setting(.12, "容器释放区域边长", "以配置 XY 为中心的正方形区域，保留实测容器边界和碰撞检查。", "place", unit="mm", scale=1000, step=1, gt=0, le=1)
    place_bottom_clearance_m: float = setting(.02, "桌面释放高度余量", "末端 Z = 实测桌面 Z + 物体高度 + 余量；默认20mm，含为保守附着包络增加的10mm。",
                                       "place", unit="mm", scale=1000, step=1, ge=0, le=1)


    gripper_transition_s: float = setting(.4, "连续动作夹爪开合时长", "三段轨迹中预留的夹爪驱动时间，不等待夹持反馈或视觉判断。", "execution", unit="s", step=.05, gt=0, le=3)
    time_scale: float = setting(3., "时间伸长／补点倍率（三段共用）", "使用五次 Hermite 插值伸长执行时间，默认 3 倍。",
                                "execution", unit="倍", step=.1, ge=1, le=10)
    pointcloud_filter: PointcloudFilterSettings = Field(default_factory=PointcloudFilterSettings)
    observation: ObservationSettings = Field(default_factory=ObservationSettings)
    attachment: AttachmentSettings = Field(default_factory=AttachmentSettings)
    vla_unpack: VLAUnpackSettings = Field(default_factory=VLAUnpackSettings)

    @model_validator(mode="before")
    @classmethod
    def migrate_saved_settings(cls, values):
        if not isinstance(values, dict):
            return values
        migrated = dict(values)
        # GraspGen already returns its contact TCP. Align it with the actual
        # arm's front TCP; a saved legacy insertion must never move it again.
        migrated.pop('grasp_insertion_m', None)
        # The fixed-open pregrasp place preflight was removed. Retire its
        # saved option instead of exposing an advanced setting with no effect.
        migrated.pop("place_max_candidates", None)
        migrated.pop("place_max_release_gap_m", None)
        for retired in ('dwell_approach_s', 'dwell_grasp_s', 'dwell_lift_s', 'dwell_place_approach_s',
                        'dwell_place_s', 'dwell_release_s', 'dwell_retreat_s'):
            migrated.pop(retired, None)
        for legacy_name in ("release_margin_m", "place_release_margin_m", "plan_timeout_s", "place_timeout_s"):
            migrated.pop(legacy_name, None)
        migrated.update(CUROBO_DEFAULTS)
        migrated.update(GRASP_LIFT_POLICY)
        migrated.pop('container_release_gap_m', None)
        migrated.pop('place_release_height_m', None)
        migrated.pop('place_tcp_z_m', None)
        migrated.pop('container_release_tcp_z_m', None)
        # Old values described TCP height, not object-bottom clearance.
        migrated.pop('place_tcp_height_m', None)
        migrated.pop('container_release_tcp_height_m', None)
        migrated.update(container_release_fixed_xy=True)
        return migrated

    @model_validator(mode="after")
    def validate_relationships(self):
        if self.gripper_timeout_s <= self.holding_stable_window_s:
            raise ValueError("夹爪超时必须大于稳定窗口")
        if self.motion_settle_timeout_s <= self.motion_stable_window_s:
            raise ValueError("运动到位超时必须大于稳定窗口")
        if self.trajopt_num_seeds > self.ik_num_seeds:
            raise ValueError("TrajOpt seeds 不能大于 IK 优化 seeds，否则 SDK 会增加内部 IK seeds")
        if self.ik_num_seeds > self.ik_seed_solver_num_seeds:
            raise ValueError("IK 优化 seeds 不能大于 LM seeds，否则 SDK 会增加实际 LM seeds")
        if self.optimizer_collision_activation_distance_m < self.trajectory_collision_clearance_m:
            raise ValueError("soft activation 必须不小于 hard clearance")
        if self.approach_offset_m == 0 or self.lift_offset_m == 0:
            raise ValueError("Approach / Lift 偏移不能为零")
        return self

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        schema = super().model_json_schema(*args, **kwargs)
        for name in CUROBO_DEFAULTS:
            schema['properties'][name]['readOnly'] = True
            schema['properties'][name]['official_default'] = True
            schema['properties'][name]['description'] = 'cuRobo v0.9.0 official default'
        for name in ('container_release_fixed_xy',):
            schema['properties'][name]['readOnly'] = True
        for name in GRASP_LIFT_POLICY:
            schema['properties'][name]['readOnly'] = True
        return schema

    def revision(self):
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]

    def solver_payload(self):
        names = ("ik_num_seeds", "ik_seed_solver_num_seeds", "ik_position_threshold_m",
                 "ik_rotation_threshold_rad", "random_seed", "self_collision_check",
                 "trajectory_collision_clearance_m", "optimizer_collision_activation_distance_m",
                 "trajopt_num_seeds", "plan_max_attempts", "finetune_attempts", "finetune_dt_scale", "freq_hz")
        values = self.model_dump()
        return {name: values[name] for name in names}


def load_settings(path: Path):
    if not path.exists():
        return PlanningSettings()
    values = json.loads(path.read_text(encoding="utf-8"))
    attachment = values.get("attachment", {}) if isinstance(values, dict) else {}
    budget = attachment.get("sphere_budget") if isinstance(attachment, dict) else None
    if type(budget) is int and 32 <= budget <= 256:
        # Migrate saved configuration only; never silently change a live held
        # binding or accept an out-of-policy request from the settings API.
        warnings.warn(f"saved attachment sphere_budget {budget} migrated to 24 under the 31-ball policy",
                      UserWarning, stacklevel=2)
        values = {**values, "attachment": {**attachment, "sphere_budget": 24}}
    return PlanningSettings.model_validate(values)


def save_settings(path: Path, settings: PlanningSettings):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(settings.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
