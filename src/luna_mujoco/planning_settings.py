"""One validated settings snapshot for the UI, both IK solvers, planning and execution."""
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator

from .observation_motion import observation_joint_limits, observation_poses

RELEASE_ENDPOINT_POSITION_TOLERANCE_M = .002


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


class PlanningSettings(BaseModel):
    experimental_semantic_only_mapping: bool = setting(True, "实验场景仅保留已识别物体",
        "仅在双臂观测位姿分析完成后，保留SAM3物体/容器及桌面；删除未分类背景。初始运动地图不使用此规则。",
        "filter")
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    holding_position_tolerance_fraction: float = setting(.02, "夹持位置容差", "相对空夹闭合位置的容差，占有效行程的比例。", "verification", ge=.001, le=.1, step=.005)
    holding_stable_window_s: float = setting(.2, "夹持稳定窗口", "只使用窗口内的夹爪位置判断稳定，不使用力或电流。", "verification", unit="s", ge=.05, le=1., step=.05)

    ik_num_seeds: int = setting(16, "IK 优化 seeds", "每目标 IK 优化起点数；16 为锁定 cuRobo batch planner 默认，两处 IK 显式对齐。",
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
    trajectory_collision_clearance_m: float = setting(.01, "碰撞间距（hard clearance）",
        "独立 IK、内部 IK 和三段轨迹复核共用。目标物体仅从 cuRobo world 排除。",
        "shared", unit="mm", scale=1000, ge=.01, le=.5)

    optimizer_collision_activation_distance_m: float = setting(.03, "碰撞代价激活距离（soft activation）",
        "TrajOpt 在该距离内产生避障代价，不等于 hard clearance；必须不小于 hard clearance。",
        "planner", unit="mm", scale=1000, ge=.03, le=.5)
    trajopt_num_seeds: int = setting(2, "TrajOpt seeds", "每目标轨迹优化起点数；2 为官方 batch 默认。项目要求 ≤ IK seeds，避免 SDK 自动上调。",
                                    "planner", ge=1, le=8, strict=True)
    plan_max_attempts: int = setting(2, "最大规划 attempts", "普通plan_pose/plan_cspace的公开尝试预算。plan_grasp内部阶段采用官方默认，公共接口未暴露此参数。",
                                     "planner", ge=1, le=10, strict=True)
    finetune_attempts: int = setting(1, "finetune attempts", "仅批量普通plan_pose公开支持；单臂plan_pose及plan_grasp内部阶段不覆盖。",
                                    "planner", ge=1, le=10, strict=True)
    finetune_dt_scale: float = setting(.55, "finetune dt scale", "仅批量普通plan_pose公开支持；plan_grasp内部用官方默认。独立于执行time_scale。",
                                       "planner", step=.05, gt=0, le=1)
    freq_hz: int = setting(50, "SDK 轨迹采样频率", "实际映射 cuRobo interpolation_dt=1/frequency；不是 MuJoCo 物理步频。",
                           "planner", unit="Hz", ge=20, le=100, strict=True)

    approach_axis: Literal["x", "y", "z"] = setting("z", "Approach 轴", "生成接近位姿的坐标轴。", "grasp")
    approach_offset_m: float = setting(-.1, "Approach 偏移", "沿指定坐标轴相对抓取位姿偏移；不能为零。",
                                       "grasp", unit="mm", scale=1000, ge=-.5, le=.5)
    approach_in_tool_frame: bool = setting(True, "Approach 使用 tool frame", "开启为 tool frame；关闭为 world frame。", "grasp")
    lift_axis: Literal["x", "y", "z"] = setting("z", "Lift 轴", "生成提升位姿的坐标轴。", "grasp")
    lift_offset_m: float = setting(.1, "Lift 偏移", "默认 world +Z 100 mm；demo按实测源盒/邻件占据边界与完整持物球包络推导离箱高度，至少保留此值并归档有效值。不能为零。",
                                   "grasp", unit="mm", scale=1000, ge=-.5, le=.5)
    lift_in_tool_frame: bool = setting(False, "Lift 使用 tool frame", "开启为 tool frame；关闭为 world frame。", "grasp")

    place_approach_axis: Literal["x", "y", "z"] = setting("z", "Approach 轴", "官方grasp_approach_axis；放置接近段使用。", "place")
    place_approach_offset_m: float = setting(.1, "Approach 偏移", "官方grasp_approach_offset；默认在放置位姿上方 world +Z 100 mm，不能为零。",
                                             "place", unit="mm", scale=1000, ge=-.5, le=.5)
    place_approach_in_tool_frame: bool = setting(False, "Approach 使用 tool frame", "开启为 tool frame；关闭为 world frame。", "place")
    place_retreat_axis: Literal["x", "y", "z"] = setting("z", "Lift 轴", "官方grasp_lift_axis语义；开爪detach后的提升段使用。", "place")
    place_retreat_offset_m: float = setting(.1, "Lift 偏移", "官方grasp_lift_offset语义；默认沿 world +Z 提升100 mm，不能为零。",
                                            "place", unit="mm", scale=1000, ge=-.5, le=.5)
    place_retreat_in_tool_frame: bool = setting(False, "Lift 使用 tool frame", "开启为 tool frame；关闭为 world frame。", "place")
    place_release_height_m: float = setting(.01, "持物球底最小间隙", "只读，等于hard clearance，默认10mm。固定抓取旋转，考虑attachment与夹爪最低点及实际碰撞场计算尽量低的安全Z；实际物体包络在目标区域内即可，不设置2mm位置误差上限。",
                                       "place", unit="mm", scale=1000, step=1, ge=0, le=.502)

    time_scale: float = setting(2., "时间伸长／补点倍率（三段共用）", "cuRobo 完成后用 q/v/a 做五次 Hermite 补点；默认 2 倍，执行点间隔保持 SDK dt。",
                                "execution", unit="倍", step=.1, ge=1, le=10)
    dwell_approach_s: float = setting(1., "Approach 后停留", "仿真保持关节目标；停留不乘时间伸长倍率。",
                                     "execution", unit="s", step=.1, ge=0, le=30)
    dwell_grasp_s: float = setting(1., "夹爪闭合后停留", "停留后以实测可见点云和 TCP 反馈记录抓取参照，再执行原有 Lift。",
                                  "execution", unit="s", step=.1, ge=0, le=30)
    dwell_lift_s: float = setting(1., "Lift 后停留", "停留后只采集 post_grasp RGBD；verify_grasp 计算持物结果并建立 attachment。",
                                 "execution", unit="s", step=.1, ge=0, le=30)
    dwell_place_approach_s: float = setting(1., "放置 approach 后停留", "保持关节目标后进入放置阶段。",
                                           "execution", unit="s", step=.1, ge=0, le=30)
    dwell_place_s: float = setting(1., "放置后开爪前停留", "到达放置位姿后保持关节目标，再检查开爪。",
                                   "execution", unit="s", step=.1, ge=0, le=30)
    dwell_release_s: float = setting(1., "释放后稳定等待", "开爪并 detach 后等待，再执行预先规划的撤离；不在此观察。",
                                     "execution", unit="s", step=.1, ge=0, le=30)
    dwell_retreat_s: float = setting(1., "撤离后停留", "撤离后等待，再观察一次以确认可见放置结果。",
                                     "execution", unit="s", step=.1, ge=0, le=30)

    pointcloud_filter: PointcloudFilterSettings = Field(default_factory=PointcloudFilterSettings)
    observation: ObservationSettings = Field(default_factory=ObservationSettings)
    attachment: AttachmentSettings = Field(default_factory=AttachmentSettings)

    @model_validator(mode="before")
    @classmethod
    def migrate_saved_settings(cls, values):
        if not isinstance(values, dict):
            return values
        migrated = dict(values)
        # The fixed-open pregrasp place preflight was removed. Retire its
        # saved option instead of exposing an advanced setting with no effect.
        migrated.pop("place_max_candidates", None)
        for legacy_name in ("release_margin_m", "place_release_margin_m", "plan_timeout_s", "place_timeout_s"):
            migrated.pop(legacy_name, None)
        migrated['place_release_height_m'] = migrated.get('trajectory_collision_clearance_m', .01)
        return migrated

    @model_validator(mode="after")
    def validate_relationships(self):
        if self.trajopt_num_seeds > self.ik_num_seeds:
            raise ValueError("TrajOpt seeds 不能大于 IK 优化 seeds，否则 SDK 会增加内部 IK seeds")
        if self.ik_num_seeds > self.ik_seed_solver_num_seeds:
            raise ValueError("IK 优化 seeds 不能大于 LM seeds，否则 SDK 会增加实际 LM seeds")
        if self.optimizer_collision_activation_distance_m < self.trajectory_collision_clearance_m:
            raise ValueError("soft activation 必须不小于 hard clearance")
        if self.approach_offset_m == 0 or self.lift_offset_m == 0:
            raise ValueError("Approach / Lift 偏移不能为零")
        if self.place_approach_offset_m == 0 or self.place_retreat_offset_m == 0:
            raise ValueError("Place approach / retreat 偏移不能为零")
        return self

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
    clearance = values.get("trajectory_collision_clearance_m") if isinstance(values, dict) else None
    if type(clearance) in (int, float) and .005 <= clearance < .01:
        # Only loading an older saved policy may upgrade its clearance. Live
        # requests below 10 mm remain invalid; existing plans/bindings are not rewritten.
        warnings.warn(f"saved collision clearance {clearance} migrated to 0.01 m under the 10 mm policy",
                      UserWarning, stacklevel=2)
        values = {**values, "trajectory_collision_clearance_m": .01}
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
