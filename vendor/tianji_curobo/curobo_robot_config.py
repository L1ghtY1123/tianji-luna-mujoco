"""从一份 Marvin 双臂 cuRobo 配置派生独立的 A/B 单臂配置。"""

from copy import deepcopy
from itertools import combinations
from pathlib import Path
# ========================================
# MODIFICATION START: type collision-link exclusion inputs
# ========================================
from typing import Any, Dict, Sequence
# ========================================
# MODIFICATION END: type collision-link exclusion inputs
# ========================================


SHARED_COLLISION_LINKS = {"Link_Base", "Link_Stand"}

# A/B 与 GUI 的定义保持一致：A 是左臂 _L，B 是右臂 _R。
ARM_CONFIG_SPECS = {
    "A": {
        "joint_names": [f"Joint{index}_L" for index in range(1, 8)],
        "tool_frame": "TCP_Link_L",
        "link_suffix": "_L",
        "gripper_prefix": "gripL_",
    },
    "B": {
        "joint_names": [f"Joint{index}_R" for index in range(1, 8)],
        "tool_frame": "TCP_Link_R",
        "link_suffix": "_R",
        "gripper_prefix": "gripR_",
    },
}


def collision_links_for_arm(
    combined_robot_config: Dict[str, Any],
    arm: str,
    *,
    include_shared: bool = False,
) -> list[str]:
    """Return the collision links that belong to one Marvin arm."""

    if arm not in ARM_CONFIG_SPECS:
        raise ValueError("arm must be 'A' or 'B'")
    kinematics = combined_robot_config.get(
        "robot_cfg", combined_robot_config
    )["kinematics"]
    spec = ARM_CONFIG_SPECS[arm]
    chain_links = {
        f"Base{spec['link_suffix']}",
        *(f"Link{index}{spec['link_suffix']}" for index in range(1, 8)),
    }
    return [
        link_name
        for link_name in kinematics["collision_link_names"]
        if (include_shared and link_name in SHARED_COLLISION_LINKS)
        or link_name in chain_links
        or link_name == spec["tool_frame"]
        or link_name.startswith(spec["gripper_prefix"])
    ]


# ==================== SDK_CUROBO 修改开始：加载时把序列化绝对路径重定位到当前仓库 ====================
def rebase_robot_asset_paths(
    combined_robot_config: Dict[str, Any],
    *,
    urdf_path: Path,
    asset_root_path: Path | None = None,
) -> Dict[str, Any]:
    """Return a copy whose resources come from the model capability.

    The cuRobo Extension never guesses a repository root.  Its owner resolves
    the URDF and mesh resources through ``jodell.robot.model.resources/v1``
    and passes those explicit paths into the isolated planner process.
    """

    resolved_urdf = Path(urdf_path).expanduser().resolve()
    if not resolved_urdf.is_file():
        raise FileNotFoundError(f"Marvin URDF not found: {resolved_urdf}")
    resolved_root = Path(
        asset_root_path if asset_root_path is not None else resolved_urdf.parent
    ).expanduser().resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"Marvin asset root not found: {resolved_root}")

    robot_config = deepcopy(combined_robot_config)
    kinematics = robot_config.get("robot_cfg", robot_config)["kinematics"]
    kinematics["asset_root_path"] = str(resolved_root)
    from tianji_curobo.urdf_axis_adapter import regularize_joint_axes

    # The authoritative model stays read-only; only bounded axis roundoff is
    # corrected in a separately audited writable-cache copy for this SDK.
    kinematics["urdf_path"] = str(regularize_joint_axes(resolved_urdf))
    return robot_config
# ==================== SDK_CUROBO 修改结束 ====================


def create_robot_segmentation_config(
    combined_robot_config: Dict[str, Any],
    *,
    urdf_path: Path,
    asset_root_path: Path | None = None,
) -> Dict[str, Any]:
    """Keep the complete robot collision model for depth segmentation."""

    robot_config = rebase_robot_asset_paths(
        combined_robot_config,
        urdf_path=urdf_path,
        asset_root_path=asset_root_path,
    )
    kinematics = robot_config.get("robot_cfg", robot_config)["kinematics"]
    cspace = kinematics.get("cspace")
    if cspace is None:
        return robot_config
    segmenter_joint_names = (
        ARM_CONFIG_SPECS["A"]["joint_names"]
        + ARM_CONFIG_SPECS["B"]["joint_names"]
    )
    configured_segmenter_joints = [
        joint_name
        for joint_name in cspace["joint_names"]
        if joint_name in segmenter_joint_names
    ]
    if configured_segmenter_joints != segmenter_joint_names:
        raise ValueError("robot segmentation cspace must contain both arm joint chains")
    position_limit_clip = cspace.get("position_limit_clip")
    if not isinstance(position_limit_clip, list):
        raise ValueError("robot segmentation position_limit_clip must be a list")
    single_arm_dof = len(ARM_CONFIG_SPECS["A"]["joint_names"])
    if len(position_limit_clip) == single_arm_dof:
        cspace["position_limit_clip"] = position_limit_clip * 2
    elif len(position_limit_clip) != len(segmenter_joint_names):
        raise ValueError(
            "robot segmentation position_limit_clip must match both arm joint chains"
        )
    return robot_config


def create_dual_arm_robot_config(combined_robot_config, *, urdf_path, asset_root_path=None):
    """Keep both moving chains and the original complete collision model."""
    result = create_robot_segmentation_config(
        combined_robot_config, urdf_path=urdf_path, asset_root_path=asset_root_path,
    )
    kinematics = result.get("robot_cfg", result)["kinematics"]
    names = ARM_CONFIG_SPECS["A"]["joint_names"] + ARM_CONFIG_SPECS["B"]["joint_names"]
    cspace = kinematics["cspace"]
    original_names = cspace["joint_names"]
    indices = [original_names.index(name) for name in names]
    for key, value in list(cspace.items()):
        if isinstance(value, list) and len(value) == len(original_names):
            cspace[key] = [value[index] for index in indices]
    cspace["joint_names"] = names
    kinematics["lock_joints"] = {
        name: value for name, value in kinematics.get("lock_joints", {}).items()
        if name not in names
    }
    kinematics["tool_frames"] = ["TCP_Link_L", "TCP_Link_R"]
    return result


def create_frozen_peer_arm_robot_config(
    combined_robot_config: Dict[str, Any],
    arm: str,
    *,
    urdf_path: Path,
    asset_root_path: Path | None = None,
) -> Dict[str, Any]:
    """Keep both arms' collision spheres while exposing one active arm."""

    if arm not in ARM_CONFIG_SPECS:
        raise ValueError("arm must be 'A' or 'B'")
    robot_config = rebase_robot_asset_paths(
        combined_robot_config,
        urdf_path=urdf_path,
        asset_root_path=asset_root_path,
    )
    kinematics = robot_config.get("robot_cfg", robot_config)["kinematics"]
    active_spec = ARM_CONFIG_SPECS[arm]
    peer_spec = ARM_CONFIG_SPECS["B" if arm == "A" else "A"]
    kinematics["tool_frames"] = [active_spec["tool_frame"]]
    locked_joints = dict(kinematics.get("lock_joints", {}))
    locked_joints.update(
        {joint_name: 0.0 for joint_name in peer_spec["joint_names"]}
    )
    kinematics["lock_joints"] = locked_joints
    peer_arm = "B" if arm == "A" else "A"
    peer_collision_links = collision_links_for_arm(robot_config, peer_arm)
    _ignore_internal_self_collision_pairs(
        kinematics,
        peer_collision_links,
    )
    return robot_config


def _ignore_internal_self_collision_pairs(
    kinematics: Dict[str, Any],
    link_names: Sequence[str],
) -> None:
    """Ignore only pairs whose two links belong to the frozen peer arm."""

    collision_links = set(kinematics.get("collision_link_names", []))
    peer_links = [
        link_name
        for link_name in link_names
        if link_name in collision_links
    ]
    ignored_pairs = kinematics.setdefault("self_collision_ignore", {})
    for link_name in peer_links:
        ignored_pairs.setdefault(link_name, [])
    for first_link, second_link in combinations(peer_links, 2):
        if second_link not in ignored_pairs[first_link]:
            ignored_pairs[first_link].append(second_link)
        if first_link not in ignored_pairs[second_link]:
            ignored_pairs[second_link].append(first_link)


def create_single_arm_robot_config(
    combined_robot_config: Dict[str, Any],
    arm: str,
    *,
    urdf_path: Path,
    asset_root_path: Path | None = None,
) -> Dict[str, Any]:
    """从完整双臂配置的深拷贝中筛选一条 7 自由度机械臂。

    原始 ``combined_robot_config`` 不会被修改，因此同一份 YAML 可以先后派生
    A、B 两个 planner。筛选范围包括碰撞 link、碰撞球、自碰撞矩阵、工具坐标、
    夹爪锁定关节以及所有与 cspace 关节数量对应的参数数组。
    """

    if arm not in ARM_CONFIG_SPECS:
        raise ValueError("arm must be 'A' or 'B'")

    # ==================== SDK_CUROBO 修改开始：所有派生入口统一重定位 URDF/mesh 根目录 ====================
    robot_config = rebase_robot_asset_paths(
        combined_robot_config,
        urdf_path=urdf_path,
        asset_root_path=asset_root_path,
    )
    # ==================== SDK_CUROBO 修改结束 ====================
    kinematics = robot_config.get("robot_cfg", robot_config)["kinematics"]
    spec = ARM_CONFIG_SPECS[arm]

    collision_links = collision_links_for_arm(
        robot_config,
        arm,
        include_shared=True,
    )
    collision_link_set = set(collision_links)
    kinematics["collision_link_names"] = collision_links
    kinematics["mesh_link_names"] = collision_links.copy()

    # 下面三项必须使用同一 link 集合，否则球体索引和自碰撞矩阵会不一致。
    kinematics["collision_spheres"] = {
        link_name: spheres
        for link_name, spheres in kinematics["collision_spheres"].items()
        if link_name in collision_link_set
    }
    collision_sphere_buffer = kinematics.get("collision_sphere_buffer")
    if isinstance(collision_sphere_buffer, dict):
        kinematics["collision_sphere_buffer"] = {
            link_name: distance
            for link_name, distance in collision_sphere_buffer.items()
            if link_name in collision_link_set
        }
    kinematics["self_collision_buffer"] = {
        link_name: distance
        for link_name, distance in kinematics["self_collision_buffer"].items()
        if link_name in collision_link_set
    }
    kinematics["self_collision_ignore"] = {
        link_name: [
            ignored_link
            for ignored_link in ignored_links
            if ignored_link in collision_link_set
        ]
        for link_name, ignored_links in kinematics[
            "self_collision_ignore"
        ].items()
        if link_name in collision_link_set
    }
    kinematics["tool_frames"] = [spec["tool_frame"]]

    # 夹爪参与碰撞，但不进入 N×7 轨迹，因此保留本臂夹爪的固定开合角。
    cspace = kinematics["cspace"]
    combined_joint_names = list(cspace["joint_names"])
    selected_joint_names = list(spec["joint_names"])
    selected_indices = [
        combined_joint_names.index(joint_name)
        for joint_name in selected_joint_names
    ]

    # ==================== SDK_CUROBO 修改开始：精确筛选后只锁定当前手臂的夹爪驱动关节 ====================
    # 另一条手臂已不在当前 tool/collision 形成的 kinematic tree 中，不能再写入
    # lock_joints；当前夹爪参与碰撞但不进入 N×7 轨迹，因此固定在配置角度。
    lock_joints = kinematics.get("lock_joints", {})
    kinematics["lock_joints"] = {
        joint_name: value
        for joint_name, value in lock_joints.items()
        if joint_name.startswith(spec["gripper_prefix"])
    }
    # ==================== SDK_CUROBO 修改结束 ====================

    # default position、速度/加速度缩放等数组都与 joint_names 等长，需同步裁剪。
    for setting_name, setting_value in list(cspace.items()):
        if isinstance(setting_value, list) and len(setting_value) == len(
            combined_joint_names
        ):
            cspace[setting_name] = [
                setting_value[index] for index in selected_indices
            ]
    cspace["joint_names"] = selected_joint_names
    return robot_config
# ========================================
# MODIFICATION START: centralize collision-link exclusion
# ========================================
def _exclude_collision_links(
    robot_config: Any,
    excluded_links: Sequence[str],
) -> Any:
    """Return a copy with selected links removed from collision fields."""

    excluded = {str(link_name) for link_name in excluded_links}
    if not excluded:
        return robot_config

    copied = deepcopy(robot_config)
    kinematics = copied.get("robot_cfg", copied)["kinematics"]
    for list_name in ("collision_link_names", "mesh_link_names"):
        kinematics[list_name] = [
            link_name
            for link_name in kinematics.get(list_name, [])
            if link_name not in excluded
        ]
    for mapping_name in (
        "collision_spheres",
        "self_collision_buffer",
        "self_collision_ignore",
    ):
        kinematics[mapping_name] = {
            link_name: value
            for link_name, value in kinematics.get(mapping_name, {}).items()
            if link_name not in excluded
        }
    collision_sphere_buffer = kinematics.get("collision_sphere_buffer")
    if isinstance(collision_sphere_buffer, dict):
        kinematics["collision_sphere_buffer"] = {
            link_name: distance
            for link_name, distance in collision_sphere_buffer.items()
            if link_name not in excluded
        }
    kinematics["self_collision_ignore"] = {
        link_name: [
            ignored_link
            for ignored_link in ignored_links
            if ignored_link not in excluded
        ]
        for link_name, ignored_links in kinematics[
            "self_collision_ignore"
        ].items()
    }
    return copied
# ========================================
# MODIFICATION END: centralize collision-link exclusion
# ========================================
