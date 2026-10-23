"""Check adapter keyword names against the installed, pinned public cuRobo API."""
import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from curobo.batch_motion_planner import BatchMotionPlanner
from curobo.inverse_kinematics import InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState, ToolPoseCriteria

from luna_mujoco.backend_builder import create_solver_configs
from luna_mujoco.planning_settings import PlanningSettings

ROOT = Path(__file__).resolve().parents[1]


def audit():
    rows = []

    def check(label, function, keywords):
        signature = inspect.signature(function)
        assert set(keywords) <= set(signature.parameters), (label, set(keywords) - set(signature.parameters))
        rows.append({'api': label, 'keywords': sorted(keywords)})

    def ik_config(**kwargs):
        check('InverseKinematicsCfg.create', InverseKinematicsCfg.create, kwargs)
        return SimpleNamespace(robot_config=kwargs['robot'])

    def motion_config(**kwargs):
        check('MotionPlannerCfg.create', MotionPlannerCfg.create, kwargs)
        return SimpleNamespace()

    create_solver_configs(PlanningSettings(), robot={}, scene_model={}, collision_cache={}, batch_size=4,
        ik_config_type=SimpleNamespace(create=ik_config), motion_config_type=SimpleNamespace(create=motion_config))
    # Explicit grasp keywords in project and the frozen transport adapter.
    for file in [ROOT / 'src/luna_mujoco/backend_builder.py', *sorted((ROOT / 'vendor/tianji_curobo').glob('*.py'))]:
        tree = ast.parse(file.read_text(encoding='utf8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != 'plan_grasp':
                continue
            keywords = [kw.arg for kw in node.keywords if kw.arg is not None]
            for cls in (MotionPlanner, BatchMotionPlanner):
                check(f'{cls.__name__}.plan_grasp ({file.name}:{node.lineno})', cls.plan_grasp, keywords)
    check('MotionPlanner.plan_pose', MotionPlanner.plan_pose, ['max_attempts'])
    check('BatchMotionPlanner.plan_pose', BatchMotionPlanner.plan_pose,
          ['max_attempts', 'finetune_attempts', 'finetune_dt_scale'])
    check('ToolPoseCriteria.linear_motion', ToolPoseCriteria.linear_motion,
          ['axis', 'non_terminal_scale', 'project_distance_to_goal'])
    assert 'dt' in JointState.__dataclass_fields__
    for cls in (MotionPlanner, BatchMotionPlanner):
        assert 'max_attempts' not in inspect.signature(cls.plan_grasp).parameters
        assert 'place_timeout_s' not in inspect.signature(cls.plan_grasp).parameters
    return {'checks': rows, 'JointState_dt': str(JointState.__dataclass_fields__['dt'].type),
            'retired_settings_absent': all(name not in PlanningSettings.model_fields
                for name in ('plan_timeout_s', 'place_timeout_s')),
            'scope': 'config constructor and explicit grasp/pose/linear-motion keywords; no robot motion'}


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
