from pathlib import Path


def test_operations_have_no_fixed_unpack_workflow_dependencies():
    source = (Path(__file__).parents[1] / 'src/luna_mujoco/robot_skills/operations.py').read_text(encoding='utf-8')
    for old_name in ('UnpackWorkflow', 'current_lifecycle', 'unpack_task', 'demo_task', 'set_observation_stage'):
        assert old_name not in source


def test_independent_skill_modules_exist():
    from luna_mujoco.robot_skills.container_geometry import container_occupied_bounds
    from luna_mujoco.robot_skills.processing import ProcessingSkills
    assert callable(container_occupied_bounds)
    assert hasattr(ProcessingSkills, '_verify_unpack')
