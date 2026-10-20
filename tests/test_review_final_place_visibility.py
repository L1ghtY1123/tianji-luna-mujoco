"""A physical release check preserves the separate Decisions observation."""

from test_rgbd_workflow import SensorRobot, VisionAlgorithms

from jev_mujoco.runner import Runner


async def test_explicit_postplace_visual_result_survives_release_feedback_check(tmp_path):
    robot = SensorRobot()
    robot.stage = 5
    algorithms = VisionAlgorithms()
    capture = robot.capture_rgbd()
    runner = Runner(robot, algorithms, None, tmp_path)
    runner.state.update(generation=1, task='grasp_place', target_id='source',
                        destination_id='destination', arm='A')
    runner.state['facts'].update(released=True, place_execution_completed=True)
    context = {'visual_plan': {'post_place': 'initialize_inventory'}}
    observation = await runner.observe(context, checkpoint='post_place', capture=capture)
    result = await runner.verify_place(context)
    assert result['status'] == 'confirmed'
    assert result['verification_scope'] == 'release_only'
    assert context['observation']['visual_inventory'] == observation['visual_inventory']
    assert algorithms.checkpoints == ['post_place']
