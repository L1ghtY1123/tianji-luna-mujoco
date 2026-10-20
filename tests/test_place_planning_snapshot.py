"""One stationary planning operation must not invalidate its freshly prepared model."""
import copy


async def test_attachment_and_place_planning_share_one_measured_snapshot(tmp_path):
    from test_current_grasp_attachment import fixture

    runner, context, _, _, _, _ = fixture(tmp_path)
    await runner.verify_grasp(context)
    runner.task_progress = None
    runner.state.update(destination_id='destination', current_step={})
    original_snapshot = runner.simulation.robot_snapshot
    snapshots = []
    inputs = []

    def snapshot():
        result = copy.deepcopy(original_snapshot())
        result['arms']['A']['gripper_position_rad'] += len(snapshots) * .000001
        snapshots.append(result)
        return result

    def prepare(observation, *args, **kwargs):
        inputs.append(observation)
        return {'attached': True}

    def plan(observation, *args, **kwargs):
        inputs.append(observation)
        return {'ok': True}

    async def verify_scene(context):
        pass

    async def call(function, *args, **kwargs):
        kwargs.pop('report_progress', None)
        kwargs.pop('timeout_s', None)
        return function(*args, **kwargs)

    async def stage(context, name, function, *args, **kwargs):
        return await function(*args, **kwargs)

    runner.simulation.robot_snapshot = snapshot
    runner.algorithms.prepare_carrying = prepare
    runner.algorithms.plan_place = plan
    runner.verify_scene = verify_scene
    runner.call_algorithm = call
    runner.execution_stage = stage
    result = await runner.execute('plan_place', context)
    assert result['ok']
    assert len(inputs) == 2
    assert inputs[0] is inputs[1]
    assert inputs[0]['arms']['A']['gripper_position_rad'] == snapshots[0]['arms']['A']['gripper_position_rad']
