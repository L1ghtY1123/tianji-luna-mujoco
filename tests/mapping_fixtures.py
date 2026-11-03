"""Unhandled evidence returns to the bounded model revision boundary, never a fixed route."""
from copy import deepcopy

from test_agent_model_layer import Client, task_data
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

from luna_mujoco.agent.decisions import DecisionsJudgments
from luna_mujoco.agent.responses import ResponsesCompiler
from luna_mujoco.agent.schema import TaskSpec
from luna_mujoco.runtime.controller import PolicyController, TargetArguments
from luna_mujoco.runtime.inventory import PolicyTaskProgress
from luna_mujoco.skills import Contract


def setup(tmp_path, results):
    task = TaskSpec.model_validate(task_data() | {'scope': 'ids', 'target_refs': ['done', 'pending']})
    http = Client([])
    compiler = ResponsesCompiler(registry={}, api_key='test', client=http)
    decision_http = Client([])
    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path,
        planning_settings=TEST_SETTINGS, compiler=compiler,
        judgments=DecisionsJudgments(api_key='test', client=decision_http))
    runner.revision_decision_http = decision_http
    runner.state.update(generation=1, target_id='pending', status='running')
    runner.state['facts'].update(grasp_held=False, recovery_needed=False, inventory_finalized=True, task_assessed=True)
    runner.task_spec = task
    runner.task_inventory = {'selected_ids': ['done', 'pending'], 'source_object_ids': ['done', 'pending'],
                             'source_id': 'table', 'destination_id': 'tray'}
    runner.task_progress = PolicyTaskProgress(task.instruction)
    runner.task_progress.records = {
        'done': {'id': 'done', 'status': 'completed', 'current_region': 'tray',
                 'destination_id': 'tray', 'confirmed_observation_id': 'done-observation'},
        'pending': {'id': 'pending', 'status': 'observed', 'current_region': 'table', 'destination_id': 'tray'}}
    runner.task_progress.membership = {'done': 'required', 'pending': 'required'}
    runner.task_progress.configure_task(task, ['done', 'pending'])
    runner.task_progress.inventory_finalized = True
    runner.task_progress.completed_ids = ['done']
    calls = []

    async def probe(context, arguments):
        calls.append(arguments.target_id)
        result = deepcopy(results.pop(0))
        if result.get('success'):
            runner.task_progress.complete(arguments.target_id, observation_id='verified-new')
        return result

    runner.registry.contracts['probe'] = Contract(id='probe', description='Read current evidence',
        requires={}, provides=[], invalidates_observation=False)
    runner.registry.register('probe', TargetArguments, probe)
    async def verified(context, arguments):
        return {'success': True}
    runner.registry.handlers['verify_task_completion'] = verified
    compiler.registry = runner.registry
    compiler.original_task = task
    compiler.budget.reserve()
    capture = runner.simulation.capture_rgbd()
    context = {'latest_capture': capture, 'observation': capture | {'source': 'rgbd', 'objects': []},
               'planning_settings': TEST_SETTINGS.model_dump(),
               'capture_epochs': {capture['observation_id']: runner.observation_epoch},
               'visual_inventory_reference': {}}
    return runner, context, http, calls
