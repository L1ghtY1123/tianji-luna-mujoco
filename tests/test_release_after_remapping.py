"""A new measured map does not erase a completed release's execution evidence."""
from copy import deepcopy

import pytest
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision, TestPolicyController


@pytest.mark.parametrize('recheck', [False, True])
async def test_release_can_be_verified_after_explicit_observe_and_rebuild(tmp_path, recheck):
    class FreshMaps(DemoVision):
        def build_scene(self, observation, target_id, **kwargs):
            scene = super().build_scene(observation, target_id, **kwargs)
            scene.update(scene_token='map-' + observation['observation_id'],
                coverage={'inventory_complete': True}, inventory_map={'entities': {
                    'pink-1': {'status': 'observed', 'last_observation_id': observation['observation_id'],
                               'last_generation': observation['generation']}}})
            return scene

    class RebuildBeforeVerification(TestPolicyController):
        checked = False

        async def execute(self, skill, context, arguments=None):
            if skill != 'verify_place':
                return await super().execute(skill, context, arguments)
            original_feedback = None
            if recheck:
                self.algorithms.progress_status = 'unknown'
                first = await super().execute(skill, context, arguments)
                assert first['status'] == 'unknown'
                original_feedback = first['feedback_id']
            await super().execute('observe_scene', context)
            context['mapping_lifecycle'].update(requested_scope='global', strategy_source='responses',
                scope_reason='test graph explicitly refreshes released object')
            await self.build_scene(context, 'selected')
            assert 'place_plan' not in context and 'pending_transfer' not in context
            scene = deepcopy(context['scene'])
            self.algorithms.progress_status = 'at_destination'
            result = await super().execute(skill, context, arguments)
            assert result['success'] is True
            if original_feedback:
                assert result['feedback_id'] == original_feedback
            assert context['scene'] == scene
            assert not context.get('scene_refresh_required')
            assert self.state['facts']['scene_available'] is True
            assert not any(row[0] == 'relocate' for row in self.algorithms.map_order)
            assert not context.get('holding') and self.holding_state is None
            self.checked = True
            return result

    runner = RebuildBeforeVerification(DemoRobot(), FreshMaps(1), None, tmp_path,
                                      planning_settings=TEST_SETTINGS)
    await runner.start('move all pink objects')
    await runner.worker
    assert runner.checked, runner.state.get('error')
    assert runner.task_progress.completed_ids == ['pink-1']


@pytest.mark.parametrize('mutation', ['owner', 'target', 'generation', 'held', 'attachment', 'geometry', 'support', 'feedback', 'ledger'])
async def test_frozen_release_rejects_changed_evidence_or_ledger_before_visual(tmp_path, mutation):
    class ChangedEvidence(TestPolicyController):
        async def execute(self, skill, context, arguments=None):
            if skill == 'verify_place':
                proof = context['release_execution_evidence']
                assert 'phases' not in proof['place'] and 'command' not in proof['place']
                if mutation == 'owner':
                    self.run_id = 'different-owner'
                elif mutation == 'target':
                    proof['target_id'] = 'different-target'
                elif mutation == 'generation':
                    proof['generation'] += 1
                elif mutation == 'held':
                    proof['held']['verification']['status'] = 'unknown'
                elif mutation == 'attachment':
                    proof['held']['attachment_evidence']['attached'] = False
                elif mutation == 'geometry':
                    proof['held']['points_tool_m'][0][0] += .1
                elif mutation == 'support':
                    proof['held']['grasp_support_reference']['T_world_tcp_at_grasp'][0][3] += .1
                elif mutation == 'feedback':
                    proof['completed_feedback']['time'] += 1
                else:
                    self.task_progress.records[self.state['target_id']]['status'] = 'completed'
            return await super().execute(skill, context, arguments)

    algorithms = DemoVision(1)
    runner = ChangedEvidence(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start('move all pink objects')
    await runner.worker
    assert runner.state['status'] == 'needs_attention'
    assert algorithms.release_calls == 0


@pytest.mark.parametrize('damage', ['old_capture', 'missing_capture', 'stale', 'incomplete', 'different_generation'])
async def test_remapped_release_without_fresh_target_proof_only_invalidates(tmp_path, damage):
    class InvalidateOnly(DemoVision):
        def invalidate_scene_entity(self, scene, target, action, *, source):
            assert source == 'verified_release_action'
            self.map_order.append(('invalidate', target, action))
            return scene | {'scene_token': scene['scene_token'] + ':invalidated'}

    class IncompleteRemap(TestPolicyController):
        checked = False

        async def execute(self, skill, context, arguments=None):
            if skill != 'verify_place':
                return await super().execute(skill, context, arguments)
            await super().execute('observe_scene', context)
            context['mapping_lifecycle'].update(requested_scope='global', strategy_source='responses',
                scope_reason='explicit graph update')
            await self.build_scene(context, 'selected')
            current = context['scene_observation']
            record = {'status': 'observed', 'last_generation': current['generation'],
                      'last_observation_id': current['observation_id']}
            context['scene'].update(scene_token='new-map', coverage={'inventory_complete': True},
                                    inventory_map={'entities': {'pink-1': record}})
            if damage == 'old_capture':
                record['last_observation_id'] = context['release_execution_evidence']['scene']['observation_id']
            elif damage == 'missing_capture':
                record['last_observation_id'] = 'no-capture-with-this-id'
            elif damage == 'stale':
                record['status'] = 'stale'
            elif damage == 'incomplete':
                context['scene']['coverage']['inventory_complete'] = False
            else:
                record['last_generation'] += 1
            result = await super().execute(skill, context, arguments)
            assert result['success'] is True
            assert context['scene_refresh_required'] is True
            assert self.state['facts']['scene_available'] is False
            assert context['mapping_lifecycle']['invalidated_entity_ids'] == ['pink-1']
            assert [row[0] for row in self.algorithms.map_order].count('invalidate') == 1
            assert not any(row[0] == 'relocate' for row in self.algorithms.map_order)
            self.checked = True
            return result

    runner = IncompleteRemap(DemoRobot(), InvalidateOnly(1), None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start('move all pink objects')
    await runner.worker
    assert runner.checked, runner.state.get('error')


async def test_release_proof_keeps_execution_fields_without_duplicate_sensor_snapshots(tmp_path):
    class InspectProof(TestPolicyController):
        checked = False

        async def execute(self, skill, context, arguments=None):
            if skill != 'execute_place':
                return await super().execute(skill, context, arguments)
            source = context['holding']
            snapshots = {key: source[key] for key in ('observation', 'scene', 'scene_observation')}
            expected_keys = set(source) - set(snapshots)
            result = await super().execute(skill, context, arguments)
            proof = context['release_execution_evidence']
            assert not set(snapshots).intersection(proof['held'])
            assert expected_keys <= set(proof['held'])
            assert all(source[key] is value for key, value in snapshots.items())
            assert proof['held']['verification']['status'] == 'held'
            assert proof['held']['attachment_evidence']['attached'] is True
            assert proof['held']['grasp_support_reference'] == source['grasp_support_reference']
            self.checked = True
            return result

    runner = InspectProof(DemoRobot(), DemoVision(1), None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start('move all pink objects')
    await runner.worker
    assert runner.checked, runner.state.get('error')
    assert runner.task_progress.completed_ids == ['pink-1']
