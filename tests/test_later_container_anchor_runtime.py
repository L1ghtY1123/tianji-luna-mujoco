"""A confirmed later capture can establish container anchors in the actual runner."""
from copy import deepcopy

import pytest
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision, TestPolicyController


@pytest.mark.parametrize('damage', [None, 'reference', 'anchor', 'generation', 'owner'])
async def test_runner_adopts_first_confirmed_later_container_anchor(tmp_path, damage):
    class LaterAnchorVision(DemoVision):
        calls = 0

        def observe(self, capture, **kwargs):
            result = super().observe(capture, **kwargs)
            source_reference = (kwargs.get('source_hint') or {}).get('visual_reference')
            if not source_reference:
                return result
            self.calls += 1
            if self.calls == 1:
                reference = deepcopy(source_reference)
                assert reference['observation_id'] != capture['observation_id']
                track = {'source': 'first_verified_decisions_registered_rgbd',
                    'reference_observation_id': reference['observation_id'],
                    'anchor_observation_id': capture['observation_id'],
                    'generation': capture['generation'], 'owner_id': reference['owner_id']}
                if damage == 'reference': track['reference_observation_id'] = 'wrong-reference'
                elif damage == 'anchor': track['anchor_observation_id'] = 'wrong-anchor'
                elif damage == 'generation': track['generation'] += 1
                elif damage == 'owner': track['owner_id'] = 'another-run'
                result['visual_reference'] = {**reference, 'container_identity_tracks': track}
            return result

    class FreshFirstObservation(TestPolicyController):
        observed_anchor = None

        async def execute(self, skill, context, arguments=None):
            first = skill == 'observe_scene' and self.algorithms.calls == 0
            if first:
                self.observation_epoch += 1  # A physical change requires a genuinely new capture.
            result = await super().execute(skill, context, {'capture_mode': 'fresh'} if first else arguments)
            if first:
                self.observed_anchor = deepcopy(context['visual_inventory_reference']['container_identity_tracks'])
            return result

    runner = FreshFirstObservation(DemoRobot(), LaterAnchorVision(1), None, tmp_path,
                                  planning_settings=TEST_SETTINGS)
    await runner.start('move all pink objects')
    await runner.worker
    if damage is None:
        assert runner.observed_anchor is not None, runner.state.get('error')
        assert runner.observed_anchor['reference_observation_id'] != runner.observed_anchor['anchor_observation_id']
        assert runner.task_progress.completed_ids == ['pink-1']
    else:
        assert runner.observed_anchor is None
        assert 'container_tracking_reference_mismatch' in runner.state['error']
