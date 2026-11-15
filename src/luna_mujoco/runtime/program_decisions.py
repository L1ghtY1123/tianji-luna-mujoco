"""Choices already determined by numeric feedback or an authored recovery route."""


def observation_recovery(choices, branches, *, needs_viewpoint_change, empty_hand, at_observation_pose):
    if not needs_viewpoint_change or not empty_hand or at_observation_pose:
        return 'stop' if 'stop' in choices else 'unknown'
    for branch in branches:
        if (branch['outcome'] == 'return' and 'return' in choices and branch['remaining_traversals'] > 0):
            return branch['outcome']
    return 'stop' if 'stop' in choices else 'unknown'


def observation_evidence(context, state):
    """Diagnose missing current IDs/depth without inferring occlusion from a failed probability."""
    observation = context.get('observation', {})
    objects = {obj['id']: obj for obj in observation.get('objects', [])}
    required = {identifier for identifier in (state.get('target_id'), state.get('destination_id')) if identifier}
    missing = sorted(required - objects.keys())
    geometry_missing = [identifier for identifier in required & objects.keys()
                        if objects[identifier].get('geometry_status') == 'missing'
                        or (objects[identifier].get('kind') not in {'table', 'marker'}
                            and not len(objects[identifier].get('points_world_m', [])))]
    views = context.get('latest_capture', observation).get('views', {})
    invalid_rgb = not views or any('rgb' not in view for view in views.values())
    held = context.get('holding')
    release = context.get('demo_place_execution', {})
    released = (release.get('released') is True and release.get('release_commanded') is True
                and not (held or {}).get('attached'))
    return {'needs_viewpoint_change': bool(missing or geometry_missing or invalid_rgb),
            'empty_hand': (not held and not state.get('facts', {}).get('grasp_held')
                           and not state.get('facts', {}).get('recovery_needed')) or released,
            'missing_instance_ids': missing, 'missing_geometry_ids': geometry_missing,
            'invalid_rgb': invalid_rgb}
