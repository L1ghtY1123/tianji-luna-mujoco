"""Bind current RGB to each release fact; share historical identity only."""
import json
from copy import deepcopy


def text_part(text):
    return {'type': 'input_text', 'text': text}


def scoped_release_payload(payload, support_id):
    """For a region already authorized in full, do not rejudge the overall task."""
    content = payload['input'][0]['content']
    context = json.loads(content[0]['text'])
    raw, annotated, camera_views = [], [], {}
    for index, part in enumerate(content):
        if part['type'] != 'input_text' or not part['text'].startswith('CURRENT RGB camera '):
            continue
        pair = deepcopy(content[index:index + 2])
        raw.extend(pair)
        camera = part['text'].rsplit(' ', 1)[1]
        camera_views[camera] = pair
        following = content[index + 2] if index + 2 < len(content) else {}
        if 'release_target_region_reference' in following.get('text', ''):
            annotated.extend([text_part('CURRENT camera ' + camera
                + ': the cyan polygon is the allowed destination region on the support surface.'),
                deepcopy(content[index + 3])])
        else:
            annotated.extend(pair)

    shared_context = {key: deepcopy(context[key]) for key in (
        'observation_id', 'binding_id', 'target_id', 'destination_id', 'appearance',
        'destination_and_reference_appearances', 'placement_intent') if key in context}
    has_reference_image = content[2].get('type') == 'input_image'
    reference_scope = (
        'REFERENCE identity only. The reference crop predates this move; its background and old '
        'position do not describe the current scene.' if has_reference_image else
        'The supplied appearance describes the bound target identity. No historical reference image is supplied.')
    shared = [text_part(json.dumps(shared_context)), text_part(reference_scope), deepcopy(content[2])]
    appearance = context['appearance']
    target = ' '.join(str(appearance.get(key, '')) for key in ('color', 'kind')).strip() or 'target object'
    target += ' shown in REFERENCE' if has_reference_image else ' bound as the target'
    destination = context.get('destination_and_reference_appearances', {}).get(support_id, {})
    destination = ' '.join(str(destination.get(key, '')) for key in ('color', 'kind')).strip() or 'identified destination'
    relation = 'inside' if context.get('placement_intent', {}).get('relation') == 'inside' else 'on'
    # Fixed calibrated rig: A/left uses wrist camera 2; B/right uses camera 1.
    arm = context['measured_gripper_state']['arm']
    side, wrist = {'A': ('left', '2'), 'B': ('right', '1')}[arm]
    statements = {
        'place_result': f'In these CURRENT camera images, is the {target} {relation} the {destination}? '
            'Judge the visible location only; gripper retention is a separate question.',
        'placement_near_target': 'In these CURRENT images, is the movable reference object inside or approximately near the cyan outlined destination region? '
            'The polygon is a reference on the support plane. Compare the object base, allowing perspective and object height. The whole region is acceptable.',
        'workpiece_still_held': f'In these CURRENT images, is the {target} physically held by the {side} gripper '
            f'(arm {arm}, wrist camera {wrist})? Camera 1 is the right wrist, camera 2 is the left wrist, and camera 3 is the head. '
            'Judge this active gripper only; the other arm being empty does not mean this gripper is empty. '
            'Judge the object and the fingers together using the current camera views and actual finger opening. '
            'The feedback describes the gripper, not the object location. '
            + json.dumps({'current_gripper_feedback': context['measured_gripper_state']}),
    }
    questions = []
    for question in payload['questions']:
        name = question['name']
        instructions = [text_part(statements[name]
            + ' Use these views together; one clear view is sufficient, and a cropped view does not negate it. '
            'If none of these views establishes the relationship, remain uncertain.')]
        views = annotated if name == 'placement_near_target' else raw
        if name == 'workpiece_still_held':
            # End the evidence sequence with the active hand's direct view,
            # retaining the head and peer wrist for cross-view comparison.
            order = [camera for camera in camera_views if camera != wrist] + [wrist]
            views = [part for camera in order for part in camera_views.get(camera, [])]
        instructions.extend(deepcopy(views))
        questions.append({**question, 'instructions': instructions})
    return {**payload, 'input': [{'role': 'user', 'content': shared}], 'questions': questions}
