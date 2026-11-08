"""Independent release facts sharing one current image per camera."""
import json
from copy import deepcopy


def text_part(text):
    return {'type': 'input_text', 'text': text}


def scoped_release_payload(payload, support_id):
    """Judge only the requested location or placement constraint."""
    if len(payload['questions']) != 1:
        raise ValueError('release_request_requires_one_question')
    content = payload['input'][0]['content']
    context = json.loads(content[0]['text'])
    question_name = payload['questions'][0]['name']
    current_views, region_views = [], []
    outlined_destination = False
    for index, part in enumerate(content):
        if part['type'] != 'input_text' or not part['text'].startswith('CURRENT RGB camera '):
            continue
        pair = deepcopy(content[index:index + 2])
        camera = part['text'].rsplit(' ', 1)[1]
        following = content[index + 2] if index + 2 < len(content) else {}
        if 'release_target_region_reference' in following.get('text', ''):
            outlined_destination = True
            region_views.extend([text_part('CURRENT camera ' + camera
                + ': the cyan polygon is the allowed destination region on the support surface.'),
                deepcopy(content[index + 3])])
        else:
            current_views.extend(pair)
    # Present the question's spatial reference first; retain every other camera.
    current_views = region_views + current_views

    has_reference_image = content[2].get('type') == 'input_image'
    reference_scope = (
        'REFERENCE identity only. The reference crop predates this move; its background and old '
        'position do not describe the current scene.' if has_reference_image else
        'The supplied appearance describes the bound target identity. No historical reference image is supplied.')
    appearance = context['appearance']
    target = ' '.join(str(appearance.get(key, '')) for key in ('color', 'kind')).strip() or 'target object'
    target += ' shown in REFERENCE' if has_reference_image else ' bound as the target'
    destination_record = context.get('destination_and_reference_appearances', {}).get(support_id, {})
    destination = (destination_record.get('description')
        or ' '.join(str(destination_record.get(key, '')) for key in ('color', 'kind')).strip() or 'identified destination')
    if outlined_destination:
        destination += ' identified by its cyan outlined destination region'
    shared_context = {'observation_id': context['observation_id'],
                      'target_appearance': deepcopy(appearance),
                      'destination_id': support_id, 'destination': destination}
    constrained = context.get('placement_semantic_scope') != 'entire_region'
    if constrained and question_name in {'placement_near_target', 'place_result'}:
        shared_context.update({key: deepcopy(context[key]) for key in (
            'placement_intent', 'original_placement_requirement', 'active_object_placement_requirement',
            'current_transfer', 'target_region_reference', 'inventory_reference_descriptions') if key in context})
    shared = [text_part(json.dumps(shared_context)), text_part(reference_scope), deepcopy(content[2]), *current_views]
    relation = 'inside' if context.get('placement_intent', {}).get('relation') == 'inside' else 'on'
    statements = {
        'place_result': f'In these CURRENT camera images, is the {target} {relation} the {destination}? '
            'Judge the visible location only.',
        'placement_near_target': f'In the CURRENT images, is the {target} approximately within the cyan outlined '
            f'region {relation} the {destination}? Judge where this object rests on the surface. '
            'Allow small placement error and perspective due to object height.',

    }
    questions = [{**question, 'instructions': statements[question['name']]
        + ' Use all three CURRENT views. One clear view is sufficient.'} for question in payload['questions']]
    if constrained and question_name in {'placement_near_target', 'place_result'}:
        # Preserve the authored spatial requirement; the scope projection must not weaken it.
        questions = deepcopy(payload['questions'])
    return {**payload, 'input': [{'role': 'user', 'content': shared}], 'questions': questions}
