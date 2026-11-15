"""Small typed visual question context, with sensor images carried separately."""
from copy import deepcopy

SCOPE_FIELDS = {
    'current_support': {'subject', 'support_candidates', 'support_hypothesis'},
    'grasp_result': {'target_id', 'selected_arm', 'primary_wrist_camera_id', 'camera_roles', 'visual_camera_ids',
                     'holding_presence', 'visual_modalities', 'holding_segmentation'},
    'instance_registration': {'proposal', 'reference_states', 'current_measurement',
        'candidate', 'current_instances', 'description_ref', 'description', 'description_camera_id',
        'geometry_correspondence'},
    'input_membership': {'task', 'subject', 'subject_history'},
    'target_selection': {'task', 'candidate_objects'},
    'graph_recovery': {'subject', 'required_judgment'},
    'policy_condition': {'task', 'object_task', 'policy_bindings'},
    'task_completion': {'task', 'object_task'},
}


def judgment_state(context, question):
    scope = context.get('verification_scope')
    if scope not in SCOPE_FIELDS:
        raise ValueError('nonvisual_decision_scope_removed:' + str(scope))
    result = {'verification_scope': scope}
    if scope == 'grasp_result' and question.name in {'object_held', 'failure_gripper_holds'}:
        from ..robot_skills.grasp_verification import HOLDING_VISUAL_CRITERIA

        result['holding_visual_criteria'] = deepcopy(HOLDING_VISUAL_CRITERIA)
        if 'gripper_feedback' in context:
            result['gripper_feedback'] = deepcopy(context['gripper_feedback'])
    subject = context.get('condition_subjects', {}).get(question.name)
    if scope == 'task_completion' and subject is not None:
        result['condition_subject'] = deepcopy(subject)
    for key in SCOPE_FIELDS[scope]:
        if key in context:
            value = deepcopy(context[key])
            if key == 'subject':
                value = {k: v for k, v in value.items() if k in {'id', 'kind', 'color', 'description', 'bbox', 'reference_camera_id'}}
            result[key] = value
    return result
