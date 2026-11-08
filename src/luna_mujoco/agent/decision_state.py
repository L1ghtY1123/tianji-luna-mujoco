"""Build the context sent with a single semantic question."""
from copy import deepcopy

from ..core.outcome_evidence import outcome_state, scalar_fields

TASK_FIELDS = {'task', 'object_task', 'environment_assumptions', 'target_id', 'destination_id'}
SCOPE_FIELDS = {
    'current_support': {'subject', 'support_candidates', 'current_measured_geometry', 'support_memory',
                        'support_hypothesis', 'rejected_support_hypothesis', 'interpretation'},
    'instance_registration': {'proposal', 'subject_geometry', 'reference_projection'},
    'input_membership': TASK_FIELDS | {'subject', 'subject_history'},
    'next_input_selection': TASK_FIELDS | {'selection_candidates', 'selection_support_geometry'},
    'mapping_update': TASK_FIELDS | {'mapping_evidence'},
    'intermediate_destination': TASK_FIELDS | {
        'intermediate_destination_id', 'destination_kind', 'final_destination_id', 'destination_evidence',
        'intermediate_placement_mode', 'declared_destination_references'},
    'execution_arm': TASK_FIELDS | {'feasible_arms', 'workspace_evaluations', 'robot_arm_names',
                                   'current_measured_entities', 'previous_same_target_failure'},
    'graph_recovery': {'failed_result', 'failed_node', 'subject', 'required_judgment'},
    'policy_condition': TASK_FIELDS | {'policy_bindings', 'last_result', 'last_failure', 'declared_branches',
        'declared_subgraphs', 'declared_continuations', 'current_facts', 'skill_contracts', 'selected_arm', 'arm_evaluations',
        'edge_selection_only', 'verified_obligations'},
    'task_completion': TASK_FIELDS | {'verified_obligations', 'verified_robot_pose'},
    'placement_region_scope': TASK_FIELDS | {'original_requirement', 'active_object_requirement',
        'bound_destination_ref', 'current_transfer', 'measured_region', 'configured_region_constraints',
        'declared_destination_references'},
    'placement_candidates': TASK_FIELDS | {'original_requirement', 'active_object_requirement',
        'bound_destination_ref', 'current_transfer', 'semantic_frames', 'physical_relation',
        'current_inventory_entities', 'occupancy_evidence', 'placement_candidates'},
    'placement_xy_proposal': {'scene_token', 'target_id', 'region_authorization', 'current_transfer',
        'placement_zone', 'subdivision_cells', 'xy_candidates', 'previous_candidate_failures', 'image_evidence_scope',
        'placement_preference', 'desired_clearance_m'},
}


def judgment_state(context, question):
    """Supply the evidence for this question, with no write access to its producers."""
    scope = context.get('verification_scope')
    if scope not in SCOPE_FIELDS:
        raise ValueError('decision_scope_not_declared:' + str(scope))
    fields = SCOPE_FIELDS[scope]
    if scope == 'task_completion' and question.name == 'source_has_remaining_input':
        fields = {'task'}
    elif scope == 'placement_region_scope' and question.name == 'placement_additional_spatial_constraint':
        fields = {'original_requirement', 'active_object_requirement', 'current_transfer', 'configured_region_constraints'}
    state = {'verification_scope': scope, **{key: deepcopy(value) for key, value in context.items() if key in fields}}
    if scope == 'instance_registration':
        state['proposal'] = {key: value for key, value in state['proposal'].items() if key in {
            'ref', 'existing_id', 'kind', 'color', 'description', 'appearance_keywords',
            'reference_camera_id', 'bbox'}}
    if scope == 'task_completion' and question.name == 'source_has_remaining_input':
        state['task'] = {key: value for key, value in state['task'].items()
                         if key in {'instruction', 'source_ref', 'target_selector'}}
    for key in ('last_result', 'last_failure', 'failed_result', 'previous_same_target_failure'):
        if key in state:
            state[key] = outcome_state(state[key])
    if isinstance(state.get('arm_evaluations'), dict):
        state['arm_evaluations'] = {arm: scalar_fields(evaluation)
                                    for arm, evaluation in state['arm_evaluations'].items()}
    return state
