"""Measured gripper state, closed-hand vision and inherited target identity."""

import math

from ..core.decisions import JudgmentRequest
from ..raw_visual_input import CAMERA_ROLES

WRIST_CAMERAS = {'A': '2', 'B': '1'}
GRIPPER_ENDPOINT_TOLERANCE_FRACTION = 0.02

HOLDING_POSITION_RULE = (
    '程序按当前实测关节位置判定：完全张开（fully_open）为未持物；'
    '动作结束后既未全开又未全闭（partially_closed）为持物，不调用视觉模型；'
    '完全闭合（fully_closed）不能单凭位置判断，须看 CURRENT 图中两指之间是否有物体材料。'
    '全闭时薄物或受压变形的材料仍可能被夹住，也可能为空爪。'
    '仅完全闭合时调用 Decisions。使用同次实测位置，不把开合命令当成实测位置；反馈不可用时保留 unknown。'
)

HOLDING_VISUAL_CRITERIA = {
    'jaw_state': HOLDING_POSITION_RULE,
    'held': '完全闭合时，CURRENT 图中两指之间可见被夹持的物体材料。',
    'empty': '完全闭合时，CURRENT 图中两指之间清晰可见且没有物体材料。',
    'deformation': '全闭不等于空爪。柔软物体可被压薄、压扁或折叠，不要求保持抓取前厚度或轮廓。',
    'visibility': '结合指腹附近的轮廓和遮挡关系观察夹持处。远处桌面或盒内的物体属于背景。'
                  '深色或黑色物体也属于有效视觉证据。夹持处不可见时保留不确定性。',
    'annotations': '当前夹持局部 RGB 不带实例标签。直接观察两指夹持处的材料，持物判断不依赖 SAM3 检出。',
    'comparison': '持物题只使用 CURRENT 夹持接触区及同次实测反馈，不用不同姿态和尺度的历史图推断当前夹持。',
}

def inherit_grasp_identity(holding, execution, capture, *, target_id, arm, owner_id, motion_epoch, action_epoch):
    """Validate execution continuity; this is not a post-grasp visual identity verdict."""
    expected = {'arm': arm, 'owner_id': owner_id, 'generation': capture['generation']}
    if (not target_id or not owner_id or arm not in {'A', 'B'}
            or not capture.get('observation_id') or capture.get('observation_id') == holding.get('observation_id')
            or type(motion_epoch) is not int or motion_epoch < 0
            or type(action_epoch) is not int or action_epoch < 0
            or holding.get('instance_id') != target_id
            or execution.get('target_id') != target_id
            or any(holding.get(key) != value or execution.get(key) != value
                   for key, value in expected.items())
            or execution.get('completed') is not True
            or execution.get('motion_epoch') != motion_epoch
            or execution.get('action_epoch') != action_epoch
            or any(not holding.get(key) or execution.get(key) != holding[key]
                   for key in ('observation_id', 'scene_token'))):
        raise ValueError('grasp_identity_execution_chain_broken')
    completed = execution.get('completed_feedback_time_s')
    current = capture.get('time')
    if (type(completed) not in (int, float) or not math.isfinite(completed)
            or type(current) not in (int, float) or not math.isfinite(current) or current < completed):
        raise ValueError('grasp_identity_capture_precedes_execution')
    reference = holding.get('identity_reference', {}).get('identity_references', {}).get(target_id, {})
    if (not reference.get('observation_id') or not reference.get('request_hash')
            or reference.get('source') != 'openai_decisions'
            or reference.get('generation') != capture['generation']):
        raise ValueError('grasp_target_reference_missing')
    return {'source': 'continuous_grasp_execution', 'target_id': target_id, **expected,
            'source_observation_id': holding['observation_id'],
            'reference_observation_id': reference['observation_id'],
            'holding_observation_id': capture['observation_id'],
            'scene_token': holding['scene_token'], 'motion_epoch': motion_epoch, 'action_epoch': action_epoch,
            'completed_feedback_time_s': completed, 'visual_identity_verified': False,
            'assumption': 'The confirmed pregrasp target was acquired by its bound uninterrupted grasp sequence.'}


def gripper_feedback_state(capture, arm):
    """Present same-capture joint feedback without deriving a holding verdict."""
    feedback = capture.get('arms', {}).get(arm, {})
    fraction = feedback.get('gripper_closed_fraction')
    result = {'observation_id': capture['observation_id'], 'generation': capture.get('generation'),
              'time_s': capture.get('time'), 'arm': arm,
              'interpretation': HOLDING_POSITION_RULE}
    if type(fraction) in (int, float) and math.isfinite(fraction) and 0 <= fraction <= 1:
        if fraction <= GRIPPER_ENDPOINT_TOLERANCE_FRACTION:
            position_state = 'fully_open'
        elif fraction >= 1 - GRIPPER_ENDPOINT_TOLERANCE_FRACTION:
            position_state = 'fully_closed'
        else:
            position_state = 'partially_closed'
        result['measured'] = {'closed_fraction': fraction,
            'position_state': position_state,
            'endpoint_tolerance_fraction': GRIPPER_ENDPOINT_TOLERANCE_FRACTION,
            'scale': '关节位置归一化值，0 为机械全开，1 为机械全闭。该数值不表示夹持力。'}
    else:
        result['measured'] = {'status': 'unavailable'}
    commanded = feedback.get('gripper_closed_command')
    if type(commanded) is bool:
        result['command'] = {'full_close_requested': commanded,
            'meaning': '是否下发到机械全闭端的目标。命令只说明动作意图，持物判断使用实测位置。'}
    return result


def holding_from_feedback(capture, arm):
    """Open and partial closure are measured facts; only full closure needs vision."""
    feedback = gripper_feedback_state(capture, arm)
    position = feedback['measured'].get('position_state')
    return {'status': {'fully_open': 'no', 'partially_closed': 'yes'}.get(position, 'unknown'),
            'requires_visual': position == 'fully_closed', 'source': 'measured_gripper_position',
            'reason': position or 'gripper_feedback_unavailable', 'feedback': feedback}


def gripper_evidence_context(capture, evidence):
    """Bind annotations to the frozen capture used by the visual question."""
    if any(evidence.get(key) != capture.get(key) for key in ('observation_id', 'generation')):
        raise ValueError('gripper_evidence_capture_mismatch')
    return {'visual_masks': evidence['visual_masks'], 'holding_segmentation': evidence['attempts']}


def grasp_visual_context(capture, holding):
    return {'verification_scope': 'grasp_result', 'target_id': holding['instance_id'],
            'selected_arm': holding['arm'], 'primary_wrist_camera_id': WRIST_CAMERAS[holding['arm']],
            'camera_roles': CAMERA_ROLES, 'visual_modalities': ['rgb']}


def object_held_question(arm):
    return JudgmentRequest(name='object_held',
        instructions=f'执行臂 {arm} 已实测完全闭合。CURRENT 图中两指之间仍夹持着物体材料。'
            '薄物或受压变形材料也算持物；清晰可见且无材料为否，夹持处不可见时保留未知。')
