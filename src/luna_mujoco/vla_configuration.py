"""Public VLA input constraints, independent of task order and simulation truth."""
import json
import math
from pathlib import Path


def default_vla_input_xy():
    from .demo_scene import demo_region
    return demo_region()['center_world_xy_m']


def default_vla_input_size():
    from .demo_scene import demo_region
    return demo_region()['size_m']


def vla_unpack_configuration(settings=None):
    from .demo_scene import demo_region

    values = settings.model_dump() if hasattr(settings, 'model_dump') else (settings or {})
    configured = values.get('vla_unpack', {})
    defaults = default_vla_input_xy()
    xy = [configured.get('input_x_m', defaults[0]), configured.get('input_y_m', defaults[1])]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in xy):
        raise ValueError('vla_input_xy_invalid')
    region = demo_region()
    size = configured.get('input_size_m', region['size_m'])
    if type(size) not in (int, float) or not math.isfinite(size) or not .02 <= size <= .5:
        raise ValueError('vla_input_size_invalid')
    region['size_m'] = size
    capability = json.loads((Path(__file__).resolve().parents[2] / 'resources/vla_unpack.json').read_text())
    preparation = capability['preparation_joints_deg']
    half = region['size_m'] / 2
    region.update(region_id='vla-unpack-input', center_world_xy_m=xy,
        region_polygon_world_xy_m=[[xy[0] + dx, xy[1] + dy]
                                  for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half))],
        source='configured_vla_input_intent', precise_xy=False, physical_marker=False,
        required_robot_configurations=[{'id': 'vla_unpack_preparation',
            'scope': 'specified_arms', 'joints_deg': preparation, 'gripper_positions_rad': {'A': 0., 'B': 0.}}])
    return {'required_arms': ['A', 'B'], 'input_xy_m': xy,
        'input_position_constraint': 'object_within_configured_square',
        'input_orientation_tolerance_rad': .05,
        'input_region_id': 'vla-unpack-input', 'input_region': region,
        'input_orientation': 'preserve_existing_grasp_orientation',
        'preparation_joints_deg': preparation,
        'visibility_requirement': ('Both arms must reach the fixed VLA preparation joints before the effect. '
            'The output is random within the configured right tabletop region. '
            'Visibility still requires current three-camera verification. '
            'The effect invalidates the map; an unverified hidden output cannot authorize a blind recovery motion.'),
        'output_region': capability['output_region'],
        'post_effect_arm_configuration': 'unchanged_from_preparation; no automatic arm return',
        'output_position': 'unknown_until_current_visual_measurement'}
