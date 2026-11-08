"""Measured placement regions bound to their support entities."""

def identify_placement_regions(observation):
    """Region identity follows the stable support entity, never detection order."""
    for obj in observation['objects']:
        reference = obj.get('placement_reference')
        if reference is None:
            continue
        reference.update(region_id='region:' + obj['id'], support_entity_id=obj['id'],
                         observation_id=observation.get('observation_id'),
                         plane={'normal_world': [0., 0., 1.], 'height_m': reference['floor_z_m']})
