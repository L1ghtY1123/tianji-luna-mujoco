import hashlib

from processing_fixtures import before_kwargs, capture_and_action, client, release_evidence, work_area

from luna_mujoco.robot_skills.processing_evidence import verify_unpacking


def test_verification_capture_binds_actual_rgb_and_time():
    capture, action = capture_and_action()
    visual, _ = client()
    result = verify_unpacking(visual, capture, target_id='pink', release_evidence=release_evidence(),
        action_evidence=action, owner_id='run', work_area=work_area(), **before_kwargs())
    summary = result['verification_capture']
    assert summary == result['visual_evidence']['verification_capture']
    assert summary['source'] == 'post_unpack_current_rgb'
    for key in ('observation_id', 'generation', 'time', 'captured_perf_counter_ns'):
        assert summary[key] == capture[key]
    for camera, image in summary['images'].items():
        rgb = capture['views'][camera]['rgb']
        assert image['rgb_sha256'] == hashlib.sha256(rgb.tobytes()).hexdigest()
        assert image['shape'] == list(rgb.shape)


def verified_result():
    capture, action = capture_and_action()
    action['process_id'] = 'process-1'
    visual, _ = client()
    result = verify_unpacking(visual, capture, target_id='pink', release_evidence=release_evidence(),
        action_evidence=action, owner_id='run', work_area=work_area(), **before_kwargs(action=action))
    return capture, action, result


def test_processing_output_has_independent_identity_and_verified_lineage():
    from luna_mujoco.robot_skills.processing import create_unpack_product

    capture, action, result = verified_result()
    product = create_unpack_product('pink', action, result, capture, owner_id='run')
    assert product['id'] not in {'pink', action['process_id']}
    assert product['input_entity_id'] == 'pink'
    assert product['process_id'] == action['process_id']
    assert product['creation_evidence']['observation_id'] == capture['observation_id']
    assert product['creation_evidence']['request_hash'] == result['request_hash']


def test_unconfirmed_mismatched_or_low_confidence_result_cannot_create_product():
    from copy import deepcopy

    import pytest

    from luna_mujoco.robot_skills.processing import create_unpack_product

    capture, action, result = verified_result()
    for field, value in [('success', False), ('target_id', 'other'), ('observation_id', 'old'),
                         ('generation', -1), ('request_hash', '')]:
        invalid = deepcopy(result)
        invalid[field] = value
        with pytest.raises(ValueError, match='product_requires_confirmed_visible_result'):
            create_unpack_product('pink', action, invalid, capture, owner_id='run')
    result['visual_evidence']['confidence'] = .79
    with pytest.raises(ValueError, match='product_requires_confirmed_visible_result'):
        create_unpack_product('pink', action, result, capture, owner_id='run')
