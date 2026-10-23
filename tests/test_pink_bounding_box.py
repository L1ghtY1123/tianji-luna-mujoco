"""The simplified pink body uses the CAD bounds for both RGB and contact."""
import mujoco
import numpy as np

from luna_mujoco.cad_assets import cad_asset
from luna_mujoco.model_builder import ModelBuilder, normalize_scene


def test_saved_original_cad_size_migrates_only_to_authorized_trimmed_bounds():
    old = [.1110853500366211, .06278736877441407, .11]
    scene = normalize_scene({'objects': [{'id': 'pink', 'kind': 'cad', 'asset_id': 'pink_wrapped',
        'position': [.5, .1, .7], 'size': old}]})
    size = scene['objects'][0]['size']
    assert .079 < size[0] < .080 and size[1:] == old[1:]


def test_pink_is_one_visible_physical_box_with_exact_cad_bounds():
    asset = cad_asset('pink_wrapped')
    builder = ModelBuilder()
    try:
        scene = normalize_scene({'objects': [{'id': 'pink', 'kind': 'cad',
            'asset_id': 'pink_wrapped', 'position': [.5, .1, .7], 'yaw_deg': 180,
            'color': 'pink', 'mass_kg': .08}]})
        model = builder.build(scene)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        indices = np.flatnonzero(model.geom_bodyid == model.body('object_pink').id)
        assert len(indices) == 1
        index = indices[0]
        assert model.geom_type[index] == mujoco.mjtGeom.mjGEOM_BOX
        assert model.geom_contype[index] and model.geom_conaffinity[index]
        assert model.geom_group[index] != 3
        np.testing.assert_allclose(model.geom_size[index] * 2, asset['size_m'], atol=1e-9)
        np.testing.assert_allclose(model.geom_pos[index], 0)
        assert model.body_gravcomp[model.body('object_pink').id] == 0
        assert model.joint('free_pink').type[0] == mujoco.mjtJoint.mjJNT_FREE
        rows = [r for r in builder.render_geometries(model, data) if r.get('object_id') == 'pink']
        assert len(rows) == 1 and rows[0]['type'] == 'box'
        assert 'mesh_url' not in rows[0]
    finally:
        builder.close()
