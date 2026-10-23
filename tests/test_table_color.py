"""Table appearance survives scene validation and actual MuJoCo compilation."""
from copy import deepcopy

import mujoco
import numpy as np
import pytest

from luna_mujoco.model_builder import DEFAULT_SCENE, ModelBuilder, color_rgba, normalize_scene


def test_old_scene_defaults_to_white_table():
    scene = deepcopy(DEFAULT_SCENE)
    scene['table'].pop('color', None)
    assert normalize_scene(scene)['table']['color'] == 'white'


@pytest.mark.parametrize('color', ['white', '#3a7b91'])
def test_table_color_reaches_physics_render_model(color):
    scene = deepcopy(DEFAULT_SCENE)
    scene['objects'] = []
    scene['table']['color'] = color
    normalized = normalize_scene(scene)
    assert normalized['table']['color'] == color
    builder = ModelBuilder()
    try:
        model = builder.build(normalized)
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'table')
        np.testing.assert_allclose(model.geom_rgba[geom], color_rgba(color), atol=1e-7)
    finally:
        builder.close()


def test_invalid_table_color_is_rejected_before_scene_changes():
    scene = deepcopy(DEFAULT_SCENE)
    scene['table']['color'] = 'invalid-color'
    with pytest.raises(ValueError, match='unsupported color'):
        normalize_scene(scene)
