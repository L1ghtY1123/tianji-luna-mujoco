"""Shared workcell geometry used by the real and simulated Marvin systems."""

# ========================================
# MODIFICATION START: SDK_CUROBO shared scene geometry
# ========================================
SCENES = ("none", "pickplace")
# Shared by TSDF fusion, ESDF collision grids and voxel point reduction.
SCENE_VOXEL_SIZE_M = 0.01

TABLE_TOP_Z = 0.60
TABLE_THICK = 0.04
TABLE_CENTER = (0.50, 0.0)
TABLE_HALF = (0.30, 0.45)
LEG_HALF = 0.03
TABLE_COLOR = (0.45, 0.30, 0.18)

CUBE_SIZE = 0.05
CUBE_MASS = 0.08
CUBES = [
    (
        "cube_left",
        (0.38, 0.20, TABLE_TOP_Z + CUBE_SIZE / 2.0),
        (0.85, 0.15, 0.10),
    ),
    (
        "cube_right",
        (0.38, -0.20, TABLE_TOP_Z + CUBE_SIZE / 2.0),
        (0.10, 0.30, 0.85),
    ),
    (
        "cube_center",
        (0.55, 0.00, TABLE_TOP_Z + CUBE_SIZE / 2.0),
        (0.15, 0.70, 0.20),
    ),
]


def table_boxes():
    """Return the tabletop and four legs as center/half-extent pairs."""

    center_x, center_y = TABLE_CENTER
    half_x, half_y = TABLE_HALF
    slab_center_z = TABLE_TOP_Z - TABLE_THICK / 2.0
    boxes = [
        (
            (center_x, center_y, slab_center_z),
            (half_x, half_y, TABLE_THICK / 2.0),
        )
    ]
    leg_height = (TABLE_TOP_Z - TABLE_THICK) / 2.0
    for x_sign in (-1, 1):
        for y_sign in (-1, 1):
            boxes.append(
                (
                    (
                        center_x + x_sign * (half_x - LEG_HALF),
                        center_y + y_sign * (half_y - LEG_HALF),
                        leg_height,
                    ),
                    (LEG_HALF, LEG_HALF, leg_height),
                )
            )
    return boxes


def static_collision_boxes(scene):
    """Return named fixed obstacles; dynamic target cubes are excluded."""

    if scene not in SCENES:
        raise ValueError(f"unknown scene: {scene!r}")
    if scene == "none":
        return []
    return [
        (f"scene_table_{index}", tuple(center), tuple(half_extents))
        for index, (center, half_extents) in enumerate(table_boxes())
    ]
# ========================================
# MODIFICATION END: SDK_CUROBO shared scene geometry
# ========================================
