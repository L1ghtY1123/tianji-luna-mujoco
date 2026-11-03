# Measured table surface support

The initial inventory table is fitted from registered RGBD inside the model's
table rectangle, with foreground/robot exclusions. A rectangle is a proposal,
so a return near the fitted plane does not by itself establish table membership.
Both the main-view seed and supplemental views now use the same existing local
surface test before their points can determine the table cuboid's footprint.

The test uses 16 nearest measured neighbours within 20 mm. Their covariance must
have a second eigenvalue greater than `1e-6 m²`, smallest eigenvalue at most one
tenth of the second, and a local normal within 5 degrees of the fitted plane's
normal. Neighbourhoods are formed in the original 30 mm plane neighbourhood,
before the final 6 mm plane-distance test. This prevents a thin height slice of
a vertical wall from masquerading as a horizontal supporting patch. These are
the unchanged supplemental-view thresholds, now shared by the seed.

RANSAC's 256 iterations, 6,000-point fitting sample, 6 mm plane threshold and
5-degree maximum tilt remain unchanged. At least 100 supported seed points are
required; insufficient local support raises `demo_inventory_table_unobservable`.
There is no sparse-data fallback that fabricates a support surface. The result
records `fit.seed_surface_support` input/accepted counts and thresholds. Raw
RGBD stays unchanged. Color, catalog assets, task destinations and simulation
geometry are not inputs to the surface test.

The published footprint remains the existing axis-aligned envelope of accepted
measured surface points, with the existing bounded downward cuboid completion.
It is an explicit collision approximation, not evidence that every interior
pixel was observed. This change does not turn an arbitrary concave or holed
support into a verified solid surface, enlarge placement permissions, modify
collision clearance, or prove a placement trajectory.

## Frozen B4 regression

Run `a60ee0113c914cd9a38e2a3edb98b8cb` used a historical table reference and
per-camera foreground rectangles. The right bin's reference rectangle was in
camera 1, while the camera 3 table rectangle also contained a strip of its wall.
Camera 3 pixel `(row=218, column=586)` measured
`(0.594811678, -0.560602486, 0.568226457) m`. Its local normal is 87.322 degrees
from the fitted table plane. The former height-only seed rule admitted this
wall strip and set the table's minimum Y to `-0.560602486 m`.

The frozen pre-fix CPU replay reproduces every published fit field, including
counts and bounds. Applying the shared support test gives minimum Y
`-0.425992548 m`, dimensions approximately `0.599952 × 0.799980 m`, and preserves
the measured top Z `0.571999967 m`. This is a measurement classification fix;
the resulting scene has not been accepted by a live placement run.

`tests/fixtures/black_b4_table_seed.{json,npz,png}` contains the original frozen
depth/calibration, original head RGB and archive hashes. Regressions check the
wall cannot expand the cuboid, measured horizontal table points remain, dense
horizontal extents are preserved, sparse coplanar points fail closed, and inputs
are not mutated. The CPU attribution script and B3/B4 comparison are under
`data/cad-obb-iteration-20261030/black-B4-placement/`.
