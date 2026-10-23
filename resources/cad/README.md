# User CAD assets

The active originals in `source/` are byte-for-byte copies supplied by the
user from `C:/Users/l1ght/OneDrive/Desktop/CAD/`. SHA256 values, source bounds,
conversion parameters and every generated mesh hash are in `manifest.json`.
The STEP SI_UNIT and both STL headers explicitly declare millimetres.
`source/pink.stl` is retained only as the superseded original for historical
provenance; it is not an additional catalog choice. The active pink source is
`tripo_convert_891be992-2e91-4d83-b19a-dbbfcfa1acd1_flat_bottom_110mm.stl`.

| Asset ID | Native X × Y × Z (mm) | Original topology |
| --- | --- | --- |
| `blue_box` | 410 × 310 × 145 | One closed, consistently oriented solid; +Z opening |
| `pink_wrapped` | 111.085350 × 62.787369 × 110 | 19 face-connected surfaces, not watertight; inconsistent winding; authored flat base |
| `black_core` | 44 × 56 × 74.5 | Thirteen surface components, not watertight |

Coordinates retain the source axes. Subtract the recorded source AABB center,
then multiply by 0.001. Instance positions therefore mean the centered CAD AABB,
not center of mass. Visual triangle positions and winding are retained; no
hole filling, remeshing, normal reversal or source repair is performed. The
black file itself describes an assumed visual sample, not validated physics.

Rebuild with the existing numpy/scipy/trimesh Python environment and Node:

```text
python scripts/build_cad_assets.py --node <node-executable>
python scripts/build_cad_assets.py --asset pink_wrapped --node <node-executable>
```

`--asset` updates only the selected source/visual/collision records in an
existing manifest; unrelated asset files and records are preserved. The flat
pink replacement has 10,924 source triangles, including 368 triangles on Z=0
with -Z normals. Its source AABB center is [0,0,55] mm. The existing normalization
therefore places that base at local Z=-.055 m without altering any triangle.

The generator downloads the integrity-pinned `occt-import-js` 0.0.23 converter
only into `data/cad-build/` when absent. An existing verified module can instead
be supplied with `--occt-module <occt-import-js.js>`. Its public `ReadStepFile`
uses millimetre output, 0.1 mm absolute chord deflection and 0.15 rad angular
deflection. There is no converter, download or new dependency in the runtime
application. The [converter's official API](https://github.com/kovacsv/occt-import-js/blob/main/README.md)
documents these options. No OpenCascade or MuJoCo SDK source is modified.

MuJoCo 3.10 ordinary mesh collision uses convex hulls. A single hull of the
blue box would close its usable cavity. The generated collision representation
instead contains a separate floor hull and 24 angular wall sectors in two
height bands, 49 convex pieces total. External ribs and handle recesses within
a sector may be filled conservatively. Pink/black collision each uses eight
axial convex bands; local concavities, the wrapping's central opening and thread
detail are filled for rigid contact. The original visual mesh remains visible.
These are declared collision approximations, not repaired CAD solids.
[MuJoCo 3.10 mesh specification](https://mujoco.readthedocs.io/en/3.10.0/XMLreference.html#asset-mesh).

Masses are explicit demo settings (blue 0.5 kg, pink 0.08 kg, black 0.06 kg),
not inferred material densities. Each collision piece receives a volume share
of that total and uses convex inertia; visual geoms have zero mass and no
contact. The parts share one body/free joint. This permits transactional mock
replacement without changing robot/body/joint integration layout.
[MuJoCo model-editing API](https://mujoco.readthedocs.io/en/3.10.0/programming/modeledit.html).

The bin interior floor is centered Z=-0.0675 m. Its walls taper outward and
its bottom corners are rounded. `cavity_bottom_section` is the closed source
tessellation cut 1 micrometre above that floor; `cavity_floor_plus_1mm_section`
is a second cut 1 mm above it. Their XY polygons include inner corner rounds.
The latter is wider and must not be used as a claim that the bottom fillet is
empty at floor height. These asset-local shape priors contain no observed pose.
The independent QA combines the lower polygon with separately checked floor
and top bounds; runtime registration must still derive pose from current RGBD.

MuJoCo may recenter/rotate each mesh to its inertia frame during compilation.
`render_geometries` exposes the compiled geom-local position and quaternion for
preview composition. Display mesh URLs are content hashes, preserving historical
geometry when CAD replacement changes mesh indices.
