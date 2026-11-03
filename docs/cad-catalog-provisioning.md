2026-11-02：当前资产、直接位姿、分层XY与观测复用规则以[本轮策略](cad-runtime-policy-20261102.md)为准，覆盖下文历史检查链路。

# CAD catalog references and FoundationPose registry

`resources/cad_catalog.json` binds the two user-configured rigid model descriptions,
source hashes and static preview hashes. `scripts/render_cad_catalog.py` reads the
verified metric mesh triangles through `CadCatalog.geometry`, uses orthographic
CPU depth-buffer rasterization, and writes three-view reference sheets. They contain
no scene pose, simulator object or generated geometry. Colour is configured display
material, not evidence inferred from the CAD file. The black model is a hollow
rectangular housing with corner holes/columns and a short threaded protrusion.

Preview generation requires NumPy, Trimesh, Pillow and Matplotlib. Run from the
project environment with Matplotlib available:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/render_cad_catalog.py
```

The script updates both preview hashes. Existing configured images are read only
after their SHA256 matches the catalog. Inspect the new PNGs before accepting a
catalog geometry or renderer change.

After a CAD match and its current pose both pass validation, the resolver retains
the model association under `(generation, permanent instance ID)`. Later
identity-confirmed observations of that same instance reuse only the stable model
choice. They still perform FoundationPose and numerical geometry validation on the
new frozen RGBD. Match evidence retains its original observation ID and separately
records the current identity binding. Unknown matches and failed initial pose fits
are not cached; a generation or source-hash change invalidates reuse. Two instances
of the same box model have independent cache entries.

## Registry maintained in tianji-algorithms

`../tianji-algorithms/scripts/provision_luna_registry.py` uses the unchanged public
`tianji_foundationpose.registry.Registry` API bundled in
`jodell-foundationpose:0.2.2-guangyu`, pinned to FoundationPose commit
`a1b694b83e633c2cb6115b9063d940a687759392`.

It validates the CAD library index, source file hashes and units against the project
catalog and geometry manifest before writing. It copies the immutable, verified blue
registration and cache from the old read-only registry. It registers the original
black STL with `Registry.register('black_core', source, 'mm', 'rigid_reference')`.
Both results must retain source axes, the expected source-to-canonical translation
and expected dimensions. Existing registrations are verified and reused; changed
geometry is rejected. Source repositories and the seed registry are never modified.

The helper uses CPU only: NumPy, Pillow and Trimesh are already in this image. It
does not import the inference backend, PyTorch or CUDA, and requires no GPU flag.
Blue STEP conversion is unnecessary because its verified cache is copied; gmsh or
CADQuery is not needed by this command. No image build is needed.

On Ubuntu, from `/home/guangyu/projects/tianji-luna-mujoco`, after making room within
the project's combined four-core quota, the coordinator can run:

```bash
mkdir -p ../tianji-algorithms/data/luna/foundationpose-registry
docker run --rm --network none --cpus 0.25 --memory 2g \
  --user "$(id -u):$(id -g)" --read-only --tmpfs /tmp:rw,size=256m \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -v /home/guangyu/projects/tianji-luna-mujoco:/project:ro \
  -v /home/guangyu/projects/tianji-algorithms/scripts:/algorithm-scripts:ro \
  -v /home/guangyu/projects/tianji-algorithms/data/luna/foundationpose-registry:/data/registry:rw \
  -v /home/guangyu/projects/tianji-algorithms/data/foundationpose-registry:/seed-registry:ro \
  -v /home/guangyu/projects/CAD:/cad:ro \
  jodell-foundationpose:0.2.2-guangyu \
  /algorithm-scripts/provision_luna_registry.py \
  --project /project --registry /data/registry --seed-registry /seed-registry --cad-root /cad
```

The image entrypoint is already `python3`, and its `PYTHONPATH` is
`/opt/algorithm-services`. The output contains both validated model records. Mount
this project's resulting registry read-only when the coordinator recreates the
FoundationPose service. Provisioning does not start or change that service.

Expected source-to-canonical translations in metres are blue `(0, 0, +0.0725)`
and black `(0, +0.006, -0.12525)`. Registry meshes use millimetres internally; the
existing inference backend scales them by `0.001`. Project planning meshes remain
the separately verified canonical metre meshes.

## Verification performed

The script ran locally against an audit copy of the old blue registry and the
read-only `C:/project/CAD` sources, using the unchanged package from
`C:/project/tianji-algorithms/services`. Both model records passed source hashes,
dimensions and canonical transforms. Local verification output is under
`data/cad-obb-iteration-20261030/pose-audit/provision-check`. Registry tests reject
wrong canonical origins and paths escaping the declared CAD root. Both previews
were visually inspected after rendering. No container was started or altered for
this artifact preparation; deployment and live FoundationPose validation are
separate coordinator steps.
