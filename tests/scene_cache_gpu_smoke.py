"""Run in an isolated pinned cuRobo process after coordinating the shared GPU.

Checks the production world-upload path with real CUDA graph replay. Synthetic
ESDF fields are an isolated collision-buffer fixture, never task evidence.
"""
import json
import time


def main():
    import torch
    from curobo.scene import VoxelGrid
    from tianji_curobo.contracts import CuroboToolError
    from tianji_curobo.curobo_service import materialize_urdf

    from jev_mujoco.geometry_service import ROOT, GeometryRuntime, sdk_identity

    assert sdk_identity()[0] == "8e734f3ced1df898990bcd92de40abce475907db"
    urdf = materialize_urdf(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf",
        marvin_meshes=ROOT / "assets/marvin_urdf/meshes", rag_meshes=ROOT / "assets/marvin_urdf/rag100_050/meshes")
    runtime = GeometryRuntime(urdf_path=urdf, asset_root_path=urdf.parent)
    runtime.geometry_scene = {"scene_token": "first", "target_id": "test-only", "disabled_obstacle_names": ()}
    runtime.geometry_model = {"voxel": {"background": {"dims": [.2, .2, .2], "voxel_size": .01}}}

    def grid(value):
        return VoxelGrid(name="background", pose=[.5, 0., .8, 1., 0., 0., 0.],
                         dims=[.2, .2, .2], voxel_size=.01, feature_dtype=torch.float16,
                         feature_tensor=torch.full((20, 20, 20), value, dtype=torch.float16, device="cuda:0"))

    runtime.geometry_grids = [grid(.1)]
    runtime._prepare_scene_cache_unlocked()
    started = time.perf_counter()
    backend = runtime._create_real_backend_unlocked("A", batch_size=1, ik_only=True)
    torch.cuda.synchronize()
    report = {"sdk_commit": sdk_identity()[0], "torch_version": torch.__version__,
              "backend_construction_wall_s": time.perf_counter() - started}
    graph = None
    try:
        runtime._ik_batch_backends["test"] = backend
        runtime._refresh_scene_unlocked(arm="A", backend=backend)
        voxels = backend.motion_planner.scene_collision_checker.data.voxels
        pointer = voxels.features.data_ptr()
        spheres = torch.tensor([[[[.5, 0., .8, .01]]]], device="cuda:0")
        checker = backend._motion_clearance_checker
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                checker.get_collision_distance(spheres)
        torch.cuda.current_stream().wait_stream(side_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_cost = checker.get_collision_distance(spheres)
        graph.replay()
        before = captured_cost.detach().cpu().clone()

        runtime.geometry_scene = runtime.geometry_scene | {"scene_token": "second"}
        runtime.geometry_grids = [grid(-.1)]
        started = time.perf_counter()
        runtime._prepare_scene_cache_unlocked()
        runtime._refresh_scene_unlocked(arm="A", backend=backend)
        torch.cuda.synchronize()
        report["warm_world_upload_wall_s"] = time.perf_counter() - started
        assert runtime._ik_batch_backends["test"] is backend
        assert voxels.features.data_ptr() == pointer
        graph.replay()
        after = captured_cost.detach().cpu().clone()
        eager = checker.get_collision_distance(spheres).detach().cpu().clone()
        assert torch.allclose(after, eager)
        assert float(torch.max(torch.abs(after - before))) > .01
        try:
            runtime._require_scene_token_unchanged_unlocked("first")
        except CuroboToolError as error:
            assert error.code == "CUROBO_SCENE_CHANGED"
        else:
            raise AssertionError("old scene token was accepted")
        report.update(same_solver=True, stable_voxel_pointer=True, captured_graph_reads_new_field=True,
                      before_cost=before.tolist(), after_cost=after.tolist(), old_trajectory_token_rejected=True)
        print(json.dumps(report), flush=True)
    finally:
        if graph is not None:
            graph.reset()
        runtime._clear_planners_unlocked()


if __name__ == "__main__":
    main()
