"""Single process web workbench; MuJoCo owns its independent stepping thread."""
import argparse
import asyncio
import copy
import os
import time
from contextlib import asynccontextmanager
from functools import partial
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .experiment_api import experiment_router
from .experiments import ExperimentOptions, ExperimentStore
from .planning_settings import PlanningSettings, load_settings, save_settings
from .runner import public_summary

ROOT = Path(__file__).resolve().parents[2]


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instruction: str = Field(min_length=1, max_length=200)
    generation: int | None = Field(default=None, ge=1)
    settings_revision: str | None = None
    experiment: ExperimentOptions = Field(default_factory=ExperimentOptions)


class SettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    settings: PlanningSettings
    expected_revision: str


class DemoPresetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object_count: int = Field(default=2, ge=1, le=4, strict=True)


def create_app(simulation=None, algorithms=None, decisions_factory=None, data_dir=None, *,
               experiment_dir=None, demo_mode=True):
    server_instance_id = str(uuid4())
    if not demo_mode:
        raise ValueError("Only the Luna Responses program controller is available")
    load_dotenv(ROOT / ".env", override=False)
    if simulation is None:
        from .demo_scene import demo_observation_poses, demo_scene
        from .simulation import Simulation
        simulation = Simulation(scene=demo_scene(), initial_joints=demo_observation_poses())
    default_visual_inventory = algorithms is None
    if algorithms is None:
        from .demo_algorithms import DemoAlgorithms
        from .demo_perception import DemoPerception
        algorithms = DemoAlgorithms(perception=DemoPerception.from_environment())
    run_directory = Path(data_dir) if data_dir is not None else ROOT / "data" / "runs"
    settings_path = run_directory.parent / "planning-settings.json"
    settings = load_settings(settings_path)
    from .demo_scene import demo_observation_poses
    settings = PlanningSettings.model_validate(
        {**settings.model_dump(), "observation": demo_observation_poses(),
         "attachment": {**settings.attachment.model_dump(),
                        **({"sphere_budget": 24} if not settings_path.exists() else {})}})
    if not settings_path.exists() and os.environ.get("LUNA_PLAN_TIMEOUT_S"):
        settings = PlanningSettings.model_validate(
            {**settings.model_dump(), "plan_timeout_s": float(os.environ["LUNA_PLAN_TIMEOUT_S"])})
    from .visual_assistance import VisualAssistance
    visual_assistance = None if default_visual_inventory else VisualAssistance.from_environment()
    experiment_root = experiment_dir or os.environ.get("LUNA_EXPERIMENT_DIR") or (
        Path(data_dir) / "experiments" if data_dir is not None else
        ROOT / "data" / "experiments")
    experiments = ExperimentStore(experiment_root)
    from .demo_runner import DemoRunner
    runner = DemoRunner(simulation, algorithms, decisions_factory, run_directory,
                    planning_settings=settings, visual_assistance=visual_assistance, experiments=experiments)
    operations = asyncio.Lock()
    health_cache = {"at": 0., "value": {}}
    from .workspace import WorkspaceAtlas
    workspace_atlas = WorkspaceAtlas()
    camera_lock = asyncio.Lock()
    camera_frames = {}
    preview_frames = {}

    @asynccontextmanager
    async def lifespan(app):
        simulation.start()
        yield
        await runner.cancel()
        if runner.planning_worker is not None:
            await asyncio.gather(asyncio.shield(runner.planning_worker), return_exceptions=True)
        if runner.algorithm_cleanup_worker is not None:
            await asyncio.shield(runner.algorithm_cleanup_worker)
        if callable(getattr(algorithms, 'close', None)):
            await asyncio.to_thread(algorithms.close)
        await asyncio.to_thread(experiments.close)
        await asyncio.to_thread(simulation.close)

    app = FastAPI(title="Tianji Luna MuJoCo", lifespan=lifespan)
    app.state.runner = runner
    app.state.experiments = experiments
    app.include_router(experiment_router(experiments))

    def state(event_start=0):
        run = {**copy.deepcopy(runner.state), "algorithm_pending": runner.algorithm_pending,
               "algorithm_cleanup_error": runner.algorithm_cleanup_error}
        if runner.experiment is not None:
            run["experiment"] = runner.experiment.summary()
        holding = runner.holding_state
        run["holding"] = None if holding is None else public_summary({
            key: holding[key] for key in ("state", "arm", "verified", "attached", "reason", "observation_id",
                "verification", "last_visual_evidence", "attachment_evidence", "attachment_detach_pending")
            if key in holding})
        return {"simulation": simulation.snapshot(), "run": run,
                "events": copy.deepcopy(runner.events[event_start:]), "server_time": time.time(),
                "server_instance_id": server_instance_id,
                "planning_settings": settings_state()}

    def settings_state():
        return {"settings": runner.planning_settings.model_dump(),
                "revision": runner.planning_settings.revision()}

    @app.get("/api/planning-settings")
    async def get_planning_settings():
        return {**settings_state(), "schema": PlanningSettings.model_json_schema()}

    @app.post("/api/planning-settings")
    async def set_planning_settings(request: SettingsRequest):
        async with operations:
            if runner.active:
                raise HTTPException(409, "run_active_reset_first")
            if runner.algorithm_pending:
                raise HTTPException(409, "algorithm_request_pending")
            if request.expected_revision != runner.planning_settings.revision():
                raise HTTPException(409, "planning_settings_changed")
            try:
                await asyncio.to_thread(save_settings, settings_path, request.settings)
            except OSError as error:
                raise HTTPException(503, "planning_settings_save_failed") from error
            runner.planning_settings = request.settings
            return {**settings_state(), "schema": PlanningSettings.model_json_schema()}

    @app.get("/api/state")
    async def get_state():
        return state()

    @app.get("/api/run-progress")
    async def get_run_progress():
        # Polling a timer must not serialize all historical clouds/plans again.
        return {"run": copy.deepcopy({key: runner.state.get(key) for key in
            ("run_id", "status", "current_step", "target_id", "error", "metrics")}),
            "server_time": time.time()}

    @app.get("/api/manifest")
    async def manifest():
        return await asyncio.to_thread(simulation.manifest)

    @app.get("/api/cad-assets")
    async def cad_assets():
        from .cad_assets import cad_catalog

        return cad_catalog()

    @app.get("/api/camera-preview")
    async def camera_preview():
        capture = await asyncio.to_thread(simulation.preview_rgb)
        if capture is None or capture["generation"] != simulation.robot_snapshot()["generation"]:
            raise HTTPException(503, "camera_preview_pending")
        identifier = capture["preview_frame_id"]
        preview_frames[identifier] = capture
        while len(preview_frames) > 2:
            preview_frames.pop(next(iter(preview_frames)))
        labels = {"2": "左腕", "3": "头部", "1": "右腕"}
        return {"preview_frame_id": identifier, "generation": capture["generation"], "time": capture["time"],
                "cameras": [{"id": c, "label": labels[c], "frame_id": identifier,
                             "rgb_url": f"/api/camera-preview/{c}/rgb?frame_id={identifier}"}
                            for c in ("2", "3", "1")]}

    @app.get("/api/camera-preview/{camera_id}/rgb")
    async def preview_image(camera_id: str, frame_id: str):
        capture = preview_frames.get(frame_id)
        if capture is None or capture["generation"] != simulation.robot_snapshot()["generation"]:
            raise HTTPException(409, "camera_frame_changed")
        if camera_id not in capture["views"]:
            raise HTTPException(404, "unknown_camera")
        def encode():
            from PIL import Image
            buffer = BytesIO()
            Image.fromarray(capture["views"][camera_id]["rgb"]).save(buffer, format="JPEG", quality=85)
            return buffer.getvalue()
        return Response(await asyncio.to_thread(encode), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/cameras")
    async def cameras():
        async with camera_lock:
            capture = simulation.latest_rgbd()
            if capture is None or capture["generation"] != simulation.robot_snapshot()["generation"]:
                raise HTTPException(503, "camera_observation_unavailable")
            identifier = capture["observation_id"]
            camera_frames[identifier] = capture
            while len(camera_frames) > 2:
                camera_frames.pop(next(iter(camera_frames)))
            labels = {"1": "右腕 RGBD", "2": "左腕 RGBD", "3": "头部 RGBD"}
            return {"observation_id": identifier, "generation": capture["generation"], "time": capture["time"],
                    "object_labels": copy.deepcopy(runner.state.get('object_labels'))
                    if runner.state.get('object_labels', {}).get('observation_id') == identifier else None,
                    "cameras": [{"id": camera_id, "label": labels[camera_id], "frame_id": identifier,
                                 "rgb_url": f"/api/cameras/{camera_id}/rgb?frame_id={identifier}",
                                 "depth_url": f"/api/cameras/{camera_id}/depth?frame_id={identifier}"}
                                for camera_id in ("1", "2", "3")]}

    @app.get("/api/cameras/{camera_id}/{channel}")
    async def camera_image(camera_id: str, channel: str, frame_id: str):
        capture = camera_frames.get(frame_id)
        if capture is None or capture["generation"] != simulation.robot_snapshot()["generation"]:
            raise HTTPException(409, "camera_frame_changed")
        if camera_id not in capture["views"] or channel not in ("rgb", "depth"):
            raise HTTPException(404, "unknown_camera_channel")

        def encode():
            import numpy as np
            from PIL import Image

            view = capture["views"][camera_id]
            if channel == "rgb":
                pixels = view["rgb"]
            else:
                # A display-only 0..2 m grayscale. Control consumes original float depth.
                pixels = np.clip(view["depth_m"] / 2. * 255., 0, 255).astype(np.uint8)
                pixels[~view["valid_depth"]] = 0
            buffer = BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            return buffer.getvalue()

        return Response(await asyncio.to_thread(encode), media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/perception-preview")
    async def perception_preview():
        try:
            preview = await asyncio.to_thread(algorithms.request, "/perception/preview")
        except (RuntimeError, ValueError, OSError) as error:
            raise HTTPException(503, "perception_preview_unavailable") from error
        if preview.get("generation") != simulation.robot_snapshot()["generation"]:
            raise HTTPException(409, "perception_preview_generation_changed")
        return preview

    @app.get("/api/collision-spheres")
    async def collision_spheres():
        return await asyncio.to_thread(simulation.collision_snapshot)

    @app.get("/api/workspaces")
    async def workspaces():
        try:
            return JSONResponse(await asyncio.to_thread(workspace_atlas.preview))
        except (ValueError, OSError) as error:
            raise HTTPException(503, f"工作空间不可用：{error}") from error

    @app.get("/api/workspace-scene")
    async def workspace_scene():
        return {'simulation': simulation.snapshot(), 'server_instance_id': server_instance_id}

    @app.get("/api/health")
    async def health():
        if time.monotonic() - health_cache["at"] > 5:
            try:
                status = await asyncio.to_thread(algorithms.health)
            except Exception as error:  # noqa: BLE001 — service boundary reports unavailable, never success
                status = {"algorithms": {"ready": False, "error": type(error).__name__}}
            if demo_mode:
                status["controller"] = {"ready": True, "name": "responses_program"}
                status["luna"] = {"ready": bool(runner.visual_inventory and runner.visual_inventory.api_key),
                                  "model": "gpt-6-luna", "api_verified": any(
                                      e["kind"] == "visual_decisions_call" and not e["data"].get("error")
                                      for e in runner.events)}
            status["mujoco"] = {"ready": True, "time": simulation.snapshot()["time"]}
            health_cache.update(at=time.monotonic(), value=status)
        return health_cache["value"]

    async def scene_action(function, *args):
        async with operations:
            if runner.active:
                raise HTTPException(409, "run_active_reset_first")
            if runner.holding_state is not None:
                raise HTTPException(409, "holding_state_reset_required")
            try:
                previous_generation = simulation.robot_snapshot()["generation"]
                result = await asyncio.to_thread(function, *args)
                runner.clear()
                if simulation.robot_snapshot()["generation"] != previous_generation:
                    runner.clear_algorithm_state()
                return result
            except (ValueError, RuntimeError) as error:
                raise HTTPException(422, str(error)) from error

    @app.post("/api/scene")
    async def configure(scene: dict):
        return await scene_action(simulation.configure, scene)

    if demo_mode:
        @app.post("/api/demo/preset")
        async def demo_preset(request: DemoPresetRequest):
            from .demo_scene import demo_observation_poses, demo_scene
            return await scene_action(partial(simulation.configure, initial_joints=demo_observation_poses()),
                                      demo_scene(request.object_count))

    @app.post("/api/settle")
    async def settle():
        return await scene_action(simulation.settle)

    @app.post("/api/confirm")
    async def confirm():
        return await scene_action(simulation.confirm)

    @app.post("/api/start")
    async def start(request: StartRequest):
        async with operations:
            if (request.settings_revision is not None
                    and request.settings_revision != runner.planning_settings.revision()):
                raise HTTPException(409, "planning_settings_changed")
            if request.generation is not None and request.generation != simulation.snapshot()["generation"]:
                raise HTTPException(409, "stale_start_after_reset")
            try:
                return await runner.start(request.instruction, experiment=request.experiment)
            except ValueError as error:
                raise HTTPException(409, str(error)) from error

    @app.post("/api/reset")
    async def reset():
        async with operations:
            # Physics invalidation happens first, so an outstanding command cannot overwrite the reset.
            try:
                result = await asyncio.to_thread(simulation.reset)
                await runner.cancel()
                runner.clear()
                runner.clear_algorithm_state()
                return result
            except (ValueError, RuntimeError) as error:
                raise HTTPException(409, str(error)) from error

    @app.websocket("/ws")
    async def updates(socket: WebSocket):
        await socket.accept()
        last_event = 0
        last_run = None
        try:
            while True:
                if last_run != runner.run_id:
                    last_event = 0
                    last_run = runner.run_id
                packet = state(last_event)
                packet["events_mode"] = "append" if last_event else "replace"
                last_event += len(packet["events"])
                await socket.send_json(packet)
                await asyncio.sleep(1 / 15)
        except (WebSocketDisconnect, RuntimeError):
            return

    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    app.mount("/assets", StaticFiles(directory=assets), name="assets")
    web = ROOT / "web" / "dist"
    if web.exists():
        app.mount("/web-assets", StaticFiles(directory=web / "web-assets"), name="web-assets")

    @app.get("/")
    async def index():
        path = web / "index.html"
        if not path.exists():
            raise HTTPException(503, "frontend_not_built: cd web && npm install && npm run build")
        return FileResponse(path)

    @app.get("/collision-spheres.html")
    async def collision_view():
        path = web / "collision-spheres.html"
        if not path.exists():
            raise HTTPException(503, "collision_view_not_built: cd web && pnpm build")
        return FileResponse(path)

    @app.get("/replay.html")
    async def experiment_replay():
        path = web / "replay.html"
        if not path.exists():
            raise HTTPException(503, "experiment_replay_not_built")
        return FileResponse(path)

    @app.get("/workspaces.html")
    async def workspace_view():
        path = web / "workspaces.html"
        if not path.exists():
            raise HTTPException(503, "workspace_view_not_built: cd web && pnpm build")
        return FileResponse(path)

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8340)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
