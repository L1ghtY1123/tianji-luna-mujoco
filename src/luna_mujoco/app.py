"""Luna task console with a single tianji_base robot boundary."""
import argparse
import asyncio
import copy
import json
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from urllib.parse import quote
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
    instruction: str = Field(min_length=1, max_length=4000)
    generation: int | None = Field(default=None, ge=1)
    settings_revision: str | None = None
    experiment: ExperimentOptions = Field(default_factory=ExperimentOptions)


class SettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    settings: PlanningSettings
    expected_revision: str


def create_app(simulation=None, algorithms=None, decisions_factory=None, data_dir=None, *,
               experiment_dir=None, environment_assumptions=None):
    server_instance_id = str(uuid4())
    if decisions_factory is not None:
        raise ValueError("legacy_decisions_factory_not_supported")
    load_dotenv(ROOT / ".env", override=False)
    if simulation is None:
        from tianji_curobo.robot_geometry import robot_geometry_client
        from .base_client import BaseClient
        from .robot_runtime import RobotRuntime
        base_url = os.getenv('LUNA_BASE_URL', 'http://127.0.0.1:8000')
        gateway = BaseClient(base_url, public_url=os.getenv('LUNA_BASE_PUBLIC_URL', base_url))
        simulation = RobotRuntime(gateway, robot_geometry_client())
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
    experiment_root = experiment_dir or os.environ.get("LUNA_EXPERIMENT_DIR") or (
        Path(data_dir) / "experiments" if data_dir is not None else
        ROOT / "data" / "experiments")
    experiments = ExperimentStore(experiment_root)
    from .runtime.controller import PolicyController
    if environment_assumptions is None:
        conditions_path = ROOT / 'resources/workcell_assumptions.json'
        environment_assumptions = json.loads(conditions_path.read_text(encoding='utf-8')) if conditions_path.is_file() else {}
    catalog = getattr(getattr(getattr(algorithms, 'perception', None), 'cad_resolver', None), 'catalog', None)
    known_assets = {code: {'label': row['display_name']} for row in (catalog.records.values() if catalog else ())
                    for code in {row['asset_id'], row.get('source_asset_id')} if code}
    runner = PolicyController(simulation, algorithms, decisions_factory, run_directory, known_assets=known_assets,
                    planning_settings=settings, experiments=experiments,
                    environment_assumptions=environment_assumptions)
    operations = asyncio.Lock()
    from .simulation_scene_client import SimulationSceneClient
    from .scene_editor import SimulationSceneEditor
    from .scene_editor_api import scene_editor_router
    scene_gateway = SimulationSceneClient(simulation.snapshot, scene_url=os.getenv('LUNA_SIMULATION_SCENE_URL'))
    scene_editor = SimulationSceneEditor(scene_gateway)
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
        scene_gateway.close()
        if runner.planning_worker is not None:
            await asyncio.gather(asyncio.shield(runner.planning_worker), return_exceptions=True)
        if runner.algorithm_cleanup_worker is not None:
            await asyncio.shield(runner.algorithm_cleanup_worker)
        if callable(getattr(algorithms, 'close', None)):
            await asyncio.to_thread(algorithms.close)
        await asyncio.to_thread(experiments.close)
        await asyncio.to_thread(simulation.close)

    app = FastAPI(title="Tianji Luna", lifespan=lifespan)
    app.state.runner = runner
    app.state.experiments = experiments
    app.include_router(experiment_router(experiments))
    app.include_router(scene_editor_router(scene_editor, operations=operations, runner=runner))

    @app.get('/api/objects')
    async def object_memory():
        world_key = simulation.world_id
        run_id, database = runner.run_id, runner.object_database
        return {'run_id': run_id, 'objects': (await asyncio.to_thread(database.entities, world_key=world_key)
                if run_id and database is not None and world_key is not None else [])}

    @app.get('/api/objects/{identifier}/history')
    async def object_history(identifier: str, run_id: str | None = None):
        current_run, database = runner.run_id, runner.object_database
        if run_id is not None and run_id != current_run:
            raise HTTPException(409, 'instance_belongs_to_a_different_experiment')
        if not runner.run_id or database is None:
            raise HTTPException(404, 'no_active_experiment_instance')
        try:
            return {**await asyncio.to_thread(database.get, identifier), 'run_id': current_run}
        except ValueError as error:
            raise HTTPException(404, str(error)) from error

    def state(event_start=0):
        run = {**copy.deepcopy(runner.state), "algorithm_pending": runner.algorithm_pending,
               "algorithm_cleanup_error": runner.algorithm_cleanup_error}
        # Failure archives keep complete solver inputs; live JSON uses the same
        # bounded sensor/array summaries as events. Preserve the structured
        # failure_report and policy graph without truncating their checks/nodes.
        for key in ('error_details', 'policy_failure'):
            if key in run:
                run[key] = public_summary(run[key])
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

    @app.get("/api/camera-preview")
    async def camera_preview():
        capture = await asyncio.to_thread(simulation.preview_rgb)
        if capture is None or capture["generation"] != simulation.robot_snapshot(include_tcp=False)["generation"]:
            feedback = simulation.robot_snapshot(include_tcp=False)
            return {'preview_frame_id': None, 'generation': feedback['generation'],
                    'time': feedback['time'], 'cameras': [], 'source': 'awaiting_graph_observation'}
        identifier = capture["preview_frame_id"]
        encoded_identifier = quote(identifier, safe="~!*'()")
        preview_frames[identifier] = capture
        while len(preview_frames) > 2:
            preview_frames.pop(next(iter(preview_frames)))
        labels = {"2": "左腕", "3": "头部", "1": "右腕"}
        return {"preview_frame_id": identifier, "generation": capture["generation"], "time": capture["time"],
                "source": capture.get('source'), "is_task_observation": False, "errors": capture.get('errors', {}),
                "cameras": [{"id": c, "label": labels[c], "frame_id": identifier,
                             "rgb_url": f"/api/camera-preview/{c}/rgb?frame_id={encoded_identifier}"}
                            for c in ("2", "3", "1") if c in capture['views']]}

    @app.get("/api/camera-preview/{camera_id}/rgb")
    async def preview_image(camera_id: str, frame_id: str):
        capture = preview_frames.get(frame_id)
        if capture is None or capture["generation"] != simulation.robot_snapshot(include_tcp=False)["generation"]:
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
            if capture is None or capture["generation"] != simulation.robot_snapshot(include_tcp=False)["generation"]:
                raise HTTPException(503, "camera_observation_unavailable")
            identifier = capture["observation_id"]
            encoded_identifier = quote(identifier, safe="~!*'()")
            camera_frames[identifier] = capture
            while len(camera_frames) > 2:
                camera_frames.pop(next(iter(camera_frames)))
            labels = {"1": "右腕 RGBD", "2": "左腕 RGBD", "3": "头部 RGBD"}
            return {"observation_id": identifier, "generation": capture["generation"], "time": capture["time"],
                    'readiness': capture.get('readiness'), 'synchronization': capture.get('synchronization'),
                    "object_labels": copy.deepcopy(runner.state.get('object_labels'))
                    if runner.state.get('object_labels', {}).get('observation_id') == identifier else None,
                    "cameras": [{"id": camera_id, "label": labels[camera_id], "frame_id": identifier,
                                 "rgb_url": f"/api/cameras/{camera_id}/rgb?frame_id={encoded_identifier}",
                                 "depth_url": f"/api/cameras/{camera_id}/depth?frame_id={encoded_identifier}"}
                                for camera_id in ("1", "2", "3")]}

    @app.get("/api/cameras/{camera_id}/{channel}")
    async def camera_image(camera_id: str, channel: str, frame_id: str):
        capture = camera_frames.get(frame_id)
        if capture is None or capture["generation"] != simulation.robot_snapshot(include_tcp=False)["generation"]:
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
        if preview.get("generation") != simulation.robot_snapshot(include_tcp=False)["generation"]:
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
        return {'simulation': await asyncio.to_thread(simulation.recording_snapshot), 'server_instance_id': server_instance_id}

    @app.get("/api/health")
    async def health():
        if time.monotonic() - health_cache["at"] > 5:
            try:
                status = await asyncio.to_thread(algorithms.health)
            except Exception as error:  # noqa: BLE001 — service boundary reports unavailable, never success
                status = {"algorithms": {"ready": False, "error": type(error).__name__}}
            status["controller"] = {"ready": True, "name": "responses_program"}
            status["luna"] = {"ready": bool(runner.visual_identity and runner.visual_identity.api_key),
                              "model": "gpt-6-luna", "decisions_model": "gpt-6-luna",
                              "responses_model": runner.compiler.model,
                              "responses_reasoning_effort": runner.compiler.reasoning_effort,
                              "api_verified": any(
                                  e["kind"] == "visual_decisions_call" and not e["data"].get("error")
                                  for e in runner.events)}
            status["tianji_base"] = {"ready": simulation.snapshot().get("available", False), "url": simulation.url}
            health_cache.update(at=time.monotonic(), value=status)
        return health_cache["value"]

    @app.post('/api/prepare')
    async def prepare():
        async with operations:
            if runner.active:
                raise HTTPException(409, 'run_active')
            try:
                return await asyncio.to_thread(simulation.prepare)
            except (ValueError, RuntimeError, OSError) as error:
                raise HTTPException(503, str(error)) from error

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

    @app.post('/api/stop')
    async def stop():
        async with operations:
            result = await asyncio.to_thread(simulation.stop_motion)
            await runner.cancel()
            return {**result, 'status': runner.state['status']}

    @app.post("/api/reset")
    async def reset():
        async with operations:
            try:
                before = await asyncio.to_thread(simulation.snapshot)
                result = await asyncio.to_thread(simulation.reset)
                await runner.cancel()
                if result.get('ok') is not True:
                    raise ValueError('task_stop_not_confirmed')
                runner.clear()
                try:
                    if before.get('runtime_mode') == 'simulation':
                        scene = await asyncio.to_thread(scene_gateway.reset, initial_state=before)
                        result.update(scope='simulation_and_task', simulation_reset=scene,
                                      robot=await asyncio.to_thread(simulation.snapshot))
                    return result
                finally:
                    runner.clear_algorithm_state()
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
