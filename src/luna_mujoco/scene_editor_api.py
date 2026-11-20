"""Separate configuration routes, excluded from policy/runtime interfaces."""
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class RevisionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    expected_revision: int = Field(ge=1)


class SceneRequest(RevisionRequest):
    configuration: dict


class MeshRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    asset_id: str = Field(min_length=1, max_length=128)


def scene_editor_router(editor, *, operations, runner):
    router = APIRouter(prefix='/api/simulation-scene')

    async def call(operation, *args, write=True):
        async with operations:
            if write and (runner.active or runner.algorithm_pending):
                raise HTTPException(409, '请先结束当前任务，再编辑仿真场景')
            try:
                return await asyncio.to_thread(operation, *args)
            except (ValueError, KeyError) as error:
                raise HTTPException(400, str(error)) from error
            except Exception as error:
                raise HTTPException(503, 'Simulation 场景接口不可用') from error

    @router.get('')
    async def snapshot():
        scene = await call(editor.gateway.snapshot, write=False)
        if runner.active or runner.algorithm_pending:
            scene.update(editable=False, message='任务正在运行或清理，请在结束后编辑仿真场景')
        return scene

    @router.get('/cad-assets')
    async def assets():
        return {'assets': await call(editor.assets, write=False)}

    @router.post('/cad-assets')
    async def import_mesh(request: MeshRequest):
        return await call(editor.import_mesh, request.asset_id)

    @router.post('')
    async def replace(request: SceneRequest):
        return await call(editor.gateway.replace, request.configuration, request.expected_revision)

    @router.post('/reset')
    async def reset(request: RevisionRequest):
        return await call(editor.gateway.reset, request.expected_revision)

    return router
