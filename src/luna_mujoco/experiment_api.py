"""Read archived experiments without invoking perception, decisions or physics."""
import asyncio

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .archive_evidence import read_array_preview


class SaveExperimentRequest(BaseModel):
    name: str | None = Field(default=None, max_length=120)


class ArchiveFileResponse(FileResponse):
    def __init__(self, store, identifier, relative):
        super().__init__(relative)
        self.store = store
        self.identifier = identifier
        self.relative = relative

    async def __call__(self, scope, receive, send):
        # Acquire at send time, and release even when the browser disconnects.
        lease = self.store.read_file(self.identifier, self.relative)
        try:
            self.path = lease.__enter__()
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        except OSError as error:
            raise HTTPException(503, "experiment_storage_unavailable") from error
        try:
            await super().__call__(scope, receive, send)
        finally:
            lease.__exit__(None, None, None)


class ArchiveExportResponse(FileResponse):
    def __init__(self, store, identifier, mode):
        super().__init__("", filename=f"experiment-{identifier}-{mode}.zip")
        self.store = store
        self.identifier = identifier
        self.mode = mode

    @staticmethod
    def remove_completed_export(task):
        # A cancelled request cannot stop a ZIP writer already running in a thread.
        if not task.cancelled() and task.exception() is None:
            task.result().unlink(missing_ok=True)

    async def __call__(self, scope, receive, send):
        lease = self.store.read_archive(self.identifier)
        entered = False
        package = None
        try:
            try:
                lease.__enter__()
                entered = True
                build = asyncio.create_task(asyncio.to_thread(self.store.export, self.identifier, self.mode))
                try:
                    package = await asyncio.shield(build)
                except asyncio.CancelledError:
                    build.add_done_callback(self.remove_completed_export)
                    raise
                self.path = package
            except FileNotFoundError as error:
                raise HTTPException(404, str(error)) from error
            except ValueError as error:
                raise HTTPException(409, str(error)) from error
            except OSError as error:
                raise HTTPException(503, "experiment_storage_unavailable") from error
            await super().__call__(scope, receive, send)
        finally:
            try:
                if package is not None:
                    package.unlink(missing_ok=True)
            finally:
                if entered:
                    lease.__exit__(None, None, None)


def experiment_router(store):
    router = APIRouter(prefix="/api/experiments")

    async def call(function, *args):
        try:
            return await asyncio.to_thread(function, *args)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        except OSError as error:
            raise HTTPException(503, "experiment_storage_unavailable") from error

    @router.get("")
    async def list_experiments():
        return {"root": str(store.root), "experiments": await call(store.list)}

    @router.get("/{identifier}")
    async def get_experiment(identifier: str):
        return await call(store.get, identifier)

    @router.get("/{identifier}/events")
    async def events(identifier: str):
        return {"events": await call(store.events, identifier)}

    @router.get("/{identifier}/arrays")
    async def array_preview(identifier: str, path: str, key: str,
                            max_points: int = Query(default=30000, ge=1, le=30000)):
        return await call(read_array_preview, store, identifier, path, key, max_points)

    @router.get("/{identifier}/files/{relative:path}")
    async def file(identifier: str, relative: str):
        return ArchiveFileResponse(store, identifier, relative)

    @router.post("/{identifier}/save")
    async def save(identifier: str, request: SaveExperimentRequest):
        return await call(store.save, identifier, request.name)

    @router.delete("/{identifier}")
    async def delete(identifier: str):
        return await call(store.delete, identifier)

    @router.get("/{identifier}/export")
    async def export(identifier: str, mode: str = "review"):
        return ArchiveExportResponse(store, identifier, mode)

    return router
