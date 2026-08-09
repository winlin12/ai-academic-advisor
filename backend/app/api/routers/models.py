"""Model selection: list the local models available and switch which one is running.

See ``services/model_manager.py`` for why switching means stopping and relaunching
llama-server rather than an instant toggle — only one 26-35B-class model fits in this box's
VRAM at a time, so ``POST /{name}/select`` genuinely blocks for the swap (30-90s) before
answering. The frontend is expected to show that as a loading state, not a spinner-free wait.
"""

from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import ModelStatusResponse
from app.services.model_manager import ModelManager, ModelSwitchError, UnknownModelError

router = APIRouter(prefix="/models", tags=["models"])


def _manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


@router.get("", response_model=ModelStatusResponse)
def list_models(request: Request):
    return _manager(request).status()


@router.post("/{name}/select", response_model=ModelStatusResponse)
async def select_model(name: str, request: Request):
    manager = _manager(request)
    try:
        await manager.switch(name)
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelSwitchError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return manager.status()
