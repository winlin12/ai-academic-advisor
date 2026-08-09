"""Route aggregation (TODO Priority 5).

The old flat, untagged router is gone. Each API domain now lives in its own tagged module
under ``routers/`` and is composed here into two mount points:

* ``api_v1_router`` — every product route, under a single ``/v1`` prefix. New breaking
  changes get a ``/v2`` sibling instead of mutating these paths.
* ``system_router`` — ``/health*`` probes, exported separately so ``main.py`` can mount
  them unversioned (infra checks shouldn't chase API versions).
"""

from fastapi import APIRouter

from app.api.routers import academic, admin, advisor, models, planning, system

api_v1_router = APIRouter(prefix="/v1")
api_v1_router.include_router(academic.router)
api_v1_router.include_router(planning.router)
api_v1_router.include_router(advisor.router)
api_v1_router.include_router(models.router)
api_v1_router.include_router(admin.router)

system_router = system.router
