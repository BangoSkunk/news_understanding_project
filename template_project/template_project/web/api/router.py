from fastapi.routing import APIRouter

from template_project.web.api import monitoring
from template_project.web.api import endpoints

api_router = APIRouter()
api_router.include_router(monitoring.router)
api_router.include_router(endpoints.router)
