from fastapi import FastAPI, Request
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from routers import api

app = FastAPI(
    title="ML-Project",
    description="Magankov ML-Project - IVT-301",
    version="1.0"
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.include_router(api.router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )
