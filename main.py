from fastapi import FastAPI, Request
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from routers import api

app = FastAPI(
    title="Machine Learning Api",
    description="Magankov K.S (IVT-301) © 2024",
    contact={
        "name": "Kirill Magankov",
        "url": "https://t.me/zntnaxbi_mk",
    },
    version="1.0"
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.include_router(api.router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )
