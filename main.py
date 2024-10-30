import base64
from typing import Annotated, Literal

from fastapi import FastAPI, Request, File
from fastapi.params import Body
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from routers import api
from routers.api import tensorflow_fashion

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


def common_context():
    return {
        'has_header': True,
        'has_footer': True,
    }


@app.get("/", include_in_schema=False)
async def index(request: Request):
    context = {
        'title': 'Home',
        **common_context(),
    }
    return templates.TemplateResponse(
        request=request, name="index.html",
        context=context
    )


@app.post('/', include_in_schema=False)
async def index_image_upload(request: Request,
                             image: Annotated[bytes, File()],
                             neural_network: Annotated[Literal['mlp', 'cnn'], Body()], ):

    if not image or not neural_network: return RedirectResponse(url='/', status_code=301)

    context = {
        'title': 'Home',
        'image': base64.b64encode(image).decode(),
        'prediction': await tensorflow_fashion(image, neural_network),
        **common_context(),
    }

    return templates.TemplateResponse(
        request=request, name="index.html",
        context=context
    )
