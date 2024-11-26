import base64
from typing import Annotated, Literal, Optional, List

from fastapi import FastAPI, Request, File
from fastapi.params import Body, Form
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from routers import api
from routers.api import tensorflow_fashion, tensorflow_cars, yolov5_detecting

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


@app.get("/car-bikes", include_in_schema=False)
async def custom_dataset(request: Request):
    context = {
        'title': 'Car/Bikes',
        **common_context(),
    }

    return templates.TemplateResponse(
        request=request, name="car_bikes.html",
        context=context
    )


@app.post('/car-bikes', include_in_schema=False)
async def custom_image_upload(request: Request,
                              image: Annotated[bytes, File()],
                              category: Annotated[Literal['custom-dataset', 'transfer-learning'], Body()]):
    if not image or not category: return RedirectResponse(url='/', status_code=301)

    context = {
        'title': 'Car/Bikes',
        'image': base64.b64encode(image).decode(),
        'prediction': await tensorflow_cars(image, category),
        **common_context(),
    }

    return templates.TemplateResponse(
        request=request, name="car_bikes.html",
        context=context
    )


@app.get('/yolo', include_in_schema=False)
async def yolo_view(request: Request):
    context = {
        'title': 'Yolo5 Detections',
        'threshold': 0.25,
        **common_context(),
    }

    return templates.TemplateResponse(
        request=request, name="yolo.html",
        context=context
    )


@app.post('/yolo', include_in_schema=False)
async def yolo_image_upload(request: Request, image: Annotated[bytes, File()], confidence_threshold: float = Form(),
                            desired_classes: Optional[str] = Form(None)):
    desired_classes_list: list = desired_classes.lower().strip().split(',') \
        if desired_classes else None

    response = await yolov5_detecting(image, confidence_threshold, desired_classes_list)

    context = {
        'title': 'Yolo5 Detections',
        'image': response.get('image'),
        'cropped_images': response.get('cropped_images'),
        'threshold': confidence_threshold,
        'desired_classes': desired_classes,
        **common_context(),
    }

    return templates.TemplateResponse(
        request=request, name="yolo.html",
        context=context
    )
