import base64
import io
import math
from io import BytesIO
from typing import Annotated, Literal, Optional, List

import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, Body, Form
from keras.src.utils import img_to_array
from keras.src.utils.module_utils import tensorflow
from pydantic import BaseModel

from constants import DIABETES_STATUS, GENDER_LIST, FASHION_MNIST, CAR_BIKES
from helpers import model_reg, model_class, gender_shoe_model, shoe_model, diabetes_model, \
    diabetes_tree_model, get_classification_metrics, ModelTypes, get_regression_metrics, fashion_mnist, normalize_image, \
    fashion_cnn, car_bikes, car_bikes_tl, yolov5_model, image_from_array

router = APIRouter()


# region Tensorflow Models

class TFRegression(BaseModel):
    temperature: int
    humidity: int
    wind_speed: int


class TFClassification(BaseModel):
    age: int
    income: int
    experience: int


@router.post("/tensorflow-regression", tags=["tensorflow"])
async def tensorflow_regression(data: TFRegression):
    pred = model_reg.predict(np.array([[
        data.temperature,
        data.temperature,
        data.wind_speed,
    ]]), verbose=False)[0][0]
    return {
        "msg": f"Потребление энергии: {round(float(pred), 2)} (киловатт-часы)",
        "inputs": data,
    }


@router.post("/tensorflow-classification", tags=["tensorflow"])
async def tensorflow_classification(data: TFClassification):
    pred = model_class.predict(np.array([[
        data.age - 35,
        data.income - 65_000,
        data.experience,
    ]]), verbose=False)
    result = np.where(pred > 0.5, "Высокий", "Низкий")[0][0]
    return {
        "msg": f"Уровень дохода: {result}",
        "probability": str(round(pred[0][0], 8))
    }


@router.post("/tensorflow-fashion", tags=["tensorflow"])
async def tensorflow_fashion(image: Annotated[bytes, File()],
                             neural_network: Annotated[Literal['mlp', 'cnn'], Body()]):
    _model = fashion_cnn if neural_network == 'cnn' else fashion_mnist

    x = normalize_image(image, neural_network)
    prediction = _model.predict(x, verbose=False)
    class_index = np.argmax(prediction)

    return {
        "msg": FASHION_MNIST[class_index],
        "network": neural_network,
        "probability": str(round(prediction[0][class_index], 3)),
    }


@router.post("/tensorflow-cars", tags=["tensorflow"])
async def tensorflow_cars(image: Annotated[bytes, File()],
                          category: Annotated[Literal['custom-dataset', 'transfer-learning'], Body()]):
    rp = (360, 360) if category == 'custom-dataset' else (160, 160)  # resize point
    _model = car_bikes if category == 'custom-dataset' else car_bikes_tl

    img = Image.open(BytesIO(image)).resize(rp)

    x = img_to_array(img)
    x = tensorflow.expand_dims(x, 0)
    prediction = _model.predict(x, verbose=False)
    class_index = np.argmax(prediction)

    return {
        "msg": CAR_BIKES[class_index],
        "category": category,
        "probability": str(round(prediction[0][class_index], 3)),
    }


# endregion


class YoloDetectingModel(BaseModel):
    confidence_threshold: float
    desired_classes: Optional[List[str]] = None


@router.post("/yolo5-detecting", tags=["Yolo"])
async def yolov5_detecting(image: Annotated[bytes, File()],
                           confidence_threshold: float = Form(),
                           desired_classes: Optional[List[str]] = Form(None),
                           ):
    try:
        image = Image.open(BytesIO(image))
        results = yolov5_model(image)

        filtered_results = results.pandas().xyxy[0]
        filtered_results = filtered_results[
            ((filtered_results['name'].isin(desired_classes)) if desired_classes else True) & (
                    filtered_results['confidence'] >= confidence_threshold
            )]

        filtered_img = np.array(image)

        for _, row in filtered_results.iterrows():
            label = row['name']
            conf = row['confidence']
            xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
            color = (0, 255, 0)  # Зеленый цвет для рамки
            cv2.rectangle(filtered_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            cv2.putText(filtered_img, f'{label} {conf:.2f}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2)

        img_byte_arr = image_from_array(filtered_img)

        filtered_results = []
        for *box, conf, cls in results.xyxy[0]:
            label = yolov5_model.names[int(cls)]
            if (label in desired_classes if desired_classes else True) and conf >= confidence_threshold:
                filtered_results.append((box, conf, cls))

        # Вырезание и отображение отфильтрованных объектов
        cropped_images = []
        for box, conf, cls in filtered_results:
            xmin, ymin, xmax, ymax = map(int, box)
            cropped_image = filtered_img[ymin:ymax, xmin:xmax]
            cropped_images.append(cropped_image)

        return {
            'status': 'ok',
            'image': base64.b64encode(img_byte_arr).decode(),
            'cropped_images': list(map(lambda x: base64.b64encode(image_from_array(x)).decode(), cropped_images)),
            'error': None
        }

    except Exception as e:
        return {
            'status': 'error',
            'image': None,
            'error': {
                'message': str(e),
                'type': type(e).__name__
            }
        }


# region Basic Models

class KNNModel(BaseModel):
    height: float
    weight: float
    shoe_size: float


class ShoeSizeModel(BaseModel):
    height: float
    weight: float
    gender: int


class DiabetesModel(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressures: float
    skin_thickness: float
    insulin: float
    bmi: float
    age: int


@router.post('/knn')
def knn_model_prediction(data: KNNModel):
    pred = gender_shoe_model.predict(np.array([[
        data.height,
        data.weight,
        data.shoe_size
    ]]))
    return {"msg": f"Gender: {GENDER_LIST[pred[0]]}"}


@router.post('/shoe-size')
def shoe_size_prediction(data: ShoeSizeModel):
    pred = shoe_model.predict(np.array([[
        data.height,
        data.weight,
        data.gender,
    ]]))
    return {"msg": f"Shoe size: {math.ceil(pred[0])}"}


@router.post('/diabetes')
def diabetes_predication(data: DiabetesModel):
    pred = diabetes_model.predict(np.array([[
        data.pregnancies,
        data.glucose,
        data.blood_pressures,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.age
    ]]))
    return {"msg": f"Diabetes: {DIABETES_STATUS[pred[0]]}"}


@router.post('/diabetes-tree')
def diabetes_tree_prediction(data: DiabetesModel):
    pred = diabetes_tree_model.predict(np.array([[
        data.pregnancies,
        data.glucose,
        data.blood_pressures,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.age
    ]]))
    return {"msg": f"Diabetes: {DIABETES_STATUS[pred[0]]}"}


@router.get('/metrics')
def model_metrics():
    metrics = {
        "diabetes_decision_tree": get_classification_metrics(ModelTypes.DIABETES_TREE, True),
        "diabetes_model": get_classification_metrics(ModelTypes.DIABETES, True),
        "gender_shoe_model": get_classification_metrics(ModelTypes.GENDER_SHOE, True),
        "shoe_model": get_regression_metrics(ModelTypes.SHOE_MODEL)
    }

    return {"msg": metrics}

# endregion
