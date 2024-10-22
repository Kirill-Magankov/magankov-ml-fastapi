import math
from typing import Annotated

import numpy as np
from fastapi import APIRouter, File
from pydantic import BaseModel

from constants import DIABETES_STATUS, GENDER_LIST, FASHION_MNIST
from helpers import model_reg, model_class, gender_shoe_model, shoe_model, diabetes_model, \
    diabetes_tree_model, get_classification_metrics, ModelTypes, get_regression_metrics, fashion_mnist, normalize_image

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
async def tensorflow_fashion(image: Annotated[bytes, File()]):
    x = normalize_image(image)
    prediction = fashion_mnist.predict(x, verbose=False)
    class_index = np.argmax(fashion_mnist.predict(x, verbose=False))

    return {
        "msg": FASHION_MNIST[class_index],
        "probability": str(round(prediction[0][class_index], 3)),
    }


# endregion


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
