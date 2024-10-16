import math

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from helpers import model_reg, model_class, gender_shoe_model, GENDER_LIST, shoe_model, diabetes_model, DIABETES_STATUS, \
    diabetes_tree_model, get_classification_metrics, ModelTypes, get_regression_metrics

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


@router.post("/tensorflow-regression")
async def tensorflow_regression(data: TFRegression):
    input_data = np.array([[
        data.temperature,
        data.temperature,
        data.wind_speed,
    ]])

    pred = model_reg.predict(input_data)[0][0]
    return {
        "msg": f"Потребление энергии: {round(float(pred), 2)} (киловатт-часы)",
        "inputs": data,
    }


@router.post("/tensorflow-classification")
async def tensorflow_classification(data: TFClassification):
    input_data = np.array([[
        data.age - 35,
        data.income - 65_000,
        data.experience,
    ]])

    pred = model_class.predict(input_data)
    result = np.where(pred > 0.5, "Высокий", "Низкий")[0][0]
    return {
        "msg": f"Уровень дохода: {result}",
        "accuracy": str(round(pred[0][0], 6))
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
    X_new = np.array([[
        data.height,
        data.weight,
        data.shoe_size
    ]])

    pred = gender_shoe_model.predict(X_new)
    return {"msg": f"Gender: {GENDER_LIST[pred[0]]}"}


@router.post('/shoe-size')
def shoe_size_prediction(data: ShoeSizeModel):
    X_new = np.array([[
        data.height,
        data.weight,
        data.gender,
    ]])

    pred = shoe_model.predict(X_new)
    return {"msg": f"Shoe size: {math.ceil(pred[0])}"}


@router.post('/diabetes')
def diabetes_predication(data: DiabetesModel):
    X_new = np.array([[
        data.pregnancies,
        data.glucose,
        data.blood_pressures,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.age
    ]])

    pred = diabetes_model.predict(X_new)
    return {"msg": f"Diabetes: {DIABETES_STATUS[pred[0]]}"}


@router.post('/diabetes-tree')
def api_diabetes_tree_get(data: DiabetesModel):
    X_new = np.array([[
        data.pregnancies,
        data.glucose,
        data.blood_pressures,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.age
    ]])

    pred = diabetes_tree_model.predict(X_new)
    return {"msg": f"Diabetes: {DIABETES_STATUS[pred[0]]}"}


@router.get('/metrics')
def api_metrics_get():
    metrics = {
        "diabetes_decision_tree": get_classification_metrics(ModelTypes.DIABETES_TREE, True),
        "diabetes_model": get_classification_metrics(ModelTypes.DIABETES, True),
        "gender_shoe_model": get_classification_metrics(ModelTypes.GENDER_SHOE, True),
        "shoe_model": get_regression_metrics(ModelTypes.SHOE_MODEL)
    }

    return {"msg": metrics}

# endregion
