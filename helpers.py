import os
import pickle
import warnings
from io import BytesIO

import dotenv
import gdown
import keras
import tensorflow as tf

import math
import numpy as np
import pandas as pd
import torch
from PIL import Image
from keras.src.utils import img_to_array

from constants import BASE_DIR, ModelTypes
from ml_data.neuron.neuron import SingleNeuron
from ml_data.test_split import get_shoe_size_test_set, get_shoe_size_gender_test_set, get_diabetes_test_set

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score, \
    accuracy_score, precision_score, recall_score, f1_score

dotenv.load_dotenv()

warnings.simplefilter(action='ignore', category=FutureWarning)

new_neuron = SingleNeuron(input_size=3)
new_neuron.load_weights(BASE_DIR / 'ml_data/neuron/neuron_weights.txt')

diabetes_model = pickle.load(open(BASE_DIR / 'ml_data/diabetes.pickle', 'rb'))
diabetes_tree_model = pickle.load(open(BASE_DIR / 'ml_data/diabetes_decision_tree.pickle', 'rb'))
gender_shoe_model = pickle.load(open(BASE_DIR / 'ml_data/shoe-size-gender.pickle', 'rb'))
shoe_model = pickle.load(open(BASE_DIR / 'ml_data/shoe-size_predict.pickle', 'rb'))

model_class = tf.keras.models.load_model(BASE_DIR / 'ml_data/tensorflow/classification_model.h5')  # noqa
model_reg = tf.keras.models.load_model(BASE_DIR / 'ml_data/tensorflow/regression_model.h5')  # noqa
fashion_mnist = keras.saving.load_model(BASE_DIR / 'ml_data/tensorflow/fashion_mnist.keras')
fashion_cnn = keras.saving.load_model(BASE_DIR / 'ml_data/tensorflow/fashion_cnn.keras')


def get_car_bikes_model():
    if hasattr(get_car_bikes_model, 'model'): return get_car_bikes_model.model

    model_path = BASE_DIR / 'ml_data/tensorflow/car_bikes.keras'

    url = os.getenv('CAR_BIKES_KERAS_URL')
    if not url: raise Exception("[CAR_BIKES_KERAS_URL] No model url provided")

    if not model_path.exists():
        gdown.download(url, model_path.__str__())

    loaded_model = keras.saving.load_model(model_path)
    get_car_bikes_model.model = loaded_model
    return loaded_model


car_bikes = get_car_bikes_model()

# transfer-learning model (mobileNet)
car_bikes_tl = keras.saving.load_model(BASE_DIR / 'ml_data/tensorflow/car_bike_transfer.keras')


def get_yolov5():  # noqa
    if hasattr(get_yolov5, 'model'): return get_yolov5.model

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    get_yolov5.model = model
    return model


yolov5_model = get_yolov5()  # noqa


def get_regression_metrics(model_type: ModelTypes):
    if model_type == ModelTypes.SHOE_MODEL:
        y_test, x_test = get_shoe_size_test_set()
        y_pred = shoe_model.predict(x_test)  # noqa

    else:
        return None

    return [{"title": "MSE", "value": round(mean_squared_error(y_test, y_pred), 4)},
            {"title": "RMSE", "value": round(math.sqrt(mean_absolute_error(y_test, y_pred)), 4)},
            {"title": "MSPE", "value": f"{round(np.mean(np.square((y_test - y_pred) / y_test)) * 100, 2)} %"},
            {"title": "MAE", "value": round(mean_absolute_error(y_test, y_pred), 4)},
            {"title": "MAPE", "value": f"{round(mean_absolute_percentage_error(y_test, y_pred), 2)} %"},
            {"title": "MRE", "value": round(mean_squared_error(y_test, y_pred), 4)},
            {"title": "R-Квадрат", "value": round(r2_score(y_test, y_pred), 4)}]


def get_classification_metrics(model_type: ModelTypes, to_json=False):
    if model_type == ModelTypes.GENDER_SHOE:
        y_test, x_test = get_shoe_size_gender_test_set()
        y_pred = gender_shoe_model.predict(x_test)  # noqa

    elif model_type == ModelTypes.DIABETES:
        y_test, x_test = get_diabetes_test_set()
        y_pred = diabetes_model.predict(x_test)  # noqa

    elif model_type == ModelTypes.DIABETES_TREE:
        y_test, x_test = get_diabetes_test_set()
        y_pred = diabetes_tree_model.predict(x_test)  # noqa

    else:
        return

    frame = {'y_Actual': y_test, 'y_Predicted': y_pred}
    df = pd.DataFrame(frame, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    return [{"title": "Confusion matrix", "value": confusion_matrix.to_dict() if to_json else confusion_matrix},
            {"title": "Accuracy", "value": round(accuracy_score(y_test, y_pred), 4)},
            {"title": "Precision", "value": round(precision_score(y_test, y_pred), 4)},
            {"title": "Recall", "value": round(recall_score(y_test, y_pred), 4)},
            {"title": "F1-мера", "value": round(f1_score(y_test, y_pred), 4)},
            ]


def normalize_image(file, neural_network='mlp'):
    img = Image.open(BytesIO(file)).resize((28, 28)).convert('L')

    x = 255 - img_to_array(img)

    x = x.reshape(1, 784) / 255 \
        if neural_network == 'mlp' \
        else np.expand_dims(x, axis=0)

    return x


def image_from_array(obj) -> bytes:
    img = Image.fromarray(obj)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()
