from enum import Enum, auto
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

API_URL = '/api/v1'
HOST = 'http://127.0.0.1:8000'

DIABETES_STATUS = ["нет", "есть"]
GENDER_LIST = ["female", "male"]

FASHION_MNIST = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


class ModelTypes(Enum):
    DIABETES = auto()
    DIABETES_TREE = auto()
    GENDER_SHOE = auto()
    SHOE_MODEL = auto()
