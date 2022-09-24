from sklearn.datasets import load_iris
import joblib
from PIL import Image


model = open('iris_Classifier.pkl', 'rb')
model_ = joblib.load(model)


def params_list():
    return load_iris().feature_names

def images():
    setosa= Image.open('setosa.png')
    versicolor= Image.open('versicolor.png')
    virginica = Image.open('virginica.png')
    list_ = [setosa, versicolor, virginica]
    return list_

