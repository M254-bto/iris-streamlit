import streamlit as st
import pandas as pd
import joblib
from PIL import Image

with open('iris_Classifier.pkl','rb') as f:
    model_ = joblib.load(f)


def params_list():
    data = pd.read_csv('Iris.csv')
    data = data.drop(data[['Id', 'Species']], axis=1)
    return data.columns


def images():
    setosa= Image.open('setosa.png')
    versicolor= Image.open('versicolor.png')
    virginica = Image.open('virginica.png')
    list_ = [setosa, versicolor, virginica]
    return list_





st.title('Iris classification site')

parameter_list = params_list()
input_params = []
input_params_default = ['5.2','3.2','4.2','1.2']

values = []




for parameter, parameter_df in zip(parameter_list, input_params_default):
 
    values= st.slider(label=parameter, key=parameter,value=float(parameter_df), min_value=0.0, max_value=8.0, step=0.1)
    input_params.append(values)
 
input_params=pd.DataFrame([input_params],columns=parameter_list,dtype=float)
st.write('\n\n')



if st.button('Classify'):
    pred = model_.predict(input_params)
    for i in range(3):
        if pred == i:
            st.image(images()[i])
