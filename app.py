import pickle
import streamlit as st
import numpy as np

# Load the trained model
with open('knn_model.pkl', 'rb') as file:
    model = pickle.load(file)
# App title
st.title('🌸 Iris Flower Classification')

# Input sliders
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, step=0.1)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, step=0.1)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, step=0.1)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, step=0.1)

# Predict button
if st.button('Predict'):
    # Prepare the input
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make prediction
    prediction = model.predict(data)

    # Show result
    st.success(f'The predicted species is: **{prediction[0]}**')


