import pandas as pd
import time
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


with open('animal_names.txt') as f:
    animal_names = [line.strip() for line in f]

with open('history.txt', 'r') as f:
    history = json.load(f)


le = preprocessing.LabelEncoder()
le.fit(animal_names)

model = models.load_model('0448P-0422-model')

st.markdown("<h2>Animal Image Classifier | <span style='color:green'>Group 6</span></h2>", unsafe_allow_html=True)
st.markdown("<p><span style='color:red'>Tanner Bibb</span> | <span style='color:red'>Samantha Kuenzi</span> | <span style='color:red'>Tyler Chanes</span></p>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

col1, col2, col3 = st.columns([0.2, 4, 0.2])


# Define a CSS style for the container
container_style = """
    border-radius: 10px;
    padding: 10px;
    height: 300px;
    text-align: center;
"""

# Define a CSS style for the table
table_style = """
    width: 100%;
"""


# Wrap the container with a <div> tag and center it
with st.sidebar:

    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)

    # Apply the style to the container
    st.markdown(
        f"""
        <div style="{container_style}">
            <table style="{table_style}">
            <tr><td style='font-size:25px; color:#999791;'><strong>Animal Name</strong></td></tr>
            {''.join([f"<tr><td style='font-size: 20px;'>{name}</td></tr>" for name in animal_names])}
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Close the <div> tag
    st.markdown("</div>", unsafe_allow_html=True)



    


if uploaded_file is not None:

    with col1:
        st.write("")

    with col2:
        st.image(uploaded_file, use_column_width=True)     
        
        # Process Image and Make Prediction
        experiment_imgs = []
        img = Image.open(uploaded_file).convert("RGB")

        resized_img = img.resize((224,224))
        resized_img = np.array(resized_img) / 255.0

        experiment_imgs.append(resized_img)
        
        experiment_imgs = np.array(experiment_imgs, dtype = 'float32')

        prediction = model.predict(experiment_imgs)
        result = np.argmax(prediction, axis=1)

        # Write Prediction
        st.markdown(f"<h3>Prediction: <span style='color:orange'>{le.inverse_transform([result[0]])[0].capitalize()}</span></h3>", unsafe_allow_html=True)
        # Plot Accuracies
        train_acc = history['accuracy']
        val_acc = history['val_accuracy']
        epochs = range(1, len(val_acc) + 1)
        plt.rcParams.update({'font.size': 30})
        fig, ax = plt.subplots(figsize=(30, 20))
        ax.plot(epochs, train_acc, 'go-', label='Training Accuracy')
        ax.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
        ax.set(title='Training & Validation Accuracy', xlabel='Epochs', ylabel='Accuracy')
        ax.legend()
        st.pyplot(fig)

        st.markdown(
            '<pre style="text-align: center; margin-bottom:20px;">'
            f'Model Input Shape: {model.input_shape}'
            '</pre>',
            unsafe_allow_html=True)

        st.markdown(
            '<pre style="text-align: center; margin-top:20px;">'
            f'{model.summary(print_fn=lambda x: st.text(x))}'
            '</pre>',
            unsafe_allow_html=True) 


        st.image("./streamlit-images/confusion.png", use_column_width=True)

    with col3:
        st.write("")

    






