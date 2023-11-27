from pyexpat import model
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras


# model = keras.models.load_model("your_trained_model.h5")
def preprocess_image(image):
    image = image.resize((100, 50))
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0) 
    return image
st.title("Captcha Solver")
def solve_captcha(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    captcha_text = decode_prediction(prediction)
    return captcha_text
def decode_prediction(prediction):
    captcha_text = ""
    for pred in prediction:
        captcha_text += str(np.argmax(pred))
    return captcha_text
uploaded_image = st.file_uploader("Upload a Captcha Image", type=["jpg", "png", "jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Captcha", use_column_width=True)
    if st.button("Solve Captcha"):
        captcha_text = solve_captcha(image)
        st.success(f"Solved Captcha: {captcha_text}")
if __name__ == '__main__':
    st.write("This is a simplified example. Achieving high accuracy and speed requires advanced model training and optimization.")
