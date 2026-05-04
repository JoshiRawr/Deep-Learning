
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

model = tf.keras.models.load_model("waste_model.keras", compile=False)

with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

st.title("Waste Segregation Classifier")
st.write("Upload a waste image to classify it.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.subheader("Prediction")
    st.write(f"**Class:** {class_names[pred_class]}")
    st.write(f"**Confidence:** {confidence:.2f}%")
