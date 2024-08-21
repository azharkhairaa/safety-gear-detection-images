import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# model path
model_path = "improved_model_2.keras"

# load model
model = tf.keras.models.load_model(model_path)

# preprocessing paramter
target_size = (224, 224)
rescale = 1./255

# ImageDataGenerator preprocessing
infer_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=rescale
)

# predict func
def predict_image(img):
    img = img.resize(target_size)
    img_array = np.array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = infer_datagen.standardize(img_array)
    
    # predict
    prediction = model.predict(img_array)
    return prediction

st.title("Safety Gear Detection in Images")
st.write("Hacktiv8 - Phase 2 - Full Time Data Analytics")
st.write("Graded Challenge 7 - RMT 033 - Muhammad Azhar Khaira")
st.markdown('---')

st.sidebar.subheader("Upload Image here")
uploaded_files = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        img = Image.open(uploaded_file)
        
        # predict
        prediction = predict_image(img)
        
        # prediction result
        if prediction[0] > 0.5:
            st.subheader(f"The image {uploaded_file.name} is classified as \n")
            st.subheader(f"Wearing Safety Gear.")
        else:
            st.subheader(f"The image {uploaded_file.name} is classified as \n")
            st.subheader(f"Not Wearing Safety Gear.")
        st.markdown('---')