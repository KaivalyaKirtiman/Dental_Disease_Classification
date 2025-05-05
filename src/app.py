import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from pathlib import Path


MODEL_PATH = Path(__file__).parent.parent / 'models' / 'resnet50v2_oral_disease.keras'

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["Calculus", "Caries", "Gingivitis", 
               "Ulcers", "Tooth-Discoloration", "Hypodontia"]

def preprocess(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224))
    return np.expand_dims(np.array(img)/255.0, axis=0)


st.title("Dental Disease Classifier")
upload = st.file_uploader("Upload tooth image:", type=["jpg","png","jpeg"])

if upload:
    st.image(upload, width=300)
    if st.button('Diagnose'):
        model = load_model()
        if model:
            img = preprocess(upload)
            pred = model.predict(img)[0]
            st.success(f"""Diagnosis: {CLASS_NAMES[np.argmax(pred)]} 
                        ({np.max(pred)*100:.1f}% confidence)""")
            st.bar_chart({k:v for k,v in zip(CLASS_NAMES, pred)})