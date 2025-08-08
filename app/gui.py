# app.py, front/back end
# Copyright (C) 2025 ibrar-syed <syed.ibraras@gmail.com>
# This file is part of the Skin Cancer Detection Project.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import streamlit as st
import numpy as np
import os
import cv2
from PIL import Image
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from inference.predict_single import predict_single_image
from config import Config

# Set page
st.set_page_config(page_title=" Skin Cancer Classifier", layout="wide")
st.title(" Dermoscopic Image Classification")
st.markdown("---")

# === Sidebar ===
st.sidebar.header("⚙️ Configuration")
model_files = [f for f in os.listdir(Config.MODEL_DIR) if f.endswith('.h5')]

if not model_files:
    st.sidebar.error(" No trained model found in 'saved_models/'. Please train a model first.")
else:
    selected_model = st.sidebar.selectbox("Select a Trained Model", model_files)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader(" Upload a Dermoscopic Image", type=["png", "jpg", "jpeg"])

# === Main Section ===
if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=350)

    # Prediction Trigger
    if st.button("🔎 Run Inference"):
        with st.spinner("Running inference..."):

            # Run prediction
            predictions = predict_single_image(
                image=image,
                model_path=os.path.join(Config.MODEL_DIR, selected_model),
                input_shape=Config.IMAGE_SHAPE,
                label_map=Config.LABEL_MAP,
                top_k=3
            )

            # Display predictions
            st.success(" Prediction complete!")
            top_class = predictions[0]
            st.subheader(f" Top Prediction: `{top_class['label']}`")
            st.markdown(f"**Confidence:** {top_class['confidence']:.2f}")

            # Bar chart of class probabilities
            st.markdown("###  Confidence Scores")
            fig, ax = plt.subplots()
            sns.barplot(
                x=[p['confidence'] for p in predictions],
                y=[p['label'] for p in predictions],
                ax=ax,
                palette='Blues_d'
            )
            ax.set_xlabel("Confidence")
            ax.set_title("Top Class Predictions")
            st.pyplot(fig)

            # Raw table
            st.markdown("### 📊 Prediction Table")
            st.table(predictions)

else:
    st.info(" Please upload an image from the sidebar to begin.")
