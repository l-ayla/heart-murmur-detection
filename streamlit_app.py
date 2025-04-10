import streamlit as st
import librosa
from preprocessing import *
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from keras.models import load_model
from io import BytesIO

from scipy import signal
from PIL import Image

# @st.cache_resource
# model = load_model(pcg.keras)

# ==== contsants ======
TARGET_SR = 2000
SEGMENT_DURATION = 3  # in seconds
BANDPASS_LOW = 25
BANDPASS_HIGH = 500
N_MELS = 64
HOP_LENGTH = 128
NFFT = 512
WIN_LENGTH = 400

LABEL_MAP = {
    "Present": "Murmur Detected",
    "Absent": "Murmur Absent",
    "Unknown": "Unknown"
}

# @st.cache_resource
# model = load_model(pcg.keras)
CLASS_LABELS = ["Murmur Detected", "Murmur Absent", "Unknown"]

# ==== GUI ======
st.set_page_config(page_title="Heart Sound Classifier", layout="centered")
st.title("🫀 Heart Murmur Detection")

uploaded_file = st.file_uploader("Upload a heart sound `.wav` file (minimum 9 seconds)", type=["wav"], accept_multiple_files = False)
col1, col2 = st.columns(2)
view_clicked = col1.button("View as Spectrogram", key="view_button")
analyse_clicked = col2.button("Analyse", key="analyse_button")

# ==== save and preprocess uploaded file ===
if uploaded_file:
    # Save uploaded file temporarily
    wav_bytes = uploaded_file.read()
    wav_buffer = BytesIO(wav_bytes)
    wav_buffer.seek(0)

    # Process file
    [segments, mel_specs] = preprocess_and_segment(wav_buffer)


# ==== View Spectrograms ====
if view_clicked:
    if uploaded_file:
        for i in range(len(segments)):
            st.markdown(f"**Segment {i + 1}**")
            fig = plot_signal_and_spectrogram(
                audio=segments[i],
                mel_spectrogram=mel_specs[i],
                sr=TARGET_SR,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS
            )
            st.pyplot(fig)
    else:
        st.warning("Please upload a file first.")

# ==== Analyse ==========
if analyse_clicked:
    if uploaded_file:
        # Prepare input: [num_segments, height, width, 1]
        input_data = np.stack(mel_specs)
        input_data = input_data[..., np.newaxis]  # add channel dimension

        predictions = model.predict(input_data)
        avg_probs = np.mean(predictions, axis=0)
        predicted_class = np.argmax(avg_probs)
        class_labels = ["Murmur Detected", "Murmur Absent", "Unknown"]
        confidence = avg_probs[predicted_class] * 100

        st.subheader("🩺 Analysis Result")
        st.write(f"**Prediction:** {class_labels[predicted_class]}")
        st.write(f"**Confidence:** {confidence:.2f}%")
    else: st.warning("Please upload a file first")