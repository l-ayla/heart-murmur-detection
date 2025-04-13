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


# ==== contsants ======
TARGET_SR = 2000
BANDPASS_LOW = 20
BANDPASS_HIGH = 900
N_MELS = 128
SEGMENT_DURATION = 3
NFFT = 256
WIN_LENGTH = 256
HOP_LENGTH = 47

LABEL_MAP = {
    "Absent": "Murmur Absent",
    "Present": "Murmur Detected",
    "Unknown": "Unknown"
}

# @st.cache_resource
# model = load_model(pcg.keras)
CLASS_LABELS = ["Murmur Detected", "Murmur Absent", "Unknown"]

# ==== GUI ======
st.set_page_config(page_title="Heart Sound Classifier", layout="centered")
st.title("🫀 Heart Murmur Detection")

@st.cache_resource
def load_cnn_model():
    return load_model('pcg_whar.h5') # i dont know why i have to declare this as a seperate function but it just works

model = load_cnn_model()

uploaded_file = st.file_uploader("Upload one or more heart sound recordings in `.wav` format from the same patient (minimum 9 seconds)", type=["wav"], accept_multiple_files = True)
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
    if uploaded_files:
        all_predictions = []

        for uploaded_file in uploaded_files:
            wav_bytes = uploaded_file.read()
            wav_buffer = BytesIO(wav_bytes)
            wav_buffer.seek(0)

            input_data = np.stack(mel_specs).astype(np.float32)
            input_data = input_data[..., np.newaxis]  # [segments, height, width, 1]

            predictions = model.predict(input_data)  # shape: [num_segments, 3]
            all_predictions.append(np.mean(predictions, axis=0))  # average for this file

        # Now average across all uploaded files
        avg_probs = np.mean(all_predictions, axis=0)
        predicted_class = np.argmax(avg_probs)

        class_labels = ["Murmur Absent", "Murmur Detected", "Unknown"]
        if avg_probs[predicted_class] < 0.5 and predicted_class in [0, 1]:
            predicted_class = 2

        confidence = avg_probs[predicted_class] * 100

        st.subheader("🩺 Analysis Result")
        st.write(f"**Prediction:** {class_labels[predicted_class]}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.warning("⚠️ This tool is for educational and research purposes only. It does **not** provide medical advice or diagnosis. If you have concerns about your heart health, please consult a qualified medical professional.")
    else:
        st.warning("Please upload one or more files.")