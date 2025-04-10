import streamlit as st
import librosa
from preprocessing import *
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from keras.models import load_model

from scipy import signal
from PIL import Image

@st.cache_resource
model = load_model(pcg.keras)

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

# ==== GUI ======
st.set_page_config(page_title="Heart Sound Classifier", layout="centered")
st.title("🫀 Heart Murmur Detection")
st.markdown("Upload a `.wav` file to view its spectrograms and analyse for murmur detection.")

uploaded_file = st.file_uploader("Upload a heart sound `.wav` file", type=["wav"], accept_multiple_files = False)


# ==== View Spectrograms ====
if uploaded_file and st.button("View as Spectrogram"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        wav_path = tmp_file.name

    st.subheader("Mel Spectrogram Segments")
    mel_specs = preprocess_and_segment(wav_path)

    for i, mel in enumerate(mel_specs):
        fig, ax = plt.subplots()
        img = librosa.display.specshow(mel, sr=TARGET_SR, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', ax=ax)
        ax.set_title(f"Segment {i + 1}")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        st.pyplot(fig)

# ==== Analyse ====
if uploaded_file and st.button("Analyse"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        wav_path = tmp_file.name

    mel_specs = preprocess_and_segment(wav_path)

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
    st.write(f"**Confidence:** {confidence:.2f}%"
