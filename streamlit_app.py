import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from io import BytesIO
from preprocessing import *  # Assumes all preprocessing functions are in here

# ==== Constants ======
TARGET_SR = 2000
BANDPASS_LOW = 20
BANDPASS_HIGH = 900
N_MELS = 128
SEGMENT_DURATION = 3
NFFT = 256
WIN_LENGTH = 256
HOP_LENGTH = 47
CLASS_LABELS = ["Murmur Absent", "Murmur Detected", "Unknown"]
THRESHOLD = 0.5

# ==== Streamlit GUI Setup =====
st.set_page_config(page_title="Heart Sound Classifier", layout="centered")
st.title("🫀 Heart Murmur Detection")
st.markdown("<small><i>This app is not intended to provide medical advice. Please consult a doctor if you are concerned about your heart health.</i></small>", unsafe_allow_html=True)

@st.cache_resource
def load_cnn_model():
    return load_model('pcg_whar.h5')

model = load_cnn_model()

uploaded_files = st.file_uploader("Upload one or more heart sound `.wav` files (each >= 9 seconds)", type=["wav"], accept_multiple_files=True)

col1, col2 = st.columns(2)
view_clicked = col1.button("View as Spectrogram")
analyse_clicked = col2.button("Analyse")

all_segments = []
all_predictions = []

if uploaded_files:
    for file_index, file in enumerate(uploaded_files):
        st.markdown(f"### 🔊 File {file_index + 1}: `{file.name}`")
        wav_bytes = file.read()
        wav_buffer = BytesIO(wav_bytes)
        wav_buffer.seek(0)

        segments, mel_specs = preprocess_and_segment(wav_buffer)
        all_segments.append((file.name, segments, mel_specs))

        input_data = np.stack(mel_specs).astype(np.float32)[..., np.newaxis]
        preds = model.predict(input_data)
        all_predictions.append(preds)

    # === View Spectrograms ===
    if view_clicked:
        for file_name, segments, mel_specs in all_segments:
            st.markdown(f"#### 📊 Spectrograms for `{file_name}`")
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

    # === Analyse ===
    if analyse_clicked:
        all_preds_concat = np.concatenate(all_predictions, axis=0)
        avg_probs = np.mean(all_preds_concat, axis=0)
        predicted_class = np.argmax(avg_probs)

        if avg_probs[predicted_class] < THRESHOLD and predicted_class in [0, 1]:
            predicted_class = 2

        confidence = avg_probs[predicted_class] * 100

        st.subheader("🩺 Combined Patient Analysis")
        st.write(f"**Prediction:** {CLASS_LABELS[predicted_class]}")
        st.write(f"**Confidence:** {confidence:.2f}%")