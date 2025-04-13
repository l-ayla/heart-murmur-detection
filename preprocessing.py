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


# ==== contsants ======
TARGET_SR = 2000
TARGET_SR = 2000
BANDPASS_LOW = 20
BANDPASS_HIGH = 900
N_MELS = 128
SEGMENT_DURATION = 3
NFFT = 256
WIN_LENGTH = 256
HOP_LENGTH = 47

LABEL_MAP = {
    "Present": "Murmur Detected",
    "Absent": "Murmur Absent",
    "Unknown": "Unknown"
}



#============functions======================

def load_and_downsample(file_path, target_sr):
    audio, original_sr = librosa.load(file_path, sr=None)
    return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr), target_sr

def z_score_normalize(audio):
    mean, std = np.mean(audio), np.std(audio)
    return (audio - mean) / std if std != 0 else audio

def bandpass_filter(audio, sr, lowcut, highcut):
    nyquist = 0.5 * sr
    b, a = signal.butter(6, [lowcut/nyquist, highcut/nyquist], btype="band")
    return signal.filtfilt(b, a, audio)

def normalise_segment(segment):
    min_val, max_val = np.min(segment), np.max(segment)
    return ((segment - min_val) / (max_val - min_val) * 2 - 1) if (max_val - min_val) != 0 else segment

def segment_audio(audio, sr, segment_duration):
    segment_samples = int(sr * segment_duration)
    return [audio[i:i + segment_samples] for i in range(0, len(audio), segment_samples)
            if len(audio[i:i + segment_samples]) == segment_samples]

def generate_mel_spectrogram(audio, sr, n_mels, hop_length, n_fft, win_length):
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels,
                                     hop_length=hop_length, n_fft=n_fft,
                                     win_length=win_length)
    return librosa.power_to_db(S, ref=np.max)

def view_mel_spectrogram(mel_spectrogram, sr, output_path):
    # Create 128x128 image
    plt.figure(figsize=(1.28, 1.28), dpi=100)
    ax = plt.gca()

    # Show spectrogram without axes
    librosa.display.specshow(mel_spectrogram,
                           sr=sr,
                           x_axis=None,
                           y_axis=None,
                           cmap="magma",
                           ax=ax)

    # Remove all margins and axes
    plt.axis('off')
    plt.tight_layout(pad=0)

    # Save without any extra space
    plt.savefig(output_path,
               bbox_inches='tight',
               pad_inches=0)
    plt.close()

def save_mel_spectrogram_npy(mel_spectrogram, output_path):
    np.save(output_path, mel_spectrogram)


def preprocess_and_segment(wav_path):
    audio, sr = load_and_downsample(wav_path, TARGET_SR)
    audio = z_score_normalize(audio)
    audio = bandpass_filter(audio, sr, BANDPASS_LOW, BANDPASS_HIGH)
    segments = segment_audio(audio, sr, SEGMENT_DURATION)

    mel_specs = []
    for segment in segments:
        segment = normalise_segment(segment)
        mel = generate_mel_spectrogram(segment, sr, N_MELS, HOP_LENGTH, NFFT, WIN_LENGTH)
        mel_specs.append(mel)

    return segments, mel_specs

def plot_signal_and_spectrogram(audio, mel_spectrogram, sr, hop_length, n_mels):
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [1, 3]})

    # Plot raw signal
    time = np.linspace(0, len(audio) / sr, len(audio))
    ax[0].plot(time, audio, color='black')
    ax[0].set_title("Raw Signal")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlim([0, len(audio) / sr])

    # Plot mel-spectrogram
    librosa.display.specshow(mel_spectrogram, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel", cmap="magma", ax=ax[1])
    ax[1].set_title("Mel-Spectrogram")
    ax[1].set_ylabel("Mel bin")
    ax[1].set_yticks([])

    plt.tight_layout()
    return fig