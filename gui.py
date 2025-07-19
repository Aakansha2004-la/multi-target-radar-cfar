import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from utils.radar_utils import (
    simulate_targets,
    generate_doppler_signal,
    add_noise,
    cfar_1d,
    generate_chirp_signal
)
from utils.plot_utils import plot_doppler_with_cfar

# Constants
c = 3e8
fc = 77e9
B = 150e6
T_chirp = 1e-6
NUM_CHIRPS = 64
SAMPLES_PER_CHIRP = 128
fs = 1e6
duration = 0.1

st.set_page_config(layout="wide")
st.title(" Multi-Target Radar Simulation Toolkit")

# Sidebar Parameters
with st.sidebar:
    st.header(" Simulation Parameters")
    num_targets = st.slider("Number of Targets", 1, 5, 3)
    snr_db = st.slider("SNR (dB)", 0, 30, 15)
    show_rdm = st.checkbox("Show Range-Doppler Map", value=True)
    apply_cfar = st.checkbox("Apply CFAR on Doppler Spectrum", value=True)
    show_gt = st.checkbox("Overlay Ground Truth on RDM", value=True)

# Ground Truth Simulation
ranges, velocities = simulate_targets(
    num_targets=num_targets,
    max_range=200,
    max_velocity=50,
    duration=duration,
    dt=0.1
)

st.subheader(" Target Ground Truth")
st.write("Ranges (m):", np.round(ranges, 2))
st.write("Velocities (m/s):", np.round(velocities, 2))

# Doppler Signal Processing
signal, _ = generate_doppler_signal(ranges, velocities, fc, fs, duration)
noisy_signal = add_noise(signal, snr_db=snr_db)

fft_spectrum = np.fft.fftshift(np.fft.fft(noisy_signal))
fft_mag = np.abs(fft_spectrum)
fft_db = 20 * np.log10(fft_mag + 1e-10)

# CFAR Detection
if apply_cfar:
    threshold, detection_mask, detection_indices = cfar_1d(fft_mag)
else:
    threshold = np.zeros_like(fft_mag)
    detection_mask = np.zeros_like(fft_mag)
    detection_indices = []

# Plotting Doppler FFT with CFAR
st.subheader(" Doppler FFT with CFAR Overlay")
fig = plot_doppler_with_cfar(fft_mag, threshold, detection_indices, fs, use_db=True)
st.pyplot(fig)

# Debug Info
st.write(" Detection Indices:", detection_indices)
st.write("FFT Peak (dB):", np.max(fft_db))

# Range-Doppler Map
if show_rdm:
    st.subheader(" Range-Doppler Map (RDM)")

    rd_signal = generate_chirp_signal(ranges, velocities, fc, B, T_chirp, NUM_CHIRPS, SAMPLES_PER_CHIRP)
    rd_fft2 = np.fft.fftshift(np.fft.fft2(rd_signal), axes=(0, 1))
    rd_mag = 20 * np.log10(np.abs(rd_fft2) + 1e-6)

    range_res = c / (2 * B)
    vel_res = c / (2 * fc * T_chirp * NUM_CHIRPS)
    range_axis = np.arange(SAMPLES_PER_CHIRP) * range_res
    vel_axis = np.linspace(-NUM_CHIRPS/2, NUM_CHIRPS/2 - 1, NUM_CHIRPS) * vel_res
    extent = [range_axis[0], range_axis[-1], vel_axis[0], vel_axis[-1]]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    im = ax2.imshow(rd_mag, aspect='auto', cmap='viridis', origin='lower', extent=extent)
    ax2.set_title("Range-Doppler Map")
    ax2.set_xlabel("Range (m)")
    ax2.set_ylabel("Velocity (m/s)")

    if show_gt:
        for r, v in zip(ranges, velocities):
            ax2.plot(r, v, 'rx', markersize=8, label='Target' if 'Target' not in ax2.get_legend_handles_labels()[1] else "")
        ax2.legend()

    fig2.colorbar(im, ax=ax2, label="Magnitude (dB)")
    st.pyplot(fig2)















