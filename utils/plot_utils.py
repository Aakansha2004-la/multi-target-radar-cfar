import matplotlib.pyplot as plt
import numpy as np
def get_extent_from_params(fc, B, T, SAMPLES_PER_CHIRP, NUM_CHIRPS):
    c = 3e8
    range_res = c / (2 * B)
    velocity_res = c / (2 * fc * T * NUM_CHIRPS)
    range_axis = np.arange(SAMPLES_PER_CHIRP) * range_res
    vel_axis = np.linspace(-NUM_CHIRPS/2, NUM_CHIRPS/2 - 1, NUM_CHIRPS) * velocity_res
    extent = [range_axis[0], range_axis[-1], vel_axis[0], vel_axis[-1]]
    return extent, range_axis, vel_axis


def plot_doppler_with_cfar(fft_mag, threshold, detection_indices, fs, use_db=False):
    N = len(fft_mag)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))

    fft_mag_shifted = np.fft.fftshift(fft_mag)
    threshold_shifted = np.fft.fftshift(threshold)

    if use_db:
        fft_mag_shifted = 20 * np.log10(fft_mag_shifted + 1e-6)
        threshold_shifted = 20 * np.log10(threshold_shifted + 1e-6)

    # Align detection indices to shifted FFT
    detection_indices_shifted = (detection_indices + N//2) % N

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, fft_mag_shifted, 'b-', label="FFT Magnitude")
    ax.plot(freqs, threshold_shifted, 'r--', label="CFAR Threshold")

    if len(detection_indices_shifted) > 0:
        detection_freqs = freqs[detection_indices_shifted]
        detection_amps = fft_mag_shifted[detection_indices_shifted]
        ax.plot(detection_freqs, detection_amps, 'ro', label="Detections")

    ax.set_title("Doppler FFT with CFAR Detection")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude" + (" (dB)" if use_db else ""))
    ax.grid(True)
    ax.legend()
    return fig









