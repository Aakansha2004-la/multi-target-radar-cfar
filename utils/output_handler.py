# utils/output_handler.py

import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure output directory exists
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_signal_plot(time, signal, title="Signal", filename="signal_plot.png"):
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"[Saved] Signal plot -> {filepath}")

def save_cfar_plot(time, signal, detections, title="CFAR Detection", filename="cfar_detection.png"):
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal, label='Signal')
    if len(detections) > 0:
        plt.plot(time[detections], signal[detections], 'ro', label='Detections')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"[Saved] CFAR plot -> {filepath}")

def save_range_doppler_plot(rd_map, detections=None, true_targets=None, vmax=None, extent=None, filename="range_doppler.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    rd_map_db = 20 * np.log10(np.abs(rd_map) + 1e-12)
    rd_map_db = np.fft.fftshift(rd_map_db, axes=0)

    cax = ax.imshow(
        rd_map_db,
        aspect='auto',
        cmap='viridis',
        origin='lower',
        extent=extent,
        vmax=vmax
    )

    ax.set_title("Range-Doppler Map")
    ax.set_xlabel("Velocity (m/s)" if extent else "Doppler Bins")
    ax.set_ylabel("Range (m)" if extent else "Range Bins")

    if detections is not None and len(detections) > 0:
        for (doppler_idx, range_idx) in detections:
            if extent:
                vmin, vmax_, rmin, rmax = extent
                vel_res = (vmax_ - vmin) / rd_map.shape[1]
                rng_res = (rmax - rmin) / rd_map.shape[0]
                x = vmin + range_idx * vel_res
                y = rmin + doppler_idx * rng_res
            else:
                x, y = range_idx, doppler_idx

            ax.plot(x, y, 'ro', label='CFAR Detection' if 'CFAR Detection' not in ax.get_legend_handles_labels()[1] else "")

    if true_targets is not None and len(true_targets) > 0:
        for target in true_targets:
            ax.plot(target['range_idx'], target['doppler_idx'], 'yx', label='Ground Truth' if 'Ground Truth' not in ax.get_legend_handles_labels()[1] else "")

    if ax.get_legend_handles_labels()[1]:
        ax.legend()

    fig.colorbar(cax, ax=ax, label="Magnitude (dB)")
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"[Saved] Range-Doppler map -> {filepath}")

def save_tracking_visual(rd_map, tracks, vmax=None, extent=None, filename="rd_tracking.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    rd_map_db = 20 * np.log10(np.abs(rd_map) + 1e-12)
    rd_map_db = np.fft.fftshift(rd_map_db, axes=0)

    cax = ax.imshow(
        rd_map_db,
        aspect='auto',
        cmap='viridis',
        origin='lower',
        extent=extent,
        vmax=vmax
    )
    ax.set_title("Range-Doppler with Kalman Tracks")
    ax.set_xlabel("Velocity (m/s)" if extent else "Doppler Bins")
    ax.set_ylabel("Range (m)" if extent else "Range Bins")

    for i, track in enumerate(tracks):
        if extent:
            vmin, vmax_, rmin, rmax = extent
            vel_res = (vmax_ - vmin) / rd_map.shape[1]
            rng_res = (rmax - rmin) / rd_map.shape[0]

            vel_coord = vmin + track['vel'] * vel_res
            rng_coord = rmin + track['rng'] * rng_res
        else:
            vel_coord = track['vel']
            rng_coord = track['rng']

        ax.plot(vel_coord, rng_coord, 'rx')
        ax.text(vel_coord, rng_coord, f"ID {track['id']}", color='white', fontsize=8)

    if tracks:
        ax.legend(["Kalman Track"])

    fig.colorbar(cax, ax=ax, label="Magnitude (dB)")
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"[Saved] RD map with tracking -> {filepath}")




