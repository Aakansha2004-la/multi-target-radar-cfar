import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.radar_utils import simulate_targets, generate_chirp_signal, cfar_1d
from utils.plot_utils import get_extent_from_params

# --- Simulation Parameters ---
NUM_FRAMES = 30
NUM_TARGETS = 3
SAMPLES_PER_CHIRP = 128
NUM_CHIRPS = 64
dt = 0.1  # time step between frames

# --- Radar Configuration ---
fc = 77e9       # Carrier frequency in Hz
B = 150e6       # Bandwidth in Hz
T = 1e-6        # Chirp duration in sec
c = 3e8         # Speed of light

# --- Axes Setup ---
range_res = c / (2 * B)
velocity_res = c / (2 * fc * T * NUM_CHIRPS)
range_axis = np.arange(SAMPLES_PER_CHIRP) * range_res
vel_axis = np.linspace(-NUM_CHIRPS/2, NUM_CHIRPS/2 - 1, NUM_CHIRPS) * velocity_res
extent = [range_axis[0], range_axis[-1], vel_axis[0], vel_axis[-1]]

# --- Target Initialization ---
ranges, velocities = simulate_targets(NUM_TARGETS)

# --- Plot Setup ---
fig, ax = plt.subplots(figsize=(8, 6))
initial_image = np.zeros((NUM_CHIRPS, SAMPLES_PER_CHIRP))
img = ax.imshow(initial_image, aspect='auto', cmap='viridis', origin='lower',
                extent=extent, vmin=-80, vmax=0)
scat = ax.scatter([], [], c='red', marker='o', label='Detections')  # CFAR detections
gt_scat = ax.scatter([], [], c='cyan', marker='x', label='True Targets')  # Ground truth

ax.set_title("Live Range-Doppler Map with CFAR Detections")
ax.set_xlabel("Range (m)")
ax.set_ylabel("Velocity (m/s)")
ax.legend(loc='upper right')

# --- Update Function ---
def update(frame_idx):
    global ranges

    # Move targets
    ranges += velocities * dt

    # Generate chirp signal
    raw_signal = generate_chirp_signal(ranges, velocities, fc, B, T, NUM_CHIRPS, SAMPLES_PER_CHIRP)

    # Compute Range-Doppler Map
    rd_map = np.fft.fftshift(np.fft.fft2(raw_signal), axes=(0, 1))
    rd_mag = np.abs(rd_map)
    rd_db = 20 * np.log10(rd_mag + 1e-12)

    # Update heatmap
    img.set_array(rd_db)
    img.set_clim(vmin=np.min(rd_db), vmax=np.max(rd_db))

    # CFAR Detection across Doppler bins
    detections_x = []
    detections_y = []
    for doppler_idx in range(NUM_CHIRPS):
        row_mag = rd_mag[doppler_idx, :]
        _, mask, _ = cfar_1d(row_mag)
        detected_indices = np.where(mask == 1)[0]
        for idx in detected_indices:
            detections_x.append(range_axis[idx])
            detections_y.append(vel_axis[doppler_idx])

    # Update detections
    scat.set_offsets(np.column_stack((detections_x, detections_y)))

    # Update true targets
    gt_scat.set_offsets(np.column_stack((ranges, velocities)))

    return img, scat, gt_scat

# --- Run Animation ---
ani = animation.FuncAnimation(fig, update, frames=NUM_FRAMES, interval=500, blit=False, repeat=False)
plt.tight_layout()
plt.show()








