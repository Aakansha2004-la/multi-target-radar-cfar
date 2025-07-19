import numpy as np

def simulate_targets(num_targets, max_range, max_velocity, duration, dt):
    """
    Simulates random target ranges and velocities.
    """
    ranges = np.random.uniform(0, max_range, num_targets)
    velocities = np.random.uniform(-max_velocity, max_velocity, num_targets)
    return ranges, velocities

def generate_doppler_signal(ranges, velocities, fc, fs, duration):
    """
    Generates a Doppler signal for given targets.
    """
    t = np.arange(0, duration, 1/fs)
    signal = np.zeros_like(t, dtype=complex)
    lambda_radar = 3e8 / fc
    for r, v in zip(ranges, velocities):
        doppler_shift = 2 * v / lambda_radar
        phase_shift = 2 * np.pi * doppler_shift * t
        signal += np.exp(1j * phase_shift)
    return signal, t

def add_noise(signal, snr_db):
    """
    Adds complex Gaussian noise to a signal at the given SNR (in dB).
    """
    signal = np.asarray(signal)
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )
    return signal + noise

def cfar_1d(
    magnitude_signal,
    num_train=20,
    num_guard=4,
    rate_fa=1e-3,
    alpha_scale=1.5,
    min_signal_threshold=1e-3,
    suppress_isolated=True,
    isolation_window=1
):
    """
    Improved 1D CFAR detection with:
    - Adjustable scaling factor for threshold (alpha_scale)
    - Suppression of weak signals (min_signal_threshold)
    - Option to remove isolated detections (suppress_isolated)

    Returns:
        threshold: threshold values at each point
        detection_mask: binary mask where detections occurred
        detection_indices: array of detection indices
    """
    n = len(magnitude_signal)
    threshold = np.zeros(n)
    detection_mask = np.zeros(n)

    alpha = alpha_scale * num_train * (rate_fa ** (-1 / num_train) - 1)

    for i in range(num_train + num_guard, n - num_train - num_guard):
        training_cells = np.concatenate((
            magnitude_signal[i - num_guard - num_train:i - num_guard],
            magnitude_signal[i + num_guard + 1:i + num_guard + 1 + num_train]
        ))
        noise_level = np.mean(training_cells)
        threshold[i] = alpha * noise_level

        if (
            magnitude_signal[i] > threshold[i]
            and magnitude_signal[i] > min_signal_threshold
        ):
            detection_mask[i] = 1

    if suppress_isolated:
        # Suppress isolated peaks (non-clustered detections)
        for i in range(isolation_window, n - isolation_window):
            if detection_mask[i]:
                local_window = detection_mask[i - isolation_window:i + isolation_window + 1]
                if np.sum(local_window) <= 1:
                    detection_mask[i] = 0  # Suppress isolated point

    detection_indices = np.where(detection_mask == 1)[0]
    return threshold, detection_mask, detection_indices

def generate_chirp_signal(ranges, velocities, fc, B, T_chirp, num_chirps, samples_per_chirp):
    """
    Generates an FMCW chirp signal for a given set of targets.
    """
    c = 3e8
    t = np.linspace(0, T_chirp, samples_per_chirp)
    signal = np.zeros((num_chirps, samples_per_chirp), dtype=complex)

    for i in range(num_chirps):
        chirp = np.zeros(samples_per_chirp, dtype=complex)
        for r, v in zip(ranges, velocities):
            tau = 2 * (r + v * i * T_chirp) / c
            phase = 2 * np.pi * (fc * tau + 0.5 * B / T_chirp * tau**2)
            chirp += np.exp(1j * (2 * np.pi * fc * t - phase))
        signal[i, :] = chirp
    return signal
















