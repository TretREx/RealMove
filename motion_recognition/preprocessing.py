import numpy as np
from scipy.signal import butter, filtfilt

class MotionPreprocessor:
    def __init__(self, sampling_rate=100, cutoff_freq=15):
        """Initialize preprocessing parameters"""
        self.sampling_rate = sampling_rate
        self.cutoff_freq = cutoff_freq
        self.filter_coeff = self._create_butterworth_filter()
        
    def _create_butterworth_filter(self):
        """Create Butterworth filter coefficients"""
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = self.cutoff_freq / nyquist
        b, a = butter(4, normal_cutoff, btype='low', analog=False)
        return b, a
        
    def apply_filter(self, data):
        """Apply low-pass filter to sensor data"""
        return filtfilt(self.filter_coeff[0], self.filter_coeff[1], data, axis=0)
        
    def complementary_filter(self, accel, gyro, dt, alpha=0.98):
        """Fuse accelerometer and gyroscope data"""
        angle = np.zeros_like(gyro)
        for i in range(1, len(gyro)):
            angle[i] = alpha * (angle[i-1] + gyro[i] * dt) + (1 - alpha) * accel[i]
        return angle
        
    def normalize_data(self, data, ranges):
        """Normalize sensor data to [-1, 1] range"""
        return 2 * ((data - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0])) - 1
        
    def segment_windows(self, data, window_size=120, overlap=0.5):
        """Segment data into overlapping windows"""
        step = int(window_size * (1 - overlap))
        windows = []
        for start in range(0, len(data) - window_size + 1, step):
            windows.append(data[start:start + window_size])
        return np.array(windows)
