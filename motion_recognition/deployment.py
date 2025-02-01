import numpy as np
import tensorflow as tf
from tensorflow import tf

class RealTimeInference:
    def __init__(self, model_path, window_size=120, overlap=0.5):
        """Initialize real-time inference pipeline"""
        self.window_size = window_size
        self.step = int(window_size * (1 - overlap))
        self.buffer = np.zeros((window_size, 9))  # 9-axis sensor data
        self.model = tf.lite.Interpreter(model_path=model_path)
        self.model.allocate_tensors()
        
    def process_frame(self, new_data):
        """Process new sensor data frame"""
        # Update ring buffer
        self.buffer = np.roll(self.buffer, -len(new_data), axis=0)
        self.buffer[-len(new_data):] = new_data
        
        # Get input details
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        
        # Prepare input tensor
        input_data = self.buffer[np.newaxis, ...].astype(np.float32)
        self.model.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        self.model.invoke()
        
        # Get output
        output = self.model.get_tensor(output_details[0]['index'])
        return np.argmax(output[0])
        
def quantize_model(keras_model):
    """Convert and optimize model for deployment"""
    converter = TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    return tflite_model

def save_tflite_model(tflite_model, path):
    """Save quantized model to file"""
    with open(path, 'wb') as f:
        f.write(tflite_model)

class PerformanceMonitor:
    def __init__(self):
        self.latencies = []
        self.power_readings = []
        
    def record_latency(self, latency):
        self.latencies.append(latency)
        
    def record_power(self, power):
        self.power_readings.append(power)
        
    def get_stats(self):
        return {
            'avg_latency': np.mean(self.latencies),
            'max_latency': np.max(self.latencies),
            'avg_power': np.mean(self.power_readings),
            'max_power': np.max(self.power_readings)
        }
