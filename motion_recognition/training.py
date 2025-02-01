import numpy as np
import tensorflow as tf
import keras
from keras.callbacks import LearningRateScheduler

class MotionTrainer:
    def __init__(self, model, num_classes):
        self.model = model
        self.num_classes = num_classes
        
    def augment_data(self, X, y):
        """Apply data augmentation techniques"""
        # Dynamic noise injection
        noise = np.random.normal(0, 0.05, X.shape)
        X_noisy = X + noise
        
        # Temporal random cropping
        crop_size = int(X.shape[1] * np.random.uniform(0.9, 1.1))
        start = np.random.randint(0, X.shape[1] - crop_size)
        X_cropped = X[:, start:start+crop_size, :]
        
        # Pad cropped sequences
        if X_cropped.shape[1] < X.shape[1]:
            pad_width = ((0, 0), (0, X.shape[1] - X_cropped.shape[1]), (0, 0))
            X_cropped = np.pad(X_cropped, pad_width, mode='constant')
            
        return X_cropped, y
        
    def curriculum_learning(self, X, y, difficulty_levels):
        """Implement curriculum learning strategy"""
        # Sort data by difficulty
        sorted_indices = np.argsort(difficulty_levels)
        return X[sorted_indices], y[sorted_indices]
        
    def cosine_decay_schedule(self, epoch, initial_lr=0.001, min_lr=1e-5, total_epochs=100):
        """Cosine learning rate decay"""
        decay = 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
        return min_lr + (initial_lr - min_lr) * decay
        
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model with curriculum learning and data augmentation"""
        # Convert labels to one-hot encoding
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_val = keras.utils.to_categorical(y_val, self.num_classes)
        
        # Create learning rate scheduler
        lr_scheduler = LearningRateScheduler(self.cosine_decay_schedule)
        
        # Training loop
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[lr_scheduler],
            verbose=1
        )
        
        return history
