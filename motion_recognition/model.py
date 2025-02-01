import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.W = Dense(1, activation='tanh')  # Learn attention weights
    
    def call(self, inputs):
        attention_weights = tf.nn.softmax(self.W(inputs), axis=1)
        return tf.reduce_sum(inputs * attention_weights, axis=1)

def create_model(input_shape, num_classes):
    """Create 1D CNN model with attention mechanism"""
    model = Sequential([
        # Feature extraction layers
        Conv1D(64, 5, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        
        # Attention layer
        AttentionLayer(),
        
        # Classification layers
        Flatten(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    # Focal loss implementation
    def focal_loss(gamma=2., alpha=0.25):
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)
            pt = tf.where(tf.keras.backend.equal(y_true, 1), y_pred, 1 - y_pred)
            return -tf.keras.backend.sum(alpha * tf.keras.backend.pow(1. - pt, gamma) * tf.keras.backend.log(pt))
        return focal_loss_fixed
    
    model.compile(optimizer='adam',
                 loss=focal_loss(),
                 metrics=['accuracy'])
    
    return model
