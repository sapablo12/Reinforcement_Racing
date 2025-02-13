import tensorflow as tf
from tensorflow.keras import layers

def create_critic_network(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Hidden layers with ReLU activation
    out = layers.Dense(256, activation='relu')(inputs)
    out = layers.BatchNormalization()(out)  # Optional: Stabilize training
    out = layers.Dense(128, activation='relu')(out)
    out = layers.Dense(64, activation='relu')(out)  # Added a third hidden layer
    
    # Output layer with linear activation
    out = layers.Dense(1, activation='linear')(out)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=out)
    return model

# Example usage:
input_shape = (5,)  # Example input shape with 5 inputs
critic_network = create_critic_network(input_shape)
critic_network.summary()
