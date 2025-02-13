import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Initialize weights in a normal distribution
initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

model = Sequential()
    
model.add(Dense(64, input_dim=5, activation='relu', kernel_initializer=initializer))
model.add(Dense(64, activation='relu', kernel_initializer=initializer))
model.add(Dense(2, activation='relu', kernel_initializer=initializer))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

def flatten_weights(weights):
        flat_weights=tf.concat([tf.reshape(w, [-1]) for w in weights], axis=0)
        return flat_weights.numpy()

def assign_weights(model, flat_weights):
    #flat_weights is np array
    flat_weights = tf.constant(flat_weights)
    reshaped_weights = []
    start = 0
    for var in model.trainable_variables:
        shape = var.shape
        size = tf.size(var).numpy()
        reshaped_weights.append(tf.reshape(flat_weights[start:start + size], shape))
        start += size

    # Assign the new weights to the model
    for var, new_weight in zip(model.trainable_variables, reshaped_weights):
        var.assign(new_weight)


def compare_models(model1, model2):
    weights1 = flatten_weights(model1.trainable_variables)
    weights2 = flatten_weights(model2.trainable_variables)
    print(weights1)
    print(weights2)
    if len(weights1) != len(weights2):
        return False
    j=0
    for w1, w2 in zip(weights1, weights2):
        if w1!=w2:
            j=j+1
    
    return j

model.save("model.h5")
