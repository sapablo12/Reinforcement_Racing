import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()

print("Detected GPUs:", tf.config.list_physical_devices("GPU"))

size = 100
exploration = 0.7
iterations = 100

exp_values = []
for i in range(iterations):
    exp = exploration - (exploration - 0.05) * (i / size)
    exp_values.append(exp)

plt.plot(range(iterations), exp_values)
plt.xlabel("Iteration")
plt.ylabel("Exploration")
plt.title("Exploration decay")
plt.show()

print("Last exploration value:", exp_values[-1])
