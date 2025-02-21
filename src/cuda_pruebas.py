import tensorflow as tf
from dotenv import load_dotenv
import numpy as np
load_dotenv()

print(tf.config.list_physical_devices('GPU'))

import matplotlib.pyplot as plt

size = 100  # Define size
exploration = 0.7  # Define exploration
N = 100  # Number of iterations

k = -1 * (1 / size) * np.log(0.05 / exploration)
exp_values = []

for i in range(N):
    exp = exploration - (exploration - 0.05) * (i / size)
    exp_values.append(exp)

# Plotting exp over N
plt.plot(range(N), exp_values)
plt.xlabel('N')
plt.ylabel('exp')
plt.title('exp over N')
plt.show()
print(exp)
a=[1,4,5,67,7,8,9]
print