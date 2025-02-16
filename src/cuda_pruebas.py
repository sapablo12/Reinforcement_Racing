import tensorflow as tf
from dotenv import load_dotenv
load_dotenv()

print(tf.config.list_physical_devices('GPU'))