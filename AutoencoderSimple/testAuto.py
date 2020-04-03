import numpy as np
import tensorflow as tf

encoder = tf.keras.models.load_model(r'./weights/encoder_weights.h5')
decoder = tf.keras.models.load_model(r'./weights/decoder_weights.h5')

inputs = np.array([[2,1,2]])
x = encoder.predict(inputs)
y = decoder.predict(x)

print('Input: {}'.format(inputs))
print('Encoded: {}'.format(x))
print('Decoded: {}'.format(y))