import tensorflow as tf
from tensorflow.keras import layers

def Discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (4, 4), strides=2, padding='same', input_shape=(256, 256, 3)),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, (4, 4), strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2D(256, (4, 4), strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2D(512, (4, 4), strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model