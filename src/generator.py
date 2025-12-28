import tensorflow as tf
from tensorflow.keras import layers, models

def build_generator(latent_dim):
    model = models.Sequential()
    
    model.add(layers.Dense(128 * 16 * 16, activation="relu", input_dim=latent_dim))
    model.add(layers.Reshape((16, 16, 128)))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(128, kernel_size=3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(64, kernel_size=3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Conv2D(3, kernel_size=3, padding="same", activation='tanh'))
    
    return model

def generate_images(generator, latent_dim, num_images):
    noise = tf.random.normal([num_images, latent_dim])
    generated_images = generator(noise)
    return generated_images.numpy()

def Generator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (4, 4), strides=2, padding='same', input_shape=(256, 256, 3)),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, (4, 4), strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2D(256, (4, 4), strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(3, (4, 4), strides=2, padding='same', activation='tanh')
    ])
    return model