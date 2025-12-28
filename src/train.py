import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

try:
    from generator import Generator
    from discriminator import Discriminator
    from utils import load_data, save_generated_images
except ImportError as e:
    print(f"Import error: {e}. Ensure generator.py, discriminator.py, and utils.py exist in src/.")
    exit(1)

# Set parameters
EPOCHS = 1000  # Reduced for practicality
BATCH_SIZE = 8  # Reduced for memory
SAVE_INTERVAL = 100

# Load dataset
try:
    low_data, normal_data = load_data('data/raw/LOL')
    print(f"Data loaded: {low_data.shape[0]} low-light images, {normal_data.shape[0]} normal images.")
except Exception as e:
    print(f"Data loading error: {e}. Check path 'data/raw/LOL' and ensure LOL dataset is downloaded.")
    exit(1)

# Initialize models
generator = Generator()
discriminator = Discriminator()

# Compile discriminator
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Create GAN model
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(256, 256, 3))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Training loop
for epoch in range(EPOCHS):
    # Sample batch
    idx = np.random.randint(0, low_data.shape[0], BATCH_SIZE)
    real_images = normal_data[idx]
    low_images = low_data[idx]
    generated_images = generator.predict(low_images)

    # Train discriminator
    d_loss_real = discriminator.train_on_batch(real_images, np.ones((BATCH_SIZE, 1)))
    d_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((BATCH_SIZE, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train generator
    g_loss = gan.train_on_batch(low_images, np.ones((BATCH_SIZE, 1)))

    # Print progress
    if epoch % SAVE_INTERVAL == 0:
        print(f'Epoch: {epoch}, D Loss: {d_loss[0]}, D Acc.: {100 * d_loss[1]}, G Loss: {g_loss}')
        try:
            save_generated_images(generator, epoch, low_images)
        except Exception as e:
            print(f"Save error: {e}")

# Save the models
generator.save('models/generator.h5')
discriminator.save('models/discriminator.h5')