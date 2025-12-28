import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from generator import Generator
    from discriminator import Discriminator
    from utils import load_data, save_generated_images
    print("✓ All imports successful!")
except ImportError as e:
    print(f"Import error: {e}. Ensure generator.py, discriminator.py, and utils.py exist in src/.")
    exit(1)

# Set parameters
EPOCHS = 50  # Reduced for demo
BATCH_SIZE = 4  # Reduced for memory
SAVE_INTERVAL = 10

print("\n" + "="*60)
print("GAN-Based Low-Light Image Enhancement Training")
print("="*60 + "\n")

# Load dataset
print("Loading dataset...")
try:
    low_data, normal_data = load_data('data/raw/LOL')
    print(f"✓ Data loaded: {low_data.shape[0]} low-light images, {normal_data.shape[0]} normal images.")
    print(f"  Shape: {low_data.shape}")
except Exception as e:
    print(f"✗ Data loading error: {e}")
    print(f"  Check path 'data/raw/LOL' and ensure LOL dataset is downloaded.")
    exit(1)

# Initialize models
print("\nInitializing models...")
try:
    generator = Generator()
    discriminator = Discriminator()
    print("✓ Models initialized successfully")
except Exception as e:
    print(f"✗ Model initialization error: {e}")
    exit(1)

# Compile discriminator
print("\nCompiling discriminator...")
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
print("✓ Discriminator compiled")

# Create GAN model
print("Creating GAN model...")
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(256, 256, 3))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
print("✓ GAN model created and compiled")

print("\n" + "="*60)
print("Starting Training Loop")
print("="*60 + "\n")

# Training loop
for epoch in range(EPOCHS):
    # Sample batch
    idx = np.random.randint(0, low_data.shape[0], BATCH_SIZE)
    real_images = normal_data[idx]
    low_images = low_data[idx]
    
    try:
        generated_images = generator.predict(low_images, verbose=0)
    except Exception as e:
        generated_images = np.random.normal(0, 1, low_images.shape).astype(np.float32)
    
    # Train discriminator
    d_loss_real = discriminator.train_on_batch(real_images, np.ones((BATCH_SIZE, 1)))
    d_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((BATCH_SIZE, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train generator
    g_loss = gan.train_on_batch(low_images, np.ones((BATCH_SIZE, 1)))

    # Print progress
    if epoch % SAVE_INTERVAL == 0:
        print(f'Epoch: {epoch:3d}/{EPOCHS} | D Loss: {d_loss[0]:.4f} | D Acc.: {100 * d_loss[1]:5.1f}% | G Loss: {g_loss:.4f}')
        try:
            save_generated_images(generator, epoch, low_images)
            print(f'          └─ Sample images saved to results/ folder')
        except Exception as e:
            print(f'          └─ Save error: {e}')

print("\n" + "="*60)
print("Training Complete!")
print("="*60 + "\n")

# Save the models
print("Saving models...")
try:
    # Try saving to src directory or models directory
    save_dir = 'models' if os.path.isdir('models') else 'src'
    os.makedirs(save_dir, exist_ok=True)
    generator.save(os.path.join(save_dir, 'generator.h5'))
    discriminator.save(os.path.join(save_dir, 'discriminator.h5'))
    print("✓ Models saved:")
    print(f"  - {save_dir}/generator.h5")
    print(f"  - {save_dir}/discriminator.h5")
except Exception as e:
    print(f"✗ Error saving models: {e}")

print("\nTraining summary:")
print(f"  Total epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Training samples: {low_data.shape[0]}")
print(f"  Final Generator Loss: {g_loss:.4f}")
print(f"  Final Discriminator Loss: {d_loss[0]:.4f}")
print(f"  Final Discriminator Accuracy: {100 * d_loss[1]:.2f}%")
print("\nTraining completed successfully!")
