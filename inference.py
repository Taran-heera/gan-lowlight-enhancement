"""
Inference script for GAN-based low-light image enhancement
Usage: python inference.py <image_path>
"""
import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf

# Add src to path
sys.path.insert(0, 'src')

def load_model(model_path):
    """Load a saved model"""
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded: {model_path}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

def preprocess_image(image_path, size=256):
    """Load and preprocess image"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((size, size))
        img_array = np.array(img) / 127.5 - 1  # Normalize to [-1, 1]
        return np.expand_dims(img_array, axis=0), img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None

def postprocess_image(img_array):
    """Convert from model output to PIL Image"""
    # Denormalize from [-1, 1] to [0, 255]
    img = ((img_array + 1) * 127.5).astype(np.uint8)
    return Image.fromarray(img)

def enhance_image(image_path, model_path='src/generator.h5', output_path='enhanced.png'):
    """Enhance a single low-light image"""
    
    print("\n" + "="*60)
    print("GAN-Based Low-Light Image Enhancement")
    print("="*60 + "\n")
    
    # Load model
    print("Loading generator model...")
    generator = load_model(model_path)
    if generator is None:
        return False
    
    # Load and preprocess image
    print(f"Loading image: {image_path}")
    img_input, original_img = preprocess_image(image_path)
    if img_input is None:
        return False
    
    # Generate enhanced image
    print("Generating enhancement...")
    try:
        enhanced = generator.predict(img_input, verbose=0)
        enhanced_img = postprocess_image(enhanced[0])
        
        # Save result
        enhanced_img.save(output_path)
        print(f"✓ Enhanced image saved: {output_path}")
        
        # Display info
        print(f"\nResult:")
        print(f"  Input size: {original_img.size}")
        print(f"  Output size: {enhanced_img.size}")
        print(f"  Format: PNG")
        
        return True
    except Exception as e:
        print(f"✗ Error during enhancement: {e}")
        return False

def batch_enhance(input_dir, output_dir='enhanced_images', model_path='src/generator.h5'):
    """Enhance all images in a directory"""
    
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = [f for f in os.listdir(input_dir) 
              if f.lower().endswith(extensions)]
    
    if not images:
        print(f"No images found in {input_dir}")
        return
    
    print(f"\nFound {len(images)} images to enhance\n")
    
    # Load model once
    generator = load_model(model_path)
    if generator is None:
        return
    
    # Process each image
    for i, filename in enumerate(images, 1):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f'enhanced_{i:03d}.png')
        
        print(f"[{i}/{len(images)}] Processing: {filename}...", end=' ')
        
        img_input, _ = preprocess_image(input_path)
        if img_input is None:
            print("✗ Failed to load")
            continue
        
        try:
            enhanced = generator.predict(img_input, verbose=0)
            enhanced_img = postprocess_image(enhanced[0])
            enhanced_img.save(output_path)
            print(f"✓ Saved as {os.path.basename(output_path)}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\n✓ Batch processing complete! Results in: {output_dir}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image:  python inference.py <image_path>")
        print("  Batch images:  python inference.py --batch <input_dir> <output_dir>")
        print("\nExample:")
        print("  python inference.py test.jpg")
        print("  python inference.py --batch data/test enhanced_output")
        sys.exit(1)
    
    if sys.argv[1] == '--batch':
        input_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/test'
        output_dir = sys.argv[3] if len(sys.argv) > 3 else 'enhanced_images'
        batch_enhance(input_dir, output_dir)
    else:
        image_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'enhanced.png'
        enhance_image(image_path, output_path=output_path)
