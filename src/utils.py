import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Dataset class
class LowLightDataset(Dataset):
    def __init__(self, low_light_dir, normal_light_dir, img_size=256, is_train=True):
        self.low_light_dir = low_light_dir
        self.normal_light_dir = normal_light_dir
        self.img_size = img_size
        self.is_train = is_train
        
        self.low_images = sorted([f for f in os.listdir(low_light_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.normal_images = sorted([f for f in os.listdir(normal_light_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        assert len(self.low_images) == len(self.normal_images), "Number of low-light and normal-light images must match"
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]) if not is_train else transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.low_images)
    
    def __getitem__(self, idx):
        low_img = Image.open(os.path.join(self.low_light_dir, self.low_images[idx])).convert('RGB')
        normal_img = Image.open(os.path.join(self.normal_light_dir, self.normal_images[idx])).convert('RGB')
        
        low_img = self.transform(low_img)
        normal_img = self.transform(normal_img)
        
        return {'low': low_img, 'normal': normal_img, 'filename': self.low_images[idx]}

# Visualization functions
def denormalize(tensor):
    return (tensor + 1) / 2

def save_sample_images(low_imgs, enhanced_imgs, normal_imgs, save_path):
    low_imgs = denormalize(low_imgs).cpu()
    enhanced_imgs = denormalize(enhanced_imgs).cpu()
    normal_imgs = denormalize(normal_imgs).cpu()
    
    fig, axes = plt.subplots(3, min(4, low_imgs.size(0)), figsize=(15, 12))
    if low_imgs.size(0) == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(min(4, low_imgs.size(0))):
        axes[0, i].imshow(low_imgs[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title('Low-Light Input')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(enhanced_imgs[i].permute(1, 2, 0).numpy())
        axes[1, i].set_title('Enhanced Output')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(normal_imgs[i].permute(1, 2, 0).numpy())
        axes[2, i].set_title('Ground Truth')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# Metrics functions
def calculate_psnr(img1, img2, max_value=1.0):
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    if len(img1.shape) == 4:
        psnr_values = []
        for i in range(img1.shape[0]):
            im1 = np.transpose(img1[i], (1, 2, 0))
            im2 = np.transpose(img2[i], (1, 2, 0))
            psnr_values.append(psnr(im1, im2, data_range=max_value))
        return np.mean(psnr_values)
    else:
        return psnr(img1, img2, data_range=max_value)

def calculate_ssim(img1, img2, max_value=1.0):
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    if len(img1.shape) == 4:
        ssim_values = []
        for i in range(img1.shape[0]):
            im1 = np.transpose(img1[i], (1, 2, 0))
            im2 = np.transpose(img2[i], (1, 2, 0))
            ssim_values.append(ssim(im1, im2, data_range=max_value, channel_axis=2))
        return np.mean(ssim_values)
    else:
        return ssim(img1, img2, data_range=max_value, channel_axis=2)

# Logging function (updated from your original)
def log_training_progress(epoch, g_loss, d_loss):
    print(f'Epoch: {epoch}, Generator Loss: {g_loss:.4f}, Discriminator Loss: {d_loss:.4f}')

# Additional helper functions
def preprocess_image(image):
    # Resize and normalize for GAN input
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image)

def save_image(image, path):
    # Denormalize and save
    image = denormalize(image).cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    plt.imsave(path, image)

def load_data(data_path):
    low_dir = os.path.join(data_path, 'train', 'low')
    normal_dir = os.path.join(data_path, 'train', 'normal')
    low_images = []
    normal_images = []
    for img_name in os.listdir(low_dir):
        low_img = np.array(Image.open(os.path.join(low_dir, img_name)).resize((256, 256))) / 127.5 - 1
        normal_img = np.array(Image.open(os.path.join(normal_dir, img_name)).resize((256, 256))) / 127.5 - 1
        low_images.append(low_img)
        normal_images.append(normal_img)
    return np.array(low_images), np.array(normal_images)

def save_generated_images(generator, epoch, low_images, result_dir='results'):
    os.makedirs(result_dir, exist_ok=True)
    noise = np.random.normal(0, 1, (4, 256, 256, 3))  # Dummy for demo; use actual low_images
    generated = generator.predict(noise)
    for i in range(4):
        img = Image.fromarray(((generated[i] + 1) * 127.5).astype(np.uint8))
        img.save(os.path.join(result_dir, f'generated_{epoch}_{i}.png'))