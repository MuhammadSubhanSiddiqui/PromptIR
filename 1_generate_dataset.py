import cv2
import numpy as np
import os
import random
import urllib.request
import zipfile

# ==========================================
# 1. DOWNLOADING & SETUP
# ==========================================
def download_div2k(data_dir="raw_data"):
    """Downloads and extracts the DIV2K Validation dataset."""
    url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    zip_path = os.path.join(data_dir, "DIV2K_valid_HR.zip")
    extract_path = os.path.join(data_dir, "DIV2K_valid_HR")

    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(extract_path):
        print("Downloading DIV2K Validation Dataset (approx. 700MB)...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)
        print("Download complete.")
    else:
        print("DIV2K dataset already exists locally.")
        
    return extract_path

# ==========================================
# 2. DEGRADATION FUNCTIONS
# ==========================================
def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_img = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def add_synthetic_rain(image):
    h, w, _ = image.shape
    rain_drops = np.zeros((h, w), dtype=np.uint8)
    num_drops = random.randint(300, 600)
    for _ in range(num_drops):
        x, y = random.randint(0, w-1), random.randint(0, h-1)
        rain_drops[y, x] = 255
        
    kernel = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
    rain_streaks = cv2.filter2D(rain_drops, -1, kernel)
    rain_streaks = cv2.blur(rain_streaks, (3, 3))
    
    rain_streaks_colored = cv2.cvtColor(rain_streaks, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image, 0.8, rain_streaks_colored, 0.3, 0)

def add_dust_haze(image, intensity=0.6):
    dust_color = np.full_like(image, (130, 180, 210), dtype=np.uint8) 
    hazed_img = cv2.addWeighted(image, 1 - intensity, dust_color, intensity, 0)
    return cv2.GaussianBlur(hazed_img, (7, 7), 0)

# ==========================================
# 3. PATCH EXTRACTION & PIPELINE
# ==========================================
def extract_random_patch(image, patch_size=512):
    """Extracts a random square patch from a high-res image."""
    h, w, _ = image.shape
    if h <= patch_size or w <= patch_size:
        return cv2.resize(image, (patch_size, patch_size))
    
    y = random.randint(0, h - patch_size)
    x = random.randint(0, w - patch_size)
    return image[y:y+patch_size, x:x+patch_size]

def build_fine_tuning_dataset(input_dir, output_dir="custom_dataset"):
    splits = ['train', 'test']
    categories = ['clean', 'noisy', 'rainy', 'dusty']
    
    for split in splits:
        for cat in categories:
            os.makedirs(os.path.join(output_dir, split, cat), exist_ok=True)

    images = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))]
    random.shuffle(images)
    
    total_clean_patches = 0
    print("Generating 200 Clean Patches and Synthesizing Degradations...")

    for i, filename in enumerate(images):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None: continue

        # Extract 2 distinct patches per image (100 images * 2 = 200 patches)
        for patch_idx in range(2):
            patch = extract_random_patch(img)
            
            # 80/20 Split Route
            split_name = 'train' if total_clean_patches < 160 else 'test'
            base_name = f"patch_{i}_{patch_idx}.png"

            # Apply degradations
            noisy = add_gaussian_noise(patch)
            rainy = add_synthetic_rain(patch)
            dusty = add_dust_haze(patch)

            # Save
            cv2.imwrite(os.path.join(output_dir, split_name, 'clean', base_name), patch)
            cv2.imwrite(os.path.join(output_dir, split_name, 'noisy', base_name), noisy)
            cv2.imwrite(os.path.join(output_dir, split_name, 'rainy', base_name), rainy)
            cv2.imwrite(os.path.join(output_dir, split_name, 'dusty', base_name), dusty)
            
            total_clean_patches += 1

    print(f"Success! Engineered {total_clean_patches} pairs. Saved in '{output_dir}'.")

if __name__ == "__main__":
    raw_path = download_div2k()
    build_fine_tuning_dataset(input_dir=raw_path)