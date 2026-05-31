import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# Ensure local 'net' package is found
sys.path.append(os.getcwd())
from net.model import PromptIR

def evaluate_test_set():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Evaluation on: {device}...")
    
    # 1. Initialize and load your fine-tuned weights
    model = PromptIR(decoder=True).to(device)
    
    weights_path = 'custom_promptir_weights.pth'
    if not os.path.exists(weights_path):
        print(f"Error: Could not find {weights_path}. Did you run the training script?")
        return
        
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # 2. Setup transforms and directories
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    test_clean_dir = 'custom_dataset/test/clean'
    degradation_types = ['noisy', 'rainy', 'dusty']

    # 3. Print table header
    print("\n" + "="*75)
    print(f"{'Degradation':<15} | {'Degraded PSNR':<15} | {'Restored PSNR':<15} | {'Restored SSIM':<15}")
    print("-" * 75)

    # 4. Evaluation Loop
    for deg in degradation_types:
        deg_dir = f'custom_dataset/test/{deg}'
        
        if not os.path.exists(deg_dir) or not os.path.exists(test_clean_dir):
            print(f"Skipping {deg}: Test directories not found.")
            continue
            
        images = [f for f in os.listdir(test_clean_dir) if f.endswith(('.png', '.jpg'))]
        
        deg_psnr_list = []
        restored_psnr_list = []
        restored_ssim_list = []

        for img_name in images:
            # Load clean and degraded images as standard NumPy arrays for metrics
            clean_path = os.path.join(test_clean_dir, img_name)
            deg_path = os.path.join(deg_dir, img_name)
            
            clean_np = cv2.resize(cv2.imread(clean_path), (128, 128))
            deg_np = cv2.resize(cv2.imread(deg_path), (128, 128))
            
            # Prepare PIL image for the PyTorch model
            deg_pil = Image.open(deg_path).convert('RGB')
            input_tensor = transform(deg_pil).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                restored_tensor = model(input_tensor)
            
            # Convert restored tensor back to a standard NumPy uint8 image
            restored_np = restored_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
            restored_np = np.clip(restored_np * 255.0, 0, 255).astype(np.uint8)
            restored_np = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)

            # Calculate metrics
            # PSNR: Higher is better (less noise)
            # SSIM: Closer to 1.0 is better (structural match)
            d_psnr = psnr_metric(clean_np, deg_np)
            r_psnr = psnr_metric(clean_np, restored_np)
            
            # win_size must be odd and smaller than the image dimensions. 
            # channel_axis=2 explicitly tells it we are using color images (H, W, C)
            r_ssim = ssim_metric(clean_np, restored_np, channel_axis=2, data_range=255)

            deg_psnr_list.append(d_psnr)
            restored_psnr_list.append(r_psnr)
            restored_ssim_list.append(r_ssim)

        # Print the averaged results for this specific degradation
        avg_deg_psnr = np.mean(deg_psnr_list)
        avg_res_psnr = np.mean(restored_psnr_list)
        avg_res_ssim = np.mean(restored_ssim_list)
        
        print(f"{deg.capitalize():<15} | {avg_deg_psnr:<15.2f} | {avg_res_psnr:<15.2f} | {avg_res_ssim:<15.4f}")

    print("=" * 75 + "\n")

if __name__ == "__main__":
    evaluate_test_set()