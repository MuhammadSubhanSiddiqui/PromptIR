import torch
import matplotlib.pyplot as plt
from net.model import PromptIR
from PIL import Image
import torchvision.transforms as transforms
import os
import random

def test_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PromptIR(decoder=True).to(device)
    model.load_state_dict(torch.load('custom_promptir_weights.pth'))
    model.eval()

    # Pick a random degraded image from the test set
    test_clean_dir = 'custom_dataset/test/clean'
    test_noisy_dir = 'custom_dataset/test/noisy'
    img_name = random.choice(os.listdir(test_clean_dir))

    clean_img = Image.open(os.path.join(test_clean_dir, img_name)).convert('RGB')
    degraded_img = Image.open(os.path.join(test_noisy_dir, img_name)).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    input_tensor = transform(degraded_img).unsqueeze(0).to(device)

    with torch.no_grad():
        restored_tensor = model(input_tensor)

    # Plotting
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(degraded_img.resize((128, 128)))
    plt.title('Degraded Input')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(restored_tensor.squeeze().cpu().permute(1, 2, 0).clamp(0, 1))
    plt.title('PromptIR Restored')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(clean_img.resize((128, 128)))
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

test_inference()