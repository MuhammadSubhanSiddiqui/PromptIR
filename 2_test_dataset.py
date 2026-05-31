import matplotlib.pyplot as plt
import cv2
import os
import random

def verify_dataset(dataset_path="custom_dataset", split="train"):
    """Randomly pulls a pair from the dataset and plots them side-by-side."""
    clean_dir = os.path.join(dataset_path, split, 'clean')
    
    if not os.path.exists(clean_dir):
        print(f"Error: Could not find {clean_dir}. Run the dataset generation script first.")
        return

    # Pick a random image filename
    images = os.listdir(clean_dir)
    if not images:
        print("Directory is empty.")
        return
        
    random_img_name = random.choice(images)
    
    # Construct paths
    paths = {
        "Ground Truth (Clean)": os.path.join(dataset_path, split, 'clean', random_img_name),
        "Gaussian Noise": os.path.join(dataset_path, split, 'noisy', random_img_name),
        "Synthetic Rain": os.path.join(dataset_path, split, 'rainy', random_img_name),
        "Synthetic Dust": os.path.join(dataset_path, split, 'dusty', random_img_name)
    }

    # Plotting
    plt.figure(figsize=(16, 4))
    plt.suptitle(f"Dataset Verification: {random_img_name} ({split.upper()} Set)", fontsize=16)

    for i, (title, img_path) in enumerate(paths.items()):
        if not os.path.exists(img_path):
            print(f"Missing file: {img_path}")
            continue
            
        # OpenCV reads in BGR, Matplotlib expects RGB
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, 4, i + 1)
        plt.imshow(img_rgb)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run this to visually test the training set output
    verify_dataset(split="train")