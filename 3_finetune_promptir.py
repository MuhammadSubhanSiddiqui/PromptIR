import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Adding current directory to path to ensure local 'net' package is found
sys.path.append(os.getcwd())

# Import the model from the restored directory
from net.model import PromptIR

# ==========================================
# 1. CUSTOM DATALOADER
# ==========================================
class SyntheticDegradationDataset(Dataset):
    """Loads paired clean and degraded images from the custom Layer 1 dataset."""
    def __init__(self, dataset_dir="custom_dataset", split="train"):
        self.clean_dir = os.path.join(dataset_dir, split, "clean")
        self.degraded_dirs = [
            os.path.join(dataset_dir, split, "noisy"),
            os.path.join(dataset_dir, split, "rainy"),
            os.path.join(dataset_dir, split, "dusty")
        ]
        self.image_names = os.listdir(self.clean_dir)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names) * 3

    def __getitem__(self, idx):
        clean_idx = idx // 3
        deg_type_idx = idx % 3
        img_name = self.image_names[clean_idx]

        clean_path = os.path.join(self.clean_dir, img_name)
        deg_path = os.path.join(self.degraded_dirs[deg_type_idx], img_name)

        clean_img = cv2.cvtColor(cv2.imread(clean_path), cv2.COLOR_BGR2RGB)
        deg_img = cv2.cvtColor(cv2.imread(deg_path), cv2.COLOR_BGR2RGB)

        return self.transform(deg_img), self.transform(clean_img)

# ==========================================
# 2. MODEL FREEZING LOGIC
# ==========================================
def setup_model_for_finetuning(model):
    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layers related to the decoder and prompts based on naming convention
    for name, param in model.named_parameters():
        if any(key in name.lower() for key in ["dec_level", "up", "reduce", "prompt"]):
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {trainable_params:,}")
    return model

# ==========================================
# 3. THE TRAINING LOOP
# ==========================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")

    train_dataset = SyntheticDegradationDataset(split="train")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    # decoder=True is critical to align channels in the skip connections
    model = PromptIR(decoder=True)

    # Load weights if available
    if os.path.exists("pretrained_weights.pth"):
        model.load_state_dict(torch.load("pretrained_weights.pth"), strict=False)

    model = setup_model_for_finetuning(model).to(device)
    criterion = nn.L1Loss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=1e-4)

    epochs = 3
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for degraded_imgs, clean_imgs in train_loader:
            degraded_imgs, clean_imgs = degraded_imgs.to(device), clean_imgs.to(device)

            optimizer.zero_grad()
            outputs = model(degraded_imgs)
            loss = criterion(outputs, clean_imgs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Average L1 Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "custom_promptir_weights.pth")
    print("Training complete. Weights saved to custom_promptir_weights.pth.")

if __name__ == "__main__":
    train_model()