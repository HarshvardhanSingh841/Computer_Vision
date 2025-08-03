import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


torch.manual_seed(42)
np.random.seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import os
import numpy as np

# Define RGB -> class index mapping based on your unique RGBs
RGB_TO_CLASS = {
    (0, 0, 0): 0,      # Background
    (1, 0, 0): 1,
    (2, 0, 0): 2,
    (3, 0, 0): 3,
    (4, 0, 0): 4,
    (5, 0, 0): 5,
    (6, 0, 0): 6,
    (7, 0, 0): 7,
    (8, 0, 0): 8,
    (9, 0, 0): 9,
    (10, 0, 0): 10,
    (11, 0, 0): 11,
    (12, 0, 0): 12,
}

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        self.images = []
        for f in os.listdir(images_dir):
            img_path = os.path.join(images_dir, f)
            mask_path = os.path.join(masks_dir, f)
            if os.path.isfile(img_path) and os.path.isfile(mask_path):
                self.images.append(f)
        
        print(f"Found {len(self.images)} valid image-mask pairs in {images_dir}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("RGB")  # force RGB mode

            if self.transform:
                image = self.transform(image)

            # Convert mask (H, W, 3) → (H, W) with class indices
            mask_np = np.array(mask)
            mask_class = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)

            for rgb, cls in RGB_TO_CLASS.items():
                matches = np.all(mask_np == rgb, axis=-1)
                mask_class[matches] = cls

            mask_tensor = torch.from_numpy(mask_class).long()

            return image, mask_tensor

        except Exception as e:
            # print(f"Error loading {img_path} or {mask_path}: {e}")
            placeholder_img = torch.zeros((3, 256, 256), dtype=torch.float32)
            placeholder_mask = torch.zeros((256, 256), dtype=torch.long)
            return placeholder_img, placeholder_mask
        
        
# Define simple transforms using standard torchvision transforms
def get_train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# Define the U-Net architecture
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=13):  # 13 classes as per output dimension
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        # Fixing the input channels for decoder blocks
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # 512 + 512 = 1024
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)   # 256 + 256 = 512
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)   # 128 + 128 = 256
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)    # 64 + 64 = 128
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.up4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        
        x = self.up3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Final output
        out = self.final_conv(x)
        return out
    
def mean_iou(pred, target, num_classes=13):
    ious = []
    pred = pred.argmax(dim=1).cpu().numpy()
    target = target.squeeze(1).cpu().numpy()
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        
        if union > 0:
            ious.append(float(intersection) / float(union))
    
    return np.mean(ious) if len(ious) > 0 else 0.0


# Training function with robust error handling
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    train_losses = []
    train_ious = []
    val_losses = []
    val_ious = []
    
    best_val_iou = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        epoch_train_iou = 0
        batch_count = 0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            try:
                # Move data to device
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks.squeeze(1))
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Calculate metrics
                batch_loss = loss.item()
                batch_iou = mean_iou(outputs, masks)
                
                epoch_train_loss += batch_loss
                epoch_train_iou += batch_iou
                batch_count += 1
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        # Calculate epoch metrics
        if batch_count > 0:
            epoch_train_loss /= batch_count
            epoch_train_iou /= batch_count
            
        train_losses.append(epoch_train_loss)
        train_ious.append(epoch_train_iou)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        epoch_val_iou = 0
        batch_count = 0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                try:
                    # Move data to device
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, masks.squeeze(1))
                    
                    # Calculate metrics
                    batch_loss = loss.item()
                    batch_iou = mean_iou(outputs, masks)
                    
                    epoch_val_loss += batch_loss
                    epoch_val_iou += batch_iou
                    batch_count += 1
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        # Calculate epoch metrics
        if batch_count > 0:
            epoch_val_loss /= batch_count
            epoch_val_iou /= batch_count
            
        val_losses.append(epoch_val_loss)
        val_ious.append(epoch_val_iou)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train mIoU: {epoch_train_iou:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val mIoU: {epoch_val_iou:.4f}")
        
        # Save the best model
        if epoch_val_iou > best_val_iou:
            best_val_iou = epoch_val_iou
            try:
                torch.save(model.state_dict(), "best_unet_model.pth")
                print("Best model saved!")
            except Exception as e:
                print(f"Error saving model: {e}")
    
    return train_losses, train_ious, val_losses, val_ious

# Visualization function
def visualize_predictions(model, test_loader, num_samples=5):
    model.eval()
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    if num_samples == 1:
        axs = [axs]  # Handle the case when num_samples = 1
    
    sample_idx = 0
    
    with torch.no_grad():
        for images, masks in test_loader:
            if sample_idx >= num_samples:
                break
                
            try:
                # Move data to device
                images = images.to(device)
                
                # Forward pass
                outputs = model(images)
                pred_masks = outputs.argmax(dim=1)
                
                # Convert tensors to numpy for visualization
                image = images[0].cpu().permute(1, 2, 0).numpy()
                # Denormalize
                image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                image = np.clip(image, 0, 1)
                
                true_mask = masks[0].squeeze().cpu().numpy()
                pred_mask = pred_masks[0].cpu().numpy()
                
                # Create a colormap for better visualization
                cmap = plt.cm.get_cmap('viridis', 13)  # 13 classes
                
                # Plot
                axs[sample_idx][0].imshow(image)
                axs[sample_idx][0].set_title('Input Image')
                axs[sample_idx][0].axis('off')
                
                axs[sample_idx][1].imshow(true_mask, cmap=cmap, vmin=0, vmax=12)
                axs[sample_idx][1].set_title('Ground Truth')
                axs[sample_idx][1].axis('off')
                
                axs[sample_idx][2].imshow(pred_mask, cmap=cmap, vmin=0, vmax=12)
                axs[sample_idx][2].set_title('Prediction')
                axs[sample_idx][2].axis('off')
                
                sample_idx += 1
                
            except Exception as e:
                print(f"Error visualizing sample: {e}")
                continue
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()
    

# Plot metrics function
def plot_metrics(train_losses, train_ious, val_losses, val_ious):
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_ious, label='Training mIoU')
    plt.plot(val_ious, label='Validation mIoU')
    plt.title('mIoU over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()
    
# Evaluate on test set
def evaluate_model(model, test_loader):
    model.eval()
    test_iou = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating on test set"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            test_iou += mean_iou(outputs, masks)
    
    test_iou /= len(test_loader)
    print(f"Test mIoU: {test_iou:.4f}")
    return test_iou

# Main execution function
def main():
    # Set paths to your dataset
    data_dir = "./dataset_256"
    train_img_dir = os.path.join(data_dir, "train/images")
    train_mask_dir = os.path.join(data_dir, "train/labels")
    test_img_dir = os.path.join(data_dir, "test/images")
    test_mask_dir = os.path.join(data_dir, "test/labels")
    
    # Create datasets
    train_dataset = SegmentationDataset(
        train_img_dir, 
        train_mask_dir,
        transform=get_train_transform()
    )
    
    # Split into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    test_dataset = SegmentationDataset(
        test_img_dir,
        test_mask_dir,
        transform=get_test_transform()
    )
    
    # Create data loaders with safer settings
    batch_size = 8
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,  # Start with 0, increase if stable
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        #prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        #prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        #prefetch_factor=2
    )
    
    # Create the model
    model = UNet(in_channels=3, out_channels=13).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("Starting training...")
    train_losses, train_ious, val_losses, val_ious = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=30
    )
    
    # Plot training metrics
    plot_metrics(train_losses, train_ious, val_losses, val_ious)
    
    # Load best model for evaluation
    try:
        model.load_state_dict(torch.load("best_unet_model.pth"))
        print("Loaded best model for evaluation")
    except Exception as e:
        print(f"Error loading best model: {e}")
    
    # Evaluate on test set
    model.eval()
    test_iou = 0
    test_batches = 0
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            try:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                test_iou += mean_iou(outputs, masks)
                test_batches += 1
            except Exception as e:
                print(f"Error in test evaluation: {e}")
                continue
    
    if test_batches > 0:
        avg_test_iou = test_iou / test_batches
        print(f"Test mIoU: {avg_test_iou:.4f}")
    
    # Visualize predictions
    print("Visualizing predictions...")
    visualize_predictions(model, test_loader, num_samples=5)

if __name__ == "__main__":
    main()