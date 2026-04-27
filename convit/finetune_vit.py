import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import glob
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ConViT-Finetune")  # Updated logger name for ConViT model

# Global device setup - do this only once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)

# Create necessary directories - do this only once at the start
os.makedirs('models', exist_ok=True)
logger.info("Created 'models' directory (if it didn't exist)")

def get_matching_image_pairs(target_folder, pixelILT_folder, report_mismatches=True):
    """Get matching image pairs between target and pixelILT folders with detailed reporting"""

    target_folder = Path(target_folder)
    pixelILT_folder = Path(pixelILT_folder)
    valid_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

    target_files = {f.stem: f for f in target_folder.glob('*.*')
                   if f.suffix.lower() in valid_extensions}
    pixelILT_files = {f.stem: f for f in pixelILT_folder.glob('*.*')
                    if f.suffix.lower() in valid_extensions}

    # Find common stems for matching pairs
    common_stems = set(target_files.keys()) & set(pixelILT_files.keys())
    
    # Report matching statistics
    logger.info(f"Found {len(target_files)} files in target folder")
    logger.info(f"Found {len(pixelILT_files)} files in pixelILT folder")
    logger.info(f"Found {len(common_stems)} matching image pairs")

    # Report mismatches if requested
    if report_mismatches and (len(target_files) != len(common_stems) or len(pixelILT_files) != len(common_stems)):
        target_only = set(target_files.keys()) - set(pixelILT_files.keys())
        pixelILT_only = set(pixelILT_files.keys()) - set(target_files.keys())
        
        if target_only:
            logger.warning(f"Found {len(target_only)} files in target folder with no match in pixelILT folder:")
            for stem in sorted(list(target_only)[:10]):  # Show max 10 examples
                logger.warning(f"  - {stem}{target_files[stem].suffix}")
            if len(target_only) > 10:
                logger.warning(f"  ... and {len(target_only) - 10} more")
                
        if pixelILT_only:
            logger.warning(f"Found {len(pixelILT_only)} files in pixelILT folder with no match in target folder:")
            for stem in sorted(list(pixelILT_only)[:10]):  # Show max 10 examples
                logger.warning(f"  - {stem}{pixelILT_files[stem].suffix}")
            if len(pixelILT_only) > 10:
                logger.warning(f"  ... and {len(pixelILT_only) - 10} more")

    if len(common_stems) == 0:
        logger.error("No matching image pairs found! Please check your data directories.")
        raise ValueError("No matching image pairs found between target and pixelILT folders")

    # Create and return the pairs
    pairs = [(str(target_files[stem]), str(pixelILT_files[stem])) for stem in common_stems]
    return pairs

# Custom dataset class for image-to-image translation
class PixelILTDataset(Dataset):
    def __init__(self, target_dir, pixelilt_dir, transform=None):
        self.target_dir = target_dir
        self.pixelilt_dir = pixelilt_dir
        self.transform = transform
        
        # Get matching image pairs using the more robust function
        self.image_pairs = get_matching_image_pairs(target_dir, pixelilt_dir)
        logger.info(f"Dataset loaded with {len(self.image_pairs)} image pairs")
        
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        # Get the image pair
        target_path, pixelilt_path = self.image_pairs[idx]
        
        # Load images
        target_img = Image.open(target_path).convert('RGB')
        pixelilt_img = Image.open(pixelilt_path).convert('RGB')
        
        # Apply transforms if specified
        if self.transform:
            target_img = self.transform(target_img)
            pixelilt_img = self.transform(pixelilt_img)
        
        return target_img, pixelilt_img

# Define the ConViT-based image-to-image translation model
class ConvitImageTranslator(nn.Module):
    def __init__(self, pretrained_model_name="convit_small", num_classes=None, use_pretrained=True):
        super(ConvitImageTranslator, self).__init__()
        
        # Load the ConViT model with pretrained weights for better detail capture
        self.convit_encoder = timm.create_model(
            pretrained_model_name, 
            pretrained=use_pretrained,  # Use pretrained weights for better feature extraction
            num_classes=0  # Remove classifier head
        )
        
        # Get feature dimension from the ConViT model
        if 'tiny' in pretrained_model_name:
            convit_feature_dim = 192
        elif 'small' in pretrained_model_name:
            convit_feature_dim = 384
        elif 'base' in pretrained_model_name:
            convit_feature_dim = 768
        else:
            # Default, will try to get from model properties
            try:
                convit_feature_dim = self.convit_encoder.head.in_features
            except:
                logger.warning("Could not detect ConViT feature dimension, using default 384")
                convit_feature_dim = 384
                
        logger.info(f"Using ConViT model with feature dimension: {convit_feature_dim}")
        
        # Modified decoder network to ensure exactly 224x224 output
        self.decoder = nn.Sequential(
            nn.Linear(432, 4096),  # Fixed dimension to match ConViT output
            nn.ReLU(),
            nn.Linear(4096, 14 * 14 * 256),  # 14×14×256 features to start from
            nn.ReLU(),
            nn.Unflatten(1, (256, 14, 14)),  # Reshape to [batch, 256, 14, 14]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [batch, 128, 28, 28]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [batch, 64, 56, 56]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [batch, 32, 112, 112]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # [batch, 3, 224, 224]
            nn.Tanh()
        )
        
        # Move all components to the same device immediately after initialization
        self.to(device)
        
        # Debug shape issues with a simple forward pass
        self._debug_shapes()
    
    def _debug_shapes(self):
        """Debug the output shape of each layer"""
        try:
            # Create a dummy input of the right size
            dummy_input = torch.randn(1, 3, 224, 224)
            
            # Run a forward pass through the encoder
            with torch.no_grad():
                features = self.convit_encoder(dummy_input)
                print(f"Encoder output shape: {features.shape}")
                
                # Trace through the decoder
                x = self.decoder[0](features)
                print(f"After first linear: {x.shape}")
                
                x = self.decoder[1](x)  # ReLU
                x = self.decoder[2](x)  # Second linear
                print(f"After second linear: {x.shape}")
                
                x = self.decoder[3](x)  # ReLU
                x = self.decoder[4](x)  # Unflatten
                print(f"After unflatten: {x.shape}")
                
                # Check each ConvTranspose
                x = self.decoder[5](x)
                print(f"After ConvTranspose 1: {x.shape}")
                
                x = self.decoder[6](x)  # ReLU
                x = self.decoder[7](x)
                print(f"After ConvTranspose 2: {x.shape}")
                
                x = self.decoder[8](x)  # ReLU
                x = self.decoder[9](x)
                print(f"After ConvTranspose 3: {x.shape}")
                
                x = self.decoder[10](x)  # ReLU
                x = self.decoder[11](x)
                print(f"After ConvTranspose 4 (final): {x.shape}")
        except Exception as e:
            print(f"Debug shapes error (can be ignored): {e}")
    
    def forward(self, x):
        # Extract features using the ConViT encoder
        # Make sure everything is on the same device
        x = x.to(next(self.parameters()).device)
        
        # Ensure ConViT encoder is on the same device as input
        self.convit_encoder = self.convit_encoder.to(x.device)
        
        # Get features
        features = self.convit_encoder(x)
        
        # Generate output image using the decoder (ensure on same device)
        output = self.decoder(features)
        
        # Ensure output has the right size (defensive programming)
        if output.shape[-1] != 224 or output.shape[-2] != 224:
            output = torch.nn.functional.interpolate(output, size=(224, 224), mode='bilinear', align_corners=False)
            
        return output

# Set up data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ConViT typically uses 224×224 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_model(model, train_loader, val_loader, num_epochs=30, lr=0.0001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Update LR scheduler
        scheduler.step(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join('models', 'best_convit_translator.pth'))
            print(f"Model saved with validation loss: {best_val_loss:.4f}")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join('models', 'convit_loss_curve.png'))
    plt.show()
    
    return model

def visualize_results(model, test_loader, num_samples=5):
    model.eval()
    
    # Get a batch of test data
    test_inputs, test_targets = next(iter(test_loader))
    test_inputs, test_targets = test_inputs[:num_samples], test_targets[:num_samples]
    
    # Generate outputs
    with torch.no_grad():
        test_inputs_gpu = test_inputs.to(device)
        test_outputs = model(test_inputs_gpu).cpu()
    
    # Denormalize images for visualization
    denorm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    test_inputs = torch.stack([denorm(img) for img in test_inputs])
    test_outputs = torch.stack([denorm(img) for img in test_outputs])
    test_targets = torch.stack([denorm(img) for img in test_targets])
    
    # Convert to numpy for visualization
    test_inputs = torch.clamp(test_inputs, 0, 1).numpy().transpose(0, 2, 3, 1)
    test_outputs = torch.clamp(test_outputs, 0, 1).numpy().transpose(0, 2, 3, 1)
    test_targets = torch.clamp(test_targets, 0, 1).numpy().transpose(0, 2, 3, 1)
    
    # Plot results
    plt.figure(figsize=(15, 5 * num_samples))
    for i in range(num_samples):
        # Display input image
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(test_inputs[i])
        plt.title('Input (Target)')
        plt.axis('off')
        
        # Display output image
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(test_outputs[i])
        plt.title('Output (Generated)')
        plt.axis('off')
        
        # Display ground truth
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(test_targets[i])
        plt.title('Ground Truth (PixelILT)')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('models/convit_results_visualization.png')
    plt.show()

def main():
    # Define paths to your dataset
    target_dir = "target"
    pixelilt_dir = "pixelILT"
    
    # Create dataset
    full_dataset = PixelILTDataset(
        target_dir=target_dir,
        pixelilt_dir=pixelilt_dir,
        transform=transform
    )
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, temp_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size + test_size]
    )
    val_dataset, test_dataset = torch.utils.data.random_split(
        temp_dataset, [val_size, test_size]
    )
    
    # Create data loaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model - updated to use ConViT
    model = ConvitImageTranslator(pretrained_model_name="convit_small", use_pretrained=True)
    model = model.to(device)  # Use the global device
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in the model: {total_params:,}")
    
    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=30, lr=0.0001)
    
    # Load best model for evaluation
    model_path = os.path.join('models', 'best_convit_translator.pth')
    model.load_state_dict(torch.load(model_path))
    
    # Visualize results
    visualize_results(model, test_loader, num_samples=5)
    
    print("Training and evaluation completed successfully!")

if __name__ == "__main__":
    main()