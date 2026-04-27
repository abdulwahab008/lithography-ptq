import os
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import numpy as np
import glob
from tqdm import tqdm
import cv2
from skimage import exposure

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
                print("Could not detect ConViT feature dimension, using default 384")
                convit_feature_dim = 384
                
        print(f"Using ConViT model with feature dimension: {convit_feature_dim}")
        
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
    
    def forward(self, x):
        # Extract features using the ConViT encoder
        # Make sure everything is on the same device
        x = x.to(next(self.parameters()).device)
        
        # Ensure ConViT encoder is on the same device as input
        self.convit_encoder = self.convit_encoder.to(x.device)
        
        # Get features
        features = self.convit_encoder(x)
        
        # Generate output image using the decoder
        output = self.decoder(features)
        
        # Ensure output has the right size (defensive programming)
        if output.shape[-1] != 224 or output.shape[-2] != 224:
            output = torch.nn.functional.interpolate(output, size=(224, 224), mode='bilinear', align_corners=False)
            
        return output

def load_model(model_path):
    """Load the trained model from the specified path."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model - updated to use ConvitImageTranslator
    model = ConvitImageTranslator(pretrained_model_name="convit_small", use_pretrained=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

def enhance_details(image, detail_factor=2.0):
    """Enhance details in the image before thresholding"""
    # Convert to float for processing
    image_float = image.astype(np.float32)
    
    # Apply contrast enhancement
    image_enhanced = exposure.equalize_adapthist(image_float) 
    
    # Apply unsharp mask to enhance edges and details
    blurred = cv2.GaussianBlur(image_enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(image_enhanced, 1.0 + detail_factor, blurred, -detail_factor, 0)
    
    # Normalize back to 0-1 range
    sharpened = np.clip(sharpened, 0, 1)
    
    return sharpened

def process_image_to_pure_black_white(output_image, threshold=0.3, detail_factor=1.5):
    """Process the image to have pure black background and pure white patterns with enhanced details"""
    # Convert to grayscale by taking mean across channels
    if len(output_image.shape) == 3 and output_image.shape[2] == 3:
        grayscale = np.mean(output_image, axis=2)
    else:
        grayscale = output_image
    
    # Enhance details
    grayscale_enhanced = enhance_details(grayscale, detail_factor)
    
    # Apply threshold to make pure black and white
    binary = np.zeros_like(grayscale_enhanced)
    binary[grayscale_enhanced > threshold] = 1.0  # Pure white
    
    # Convert to PIL image for direct saving (pure black and white)
    return Image.fromarray((binary * 255).astype(np.uint8))

def predict(model, image_path, device, threshold=0.3, detail_factor=1.5):
    """Generate a pixelILT image from the input target image using the ConViT model"""
    # Set up image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate output
    with torch.no_grad():
        output = model(image_tensor)
    
    # Denormalize output
    denorm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    output = denorm(output.squeeze(0).cpu())
    output = torch.clamp(output, 0, 1).permute(1, 2, 0).numpy()
    
    # Process to pure black and white with enhanced details
    bw_image = process_image_to_pure_black_white(output, threshold, detail_factor)
    
    return bw_image

def process_all_images(model_path, target_dir, output_dir, threshold=0.3, detail_factor=1.5, device=None):
    """Process all images in the target directory and save results in output directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    if device is None:
        model, device = load_model(model_path)
    else:
        model, _ = load_model(model_path)
    
    # Get all image files
    image_files = glob.glob(os.path.join(target_dir, "*.png"))
    
    print(f"Found {len(image_files)} images to process")
    print(f"Using threshold: {threshold}, detail enhancement factor: {detail_factor}")
    
    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Get filename without extension
            filename = os.path.basename(image_path)
            base_filename = os.path.splitext(filename)[0]
            
            # Process image
            output_image = predict(model, image_path, device, threshold, detail_factor)
            
            # Save processed image directly as pure black and white
            output_path = os.path.join(output_dir, f"{base_filename}_pixelILT_convit.png")
            output_image.save(output_path)
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    print(f"All images processed and saved to {output_dir}")

def main():
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.abspath(os.path.join(_here, ".."))
    _default_model = os.path.join(_root, "checkpoints", "best_convit_translator.pth")
    _default_targets = os.path.join(_root, "data", "sample_targets")
    _default_out = os.path.join(_root, "results", "convit_fp32")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate pixelILT images using a trained ConViT model")
    parser.add_argument('--model', type=str, default=_default_model, help='Path to the trained model')
    parser.add_argument('--image', type=str, help='Path to a single input target image')
    parser.add_argument('--target-dir', type=str, default=_default_targets, help='Directory containing target images to process')
    parser.add_argument('--output-dir', type=str, default=_default_out, help='Directory to save output pixelILT images')
    parser.add_argument('--threshold', type=float, default=0.3, help='Threshold for black/white conversion (0.0-1.0)')
    parser.add_argument('--detail', type=float, default=2.0, help='Detail enhancement factor (higher = more detail)')
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    if not os.path.exists(args.model):
        print(f"Warning: Model file {args.model} not found. Training may not have completed.")
        model_dir = os.path.dirname(args.model)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
    
    # Process a single image if specified
    if args.image:
        # Load the model
        model, device = load_model(args.model)
        
        # Generate pixelILT image
        output_image = predict(model, args.image, device, args.threshold, args.detail)
        
        # Create output directory if needed
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save the output
        filename = os.path.basename(args.image)
        base_filename = os.path.splitext(filename)[0]
        output_path = os.path.join(args.output_dir, f"{base_filename}_pixelILT_convit.png")
        output_image.save(output_path)
        print(f"Output saved to {output_path}")
    
    else:
        # Process all images in the target directory
        process_all_images(args.model, args.target_dir, args.output_dir, args.threshold, args.detail)

if __name__ == "__main__":
    main()