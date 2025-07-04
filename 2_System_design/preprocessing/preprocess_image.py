# preprocess_image.py
# This script preprocesses a drone or satellite image for the Lendurai neural network system.
# It resizes the image to 224x224 pixels, normalizes pixel values to [0,1], and converts to CHW format (channels, height, width)
# for PyTorch model input. The script supports PNG, JPEG, and TIFF formats, compatible with the C++ program.

import numpy as np
from PIL import Image
import argparse

def preprocess_image(image_path, output_path=None):
    """
    Preprocess an input image:
    - Resize to 224x224 pixels
    - Normalize pixel values to [0,1]
    - Convert to CHW format (channels, height, width)
    Args:
        image_path (str): Path to input image (PNG, JPEG, or TIFF)
        output_path (str, optional): Path to save preprocessed image (as numpy array, .npy)
    Returns:
        np.ndarray: Preprocessed image in CHW format (shape: [3, 224, 224])
    """
    try:
        # Load image using PIL
        image = Image.open(image_path).convert("RGB")  # Ensure RGB format
        print(f"Loaded image: {image_path}, original size: {image.size}")

        # Resize to 224x224
        image = image.resize((224, 224), Image.LANCZOS)  # High-quality resampling
        print("Resized to 224x224 pixels")

        # Convert to numpy array and normalize to [0,1]
        image_array = np.array(image, dtype=np.float32) / 255.0
        print("Normalized pixel values to [0,1]")

        # Convert to CHW format (from HWC)
        image_array = image_array.transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]
        print("Converted to CHW format, shape:", image_array.shape)

        # Optionally save preprocessed image as numpy array
        if output_path:
            np.save(output_path, image_array)
            print(f"Saved preprocessed image to {output_path}")

        return image_array

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Preprocess drone/satellite image for neural network input")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (PNG, JPEG, TIFF)")
    parser.add_argument("--output", type=str, help="Path to save preprocessed image (.npy, optional)")
    args = parser.parse_args()

    # Preprocess the image
    preprocessed_image = preprocess_image(args.image, args.output)
    if preprocessed_image is not None:
        print("Preprocessing complete. Ready for neural network input.")
    else:
        print("Preprocessing failed. Check the input image and try again.")