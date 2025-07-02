import h5py
import numpy as np
from PIL import Image
import os

# Function to extract metadata and save to a text file
def export_metadata(hdf_file, output_txt):
    with h5py.File(hdf_file, 'r') as f:
        with open(output_txt, 'w') as txt_file:
            txt_file.write("HDF Image Metadata\n")
            txt_file.write("=================\n\n")
            # File-level attributes
            for attr, value in f.attrs.items():
                txt_file.write(f"File Attribute: {attr} = {value}\n")
            # Dataset-level metadata
            for dataset_name in f.keys():
                txt_file.write(f"\nDataset: {dataset_name}\n")
                dataset = f[dataset_name]
                txt_file.write(f"  Shape: {dataset.shape}\n")
                txt_file.write(f"  Data Type: {dataset.dtype}\n")
                for attr, value in dataset.attrs.items():
                    txt_file.write(f"  Attribute: {attr} = {value}\n")
    print(f"Metadata saved to {output_txt}")

# Function to read HDF image, resize, and save as JPEG
def process_hdf_image(hdf_file, dataset_name, output_jpeg, resolution_m_per_pixel=0.5):
    try:
        # Open HDF5 file
        with h5py.File(hdf_file, 'r') as f:
            if dataset_name not in f:
                raise ValueError(f"Dataset '{dataset_name}' not found in HDF file")
            # Read the dataset (assuming it's an image array)
            data = f[dataset_name][:]
            
            # Handle different array shapes (2D for grayscale, 3D for RGB)
            if len(data.shape) == 2:
                # Grayscale image
                image_array = data
            elif len(data.shape) == 3 and data.shape[2] in [1, 3]:
                # RGB or single-channel 3D array
                image_array = data if data.shape[2] == 3 else data[:, :, 0]
            else:
                raise ValueError("Unsupported dataset shape. Expected 2D or 3D (H, W, 1 or 3).")

            # Ensure uint8 data type for image
            if image_array.dtype != np.uint8:
                # Normalize to 0-255 if needed
                image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8)

            # Create PIL image
            image = Image.fromarray(image_array)

            # Calculate target size for 1 km^2 (assuming resolution_m_per_pixel)
            target_pixels = int(1000 / resolution_m_per_pixel)  # 1 km = 1000 m
            target_size = (target_pixels, target_pixels)  # 2000x2000 for 0.5 m/pixel

            # Crop from top-right corner
            width, height = image.size
            left = width - target_pixels
            top = 0
            right = width
            bottom = target_pixels
            if left < 0 or bottom > height:
                raise ValueError("Image is too small to crop 1 km^2 from top-right")
            image_cropped = image.crop((left, top, right, bottom))

            # Resize to ensure exact 1 km^2 (in case of rounding errors)
            image_resized = image_cropped.resize(target_size, Image.LANCZOS)

            # Save as JPEG
            image_resized.save(output_jpeg, "JPEG")
            print(f"Image saved as {output_jpeg}")
            
    except Exception as e:
        print(f"Error processing HDF file: {e}")

if __name__ == "__main__":
    # Input parameters
    hdf_file = "satellite_image.hdf"  # Replace with your HDF file path
    dataset_name = "image"  # Replace with the actual dataset name in your HDF file
    output_txt = "metadata.txt"
    output_jpeg = "output_image.jpg"
    resolution_m_per_pixel = 0.5  # Assumed resolution (m/pixel)

    # Export metadata
    if os.path.exists(hdf_file):
        export_metadata(hdf_file, output_txt)
        process_hdf_image(hdf_file, dataset_name, output_jpeg, resolution_m_per_pixel)
    else:
        print(f"Error: HDF file '{hdf_file}' not found")