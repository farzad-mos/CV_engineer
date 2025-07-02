
1- Download satellite image on: [Dropbox](https://www.dropbox.com/scl/fi/go1o1o2ms2ukvx7yenrpw/MOD02QKM.A2024249.0835.061.2024249191823.hdf?rlkey=2jjhxvc9ybygni5nnkyn86spr&st=znb37106&dl=0)

2- run the script

----


This script:
1. Reads the HDF image using the `h5py` library (HDF5).
2. Exports metadata to a text file.
3. Resizes the image to cover 1 km² (from the top-right corner, assuming a known resolution).
4. Saves the image as a JPEG file.


### Explanation of the Code
- **Reading HDF Image**: Uses `h5py` to open the HDF5 file and read the specified dataset (e.g., "image"). Handles both 2D (grayscale) and 3D (RGB or single-channel) arrays. Normalizes data to uint8 for image compatibility if needed.
- **Exporting Metadata**: Extracts file and dataset attributes (e.g., shape, data type, resolution) and saves them to `metadata.txt`. This includes any available metadata like units or capture details, depending on the HDF file.
- **Resizing to 1 km²**: Assumes 0.5 m/pixel resolution, so 1 km² = 2000x2000 pixels. Crops from the top-right corner and resizes to ensure the exact size using PIL’s `resize` with LANCZOS for high-quality resampling.
- **Saving as JPEG**: Converts the processed image to JPEG format using PIL’s `save` method. Note that JPEG is lossy, but it’s compatible with the C++ program.
- **Error Handling**: Checks for file existence, valid dataset, and sufficient image size. Prints clear error messages for issues like missing files or unsupported shapes.

### Manual for Running the Code
1. **Prerequisites**:
   - **Ubuntu 24.04** (or any system with Python 3.x).
   - **Dependencies**: Install `h5py` and `Pillow`:
     ```bash
     pip install h5py Pillow
     ```
   - **Input HDF File**: Ensure you have an HDF5 file (e.g., `satellite_image.hdf`) with an image dataset (e.g., named "image"). If it’s HDF4, install `pyhdf` instead and modify the code (see note below).

2. **Setup**:
   - Save the Python script as `convert_hdf_to_jpeg.py`.
   - Update the script’s `hdf_file` variable with the path to your HDF file and `dataset_name` with the name of the image dataset (check using an HDF viewer like HDFView if unknown).

3. **Run the Script**:
   ```bash
   python3 convert_hdf_to_jpeg.py
   ```
   - **Output**:
     - `metadata.txt`: Contains metadata like dataset shape, data type, and attributes.
     - `output_image.jpg`: The resized (2000x2000 pixels) JPEG image cropped from the top-right.

4. **Using with C++ Program**:
   - Use the generated `output_image.jpg` as input for the C++ program:
     ```bash
     ./drone_camera_sim --image output_image.jpg --x 0.5 --y 0.5 --altitude 100 --tilt 30 --output drone_output.png
     ```

### Notes
- **HDF4 vs. HDF5**: If your file is HDF4 (common for older satellite data like MODIS), replace `h5py` with `pyhdf.SD`. Modify the reading logic:
  ```python
  from pyhdf.SD import SD, SDC
  hdf = SD(hdf_file, SDC.READ)
  data = hdf.select(dataset_name)[:]
  ```
  Install `pyhdf`:
  ```bash
  pip install pyhdf
  ```
  Update metadata extraction to use `hdf.attributes()` and `hdf.select(dataset_name).attributes()`.

- **Dataset Name**: You need to know the dataset name containing the image (e.g., "image"). Use tools like HDFView or `h5py` exploration (`print(list(f.keys()))`) to find it.
- **Resolution**: The script assumes 0.5 m/pixel. If your HDF file includes resolution metadata, adjust `resolution_m_per_pixel` accordingly.
- **Metadata**: The script extracts all available attributes. If your HDF file lacks metadata, the `metadata.txt` file will only include basic info like shape and data type.
- **JPEG Quality**: JPEG is lossy, so some detail may be lost. If you need lossless output, you could modify the script to save as PNG or TIFF instead.

### Example Output
- **metadata.txt**:
  ```
  HDF Image Metadata
  =================

  File Attribute: source = Satellite Data
  File Attribute: date = 2025-01-01

  Dataset: image
    Shape: (4000, 4000, 3)
    Data Type: float32
    Attribute: units = reflectance
    Attribute: resolution = 0.5 m/pixel
  ```
- **output_image.jpg**: A 2000x2000 pixel JPEG image covering 1 km² from the top-right of the original image.

