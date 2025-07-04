# Manual for Running the Drone Image Matching System

This manual provides instructions on how to run the `image_matching_system.py` program, which matches a drone camera image (640x480 pixels) to one of 100+ satellite image tiles (2000x2000 pixels, 0.5 m/pixel) and predicts coordinates within that tile. The system is designed for real-time inference on a Raspberry Pi 5 (RPi5) using a MobileNetV3-based neural network. The instructions are beginner-friendly and include setup, execution, and deployment steps for both Ubuntu 24.04 and RPi5.

## Prerequisites

### Software
- **Python 3.8+**: Required to run the PyTorch script.
- **PyTorch and torchvision**: For the neural network (MobileNetV3-Small) and image processing.
  - Install with: `pip3 install torch torchvision`
  - For RPi5, use a compatible PyTorch version (e.g., `torch==2.0.0`).
- **NumPy**: For handling synthetic data and arrays.
  - Install with: `pip3 install numpy`
- **ONNX Runtime** (optional, for RPi5 deployment): For optimized inference.
  - Install with: `pip3 install onnxruntime`
- **Operating System**:
  - Ubuntu 24.04 (for development and testing).
  - Raspberry Pi OS (64-bit, based on Debian) for RPi5 deployment.

### Hardware
- **Development Machine**: Any system running Ubuntu 24.04 (or similar) with at least 4GB RAM and a CPU for testing.
- **Raspberry Pi 5**: Quad-core Cortex-A76, 8GB RAM, for real-time inference.
  - Ensure a microSD card (16GB+), power supply, and internet connection.

### Data
- **Drone Image**: A 640x480 RGB image (e.g., output from the first assignment’s C++ program). For prototyping, the code uses a synthetic random image.
- **Satellite Tiles**: 100+ tiles (2000x2000 pixels, 0.5 m/pixel) with labels (tile ID, x, y coordinates). For this prototype, synthetic data (random tile ID and coordinates) is used since no real dataset is provided.
- **Optional**: A real dataset of paired drone images and satellite tiles for testing or training (not required for prototype).

### Installation Steps
1. **Set Up Python Environment**:
   - On Ubuntu 24.04:
     ```bash
     sudo apt-get update
     sudo apt-get install python3-pip
     pip3 install torch torchvision numpy onnxruntime
     ```
   - On RPi5 (Raspberry Pi OS):
     ```bash
     sudo apt-get update
     sudo apt-get install python3-pip
     pip3 install torch==2.0.0 torchvision==0.15.0 numpy onnxruntime
     ```
     Note: Check PyTorch’s official site for RPi5-compatible wheels if needed.

2. **Verify Dependencies**:
   - Check Python: `python3 --version`
   - Check PyTorch: `python3 -c "import torch; print(torch.__version__)"`
   - Ensure NumPy and ONNX Runtime are installed: `pip3 show numpy onnxruntime`

3. **Prepare Input Data**:
   - For testing, the code uses synthetic data (random 224x224 image, tile ID, coordinates).
   - For real use, prepare a 640x480 RGB drone image (e.g., `drone_image.png`) and preprocess it:
     - Resize to 224x224 pixels.
     - Normalize pixel values to [0,1].
     - Convert to CHW format (channels, height, width).

## Running the Program

1. **Save the Code**:
   - Save the provided script as `image_matching_system.py` in a folder (e.g., `drone_matching`).

2. **Run on Ubuntu 24.04 (Testing)**:
   - Navigate to the folder:
     ```bash
     cd drone_matching
     ```
   - Run the script to test with synthetic data:
     ```bash
     python3 image_matching_system.py
     ```
   - **Output**:
     - Model parameter count (e.g., ~2.5M parameters).
     - Sample inference results: `Predicted Tile ID: <number>, Coordinates: [x, y]`.
     - Overfitting results: Loss values over 100 epochs, final predicted tile ID (e.g., 42), and coordinates (e.g., [0.5, 0.5]).

3. **Run on RPi5 (Deployment)**:
   - **Export to ONNX** (do this on Ubuntu first):
     ```python
     import torch
     from image_matching_system import ImageMatchingNet
     model = ImageMatchingNet(num_tiles=100)
     model.eval()
     torch.onnx.export(model, torch.rand(1, 3, 224, 224), "model.onnx", input_names=["input"], output_names=["tile_logits", "coords"])
     ```
   - Copy `model.onnx` and the following script to RPi5:
     ```python
     import onnxruntime as ort
     import numpy as np
     session = ort.InferenceSession("model.onnx")
     input_image = np.random.rand(1, 3, 224, 224).astype(np.float32)  # Replace with real preprocessed image
     tile_logits, coords = session.run(None, {"input": input_image})
     pred_tile = np.argmax(tile_logits, axis=1)
     print(f"Predicted Tile ID: {pred_tile[0]}, Coordinates: {coords[0]}")
     ```
   - Save as `run_onnx.py` and run on RPi5:
     ```bash
     python3 run_onnx.py
     ```
   - **Output**: Predicted tile ID and coordinates for the input image.

4. **Using Real Drone Images**:
   - Preprocess the image (e.g., using Python with PIL):
     ```python
     from PIL import Image
     import numpy as np
     image = Image.open("drone_image.png").resize((224, 224))
     image_array = np.array(image).astype(np.float32) / 255.0  # Normalize to [0,1]
     image_array = image_array.transpose(2, 0, 1)  # Convert to CHW
     image_array = image_array[np.newaxis, :]  # Add batch dimension
     ```
   - Replace the synthetic input (`np.random.rand(1, 3, 224, 224)`) in `run_onnx.py` or `image_matching_system.py` with `image_array`.

## Troubleshooting
- **Module Not Found**: Ensure PyTorch, torchvision, NumPy, and ONNX Runtime are installed correctly. Check versions with `pip3 list`.
- **ONNX Export Fails**: Verify the model is in evaluation mode (`model.eval()`) and the input tensor shape is correct.
- **Slow Inference on RPi5**: Confirm ONNX Runtime is used. Test with a smaller input size (e.g., 224x224). Check latency with `time python3 run_onnx.py`.
- **No Dataset**: The code uses synthetic data for prototyping. For real use, replace with a dataset of drone images and labeled satellite tiles (tile ID, x, y coordinates).

## Notes
- The script uses synthetic data (random image, tile ID 42, coordinates [0.5, 0.5]) for testing since no real dataset is provided.
- Inference time on RPi5 should be <100ms with MobileNetV3-Small and ONNX optimization.
- The model is not trained; the overfitting loop demonstrates learning capability on one sample.
- For real-world use, train the model with a dataset of paired drone and satellite images.

For more details, see the inline comments in `image_matching_system.py`, which include a report on the architecture, loss function, and optimizations.