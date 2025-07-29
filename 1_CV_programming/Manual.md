# Drone Camera Image Simulator

This program generates a simulated drone camera image from a satellite image. It takes a satellite image and parameters (x, y, altitude, tilt) to produce an output image mimicking a drone's view. It supports optional winter and thermal effects.

## Prerequisites
- Ubuntu 24.04
- OpenCV 4.x
- CMake 3.10+
- A satellite image (PNG, JPEG, or TIFF, at least 1 km²)

## Installation
1. Install dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install libopencv-dev cmake
   ```
2. Build the program:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

## Usage
Run the program with:
```bash
./drone_camera_sim --image <path_to_image> --x <x_coord> --y <y_coord> --altitude <altitude_m> --tilt <tilt_degrees> --output <output.png> [--winter] [--thermal]
```
Example:
```bash
./drone_camera_sim --image satellite.png --x 0.5 --y 0.5 --altitude 100 --tilt 30 --output output.png --winter
```

## Parameters
- `--image`: Path to satellite image (PNG/JPEG/TIFF).
- `--x`, `--y`: Normalized coordinates [0,1] (0.5, 0.5 is image center).
- `--altitude`: Altitude in meters (>0).
- `--tilt`: Camera tilt angle in degrees [-90, 90].
- `--output`: Output image path (PNG).
- `--winter`: Apply winter effect (optional).
- `--thermal`: Apply thermal camera effect (optional).

## Notes
- The program assumes the satellite image has a resolution of 0.5 m/pixel.
- Output image size is 640x480 pixels.
- The `--winter` and `--thermal` options cannot be used together in this version.
- The program does not fetch images from an API; use a local image.

## Testing
Tested with sample satellite images (urban, rural) using various parameters:
- x, y: 0.5, 0.5 (center), 0.1, 0.9 (edges)
- Altitude: 50m, 100m, 500m
- Tilt: 0°, 30°, 60°
