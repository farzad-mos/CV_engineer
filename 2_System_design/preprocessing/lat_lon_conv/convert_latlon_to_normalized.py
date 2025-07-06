# convert_latlon_to_normalized.py
# This script converts latitude and longitude coordinates to normalized [0,1] values
# relative to a satellite image tile's geographical bounding box (e.g., 1 km²).
# It is designed for the Lendurai Computer Vision Engineer assignment to prepare coordinates
# for the neural network or C++ program, which expect normalized x, y values.

import numpy as np
import argparse

def convert_latlon_to_normalized(lat, lon, lat_min, lat_max, lon_min, lon_max):
    """
    Convert latitude and longitude to normalized [0,1] coordinates.
    Args:
        lat (float): Latitude of the point (in degrees).
        lon (float): Longitude of the point (in degrees).
        lat_min, lat_max (float): Latitude bounds of the tile (in degrees).
        lon_min, lon_max (float): Longitude bounds of the tile (in degrees).
    Returns:
        tuple: Normalized (x, y) coordinates in [0,1], or None if invalid.
    """
    try:
        # Validate inputs
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            raise ValueError("Latitude or longitude is outside the tile's bounding box")

        # Normalize latitude to [0,1]: (lat - lat_min) / (lat_max - lat_min)
        norm_y = (lat - lat_min) / (lat_max - lat_min)
        # Normalize longitude to [0,1]: (lon - lon_min) / (lon_max - lon_min)
        norm_x = (lon - lon_min) / (lon_max - lon_min)

        # Ensure values are within [0,1]
        norm_x = np.clip(norm_x, 0.0, 1.0)
        norm_y = np.clip(norm_y, 0.0, 1.0)

        print(f"Input Lat: {lat}, Lon: {lon}")
        print(f"Normalized x: {norm_x:.6f}, y: {norm_y:.6f}")
        return norm_x, norm_y

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert latitude/longitude to normalized [0,1] coordinates")
    parser.add_argument("--lat", type=float, required=True, help="Latitude of the point (degrees)")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of the point (degrees)")
    parser.add_argument("--lat_min", type=float, default=0.0, help="Minimum latitude of the tile (degrees)")
    parser.add_argument("--lat_max", type=float, default=0.01, help="Maximum latitude of the tile (degrees)")
    parser.add_argument("--lon_min", type=float, default=0.0, help="Minimum longitude of the tile (degrees)")
    parser.add_argument("--lon_max", type=float, default=0.01, help="Maximum longitude of the tile (degrees)")
    args = parser.parse_args()

    # Convert coordinates
    result = convert_latlon_to_normalized(
        args.lat, args.lon, args.lat_min, args.lat_max, args.lon_min, args.lon_max
    )
    if result is not None:
        norm_x, norm_y = result
        print(f"Successfully converted to normalized coordinates: x={norm_x:.6f}, y={norm_y:.6f}")
    else:
        print("Conversion failed. Check input values and bounding box.")