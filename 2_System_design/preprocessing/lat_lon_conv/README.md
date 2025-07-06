Python script that converts latitude and longitude coordinates to normalized values in the range [0,1]
---

### Manual

1. **Understanding the Task**:
   - The developed code for the assignments work with normalized coordinates (x, y in [0,1]) to represent positions in a satellite image tile (1 km², 2000x2000 pixels at 0.5 m/pixel).
   - A Python script that takes latitude and longitude (in degrees) and converts them to these normalized values, assuming the tile has known geographical bounds (e.g., min/max latitude and longitude).
   
2. **Planning the Solution**:
   - The normalizion of coordeinates has been done like scaling numbers to fit between 0 and 1. For example, if latitude ranges from 40.0 to 40.01 degrees, a latitude of 40.005 would be normalized to (40.005 - 40.0) / (40.01 - 40.0) = 0.5.
   - The code has been written using NumPy for math operations and clipping values to [0,1].
   - The script needed to take inputs (latitude, longitude, and the tile’s bounds) via the command line for flexibility, and it should print the results clearly.
   - I planned to add error checks to make sure the coordinates are valid and within the tile’s bounds.

3. **Writing the Code**:
   - **Step 1: Setting Up Inputs**: I used the `argparse` library to let users enter latitude, longitude, and the tile’s bounds (min/max latitude and longitude) from the command line. This makes the script easy to use, like `./convert_latlon_to_normalized.py --lat 40.005 --lon -74.005`.
   - **Step 2: Normalization Formula**:
     - For longitude (x), I used: `(lon - lon_min) / (lon_max - lon_min)` to get a value between 0 and 1.
     - For latitude (y), I used: `(lat - lat_min) / (lat_max - lat_min)`.
     - These formulas map the coordinates to the tile’s width and height, assuming the tile is a rectangle in geographical space.
   - **Step 3: Error Handling**: I added checks to ensure the input coordinates are within the tile’s bounds (e.g., `lat_min <= lat <= lat_max`). If not, the script shows an error message.
   - **Step 4: Clipping Values**: I used NumPy’s `clip` function to make sure the normalized values stay between 0 and 1, even if there’s a small math error.
   - **Step 5: Output**: The script prints the normalized x, y values and returns them for use in other programs (e.g., the neural network or C++ code).
   - **Step 6: Testing**: I tested with sample coordinates (e.g., lat=40.005, lon=-74.005 with bounds lat_min=40.0, lat_max=40.01, lon_min=-74.01, lon_max=-74.0) to ensure the output is correct (e.g., x=0.5, y=0.5 for the center).
