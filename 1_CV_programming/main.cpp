// main.cpp
// Drone Camera Image Simulator for Lendurai Job Assignment
// This program takes a satellite image and parameters (x, y, altitude, tilt) to generate a simulated drone camera image.
// It uses OpenCV for image processing and applies a perspective transformation to mimic the drone's view.
// Bonus features: seasonal change (winter) and thermal camera simulation using simple color adjustments.
// The code is designed to be simple, well-documented, and runnable on Ubuntu 24.04.

// Report:
// - Approach: We use OpenCV to load a satellite image and apply a perspective transformation based on a pinhole camera model.
//   The x, y coordinates (normalized [0,1]) are mapped to pixel coordinates, altitude determines the field of view size,
//   and tilt_degrees adjusts the perspective. The output is saved as a PNG file.
// - Camera Model: Pinhole camera with 60-degree FOV. Field of view size is calculated as 2 * altitude * tan(FOV/2).
// - Assumptions: Satellite image resolution is 0.5 m/pixel, output image is 640x480 pixels, FOV is 60 degrees.
// - Bonus Features:
//   - Seasonal Change: Adjusts HSV colors to mimic winter (green to white).
//   - Thermal Camera: Converts to grayscale with a jet colormap for thermal effect.
//   - Satellite Image Acquisition: Not implemented due to complexity of API integration; uses local image as fallback.
// - Testing: Tested with sample PNG images (urban, rural) for various x, y, altitude, and tilt values.
// - Edge Cases: Handles invalid inputs (e.g., x, y outside [0,1], negative altitude) with error messages.

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <string>

using namespace cv;
using namespace std;

// Function to validate input parameters
bool validateInputs(float x, float y, float altitude_m, float tilt_degrees) {
    if (x < 0.0f || x > 1.0f || y < 0.0f || y > 1.0f) {
        cout << "Error: x and y must be in [0,1]." << endl;
        return false;
    }
    if (altitude_m <= 0.0f) {
        cout << "Error: altitude_m must be positive." << endl;
        return false;
    }
    if (tilt_degrees < -90.0f || tilt_degrees > 90.0f) {
        cout << "Error: tilt_degrees must be in [-90, 90]." << endl;
        return false;
    }
    return true;
}

// Function to apply seasonal change (winter effect)
Mat applyWinterEffect(const Mat& image) {
    Mat hsv_image;
    cvtColor(image, hsv_image, COLOR_BGR2HSV);
    for (int i = 0; i < hsv_image.rows; i++) {
        for (int j = 0; j < hsv_image.cols; j++) {
            Vec3b& pixel = hsv_image.at<Vec3b>(i, j);
            // Increase value (brightness) to mimic snow for green pixels (hue ~60-120)
            if (pixel[0] >= 30 && pixel[0] <= 60) {
                pixel[2] = min(pixel[2] + 50, 255); // Brighten
                pixel[1] = max(pixel[1] - 50, 0);   // Reduce saturation
            }
        }
    }
    Mat result;
    cvtColor(hsv_image, result, COLOR_HSV2BGR);
    return result;
}

// Function to apply thermal camera effect (jet colormap)
Mat applyThermalEffect(const Mat& image) {
    Mat gray_image, thermal_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);
    applyColorMap(gray_image, thermal_image, COLORMAP_JET);
    return thermal_image;
}

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    string image_path, output_path;
    float x = 0.5f, y = 0.5f, altitude_m = 100.0f, tilt_degrees = 0.0f;
    bool winter = false, thermal = false;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--image" && i + 1 < argc) image_path = argv[++i];
        else if (arg == "--x" && i + 1 < argc) x = stof(argv[++i]);
        else if (arg == "--y" && i + 1 < argc) y = stof(argv[++i]);
        else if (arg == "--altitude" && i + 1 < argc) altitude_m = stof(argv[++i]);
        else if (arg == "--tilt" && i + 1 < argc) tilt_degrees = stof(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--winter") winter = true;
        else if (arg == "--thermal") thermal = true;
        else {
            cout << "Unknown or incomplete argument: " << arg << endl;
            return 1;
        }
    }

    // Validate inputs
    if (image_path.empty() || output_path.empty()) {
        cout << "Usage: ./drone_camera_sim --image <path> --x <x_coord> --y <y_coord> "
             << "--altitude <altitude_m> --tilt <tilt_degrees> --output <output.png> "
             << "[--winter] [--thermal]" << endl;
        return 1;
    }
    if (!validateInputs(x, y, altitude_m, tilt_degrees)) {
        return 1;
    }

    // Load satellite image
    Mat image = imread(image_path, IMREAD_COLOR);
    if (image.empty()) {
        cout << "Error: Could not load image at " << image_path << endl;
        return 1;
    }

    // Assumptions
    const float resolution_m_per_pixel = 0.5f; // Satellite image resolution: 0.5 m/pixel
    const float fov_degrees = 60.0f;          // Camera field of view
    const int output_width = 640;             // Output image dimensions
    const int output_height = 480;

    // Map normalized coordinates to pixel coordinates
    float center_x = x * image.cols;
    float center_y = y * image.rows;

    // Calculate field of view size in meters
    float fov_size_m = 2.0f * altitude_m * tan((fov_degrees * CV_PI / 180.0f) / 2.0f);
    float fov_size_pixels = fov_size_m / resolution_m_per_pixel;

    // Define source points (a rectangle in the satellite image)
    float half_fov_pixels = fov_size_pixels / 2.0f;
    vector<Point2f> src_points = {
        Point2f(center_x - half_fov_pixels, center_y - half_fov_pixels),
        Point2f(center_x + half_fov_pixels, center_y - half_fov_pixels),
        Point2f(center_x + half_fov_pixels, center_y + half_fov_pixels),
        Point2f(center_x - half_fov_pixels, center_y + half_fov_pixels)
    };

    // Define destination points (output image with perspective tilt)
    float tilt_rad = tilt_degrees * CV_PI / 180.0f;
    float perspective_factor = cos(tilt_rad); // Simulates perspective by compressing top
    vector<Point2f> dst_points = {
        Point2f(0, 0),
        Point2f(output_width, 0),
        Point2f(output_width, output_height * perspective_factor),
        Point2f(0, output_height * perspective_factor)
    };

    // Compute homography and warp image
    Mat homography = findHomography(src_points, dst_points);
    Mat output_image;
    warpPerspective(image, output_image, homography, Size(output_width, output_height));

    // Apply bonus effects if requested
    if (winter) {
        output_image = applyWinterEffect(output_image);
    }
    if (thermal) {
        output_image = applyThermalEffect(output_image);
    }

    // Save output image
    if (!imwrite(output_path, output_image)) {
        cout << "Error: Could not save output image to " << output_path << endl;
        return 1;
    }

    cout << "Output image saved to " << output_path << endl;
    return 0;
}
