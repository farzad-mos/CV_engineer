cmake_minimum_required(VERSION 3.10)
project(DroneCameraSim)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(drone_camera_sim main.cpp)

# Link OpenCV libraries
target_link_libraries(drone_camera_sim ${OpenCV_LIBS})

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)