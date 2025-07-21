# Rubik's Cube Computer Vision Detection System

A high-performance dual-camera computer vision system for real-time Rubik's cube state detection and analysis. This system uses advanced HSV color space processing and optimized lookup tables to achieve fast, accurate cube state recognition.

## Overview

This project implements a sophisticated computer vision pipeline that captures and processes Rubik's cube states using two strategically positioned cameras. The system is designed for real-time detection with minimal latency, making it suitable for speedcubing analysis, automatic solving systems, and educational demonstrations.

### Key Features

- **Dual-Camera Architecture**: Simultaneous capture from two cameras for complete cube visibility
- **Real-Time Processing**: Optimized algorithms for sub-100ms detection times
- **Interactive Calibration**: User-friendly position and color calibration interfaces
- **Robust Color Detection**: HSV-based color space analysis with lighting compensation
- **Cross-Platform Support**: Compatible with Linux and Windows systems
- **Flexible Configuration**: Customizable camera settings and detection parameters

## System Requirements

### Hardware Requirements
- **Cameras**: 2x USB cameras 
- **USB**: Multiple USB 2.0+ ports for camera connections. Bottleneck possible when using a single port.

### Software Requirements
- **Operating System**: Linux (Ubuntu 18.04+) or Windows 10+
- **CMake**: Version 3.31 or higher
- **C++ Compiler**: C++20 compatible compiler (GCC 10+, Clang 10+, MSVC 2019+)
- **OpenCV**: Version 4.0 or higher
- **V4L2**: Linux camera drivers (Linux only)

## Quick Start

### Automated Build (Recommended)

Use the provided build script for automatic dependency installation and compilation:

```bash
# Linux
./build.sh

# Windows
build.bat
```

### Manual Build

If you prefer manual compilation or need custom configuration:

```bash
# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake ..

# Compile
make

# Run the application
./rubiks_cube_cpp_final
```

## Camera Setup

### Physical Positioning

Position your cameras to capture complementary views of the cube:

- **Camera 1**: Front, Right, and Top faces
- **Camera 2**: Back, Left, and Bottom faces

Ensure both cameras have clear, unobstructed views of their respective cube faces with consistent lighting.


## Calibration Process

The system requires two calibration steps for optimal performance:

### 1. Position Calibration

Calibrate camera positions to map facelet detection points:

```bash
# Run the application and select option 'c'
./rubiks_cube_cpp_final
> c
```

**Visual Calibration Demo**
<!-- Placeholder for position calibration GIF/video -->
*[Demo of interactive position calibration process will be added here]*

- Click on each facelet center in the displayed order
- Visual feedback shows detected points in real-time
- Calibration data saved to `pos_1.txt` and `pos_2.txt`

### 2. Color Calibration

Fine-tune HSV color ranges for accurate detection:

```bash
# Run the application and select option 'k'
./rubiks_cube_cpp_final
> k
```

- Interactive trackbars for HSV range adjustment
- Real-time color detection preview
- Press 's' to save each color, 'r' to reset
- Calibration data saved to `range.txt`

## Detection Modes

### Full Detection Mode

Complete cube state detection with performance metrics:

```bash
# Select option 'j' from the main menu
> j
```

**Detection Demo**
<!-- Placeholder for detection process GIF/video -->
*[Demo of real-time cube detection will be added here]*

Features:
- Parallel processing for maximum speed
- Real-time cube state visualization
- Performance metrics and timing data
- Automatic cube validation

### Benchmark Mode

Test detection performance with standard settings:

```bash
# Select option 'b' from the main menu
> b
```

### Dual Camera View

Live preview from both cameras for setup verification:

```bash
# Select option 'd' from the main menu
> d
```

- Side-by-side camera feeds
- 3x3 grid overlay for alignment assistance
- Snapshot capture capability (SPACE key)
- Exit with ESC or Q

## Configuration Files

The system generates and uses several configuration files:

- **`pos_1.txt`** / **`pos_2.txt`**: Camera facelet coordinates (27 points each)
- **`range.txt`**: Custom HSV color ranges for each cube color
- **`config.txt`**: Additional system configuration parameters

## Architecture Overview

### Core Components

- **`main.cpp`**: Main application logic and detection algorithms
- **`PS3EyeCamera.h/.cpp`**: Camera abstraction and configuration
- **`arduino_detection.h/.cpp`**: Additional detection utilities

### Data Structures

- **Color LUT**: 3D lookup table `[180][256][256]` for O(1) HSV-to-color mapping
- **Camera Points**: Pre-calibrated pixel coordinates for facelet sampling
- **Cube State**: 6-face × 9-sticker representation of cube configuration

### Performance Optimizations

- Lookup table approach eliminates conditional branching
- Parallel processing with `std::thread` for dual-camera capture
- Point-based sampling avoids expensive blob detection
- HSV color space provides lighting invariance

## Troubleshooting

### Common Issues

**Camera Not Detected**
- Verify camera connections and USB port functionality
- Check V4L2 drivers on Linux systems
- Ensure cameras are not in use by other applications

**Poor Color Detection**
- Recalibrate color ranges in varying lighting conditions
- Ensure consistent, diffused lighting setup
- Check for color bleeding between adjacent facelets

**Performance Issues**
- Close unnecessary applications to free system resources
- Verify camera resolution and color settings 
- Check CPU usage during parallel processing

### Debug Mode

Enable verbose output for troubleshooting:

```bash
# Compile with debug flags
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

## Development

### Building from Source

```bash
# Clone the repository
git clone [repository-url]
cd rubiks_cube_cpp_final

# Install dependencies (see build script)
./build.sh --deps-only

# Manual compilation
mkdir build && cd build
cmake ..
make
```

### Code Structure

The codebase follows a modular architecture with clear separation of concerns:

- Camera management and configuration
- Computer vision processing pipeline
- User interface and calibration tools
- Performance optimization and threading

### Contributing

1. Follow C++20 coding standards
2. Maintain compatibility with OpenCV 4.0+
3. Test on both Linux and Windows platforms
4. Document new features and API changes

## Acknowledgments

- OpenCV community for computer vision libraries
- V4L2 project for Linux camera support
- Contributors to PS3 Eye camera drivers

---

For technical support or feature requests, please refer to the project documentation or submit an issue through the project repository.