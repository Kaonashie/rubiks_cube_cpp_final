#!/bin/bash

# Rubik's Cube Detection System - Universal Build Script for Linux
# This script automatically installs dependencies and builds the project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
DEPS_ONLY=false
CLEAN_BUILD=false
VERBOSE=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    elif [ -f /etc/redhat-release ]; then
        echo "rhel"
    elif [ -f /etc/debian_version ]; then
        echo "debian"
    else
        echo "unknown"
    fi
}

# Function to install dependencies based on distribution
install_dependencies() {
    local distro=$(detect_distro)
    print_status "Detected distribution: $distro"
    
    case "$distro" in
        "ubuntu"|"debian"|"pop"|"mint")
            print_status "Installing dependencies for Debian/Ubuntu-based system..."
            sudo apt update
            sudo apt install -y \
                build-essential \
                cmake \
                pkg-config \
                libopencv-dev \
                libopencv-contrib-dev \
                libv4l-dev \
                v4l-utils \
                git \
                wget
            ;;
        "fedora"|"centos"|"rhel"|"rocky"|"almalinux")
            print_status "Installing dependencies for Red Hat-based system..."
            if command -v dnf &> /dev/null; then
                PKG_MANAGER="dnf"
            else
                PKG_MANAGER="yum"
            fi
            
            sudo $PKG_MANAGER groupinstall -y "Development Tools"
            sudo $PKG_MANAGER install -y \
                cmake \
                pkg-config \
                opencv-devel \
                opencv-contrib-devel \
                libv4l-devel \
                v4l-utils \
                git \
                wget
            ;;
        "arch"|"manjaro"|"endeavouros")
            print_status "Installing dependencies for Arch-based system..."
            sudo pacman -Syu --needed --noconfirm \
                base-devel \
                cmake \
                pkg-config \
                opencv \
                opencv-samples \
                v4l-utils \
                git \
                wget
            ;;
        "opensuse"|"sles")
            print_status "Installing dependencies for openSUSE/SLES..."
            sudo zypper refresh
            sudo zypper install -y \
                -t pattern devel_basis \
                cmake \
                pkg-config \
                opencv-devel \
                libv4l-devel \
                v4l-utils \
                git \
                wget
            ;;
        *)
            print_warning "Unknown distribution. Please install the following packages manually:"
            echo "  - build-essential/development tools"
            echo "  - cmake (3.31+)"
            echo "  - pkg-config"
            echo "  - opencv-dev (4.0+)"
            echo "  - libv4l-dev"
            echo "  - v4l-utils"
            echo "  - git"
            read -p "Have you installed the required dependencies? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_error "Dependencies not confirmed. Exiting."
                exit 1
            fi
            ;;
    esac
}

# Function to check if dependencies are available
check_dependencies() {
    print_status "Checking dependencies..."
    
    local missing_deps=()
    
    # Check for essential build tools
    if ! command -v cmake &> /dev/null; then
        missing_deps+=("cmake")
    fi
    
    if ! command -v make &> /dev/null && ! command -v ninja &> /dev/null; then
        missing_deps+=("make or ninja")
    fi
    
    if ! command -v pkg-config &> /dev/null; then
        missing_deps+=("pkg-config")
    fi
    
    # Check for OpenCV
    if ! pkg-config --exists opencv4 && ! pkg-config --exists opencv; then
        missing_deps+=("opencv")
    fi
    
    # Check for V4L2
    if [ ! -f /usr/include/linux/videodev2.h ] && [ ! -f /usr/include/libv4l2.h ]; then
        missing_deps+=("v4l2")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_status "Run '$0 --install-deps' to install dependencies automatically"
        return 1
    fi
    
    print_success "All dependencies found"
    return 0
}

# Function to check CMake version
check_cmake_version() {
    local cmake_version=$(cmake --version | head -n1 | cut -d' ' -f3)
    local required_version="3.31"
    
    if ! printf '%s\n%s\n' "$required_version" "$cmake_version" | sort -V -C; then
        print_warning "CMake version $cmake_version is older than required $required_version"
        print_status "The build might still work, but consider upgrading CMake"
    else
        print_success "CMake version $cmake_version meets requirements"
    fi
}

# Function to build the project
build_project() {
    print_status "Starting build process..."
    
    # Create build directory
    if [ "$CLEAN_BUILD" = true ] && [ -d "$BUILD_DIR" ]; then
        print_status "Cleaning existing build directory..."
        rm -rf "$BUILD_DIR"
    fi
    
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Configure with CMake
    print_status "Configuring project with CMake..."
    local cmake_args=("-DCMAKE_BUILD_TYPE=Release")
    
    if [ "$VERBOSE" = true ]; then
        cmake_args+=("-DCMAKE_VERBOSE_MAKEFILE=ON")
    fi
    
    if ! cmake "${cmake_args[@]}" ..; then
        print_error "CMake configuration failed"
        exit 1
    fi
    
    # Build the project
    print_status "Building project..."
    local build_args=()
    
    if [ "$VERBOSE" = true ]; then
        build_args+=("--verbose")
    fi
    
    # Use all available cores for faster compilation
    local num_cores=$(nproc 2>/dev/null || echo 4)
    build_args+=("--parallel" "$num_cores")
    
    if ! cmake --build . "${build_args[@]}"; then
        print_error "Build failed"
        exit 1
    fi
    
    print_success "Build completed successfully!"
    print_status "Executable location: $BUILD_DIR/rubiks_cube_cpp_final"
}

# Function to verify the build
verify_build() {
    local executable="$BUILD_DIR/rubiks_cube_cpp_final"
    
    if [ -f "$executable" ]; then
        print_success "Build verification passed"
        print_status "You can now run: $executable"
        
        # Check if cameras are available
        if command -v v4l2-ctl &> /dev/null; then
            local cameras=$(v4l2-ctl --list-devices 2>/dev/null | grep -c "/dev/video" || echo "0")
            if [ "$cameras" -gt 0 ]; then
                print_success "Found $cameras camera device(s)"
            else
                print_warning "No camera devices found. Make sure your cameras are connected."
            fi
        fi
        
        return 0
    else
        print_error "Build verification failed - executable not found"
        return 1
    fi
}

# Function to show usage
show_usage() {
    echo "Rubik's Cube Detection System - Build Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --install-deps    Install system dependencies automatically"
    echo "  --deps-only       Only install dependencies, don't build"
    echo "  --clean           Clean build directory before building"
    echo "  --verbose         Enable verbose build output"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                 # Build project (dependencies must be installed)"
    echo "  $0 --install-deps  # Install dependencies and build"
    echo "  $0 --clean         # Clean build and rebuild"
    echo "  $0 --deps-only     # Only install dependencies"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --install-deps)
            INSTALL_DEPS=true
            shift
            ;;
        --deps-only)
            DEPS_ONLY=true
            INSTALL_DEPS=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
print_status "Rubik's Cube Detection System - Build Script"
print_status "============================================="

# Install dependencies if requested
if [ "$INSTALL_DEPS" = true ]; then
    install_dependencies
    if [ "$DEPS_ONLY" = true ]; then
        print_success "Dependencies installed successfully!"
        exit 0
    fi
fi

# Check dependencies
if ! check_dependencies; then
    exit 1
fi

# Check CMake version
check_cmake_version

# Build the project
build_project

# Verify the build
if verify_build; then
    print_success "Build process completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Connect your two cameras"
    echo "2. Run: $BUILD_DIR/rubiks_cube_cpp_final"
    echo "3. Follow the calibration process in the application menu"
    echo ""
    echo "For more information, see README.md"
else
    exit 1
fi