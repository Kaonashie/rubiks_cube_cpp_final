#pragma once
#include "opencv2/opencv.hpp"
#include <vector>
#include <string>
#include <array>
#include <functional>

class ArduinoStyleDetection {
private:
    // Reference RGB colors for each face (U, R, F, D, L, B)
    // These should be calibrated for your specific setup
    int reference_colors[6][3] = {
        {255, 255, 255}, // U (White) - face 0
        {255, 0, 0},     // R (Red) - face 1  
        {0, 255, 0},     // F (Green) - face 2
        {255, 165, 0},   // D (Orange) - face 3
        {0, 0, 255},     // L (Blue) - face 4
        {255, 255, 0}    // B (Yellow) - face 5
    };
    
    // Face assignments (characters for each color)
    char face_assign[6] = {'U', 'R', 'F', 'D', 'L', 'B'};
    
    // Hard-to-detect facet indices (corners that are difficult to see)
    std::vector<int> hard_facets = {0, 2, 6, 8, 9, 11, 15, 17, 18, 20, 24, 26, 27, 29, 33, 35, 36, 38, 42, 44, 45, 47, 51, 53};
    
    // Corner assignment lookup table [primary_color][secondary_color] -> third_color
    // Based on cube geometry - if you know 2 colors of a corner, you can deduce the 3rd
    char corner_assign[6][6] = {
        //   U  R  F  D  L  B
        {'?','D','L','R','F','?'}, // U
        {'D','?','U','F','?','L'}, // R  
        {'L','U','?','?','D','R'}, // F
        {'R','F','?','?','B','U'}, // D
        {'F','?','D','B','?','U'}, // L
        {'?','L','R','U','?','?'}  // B
    };
    
    std::vector<cv::Point> camera_1_points;
    std::vector<cv::Point> camera_2_points;
    
public:
    ArduinoStyleDetection();
    ~ArduinoStyleDetection();
    
    // Load position calibration files
    bool loadPositions(const std::string& pos1_file, const std::string& pos2_file);
    
    // Load color calibration (RGB reference colors)
    bool loadColorCalibration(const std::string& color_file);
    
    // Main detection function
    int detectCube(cv::Mat& frame1, cv::Mat& frame2, std::array<char, 54>& cube_state);
    
    // Helper functions
    int calculateColorDistance(int r, int g, int b, int face_index);
    int findClosestColor(int r, int g, int b);
    char applyColorDisambiguation(int min_color, int r, int g, int b);
    bool isHardFacet(int facet_index);
    char inferCornerColor(int facet_index, const std::array<char, 54>& cube_state);
    bool validateCubeConfiguration(const std::array<char, 54>& cube_state);
    
    // Calibration helpers
    typedef std::function<bool(cv::Mat&, cv::Mat&)> FrameCaptureCallback;
    void calibrateColors(FrameCaptureCallback captureFrames);
    void printCubeState(const std::array<char, 54>& cube_state);
};