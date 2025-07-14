#include "arduino_detection.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

ArduinoStyleDetection::ArduinoStyleDetection() {
    // Initialize with default values
}

ArduinoStyleDetection::~ArduinoStyleDetection() {
}

bool ArduinoStyleDetection::loadPositions(const std::string& pos1_file, const std::string& pos2_file) {
    // Load camera 1 positions
    std::ifstream file1(pos1_file);
    if (!file1.is_open()) {
        std::cerr << "Error: Could not open " << pos1_file << std::endl;
        return false;
    }
    
    camera_1_points.clear();
    int x, y;
    for (int i = 0; i < 27; i++) {
        if (file1 >> x >> y) {
            camera_1_points.emplace_back(x, y);
        } else {
            std::cerr << "Error: Not enough points in " << pos1_file << std::endl;
            return false;
        }
    }
    file1.close();
    
    // Load camera 2 positions
    std::ifstream file2(pos2_file);
    if (!file2.is_open()) {
        std::cerr << "Error: Could not open " << pos2_file << std::endl;
        return false;
    }
    
    camera_2_points.clear();
    for (int i = 0; i < 27; i++) {
        if (file2 >> x >> y) {
            camera_2_points.emplace_back(x, y);
        } else {
            std::cerr << "Error: Not enough points in " << pos2_file << std::endl;
            return false;
        }
    }
    file2.close();
    
    std::cout << "✓ Loaded positions: " << camera_1_points.size() << " points from camera 1, " 
              << camera_2_points.size() << " points from camera 2" << std::endl;
    return true;
}

bool ArduinoStyleDetection::loadColorCalibration(const std::string& color_file) {
    std::ifstream file(color_file);
    if (!file.is_open()) {
        std::cout << "Warning: Could not open " << color_file << ". Using default RGB values." << std::endl;
        return false;
    }
    
    // Expected format: W r g b, R r g b, O r g b, Y r g b, G r g b, B r g b
    std::string line;
    int color_index = 0;
    
    while (std::getline(file, line) && color_index < 6) {
        if (line.empty() || line[0] == '#') continue;
        
        std::stringstream ss(line);
        char color_char;
        int r, g, b;
        
        if (ss >> color_char >> r >> g >> b) {
            // Map color characters to indices
            int idx = -1;
            switch (color_char) {
                case 'W': idx = 0; break; // White (U)
                case 'R': idx = 1; break; // Red (R)
                case 'G': idx = 2; break; // Green (F)
                case 'O': idx = 3; break; // Orange (D)
                case 'B': idx = 4; break; // Blue (L)
                case 'Y': idx = 5; break; // Yellow (B)
            }
            
            if (idx >= 0) {
                reference_colors[idx][0] = r;
                reference_colors[idx][1] = g;
                reference_colors[idx][2] = b;
                std::cout << "Loaded " << color_char << ": RGB(" << r << "," << g << "," << b << ")" << std::endl;
            }
        }
    }
    
    file.close();
    return true;
}

int ArduinoStyleDetection::calculateColorDistance(int r, int g, int b, int face_index) {
    // Manhattan distance (same as Arduino code)
    return abs(reference_colors[face_index][0] - r) + 
           abs(reference_colors[face_index][1] - g) + 
           abs(reference_colors[face_index][2] - b);
}

int ArduinoStyleDetection::findClosestColor(int r, int g, int b) {
    int min_distance = INT_MAX;
    int closest_color = 0;
    
    for (int i = 0; i < 6; i++) {
        int distance = calculateColorDistance(r, g, b, i);
        if (distance < min_distance) {
            min_distance = distance;
            closest_color = i;
        }
    }
    
    return closest_color;
}

char ArduinoStyleDetection::applyColorDisambiguation(int min_color, int r, int g, int b) {
    // Apply the same disambiguation rules as Arduino code
    
    // Distinguish between red (0/U) and orange (3/D)
    if (min_color == 0 || min_color == 3) {
        if (b > g) {
            min_color = 0; // Red
        } else {
            min_color = 3; // Orange
        }
    }
    
    // Distinguish between orange (3/D) and yellow (5/B)
    if (min_color == 3 || min_color == 5) {
        if (g < 170) {
            min_color = 3; // Orange
        } else {
            min_color = 5; // Yellow
        }
    }
    
    // Distinguish between green (2/F) and blue (4/L)
    if (min_color == 2 || min_color == 4) {
        if (g > b) {
            min_color = 2; // Green
        } else {
            min_color = 4; // Blue
        }
    }
    
    // Distinguish between red (0/U) and blue (4/L)
    if (min_color == 0 || min_color == 4) {
        if (r > b) {
            min_color = 0; // Red
        } else {
            min_color = 4; // Blue
        }
    }
    
    // Distinguish between blue (4/L) and white (0/U)
    if (min_color == 0 && r < 120) {
        min_color = 4; // Blue
    }
    
    // Distinguish between white (0/U) and yellow (5/B)
    if (min_color == 0 && b < 180) {
        min_color = 5; // Yellow
    }
    
    return face_assign[min_color];
}

bool ArduinoStyleDetection::isHardFacet(int facet_index) {
    return std::find(hard_facets.begin(), hard_facets.end(), facet_index) != hard_facets.end();
}

char ArduinoStyleDetection::inferCornerColor(int facet_index, const std::array<char, 54>& cube_state) {
    // This is a simplified version - in a real implementation you'd need
    // to map each corner facet to its adjacent facets and use the lookup table
    
    // For now, return unknown - this would need cube geometry mapping
    return '?';
}

bool ArduinoStyleDetection::validateCubeConfiguration(const std::array<char, 54>& cube_state) {
    int count[6] = {0, 0, 0, 0, 0, 0}; // Count for each color
    
    for (int i = 0; i < 54; i++) {
        char color = cube_state[i];
        int color_index = -1;
        
        for (int j = 0; j < 6; j++) {
            if (color == face_assign[j]) {
                color_index = j;
                break;
            }
        }
        
        if (color_index >= 0) {
            count[color_index]++;
        }
    }
    
    bool valid = true;
    for (int i = 0; i < 6; i++) {
        if (count[i] != 9) {
            std::cout << "Error: Face " << face_assign[i] << " has " << count[i] << " facets" << std::endl;
            valid = false;
        }
    }
    
    if (valid) {
        std::cout << "✓ Cube Configuration is Correct" << std::endl;
    }
    
    return valid;
}

int ArduinoStyleDetection::detectCube(cv::Mat& frame1, cv::Mat& frame2, std::array<char, 54>& cube_state) {
    if (camera_1_points.empty() || camera_2_points.empty()) {
        std::cerr << "Error: Position calibration not loaded" << std::endl;
        return 0;
    }
    
    // Initialize cube state
    std::fill(cube_state.begin(), cube_state.end(), '?');
    
    // Phase 1: Detect "easy" facets using direct color analysis
    for (int face = 0; face < 6; face++) {
        for (int facet = 0; facet < 9; facet++) {
            int facet_index = face * 9 + facet;
            
            // Skip center facets (they're constant) and hard facets
            if (facet == 4 || isHardFacet(facet_index)) {
                continue;
            }
            
            cv::Vec3b bgr_pixel;
            bool pixel_valid = false;
            
            // Determine which camera to use based on face
            if (face % 2 == 1) {
                // Odd faces (1,3,5): R, D, B - use camera 2
                int point_index = face * 9 + facet;
                if (point_index < camera_2_points.size()) {
                    int x = camera_2_points[point_index].x;
                    int y = camera_2_points[point_index].y;
                    
                    if (x >= 0 && x < frame2.cols && y >= 0 && y < frame2.rows) {
                        bgr_pixel = frame2.at<cv::Vec3b>(y, x);
                        pixel_valid = true;
                    }
                }
            } else {
                // Even faces (0,2,4): U, F, L - use camera 1
                int point_index = face * 9 + facet;
                if (point_index < camera_1_points.size()) {
                    int x = camera_1_points[point_index].x;
                    int y = camera_1_points[point_index].y;
                    
                    if (x >= 0 && x < frame1.cols && y >= 0 && y < frame1.rows) {
                        bgr_pixel = frame1.at<cv::Vec3b>(y, x);
                        pixel_valid = true;
                    }
                }
            }
            
            if (pixel_valid) {
                // Convert BGR to RGB
                int r = bgr_pixel[2];
                int g = bgr_pixel[1];
                int b = bgr_pixel[0];
                
                // Find closest color and apply disambiguation
                int closest = findClosestColor(r, g, b);
                char final_color = applyColorDisambiguation(closest, r, g, b);
                
                cube_state[facet_index] = final_color;
            }
        }
    }
    
    // Set center facets (they define the face color)
    cube_state[4] = face_assign[0];   // U center
    cube_state[13] = face_assign[1];  // R center
    cube_state[22] = face_assign[2];  // F center
    cube_state[31] = face_assign[3];  // D center
    cube_state[40] = face_assign[4];  // L center
    cube_state[49] = face_assign[5];  // B center
    
    // Phase 2: Infer hard facets using geometric constraints
    // (This would require implementing the corner mapping logic)
    for (int hard_facet : hard_facets) {
        if (cube_state[hard_facet] == '?') {
            // For now, mark as unknown - full implementation would use corner inference
            cube_state[hard_facet] = 'X'; // Mark as undetected
        }
    }
    
    // Validate configuration
    return validateCubeConfiguration(cube_state) ? 1 : 0;
}

void ArduinoStyleDetection::calibrateColors(FrameCaptureCallback captureFrames) {
    std::cout << "=== Arduino-Style Color Calibration ===" << std::endl;
    std::cout << "This will help you set RGB reference values for each color." << std::endl;
    std::cout << "Point the camera at each color and press the corresponding key:" << std::endl;
    std::cout << "W=White, R=Red, G=Green, O=Orange, B=Blue, Y=Yellow, Q=Quit" << std::endl;
    
    cv::namedWindow("Arduino Color Calibration", cv::WINDOW_NORMAL);
    cv::resizeWindow("Arduino Color Calibration", 1200, 400);
    
    while (true) {
        cv::Mat frame1, frame2, display;
        
        // Capture fresh frames using callback
        if (!captureFrames(frame1, frame2)) {
            continue; // Skip if capture failed
        }
        
        if (!frame1.empty() && !frame2.empty()) {
            cv::hconcat(frame1, frame2, display);
        } else if (!frame1.empty()) {
            display = frame1.clone();
        } else if (!frame2.empty()) {
            display = frame2.clone();
        } else {
            continue; // Skip if no frames
        }
        
        // Add crosshair at center of frame1 for sampling
        if (!frame1.empty()) {
            int center_x = frame1.cols / 2;
            int center_y = frame1.rows / 2;
            cv::line(display, cv::Point(center_x - 20, center_y), cv::Point(center_x + 20, center_y), cv::Scalar(0, 255, 255), 2);
            cv::line(display, cv::Point(center_x, center_y - 20), cv::Point(center_x, center_y + 20), cv::Scalar(0, 255, 255), 2);
        }
        
        // Add instructions
        cv::putText(display, "Press: W/R/G/O/B/Y to calibrate colors, Q to quit", 
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(display, "Point cube face at crosshair before pressing key", 
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        
        // Show current reference colors
        for (int i = 0; i < 6; i++) {
            char color_name[20];
            sprintf(color_name, "%c: %d,%d,%d", face_assign[i], 
                   reference_colors[i][0], reference_colors[i][1], reference_colors[i][2]);
            cv::putText(display, color_name, cv::Point(10, 100 + i * 25), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
        
        cv::imshow("Arduino Color Calibration", display);
        
        int key = cv::waitKey(30) & 0xFF;
        if (key == 'q' || key == 'Q') break;
        
        // Sample center pixel from frame1 when a color key is pressed
        if (!frame1.empty()) {
            int center_x = frame1.cols / 2;
            int center_y = frame1.rows / 2;
            cv::Vec3b center_pixel = frame1.at<cv::Vec3b>(center_y, center_x);
            
            int r = center_pixel[2];
            int g = center_pixel[1]; 
            int b = center_pixel[0];
            
            int color_index = -1;
            switch (key) {
                case 'w': case 'W': color_index = 0; break;
                case 'r': case 'R': color_index = 1; break;
                case 'g': case 'G': color_index = 2; break;
                case 'o': case 'O': color_index = 3; break;
                case 'b': case 'B': color_index = 4; break;
                case 'y': case 'Y': color_index = 5; break;
            }
            
            if (color_index >= 0) {
                reference_colors[color_index][0] = r;
                reference_colors[color_index][1] = g;
                reference_colors[color_index][2] = b;
                std::cout << "Calibrated " << face_assign[color_index] << ": RGB(" << r << "," << g << "," << b << ")" << std::endl;
                
                // Save to file
                std::ofstream outfile("arduino_colors.txt");
                if (outfile.is_open()) {
                    char color_chars[] = {'W', 'R', 'G', 'O', 'B', 'Y'};
                    for (int i = 0; i < 6; i++) {
                        outfile << color_chars[i] << " " << reference_colors[i][0] << " " 
                               << reference_colors[i][1] << " " << reference_colors[i][2] << std::endl;
                    }
                    outfile.close();
                    std::cout << "Colors saved to arduino_colors.txt" << std::endl;
                }
            }
        }
    }
    
    cv::destroyAllWindows();
}

void ArduinoStyleDetection::printCubeState(const std::array<char, 54>& cube_state) {
    std::cout << "\n=== Cube State ===" << std::endl;
    for (int face = 0; face < 6; face++) {
        std::cout << face_assign[face] << ": ";
        for (int facet = 0; facet < 9; facet++) {
            std::cout << cube_state[face * 9 + facet];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}