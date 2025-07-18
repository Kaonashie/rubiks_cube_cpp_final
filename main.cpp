#include <chrono>
#include <map>
#include <thread>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "opencv2/opencv.hpp"
#include "arduino_detection.h"

// Configuration structure
struct Config {
    int camera_1_index = 4;
    int camera_2_index = 5;
    int camera_width = 320;
    int camera_height = 240;
    int camera_fps = 187;
    int exposure = 15;
    int gain = 10;
    int brightness = 15;
    int contrast = 9;
    int saturation = 60;
};

// Global configuration
static Config config;

class PS3EyeCamera {
private:
	int height{}, width{}, index{}, fps{};
	cv::VideoCapture video_capture;
	bool camera_initialized = false;
	std::ofstream position_out_file;
public:
	explicit PS3EyeCamera(int height = 320, int width = 240, int index = 4, int fps = 187);
	bool calibratePosition(const std::string &filename);
	void calibrateColors(const std::string &filename);
	void friend positionMouseCallback(int event, int x, int y, int flags, void* userdata);
	void capture(cv::Mat& frame);
	void optimizeForDualCamera();
};

// Function declarations
void show_dual_camera_feed(PS3EyeCamera *camera_1, PS3EyeCamera *camera_2);
void drawPositioningGrid(cv::Mat& frame);
void loadConfig(const std::string& filename);
void initializeCameras();

// Global variables for calibration
static int h_min = 0, s_min = 0, v_min = 0;
static int h_max = 179, s_max = 255, v_max = 255;
static int current_click = 0;
static int current_face = 0;
static int current_facelet = 0;
static bool is_camera_1 = true; // Track which camera is being calibrated
static std::ofstream* current_position_file = nullptr;
static cv::Mat* current_display_frame = nullptr;
static std::vector<cv::Point> clicked_points;

// Face names for display and cube solving (Camera 1: Front, Right, Up | Camera 2: Back, Left, Down)
static const char* face_names[] = {"Front", "Right", "Up", "Back", "Left", "Down"};
static const char* face_colors[] = {"Red", "Green", "White", "Orange", "Blue", "Yellow"};
static const char face_color_chars[] = {'R', 'G', 'W', 'O', 'B', 'Y'};
static const char* face_positions[] = {
    // Camera 1 faces (Front, Right, Up)
    "Corner-TL", "Edge-T", "Corner-TR", "Edge-L", "Edge-R", "Corner-BL", "Edge-B", "Corner-BR", // Front face (skip center at index 4)
    "Corner-TL", "Edge-T", "Corner-TR", "Edge-L", "Edge-R", "Corner-BL", "Edge-B", "Corner-BR", // Right face
    "Corner-TL", "Edge-T", "Corner-TR", "Edge-L", "Edge-R", "Corner-BL", "Edge-B", "Corner-BR", // Up face
    // Camera 2 faces (Back, Left, Down)  
    "Corner-TL", "Edge-T", "Corner-TR", "Edge-L", "Edge-R", "Corner-BL", "Edge-B", "Corner-BR", // Back face
    "Corner-TL", "Edge-T", "Corner-TR", "Edge-L", "Edge-R", "Corner-BL", "Edge-B", "Corner-BR", // Left face
    "Corner-TL", "Edge-T", "Corner-TR", "Edge-L", "Edge-R", "Corner-BL", "Edge-B", "Corner-BR"  // Down face
};

// Default color ranges for auto-suggestion
struct ColorRange {
    int h_min, h_max, s_min, s_max, v_min, v_max;
};

static const ColorRange default_ranges[] = {
    {0, 179, 0, 50, 150, 255},     // White
    {0, 10, 80, 255, 80, 255},     // Red (part 1)
    {9, 20, 100, 255, 100, 255},   // Orange  
    {21, 35, 80, 255, 120, 255},   // Yellow
    {45, 75, 60, 255, 60, 255},    // Green
    {100, 125, 80, 255, 80, 255}   // Blue
};

// Callback functions
static void on_trackbar(int, void*) {}

static void reset_to_defaults(int color_index) {
    if (color_index >= 0 && color_index < 6) {
        const ColorRange& range = default_ranges[color_index];
        h_min = range.h_min;
        h_max = range.h_max;
        s_min = range.s_min;
        s_max = range.s_max;
        v_min = range.v_min;
        v_max = range.v_max;
        
        // Update trackbars
        cv::setTrackbarPos("H_MIN", "Controls", h_min);
        cv::setTrackbarPos("H_MAX", "Controls", h_max);
        cv::setTrackbarPos("S_MIN", "Controls", s_min);
        cv::setTrackbarPos("S_MAX", "Controls", s_max);
        cv::setTrackbarPos("V_MIN", "Controls", v_min);
        cv::setTrackbarPos("V_MAX", "Controls", v_max);
    }
}

void PS3EyeCamera::calibrateColors(const std::string& output_filename) {
    std::ofstream outfile(output_filename, std::ios::app);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << output_filename << " for writing." << std::endl;
        return;
    }

    const char* colors[] = {"White", "Red", "Orange", "Yellow", "Green", "Blue"};
    const char color_chars[] = {'W', 'R', 'O', 'Y', 'G', 'B'};

    // Create a single large window for combined display
    cv::namedWindow("Color Calibration", cv::WINDOW_NORMAL);
    cv::resizeWindow("Color Calibration", 1400, 800);
    
    // Create trackbars window
    cv::namedWindow("Controls", cv::WINDOW_NORMAL);
    cv::resizeWindow("Controls", 400, 300);
    
    // Create trackbars
    cv::createTrackbar("H_MIN", "Controls", &h_min, 179, on_trackbar);
    cv::createTrackbar("H_MAX", "Controls", &h_max, 179, on_trackbar);
    cv::createTrackbar("S_MIN", "Controls", &s_min, 255, on_trackbar);
    cv::createTrackbar("S_MAX", "Controls", &s_max, 255, on_trackbar);
    cv::createTrackbar("V_MIN", "Controls", &v_min, 255, on_trackbar);
    cv::createTrackbar("V_MAX", "Controls", &v_max, 255, on_trackbar);

    // Position windows
    cv::moveWindow("Color Calibration", 50, 50);
    cv::moveWindow("Controls", 1500, 50);

    for (int i = 0; i < 6; ++i) {
        // Auto-suggest starting values
        reset_to_defaults(i);
        
        std::cout << "\n=== Calibrating for color: " << colors[i] << " ===" << std::endl;
        std::cout << "Controls:" << std::endl;
        std::cout << "  's' = Save current settings" << std::endl;
        std::cout << "  'r' = Reset to default range" << std::endl;
        std::cout << "  'q' = Quit calibration" << std::endl;
        std::cout << "Tip: Hold the " << colors[i] << " face up to the camera" << std::endl;

        while (true) {
            cv::Mat frame, hsv_frame, mask, preview;
            this->capture(frame);
            cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV);

            // Create mask from trackbar values
            cv::Scalar lower_bound(h_min, s_min, v_min);
            cv::Scalar upper_bound(h_max, s_max, v_max);
            cv::inRange(hsv_frame, lower_bound, upper_bound, mask);

            // Create preview showing detected pixels
            cv::bitwise_and(frame, frame, preview, mask);
            
            // Scale up the images for better visibility
            cv::Mat frame_large, mask_large, preview_large;
            cv::resize(frame, frame_large, cv::Size(400, 300));
            cv::resize(mask, mask_large, cv::Size(400, 300));
            cv::resize(preview, preview_large, cv::Size(400, 300));
            
            // Convert mask to 3-channel for concatenation
            cv::Mat mask_colored;
            cv::cvtColor(mask_large, mask_colored, cv::COLOR_GRAY2BGR);
            
            // Create combined display
            cv::Mat top_row, bottom_row, combined_display;
            cv::hconcat(frame_large, mask_colored, top_row);
            cv::hconcat(preview_large, cv::Mat::zeros(300, 400, CV_8UC3), bottom_row);
            cv::vconcat(top_row, bottom_row, combined_display);
            
            // Add labels to each section
            cv::putText(combined_display, "Original", cv::Point(10, 25), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            cv::putText(combined_display, "Mask", cv::Point(410, 25), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            cv::putText(combined_display, "Detected", cv::Point(10, 325), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            
            // Add current color and instructions
            cv::putText(combined_display, "Color: " + std::string(colors[i]), cv::Point(410, 325), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
            cv::putText(combined_display, "Hold " + std::string(colors[i]) + " face to camera", 
                       cv::Point(410, 360), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
            
            // Show current HSV ranges
            std::string range_text = "H:" + std::to_string(h_min) + "-" + std::to_string(h_max) + 
                                   " S:" + std::to_string(s_min) + "-" + std::to_string(s_max) +
                                   " V:" + std::to_string(v_min) + "-" + std::to_string(v_max);
            cv::putText(combined_display, range_text, cv::Point(10, combined_display.rows - 40), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            
            // Add control instructions
            cv::putText(combined_display, "Controls: S=Save, R=Reset, Q=Quit", cv::Point(10, combined_display.rows - 10), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

            cv::imshow("Color Calibration", combined_display);

            int key = cv::waitKey(30);
            if (key == 's') {
                // Save the calibrated values to the file
                outfile << color_chars[i] << " " << h_min << " " << h_max << " "
                        << s_min << " " << s_max << " " << v_min << " " << v_max << std::endl;
                std::cout << colors[i] << " range saved: H(" << h_min << "-" << h_max 
                         << ") S(" << s_min << "-" << s_max << ") V(" << v_min << "-" << v_max << ")" << std::endl;
                break;
            }
            if (key == 'r') {
                std::cout << "Reset to default range for " << colors[i] << std::endl;
                reset_to_defaults(i);
            }
            if (key == 'q') {
                outfile.close();
                cv::destroyAllWindows();
                return;
            }
        }
    }

    outfile.close();
    cv::destroyAllWindows();
    std::cout << "\n=== Calibration complete! Values saved to " << output_filename << " ===" << std::endl;
}

PS3EyeCamera::PS3EyeCamera(int height, int width, int index, int fps) {
	this->height = height;
	this->width = width;
	this->index = index;
	this->fps = fps;
	
	video_capture.open(index, cv::CAP_V4L2);
	
	// Set basic parameters
	// video_capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
	// video_capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
	video_capture.set(cv::CAP_PROP_FPS, fps);
	// video_capture.set(cv::CAP_PROP_BUFFERSIZE, 1); // Keep buffer small for real-time

	// Camera settings for stable image using config values
	// video_capture.set(cv::CAP_PROP_AUTO_WB, 1); // Disable Auto White Balance
	// video_capture.set(cv::CAP_PROP_AUTO_EXPOSURE, 0); // Disable Auto Exposure
	// video_capture.set(cv::CAP_PROP_EXPOSURE, config.exposure);
	// video_capture.set(cv::CAP_PROP_GAIN, config.gain);
	// video_capture.set(cv::CAP_PROP_BRIGHTNESS, config.brightness);
	// video_capture.set(cv::CAP_PROP_CONTRAST, config.contrast);
	// video_capture.set(cv::CAP_PROP_SATURATION, config.saturation);

	if (!video_capture.isOpened()) {
		std::cerr << "Camera " << index << " could not be opened" << std::endl;
		throw std::exception();
	}

	// Warmup with 5 frames to stabilize
	cv::Mat temp_frames;
	for (int i = 0; i < 5; i++) {
		video_capture.read(temp_frames);
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	
	std::cout << "Camera " << index << " initialized successfully" << std::endl;
}

void drawTargetGuide(cv::Mat& display, int face_index, int piece_index) {
    // Draw a 3x3 grid to show piece layout
    int grid_size = 120;
    int start_x = display.cols - grid_size - 20;
    int start_y = 80;
    int cell_size = grid_size / 3;
    
    // Draw grid background
    cv::rectangle(display, cv::Point(start_x - 5, start_y - 5), 
                  cv::Point(start_x + grid_size + 5, start_y + grid_size + 5),
                  cv::Scalar(50, 50, 50), -1);
    
    // Draw grid lines
    for (int i = 0; i <= 3; i++) {
        cv::line(display, cv::Point(start_x + i * cell_size, start_y),
                 cv::Point(start_x + i * cell_size, start_y + grid_size),
                 cv::Scalar(200, 200, 200), 1);
        cv::line(display, cv::Point(start_x, start_y + i * cell_size),
                 cv::Point(start_x + grid_size, start_y + i * cell_size),
                 cv::Scalar(200, 200, 200), 1);
    }
    
    // Map piece index (0-7) to grid position (skipping center at index 4)
    int grid_positions[] = {0, 1, 2, 3, 5, 6, 7, 8}; // Skip center (4)
    int grid_pos = grid_positions[piece_index];
    int grid_row = grid_pos / 3;
    int grid_col = grid_pos % 3;
    
    // Highlight target cell
    int cell_x = start_x + grid_col * cell_size;
    int cell_y = start_y + grid_row * cell_size;
    cv::rectangle(display, cv::Point(cell_x + 2, cell_y + 2),
                  cv::Point(cell_x + cell_size - 2, cell_y + cell_size - 2),
                  cv::Scalar(0, 255, 0), 3);
    
    // Draw center piece differently (blocked)
    int center_x = start_x + 1 * cell_size;
    int center_y = start_y + 1 * cell_size;
    cv::rectangle(display, cv::Point(center_x + 2, center_y + 2),
                  cv::Point(center_x + cell_size - 2, center_y + cell_size - 2),
                  cv::Scalar(128, 128, 128), -1);
    cv::putText(display, "X", cv::Point(center_x + cell_size/2 - 8, center_y + cell_size/2 + 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    
    // Add face label with color
    std::string face_label = std::string(face_names[face_index]) + " (" + face_colors[face_index] + ")";
    cv::putText(display, face_label, cv::Point(start_x, start_y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
}

void positionMouseCallback(int event, int x, int y, int flags, void *userdata) {
    if (event == cv::EVENT_LBUTTONDOWN && current_position_file && current_display_frame) {
        // Save coordinates to file
        *current_position_file << x << " " << y << "\n";
        current_position_file->flush();
        
        // Add to clicked points for visual feedback
        clicked_points.push_back(cv::Point(x, y));
        
        // Update display
        cv::Mat display = current_display_frame->clone();
        
        // Draw all previous clicks
        for (size_t i = 0; i < clicked_points.size(); i++) {
            cv::circle(display, clicked_points[i], 5, cv::Scalar(0, 255, 0), -1);
            cv::putText(display, std::to_string(i + 1), 
                       cv::Point(clicked_points[i].x + 8, clicked_points[i].y - 8),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }
        
        // Draw target guide
        drawTargetGuide(display, current_face, current_facelet);
        
        // Current target info with better description
        int total_pieces_per_face = 8; // Excluding center
        std::string position_desc = face_positions[current_face * 8 + current_facelet];
        std::string target_text = std::string(face_names[current_face]) + " (" + face_colors[current_face] + ") face: " + position_desc;
        cv::putText(display, target_text, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        // Show progress for current face
        std::string face_progress = "Face progress: " + std::to_string(current_facelet + 1) + "/8";
        cv::putText(display, face_progress, cv::Point(10, 60), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        
        // Show total progress (24 pieces total, 8 per face × 3 faces per camera)
        std::string total_progress = "Total: " + std::to_string(current_click + 1) + "/24";
        cv::putText(display, total_progress, cv::Point(10, 90), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        
        cv::imshow("calibration", display);
        
        std::cout << "✓ Saved point " << (current_click + 1) << ": (" << x << ", " << y 
                  << ") for " << face_names[current_face] << " " << position_desc << std::endl;
        
        current_click++;
        current_facelet++;
        
        // Move to next face if current face is complete (8 pieces per face)
        if (current_facelet >= 8) {
            current_facelet = 0;
            current_face++;
            
            // Check if we've completed all faces for this camera
            bool more_faces = is_camera_1 ? (current_face <= 2) : (current_face <= 5);
            
            if (more_faces) {
                std::cout << "\n--- Moving to " << face_names[current_face] << " (" << face_colors[current_face] << ") face ---" << std::endl;
                std::cout << "Click on the 8 edge/corner pieces of the " << face_names[current_face] 
                          << " face (skip center)" << std::endl;
            }
        }
    }
}


bool PS3EyeCamera::calibratePosition(const std::string &filename) {
    position_out_file.open(filename, std::ios::app);
    if (!position_out_file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
        return false;
    }

    // Initialize global variables for callback
    current_position_file = &position_out_file;
    current_click = 0;
    
    // Determine starting face based on filename
    if (filename.find("pos_1") != std::string::npos) {
        is_camera_1 = true;
        current_face = 0; // Camera 1 starts with Front face (index 0)
        std::cout << "Camera 1 will calibrate: Front (Red), Right (Green), Up (White)" << std::endl;
    } else {
        is_camera_1 = false;
        current_face = 3; // Camera 2 starts with Back face (index 3)  
        std::cout << "Camera 2 will calibrate: Back (Orange), Left (Blue), Down (Yellow)" << std::endl;
    }
    
    current_facelet = 0;
    clicked_points.clear();

    cv::Mat frame;
    video_capture.read(frame);
    current_display_frame = &frame;

    cv::namedWindow("calibration", cv::WINDOW_NORMAL);
    cv::resizeWindow("calibration", 1280, 960);

    std::cout << "\n=== Position Calibration ===" << std::endl;
    std::cout << "Instructions:" << std::endl;
    std::cout << "- Click on the 8 edge/corner pieces (SKIP CENTER PIECES)" << std::endl;
    std::cout << "- Follow the green highlight in the grid guide (right side)" << std::endl;
    std::cout << "- Green circles show your clicks with numbers" << std::endl;
    std::cout << "- Press SPACE to refresh camera feed" << std::endl;
    std::cout << "- Press 'q' to quit and save" << std::endl;
    std::cout << "- Press 'r' to restart current face" << std::endl;
    std::cout << "\nStarting with " << face_names[current_face] << " face..." << std::endl;
    std::cout << "Click on the 8 edge/corner pieces of the " << face_names[current_face] 
              << " face (center will be hardcoded)" << std::endl;

    cv::setMouseCallback("calibration", positionMouseCallback, this);

    while (current_click < 24) { // 24 pieces total (8 per face × 3 faces per camera)
        // Create display with overlays
        cv::Mat display = frame.clone();
        
        // Draw all clicked points
        for (size_t i = 0; i < clicked_points.size(); i++) {
            cv::circle(display, clicked_points[i], 5, cv::Scalar(0, 255, 0), -1);
            cv::putText(display, std::to_string(i + 1), 
                       cv::Point(clicked_points[i].x + 8, clicked_points[i].y - 8),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }
        
        // Draw target guide if still calibrating
        if (current_face < 6) {
            drawTargetGuide(display, current_face, current_facelet);
            
            std::string position_desc = face_positions[current_face * 8 + current_facelet];
            std::string target_text = std::string(face_names[current_face]) + " face: " + position_desc;
            cv::putText(display, target_text, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        }
        
        // Progress info
        std::string progress_text = "Total: " + std::to_string(current_click) + "/24";
        cv::putText(display, progress_text, cv::Point(10, 60), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        
        // Instructions
        cv::putText(display, "SPACE=refresh, R=restart face, Q=quit", 
                   cv::Point(10, display.rows - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

        cv::imshow("calibration", display);

        int key = cv::waitKey(30);
        if (key == ' ') {
            // Refresh camera feed
            video_capture.read(frame);
            current_display_frame = &frame;
            std::cout << "Camera feed refreshed" << std::endl;
        }
        else if (key == 'r' && current_facelet > 0) {
            // Restart current face
            int points_to_remove = current_facelet;
            for (int i = 0; i < points_to_remove; i++) {
                if (!clicked_points.empty()) {
                    clicked_points.pop_back();
                    current_click--;
                }
            }
            current_facelet = 0;
            std::cout << "Restarting " << face_names[current_face] << " face" << std::endl;
        }
        else if (key == 'q') {
            break;
        }
    }

    if (current_click >= 24) {
        std::cout << "\n✓ Position calibration completed successfully!" << std::endl;
        std::cout << "Saved " << current_click << " edge/corner points to " << filename << std::endl;
        std::cout << "Center pieces will use hardcoded colors during detection." << std::endl;
    } else {
        std::cout << "\nCalibration stopped. Saved " << current_click << " points to " << filename << std::endl;
    }

    current_position_file = nullptr;
    current_display_frame = nullptr;
    position_out_file.close();
    cv::destroyAllWindows();
    return true;
}

void PS3EyeCamera::capture(cv::Mat &frame) {
	video_capture.read(frame);
}

void PS3EyeCamera::optimizeForDualCamera() {
	// Optimize settings for dual camera operation
	video_capture.set(cv::CAP_PROP_BUFFERSIZE, 1); // Minimal buffer for real-time
	
	// Clear any buffered frames
	cv::Mat dummy;
	for (int i = 0; i < 3; i++) {
		video_capture.read(dummy);
	}
	
	std::cout << "Camera optimized for dual operation (keeping original FPS)" << std::endl;
}



using namespace cv;
static Mat global_frame_1, global_frame_2, global_hsv_1, global_hsv_2;
static char color_lut[180][256][256];
static std::vector<char> glob_colors_cam_1(24); // 8 pieces per face × 3 faces = 24
static std::vector<char> glob_colors_cam_2(24); // 8 pieces per face × 3 faces = 24

// Hardcoded center colors for each face (standard cube mapping)
// Front=Red, Right=Green, Up=White, Back=Orange, Left=Blue, Down=Yellow
static const std::map<std::string, char> face_centers = {
    {"Front", 'R'}, {"Right", 'G'}, {"Up", 'W'},
    {"Back", 'O'}, {"Left", 'B'}, {"Down", 'Y'}
};
static PS3EyeCamera* camera_1 = nullptr;
static PS3EyeCamera* camera_2 = nullptr;

static std::vector<Point> points_cam_1;
static std::vector<Point> points_cam_2;

char find_color_lut(Vec3b hsv_pixel) { return color_lut[hsv_pixel[0]][hsv_pixel[1]][hsv_pixel[2]]; }

void init_lut() {
	for (int h = 0; h < 180; h++) {
		for (int s = 0; s < 256; s++) {
			for (int v = 0; v < 256; v++) {
				// White - very strict
				if (s <= 50 && v >= 150) {
					color_lut[h][s][v] = 'W';
				}
				// Red - tighter range
				else if (((h >= 0 && h <= 8) || (h >= 172 && h <= 179)) && s >= 80 && v >= 80) {
					color_lut[h][s][v] = 'R';
				}
				// Orange - non-overlapping with red
				else if (h >= 9 && h <= 20 && s >= 100 && v >= 100) {
					color_lut[h][s][v] = 'O';
				}
				// Yellow - non-overlapping
				else if (h >= 21 && h <= 35 && s >= 80 && v >= 120) {
					color_lut[h][s][v] = 'Y';
				}
				// Green - tighter range
				else if (h >= 45 && h <= 75 && s >= 60 && v >= 60) {
					color_lut[h][s][v] = 'G';
				}
				// Blue - much tighter range to avoid overlap
				else if (h >= 100 && h <= 125 && s >= 80 && v >= 80) {
					color_lut[h][s][v] = 'B';
				}
				else {
					color_lut[h][s][v] = 'N';
				}
			}
		}
	}
}

void init_mat() {
	global_frame_1 = Mat::zeros(config.camera_height, config.camera_width, CV_8UC3);
	global_frame_2 = Mat::zeros(config.camera_height, config.camera_width, CV_8UC3);
	global_hsv_1 = Mat::zeros(config.camera_height, config.camera_width, CV_8UC3);
	global_hsv_2 = Mat::zeros(config.camera_height, config.camera_width, CV_8UC3);
}

char findColor(Vec3b hsv_pixel) {
	int h = hsv_pixel[0];
	int s = hsv_pixel[1];
	int v = hsv_pixel[2];
	if (((h >= 0 && h <= 10) || (h >= 170 && h <= 179)) && s >= 50 && v >= 50)
		return 'R'; // Red
	else if (h >= 10 && h <= 25 && s >= 120 && v >= 120)
		return 'O'; // Orange
	else if (h >= 25 && h <= 35 && s >= 100 && v >= 100)
		return 'Y'; // Yellow
	else if (h >= 45 && h <= 75 && s >= 80 && v >= 80)
		return 'G'; // Green
	else if (h >= 100 && h <= 130 && s >= 100 && v >= 100)
		return 'B'; // Blue
	else if (s <= 100 && v >= 100)
		return 'W'; // White
	else
		return 'N'; // None/Unknown
}


void load_position(const std::string &filename_1 , const std::string &filename_2) {
	std::ifstream in(filename_1);
	int x, y;
	for (int i = 0; i < 24; i++) { // 8 pieces × 3 faces = 24 points per camera
		in >> x >> y;
		points_cam_1.emplace_back(x, y);
	}
	in.close();
	std::ifstream in_2(filename_2);

	for (int i = 0; i < 24; i++) {
		in_2 >> x >> y;
		points_cam_2.emplace_back(x, y);
	}
	in_2.close();
}

void detect_cam_1() {
	if (camera_1) {
		camera_1->capture(global_frame_1);
		if (global_frame_1.empty()) {
			std::cerr << "Error: Camera 1 frame is empty" << std::endl;
			return;
		}
		cvtColor(global_frame_1, global_hsv_1, COLOR_BGR2HSV);
		
		for (int i = 0; i < points_cam_1.size(); i++) {
			int x = points_cam_1[i].x;
			int y = points_cam_1[i].y;
			
			// Check bounds before accessing pixel
			if (x >= 0 && x < global_hsv_1.cols && y >= 0 && y < global_hsv_1.rows) {
				const auto hsv_pixel = global_hsv_1.at<Vec3b>(y, x);
				glob_colors_cam_1[i] = find_color_lut(hsv_pixel);
			} else {
				std::cerr << "Warning: Point " << i << " (" << x << "," << y << ") is out of bounds for camera 1 frame (" 
						  << global_hsv_1.cols << "x" << global_hsv_1.rows << ")" << std::endl;
				glob_colors_cam_1[i] = 'N'; // Unknown color for out-of-bounds points
			}
		}
	}
}

void detect_cam_2() {
	if (camera_2) {
		camera_2->capture(global_frame_2);
		if (global_frame_2.empty()) {
			std::cerr << "Error: Camera 2 frame is empty" << std::endl;
			return;
		}
		cvtColor(global_frame_2, global_hsv_2, COLOR_BGR2HSV);
		
		for (int i = 0; i < points_cam_2.size(); i++) {
			int x = points_cam_2[i].x;
			int y = points_cam_2[i].y;
			
			// Check bounds before accessing pixel
			if (x >= 0 && x < global_hsv_2.cols && y >= 0 && y < global_hsv_2.rows) {
				const auto hsv_pixel = global_hsv_2.at<Vec3b>(y, x);
				glob_colors_cam_2[i] = find_color_lut(hsv_pixel);
			} else {
				std::cerr << "Warning: Point " << i << " (" << x << "," << y << ") is out of bounds for camera 2 frame (" 
						  << global_hsv_2.cols << "x" << global_hsv_2.rows << ")" << std::endl;
				glob_colors_cam_2[i] = 'N'; // Unknown color for out-of-bounds points
			}
		}
	}
}


void print_colors() {
	for (auto c: glob_colors_cam_1) {
		std::cout << c << std::endl;
	}
	for (auto c : glob_colors_cam_2) {
		std::cout << c << std::endl;
	}
}

void load_lut_from_file(const std::string& filename) {
	// Initialize LUT with 'N' (None)
	memset(color_lut, 'N', sizeof(color_lut));

	std::ifstream infile(filename);
	if (!infile.is_open()) {
		std::cerr << "Could not open LUT file: " << filename << ". Using default hardcoded LUT." << std::endl;
		init_lut(); // Fallback to the old version
		return;
	}

	std::string line;
	while (std::getline(infile, line)) {
		std::stringstream ss(line);
		char color;
		int h_min, h_max, s_min, s_max, v_min, v_max;

		ss >> color >> h_min >> h_max >> s_min >> s_max >> v_min >> v_max;

		// Handle red's hue wrap-around where h_min > h_max
		bool red_wrap = (color == 'R' && h_min > h_max);

		for (int h = 0; h < 180; ++h) {
			bool h_in_range = red_wrap ? (h >= h_min || h <= h_max) : (h >= h_min && h <= h_max);
			if (h_in_range) {
				for (int s = s_min; s <= s_max; ++s) {
					for (int v = v_min; v <= v_max; ++v) {
						color_lut[h][s][v] = color;
					}
				}
			}
		}
	}
	std::cout << "Custom color LUT loaded from " << filename << std::endl;
}

int process() {
	Mat frame, hsv, red_mask, out;
	try {
		frame = imread("cube_scrambled.png");
		if (frame.empty()) {
			std::cout << "Could not load cube_scrambled.png" << std::endl;
			return -1;
		}
	} catch (cv::Exception &e) {
		std::cout << "Exception : " << e.what() << std::endl;
		return -1;
	}
	
	cvtColor(frame, hsv, COLOR_BGR2HSV);
	
	// Define red color range for mask
	Scalar lower_red(0, 50, 50);
	Scalar upper_red(10, 255, 255);
	inRange(hsv, lower_red, upper_red, red_mask);
	
	bitwise_and(frame, frame, out, red_mask);
	imwrite("red_mask.jpg", out);
	return 0;
}

void benchmark() {
	detect_cam_1();
	detect_cam_2();
}



void show_camera_setup_guide() {
	std::cout << "\n=== Camera Setup Guide ===" << std::endl;
	std::cout << "For proper cube detection, each camera must see specific faces:" << std::endl;
	std::cout << std::endl;
	std::cout << "📹 CAMERA 1 should see these faces:" << std::endl;
	std::cout << "  • Front face (Red center)" << std::endl;
	std::cout << "  • Right face (Green center)" << std::endl;
	std::cout << "  • Up face (White center)" << std::endl;
	std::cout << std::endl;
	std::cout << "📹 CAMERA 2 should see these faces:" << std::endl;
	std::cout << "  • Back face (Orange center)" << std::endl;
	std::cout << "  • Left face (Blue center)" << std::endl;
	std::cout << "  • Down face (Yellow center)" << std::endl;
	std::cout << std::endl;
	std::cout << "💡 TIP: Position your cube so that:" << std::endl;
	std::cout << "  - Camera 1 sees the front-right corner of the cube" << std::endl;
	std::cout << "  - Camera 2 sees the back-left corner of the cube" << std::endl;
	std::cout << "  - Both cameras can see the top and bottom faces partially" << std::endl;
	std::cout << std::endl;
	std::cout << "Press any key to continue..." << std::endl;
	std::cin.get();
}

void show_dual_camera_feed(PS3EyeCamera *camera_1, PS3EyeCamera *camera_2) {
	Mat frame1, frame2, display1, display2;
	
	// Create windows
	namedWindow("Camera 1 (Front/Right/Up)", WINDOW_NORMAL);
	namedWindow("Camera 2 (Back/Left/Down)", WINDOW_NORMAL);
	resizeWindow("Camera 1 (Front/Right/Up)", 640, 480);
	resizeWindow("Camera 2 (Back/Left/Down)", 640, 480);
	
	// Position windows side by side
	moveWindow("Camera 1 (Front/Right/Up)", 50, 50);
	moveWindow("Camera 2 (Back/Left/Down)", 720, 50);
	
	std::cout << "\n=== Dual Camera Feed ===" << std::endl;
	std::cout << "Use this to position your cameras to see the cube properly." << std::endl;
	std::cout << "Camera 1 should see: Front, Right, Up faces" << std::endl;
	std::cout << "Camera 2 should see: Back, Left, Down faces" << std::endl;
	std::cout << "\nControls:" << std::endl;
	std::cout << "  ESC or Q = Quit" << std::endl;
	std::cout << "  SPACE = Take snapshot and save images" << std::endl;
	std::cout << "  F = Toggle FPS display" << std::endl;
	
	int frame_count = 0;
	bool show_fps = false;
	auto last_time = std::chrono::high_resolution_clock::now();
	double fps = 0.0;
	
	// Initial warmup - clear any buffered frames
	std::cout << "Warming up cameras..." << std::endl;
	for (int i = 0; i < 5; i++) {
		camera_1->capture(frame1);
		camera_2->capture(frame2);
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}
	std::cout << "Ready!" << std::endl;
	
	while (true) {
		auto start_time = std::chrono::high_resolution_clock::now();
		
		try {
			// Capture from both cameras simultaneously
			camera_1->capture(frame1);
			camera_2->capture(frame2);
			
			if (frame1.empty() || frame2.empty()) {
				std::cout << "Warning: Could not capture from one or both cameras" << std::endl;
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
				continue;
			}
			
			// Create display frames with info overlays
			display1 = frame1.clone();
			display2 = frame2.clone();
			
			// Add labels and info to frames
			cv::putText(display1, "Camera 1: Front/Right/Up", cv::Point(10, 30), 
						cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
			cv::putText(display1, "Expected: Red/Green/White centers", cv::Point(10, 60), 
						cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
			cv::putText(display1, "Index: 4", cv::Point(10, 90), 
						cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
			
			cv::putText(display2, "Camera 2: Back/Left/Down", cv::Point(10, 30), 
						cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
			cv::putText(display2, "Expected: Orange/Blue/Yellow centers", cv::Point(10, 60), 
						cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
			cv::putText(display2, "Index: 5", cv::Point(10, 90), 
						cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
			
			// Calculate FPS
			auto current_time = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time);
			if (duration.count() > 1000) { // Update FPS every second
				fps = frame_count * 1000.0 / duration.count();
				frame_count = 0;
				last_time = current_time;
			}
			
			// Add frame counter and FPS
			std::string info1 = "Frame: " + std::to_string(frame_count);
			std::string info2 = "Frame: " + std::to_string(frame_count);
			if (show_fps) {
				info1 += " | FPS: " + std::to_string((int)fps);
				info2 += " | FPS: " + std::to_string((int)fps);
			}
			
			cv::putText(display1, info1, cv::Point(10, display1.rows - 10), 
						cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
			cv::putText(display2, info2, cv::Point(10, display2.rows - 10), 
						cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
			
			// Draw grid overlay to help with positioning
			drawPositioningGrid(display1);
			drawPositioningGrid(display2);
			
			// Display frames
			imshow("Camera 1 (Front/Right/Up)", display1);
			imshow("Camera 2 (Back/Left/Down)", display2);
			
			frame_count++;
			
		} catch (const cv::Exception &e) {
			std::cerr << "Exception during capture: " << e.what() << std::endl;
		}
		
		// Handle keyboard input with shorter delay for smoother display
		int key = waitKey(1) & 0xFF; // ~60 FPS max display rate
		if (key == 27 || key == 'q' || key == 'Q') { // ESC or Q
			break;
		}
		else if (key == 'f' || key == 'F') { // Toggle FPS display
			show_fps = !show_fps;
			std::cout << "FPS display: " << (show_fps ? "ON" : "OFF") << std::endl;
		}
		else if (key == ' ') { // SPACE - take snapshot
			std::string timestamp = std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
				std::chrono::system_clock::now().time_since_epoch()).count());
			
			std::string filename1 = "camera1_snapshot_" + timestamp + ".jpg";
			std::string filename2 = "camera2_snapshot_" + timestamp + ".jpg";
			
			imwrite(filename1, frame1);
			imwrite(filename2, frame2);
			
			std::cout << "📸 Snapshots saved: " << filename1 << ", " << filename2 << std::endl;
			
			// Flash effect
			Mat white1 = Mat::ones(frame1.size(), frame1.type()) * 255;
			Mat white2 = Mat::ones(frame2.size(), frame2.type()) * 255;
			imshow("Camera 1 (Front/Right/Up)", white1);
			imshow("Camera 2 (Back/Left/Down)", white2);
			waitKey(100);
		}
	}
	
	cv::destroyAllWindows();
	std::cout << "Camera feed closed." << std::endl;
}

void drawPositioningGrid(Mat& frame) {
	// Draw a 3x3 grid overlay to help visualize cube positioning
	int rows = frame.rows;
	int cols = frame.cols;
	
	Scalar grid_color(100, 100, 100); // Gray color
	int thickness = 1;
	
	// Vertical lines
	for (int i = 1; i < 3; i++) {
		int x = (cols * i) / 3;
		line(frame, Point(x, 0), Point(x, rows), grid_color, thickness);
	}
	
	// Horizontal lines  
	for (int i = 1; i < 3; i++) {
		int y = (rows * i) / 3;
		line(frame, Point(0, y), Point(cols, y), grid_color, thickness);
	}
	
	// Center crosshair
	int center_x = cols / 2;
	int center_y = rows / 2;
	line(frame, Point(center_x - 10, center_y), Point(center_x + 10, center_y), 
		 Scalar(0, 255, 0), 2);
	line(frame, Point(center_x, center_y - 10), Point(center_x, center_y + 10), 
		 Scalar(0, 255, 0), 2);
}

void loadConfig(const std::string& filename) {
	std::ifstream configFile(filename);
	if (!configFile.is_open()) {
		std::cout << "Warning: Could not open config file " << filename << ". Using default values." << std::endl;
		return;
	}
	
	std::string line;
	while (std::getline(configFile, line)) {
		// Skip comments and empty lines
		if (line.empty() || line[0] == '#') {
			continue;
		}
		
		// Find the = separator
		size_t pos = line.find('=');
		if (pos == std::string::npos) {
			continue;
		}
		
		std::string key = line.substr(0, pos);
		std::string value = line.substr(pos + 1);
		
		// Parse configuration values
		if (key == "CAMERA_1_INDEX") {
			config.camera_1_index = std::stoi(value);
		} else if (key == "CAMERA_2_INDEX") {
			config.camera_2_index = std::stoi(value);
		} else if (key == "CAMERA_WIDTH") {
			config.camera_width = std::stoi(value);
		} else if (key == "CAMERA_HEIGHT") {
			config.camera_height = std::stoi(value);
		} else if (key == "CAMERA_FPS") {
			config.camera_fps = std::stoi(value);
		} else if (key == "EXPOSURE") {
			config.exposure = std::stoi(value);
		} else if (key == "GAIN") {
			config.gain = std::stoi(value);
		} else if (key == "BRIGHTNESS") {
			config.brightness = std::stoi(value);
		} else if (key == "CONTRAST") {
			config.contrast = std::stoi(value);
		} else if (key == "SATURATION") {
			config.saturation = std::stoi(value);
		}
	}
	
	configFile.close();
	std::cout << "✓ Configuration loaded from " << filename << std::endl;
	std::cout << "  Camera 1 index: " << config.camera_1_index << std::endl;
	std::cout << "  Camera 2 index: " << config.camera_2_index << std::endl;
	std::cout << "  Resolution: " << config.camera_width << "x" << config.camera_height << std::endl;
}

void initializeCameras() {
	if (camera_1) {
		delete camera_1;
		camera_1 = nullptr;
	}
	if (camera_2) {
		delete camera_2;
		camera_2 = nullptr;
	}
	
	try {
		camera_1 = new PS3EyeCamera(config.camera_height, config.camera_width, config.camera_1_index, 187);
		camera_2 = new PS3EyeCamera(config.camera_height, config.camera_width, config.camera_2_index, 187);
	} catch (const std::exception& e) {
		std::cerr << "Error initializing cameras: " << e.what() << std::endl;
		if (camera_1) {
			delete camera_1;
			camera_1 = nullptr;
		}
		if (camera_2) {
			delete camera_2;
			camera_2 = nullptr;
		}
		throw;
	}
}


void parallel_benchmark() {
	const auto start = std::chrono::high_resolution_clock::now();

	std::thread t1(detect_cam_1);
	std::thread t2(detect_cam_2);

	// #pragma omp parallel sections
	// 	{
	// #pragma omp section
	// 		detect_cam_1();
	//
	// #pragma omp section
	// 		detect_cam_2();
	// 	}
	t1.join();
	t2.join();

	// Future: add cube state assembly and validation here
	const auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	std::cout << "Dual camera time: " << duration.count() << " seconds" << std::endl;
}


bool validateCube() {
	std::map<char, int> color_counts;
	
	// Count colors from both cameras (edge/corner pieces only)
	for (const auto& color : glob_colors_cam_1) {
		color_counts[color]++;
	}
	for (const auto& color : glob_colors_cam_2) {
		color_counts[color]++;
	}
	
	// Add hardcoded center pieces (1 of each face color)
	color_counts['R']++; // Front center
	color_counts['G']++; // Right center  
	color_counts['W']++; // Up center
	color_counts['O']++; // Back center
	color_counts['B']++; // Left center
	color_counts['Y']++; // Down center
	
	// Check that we have exactly 9 of each expected color (8 detected + 1 center)
	const char expected_colors[] = {'W', 'R', 'O', 'Y', 'G', 'B'};
	bool is_valid = true;
	
	std::cout << "\n=== Cube Validation ===" << std::endl;
	std::cout << "Detected pieces: 48 total (24 detected + 6 hardcoded centers)" << std::endl;
	
	for (char color : expected_colors) {
		int detected_count = color_counts[color] - 1; // Subtract the hardcoded center
		int total_count = color_counts[color];
		
		std::cout << "Color " << color << ": " << detected_count << " detected + 1 center = " 
				  << total_count << " total";
		if (total_count == 9) {
			std::cout << " ✓" << std::endl;
		} else {
			std::cout << " ✗ (expected 9)" << std::endl;
			is_valid = false;
		}
	}
	
	// Check for unknown colors
	int unknown_count = color_counts['N'];
	if (unknown_count > 0) {
		std::cout << "Unknown/Undetected: " << unknown_count << " stickers ✗" << std::endl;
		is_valid = false;
	}
	
	if (is_valid) {
		std::cout << "✓ Cube validation PASSED - All colors detected correctly!" << std::endl;
	} else {
		std::cout << "✗ Cube validation FAILED - Color count mismatch!" << std::endl;
	}
	
	return is_valid;
}

std::string generateSolverString() {
	// Generate cube state string for solving algorithms (54 characters)
	// Standard format: Front, Right, Up, Back, Left, Down faces
	// Each face: Top row (0,1,2), Middle row (3,4,5), Bottom row (6,7,8)
	
	std::string cube_state(54, 'N'); // Initialize with 'N' (unknown)
	
	// Camera 1 handles: Front (0-8), Right (9-17), Up (18-26)
	// Camera 2 handles: Back (27-35), Left (36-44), Down (45-53)
	
	// Map detected pieces to cube positions (skipping centers at positions 4, 13, 22, 31, 40, 49)
	int detected_indices_cam1[] = {0, 1, 2, 3, 5, 6, 7, 8,     // Front face (skip 4)
								   9, 10, 11, 12, 14, 15, 16, 17, // Right face (skip 13)  
								   18, 19, 20, 21, 23, 24, 25, 26}; // Up face (skip 22)
	
	int detected_indices_cam2[] = {27, 28, 29, 30, 32, 33, 34, 35, // Back face (skip 31)
								   36, 37, 38, 39, 41, 42, 43, 44, // Left face (skip 40)
								   45, 46, 47, 48, 50, 51, 52, 53}; // Down face (skip 49)
	
	// Fill in detected pieces from camera 1
	for (int i = 0; i < 24 && i < glob_colors_cam_1.size(); i++) {
		cube_state[detected_indices_cam1[i]] = glob_colors_cam_1[i];
	}
	
	// Fill in detected pieces from camera 2  
	for (int i = 0; i < 24 && i < glob_colors_cam_2.size(); i++) {
		cube_state[detected_indices_cam2[i]] = glob_colors_cam_2[i];
	}
	
	// Fill in hardcoded center pieces
	cube_state[4] = 'R';   // Front center
	cube_state[13] = 'G';  // Right center
	cube_state[22] = 'W';  // Up center
	cube_state[31] = 'O';  // Back center
	cube_state[40] = 'B';  // Left center
	cube_state[49] = 'Y';  // Down center
	
	return cube_state;
}


std::string convertToFRUBLD(const std::string& rgbygo_string) {
	// Convert from RGBYGO color format to FRUBLD face format
	// Standard cube: F=R, R=G, U=W, B=O, L=B, D=Y
	std::string frubld_string = rgbygo_string;
	
	for (char& c : frubld_string) {
		switch (c) {
			case 'R': c = 'F'; break; // Red -> Front
			case 'G': c = 'R'; break; // Green -> Right  
			case 'W': c = 'U'; break; // White -> Up
			case 'O': c = 'B'; break; // Orange -> Back
			case 'B': c = 'L'; break; // Blue -> Left
			case 'Y': c = 'D'; break; // Yellow -> Down
			case 'N': c = 'N'; break; // Unknown stays unknown
		}
	}
	
	return frubld_string;
}

void printSolverFormat() {
	std::string cube_state = generateSolverString();
	std::string frubld_state = convertToFRUBLD(cube_state);
	
	std::cout << "\n=== Cube State for Solver ===" << std::endl;
	
	std::cout << "\nColor format (RGWYBO): " << cube_state << std::endl;
	std::cout << "Face format (FRUBLD):  " << frubld_state << std::endl;
	
	std::cout << "\nFace-by-face breakdown (Color format):" << std::endl;
	const char* face_labels[] = {"Front", "Right", "Up", "Back", "Left", "Down"};
	
	for (int face = 0; face < 6; face++) {
		std::cout << face_labels[face] << " face: ";
		for (int pos = 0; pos < 9; pos++) {
			std::cout << cube_state[face * 9 + pos];
		}
		std::cout << std::endl;
	}
	
	// Kociemba validation checks
	std::cout << "\n=== Kociemba Validation Checks ===" << std::endl;
	
	// Check 1: Exactly 54 characters
	std::cout << "String length: " << frubld_state.length() << " (should be 54)" << std::endl;
	
	// Check 2: Only valid characters (F, R, U, B, L, D)
	bool valid_chars = true;
	for (char c : frubld_state) {
		if (c != 'F' && c != 'R' && c != 'U' && c != 'B' && c != 'L' && c != 'D' && c != 'N') {
			valid_chars = false;
			break;
		}
	}
	std::cout << "Valid characters: " << (valid_chars ? "✓" : "✗") << std::endl;
	
	// Check 3: Exactly 9 of each face character
	std::map<char, int> face_counts;
	for (char c : frubld_state) {
		face_counts[c]++;
	}
	
	std::cout << "Character counts:" << std::endl;
	for (auto& pair : face_counts) {
		std::cout << "  " << pair.first << ": " << pair.second << " (should be 9)" << std::endl;
	}
	
	// Check 4: Centers must be F, R, U, B, L, D respectively
	std::cout << "Center pieces:" << std::endl;
	std::cout << "  Front center (pos 4): " << frubld_state[4] << " (should be F)" << std::endl;
	std::cout << "  Right center (pos 13): " << frubld_state[13] << " (should be R)" << std::endl;
	std::cout << "  Up center (pos 22): " << frubld_state[22] << " (should be U)" << std::endl;
	std::cout << "  Back center (pos 31): " << frubld_state[31] << " (should be B)" << std::endl;
	std::cout << "  Left center (pos 40): " << frubld_state[40] << " (should be L)" << std::endl;
	std::cout << "  Down center (pos 49): " << frubld_state[49] << " (should be D)" << std::endl;
	
	std::cout << "\n💡 Use the 'Face format (FRUBLD)' string for Kociemba!" << std::endl;
}

void visual_debug_detection() {
	if (!camera_1 || !camera_2) {
		std::cerr << "Cameras not initialized!" << std::endl;
		return;
	}
	
	// Check if position files exist
	if (points_cam_1.empty() || points_cam_2.empty()) {
		std::cerr << "No calibration points loaded. Please run position calibration first." << std::endl;
		return;
	}
	
	cv::namedWindow("Debug Camera 1", cv::WINDOW_NORMAL);
	cv::namedWindow("Debug Camera 2", cv::WINDOW_NORMAL);
	cv::resizeWindow("Debug Camera 1", 640, 480);
	cv::resizeWindow("Debug Camera 2", 640, 480);
	cv::moveWindow("Debug Camera 1", 50, 50);
	cv::moveWindow("Debug Camera 2", 720, 50);
	
	std::cout << "\n=== Visual Debug Detection Mode ===" << std::endl;
	std::cout << "Controls:" << std::endl;
	std::cout << "  SPACE = Detect colors and show on points" << std::endl;
	std::cout << "  ESC/Q = Quit" << std::endl;
	std::cout << "  R = Reset (show points without colors)" << std::endl;
	
	bool show_colors = false;
	
	// Color mapping for visualization
	std::map<char, cv::Scalar> color_map = {
		{'W', cv::Scalar(255, 255, 255)}, // White
		{'R', cv::Scalar(0, 0, 255)},     // Red
		{'O', cv::Scalar(0, 165, 255)},   // Orange
		{'Y', cv::Scalar(0, 255, 255)},   // Yellow
		{'G', cv::Scalar(0, 255, 0)},     // Green
		{'B', cv::Scalar(255, 0, 0)},     // Blue
		{'N', cv::Scalar(128, 128, 128)}  // Unknown/Gray
	};
	
	while (true) {
		cv::Mat frame1, frame2, display1, display2;
		
		// Capture frames
		camera_1->capture(frame1);
		camera_2->capture(frame2);
		
		if (frame1.empty() || frame2.empty()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
			continue;
		}
		
		display1 = frame1.clone();
		display2 = frame2.clone();
		
		// Draw calibration points
		for (int i = 0; i < points_cam_1.size(); i++) {
			cv::Point pt = points_cam_1[i];
			
			if (show_colors && i < glob_colors_cam_1.size()) {
				// Show detected color
				char detected_color = glob_colors_cam_1[i];
				cv::Scalar color = color_map[detected_color];
				
				// Draw filled circle with detected color
				cv::circle(display1, pt, 8, color, -1);
				// Draw black border for visibility
				cv::circle(display1, pt, 8, cv::Scalar(0, 0, 0), 2);
				
				// Add color text
				cv::putText(display1, std::string(1, detected_color), 
						   cv::Point(pt.x + 12, pt.y + 5),
						   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
			} else {
				// Just show calibration points
				cv::circle(display1, pt, 5, cv::Scalar(0, 255, 0), 2);
				cv::putText(display1, std::to_string(i + 1), 
						   cv::Point(pt.x + 8, pt.y - 8),
						   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
			}
		}
		
		for (int i = 0; i < points_cam_2.size(); i++) {
			cv::Point pt = points_cam_2[i];
			
			if (show_colors && i < glob_colors_cam_2.size()) {
				// Show detected color
				char detected_color = glob_colors_cam_2[i];
				cv::Scalar color = color_map[detected_color];
				
				// Draw filled circle with detected color
				cv::circle(display2, pt, 8, color, -1);
				// Draw black border for visibility
				cv::circle(display2, pt, 8, cv::Scalar(0, 0, 0), 2);
				
				// Add color text
				cv::putText(display2, std::string(1, detected_color), 
						   cv::Point(pt.x + 12, pt.y + 5),
						   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
			} else {
				// Just show calibration points
				cv::circle(display2, pt, 5, cv::Scalar(0, 255, 0), 2);
				cv::putText(display2, std::to_string(i + 1), 
						   cv::Point(pt.x + 8, pt.y - 8),
						   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
			}
		}
		
		// Add titles and instructions
		cv::putText(display1, "Camera 1 (Front/Right/Up)", cv::Point(10, 30),
				   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
		cv::putText(display2, "Camera 2 (Back/Left/Down)", cv::Point(10, 30),
				   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
		
		std::string status = show_colors ? "Showing detected colors" : "Showing calibration points";
		cv::putText(display1, status, cv::Point(10, display1.rows - 40),
				   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
		cv::putText(display2, status, cv::Point(10, display2.rows - 40),
				   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
		
		cv::putText(display1, "SPACE=Detect, R=Reset, Q=Quit", cv::Point(10, display1.rows - 10),
				   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
		cv::putText(display2, "SPACE=Detect, R=Reset, Q=Quit", cv::Point(10, display2.rows - 10),
				   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
		
		cv::imshow("Debug Camera 1", display1);
		cv::imshow("Debug Camera 2", display2);
		
		int key = cv::waitKey(30) & 0xFF;
		if (key == 27 || key == 'q' || key == 'Q') { // ESC or Q
			break;
		}
		else if (key == ' ') { // SPACE - detect colors
			std::cout << "Running color detection..." << std::endl;
			detect_cam_1();
			detect_cam_2();
			show_colors = true;
			std::cout << "Colors detected! Check the visual display." << std::endl;
		}
		else if (key == 'r' || key == 'R') { // R - reset
			show_colors = false;
			std::cout << "Reset - showing calibration points only." << std::endl;
		}
	}
	
	cv::destroyAllWindows();
	std::cout << "Visual debug mode closed." << std::endl;
}

void debug_detected_colors() {
	for (int i = 0; i < glob_colors_cam_1.size(); i++) {
		std::cout << "Point " << i << ": " << glob_colors_cam_1[i] << std::endl;
	}
	for (int i = 0; i < glob_colors_cam_2.size(); i++) {
		std::cout << "Point " << i << ": " << glob_colors_cam_2[i] << std::endl;
	}
}

int main() {
	std::cout << "\n=== Rubik's Cube Detection System ===" << std::endl;
	
	// Load configuration
	loadConfig("config.txt");
	
	// Initialize cameras once after config loading
	try {
		initializeCameras();
	} catch (const std::exception& e) {
		std::cerr << "Failed to initialize cameras. Please check your config.txt file." << std::endl;
		return 1;
	}
	
	std::cout << "\nSelect mode:" << std::endl;
	std::cout << "  c = Position calibration" << std::endl;
	std::cout << "  k = Color calibration" << std::endl;
	std::cout << "  b = Simple benchmark" << std::endl;
	std::cout << "  j = Full detection (with custom LUT)" << std::endl;
	std::cout << "  d = Show dual camera feed (positioning)" << std::endl;
	std::cout << "  v = Visual debug detection (see detection points)" << std::endl;
	std::cout << "  a = Arduino-style detection test" << std::endl;
	std::cout << "  q = Quit" << std::endl;
	std::cout << "Enter choice: ";
	
	int k = std::cin.get();
	
	try {
		if (k == 'c') {
			std::cout << "\n=== Position Calibration Mode ===" << std::endl;
			show_camera_setup_guide();
			
			std::cout << "Starting calibration for camera 1..." << std::endl;
			camera_1->calibratePosition("pos_1.txt");
			
			std::cout << "\nStarting calibration for camera 2..." << std::endl;
			camera_2->calibratePosition("pos_2.txt");
			
			std::cout << "\n✓ Position calibration completed!" << std::endl;
		} 
		else if (k == 'k') {
			std::cout << "\n=== Color Calibration Mode ===" << std::endl;
			camera_2->calibrateColors("range.txt");
		}
		else if (k == 'b') {
			std::cout << "\n=== Simple Benchmark Mode ===" << std::endl;
			init_lut();
			init_mat();
			load_position("pos_1.txt", "pos_2.txt");

			const auto dstart = std::chrono::high_resolution_clock::now();
			benchmark();
			const auto dend = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> duration = dend - dstart;
			std::cout << "Detection time: " << duration.count() << " seconds" << std::endl;
			print_colors();
			validateCube();
			printSolverFormat();
		}
		else if (k == 'j') {
			std::cout << "\n=== Full Detection Mode ===" << std::endl;
			init_lut();
			init_mat();
			load_position("pos_1.txt", "pos_2.txt");
			load_lut_from_file("range.txt");

			parallel_benchmark();
			debug_detected_colors();
			validateCube();
			printSolverFormat();
		}
		else if (k == 'd') {
			std::cout << "\n=== Dual Camera Display Mode ===" << std::endl;
			show_camera_setup_guide();
			
			// Optimize both cameras for dual operation
			camera_1->optimizeForDualCamera();
			camera_2->optimizeForDualCamera();
			
			show_dual_camera_feed(camera_1, camera_2);
		}
		else if (k == 'v') {
			std::cout << "\n=== Visual Debug Detection Mode ===" << std::endl;
			init_lut();
			init_mat();
			load_position("pos_1.txt", "pos_2.txt");
			load_lut_from_file("range.txt");
			
			visual_debug_detection();
		}
		else if (k == 'a') {
			std::cout << "\n=== Arduino-Style Detection Test ===" << std::endl;
			ArduinoStyleDetection arduino_detector;
			
			// Load position calibration
			if (!arduino_detector.loadPositions("pos_1.txt", "pos_2.txt")) {
				std::cout << "Failed to load position files. Please run position calibration first." << std::endl;
			} else {
				// Load color calibration if available
				arduino_detector.loadColorCalibration("arduino_colors.txt");
				
				std::cout << "Controls:" << std::endl;
				std::cout << "  SPACE = Detect cube" << std::endl;
				std::cout << "  C = Calibrate colors" << std::endl;
				std::cout << "  Q = Quit" << std::endl;
				
				cv::namedWindow("Arduino Detection", cv::WINDOW_NORMAL);
				cv::resizeWindow("Arduino Detection", 1200, 400);
				
				while (true) {
					cv::Mat frame1, frame2, combined;
					camera_1->capture(frame1);
					camera_2->capture(frame2);
					
					if (!frame1.empty() && !frame2.empty()) {
						cv::hconcat(frame1, frame2, combined);
						cv::putText(combined, "SPACE=Detect, C=Calibrate, Q=Quit", 
								   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
						cv::imshow("Arduino Detection", combined);
					}
					
					int key = cv::waitKey(30) & 0xFF;
					if (key == 'q' || key == 'Q') break;
					
					if (key == ' ') {
						// Run detection
						std::array<char, 54> cube_state;
						auto start = std::chrono::high_resolution_clock::now();
						int result = arduino_detector.detectCube(frame1, frame2, cube_state);
						auto end = std::chrono::high_resolution_clock::now();
						auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
						
						std::cout << "\n--- Detection Results ---" << std::endl;
						std::cout << "Detection time: " << duration.count() << " ms" << std::endl;
						std::cout << "Result: " << (result ? "SUCCESS" : "FAILED") << std::endl;
						arduino_detector.printCubeState(cube_state);
					}
					
					if (key == 'c' || key == 'C') {
						// Run color calibration with capture callback
						auto captureCallback = [&](cv::Mat& frame1, cv::Mat& frame2) -> bool {
							if (camera_1) camera_1->capture(frame1);
							if (camera_2) camera_2->capture(frame2);
							return !frame1.empty() || !frame2.empty();
						};
						arduino_detector.calibrateColors(captureCallback);
					}
				}
				
				cv::destroyAllWindows();
			}
		}
		else if (k == 'q') {
			std::cout << "Goodbye!" << std::endl;
		}
		else {
			std::cout << "Invalid option. Please try again." << std::endl;
		}
	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
	}

	// Cleanup
	if (camera_1) {
		delete camera_1;
		camera_1 = nullptr;
	}
	if (camera_2) {
		delete camera_2;
		camera_2 = nullptr;
	}
	
	return 0;
}