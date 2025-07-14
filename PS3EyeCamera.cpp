//
// Created by soma on 24/06/25.
//

#include "PS3EyeCamera.h"


// These can be file-static as they are only used by the calibration function
static int h_min = 0, s_min = 0, v_min = 0;
static int h_max = 179, s_max = 255, v_max = 255;

// A simple callback function for the trackbars (does nothing, but is required)
static void on_trackbar(int, void*) {}

void PS3EyeCamera::calibrateColors(const std::string& output_filename) {
    std::ofstream outfile(output_filename, std::ios::app);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << output_filename << " for writing." << std::endl;
        return;
    }

    const char* colors[] = {"White", "Red", "Orange", "Yellow", "Green", "Blue"};
    const char color_chars[] = {'W', 'R', 'O', 'Y', 'G', 'B'};

    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Mask", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Trackbars", cv::WINDOW_AUTOSIZE);

    // Create trackbars
    cv::createTrackbar("H_MIN", "Trackbars", &h_min, 179, on_trackbar);
    cv::createTrackbar("H_MAX", "Trackbars", &h_max, 179, on_trackbar);
    cv::createTrackbar("S_MIN", "Trackbars", &s_min, 255, on_trackbar);
    cv::createTrackbar("S_MAX", "Trackbars", &s_max, 255, on_trackbar);
    cv::createTrackbar("V_MIN", "Trackbars", &v_min, 255, on_trackbar);
    cv::createTrackbar("V_MAX", "Trackbars", &v_max, 255, on_trackbar);

    for (int i = 0; i < 6; ++i) {
        std::cout << "Calibrating for color: " << colors[i] << ". Adjust trackbars and press 's' to save." << std::endl;

        while (true) {
            cv::Mat frame, hsv_frame, mask;
            this->capture(frame); // Use the class's capture method
            cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV);

            // Create mask from trackbar values
            cv::Scalar lower_bound(h_min, s_min, v_min);
            cv::Scalar upper_bound(h_max, s_max, v_max);
            cv::inRange(hsv_frame, lower_bound, upper_bound, mask);

            imshow("Original", frame);
            imshow("Mask", mask);

            int key = cv::waitKey(30);
            if (key == 's') {
                // Save the calibrated values to the file
                outfile << color_chars[i] << " " << h_min << " " << h_max << " "
                        << s_min << " " << s_max << " " << v_min << " " << v_max << std::endl;
                std::cout << colors[i] << " range saved." << std::endl;
                break; // Move to the next color
            }
            if (key == 'q') { // Allow quitting early
                outfile.close();
                cv::destroyAllWindows();
                return;
            }
        }
    }

    outfile.close();
    cv::destroyAllWindows();
	std::cout << "Calibration complete. Values saved to " << output_filename << std::endl;
}

 PS3EyeCamera::PS3EyeCamera(int height, int width, int index, int fps) {
 	video_capture.open(index, cv::CAP_V4L2);
 	video_capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
 	video_capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
 	video_capture.set(cv::CAP_PROP_FPS, fps);
 	video_capture.set(cv::CAP_PROP_BUFFERSIZE, 1);


 	video_capture.set(cv::CAP_PROP_AUTO_WB, 1); // Disable Auto White Balance
 	video_capture.set(cv::CAP_PROP_AUTO_EXPOSURE, 0); // Disable Auto Exposure
 	video_capture.set(cv::CAP_PROP_EXPOSURE, 15);
 	video_capture.set(cv::CAP_PROP_GAIN, 10); // Set Gain (adjust as needed)
 	video_capture.set(cv::CAP_PROP_BRIGHTNESS, 15); // Set Brightness (adjust as needed)
 	video_capture.set(cv::CAP_PROP_CONTRAST, 9); // Set Contrast (adjust as needed)
 	video_capture.set(cv::CAP_PROP_SATURATION, 60);

 	if (!video_capture.isOpened()) {
		std::cerr << "Camera could not be opened" << std::endl;
 		throw std::exception();
 	}

 	// Warmup with 5 frames
 	cv::Mat temp_frames;
 	for (int i = 0; i < 5; i++) {
 		video_capture.read(temp_frames);
 	}
 }

void positionMouseCallback(int event, int x, int y, int flags, void *userdata) {
 	if (event == cv::EVENT_LBUTTONDOWN) {

 	}
 }


bool PS3EyeCamera::calibratePosition(const std::string &filename) {
	 position_out_file.open(filename, std::ios::app);
	 if (!position_out_file.is_open()) {
		 std::cerr << "Error: Could not open pos.txt for writing." << std::endl;
		 return false;
	 }
	 cv::Mat frame;
	 video_capture.read(frame);
	 cv::namedWindow("calibration", cv::WINDOW_NORMAL);
	 cv::resizeWindow("calibration", 1280, 960);

	 for (int i = 0; i < 6; i++) {
		 for (int j = 0; j < 9; j++) {
			 std::cout << "Click on face " << i << " facelet " << j;
		 }
	 }
	 cv::setMouseCallback("calibration", positionMouseCallback, this);
	 imshow("calibration", frame);
	 int k = cv::waitKey(0);
	 if (k == 'q') {
		 position_out_file.close();
		 return true;
	 }
 }

void PS3EyeCamera::capture(cv::Mat &frame) {
	video_capture.read(frame);
 }
