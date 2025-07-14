//
// Created by soma on 24/06/25.
//

#ifndef PS3EYECAMERA_H
#define PS3EYECAMERA_H

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <chrono>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>

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
};



#endif //PS3EYECAMERA_H
