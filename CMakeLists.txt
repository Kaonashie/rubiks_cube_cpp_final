cmake_minimum_required(VERSION 3.31)
project(rubiks_cube_cpp_final)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenCV REQUIRED)

add_executable(rubiks_cube_cpp_final main.cpp arduino_detection.cpp)



target_link_libraries(rubiks_cube_cpp_final ${OpenCV_LIBS})