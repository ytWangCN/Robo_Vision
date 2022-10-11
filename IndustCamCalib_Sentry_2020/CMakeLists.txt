cmake_minimum_required(VERSION 3.0.0)
project(calibrate VERSION 0.1.0)
include_directories(/home/joker/Desktop/2020_Armor/RMVideoCapture/include)
include_directories(/home/joker/Desktop/2020_Armor/serialport/include)

find_package(OpenCV 4.1 REQUIRED)

add_subdirectory(RMVideoCapture)
add_executable(calibrate main.cpp)

target_link_libraries(calibrate ${OpenCV_LIBS} libMVSDK.so RMVideoCapture)
