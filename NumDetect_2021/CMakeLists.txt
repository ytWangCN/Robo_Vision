cmake_minimum_required (VERSION 2.8)
project(Vision_NumDetect)

set(CMAKE_PREFIX_PATH ~/libtorch)

find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(modules/LeNet5)
add_subdirectory(modules/Libtorch)
add_subdirectory(modules/DNN)
add_subdirectory(RMVideoCapture)
add_subdirectory(SerialPort)


add_executable(Vision_NumDetect main.cpp)

target_link_libraries(Vision_NumDetect ${TORCH_LIBRARIES} ${OpenCV_LIBS} LeNet5 Libtorch DNN RMVideoCapture)

set_property(TARGET Vision_NumDetect PROPERTY CXX_STANDARD 14)
