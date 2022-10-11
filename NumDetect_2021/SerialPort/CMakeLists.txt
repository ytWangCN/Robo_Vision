cmake_minimum_required (VERSION 3.10)

add_library(SerialPort STATIC SerialPort.cpp)

target_include_directories(SerialPort PUBLIC include)
