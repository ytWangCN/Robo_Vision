find_package(OpenCV 4.1 REQUIRED core)

add_library(RMVideoCapture STATIC 
    RMVideoCapture.cpp)

target_link_libraries(RMVideoCapture ${OpenCV_LIBS} libMVSDK.so)
target_include_directories(RMVideoCapture PUBLIC include)
