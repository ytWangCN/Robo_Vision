# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_LENET5 变量

aux_source_directory(. DIR_LENET5)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# 生成链接库
add_library (LeNet5 ${DIR_LENET5})

target_include_directories(LeNet5 PUBLIC include)
target_link_libraries(LeNet5 ${TORCH_LIBRARIES})

