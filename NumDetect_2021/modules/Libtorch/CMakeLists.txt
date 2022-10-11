# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_TORCH 变量
aux_source_directory(. DIR_TORCH)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# 生成链接库
add_library (Libtorch ${DIR_TORCH})

target_include_directories(Libtorch PUBLIC include)
target_link_libraries(Libtorch ${TORCH_LIBRARIES})
