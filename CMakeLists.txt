cmake_minimum_required(VERSION 3.10)
project(SuperSamplingInference)

set(CMAKE_CXX_STANDARD 17)

# 1. Use absolute path for ncnn to avoid '~' expansion issues in CMake
set(ncnn_DIR "/home/raskoll/repos/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

# 2. Find OpenCV
find_package(OpenCV REQUIRED)

# 3. This line fixes the "opencv2/opencv.hpp: No such file" error
# It adds /usr/include/opencv4 to the compiler search path
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(upscaler main.cpp)

# 4. Use ${OpenCV_LIBS} which contains the list of all OpenCV library files
target_link_libraries(upscaler ncnn ${OpenCV_LIBS})
