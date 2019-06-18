cmake_minimum_required(VERSION 2.6)
project(AIProject)

#The version number
set(AIProject_VERSION_MAJOR 1)
set(AIProject_VERSION_MINOR 0)

#output path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../bin/)

#cuda required
find_package(CUDA REQUIRED)

#opencv required
find_package(OpenCV REQUIRED)

#gcc flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11")

#set tensorrt libs
set(TENSORRT_LIBS libnvcaffe_parser.so libnvinfer.so libnvinfer_plugin.so libnvparsers.so)

include_directories(/usr/local/cuda-10.0/targets/x86_64-linux/include/)
include_directories(/usr/local/cuda-10.0/include/)

link_directories(/usr/lib/x86_64-linux-gnu/)
link_directories(/usr/local/lib/)
link_directories(/usr/local/cuda-10.0/lib64/)
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(/path/to/tensorrtlib/)

add_executable(AIProject  
    ./test.cpp 
)

target_link_libraries(AIProject GammaContext ${OpenCV_LIBS}
 cudart cublas curand cudnn pthread
    ${TENSORRT_LIBS} 
    pthread stdc++fs)
    
