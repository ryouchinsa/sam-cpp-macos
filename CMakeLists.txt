cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
project(SamCPP)
find_package(OpenCV CONFIG REQUIRED)

message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

add_library(sam_cpp_lib SHARED sam.h sam.cpp)
target_include_directories(
  sam_cpp_lib PUBLIC 
  /Users/ryo/Downloads/onnxruntime-osx-universal2-1.17.1/include
)
target_link_libraries(
  sam_cpp_lib PUBLIC
  /Users/ryo/Downloads/onnxruntime-osx-universal2-1.17.1/lib/libonnxruntime.dylib
  ${OpenCV_LIBS}
)

add_executable(sam_cpp_test test.cpp)
target_link_libraries(
  sam_cpp_test PRIVATE
  sam_cpp_lib
)

