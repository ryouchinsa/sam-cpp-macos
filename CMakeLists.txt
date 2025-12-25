cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
project(SamCPP)
find_package(OpenCV CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)

add_library(sam_cpp_lib SHARED sam.h sam.cpp util.h util.cpp)
if (APPLE)
  set(onnxruntime_lib ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.dylib)
else()
  set(onnxruntime_lib ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so)
endif()

target_include_directories(
  sam_cpp_lib PUBLIC 
  ${ONNXRUNTIME_ROOT_DIR}/include
)

target_link_libraries(
  sam_cpp_lib PUBLIC
  ${onnxruntime_lib}
  ${OpenCV_LIBS}
  gflags
)

add_executable(sam_cpp_test test.cpp)
target_link_libraries(
  sam_cpp_test PRIVATE
  sam_cpp_lib
)

