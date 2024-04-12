#include <opencv2/opencv.hpp>
#define STRIP_FLAG_HELP 1
#include <gflags/gflags.h>
#include <thread>
#include "sam.h"

DEFINE_string(encoder, "mobile_sam/mobile_sam_preprocess.onnx", "Path to the encoder model");
DEFINE_string(decoder, "mobile_sam/mobile_sam.onnx", "Path to the decoder model");
DEFINE_string(image, "david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg", "Path to the image");
DEFINE_string(device, "cpu", "cpu or cuda:0(1,2,3...)");
DEFINE_bool(h, false, "Show help");

int main(int argc, char** argv) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if(FLAGS_h){
    std::cout<<"Example: ./build/sam_cpp_test -encoder=\"mobile_sam/mobile_sam_preprocess.onnx\" "
               "-decoder=\"mobile_sam/mobile_sam.onnx\" "
               "-image=\"david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg\" -device=\"cpu\""<< std::endl;
    return 0;
  }
  Sam sam;
  if(FLAGS_encoder.find("sam_hq") != std::string::npos){
    sam.changeMode(HQSAM);
  }else if(FLAGS_encoder.find("efficientsam") != std::string::npos){
    sam.changeMode(EfficientSAM);
  }else if(FLAGS_encoder.find("edge_sam") != std::string::npos){
    sam.changeMode(EdgeSAM);
  }
  std::cout<<"loadModel started"<<std::endl;
  bool successLoadModel = sam.loadModel(FLAGS_encoder, FLAGS_decoder, std::thread::hardware_concurrency(), FLAGS_device);
  if(!successLoadModel){
    std::cout<<"loadModel error"<<std::endl;
    return 1;
  }
  std::cout<<"preprocessImage started"<<std::endl;
  cv::Mat image = cv::imread(FLAGS_image, cv::IMREAD_COLOR);
  cv::Size imageSize = cv::Size(image.cols, image.rows);
  cv::Size inputSize = sam.getInputSize();
  cv::resize(image, image, inputSize);
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  bool successPreprocessImage = sam.preprocessImage(image);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;
  if(!successPreprocessImage){
    std::cout<<"preprocessImage error"<<std::endl;
    return 1;
  }
  std::cout<<"getMask started"<<std::endl;
  std::list<cv::Point> points, nagativePoints;
  std::list<cv::Rect> rects;
  // box
  int previousMaskIdx = -1;
  bool isNextGetMask = true;
  cv::Rect rect = cv::Rect(1215 * inputSize.width / imageSize.width,
                           125 * inputSize.height / imageSize.height,
                           508 * inputSize.width / imageSize.width,
                           436 * inputSize.height / imageSize.height);
  rects.push_back(rect);
  cv::Mat mask = sam.getMask(points, nagativePoints, rects, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::resize(mask, mask, imageSize, 0, 0, cv::INTER_NEAREST);
  cv::imwrite("mask-box.png", mask);
  // positive point
  isNextGetMask = false;
  cv::Point point = cv::Point(1255 * inputSize.width / imageSize.width,
                              360 * inputSize.height / imageSize.height);
  points.push_back(point);
  mask = sam.getMask(points, nagativePoints, rects, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::resize(mask, mask, imageSize, 0, 0, cv::INTER_NEAREST);
  cv::imwrite("mask-positive_point.png", mask);
  return 0;
}
