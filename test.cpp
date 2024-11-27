#include <opencv2/opencv.hpp>
#define STRIP_FLAG_HELP 1
#include <gflags/gflags.h>
#include <thread>
#include "sam.h"

DEFINE_string(encoder, "sam2.1_tiny/sam2.1_tiny_preprocess.onnx", "Path to the encoder model");
DEFINE_string(decoder, "sam2.1_tiny/sam2.1_tiny.onnx", "Path to the decoder model");
DEFINE_string(image, "david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg", "Path to the image");
DEFINE_string(device, "cpu", "cpu or cuda:0(1,2,3...)");
DEFINE_bool(h, false, "Show help");

int main(int argc, char** argv) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if(FLAGS_h){
    std::cout<<"Example: ./build/sam_cpp_test -encoder=\"sam2.1_tiny/sam2.1_tiny_preprocess.onnx\" "
               "-decoder=\"sam2.1_tiny/sam2.1_tiny.onnx\" "
               "-image=\"david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg\" -device=\"cpu\""<< std::endl;
    return 0;
  }
  Sam sam;
  if(FLAGS_encoder.find("sam2") != std::string::npos){
    sam.changeMode(SAM2);
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
  begin = std::chrono::steady_clock::now();
  std::list<cv::Rect> rects;
  std::list<cv::Point> points;
  std::vector<float> inputPointValues, inputLabelValues;
  int previousMaskIdx = -1;
  bool isNextGetMask = true;
  cv::Mat mask;
  
  cv::Rect rect1 = cv::Rect(1215 * inputSize.width / imageSize.width,
                            125 * inputSize.height / imageSize.height,
                            508 * inputSize.width / imageSize.width,
                            436 * inputSize.height / imageSize.height);
  cv::Rect rect2 = cv::Rect(890 * inputSize.width / imageSize.width,
                            85 * inputSize.height / imageSize.height,
                            315 * inputSize.width / imageSize.width,
                            460 * inputSize.height / imageSize.height);
  if(sam.getMode() == SAM2){
    rects.push_back(rect1);
    rects.push_back(rect2);
    sam.setRectsLabels(rects, &inputPointValues, &inputLabelValues);
    int batchNum = (int)rects.size();
    mask = sam.getMaskBatch(inputPointValues, inputLabelValues, batchNum, imageSize);
    cv::imwrite("mask_box_batch.png", mask);
    inputPointValues.resize(0);
    inputLabelValues.resize(0);
    rects.resize(0);
  }
  
  rects.push_back(rect1);
  sam.setRectsLabels(rects, &inputPointValues, &inputLabelValues);
  mask = sam.getMask(inputPointValues, inputLabelValues, imageSize, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::imwrite("mask_box1.png", mask);
  inputPointValues.resize(0);
  inputLabelValues.resize(0);
  rects.resize(0);
  
  rects.push_back(rect2);
  sam.setRectsLabels(rects, &inputPointValues, &inputLabelValues);
  mask = sam.getMask(inputPointValues, inputLabelValues, imageSize, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::imwrite("mask_box2.png", mask);
  inputPointValues.resize(0);
  inputLabelValues.resize(0);
  rects.resize(0);
  
  cv::Point point1 = cv::Point(1255 * inputSize.width / imageSize.width,
                               360 * inputSize.height / imageSize.height);
  cv::Point point2 = cv::Point(1500 * inputSize.width / imageSize.width,
                               420 * inputSize.height / imageSize.height);
  points.push_back(point1);
  points.push_back(point2);
  sam.setPointsLabels(points, 1, &inputPointValues, &inputLabelValues);
  mask = sam.getMask(inputPointValues, inputLabelValues, imageSize, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::imwrite("mask_point12.png", mask);
  inputPointValues.resize(0);
  inputLabelValues.resize(0);
  points.resize(0);
  
  points.push_back(point2);
  sam.setPointsLabels(points, 1, &inputPointValues, &inputLabelValues);
  mask = sam.getMask(inputPointValues, inputLabelValues, imageSize, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::imwrite("mask_point2.png", mask);
  inputPointValues.resize(0);
  inputLabelValues.resize(0);
  points.resize(0);

  points.push_back(point1);
  sam.setPointsLabels(points, 1, &inputPointValues, &inputLabelValues);
  mask = sam.getMask(inputPointValues, inputLabelValues, imageSize, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::imwrite("mask_point1.png", mask);
  inputPointValues.resize(0);
  inputLabelValues.resize(0);
  
  isNextGetMask = false;
  points.push_back(point2);
  sam.setPointsLabels(points, 1, &inputPointValues, &inputLabelValues);
  mask = sam.getMask(inputPointValues, inputLabelValues, imageSize, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::imwrite("mask_point1_then_point2.png", mask);
  inputPointValues.resize(0);
  inputLabelValues.resize(0);
  points.resize(0);
  
  end = std::chrono::steady_clock::now();
  std::cout << "sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;
  return 0;
}
