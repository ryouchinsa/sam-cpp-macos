#include <opencv2/opencv.hpp>
#include <thread>
#include "sam.h"

int main(int argc, char** argv) {
  Sam sam;
  std::string modelName = "mobile_sam";
  std::string pathEncoder = modelName + "/" + modelName + "_preprocess.onnx";
  std::string pathDecoder = modelName + "/" + modelName + ".onnx";
  std::cout<<"loadModel started"<<std::endl;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  bool successLoadModel = sam.loadModel(pathEncoder, pathDecoder, std::thread::hardware_concurrency());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time difference (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;
  if(!successLoadModel){
    std::cout<<"loadModel error"<<std::endl;
    return 1;
  }
  std::string imagePath = "david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg";
  cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
  auto inputSize = sam.getInputSize();
  cv::resize(image, image, inputSize);
  std::cout<<"preprocessImage started"<<std::endl;
  begin = std::chrono::steady_clock::now();
  bool successPreprocessImage = sam.preprocessImage(image);
  end = std::chrono::steady_clock::now();
  std::cout << "Time difference (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;
  if(!successPreprocessImage){
    std::cout<<"preprocessImage error"<<std::endl;
    return 1;
  }
  std::cout<<"getMask started"<<std::endl;
  begin = std::chrono::steady_clock::now();
  std::list<cv::Point> points, nagativePoints;
  cv::Rect roi;
  // 1st object and 1st click
  int previousMaskIdx = -1; // An index to use the previous mask result
  bool isNextGetMask = true; // Set true when start labeling a new object
  points.push_back({810, 550});
  cv::Mat mask = sam.getMask(points, nagativePoints, roi, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::imwrite("mask-object1-click1.png", mask);
  // 1st object and 2nd click
  isNextGetMask = false;
  points.push_back({940, 410});
  mask = sam.getMask(points, nagativePoints, roi, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::imwrite("mask-object1-click2.png", mask);
  end = std::chrono::steady_clock::now();
  std::cout << "Time difference (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;
  return 0;
}
