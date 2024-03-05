#include <opencv2/opencv.hpp>
#include <thread>
#include "sam.h"

int main(int argc, char** argv) {
  bool isEfficientSAM = false;
  Sam sam;
  std::string modelName = "mobile_sam";
  if(isEfficientSAM){
    sam.changeMode(EfficientSAM);
    modelName = "efficientsam_s";
  }
  std::string pathEncoder = modelName + "/" + modelName + "_preprocess.onnx";
  std::string pathDecoder = modelName + "/" + modelName + ".onnx";
  std::cout<<"loadModel started"<<std::endl;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  bool successLoadModel = sam.loadModel(pathEncoder, pathDecoder, std::thread::hardware_concurrency());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;
  if(!successLoadModel){
    std::cout<<"loadModel error"<<std::endl;
    return 1;
  }
  std::string imagePath = "david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg";
  cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
  cv::Size imageSize = cv::Size(image.cols, image.rows);
  cv::Size inputSize = sam.getInputSize();
  cv::resize(image, image, inputSize);
  std::cout<<"preprocessImage started"<<std::endl;
  begin = std::chrono::steady_clock::now();
  bool successPreprocessImage = sam.preprocessImage(image);
  end = std::chrono::steady_clock::now();
  std::cout << "sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;
  if(!successPreprocessImage){
    std::cout<<"preprocessImage error"<<std::endl;
    return 1;
  }
  std::cout<<"getMask started"<<std::endl;
  begin = std::chrono::steady_clock::now();
  std::list<cv::Point> points, nagativePoints;
  std::list<cv::Rect> rects;
  // box
  int previousMaskIdx = -1; // An index to use the previous mask result
  bool isNextGetMask = true; // Set true when start labeling a new object
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
  end = std::chrono::steady_clock::now();
  std::cout << "sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;
  return 0;
}
