#ifndef SAMCPP__SAM_H_
#define SAMCPP__SAM_H_

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <list>
#include <fstream>
#include <iostream>

enum SamMode {
  SAM,
  EfficientSAM,
};

class Sam {
  std::unique_ptr<Ort::Session> sessionEncoder, sessionDecoder;
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
  Ort::SessionOptions sessionOptions[2];
  Ort::RunOptions runOptionsEncoder;
  Ort::MemoryInfo memoryInfo{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  std::vector<int64_t> inputShapeEncoder, outputShapeEncoder;
  std::vector<float> outputTensorValuesEncoder;
  std::vector<std::vector<float>> previousMasks;
  SamMode mode = SAM;
  bool loadingModel = false;
  bool preprocessing = false;
  bool terminating = false;
  
 public:
  Sam();
  ~Sam();
  bool clearLoadModel();
  void clearPreviousMasks();
  void resizePreviousMasks(int previousMaskIdx);
  void terminatePreprocessing();
  void changeMode(SamMode modeTo);
  bool loadModel(const std::string& encoderPath, const std::string& decoderPath, int threadsNumber);
  void loadingStart();
  void loadingEnd();
  cv::Size getInputSize();
  bool preprocessImage(const cv::Mat& image);
  bool preprocessImageEfficientSAM(const cv::Mat& image);
  void preprocessingStart();
  void preprocessingEnd();
  cv::Mat getMask(const std::list<cv::Point>& points, const std::list<cv::Point>& negativePoints, const std::list<cv::Rect> &rects, int previousMaskIdx, bool isNextGetMask);
  cv::Mat getMaskEfficientSAM(const std::list<cv::Point>& points, const std::list<cv::Rect> &rects);
  std::vector<const char*> getInputNamesEncoder();
  std::vector<const char*> getOutputNamesEncoder();
  std::vector<const char*> getInputNamesDecoder();
  std::vector<const char*> getOutputNamesDecoder();
  
};

#endif
