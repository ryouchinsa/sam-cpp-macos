#ifndef SAMCPP__SAM_H_
#define SAMCPP__SAM_H_

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <list>
#include <fstream>
#include <iostream>

enum SamMode {
  SAM,
  SAM2
};

class Sam {
  std::unique_ptr<Ort::Session> sessionEncoder, sessionDecoder;
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
  Ort::SessionOptions sessionOptions[2];
  Ort::RunOptions runOptionsEncoder;
  Ort::MemoryInfo memoryInfo{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  std::vector<int64_t> inputShapeEncoder, outputShapeEncoder, highResFeatures1Shape, highResFeatures2Shape;
  std::vector<float> outputTensorValuesEncoder, highResFeatures1, highResFeatures2;
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
  SamMode getMode();
  bool loadModel(const std::string& encoderPath, const std::string& decoderPath, int threadsNumber, std::string device = "cpu");
  void loadingStart();
  void loadingEnd();
  cv::Size getInputSize();
  bool preprocessImage(const cv::Mat& image);
  void preprocessingStart();
  void preprocessingEnd();
  void setRectsLabels(const std::list<cv::Rect> &rects, std::vector<float> *inputPointValues, std::vector<float> *inputLabelValues);
  void setPointsLabels(const std::list<cv::Point>& points, int label, std::vector<float> *inputPointValues, std::vector<float> *inputLabelValues);
  void setDecorderTensorsEmbeddings(std::vector<Ort::Value> *inputTensors);
  void setDecorderTensorsPointsLabels(std::vector<float> &inputPointValues, std::vector<float> &inputLabelValues, int batchNum, int numPoints, std::vector<Ort::Value> *inputTensors);
  void setDecorderTensorsMaskInput(const size_t maskInputSize, float *maskInputValues, float *hasMaskValues, std::vector<float> &previousMaskInputValues, std::vector<Ort::Value> *inputTensors);
  cv::Mat setDecorderTensorsImageSize(std::vector<int64_t> &orig_im_size_values_int64, std::vector<float> &orig_im_size_values_float, std::vector<Ort::Value> *inputTensors);
  cv::Mat getMaskBatch(std::vector<float> &inputPointValues, std::vector<float> &inputLabelValues, int batchNum, const cv::Size &imageSize);
  cv::Mat getMask(std::vector<float> &inputPointValues, std::vector<float> &inputLabelValues, const cv::Size &imageSize, int previousMaskIdx, bool isNextGetMask);
};

#endif
