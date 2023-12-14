#ifndef SAMCPP__SAM_H_
#define SAMCPP__SAM_H_

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <list>
#include <fstream>
#include <iostream>

class Sam {
  std::unique_ptr<Ort::Session> sessionPre, sessionSam;
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
  Ort::SessionOptions sessionOptions[2];
  Ort::RunOptions runOptionsPre;
  Ort::MemoryInfo memoryInfo{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  std::vector<int64_t> inputShapePre, outputShapePre;
  std::vector<float> outputTensorValuesPre;
  std::vector<std::vector<float>> previousMasks;
  const char *inputNamesSam[6]{"image_embeddings", "point_coords", "point_labels",
                               "mask_input", "has_mask_input", "orig_im_size"},
  *outputNamesSam[3]{"masks", "iou_predictions", "low_res_masks"};
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
  bool loadModel(const std::string& preModelPath, const std::string& samModelPath, int threadsNumber);
  void loadingStart();
  void loadingEnd();
  cv::Size getInputSize();
  bool preprocessImage(const cv::Mat& image);
  void preprocessingStart();
  void preprocessingEnd();
  cv::Mat getMask(const std::list<cv::Point>& points, const std::list<cv::Point>& negativePoints, const cv::Rect& roi, int previousMaskIdx, bool isNextGetMask);
};

#endif
