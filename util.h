#ifndef UTIL_CPP_H_
#define UTIL_CPP_H_

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <list>
#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>
#include <algorithm> 

std::vector<std::string> split(const std::string &text, const char &separator);
std::tuple<std::vector<cv::Rect2f>, std::vector<int>> parse_box_prompts(const std::string &boxes);
void normalizeRects(std::vector<cv::Rect2f> *rects, const cv::Size &imageSize);
bool modelExists(const std::string& modelPath);
std::string LoadBytesFromFile(const std::string& path);
void printShape(const std::vector<int64_t> &shape);
int getShapeSize(const std::vector<int64_t> &shape);
std::vector<const char*> getInputNames(std::unique_ptr<Ort::Session> &session);
std::vector<const char*> getOutputNames(std::unique_ptr<Ort::Session> &session);
std::vector<int> sort_indexes(const std::vector<float> &v);
float calc_iou(const std::vector<int> &box1, const std::vector<int> &box2);
bool can_append_box(const std::vector<int> box, const std::vector<int> &boxes);

#endif
