#include "util.h"

std::vector<std::string> split(const std::string &text, const char &separator){
  std::vector<std::string> strings;
  std::istringstream f(text);
  std::string s;    
  while (getline(f, s, separator)) {
    strings.push_back(s);
  }
  return strings;
}

std::tuple<std::vector<cv::Rect2f>, std::vector<int>> parse_box_prompts(const std::string &boxes){
  std::vector<cv::Rect2f> rects;
  std::vector<int> labels;
  std::vector<std::string> split_semicolon = split(boxes, ';');
  for(int i = 0; i < split_semicolon.size(); i++){
    std::string coords;
    int label = 1;
    if(split_semicolon[i].rfind("pos:", 0) == 0) {
      coords = split_semicolon[i].substr(4);
      label = 1;
    }else if(split_semicolon[i].rfind("neg:", 0) == 0) {
      coords = split_semicolon[i].substr(4);
      label = 0;
    }
    std::vector<std::string> split_comma = split(coords, ',');
    if(split_comma.size() == 4){
      cv::Rect2f rect = cv::Rect2f(
        std::stof(split_comma[0]),
        std::stof(split_comma[1]),
        std::stof(split_comma[2]),
        std::stof(split_comma[3]));
      rects.push_back(rect);
      labels.push_back(label);
    }
  }
  return std::make_tuple(rects, labels);
}

void normalizeRects(std::vector<cv::Rect2f> *rects, const cv::Size &imageSize){
  for(int i = 0; i < (*rects).size(); i++){
    (*rects)[i].x /= imageSize.width;
    (*rects)[i].y /= imageSize.height;
    (*rects)[i].width /= imageSize.width;
    (*rects)[i].height /= imageSize.height;
    (*rects)[i].x = (*rects)[i].x + (*rects)[i].width / 2;
    (*rects)[i].y = (*rects)[i].y + (*rects)[i].height / 2;
  }
}

bool modelExists(const std::string& modelPath){
  std::ifstream f(modelPath);
  if (!f.good()) {
    return false;
  }
  return true;
}

std::string LoadBytesFromFile(const std::string& path) {
  std::string data;
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open " << path << std::endl;
    return data;
  }
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

void printShape(const std::vector<int64_t> &shape){
  std::string text = "";
  for(int i = 0; i < shape.size(); i++){
    text = text + std::to_string(shape[i]) + " ";
  }
  std::cout << text << std::endl;
}

int getShapeSize(const std::vector<int64_t> &shape){
  int size = 1;
  for(int i = 0; i < shape.size(); i++){
    size *= shape[i];
  }
  return size;
}

std::vector<const char*> getInputNames(std::unique_ptr<Ort::Session> &session){
  std::vector<const char*> inputNames;
  Ort::AllocatorWithDefaultOptions allocator;
  for (size_t i = 0; i < session->GetInputCount(); ++i) {
    Ort::AllocatedStringPtr name_Ptr = session->GetInputNameAllocated(i, allocator);
    char* name = name_Ptr.get();
    size_t name_length = strlen(name) + 1;
    char* name_new = new char[name_length];
    strncpy(name_new, name, name_length);
    inputNames.push_back(name_new);
  }
  return inputNames;
}

std::vector<const char*> getOutputNames(std::unique_ptr<Ort::Session> &session){
  std::vector<const char*> outputNames;
  Ort::AllocatorWithDefaultOptions allocator;
  for (size_t i = 0; i < session->GetOutputCount(); ++i) {
    Ort::AllocatedStringPtr name_Ptr = session->GetOutputNameAllocated(i, allocator);
    char* name = name_Ptr.get();
    size_t name_length = strlen(name) + 1;
    char* name_new = new char[name_length];
    strncpy(name_new, name, name_length);
    outputNames.push_back(name_new);
  }
  return outputNames;
}

std::vector<int> sort_indexes(const std::vector<float> &v) {
  std::vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  stable_sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v[i1] > v[i2];});
  return idx;
}

float calc_iou(const std::vector<int> &box1, const std::vector<int> &box2) {
  int inter_x1 = std::max(box1[0], box2[0]);
  int inter_y1 = std::max(box1[1], box2[1]);
  int inter_x2 = std::min(box1[0] + box1[2], box2[0] + box2[2]);
  int inter_y2 = std::min(box1[1] + box1[3], box2[1] + box2[3]);
  int inter_width = std::max(0, inter_x2 - inter_x1);
  int inter_height = std::max(0, inter_y2 - inter_y1);
  int inter_area = inter_width * inter_height;
  if (inter_area == 0) {
      return 0;
  }
  int area1 = box1[2] * box1[3];
  int area2 = box2[2] * box2[3];
  int union_area = area1 + area2 - inter_area;
  return (float)inter_area / union_area;
}

bool can_append_box(const std::vector<int> box, const std::vector<int> &boxes){
  float threshold = 0.9;
  int num = (int)boxes.size() / 4;
  for(int i = 0; i < num; i++){
    std::vector<int> box_tmp;
    for(int j = 0; j < 4; j++){
      box_tmp.push_back(boxes[4 * i + j]);
    }
    float iou = calc_iou(box, box_tmp);
    if(iou > threshold){
      return false;
    }
  }
  return true;
}
