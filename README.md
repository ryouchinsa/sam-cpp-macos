## Segment Anything CPP Wrapper for macOS

This code is originated from [Segment Anything CPP Wrapper](https://github.com/dinglufe/segment-anything-cpp-wrapper) and implemented on macOS app [RectLabel](https://rectlabel.com). We customized the original code so that getMask() uses the previous mask result called as low_res_logits and retain the previous mask array for undo/redo actions. 

![sam_polygon](https://github.com/ryouchinsa/sam-cpp-macos/assets/1954306/4640e139-c533-4b8c-b27b-e02a401b9bbd)

Download a zipped model folder from
[MobileSAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/mobile_sam.zip), [ViT-Large SAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam_vit_l_0b3195.zip), and [ViT-Huge SAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam_vit_h_4b8939.zip).
Put the unzipped model folder into sam-cpp-macos folder.

Edit the modelName in [test.cpp](https://github.com/ryouchinsa/sam-cpp-macos/blob/master/test.cpp).

```cpp
Sam sam;
std::string modelName = "mobile_sam";
std::string pathEncoder = modelName + "/" + modelName + "_preprocess.onnx";
std::string pathDecoder = modelName + "/" + modelName + ".onnx";
std::cout<<"loadModel started"<<std::endl;
bool successLoadModel = sam.loadModel(pathEncoder, pathDecoder, std::thread::hardware_concurrency());
if(!successLoadModel){
  std::cout<<"loadModel error"<<std::endl;
  return 1;
}
```

After loading the model, the preprocessing for the image begins. Because of CPU mode, it takes 2 seconds for "MobileSAM", 30 seconds for "ViT-Large SAM", and 60 seconds for "ViT-Huge SAM" on the Apple M1 device.

```cpp
std::string imagePath = "david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg";
cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
auto inputSize = sam.getInputSize();
cv::resize(image, image, inputSize);
cv::imwrite("resized.jpg", image);
bool terminated = false; // Check the preprocessing is terminated when the image is changed
std::cout<<"preprocessImage started"<<std::endl;
bool successPreprocessImage = sam.preprocessImage(image, &terminated);
if(!successPreprocessImage){
  std::cout<<"preprocessImage error"<<std::endl;
  return 1;
}
```

When you click a foreground point or a background point, getMask() is called and the mask result is shown. From the second click, getMask() uses the previous mask result to increase the accuracy. To support undo/redo actions, Sam class instance retains the previous mask array. previousMaskIdx is used which previous mask to use in getMask(). For the first click in the image, previousMaskIdx is set to -1. When getMask() is called, previousMaskIdx is incremented. When you start labeling a new object in the image, isNextGetMask is set to true so that getMask() does not use the previous mask result. From the second click for the object, isNextGetMask is set to false to use the previous mask result.

```cpp
std::cout<<"getMask started"<<std::endl;
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
```

Download the [ONNX Runtime v1.15.1](https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-osx-universal2-1.15.1.tgz). Edit the onnxruntime include path and lib path in CMakeLists.txt.

```bash
add_library(sam_cpp_lib SHARED sam.h sam.cpp)
target_include_directories(
  sam_cpp_lib PUBLIC 
  /Users/ryo/Downloads/onnxruntime-osx-universal2-1.15.1/include
)
target_link_libraries(
  sam_cpp_lib PUBLIC
  /Users/ryo/Downloads/onnxruntime-osx-universal2-1.15.1/lib/libonnxruntime.dylib
  ${OpenCV_LIBS}
)
```

Build and run.

```bash
cmake -S . -B build
cmake --build build
./build/sam_cpp_test
```
