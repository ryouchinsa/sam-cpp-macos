## Segment Anything CPP Wrapper for macOS

This code is originated from [Segment Anything CPP Wrapper](https://github.com/dinglufe/segment-anything-cpp-wrapper) and implemented on macOS app [RectLabel](https://rectlabel.com). We customized the original code so that getMask() uses the previous mask result and retain the previous mask array for undo/redo actions. 

![sam](https://github.com/ryouchinsa/sam-cpp-macos/assets/1954306/8d41873d-c61c-43c6-a433-51fb5cd594c1)

Download a zipped model folder from below.
- [MobileSAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/mobile_sam.zip)
- [EdgeSAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/edge_sam.zip)
- [EdgeSAM-3x](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/edge_sam_3x.zip)
- [Tiny EfficientSAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/efficientsam_ti.zip)
- [Small EfficientSAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/efficientsam_s.zip)
- [ViT-Base HQ-SAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam_hq_vit_b.zip)
- [ViT-Large HQ-SAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam_hq_vit_l.zip)
- [ViT-Huge HQ-SAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam_hq_vit_h.zip)
- [ViT-Base SAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam_vit_b_01ec64.zip)
- [ViT-Large SAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam_vit_l_0b3195.zip)
- [ViT-Huge SAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam_vit_h_4b8939.zip)

Put the unzipped model folder into sam-cpp-macos folder.

![スクリーンショット 2024-03-07 15 55 39](https://github.com/ryouchinsa/sam-cpp-macos/assets/1954306/b49f101c-b9a1-444e-a773-ecbee3d46f67)

Edit the modelName in [test.cpp](https://github.com/ryouchinsa/sam-cpp-macos/blob/master/test.cpp).

```cpp
Sam sam;
std::string modelName = "mobile_sam";
if(modelName.find("sam_hq") != std::string::npos){
  sam.changeMode(HQSAM);
}else if(modelName.find("efficientsam") != std::string::npos){
  sam.changeMode(EfficientSAM);
}else if(modelName.find("edge_sam") != std::string::npos){
  sam.changeMode(EdgeSAM);
}
std::string pathEncoder = modelName + "/" + modelName + "_preprocess.onnx";
std::string pathDecoder = modelName + "/" + modelName + ".onnx";
std::cout<<"loadModel started"<<std::endl;
bool successLoadModel = sam.loadModel(pathEncoder, pathDecoder, std::thread::hardware_concurrency());
if(!successLoadModel){
  std::cout<<"loadModel error"<<std::endl;
  return 1;
}
```

After loading the model, the preprocessing for the image begins. Because of CPU mode, it takes 2 seconds for "MobileSAM" and 30 seconds for "ViT-Large SAM" on the Apple M1 device.

```cpp
std::cout<<"preprocessImage started"<<std::endl;
std::string imagePath = "david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg";
cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
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
```

To support undo/redo actions, Sam class instance retains the previous mask array.  For the first click, previousMaskIdx is set to -1. When getMask() is called, previousMaskIdx is incremented. When you start labeling a new object, isNextGetMask is set to true so that getMask() does not use the previous mask result. From the second click for the object, isNextGetMask is set to false to use the previous mask result.

```cpp
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
```

Download the [ONNX Runtime v1.17.1](https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-osx-universal2-1.17.1.tgz). Edit the onnxruntime include path and lib path in CMakeLists.txt.

```bash
add_library(sam_cpp_lib SHARED sam.h sam.cpp)
target_include_directories(
  sam_cpp_lib PUBLIC 
  /Users/ryo/Downloads/onnxruntime-osx-universal2-1.17.1/include
)
target_link_libraries(
  sam_cpp_lib PUBLIC
  /Users/ryo/Downloads/onnxruntime-osx-universal2-1.17.1/lib/libonnxruntime.dylib
  ${OpenCV_LIBS}
)
```

Build and run.

```bash
cmake -S . -B build
cmake --build build
./build/sam_cpp_test
```

To build on the Xcode, this is our settings on the Xcode.

- General -> Frameworks, Libraries, and Embedded Content

![スクリーンショット 2024-03-06 1 36 12](https://github.com/ryouchinsa/sam-cpp-macos/assets/1954306/f13b4006-ad18-4a32-92cd-179804682887)

- Build Settings

Header Search Paths
`/Users/ryo/Downloads/onnxruntime-osx-universal2-1.17.1/include`

Library Search Paths
`/Users/ryo/Downloads/onnxruntime-osx-universal2-1.17.1/lib`

- Build Phases -> Embed Libraries

![スクリーンショット 2024-03-06 1 37 32](https://github.com/ryouchinsa/sam-cpp-macos/assets/1954306/13ccda41-5d13-4e73-8b53-830ca0efa0b4)




