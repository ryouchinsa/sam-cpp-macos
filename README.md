## Segment Anything CPP Wrapper for macOS

This code is originated from [Segment Anything CPP Wrapper](https://github.com/dinglufe/segment-anything-cpp-wrapper) and implemented on macOS app [RectLabel](https://rectlabel.com). We customized the original code so that getMask() uses the previous mask result called as low_res_logits and retain the previous mask array for undo/redo actions. 

![sam_polygon](https://github.com/ryouchinsa/sam-cpp-macos/assets/1954306/d0021004-3eb8-4873-ab88-284fcc149a5b)

Download a zipped model folder from
[MobileSAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/mobile_sam.zip), [ViT-Large SAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam_vit_l_0b3195.zip), and [ViT-Huge SAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam_vit_h_4b8939.zip).
Put the unzipped model folder into sam-cpp-macos folder.

![スクリーンショット 2023-09-15 22 14 30](https://github.com/ryouchinsa/sam-cpp-macos/assets/1954306/8d2b1ade-1f38-4f9b-a40d-73607893c903)

Edit the modelName in [test.cpp](https://github.com/ryouchinsa/sam-cpp-macos/blob/master/test.cpp).

```cpp
Sam sam;
std::string modelName = "mobile_sam";
std::string pathEncoder = modelName + "/" + modelName + "_preprocess.onnx";
std::string pathDecoder = modelName + "/" + modelName + ".onnx";
std::cout<<"loadModel started"<<std::endl;
bool terminated = false; // Check the preprocessing is terminated when the image is changed
bool successLoadModel = sam.loadModel(pathEncoder, pathDecoder, std::thread::hardware_concurrency(), &terminated);
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

If the build fails, check the OpenCV_INCLUDE_DIRS and OpenCV_LIBS are correct.

```bash
-- The C compiler identification is AppleClang 14.0.3.14030022
-- The CXX compiler identification is AppleClang 14.0.3.14030022
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found OpenCV: /opt/homebrew/Cellar/opencv/4.8.0_6 (found version "4.8.0") 
-- OpenCV_INCLUDE_DIRS = /opt/homebrew/Cellar/opencv/4.8.0_6/include/opencv4
-- OpenCV_LIBS = opencv_calib3d;opencv_core;opencv_dnn;opencv_features2d;opencv_flann;opencv_gapi;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_stitching;opencv_video;opencv_videoio;opencv_alphamat;opencv_aruco;opencv_bgsegm;opencv_bioinspired;opencv_ccalib;opencv_datasets;opencv_dnn_objdetect;opencv_dnn_superres;opencv_dpm;opencv_face;opencv_freetype;opencv_fuzzy;opencv_hfs;opencv_img_hash;opencv_intensity_transform;opencv_line_descriptor;opencv_mcc;opencv_optflow;opencv_phase_unwrapping;opencv_plot;opencv_quality;opencv_rapid;opencv_reg;opencv_rgbd;opencv_saliency;opencv_sfm;opencv_shape;opencv_stereo;opencv_structured_light;opencv_superres;opencv_surface_matching;opencv_text;opencv_tracking;opencv_videostab;opencv_viz;opencv_wechat_qrcode;opencv_xfeatures2d;opencv_ximgproc;opencv_xobjdetect;opencv_xphoto
-- Configuring done (7.9s)
-- Generating done (0.0s)
-- Build files have been written to: /Users/ryo/Downloads/sam-cpp-macos/build
```

