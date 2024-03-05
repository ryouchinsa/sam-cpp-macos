## Segment Anything CPP Wrapper for macOS

This code is originated from [Segment Anything CPP Wrapper](https://github.com/dinglufe/segment-anything-cpp-wrapper) and implemented on macOS app [RectLabel](https://rectlabel.com). We customized the original code so that getMask() uses the previous mask result and retain the previous mask array for undo/redo actions. 

![sam](https://github.com/ryouchinsa/sam-cpp-macos/assets/1954306/8d41873d-c61c-43c6-a433-51fb5cd594c1)

Download a zipped model folder from below.
- [MobileSAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/mobile_sam.zip)
- [Tiny EfficientSAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/efficientsam_ti.zip)
- [Small EfficientSAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/efficientsam_s.zip)
- [ViT-Base SAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam_vit_b_01ec64.zip)
- [ViT-Large SAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam_vit_l_0b3195.zip)
- [ViT-Huge SAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam_vit_h_4b8939.zip)

Put the unzipped model folder into sam-cpp-macos folder.

![スクリーンショット 2024-03-05 19 44 41](https://github.com/ryouchinsa/sam-cpp-macos/assets/1954306/0cd464c4-c997-4ba4-a093-8ffc5f3de08e)

Edit the modelName in [test.cpp](https://github.com/ryouchinsa/sam-cpp-macos/blob/master/test.cpp).

```cpp
Sam sam;
std::string modelName = "mobile_sam";
if(modelName.find("efficientsam") != std::string::npos){
  sam.changeMode(EfficientSAM);
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
```

After loading the model, the preprocessing for the image begins. Because of CPU mode, it takes 2 seconds for "MobileSAM", 30 seconds for "ViT-Large SAM", and 60 seconds for "ViT-Huge SAM" on the Apple M1 device.

```cpp
std::string imagePath = "david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg";
cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
auto inputSize = sam.getInputSize();
cv::resize(image, image, inputSize);
std::cout<<"preprocessImage started"<<std::endl;
bool successPreprocessImage = sam.preprocessImage(image);
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

Download the [ONNX Runtime v1.16.3](https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-osx-universal2-1.16.3.tgz). Edit the onnxruntime include path and lib path in CMakeLists.txt.

```bash
add_library(sam_cpp_lib SHARED sam.h sam.cpp)
target_include_directories(
  sam_cpp_lib PUBLIC 
  /Users/ryo/Downloads/onnxruntime-osx-universal2-1.16.3/include
)
target_link_libraries(
  sam_cpp_lib PUBLIC
  /Users/ryo/Downloads/onnxruntime-osx-universal2-1.16.3/lib/libonnxruntime.dylib
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
-- The C compiler identification is AppleClang 15.0.0.15000040
-- The CXX compiler identification is AppleClang 15.0.0.15000040
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
-- Configuring done (4.0s)
-- Generating done (0.0s)
-- Build files have been written to: /Users/ryo/Downloads/sam-cpp-macos/build
```

To build on the Xcode, this is my settings on the Xcode.

- General -> Frameworks, Libraries, and Embedded Content

![スクリーンショット 2023-12-24 15 07 26](https://github.com/ryouchinsa/sam-cpp-macos/assets/1954306/bc86a5df-d9c3-454f-8f81-5fc28ef54b42)

- Build Settings

Header Search Paths
`/Users/ryo/Downloads/onnxruntime-osx-universal2-1.16.3/include`

Library Search Paths
`/Users/ryo/Downloads/onnxruntime-osx-universal2-1.16.3/lib`

![スクリーンショット 2023-12-24 15 15 56](https://github.com/ryouchinsa/sam-cpp-macos/assets/1954306/ffa4f838-90cf-4be0-87bc-208a65c917f8)

- Build Phases -> Embed Libraries

![スクリーンショット 2023-12-24 15 16 33](https://github.com/ryouchinsa/sam-cpp-macos/assets/1954306/ef3e23ad-0482-4f61-868f-3da63a9f0b2f)



