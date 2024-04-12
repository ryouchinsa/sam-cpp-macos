## Segment Anything CPP Wrapper for macOS

This code is originated from [Segment Anything CPP Wrapper](https://github.com/dinglufe/segment-anything-cpp-wrapper) and implemented on the macOS app [RectLabel](https://rectlabel.com). We customized the original code so that getMask() uses the previous mask result and retain the previous mask array for undo/redo actions. 

![sam](https://github.com/ryouchinsa/sam-cpp-macos/assets/1954306/8d41873d-c61c-43c6-a433-51fb5cd594c1)

Download a zipped model folder.
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

Download [onnxruntime-osx-universal2-1.17.1.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-osx-universal2-1.17.1.tgz).

![スクリーンショット 2024-03-12 5 06 38](https://github.com/ryouchinsa/sam-cpp-macos/assets/1954306/cee0f920-7041-4110-9319-d825e7c3f952)

Build and run.

```bash
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/Users/ryo/Downloads/onnxruntime-osx-universal2-1.17.1
cmake --build build
./build/sam_cpp_test -encoder="mobile_sam/mobile_sam_preprocess.onnx" -decoder="mobile_sam/mobile_sam.onnx" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cpu"
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




