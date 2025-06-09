## Segment Anything Model 2 CPP Wrapper for macOS and Ubuntu GPU

This code is to run a [Segment Anything Model 2](https://github.com/facebookresearch/sam2) ONNX model in c++ code and implemented on the macOS app [RectLabel](https://rectlabel.com).

<video src="https://github.com/user-attachments/assets/812776c3-bfad-4f80-99e1-6141b21c024b" controls="controls" muted="muted" class="width-fit" style="max-height:640px; min-height: 200px"></video>

Install [CUDA, cuDNN, PyTorch, and ONNX Runtime](https://rectlabel.com/pytorch/).

Install [Segment Anything Model 2](https://github.com/facebookresearch/sam2), download checkpoints and copy yaml files in sam2/configs/sam2.1 to sam2.

```bash
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
cd checkpoints
./download_ckpts.sh
cd ..
cp sam2/configs/sam2.1/*.yaml sam2
```

Install Segment Anything Model 2 CPP Wrapper.
```bash
git clone https://github.com/ryouchinsa/sam-cpp-macos.git
cp sam-cpp-macos/export_onnx.py .
cp sam-cpp-macos/david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg .
```

Export an ONNX model and check how the ONNX model works.

```bash
python export_onnx.py --mode export
python export_onnx.py --mode import
cp -r checkpoints/sam2.1_tiny sam-cpp-macos
cd sam-cpp-macos
```

Download exported SAM 2.1 ONNX models.
- [SAM 2.1 Tiny](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam2.1_tiny.zip)
- [SAM 2.1 Small](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam2.1_small.zip)
- [SAM 2.1 BasePlus](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam2.1_base_plus.zip)
- [SAM 2.1 Large](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam2.1_large.zip)

Build and run.

```bash
# macOS
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/Users/ryo/Downloads/onnxruntime-osx-universal2-1.20.0
# Ubuntu GPU
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/root/onnxruntime-linux-x64-gpu-1.20.0

cmake --build build

# macOS
./build/sam_cpp_test -encoder="sam2.1_tiny/sam2.1_tiny_preprocess.onnx" -decoder="sam2.1_tiny/sam2.1_tiny.onnx" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cpu"
# Ubuntu GPU
./build/sam_cpp_test -encoder="sam2.1_tiny/sam2.1_tiny_preprocess.onnx" -decoder="sam2.1_tiny/sam2.1_tiny.onnx" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cuda:0"
```
