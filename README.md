## Segment Anything Model 2 CPP Wrapper for macOS and Ubuntu CPU/GPU

This code is to run a [Segment Anything Model 2](https://github.com/facebookresearch/sam2) ONNX model in c++ code and implemented on the macOS app [RectLabel](https://rectlabel.com).

<video src="https://github.com/user-attachments/assets/812776c3-bfad-4f80-99e1-6141b21c024b" controls="controls" muted="muted" class="width-fit" style="max-height:640px; min-height: 200px"></video>

Install [Segment Anything Model 2](https://github.com/facebookresearch/sam2) and download checkpoints.

![checkpoints](https://github.com/user-attachments/assets/f57c57a3-f689-466e-b883-8d8caf931d11)

Copy yaml files in sam2/configs/sam2.1 to sam2.

![configs](https://github.com/user-attachments/assets/39827d2f-76ba-4904-bc59-9e0716af6cda)

Put [export_onnx.py](https://github.com/ryouchinsa/sam-cpp-macos/blob/master/export_onnx.py) and david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg to the root folder.

![export](https://github.com/user-attachments/assets/0fc7dcce-8f38-403b-b84e-bb38fea0eeca)

To export an ONNX model.

```bash
python export_onnx.py --mode export
```

To check how the ONNX model works.

```bash
python export_onnx.py --mode import
```

Download exported SAM 2.1 ONNX models.
- [SAM 2.1 Tiny](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam2.1_tiny.zip)
- [SAM 2.1 Small](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam2.1_small.zip)
- [SAM 2.1 BasePlus](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam2.1_base_plus.zip)
- [SAM 2.1 Large](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam2.1_large.zip)

Download ONNX Runtime.
- [onnxruntime-osx-universal2-1.20.0.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-osx-universal2-1.20.0.tgz) for macOS
- [onnxruntime-linux-x64-1.20.0.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz) for Ubuntu CPU
- [onnxruntime-linux-x64-gpu-1.20.0.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-gpu-1.20.0.tgz) for Ubuntu GPU

![folders](https://github.com/user-attachments/assets/fb0d3bbf-d5e9-4cee-8b9b-7c7a5c5af573)

For Ubuntu, install packages including gflags and opencv.
```bash
sudo apt-get update
sudo apt-get install build-essential tar curl zip unzip autopoint libtool bison libx11-dev libxft-dev libxext-dev libxrandr-dev libxi-dev libxcursor-dev libxdamage-dev libxinerama-dev libxtst-dev cmake libgflags-dev libopencv-dev python3-dev
```

For Ubuntu GPU, install CUDA and cuDNN.
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt install cuda-drivers
nvidia-smi

sudo apt install cuda-toolkit-12-8
vi ~/.bashrc
export PATH="/usr/local/cuda/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
source ~/.bashrc
nvcc --version

apt-cache search libcudnn
sudo apt install libcudnn9-cuda-12
sudo apt install libcudnn9-dev-cuda-12
```

Build and run.

```bash
# macOS
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/Users/ryo/Downloads/onnxruntime-osx-universal2-1.20.0
# Ubuntu CPU
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/root/onnxruntime-linux-x64-1.20.0 -DCMAKE_TOOLCHAIN_FILE=/root/vcpkg/scripts/buildsystems/vcpkg.cmake
# Ubuntu GPU
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/root/onnxruntime-linux-x64-gpu-1.20.0 -DCMAKE_TOOLCHAIN_FILE=/root/vcpkg/scripts/buildsystems/vcpkg.cmake

cmake --build build

# macOS and Ubuntu CPU
./build/sam_cpp_test -encoder="sam2.1_tiny/sam2.1_tiny_preprocess.onnx" -decoder="sam2.1_tiny/sam2.1_tiny.onnx" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cpu"
# Ubuntu GPU
./build/sam_cpp_test -encoder="sam2.1_tiny/sam2.1_tiny_preprocess.onnx" -decoder="sam2.1_tiny/sam2.1_tiny.onnx" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cuda:0"
```
