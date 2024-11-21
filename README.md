## Segment Anything Model 2 CPP Wrapper for macOS and Ubuntu CPU/GPU

This code is to run a [Segment Anything Model 2](https://github.com/facebookresearch/sam2) ONNX model in c++ code and implemented on the macOS app [RectLabel](https://rectlabel.com).

<video src="https://github.com/user-attachments/assets/812776c3-bfad-4f80-99e1-6141b21c024b" controls="controls" muted="muted" class="width-fit" style="max-height:640px; min-height: 200px"></video>

Install [Segment Anything Model 2](https://github.com/facebookresearch/sam2) and download checkpoints.

![checkpoints](https://github.com/user-attachments/assets/0a905f19-6cb8-4231-a355-df6b1e8f1ab0)

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

You can download exported SAM 2.1 ONNX models.
- [SAM2 Tiny](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam2_tiny.zip)
- [SAM2 Small](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam2_small.zip)
- [SAM2 BasePlus](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam2_base_plus.zip)
- [SAM2 Large](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam2_large.zip)

Download an ONNX Runtime folder.
- [onnxruntime-osx-universal2-1.20.0.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-osx-universal2-1.20.0.tgz) for macOS
- [onnxruntime-linux-x64-1.20.0.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz) for Ubuntu CPU
- [onnxruntime-linux-x64-gpu-1.20.0.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-gpu-1.20.0.tgz) for Ubuntu GPU

![folder](https://github.com/user-attachments/assets/ee7c328f-17e1-4881-a2db-1942f3eee5a4)

For Ubuntu, install gflags and opencv through [vcpkg](https://github.com/microsoft/vcpkg).
```bash
git clone https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh
./vcpkg/vcpkg install gflags
./vcpkg/vcpkg install opencv
```

For Ubuntu GPU, install cuda and cudnn.
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-drivers
reboot
nvidia-smi

sudo apt install cuda-toolkit-11-8
vi ~/.bashrc
export PATH="/usr/local/cuda/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
source ~/.bashrc
which nvcc
nvcc --version

apt list libcudnn8 -a
cudnn_version=8.9.7.29
cuda_version=cuda11.8
sudo apt install libcudnn8=${cudnn_version}-1+${cuda_version}
sudo apt install libcudnn8-dev=${cudnn_version}-1+${cuda_version}
sudo apt install libcudnn8-samples=${cudnn_version}-1+${cuda_version}
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
./build/sam_cpp_test -encoder="sam2_tiny/sam2_tiny_preprocess.onnx" -decoder="sam2_tiny/sam2_tiny.onnx" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cpu"
# Ubuntu GPU
./build/sam_cpp_test -encoder="sam2_tiny/sam2_tiny_preprocess.onnx" -decoder="sam2_tiny/sam2_tiny.onnx" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cuda:0"
```
