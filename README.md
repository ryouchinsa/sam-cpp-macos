## Segment Anything Model 2 CPP Wrapper for macOS and Ubuntu

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

You can download exported ONNX models.
- [SAM2 Tiny](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam2_tiny.zip)
- [SAM2 Small](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam2_small.zip)
- [SAM2 BasePlus](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam2_base_plus.zip)
- [SAM2 Large](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam2_large.zip)

Download an ONNX Runtime folder.
- [onnxruntime-osx-universal2-1.17.1.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-osx-universal2-1.17.1.tgz) for macOS
- [onnxruntime-linux-x64-1.17.1.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz) for Ubuntu CPU
- [onnxruntime-linux-x64-gpu-1.17.1.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-gpu-1.17.1.tgz) for Ubuntu GPU

![sam_cpp_macos_folders](https://github.com/user-attachments/assets/81055a3b-0ea4-4007-96fa-0732dcf41bcc)

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
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/Users/ryo/Downloads/onnxruntime-osx-universal2-1.17.1
# Ubuntu CPU
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/root/onnxruntime-linux-x64-1.17.1 -DCMAKE_TOOLCHAIN_FILE=/root/vcpkg/scripts/buildsystems/vcpkg.cmake
# Ubuntu GPU
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/root/onnxruntime-linux-x64-gpu-1.17.1 -DCMAKE_TOOLCHAIN_FILE=/root/vcpkg/scripts/buildsystems/vcpkg.cmake

cmake --build build

# macOS and Ubuntu CPU
./build/sam_cpp_test -encoder="mobile_sam/mobile_sam_preprocess.onnx" -decoder="mobile_sam/mobile_sam.onnx" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cpu"
# Ubuntu GPU
./build/sam_cpp_test -encoder="mobile_sam/mobile_sam_preprocess.onnx" -decoder="mobile_sam/mobile_sam.onnx" -image="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg" -device="cuda:0"
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




