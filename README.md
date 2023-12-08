# Installation Guide for TensorFlow GPU, CenterNet Hourglass, Darknet, and KITTI Dataset

## TensorFlow GPU Installation

1. Install NVIDIA GPU drivers:
   - Visit the [NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx) page and download the appropriate drivers for your GPU.

2. Install CUDA Toolkit and cuDNN:
   - Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
   - Download and install [cuDNN](https://developer.nvidia.com/cudnn).

3. Install TensorFlow GPU:
   - Run the following command to install TensorFlow with GPU support:
     ```
     pip install tensorflow-gpu
     ```

## CenterNet Hourglass Installation

1. Clone the CenterNet repository:
git clone https://github.com/xingyizhou/CenterNet.git

2. Install dependencies

3. Download pre-trained models (Optional):
- You may download pre-trained models from the CenterNet [Model Zoo](https://github.com/xingyizhou/CenterNet/blob/master/readme/MODEL_ZOO.md).

## Darknet Installation

1. Clone the Darknet repository: git clone https://github.com/AlexeyAB/darknet.git

2. Build Darknet:

cd darknet
make


3. Download pre-trained YOLO weights (Optional):
- You can download pre-trained YOLO weights from the Darknet website or use the provided weights in the `darknet/weights` directory.

## KITTI Dataset Download

1. Download the KITTI Vision Benchmark Suite:
- Visit the [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/eval_object.php) website.
- Download the relevant datasets, such as `left color images of object data set`.

2. Extract the downloaded dataset:
tar -xvf your_downloaded_dataset.tar.gz


3. Organize the dataset:
- You may need to organize the dataset according to your specific requirements and the data format expected by the algorithms you are using.

## Running the Applications

1. Follow the respective documentation and examples for TensorFlow, CenterNet, and Darknet to run the applications using your GPU and the downloaded dataset.


