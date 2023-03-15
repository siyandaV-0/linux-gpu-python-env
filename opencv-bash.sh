#!/bin/bash

# Set OpenCV version
opencv_version="4.5.4"

# Set CUDA version
CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')

# Set cuDNN version
cudnn_version="8.8.1"

# Set Conda environment name
conda_env="CV2_GPU"

# Set paths to CUDA and cuDNN
cuda_path="/usr/local/cuda-${CUDA_VERSION}"
cudnn_path="/usr"

# Activate Conda environment
conda activate ${conda_env}

# Clone OpenCV repository
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout ${opencv_version}

# Create build directory
mkdir build
cd build

# Configure CMake
cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN=7.5 \
    -D CUDA_ARCH_PTX=7.5 \
    -D WITH_CUBLAS=1 \
    -D CUDA_FAST_MATH=ON \
    -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D WITH_CUDNN=ON \
    -D CUDNN_LIBRARY=${cudnn_path}/lib/x86_64-linux-gnu/libcudnn.so \
    -D CUDNN_INCLUDE_DIR=${cudnn_path}/include \
    -D PYTHON_DEFAULT_EXECUTABLE=$(which python) \
    -D PYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -D PYTHON_LIBRARY=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    -D PYTHON3_NUMPY_INCLUDE_DIRS=$(python -c "import numpy; print(numpy.get_include())") \
    -D BUILD_TIFF=ON \
	-D BUILD_EXAMPLES=ON \
    ..

# Build and install OpenCV
make -j$(nproc)
sudo make install
sudo ldconfig

# May have to link the following maunually
cd ~/anaconda3/envs/cv2_GPU/lib/python3.9/site-packages/cv2/python-3.9
sudo ln -s cv2.cpython-39-x86_64-linux-gnu.so cv2.so

# Cleanup
cd ~/
rm -rf opencv
