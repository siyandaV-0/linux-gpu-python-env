#!/bin/bash

# Set OpenCV version
OPENCV_VERSION="4.5.5"

# Set CUDA version
CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
echo "For CUDA VERSION: ${CUDA_VERSION}"

# Set paths to CUDNN
CUDNN_PATH="/usr/local/cuda-${CUDA_VERSION}"
echo "CUDNN_PATH: ${CUDNN_PATH}"

# If opencv exist within environment, remove it
pip3 uninstall opencv-python
pip3 uninstall opencv-contrib-python
pip3 uninstall opencv-headless-python

# Clone opencv and opencv-contrib repositories
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

# Checkout the version you want to install
cd opencv && git checkout ${OPENCV_VERSION} && cd ..
cd opencv_contrib && git checkout ${OPENCV_VERSION} && cd ..

cd opencv
# Create build directory
mkdir build

cd build

SYS_PREFIX=$(python3 -c "import sys; print(sys.prefix)")

# Configure CMake
cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=${SYS_PREFIX} \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN=7.5 \
    -D CUDA_ARCH_PTX=7.5 \
    -D WITH_CUBLAS=1 \
    -D CUDA_FAST_MATH=ON \
    -D ENABLE_FAST_MATH=1 \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D WITH_CUDNN=ON \
    -D CUDNN_LIBRARY=${CUDNN_PATH}/lib64/libcudnn.so \
    -D CUDNN_INCLUDE_DIR=${CUDNN_PATH}/include \
    -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
    -D PYTHON_INCLUDE_DIR=$(python3 -c "from sysconfig import get_path; print(get_path(\"include\"))") \
    -D PYTHON_LIBRARY=$(python3 -c "from sysconfig import get_path; print(get_path(\"platlib\"))")\
    -D PYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \ 
    -D BUILD_TIFF=ON \
    -D BUILD_EXAMPLES=ON \
    ..

# Build and install OpenCV
make -j$(nproc)

make install

ldconfig

cd ../../

# Cleanup
rm -rf opencv 
rm -rf opecnv_contrib

# May have to link cv2.so files to cv2.cpython*.so file  maunually
cd $(python3 -c "import os; from sysconfig import get_path; print(os.path.join(get_path(\"platlib\"), 'cv2', 'python*'))")
ln -sf cv2.cpython*.so cv2.so

