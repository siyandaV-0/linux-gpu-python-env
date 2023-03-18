#!/bin/bash

# Set Anaconda environment name
# ENV_NAME=cv2.GPU
# Set OpenCV version
OPENCV_VERSION=4.5.5
# Python version 
# PYTHON_VERSION=3.10

# Set CUDA version
CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
# Set cuDNN version
CUDNN_VERSION=$(ls /usr/lib/x86_64-linux-gnu/libcudnn.so.* | tail -1 | cut -d'.' -f3-5)


# Set paths to CUDA and cuDNN
CUDA_PATH="/usr/local/cuda-${CUDA_VERSION}"
CUDNN_PATH="/usr"

# # Source Anaconda
# source ~/anaconda3/etc/profile.d/conda.sh

# # Create a new Anaconda environment and activate it
# echo "Creating a new Anaconda environment: $ENV_NAME"
# conda create -y -n ${ENV_NAME} python=${PYTHON_VERSION}
# echo "Anaconda environment created successfully." 
# source activate ${ENV_NAME} 
# echo "Anaconda environment activated successfully."

# # Install required env packages for Computer Vision
# pip3 install -r requirements/cv-requirements.txt


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

# Configure CMake
cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN=7.5 \
    -D CUDA_ARCH_PTX=7.5 \
    -D WITH_CUBLAS=1 \
    -D CUDA_FAST_MATH=ON \
    -D ENABLE_FAST_MATH=1 \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D WITH_CUDNN=ON \
    -D CUDNN_LIBRARY=${CUDA_PATH}/lib/x86_64-linux-gnu/libcudnn.so \
    -D CUDNN_INCLUDE_DIR=${CUDNN_PATH}/include \
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


# Cleanup
rm -rf opencv
rm -rf opecnv-contrib


# May have to link the following maunually
cd ${CONDA_PREFIX}/lib/python3.10/site-packages/cv2/python-3.10
sudo ln -s cv2.cpython-39-x86_64-linux-gnu.so cv2.so

