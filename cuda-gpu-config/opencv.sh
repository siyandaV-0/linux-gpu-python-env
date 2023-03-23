#!/bin/bash

# Set OpenCV version
OPENCV_VERSION="4.5.5"

# conda install -c conda-forge libstdcxx-ng

# If opencv exist within environment, remove it
pip3 uninstall opencv-python
pip3 uninstall opencv-contrib-python
pip3 uninstall opencv-headless-python


# Set CUDA version to that corresponding to our env cv2-gpu
CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
# Set paths to CUDNN
CUDNN_PATH="/usr/local/cuda-${CUDA_VERSION}"


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
    -D CUDNN_LIBRARY=${CUDNN_PATH}/lib64 \
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

cd ../../

# Cleanup
rm -rf opencv && rm -rf opecnv-contrib


# To extract the version numbers from Python version returned by python --version
python_version=$(python -V 2>&1)
ver_digits=$(echo "${python_version}" | awk '{print $2}' | awk -F'.' '{print $1$2}')
ver_dot_digits=$(echo "${python_version}" | awk '{print $2}' | awk -F'.' '{print $1"."$2}')

# May have to link the following maunually
cd ${CONDA_PREFIX}/lib/python${ver_dot_digits}/site-packages/cv2/python-${ver_dot_digits}
sudo ln -sf cv2.cpython-${ver_digits}-x86_64-linux-gnu.so cv2.so

