#!/bin/bash

# Set the paths to the cuDNN library files
CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
# Get the version number of the installed libcudnn library files
CUDNN_VERSION=$(ls /usr/local/cuda-${CUDA_VERSION}/lib64/libcudnn.so.* | tail -1 | cut -d'.' -f3)

# Set the paths to the cuDNN library files
CUDNN_DIR=/usr/local/cuda-${CUDA_VERSION}/lib64
CUDNN_INCLUDE_DIR=/usr/local/cuda-${CUDA_VERSION}/include

# sudo ln -sf libcudnn.so.8.0.5 libcudnn.so.8 \ 
# sudo ln -sf libcudnn.so.8 libcudnn.so \ 
# sudo ln -sf libcudnn_adv_infer.so.8.0.5 libcudnn_adv_infer.so.8 \ 
# sudo ln -sf libcudnn_adv_infer.so.8 libcudnn_adv_infer.so \ 
# sudo ln -sf libcudnn_adv_train.so.8.0.5 libcudnn_adv_train.so.8 \ 
# sudo ln -sf libcudnn_adv_train.so.8 libcudnn_adv_train.so \
# sudo ln -sf libcudnn_cnn_infer.so.8.0.5 libcudnn_cnn_infer.so.8 \
# sudo ln -sf libcudnn_cnn_infer.so.8 libcudnn_cnn_infer.so \
# sudo ln -sf libcudnn_cnn_train.so.8.0.5 libcudnn_cnn_train.so.8 \
# sudo ln -sf libcudnn_cnn_train.so.8 libcudnn_cnn_train.so \
# sudo ln -sf libcudnn_ops_infer.so.8.0.5 libcudnn_ops_infer.so.8 \
# sudo ln -sf libcudnn_ops_infer.so.8 libcudnn_ops_infer.so \
# sudo ln -sf libcudnn_ops_train.so.8.0.5 libcudnn_ops_train.so.8 \
# sudo ln -sf libcudnn_ops_train.so.8 libcudnn_ops_train.so \

# Create symbolic links for the cuDNN library files
for lib in ${CUDNN_DIR}/libcudnn*${CUDNN_VERSION}*; do
  sudo ln -sf ${lib} ${CUDNN_DIR}/$(basename ${lib})
done

sudo ln -sf ${CUDNN_INCLUDE_DIR}/cudnn_version.h ${CUDNN_INCLUDE_DIR}/cudnn.h
sudo ln -sf ${CUDNN_INCLUDE_DIR}/libcudnn_version.h ${CUDNN_INCLUDE_DIR}/libcudnn.h



for lib in ${CUDNN_LIB_DIR}/libcudnn*${CUDNN_VERSION}*; do
  ls| grep  ${lib} 
  ls | grep ${CUDNN_LIB_DIR}/$(basename ${lib})'
done
