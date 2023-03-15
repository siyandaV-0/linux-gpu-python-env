#!/bin/bash

# Set the paths to the cuDNN library files
CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
CUDNN_DIR=/usr/local/${CUDA_VERSION}/lib64/
CUDNN_LIB_DIR=${CUDNN_DIR}/lib64/
CUDNN_LIB_DIR=${CUDNN_DIR}/include/

sudo ln -sf libcudnn.so.8.0.5 libcudnn.so.8 \ 
sudo ln -sf libcudnn.so.8 libcudnn.so \ 
sudo ln -sf libcudnn_adv_infer.so.8.0.5 libcudnn_adv_infer.so.8 \ 
sudo ln -sf libcudnn_adv_infer.so.8 libcudnn_adv_infer.so \ 
sudo ln -sf libcudnn_adv_train.so.8.0.5 libcudnn_adv_train.so.8 \ 
sudo ln -sf libcudnn_adv_train.so.8 libcudnn_adv_train.so \
sudo ln -sf libcudnn_cnn_infer.so.8.0.5 libcudnn_cnn_infer.so.8 \
sudo ln -sf libcudnn_cnn_infer.so.8 libcudnn_cnn_infer.so \
sudo ln -sf libcudnn_cnn_train.so.8.0.5 libcudnn_cnn_train.so.8 \
sudo ln -sf libcudnn_cnn_train.so.8 libcudnn_cnn_train.so \
sudo ln -sf libcudnn_ops_infer.so.8.0.5 libcudnn_ops_infer.so.8 \
sudo ln -sf libcudnn_ops_infer.so.8 libcudnn_ops_infer.so \
sudo ln -sf libcudnn_ops_train.so.8.0.5 libcudnn_ops_train.so.8 \
sudo ln -sf libcudnn_ops_train.so.8 libcudnn_ops_train.so \

