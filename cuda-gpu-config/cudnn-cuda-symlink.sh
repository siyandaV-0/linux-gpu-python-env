#!/bin/bash

CUDA_VERSION="$1"

if [[ -z "$CUDA_VERSION" ]]; then
  echo "Usage: $0 <cuda_version>"
  exit 1
fi

# Check if CUDA version folder exists
if [ ! -d "/usr/local/cuda-${CUDA_VERSION}/lib64" ]; then
    echo "CUDA version folder /usr/local/cuda-${CUDA_VERSION}/lib64 not found"
    exit 1
fi

CUDNN_LIB="/usr/local/cuda-${CUDA_VERSION}/lib64"

for file in ${CUDNN_LIB}/libcudnn*.so*; do
  if [ -f "$file" ]; then
    link_name=$(echo "$file" | sed -E "s/(libcudnn[^.]+)\.so\..*/\1.so/")
    if [ ! -f "${link_name}" ]; then
      echo "Creating link for $file"
      sudo ln -sf "$file" "${link_name}"
    fi
  elif [ -L "$file" ]; then
    echo "Symbolic link already exists for $file"
  fi
done

echo "Done."
