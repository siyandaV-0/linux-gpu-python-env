#!/bin/bash

cuda_version="$1"

if [[ -z "$cuda_version" ]]; then
  echo "Usage: $0 <cuda_version>"
  exit 1
fi

Check if CUDA version folder exists
if [ ! -d "/usr/local/cuda-${cuda_version}/lib64" ]; then
    echo "CUDA version folder /usr/local/cuda-${cuda_version}/lib64 not found"
    exit 1
fi

# cuda_lib_dir="testdir"
# cudnn_lib_dir="$cuda_lib_dir"
# cudnn_version=$(ls $cudnn_lib_dir | grep libcudnn.so | sed 's/[^0-9.]*\([0-9.]*\).*/\1/' | sort -r | head -n1)

cuda_dir="testdir"

for file in ${cuda_dir}/libcudnn*.so*; do
  if [ -f "$file" ]; then
    link_name=$(echo "$file" | sed -E "s/(libcudnn[^.]+)\.so\..*/\1.so/")
    if [ ! -f "${link_name}" ]; then
      echo "Creating link for $file"
      ln -s "$file" "${link_name}"
    fi
  fi
done

echo "Done."
