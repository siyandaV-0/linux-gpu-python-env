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

CUDNN_PATH="/usr/local/cuda-${CUDA_VERSION}/lib64"

cd ${CUDNN_PATH}

# Define the base name of the library files
LIBRARY_BASENAME="libcudnn"

# Find the latest version of libcudnn
LATEST_LIBCUDNN=$(ls -1 ${LIBRARY_BASENAME}*.so.*.*.* | sort -V | tail -n1)

# Extract the version number from the file name
LIBCUDNN_VERSION=$(echo ${LATEST_LIBCUDNN} | sed -E "s/.*\.so\.([0-9]+\.[0-9]+\.[0-9]+)/\1/")

# Create an array of library files
LIBRARY_FILES=($(ls -1 ${LIBRARY_BASENAME}*.so.*.*.*))

# Create the symbolic links
for library_file in "${LIBRARY_FILES[@]}"; do
  # Extract the library name from the file name
  library_name=$(echo ${library_file} | sed -E "s/(.*\.so)\.${LIBCUDNN_VERSION}.*/\1/")
  sudo ln -sf "${library_name}.${LIBCUDNN_VERSION}" "${library_name}.${LIBCUDNN_VERSION%%.*}"
  sudo ln -sf "${library_name}.${LIBCUDNN_VERSION%%.*}" "${library_name}"
done

echo "Done."
