#!/bin/bash
#set -Eueo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

DEST=${1:-${DIR}}
WHL=onnxruntime_gpu_tensorrt-1.3.0-cp36-cp36m-linux_aarch64.whl

if [ -f ${DEST}/${WHL} ]; then
    echo "Found ${WHL} in ${DEST}"
else
    # Comment this out if CMake is already > 3.13
    # let it fail if run second time
    echo "======================= Building CMake 3.13 for ONNXruntime. That will take a while ... Log is in /tmp/CMake/cmake/cmake.log==============================="
    mkdir -p /tmp/cmake && cd /tmp/cmake && \
	git clone https://github.com/Kitware/CMake.git -b v3.13.5 --depth 1
    cd /tmp/cmake/CMake && ./bootstrap && make && sudo -H make install  > cmake.log 2>&1 
    
    echo "======================= Building ONNXruntime on the host. That will take a while ... Log is in /tmp/ort/onnxruntime/ort.log ======================="
    
    mkdir -p /tmp/ort && cd /tmp/ort && \
	git clone https://github.com/microsoft/onnxruntime.git --single-branch --recursive --branch v1.3.0 && \
	cd onnxruntime && mkdir -p build/Linux/Release
    cd /tmp/ort/onnxruntime && patch -p1  < ${DIR}/ort_patch.txt 
    
    export CUDACXX="/usr/local/cuda/bin/nvcc"
    export PATH="/usr/local/cuda/bin/:${PATH}"
    
    cd /tmp/ort/onnxruntime && ./build.sh --update --config Release --build --build_wheel --use_tensorrt --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu --tensorrt_home /usr/lib/aarch64-linux-gnu > ort.log 2>&1 

    mkdir -p ${DEST} && mv /tmp/ort/onnxruntime/build/Linux/Release/dist/${WHL} ${DEST} || exit 1
    
    echo "Built ${WHL} in ${DEST}."
fi
