#!/bin/bash
#set -Eueo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DEST=${1:-${DIR}}
ARCH=${2:-${HOSTTYPE}}

WHL=onnxruntime_gpu-1.3.0-cp36-cp36m-linux_${ARCH}.whl

if [ -f ${DEST}/${WHL} ]; then
    echo "Found ${WHL} in ${DEST}"
else    
    
    mkdir -p /tmp/ort && cd /tmp/ort && \
	git clone https://github.com/microsoft/onnxruntime.git --single-branch --recursive --branch v1.3.0 && \
	cd onnxruntime && mkdir -p build/Linux/Release
    
    if [ "${ARCH}" = "aarch64" ]; then
	cd /tmp/ort/onnxruntime && patch -p1  < ${DIR}/ort_patch.txt
	# Comment this if CMake is already > 3.13
	# let it fail if run second time
	echo "======================= Building CMake 3.13 for ONNXruntime. That will take a while ... Log is in /tmp/CMake/cmake/cmake.log==============================="
	mkdir -p /tmp/cmake && cd /tmp/cmake && \
    	    git clone https://github.com/Kitware/CMake.git -b v3.13.5 --depth 1 \
	    && cd /tmp/cmake/CMake && ./bootstrap && make && sudo -H make install  > cmake.log 2>&1 
    fi
    
    export CUDACXX="/usr/local/cuda/bin/nvcc"
    export PATH="/usr/local/cuda/bin/:${PATH}"
    
    echo "======================= Building ONNXruntime on the host. That will take a while ... Log is in /tmp/ort/onnxruntime/ort.log ======================="
    cd /tmp/ort/onnxruntime && ./build.sh --update --config Release --build \
					  --build_wheel --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/lib/${ARCH}-linux-gnu  > ort.log 2>&1 

    mkdir -p ${DEST} && mv /tmp/ort/onnxruntime/build/Linux/Release/dist/${WHL} ${DEST} || exit 1
    
    echo "Built ${WHL} in ${DEST}."
fi
