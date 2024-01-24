NVCC=nvcc
NVCC_FLAGS=-std=c++17 -O3 --generate-code arch=compute_70,code=sm_70 --generate-code arch=compute_75,code=sm_75 --generate-code arch=compute_80,code=sm_80 --generate-code arch=compute_86,code=sm_86

CUFFTDX_INCLUDE_DIR=../include/
CUDA_BIN_DIR=$(shell dirname `which $(NVCC)`)
CUDA_INCLUDE_DIR=$(CUDA_BIN_DIR)/../include
NVRTC_DEFINES=-DCUDA_INCLUDE_DIR="\"$(CUDA_INCLUDE_DIR)\"" -DCUFFTDX_INCLUDE_DIRS="\"$(CUFFTDX_INCLUDE_DIR)\""

SRCS=$(filter-out nvrtc_fft_thread.cu convolution_performance.cu, $(wildcard *.cu))
TARGETS=$(patsubst %.cu,%,$(SRCS))

NVRTC_SRCS=$(wildcard nvrtc_*.cu)
NVRTC_TARGETS=$(patsubst %.cu,%,$(NVRTC_SRCS))

CUFFT_SRCS=convolution_performance.cu
CUFFT_TARGETS=convolution_performance

$(TARGETS): %: %.cu
	$(NVCC) -o $@ $< $(NVCC_FLAGS) -I$(CUFFTDX_INCLUDE_DIR)

$(NVRTC_TARGETS): %: %.cu
	$(NVCC) -o $@ $< $(NVCC_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) $(NVRTC_DEFINES) -lnvrtc -lcuda

$(CUFFT_TARGETS): %: %.cu
	$(NVCC) -o $@ $< $(NVCC_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) $(NVRTC_DEFINES) -lcuda -lcufft

.PHONY: all clean

all: $(TARGETS) $(NVRTC_TARGETS) $(CUFFT_TARGETS)
	$(echo $(NVRTC_TARGETS))

clean:
	rm -f $(TARGETS) $(NVRTC_TARGETS) $(CUFFT_TARGETS)

.DEFAULT_GOAL := all
