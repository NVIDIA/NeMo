#ifndef CUFFTDX_EXAMPLE_COMMON_HPP_
#define CUFFTDX_EXAMPLE_COMMON_HPP_

#include <cuda_runtime_api.h>

#ifndef CUDA_CHECK_AND_EXIT
#    define CUDA_CHECK_AND_EXIT(error)                                                                      \
        {                                                                                                   \
            auto status = static_cast<cudaError_t>(error);                                                  \
            if (status != cudaSuccess) {                                                                    \
                std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                                          \
            }                                                                                               \
        }
#endif // CUDA_CHECK_AND_EXIT

#ifndef CUFFT_CHECK_AND_EXIT
#    define CUFFT_CHECK_AND_EXIT(error)                                                                     \
        {                                                                                                   \
            auto status = static_cast<cufftResult>(error);                                                  \
            if (status != CUFFT_SUCCESS) {                                                                  \
                std::cout << status << " " << __FILE__ << ":" << __LINE__ << std::endl;                \
                std::exit(status);                                                                          \
            }                                                                                               \
        }
#endif // CUFFT_CHECK_AND_EXIT

namespace example {
    inline unsigned int get_cuda_device_arch() {
        int device;
        CUDA_CHECK_AND_EXIT(cudaGetDevice(&device));

        int major = 0;
        int minor = 0;
        CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

        return static_cast<unsigned>(major) * 100 + static_cast<unsigned>(minor) * 10;
    }

    inline unsigned int get_multiprocessor_count(int device) {
        int multiprocessor_count = 0;
        CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
        return multiprocessor_count;
    }

    inline unsigned int get_multiprocessor_count() {
        int device = 0;
        CUDA_CHECK_AND_EXIT(cudaGetDevice(&device));
        return get_multiprocessor_count(device);
    }

    template<template<unsigned int> class Functor>
    inline int sm_runner() {
        // Get CUDA device compute capability
        const auto cuda_device_arch = get_cuda_device_arch();

        switch (cuda_device_arch) {
            // All SM supported by cuFFTDx
            case 700: Functor<700>()(); return 0;
            case 720: Functor<720>()(); return 0;
            case 750: Functor<750>()(); return 0;
            case 800: Functor<800>()(); return 0;
            case 860: Functor<860>()(); return 0;
            default: {
                if (cuda_device_arch > 800) {
                    Functor<800>()();
                    return 0;
                }
            }
        }
        return 1;
    }
} // namespace example

#endif // CUFFTDX_EXAMPLE_COMMON_HPP_
