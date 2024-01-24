// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_DETAIL_SYSTEM_CHECKS_HPP
#define CUFFTDX_DETAIL_SYSTEM_CHECKS_HPP

// We require target architecture to be Volta+ (only checking on device)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#   error "cuFFTDx requires GPU architecture sm_70 or higher");
#endif

#ifdef __CUDACC_RTC__

// NVRTC version check
#    ifndef CUFFTDX_IGNORE_DEPRECATED_COMPILER
#        if (__CUDACC_VER_MAJOR__ < 11)
#            error cuFFTDx requires NVRTC from CUDA Toolkit 11.0 or newer
#        endif
#    endif // CUFFTDX_IGNORE_DEPRECATED_COMPILER

// NVRTC compilation checks
#    ifndef CUFFTDX_IGNORE_DEPRECATED_COMPILER
static_assert(__CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 0,
              "cuFFTDx requires CUDA Runtime 11.0 or newer to work with NVRTC");
#    endif // CUFFTDX_IGNORE_DEPRECATED_COMPILER

#else
#    include <cuda.h>

// NVCC compilation

static_assert(CUDART_VERSION >= 11000, "cuFFTDx requires CUDA Runtime 11.0 or newer");
static_assert(CUDA_VERSION >= 11000, "cuFFTDx requires CUDA Toolkit 11.0 or newer");

#    ifndef CUFFTDX_IGNORE_DEPRECATED_COMPILER

// Test for GCC 7+
#        if defined(__GNUC__) && !defined(__clang__)
#            if (__GNUC__ < 7)
#                error cuFFTDx requires GCC in version 7 or newer
#            endif
#        endif // __GNUC__

// Test for clang 9+
#        ifdef __clang__
#            if (__clang_major__ < 9)
#                error cuFFTDx requires clang in version 9 or newer (experimental support for clang as host compiler)
#            endif
#        endif // __clang__

// MSVC (Visual Studio) is not supported
#        ifdef _MSC_VER
#            error cuFFTDx does not support compilation with MSVC
#        endif // _MSC_VER

#    endif // CUFFTDX_IGNORE_DEPRECATED_COMPILER

#endif // __CUDACC_RTC__

// C++ Version
#ifndef CUFFTDX_IGNORE_DEPRECATED_DIALECT
#    if (__cplusplus < 201703L)
#        error cuFFTDx requires C++17 (or newer) enabled
#    endif
#endif // CUFFTDX_IGNORE_DEPRECATED_DIALECT

#endif // CUFFTDX_DETAIL_SYSTEM_CHECKS_HPP
