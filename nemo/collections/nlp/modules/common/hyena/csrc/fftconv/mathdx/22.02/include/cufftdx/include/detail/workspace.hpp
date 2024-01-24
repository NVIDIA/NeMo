// Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_DETAIL_WORKSPACE_HPP
#define CUFFTDX_DETAIL_WORKSPACE_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

#ifndef __CUDACC_RTC__
#   include <memory>
#   include <cassert>
#endif // __CUDACC_RTC__

#ifndef __CUDACC_RTC__
#   include <cuda_runtime_api.h>
#endif // __CUDACC_RTC__

#include "../traits.hpp"
#include "../traits/detail/bluestein_helpers.hpp"

#ifdef CUFFTDX_DETAIL_DEBUG
#    define CUFFTDX_DETAIL_CUDA_CHECK_AND_EXIT(error)                                                       \
        {                                                                                                   \
            auto status = static_cast<cudaError_t>(error);                                                  \
            if (status != cudaSuccess) {                                                                    \
                std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                                          \
            }                                                                                               \
        }
#else
#    define CUFFTDX_DETAIL_CUDA_CHECK_AND_EXIT(error) (void)error;
#endif // CUDA_CHECK_AND_EXIT

namespace cufftdx {
    namespace detail {
        struct empty_workspace {
            // CUDA device handle type
            struct device_handle {
                __forceinline__ __device__ bool valid() const { return true; }
            };

            // __host__ functions should not be visible for NVRTC
            #ifndef __CUDACC_RTC__
            static __forceinline__ __host__ empty_workspace create(cudaError_t& error_code) noexcept {
                error_code = cudaSuccess;
                return empty_workspace {};
            }

            __forceinline__ __host__ device_handle get_device_handle() const { return device_handle {}; }
            __forceinline__ __host__               operator device_handle() const { return get_device_handle(); }
            __forceinline__ __host__ bool          valid() const { return true; }
            __forceinline__ __host__ void          release() { }
            #endif // __CUDACC_RTC__
        };

        struct unknown_workspace {
            // CUDA device handle type
            struct device_handle {
                __forceinline__ __device__ bool valid() const { return false; }
            };

            // __host__ functions should not be visible for NVRTC
            #ifndef __CUDACC_RTC__
            static __forceinline__ __host__ unknown_workspace create(cudaError_t& error_code) noexcept {
                error_code = cudaSuccess;
                return unknown_workspace {};
            }

            __forceinline__ __host__ device_handle get_device_handle() const { return device_handle {}; }
            __forceinline__ __host__               operator device_handle() const { return get_device_handle(); }
            __forceinline__ __host__ bool          valid() const { return false; }
            __forceinline__ __host__ void          release() { }
            #endif // __CUDACC_RTC__
        };

        namespace __bluestein_workspace {
            template<class BluesteinFFT, unsigned int FFTSize, class FFTComplexType>
            __global__ __launch_bounds__(BluesteinFFT::max_threads_per_block) void kernel(FFTComplexType* w_time,
                                   FFTComplexType* w_freq,
                                   //    const unsigned int fft_size,
                                   //    const unsigned int fft_blue_size,
                                   const double theta0) {
                static_assert(BluesteinFFT::storage_size == BluesteinFFT::elements_per_thread,
                              "Workspace generation incorrectly configured");

                using blue_complex_type = typename BluesteinFFT::value_type;

                static constexpr auto blue_fft_size = size_of<BluesteinFFT>::value;

                blue_complex_type thread_w_time[BluesteinFFT::storage_size];
                // Generate w_time signal and store
                const unsigned int stride        = blue_fft_size / BluesteinFFT::elements_per_thread;
                unsigned int       index         = threadIdx.x;
                unsigned int       compute_index = index;
                for (unsigned int i = 0; i < BluesteinFFT::elements_per_thread; i++) {
                    if (index >= FFTSize) {
                        compute_index = blue_fft_size - index;
                    }
                    thread_w_time[i]      = 0;
                    blue_complex_type b_n = {0, 0};
                    if (compute_index < FFTSize) {
                        const double theta = theta0 * ((compute_index * compute_index) % (2 * FFTSize));
                        b_n.x              = cos(theta);
                        b_n.y              = sin(theta);
                        thread_w_time[i]   = b_n;
                    }
                    // Store conjugated value in w_time
                    b_n.y         = -b_n.y;
                    w_time[index] = FFTComplexType(b_n);
                    index += stride;
                    compute_index = index;
                }
                __syncthreads();

                // Calculate w_freq
                extern __shared__ unsigned char shared_mem[];
                BluesteinFFT().execute(thread_w_time, shared_mem);

                // Store w_freq
                index = threadIdx.x;
                for (unsigned int i = 0; i < BluesteinFFT::elements_per_thread; i++) {
                    w_freq[index] = FFTComplexType(thread_w_time[i]);
                    index += stride;
                }
            };
        } // namespace __bluestein_workspace

        template<unsigned int FFTSize, class FFTValueType, unsigned int SMValue>
        class bluestein_workspace: empty_workspace
        {
            using value_type = FFTValueType;

            // Replace size with next power-of-two, type with C2C, direction to forward, precision to double
            using bluestein_fft_type =
                typename CUFFTDX_STD::decay<decltype(Size<detail::get_bluestein_size(FFTSize)>() +
                                                     Direction<fft_direction::forward>() + Precision<double>() +
                                                     Type<fft_type::c2c>() + Block() + SM<SMValue>())>::type;

            // std::unique_ptr and std::shared_ptr undefined for NVRTC
            #ifndef __CUDACC_RTC__
            template<class Deleter>
            using d_value_type_uptr = std::unique_ptr<value_type, Deleter>;
            using d_value_type_sptr = std::shared_ptr<value_type>;
            #endif // __CUDACC_RTC__

            // Types with access to private/protected members
            template<class... Operators>
            friend class fft_block_execution;

            bluestein_workspace():
                w_time(nullptr), w_freq(nullptr) {}

        public:
            class device_handle
            {
                template<class... Operators>
                friend class fft_block_execution;

                friend class bluestein_workspace<FFTSize, FFTValueType, SMValue>;

                device_handle(value_type* w_time, value_type* w_freq):
                    w_time(w_time), w_freq(w_freq) {}

                __forceinline__ __device__ bool valid() const {
                    return (w_time != nullptr) && (w_freq != nullptr);
                }

                value_type* w_time;
                value_type* w_freq;
            };

            // __host__ functions should not be visible for NVRTC
            #ifndef __CUDACC_RTC__
            static __forceinline__ __host__ bluestein_workspace create(cudaError_t& error_code) noexcept {
                error_code = cudaSuccess;
                bluestein_workspace ws;

                static constexpr auto fft_size              = FFTSize;
                static constexpr auto fft_blue_size         = detail::get_bluestein_size(FFTSize);
                static constexpr auto ws_buffers_size_bytes = fft_blue_size * sizeof(value_type);

                auto deleter = [&error_code](value_type* ptr) { error_code = cudaFree(ptr); };
#ifdef CUFFTDX_DETAIL_DEBUG
                auto cuda_malloc = [](size_t size_bytes, cudaError_t& error_code) {
                    void* ptr  = nullptr;
                    error_code = cudaMallocManaged((void**)&ptr, size_bytes);
                    return (value_type*)ptr;
                };
#else
                auto cuda_malloc = [](size_t size_bytes, cudaError_t& error_code) {
                    void* ptr  = nullptr;
                    error_code = cudaMalloc((void**)&ptr, size_bytes);
                    return (value_type*)ptr;
                };
#endif

                d_value_type_uptr<decltype(deleter)> uptr_w_time(cuda_malloc(ws_buffers_size_bytes, error_code),
                                                                 deleter);
                if (error_code != cudaSuccess) {
                    CUFFTDX_DETAIL_CUDA_CHECK_AND_EXIT(error_code);
                    return ws;
                }
                d_value_type_uptr<decltype(deleter)> uptr_w_freq(cuda_malloc(ws_buffers_size_bytes, error_code),
                                                                 deleter);
                if (error_code != cudaSuccess) {
                    CUFFTDX_DETAIL_CUDA_CHECK_AND_EXIT(error_code);
                    return ws;
                }

                error_code = cudaMemset((void*)uptr_w_time.get(), 0, ws_buffers_size_bytes);
                if (error_code != cudaSuccess) {
                    CUFFTDX_DETAIL_CUDA_CHECK_AND_EXIT(error_code);
                    return ws;
                }
                error_code = cudaMemset((void*)uptr_w_freq.get(), 0, ws_buffers_size_bytes);
                if (error_code != cudaSuccess) {
                    CUFFTDX_DETAIL_CUDA_CHECK_AND_EXIT(error_code);
                    return ws;
                }
                // Increase max shared memory if needed
                error_code = cudaFuncSetAttribute(
                    (void*)__bluestein_workspace::kernel<bluestein_fft_type, FFTSize, FFTValueType>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    bluestein_fft_type::shared_memory_size);
                if (error_code != cudaSuccess) {
                    CUFFTDX_DETAIL_CUDA_CHECK_AND_EXIT(error_code);
                    return ws;
                }

                // Needs l-value for args
                auto tmp_w_time = uptr_w_time.get();
                auto tmp_w_freq = uptr_w_freq.get();

                double theta0 = M_PI / fft_size;
                void*  args[] = {&tmp_w_time, &tmp_w_freq, &theta0};
                dim3   blocks(1, 1, 1);
                error_code =
                    cudaLaunchKernel((void*)__bluestein_workspace::kernel<bluestein_fft_type, FFTSize, FFTValueType>,
                                     blocks,
                                     bluestein_fft_type::block_dim,
                                     args,
                                     bluestein_fft_type::shared_memory_size,
                                     0);
                if (error_code != cudaSuccess) {
                    CUFFTDX_DETAIL_CUDA_CHECK_AND_EXIT(error_code);
                    return ws;
                }
                error_code = cudaGetLastError();
                if (error_code != cudaSuccess) {
                    CUFFTDX_DETAIL_CUDA_CHECK_AND_EXIT(error_code);
                    return ws;
                }
                error_code = cudaDeviceSynchronize();
                if (error_code != cudaSuccess) {
                    CUFFTDX_DETAIL_CUDA_CHECK_AND_EXIT(error_code);
                    return ws;
                }

                auto shared_deleter = [](value_type* ptr) {
                    // User should handle and clear all CUDA RT errors which occurred before,
                    // thus we expect that cudaGetLastError() return cudaSuccess.
                    assert(cudaGetLastError() == cudaSuccess);
                    // We don't expect cudaFree to fail here
                    CUFFTDX_DETAIL_CUDA_CHECK_AND_EXIT(cudaFree(ptr));
                };
                ws.w_time = d_value_type_sptr(uptr_w_time.release(), shared_deleter);
                ws.w_freq = d_value_type_sptr(uptr_w_freq.release(), shared_deleter);
                return ws;
            }

            __forceinline__ __host__ device_handle get_device_handle() const {
                return device_handle {w_time.get(), w_freq.get()};
            }
            __forceinline__ __host__ operator device_handle() const { return get_device_handle(); }

            __forceinline__ __host__ bool valid() const {
                return (w_time.get() != nullptr) && (w_freq.get() != nullptr);
            }

            __forceinline__ __host__ void release() {
                w_time.reset();
                w_freq.reset();
            }
            #endif // __CUDACC_RTC__

        protected:
            #ifndef __CUDACC_RTC__
            d_value_type_sptr w_time;
            d_value_type_sptr w_freq;
            #endif // __CUDACC_RTC__
        };

        template<unsigned int  FFTSize,
                 class         FFTPrecisionType,
                 class         FFTValueType,
                 fft_direction FFTDirection,
                 fft_type      FFTType,
                 unsigned int  FFTSM,
                 bool          IsComplete>
        struct workspace_selector;


        template<unsigned int FFTSize,
                 typename FFTPrecisionType,
                 typename FFTValueType,
                 fft_direction FFTDirection,
                 fft_type      FFTType,
                 unsigned int  FFTSM>
        struct workspace_selector<FFTSize, FFTPrecisionType, FFTValueType, FFTDirection, FFTType, FFTSM, false> {
            using type = unknown_workspace;
        };

        namespace __workspace_selector {

            template<unsigned int FFTSize, typename FFTValueType, unsigned int FFTSM, bool IsBluesteinRequired>
            struct helper;

            template<unsigned int FFTSize, typename FFTValueType, unsigned int FFTSM>
            struct helper<FFTSize, FFTValueType, FFTSM, false> {
                using type = empty_workspace;
            };

            template<unsigned int FFTSize, typename FFTValueType, unsigned int FFTSM>
            struct helper<FFTSize, FFTValueType, FFTSM, true> {
                using type = bluestein_workspace<FFTSize, FFTValueType, FFTSM>;
            };
        } // namespace __workspace_selector

        template<unsigned int FFTSize,
                 typename FFTPrecisionType,
                 typename FFTValueType,
                 fft_direction FFTDirection,
                 fft_type      FFTType,
                 unsigned int  FFTSM>
        struct workspace_selector<FFTSize, FFTPrecisionType, FFTValueType, FFTDirection, FFTType, FFTSM, true> {
            using type = typename __workspace_selector::helper<
                FFTSize,
                FFTValueType,
                FFTSM,
                is_bluestein_required<FFTSize, FFTPrecisionType, FFTDirection, FFTType, FFTSM>::value>::type;
        };
    } // namespace detail

    template<class FFT>
    constexpr bool is_workspace_required() noexcept {
        return FFT::requires_workspace;
    }

    template<class FFT>
    auto make_workspace(cudaError_t& error_code) noexcept //
        -> typename FFT::host_workspace_type {
        static_assert(!CUFFTDX_STD::is_same<typename FFT::host_workspace_type, detail::unknown_workspace>::value,
                      "Workspace type unknown. FFT description isn't complete");
        using workspace_type = typename FFT::host_workspace_type;
        return workspace_type::create(error_code);
    }

    template<class FFT>
    auto make_workspace() -> typename FFT::host_workspace_type {
        cudaError_t error_code = cudaSuccess;
        auto        ws         = make_workspace<FFT>(error_code);
        if (error_code != cudaSuccess) {
            std::string error_what = cudaGetErrorString(error_code);
            throw std::runtime_error(error_what);
        }
        return ws;
    }
} // namespace cufftdx

#endif // CUFFTDX_DETAIL_WORKSPACE_HPP
