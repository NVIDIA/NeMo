// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_TRAITS_DETAIL_BLUESTEIN_HELPERS_HPP
#define CUFFTDX_TRAITS_DETAIL_BLUESTEIN_HELPERS_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

#include "../../database/database.hpp"

namespace cufftdx {
    namespace detail {
        // Return Bluestein size required to calculate FFT
        __device__  __host__ __forceinline__
        constexpr unsigned int get_bluestein_size(const unsigned int fft_size) {
            return
                fft_size <= 16 ? 32 :
                (fft_size <= 32 ? 64 :
                (fft_size <= 64 ? 128 :
                (fft_size <= 128 ? 256 :
                (fft_size <= 256 ? 512 :
                (fft_size <= 512 ? 1024 :
                (fft_size <= 1024 ? 2048 :
                (fft_size <= 2048 ? 4096 :
                (fft_size <= 4096 ? 8192 :
                (fft_size <= 8192 ? 16384 : 0)))))))));
        }

        template<class Description>
        __device__ __host__ __forceinline__
        constexpr unsigned int get_bluestein_size() {
            return get_bluestein_size(size_of<Description>::value);
        }

        template<unsigned int FFTSize>
        __device__ __host__ __forceinline__ constexpr bool is_bluestein_supported() {
            return get_bluestein_size(FFTSize) > 0;
        }

        template<unsigned int  FFTSize,
                 class         FFTPrecisionType,
                 fft_direction FFTDirection,
                 fft_type      FFTType,
                 unsigned int  FFTSM>
        struct is_bluestein_required {
        private:
            // Search for record in database
            using block_fft_record_t =
                database::detail::block_fft_record<FFTSize, FFTPrecisionType, FFTType, FFTDirection, FFTSM>;
        public:
            static constexpr bool value = !block_fft_record_t::defined;
        };

        template<unsigned int FFTSize,
                 typename FFTPrecisionType,
                 fft_direction FFTDirection,
                 fft_type      FFTType,
                 unsigned int  FFTSM>
        constexpr bool is_bluestein_required<FFTSize, FFTPrecisionType, FFTDirection, FFTType, FFTSM>::value;
    } // namespace detail
} // namespace cufftdx

#endif // CUFFTDX_TRAITS_DETAIL_BLUESTEIN_HELPERS_HPP
