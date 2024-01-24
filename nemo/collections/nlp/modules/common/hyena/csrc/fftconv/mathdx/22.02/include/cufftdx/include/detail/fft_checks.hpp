// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_DETAIL_FFT_CHECKS_HPP
#define CUFFTDX_DETAIL_FFT_CHECKS_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

#include <cuda_fp16.h>

#include "../operators.hpp"
#include "../traits/detail/bluestein_helpers.hpp"

namespace cufftdx {
    namespace detail {

// SM70
#define CUFFTDX_DETAIL_SM700_FP16_MAX 16384
#define CUFFTDX_DETAIL_SM700_FP32_MAX 16384
#define CUFFTDX_DETAIL_SM700_FP64_MAX 8192
// SM72
#define CUFFTDX_DETAIL_SM720_FP16_MAX 16384
#define CUFFTDX_DETAIL_SM720_FP32_MAX 16384
#define CUFFTDX_DETAIL_SM720_FP64_MAX 8192
// SM75
#define CUFFTDX_DETAIL_SM750_FP16_MAX 4096
#define CUFFTDX_DETAIL_SM750_FP32_MAX 4096
#define CUFFTDX_DETAIL_SM750_FP64_MAX 2048
// SM80
#define CUFFTDX_DETAIL_SM800_FP16_MAX 32768
#define CUFFTDX_DETAIL_SM800_FP32_MAX 32768
#define CUFFTDX_DETAIL_SM800_FP64_MAX 16384
// SM86
#define CUFFTDX_DETAIL_SM860_FP16_MAX 16384
#define CUFFTDX_DETAIL_SM860_FP32_MAX 16384
#define CUFFTDX_DETAIL_SM860_FP64_MAX 8192

        template<class Precision, unsigned int Size, unsigned int Arch>
        class is_supported: public CUFFTDX_STD::false_type
        {};

        // Max supported sizes, ignores SM
        template<class Precision, unsigned int Size>
        class is_supported<Precision, Size, unsigned(-1)>
        {
            static constexpr auto blue_size = detail::get_bluestein_size(Size);

        public:
            static constexpr bool fp16_block_value =
                CUFFTDX_STD::is_same<__half, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM800_FP16_MAX) && (Size >= 2));
            static constexpr bool fp32_block_value =
                CUFFTDX_STD::is_same<float, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM800_FP32_MAX) && (Size >= 2));
            static constexpr bool fp64_block_value =
                CUFFTDX_STD::is_same<double, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM800_FP64_MAX) && (Size >= 2));
            static constexpr bool blue_block_value = ((blue_size <= CUFFTDX_DETAIL_SM800_FP64_MAX) && (blue_size >= 2));

            static constexpr bool value = fp16_block_value || fp32_block_value || fp64_block_value || blue_block_value;
        };

        // SM70
        template<class Precision, unsigned int Size>
        class is_supported<Precision, Size, 700>
        {
            static constexpr auto blue_size = detail::get_bluestein_size(Size);

        public:
            static constexpr bool fp16_block_value =
                CUFFTDX_STD::is_same<__half, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM700_FP16_MAX) && (Size >= 2));
            static constexpr bool fp32_block_value =
                CUFFTDX_STD::is_same<float, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM700_FP32_MAX) && (Size >= 2));
            static constexpr bool fp64_block_value =
                CUFFTDX_STD::is_same<double, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM700_FP64_MAX) && (Size >= 2));
            static constexpr bool blue_block_value = ((blue_size <= CUFFTDX_DETAIL_SM700_FP64_MAX) && (blue_size >= 2));

            static constexpr bool value = fp16_block_value || fp32_block_value || fp64_block_value || blue_block_value;
        };

        // SM72
        template<class Precision, unsigned int Size>
        class is_supported<Precision, Size, 720>
        {
            static constexpr auto blue_size = detail::get_bluestein_size(Size);

        public:
            static constexpr bool fp16_block_value =
                CUFFTDX_STD::is_same<__half, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM720_FP16_MAX) && (Size >= 2));
            static constexpr bool fp32_block_value =
                CUFFTDX_STD::is_same<float, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM720_FP32_MAX) && (Size >= 2));
            static constexpr bool fp64_block_value =
                CUFFTDX_STD::is_same<double, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM720_FP64_MAX) && (Size >= 2));
            static constexpr bool blue_block_value = ((blue_size <= CUFFTDX_DETAIL_SM720_FP64_MAX) && (blue_size >= 2));

            static constexpr bool value = fp16_block_value || fp32_block_value || fp64_block_value || blue_block_value;
        };

        // SM75
        template<class Precision, unsigned int Size>
        class is_supported<Precision, Size, 750>
        {
            static constexpr auto blue_size = detail::get_bluestein_size(Size);

        public:
            static constexpr bool fp16_block_value =
                CUFFTDX_STD::is_same<__half, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM750_FP16_MAX) && (Size >= 2));
            static constexpr bool fp32_block_value =
                CUFFTDX_STD::is_same<float, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM750_FP32_MAX) && (Size >= 2));
            static constexpr bool fp64_block_value =
                CUFFTDX_STD::is_same<double, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM750_FP64_MAX) && (Size >= 2));
            static constexpr bool blue_block_value = ((blue_size <= CUFFTDX_DETAIL_SM750_FP64_MAX) && (blue_size >= 2));

            static constexpr bool value = fp16_block_value || fp32_block_value || fp64_block_value || blue_block_value;
        };

        // SM80
        template<class Precision, unsigned int Size>
        class is_supported<Precision, Size, 800>
        {
            static constexpr auto blue_size = detail::get_bluestein_size(Size);

        public:
            static constexpr bool fp16_block_value =
                CUFFTDX_STD::is_same<__half, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM800_FP16_MAX) && (Size >= 2));
            static constexpr bool fp32_block_value =
                CUFFTDX_STD::is_same<float, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM800_FP32_MAX) && (Size >= 2));
            static constexpr bool fp64_block_value =
                CUFFTDX_STD::is_same<double, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM800_FP64_MAX) && (Size >= 2));
            static constexpr bool blue_block_value = ((blue_size <= CUFFTDX_DETAIL_SM800_FP64_MAX) && (blue_size >= 2));

            static constexpr bool value = fp16_block_value || fp32_block_value || fp64_block_value || blue_block_value;
        };

        // SM86
        template<class Precision, unsigned int Size>
        class is_supported<Precision, Size, 860>
        {
            static constexpr auto blue_size = detail::get_bluestein_size(Size);

        public:
            static constexpr bool fp16_block_value =
                CUFFTDX_STD::is_same<__half, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM860_FP16_MAX) && (Size >= 2));
            static constexpr bool fp32_block_value =
                CUFFTDX_STD::is_same<float, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM860_FP32_MAX) && (Size >= 2));
            static constexpr bool fp64_block_value =
                CUFFTDX_STD::is_same<double, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM860_FP64_MAX) && (Size >= 2));
            static constexpr bool blue_block_value = ((blue_size <= CUFFTDX_DETAIL_SM860_FP64_MAX) && (blue_size >= 2));

            static constexpr bool value = fp16_block_value || fp32_block_value || fp64_block_value || blue_block_value;
        };
    } // namespace detail
} // namespace cufftdx

#endif // CUFFTDX_DETAIL_FFT_CHECKS_HPP
