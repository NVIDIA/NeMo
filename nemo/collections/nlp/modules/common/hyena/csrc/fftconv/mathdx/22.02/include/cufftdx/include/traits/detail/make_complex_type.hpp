// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_TRAITS_DETAIL_MAKE_COMPLEX_TYPE_HPP
#define CUFFTDX_TRAITS_DETAIL_MAKE_COMPLEX_TYPE_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

#include <cuda_fp16.h>

#include "../../operators/precision.hpp" // is_supported_fp_type
#include "../../types.hpp"

namespace cufftdx {
    namespace detail {
        template<class T>
        struct make_complex_type {
            static_assert(detail::is_supported_fp_type<T>::value,
                          "Only double, float, and __half floating-point types are supported");
        };

#define CUFFTDX_DETAIL_DEFINE_MAKE_COMPLEX_TYPE(scalar_type)          \
    template<>                                                        \
    struct make_complex_type<scalar_type> {                           \
        using cufftdx_type = ::cufftdx::detail::complex<scalar_type>; \
    };

        CUFFTDX_DETAIL_DEFINE_MAKE_COMPLEX_TYPE(float)
        CUFFTDX_DETAIL_DEFINE_MAKE_COMPLEX_TYPE(double)

        template<>
        struct make_complex_type<__half> {
            using cufftdx_type = ::cufftdx::detail::complex<__half2>;
        };

        template<>
        struct make_complex_type<__half2> {
            using cufftdx_type = ::cufftdx::detail::complex<__half2>;
        };

#undef CUFFTDX_DETAIL_DEFINE_MAKE_COMPLEX_TYPE
    } // namespace detail
} // namespace cufftdx

#endif // CUFFTDX_TRAITS_DETAIL_MAKE_COMPLEX_TYPE_HPP
