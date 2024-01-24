// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_OPERATORS_PRECISION_HPP
#define CUFFTDX_OPERATORS_PRECISION_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

#include <cuda_fp16.h>

#include "../detail/expressions.hpp"

namespace cufftdx {
    namespace detail {
        template<class T>
        struct is_supported_fp_type:
            CUFFTDX_STD::integral_constant<bool,
                                   CUFFTDX_STD::is_same<float, typename CUFFTDX_STD::remove_cv<T>::type>::value ||
                                       CUFFTDX_STD::is_same<double, typename CUFFTDX_STD::remove_cv<T>::type>::value ||
                                       CUFFTDX_STD::is_same<__half, typename CUFFTDX_STD::remove_cv<T>::type>::value> {};
    } // namespace detail

    template<class T = float>
    struct Precision: detail::operator_expression {
        using type = typename CUFFTDX_STD::remove_cv<T>::type;
        static_assert(detail::is_supported_fp_type<type>::value, "Precision must be double, float, or __half.");
    };
} // namespace cufftdx

#endif // CUFFTDX_OPERATORS_TYPE_HPP
