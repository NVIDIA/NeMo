// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_TRAITS_DETAIL_LDG_TYPE_HPP
#define CUFFTDX_TRAITS_DETAIL_LDG_TYPE_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

#include <cuda_fp16.h>

#include "../../types.hpp"

namespace cufftdx {
    namespace detail {
        template<class T>
        struct ldg_type {
            using type = void;
        };

#define CUFFTDX_DETAIL_DEFINE_LDG_TYPE(mytype, ldgtype) \
    template<>                                          \
    struct ldg_type<mytype> {                           \
        using type = ldgtype;                           \
    };

        CUFFTDX_DETAIL_DEFINE_LDG_TYPE(__half, __half)
        CUFFTDX_DETAIL_DEFINE_LDG_TYPE(float, float)
        CUFFTDX_DETAIL_DEFINE_LDG_TYPE(double, double)
        CUFFTDX_DETAIL_DEFINE_LDG_TYPE(::cufftdx::detail::complex<__half2>, float2)
        CUFFTDX_DETAIL_DEFINE_LDG_TYPE(::cufftdx::detail::complex<float>, float2)
        CUFFTDX_DETAIL_DEFINE_LDG_TYPE(::cufftdx::detail::complex<double>, double2)


#undef CUFFTDX_DETAIL_DEFINE_LDG_TYPE
    } // namespace detail
} // namespace cufftdx

#endif // CUFFTDX_TRAITS_DETAIL_LDG_TYPE_HPP
