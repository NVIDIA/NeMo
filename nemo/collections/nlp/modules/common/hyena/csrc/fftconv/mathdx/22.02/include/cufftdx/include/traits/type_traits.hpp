// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_TRAITS_TYPE_TRAITS_HPP
#define CUFFTDX_TRAITS_TYPE_TRAITS_HPP

#include <cuda_fp16.h>

#include "detail/make_complex_type.hpp"

namespace cufftdx {
    // Creates cuFFTDx complex type from scalar floating point type
    template<class T>
    struct make_complex_type;

    template<class T>
    using make_complex_type_t = typename make_complex_type<T>::type;

#define CUFFTDX_DETAIL_DEFINE_MAKE_COMPLEX_TYPE(scalar_type)                        \
    template<>                                                                      \
    struct make_complex_type<scalar_type> {                                         \
        using type = typename detail::make_complex_type<scalar_type>::cufftdx_type; \
    };

    CUFFTDX_DETAIL_DEFINE_MAKE_COMPLEX_TYPE(float)
    CUFFTDX_DETAIL_DEFINE_MAKE_COMPLEX_TYPE(double)
    CUFFTDX_DETAIL_DEFINE_MAKE_COMPLEX_TYPE(__half2)

#undef CUFFTDX_DETAIL_DEFINE_MAKE_COMPLEX_TYPE

} // namespace cufftdx

#endif // CUFFTDX_TRAITS_TYPE_TRAITS_HPP
