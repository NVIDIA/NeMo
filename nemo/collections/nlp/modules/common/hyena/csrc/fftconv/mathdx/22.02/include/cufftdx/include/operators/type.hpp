// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_OPERATORS_TYPE_HPP
#define CUFFTDX_OPERATORS_TYPE_HPP

#include "../detail/expressions.hpp"

namespace cufftdx {
    enum class fft_type
    {
        c2c,
        c2r,
        r2c
    };

    template<fft_type Value>
    struct Type: public detail::constant_operator_expression<fft_type, Value> {};
} // namespace cufftdx

#endif // CUFFTDX_OPERATORS_TYPE_HPP
