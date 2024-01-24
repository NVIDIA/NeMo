// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_OPERATORS_SIZE_HPP
#define CUFFTDX_OPERATORS_SIZE_HPP

#include "../detail/expressions.hpp"

namespace cufftdx {
    template<unsigned int Value>
    struct Size: public detail::constant_operator_expression<unsigned int, Value> {
        static_assert(Value > 1, "FFT size must be greater than 1");
    };
} // namespace cufftdx

#endif // CUFFTDX_OPERATORS_SIZE_HPP
