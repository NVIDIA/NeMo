// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_OPERATORS_BLOCK_OPERATORS_HPP
#define CUFFTDX_OPERATORS_BLOCK_OPERATORS_HPP

#include "../detail/expressions.hpp"

namespace cufftdx {
    template<unsigned int Value>
    struct FFTsPerBlock: detail::constant_block_operator_expression<unsigned int, Value> {};

    template<unsigned int Value>
    struct ElementsPerThread: detail::constant_block_operator_expression<unsigned int, Value> {};

    template<unsigned int X, unsigned int Y = 1, unsigned int Z = 1>
    struct BlockDim: detail::block_operator_expression {
        static constexpr unsigned int x     = X;
        static constexpr unsigned int y     = Y;
        static constexpr unsigned int z     = Z;
        static constexpr dim3         value = dim3(x, y, z);

        static constexpr unsigned int flat_size = x * y * z;
        static constexpr unsigned int rank      = (x != 1) + (y != 1) + (z != 1);
    };
} // namespace cufftdx

#endif // CUFFTDX_OPERATORS_BLOCK_OPERATORS_HPP
