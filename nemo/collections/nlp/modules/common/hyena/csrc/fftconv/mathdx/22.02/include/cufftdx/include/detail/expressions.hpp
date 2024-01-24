// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_DETAIL_EXPRESSIONS_HPP
#define CUFFTDX_DETAIL_EXPRESSIONS_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

namespace cufftdx {
    namespace detail {
        struct expression {};
        struct operator_expression: expression {};
        struct block_operator_expression: operator_expression {};

        struct description_expression: expression {};
        struct execution_description_expression: description_expression {};

        template<class ValueType, ValueType Value>
        struct constant_operator_expression:
            public operator_expression,
            public CUFFTDX_STD::integral_constant<ValueType, Value> {};

        template<class ValueType, ValueType Value>
        struct constant_block_operator_expression:
            public block_operator_expression,
            public CUFFTDX_STD::integral_constant<ValueType, Value> {};
    } // namespace detail
} // namespace cufftdx

#endif // CUFFTDX_DETAIL_EXPRESSIONS_HPP
