// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_OPERATORS_SM_HPP
#define CUFFTDX_OPERATORS_SM_HPP

#include "../detail/expressions.hpp"

namespace cufftdx {
    template<unsigned int Architecture>
    struct SM;

    template<>
    struct SM<700>: public detail::constant_operator_expression<unsigned int, 700> {};

    template<>
    struct SM<720>: public detail::constant_operator_expression<unsigned int, 720> {};

    template<>
    struct SM<750>: public detail::constant_operator_expression<unsigned int, 750> {};

    template<>
    struct SM<800>: public detail::constant_operator_expression<unsigned int, 800> {};

    template<>
    struct SM<860>: public detail::constant_operator_expression<unsigned int, 860> {};
} // namespace cufftdx

#endif // CUFFTDX_OPERATORS_SM_HPP
