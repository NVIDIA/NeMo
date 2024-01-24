// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_OPERATORS_HPP__
#define CUFFTDX_OPERATORS_HPP__

#include "operators/block_operators.hpp"
#include "operators/direction.hpp"
#include "operators/execution_operators.hpp"
#include "operators/precision.hpp"
#include "operators/size.hpp"
#include "operators/sm.hpp"
#include "operators/type.hpp"

namespace cufftdx {
    enum class fft_operator
    {
        direction,
        precision,
        size,
        sm,
        type,
        // execution
        thread,
        block,
        // block-only
        elements_per_thread,
        ffts_per_block,
        block_dim,
    };
} // namespace cufftdx

#endif // CUFFTDX_OPERATORS_HPP__
