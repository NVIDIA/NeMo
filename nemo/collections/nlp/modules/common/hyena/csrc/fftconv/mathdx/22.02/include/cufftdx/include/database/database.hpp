// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_DATABASE_DATABASE_HPP
#define CUFFTDX_DATABASE_DATABASE_HPP

#include "detail/block_fft.hpp"

namespace cufftdx {
    namespace database {
        namespace detail {
            #include "lut_fp32.hpp.inc"
            #include "lut_fp64.hpp.inc"


            #include "records/700/database_fp16_fwd.hpp.inc"
            #include "records/700/database_fp16_inv.hpp.inc"
            #include "records/700/database_fp32_fwd.hpp.inc"
            #include "records/700/database_fp32_inv.hpp.inc"
            #include "records/700/database_fp64_fwd.hpp.inc"
            #include "records/700/database_fp64_inv.hpp.inc"

            #include "records/800/database_fp16_fwd.hpp.inc"
            #include "records/800/database_fp16_inv.hpp.inc"
            #include "records/800/database_fp32_fwd.hpp.inc"
            #include "records/800/database_fp32_inv.hpp.inc"
            #include "records/800/database_fp64_fwd.hpp.inc"
            #include "records/800/database_fp64_inv.hpp.inc"

#ifndef __HALF2_TO_UI
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))
#endif

            #include "records/definitions_fp16_fwd.hpp.inc"
            #include "records/definitions_fp16_inv.hpp.inc"
            #include "records/definitions_fp32_fwd.hpp.inc"
            #include "records/definitions_fp32_inv.hpp.inc"
            #include "records/definitions_fp64_fwd.hpp.inc"
            #include "records/definitions_fp64_inv.hpp.inc"

#ifdef __HALF2_TO_UI
#undef __HALF2_TO_UI
#endif

        } // namespace detail
    }     // namespace database
} // namespace cufftdx

#endif // CUFFTDX_DATABASE_DATABASE_HPP

