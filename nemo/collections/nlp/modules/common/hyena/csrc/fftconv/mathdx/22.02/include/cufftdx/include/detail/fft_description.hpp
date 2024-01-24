// Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_DETAIL_FFT_DESCRIPTION_HPP
#define CUFFTDX_DETAIL_FFT_DESCRIPTION_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

#include <cuda_fp16.h>

#include "../operators.hpp"
#include "../traits/detail/check_and_get_trait.hpp"
#include "../traits/detail/description_traits.hpp"
#include "../traits/detail/get.hpp"
#include "../traits/detail/make_complex_type.hpp"
#include "../database/database.hpp"

#include "expressions.hpp"

namespace cufftdx {
    namespace detail {
        template<class... Operators>
        class fft_operator_wrapper: public description_expression { };

        template<class... Operators>
        class fft_description: public description_expression
        {
            using description_type = fft_operator_wrapper<Operators...>;

        protected:
            /// ---- Traits

            // Size
            // * Default value: NONE
            // * If there is no size, then dummy size is 2. This is required so this_fft_size_v does not break.
            // * Values of has_size or is_complete should be checked before using this property.
            static constexpr bool has_size = has_operator<fft_operator::size, description_type>::value;
            using dummy_default_fft_size   = Size<2>;
            using this_fft_size = get_or_default_t<fft_operator::size, description_type, dummy_default_fft_size>;
            static constexpr auto this_fft_size_v = this_fft_size::value;

            // Type (C2C, C2R, R2C)
            // * Default value: C2C
            using default_fft_type = Type<fft_type::c2c>;
            using this_fft_type    = get_or_default_t<fft_operator::type, description_type, default_fft_type>;
            static constexpr auto this_fft_type_v = this_fft_type::value;

            // Direction
            // * Default value: NONE
            // * Direction can be deduced from FFT Type
            // * If there is no direction and we can't deduced it, dummy direction is FORWARD.This is required so
            // this_fft_direction_v does not break.
            // * Values of has_size or is_complete should be checked before using this property.
            static constexpr bool has_direction = has_operator<fft_operator::direction, description_type>::value;
            using deduced_fft_direction         = deduce_direction_type_t<this_fft_type>;
            using dummy_default_fft_direction   = Direction<fft_direction::forward>;
            using this_fft_direction =
                get_or_default_t<fft_operator::direction,
                                 description_type,
                                 typename CUFFTDX_STD::conditional<!CUFFTDX_STD::is_void<deduced_fft_direction>::value,
                                                           deduced_fft_direction,
                                                           dummy_default_fft_direction>::type>;
            static constexpr auto this_fft_direction_v = this_fft_direction::value;

            // Precision
            // * Default: float
            using default_fft_precision = Precision<float>;
            using this_fft_precision =
                get_or_default_t<fft_operator::precision, description_type, default_fft_precision>;
            using this_fft_precision_t = typename this_fft_precision::type;

            // True if description is complete FFT description
            static constexpr bool is_complete = is_complete_description<description_type>::value;

            // SM
            static constexpr bool has_sm = has_operator<fft_operator::sm, description_type>::value;
            using dummy_default_fft_sm   = SM<700>;
            using this_fft_sm            = get_or_default_t<fft_operator::sm, description_type, dummy_default_fft_sm>;
            static constexpr auto this_fft_sm_v = this_fft_sm::value;

            /// ---- Constraints

            // Not-implemented-yet / disabled features

            static constexpr bool has_block_dim = has_operator<fft_operator::block_dim, description_type>::value;
#ifndef CUFFTDX_DETAIL_TEST_ENABLE_BLOCKDIM
            static_assert(!has_block_dim, "BlockDim<> feature is not implemented yet");
#endif

            // We can only have one of each option

            // Main operators
            static constexpr bool has_one_direction =
                has_at_most_one_of<fft_operator::direction, description_type>::value;
            static constexpr bool has_one_precision =
                has_at_most_one_of<fft_operator::precision, description_type>::value;
            static constexpr bool has_one_size   = has_at_most_one_of<fft_operator::size, description_type>::value;
            static constexpr bool has_one_sm     = has_at_most_one_of<fft_operator::sm, description_type>::value;
            static constexpr bool has_one_type   = has_at_most_one_of<fft_operator::type, description_type>::value;

            static_assert(has_one_direction, "Can't create FFT with two Direction<> expressions");
            static_assert(has_one_precision, "Can't create FFT with two Precision<> expressions");
            static_assert(has_one_size, "Can't create FFT with two Size<> expressions");
            static_assert(has_one_sm, "Can't create FFT with two SM<> expressions");
            static_assert(has_one_type, "Can't create FFT with two Type<> expressions");

            // Block-only operators
            static constexpr bool has_one_ept =
                has_at_most_one_of<fft_operator::elements_per_thread, description_type>::value;
            static constexpr bool has_one_fpb =
                has_at_most_one_of<fft_operator::ffts_per_block, description_type>::value;
            static constexpr bool has_one_block_dim =
                has_at_most_one_of<fft_operator::block_dim, description_type>::value;

            static_assert(has_one_ept, "Can't create FFT with two ElementsPerThread<> expressions");
            static_assert(has_one_fpb, "Can't create FFT with two FFTsPerBlock<> expressions");
            static_assert(has_one_block_dim, "Can't create FFT with two BlockDim<> expressions");

            // Mutually exclusive options
            static constexpr bool c2r_type_forward_dir =
                !has_direction || !(CUFFTDX_STD::is_same<this_fft_type, Type<fft_type::c2r>>::value &&
                                    CUFFTDX_STD::is_same<this_fft_direction, Direction<fft_direction::forward>>::value);
            static constexpr bool r2c_type_inverse_dir =
                !has_direction || !(CUFFTDX_STD::is_same<this_fft_type, Type<fft_type::r2c>>::value &&
                                    CUFFTDX_STD::is_same<this_fft_direction, Direction<fft_direction::inverse>>::value);

            static_assert(c2r_type_forward_dir, "Can't create Complex-to-Real FFT with forward direction");
            static_assert(r2c_type_inverse_dir, "Can't create Real-to-Complex FFT with inverse direction");

            /// ---- End of Constraints
        };

        template<>
        class fft_description<>: public description_expression {};
    } // namespace detail
} // namespace cufftdx

#endif // CUFFTDX_DETAIL_FFT_DESCRIPTION_HPP
