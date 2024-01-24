// Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_TRAITS_DETAIL_CHECK_AND_GET_TRAIT_HPP
#define CUFFTDX_TRAITS_DETAIL_CHECK_AND_GET_TRAIT_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#    include <cuda/std/utility>
#else
#    include <type_traits>
#    include <utility>
#endif

#include <cuda_fp16.h>

#include "../../operators.hpp"
#include "../../detail/expressions.hpp"
#include "../../database/database.hpp"

#include "../fft_traits.hpp"
#include "../replace.hpp"

#include "get.hpp"
#include "description_traits.hpp"
#include "bluestein_helpers.hpp"

namespace cufftdx {
    namespace detail {
        template<fft_operator Operator, class Description>
        class check_and_get_trait;

        namespace __get_block_config {
            template<class Description>
            class helper_block_ct
            {
                // Using SIZE, PRECISION, ARCHITECTURE, we search for optimal EPT and FFTs per block (FPB, BPB)
                static constexpr unsigned int this_fft_size_v = size_of<Description>::value;
                using this_fft_precision_t                    = precision_of_t<Description>;
                static constexpr auto this_fft_direction_v    = direction_of<Description>::value;
                static constexpr auto this_fft_type_v         = type_of<Description>::value;
                static constexpr auto this_fft_sm_v           = sm_of<Description>::value;

                // Select block_fft implementation
                // * database::detail::block_fft_record has all possible implementations in type_list named "blobs"
                // * first implementation from blobs is considered default/suggested/optimal
                using block_fft_record_t = database::detail::block_fft_record<this_fft_size_v,
                                                                              this_fft_precision_t,
                                                                              this_fft_type_v,
                                                                              this_fft_direction_v,
                                                                              this_fft_sm_v>;
                // Checks if record for requested (size, precision, type, direction, arch) exists
                static_assert(block_fft_record_t::defined, "This FFT configuration is not supported");

                // Get default (optimal) implementation
                using suggested_block_config_t =
                    typename database::detail::type_list_element<0, typename block_fft_record_t::blobs>::type;

                // Get suggested EPT and FPB
                using suggested_ept = ElementsPerThread<suggested_block_config_t::elements_per_thread>;
                using suggested_fpb = FFTsPerBlock<suggested_block_config_t::ffts_per_block>;

                // Get selected EPT (suggested or provided by user)
                using this_fft_elements_per_thread =
                    get_or_default_t<fft_operator::elements_per_thread, Description, suggested_ept>;
                static constexpr bool has_ept = has_operator<fft_operator::elements_per_thread, Description>::value;
                static constexpr auto this_fft_ept_v = this_fft_elements_per_thread::value;

                #ifdef CUFFTDX_DETAIL_BLOCK_FFT_ENFORCE_X_TRANSPOSITION
                static constexpr unsigned int this_fft_trp_option_v = 1;
                #elif defined(CUFFTDX_DETAIL_BLOCK_FFT_ENFORCE_XY_TRANSPOSITION)
                static constexpr unsigned int this_fft_trp_option_v = 2;
                #else
                static constexpr unsigned int this_fft_trp_option_v = 0;
                #endif

                // Search for implementation
                using this_fft_block_fft_implementation =
                    typename database::detail::search_by_ept<this_fft_ept_v,
                                                             this_fft_precision_t,
                                                             this_fft_trp_option_v,
                                                             typename block_fft_record_t::blobs>::type;
                // Checks if implementation for requested EPT exists within selected record
                static_assert(!CUFFTDX_STD::is_void<this_fft_block_fft_implementation>::value,
                              "This FFT configuration is not supported");

                // suggested_fpb is not used as default fpb
                // For fp16 FPB must be even, each thread processes two half complex numbers
                static constexpr auto default_fpb_v = CUFFTDX_STD::is_same<this_fft_precision_t, __half>::value ? 2 : 1;
                using default_fpb                   = FFTsPerBlock<default_fpb_v>;
                using this_ffts_per_block = get_or_default_t<fft_operator::ffts_per_block, Description, default_fpb>;
                static constexpr auto this_ffts_per_block_v = this_ffts_per_block::value;
                static_assert(!CUFFTDX_STD::is_same<this_fft_precision_t, __half>::value || (this_ffts_per_block_v % 2 == 0),
                              "FP16 block FFT can only process even number of FFTs per block");

                static constexpr unsigned int default_block_dim_v =
                    this_fft_size_v / this_fft_elements_per_thread::value;
                // Default block dimension (X = SIZE/EPT, Y = FFTs Per Block, Z = 1)
                using default_block_dim =
                    BlockDim<default_block_dim_v,
                             (CUFFTDX_STD::is_same<this_fft_precision_t, __half>::value ? this_ffts_per_block::value / 2
                                                                                : this_ffts_per_block::value)>;
                using this_block_dim = get_or_default_t<fft_operator::block_dim, Description, default_block_dim>;

            public:
                // Searches database for optimal default
                using elements_per_thread = this_fft_elements_per_thread;
                // Default calculated based on size and ept
                using block_dim = this_block_dim;
                // Defaults to 1
                using ffts_per_block = this_ffts_per_block;
                // FFT implementation type
                using block_fft_implementation = this_fft_block_fft_implementation;

                // Suggested values of EPT and FPB
                using suggested_elements_per_thread = suggested_ept;
                // If user set EPT, suggested FPB is for that EPT; otherwise it's for default EPT
                using suggested_ffts_per_block =
                    typename CUFFTDX_STD::conditional<has_ept,
                                                      FFTsPerBlock<this_fft_block_fft_implementation::ffts_per_block>,
                                                      suggested_fpb>::type;

                //
                static constexpr bool use_bluestein = false;
                static constexpr unsigned int workspace_size = 0;

            private:
                // Checks

                // Must specify EPT if user specified BlockDim
                static constexpr bool has_block_dim = has_operator<fft_operator::block_dim, Description>::value;
                static_assert(!has_block_dim || (has_block_dim && has_ept),
                              "If BlockDim<> was specifided, user must also specify ElementsPerThread<>");

                // SIZE % EPT == 0
                static constexpr bool ept_is_factor_of_size = (this_fft_size_v % elements_per_thread::value) == 0;
                static_assert(ept_is_factor_of_size, "Elements per thread must be a factor of FFT size");

                // SIZE * FFTS_PER_BLOCK <= EPT * FLAT_BLOCK_SIZE
                static constexpr auto max_elements_processed_per_block =
                    block_dim::flat_size * elements_per_thread::value *
                    (CUFFTDX_STD::is_same<this_fft_precision_t, __half>::value ? 2 : 1);
                static constexpr auto elements_to_process_per_block = this_fft_size_v * ffts_per_block::value;
                static_assert(elements_to_process_per_block <= max_elements_processed_per_block,
                              "Not enough threads in block to calculate FFT, you need to increase BlockDim<> or "
                              "ElementsPerThread<>");

                // FLAT_BLOCK_SIZE % FFTS_PER_BLOCK == 0
                static constexpr bool fpb_is_factor_of_flat_block_size =
                    (block_dim::flat_size % ffts_per_block::value) == 0;
                static_assert(ept_is_factor_of_size || has_block_dim,
                              "Elements per thread must be a factor of FFT size");
            };

            template<class Description>
            class helper_block_bluestein
            {
                static constexpr unsigned int this_fft_size_v      = size_of<Description>::value;
                static constexpr unsigned int this_fft_blue_size_v = get_bluestein_size(this_fft_size_v);

                // Create description with size changed to this_fft_blue_size_v
                using bluestein_description    = typename detail::replace_force<Description,
                                                                             true,
                                                                             Type<fft_type::c2c>,
                                                                             Size<this_fft_blue_size_v>,
                                                                             Direction<fft_direction::forward>>::type;
                using bluestein_block_helper_t = helper_block_ct<bluestein_description>;

            public:
                using elements_per_thread      = typename bluestein_block_helper_t::elements_per_thread;
                using block_dim                = typename bluestein_block_helper_t::block_dim;
                using ffts_per_block           = typename bluestein_block_helper_t::ffts_per_block;
                using block_fft_implementation = typename bluestein_block_helper_t::block_fft_implementation;

                using suggested_elements_per_thread = typename bluestein_block_helper_t::suggested_elements_per_thread;
                using suggested_ffts_per_block      = typename bluestein_block_helper_t::suggested_ffts_per_block;

                static constexpr bool use_bluestein = true;

                using complex_type = typename make_complex_type<precision_of_t<Description>>::cufftdx_type;
                static constexpr unsigned int workspace_size = 2 * this_fft_blue_size_v * sizeof(complex_type);

            private:
                // Checks
                static_assert(this_fft_blue_size_v >= (2 * this_fft_size_v - 1),
                              "cuFFTDx internal error, selected Bluestein size is too small");
                static_assert(this_fft_blue_size_v <= (4 * this_fft_size_v - 3),
                              "cuFFTDx internal error, selected Bluestein size is too big");
            };

            template<class Description, bool IsThreadExecution>
            class helper
            {
                // To suggest EPT and FPB (BPB) we need to know Size, Type, Direction, Precision + Architecture
                static constexpr bool is_complete = is_complete_description<Description>::value;
                static_assert(is_complete, "FFT description must be complete to calculate queried information");

                // Right now we go to Bluestein only if there's no CT implementation. User can't force Bluestein if
                // there is CT implementation.
#ifdef CUFFTDX_DETAIL_DISABLE_BLUESTEIN
                static constexpr bool is_bluestein_required_v  = false;
                static constexpr bool is_bluestein_supported_v = false;
#else

                static constexpr bool is_bluestein_required_v = is_bluestein_required<size_of<Description>::value,
                                                                                      precision_of_t<Description>,
                                                                                      direction_of<Description>::value,
                                                                                      type_of<Description>::value,
                                                                                      sm_of<Description>::value>::value;
                static constexpr bool is_bluestein_supported_v = is_bluestein_supported<size_of<Description>::value>();

                // Check if we have implementation or bluestein which can do requested size
                static_assert(!is_bluestein_required_v || (is_bluestein_required_v && is_bluestein_supported_v),
                              "This FFT configuration is not supported");
#endif
                using selected_block_helper_t = typename CUFFTDX_STD::conditional<is_bluestein_required_v,
                                                                          helper_block_bluestein<Description>,
                                                                          helper_block_ct<Description>>::type;

            public:
                using elements_per_thread = typename selected_block_helper_t::elements_per_thread;
                using block_dim           = typename selected_block_helper_t::block_dim;
                using ffts_per_block      = typename selected_block_helper_t::ffts_per_block;
                using fft_implementation  = typename selected_block_helper_t::block_fft_implementation;

                using suggested_elements_per_thread = typename selected_block_helper_t::suggested_elements_per_thread;
                using suggested_ffts_per_block      = typename selected_block_helper_t::suggested_ffts_per_block;

                static constexpr bool         use_bluestein      = is_bluestein_required_v;
                static constexpr bool         requires_workspace = is_bluestein_required_v;
                static constexpr unsigned int workspace_size = selected_block_helper_t::workspace_size;
            };

            template<class Description>
            class helper<Description, true>
            {
                // To suggest EPT we need to know Size, Type, Direction, Precision + Architecture
                static constexpr bool is_complete = is_complete_description<Description>::value;
                static_assert(is_complete, "FFT description must be complete to calculate queried information");

                // We don't need SM for thread FFT for description to be complete, so we select dummy SM. Every
                // thread FFT implementation will look the same no matter CUDA architecture.
                static constexpr unsigned int dummy_thread_fft_sm_v = 800;
                using block_fft_record_t = database::detail::block_fft_record<size_of<Description>::value,
                                                                              precision_of_t<Description>,
                                                                              type_of<Description>::value,
                                                                              direction_of<Description>::value,
                                                                              dummy_thread_fft_sm_v>;
                static_assert(block_fft_record_t::defined, "This FFT configuration is not supported");

                using thread_fft_implementation =
                    typename database::detail::search_by_ept<size_of<Description>::value,
                                                             precision_of_t<Description>,
                                                             0 /* trp_option */,
                                                             typename block_fft_record_t::blobs>::type;
                static_assert(!CUFFTDX_STD::is_void<thread_fft_implementation>::value,
                              "This FFT configuration is not supported");

            public:
                using elements_per_thread =
                    ElementsPerThread<get_or_default_t<fft_operator::size, Description, Size<2>>::value>;
                using fft_implementation = thread_fft_implementation;

                static constexpr bool use_bluestein = false;
                static constexpr bool requires_workspace = false;
                static constexpr unsigned int workspace_size = 0;
            };
        } // namespace __get_block_config

        template<class Description>
        class check_and_get_trait<fft_operator::block_dim, Description>
        {
            // FAIL if it's not a block execution
            static constexpr bool is_block_execution = has_operator<fft_operator::block, Description>::value;
            static_assert(is_block_execution, "Must be block execution to get ::block_dim trait");

        public:
            using type                  = typename __get_block_config::helper<Description, false>::block_dim;
            static constexpr dim3 value = type::value;
        };

        template<class Description>
        class check_and_get_trait<fft_operator::elements_per_thread, Description>
        {
            // FAIL if it's not a block execution
            static constexpr bool is_block_execution  = has_operator<fft_operator::block, Description>::value;
            static constexpr bool is_thread_execution = has_operator<fft_operator::thread, Description>::value;
            static_assert(is_block_execution || is_thread_execution,
                          "FFT must be define as either thread of block execution to get ::elements_per_thread trait");

        public:
            using type = typename __get_block_config::helper<Description, is_thread_execution>::elements_per_thread;
            static constexpr unsigned int value = type::value;
        };

        template<class Description>
        class check_and_get_trait<fft_operator::ffts_per_block, Description>
        {
            // FAIL if it's not a block execution
            static constexpr bool is_block_execution = has_operator<fft_operator::block, Description>::value;
            static_assert(is_block_execution, "Must be block execution to get ::ffts_per_block trait");

        public:
            using type = typename __get_block_config::helper<Description, false>::ffts_per_block;
            static constexpr unsigned int value = type::value;

            using suggested_type = typename __get_block_config::helper<Description, false>::suggested_ffts_per_block;
            static constexpr unsigned int suggested = suggested_type::value;
        };

        template<class Description>
        class check_and_get_fft_implementation
        {
            // FAIL if it's not a block execution
            static constexpr bool is_block_execution  = has_operator<fft_operator::block, Description>::value;
            static constexpr bool is_thread_execution = has_operator<fft_operator::thread, Description>::value;
            static_assert(is_block_execution || is_thread_execution,
                          "FFT must be define as either thread of block execution to get ::elements_per_thread trait");

            using block_config_t = __get_block_config::helper<Description, is_thread_execution>;

        public:
            using type = typename block_config_t::fft_implementation;

            static constexpr bool use_bluestein = block_config_t::use_bluestein;
            static constexpr bool requires_workspace = block_config_t::requires_workspace;
            static constexpr unsigned int workspace_size = block_config_t::workspace_size;
        };

        /// Alias template for check_and_get_fft_implementation_t<Description>::type
        template<class Description>
        using check_and_get_fft_implementation_t = typename check_and_get_fft_implementation<Description>::type;
    } // namespace detail
} // namespace cufftdx

#endif // CUFFTDX_TRAITS_DETAIL_CHECK_AND_GET_TRAIT_HPP
