// Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_DETAIL_FFT_EXECUTION_HPP
#define CUFFTDX_DETAIL_FFT_EXECUTION_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

#ifndef __CUDACC_RTC__
#include <cassert>
#endif

#ifndef __CUDACC_RTC__
#include <cuda_runtime_api.h> // cudaError_t
#endif

#include <cuda_fp16.h>

#include "fft_checks.hpp"
#include "fft_description.hpp"
#include "workspace.hpp"

#include "../traits/detail/ldg_type.hpp"

#define STRINGIFY(s)  XSTRINGIFY(s)
#define XSTRINGIFY(s) #s

namespace cufftdx {
    namespace detail {
        template<class T, unsigned int Size>
        struct normalize_helper {
            inline __device__ void operator()(T& value) { value /= Size; }
        };

        template<unsigned int Size>
        struct normalize_helper<complex<__half2>, Size> {
            inline __device__ void operator()(complex<__half2>& value) {
                value.x /= __half2 {Size, Size};
                value.y /= __half2 {Size, Size};
            }
        };

        template<unsigned int Size, class T>
        inline __device__ void normalize(T& value) {
            return normalize_helper<T, Size>()(value);
        };

        template<class T>
        inline __device__ constexpr T get_zero() {
            return 0.;
        }

        template<>
        inline __device__ constexpr __half2 get_zero<__half2>() {
            // This return __half2 with zeros everywhere
            return __half2 {};
        }

        // C2C OR (C2R AND ept == 2)
        template<class FFT, class ComplexType>
        inline __device__ auto preprocess(ComplexType * /* input */) //
            -> typename CUFFTDX_STD::enable_if<(type_of<FFT>::value == fft_type::c2c) ||
                                       ((type_of<FFT>::value == fft_type::c2r) && //
                                        (FFT::elements_per_thread == 2))>::type {
            // NOP, C2C and C2R with ept == 2 don't require any preprocess
        }

        // R2C AND ept > 2
        template<class FFT, class ComplexType>
        inline __device__ auto preprocess(ComplexType* input) //
            -> typename CUFFTDX_STD::enable_if<(type_of<FFT>::value == fft_type::r2c)>::type {
            using scalar_type         = typename ComplexType::value_type;
            static constexpr auto ept = FFT::elements_per_thread;

            // Input has packed real values (this means .y has real values), this
            // unpacks input so every complex value is {real, 0}
            for (unsigned int i = ept; i > 1; i--) {
                (reinterpret_cast<scalar_type*>(input))[2 * i - 1] = get_zero<scalar_type>();
                (reinterpret_cast<scalar_type*>(input))[2 * i - 2] = (reinterpret_cast<scalar_type*>(input))[i - 1];
            }
            // input[0].x is in the right position from the start, just need to set .y to zero
            input[0].y = get_zero<scalar_type>();
        }

        // C2R AND ept > 2
        template<class FFT, class ComplexType>
        inline __device__ auto preprocess(ComplexType* input) //
            -> typename CUFFTDX_STD::enable_if<(type_of<FFT>::value == fft_type::c2r) && (FFT::elements_per_thread > 2)>::type {
            using scalar_type         = typename ComplexType::value_type;
            static constexpr auto ept = FFT::elements_per_thread;

            // If ept is even we need to fill one value less
            static constexpr unsigned number_of_values_to_fill = (ept % 2 == 0) ? (ept / 2 - 1) : (ept / 2);
            for (unsigned int i = 0; i < number_of_values_to_fill; i++) {
                input[ept - i - 1] = input[i + 1];
                // conjugate
                input[ept - i - 1].y = -input[ept - i - 1].y;
            }
        }

        // C2C or R2C
        template<class FFT, class ComplexType>
        inline __device__ auto postprocess(ComplexType* input) //
            -> typename CUFFTDX_STD::enable_if<(type_of<FFT>::value != fft_type::c2r)>::type {
            // NOP, C2R and R2C don't require postprocess
        }

        // C2R
        template<class FFT, class ComplexType>
        inline __device__ auto postprocess(ComplexType* input) //
            -> typename CUFFTDX_STD::enable_if<(type_of<FFT>::value == fft_type::c2r)>::type {
            using scalar_type         = typename ComplexType::value_type;
            static constexpr auto ept = FFT::elements_per_thread;

            // Pack real values
            for (unsigned int i = 1; i < ept; i++) {
                (reinterpret_cast<scalar_type*>(input))[i] = (reinterpret_cast<scalar_type*>(input))[2 * i];
            }
        }

        template<class... Operators>
        class fft_execution: public fft_description<Operators...>, public execution_description_expression
        {
            using base_type      = fft_description<Operators...>;
            using execution_type = fft_execution<Operators...>;

        protected:
            // Precision type
            using typename base_type::this_fft_precision_t;

            /// ---- Constraints

            // We need Block or Thread to be specified exactly once
            static constexpr bool is_thread_execution = has_n_of<1, fft_operator::thread, execution_type>::value;
            static constexpr bool is_block_execution  = has_n_of<1, fft_operator::block, execution_type>::value;
            static_assert((is_thread_execution != is_block_execution), "Can't create FFT with two execution operators");
        };

        template<class... Operators>
        class fft_thread_execution: public fft_execution<Operators...>
        {
            using this_type = fft_thread_execution<Operators...>;
            using base_type = fft_execution<Operators...>;
            using typename base_type::this_fft_precision_t;
            using host_workspace_type = cufftdx::detail::empty_workspace;

            template<class FFT>
            friend typename FFT::host_workspace_type cufftdx::make_workspace(cudaError_t&) noexcept;

        protected:
            // Thread can't have block-only operators
            static constexpr bool has_block_only_operators = has_any_block_operator<base_type>::value;
            static_assert(!has_block_only_operators, "FFT for thread execution can't contain block-only operators");

            // Thread, Size and Precision constrains
            static constexpr bool valid_size_for_thread_fp16 =
                !base_type::has_size || // Size<> was not defined
                !CUFFTDX_STD::is_same<__half, this_fft_precision_t>::value ||
                ((base_type::this_fft_size_v <= 32) && (base_type::this_fft_size_v >= 2));
            static constexpr bool valid_size_for_thread_fp32 =
                !base_type::has_size || // Size<> was not defined
                !CUFFTDX_STD::is_same<float, this_fft_precision_t>::value ||
                ((base_type::this_fft_size_v <= 32) && (base_type::this_fft_size_v >= 2));
            static constexpr bool valid_size_for_thread_fp64 =
                !base_type::has_size || // Size<> was not defined
                !CUFFTDX_STD::is_same<double, this_fft_precision_t>::value ||
                ((base_type::this_fft_size_v <= 16) && (base_type::this_fft_size_v >= 2));
            static_assert(valid_size_for_thread_fp16,
                          "Thread execution in fp16 precision supports sizes in range [2; 32]");
            static_assert(valid_size_for_thread_fp32,
                          "Thread execution in fp32 precision supports sizes in range [2; 32]");
            static_assert(valid_size_for_thread_fp64,
                          "Thread execution in fp64 precision supports sizes in range [2; 16]");

        public:
            using value_type  = typename make_complex_type<this_fft_precision_t>::cufftdx_type;
            using input_type  = value_type;
            using output_type = value_type;
            using workspace_type = typename host_workspace_type::device_handle;
            static_assert(CUFFTDX_STD::is_same<host_workspace_type, cufftdx::detail::empty_workspace>::value,
                          "Internal cuFFTDx error, thread FFT should never require non-empty workspace");

            inline __device__ void execute(value_type* input) {
                static_assert(base_type::is_complete, "Can't execute, FFT description is not complete");

                using fft_implementation_t        = check_and_get_fft_implementation_t<this_type>;
                static constexpr auto function_id = fft_implementation_t::function_id;

                preprocess<this_type>(input);
                using scalar_type = typename value_type::value_type;
                database::detail::cufftdx_private_function_wrapper<function_id, scalar_type, 1>(input, nullptr);
                postprocess<this_type>(input);
            }

            // T - can be any type if it's alignment and size are the same as those of ::value_type
            template<class T /* TODO = typename make_vector_type<make_scalar_type<value_type>, 2>::type */>
            inline __device__ auto execute(T* input) //
                -> typename CUFFTDX_STD::enable_if<!CUFFTDX_STD::is_void<T>::value && (sizeof(T) == sizeof(value_type)) &&
                                           (alignof(T) == alignof(value_type))>::type {
                return execute(reinterpret_cast<value_type*>(input));
            }

            template<class T>
            inline __device__ auto execute(T* input) //
                -> typename CUFFTDX_STD::enable_if<CUFFTDX_STD::is_void<T>::value || (sizeof(T) != sizeof(value_type)) ||
                                           (alignof(T) != alignof(value_type))>::type {
                static constexpr bool condition =
                    CUFFTDX_STD::is_void<T>::value || (sizeof(T) != sizeof(value_type)) || (alignof(T) != alignof(value_type));
                static_assert(condition, "Incorrect value type is used, try using ::value_type");
            }

            template<class T>
            inline __device__ auto execute(T* input, workspace_type & /* workspace */) //
                -> typename CUFFTDX_STD::enable_if<!CUFFTDX_STD::is_void<T>::value && (sizeof(T) == sizeof(value_type)) &&
                                           (alignof(T) == alignof(value_type))>::type {
                return execute(reinterpret_cast<value_type*>(input));
            }

            template<class T>
            inline __device__ auto execute(T* /* input */, workspace_type & /* workspace */) //
                -> typename CUFFTDX_STD::enable_if<CUFFTDX_STD::is_void<T>::value || (sizeof(T) != sizeof(value_type)) ||
                                           (alignof(T) != alignof(value_type))>::type {
                static constexpr bool condition =
                    CUFFTDX_STD::is_void<T>::value || (sizeof(T) != sizeof(value_type)) || (alignof(T) != alignof(value_type));
                static_assert(condition, "Incorrect value type is used, try using ::value_type");
            }

            static constexpr unsigned int elements_per_thread = check_and_get_trait<fft_operator::elements_per_thread, base_type>::value;
            static constexpr unsigned int stride              = 1;
            static constexpr unsigned int storage_size        = elements_per_thread;

            static constexpr unsigned int implicit_type_batching =
                CUFFTDX_STD::is_same<this_fft_precision_t, __half>::value ? 2 : 1;
        };

        template<class... Operators>
        constexpr unsigned int fft_thread_execution<Operators...>::elements_per_thread;
        template<class... Operators>
        constexpr unsigned int fft_thread_execution<Operators...>::stride;
        template<class... Operators>
        constexpr unsigned int fft_thread_execution<Operators...>::storage_size;
        template<class... Operators>
        constexpr unsigned int fft_thread_execution<Operators...>::implicit_type_batching;

        // Registers API

        // C2C
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess(ComplexType* /* input */, ComplexType * /* smem */) //
            -> typename CUFFTDX_STD::enable_if<(type_of<FFT>::value == fft_type::c2c)>::type {
            // NOP, C2C and C2R with ept == 2 don't require any preprocess
        }

        // R2C AND ept > 2
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess(ComplexType* input, ComplexType * /* smem */) //
            -> typename CUFFTDX_STD::enable_if<(type_of<FFT>::value == fft_type::r2c)>::type {
            // Same implementation as thread_preprocess
            preprocess<FFT>(input);
        }

        // C2R, EPT == SIZE
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess(ComplexType* input, ComplexType * /* smem */) //
            -> typename CUFFTDX_STD::enable_if<(type_of<FFT>::value == fft_type::c2r) &&
                                       (FFT::elements_per_thread == size_of<FFT>::value)>::type {
            // Same implementation as thread_preprocess
            preprocess<FFT>(input);
        }

        // C2R, EPT < SIZE, CT
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess(ComplexType* input, ComplexType* smem) //
            -> typename CUFFTDX_STD::enable_if<!Bluestein && (type_of<FFT>::value == fft_type::c2r) &&
                                       (FFT::elements_per_thread < size_of<FFT>::value)>::type {
            using scalar_type                      = typename ComplexType::value_type;
            static constexpr auto ept              = FFT::elements_per_thread;
            static constexpr auto fft_size         = size_of<FFT>::value;
            static constexpr bool fft_size_is_even = (fft_size % 2) == 0;

            // Move to the part of shared memory for that FFT batch
            ComplexType* smem_fft_batch = smem + (threadIdx.y * (fft_size / 2));

            for (unsigned int i = 0; i < (ept / 2); i++) {
                if (!(threadIdx.x == 0 && i == 0)) {
                    smem_fft_batch[threadIdx.x + (i * (fft_size / ept)) - 1] = input[i];
                }
            }
            if (!fft_size_is_even) {
                constexpr unsigned int i     = ept / 2;
                unsigned int           index = threadIdx.x + (i * (fft_size / ept)) - 1;
                if (index < (fft_size / 2)) {
                    smem_fft_batch[index] = input[i];
                }
            }
            __syncthreads();

            const unsigned int reversed_thread_id = (fft_size / ept) - threadIdx.x;
            for (unsigned int i = 0; i < (ept / 2); i++) {
                if (i < ((ept / 2) - ((threadIdx.x == 0) && fft_size_is_even))) {
                    input[ept - 1 - i] = smem_fft_batch[reversed_thread_id + (i * (fft_size / ept)) - 1];
                    // conjugate
                    input[ept - 1 - i].y = -input[ept - 1 - i].y;
                }
            }
            if (!fft_size_is_even) {
                constexpr unsigned int i     = ept / 2;
                unsigned int           index = reversed_thread_id + (i * (fft_size / ept)) - 1;
                if (index < (fft_size / 2)) {
                    input[i] = smem_fft_batch[index];
                    // conjugate
                    input[i].y = -input[i].y;
                }
            }
        }

        // C2R, EPT < SIZE, Bluestein
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess(ComplexType* input, ComplexType* smem) //
            -> typename CUFFTDX_STD::enable_if<Bluestein && (type_of<FFT>::value == fft_type::c2r) &&
                                       (FFT::elements_per_thread < get_bluestein_size(size_of<FFT>::value))>::type {
            using scalar_type                           = typename ComplexType::value_type;
            static constexpr auto         ept           = FFT::elements_per_thread;
            static constexpr auto         fft_size      = size_of<FFT>::value;
            static constexpr auto         fft_blue_size = get_bluestein_size(fft_size);
            static constexpr unsigned int stride        = fft_blue_size / ept;

            // Move to the part of shared memory for that FFT batch
            ComplexType* smem_fft_batch = smem + (threadIdx.y * (fft_blue_size / 2));

            // max_meaningful_ept limits number of loops
            static constexpr unsigned int max_meaningful_ept = ((fft_size / 2 + 1) + (stride - 1)) / stride;
            for (unsigned i = 0; i < max_meaningful_ept /*ept/2*/; i++) {
                unsigned index = (i * stride) + threadIdx.x;
                if (index < (fft_size / 2 + 1)) {
                    if (!(threadIdx.x == 0 && i == 0)) {
                        smem_fft_batch[index - 1] = input[i];
                    }
                }
            }
            __syncthreads();

            // max_meaningful_ept_2 limits number of loops
            static constexpr unsigned int max_meaningful_ept_2 =
                ept > (2 * max_meaningful_ept) ? ept : (2 * max_meaningful_ept);
            for (unsigned i = (max_meaningful_ept - 1); i < max_meaningful_ept_2; i++) {
                unsigned int index = (i * stride) + threadIdx.x;
                if ((index >= (fft_size / 2 + 1)) && (index < fft_size)) {
                    input[i] = smem_fft_batch[(fft_size - index) - 1];
                    // conjugate
                    input[i].y = -input[i].y;
                }
            }
        }

        // Shared memory API

        // C2C
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess_shared_api(ComplexType* /* input */, ComplexType * /* smem */) //
            -> typename CUFFTDX_STD::enable_if<(type_of<FFT>::value == fft_type::c2c)>::type {
            // NOP, C2C and C2R with ept == 2 don't require any preprocess
        }

        // R2C AND ept > 2
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess_shared_api(ComplexType* input, ComplexType * /* smem */) //
            -> typename CUFFTDX_STD::enable_if<(type_of<FFT>::value == fft_type::r2c)>::type {
            // Same implementation as thread_preprocess
            preprocess<FFT>(input);
        }

        // C2R, EPT == SIZE
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess_shared_api(ComplexType* input, ComplexType * /* smem */) //
            -> typename CUFFTDX_STD::enable_if<(type_of<FFT>::value == fft_type::c2r) &&
                                       (FFT::elements_per_thread == size_of<FFT>::value)>::type {
            // Same implementation as thread_preprocess
            preprocess<FFT>(input);
        }

        // C2R, EPT < SIZE, CT
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess_shared_api(ComplexType* input, ComplexType* smem) //
            -> typename CUFFTDX_STD::enable_if<!Bluestein && (type_of<FFT>::value == fft_type::c2r) &&
                                       (FFT::elements_per_thread < size_of<FFT>::value)>::type {
            using scalar_type                      = typename ComplexType::value_type;
            static constexpr auto ept              = FFT::elements_per_thread;
            static constexpr auto fft_size         = size_of<FFT>::value;
            static constexpr bool fft_size_is_even = (fft_size % 2) == 0;

            // Move to the part of shared memory for that FFT batch
            ComplexType* smem_fft_batch = smem + (threadIdx.y * (fft_size / 2 + 1));

            const unsigned int reversed_thread_id = (fft_size / ept) - threadIdx.x;
            for (unsigned int i = 0; i < (ept / 2); i++) {
                if (i < ((ept / 2) - ((threadIdx.x == 0) && fft_size_is_even))) {
                    input[ept - 1 - i] = smem_fft_batch[reversed_thread_id + (i * (fft_size / ept))];
                    // conjugate
                    input[ept - 1 - i].y = -input[ept - 1 - i].y;
                }
            }
            if (!fft_size_is_even) {
                constexpr unsigned int i     = ept / 2;
                unsigned int           index = reversed_thread_id + (i * (fft_size / ept));
                if (index < (fft_size / 2) + 1) {
                    input[i] = smem_fft_batch[index];
                    // conjugate
                    input[i].y = -input[i].y;
                }
            }
        }

        // C2R, EPT < SIZE, Bluestein
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess_shared_api(ComplexType* input, ComplexType* smem) //
            -> typename CUFFTDX_STD::enable_if<Bluestein && (type_of<FFT>::value == fft_type::c2r) &&
                                       (FFT::elements_per_thread < get_bluestein_size(size_of<FFT>::value))>::type {
            using scalar_type                           = typename ComplexType::value_type;
            static constexpr auto         ept           = FFT::elements_per_thread;
            static constexpr auto         fft_size      = size_of<FFT>::value;
            static constexpr auto         fft_blue_size = get_bluestein_size(fft_size);
            static constexpr unsigned int stride        = fft_blue_size / ept;

            // Move to the part of shared memory for that FFT batch
            ComplexType* smem_fft_batch = smem + (threadIdx.y * (fft_size / 2 + 1));

            static constexpr unsigned int first_missing_index = ((fft_size / 2 + 1) + (stride - 1)) / stride;
            static constexpr unsigned int last_missing_index =
                ept > (2 * first_missing_index) ? ept : (2 * first_missing_index);
            for (unsigned i = (first_missing_index - 1); i < last_missing_index; i++) {
                unsigned int index = (i * stride) + threadIdx.x;
                if ((index >= (fft_size / 2 + 1)) && (index < fft_size)) {
                    input[i] = smem_fft_batch[(fft_size - index)];
                    // conjugate
                    input[i].y = -input[i].y;
                }
            }
        }

#ifdef __CUDACC_RTC__
        template<class... Operators>
        class fft_block_execution_partial: public fft_execution<Operators...>
        {
            using base_type = fft_execution<Operators...>;
            using typename base_type::this_fft_precision_t;

        public:
            using value_type = typename make_complex_type<this_fft_precision_t>::cufftdx_type;
            using input_type  = value_type;
            using output_type = value_type;
        };
#endif

        template<class... Operators>
        class fft_block_execution: public fft_execution<Operators...>
        {
            using this_type = fft_block_execution<Operators...>;
            using base_type = fft_execution<Operators...>;
            using typename base_type::this_fft_precision_t;

        public:
            using value_type = typename make_complex_type<this_fft_precision_t>::cufftdx_type;

        private:
            template<class FFT>
            friend typename FFT::host_workspace_type cufftdx::make_workspace(cudaError_t&) noexcept;
            using host_workspace_type = typename workspace_selector<base_type::this_fft_size_v,
                                                                    this_fft_precision_t,
                                                                    value_type,
                                                                    base_type::this_fft_direction_v,
                                                                    base_type::this_fft_type_v,
                                                                    base_type::this_fft_sm_v,
                                                                    base_type::is_complete>::type;

            // Return false if fft's precision matches 'Precision', sm matches 'SM'
            // and test can not be executed
            template<class Precision, unsigned SM>
            static constexpr bool is_valid_size_for_block() {
                using is_size_supported = is_supported<Precision, base_type::this_fft_size_v, base_type::this_fft_sm_v>;
                return !base_type::has_size || // Size<> was not defined
                       !(base_type::this_fft_sm_v == SM && base_type::has_sm) ||
                       !(CUFFTDX_STD::is_same<this_fft_precision_t, Precision>::value) ||
                       ((CUFFTDX_STD::is_same<this_fft_precision_t, __half>::value) && is_size_supported::fp16_block_value) ||
                       ((CUFFTDX_STD::is_same<this_fft_precision_t, float>::value) && is_size_supported::fp32_block_value) ||
                       ((CUFFTDX_STD::is_same<this_fft_precision_t, double>::value) && is_size_supported::fp64_block_value);
            }

            // Check requirements for Bluestein size
            // If we need Bluestein, we need to generate chirp using FP64 FFT of size next-power_of_2(2*N - 1)
            template<unsigned SM>
            static constexpr bool is_valid_size_for_bluestein() {
                return !CUFFTDX_STD::is_same<host_workspace_type, detail::bluestein_workspace<base_type::this_fft_size_v, value_type, base_type::this_fft_sm_v>>::value ||
                       !base_type::has_size || // Size<> was not defined
                       !(base_type::this_fft_sm_v == SM && base_type::has_sm) ||
                       is_supported<double, base_type::this_fft_size_v, base_type::this_fft_sm_v>::blue_block_value;
            }

        protected:

            // Block, Size and Precision constrains

            // SM70
            static constexpr bool valid_size_for_block_fp16_sm70 = is_valid_size_for_block<__half, 700>();
            static constexpr bool valid_size_for_block_fp32_sm70 = is_valid_size_for_block<float, 700>();
            static constexpr bool valid_size_for_block_fp64_sm70 = is_valid_size_for_block<double, 700>();
            static_assert(valid_size_for_block_fp16_sm70,
                          "Block execution in fp16 precision on SM70 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM700_FP16_MAX) "]");
            static_assert(valid_size_for_block_fp32_sm70,
                          "Block execution in fp32 precision on SM70 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM700_FP32_MAX) "]");
            static_assert(valid_size_for_block_fp64_sm70,
                          "Block execution in fp64 precision on SM70 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM700_FP64_MAX) "]");

            static constexpr bool valid_size_for_bluestein_sm70 = is_valid_size_for_bluestein<700>();
            static_assert(valid_size_for_bluestein_sm70,
                          "Block execution for this size is not supported");

            // SM72
            static constexpr bool valid_size_for_block_fp16_sm72 = is_valid_size_for_block<__half, 720>();
            static constexpr bool valid_size_for_block_fp32_sm72 = is_valid_size_for_block<float, 720>();
            static constexpr bool valid_size_for_block_fp64_sm72 = is_valid_size_for_block<double, 720>();
            static_assert(valid_size_for_block_fp16_sm72,
                          "Block execution in fp16 precision on SM72 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM720_FP16_MAX) "]");
            static_assert(valid_size_for_block_fp32_sm72,
                          "Block execution in fp32 precision on SM72 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM720_FP32_MAX) "]");
            static_assert(valid_size_for_block_fp64_sm72,
                          "Block execution in fp64 precision on SM72 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM720_FP64_MAX) "]");

            static constexpr bool valid_size_for_bluestein_sm72 = is_valid_size_for_bluestein<720>();
            static_assert(valid_size_for_bluestein_sm72,
                          "Block execution for this size is not supported");

            // SM75
            static constexpr bool valid_size_for_block_fp16_sm75 = is_valid_size_for_block<__half, 750>();
            static constexpr bool valid_size_for_block_fp32_sm75 = is_valid_size_for_block<float, 750>();
            static constexpr bool valid_size_for_block_fp64_sm75 = is_valid_size_for_block<double, 750>();
            static_assert(valid_size_for_block_fp16_sm75,
                          "Block execution in fp16 precision on SM75 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM750_FP16_MAX) "]");
            static_assert(valid_size_for_block_fp32_sm75,
                          "Block execution in fp32 precision on SM75 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM750_FP32_MAX) "]");
            static_assert(valid_size_for_block_fp64_sm75,
                          "Block execution in fp64 precision on SM75 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM750_FP64_MAX) "]");

            static constexpr bool valid_size_for_bluestein_sm75 = is_valid_size_for_bluestein<750>();
            static_assert(valid_size_for_bluestein_sm75,
                          "Block execution for this size is not supported");

            // SM80
            static constexpr bool valid_size_for_block_fp16_sm80 = is_valid_size_for_block<__half, 800>();
            static constexpr bool valid_size_for_block_fp32_sm80 = is_valid_size_for_block<float, 800>();
            static constexpr bool valid_size_for_block_fp64_sm80 = is_valid_size_for_block<double, 800>();
            static_assert(valid_size_for_block_fp16_sm80,
                          "Block execution in fp16 precision on SM80 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM800_FP16_MAX) "]");
            static_assert(valid_size_for_block_fp32_sm80,
                          "Block execution in fp32 precision on SM80 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM800_FP32_MAX) "]");
            static_assert(valid_size_for_block_fp64_sm80,
                          "Block execution in fp64 precision on SM80 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM800_FP64_MAX) "]");

            static constexpr bool valid_size_for_bluestein_sm80 = is_valid_size_for_bluestein<800>();
            static_assert(valid_size_for_bluestein_sm80,
                          "Block execution for this size is not supported");

            // SM86
            static constexpr bool valid_size_for_block_fp16_sm86 = is_valid_size_for_block<__half, 860>();
            static constexpr bool valid_size_for_block_fp32_sm86 = is_valid_size_for_block<float, 860>();
            static constexpr bool valid_size_for_block_fp64_sm86 = is_valid_size_for_block<double, 860>();
            static_assert(valid_size_for_block_fp16_sm86,
                          "Block execution in fp16 precision on SM86 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM860_FP16_MAX) "]");
            static_assert(valid_size_for_block_fp32_sm86,
                          "Block execution in fp32 precision on SM86 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM860_FP32_MAX) "]");
            static_assert(valid_size_for_block_fp64_sm86,
                          "Block execution in fp64 precision on SM86 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM860_FP64_MAX) "]");

            static constexpr bool valid_size_for_bluestein_sm86 = is_valid_size_for_bluestein<860>();
            static_assert(valid_size_for_bluestein_sm86,
                          "Block execution for this size is not supported");

            // MAX (No SM must be defined)
            static constexpr bool valid_size_for_block_fp16_max =
                !base_type::has_size || // Size<> was not defined
                !(CUFFTDX_STD::is_same<this_fft_precision_t, __half>::value) ||
                is_supported<__half, base_type::this_fft_size_v, unsigned(-1)>::fp16_block_value;
            static constexpr bool valid_size_for_block_fp32_max =
                !base_type::has_size || // Size<> was not defined
                !(CUFFTDX_STD::is_same<this_fft_precision_t, float>::value) ||
                is_supported<float, base_type::this_fft_size_v, unsigned(-1)>::fp32_block_value;
            static constexpr bool valid_size_for_block_fp64_max =
                !base_type::has_size || // Size<> was not defined
                !(CUFFTDX_STD::is_same<this_fft_precision_t, double>::value) ||
                is_supported<double, base_type::this_fft_size_v, unsigned(-1)>::fp64_block_value;
            static_assert(valid_size_for_block_fp16_max,
                          "Block execution in fp16 precision supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM800_FP16_MAX) "]");
            static_assert(valid_size_for_block_fp32_max,
                          "Block execution in fp32 precision supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM800_FP32_MAX) "]");
            static_assert(valid_size_for_block_fp64_max,
                          "Block execution in fp64 precision supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM800_FP64_MAX) "]");

            // MAX (No SM must be defined)
            // Check requirements for Bluestein size
            // If we need Bluestein, we need to generate chirp using FP64 FFT of size next-power_of_2(2*N - 1)
            static constexpr bool valid_size_for_bluestein_max =
                !CUFFTDX_STD::is_same<host_workspace_type, detail::bluestein_workspace<base_type::this_fft_size_v, value_type, base_type::this_fft_sm_v>>::value ||
                !base_type::has_size || // Size<> was not defined
                is_supported<double, base_type::this_fft_size_v, unsigned(-1)>::blue_block_value;
            static_assert(valid_size_for_bluestein_max,
                          "Block execution for this size is not supported");

        public:
            using input_type  = value_type;
            using output_type = value_type;
            using workspace_type = typename host_workspace_type::device_handle;

            template<class T>
            inline __device__ void execute(T* shared_memory_input) {
#if !defined(NDEBUG) && !defined(CUFFTDX_DISABLE_RUNTIME_ASSERTS) && !defined(__CUDACC_RTC__)
                const bool block_dimension_x_is_correct = (blockDim.x == block_dim.x);
                assert(block_dimension_x_is_correct);
                const bool block_dimension_y_is_correct = (blockDim.y == block_dim.y);
                assert(block_dimension_y_is_correct);
#endif
                static_assert(base_type::is_complete, "Can't execute, FFT description is not complete");
                static_assert(!requires_workspace, "This FFT configuration requires workspace");

                static constexpr bool use_bluestein = check_and_get_fft_implementation<this_type>::use_bluestein;

                value_type thread_data[storage_size];
                shared_to_registers(shared_memory_input, thread_data);

                block_preprocess_shared_api<this_type, use_bluestein>(thread_data, reinterpret_cast<value_type*>(shared_memory_input));
                workspace_type dummy_workspace;
                internal_execute<use_bluestein>(thread_data, shared_memory_input, dummy_workspace);
                postprocess<this_type>(thread_data);

                registers_to_shared(thread_data, shared_memory_input);
            }

            template<class T>
            inline __device__ void execute(T* shared_memory_input, workspace_type& workspace) {
#if !defined(NDEBUG) && !defined(CUFFTDX_DISABLE_RUNTIME_ASSERTS) && !defined(__CUDACC_RTC__)
                const bool block_dimension_x_is_correct = (blockDim.x == block_dim.x);
                assert(block_dimension_x_is_correct);
                const bool block_dimension_y_is_correct = (blockDim.y == block_dim.y);
                assert(block_dimension_y_is_correct);
#endif
                static_assert(base_type::is_complete, "Can't execute, FFT description is not complete");

                value_type thread_data[storage_size];
                shared_to_registers(shared_memory_input, thread_data);

                static constexpr bool use_bluestein = check_and_get_fft_implementation<this_type>::use_bluestein;
                block_preprocess_shared_api<this_type, use_bluestein>(thread_data, reinterpret_cast<value_type*>(shared_memory_input));
                internal_execute<use_bluestein>(thread_data, shared_memory_input, workspace);
                postprocess<this_type>(thread_data);

                registers_to_shared(thread_data, shared_memory_input);
            }

            inline __device__ void execute(value_type* input, void* shared_memory) {
#if !defined(NDEBUG) && !defined(CUFFTDX_DISABLE_RUNTIME_ASSERTS) && !defined(__CUDACC_RTC__)
                const bool block_dimension_x_is_correct = (blockDim.x == block_dim.x);
                assert(block_dimension_x_is_correct);
                const bool block_dimension_y_is_correct = (blockDim.y == block_dim.y);
                assert(block_dimension_y_is_correct);
#endif

                static_assert(base_type::is_complete, "Can't execute, FFT description is not complete");
                static_assert(!requires_workspace, "This FFT configuration requires workspace");

                static constexpr bool use_bluestein = check_and_get_fft_implementation<this_type>::use_bluestein;
                block_preprocess<this_type, use_bluestein>(input, reinterpret_cast<value_type*>(shared_memory));
                workspace_type dummy_workspace;
                internal_execute<use_bluestein>(input, shared_memory, dummy_workspace);
                postprocess<this_type>(input);
            }

            // T - can be any type if its alignment and size are the same as those of ::value_type
            template<class T /* TODO = typename make_vector_type<make_scalar_type<value_type>, 2>::type */>
            inline __device__ auto execute(T* input, void* shared_memory) //
                -> typename CUFFTDX_STD::enable_if<!CUFFTDX_STD::is_void<T>::value && (sizeof(T) == sizeof(value_type)) &&
                                           (alignof(T) == alignof(value_type))>::type {
                return execute(reinterpret_cast<value_type*>(input), shared_memory);
            }

            template<class T>
            inline __device__ auto execute(T* /* input */, void * /* shared_memory */) //
                -> typename CUFFTDX_STD::enable_if<CUFFTDX_STD::is_void<T>::value || (sizeof(T) != sizeof(value_type)) ||
                                           (alignof(T) != alignof(value_type))>::type {
                static constexpr bool condition =
                    CUFFTDX_STD::is_void<T>::value || (sizeof(T) != sizeof(value_type)) || (alignof(T) != alignof(value_type));
                static_assert(condition, "Incorrect value type is used, try using ::value_type");
            }

            inline __device__ void execute(value_type* input, void* shared_memory, workspace_type& workspace) {
#if !defined(NDEBUG) && !defined(CUFFTDX_DISABLE_RUNTIME_ASSERTS) && !defined(__CUDACC_RTC__)
                const bool block_dimension_x_is_correct = (blockDim.x == block_dim.x);
                assert(block_dimension_x_is_correct);
                const bool block_dimension_y_is_correct = (blockDim.y == block_dim.y);
                assert(block_dimension_y_is_correct);
#endif
                static_assert(base_type::is_complete, "Can't execute, FFT description is not complete");

                static constexpr bool use_bluestein = check_and_get_fft_implementation<this_type>::use_bluestein;
                block_preprocess<this_type, use_bluestein>(input, reinterpret_cast<value_type*>(shared_memory));
                internal_execute<use_bluestein>(input, shared_memory, workspace);
                postprocess<this_type>(input);
            }

            // T - can be any type if its alignment and size are the same as those of ::value_type
            template<class T>
            inline __device__ auto execute(T* input, void* shared_memory, workspace_type& workspace) //
                -> typename CUFFTDX_STD::enable_if<!CUFFTDX_STD::is_void<T>::value && (sizeof(T) == sizeof(value_type)) &&
                                           (alignof(T) == alignof(value_type))>::type {
                return execute(reinterpret_cast<value_type*>(input), shared_memory, workspace);
            }

            template<class T>
            inline __device__ auto execute(T* /* input */, void* /* shared_memory */, workspace_type &
                                           /* workspace */) //
                -> typename CUFFTDX_STD::enable_if<CUFFTDX_STD::is_void<T>::value || (sizeof(T) != sizeof(value_type)) ||
                                           (alignof(T) != alignof(value_type))>::type {
                static constexpr bool condition =
                    CUFFTDX_STD::is_void<T>::value || (sizeof(T) != sizeof(value_type)) || (alignof(T) != alignof(value_type));
                static_assert(condition, "Incorrect value type is used, try using ::value_type");
            }

        private:

            template<unsigned int N, class T>
            inline __device__ void shared_to_registers_impl(T* shared_memory, T* thread_data) {
                const unsigned int batch_offset = threadIdx.y * N;
                unsigned int index = threadIdx.x;
                for(unsigned int i = 0; i < elements_per_thread; i++) {
                    if (index < N) {
                        thread_data[i] = shared_memory[batch_offset + index];
                    }
                    index += stride;
                }
            }

            template<unsigned int N, class T>
            inline __device__ void registers_to_shared_impl(T* thread_data, T* shared_memory) {
                const unsigned int batch_offset = threadIdx.y * N;
                unsigned int index = threadIdx.x;
                for(unsigned int i = 0; i < elements_per_thread; i++) {
                    if (index < N) {
                        shared_memory[batch_offset + index] = thread_data[i];
                    }
                    index += stride;
                }
            }

            template<class V>
            inline __device__ void shared_to_registers(void* shared_memory, V* thread_data) {
                if (base_type::this_fft_type_v == fft_type::c2c) {
                    shared_to_registers_impl<base_type::this_fft_size_v>(
                        reinterpret_cast<value_type*>(shared_memory), reinterpret_cast<value_type*>(thread_data));
                } else if (base_type::this_fft_type_v == fft_type::c2r) {
                    shared_to_registers_impl<base_type::this_fft_size_v / 2 + 1>(
                        reinterpret_cast<value_type*>(shared_memory), reinterpret_cast<value_type*>(thread_data));
                } else if (base_type::this_fft_type_v == fft_type::r2c) {
                    shared_to_registers_impl<base_type::this_fft_size_v>(
                        reinterpret_cast<typename value_type::value_type*>(shared_memory),
                        reinterpret_cast<typename value_type::value_type*>(thread_data));
                }
                __syncthreads();
            }

            template<class V>
            inline __device__ void registers_to_shared(void* shared_memory, V* thread_data) {
                __syncthreads();
                if (base_type::this_fft_type_v == fft_type::c2c) {
                    registers_to_shared_impl<base_type::this_fft_size_v>(
                        reinterpret_cast<value_type*>(shared_memory), reinterpret_cast<value_type*>(thread_data));
                } else if (base_type::this_fft_type_v == fft_type::c2r) {
                    registers_to_shared_impl<base_type::this_fft_size_v>(
                        reinterpret_cast<typename value_type::value_type*>(shared_memory),
                        reinterpret_cast<typename value_type::value_type*>(thread_data));
                } else if (base_type::this_fft_type_v == fft_type::r2c) {
                    registers_to_shared_impl<base_type::this_fft_size_v / 2 + 1>(
                        reinterpret_cast<value_type*>(shared_memory), reinterpret_cast<value_type*>(thread_data));
                }
            }

            // Cooley-Tukey execution
            template<bool UseBluestein>
            inline __device__ auto internal_execute(value_type* input,
                                                    void*       shared_memory,
                                                    workspace_type& /* workspace */,
                                                    const unsigned int /* fft_id */ = threadIdx.x) //
                -> typename CUFFTDX_STD::enable_if<!UseBluestein>::type {
                using fft_implementation_t        = check_and_get_fft_implementation_t<this_type>;
                static constexpr auto function_id = fft_implementation_t::function_id;
                using scalar_type                 = typename value_type::value_type;
                database::detail::cufftdx_private_function_wrapper<function_id, scalar_type, 1 /* dynamic */>(input,
                                                                                                              shared_memory);
            }

            // Bluestein execution
            // Assumptions:
            // * fft_id is threadIdx.x -> user must use our block dimension
            template<bool UseBluestein>
            inline __device__ auto internal_execute(value_type*        input,
                                                    void*              shared_memory,
                                                    workspace_type&    workspace,
                                                    const unsigned int fft_id = threadIdx.x) //
                -> typename CUFFTDX_STD::enable_if<UseBluestein>::type {
#if !defined(NDEBUG)
                const bool workspace_valid = workspace.valid();
                assert(workspace_valid && "Workspace is invalid, check if workspace was created successfully before passing it to kernel");
#endif

                using scalar_type = typename value_type::value_type;
                using ldg_type    = typename ldg_type<value_type>::type;

                using fft_implementation_t          = check_and_get_fft_implementation_t<this_type>;
                static constexpr auto function_id   = fft_implementation_t::function_id;
                static constexpr auto fft_blue_size = get_bluestein_size(base_type::this_fft_size_v);

                unsigned int                  index  = fft_id;
                static constexpr unsigned int stride = fft_blue_size / elements_per_thread;

                // Only first fft_size values are meaningful, others should be zero.
                static constexpr unsigned int max_meaningful_ept = (base_type::this_fft_size_v + (stride - 1)) / stride;
                // In this case user is expected to zero-padded input.
                // for (unsigned int i = 0; i < max_meaningful_ept; ++i) {
                //     auto v = __ldg((ldg_type*)workspace.w_time + index);
                //     input[i] *= *(reinterpret_cast<value_type*>(&v));
                //     index += stride;
                // }
                // This zeroes the padding.
                for (unsigned int i = 0; i < elements_per_thread; ++i) {
                    // Make swap real<->imag for inverse FFT
                    if (base_type::this_fft_direction_v == fft_direction::inverse) {
                        const auto tmp = input[i].x;
                        input[i].x     = input[i].y;
                        input[i].y     = tmp;
                    }

                    if ((i * stride + fft_id) < base_type::this_fft_size_v) {
                        auto v = __ldg((ldg_type*)workspace.w_time + index);
                        // For half precision we're loading float2 in ldg, so we need
                        // to reinterpret in to complex<__half2> in order to have correct
                        // multiplication performed.
                        input[i] *= *(reinterpret_cast<value_type*>(&v));
                    } else {
                        input[i] = value_type(0., 0.);
                    }
                    index += stride;
                }

                database::detail::cufftdx_private_function_wrapper<function_id, scalar_type, 1 /* dynamic */>(input,
                                                                                                              shared_memory);
                index = fft_id;
                for (unsigned int i = 0; i < elements_per_thread; ++i) {
                    auto v = __ldg((ldg_type*)workspace.w_freq + index);
                    input[i] *= *(reinterpret_cast<value_type*>(&v));
                    input[i].y = -input[i].y; // conjugate
                    index += stride;
                }

                database::detail::cufftdx_private_function_wrapper<function_id, scalar_type, 1 /* dynamic */>(input,
                                                                                                              shared_memory);

                // We can limit the last loop to just max_meaningful_ept, other values are not needed.
                index = fft_id;
                for (unsigned int i = 0; i < max_meaningful_ept; ++i) {
                    input[i].y = -input[i].y; // conjugate
                    // normalize; input[i] /= fft_blue_size; // divide by xsize, for ifft
                    normalize<fft_blue_size>( input[i]);
                    auto v = __ldg((ldg_type*)workspace.w_time + index);
                    input[i] *= *(reinterpret_cast<value_type*>(&v));
                    index += stride;

                    // Make swap real<->imag for inverse FFT
                    if (base_type::this_fft_direction_v == fft_direction::inverse) {
                        const auto tmp = input[i].x;
                        input[i].x     = input[i].y;
                        input[i].y     = tmp;
                    }
                }
            }

            inline static constexpr unsigned int get_shared_memory_size() {
                static_assert(base_type::is_complete, "Can't calculate shared memory, FFT description is not complete");
                using fft_implementation_t = check_and_get_fft_implementation_t<this_type>;
                return fft_implementation_t::shared_memory_size * ffts_per_block;
            }

            inline static constexpr unsigned int get_storage_size() {
                static_assert(base_type::is_complete, "Can't calculate storage_size, FFT description is not complete");
                using fft_implementation_t = check_and_get_fft_implementation_t<this_type>;
                return fft_implementation_t::storage_size;
            }

        public:
            static constexpr dim3         block_dim = check_and_get_trait<fft_operator::block_dim, base_type>::value;
            static constexpr unsigned int ffts_per_block =
                check_and_get_trait<fft_operator::ffts_per_block, base_type>::value;
            static constexpr unsigned int elements_per_thread = check_and_get_trait<fft_operator::elements_per_thread, base_type>::value;
            static constexpr unsigned int stride              = block_dim.x;

            static constexpr unsigned int suggested_ffts_per_block =
                check_and_get_trait<fft_operator::ffts_per_block, base_type>::suggested;

            static constexpr unsigned int storage_size       = get_storage_size();
            static constexpr unsigned int shared_memory_size = get_shared_memory_size();

            static constexpr unsigned int max_threads_per_block         = block_dim.x * block_dim.y * block_dim.z;

            static constexpr unsigned int implicit_type_batching =
                CUFFTDX_STD::is_same<this_fft_precision_t, __half>::value ? 2 : 1;

            static constexpr bool requires_workspace = check_and_get_fft_implementation<this_type>::requires_workspace;
            static constexpr unsigned int workspace_size = check_and_get_fft_implementation<this_type>::workspace_size;
        };

        template<class... Operators>
        constexpr dim3 fft_block_execution<Operators...>::block_dim;
        template<class... Operators>
        constexpr unsigned int fft_block_execution<Operators...>::ffts_per_block;
        template<class... Operators>
        constexpr unsigned int fft_block_execution<Operators...>::elements_per_thread;
        template<class... Operators>
        constexpr unsigned int fft_block_execution<Operators...>::stride;
        template<class... Operators>
        constexpr unsigned int fft_block_execution<Operators...>::suggested_ffts_per_block;
        template<class... Operators>
        constexpr unsigned int fft_block_execution<Operators...>::storage_size;
        template<class... Operators>
        constexpr unsigned int fft_block_execution<Operators...>::shared_memory_size;
        template<class... Operators>
        constexpr unsigned int fft_block_execution<Operators...>::max_threads_per_block;
        template<class... Operators>
        constexpr unsigned int fft_block_execution<Operators...>::implicit_type_batching;
        template<class... Operators>
        constexpr bool fft_block_execution<Operators...>::requires_workspace;
        template<class... Operators>
        constexpr unsigned int fft_block_execution<Operators...>::workspace_size;


        // [NOTE] Idea for testing static assert.
        //
        // Switch (macro) which changes behaviour from going to static_asserts
        // to returning description_error type in operator+(). That would required more indirection
        // in creating fft_description and fft_execution types.

        template<class... Operators>
        struct make_description {
        private:
            static constexpr bool has_block_operator =
                has_operator<fft_operator::block, fft_execution<Operators...>>::value;
            static constexpr bool has_thread_operator =
                has_operator<fft_operator::thread, fft_execution<Operators...>>::value;
            static constexpr bool has_execution_operator = has_block_operator || has_thread_operator;

            // Workaround (NVRTC)
            //
            // For NVRTC we need to utilize a in-between class called fft_block_execution_partial, otherwise
            // we run into a complation error if Block() is added to description before FFT description is
            // complete, example:
            //
            // Fails on NVRTC:
            //     Size<...>() + Direction<...>() + Type<...>() + Precision<...>() + Block() + SM<700>()
            // Works on NVRTC:
            //     Size<...>() + Direction<...>() + Type<...>() + Precision<...>() + SM<700>() + Block()
            //
            // This workaround disables some useful diagnostics based on static_asserts.
#ifdef __CUDACC_RTC__
            using operator_wrapper_type = fft_operator_wrapper<Operators...>;
            using fft_block_execution_type =
                typename CUFFTDX_STD::conditional<is_complete_fft<operator_wrapper_type>::value,
                                                  fft_block_execution<Operators...>,
                                                  fft_block_execution_partial<Operators...>>::type;
#else
            using fft_block_execution_type = fft_block_execution<Operators...>;
#endif

            using description_type = fft_description<Operators...>;
            using execution_type   = typename CUFFTDX_STD::conditional<has_block_operator,
                                                            fft_block_execution_type,
                                                            fft_thread_execution<Operators...>>::type;

        public:
            using type = typename CUFFTDX_STD::conditional<has_execution_operator, execution_type, description_type>::type;
        };

        template<class... Operators>
        using make_description_t = typename make_description<Operators...>::type;
    } // namespace detail

    template<class Operator1, class Operator2>
    __host__ __device__ __forceinline__ auto operator+(const Operator1&, const Operator2&) //
        -> typename CUFFTDX_STD::enable_if<detail::are_operator_expressions<Operator1, Operator2>::value,
                                   detail::make_description_t<Operator1, Operator2>>::type {
        return detail::make_description_t<Operator1, Operator2>();
    }

    template<class... Operators1, class Operator2>
    __host__ __device__ __forceinline__ auto operator+(const detail::fft_description<Operators1...>&,
                                                       const Operator2&) //
        -> typename CUFFTDX_STD::enable_if<detail::is_operator_expression<Operator2>::value,
                                   detail::make_description_t<Operators1..., Operator2>>::type {
        return detail::make_description_t<Operators1..., Operator2>();
    }

    template<class Operator1, class... Operators2>
    __host__ __device__ __forceinline__ auto operator+(const Operator1&,
                                                       const detail::fft_description<Operators2...>&) //
        -> typename CUFFTDX_STD::enable_if<detail::is_operator_expression<Operator1>::value,
                                   detail::make_description_t<Operator1, Operators2...>>::type {
        return detail::make_description_t<Operator1, Operators2...>();
    }

    template<class... Operators1, class... Operators2>
    __host__ __device__ __forceinline__ auto operator+(const detail::fft_description<Operators1...>&,
                                                       const detail::fft_description<Operators2...>&) //
        -> detail::make_description_t<Operators1..., Operators2...> {
        return detail::make_description_t<Operators1..., Operators2...>();
    }
} // namespace cufftdx

#undef STRINGIFY
#undef XSTRINGIFY

#endif // CUFFTDX_DETAIL_FFT_EXECUTION_HPP
