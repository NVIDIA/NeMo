// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_DATABASE_DETAIL_BLOCK_FFT_HPP
#define CUFFTDX_DATABASE_DETAIL_BLOCK_FFT_HPP

#include "cuda_fp16.h"

#include "../../operators.hpp"
#include "../../traits/detail/make_complex_type.hpp"
#include "type_list.hpp"

namespace cufftdx {
    namespace database {
        namespace detail {
            template<unsigned int Size /* FFT size */,
                     class Precision,
                     fft_type      Type,
                     fft_direction Direction,
                     unsigned int  Architecture>
            struct block_fft_record {
                static constexpr bool defined = false;
            };

            // Forward SM72 records to SM70 records
            template<unsigned int Size /* FFT size */, class Precision, fft_type Type, fft_direction Direction>
            struct block_fft_record<Size, Precision, Type, Direction, 720>:
                public block_fft_record<Size, Precision, Type, Direction, 700> {};

            // Forward SM75 records to SM70 records
            template<unsigned int Size /* FFT size */, class Precision, fft_type Type, fft_direction Direction>
            struct block_fft_record<Size, Precision, Type, Direction, 750>:
                public block_fft_record<Size, Precision, Type, Direction, 700> {};

            // Forward SM86 records to SM70 records
            template<unsigned int Size /* FFT size */, class Precision, fft_type Type, fft_direction Direction>
            struct block_fft_record<Size, Precision, Type, Direction, 860>:
                public block_fft_record<Size, Precision, Type, Direction, 700> {};

            // Forward R2C records to C2C records
            template<unsigned int Size /* FFT size */, class Precision>
            struct block_fft_record<Size, Precision, fft_type::r2c, fft_direction::forward, 800>:
                public block_fft_record<Size, Precision, fft_type::c2c, fft_direction::forward, 800> {};
            template<unsigned int Size /* FFT size */, class Precision>
            struct block_fft_record<Size, Precision, fft_type::r2c, fft_direction::forward, 700>:
                public block_fft_record<Size, Precision, fft_type::c2c, fft_direction::forward, 700> {};

            // Forward C2R records to C2C records
            template<unsigned int Size /* FFT size */, class Precision>
            struct block_fft_record<Size, Precision, fft_type::c2r, fft_direction::inverse, 800>:
                public block_fft_record<Size, Precision, fft_type::c2c, fft_direction::inverse, 800> {};
            template<unsigned int Size /* FFT size */, class Precision>
            struct block_fft_record<Size, Precision, fft_type::c2r, fft_direction::inverse, 700>:
                public block_fft_record<Size, Precision, fft_type::c2c, fft_direction::inverse, 700> {};

            template<unsigned int ElementsPerThread /* Number of elements processed per thread */,
                     unsigned int StorageSize /* Storage size, number of elements in input/output array */,
                     unsigned int ThreadsPerFFT,
                     unsigned int FFTsPerBlock,
                     unsigned int SharedMemorySize /* Size of shared mem. required by one FFT */,
                     unsigned int FunctionId>
            struct block_fft_implementation {
                static constexpr unsigned int elements_per_thread = ElementsPerThread;
                static constexpr unsigned int storage_size        = StorageSize;
                static constexpr unsigned int threads_per_fft     = ThreadsPerFFT;
                static constexpr unsigned int ffts_per_block      = FFTsPerBlock;
                static constexpr unsigned int shared_memory_size  = SharedMemorySize;
                static constexpr unsigned int function_id         = FunctionId;
            };

            template<class Implementation, typename PrecisionType, unsigned int TRPOption>
            struct enforce_trp {
                static constexpr bool matches = true;
            };

            template<class Implementation, typename PrecisionType>
            struct enforce_trp<Implementation, PrecisionType, 1 /* X */> {
                static constexpr bool matches =
                    sizeof(PrecisionType) * Implementation::elements_per_thread * Implementation::threads_per_fft ==
                    Implementation::shared_memory_size;
            };

            template<class Implementation, typename PrecisionType>
            struct enforce_trp<Implementation, PrecisionType, 2 /* XY */> {
                static constexpr bool matches =
                    sizeof(PrecisionType) * 2 * Implementation::elements_per_thread * Implementation::threads_per_fft ==
                    Implementation::shared_memory_size;
            };

            // Selects block_fft_implementation from type_list based on ElementsPerThread,
            // if there is no such implementation in list search_by_ept::type is set to void.
            template<unsigned int ElementsPerThread,
                     typename PrecisionType,
                     unsigned int TRPOption,
                     class ImplementationList>
            struct search_by_ept;

            template<unsigned int ElementsPerThread,
                     typename PrecisionType,
                     unsigned int TRPOption,
                     class Implementation>
            struct search_by_ept<ElementsPerThread, PrecisionType, TRPOption, type_list<Implementation>> {
                using type = typename CUFFTDX_STD::conditional<
                    (Implementation::elements_per_thread == ElementsPerThread &&
                     (Implementation::threads_per_fft == 1 ||
                      enforce_trp<Implementation, PrecisionType, TRPOption>::matches)),
                    Implementation,
                    void>::type;
            };

            template<unsigned int ElementsPerThread,
                     typename PrecisionType,
                     unsigned int TRPOption,
                     class Head,
                     class... Tail>
            struct search_by_ept<ElementsPerThread, PrecisionType, TRPOption, type_list<Head, Tail...>> {
                using type = typename CUFFTDX_STD::conditional<
                    (Head::elements_per_thread == ElementsPerThread &&
                     (Head::threads_per_fft == 1 || enforce_trp<Head, PrecisionType, TRPOption>::matches)),
                    Head,
                    typename search_by_ept<ElementsPerThread, PrecisionType, TRPOption, type_list<Tail...>>::type>::
                    type;
            };

            template<unsigned int FunctionID, typename T, unsigned int FFTsPerBlock>
            __device__ void cufftdx_private_function(typename cufftdx::detail::make_complex_type<T>::cufftdx_type* rmem,
                                                     unsigned smem);


            template<unsigned int FunctionID, typename T, unsigned int FFTsPerBlock>
            __device__ void cufftdx_private_function_wrapper(typename cufftdx::detail::make_complex_type<T>::cufftdx_type* rmem,
                                                             void* smem) {
                unsigned smem32 = static_cast<unsigned>(__cvta_generic_to_shared(smem));
                cufftdx_private_function<FunctionID, T, FFTsPerBlock>(rmem, smem32);
            }
        } // namespace detail
    }     // namespace database
} // namespace cufftdx

#endif // CUFFTDX_DATABASE_DETAIL_BLOCK_FFT_HPP
