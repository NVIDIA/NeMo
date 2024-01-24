// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_TRAITS_DETAIL_GET_HPP
#define CUFFTDX_TRAITS_DETAIL_GET_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

namespace cufftdx {
    namespace detail {
        // Forward declaration
        template<fft_operator OperatorType, class T>
        struct is_operator;

        namespace __get {
            template<fft_operator OperatorType, class T>
            struct helper {
                using type = typename CUFFTDX_STD::conditional<is_operator<OperatorType, T>::value, T, void>::type;
            };

            // clang-format off
            template<fft_operator OperatorType,
                     template<class...> class DescriptionType,
                     class TypeHead,
                     class... TailTypes>
            struct helper<OperatorType, DescriptionType<TypeHead, TailTypes...>> {
                using type = typename CUFFTDX_STD::conditional<
                    is_operator<OperatorType, TypeHead>::value,
                    TypeHead,
                    typename helper<OperatorType, DescriptionType<TailTypes...>>::type>::type;
            };
            // clang-format on
        } // namespace __get

        /// get

        template<fft_operator OperatorType, class Description>
        struct get {
            using type = typename __get::helper<OperatorType, Description>::type;
        };

        template<fft_operator OperatorType, class Description>
        using get_t = typename get<OperatorType, Description>::type;

        /// get_or_default

        template<fft_operator OperatorType, class Description, class Default = void>
        struct get_or_default {
        private:
            using get_type = get_t<OperatorType, Description>;

        public:
            using type = typename CUFFTDX_STD::conditional<CUFFTDX_STD::is_void<get_type>::value, Default, get_type>::type;
        };

        template<fft_operator OperatorType, class Description, class Default = void>
        using get_or_default_t = typename get_or_default<OperatorType, Description, Default>::type;
    } // namespace detail
} // namespace cufftdx

#endif // CUFFTDX_TRAITS_DETAIL_GET_HPP
