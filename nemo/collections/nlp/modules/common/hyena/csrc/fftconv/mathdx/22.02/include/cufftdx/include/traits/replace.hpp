// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_TRAITS_REPLACE_HPP
#define CUFFTDX_TRAITS_REPLACE_HPP

#include "fft_traits.hpp"

namespace cufftdx {
    // Implementation of replace
    namespace detail {
        template<class... Operators>
        struct check_operators;

        template<class Operator>
        struct check_operators<Operator> {
            static constexpr bool value = detail::is_operator_expression<Operator>::value;
        };

        template<class HeadOperators, class... TailOperators>
        struct check_operators<HeadOperators, TailOperators...> {
            static constexpr bool value =
                detail::is_operator_expression<HeadOperators>::value && check_operators<TailOperators...>::value;
        };

        template<class Description, class... NewOperators>
        struct remove_operators;

        template<class Description, class... NewOperators>
        using remove_operators_t = typename remove_operators<Description, NewOperators...>::type;

        template<class Description, class NewOperator>
        struct remove_operators<Description, NewOperator> {
            static constexpr fft_operator operator_type   = get_operator_type<NewOperator>::value;
            static constexpr bool         can_be_replaced = !(is_operator<fft_operator::block, NewOperator>::value ||
                                                      is_operator<fft_operator::thread, NewOperator>::value);
            using type =
                typename CUFFTDX_STD::conditional<can_be_replaced, filter_t<Description, operator_type>, Description>::type;
        };

        template<class Description, class HeadNewOperators, class... TailNewOperators>
        struct remove_operators<Description, HeadNewOperators, TailNewOperators...> {
            static constexpr fft_operator operator_type = get_operator_type<HeadNewOperators>::value;
            static constexpr bool can_be_replaced       = !(is_operator<fft_operator::block, HeadNewOperators>::value ||
                                                      is_operator<fft_operator::thread, HeadNewOperators>::value);
            using filtered_type                         = filter_t<Description, operator_type>;
            using type                                  = typename CUFFTDX_STD::conditional<can_be_replaced,
                                                   remove_operators_t<filtered_type, TailNewOperators...>,
                                                   remove_operators_t<Description, TailNewOperators...>>::type;
        };

        template<class Description, class OriginalDescription, bool Force, class... NewOperators>
        struct add_operators;

        template<class Description, class OriginalDescription, bool Force, class... NewOperators>
        using add_operators_t = typename add_operators<Description, OriginalDescription, Force, NewOperators...>::type;

        template<class Description, class OriginalDescription, bool Force, class NewOperator>
        struct add_operators<Description, OriginalDescription, Force, NewOperator> {
            static constexpr fft_operator operator_type = get_operator_type<NewOperator>::value;
            static constexpr bool         had_operator  = has_operator<operator_type, OriginalDescription>::value;
            static constexpr bool         can_be_added =
                (had_operator || Force) && !(is_operator<fft_operator::block, NewOperator>::value ||
                                             is_operator<fft_operator::thread, NewOperator>::value);
            using type = typename CUFFTDX_STD::
                conditional<can_be_added, concatenate_description_t<NewOperator, Description>, Description>::type;
        };

        template<class Description, class OriginalDescription, bool Force, class HeadNewOperators, class... TailNewOperators>
        struct add_operators<Description, OriginalDescription, Force, HeadNewOperators, TailNewOperators...> {
            static constexpr fft_operator operator_type = get_operator_type<HeadNewOperators>::value;
            static constexpr bool         had_operator  = has_operator<operator_type, OriginalDescription>::value;
            // We only add operator if:
            // * it's not block or thread operator
            // * there is the same type of operator in the original FFT description
            static constexpr bool can_be_added =
                (had_operator || Force) && !(is_operator<fft_operator::block, HeadNewOperators>::value ||
                                             is_operator<fft_operator::thread, HeadNewOperators>::value);
            using concatenated_type = concatenate_description_t<HeadNewOperators, Description>;
            using type =
                typename CUFFTDX_STD::conditional<can_be_added,
                                          add_operators_t<concatenated_type, OriginalDescription, Force, TailNewOperators...>,
                                          add_operators_t<Description, OriginalDescription, Force, TailNewOperators...>>::type;
        };

        template<class Description, bool Force, class... NewOperators>
        struct replace_force {
            static_assert(is_fft<Description>::value, "Description is not a cuFFTDx FFT description");
            static_assert(detail::check_operators<NewOperators...>::value,
                          "One of operators is not a cuFFTDx FFT operator");

            // First remove all operators of the same types as NewOperators...,
            // and then add NewOperators... to the description.
            using filtered_description = detail::remove_operators_t<Description, NewOperators...>;
            using replaced_description =
                detail::add_operators_t<filtered_description, Description, Force, NewOperators...>;
            using new_fft_description = typename CUFFTDX_STD::conditional<
                is_fft_execution<replaced_description>::value,
                replaced_description,
                typename detail::convert_to_fft_description<replaced_description>::type>::type;

        public:
            /// cuFFTDx FFT description with replaced operators
            using type = replaced_description;
        };
    } // namespace detail

    /// \class replace
    /// \brief Replaces operators of the same type as \p NewOperators in \p Description with \p NewOperators.
    ///
    /// \par Overview
    /// * Replaces operators of the same type as \p NewOperators in \p Description with \p NewOperators.
    /// * cufftdx::Thread and cufftdx::Block operators in \p NewOperators are ignored.
    ///
    /// \tparam Description - cuFFTDx FFT description type to process
    /// \tparam NewOperators - the list of operators to use as replacement
    template<class Description, class... NewOperators>
    struct replace {
        /// cuFFTDx FFT description with replaced operators
        using type = typename detail::replace_force<Description, false, NewOperators...>::type;
    };

    /// Alias template for replace<Description, NewOperators...>::type
    template<class Description, class... NewOperators>
    using replace_t = typename replace<Description, NewOperators...>::type;
} // namespace cufftdx

#endif // CUFFTDX_TRAITS_REPLACE_HPP
