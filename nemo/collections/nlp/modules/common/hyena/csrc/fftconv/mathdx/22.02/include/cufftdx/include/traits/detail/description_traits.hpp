// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP
#define CUFFTDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

#include "../../operators.hpp"
#include "../../detail/expressions.hpp"

#include "get.hpp"

namespace cufftdx {
    namespace detail {
        /// is_expression

        template<class T>
        struct is_expression: public CUFFTDX_STD::is_base_of<expression, T> {};

        template<class T, class U>
        struct are_expressions:
            public CUFFTDX_STD::integral_constant<bool, is_expression<T>::value && is_expression<U>::value> {};

        /// is_operator_expression

        template<class T>
        struct is_operator_expression: public CUFFTDX_STD::is_base_of<operator_expression, T> {};

        template<class T, class U>
        struct are_operator_expressions:
            public CUFFTDX_STD::integral_constant<bool, is_operator_expression<T>::value && is_operator_expression<U>::value> {
        };

        /// is_description_expression

        template<class T>
        struct is_description_expression: public CUFFTDX_STD::is_base_of<description_expression, T> {};

        /// is_operator

        template<fft_operator OperatorType, class T>
        struct is_operator: CUFFTDX_STD::false_type {};

        template<fft_direction D>
        struct is_operator<fft_operator::direction, Direction<D>>: CUFFTDX_STD::true_type {};

        template<class T>
        struct is_operator<fft_operator::precision, Precision<T>>: CUFFTDX_STD::true_type {};

        template<unsigned int N>
        struct is_operator<fft_operator::size, Size<N>>: CUFFTDX_STD::true_type {};

        template<unsigned int Architecture>
        struct is_operator<fft_operator::sm, SM<Architecture>>: CUFFTDX_STD::true_type {};

        template<fft_type T>
        struct is_operator<fft_operator::type, Type<T>>: CUFFTDX_STD::true_type {};

        template<>
        struct is_operator<fft_operator::thread, Thread>: CUFFTDX_STD::true_type {};

        template<>
        struct is_operator<fft_operator::block, Block>: CUFFTDX_STD::true_type {};

        template<unsigned int N>
        struct is_operator<fft_operator::elements_per_thread, ElementsPerThread<N>>: CUFFTDX_STD::true_type {};

        template<unsigned int N>
        struct is_operator<fft_operator::ffts_per_block, FFTsPerBlock<N>>: CUFFTDX_STD::true_type {};

        template<unsigned int X, unsigned int Y, unsigned int Z>
        struct is_operator<fft_operator::block_dim, BlockDim<X, Y, Z>>: CUFFTDX_STD::true_type {};

        // get_operator_type, TODO: Consider moving that info inside operator class

        template<class T>
        struct get_operator_type;

        template<fft_direction D>
        struct get_operator_type<Direction<D>> {
            static constexpr fft_operator value = fft_operator::direction;
        };

        template<class T>
        struct get_operator_type<Precision<T>> {
            static constexpr fft_operator value = fft_operator::precision;
        };

        template<unsigned int N>
        struct get_operator_type<Size<N>> {
            static constexpr fft_operator value = fft_operator::size;
        };

        template<unsigned int Architecture>
        struct get_operator_type<SM<Architecture>> {
            static constexpr fft_operator value = fft_operator::sm;
        };

        template<fft_type T>
        struct get_operator_type<Type<T>> {
            static constexpr fft_operator value = fft_operator::type;
        };

        template<>
        struct get_operator_type<Thread> {
            static constexpr fft_operator value = fft_operator::thread;
        };

        template<>
        struct get_operator_type<Block> {
            static constexpr fft_operator value = fft_operator::block;
        };

        template<unsigned int N>
        struct get_operator_type<ElementsPerThread<N>> {
            static constexpr fft_operator value = fft_operator::elements_per_thread;
        };

        template<unsigned int N>
        struct get_operator_type<FFTsPerBlock<N>> {
            static constexpr fft_operator value = fft_operator::ffts_per_block;
        };

        template<unsigned int X, unsigned int Y, unsigned int Z>
        struct get_operator_type<BlockDim<X, Y, Z>> {
            static constexpr fft_operator value = fft_operator::block_dim;
        };

        /// has_n_of

        namespace __has_n_of {
            template<unsigned int Counter, fft_operator OperatorType, class Head, class... Types>
            struct counter_helper {
                static constexpr unsigned int value = is_operator<OperatorType, Head>::value
                                                          ? counter_helper<(Counter + 1), OperatorType, Types...>::value
                                                          : counter_helper<Counter, OperatorType, Types...>::value;
            };

            template<unsigned int Counter, fft_operator OperatorType, class Head>
            struct counter_helper<Counter, OperatorType, Head> {
                static constexpr unsigned int value = is_operator<OperatorType, Head>::value ? Counter + 1 : Counter;
            };

            template<fft_operator OperatorType, class Operator>
            struct counter: CUFFTDX_STD::integral_constant<unsigned int, is_operator<OperatorType, Operator>::value> {};

            template<fft_operator OperatorType, template<class...> class Description, class... Types>
            struct counter<OperatorType, Description<Types...>>:
                CUFFTDX_STD::integral_constant<unsigned int, counter_helper<0, OperatorType, Types...>::value> {};
        } // namespace __has_n_of

        template<unsigned int N, fft_operator OperatorType, class Description>
        struct has_n_of: CUFFTDX_STD::integral_constant<bool, __has_n_of::counter<OperatorType, Description>::value == N> {};

        template<fft_operator OperatorType, class Description>
        struct has_at_most_one_of:
            CUFFTDX_STD::integral_constant<bool, (__has_n_of::counter<OperatorType, Description>::value <= 1)> {};

        /// has_block_operator
        namespace __has_block_operator {
            template<unsigned int Counter, class Head, class... Types>
            struct counter_helper {
                static constexpr unsigned int value = CUFFTDX_STD::is_base_of<block_operator_expression, Head>::value
                                                          ? counter_helper<(Counter + 1), Types...>::value
                                                          : counter_helper<Counter, Types...>::value;
            };

            template<unsigned int Counter, class Head>
            struct counter_helper<Counter, Head> {
                static constexpr unsigned int value =
                    CUFFTDX_STD::is_base_of<block_operator_expression, Head>::value ? Counter + 1 : Counter;
            };

            template<class Operator>
            struct counter:
                CUFFTDX_STD::integral_constant<unsigned int, CUFFTDX_STD::is_base_of<block_operator_expression, Operator>::value> {};

            template<template<class...> class Description, class... Types>
            struct counter<Description<Types...>>:
                CUFFTDX_STD::integral_constant<unsigned int, counter_helper<0, Types...>::value> {};
        } // namespace __has_block_operator

        template<class Description>
        struct has_any_block_operator:
            CUFFTDX_STD::integral_constant<bool, (__has_block_operator::counter<Description>::value > 0)> {};

        /// has_operator

        template<fft_operator OperatorType, class Description>
        struct has_operator:
            CUFFTDX_STD::integral_constant<bool, (__has_n_of::counter<OperatorType, Description>::value > 0)> {};

        /// deduce_direction_type

        template<class T>
        struct deduce_direction_type {
            using type = void;
        };

        template<>
        struct deduce_direction_type<Type<fft_type::c2r>> {
            using type = Direction<fft_direction::inverse>;
        };

        template<>
        struct deduce_direction_type<Type<fft_type::r2c>> {
            using type = Direction<fft_direction::forward>;
        };

        template<class T>
        using deduce_direction_type_t = typename deduce_direction_type<T>::type;

        // is_complete_description

        namespace __is_complete_description {
            template<class Description, class Enable = void>
            struct helper: CUFFTDX_STD::false_type {};

            template<template<class...> class Description, class... Types>
            struct helper<Description<Types...>,
                          typename CUFFTDX_STD::enable_if<is_description_expression<Description<Types...>>::value>::type> {
                using description_type = Description<Types...>;

                // Extract and/or deduce description types

                // Size
                using this_fft_size = get_t<fft_operator::size, description_type>;
                // Type (C2C, C2R, R2C)
                using default_fft_type = Type<fft_type::c2c>;
                using this_fft_type    = get_or_default_t<fft_operator::type, description_type, default_fft_type>;
                // Direction
                using deduced_fft_direction = deduce_direction_type_t<this_fft_type>;
                using this_fft_direction =
                    get_or_default_t<fft_operator::direction, description_type, deduced_fft_direction>;
                // SM
                using this_fft_sm = get_t<fft_operator::sm, description_type>;
                // Thread FFT
                static constexpr bool is_thread_execution = has_operator<fft_operator::thread, description_type>::value;

                static constexpr bool value =
                    !(CUFFTDX_STD::is_void<this_fft_size>::value || CUFFTDX_STD::is_void<this_fft_type>::value ||
                      CUFFTDX_STD::is_void<this_fft_direction>::value ||
                      // If we not that FFT is a thread FFT, then we don't require SM for completness
                      (CUFFTDX_STD::is_void<this_fft_sm>::value && !is_thread_execution));
            };
        } // namespace __is_complete_description

        template<class Description>
        struct is_complete_description:
            CUFFTDX_STD::integral_constant<bool, __is_complete_description::helper<Description>::value> {};
    } // namespace detail
} // namespace cufftdx

#endif // CUFFTDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP
