// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_TRAITS_FFT_TRAITS_HPP
#define CUFFTDX_TRAITS_FFT_TRAITS_HPP

#include "../detail/fft_description_fd.hpp"

#include "../operators.hpp"

#include "detail/get.hpp"
#include "detail/description_traits.hpp"
#include "detail/make_complex_type.hpp"

namespace cufftdx {
    template<class Description>
    struct size_of {
    private:
        static constexpr bool has_size = detail::has_operator<fft_operator::size, Description>::value;
        static_assert(has_size, "Description does not have size defined");

    public:
        using value_type                  = unsigned int;
        static constexpr value_type value = detail::get_t<fft_operator::size, Description>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    constexpr unsigned int size_of<Description>::value;

    template<class Description>
    struct sm_of {
    private:
        static constexpr bool has_sm = detail::has_operator<fft_operator::sm, Description>::value;
        static_assert(has_sm, "Description does not have CUDA architecture defined");

    public:
        using value_type                  = unsigned int;
        static constexpr value_type value = detail::get_t<fft_operator::sm, Description>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    constexpr unsigned int sm_of<Description>::value;

    template<class Description>
    struct type_of {
        using value_type = fft_type;
        static constexpr value_type value =
            detail::get_or_default_t<fft_operator::type, Description, Type<fft_type::c2c>>::value;
        constexpr operator value_type() const noexcept { return value; }
    };

    template<class Description>
    constexpr fft_type type_of<Description>::value;

    template<class Description>
    struct direction_of {
    private:
        using deduced_fft_direction = detail::deduce_direction_type_t<Type<type_of<Description>::value>>;
        using this_fft_direction =
            detail::get_or_default_t<fft_operator::direction, Description, deduced_fft_direction>;

        static_assert(!CUFFTDX_STD::is_void<this_fft_direction>::value,
                      "Description has neither direction defined, nor it can be deduced from its type");

    public:
        using value_type                  = fft_direction;
        static constexpr value_type value = this_fft_direction::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    constexpr fft_direction direction_of<Description>::value;

    template<class Description>
    struct precision_of {
        using type = typename detail::get_or_default_t<fft_operator::precision, Description, Precision<float>>::type;
    };

    template<class Description>
    using precision_of_t = typename precision_of<Description>::type;

    template<class Description>
    struct is_fft {
        using value_type                  = bool;
        static constexpr value_type value = detail::is_expression<Description>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    constexpr bool is_fft<Description>::value;

    template<class Description>
    struct is_fft_execution {
        static constexpr auto block  = detail::has_operator<fft_operator::block, Description>::value;
        static constexpr auto thread = detail::has_operator<fft_operator::thread, Description>::value;

    public:
        using value_type                  = bool;
        static constexpr value_type value = is_fft<Description>::value && (thread || block);
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    constexpr bool is_fft_execution<Description>::value;

    template<class Description>
    struct is_complete_fft {
        using value_type = bool;
        static constexpr value_type value =
            is_fft<Description>::value && detail::is_complete_description<Description>::value;
        constexpr operator value_type() const noexcept { return value; }
    };

    template<class Description>
    constexpr bool is_complete_fft<Description>::value;

    template<class Description>
    struct is_complete_fft_execution {
        using value_type                  = bool;
        static constexpr value_type value = is_fft_execution<Description>::value && is_complete_fft<Description>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    constexpr bool is_complete_fft_execution<Description>::value;

    namespace detail {
        // Concatenates OperatorType to the description (faster than using decltype and adding operators)
        template<class OperatorType, class Description>
        struct concatenate_description;

        template<class OperatorType, template<class...> class Description, class... Operators>
        struct concatenate_description<OperatorType, Description<Operators...>> {
            using type = Description<OperatorType, Operators...>;
        };

        template<class OperatorType, class Description>
        using concatenate_description_t = typename concatenate_description<OperatorType, Description>::type;

        // Removes give OperatorType from an FFT description
        template<class Description, fft_operator OperatorType>
        struct filter {
            using type = void;
        };

        template<class Description, fft_operator OperatorType>
        using filter_t = typename filter<Description, OperatorType>::type;

        template<template<class...> class Description, fft_operator OperatorType>
        struct filter<Description<>, OperatorType> {
            using type = Description<>;
        };

        template<template<class...> class Description, fft_operator OperatorType, class Head, class... Tail>
        struct filter<Description<Head, Tail...>, OperatorType> {
            using type = typename CUFFTDX_STD::conditional<
                is_operator<OperatorType, Head>::value,
                filter_t<Description<Tail...>, OperatorType>,
                concatenate_description_t<Head, typename filter<Description<Tail...>, OperatorType>::type> //
                >::type;
        };

        template<class Description>
        struct convert_to_fft_description {
            using type = void;
        };

        template<template<class...> class Description, class... Types>
        struct convert_to_fft_description<Description<Types...>> {
            using type = typename detail::fft_description<Types...>;
        };
    } // namespace detail

    // This extracts an FFT description from FFT execution description.
    template<class Description>
    struct extract_fft_description {
    private:
        // Converts execution description to simple description, filter_t will remove Thread and Block operators
        using fft_description_type = typename detail::convert_to_fft_description<Description>::type;

    public:
        static_assert(is_fft<Description>::value, "Description is not a cuFFDx FFT description");
        using type = typename CUFFTDX_STD::conditional<
            detail::is_operator_expression<Description>::value,
            Description, // For single operator or if Description just return Description
            detail::filter_t<detail::filter_t<fft_description_type, fft_operator::block>, fft_operator::thread> //
            >::type;
    };

    template<class Description>
    using extract_fft_description_t = typename extract_fft_description<Description>::type;
} // namespace cufftdx

#endif // CUFFTDX_TRAITS_FFT_TRAITS_HPP
