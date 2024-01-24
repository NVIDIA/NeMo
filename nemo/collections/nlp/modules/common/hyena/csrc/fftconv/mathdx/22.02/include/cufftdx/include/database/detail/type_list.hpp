// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_DATABASE_DETAIL_TYPE_LIST_HPP
#define CUFFTDX_DATABASE_DETAIL_TYPE_LIST_HPP

namespace cufftdx {
    namespace database {
        namespace detail {
            template<size_t Index, class T>
            struct type_list_element;

            template<class... Elements>
            struct type_list {};

            template<CUFFTDX_STD::size_t Index, class Head, class... Tail>
            struct type_list_element<Index, type_list<Head, Tail...>>:
                type_list_element<Index - 1, type_list<Tail...>> {};

            template<class Head, class... Tail>
            struct type_list_element<0, type_list<Head, Tail...>> {
                using type = Head;
            };
        } // namespace detail
    }     // namespace database
} // namespace cufftdx

#endif // CUFFTDX_DATABASE_DETAIL_TYPE_LIST_HPP
