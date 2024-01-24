// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_TYPES_HPP__
#define CUFFTDX_TYPES_HPP__

#include <cuda_fp16.h>

namespace cufftdx {
    namespace detail {
        template<class T>
        struct complex_base {
            using value_type = T;

            complex_base()                    = default;
            complex_base(const complex_base&) = default;

            __device__ __forceinline__ __host__ constexpr complex_base(value_type re, value_type im): x(re), y(im) {}

            __device__ __forceinline__ __host__ constexpr value_type real() const { return x; }
            __device__ __forceinline__ __host__ constexpr value_type imag() const { return y; }
            __device__ __forceinline__ __host__ void real(value_type re) { x = re; }
            __device__ __forceinline__ __host__ void imag(value_type im) { y = im; }

            __device__ __forceinline__ __host__ complex_base& operator=(value_type re) {
                x = re;
                y = value_type();
                return *this;
            }
            __device__ __forceinline__ __host__ complex_base& operator+=(value_type re) {
                x += re;
                return *this;
            }
            __device__ __forceinline__ __host__ complex_base& operator-=(value_type re) {
                x -= re;
                return *this;
            }
            __device__ __forceinline__ __host__ complex_base& operator*=(value_type re) {
                x *= re;
                y *= re;
                return *this;
            }
            __device__ __forceinline__ __host__ complex_base& operator/=(value_type re) {
                x /= re;
                y /= re;
                return *this;
            }

            template<class K>
            __device__ __forceinline__ __host__ complex_base& operator=(const complex_base<K>& other) {
                x = other.real();
                y = other.imag();
                return *this;
            }

            template<class OtherType>
            __device__ __forceinline__ __host__ complex_base& operator+=(const OtherType& other) {
                x = x + other.x;
                y = y + other.y;
                return *this;
            }

            template<class OtherType>
            __device__ __forceinline__ __host__ complex_base& operator-=(const OtherType& other) {
                x = x - other.x;
                y = y - other.y;
                return *this;
            }

            template<class OtherType>
            __device__ __forceinline__ __host__ complex_base& operator*=(const OtherType& other) {
                auto saved_x = x;
                x            = x * other.x - y * other.y;
                y            = saved_x * other.y + y * other.x;
                return *this;
            }

            /// \internal
            value_type x, y;
        };

        template<class T>
        struct complex;

        template<>
        struct alignas(2 * sizeof(float)) complex<float>: complex_base<float> {
        private:
            using base_type = complex_base<float>;

        public:
            using value_type        = float;
            complex()               = default;
            complex(const complex&) = default;
            __device__ __forceinline__ __host__ constexpr complex(float re, float im): base_type(re, im) {}
            __device__ __forceinline__ __host__ explicit constexpr complex(const complex<double>& other);
            using base_type::operator+=;
            using base_type::operator-=;
            using base_type::operator*=;
            using base_type::operator/=;
            using base_type::operator=;
        };

        template<>
        struct alignas(2 * sizeof(double)) complex<double>: complex_base<double> {
        private:
            using base_type = complex_base<double>;

        public:
            using value_type        = double;
            complex()               = default;
            complex(const complex&) = default;
            __device__ __forceinline__ __host__ constexpr complex(double re, double im): base_type(re, im) {}
            __device__ __forceinline__ __host__ explicit constexpr complex(const complex<float>& other);
            using base_type::operator+=;
            using base_type::operator-=;
            using base_type::operator*=;
            using base_type::operator/=;
            using base_type::operator=;
        };

        // For FFT computations, complex<half2> should be in RRII layout.
        template<>
        struct alignas(2 * sizeof(__half2)) complex<__half2> {
            using value_type        = __half2;
            complex()               = default;
            complex(const complex&) = default;

            __device__ __forceinline__ __host__ complex(value_type re, value_type im): x(re), y(im) {}
#    if CUDA_VERSION < 11000
            __device__ __forceinline__ __host__ complex(double re, double im):
                x(__float2half2_rn(re)), y(__float2half2_rn(im)) {}
#    else
            __device__ __forceinline__ __host__ complex(double re, double im)

            {
                __half hre = __double2half(re);
                x          = __half2(hre, hre);
                __half him = __double2half(im);
                y          = __half2(him, him);
            }

#    endif
            __device__ __forceinline__ __host__ complex(float re, float im):
                x(__float2half2_rn(re)), y(__float2half2_rn(im)) {}

#    if CUDA_VERSION < 11000
            __device__ __forceinline__ __host__ explicit complex(const complex<double>& other):
                x(__float2half2_rn(other.real())), y(__float2half2_rn(other.imag())) {}
#    else
            __device__ __forceinline__ __host__ explicit complex(const complex<double>& other) {

                __half hre = __double2half(other.real());
                x          = __half2(hre, hre);
                __half him = __double2half(other.imag());
                y          = __half2(him, him);
            }
#    endif
            __device__ __forceinline__ __host__ explicit complex(const complex<float>& other):
                x(__float2half2_rn(other.real())), y(__float2half2_rn(other.imag())) {}

            __device__ __forceinline__ __host__ value_type real() const { return x; }
            __device__ __forceinline__ __host__ value_type imag() const { return y; }
            __device__ __forceinline__ __host__ void       real(value_type re) { x = re; }
            __device__ __forceinline__ __host__ void       imag(value_type im) { y = im; }

            __device__ __forceinline__ __host__ complex& operator=(value_type re) {
                x = re;
                y = value_type();
                return *this;
            }
            __device__ __forceinline__ complex& operator+=(value_type re) {
                x += re;
                return *this;
            }
            __device__ __forceinline__ complex& operator-=(value_type re) {
                x -= re;
                return *this;
            }
            __device__ __forceinline__ complex& operator*=(value_type re) {
                x *= re;
                y *= re;
                return *this;
            }
            __device__ __forceinline__ complex& operator/=(value_type re) {
                x /= re;
                y /= re;
                return *this;
            }

            __device__ __forceinline__ __host__ complex& operator=(const complex& other) {
                x = other.real();
                y = other.imag();
                return *this;
            }

            __device__ __forceinline__ complex& operator+=(const complex& other) {
                x = x + other.x;
                y = y + other.y;
                return *this;
            }

            __device__ __forceinline__ complex& operator-=(const complex& other) {
                x = x - other.x;
                y = y - other.y;
                return *this;
            }

            __device__ __forceinline__ complex& operator*=(const complex& other) {
                auto saved_x = x;
                x            = __hfma2(x, other.x, - y * other.y);
                y            = __hfma2(saved_x, other.y, y * other.x);
                return *this;
            }

            /// \internal
            value_type x, y;
        };

        __forceinline__ constexpr complex<float>::complex(const complex<double>& other):
            complex_base<float>(other.real(), other.imag()) {};

        __forceinline__ constexpr complex<double>::complex(const complex<float>& other):
            complex_base<double>(other.real(), other.imag()) {};
    } // namespace detail

    template<class T>
    using complex = typename detail::complex<T>;
} // namespace cufftdx

#endif // CUFFTDX_TYPES_HPP__
