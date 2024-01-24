
#ifndef CUFFTDX_EXAMPLE_FP16_COMMON_HPP_
#define CUFFTDX_EXAMPLE_FP16_COMMON_HPP_

namespace example {
    // Changes layout of complex<__half2> value from ((Real, Imag), (Real, Imag)) layout to
    // ((Real, Real), (Imag, Imag)) layout.
    __device__ __host__ __forceinline__ cufftdx::complex<__half2> to_rrii(
        cufftdx::complex<__half2> riri) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0)
        cufftdx::complex<__half2> rrii(__lows2half2(riri.x, riri.y),
                                       __highs2half2(riri.x, riri.y));
#else
        cufftdx::complex<__half2> rrii(__half2 {riri.x.x, riri.y.x},
                                       __half2 {riri.x.y, riri.y.y});
#endif
        return rrii;
    }

    // Converts to __half complex values to complex<__half2> in ((Real, Real), (Imag, Imag)) layout.
    __device__ __host__ __forceinline__ cufftdx::complex<__half2> to_rrii(
        __half2 ri1,
        __half2 ri2) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0)
        cufftdx::complex<__half2> rrii(__lows2half2(ri1, ri2),
                                       __highs2half2(ri1, ri2));
#else
        cufftdx::complex<__half2> rrii(__half2 {ri1.x, ri2.x},
                                       __half2 {ri1.y, ri2.y});
#endif
        return rrii;
    }

    // Changes layout of complex<__half2> value from ((Real, Real), (Imag, Imag)) layout to
    // ((Real, Imag), (Real, Imag)) layout.
    __device__ __host__ __forceinline__ cufftdx::complex<__half2> to_riri(
        cufftdx::complex<__half2> rrii) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0)
        cufftdx::complex<__half2> riri(__lows2half2(rrii.x, rrii.y),
                                       __highs2half2(rrii.x, rrii.y));
#else
        cufftdx::complex<__half2> riri(__half2 {rrii.x.x, rrii.y.x},
                                       __half2 {rrii.x.y, rrii.y.y});
#endif
        return riri;
    }

    // Return the first half complex number (as __half2) from complex<__half2> value with
    // ((Real, Real), (Imag, Imag)) layout.
    // Example: for rrii equal to ((1,2), (3,4)), it return __half2 (1, 3).
    __device__ __host__ __forceinline__ __half2 to_ri1(cufftdx::complex<__half2> rrii) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0)
        return __lows2half2(rrii.x, rrii.y);
#else
        return __half2 {rrii.x.x, rrii.y.x};
#endif
    }

    // Return the second half complex number (as __half2) from complex<__half2> value with
    // ((Real, Real), (Imag, Imag)) layout.
    // Example: for rrii equal to ((1,2), (3,4)), it return __half2 (2, 4).
    __device__ __host__ __forceinline__ __half2 to_ri2(cufftdx::complex<__half2> rrii) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0)
        return __highs2half2(rrii.x, rrii.y);
#else
        return __half2 {rrii.x.y, rrii.y.y};
#endif
    }
} // namespace example

#endif // CUFFTDX_EXAMPLE_FP16_COMMON_HPP_
