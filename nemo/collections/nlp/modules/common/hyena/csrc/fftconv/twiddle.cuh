#pragma once

#include <cufftdx.hpp>
#include "lut.h"

// index must be less than or equal to FFT_SIZE / 8
template<int FFT_SIZE> inline __device__ float2 twiddle_lut(int index);
template<> inline __device__ float2 twiddle_lut<8>(int index) { return cufftdx::database::detail::lut_sp_4_8[index]; };
template<> inline __device__ float2 twiddle_lut<16>(int index) { return cufftdx::database::detail::lut_sp_4_16[index]; };
template<> inline __device__ float2 twiddle_lut<32>(int index) { return cufftdx::database::detail::lut_sp_4_32[index]; };
template<> inline __device__ float2 twiddle_lut<64>(int index) { return cufftdx::database::detail::lut_sp_4_64[index]; };
template<> inline __device__ float2 twiddle_lut<128>(int index) { return cufftdx::database::detail::lut_sp_4_128[index]; };
template<> inline __device__ float2 twiddle_lut<256>(int index) { return cufftdx::database::detail::lut_sp_4_256[index]; };
template<> inline __device__ float2 twiddle_lut<512>(int index) { return cufftdx::database::detail::lut_sp_4_512[index]; };
template<> inline __device__ float2 twiddle_lut<1024>(int index) { return cufftdx::database::detail::lut_sp_4_1024[index]; };
template<> inline __device__ float2 twiddle_lut<2048>(int index) { return cufftdx::database::detail::lut_sp_4_2048[index]; };
template<> inline __device__ float2 twiddle_lut<4096>(int index) { return cufftdx::database::detail::lut_sp_4_4096[index]; };
// Doesn't work with 8192 because of the edge case where the index is equal to FFT_SIZE / 8, and the
// lookup table doesn't have that value. So we have to use our own lookup table.
template<> inline __device__ float2 twiddle_lut<8192>(int index) { return cufftdx::database::detail::lut_mine_sp_8_8192[index]; };
template<> inline __device__ float2 twiddle_lut<16384>(int index) { return cufftdx::database::detail::lut_mine_sp_8_16384[index]; };

// The quadrant argument is not strictly necessary but we can compute it from the loop index,
// which will be unrolled and so it avoids branching.
template<int FFT_SIZE>
inline __device__ c10::complex<float> twiddle_from_lut(int quadrant, int index) {
    using cfloat_t = c10::complex<float>;
    if (quadrant == 0) {
        float2 twiddle = twiddle_lut<FFT_SIZE>(index);
        return cfloat_t(twiddle.x, twiddle.y);
    } else if (quadrant == 1) {
        float2 twiddle = twiddle_lut<FFT_SIZE>(FFT_SIZE / 4 - index);
        return cfloat_t(-twiddle.y, -twiddle.x);
    } else if (quadrant == 2) {
        float2 twiddle = twiddle_lut<FFT_SIZE>(index - FFT_SIZE / 4);
        return cfloat_t(twiddle.y, -twiddle.x);
    } else if (quadrant == 3) {
        float2 twiddle = twiddle_lut<FFT_SIZE>(FFT_SIZE / 2 - index);
        return cfloat_t(-twiddle.x, twiddle.y);
    } else {
        assert(false);
    }
}
