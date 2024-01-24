// Copyright (c) 2022 Tri Dao, Dan Fu

#include <torch/torch.h>

#include <stdio.h>
#include <cuda/std/complex>

#include <cufftdx.hpp>

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>

#include <c10/cuda/CUDAException.h>  // For C10_CUDA_KERNEL_LAUNCH_CHECK

#include "static_switch.h"
#include "twiddle.cuh"

// *************** FOR ERROR CHECKING *******************
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL
// *************** FOR ERROR CHECKING *******************

template<int N>
inline __device__ void gelu(float (&output)[N], const float (&input)[N]) {
    constexpr float kAlpha = M_SQRT1_2;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        output[i] = input[i] * 0.5 * (1 + erff(input[i] * kAlpha));
    }
}

template<int N>
inline __device__ void dgelu(float (&grad_input)[N], const float (&grad_output)[N], const float (&input)[N]) {
    constexpr float kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
    constexpr float kAlpha = M_SQRT1_2;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        const float cdf = 0.5 * (1 + erff(input[i] * kAlpha));
        const float pdf = expf(-0.5 * input[i] * input[i]) * kBeta;
        grad_input[i] = grad_output[i] * (cdf + input[i] * pdf);
    }
}

// GeLU(input0) * input1
template<int N>
inline __device__ void geglu(float (&output)[N], const float (&input0)[N], const float (&input1)[N]) {
    constexpr float kAlpha = M_SQRT1_2;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        output[i] = input1 * (input0[i] * 0.5 * (1 + erff(input0[i] * kAlpha)));
    }
}

template<int N>
inline __device__ void dgeglu(float (&grad_input0)[N], float (&grad_input1)[N],
                              const float (&grad_output)[N], const float (&input0)[N], const float (&input1)[N]) {
    constexpr float kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
    constexpr float kAlpha = M_SQRT1_2;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        const float cdf = 0.5 * (1 + erff(input0[i] * kAlpha));
        const float pdf = expf(-0.5 * input0[i] * input0[i]) * kBeta;
        grad_input0[i] = grad_output[i] * input1[i] * (cdf + input0[i] * pdf);
        grad_input1[i] = grad_output[i] * input0[i] * cdf;
    }
}

template<typename T>
__device__ c10::complex<T> pointwise_mul(const c10::complex<T> a, const c10::complex<T> b) {
    return c10::complex<T>(a.real_ * b.real_, a.imag_ * b.imag_);
}


inline __device__ void read_rrii(cufftdx::detail::complex<__half2> val, c10::complex<float> result [2]) {
    using cfloat_t = c10::complex<float>;
    result[0] = cfloat_t(__half2float(val.x.x), __half2float(val.y.x));
    result[1] = cfloat_t(__half2float(val.x.y), __half2float(val.y.y));
}

inline __device__ cufftdx::detail::complex<__half2> write_rrii(c10::complex<float> val [2]) {
    using complex_t = typename cufftdx::detail::complex<__half2>;
    return complex_t {
        __float22half2_rn(float2 {val[0].real(), val[1].real()}),
        __float22half2_rn(float2 {val[0].imag(), val[1].imag()}),
    };
}

// Implement a real FFT of size 2 * N by calling a complex FFT of size N.
// http://www.robinscheibler.org/2013/02/13/real-fft.html
template<typename FFT>
inline __device__ void rfft(c10::complex<float> (&thread_data)[FFT::elements_per_thread],
                            c10::complex<float> *shared_mem){
    using cfloat_t = typename c10::complex<float>;
    using complex_t = typename cufftdx::detail::complex<float>;
    constexpr int N = cufftdx::size_of<FFT>::value;
    constexpr int EPT = FFT::elements_per_thread;

    complex_t *smem_c = reinterpret_cast<complex_t *>(shared_mem);
    complex_t (&thread_data_fft)[EPT] = reinterpret_cast<complex_t (&)[EPT]>(thread_data);
    FFT().execute(thread_data_fft, smem_c);
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < EPT; ++i) {
        smem_c[threadIdx.x + FFT::stride * i] = thread_data_fft[i];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < EPT; ++i) {
        if ((threadIdx.x == 0) && (i == 0)) {
            cfloat_t smem_val = shared_mem[0];
            thread_data[i] = cfloat_t(smem_val.real_ + smem_val.imag_, smem_val.real_ - smem_val.imag_);
        } else {
            int index = threadIdx.x + FFT::stride * i;
            cfloat_t smem_val_0 = shared_mem[index], smem_val_1 = shared_mem[N - index];
            cfloat_t X_even = smem_val_0 + std::conj(smem_val_1);
            // constexpr cfloat_t j = cfloat_t(0.f, 1.f);
            // cfloat_t X_odd = -j * (smem_val_0 - std::conj(smem_val_1));
            // Algebraic simplification
            cfloat_t X_odd = cfloat_t(smem_val_0.imag_ + smem_val_1.imag_, -smem_val_0.real_ + smem_val_1.real_);
            // cfloat_t twiddle;
            // sincospif(-float(index) / N, reinterpret_cast<float *>(&twiddle) + 1,
            //           reinterpret_cast<float *>(&twiddle));
            // Reading from lookup table is faster than computing the twiddle
            int quadrant = i / (EPT / 4);
            cfloat_t twiddle = twiddle_from_lut<N * 2>(quadrant, index);
            thread_data[i] = (X_even + X_odd * twiddle) / 2;
        }
    }
}

// Implement a conjugate symmetric inverse FFT of size 2 * N by calling a complex iFFT of size N.
// http://www.robinscheibler.org/2013/02/13/real-fft.html
template<typename IFFT>
inline __device__ void irfft(c10::complex<float> (&thread_data)[IFFT::elements_per_thread],
                             c10::complex<float> *shared_mem){
    using cfloat_t = typename c10::complex<float>;
    using complex_t = typename cufftdx::detail::complex<float>;
    constexpr int N = cufftdx::size_of<IFFT>::value;
    constexpr int EPT = IFFT::elements_per_thread;

    #pragma unroll
    for (int i = 0; i < EPT; ++i) {
        shared_mem[threadIdx.x + IFFT::stride * i] = thread_data[i];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < EPT; ++i) {
        if ((threadIdx.x == 0) && (i == 0)) {
            cfloat_t smem_val = shared_mem[0];
            thread_data[i] = cfloat_t(smem_val.real_ + smem_val.imag_, smem_val.real_ - smem_val.imag_);
            // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
            //     printf("%f.4f+%.4fi, ", thread_data[i].real_, thread_data[i].imag_);
            // }
        } else {
            int index = threadIdx.x + IFFT::stride * i;
            cfloat_t smem_val_0 = shared_mem[index], smem_val_1 = shared_mem[N - index];
            cfloat_t X_even = smem_val_0 + std::conj(smem_val_1);
            // cfloat_t twiddle;
            // sincospif(float(index) / N, reinterpret_cast<float *>(&twiddle) + 1,
            //           reinterpret_cast<float *>(&twiddle));;
            // Reading from lookup table is faster than computing the twiddle
            int quadrant = i / (EPT / 4);
            cfloat_t twiddle = std::conj(twiddle_from_lut<N * 2>(quadrant, index));
            // cfloat_t X_odd = (smem_val_0 - std::conj(smem_val_1)) * twiddle;
            // constexpr cfloat_t j = cfloat_t(0.f, 1.f);
            // thread_data[i] = (X_even + j * X_odd) / 2;
            // Algebraic simplification
            cfloat_t X_odd_j = cfloat_t(-smem_val_0.imag_ - smem_val_1.imag_, smem_val_0.real_ - smem_val_1.real_) * twiddle;
            thread_data[i] = X_even + X_odd_j;
        }
    }
    __syncthreads();
    IFFT().execute(reinterpret_cast<complex_t (&)[EPT]>(thread_data),
                   reinterpret_cast<complex_t *>(shared_mem));
}

// // Implement a real FFT of size 2 * N by calling a complex FFT of size N.
// // http://www.robinscheibler.org/2013/02/13/real-fft.html
// template<typename FFT>
// inline __device__ void rfftfp16(cufftdx::detail::complex<__half2> (&thread_data)[FFT::elements_per_thread], cufftdx::detail::complex<__half2> *shared_mem){
//     using cfloat_t = typename c10::complex<float>;
//     using complex_t = typename cufftdx::detail::complex<__half2>;
//     constexpr int N = cufftdx::size_of<FFT>::value;
//     constexpr int EPT = FFT::elements_per_thread;

//     // complex_t *smem_c = reinterpret_cast<complex_t *>(shared_mem);
//     // complex_t (&thread_data_fft)[EPT] = reinterpret_cast<complex_t (&)[EPT]>(thread_data);
//     FFT().execute(thread_data, shared_mem);
//     __syncthreads();
//     #pragma unroll
//     for (int i = 0; i < EPT; ++i) {
//         shared_mem[threadIdx.x + FFT::stride * i] = thread_data[i];
//     }
//     __syncthreads();
//     #pragma unroll
//     for (int i = 0; i < EPT; ++i) {
//         if ((threadIdx.x == 0) && (i == 0)) {
//             complex_t smem_val_half = shared_mem[0];
//             cfloat_t smem_val[2];
//             read_rrii(smem_val_half, smem_val);
//             // thread_data[i] = cfloat_t(smem_val.real_ + smem_val.imag_, smem_val.real_ - smem_val.imag_);
//             cfloat_t res[2] = {
//                 cfloat_t(smem_val[0].real() + smem_val[0].imag(), smem_val[0].real() - smem_val[0].imag()),
//                 cfloat_t(smem_val[1].real() + smem_val[1].imag(), smem_val[1].real() - smem_val[1].imag())
//             };
//             thread_data[i] = write_rrii(res);
//         } else {
//             int index = threadIdx.x + FFT::stride * i;
//             // cfloat_t smem_val_0 = shared_mem[index], smem_val_1 = shared_mem[N - index];
//             complex_t smem_val_0_half = shared_mem[index], smem_val_1_half = shared_mem[N - index];
//             cfloat_t smem_val_0[2], smem_val_1[2];
//             read_rrii(smem_val_0_half, smem_val_0);
//             read_rrii(smem_val_1_half, smem_val_1);

//             // cfloat_t X_even = smem_val_0 + std::conj(smem_val_1);
//             cfloat_t X_even[2] = {
//                 smem_val_0[0] + std::conj(smem_val_1[0]),
//                 smem_val_0[1] + std::conj(smem_val_1[1])
//             };

//             // constexpr cfloat_t j = cfloat_t(0.f, 1.f);
//             // cfloat_t X_odd = -j * (smem_val_0 - std::conj(smem_val_1));
//             // Algebraic simplification
//             // cfloat_t X_odd = cfloat_t(smem_val_0.imag_ + smem_val_1.imag_, -smem_val_0.real_ + smem_val_1.real_);
//             cfloat_t X_odd[2] = {
//                 cfloat_t(smem_val_0[0].imag() + smem_val_1[0].imag(), -smem_val_0[0].real() + smem_val_1[0].real()),
//                 cfloat_t(smem_val_0[1].imag() + smem_val_1[1].imag(), -smem_val_0[1].real() + smem_val_1[1].real())
//             };

//             // cfloat_t twiddle;
//             // sincospif(-float(index) / N, reinterpret_cast<float *>(&twiddle) + 1,
//             //           reinterpret_cast<float *>(&twiddle));
//             // Reading from lookup table is faster than computing the twiddle
//             int quadrant = i / (EPT / 4);
//             cfloat_t twiddle = twiddle_from_lut<N * 2>(quadrant, index);

//             // thread_data[i] = (X_even + X_odd * twiddle) / 2;
//             cfloat_t result[2] = {
//                 (X_even[0] + X_odd[0] * twiddle) / 2,
//                 (X_even[1] + X_odd[1] * twiddle) / 2
//             };
//             thread_data[i] = write_rrii(result);
//         }
//     }
// }

// // Implement a conjugate symmetric inverse FFT of size 2 * N by calling a complex iFFT of size N.
// // http://www.robinscheibler.org/2013/02/13/real-fft.html
// template<typename IFFT>
// inline __device__ void irfftfp16(cufftdx::detail::complex<__half2> (&thread_data)[IFFT::elements_per_thread],
// cufftdx::detail::complex<__half2> *shared_mem){
//     using cfloat_t = typename c10::complex<float>;
//     using complex_t = typename cufftdx::detail::complex<__half2>;
//     constexpr int N = cufftdx::size_of<IFFT>::value;
//     constexpr int EPT = IFFT::elements_per_thread;

//     #pragma unroll
//     for (int i = 0; i < EPT; ++i) {
//         shared_mem[threadIdx.x + IFFT::stride * i] = thread_data[i];
//     }
//     __syncthreads();
//     #pragma unroll
//     for (int i = 0; i < EPT; ++i) {
//         if ((threadIdx.x == 0) && (i == 0)) {
//             complex_t smem_val_half = shared_mem[0];
//             cfloat_t smem_val[2];
//             read_rrii(smem_val_half, smem_val);
//             // thread_data[i] = cfloat_t(smem_val.real_ + smem_val.imag_, smem_val.real_ - smem_val.imag_);
//             cfloat_t res[2] = {
//                 cfloat_t(smem_val[0].real() + smem_val[0].imag(), smem_val[0].real() - smem_val[0].imag()),
//                 cfloat_t(smem_val[1].real() + smem_val[1].imag(), smem_val[1].real() - smem_val[1].imag())
//             };
//             thread_data[i] = write_rrii(res);
//             // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
//             //     printf("%f.4f+%.4fi, ", thread_data[i].real_, thread_data[i].imag_);
//             // }
//         } else {
//             int index = threadIdx.x + IFFT::stride * i;
//             // cfloat_t smem_val_0 = shared_mem[index], smem_val_1 = shared_mem[N - index];
//             complex_t smem_val_0_half = shared_mem[index], smem_val_1_half = shared_mem[N - index];
//             cfloat_t smem_val_0[2], smem_val_1[2];
//             read_rrii(smem_val_0_half, smem_val_0);
//             read_rrii(smem_val_1_half, smem_val_1);

//             // cfloat_t X_even = smem_val_0 + std::conj(smem_val_1);
//             cfloat_t X_even[2] = {
//                 smem_val_0[0] + std::conj(smem_val_1[0]),
//                 smem_val_0[1] + std::conj(smem_val_1[1])
//             };

//             // cfloat_t twiddle;
//             // sincospif(float(index) / N, reinterpret_cast<float *>(&twiddle) + 1,
//             //           reinterpret_cast<float *>(&twiddle));;
//             // Reading from lookup table is faster than computing the twiddle
//             int quadrant = i / (EPT / 4);
//             cfloat_t twiddle = std::conj(twiddle_from_lut<N * 2>(quadrant, index));

//             // cfloat_t X_odd = (smem_val_0 - std::conj(smem_val_1)) * twiddle;
//             // constexpr cfloat_t j = cfloat_t(0.f, 1.f);
//             // thread_data[i] = (X_even + j * X_odd) / 2;
//             // Algebraic simplification
//             // cfloat_t X_odd_j = cfloat_t(-smem_val_0.imag_ - smem_val_1.imag_, smem_val_0.real_ - smem_val_1.real_) * twiddle;
//             cfloat_t X_odd_j[2] = {
//                 cfloat_t(-smem_val_0[0].imag() - smem_val_1[0].imag(), smem_val_0[0].real() - smem_val_1[0].real()) * twiddle,
//                 cfloat_t(-smem_val_0[1].imag() - smem_val_1[1].imag(), smem_val_0[1].real() - smem_val_1[1].real()) * twiddle
//             };

//             // thread_data[i] = X_even + X_odd_j;
//             cfloat_t res[2] = {
//                 (X_even[0] + X_odd_j[0]),
//                 (X_even[1] + X_odd_j[1])
//             };
//             thread_data[i] = write_rrii(res);
//         }
//     }
//     __syncthreads();
//     IFFT().execute(reinterpret_cast<complex_t (&)[EPT]>(thread_data),
//                    reinterpret_cast<complex_t *>(shared_mem));
// }

template<bool QV, int HEADDIM, typename FFT, typename IFFT, typename input_t, typename output_t=input_t, bool GELU_INPUT=false, bool GELU_OUTPUT=true, bool GELU_Q=false>
__launch_bounds__( FFT::max_threads_per_block )
__global__ void fftconv_fwd_kernel(const input_t *__restrict__ inputData,
                                   const c10::complex<float> *__restrict__ filterData,
                                   const input_t *__restrict__ inputMulVData,
                                   const input_t *__restrict__ inputMulQData,
                                   const float *__restrict__ DData,
                                   const float *__restrict__ dropmaskData,
                                   output_t *__restrict__ outputData,
                                   int batch_size,
                                   int H,
                                   int signal_size,
                                   size_t batch_stride, size_t H_stride,
                                   bool output_hbl_layout) {

    using complex_t = typename cufftdx::detail::complex<float>;
    using cfloat_t = typename c10::complex<float>;
    constexpr int N = cufftdx::size_of<FFT>::value;
    constexpr int EPT = FFT::elements_per_thread;
    static_assert(FFT::storage_size == EPT);
    static_assert(IFFT::storage_size == EPT);

    using BlockLoad_input = cub::BlockLoad<cfloat_t, FFT::block_dim.x, EPT / 2, cub::BLOCK_LOAD_STRIPED>;
    using BlockLoad_filter = cub::BlockLoad<cfloat_t, FFT::block_dim.x, EPT, cub::BLOCK_LOAD_STRIPED>;
    using BlockStore_output = cub::BlockStore<c10::complex<output_t>, FFT::block_dim.x, EPT / 2, cub::BLOCK_STORE_STRIPED>;

    extern __shared__ cfloat_t shared_mem[];

    float result_data[EPT] = { 0 };

    cfloat_t filter_data[EPT];
    // Adjust for head dim
    unsigned int filter_id = blockIdx.y;
    BlockLoad_filter().Load(filterData + filter_id * (N + 1), filter_data);
    // CHECK THIS!!!
    if (threadIdx.x == 0) {
        filter_data[0].imag_ = *(reinterpret_cast<const float *>(filterData + filter_id * (N + 1) + N));
    }
    #pragma unroll
    for ( int i = 0; i < EPT; i++ ) { filter_data[i] /= 2 * N; }

    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     for (int i = 0; i < FFT::storage_size / 2; i++) {
    //         printf("%.4f+%.4fi, ", filter_data[i].real_, filter_data[i].imag_);
    //     }
    //     printf("\n");
    // }

    // CHECK THIS!!!
    float D_val = DData[filter_id];
    unsigned int dropmask_id = blockIdx.x * H + blockIdx.y;
    float dropmask_val = dropmaskData == nullptr ? 1.f : dropmaskData[dropmask_id];

    float v_data[EPT];
    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    // Used for inputMulVData only
    size_t v_offset = blockIdx.x * batch_stride + (blockIdx.y * HEADDIM + blockIdx.z) * H_stride;
    if (QV) {
        BlockLoad_input().Load(reinterpret_cast<const c10::complex<input_t> *>(inputMulVData + v_offset),
                               reinterpret_cast<cfloat_t (&)[EPT / 2]>(v_data),
                               signal_size / 2, cfloat_t(0.f));
        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
        //     printf("v_data: ");
        //     for (int i = 0; i < EPT; i++) {
        //         printf("%.4f, ", v_data[i]);
        //     }
        //     printf("\n");
        // }
    }

    // Doesn't seem to matter if we put #pragma unroll
    // #pragma unroll
    for (int head_i = 0; head_i < HEADDIM; head_i++) {
        // Local array and copy data into it
        float u_og_data[EPT];
        cfloat_t thread_data[EPT];

        // Id for inputData and inputMulQData
        size_t u_offset = blockIdx.x * batch_stride + (blockIdx.y * HEADDIM + head_i) * H_stride;

        BlockLoad_input().Load(reinterpret_cast<const c10::complex<input_t> *>(inputData + u_offset),
                               reinterpret_cast<cfloat_t (&)[EPT / 2]>(u_og_data),
                               signal_size / 2, cfloat_t(0.f));
        // TODO: what if signal_size is odd
        if (GELU_INPUT) { gelu(u_og_data, u_og_data); }

        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
        //     printf("u_og_data: ");
        //     for (int i = 0; i < EPT; i++) {
        //         printf("%.4f, ", u_og_data[i]);
        //     }
        //     printf("\n");
        // }

        if (QV) {
            #pragma unroll
            for (int i = 0; i < EPT; ++i) {
                u_og_data[i] *= v_data[i];
            }
        }

        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
        //     printf("u_og_data: ");
        //     for (int i = 0; i < EPT; i++) {
        //         printf("%.4f, ", u_og_data[i]);
        //     }
        //     printf("\n");
        // }

        #pragma unroll
        for (int i = 0; i < EPT; ++i) {
            thread_data[i] = i < EPT / 2 ? cfloat_t(u_og_data[i * 2], u_og_data[i * 2 + 1]) : cfloat_t(0.f);
        }

        if (head_i > 0) { __syncthreads(); }
        // Execute FFT
        rfft<FFT>(thread_data, shared_mem);

        #pragma unroll
        for ( int i = 0; i < EPT; i++ ) {
            thread_data[i] = (threadIdx.x == 0) && (i == 0) ?
                pointwise_mul(thread_data[i], filter_data[i]) : thread_data[i] * filter_data[i];
        }

        // Execute FFT
        __syncthreads();
        irfft<IFFT>(thread_data, shared_mem);

        float out_data[EPT] {};

        #pragma unroll
        for ( int i = 0; i < EPT; i++ ) {
            out_data[i] = reinterpret_cast<float (&)[EPT * 2]>(thread_data)[i] + u_og_data[i] * D_val;
        }

        // GELU_OUTPUT and dropout
        // https://github.com/pytorch/pytorch/blob/dc169d53aa266560750ea25ee0cf31c7e614550d/aten/src/ATen/native/cuda/Activation.cu#L395
        if (GELU_OUTPUT) { gelu(out_data, out_data); }
        #pragma unroll
        for ( int i = 0; i < EPT; i++ ) {
            out_data[i] *= dropmask_val;
        }

        float q_data[EPT];

        if (QV) {
            BlockLoad_input().Load(reinterpret_cast<const c10::complex<input_t> *>(inputMulQData + u_offset),
                                   reinterpret_cast<cfloat_t (&)[EPT / 2]>(q_data),
                                   signal_size / 2, cfloat_t(0.f));

            // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
            //     printf("q_data: ");
            //     for (int i = 0; i < EPT; i++) {
            //         printf("%.4f, ", q_data[i]);
            //     }
            //     printf("\n");
            // }

            if (GELU_Q) { gelu(q_data, q_data); }
            #pragma unroll
            for (int i = 0; i < EPT; ++i) {
                out_data[i] *= q_data[i];
            }
        }

        #pragma unroll
        for (int i = 0; i < EPT; ++i) {
            result_data[i] += out_data[i];
        }
    }

    // Save results
    c10::complex<output_t> write_data[EPT / 2];
    #pragma unroll
    for (int i = 0; i < EPT / 2; ++i) {
        write_data[i] = c10::complex(output_t(result_data[i * 2]), output_t(result_data[i * 2 + 1]));
    }
    unsigned int output_fft_id = !output_hbl_layout ? blockIdx.x * H + blockIdx.y * HEADDIM + blockIdx.z : blockIdx.x + (blockIdx.y * HEADDIM + blockIdx.z) * batch_size;
    BlockStore_output().Store(reinterpret_cast<c10::complex<output_t> *>(outputData + output_fft_id * signal_size),
                              write_data, signal_size / 2);
    // TODO: what if signal_size is odd?
}


// template<bool QV, int HEADDIM, typename FFT, typename IFFT, typename input_t, typename output_t=input_t, bool GELU_INPUT=false, bool GELU_OUTPUT=true, bool GELU_Q=false>
// __launch_bounds__( FFT::max_threads_per_block )
// __global__ void fftconv_fwd_kernelfp16(const input_t *__restrict__ inputData,
//                                    const c10::complex<float> *__restrict__ filterData,
//                                    const input_t *__restrict__ inputMulVData,
//                                    const input_t *__restrict__ inputMulQData,
//                                    const float *__restrict__ DData,
//                                    const float *__restrict__ dropmaskData,
//                                    output_t *__restrict__ outputData,
//                                    int batch_size,
//                                    int H,
//                                    int signal_size,
//                                    bool output_hbl_layout) {

//     using complex_t = typename cufftdx::detail::complex<__half2>;
//     using cfloat_t = typename c10::complex<float>;
//     constexpr int N = cufftdx::size_of<FFT>::value;
//     constexpr int EPT = FFT::elements_per_thread;
//     static_assert(FFT::storage_size == EPT);
//     static_assert(IFFT::storage_size == EPT);

//     using BlockLoad_input = cub::BlockLoad<cfloat_t, FFT::block_dim.x, EPT / 2, cub::BLOCK_LOAD_STRIPED>;
//     using BlockLoad_filter = cub::BlockLoad<cfloat_t, FFT::block_dim.x, EPT, cub::BLOCK_LOAD_STRIPED>;
//     using BlockStore_output = cub::BlockStore<c10::complex<output_t>, FFT::block_dim.x, EPT / 2, cub::BLOCK_STORE_STRIPED>;

//     extern __shared__ cfloat_t shared_mem[];

//     float result_data[2][EPT] = { 0 };

//     cfloat_t filter_data[EPT];
//     unsigned int filter_id = blockIdx.y;
//     BlockLoad_filter().Load(filterData + filter_id * (N + 1), filter_data);
//     if (threadIdx.x == 0) {
//         filter_data[0].imag_ = *(reinterpret_cast<const float *>(filterData + filter_id * (N + 1) + N));
//     }
//     #pragma unroll
//     for ( int i = 0; i < EPT; i++ ) { filter_data[i] /= 2 * N; }

//     // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)) {
//     //     for (int i = 0; i < FFT::storage_size / 2; i++) {
//     //         printf("%.4f+%.4fi, ", filter_data[i].real_, filter_data[i].imag_);
//     //     }
//     //     printf("\n");
//     // }

//     float D_val = DData[filter_id];
//     unsigned int dropmask_id = blockIdx.x * H + blockIdx.y;
//     float dropmask_val = dropmaskData == nullptr ? 1.f : dropmaskData[dropmask_id];

//     // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
//     // Used for inputMulVData only
//     unsigned int global_fft_id = blockIdx.x * H * FFT::ffts_per_block + blockIdx.y * HEADDIM + blockIdx.z;

//     // do not pragma unroll this!!
//     for (int head_i = 0; head_i < HEADDIM; head_i++) {
//         // Local array and copy data into it
//         float u_og_data[2][EPT];
//         float v_data[2][EPT];
//         complex_t thread_data[EPT];

//         // Id for inputData and inputMulQData
//         unsigned int head_fft_id = blockIdx.x * H * FFT::ffts_per_block + blockIdx.y * HEADDIM + head_i;

//         BlockLoad_input().Load(reinterpret_cast<const c10::complex<input_t> *>(inputData + head_fft_id * signal_size),
//                             reinterpret_cast<cfloat_t (&)[EPT / 2]>(u_og_data[0]),
//                             signal_size / 2, cfloat_t(0.f));
//         BlockLoad_input().Load(reinterpret_cast<const c10::complex<input_t> *>(inputData + head_fft_id * signal_size + H * signal_size),
//                             reinterpret_cast<cfloat_t (&)[EPT / 2]>(u_og_data[1]),
//                             signal_size / 2, cfloat_t(0.f));
//         // TODO: what if signal_size is odd
//         if (GELU_INPUT) {
//             gelu(u_og_data[0], u_og_data[0]);
//             gelu(u_og_data[1], u_og_data[1]);
//         }

//         // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) && (head_i == 0)) {
//         //     printf("u_og_data[0]: ");
//         //     for (int i = 0; i < EPT; i++) {
//         //         printf("%.4f, ", u_og_data[0][i]);
//         //     }
//         //     printf("\n");
//         //     printf("u_og_data[1]: ");
//         //     for (int i = 0; i < EPT; i++) {
//         //         printf("%.4f, ", u_og_data[1][i]);
//         //     }
//         //     printf("\n");
//         // }

//         BlockLoad_input().Load(reinterpret_cast<const c10::complex<input_t> *>(inputMulVData + global_fft_id * signal_size),
//                         reinterpret_cast<cfloat_t (&)[EPT / 2]>(v_data[0]),
//                         signal_size / 2, cfloat_t(0.f));
//         BlockLoad_input().Load(reinterpret_cast<const c10::complex<input_t> *>(inputMulVData + global_fft_id * signal_size + H * signal_size),
//                         reinterpret_cast<cfloat_t (&)[EPT / 2]>(v_data[1]),
//                         signal_size / 2, cfloat_t(0.f));

//         #pragma unroll
//         for (int i = 0; i < EPT; ++i) {
//             u_og_data[0][i] = u_og_data[0][i] * v_data[0][i];
//             u_og_data[1][i] = u_og_data[1][i] * v_data[1][i];
//         }

//         // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) && (head_i == 0)) {
//         //     printf("u_og_data[0]: ");
//         //     for (int i = 0; i < EPT; i++) {
//         //         printf("%.4f, ", u_og_data[0][i]);
//         //     }
//         //     printf("\n");
//         //     printf("u_og_data[1]: ");
//         //     for (int i = 0; i < EPT; i++) {
//         //         printf("%.4f, ", u_og_data[1][i]);
//         //     }
//         //     printf("\n");
//         // }

//         #pragma unroll
//         for (int i = 0; i < EPT; ++i) {
//             thread_data[i] = i < EPT / 2 ? complex_t {
//                 __float22half2_rn({u_og_data[0][i * 2], u_og_data[1][i * 2]}),
//                 __float22half2_rn({u_og_data[0][i * 2 + 1], u_og_data[1][i * 2 + 1]})
//             } : complex_t { __float22half2_rn({0.f, 0.f}), __float22half2_rn({0.f, 0.f}) };
//         }

//         if (head_i > 0) { __syncthreads(); }
//         // Execute FFT
//         rfftfp16<FFT>(thread_data, reinterpret_cast<cufftdx::detail::complex<__half2> *>(shared_mem));

//         // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) && (head_i == 0)) {
//         //     cfloat_t thread_floats [2];
//         //     printf("fft(u)[0]: ");
//         //     for (int i = 0; i < EPT; i++) {
//         //         read_rrii(thread_data[i], thread_floats);
//         //         printf("%.4f+%.4fi, ", thread_floats[0].real_, thread_floats[0].imag_);
//         //     }
//         //     printf("\n");
//         //     printf("fft(u)[1]: ");
//         //     for (int i = 0; i < EPT; i++) {
//         //         read_rrii(thread_data[i], thread_floats);
//         //         printf("%.4f+%.4fi, ", thread_floats[1].real_, thread_floats[1].imag_);
//         //     }
//         //     printf("\n");
//         // }

//         // here, do a pointwise mul converting from rr fp16 to fp32
//         cfloat_t thread_floats [2];
//         cfloat_t res [2];
//         #pragma unroll
//         for ( int i = 0; i < EPT; i++ ) {
//             read_rrii(thread_data[i], thread_floats);
//             for ( int j = 0; j < 2; j++ ) {
//                 res[j] = (threadIdx.x == 0) && (i == 0) ?
//                 pointwise_mul(thread_floats[j], filter_data[i]) : thread_floats[j] * filter_data[i];
//             }
//             thread_data[i] = write_rrii(res);
//             // thread_data[i] = (threadIdx.x == 0) && (i == 0) ?
//             //     pointwise_mul(thread_data[i], filter_data[i]) : thread_data[i] * filter_data[i];
//         }

//         // Execute FFT
//         __syncthreads();
//         irfftfp16<IFFT>(thread_data, reinterpret_cast<cufftdx::detail::complex<__half2> *>(shared_mem));


//         float out_data[2][EPT] {};

//         #pragma unroll
//         for ( int i = 0; i < EPT; i++ ) {
//             out_data[0][i] = __half2float(reinterpret_cast<__half2 (&)[EPT * 2]>(thread_data)[i].x) + u_og_data[0][i] * D_val;
//             out_data[1][i] = __half2float(reinterpret_cast<__half2 (&)[EPT * 2]>(thread_data)[i].y) + u_og_data[1][i] * D_val;
//         }

//         // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) && (head_i == 0)) {
//         //     printf("out[0]: ");
//         //     for (int i = 0; i < EPT; i++) {
//         //         printf("%.4f, ", out_data[0][i]);
//         //     }
//         //     printf("\n");
//         //     printf("out[1]: ");
//         //     for (int i = 0; i < EPT; i++) {
//         //         printf("%.4f, ", out_data[1][i]);
//         //     }
//         //     printf("\n");
//         // }

//         // GELU and dropout
//         // https://github.com/pytorch/pytorch/blob/dc169d53aa266560750ea25ee0cf31c7e614550d/aten/src/ATen/native/cuda/Activation.cu#L395

//         if (GELU_OUTPUT) {
//             gelu(out_data[0], out_data[0]);
//             gelu(out_data[1], out_data[1]);
//         }
//         #pragma unroll
//         for ( int i = 0; i < EPT; i++ ) {
//             out_data[0][i] = out_data[0][i] * dropmask_val;
//             out_data[1][i] = out_data[1][i] * dropmask_val;
//         }

//         float q_data[2][EPT];

//         BlockLoad_input().Load(reinterpret_cast<const c10::complex<input_t> *>(inputMulQData + head_fft_id * signal_size),
//                         reinterpret_cast<cfloat_t (&)[EPT / 2]>(q_data[0]),
//                         signal_size / 2, cfloat_t(0.f));
//         BlockLoad_input().Load(reinterpret_cast<const c10::complex<input_t> *>(inputMulQData + head_fft_id * signal_size + H * signal_size),
//                         reinterpret_cast<cfloat_t (&)[EPT / 2]>(q_data[1]),
//                         signal_size / 2, cfloat_t(0.f));

//         // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) && (head_i == 0)) {
//         //     printf("q[0]: ");
//         //     for (int i = 0; i < EPT; i++) {
//         //         printf("%.4f, ", q_data[0][i]);
//         //     }
//         //     printf("\n");
//         //     printf("q[1]: ");
//         //     for (int i = 0; i < EPT; i++) {
//         //         printf("%.4f, ", q_data[1][i]);
//         //     }
//         //     printf("\n");
//         // }

//         #pragma unroll
//         for (int i = 0; i < EPT; ++i) {
//             out_data[0][i] = q_data[0][i] * out_data[0][i];
//             out_data[1][i] = q_data[1][i] * out_data[1][i];
//         }

//         #pragma unroll
//         for (int i = 0; i < EPT; ++i) {
//             result_data[0][i] += out_data[0][i];
//             result_data[1][i] += out_data[1][i];
//         }
//     }

//     // Save results
//     c10::complex<output_t> write_data[2][EPT / 2];
//     #pragma unroll
//     for (int i = 0; i < EPT / 2; ++i) {
//         write_data[0][i] = c10::complex(output_t(result_data[0][i * 2]), output_t(result_data[0][i * 2 + 1]));
//         write_data[1][i] = c10::complex(output_t(result_data[1][i * 2]), output_t(result_data[1][i * 2 + 1]));
//     }

//     unsigned int output_fft_id = !output_hbl_layout ? blockIdx.x * H * FFT::ffts_per_block + blockIdx.y * HEADDIM + blockIdx.z : blockIdx.x * FFT::ffts_per_block + (blockIdx.y * HEADDIM + blockIdx.z) * batch_size;
//     BlockStore_output().Store(reinterpret_cast<c10::complex<output_t> *>(outputData + output_fft_id * signal_size),
//                               write_data[0], signal_size / 2);
//     BlockStore_output().Store(reinterpret_cast<c10::complex<output_t> *>(outputData + output_fft_id * signal_size + (!output_hbl_layout ? H * signal_size : signal_size)),
//                               write_data[1], signal_size / 2);
//     // TODO: what if signal_size is odd?
// }

template <bool GELU_OUTPUT, uint FFT_SIZE, uint EPT, typename input_t, typename output_t=input_t>
void fftconv_fwd_cuda(const input_t *u, const c10::complex<float> *filter,
                      const input_t *v, int head_dim, const input_t *q,
                      const float *D, const float *dropout_mask, output_t *out,
                      bool gelu_inp, bool gelu_q, int batch_size, int H, int signal_size,
                      size_t batch_stride, size_t H_stride, bool output_hbl_layout, bool fftfp16) {
#if defined(__CUDA_ARCH__)
    constexpr uint ARCH = __CUDA_ARCH__;
#else
    constexpr uint ARCH = 700;
#endif
    
    (void) gelu_inp; // these options are not supported right now
    (void) gelu_q;   // these options are not supported right now

    switch (head_dim) {
        case 1:
        {
            constexpr uint FPB = 1;
            // FFT is defined, its: size, type, direction, precision. Block() operator
            // informs that FFT will be executed on block level. Shared memory is
            // required for co-operation between threads.
            
            using FFT_base = decltype(cufftdx::Block() + cufftdx::Size<FFT_SIZE>() + cufftdx::Precision<float>() +
                                    cufftdx::ElementsPerThread<EPT>() + cufftdx::FFTsPerBlock<FPB>() + cufftdx::SM<ARCH>()
                                    + cufftdx::Type<cufftdx::fft_type::c2c>());

            using FFT = decltype(FFT_base() + cufftdx::Direction<cufftdx::fft_direction::forward>());
            using IFFT = decltype(FFT_base() + cufftdx::Direction<cufftdx::fft_direction::inverse>());

            // By default the shared memory size is 4 * FFT_SIZE (idk how).
            // So it wouldn't work for our rfft and irfft functions.
            const auto shared_memory_size = std::max({FFT::shared_memory_size, IFFT::shared_memory_size,
                                                    8 * FFT_SIZE});
            // printf("shared_memory_size = %d\n", shared_memory_size);
            
            // unsigned int blocks_per_grid { static_cast<unsigned int>( std::ceil( batch_size / FPB ) ) };
            unsigned int H_per_grid { static_cast<unsigned int>( std::ceil( H / FPB ) ) };
            dim3 block(batch_size, H_per_grid / head_dim, head_dim);
            BOOL_SWITCH(v != nullptr, QV, [&] {
                auto kernel = &fftconv_fwd_kernel<QV, 1, FFT, IFFT, input_t, output_t, false, GELU_OUTPUT, false>;
                // Increase dynamic memory limit if required.
                CUDA_RT_CALL( cudaFuncSetAttribute(kernel,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    shared_memory_size ));
                kernel<<<block, FFT::block_dim, shared_memory_size>>>(u, filter, v, q, D, dropout_mask, out, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout);
            });
            break;
        }
        case 8:
        {
            if (fftfp16) {
                // constexpr uint FPB = 2;

                // using FFT_base = decltype(cufftdx::Block() + cufftdx::Size<FFT_SIZE>() + cufftdx::Precision<__half>() +
                //                         cufftdx::ElementsPerThread<EPT>() + cufftdx::FFTsPerBlock<FPB>() + cufftdx::SM<ARCH>()
                //                         + cufftdx::Type<cufftdx::fft_type::c2c>());

                // using FFT = decltype(FFT_base() + cufftdx::Direction<cufftdx::fft_direction::forward>());
                // using IFFT = decltype(FFT_base() + cufftdx::Direction<cufftdx::fft_direction::inverse>());

                // // By default the shared memory size is 4 * FFT_SIZE (idk how).
                // // So it wouldn't work for our rfft and irfft functions.
                // const auto shared_memory_size = std::max({FFT::shared_memory_size, IFFT::shared_memory_size,
                //                                         8 * FFT_SIZE});
                // // printf("shared_memory_size = %d\n", shared_memory_size);

                // unsigned int blocks_per_grid { static_cast<unsigned int>( std::ceil( batch_size / FPB ) ) };
                // // unsigned int H_per_grid { static_cast<unsigned int>( std::ceil( H / FPB ) ) };
                // dim3 block(blocks_per_grid, H / head_dim, head_dim);
                // constexpr bool QV = true;  // Multi-head requires QV

                // auto kernel = &fftconv_fwd_kernelfp16<QV, 8, FFT, IFFT, input_t, output_t, false, GELU_OUTPUT, false>;

                // // Increase dynamic memory limit if required.
                // CUDA_RT_CALL( cudaFuncSetAttribute(kernel,
                //                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                //                                     shared_memory_size ));
                // kernel<<<block, FFT::block_dim, shared_memory_size>>>(u, filter, v, q, D, dropout_mask, out, batch_size, H, signal_size, output_hbl_layout);
            }
            else {
                // uncomment this and the kernel line below to go back to fp32
                constexpr uint FPB = 1;
                
                using FFT_base = decltype(cufftdx::Block() + cufftdx::Size<FFT_SIZE>() + cufftdx::Precision<float>() +
                                        cufftdx::ElementsPerThread<EPT>() + cufftdx::FFTsPerBlock<FPB>() + cufftdx::SM<ARCH>()
                                        + cufftdx::Type<cufftdx::fft_type::c2c>());

                using FFT = decltype(FFT_base() + cufftdx::Direction<cufftdx::fft_direction::forward>());
                using IFFT = decltype(FFT_base() + cufftdx::Direction<cufftdx::fft_direction::inverse>());

                // By default the shared memory size is 4 * FFT_SIZE (idk how).
                // So it wouldn't work for our rfft and irfft functions.
                const auto shared_memory_size = std::max({FFT::shared_memory_size, IFFT::shared_memory_size,
                                                        8 * FFT_SIZE});
                // printf("shared_memory_size = %d\n", shared_memory_size);
                
                unsigned int blocks_per_grid { static_cast<unsigned int>( std::ceil( batch_size / FPB ) ) };
                // unsigned int H_per_grid { static_cast<unsigned int>( std::ceil( H / FPB ) ) };
                dim3 block(blocks_per_grid, H / head_dim, head_dim);
                constexpr bool QV = true;  // Multi-head requires QV

                // change this line to go back to fp32
                auto kernel = &fftconv_fwd_kernel<QV, 8, FFT, IFFT, input_t, output_t, false, GELU_OUTPUT, false>;

                // Increase dynamic memory limit if required.
                CUDA_RT_CALL( cudaFuncSetAttribute(kernel,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    shared_memory_size ));
                kernel<<<block, FFT::block_dim, shared_memory_size>>>(u, filter, v, q, D, dropout_mask, out, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout);
            }
            break;
        }
        default:
            AT_ERROR("fftconv forward not implemented for this head_dim");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
};

template <typename input_t, typename output_t=input_t>
void fftconv_fwd_cuda_dispatch(const input_t *u, const c10::complex<float> *filter,
                               const input_t *v, int head_dim, const input_t *q,
                               const float *D, const float *dropout_mask, output_t *out,
                               bool gelu, bool gelu_inp, bool gelu_q, int batch_size, int H, int signal_size,
                               size_t batch_stride, size_t H_stride, int fft_size,
                               bool output_hbl_layout, bool fftfp16) {
    BOOL_SWITCH(gelu, GELU_OUTPUT, [&] {
        switch(fft_size) {
            case 16:
                fftconv_fwd_cuda<GELU_OUTPUT, 8, 4, input_t, output_t>(
                    u, filter, v, head_dim, q, D, dropout_mask, out, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            case 32:
                fftconv_fwd_cuda<GELU_OUTPUT, 16, 4, input_t, output_t>(
                    u, filter, v, head_dim, q, D, dropout_mask, out, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            case 64:
                fftconv_fwd_cuda<GELU_OUTPUT, 32, 4, input_t, output_t>(
                    u, filter, v, head_dim, q, D, dropout_mask, out, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            case 128:
                fftconv_fwd_cuda<GELU_OUTPUT, 64, 4, input_t, output_t>(
                    u, filter, v, head_dim, q, D, dropout_mask, out, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            case 256:
                fftconv_fwd_cuda<GELU_OUTPUT, 128, 4, input_t, output_t>(
                    u, filter, v, head_dim, q, D, dropout_mask, out, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            case 512:
                fftconv_fwd_cuda<GELU_OUTPUT, 256, 8, input_t, output_t>(
                    u, filter, v, head_dim, q, D, dropout_mask, out, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            case 1024:
                fftconv_fwd_cuda<GELU_OUTPUT, 512, 16, input_t, output_t>(
                    u, filter, v, head_dim, q, D, dropout_mask, out, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            case 2048:
                fftconv_fwd_cuda<GELU_OUTPUT, 1024, 16, input_t, output_t>(
                    u, filter, v, head_dim, q, D, dropout_mask, out, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            case 4096:
                fftconv_fwd_cuda<GELU_OUTPUT, 2048, 8, input_t, output_t>(
                    u, filter, v, head_dim, q, D, dropout_mask, out, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            case 8192:
                fftconv_fwd_cuda<GELU_OUTPUT, 4096, 8, input_t, output_t>(
                    u, filter, v, head_dim, q, D, dropout_mask, out, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            case 16384:
                fftconv_fwd_cuda<GELU_OUTPUT, 8192, 8, input_t, output_t>(
                    u, filter, v, head_dim, q, D, dropout_mask, out, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            default:
                AT_ERROR("fftconv forward not implemented for this fft_size");
        }
    });
}

template<bool QV, int HEADDIM, typename FFT, typename IFFT, typename input_t, typename output_t=input_t, bool GELU_INPUT=false, bool GELU_OUTPUT=true, bool GELU_Q=false>
__launch_bounds__( FFT::max_threads_per_block )
__global__ void fftconv_bwd_kernel(const output_t *__restrict__ doutData,
                                   const input_t *__restrict__ inputData,
                                   const c10::complex<float> *__restrict__ filterData,
                                   const input_t *__restrict__ inputMulVData,
                                   const input_t *__restrict__ inputMulQData,
                                   const float *__restrict__ DData,
                                   const float *__restrict__ dropmaskData,
                                   input_t *__restrict__ duData,
                                   c10::complex<float> *__restrict__ dfilterData,
                                   float *__restrict__ dDData,
                                   float *__restrict__ dvData,
                                   input_t *__restrict__ dqData,
                                   int batch_size,
                                   int H,
                                   int signal_size,
                                   size_t batch_stride, size_t H_stride,
                                   bool output_hbl_layout) {

    using complex_t = typename cufftdx::detail::complex<float>;
    using cfloat_t = typename c10::complex<float>;
    constexpr int N = cufftdx::size_of<FFT>::value;
    constexpr int EPT = FFT::elements_per_thread;
    static_assert(FFT::storage_size == EPT);
    static_assert(IFFT::storage_size == EPT);

    using BlockLoad_input = cub::BlockLoad<cfloat_t, FFT::block_dim.x, EPT / 2, cub::BLOCK_LOAD_STRIPED>;
    using BlockLoad_filter = cub::BlockLoad<cfloat_t, FFT::block_dim.x, EPT, cub::BLOCK_LOAD_STRIPED>;
    using BlockLoad_dout = cub::BlockLoad<cfloat_t, FFT::block_dim.x, EPT / 2, cub::BLOCK_LOAD_STRIPED>;
    using BlockStore_dinput = cub::BlockStore<c10::complex<input_t>, FFT::block_dim.x, EPT / 2, cub::BLOCK_STORE_STRIPED>;
    using BlockStore_dv = cub::BlockStore<cfloat_t, FFT::block_dim.x, EPT / 2, cub::BLOCK_STORE_STRIPED>;
    using BlockStore_dfilter = cub::BlockStore<cfloat_t, FFT::block_dim.x, EPT, cub::BLOCK_STORE_STRIPED>;

    extern __shared__ cfloat_t shared_mem[];

    float du_data[EPT] = { 0 };
    float dq_data[EPT] = { 0 };
    cfloat_t dfilter_data[EPT] = { 0 };
    float dD_val = 0.f;
                                    
    // #pragma unroll
    // for ( int i = 0; i < EPT; i++ ) {
    //     dfilter_data[i] = cfloat_t(0, 0);
    // }

    // Local array and copy data into it
    float u_og_data_before_gelu[EPT];
    float u_og_data[EPT];
    float q_data[EPT];

    // Id for inputData and inputMulQData
    size_t u_offset = blockIdx.x * batch_stride + (blockIdx.y * HEADDIM + blockIdx.z) * H_stride;
    BlockLoad_input().Load(reinterpret_cast<const c10::complex<input_t> *>(inputData + u_offset),
                           reinterpret_cast<cfloat_t (&)[EPT / 2]>(u_og_data_before_gelu),
                           signal_size / 2, cfloat_t(0.f));
    // TODO: what if signal_size is odd
    if (GELU_INPUT) {
        gelu(u_og_data, u_og_data_before_gelu);
    } else {
        #pragma unroll
        for ( int i = 0; i < EPT; i++ ) { u_og_data[i] = u_og_data_before_gelu[i]; }
    }

    cfloat_t filter_data[EPT];

    unsigned int filter_id = blockIdx.y;
    BlockLoad_filter().Load(filterData + filter_id * (N + 1), filter_data);
    if (threadIdx.x == 0) {
        filter_data[0].imag_ = *(reinterpret_cast<const float *>(filterData + filter_id * (N + 1) + N));
    }
    #pragma unroll
    for ( int i = 0; i < EPT; i++ ) { filter_data[i] /= 2 * N; }

    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     for (int i = 0; i < FFT::storage_size / 2; i++) {
    //         printf("%.4f+%.4fi, ", filter_data[i].real_, filter_data[i].imag_);
    //     }
    //     printf("\n");
    // }

    float D_val = DData[filter_id];
    unsigned int dropmask_id = blockIdx.x * H + blockIdx.y;
    float dropmask_val = dropmaskData == nullptr ? 1.f : dropmaskData[dropmask_id];

    if (QV) {
        // Will need to change this if head_dim is not 1
        BlockLoad_input().Load(reinterpret_cast<const c10::complex<input_t> *>(inputMulQData + u_offset),
                               reinterpret_cast<cfloat_t (&)[EPT / 2]>(q_data),
                               signal_size / 2, cfloat_t(0.f));

        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
        //     printf("q_data: ");
        //     for (int i = 0; i < EPT; i++) {
        //         printf("%.4f, ", q_data[i]);
        //     }
        //     printf("\n");
        // }
    }

    // do not pragma unroll this!!
    for (int head_i = 0; head_i < HEADDIM; head_i++) {
        float k_data[EPT];
        float v_data[EPT];
        cfloat_t thread_data[EPT];
        float grad_data[EPT];

        #pragma unroll
        for (int i = 0; i < EPT; ++i) { k_data[i] = u_og_data[i]; }

        size_t v_offset = blockIdx.x * batch_stride + (blockIdx.y * HEADDIM + head_i) * H_stride;
        if (QV) {
            BlockLoad_input().Load(reinterpret_cast<const c10::complex<input_t> *>(inputMulVData + v_offset),
                                   reinterpret_cast<cfloat_t (&)[EPT / 2]>(v_data),
                                   signal_size / 2, cfloat_t(0.f));

            // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
            //     printf("v_data: ");
            //     for (int i = 0; i < EPT; i++) {
            //         printf("%.4f, ", v_data[i]);
            //     }
            //     printf("\n");
            // }
            
            #pragma unroll
            for (int i = 0; i < EPT; ++i) {
                k_data[i] *= v_data[i];
            }
        }

        #pragma unroll
        for (int i = 0; i < EPT; ++i) {
            thread_data[i] = i < EPT / 2 ? cfloat_t(k_data[i * 2], k_data[i * 2 + 1]) : cfloat_t(0.f);
        }

        __syncthreads();
        // Execute FFT
        rfft<FFT>(thread_data, shared_mem);

        cfloat_t u_f[EPT];
        #pragma unroll
        for ( int i = 0; i < EPT; i++ ) { u_f[i] = thread_data[i]; }

        #pragma unroll
        for ( int i = 0; i < EPT; i++ ) {
            thread_data[i] = (threadIdx.x == 0) && (i == 0) ?
                pointwise_mul(thread_data[i], filter_data[i]) : thread_data[i] * filter_data[i];
        }

        // Execute FFT
        __syncthreads();
        irfft<IFFT>(thread_data, shared_mem);


        float out_data[EPT] {};

        #pragma unroll
        for ( int i = 0; i < EPT; i++ ) {
            out_data[i] = reinterpret_cast<float (&)[EPT * 2]>(thread_data)[i] + k_data[i] * D_val;
        }

        unsigned int output_fft_id = !output_hbl_layout ? blockIdx.x * H + blockIdx.y * HEADDIM + head_i : blockIdx.x + (blockIdx.y * HEADDIM + head_i) * batch_size;
        BlockLoad_dout().Load(reinterpret_cast<const c10::complex<output_t> *>(doutData + output_fft_id * signal_size),
                              reinterpret_cast<cfloat_t (&)[EPT / 2]>(grad_data),
                              signal_size / 2, cfloat_t(0.f));

        float out_data_before_gelu[EPT];
        #pragma unroll
        for (int i = 0; i < EPT; ++i) { out_data_before_gelu[i] = out_data[i]; };
        if (GELU_OUTPUT) { gelu(out_data, out_data); }
        #pragma unroll
        for ( int i = 0; i < EPT; i++ ) {
            out_data[i] *= dropmask_val;
        }

        // dQ
        if (QV) {
            #pragma unroll
            for (int i = 0; i < EPT; ++i) {
                if (GELU_Q) {
                    constexpr float kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
                    constexpr float kAlpha = M_SQRT1_2;
                    const float cdf = 0.5 * (1 + erff(q_data[i] * kAlpha));
                    const float pdf = expf(-0.5 * q_data[i] * q_data[i]) * kBeta;
                    dq_data[i] += (cdf + q_data[i] * pdf) * grad_data[i] * out_data[i];
                    grad_data[i] *= q_data[i] * cdf;
                } else {
                    dq_data[i] += grad_data[i] * out_data[i];
                    grad_data[i] *= q_data[i];
                }
            }
        }

        // dGELU and dropout
        // https://github.com/pytorch/pytorch/blob/dc169d53aa266560750ea25ee0cf31c7e614550d/aten/src/ATen/native/cuda/Activation.cu#L418
        #pragma unroll
        for ( int i = 0; i < EPT; ++i) { grad_data[i] *= dropmask_val; }
        if (GELU_OUTPUT) { dgelu(grad_data, grad_data, out_data_before_gelu); }

        // CHANGE THIS!!!
        #pragma unroll
        for ( int i = 0; i < EPT; i++ ) {
            dD_val += grad_data[i] * k_data[i];
        }

        cfloat_t grad_data_c[EPT];
        #pragma unroll
        for ( int i = 0; i < EPT; i++ ) {
            grad_data_c[i] = i < EPT / 2 ? cfloat_t(grad_data[i * 2], grad_data[i * 2 + 1]) : cfloat_t(0.f);
        }

        __syncthreads();
        rfft<FFT>(grad_data_c, shared_mem);

        // CHANGE THIS!!!
        #pragma unroll
        for ( int i = 0; i < EPT; i++ ) {
            dfilter_data[i] += ((threadIdx.x == 0) && (i == 0) ?
                            pointwise_mul(grad_data_c[i], u_f[i]) : grad_data_c[i] * std::conj(u_f[i])) / (2 * N);
        }

        #pragma unroll
        for ( int i = 0; i < EPT; i++ ) {
            grad_data_c[i] = (threadIdx.x == 0) && (i == 0) ?
                pointwise_mul(grad_data_c[i], filter_data[i]) : grad_data_c[i] * std::conj(filter_data[i]);
        }

        __syncthreads();
        irfft<IFFT>(grad_data_c, shared_mem);

        float du_data_local[EPT];
        #pragma unroll
        for ( int i = 0; i < EPT; i++ ) {
            du_data_local[i] = reinterpret_cast<float (&)[EPT * 2]>(grad_data_c)[i] + grad_data[i] * D_val;
        }

        float dv_data[EPT];
        // compute dv, and update du
        if (QV) {
            #pragma unroll
            for ( int i = 0; i < EPT; i++ ) {
                // dv
                constexpr float kAlpha = M_SQRT1_2;
                dv_data[i] = du_data_local[i] * (GELU_INPUT ? (u_og_data_before_gelu[i] * 0.5 * (1 + erff(u_og_data_before_gelu[i] * kAlpha))) : u_og_data_before_gelu[i]);

                // update du
                du_data_local[i] = du_data_local[i] * v_data[i];
                if (GELU_INPUT) {
                    constexpr float kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
                    const float cdf = 0.5 * (1 + erff(u_og_data_before_gelu[i] * kAlpha));
                    const float pdf = expf(-0.5 * u_og_data_before_gelu[i] * u_og_data_before_gelu[i]) * kBeta;
                    du_data_local[i] = (cdf + u_og_data_before_gelu[i] * pdf) * du_data_local[i];
                }
            }
        }

        #pragma unroll
        for ( int i = 0; i < EPT; i++ ) {
            du_data[i] += du_data_local[i];
        }        
        
        // store dv using atomic add
        if (QV) {
            unsigned int dv_data_idx;
            unsigned int thread_id = threadIdx.x;
            #pragma unroll
            for (int i = 0; i < EPT / 2; ++i) {
                // compute index based on thread idx, i, and head_i
                dv_data_idx = FFT::block_dim.x * i + thread_id;
                if (dv_data_idx < signal_size / 2) {
                    // add the real and imaginary parts separately
                    cfloat_t *loc = &reinterpret_cast<cfloat_t *>(dvData + v_offset)[dv_data_idx];
                    atomicAdd(reinterpret_cast<float *>(loc), dv_data[i * 2]);
                    atomicAdd(reinterpret_cast<float *>(loc) + 1, dv_data[i * 2 + 1]);
                }
            }
        }
        // TODO: what if signal_size is odd?
    }

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    unsigned int dfilter_id = blockIdx.x * H + blockIdx.y * HEADDIM + blockIdx.z;

    // There may be something wrong here??
    // Save dD
    using BlockReduceT = cub::BlockReduce<float, FFT::block_dim.x>;
    using TempStorageT = typename BlockReduceT::TempStorage;
    __syncthreads();
    dD_val = BlockReduceT(reinterpret_cast<TempStorageT&>(shared_mem)).Sum(dD_val);
    if (threadIdx.x == 0) { *(dDData + dfilter_id) = dD_val; }

    // Save dfilter
    float dfilter_extra = 0.f;
    if (threadIdx.x == 0) {
        dfilter_extra = dfilter_data[0].imag_;
        dfilter_data[0].imag_ = 0.f;
    }

    BlockStore_dfilter().Store(dfilterData + dfilter_id * (N + 1), dfilter_data);
    if (threadIdx.x == 0) {
        *(dfilterData + dfilter_id * (N + 1) + N) = cfloat_t(dfilter_extra, 0.f);
    }

    // Save results
    c10::complex<input_t> du_data_c[EPT / 2];
    #pragma unroll
    for (int i = 0; i < EPT / 2; ++i) {
        du_data_c[i] = c10::complex(input_t(du_data[i * 2]), input_t(du_data[i * 2 + 1]));
    }
    BlockStore_dinput().Store(reinterpret_cast<c10::complex<input_t> *>(duData + u_offset),
                              du_data_c, signal_size / 2);
    if (QV) {
        c10::complex<input_t> dq_data_c[EPT / 2];
        #pragma unroll
        for (int i = 0; i < EPT / 2; ++i) {
            dq_data_c[i] = c10::complex(input_t(dq_data[i * 2]), input_t(dq_data[i * 2 + 1]));
        }

        BlockStore_dinput().Store(reinterpret_cast<c10::complex<input_t> *>(dqData + u_offset),  // check this pointer arithmetic
                                  dq_data_c, signal_size / 2);
    }
}

// template<bool QV, int HEADDIM, typename FFT, typename IFFT, typename input_t, typename output_t=input_t, bool GELU_INPUT=false, bool GELU_OUTPUT=true, bool GELU_Q=false>
// __launch_bounds__( FFT::max_threads_per_block )
// __global__ void fftconv_bwd_kernelfp16(const output_t *__restrict__ doutData,
//                                    const input_t *__restrict__ inputData,
//                                    const c10::complex<float> *__restrict__ filterData,
//                                    const input_t *__restrict__ inputMulVData,
//                                    const input_t *__restrict__ inputMulQData,
//                                    const float *__restrict__ DData,
//                                    const float *__restrict__ dropmaskData,
//                                    input_t *__restrict__ duData,
//                                    c10::complex<float> *__restrict__ dfilterData,
//                                    float *__restrict__ dDData,
//                                    float *__restrict__ dvData,
//                                    input_t *__restrict__ dqData,
//                                    int batch_size,
//                                    int H,
//                                    int signal_size,
//                                    bool output_hbl_layout) {

//     using complex_t = typename cufftdx::detail::complex<__half2>;
//     using cfloat_t = typename c10::complex<float>;
//     constexpr int N = cufftdx::size_of<FFT>::value;
//     constexpr int EPT = FFT::elements_per_thread;
//     static_assert(FFT::storage_size == EPT);
//     static_assert(IFFT::storage_size == EPT);

//     using BlockLoad_input = cub::BlockLoad<cfloat_t, FFT::block_dim.x, EPT / 2, cub::BLOCK_LOAD_STRIPED>;
//     using BlockLoad_filter = cub::BlockLoad<cfloat_t, FFT::block_dim.x, EPT, cub::BLOCK_LOAD_STRIPED>;
//     using BlockLoad_dout = cub::BlockLoad<cfloat_t, FFT::block_dim.x, EPT / 2, cub::BLOCK_LOAD_STRIPED>;
//     using BlockStore_dinput = cub::BlockStore<c10::complex<input_t>, FFT::block_dim.x, EPT / 2, cub::BLOCK_STORE_STRIPED>;
//     using BlockStore_dv = cub::BlockStore<cfloat_t, FFT::block_dim.x, EPT / 2, cub::BLOCK_STORE_STRIPED>;
//     using BlockStore_dfilter = cub::BlockStore<cfloat_t, FFT::block_dim.x, EPT, cub::BLOCK_STORE_STRIPED>;

//     extern __shared__ cfloat_t shared_mem[];

//     float du_data[2][EPT] = { 0 };
//     float dq_data[2][EPT] = { 0 };
//     cfloat_t dfilter_data[2][EPT] = { 0 };
//     float dD_val [2] = { 0.f, 0.f };

//     // Local array and copy data into it
//     float u_og_data_before_gelu[2][EPT];
//     float u_og_data[2][EPT];
//     float q_data[2][EPT];

//     // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
//     unsigned int global_fft_id = blockIdx.x * H * FFT::ffts_per_block + blockIdx.y * HEADDIM + blockIdx.z;

//     BlockLoad_input().Load(reinterpret_cast<const c10::complex<input_t> *>(inputData + global_fft_id * signal_size),
//                            reinterpret_cast<cfloat_t (&)[EPT / 2]>(u_og_data_before_gelu[0]),
//                            signal_size / 2, cfloat_t(0.f));
//     BlockLoad_input().Load(reinterpret_cast<const c10::complex<input_t> *>(inputData + global_fft_id * signal_size + H * signal_size),
//                            reinterpret_cast<cfloat_t (&)[EPT / 2]>(u_og_data_before_gelu[1]),
//                            signal_size / 2, cfloat_t(0.f));
//     // TODO: what if signal_size is odd
//     if (GELU_INPUT) {
//         gelu(u_og_data[0], u_og_data_before_gelu[0]);
//         gelu(u_og_data[1], u_og_data_before_gelu[1]);
//     } else {
//         #pragma unroll
//         for ( int i = 0; i < EPT; i++ ) {
//             u_og_data[0][i] = u_og_data_before_gelu[0][i];
//             u_og_data[1][i] = u_og_data_before_gelu[1][i];
//         }
//     }

//     cfloat_t filter_data[EPT];

//     unsigned int filter_id = blockIdx.y;
//     BlockLoad_filter().Load(filterData + filter_id * (N + 1), filter_data);
//     if (threadIdx.x == 0) {
//         filter_data[0].imag_ = *(reinterpret_cast<const float *>(filterData + filter_id * (N + 1) + N));
//     }
//     #pragma unroll
//     for ( int i = 0; i < EPT; i++ ) { filter_data[i] /= 2 * N; }

//     // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
//     //     for (int i = 0; i < FFT::storage_size / 2; i++) {
//     //         printf("%.4f+%.4fi, ", filter_data[i].real_, filter_data[i].imag_);
//     //     }
//     //     printf("\n");
//     // }

//     float D_val = DData[filter_id];
//     unsigned int dropmask_id = blockIdx.x * H + blockIdx.y;
//     float dropmask_val = dropmaskData == nullptr ? 1.f : dropmaskData[dropmask_id];

//     if (QV) {
//         // Will need to change this if head_dim is not 1
//         BlockLoad_input().Load(reinterpret_cast<const c10::complex<input_t> *>(inputMulQData + global_fft_id * signal_size),
//                                reinterpret_cast<cfloat_t (&)[EPT / 2]>(q_data[0]),
//                                signal_size / 2, cfloat_t(0.f));
//         BlockLoad_input().Load(reinterpret_cast<const c10::complex<input_t> *>(inputMulQData + global_fft_id * signal_size + H * signal_size),
//                                reinterpret_cast<cfloat_t (&)[EPT / 2]>(q_data[1]),
//                                signal_size / 2, cfloat_t(0.f));

//         // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
//         //     printf("q_data: ");
//         //     for (int i = 0; i < EPT; i++) {
//         //         printf("%.4f, ", q_data[i]);
//         //     }
//         //     printf("\n");
//         // }
//     }

//     // do not pragma unroll this!!
//     for (int head_i = 0; head_i < HEADDIM; head_i++) {
//         float k_data[2][EPT];
//         float v_data[2][EPT];
//         complex_t thread_data[EPT];
//         float grad_data[2][EPT];

//         #pragma unroll
//         for (int i = 0; i < EPT; ++i) {
//             k_data[0][i] = u_og_data[0][i];
//             k_data[1][i] = u_og_data[1][i];
//         }

//         unsigned int head_fft_id = blockIdx.x * H * FFT::ffts_per_block + blockIdx.y * HEADDIM + head_i;

//         if (QV) {
//             BlockLoad_input().Load(reinterpret_cast<const c10::complex<input_t> *>(inputMulVData + head_fft_id * signal_size),
//                                    reinterpret_cast<cfloat_t (&)[EPT / 2]>(v_data[0]),
//                                    signal_size / 2, cfloat_t(0.f));
//             BlockLoad_input().Load(reinterpret_cast<const c10::complex<input_t> *>(inputMulVData + head_fft_id * signal_size + H * signal_size),
//                                    reinterpret_cast<cfloat_t (&)[EPT / 2]>(v_data[1]),
//                                    signal_size / 2, cfloat_t(0.f));

//             // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
//             //     printf("v_data: ");
//             //     for (int i = 0; i < EPT; i++) {
//             //         printf("%.4f, ", v_data[i]);
//             //     }
//             //     printf("\n");
//             // }

//             #pragma unroll
//             for (int i = 0; i < EPT; ++i) {
//                 k_data[0][i] *= v_data[0][i];
//                 k_data[1][i] *= v_data[1][i];
//             }
//         }

//         #pragma unroll
//         for (int i = 0; i < EPT; ++i) {
//             thread_data[i] = i < EPT / 2 ? complex_t {
//                 __float22half2_rn({k_data[0][i * 2], k_data[1][i * 2]}),
//                 __float22half2_rn({k_data[0][i * 2 + 1], k_data[1][i * 2 + 1]})
//             } : complex_t { __float22half2_rn({0.f, 0.f}), __float22half2_rn({0.f, 0.f}) };
//         }

//         if (head_i > 0) { __syncthreads(); }
//         // Execute FFT
//         rfftfp16<FFT>(thread_data, reinterpret_cast<cufftdx::detail::complex<__half2> *>(shared_mem));

//         cfloat_t u_f[2][EPT];
//         cfloat_t thread_floats[2];
//         #pragma unroll
//         for ( int i = 0; i < EPT; i++ ) {
//             read_rrii(thread_data[i], thread_floats);
//             u_f[0][i] = thread_floats[0];
//             u_f[1][i] = thread_floats[1];
//         }

//         cfloat_t res [2];
//         #pragma unroll
//         for ( int i = 0; i < EPT; i++ ) {
//             read_rrii(thread_data[i], thread_floats);
//             for ( int j = 0; j < 2; j++ ) {
//                 res[j] = (threadIdx.x == 0) && (i == 0) ?
//                 pointwise_mul(thread_floats[j], filter_data[i]) : thread_floats[j] * filter_data[i];
//             }
//             thread_data[i] = write_rrii(res);
//             // thread_data[i] = (threadIdx.x == 0) && (i == 0) ?
//             //     pointwise_mul(thread_data[i], filter_data[i]) : thread_data[i] * filter_data[i];
//         }

//         // Execute FFT
//         __syncthreads();
//         irfftfp16<IFFT>(thread_data, reinterpret_cast<cufftdx::detail::complex<__half2> *>(shared_mem));


//         float out_data[2][EPT] {};

//         #pragma unroll
//         for ( int i = 0; i < EPT; i++ ) {
//             out_data[0][i] = __half2float(reinterpret_cast<__half2 (&)[EPT * 2]>(thread_data)[i].x) + k_data[0][i] * D_val;
//             out_data[1][i] = __half2float(reinterpret_cast<__half2 (&)[EPT * 2]>(thread_data)[i].y) + k_data[1][i] * D_val;
//         }

//         unsigned int output_fft_id = !output_hbl_layout ? blockIdx.x * H * FFT::ffts_per_block + blockIdx.y * HEADDIM + head_i : blockIdx.x * FFT::ffts_per_block + (blockIdx.y * HEADDIM + head_i) * batch_size;
//         BlockLoad_dout().Load(reinterpret_cast<const c10::complex<output_t> *>(doutData + output_fft_id * signal_size),
//                               reinterpret_cast<cfloat_t (&)[EPT / 2]>(grad_data[0]),
//                               signal_size / 2, cfloat_t(0.f));
//         BlockLoad_dout().Load(reinterpret_cast<const c10::complex<output_t> *>(doutData + output_fft_id * signal_size + (!output_hbl_layout ? H * signal_size : signal_size)),
//                               reinterpret_cast<cfloat_t (&)[EPT / 2]>(grad_data[1]),
//                               signal_size / 2, cfloat_t(0.f));

//         float out_data_before_gelu[2][EPT];
//         #pragma unroll
//         for (int i = 0; i < EPT; ++i) {
//             out_data_before_gelu[0][i] = out_data[0][i];
//             out_data_before_gelu[1][i] = out_data[1][i];
//         };
//         if (GELU_OUTPUT) {
//             gelu(out_data[0], out_data[0]);
//             gelu(out_data[1], out_data[1]);
//         }
//         #pragma unroll
//         for ( int i = 0; i < EPT; i++ ) {
//             out_data[0][i] *= dropmask_val;
//             out_data[1][i] *= dropmask_val;
//         }

//         // dQ
//         if (QV) {
//             #pragma unroll
//             for (int i = 0; i < EPT; ++i) {
//                 for (int j = 0; j < 2; ++j) {
//                     if (GELU_Q) {
//                         constexpr float kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
//                         constexpr float kAlpha = M_SQRT1_2;
//                         const float cdf = 0.5 * (1 + erff(q_data[j][i] * kAlpha));
//                         const float pdf = expf(-0.5 * q_data[j][i] * q_data[j][i]) * kBeta;
//                         dq_data[j][i] += (cdf + q_data[j][i] * pdf) * grad_data[j][i] * out_data[j][i];
//                         grad_data[j][i] *= q_data[j][i] * cdf;
//                     } else {
//                         dq_data[j][i] += grad_data[j][i] * out_data[j][i];
//                         grad_data[j][i] *= q_data[j][i];
//                     }
//                 }
//             }
//         }

//         // dGELU and dropout
//         // https://github.com/pytorch/pytorch/blob/dc169d53aa266560750ea25ee0cf31c7e614550d/aten/src/ATen/native/cuda/Activation.cu#L418
//         #pragma unroll
//         for ( int i = 0; i < EPT; ++i) {
//             grad_data[0][i] *= dropmask_val;
//             grad_data[1][i] *= dropmask_val;
//         }
//         if (GELU_OUTPUT) {
//             dgelu(grad_data[0], grad_data[0], out_data_before_gelu[0]);
//             dgelu(grad_data[1], grad_data[1], out_data_before_gelu[1]);
//         }

//         // CHANGE THIS!!!
//         #pragma unroll
//         for ( int i = 0; i < EPT; i++ ) {
//             dD_val[0] += grad_data[0][i] * k_data[0][i];
//             dD_val[1] += grad_data[1][i] * k_data[1][i];
//         }

//         complex_t grad_data_c[EPT];
//         #pragma unroll
//         for ( int i = 0; i < EPT; i++ ) {
//             // grad_data_c[i] = i < EPT / 2 ? cfloat_t(grad_data[i * 2], grad_data[i * 2 + 1]) : cfloat_t(0.f);
//             grad_data_c[i] = i < EPT / 2 ? complex_t {
//                 __float22half2_rn({grad_data[0][i * 2], grad_data[1][i * 2]}),
//                 __float22half2_rn({grad_data[0][i * 2 + 1], grad_data[1][i * 2 + 1]})
//             } : complex_t { __float22half2_rn({0.f, 0.f}), __float22half2_rn({0.f, 0.f}) };
//         }

//         __syncthreads();
//         rfftfp16<FFT>(grad_data_c, reinterpret_cast<cufftdx::detail::complex<__half2> *>(shared_mem));

//         cfloat_t grad_floats [2];
//         #pragma unroll
//         for ( int i = 0; i < EPT; i++ ) {
//             read_rrii(grad_data_c[i], grad_floats);
//             for ( int j = 0; j < 2; j++ ) {
//                 dfilter_data[j][i] += ((threadIdx.x == 0) && (i == 0) ?
//                     pointwise_mul(grad_floats[j], u_f[j][i]) : grad_floats[j] * std::conj(u_f[j][i])) / (2 * N);
//             }
//             // dfilter_data[i] += ((threadIdx.x == 0) && (i == 0) ?
//             //                 pointwise_mul(grad_data_c[i], u_f[i]) : grad_data_c[i] * std::conj(u_f[i])) / (2 * N);
//         }

//         #pragma unroll
//         for ( int i = 0; i < EPT; i++ ) {
//             read_rrii(grad_data_c[i], grad_floats);
//             for ( int j = 0; j < 2; j++ ) {
//                 res[j] = (threadIdx.x == 0) && (i == 0) ?
//                 pointwise_mul(grad_floats[j], filter_data[i]) : grad_floats[j] * std::conj(filter_data[i]);
//             }
//             grad_data_c[i] = write_rrii(res);
//             // grad_data_c[i] = (threadIdx.x == 0) && (i == 0) ?
//             //     pointwise_mul(grad_data_c[i], filter_data[i]) : grad_data_c[i] * std::conj(filter_data[i]);
//         }

//         __syncthreads();
//         irfftfp16<IFFT>(grad_data_c, reinterpret_cast<cufftdx::detail::complex<__half2> *>(shared_mem));

//         float du_data_local[2][EPT];
//         #pragma unroll
//         for ( int i = 0; i < EPT; i++ ) {
//             // du_data_local[i] = reinterpret_cast<float (&)[EPT * 2]>(grad_data_c)[i] + grad_data[i] * D_val;
//             du_data_local[0][i] = __half2float(reinterpret_cast<__half2 (&)[EPT * 2]>(grad_data_c)[i].x) + grad_data[0][i] * D_val;
//             du_data_local[1][i] = __half2float(reinterpret_cast<__half2 (&)[EPT * 2]>(grad_data_c)[i].y) + grad_data[1][i] * D_val;
//         }

//         float dv_data[2][EPT];
//         // compute dv, and update du
//         if (QV) {
//             #pragma unroll
//             for ( int i = 0; i < EPT; i++ ) {
//                 // dv
//                 dv_data[0][i] = du_data_local[0][i] * u_og_data_before_gelu[0][i];
//                 dv_data[1][i] = du_data_local[1][i] * u_og_data_before_gelu[1][i];

//                 // update du
//                 du_data_local[0][i] = du_data_local[0][i] * v_data[0][i];
//                 du_data_local[1][i] = du_data_local[1][i] * v_data[1][i];
//             }
//         }

//         // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) && (head_i == 0)) {
//         //     printf("u_og_data_before_gelu[0]: ");
//         //     for (int i = 0; i < EPT; i++) {
//         //         printf("%.4f, ", u_og_data_before_gelu[0][i]);
//         //     }
//         //     printf("\n");
//         //     printf("u_og_data_before_gelu[1]: ");
//         //     for (int i = 0; i < EPT; i++) {
//         //         printf("%.4f, ", u_og_data_before_gelu[1][i]);
//         //     }
//         //     printf("\n");
//         // }
//         // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) && (head_i == 0)) {
//         //     printf("dv_data[0]: ");
//         //     for (int i = 0; i < EPT; i++) {
//         //         printf("%.4f, ", dv_data[0][i]);
//         //     }
//         //     printf("\n");
//         //     printf("dv_data[1]: ");
//         //     for (int i = 0; i < EPT; i++) {
//         //         printf("%.4f, ", dv_data[1][i]);
//         //     }
//         //     printf("\n");
//         // }

//         #pragma unroll
//         for ( int i = 0; i < EPT; i++ ) {
//             du_data[0][i] += du_data_local[0][i];
//             du_data[1][i] += du_data_local[1][i];
//         }

//         // store dv using atomic add
//         if (QV) {
//             unsigned int dv_data_idx;
//             unsigned int thread_id = threadIdx.x;
//             #pragma unroll
//             for (int i = 0; i < EPT / 2; ++i) {
//                 // compute index based on thread idx, i, and head_i
//                 dv_data_idx = FFT::block_dim.x * i + thread_id;
//                 if (dv_data_idx < signal_size / 2) {
//                     // add the real and imaginary parts separately
//                     cfloat_t *loc = &reinterpret_cast<cfloat_t *>(dvData + head_fft_id * signal_size)[dv_data_idx];
//                     atomicAdd(reinterpret_cast<float *>(loc), dv_data[0][i * 2]);
//                     atomicAdd(reinterpret_cast<float *>(loc) + 1, dv_data[0][i * 2 + 1]);

//                     // add the real and imaginary parts separately
//                     loc = &reinterpret_cast<cfloat_t *>(dvData + head_fft_id * signal_size + H * signal_size)[dv_data_idx];
//                     atomicAdd(reinterpret_cast<float *>(loc), dv_data[1][i * 2]);
//                     atomicAdd(reinterpret_cast<float *>(loc) + 1, dv_data[1][i * 2 + 1]);
//                 }
//             }
//         }
//         // TODO: what if signal_size is odd?
//     }

//     unsigned int dfilter_id = global_fft_id;

//     // There may be something wrong here??
//     // Save dD
//     using BlockReduceT = cub::BlockReduce<float, FFT::block_dim.x>;
//     using TempStorageT = typename BlockReduceT::TempStorage;
//     __syncthreads();
//     dD_val[0] = BlockReduceT(reinterpret_cast<TempStorageT&>(shared_mem)).Sum(dD_val[0]);
//     dD_val[1] = BlockReduceT(reinterpret_cast<TempStorageT&>(shared_mem)).Sum(dD_val[1]);
//     if (threadIdx.x == 0) {
//         *(dDData + dfilter_id) = dD_val[0];
//         *(dDData + dfilter_id + H) = dD_val[1];
//     }

//     // Save dfilter
//     float dfilter_extra [2] = { 0.f, 0.f };
//     if (threadIdx.x == 0) {
//         dfilter_extra[0] = dfilter_data[0][0].imag_;
//         dfilter_data[0][0].imag_ = 0.f;
//         dfilter_extra[1] = dfilter_data[1][0].imag_;
//         dfilter_data[1][0].imag_ = 0.f;
//     }

//     BlockStore_dfilter().Store(dfilterData + dfilter_id * (N + 1), dfilter_data[0]);
//     BlockStore_dfilter().Store(dfilterData + (dfilter_id + H) * (N + 1), dfilter_data[1]);
//     if (threadIdx.x == 0) {
//         *(dfilterData + dfilter_id * (N + 1) + N) = cfloat_t(dfilter_extra[0], 0.f);
//         *(dfilterData + (dfilter_id + H) * (N + 1) + N) = cfloat_t(dfilter_extra[1], 0.f);
//     }

//     // Save results
//     c10::complex<input_t> du_data_c[2][EPT / 2];
//     #pragma unroll
//     for (int i = 0; i < EPT / 2; ++i) {
//         du_data_c[0][i] = c10::complex(input_t(du_data[0][i * 2]), input_t(du_data[0][i * 2 + 1]));
//         du_data_c[1][i] = c10::complex(input_t(du_data[1][i * 2]), input_t(du_data[1][i * 2 + 1]));
//     }
//     BlockStore_dinput().Store(reinterpret_cast<c10::complex<input_t> *>(duData + global_fft_id * signal_size),
//                               du_data_c[0], signal_size / 2);
//     BlockStore_dinput().Store(reinterpret_cast<c10::complex<input_t> *>(duData + global_fft_id * signal_size + H * signal_size),
//                               du_data_c[1], signal_size / 2);
//     if (QV) {
//         c10::complex<input_t> dq_data_c[2][EPT / 2];
//         #pragma unroll
//         for (int i = 0; i < EPT / 2; ++i) {
//             dq_data_c[0][i] = c10::complex(input_t(dq_data[0][i * 2]), input_t(dq_data[0][i * 2 + 1]));
//             dq_data_c[1][i] = c10::complex(input_t(dq_data[1][i * 2]), input_t(dq_data[1][i * 2 + 1]));
//         }

//         BlockStore_dinput().Store(reinterpret_cast<c10::complex<input_t> *>(dqData + global_fft_id * signal_size),  // check this pointer arithmetic
//                               dq_data_c[0], signal_size / 2);
//         BlockStore_dinput().Store(reinterpret_cast<c10::complex<input_t> *>(dqData + global_fft_id * signal_size + H * signal_size),  // check this pointer arithmetic
//                               dq_data_c[1], signal_size / 2);
//     }
// }

template <bool GELU_OUTPUT, uint FFT_SIZE, uint EPT, typename input_t, typename output_t=input_t>
void fftconv_bwd_cuda(
    const output_t *dout, const input_t *u,
    const c10::complex<float> *filter,
    const input_t *v, int head_dim, const input_t *q,
    const float *D, const float *dropout_mask,
    input_t *du, c10::complex<float> *dfilter,
    float *dD,
    float *dv, input_t *dq,
    bool gelu, bool gelu_inp, bool gelu_q, int batch_size, int H, int signal_size,
    size_t batch_stride, size_t H_stride, bool output_hbl_layout, bool fftfp16
) {
#if defined(__CUDA_ARCH__)
    constexpr uint ARCH = __CUDA_ARCH__;
#else
    constexpr uint ARCH = 700;
#endif

    (void) gelu_inp;
    (void) gelu_q;

    switch (head_dim) {
        case 1:
        {   
            constexpr uint FPB = 1;

            // FFT is defined, its: size, type, direction, precision. Block() operator
            // informs that FFT will be executed on block level. Shared memory is
            // required for co-operation between threads.
            using FFT_base = decltype(cufftdx::Block() + cufftdx::Size<FFT_SIZE>() + cufftdx::Precision<float>() +
            cufftdx::ElementsPerThread<EPT>() + cufftdx::FFTsPerBlock<FPB>() + cufftdx::SM<ARCH>()
            + cufftdx::Type<cufftdx::fft_type::c2c>());

            using FFT = decltype(FFT_base() + cufftdx::Direction<cufftdx::fft_direction::forward>());
            using IFFT = decltype(FFT_base() + cufftdx::Direction<cufftdx::fft_direction::inverse>());

            // By default the shared memory size is 4 * FFT_SIZE (idk how).
            // So it wouldn't work for our rfft and irfft functions.
            const auto shared_memory_size = std::max({FFT::shared_memory_size, IFFT::shared_memory_size,
                8 * FFT_SIZE});
            // printf("shared_memory_size = %d\n", shared_memory_size);

            // unsigned int blocks_per_grid { static_cast<unsigned int>( std::ceil( batch_size / FPB ) ) };
            unsigned int H_per_grid { static_cast<unsigned int>( std::ceil( H / FPB ) ) };
            dim3 block(batch_size, H_per_grid / head_dim, head_dim);

            BOOL_SWITCH(v != nullptr, QV, [&] {
                auto kernel = &fftconv_bwd_kernel<QV, 1, FFT, IFFT, input_t, output_t, false, GELU_OUTPUT, false>;
                // Increase dynamic memory limit if required.
                CUDA_RT_CALL( cudaFuncSetAttribute(kernel,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   shared_memory_size ));
                kernel<<<block, FFT::block_dim, shared_memory_size>>>(
                    dout, u, filter, v, q, D, dropout_mask, du, dfilter, dD, dv, dq, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout);
            });
            break;
        }
        case 8:
        {
            if (fftfp16) {
                // constexpr uint FPB = 2;

                // using FFT_base = decltype(cufftdx::Block() + cufftdx::Size<FFT_SIZE>() + cufftdx::Precision<__half>() +
                //     cufftdx::ElementsPerThread<EPT>() + cufftdx::FFTsPerBlock<FPB>() + cufftdx::SM<ARCH>()
                //     + cufftdx::Type<cufftdx::fft_type::c2c>());

                // using FFT = decltype(FFT_base() + cufftdx::Direction<cufftdx::fft_direction::forward>());
                // using IFFT = decltype(FFT_base() + cufftdx::Direction<cufftdx::fft_direction::inverse>());

                // // By default the shared memory size is 4 * FFT_SIZE (idk how).
                // // So it wouldn't work for our rfft and irfft functions.
                // const auto shared_memory_size = std::max({FFT::shared_memory_size, IFFT::shared_memory_size,
                //     8 * FFT_SIZE});
                // // printf("shared_memory_size = %d\n", shared_memory_size);

                // unsigned int blocks_per_grid { static_cast<unsigned int>( std::ceil( batch_size / FPB ) ) };
                // // unsigned int H_per_grid { static_cast<unsigned int>( std::ceil( H / FPB ) ) };
                // dim3 block(blocks_per_grid, H / head_dim, head_dim);

                // constexpr bool QV = true;  // Multi-head requires QV
                // auto kernel = &fftconv_bwd_kernelfp16<QV, 8, FFT, IFFT, input_t, output_t, false, GELU_OUTPUT, false>;
                // // Increase dynamic memory limit if required.
                // CUDA_RT_CALL( cudaFuncSetAttribute(kernel,
                //                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                //                                 shared_memory_size ));
                // kernel<<<block, FFT::block_dim, shared_memory_size>>>(
                //     dout, u, filter, v, q, D, dropout_mask, du, dfilter, dD, dv, dq, batch_size, H, signal_size, output_hbl_layout);
            }
            else {
                // to go back to fp32
                constexpr uint FPB = 1;

                using FFT_base = decltype(cufftdx::Block() + cufftdx::Size<FFT_SIZE>() + cufftdx::Precision<float>() +
                    cufftdx::ElementsPerThread<EPT>() + cufftdx::FFTsPerBlock<FPB>() + cufftdx::SM<ARCH>()
                    + cufftdx::Type<cufftdx::fft_type::c2c>());

                using FFT = decltype(FFT_base() + cufftdx::Direction<cufftdx::fft_direction::forward>());
                using IFFT = decltype(FFT_base() + cufftdx::Direction<cufftdx::fft_direction::inverse>());

                // By default the shared memory size is 4 * FFT_SIZE (idk how).
                // So it wouldn't work for our rfft and irfft functions.
                const auto shared_memory_size = std::max({FFT::shared_memory_size, IFFT::shared_memory_size,
                    8 * FFT_SIZE});
                // printf("shared_memory_size = %d\n", shared_memory_size);

                unsigned int blocks_per_grid { static_cast<unsigned int>( std::ceil( batch_size / FPB ) ) };
                // unsigned int H_per_grid { static_cast<unsigned int>( std::ceil( H / FPB ) ) };
                dim3 block(blocks_per_grid, H / head_dim, head_dim);

                constexpr bool QV = true;  // Multi-head requires QV
                auto kernel = &fftconv_bwd_kernel<QV, 8, FFT, IFFT, input_t, output_t, false, GELU_OUTPUT, false>;
                // Increase dynamic memory limit if required.
                CUDA_RT_CALL( cudaFuncSetAttribute(kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                shared_memory_size ));
                kernel<<<block, FFT::block_dim, shared_memory_size>>>(
                    dout, u, filter, v, q, D, dropout_mask, du, dfilter, dD, dv, dq, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout);
            }
            break;
        }
        default:
            AT_ERROR("fftconv backward not implemented for this head_dim");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
};

template <typename input_t, typename output_t=input_t>
void fftconv_bwd_cuda_dispatch(
    const output_t *dout,
    const input_t *u, const c10::complex<float> *filter,
    const input_t *v, int head_dim, const input_t *q,
    const float *D, const float *dropout_mask,
    input_t *du, c10::complex<float> *dfilter, float *dD,
    float *dv, input_t *dq,
    bool gelu, bool gelu_inp, bool gelu_q, int batch_size, int H, int signal_size,
    size_t batch_stride, size_t H_stride, int fft_size,
    bool output_hbl_layout, bool fftfp16
) {
    BOOL_SWITCH(gelu, GELU_OUTPUT, [&] {
        switch(fft_size) {
            case 16:
                fftconv_bwd_cuda<GELU_OUTPUT, 8, 4, input_t, output_t>(
                    dout, u, filter, v, head_dim, q, D, dropout_mask,
                    du, dfilter, dD, dv, dq,
                    gelu, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            case 32:
                fftconv_bwd_cuda<GELU_OUTPUT, 16, 4, input_t, output_t>(
                    dout, u, filter, v, head_dim, q, D, dropout_mask,
                    du, dfilter, dD, dv, dq,
                    gelu, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            case 64:
                fftconv_bwd_cuda<GELU_OUTPUT, 32, 4, input_t, output_t>(
                    dout, u, filter, v, head_dim, q, D, dropout_mask,
                    du, dfilter, dD, dv, dq,
                    gelu, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            case 128:
                fftconv_bwd_cuda<GELU_OUTPUT, 64, 4, input_t, output_t>(
                    dout, u, filter, v, head_dim, q, D, dropout_mask,
                    du, dfilter, dD, dv, dq,
                    gelu, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            case 256:
                fftconv_bwd_cuda<GELU_OUTPUT, 128, 4, input_t, output_t>(
                    dout, u, filter, v, head_dim, q, D, dropout_mask,
                    du, dfilter, dD, dv, dq,
                    gelu, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            case 512:
                fftconv_bwd_cuda<GELU_OUTPUT, 256, 8, input_t, output_t>(
                    dout, u, filter, v, head_dim, q, D, dropout_mask,
                    du, dfilter, dD, dv, dq,
                    gelu, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            case 1024:
                fftconv_bwd_cuda<GELU_OUTPUT, 512, 16, input_t, output_t>(
                    dout, u, filter, v, head_dim, q, D, dropout_mask,
                    du, dfilter, dD, dv, dq,
                    gelu, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            case 2048:
                fftconv_bwd_cuda<GELU_OUTPUT, 1024, 16, input_t, output_t>(
                    dout, u, filter, v, head_dim, q, D, dropout_mask,
                    du, dfilter, dD, dv, dq,
                    gelu, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            case 4096:
                fftconv_bwd_cuda<GELU_OUTPUT, 2048, 8, input_t, output_t>(
                    dout, u, filter, v, head_dim, q, D, dropout_mask,
                    du, dfilter, dD, dv, dq,
                    gelu, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            case 8192:
                fftconv_bwd_cuda<GELU_OUTPUT, 4096, 8, input_t, output_t>(
                    dout, u, filter, v, head_dim, q, D, dropout_mask,
                    du, dfilter, dD, dv, dq,
                    gelu, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            case 16384:
                fftconv_bwd_cuda<GELU_OUTPUT, 8192, 8, input_t, output_t>(
                    dout, u, filter, v, head_dim, q, D, dropout_mask,
                    du, dfilter, dD, dv, dq,
                    gelu, gelu_inp, gelu_q, batch_size, H, signal_size, batch_stride, H_stride, output_hbl_layout, fftfp16);
                break;
            default:
                AT_ERROR("fftconv backward not implemented for this fft_size");
        }
    });
};

template void fftconv_fwd_cuda_dispatch<float, float>(
    const float *u, const c10::complex<float> *filter,
    const float *v, int head_dim, const float *q,
    const float *D, const float *dropout_mask, float *out,
    bool gelu, bool gelu_inp, bool gelu_q, int batch_size, int H, int signal_size,
    size_t batch_stride, size_t H_stride, int fft_size,
    bool output_hbl_layout, bool fftfp16);

template void fftconv_fwd_cuda_dispatch<float, at::Half>(
    const float *u, const c10::complex<float> *filter,
    const float *v, int head_dim, const float *q,
    const float *D, const float *dropout_mask, at::Half *out,
    bool gelu, bool gelu_inp, bool gelu_q, int batch_size, int H, int signal_size,
    size_t batch_stride, size_t H_stride, int fft_size,
    bool output_hbl_layout, bool fftfp16);

template void fftconv_fwd_cuda_dispatch<at::Half, at::Half>(
    const at::Half *u, const c10::complex<float> *filter,
    const at::Half *v, int head_dim, const at::Half *q,
    const float *D, const float *dropout_mask, at::Half *out,
    bool gelu, bool gelu_inp, bool gelu_q, int batch_size, int H, int signal_size,
    size_t batch_stride, size_t H_stride, int fft_size,
    bool output_hbl_layout, bool fftfp16);

template void fftconv_fwd_cuda_dispatch<at::BFloat16, at::BFloat16>(
    const at::BFloat16 *u, const c10::complex<float> *filter,
    const at::BFloat16 *v, int head_dim, const at::BFloat16 *q,
    const float *D, const float *dropout_mask, at::BFloat16 *out,
    bool gelu, bool gelu_inp, bool gelu_q, int batch_size, int H, int signal_size,
    size_t batch_stride, size_t H_stride, int fft_size,
    bool output_hbl_layout, bool fftfp16);

template void fftconv_bwd_cuda_dispatch<float, float>(
    const float *dout,
    const float *u, const c10::complex<float> *filter,
    const float *v, int head_dim, const float *q,
    const float *D, const float *dropout_mask,
    float *du, c10::complex<float> *dfilter, float *dD,
    float *dv, float *dq,
    bool gelu, bool gelu_inp, bool gelu_q, int batch_size, int H, int signal_size,
    size_t batch_stride, size_t H_stride, int fft_size,
    bool output_hbl_layout, bool fftfp16);

template void fftconv_bwd_cuda_dispatch<float, at::Half>(
    const at::Half *dout,
    const float *u, const c10::complex<float> *filter,
    const float *v, int head_dim, const float *q,
    const float *D, const float *dropout_mask,
    float *du, c10::complex<float> *dfilter, float *dD,
    float *dv, float *dq,
    bool gelu, bool gelu_inp, bool gelu_q, int batch_size, int H, int signal_size,
    size_t batch_stride, size_t H_stride, int fft_size,
    bool output_hbl_layout, bool fftfp16);

template void fftconv_bwd_cuda_dispatch<at::Half, at::Half>(
    const at::Half *dout,
    const at::Half *u, const c10::complex<float> *filter,
    const at::Half *v, int head_dim, const at::Half *q,
    const float *D, const float *dropout_mask,
    at::Half *du, c10::complex<float> *dfilter, float *dD,
    float *dv, at::Half *dq,
    bool gelu, bool gelu_inp, bool gelu_q, int batch_size, int H, int signal_size,
    size_t batch_stride, size_t H_stride, int fft_size,
    bool output_hbl_layout, bool fftfp16);

template void fftconv_bwd_cuda_dispatch<at::BFloat16, at::BFloat16>(
    const at::BFloat16 *dout,
    const at::BFloat16 *u, const c10::complex<float> *filter,
    const at::BFloat16 *v, int head_dim, const at::BFloat16 *q,
    const float *D, const float *dropout_mask,
    at::BFloat16 *du, c10::complex<float> *dfilter, float *dD,
    float *dv, at::BFloat16 *dq,
    bool gelu, bool gelu_inp, bool gelu_q, int batch_size, int H, int signal_size,
    size_t batch_stride, size_t H_stride, int fft_size,
    bool output_hbl_layout, bool fftfp16);