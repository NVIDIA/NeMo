#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "block_io.hpp"
#include "common.hpp"

template<class FFTR2C, class FFTC2R>
__launch_bounds__(FFTR2C::max_threads_per_block) __global__ void convolution_kernel(cufftdx::precision_of_t<FFTR2C>* data) {
    using complex_type = typename FFTR2C::value_type;
    using scalar_type  = typename complex_type::value_type;

    // Local array for thread
    complex_type thread_data[FFTR2C::storage_size];

    // ID of FFT in CUDA block, in range [0; FFTR2C::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Load data from global memory to registers
    example::io<FFTR2C>::load_r2c(data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ complex_type shared_mem[];
    FFTR2C().execute(thread_data, shared_mem);

    // Scale values
    scalar_type scale = 1.0 / cufftdx::size_of<FFTR2C>::value;
    for (unsigned int i = 0; i < FFTR2C::elements_per_thread; i++) {
        thread_data[i].x *= scale;
        thread_data[i].y *= scale;
    }

    // Execute inverse FFT
    FFTC2R().execute(thread_data, shared_mem);

    // Save results
    example::io<FFTC2R>::store_c2r(thread_data, data, local_fft_id);
}

// This example demonstrates how to use cuFFTDx t operform a convolution using one-dimensional FFTs.
//
// One block is run, it calculates two 128-point convolutions by first doing forward FFT, then
// applying pointwise operation, and ending with inverse FFT.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
template<unsigned int Arch>
void convolution() {
    using namespace cufftdx;

    static constexpr unsigned int ffts_per_block = 2;
    static constexpr unsigned int fft_size       = 128;
    // FFT_base defined common options for FFT and IFFT. FFT_base is not a complete FFT description.
    // In order to complete FFT description directions are specified: forward for FFT, inverse for IFFT.
    using FFT_base  = decltype(Block() + Size<fft_size>() + Precision<float>() +
                              ElementsPerThread<8>() + FFTsPerBlock<ffts_per_block>() + SM<Arch>());
    using FFTR2C    = decltype(FFT_base() + Type<fft_type::r2c>());
    using FFTC2R    = decltype(FFT_base() + Type<fft_type::c2r>());
    using real_type = precision_of_t<FFTR2C>;

    // Allocate managed memory for input/output
    real_type* data;
    auto       size       = ffts_per_block * fft_size;
    auto       size_bytes = size * sizeof(real_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&data, size_bytes));
    for (size_t i = 0; i < size; i++) {
        data[i] = float(i);
    }

    std::cout << "input [1st FFT]:\n";
    for (size_t i = 0; i < fft_size; i++) {
        std::cout << data[i] << std::endl;
    }

    const auto shared_memory_size = std::max(FFTR2C::shared_memory_size, FFTC2R::shared_memory_size);
    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        convolution_kernel<FFTR2C, FFTC2R>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size));

    // Invokes convolution kernel with FFT::block_dim threads in CUDA block
    convolution_kernel<FFTR2C, FFTC2R><<<1, FFTR2C::block_dim, shared_memory_size>>>(data);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::cout << "output [1st FFT]:\n";
    for (size_t i = 0; i < fft_size; i++) {
        std::cout << data[i] << std::endl;
    }

    CUDA_CHECK_AND_EXIT(cudaFree(data));
    std::cout << "Success" << std::endl;
}

template<unsigned int Arch>
struct convolution_functor {
    void operator()() { return convolution<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<convolution_functor>();
}
