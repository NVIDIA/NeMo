#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "block_io.hpp"
#include "common.hpp"
#include "fp16_common.hpp"

template<class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void block_fft_kernel(__half2* data) {
    using complex_type = typename FFT::value_type;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Load data from global memory to registers
    example::io_fp16<FFT>::load(data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Save results
    example::io_fp16<FFT>::store(thread_data, data, local_fft_id);
}

// In this example a one-dimensional complex-to-complex transform is performed by a CUDA block.
//
// One block is run, and it calculates four 128-point C2C half precision FFTs.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
//
// Here, we're using __half2 as the type of the input/output data passed to kernel, and later on
// the device we use special example::io_fp16 struct template to load values from two batches
// into an array of complex<half2> with ((Real, Real), (Imag, Imag)) layout.
template<unsigned int Arch>
void simple_block_fft_half2() {
    using namespace cufftdx;

    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
    // will be executed on block level. Shared memory is required for co-operation between threads.
    // Additionally,
    using FFT = decltype(Block() + Size<128>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                         Precision<__half>() + ElementsPerThread<8>() + FFTsPerBlock<4>() + SM<Arch>());

    // Allocate managed memory for input/output
    __half2* data;
    auto     size       = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto     size_bytes = size * sizeof(__half2);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&data, size_bytes));
    for (size_t i = 0; i < size; i++) {
        data[i] = __half2 {float(i), -float(i)};
    }

    std::cout << "input [1st FFT]:\n";
    for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
        std::cout << __half2float(data[i].x) << " " << __half2float(data[i].y) << std::endl;
    }

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        block_fft_kernel<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        FFT::shared_memory_size));

    // Invokes kernel with FFT::block_dim threads in CUDA block
    block_fft_kernel<FFT><<<1, FFT::block_dim, FFT::shared_memory_size>>>(data);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::cout << "output [1st FFT]:\n";
    for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
        std::cout << __half2float(data[i].x) << " " << __half2float(data[i].y) << std::endl;
    }

    CUDA_CHECK_AND_EXIT(cudaFree(data));
    std::cout << "Success" << std::endl;
}

template<unsigned int Arch>
struct simple_block_fft_half2_functor {
    void operator()() { return simple_block_fft_half2<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<simple_block_fft_half2_functor>();
}
