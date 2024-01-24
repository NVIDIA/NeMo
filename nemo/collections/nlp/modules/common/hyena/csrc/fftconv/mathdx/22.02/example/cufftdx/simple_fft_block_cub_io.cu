#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include <cub/cub.cuh>
#include <cub/version.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>

#include "block_io.hpp"
#include "common.hpp"

#if CUB_VERSION < 101300
int main(int, char**) {
    std::cout << "Example disabled, BLOCK_LOAD_STRIPED/BLOCK_STORE_STRIPED is only supported since CUB 1.13 (CUDA 11.5)" << std::endl;
    return 0;
}
#else

template<class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void block_fft_kernel(typename FFT::value_type* data) {
    using complex_type = typename FFT::value_type;

    // FFT::stride == FFT::block_dim.x in most cases
    using BlockLoad  = cub::BlockLoad <complex_type, FFT::stride /* BLOCK_DIM_X */, FFT::storage_size, cub::BLOCK_LOAD_STRIPED>;
    using BlockStore = cub::BlockStore<complex_type, FFT::stride, FFT::storage_size, cub::BLOCK_STORE_STRIPED>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // ID of FFT in CUDA grid
    unsigned int global_fft_id =
        (FFT::ffts_per_block == 1) ? local_fft_id : ( blockIdx.x * FFT::ffts_per_block + local_fft_id);

    // Load data from global memory to registers
    auto fft_data = data + (global_fft_id * cufftdx::size_of<FFT>::value);
    BlockLoad().Load(fft_data, thread_data, cufftdx::size_of<FFT>::value, complex_type { 0.0, 0.0 });

    // Execute FFT
    extern __shared__ complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Save results
    BlockStore().Store(fft_data, thread_data, cufftdx::size_of<FFT>::value);
}

// In this example a one-dimensional complex-to-complex transform is performed by a CUDA block. CUB
// library is used for IO in kernel.
//
// One block is run, it calculates two 128-point C2C float precision FFTs.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
template<unsigned int Arch>
void simple_block_fft() {
    using namespace cufftdx;

    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
    // will be executed on block level. Shared memory is required for co-operation between threads.
    // Additionally,
    using FFT          = decltype(Block() + Size<128>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                         Precision<float>() + ElementsPerThread<8>() + FFTsPerBlock<2>() + SM<Arch>());
    using complex_type = typename FFT::value_type;

    // Allocate managed memory for input/output
    complex_type* data;
    auto          size       = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto          size_bytes = size * sizeof(complex_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&data, size_bytes));
    for (size_t i = 0; i < size; i++) {
        data[i] = complex_type {float(i), -float(i)};
    }

    std::cout << "input [1st FFT]:\n";
    for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
        std::cout << data[i].x << " " << data[i].y << std::endl;
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
        std::cout << data[i].x << " " << data[i].y << std::endl;
    }
    auto sum = data[0].x;

    CUDA_CHECK_AND_EXIT(cudaFree(data));
    if(std::abs(sum - ((cufftdx::size_of<FFT>::value-1) * cufftdx::size_of<FFT>::value / 2)) > 0.1) {
        std::cout << "Failed" << std::endl;
        return;
    }
    std::cout << "Success" << std::endl;
}

template<unsigned int Arch>
struct simple_block_fft_functor {
    void operator()() { return simple_block_fft<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<simple_block_fft_functor>();
}
#endif // CUB_VERSION < 101300
