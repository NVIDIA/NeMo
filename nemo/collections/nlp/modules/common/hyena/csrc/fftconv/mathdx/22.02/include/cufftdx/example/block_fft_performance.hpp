#ifndef CUFFTDX_EXAMPLE_BLOCK_FFT_PERFORMANCE_HPP_
#define CUFFTDX_EXAMPLE_BLOCK_FFT_PERFORMANCE_HPP_

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "block_io.hpp"
#include "common.hpp"
#include "random.hpp"

template<class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void block_fft_kernel(typename FFT::value_type*    data,
                                                                               unsigned int                 repeats,
                                                                               typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;
    extern __shared__ complex_type shared_mem[];

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Load data from global memory to registers
    example::io<FFT>::load(data, thread_data, local_fft_id);

// Execute FFT
#pragma unroll 1
    for (unsigned int i = 0; i < repeats; i++) {
        FFT().execute(thread_data, shared_mem, workspace);
    }

    // Save results
    example::io<FFT>::store(thread_data, data, local_fft_id);
}

template<bool UseGraphs = false>
struct measure {
    // Returns execution time in ms
    template<unsigned int WarmUpRuns, typename Kernel>
    static float execution(Kernel&& kernel, cudaStream_t stream) {
        cudaEvent_t startEvent, stopEvent;
        CUDA_CHECK_AND_EXIT(cudaEventCreate(&startEvent));
        CUDA_CHECK_AND_EXIT(cudaEventCreate(&stopEvent));
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        for (size_t i = 0; i < WarmUpRuns; i++) {
            kernel();
        }

        CUDA_CHECK_AND_EXIT(cudaGetLastError());
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        CUDA_CHECK_AND_EXIT(cudaEventRecord(startEvent, stream));
        kernel();
        CUDA_CHECK_AND_EXIT(cudaEventRecord(stopEvent, stream));
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        float time;
        CUDA_CHECK_AND_EXIT(cudaEventElapsedTime(&time, startEvent, stopEvent));
        CUDA_CHECK_AND_EXIT(cudaEventDestroy(startEvent));
        CUDA_CHECK_AND_EXIT(cudaEventDestroy(stopEvent));
        return time;
    }
};

template<class FFTBase, unsigned int S /* Size */, unsigned int EPT, unsigned int FPB = 1, bool UseSuggested = false>
void benchmark_block_fft(const cudaStream_t& stream, bool verbose = false) {
    using namespace cufftdx;

    // Create complete FFT description, only now we can query EPT and suggested FFTs per block
    using FFT_complete = decltype(FFTBase() + Size<S>());

    static constexpr unsigned int inside_repeats = 4000;
    static constexpr unsigned int kernel_repeats = 1;
    static constexpr unsigned int warm_up_runs   = 1;

    static constexpr unsigned int fft_size            = S;
    static constexpr unsigned int elements_per_thread = UseSuggested ? FFT_complete::elements_per_thread : EPT;
    static constexpr unsigned int ffts_per_block      = UseSuggested ? FFT_complete::suggested_ffts_per_block : FPB;

    using FFT = decltype(FFT_complete() + ElementsPerThread<elements_per_thread>() + FFTsPerBlock<ffts_per_block>());
    using complex_type = typename FFT::value_type;

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        block_fft_kernel<FFT>, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

    int blocks_per_multiprocessor = 0;
    CUDA_CHECK_AND_EXIT(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_multiprocessor,
                                                      block_fft_kernel<FFT>,
                                                      FFT::block_dim.x * FFT::block_dim.y * FFT::block_dim.z,
                                                      FFT::shared_memory_size));

    unsigned int multiprocessor_count = example::get_multiprocessor_count();
    unsigned int cuda_blocks = blocks_per_multiprocessor * multiprocessor_count;

    // The memory required to run fft (number of complex_type values that must be allocated).
    // For r2c, the input consists of fft_size real numbers and the output consists of (fft_size / 2 + 1) complex numbers.
    // One memory block will be used to store input and output, so the memory block must fit
    // max((fft_size + 1) / 2, fft_size / 2 + 1) = (fft_size / 2 + 1) complex numbers.
    // For c2r, the input consists of (fft_size / 2 + 1) complex numbers and the output consists of fft_size real numbers,
    // so the minimal required memory size is the same.
    unsigned int input_size =
        ffts_per_block * cuda_blocks * (type_of<FFT>::value == fft_type::c2c ? fft_size : (fft_size / 2 + 1));

    // Host data
    std::vector<complex_type> input =
        example::get_random_complex_data<typename complex_type::value_type>(input_size, -10, 10);

    // Device data
    complex_type* device_buffer;
    auto          size_bytes = input.size() * sizeof(complex_type);
    CUDA_CHECK_AND_EXIT(cudaMalloc(&device_buffer, size_bytes));
    // Copy host to device
    CUDA_CHECK_AND_EXIT(cudaMemcpy(device_buffer, input.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    cudaError_t error_code = cudaSuccess;
    auto        workspace  = make_workspace<FFT>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    CUDA_CHECK_AND_EXIT(cudaGetLastError());

    // Measure performance of N trials
    double ms_n = measure<>::execution<warm_up_runs>(
        [&]() {
            for (unsigned int i = 0; i < kernel_repeats; i++) {
                block_fft_kernel<FFT><<<cuda_blocks, FFT::block_dim, FFT::shared_memory_size, stream>>>(
                    device_buffer, inside_repeats, workspace);
            }
        },
        stream);

    // Check kernel error
    CUDA_CHECK_AND_EXIT(cudaGetLastError());

    // Copy host to device
    CUDA_CHECK_AND_EXIT(cudaMemcpy(device_buffer, input.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Measure performance of 2*N trials
    double ms_n2 = measure<>::execution<warm_up_runs>(
        [&]() {
            for (unsigned int i = 0; i < kernel_repeats; i++) {
                block_fft_kernel<FFT><<<cuda_blocks, FFT::block_dim, FFT::shared_memory_size, stream>>>(
                    device_buffer, 2 * inside_repeats, workspace);
            }
        },
        stream);

    CUDA_CHECK_AND_EXIT(cudaFree(device_buffer));

    // Time for N repeats without overhead
    auto   time_n = ms_n2 - ms_n;
    double gflops = 1.0 * kernel_repeats * inside_repeats * ffts_per_block * cuda_blocks * 5.0 * fft_size *
                    (std::log(fft_size) / std::log(2)) / time_n / 1000000.0;

    static const std::string fft_type_name = type_of<FFT>::value == fft_type::c2c ? "c2c" :
                                             (type_of<FFT>::value == fft_type::c2r ? "c2r" :
                                             "r2c");
    if (verbose) {
        std::cout << "FFT type: " << fft_type_name << std::endl;
        std::cout << "FFT size: " << fft_size << std::endl;
        std::cout << "FFTs elements per thread: " << FFT::elements_per_thread << std::endl;
        std::cout << "FFTs per block: " << ffts_per_block << std::endl;
        std::cout << "CUDA blocks: " << cuda_blocks << std::endl;
        std::cout << "Blocks per multiprocessor: " << blocks_per_multiprocessor << std::endl;
        std::cout << "FFTs run: " << ffts_per_block * cuda_blocks << std::endl;
        std::cout << "Shared memory: " << FFT::shared_memory_size << std::endl;
        std::cout << "Avg Time [ms_n]: " << time_n / (inside_repeats * kernel_repeats) << std::endl;
        std::cout << "Time (all) [ms_n]: " << time_n << std::endl;
        std::cout << "Performance [GFLOPS]: " << gflops << std::endl;
    } else {
        std::cout << fft_type_name << ", " << fft_size << ", " << gflops << ", "
                  << time_n / (inside_repeats * kernel_repeats) << ", " << std::endl;
    }
}

#endif // CUFFTDX_EXAMPLE_BLOCK_FFT_PERFORMANCE_HPP_
