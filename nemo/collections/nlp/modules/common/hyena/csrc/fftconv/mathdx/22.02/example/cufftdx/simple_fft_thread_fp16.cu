#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "common.hpp"
#include "fp16_common.hpp"

template<class FFT>
__global__ void thread_fft_kernel(typename FFT::value_type* data) {
    using complex_type = typename FFT::value_type;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // Load data from global memory to registers.
    // thread_data should have all input data in order.
    unsigned int index = threadIdx.x * FFT::elements_per_thread;
    for (size_t i = 0; i < FFT::elements_per_thread; i++) {
        // complex<half2> values are processed with assumtion that they are in RRII layout,
        // but data has them in RIRI layout. example::to_rrii converts RIRI to RRII.
        thread_data[i] = example::to_rrii(data[index + i]);
    }

    // Execute FFT
    FFT().execute(thread_data);

    // Save results
    for (size_t i = 0; i < FFT::elements_per_thread; i++) {
        // converting back form RRII to RIRI layout
        data[index + i] = example::to_riri(thread_data[i]);
    }
}

// In this example a one-dimensional half-precision complex-to-complex transform is perform by each CUDA thread.
//
// Three (threads_count) threads are run, and each thread calculates two 8-point (fft_size) C2C half precision FFTs.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
//
// Note: In half precision cuFFTDx uses complex<half2> type and processes values in implicit batches of two FFTs, ie.
// each thread processes two FFTs.
int main(int, char**) {
    using namespace cufftdx;

    // Number of threads to execute
    // In case of half precision each thread caluclates two FFTs
    static constexpr unsigned int threads_count = 3;

    // FFT is defined, its: size, type, direction, precision.
    // Thread() operator informs that FFT will be executed on a thread level.
    using FFT          = decltype(Thread() + Size<8>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                         Precision<__half>());
    using complex_type = typename FFT::value_type;

    // Host data
    std::vector<complex_type> input(cufftdx::size_of<FFT>::value);
    for (size_t i = 0; i < input.size(); i++) {
        float v1 = i;
        float v2 = i + input.size();
        // Populate input with complex<half2> values in ((Real, Imag), (Real, Imag)) layout
        input[i] = complex_type {__half2 {v1, -v1}, __half2 {v2, -v2}};
    }

    std::cout << "input [1st FFT]:\n";
    for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
        std::cout << __half2float(input[i].x.x) << " " << __half2float(input[i].x.y) << std::endl;
    }

    // Device data
    complex_type* device_buffer;
    auto          size_bytes = input.size() * sizeof(complex_type);
    CUDA_CHECK_AND_EXIT(cudaMalloc(&device_buffer, size_bytes));
    // Copy host to device
    CUDA_CHECK_AND_EXIT(cudaMemcpy(device_buffer, input.data(), size_bytes, cudaMemcpyHostToDevice));

    // Invokes kernel with 'threads_count' threads in block, each thread calculates two FFTs of size 8
    thread_fft_kernel<FFT><<<1, threads_count>>>(device_buffer);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy device to host
    std::vector<complex_type> output(input.size());
    CUDA_CHECK_AND_EXIT(cudaMemcpy(output.data(), device_buffer, size_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaFree(device_buffer));

    std::cout << "output [1st FFT]:\n";
    for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
        std::cout << __half2float(output[i].x.x) << " " << __half2float(output[i].x.y) << std::endl;
    }

    std::cout << "Success" << std::endl;
}
