#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include <cufft.h>
#ifdef CUFFTDX_EXAMPLES_CUFFT_CALLBACK
#include <cufftXt.h>
#endif

#include "block_io.hpp"
#include "common.hpp"
#include "random.hpp"

// Returns execution time in ms
template<unsigned int WarmUpRuns, typename Kernel>
float measure_execution(Kernel&& kernel, cudaStream_t stream) {
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK_AND_EXIT(cudaEventCreate(&startEvent));
    CUDA_CHECK_AND_EXIT(cudaEventCreate(&stopEvent));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    for (size_t i = 0; i < WarmUpRuns; i++) {
        kernel();
    }
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


template<class FFT, class IFFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void convolution_kernel(typename FFT::value_type*     data,
                                                                                 typename FFT::workspace_type  workspace,
                                                                                 typename IFFT::workspace_type workspace_inverse) {
    using complex_type = typename FFT::value_type;
    using scalar_type  = typename complex_type::value_type;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;

    // Load data from global memory to registers
    example::io<FFT>::load(data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem, workspace);

    // Scale values
    scalar_type scale = 1.0 / cufftdx::size_of<FFT>::value;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        thread_data[i].x *= scale;
        thread_data[i].y *= scale;
    }

    // Execute inverse FFT
    IFFT().execute(thread_data, shared_mem, workspace_inverse);

    // Save results
    example::io<FFT>::store(thread_data, data, local_fft_id);
}

// Scaling kernel; transforms data between cuFFTs.
template<unsigned int fft_size>
__global__ void scaling_kernel(cufftComplex*      data,
                               const unsigned int input_size,
                               const unsigned int ept) {

    static constexpr float scale = 1.0 / fft_size;

    cufftComplex temp;
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = 0; i < ept; i++) {
        if (index < input_size) {
            temp = data[index];
            temp.x *= scale;
            temp.y *= scale;
            data[index] = temp;
            index += blockDim.x * gridDim.x;
        }
    }
}

#ifdef CUFFTDX_EXAMPLES_CUFFT_CALLBACK
template<unsigned int fft_size>
__device__  cufftComplex scaling_callback(void *dataIn,
                                          size_t offset,
                                          void *callerInfo,
                                          void *sharedPtr) {
    static constexpr float scale = 1.0 / fft_size;

    cufftComplex value = static_cast<cufftComplex*>(dataIn)[offset];
    value.x *= scale;
    value.y *= scale;
    return value;
}

__device__ __managed__ cufftCallbackLoadC scaling_callback_ptr = scaling_callback<128>;
#endif

template<class FFT, class IFFT, unsigned int warm_up_runs>
double measure_cufftdx(const unsigned int&       kernel_repeats,
                       const unsigned int&       cuda_blocks,
                       typename FFT::value_type* device_buffer,
                       cudaStream_t              stream) {

    using namespace cufftdx;
    using complex_type = typename FFT::value_type;

    // create workspaces for FFT and IFFT
    cudaError_t error_code = cudaSuccess;
    auto        workspace  = make_workspace<FFT>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);
    auto workspace_inverse = make_workspace<IFFT>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);

    // run cuFFTDx
    double time = measure_execution<warm_up_runs>(
        [&]() {
            for (unsigned int i = 0; i < kernel_repeats; i++) {
                // There are (ffts_per_block * fft_size * cuda_blocks) elements
                convolution_kernel<FFT, IFFT><<<cuda_blocks, FFT::block_dim, FFT::shared_memory_size, stream>>>(
                    device_buffer, workspace, workspace_inverse);
            }
        },
        stream);

    return time;
}

template<unsigned int fft_size, unsigned int warm_up_runs>
double measure_cufft(const unsigned int& kernel_repeats,
                     const unsigned int& batch_size,
                     cufftComplex*       device_buffer,
                     cudaStream_t        stream) {

    static constexpr unsigned int block_dim_scaling_kernel = 1024;

    // Calculating parameters for scaling_kernel execution.
    // Get maximum number of running CUDA blocks per multiprocessor.
    int blocks_per_multiprocessor = 0;
    CUDA_CHECK_AND_EXIT(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_multiprocessor,
                                                      scaling_kernel<fft_size>,
                                                      block_dim_scaling_kernel,
                                                      0));

    // Get maximum number of CUDA blocks running on all multiprocessors.
    // This many CUDA blocks will be run for simple_kernel.
    const unsigned int cuda_blocks = blocks_per_multiprocessor * example::get_multiprocessor_count();

    const unsigned int input_length        = fft_size * batch_size;
    const unsigned int elements_per_block  = (input_length + cuda_blocks - 1) / cuda_blocks;
    const unsigned int elements_per_thread = (elements_per_block + block_dim_scaling_kernel - 1) / block_dim_scaling_kernel;

    // prepare cuFFT runs
    cufftHandle plan;
    CUFFT_CHECK_AND_EXIT(cufftPlan1d(&plan, fft_size, CUFFT_C2C, batch_size));
    CUFFT_CHECK_AND_EXIT(cufftSetStream(plan, stream));

    // run convolution
    double time_cufft = measure_execution<warm_up_runs>(
        [&]() {
            for (unsigned int i = 0; i < kernel_repeats; i++) {

                if (cufftExecC2C(plan, device_buffer, device_buffer, CUFFT_FORWARD) != CUFFT_SUCCESS) {
                    fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
                    return;
                }

                scaling_kernel<fft_size>
                    <<<cuda_blocks, block_dim_scaling_kernel, 0, stream>>>(device_buffer, input_length, elements_per_thread);

                if (cufftExecC2C(plan, device_buffer, device_buffer, CUFFT_INVERSE) != CUFFT_SUCCESS) {
                    fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
                    return;
                }
            }
        },
        stream);

    CUFFT_CHECK_AND_EXIT(cufftDestroy(plan));
    return time_cufft;
}

#ifdef CUFFTDX_EXAMPLES_CUFFT_CALLBACK
template<unsigned int fft_size, unsigned int warm_up_runs>
double measure_cufft_callback(const unsigned int& kernel_repeats,
                              const unsigned int& batch_size,
                              cufftComplex*       device_buffer,
                              cudaStream_t        stream) {

    static constexpr unsigned int block_dim_scaling_kernel = 1024;

    // Calculating parameters for scaling_kernel execution.
    // Get maximum number of running CUDA blocks per multiprocessor.
    int blocks_per_multiprocessor = 0;
    CUDA_CHECK_AND_EXIT(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_multiprocessor,
                                                      scaling_kernel<fft_size>,
                                                      block_dim_scaling_kernel,
                                                      0));
    // prepare cuFFT runs
    cufftHandle plan_in;
    CUFFT_CHECK_AND_EXIT(cufftPlan1d(&plan_in, fft_size, CUFFT_C2C, batch_size));
    CUFFT_CHECK_AND_EXIT(cufftSetStream(plan_in, stream));
    cufftHandle plan_out;
    CUFFT_CHECK_AND_EXIT(cufftPlan1d(&plan_out, fft_size, CUFFT_C2C, batch_size));
    CUFFT_CHECK_AND_EXIT(cufftSetStream(plan_out, stream));

    // Set input callback
    CUFFT_CHECK_AND_EXIT(cufftXtSetCallback(plan_in,
                                            reinterpret_cast<void**>(&scaling_callback_ptr),
                                            CUFFT_CB_LD_COMPLEX,
                                            nullptr));

    // run convolution
    double time_cufft = measure_execution<warm_up_runs>(
        [&]() {
            for (unsigned int i = 0; i < kernel_repeats; i++) {
                CUFFT_CHECK_AND_EXIT(cufftExecC2C(plan_in, device_buffer, device_buffer, CUFFT_FORWARD));
                CUFFT_CHECK_AND_EXIT(cufftExecC2C(plan_out, device_buffer, device_buffer, CUFFT_INVERSE));
            }
        },
        stream);

    CUFFT_CHECK_AND_EXIT(cufftDestroy(plan_in));
    CUFFT_CHECK_AND_EXIT(cufftDestroy(plan_out));
    return time_cufft;
}
#endif // CUFFTDX_EXAMPLES_CUFFT_CALLBACK

// This example compares performance of cuFFT and cuFFTDx when performing C2C convolution.
// Data is generated on host, copied to device buffer and processed by FFTs.
// Each cuFFTDx execution runs one kernel, each cuFFT execution - three kernels.
// The experiment runs with the following principles:
// - at least 1GB of data is allocated in GPU and transformed by both convolutions,
// - for cuFFTDx kernel run, number of CUDA blocks is divisible
//   by maximum number of CUDA blocks that can run simultaneously on the GPU.
template<unsigned int Arch>
void convolution() {
    using namespace cufftdx;

    static constexpr unsigned int minimum_input_size_bytes = (1 << 30); // At least one GB of data will be processed by FFTs.
    static constexpr unsigned int fft_size                 = 512;
    static constexpr unsigned int kernel_repeats           = 10;
    static constexpr unsigned int warm_up_runs             = 1;
    static constexpr bool         verbose                  = true;

    static constexpr bool         use_suggested              = true; // Whether to use suggested FPB and EPT values or custom.
    static constexpr unsigned int custom_ffts_per_block      = 2;
    static constexpr unsigned int custom_elements_per_thread = 8;

    // To determine the total input length (number of fft batches to run), the maximum number of
    // simultanously running cuFFTDx CUDA blocks is calculated.

    // Declaration of cuFFTDx run
    using fft_incomplete = decltype(Block() + Size<fft_size>() + Type<fft_type::c2c>() + Precision<float>() + SM<Arch>());
    using fft_base       = decltype(fft_incomplete() + Direction<fft_direction::forward>());
    using ifft_base      = decltype(fft_incomplete() + Direction<fft_direction::inverse>());

    static constexpr unsigned int elements_per_thread = use_suggested ? fft_base::elements_per_thread : custom_elements_per_thread;
    static constexpr unsigned int ffts_per_block      = use_suggested ? fft_base::suggested_ffts_per_block : custom_ffts_per_block;

    using fft          = decltype(fft_base() + ElementsPerThread<elements_per_thread>() + FFTsPerBlock<ffts_per_block>());
    using ifft         = decltype(ifft_base() + ElementsPerThread<elements_per_thread>() + FFTsPerBlock<ffts_per_block>());
    using complex_type = typename fft::value_type;

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        convolution_kernel<fft, ifft>, cudaFuncAttributeMaxDynamicSharedMemorySize, fft::shared_memory_size));

    // Get maximum number of running CUDA blocks per multiprocessor
    int blocks_per_multiprocessor = 0;
    CUDA_CHECK_AND_EXIT(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_multiprocessor,
                                                      convolution_kernel<fft, ifft>,
                                                      fft::block_dim.x * fft::block_dim.y * fft::block_dim.z,
                                                      fft::shared_memory_size));

    // Get maximum number of CUDA blocks running on all multiprocessors
    const unsigned int device_blocks = blocks_per_multiprocessor * example::get_multiprocessor_count();

    // Input size in bytes if device_blocks CUDA blocks were run.
    const unsigned int data_size_device_blocks_bytes = device_blocks * ffts_per_block * fft_size * sizeof(complex_type);

    // cuda_blocks = minimal number of CUDA blocks to run, such that:
    //   - cuda_blocks is divisible by device_blocks,
    //   - total input size is not less than minimum_input_size_bytes.
    // executed_blocks_multiplyer = cuda_blocks / device_blocks
    const unsigned int executed_blocks_multiplyer =
        (minimum_input_size_bytes + data_size_device_blocks_bytes - 1) / data_size_device_blocks_bytes;
    const unsigned int cuda_blocks  = device_blocks * executed_blocks_multiplyer;
    const unsigned int input_length = ffts_per_block * cuda_blocks * fft_size;

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    // Host data
    std::vector<complex_type> input =
        example::get_random_complex_data<typename complex_type::value_type>(input_length, -10, 10);

    // Device data
    complex_type* device_buffer;
    auto          input_size_bytes = input.size() * sizeof(complex_type);
    CUDA_CHECK_AND_EXIT(cudaMalloc(&device_buffer, input_size_bytes));

    // Copy host to device
    CUDA_CHECK_AND_EXIT(cudaMemcpy(device_buffer, input.data(), input_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Measure performance
    double time_cufftdx = measure_cufftdx<fft, ifft, warm_up_runs>(kernel_repeats, cuda_blocks, device_buffer, stream);
    double time_cufft   = measure_cufft<fft_size, warm_up_runs>(
        kernel_repeats, cuda_blocks * ffts_per_block, (cufftComplex*)device_buffer, stream);
    #ifdef CUFFTDX_EXAMPLES_CUFFT_CALLBACK
    double time_cufft_cb = measure_cufft_callback<fft_size, warm_up_runs>(
        kernel_repeats, cuda_blocks * ffts_per_block, (cufftComplex*)device_buffer, stream);
    #endif

    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    CUDA_CHECK_AND_EXIT(cudaFree(device_buffer));

    // Report results.
    auto report_time_and_performance = [&](std::string name, double time) -> void {
        double gflops = 1.0 * kernel_repeats * ffts_per_block * cuda_blocks * 5.0 * fft_size *
                        (std::log(fft_size) / std::log(2)) / time / 1000000.0;

        std::cout << std::endl;
        std::cout << name << std::endl;
        std::cout << "Avg Time [ms_n]: " << time / kernel_repeats << std::endl;
        std::cout << "Time (all) [ms_n]: " << time << std::endl;
        std::cout << "Performance [GFLOPS]: " << gflops << std::endl;
    };

    if (verbose) {
        std::cout << "FFT size: " << fft_size << std::endl;
        std::cout << "FFTs run: " << ffts_per_block * cuda_blocks << std::endl;
        report_time_and_performance("cuFFTDx", time_cufftdx);
        std::cout << "FFTs elements per thread: " << fft::elements_per_thread << std::endl;
        std::cout << "FFTs per block: " << ffts_per_block << std::endl;
        std::cout << "CUDA blocks: " << cuda_blocks << std::endl;
        std::cout << "Blocks per multiprocessor: " << blocks_per_multiprocessor << std::endl;

        report_time_and_performance("cuFFT", time_cufft);
        #ifdef CUFFTDX_EXAMPLES_CUFFT_CALLBACK
        report_time_and_performance("cuFFT Callback", time_cufft_cb);
        #endif
    } else {
        double gflops_cufftdx = 1.0 * kernel_repeats * ffts_per_block * cuda_blocks * 5.0 * fft_size *
                                (std::log(fft_size) / std::log(2)) / time_cufftdx / 1000000.0;
        double gflops_cufft = 1.0 * kernel_repeats * ffts_per_block * cuda_blocks * 5.0 * fft_size *
                              (std::log(fft_size) / std::log(2)) / time_cufft / 1000000.0;
        std::cout << fft_size << ": " << std::endl
                  << gflops_cufftdx << ", " << time_cufftdx / kernel_repeats << ", " << std::endl
                  << gflops_cufft << ", " << time_cufft / kernel_repeats;
    }
}

template<unsigned int Arch>
struct convolution_functor {
    void operator()() {
        return convolution<Arch>();
    }
};

int main(int, char**) {
    return example::sm_runner<convolution_functor>();
}
