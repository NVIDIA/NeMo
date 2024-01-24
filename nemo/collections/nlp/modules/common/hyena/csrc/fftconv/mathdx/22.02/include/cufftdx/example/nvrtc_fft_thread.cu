#include <vector>
#include <iostream>
#include <string>
#include <algorithm>

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cufftdx.hpp>

#include "common.hpp"

#define NVRTC_SAFE_CALL(x)                                                                            \
    do {                                                                                              \
        nvrtcResult result = x;                                                                       \
        if (result != NVRTC_SUCCESS) {                                                                \
            std::cerr << "\nerror: " #x " failed with error " << nvrtcGetErrorString(result) << '\n'; \
            exit(1);                                                                                  \
        }                                                                                             \
    } while (0)

const char* thread_fft_kernel = R"kernel(
#include <cufftdx.hpp>

using namespace cufftdx;

// FFT
using size_desc  = Size<FFT_SIZE>;
using dir_desc   = Direction<fft_direction::inverse>;
using type_c2c   = Type<fft_type::c2c>;
using FFT        = decltype(size_desc() + dir_desc() + type_c2c() + Thread() + Precision<double>());

extern "C" __global__ void thread_fft_kernel(typename FFT::value_type *data)
{
    // Local array for thread
    typename FFT::value_type thread_data[FFT::storage_size];

    // Load data from global memory to registers.
    // thread_data should have all input data in order.
    unsigned int index = threadIdx.x * FFT::elements_per_thread;
    for (size_t i = 0; i < FFT::elements_per_thread; i++) {
        thread_data[i] = data[index + i];
    }

    // Execute FFT
    FFT().execute(thread_data);

    // Save results
    for (size_t i = 0; i < FFT::elements_per_thread; i++) {
        data[index + i] = thread_data[i];
    }
}
)kernel";

int main(int, char**) {
    // Define FFT
    using namespace cufftdx;

    static constexpr unsigned int fft_size = 16;

    // FFT Operators
    using size_desc  = Size<fft_size>;
    using dir_desc   = Direction<fft_direction::inverse>;
    using type_c2c   = Type<fft_type::c2c>;
    using FFT        = decltype(size_desc() + dir_desc() + type_c2c() + Thread() + Precision<double>());
    using value_type = typename FFT::value_type;

    std::string fft_size_definition = "-DFFT_SIZE=" + std::to_string(fft_size);
    // Parse cuFFTDx include dirs
    std::vector<std::string> cufftdx_include_dirs_array;
    {
        std::string cufftdx_include_dirs = CUFFTDX_INCLUDE_DIRS;
        std::string delim                = ";";
        auto        start                = 0U;
        auto        end                  = cufftdx_include_dirs.find(delim);
        while (end != std::string::npos) {
            cufftdx_include_dirs_array.push_back("--include-path=" + cufftdx_include_dirs.substr(start, end - start));
            start = end + delim.length();
            end   = cufftdx_include_dirs.find(delim, start);
        }
        cufftdx_include_dirs_array.push_back("--include-path=" + cufftdx_include_dirs.substr(start, end - start));
    }

    // Get architecture of current device
    int device;
    CUDA_CHECK_AND_EXIT(cudaGetDevice(&device));
    int major = 0;
    int minor = 0;
    CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    std::string gpu_architecture_option =
        "--gpu-architecture=compute_" + std::to_string(major * 10 + minor);

    // Create a program
    nvrtcProgram program;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&program,               // program
                                       thread_fft_kernel,      // buffer
                                       "thread_fft_kernel.cu", // name
                                       0,                      // numHeaders
                                       NULL,                   // headers
                                       NULL));                 // includeNames

    // Prepare compilation options
    std::vector<const char*> opts = {
        "--std=c++17",
        "--device-as-default-execution-space",
        "--include-path=" CUDA_INCLUDE_DIR // Add path to CUDA include directory
    };
    // Include cuFFTDx dir in opts
    for (auto& d : cufftdx_include_dirs_array) {
        opts.push_back(d.c_str());
    }
    // Add FFT_SIZE definition
    opts.push_back(fft_size_definition.c_str());
    // Add gpu-architecture flag
    opts.push_back(gpu_architecture_option.c_str());

    nvrtcResult compileResult = nvrtcCompileProgram(program,      // program
                                                    opts.size(),  // numOptions
                                                    opts.data()); // options

    // Obtain compilation log from the program
    if (compileResult != NVRTC_SUCCESS) {
        size_t log_size;
        NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(program, &log_size));
        char* log = new char[log_size];
        NVRTC_SAFE_CALL(nvrtcGetProgramLog(program, log));
        std::cout << log << '\n';
        delete[] log;
        std::exit(1);
    }

    // Obtain PTX from the program.
    size_t ptx_size;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(program, &ptx_size));
    char* ptx = new char[ptx_size];
    NVRTC_SAFE_CALL(nvrtcGetPTX(program, ptx));

    // Destroy the program.
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&program));

    // Load the generated PTX and get a handle to the thread_fft_kernel
    CUcontext  context;
    CUmodule   module;
    CUfunction kernel;
    CUDA_CHECK_AND_EXIT(cudaFree(0));               // Initialize CUDA context
    CUDA_CHECK_AND_EXIT(cuCtxGetCurrent(&context)); // Get current context
    CUDA_CHECK_AND_EXIT(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
    CUDA_CHECK_AND_EXIT(cuModuleGetFunction(&kernel, module, "thread_fft_kernel"));

    // Generate input for execution
    std::vector<value_type> host_input(cufftdx::size_of<FFT>::value);
    float                   i = 0.0f;
    for (auto& v : host_input) {
        v.x = i++;
        v.y = 0;
    }

    size_t fft_buffer_size = cufftdx::size_of<FFT>::value * sizeof(value_type);
    void*  device_values;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&device_values, fft_buffer_size));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(device_values, host_input.data(), fft_buffer_size, cudaMemcpyHostToDevice));

    // Execute thread_fft_kernel
    void* args[] = {&device_values};
    CUDA_CHECK_AND_EXIT(cuLaunchKernel(kernel,
                                       1, // number of blocks
                                       1,
                                       1,
                                       1, // number of threads
                                       1,
                                       1,
                                       0,    // no shared memory
                                       NULL, // NULL stream
                                       args,
                                       0));
    CUDA_CHECK_AND_EXIT(cuCtxSynchronize());

    // Retrieve and print output.
    std::vector<value_type> host_output(cufftdx::size_of<FFT>::value);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(host_output.data(), device_values, fft_buffer_size, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < cufftdx::size_of<FFT>::value; ++i) {
        std::cout << i << ": (" << host_output[i].x << ", " << host_output[i].y << ")" << std::endl;
    }

    // Release resources.
    CUDA_CHECK_AND_EXIT(cudaFree(device_values));
    CUDA_CHECK_AND_EXIT(cuModuleUnload(module));

    double expected_value = (fft_size * (fft_size + 1)) / 2;
    if ((host_output[0].x - expected_value) > 0.01) {
        std::cout << "Failed" << std::endl;
        return 1;
    }
    std::cout << "Success" << std::endl;
    return 0;
}
