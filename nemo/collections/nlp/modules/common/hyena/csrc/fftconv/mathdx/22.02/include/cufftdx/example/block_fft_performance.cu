#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#include "block_fft_performance.hpp"

template<unsigned int Arch>
void block_fft_performance() {
    using namespace cufftdx;

    using fft_base = decltype(Block() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                              Precision<float>() + SM<Arch>());

    static constexpr unsigned int elements_per_thread = 8;
    static constexpr unsigned int fft_size            = 512;
    static constexpr unsigned int ffts_per_block      = 1;

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream))
    benchmark_block_fft<fft_base, fft_size, elements_per_thread, ffts_per_block>(stream, true);
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
}

template<unsigned int Arch>
struct block_fft_performance_functor {
    void operator()() { return block_fft_performance<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<block_fft_performance_functor>();
}
