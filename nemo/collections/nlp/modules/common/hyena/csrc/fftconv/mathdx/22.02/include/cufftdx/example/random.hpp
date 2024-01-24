
#ifndef CUFFTDX_EXAMPLE_RANDOM_HPP_
#define CUFFTDX_EXAMPLE_RANDOM_HPP_

#include <algorithm>
#include <random>
#include <vector>
#include <type_traits>

#include <cuda.h>
#include <cufftdx.hpp>

namespace example {
    template<class T>
    inline auto get_random_complex_data(size_t size, T min, T max) ->
        typename std::enable_if<std::is_floating_point<T>::value,
                                std::vector<cufftdx::make_complex_type_t<T>>>::type {
        using complex_type = cufftdx::make_complex_type_t<T>;
        std::random_device                rd;
        std::default_random_engine        gen(rd());
        std::uniform_real_distribution<T> distribution(min, max);
        std::vector<complex_type>         output(size);
        std::generate(output.begin(), output.end(), [&]() {
            return complex_type {distribution(gen), distribution(gen)};
        });
        return output;
    }

    template<class T>
    inline auto get_random_complex_data(size_t size, T min, T max) ->
        typename std::enable_if<std::is_same<T, __half>::value,
                                std::vector<cufftdx::make_complex_type_t<__half2>>>::type {
        using complex_type = cufftdx::make_complex_type_t<__half2>;
        std::random_device                    rd;
        std::default_random_engine            gen(rd());
        std::uniform_real_distribution<float> distribution(min, max);
        std::vector<complex_type>             output(size);
        std::generate(output.begin(), output.end(), [&]() {
            auto xx = __float2half(distribution(gen));
            auto xy = __float2half(distribution(gen));
            auto yx = __float2half(distribution(gen));
            auto yy = __float2half(distribution(gen));
            auto x  = __half2 {xx, xy};
            auto y  = __half2 {yx, yy};
            return complex_type {x, y};
        });
        return output;
    }
} // namespace example

#endif // CUFFTDX_EXAMPLE_RANDOM_HPP_
