#ifndef CUFFTDX_FFT_2_FP64_INV_PTX_HPP
#define CUFFTDX_FFT_2_FP64_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<591, double, 1>(cufftdx::detail::complex<double> *rmem, unsigned smem){

asm volatile (R"({
.reg .f64 fd<13>;
.reg .b64 rd<2>;
add.f64 %1, %5, %7;
add.f64 %0, %4, %6;
sub.f64 %3, %5, %7;
sub.f64 %2, %4, %6;
})"
     : "=d"(rmem[0].x), "=d"(rmem[0].y), "=d"(rmem[1].x), "=d"(rmem[1].y): "d"(rmem[0].x), "d"(rmem[0].y), "d"(rmem[1].x), "d"(rmem[1].y));
};


#endif
