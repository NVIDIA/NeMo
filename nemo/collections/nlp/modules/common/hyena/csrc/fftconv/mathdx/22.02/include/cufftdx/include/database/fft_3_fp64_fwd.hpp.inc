#ifndef CUFFTDX_FFT_3_FP64_FWD_PTX_HPP
#define CUFFTDX_FFT_3_FP64_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<500, double, 1>(cufftdx::detail::complex<double> *rmem, unsigned smem){

asm volatile (R"({
.reg .f64 fd<29>;
.reg .b64 rd<2>;
add.f64 fd13, %8, %11;
add.f64 fd14, %10, %12;
mul.f64 fd15, fd13, 0d3FE0000000000000;
sub.f64 fd16, %6, fd15;
sub.f64 fd17, %10, %12;
mul.f64 fd18, fd17, 0d3FEBB67AE8584CAA;
mul.f64 fd19, fd14, 0d3FE0000000000000;
sub.f64 fd20, %7, fd19;
sub.f64 fd21, %8, %11;
mul.f64 fd22, fd21, 0d3FEBB67AE8584CAA;
add.f64 %1, %7, fd14;
add.f64 %0, %6, fd13;
sub.f64 %3, fd20, fd22;
add.f64 %2, fd18, fd16;
add.f64 %5, fd22, fd20;
sub.f64 %4, fd16, fd18;
})"
     : "=d"(rmem[0].x), "=d"(rmem[0].y), "=d"(rmem[1].x), "=d"(rmem[1].y), "=d"(rmem[2].x), "=d"(rmem[2].y): "d"(rmem[0].x), "d"(rmem[0].y), "d"(rmem[1].x), "d"(rmem[1].y), "d"(rmem[1].y), "d"(rmem[2].x), "d"(rmem[2].y));
};


#endif
