#ifndef CUFFTDX_FFT_5_FP64_FWD_PTX_HPP
#define CUFFTDX_FFT_5_FP64_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<532, double, 1>(cufftdx::detail::complex<double> *rmem, unsigned smem){

asm volatile (R"({
.reg .f64 fd<65>;
.reg .b64 rd<2>;
add.f64 fd21, %12, %20;
add.f64 fd22, %10, fd21;
add.f64 fd23, %15, %18;
add.f64 fd24, %14, %21;
add.f64 fd25, %11, fd24;
add.f64 fd26, %17, %19;
fma.rn.f64 fd27, fd21, 0d3FD3C6EF372FE950, %10;
mul.f64 fd28, fd23, 0d3FE9E3779B97F4A8;
sub.f64 fd29, fd27, fd28;
sub.f64 fd30, %14, %21;
mul.f64 fd31, fd30, 0d3FEE6F0E134454FF;
sub.f64 fd32, %17, %19;
mul.f64 fd33, fd32, 0dBFE2CF2304755A5E;
sub.f64 fd34, fd33, fd31;
mul.f64 fd35, fd21, 0d3FE9E3779B97F4A8;
sub.f64 fd36, %10, fd35;
fma.rn.f64 fd37, fd23, 0d3FD3C6EF372FE950, fd36;
mul.f64 fd38, fd30, 0d3FE2CF2304755A5E;
mul.f64 fd39, fd32, 0d3FEE6F0E134454FF;
sub.f64 fd40, fd39, fd38;
fma.rn.f64 fd41, fd24, 0d3FD3C6EF372FE950, %11;
mul.f64 fd42, fd26, 0d3FE9E3779B97F4A8;
sub.f64 fd43, fd41, fd42;
sub.f64 fd44, %12, %20;
mul.f64 fd45, fd44, 0d3FEE6F0E134454FF;
sub.f64 fd46, %15, %18;
mul.f64 fd47, fd46, 0dBFE2CF2304755A5E;
sub.f64 fd48, fd47, fd45;
mul.f64 fd49, fd24, 0d3FE9E3779B97F4A8;
sub.f64 fd50, %11, fd49;
fma.rn.f64 fd51, fd26, 0d3FD3C6EF372FE950, fd50;
mul.f64 fd52, fd44, 0d3FE2CF2304755A5E;
mul.f64 fd53, fd46, 0d3FEE6F0E134454FF;
sub.f64 fd54, fd53, fd52;
add.f64 %1, fd26, fd25;
add.f64 %0, fd23, fd22;
add.f64 %3, fd48, fd43;
sub.f64 %2, fd29, fd34;
add.f64 %5, fd54, fd51;
sub.f64 %4, fd37, fd40;
sub.f64 %7, fd51, fd54;
add.f64 %6, fd40, fd37;
sub.f64 %9, fd43, fd48;
add.f64 %8, fd34, fd29;
})"
     : "=d"(rmem[0].x), "=d"(rmem[0].y), "=d"(rmem[1].x), "=d"(rmem[1].y), "=d"(rmem[2].x), "=d"(rmem[2].y), "=d"(rmem[3].x), "=d"(rmem[3].y), "=d"(rmem[4].x), "=d"(rmem[4].y): "d"(rmem[0].x), "d"(rmem[0].y), "d"(rmem[1].x), "d"(rmem[1].y), "d"(rmem[1].y), "d"(rmem[2].x), "d"(rmem[2].y), "d"(rmem[2].y), "d"(rmem[3].x), "d"(rmem[3].y), "d"(rmem[4].x), "d"(rmem[4].y));
};


#endif
