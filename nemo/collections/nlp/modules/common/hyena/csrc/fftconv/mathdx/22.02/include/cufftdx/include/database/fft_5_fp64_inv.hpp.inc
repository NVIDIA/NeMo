#ifndef CUFFTDX_FFT_5_FP64_INV_PTX_HPP
#define CUFFTDX_FFT_5_FP64_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<703, double, 1>(cufftdx::detail::complex<double> *rmem, unsigned smem){

asm volatile (R"({
.reg .f64 fd<63>;
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
fma.rn.f64 fd33, fd32, 0d3FE2CF2304755A5E, fd31;
mul.f64 fd34, fd21, 0d3FE9E3779B97F4A8;
sub.f64 fd35, %10, fd34;
fma.rn.f64 fd36, fd23, 0d3FD3C6EF372FE950, fd35;
mul.f64 fd37, fd30, 0d3FE2CF2304755A5E;
mul.f64 fd38, fd32, 0d3FEE6F0E134454FF;
sub.f64 fd39, fd37, fd38;
fma.rn.f64 fd40, fd24, 0d3FD3C6EF372FE950, %11;
mul.f64 fd41, fd26, 0d3FE9E3779B97F4A8;
sub.f64 fd42, fd40, fd41;
sub.f64 fd43, %12, %20;
mul.f64 fd44, fd43, 0d3FEE6F0E134454FF;
sub.f64 fd45, %15, %18;
fma.rn.f64 fd46, fd45, 0d3FE2CF2304755A5E, fd44;
mul.f64 fd47, fd24, 0d3FE9E3779B97F4A8;
sub.f64 fd48, %11, fd47;
fma.rn.f64 fd49, fd26, 0d3FD3C6EF372FE950, fd48;
mul.f64 fd50, fd43, 0d3FE2CF2304755A5E;
mul.f64 fd51, fd45, 0d3FEE6F0E134454FF;
sub.f64 fd52, fd50, fd51;
add.f64 %1, fd26, fd25;
add.f64 %0, fd23, fd22;
add.f64 %3, fd46, fd42;
sub.f64 %2, fd29, fd33;
add.f64 %5, fd52, fd49;
sub.f64 %4, fd36, fd39;
sub.f64 %7, fd49, fd52;
add.f64 %6, fd39, fd36;
sub.f64 %9, fd42, fd46;
add.f64 %8, fd33, fd29;
})"
     : "=d"(rmem[0].x), "=d"(rmem[0].y), "=d"(rmem[1].x), "=d"(rmem[1].y), "=d"(rmem[2].x), "=d"(rmem[2].y), "=d"(rmem[3].x), "=d"(rmem[3].y), "=d"(rmem[4].x), "=d"(rmem[4].y): "d"(rmem[0].x), "d"(rmem[0].y), "d"(rmem[1].x), "d"(rmem[1].y), "d"(rmem[1].y), "d"(rmem[2].x), "d"(rmem[2].y), "d"(rmem[2].y), "d"(rmem[3].x), "d"(rmem[3].y), "d"(rmem[4].x), "d"(rmem[4].y));
};


#endif
