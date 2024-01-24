#ifndef CUFFTDX_FFT_6_FP64_FWD_PTX_HPP
#define CUFFTDX_FFT_6_FP64_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<558, double, 1>(cufftdx::detail::complex<double> *rmem, unsigned smem){

asm volatile (R"({
.reg .f64 fd<79>;
.reg .b64 rd<2>;
add.f64 fd25, %17, %22;
add.f64 fd26, %12, fd25;
add.f64 fd27, %19, %24;
add.f64 fd28, %13, fd27;
mul.f64 fd29, fd25, 0d3FE0000000000000;
sub.f64 fd30, %12, fd29;
sub.f64 fd31, %19, %24;
mul.f64 fd32, fd31, 0d3FEBB67AE8584CAA;
add.f64 fd33, fd32, fd30;
sub.f64 fd34, fd30, fd32;
mul.f64 fd35, fd27, 0d3FE0000000000000;
sub.f64 fd36, %13, fd35;
sub.f64 fd37, %17, %22;
mul.f64 fd38, fd37, 0d3FEBB67AE8584CAA;
sub.f64 fd39, fd36, fd38;
add.f64 fd40, fd38, fd36;
add.f64 fd41, %20, %25;
add.f64 fd42, %14, fd41;
add.f64 fd43, %21, %26;
add.f64 fd44, %16, fd43;
mul.f64 fd45, fd41, 0d3FE0000000000000;
sub.f64 fd46, %14, fd45;
sub.f64 fd47, %21, %26;
mul.f64 fd48, fd47, 0d3FEBB67AE8584CAA;
add.f64 fd49, fd48, fd46;
sub.f64 fd50, fd46, fd48;
mul.f64 fd51, fd43, 0d3FE0000000000000;
sub.f64 fd52, %16, fd51;
sub.f64 fd53, %20, %25;
mul.f64 fd54, fd53, 0d3FEBB67AE8584CAA;
sub.f64 fd55, fd52, fd54;
add.f64 fd56, fd54, fd52;
mul.f64 fd57, fd49, 0d3FE0000000000000;
mul.f64 fd58, fd55, 0dBFEBB67AE8584CAA;
sub.f64 fd59, fd57, fd58;
mul.f64 fd60, fd55, 0d3FE0000000000000;
fma.rn.f64 fd61, fd49, 0dBFEBB67AE8584CAA, fd60;
mul.f64 fd62, fd50, 0dBFE0000000000000;
mul.f64 fd63, fd56, 0dBFEBB67AE8584CAA;
sub.f64 fd64, fd62, fd63;
mul.f64 fd65, fd56, 0dBFE0000000000000;
fma.rn.f64 fd66, fd50, 0dBFEBB67AE8584CAA, fd65;
add.f64 %1, fd28, fd44;
add.f64 %0, fd26, fd42;
add.f64 %3, fd39, fd61;
add.f64 %2, fd33, fd59;
add.f64 %5, fd40, fd66;
add.f64 %4, fd34, fd64;
sub.f64 %7, fd28, fd44;
sub.f64 %6, fd26, fd42;
sub.f64 %9, fd39, fd61;
sub.f64 %8, fd33, fd59;
sub.f64 %11, fd40, fd66;
sub.f64 %10, fd34, fd64;
})"
     : "=d"(rmem[0].x), "=d"(rmem[0].y), "=d"(rmem[1].x), "=d"(rmem[1].y), "=d"(rmem[2].x), "=d"(rmem[2].y), "=d"(rmem[3].x), "=d"(rmem[3].y), "=d"(rmem[4].x), "=d"(rmem[4].y), "=d"(rmem[5].x), "=d"(rmem[5].y): "d"(rmem[0].x), "d"(rmem[0].y), "d"(rmem[1].x), "d"(rmem[1].y), "d"(rmem[1].y), "d"(rmem[2].x), "d"(rmem[2].y), "d"(rmem[2].y), "d"(rmem[3].x), "d"(rmem[3].y), "d"(rmem[4].x), "d"(rmem[4].y), "d"(rmem[4].y), "d"(rmem[5].x), "d"(rmem[5].y));
};


#endif
