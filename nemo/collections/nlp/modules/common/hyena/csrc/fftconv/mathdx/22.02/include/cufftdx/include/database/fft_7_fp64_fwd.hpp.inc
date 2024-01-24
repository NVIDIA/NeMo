#ifndef CUFFTDX_FFT_7_FP64_FWD_PTX_HPP
#define CUFFTDX_FFT_7_FP64_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<546, double, 1>(cufftdx::detail::complex<double> *rmem, unsigned smem){

asm volatile (R"({
.reg .f64 fd<117>;
.reg .b64 rd<2>;
add.f64 fd29, %16, %30;
add.f64 fd30, %14, fd29;
add.f64 fd31, %19, %27;
add.f64 fd32, fd31, fd30;
add.f64 fd33, %22, %24;
add.f64 fd34, %18, %31;
add.f64 fd35, %15, fd34;
add.f64 fd36, %21, %29;
add.f64 fd37, fd36, fd35;
add.f64 fd38, %23, %26;
fma.rn.f64 fd39, fd29, 0d3FE3F3A0E28BEDD1, %14;
mul.f64 fd40, fd31, 0d3FCC7B90E3024582;
sub.f64 fd41, fd39, fd40;
mul.f64 fd42, fd33, 0d3FECD4BCA9CB5C71;
sub.f64 fd43, fd41, fd42;
sub.f64 fd44, %18, %31;
mul.f64 fd45, fd44, 0d3FE904C37505DE4B;
sub.f64 fd46, %21, %29;
mul.f64 fd47, fd46, 0dBFEF329C0558E969;
sub.f64 fd48, fd47, fd45;
sub.f64 fd49, %23, %26;
mul.f64 fd50, fd49, 0d3FDBC4C04D71ABC1;
sub.f64 fd51, fd48, fd50;
mul.f64 fd52, fd29, 0d3FCC7B90E3024582;
sub.f64 fd53, %14, fd52;
mul.f64 fd54, fd31, 0d3FECD4BCA9CB5C71;
sub.f64 fd55, fd53, fd54;
fma.rn.f64 fd56, fd33, 0d3FE3F3A0E28BEDD1, fd55;
mul.f64 fd57, fd44, 0d3FEF329C0558E969;
mul.f64 fd58, fd46, 0d3FDBC4C04D71ABC1;
sub.f64 fd59, fd58, fd57;
fma.rn.f64 fd60, fd49, 0d3FE904C37505DE4B, fd59;
mul.f64 fd61, fd29, 0d3FECD4BCA9CB5C71;
sub.f64 fd62, %14, fd61;
fma.rn.f64 fd63, fd31, 0d3FE3F3A0E28BEDD1, fd62;
mul.f64 fd64, fd33, 0d3FCC7B90E3024582;
sub.f64 fd65, fd63, fd64;
mul.f64 fd66, fd44, 0d3FDBC4C04D71ABC1;
mul.f64 fd67, fd46, 0d3FE904C37505DE4B;
sub.f64 fd68, fd67, fd66;
mul.f64 fd69, fd49, 0d3FEF329C0558E969;
sub.f64 fd70, fd68, fd69;
fma.rn.f64 fd71, fd34, 0d3FE3F3A0E28BEDD1, %15;
mul.f64 fd72, fd36, 0d3FCC7B90E3024582;
sub.f64 fd73, fd71, fd72;
mul.f64 fd74, fd38, 0d3FECD4BCA9CB5C71;
sub.f64 fd75, fd73, fd74;
sub.f64 fd76, %16, %30;
mul.f64 fd77, fd76, 0d3FE904C37505DE4B;
sub.f64 fd78, %19, %27;
mul.f64 fd79, fd78, 0dBFEF329C0558E969;
sub.f64 fd80, fd79, fd77;
sub.f64 fd81, %22, %24;
mul.f64 fd82, fd81, 0d3FDBC4C04D71ABC1;
sub.f64 fd83, fd80, fd82;
mul.f64 fd84, fd34, 0d3FCC7B90E3024582;
sub.f64 fd85, %15, fd84;
mul.f64 fd86, fd36, 0d3FECD4BCA9CB5C71;
sub.f64 fd87, fd85, fd86;
fma.rn.f64 fd88, fd38, 0d3FE3F3A0E28BEDD1, fd87;
mul.f64 fd89, fd76, 0d3FEF329C0558E969;
mul.f64 fd90, fd78, 0d3FDBC4C04D71ABC1;
sub.f64 fd91, fd90, fd89;
fma.rn.f64 fd92, fd81, 0d3FE904C37505DE4B, fd91;
mul.f64 fd93, fd34, 0d3FECD4BCA9CB5C71;
sub.f64 fd94, %15, fd93;
fma.rn.f64 fd95, fd36, 0d3FE3F3A0E28BEDD1, fd94;
mul.f64 fd96, fd38, 0d3FCC7B90E3024582;
sub.f64 fd97, fd95, fd96;
mul.f64 fd98, fd76, 0d3FDBC4C04D71ABC1;
mul.f64 fd99, fd78, 0d3FE904C37505DE4B;
sub.f64 fd100, fd99, fd98;
mul.f64 fd101, fd81, 0d3FEF329C0558E969;
sub.f64 fd102, fd100, fd101;
add.f64 %1, fd38, fd37;
add.f64 %0, fd33, fd32;
add.f64 %3, fd83, fd75;
sub.f64 %2, fd43, fd51;
add.f64 %5, fd92, fd88;
sub.f64 %4, fd56, fd60;
add.f64 %7, fd102, fd97;
sub.f64 %6, fd65, fd70;
sub.f64 %9, fd97, fd102;
add.f64 %8, fd70, fd65;
sub.f64 %11, fd88, fd92;
add.f64 %10, fd60, fd56;
sub.f64 %13, fd75, fd83;
add.f64 %12, fd51, fd43;
})"
     : "=d"(rmem[0].x), "=d"(rmem[0].y), "=d"(rmem[1].x), "=d"(rmem[1].y), "=d"(rmem[2].x), "=d"(rmem[2].y), "=d"(rmem[3].x), "=d"(rmem[3].y), "=d"(rmem[4].x), "=d"(rmem[4].y), "=d"(rmem[5].x), "=d"(rmem[5].y), "=d"(rmem[6].x), "=d"(rmem[6].y): "d"(rmem[0].x), "d"(rmem[0].y), "d"(rmem[1].x), "d"(rmem[1].y), "d"(rmem[1].y), "d"(rmem[2].x), "d"(rmem[2].y), "d"(rmem[2].y), "d"(rmem[3].x), "d"(rmem[3].y), "d"(rmem[4].x), "d"(rmem[4].y), "d"(rmem[4].y), "d"(rmem[5].x), "d"(rmem[5].y), "d"(rmem[5].y), "d"(rmem[6].x), "d"(rmem[6].y));
};


#endif
