#ifndef CUFFTDX_FFT_7_FP64_INV_PTX_HPP
#define CUFFTDX_FFT_7_FP64_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<717, double, 1>(cufftdx::detail::complex<double> *rmem, unsigned smem){

asm volatile (R"({
.reg .f64 fd<113>;
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
fma.rn.f64 fd47, fd46, 0d3FEF329C0558E969, fd45;
sub.f64 fd48, %23, %26;
fma.rn.f64 fd49, fd48, 0d3FDBC4C04D71ABC1, fd47;
mul.f64 fd50, fd29, 0d3FCC7B90E3024582;
sub.f64 fd51, %14, fd50;
mul.f64 fd52, fd31, 0d3FECD4BCA9CB5C71;
sub.f64 fd53, fd51, fd52;
fma.rn.f64 fd54, fd33, 0d3FE3F3A0E28BEDD1, fd53;
mul.f64 fd55, fd44, 0d3FEF329C0558E969;
mul.f64 fd56, fd46, 0d3FDBC4C04D71ABC1;
sub.f64 fd57, fd55, fd56;
mul.f64 fd58, fd48, 0d3FE904C37505DE4B;
sub.f64 fd59, fd57, fd58;
mul.f64 fd60, fd29, 0d3FECD4BCA9CB5C71;
sub.f64 fd61, %14, fd60;
fma.rn.f64 fd62, fd31, 0d3FE3F3A0E28BEDD1, fd61;
mul.f64 fd63, fd33, 0d3FCC7B90E3024582;
sub.f64 fd64, fd62, fd63;
mul.f64 fd65, fd44, 0d3FDBC4C04D71ABC1;
mul.f64 fd66, fd46, 0d3FE904C37505DE4B;
sub.f64 fd67, fd65, fd66;
fma.rn.f64 fd68, fd48, 0d3FEF329C0558E969, fd67;
fma.rn.f64 fd69, fd34, 0d3FE3F3A0E28BEDD1, %15;
mul.f64 fd70, fd36, 0d3FCC7B90E3024582;
sub.f64 fd71, fd69, fd70;
mul.f64 fd72, fd38, 0d3FECD4BCA9CB5C71;
sub.f64 fd73, fd71, fd72;
sub.f64 fd74, %16, %30;
mul.f64 fd75, fd74, 0d3FE904C37505DE4B;
sub.f64 fd76, %19, %27;
fma.rn.f64 fd77, fd76, 0d3FEF329C0558E969, fd75;
sub.f64 fd78, %22, %24;
fma.rn.f64 fd79, fd78, 0d3FDBC4C04D71ABC1, fd77;
mul.f64 fd80, fd34, 0d3FCC7B90E3024582;
sub.f64 fd81, %15, fd80;
mul.f64 fd82, fd36, 0d3FECD4BCA9CB5C71;
sub.f64 fd83, fd81, fd82;
fma.rn.f64 fd84, fd38, 0d3FE3F3A0E28BEDD1, fd83;
mul.f64 fd85, fd74, 0d3FEF329C0558E969;
mul.f64 fd86, fd76, 0d3FDBC4C04D71ABC1;
sub.f64 fd87, fd85, fd86;
mul.f64 fd88, fd78, 0d3FE904C37505DE4B;
sub.f64 fd89, fd87, fd88;
mul.f64 fd90, fd34, 0d3FECD4BCA9CB5C71;
sub.f64 fd91, %15, fd90;
fma.rn.f64 fd92, fd36, 0d3FE3F3A0E28BEDD1, fd91;
mul.f64 fd93, fd38, 0d3FCC7B90E3024582;
sub.f64 fd94, fd92, fd93;
mul.f64 fd95, fd74, 0d3FDBC4C04D71ABC1;
mul.f64 fd96, fd76, 0d3FE904C37505DE4B;
sub.f64 fd97, fd95, fd96;
fma.rn.f64 fd98, fd78, 0d3FEF329C0558E969, fd97;
add.f64 %1, fd38, fd37;
add.f64 %0, fd33, fd32;
add.f64 %3, fd79, fd73;
sub.f64 %2, fd43, fd49;
add.f64 %5, fd89, fd84;
sub.f64 %4, fd54, fd59;
add.f64 %7, fd98, fd94;
sub.f64 %6, fd64, fd68;
sub.f64 %9, fd94, fd98;
add.f64 %8, fd68, fd64;
sub.f64 %11, fd84, fd89;
add.f64 %10, fd59, fd54;
sub.f64 %13, fd73, fd79;
add.f64 %12, fd49, fd43;
})"
     : "=d"(rmem[0].x), "=d"(rmem[0].y), "=d"(rmem[1].x), "=d"(rmem[1].y), "=d"(rmem[2].x), "=d"(rmem[2].y), "=d"(rmem[3].x), "=d"(rmem[3].y), "=d"(rmem[4].x), "=d"(rmem[4].y), "=d"(rmem[5].x), "=d"(rmem[5].y), "=d"(rmem[6].x), "=d"(rmem[6].y): "d"(rmem[0].x), "d"(rmem[0].y), "d"(rmem[1].x), "d"(rmem[1].y), "d"(rmem[1].y), "d"(rmem[2].x), "d"(rmem[2].y), "d"(rmem[2].y), "d"(rmem[3].x), "d"(rmem[3].y), "d"(rmem[4].x), "d"(rmem[4].y), "d"(rmem[4].y), "d"(rmem[5].x), "d"(rmem[5].y), "d"(rmem[5].y), "d"(rmem[6].x), "d"(rmem[6].y));
};


#endif
