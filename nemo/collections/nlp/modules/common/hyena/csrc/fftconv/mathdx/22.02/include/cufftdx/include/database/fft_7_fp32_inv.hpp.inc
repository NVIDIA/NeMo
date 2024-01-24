#ifndef CUFFTDX_FFT_7_FP32_INV_PTX_HPP
#define CUFFTDX_FFT_7_FP32_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<373, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<113>;
.reg .b64 rd<2>;
add.f32 f29, %16, %30;
add.f32 f30, %14, f29;
add.f32 f31, %19, %27;
add.f32 f32, f31, f30;
add.f32 f33, %22, %24;
add.f32 f34, %18, %31;
add.f32 f35, %15, f34;
add.f32 f36, %21, %29;
add.f32 f37, f36, f35;
add.f32 f38, %23, %26;
fma.rn.f32 f39, f29, 0f3F1F9D07, %14;
mul.f32 f40, f31, 0f3E63DC87;
sub.f32 f41, f39, f40;
mul.f32 f42, f33, 0f3F66A5E5;
sub.f32 f43, f41, f42;
sub.f32 f44, %18, %31;
mul.f32 f45, f44, 0f3F48261C;
sub.f32 f46, %21, %29;
fma.rn.f32 f47, f46, 0f3F7994E0, f45;
sub.f32 f48, %23, %26;
fma.rn.f32 f49, f48, 0f3EDE2602, f47;
mul.f32 f50, f29, 0f3E63DC87;
sub.f32 f51, %14, f50;
mul.f32 f52, f31, 0f3F66A5E5;
sub.f32 f53, f51, f52;
fma.rn.f32 f54, f33, 0f3F1F9D07, f53;
mul.f32 f55, f44, 0f3F7994E0;
mul.f32 f56, f46, 0f3EDE2602;
sub.f32 f57, f55, f56;
mul.f32 f58, f48, 0f3F48261C;
sub.f32 f59, f57, f58;
mul.f32 f60, f29, 0f3F66A5E5;
sub.f32 f61, %14, f60;
fma.rn.f32 f62, f31, 0f3F1F9D07, f61;
mul.f32 f63, f33, 0f3E63DC87;
sub.f32 f64, f62, f63;
mul.f32 f65, f44, 0f3EDE2602;
mul.f32 f66, f46, 0f3F48261C;
sub.f32 f67, f65, f66;
fma.rn.f32 f68, f48, 0f3F7994E0, f67;
fma.rn.f32 f69, f34, 0f3F1F9D07, %15;
mul.f32 f70, f36, 0f3E63DC87;
sub.f32 f71, f69, f70;
mul.f32 f72, f38, 0f3F66A5E5;
sub.f32 f73, f71, f72;
sub.f32 f74, %16, %30;
mul.f32 f75, f74, 0f3F48261C;
sub.f32 f76, %19, %27;
fma.rn.f32 f77, f76, 0f3F7994E0, f75;
sub.f32 f78, %22, %24;
fma.rn.f32 f79, f78, 0f3EDE2602, f77;
mul.f32 f80, f34, 0f3E63DC87;
sub.f32 f81, %15, f80;
mul.f32 f82, f36, 0f3F66A5E5;
sub.f32 f83, f81, f82;
fma.rn.f32 f84, f38, 0f3F1F9D07, f83;
mul.f32 f85, f74, 0f3F7994E0;
mul.f32 f86, f76, 0f3EDE2602;
sub.f32 f87, f85, f86;
mul.f32 f88, f78, 0f3F48261C;
sub.f32 f89, f87, f88;
mul.f32 f90, f34, 0f3F66A5E5;
sub.f32 f91, %15, f90;
fma.rn.f32 f92, f36, 0f3F1F9D07, f91;
mul.f32 f93, f38, 0f3E63DC87;
sub.f32 f94, f92, f93;
mul.f32 f95, f74, 0f3EDE2602;
mul.f32 f96, f76, 0f3F48261C;
sub.f32 f97, f95, f96;
fma.rn.f32 f98, f78, 0f3F7994E0, f97;
add.f32 %1, f38, f37;
add.f32 %0, f33, f32;
add.f32 %3, f79, f73;
sub.f32 %2, f43, f49;
add.f32 %5, f89, f84;
sub.f32 %4, f54, f59;
add.f32 %7, f98, f94;
sub.f32 %6, f64, f68;
sub.f32 %9, f94, f98;
add.f32 %8, f68, f64;
sub.f32 %11, f84, f89;
add.f32 %10, f59, f54;
sub.f32 %13, f73, f79;
add.f32 %12, f49, f43;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y), "=f"(rmem[2].x), "=f"(rmem[2].y), "=f"(rmem[3].x), "=f"(rmem[3].y), "=f"(rmem[4].x), "=f"(rmem[4].y), "=f"(rmem[5].x), "=f"(rmem[5].y), "=f"(rmem[6].x), "=f"(rmem[6].y): "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y), "f"(rmem[1].y), "f"(rmem[2].x), "f"(rmem[2].y), "f"(rmem[2].y), "f"(rmem[3].x), "f"(rmem[3].y), "f"(rmem[4].x), "f"(rmem[4].y), "f"(rmem[4].y), "f"(rmem[5].x), "f"(rmem[5].y), "f"(rmem[5].y), "f"(rmem[6].x), "f"(rmem[6].y));
};


#endif
