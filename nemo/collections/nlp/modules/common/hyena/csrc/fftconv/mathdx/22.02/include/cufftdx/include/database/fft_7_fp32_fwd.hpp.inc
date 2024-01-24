#ifndef CUFFTDX_FFT_7_FP32_FWD_PTX_HPP
#define CUFFTDX_FFT_7_FP32_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<171, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<117>;
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
mul.f32 f47, f46, 0fBF7994E0;
sub.f32 f48, f47, f45;
sub.f32 f49, %23, %26;
mul.f32 f50, f49, 0f3EDE2602;
sub.f32 f51, f48, f50;
mul.f32 f52, f29, 0f3E63DC87;
sub.f32 f53, %14, f52;
mul.f32 f54, f31, 0f3F66A5E5;
sub.f32 f55, f53, f54;
fma.rn.f32 f56, f33, 0f3F1F9D07, f55;
mul.f32 f57, f44, 0f3F7994E0;
mul.f32 f58, f46, 0f3EDE2602;
sub.f32 f59, f58, f57;
fma.rn.f32 f60, f49, 0f3F48261C, f59;
mul.f32 f61, f29, 0f3F66A5E5;
sub.f32 f62, %14, f61;
fma.rn.f32 f63, f31, 0f3F1F9D07, f62;
mul.f32 f64, f33, 0f3E63DC87;
sub.f32 f65, f63, f64;
mul.f32 f66, f44, 0f3EDE2602;
mul.f32 f67, f46, 0f3F48261C;
sub.f32 f68, f67, f66;
mul.f32 f69, f49, 0f3F7994E0;
sub.f32 f70, f68, f69;
fma.rn.f32 f71, f34, 0f3F1F9D07, %15;
mul.f32 f72, f36, 0f3E63DC87;
sub.f32 f73, f71, f72;
mul.f32 f74, f38, 0f3F66A5E5;
sub.f32 f75, f73, f74;
sub.f32 f76, %16, %30;
mul.f32 f77, f76, 0f3F48261C;
sub.f32 f78, %19, %27;
mul.f32 f79, f78, 0fBF7994E0;
sub.f32 f80, f79, f77;
sub.f32 f81, %22, %24;
mul.f32 f82, f81, 0f3EDE2602;
sub.f32 f83, f80, f82;
mul.f32 f84, f34, 0f3E63DC87;
sub.f32 f85, %15, f84;
mul.f32 f86, f36, 0f3F66A5E5;
sub.f32 f87, f85, f86;
fma.rn.f32 f88, f38, 0f3F1F9D07, f87;
mul.f32 f89, f76, 0f3F7994E0;
mul.f32 f90, f78, 0f3EDE2602;
sub.f32 f91, f90, f89;
fma.rn.f32 f92, f81, 0f3F48261C, f91;
mul.f32 f93, f34, 0f3F66A5E5;
sub.f32 f94, %15, f93;
fma.rn.f32 f95, f36, 0f3F1F9D07, f94;
mul.f32 f96, f38, 0f3E63DC87;
sub.f32 f97, f95, f96;
mul.f32 f98, f76, 0f3EDE2602;
mul.f32 f99, f78, 0f3F48261C;
sub.f32 f100, f99, f98;
mul.f32 f101, f81, 0f3F7994E0;
sub.f32 f102, f100, f101;
add.f32 %1, f38, f37;
add.f32 %0, f33, f32;
add.f32 %3, f83, f75;
sub.f32 %2, f43, f51;
add.f32 %5, f92, f88;
sub.f32 %4, f56, f60;
add.f32 %7, f102, f97;
sub.f32 %6, f65, f70;
sub.f32 %9, f97, f102;
add.f32 %8, f70, f65;
sub.f32 %11, f88, f92;
add.f32 %10, f60, f56;
sub.f32 %13, f75, f83;
add.f32 %12, f51, f43;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y), "=f"(rmem[2].x), "=f"(rmem[2].y), "=f"(rmem[3].x), "=f"(rmem[3].y), "=f"(rmem[4].x), "=f"(rmem[4].y), "=f"(rmem[5].x), "=f"(rmem[5].y), "=f"(rmem[6].x), "=f"(rmem[6].y): "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y), "f"(rmem[1].y), "f"(rmem[2].x), "f"(rmem[2].y), "f"(rmem[2].y), "f"(rmem[3].x), "f"(rmem[3].y), "f"(rmem[4].x), "f"(rmem[4].y), "f"(rmem[4].y), "f"(rmem[5].x), "f"(rmem[5].y), "f"(rmem[5].y), "f"(rmem[6].x), "f"(rmem[6].y));
};


#endif
