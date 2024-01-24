#ifndef CUFFTDX_FFT_5_FP32_INV_PTX_HPP
#define CUFFTDX_FFT_5_FP32_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<356, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<63>;
.reg .b64 rd<2>;
add.f32 f21, %12, %20;
add.f32 f22, %10, f21;
add.f32 f23, %15, %18;
add.f32 f24, %14, %21;
add.f32 f25, %11, f24;
add.f32 f26, %17, %19;
fma.rn.f32 f27, f21, 0f3E9E377A, %10;
mul.f32 f28, f23, 0f3F4F1BBD;
sub.f32 f29, f27, f28;
sub.f32 f30, %14, %21;
mul.f32 f31, f30, 0f3F737871;
sub.f32 f32, %17, %19;
fma.rn.f32 f33, f32, 0f3F167918, f31;
mul.f32 f34, f21, 0f3F4F1BBD;
sub.f32 f35, %10, f34;
fma.rn.f32 f36, f23, 0f3E9E377A, f35;
mul.f32 f37, f30, 0f3F167918;
mul.f32 f38, f32, 0f3F737871;
sub.f32 f39, f37, f38;
fma.rn.f32 f40, f24, 0f3E9E377A, %11;
mul.f32 f41, f26, 0f3F4F1BBD;
sub.f32 f42, f40, f41;
sub.f32 f43, %12, %20;
mul.f32 f44, f43, 0f3F737871;
sub.f32 f45, %15, %18;
fma.rn.f32 f46, f45, 0f3F167918, f44;
mul.f32 f47, f24, 0f3F4F1BBD;
sub.f32 f48, %11, f47;
fma.rn.f32 f49, f26, 0f3E9E377A, f48;
mul.f32 f50, f43, 0f3F167918;
mul.f32 f51, f45, 0f3F737871;
sub.f32 f52, f50, f51;
add.f32 %1, f26, f25;
add.f32 %0, f23, f22;
add.f32 %3, f46, f42;
sub.f32 %2, f29, f33;
add.f32 %5, f52, f49;
sub.f32 %4, f36, f39;
sub.f32 %7, f49, f52;
add.f32 %6, f39, f36;
sub.f32 %9, f42, f46;
add.f32 %8, f33, f29;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y), "=f"(rmem[2].x), "=f"(rmem[2].y), "=f"(rmem[3].x), "=f"(rmem[3].y), "=f"(rmem[4].x), "=f"(rmem[4].y): "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y), "f"(rmem[1].y), "f"(rmem[2].x), "f"(rmem[2].y), "f"(rmem[2].y), "f"(rmem[3].x), "f"(rmem[3].y), "f"(rmem[4].x), "f"(rmem[4].y));
};


#endif
