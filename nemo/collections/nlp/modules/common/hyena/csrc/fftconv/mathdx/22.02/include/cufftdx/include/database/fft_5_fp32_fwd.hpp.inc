#ifndef CUFFTDX_FFT_5_FP32_FWD_PTX_HPP
#define CUFFTDX_FFT_5_FP32_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<154, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<65>;
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
mul.f32 f33, f32, 0fBF167918;
sub.f32 f34, f33, f31;
mul.f32 f35, f21, 0f3F4F1BBD;
sub.f32 f36, %10, f35;
fma.rn.f32 f37, f23, 0f3E9E377A, f36;
mul.f32 f38, f30, 0f3F167918;
mul.f32 f39, f32, 0f3F737871;
sub.f32 f40, f39, f38;
fma.rn.f32 f41, f24, 0f3E9E377A, %11;
mul.f32 f42, f26, 0f3F4F1BBD;
sub.f32 f43, f41, f42;
sub.f32 f44, %12, %20;
mul.f32 f45, f44, 0f3F737871;
sub.f32 f46, %15, %18;
mul.f32 f47, f46, 0fBF167918;
sub.f32 f48, f47, f45;
mul.f32 f49, f24, 0f3F4F1BBD;
sub.f32 f50, %11, f49;
fma.rn.f32 f51, f26, 0f3E9E377A, f50;
mul.f32 f52, f44, 0f3F167918;
mul.f32 f53, f46, 0f3F737871;
sub.f32 f54, f53, f52;
add.f32 %1, f26, f25;
add.f32 %0, f23, f22;
add.f32 %3, f48, f43;
sub.f32 %2, f29, f34;
add.f32 %5, f54, f51;
sub.f32 %4, f37, f40;
sub.f32 %7, f51, f54;
add.f32 %6, f40, f37;
sub.f32 %9, f43, f48;
add.f32 %8, f34, f29;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y), "=f"(rmem[2].x), "=f"(rmem[2].y), "=f"(rmem[3].x), "=f"(rmem[3].y), "=f"(rmem[4].x), "=f"(rmem[4].y): "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y), "f"(rmem[1].y), "f"(rmem[2].x), "f"(rmem[2].y), "f"(rmem[2].y), "f"(rmem[3].x), "f"(rmem[3].y), "f"(rmem[4].x), "f"(rmem[4].y));
};


#endif
