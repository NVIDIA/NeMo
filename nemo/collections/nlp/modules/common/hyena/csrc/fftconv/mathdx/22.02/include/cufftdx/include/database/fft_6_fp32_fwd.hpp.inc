#ifndef CUFFTDX_FFT_6_FP32_FWD_PTX_HPP
#define CUFFTDX_FFT_6_FP32_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<183, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<79>;
.reg .b64 rd<2>;
add.f32 f25, %17, %22;
add.f32 f26, %12, f25;
add.f32 f27, %19, %24;
add.f32 f28, %13, f27;
mul.f32 f29, f25, 0f3F000000;
sub.f32 f30, %12, f29;
sub.f32 f31, %19, %24;
mul.f32 f32, f31, 0f3F5DB3D7;
add.f32 f33, f32, f30;
sub.f32 f34, f30, f32;
mul.f32 f35, f27, 0f3F000000;
sub.f32 f36, %13, f35;
sub.f32 f37, %17, %22;
mul.f32 f38, f37, 0f3F5DB3D7;
sub.f32 f39, f36, f38;
add.f32 f40, f38, f36;
add.f32 f41, %20, %25;
add.f32 f42, %14, f41;
add.f32 f43, %21, %26;
add.f32 f44, %16, f43;
mul.f32 f45, f41, 0f3F000000;
sub.f32 f46, %14, f45;
sub.f32 f47, %21, %26;
mul.f32 f48, f47, 0f3F5DB3D7;
add.f32 f49, f48, f46;
sub.f32 f50, f46, f48;
mul.f32 f51, f43, 0f3F000000;
sub.f32 f52, %16, f51;
sub.f32 f53, %20, %25;
mul.f32 f54, f53, 0f3F5DB3D7;
sub.f32 f55, f52, f54;
add.f32 f56, f54, f52;
mul.f32 f57, f49, 0f3F000000;
mul.f32 f58, f55, 0fBF5DB3D7;
sub.f32 f59, f57, f58;
mul.f32 f60, f55, 0f3F000000;
fma.rn.f32 f61, f49, 0fBF5DB3D7, f60;
mul.f32 f62, f50, 0fBF000000;
mul.f32 f63, f56, 0fBF5DB3D7;
sub.f32 f64, f62, f63;
mul.f32 f65, f56, 0fBF000000;
fma.rn.f32 f66, f50, 0fBF5DB3D7, f65;
add.f32 %1, f28, f44;
add.f32 %0, f26, f42;
add.f32 %3, f39, f61;
add.f32 %2, f33, f59;
add.f32 %5, f40, f66;
add.f32 %4, f34, f64;
sub.f32 %7, f28, f44;
sub.f32 %6, f26, f42;
sub.f32 %9, f39, f61;
sub.f32 %8, f33, f59;
sub.f32 %11, f40, f66;
sub.f32 %10, f34, f64;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y), "=f"(rmem[2].x), "=f"(rmem[2].y), "=f"(rmem[3].x), "=f"(rmem[3].y), "=f"(rmem[4].x), "=f"(rmem[4].y), "=f"(rmem[5].x), "=f"(rmem[5].y): "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y), "f"(rmem[1].y), "f"(rmem[2].x), "f"(rmem[2].y), "f"(rmem[2].y), "f"(rmem[3].x), "f"(rmem[3].y), "f"(rmem[4].x), "f"(rmem[4].y), "f"(rmem[4].y), "f"(rmem[5].x), "f"(rmem[5].y));
};


#endif
