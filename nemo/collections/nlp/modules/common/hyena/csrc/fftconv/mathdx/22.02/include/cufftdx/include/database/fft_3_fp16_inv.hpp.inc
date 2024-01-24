#ifndef CUFFTDX_FFT_3_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_3_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1065, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<3>;
.reg .b32 r<85>;
.reg .f64 fd<3>;
.reg .b64 rd<2>;
mov.f64 fd1, 0dBFE0000000000000;
{
cvt.rn.f16.f64 rs1, fd1;
}
mov.b32 r72, {rs1, rs1};
mov.f64 fd2, 0dBFEBB67AE8584CAA;
{
cvt.rn.f16.f64 rs2, fd2;
}
mov.b32 r81, {rs2, rs2};
{
add.f16x2 r1, %6, %7;
}
{
add.f16x2 %0, %8, r1;
}
{
add.f16x2 r7, %9, %10;
}
{
add.f16x2 %1, %11, r7;
}
{
add.f16x2 r13, %6, %7;
}
{
mul.f16x2 r16, r13, r72;
}
{
add.f16x2 r19, %8, r16;
}
{
sub.f16x2 r22, %9, %10;
}
{
mul.f16x2 r25, r22, r81;
}
{
add.f16x2 %2, r19, r25;
}
{
add.f16x2 r31, %6, %7;
}
{
mul.f16x2 r34, r31, r72;
}
{
add.f16x2 r37, %8, r34;
}
{
sub.f16x2 r40, %9, %10;
}
{
mul.f16x2 r43, r40, r81;
}
{
sub.f16x2 %4, r37, r43;
}
{
add.f16x2 r49, %9, %10;
}
{
mul.f16x2 r52, r49, r72;
}
{
add.f16x2 r55, %11, r52;
}
{
sub.f16x2 r58, %6, %7;
}
{
mul.f16x2 r61, r58, r81;
}
{
sub.f16x2 %3, r55, r61;
}
{
add.f16x2 r67, %9, %10;
}
{
mul.f16x2 r70, r67, r72;
}
{
add.f16x2 r73, %11, r70;
}
{
sub.f16x2 r76, %6, %7;
}
{
mul.f16x2 r79, r76, r81;
}
{
add.f16x2 %5, r73, r79;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[0].y)));
};


#endif
