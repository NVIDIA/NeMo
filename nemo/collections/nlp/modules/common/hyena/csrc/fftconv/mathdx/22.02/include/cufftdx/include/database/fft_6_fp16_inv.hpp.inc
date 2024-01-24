#ifndef CUFFTDX_FFT_6_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_6_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1131, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<15>;
.reg .b32 r<237>;
.reg .f64 fd<15>;
.reg .b64 rd<2>;
mov.f64 fd7, 0dBFE0000000000000;
{
cvt.rn.f16.f64 rs1, fd7;
}
mov.b32 r72, {rs1, rs1};
mov.f64 fd4, 0dBFEBB67AE8584CAA;
{
cvt.rn.f16.f64 rs2, fd4;
}
mov.b32 r81, {rs2, rs2};
{
add.f16x2 r1, %17, %13;
}
{
add.f16x2 r4, %18, r1;
}
{
add.f16x2 r7, %19, %14;
}
{
add.f16x2 r10, %20, r7;
}
{
add.f16x2 r13, %17, %13;
}
{
mul.f16x2 r16, r13, r72;
}
{
add.f16x2 r19, %18, r16;
}
{
sub.f16x2 r22, %19, %14;
}
{
mul.f16x2 r25, r22, r81;
}
{
add.f16x2 r28, r19, r25;
}
{
add.f16x2 r31, %17, %13;
}
{
mul.f16x2 r34, r31, r72;
}
{
add.f16x2 r37, %18, r34;
}
{
sub.f16x2 r40, %19, %14;
}
{
mul.f16x2 r43, r40, r81;
}
{
sub.f16x2 r46, r37, r43;
}
{
add.f16x2 r49, %19, %14;
}
{
mul.f16x2 r52, r49, r72;
}
{
add.f16x2 r55, %20, r52;
}
{
sub.f16x2 r58, %17, %13;
}
{
mul.f16x2 r61, r58, r81;
}
{
sub.f16x2 r64, r55, r61;
}
{
add.f16x2 r67, %19, %14;
}
{
mul.f16x2 r70, r67, r72;
}
{
add.f16x2 r73, %20, r70;
}
{
sub.f16x2 r76, %17, %13;
}
{
mul.f16x2 r79, r76, r81;
}
{
add.f16x2 r82, r73, r79;
}
{
cvt.rn.f16.f64 rs3, fd7;
}
mov.b32 r156, {rs3, rs3};
{
cvt.rn.f16.f64 rs4, fd4;
}
mov.b32 r165, {rs4, rs4};
{
add.f16x2 r85, %21, %15;
}
{
add.f16x2 r88, %22, r85;
}
{
add.f16x2 r91, %12, %16;
}
{
add.f16x2 r94, %23, r91;
}
{
add.f16x2 r97, %21, %15;
}
{
mul.f16x2 r100, r97, r156;
}
{
add.f16x2 r103, %22, r100;
}
{
sub.f16x2 r106, %12, %16;
}
{
mul.f16x2 r109, r106, r165;
}
{
add.f16x2 r112, r103, r109;
}
{
add.f16x2 r115, %21, %15;
}
{
mul.f16x2 r118, r115, r156;
}
{
add.f16x2 r121, %22, r118;
}
{
sub.f16x2 r124, %12, %16;
}
{
mul.f16x2 r127, r124, r165;
}
{
sub.f16x2 r130, r121, r127;
}
{
add.f16x2 r133, %12, %16;
}
{
mul.f16x2 r136, r133, r156;
}
{
add.f16x2 r139, %23, r136;
}
{
sub.f16x2 r142, %21, %15;
}
{
mul.f16x2 r145, r142, r165;
}
{
sub.f16x2 r148, r139, r145;
}
{
add.f16x2 r151, %12, %16;
}
{
mul.f16x2 r154, r151, r156;
}
{
add.f16x2 r157, %23, r154;
}
{
sub.f16x2 r160, %21, %15;
}
{
mul.f16x2 r163, r160, r165;
}
{
add.f16x2 r166, r157, r163;
}
mov.f64 fd5, 0d3FE0000000000000;
{
cvt.rn.f16.f64 rs5, fd5;
}
mov.f64 fd8, 0d3FEBB67AE8584CAA;
{
cvt.rn.f16.f64 rs6, fd8;
}
{
cvt.rn.f16.f64 rs7, fd7;
}
{
cvt.rn.f16.f64 rs8, fd8;
}
mov.b32 r183, {rs5, rs5};
{
mul.f16x2 r169, r112, r183;
}
mov.b32 r180, {rs6, rs6};
{
mul.f16x2 r172, r148, r180;
}
{
sub.f16x2 r175, r169, r172;
}
{
mul.f16x2 r178, r112, r180;
}
{
fma.rn.f16x2 r181, r148, r183, r178;
}
mov.b32 r199, {rs7, rs7};
{
mul.f16x2 r185, r130, r199;
}
mov.b32 r196, {rs8, rs8};
{
mul.f16x2 r188, r166, r196;
}
{
sub.f16x2 r191, r185, r188;
}
{
mul.f16x2 r194, r130, r196;
}
{
fma.rn.f16x2 r197, r166, r199, r194;
}
{
add.f16x2 %0, r4, r88;
}
{
add.f16x2 %1, r10, r94;
}
{
sub.f16x2 %6, r4, r88;
}
{
sub.f16x2 %7, r10, r94;
}
{
add.f16x2 %2, r28, r175;
}
{
add.f16x2 %3, r64, r181;
}
{
sub.f16x2 %8, r28, r175;
}
{
sub.f16x2 %9, r64, r181;
}
{
add.f16x2 %4, r46, r191;
}
{
add.f16x2 %5, r82, r197;
}
{
sub.f16x2 %10, r46, r191;
}
{
sub.f16x2 %11, r82, r197;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)): "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)));
};


#endif
