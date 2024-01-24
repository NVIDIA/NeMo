#ifndef CUFFTDX_FFT_5_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_5_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<900, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<9>;
.reg .b32 r<313>;
.reg .f64 fd<7>;
.reg .b64 rd<2>;
mov.f64 fd5, 0d3FD3C6EF372FE950;
{
cvt.rn.f16.f64 rs1, fd5;
}
mov.b32 r210, {rs1, rs1};
mov.f64 fd6, 0dBFEE6F0E134454FF;
{
cvt.rn.f16.f64 rs2, fd6;
}
mov.b32 r228, {rs2, rs2};
mov.f64 fd3, 0dBFE9E3779B97F4A8;
{
cvt.rn.f16.f64 rs3, fd3;
}
mov.b32 r282, {rs3, rs3};
mov.f64 fd4, 0dBFE2CF2304755A5E;
{
cvt.rn.f16.f64 rs4, fd4;
}
mov.b32 r300, {rs4, rs4};
{
cvt.rn.f16.f64 rs5, fd5;
}
mov.b32 r291, {rs5, rs5};
{
cvt.rn.f16.f64 rs6, fd6;
}
{
neg.f16 rs7, rs6;
}
mov.b32 r306, {rs7, rs7};
{
add.f16x2 r1, %11, %12;
}
{
add.f16x2 r4, %13, r1;
}
{
add.f16x2 r7, %14, %15;
}
{
add.f16x2 %0, r4, r7;
}
{
add.f16x2 r13, %16, %10;
}
{
add.f16x2 r16, %17, r13;
}
{
add.f16x2 r19, %18, %19;
}
{
add.f16x2 %1, r16, r19;
}
{
add.f16x2 r25, %11, %12;
}
{
mul.f16x2 r28, r25, r210;
}
{
add.f16x2 r31, %13, r28;
}
{
add.f16x2 r34, %14, %15;
}
{
mul.f16x2 r37, r34, r282;
}
{
add.f16x2 r40, r31, r37;
}
{
sub.f16x2 r43, %16, %10;
}
{
mul.f16x2 r46, r43, r228;
}
{
sub.f16x2 r49, %18, %19;
}
{
mul.f16x2 r52, r49, r300;
}
{
add.f16x2 r55, r46, r52;
}
{
sub.f16x2 %2, r40, r55;
}
{
add.f16x2 r61, %11, %12;
}
{
mul.f16x2 r64, r61, r210;
}
{
add.f16x2 r67, %13, r64;
}
{
add.f16x2 r70, %14, %15;
}
{
mul.f16x2 r73, r70, r282;
}
{
add.f16x2 r76, r67, r73;
}
{
sub.f16x2 r79, %16, %10;
}
{
mul.f16x2 r82, r79, r228;
}
{
sub.f16x2 r85, %18, %19;
}
{
mul.f16x2 r88, r85, r300;
}
{
add.f16x2 r91, r82, r88;
}
{
add.f16x2 %8, r76, r91;
}
{
add.f16x2 r97, %11, %12;
}
{
mul.f16x2 r100, r97, r282;
}
{
add.f16x2 r103, %13, r100;
}
{
add.f16x2 r106, %14, %15;
}
{
mul.f16x2 r109, r106, r291;
}
{
add.f16x2 r112, r103, r109;
}
{
sub.f16x2 r115, %16, %10;
}
{
mul.f16x2 r118, r115, r300;
}
{
sub.f16x2 r121, %18, %19;
}
{
mul.f16x2 r124, r121, r306;
}
{
add.f16x2 r127, r118, r124;
}
{
sub.f16x2 %4, r112, r127;
}
{
add.f16x2 r133, %11, %12;
}
{
mul.f16x2 r136, r133, r282;
}
{
add.f16x2 r139, %13, r136;
}
{
add.f16x2 r142, %14, %15;
}
{
mul.f16x2 r145, r142, r291;
}
{
add.f16x2 r148, r139, r145;
}
{
sub.f16x2 r151, %16, %10;
}
{
mul.f16x2 r154, r151, r300;
}
{
sub.f16x2 r157, %18, %19;
}
{
mul.f16x2 r160, r157, r306;
}
{
add.f16x2 r163, r154, r160;
}
{
add.f16x2 %6, r148, r163;
}
{
add.f16x2 r169, %16, %10;
}
{
mul.f16x2 r172, r169, r210;
}
{
add.f16x2 r175, %17, r172;
}
{
add.f16x2 r178, %18, %19;
}
{
mul.f16x2 r181, r178, r282;
}
{
add.f16x2 r184, r175, r181;
}
{
sub.f16x2 r187, %11, %12;
}
{
mul.f16x2 r190, r187, r228;
}
{
sub.f16x2 r193, %14, %15;
}
{
mul.f16x2 r196, r193, r300;
}
{
add.f16x2 r199, r190, r196;
}
{
add.f16x2 %3, r184, r199;
}
{
add.f16x2 r205, %16, %10;
}
{
mul.f16x2 r208, r205, r210;
}
{
add.f16x2 r211, %17, r208;
}
{
add.f16x2 r214, %18, %19;
}
{
mul.f16x2 r217, r214, r282;
}
{
add.f16x2 r220, r211, r217;
}
{
sub.f16x2 r223, %11, %12;
}
{
mul.f16x2 r226, r223, r228;
}
{
sub.f16x2 r229, %14, %15;
}
{
mul.f16x2 r232, r229, r300;
}
{
add.f16x2 r235, r226, r232;
}
{
sub.f16x2 %9, r220, r235;
}
{
add.f16x2 r241, %16, %10;
}
{
mul.f16x2 r244, r241, r282;
}
{
add.f16x2 r247, %17, r244;
}
{
add.f16x2 r250, %18, %19;
}
{
mul.f16x2 r253, r250, r291;
}
{
add.f16x2 r256, r247, r253;
}
{
sub.f16x2 r259, %11, %12;
}
{
mul.f16x2 r262, r259, r300;
}
{
sub.f16x2 r265, %14, %15;
}
{
mul.f16x2 r268, r265, r306;
}
{
add.f16x2 r271, r262, r268;
}
{
add.f16x2 %5, r256, r271;
}
{
add.f16x2 r277, %16, %10;
}
{
mul.f16x2 r280, r277, r282;
}
{
add.f16x2 r283, %17, r280;
}
{
add.f16x2 r286, %18, %19;
}
{
mul.f16x2 r289, r286, r291;
}
{
add.f16x2 r292, r283, r289;
}
{
sub.f16x2 r295, %11, %12;
}
{
mul.f16x2 r298, r295, r300;
}
{
sub.f16x2 r301, %14, %15;
}
{
mul.f16x2 r304, r301, r306;
}
{
add.f16x2 r307, r298, r304;
}
{
sub.f16x2 %7, r292, r307;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)): "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].y)));
};


#endif
