#ifndef CUFFTDX_FFT_12_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_12_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1145, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<51>;
.reg .b32 r<611>;
.reg .f64 fd<51>;
.reg .b64 rd<2>;
mov.f64 fd35, 0dBFE0000000000000;
{
cvt.rn.f16.f64 rs1, fd35;
}
mov.b32 r72, {rs1, rs1};
mov.f64 fd37, 0dBFEBB67AE8584CAA;
{
cvt.rn.f16.f64 rs2, fd37;
}
mov.b32 r81, {rs2, rs2};
{
add.f16x2 r1, %32, %40;
}
{
add.f16x2 r4, %24, r1;
}
{
add.f16x2 r7, %33, %41;
}
{
add.f16x2 r10, %25, r7;
}
{
add.f16x2 r13, %32, %40;
}
{
mul.f16x2 r16, r13, r72;
}
{
add.f16x2 r19, %24, r16;
}
{
sub.f16x2 r22, %33, %41;
}
{
mul.f16x2 r25, r22, r81;
}
{
add.f16x2 r28, r19, r25;
}
{
add.f16x2 r31, %32, %40;
}
{
mul.f16x2 r34, r31, r72;
}
{
add.f16x2 r37, %24, r34;
}
{
sub.f16x2 r40, %33, %41;
}
{
mul.f16x2 r43, r40, r81;
}
{
sub.f16x2 r46, r37, r43;
}
{
add.f16x2 r49, %33, %41;
}
{
mul.f16x2 r52, r49, r72;
}
{
add.f16x2 r55, %25, r52;
}
{
sub.f16x2 r58, %32, %40;
}
{
mul.f16x2 r61, r58, r81;
}
{
sub.f16x2 r64, r55, r61;
}
{
add.f16x2 r67, %33, %41;
}
{
mul.f16x2 r70, r67, r72;
}
{
add.f16x2 r73, %25, r70;
}
{
sub.f16x2 r76, %32, %40;
}
{
mul.f16x2 r79, r76, r81;
}
{
add.f16x2 r82, r73, r79;
}
{
cvt.rn.f16.f64 rs3, fd35;
}
mov.b32 r156, {rs3, rs3};
{
cvt.rn.f16.f64 rs4, fd37;
}
mov.b32 r165, {rs4, rs4};
{
add.f16x2 r85, %36, %44;
}
{
add.f16x2 r88, %28, r85;
}
{
add.f16x2 r91, %37, %45;
}
{
add.f16x2 r94, %29, r91;
}
{
add.f16x2 r97, %36, %44;
}
{
mul.f16x2 r100, r97, r156;
}
{
add.f16x2 r103, %28, r100;
}
{
sub.f16x2 r106, %37, %45;
}
{
mul.f16x2 r109, r106, r165;
}
{
add.f16x2 r112, r103, r109;
}
{
add.f16x2 r115, %36, %44;
}
{
mul.f16x2 r118, r115, r156;
}
{
add.f16x2 r121, %28, r118;
}
{
sub.f16x2 r124, %37, %45;
}
{
mul.f16x2 r127, r124, r165;
}
{
sub.f16x2 r130, r121, r127;
}
{
add.f16x2 r133, %37, %45;
}
{
mul.f16x2 r136, r133, r156;
}
{
add.f16x2 r139, %29, r136;
}
{
sub.f16x2 r142, %36, %44;
}
{
mul.f16x2 r145, r142, r165;
}
{
sub.f16x2 r148, r139, r145;
}
{
add.f16x2 r151, %37, %45;
}
{
mul.f16x2 r154, r151, r156;
}
{
add.f16x2 r157, %29, r154;
}
{
sub.f16x2 r160, %36, %44;
}
{
mul.f16x2 r163, r160, r165;
}
{
add.f16x2 r166, r157, r163;
}
mov.f64 fd38, 0d3FE0000000000000;
{
cvt.rn.f16.f64 rs5, fd38;
}
mov.f64 fd36, 0d3FEBB67AE8584CAA;
{
cvt.rn.f16.f64 rs6, fd36;
}
{
cvt.rn.f16.f64 rs7, fd35;
}
{
cvt.rn.f16.f64 rs8, fd36;
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
add.f16x2 r201, r4, r88;
}
{
add.f16x2 r204, r10, r94;
}
{
sub.f16x2 r207, r4, r88;
}
{
sub.f16x2 r210, r10, r94;
}
{
add.f16x2 r213, r28, r175;
}
{
add.f16x2 r216, r64, r181;
}
{
sub.f16x2 r219, r28, r175;
}
{
sub.f16x2 r222, r64, r181;
}
{
add.f16x2 r225, r46, r191;
}
{
add.f16x2 r228, r82, r197;
}
{
sub.f16x2 r231, r46, r191;
}
{
sub.f16x2 r234, r82, r197;
}
{
cvt.rn.f16.f64 rs15, fd35;
}
mov.b32 r308, {rs15, rs15};
{
cvt.rn.f16.f64 rs16, fd37;
}
mov.b32 r317, {rs16, rs16};
{
add.f16x2 r237, %34, %42;
}
{
add.f16x2 r240, %26, r237;
}
{
add.f16x2 r243, %35, %43;
}
{
add.f16x2 r246, %27, r243;
}
{
add.f16x2 r249, %34, %42;
}
{
mul.f16x2 r252, r249, r308;
}
{
add.f16x2 r255, %26, r252;
}
{
sub.f16x2 r258, %35, %43;
}
{
mul.f16x2 r261, r258, r317;
}
{
add.f16x2 r264, r255, r261;
}
{
add.f16x2 r267, %34, %42;
}
{
mul.f16x2 r270, r267, r308;
}
{
add.f16x2 r273, %26, r270;
}
{
sub.f16x2 r276, %35, %43;
}
{
mul.f16x2 r279, r276, r317;
}
{
sub.f16x2 r282, r273, r279;
}
{
add.f16x2 r285, %35, %43;
}
{
mul.f16x2 r288, r285, r308;
}
{
add.f16x2 r291, %27, r288;
}
{
sub.f16x2 r294, %34, %42;
}
{
mul.f16x2 r297, r294, r317;
}
{
sub.f16x2 r300, r291, r297;
}
{
add.f16x2 r303, %35, %43;
}
{
mul.f16x2 r306, r303, r308;
}
{
add.f16x2 r309, %27, r306;
}
{
sub.f16x2 r312, %34, %42;
}
{
mul.f16x2 r315, r312, r317;
}
{
add.f16x2 r318, r309, r315;
}
{
cvt.rn.f16.f64 rs17, fd35;
}
mov.b32 r392, {rs17, rs17};
{
cvt.rn.f16.f64 rs18, fd37;
}
mov.b32 r401, {rs18, rs18};
{
add.f16x2 r321, %38, %46;
}
{
add.f16x2 r324, %30, r321;
}
{
add.f16x2 r327, %39, %47;
}
{
add.f16x2 r330, %31, r327;
}
{
add.f16x2 r333, %38, %46;
}
{
mul.f16x2 r336, r333, r392;
}
{
add.f16x2 r339, %30, r336;
}
{
sub.f16x2 r342, %39, %47;
}
{
mul.f16x2 r345, r342, r401;
}
{
add.f16x2 r348, r339, r345;
}
{
add.f16x2 r351, %38, %46;
}
{
mul.f16x2 r354, r351, r392;
}
{
add.f16x2 r357, %30, r354;
}
{
sub.f16x2 r360, %39, %47;
}
{
mul.f16x2 r363, r360, r401;
}
{
sub.f16x2 r366, r357, r363;
}
{
add.f16x2 r369, %39, %47;
}
{
mul.f16x2 r372, r369, r392;
}
{
add.f16x2 r375, %31, r372;
}
{
sub.f16x2 r378, %38, %46;
}
{
mul.f16x2 r381, r378, r401;
}
{
sub.f16x2 r384, r375, r381;
}
{
add.f16x2 r387, %39, %47;
}
{
mul.f16x2 r390, r387, r392;
}
{
add.f16x2 r393, %31, r390;
}
{
sub.f16x2 r396, %38, %46;
}
{
mul.f16x2 r399, r396, r401;
}
{
add.f16x2 r402, r393, r399;
}
{
cvt.rn.f16.f64 rs19, fd38;
}
{
cvt.rn.f16.f64 rs20, fd36;
}
{
cvt.rn.f16.f64 rs21, fd35;
}
{
cvt.rn.f16.f64 rs22, fd36;
}
mov.b32 r419, {rs19, rs19};
{
mul.f16x2 r405, r348, r419;
}
mov.b32 r416, {rs20, rs20};
{
mul.f16x2 r408, r384, r416;
}
{
sub.f16x2 r411, r405, r408;
}
{
mul.f16x2 r414, r348, r416;
}
{
fma.rn.f16x2 r417, r384, r419, r414;
}
mov.b32 r435, {rs21, rs21};
{
mul.f16x2 r421, r366, r435;
}
mov.b32 r432, {rs22, rs22};
{
mul.f16x2 r424, r402, r432;
}
{
sub.f16x2 r427, r421, r424;
}
{
mul.f16x2 r430, r366, r432;
}
{
fma.rn.f16x2 r433, r402, r435, r430;
}
{
add.f16x2 r437, r240, r324;
}
{
add.f16x2 r440, r246, r330;
}
{
sub.f16x2 r443, r240, r324;
}
{
sub.f16x2 r446, r246, r330;
}
{
add.f16x2 r449, r264, r411;
}
{
add.f16x2 r452, r300, r417;
}
{
sub.f16x2 r455, r264, r411;
}
{
sub.f16x2 r458, r300, r417;
}
{
add.f16x2 r461, r282, r427;
}
{
add.f16x2 r464, r318, r433;
}
{
sub.f16x2 r467, r282, r427;
}
{
sub.f16x2 r470, r318, r433;
}
{
cvt.rn.f16.f64 rs29, fd36;
}
{
cvt.rn.f16.f64 rs30, fd38;
}
{
cvt.rn.f16.f64 rs31, fd38;
}
{
cvt.rn.f16.f64 rs32, fd36;
}
{
cvt.rn.f16.f64 rs35, fd35;
}
{
cvt.rn.f16.f64 rs36, fd36;
}
{
cvt.rn.f16.f64 rs37, fd37;
}
{
cvt.rn.f16.f64 rs38, fd38;
}
mov.b32 r487, {rs29, rs29};
{
mul.f16x2 r473, r449, r487;
}
mov.b32 r484, {rs30, rs30};
{
mul.f16x2 r476, r452, r484;
}
{
sub.f16x2 r479, r473, r476;
}
{
mul.f16x2 r482, r449, r484;
}
{
fma.rn.f16x2 r485, r452, r487, r482;
}
mov.b32 r503, {rs31, rs31};
{
mul.f16x2 r489, r461, r503;
}
mov.b32 r500, {rs32, rs32};
{
mul.f16x2 r492, r464, r500;
}
{
sub.f16x2 r495, r489, r492;
}
{
mul.f16x2 r498, r461, r500;
}
{
fma.rn.f16x2 r501, r464, r503, r498;
}
{
neg.f16x2 r505, r446;
}
mov.b32 r521, {rs35, rs35};
{
mul.f16x2 r507, r455, r521;
}
mov.b32 r518, {rs36, rs36};
{
mul.f16x2 r510, r458, r518;
}
{
sub.f16x2 r513, r507, r510;
}
{
mul.f16x2 r516, r455, r518;
}
{
fma.rn.f16x2 r519, r458, r521, r516;
}
mov.b32 r537, {rs37, rs37};
{
mul.f16x2 r523, r467, r537;
}
mov.b32 r534, {rs38, rs38};
{
mul.f16x2 r526, r470, r534;
}
{
sub.f16x2 r529, r523, r526;
}
{
mul.f16x2 r532, r467, r534;
}
{
fma.rn.f16x2 r535, r470, r537, r532;
}
{
add.f16x2 %0, r201, r437;
}
{
add.f16x2 %1, r204, r440;
}
{
sub.f16x2 %12, r201, r437;
}
{
sub.f16x2 %13, r204, r440;
}
{
add.f16x2 %2, r213, r479;
}
{
add.f16x2 %3, r216, r485;
}
{
sub.f16x2 %14, r213, r479;
}
{
sub.f16x2 %15, r216, r485;
}
{
add.f16x2 %4, r225, r495;
}
{
add.f16x2 %5, r228, r501;
}
{
sub.f16x2 %16, r225, r495;
}
{
sub.f16x2 %17, r228, r501;
}
{
add.f16x2 %6, r207, r505;
}
{
add.f16x2 %7, r210, r443;
}
{
sub.f16x2 %18, r207, r505;
}
{
sub.f16x2 %19, r210, r443;
}
{
add.f16x2 %8, r219, r513;
}
{
add.f16x2 %9, r222, r519;
}
{
sub.f16x2 %20, r219, r513;
}
{
sub.f16x2 %21, r222, r519;
}
{
add.f16x2 %10, r231, r529;
}
{
add.f16x2 %11, r234, r535;
}
{
sub.f16x2 %22, r231, r529;
}
{
sub.f16x2 %23, r234, r535;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)));
};


#endif
