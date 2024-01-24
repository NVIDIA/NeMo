#ifndef CUFFTDX_FFT_9_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_9_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1066, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<29>;
.reg .b32 r<569>;
.reg .f64 fd<29>;
.reg .b64 rd<2>;
mov.f64 fd27, 0dBFE0000000000000;
{
cvt.rn.f16.f64 rs1, fd27;
}
mov.b32 r72, {rs1, rs1};
mov.f64 fd28, 0dBFEBB67AE8584CAA;
{
cvt.rn.f16.f64 rs2, fd28;
}
mov.b32 r81, {rs2, rs2};
{
add.f16x2 r1, %24, %30;
}
{
add.f16x2 r4, %18, r1;
}
{
add.f16x2 r7, %25, %31;
}
{
add.f16x2 r10, %19, r7;
}
{
add.f16x2 r13, %24, %30;
}
{
mul.f16x2 r16, r13, r72;
}
{
add.f16x2 r19, %18, r16;
}
{
sub.f16x2 r22, %25, %31;
}
{
mul.f16x2 r25, r22, r81;
}
{
add.f16x2 r28, r19, r25;
}
{
add.f16x2 r31, %24, %30;
}
{
mul.f16x2 r34, r31, r72;
}
{
add.f16x2 r37, %18, r34;
}
{
sub.f16x2 r40, %25, %31;
}
{
mul.f16x2 r43, r40, r81;
}
{
sub.f16x2 r46, r37, r43;
}
{
add.f16x2 r49, %25, %31;
}
{
mul.f16x2 r52, r49, r72;
}
{
add.f16x2 r55, %19, r52;
}
{
sub.f16x2 r58, %24, %30;
}
{
mul.f16x2 r61, r58, r81;
}
{
sub.f16x2 r64, r55, r61;
}
{
add.f16x2 r67, %25, %31;
}
{
mul.f16x2 r70, r67, r72;
}
{
add.f16x2 r73, %19, r70;
}
{
sub.f16x2 r76, %24, %30;
}
{
mul.f16x2 r79, r76, r81;
}
{
add.f16x2 r82, r73, r79;
}
{
cvt.rn.f16.f64 rs3, fd27;
}
mov.b32 r156, {rs3, rs3};
{
cvt.rn.f16.f64 rs4, fd28;
}
mov.b32 r165, {rs4, rs4};
{
add.f16x2 r85, %26, %32;
}
{
add.f16x2 r88, %20, r85;
}
{
add.f16x2 r91, %27, %33;
}
{
add.f16x2 r94, %21, r91;
}
{
add.f16x2 r97, %26, %32;
}
{
mul.f16x2 r100, r97, r156;
}
{
add.f16x2 r103, %20, r100;
}
{
sub.f16x2 r106, %27, %33;
}
{
mul.f16x2 r109, r106, r165;
}
{
add.f16x2 r112, r103, r109;
}
{
add.f16x2 r115, %26, %32;
}
{
mul.f16x2 r118, r115, r156;
}
{
add.f16x2 r121, %20, r118;
}
{
sub.f16x2 r124, %27, %33;
}
{
mul.f16x2 r127, r124, r165;
}
{
sub.f16x2 r130, r121, r127;
}
{
add.f16x2 r133, %27, %33;
}
{
mul.f16x2 r136, r133, r156;
}
{
add.f16x2 r139, %21, r136;
}
{
sub.f16x2 r142, %26, %32;
}
{
mul.f16x2 r145, r142, r165;
}
{
sub.f16x2 r148, r139, r145;
}
{
add.f16x2 r151, %27, %33;
}
{
mul.f16x2 r154, r151, r156;
}
{
add.f16x2 r157, %21, r154;
}
{
sub.f16x2 r160, %26, %32;
}
{
mul.f16x2 r163, r160, r165;
}
{
add.f16x2 r166, r157, r163;
}
{
cvt.rn.f16.f64 rs5, fd27;
}
mov.b32 r240, {rs5, rs5};
{
cvt.rn.f16.f64 rs6, fd28;
}
mov.b32 r249, {rs6, rs6};
{
add.f16x2 r169, %28, %34;
}
{
add.f16x2 r172, %22, r169;
}
{
add.f16x2 r175, %29, %35;
}
{
add.f16x2 r178, %23, r175;
}
{
add.f16x2 r181, %28, %34;
}
{
mul.f16x2 r184, r181, r240;
}
{
add.f16x2 r187, %22, r184;
}
{
sub.f16x2 r190, %29, %35;
}
{
mul.f16x2 r193, r190, r249;
}
{
add.f16x2 r196, r187, r193;
}
{
add.f16x2 r199, %28, %34;
}
{
mul.f16x2 r202, r199, r240;
}
{
add.f16x2 r205, %22, r202;
}
{
sub.f16x2 r208, %29, %35;
}
{
mul.f16x2 r211, r208, r249;
}
{
sub.f16x2 r214, r205, r211;
}
{
add.f16x2 r217, %29, %35;
}
{
mul.f16x2 r220, r217, r240;
}
{
add.f16x2 r223, %23, r220;
}
{
sub.f16x2 r226, %28, %34;
}
{
mul.f16x2 r229, r226, r249;
}
{
sub.f16x2 r232, r223, r229;
}
{
add.f16x2 r235, %29, %35;
}
{
mul.f16x2 r238, r235, r240;
}
{
add.f16x2 r241, %23, r238;
}
{
sub.f16x2 r244, %28, %34;
}
{
mul.f16x2 r247, r244, r249;
}
{
add.f16x2 r250, r241, r247;
}
mov.f64 fd7, 0d3FE8836FA2CF5039;
{
cvt.rn.f16.f64 rs7, fd7;
}
mov.f64 fd8, 0d3FE491B7523C161D;
{
cvt.rn.f16.f64 rs8, fd8;
}
mov.f64 fd9, 0d3FC63A1A7E0B738A;
{
cvt.rn.f16.f64 rs9, fd9;
}
mov.f64 fd10, 0d3FEF838B8C811C17;
{
cvt.rn.f16.f64 rs10, fd10;
}
mov.f64 fd13, 0dBFEE11F642522D1C;
{
cvt.rn.f16.f64 rs13, fd13;
}
mov.f64 fd14, 0d3FD5E3A8748A0BF5;
{
cvt.rn.f16.f64 rs14, fd14;
}
mov.b32 r267, {rs7, rs7};
{
mul.f16x2 r253, r112, r267;
}
mov.b32 r264, {rs8, rs8};
{
mul.f16x2 r256, r148, r264;
}
{
sub.f16x2 r259, r253, r256;
}
{
mul.f16x2 r262, r112, r264;
}
{
fma.rn.f16x2 r265, r148, r267, r262;
}
mov.b32 r299, {rs9, rs9};
{
mul.f16x2 r269, r196, r299;
}
mov.b32 r296, {rs10, rs10};
{
mul.f16x2 r272, r232, r296;
}
{
sub.f16x2 r275, r269, r272;
}
{
mul.f16x2 r278, r196, r296;
}
{
fma.rn.f16x2 r281, r232, r299, r278;
}
{
mul.f16x2 r285, r130, r299;
}
{
mul.f16x2 r288, r166, r296;
}
{
sub.f16x2 r291, r285, r288;
}
{
mul.f16x2 r294, r130, r296;
}
{
fma.rn.f16x2 r297, r166, r299, r294;
}
mov.b32 r315, {rs13, rs13};
{
mul.f16x2 r301, r214, r315;
}
mov.b32 r312, {rs14, rs14};
{
mul.f16x2 r304, r250, r312;
}
{
sub.f16x2 r307, r301, r304;
}
{
mul.f16x2 r310, r214, r312;
}
{
fma.rn.f16x2 r313, r250, r315, r310;
}
{
cvt.rn.f16.f64 rs23, fd27;
}
mov.b32 r388, {rs23, rs23};
{
cvt.rn.f16.f64 rs24, fd28;
}
mov.b32 r397, {rs24, rs24};
{
add.f16x2 r317, r88, r172;
}
{
add.f16x2 %0, r4, r317;
}
{
add.f16x2 r323, r94, r178;
}
{
add.f16x2 %1, r10, r323;
}
{
add.f16x2 r329, r88, r172;
}
{
mul.f16x2 r332, r329, r388;
}
{
add.f16x2 r335, r4, r332;
}
{
sub.f16x2 r338, r94, r178;
}
{
mul.f16x2 r341, r338, r397;
}
{
add.f16x2 %6, r335, r341;
}
{
add.f16x2 r347, r88, r172;
}
{
mul.f16x2 r350, r347, r388;
}
{
add.f16x2 r353, r4, r350;
}
{
sub.f16x2 r356, r94, r178;
}
{
mul.f16x2 r359, r356, r397;
}
{
sub.f16x2 %12, r353, r359;
}
{
add.f16x2 r365, r94, r178;
}
{
mul.f16x2 r368, r365, r388;
}
{
add.f16x2 r371, r10, r368;
}
{
sub.f16x2 r374, r88, r172;
}
{
mul.f16x2 r377, r374, r397;
}
{
sub.f16x2 %7, r371, r377;
}
{
add.f16x2 r383, r94, r178;
}
{
mul.f16x2 r386, r383, r388;
}
{
add.f16x2 r389, r10, r386;
}
{
sub.f16x2 r392, r88, r172;
}
{
mul.f16x2 r395, r392, r397;
}
{
add.f16x2 %13, r389, r395;
}
{
cvt.rn.f16.f64 rs25, fd27;
}
mov.b32 r472, {rs25, rs25};
{
cvt.rn.f16.f64 rs26, fd28;
}
mov.b32 r481, {rs26, rs26};
{
add.f16x2 r401, r259, r275;
}
{
add.f16x2 %2, r28, r401;
}
{
add.f16x2 r407, r265, r281;
}
{
add.f16x2 %3, r64, r407;
}
{
add.f16x2 r413, r259, r275;
}
{
mul.f16x2 r416, r413, r472;
}
{
add.f16x2 r419, r28, r416;
}
{
sub.f16x2 r422, r265, r281;
}
{
mul.f16x2 r425, r422, r481;
}
{
add.f16x2 %8, r419, r425;
}
{
add.f16x2 r431, r259, r275;
}
{
mul.f16x2 r434, r431, r472;
}
{
add.f16x2 r437, r28, r434;
}
{
sub.f16x2 r440, r265, r281;
}
{
mul.f16x2 r443, r440, r481;
}
{
sub.f16x2 %14, r437, r443;
}
{
add.f16x2 r449, r265, r281;
}
{
mul.f16x2 r452, r449, r472;
}
{
add.f16x2 r455, r64, r452;
}
{
sub.f16x2 r458, r259, r275;
}
{
mul.f16x2 r461, r458, r481;
}
{
sub.f16x2 %9, r455, r461;
}
{
add.f16x2 r467, r265, r281;
}
{
mul.f16x2 r470, r467, r472;
}
{
add.f16x2 r473, r64, r470;
}
{
sub.f16x2 r476, r259, r275;
}
{
mul.f16x2 r479, r476, r481;
}
{
add.f16x2 %15, r473, r479;
}
{
cvt.rn.f16.f64 rs27, fd27;
}
mov.b32 r556, {rs27, rs27};
{
cvt.rn.f16.f64 rs28, fd28;
}
mov.b32 r565, {rs28, rs28};
{
add.f16x2 r485, r291, r307;
}
{
add.f16x2 %4, r46, r485;
}
{
add.f16x2 r491, r297, r313;
}
{
add.f16x2 %5, r82, r491;
}
{
add.f16x2 r497, r291, r307;
}
{
mul.f16x2 r500, r497, r556;
}
{
add.f16x2 r503, r46, r500;
}
{
sub.f16x2 r506, r297, r313;
}
{
mul.f16x2 r509, r506, r565;
}
{
add.f16x2 %10, r503, r509;
}
{
add.f16x2 r515, r291, r307;
}
{
mul.f16x2 r518, r515, r556;
}
{
add.f16x2 r521, r46, r518;
}
{
sub.f16x2 r524, r297, r313;
}
{
mul.f16x2 r527, r524, r565;
}
{
sub.f16x2 %16, r521, r527;
}
{
add.f16x2 r533, r297, r313;
}
{
mul.f16x2 r536, r533, r556;
}
{
add.f16x2 r539, r82, r536;
}
{
sub.f16x2 r542, r291, r307;
}
{
mul.f16x2 r545, r542, r565;
}
{
sub.f16x2 %11, r539, r545;
}
{
add.f16x2 r551, r297, r313;
}
{
mul.f16x2 r554, r551, r556;
}
{
add.f16x2 r557, r82, r554;
}
{
sub.f16x2 r560, r291, r307;
}
{
mul.f16x2 r563, r560, r565;
}
{
add.f16x2 %17, r557, r563;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1067, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<20>;
.reg .b32 r<261>;
.reg .b64 rd<4>;
mov.u32 r250, %tid.y;
mov.u32 r251, %6;
mad.lo.s32 r252, r250, 72, r251;
mov.u32 r253, %tid.x;
mov.f32 f14, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f14;
cvt.rn.f16.f32 high, f14;
mov.b32 r1, {low, high};
}
mov.f32 f16, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f16;
cvt.rn.f16.f32 high, f16;
mov.b32 r2, {low, high};
}
{
add.f16x2 r3, %9, %11;
}
{
add.f16x2 r6, %7, r3;
}
{
add.f16x2 r9, %10, %12;
}
{
add.f16x2 r12, %8, r9;
}
{
add.f16x2 r15, %9, %11;
}
{
mul.f16x2 r18, r15, r1;
}
{
add.f16x2 r21, %7, r18;
}
{
sub.f16x2 r24, %10, %12;
}
{
mul.f16x2 r27, r24, r2;
}
{
add.f16x2 r30, r21, r27;
}
{
add.f16x2 r33, %9, %11;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %7, r36;
}
{
sub.f16x2 r42, %10, %12;
}
{
mul.f16x2 r45, r42, r2;
}
{
sub.f16x2 r48, r39, r45;
}
{
add.f16x2 r51, %10, %12;
}
{
mul.f16x2 r54, r51, r1;
}
{
add.f16x2 r57, %8, r54;
}
{
sub.f16x2 r60, %9, %11;
}
{
mul.f16x2 r63, r60, r2;
}
{
sub.f16x2 r66, r57, r63;
}
{
add.f16x2 r69, %10, %12;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %8, r72;
}
{
sub.f16x2 r78, %9, %11;
}
{
mul.f16x2 r81, r78, r2;
}
{
add.f16x2 r84, r75, r81;
}
mul.wide.u32 rd2, r253, -1431655765;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r254, rd3;
mul.lo.s32 r255, r254, 3;
sub.s32 r256, r253, r255;
mad.lo.s32 r257, r254, 72, r252;
cvt.rn.f32.u32 f17, r256;
mul.f32 f18, f17, 0f3F32B8C2;
cos.approx.f32 f5, f18;
sin.approx.f32 f19, f18;
neg.f32 f6, f19;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f5;
cvt.rn.f16.f32 high, f6;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r90, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r92, {high, high};
}
{
mul.f16x2 r94, r66, r92;
}
{
fma.rn.f16x2 r97, r30, r90, r94;
}
{
mul.f16x2 r101, r30, r92;
}
{
neg.f16x2 r104, r101;
}
{
fma.rn.f16x2 r106, r66, r90, r104;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r110, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r112, {high, high};
}
mov.f32 f9, 0fBF800000;
mov.f32 f10, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f9;
cvt.rn.f16.f32 high, f10;
mov.b32 r114, {low, high};
}
{
mul.f16x2 r115, r112, r114;
}
{
mul.f16x2 r118, r87, r110;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r121, {high, low};
}
{
fma.rn.f16x2 r123, r115, r121, r118;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r123;
mov.b32 r127, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r123;
mov.b32 r129, {high, high};
}
{
mul.f16x2 r131, r84, r129;
}
{
fma.rn.f16x2 r134, r48, r127, r131;
}
{
mul.f16x2 r138, r48, r129;
}
{
neg.f16x2 r141, r138;
}
{
fma.rn.f16x2 r143, r84, r127, r141;
}
barrier.sync 0;
mad.lo.s32 r258, r256, 24, r257;
st.shared.v2.f32 [r258], {r6, r12};
st.shared.v2.f32 [r258+8], {r97, r106};
st.shared.v2.f32 [r258+16], {r134, r143};
barrier.sync 0;
shl.b32 r259, r256, 4;
sub.s32 r260, r258, r259;
ld.shared.u32 r170, [r260];
ld.shared.u32 r176, [r260+4];
ld.shared.u32 r167, [r260+24];
ld.shared.u32 r173, [r260+28];
ld.shared.u32 r168, [r260+48];
ld.shared.u32 r174, [r260+52];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f14;
cvt.rn.f16.f32 high, f14;
mov.b32 r164, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f16;
cvt.rn.f16.f32 high, f16;
mov.b32 r165, {low, high};
}
{
add.f16x2 r166, r167, r168;
}
{
add.f16x2 %0, r170, r166;
}
{
add.f16x2 r172, r173, r174;
}
{
add.f16x2 %1, r176, r172;
}
{
add.f16x2 r178, r167, r168;
}
{
mul.f16x2 r181, r178, r164;
}
{
add.f16x2 r184, r170, r181;
}
{
sub.f16x2 r187, r173, r174;
}
{
mul.f16x2 r190, r187, r165;
}
{
add.f16x2 %2, r184, r190;
}
{
add.f16x2 r196, r167, r168;
}
{
mul.f16x2 r199, r196, r164;
}
{
add.f16x2 r202, r170, r199;
}
{
sub.f16x2 r205, r173, r174;
}
{
mul.f16x2 r208, r205, r165;
}
{
sub.f16x2 %4, r202, r208;
}
{
add.f16x2 r214, r173, r174;
}
{
mul.f16x2 r217, r214, r164;
}
{
add.f16x2 r220, r176, r217;
}
{
sub.f16x2 r223, r167, r168;
}
{
mul.f16x2 r226, r223, r165;
}
{
sub.f16x2 %3, r220, r226;
}
{
add.f16x2 r232, r173, r174;
}
{
mul.f16x2 r235, r232, r164;
}
{
add.f16x2 r238, r176, r235;
}
{
sub.f16x2 r241, r167, r168;
}
{
mul.f16x2 r244, r241, r165;
}
{
add.f16x2 %5, r238, r244;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1068, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<20>;
.reg .b32 r<261>;
.reg .b64 rd<4>;
mov.u32 r250, %tid.y;
mov.u32 r251, %6;
mad.lo.s32 r252, r250, 36, r251;
mov.u32 r253, %tid.x;
mov.f32 f14, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f14;
cvt.rn.f16.f32 high, f14;
mov.b32 r1, {low, high};
}
mov.f32 f16, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f16;
cvt.rn.f16.f32 high, f16;
mov.b32 r2, {low, high};
}
{
add.f16x2 r3, %9, %11;
}
{
add.f16x2 r6, %7, r3;
}
{
add.f16x2 r9, %10, %12;
}
{
add.f16x2 r12, %8, r9;
}
{
add.f16x2 r15, %9, %11;
}
{
mul.f16x2 r18, r15, r1;
}
{
add.f16x2 r21, %7, r18;
}
{
sub.f16x2 r24, %10, %12;
}
{
mul.f16x2 r27, r24, r2;
}
{
add.f16x2 r30, r21, r27;
}
{
add.f16x2 r33, %9, %11;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %7, r36;
}
{
sub.f16x2 r42, %10, %12;
}
{
mul.f16x2 r45, r42, r2;
}
{
sub.f16x2 r48, r39, r45;
}
{
add.f16x2 r51, %10, %12;
}
{
mul.f16x2 r54, r51, r1;
}
{
add.f16x2 r57, %8, r54;
}
{
sub.f16x2 r60, %9, %11;
}
{
mul.f16x2 r63, r60, r2;
}
{
sub.f16x2 r66, r57, r63;
}
{
add.f16x2 r69, %10, %12;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %8, r72;
}
{
sub.f16x2 r78, %9, %11;
}
{
mul.f16x2 r81, r78, r2;
}
{
add.f16x2 r84, r75, r81;
}
mul.wide.u32 rd2, r253, -1431655765;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r254, rd3;
mul.lo.s32 r255, r254, 3;
sub.s32 r256, r253, r255;
mad.lo.s32 r257, r254, 36, r252;
cvt.rn.f32.u32 f17, r256;
mul.f32 f18, f17, 0f3F32B8C2;
cos.approx.f32 f5, f18;
sin.approx.f32 f19, f18;
neg.f32 f6, f19;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f5;
cvt.rn.f16.f32 high, f6;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r90, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r92, {high, high};
}
{
mul.f16x2 r94, r66, r92;
}
{
fma.rn.f16x2 r97, r30, r90, r94;
}
{
mul.f16x2 r101, r30, r92;
}
{
neg.f16x2 r104, r101;
}
{
fma.rn.f16x2 r106, r66, r90, r104;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r110, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r112, {high, high};
}
mov.f32 f9, 0fBF800000;
mov.f32 f10, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f9;
cvt.rn.f16.f32 high, f10;
mov.b32 r114, {low, high};
}
{
mul.f16x2 r115, r112, r114;
}
{
mul.f16x2 r118, r87, r110;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r121, {high, low};
}
{
fma.rn.f16x2 r123, r115, r121, r118;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r123;
mov.b32 r127, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r123;
mov.b32 r129, {high, high};
}
{
mul.f16x2 r131, r84, r129;
}
{
fma.rn.f16x2 r134, r48, r127, r131;
}
{
mul.f16x2 r138, r48, r129;
}
{
neg.f16x2 r141, r138;
}
{
fma.rn.f16x2 r143, r84, r127, r141;
}
barrier.sync 0;
mad.lo.s32 r258, r256, 12, r257;
st.shared.u32 [r258], r6;
st.shared.u32 [r258+4], r97;
st.shared.u32 [r258+8], r134;
barrier.sync 0;
shl.b32 r259, r256, 3;
sub.s32 r260, r258, r259;
ld.shared.u32 r170, [r260];
ld.shared.u32 r167, [r260+12];
ld.shared.u32 r168, [r260+24];
barrier.sync 0;
st.shared.u32 [r258], r12;
st.shared.u32 [r258+4], r106;
st.shared.u32 [r258+8], r143;
barrier.sync 0;
ld.shared.u32 r176, [r260];
ld.shared.u32 r173, [r260+12];
ld.shared.u32 r174, [r260+24];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f14;
cvt.rn.f16.f32 high, f14;
mov.b32 r164, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f16;
cvt.rn.f16.f32 high, f16;
mov.b32 r165, {low, high};
}
{
add.f16x2 r166, r167, r168;
}
{
add.f16x2 %0, r170, r166;
}
{
add.f16x2 r172, r173, r174;
}
{
add.f16x2 %1, r176, r172;
}
{
add.f16x2 r178, r167, r168;
}
{
mul.f16x2 r181, r178, r164;
}
{
add.f16x2 r184, r170, r181;
}
{
sub.f16x2 r187, r173, r174;
}
{
mul.f16x2 r190, r187, r165;
}
{
add.f16x2 %2, r184, r190;
}
{
add.f16x2 r196, r167, r168;
}
{
mul.f16x2 r199, r196, r164;
}
{
add.f16x2 r202, r170, r199;
}
{
sub.f16x2 r205, r173, r174;
}
{
mul.f16x2 r208, r205, r165;
}
{
sub.f16x2 %4, r202, r208;
}
{
add.f16x2 r214, r173, r174;
}
{
mul.f16x2 r217, r214, r164;
}
{
add.f16x2 r220, r176, r217;
}
{
sub.f16x2 r223, r167, r168;
}
{
mul.f16x2 r226, r223, r165;
}
{
sub.f16x2 %3, r220, r226;
}
{
add.f16x2 r232, r173, r174;
}
{
mul.f16x2 r235, r232, r164;
}
{
add.f16x2 r238, r176, r235;
}
{
sub.f16x2 r241, r167, r168;
}
{
mul.f16x2 r244, r241, r165;
}
{
add.f16x2 %5, r238, r244;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)));
};


#endif
