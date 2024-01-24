#ifndef CUFFTDX_FFT_9_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_9_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<864, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<41>;
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
{
neg.f16 rs3, rs2;
}
mov.b32 r81, {rs3, rs3};
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
cvt.rn.f16.f64 rs5, fd27;
}
mov.b32 r156, {rs5, rs5};
{
cvt.rn.f16.f64 rs6, fd28;
}
{
neg.f16 rs7, rs6;
}
mov.b32 r165, {rs7, rs7};
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
cvt.rn.f16.f64 rs9, fd27;
}
mov.b32 r240, {rs9, rs9};
{
cvt.rn.f16.f64 rs10, fd28;
}
{
neg.f16 rs11, rs10;
}
mov.b32 r249, {rs11, rs11};
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
cvt.rn.f16.f64 rs13, fd7;
}
mov.f64 fd8, 0dBFE491B7523C161D;
{
cvt.rn.f16.f64 rs14, fd8;
}
mov.f64 fd9, 0d3FC63A1A7E0B738A;
{
cvt.rn.f16.f64 rs15, fd9;
}
mov.f64 fd10, 0dBFEF838B8C811C17;
{
cvt.rn.f16.f64 rs16, fd10;
}
mov.f64 fd13, 0dBFEE11F642522D1C;
{
cvt.rn.f16.f64 rs19, fd13;
}
mov.f64 fd14, 0dBFD5E3A8748A0BF5;
{
cvt.rn.f16.f64 rs20, fd14;
}
mov.b32 r267, {rs13, rs13};
{
mul.f16x2 r253, r112, r267;
}
mov.b32 r264, {rs14, rs14};
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
mov.b32 r299, {rs15, rs15};
{
mul.f16x2 r269, r196, r299;
}
mov.b32 r296, {rs16, rs16};
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
mov.b32 r315, {rs19, rs19};
{
mul.f16x2 r301, r214, r315;
}
mov.b32 r312, {rs20, rs20};
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
cvt.rn.f16.f64 rs29, fd27;
}
mov.b32 r388, {rs29, rs29};
{
cvt.rn.f16.f64 rs30, fd28;
}
{
neg.f16 rs31, rs30;
}
mov.b32 r397, {rs31, rs31};
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
cvt.rn.f16.f64 rs33, fd27;
}
mov.b32 r472, {rs33, rs33};
{
cvt.rn.f16.f64 rs34, fd28;
}
{
neg.f16 rs35, rs34;
}
mov.b32 r481, {rs35, rs35};
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
cvt.rn.f16.f64 rs37, fd27;
}
mov.b32 r556, {rs37, rs37};
{
cvt.rn.f16.f64 rs38, fd28;
}
{
neg.f16 rs39, rs38;
}
mov.b32 r565, {rs39, rs39};
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




template<> __forceinline__ __device__ void cufftdx_private_function<865, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<20>;
.reg .b32 r<265>;
.reg .b64 rd<4>;
mov.u32 r254, %tid.y;
mov.u32 r255, %6;
mad.lo.s32 r256, r254, 72, r255;
mov.u32 r257, %tid.x;
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
neg.f16x2 r3, r2;
}
{
add.f16x2 r5, %9, %11;
}
{
add.f16x2 r8, %7, r5;
}
{
add.f16x2 r11, %10, %12;
}
{
add.f16x2 r14, %8, r11;
}
{
add.f16x2 r17, %9, %11;
}
{
mul.f16x2 r20, r17, r1;
}
{
add.f16x2 r23, %7, r20;
}
{
sub.f16x2 r26, %10, %12;
}
{
mul.f16x2 r29, r26, r3;
}
{
add.f16x2 r32, r23, r29;
}
{
add.f16x2 r35, %9, %11;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %7, r38;
}
{
sub.f16x2 r44, %10, %12;
}
{
mul.f16x2 r47, r44, r3;
}
{
sub.f16x2 r50, r41, r47;
}
{
add.f16x2 r53, %10, %12;
}
{
mul.f16x2 r56, r53, r1;
}
{
add.f16x2 r59, %8, r56;
}
{
sub.f16x2 r62, %9, %11;
}
{
mul.f16x2 r65, r62, r3;
}
{
sub.f16x2 r68, r59, r65;
}
{
add.f16x2 r71, %10, %12;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %8, r74;
}
{
sub.f16x2 r80, %9, %11;
}
{
mul.f16x2 r83, r80, r3;
}
{
add.f16x2 r86, r77, r83;
}
mul.wide.u32 rd2, r257, -1431655765;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r258, rd3;
mul.lo.s32 r259, r258, 3;
sub.s32 r260, r257, r259;
mad.lo.s32 r261, r258, 72, r256;
cvt.rn.f32.u32 f17, r260;
mul.f32 f18, f17, 0f3F32B8C2;
cos.approx.f32 f5, f18;
sin.approx.f32 f19, f18;
neg.f32 f6, f19;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f5;
cvt.rn.f16.f32 high, f6;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r92, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r94, {high, high};
}
{
mul.f16x2 r96, r68, r94;
}
{
neg.f16x2 r99, r96;
}
{
fma.rn.f16x2 r101, r32, r92, r99;
}
{
mul.f16x2 r105, r32, r94;
}
{
fma.rn.f16x2 r108, r68, r92, r105;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r112, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r114, {high, high};
}
mov.f32 f9, 0fBF800000;
mov.f32 f10, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f9;
cvt.rn.f16.f32 high, f10;
mov.b32 r116, {low, high};
}
{
mul.f16x2 r117, r114, r116;
}
{
mul.f16x2 r120, r89, r112;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r123, {high, low};
}
{
fma.rn.f16x2 r125, r117, r123, r120;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r125;
mov.b32 r129, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r125;
mov.b32 r131, {high, high};
}
{
mul.f16x2 r133, r86, r131;
}
{
neg.f16x2 r136, r133;
}
{
fma.rn.f16x2 r138, r50, r129, r136;
}
{
mul.f16x2 r142, r50, r131;
}
{
fma.rn.f16x2 r145, r86, r129, r142;
}
barrier.sync 0;
mad.lo.s32 r262, r260, 24, r261;
st.shared.v2.f32 [r262], {r8, r14};
st.shared.v2.f32 [r262+8], {r101, r108};
st.shared.v2.f32 [r262+16], {r138, r145};
barrier.sync 0;
shl.b32 r263, r260, 4;
sub.s32 r264, r262, r263;
ld.shared.u32 r174, [r264];
ld.shared.u32 r180, [r264+4];
ld.shared.u32 r171, [r264+24];
ld.shared.u32 r177, [r264+28];
ld.shared.u32 r172, [r264+48];
ld.shared.u32 r178, [r264+52];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f14;
cvt.rn.f16.f32 high, f14;
mov.b32 r166, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f16;
cvt.rn.f16.f32 high, f16;
mov.b32 r167, {low, high};
}
{
neg.f16x2 r168, r167;
}
{
add.f16x2 r170, r171, r172;
}
{
add.f16x2 %0, r174, r170;
}
{
add.f16x2 r176, r177, r178;
}
{
add.f16x2 %1, r180, r176;
}
{
add.f16x2 r182, r171, r172;
}
{
mul.f16x2 r185, r182, r166;
}
{
add.f16x2 r188, r174, r185;
}
{
sub.f16x2 r191, r177, r178;
}
{
mul.f16x2 r194, r191, r168;
}
{
add.f16x2 %2, r188, r194;
}
{
add.f16x2 r200, r171, r172;
}
{
mul.f16x2 r203, r200, r166;
}
{
add.f16x2 r206, r174, r203;
}
{
sub.f16x2 r209, r177, r178;
}
{
mul.f16x2 r212, r209, r168;
}
{
sub.f16x2 %4, r206, r212;
}
{
add.f16x2 r218, r177, r178;
}
{
mul.f16x2 r221, r218, r166;
}
{
add.f16x2 r224, r180, r221;
}
{
sub.f16x2 r227, r171, r172;
}
{
mul.f16x2 r230, r227, r168;
}
{
sub.f16x2 %3, r224, r230;
}
{
add.f16x2 r236, r177, r178;
}
{
mul.f16x2 r239, r236, r166;
}
{
add.f16x2 r242, r180, r239;
}
{
sub.f16x2 r245, r171, r172;
}
{
mul.f16x2 r248, r245, r168;
}
{
add.f16x2 %5, r242, r248;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<866, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<20>;
.reg .b32 r<265>;
.reg .b64 rd<4>;
mov.u32 r254, %tid.y;
mov.u32 r255, %6;
mad.lo.s32 r256, r254, 36, r255;
mov.u32 r257, %tid.x;
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
neg.f16x2 r3, r2;
}
{
add.f16x2 r5, %9, %11;
}
{
add.f16x2 r8, %7, r5;
}
{
add.f16x2 r11, %10, %12;
}
{
add.f16x2 r14, %8, r11;
}
{
add.f16x2 r17, %9, %11;
}
{
mul.f16x2 r20, r17, r1;
}
{
add.f16x2 r23, %7, r20;
}
{
sub.f16x2 r26, %10, %12;
}
{
mul.f16x2 r29, r26, r3;
}
{
add.f16x2 r32, r23, r29;
}
{
add.f16x2 r35, %9, %11;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %7, r38;
}
{
sub.f16x2 r44, %10, %12;
}
{
mul.f16x2 r47, r44, r3;
}
{
sub.f16x2 r50, r41, r47;
}
{
add.f16x2 r53, %10, %12;
}
{
mul.f16x2 r56, r53, r1;
}
{
add.f16x2 r59, %8, r56;
}
{
sub.f16x2 r62, %9, %11;
}
{
mul.f16x2 r65, r62, r3;
}
{
sub.f16x2 r68, r59, r65;
}
{
add.f16x2 r71, %10, %12;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %8, r74;
}
{
sub.f16x2 r80, %9, %11;
}
{
mul.f16x2 r83, r80, r3;
}
{
add.f16x2 r86, r77, r83;
}
mul.wide.u32 rd2, r257, -1431655765;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r258, rd3;
mul.lo.s32 r259, r258, 3;
sub.s32 r260, r257, r259;
mad.lo.s32 r261, r258, 36, r256;
cvt.rn.f32.u32 f17, r260;
mul.f32 f18, f17, 0f3F32B8C2;
cos.approx.f32 f5, f18;
sin.approx.f32 f19, f18;
neg.f32 f6, f19;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f5;
cvt.rn.f16.f32 high, f6;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r92, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r94, {high, high};
}
{
mul.f16x2 r96, r68, r94;
}
{
neg.f16x2 r99, r96;
}
{
fma.rn.f16x2 r101, r32, r92, r99;
}
{
mul.f16x2 r105, r32, r94;
}
{
fma.rn.f16x2 r108, r68, r92, r105;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r112, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r114, {high, high};
}
mov.f32 f9, 0fBF800000;
mov.f32 f10, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f9;
cvt.rn.f16.f32 high, f10;
mov.b32 r116, {low, high};
}
{
mul.f16x2 r117, r114, r116;
}
{
mul.f16x2 r120, r89, r112;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r123, {high, low};
}
{
fma.rn.f16x2 r125, r117, r123, r120;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r125;
mov.b32 r129, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r125;
mov.b32 r131, {high, high};
}
{
mul.f16x2 r133, r86, r131;
}
{
neg.f16x2 r136, r133;
}
{
fma.rn.f16x2 r138, r50, r129, r136;
}
{
mul.f16x2 r142, r50, r131;
}
{
fma.rn.f16x2 r145, r86, r129, r142;
}
barrier.sync 0;
mad.lo.s32 r262, r260, 12, r261;
st.shared.u32 [r262], r8;
st.shared.u32 [r262+4], r101;
st.shared.u32 [r262+8], r138;
barrier.sync 0;
shl.b32 r263, r260, 3;
sub.s32 r264, r262, r263;
ld.shared.u32 r174, [r264];
ld.shared.u32 r171, [r264+12];
ld.shared.u32 r172, [r264+24];
barrier.sync 0;
st.shared.u32 [r262], r14;
st.shared.u32 [r262+4], r108;
st.shared.u32 [r262+8], r145;
barrier.sync 0;
ld.shared.u32 r180, [r264];
ld.shared.u32 r177, [r264+12];
ld.shared.u32 r178, [r264+24];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f14;
cvt.rn.f16.f32 high, f14;
mov.b32 r166, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f16;
cvt.rn.f16.f32 high, f16;
mov.b32 r167, {low, high};
}
{
neg.f16x2 r168, r167;
}
{
add.f16x2 r170, r171, r172;
}
{
add.f16x2 %0, r174, r170;
}
{
add.f16x2 r176, r177, r178;
}
{
add.f16x2 %1, r180, r176;
}
{
add.f16x2 r182, r171, r172;
}
{
mul.f16x2 r185, r182, r166;
}
{
add.f16x2 r188, r174, r185;
}
{
sub.f16x2 r191, r177, r178;
}
{
mul.f16x2 r194, r191, r168;
}
{
add.f16x2 %2, r188, r194;
}
{
add.f16x2 r200, r171, r172;
}
{
mul.f16x2 r203, r200, r166;
}
{
add.f16x2 r206, r174, r203;
}
{
sub.f16x2 r209, r177, r178;
}
{
mul.f16x2 r212, r209, r168;
}
{
sub.f16x2 %4, r206, r212;
}
{
add.f16x2 r218, r177, r178;
}
{
mul.f16x2 r221, r218, r166;
}
{
add.f16x2 r224, r180, r221;
}
{
sub.f16x2 r227, r171, r172;
}
{
mul.f16x2 r230, r227, r168;
}
{
sub.f16x2 %3, r224, r230;
}
{
add.f16x2 r236, r177, r178;
}
{
mul.f16x2 r239, r236, r166;
}
{
add.f16x2 r242, r180, r239;
}
{
sub.f16x2 r245, r171, r172;
}
{
mul.f16x2 r248, r245, r168;
}
{
add.f16x2 %5, r242, r248;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)));
};


#endif
