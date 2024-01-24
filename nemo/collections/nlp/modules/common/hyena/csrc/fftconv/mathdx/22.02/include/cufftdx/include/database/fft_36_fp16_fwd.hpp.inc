#ifndef CUFFTDX_FFT_36_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_36_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<931, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<74>;
.reg .b32 r<710>;
.reg .b64 rd<5>;
mov.u32 r697, %tid.y;
shl.b32 r698, r697, 1;
mov.u32 r699, %12;
mad.lo.s32 r700, r698, 144, r699;
mov.u32 r701, %tid.x;
mov.f32 f56, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r1, {low, high};
}
mov.f32 f58, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r2, {low, high};
}
{
neg.f16x2 r3, r2;
}
{
add.f16x2 r5, %17, %21;
}
{
add.f16x2 r8, %13, r5;
}
{
add.f16x2 r11, %18, %22;
}
{
add.f16x2 r14, %14, r11;
}
{
add.f16x2 r17, %17, %21;
}
{
mul.f16x2 r20, r17, r1;
}
{
add.f16x2 r23, %13, r20;
}
{
sub.f16x2 r26, %18, %22;
}
{
mul.f16x2 r29, r26, r3;
}
{
add.f16x2 r32, r23, r29;
}
{
add.f16x2 r35, %17, %21;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %13, r38;
}
{
sub.f16x2 r44, %18, %22;
}
{
mul.f16x2 r47, r44, r3;
}
{
sub.f16x2 r50, r41, r47;
}
{
add.f16x2 r53, %18, %22;
}
{
mul.f16x2 r56, r53, r1;
}
{
add.f16x2 r59, %14, r56;
}
{
sub.f16x2 r62, %17, %21;
}
{
mul.f16x2 r65, r62, r3;
}
{
sub.f16x2 r68, r59, r65;
}
{
add.f16x2 r71, %18, %22;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %14, r74;
}
{
sub.f16x2 r80, %17, %21;
}
{
mul.f16x2 r83, r80, r3;
}
{
add.f16x2 r86, r77, r83;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r90, {low, high};
}
{
neg.f16x2 r91, r90;
}
{
add.f16x2 r93, %19, %23;
}
{
add.f16x2 r96, %15, r93;
}
{
add.f16x2 r99, %20, %24;
}
{
add.f16x2 r102, %16, r99;
}
{
add.f16x2 r105, %19, %23;
}
{
mul.f16x2 r108, r105, r89;
}
{
add.f16x2 r111, %15, r108;
}
{
sub.f16x2 r114, %20, %24;
}
{
mul.f16x2 r117, r114, r91;
}
{
add.f16x2 r120, r111, r117;
}
{
add.f16x2 r123, %19, %23;
}
{
mul.f16x2 r126, r123, r89;
}
{
add.f16x2 r129, %15, r126;
}
{
sub.f16x2 r132, %20, %24;
}
{
mul.f16x2 r135, r132, r91;
}
{
sub.f16x2 r138, r129, r135;
}
{
add.f16x2 r141, %20, %24;
}
{
mul.f16x2 r144, r141, r89;
}
{
add.f16x2 r147, %16, r144;
}
{
sub.f16x2 r150, %19, %23;
}
{
mul.f16x2 r153, r150, r91;
}
{
sub.f16x2 r156, r147, r153;
}
{
add.f16x2 r159, %20, %24;
}
{
mul.f16x2 r162, r159, r89;
}
{
add.f16x2 r165, %16, r162;
}
{
sub.f16x2 r168, %19, %23;
}
{
mul.f16x2 r171, r168, r91;
}
{
add.f16x2 r174, r165, r171;
}
mov.f32 f52, 0f3F000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r177, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r178, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r179, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r180, {low, high};
}
mov.f32 f39, 0fBF800000;
{
mul.f16x2 r187, r120, r177;
}
{
mul.f16x2 r190, r156, r178;
}
{
sub.f16x2 r193, r187, r190;
}
{
mul.f16x2 r196, r120, r178;
}
{
fma.rn.f16x2 r199, r156, r177, r196;
}
{
mul.f16x2 r203, r138, r179;
}
{
mul.f16x2 r206, r174, r180;
}
{
sub.f16x2 r209, r203, r206;
}
{
mul.f16x2 r212, r138, r180;
}
{
fma.rn.f16x2 r215, r174, r179, r212;
}
{
add.f16x2 r219, r8, r96;
}
{
add.f16x2 r222, r14, r102;
}
{
sub.f16x2 r225, r8, r96;
}
{
sub.f16x2 r228, r14, r102;
}
{
add.f16x2 r231, r32, r193;
}
{
add.f16x2 r234, r68, r199;
}
{
sub.f16x2 r237, r32, r193;
}
{
sub.f16x2 r240, r68, r199;
}
{
add.f16x2 r243, r50, r209;
}
{
add.f16x2 r246, r86, r215;
}
{
sub.f16x2 r249, r50, r209;
}
{
sub.f16x2 r252, r86, r215;
}
mul.wide.u32 rd2, r701, -1431655765;
shr.u64 rd3, rd2, 34;
cvt.u32.u64 r702, rd3;
mul.lo.s32 r703, r702, 6;
sub.s32 r704, r701, r703;
shr.u64 rd4, rd2, 33;
cvt.u32.u64 r705, rd4;
and.b32 r706, r705, 2147483646;
mad.lo.s32 r707, r706, 144, r700;
cvt.rn.f32.u32 f71, r704;
mul.f32 f72, f71, 0f3E32B8C2;
cos.approx.f32 f29, f72;
sin.approx.f32 f73, f72;
neg.f32 f30, f73;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f29;
cvt.rn.f16.f32 high, f30;
mov.b32 r255, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r258, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r260, {high, high};
}
{
mul.f16x2 r262, r234, r260;
}
{
neg.f16x2 r265, r262;
}
{
fma.rn.f16x2 r267, r231, r258, r265;
}
{
mul.f16x2 r271, r231, r260;
}
{
fma.rn.f16x2 r274, r234, r258, r271;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r278, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r280, {high, high};
}
mov.f32 f40, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f39;
cvt.rn.f16.f32 high, f40;
mov.b32 r282, {low, high};
}
{
mul.f16x2 r283, r280, r282;
}
{
mul.f16x2 r286, r255, r278;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r289, {high, low};
}
{
fma.rn.f16x2 r291, r283, r289, r286;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r291;
mov.b32 r295, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r291;
mov.b32 r297, {high, high};
}
{
mul.f16x2 r299, r246, r297;
}
{
neg.f16x2 r302, r299;
}
{
fma.rn.f16x2 r304, r243, r295, r302;
}
{
mul.f16x2 r308, r243, r297;
}
{
fma.rn.f16x2 r311, r246, r295, r308;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r315, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r317, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f39;
cvt.rn.f16.f32 high, f40;
mov.b32 r319, {low, high};
}
{
mul.f16x2 r320, r317, r319;
}
{
mul.f16x2 r323, r291, r315;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r291;
mov.b32 r326, {high, low};
}
{
fma.rn.f16x2 r328, r320, r326, r323;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r328;
mov.b32 r332, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r328;
mov.b32 r334, {high, high};
}
{
mul.f16x2 r336, r228, r334;
}
{
neg.f16x2 r339, r336;
}
{
fma.rn.f16x2 r341, r225, r332, r339;
}
{
mul.f16x2 r345, r225, r334;
}
{
fma.rn.f16x2 r348, r228, r332, r345;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r352, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r354, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f39;
cvt.rn.f16.f32 high, f40;
mov.b32 r356, {low, high};
}
{
mul.f16x2 r357, r354, r356;
}
{
mul.f16x2 r360, r328, r352;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r328;
mov.b32 r363, {high, low};
}
{
fma.rn.f16x2 r365, r357, r363, r360;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r365;
mov.b32 r369, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r365;
mov.b32 r371, {high, high};
}
{
mul.f16x2 r373, r240, r371;
}
{
neg.f16x2 r376, r373;
}
{
fma.rn.f16x2 r378, r237, r369, r376;
}
{
mul.f16x2 r382, r237, r371;
}
{
fma.rn.f16x2 r385, r240, r369, r382;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r389, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r391, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f39;
cvt.rn.f16.f32 high, f40;
mov.b32 r393, {low, high};
}
{
mul.f16x2 r394, r391, r393;
}
{
mul.f16x2 r397, r365, r389;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r365;
mov.b32 r400, {high, low};
}
{
fma.rn.f16x2 r402, r394, r400, r397;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r402;
mov.b32 r406, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r402;
mov.b32 r408, {high, high};
}
{
mul.f16x2 r410, r252, r408;
}
{
neg.f16x2 r413, r410;
}
{
fma.rn.f16x2 r415, r249, r406, r413;
}
{
mul.f16x2 r419, r249, r408;
}
{
fma.rn.f16x2 r422, r252, r406, r419;
}
barrier.sync 0;
mad.lo.s32 r708, r704, 48, r707;
st.shared.v2.f32 [r708], {r219, r222};
st.shared.v2.f32 [r708+8], {r267, r274};
st.shared.v2.f32 [r708+16], {r304, r311};
st.shared.v2.f32 [r708+24], {r341, r348};
st.shared.v2.f32 [r708+32], {r378, r385};
st.shared.v2.f32 [r708+40], {r415, r422};
barrier.sync 0;
mad.lo.s32 r709, r704, -40, r708;
ld.shared.u32 r451, [r709];
ld.shared.u32 r457, [r709+4];
ld.shared.u32 r539, [r709+48];
ld.shared.u32 r545, [r709+52];
ld.shared.u32 r448, [r709+96];
ld.shared.u32 r454, [r709+100];
ld.shared.u32 r536, [r709+144];
ld.shared.u32 r542, [r709+148];
ld.shared.u32 r449, [r709+192];
ld.shared.u32 r455, [r709+196];
ld.shared.u32 r537, [r709+240];
ld.shared.u32 r543, [r709+244];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r443, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r444, {low, high};
}
{
neg.f16x2 r445, r444;
}
{
add.f16x2 r447, r448, r449;
}
{
add.f16x2 r450, r451, r447;
}
{
add.f16x2 r453, r454, r455;
}
{
add.f16x2 r456, r457, r453;
}
{
add.f16x2 r459, r448, r449;
}
{
mul.f16x2 r462, r459, r443;
}
{
add.f16x2 r465, r451, r462;
}
{
sub.f16x2 r468, r454, r455;
}
{
mul.f16x2 r471, r468, r445;
}
{
add.f16x2 r474, r465, r471;
}
{
add.f16x2 r477, r448, r449;
}
{
mul.f16x2 r480, r477, r443;
}
{
add.f16x2 r483, r451, r480;
}
{
sub.f16x2 r486, r454, r455;
}
{
mul.f16x2 r489, r486, r445;
}
{
sub.f16x2 r492, r483, r489;
}
{
add.f16x2 r495, r454, r455;
}
{
mul.f16x2 r498, r495, r443;
}
{
add.f16x2 r501, r457, r498;
}
{
sub.f16x2 r504, r448, r449;
}
{
mul.f16x2 r507, r504, r445;
}
{
sub.f16x2 r510, r501, r507;
}
{
add.f16x2 r513, r454, r455;
}
{
mul.f16x2 r516, r513, r443;
}
{
add.f16x2 r519, r457, r516;
}
{
sub.f16x2 r522, r448, r449;
}
{
mul.f16x2 r525, r522, r445;
}
{
add.f16x2 r528, r519, r525;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r531, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r532, {low, high};
}
{
neg.f16x2 r533, r532;
}
{
add.f16x2 r535, r536, r537;
}
{
add.f16x2 r538, r539, r535;
}
{
add.f16x2 r541, r542, r543;
}
{
add.f16x2 r544, r545, r541;
}
{
add.f16x2 r547, r536, r537;
}
{
mul.f16x2 r550, r547, r531;
}
{
add.f16x2 r553, r539, r550;
}
{
sub.f16x2 r556, r542, r543;
}
{
mul.f16x2 r559, r556, r533;
}
{
add.f16x2 r562, r553, r559;
}
{
add.f16x2 r565, r536, r537;
}
{
mul.f16x2 r568, r565, r531;
}
{
add.f16x2 r571, r539, r568;
}
{
sub.f16x2 r574, r542, r543;
}
{
mul.f16x2 r577, r574, r533;
}
{
sub.f16x2 r580, r571, r577;
}
{
add.f16x2 r583, r542, r543;
}
{
mul.f16x2 r586, r583, r531;
}
{
add.f16x2 r589, r545, r586;
}
{
sub.f16x2 r592, r536, r537;
}
{
mul.f16x2 r595, r592, r533;
}
{
sub.f16x2 r598, r589, r595;
}
{
add.f16x2 r601, r542, r543;
}
{
mul.f16x2 r604, r601, r531;
}
{
add.f16x2 r607, r545, r604;
}
{
sub.f16x2 r610, r536, r537;
}
{
mul.f16x2 r613, r610, r533;
}
{
add.f16x2 r616, r607, r613;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r619, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r620, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r621, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r622, {low, high};
}
{
mul.f16x2 r629, r562, r619;
}
{
mul.f16x2 r632, r598, r620;
}
{
sub.f16x2 r635, r629, r632;
}
{
mul.f16x2 r638, r562, r620;
}
{
fma.rn.f16x2 r641, r598, r619, r638;
}
{
mul.f16x2 r645, r580, r621;
}
{
mul.f16x2 r648, r616, r622;
}
{
sub.f16x2 r651, r645, r648;
}
{
mul.f16x2 r654, r580, r622;
}
{
fma.rn.f16x2 r657, r616, r621, r654;
}
{
add.f16x2 %0, r450, r538;
}
{
add.f16x2 %1, r456, r544;
}
{
sub.f16x2 %6, r450, r538;
}
{
sub.f16x2 %7, r456, r544;
}
{
add.f16x2 %2, r474, r635;
}
{
add.f16x2 %3, r510, r641;
}
{
sub.f16x2 %8, r474, r635;
}
{
sub.f16x2 %9, r510, r641;
}
{
add.f16x2 %4, r492, r651;
}
{
add.f16x2 %5, r528, r657;
}
{
sub.f16x2 %10, r492, r651;
}
{
sub.f16x2 %11, r528, r657;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<930, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<74>;
.reg .b32 r<707>;
.reg .b64 rd<4>;
mov.u32 r697, %tid.y;
mov.u32 r698, %12;
mad.lo.s32 r699, r697, 144, r698;
mov.u32 r700, %tid.x;
mov.f32 f56, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r1, {low, high};
}
mov.f32 f58, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r2, {low, high};
}
{
neg.f16x2 r3, r2;
}
{
add.f16x2 r5, %17, %21;
}
{
add.f16x2 r8, %13, r5;
}
{
add.f16x2 r11, %18, %22;
}
{
add.f16x2 r14, %14, r11;
}
{
add.f16x2 r17, %17, %21;
}
{
mul.f16x2 r20, r17, r1;
}
{
add.f16x2 r23, %13, r20;
}
{
sub.f16x2 r26, %18, %22;
}
{
mul.f16x2 r29, r26, r3;
}
{
add.f16x2 r32, r23, r29;
}
{
add.f16x2 r35, %17, %21;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %13, r38;
}
{
sub.f16x2 r44, %18, %22;
}
{
mul.f16x2 r47, r44, r3;
}
{
sub.f16x2 r50, r41, r47;
}
{
add.f16x2 r53, %18, %22;
}
{
mul.f16x2 r56, r53, r1;
}
{
add.f16x2 r59, %14, r56;
}
{
sub.f16x2 r62, %17, %21;
}
{
mul.f16x2 r65, r62, r3;
}
{
sub.f16x2 r68, r59, r65;
}
{
add.f16x2 r71, %18, %22;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %14, r74;
}
{
sub.f16x2 r80, %17, %21;
}
{
mul.f16x2 r83, r80, r3;
}
{
add.f16x2 r86, r77, r83;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r90, {low, high};
}
{
neg.f16x2 r91, r90;
}
{
add.f16x2 r93, %19, %23;
}
{
add.f16x2 r96, %15, r93;
}
{
add.f16x2 r99, %20, %24;
}
{
add.f16x2 r102, %16, r99;
}
{
add.f16x2 r105, %19, %23;
}
{
mul.f16x2 r108, r105, r89;
}
{
add.f16x2 r111, %15, r108;
}
{
sub.f16x2 r114, %20, %24;
}
{
mul.f16x2 r117, r114, r91;
}
{
add.f16x2 r120, r111, r117;
}
{
add.f16x2 r123, %19, %23;
}
{
mul.f16x2 r126, r123, r89;
}
{
add.f16x2 r129, %15, r126;
}
{
sub.f16x2 r132, %20, %24;
}
{
mul.f16x2 r135, r132, r91;
}
{
sub.f16x2 r138, r129, r135;
}
{
add.f16x2 r141, %20, %24;
}
{
mul.f16x2 r144, r141, r89;
}
{
add.f16x2 r147, %16, r144;
}
{
sub.f16x2 r150, %19, %23;
}
{
mul.f16x2 r153, r150, r91;
}
{
sub.f16x2 r156, r147, r153;
}
{
add.f16x2 r159, %20, %24;
}
{
mul.f16x2 r162, r159, r89;
}
{
add.f16x2 r165, %16, r162;
}
{
sub.f16x2 r168, %19, %23;
}
{
mul.f16x2 r171, r168, r91;
}
{
add.f16x2 r174, r165, r171;
}
mov.f32 f52, 0f3F000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r177, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r178, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r179, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r180, {low, high};
}
mov.f32 f39, 0fBF800000;
{
mul.f16x2 r187, r120, r177;
}
{
mul.f16x2 r190, r156, r178;
}
{
sub.f16x2 r193, r187, r190;
}
{
mul.f16x2 r196, r120, r178;
}
{
fma.rn.f16x2 r199, r156, r177, r196;
}
{
mul.f16x2 r203, r138, r179;
}
{
mul.f16x2 r206, r174, r180;
}
{
sub.f16x2 r209, r203, r206;
}
{
mul.f16x2 r212, r138, r180;
}
{
fma.rn.f16x2 r215, r174, r179, r212;
}
{
add.f16x2 r219, r8, r96;
}
{
add.f16x2 r222, r14, r102;
}
{
sub.f16x2 r225, r8, r96;
}
{
sub.f16x2 r228, r14, r102;
}
{
add.f16x2 r231, r32, r193;
}
{
add.f16x2 r234, r68, r199;
}
{
sub.f16x2 r237, r32, r193;
}
{
sub.f16x2 r240, r68, r199;
}
{
add.f16x2 r243, r50, r209;
}
{
add.f16x2 r246, r86, r215;
}
{
sub.f16x2 r249, r50, r209;
}
{
sub.f16x2 r252, r86, r215;
}
mul.wide.u32 rd2, r700, -1431655765;
shr.u64 rd3, rd2, 34;
cvt.u32.u64 r701, rd3;
mul.lo.s32 r702, r701, 6;
sub.s32 r703, r700, r702;
mad.lo.s32 r704, r701, 144, r699;
cvt.rn.f32.u32 f71, r703;
mul.f32 f72, f71, 0f3E32B8C2;
cos.approx.f32 f29, f72;
sin.approx.f32 f73, f72;
neg.f32 f30, f73;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f29;
cvt.rn.f16.f32 high, f30;
mov.b32 r255, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r258, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r260, {high, high};
}
{
mul.f16x2 r262, r234, r260;
}
{
neg.f16x2 r265, r262;
}
{
fma.rn.f16x2 r267, r231, r258, r265;
}
{
mul.f16x2 r271, r231, r260;
}
{
fma.rn.f16x2 r274, r234, r258, r271;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r278, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r280, {high, high};
}
mov.f32 f40, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f39;
cvt.rn.f16.f32 high, f40;
mov.b32 r282, {low, high};
}
{
mul.f16x2 r283, r280, r282;
}
{
mul.f16x2 r286, r255, r278;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r289, {high, low};
}
{
fma.rn.f16x2 r291, r283, r289, r286;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r291;
mov.b32 r295, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r291;
mov.b32 r297, {high, high};
}
{
mul.f16x2 r299, r246, r297;
}
{
neg.f16x2 r302, r299;
}
{
fma.rn.f16x2 r304, r243, r295, r302;
}
{
mul.f16x2 r308, r243, r297;
}
{
fma.rn.f16x2 r311, r246, r295, r308;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r315, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r317, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f39;
cvt.rn.f16.f32 high, f40;
mov.b32 r319, {low, high};
}
{
mul.f16x2 r320, r317, r319;
}
{
mul.f16x2 r323, r291, r315;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r291;
mov.b32 r326, {high, low};
}
{
fma.rn.f16x2 r328, r320, r326, r323;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r328;
mov.b32 r332, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r328;
mov.b32 r334, {high, high};
}
{
mul.f16x2 r336, r228, r334;
}
{
neg.f16x2 r339, r336;
}
{
fma.rn.f16x2 r341, r225, r332, r339;
}
{
mul.f16x2 r345, r225, r334;
}
{
fma.rn.f16x2 r348, r228, r332, r345;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r352, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r354, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f39;
cvt.rn.f16.f32 high, f40;
mov.b32 r356, {low, high};
}
{
mul.f16x2 r357, r354, r356;
}
{
mul.f16x2 r360, r328, r352;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r328;
mov.b32 r363, {high, low};
}
{
fma.rn.f16x2 r365, r357, r363, r360;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r365;
mov.b32 r369, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r365;
mov.b32 r371, {high, high};
}
{
mul.f16x2 r373, r240, r371;
}
{
neg.f16x2 r376, r373;
}
{
fma.rn.f16x2 r378, r237, r369, r376;
}
{
mul.f16x2 r382, r237, r371;
}
{
fma.rn.f16x2 r385, r240, r369, r382;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r389, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r255;
mov.b32 r391, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f39;
cvt.rn.f16.f32 high, f40;
mov.b32 r393, {low, high};
}
{
mul.f16x2 r394, r391, r393;
}
{
mul.f16x2 r397, r365, r389;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r365;
mov.b32 r400, {high, low};
}
{
fma.rn.f16x2 r402, r394, r400, r397;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r402;
mov.b32 r406, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r402;
mov.b32 r408, {high, high};
}
{
mul.f16x2 r410, r252, r408;
}
{
neg.f16x2 r413, r410;
}
{
fma.rn.f16x2 r415, r249, r406, r413;
}
{
mul.f16x2 r419, r249, r408;
}
{
fma.rn.f16x2 r422, r252, r406, r419;
}
barrier.sync 0;
mad.lo.s32 r705, r703, 24, r704;
st.shared.v2.f32 [r705], {r219, r267};
st.shared.v2.f32 [r705+8], {r304, r341};
st.shared.v2.f32 [r705+16], {r378, r415};
barrier.sync 0;
mad.lo.s32 r706, r703, -20, r705;
ld.shared.u32 r451, [r706];
ld.shared.u32 r539, [r706+24];
ld.shared.u32 r448, [r706+48];
ld.shared.u32 r536, [r706+72];
ld.shared.u32 r449, [r706+96];
ld.shared.u32 r537, [r706+120];
barrier.sync 0;
st.shared.v2.f32 [r705], {r222, r274};
st.shared.v2.f32 [r705+8], {r311, r348};
st.shared.v2.f32 [r705+16], {r385, r422};
barrier.sync 0;
ld.shared.u32 r457, [r706];
ld.shared.u32 r545, [r706+24];
ld.shared.u32 r454, [r706+48];
ld.shared.u32 r542, [r706+72];
ld.shared.u32 r455, [r706+96];
ld.shared.u32 r543, [r706+120];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r443, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r444, {low, high};
}
{
neg.f16x2 r445, r444;
}
{
add.f16x2 r447, r448, r449;
}
{
add.f16x2 r450, r451, r447;
}
{
add.f16x2 r453, r454, r455;
}
{
add.f16x2 r456, r457, r453;
}
{
add.f16x2 r459, r448, r449;
}
{
mul.f16x2 r462, r459, r443;
}
{
add.f16x2 r465, r451, r462;
}
{
sub.f16x2 r468, r454, r455;
}
{
mul.f16x2 r471, r468, r445;
}
{
add.f16x2 r474, r465, r471;
}
{
add.f16x2 r477, r448, r449;
}
{
mul.f16x2 r480, r477, r443;
}
{
add.f16x2 r483, r451, r480;
}
{
sub.f16x2 r486, r454, r455;
}
{
mul.f16x2 r489, r486, r445;
}
{
sub.f16x2 r492, r483, r489;
}
{
add.f16x2 r495, r454, r455;
}
{
mul.f16x2 r498, r495, r443;
}
{
add.f16x2 r501, r457, r498;
}
{
sub.f16x2 r504, r448, r449;
}
{
mul.f16x2 r507, r504, r445;
}
{
sub.f16x2 r510, r501, r507;
}
{
add.f16x2 r513, r454, r455;
}
{
mul.f16x2 r516, r513, r443;
}
{
add.f16x2 r519, r457, r516;
}
{
sub.f16x2 r522, r448, r449;
}
{
mul.f16x2 r525, r522, r445;
}
{
add.f16x2 r528, r519, r525;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r531, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r532, {low, high};
}
{
neg.f16x2 r533, r532;
}
{
add.f16x2 r535, r536, r537;
}
{
add.f16x2 r538, r539, r535;
}
{
add.f16x2 r541, r542, r543;
}
{
add.f16x2 r544, r545, r541;
}
{
add.f16x2 r547, r536, r537;
}
{
mul.f16x2 r550, r547, r531;
}
{
add.f16x2 r553, r539, r550;
}
{
sub.f16x2 r556, r542, r543;
}
{
mul.f16x2 r559, r556, r533;
}
{
add.f16x2 r562, r553, r559;
}
{
add.f16x2 r565, r536, r537;
}
{
mul.f16x2 r568, r565, r531;
}
{
add.f16x2 r571, r539, r568;
}
{
sub.f16x2 r574, r542, r543;
}
{
mul.f16x2 r577, r574, r533;
}
{
sub.f16x2 r580, r571, r577;
}
{
add.f16x2 r583, r542, r543;
}
{
mul.f16x2 r586, r583, r531;
}
{
add.f16x2 r589, r545, r586;
}
{
sub.f16x2 r592, r536, r537;
}
{
mul.f16x2 r595, r592, r533;
}
{
sub.f16x2 r598, r589, r595;
}
{
add.f16x2 r601, r542, r543;
}
{
mul.f16x2 r604, r601, r531;
}
{
add.f16x2 r607, r545, r604;
}
{
sub.f16x2 r610, r536, r537;
}
{
mul.f16x2 r613, r610, r533;
}
{
add.f16x2 r616, r607, r613;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r619, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r620, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r621, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r622, {low, high};
}
{
mul.f16x2 r629, r562, r619;
}
{
mul.f16x2 r632, r598, r620;
}
{
sub.f16x2 r635, r629, r632;
}
{
mul.f16x2 r638, r562, r620;
}
{
fma.rn.f16x2 r641, r598, r619, r638;
}
{
mul.f16x2 r645, r580, r621;
}
{
mul.f16x2 r648, r616, r622;
}
{
sub.f16x2 r651, r645, r648;
}
{
mul.f16x2 r654, r580, r622;
}
{
fma.rn.f16x2 r657, r616, r621, r654;
}
{
add.f16x2 %0, r450, r538;
}
{
add.f16x2 %1, r456, r544;
}
{
sub.f16x2 %6, r450, r538;
}
{
sub.f16x2 %7, r456, r544;
}
{
add.f16x2 %2, r474, r635;
}
{
add.f16x2 %3, r510, r641;
}
{
sub.f16x2 %8, r474, r635;
}
{
sub.f16x2 %9, r510, r641;
}
{
add.f16x2 %4, r492, r651;
}
{
add.f16x2 %5, r528, r657;
}
{
sub.f16x2 %10, r492, r651;
}
{
sub.f16x2 %11, r528, r657;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)));
};


#endif
