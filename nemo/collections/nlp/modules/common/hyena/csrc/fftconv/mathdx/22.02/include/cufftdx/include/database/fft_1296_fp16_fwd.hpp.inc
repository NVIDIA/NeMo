#ifndef CUFFTDX_FFT_1296_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_1296_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<934, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<164>;
.reg .b32 r<1606>;
.reg .b64 rd<8>;
mov.u32 r1581, %tid.y;
shl.b32 r1582, r1581, 1;
mov.u32 r1583, %12;
mad.lo.s32 r1584, r1582, 5184, r1583;
mov.u32 r1585, %tid.x;
mov.f32 f140, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1, {low, high};
}
mov.f32 f142, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
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
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
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
mov.f32 f136, 0f3F000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f136;
cvt.rn.f16.f32 high, f136;
mov.b32 r177, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r178, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r179, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r180, {low, high};
}
mov.f32 f123, 0fBF800000;
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
shr.u32 r1586, r1585, 3;
mul.wide.u32 rd2, r1586, 159072863;
shr.u64 rd3, rd2, 32;
cvt.u32.u64 r1587, rd3;
mul.lo.s32 r1588, r1587, 216;
sub.s32 r1589, r1585, r1588;
shl.b32 r1590, r1587, 1;
mad.lo.s32 r1591, r1590, 5184, r1584;
cvt.rn.f32.u32 f155, r1589;
mul.f32 f156, f155, 0f3B9EDD1F;
cos.approx.f32 f29, f156;
sin.approx.f32 f157, f156;
neg.f32 f30, f157;
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
mov.f32 f124, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
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
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
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
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
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
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
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
mad.lo.s32 r1592, r1589, 48, r1591;
st.shared.v2.f32 [r1592], {r219, r222};
st.shared.v2.f32 [r1592+8], {r267, r274};
st.shared.v2.f32 [r1592+16], {r304, r311};
st.shared.v2.f32 [r1592+24], {r341, r348};
st.shared.v2.f32 [r1592+32], {r378, r385};
st.shared.v2.f32 [r1592+40], {r415, r422};
barrier.sync 0;
mad.lo.s32 r1593, r1589, -40, r1592;
ld.shared.u32 r451, [r1593];
ld.shared.u32 r457, [r1593+4];
ld.shared.u32 r539, [r1593+1728];
ld.shared.u32 r545, [r1593+1732];
ld.shared.u32 r448, [r1593+3456];
ld.shared.u32 r454, [r1593+3460];
ld.shared.u32 r536, [r1593+5184];
ld.shared.u32 r542, [r1593+5188];
ld.shared.u32 r449, [r1593+6912];
ld.shared.u32 r455, [r1593+6916];
ld.shared.u32 r537, [r1593+8640];
ld.shared.u32 r543, [r1593+8644];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r443, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
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
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r531, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
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
cvt.rn.f16.f32 low, f136;
cvt.rn.f16.f32 high, f136;
mov.b32 r619, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r620, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r621, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
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
add.f16x2 r661, r450, r538;
}
{
add.f16x2 r664, r456, r544;
}
{
sub.f16x2 r667, r450, r538;
}
{
sub.f16x2 r670, r456, r544;
}
{
add.f16x2 r673, r474, r635;
}
{
add.f16x2 r676, r510, r641;
}
{
sub.f16x2 r679, r474, r635;
}
{
sub.f16x2 r682, r510, r641;
}
{
add.f16x2 r685, r492, r651;
}
{
add.f16x2 r688, r528, r657;
}
{
sub.f16x2 r691, r492, r651;
}
{
sub.f16x2 r694, r528, r657;
}
mul.wide.u32 rd4, r1589, -1431655765;
shr.u64 rd5, rd4, 34;
cvt.u32.u64 r1594, rd5;
cvt.rn.f32.u32 f158, r1594;
mul.f32 f159, f158, 0f3CEE4BAE;
cos.approx.f32 f71, f159;
sin.approx.f32 f160, f159;
neg.f32 f72, f160;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f71;
cvt.rn.f16.f32 high, f72;
mov.b32 r697, {low, high};
}
mul.lo.s32 r1595, r1594, 6;
sub.s32 r1596, r1589, r1595;
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r700, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r702, {high, high};
}
{
mul.f16x2 r704, r676, r702;
}
{
neg.f16x2 r707, r704;
}
{
fma.rn.f16x2 r709, r673, r700, r707;
}
{
mul.f16x2 r713, r673, r702;
}
{
fma.rn.f16x2 r716, r676, r700, r713;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r720, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r722, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r724, {low, high};
}
{
mul.f16x2 r725, r722, r724;
}
{
mul.f16x2 r728, r697, r720;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r731, {high, low};
}
{
fma.rn.f16x2 r733, r725, r731, r728;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r733;
mov.b32 r737, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r733;
mov.b32 r739, {high, high};
}
{
mul.f16x2 r741, r688, r739;
}
{
neg.f16x2 r744, r741;
}
{
fma.rn.f16x2 r746, r685, r737, r744;
}
{
mul.f16x2 r750, r685, r739;
}
{
fma.rn.f16x2 r753, r688, r737, r750;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r757, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r759, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r761, {low, high};
}
{
mul.f16x2 r762, r759, r761;
}
{
mul.f16x2 r765, r733, r757;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r733;
mov.b32 r768, {high, low};
}
{
fma.rn.f16x2 r770, r762, r768, r765;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r770;
mov.b32 r774, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r770;
mov.b32 r776, {high, high};
}
{
mul.f16x2 r778, r670, r776;
}
{
neg.f16x2 r781, r778;
}
{
fma.rn.f16x2 r783, r667, r774, r781;
}
{
mul.f16x2 r787, r667, r776;
}
{
fma.rn.f16x2 r790, r670, r774, r787;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r794, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r796, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r798, {low, high};
}
{
mul.f16x2 r799, r796, r798;
}
{
mul.f16x2 r802, r770, r794;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r770;
mov.b32 r805, {high, low};
}
{
fma.rn.f16x2 r807, r799, r805, r802;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r807;
mov.b32 r811, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r807;
mov.b32 r813, {high, high};
}
{
mul.f16x2 r815, r682, r813;
}
{
neg.f16x2 r818, r815;
}
{
fma.rn.f16x2 r820, r679, r811, r818;
}
{
mul.f16x2 r824, r679, r813;
}
{
fma.rn.f16x2 r827, r682, r811, r824;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r831, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r833, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r835, {low, high};
}
{
mul.f16x2 r836, r833, r835;
}
{
mul.f16x2 r839, r807, r831;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r807;
mov.b32 r842, {high, low};
}
{
fma.rn.f16x2 r844, r836, r842, r839;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r844;
mov.b32 r848, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r844;
mov.b32 r850, {high, high};
}
{
mul.f16x2 r852, r694, r850;
}
{
neg.f16x2 r855, r852;
}
{
fma.rn.f16x2 r857, r691, r848, r855;
}
{
mul.f16x2 r861, r691, r850;
}
{
fma.rn.f16x2 r864, r694, r848, r861;
}
shl.b32 r1597, r1596, 3;
add.s32 r1598, r1591, r1597;
barrier.sync 0;
mad.lo.s32 r1599, r1594, 288, r1598;
st.shared.u32 [r1599], r661;
st.shared.u32 [r1599+4], r664;
st.shared.u32 [r1599+48], r709;
st.shared.u32 [r1599+52], r716;
st.shared.u32 [r1599+96], r746;
st.shared.u32 [r1599+100], r753;
st.shared.u32 [r1599+144], r783;
st.shared.u32 [r1599+148], r790;
st.shared.u32 [r1599+192], r820;
st.shared.u32 [r1599+196], r827;
st.shared.u32 [r1599+240], r857;
st.shared.u32 [r1599+244], r864;
barrier.sync 0;
ld.shared.u32 r893, [r1593];
ld.shared.u32 r899, [r1593+4];
ld.shared.u32 r981, [r1593+1728];
ld.shared.u32 r987, [r1593+1732];
ld.shared.u32 r890, [r1593+3456];
ld.shared.u32 r896, [r1593+3460];
ld.shared.u32 r978, [r1593+5184];
ld.shared.u32 r984, [r1593+5188];
ld.shared.u32 r891, [r1593+6912];
ld.shared.u32 r897, [r1593+6916];
ld.shared.u32 r979, [r1593+8640];
ld.shared.u32 r985, [r1593+8644];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r885, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r886, {low, high};
}
{
neg.f16x2 r887, r886;
}
{
add.f16x2 r889, r890, r891;
}
{
add.f16x2 r892, r893, r889;
}
{
add.f16x2 r895, r896, r897;
}
{
add.f16x2 r898, r899, r895;
}
{
add.f16x2 r901, r890, r891;
}
{
mul.f16x2 r904, r901, r885;
}
{
add.f16x2 r907, r893, r904;
}
{
sub.f16x2 r910, r896, r897;
}
{
mul.f16x2 r913, r910, r887;
}
{
add.f16x2 r916, r907, r913;
}
{
add.f16x2 r919, r890, r891;
}
{
mul.f16x2 r922, r919, r885;
}
{
add.f16x2 r925, r893, r922;
}
{
sub.f16x2 r928, r896, r897;
}
{
mul.f16x2 r931, r928, r887;
}
{
sub.f16x2 r934, r925, r931;
}
{
add.f16x2 r937, r896, r897;
}
{
mul.f16x2 r940, r937, r885;
}
{
add.f16x2 r943, r899, r940;
}
{
sub.f16x2 r946, r890, r891;
}
{
mul.f16x2 r949, r946, r887;
}
{
sub.f16x2 r952, r943, r949;
}
{
add.f16x2 r955, r896, r897;
}
{
mul.f16x2 r958, r955, r885;
}
{
add.f16x2 r961, r899, r958;
}
{
sub.f16x2 r964, r890, r891;
}
{
mul.f16x2 r967, r964, r887;
}
{
add.f16x2 r970, r961, r967;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r973, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r974, {low, high};
}
{
neg.f16x2 r975, r974;
}
{
add.f16x2 r977, r978, r979;
}
{
add.f16x2 r980, r981, r977;
}
{
add.f16x2 r983, r984, r985;
}
{
add.f16x2 r986, r987, r983;
}
{
add.f16x2 r989, r978, r979;
}
{
mul.f16x2 r992, r989, r973;
}
{
add.f16x2 r995, r981, r992;
}
{
sub.f16x2 r998, r984, r985;
}
{
mul.f16x2 r1001, r998, r975;
}
{
add.f16x2 r1004, r995, r1001;
}
{
add.f16x2 r1007, r978, r979;
}
{
mul.f16x2 r1010, r1007, r973;
}
{
add.f16x2 r1013, r981, r1010;
}
{
sub.f16x2 r1016, r984, r985;
}
{
mul.f16x2 r1019, r1016, r975;
}
{
sub.f16x2 r1022, r1013, r1019;
}
{
add.f16x2 r1025, r984, r985;
}
{
mul.f16x2 r1028, r1025, r973;
}
{
add.f16x2 r1031, r987, r1028;
}
{
sub.f16x2 r1034, r978, r979;
}
{
mul.f16x2 r1037, r1034, r975;
}
{
sub.f16x2 r1040, r1031, r1037;
}
{
add.f16x2 r1043, r984, r985;
}
{
mul.f16x2 r1046, r1043, r973;
}
{
add.f16x2 r1049, r987, r1046;
}
{
sub.f16x2 r1052, r978, r979;
}
{
mul.f16x2 r1055, r1052, r975;
}
{
add.f16x2 r1058, r1049, r1055;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f136;
cvt.rn.f16.f32 high, f136;
mov.b32 r1061, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r1062, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1063, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r1064, {low, high};
}
{
mul.f16x2 r1071, r1004, r1061;
}
{
mul.f16x2 r1074, r1040, r1062;
}
{
sub.f16x2 r1077, r1071, r1074;
}
{
mul.f16x2 r1080, r1004, r1062;
}
{
fma.rn.f16x2 r1083, r1040, r1061, r1080;
}
{
mul.f16x2 r1087, r1022, r1063;
}
{
mul.f16x2 r1090, r1058, r1064;
}
{
sub.f16x2 r1093, r1087, r1090;
}
{
mul.f16x2 r1096, r1022, r1064;
}
{
fma.rn.f16x2 r1099, r1058, r1063, r1096;
}
{
add.f16x2 r1103, r892, r980;
}
{
add.f16x2 r1106, r898, r986;
}
{
sub.f16x2 r1109, r892, r980;
}
{
sub.f16x2 r1112, r898, r986;
}
{
add.f16x2 r1115, r916, r1077;
}
{
add.f16x2 r1118, r952, r1083;
}
{
sub.f16x2 r1121, r916, r1077;
}
{
sub.f16x2 r1124, r952, r1083;
}
{
add.f16x2 r1127, r934, r1093;
}
{
add.f16x2 r1130, r970, r1099;
}
{
sub.f16x2 r1133, r934, r1093;
}
{
sub.f16x2 r1136, r970, r1099;
}
mul.wide.u32 rd6, r1589, 954437177;
shr.u64 rd7, rd6, 35;
cvt.u32.u64 r1600, rd7;
cvt.rn.f32.u32 f161, r1600;
mul.f32 f162, f161, 0f3E32B8C2;
cos.approx.f32 f113, f162;
sin.approx.f32 f163, f162;
neg.f32 f114, f163;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f113;
cvt.rn.f16.f32 high, f114;
mov.b32 r1139, {low, high};
}
mul.lo.s32 r1601, r1600, 36;
sub.s32 r1602, r1589, r1601;
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1142, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1144, {high, high};
}
{
mul.f16x2 r1146, r1118, r1144;
}
{
neg.f16x2 r1149, r1146;
}
{
fma.rn.f16x2 r1151, r1115, r1142, r1149;
}
{
mul.f16x2 r1155, r1115, r1144;
}
{
fma.rn.f16x2 r1158, r1118, r1142, r1155;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1162, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1164, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r1166, {low, high};
}
{
mul.f16x2 r1167, r1164, r1166;
}
{
mul.f16x2 r1170, r1139, r1162;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1173, {high, low};
}
{
fma.rn.f16x2 r1175, r1167, r1173, r1170;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1175;
mov.b32 r1179, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1175;
mov.b32 r1181, {high, high};
}
{
mul.f16x2 r1183, r1130, r1181;
}
{
neg.f16x2 r1186, r1183;
}
{
fma.rn.f16x2 r1188, r1127, r1179, r1186;
}
{
mul.f16x2 r1192, r1127, r1181;
}
{
fma.rn.f16x2 r1195, r1130, r1179, r1192;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1199, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1201, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r1203, {low, high};
}
{
mul.f16x2 r1204, r1201, r1203;
}
{
mul.f16x2 r1207, r1175, r1199;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1175;
mov.b32 r1210, {high, low};
}
{
fma.rn.f16x2 r1212, r1204, r1210, r1207;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1212;
mov.b32 r1216, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1212;
mov.b32 r1218, {high, high};
}
{
mul.f16x2 r1220, r1112, r1218;
}
{
neg.f16x2 r1223, r1220;
}
{
fma.rn.f16x2 r1225, r1109, r1216, r1223;
}
{
mul.f16x2 r1229, r1109, r1218;
}
{
fma.rn.f16x2 r1232, r1112, r1216, r1229;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1236, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1238, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r1240, {low, high};
}
{
mul.f16x2 r1241, r1238, r1240;
}
{
mul.f16x2 r1244, r1212, r1236;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1212;
mov.b32 r1247, {high, low};
}
{
fma.rn.f16x2 r1249, r1241, r1247, r1244;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1249;
mov.b32 r1253, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1249;
mov.b32 r1255, {high, high};
}
{
mul.f16x2 r1257, r1124, r1255;
}
{
neg.f16x2 r1260, r1257;
}
{
fma.rn.f16x2 r1262, r1121, r1253, r1260;
}
{
mul.f16x2 r1266, r1121, r1255;
}
{
fma.rn.f16x2 r1269, r1124, r1253, r1266;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1273, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1275, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r1277, {low, high};
}
{
mul.f16x2 r1278, r1275, r1277;
}
{
mul.f16x2 r1281, r1249, r1273;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1249;
mov.b32 r1284, {high, low};
}
{
fma.rn.f16x2 r1286, r1278, r1284, r1281;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1286;
mov.b32 r1290, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1286;
mov.b32 r1292, {high, high};
}
{
mul.f16x2 r1294, r1136, r1292;
}
{
neg.f16x2 r1297, r1294;
}
{
fma.rn.f16x2 r1299, r1133, r1290, r1297;
}
{
mul.f16x2 r1303, r1133, r1292;
}
{
fma.rn.f16x2 r1306, r1136, r1290, r1303;
}
shl.b32 r1603, r1602, 3;
add.s32 r1604, r1591, r1603;
barrier.sync 0;
mad.lo.s32 r1605, r1600, 1728, r1604;
st.shared.u32 [r1605], r1103;
st.shared.u32 [r1605+4], r1106;
st.shared.u32 [r1605+288], r1151;
st.shared.u32 [r1605+292], r1158;
st.shared.u32 [r1605+576], r1188;
st.shared.u32 [r1605+580], r1195;
st.shared.u32 [r1605+864], r1225;
st.shared.u32 [r1605+868], r1232;
st.shared.u32 [r1605+1152], r1262;
st.shared.u32 [r1605+1156], r1269;
st.shared.u32 [r1605+1440], r1299;
st.shared.u32 [r1605+1444], r1306;
barrier.sync 0;
ld.shared.u32 r1335, [r1593];
ld.shared.u32 r1341, [r1593+4];
ld.shared.u32 r1423, [r1593+1728];
ld.shared.u32 r1429, [r1593+1732];
ld.shared.u32 r1332, [r1593+3456];
ld.shared.u32 r1338, [r1593+3460];
ld.shared.u32 r1420, [r1593+5184];
ld.shared.u32 r1426, [r1593+5188];
ld.shared.u32 r1333, [r1593+6912];
ld.shared.u32 r1339, [r1593+6916];
ld.shared.u32 r1421, [r1593+8640];
ld.shared.u32 r1427, [r1593+8644];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1327, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r1328, {low, high};
}
{
neg.f16x2 r1329, r1328;
}
{
add.f16x2 r1331, r1332, r1333;
}
{
add.f16x2 r1334, r1335, r1331;
}
{
add.f16x2 r1337, r1338, r1339;
}
{
add.f16x2 r1340, r1341, r1337;
}
{
add.f16x2 r1343, r1332, r1333;
}
{
mul.f16x2 r1346, r1343, r1327;
}
{
add.f16x2 r1349, r1335, r1346;
}
{
sub.f16x2 r1352, r1338, r1339;
}
{
mul.f16x2 r1355, r1352, r1329;
}
{
add.f16x2 r1358, r1349, r1355;
}
{
add.f16x2 r1361, r1332, r1333;
}
{
mul.f16x2 r1364, r1361, r1327;
}
{
add.f16x2 r1367, r1335, r1364;
}
{
sub.f16x2 r1370, r1338, r1339;
}
{
mul.f16x2 r1373, r1370, r1329;
}
{
sub.f16x2 r1376, r1367, r1373;
}
{
add.f16x2 r1379, r1338, r1339;
}
{
mul.f16x2 r1382, r1379, r1327;
}
{
add.f16x2 r1385, r1341, r1382;
}
{
sub.f16x2 r1388, r1332, r1333;
}
{
mul.f16x2 r1391, r1388, r1329;
}
{
sub.f16x2 r1394, r1385, r1391;
}
{
add.f16x2 r1397, r1338, r1339;
}
{
mul.f16x2 r1400, r1397, r1327;
}
{
add.f16x2 r1403, r1341, r1400;
}
{
sub.f16x2 r1406, r1332, r1333;
}
{
mul.f16x2 r1409, r1406, r1329;
}
{
add.f16x2 r1412, r1403, r1409;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1415, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r1416, {low, high};
}
{
neg.f16x2 r1417, r1416;
}
{
add.f16x2 r1419, r1420, r1421;
}
{
add.f16x2 r1422, r1423, r1419;
}
{
add.f16x2 r1425, r1426, r1427;
}
{
add.f16x2 r1428, r1429, r1425;
}
{
add.f16x2 r1431, r1420, r1421;
}
{
mul.f16x2 r1434, r1431, r1415;
}
{
add.f16x2 r1437, r1423, r1434;
}
{
sub.f16x2 r1440, r1426, r1427;
}
{
mul.f16x2 r1443, r1440, r1417;
}
{
add.f16x2 r1446, r1437, r1443;
}
{
add.f16x2 r1449, r1420, r1421;
}
{
mul.f16x2 r1452, r1449, r1415;
}
{
add.f16x2 r1455, r1423, r1452;
}
{
sub.f16x2 r1458, r1426, r1427;
}
{
mul.f16x2 r1461, r1458, r1417;
}
{
sub.f16x2 r1464, r1455, r1461;
}
{
add.f16x2 r1467, r1426, r1427;
}
{
mul.f16x2 r1470, r1467, r1415;
}
{
add.f16x2 r1473, r1429, r1470;
}
{
sub.f16x2 r1476, r1420, r1421;
}
{
mul.f16x2 r1479, r1476, r1417;
}
{
sub.f16x2 r1482, r1473, r1479;
}
{
add.f16x2 r1485, r1426, r1427;
}
{
mul.f16x2 r1488, r1485, r1415;
}
{
add.f16x2 r1491, r1429, r1488;
}
{
sub.f16x2 r1494, r1420, r1421;
}
{
mul.f16x2 r1497, r1494, r1417;
}
{
add.f16x2 r1500, r1491, r1497;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f136;
cvt.rn.f16.f32 high, f136;
mov.b32 r1503, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r1504, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1505, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r1506, {low, high};
}
{
mul.f16x2 r1513, r1446, r1503;
}
{
mul.f16x2 r1516, r1482, r1504;
}
{
sub.f16x2 r1519, r1513, r1516;
}
{
mul.f16x2 r1522, r1446, r1504;
}
{
fma.rn.f16x2 r1525, r1482, r1503, r1522;
}
{
mul.f16x2 r1529, r1464, r1505;
}
{
mul.f16x2 r1532, r1500, r1506;
}
{
sub.f16x2 r1535, r1529, r1532;
}
{
mul.f16x2 r1538, r1464, r1506;
}
{
fma.rn.f16x2 r1541, r1500, r1505, r1538;
}
{
add.f16x2 %0, r1334, r1422;
}
{
add.f16x2 %1, r1340, r1428;
}
{
sub.f16x2 %6, r1334, r1422;
}
{
sub.f16x2 %7, r1340, r1428;
}
{
add.f16x2 %2, r1358, r1519;
}
{
add.f16x2 %3, r1394, r1525;
}
{
sub.f16x2 %8, r1358, r1519;
}
{
sub.f16x2 %9, r1394, r1525;
}
{
add.f16x2 %4, r1376, r1535;
}
{
add.f16x2 %5, r1412, r1541;
}
{
sub.f16x2 %10, r1376, r1535;
}
{
sub.f16x2 %11, r1412, r1541;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<935, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<164>;
.reg .b32 r<1604>;
.reg .b64 rd<8>;
mov.u32 r1581, %tid.y;
mov.u32 r1582, %12;
mad.lo.s32 r1583, r1581, 5184, r1582;
mov.u32 r1584, %tid.x;
mov.f32 f140, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1, {low, high};
}
mov.f32 f142, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
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
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
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
mov.f32 f136, 0f3F000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f136;
cvt.rn.f16.f32 high, f136;
mov.b32 r177, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r178, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r179, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r180, {low, high};
}
mov.f32 f123, 0fBF800000;
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
shr.u32 r1585, r1584, 3;
mul.wide.u32 rd2, r1585, 159072863;
shr.u64 rd3, rd2, 32;
cvt.u32.u64 r1586, rd3;
mul.lo.s32 r1587, r1586, 216;
sub.s32 r1588, r1584, r1587;
mad.lo.s32 r1589, r1586, 5184, r1583;
cvt.rn.f32.u32 f155, r1588;
mul.f32 f156, f155, 0f3B9EDD1F;
cos.approx.f32 f29, f156;
sin.approx.f32 f157, f156;
neg.f32 f30, f157;
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
mov.f32 f124, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
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
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
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
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
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
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
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
mad.lo.s32 r1590, r1588, 24, r1589;
st.shared.v2.f32 [r1590], {r219, r267};
st.shared.v2.f32 [r1590+8], {r304, r341};
st.shared.v2.f32 [r1590+16], {r378, r415};
barrier.sync 0;
mad.lo.s32 r1591, r1588, -20, r1590;
ld.shared.u32 r451, [r1591];
ld.shared.u32 r539, [r1591+864];
ld.shared.u32 r448, [r1591+1728];
ld.shared.u32 r536, [r1591+2592];
ld.shared.u32 r449, [r1591+3456];
ld.shared.u32 r537, [r1591+4320];
barrier.sync 0;
st.shared.v2.f32 [r1590], {r222, r274};
st.shared.v2.f32 [r1590+8], {r311, r348};
st.shared.v2.f32 [r1590+16], {r385, r422};
barrier.sync 0;
ld.shared.u32 r457, [r1591];
ld.shared.u32 r545, [r1591+864];
ld.shared.u32 r454, [r1591+1728];
ld.shared.u32 r542, [r1591+2592];
ld.shared.u32 r455, [r1591+3456];
ld.shared.u32 r543, [r1591+4320];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r443, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
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
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r531, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
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
cvt.rn.f16.f32 low, f136;
cvt.rn.f16.f32 high, f136;
mov.b32 r619, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r620, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r621, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
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
add.f16x2 r661, r450, r538;
}
{
add.f16x2 r664, r456, r544;
}
{
sub.f16x2 r667, r450, r538;
}
{
sub.f16x2 r670, r456, r544;
}
{
add.f16x2 r673, r474, r635;
}
{
add.f16x2 r676, r510, r641;
}
{
sub.f16x2 r679, r474, r635;
}
{
sub.f16x2 r682, r510, r641;
}
{
add.f16x2 r685, r492, r651;
}
{
add.f16x2 r688, r528, r657;
}
{
sub.f16x2 r691, r492, r651;
}
{
sub.f16x2 r694, r528, r657;
}
mul.wide.u32 rd4, r1588, -1431655765;
shr.u64 rd5, rd4, 34;
cvt.u32.u64 r1592, rd5;
mul.lo.s32 r1593, r1592, 6;
sub.s32 r1594, r1588, r1593;
shl.b32 r1595, r1594, 2;
add.s32 r1596, r1589, r1595;
cvt.rn.f32.u32 f158, r1592;
mul.f32 f159, f158, 0f3CEE4BAE;
cos.approx.f32 f71, f159;
sin.approx.f32 f160, f159;
neg.f32 f72, f160;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f71;
cvt.rn.f16.f32 high, f72;
mov.b32 r697, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r700, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r702, {high, high};
}
{
mul.f16x2 r704, r676, r702;
}
{
neg.f16x2 r707, r704;
}
{
fma.rn.f16x2 r709, r673, r700, r707;
}
{
mul.f16x2 r713, r673, r702;
}
{
fma.rn.f16x2 r716, r676, r700, r713;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r720, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r722, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r724, {low, high};
}
{
mul.f16x2 r725, r722, r724;
}
{
mul.f16x2 r728, r697, r720;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r731, {high, low};
}
{
fma.rn.f16x2 r733, r725, r731, r728;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r733;
mov.b32 r737, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r733;
mov.b32 r739, {high, high};
}
{
mul.f16x2 r741, r688, r739;
}
{
neg.f16x2 r744, r741;
}
{
fma.rn.f16x2 r746, r685, r737, r744;
}
{
mul.f16x2 r750, r685, r739;
}
{
fma.rn.f16x2 r753, r688, r737, r750;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r757, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r759, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r761, {low, high};
}
{
mul.f16x2 r762, r759, r761;
}
{
mul.f16x2 r765, r733, r757;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r733;
mov.b32 r768, {high, low};
}
{
fma.rn.f16x2 r770, r762, r768, r765;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r770;
mov.b32 r774, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r770;
mov.b32 r776, {high, high};
}
{
mul.f16x2 r778, r670, r776;
}
{
neg.f16x2 r781, r778;
}
{
fma.rn.f16x2 r783, r667, r774, r781;
}
{
mul.f16x2 r787, r667, r776;
}
{
fma.rn.f16x2 r790, r670, r774, r787;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r794, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r796, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r798, {low, high};
}
{
mul.f16x2 r799, r796, r798;
}
{
mul.f16x2 r802, r770, r794;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r770;
mov.b32 r805, {high, low};
}
{
fma.rn.f16x2 r807, r799, r805, r802;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r807;
mov.b32 r811, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r807;
mov.b32 r813, {high, high};
}
{
mul.f16x2 r815, r682, r813;
}
{
neg.f16x2 r818, r815;
}
{
fma.rn.f16x2 r820, r679, r811, r818;
}
{
mul.f16x2 r824, r679, r813;
}
{
fma.rn.f16x2 r827, r682, r811, r824;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r831, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r833, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r835, {low, high};
}
{
mul.f16x2 r836, r833, r835;
}
{
mul.f16x2 r839, r807, r831;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r807;
mov.b32 r842, {high, low};
}
{
fma.rn.f16x2 r844, r836, r842, r839;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r844;
mov.b32 r848, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r844;
mov.b32 r850, {high, high};
}
{
mul.f16x2 r852, r694, r850;
}
{
neg.f16x2 r855, r852;
}
{
fma.rn.f16x2 r857, r691, r848, r855;
}
{
mul.f16x2 r861, r691, r850;
}
{
fma.rn.f16x2 r864, r694, r848, r861;
}
barrier.sync 0;
mad.lo.s32 r1597, r1592, 144, r1596;
st.shared.u32 [r1597], r661;
st.shared.u32 [r1597+24], r709;
st.shared.u32 [r1597+48], r746;
st.shared.u32 [r1597+72], r783;
st.shared.u32 [r1597+96], r820;
st.shared.u32 [r1597+120], r857;
barrier.sync 0;
ld.shared.u32 r893, [r1591];
ld.shared.u32 r981, [r1591+864];
ld.shared.u32 r890, [r1591+1728];
ld.shared.u32 r978, [r1591+2592];
ld.shared.u32 r891, [r1591+3456];
ld.shared.u32 r979, [r1591+4320];
barrier.sync 0;
st.shared.u32 [r1597], r664;
st.shared.u32 [r1597+24], r716;
st.shared.u32 [r1597+48], r753;
st.shared.u32 [r1597+72], r790;
st.shared.u32 [r1597+96], r827;
st.shared.u32 [r1597+120], r864;
barrier.sync 0;
ld.shared.u32 r899, [r1591];
ld.shared.u32 r987, [r1591+864];
ld.shared.u32 r896, [r1591+1728];
ld.shared.u32 r984, [r1591+2592];
ld.shared.u32 r897, [r1591+3456];
ld.shared.u32 r985, [r1591+4320];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r885, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r886, {low, high};
}
{
neg.f16x2 r887, r886;
}
{
add.f16x2 r889, r890, r891;
}
{
add.f16x2 r892, r893, r889;
}
{
add.f16x2 r895, r896, r897;
}
{
add.f16x2 r898, r899, r895;
}
{
add.f16x2 r901, r890, r891;
}
{
mul.f16x2 r904, r901, r885;
}
{
add.f16x2 r907, r893, r904;
}
{
sub.f16x2 r910, r896, r897;
}
{
mul.f16x2 r913, r910, r887;
}
{
add.f16x2 r916, r907, r913;
}
{
add.f16x2 r919, r890, r891;
}
{
mul.f16x2 r922, r919, r885;
}
{
add.f16x2 r925, r893, r922;
}
{
sub.f16x2 r928, r896, r897;
}
{
mul.f16x2 r931, r928, r887;
}
{
sub.f16x2 r934, r925, r931;
}
{
add.f16x2 r937, r896, r897;
}
{
mul.f16x2 r940, r937, r885;
}
{
add.f16x2 r943, r899, r940;
}
{
sub.f16x2 r946, r890, r891;
}
{
mul.f16x2 r949, r946, r887;
}
{
sub.f16x2 r952, r943, r949;
}
{
add.f16x2 r955, r896, r897;
}
{
mul.f16x2 r958, r955, r885;
}
{
add.f16x2 r961, r899, r958;
}
{
sub.f16x2 r964, r890, r891;
}
{
mul.f16x2 r967, r964, r887;
}
{
add.f16x2 r970, r961, r967;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r973, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r974, {low, high};
}
{
neg.f16x2 r975, r974;
}
{
add.f16x2 r977, r978, r979;
}
{
add.f16x2 r980, r981, r977;
}
{
add.f16x2 r983, r984, r985;
}
{
add.f16x2 r986, r987, r983;
}
{
add.f16x2 r989, r978, r979;
}
{
mul.f16x2 r992, r989, r973;
}
{
add.f16x2 r995, r981, r992;
}
{
sub.f16x2 r998, r984, r985;
}
{
mul.f16x2 r1001, r998, r975;
}
{
add.f16x2 r1004, r995, r1001;
}
{
add.f16x2 r1007, r978, r979;
}
{
mul.f16x2 r1010, r1007, r973;
}
{
add.f16x2 r1013, r981, r1010;
}
{
sub.f16x2 r1016, r984, r985;
}
{
mul.f16x2 r1019, r1016, r975;
}
{
sub.f16x2 r1022, r1013, r1019;
}
{
add.f16x2 r1025, r984, r985;
}
{
mul.f16x2 r1028, r1025, r973;
}
{
add.f16x2 r1031, r987, r1028;
}
{
sub.f16x2 r1034, r978, r979;
}
{
mul.f16x2 r1037, r1034, r975;
}
{
sub.f16x2 r1040, r1031, r1037;
}
{
add.f16x2 r1043, r984, r985;
}
{
mul.f16x2 r1046, r1043, r973;
}
{
add.f16x2 r1049, r987, r1046;
}
{
sub.f16x2 r1052, r978, r979;
}
{
mul.f16x2 r1055, r1052, r975;
}
{
add.f16x2 r1058, r1049, r1055;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f136;
cvt.rn.f16.f32 high, f136;
mov.b32 r1061, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r1062, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1063, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r1064, {low, high};
}
{
mul.f16x2 r1071, r1004, r1061;
}
{
mul.f16x2 r1074, r1040, r1062;
}
{
sub.f16x2 r1077, r1071, r1074;
}
{
mul.f16x2 r1080, r1004, r1062;
}
{
fma.rn.f16x2 r1083, r1040, r1061, r1080;
}
{
mul.f16x2 r1087, r1022, r1063;
}
{
mul.f16x2 r1090, r1058, r1064;
}
{
sub.f16x2 r1093, r1087, r1090;
}
{
mul.f16x2 r1096, r1022, r1064;
}
{
fma.rn.f16x2 r1099, r1058, r1063, r1096;
}
{
add.f16x2 r1103, r892, r980;
}
{
add.f16x2 r1106, r898, r986;
}
{
sub.f16x2 r1109, r892, r980;
}
{
sub.f16x2 r1112, r898, r986;
}
{
add.f16x2 r1115, r916, r1077;
}
{
add.f16x2 r1118, r952, r1083;
}
{
sub.f16x2 r1121, r916, r1077;
}
{
sub.f16x2 r1124, r952, r1083;
}
{
add.f16x2 r1127, r934, r1093;
}
{
add.f16x2 r1130, r970, r1099;
}
{
sub.f16x2 r1133, r934, r1093;
}
{
sub.f16x2 r1136, r970, r1099;
}
mul.wide.u32 rd6, r1588, 954437177;
shr.u64 rd7, rd6, 35;
cvt.u32.u64 r1598, rd7;
mul.lo.s32 r1599, r1598, 36;
sub.s32 r1600, r1588, r1599;
shl.b32 r1601, r1600, 2;
add.s32 r1602, r1589, r1601;
cvt.rn.f32.u32 f161, r1598;
mul.f32 f162, f161, 0f3E32B8C2;
cos.approx.f32 f113, f162;
sin.approx.f32 f163, f162;
neg.f32 f114, f163;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f113;
cvt.rn.f16.f32 high, f114;
mov.b32 r1139, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1142, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1144, {high, high};
}
{
mul.f16x2 r1146, r1118, r1144;
}
{
neg.f16x2 r1149, r1146;
}
{
fma.rn.f16x2 r1151, r1115, r1142, r1149;
}
{
mul.f16x2 r1155, r1115, r1144;
}
{
fma.rn.f16x2 r1158, r1118, r1142, r1155;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1162, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1164, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r1166, {low, high};
}
{
mul.f16x2 r1167, r1164, r1166;
}
{
mul.f16x2 r1170, r1139, r1162;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1173, {high, low};
}
{
fma.rn.f16x2 r1175, r1167, r1173, r1170;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1175;
mov.b32 r1179, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1175;
mov.b32 r1181, {high, high};
}
{
mul.f16x2 r1183, r1130, r1181;
}
{
neg.f16x2 r1186, r1183;
}
{
fma.rn.f16x2 r1188, r1127, r1179, r1186;
}
{
mul.f16x2 r1192, r1127, r1181;
}
{
fma.rn.f16x2 r1195, r1130, r1179, r1192;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1199, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1201, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r1203, {low, high};
}
{
mul.f16x2 r1204, r1201, r1203;
}
{
mul.f16x2 r1207, r1175, r1199;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1175;
mov.b32 r1210, {high, low};
}
{
fma.rn.f16x2 r1212, r1204, r1210, r1207;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1212;
mov.b32 r1216, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1212;
mov.b32 r1218, {high, high};
}
{
mul.f16x2 r1220, r1112, r1218;
}
{
neg.f16x2 r1223, r1220;
}
{
fma.rn.f16x2 r1225, r1109, r1216, r1223;
}
{
mul.f16x2 r1229, r1109, r1218;
}
{
fma.rn.f16x2 r1232, r1112, r1216, r1229;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1236, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1238, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r1240, {low, high};
}
{
mul.f16x2 r1241, r1238, r1240;
}
{
mul.f16x2 r1244, r1212, r1236;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1212;
mov.b32 r1247, {high, low};
}
{
fma.rn.f16x2 r1249, r1241, r1247, r1244;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1249;
mov.b32 r1253, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1249;
mov.b32 r1255, {high, high};
}
{
mul.f16x2 r1257, r1124, r1255;
}
{
neg.f16x2 r1260, r1257;
}
{
fma.rn.f16x2 r1262, r1121, r1253, r1260;
}
{
mul.f16x2 r1266, r1121, r1255;
}
{
fma.rn.f16x2 r1269, r1124, r1253, r1266;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1273, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1139;
mov.b32 r1275, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r1277, {low, high};
}
{
mul.f16x2 r1278, r1275, r1277;
}
{
mul.f16x2 r1281, r1249, r1273;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1249;
mov.b32 r1284, {high, low};
}
{
fma.rn.f16x2 r1286, r1278, r1284, r1281;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1286;
mov.b32 r1290, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1286;
mov.b32 r1292, {high, high};
}
{
mul.f16x2 r1294, r1136, r1292;
}
{
neg.f16x2 r1297, r1294;
}
{
fma.rn.f16x2 r1299, r1133, r1290, r1297;
}
{
mul.f16x2 r1303, r1133, r1292;
}
{
fma.rn.f16x2 r1306, r1136, r1290, r1303;
}
barrier.sync 0;
mad.lo.s32 r1603, r1598, 864, r1602;
st.shared.u32 [r1603], r1103;
st.shared.u32 [r1603+144], r1151;
st.shared.u32 [r1603+288], r1188;
st.shared.u32 [r1603+432], r1225;
st.shared.u32 [r1603+576], r1262;
st.shared.u32 [r1603+720], r1299;
barrier.sync 0;
ld.shared.u32 r1335, [r1591];
ld.shared.u32 r1423, [r1591+864];
ld.shared.u32 r1332, [r1591+1728];
ld.shared.u32 r1420, [r1591+2592];
ld.shared.u32 r1333, [r1591+3456];
ld.shared.u32 r1421, [r1591+4320];
barrier.sync 0;
st.shared.u32 [r1603], r1106;
st.shared.u32 [r1603+144], r1158;
st.shared.u32 [r1603+288], r1195;
st.shared.u32 [r1603+432], r1232;
st.shared.u32 [r1603+576], r1269;
st.shared.u32 [r1603+720], r1306;
barrier.sync 0;
ld.shared.u32 r1341, [r1591];
ld.shared.u32 r1429, [r1591+864];
ld.shared.u32 r1338, [r1591+1728];
ld.shared.u32 r1426, [r1591+2592];
ld.shared.u32 r1339, [r1591+3456];
ld.shared.u32 r1427, [r1591+4320];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1327, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r1328, {low, high};
}
{
neg.f16x2 r1329, r1328;
}
{
add.f16x2 r1331, r1332, r1333;
}
{
add.f16x2 r1334, r1335, r1331;
}
{
add.f16x2 r1337, r1338, r1339;
}
{
add.f16x2 r1340, r1341, r1337;
}
{
add.f16x2 r1343, r1332, r1333;
}
{
mul.f16x2 r1346, r1343, r1327;
}
{
add.f16x2 r1349, r1335, r1346;
}
{
sub.f16x2 r1352, r1338, r1339;
}
{
mul.f16x2 r1355, r1352, r1329;
}
{
add.f16x2 r1358, r1349, r1355;
}
{
add.f16x2 r1361, r1332, r1333;
}
{
mul.f16x2 r1364, r1361, r1327;
}
{
add.f16x2 r1367, r1335, r1364;
}
{
sub.f16x2 r1370, r1338, r1339;
}
{
mul.f16x2 r1373, r1370, r1329;
}
{
sub.f16x2 r1376, r1367, r1373;
}
{
add.f16x2 r1379, r1338, r1339;
}
{
mul.f16x2 r1382, r1379, r1327;
}
{
add.f16x2 r1385, r1341, r1382;
}
{
sub.f16x2 r1388, r1332, r1333;
}
{
mul.f16x2 r1391, r1388, r1329;
}
{
sub.f16x2 r1394, r1385, r1391;
}
{
add.f16x2 r1397, r1338, r1339;
}
{
mul.f16x2 r1400, r1397, r1327;
}
{
add.f16x2 r1403, r1341, r1400;
}
{
sub.f16x2 r1406, r1332, r1333;
}
{
mul.f16x2 r1409, r1406, r1329;
}
{
add.f16x2 r1412, r1403, r1409;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1415, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r1416, {low, high};
}
{
neg.f16x2 r1417, r1416;
}
{
add.f16x2 r1419, r1420, r1421;
}
{
add.f16x2 r1422, r1423, r1419;
}
{
add.f16x2 r1425, r1426, r1427;
}
{
add.f16x2 r1428, r1429, r1425;
}
{
add.f16x2 r1431, r1420, r1421;
}
{
mul.f16x2 r1434, r1431, r1415;
}
{
add.f16x2 r1437, r1423, r1434;
}
{
sub.f16x2 r1440, r1426, r1427;
}
{
mul.f16x2 r1443, r1440, r1417;
}
{
add.f16x2 r1446, r1437, r1443;
}
{
add.f16x2 r1449, r1420, r1421;
}
{
mul.f16x2 r1452, r1449, r1415;
}
{
add.f16x2 r1455, r1423, r1452;
}
{
sub.f16x2 r1458, r1426, r1427;
}
{
mul.f16x2 r1461, r1458, r1417;
}
{
sub.f16x2 r1464, r1455, r1461;
}
{
add.f16x2 r1467, r1426, r1427;
}
{
mul.f16x2 r1470, r1467, r1415;
}
{
add.f16x2 r1473, r1429, r1470;
}
{
sub.f16x2 r1476, r1420, r1421;
}
{
mul.f16x2 r1479, r1476, r1417;
}
{
sub.f16x2 r1482, r1473, r1479;
}
{
add.f16x2 r1485, r1426, r1427;
}
{
mul.f16x2 r1488, r1485, r1415;
}
{
add.f16x2 r1491, r1429, r1488;
}
{
sub.f16x2 r1494, r1420, r1421;
}
{
mul.f16x2 r1497, r1494, r1417;
}
{
add.f16x2 r1500, r1491, r1497;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f136;
cvt.rn.f16.f32 high, f136;
mov.b32 r1503, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r1504, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1505, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r1506, {low, high};
}
{
mul.f16x2 r1513, r1446, r1503;
}
{
mul.f16x2 r1516, r1482, r1504;
}
{
sub.f16x2 r1519, r1513, r1516;
}
{
mul.f16x2 r1522, r1446, r1504;
}
{
fma.rn.f16x2 r1525, r1482, r1503, r1522;
}
{
mul.f16x2 r1529, r1464, r1505;
}
{
mul.f16x2 r1532, r1500, r1506;
}
{
sub.f16x2 r1535, r1529, r1532;
}
{
mul.f16x2 r1538, r1464, r1506;
}
{
fma.rn.f16x2 r1541, r1500, r1505, r1538;
}
{
add.f16x2 %0, r1334, r1422;
}
{
add.f16x2 %1, r1340, r1428;
}
{
sub.f16x2 %6, r1334, r1422;
}
{
sub.f16x2 %7, r1340, r1428;
}
{
add.f16x2 %2, r1358, r1519;
}
{
add.f16x2 %3, r1394, r1525;
}
{
sub.f16x2 %8, r1358, r1519;
}
{
sub.f16x2 %9, r1394, r1525;
}
{
add.f16x2 %4, r1376, r1535;
}
{
add.f16x2 %5, r1412, r1541;
}
{
sub.f16x2 %10, r1376, r1535;
}
{
sub.f16x2 %11, r1412, r1541;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)));
};


#endif
