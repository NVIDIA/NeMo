#ifndef CUFFTDX_FFT_100_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_100_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1140, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<146>;
.reg .b32 r<1922>;
.reg .b64 rd<5>;
mov.u32 r1909, %tid.y;
shl.b32 r1910, r1909, 1;
mov.u32 r1911, %20;
mad.lo.s32 r1912, r1910, 400, r1911;
mov.u32 r1913, %tid.x;
mov.f32 f112, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f112;
cvt.rn.f16.f32 high, f112;
mov.b32 r1, {low, high};
}
mov.f32 f106, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r2, {low, high};
}
{
neg.f16x2 r3, r2;
}
mov.f32 f120, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f120;
cvt.rn.f16.f32 high, f120;
mov.b32 r5, {low, high};
}
mov.f32 f102, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f112;
cvt.rn.f16.f32 high, f112;
mov.b32 r9, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r10, {low, high};
}
{
add.f16x2 r11, %25, %37;
}
{
add.f16x2 r14, %21, r11;
}
{
add.f16x2 r17, %29, %33;
}
{
add.f16x2 r20, r14, r17;
}
{
add.f16x2 r23, %26, %38;
}
{
add.f16x2 r26, %22, r23;
}
{
add.f16x2 r29, %30, %34;
}
{
add.f16x2 r32, r26, r29;
}
{
add.f16x2 r35, %25, %37;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %21, r38;
}
{
add.f16x2 r44, %29, %33;
}
{
mul.f16x2 r47, r44, r5;
}
{
add.f16x2 r50, r41, r47;
}
{
sub.f16x2 r53, %26, %38;
}
{
mul.f16x2 r56, r53, r3;
}
{
sub.f16x2 r59, %30, %34;
}
{
mul.f16x2 r62, r59, r7;
}
{
add.f16x2 r65, r56, r62;
}
{
sub.f16x2 r68, r50, r65;
}
{
add.f16x2 r71, %25, %37;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %21, r74;
}
{
add.f16x2 r80, %29, %33;
}
{
mul.f16x2 r83, r80, r5;
}
{
add.f16x2 r86, r77, r83;
}
{
sub.f16x2 r89, %26, %38;
}
{
mul.f16x2 r92, r89, r3;
}
{
sub.f16x2 r95, %30, %34;
}
{
mul.f16x2 r98, r95, r7;
}
{
add.f16x2 r101, r92, r98;
}
{
add.f16x2 r104, r86, r101;
}
{
add.f16x2 r107, %25, %37;
}
{
mul.f16x2 r110, r107, r5;
}
{
add.f16x2 r113, %21, r110;
}
{
add.f16x2 r116, %29, %33;
}
{
mul.f16x2 r119, r116, r9;
}
{
add.f16x2 r122, r113, r119;
}
{
sub.f16x2 r125, %26, %38;
}
{
mul.f16x2 r128, r125, r7;
}
{
sub.f16x2 r131, %30, %34;
}
{
mul.f16x2 r134, r131, r10;
}
{
add.f16x2 r137, r128, r134;
}
{
sub.f16x2 r140, r122, r137;
}
{
add.f16x2 r143, %25, %37;
}
{
mul.f16x2 r146, r143, r5;
}
{
add.f16x2 r149, %21, r146;
}
{
add.f16x2 r152, %29, %33;
}
{
mul.f16x2 r155, r152, r9;
}
{
add.f16x2 r158, r149, r155;
}
{
sub.f16x2 r161, %26, %38;
}
{
mul.f16x2 r164, r161, r7;
}
{
sub.f16x2 r167, %30, %34;
}
{
mul.f16x2 r170, r167, r10;
}
{
add.f16x2 r173, r164, r170;
}
{
add.f16x2 r176, r158, r173;
}
{
add.f16x2 r179, %26, %38;
}
{
mul.f16x2 r182, r179, r1;
}
{
add.f16x2 r185, %22, r182;
}
{
add.f16x2 r188, %30, %34;
}
{
mul.f16x2 r191, r188, r5;
}
{
add.f16x2 r194, r185, r191;
}
{
sub.f16x2 r197, %25, %37;
}
{
mul.f16x2 r200, r197, r3;
}
{
sub.f16x2 r203, %29, %33;
}
{
mul.f16x2 r206, r203, r7;
}
{
add.f16x2 r209, r200, r206;
}
{
add.f16x2 r212, r194, r209;
}
{
add.f16x2 r215, %26, %38;
}
{
mul.f16x2 r218, r215, r1;
}
{
add.f16x2 r221, %22, r218;
}
{
add.f16x2 r224, %30, %34;
}
{
mul.f16x2 r227, r224, r5;
}
{
add.f16x2 r230, r221, r227;
}
{
sub.f16x2 r233, %25, %37;
}
{
mul.f16x2 r236, r233, r3;
}
{
sub.f16x2 r239, %29, %33;
}
{
mul.f16x2 r242, r239, r7;
}
{
add.f16x2 r245, r236, r242;
}
{
sub.f16x2 r248, r230, r245;
}
{
add.f16x2 r251, %26, %38;
}
{
mul.f16x2 r254, r251, r5;
}
{
add.f16x2 r257, %22, r254;
}
{
add.f16x2 r260, %30, %34;
}
{
mul.f16x2 r263, r260, r9;
}
{
add.f16x2 r266, r257, r263;
}
{
sub.f16x2 r269, %25, %37;
}
{
mul.f16x2 r272, r269, r7;
}
{
sub.f16x2 r275, %29, %33;
}
{
mul.f16x2 r278, r275, r10;
}
{
add.f16x2 r281, r272, r278;
}
{
add.f16x2 r284, r266, r281;
}
{
add.f16x2 r287, %26, %38;
}
{
mul.f16x2 r290, r287, r5;
}
{
add.f16x2 r293, %22, r290;
}
{
add.f16x2 r296, %30, %34;
}
{
mul.f16x2 r299, r296, r9;
}
{
add.f16x2 r302, r293, r299;
}
{
sub.f16x2 r305, %25, %37;
}
{
mul.f16x2 r308, r305, r7;
}
{
sub.f16x2 r311, %29, %33;
}
{
mul.f16x2 r314, r311, r10;
}
{
add.f16x2 r317, r308, r314;
}
{
sub.f16x2 r320, r302, r317;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f112;
cvt.rn.f16.f32 high, f112;
mov.b32 r323, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r324, {low, high};
}
{
neg.f16x2 r325, r324;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f120;
cvt.rn.f16.f32 high, f120;
mov.b32 r327, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r328, {low, high};
}
{
neg.f16x2 r329, r328;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f112;
cvt.rn.f16.f32 high, f112;
mov.b32 r331, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r332, {low, high};
}
{
add.f16x2 r333, %27, %39;
}
{
add.f16x2 r336, %23, r333;
}
{
add.f16x2 r339, %31, %35;
}
{
add.f16x2 r342, r336, r339;
}
{
add.f16x2 r345, %28, %40;
}
{
add.f16x2 r348, %24, r345;
}
{
add.f16x2 r351, %32, %36;
}
{
add.f16x2 r354, r348, r351;
}
{
add.f16x2 r357, %27, %39;
}
{
mul.f16x2 r360, r357, r323;
}
{
add.f16x2 r363, %23, r360;
}
{
add.f16x2 r366, %31, %35;
}
{
mul.f16x2 r369, r366, r327;
}
{
add.f16x2 r372, r363, r369;
}
{
sub.f16x2 r375, %28, %40;
}
{
mul.f16x2 r378, r375, r325;
}
{
sub.f16x2 r381, %32, %36;
}
{
mul.f16x2 r384, r381, r329;
}
{
add.f16x2 r387, r378, r384;
}
{
sub.f16x2 r390, r372, r387;
}
{
add.f16x2 r393, %27, %39;
}
{
mul.f16x2 r396, r393, r323;
}
{
add.f16x2 r399, %23, r396;
}
{
add.f16x2 r402, %31, %35;
}
{
mul.f16x2 r405, r402, r327;
}
{
add.f16x2 r408, r399, r405;
}
{
sub.f16x2 r411, %28, %40;
}
{
mul.f16x2 r414, r411, r325;
}
{
sub.f16x2 r417, %32, %36;
}
{
mul.f16x2 r420, r417, r329;
}
{
add.f16x2 r423, r414, r420;
}
{
add.f16x2 r426, r408, r423;
}
{
add.f16x2 r429, %27, %39;
}
{
mul.f16x2 r432, r429, r327;
}
{
add.f16x2 r435, %23, r432;
}
{
add.f16x2 r438, %31, %35;
}
{
mul.f16x2 r441, r438, r331;
}
{
add.f16x2 r444, r435, r441;
}
{
sub.f16x2 r447, %28, %40;
}
{
mul.f16x2 r450, r447, r329;
}
{
sub.f16x2 r453, %32, %36;
}
{
mul.f16x2 r456, r453, r332;
}
{
add.f16x2 r459, r450, r456;
}
{
sub.f16x2 r462, r444, r459;
}
{
add.f16x2 r465, %27, %39;
}
{
mul.f16x2 r468, r465, r327;
}
{
add.f16x2 r471, %23, r468;
}
{
add.f16x2 r474, %31, %35;
}
{
mul.f16x2 r477, r474, r331;
}
{
add.f16x2 r480, r471, r477;
}
{
sub.f16x2 r483, %28, %40;
}
{
mul.f16x2 r486, r483, r329;
}
{
sub.f16x2 r489, %32, %36;
}
{
mul.f16x2 r492, r489, r332;
}
{
add.f16x2 r495, r486, r492;
}
{
add.f16x2 r498, r480, r495;
}
{
add.f16x2 r501, %28, %40;
}
{
mul.f16x2 r504, r501, r323;
}
{
add.f16x2 r507, %24, r504;
}
{
add.f16x2 r510, %32, %36;
}
{
mul.f16x2 r513, r510, r327;
}
{
add.f16x2 r516, r507, r513;
}
{
sub.f16x2 r519, %27, %39;
}
{
mul.f16x2 r522, r519, r325;
}
{
sub.f16x2 r525, %31, %35;
}
{
mul.f16x2 r528, r525, r329;
}
{
add.f16x2 r531, r522, r528;
}
{
add.f16x2 r534, r516, r531;
}
{
add.f16x2 r537, %28, %40;
}
{
mul.f16x2 r540, r537, r323;
}
{
add.f16x2 r543, %24, r540;
}
{
add.f16x2 r546, %32, %36;
}
{
mul.f16x2 r549, r546, r327;
}
{
add.f16x2 r552, r543, r549;
}
{
sub.f16x2 r555, %27, %39;
}
{
mul.f16x2 r558, r555, r325;
}
{
sub.f16x2 r561, %31, %35;
}
{
mul.f16x2 r564, r561, r329;
}
{
add.f16x2 r567, r558, r564;
}
{
sub.f16x2 r570, r552, r567;
}
{
add.f16x2 r573, %28, %40;
}
{
mul.f16x2 r576, r573, r327;
}
{
add.f16x2 r579, %24, r576;
}
{
add.f16x2 r582, %32, %36;
}
{
mul.f16x2 r585, r582, r331;
}
{
add.f16x2 r588, r579, r585;
}
{
sub.f16x2 r591, %27, %39;
}
{
mul.f16x2 r594, r591, r329;
}
{
sub.f16x2 r597, %31, %35;
}
{
mul.f16x2 r600, r597, r332;
}
{
add.f16x2 r603, r594, r600;
}
{
add.f16x2 r606, r588, r603;
}
{
add.f16x2 r609, %28, %40;
}
{
mul.f16x2 r612, r609, r327;
}
{
add.f16x2 r615, %24, r612;
}
{
add.f16x2 r618, %32, %36;
}
{
mul.f16x2 r621, r618, r331;
}
{
add.f16x2 r624, r615, r621;
}
{
sub.f16x2 r627, %27, %39;
}
{
mul.f16x2 r630, r627, r329;
}
{
sub.f16x2 r633, %31, %35;
}
{
mul.f16x2 r636, r633, r332;
}
{
add.f16x2 r639, r630, r636;
}
{
sub.f16x2 r642, r624, r639;
}
mov.f32 f108, 0f3F4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r645, {low, high};
}
mov.f32 f122, 0f3F167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r646, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f112;
cvt.rn.f16.f32 high, f112;
mov.b32 r647, {low, high};
}
mov.f32 f118, 0f3F737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f118;
cvt.rn.f16.f32 high, f118;
mov.b32 r648, {low, high};
}
mov.f32 f116, 0fBE9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f116;
cvt.rn.f16.f32 high, f116;
mov.b32 r649, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f118;
cvt.rn.f16.f32 high, f118;
mov.b32 r650, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f120;
cvt.rn.f16.f32 high, f120;
mov.b32 r651, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r652, {low, high};
}
mov.f32 f79, 0fBF800000;
{
mul.f16x2 r663, r390, r645;
}
{
mul.f16x2 r666, r534, r646;
}
{
sub.f16x2 r669, r663, r666;
}
{
mul.f16x2 r672, r390, r646;
}
{
fma.rn.f16x2 r675, r534, r645, r672;
}
{
mul.f16x2 r679, r462, r647;
}
{
mul.f16x2 r682, r606, r648;
}
{
sub.f16x2 r685, r679, r682;
}
{
mul.f16x2 r688, r462, r648;
}
{
fma.rn.f16x2 r691, r606, r647, r688;
}
{
mul.f16x2 r695, r498, r649;
}
{
mul.f16x2 r698, r642, r650;
}
{
sub.f16x2 r701, r695, r698;
}
{
mul.f16x2 r704, r498, r650;
}
{
fma.rn.f16x2 r707, r642, r649, r704;
}
{
mul.f16x2 r711, r426, r651;
}
{
mul.f16x2 r714, r570, r652;
}
{
sub.f16x2 r717, r711, r714;
}
{
mul.f16x2 r720, r426, r652;
}
{
fma.rn.f16x2 r723, r570, r651, r720;
}
{
add.f16x2 r727, r20, r342;
}
{
add.f16x2 r730, r32, r354;
}
{
sub.f16x2 r733, r20, r342;
}
{
sub.f16x2 r736, r32, r354;
}
{
add.f16x2 r739, r68, r669;
}
{
add.f16x2 r742, r212, r675;
}
{
sub.f16x2 r745, r68, r669;
}
{
sub.f16x2 r748, r212, r675;
}
{
add.f16x2 r751, r140, r685;
}
{
add.f16x2 r754, r284, r691;
}
{
sub.f16x2 r757, r140, r685;
}
{
sub.f16x2 r760, r284, r691;
}
{
add.f16x2 r763, r176, r701;
}
{
add.f16x2 r766, r320, r707;
}
{
sub.f16x2 r769, r176, r701;
}
{
sub.f16x2 r772, r320, r707;
}
{
add.f16x2 r775, r104, r717;
}
{
add.f16x2 r778, r248, r723;
}
{
sub.f16x2 r781, r104, r717;
}
{
sub.f16x2 r784, r248, r723;
}
mul.wide.u32 rd2, r1913, -858993459;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r1914, rd3;
mul.lo.s32 r1915, r1914, 10;
sub.s32 r1916, r1913, r1915;
shr.u64 rd4, rd2, 34;
cvt.u32.u64 r1917, rd4;
and.b32 r1918, r1917, 1073741822;
mad.lo.s32 r1919, r1918, 400, r1912;
cvt.rn.f32.u32 f143, r1916;
mul.f32 f144, f143, 0f3D80ADFD;
cos.approx.f32 f61, f144;
sin.approx.f32 f145, f144;
neg.f32 f62, f145;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f61;
cvt.rn.f16.f32 high, f62;
mov.b32 r787, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r790, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r792, {high, high};
}
{
mul.f16x2 r794, r742, r792;
}
{
fma.rn.f16x2 r797, r739, r790, r794;
}
{
mul.f16x2 r801, r739, r792;
}
{
neg.f16x2 r804, r801;
}
{
fma.rn.f16x2 r806, r742, r790, r804;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r810, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r812, {high, high};
}
mov.f32 f80, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f79;
cvt.rn.f16.f32 high, f80;
mov.b32 r814, {low, high};
}
{
mul.f16x2 r815, r812, r814;
}
{
mul.f16x2 r818, r787, r810;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r821, {high, low};
}
{
fma.rn.f16x2 r823, r815, r821, r818;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r823;
mov.b32 r827, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r823;
mov.b32 r829, {high, high};
}
{
mul.f16x2 r831, r754, r829;
}
{
fma.rn.f16x2 r834, r751, r827, r831;
}
{
mul.f16x2 r838, r751, r829;
}
{
neg.f16x2 r841, r838;
}
{
fma.rn.f16x2 r843, r754, r827, r841;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r847, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r849, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f79;
cvt.rn.f16.f32 high, f80;
mov.b32 r851, {low, high};
}
{
mul.f16x2 r852, r849, r851;
}
{
mul.f16x2 r855, r823, r847;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r823;
mov.b32 r858, {high, low};
}
{
fma.rn.f16x2 r860, r852, r858, r855;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r860;
mov.b32 r864, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r860;
mov.b32 r866, {high, high};
}
{
mul.f16x2 r868, r766, r866;
}
{
fma.rn.f16x2 r871, r763, r864, r868;
}
{
mul.f16x2 r875, r763, r866;
}
{
neg.f16x2 r878, r875;
}
{
fma.rn.f16x2 r880, r766, r864, r878;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r884, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r886, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f79;
cvt.rn.f16.f32 high, f80;
mov.b32 r888, {low, high};
}
{
mul.f16x2 r889, r886, r888;
}
{
mul.f16x2 r892, r860, r884;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r860;
mov.b32 r895, {high, low};
}
{
fma.rn.f16x2 r897, r889, r895, r892;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r897;
mov.b32 r901, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r897;
mov.b32 r903, {high, high};
}
{
mul.f16x2 r905, r778, r903;
}
{
fma.rn.f16x2 r908, r775, r901, r905;
}
{
mul.f16x2 r912, r775, r903;
}
{
neg.f16x2 r915, r912;
}
{
fma.rn.f16x2 r917, r778, r901, r915;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r921, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r923, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f79;
cvt.rn.f16.f32 high, f80;
mov.b32 r925, {low, high};
}
{
mul.f16x2 r926, r923, r925;
}
{
mul.f16x2 r929, r897, r921;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r897;
mov.b32 r932, {high, low};
}
{
fma.rn.f16x2 r934, r926, r932, r929;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r934;
mov.b32 r938, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r934;
mov.b32 r940, {high, high};
}
{
mul.f16x2 r942, r736, r940;
}
{
fma.rn.f16x2 r945, r733, r938, r942;
}
{
mul.f16x2 r949, r733, r940;
}
{
neg.f16x2 r952, r949;
}
{
fma.rn.f16x2 r954, r736, r938, r952;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r958, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r960, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f79;
cvt.rn.f16.f32 high, f80;
mov.b32 r962, {low, high};
}
{
mul.f16x2 r963, r960, r962;
}
{
mul.f16x2 r966, r934, r958;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r934;
mov.b32 r969, {high, low};
}
{
fma.rn.f16x2 r971, r963, r969, r966;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r971;
mov.b32 r975, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r971;
mov.b32 r977, {high, high};
}
{
mul.f16x2 r979, r748, r977;
}
{
fma.rn.f16x2 r982, r745, r975, r979;
}
{
mul.f16x2 r986, r745, r977;
}
{
neg.f16x2 r989, r986;
}
{
fma.rn.f16x2 r991, r748, r975, r989;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r995, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r997, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f79;
cvt.rn.f16.f32 high, f80;
mov.b32 r999, {low, high};
}
{
mul.f16x2 r1000, r997, r999;
}
{
mul.f16x2 r1003, r971, r995;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r971;
mov.b32 r1006, {high, low};
}
{
fma.rn.f16x2 r1008, r1000, r1006, r1003;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1008;
mov.b32 r1012, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1008;
mov.b32 r1014, {high, high};
}
{
mul.f16x2 r1016, r760, r1014;
}
{
fma.rn.f16x2 r1019, r757, r1012, r1016;
}
{
mul.f16x2 r1023, r757, r1014;
}
{
neg.f16x2 r1026, r1023;
}
{
fma.rn.f16x2 r1028, r760, r1012, r1026;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r1032, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r1034, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f79;
cvt.rn.f16.f32 high, f80;
mov.b32 r1036, {low, high};
}
{
mul.f16x2 r1037, r1034, r1036;
}
{
mul.f16x2 r1040, r1008, r1032;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1008;
mov.b32 r1043, {high, low};
}
{
fma.rn.f16x2 r1045, r1037, r1043, r1040;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1045;
mov.b32 r1049, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1045;
mov.b32 r1051, {high, high};
}
{
mul.f16x2 r1053, r772, r1051;
}
{
fma.rn.f16x2 r1056, r769, r1049, r1053;
}
{
mul.f16x2 r1060, r769, r1051;
}
{
neg.f16x2 r1063, r1060;
}
{
fma.rn.f16x2 r1065, r772, r1049, r1063;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r1069, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r1071, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f79;
cvt.rn.f16.f32 high, f80;
mov.b32 r1073, {low, high};
}
{
mul.f16x2 r1074, r1071, r1073;
}
{
mul.f16x2 r1077, r1045, r1069;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1045;
mov.b32 r1080, {high, low};
}
{
fma.rn.f16x2 r1082, r1074, r1080, r1077;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1082;
mov.b32 r1086, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1082;
mov.b32 r1088, {high, high};
}
{
mul.f16x2 r1090, r784, r1088;
}
{
fma.rn.f16x2 r1093, r781, r1086, r1090;
}
{
mul.f16x2 r1097, r781, r1088;
}
{
neg.f16x2 r1100, r1097;
}
{
fma.rn.f16x2 r1102, r784, r1086, r1100;
}
barrier.sync 0;
mad.lo.s32 r1920, r1916, 80, r1919;
st.shared.v2.f32 [r1920], {r727, r730};
st.shared.v2.f32 [r1920+8], {r797, r806};
st.shared.v2.f32 [r1920+16], {r834, r843};
st.shared.v2.f32 [r1920+24], {r871, r880};
st.shared.v2.f32 [r1920+32], {r908, r917};
st.shared.v2.f32 [r1920+40], {r945, r954};
st.shared.v2.f32 [r1920+48], {r982, r991};
st.shared.v2.f32 [r1920+56], {r1019, r1028};
st.shared.v2.f32 [r1920+64], {r1056, r1065};
st.shared.v2.f32 [r1920+72], {r1093, r1102};
barrier.sync 0;
mad.lo.s32 r1921, r1916, -72, r1920;
ld.shared.u32 r1137, [r1921];
ld.shared.u32 r1149, [r1921+4];
ld.shared.u32 r1459, [r1921+80];
ld.shared.u32 r1471, [r1921+84];
ld.shared.u32 r1134, [r1921+160];
ld.shared.u32 r1146, [r1921+164];
ld.shared.u32 r1456, [r1921+240];
ld.shared.u32 r1468, [r1921+244];
ld.shared.u32 r1140, [r1921+320];
ld.shared.u32 r1152, [r1921+324];
ld.shared.u32 r1462, [r1921+400];
ld.shared.u32 r1474, [r1921+404];
ld.shared.u32 r1141, [r1921+480];
ld.shared.u32 r1153, [r1921+484];
ld.shared.u32 r1463, [r1921+560];
ld.shared.u32 r1475, [r1921+564];
ld.shared.u32 r1135, [r1921+640];
ld.shared.u32 r1147, [r1921+644];
ld.shared.u32 r1457, [r1921+720];
ld.shared.u32 r1469, [r1921+724];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f112;
cvt.rn.f16.f32 high, f112;
mov.b32 r1123, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1124, {low, high};
}
{
neg.f16x2 r1125, r1124;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f120;
cvt.rn.f16.f32 high, f120;
mov.b32 r1127, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r1128, {low, high};
}
{
neg.f16x2 r1129, r1128;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f112;
cvt.rn.f16.f32 high, f112;
mov.b32 r1131, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1132, {low, high};
}
{
add.f16x2 r1133, r1134, r1135;
}
{
add.f16x2 r1136, r1137, r1133;
}
{
add.f16x2 r1139, r1140, r1141;
}
{
add.f16x2 r1142, r1136, r1139;
}
{
add.f16x2 r1145, r1146, r1147;
}
{
add.f16x2 r1148, r1149, r1145;
}
{
add.f16x2 r1151, r1152, r1153;
}
{
add.f16x2 r1154, r1148, r1151;
}
{
add.f16x2 r1157, r1134, r1135;
}
{
mul.f16x2 r1160, r1157, r1123;
}
{
add.f16x2 r1163, r1137, r1160;
}
{
add.f16x2 r1166, r1140, r1141;
}
{
mul.f16x2 r1169, r1166, r1127;
}
{
add.f16x2 r1172, r1163, r1169;
}
{
sub.f16x2 r1175, r1146, r1147;
}
{
mul.f16x2 r1178, r1175, r1125;
}
{
sub.f16x2 r1181, r1152, r1153;
}
{
mul.f16x2 r1184, r1181, r1129;
}
{
add.f16x2 r1187, r1178, r1184;
}
{
sub.f16x2 r1190, r1172, r1187;
}
{
add.f16x2 r1193, r1134, r1135;
}
{
mul.f16x2 r1196, r1193, r1123;
}
{
add.f16x2 r1199, r1137, r1196;
}
{
add.f16x2 r1202, r1140, r1141;
}
{
mul.f16x2 r1205, r1202, r1127;
}
{
add.f16x2 r1208, r1199, r1205;
}
{
sub.f16x2 r1211, r1146, r1147;
}
{
mul.f16x2 r1214, r1211, r1125;
}
{
sub.f16x2 r1217, r1152, r1153;
}
{
mul.f16x2 r1220, r1217, r1129;
}
{
add.f16x2 r1223, r1214, r1220;
}
{
add.f16x2 r1226, r1208, r1223;
}
{
add.f16x2 r1229, r1134, r1135;
}
{
mul.f16x2 r1232, r1229, r1127;
}
{
add.f16x2 r1235, r1137, r1232;
}
{
add.f16x2 r1238, r1140, r1141;
}
{
mul.f16x2 r1241, r1238, r1131;
}
{
add.f16x2 r1244, r1235, r1241;
}
{
sub.f16x2 r1247, r1146, r1147;
}
{
mul.f16x2 r1250, r1247, r1129;
}
{
sub.f16x2 r1253, r1152, r1153;
}
{
mul.f16x2 r1256, r1253, r1132;
}
{
add.f16x2 r1259, r1250, r1256;
}
{
sub.f16x2 r1262, r1244, r1259;
}
{
add.f16x2 r1265, r1134, r1135;
}
{
mul.f16x2 r1268, r1265, r1127;
}
{
add.f16x2 r1271, r1137, r1268;
}
{
add.f16x2 r1274, r1140, r1141;
}
{
mul.f16x2 r1277, r1274, r1131;
}
{
add.f16x2 r1280, r1271, r1277;
}
{
sub.f16x2 r1283, r1146, r1147;
}
{
mul.f16x2 r1286, r1283, r1129;
}
{
sub.f16x2 r1289, r1152, r1153;
}
{
mul.f16x2 r1292, r1289, r1132;
}
{
add.f16x2 r1295, r1286, r1292;
}
{
add.f16x2 r1298, r1280, r1295;
}
{
add.f16x2 r1301, r1146, r1147;
}
{
mul.f16x2 r1304, r1301, r1123;
}
{
add.f16x2 r1307, r1149, r1304;
}
{
add.f16x2 r1310, r1152, r1153;
}
{
mul.f16x2 r1313, r1310, r1127;
}
{
add.f16x2 r1316, r1307, r1313;
}
{
sub.f16x2 r1319, r1134, r1135;
}
{
mul.f16x2 r1322, r1319, r1125;
}
{
sub.f16x2 r1325, r1140, r1141;
}
{
mul.f16x2 r1328, r1325, r1129;
}
{
add.f16x2 r1331, r1322, r1328;
}
{
add.f16x2 r1334, r1316, r1331;
}
{
add.f16x2 r1337, r1146, r1147;
}
{
mul.f16x2 r1340, r1337, r1123;
}
{
add.f16x2 r1343, r1149, r1340;
}
{
add.f16x2 r1346, r1152, r1153;
}
{
mul.f16x2 r1349, r1346, r1127;
}
{
add.f16x2 r1352, r1343, r1349;
}
{
sub.f16x2 r1355, r1134, r1135;
}
{
mul.f16x2 r1358, r1355, r1125;
}
{
sub.f16x2 r1361, r1140, r1141;
}
{
mul.f16x2 r1364, r1361, r1129;
}
{
add.f16x2 r1367, r1358, r1364;
}
{
sub.f16x2 r1370, r1352, r1367;
}
{
add.f16x2 r1373, r1146, r1147;
}
{
mul.f16x2 r1376, r1373, r1127;
}
{
add.f16x2 r1379, r1149, r1376;
}
{
add.f16x2 r1382, r1152, r1153;
}
{
mul.f16x2 r1385, r1382, r1131;
}
{
add.f16x2 r1388, r1379, r1385;
}
{
sub.f16x2 r1391, r1134, r1135;
}
{
mul.f16x2 r1394, r1391, r1129;
}
{
sub.f16x2 r1397, r1140, r1141;
}
{
mul.f16x2 r1400, r1397, r1132;
}
{
add.f16x2 r1403, r1394, r1400;
}
{
add.f16x2 r1406, r1388, r1403;
}
{
add.f16x2 r1409, r1146, r1147;
}
{
mul.f16x2 r1412, r1409, r1127;
}
{
add.f16x2 r1415, r1149, r1412;
}
{
add.f16x2 r1418, r1152, r1153;
}
{
mul.f16x2 r1421, r1418, r1131;
}
{
add.f16x2 r1424, r1415, r1421;
}
{
sub.f16x2 r1427, r1134, r1135;
}
{
mul.f16x2 r1430, r1427, r1129;
}
{
sub.f16x2 r1433, r1140, r1141;
}
{
mul.f16x2 r1436, r1433, r1132;
}
{
add.f16x2 r1439, r1430, r1436;
}
{
sub.f16x2 r1442, r1424, r1439;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f112;
cvt.rn.f16.f32 high, f112;
mov.b32 r1445, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1446, {low, high};
}
{
neg.f16x2 r1447, r1446;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f120;
cvt.rn.f16.f32 high, f120;
mov.b32 r1449, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r1450, {low, high};
}
{
neg.f16x2 r1451, r1450;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f112;
cvt.rn.f16.f32 high, f112;
mov.b32 r1453, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1454, {low, high};
}
{
add.f16x2 r1455, r1456, r1457;
}
{
add.f16x2 r1458, r1459, r1455;
}
{
add.f16x2 r1461, r1462, r1463;
}
{
add.f16x2 r1464, r1458, r1461;
}
{
add.f16x2 r1467, r1468, r1469;
}
{
add.f16x2 r1470, r1471, r1467;
}
{
add.f16x2 r1473, r1474, r1475;
}
{
add.f16x2 r1476, r1470, r1473;
}
{
add.f16x2 r1479, r1456, r1457;
}
{
mul.f16x2 r1482, r1479, r1445;
}
{
add.f16x2 r1485, r1459, r1482;
}
{
add.f16x2 r1488, r1462, r1463;
}
{
mul.f16x2 r1491, r1488, r1449;
}
{
add.f16x2 r1494, r1485, r1491;
}
{
sub.f16x2 r1497, r1468, r1469;
}
{
mul.f16x2 r1500, r1497, r1447;
}
{
sub.f16x2 r1503, r1474, r1475;
}
{
mul.f16x2 r1506, r1503, r1451;
}
{
add.f16x2 r1509, r1500, r1506;
}
{
sub.f16x2 r1512, r1494, r1509;
}
{
add.f16x2 r1515, r1456, r1457;
}
{
mul.f16x2 r1518, r1515, r1445;
}
{
add.f16x2 r1521, r1459, r1518;
}
{
add.f16x2 r1524, r1462, r1463;
}
{
mul.f16x2 r1527, r1524, r1449;
}
{
add.f16x2 r1530, r1521, r1527;
}
{
sub.f16x2 r1533, r1468, r1469;
}
{
mul.f16x2 r1536, r1533, r1447;
}
{
sub.f16x2 r1539, r1474, r1475;
}
{
mul.f16x2 r1542, r1539, r1451;
}
{
add.f16x2 r1545, r1536, r1542;
}
{
add.f16x2 r1548, r1530, r1545;
}
{
add.f16x2 r1551, r1456, r1457;
}
{
mul.f16x2 r1554, r1551, r1449;
}
{
add.f16x2 r1557, r1459, r1554;
}
{
add.f16x2 r1560, r1462, r1463;
}
{
mul.f16x2 r1563, r1560, r1453;
}
{
add.f16x2 r1566, r1557, r1563;
}
{
sub.f16x2 r1569, r1468, r1469;
}
{
mul.f16x2 r1572, r1569, r1451;
}
{
sub.f16x2 r1575, r1474, r1475;
}
{
mul.f16x2 r1578, r1575, r1454;
}
{
add.f16x2 r1581, r1572, r1578;
}
{
sub.f16x2 r1584, r1566, r1581;
}
{
add.f16x2 r1587, r1456, r1457;
}
{
mul.f16x2 r1590, r1587, r1449;
}
{
add.f16x2 r1593, r1459, r1590;
}
{
add.f16x2 r1596, r1462, r1463;
}
{
mul.f16x2 r1599, r1596, r1453;
}
{
add.f16x2 r1602, r1593, r1599;
}
{
sub.f16x2 r1605, r1468, r1469;
}
{
mul.f16x2 r1608, r1605, r1451;
}
{
sub.f16x2 r1611, r1474, r1475;
}
{
mul.f16x2 r1614, r1611, r1454;
}
{
add.f16x2 r1617, r1608, r1614;
}
{
add.f16x2 r1620, r1602, r1617;
}
{
add.f16x2 r1623, r1468, r1469;
}
{
mul.f16x2 r1626, r1623, r1445;
}
{
add.f16x2 r1629, r1471, r1626;
}
{
add.f16x2 r1632, r1474, r1475;
}
{
mul.f16x2 r1635, r1632, r1449;
}
{
add.f16x2 r1638, r1629, r1635;
}
{
sub.f16x2 r1641, r1456, r1457;
}
{
mul.f16x2 r1644, r1641, r1447;
}
{
sub.f16x2 r1647, r1462, r1463;
}
{
mul.f16x2 r1650, r1647, r1451;
}
{
add.f16x2 r1653, r1644, r1650;
}
{
add.f16x2 r1656, r1638, r1653;
}
{
add.f16x2 r1659, r1468, r1469;
}
{
mul.f16x2 r1662, r1659, r1445;
}
{
add.f16x2 r1665, r1471, r1662;
}
{
add.f16x2 r1668, r1474, r1475;
}
{
mul.f16x2 r1671, r1668, r1449;
}
{
add.f16x2 r1674, r1665, r1671;
}
{
sub.f16x2 r1677, r1456, r1457;
}
{
mul.f16x2 r1680, r1677, r1447;
}
{
sub.f16x2 r1683, r1462, r1463;
}
{
mul.f16x2 r1686, r1683, r1451;
}
{
add.f16x2 r1689, r1680, r1686;
}
{
sub.f16x2 r1692, r1674, r1689;
}
{
add.f16x2 r1695, r1468, r1469;
}
{
mul.f16x2 r1698, r1695, r1449;
}
{
add.f16x2 r1701, r1471, r1698;
}
{
add.f16x2 r1704, r1474, r1475;
}
{
mul.f16x2 r1707, r1704, r1453;
}
{
add.f16x2 r1710, r1701, r1707;
}
{
sub.f16x2 r1713, r1456, r1457;
}
{
mul.f16x2 r1716, r1713, r1451;
}
{
sub.f16x2 r1719, r1462, r1463;
}
{
mul.f16x2 r1722, r1719, r1454;
}
{
add.f16x2 r1725, r1716, r1722;
}
{
add.f16x2 r1728, r1710, r1725;
}
{
add.f16x2 r1731, r1468, r1469;
}
{
mul.f16x2 r1734, r1731, r1449;
}
{
add.f16x2 r1737, r1471, r1734;
}
{
add.f16x2 r1740, r1474, r1475;
}
{
mul.f16x2 r1743, r1740, r1453;
}
{
add.f16x2 r1746, r1737, r1743;
}
{
sub.f16x2 r1749, r1456, r1457;
}
{
mul.f16x2 r1752, r1749, r1451;
}
{
sub.f16x2 r1755, r1462, r1463;
}
{
mul.f16x2 r1758, r1755, r1454;
}
{
add.f16x2 r1761, r1752, r1758;
}
{
sub.f16x2 r1764, r1746, r1761;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1767, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r1768, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f112;
cvt.rn.f16.f32 high, f112;
mov.b32 r1769, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f118;
cvt.rn.f16.f32 high, f118;
mov.b32 r1770, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f116;
cvt.rn.f16.f32 high, f116;
mov.b32 r1771, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f118;
cvt.rn.f16.f32 high, f118;
mov.b32 r1772, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f120;
cvt.rn.f16.f32 high, f120;
mov.b32 r1773, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r1774, {low, high};
}
{
mul.f16x2 r1785, r1512, r1767;
}
{
mul.f16x2 r1788, r1656, r1768;
}
{
sub.f16x2 r1791, r1785, r1788;
}
{
mul.f16x2 r1794, r1512, r1768;
}
{
fma.rn.f16x2 r1797, r1656, r1767, r1794;
}
{
mul.f16x2 r1801, r1584, r1769;
}
{
mul.f16x2 r1804, r1728, r1770;
}
{
sub.f16x2 r1807, r1801, r1804;
}
{
mul.f16x2 r1810, r1584, r1770;
}
{
fma.rn.f16x2 r1813, r1728, r1769, r1810;
}
{
mul.f16x2 r1817, r1620, r1771;
}
{
mul.f16x2 r1820, r1764, r1772;
}
{
sub.f16x2 r1823, r1817, r1820;
}
{
mul.f16x2 r1826, r1620, r1772;
}
{
fma.rn.f16x2 r1829, r1764, r1771, r1826;
}
{
mul.f16x2 r1833, r1548, r1773;
}
{
mul.f16x2 r1836, r1692, r1774;
}
{
sub.f16x2 r1839, r1833, r1836;
}
{
mul.f16x2 r1842, r1548, r1774;
}
{
fma.rn.f16x2 r1845, r1692, r1773, r1842;
}
{
add.f16x2 %0, r1142, r1464;
}
{
add.f16x2 %1, r1154, r1476;
}
{
sub.f16x2 %10, r1142, r1464;
}
{
sub.f16x2 %11, r1154, r1476;
}
{
add.f16x2 %2, r1190, r1791;
}
{
add.f16x2 %3, r1334, r1797;
}
{
sub.f16x2 %12, r1190, r1791;
}
{
sub.f16x2 %13, r1334, r1797;
}
{
add.f16x2 %4, r1262, r1807;
}
{
add.f16x2 %5, r1406, r1813;
}
{
sub.f16x2 %14, r1262, r1807;
}
{
sub.f16x2 %15, r1406, r1813;
}
{
add.f16x2 %6, r1298, r1823;
}
{
add.f16x2 %7, r1442, r1829;
}
{
sub.f16x2 %16, r1298, r1823;
}
{
sub.f16x2 %17, r1442, r1829;
}
{
add.f16x2 %8, r1226, r1839;
}
{
add.f16x2 %9, r1370, r1845;
}
{
sub.f16x2 %18, r1226, r1839;
}
{
sub.f16x2 %19, r1370, r1845;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1139, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<146>;
.reg .b32 r<1919>;
.reg .b64 rd<4>;
mov.u32 r1909, %tid.y;
mov.u32 r1910, %20;
mad.lo.s32 r1911, r1909, 400, r1910;
mov.u32 r1912, %tid.x;
mov.f32 f112, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f112;
cvt.rn.f16.f32 high, f112;
mov.b32 r1, {low, high};
}
mov.f32 f106, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r2, {low, high};
}
{
neg.f16x2 r3, r2;
}
mov.f32 f120, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f120;
cvt.rn.f16.f32 high, f120;
mov.b32 r5, {low, high};
}
mov.f32 f102, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f112;
cvt.rn.f16.f32 high, f112;
mov.b32 r9, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r10, {low, high};
}
{
add.f16x2 r11, %25, %37;
}
{
add.f16x2 r14, %21, r11;
}
{
add.f16x2 r17, %29, %33;
}
{
add.f16x2 r20, r14, r17;
}
{
add.f16x2 r23, %26, %38;
}
{
add.f16x2 r26, %22, r23;
}
{
add.f16x2 r29, %30, %34;
}
{
add.f16x2 r32, r26, r29;
}
{
add.f16x2 r35, %25, %37;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %21, r38;
}
{
add.f16x2 r44, %29, %33;
}
{
mul.f16x2 r47, r44, r5;
}
{
add.f16x2 r50, r41, r47;
}
{
sub.f16x2 r53, %26, %38;
}
{
mul.f16x2 r56, r53, r3;
}
{
sub.f16x2 r59, %30, %34;
}
{
mul.f16x2 r62, r59, r7;
}
{
add.f16x2 r65, r56, r62;
}
{
sub.f16x2 r68, r50, r65;
}
{
add.f16x2 r71, %25, %37;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %21, r74;
}
{
add.f16x2 r80, %29, %33;
}
{
mul.f16x2 r83, r80, r5;
}
{
add.f16x2 r86, r77, r83;
}
{
sub.f16x2 r89, %26, %38;
}
{
mul.f16x2 r92, r89, r3;
}
{
sub.f16x2 r95, %30, %34;
}
{
mul.f16x2 r98, r95, r7;
}
{
add.f16x2 r101, r92, r98;
}
{
add.f16x2 r104, r86, r101;
}
{
add.f16x2 r107, %25, %37;
}
{
mul.f16x2 r110, r107, r5;
}
{
add.f16x2 r113, %21, r110;
}
{
add.f16x2 r116, %29, %33;
}
{
mul.f16x2 r119, r116, r9;
}
{
add.f16x2 r122, r113, r119;
}
{
sub.f16x2 r125, %26, %38;
}
{
mul.f16x2 r128, r125, r7;
}
{
sub.f16x2 r131, %30, %34;
}
{
mul.f16x2 r134, r131, r10;
}
{
add.f16x2 r137, r128, r134;
}
{
sub.f16x2 r140, r122, r137;
}
{
add.f16x2 r143, %25, %37;
}
{
mul.f16x2 r146, r143, r5;
}
{
add.f16x2 r149, %21, r146;
}
{
add.f16x2 r152, %29, %33;
}
{
mul.f16x2 r155, r152, r9;
}
{
add.f16x2 r158, r149, r155;
}
{
sub.f16x2 r161, %26, %38;
}
{
mul.f16x2 r164, r161, r7;
}
{
sub.f16x2 r167, %30, %34;
}
{
mul.f16x2 r170, r167, r10;
}
{
add.f16x2 r173, r164, r170;
}
{
add.f16x2 r176, r158, r173;
}
{
add.f16x2 r179, %26, %38;
}
{
mul.f16x2 r182, r179, r1;
}
{
add.f16x2 r185, %22, r182;
}
{
add.f16x2 r188, %30, %34;
}
{
mul.f16x2 r191, r188, r5;
}
{
add.f16x2 r194, r185, r191;
}
{
sub.f16x2 r197, %25, %37;
}
{
mul.f16x2 r200, r197, r3;
}
{
sub.f16x2 r203, %29, %33;
}
{
mul.f16x2 r206, r203, r7;
}
{
add.f16x2 r209, r200, r206;
}
{
add.f16x2 r212, r194, r209;
}
{
add.f16x2 r215, %26, %38;
}
{
mul.f16x2 r218, r215, r1;
}
{
add.f16x2 r221, %22, r218;
}
{
add.f16x2 r224, %30, %34;
}
{
mul.f16x2 r227, r224, r5;
}
{
add.f16x2 r230, r221, r227;
}
{
sub.f16x2 r233, %25, %37;
}
{
mul.f16x2 r236, r233, r3;
}
{
sub.f16x2 r239, %29, %33;
}
{
mul.f16x2 r242, r239, r7;
}
{
add.f16x2 r245, r236, r242;
}
{
sub.f16x2 r248, r230, r245;
}
{
add.f16x2 r251, %26, %38;
}
{
mul.f16x2 r254, r251, r5;
}
{
add.f16x2 r257, %22, r254;
}
{
add.f16x2 r260, %30, %34;
}
{
mul.f16x2 r263, r260, r9;
}
{
add.f16x2 r266, r257, r263;
}
{
sub.f16x2 r269, %25, %37;
}
{
mul.f16x2 r272, r269, r7;
}
{
sub.f16x2 r275, %29, %33;
}
{
mul.f16x2 r278, r275, r10;
}
{
add.f16x2 r281, r272, r278;
}
{
add.f16x2 r284, r266, r281;
}
{
add.f16x2 r287, %26, %38;
}
{
mul.f16x2 r290, r287, r5;
}
{
add.f16x2 r293, %22, r290;
}
{
add.f16x2 r296, %30, %34;
}
{
mul.f16x2 r299, r296, r9;
}
{
add.f16x2 r302, r293, r299;
}
{
sub.f16x2 r305, %25, %37;
}
{
mul.f16x2 r308, r305, r7;
}
{
sub.f16x2 r311, %29, %33;
}
{
mul.f16x2 r314, r311, r10;
}
{
add.f16x2 r317, r308, r314;
}
{
sub.f16x2 r320, r302, r317;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f112;
cvt.rn.f16.f32 high, f112;
mov.b32 r323, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r324, {low, high};
}
{
neg.f16x2 r325, r324;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f120;
cvt.rn.f16.f32 high, f120;
mov.b32 r327, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r328, {low, high};
}
{
neg.f16x2 r329, r328;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f112;
cvt.rn.f16.f32 high, f112;
mov.b32 r331, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r332, {low, high};
}
{
add.f16x2 r333, %27, %39;
}
{
add.f16x2 r336, %23, r333;
}
{
add.f16x2 r339, %31, %35;
}
{
add.f16x2 r342, r336, r339;
}
{
add.f16x2 r345, %28, %40;
}
{
add.f16x2 r348, %24, r345;
}
{
add.f16x2 r351, %32, %36;
}
{
add.f16x2 r354, r348, r351;
}
{
add.f16x2 r357, %27, %39;
}
{
mul.f16x2 r360, r357, r323;
}
{
add.f16x2 r363, %23, r360;
}
{
add.f16x2 r366, %31, %35;
}
{
mul.f16x2 r369, r366, r327;
}
{
add.f16x2 r372, r363, r369;
}
{
sub.f16x2 r375, %28, %40;
}
{
mul.f16x2 r378, r375, r325;
}
{
sub.f16x2 r381, %32, %36;
}
{
mul.f16x2 r384, r381, r329;
}
{
add.f16x2 r387, r378, r384;
}
{
sub.f16x2 r390, r372, r387;
}
{
add.f16x2 r393, %27, %39;
}
{
mul.f16x2 r396, r393, r323;
}
{
add.f16x2 r399, %23, r396;
}
{
add.f16x2 r402, %31, %35;
}
{
mul.f16x2 r405, r402, r327;
}
{
add.f16x2 r408, r399, r405;
}
{
sub.f16x2 r411, %28, %40;
}
{
mul.f16x2 r414, r411, r325;
}
{
sub.f16x2 r417, %32, %36;
}
{
mul.f16x2 r420, r417, r329;
}
{
add.f16x2 r423, r414, r420;
}
{
add.f16x2 r426, r408, r423;
}
{
add.f16x2 r429, %27, %39;
}
{
mul.f16x2 r432, r429, r327;
}
{
add.f16x2 r435, %23, r432;
}
{
add.f16x2 r438, %31, %35;
}
{
mul.f16x2 r441, r438, r331;
}
{
add.f16x2 r444, r435, r441;
}
{
sub.f16x2 r447, %28, %40;
}
{
mul.f16x2 r450, r447, r329;
}
{
sub.f16x2 r453, %32, %36;
}
{
mul.f16x2 r456, r453, r332;
}
{
add.f16x2 r459, r450, r456;
}
{
sub.f16x2 r462, r444, r459;
}
{
add.f16x2 r465, %27, %39;
}
{
mul.f16x2 r468, r465, r327;
}
{
add.f16x2 r471, %23, r468;
}
{
add.f16x2 r474, %31, %35;
}
{
mul.f16x2 r477, r474, r331;
}
{
add.f16x2 r480, r471, r477;
}
{
sub.f16x2 r483, %28, %40;
}
{
mul.f16x2 r486, r483, r329;
}
{
sub.f16x2 r489, %32, %36;
}
{
mul.f16x2 r492, r489, r332;
}
{
add.f16x2 r495, r486, r492;
}
{
add.f16x2 r498, r480, r495;
}
{
add.f16x2 r501, %28, %40;
}
{
mul.f16x2 r504, r501, r323;
}
{
add.f16x2 r507, %24, r504;
}
{
add.f16x2 r510, %32, %36;
}
{
mul.f16x2 r513, r510, r327;
}
{
add.f16x2 r516, r507, r513;
}
{
sub.f16x2 r519, %27, %39;
}
{
mul.f16x2 r522, r519, r325;
}
{
sub.f16x2 r525, %31, %35;
}
{
mul.f16x2 r528, r525, r329;
}
{
add.f16x2 r531, r522, r528;
}
{
add.f16x2 r534, r516, r531;
}
{
add.f16x2 r537, %28, %40;
}
{
mul.f16x2 r540, r537, r323;
}
{
add.f16x2 r543, %24, r540;
}
{
add.f16x2 r546, %32, %36;
}
{
mul.f16x2 r549, r546, r327;
}
{
add.f16x2 r552, r543, r549;
}
{
sub.f16x2 r555, %27, %39;
}
{
mul.f16x2 r558, r555, r325;
}
{
sub.f16x2 r561, %31, %35;
}
{
mul.f16x2 r564, r561, r329;
}
{
add.f16x2 r567, r558, r564;
}
{
sub.f16x2 r570, r552, r567;
}
{
add.f16x2 r573, %28, %40;
}
{
mul.f16x2 r576, r573, r327;
}
{
add.f16x2 r579, %24, r576;
}
{
add.f16x2 r582, %32, %36;
}
{
mul.f16x2 r585, r582, r331;
}
{
add.f16x2 r588, r579, r585;
}
{
sub.f16x2 r591, %27, %39;
}
{
mul.f16x2 r594, r591, r329;
}
{
sub.f16x2 r597, %31, %35;
}
{
mul.f16x2 r600, r597, r332;
}
{
add.f16x2 r603, r594, r600;
}
{
add.f16x2 r606, r588, r603;
}
{
add.f16x2 r609, %28, %40;
}
{
mul.f16x2 r612, r609, r327;
}
{
add.f16x2 r615, %24, r612;
}
{
add.f16x2 r618, %32, %36;
}
{
mul.f16x2 r621, r618, r331;
}
{
add.f16x2 r624, r615, r621;
}
{
sub.f16x2 r627, %27, %39;
}
{
mul.f16x2 r630, r627, r329;
}
{
sub.f16x2 r633, %31, %35;
}
{
mul.f16x2 r636, r633, r332;
}
{
add.f16x2 r639, r630, r636;
}
{
sub.f16x2 r642, r624, r639;
}
mov.f32 f108, 0f3F4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r645, {low, high};
}
mov.f32 f122, 0f3F167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r646, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f112;
cvt.rn.f16.f32 high, f112;
mov.b32 r647, {low, high};
}
mov.f32 f118, 0f3F737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f118;
cvt.rn.f16.f32 high, f118;
mov.b32 r648, {low, high};
}
mov.f32 f116, 0fBE9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f116;
cvt.rn.f16.f32 high, f116;
mov.b32 r649, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f118;
cvt.rn.f16.f32 high, f118;
mov.b32 r650, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f120;
cvt.rn.f16.f32 high, f120;
mov.b32 r651, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r652, {low, high};
}
mov.f32 f79, 0fBF800000;
{
mul.f16x2 r663, r390, r645;
}
{
mul.f16x2 r666, r534, r646;
}
{
sub.f16x2 r669, r663, r666;
}
{
mul.f16x2 r672, r390, r646;
}
{
fma.rn.f16x2 r675, r534, r645, r672;
}
{
mul.f16x2 r679, r462, r647;
}
{
mul.f16x2 r682, r606, r648;
}
{
sub.f16x2 r685, r679, r682;
}
{
mul.f16x2 r688, r462, r648;
}
{
fma.rn.f16x2 r691, r606, r647, r688;
}
{
mul.f16x2 r695, r498, r649;
}
{
mul.f16x2 r698, r642, r650;
}
{
sub.f16x2 r701, r695, r698;
}
{
mul.f16x2 r704, r498, r650;
}
{
fma.rn.f16x2 r707, r642, r649, r704;
}
{
mul.f16x2 r711, r426, r651;
}
{
mul.f16x2 r714, r570, r652;
}
{
sub.f16x2 r717, r711, r714;
}
{
mul.f16x2 r720, r426, r652;
}
{
fma.rn.f16x2 r723, r570, r651, r720;
}
{
add.f16x2 r727, r20, r342;
}
{
add.f16x2 r730, r32, r354;
}
{
sub.f16x2 r733, r20, r342;
}
{
sub.f16x2 r736, r32, r354;
}
{
add.f16x2 r739, r68, r669;
}
{
add.f16x2 r742, r212, r675;
}
{
sub.f16x2 r745, r68, r669;
}
{
sub.f16x2 r748, r212, r675;
}
{
add.f16x2 r751, r140, r685;
}
{
add.f16x2 r754, r284, r691;
}
{
sub.f16x2 r757, r140, r685;
}
{
sub.f16x2 r760, r284, r691;
}
{
add.f16x2 r763, r176, r701;
}
{
add.f16x2 r766, r320, r707;
}
{
sub.f16x2 r769, r176, r701;
}
{
sub.f16x2 r772, r320, r707;
}
{
add.f16x2 r775, r104, r717;
}
{
add.f16x2 r778, r248, r723;
}
{
sub.f16x2 r781, r104, r717;
}
{
sub.f16x2 r784, r248, r723;
}
mul.wide.u32 rd2, r1912, -858993459;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r1913, rd3;
mul.lo.s32 r1914, r1913, 10;
sub.s32 r1915, r1912, r1914;
mad.lo.s32 r1916, r1913, 400, r1911;
cvt.rn.f32.u32 f143, r1915;
mul.f32 f144, f143, 0f3D80ADFD;
cos.approx.f32 f61, f144;
sin.approx.f32 f145, f144;
neg.f32 f62, f145;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f61;
cvt.rn.f16.f32 high, f62;
mov.b32 r787, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r790, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r792, {high, high};
}
{
mul.f16x2 r794, r742, r792;
}
{
fma.rn.f16x2 r797, r739, r790, r794;
}
{
mul.f16x2 r801, r739, r792;
}
{
neg.f16x2 r804, r801;
}
{
fma.rn.f16x2 r806, r742, r790, r804;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r810, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r812, {high, high};
}
mov.f32 f80, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f79;
cvt.rn.f16.f32 high, f80;
mov.b32 r814, {low, high};
}
{
mul.f16x2 r815, r812, r814;
}
{
mul.f16x2 r818, r787, r810;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r821, {high, low};
}
{
fma.rn.f16x2 r823, r815, r821, r818;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r823;
mov.b32 r827, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r823;
mov.b32 r829, {high, high};
}
{
mul.f16x2 r831, r754, r829;
}
{
fma.rn.f16x2 r834, r751, r827, r831;
}
{
mul.f16x2 r838, r751, r829;
}
{
neg.f16x2 r841, r838;
}
{
fma.rn.f16x2 r843, r754, r827, r841;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r847, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r849, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f79;
cvt.rn.f16.f32 high, f80;
mov.b32 r851, {low, high};
}
{
mul.f16x2 r852, r849, r851;
}
{
mul.f16x2 r855, r823, r847;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r823;
mov.b32 r858, {high, low};
}
{
fma.rn.f16x2 r860, r852, r858, r855;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r860;
mov.b32 r864, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r860;
mov.b32 r866, {high, high};
}
{
mul.f16x2 r868, r766, r866;
}
{
fma.rn.f16x2 r871, r763, r864, r868;
}
{
mul.f16x2 r875, r763, r866;
}
{
neg.f16x2 r878, r875;
}
{
fma.rn.f16x2 r880, r766, r864, r878;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r884, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r886, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f79;
cvt.rn.f16.f32 high, f80;
mov.b32 r888, {low, high};
}
{
mul.f16x2 r889, r886, r888;
}
{
mul.f16x2 r892, r860, r884;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r860;
mov.b32 r895, {high, low};
}
{
fma.rn.f16x2 r897, r889, r895, r892;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r897;
mov.b32 r901, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r897;
mov.b32 r903, {high, high};
}
{
mul.f16x2 r905, r778, r903;
}
{
fma.rn.f16x2 r908, r775, r901, r905;
}
{
mul.f16x2 r912, r775, r903;
}
{
neg.f16x2 r915, r912;
}
{
fma.rn.f16x2 r917, r778, r901, r915;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r921, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r923, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f79;
cvt.rn.f16.f32 high, f80;
mov.b32 r925, {low, high};
}
{
mul.f16x2 r926, r923, r925;
}
{
mul.f16x2 r929, r897, r921;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r897;
mov.b32 r932, {high, low};
}
{
fma.rn.f16x2 r934, r926, r932, r929;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r934;
mov.b32 r938, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r934;
mov.b32 r940, {high, high};
}
{
mul.f16x2 r942, r736, r940;
}
{
fma.rn.f16x2 r945, r733, r938, r942;
}
{
mul.f16x2 r949, r733, r940;
}
{
neg.f16x2 r952, r949;
}
{
fma.rn.f16x2 r954, r736, r938, r952;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r958, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r960, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f79;
cvt.rn.f16.f32 high, f80;
mov.b32 r962, {low, high};
}
{
mul.f16x2 r963, r960, r962;
}
{
mul.f16x2 r966, r934, r958;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r934;
mov.b32 r969, {high, low};
}
{
fma.rn.f16x2 r971, r963, r969, r966;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r971;
mov.b32 r975, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r971;
mov.b32 r977, {high, high};
}
{
mul.f16x2 r979, r748, r977;
}
{
fma.rn.f16x2 r982, r745, r975, r979;
}
{
mul.f16x2 r986, r745, r977;
}
{
neg.f16x2 r989, r986;
}
{
fma.rn.f16x2 r991, r748, r975, r989;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r995, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r997, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f79;
cvt.rn.f16.f32 high, f80;
mov.b32 r999, {low, high};
}
{
mul.f16x2 r1000, r997, r999;
}
{
mul.f16x2 r1003, r971, r995;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r971;
mov.b32 r1006, {high, low};
}
{
fma.rn.f16x2 r1008, r1000, r1006, r1003;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1008;
mov.b32 r1012, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1008;
mov.b32 r1014, {high, high};
}
{
mul.f16x2 r1016, r760, r1014;
}
{
fma.rn.f16x2 r1019, r757, r1012, r1016;
}
{
mul.f16x2 r1023, r757, r1014;
}
{
neg.f16x2 r1026, r1023;
}
{
fma.rn.f16x2 r1028, r760, r1012, r1026;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r1032, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r1034, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f79;
cvt.rn.f16.f32 high, f80;
mov.b32 r1036, {low, high};
}
{
mul.f16x2 r1037, r1034, r1036;
}
{
mul.f16x2 r1040, r1008, r1032;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1008;
mov.b32 r1043, {high, low};
}
{
fma.rn.f16x2 r1045, r1037, r1043, r1040;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1045;
mov.b32 r1049, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1045;
mov.b32 r1051, {high, high};
}
{
mul.f16x2 r1053, r772, r1051;
}
{
fma.rn.f16x2 r1056, r769, r1049, r1053;
}
{
mul.f16x2 r1060, r769, r1051;
}
{
neg.f16x2 r1063, r1060;
}
{
fma.rn.f16x2 r1065, r772, r1049, r1063;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r1069, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r787;
mov.b32 r1071, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f79;
cvt.rn.f16.f32 high, f80;
mov.b32 r1073, {low, high};
}
{
mul.f16x2 r1074, r1071, r1073;
}
{
mul.f16x2 r1077, r1045, r1069;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1045;
mov.b32 r1080, {high, low};
}
{
fma.rn.f16x2 r1082, r1074, r1080, r1077;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1082;
mov.b32 r1086, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1082;
mov.b32 r1088, {high, high};
}
{
mul.f16x2 r1090, r784, r1088;
}
{
fma.rn.f16x2 r1093, r781, r1086, r1090;
}
{
mul.f16x2 r1097, r781, r1088;
}
{
neg.f16x2 r1100, r1097;
}
{
fma.rn.f16x2 r1102, r784, r1086, r1100;
}
barrier.sync 0;
mad.lo.s32 r1917, r1915, 40, r1916;
st.shared.v2.f32 [r1917], {r727, r797};
st.shared.v2.f32 [r1917+8], {r834, r871};
st.shared.v2.f32 [r1917+16], {r908, r945};
st.shared.v2.f32 [r1917+24], {r982, r1019};
st.shared.v2.f32 [r1917+32], {r1056, r1093};
barrier.sync 0;
mad.lo.s32 r1918, r1915, -36, r1917;
ld.shared.u32 r1137, [r1918];
ld.shared.u32 r1459, [r1918+40];
ld.shared.u32 r1134, [r1918+80];
ld.shared.u32 r1456, [r1918+120];
ld.shared.u32 r1140, [r1918+160];
ld.shared.u32 r1462, [r1918+200];
ld.shared.u32 r1141, [r1918+240];
ld.shared.u32 r1463, [r1918+280];
ld.shared.u32 r1135, [r1918+320];
ld.shared.u32 r1457, [r1918+360];
barrier.sync 0;
st.shared.v2.f32 [r1917], {r730, r806};
st.shared.v2.f32 [r1917+8], {r843, r880};
st.shared.v2.f32 [r1917+16], {r917, r954};
st.shared.v2.f32 [r1917+24], {r991, r1028};
st.shared.v2.f32 [r1917+32], {r1065, r1102};
barrier.sync 0;
ld.shared.u32 r1149, [r1918];
ld.shared.u32 r1471, [r1918+40];
ld.shared.u32 r1146, [r1918+80];
ld.shared.u32 r1468, [r1918+120];
ld.shared.u32 r1152, [r1918+160];
ld.shared.u32 r1474, [r1918+200];
ld.shared.u32 r1153, [r1918+240];
ld.shared.u32 r1475, [r1918+280];
ld.shared.u32 r1147, [r1918+320];
ld.shared.u32 r1469, [r1918+360];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f112;
cvt.rn.f16.f32 high, f112;
mov.b32 r1123, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1124, {low, high};
}
{
neg.f16x2 r1125, r1124;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f120;
cvt.rn.f16.f32 high, f120;
mov.b32 r1127, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r1128, {low, high};
}
{
neg.f16x2 r1129, r1128;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f112;
cvt.rn.f16.f32 high, f112;
mov.b32 r1131, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1132, {low, high};
}
{
add.f16x2 r1133, r1134, r1135;
}
{
add.f16x2 r1136, r1137, r1133;
}
{
add.f16x2 r1139, r1140, r1141;
}
{
add.f16x2 r1142, r1136, r1139;
}
{
add.f16x2 r1145, r1146, r1147;
}
{
add.f16x2 r1148, r1149, r1145;
}
{
add.f16x2 r1151, r1152, r1153;
}
{
add.f16x2 r1154, r1148, r1151;
}
{
add.f16x2 r1157, r1134, r1135;
}
{
mul.f16x2 r1160, r1157, r1123;
}
{
add.f16x2 r1163, r1137, r1160;
}
{
add.f16x2 r1166, r1140, r1141;
}
{
mul.f16x2 r1169, r1166, r1127;
}
{
add.f16x2 r1172, r1163, r1169;
}
{
sub.f16x2 r1175, r1146, r1147;
}
{
mul.f16x2 r1178, r1175, r1125;
}
{
sub.f16x2 r1181, r1152, r1153;
}
{
mul.f16x2 r1184, r1181, r1129;
}
{
add.f16x2 r1187, r1178, r1184;
}
{
sub.f16x2 r1190, r1172, r1187;
}
{
add.f16x2 r1193, r1134, r1135;
}
{
mul.f16x2 r1196, r1193, r1123;
}
{
add.f16x2 r1199, r1137, r1196;
}
{
add.f16x2 r1202, r1140, r1141;
}
{
mul.f16x2 r1205, r1202, r1127;
}
{
add.f16x2 r1208, r1199, r1205;
}
{
sub.f16x2 r1211, r1146, r1147;
}
{
mul.f16x2 r1214, r1211, r1125;
}
{
sub.f16x2 r1217, r1152, r1153;
}
{
mul.f16x2 r1220, r1217, r1129;
}
{
add.f16x2 r1223, r1214, r1220;
}
{
add.f16x2 r1226, r1208, r1223;
}
{
add.f16x2 r1229, r1134, r1135;
}
{
mul.f16x2 r1232, r1229, r1127;
}
{
add.f16x2 r1235, r1137, r1232;
}
{
add.f16x2 r1238, r1140, r1141;
}
{
mul.f16x2 r1241, r1238, r1131;
}
{
add.f16x2 r1244, r1235, r1241;
}
{
sub.f16x2 r1247, r1146, r1147;
}
{
mul.f16x2 r1250, r1247, r1129;
}
{
sub.f16x2 r1253, r1152, r1153;
}
{
mul.f16x2 r1256, r1253, r1132;
}
{
add.f16x2 r1259, r1250, r1256;
}
{
sub.f16x2 r1262, r1244, r1259;
}
{
add.f16x2 r1265, r1134, r1135;
}
{
mul.f16x2 r1268, r1265, r1127;
}
{
add.f16x2 r1271, r1137, r1268;
}
{
add.f16x2 r1274, r1140, r1141;
}
{
mul.f16x2 r1277, r1274, r1131;
}
{
add.f16x2 r1280, r1271, r1277;
}
{
sub.f16x2 r1283, r1146, r1147;
}
{
mul.f16x2 r1286, r1283, r1129;
}
{
sub.f16x2 r1289, r1152, r1153;
}
{
mul.f16x2 r1292, r1289, r1132;
}
{
add.f16x2 r1295, r1286, r1292;
}
{
add.f16x2 r1298, r1280, r1295;
}
{
add.f16x2 r1301, r1146, r1147;
}
{
mul.f16x2 r1304, r1301, r1123;
}
{
add.f16x2 r1307, r1149, r1304;
}
{
add.f16x2 r1310, r1152, r1153;
}
{
mul.f16x2 r1313, r1310, r1127;
}
{
add.f16x2 r1316, r1307, r1313;
}
{
sub.f16x2 r1319, r1134, r1135;
}
{
mul.f16x2 r1322, r1319, r1125;
}
{
sub.f16x2 r1325, r1140, r1141;
}
{
mul.f16x2 r1328, r1325, r1129;
}
{
add.f16x2 r1331, r1322, r1328;
}
{
add.f16x2 r1334, r1316, r1331;
}
{
add.f16x2 r1337, r1146, r1147;
}
{
mul.f16x2 r1340, r1337, r1123;
}
{
add.f16x2 r1343, r1149, r1340;
}
{
add.f16x2 r1346, r1152, r1153;
}
{
mul.f16x2 r1349, r1346, r1127;
}
{
add.f16x2 r1352, r1343, r1349;
}
{
sub.f16x2 r1355, r1134, r1135;
}
{
mul.f16x2 r1358, r1355, r1125;
}
{
sub.f16x2 r1361, r1140, r1141;
}
{
mul.f16x2 r1364, r1361, r1129;
}
{
add.f16x2 r1367, r1358, r1364;
}
{
sub.f16x2 r1370, r1352, r1367;
}
{
add.f16x2 r1373, r1146, r1147;
}
{
mul.f16x2 r1376, r1373, r1127;
}
{
add.f16x2 r1379, r1149, r1376;
}
{
add.f16x2 r1382, r1152, r1153;
}
{
mul.f16x2 r1385, r1382, r1131;
}
{
add.f16x2 r1388, r1379, r1385;
}
{
sub.f16x2 r1391, r1134, r1135;
}
{
mul.f16x2 r1394, r1391, r1129;
}
{
sub.f16x2 r1397, r1140, r1141;
}
{
mul.f16x2 r1400, r1397, r1132;
}
{
add.f16x2 r1403, r1394, r1400;
}
{
add.f16x2 r1406, r1388, r1403;
}
{
add.f16x2 r1409, r1146, r1147;
}
{
mul.f16x2 r1412, r1409, r1127;
}
{
add.f16x2 r1415, r1149, r1412;
}
{
add.f16x2 r1418, r1152, r1153;
}
{
mul.f16x2 r1421, r1418, r1131;
}
{
add.f16x2 r1424, r1415, r1421;
}
{
sub.f16x2 r1427, r1134, r1135;
}
{
mul.f16x2 r1430, r1427, r1129;
}
{
sub.f16x2 r1433, r1140, r1141;
}
{
mul.f16x2 r1436, r1433, r1132;
}
{
add.f16x2 r1439, r1430, r1436;
}
{
sub.f16x2 r1442, r1424, r1439;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f112;
cvt.rn.f16.f32 high, f112;
mov.b32 r1445, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1446, {low, high};
}
{
neg.f16x2 r1447, r1446;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f120;
cvt.rn.f16.f32 high, f120;
mov.b32 r1449, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r1450, {low, high};
}
{
neg.f16x2 r1451, r1450;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f112;
cvt.rn.f16.f32 high, f112;
mov.b32 r1453, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1454, {low, high};
}
{
add.f16x2 r1455, r1456, r1457;
}
{
add.f16x2 r1458, r1459, r1455;
}
{
add.f16x2 r1461, r1462, r1463;
}
{
add.f16x2 r1464, r1458, r1461;
}
{
add.f16x2 r1467, r1468, r1469;
}
{
add.f16x2 r1470, r1471, r1467;
}
{
add.f16x2 r1473, r1474, r1475;
}
{
add.f16x2 r1476, r1470, r1473;
}
{
add.f16x2 r1479, r1456, r1457;
}
{
mul.f16x2 r1482, r1479, r1445;
}
{
add.f16x2 r1485, r1459, r1482;
}
{
add.f16x2 r1488, r1462, r1463;
}
{
mul.f16x2 r1491, r1488, r1449;
}
{
add.f16x2 r1494, r1485, r1491;
}
{
sub.f16x2 r1497, r1468, r1469;
}
{
mul.f16x2 r1500, r1497, r1447;
}
{
sub.f16x2 r1503, r1474, r1475;
}
{
mul.f16x2 r1506, r1503, r1451;
}
{
add.f16x2 r1509, r1500, r1506;
}
{
sub.f16x2 r1512, r1494, r1509;
}
{
add.f16x2 r1515, r1456, r1457;
}
{
mul.f16x2 r1518, r1515, r1445;
}
{
add.f16x2 r1521, r1459, r1518;
}
{
add.f16x2 r1524, r1462, r1463;
}
{
mul.f16x2 r1527, r1524, r1449;
}
{
add.f16x2 r1530, r1521, r1527;
}
{
sub.f16x2 r1533, r1468, r1469;
}
{
mul.f16x2 r1536, r1533, r1447;
}
{
sub.f16x2 r1539, r1474, r1475;
}
{
mul.f16x2 r1542, r1539, r1451;
}
{
add.f16x2 r1545, r1536, r1542;
}
{
add.f16x2 r1548, r1530, r1545;
}
{
add.f16x2 r1551, r1456, r1457;
}
{
mul.f16x2 r1554, r1551, r1449;
}
{
add.f16x2 r1557, r1459, r1554;
}
{
add.f16x2 r1560, r1462, r1463;
}
{
mul.f16x2 r1563, r1560, r1453;
}
{
add.f16x2 r1566, r1557, r1563;
}
{
sub.f16x2 r1569, r1468, r1469;
}
{
mul.f16x2 r1572, r1569, r1451;
}
{
sub.f16x2 r1575, r1474, r1475;
}
{
mul.f16x2 r1578, r1575, r1454;
}
{
add.f16x2 r1581, r1572, r1578;
}
{
sub.f16x2 r1584, r1566, r1581;
}
{
add.f16x2 r1587, r1456, r1457;
}
{
mul.f16x2 r1590, r1587, r1449;
}
{
add.f16x2 r1593, r1459, r1590;
}
{
add.f16x2 r1596, r1462, r1463;
}
{
mul.f16x2 r1599, r1596, r1453;
}
{
add.f16x2 r1602, r1593, r1599;
}
{
sub.f16x2 r1605, r1468, r1469;
}
{
mul.f16x2 r1608, r1605, r1451;
}
{
sub.f16x2 r1611, r1474, r1475;
}
{
mul.f16x2 r1614, r1611, r1454;
}
{
add.f16x2 r1617, r1608, r1614;
}
{
add.f16x2 r1620, r1602, r1617;
}
{
add.f16x2 r1623, r1468, r1469;
}
{
mul.f16x2 r1626, r1623, r1445;
}
{
add.f16x2 r1629, r1471, r1626;
}
{
add.f16x2 r1632, r1474, r1475;
}
{
mul.f16x2 r1635, r1632, r1449;
}
{
add.f16x2 r1638, r1629, r1635;
}
{
sub.f16x2 r1641, r1456, r1457;
}
{
mul.f16x2 r1644, r1641, r1447;
}
{
sub.f16x2 r1647, r1462, r1463;
}
{
mul.f16x2 r1650, r1647, r1451;
}
{
add.f16x2 r1653, r1644, r1650;
}
{
add.f16x2 r1656, r1638, r1653;
}
{
add.f16x2 r1659, r1468, r1469;
}
{
mul.f16x2 r1662, r1659, r1445;
}
{
add.f16x2 r1665, r1471, r1662;
}
{
add.f16x2 r1668, r1474, r1475;
}
{
mul.f16x2 r1671, r1668, r1449;
}
{
add.f16x2 r1674, r1665, r1671;
}
{
sub.f16x2 r1677, r1456, r1457;
}
{
mul.f16x2 r1680, r1677, r1447;
}
{
sub.f16x2 r1683, r1462, r1463;
}
{
mul.f16x2 r1686, r1683, r1451;
}
{
add.f16x2 r1689, r1680, r1686;
}
{
sub.f16x2 r1692, r1674, r1689;
}
{
add.f16x2 r1695, r1468, r1469;
}
{
mul.f16x2 r1698, r1695, r1449;
}
{
add.f16x2 r1701, r1471, r1698;
}
{
add.f16x2 r1704, r1474, r1475;
}
{
mul.f16x2 r1707, r1704, r1453;
}
{
add.f16x2 r1710, r1701, r1707;
}
{
sub.f16x2 r1713, r1456, r1457;
}
{
mul.f16x2 r1716, r1713, r1451;
}
{
sub.f16x2 r1719, r1462, r1463;
}
{
mul.f16x2 r1722, r1719, r1454;
}
{
add.f16x2 r1725, r1716, r1722;
}
{
add.f16x2 r1728, r1710, r1725;
}
{
add.f16x2 r1731, r1468, r1469;
}
{
mul.f16x2 r1734, r1731, r1449;
}
{
add.f16x2 r1737, r1471, r1734;
}
{
add.f16x2 r1740, r1474, r1475;
}
{
mul.f16x2 r1743, r1740, r1453;
}
{
add.f16x2 r1746, r1737, r1743;
}
{
sub.f16x2 r1749, r1456, r1457;
}
{
mul.f16x2 r1752, r1749, r1451;
}
{
sub.f16x2 r1755, r1462, r1463;
}
{
mul.f16x2 r1758, r1755, r1454;
}
{
add.f16x2 r1761, r1752, r1758;
}
{
sub.f16x2 r1764, r1746, r1761;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1767, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r1768, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f112;
cvt.rn.f16.f32 high, f112;
mov.b32 r1769, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f118;
cvt.rn.f16.f32 high, f118;
mov.b32 r1770, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f116;
cvt.rn.f16.f32 high, f116;
mov.b32 r1771, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f118;
cvt.rn.f16.f32 high, f118;
mov.b32 r1772, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f120;
cvt.rn.f16.f32 high, f120;
mov.b32 r1773, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r1774, {low, high};
}
{
mul.f16x2 r1785, r1512, r1767;
}
{
mul.f16x2 r1788, r1656, r1768;
}
{
sub.f16x2 r1791, r1785, r1788;
}
{
mul.f16x2 r1794, r1512, r1768;
}
{
fma.rn.f16x2 r1797, r1656, r1767, r1794;
}
{
mul.f16x2 r1801, r1584, r1769;
}
{
mul.f16x2 r1804, r1728, r1770;
}
{
sub.f16x2 r1807, r1801, r1804;
}
{
mul.f16x2 r1810, r1584, r1770;
}
{
fma.rn.f16x2 r1813, r1728, r1769, r1810;
}
{
mul.f16x2 r1817, r1620, r1771;
}
{
mul.f16x2 r1820, r1764, r1772;
}
{
sub.f16x2 r1823, r1817, r1820;
}
{
mul.f16x2 r1826, r1620, r1772;
}
{
fma.rn.f16x2 r1829, r1764, r1771, r1826;
}
{
mul.f16x2 r1833, r1548, r1773;
}
{
mul.f16x2 r1836, r1692, r1774;
}
{
sub.f16x2 r1839, r1833, r1836;
}
{
mul.f16x2 r1842, r1548, r1774;
}
{
fma.rn.f16x2 r1845, r1692, r1773, r1842;
}
{
add.f16x2 %0, r1142, r1464;
}
{
add.f16x2 %1, r1154, r1476;
}
{
sub.f16x2 %10, r1142, r1464;
}
{
sub.f16x2 %11, r1154, r1476;
}
{
add.f16x2 %2, r1190, r1791;
}
{
add.f16x2 %3, r1334, r1797;
}
{
sub.f16x2 %12, r1190, r1791;
}
{
sub.f16x2 %13, r1334, r1797;
}
{
add.f16x2 %4, r1262, r1807;
}
{
add.f16x2 %5, r1406, r1813;
}
{
sub.f16x2 %14, r1262, r1807;
}
{
sub.f16x2 %15, r1406, r1813;
}
{
add.f16x2 %6, r1298, r1823;
}
{
add.f16x2 %7, r1442, r1829;
}
{
sub.f16x2 %16, r1298, r1823;
}
{
sub.f16x2 %17, r1442, r1829;
}
{
add.f16x2 %8, r1226, r1839;
}
{
add.f16x2 %9, r1370, r1845;
}
{
sub.f16x2 %18, r1226, r1839;
}
{
sub.f16x2 %19, r1370, r1845;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)));
};


#endif
