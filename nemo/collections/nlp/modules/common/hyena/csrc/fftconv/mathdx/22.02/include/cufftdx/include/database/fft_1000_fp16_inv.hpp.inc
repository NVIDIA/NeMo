#ifndef CUFFTDX_FFT_1000_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_1000_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1141, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<231>;
.reg .b32 r<3050>;
.reg .b64 rd<7>;
mov.u32 r3031, %tid.y;
shl.b32 r3032, r3031, 1;
mov.u32 r3033, %20;
mad.lo.s32 r3034, r3032, 4000, r3033;
mov.u32 r3035, %tid.x;
mov.f32 f194, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1, {low, high};
}
mov.f32 f188, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r2, {low, high};
}
{
neg.f16x2 r3, r2;
}
mov.f32 f202, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r5, {low, high};
}
mov.f32 f184, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f184;
cvt.rn.f16.f32 high, f184;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r9, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
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
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r323, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r324, {low, high};
}
{
neg.f16x2 r325, r324;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r327, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f184;
cvt.rn.f16.f32 high, f184;
mov.b32 r328, {low, high};
}
{
neg.f16x2 r329, r328;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r331, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
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
mov.f32 f190, 0f3F4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r645, {low, high};
}
mov.f32 f204, 0f3F167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r646, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r647, {low, high};
}
mov.f32 f200, 0f3F737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r648, {low, high};
}
mov.f32 f198, 0fBE9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r649, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r650, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r651, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r652, {low, high};
}
mov.f32 f161, 0fBF800000;
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
mul.wide.u32 rd2, r3035, 1374389535;
shr.u64 rd3, rd2, 37;
cvt.u32.u64 r3036, rd3;
mul.lo.s32 r3037, r3036, 100;
sub.s32 r3038, r3035, r3037;
shr.u64 rd4, rd2, 36;
cvt.u32.u64 r3039, rd4;
and.b32 r3040, r3039, 268435454;
mad.lo.s32 r3041, r3040, 4000, r3034;
cvt.rn.f32.u32 f225, r3038;
mul.f32 f226, f225, 0f3BCDE32E;
cos.approx.f32 f61, f226;
sin.approx.f32 f227, f226;
neg.f32 f62, f227;
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
mov.f32 f162, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
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
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
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
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
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
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
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
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
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
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
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
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
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
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
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
mad.lo.s32 r3042, r3038, 80, r3041;
st.shared.v2.f32 [r3042], {r727, r730};
st.shared.v2.f32 [r3042+8], {r797, r806};
st.shared.v2.f32 [r3042+16], {r834, r843};
st.shared.v2.f32 [r3042+24], {r871, r880};
st.shared.v2.f32 [r3042+32], {r908, r917};
st.shared.v2.f32 [r3042+40], {r945, r954};
st.shared.v2.f32 [r3042+48], {r982, r991};
st.shared.v2.f32 [r3042+56], {r1019, r1028};
st.shared.v2.f32 [r3042+64], {r1056, r1065};
st.shared.v2.f32 [r3042+72], {r1093, r1102};
barrier.sync 0;
mad.lo.s32 r3043, r3038, -72, r3042;
ld.shared.u32 r1137, [r3043];
ld.shared.u32 r1149, [r3043+4];
ld.shared.u32 r1459, [r3043+800];
ld.shared.u32 r1471, [r3043+804];
ld.shared.u32 r1134, [r3043+1600];
ld.shared.u32 r1146, [r3043+1604];
ld.shared.u32 r1456, [r3043+2400];
ld.shared.u32 r1468, [r3043+2404];
ld.shared.u32 r1140, [r3043+3200];
ld.shared.u32 r1152, [r3043+3204];
ld.shared.u32 r1462, [r3043+4000];
ld.shared.u32 r1474, [r3043+4004];
ld.shared.u32 r1141, [r3043+4800];
ld.shared.u32 r1153, [r3043+4804];
ld.shared.u32 r1463, [r3043+5600];
ld.shared.u32 r1475, [r3043+5604];
ld.shared.u32 r1135, [r3043+6400];
ld.shared.u32 r1147, [r3043+6404];
ld.shared.u32 r1457, [r3043+7200];
ld.shared.u32 r1469, [r3043+7204];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1123, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r1124, {low, high};
}
{
neg.f16x2 r1125, r1124;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1127, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f184;
cvt.rn.f16.f32 high, f184;
mov.b32 r1128, {low, high};
}
{
neg.f16x2 r1129, r1128;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1131, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
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
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1445, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r1446, {low, high};
}
{
neg.f16x2 r1447, r1446;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1449, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f184;
cvt.rn.f16.f32 high, f184;
mov.b32 r1450, {low, high};
}
{
neg.f16x2 r1451, r1450;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1453, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
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
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r1767, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r1768, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1769, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1770, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1771, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1772, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1773, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
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
add.f16x2 r1849, r1142, r1464;
}
{
add.f16x2 r1852, r1154, r1476;
}
{
sub.f16x2 r1855, r1142, r1464;
}
{
sub.f16x2 r1858, r1154, r1476;
}
{
add.f16x2 r1861, r1190, r1791;
}
{
add.f16x2 r1864, r1334, r1797;
}
{
sub.f16x2 r1867, r1190, r1791;
}
{
sub.f16x2 r1870, r1334, r1797;
}
{
add.f16x2 r1873, r1262, r1807;
}
{
add.f16x2 r1876, r1406, r1813;
}
{
sub.f16x2 r1879, r1262, r1807;
}
{
sub.f16x2 r1882, r1406, r1813;
}
{
add.f16x2 r1885, r1298, r1823;
}
{
add.f16x2 r1888, r1442, r1829;
}
{
sub.f16x2 r1891, r1298, r1823;
}
{
sub.f16x2 r1894, r1442, r1829;
}
{
add.f16x2 r1897, r1226, r1839;
}
{
add.f16x2 r1900, r1370, r1845;
}
{
sub.f16x2 r1903, r1226, r1839;
}
{
sub.f16x2 r1906, r1370, r1845;
}
mul.wide.u32 rd5, r3038, -858993459;
shr.u64 rd6, rd5, 35;
cvt.u32.u64 r3044, rd6;
cvt.rn.f32.u32 f228, r3044;
mul.f32 f229, f228, 0f3D80ADFD;
cos.approx.f32 f143, f229;
sin.approx.f32 f230, f229;
neg.f32 f144, f230;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f143;
cvt.rn.f16.f32 high, f144;
mov.b32 r1909, {low, high};
}
mul.lo.s32 r3045, r3044, 10;
sub.s32 r3046, r3038, r3045;
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r1912, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r1914, {high, high};
}
{
mul.f16x2 r1916, r1864, r1914;
}
{
fma.rn.f16x2 r1919, r1861, r1912, r1916;
}
{
mul.f16x2 r1923, r1861, r1914;
}
{
neg.f16x2 r1926, r1923;
}
{
fma.rn.f16x2 r1928, r1864, r1912, r1926;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r1932, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r1934, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r1936, {low, high};
}
{
mul.f16x2 r1937, r1934, r1936;
}
{
mul.f16x2 r1940, r1909, r1932;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r1943, {high, low};
}
{
fma.rn.f16x2 r1945, r1937, r1943, r1940;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1945;
mov.b32 r1949, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1945;
mov.b32 r1951, {high, high};
}
{
mul.f16x2 r1953, r1876, r1951;
}
{
fma.rn.f16x2 r1956, r1873, r1949, r1953;
}
{
mul.f16x2 r1960, r1873, r1951;
}
{
neg.f16x2 r1963, r1960;
}
{
fma.rn.f16x2 r1965, r1876, r1949, r1963;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r1969, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r1971, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r1973, {low, high};
}
{
mul.f16x2 r1974, r1971, r1973;
}
{
mul.f16x2 r1977, r1945, r1969;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1945;
mov.b32 r1980, {high, low};
}
{
fma.rn.f16x2 r1982, r1974, r1980, r1977;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1982;
mov.b32 r1986, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1982;
mov.b32 r1988, {high, high};
}
{
mul.f16x2 r1990, r1888, r1988;
}
{
fma.rn.f16x2 r1993, r1885, r1986, r1990;
}
{
mul.f16x2 r1997, r1885, r1988;
}
{
neg.f16x2 r2000, r1997;
}
{
fma.rn.f16x2 r2002, r1888, r1986, r2000;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2006, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2008, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2010, {low, high};
}
{
mul.f16x2 r2011, r2008, r2010;
}
{
mul.f16x2 r2014, r1982, r2006;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1982;
mov.b32 r2017, {high, low};
}
{
fma.rn.f16x2 r2019, r2011, r2017, r2014;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2019;
mov.b32 r2023, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2019;
mov.b32 r2025, {high, high};
}
{
mul.f16x2 r2027, r1900, r2025;
}
{
fma.rn.f16x2 r2030, r1897, r2023, r2027;
}
{
mul.f16x2 r2034, r1897, r2025;
}
{
neg.f16x2 r2037, r2034;
}
{
fma.rn.f16x2 r2039, r1900, r2023, r2037;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2043, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2045, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2047, {low, high};
}
{
mul.f16x2 r2048, r2045, r2047;
}
{
mul.f16x2 r2051, r2019, r2043;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2019;
mov.b32 r2054, {high, low};
}
{
fma.rn.f16x2 r2056, r2048, r2054, r2051;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2056;
mov.b32 r2060, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2056;
mov.b32 r2062, {high, high};
}
{
mul.f16x2 r2064, r1858, r2062;
}
{
fma.rn.f16x2 r2067, r1855, r2060, r2064;
}
{
mul.f16x2 r2071, r1855, r2062;
}
{
neg.f16x2 r2074, r2071;
}
{
fma.rn.f16x2 r2076, r1858, r2060, r2074;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2080, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2082, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2084, {low, high};
}
{
mul.f16x2 r2085, r2082, r2084;
}
{
mul.f16x2 r2088, r2056, r2080;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2056;
mov.b32 r2091, {high, low};
}
{
fma.rn.f16x2 r2093, r2085, r2091, r2088;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2093;
mov.b32 r2097, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2093;
mov.b32 r2099, {high, high};
}
{
mul.f16x2 r2101, r1870, r2099;
}
{
fma.rn.f16x2 r2104, r1867, r2097, r2101;
}
{
mul.f16x2 r2108, r1867, r2099;
}
{
neg.f16x2 r2111, r2108;
}
{
fma.rn.f16x2 r2113, r1870, r2097, r2111;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2117, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2119, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2121, {low, high};
}
{
mul.f16x2 r2122, r2119, r2121;
}
{
mul.f16x2 r2125, r2093, r2117;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2093;
mov.b32 r2128, {high, low};
}
{
fma.rn.f16x2 r2130, r2122, r2128, r2125;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2130;
mov.b32 r2134, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2130;
mov.b32 r2136, {high, high};
}
{
mul.f16x2 r2138, r1882, r2136;
}
{
fma.rn.f16x2 r2141, r1879, r2134, r2138;
}
{
mul.f16x2 r2145, r1879, r2136;
}
{
neg.f16x2 r2148, r2145;
}
{
fma.rn.f16x2 r2150, r1882, r2134, r2148;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2154, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2156, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2158, {low, high};
}
{
mul.f16x2 r2159, r2156, r2158;
}
{
mul.f16x2 r2162, r2130, r2154;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2130;
mov.b32 r2165, {high, low};
}
{
fma.rn.f16x2 r2167, r2159, r2165, r2162;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2167;
mov.b32 r2171, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2167;
mov.b32 r2173, {high, high};
}
{
mul.f16x2 r2175, r1894, r2173;
}
{
fma.rn.f16x2 r2178, r1891, r2171, r2175;
}
{
mul.f16x2 r2182, r1891, r2173;
}
{
neg.f16x2 r2185, r2182;
}
{
fma.rn.f16x2 r2187, r1894, r2171, r2185;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2191, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2193, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2195, {low, high};
}
{
mul.f16x2 r2196, r2193, r2195;
}
{
mul.f16x2 r2199, r2167, r2191;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2167;
mov.b32 r2202, {high, low};
}
{
fma.rn.f16x2 r2204, r2196, r2202, r2199;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2204;
mov.b32 r2208, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2204;
mov.b32 r2210, {high, high};
}
{
mul.f16x2 r2212, r1906, r2210;
}
{
fma.rn.f16x2 r2215, r1903, r2208, r2212;
}
{
mul.f16x2 r2219, r1903, r2210;
}
{
neg.f16x2 r2222, r2219;
}
{
fma.rn.f16x2 r2224, r1906, r2208, r2222;
}
shl.b32 r3047, r3046, 3;
add.s32 r3048, r3041, r3047;
barrier.sync 0;
mad.lo.s32 r3049, r3044, 800, r3048;
st.shared.u32 [r3049], r1849;
st.shared.u32 [r3049+4], r1852;
st.shared.u32 [r3049+80], r1919;
st.shared.u32 [r3049+84], r1928;
st.shared.u32 [r3049+160], r1956;
st.shared.u32 [r3049+164], r1965;
st.shared.u32 [r3049+240], r1993;
st.shared.u32 [r3049+244], r2002;
st.shared.u32 [r3049+320], r2030;
st.shared.u32 [r3049+324], r2039;
st.shared.u32 [r3049+400], r2067;
st.shared.u32 [r3049+404], r2076;
st.shared.u32 [r3049+480], r2104;
st.shared.u32 [r3049+484], r2113;
st.shared.u32 [r3049+560], r2141;
st.shared.u32 [r3049+564], r2150;
st.shared.u32 [r3049+640], r2178;
st.shared.u32 [r3049+644], r2187;
st.shared.u32 [r3049+720], r2215;
st.shared.u32 [r3049+724], r2224;
barrier.sync 0;
ld.shared.u32 r2259, [r3043];
ld.shared.u32 r2271, [r3043+4];
ld.shared.u32 r2581, [r3043+800];
ld.shared.u32 r2593, [r3043+804];
ld.shared.u32 r2256, [r3043+1600];
ld.shared.u32 r2268, [r3043+1604];
ld.shared.u32 r2578, [r3043+2400];
ld.shared.u32 r2590, [r3043+2404];
ld.shared.u32 r2262, [r3043+3200];
ld.shared.u32 r2274, [r3043+3204];
ld.shared.u32 r2584, [r3043+4000];
ld.shared.u32 r2596, [r3043+4004];
ld.shared.u32 r2263, [r3043+4800];
ld.shared.u32 r2275, [r3043+4804];
ld.shared.u32 r2585, [r3043+5600];
ld.shared.u32 r2597, [r3043+5604];
ld.shared.u32 r2257, [r3043+6400];
ld.shared.u32 r2269, [r3043+6404];
ld.shared.u32 r2579, [r3043+7200];
ld.shared.u32 r2591, [r3043+7204];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r2245, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r2246, {low, high};
}
{
neg.f16x2 r2247, r2246;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r2249, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f184;
cvt.rn.f16.f32 high, f184;
mov.b32 r2250, {low, high};
}
{
neg.f16x2 r2251, r2250;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r2253, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r2254, {low, high};
}
{
add.f16x2 r2255, r2256, r2257;
}
{
add.f16x2 r2258, r2259, r2255;
}
{
add.f16x2 r2261, r2262, r2263;
}
{
add.f16x2 r2264, r2258, r2261;
}
{
add.f16x2 r2267, r2268, r2269;
}
{
add.f16x2 r2270, r2271, r2267;
}
{
add.f16x2 r2273, r2274, r2275;
}
{
add.f16x2 r2276, r2270, r2273;
}
{
add.f16x2 r2279, r2256, r2257;
}
{
mul.f16x2 r2282, r2279, r2245;
}
{
add.f16x2 r2285, r2259, r2282;
}
{
add.f16x2 r2288, r2262, r2263;
}
{
mul.f16x2 r2291, r2288, r2249;
}
{
add.f16x2 r2294, r2285, r2291;
}
{
sub.f16x2 r2297, r2268, r2269;
}
{
mul.f16x2 r2300, r2297, r2247;
}
{
sub.f16x2 r2303, r2274, r2275;
}
{
mul.f16x2 r2306, r2303, r2251;
}
{
add.f16x2 r2309, r2300, r2306;
}
{
sub.f16x2 r2312, r2294, r2309;
}
{
add.f16x2 r2315, r2256, r2257;
}
{
mul.f16x2 r2318, r2315, r2245;
}
{
add.f16x2 r2321, r2259, r2318;
}
{
add.f16x2 r2324, r2262, r2263;
}
{
mul.f16x2 r2327, r2324, r2249;
}
{
add.f16x2 r2330, r2321, r2327;
}
{
sub.f16x2 r2333, r2268, r2269;
}
{
mul.f16x2 r2336, r2333, r2247;
}
{
sub.f16x2 r2339, r2274, r2275;
}
{
mul.f16x2 r2342, r2339, r2251;
}
{
add.f16x2 r2345, r2336, r2342;
}
{
add.f16x2 r2348, r2330, r2345;
}
{
add.f16x2 r2351, r2256, r2257;
}
{
mul.f16x2 r2354, r2351, r2249;
}
{
add.f16x2 r2357, r2259, r2354;
}
{
add.f16x2 r2360, r2262, r2263;
}
{
mul.f16x2 r2363, r2360, r2253;
}
{
add.f16x2 r2366, r2357, r2363;
}
{
sub.f16x2 r2369, r2268, r2269;
}
{
mul.f16x2 r2372, r2369, r2251;
}
{
sub.f16x2 r2375, r2274, r2275;
}
{
mul.f16x2 r2378, r2375, r2254;
}
{
add.f16x2 r2381, r2372, r2378;
}
{
sub.f16x2 r2384, r2366, r2381;
}
{
add.f16x2 r2387, r2256, r2257;
}
{
mul.f16x2 r2390, r2387, r2249;
}
{
add.f16x2 r2393, r2259, r2390;
}
{
add.f16x2 r2396, r2262, r2263;
}
{
mul.f16x2 r2399, r2396, r2253;
}
{
add.f16x2 r2402, r2393, r2399;
}
{
sub.f16x2 r2405, r2268, r2269;
}
{
mul.f16x2 r2408, r2405, r2251;
}
{
sub.f16x2 r2411, r2274, r2275;
}
{
mul.f16x2 r2414, r2411, r2254;
}
{
add.f16x2 r2417, r2408, r2414;
}
{
add.f16x2 r2420, r2402, r2417;
}
{
add.f16x2 r2423, r2268, r2269;
}
{
mul.f16x2 r2426, r2423, r2245;
}
{
add.f16x2 r2429, r2271, r2426;
}
{
add.f16x2 r2432, r2274, r2275;
}
{
mul.f16x2 r2435, r2432, r2249;
}
{
add.f16x2 r2438, r2429, r2435;
}
{
sub.f16x2 r2441, r2256, r2257;
}
{
mul.f16x2 r2444, r2441, r2247;
}
{
sub.f16x2 r2447, r2262, r2263;
}
{
mul.f16x2 r2450, r2447, r2251;
}
{
add.f16x2 r2453, r2444, r2450;
}
{
add.f16x2 r2456, r2438, r2453;
}
{
add.f16x2 r2459, r2268, r2269;
}
{
mul.f16x2 r2462, r2459, r2245;
}
{
add.f16x2 r2465, r2271, r2462;
}
{
add.f16x2 r2468, r2274, r2275;
}
{
mul.f16x2 r2471, r2468, r2249;
}
{
add.f16x2 r2474, r2465, r2471;
}
{
sub.f16x2 r2477, r2256, r2257;
}
{
mul.f16x2 r2480, r2477, r2247;
}
{
sub.f16x2 r2483, r2262, r2263;
}
{
mul.f16x2 r2486, r2483, r2251;
}
{
add.f16x2 r2489, r2480, r2486;
}
{
sub.f16x2 r2492, r2474, r2489;
}
{
add.f16x2 r2495, r2268, r2269;
}
{
mul.f16x2 r2498, r2495, r2249;
}
{
add.f16x2 r2501, r2271, r2498;
}
{
add.f16x2 r2504, r2274, r2275;
}
{
mul.f16x2 r2507, r2504, r2253;
}
{
add.f16x2 r2510, r2501, r2507;
}
{
sub.f16x2 r2513, r2256, r2257;
}
{
mul.f16x2 r2516, r2513, r2251;
}
{
sub.f16x2 r2519, r2262, r2263;
}
{
mul.f16x2 r2522, r2519, r2254;
}
{
add.f16x2 r2525, r2516, r2522;
}
{
add.f16x2 r2528, r2510, r2525;
}
{
add.f16x2 r2531, r2268, r2269;
}
{
mul.f16x2 r2534, r2531, r2249;
}
{
add.f16x2 r2537, r2271, r2534;
}
{
add.f16x2 r2540, r2274, r2275;
}
{
mul.f16x2 r2543, r2540, r2253;
}
{
add.f16x2 r2546, r2537, r2543;
}
{
sub.f16x2 r2549, r2256, r2257;
}
{
mul.f16x2 r2552, r2549, r2251;
}
{
sub.f16x2 r2555, r2262, r2263;
}
{
mul.f16x2 r2558, r2555, r2254;
}
{
add.f16x2 r2561, r2552, r2558;
}
{
sub.f16x2 r2564, r2546, r2561;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r2567, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r2568, {low, high};
}
{
neg.f16x2 r2569, r2568;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r2571, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f184;
cvt.rn.f16.f32 high, f184;
mov.b32 r2572, {low, high};
}
{
neg.f16x2 r2573, r2572;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r2575, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r2576, {low, high};
}
{
add.f16x2 r2577, r2578, r2579;
}
{
add.f16x2 r2580, r2581, r2577;
}
{
add.f16x2 r2583, r2584, r2585;
}
{
add.f16x2 r2586, r2580, r2583;
}
{
add.f16x2 r2589, r2590, r2591;
}
{
add.f16x2 r2592, r2593, r2589;
}
{
add.f16x2 r2595, r2596, r2597;
}
{
add.f16x2 r2598, r2592, r2595;
}
{
add.f16x2 r2601, r2578, r2579;
}
{
mul.f16x2 r2604, r2601, r2567;
}
{
add.f16x2 r2607, r2581, r2604;
}
{
add.f16x2 r2610, r2584, r2585;
}
{
mul.f16x2 r2613, r2610, r2571;
}
{
add.f16x2 r2616, r2607, r2613;
}
{
sub.f16x2 r2619, r2590, r2591;
}
{
mul.f16x2 r2622, r2619, r2569;
}
{
sub.f16x2 r2625, r2596, r2597;
}
{
mul.f16x2 r2628, r2625, r2573;
}
{
add.f16x2 r2631, r2622, r2628;
}
{
sub.f16x2 r2634, r2616, r2631;
}
{
add.f16x2 r2637, r2578, r2579;
}
{
mul.f16x2 r2640, r2637, r2567;
}
{
add.f16x2 r2643, r2581, r2640;
}
{
add.f16x2 r2646, r2584, r2585;
}
{
mul.f16x2 r2649, r2646, r2571;
}
{
add.f16x2 r2652, r2643, r2649;
}
{
sub.f16x2 r2655, r2590, r2591;
}
{
mul.f16x2 r2658, r2655, r2569;
}
{
sub.f16x2 r2661, r2596, r2597;
}
{
mul.f16x2 r2664, r2661, r2573;
}
{
add.f16x2 r2667, r2658, r2664;
}
{
add.f16x2 r2670, r2652, r2667;
}
{
add.f16x2 r2673, r2578, r2579;
}
{
mul.f16x2 r2676, r2673, r2571;
}
{
add.f16x2 r2679, r2581, r2676;
}
{
add.f16x2 r2682, r2584, r2585;
}
{
mul.f16x2 r2685, r2682, r2575;
}
{
add.f16x2 r2688, r2679, r2685;
}
{
sub.f16x2 r2691, r2590, r2591;
}
{
mul.f16x2 r2694, r2691, r2573;
}
{
sub.f16x2 r2697, r2596, r2597;
}
{
mul.f16x2 r2700, r2697, r2576;
}
{
add.f16x2 r2703, r2694, r2700;
}
{
sub.f16x2 r2706, r2688, r2703;
}
{
add.f16x2 r2709, r2578, r2579;
}
{
mul.f16x2 r2712, r2709, r2571;
}
{
add.f16x2 r2715, r2581, r2712;
}
{
add.f16x2 r2718, r2584, r2585;
}
{
mul.f16x2 r2721, r2718, r2575;
}
{
add.f16x2 r2724, r2715, r2721;
}
{
sub.f16x2 r2727, r2590, r2591;
}
{
mul.f16x2 r2730, r2727, r2573;
}
{
sub.f16x2 r2733, r2596, r2597;
}
{
mul.f16x2 r2736, r2733, r2576;
}
{
add.f16x2 r2739, r2730, r2736;
}
{
add.f16x2 r2742, r2724, r2739;
}
{
add.f16x2 r2745, r2590, r2591;
}
{
mul.f16x2 r2748, r2745, r2567;
}
{
add.f16x2 r2751, r2593, r2748;
}
{
add.f16x2 r2754, r2596, r2597;
}
{
mul.f16x2 r2757, r2754, r2571;
}
{
add.f16x2 r2760, r2751, r2757;
}
{
sub.f16x2 r2763, r2578, r2579;
}
{
mul.f16x2 r2766, r2763, r2569;
}
{
sub.f16x2 r2769, r2584, r2585;
}
{
mul.f16x2 r2772, r2769, r2573;
}
{
add.f16x2 r2775, r2766, r2772;
}
{
add.f16x2 r2778, r2760, r2775;
}
{
add.f16x2 r2781, r2590, r2591;
}
{
mul.f16x2 r2784, r2781, r2567;
}
{
add.f16x2 r2787, r2593, r2784;
}
{
add.f16x2 r2790, r2596, r2597;
}
{
mul.f16x2 r2793, r2790, r2571;
}
{
add.f16x2 r2796, r2787, r2793;
}
{
sub.f16x2 r2799, r2578, r2579;
}
{
mul.f16x2 r2802, r2799, r2569;
}
{
sub.f16x2 r2805, r2584, r2585;
}
{
mul.f16x2 r2808, r2805, r2573;
}
{
add.f16x2 r2811, r2802, r2808;
}
{
sub.f16x2 r2814, r2796, r2811;
}
{
add.f16x2 r2817, r2590, r2591;
}
{
mul.f16x2 r2820, r2817, r2571;
}
{
add.f16x2 r2823, r2593, r2820;
}
{
add.f16x2 r2826, r2596, r2597;
}
{
mul.f16x2 r2829, r2826, r2575;
}
{
add.f16x2 r2832, r2823, r2829;
}
{
sub.f16x2 r2835, r2578, r2579;
}
{
mul.f16x2 r2838, r2835, r2573;
}
{
sub.f16x2 r2841, r2584, r2585;
}
{
mul.f16x2 r2844, r2841, r2576;
}
{
add.f16x2 r2847, r2838, r2844;
}
{
add.f16x2 r2850, r2832, r2847;
}
{
add.f16x2 r2853, r2590, r2591;
}
{
mul.f16x2 r2856, r2853, r2571;
}
{
add.f16x2 r2859, r2593, r2856;
}
{
add.f16x2 r2862, r2596, r2597;
}
{
mul.f16x2 r2865, r2862, r2575;
}
{
add.f16x2 r2868, r2859, r2865;
}
{
sub.f16x2 r2871, r2578, r2579;
}
{
mul.f16x2 r2874, r2871, r2573;
}
{
sub.f16x2 r2877, r2584, r2585;
}
{
mul.f16x2 r2880, r2877, r2576;
}
{
add.f16x2 r2883, r2874, r2880;
}
{
sub.f16x2 r2886, r2868, r2883;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r2889, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r2890, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r2891, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2892, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r2893, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2894, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r2895, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r2896, {low, high};
}
{
mul.f16x2 r2907, r2634, r2889;
}
{
mul.f16x2 r2910, r2778, r2890;
}
{
sub.f16x2 r2913, r2907, r2910;
}
{
mul.f16x2 r2916, r2634, r2890;
}
{
fma.rn.f16x2 r2919, r2778, r2889, r2916;
}
{
mul.f16x2 r2923, r2706, r2891;
}
{
mul.f16x2 r2926, r2850, r2892;
}
{
sub.f16x2 r2929, r2923, r2926;
}
{
mul.f16x2 r2932, r2706, r2892;
}
{
fma.rn.f16x2 r2935, r2850, r2891, r2932;
}
{
mul.f16x2 r2939, r2742, r2893;
}
{
mul.f16x2 r2942, r2886, r2894;
}
{
sub.f16x2 r2945, r2939, r2942;
}
{
mul.f16x2 r2948, r2742, r2894;
}
{
fma.rn.f16x2 r2951, r2886, r2893, r2948;
}
{
mul.f16x2 r2955, r2670, r2895;
}
{
mul.f16x2 r2958, r2814, r2896;
}
{
sub.f16x2 r2961, r2955, r2958;
}
{
mul.f16x2 r2964, r2670, r2896;
}
{
fma.rn.f16x2 r2967, r2814, r2895, r2964;
}
{
add.f16x2 %0, r2264, r2586;
}
{
add.f16x2 %1, r2276, r2598;
}
{
sub.f16x2 %10, r2264, r2586;
}
{
sub.f16x2 %11, r2276, r2598;
}
{
add.f16x2 %2, r2312, r2913;
}
{
add.f16x2 %3, r2456, r2919;
}
{
sub.f16x2 %12, r2312, r2913;
}
{
sub.f16x2 %13, r2456, r2919;
}
{
add.f16x2 %4, r2384, r2929;
}
{
add.f16x2 %5, r2528, r2935;
}
{
sub.f16x2 %14, r2384, r2929;
}
{
sub.f16x2 %15, r2528, r2935;
}
{
add.f16x2 %6, r2420, r2945;
}
{
add.f16x2 %7, r2564, r2951;
}
{
sub.f16x2 %16, r2420, r2945;
}
{
sub.f16x2 %17, r2564, r2951;
}
{
add.f16x2 %8, r2348, r2961;
}
{
add.f16x2 %9, r2492, r2967;
}
{
sub.f16x2 %18, r2348, r2961;
}
{
sub.f16x2 %19, r2492, r2967;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1142, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<231>;
.reg .b32 r<3047>;
.reg .b64 rd<6>;
mov.u32 r3031, %tid.y;
mov.u32 r3032, %20;
mad.lo.s32 r3033, r3031, 4000, r3032;
mov.u32 r3034, %tid.x;
mov.f32 f194, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1, {low, high};
}
mov.f32 f188, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r2, {low, high};
}
{
neg.f16x2 r3, r2;
}
mov.f32 f202, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r5, {low, high};
}
mov.f32 f184, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f184;
cvt.rn.f16.f32 high, f184;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r9, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
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
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r323, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r324, {low, high};
}
{
neg.f16x2 r325, r324;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r327, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f184;
cvt.rn.f16.f32 high, f184;
mov.b32 r328, {low, high};
}
{
neg.f16x2 r329, r328;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r331, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
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
mov.f32 f190, 0f3F4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r645, {low, high};
}
mov.f32 f204, 0f3F167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r646, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r647, {low, high};
}
mov.f32 f200, 0f3F737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r648, {low, high};
}
mov.f32 f198, 0fBE9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r649, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r650, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r651, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r652, {low, high};
}
mov.f32 f161, 0fBF800000;
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
mul.wide.u32 rd2, r3034, 1374389535;
shr.u64 rd3, rd2, 37;
cvt.u32.u64 r3035, rd3;
mul.lo.s32 r3036, r3035, 100;
sub.s32 r3037, r3034, r3036;
mad.lo.s32 r3038, r3035, 4000, r3033;
cvt.rn.f32.u32 f225, r3037;
mul.f32 f226, f225, 0f3BCDE32E;
cos.approx.f32 f61, f226;
sin.approx.f32 f227, f226;
neg.f32 f62, f227;
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
mov.f32 f162, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
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
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
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
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
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
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
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
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
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
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
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
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
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
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
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
mad.lo.s32 r3039, r3037, 40, r3038;
st.shared.v2.f32 [r3039], {r727, r797};
st.shared.v2.f32 [r3039+8], {r834, r871};
st.shared.v2.f32 [r3039+16], {r908, r945};
st.shared.v2.f32 [r3039+24], {r982, r1019};
st.shared.v2.f32 [r3039+32], {r1056, r1093};
barrier.sync 0;
mad.lo.s32 r3040, r3037, -36, r3039;
ld.shared.u32 r1137, [r3040];
ld.shared.u32 r1459, [r3040+400];
ld.shared.u32 r1134, [r3040+800];
ld.shared.u32 r1456, [r3040+1200];
ld.shared.u32 r1140, [r3040+1600];
ld.shared.u32 r1462, [r3040+2000];
ld.shared.u32 r1141, [r3040+2400];
ld.shared.u32 r1463, [r3040+2800];
ld.shared.u32 r1135, [r3040+3200];
ld.shared.u32 r1457, [r3040+3600];
barrier.sync 0;
st.shared.v2.f32 [r3039], {r730, r806};
st.shared.v2.f32 [r3039+8], {r843, r880};
st.shared.v2.f32 [r3039+16], {r917, r954};
st.shared.v2.f32 [r3039+24], {r991, r1028};
st.shared.v2.f32 [r3039+32], {r1065, r1102};
barrier.sync 0;
ld.shared.u32 r1149, [r3040];
ld.shared.u32 r1471, [r3040+400];
ld.shared.u32 r1146, [r3040+800];
ld.shared.u32 r1468, [r3040+1200];
ld.shared.u32 r1152, [r3040+1600];
ld.shared.u32 r1474, [r3040+2000];
ld.shared.u32 r1153, [r3040+2400];
ld.shared.u32 r1475, [r3040+2800];
ld.shared.u32 r1147, [r3040+3200];
ld.shared.u32 r1469, [r3040+3600];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1123, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r1124, {low, high};
}
{
neg.f16x2 r1125, r1124;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1127, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f184;
cvt.rn.f16.f32 high, f184;
mov.b32 r1128, {low, high};
}
{
neg.f16x2 r1129, r1128;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1131, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
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
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1445, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r1446, {low, high};
}
{
neg.f16x2 r1447, r1446;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1449, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f184;
cvt.rn.f16.f32 high, f184;
mov.b32 r1450, {low, high};
}
{
neg.f16x2 r1451, r1450;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1453, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
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
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r1767, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r1768, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1769, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1770, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1771, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1772, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1773, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
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
add.f16x2 r1849, r1142, r1464;
}
{
add.f16x2 r1852, r1154, r1476;
}
{
sub.f16x2 r1855, r1142, r1464;
}
{
sub.f16x2 r1858, r1154, r1476;
}
{
add.f16x2 r1861, r1190, r1791;
}
{
add.f16x2 r1864, r1334, r1797;
}
{
sub.f16x2 r1867, r1190, r1791;
}
{
sub.f16x2 r1870, r1334, r1797;
}
{
add.f16x2 r1873, r1262, r1807;
}
{
add.f16x2 r1876, r1406, r1813;
}
{
sub.f16x2 r1879, r1262, r1807;
}
{
sub.f16x2 r1882, r1406, r1813;
}
{
add.f16x2 r1885, r1298, r1823;
}
{
add.f16x2 r1888, r1442, r1829;
}
{
sub.f16x2 r1891, r1298, r1823;
}
{
sub.f16x2 r1894, r1442, r1829;
}
{
add.f16x2 r1897, r1226, r1839;
}
{
add.f16x2 r1900, r1370, r1845;
}
{
sub.f16x2 r1903, r1226, r1839;
}
{
sub.f16x2 r1906, r1370, r1845;
}
mul.wide.u32 rd4, r3037, -858993459;
shr.u64 rd5, rd4, 35;
cvt.u32.u64 r3041, rd5;
mul.lo.s32 r3042, r3041, 10;
sub.s32 r3043, r3037, r3042;
shl.b32 r3044, r3043, 2;
add.s32 r3045, r3038, r3044;
cvt.rn.f32.u32 f228, r3041;
mul.f32 f229, f228, 0f3D80ADFD;
cos.approx.f32 f143, f229;
sin.approx.f32 f230, f229;
neg.f32 f144, f230;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f143;
cvt.rn.f16.f32 high, f144;
mov.b32 r1909, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r1912, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r1914, {high, high};
}
{
mul.f16x2 r1916, r1864, r1914;
}
{
fma.rn.f16x2 r1919, r1861, r1912, r1916;
}
{
mul.f16x2 r1923, r1861, r1914;
}
{
neg.f16x2 r1926, r1923;
}
{
fma.rn.f16x2 r1928, r1864, r1912, r1926;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r1932, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r1934, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r1936, {low, high};
}
{
mul.f16x2 r1937, r1934, r1936;
}
{
mul.f16x2 r1940, r1909, r1932;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r1943, {high, low};
}
{
fma.rn.f16x2 r1945, r1937, r1943, r1940;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1945;
mov.b32 r1949, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1945;
mov.b32 r1951, {high, high};
}
{
mul.f16x2 r1953, r1876, r1951;
}
{
fma.rn.f16x2 r1956, r1873, r1949, r1953;
}
{
mul.f16x2 r1960, r1873, r1951;
}
{
neg.f16x2 r1963, r1960;
}
{
fma.rn.f16x2 r1965, r1876, r1949, r1963;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r1969, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r1971, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r1973, {low, high};
}
{
mul.f16x2 r1974, r1971, r1973;
}
{
mul.f16x2 r1977, r1945, r1969;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1945;
mov.b32 r1980, {high, low};
}
{
fma.rn.f16x2 r1982, r1974, r1980, r1977;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1982;
mov.b32 r1986, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1982;
mov.b32 r1988, {high, high};
}
{
mul.f16x2 r1990, r1888, r1988;
}
{
fma.rn.f16x2 r1993, r1885, r1986, r1990;
}
{
mul.f16x2 r1997, r1885, r1988;
}
{
neg.f16x2 r2000, r1997;
}
{
fma.rn.f16x2 r2002, r1888, r1986, r2000;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2006, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2008, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2010, {low, high};
}
{
mul.f16x2 r2011, r2008, r2010;
}
{
mul.f16x2 r2014, r1982, r2006;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1982;
mov.b32 r2017, {high, low};
}
{
fma.rn.f16x2 r2019, r2011, r2017, r2014;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2019;
mov.b32 r2023, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2019;
mov.b32 r2025, {high, high};
}
{
mul.f16x2 r2027, r1900, r2025;
}
{
fma.rn.f16x2 r2030, r1897, r2023, r2027;
}
{
mul.f16x2 r2034, r1897, r2025;
}
{
neg.f16x2 r2037, r2034;
}
{
fma.rn.f16x2 r2039, r1900, r2023, r2037;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2043, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2045, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2047, {low, high};
}
{
mul.f16x2 r2048, r2045, r2047;
}
{
mul.f16x2 r2051, r2019, r2043;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2019;
mov.b32 r2054, {high, low};
}
{
fma.rn.f16x2 r2056, r2048, r2054, r2051;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2056;
mov.b32 r2060, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2056;
mov.b32 r2062, {high, high};
}
{
mul.f16x2 r2064, r1858, r2062;
}
{
fma.rn.f16x2 r2067, r1855, r2060, r2064;
}
{
mul.f16x2 r2071, r1855, r2062;
}
{
neg.f16x2 r2074, r2071;
}
{
fma.rn.f16x2 r2076, r1858, r2060, r2074;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2080, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2082, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2084, {low, high};
}
{
mul.f16x2 r2085, r2082, r2084;
}
{
mul.f16x2 r2088, r2056, r2080;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2056;
mov.b32 r2091, {high, low};
}
{
fma.rn.f16x2 r2093, r2085, r2091, r2088;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2093;
mov.b32 r2097, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2093;
mov.b32 r2099, {high, high};
}
{
mul.f16x2 r2101, r1870, r2099;
}
{
fma.rn.f16x2 r2104, r1867, r2097, r2101;
}
{
mul.f16x2 r2108, r1867, r2099;
}
{
neg.f16x2 r2111, r2108;
}
{
fma.rn.f16x2 r2113, r1870, r2097, r2111;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2117, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2119, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2121, {low, high};
}
{
mul.f16x2 r2122, r2119, r2121;
}
{
mul.f16x2 r2125, r2093, r2117;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2093;
mov.b32 r2128, {high, low};
}
{
fma.rn.f16x2 r2130, r2122, r2128, r2125;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2130;
mov.b32 r2134, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2130;
mov.b32 r2136, {high, high};
}
{
mul.f16x2 r2138, r1882, r2136;
}
{
fma.rn.f16x2 r2141, r1879, r2134, r2138;
}
{
mul.f16x2 r2145, r1879, r2136;
}
{
neg.f16x2 r2148, r2145;
}
{
fma.rn.f16x2 r2150, r1882, r2134, r2148;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2154, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2156, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2158, {low, high};
}
{
mul.f16x2 r2159, r2156, r2158;
}
{
mul.f16x2 r2162, r2130, r2154;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2130;
mov.b32 r2165, {high, low};
}
{
fma.rn.f16x2 r2167, r2159, r2165, r2162;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2167;
mov.b32 r2171, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2167;
mov.b32 r2173, {high, high};
}
{
mul.f16x2 r2175, r1894, r2173;
}
{
fma.rn.f16x2 r2178, r1891, r2171, r2175;
}
{
mul.f16x2 r2182, r1891, r2173;
}
{
neg.f16x2 r2185, r2182;
}
{
fma.rn.f16x2 r2187, r1894, r2171, r2185;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2191, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1909;
mov.b32 r2193, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2195, {low, high};
}
{
mul.f16x2 r2196, r2193, r2195;
}
{
mul.f16x2 r2199, r2167, r2191;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2167;
mov.b32 r2202, {high, low};
}
{
fma.rn.f16x2 r2204, r2196, r2202, r2199;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2204;
mov.b32 r2208, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2204;
mov.b32 r2210, {high, high};
}
{
mul.f16x2 r2212, r1906, r2210;
}
{
fma.rn.f16x2 r2215, r1903, r2208, r2212;
}
{
mul.f16x2 r2219, r1903, r2210;
}
{
neg.f16x2 r2222, r2219;
}
{
fma.rn.f16x2 r2224, r1906, r2208, r2222;
}
barrier.sync 0;
mad.lo.s32 r3046, r3041, 400, r3045;
st.shared.u32 [r3046], r1849;
st.shared.u32 [r3046+40], r1919;
st.shared.u32 [r3046+80], r1956;
st.shared.u32 [r3046+120], r1993;
st.shared.u32 [r3046+160], r2030;
st.shared.u32 [r3046+200], r2067;
st.shared.u32 [r3046+240], r2104;
st.shared.u32 [r3046+280], r2141;
st.shared.u32 [r3046+320], r2178;
st.shared.u32 [r3046+360], r2215;
barrier.sync 0;
ld.shared.u32 r2259, [r3040];
ld.shared.u32 r2581, [r3040+400];
ld.shared.u32 r2256, [r3040+800];
ld.shared.u32 r2578, [r3040+1200];
ld.shared.u32 r2262, [r3040+1600];
ld.shared.u32 r2584, [r3040+2000];
ld.shared.u32 r2263, [r3040+2400];
ld.shared.u32 r2585, [r3040+2800];
ld.shared.u32 r2257, [r3040+3200];
ld.shared.u32 r2579, [r3040+3600];
barrier.sync 0;
st.shared.u32 [r3046], r1852;
st.shared.u32 [r3046+40], r1928;
st.shared.u32 [r3046+80], r1965;
st.shared.u32 [r3046+120], r2002;
st.shared.u32 [r3046+160], r2039;
st.shared.u32 [r3046+200], r2076;
st.shared.u32 [r3046+240], r2113;
st.shared.u32 [r3046+280], r2150;
st.shared.u32 [r3046+320], r2187;
st.shared.u32 [r3046+360], r2224;
barrier.sync 0;
ld.shared.u32 r2271, [r3040];
ld.shared.u32 r2593, [r3040+400];
ld.shared.u32 r2268, [r3040+800];
ld.shared.u32 r2590, [r3040+1200];
ld.shared.u32 r2274, [r3040+1600];
ld.shared.u32 r2596, [r3040+2000];
ld.shared.u32 r2275, [r3040+2400];
ld.shared.u32 r2597, [r3040+2800];
ld.shared.u32 r2269, [r3040+3200];
ld.shared.u32 r2591, [r3040+3600];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r2245, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r2246, {low, high};
}
{
neg.f16x2 r2247, r2246;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r2249, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f184;
cvt.rn.f16.f32 high, f184;
mov.b32 r2250, {low, high};
}
{
neg.f16x2 r2251, r2250;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r2253, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r2254, {low, high};
}
{
add.f16x2 r2255, r2256, r2257;
}
{
add.f16x2 r2258, r2259, r2255;
}
{
add.f16x2 r2261, r2262, r2263;
}
{
add.f16x2 r2264, r2258, r2261;
}
{
add.f16x2 r2267, r2268, r2269;
}
{
add.f16x2 r2270, r2271, r2267;
}
{
add.f16x2 r2273, r2274, r2275;
}
{
add.f16x2 r2276, r2270, r2273;
}
{
add.f16x2 r2279, r2256, r2257;
}
{
mul.f16x2 r2282, r2279, r2245;
}
{
add.f16x2 r2285, r2259, r2282;
}
{
add.f16x2 r2288, r2262, r2263;
}
{
mul.f16x2 r2291, r2288, r2249;
}
{
add.f16x2 r2294, r2285, r2291;
}
{
sub.f16x2 r2297, r2268, r2269;
}
{
mul.f16x2 r2300, r2297, r2247;
}
{
sub.f16x2 r2303, r2274, r2275;
}
{
mul.f16x2 r2306, r2303, r2251;
}
{
add.f16x2 r2309, r2300, r2306;
}
{
sub.f16x2 r2312, r2294, r2309;
}
{
add.f16x2 r2315, r2256, r2257;
}
{
mul.f16x2 r2318, r2315, r2245;
}
{
add.f16x2 r2321, r2259, r2318;
}
{
add.f16x2 r2324, r2262, r2263;
}
{
mul.f16x2 r2327, r2324, r2249;
}
{
add.f16x2 r2330, r2321, r2327;
}
{
sub.f16x2 r2333, r2268, r2269;
}
{
mul.f16x2 r2336, r2333, r2247;
}
{
sub.f16x2 r2339, r2274, r2275;
}
{
mul.f16x2 r2342, r2339, r2251;
}
{
add.f16x2 r2345, r2336, r2342;
}
{
add.f16x2 r2348, r2330, r2345;
}
{
add.f16x2 r2351, r2256, r2257;
}
{
mul.f16x2 r2354, r2351, r2249;
}
{
add.f16x2 r2357, r2259, r2354;
}
{
add.f16x2 r2360, r2262, r2263;
}
{
mul.f16x2 r2363, r2360, r2253;
}
{
add.f16x2 r2366, r2357, r2363;
}
{
sub.f16x2 r2369, r2268, r2269;
}
{
mul.f16x2 r2372, r2369, r2251;
}
{
sub.f16x2 r2375, r2274, r2275;
}
{
mul.f16x2 r2378, r2375, r2254;
}
{
add.f16x2 r2381, r2372, r2378;
}
{
sub.f16x2 r2384, r2366, r2381;
}
{
add.f16x2 r2387, r2256, r2257;
}
{
mul.f16x2 r2390, r2387, r2249;
}
{
add.f16x2 r2393, r2259, r2390;
}
{
add.f16x2 r2396, r2262, r2263;
}
{
mul.f16x2 r2399, r2396, r2253;
}
{
add.f16x2 r2402, r2393, r2399;
}
{
sub.f16x2 r2405, r2268, r2269;
}
{
mul.f16x2 r2408, r2405, r2251;
}
{
sub.f16x2 r2411, r2274, r2275;
}
{
mul.f16x2 r2414, r2411, r2254;
}
{
add.f16x2 r2417, r2408, r2414;
}
{
add.f16x2 r2420, r2402, r2417;
}
{
add.f16x2 r2423, r2268, r2269;
}
{
mul.f16x2 r2426, r2423, r2245;
}
{
add.f16x2 r2429, r2271, r2426;
}
{
add.f16x2 r2432, r2274, r2275;
}
{
mul.f16x2 r2435, r2432, r2249;
}
{
add.f16x2 r2438, r2429, r2435;
}
{
sub.f16x2 r2441, r2256, r2257;
}
{
mul.f16x2 r2444, r2441, r2247;
}
{
sub.f16x2 r2447, r2262, r2263;
}
{
mul.f16x2 r2450, r2447, r2251;
}
{
add.f16x2 r2453, r2444, r2450;
}
{
add.f16x2 r2456, r2438, r2453;
}
{
add.f16x2 r2459, r2268, r2269;
}
{
mul.f16x2 r2462, r2459, r2245;
}
{
add.f16x2 r2465, r2271, r2462;
}
{
add.f16x2 r2468, r2274, r2275;
}
{
mul.f16x2 r2471, r2468, r2249;
}
{
add.f16x2 r2474, r2465, r2471;
}
{
sub.f16x2 r2477, r2256, r2257;
}
{
mul.f16x2 r2480, r2477, r2247;
}
{
sub.f16x2 r2483, r2262, r2263;
}
{
mul.f16x2 r2486, r2483, r2251;
}
{
add.f16x2 r2489, r2480, r2486;
}
{
sub.f16x2 r2492, r2474, r2489;
}
{
add.f16x2 r2495, r2268, r2269;
}
{
mul.f16x2 r2498, r2495, r2249;
}
{
add.f16x2 r2501, r2271, r2498;
}
{
add.f16x2 r2504, r2274, r2275;
}
{
mul.f16x2 r2507, r2504, r2253;
}
{
add.f16x2 r2510, r2501, r2507;
}
{
sub.f16x2 r2513, r2256, r2257;
}
{
mul.f16x2 r2516, r2513, r2251;
}
{
sub.f16x2 r2519, r2262, r2263;
}
{
mul.f16x2 r2522, r2519, r2254;
}
{
add.f16x2 r2525, r2516, r2522;
}
{
add.f16x2 r2528, r2510, r2525;
}
{
add.f16x2 r2531, r2268, r2269;
}
{
mul.f16x2 r2534, r2531, r2249;
}
{
add.f16x2 r2537, r2271, r2534;
}
{
add.f16x2 r2540, r2274, r2275;
}
{
mul.f16x2 r2543, r2540, r2253;
}
{
add.f16x2 r2546, r2537, r2543;
}
{
sub.f16x2 r2549, r2256, r2257;
}
{
mul.f16x2 r2552, r2549, r2251;
}
{
sub.f16x2 r2555, r2262, r2263;
}
{
mul.f16x2 r2558, r2555, r2254;
}
{
add.f16x2 r2561, r2552, r2558;
}
{
sub.f16x2 r2564, r2546, r2561;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r2567, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r2568, {low, high};
}
{
neg.f16x2 r2569, r2568;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r2571, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f184;
cvt.rn.f16.f32 high, f184;
mov.b32 r2572, {low, high};
}
{
neg.f16x2 r2573, r2572;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r2575, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r2576, {low, high};
}
{
add.f16x2 r2577, r2578, r2579;
}
{
add.f16x2 r2580, r2581, r2577;
}
{
add.f16x2 r2583, r2584, r2585;
}
{
add.f16x2 r2586, r2580, r2583;
}
{
add.f16x2 r2589, r2590, r2591;
}
{
add.f16x2 r2592, r2593, r2589;
}
{
add.f16x2 r2595, r2596, r2597;
}
{
add.f16x2 r2598, r2592, r2595;
}
{
add.f16x2 r2601, r2578, r2579;
}
{
mul.f16x2 r2604, r2601, r2567;
}
{
add.f16x2 r2607, r2581, r2604;
}
{
add.f16x2 r2610, r2584, r2585;
}
{
mul.f16x2 r2613, r2610, r2571;
}
{
add.f16x2 r2616, r2607, r2613;
}
{
sub.f16x2 r2619, r2590, r2591;
}
{
mul.f16x2 r2622, r2619, r2569;
}
{
sub.f16x2 r2625, r2596, r2597;
}
{
mul.f16x2 r2628, r2625, r2573;
}
{
add.f16x2 r2631, r2622, r2628;
}
{
sub.f16x2 r2634, r2616, r2631;
}
{
add.f16x2 r2637, r2578, r2579;
}
{
mul.f16x2 r2640, r2637, r2567;
}
{
add.f16x2 r2643, r2581, r2640;
}
{
add.f16x2 r2646, r2584, r2585;
}
{
mul.f16x2 r2649, r2646, r2571;
}
{
add.f16x2 r2652, r2643, r2649;
}
{
sub.f16x2 r2655, r2590, r2591;
}
{
mul.f16x2 r2658, r2655, r2569;
}
{
sub.f16x2 r2661, r2596, r2597;
}
{
mul.f16x2 r2664, r2661, r2573;
}
{
add.f16x2 r2667, r2658, r2664;
}
{
add.f16x2 r2670, r2652, r2667;
}
{
add.f16x2 r2673, r2578, r2579;
}
{
mul.f16x2 r2676, r2673, r2571;
}
{
add.f16x2 r2679, r2581, r2676;
}
{
add.f16x2 r2682, r2584, r2585;
}
{
mul.f16x2 r2685, r2682, r2575;
}
{
add.f16x2 r2688, r2679, r2685;
}
{
sub.f16x2 r2691, r2590, r2591;
}
{
mul.f16x2 r2694, r2691, r2573;
}
{
sub.f16x2 r2697, r2596, r2597;
}
{
mul.f16x2 r2700, r2697, r2576;
}
{
add.f16x2 r2703, r2694, r2700;
}
{
sub.f16x2 r2706, r2688, r2703;
}
{
add.f16x2 r2709, r2578, r2579;
}
{
mul.f16x2 r2712, r2709, r2571;
}
{
add.f16x2 r2715, r2581, r2712;
}
{
add.f16x2 r2718, r2584, r2585;
}
{
mul.f16x2 r2721, r2718, r2575;
}
{
add.f16x2 r2724, r2715, r2721;
}
{
sub.f16x2 r2727, r2590, r2591;
}
{
mul.f16x2 r2730, r2727, r2573;
}
{
sub.f16x2 r2733, r2596, r2597;
}
{
mul.f16x2 r2736, r2733, r2576;
}
{
add.f16x2 r2739, r2730, r2736;
}
{
add.f16x2 r2742, r2724, r2739;
}
{
add.f16x2 r2745, r2590, r2591;
}
{
mul.f16x2 r2748, r2745, r2567;
}
{
add.f16x2 r2751, r2593, r2748;
}
{
add.f16x2 r2754, r2596, r2597;
}
{
mul.f16x2 r2757, r2754, r2571;
}
{
add.f16x2 r2760, r2751, r2757;
}
{
sub.f16x2 r2763, r2578, r2579;
}
{
mul.f16x2 r2766, r2763, r2569;
}
{
sub.f16x2 r2769, r2584, r2585;
}
{
mul.f16x2 r2772, r2769, r2573;
}
{
add.f16x2 r2775, r2766, r2772;
}
{
add.f16x2 r2778, r2760, r2775;
}
{
add.f16x2 r2781, r2590, r2591;
}
{
mul.f16x2 r2784, r2781, r2567;
}
{
add.f16x2 r2787, r2593, r2784;
}
{
add.f16x2 r2790, r2596, r2597;
}
{
mul.f16x2 r2793, r2790, r2571;
}
{
add.f16x2 r2796, r2787, r2793;
}
{
sub.f16x2 r2799, r2578, r2579;
}
{
mul.f16x2 r2802, r2799, r2569;
}
{
sub.f16x2 r2805, r2584, r2585;
}
{
mul.f16x2 r2808, r2805, r2573;
}
{
add.f16x2 r2811, r2802, r2808;
}
{
sub.f16x2 r2814, r2796, r2811;
}
{
add.f16x2 r2817, r2590, r2591;
}
{
mul.f16x2 r2820, r2817, r2571;
}
{
add.f16x2 r2823, r2593, r2820;
}
{
add.f16x2 r2826, r2596, r2597;
}
{
mul.f16x2 r2829, r2826, r2575;
}
{
add.f16x2 r2832, r2823, r2829;
}
{
sub.f16x2 r2835, r2578, r2579;
}
{
mul.f16x2 r2838, r2835, r2573;
}
{
sub.f16x2 r2841, r2584, r2585;
}
{
mul.f16x2 r2844, r2841, r2576;
}
{
add.f16x2 r2847, r2838, r2844;
}
{
add.f16x2 r2850, r2832, r2847;
}
{
add.f16x2 r2853, r2590, r2591;
}
{
mul.f16x2 r2856, r2853, r2571;
}
{
add.f16x2 r2859, r2593, r2856;
}
{
add.f16x2 r2862, r2596, r2597;
}
{
mul.f16x2 r2865, r2862, r2575;
}
{
add.f16x2 r2868, r2859, r2865;
}
{
sub.f16x2 r2871, r2578, r2579;
}
{
mul.f16x2 r2874, r2871, r2573;
}
{
sub.f16x2 r2877, r2584, r2585;
}
{
mul.f16x2 r2880, r2877, r2576;
}
{
add.f16x2 r2883, r2874, r2880;
}
{
sub.f16x2 r2886, r2868, r2883;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r2889, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r2890, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r2891, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2892, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r2893, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2894, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r2895, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r2896, {low, high};
}
{
mul.f16x2 r2907, r2634, r2889;
}
{
mul.f16x2 r2910, r2778, r2890;
}
{
sub.f16x2 r2913, r2907, r2910;
}
{
mul.f16x2 r2916, r2634, r2890;
}
{
fma.rn.f16x2 r2919, r2778, r2889, r2916;
}
{
mul.f16x2 r2923, r2706, r2891;
}
{
mul.f16x2 r2926, r2850, r2892;
}
{
sub.f16x2 r2929, r2923, r2926;
}
{
mul.f16x2 r2932, r2706, r2892;
}
{
fma.rn.f16x2 r2935, r2850, r2891, r2932;
}
{
mul.f16x2 r2939, r2742, r2893;
}
{
mul.f16x2 r2942, r2886, r2894;
}
{
sub.f16x2 r2945, r2939, r2942;
}
{
mul.f16x2 r2948, r2742, r2894;
}
{
fma.rn.f16x2 r2951, r2886, r2893, r2948;
}
{
mul.f16x2 r2955, r2670, r2895;
}
{
mul.f16x2 r2958, r2814, r2896;
}
{
sub.f16x2 r2961, r2955, r2958;
}
{
mul.f16x2 r2964, r2670, r2896;
}
{
fma.rn.f16x2 r2967, r2814, r2895, r2964;
}
{
add.f16x2 %0, r2264, r2586;
}
{
add.f16x2 %1, r2276, r2598;
}
{
sub.f16x2 %10, r2264, r2586;
}
{
sub.f16x2 %11, r2276, r2598;
}
{
add.f16x2 %2, r2312, r2913;
}
{
add.f16x2 %3, r2456, r2919;
}
{
sub.f16x2 %12, r2312, r2913;
}
{
sub.f16x2 %13, r2456, r2919;
}
{
add.f16x2 %4, r2384, r2929;
}
{
add.f16x2 %5, r2528, r2935;
}
{
sub.f16x2 %14, r2384, r2929;
}
{
sub.f16x2 %15, r2528, r2935;
}
{
add.f16x2 %6, r2420, r2945;
}
{
add.f16x2 %7, r2564, r2951;
}
{
sub.f16x2 %16, r2420, r2945;
}
{
sub.f16x2 %17, r2564, r2951;
}
{
add.f16x2 %8, r2348, r2961;
}
{
add.f16x2 %9, r2492, r2967;
}
{
sub.f16x2 %18, r2348, r2961;
}
{
sub.f16x2 %19, r2492, r2967;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)));
};


#endif
