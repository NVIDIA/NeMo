#ifndef CUFFTDX_FFT_3125_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_3125_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1115, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<603>;
.reg .b32 r<10508>;
.reg .b64 rd<6>;
mov.u32 r10442, %50;
mov.u32 r10507, %tid.y;
mad.lo.s32 r10443, r10507, 25000, r10442;
mov.u32 r10444, %tid.x;
mov.f32 f594, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1, {low, high};
}
mov.f32 f596, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2, {low, high};
}
{
neg.f16x2 r3, r2;
}
mov.f32 f590, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r5, {low, high};
}
mov.f32 f592, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r10, {low, high};
}
{
add.f16x2 r11, %67, %54;
}
{
add.f16x2 r14, %74, r11;
}
{
add.f16x2 r17, %81, %65;
}
{
add.f16x2 r20, r14, r17;
}
{
add.f16x2 r23, %94, %79;
}
{
add.f16x2 r26, %51, r23;
}
{
add.f16x2 r29, %57, %92;
}
{
add.f16x2 r32, r26, r29;
}
{
add.f16x2 r35, %67, %54;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %74, r38;
}
{
add.f16x2 r44, %81, %65;
}
{
mul.f16x2 r47, r44, r5;
}
{
add.f16x2 r50, r41, r47;
}
{
sub.f16x2 r53, %94, %79;
}
{
mul.f16x2 r56, r53, r3;
}
{
sub.f16x2 r59, %57, %92;
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
add.f16x2 r71, %67, %54;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %74, r74;
}
{
add.f16x2 r80, %81, %65;
}
{
mul.f16x2 r83, r80, r5;
}
{
add.f16x2 r86, r77, r83;
}
{
sub.f16x2 r89, %94, %79;
}
{
mul.f16x2 r92, r89, r3;
}
{
sub.f16x2 r95, %57, %92;
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
add.f16x2 r107, %67, %54;
}
{
mul.f16x2 r110, r107, r5;
}
{
add.f16x2 r113, %74, r110;
}
{
add.f16x2 r116, %81, %65;
}
{
mul.f16x2 r119, r116, r9;
}
{
add.f16x2 r122, r113, r119;
}
{
sub.f16x2 r125, %94, %79;
}
{
mul.f16x2 r128, r125, r7;
}
{
sub.f16x2 r131, %57, %92;
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
add.f16x2 r143, %67, %54;
}
{
mul.f16x2 r146, r143, r5;
}
{
add.f16x2 r149, %74, r146;
}
{
add.f16x2 r152, %81, %65;
}
{
mul.f16x2 r155, r152, r9;
}
{
add.f16x2 r158, r149, r155;
}
{
sub.f16x2 r161, %94, %79;
}
{
mul.f16x2 r164, r161, r7;
}
{
sub.f16x2 r167, %57, %92;
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
add.f16x2 r179, %94, %79;
}
{
mul.f16x2 r182, r179, r1;
}
{
add.f16x2 r185, %51, r182;
}
{
add.f16x2 r188, %57, %92;
}
{
mul.f16x2 r191, r188, r5;
}
{
add.f16x2 r194, r185, r191;
}
{
sub.f16x2 r197, %67, %54;
}
{
mul.f16x2 r200, r197, r3;
}
{
sub.f16x2 r203, %81, %65;
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
add.f16x2 r215, %94, %79;
}
{
mul.f16x2 r218, r215, r1;
}
{
add.f16x2 r221, %51, r218;
}
{
add.f16x2 r224, %57, %92;
}
{
mul.f16x2 r227, r224, r5;
}
{
add.f16x2 r230, r221, r227;
}
{
sub.f16x2 r233, %67, %54;
}
{
mul.f16x2 r236, r233, r3;
}
{
sub.f16x2 r239, %81, %65;
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
add.f16x2 r251, %94, %79;
}
{
mul.f16x2 r254, r251, r5;
}
{
add.f16x2 r257, %51, r254;
}
{
add.f16x2 r260, %57, %92;
}
{
mul.f16x2 r263, r260, r9;
}
{
add.f16x2 r266, r257, r263;
}
{
sub.f16x2 r269, %67, %54;
}
{
mul.f16x2 r272, r269, r7;
}
{
sub.f16x2 r275, %81, %65;
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
add.f16x2 r287, %94, %79;
}
{
mul.f16x2 r290, r287, r5;
}
{
add.f16x2 r293, %51, r290;
}
{
add.f16x2 r296, %57, %92;
}
{
mul.f16x2 r299, r296, r9;
}
{
add.f16x2 r302, r293, r299;
}
{
sub.f16x2 r305, %67, %54;
}
{
mul.f16x2 r308, r305, r7;
}
{
sub.f16x2 r311, %81, %65;
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
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r323, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r324, {low, high};
}
{
neg.f16x2 r325, r324;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r327, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r328, {low, high};
}
{
neg.f16x2 r329, r328;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r331, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r332, {low, high};
}
{
add.f16x2 r333, %89, %77;
}
{
add.f16x2 r336, %96, r333;
}
{
add.f16x2 r339, %55, %87;
}
{
add.f16x2 r342, r336, r339;
}
{
add.f16x2 r345, %66, %52;
}
{
add.f16x2 r348, %73, r345;
}
{
add.f16x2 r351, %80, %64;
}
{
add.f16x2 r354, r348, r351;
}
{
add.f16x2 r357, %89, %77;
}
{
mul.f16x2 r360, r357, r323;
}
{
add.f16x2 r363, %96, r360;
}
{
add.f16x2 r366, %55, %87;
}
{
mul.f16x2 r369, r366, r327;
}
{
add.f16x2 r372, r363, r369;
}
{
sub.f16x2 r375, %66, %52;
}
{
mul.f16x2 r378, r375, r325;
}
{
sub.f16x2 r381, %80, %64;
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
add.f16x2 r393, %89, %77;
}
{
mul.f16x2 r396, r393, r323;
}
{
add.f16x2 r399, %96, r396;
}
{
add.f16x2 r402, %55, %87;
}
{
mul.f16x2 r405, r402, r327;
}
{
add.f16x2 r408, r399, r405;
}
{
sub.f16x2 r411, %66, %52;
}
{
mul.f16x2 r414, r411, r325;
}
{
sub.f16x2 r417, %80, %64;
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
add.f16x2 r429, %89, %77;
}
{
mul.f16x2 r432, r429, r327;
}
{
add.f16x2 r435, %96, r432;
}
{
add.f16x2 r438, %55, %87;
}
{
mul.f16x2 r441, r438, r331;
}
{
add.f16x2 r444, r435, r441;
}
{
sub.f16x2 r447, %66, %52;
}
{
mul.f16x2 r450, r447, r329;
}
{
sub.f16x2 r453, %80, %64;
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
add.f16x2 r465, %89, %77;
}
{
mul.f16x2 r468, r465, r327;
}
{
add.f16x2 r471, %96, r468;
}
{
add.f16x2 r474, %55, %87;
}
{
mul.f16x2 r477, r474, r331;
}
{
add.f16x2 r480, r471, r477;
}
{
sub.f16x2 r483, %66, %52;
}
{
mul.f16x2 r486, r483, r329;
}
{
sub.f16x2 r489, %80, %64;
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
add.f16x2 r501, %66, %52;
}
{
mul.f16x2 r504, r501, r323;
}
{
add.f16x2 r507, %73, r504;
}
{
add.f16x2 r510, %80, %64;
}
{
mul.f16x2 r513, r510, r327;
}
{
add.f16x2 r516, r507, r513;
}
{
sub.f16x2 r519, %89, %77;
}
{
mul.f16x2 r522, r519, r325;
}
{
sub.f16x2 r525, %55, %87;
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
add.f16x2 r537, %66, %52;
}
{
mul.f16x2 r540, r537, r323;
}
{
add.f16x2 r543, %73, r540;
}
{
add.f16x2 r546, %80, %64;
}
{
mul.f16x2 r549, r546, r327;
}
{
add.f16x2 r552, r543, r549;
}
{
sub.f16x2 r555, %89, %77;
}
{
mul.f16x2 r558, r555, r325;
}
{
sub.f16x2 r561, %55, %87;
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
add.f16x2 r573, %66, %52;
}
{
mul.f16x2 r576, r573, r327;
}
{
add.f16x2 r579, %73, r576;
}
{
add.f16x2 r582, %80, %64;
}
{
mul.f16x2 r585, r582, r331;
}
{
add.f16x2 r588, r579, r585;
}
{
sub.f16x2 r591, %89, %77;
}
{
mul.f16x2 r594, r591, r329;
}
{
sub.f16x2 r597, %55, %87;
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
add.f16x2 r609, %66, %52;
}
{
mul.f16x2 r612, r609, r327;
}
{
add.f16x2 r615, %73, r612;
}
{
add.f16x2 r618, %80, %64;
}
{
mul.f16x2 r621, r618, r331;
}
{
add.f16x2 r624, r615, r621;
}
{
sub.f16x2 r627, %89, %77;
}
{
mul.f16x2 r630, r627, r329;
}
{
sub.f16x2 r633, %55, %87;
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
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r645, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r646, {low, high};
}
{
neg.f16x2 r647, r646;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r649, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r650, {low, high};
}
{
neg.f16x2 r651, r650;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r653, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r654, {low, high};
}
{
add.f16x2 r655, %62, %99;
}
{
add.f16x2 r658, %69, r655;
}
{
add.f16x2 r661, %78, %60;
}
{
add.f16x2 r664, r658, r661;
}
{
add.f16x2 r667, %88, %75;
}
{
add.f16x2 r670, %95, r667;
}
{
add.f16x2 r673, %53, %86;
}
{
add.f16x2 r676, r670, r673;
}
{
add.f16x2 r679, %62, %99;
}
{
mul.f16x2 r682, r679, r645;
}
{
add.f16x2 r685, %69, r682;
}
{
add.f16x2 r688, %78, %60;
}
{
mul.f16x2 r691, r688, r649;
}
{
add.f16x2 r694, r685, r691;
}
{
sub.f16x2 r697, %88, %75;
}
{
mul.f16x2 r700, r697, r647;
}
{
sub.f16x2 r703, %53, %86;
}
{
mul.f16x2 r706, r703, r651;
}
{
add.f16x2 r709, r700, r706;
}
{
sub.f16x2 r712, r694, r709;
}
{
add.f16x2 r715, %62, %99;
}
{
mul.f16x2 r718, r715, r645;
}
{
add.f16x2 r721, %69, r718;
}
{
add.f16x2 r724, %78, %60;
}
{
mul.f16x2 r727, r724, r649;
}
{
add.f16x2 r730, r721, r727;
}
{
sub.f16x2 r733, %88, %75;
}
{
mul.f16x2 r736, r733, r647;
}
{
sub.f16x2 r739, %53, %86;
}
{
mul.f16x2 r742, r739, r651;
}
{
add.f16x2 r745, r736, r742;
}
{
add.f16x2 r748, r730, r745;
}
{
add.f16x2 r751, %62, %99;
}
{
mul.f16x2 r754, r751, r649;
}
{
add.f16x2 r757, %69, r754;
}
{
add.f16x2 r760, %78, %60;
}
{
mul.f16x2 r763, r760, r653;
}
{
add.f16x2 r766, r757, r763;
}
{
sub.f16x2 r769, %88, %75;
}
{
mul.f16x2 r772, r769, r651;
}
{
sub.f16x2 r775, %53, %86;
}
{
mul.f16x2 r778, r775, r654;
}
{
add.f16x2 r781, r772, r778;
}
{
sub.f16x2 r784, r766, r781;
}
{
add.f16x2 r787, %62, %99;
}
{
mul.f16x2 r790, r787, r649;
}
{
add.f16x2 r793, %69, r790;
}
{
add.f16x2 r796, %78, %60;
}
{
mul.f16x2 r799, r796, r653;
}
{
add.f16x2 r802, r793, r799;
}
{
sub.f16x2 r805, %88, %75;
}
{
mul.f16x2 r808, r805, r651;
}
{
sub.f16x2 r811, %53, %86;
}
{
mul.f16x2 r814, r811, r654;
}
{
add.f16x2 r817, r808, r814;
}
{
add.f16x2 r820, r802, r817;
}
{
add.f16x2 r823, %88, %75;
}
{
mul.f16x2 r826, r823, r645;
}
{
add.f16x2 r829, %95, r826;
}
{
add.f16x2 r832, %53, %86;
}
{
mul.f16x2 r835, r832, r649;
}
{
add.f16x2 r838, r829, r835;
}
{
sub.f16x2 r841, %62, %99;
}
{
mul.f16x2 r844, r841, r647;
}
{
sub.f16x2 r847, %78, %60;
}
{
mul.f16x2 r850, r847, r651;
}
{
add.f16x2 r853, r844, r850;
}
{
add.f16x2 r856, r838, r853;
}
{
add.f16x2 r859, %88, %75;
}
{
mul.f16x2 r862, r859, r645;
}
{
add.f16x2 r865, %95, r862;
}
{
add.f16x2 r868, %53, %86;
}
{
mul.f16x2 r871, r868, r649;
}
{
add.f16x2 r874, r865, r871;
}
{
sub.f16x2 r877, %62, %99;
}
{
mul.f16x2 r880, r877, r647;
}
{
sub.f16x2 r883, %78, %60;
}
{
mul.f16x2 r886, r883, r651;
}
{
add.f16x2 r889, r880, r886;
}
{
sub.f16x2 r892, r874, r889;
}
{
add.f16x2 r895, %88, %75;
}
{
mul.f16x2 r898, r895, r649;
}
{
add.f16x2 r901, %95, r898;
}
{
add.f16x2 r904, %53, %86;
}
{
mul.f16x2 r907, r904, r653;
}
{
add.f16x2 r910, r901, r907;
}
{
sub.f16x2 r913, %62, %99;
}
{
mul.f16x2 r916, r913, r651;
}
{
sub.f16x2 r919, %78, %60;
}
{
mul.f16x2 r922, r919, r654;
}
{
add.f16x2 r925, r916, r922;
}
{
add.f16x2 r928, r910, r925;
}
{
add.f16x2 r931, %88, %75;
}
{
mul.f16x2 r934, r931, r649;
}
{
add.f16x2 r937, %95, r934;
}
{
add.f16x2 r940, %53, %86;
}
{
mul.f16x2 r943, r940, r653;
}
{
add.f16x2 r946, r937, r943;
}
{
sub.f16x2 r949, %62, %99;
}
{
mul.f16x2 r952, r949, r651;
}
{
sub.f16x2 r955, %78, %60;
}
{
mul.f16x2 r958, r955, r654;
}
{
add.f16x2 r961, r952, r958;
}
{
sub.f16x2 r964, r946, r961;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r967, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r968, {low, high};
}
{
neg.f16x2 r969, r968;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r971, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r972, {low, high};
}
{
neg.f16x2 r973, r972;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r975, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r976, {low, high};
}
{
add.f16x2 r977, %85, %71;
}
{
add.f16x2 r980, %91, r977;
}
{
add.f16x2 r983, %100, %83;
}
{
add.f16x2 r986, r980, r983;
}
{
add.f16x2 r989, %61, %97;
}
{
add.f16x2 r992, %68, r989;
}
{
add.f16x2 r995, %76, %59;
}
{
add.f16x2 r998, r992, r995;
}
{
add.f16x2 r1001, %85, %71;
}
{
mul.f16x2 r1004, r1001, r967;
}
{
add.f16x2 r1007, %91, r1004;
}
{
add.f16x2 r1010, %100, %83;
}
{
mul.f16x2 r1013, r1010, r971;
}
{
add.f16x2 r1016, r1007, r1013;
}
{
sub.f16x2 r1019, %61, %97;
}
{
mul.f16x2 r1022, r1019, r969;
}
{
sub.f16x2 r1025, %76, %59;
}
{
mul.f16x2 r1028, r1025, r973;
}
{
add.f16x2 r1031, r1022, r1028;
}
{
sub.f16x2 r1034, r1016, r1031;
}
{
add.f16x2 r1037, %85, %71;
}
{
mul.f16x2 r1040, r1037, r967;
}
{
add.f16x2 r1043, %91, r1040;
}
{
add.f16x2 r1046, %100, %83;
}
{
mul.f16x2 r1049, r1046, r971;
}
{
add.f16x2 r1052, r1043, r1049;
}
{
sub.f16x2 r1055, %61, %97;
}
{
mul.f16x2 r1058, r1055, r969;
}
{
sub.f16x2 r1061, %76, %59;
}
{
mul.f16x2 r1064, r1061, r973;
}
{
add.f16x2 r1067, r1058, r1064;
}
{
add.f16x2 r1070, r1052, r1067;
}
{
add.f16x2 r1073, %85, %71;
}
{
mul.f16x2 r1076, r1073, r971;
}
{
add.f16x2 r1079, %91, r1076;
}
{
add.f16x2 r1082, %100, %83;
}
{
mul.f16x2 r1085, r1082, r975;
}
{
add.f16x2 r1088, r1079, r1085;
}
{
sub.f16x2 r1091, %61, %97;
}
{
mul.f16x2 r1094, r1091, r973;
}
{
sub.f16x2 r1097, %76, %59;
}
{
mul.f16x2 r1100, r1097, r976;
}
{
add.f16x2 r1103, r1094, r1100;
}
{
sub.f16x2 r1106, r1088, r1103;
}
{
add.f16x2 r1109, %85, %71;
}
{
mul.f16x2 r1112, r1109, r971;
}
{
add.f16x2 r1115, %91, r1112;
}
{
add.f16x2 r1118, %100, %83;
}
{
mul.f16x2 r1121, r1118, r975;
}
{
add.f16x2 r1124, r1115, r1121;
}
{
sub.f16x2 r1127, %61, %97;
}
{
mul.f16x2 r1130, r1127, r973;
}
{
sub.f16x2 r1133, %76, %59;
}
{
mul.f16x2 r1136, r1133, r976;
}
{
add.f16x2 r1139, r1130, r1136;
}
{
add.f16x2 r1142, r1124, r1139;
}
{
add.f16x2 r1145, %61, %97;
}
{
mul.f16x2 r1148, r1145, r967;
}
{
add.f16x2 r1151, %68, r1148;
}
{
add.f16x2 r1154, %76, %59;
}
{
mul.f16x2 r1157, r1154, r971;
}
{
add.f16x2 r1160, r1151, r1157;
}
{
sub.f16x2 r1163, %85, %71;
}
{
mul.f16x2 r1166, r1163, r969;
}
{
sub.f16x2 r1169, %100, %83;
}
{
mul.f16x2 r1172, r1169, r973;
}
{
add.f16x2 r1175, r1166, r1172;
}
{
add.f16x2 r1178, r1160, r1175;
}
{
add.f16x2 r1181, %61, %97;
}
{
mul.f16x2 r1184, r1181, r967;
}
{
add.f16x2 r1187, %68, r1184;
}
{
add.f16x2 r1190, %76, %59;
}
{
mul.f16x2 r1193, r1190, r971;
}
{
add.f16x2 r1196, r1187, r1193;
}
{
sub.f16x2 r1199, %85, %71;
}
{
mul.f16x2 r1202, r1199, r969;
}
{
sub.f16x2 r1205, %100, %83;
}
{
mul.f16x2 r1208, r1205, r973;
}
{
add.f16x2 r1211, r1202, r1208;
}
{
sub.f16x2 r1214, r1196, r1211;
}
{
add.f16x2 r1217, %61, %97;
}
{
mul.f16x2 r1220, r1217, r971;
}
{
add.f16x2 r1223, %68, r1220;
}
{
add.f16x2 r1226, %76, %59;
}
{
mul.f16x2 r1229, r1226, r975;
}
{
add.f16x2 r1232, r1223, r1229;
}
{
sub.f16x2 r1235, %85, %71;
}
{
mul.f16x2 r1238, r1235, r973;
}
{
sub.f16x2 r1241, %100, %83;
}
{
mul.f16x2 r1244, r1241, r976;
}
{
add.f16x2 r1247, r1238, r1244;
}
{
add.f16x2 r1250, r1232, r1247;
}
{
add.f16x2 r1253, %61, %97;
}
{
mul.f16x2 r1256, r1253, r971;
}
{
add.f16x2 r1259, %68, r1256;
}
{
add.f16x2 r1262, %76, %59;
}
{
mul.f16x2 r1265, r1262, r975;
}
{
add.f16x2 r1268, r1259, r1265;
}
{
sub.f16x2 r1271, %85, %71;
}
{
mul.f16x2 r1274, r1271, r973;
}
{
sub.f16x2 r1277, %100, %83;
}
{
mul.f16x2 r1280, r1277, r976;
}
{
add.f16x2 r1283, r1274, r1280;
}
{
sub.f16x2 r1286, r1268, r1283;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1289, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r1290, {low, high};
}
{
neg.f16x2 r1291, r1290;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r1293, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r1294, {low, high};
}
{
neg.f16x2 r1295, r1294;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r1298, {low, high};
}
{
add.f16x2 r1299, %58, %93;
}
{
add.f16x2 r1302, %63, r1299;
}
{
add.f16x2 r1305, %72, %56;
}
{
add.f16x2 r1308, r1302, r1305;
}
{
add.f16x2 r1311, %84, %70;
}
{
add.f16x2 r1314, %90, r1311;
}
{
add.f16x2 r1317, %98, %82;
}
{
add.f16x2 r1320, r1314, r1317;
}
{
add.f16x2 r1323, %58, %93;
}
{
mul.f16x2 r1326, r1323, r1289;
}
{
add.f16x2 r1329, %63, r1326;
}
{
add.f16x2 r1332, %72, %56;
}
{
mul.f16x2 r1335, r1332, r1293;
}
{
add.f16x2 r1338, r1329, r1335;
}
{
sub.f16x2 r1341, %84, %70;
}
{
mul.f16x2 r1344, r1341, r1291;
}
{
sub.f16x2 r1347, %98, %82;
}
{
mul.f16x2 r1350, r1347, r1295;
}
{
add.f16x2 r1353, r1344, r1350;
}
{
sub.f16x2 r1356, r1338, r1353;
}
{
add.f16x2 r1359, %58, %93;
}
{
mul.f16x2 r1362, r1359, r1289;
}
{
add.f16x2 r1365, %63, r1362;
}
{
add.f16x2 r1368, %72, %56;
}
{
mul.f16x2 r1371, r1368, r1293;
}
{
add.f16x2 r1374, r1365, r1371;
}
{
sub.f16x2 r1377, %84, %70;
}
{
mul.f16x2 r1380, r1377, r1291;
}
{
sub.f16x2 r1383, %98, %82;
}
{
mul.f16x2 r1386, r1383, r1295;
}
{
add.f16x2 r1389, r1380, r1386;
}
{
add.f16x2 r1392, r1374, r1389;
}
{
add.f16x2 r1395, %58, %93;
}
{
mul.f16x2 r1398, r1395, r1293;
}
{
add.f16x2 r1401, %63, r1398;
}
{
add.f16x2 r1404, %72, %56;
}
{
mul.f16x2 r1407, r1404, r1297;
}
{
add.f16x2 r1410, r1401, r1407;
}
{
sub.f16x2 r1413, %84, %70;
}
{
mul.f16x2 r1416, r1413, r1295;
}
{
sub.f16x2 r1419, %98, %82;
}
{
mul.f16x2 r1422, r1419, r1298;
}
{
add.f16x2 r1425, r1416, r1422;
}
{
sub.f16x2 r1428, r1410, r1425;
}
{
add.f16x2 r1431, %58, %93;
}
{
mul.f16x2 r1434, r1431, r1293;
}
{
add.f16x2 r1437, %63, r1434;
}
{
add.f16x2 r1440, %72, %56;
}
{
mul.f16x2 r1443, r1440, r1297;
}
{
add.f16x2 r1446, r1437, r1443;
}
{
sub.f16x2 r1449, %84, %70;
}
{
mul.f16x2 r1452, r1449, r1295;
}
{
sub.f16x2 r1455, %98, %82;
}
{
mul.f16x2 r1458, r1455, r1298;
}
{
add.f16x2 r1461, r1452, r1458;
}
{
add.f16x2 r1464, r1446, r1461;
}
{
add.f16x2 r1467, %84, %70;
}
{
mul.f16x2 r1470, r1467, r1289;
}
{
add.f16x2 r1473, %90, r1470;
}
{
add.f16x2 r1476, %98, %82;
}
{
mul.f16x2 r1479, r1476, r1293;
}
{
add.f16x2 r1482, r1473, r1479;
}
{
sub.f16x2 r1485, %58, %93;
}
{
mul.f16x2 r1488, r1485, r1291;
}
{
sub.f16x2 r1491, %72, %56;
}
{
mul.f16x2 r1494, r1491, r1295;
}
{
add.f16x2 r1497, r1488, r1494;
}
{
add.f16x2 r1500, r1482, r1497;
}
{
add.f16x2 r1503, %84, %70;
}
{
mul.f16x2 r1506, r1503, r1289;
}
{
add.f16x2 r1509, %90, r1506;
}
{
add.f16x2 r1512, %98, %82;
}
{
mul.f16x2 r1515, r1512, r1293;
}
{
add.f16x2 r1518, r1509, r1515;
}
{
sub.f16x2 r1521, %58, %93;
}
{
mul.f16x2 r1524, r1521, r1291;
}
{
sub.f16x2 r1527, %72, %56;
}
{
mul.f16x2 r1530, r1527, r1295;
}
{
add.f16x2 r1533, r1524, r1530;
}
{
sub.f16x2 r1536, r1518, r1533;
}
{
add.f16x2 r1539, %84, %70;
}
{
mul.f16x2 r1542, r1539, r1293;
}
{
add.f16x2 r1545, %90, r1542;
}
{
add.f16x2 r1548, %98, %82;
}
{
mul.f16x2 r1551, r1548, r1297;
}
{
add.f16x2 r1554, r1545, r1551;
}
{
sub.f16x2 r1557, %58, %93;
}
{
mul.f16x2 r1560, r1557, r1295;
}
{
sub.f16x2 r1563, %72, %56;
}
{
mul.f16x2 r1566, r1563, r1298;
}
{
add.f16x2 r1569, r1560, r1566;
}
{
add.f16x2 r1572, r1554, r1569;
}
{
add.f16x2 r1575, %84, %70;
}
{
mul.f16x2 r1578, r1575, r1293;
}
{
add.f16x2 r1581, %90, r1578;
}
{
add.f16x2 r1584, %98, %82;
}
{
mul.f16x2 r1587, r1584, r1297;
}
{
add.f16x2 r1590, r1581, r1587;
}
{
sub.f16x2 r1593, %58, %93;
}
{
mul.f16x2 r1596, r1593, r1295;
}
{
sub.f16x2 r1599, %72, %56;
}
{
mul.f16x2 r1602, r1599, r1298;
}
{
add.f16x2 r1605, r1596, r1602;
}
{
sub.f16x2 r1608, r1590, r1605;
}
mov.f32 f330, 0f3F77F511;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r1611, {low, high};
}
mov.f32 f332, 0f3E7EA890;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f332;
cvt.rn.f16.f32 high, f332;
mov.b32 r1612, {low, high};
}
mov.f32 f334, 0f3F6055A2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f334;
cvt.rn.f16.f32 high, f334;
mov.b32 r1613, {low, high};
}
mov.f32 f336, 0f3EF6A86B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f336;
cvt.rn.f16.f32 high, f336;
mov.b32 r1614, {low, high};
}
mov.f32 f338, 0f3F3A9DB0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f338;
cvt.rn.f16.f32 high, f338;
mov.b32 r1615, {low, high};
}
mov.f32 f340, 0f3F2F3E7B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r1616, {low, high};
}
mov.f32 f342, 0f3F092BF2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f342;
cvt.rn.f16.f32 high, f342;
mov.b32 r1617, {low, high};
}
mov.f32 f344, 0f3F5825E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f344;
cvt.rn.f16.f32 high, f344;
mov.b32 r1618, {low, high};
}
mov.f32 f350, 0f3D809851;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f350;
cvt.rn.f16.f32 high, f350;
mov.b32 r1621, {low, high};
}
mov.f32 f352, 0f3F7F7EAE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r1622, {low, high};
}
mov.f32 f358, 0fBED9FFBE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f358;
cvt.rn.f16.f32 high, f358;
mov.b32 r1625, {low, high};
}
mov.f32 f360, 0f3F67A2BF;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f360;
cvt.rn.f16.f32 high, f360;
mov.b32 r1626, {low, high};
}
mov.f32 f390, 0fBF232E38;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f390;
cvt.rn.f16.f32 high, f390;
mov.b32 r1627, {low, high};
}
mov.f32 f364, 0f3F45405B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f364;
cvt.rn.f16.f32 high, f364;
mov.b32 r1628, {low, high};
}
mov.f32 f374, 0fBF7DFB3B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f374;
cvt.rn.f16.f32 high, f374;
mov.b32 r1633, {low, high};
}
mov.f32 f376, 0f3E00575B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r1634, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f390;
cvt.rn.f16.f32 high, f390;
mov.b32 r1641, {low, high};
}
mov.f32 f392, 0fBF45405B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f392;
cvt.rn.f16.f32 high, f392;
mov.b32 r1642, {low, high};
}
{
mul.f16x2 r1659, r390, r1611;
}
{
mul.f16x2 r1662, r534, r1612;
}
{
sub.f16x2 r1665, r1659, r1662;
}
{
mul.f16x2 r1668, r390, r1612;
}
{
fma.rn.f16x2 r1671, r534, r1611, r1668;
}
{
mul.f16x2 r1675, r712, r1613;
}
{
mul.f16x2 r1678, r856, r1614;
}
{
sub.f16x2 r1681, r1675, r1678;
}
{
mul.f16x2 r1684, r712, r1614;
}
{
fma.rn.f16x2 r1687, r856, r1613, r1684;
}
{
mul.f16x2 r1691, r1034, r1615;
}
{
mul.f16x2 r1694, r1178, r1616;
}
{
sub.f16x2 r1697, r1691, r1694;
}
{
mul.f16x2 r1700, r1034, r1616;
}
{
fma.rn.f16x2 r1703, r1178, r1615, r1700;
}
{
mul.f16x2 r1707, r1356, r1617;
}
{
mul.f16x2 r1710, r1500, r1618;
}
{
sub.f16x2 r1713, r1707, r1710;
}
{
mul.f16x2 r1716, r1356, r1618;
}
{
fma.rn.f16x2 r1719, r1500, r1617, r1716;
}
{
mul.f16x2 r1723, r462, r1613;
}
{
mul.f16x2 r1726, r606, r1614;
}
{
sub.f16x2 r1729, r1723, r1726;
}
{
mul.f16x2 r1732, r462, r1614;
}
{
fma.rn.f16x2 r1735, r606, r1613, r1732;
}
{
mul.f16x2 r1739, r784, r1617;
}
{
mul.f16x2 r1742, r928, r1618;
}
{
sub.f16x2 r1745, r1739, r1742;
}
{
mul.f16x2 r1748, r784, r1618;
}
{
fma.rn.f16x2 r1751, r928, r1617, r1748;
}
{
mul.f16x2 r1755, r1106, r1621;
}
{
mul.f16x2 r1758, r1250, r1622;
}
{
sub.f16x2 r1761, r1755, r1758;
}
{
mul.f16x2 r1764, r1106, r1622;
}
{
fma.rn.f16x2 r1767, r1250, r1621, r1764;
}
{
mul.f16x2 r1771, r1428, r1625;
}
{
mul.f16x2 r1774, r1572, r1626;
}
{
sub.f16x2 r1777, r1771, r1774;
}
{
mul.f16x2 r1780, r1428, r1626;
}
{
fma.rn.f16x2 r1783, r1572, r1625, r1780;
}
{
mul.f16x2 r1787, r498, r1615;
}
{
mul.f16x2 r1790, r642, r1616;
}
{
sub.f16x2 r1793, r1787, r1790;
}
{
mul.f16x2 r1796, r498, r1616;
}
{
fma.rn.f16x2 r1799, r642, r1615, r1796;
}
{
mul.f16x2 r1803, r820, r1621;
}
{
mul.f16x2 r1806, r964, r1622;
}
{
sub.f16x2 r1809, r1803, r1806;
}
{
mul.f16x2 r1812, r820, r1622;
}
{
fma.rn.f16x2 r1815, r964, r1621, r1812;
}
{
mul.f16x2 r1819, r1142, r1627;
}
{
mul.f16x2 r1822, r1286, r1628;
}
{
sub.f16x2 r1825, r1819, r1822;
}
{
mul.f16x2 r1828, r1142, r1628;
}
{
fma.rn.f16x2 r1831, r1286, r1627, r1828;
}
{
mul.f16x2 r1835, r1464, r1633;
}
{
mul.f16x2 r1838, r1608, r1634;
}
{
sub.f16x2 r1841, r1835, r1838;
}
{
mul.f16x2 r1844, r1464, r1634;
}
{
fma.rn.f16x2 r1847, r1608, r1633, r1844;
}
{
mul.f16x2 r1851, r426, r1617;
}
{
mul.f16x2 r1854, r570, r1618;
}
{
sub.f16x2 r1857, r1851, r1854;
}
{
mul.f16x2 r1860, r426, r1618;
}
{
fma.rn.f16x2 r1863, r570, r1617, r1860;
}
{
mul.f16x2 r1867, r748, r1625;
}
{
mul.f16x2 r1870, r892, r1626;
}
{
sub.f16x2 r1873, r1867, r1870;
}
{
mul.f16x2 r1876, r748, r1626;
}
{
fma.rn.f16x2 r1879, r892, r1625, r1876;
}
{
mul.f16x2 r1883, r1070, r1633;
}
{
mul.f16x2 r1886, r1214, r1634;
}
{
sub.f16x2 r1889, r1883, r1886;
}
{
mul.f16x2 r1892, r1070, r1634;
}
{
fma.rn.f16x2 r1895, r1214, r1633, r1892;
}
{
mul.f16x2 r1899, r1392, r1641;
}
{
mul.f16x2 r1902, r1536, r1642;
}
{
sub.f16x2 r1905, r1899, r1902;
}
{
mul.f16x2 r1908, r1392, r1642;
}
{
fma.rn.f16x2 r1911, r1536, r1641, r1908;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1915, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r1916, {low, high};
}
{
neg.f16x2 r1917, r1916;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r1919, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r1920, {low, high};
}
{
neg.f16x2 r1921, r1920;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1923, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r1924, {low, high};
}
{
add.f16x2 r1925, r342, r1308;
}
{
add.f16x2 r1928, r20, r1925;
}
{
add.f16x2 r1931, r664, r986;
}
{
add.f16x2 r1934, r1928, r1931;
}
{
add.f16x2 r1937, r354, r1320;
}
{
add.f16x2 r1940, r32, r1937;
}
{
add.f16x2 r1943, r676, r998;
}
{
add.f16x2 r1946, r1940, r1943;
}
{
add.f16x2 r1949, r342, r1308;
}
{
mul.f16x2 r1952, r1949, r1915;
}
{
add.f16x2 r1955, r20, r1952;
}
{
add.f16x2 r1958, r664, r986;
}
{
mul.f16x2 r1961, r1958, r1919;
}
{
add.f16x2 r1964, r1955, r1961;
}
{
sub.f16x2 r1967, r354, r1320;
}
{
mul.f16x2 r1970, r1967, r1917;
}
{
sub.f16x2 r1973, r676, r998;
}
{
mul.f16x2 r1976, r1973, r1921;
}
{
add.f16x2 r1979, r1970, r1976;
}
{
sub.f16x2 r1982, r1964, r1979;
}
{
add.f16x2 r1985, r342, r1308;
}
{
mul.f16x2 r1988, r1985, r1915;
}
{
add.f16x2 r1991, r20, r1988;
}
{
add.f16x2 r1994, r664, r986;
}
{
mul.f16x2 r1997, r1994, r1919;
}
{
add.f16x2 r2000, r1991, r1997;
}
{
sub.f16x2 r2003, r354, r1320;
}
{
mul.f16x2 r2006, r2003, r1917;
}
{
sub.f16x2 r2009, r676, r998;
}
{
mul.f16x2 r2012, r2009, r1921;
}
{
add.f16x2 r2015, r2006, r2012;
}
{
add.f16x2 r2018, r2000, r2015;
}
{
add.f16x2 r2021, r342, r1308;
}
{
mul.f16x2 r2024, r2021, r1919;
}
{
add.f16x2 r2027, r20, r2024;
}
{
add.f16x2 r2030, r664, r986;
}
{
mul.f16x2 r2033, r2030, r1923;
}
{
add.f16x2 r2036, r2027, r2033;
}
{
sub.f16x2 r2039, r354, r1320;
}
{
mul.f16x2 r2042, r2039, r1921;
}
{
sub.f16x2 r2045, r676, r998;
}
{
mul.f16x2 r2048, r2045, r1924;
}
{
add.f16x2 r2051, r2042, r2048;
}
{
sub.f16x2 r2054, r2036, r2051;
}
{
add.f16x2 r2057, r342, r1308;
}
{
mul.f16x2 r2060, r2057, r1919;
}
{
add.f16x2 r2063, r20, r2060;
}
{
add.f16x2 r2066, r664, r986;
}
{
mul.f16x2 r2069, r2066, r1923;
}
{
add.f16x2 r2072, r2063, r2069;
}
{
sub.f16x2 r2075, r354, r1320;
}
{
mul.f16x2 r2078, r2075, r1921;
}
{
sub.f16x2 r2081, r676, r998;
}
{
mul.f16x2 r2084, r2081, r1924;
}
{
add.f16x2 r2087, r2078, r2084;
}
{
add.f16x2 r2090, r2072, r2087;
}
{
add.f16x2 r2093, r354, r1320;
}
{
mul.f16x2 r2096, r2093, r1915;
}
{
add.f16x2 r2099, r32, r2096;
}
{
add.f16x2 r2102, r676, r998;
}
{
mul.f16x2 r2105, r2102, r1919;
}
{
add.f16x2 r2108, r2099, r2105;
}
{
sub.f16x2 r2111, r342, r1308;
}
{
mul.f16x2 r2114, r2111, r1917;
}
{
sub.f16x2 r2117, r664, r986;
}
{
mul.f16x2 r2120, r2117, r1921;
}
{
add.f16x2 r2123, r2114, r2120;
}
{
add.f16x2 r2126, r2108, r2123;
}
{
add.f16x2 r2129, r354, r1320;
}
{
mul.f16x2 r2132, r2129, r1915;
}
{
add.f16x2 r2135, r32, r2132;
}
{
add.f16x2 r2138, r676, r998;
}
{
mul.f16x2 r2141, r2138, r1919;
}
{
add.f16x2 r2144, r2135, r2141;
}
{
sub.f16x2 r2147, r342, r1308;
}
{
mul.f16x2 r2150, r2147, r1917;
}
{
sub.f16x2 r2153, r664, r986;
}
{
mul.f16x2 r2156, r2153, r1921;
}
{
add.f16x2 r2159, r2150, r2156;
}
{
sub.f16x2 r2162, r2144, r2159;
}
{
add.f16x2 r2165, r354, r1320;
}
{
mul.f16x2 r2168, r2165, r1919;
}
{
add.f16x2 r2171, r32, r2168;
}
{
add.f16x2 r2174, r676, r998;
}
{
mul.f16x2 r2177, r2174, r1923;
}
{
add.f16x2 r2180, r2171, r2177;
}
{
sub.f16x2 r2183, r342, r1308;
}
{
mul.f16x2 r2186, r2183, r1921;
}
{
sub.f16x2 r2189, r664, r986;
}
{
mul.f16x2 r2192, r2189, r1924;
}
{
add.f16x2 r2195, r2186, r2192;
}
{
add.f16x2 r2198, r2180, r2195;
}
{
add.f16x2 r2201, r354, r1320;
}
{
mul.f16x2 r2204, r2201, r1919;
}
{
add.f16x2 r2207, r32, r2204;
}
{
add.f16x2 r2210, r676, r998;
}
{
mul.f16x2 r2213, r2210, r1923;
}
{
add.f16x2 r2216, r2207, r2213;
}
{
sub.f16x2 r2219, r342, r1308;
}
{
mul.f16x2 r2222, r2219, r1921;
}
{
sub.f16x2 r2225, r664, r986;
}
{
mul.f16x2 r2228, r2225, r1924;
}
{
add.f16x2 r2231, r2222, r2228;
}
{
sub.f16x2 r2234, r2216, r2231;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2237, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2238, {low, high};
}
{
neg.f16x2 r2239, r2238;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r2241, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r2242, {low, high};
}
{
neg.f16x2 r2243, r2242;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2245, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2246, {low, high};
}
{
add.f16x2 r2247, r1665, r1713;
}
{
add.f16x2 r2250, r68, r2247;
}
{
add.f16x2 r2253, r1681, r1697;
}
{
add.f16x2 r2256, r2250, r2253;
}
{
add.f16x2 r2259, r1671, r1719;
}
{
add.f16x2 r2262, r212, r2259;
}
{
add.f16x2 r2265, r1687, r1703;
}
{
add.f16x2 r2268, r2262, r2265;
}
{
add.f16x2 r2271, r1665, r1713;
}
{
mul.f16x2 r2274, r2271, r2237;
}
{
add.f16x2 r2277, r68, r2274;
}
{
add.f16x2 r2280, r1681, r1697;
}
{
mul.f16x2 r2283, r2280, r2241;
}
{
add.f16x2 r2286, r2277, r2283;
}
{
sub.f16x2 r2289, r1671, r1719;
}
{
mul.f16x2 r2292, r2289, r2239;
}
{
sub.f16x2 r2295, r1687, r1703;
}
{
mul.f16x2 r2298, r2295, r2243;
}
{
add.f16x2 r2301, r2292, r2298;
}
{
sub.f16x2 r2304, r2286, r2301;
}
{
add.f16x2 r2307, r1665, r1713;
}
{
mul.f16x2 r2310, r2307, r2237;
}
{
add.f16x2 r2313, r68, r2310;
}
{
add.f16x2 r2316, r1681, r1697;
}
{
mul.f16x2 r2319, r2316, r2241;
}
{
add.f16x2 r2322, r2313, r2319;
}
{
sub.f16x2 r2325, r1671, r1719;
}
{
mul.f16x2 r2328, r2325, r2239;
}
{
sub.f16x2 r2331, r1687, r1703;
}
{
mul.f16x2 r2334, r2331, r2243;
}
{
add.f16x2 r2337, r2328, r2334;
}
{
add.f16x2 r2340, r2322, r2337;
}
{
add.f16x2 r2343, r1665, r1713;
}
{
mul.f16x2 r2346, r2343, r2241;
}
{
add.f16x2 r2349, r68, r2346;
}
{
add.f16x2 r2352, r1681, r1697;
}
{
mul.f16x2 r2355, r2352, r2245;
}
{
add.f16x2 r2358, r2349, r2355;
}
{
sub.f16x2 r2361, r1671, r1719;
}
{
mul.f16x2 r2364, r2361, r2243;
}
{
sub.f16x2 r2367, r1687, r1703;
}
{
mul.f16x2 r2370, r2367, r2246;
}
{
add.f16x2 r2373, r2364, r2370;
}
{
sub.f16x2 r2376, r2358, r2373;
}
{
add.f16x2 r2379, r1665, r1713;
}
{
mul.f16x2 r2382, r2379, r2241;
}
{
add.f16x2 r2385, r68, r2382;
}
{
add.f16x2 r2388, r1681, r1697;
}
{
mul.f16x2 r2391, r2388, r2245;
}
{
add.f16x2 r2394, r2385, r2391;
}
{
sub.f16x2 r2397, r1671, r1719;
}
{
mul.f16x2 r2400, r2397, r2243;
}
{
sub.f16x2 r2403, r1687, r1703;
}
{
mul.f16x2 r2406, r2403, r2246;
}
{
add.f16x2 r2409, r2400, r2406;
}
{
add.f16x2 r2412, r2394, r2409;
}
{
add.f16x2 r2415, r1671, r1719;
}
{
mul.f16x2 r2418, r2415, r2237;
}
{
add.f16x2 r2421, r212, r2418;
}
{
add.f16x2 r2424, r1687, r1703;
}
{
mul.f16x2 r2427, r2424, r2241;
}
{
add.f16x2 r2430, r2421, r2427;
}
{
sub.f16x2 r2433, r1665, r1713;
}
{
mul.f16x2 r2436, r2433, r2239;
}
{
sub.f16x2 r2439, r1681, r1697;
}
{
mul.f16x2 r2442, r2439, r2243;
}
{
add.f16x2 r2445, r2436, r2442;
}
{
add.f16x2 r2448, r2430, r2445;
}
{
add.f16x2 r2451, r1671, r1719;
}
{
mul.f16x2 r2454, r2451, r2237;
}
{
add.f16x2 r2457, r212, r2454;
}
{
add.f16x2 r2460, r1687, r1703;
}
{
mul.f16x2 r2463, r2460, r2241;
}
{
add.f16x2 r2466, r2457, r2463;
}
{
sub.f16x2 r2469, r1665, r1713;
}
{
mul.f16x2 r2472, r2469, r2239;
}
{
sub.f16x2 r2475, r1681, r1697;
}
{
mul.f16x2 r2478, r2475, r2243;
}
{
add.f16x2 r2481, r2472, r2478;
}
{
sub.f16x2 r2484, r2466, r2481;
}
{
add.f16x2 r2487, r1671, r1719;
}
{
mul.f16x2 r2490, r2487, r2241;
}
{
add.f16x2 r2493, r212, r2490;
}
{
add.f16x2 r2496, r1687, r1703;
}
{
mul.f16x2 r2499, r2496, r2245;
}
{
add.f16x2 r2502, r2493, r2499;
}
{
sub.f16x2 r2505, r1665, r1713;
}
{
mul.f16x2 r2508, r2505, r2243;
}
{
sub.f16x2 r2511, r1681, r1697;
}
{
mul.f16x2 r2514, r2511, r2246;
}
{
add.f16x2 r2517, r2508, r2514;
}
{
add.f16x2 r2520, r2502, r2517;
}
{
add.f16x2 r2523, r1671, r1719;
}
{
mul.f16x2 r2526, r2523, r2241;
}
{
add.f16x2 r2529, r212, r2526;
}
{
add.f16x2 r2532, r1687, r1703;
}
{
mul.f16x2 r2535, r2532, r2245;
}
{
add.f16x2 r2538, r2529, r2535;
}
{
sub.f16x2 r2541, r1665, r1713;
}
{
mul.f16x2 r2544, r2541, r2243;
}
{
sub.f16x2 r2547, r1681, r1697;
}
{
mul.f16x2 r2550, r2547, r2246;
}
{
add.f16x2 r2553, r2544, r2550;
}
{
sub.f16x2 r2556, r2538, r2553;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2559, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2560, {low, high};
}
{
neg.f16x2 r2561, r2560;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r2563, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r2564, {low, high};
}
{
neg.f16x2 r2565, r2564;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2567, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2568, {low, high};
}
{
add.f16x2 r2569, r1729, r1777;
}
{
add.f16x2 r2572, r140, r2569;
}
{
add.f16x2 r2575, r1745, r1761;
}
{
add.f16x2 r2578, r2572, r2575;
}
{
add.f16x2 r2581, r1735, r1783;
}
{
add.f16x2 r2584, r284, r2581;
}
{
add.f16x2 r2587, r1751, r1767;
}
{
add.f16x2 r2590, r2584, r2587;
}
{
add.f16x2 r2593, r1729, r1777;
}
{
mul.f16x2 r2596, r2593, r2559;
}
{
add.f16x2 r2599, r140, r2596;
}
{
add.f16x2 r2602, r1745, r1761;
}
{
mul.f16x2 r2605, r2602, r2563;
}
{
add.f16x2 r2608, r2599, r2605;
}
{
sub.f16x2 r2611, r1735, r1783;
}
{
mul.f16x2 r2614, r2611, r2561;
}
{
sub.f16x2 r2617, r1751, r1767;
}
{
mul.f16x2 r2620, r2617, r2565;
}
{
add.f16x2 r2623, r2614, r2620;
}
{
sub.f16x2 r2626, r2608, r2623;
}
{
add.f16x2 r2629, r1729, r1777;
}
{
mul.f16x2 r2632, r2629, r2559;
}
{
add.f16x2 r2635, r140, r2632;
}
{
add.f16x2 r2638, r1745, r1761;
}
{
mul.f16x2 r2641, r2638, r2563;
}
{
add.f16x2 r2644, r2635, r2641;
}
{
sub.f16x2 r2647, r1735, r1783;
}
{
mul.f16x2 r2650, r2647, r2561;
}
{
sub.f16x2 r2653, r1751, r1767;
}
{
mul.f16x2 r2656, r2653, r2565;
}
{
add.f16x2 r2659, r2650, r2656;
}
{
add.f16x2 r2662, r2644, r2659;
}
{
add.f16x2 r2665, r1729, r1777;
}
{
mul.f16x2 r2668, r2665, r2563;
}
{
add.f16x2 r2671, r140, r2668;
}
{
add.f16x2 r2674, r1745, r1761;
}
{
mul.f16x2 r2677, r2674, r2567;
}
{
add.f16x2 r2680, r2671, r2677;
}
{
sub.f16x2 r2683, r1735, r1783;
}
{
mul.f16x2 r2686, r2683, r2565;
}
{
sub.f16x2 r2689, r1751, r1767;
}
{
mul.f16x2 r2692, r2689, r2568;
}
{
add.f16x2 r2695, r2686, r2692;
}
{
sub.f16x2 r2698, r2680, r2695;
}
{
add.f16x2 r2701, r1729, r1777;
}
{
mul.f16x2 r2704, r2701, r2563;
}
{
add.f16x2 r2707, r140, r2704;
}
{
add.f16x2 r2710, r1745, r1761;
}
{
mul.f16x2 r2713, r2710, r2567;
}
{
add.f16x2 r2716, r2707, r2713;
}
{
sub.f16x2 r2719, r1735, r1783;
}
{
mul.f16x2 r2722, r2719, r2565;
}
{
sub.f16x2 r2725, r1751, r1767;
}
{
mul.f16x2 r2728, r2725, r2568;
}
{
add.f16x2 r2731, r2722, r2728;
}
{
add.f16x2 r2734, r2716, r2731;
}
{
add.f16x2 r2737, r1735, r1783;
}
{
mul.f16x2 r2740, r2737, r2559;
}
{
add.f16x2 r2743, r284, r2740;
}
{
add.f16x2 r2746, r1751, r1767;
}
{
mul.f16x2 r2749, r2746, r2563;
}
{
add.f16x2 r2752, r2743, r2749;
}
{
sub.f16x2 r2755, r1729, r1777;
}
{
mul.f16x2 r2758, r2755, r2561;
}
{
sub.f16x2 r2761, r1745, r1761;
}
{
mul.f16x2 r2764, r2761, r2565;
}
{
add.f16x2 r2767, r2758, r2764;
}
{
add.f16x2 r2770, r2752, r2767;
}
{
add.f16x2 r2773, r1735, r1783;
}
{
mul.f16x2 r2776, r2773, r2559;
}
{
add.f16x2 r2779, r284, r2776;
}
{
add.f16x2 r2782, r1751, r1767;
}
{
mul.f16x2 r2785, r2782, r2563;
}
{
add.f16x2 r2788, r2779, r2785;
}
{
sub.f16x2 r2791, r1729, r1777;
}
{
mul.f16x2 r2794, r2791, r2561;
}
{
sub.f16x2 r2797, r1745, r1761;
}
{
mul.f16x2 r2800, r2797, r2565;
}
{
add.f16x2 r2803, r2794, r2800;
}
{
sub.f16x2 r2806, r2788, r2803;
}
{
add.f16x2 r2809, r1735, r1783;
}
{
mul.f16x2 r2812, r2809, r2563;
}
{
add.f16x2 r2815, r284, r2812;
}
{
add.f16x2 r2818, r1751, r1767;
}
{
mul.f16x2 r2821, r2818, r2567;
}
{
add.f16x2 r2824, r2815, r2821;
}
{
sub.f16x2 r2827, r1729, r1777;
}
{
mul.f16x2 r2830, r2827, r2565;
}
{
sub.f16x2 r2833, r1745, r1761;
}
{
mul.f16x2 r2836, r2833, r2568;
}
{
add.f16x2 r2839, r2830, r2836;
}
{
add.f16x2 r2842, r2824, r2839;
}
{
add.f16x2 r2845, r1735, r1783;
}
{
mul.f16x2 r2848, r2845, r2563;
}
{
add.f16x2 r2851, r284, r2848;
}
{
add.f16x2 r2854, r1751, r1767;
}
{
mul.f16x2 r2857, r2854, r2567;
}
{
add.f16x2 r2860, r2851, r2857;
}
{
sub.f16x2 r2863, r1729, r1777;
}
{
mul.f16x2 r2866, r2863, r2565;
}
{
sub.f16x2 r2869, r1745, r1761;
}
{
mul.f16x2 r2872, r2869, r2568;
}
{
add.f16x2 r2875, r2866, r2872;
}
{
sub.f16x2 r2878, r2860, r2875;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2881, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2882, {low, high};
}
{
neg.f16x2 r2883, r2882;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r2885, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r2886, {low, high};
}
{
neg.f16x2 r2887, r2886;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2889, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2890, {low, high};
}
{
add.f16x2 r2891, r1793, r1841;
}
{
add.f16x2 r2894, r176, r2891;
}
{
add.f16x2 r2897, r1809, r1825;
}
{
add.f16x2 r2900, r2894, r2897;
}
{
add.f16x2 r2903, r1799, r1847;
}
{
add.f16x2 r2906, r320, r2903;
}
{
add.f16x2 r2909, r1815, r1831;
}
{
add.f16x2 r2912, r2906, r2909;
}
{
add.f16x2 r2915, r1793, r1841;
}
{
mul.f16x2 r2918, r2915, r2881;
}
{
add.f16x2 r2921, r176, r2918;
}
{
add.f16x2 r2924, r1809, r1825;
}
{
mul.f16x2 r2927, r2924, r2885;
}
{
add.f16x2 r2930, r2921, r2927;
}
{
sub.f16x2 r2933, r1799, r1847;
}
{
mul.f16x2 r2936, r2933, r2883;
}
{
sub.f16x2 r2939, r1815, r1831;
}
{
mul.f16x2 r2942, r2939, r2887;
}
{
add.f16x2 r2945, r2936, r2942;
}
{
sub.f16x2 r2948, r2930, r2945;
}
{
add.f16x2 r2951, r1793, r1841;
}
{
mul.f16x2 r2954, r2951, r2881;
}
{
add.f16x2 r2957, r176, r2954;
}
{
add.f16x2 r2960, r1809, r1825;
}
{
mul.f16x2 r2963, r2960, r2885;
}
{
add.f16x2 r2966, r2957, r2963;
}
{
sub.f16x2 r2969, r1799, r1847;
}
{
mul.f16x2 r2972, r2969, r2883;
}
{
sub.f16x2 r2975, r1815, r1831;
}
{
mul.f16x2 r2978, r2975, r2887;
}
{
add.f16x2 r2981, r2972, r2978;
}
{
add.f16x2 r2984, r2966, r2981;
}
{
add.f16x2 r2987, r1793, r1841;
}
{
mul.f16x2 r2990, r2987, r2885;
}
{
add.f16x2 r2993, r176, r2990;
}
{
add.f16x2 r2996, r1809, r1825;
}
{
mul.f16x2 r2999, r2996, r2889;
}
{
add.f16x2 r3002, r2993, r2999;
}
{
sub.f16x2 r3005, r1799, r1847;
}
{
mul.f16x2 r3008, r3005, r2887;
}
{
sub.f16x2 r3011, r1815, r1831;
}
{
mul.f16x2 r3014, r3011, r2890;
}
{
add.f16x2 r3017, r3008, r3014;
}
{
sub.f16x2 r3020, r3002, r3017;
}
{
add.f16x2 r3023, r1793, r1841;
}
{
mul.f16x2 r3026, r3023, r2885;
}
{
add.f16x2 r3029, r176, r3026;
}
{
add.f16x2 r3032, r1809, r1825;
}
{
mul.f16x2 r3035, r3032, r2889;
}
{
add.f16x2 r3038, r3029, r3035;
}
{
sub.f16x2 r3041, r1799, r1847;
}
{
mul.f16x2 r3044, r3041, r2887;
}
{
sub.f16x2 r3047, r1815, r1831;
}
{
mul.f16x2 r3050, r3047, r2890;
}
{
add.f16x2 r3053, r3044, r3050;
}
{
add.f16x2 r3056, r3038, r3053;
}
{
add.f16x2 r3059, r1799, r1847;
}
{
mul.f16x2 r3062, r3059, r2881;
}
{
add.f16x2 r3065, r320, r3062;
}
{
add.f16x2 r3068, r1815, r1831;
}
{
mul.f16x2 r3071, r3068, r2885;
}
{
add.f16x2 r3074, r3065, r3071;
}
{
sub.f16x2 r3077, r1793, r1841;
}
{
mul.f16x2 r3080, r3077, r2883;
}
{
sub.f16x2 r3083, r1809, r1825;
}
{
mul.f16x2 r3086, r3083, r2887;
}
{
add.f16x2 r3089, r3080, r3086;
}
{
add.f16x2 r3092, r3074, r3089;
}
{
add.f16x2 r3095, r1799, r1847;
}
{
mul.f16x2 r3098, r3095, r2881;
}
{
add.f16x2 r3101, r320, r3098;
}
{
add.f16x2 r3104, r1815, r1831;
}
{
mul.f16x2 r3107, r3104, r2885;
}
{
add.f16x2 r3110, r3101, r3107;
}
{
sub.f16x2 r3113, r1793, r1841;
}
{
mul.f16x2 r3116, r3113, r2883;
}
{
sub.f16x2 r3119, r1809, r1825;
}
{
mul.f16x2 r3122, r3119, r2887;
}
{
add.f16x2 r3125, r3116, r3122;
}
{
sub.f16x2 r3128, r3110, r3125;
}
{
add.f16x2 r3131, r1799, r1847;
}
{
mul.f16x2 r3134, r3131, r2885;
}
{
add.f16x2 r3137, r320, r3134;
}
{
add.f16x2 r3140, r1815, r1831;
}
{
mul.f16x2 r3143, r3140, r2889;
}
{
add.f16x2 r3146, r3137, r3143;
}
{
sub.f16x2 r3149, r1793, r1841;
}
{
mul.f16x2 r3152, r3149, r2887;
}
{
sub.f16x2 r3155, r1809, r1825;
}
{
mul.f16x2 r3158, r3155, r2890;
}
{
add.f16x2 r3161, r3152, r3158;
}
{
add.f16x2 r3164, r3146, r3161;
}
{
add.f16x2 r3167, r1799, r1847;
}
{
mul.f16x2 r3170, r3167, r2885;
}
{
add.f16x2 r3173, r320, r3170;
}
{
add.f16x2 r3176, r1815, r1831;
}
{
mul.f16x2 r3179, r3176, r2889;
}
{
add.f16x2 r3182, r3173, r3179;
}
{
sub.f16x2 r3185, r1793, r1841;
}
{
mul.f16x2 r3188, r3185, r2887;
}
{
sub.f16x2 r3191, r1809, r1825;
}
{
mul.f16x2 r3194, r3191, r2890;
}
{
add.f16x2 r3197, r3188, r3194;
}
{
sub.f16x2 r3200, r3182, r3197;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r3203, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r3204, {low, high};
}
{
neg.f16x2 r3205, r3204;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r3207, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r3208, {low, high};
}
{
neg.f16x2 r3209, r3208;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r3211, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r3212, {low, high};
}
{
add.f16x2 r3213, r1857, r1905;
}
{
add.f16x2 r3216, r104, r3213;
}
{
add.f16x2 r3219, r1873, r1889;
}
{
add.f16x2 r3222, r3216, r3219;
}
{
add.f16x2 r3225, r1863, r1911;
}
{
add.f16x2 r3228, r248, r3225;
}
{
add.f16x2 r3231, r1879, r1895;
}
{
add.f16x2 r3234, r3228, r3231;
}
{
add.f16x2 r3237, r1857, r1905;
}
{
mul.f16x2 r3240, r3237, r3203;
}
{
add.f16x2 r3243, r104, r3240;
}
{
add.f16x2 r3246, r1873, r1889;
}
{
mul.f16x2 r3249, r3246, r3207;
}
{
add.f16x2 r3252, r3243, r3249;
}
{
sub.f16x2 r3255, r1863, r1911;
}
{
mul.f16x2 r3258, r3255, r3205;
}
{
sub.f16x2 r3261, r1879, r1895;
}
{
mul.f16x2 r3264, r3261, r3209;
}
{
add.f16x2 r3267, r3258, r3264;
}
{
sub.f16x2 r3270, r3252, r3267;
}
{
add.f16x2 r3273, r1857, r1905;
}
{
mul.f16x2 r3276, r3273, r3203;
}
{
add.f16x2 r3279, r104, r3276;
}
{
add.f16x2 r3282, r1873, r1889;
}
{
mul.f16x2 r3285, r3282, r3207;
}
{
add.f16x2 r3288, r3279, r3285;
}
{
sub.f16x2 r3291, r1863, r1911;
}
{
mul.f16x2 r3294, r3291, r3205;
}
{
sub.f16x2 r3297, r1879, r1895;
}
{
mul.f16x2 r3300, r3297, r3209;
}
{
add.f16x2 r3303, r3294, r3300;
}
{
add.f16x2 r3306, r3288, r3303;
}
{
add.f16x2 r3309, r1857, r1905;
}
{
mul.f16x2 r3312, r3309, r3207;
}
{
add.f16x2 r3315, r104, r3312;
}
{
add.f16x2 r3318, r1873, r1889;
}
{
mul.f16x2 r3321, r3318, r3211;
}
{
add.f16x2 r3324, r3315, r3321;
}
{
sub.f16x2 r3327, r1863, r1911;
}
{
mul.f16x2 r3330, r3327, r3209;
}
{
sub.f16x2 r3333, r1879, r1895;
}
{
mul.f16x2 r3336, r3333, r3212;
}
{
add.f16x2 r3339, r3330, r3336;
}
{
sub.f16x2 r3342, r3324, r3339;
}
{
add.f16x2 r3345, r1857, r1905;
}
{
mul.f16x2 r3348, r3345, r3207;
}
{
add.f16x2 r3351, r104, r3348;
}
{
add.f16x2 r3354, r1873, r1889;
}
{
mul.f16x2 r3357, r3354, r3211;
}
{
add.f16x2 r3360, r3351, r3357;
}
{
sub.f16x2 r3363, r1863, r1911;
}
{
mul.f16x2 r3366, r3363, r3209;
}
{
sub.f16x2 r3369, r1879, r1895;
}
{
mul.f16x2 r3372, r3369, r3212;
}
{
add.f16x2 r3375, r3366, r3372;
}
{
add.f16x2 r3378, r3360, r3375;
}
{
add.f16x2 r3381, r1863, r1911;
}
{
mul.f16x2 r3384, r3381, r3203;
}
{
add.f16x2 r3387, r248, r3384;
}
{
add.f16x2 r3390, r1879, r1895;
}
{
mul.f16x2 r3393, r3390, r3207;
}
{
add.f16x2 r3396, r3387, r3393;
}
{
sub.f16x2 r3399, r1857, r1905;
}
{
mul.f16x2 r3402, r3399, r3205;
}
{
sub.f16x2 r3405, r1873, r1889;
}
{
mul.f16x2 r3408, r3405, r3209;
}
{
add.f16x2 r3411, r3402, r3408;
}
{
add.f16x2 r3414, r3396, r3411;
}
{
add.f16x2 r3417, r1863, r1911;
}
{
mul.f16x2 r3420, r3417, r3203;
}
{
add.f16x2 r3423, r248, r3420;
}
{
add.f16x2 r3426, r1879, r1895;
}
{
mul.f16x2 r3429, r3426, r3207;
}
{
add.f16x2 r3432, r3423, r3429;
}
{
sub.f16x2 r3435, r1857, r1905;
}
{
mul.f16x2 r3438, r3435, r3205;
}
{
sub.f16x2 r3441, r1873, r1889;
}
{
mul.f16x2 r3444, r3441, r3209;
}
{
add.f16x2 r3447, r3438, r3444;
}
{
sub.f16x2 r3450, r3432, r3447;
}
{
add.f16x2 r3453, r1863, r1911;
}
{
mul.f16x2 r3456, r3453, r3207;
}
{
add.f16x2 r3459, r248, r3456;
}
{
add.f16x2 r3462, r1879, r1895;
}
{
mul.f16x2 r3465, r3462, r3211;
}
{
add.f16x2 r3468, r3459, r3465;
}
{
sub.f16x2 r3471, r1857, r1905;
}
{
mul.f16x2 r3474, r3471, r3209;
}
{
sub.f16x2 r3477, r1873, r1889;
}
{
mul.f16x2 r3480, r3477, r3212;
}
{
add.f16x2 r3483, r3474, r3480;
}
{
add.f16x2 r3486, r3468, r3483;
}
{
add.f16x2 r3489, r1863, r1911;
}
{
mul.f16x2 r3492, r3489, r3207;
}
{
add.f16x2 r3495, r248, r3492;
}
{
add.f16x2 r3498, r1879, r1895;
}
{
mul.f16x2 r3501, r3498, r3211;
}
{
add.f16x2 r3504, r3495, r3501;
}
{
sub.f16x2 r3507, r1857, r1905;
}
{
mul.f16x2 r3510, r3507, r3209;
}
{
sub.f16x2 r3513, r1873, r1889;
}
{
mul.f16x2 r3516, r3513, r3212;
}
{
add.f16x2 r3519, r3510, r3516;
}
{
sub.f16x2 r3522, r3504, r3519;
}
mul.wide.u32 rd2, r10444, 274877907;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r10445, rd3;
mul.lo.s32 r10446, r10445, 125;
sub.s32 r10447, r10444, r10446;
cvt.rn.f32.u32 f597, r10447;
mul.f32 f598, f597, 0f3B03C498;
cos.approx.f32 f217, f598;
sin.approx.f32 f599, f598;
neg.f32 f218, f599;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f217;
cvt.rn.f16.f32 high, f218;
mov.b32 r3525, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3528, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3530, {high, high};
}
{
mul.f16x2 r3532, r2268, r3530;
}
{
fma.rn.f16x2 r3535, r2256, r3528, r3532;
}
{
mul.f16x2 r3539, r2256, r3530;
}
{
neg.f16x2 r3542, r3539;
}
{
fma.rn.f16x2 r3544, r2268, r3528, r3542;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3548, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3550, {high, high};
}
mov.f32 f533, 0fBF800000;
mov.f32 f534, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3552, {low, high};
}
{
mul.f16x2 r3553, r3550, r3552;
}
{
mul.f16x2 r3556, r3525, r3548;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3559, {high, low};
}
{
fma.rn.f16x2 r3561, r3553, r3559, r3556;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3561;
mov.b32 r3565, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3561;
mov.b32 r3567, {high, high};
}
{
mul.f16x2 r3569, r2590, r3567;
}
{
fma.rn.f16x2 r3572, r2578, r3565, r3569;
}
{
mul.f16x2 r3576, r2578, r3567;
}
{
neg.f16x2 r3579, r3576;
}
{
fma.rn.f16x2 r3581, r2590, r3565, r3579;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3585, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3587, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3589, {low, high};
}
{
mul.f16x2 r3590, r3587, r3589;
}
{
mul.f16x2 r3593, r3561, r3585;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3561;
mov.b32 r3596, {high, low};
}
{
fma.rn.f16x2 r3598, r3590, r3596, r3593;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3598;
mov.b32 r3602, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3598;
mov.b32 r3604, {high, high};
}
{
mul.f16x2 r3606, r2912, r3604;
}
{
fma.rn.f16x2 r3609, r2900, r3602, r3606;
}
{
mul.f16x2 r3613, r2900, r3604;
}
{
neg.f16x2 r3616, r3613;
}
{
fma.rn.f16x2 r3618, r2912, r3602, r3616;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3622, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3624, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3626, {low, high};
}
{
mul.f16x2 r3627, r3624, r3626;
}
{
mul.f16x2 r3630, r3598, r3622;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3598;
mov.b32 r3633, {high, low};
}
{
fma.rn.f16x2 r3635, r3627, r3633, r3630;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3635;
mov.b32 r3639, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3635;
mov.b32 r3641, {high, high};
}
{
mul.f16x2 r3643, r3234, r3641;
}
{
fma.rn.f16x2 r3646, r3222, r3639, r3643;
}
{
mul.f16x2 r3650, r3222, r3641;
}
{
neg.f16x2 r3653, r3650;
}
{
fma.rn.f16x2 r3655, r3234, r3639, r3653;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3659, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3661, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3663, {low, high};
}
{
mul.f16x2 r3664, r3661, r3663;
}
{
mul.f16x2 r3667, r3635, r3659;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3635;
mov.b32 r3670, {high, low};
}
{
fma.rn.f16x2 r3672, r3664, r3670, r3667;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3672;
mov.b32 r3676, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3672;
mov.b32 r3678, {high, high};
}
{
mul.f16x2 r3680, r2126, r3678;
}
{
fma.rn.f16x2 r3683, r1982, r3676, r3680;
}
{
mul.f16x2 r3687, r1982, r3678;
}
{
neg.f16x2 r3690, r3687;
}
{
fma.rn.f16x2 r3692, r2126, r3676, r3690;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3696, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3698, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3700, {low, high};
}
{
mul.f16x2 r3701, r3698, r3700;
}
{
mul.f16x2 r3704, r3672, r3696;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3672;
mov.b32 r3707, {high, low};
}
{
fma.rn.f16x2 r3709, r3701, r3707, r3704;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3709;
mov.b32 r3713, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3709;
mov.b32 r3715, {high, high};
}
{
mul.f16x2 r3717, r2448, r3715;
}
{
fma.rn.f16x2 r3720, r2304, r3713, r3717;
}
{
mul.f16x2 r3724, r2304, r3715;
}
{
neg.f16x2 r3727, r3724;
}
{
fma.rn.f16x2 r3729, r2448, r3713, r3727;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3733, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3735, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3737, {low, high};
}
{
mul.f16x2 r3738, r3735, r3737;
}
{
mul.f16x2 r3741, r3709, r3733;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3709;
mov.b32 r3744, {high, low};
}
{
fma.rn.f16x2 r3746, r3738, r3744, r3741;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3746;
mov.b32 r3750, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3746;
mov.b32 r3752, {high, high};
}
{
mul.f16x2 r3754, r2770, r3752;
}
{
fma.rn.f16x2 r3757, r2626, r3750, r3754;
}
{
mul.f16x2 r3761, r2626, r3752;
}
{
neg.f16x2 r3764, r3761;
}
{
fma.rn.f16x2 r3766, r2770, r3750, r3764;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3770, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3772, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3774, {low, high};
}
{
mul.f16x2 r3775, r3772, r3774;
}
{
mul.f16x2 r3778, r3746, r3770;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3746;
mov.b32 r3781, {high, low};
}
{
fma.rn.f16x2 r3783, r3775, r3781, r3778;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3783;
mov.b32 r3787, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3783;
mov.b32 r3789, {high, high};
}
{
mul.f16x2 r3791, r3092, r3789;
}
{
fma.rn.f16x2 r3794, r2948, r3787, r3791;
}
{
mul.f16x2 r3798, r2948, r3789;
}
{
neg.f16x2 r3801, r3798;
}
{
fma.rn.f16x2 r3803, r3092, r3787, r3801;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3807, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3809, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3811, {low, high};
}
{
mul.f16x2 r3812, r3809, r3811;
}
{
mul.f16x2 r3815, r3783, r3807;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3783;
mov.b32 r3818, {high, low};
}
{
fma.rn.f16x2 r3820, r3812, r3818, r3815;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3820;
mov.b32 r3824, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3820;
mov.b32 r3826, {high, high};
}
{
mul.f16x2 r3828, r3414, r3826;
}
{
fma.rn.f16x2 r3831, r3270, r3824, r3828;
}
{
mul.f16x2 r3835, r3270, r3826;
}
{
neg.f16x2 r3838, r3835;
}
{
fma.rn.f16x2 r3840, r3414, r3824, r3838;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3844, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3846, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3848, {low, high};
}
{
mul.f16x2 r3849, r3846, r3848;
}
{
mul.f16x2 r3852, r3820, r3844;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3820;
mov.b32 r3855, {high, low};
}
{
fma.rn.f16x2 r3857, r3849, r3855, r3852;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3857;
mov.b32 r3861, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3857;
mov.b32 r3863, {high, high};
}
{
mul.f16x2 r3865, r2198, r3863;
}
{
fma.rn.f16x2 r3868, r2054, r3861, r3865;
}
{
mul.f16x2 r3872, r2054, r3863;
}
{
neg.f16x2 r3875, r3872;
}
{
fma.rn.f16x2 r3877, r2198, r3861, r3875;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3881, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3883, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3885, {low, high};
}
{
mul.f16x2 r3886, r3883, r3885;
}
{
mul.f16x2 r3889, r3857, r3881;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3857;
mov.b32 r3892, {high, low};
}
{
fma.rn.f16x2 r3894, r3886, r3892, r3889;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3894;
mov.b32 r3898, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3894;
mov.b32 r3900, {high, high};
}
{
mul.f16x2 r3902, r2520, r3900;
}
{
fma.rn.f16x2 r3905, r2376, r3898, r3902;
}
{
mul.f16x2 r3909, r2376, r3900;
}
{
neg.f16x2 r3912, r3909;
}
{
fma.rn.f16x2 r3914, r2520, r3898, r3912;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3918, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3920, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3922, {low, high};
}
{
mul.f16x2 r3923, r3920, r3922;
}
{
mul.f16x2 r3926, r3894, r3918;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3894;
mov.b32 r3929, {high, low};
}
{
fma.rn.f16x2 r3931, r3923, r3929, r3926;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3931;
mov.b32 r3935, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3931;
mov.b32 r3937, {high, high};
}
{
mul.f16x2 r3939, r2842, r3937;
}
{
fma.rn.f16x2 r3942, r2698, r3935, r3939;
}
{
mul.f16x2 r3946, r2698, r3937;
}
{
neg.f16x2 r3949, r3946;
}
{
fma.rn.f16x2 r3951, r2842, r3935, r3949;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3955, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3957, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3959, {low, high};
}
{
mul.f16x2 r3960, r3957, r3959;
}
{
mul.f16x2 r3963, r3931, r3955;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3931;
mov.b32 r3966, {high, low};
}
{
fma.rn.f16x2 r3968, r3960, r3966, r3963;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3968;
mov.b32 r3972, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3968;
mov.b32 r3974, {high, high};
}
{
mul.f16x2 r3976, r3164, r3974;
}
{
fma.rn.f16x2 r3979, r3020, r3972, r3976;
}
{
mul.f16x2 r3983, r3020, r3974;
}
{
neg.f16x2 r3986, r3983;
}
{
fma.rn.f16x2 r3988, r3164, r3972, r3986;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3992, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3994, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3996, {low, high};
}
{
mul.f16x2 r3997, r3994, r3996;
}
{
mul.f16x2 r4000, r3968, r3992;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3968;
mov.b32 r4003, {high, low};
}
{
fma.rn.f16x2 r4005, r3997, r4003, r4000;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4005;
mov.b32 r4009, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4005;
mov.b32 r4011, {high, high};
}
{
mul.f16x2 r4013, r3486, r4011;
}
{
fma.rn.f16x2 r4016, r3342, r4009, r4013;
}
{
mul.f16x2 r4020, r3342, r4011;
}
{
neg.f16x2 r4023, r4020;
}
{
fma.rn.f16x2 r4025, r3486, r4009, r4023;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4029, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4031, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4033, {low, high};
}
{
mul.f16x2 r4034, r4031, r4033;
}
{
mul.f16x2 r4037, r4005, r4029;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4005;
mov.b32 r4040, {high, low};
}
{
fma.rn.f16x2 r4042, r4034, r4040, r4037;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4042;
mov.b32 r4046, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4042;
mov.b32 r4048, {high, high};
}
{
mul.f16x2 r4050, r2234, r4048;
}
{
fma.rn.f16x2 r4053, r2090, r4046, r4050;
}
{
mul.f16x2 r4057, r2090, r4048;
}
{
neg.f16x2 r4060, r4057;
}
{
fma.rn.f16x2 r4062, r2234, r4046, r4060;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4066, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4068, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4070, {low, high};
}
{
mul.f16x2 r4071, r4068, r4070;
}
{
mul.f16x2 r4074, r4042, r4066;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4042;
mov.b32 r4077, {high, low};
}
{
fma.rn.f16x2 r4079, r4071, r4077, r4074;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4079;
mov.b32 r4083, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4079;
mov.b32 r4085, {high, high};
}
{
mul.f16x2 r4087, r2556, r4085;
}
{
fma.rn.f16x2 r4090, r2412, r4083, r4087;
}
{
mul.f16x2 r4094, r2412, r4085;
}
{
neg.f16x2 r4097, r4094;
}
{
fma.rn.f16x2 r4099, r2556, r4083, r4097;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4103, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4105, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4107, {low, high};
}
{
mul.f16x2 r4108, r4105, r4107;
}
{
mul.f16x2 r4111, r4079, r4103;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4079;
mov.b32 r4114, {high, low};
}
{
fma.rn.f16x2 r4116, r4108, r4114, r4111;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4116;
mov.b32 r4120, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4116;
mov.b32 r4122, {high, high};
}
{
mul.f16x2 r4124, r2878, r4122;
}
{
fma.rn.f16x2 r4127, r2734, r4120, r4124;
}
{
mul.f16x2 r4131, r2734, r4122;
}
{
neg.f16x2 r4134, r4131;
}
{
fma.rn.f16x2 r4136, r2878, r4120, r4134;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4140, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4142, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4144, {low, high};
}
{
mul.f16x2 r4145, r4142, r4144;
}
{
mul.f16x2 r4148, r4116, r4140;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4116;
mov.b32 r4151, {high, low};
}
{
fma.rn.f16x2 r4153, r4145, r4151, r4148;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4153;
mov.b32 r4157, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4153;
mov.b32 r4159, {high, high};
}
{
mul.f16x2 r4161, r3200, r4159;
}
{
fma.rn.f16x2 r4164, r3056, r4157, r4161;
}
{
mul.f16x2 r4168, r3056, r4159;
}
{
neg.f16x2 r4171, r4168;
}
{
fma.rn.f16x2 r4173, r3200, r4157, r4171;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4177, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4179, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4181, {low, high};
}
{
mul.f16x2 r4182, r4179, r4181;
}
{
mul.f16x2 r4185, r4153, r4177;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4153;
mov.b32 r4188, {high, low};
}
{
fma.rn.f16x2 r4190, r4182, r4188, r4185;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4190;
mov.b32 r4194, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4190;
mov.b32 r4196, {high, high};
}
{
mul.f16x2 r4198, r3522, r4196;
}
{
fma.rn.f16x2 r4201, r3378, r4194, r4198;
}
{
mul.f16x2 r4205, r3378, r4196;
}
{
neg.f16x2 r4208, r4205;
}
{
fma.rn.f16x2 r4210, r3522, r4194, r4208;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4214, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4216, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4218, {low, high};
}
{
mul.f16x2 r4219, r4216, r4218;
}
{
mul.f16x2 r4222, r4190, r4214;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4190;
mov.b32 r4225, {high, low};
}
{
fma.rn.f16x2 r4227, r4219, r4225, r4222;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4227;
mov.b32 r4231, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4227;
mov.b32 r4233, {high, high};
}
{
mul.f16x2 r4235, r2162, r4233;
}
{
fma.rn.f16x2 r4238, r2018, r4231, r4235;
}
{
mul.f16x2 r4242, r2018, r4233;
}
{
neg.f16x2 r4245, r4242;
}
{
fma.rn.f16x2 r4247, r2162, r4231, r4245;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4251, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4253, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4255, {low, high};
}
{
mul.f16x2 r4256, r4253, r4255;
}
{
mul.f16x2 r4259, r4227, r4251;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4227;
mov.b32 r4262, {high, low};
}
{
fma.rn.f16x2 r4264, r4256, r4262, r4259;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4264;
mov.b32 r4268, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4264;
mov.b32 r4270, {high, high};
}
{
mul.f16x2 r4272, r2484, r4270;
}
{
fma.rn.f16x2 r4275, r2340, r4268, r4272;
}
{
mul.f16x2 r4279, r2340, r4270;
}
{
neg.f16x2 r4282, r4279;
}
{
fma.rn.f16x2 r4284, r2484, r4268, r4282;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4288, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4290, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4292, {low, high};
}
{
mul.f16x2 r4293, r4290, r4292;
}
{
mul.f16x2 r4296, r4264, r4288;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4264;
mov.b32 r4299, {high, low};
}
{
fma.rn.f16x2 r4301, r4293, r4299, r4296;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4301;
mov.b32 r4305, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4301;
mov.b32 r4307, {high, high};
}
{
mul.f16x2 r4309, r2806, r4307;
}
{
fma.rn.f16x2 r4312, r2662, r4305, r4309;
}
{
mul.f16x2 r4316, r2662, r4307;
}
{
neg.f16x2 r4319, r4316;
}
{
fma.rn.f16x2 r4321, r2806, r4305, r4319;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4325, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4327, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4329, {low, high};
}
{
mul.f16x2 r4330, r4327, r4329;
}
{
mul.f16x2 r4333, r4301, r4325;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4301;
mov.b32 r4336, {high, low};
}
{
fma.rn.f16x2 r4338, r4330, r4336, r4333;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4338;
mov.b32 r4342, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4338;
mov.b32 r4344, {high, high};
}
{
mul.f16x2 r4346, r3128, r4344;
}
{
fma.rn.f16x2 r4349, r2984, r4342, r4346;
}
{
mul.f16x2 r4353, r2984, r4344;
}
{
neg.f16x2 r4356, r4353;
}
{
fma.rn.f16x2 r4358, r3128, r4342, r4356;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4362, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4364, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4366, {low, high};
}
{
mul.f16x2 r4367, r4364, r4366;
}
{
mul.f16x2 r4370, r4338, r4362;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4338;
mov.b32 r4373, {high, low};
}
{
fma.rn.f16x2 r4375, r4367, r4373, r4370;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4375;
mov.b32 r4379, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4375;
mov.b32 r4381, {high, high};
}
{
mul.f16x2 r4383, r3450, r4381;
}
{
fma.rn.f16x2 r4386, r3306, r4379, r4383;
}
{
mul.f16x2 r4390, r3306, r4381;
}
{
neg.f16x2 r4393, r4390;
}
{
fma.rn.f16x2 r4395, r3450, r4379, r4393;
}
mad.lo.s32 r10448, r10445, 25000, r10443;
barrier.sync 0;
mad.lo.s32 r10449, r10447, 200, r10448;
st.shared.v2.f32 [r10449], {r1934, r1946};
st.shared.v2.f32 [r10449+8], {r3535, r3544};
st.shared.v2.f32 [r10449+16], {r3572, r3581};
st.shared.v2.f32 [r10449+24], {r3609, r3618};
st.shared.v2.f32 [r10449+32], {r3646, r3655};
st.shared.v2.f32 [r10449+40], {r3683, r3692};
st.shared.v2.f32 [r10449+48], {r3720, r3729};
st.shared.v2.f32 [r10449+56], {r3757, r3766};
st.shared.v2.f32 [r10449+64], {r3794, r3803};
st.shared.v2.f32 [r10449+72], {r3831, r3840};
st.shared.v2.f32 [r10449+80], {r3868, r3877};
st.shared.v2.f32 [r10449+88], {r3905, r3914};
st.shared.v2.f32 [r10449+96], {r3942, r3951};
st.shared.v2.f32 [r10449+104], {r3979, r3988};
st.shared.v2.f32 [r10449+112], {r4016, r4025};
st.shared.v2.f32 [r10449+120], {r4053, r4062};
st.shared.v2.f32 [r10449+128], {r4090, r4099};
st.shared.v2.f32 [r10449+136], {r4127, r4136};
st.shared.v2.f32 [r10449+144], {r4164, r4173};
st.shared.v2.f32 [r10449+152], {r4201, r4210};
st.shared.v2.f32 [r10449+160], {r4238, r4247};
st.shared.v2.f32 [r10449+168], {r4275, r4284};
st.shared.v2.f32 [r10449+176], {r4312, r4321};
st.shared.v2.f32 [r10449+184], {r4349, r4358};
st.shared.v2.f32 [r10449+192], {r4386, r4395};
barrier.sync 0;
mad.lo.s32 r10450, r10447, -192, r10449;
ld.shared.u32 r4430, [r10450];
ld.shared.u32 r4442, [r10450+4];
ld.shared.u32 r4752, [r10450+1000];
ld.shared.u32 r4764, [r10450+1004];
ld.shared.u32 r5074, [r10450+2000];
ld.shared.u32 r5086, [r10450+2004];
ld.shared.u32 r5396, [r10450+3000];
ld.shared.u32 r5408, [r10450+3004];
ld.shared.u32 r5718, [r10450+4000];
ld.shared.u32 r5730, [r10450+4004];
ld.shared.u32 r4427, [r10450+5000];
ld.shared.u32 r4439, [r10450+5004];
ld.shared.u32 r4749, [r10450+6000];
ld.shared.u32 r4761, [r10450+6004];
ld.shared.u32 r5071, [r10450+7000];
ld.shared.u32 r5083, [r10450+7004];
ld.shared.u32 r5393, [r10450+8000];
ld.shared.u32 r5405, [r10450+8004];
ld.shared.u32 r5715, [r10450+9000];
ld.shared.u32 r5727, [r10450+9004];
ld.shared.u32 r4433, [r10450+10000];
ld.shared.u32 r4445, [r10450+10004];
ld.shared.u32 r4755, [r10450+11000];
ld.shared.u32 r4767, [r10450+11004];
ld.shared.u32 r5077, [r10450+12000];
ld.shared.u32 r5089, [r10450+12004];
ld.shared.u32 r5399, [r10450+13000];
ld.shared.u32 r5411, [r10450+13004];
ld.shared.u32 r5721, [r10450+14000];
ld.shared.u32 r5733, [r10450+14004];
ld.shared.u32 r4434, [r10450+15000];
ld.shared.u32 r4446, [r10450+15004];
ld.shared.u32 r4756, [r10450+16000];
ld.shared.u32 r4768, [r10450+16004];
ld.shared.u32 r5078, [r10450+17000];
ld.shared.u32 r5090, [r10450+17004];
ld.shared.u32 r5400, [r10450+18000];
ld.shared.u32 r5412, [r10450+18004];
ld.shared.u32 r5722, [r10450+19000];
ld.shared.u32 r5734, [r10450+19004];
ld.shared.u32 r4428, [r10450+20000];
ld.shared.u32 r4440, [r10450+20004];
ld.shared.u32 r4750, [r10450+21000];
ld.shared.u32 r4762, [r10450+21004];
ld.shared.u32 r5072, [r10450+22000];
ld.shared.u32 r5084, [r10450+22004];
ld.shared.u32 r5394, [r10450+23000];
ld.shared.u32 r5406, [r10450+23004];
ld.shared.u32 r5716, [r10450+24000];
ld.shared.u32 r5728, [r10450+24004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r4416, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r4417, {low, high};
}
{
neg.f16x2 r4418, r4417;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r4420, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r4421, {low, high};
}
{
neg.f16x2 r4422, r4421;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r4424, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r4425, {low, high};
}
{
add.f16x2 r4426, r4427, r4428;
}
{
add.f16x2 r4429, r4430, r4426;
}
{
add.f16x2 r4432, r4433, r4434;
}
{
add.f16x2 r4435, r4429, r4432;
}
{
add.f16x2 r4438, r4439, r4440;
}
{
add.f16x2 r4441, r4442, r4438;
}
{
add.f16x2 r4444, r4445, r4446;
}
{
add.f16x2 r4447, r4441, r4444;
}
{
add.f16x2 r4450, r4427, r4428;
}
{
mul.f16x2 r4453, r4450, r4416;
}
{
add.f16x2 r4456, r4430, r4453;
}
{
add.f16x2 r4459, r4433, r4434;
}
{
mul.f16x2 r4462, r4459, r4420;
}
{
add.f16x2 r4465, r4456, r4462;
}
{
sub.f16x2 r4468, r4439, r4440;
}
{
mul.f16x2 r4471, r4468, r4418;
}
{
sub.f16x2 r4474, r4445, r4446;
}
{
mul.f16x2 r4477, r4474, r4422;
}
{
add.f16x2 r4480, r4471, r4477;
}
{
sub.f16x2 r4483, r4465, r4480;
}
{
add.f16x2 r4486, r4427, r4428;
}
{
mul.f16x2 r4489, r4486, r4416;
}
{
add.f16x2 r4492, r4430, r4489;
}
{
add.f16x2 r4495, r4433, r4434;
}
{
mul.f16x2 r4498, r4495, r4420;
}
{
add.f16x2 r4501, r4492, r4498;
}
{
sub.f16x2 r4504, r4439, r4440;
}
{
mul.f16x2 r4507, r4504, r4418;
}
{
sub.f16x2 r4510, r4445, r4446;
}
{
mul.f16x2 r4513, r4510, r4422;
}
{
add.f16x2 r4516, r4507, r4513;
}
{
add.f16x2 r4519, r4501, r4516;
}
{
add.f16x2 r4522, r4427, r4428;
}
{
mul.f16x2 r4525, r4522, r4420;
}
{
add.f16x2 r4528, r4430, r4525;
}
{
add.f16x2 r4531, r4433, r4434;
}
{
mul.f16x2 r4534, r4531, r4424;
}
{
add.f16x2 r4537, r4528, r4534;
}
{
sub.f16x2 r4540, r4439, r4440;
}
{
mul.f16x2 r4543, r4540, r4422;
}
{
sub.f16x2 r4546, r4445, r4446;
}
{
mul.f16x2 r4549, r4546, r4425;
}
{
add.f16x2 r4552, r4543, r4549;
}
{
sub.f16x2 r4555, r4537, r4552;
}
{
add.f16x2 r4558, r4427, r4428;
}
{
mul.f16x2 r4561, r4558, r4420;
}
{
add.f16x2 r4564, r4430, r4561;
}
{
add.f16x2 r4567, r4433, r4434;
}
{
mul.f16x2 r4570, r4567, r4424;
}
{
add.f16x2 r4573, r4564, r4570;
}
{
sub.f16x2 r4576, r4439, r4440;
}
{
mul.f16x2 r4579, r4576, r4422;
}
{
sub.f16x2 r4582, r4445, r4446;
}
{
mul.f16x2 r4585, r4582, r4425;
}
{
add.f16x2 r4588, r4579, r4585;
}
{
add.f16x2 r4591, r4573, r4588;
}
{
add.f16x2 r4594, r4439, r4440;
}
{
mul.f16x2 r4597, r4594, r4416;
}
{
add.f16x2 r4600, r4442, r4597;
}
{
add.f16x2 r4603, r4445, r4446;
}
{
mul.f16x2 r4606, r4603, r4420;
}
{
add.f16x2 r4609, r4600, r4606;
}
{
sub.f16x2 r4612, r4427, r4428;
}
{
mul.f16x2 r4615, r4612, r4418;
}
{
sub.f16x2 r4618, r4433, r4434;
}
{
mul.f16x2 r4621, r4618, r4422;
}
{
add.f16x2 r4624, r4615, r4621;
}
{
add.f16x2 r4627, r4609, r4624;
}
{
add.f16x2 r4630, r4439, r4440;
}
{
mul.f16x2 r4633, r4630, r4416;
}
{
add.f16x2 r4636, r4442, r4633;
}
{
add.f16x2 r4639, r4445, r4446;
}
{
mul.f16x2 r4642, r4639, r4420;
}
{
add.f16x2 r4645, r4636, r4642;
}
{
sub.f16x2 r4648, r4427, r4428;
}
{
mul.f16x2 r4651, r4648, r4418;
}
{
sub.f16x2 r4654, r4433, r4434;
}
{
mul.f16x2 r4657, r4654, r4422;
}
{
add.f16x2 r4660, r4651, r4657;
}
{
sub.f16x2 r4663, r4645, r4660;
}
{
add.f16x2 r4666, r4439, r4440;
}
{
mul.f16x2 r4669, r4666, r4420;
}
{
add.f16x2 r4672, r4442, r4669;
}
{
add.f16x2 r4675, r4445, r4446;
}
{
mul.f16x2 r4678, r4675, r4424;
}
{
add.f16x2 r4681, r4672, r4678;
}
{
sub.f16x2 r4684, r4427, r4428;
}
{
mul.f16x2 r4687, r4684, r4422;
}
{
sub.f16x2 r4690, r4433, r4434;
}
{
mul.f16x2 r4693, r4690, r4425;
}
{
add.f16x2 r4696, r4687, r4693;
}
{
add.f16x2 r4699, r4681, r4696;
}
{
add.f16x2 r4702, r4439, r4440;
}
{
mul.f16x2 r4705, r4702, r4420;
}
{
add.f16x2 r4708, r4442, r4705;
}
{
add.f16x2 r4711, r4445, r4446;
}
{
mul.f16x2 r4714, r4711, r4424;
}
{
add.f16x2 r4717, r4708, r4714;
}
{
sub.f16x2 r4720, r4427, r4428;
}
{
mul.f16x2 r4723, r4720, r4422;
}
{
sub.f16x2 r4726, r4433, r4434;
}
{
mul.f16x2 r4729, r4726, r4425;
}
{
add.f16x2 r4732, r4723, r4729;
}
{
sub.f16x2 r4735, r4717, r4732;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r4738, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r4739, {low, high};
}
{
neg.f16x2 r4740, r4739;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r4742, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r4743, {low, high};
}
{
neg.f16x2 r4744, r4743;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r4746, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r4747, {low, high};
}
{
add.f16x2 r4748, r4749, r4750;
}
{
add.f16x2 r4751, r4752, r4748;
}
{
add.f16x2 r4754, r4755, r4756;
}
{
add.f16x2 r4757, r4751, r4754;
}
{
add.f16x2 r4760, r4761, r4762;
}
{
add.f16x2 r4763, r4764, r4760;
}
{
add.f16x2 r4766, r4767, r4768;
}
{
add.f16x2 r4769, r4763, r4766;
}
{
add.f16x2 r4772, r4749, r4750;
}
{
mul.f16x2 r4775, r4772, r4738;
}
{
add.f16x2 r4778, r4752, r4775;
}
{
add.f16x2 r4781, r4755, r4756;
}
{
mul.f16x2 r4784, r4781, r4742;
}
{
add.f16x2 r4787, r4778, r4784;
}
{
sub.f16x2 r4790, r4761, r4762;
}
{
mul.f16x2 r4793, r4790, r4740;
}
{
sub.f16x2 r4796, r4767, r4768;
}
{
mul.f16x2 r4799, r4796, r4744;
}
{
add.f16x2 r4802, r4793, r4799;
}
{
sub.f16x2 r4805, r4787, r4802;
}
{
add.f16x2 r4808, r4749, r4750;
}
{
mul.f16x2 r4811, r4808, r4738;
}
{
add.f16x2 r4814, r4752, r4811;
}
{
add.f16x2 r4817, r4755, r4756;
}
{
mul.f16x2 r4820, r4817, r4742;
}
{
add.f16x2 r4823, r4814, r4820;
}
{
sub.f16x2 r4826, r4761, r4762;
}
{
mul.f16x2 r4829, r4826, r4740;
}
{
sub.f16x2 r4832, r4767, r4768;
}
{
mul.f16x2 r4835, r4832, r4744;
}
{
add.f16x2 r4838, r4829, r4835;
}
{
add.f16x2 r4841, r4823, r4838;
}
{
add.f16x2 r4844, r4749, r4750;
}
{
mul.f16x2 r4847, r4844, r4742;
}
{
add.f16x2 r4850, r4752, r4847;
}
{
add.f16x2 r4853, r4755, r4756;
}
{
mul.f16x2 r4856, r4853, r4746;
}
{
add.f16x2 r4859, r4850, r4856;
}
{
sub.f16x2 r4862, r4761, r4762;
}
{
mul.f16x2 r4865, r4862, r4744;
}
{
sub.f16x2 r4868, r4767, r4768;
}
{
mul.f16x2 r4871, r4868, r4747;
}
{
add.f16x2 r4874, r4865, r4871;
}
{
sub.f16x2 r4877, r4859, r4874;
}
{
add.f16x2 r4880, r4749, r4750;
}
{
mul.f16x2 r4883, r4880, r4742;
}
{
add.f16x2 r4886, r4752, r4883;
}
{
add.f16x2 r4889, r4755, r4756;
}
{
mul.f16x2 r4892, r4889, r4746;
}
{
add.f16x2 r4895, r4886, r4892;
}
{
sub.f16x2 r4898, r4761, r4762;
}
{
mul.f16x2 r4901, r4898, r4744;
}
{
sub.f16x2 r4904, r4767, r4768;
}
{
mul.f16x2 r4907, r4904, r4747;
}
{
add.f16x2 r4910, r4901, r4907;
}
{
add.f16x2 r4913, r4895, r4910;
}
{
add.f16x2 r4916, r4761, r4762;
}
{
mul.f16x2 r4919, r4916, r4738;
}
{
add.f16x2 r4922, r4764, r4919;
}
{
add.f16x2 r4925, r4767, r4768;
}
{
mul.f16x2 r4928, r4925, r4742;
}
{
add.f16x2 r4931, r4922, r4928;
}
{
sub.f16x2 r4934, r4749, r4750;
}
{
mul.f16x2 r4937, r4934, r4740;
}
{
sub.f16x2 r4940, r4755, r4756;
}
{
mul.f16x2 r4943, r4940, r4744;
}
{
add.f16x2 r4946, r4937, r4943;
}
{
add.f16x2 r4949, r4931, r4946;
}
{
add.f16x2 r4952, r4761, r4762;
}
{
mul.f16x2 r4955, r4952, r4738;
}
{
add.f16x2 r4958, r4764, r4955;
}
{
add.f16x2 r4961, r4767, r4768;
}
{
mul.f16x2 r4964, r4961, r4742;
}
{
add.f16x2 r4967, r4958, r4964;
}
{
sub.f16x2 r4970, r4749, r4750;
}
{
mul.f16x2 r4973, r4970, r4740;
}
{
sub.f16x2 r4976, r4755, r4756;
}
{
mul.f16x2 r4979, r4976, r4744;
}
{
add.f16x2 r4982, r4973, r4979;
}
{
sub.f16x2 r4985, r4967, r4982;
}
{
add.f16x2 r4988, r4761, r4762;
}
{
mul.f16x2 r4991, r4988, r4742;
}
{
add.f16x2 r4994, r4764, r4991;
}
{
add.f16x2 r4997, r4767, r4768;
}
{
mul.f16x2 r5000, r4997, r4746;
}
{
add.f16x2 r5003, r4994, r5000;
}
{
sub.f16x2 r5006, r4749, r4750;
}
{
mul.f16x2 r5009, r5006, r4744;
}
{
sub.f16x2 r5012, r4755, r4756;
}
{
mul.f16x2 r5015, r5012, r4747;
}
{
add.f16x2 r5018, r5009, r5015;
}
{
add.f16x2 r5021, r5003, r5018;
}
{
add.f16x2 r5024, r4761, r4762;
}
{
mul.f16x2 r5027, r5024, r4742;
}
{
add.f16x2 r5030, r4764, r5027;
}
{
add.f16x2 r5033, r4767, r4768;
}
{
mul.f16x2 r5036, r5033, r4746;
}
{
add.f16x2 r5039, r5030, r5036;
}
{
sub.f16x2 r5042, r4749, r4750;
}
{
mul.f16x2 r5045, r5042, r4744;
}
{
sub.f16x2 r5048, r4755, r4756;
}
{
mul.f16x2 r5051, r5048, r4747;
}
{
add.f16x2 r5054, r5045, r5051;
}
{
sub.f16x2 r5057, r5039, r5054;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5060, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5061, {low, high};
}
{
neg.f16x2 r5062, r5061;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r5064, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r5065, {low, high};
}
{
neg.f16x2 r5066, r5065;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5068, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5069, {low, high};
}
{
add.f16x2 r5070, r5071, r5072;
}
{
add.f16x2 r5073, r5074, r5070;
}
{
add.f16x2 r5076, r5077, r5078;
}
{
add.f16x2 r5079, r5073, r5076;
}
{
add.f16x2 r5082, r5083, r5084;
}
{
add.f16x2 r5085, r5086, r5082;
}
{
add.f16x2 r5088, r5089, r5090;
}
{
add.f16x2 r5091, r5085, r5088;
}
{
add.f16x2 r5094, r5071, r5072;
}
{
mul.f16x2 r5097, r5094, r5060;
}
{
add.f16x2 r5100, r5074, r5097;
}
{
add.f16x2 r5103, r5077, r5078;
}
{
mul.f16x2 r5106, r5103, r5064;
}
{
add.f16x2 r5109, r5100, r5106;
}
{
sub.f16x2 r5112, r5083, r5084;
}
{
mul.f16x2 r5115, r5112, r5062;
}
{
sub.f16x2 r5118, r5089, r5090;
}
{
mul.f16x2 r5121, r5118, r5066;
}
{
add.f16x2 r5124, r5115, r5121;
}
{
sub.f16x2 r5127, r5109, r5124;
}
{
add.f16x2 r5130, r5071, r5072;
}
{
mul.f16x2 r5133, r5130, r5060;
}
{
add.f16x2 r5136, r5074, r5133;
}
{
add.f16x2 r5139, r5077, r5078;
}
{
mul.f16x2 r5142, r5139, r5064;
}
{
add.f16x2 r5145, r5136, r5142;
}
{
sub.f16x2 r5148, r5083, r5084;
}
{
mul.f16x2 r5151, r5148, r5062;
}
{
sub.f16x2 r5154, r5089, r5090;
}
{
mul.f16x2 r5157, r5154, r5066;
}
{
add.f16x2 r5160, r5151, r5157;
}
{
add.f16x2 r5163, r5145, r5160;
}
{
add.f16x2 r5166, r5071, r5072;
}
{
mul.f16x2 r5169, r5166, r5064;
}
{
add.f16x2 r5172, r5074, r5169;
}
{
add.f16x2 r5175, r5077, r5078;
}
{
mul.f16x2 r5178, r5175, r5068;
}
{
add.f16x2 r5181, r5172, r5178;
}
{
sub.f16x2 r5184, r5083, r5084;
}
{
mul.f16x2 r5187, r5184, r5066;
}
{
sub.f16x2 r5190, r5089, r5090;
}
{
mul.f16x2 r5193, r5190, r5069;
}
{
add.f16x2 r5196, r5187, r5193;
}
{
sub.f16x2 r5199, r5181, r5196;
}
{
add.f16x2 r5202, r5071, r5072;
}
{
mul.f16x2 r5205, r5202, r5064;
}
{
add.f16x2 r5208, r5074, r5205;
}
{
add.f16x2 r5211, r5077, r5078;
}
{
mul.f16x2 r5214, r5211, r5068;
}
{
add.f16x2 r5217, r5208, r5214;
}
{
sub.f16x2 r5220, r5083, r5084;
}
{
mul.f16x2 r5223, r5220, r5066;
}
{
sub.f16x2 r5226, r5089, r5090;
}
{
mul.f16x2 r5229, r5226, r5069;
}
{
add.f16x2 r5232, r5223, r5229;
}
{
add.f16x2 r5235, r5217, r5232;
}
{
add.f16x2 r5238, r5083, r5084;
}
{
mul.f16x2 r5241, r5238, r5060;
}
{
add.f16x2 r5244, r5086, r5241;
}
{
add.f16x2 r5247, r5089, r5090;
}
{
mul.f16x2 r5250, r5247, r5064;
}
{
add.f16x2 r5253, r5244, r5250;
}
{
sub.f16x2 r5256, r5071, r5072;
}
{
mul.f16x2 r5259, r5256, r5062;
}
{
sub.f16x2 r5262, r5077, r5078;
}
{
mul.f16x2 r5265, r5262, r5066;
}
{
add.f16x2 r5268, r5259, r5265;
}
{
add.f16x2 r5271, r5253, r5268;
}
{
add.f16x2 r5274, r5083, r5084;
}
{
mul.f16x2 r5277, r5274, r5060;
}
{
add.f16x2 r5280, r5086, r5277;
}
{
add.f16x2 r5283, r5089, r5090;
}
{
mul.f16x2 r5286, r5283, r5064;
}
{
add.f16x2 r5289, r5280, r5286;
}
{
sub.f16x2 r5292, r5071, r5072;
}
{
mul.f16x2 r5295, r5292, r5062;
}
{
sub.f16x2 r5298, r5077, r5078;
}
{
mul.f16x2 r5301, r5298, r5066;
}
{
add.f16x2 r5304, r5295, r5301;
}
{
sub.f16x2 r5307, r5289, r5304;
}
{
add.f16x2 r5310, r5083, r5084;
}
{
mul.f16x2 r5313, r5310, r5064;
}
{
add.f16x2 r5316, r5086, r5313;
}
{
add.f16x2 r5319, r5089, r5090;
}
{
mul.f16x2 r5322, r5319, r5068;
}
{
add.f16x2 r5325, r5316, r5322;
}
{
sub.f16x2 r5328, r5071, r5072;
}
{
mul.f16x2 r5331, r5328, r5066;
}
{
sub.f16x2 r5334, r5077, r5078;
}
{
mul.f16x2 r5337, r5334, r5069;
}
{
add.f16x2 r5340, r5331, r5337;
}
{
add.f16x2 r5343, r5325, r5340;
}
{
add.f16x2 r5346, r5083, r5084;
}
{
mul.f16x2 r5349, r5346, r5064;
}
{
add.f16x2 r5352, r5086, r5349;
}
{
add.f16x2 r5355, r5089, r5090;
}
{
mul.f16x2 r5358, r5355, r5068;
}
{
add.f16x2 r5361, r5352, r5358;
}
{
sub.f16x2 r5364, r5071, r5072;
}
{
mul.f16x2 r5367, r5364, r5066;
}
{
sub.f16x2 r5370, r5077, r5078;
}
{
mul.f16x2 r5373, r5370, r5069;
}
{
add.f16x2 r5376, r5367, r5373;
}
{
sub.f16x2 r5379, r5361, r5376;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5382, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5383, {low, high};
}
{
neg.f16x2 r5384, r5383;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r5386, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r5387, {low, high};
}
{
neg.f16x2 r5388, r5387;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5390, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5391, {low, high};
}
{
add.f16x2 r5392, r5393, r5394;
}
{
add.f16x2 r5395, r5396, r5392;
}
{
add.f16x2 r5398, r5399, r5400;
}
{
add.f16x2 r5401, r5395, r5398;
}
{
add.f16x2 r5404, r5405, r5406;
}
{
add.f16x2 r5407, r5408, r5404;
}
{
add.f16x2 r5410, r5411, r5412;
}
{
add.f16x2 r5413, r5407, r5410;
}
{
add.f16x2 r5416, r5393, r5394;
}
{
mul.f16x2 r5419, r5416, r5382;
}
{
add.f16x2 r5422, r5396, r5419;
}
{
add.f16x2 r5425, r5399, r5400;
}
{
mul.f16x2 r5428, r5425, r5386;
}
{
add.f16x2 r5431, r5422, r5428;
}
{
sub.f16x2 r5434, r5405, r5406;
}
{
mul.f16x2 r5437, r5434, r5384;
}
{
sub.f16x2 r5440, r5411, r5412;
}
{
mul.f16x2 r5443, r5440, r5388;
}
{
add.f16x2 r5446, r5437, r5443;
}
{
sub.f16x2 r5449, r5431, r5446;
}
{
add.f16x2 r5452, r5393, r5394;
}
{
mul.f16x2 r5455, r5452, r5382;
}
{
add.f16x2 r5458, r5396, r5455;
}
{
add.f16x2 r5461, r5399, r5400;
}
{
mul.f16x2 r5464, r5461, r5386;
}
{
add.f16x2 r5467, r5458, r5464;
}
{
sub.f16x2 r5470, r5405, r5406;
}
{
mul.f16x2 r5473, r5470, r5384;
}
{
sub.f16x2 r5476, r5411, r5412;
}
{
mul.f16x2 r5479, r5476, r5388;
}
{
add.f16x2 r5482, r5473, r5479;
}
{
add.f16x2 r5485, r5467, r5482;
}
{
add.f16x2 r5488, r5393, r5394;
}
{
mul.f16x2 r5491, r5488, r5386;
}
{
add.f16x2 r5494, r5396, r5491;
}
{
add.f16x2 r5497, r5399, r5400;
}
{
mul.f16x2 r5500, r5497, r5390;
}
{
add.f16x2 r5503, r5494, r5500;
}
{
sub.f16x2 r5506, r5405, r5406;
}
{
mul.f16x2 r5509, r5506, r5388;
}
{
sub.f16x2 r5512, r5411, r5412;
}
{
mul.f16x2 r5515, r5512, r5391;
}
{
add.f16x2 r5518, r5509, r5515;
}
{
sub.f16x2 r5521, r5503, r5518;
}
{
add.f16x2 r5524, r5393, r5394;
}
{
mul.f16x2 r5527, r5524, r5386;
}
{
add.f16x2 r5530, r5396, r5527;
}
{
add.f16x2 r5533, r5399, r5400;
}
{
mul.f16x2 r5536, r5533, r5390;
}
{
add.f16x2 r5539, r5530, r5536;
}
{
sub.f16x2 r5542, r5405, r5406;
}
{
mul.f16x2 r5545, r5542, r5388;
}
{
sub.f16x2 r5548, r5411, r5412;
}
{
mul.f16x2 r5551, r5548, r5391;
}
{
add.f16x2 r5554, r5545, r5551;
}
{
add.f16x2 r5557, r5539, r5554;
}
{
add.f16x2 r5560, r5405, r5406;
}
{
mul.f16x2 r5563, r5560, r5382;
}
{
add.f16x2 r5566, r5408, r5563;
}
{
add.f16x2 r5569, r5411, r5412;
}
{
mul.f16x2 r5572, r5569, r5386;
}
{
add.f16x2 r5575, r5566, r5572;
}
{
sub.f16x2 r5578, r5393, r5394;
}
{
mul.f16x2 r5581, r5578, r5384;
}
{
sub.f16x2 r5584, r5399, r5400;
}
{
mul.f16x2 r5587, r5584, r5388;
}
{
add.f16x2 r5590, r5581, r5587;
}
{
add.f16x2 r5593, r5575, r5590;
}
{
add.f16x2 r5596, r5405, r5406;
}
{
mul.f16x2 r5599, r5596, r5382;
}
{
add.f16x2 r5602, r5408, r5599;
}
{
add.f16x2 r5605, r5411, r5412;
}
{
mul.f16x2 r5608, r5605, r5386;
}
{
add.f16x2 r5611, r5602, r5608;
}
{
sub.f16x2 r5614, r5393, r5394;
}
{
mul.f16x2 r5617, r5614, r5384;
}
{
sub.f16x2 r5620, r5399, r5400;
}
{
mul.f16x2 r5623, r5620, r5388;
}
{
add.f16x2 r5626, r5617, r5623;
}
{
sub.f16x2 r5629, r5611, r5626;
}
{
add.f16x2 r5632, r5405, r5406;
}
{
mul.f16x2 r5635, r5632, r5386;
}
{
add.f16x2 r5638, r5408, r5635;
}
{
add.f16x2 r5641, r5411, r5412;
}
{
mul.f16x2 r5644, r5641, r5390;
}
{
add.f16x2 r5647, r5638, r5644;
}
{
sub.f16x2 r5650, r5393, r5394;
}
{
mul.f16x2 r5653, r5650, r5388;
}
{
sub.f16x2 r5656, r5399, r5400;
}
{
mul.f16x2 r5659, r5656, r5391;
}
{
add.f16x2 r5662, r5653, r5659;
}
{
add.f16x2 r5665, r5647, r5662;
}
{
add.f16x2 r5668, r5405, r5406;
}
{
mul.f16x2 r5671, r5668, r5386;
}
{
add.f16x2 r5674, r5408, r5671;
}
{
add.f16x2 r5677, r5411, r5412;
}
{
mul.f16x2 r5680, r5677, r5390;
}
{
add.f16x2 r5683, r5674, r5680;
}
{
sub.f16x2 r5686, r5393, r5394;
}
{
mul.f16x2 r5689, r5686, r5388;
}
{
sub.f16x2 r5692, r5399, r5400;
}
{
mul.f16x2 r5695, r5692, r5391;
}
{
add.f16x2 r5698, r5689, r5695;
}
{
sub.f16x2 r5701, r5683, r5698;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5704, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5705, {low, high};
}
{
neg.f16x2 r5706, r5705;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r5708, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r5709, {low, high};
}
{
neg.f16x2 r5710, r5709;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5712, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5713, {low, high};
}
{
add.f16x2 r5714, r5715, r5716;
}
{
add.f16x2 r5717, r5718, r5714;
}
{
add.f16x2 r5720, r5721, r5722;
}
{
add.f16x2 r5723, r5717, r5720;
}
{
add.f16x2 r5726, r5727, r5728;
}
{
add.f16x2 r5729, r5730, r5726;
}
{
add.f16x2 r5732, r5733, r5734;
}
{
add.f16x2 r5735, r5729, r5732;
}
{
add.f16x2 r5738, r5715, r5716;
}
{
mul.f16x2 r5741, r5738, r5704;
}
{
add.f16x2 r5744, r5718, r5741;
}
{
add.f16x2 r5747, r5721, r5722;
}
{
mul.f16x2 r5750, r5747, r5708;
}
{
add.f16x2 r5753, r5744, r5750;
}
{
sub.f16x2 r5756, r5727, r5728;
}
{
mul.f16x2 r5759, r5756, r5706;
}
{
sub.f16x2 r5762, r5733, r5734;
}
{
mul.f16x2 r5765, r5762, r5710;
}
{
add.f16x2 r5768, r5759, r5765;
}
{
sub.f16x2 r5771, r5753, r5768;
}
{
add.f16x2 r5774, r5715, r5716;
}
{
mul.f16x2 r5777, r5774, r5704;
}
{
add.f16x2 r5780, r5718, r5777;
}
{
add.f16x2 r5783, r5721, r5722;
}
{
mul.f16x2 r5786, r5783, r5708;
}
{
add.f16x2 r5789, r5780, r5786;
}
{
sub.f16x2 r5792, r5727, r5728;
}
{
mul.f16x2 r5795, r5792, r5706;
}
{
sub.f16x2 r5798, r5733, r5734;
}
{
mul.f16x2 r5801, r5798, r5710;
}
{
add.f16x2 r5804, r5795, r5801;
}
{
add.f16x2 r5807, r5789, r5804;
}
{
add.f16x2 r5810, r5715, r5716;
}
{
mul.f16x2 r5813, r5810, r5708;
}
{
add.f16x2 r5816, r5718, r5813;
}
{
add.f16x2 r5819, r5721, r5722;
}
{
mul.f16x2 r5822, r5819, r5712;
}
{
add.f16x2 r5825, r5816, r5822;
}
{
sub.f16x2 r5828, r5727, r5728;
}
{
mul.f16x2 r5831, r5828, r5710;
}
{
sub.f16x2 r5834, r5733, r5734;
}
{
mul.f16x2 r5837, r5834, r5713;
}
{
add.f16x2 r5840, r5831, r5837;
}
{
sub.f16x2 r5843, r5825, r5840;
}
{
add.f16x2 r5846, r5715, r5716;
}
{
mul.f16x2 r5849, r5846, r5708;
}
{
add.f16x2 r5852, r5718, r5849;
}
{
add.f16x2 r5855, r5721, r5722;
}
{
mul.f16x2 r5858, r5855, r5712;
}
{
add.f16x2 r5861, r5852, r5858;
}
{
sub.f16x2 r5864, r5727, r5728;
}
{
mul.f16x2 r5867, r5864, r5710;
}
{
sub.f16x2 r5870, r5733, r5734;
}
{
mul.f16x2 r5873, r5870, r5713;
}
{
add.f16x2 r5876, r5867, r5873;
}
{
add.f16x2 r5879, r5861, r5876;
}
{
add.f16x2 r5882, r5727, r5728;
}
{
mul.f16x2 r5885, r5882, r5704;
}
{
add.f16x2 r5888, r5730, r5885;
}
{
add.f16x2 r5891, r5733, r5734;
}
{
mul.f16x2 r5894, r5891, r5708;
}
{
add.f16x2 r5897, r5888, r5894;
}
{
sub.f16x2 r5900, r5715, r5716;
}
{
mul.f16x2 r5903, r5900, r5706;
}
{
sub.f16x2 r5906, r5721, r5722;
}
{
mul.f16x2 r5909, r5906, r5710;
}
{
add.f16x2 r5912, r5903, r5909;
}
{
add.f16x2 r5915, r5897, r5912;
}
{
add.f16x2 r5918, r5727, r5728;
}
{
mul.f16x2 r5921, r5918, r5704;
}
{
add.f16x2 r5924, r5730, r5921;
}
{
add.f16x2 r5927, r5733, r5734;
}
{
mul.f16x2 r5930, r5927, r5708;
}
{
add.f16x2 r5933, r5924, r5930;
}
{
sub.f16x2 r5936, r5715, r5716;
}
{
mul.f16x2 r5939, r5936, r5706;
}
{
sub.f16x2 r5942, r5721, r5722;
}
{
mul.f16x2 r5945, r5942, r5710;
}
{
add.f16x2 r5948, r5939, r5945;
}
{
sub.f16x2 r5951, r5933, r5948;
}
{
add.f16x2 r5954, r5727, r5728;
}
{
mul.f16x2 r5957, r5954, r5708;
}
{
add.f16x2 r5960, r5730, r5957;
}
{
add.f16x2 r5963, r5733, r5734;
}
{
mul.f16x2 r5966, r5963, r5712;
}
{
add.f16x2 r5969, r5960, r5966;
}
{
sub.f16x2 r5972, r5715, r5716;
}
{
mul.f16x2 r5975, r5972, r5710;
}
{
sub.f16x2 r5978, r5721, r5722;
}
{
mul.f16x2 r5981, r5978, r5713;
}
{
add.f16x2 r5984, r5975, r5981;
}
{
add.f16x2 r5987, r5969, r5984;
}
{
add.f16x2 r5990, r5727, r5728;
}
{
mul.f16x2 r5993, r5990, r5708;
}
{
add.f16x2 r5996, r5730, r5993;
}
{
add.f16x2 r5999, r5733, r5734;
}
{
mul.f16x2 r6002, r5999, r5712;
}
{
add.f16x2 r6005, r5996, r6002;
}
{
sub.f16x2 r6008, r5715, r5716;
}
{
mul.f16x2 r6011, r6008, r5710;
}
{
sub.f16x2 r6014, r5721, r5722;
}
{
mul.f16x2 r6017, r6014, r5713;
}
{
add.f16x2 r6020, r6011, r6017;
}
{
sub.f16x2 r6023, r6005, r6020;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r6026, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f332;
cvt.rn.f16.f32 high, f332;
mov.b32 r6027, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f334;
cvt.rn.f16.f32 high, f334;
mov.b32 r6028, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f336;
cvt.rn.f16.f32 high, f336;
mov.b32 r6029, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f338;
cvt.rn.f16.f32 high, f338;
mov.b32 r6030, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r6031, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f342;
cvt.rn.f16.f32 high, f342;
mov.b32 r6032, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f344;
cvt.rn.f16.f32 high, f344;
mov.b32 r6033, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f350;
cvt.rn.f16.f32 high, f350;
mov.b32 r6036, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r6037, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f358;
cvt.rn.f16.f32 high, f358;
mov.b32 r6040, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f360;
cvt.rn.f16.f32 high, f360;
mov.b32 r6041, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f390;
cvt.rn.f16.f32 high, f390;
mov.b32 r6042, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f364;
cvt.rn.f16.f32 high, f364;
mov.b32 r6043, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f374;
cvt.rn.f16.f32 high, f374;
mov.b32 r6048, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r6049, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f390;
cvt.rn.f16.f32 high, f390;
mov.b32 r6056, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f392;
cvt.rn.f16.f32 high, f392;
mov.b32 r6057, {low, high};
}
{
mul.f16x2 r6074, r4805, r6026;
}
{
mul.f16x2 r6077, r4949, r6027;
}
{
sub.f16x2 r6080, r6074, r6077;
}
{
mul.f16x2 r6083, r4805, r6027;
}
{
fma.rn.f16x2 r6086, r4949, r6026, r6083;
}
{
mul.f16x2 r6090, r5127, r6028;
}
{
mul.f16x2 r6093, r5271, r6029;
}
{
sub.f16x2 r6096, r6090, r6093;
}
{
mul.f16x2 r6099, r5127, r6029;
}
{
fma.rn.f16x2 r6102, r5271, r6028, r6099;
}
{
mul.f16x2 r6106, r5449, r6030;
}
{
mul.f16x2 r6109, r5593, r6031;
}
{
sub.f16x2 r6112, r6106, r6109;
}
{
mul.f16x2 r6115, r5449, r6031;
}
{
fma.rn.f16x2 r6118, r5593, r6030, r6115;
}
{
mul.f16x2 r6122, r5771, r6032;
}
{
mul.f16x2 r6125, r5915, r6033;
}
{
sub.f16x2 r6128, r6122, r6125;
}
{
mul.f16x2 r6131, r5771, r6033;
}
{
fma.rn.f16x2 r6134, r5915, r6032, r6131;
}
{
mul.f16x2 r6138, r4877, r6028;
}
{
mul.f16x2 r6141, r5021, r6029;
}
{
sub.f16x2 r6144, r6138, r6141;
}
{
mul.f16x2 r6147, r4877, r6029;
}
{
fma.rn.f16x2 r6150, r5021, r6028, r6147;
}
{
mul.f16x2 r6154, r5199, r6032;
}
{
mul.f16x2 r6157, r5343, r6033;
}
{
sub.f16x2 r6160, r6154, r6157;
}
{
mul.f16x2 r6163, r5199, r6033;
}
{
fma.rn.f16x2 r6166, r5343, r6032, r6163;
}
{
mul.f16x2 r6170, r5521, r6036;
}
{
mul.f16x2 r6173, r5665, r6037;
}
{
sub.f16x2 r6176, r6170, r6173;
}
{
mul.f16x2 r6179, r5521, r6037;
}
{
fma.rn.f16x2 r6182, r5665, r6036, r6179;
}
{
mul.f16x2 r6186, r5843, r6040;
}
{
mul.f16x2 r6189, r5987, r6041;
}
{
sub.f16x2 r6192, r6186, r6189;
}
{
mul.f16x2 r6195, r5843, r6041;
}
{
fma.rn.f16x2 r6198, r5987, r6040, r6195;
}
{
mul.f16x2 r6202, r4913, r6030;
}
{
mul.f16x2 r6205, r5057, r6031;
}
{
sub.f16x2 r6208, r6202, r6205;
}
{
mul.f16x2 r6211, r4913, r6031;
}
{
fma.rn.f16x2 r6214, r5057, r6030, r6211;
}
{
mul.f16x2 r6218, r5235, r6036;
}
{
mul.f16x2 r6221, r5379, r6037;
}
{
sub.f16x2 r6224, r6218, r6221;
}
{
mul.f16x2 r6227, r5235, r6037;
}
{
fma.rn.f16x2 r6230, r5379, r6036, r6227;
}
{
mul.f16x2 r6234, r5557, r6042;
}
{
mul.f16x2 r6237, r5701, r6043;
}
{
sub.f16x2 r6240, r6234, r6237;
}
{
mul.f16x2 r6243, r5557, r6043;
}
{
fma.rn.f16x2 r6246, r5701, r6042, r6243;
}
{
mul.f16x2 r6250, r5879, r6048;
}
{
mul.f16x2 r6253, r6023, r6049;
}
{
sub.f16x2 r6256, r6250, r6253;
}
{
mul.f16x2 r6259, r5879, r6049;
}
{
fma.rn.f16x2 r6262, r6023, r6048, r6259;
}
{
mul.f16x2 r6266, r4841, r6032;
}
{
mul.f16x2 r6269, r4985, r6033;
}
{
sub.f16x2 r6272, r6266, r6269;
}
{
mul.f16x2 r6275, r4841, r6033;
}
{
fma.rn.f16x2 r6278, r4985, r6032, r6275;
}
{
mul.f16x2 r6282, r5163, r6040;
}
{
mul.f16x2 r6285, r5307, r6041;
}
{
sub.f16x2 r6288, r6282, r6285;
}
{
mul.f16x2 r6291, r5163, r6041;
}
{
fma.rn.f16x2 r6294, r5307, r6040, r6291;
}
{
mul.f16x2 r6298, r5485, r6048;
}
{
mul.f16x2 r6301, r5629, r6049;
}
{
sub.f16x2 r6304, r6298, r6301;
}
{
mul.f16x2 r6307, r5485, r6049;
}
{
fma.rn.f16x2 r6310, r5629, r6048, r6307;
}
{
mul.f16x2 r6314, r5807, r6056;
}
{
mul.f16x2 r6317, r5951, r6057;
}
{
sub.f16x2 r6320, r6314, r6317;
}
{
mul.f16x2 r6323, r5807, r6057;
}
{
fma.rn.f16x2 r6326, r5951, r6056, r6323;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6330, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6331, {low, high};
}
{
neg.f16x2 r6332, r6331;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r6334, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r6335, {low, high};
}
{
neg.f16x2 r6336, r6335;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6338, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6339, {low, high};
}
{
add.f16x2 r6340, r4757, r5723;
}
{
add.f16x2 r6343, r4435, r6340;
}
{
add.f16x2 r6346, r5079, r5401;
}
{
add.f16x2 r6349, r6343, r6346;
}
{
add.f16x2 r6352, r4769, r5735;
}
{
add.f16x2 r6355, r4447, r6352;
}
{
add.f16x2 r6358, r5091, r5413;
}
{
add.f16x2 r6361, r6355, r6358;
}
{
add.f16x2 r6364, r4757, r5723;
}
{
mul.f16x2 r6367, r6364, r6330;
}
{
add.f16x2 r6370, r4435, r6367;
}
{
add.f16x2 r6373, r5079, r5401;
}
{
mul.f16x2 r6376, r6373, r6334;
}
{
add.f16x2 r6379, r6370, r6376;
}
{
sub.f16x2 r6382, r4769, r5735;
}
{
mul.f16x2 r6385, r6382, r6332;
}
{
sub.f16x2 r6388, r5091, r5413;
}
{
mul.f16x2 r6391, r6388, r6336;
}
{
add.f16x2 r6394, r6385, r6391;
}
{
sub.f16x2 r6397, r6379, r6394;
}
{
add.f16x2 r6400, r4757, r5723;
}
{
mul.f16x2 r6403, r6400, r6330;
}
{
add.f16x2 r6406, r4435, r6403;
}
{
add.f16x2 r6409, r5079, r5401;
}
{
mul.f16x2 r6412, r6409, r6334;
}
{
add.f16x2 r6415, r6406, r6412;
}
{
sub.f16x2 r6418, r4769, r5735;
}
{
mul.f16x2 r6421, r6418, r6332;
}
{
sub.f16x2 r6424, r5091, r5413;
}
{
mul.f16x2 r6427, r6424, r6336;
}
{
add.f16x2 r6430, r6421, r6427;
}
{
add.f16x2 r6433, r6415, r6430;
}
{
add.f16x2 r6436, r4757, r5723;
}
{
mul.f16x2 r6439, r6436, r6334;
}
{
add.f16x2 r6442, r4435, r6439;
}
{
add.f16x2 r6445, r5079, r5401;
}
{
mul.f16x2 r6448, r6445, r6338;
}
{
add.f16x2 r6451, r6442, r6448;
}
{
sub.f16x2 r6454, r4769, r5735;
}
{
mul.f16x2 r6457, r6454, r6336;
}
{
sub.f16x2 r6460, r5091, r5413;
}
{
mul.f16x2 r6463, r6460, r6339;
}
{
add.f16x2 r6466, r6457, r6463;
}
{
sub.f16x2 r6469, r6451, r6466;
}
{
add.f16x2 r6472, r4757, r5723;
}
{
mul.f16x2 r6475, r6472, r6334;
}
{
add.f16x2 r6478, r4435, r6475;
}
{
add.f16x2 r6481, r5079, r5401;
}
{
mul.f16x2 r6484, r6481, r6338;
}
{
add.f16x2 r6487, r6478, r6484;
}
{
sub.f16x2 r6490, r4769, r5735;
}
{
mul.f16x2 r6493, r6490, r6336;
}
{
sub.f16x2 r6496, r5091, r5413;
}
{
mul.f16x2 r6499, r6496, r6339;
}
{
add.f16x2 r6502, r6493, r6499;
}
{
add.f16x2 r6505, r6487, r6502;
}
{
add.f16x2 r6508, r4769, r5735;
}
{
mul.f16x2 r6511, r6508, r6330;
}
{
add.f16x2 r6514, r4447, r6511;
}
{
add.f16x2 r6517, r5091, r5413;
}
{
mul.f16x2 r6520, r6517, r6334;
}
{
add.f16x2 r6523, r6514, r6520;
}
{
sub.f16x2 r6526, r4757, r5723;
}
{
mul.f16x2 r6529, r6526, r6332;
}
{
sub.f16x2 r6532, r5079, r5401;
}
{
mul.f16x2 r6535, r6532, r6336;
}
{
add.f16x2 r6538, r6529, r6535;
}
{
add.f16x2 r6541, r6523, r6538;
}
{
add.f16x2 r6544, r4769, r5735;
}
{
mul.f16x2 r6547, r6544, r6330;
}
{
add.f16x2 r6550, r4447, r6547;
}
{
add.f16x2 r6553, r5091, r5413;
}
{
mul.f16x2 r6556, r6553, r6334;
}
{
add.f16x2 r6559, r6550, r6556;
}
{
sub.f16x2 r6562, r4757, r5723;
}
{
mul.f16x2 r6565, r6562, r6332;
}
{
sub.f16x2 r6568, r5079, r5401;
}
{
mul.f16x2 r6571, r6568, r6336;
}
{
add.f16x2 r6574, r6565, r6571;
}
{
sub.f16x2 r6577, r6559, r6574;
}
{
add.f16x2 r6580, r4769, r5735;
}
{
mul.f16x2 r6583, r6580, r6334;
}
{
add.f16x2 r6586, r4447, r6583;
}
{
add.f16x2 r6589, r5091, r5413;
}
{
mul.f16x2 r6592, r6589, r6338;
}
{
add.f16x2 r6595, r6586, r6592;
}
{
sub.f16x2 r6598, r4757, r5723;
}
{
mul.f16x2 r6601, r6598, r6336;
}
{
sub.f16x2 r6604, r5079, r5401;
}
{
mul.f16x2 r6607, r6604, r6339;
}
{
add.f16x2 r6610, r6601, r6607;
}
{
add.f16x2 r6613, r6595, r6610;
}
{
add.f16x2 r6616, r4769, r5735;
}
{
mul.f16x2 r6619, r6616, r6334;
}
{
add.f16x2 r6622, r4447, r6619;
}
{
add.f16x2 r6625, r5091, r5413;
}
{
mul.f16x2 r6628, r6625, r6338;
}
{
add.f16x2 r6631, r6622, r6628;
}
{
sub.f16x2 r6634, r4757, r5723;
}
{
mul.f16x2 r6637, r6634, r6336;
}
{
sub.f16x2 r6640, r5079, r5401;
}
{
mul.f16x2 r6643, r6640, r6339;
}
{
add.f16x2 r6646, r6637, r6643;
}
{
sub.f16x2 r6649, r6631, r6646;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6652, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6653, {low, high};
}
{
neg.f16x2 r6654, r6653;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r6656, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r6657, {low, high};
}
{
neg.f16x2 r6658, r6657;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6660, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6661, {low, high};
}
{
add.f16x2 r6662, r6080, r6128;
}
{
add.f16x2 r6665, r4483, r6662;
}
{
add.f16x2 r6668, r6096, r6112;
}
{
add.f16x2 r6671, r6665, r6668;
}
{
add.f16x2 r6674, r6086, r6134;
}
{
add.f16x2 r6677, r4627, r6674;
}
{
add.f16x2 r6680, r6102, r6118;
}
{
add.f16x2 r6683, r6677, r6680;
}
{
add.f16x2 r6686, r6080, r6128;
}
{
mul.f16x2 r6689, r6686, r6652;
}
{
add.f16x2 r6692, r4483, r6689;
}
{
add.f16x2 r6695, r6096, r6112;
}
{
mul.f16x2 r6698, r6695, r6656;
}
{
add.f16x2 r6701, r6692, r6698;
}
{
sub.f16x2 r6704, r6086, r6134;
}
{
mul.f16x2 r6707, r6704, r6654;
}
{
sub.f16x2 r6710, r6102, r6118;
}
{
mul.f16x2 r6713, r6710, r6658;
}
{
add.f16x2 r6716, r6707, r6713;
}
{
sub.f16x2 r6719, r6701, r6716;
}
{
add.f16x2 r6722, r6080, r6128;
}
{
mul.f16x2 r6725, r6722, r6652;
}
{
add.f16x2 r6728, r4483, r6725;
}
{
add.f16x2 r6731, r6096, r6112;
}
{
mul.f16x2 r6734, r6731, r6656;
}
{
add.f16x2 r6737, r6728, r6734;
}
{
sub.f16x2 r6740, r6086, r6134;
}
{
mul.f16x2 r6743, r6740, r6654;
}
{
sub.f16x2 r6746, r6102, r6118;
}
{
mul.f16x2 r6749, r6746, r6658;
}
{
add.f16x2 r6752, r6743, r6749;
}
{
add.f16x2 r6755, r6737, r6752;
}
{
add.f16x2 r6758, r6080, r6128;
}
{
mul.f16x2 r6761, r6758, r6656;
}
{
add.f16x2 r6764, r4483, r6761;
}
{
add.f16x2 r6767, r6096, r6112;
}
{
mul.f16x2 r6770, r6767, r6660;
}
{
add.f16x2 r6773, r6764, r6770;
}
{
sub.f16x2 r6776, r6086, r6134;
}
{
mul.f16x2 r6779, r6776, r6658;
}
{
sub.f16x2 r6782, r6102, r6118;
}
{
mul.f16x2 r6785, r6782, r6661;
}
{
add.f16x2 r6788, r6779, r6785;
}
{
sub.f16x2 r6791, r6773, r6788;
}
{
add.f16x2 r6794, r6080, r6128;
}
{
mul.f16x2 r6797, r6794, r6656;
}
{
add.f16x2 r6800, r4483, r6797;
}
{
add.f16x2 r6803, r6096, r6112;
}
{
mul.f16x2 r6806, r6803, r6660;
}
{
add.f16x2 r6809, r6800, r6806;
}
{
sub.f16x2 r6812, r6086, r6134;
}
{
mul.f16x2 r6815, r6812, r6658;
}
{
sub.f16x2 r6818, r6102, r6118;
}
{
mul.f16x2 r6821, r6818, r6661;
}
{
add.f16x2 r6824, r6815, r6821;
}
{
add.f16x2 r6827, r6809, r6824;
}
{
add.f16x2 r6830, r6086, r6134;
}
{
mul.f16x2 r6833, r6830, r6652;
}
{
add.f16x2 r6836, r4627, r6833;
}
{
add.f16x2 r6839, r6102, r6118;
}
{
mul.f16x2 r6842, r6839, r6656;
}
{
add.f16x2 r6845, r6836, r6842;
}
{
sub.f16x2 r6848, r6080, r6128;
}
{
mul.f16x2 r6851, r6848, r6654;
}
{
sub.f16x2 r6854, r6096, r6112;
}
{
mul.f16x2 r6857, r6854, r6658;
}
{
add.f16x2 r6860, r6851, r6857;
}
{
add.f16x2 r6863, r6845, r6860;
}
{
add.f16x2 r6866, r6086, r6134;
}
{
mul.f16x2 r6869, r6866, r6652;
}
{
add.f16x2 r6872, r4627, r6869;
}
{
add.f16x2 r6875, r6102, r6118;
}
{
mul.f16x2 r6878, r6875, r6656;
}
{
add.f16x2 r6881, r6872, r6878;
}
{
sub.f16x2 r6884, r6080, r6128;
}
{
mul.f16x2 r6887, r6884, r6654;
}
{
sub.f16x2 r6890, r6096, r6112;
}
{
mul.f16x2 r6893, r6890, r6658;
}
{
add.f16x2 r6896, r6887, r6893;
}
{
sub.f16x2 r6899, r6881, r6896;
}
{
add.f16x2 r6902, r6086, r6134;
}
{
mul.f16x2 r6905, r6902, r6656;
}
{
add.f16x2 r6908, r4627, r6905;
}
{
add.f16x2 r6911, r6102, r6118;
}
{
mul.f16x2 r6914, r6911, r6660;
}
{
add.f16x2 r6917, r6908, r6914;
}
{
sub.f16x2 r6920, r6080, r6128;
}
{
mul.f16x2 r6923, r6920, r6658;
}
{
sub.f16x2 r6926, r6096, r6112;
}
{
mul.f16x2 r6929, r6926, r6661;
}
{
add.f16x2 r6932, r6923, r6929;
}
{
add.f16x2 r6935, r6917, r6932;
}
{
add.f16x2 r6938, r6086, r6134;
}
{
mul.f16x2 r6941, r6938, r6656;
}
{
add.f16x2 r6944, r4627, r6941;
}
{
add.f16x2 r6947, r6102, r6118;
}
{
mul.f16x2 r6950, r6947, r6660;
}
{
add.f16x2 r6953, r6944, r6950;
}
{
sub.f16x2 r6956, r6080, r6128;
}
{
mul.f16x2 r6959, r6956, r6658;
}
{
sub.f16x2 r6962, r6096, r6112;
}
{
mul.f16x2 r6965, r6962, r6661;
}
{
add.f16x2 r6968, r6959, r6965;
}
{
sub.f16x2 r6971, r6953, r6968;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6974, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6975, {low, high};
}
{
neg.f16x2 r6976, r6975;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r6978, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r6979, {low, high};
}
{
neg.f16x2 r6980, r6979;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6982, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6983, {low, high};
}
{
add.f16x2 r6984, r6144, r6192;
}
{
add.f16x2 r6987, r4555, r6984;
}
{
add.f16x2 r6990, r6160, r6176;
}
{
add.f16x2 r6993, r6987, r6990;
}
{
add.f16x2 r6996, r6150, r6198;
}
{
add.f16x2 r6999, r4699, r6996;
}
{
add.f16x2 r7002, r6166, r6182;
}
{
add.f16x2 r7005, r6999, r7002;
}
{
add.f16x2 r7008, r6144, r6192;
}
{
mul.f16x2 r7011, r7008, r6974;
}
{
add.f16x2 r7014, r4555, r7011;
}
{
add.f16x2 r7017, r6160, r6176;
}
{
mul.f16x2 r7020, r7017, r6978;
}
{
add.f16x2 r7023, r7014, r7020;
}
{
sub.f16x2 r7026, r6150, r6198;
}
{
mul.f16x2 r7029, r7026, r6976;
}
{
sub.f16x2 r7032, r6166, r6182;
}
{
mul.f16x2 r7035, r7032, r6980;
}
{
add.f16x2 r7038, r7029, r7035;
}
{
sub.f16x2 r7041, r7023, r7038;
}
{
add.f16x2 r7044, r6144, r6192;
}
{
mul.f16x2 r7047, r7044, r6974;
}
{
add.f16x2 r7050, r4555, r7047;
}
{
add.f16x2 r7053, r6160, r6176;
}
{
mul.f16x2 r7056, r7053, r6978;
}
{
add.f16x2 r7059, r7050, r7056;
}
{
sub.f16x2 r7062, r6150, r6198;
}
{
mul.f16x2 r7065, r7062, r6976;
}
{
sub.f16x2 r7068, r6166, r6182;
}
{
mul.f16x2 r7071, r7068, r6980;
}
{
add.f16x2 r7074, r7065, r7071;
}
{
add.f16x2 r7077, r7059, r7074;
}
{
add.f16x2 r7080, r6144, r6192;
}
{
mul.f16x2 r7083, r7080, r6978;
}
{
add.f16x2 r7086, r4555, r7083;
}
{
add.f16x2 r7089, r6160, r6176;
}
{
mul.f16x2 r7092, r7089, r6982;
}
{
add.f16x2 r7095, r7086, r7092;
}
{
sub.f16x2 r7098, r6150, r6198;
}
{
mul.f16x2 r7101, r7098, r6980;
}
{
sub.f16x2 r7104, r6166, r6182;
}
{
mul.f16x2 r7107, r7104, r6983;
}
{
add.f16x2 r7110, r7101, r7107;
}
{
sub.f16x2 r7113, r7095, r7110;
}
{
add.f16x2 r7116, r6144, r6192;
}
{
mul.f16x2 r7119, r7116, r6978;
}
{
add.f16x2 r7122, r4555, r7119;
}
{
add.f16x2 r7125, r6160, r6176;
}
{
mul.f16x2 r7128, r7125, r6982;
}
{
add.f16x2 r7131, r7122, r7128;
}
{
sub.f16x2 r7134, r6150, r6198;
}
{
mul.f16x2 r7137, r7134, r6980;
}
{
sub.f16x2 r7140, r6166, r6182;
}
{
mul.f16x2 r7143, r7140, r6983;
}
{
add.f16x2 r7146, r7137, r7143;
}
{
add.f16x2 r7149, r7131, r7146;
}
{
add.f16x2 r7152, r6150, r6198;
}
{
mul.f16x2 r7155, r7152, r6974;
}
{
add.f16x2 r7158, r4699, r7155;
}
{
add.f16x2 r7161, r6166, r6182;
}
{
mul.f16x2 r7164, r7161, r6978;
}
{
add.f16x2 r7167, r7158, r7164;
}
{
sub.f16x2 r7170, r6144, r6192;
}
{
mul.f16x2 r7173, r7170, r6976;
}
{
sub.f16x2 r7176, r6160, r6176;
}
{
mul.f16x2 r7179, r7176, r6980;
}
{
add.f16x2 r7182, r7173, r7179;
}
{
add.f16x2 r7185, r7167, r7182;
}
{
add.f16x2 r7188, r6150, r6198;
}
{
mul.f16x2 r7191, r7188, r6974;
}
{
add.f16x2 r7194, r4699, r7191;
}
{
add.f16x2 r7197, r6166, r6182;
}
{
mul.f16x2 r7200, r7197, r6978;
}
{
add.f16x2 r7203, r7194, r7200;
}
{
sub.f16x2 r7206, r6144, r6192;
}
{
mul.f16x2 r7209, r7206, r6976;
}
{
sub.f16x2 r7212, r6160, r6176;
}
{
mul.f16x2 r7215, r7212, r6980;
}
{
add.f16x2 r7218, r7209, r7215;
}
{
sub.f16x2 r7221, r7203, r7218;
}
{
add.f16x2 r7224, r6150, r6198;
}
{
mul.f16x2 r7227, r7224, r6978;
}
{
add.f16x2 r7230, r4699, r7227;
}
{
add.f16x2 r7233, r6166, r6182;
}
{
mul.f16x2 r7236, r7233, r6982;
}
{
add.f16x2 r7239, r7230, r7236;
}
{
sub.f16x2 r7242, r6144, r6192;
}
{
mul.f16x2 r7245, r7242, r6980;
}
{
sub.f16x2 r7248, r6160, r6176;
}
{
mul.f16x2 r7251, r7248, r6983;
}
{
add.f16x2 r7254, r7245, r7251;
}
{
add.f16x2 r7257, r7239, r7254;
}
{
add.f16x2 r7260, r6150, r6198;
}
{
mul.f16x2 r7263, r7260, r6978;
}
{
add.f16x2 r7266, r4699, r7263;
}
{
add.f16x2 r7269, r6166, r6182;
}
{
mul.f16x2 r7272, r7269, r6982;
}
{
add.f16x2 r7275, r7266, r7272;
}
{
sub.f16x2 r7278, r6144, r6192;
}
{
mul.f16x2 r7281, r7278, r6980;
}
{
sub.f16x2 r7284, r6160, r6176;
}
{
mul.f16x2 r7287, r7284, r6983;
}
{
add.f16x2 r7290, r7281, r7287;
}
{
sub.f16x2 r7293, r7275, r7290;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r7296, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r7297, {low, high};
}
{
neg.f16x2 r7298, r7297;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r7300, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r7301, {low, high};
}
{
neg.f16x2 r7302, r7301;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r7304, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r7305, {low, high};
}
{
add.f16x2 r7306, r6208, r6256;
}
{
add.f16x2 r7309, r4591, r7306;
}
{
add.f16x2 r7312, r6224, r6240;
}
{
add.f16x2 r7315, r7309, r7312;
}
{
add.f16x2 r7318, r6214, r6262;
}
{
add.f16x2 r7321, r4735, r7318;
}
{
add.f16x2 r7324, r6230, r6246;
}
{
add.f16x2 r7327, r7321, r7324;
}
{
add.f16x2 r7330, r6208, r6256;
}
{
mul.f16x2 r7333, r7330, r7296;
}
{
add.f16x2 r7336, r4591, r7333;
}
{
add.f16x2 r7339, r6224, r6240;
}
{
mul.f16x2 r7342, r7339, r7300;
}
{
add.f16x2 r7345, r7336, r7342;
}
{
sub.f16x2 r7348, r6214, r6262;
}
{
mul.f16x2 r7351, r7348, r7298;
}
{
sub.f16x2 r7354, r6230, r6246;
}
{
mul.f16x2 r7357, r7354, r7302;
}
{
add.f16x2 r7360, r7351, r7357;
}
{
sub.f16x2 r7363, r7345, r7360;
}
{
add.f16x2 r7366, r6208, r6256;
}
{
mul.f16x2 r7369, r7366, r7296;
}
{
add.f16x2 r7372, r4591, r7369;
}
{
add.f16x2 r7375, r6224, r6240;
}
{
mul.f16x2 r7378, r7375, r7300;
}
{
add.f16x2 r7381, r7372, r7378;
}
{
sub.f16x2 r7384, r6214, r6262;
}
{
mul.f16x2 r7387, r7384, r7298;
}
{
sub.f16x2 r7390, r6230, r6246;
}
{
mul.f16x2 r7393, r7390, r7302;
}
{
add.f16x2 r7396, r7387, r7393;
}
{
add.f16x2 r7399, r7381, r7396;
}
{
add.f16x2 r7402, r6208, r6256;
}
{
mul.f16x2 r7405, r7402, r7300;
}
{
add.f16x2 r7408, r4591, r7405;
}
{
add.f16x2 r7411, r6224, r6240;
}
{
mul.f16x2 r7414, r7411, r7304;
}
{
add.f16x2 r7417, r7408, r7414;
}
{
sub.f16x2 r7420, r6214, r6262;
}
{
mul.f16x2 r7423, r7420, r7302;
}
{
sub.f16x2 r7426, r6230, r6246;
}
{
mul.f16x2 r7429, r7426, r7305;
}
{
add.f16x2 r7432, r7423, r7429;
}
{
sub.f16x2 r7435, r7417, r7432;
}
{
add.f16x2 r7438, r6208, r6256;
}
{
mul.f16x2 r7441, r7438, r7300;
}
{
add.f16x2 r7444, r4591, r7441;
}
{
add.f16x2 r7447, r6224, r6240;
}
{
mul.f16x2 r7450, r7447, r7304;
}
{
add.f16x2 r7453, r7444, r7450;
}
{
sub.f16x2 r7456, r6214, r6262;
}
{
mul.f16x2 r7459, r7456, r7302;
}
{
sub.f16x2 r7462, r6230, r6246;
}
{
mul.f16x2 r7465, r7462, r7305;
}
{
add.f16x2 r7468, r7459, r7465;
}
{
add.f16x2 r7471, r7453, r7468;
}
{
add.f16x2 r7474, r6214, r6262;
}
{
mul.f16x2 r7477, r7474, r7296;
}
{
add.f16x2 r7480, r4735, r7477;
}
{
add.f16x2 r7483, r6230, r6246;
}
{
mul.f16x2 r7486, r7483, r7300;
}
{
add.f16x2 r7489, r7480, r7486;
}
{
sub.f16x2 r7492, r6208, r6256;
}
{
mul.f16x2 r7495, r7492, r7298;
}
{
sub.f16x2 r7498, r6224, r6240;
}
{
mul.f16x2 r7501, r7498, r7302;
}
{
add.f16x2 r7504, r7495, r7501;
}
{
add.f16x2 r7507, r7489, r7504;
}
{
add.f16x2 r7510, r6214, r6262;
}
{
mul.f16x2 r7513, r7510, r7296;
}
{
add.f16x2 r7516, r4735, r7513;
}
{
add.f16x2 r7519, r6230, r6246;
}
{
mul.f16x2 r7522, r7519, r7300;
}
{
add.f16x2 r7525, r7516, r7522;
}
{
sub.f16x2 r7528, r6208, r6256;
}
{
mul.f16x2 r7531, r7528, r7298;
}
{
sub.f16x2 r7534, r6224, r6240;
}
{
mul.f16x2 r7537, r7534, r7302;
}
{
add.f16x2 r7540, r7531, r7537;
}
{
sub.f16x2 r7543, r7525, r7540;
}
{
add.f16x2 r7546, r6214, r6262;
}
{
mul.f16x2 r7549, r7546, r7300;
}
{
add.f16x2 r7552, r4735, r7549;
}
{
add.f16x2 r7555, r6230, r6246;
}
{
mul.f16x2 r7558, r7555, r7304;
}
{
add.f16x2 r7561, r7552, r7558;
}
{
sub.f16x2 r7564, r6208, r6256;
}
{
mul.f16x2 r7567, r7564, r7302;
}
{
sub.f16x2 r7570, r6224, r6240;
}
{
mul.f16x2 r7573, r7570, r7305;
}
{
add.f16x2 r7576, r7567, r7573;
}
{
add.f16x2 r7579, r7561, r7576;
}
{
add.f16x2 r7582, r6214, r6262;
}
{
mul.f16x2 r7585, r7582, r7300;
}
{
add.f16x2 r7588, r4735, r7585;
}
{
add.f16x2 r7591, r6230, r6246;
}
{
mul.f16x2 r7594, r7591, r7304;
}
{
add.f16x2 r7597, r7588, r7594;
}
{
sub.f16x2 r7600, r6208, r6256;
}
{
mul.f16x2 r7603, r7600, r7302;
}
{
sub.f16x2 r7606, r6224, r6240;
}
{
mul.f16x2 r7609, r7606, r7305;
}
{
add.f16x2 r7612, r7603, r7609;
}
{
sub.f16x2 r7615, r7597, r7612;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r7618, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r7619, {low, high};
}
{
neg.f16x2 r7620, r7619;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r7622, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r7623, {low, high};
}
{
neg.f16x2 r7624, r7623;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r7626, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r7627, {low, high};
}
{
add.f16x2 r7628, r6272, r6320;
}
{
add.f16x2 r7631, r4519, r7628;
}
{
add.f16x2 r7634, r6288, r6304;
}
{
add.f16x2 r7637, r7631, r7634;
}
{
add.f16x2 r7640, r6278, r6326;
}
{
add.f16x2 r7643, r4663, r7640;
}
{
add.f16x2 r7646, r6294, r6310;
}
{
add.f16x2 r7649, r7643, r7646;
}
{
add.f16x2 r7652, r6272, r6320;
}
{
mul.f16x2 r7655, r7652, r7618;
}
{
add.f16x2 r7658, r4519, r7655;
}
{
add.f16x2 r7661, r6288, r6304;
}
{
mul.f16x2 r7664, r7661, r7622;
}
{
add.f16x2 r7667, r7658, r7664;
}
{
sub.f16x2 r7670, r6278, r6326;
}
{
mul.f16x2 r7673, r7670, r7620;
}
{
sub.f16x2 r7676, r6294, r6310;
}
{
mul.f16x2 r7679, r7676, r7624;
}
{
add.f16x2 r7682, r7673, r7679;
}
{
sub.f16x2 r7685, r7667, r7682;
}
{
add.f16x2 r7688, r6272, r6320;
}
{
mul.f16x2 r7691, r7688, r7618;
}
{
add.f16x2 r7694, r4519, r7691;
}
{
add.f16x2 r7697, r6288, r6304;
}
{
mul.f16x2 r7700, r7697, r7622;
}
{
add.f16x2 r7703, r7694, r7700;
}
{
sub.f16x2 r7706, r6278, r6326;
}
{
mul.f16x2 r7709, r7706, r7620;
}
{
sub.f16x2 r7712, r6294, r6310;
}
{
mul.f16x2 r7715, r7712, r7624;
}
{
add.f16x2 r7718, r7709, r7715;
}
{
add.f16x2 r7721, r7703, r7718;
}
{
add.f16x2 r7724, r6272, r6320;
}
{
mul.f16x2 r7727, r7724, r7622;
}
{
add.f16x2 r7730, r4519, r7727;
}
{
add.f16x2 r7733, r6288, r6304;
}
{
mul.f16x2 r7736, r7733, r7626;
}
{
add.f16x2 r7739, r7730, r7736;
}
{
sub.f16x2 r7742, r6278, r6326;
}
{
mul.f16x2 r7745, r7742, r7624;
}
{
sub.f16x2 r7748, r6294, r6310;
}
{
mul.f16x2 r7751, r7748, r7627;
}
{
add.f16x2 r7754, r7745, r7751;
}
{
sub.f16x2 r7757, r7739, r7754;
}
{
add.f16x2 r7760, r6272, r6320;
}
{
mul.f16x2 r7763, r7760, r7622;
}
{
add.f16x2 r7766, r4519, r7763;
}
{
add.f16x2 r7769, r6288, r6304;
}
{
mul.f16x2 r7772, r7769, r7626;
}
{
add.f16x2 r7775, r7766, r7772;
}
{
sub.f16x2 r7778, r6278, r6326;
}
{
mul.f16x2 r7781, r7778, r7624;
}
{
sub.f16x2 r7784, r6294, r6310;
}
{
mul.f16x2 r7787, r7784, r7627;
}
{
add.f16x2 r7790, r7781, r7787;
}
{
add.f16x2 r7793, r7775, r7790;
}
{
add.f16x2 r7796, r6278, r6326;
}
{
mul.f16x2 r7799, r7796, r7618;
}
{
add.f16x2 r7802, r4663, r7799;
}
{
add.f16x2 r7805, r6294, r6310;
}
{
mul.f16x2 r7808, r7805, r7622;
}
{
add.f16x2 r7811, r7802, r7808;
}
{
sub.f16x2 r7814, r6272, r6320;
}
{
mul.f16x2 r7817, r7814, r7620;
}
{
sub.f16x2 r7820, r6288, r6304;
}
{
mul.f16x2 r7823, r7820, r7624;
}
{
add.f16x2 r7826, r7817, r7823;
}
{
add.f16x2 r7829, r7811, r7826;
}
{
add.f16x2 r7832, r6278, r6326;
}
{
mul.f16x2 r7835, r7832, r7618;
}
{
add.f16x2 r7838, r4663, r7835;
}
{
add.f16x2 r7841, r6294, r6310;
}
{
mul.f16x2 r7844, r7841, r7622;
}
{
add.f16x2 r7847, r7838, r7844;
}
{
sub.f16x2 r7850, r6272, r6320;
}
{
mul.f16x2 r7853, r7850, r7620;
}
{
sub.f16x2 r7856, r6288, r6304;
}
{
mul.f16x2 r7859, r7856, r7624;
}
{
add.f16x2 r7862, r7853, r7859;
}
{
sub.f16x2 r7865, r7847, r7862;
}
{
add.f16x2 r7868, r6278, r6326;
}
{
mul.f16x2 r7871, r7868, r7622;
}
{
add.f16x2 r7874, r4663, r7871;
}
{
add.f16x2 r7877, r6294, r6310;
}
{
mul.f16x2 r7880, r7877, r7626;
}
{
add.f16x2 r7883, r7874, r7880;
}
{
sub.f16x2 r7886, r6272, r6320;
}
{
mul.f16x2 r7889, r7886, r7624;
}
{
sub.f16x2 r7892, r6288, r6304;
}
{
mul.f16x2 r7895, r7892, r7627;
}
{
add.f16x2 r7898, r7889, r7895;
}
{
add.f16x2 r7901, r7883, r7898;
}
{
add.f16x2 r7904, r6278, r6326;
}
{
mul.f16x2 r7907, r7904, r7622;
}
{
add.f16x2 r7910, r4663, r7907;
}
{
add.f16x2 r7913, r6294, r6310;
}
{
mul.f16x2 r7916, r7913, r7626;
}
{
add.f16x2 r7919, r7910, r7916;
}
{
sub.f16x2 r7922, r6272, r6320;
}
{
mul.f16x2 r7925, r7922, r7624;
}
{
sub.f16x2 r7928, r6288, r6304;
}
{
mul.f16x2 r7931, r7928, r7627;
}
{
add.f16x2 r7934, r7925, r7931;
}
{
sub.f16x2 r7937, r7919, r7934;
}
mul.wide.u32 rd4, r10447, 1374389535;
shr.u64 rd5, rd4, 35;
cvt.u32.u64 r10451, rd5;
cvt.rn.f32.u32 f600, r10451;
mul.f32 f601, f600, 0f3D4DE32E;
cos.approx.f32 f485, f601;
sin.approx.f32 f602, f601;
neg.f32 f486, f602;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f485;
cvt.rn.f16.f32 high, f486;
mov.b32 r7940, {low, high};
}
mul.lo.s32 r10452, r10451, 25;
sub.s32 r10453, r10447, r10452;
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r7943, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r7945, {high, high};
}
{
mul.f16x2 r7947, r6683, r7945;
}
{
fma.rn.f16x2 r7950, r6671, r7943, r7947;
}
{
mul.f16x2 r7954, r6671, r7945;
}
{
neg.f16x2 r7957, r7954;
}
{
fma.rn.f16x2 r7959, r6683, r7943, r7957;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r7963, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r7965, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r7967, {low, high};
}
{
mul.f16x2 r7968, r7965, r7967;
}
{
mul.f16x2 r7971, r7940, r7963;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r7974, {high, low};
}
{
fma.rn.f16x2 r7976, r7968, r7974, r7971;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7976;
mov.b32 r7980, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7976;
mov.b32 r7982, {high, high};
}
{
mul.f16x2 r7984, r7005, r7982;
}
{
fma.rn.f16x2 r7987, r6993, r7980, r7984;
}
{
mul.f16x2 r7991, r6993, r7982;
}
{
neg.f16x2 r7994, r7991;
}
{
fma.rn.f16x2 r7996, r7005, r7980, r7994;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8000, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8002, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8004, {low, high};
}
{
mul.f16x2 r8005, r8002, r8004;
}
{
mul.f16x2 r8008, r7976, r8000;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7976;
mov.b32 r8011, {high, low};
}
{
fma.rn.f16x2 r8013, r8005, r8011, r8008;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8013;
mov.b32 r8017, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8013;
mov.b32 r8019, {high, high};
}
{
mul.f16x2 r8021, r7327, r8019;
}
{
fma.rn.f16x2 r8024, r7315, r8017, r8021;
}
{
mul.f16x2 r8028, r7315, r8019;
}
{
neg.f16x2 r8031, r8028;
}
{
fma.rn.f16x2 r8033, r7327, r8017, r8031;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8037, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8039, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8041, {low, high};
}
{
mul.f16x2 r8042, r8039, r8041;
}
{
mul.f16x2 r8045, r8013, r8037;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8013;
mov.b32 r8048, {high, low};
}
{
fma.rn.f16x2 r8050, r8042, r8048, r8045;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8050;
mov.b32 r8054, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8050;
mov.b32 r8056, {high, high};
}
{
mul.f16x2 r8058, r7649, r8056;
}
{
fma.rn.f16x2 r8061, r7637, r8054, r8058;
}
{
mul.f16x2 r8065, r7637, r8056;
}
{
neg.f16x2 r8068, r8065;
}
{
fma.rn.f16x2 r8070, r7649, r8054, r8068;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8074, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8076, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8078, {low, high};
}
{
mul.f16x2 r8079, r8076, r8078;
}
{
mul.f16x2 r8082, r8050, r8074;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8050;
mov.b32 r8085, {high, low};
}
{
fma.rn.f16x2 r8087, r8079, r8085, r8082;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8087;
mov.b32 r8091, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8087;
mov.b32 r8093, {high, high};
}
{
mul.f16x2 r8095, r6541, r8093;
}
{
fma.rn.f16x2 r8098, r6397, r8091, r8095;
}
{
mul.f16x2 r8102, r6397, r8093;
}
{
neg.f16x2 r8105, r8102;
}
{
fma.rn.f16x2 r8107, r6541, r8091, r8105;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8111, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8113, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8115, {low, high};
}
{
mul.f16x2 r8116, r8113, r8115;
}
{
mul.f16x2 r8119, r8087, r8111;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8087;
mov.b32 r8122, {high, low};
}
{
fma.rn.f16x2 r8124, r8116, r8122, r8119;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8124;
mov.b32 r8128, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8124;
mov.b32 r8130, {high, high};
}
{
mul.f16x2 r8132, r6863, r8130;
}
{
fma.rn.f16x2 r8135, r6719, r8128, r8132;
}
{
mul.f16x2 r8139, r6719, r8130;
}
{
neg.f16x2 r8142, r8139;
}
{
fma.rn.f16x2 r8144, r6863, r8128, r8142;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8148, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8150, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8152, {low, high};
}
{
mul.f16x2 r8153, r8150, r8152;
}
{
mul.f16x2 r8156, r8124, r8148;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8124;
mov.b32 r8159, {high, low};
}
{
fma.rn.f16x2 r8161, r8153, r8159, r8156;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8161;
mov.b32 r8165, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8161;
mov.b32 r8167, {high, high};
}
{
mul.f16x2 r8169, r7185, r8167;
}
{
fma.rn.f16x2 r8172, r7041, r8165, r8169;
}
{
mul.f16x2 r8176, r7041, r8167;
}
{
neg.f16x2 r8179, r8176;
}
{
fma.rn.f16x2 r8181, r7185, r8165, r8179;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8185, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8187, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8189, {low, high};
}
{
mul.f16x2 r8190, r8187, r8189;
}
{
mul.f16x2 r8193, r8161, r8185;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8161;
mov.b32 r8196, {high, low};
}
{
fma.rn.f16x2 r8198, r8190, r8196, r8193;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8198;
mov.b32 r8202, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8198;
mov.b32 r8204, {high, high};
}
{
mul.f16x2 r8206, r7507, r8204;
}
{
fma.rn.f16x2 r8209, r7363, r8202, r8206;
}
{
mul.f16x2 r8213, r7363, r8204;
}
{
neg.f16x2 r8216, r8213;
}
{
fma.rn.f16x2 r8218, r7507, r8202, r8216;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8222, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8224, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8226, {low, high};
}
{
mul.f16x2 r8227, r8224, r8226;
}
{
mul.f16x2 r8230, r8198, r8222;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8198;
mov.b32 r8233, {high, low};
}
{
fma.rn.f16x2 r8235, r8227, r8233, r8230;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8235;
mov.b32 r8239, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8235;
mov.b32 r8241, {high, high};
}
{
mul.f16x2 r8243, r7829, r8241;
}
{
fma.rn.f16x2 r8246, r7685, r8239, r8243;
}
{
mul.f16x2 r8250, r7685, r8241;
}
{
neg.f16x2 r8253, r8250;
}
{
fma.rn.f16x2 r8255, r7829, r8239, r8253;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8259, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8261, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8263, {low, high};
}
{
mul.f16x2 r8264, r8261, r8263;
}
{
mul.f16x2 r8267, r8235, r8259;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8235;
mov.b32 r8270, {high, low};
}
{
fma.rn.f16x2 r8272, r8264, r8270, r8267;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8272;
mov.b32 r8276, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8272;
mov.b32 r8278, {high, high};
}
{
mul.f16x2 r8280, r6613, r8278;
}
{
fma.rn.f16x2 r8283, r6469, r8276, r8280;
}
{
mul.f16x2 r8287, r6469, r8278;
}
{
neg.f16x2 r8290, r8287;
}
{
fma.rn.f16x2 r8292, r6613, r8276, r8290;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8296, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8298, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8300, {low, high};
}
{
mul.f16x2 r8301, r8298, r8300;
}
{
mul.f16x2 r8304, r8272, r8296;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8272;
mov.b32 r8307, {high, low};
}
{
fma.rn.f16x2 r8309, r8301, r8307, r8304;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8309;
mov.b32 r8313, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8309;
mov.b32 r8315, {high, high};
}
{
mul.f16x2 r8317, r6935, r8315;
}
{
fma.rn.f16x2 r8320, r6791, r8313, r8317;
}
{
mul.f16x2 r8324, r6791, r8315;
}
{
neg.f16x2 r8327, r8324;
}
{
fma.rn.f16x2 r8329, r6935, r8313, r8327;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8333, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8335, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8337, {low, high};
}
{
mul.f16x2 r8338, r8335, r8337;
}
{
mul.f16x2 r8341, r8309, r8333;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8309;
mov.b32 r8344, {high, low};
}
{
fma.rn.f16x2 r8346, r8338, r8344, r8341;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8346;
mov.b32 r8350, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8346;
mov.b32 r8352, {high, high};
}
{
mul.f16x2 r8354, r7257, r8352;
}
{
fma.rn.f16x2 r8357, r7113, r8350, r8354;
}
{
mul.f16x2 r8361, r7113, r8352;
}
{
neg.f16x2 r8364, r8361;
}
{
fma.rn.f16x2 r8366, r7257, r8350, r8364;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8370, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8372, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8374, {low, high};
}
{
mul.f16x2 r8375, r8372, r8374;
}
{
mul.f16x2 r8378, r8346, r8370;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8346;
mov.b32 r8381, {high, low};
}
{
fma.rn.f16x2 r8383, r8375, r8381, r8378;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8383;
mov.b32 r8387, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8383;
mov.b32 r8389, {high, high};
}
{
mul.f16x2 r8391, r7579, r8389;
}
{
fma.rn.f16x2 r8394, r7435, r8387, r8391;
}
{
mul.f16x2 r8398, r7435, r8389;
}
{
neg.f16x2 r8401, r8398;
}
{
fma.rn.f16x2 r8403, r7579, r8387, r8401;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8407, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8409, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8411, {low, high};
}
{
mul.f16x2 r8412, r8409, r8411;
}
{
mul.f16x2 r8415, r8383, r8407;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8383;
mov.b32 r8418, {high, low};
}
{
fma.rn.f16x2 r8420, r8412, r8418, r8415;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8420;
mov.b32 r8424, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8420;
mov.b32 r8426, {high, high};
}
{
mul.f16x2 r8428, r7901, r8426;
}
{
fma.rn.f16x2 r8431, r7757, r8424, r8428;
}
{
mul.f16x2 r8435, r7757, r8426;
}
{
neg.f16x2 r8438, r8435;
}
{
fma.rn.f16x2 r8440, r7901, r8424, r8438;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8444, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8446, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8448, {low, high};
}
{
mul.f16x2 r8449, r8446, r8448;
}
{
mul.f16x2 r8452, r8420, r8444;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8420;
mov.b32 r8455, {high, low};
}
{
fma.rn.f16x2 r8457, r8449, r8455, r8452;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8457;
mov.b32 r8461, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8457;
mov.b32 r8463, {high, high};
}
{
mul.f16x2 r8465, r6649, r8463;
}
{
fma.rn.f16x2 r8468, r6505, r8461, r8465;
}
{
mul.f16x2 r8472, r6505, r8463;
}
{
neg.f16x2 r8475, r8472;
}
{
fma.rn.f16x2 r8477, r6649, r8461, r8475;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8481, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8483, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8485, {low, high};
}
{
mul.f16x2 r8486, r8483, r8485;
}
{
mul.f16x2 r8489, r8457, r8481;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8457;
mov.b32 r8492, {high, low};
}
{
fma.rn.f16x2 r8494, r8486, r8492, r8489;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8494;
mov.b32 r8498, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8494;
mov.b32 r8500, {high, high};
}
{
mul.f16x2 r8502, r6971, r8500;
}
{
fma.rn.f16x2 r8505, r6827, r8498, r8502;
}
{
mul.f16x2 r8509, r6827, r8500;
}
{
neg.f16x2 r8512, r8509;
}
{
fma.rn.f16x2 r8514, r6971, r8498, r8512;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8518, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8520, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8522, {low, high};
}
{
mul.f16x2 r8523, r8520, r8522;
}
{
mul.f16x2 r8526, r8494, r8518;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8494;
mov.b32 r8529, {high, low};
}
{
fma.rn.f16x2 r8531, r8523, r8529, r8526;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8531;
mov.b32 r8535, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8531;
mov.b32 r8537, {high, high};
}
{
mul.f16x2 r8539, r7293, r8537;
}
{
fma.rn.f16x2 r8542, r7149, r8535, r8539;
}
{
mul.f16x2 r8546, r7149, r8537;
}
{
neg.f16x2 r8549, r8546;
}
{
fma.rn.f16x2 r8551, r7293, r8535, r8549;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8555, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8557, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8559, {low, high};
}
{
mul.f16x2 r8560, r8557, r8559;
}
{
mul.f16x2 r8563, r8531, r8555;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8531;
mov.b32 r8566, {high, low};
}
{
fma.rn.f16x2 r8568, r8560, r8566, r8563;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8568;
mov.b32 r8572, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8568;
mov.b32 r8574, {high, high};
}
{
mul.f16x2 r8576, r7615, r8574;
}
{
fma.rn.f16x2 r8579, r7471, r8572, r8576;
}
{
mul.f16x2 r8583, r7471, r8574;
}
{
neg.f16x2 r8586, r8583;
}
{
fma.rn.f16x2 r8588, r7615, r8572, r8586;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8592, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8594, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8596, {low, high};
}
{
mul.f16x2 r8597, r8594, r8596;
}
{
mul.f16x2 r8600, r8568, r8592;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8568;
mov.b32 r8603, {high, low};
}
{
fma.rn.f16x2 r8605, r8597, r8603, r8600;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8605;
mov.b32 r8609, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8605;
mov.b32 r8611, {high, high};
}
{
mul.f16x2 r8613, r7937, r8611;
}
{
fma.rn.f16x2 r8616, r7793, r8609, r8613;
}
{
mul.f16x2 r8620, r7793, r8611;
}
{
neg.f16x2 r8623, r8620;
}
{
fma.rn.f16x2 r8625, r7937, r8609, r8623;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8629, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8631, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8633, {low, high};
}
{
mul.f16x2 r8634, r8631, r8633;
}
{
mul.f16x2 r8637, r8605, r8629;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8605;
mov.b32 r8640, {high, low};
}
{
fma.rn.f16x2 r8642, r8634, r8640, r8637;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8642;
mov.b32 r8646, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8642;
mov.b32 r8648, {high, high};
}
{
mul.f16x2 r8650, r6577, r8648;
}
{
fma.rn.f16x2 r8653, r6433, r8646, r8650;
}
{
mul.f16x2 r8657, r6433, r8648;
}
{
neg.f16x2 r8660, r8657;
}
{
fma.rn.f16x2 r8662, r6577, r8646, r8660;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8666, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8668, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8670, {low, high};
}
{
mul.f16x2 r8671, r8668, r8670;
}
{
mul.f16x2 r8674, r8642, r8666;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8642;
mov.b32 r8677, {high, low};
}
{
fma.rn.f16x2 r8679, r8671, r8677, r8674;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8679;
mov.b32 r8683, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8679;
mov.b32 r8685, {high, high};
}
{
mul.f16x2 r8687, r6899, r8685;
}
{
fma.rn.f16x2 r8690, r6755, r8683, r8687;
}
{
mul.f16x2 r8694, r6755, r8685;
}
{
neg.f16x2 r8697, r8694;
}
{
fma.rn.f16x2 r8699, r6899, r8683, r8697;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8703, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8705, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8707, {low, high};
}
{
mul.f16x2 r8708, r8705, r8707;
}
{
mul.f16x2 r8711, r8679, r8703;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8679;
mov.b32 r8714, {high, low};
}
{
fma.rn.f16x2 r8716, r8708, r8714, r8711;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8716;
mov.b32 r8720, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8716;
mov.b32 r8722, {high, high};
}
{
mul.f16x2 r8724, r7221, r8722;
}
{
fma.rn.f16x2 r8727, r7077, r8720, r8724;
}
{
mul.f16x2 r8731, r7077, r8722;
}
{
neg.f16x2 r8734, r8731;
}
{
fma.rn.f16x2 r8736, r7221, r8720, r8734;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8740, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8742, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8744, {low, high};
}
{
mul.f16x2 r8745, r8742, r8744;
}
{
mul.f16x2 r8748, r8716, r8740;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8716;
mov.b32 r8751, {high, low};
}
{
fma.rn.f16x2 r8753, r8745, r8751, r8748;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8753;
mov.b32 r8757, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8753;
mov.b32 r8759, {high, high};
}
{
mul.f16x2 r8761, r7543, r8759;
}
{
fma.rn.f16x2 r8764, r7399, r8757, r8761;
}
{
mul.f16x2 r8768, r7399, r8759;
}
{
neg.f16x2 r8771, r8768;
}
{
fma.rn.f16x2 r8773, r7543, r8757, r8771;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8777, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8779, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8781, {low, high};
}
{
mul.f16x2 r8782, r8779, r8781;
}
{
mul.f16x2 r8785, r8753, r8777;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8753;
mov.b32 r8788, {high, low};
}
{
fma.rn.f16x2 r8790, r8782, r8788, r8785;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8790;
mov.b32 r8794, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8790;
mov.b32 r8796, {high, high};
}
{
mul.f16x2 r8798, r7865, r8796;
}
{
fma.rn.f16x2 r8801, r7721, r8794, r8798;
}
{
mul.f16x2 r8805, r7721, r8796;
}
{
neg.f16x2 r8808, r8805;
}
{
fma.rn.f16x2 r8810, r7865, r8794, r8808;
}
shl.b32 r10454, r10453, 3;
add.s32 r10455, r10448, r10454;
barrier.sync 0;
mad.lo.s32 r10456, r10451, 5000, r10455;
st.shared.u32 [r10456], r6349;
st.shared.u32 [r10456+4], r6361;
st.shared.u32 [r10456+200], r7950;
st.shared.u32 [r10456+204], r7959;
st.shared.u32 [r10456+400], r7987;
st.shared.u32 [r10456+404], r7996;
st.shared.u32 [r10456+600], r8024;
st.shared.u32 [r10456+604], r8033;
st.shared.u32 [r10456+800], r8061;
st.shared.u32 [r10456+804], r8070;
st.shared.u32 [r10456+1000], r8098;
st.shared.u32 [r10456+1004], r8107;
st.shared.u32 [r10456+1200], r8135;
st.shared.u32 [r10456+1204], r8144;
st.shared.u32 [r10456+1400], r8172;
st.shared.u32 [r10456+1404], r8181;
st.shared.u32 [r10456+1600], r8209;
st.shared.u32 [r10456+1604], r8218;
st.shared.u32 [r10456+1800], r8246;
st.shared.u32 [r10456+1804], r8255;
st.shared.u32 [r10456+2000], r8283;
st.shared.u32 [r10456+2004], r8292;
st.shared.u32 [r10456+2200], r8320;
st.shared.u32 [r10456+2204], r8329;
st.shared.u32 [r10456+2400], r8357;
st.shared.u32 [r10456+2404], r8366;
st.shared.u32 [r10456+2600], r8394;
st.shared.u32 [r10456+2604], r8403;
st.shared.u32 [r10456+2800], r8431;
st.shared.u32 [r10456+2804], r8440;
st.shared.u32 [r10456+3000], r8468;
st.shared.u32 [r10456+3004], r8477;
st.shared.u32 [r10456+3200], r8505;
st.shared.u32 [r10456+3204], r8514;
st.shared.u32 [r10456+3400], r8542;
st.shared.u32 [r10456+3404], r8551;
st.shared.u32 [r10456+3600], r8579;
st.shared.u32 [r10456+3604], r8588;
st.shared.u32 [r10456+3800], r8616;
st.shared.u32 [r10456+3804], r8625;
st.shared.u32 [r10456+4000], r8653;
st.shared.u32 [r10456+4004], r8662;
st.shared.u32 [r10456+4200], r8690;
st.shared.u32 [r10456+4204], r8699;
st.shared.u32 [r10456+4400], r8727;
st.shared.u32 [r10456+4404], r8736;
st.shared.u32 [r10456+4600], r8764;
st.shared.u32 [r10456+4604], r8773;
st.shared.u32 [r10456+4800], r8801;
st.shared.u32 [r10456+4804], r8810;
barrier.sync 0;
ld.shared.u32 r8845, [r10450];
ld.shared.u32 r8857, [r10450+4];
ld.shared.u32 r9167, [r10450+1000];
ld.shared.u32 r9179, [r10450+1004];
ld.shared.u32 r9489, [r10450+2000];
ld.shared.u32 r9501, [r10450+2004];
ld.shared.u32 r9811, [r10450+3000];
ld.shared.u32 r9823, [r10450+3004];
ld.shared.u32 r10133, [r10450+4000];
ld.shared.u32 r10145, [r10450+4004];
ld.shared.u32 r8842, [r10450+5000];
ld.shared.u32 r8854, [r10450+5004];
ld.shared.u32 r9164, [r10450+6000];
ld.shared.u32 r9176, [r10450+6004];
ld.shared.u32 r9486, [r10450+7000];
ld.shared.u32 r9498, [r10450+7004];
ld.shared.u32 r9808, [r10450+8000];
ld.shared.u32 r9820, [r10450+8004];
ld.shared.u32 r10130, [r10450+9000];
ld.shared.u32 r10142, [r10450+9004];
ld.shared.u32 r8848, [r10450+10000];
ld.shared.u32 r8860, [r10450+10004];
ld.shared.u32 r9170, [r10450+11000];
ld.shared.u32 r9182, [r10450+11004];
ld.shared.u32 r9492, [r10450+12000];
ld.shared.u32 r9504, [r10450+12004];
ld.shared.u32 r9814, [r10450+13000];
ld.shared.u32 r9826, [r10450+13004];
ld.shared.u32 r10136, [r10450+14000];
ld.shared.u32 r10148, [r10450+14004];
ld.shared.u32 r8849, [r10450+15000];
ld.shared.u32 r8861, [r10450+15004];
ld.shared.u32 r9171, [r10450+16000];
ld.shared.u32 r9183, [r10450+16004];
ld.shared.u32 r9493, [r10450+17000];
ld.shared.u32 r9505, [r10450+17004];
ld.shared.u32 r9815, [r10450+18000];
ld.shared.u32 r9827, [r10450+18004];
ld.shared.u32 r10137, [r10450+19000];
ld.shared.u32 r10149, [r10450+19004];
ld.shared.u32 r8843, [r10450+20000];
ld.shared.u32 r8855, [r10450+20004];
ld.shared.u32 r9165, [r10450+21000];
ld.shared.u32 r9177, [r10450+21004];
ld.shared.u32 r9487, [r10450+22000];
ld.shared.u32 r9499, [r10450+22004];
ld.shared.u32 r9809, [r10450+23000];
ld.shared.u32 r9821, [r10450+23004];
ld.shared.u32 r10131, [r10450+24000];
ld.shared.u32 r10143, [r10450+24004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r8831, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r8832, {low, high};
}
{
neg.f16x2 r8833, r8832;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r8835, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r8836, {low, high};
}
{
neg.f16x2 r8837, r8836;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r8839, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r8840, {low, high};
}
{
add.f16x2 r8841, r8842, r8843;
}
{
add.f16x2 r8844, r8845, r8841;
}
{
add.f16x2 r8847, r8848, r8849;
}
{
add.f16x2 %0, r8844, r8847;
}
{
add.f16x2 r8853, r8854, r8855;
}
{
add.f16x2 r8856, r8857, r8853;
}
{
add.f16x2 r8859, r8860, r8861;
}
{
add.f16x2 %1, r8856, r8859;
}
{
add.f16x2 r8865, r8842, r8843;
}
{
mul.f16x2 r8868, r8865, r8831;
}
{
add.f16x2 r8871, r8845, r8868;
}
{
add.f16x2 r8874, r8848, r8849;
}
{
mul.f16x2 r8877, r8874, r8835;
}
{
add.f16x2 r8880, r8871, r8877;
}
{
sub.f16x2 r8883, r8854, r8855;
}
{
mul.f16x2 r8886, r8883, r8833;
}
{
sub.f16x2 r8889, r8860, r8861;
}
{
mul.f16x2 r8892, r8889, r8837;
}
{
add.f16x2 r8895, r8886, r8892;
}
{
sub.f16x2 %10, r8880, r8895;
}
{
add.f16x2 r8901, r8842, r8843;
}
{
mul.f16x2 r8904, r8901, r8831;
}
{
add.f16x2 r8907, r8845, r8904;
}
{
add.f16x2 r8910, r8848, r8849;
}
{
mul.f16x2 r8913, r8910, r8835;
}
{
add.f16x2 r8916, r8907, r8913;
}
{
sub.f16x2 r8919, r8854, r8855;
}
{
mul.f16x2 r8922, r8919, r8833;
}
{
sub.f16x2 r8925, r8860, r8861;
}
{
mul.f16x2 r8928, r8925, r8837;
}
{
add.f16x2 r8931, r8922, r8928;
}
{
add.f16x2 %40, r8916, r8931;
}
{
add.f16x2 r8937, r8842, r8843;
}
{
mul.f16x2 r8940, r8937, r8835;
}
{
add.f16x2 r8943, r8845, r8940;
}
{
add.f16x2 r8946, r8848, r8849;
}
{
mul.f16x2 r8949, r8946, r8839;
}
{
add.f16x2 r8952, r8943, r8949;
}
{
sub.f16x2 r8955, r8854, r8855;
}
{
mul.f16x2 r8958, r8955, r8837;
}
{
sub.f16x2 r8961, r8860, r8861;
}
{
mul.f16x2 r8964, r8961, r8840;
}
{
add.f16x2 r8967, r8958, r8964;
}
{
sub.f16x2 %20, r8952, r8967;
}
{
add.f16x2 r8973, r8842, r8843;
}
{
mul.f16x2 r8976, r8973, r8835;
}
{
add.f16x2 r8979, r8845, r8976;
}
{
add.f16x2 r8982, r8848, r8849;
}
{
mul.f16x2 r8985, r8982, r8839;
}
{
add.f16x2 r8988, r8979, r8985;
}
{
sub.f16x2 r8991, r8854, r8855;
}
{
mul.f16x2 r8994, r8991, r8837;
}
{
sub.f16x2 r8997, r8860, r8861;
}
{
mul.f16x2 r9000, r8997, r8840;
}
{
add.f16x2 r9003, r8994, r9000;
}
{
add.f16x2 %30, r8988, r9003;
}
{
add.f16x2 r9009, r8854, r8855;
}
{
mul.f16x2 r9012, r9009, r8831;
}
{
add.f16x2 r9015, r8857, r9012;
}
{
add.f16x2 r9018, r8860, r8861;
}
{
mul.f16x2 r9021, r9018, r8835;
}
{
add.f16x2 r9024, r9015, r9021;
}
{
sub.f16x2 r9027, r8842, r8843;
}
{
mul.f16x2 r9030, r9027, r8833;
}
{
sub.f16x2 r9033, r8848, r8849;
}
{
mul.f16x2 r9036, r9033, r8837;
}
{
add.f16x2 r9039, r9030, r9036;
}
{
add.f16x2 %11, r9024, r9039;
}
{
add.f16x2 r9045, r8854, r8855;
}
{
mul.f16x2 r9048, r9045, r8831;
}
{
add.f16x2 r9051, r8857, r9048;
}
{
add.f16x2 r9054, r8860, r8861;
}
{
mul.f16x2 r9057, r9054, r8835;
}
{
add.f16x2 r9060, r9051, r9057;
}
{
sub.f16x2 r9063, r8842, r8843;
}
{
mul.f16x2 r9066, r9063, r8833;
}
{
sub.f16x2 r9069, r8848, r8849;
}
{
mul.f16x2 r9072, r9069, r8837;
}
{
add.f16x2 r9075, r9066, r9072;
}
{
sub.f16x2 %41, r9060, r9075;
}
{
add.f16x2 r9081, r8854, r8855;
}
{
mul.f16x2 r9084, r9081, r8835;
}
{
add.f16x2 r9087, r8857, r9084;
}
{
add.f16x2 r9090, r8860, r8861;
}
{
mul.f16x2 r9093, r9090, r8839;
}
{
add.f16x2 r9096, r9087, r9093;
}
{
sub.f16x2 r9099, r8842, r8843;
}
{
mul.f16x2 r9102, r9099, r8837;
}
{
sub.f16x2 r9105, r8848, r8849;
}
{
mul.f16x2 r9108, r9105, r8840;
}
{
add.f16x2 r9111, r9102, r9108;
}
{
add.f16x2 %21, r9096, r9111;
}
{
add.f16x2 r9117, r8854, r8855;
}
{
mul.f16x2 r9120, r9117, r8835;
}
{
add.f16x2 r9123, r8857, r9120;
}
{
add.f16x2 r9126, r8860, r8861;
}
{
mul.f16x2 r9129, r9126, r8839;
}
{
add.f16x2 r9132, r9123, r9129;
}
{
sub.f16x2 r9135, r8842, r8843;
}
{
mul.f16x2 r9138, r9135, r8837;
}
{
sub.f16x2 r9141, r8848, r8849;
}
{
mul.f16x2 r9144, r9141, r8840;
}
{
add.f16x2 r9147, r9138, r9144;
}
{
sub.f16x2 %31, r9132, r9147;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9153, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9154, {low, high};
}
{
neg.f16x2 r9155, r9154;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r9157, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r9158, {low, high};
}
{
neg.f16x2 r9159, r9158;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9161, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9162, {low, high};
}
{
add.f16x2 r9163, r9164, r9165;
}
{
add.f16x2 r9166, r9167, r9163;
}
{
add.f16x2 r9169, r9170, r9171;
}
{
add.f16x2 %2, r9166, r9169;
}
{
add.f16x2 r9175, r9176, r9177;
}
{
add.f16x2 r9178, r9179, r9175;
}
{
add.f16x2 r9181, r9182, r9183;
}
{
add.f16x2 %3, r9178, r9181;
}
{
add.f16x2 r9187, r9164, r9165;
}
{
mul.f16x2 r9190, r9187, r9153;
}
{
add.f16x2 r9193, r9167, r9190;
}
{
add.f16x2 r9196, r9170, r9171;
}
{
mul.f16x2 r9199, r9196, r9157;
}
{
add.f16x2 r9202, r9193, r9199;
}
{
sub.f16x2 r9205, r9176, r9177;
}
{
mul.f16x2 r9208, r9205, r9155;
}
{
sub.f16x2 r9211, r9182, r9183;
}
{
mul.f16x2 r9214, r9211, r9159;
}
{
add.f16x2 r9217, r9208, r9214;
}
{
sub.f16x2 %12, r9202, r9217;
}
{
add.f16x2 r9223, r9164, r9165;
}
{
mul.f16x2 r9226, r9223, r9153;
}
{
add.f16x2 r9229, r9167, r9226;
}
{
add.f16x2 r9232, r9170, r9171;
}
{
mul.f16x2 r9235, r9232, r9157;
}
{
add.f16x2 r9238, r9229, r9235;
}
{
sub.f16x2 r9241, r9176, r9177;
}
{
mul.f16x2 r9244, r9241, r9155;
}
{
sub.f16x2 r9247, r9182, r9183;
}
{
mul.f16x2 r9250, r9247, r9159;
}
{
add.f16x2 r9253, r9244, r9250;
}
{
add.f16x2 %42, r9238, r9253;
}
{
add.f16x2 r9259, r9164, r9165;
}
{
mul.f16x2 r9262, r9259, r9157;
}
{
add.f16x2 r9265, r9167, r9262;
}
{
add.f16x2 r9268, r9170, r9171;
}
{
mul.f16x2 r9271, r9268, r9161;
}
{
add.f16x2 r9274, r9265, r9271;
}
{
sub.f16x2 r9277, r9176, r9177;
}
{
mul.f16x2 r9280, r9277, r9159;
}
{
sub.f16x2 r9283, r9182, r9183;
}
{
mul.f16x2 r9286, r9283, r9162;
}
{
add.f16x2 r9289, r9280, r9286;
}
{
sub.f16x2 %22, r9274, r9289;
}
{
add.f16x2 r9295, r9164, r9165;
}
{
mul.f16x2 r9298, r9295, r9157;
}
{
add.f16x2 r9301, r9167, r9298;
}
{
add.f16x2 r9304, r9170, r9171;
}
{
mul.f16x2 r9307, r9304, r9161;
}
{
add.f16x2 r9310, r9301, r9307;
}
{
sub.f16x2 r9313, r9176, r9177;
}
{
mul.f16x2 r9316, r9313, r9159;
}
{
sub.f16x2 r9319, r9182, r9183;
}
{
mul.f16x2 r9322, r9319, r9162;
}
{
add.f16x2 r9325, r9316, r9322;
}
{
add.f16x2 %32, r9310, r9325;
}
{
add.f16x2 r9331, r9176, r9177;
}
{
mul.f16x2 r9334, r9331, r9153;
}
{
add.f16x2 r9337, r9179, r9334;
}
{
add.f16x2 r9340, r9182, r9183;
}
{
mul.f16x2 r9343, r9340, r9157;
}
{
add.f16x2 r9346, r9337, r9343;
}
{
sub.f16x2 r9349, r9164, r9165;
}
{
mul.f16x2 r9352, r9349, r9155;
}
{
sub.f16x2 r9355, r9170, r9171;
}
{
mul.f16x2 r9358, r9355, r9159;
}
{
add.f16x2 r9361, r9352, r9358;
}
{
add.f16x2 %13, r9346, r9361;
}
{
add.f16x2 r9367, r9176, r9177;
}
{
mul.f16x2 r9370, r9367, r9153;
}
{
add.f16x2 r9373, r9179, r9370;
}
{
add.f16x2 r9376, r9182, r9183;
}
{
mul.f16x2 r9379, r9376, r9157;
}
{
add.f16x2 r9382, r9373, r9379;
}
{
sub.f16x2 r9385, r9164, r9165;
}
{
mul.f16x2 r9388, r9385, r9155;
}
{
sub.f16x2 r9391, r9170, r9171;
}
{
mul.f16x2 r9394, r9391, r9159;
}
{
add.f16x2 r9397, r9388, r9394;
}
{
sub.f16x2 %43, r9382, r9397;
}
{
add.f16x2 r9403, r9176, r9177;
}
{
mul.f16x2 r9406, r9403, r9157;
}
{
add.f16x2 r9409, r9179, r9406;
}
{
add.f16x2 r9412, r9182, r9183;
}
{
mul.f16x2 r9415, r9412, r9161;
}
{
add.f16x2 r9418, r9409, r9415;
}
{
sub.f16x2 r9421, r9164, r9165;
}
{
mul.f16x2 r9424, r9421, r9159;
}
{
sub.f16x2 r9427, r9170, r9171;
}
{
mul.f16x2 r9430, r9427, r9162;
}
{
add.f16x2 r9433, r9424, r9430;
}
{
add.f16x2 %23, r9418, r9433;
}
{
add.f16x2 r9439, r9176, r9177;
}
{
mul.f16x2 r9442, r9439, r9157;
}
{
add.f16x2 r9445, r9179, r9442;
}
{
add.f16x2 r9448, r9182, r9183;
}
{
mul.f16x2 r9451, r9448, r9161;
}
{
add.f16x2 r9454, r9445, r9451;
}
{
sub.f16x2 r9457, r9164, r9165;
}
{
mul.f16x2 r9460, r9457, r9159;
}
{
sub.f16x2 r9463, r9170, r9171;
}
{
mul.f16x2 r9466, r9463, r9162;
}
{
add.f16x2 r9469, r9460, r9466;
}
{
sub.f16x2 %33, r9454, r9469;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9475, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9476, {low, high};
}
{
neg.f16x2 r9477, r9476;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r9479, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r9480, {low, high};
}
{
neg.f16x2 r9481, r9480;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9483, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9484, {low, high};
}
{
add.f16x2 r9485, r9486, r9487;
}
{
add.f16x2 r9488, r9489, r9485;
}
{
add.f16x2 r9491, r9492, r9493;
}
{
add.f16x2 %4, r9488, r9491;
}
{
add.f16x2 r9497, r9498, r9499;
}
{
add.f16x2 r9500, r9501, r9497;
}
{
add.f16x2 r9503, r9504, r9505;
}
{
add.f16x2 %5, r9500, r9503;
}
{
add.f16x2 r9509, r9486, r9487;
}
{
mul.f16x2 r9512, r9509, r9475;
}
{
add.f16x2 r9515, r9489, r9512;
}
{
add.f16x2 r9518, r9492, r9493;
}
{
mul.f16x2 r9521, r9518, r9479;
}
{
add.f16x2 r9524, r9515, r9521;
}
{
sub.f16x2 r9527, r9498, r9499;
}
{
mul.f16x2 r9530, r9527, r9477;
}
{
sub.f16x2 r9533, r9504, r9505;
}
{
mul.f16x2 r9536, r9533, r9481;
}
{
add.f16x2 r9539, r9530, r9536;
}
{
sub.f16x2 %14, r9524, r9539;
}
{
add.f16x2 r9545, r9486, r9487;
}
{
mul.f16x2 r9548, r9545, r9475;
}
{
add.f16x2 r9551, r9489, r9548;
}
{
add.f16x2 r9554, r9492, r9493;
}
{
mul.f16x2 r9557, r9554, r9479;
}
{
add.f16x2 r9560, r9551, r9557;
}
{
sub.f16x2 r9563, r9498, r9499;
}
{
mul.f16x2 r9566, r9563, r9477;
}
{
sub.f16x2 r9569, r9504, r9505;
}
{
mul.f16x2 r9572, r9569, r9481;
}
{
add.f16x2 r9575, r9566, r9572;
}
{
add.f16x2 %44, r9560, r9575;
}
{
add.f16x2 r9581, r9486, r9487;
}
{
mul.f16x2 r9584, r9581, r9479;
}
{
add.f16x2 r9587, r9489, r9584;
}
{
add.f16x2 r9590, r9492, r9493;
}
{
mul.f16x2 r9593, r9590, r9483;
}
{
add.f16x2 r9596, r9587, r9593;
}
{
sub.f16x2 r9599, r9498, r9499;
}
{
mul.f16x2 r9602, r9599, r9481;
}
{
sub.f16x2 r9605, r9504, r9505;
}
{
mul.f16x2 r9608, r9605, r9484;
}
{
add.f16x2 r9611, r9602, r9608;
}
{
sub.f16x2 %24, r9596, r9611;
}
{
add.f16x2 r9617, r9486, r9487;
}
{
mul.f16x2 r9620, r9617, r9479;
}
{
add.f16x2 r9623, r9489, r9620;
}
{
add.f16x2 r9626, r9492, r9493;
}
{
mul.f16x2 r9629, r9626, r9483;
}
{
add.f16x2 r9632, r9623, r9629;
}
{
sub.f16x2 r9635, r9498, r9499;
}
{
mul.f16x2 r9638, r9635, r9481;
}
{
sub.f16x2 r9641, r9504, r9505;
}
{
mul.f16x2 r9644, r9641, r9484;
}
{
add.f16x2 r9647, r9638, r9644;
}
{
add.f16x2 %34, r9632, r9647;
}
{
add.f16x2 r9653, r9498, r9499;
}
{
mul.f16x2 r9656, r9653, r9475;
}
{
add.f16x2 r9659, r9501, r9656;
}
{
add.f16x2 r9662, r9504, r9505;
}
{
mul.f16x2 r9665, r9662, r9479;
}
{
add.f16x2 r9668, r9659, r9665;
}
{
sub.f16x2 r9671, r9486, r9487;
}
{
mul.f16x2 r9674, r9671, r9477;
}
{
sub.f16x2 r9677, r9492, r9493;
}
{
mul.f16x2 r9680, r9677, r9481;
}
{
add.f16x2 r9683, r9674, r9680;
}
{
add.f16x2 %15, r9668, r9683;
}
{
add.f16x2 r9689, r9498, r9499;
}
{
mul.f16x2 r9692, r9689, r9475;
}
{
add.f16x2 r9695, r9501, r9692;
}
{
add.f16x2 r9698, r9504, r9505;
}
{
mul.f16x2 r9701, r9698, r9479;
}
{
add.f16x2 r9704, r9695, r9701;
}
{
sub.f16x2 r9707, r9486, r9487;
}
{
mul.f16x2 r9710, r9707, r9477;
}
{
sub.f16x2 r9713, r9492, r9493;
}
{
mul.f16x2 r9716, r9713, r9481;
}
{
add.f16x2 r9719, r9710, r9716;
}
{
sub.f16x2 %45, r9704, r9719;
}
{
add.f16x2 r9725, r9498, r9499;
}
{
mul.f16x2 r9728, r9725, r9479;
}
{
add.f16x2 r9731, r9501, r9728;
}
{
add.f16x2 r9734, r9504, r9505;
}
{
mul.f16x2 r9737, r9734, r9483;
}
{
add.f16x2 r9740, r9731, r9737;
}
{
sub.f16x2 r9743, r9486, r9487;
}
{
mul.f16x2 r9746, r9743, r9481;
}
{
sub.f16x2 r9749, r9492, r9493;
}
{
mul.f16x2 r9752, r9749, r9484;
}
{
add.f16x2 r9755, r9746, r9752;
}
{
add.f16x2 %25, r9740, r9755;
}
{
add.f16x2 r9761, r9498, r9499;
}
{
mul.f16x2 r9764, r9761, r9479;
}
{
add.f16x2 r9767, r9501, r9764;
}
{
add.f16x2 r9770, r9504, r9505;
}
{
mul.f16x2 r9773, r9770, r9483;
}
{
add.f16x2 r9776, r9767, r9773;
}
{
sub.f16x2 r9779, r9486, r9487;
}
{
mul.f16x2 r9782, r9779, r9481;
}
{
sub.f16x2 r9785, r9492, r9493;
}
{
mul.f16x2 r9788, r9785, r9484;
}
{
add.f16x2 r9791, r9782, r9788;
}
{
sub.f16x2 %35, r9776, r9791;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9797, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9798, {low, high};
}
{
neg.f16x2 r9799, r9798;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r9801, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r9802, {low, high};
}
{
neg.f16x2 r9803, r9802;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9805, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9806, {low, high};
}
{
add.f16x2 r9807, r9808, r9809;
}
{
add.f16x2 r9810, r9811, r9807;
}
{
add.f16x2 r9813, r9814, r9815;
}
{
add.f16x2 %6, r9810, r9813;
}
{
add.f16x2 r9819, r9820, r9821;
}
{
add.f16x2 r9822, r9823, r9819;
}
{
add.f16x2 r9825, r9826, r9827;
}
{
add.f16x2 %7, r9822, r9825;
}
{
add.f16x2 r9831, r9808, r9809;
}
{
mul.f16x2 r9834, r9831, r9797;
}
{
add.f16x2 r9837, r9811, r9834;
}
{
add.f16x2 r9840, r9814, r9815;
}
{
mul.f16x2 r9843, r9840, r9801;
}
{
add.f16x2 r9846, r9837, r9843;
}
{
sub.f16x2 r9849, r9820, r9821;
}
{
mul.f16x2 r9852, r9849, r9799;
}
{
sub.f16x2 r9855, r9826, r9827;
}
{
mul.f16x2 r9858, r9855, r9803;
}
{
add.f16x2 r9861, r9852, r9858;
}
{
sub.f16x2 %16, r9846, r9861;
}
{
add.f16x2 r9867, r9808, r9809;
}
{
mul.f16x2 r9870, r9867, r9797;
}
{
add.f16x2 r9873, r9811, r9870;
}
{
add.f16x2 r9876, r9814, r9815;
}
{
mul.f16x2 r9879, r9876, r9801;
}
{
add.f16x2 r9882, r9873, r9879;
}
{
sub.f16x2 r9885, r9820, r9821;
}
{
mul.f16x2 r9888, r9885, r9799;
}
{
sub.f16x2 r9891, r9826, r9827;
}
{
mul.f16x2 r9894, r9891, r9803;
}
{
add.f16x2 r9897, r9888, r9894;
}
{
add.f16x2 %46, r9882, r9897;
}
{
add.f16x2 r9903, r9808, r9809;
}
{
mul.f16x2 r9906, r9903, r9801;
}
{
add.f16x2 r9909, r9811, r9906;
}
{
add.f16x2 r9912, r9814, r9815;
}
{
mul.f16x2 r9915, r9912, r9805;
}
{
add.f16x2 r9918, r9909, r9915;
}
{
sub.f16x2 r9921, r9820, r9821;
}
{
mul.f16x2 r9924, r9921, r9803;
}
{
sub.f16x2 r9927, r9826, r9827;
}
{
mul.f16x2 r9930, r9927, r9806;
}
{
add.f16x2 r9933, r9924, r9930;
}
{
sub.f16x2 %26, r9918, r9933;
}
{
add.f16x2 r9939, r9808, r9809;
}
{
mul.f16x2 r9942, r9939, r9801;
}
{
add.f16x2 r9945, r9811, r9942;
}
{
add.f16x2 r9948, r9814, r9815;
}
{
mul.f16x2 r9951, r9948, r9805;
}
{
add.f16x2 r9954, r9945, r9951;
}
{
sub.f16x2 r9957, r9820, r9821;
}
{
mul.f16x2 r9960, r9957, r9803;
}
{
sub.f16x2 r9963, r9826, r9827;
}
{
mul.f16x2 r9966, r9963, r9806;
}
{
add.f16x2 r9969, r9960, r9966;
}
{
add.f16x2 %36, r9954, r9969;
}
{
add.f16x2 r9975, r9820, r9821;
}
{
mul.f16x2 r9978, r9975, r9797;
}
{
add.f16x2 r9981, r9823, r9978;
}
{
add.f16x2 r9984, r9826, r9827;
}
{
mul.f16x2 r9987, r9984, r9801;
}
{
add.f16x2 r9990, r9981, r9987;
}
{
sub.f16x2 r9993, r9808, r9809;
}
{
mul.f16x2 r9996, r9993, r9799;
}
{
sub.f16x2 r9999, r9814, r9815;
}
{
mul.f16x2 r10002, r9999, r9803;
}
{
add.f16x2 r10005, r9996, r10002;
}
{
add.f16x2 %17, r9990, r10005;
}
{
add.f16x2 r10011, r9820, r9821;
}
{
mul.f16x2 r10014, r10011, r9797;
}
{
add.f16x2 r10017, r9823, r10014;
}
{
add.f16x2 r10020, r9826, r9827;
}
{
mul.f16x2 r10023, r10020, r9801;
}
{
add.f16x2 r10026, r10017, r10023;
}
{
sub.f16x2 r10029, r9808, r9809;
}
{
mul.f16x2 r10032, r10029, r9799;
}
{
sub.f16x2 r10035, r9814, r9815;
}
{
mul.f16x2 r10038, r10035, r9803;
}
{
add.f16x2 r10041, r10032, r10038;
}
{
sub.f16x2 %47, r10026, r10041;
}
{
add.f16x2 r10047, r9820, r9821;
}
{
mul.f16x2 r10050, r10047, r9801;
}
{
add.f16x2 r10053, r9823, r10050;
}
{
add.f16x2 r10056, r9826, r9827;
}
{
mul.f16x2 r10059, r10056, r9805;
}
{
add.f16x2 r10062, r10053, r10059;
}
{
sub.f16x2 r10065, r9808, r9809;
}
{
mul.f16x2 r10068, r10065, r9803;
}
{
sub.f16x2 r10071, r9814, r9815;
}
{
mul.f16x2 r10074, r10071, r9806;
}
{
add.f16x2 r10077, r10068, r10074;
}
{
add.f16x2 %27, r10062, r10077;
}
{
add.f16x2 r10083, r9820, r9821;
}
{
mul.f16x2 r10086, r10083, r9801;
}
{
add.f16x2 r10089, r9823, r10086;
}
{
add.f16x2 r10092, r9826, r9827;
}
{
mul.f16x2 r10095, r10092, r9805;
}
{
add.f16x2 r10098, r10089, r10095;
}
{
sub.f16x2 r10101, r9808, r9809;
}
{
mul.f16x2 r10104, r10101, r9803;
}
{
sub.f16x2 r10107, r9814, r9815;
}
{
mul.f16x2 r10110, r10107, r9806;
}
{
add.f16x2 r10113, r10104, r10110;
}
{
sub.f16x2 %37, r10098, r10113;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r10119, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r10120, {low, high};
}
{
neg.f16x2 r10121, r10120;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r10123, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r10124, {low, high};
}
{
neg.f16x2 r10125, r10124;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r10127, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r10128, {low, high};
}
{
add.f16x2 r10129, r10130, r10131;
}
{
add.f16x2 r10132, r10133, r10129;
}
{
add.f16x2 r10135, r10136, r10137;
}
{
add.f16x2 %8, r10132, r10135;
}
{
add.f16x2 r10141, r10142, r10143;
}
{
add.f16x2 r10144, r10145, r10141;
}
{
add.f16x2 r10147, r10148, r10149;
}
{
add.f16x2 %9, r10144, r10147;
}
{
add.f16x2 r10153, r10130, r10131;
}
{
mul.f16x2 r10156, r10153, r10119;
}
{
add.f16x2 r10159, r10133, r10156;
}
{
add.f16x2 r10162, r10136, r10137;
}
{
mul.f16x2 r10165, r10162, r10123;
}
{
add.f16x2 r10168, r10159, r10165;
}
{
sub.f16x2 r10171, r10142, r10143;
}
{
mul.f16x2 r10174, r10171, r10121;
}
{
sub.f16x2 r10177, r10148, r10149;
}
{
mul.f16x2 r10180, r10177, r10125;
}
{
add.f16x2 r10183, r10174, r10180;
}
{
sub.f16x2 %18, r10168, r10183;
}
{
add.f16x2 r10189, r10130, r10131;
}
{
mul.f16x2 r10192, r10189, r10119;
}
{
add.f16x2 r10195, r10133, r10192;
}
{
add.f16x2 r10198, r10136, r10137;
}
{
mul.f16x2 r10201, r10198, r10123;
}
{
add.f16x2 r10204, r10195, r10201;
}
{
sub.f16x2 r10207, r10142, r10143;
}
{
mul.f16x2 r10210, r10207, r10121;
}
{
sub.f16x2 r10213, r10148, r10149;
}
{
mul.f16x2 r10216, r10213, r10125;
}
{
add.f16x2 r10219, r10210, r10216;
}
{
add.f16x2 %48, r10204, r10219;
}
{
add.f16x2 r10225, r10130, r10131;
}
{
mul.f16x2 r10228, r10225, r10123;
}
{
add.f16x2 r10231, r10133, r10228;
}
{
add.f16x2 r10234, r10136, r10137;
}
{
mul.f16x2 r10237, r10234, r10127;
}
{
add.f16x2 r10240, r10231, r10237;
}
{
sub.f16x2 r10243, r10142, r10143;
}
{
mul.f16x2 r10246, r10243, r10125;
}
{
sub.f16x2 r10249, r10148, r10149;
}
{
mul.f16x2 r10252, r10249, r10128;
}
{
add.f16x2 r10255, r10246, r10252;
}
{
sub.f16x2 %28, r10240, r10255;
}
{
add.f16x2 r10261, r10130, r10131;
}
{
mul.f16x2 r10264, r10261, r10123;
}
{
add.f16x2 r10267, r10133, r10264;
}
{
add.f16x2 r10270, r10136, r10137;
}
{
mul.f16x2 r10273, r10270, r10127;
}
{
add.f16x2 r10276, r10267, r10273;
}
{
sub.f16x2 r10279, r10142, r10143;
}
{
mul.f16x2 r10282, r10279, r10125;
}
{
sub.f16x2 r10285, r10148, r10149;
}
{
mul.f16x2 r10288, r10285, r10128;
}
{
add.f16x2 r10291, r10282, r10288;
}
{
add.f16x2 %38, r10276, r10291;
}
{
add.f16x2 r10297, r10142, r10143;
}
{
mul.f16x2 r10300, r10297, r10119;
}
{
add.f16x2 r10303, r10145, r10300;
}
{
add.f16x2 r10306, r10148, r10149;
}
{
mul.f16x2 r10309, r10306, r10123;
}
{
add.f16x2 r10312, r10303, r10309;
}
{
sub.f16x2 r10315, r10130, r10131;
}
{
mul.f16x2 r10318, r10315, r10121;
}
{
sub.f16x2 r10321, r10136, r10137;
}
{
mul.f16x2 r10324, r10321, r10125;
}
{
add.f16x2 r10327, r10318, r10324;
}
{
add.f16x2 %19, r10312, r10327;
}
{
add.f16x2 r10333, r10142, r10143;
}
{
mul.f16x2 r10336, r10333, r10119;
}
{
add.f16x2 r10339, r10145, r10336;
}
{
add.f16x2 r10342, r10148, r10149;
}
{
mul.f16x2 r10345, r10342, r10123;
}
{
add.f16x2 r10348, r10339, r10345;
}
{
sub.f16x2 r10351, r10130, r10131;
}
{
mul.f16x2 r10354, r10351, r10121;
}
{
sub.f16x2 r10357, r10136, r10137;
}
{
mul.f16x2 r10360, r10357, r10125;
}
{
add.f16x2 r10363, r10354, r10360;
}
{
sub.f16x2 %49, r10348, r10363;
}
{
add.f16x2 r10369, r10142, r10143;
}
{
mul.f16x2 r10372, r10369, r10123;
}
{
add.f16x2 r10375, r10145, r10372;
}
{
add.f16x2 r10378, r10148, r10149;
}
{
mul.f16x2 r10381, r10378, r10127;
}
{
add.f16x2 r10384, r10375, r10381;
}
{
sub.f16x2 r10387, r10130, r10131;
}
{
mul.f16x2 r10390, r10387, r10125;
}
{
sub.f16x2 r10393, r10136, r10137;
}
{
mul.f16x2 r10396, r10393, r10128;
}
{
add.f16x2 r10399, r10390, r10396;
}
{
add.f16x2 %29, r10384, r10399;
}
{
add.f16x2 r10405, r10142, r10143;
}
{
mul.f16x2 r10408, r10405, r10123;
}
{
add.f16x2 r10411, r10145, r10408;
}
{
add.f16x2 r10414, r10148, r10149;
}
{
mul.f16x2 r10417, r10414, r10127;
}
{
add.f16x2 r10420, r10411, r10417;
}
{
sub.f16x2 r10423, r10130, r10131;
}
{
mul.f16x2 r10426, r10423, r10125;
}
{
sub.f16x2 r10429, r10136, r10137;
}
{
mul.f16x2 r10432, r10429, r10128;
}
{
add.f16x2 r10435, r10426, r10432;
}
{
sub.f16x2 %39, r10420, r10435;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[13].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1114, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<603>;
.reg .b32 r<10508>;
.reg .b64 rd<6>;
mov.u32 r10442, %50;
mov.u32 r10507, %tid.y;
mad.lo.s32 r10443, r10507, 12500, r10442;
mov.u32 r10444, %tid.x;
mov.f32 f594, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1, {low, high};
}
mov.f32 f596, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2, {low, high};
}
{
neg.f16x2 r3, r2;
}
mov.f32 f590, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r5, {low, high};
}
mov.f32 f592, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r10, {low, high};
}
{
add.f16x2 r11, %67, %54;
}
{
add.f16x2 r14, %74, r11;
}
{
add.f16x2 r17, %81, %65;
}
{
add.f16x2 r20, r14, r17;
}
{
add.f16x2 r23, %94, %79;
}
{
add.f16x2 r26, %51, r23;
}
{
add.f16x2 r29, %57, %92;
}
{
add.f16x2 r32, r26, r29;
}
{
add.f16x2 r35, %67, %54;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %74, r38;
}
{
add.f16x2 r44, %81, %65;
}
{
mul.f16x2 r47, r44, r5;
}
{
add.f16x2 r50, r41, r47;
}
{
sub.f16x2 r53, %94, %79;
}
{
mul.f16x2 r56, r53, r3;
}
{
sub.f16x2 r59, %57, %92;
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
add.f16x2 r71, %67, %54;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %74, r74;
}
{
add.f16x2 r80, %81, %65;
}
{
mul.f16x2 r83, r80, r5;
}
{
add.f16x2 r86, r77, r83;
}
{
sub.f16x2 r89, %94, %79;
}
{
mul.f16x2 r92, r89, r3;
}
{
sub.f16x2 r95, %57, %92;
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
add.f16x2 r107, %67, %54;
}
{
mul.f16x2 r110, r107, r5;
}
{
add.f16x2 r113, %74, r110;
}
{
add.f16x2 r116, %81, %65;
}
{
mul.f16x2 r119, r116, r9;
}
{
add.f16x2 r122, r113, r119;
}
{
sub.f16x2 r125, %94, %79;
}
{
mul.f16x2 r128, r125, r7;
}
{
sub.f16x2 r131, %57, %92;
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
add.f16x2 r143, %67, %54;
}
{
mul.f16x2 r146, r143, r5;
}
{
add.f16x2 r149, %74, r146;
}
{
add.f16x2 r152, %81, %65;
}
{
mul.f16x2 r155, r152, r9;
}
{
add.f16x2 r158, r149, r155;
}
{
sub.f16x2 r161, %94, %79;
}
{
mul.f16x2 r164, r161, r7;
}
{
sub.f16x2 r167, %57, %92;
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
add.f16x2 r179, %94, %79;
}
{
mul.f16x2 r182, r179, r1;
}
{
add.f16x2 r185, %51, r182;
}
{
add.f16x2 r188, %57, %92;
}
{
mul.f16x2 r191, r188, r5;
}
{
add.f16x2 r194, r185, r191;
}
{
sub.f16x2 r197, %67, %54;
}
{
mul.f16x2 r200, r197, r3;
}
{
sub.f16x2 r203, %81, %65;
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
add.f16x2 r215, %94, %79;
}
{
mul.f16x2 r218, r215, r1;
}
{
add.f16x2 r221, %51, r218;
}
{
add.f16x2 r224, %57, %92;
}
{
mul.f16x2 r227, r224, r5;
}
{
add.f16x2 r230, r221, r227;
}
{
sub.f16x2 r233, %67, %54;
}
{
mul.f16x2 r236, r233, r3;
}
{
sub.f16x2 r239, %81, %65;
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
add.f16x2 r251, %94, %79;
}
{
mul.f16x2 r254, r251, r5;
}
{
add.f16x2 r257, %51, r254;
}
{
add.f16x2 r260, %57, %92;
}
{
mul.f16x2 r263, r260, r9;
}
{
add.f16x2 r266, r257, r263;
}
{
sub.f16x2 r269, %67, %54;
}
{
mul.f16x2 r272, r269, r7;
}
{
sub.f16x2 r275, %81, %65;
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
add.f16x2 r287, %94, %79;
}
{
mul.f16x2 r290, r287, r5;
}
{
add.f16x2 r293, %51, r290;
}
{
add.f16x2 r296, %57, %92;
}
{
mul.f16x2 r299, r296, r9;
}
{
add.f16x2 r302, r293, r299;
}
{
sub.f16x2 r305, %67, %54;
}
{
mul.f16x2 r308, r305, r7;
}
{
sub.f16x2 r311, %81, %65;
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
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r323, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r324, {low, high};
}
{
neg.f16x2 r325, r324;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r327, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r328, {low, high};
}
{
neg.f16x2 r329, r328;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r331, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r332, {low, high};
}
{
add.f16x2 r333, %89, %77;
}
{
add.f16x2 r336, %96, r333;
}
{
add.f16x2 r339, %55, %87;
}
{
add.f16x2 r342, r336, r339;
}
{
add.f16x2 r345, %66, %52;
}
{
add.f16x2 r348, %73, r345;
}
{
add.f16x2 r351, %80, %64;
}
{
add.f16x2 r354, r348, r351;
}
{
add.f16x2 r357, %89, %77;
}
{
mul.f16x2 r360, r357, r323;
}
{
add.f16x2 r363, %96, r360;
}
{
add.f16x2 r366, %55, %87;
}
{
mul.f16x2 r369, r366, r327;
}
{
add.f16x2 r372, r363, r369;
}
{
sub.f16x2 r375, %66, %52;
}
{
mul.f16x2 r378, r375, r325;
}
{
sub.f16x2 r381, %80, %64;
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
add.f16x2 r393, %89, %77;
}
{
mul.f16x2 r396, r393, r323;
}
{
add.f16x2 r399, %96, r396;
}
{
add.f16x2 r402, %55, %87;
}
{
mul.f16x2 r405, r402, r327;
}
{
add.f16x2 r408, r399, r405;
}
{
sub.f16x2 r411, %66, %52;
}
{
mul.f16x2 r414, r411, r325;
}
{
sub.f16x2 r417, %80, %64;
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
add.f16x2 r429, %89, %77;
}
{
mul.f16x2 r432, r429, r327;
}
{
add.f16x2 r435, %96, r432;
}
{
add.f16x2 r438, %55, %87;
}
{
mul.f16x2 r441, r438, r331;
}
{
add.f16x2 r444, r435, r441;
}
{
sub.f16x2 r447, %66, %52;
}
{
mul.f16x2 r450, r447, r329;
}
{
sub.f16x2 r453, %80, %64;
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
add.f16x2 r465, %89, %77;
}
{
mul.f16x2 r468, r465, r327;
}
{
add.f16x2 r471, %96, r468;
}
{
add.f16x2 r474, %55, %87;
}
{
mul.f16x2 r477, r474, r331;
}
{
add.f16x2 r480, r471, r477;
}
{
sub.f16x2 r483, %66, %52;
}
{
mul.f16x2 r486, r483, r329;
}
{
sub.f16x2 r489, %80, %64;
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
add.f16x2 r501, %66, %52;
}
{
mul.f16x2 r504, r501, r323;
}
{
add.f16x2 r507, %73, r504;
}
{
add.f16x2 r510, %80, %64;
}
{
mul.f16x2 r513, r510, r327;
}
{
add.f16x2 r516, r507, r513;
}
{
sub.f16x2 r519, %89, %77;
}
{
mul.f16x2 r522, r519, r325;
}
{
sub.f16x2 r525, %55, %87;
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
add.f16x2 r537, %66, %52;
}
{
mul.f16x2 r540, r537, r323;
}
{
add.f16x2 r543, %73, r540;
}
{
add.f16x2 r546, %80, %64;
}
{
mul.f16x2 r549, r546, r327;
}
{
add.f16x2 r552, r543, r549;
}
{
sub.f16x2 r555, %89, %77;
}
{
mul.f16x2 r558, r555, r325;
}
{
sub.f16x2 r561, %55, %87;
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
add.f16x2 r573, %66, %52;
}
{
mul.f16x2 r576, r573, r327;
}
{
add.f16x2 r579, %73, r576;
}
{
add.f16x2 r582, %80, %64;
}
{
mul.f16x2 r585, r582, r331;
}
{
add.f16x2 r588, r579, r585;
}
{
sub.f16x2 r591, %89, %77;
}
{
mul.f16x2 r594, r591, r329;
}
{
sub.f16x2 r597, %55, %87;
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
add.f16x2 r609, %66, %52;
}
{
mul.f16x2 r612, r609, r327;
}
{
add.f16x2 r615, %73, r612;
}
{
add.f16x2 r618, %80, %64;
}
{
mul.f16x2 r621, r618, r331;
}
{
add.f16x2 r624, r615, r621;
}
{
sub.f16x2 r627, %89, %77;
}
{
mul.f16x2 r630, r627, r329;
}
{
sub.f16x2 r633, %55, %87;
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
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r645, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r646, {low, high};
}
{
neg.f16x2 r647, r646;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r649, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r650, {low, high};
}
{
neg.f16x2 r651, r650;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r653, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r654, {low, high};
}
{
add.f16x2 r655, %62, %99;
}
{
add.f16x2 r658, %69, r655;
}
{
add.f16x2 r661, %78, %60;
}
{
add.f16x2 r664, r658, r661;
}
{
add.f16x2 r667, %88, %75;
}
{
add.f16x2 r670, %95, r667;
}
{
add.f16x2 r673, %53, %86;
}
{
add.f16x2 r676, r670, r673;
}
{
add.f16x2 r679, %62, %99;
}
{
mul.f16x2 r682, r679, r645;
}
{
add.f16x2 r685, %69, r682;
}
{
add.f16x2 r688, %78, %60;
}
{
mul.f16x2 r691, r688, r649;
}
{
add.f16x2 r694, r685, r691;
}
{
sub.f16x2 r697, %88, %75;
}
{
mul.f16x2 r700, r697, r647;
}
{
sub.f16x2 r703, %53, %86;
}
{
mul.f16x2 r706, r703, r651;
}
{
add.f16x2 r709, r700, r706;
}
{
sub.f16x2 r712, r694, r709;
}
{
add.f16x2 r715, %62, %99;
}
{
mul.f16x2 r718, r715, r645;
}
{
add.f16x2 r721, %69, r718;
}
{
add.f16x2 r724, %78, %60;
}
{
mul.f16x2 r727, r724, r649;
}
{
add.f16x2 r730, r721, r727;
}
{
sub.f16x2 r733, %88, %75;
}
{
mul.f16x2 r736, r733, r647;
}
{
sub.f16x2 r739, %53, %86;
}
{
mul.f16x2 r742, r739, r651;
}
{
add.f16x2 r745, r736, r742;
}
{
add.f16x2 r748, r730, r745;
}
{
add.f16x2 r751, %62, %99;
}
{
mul.f16x2 r754, r751, r649;
}
{
add.f16x2 r757, %69, r754;
}
{
add.f16x2 r760, %78, %60;
}
{
mul.f16x2 r763, r760, r653;
}
{
add.f16x2 r766, r757, r763;
}
{
sub.f16x2 r769, %88, %75;
}
{
mul.f16x2 r772, r769, r651;
}
{
sub.f16x2 r775, %53, %86;
}
{
mul.f16x2 r778, r775, r654;
}
{
add.f16x2 r781, r772, r778;
}
{
sub.f16x2 r784, r766, r781;
}
{
add.f16x2 r787, %62, %99;
}
{
mul.f16x2 r790, r787, r649;
}
{
add.f16x2 r793, %69, r790;
}
{
add.f16x2 r796, %78, %60;
}
{
mul.f16x2 r799, r796, r653;
}
{
add.f16x2 r802, r793, r799;
}
{
sub.f16x2 r805, %88, %75;
}
{
mul.f16x2 r808, r805, r651;
}
{
sub.f16x2 r811, %53, %86;
}
{
mul.f16x2 r814, r811, r654;
}
{
add.f16x2 r817, r808, r814;
}
{
add.f16x2 r820, r802, r817;
}
{
add.f16x2 r823, %88, %75;
}
{
mul.f16x2 r826, r823, r645;
}
{
add.f16x2 r829, %95, r826;
}
{
add.f16x2 r832, %53, %86;
}
{
mul.f16x2 r835, r832, r649;
}
{
add.f16x2 r838, r829, r835;
}
{
sub.f16x2 r841, %62, %99;
}
{
mul.f16x2 r844, r841, r647;
}
{
sub.f16x2 r847, %78, %60;
}
{
mul.f16x2 r850, r847, r651;
}
{
add.f16x2 r853, r844, r850;
}
{
add.f16x2 r856, r838, r853;
}
{
add.f16x2 r859, %88, %75;
}
{
mul.f16x2 r862, r859, r645;
}
{
add.f16x2 r865, %95, r862;
}
{
add.f16x2 r868, %53, %86;
}
{
mul.f16x2 r871, r868, r649;
}
{
add.f16x2 r874, r865, r871;
}
{
sub.f16x2 r877, %62, %99;
}
{
mul.f16x2 r880, r877, r647;
}
{
sub.f16x2 r883, %78, %60;
}
{
mul.f16x2 r886, r883, r651;
}
{
add.f16x2 r889, r880, r886;
}
{
sub.f16x2 r892, r874, r889;
}
{
add.f16x2 r895, %88, %75;
}
{
mul.f16x2 r898, r895, r649;
}
{
add.f16x2 r901, %95, r898;
}
{
add.f16x2 r904, %53, %86;
}
{
mul.f16x2 r907, r904, r653;
}
{
add.f16x2 r910, r901, r907;
}
{
sub.f16x2 r913, %62, %99;
}
{
mul.f16x2 r916, r913, r651;
}
{
sub.f16x2 r919, %78, %60;
}
{
mul.f16x2 r922, r919, r654;
}
{
add.f16x2 r925, r916, r922;
}
{
add.f16x2 r928, r910, r925;
}
{
add.f16x2 r931, %88, %75;
}
{
mul.f16x2 r934, r931, r649;
}
{
add.f16x2 r937, %95, r934;
}
{
add.f16x2 r940, %53, %86;
}
{
mul.f16x2 r943, r940, r653;
}
{
add.f16x2 r946, r937, r943;
}
{
sub.f16x2 r949, %62, %99;
}
{
mul.f16x2 r952, r949, r651;
}
{
sub.f16x2 r955, %78, %60;
}
{
mul.f16x2 r958, r955, r654;
}
{
add.f16x2 r961, r952, r958;
}
{
sub.f16x2 r964, r946, r961;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r967, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r968, {low, high};
}
{
neg.f16x2 r969, r968;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r971, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r972, {low, high};
}
{
neg.f16x2 r973, r972;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r975, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r976, {low, high};
}
{
add.f16x2 r977, %85, %71;
}
{
add.f16x2 r980, %91, r977;
}
{
add.f16x2 r983, %100, %83;
}
{
add.f16x2 r986, r980, r983;
}
{
add.f16x2 r989, %61, %97;
}
{
add.f16x2 r992, %68, r989;
}
{
add.f16x2 r995, %76, %59;
}
{
add.f16x2 r998, r992, r995;
}
{
add.f16x2 r1001, %85, %71;
}
{
mul.f16x2 r1004, r1001, r967;
}
{
add.f16x2 r1007, %91, r1004;
}
{
add.f16x2 r1010, %100, %83;
}
{
mul.f16x2 r1013, r1010, r971;
}
{
add.f16x2 r1016, r1007, r1013;
}
{
sub.f16x2 r1019, %61, %97;
}
{
mul.f16x2 r1022, r1019, r969;
}
{
sub.f16x2 r1025, %76, %59;
}
{
mul.f16x2 r1028, r1025, r973;
}
{
add.f16x2 r1031, r1022, r1028;
}
{
sub.f16x2 r1034, r1016, r1031;
}
{
add.f16x2 r1037, %85, %71;
}
{
mul.f16x2 r1040, r1037, r967;
}
{
add.f16x2 r1043, %91, r1040;
}
{
add.f16x2 r1046, %100, %83;
}
{
mul.f16x2 r1049, r1046, r971;
}
{
add.f16x2 r1052, r1043, r1049;
}
{
sub.f16x2 r1055, %61, %97;
}
{
mul.f16x2 r1058, r1055, r969;
}
{
sub.f16x2 r1061, %76, %59;
}
{
mul.f16x2 r1064, r1061, r973;
}
{
add.f16x2 r1067, r1058, r1064;
}
{
add.f16x2 r1070, r1052, r1067;
}
{
add.f16x2 r1073, %85, %71;
}
{
mul.f16x2 r1076, r1073, r971;
}
{
add.f16x2 r1079, %91, r1076;
}
{
add.f16x2 r1082, %100, %83;
}
{
mul.f16x2 r1085, r1082, r975;
}
{
add.f16x2 r1088, r1079, r1085;
}
{
sub.f16x2 r1091, %61, %97;
}
{
mul.f16x2 r1094, r1091, r973;
}
{
sub.f16x2 r1097, %76, %59;
}
{
mul.f16x2 r1100, r1097, r976;
}
{
add.f16x2 r1103, r1094, r1100;
}
{
sub.f16x2 r1106, r1088, r1103;
}
{
add.f16x2 r1109, %85, %71;
}
{
mul.f16x2 r1112, r1109, r971;
}
{
add.f16x2 r1115, %91, r1112;
}
{
add.f16x2 r1118, %100, %83;
}
{
mul.f16x2 r1121, r1118, r975;
}
{
add.f16x2 r1124, r1115, r1121;
}
{
sub.f16x2 r1127, %61, %97;
}
{
mul.f16x2 r1130, r1127, r973;
}
{
sub.f16x2 r1133, %76, %59;
}
{
mul.f16x2 r1136, r1133, r976;
}
{
add.f16x2 r1139, r1130, r1136;
}
{
add.f16x2 r1142, r1124, r1139;
}
{
add.f16x2 r1145, %61, %97;
}
{
mul.f16x2 r1148, r1145, r967;
}
{
add.f16x2 r1151, %68, r1148;
}
{
add.f16x2 r1154, %76, %59;
}
{
mul.f16x2 r1157, r1154, r971;
}
{
add.f16x2 r1160, r1151, r1157;
}
{
sub.f16x2 r1163, %85, %71;
}
{
mul.f16x2 r1166, r1163, r969;
}
{
sub.f16x2 r1169, %100, %83;
}
{
mul.f16x2 r1172, r1169, r973;
}
{
add.f16x2 r1175, r1166, r1172;
}
{
add.f16x2 r1178, r1160, r1175;
}
{
add.f16x2 r1181, %61, %97;
}
{
mul.f16x2 r1184, r1181, r967;
}
{
add.f16x2 r1187, %68, r1184;
}
{
add.f16x2 r1190, %76, %59;
}
{
mul.f16x2 r1193, r1190, r971;
}
{
add.f16x2 r1196, r1187, r1193;
}
{
sub.f16x2 r1199, %85, %71;
}
{
mul.f16x2 r1202, r1199, r969;
}
{
sub.f16x2 r1205, %100, %83;
}
{
mul.f16x2 r1208, r1205, r973;
}
{
add.f16x2 r1211, r1202, r1208;
}
{
sub.f16x2 r1214, r1196, r1211;
}
{
add.f16x2 r1217, %61, %97;
}
{
mul.f16x2 r1220, r1217, r971;
}
{
add.f16x2 r1223, %68, r1220;
}
{
add.f16x2 r1226, %76, %59;
}
{
mul.f16x2 r1229, r1226, r975;
}
{
add.f16x2 r1232, r1223, r1229;
}
{
sub.f16x2 r1235, %85, %71;
}
{
mul.f16x2 r1238, r1235, r973;
}
{
sub.f16x2 r1241, %100, %83;
}
{
mul.f16x2 r1244, r1241, r976;
}
{
add.f16x2 r1247, r1238, r1244;
}
{
add.f16x2 r1250, r1232, r1247;
}
{
add.f16x2 r1253, %61, %97;
}
{
mul.f16x2 r1256, r1253, r971;
}
{
add.f16x2 r1259, %68, r1256;
}
{
add.f16x2 r1262, %76, %59;
}
{
mul.f16x2 r1265, r1262, r975;
}
{
add.f16x2 r1268, r1259, r1265;
}
{
sub.f16x2 r1271, %85, %71;
}
{
mul.f16x2 r1274, r1271, r973;
}
{
sub.f16x2 r1277, %100, %83;
}
{
mul.f16x2 r1280, r1277, r976;
}
{
add.f16x2 r1283, r1274, r1280;
}
{
sub.f16x2 r1286, r1268, r1283;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1289, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r1290, {low, high};
}
{
neg.f16x2 r1291, r1290;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r1293, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r1294, {low, high};
}
{
neg.f16x2 r1295, r1294;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r1298, {low, high};
}
{
add.f16x2 r1299, %58, %93;
}
{
add.f16x2 r1302, %63, r1299;
}
{
add.f16x2 r1305, %72, %56;
}
{
add.f16x2 r1308, r1302, r1305;
}
{
add.f16x2 r1311, %84, %70;
}
{
add.f16x2 r1314, %90, r1311;
}
{
add.f16x2 r1317, %98, %82;
}
{
add.f16x2 r1320, r1314, r1317;
}
{
add.f16x2 r1323, %58, %93;
}
{
mul.f16x2 r1326, r1323, r1289;
}
{
add.f16x2 r1329, %63, r1326;
}
{
add.f16x2 r1332, %72, %56;
}
{
mul.f16x2 r1335, r1332, r1293;
}
{
add.f16x2 r1338, r1329, r1335;
}
{
sub.f16x2 r1341, %84, %70;
}
{
mul.f16x2 r1344, r1341, r1291;
}
{
sub.f16x2 r1347, %98, %82;
}
{
mul.f16x2 r1350, r1347, r1295;
}
{
add.f16x2 r1353, r1344, r1350;
}
{
sub.f16x2 r1356, r1338, r1353;
}
{
add.f16x2 r1359, %58, %93;
}
{
mul.f16x2 r1362, r1359, r1289;
}
{
add.f16x2 r1365, %63, r1362;
}
{
add.f16x2 r1368, %72, %56;
}
{
mul.f16x2 r1371, r1368, r1293;
}
{
add.f16x2 r1374, r1365, r1371;
}
{
sub.f16x2 r1377, %84, %70;
}
{
mul.f16x2 r1380, r1377, r1291;
}
{
sub.f16x2 r1383, %98, %82;
}
{
mul.f16x2 r1386, r1383, r1295;
}
{
add.f16x2 r1389, r1380, r1386;
}
{
add.f16x2 r1392, r1374, r1389;
}
{
add.f16x2 r1395, %58, %93;
}
{
mul.f16x2 r1398, r1395, r1293;
}
{
add.f16x2 r1401, %63, r1398;
}
{
add.f16x2 r1404, %72, %56;
}
{
mul.f16x2 r1407, r1404, r1297;
}
{
add.f16x2 r1410, r1401, r1407;
}
{
sub.f16x2 r1413, %84, %70;
}
{
mul.f16x2 r1416, r1413, r1295;
}
{
sub.f16x2 r1419, %98, %82;
}
{
mul.f16x2 r1422, r1419, r1298;
}
{
add.f16x2 r1425, r1416, r1422;
}
{
sub.f16x2 r1428, r1410, r1425;
}
{
add.f16x2 r1431, %58, %93;
}
{
mul.f16x2 r1434, r1431, r1293;
}
{
add.f16x2 r1437, %63, r1434;
}
{
add.f16x2 r1440, %72, %56;
}
{
mul.f16x2 r1443, r1440, r1297;
}
{
add.f16x2 r1446, r1437, r1443;
}
{
sub.f16x2 r1449, %84, %70;
}
{
mul.f16x2 r1452, r1449, r1295;
}
{
sub.f16x2 r1455, %98, %82;
}
{
mul.f16x2 r1458, r1455, r1298;
}
{
add.f16x2 r1461, r1452, r1458;
}
{
add.f16x2 r1464, r1446, r1461;
}
{
add.f16x2 r1467, %84, %70;
}
{
mul.f16x2 r1470, r1467, r1289;
}
{
add.f16x2 r1473, %90, r1470;
}
{
add.f16x2 r1476, %98, %82;
}
{
mul.f16x2 r1479, r1476, r1293;
}
{
add.f16x2 r1482, r1473, r1479;
}
{
sub.f16x2 r1485, %58, %93;
}
{
mul.f16x2 r1488, r1485, r1291;
}
{
sub.f16x2 r1491, %72, %56;
}
{
mul.f16x2 r1494, r1491, r1295;
}
{
add.f16x2 r1497, r1488, r1494;
}
{
add.f16x2 r1500, r1482, r1497;
}
{
add.f16x2 r1503, %84, %70;
}
{
mul.f16x2 r1506, r1503, r1289;
}
{
add.f16x2 r1509, %90, r1506;
}
{
add.f16x2 r1512, %98, %82;
}
{
mul.f16x2 r1515, r1512, r1293;
}
{
add.f16x2 r1518, r1509, r1515;
}
{
sub.f16x2 r1521, %58, %93;
}
{
mul.f16x2 r1524, r1521, r1291;
}
{
sub.f16x2 r1527, %72, %56;
}
{
mul.f16x2 r1530, r1527, r1295;
}
{
add.f16x2 r1533, r1524, r1530;
}
{
sub.f16x2 r1536, r1518, r1533;
}
{
add.f16x2 r1539, %84, %70;
}
{
mul.f16x2 r1542, r1539, r1293;
}
{
add.f16x2 r1545, %90, r1542;
}
{
add.f16x2 r1548, %98, %82;
}
{
mul.f16x2 r1551, r1548, r1297;
}
{
add.f16x2 r1554, r1545, r1551;
}
{
sub.f16x2 r1557, %58, %93;
}
{
mul.f16x2 r1560, r1557, r1295;
}
{
sub.f16x2 r1563, %72, %56;
}
{
mul.f16x2 r1566, r1563, r1298;
}
{
add.f16x2 r1569, r1560, r1566;
}
{
add.f16x2 r1572, r1554, r1569;
}
{
add.f16x2 r1575, %84, %70;
}
{
mul.f16x2 r1578, r1575, r1293;
}
{
add.f16x2 r1581, %90, r1578;
}
{
add.f16x2 r1584, %98, %82;
}
{
mul.f16x2 r1587, r1584, r1297;
}
{
add.f16x2 r1590, r1581, r1587;
}
{
sub.f16x2 r1593, %58, %93;
}
{
mul.f16x2 r1596, r1593, r1295;
}
{
sub.f16x2 r1599, %72, %56;
}
{
mul.f16x2 r1602, r1599, r1298;
}
{
add.f16x2 r1605, r1596, r1602;
}
{
sub.f16x2 r1608, r1590, r1605;
}
mov.f32 f330, 0f3F77F511;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r1611, {low, high};
}
mov.f32 f332, 0f3E7EA890;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f332;
cvt.rn.f16.f32 high, f332;
mov.b32 r1612, {low, high};
}
mov.f32 f334, 0f3F6055A2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f334;
cvt.rn.f16.f32 high, f334;
mov.b32 r1613, {low, high};
}
mov.f32 f336, 0f3EF6A86B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f336;
cvt.rn.f16.f32 high, f336;
mov.b32 r1614, {low, high};
}
mov.f32 f338, 0f3F3A9DB0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f338;
cvt.rn.f16.f32 high, f338;
mov.b32 r1615, {low, high};
}
mov.f32 f340, 0f3F2F3E7B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r1616, {low, high};
}
mov.f32 f342, 0f3F092BF2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f342;
cvt.rn.f16.f32 high, f342;
mov.b32 r1617, {low, high};
}
mov.f32 f344, 0f3F5825E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f344;
cvt.rn.f16.f32 high, f344;
mov.b32 r1618, {low, high};
}
mov.f32 f350, 0f3D809851;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f350;
cvt.rn.f16.f32 high, f350;
mov.b32 r1621, {low, high};
}
mov.f32 f352, 0f3F7F7EAE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r1622, {low, high};
}
mov.f32 f358, 0fBED9FFBE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f358;
cvt.rn.f16.f32 high, f358;
mov.b32 r1625, {low, high};
}
mov.f32 f360, 0f3F67A2BF;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f360;
cvt.rn.f16.f32 high, f360;
mov.b32 r1626, {low, high};
}
mov.f32 f390, 0fBF232E38;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f390;
cvt.rn.f16.f32 high, f390;
mov.b32 r1627, {low, high};
}
mov.f32 f364, 0f3F45405B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f364;
cvt.rn.f16.f32 high, f364;
mov.b32 r1628, {low, high};
}
mov.f32 f374, 0fBF7DFB3B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f374;
cvt.rn.f16.f32 high, f374;
mov.b32 r1633, {low, high};
}
mov.f32 f376, 0f3E00575B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r1634, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f390;
cvt.rn.f16.f32 high, f390;
mov.b32 r1641, {low, high};
}
mov.f32 f392, 0fBF45405B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f392;
cvt.rn.f16.f32 high, f392;
mov.b32 r1642, {low, high};
}
{
mul.f16x2 r1659, r390, r1611;
}
{
mul.f16x2 r1662, r534, r1612;
}
{
sub.f16x2 r1665, r1659, r1662;
}
{
mul.f16x2 r1668, r390, r1612;
}
{
fma.rn.f16x2 r1671, r534, r1611, r1668;
}
{
mul.f16x2 r1675, r712, r1613;
}
{
mul.f16x2 r1678, r856, r1614;
}
{
sub.f16x2 r1681, r1675, r1678;
}
{
mul.f16x2 r1684, r712, r1614;
}
{
fma.rn.f16x2 r1687, r856, r1613, r1684;
}
{
mul.f16x2 r1691, r1034, r1615;
}
{
mul.f16x2 r1694, r1178, r1616;
}
{
sub.f16x2 r1697, r1691, r1694;
}
{
mul.f16x2 r1700, r1034, r1616;
}
{
fma.rn.f16x2 r1703, r1178, r1615, r1700;
}
{
mul.f16x2 r1707, r1356, r1617;
}
{
mul.f16x2 r1710, r1500, r1618;
}
{
sub.f16x2 r1713, r1707, r1710;
}
{
mul.f16x2 r1716, r1356, r1618;
}
{
fma.rn.f16x2 r1719, r1500, r1617, r1716;
}
{
mul.f16x2 r1723, r462, r1613;
}
{
mul.f16x2 r1726, r606, r1614;
}
{
sub.f16x2 r1729, r1723, r1726;
}
{
mul.f16x2 r1732, r462, r1614;
}
{
fma.rn.f16x2 r1735, r606, r1613, r1732;
}
{
mul.f16x2 r1739, r784, r1617;
}
{
mul.f16x2 r1742, r928, r1618;
}
{
sub.f16x2 r1745, r1739, r1742;
}
{
mul.f16x2 r1748, r784, r1618;
}
{
fma.rn.f16x2 r1751, r928, r1617, r1748;
}
{
mul.f16x2 r1755, r1106, r1621;
}
{
mul.f16x2 r1758, r1250, r1622;
}
{
sub.f16x2 r1761, r1755, r1758;
}
{
mul.f16x2 r1764, r1106, r1622;
}
{
fma.rn.f16x2 r1767, r1250, r1621, r1764;
}
{
mul.f16x2 r1771, r1428, r1625;
}
{
mul.f16x2 r1774, r1572, r1626;
}
{
sub.f16x2 r1777, r1771, r1774;
}
{
mul.f16x2 r1780, r1428, r1626;
}
{
fma.rn.f16x2 r1783, r1572, r1625, r1780;
}
{
mul.f16x2 r1787, r498, r1615;
}
{
mul.f16x2 r1790, r642, r1616;
}
{
sub.f16x2 r1793, r1787, r1790;
}
{
mul.f16x2 r1796, r498, r1616;
}
{
fma.rn.f16x2 r1799, r642, r1615, r1796;
}
{
mul.f16x2 r1803, r820, r1621;
}
{
mul.f16x2 r1806, r964, r1622;
}
{
sub.f16x2 r1809, r1803, r1806;
}
{
mul.f16x2 r1812, r820, r1622;
}
{
fma.rn.f16x2 r1815, r964, r1621, r1812;
}
{
mul.f16x2 r1819, r1142, r1627;
}
{
mul.f16x2 r1822, r1286, r1628;
}
{
sub.f16x2 r1825, r1819, r1822;
}
{
mul.f16x2 r1828, r1142, r1628;
}
{
fma.rn.f16x2 r1831, r1286, r1627, r1828;
}
{
mul.f16x2 r1835, r1464, r1633;
}
{
mul.f16x2 r1838, r1608, r1634;
}
{
sub.f16x2 r1841, r1835, r1838;
}
{
mul.f16x2 r1844, r1464, r1634;
}
{
fma.rn.f16x2 r1847, r1608, r1633, r1844;
}
{
mul.f16x2 r1851, r426, r1617;
}
{
mul.f16x2 r1854, r570, r1618;
}
{
sub.f16x2 r1857, r1851, r1854;
}
{
mul.f16x2 r1860, r426, r1618;
}
{
fma.rn.f16x2 r1863, r570, r1617, r1860;
}
{
mul.f16x2 r1867, r748, r1625;
}
{
mul.f16x2 r1870, r892, r1626;
}
{
sub.f16x2 r1873, r1867, r1870;
}
{
mul.f16x2 r1876, r748, r1626;
}
{
fma.rn.f16x2 r1879, r892, r1625, r1876;
}
{
mul.f16x2 r1883, r1070, r1633;
}
{
mul.f16x2 r1886, r1214, r1634;
}
{
sub.f16x2 r1889, r1883, r1886;
}
{
mul.f16x2 r1892, r1070, r1634;
}
{
fma.rn.f16x2 r1895, r1214, r1633, r1892;
}
{
mul.f16x2 r1899, r1392, r1641;
}
{
mul.f16x2 r1902, r1536, r1642;
}
{
sub.f16x2 r1905, r1899, r1902;
}
{
mul.f16x2 r1908, r1392, r1642;
}
{
fma.rn.f16x2 r1911, r1536, r1641, r1908;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1915, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r1916, {low, high};
}
{
neg.f16x2 r1917, r1916;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r1919, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r1920, {low, high};
}
{
neg.f16x2 r1921, r1920;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1923, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r1924, {low, high};
}
{
add.f16x2 r1925, r342, r1308;
}
{
add.f16x2 r1928, r20, r1925;
}
{
add.f16x2 r1931, r664, r986;
}
{
add.f16x2 r1934, r1928, r1931;
}
{
add.f16x2 r1937, r354, r1320;
}
{
add.f16x2 r1940, r32, r1937;
}
{
add.f16x2 r1943, r676, r998;
}
{
add.f16x2 r1946, r1940, r1943;
}
{
add.f16x2 r1949, r342, r1308;
}
{
mul.f16x2 r1952, r1949, r1915;
}
{
add.f16x2 r1955, r20, r1952;
}
{
add.f16x2 r1958, r664, r986;
}
{
mul.f16x2 r1961, r1958, r1919;
}
{
add.f16x2 r1964, r1955, r1961;
}
{
sub.f16x2 r1967, r354, r1320;
}
{
mul.f16x2 r1970, r1967, r1917;
}
{
sub.f16x2 r1973, r676, r998;
}
{
mul.f16x2 r1976, r1973, r1921;
}
{
add.f16x2 r1979, r1970, r1976;
}
{
sub.f16x2 r1982, r1964, r1979;
}
{
add.f16x2 r1985, r342, r1308;
}
{
mul.f16x2 r1988, r1985, r1915;
}
{
add.f16x2 r1991, r20, r1988;
}
{
add.f16x2 r1994, r664, r986;
}
{
mul.f16x2 r1997, r1994, r1919;
}
{
add.f16x2 r2000, r1991, r1997;
}
{
sub.f16x2 r2003, r354, r1320;
}
{
mul.f16x2 r2006, r2003, r1917;
}
{
sub.f16x2 r2009, r676, r998;
}
{
mul.f16x2 r2012, r2009, r1921;
}
{
add.f16x2 r2015, r2006, r2012;
}
{
add.f16x2 r2018, r2000, r2015;
}
{
add.f16x2 r2021, r342, r1308;
}
{
mul.f16x2 r2024, r2021, r1919;
}
{
add.f16x2 r2027, r20, r2024;
}
{
add.f16x2 r2030, r664, r986;
}
{
mul.f16x2 r2033, r2030, r1923;
}
{
add.f16x2 r2036, r2027, r2033;
}
{
sub.f16x2 r2039, r354, r1320;
}
{
mul.f16x2 r2042, r2039, r1921;
}
{
sub.f16x2 r2045, r676, r998;
}
{
mul.f16x2 r2048, r2045, r1924;
}
{
add.f16x2 r2051, r2042, r2048;
}
{
sub.f16x2 r2054, r2036, r2051;
}
{
add.f16x2 r2057, r342, r1308;
}
{
mul.f16x2 r2060, r2057, r1919;
}
{
add.f16x2 r2063, r20, r2060;
}
{
add.f16x2 r2066, r664, r986;
}
{
mul.f16x2 r2069, r2066, r1923;
}
{
add.f16x2 r2072, r2063, r2069;
}
{
sub.f16x2 r2075, r354, r1320;
}
{
mul.f16x2 r2078, r2075, r1921;
}
{
sub.f16x2 r2081, r676, r998;
}
{
mul.f16x2 r2084, r2081, r1924;
}
{
add.f16x2 r2087, r2078, r2084;
}
{
add.f16x2 r2090, r2072, r2087;
}
{
add.f16x2 r2093, r354, r1320;
}
{
mul.f16x2 r2096, r2093, r1915;
}
{
add.f16x2 r2099, r32, r2096;
}
{
add.f16x2 r2102, r676, r998;
}
{
mul.f16x2 r2105, r2102, r1919;
}
{
add.f16x2 r2108, r2099, r2105;
}
{
sub.f16x2 r2111, r342, r1308;
}
{
mul.f16x2 r2114, r2111, r1917;
}
{
sub.f16x2 r2117, r664, r986;
}
{
mul.f16x2 r2120, r2117, r1921;
}
{
add.f16x2 r2123, r2114, r2120;
}
{
add.f16x2 r2126, r2108, r2123;
}
{
add.f16x2 r2129, r354, r1320;
}
{
mul.f16x2 r2132, r2129, r1915;
}
{
add.f16x2 r2135, r32, r2132;
}
{
add.f16x2 r2138, r676, r998;
}
{
mul.f16x2 r2141, r2138, r1919;
}
{
add.f16x2 r2144, r2135, r2141;
}
{
sub.f16x2 r2147, r342, r1308;
}
{
mul.f16x2 r2150, r2147, r1917;
}
{
sub.f16x2 r2153, r664, r986;
}
{
mul.f16x2 r2156, r2153, r1921;
}
{
add.f16x2 r2159, r2150, r2156;
}
{
sub.f16x2 r2162, r2144, r2159;
}
{
add.f16x2 r2165, r354, r1320;
}
{
mul.f16x2 r2168, r2165, r1919;
}
{
add.f16x2 r2171, r32, r2168;
}
{
add.f16x2 r2174, r676, r998;
}
{
mul.f16x2 r2177, r2174, r1923;
}
{
add.f16x2 r2180, r2171, r2177;
}
{
sub.f16x2 r2183, r342, r1308;
}
{
mul.f16x2 r2186, r2183, r1921;
}
{
sub.f16x2 r2189, r664, r986;
}
{
mul.f16x2 r2192, r2189, r1924;
}
{
add.f16x2 r2195, r2186, r2192;
}
{
add.f16x2 r2198, r2180, r2195;
}
{
add.f16x2 r2201, r354, r1320;
}
{
mul.f16x2 r2204, r2201, r1919;
}
{
add.f16x2 r2207, r32, r2204;
}
{
add.f16x2 r2210, r676, r998;
}
{
mul.f16x2 r2213, r2210, r1923;
}
{
add.f16x2 r2216, r2207, r2213;
}
{
sub.f16x2 r2219, r342, r1308;
}
{
mul.f16x2 r2222, r2219, r1921;
}
{
sub.f16x2 r2225, r664, r986;
}
{
mul.f16x2 r2228, r2225, r1924;
}
{
add.f16x2 r2231, r2222, r2228;
}
{
sub.f16x2 r2234, r2216, r2231;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2237, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2238, {low, high};
}
{
neg.f16x2 r2239, r2238;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r2241, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r2242, {low, high};
}
{
neg.f16x2 r2243, r2242;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2245, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2246, {low, high};
}
{
add.f16x2 r2247, r1665, r1713;
}
{
add.f16x2 r2250, r68, r2247;
}
{
add.f16x2 r2253, r1681, r1697;
}
{
add.f16x2 r2256, r2250, r2253;
}
{
add.f16x2 r2259, r1671, r1719;
}
{
add.f16x2 r2262, r212, r2259;
}
{
add.f16x2 r2265, r1687, r1703;
}
{
add.f16x2 r2268, r2262, r2265;
}
{
add.f16x2 r2271, r1665, r1713;
}
{
mul.f16x2 r2274, r2271, r2237;
}
{
add.f16x2 r2277, r68, r2274;
}
{
add.f16x2 r2280, r1681, r1697;
}
{
mul.f16x2 r2283, r2280, r2241;
}
{
add.f16x2 r2286, r2277, r2283;
}
{
sub.f16x2 r2289, r1671, r1719;
}
{
mul.f16x2 r2292, r2289, r2239;
}
{
sub.f16x2 r2295, r1687, r1703;
}
{
mul.f16x2 r2298, r2295, r2243;
}
{
add.f16x2 r2301, r2292, r2298;
}
{
sub.f16x2 r2304, r2286, r2301;
}
{
add.f16x2 r2307, r1665, r1713;
}
{
mul.f16x2 r2310, r2307, r2237;
}
{
add.f16x2 r2313, r68, r2310;
}
{
add.f16x2 r2316, r1681, r1697;
}
{
mul.f16x2 r2319, r2316, r2241;
}
{
add.f16x2 r2322, r2313, r2319;
}
{
sub.f16x2 r2325, r1671, r1719;
}
{
mul.f16x2 r2328, r2325, r2239;
}
{
sub.f16x2 r2331, r1687, r1703;
}
{
mul.f16x2 r2334, r2331, r2243;
}
{
add.f16x2 r2337, r2328, r2334;
}
{
add.f16x2 r2340, r2322, r2337;
}
{
add.f16x2 r2343, r1665, r1713;
}
{
mul.f16x2 r2346, r2343, r2241;
}
{
add.f16x2 r2349, r68, r2346;
}
{
add.f16x2 r2352, r1681, r1697;
}
{
mul.f16x2 r2355, r2352, r2245;
}
{
add.f16x2 r2358, r2349, r2355;
}
{
sub.f16x2 r2361, r1671, r1719;
}
{
mul.f16x2 r2364, r2361, r2243;
}
{
sub.f16x2 r2367, r1687, r1703;
}
{
mul.f16x2 r2370, r2367, r2246;
}
{
add.f16x2 r2373, r2364, r2370;
}
{
sub.f16x2 r2376, r2358, r2373;
}
{
add.f16x2 r2379, r1665, r1713;
}
{
mul.f16x2 r2382, r2379, r2241;
}
{
add.f16x2 r2385, r68, r2382;
}
{
add.f16x2 r2388, r1681, r1697;
}
{
mul.f16x2 r2391, r2388, r2245;
}
{
add.f16x2 r2394, r2385, r2391;
}
{
sub.f16x2 r2397, r1671, r1719;
}
{
mul.f16x2 r2400, r2397, r2243;
}
{
sub.f16x2 r2403, r1687, r1703;
}
{
mul.f16x2 r2406, r2403, r2246;
}
{
add.f16x2 r2409, r2400, r2406;
}
{
add.f16x2 r2412, r2394, r2409;
}
{
add.f16x2 r2415, r1671, r1719;
}
{
mul.f16x2 r2418, r2415, r2237;
}
{
add.f16x2 r2421, r212, r2418;
}
{
add.f16x2 r2424, r1687, r1703;
}
{
mul.f16x2 r2427, r2424, r2241;
}
{
add.f16x2 r2430, r2421, r2427;
}
{
sub.f16x2 r2433, r1665, r1713;
}
{
mul.f16x2 r2436, r2433, r2239;
}
{
sub.f16x2 r2439, r1681, r1697;
}
{
mul.f16x2 r2442, r2439, r2243;
}
{
add.f16x2 r2445, r2436, r2442;
}
{
add.f16x2 r2448, r2430, r2445;
}
{
add.f16x2 r2451, r1671, r1719;
}
{
mul.f16x2 r2454, r2451, r2237;
}
{
add.f16x2 r2457, r212, r2454;
}
{
add.f16x2 r2460, r1687, r1703;
}
{
mul.f16x2 r2463, r2460, r2241;
}
{
add.f16x2 r2466, r2457, r2463;
}
{
sub.f16x2 r2469, r1665, r1713;
}
{
mul.f16x2 r2472, r2469, r2239;
}
{
sub.f16x2 r2475, r1681, r1697;
}
{
mul.f16x2 r2478, r2475, r2243;
}
{
add.f16x2 r2481, r2472, r2478;
}
{
sub.f16x2 r2484, r2466, r2481;
}
{
add.f16x2 r2487, r1671, r1719;
}
{
mul.f16x2 r2490, r2487, r2241;
}
{
add.f16x2 r2493, r212, r2490;
}
{
add.f16x2 r2496, r1687, r1703;
}
{
mul.f16x2 r2499, r2496, r2245;
}
{
add.f16x2 r2502, r2493, r2499;
}
{
sub.f16x2 r2505, r1665, r1713;
}
{
mul.f16x2 r2508, r2505, r2243;
}
{
sub.f16x2 r2511, r1681, r1697;
}
{
mul.f16x2 r2514, r2511, r2246;
}
{
add.f16x2 r2517, r2508, r2514;
}
{
add.f16x2 r2520, r2502, r2517;
}
{
add.f16x2 r2523, r1671, r1719;
}
{
mul.f16x2 r2526, r2523, r2241;
}
{
add.f16x2 r2529, r212, r2526;
}
{
add.f16x2 r2532, r1687, r1703;
}
{
mul.f16x2 r2535, r2532, r2245;
}
{
add.f16x2 r2538, r2529, r2535;
}
{
sub.f16x2 r2541, r1665, r1713;
}
{
mul.f16x2 r2544, r2541, r2243;
}
{
sub.f16x2 r2547, r1681, r1697;
}
{
mul.f16x2 r2550, r2547, r2246;
}
{
add.f16x2 r2553, r2544, r2550;
}
{
sub.f16x2 r2556, r2538, r2553;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2559, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2560, {low, high};
}
{
neg.f16x2 r2561, r2560;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r2563, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r2564, {low, high};
}
{
neg.f16x2 r2565, r2564;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2567, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2568, {low, high};
}
{
add.f16x2 r2569, r1729, r1777;
}
{
add.f16x2 r2572, r140, r2569;
}
{
add.f16x2 r2575, r1745, r1761;
}
{
add.f16x2 r2578, r2572, r2575;
}
{
add.f16x2 r2581, r1735, r1783;
}
{
add.f16x2 r2584, r284, r2581;
}
{
add.f16x2 r2587, r1751, r1767;
}
{
add.f16x2 r2590, r2584, r2587;
}
{
add.f16x2 r2593, r1729, r1777;
}
{
mul.f16x2 r2596, r2593, r2559;
}
{
add.f16x2 r2599, r140, r2596;
}
{
add.f16x2 r2602, r1745, r1761;
}
{
mul.f16x2 r2605, r2602, r2563;
}
{
add.f16x2 r2608, r2599, r2605;
}
{
sub.f16x2 r2611, r1735, r1783;
}
{
mul.f16x2 r2614, r2611, r2561;
}
{
sub.f16x2 r2617, r1751, r1767;
}
{
mul.f16x2 r2620, r2617, r2565;
}
{
add.f16x2 r2623, r2614, r2620;
}
{
sub.f16x2 r2626, r2608, r2623;
}
{
add.f16x2 r2629, r1729, r1777;
}
{
mul.f16x2 r2632, r2629, r2559;
}
{
add.f16x2 r2635, r140, r2632;
}
{
add.f16x2 r2638, r1745, r1761;
}
{
mul.f16x2 r2641, r2638, r2563;
}
{
add.f16x2 r2644, r2635, r2641;
}
{
sub.f16x2 r2647, r1735, r1783;
}
{
mul.f16x2 r2650, r2647, r2561;
}
{
sub.f16x2 r2653, r1751, r1767;
}
{
mul.f16x2 r2656, r2653, r2565;
}
{
add.f16x2 r2659, r2650, r2656;
}
{
add.f16x2 r2662, r2644, r2659;
}
{
add.f16x2 r2665, r1729, r1777;
}
{
mul.f16x2 r2668, r2665, r2563;
}
{
add.f16x2 r2671, r140, r2668;
}
{
add.f16x2 r2674, r1745, r1761;
}
{
mul.f16x2 r2677, r2674, r2567;
}
{
add.f16x2 r2680, r2671, r2677;
}
{
sub.f16x2 r2683, r1735, r1783;
}
{
mul.f16x2 r2686, r2683, r2565;
}
{
sub.f16x2 r2689, r1751, r1767;
}
{
mul.f16x2 r2692, r2689, r2568;
}
{
add.f16x2 r2695, r2686, r2692;
}
{
sub.f16x2 r2698, r2680, r2695;
}
{
add.f16x2 r2701, r1729, r1777;
}
{
mul.f16x2 r2704, r2701, r2563;
}
{
add.f16x2 r2707, r140, r2704;
}
{
add.f16x2 r2710, r1745, r1761;
}
{
mul.f16x2 r2713, r2710, r2567;
}
{
add.f16x2 r2716, r2707, r2713;
}
{
sub.f16x2 r2719, r1735, r1783;
}
{
mul.f16x2 r2722, r2719, r2565;
}
{
sub.f16x2 r2725, r1751, r1767;
}
{
mul.f16x2 r2728, r2725, r2568;
}
{
add.f16x2 r2731, r2722, r2728;
}
{
add.f16x2 r2734, r2716, r2731;
}
{
add.f16x2 r2737, r1735, r1783;
}
{
mul.f16x2 r2740, r2737, r2559;
}
{
add.f16x2 r2743, r284, r2740;
}
{
add.f16x2 r2746, r1751, r1767;
}
{
mul.f16x2 r2749, r2746, r2563;
}
{
add.f16x2 r2752, r2743, r2749;
}
{
sub.f16x2 r2755, r1729, r1777;
}
{
mul.f16x2 r2758, r2755, r2561;
}
{
sub.f16x2 r2761, r1745, r1761;
}
{
mul.f16x2 r2764, r2761, r2565;
}
{
add.f16x2 r2767, r2758, r2764;
}
{
add.f16x2 r2770, r2752, r2767;
}
{
add.f16x2 r2773, r1735, r1783;
}
{
mul.f16x2 r2776, r2773, r2559;
}
{
add.f16x2 r2779, r284, r2776;
}
{
add.f16x2 r2782, r1751, r1767;
}
{
mul.f16x2 r2785, r2782, r2563;
}
{
add.f16x2 r2788, r2779, r2785;
}
{
sub.f16x2 r2791, r1729, r1777;
}
{
mul.f16x2 r2794, r2791, r2561;
}
{
sub.f16x2 r2797, r1745, r1761;
}
{
mul.f16x2 r2800, r2797, r2565;
}
{
add.f16x2 r2803, r2794, r2800;
}
{
sub.f16x2 r2806, r2788, r2803;
}
{
add.f16x2 r2809, r1735, r1783;
}
{
mul.f16x2 r2812, r2809, r2563;
}
{
add.f16x2 r2815, r284, r2812;
}
{
add.f16x2 r2818, r1751, r1767;
}
{
mul.f16x2 r2821, r2818, r2567;
}
{
add.f16x2 r2824, r2815, r2821;
}
{
sub.f16x2 r2827, r1729, r1777;
}
{
mul.f16x2 r2830, r2827, r2565;
}
{
sub.f16x2 r2833, r1745, r1761;
}
{
mul.f16x2 r2836, r2833, r2568;
}
{
add.f16x2 r2839, r2830, r2836;
}
{
add.f16x2 r2842, r2824, r2839;
}
{
add.f16x2 r2845, r1735, r1783;
}
{
mul.f16x2 r2848, r2845, r2563;
}
{
add.f16x2 r2851, r284, r2848;
}
{
add.f16x2 r2854, r1751, r1767;
}
{
mul.f16x2 r2857, r2854, r2567;
}
{
add.f16x2 r2860, r2851, r2857;
}
{
sub.f16x2 r2863, r1729, r1777;
}
{
mul.f16x2 r2866, r2863, r2565;
}
{
sub.f16x2 r2869, r1745, r1761;
}
{
mul.f16x2 r2872, r2869, r2568;
}
{
add.f16x2 r2875, r2866, r2872;
}
{
sub.f16x2 r2878, r2860, r2875;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2881, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2882, {low, high};
}
{
neg.f16x2 r2883, r2882;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r2885, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r2886, {low, high};
}
{
neg.f16x2 r2887, r2886;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2889, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2890, {low, high};
}
{
add.f16x2 r2891, r1793, r1841;
}
{
add.f16x2 r2894, r176, r2891;
}
{
add.f16x2 r2897, r1809, r1825;
}
{
add.f16x2 r2900, r2894, r2897;
}
{
add.f16x2 r2903, r1799, r1847;
}
{
add.f16x2 r2906, r320, r2903;
}
{
add.f16x2 r2909, r1815, r1831;
}
{
add.f16x2 r2912, r2906, r2909;
}
{
add.f16x2 r2915, r1793, r1841;
}
{
mul.f16x2 r2918, r2915, r2881;
}
{
add.f16x2 r2921, r176, r2918;
}
{
add.f16x2 r2924, r1809, r1825;
}
{
mul.f16x2 r2927, r2924, r2885;
}
{
add.f16x2 r2930, r2921, r2927;
}
{
sub.f16x2 r2933, r1799, r1847;
}
{
mul.f16x2 r2936, r2933, r2883;
}
{
sub.f16x2 r2939, r1815, r1831;
}
{
mul.f16x2 r2942, r2939, r2887;
}
{
add.f16x2 r2945, r2936, r2942;
}
{
sub.f16x2 r2948, r2930, r2945;
}
{
add.f16x2 r2951, r1793, r1841;
}
{
mul.f16x2 r2954, r2951, r2881;
}
{
add.f16x2 r2957, r176, r2954;
}
{
add.f16x2 r2960, r1809, r1825;
}
{
mul.f16x2 r2963, r2960, r2885;
}
{
add.f16x2 r2966, r2957, r2963;
}
{
sub.f16x2 r2969, r1799, r1847;
}
{
mul.f16x2 r2972, r2969, r2883;
}
{
sub.f16x2 r2975, r1815, r1831;
}
{
mul.f16x2 r2978, r2975, r2887;
}
{
add.f16x2 r2981, r2972, r2978;
}
{
add.f16x2 r2984, r2966, r2981;
}
{
add.f16x2 r2987, r1793, r1841;
}
{
mul.f16x2 r2990, r2987, r2885;
}
{
add.f16x2 r2993, r176, r2990;
}
{
add.f16x2 r2996, r1809, r1825;
}
{
mul.f16x2 r2999, r2996, r2889;
}
{
add.f16x2 r3002, r2993, r2999;
}
{
sub.f16x2 r3005, r1799, r1847;
}
{
mul.f16x2 r3008, r3005, r2887;
}
{
sub.f16x2 r3011, r1815, r1831;
}
{
mul.f16x2 r3014, r3011, r2890;
}
{
add.f16x2 r3017, r3008, r3014;
}
{
sub.f16x2 r3020, r3002, r3017;
}
{
add.f16x2 r3023, r1793, r1841;
}
{
mul.f16x2 r3026, r3023, r2885;
}
{
add.f16x2 r3029, r176, r3026;
}
{
add.f16x2 r3032, r1809, r1825;
}
{
mul.f16x2 r3035, r3032, r2889;
}
{
add.f16x2 r3038, r3029, r3035;
}
{
sub.f16x2 r3041, r1799, r1847;
}
{
mul.f16x2 r3044, r3041, r2887;
}
{
sub.f16x2 r3047, r1815, r1831;
}
{
mul.f16x2 r3050, r3047, r2890;
}
{
add.f16x2 r3053, r3044, r3050;
}
{
add.f16x2 r3056, r3038, r3053;
}
{
add.f16x2 r3059, r1799, r1847;
}
{
mul.f16x2 r3062, r3059, r2881;
}
{
add.f16x2 r3065, r320, r3062;
}
{
add.f16x2 r3068, r1815, r1831;
}
{
mul.f16x2 r3071, r3068, r2885;
}
{
add.f16x2 r3074, r3065, r3071;
}
{
sub.f16x2 r3077, r1793, r1841;
}
{
mul.f16x2 r3080, r3077, r2883;
}
{
sub.f16x2 r3083, r1809, r1825;
}
{
mul.f16x2 r3086, r3083, r2887;
}
{
add.f16x2 r3089, r3080, r3086;
}
{
add.f16x2 r3092, r3074, r3089;
}
{
add.f16x2 r3095, r1799, r1847;
}
{
mul.f16x2 r3098, r3095, r2881;
}
{
add.f16x2 r3101, r320, r3098;
}
{
add.f16x2 r3104, r1815, r1831;
}
{
mul.f16x2 r3107, r3104, r2885;
}
{
add.f16x2 r3110, r3101, r3107;
}
{
sub.f16x2 r3113, r1793, r1841;
}
{
mul.f16x2 r3116, r3113, r2883;
}
{
sub.f16x2 r3119, r1809, r1825;
}
{
mul.f16x2 r3122, r3119, r2887;
}
{
add.f16x2 r3125, r3116, r3122;
}
{
sub.f16x2 r3128, r3110, r3125;
}
{
add.f16x2 r3131, r1799, r1847;
}
{
mul.f16x2 r3134, r3131, r2885;
}
{
add.f16x2 r3137, r320, r3134;
}
{
add.f16x2 r3140, r1815, r1831;
}
{
mul.f16x2 r3143, r3140, r2889;
}
{
add.f16x2 r3146, r3137, r3143;
}
{
sub.f16x2 r3149, r1793, r1841;
}
{
mul.f16x2 r3152, r3149, r2887;
}
{
sub.f16x2 r3155, r1809, r1825;
}
{
mul.f16x2 r3158, r3155, r2890;
}
{
add.f16x2 r3161, r3152, r3158;
}
{
add.f16x2 r3164, r3146, r3161;
}
{
add.f16x2 r3167, r1799, r1847;
}
{
mul.f16x2 r3170, r3167, r2885;
}
{
add.f16x2 r3173, r320, r3170;
}
{
add.f16x2 r3176, r1815, r1831;
}
{
mul.f16x2 r3179, r3176, r2889;
}
{
add.f16x2 r3182, r3173, r3179;
}
{
sub.f16x2 r3185, r1793, r1841;
}
{
mul.f16x2 r3188, r3185, r2887;
}
{
sub.f16x2 r3191, r1809, r1825;
}
{
mul.f16x2 r3194, r3191, r2890;
}
{
add.f16x2 r3197, r3188, r3194;
}
{
sub.f16x2 r3200, r3182, r3197;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r3203, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r3204, {low, high};
}
{
neg.f16x2 r3205, r3204;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r3207, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r3208, {low, high};
}
{
neg.f16x2 r3209, r3208;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r3211, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r3212, {low, high};
}
{
add.f16x2 r3213, r1857, r1905;
}
{
add.f16x2 r3216, r104, r3213;
}
{
add.f16x2 r3219, r1873, r1889;
}
{
add.f16x2 r3222, r3216, r3219;
}
{
add.f16x2 r3225, r1863, r1911;
}
{
add.f16x2 r3228, r248, r3225;
}
{
add.f16x2 r3231, r1879, r1895;
}
{
add.f16x2 r3234, r3228, r3231;
}
{
add.f16x2 r3237, r1857, r1905;
}
{
mul.f16x2 r3240, r3237, r3203;
}
{
add.f16x2 r3243, r104, r3240;
}
{
add.f16x2 r3246, r1873, r1889;
}
{
mul.f16x2 r3249, r3246, r3207;
}
{
add.f16x2 r3252, r3243, r3249;
}
{
sub.f16x2 r3255, r1863, r1911;
}
{
mul.f16x2 r3258, r3255, r3205;
}
{
sub.f16x2 r3261, r1879, r1895;
}
{
mul.f16x2 r3264, r3261, r3209;
}
{
add.f16x2 r3267, r3258, r3264;
}
{
sub.f16x2 r3270, r3252, r3267;
}
{
add.f16x2 r3273, r1857, r1905;
}
{
mul.f16x2 r3276, r3273, r3203;
}
{
add.f16x2 r3279, r104, r3276;
}
{
add.f16x2 r3282, r1873, r1889;
}
{
mul.f16x2 r3285, r3282, r3207;
}
{
add.f16x2 r3288, r3279, r3285;
}
{
sub.f16x2 r3291, r1863, r1911;
}
{
mul.f16x2 r3294, r3291, r3205;
}
{
sub.f16x2 r3297, r1879, r1895;
}
{
mul.f16x2 r3300, r3297, r3209;
}
{
add.f16x2 r3303, r3294, r3300;
}
{
add.f16x2 r3306, r3288, r3303;
}
{
add.f16x2 r3309, r1857, r1905;
}
{
mul.f16x2 r3312, r3309, r3207;
}
{
add.f16x2 r3315, r104, r3312;
}
{
add.f16x2 r3318, r1873, r1889;
}
{
mul.f16x2 r3321, r3318, r3211;
}
{
add.f16x2 r3324, r3315, r3321;
}
{
sub.f16x2 r3327, r1863, r1911;
}
{
mul.f16x2 r3330, r3327, r3209;
}
{
sub.f16x2 r3333, r1879, r1895;
}
{
mul.f16x2 r3336, r3333, r3212;
}
{
add.f16x2 r3339, r3330, r3336;
}
{
sub.f16x2 r3342, r3324, r3339;
}
{
add.f16x2 r3345, r1857, r1905;
}
{
mul.f16x2 r3348, r3345, r3207;
}
{
add.f16x2 r3351, r104, r3348;
}
{
add.f16x2 r3354, r1873, r1889;
}
{
mul.f16x2 r3357, r3354, r3211;
}
{
add.f16x2 r3360, r3351, r3357;
}
{
sub.f16x2 r3363, r1863, r1911;
}
{
mul.f16x2 r3366, r3363, r3209;
}
{
sub.f16x2 r3369, r1879, r1895;
}
{
mul.f16x2 r3372, r3369, r3212;
}
{
add.f16x2 r3375, r3366, r3372;
}
{
add.f16x2 r3378, r3360, r3375;
}
{
add.f16x2 r3381, r1863, r1911;
}
{
mul.f16x2 r3384, r3381, r3203;
}
{
add.f16x2 r3387, r248, r3384;
}
{
add.f16x2 r3390, r1879, r1895;
}
{
mul.f16x2 r3393, r3390, r3207;
}
{
add.f16x2 r3396, r3387, r3393;
}
{
sub.f16x2 r3399, r1857, r1905;
}
{
mul.f16x2 r3402, r3399, r3205;
}
{
sub.f16x2 r3405, r1873, r1889;
}
{
mul.f16x2 r3408, r3405, r3209;
}
{
add.f16x2 r3411, r3402, r3408;
}
{
add.f16x2 r3414, r3396, r3411;
}
{
add.f16x2 r3417, r1863, r1911;
}
{
mul.f16x2 r3420, r3417, r3203;
}
{
add.f16x2 r3423, r248, r3420;
}
{
add.f16x2 r3426, r1879, r1895;
}
{
mul.f16x2 r3429, r3426, r3207;
}
{
add.f16x2 r3432, r3423, r3429;
}
{
sub.f16x2 r3435, r1857, r1905;
}
{
mul.f16x2 r3438, r3435, r3205;
}
{
sub.f16x2 r3441, r1873, r1889;
}
{
mul.f16x2 r3444, r3441, r3209;
}
{
add.f16x2 r3447, r3438, r3444;
}
{
sub.f16x2 r3450, r3432, r3447;
}
{
add.f16x2 r3453, r1863, r1911;
}
{
mul.f16x2 r3456, r3453, r3207;
}
{
add.f16x2 r3459, r248, r3456;
}
{
add.f16x2 r3462, r1879, r1895;
}
{
mul.f16x2 r3465, r3462, r3211;
}
{
add.f16x2 r3468, r3459, r3465;
}
{
sub.f16x2 r3471, r1857, r1905;
}
{
mul.f16x2 r3474, r3471, r3209;
}
{
sub.f16x2 r3477, r1873, r1889;
}
{
mul.f16x2 r3480, r3477, r3212;
}
{
add.f16x2 r3483, r3474, r3480;
}
{
add.f16x2 r3486, r3468, r3483;
}
{
add.f16x2 r3489, r1863, r1911;
}
{
mul.f16x2 r3492, r3489, r3207;
}
{
add.f16x2 r3495, r248, r3492;
}
{
add.f16x2 r3498, r1879, r1895;
}
{
mul.f16x2 r3501, r3498, r3211;
}
{
add.f16x2 r3504, r3495, r3501;
}
{
sub.f16x2 r3507, r1857, r1905;
}
{
mul.f16x2 r3510, r3507, r3209;
}
{
sub.f16x2 r3513, r1873, r1889;
}
{
mul.f16x2 r3516, r3513, r3212;
}
{
add.f16x2 r3519, r3510, r3516;
}
{
sub.f16x2 r3522, r3504, r3519;
}
mul.wide.u32 rd2, r10444, 274877907;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r10445, rd3;
mul.lo.s32 r10446, r10445, 125;
sub.s32 r10447, r10444, r10446;
mad.lo.s32 r10448, r10445, 12500, r10443;
cvt.rn.f32.u32 f597, r10447;
mul.f32 f598, f597, 0f3B03C498;
cos.approx.f32 f217, f598;
sin.approx.f32 f599, f598;
neg.f32 f218, f599;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f217;
cvt.rn.f16.f32 high, f218;
mov.b32 r3525, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3528, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3530, {high, high};
}
{
mul.f16x2 r3532, r2268, r3530;
}
{
fma.rn.f16x2 r3535, r2256, r3528, r3532;
}
{
mul.f16x2 r3539, r2256, r3530;
}
{
neg.f16x2 r3542, r3539;
}
{
fma.rn.f16x2 r3544, r2268, r3528, r3542;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3548, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3550, {high, high};
}
mov.f32 f533, 0fBF800000;
mov.f32 f534, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3552, {low, high};
}
{
mul.f16x2 r3553, r3550, r3552;
}
{
mul.f16x2 r3556, r3525, r3548;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3559, {high, low};
}
{
fma.rn.f16x2 r3561, r3553, r3559, r3556;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3561;
mov.b32 r3565, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3561;
mov.b32 r3567, {high, high};
}
{
mul.f16x2 r3569, r2590, r3567;
}
{
fma.rn.f16x2 r3572, r2578, r3565, r3569;
}
{
mul.f16x2 r3576, r2578, r3567;
}
{
neg.f16x2 r3579, r3576;
}
{
fma.rn.f16x2 r3581, r2590, r3565, r3579;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3585, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3587, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3589, {low, high};
}
{
mul.f16x2 r3590, r3587, r3589;
}
{
mul.f16x2 r3593, r3561, r3585;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3561;
mov.b32 r3596, {high, low};
}
{
fma.rn.f16x2 r3598, r3590, r3596, r3593;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3598;
mov.b32 r3602, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3598;
mov.b32 r3604, {high, high};
}
{
mul.f16x2 r3606, r2912, r3604;
}
{
fma.rn.f16x2 r3609, r2900, r3602, r3606;
}
{
mul.f16x2 r3613, r2900, r3604;
}
{
neg.f16x2 r3616, r3613;
}
{
fma.rn.f16x2 r3618, r2912, r3602, r3616;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3622, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3624, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3626, {low, high};
}
{
mul.f16x2 r3627, r3624, r3626;
}
{
mul.f16x2 r3630, r3598, r3622;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3598;
mov.b32 r3633, {high, low};
}
{
fma.rn.f16x2 r3635, r3627, r3633, r3630;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3635;
mov.b32 r3639, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3635;
mov.b32 r3641, {high, high};
}
{
mul.f16x2 r3643, r3234, r3641;
}
{
fma.rn.f16x2 r3646, r3222, r3639, r3643;
}
{
mul.f16x2 r3650, r3222, r3641;
}
{
neg.f16x2 r3653, r3650;
}
{
fma.rn.f16x2 r3655, r3234, r3639, r3653;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3659, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3661, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3663, {low, high};
}
{
mul.f16x2 r3664, r3661, r3663;
}
{
mul.f16x2 r3667, r3635, r3659;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3635;
mov.b32 r3670, {high, low};
}
{
fma.rn.f16x2 r3672, r3664, r3670, r3667;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3672;
mov.b32 r3676, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3672;
mov.b32 r3678, {high, high};
}
{
mul.f16x2 r3680, r2126, r3678;
}
{
fma.rn.f16x2 r3683, r1982, r3676, r3680;
}
{
mul.f16x2 r3687, r1982, r3678;
}
{
neg.f16x2 r3690, r3687;
}
{
fma.rn.f16x2 r3692, r2126, r3676, r3690;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3696, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3698, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3700, {low, high};
}
{
mul.f16x2 r3701, r3698, r3700;
}
{
mul.f16x2 r3704, r3672, r3696;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3672;
mov.b32 r3707, {high, low};
}
{
fma.rn.f16x2 r3709, r3701, r3707, r3704;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3709;
mov.b32 r3713, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3709;
mov.b32 r3715, {high, high};
}
{
mul.f16x2 r3717, r2448, r3715;
}
{
fma.rn.f16x2 r3720, r2304, r3713, r3717;
}
{
mul.f16x2 r3724, r2304, r3715;
}
{
neg.f16x2 r3727, r3724;
}
{
fma.rn.f16x2 r3729, r2448, r3713, r3727;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3733, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3735, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3737, {low, high};
}
{
mul.f16x2 r3738, r3735, r3737;
}
{
mul.f16x2 r3741, r3709, r3733;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3709;
mov.b32 r3744, {high, low};
}
{
fma.rn.f16x2 r3746, r3738, r3744, r3741;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3746;
mov.b32 r3750, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3746;
mov.b32 r3752, {high, high};
}
{
mul.f16x2 r3754, r2770, r3752;
}
{
fma.rn.f16x2 r3757, r2626, r3750, r3754;
}
{
mul.f16x2 r3761, r2626, r3752;
}
{
neg.f16x2 r3764, r3761;
}
{
fma.rn.f16x2 r3766, r2770, r3750, r3764;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3770, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3772, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3774, {low, high};
}
{
mul.f16x2 r3775, r3772, r3774;
}
{
mul.f16x2 r3778, r3746, r3770;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3746;
mov.b32 r3781, {high, low};
}
{
fma.rn.f16x2 r3783, r3775, r3781, r3778;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3783;
mov.b32 r3787, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3783;
mov.b32 r3789, {high, high};
}
{
mul.f16x2 r3791, r3092, r3789;
}
{
fma.rn.f16x2 r3794, r2948, r3787, r3791;
}
{
mul.f16x2 r3798, r2948, r3789;
}
{
neg.f16x2 r3801, r3798;
}
{
fma.rn.f16x2 r3803, r3092, r3787, r3801;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3807, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3809, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3811, {low, high};
}
{
mul.f16x2 r3812, r3809, r3811;
}
{
mul.f16x2 r3815, r3783, r3807;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3783;
mov.b32 r3818, {high, low};
}
{
fma.rn.f16x2 r3820, r3812, r3818, r3815;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3820;
mov.b32 r3824, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3820;
mov.b32 r3826, {high, high};
}
{
mul.f16x2 r3828, r3414, r3826;
}
{
fma.rn.f16x2 r3831, r3270, r3824, r3828;
}
{
mul.f16x2 r3835, r3270, r3826;
}
{
neg.f16x2 r3838, r3835;
}
{
fma.rn.f16x2 r3840, r3414, r3824, r3838;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3844, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3846, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3848, {low, high};
}
{
mul.f16x2 r3849, r3846, r3848;
}
{
mul.f16x2 r3852, r3820, r3844;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3820;
mov.b32 r3855, {high, low};
}
{
fma.rn.f16x2 r3857, r3849, r3855, r3852;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3857;
mov.b32 r3861, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3857;
mov.b32 r3863, {high, high};
}
{
mul.f16x2 r3865, r2198, r3863;
}
{
fma.rn.f16x2 r3868, r2054, r3861, r3865;
}
{
mul.f16x2 r3872, r2054, r3863;
}
{
neg.f16x2 r3875, r3872;
}
{
fma.rn.f16x2 r3877, r2198, r3861, r3875;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3881, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3883, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3885, {low, high};
}
{
mul.f16x2 r3886, r3883, r3885;
}
{
mul.f16x2 r3889, r3857, r3881;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3857;
mov.b32 r3892, {high, low};
}
{
fma.rn.f16x2 r3894, r3886, r3892, r3889;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3894;
mov.b32 r3898, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3894;
mov.b32 r3900, {high, high};
}
{
mul.f16x2 r3902, r2520, r3900;
}
{
fma.rn.f16x2 r3905, r2376, r3898, r3902;
}
{
mul.f16x2 r3909, r2376, r3900;
}
{
neg.f16x2 r3912, r3909;
}
{
fma.rn.f16x2 r3914, r2520, r3898, r3912;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3918, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3920, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3922, {low, high};
}
{
mul.f16x2 r3923, r3920, r3922;
}
{
mul.f16x2 r3926, r3894, r3918;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3894;
mov.b32 r3929, {high, low};
}
{
fma.rn.f16x2 r3931, r3923, r3929, r3926;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3931;
mov.b32 r3935, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3931;
mov.b32 r3937, {high, high};
}
{
mul.f16x2 r3939, r2842, r3937;
}
{
fma.rn.f16x2 r3942, r2698, r3935, r3939;
}
{
mul.f16x2 r3946, r2698, r3937;
}
{
neg.f16x2 r3949, r3946;
}
{
fma.rn.f16x2 r3951, r2842, r3935, r3949;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3955, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3957, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3959, {low, high};
}
{
mul.f16x2 r3960, r3957, r3959;
}
{
mul.f16x2 r3963, r3931, r3955;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3931;
mov.b32 r3966, {high, low};
}
{
fma.rn.f16x2 r3968, r3960, r3966, r3963;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3968;
mov.b32 r3972, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3968;
mov.b32 r3974, {high, high};
}
{
mul.f16x2 r3976, r3164, r3974;
}
{
fma.rn.f16x2 r3979, r3020, r3972, r3976;
}
{
mul.f16x2 r3983, r3020, r3974;
}
{
neg.f16x2 r3986, r3983;
}
{
fma.rn.f16x2 r3988, r3164, r3972, r3986;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3992, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r3994, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3996, {low, high};
}
{
mul.f16x2 r3997, r3994, r3996;
}
{
mul.f16x2 r4000, r3968, r3992;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3968;
mov.b32 r4003, {high, low};
}
{
fma.rn.f16x2 r4005, r3997, r4003, r4000;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4005;
mov.b32 r4009, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4005;
mov.b32 r4011, {high, high};
}
{
mul.f16x2 r4013, r3486, r4011;
}
{
fma.rn.f16x2 r4016, r3342, r4009, r4013;
}
{
mul.f16x2 r4020, r3342, r4011;
}
{
neg.f16x2 r4023, r4020;
}
{
fma.rn.f16x2 r4025, r3486, r4009, r4023;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4029, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4031, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4033, {low, high};
}
{
mul.f16x2 r4034, r4031, r4033;
}
{
mul.f16x2 r4037, r4005, r4029;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4005;
mov.b32 r4040, {high, low};
}
{
fma.rn.f16x2 r4042, r4034, r4040, r4037;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4042;
mov.b32 r4046, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4042;
mov.b32 r4048, {high, high};
}
{
mul.f16x2 r4050, r2234, r4048;
}
{
fma.rn.f16x2 r4053, r2090, r4046, r4050;
}
{
mul.f16x2 r4057, r2090, r4048;
}
{
neg.f16x2 r4060, r4057;
}
{
fma.rn.f16x2 r4062, r2234, r4046, r4060;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4066, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4068, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4070, {low, high};
}
{
mul.f16x2 r4071, r4068, r4070;
}
{
mul.f16x2 r4074, r4042, r4066;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4042;
mov.b32 r4077, {high, low};
}
{
fma.rn.f16x2 r4079, r4071, r4077, r4074;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4079;
mov.b32 r4083, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4079;
mov.b32 r4085, {high, high};
}
{
mul.f16x2 r4087, r2556, r4085;
}
{
fma.rn.f16x2 r4090, r2412, r4083, r4087;
}
{
mul.f16x2 r4094, r2412, r4085;
}
{
neg.f16x2 r4097, r4094;
}
{
fma.rn.f16x2 r4099, r2556, r4083, r4097;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4103, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4105, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4107, {low, high};
}
{
mul.f16x2 r4108, r4105, r4107;
}
{
mul.f16x2 r4111, r4079, r4103;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4079;
mov.b32 r4114, {high, low};
}
{
fma.rn.f16x2 r4116, r4108, r4114, r4111;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4116;
mov.b32 r4120, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4116;
mov.b32 r4122, {high, high};
}
{
mul.f16x2 r4124, r2878, r4122;
}
{
fma.rn.f16x2 r4127, r2734, r4120, r4124;
}
{
mul.f16x2 r4131, r2734, r4122;
}
{
neg.f16x2 r4134, r4131;
}
{
fma.rn.f16x2 r4136, r2878, r4120, r4134;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4140, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4142, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4144, {low, high};
}
{
mul.f16x2 r4145, r4142, r4144;
}
{
mul.f16x2 r4148, r4116, r4140;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4116;
mov.b32 r4151, {high, low};
}
{
fma.rn.f16x2 r4153, r4145, r4151, r4148;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4153;
mov.b32 r4157, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4153;
mov.b32 r4159, {high, high};
}
{
mul.f16x2 r4161, r3200, r4159;
}
{
fma.rn.f16x2 r4164, r3056, r4157, r4161;
}
{
mul.f16x2 r4168, r3056, r4159;
}
{
neg.f16x2 r4171, r4168;
}
{
fma.rn.f16x2 r4173, r3200, r4157, r4171;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4177, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4179, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4181, {low, high};
}
{
mul.f16x2 r4182, r4179, r4181;
}
{
mul.f16x2 r4185, r4153, r4177;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4153;
mov.b32 r4188, {high, low};
}
{
fma.rn.f16x2 r4190, r4182, r4188, r4185;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4190;
mov.b32 r4194, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4190;
mov.b32 r4196, {high, high};
}
{
mul.f16x2 r4198, r3522, r4196;
}
{
fma.rn.f16x2 r4201, r3378, r4194, r4198;
}
{
mul.f16x2 r4205, r3378, r4196;
}
{
neg.f16x2 r4208, r4205;
}
{
fma.rn.f16x2 r4210, r3522, r4194, r4208;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4214, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4216, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4218, {low, high};
}
{
mul.f16x2 r4219, r4216, r4218;
}
{
mul.f16x2 r4222, r4190, r4214;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4190;
mov.b32 r4225, {high, low};
}
{
fma.rn.f16x2 r4227, r4219, r4225, r4222;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4227;
mov.b32 r4231, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4227;
mov.b32 r4233, {high, high};
}
{
mul.f16x2 r4235, r2162, r4233;
}
{
fma.rn.f16x2 r4238, r2018, r4231, r4235;
}
{
mul.f16x2 r4242, r2018, r4233;
}
{
neg.f16x2 r4245, r4242;
}
{
fma.rn.f16x2 r4247, r2162, r4231, r4245;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4251, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4253, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4255, {low, high};
}
{
mul.f16x2 r4256, r4253, r4255;
}
{
mul.f16x2 r4259, r4227, r4251;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4227;
mov.b32 r4262, {high, low};
}
{
fma.rn.f16x2 r4264, r4256, r4262, r4259;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4264;
mov.b32 r4268, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4264;
mov.b32 r4270, {high, high};
}
{
mul.f16x2 r4272, r2484, r4270;
}
{
fma.rn.f16x2 r4275, r2340, r4268, r4272;
}
{
mul.f16x2 r4279, r2340, r4270;
}
{
neg.f16x2 r4282, r4279;
}
{
fma.rn.f16x2 r4284, r2484, r4268, r4282;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4288, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4290, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4292, {low, high};
}
{
mul.f16x2 r4293, r4290, r4292;
}
{
mul.f16x2 r4296, r4264, r4288;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4264;
mov.b32 r4299, {high, low};
}
{
fma.rn.f16x2 r4301, r4293, r4299, r4296;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4301;
mov.b32 r4305, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4301;
mov.b32 r4307, {high, high};
}
{
mul.f16x2 r4309, r2806, r4307;
}
{
fma.rn.f16x2 r4312, r2662, r4305, r4309;
}
{
mul.f16x2 r4316, r2662, r4307;
}
{
neg.f16x2 r4319, r4316;
}
{
fma.rn.f16x2 r4321, r2806, r4305, r4319;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4325, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4327, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4329, {low, high};
}
{
mul.f16x2 r4330, r4327, r4329;
}
{
mul.f16x2 r4333, r4301, r4325;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4301;
mov.b32 r4336, {high, low};
}
{
fma.rn.f16x2 r4338, r4330, r4336, r4333;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4338;
mov.b32 r4342, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4338;
mov.b32 r4344, {high, high};
}
{
mul.f16x2 r4346, r3128, r4344;
}
{
fma.rn.f16x2 r4349, r2984, r4342, r4346;
}
{
mul.f16x2 r4353, r2984, r4344;
}
{
neg.f16x2 r4356, r4353;
}
{
fma.rn.f16x2 r4358, r3128, r4342, r4356;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4362, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3525;
mov.b32 r4364, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4366, {low, high};
}
{
mul.f16x2 r4367, r4364, r4366;
}
{
mul.f16x2 r4370, r4338, r4362;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4338;
mov.b32 r4373, {high, low};
}
{
fma.rn.f16x2 r4375, r4367, r4373, r4370;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4375;
mov.b32 r4379, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4375;
mov.b32 r4381, {high, high};
}
{
mul.f16x2 r4383, r3450, r4381;
}
{
fma.rn.f16x2 r4386, r3306, r4379, r4383;
}
{
mul.f16x2 r4390, r3306, r4381;
}
{
neg.f16x2 r4393, r4390;
}
{
fma.rn.f16x2 r4395, r3450, r4379, r4393;
}
barrier.sync 0;
mad.lo.s32 r10449, r10447, 100, r10448;
st.shared.u32 [r10449], r1934;
st.shared.u32 [r10449+4], r3535;
st.shared.u32 [r10449+8], r3572;
st.shared.u32 [r10449+12], r3609;
st.shared.u32 [r10449+16], r3646;
st.shared.u32 [r10449+20], r3683;
st.shared.u32 [r10449+24], r3720;
st.shared.u32 [r10449+28], r3757;
st.shared.u32 [r10449+32], r3794;
st.shared.u32 [r10449+36], r3831;
st.shared.u32 [r10449+40], r3868;
st.shared.u32 [r10449+44], r3905;
st.shared.u32 [r10449+48], r3942;
st.shared.u32 [r10449+52], r3979;
st.shared.u32 [r10449+56], r4016;
st.shared.u32 [r10449+60], r4053;
st.shared.u32 [r10449+64], r4090;
st.shared.u32 [r10449+68], r4127;
st.shared.u32 [r10449+72], r4164;
st.shared.u32 [r10449+76], r4201;
st.shared.u32 [r10449+80], r4238;
st.shared.u32 [r10449+84], r4275;
st.shared.u32 [r10449+88], r4312;
st.shared.u32 [r10449+92], r4349;
st.shared.u32 [r10449+96], r4386;
barrier.sync 0;
mad.lo.s32 r10450, r10447, -96, r10449;
ld.shared.u32 r4430, [r10450];
ld.shared.u32 r4752, [r10450+500];
ld.shared.u32 r5074, [r10450+1000];
ld.shared.u32 r5396, [r10450+1500];
ld.shared.u32 r5718, [r10450+2000];
ld.shared.u32 r4427, [r10450+2500];
ld.shared.u32 r4749, [r10450+3000];
ld.shared.u32 r5071, [r10450+3500];
ld.shared.u32 r5393, [r10450+4000];
ld.shared.u32 r5715, [r10450+4500];
ld.shared.u32 r4433, [r10450+5000];
ld.shared.u32 r4755, [r10450+5500];
ld.shared.u32 r5077, [r10450+6000];
ld.shared.u32 r5399, [r10450+6500];
ld.shared.u32 r5721, [r10450+7000];
ld.shared.u32 r4434, [r10450+7500];
ld.shared.u32 r4756, [r10450+8000];
ld.shared.u32 r5078, [r10450+8500];
ld.shared.u32 r5400, [r10450+9000];
ld.shared.u32 r5722, [r10450+9500];
ld.shared.u32 r4428, [r10450+10000];
ld.shared.u32 r4750, [r10450+10500];
ld.shared.u32 r5072, [r10450+11000];
ld.shared.u32 r5394, [r10450+11500];
ld.shared.u32 r5716, [r10450+12000];
barrier.sync 0;
st.shared.u32 [r10449], r1946;
st.shared.u32 [r10449+4], r3544;
st.shared.u32 [r10449+8], r3581;
st.shared.u32 [r10449+12], r3618;
st.shared.u32 [r10449+16], r3655;
st.shared.u32 [r10449+20], r3692;
st.shared.u32 [r10449+24], r3729;
st.shared.u32 [r10449+28], r3766;
st.shared.u32 [r10449+32], r3803;
st.shared.u32 [r10449+36], r3840;
st.shared.u32 [r10449+40], r3877;
st.shared.u32 [r10449+44], r3914;
st.shared.u32 [r10449+48], r3951;
st.shared.u32 [r10449+52], r3988;
st.shared.u32 [r10449+56], r4025;
st.shared.u32 [r10449+60], r4062;
st.shared.u32 [r10449+64], r4099;
st.shared.u32 [r10449+68], r4136;
st.shared.u32 [r10449+72], r4173;
st.shared.u32 [r10449+76], r4210;
st.shared.u32 [r10449+80], r4247;
st.shared.u32 [r10449+84], r4284;
st.shared.u32 [r10449+88], r4321;
st.shared.u32 [r10449+92], r4358;
st.shared.u32 [r10449+96], r4395;
barrier.sync 0;
ld.shared.u32 r4442, [r10450];
ld.shared.u32 r4764, [r10450+500];
ld.shared.u32 r5086, [r10450+1000];
ld.shared.u32 r5408, [r10450+1500];
ld.shared.u32 r5730, [r10450+2000];
ld.shared.u32 r4439, [r10450+2500];
ld.shared.u32 r4761, [r10450+3000];
ld.shared.u32 r5083, [r10450+3500];
ld.shared.u32 r5405, [r10450+4000];
ld.shared.u32 r5727, [r10450+4500];
ld.shared.u32 r4445, [r10450+5000];
ld.shared.u32 r4767, [r10450+5500];
ld.shared.u32 r5089, [r10450+6000];
ld.shared.u32 r5411, [r10450+6500];
ld.shared.u32 r5733, [r10450+7000];
ld.shared.u32 r4446, [r10450+7500];
ld.shared.u32 r4768, [r10450+8000];
ld.shared.u32 r5090, [r10450+8500];
ld.shared.u32 r5412, [r10450+9000];
ld.shared.u32 r5734, [r10450+9500];
ld.shared.u32 r4440, [r10450+10000];
ld.shared.u32 r4762, [r10450+10500];
ld.shared.u32 r5084, [r10450+11000];
ld.shared.u32 r5406, [r10450+11500];
ld.shared.u32 r5728, [r10450+12000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r4416, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r4417, {low, high};
}
{
neg.f16x2 r4418, r4417;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r4420, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r4421, {low, high};
}
{
neg.f16x2 r4422, r4421;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r4424, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r4425, {low, high};
}
{
add.f16x2 r4426, r4427, r4428;
}
{
add.f16x2 r4429, r4430, r4426;
}
{
add.f16x2 r4432, r4433, r4434;
}
{
add.f16x2 r4435, r4429, r4432;
}
{
add.f16x2 r4438, r4439, r4440;
}
{
add.f16x2 r4441, r4442, r4438;
}
{
add.f16x2 r4444, r4445, r4446;
}
{
add.f16x2 r4447, r4441, r4444;
}
{
add.f16x2 r4450, r4427, r4428;
}
{
mul.f16x2 r4453, r4450, r4416;
}
{
add.f16x2 r4456, r4430, r4453;
}
{
add.f16x2 r4459, r4433, r4434;
}
{
mul.f16x2 r4462, r4459, r4420;
}
{
add.f16x2 r4465, r4456, r4462;
}
{
sub.f16x2 r4468, r4439, r4440;
}
{
mul.f16x2 r4471, r4468, r4418;
}
{
sub.f16x2 r4474, r4445, r4446;
}
{
mul.f16x2 r4477, r4474, r4422;
}
{
add.f16x2 r4480, r4471, r4477;
}
{
sub.f16x2 r4483, r4465, r4480;
}
{
add.f16x2 r4486, r4427, r4428;
}
{
mul.f16x2 r4489, r4486, r4416;
}
{
add.f16x2 r4492, r4430, r4489;
}
{
add.f16x2 r4495, r4433, r4434;
}
{
mul.f16x2 r4498, r4495, r4420;
}
{
add.f16x2 r4501, r4492, r4498;
}
{
sub.f16x2 r4504, r4439, r4440;
}
{
mul.f16x2 r4507, r4504, r4418;
}
{
sub.f16x2 r4510, r4445, r4446;
}
{
mul.f16x2 r4513, r4510, r4422;
}
{
add.f16x2 r4516, r4507, r4513;
}
{
add.f16x2 r4519, r4501, r4516;
}
{
add.f16x2 r4522, r4427, r4428;
}
{
mul.f16x2 r4525, r4522, r4420;
}
{
add.f16x2 r4528, r4430, r4525;
}
{
add.f16x2 r4531, r4433, r4434;
}
{
mul.f16x2 r4534, r4531, r4424;
}
{
add.f16x2 r4537, r4528, r4534;
}
{
sub.f16x2 r4540, r4439, r4440;
}
{
mul.f16x2 r4543, r4540, r4422;
}
{
sub.f16x2 r4546, r4445, r4446;
}
{
mul.f16x2 r4549, r4546, r4425;
}
{
add.f16x2 r4552, r4543, r4549;
}
{
sub.f16x2 r4555, r4537, r4552;
}
{
add.f16x2 r4558, r4427, r4428;
}
{
mul.f16x2 r4561, r4558, r4420;
}
{
add.f16x2 r4564, r4430, r4561;
}
{
add.f16x2 r4567, r4433, r4434;
}
{
mul.f16x2 r4570, r4567, r4424;
}
{
add.f16x2 r4573, r4564, r4570;
}
{
sub.f16x2 r4576, r4439, r4440;
}
{
mul.f16x2 r4579, r4576, r4422;
}
{
sub.f16x2 r4582, r4445, r4446;
}
{
mul.f16x2 r4585, r4582, r4425;
}
{
add.f16x2 r4588, r4579, r4585;
}
{
add.f16x2 r4591, r4573, r4588;
}
{
add.f16x2 r4594, r4439, r4440;
}
{
mul.f16x2 r4597, r4594, r4416;
}
{
add.f16x2 r4600, r4442, r4597;
}
{
add.f16x2 r4603, r4445, r4446;
}
{
mul.f16x2 r4606, r4603, r4420;
}
{
add.f16x2 r4609, r4600, r4606;
}
{
sub.f16x2 r4612, r4427, r4428;
}
{
mul.f16x2 r4615, r4612, r4418;
}
{
sub.f16x2 r4618, r4433, r4434;
}
{
mul.f16x2 r4621, r4618, r4422;
}
{
add.f16x2 r4624, r4615, r4621;
}
{
add.f16x2 r4627, r4609, r4624;
}
{
add.f16x2 r4630, r4439, r4440;
}
{
mul.f16x2 r4633, r4630, r4416;
}
{
add.f16x2 r4636, r4442, r4633;
}
{
add.f16x2 r4639, r4445, r4446;
}
{
mul.f16x2 r4642, r4639, r4420;
}
{
add.f16x2 r4645, r4636, r4642;
}
{
sub.f16x2 r4648, r4427, r4428;
}
{
mul.f16x2 r4651, r4648, r4418;
}
{
sub.f16x2 r4654, r4433, r4434;
}
{
mul.f16x2 r4657, r4654, r4422;
}
{
add.f16x2 r4660, r4651, r4657;
}
{
sub.f16x2 r4663, r4645, r4660;
}
{
add.f16x2 r4666, r4439, r4440;
}
{
mul.f16x2 r4669, r4666, r4420;
}
{
add.f16x2 r4672, r4442, r4669;
}
{
add.f16x2 r4675, r4445, r4446;
}
{
mul.f16x2 r4678, r4675, r4424;
}
{
add.f16x2 r4681, r4672, r4678;
}
{
sub.f16x2 r4684, r4427, r4428;
}
{
mul.f16x2 r4687, r4684, r4422;
}
{
sub.f16x2 r4690, r4433, r4434;
}
{
mul.f16x2 r4693, r4690, r4425;
}
{
add.f16x2 r4696, r4687, r4693;
}
{
add.f16x2 r4699, r4681, r4696;
}
{
add.f16x2 r4702, r4439, r4440;
}
{
mul.f16x2 r4705, r4702, r4420;
}
{
add.f16x2 r4708, r4442, r4705;
}
{
add.f16x2 r4711, r4445, r4446;
}
{
mul.f16x2 r4714, r4711, r4424;
}
{
add.f16x2 r4717, r4708, r4714;
}
{
sub.f16x2 r4720, r4427, r4428;
}
{
mul.f16x2 r4723, r4720, r4422;
}
{
sub.f16x2 r4726, r4433, r4434;
}
{
mul.f16x2 r4729, r4726, r4425;
}
{
add.f16x2 r4732, r4723, r4729;
}
{
sub.f16x2 r4735, r4717, r4732;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r4738, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r4739, {low, high};
}
{
neg.f16x2 r4740, r4739;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r4742, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r4743, {low, high};
}
{
neg.f16x2 r4744, r4743;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r4746, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r4747, {low, high};
}
{
add.f16x2 r4748, r4749, r4750;
}
{
add.f16x2 r4751, r4752, r4748;
}
{
add.f16x2 r4754, r4755, r4756;
}
{
add.f16x2 r4757, r4751, r4754;
}
{
add.f16x2 r4760, r4761, r4762;
}
{
add.f16x2 r4763, r4764, r4760;
}
{
add.f16x2 r4766, r4767, r4768;
}
{
add.f16x2 r4769, r4763, r4766;
}
{
add.f16x2 r4772, r4749, r4750;
}
{
mul.f16x2 r4775, r4772, r4738;
}
{
add.f16x2 r4778, r4752, r4775;
}
{
add.f16x2 r4781, r4755, r4756;
}
{
mul.f16x2 r4784, r4781, r4742;
}
{
add.f16x2 r4787, r4778, r4784;
}
{
sub.f16x2 r4790, r4761, r4762;
}
{
mul.f16x2 r4793, r4790, r4740;
}
{
sub.f16x2 r4796, r4767, r4768;
}
{
mul.f16x2 r4799, r4796, r4744;
}
{
add.f16x2 r4802, r4793, r4799;
}
{
sub.f16x2 r4805, r4787, r4802;
}
{
add.f16x2 r4808, r4749, r4750;
}
{
mul.f16x2 r4811, r4808, r4738;
}
{
add.f16x2 r4814, r4752, r4811;
}
{
add.f16x2 r4817, r4755, r4756;
}
{
mul.f16x2 r4820, r4817, r4742;
}
{
add.f16x2 r4823, r4814, r4820;
}
{
sub.f16x2 r4826, r4761, r4762;
}
{
mul.f16x2 r4829, r4826, r4740;
}
{
sub.f16x2 r4832, r4767, r4768;
}
{
mul.f16x2 r4835, r4832, r4744;
}
{
add.f16x2 r4838, r4829, r4835;
}
{
add.f16x2 r4841, r4823, r4838;
}
{
add.f16x2 r4844, r4749, r4750;
}
{
mul.f16x2 r4847, r4844, r4742;
}
{
add.f16x2 r4850, r4752, r4847;
}
{
add.f16x2 r4853, r4755, r4756;
}
{
mul.f16x2 r4856, r4853, r4746;
}
{
add.f16x2 r4859, r4850, r4856;
}
{
sub.f16x2 r4862, r4761, r4762;
}
{
mul.f16x2 r4865, r4862, r4744;
}
{
sub.f16x2 r4868, r4767, r4768;
}
{
mul.f16x2 r4871, r4868, r4747;
}
{
add.f16x2 r4874, r4865, r4871;
}
{
sub.f16x2 r4877, r4859, r4874;
}
{
add.f16x2 r4880, r4749, r4750;
}
{
mul.f16x2 r4883, r4880, r4742;
}
{
add.f16x2 r4886, r4752, r4883;
}
{
add.f16x2 r4889, r4755, r4756;
}
{
mul.f16x2 r4892, r4889, r4746;
}
{
add.f16x2 r4895, r4886, r4892;
}
{
sub.f16x2 r4898, r4761, r4762;
}
{
mul.f16x2 r4901, r4898, r4744;
}
{
sub.f16x2 r4904, r4767, r4768;
}
{
mul.f16x2 r4907, r4904, r4747;
}
{
add.f16x2 r4910, r4901, r4907;
}
{
add.f16x2 r4913, r4895, r4910;
}
{
add.f16x2 r4916, r4761, r4762;
}
{
mul.f16x2 r4919, r4916, r4738;
}
{
add.f16x2 r4922, r4764, r4919;
}
{
add.f16x2 r4925, r4767, r4768;
}
{
mul.f16x2 r4928, r4925, r4742;
}
{
add.f16x2 r4931, r4922, r4928;
}
{
sub.f16x2 r4934, r4749, r4750;
}
{
mul.f16x2 r4937, r4934, r4740;
}
{
sub.f16x2 r4940, r4755, r4756;
}
{
mul.f16x2 r4943, r4940, r4744;
}
{
add.f16x2 r4946, r4937, r4943;
}
{
add.f16x2 r4949, r4931, r4946;
}
{
add.f16x2 r4952, r4761, r4762;
}
{
mul.f16x2 r4955, r4952, r4738;
}
{
add.f16x2 r4958, r4764, r4955;
}
{
add.f16x2 r4961, r4767, r4768;
}
{
mul.f16x2 r4964, r4961, r4742;
}
{
add.f16x2 r4967, r4958, r4964;
}
{
sub.f16x2 r4970, r4749, r4750;
}
{
mul.f16x2 r4973, r4970, r4740;
}
{
sub.f16x2 r4976, r4755, r4756;
}
{
mul.f16x2 r4979, r4976, r4744;
}
{
add.f16x2 r4982, r4973, r4979;
}
{
sub.f16x2 r4985, r4967, r4982;
}
{
add.f16x2 r4988, r4761, r4762;
}
{
mul.f16x2 r4991, r4988, r4742;
}
{
add.f16x2 r4994, r4764, r4991;
}
{
add.f16x2 r4997, r4767, r4768;
}
{
mul.f16x2 r5000, r4997, r4746;
}
{
add.f16x2 r5003, r4994, r5000;
}
{
sub.f16x2 r5006, r4749, r4750;
}
{
mul.f16x2 r5009, r5006, r4744;
}
{
sub.f16x2 r5012, r4755, r4756;
}
{
mul.f16x2 r5015, r5012, r4747;
}
{
add.f16x2 r5018, r5009, r5015;
}
{
add.f16x2 r5021, r5003, r5018;
}
{
add.f16x2 r5024, r4761, r4762;
}
{
mul.f16x2 r5027, r5024, r4742;
}
{
add.f16x2 r5030, r4764, r5027;
}
{
add.f16x2 r5033, r4767, r4768;
}
{
mul.f16x2 r5036, r5033, r4746;
}
{
add.f16x2 r5039, r5030, r5036;
}
{
sub.f16x2 r5042, r4749, r4750;
}
{
mul.f16x2 r5045, r5042, r4744;
}
{
sub.f16x2 r5048, r4755, r4756;
}
{
mul.f16x2 r5051, r5048, r4747;
}
{
add.f16x2 r5054, r5045, r5051;
}
{
sub.f16x2 r5057, r5039, r5054;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5060, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5061, {low, high};
}
{
neg.f16x2 r5062, r5061;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r5064, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r5065, {low, high};
}
{
neg.f16x2 r5066, r5065;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5068, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5069, {low, high};
}
{
add.f16x2 r5070, r5071, r5072;
}
{
add.f16x2 r5073, r5074, r5070;
}
{
add.f16x2 r5076, r5077, r5078;
}
{
add.f16x2 r5079, r5073, r5076;
}
{
add.f16x2 r5082, r5083, r5084;
}
{
add.f16x2 r5085, r5086, r5082;
}
{
add.f16x2 r5088, r5089, r5090;
}
{
add.f16x2 r5091, r5085, r5088;
}
{
add.f16x2 r5094, r5071, r5072;
}
{
mul.f16x2 r5097, r5094, r5060;
}
{
add.f16x2 r5100, r5074, r5097;
}
{
add.f16x2 r5103, r5077, r5078;
}
{
mul.f16x2 r5106, r5103, r5064;
}
{
add.f16x2 r5109, r5100, r5106;
}
{
sub.f16x2 r5112, r5083, r5084;
}
{
mul.f16x2 r5115, r5112, r5062;
}
{
sub.f16x2 r5118, r5089, r5090;
}
{
mul.f16x2 r5121, r5118, r5066;
}
{
add.f16x2 r5124, r5115, r5121;
}
{
sub.f16x2 r5127, r5109, r5124;
}
{
add.f16x2 r5130, r5071, r5072;
}
{
mul.f16x2 r5133, r5130, r5060;
}
{
add.f16x2 r5136, r5074, r5133;
}
{
add.f16x2 r5139, r5077, r5078;
}
{
mul.f16x2 r5142, r5139, r5064;
}
{
add.f16x2 r5145, r5136, r5142;
}
{
sub.f16x2 r5148, r5083, r5084;
}
{
mul.f16x2 r5151, r5148, r5062;
}
{
sub.f16x2 r5154, r5089, r5090;
}
{
mul.f16x2 r5157, r5154, r5066;
}
{
add.f16x2 r5160, r5151, r5157;
}
{
add.f16x2 r5163, r5145, r5160;
}
{
add.f16x2 r5166, r5071, r5072;
}
{
mul.f16x2 r5169, r5166, r5064;
}
{
add.f16x2 r5172, r5074, r5169;
}
{
add.f16x2 r5175, r5077, r5078;
}
{
mul.f16x2 r5178, r5175, r5068;
}
{
add.f16x2 r5181, r5172, r5178;
}
{
sub.f16x2 r5184, r5083, r5084;
}
{
mul.f16x2 r5187, r5184, r5066;
}
{
sub.f16x2 r5190, r5089, r5090;
}
{
mul.f16x2 r5193, r5190, r5069;
}
{
add.f16x2 r5196, r5187, r5193;
}
{
sub.f16x2 r5199, r5181, r5196;
}
{
add.f16x2 r5202, r5071, r5072;
}
{
mul.f16x2 r5205, r5202, r5064;
}
{
add.f16x2 r5208, r5074, r5205;
}
{
add.f16x2 r5211, r5077, r5078;
}
{
mul.f16x2 r5214, r5211, r5068;
}
{
add.f16x2 r5217, r5208, r5214;
}
{
sub.f16x2 r5220, r5083, r5084;
}
{
mul.f16x2 r5223, r5220, r5066;
}
{
sub.f16x2 r5226, r5089, r5090;
}
{
mul.f16x2 r5229, r5226, r5069;
}
{
add.f16x2 r5232, r5223, r5229;
}
{
add.f16x2 r5235, r5217, r5232;
}
{
add.f16x2 r5238, r5083, r5084;
}
{
mul.f16x2 r5241, r5238, r5060;
}
{
add.f16x2 r5244, r5086, r5241;
}
{
add.f16x2 r5247, r5089, r5090;
}
{
mul.f16x2 r5250, r5247, r5064;
}
{
add.f16x2 r5253, r5244, r5250;
}
{
sub.f16x2 r5256, r5071, r5072;
}
{
mul.f16x2 r5259, r5256, r5062;
}
{
sub.f16x2 r5262, r5077, r5078;
}
{
mul.f16x2 r5265, r5262, r5066;
}
{
add.f16x2 r5268, r5259, r5265;
}
{
add.f16x2 r5271, r5253, r5268;
}
{
add.f16x2 r5274, r5083, r5084;
}
{
mul.f16x2 r5277, r5274, r5060;
}
{
add.f16x2 r5280, r5086, r5277;
}
{
add.f16x2 r5283, r5089, r5090;
}
{
mul.f16x2 r5286, r5283, r5064;
}
{
add.f16x2 r5289, r5280, r5286;
}
{
sub.f16x2 r5292, r5071, r5072;
}
{
mul.f16x2 r5295, r5292, r5062;
}
{
sub.f16x2 r5298, r5077, r5078;
}
{
mul.f16x2 r5301, r5298, r5066;
}
{
add.f16x2 r5304, r5295, r5301;
}
{
sub.f16x2 r5307, r5289, r5304;
}
{
add.f16x2 r5310, r5083, r5084;
}
{
mul.f16x2 r5313, r5310, r5064;
}
{
add.f16x2 r5316, r5086, r5313;
}
{
add.f16x2 r5319, r5089, r5090;
}
{
mul.f16x2 r5322, r5319, r5068;
}
{
add.f16x2 r5325, r5316, r5322;
}
{
sub.f16x2 r5328, r5071, r5072;
}
{
mul.f16x2 r5331, r5328, r5066;
}
{
sub.f16x2 r5334, r5077, r5078;
}
{
mul.f16x2 r5337, r5334, r5069;
}
{
add.f16x2 r5340, r5331, r5337;
}
{
add.f16x2 r5343, r5325, r5340;
}
{
add.f16x2 r5346, r5083, r5084;
}
{
mul.f16x2 r5349, r5346, r5064;
}
{
add.f16x2 r5352, r5086, r5349;
}
{
add.f16x2 r5355, r5089, r5090;
}
{
mul.f16x2 r5358, r5355, r5068;
}
{
add.f16x2 r5361, r5352, r5358;
}
{
sub.f16x2 r5364, r5071, r5072;
}
{
mul.f16x2 r5367, r5364, r5066;
}
{
sub.f16x2 r5370, r5077, r5078;
}
{
mul.f16x2 r5373, r5370, r5069;
}
{
add.f16x2 r5376, r5367, r5373;
}
{
sub.f16x2 r5379, r5361, r5376;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5382, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5383, {low, high};
}
{
neg.f16x2 r5384, r5383;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r5386, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r5387, {low, high};
}
{
neg.f16x2 r5388, r5387;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5390, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5391, {low, high};
}
{
add.f16x2 r5392, r5393, r5394;
}
{
add.f16x2 r5395, r5396, r5392;
}
{
add.f16x2 r5398, r5399, r5400;
}
{
add.f16x2 r5401, r5395, r5398;
}
{
add.f16x2 r5404, r5405, r5406;
}
{
add.f16x2 r5407, r5408, r5404;
}
{
add.f16x2 r5410, r5411, r5412;
}
{
add.f16x2 r5413, r5407, r5410;
}
{
add.f16x2 r5416, r5393, r5394;
}
{
mul.f16x2 r5419, r5416, r5382;
}
{
add.f16x2 r5422, r5396, r5419;
}
{
add.f16x2 r5425, r5399, r5400;
}
{
mul.f16x2 r5428, r5425, r5386;
}
{
add.f16x2 r5431, r5422, r5428;
}
{
sub.f16x2 r5434, r5405, r5406;
}
{
mul.f16x2 r5437, r5434, r5384;
}
{
sub.f16x2 r5440, r5411, r5412;
}
{
mul.f16x2 r5443, r5440, r5388;
}
{
add.f16x2 r5446, r5437, r5443;
}
{
sub.f16x2 r5449, r5431, r5446;
}
{
add.f16x2 r5452, r5393, r5394;
}
{
mul.f16x2 r5455, r5452, r5382;
}
{
add.f16x2 r5458, r5396, r5455;
}
{
add.f16x2 r5461, r5399, r5400;
}
{
mul.f16x2 r5464, r5461, r5386;
}
{
add.f16x2 r5467, r5458, r5464;
}
{
sub.f16x2 r5470, r5405, r5406;
}
{
mul.f16x2 r5473, r5470, r5384;
}
{
sub.f16x2 r5476, r5411, r5412;
}
{
mul.f16x2 r5479, r5476, r5388;
}
{
add.f16x2 r5482, r5473, r5479;
}
{
add.f16x2 r5485, r5467, r5482;
}
{
add.f16x2 r5488, r5393, r5394;
}
{
mul.f16x2 r5491, r5488, r5386;
}
{
add.f16x2 r5494, r5396, r5491;
}
{
add.f16x2 r5497, r5399, r5400;
}
{
mul.f16x2 r5500, r5497, r5390;
}
{
add.f16x2 r5503, r5494, r5500;
}
{
sub.f16x2 r5506, r5405, r5406;
}
{
mul.f16x2 r5509, r5506, r5388;
}
{
sub.f16x2 r5512, r5411, r5412;
}
{
mul.f16x2 r5515, r5512, r5391;
}
{
add.f16x2 r5518, r5509, r5515;
}
{
sub.f16x2 r5521, r5503, r5518;
}
{
add.f16x2 r5524, r5393, r5394;
}
{
mul.f16x2 r5527, r5524, r5386;
}
{
add.f16x2 r5530, r5396, r5527;
}
{
add.f16x2 r5533, r5399, r5400;
}
{
mul.f16x2 r5536, r5533, r5390;
}
{
add.f16x2 r5539, r5530, r5536;
}
{
sub.f16x2 r5542, r5405, r5406;
}
{
mul.f16x2 r5545, r5542, r5388;
}
{
sub.f16x2 r5548, r5411, r5412;
}
{
mul.f16x2 r5551, r5548, r5391;
}
{
add.f16x2 r5554, r5545, r5551;
}
{
add.f16x2 r5557, r5539, r5554;
}
{
add.f16x2 r5560, r5405, r5406;
}
{
mul.f16x2 r5563, r5560, r5382;
}
{
add.f16x2 r5566, r5408, r5563;
}
{
add.f16x2 r5569, r5411, r5412;
}
{
mul.f16x2 r5572, r5569, r5386;
}
{
add.f16x2 r5575, r5566, r5572;
}
{
sub.f16x2 r5578, r5393, r5394;
}
{
mul.f16x2 r5581, r5578, r5384;
}
{
sub.f16x2 r5584, r5399, r5400;
}
{
mul.f16x2 r5587, r5584, r5388;
}
{
add.f16x2 r5590, r5581, r5587;
}
{
add.f16x2 r5593, r5575, r5590;
}
{
add.f16x2 r5596, r5405, r5406;
}
{
mul.f16x2 r5599, r5596, r5382;
}
{
add.f16x2 r5602, r5408, r5599;
}
{
add.f16x2 r5605, r5411, r5412;
}
{
mul.f16x2 r5608, r5605, r5386;
}
{
add.f16x2 r5611, r5602, r5608;
}
{
sub.f16x2 r5614, r5393, r5394;
}
{
mul.f16x2 r5617, r5614, r5384;
}
{
sub.f16x2 r5620, r5399, r5400;
}
{
mul.f16x2 r5623, r5620, r5388;
}
{
add.f16x2 r5626, r5617, r5623;
}
{
sub.f16x2 r5629, r5611, r5626;
}
{
add.f16x2 r5632, r5405, r5406;
}
{
mul.f16x2 r5635, r5632, r5386;
}
{
add.f16x2 r5638, r5408, r5635;
}
{
add.f16x2 r5641, r5411, r5412;
}
{
mul.f16x2 r5644, r5641, r5390;
}
{
add.f16x2 r5647, r5638, r5644;
}
{
sub.f16x2 r5650, r5393, r5394;
}
{
mul.f16x2 r5653, r5650, r5388;
}
{
sub.f16x2 r5656, r5399, r5400;
}
{
mul.f16x2 r5659, r5656, r5391;
}
{
add.f16x2 r5662, r5653, r5659;
}
{
add.f16x2 r5665, r5647, r5662;
}
{
add.f16x2 r5668, r5405, r5406;
}
{
mul.f16x2 r5671, r5668, r5386;
}
{
add.f16x2 r5674, r5408, r5671;
}
{
add.f16x2 r5677, r5411, r5412;
}
{
mul.f16x2 r5680, r5677, r5390;
}
{
add.f16x2 r5683, r5674, r5680;
}
{
sub.f16x2 r5686, r5393, r5394;
}
{
mul.f16x2 r5689, r5686, r5388;
}
{
sub.f16x2 r5692, r5399, r5400;
}
{
mul.f16x2 r5695, r5692, r5391;
}
{
add.f16x2 r5698, r5689, r5695;
}
{
sub.f16x2 r5701, r5683, r5698;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5704, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5705, {low, high};
}
{
neg.f16x2 r5706, r5705;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r5708, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r5709, {low, high};
}
{
neg.f16x2 r5710, r5709;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5712, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5713, {low, high};
}
{
add.f16x2 r5714, r5715, r5716;
}
{
add.f16x2 r5717, r5718, r5714;
}
{
add.f16x2 r5720, r5721, r5722;
}
{
add.f16x2 r5723, r5717, r5720;
}
{
add.f16x2 r5726, r5727, r5728;
}
{
add.f16x2 r5729, r5730, r5726;
}
{
add.f16x2 r5732, r5733, r5734;
}
{
add.f16x2 r5735, r5729, r5732;
}
{
add.f16x2 r5738, r5715, r5716;
}
{
mul.f16x2 r5741, r5738, r5704;
}
{
add.f16x2 r5744, r5718, r5741;
}
{
add.f16x2 r5747, r5721, r5722;
}
{
mul.f16x2 r5750, r5747, r5708;
}
{
add.f16x2 r5753, r5744, r5750;
}
{
sub.f16x2 r5756, r5727, r5728;
}
{
mul.f16x2 r5759, r5756, r5706;
}
{
sub.f16x2 r5762, r5733, r5734;
}
{
mul.f16x2 r5765, r5762, r5710;
}
{
add.f16x2 r5768, r5759, r5765;
}
{
sub.f16x2 r5771, r5753, r5768;
}
{
add.f16x2 r5774, r5715, r5716;
}
{
mul.f16x2 r5777, r5774, r5704;
}
{
add.f16x2 r5780, r5718, r5777;
}
{
add.f16x2 r5783, r5721, r5722;
}
{
mul.f16x2 r5786, r5783, r5708;
}
{
add.f16x2 r5789, r5780, r5786;
}
{
sub.f16x2 r5792, r5727, r5728;
}
{
mul.f16x2 r5795, r5792, r5706;
}
{
sub.f16x2 r5798, r5733, r5734;
}
{
mul.f16x2 r5801, r5798, r5710;
}
{
add.f16x2 r5804, r5795, r5801;
}
{
add.f16x2 r5807, r5789, r5804;
}
{
add.f16x2 r5810, r5715, r5716;
}
{
mul.f16x2 r5813, r5810, r5708;
}
{
add.f16x2 r5816, r5718, r5813;
}
{
add.f16x2 r5819, r5721, r5722;
}
{
mul.f16x2 r5822, r5819, r5712;
}
{
add.f16x2 r5825, r5816, r5822;
}
{
sub.f16x2 r5828, r5727, r5728;
}
{
mul.f16x2 r5831, r5828, r5710;
}
{
sub.f16x2 r5834, r5733, r5734;
}
{
mul.f16x2 r5837, r5834, r5713;
}
{
add.f16x2 r5840, r5831, r5837;
}
{
sub.f16x2 r5843, r5825, r5840;
}
{
add.f16x2 r5846, r5715, r5716;
}
{
mul.f16x2 r5849, r5846, r5708;
}
{
add.f16x2 r5852, r5718, r5849;
}
{
add.f16x2 r5855, r5721, r5722;
}
{
mul.f16x2 r5858, r5855, r5712;
}
{
add.f16x2 r5861, r5852, r5858;
}
{
sub.f16x2 r5864, r5727, r5728;
}
{
mul.f16x2 r5867, r5864, r5710;
}
{
sub.f16x2 r5870, r5733, r5734;
}
{
mul.f16x2 r5873, r5870, r5713;
}
{
add.f16x2 r5876, r5867, r5873;
}
{
add.f16x2 r5879, r5861, r5876;
}
{
add.f16x2 r5882, r5727, r5728;
}
{
mul.f16x2 r5885, r5882, r5704;
}
{
add.f16x2 r5888, r5730, r5885;
}
{
add.f16x2 r5891, r5733, r5734;
}
{
mul.f16x2 r5894, r5891, r5708;
}
{
add.f16x2 r5897, r5888, r5894;
}
{
sub.f16x2 r5900, r5715, r5716;
}
{
mul.f16x2 r5903, r5900, r5706;
}
{
sub.f16x2 r5906, r5721, r5722;
}
{
mul.f16x2 r5909, r5906, r5710;
}
{
add.f16x2 r5912, r5903, r5909;
}
{
add.f16x2 r5915, r5897, r5912;
}
{
add.f16x2 r5918, r5727, r5728;
}
{
mul.f16x2 r5921, r5918, r5704;
}
{
add.f16x2 r5924, r5730, r5921;
}
{
add.f16x2 r5927, r5733, r5734;
}
{
mul.f16x2 r5930, r5927, r5708;
}
{
add.f16x2 r5933, r5924, r5930;
}
{
sub.f16x2 r5936, r5715, r5716;
}
{
mul.f16x2 r5939, r5936, r5706;
}
{
sub.f16x2 r5942, r5721, r5722;
}
{
mul.f16x2 r5945, r5942, r5710;
}
{
add.f16x2 r5948, r5939, r5945;
}
{
sub.f16x2 r5951, r5933, r5948;
}
{
add.f16x2 r5954, r5727, r5728;
}
{
mul.f16x2 r5957, r5954, r5708;
}
{
add.f16x2 r5960, r5730, r5957;
}
{
add.f16x2 r5963, r5733, r5734;
}
{
mul.f16x2 r5966, r5963, r5712;
}
{
add.f16x2 r5969, r5960, r5966;
}
{
sub.f16x2 r5972, r5715, r5716;
}
{
mul.f16x2 r5975, r5972, r5710;
}
{
sub.f16x2 r5978, r5721, r5722;
}
{
mul.f16x2 r5981, r5978, r5713;
}
{
add.f16x2 r5984, r5975, r5981;
}
{
add.f16x2 r5987, r5969, r5984;
}
{
add.f16x2 r5990, r5727, r5728;
}
{
mul.f16x2 r5993, r5990, r5708;
}
{
add.f16x2 r5996, r5730, r5993;
}
{
add.f16x2 r5999, r5733, r5734;
}
{
mul.f16x2 r6002, r5999, r5712;
}
{
add.f16x2 r6005, r5996, r6002;
}
{
sub.f16x2 r6008, r5715, r5716;
}
{
mul.f16x2 r6011, r6008, r5710;
}
{
sub.f16x2 r6014, r5721, r5722;
}
{
mul.f16x2 r6017, r6014, r5713;
}
{
add.f16x2 r6020, r6011, r6017;
}
{
sub.f16x2 r6023, r6005, r6020;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r6026, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f332;
cvt.rn.f16.f32 high, f332;
mov.b32 r6027, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f334;
cvt.rn.f16.f32 high, f334;
mov.b32 r6028, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f336;
cvt.rn.f16.f32 high, f336;
mov.b32 r6029, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f338;
cvt.rn.f16.f32 high, f338;
mov.b32 r6030, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r6031, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f342;
cvt.rn.f16.f32 high, f342;
mov.b32 r6032, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f344;
cvt.rn.f16.f32 high, f344;
mov.b32 r6033, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f350;
cvt.rn.f16.f32 high, f350;
mov.b32 r6036, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r6037, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f358;
cvt.rn.f16.f32 high, f358;
mov.b32 r6040, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f360;
cvt.rn.f16.f32 high, f360;
mov.b32 r6041, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f390;
cvt.rn.f16.f32 high, f390;
mov.b32 r6042, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f364;
cvt.rn.f16.f32 high, f364;
mov.b32 r6043, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f374;
cvt.rn.f16.f32 high, f374;
mov.b32 r6048, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r6049, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f390;
cvt.rn.f16.f32 high, f390;
mov.b32 r6056, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f392;
cvt.rn.f16.f32 high, f392;
mov.b32 r6057, {low, high};
}
{
mul.f16x2 r6074, r4805, r6026;
}
{
mul.f16x2 r6077, r4949, r6027;
}
{
sub.f16x2 r6080, r6074, r6077;
}
{
mul.f16x2 r6083, r4805, r6027;
}
{
fma.rn.f16x2 r6086, r4949, r6026, r6083;
}
{
mul.f16x2 r6090, r5127, r6028;
}
{
mul.f16x2 r6093, r5271, r6029;
}
{
sub.f16x2 r6096, r6090, r6093;
}
{
mul.f16x2 r6099, r5127, r6029;
}
{
fma.rn.f16x2 r6102, r5271, r6028, r6099;
}
{
mul.f16x2 r6106, r5449, r6030;
}
{
mul.f16x2 r6109, r5593, r6031;
}
{
sub.f16x2 r6112, r6106, r6109;
}
{
mul.f16x2 r6115, r5449, r6031;
}
{
fma.rn.f16x2 r6118, r5593, r6030, r6115;
}
{
mul.f16x2 r6122, r5771, r6032;
}
{
mul.f16x2 r6125, r5915, r6033;
}
{
sub.f16x2 r6128, r6122, r6125;
}
{
mul.f16x2 r6131, r5771, r6033;
}
{
fma.rn.f16x2 r6134, r5915, r6032, r6131;
}
{
mul.f16x2 r6138, r4877, r6028;
}
{
mul.f16x2 r6141, r5021, r6029;
}
{
sub.f16x2 r6144, r6138, r6141;
}
{
mul.f16x2 r6147, r4877, r6029;
}
{
fma.rn.f16x2 r6150, r5021, r6028, r6147;
}
{
mul.f16x2 r6154, r5199, r6032;
}
{
mul.f16x2 r6157, r5343, r6033;
}
{
sub.f16x2 r6160, r6154, r6157;
}
{
mul.f16x2 r6163, r5199, r6033;
}
{
fma.rn.f16x2 r6166, r5343, r6032, r6163;
}
{
mul.f16x2 r6170, r5521, r6036;
}
{
mul.f16x2 r6173, r5665, r6037;
}
{
sub.f16x2 r6176, r6170, r6173;
}
{
mul.f16x2 r6179, r5521, r6037;
}
{
fma.rn.f16x2 r6182, r5665, r6036, r6179;
}
{
mul.f16x2 r6186, r5843, r6040;
}
{
mul.f16x2 r6189, r5987, r6041;
}
{
sub.f16x2 r6192, r6186, r6189;
}
{
mul.f16x2 r6195, r5843, r6041;
}
{
fma.rn.f16x2 r6198, r5987, r6040, r6195;
}
{
mul.f16x2 r6202, r4913, r6030;
}
{
mul.f16x2 r6205, r5057, r6031;
}
{
sub.f16x2 r6208, r6202, r6205;
}
{
mul.f16x2 r6211, r4913, r6031;
}
{
fma.rn.f16x2 r6214, r5057, r6030, r6211;
}
{
mul.f16x2 r6218, r5235, r6036;
}
{
mul.f16x2 r6221, r5379, r6037;
}
{
sub.f16x2 r6224, r6218, r6221;
}
{
mul.f16x2 r6227, r5235, r6037;
}
{
fma.rn.f16x2 r6230, r5379, r6036, r6227;
}
{
mul.f16x2 r6234, r5557, r6042;
}
{
mul.f16x2 r6237, r5701, r6043;
}
{
sub.f16x2 r6240, r6234, r6237;
}
{
mul.f16x2 r6243, r5557, r6043;
}
{
fma.rn.f16x2 r6246, r5701, r6042, r6243;
}
{
mul.f16x2 r6250, r5879, r6048;
}
{
mul.f16x2 r6253, r6023, r6049;
}
{
sub.f16x2 r6256, r6250, r6253;
}
{
mul.f16x2 r6259, r5879, r6049;
}
{
fma.rn.f16x2 r6262, r6023, r6048, r6259;
}
{
mul.f16x2 r6266, r4841, r6032;
}
{
mul.f16x2 r6269, r4985, r6033;
}
{
sub.f16x2 r6272, r6266, r6269;
}
{
mul.f16x2 r6275, r4841, r6033;
}
{
fma.rn.f16x2 r6278, r4985, r6032, r6275;
}
{
mul.f16x2 r6282, r5163, r6040;
}
{
mul.f16x2 r6285, r5307, r6041;
}
{
sub.f16x2 r6288, r6282, r6285;
}
{
mul.f16x2 r6291, r5163, r6041;
}
{
fma.rn.f16x2 r6294, r5307, r6040, r6291;
}
{
mul.f16x2 r6298, r5485, r6048;
}
{
mul.f16x2 r6301, r5629, r6049;
}
{
sub.f16x2 r6304, r6298, r6301;
}
{
mul.f16x2 r6307, r5485, r6049;
}
{
fma.rn.f16x2 r6310, r5629, r6048, r6307;
}
{
mul.f16x2 r6314, r5807, r6056;
}
{
mul.f16x2 r6317, r5951, r6057;
}
{
sub.f16x2 r6320, r6314, r6317;
}
{
mul.f16x2 r6323, r5807, r6057;
}
{
fma.rn.f16x2 r6326, r5951, r6056, r6323;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6330, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6331, {low, high};
}
{
neg.f16x2 r6332, r6331;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r6334, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r6335, {low, high};
}
{
neg.f16x2 r6336, r6335;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6338, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6339, {low, high};
}
{
add.f16x2 r6340, r4757, r5723;
}
{
add.f16x2 r6343, r4435, r6340;
}
{
add.f16x2 r6346, r5079, r5401;
}
{
add.f16x2 r6349, r6343, r6346;
}
{
add.f16x2 r6352, r4769, r5735;
}
{
add.f16x2 r6355, r4447, r6352;
}
{
add.f16x2 r6358, r5091, r5413;
}
{
add.f16x2 r6361, r6355, r6358;
}
{
add.f16x2 r6364, r4757, r5723;
}
{
mul.f16x2 r6367, r6364, r6330;
}
{
add.f16x2 r6370, r4435, r6367;
}
{
add.f16x2 r6373, r5079, r5401;
}
{
mul.f16x2 r6376, r6373, r6334;
}
{
add.f16x2 r6379, r6370, r6376;
}
{
sub.f16x2 r6382, r4769, r5735;
}
{
mul.f16x2 r6385, r6382, r6332;
}
{
sub.f16x2 r6388, r5091, r5413;
}
{
mul.f16x2 r6391, r6388, r6336;
}
{
add.f16x2 r6394, r6385, r6391;
}
{
sub.f16x2 r6397, r6379, r6394;
}
{
add.f16x2 r6400, r4757, r5723;
}
{
mul.f16x2 r6403, r6400, r6330;
}
{
add.f16x2 r6406, r4435, r6403;
}
{
add.f16x2 r6409, r5079, r5401;
}
{
mul.f16x2 r6412, r6409, r6334;
}
{
add.f16x2 r6415, r6406, r6412;
}
{
sub.f16x2 r6418, r4769, r5735;
}
{
mul.f16x2 r6421, r6418, r6332;
}
{
sub.f16x2 r6424, r5091, r5413;
}
{
mul.f16x2 r6427, r6424, r6336;
}
{
add.f16x2 r6430, r6421, r6427;
}
{
add.f16x2 r6433, r6415, r6430;
}
{
add.f16x2 r6436, r4757, r5723;
}
{
mul.f16x2 r6439, r6436, r6334;
}
{
add.f16x2 r6442, r4435, r6439;
}
{
add.f16x2 r6445, r5079, r5401;
}
{
mul.f16x2 r6448, r6445, r6338;
}
{
add.f16x2 r6451, r6442, r6448;
}
{
sub.f16x2 r6454, r4769, r5735;
}
{
mul.f16x2 r6457, r6454, r6336;
}
{
sub.f16x2 r6460, r5091, r5413;
}
{
mul.f16x2 r6463, r6460, r6339;
}
{
add.f16x2 r6466, r6457, r6463;
}
{
sub.f16x2 r6469, r6451, r6466;
}
{
add.f16x2 r6472, r4757, r5723;
}
{
mul.f16x2 r6475, r6472, r6334;
}
{
add.f16x2 r6478, r4435, r6475;
}
{
add.f16x2 r6481, r5079, r5401;
}
{
mul.f16x2 r6484, r6481, r6338;
}
{
add.f16x2 r6487, r6478, r6484;
}
{
sub.f16x2 r6490, r4769, r5735;
}
{
mul.f16x2 r6493, r6490, r6336;
}
{
sub.f16x2 r6496, r5091, r5413;
}
{
mul.f16x2 r6499, r6496, r6339;
}
{
add.f16x2 r6502, r6493, r6499;
}
{
add.f16x2 r6505, r6487, r6502;
}
{
add.f16x2 r6508, r4769, r5735;
}
{
mul.f16x2 r6511, r6508, r6330;
}
{
add.f16x2 r6514, r4447, r6511;
}
{
add.f16x2 r6517, r5091, r5413;
}
{
mul.f16x2 r6520, r6517, r6334;
}
{
add.f16x2 r6523, r6514, r6520;
}
{
sub.f16x2 r6526, r4757, r5723;
}
{
mul.f16x2 r6529, r6526, r6332;
}
{
sub.f16x2 r6532, r5079, r5401;
}
{
mul.f16x2 r6535, r6532, r6336;
}
{
add.f16x2 r6538, r6529, r6535;
}
{
add.f16x2 r6541, r6523, r6538;
}
{
add.f16x2 r6544, r4769, r5735;
}
{
mul.f16x2 r6547, r6544, r6330;
}
{
add.f16x2 r6550, r4447, r6547;
}
{
add.f16x2 r6553, r5091, r5413;
}
{
mul.f16x2 r6556, r6553, r6334;
}
{
add.f16x2 r6559, r6550, r6556;
}
{
sub.f16x2 r6562, r4757, r5723;
}
{
mul.f16x2 r6565, r6562, r6332;
}
{
sub.f16x2 r6568, r5079, r5401;
}
{
mul.f16x2 r6571, r6568, r6336;
}
{
add.f16x2 r6574, r6565, r6571;
}
{
sub.f16x2 r6577, r6559, r6574;
}
{
add.f16x2 r6580, r4769, r5735;
}
{
mul.f16x2 r6583, r6580, r6334;
}
{
add.f16x2 r6586, r4447, r6583;
}
{
add.f16x2 r6589, r5091, r5413;
}
{
mul.f16x2 r6592, r6589, r6338;
}
{
add.f16x2 r6595, r6586, r6592;
}
{
sub.f16x2 r6598, r4757, r5723;
}
{
mul.f16x2 r6601, r6598, r6336;
}
{
sub.f16x2 r6604, r5079, r5401;
}
{
mul.f16x2 r6607, r6604, r6339;
}
{
add.f16x2 r6610, r6601, r6607;
}
{
add.f16x2 r6613, r6595, r6610;
}
{
add.f16x2 r6616, r4769, r5735;
}
{
mul.f16x2 r6619, r6616, r6334;
}
{
add.f16x2 r6622, r4447, r6619;
}
{
add.f16x2 r6625, r5091, r5413;
}
{
mul.f16x2 r6628, r6625, r6338;
}
{
add.f16x2 r6631, r6622, r6628;
}
{
sub.f16x2 r6634, r4757, r5723;
}
{
mul.f16x2 r6637, r6634, r6336;
}
{
sub.f16x2 r6640, r5079, r5401;
}
{
mul.f16x2 r6643, r6640, r6339;
}
{
add.f16x2 r6646, r6637, r6643;
}
{
sub.f16x2 r6649, r6631, r6646;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6652, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6653, {low, high};
}
{
neg.f16x2 r6654, r6653;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r6656, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r6657, {low, high};
}
{
neg.f16x2 r6658, r6657;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6660, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6661, {low, high};
}
{
add.f16x2 r6662, r6080, r6128;
}
{
add.f16x2 r6665, r4483, r6662;
}
{
add.f16x2 r6668, r6096, r6112;
}
{
add.f16x2 r6671, r6665, r6668;
}
{
add.f16x2 r6674, r6086, r6134;
}
{
add.f16x2 r6677, r4627, r6674;
}
{
add.f16x2 r6680, r6102, r6118;
}
{
add.f16x2 r6683, r6677, r6680;
}
{
add.f16x2 r6686, r6080, r6128;
}
{
mul.f16x2 r6689, r6686, r6652;
}
{
add.f16x2 r6692, r4483, r6689;
}
{
add.f16x2 r6695, r6096, r6112;
}
{
mul.f16x2 r6698, r6695, r6656;
}
{
add.f16x2 r6701, r6692, r6698;
}
{
sub.f16x2 r6704, r6086, r6134;
}
{
mul.f16x2 r6707, r6704, r6654;
}
{
sub.f16x2 r6710, r6102, r6118;
}
{
mul.f16x2 r6713, r6710, r6658;
}
{
add.f16x2 r6716, r6707, r6713;
}
{
sub.f16x2 r6719, r6701, r6716;
}
{
add.f16x2 r6722, r6080, r6128;
}
{
mul.f16x2 r6725, r6722, r6652;
}
{
add.f16x2 r6728, r4483, r6725;
}
{
add.f16x2 r6731, r6096, r6112;
}
{
mul.f16x2 r6734, r6731, r6656;
}
{
add.f16x2 r6737, r6728, r6734;
}
{
sub.f16x2 r6740, r6086, r6134;
}
{
mul.f16x2 r6743, r6740, r6654;
}
{
sub.f16x2 r6746, r6102, r6118;
}
{
mul.f16x2 r6749, r6746, r6658;
}
{
add.f16x2 r6752, r6743, r6749;
}
{
add.f16x2 r6755, r6737, r6752;
}
{
add.f16x2 r6758, r6080, r6128;
}
{
mul.f16x2 r6761, r6758, r6656;
}
{
add.f16x2 r6764, r4483, r6761;
}
{
add.f16x2 r6767, r6096, r6112;
}
{
mul.f16x2 r6770, r6767, r6660;
}
{
add.f16x2 r6773, r6764, r6770;
}
{
sub.f16x2 r6776, r6086, r6134;
}
{
mul.f16x2 r6779, r6776, r6658;
}
{
sub.f16x2 r6782, r6102, r6118;
}
{
mul.f16x2 r6785, r6782, r6661;
}
{
add.f16x2 r6788, r6779, r6785;
}
{
sub.f16x2 r6791, r6773, r6788;
}
{
add.f16x2 r6794, r6080, r6128;
}
{
mul.f16x2 r6797, r6794, r6656;
}
{
add.f16x2 r6800, r4483, r6797;
}
{
add.f16x2 r6803, r6096, r6112;
}
{
mul.f16x2 r6806, r6803, r6660;
}
{
add.f16x2 r6809, r6800, r6806;
}
{
sub.f16x2 r6812, r6086, r6134;
}
{
mul.f16x2 r6815, r6812, r6658;
}
{
sub.f16x2 r6818, r6102, r6118;
}
{
mul.f16x2 r6821, r6818, r6661;
}
{
add.f16x2 r6824, r6815, r6821;
}
{
add.f16x2 r6827, r6809, r6824;
}
{
add.f16x2 r6830, r6086, r6134;
}
{
mul.f16x2 r6833, r6830, r6652;
}
{
add.f16x2 r6836, r4627, r6833;
}
{
add.f16x2 r6839, r6102, r6118;
}
{
mul.f16x2 r6842, r6839, r6656;
}
{
add.f16x2 r6845, r6836, r6842;
}
{
sub.f16x2 r6848, r6080, r6128;
}
{
mul.f16x2 r6851, r6848, r6654;
}
{
sub.f16x2 r6854, r6096, r6112;
}
{
mul.f16x2 r6857, r6854, r6658;
}
{
add.f16x2 r6860, r6851, r6857;
}
{
add.f16x2 r6863, r6845, r6860;
}
{
add.f16x2 r6866, r6086, r6134;
}
{
mul.f16x2 r6869, r6866, r6652;
}
{
add.f16x2 r6872, r4627, r6869;
}
{
add.f16x2 r6875, r6102, r6118;
}
{
mul.f16x2 r6878, r6875, r6656;
}
{
add.f16x2 r6881, r6872, r6878;
}
{
sub.f16x2 r6884, r6080, r6128;
}
{
mul.f16x2 r6887, r6884, r6654;
}
{
sub.f16x2 r6890, r6096, r6112;
}
{
mul.f16x2 r6893, r6890, r6658;
}
{
add.f16x2 r6896, r6887, r6893;
}
{
sub.f16x2 r6899, r6881, r6896;
}
{
add.f16x2 r6902, r6086, r6134;
}
{
mul.f16x2 r6905, r6902, r6656;
}
{
add.f16x2 r6908, r4627, r6905;
}
{
add.f16x2 r6911, r6102, r6118;
}
{
mul.f16x2 r6914, r6911, r6660;
}
{
add.f16x2 r6917, r6908, r6914;
}
{
sub.f16x2 r6920, r6080, r6128;
}
{
mul.f16x2 r6923, r6920, r6658;
}
{
sub.f16x2 r6926, r6096, r6112;
}
{
mul.f16x2 r6929, r6926, r6661;
}
{
add.f16x2 r6932, r6923, r6929;
}
{
add.f16x2 r6935, r6917, r6932;
}
{
add.f16x2 r6938, r6086, r6134;
}
{
mul.f16x2 r6941, r6938, r6656;
}
{
add.f16x2 r6944, r4627, r6941;
}
{
add.f16x2 r6947, r6102, r6118;
}
{
mul.f16x2 r6950, r6947, r6660;
}
{
add.f16x2 r6953, r6944, r6950;
}
{
sub.f16x2 r6956, r6080, r6128;
}
{
mul.f16x2 r6959, r6956, r6658;
}
{
sub.f16x2 r6962, r6096, r6112;
}
{
mul.f16x2 r6965, r6962, r6661;
}
{
add.f16x2 r6968, r6959, r6965;
}
{
sub.f16x2 r6971, r6953, r6968;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6974, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6975, {low, high};
}
{
neg.f16x2 r6976, r6975;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r6978, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r6979, {low, high};
}
{
neg.f16x2 r6980, r6979;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6982, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6983, {low, high};
}
{
add.f16x2 r6984, r6144, r6192;
}
{
add.f16x2 r6987, r4555, r6984;
}
{
add.f16x2 r6990, r6160, r6176;
}
{
add.f16x2 r6993, r6987, r6990;
}
{
add.f16x2 r6996, r6150, r6198;
}
{
add.f16x2 r6999, r4699, r6996;
}
{
add.f16x2 r7002, r6166, r6182;
}
{
add.f16x2 r7005, r6999, r7002;
}
{
add.f16x2 r7008, r6144, r6192;
}
{
mul.f16x2 r7011, r7008, r6974;
}
{
add.f16x2 r7014, r4555, r7011;
}
{
add.f16x2 r7017, r6160, r6176;
}
{
mul.f16x2 r7020, r7017, r6978;
}
{
add.f16x2 r7023, r7014, r7020;
}
{
sub.f16x2 r7026, r6150, r6198;
}
{
mul.f16x2 r7029, r7026, r6976;
}
{
sub.f16x2 r7032, r6166, r6182;
}
{
mul.f16x2 r7035, r7032, r6980;
}
{
add.f16x2 r7038, r7029, r7035;
}
{
sub.f16x2 r7041, r7023, r7038;
}
{
add.f16x2 r7044, r6144, r6192;
}
{
mul.f16x2 r7047, r7044, r6974;
}
{
add.f16x2 r7050, r4555, r7047;
}
{
add.f16x2 r7053, r6160, r6176;
}
{
mul.f16x2 r7056, r7053, r6978;
}
{
add.f16x2 r7059, r7050, r7056;
}
{
sub.f16x2 r7062, r6150, r6198;
}
{
mul.f16x2 r7065, r7062, r6976;
}
{
sub.f16x2 r7068, r6166, r6182;
}
{
mul.f16x2 r7071, r7068, r6980;
}
{
add.f16x2 r7074, r7065, r7071;
}
{
add.f16x2 r7077, r7059, r7074;
}
{
add.f16x2 r7080, r6144, r6192;
}
{
mul.f16x2 r7083, r7080, r6978;
}
{
add.f16x2 r7086, r4555, r7083;
}
{
add.f16x2 r7089, r6160, r6176;
}
{
mul.f16x2 r7092, r7089, r6982;
}
{
add.f16x2 r7095, r7086, r7092;
}
{
sub.f16x2 r7098, r6150, r6198;
}
{
mul.f16x2 r7101, r7098, r6980;
}
{
sub.f16x2 r7104, r6166, r6182;
}
{
mul.f16x2 r7107, r7104, r6983;
}
{
add.f16x2 r7110, r7101, r7107;
}
{
sub.f16x2 r7113, r7095, r7110;
}
{
add.f16x2 r7116, r6144, r6192;
}
{
mul.f16x2 r7119, r7116, r6978;
}
{
add.f16x2 r7122, r4555, r7119;
}
{
add.f16x2 r7125, r6160, r6176;
}
{
mul.f16x2 r7128, r7125, r6982;
}
{
add.f16x2 r7131, r7122, r7128;
}
{
sub.f16x2 r7134, r6150, r6198;
}
{
mul.f16x2 r7137, r7134, r6980;
}
{
sub.f16x2 r7140, r6166, r6182;
}
{
mul.f16x2 r7143, r7140, r6983;
}
{
add.f16x2 r7146, r7137, r7143;
}
{
add.f16x2 r7149, r7131, r7146;
}
{
add.f16x2 r7152, r6150, r6198;
}
{
mul.f16x2 r7155, r7152, r6974;
}
{
add.f16x2 r7158, r4699, r7155;
}
{
add.f16x2 r7161, r6166, r6182;
}
{
mul.f16x2 r7164, r7161, r6978;
}
{
add.f16x2 r7167, r7158, r7164;
}
{
sub.f16x2 r7170, r6144, r6192;
}
{
mul.f16x2 r7173, r7170, r6976;
}
{
sub.f16x2 r7176, r6160, r6176;
}
{
mul.f16x2 r7179, r7176, r6980;
}
{
add.f16x2 r7182, r7173, r7179;
}
{
add.f16x2 r7185, r7167, r7182;
}
{
add.f16x2 r7188, r6150, r6198;
}
{
mul.f16x2 r7191, r7188, r6974;
}
{
add.f16x2 r7194, r4699, r7191;
}
{
add.f16x2 r7197, r6166, r6182;
}
{
mul.f16x2 r7200, r7197, r6978;
}
{
add.f16x2 r7203, r7194, r7200;
}
{
sub.f16x2 r7206, r6144, r6192;
}
{
mul.f16x2 r7209, r7206, r6976;
}
{
sub.f16x2 r7212, r6160, r6176;
}
{
mul.f16x2 r7215, r7212, r6980;
}
{
add.f16x2 r7218, r7209, r7215;
}
{
sub.f16x2 r7221, r7203, r7218;
}
{
add.f16x2 r7224, r6150, r6198;
}
{
mul.f16x2 r7227, r7224, r6978;
}
{
add.f16x2 r7230, r4699, r7227;
}
{
add.f16x2 r7233, r6166, r6182;
}
{
mul.f16x2 r7236, r7233, r6982;
}
{
add.f16x2 r7239, r7230, r7236;
}
{
sub.f16x2 r7242, r6144, r6192;
}
{
mul.f16x2 r7245, r7242, r6980;
}
{
sub.f16x2 r7248, r6160, r6176;
}
{
mul.f16x2 r7251, r7248, r6983;
}
{
add.f16x2 r7254, r7245, r7251;
}
{
add.f16x2 r7257, r7239, r7254;
}
{
add.f16x2 r7260, r6150, r6198;
}
{
mul.f16x2 r7263, r7260, r6978;
}
{
add.f16x2 r7266, r4699, r7263;
}
{
add.f16x2 r7269, r6166, r6182;
}
{
mul.f16x2 r7272, r7269, r6982;
}
{
add.f16x2 r7275, r7266, r7272;
}
{
sub.f16x2 r7278, r6144, r6192;
}
{
mul.f16x2 r7281, r7278, r6980;
}
{
sub.f16x2 r7284, r6160, r6176;
}
{
mul.f16x2 r7287, r7284, r6983;
}
{
add.f16x2 r7290, r7281, r7287;
}
{
sub.f16x2 r7293, r7275, r7290;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r7296, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r7297, {low, high};
}
{
neg.f16x2 r7298, r7297;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r7300, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r7301, {low, high};
}
{
neg.f16x2 r7302, r7301;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r7304, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r7305, {low, high};
}
{
add.f16x2 r7306, r6208, r6256;
}
{
add.f16x2 r7309, r4591, r7306;
}
{
add.f16x2 r7312, r6224, r6240;
}
{
add.f16x2 r7315, r7309, r7312;
}
{
add.f16x2 r7318, r6214, r6262;
}
{
add.f16x2 r7321, r4735, r7318;
}
{
add.f16x2 r7324, r6230, r6246;
}
{
add.f16x2 r7327, r7321, r7324;
}
{
add.f16x2 r7330, r6208, r6256;
}
{
mul.f16x2 r7333, r7330, r7296;
}
{
add.f16x2 r7336, r4591, r7333;
}
{
add.f16x2 r7339, r6224, r6240;
}
{
mul.f16x2 r7342, r7339, r7300;
}
{
add.f16x2 r7345, r7336, r7342;
}
{
sub.f16x2 r7348, r6214, r6262;
}
{
mul.f16x2 r7351, r7348, r7298;
}
{
sub.f16x2 r7354, r6230, r6246;
}
{
mul.f16x2 r7357, r7354, r7302;
}
{
add.f16x2 r7360, r7351, r7357;
}
{
sub.f16x2 r7363, r7345, r7360;
}
{
add.f16x2 r7366, r6208, r6256;
}
{
mul.f16x2 r7369, r7366, r7296;
}
{
add.f16x2 r7372, r4591, r7369;
}
{
add.f16x2 r7375, r6224, r6240;
}
{
mul.f16x2 r7378, r7375, r7300;
}
{
add.f16x2 r7381, r7372, r7378;
}
{
sub.f16x2 r7384, r6214, r6262;
}
{
mul.f16x2 r7387, r7384, r7298;
}
{
sub.f16x2 r7390, r6230, r6246;
}
{
mul.f16x2 r7393, r7390, r7302;
}
{
add.f16x2 r7396, r7387, r7393;
}
{
add.f16x2 r7399, r7381, r7396;
}
{
add.f16x2 r7402, r6208, r6256;
}
{
mul.f16x2 r7405, r7402, r7300;
}
{
add.f16x2 r7408, r4591, r7405;
}
{
add.f16x2 r7411, r6224, r6240;
}
{
mul.f16x2 r7414, r7411, r7304;
}
{
add.f16x2 r7417, r7408, r7414;
}
{
sub.f16x2 r7420, r6214, r6262;
}
{
mul.f16x2 r7423, r7420, r7302;
}
{
sub.f16x2 r7426, r6230, r6246;
}
{
mul.f16x2 r7429, r7426, r7305;
}
{
add.f16x2 r7432, r7423, r7429;
}
{
sub.f16x2 r7435, r7417, r7432;
}
{
add.f16x2 r7438, r6208, r6256;
}
{
mul.f16x2 r7441, r7438, r7300;
}
{
add.f16x2 r7444, r4591, r7441;
}
{
add.f16x2 r7447, r6224, r6240;
}
{
mul.f16x2 r7450, r7447, r7304;
}
{
add.f16x2 r7453, r7444, r7450;
}
{
sub.f16x2 r7456, r6214, r6262;
}
{
mul.f16x2 r7459, r7456, r7302;
}
{
sub.f16x2 r7462, r6230, r6246;
}
{
mul.f16x2 r7465, r7462, r7305;
}
{
add.f16x2 r7468, r7459, r7465;
}
{
add.f16x2 r7471, r7453, r7468;
}
{
add.f16x2 r7474, r6214, r6262;
}
{
mul.f16x2 r7477, r7474, r7296;
}
{
add.f16x2 r7480, r4735, r7477;
}
{
add.f16x2 r7483, r6230, r6246;
}
{
mul.f16x2 r7486, r7483, r7300;
}
{
add.f16x2 r7489, r7480, r7486;
}
{
sub.f16x2 r7492, r6208, r6256;
}
{
mul.f16x2 r7495, r7492, r7298;
}
{
sub.f16x2 r7498, r6224, r6240;
}
{
mul.f16x2 r7501, r7498, r7302;
}
{
add.f16x2 r7504, r7495, r7501;
}
{
add.f16x2 r7507, r7489, r7504;
}
{
add.f16x2 r7510, r6214, r6262;
}
{
mul.f16x2 r7513, r7510, r7296;
}
{
add.f16x2 r7516, r4735, r7513;
}
{
add.f16x2 r7519, r6230, r6246;
}
{
mul.f16x2 r7522, r7519, r7300;
}
{
add.f16x2 r7525, r7516, r7522;
}
{
sub.f16x2 r7528, r6208, r6256;
}
{
mul.f16x2 r7531, r7528, r7298;
}
{
sub.f16x2 r7534, r6224, r6240;
}
{
mul.f16x2 r7537, r7534, r7302;
}
{
add.f16x2 r7540, r7531, r7537;
}
{
sub.f16x2 r7543, r7525, r7540;
}
{
add.f16x2 r7546, r6214, r6262;
}
{
mul.f16x2 r7549, r7546, r7300;
}
{
add.f16x2 r7552, r4735, r7549;
}
{
add.f16x2 r7555, r6230, r6246;
}
{
mul.f16x2 r7558, r7555, r7304;
}
{
add.f16x2 r7561, r7552, r7558;
}
{
sub.f16x2 r7564, r6208, r6256;
}
{
mul.f16x2 r7567, r7564, r7302;
}
{
sub.f16x2 r7570, r6224, r6240;
}
{
mul.f16x2 r7573, r7570, r7305;
}
{
add.f16x2 r7576, r7567, r7573;
}
{
add.f16x2 r7579, r7561, r7576;
}
{
add.f16x2 r7582, r6214, r6262;
}
{
mul.f16x2 r7585, r7582, r7300;
}
{
add.f16x2 r7588, r4735, r7585;
}
{
add.f16x2 r7591, r6230, r6246;
}
{
mul.f16x2 r7594, r7591, r7304;
}
{
add.f16x2 r7597, r7588, r7594;
}
{
sub.f16x2 r7600, r6208, r6256;
}
{
mul.f16x2 r7603, r7600, r7302;
}
{
sub.f16x2 r7606, r6224, r6240;
}
{
mul.f16x2 r7609, r7606, r7305;
}
{
add.f16x2 r7612, r7603, r7609;
}
{
sub.f16x2 r7615, r7597, r7612;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r7618, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r7619, {low, high};
}
{
neg.f16x2 r7620, r7619;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r7622, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r7623, {low, high};
}
{
neg.f16x2 r7624, r7623;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r7626, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r7627, {low, high};
}
{
add.f16x2 r7628, r6272, r6320;
}
{
add.f16x2 r7631, r4519, r7628;
}
{
add.f16x2 r7634, r6288, r6304;
}
{
add.f16x2 r7637, r7631, r7634;
}
{
add.f16x2 r7640, r6278, r6326;
}
{
add.f16x2 r7643, r4663, r7640;
}
{
add.f16x2 r7646, r6294, r6310;
}
{
add.f16x2 r7649, r7643, r7646;
}
{
add.f16x2 r7652, r6272, r6320;
}
{
mul.f16x2 r7655, r7652, r7618;
}
{
add.f16x2 r7658, r4519, r7655;
}
{
add.f16x2 r7661, r6288, r6304;
}
{
mul.f16x2 r7664, r7661, r7622;
}
{
add.f16x2 r7667, r7658, r7664;
}
{
sub.f16x2 r7670, r6278, r6326;
}
{
mul.f16x2 r7673, r7670, r7620;
}
{
sub.f16x2 r7676, r6294, r6310;
}
{
mul.f16x2 r7679, r7676, r7624;
}
{
add.f16x2 r7682, r7673, r7679;
}
{
sub.f16x2 r7685, r7667, r7682;
}
{
add.f16x2 r7688, r6272, r6320;
}
{
mul.f16x2 r7691, r7688, r7618;
}
{
add.f16x2 r7694, r4519, r7691;
}
{
add.f16x2 r7697, r6288, r6304;
}
{
mul.f16x2 r7700, r7697, r7622;
}
{
add.f16x2 r7703, r7694, r7700;
}
{
sub.f16x2 r7706, r6278, r6326;
}
{
mul.f16x2 r7709, r7706, r7620;
}
{
sub.f16x2 r7712, r6294, r6310;
}
{
mul.f16x2 r7715, r7712, r7624;
}
{
add.f16x2 r7718, r7709, r7715;
}
{
add.f16x2 r7721, r7703, r7718;
}
{
add.f16x2 r7724, r6272, r6320;
}
{
mul.f16x2 r7727, r7724, r7622;
}
{
add.f16x2 r7730, r4519, r7727;
}
{
add.f16x2 r7733, r6288, r6304;
}
{
mul.f16x2 r7736, r7733, r7626;
}
{
add.f16x2 r7739, r7730, r7736;
}
{
sub.f16x2 r7742, r6278, r6326;
}
{
mul.f16x2 r7745, r7742, r7624;
}
{
sub.f16x2 r7748, r6294, r6310;
}
{
mul.f16x2 r7751, r7748, r7627;
}
{
add.f16x2 r7754, r7745, r7751;
}
{
sub.f16x2 r7757, r7739, r7754;
}
{
add.f16x2 r7760, r6272, r6320;
}
{
mul.f16x2 r7763, r7760, r7622;
}
{
add.f16x2 r7766, r4519, r7763;
}
{
add.f16x2 r7769, r6288, r6304;
}
{
mul.f16x2 r7772, r7769, r7626;
}
{
add.f16x2 r7775, r7766, r7772;
}
{
sub.f16x2 r7778, r6278, r6326;
}
{
mul.f16x2 r7781, r7778, r7624;
}
{
sub.f16x2 r7784, r6294, r6310;
}
{
mul.f16x2 r7787, r7784, r7627;
}
{
add.f16x2 r7790, r7781, r7787;
}
{
add.f16x2 r7793, r7775, r7790;
}
{
add.f16x2 r7796, r6278, r6326;
}
{
mul.f16x2 r7799, r7796, r7618;
}
{
add.f16x2 r7802, r4663, r7799;
}
{
add.f16x2 r7805, r6294, r6310;
}
{
mul.f16x2 r7808, r7805, r7622;
}
{
add.f16x2 r7811, r7802, r7808;
}
{
sub.f16x2 r7814, r6272, r6320;
}
{
mul.f16x2 r7817, r7814, r7620;
}
{
sub.f16x2 r7820, r6288, r6304;
}
{
mul.f16x2 r7823, r7820, r7624;
}
{
add.f16x2 r7826, r7817, r7823;
}
{
add.f16x2 r7829, r7811, r7826;
}
{
add.f16x2 r7832, r6278, r6326;
}
{
mul.f16x2 r7835, r7832, r7618;
}
{
add.f16x2 r7838, r4663, r7835;
}
{
add.f16x2 r7841, r6294, r6310;
}
{
mul.f16x2 r7844, r7841, r7622;
}
{
add.f16x2 r7847, r7838, r7844;
}
{
sub.f16x2 r7850, r6272, r6320;
}
{
mul.f16x2 r7853, r7850, r7620;
}
{
sub.f16x2 r7856, r6288, r6304;
}
{
mul.f16x2 r7859, r7856, r7624;
}
{
add.f16x2 r7862, r7853, r7859;
}
{
sub.f16x2 r7865, r7847, r7862;
}
{
add.f16x2 r7868, r6278, r6326;
}
{
mul.f16x2 r7871, r7868, r7622;
}
{
add.f16x2 r7874, r4663, r7871;
}
{
add.f16x2 r7877, r6294, r6310;
}
{
mul.f16x2 r7880, r7877, r7626;
}
{
add.f16x2 r7883, r7874, r7880;
}
{
sub.f16x2 r7886, r6272, r6320;
}
{
mul.f16x2 r7889, r7886, r7624;
}
{
sub.f16x2 r7892, r6288, r6304;
}
{
mul.f16x2 r7895, r7892, r7627;
}
{
add.f16x2 r7898, r7889, r7895;
}
{
add.f16x2 r7901, r7883, r7898;
}
{
add.f16x2 r7904, r6278, r6326;
}
{
mul.f16x2 r7907, r7904, r7622;
}
{
add.f16x2 r7910, r4663, r7907;
}
{
add.f16x2 r7913, r6294, r6310;
}
{
mul.f16x2 r7916, r7913, r7626;
}
{
add.f16x2 r7919, r7910, r7916;
}
{
sub.f16x2 r7922, r6272, r6320;
}
{
mul.f16x2 r7925, r7922, r7624;
}
{
sub.f16x2 r7928, r6288, r6304;
}
{
mul.f16x2 r7931, r7928, r7627;
}
{
add.f16x2 r7934, r7925, r7931;
}
{
sub.f16x2 r7937, r7919, r7934;
}
mul.wide.u32 rd4, r10447, 1374389535;
shr.u64 rd5, rd4, 35;
cvt.u32.u64 r10451, rd5;
mul.lo.s32 r10452, r10451, 25;
sub.s32 r10453, r10447, r10452;
shl.b32 r10454, r10453, 2;
add.s32 r10455, r10448, r10454;
cvt.rn.f32.u32 f600, r10451;
mul.f32 f601, f600, 0f3D4DE32E;
cos.approx.f32 f485, f601;
sin.approx.f32 f602, f601;
neg.f32 f486, f602;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f485;
cvt.rn.f16.f32 high, f486;
mov.b32 r7940, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r7943, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r7945, {high, high};
}
{
mul.f16x2 r7947, r6683, r7945;
}
{
fma.rn.f16x2 r7950, r6671, r7943, r7947;
}
{
mul.f16x2 r7954, r6671, r7945;
}
{
neg.f16x2 r7957, r7954;
}
{
fma.rn.f16x2 r7959, r6683, r7943, r7957;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r7963, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r7965, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r7967, {low, high};
}
{
mul.f16x2 r7968, r7965, r7967;
}
{
mul.f16x2 r7971, r7940, r7963;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r7974, {high, low};
}
{
fma.rn.f16x2 r7976, r7968, r7974, r7971;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7976;
mov.b32 r7980, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7976;
mov.b32 r7982, {high, high};
}
{
mul.f16x2 r7984, r7005, r7982;
}
{
fma.rn.f16x2 r7987, r6993, r7980, r7984;
}
{
mul.f16x2 r7991, r6993, r7982;
}
{
neg.f16x2 r7994, r7991;
}
{
fma.rn.f16x2 r7996, r7005, r7980, r7994;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8000, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8002, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8004, {low, high};
}
{
mul.f16x2 r8005, r8002, r8004;
}
{
mul.f16x2 r8008, r7976, r8000;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7976;
mov.b32 r8011, {high, low};
}
{
fma.rn.f16x2 r8013, r8005, r8011, r8008;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8013;
mov.b32 r8017, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8013;
mov.b32 r8019, {high, high};
}
{
mul.f16x2 r8021, r7327, r8019;
}
{
fma.rn.f16x2 r8024, r7315, r8017, r8021;
}
{
mul.f16x2 r8028, r7315, r8019;
}
{
neg.f16x2 r8031, r8028;
}
{
fma.rn.f16x2 r8033, r7327, r8017, r8031;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8037, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8039, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8041, {low, high};
}
{
mul.f16x2 r8042, r8039, r8041;
}
{
mul.f16x2 r8045, r8013, r8037;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8013;
mov.b32 r8048, {high, low};
}
{
fma.rn.f16x2 r8050, r8042, r8048, r8045;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8050;
mov.b32 r8054, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8050;
mov.b32 r8056, {high, high};
}
{
mul.f16x2 r8058, r7649, r8056;
}
{
fma.rn.f16x2 r8061, r7637, r8054, r8058;
}
{
mul.f16x2 r8065, r7637, r8056;
}
{
neg.f16x2 r8068, r8065;
}
{
fma.rn.f16x2 r8070, r7649, r8054, r8068;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8074, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8076, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8078, {low, high};
}
{
mul.f16x2 r8079, r8076, r8078;
}
{
mul.f16x2 r8082, r8050, r8074;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8050;
mov.b32 r8085, {high, low};
}
{
fma.rn.f16x2 r8087, r8079, r8085, r8082;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8087;
mov.b32 r8091, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8087;
mov.b32 r8093, {high, high};
}
{
mul.f16x2 r8095, r6541, r8093;
}
{
fma.rn.f16x2 r8098, r6397, r8091, r8095;
}
{
mul.f16x2 r8102, r6397, r8093;
}
{
neg.f16x2 r8105, r8102;
}
{
fma.rn.f16x2 r8107, r6541, r8091, r8105;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8111, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8113, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8115, {low, high};
}
{
mul.f16x2 r8116, r8113, r8115;
}
{
mul.f16x2 r8119, r8087, r8111;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8087;
mov.b32 r8122, {high, low};
}
{
fma.rn.f16x2 r8124, r8116, r8122, r8119;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8124;
mov.b32 r8128, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8124;
mov.b32 r8130, {high, high};
}
{
mul.f16x2 r8132, r6863, r8130;
}
{
fma.rn.f16x2 r8135, r6719, r8128, r8132;
}
{
mul.f16x2 r8139, r6719, r8130;
}
{
neg.f16x2 r8142, r8139;
}
{
fma.rn.f16x2 r8144, r6863, r8128, r8142;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8148, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8150, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8152, {low, high};
}
{
mul.f16x2 r8153, r8150, r8152;
}
{
mul.f16x2 r8156, r8124, r8148;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8124;
mov.b32 r8159, {high, low};
}
{
fma.rn.f16x2 r8161, r8153, r8159, r8156;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8161;
mov.b32 r8165, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8161;
mov.b32 r8167, {high, high};
}
{
mul.f16x2 r8169, r7185, r8167;
}
{
fma.rn.f16x2 r8172, r7041, r8165, r8169;
}
{
mul.f16x2 r8176, r7041, r8167;
}
{
neg.f16x2 r8179, r8176;
}
{
fma.rn.f16x2 r8181, r7185, r8165, r8179;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8185, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8187, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8189, {low, high};
}
{
mul.f16x2 r8190, r8187, r8189;
}
{
mul.f16x2 r8193, r8161, r8185;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8161;
mov.b32 r8196, {high, low};
}
{
fma.rn.f16x2 r8198, r8190, r8196, r8193;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8198;
mov.b32 r8202, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8198;
mov.b32 r8204, {high, high};
}
{
mul.f16x2 r8206, r7507, r8204;
}
{
fma.rn.f16x2 r8209, r7363, r8202, r8206;
}
{
mul.f16x2 r8213, r7363, r8204;
}
{
neg.f16x2 r8216, r8213;
}
{
fma.rn.f16x2 r8218, r7507, r8202, r8216;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8222, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8224, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8226, {low, high};
}
{
mul.f16x2 r8227, r8224, r8226;
}
{
mul.f16x2 r8230, r8198, r8222;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8198;
mov.b32 r8233, {high, low};
}
{
fma.rn.f16x2 r8235, r8227, r8233, r8230;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8235;
mov.b32 r8239, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8235;
mov.b32 r8241, {high, high};
}
{
mul.f16x2 r8243, r7829, r8241;
}
{
fma.rn.f16x2 r8246, r7685, r8239, r8243;
}
{
mul.f16x2 r8250, r7685, r8241;
}
{
neg.f16x2 r8253, r8250;
}
{
fma.rn.f16x2 r8255, r7829, r8239, r8253;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8259, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8261, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8263, {low, high};
}
{
mul.f16x2 r8264, r8261, r8263;
}
{
mul.f16x2 r8267, r8235, r8259;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8235;
mov.b32 r8270, {high, low};
}
{
fma.rn.f16x2 r8272, r8264, r8270, r8267;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8272;
mov.b32 r8276, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8272;
mov.b32 r8278, {high, high};
}
{
mul.f16x2 r8280, r6613, r8278;
}
{
fma.rn.f16x2 r8283, r6469, r8276, r8280;
}
{
mul.f16x2 r8287, r6469, r8278;
}
{
neg.f16x2 r8290, r8287;
}
{
fma.rn.f16x2 r8292, r6613, r8276, r8290;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8296, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8298, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8300, {low, high};
}
{
mul.f16x2 r8301, r8298, r8300;
}
{
mul.f16x2 r8304, r8272, r8296;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8272;
mov.b32 r8307, {high, low};
}
{
fma.rn.f16x2 r8309, r8301, r8307, r8304;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8309;
mov.b32 r8313, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8309;
mov.b32 r8315, {high, high};
}
{
mul.f16x2 r8317, r6935, r8315;
}
{
fma.rn.f16x2 r8320, r6791, r8313, r8317;
}
{
mul.f16x2 r8324, r6791, r8315;
}
{
neg.f16x2 r8327, r8324;
}
{
fma.rn.f16x2 r8329, r6935, r8313, r8327;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8333, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8335, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8337, {low, high};
}
{
mul.f16x2 r8338, r8335, r8337;
}
{
mul.f16x2 r8341, r8309, r8333;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8309;
mov.b32 r8344, {high, low};
}
{
fma.rn.f16x2 r8346, r8338, r8344, r8341;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8346;
mov.b32 r8350, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8346;
mov.b32 r8352, {high, high};
}
{
mul.f16x2 r8354, r7257, r8352;
}
{
fma.rn.f16x2 r8357, r7113, r8350, r8354;
}
{
mul.f16x2 r8361, r7113, r8352;
}
{
neg.f16x2 r8364, r8361;
}
{
fma.rn.f16x2 r8366, r7257, r8350, r8364;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8370, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8372, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8374, {low, high};
}
{
mul.f16x2 r8375, r8372, r8374;
}
{
mul.f16x2 r8378, r8346, r8370;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8346;
mov.b32 r8381, {high, low};
}
{
fma.rn.f16x2 r8383, r8375, r8381, r8378;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8383;
mov.b32 r8387, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8383;
mov.b32 r8389, {high, high};
}
{
mul.f16x2 r8391, r7579, r8389;
}
{
fma.rn.f16x2 r8394, r7435, r8387, r8391;
}
{
mul.f16x2 r8398, r7435, r8389;
}
{
neg.f16x2 r8401, r8398;
}
{
fma.rn.f16x2 r8403, r7579, r8387, r8401;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8407, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8409, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8411, {low, high};
}
{
mul.f16x2 r8412, r8409, r8411;
}
{
mul.f16x2 r8415, r8383, r8407;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8383;
mov.b32 r8418, {high, low};
}
{
fma.rn.f16x2 r8420, r8412, r8418, r8415;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8420;
mov.b32 r8424, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8420;
mov.b32 r8426, {high, high};
}
{
mul.f16x2 r8428, r7901, r8426;
}
{
fma.rn.f16x2 r8431, r7757, r8424, r8428;
}
{
mul.f16x2 r8435, r7757, r8426;
}
{
neg.f16x2 r8438, r8435;
}
{
fma.rn.f16x2 r8440, r7901, r8424, r8438;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8444, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8446, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8448, {low, high};
}
{
mul.f16x2 r8449, r8446, r8448;
}
{
mul.f16x2 r8452, r8420, r8444;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8420;
mov.b32 r8455, {high, low};
}
{
fma.rn.f16x2 r8457, r8449, r8455, r8452;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8457;
mov.b32 r8461, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8457;
mov.b32 r8463, {high, high};
}
{
mul.f16x2 r8465, r6649, r8463;
}
{
fma.rn.f16x2 r8468, r6505, r8461, r8465;
}
{
mul.f16x2 r8472, r6505, r8463;
}
{
neg.f16x2 r8475, r8472;
}
{
fma.rn.f16x2 r8477, r6649, r8461, r8475;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8481, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8483, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8485, {low, high};
}
{
mul.f16x2 r8486, r8483, r8485;
}
{
mul.f16x2 r8489, r8457, r8481;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8457;
mov.b32 r8492, {high, low};
}
{
fma.rn.f16x2 r8494, r8486, r8492, r8489;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8494;
mov.b32 r8498, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8494;
mov.b32 r8500, {high, high};
}
{
mul.f16x2 r8502, r6971, r8500;
}
{
fma.rn.f16x2 r8505, r6827, r8498, r8502;
}
{
mul.f16x2 r8509, r6827, r8500;
}
{
neg.f16x2 r8512, r8509;
}
{
fma.rn.f16x2 r8514, r6971, r8498, r8512;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8518, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8520, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8522, {low, high};
}
{
mul.f16x2 r8523, r8520, r8522;
}
{
mul.f16x2 r8526, r8494, r8518;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8494;
mov.b32 r8529, {high, low};
}
{
fma.rn.f16x2 r8531, r8523, r8529, r8526;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8531;
mov.b32 r8535, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8531;
mov.b32 r8537, {high, high};
}
{
mul.f16x2 r8539, r7293, r8537;
}
{
fma.rn.f16x2 r8542, r7149, r8535, r8539;
}
{
mul.f16x2 r8546, r7149, r8537;
}
{
neg.f16x2 r8549, r8546;
}
{
fma.rn.f16x2 r8551, r7293, r8535, r8549;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8555, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8557, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8559, {low, high};
}
{
mul.f16x2 r8560, r8557, r8559;
}
{
mul.f16x2 r8563, r8531, r8555;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8531;
mov.b32 r8566, {high, low};
}
{
fma.rn.f16x2 r8568, r8560, r8566, r8563;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8568;
mov.b32 r8572, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8568;
mov.b32 r8574, {high, high};
}
{
mul.f16x2 r8576, r7615, r8574;
}
{
fma.rn.f16x2 r8579, r7471, r8572, r8576;
}
{
mul.f16x2 r8583, r7471, r8574;
}
{
neg.f16x2 r8586, r8583;
}
{
fma.rn.f16x2 r8588, r7615, r8572, r8586;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8592, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8594, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8596, {low, high};
}
{
mul.f16x2 r8597, r8594, r8596;
}
{
mul.f16x2 r8600, r8568, r8592;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8568;
mov.b32 r8603, {high, low};
}
{
fma.rn.f16x2 r8605, r8597, r8603, r8600;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8605;
mov.b32 r8609, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8605;
mov.b32 r8611, {high, high};
}
{
mul.f16x2 r8613, r7937, r8611;
}
{
fma.rn.f16x2 r8616, r7793, r8609, r8613;
}
{
mul.f16x2 r8620, r7793, r8611;
}
{
neg.f16x2 r8623, r8620;
}
{
fma.rn.f16x2 r8625, r7937, r8609, r8623;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8629, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8631, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8633, {low, high};
}
{
mul.f16x2 r8634, r8631, r8633;
}
{
mul.f16x2 r8637, r8605, r8629;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8605;
mov.b32 r8640, {high, low};
}
{
fma.rn.f16x2 r8642, r8634, r8640, r8637;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8642;
mov.b32 r8646, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8642;
mov.b32 r8648, {high, high};
}
{
mul.f16x2 r8650, r6577, r8648;
}
{
fma.rn.f16x2 r8653, r6433, r8646, r8650;
}
{
mul.f16x2 r8657, r6433, r8648;
}
{
neg.f16x2 r8660, r8657;
}
{
fma.rn.f16x2 r8662, r6577, r8646, r8660;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8666, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8668, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8670, {low, high};
}
{
mul.f16x2 r8671, r8668, r8670;
}
{
mul.f16x2 r8674, r8642, r8666;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8642;
mov.b32 r8677, {high, low};
}
{
fma.rn.f16x2 r8679, r8671, r8677, r8674;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8679;
mov.b32 r8683, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8679;
mov.b32 r8685, {high, high};
}
{
mul.f16x2 r8687, r6899, r8685;
}
{
fma.rn.f16x2 r8690, r6755, r8683, r8687;
}
{
mul.f16x2 r8694, r6755, r8685;
}
{
neg.f16x2 r8697, r8694;
}
{
fma.rn.f16x2 r8699, r6899, r8683, r8697;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8703, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8705, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8707, {low, high};
}
{
mul.f16x2 r8708, r8705, r8707;
}
{
mul.f16x2 r8711, r8679, r8703;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8679;
mov.b32 r8714, {high, low};
}
{
fma.rn.f16x2 r8716, r8708, r8714, r8711;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8716;
mov.b32 r8720, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8716;
mov.b32 r8722, {high, high};
}
{
mul.f16x2 r8724, r7221, r8722;
}
{
fma.rn.f16x2 r8727, r7077, r8720, r8724;
}
{
mul.f16x2 r8731, r7077, r8722;
}
{
neg.f16x2 r8734, r8731;
}
{
fma.rn.f16x2 r8736, r7221, r8720, r8734;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8740, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8742, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8744, {low, high};
}
{
mul.f16x2 r8745, r8742, r8744;
}
{
mul.f16x2 r8748, r8716, r8740;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8716;
mov.b32 r8751, {high, low};
}
{
fma.rn.f16x2 r8753, r8745, r8751, r8748;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8753;
mov.b32 r8757, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8753;
mov.b32 r8759, {high, high};
}
{
mul.f16x2 r8761, r7543, r8759;
}
{
fma.rn.f16x2 r8764, r7399, r8757, r8761;
}
{
mul.f16x2 r8768, r7399, r8759;
}
{
neg.f16x2 r8771, r8768;
}
{
fma.rn.f16x2 r8773, r7543, r8757, r8771;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8777, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7940;
mov.b32 r8779, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8781, {low, high};
}
{
mul.f16x2 r8782, r8779, r8781;
}
{
mul.f16x2 r8785, r8753, r8777;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8753;
mov.b32 r8788, {high, low};
}
{
fma.rn.f16x2 r8790, r8782, r8788, r8785;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8790;
mov.b32 r8794, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8790;
mov.b32 r8796, {high, high};
}
{
mul.f16x2 r8798, r7865, r8796;
}
{
fma.rn.f16x2 r8801, r7721, r8794, r8798;
}
{
mul.f16x2 r8805, r7721, r8796;
}
{
neg.f16x2 r8808, r8805;
}
{
fma.rn.f16x2 r8810, r7865, r8794, r8808;
}
barrier.sync 0;
mad.lo.s32 r10456, r10451, 2500, r10455;
st.shared.u32 [r10456], r6349;
st.shared.u32 [r10456+100], r7950;
st.shared.u32 [r10456+200], r7987;
st.shared.u32 [r10456+300], r8024;
st.shared.u32 [r10456+400], r8061;
st.shared.u32 [r10456+500], r8098;
st.shared.u32 [r10456+600], r8135;
st.shared.u32 [r10456+700], r8172;
st.shared.u32 [r10456+800], r8209;
st.shared.u32 [r10456+900], r8246;
st.shared.u32 [r10456+1000], r8283;
st.shared.u32 [r10456+1100], r8320;
st.shared.u32 [r10456+1200], r8357;
st.shared.u32 [r10456+1300], r8394;
st.shared.u32 [r10456+1400], r8431;
st.shared.u32 [r10456+1500], r8468;
st.shared.u32 [r10456+1600], r8505;
st.shared.u32 [r10456+1700], r8542;
st.shared.u32 [r10456+1800], r8579;
st.shared.u32 [r10456+1900], r8616;
st.shared.u32 [r10456+2000], r8653;
st.shared.u32 [r10456+2100], r8690;
st.shared.u32 [r10456+2200], r8727;
st.shared.u32 [r10456+2300], r8764;
st.shared.u32 [r10456+2400], r8801;
barrier.sync 0;
ld.shared.u32 r8845, [r10450];
ld.shared.u32 r9167, [r10450+500];
ld.shared.u32 r9489, [r10450+1000];
ld.shared.u32 r9811, [r10450+1500];
ld.shared.u32 r10133, [r10450+2000];
ld.shared.u32 r8842, [r10450+2500];
ld.shared.u32 r9164, [r10450+3000];
ld.shared.u32 r9486, [r10450+3500];
ld.shared.u32 r9808, [r10450+4000];
ld.shared.u32 r10130, [r10450+4500];
ld.shared.u32 r8848, [r10450+5000];
ld.shared.u32 r9170, [r10450+5500];
ld.shared.u32 r9492, [r10450+6000];
ld.shared.u32 r9814, [r10450+6500];
ld.shared.u32 r10136, [r10450+7000];
ld.shared.u32 r8849, [r10450+7500];
ld.shared.u32 r9171, [r10450+8000];
ld.shared.u32 r9493, [r10450+8500];
ld.shared.u32 r9815, [r10450+9000];
ld.shared.u32 r10137, [r10450+9500];
ld.shared.u32 r8843, [r10450+10000];
ld.shared.u32 r9165, [r10450+10500];
ld.shared.u32 r9487, [r10450+11000];
ld.shared.u32 r9809, [r10450+11500];
ld.shared.u32 r10131, [r10450+12000];
barrier.sync 0;
st.shared.u32 [r10456], r6361;
st.shared.u32 [r10456+100], r7959;
st.shared.u32 [r10456+200], r7996;
st.shared.u32 [r10456+300], r8033;
st.shared.u32 [r10456+400], r8070;
st.shared.u32 [r10456+500], r8107;
st.shared.u32 [r10456+600], r8144;
st.shared.u32 [r10456+700], r8181;
st.shared.u32 [r10456+800], r8218;
st.shared.u32 [r10456+900], r8255;
st.shared.u32 [r10456+1000], r8292;
st.shared.u32 [r10456+1100], r8329;
st.shared.u32 [r10456+1200], r8366;
st.shared.u32 [r10456+1300], r8403;
st.shared.u32 [r10456+1400], r8440;
st.shared.u32 [r10456+1500], r8477;
st.shared.u32 [r10456+1600], r8514;
st.shared.u32 [r10456+1700], r8551;
st.shared.u32 [r10456+1800], r8588;
st.shared.u32 [r10456+1900], r8625;
st.shared.u32 [r10456+2000], r8662;
st.shared.u32 [r10456+2100], r8699;
st.shared.u32 [r10456+2200], r8736;
st.shared.u32 [r10456+2300], r8773;
st.shared.u32 [r10456+2400], r8810;
barrier.sync 0;
ld.shared.u32 r8857, [r10450];
ld.shared.u32 r9179, [r10450+500];
ld.shared.u32 r9501, [r10450+1000];
ld.shared.u32 r9823, [r10450+1500];
ld.shared.u32 r10145, [r10450+2000];
ld.shared.u32 r8854, [r10450+2500];
ld.shared.u32 r9176, [r10450+3000];
ld.shared.u32 r9498, [r10450+3500];
ld.shared.u32 r9820, [r10450+4000];
ld.shared.u32 r10142, [r10450+4500];
ld.shared.u32 r8860, [r10450+5000];
ld.shared.u32 r9182, [r10450+5500];
ld.shared.u32 r9504, [r10450+6000];
ld.shared.u32 r9826, [r10450+6500];
ld.shared.u32 r10148, [r10450+7000];
ld.shared.u32 r8861, [r10450+7500];
ld.shared.u32 r9183, [r10450+8000];
ld.shared.u32 r9505, [r10450+8500];
ld.shared.u32 r9827, [r10450+9000];
ld.shared.u32 r10149, [r10450+9500];
ld.shared.u32 r8855, [r10450+10000];
ld.shared.u32 r9177, [r10450+10500];
ld.shared.u32 r9499, [r10450+11000];
ld.shared.u32 r9821, [r10450+11500];
ld.shared.u32 r10143, [r10450+12000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r8831, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r8832, {low, high};
}
{
neg.f16x2 r8833, r8832;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r8835, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r8836, {low, high};
}
{
neg.f16x2 r8837, r8836;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r8839, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r8840, {low, high};
}
{
add.f16x2 r8841, r8842, r8843;
}
{
add.f16x2 r8844, r8845, r8841;
}
{
add.f16x2 r8847, r8848, r8849;
}
{
add.f16x2 %0, r8844, r8847;
}
{
add.f16x2 r8853, r8854, r8855;
}
{
add.f16x2 r8856, r8857, r8853;
}
{
add.f16x2 r8859, r8860, r8861;
}
{
add.f16x2 %1, r8856, r8859;
}
{
add.f16x2 r8865, r8842, r8843;
}
{
mul.f16x2 r8868, r8865, r8831;
}
{
add.f16x2 r8871, r8845, r8868;
}
{
add.f16x2 r8874, r8848, r8849;
}
{
mul.f16x2 r8877, r8874, r8835;
}
{
add.f16x2 r8880, r8871, r8877;
}
{
sub.f16x2 r8883, r8854, r8855;
}
{
mul.f16x2 r8886, r8883, r8833;
}
{
sub.f16x2 r8889, r8860, r8861;
}
{
mul.f16x2 r8892, r8889, r8837;
}
{
add.f16x2 r8895, r8886, r8892;
}
{
sub.f16x2 %10, r8880, r8895;
}
{
add.f16x2 r8901, r8842, r8843;
}
{
mul.f16x2 r8904, r8901, r8831;
}
{
add.f16x2 r8907, r8845, r8904;
}
{
add.f16x2 r8910, r8848, r8849;
}
{
mul.f16x2 r8913, r8910, r8835;
}
{
add.f16x2 r8916, r8907, r8913;
}
{
sub.f16x2 r8919, r8854, r8855;
}
{
mul.f16x2 r8922, r8919, r8833;
}
{
sub.f16x2 r8925, r8860, r8861;
}
{
mul.f16x2 r8928, r8925, r8837;
}
{
add.f16x2 r8931, r8922, r8928;
}
{
add.f16x2 %40, r8916, r8931;
}
{
add.f16x2 r8937, r8842, r8843;
}
{
mul.f16x2 r8940, r8937, r8835;
}
{
add.f16x2 r8943, r8845, r8940;
}
{
add.f16x2 r8946, r8848, r8849;
}
{
mul.f16x2 r8949, r8946, r8839;
}
{
add.f16x2 r8952, r8943, r8949;
}
{
sub.f16x2 r8955, r8854, r8855;
}
{
mul.f16x2 r8958, r8955, r8837;
}
{
sub.f16x2 r8961, r8860, r8861;
}
{
mul.f16x2 r8964, r8961, r8840;
}
{
add.f16x2 r8967, r8958, r8964;
}
{
sub.f16x2 %20, r8952, r8967;
}
{
add.f16x2 r8973, r8842, r8843;
}
{
mul.f16x2 r8976, r8973, r8835;
}
{
add.f16x2 r8979, r8845, r8976;
}
{
add.f16x2 r8982, r8848, r8849;
}
{
mul.f16x2 r8985, r8982, r8839;
}
{
add.f16x2 r8988, r8979, r8985;
}
{
sub.f16x2 r8991, r8854, r8855;
}
{
mul.f16x2 r8994, r8991, r8837;
}
{
sub.f16x2 r8997, r8860, r8861;
}
{
mul.f16x2 r9000, r8997, r8840;
}
{
add.f16x2 r9003, r8994, r9000;
}
{
add.f16x2 %30, r8988, r9003;
}
{
add.f16x2 r9009, r8854, r8855;
}
{
mul.f16x2 r9012, r9009, r8831;
}
{
add.f16x2 r9015, r8857, r9012;
}
{
add.f16x2 r9018, r8860, r8861;
}
{
mul.f16x2 r9021, r9018, r8835;
}
{
add.f16x2 r9024, r9015, r9021;
}
{
sub.f16x2 r9027, r8842, r8843;
}
{
mul.f16x2 r9030, r9027, r8833;
}
{
sub.f16x2 r9033, r8848, r8849;
}
{
mul.f16x2 r9036, r9033, r8837;
}
{
add.f16x2 r9039, r9030, r9036;
}
{
add.f16x2 %11, r9024, r9039;
}
{
add.f16x2 r9045, r8854, r8855;
}
{
mul.f16x2 r9048, r9045, r8831;
}
{
add.f16x2 r9051, r8857, r9048;
}
{
add.f16x2 r9054, r8860, r8861;
}
{
mul.f16x2 r9057, r9054, r8835;
}
{
add.f16x2 r9060, r9051, r9057;
}
{
sub.f16x2 r9063, r8842, r8843;
}
{
mul.f16x2 r9066, r9063, r8833;
}
{
sub.f16x2 r9069, r8848, r8849;
}
{
mul.f16x2 r9072, r9069, r8837;
}
{
add.f16x2 r9075, r9066, r9072;
}
{
sub.f16x2 %41, r9060, r9075;
}
{
add.f16x2 r9081, r8854, r8855;
}
{
mul.f16x2 r9084, r9081, r8835;
}
{
add.f16x2 r9087, r8857, r9084;
}
{
add.f16x2 r9090, r8860, r8861;
}
{
mul.f16x2 r9093, r9090, r8839;
}
{
add.f16x2 r9096, r9087, r9093;
}
{
sub.f16x2 r9099, r8842, r8843;
}
{
mul.f16x2 r9102, r9099, r8837;
}
{
sub.f16x2 r9105, r8848, r8849;
}
{
mul.f16x2 r9108, r9105, r8840;
}
{
add.f16x2 r9111, r9102, r9108;
}
{
add.f16x2 %21, r9096, r9111;
}
{
add.f16x2 r9117, r8854, r8855;
}
{
mul.f16x2 r9120, r9117, r8835;
}
{
add.f16x2 r9123, r8857, r9120;
}
{
add.f16x2 r9126, r8860, r8861;
}
{
mul.f16x2 r9129, r9126, r8839;
}
{
add.f16x2 r9132, r9123, r9129;
}
{
sub.f16x2 r9135, r8842, r8843;
}
{
mul.f16x2 r9138, r9135, r8837;
}
{
sub.f16x2 r9141, r8848, r8849;
}
{
mul.f16x2 r9144, r9141, r8840;
}
{
add.f16x2 r9147, r9138, r9144;
}
{
sub.f16x2 %31, r9132, r9147;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9153, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9154, {low, high};
}
{
neg.f16x2 r9155, r9154;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r9157, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r9158, {low, high};
}
{
neg.f16x2 r9159, r9158;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9161, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9162, {low, high};
}
{
add.f16x2 r9163, r9164, r9165;
}
{
add.f16x2 r9166, r9167, r9163;
}
{
add.f16x2 r9169, r9170, r9171;
}
{
add.f16x2 %2, r9166, r9169;
}
{
add.f16x2 r9175, r9176, r9177;
}
{
add.f16x2 r9178, r9179, r9175;
}
{
add.f16x2 r9181, r9182, r9183;
}
{
add.f16x2 %3, r9178, r9181;
}
{
add.f16x2 r9187, r9164, r9165;
}
{
mul.f16x2 r9190, r9187, r9153;
}
{
add.f16x2 r9193, r9167, r9190;
}
{
add.f16x2 r9196, r9170, r9171;
}
{
mul.f16x2 r9199, r9196, r9157;
}
{
add.f16x2 r9202, r9193, r9199;
}
{
sub.f16x2 r9205, r9176, r9177;
}
{
mul.f16x2 r9208, r9205, r9155;
}
{
sub.f16x2 r9211, r9182, r9183;
}
{
mul.f16x2 r9214, r9211, r9159;
}
{
add.f16x2 r9217, r9208, r9214;
}
{
sub.f16x2 %12, r9202, r9217;
}
{
add.f16x2 r9223, r9164, r9165;
}
{
mul.f16x2 r9226, r9223, r9153;
}
{
add.f16x2 r9229, r9167, r9226;
}
{
add.f16x2 r9232, r9170, r9171;
}
{
mul.f16x2 r9235, r9232, r9157;
}
{
add.f16x2 r9238, r9229, r9235;
}
{
sub.f16x2 r9241, r9176, r9177;
}
{
mul.f16x2 r9244, r9241, r9155;
}
{
sub.f16x2 r9247, r9182, r9183;
}
{
mul.f16x2 r9250, r9247, r9159;
}
{
add.f16x2 r9253, r9244, r9250;
}
{
add.f16x2 %42, r9238, r9253;
}
{
add.f16x2 r9259, r9164, r9165;
}
{
mul.f16x2 r9262, r9259, r9157;
}
{
add.f16x2 r9265, r9167, r9262;
}
{
add.f16x2 r9268, r9170, r9171;
}
{
mul.f16x2 r9271, r9268, r9161;
}
{
add.f16x2 r9274, r9265, r9271;
}
{
sub.f16x2 r9277, r9176, r9177;
}
{
mul.f16x2 r9280, r9277, r9159;
}
{
sub.f16x2 r9283, r9182, r9183;
}
{
mul.f16x2 r9286, r9283, r9162;
}
{
add.f16x2 r9289, r9280, r9286;
}
{
sub.f16x2 %22, r9274, r9289;
}
{
add.f16x2 r9295, r9164, r9165;
}
{
mul.f16x2 r9298, r9295, r9157;
}
{
add.f16x2 r9301, r9167, r9298;
}
{
add.f16x2 r9304, r9170, r9171;
}
{
mul.f16x2 r9307, r9304, r9161;
}
{
add.f16x2 r9310, r9301, r9307;
}
{
sub.f16x2 r9313, r9176, r9177;
}
{
mul.f16x2 r9316, r9313, r9159;
}
{
sub.f16x2 r9319, r9182, r9183;
}
{
mul.f16x2 r9322, r9319, r9162;
}
{
add.f16x2 r9325, r9316, r9322;
}
{
add.f16x2 %32, r9310, r9325;
}
{
add.f16x2 r9331, r9176, r9177;
}
{
mul.f16x2 r9334, r9331, r9153;
}
{
add.f16x2 r9337, r9179, r9334;
}
{
add.f16x2 r9340, r9182, r9183;
}
{
mul.f16x2 r9343, r9340, r9157;
}
{
add.f16x2 r9346, r9337, r9343;
}
{
sub.f16x2 r9349, r9164, r9165;
}
{
mul.f16x2 r9352, r9349, r9155;
}
{
sub.f16x2 r9355, r9170, r9171;
}
{
mul.f16x2 r9358, r9355, r9159;
}
{
add.f16x2 r9361, r9352, r9358;
}
{
add.f16x2 %13, r9346, r9361;
}
{
add.f16x2 r9367, r9176, r9177;
}
{
mul.f16x2 r9370, r9367, r9153;
}
{
add.f16x2 r9373, r9179, r9370;
}
{
add.f16x2 r9376, r9182, r9183;
}
{
mul.f16x2 r9379, r9376, r9157;
}
{
add.f16x2 r9382, r9373, r9379;
}
{
sub.f16x2 r9385, r9164, r9165;
}
{
mul.f16x2 r9388, r9385, r9155;
}
{
sub.f16x2 r9391, r9170, r9171;
}
{
mul.f16x2 r9394, r9391, r9159;
}
{
add.f16x2 r9397, r9388, r9394;
}
{
sub.f16x2 %43, r9382, r9397;
}
{
add.f16x2 r9403, r9176, r9177;
}
{
mul.f16x2 r9406, r9403, r9157;
}
{
add.f16x2 r9409, r9179, r9406;
}
{
add.f16x2 r9412, r9182, r9183;
}
{
mul.f16x2 r9415, r9412, r9161;
}
{
add.f16x2 r9418, r9409, r9415;
}
{
sub.f16x2 r9421, r9164, r9165;
}
{
mul.f16x2 r9424, r9421, r9159;
}
{
sub.f16x2 r9427, r9170, r9171;
}
{
mul.f16x2 r9430, r9427, r9162;
}
{
add.f16x2 r9433, r9424, r9430;
}
{
add.f16x2 %23, r9418, r9433;
}
{
add.f16x2 r9439, r9176, r9177;
}
{
mul.f16x2 r9442, r9439, r9157;
}
{
add.f16x2 r9445, r9179, r9442;
}
{
add.f16x2 r9448, r9182, r9183;
}
{
mul.f16x2 r9451, r9448, r9161;
}
{
add.f16x2 r9454, r9445, r9451;
}
{
sub.f16x2 r9457, r9164, r9165;
}
{
mul.f16x2 r9460, r9457, r9159;
}
{
sub.f16x2 r9463, r9170, r9171;
}
{
mul.f16x2 r9466, r9463, r9162;
}
{
add.f16x2 r9469, r9460, r9466;
}
{
sub.f16x2 %33, r9454, r9469;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9475, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9476, {low, high};
}
{
neg.f16x2 r9477, r9476;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r9479, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r9480, {low, high};
}
{
neg.f16x2 r9481, r9480;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9483, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9484, {low, high};
}
{
add.f16x2 r9485, r9486, r9487;
}
{
add.f16x2 r9488, r9489, r9485;
}
{
add.f16x2 r9491, r9492, r9493;
}
{
add.f16x2 %4, r9488, r9491;
}
{
add.f16x2 r9497, r9498, r9499;
}
{
add.f16x2 r9500, r9501, r9497;
}
{
add.f16x2 r9503, r9504, r9505;
}
{
add.f16x2 %5, r9500, r9503;
}
{
add.f16x2 r9509, r9486, r9487;
}
{
mul.f16x2 r9512, r9509, r9475;
}
{
add.f16x2 r9515, r9489, r9512;
}
{
add.f16x2 r9518, r9492, r9493;
}
{
mul.f16x2 r9521, r9518, r9479;
}
{
add.f16x2 r9524, r9515, r9521;
}
{
sub.f16x2 r9527, r9498, r9499;
}
{
mul.f16x2 r9530, r9527, r9477;
}
{
sub.f16x2 r9533, r9504, r9505;
}
{
mul.f16x2 r9536, r9533, r9481;
}
{
add.f16x2 r9539, r9530, r9536;
}
{
sub.f16x2 %14, r9524, r9539;
}
{
add.f16x2 r9545, r9486, r9487;
}
{
mul.f16x2 r9548, r9545, r9475;
}
{
add.f16x2 r9551, r9489, r9548;
}
{
add.f16x2 r9554, r9492, r9493;
}
{
mul.f16x2 r9557, r9554, r9479;
}
{
add.f16x2 r9560, r9551, r9557;
}
{
sub.f16x2 r9563, r9498, r9499;
}
{
mul.f16x2 r9566, r9563, r9477;
}
{
sub.f16x2 r9569, r9504, r9505;
}
{
mul.f16x2 r9572, r9569, r9481;
}
{
add.f16x2 r9575, r9566, r9572;
}
{
add.f16x2 %44, r9560, r9575;
}
{
add.f16x2 r9581, r9486, r9487;
}
{
mul.f16x2 r9584, r9581, r9479;
}
{
add.f16x2 r9587, r9489, r9584;
}
{
add.f16x2 r9590, r9492, r9493;
}
{
mul.f16x2 r9593, r9590, r9483;
}
{
add.f16x2 r9596, r9587, r9593;
}
{
sub.f16x2 r9599, r9498, r9499;
}
{
mul.f16x2 r9602, r9599, r9481;
}
{
sub.f16x2 r9605, r9504, r9505;
}
{
mul.f16x2 r9608, r9605, r9484;
}
{
add.f16x2 r9611, r9602, r9608;
}
{
sub.f16x2 %24, r9596, r9611;
}
{
add.f16x2 r9617, r9486, r9487;
}
{
mul.f16x2 r9620, r9617, r9479;
}
{
add.f16x2 r9623, r9489, r9620;
}
{
add.f16x2 r9626, r9492, r9493;
}
{
mul.f16x2 r9629, r9626, r9483;
}
{
add.f16x2 r9632, r9623, r9629;
}
{
sub.f16x2 r9635, r9498, r9499;
}
{
mul.f16x2 r9638, r9635, r9481;
}
{
sub.f16x2 r9641, r9504, r9505;
}
{
mul.f16x2 r9644, r9641, r9484;
}
{
add.f16x2 r9647, r9638, r9644;
}
{
add.f16x2 %34, r9632, r9647;
}
{
add.f16x2 r9653, r9498, r9499;
}
{
mul.f16x2 r9656, r9653, r9475;
}
{
add.f16x2 r9659, r9501, r9656;
}
{
add.f16x2 r9662, r9504, r9505;
}
{
mul.f16x2 r9665, r9662, r9479;
}
{
add.f16x2 r9668, r9659, r9665;
}
{
sub.f16x2 r9671, r9486, r9487;
}
{
mul.f16x2 r9674, r9671, r9477;
}
{
sub.f16x2 r9677, r9492, r9493;
}
{
mul.f16x2 r9680, r9677, r9481;
}
{
add.f16x2 r9683, r9674, r9680;
}
{
add.f16x2 %15, r9668, r9683;
}
{
add.f16x2 r9689, r9498, r9499;
}
{
mul.f16x2 r9692, r9689, r9475;
}
{
add.f16x2 r9695, r9501, r9692;
}
{
add.f16x2 r9698, r9504, r9505;
}
{
mul.f16x2 r9701, r9698, r9479;
}
{
add.f16x2 r9704, r9695, r9701;
}
{
sub.f16x2 r9707, r9486, r9487;
}
{
mul.f16x2 r9710, r9707, r9477;
}
{
sub.f16x2 r9713, r9492, r9493;
}
{
mul.f16x2 r9716, r9713, r9481;
}
{
add.f16x2 r9719, r9710, r9716;
}
{
sub.f16x2 %45, r9704, r9719;
}
{
add.f16x2 r9725, r9498, r9499;
}
{
mul.f16x2 r9728, r9725, r9479;
}
{
add.f16x2 r9731, r9501, r9728;
}
{
add.f16x2 r9734, r9504, r9505;
}
{
mul.f16x2 r9737, r9734, r9483;
}
{
add.f16x2 r9740, r9731, r9737;
}
{
sub.f16x2 r9743, r9486, r9487;
}
{
mul.f16x2 r9746, r9743, r9481;
}
{
sub.f16x2 r9749, r9492, r9493;
}
{
mul.f16x2 r9752, r9749, r9484;
}
{
add.f16x2 r9755, r9746, r9752;
}
{
add.f16x2 %25, r9740, r9755;
}
{
add.f16x2 r9761, r9498, r9499;
}
{
mul.f16x2 r9764, r9761, r9479;
}
{
add.f16x2 r9767, r9501, r9764;
}
{
add.f16x2 r9770, r9504, r9505;
}
{
mul.f16x2 r9773, r9770, r9483;
}
{
add.f16x2 r9776, r9767, r9773;
}
{
sub.f16x2 r9779, r9486, r9487;
}
{
mul.f16x2 r9782, r9779, r9481;
}
{
sub.f16x2 r9785, r9492, r9493;
}
{
mul.f16x2 r9788, r9785, r9484;
}
{
add.f16x2 r9791, r9782, r9788;
}
{
sub.f16x2 %35, r9776, r9791;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9797, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9798, {low, high};
}
{
neg.f16x2 r9799, r9798;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r9801, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r9802, {low, high};
}
{
neg.f16x2 r9803, r9802;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9805, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9806, {low, high};
}
{
add.f16x2 r9807, r9808, r9809;
}
{
add.f16x2 r9810, r9811, r9807;
}
{
add.f16x2 r9813, r9814, r9815;
}
{
add.f16x2 %6, r9810, r9813;
}
{
add.f16x2 r9819, r9820, r9821;
}
{
add.f16x2 r9822, r9823, r9819;
}
{
add.f16x2 r9825, r9826, r9827;
}
{
add.f16x2 %7, r9822, r9825;
}
{
add.f16x2 r9831, r9808, r9809;
}
{
mul.f16x2 r9834, r9831, r9797;
}
{
add.f16x2 r9837, r9811, r9834;
}
{
add.f16x2 r9840, r9814, r9815;
}
{
mul.f16x2 r9843, r9840, r9801;
}
{
add.f16x2 r9846, r9837, r9843;
}
{
sub.f16x2 r9849, r9820, r9821;
}
{
mul.f16x2 r9852, r9849, r9799;
}
{
sub.f16x2 r9855, r9826, r9827;
}
{
mul.f16x2 r9858, r9855, r9803;
}
{
add.f16x2 r9861, r9852, r9858;
}
{
sub.f16x2 %16, r9846, r9861;
}
{
add.f16x2 r9867, r9808, r9809;
}
{
mul.f16x2 r9870, r9867, r9797;
}
{
add.f16x2 r9873, r9811, r9870;
}
{
add.f16x2 r9876, r9814, r9815;
}
{
mul.f16x2 r9879, r9876, r9801;
}
{
add.f16x2 r9882, r9873, r9879;
}
{
sub.f16x2 r9885, r9820, r9821;
}
{
mul.f16x2 r9888, r9885, r9799;
}
{
sub.f16x2 r9891, r9826, r9827;
}
{
mul.f16x2 r9894, r9891, r9803;
}
{
add.f16x2 r9897, r9888, r9894;
}
{
add.f16x2 %46, r9882, r9897;
}
{
add.f16x2 r9903, r9808, r9809;
}
{
mul.f16x2 r9906, r9903, r9801;
}
{
add.f16x2 r9909, r9811, r9906;
}
{
add.f16x2 r9912, r9814, r9815;
}
{
mul.f16x2 r9915, r9912, r9805;
}
{
add.f16x2 r9918, r9909, r9915;
}
{
sub.f16x2 r9921, r9820, r9821;
}
{
mul.f16x2 r9924, r9921, r9803;
}
{
sub.f16x2 r9927, r9826, r9827;
}
{
mul.f16x2 r9930, r9927, r9806;
}
{
add.f16x2 r9933, r9924, r9930;
}
{
sub.f16x2 %26, r9918, r9933;
}
{
add.f16x2 r9939, r9808, r9809;
}
{
mul.f16x2 r9942, r9939, r9801;
}
{
add.f16x2 r9945, r9811, r9942;
}
{
add.f16x2 r9948, r9814, r9815;
}
{
mul.f16x2 r9951, r9948, r9805;
}
{
add.f16x2 r9954, r9945, r9951;
}
{
sub.f16x2 r9957, r9820, r9821;
}
{
mul.f16x2 r9960, r9957, r9803;
}
{
sub.f16x2 r9963, r9826, r9827;
}
{
mul.f16x2 r9966, r9963, r9806;
}
{
add.f16x2 r9969, r9960, r9966;
}
{
add.f16x2 %36, r9954, r9969;
}
{
add.f16x2 r9975, r9820, r9821;
}
{
mul.f16x2 r9978, r9975, r9797;
}
{
add.f16x2 r9981, r9823, r9978;
}
{
add.f16x2 r9984, r9826, r9827;
}
{
mul.f16x2 r9987, r9984, r9801;
}
{
add.f16x2 r9990, r9981, r9987;
}
{
sub.f16x2 r9993, r9808, r9809;
}
{
mul.f16x2 r9996, r9993, r9799;
}
{
sub.f16x2 r9999, r9814, r9815;
}
{
mul.f16x2 r10002, r9999, r9803;
}
{
add.f16x2 r10005, r9996, r10002;
}
{
add.f16x2 %17, r9990, r10005;
}
{
add.f16x2 r10011, r9820, r9821;
}
{
mul.f16x2 r10014, r10011, r9797;
}
{
add.f16x2 r10017, r9823, r10014;
}
{
add.f16x2 r10020, r9826, r9827;
}
{
mul.f16x2 r10023, r10020, r9801;
}
{
add.f16x2 r10026, r10017, r10023;
}
{
sub.f16x2 r10029, r9808, r9809;
}
{
mul.f16x2 r10032, r10029, r9799;
}
{
sub.f16x2 r10035, r9814, r9815;
}
{
mul.f16x2 r10038, r10035, r9803;
}
{
add.f16x2 r10041, r10032, r10038;
}
{
sub.f16x2 %47, r10026, r10041;
}
{
add.f16x2 r10047, r9820, r9821;
}
{
mul.f16x2 r10050, r10047, r9801;
}
{
add.f16x2 r10053, r9823, r10050;
}
{
add.f16x2 r10056, r9826, r9827;
}
{
mul.f16x2 r10059, r10056, r9805;
}
{
add.f16x2 r10062, r10053, r10059;
}
{
sub.f16x2 r10065, r9808, r9809;
}
{
mul.f16x2 r10068, r10065, r9803;
}
{
sub.f16x2 r10071, r9814, r9815;
}
{
mul.f16x2 r10074, r10071, r9806;
}
{
add.f16x2 r10077, r10068, r10074;
}
{
add.f16x2 %27, r10062, r10077;
}
{
add.f16x2 r10083, r9820, r9821;
}
{
mul.f16x2 r10086, r10083, r9801;
}
{
add.f16x2 r10089, r9823, r10086;
}
{
add.f16x2 r10092, r9826, r9827;
}
{
mul.f16x2 r10095, r10092, r9805;
}
{
add.f16x2 r10098, r10089, r10095;
}
{
sub.f16x2 r10101, r9808, r9809;
}
{
mul.f16x2 r10104, r10101, r9803;
}
{
sub.f16x2 r10107, r9814, r9815;
}
{
mul.f16x2 r10110, r10107, r9806;
}
{
add.f16x2 r10113, r10104, r10110;
}
{
sub.f16x2 %37, r10098, r10113;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r10119, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r10120, {low, high};
}
{
neg.f16x2 r10121, r10120;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r10123, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r10124, {low, high};
}
{
neg.f16x2 r10125, r10124;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r10127, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r10128, {low, high};
}
{
add.f16x2 r10129, r10130, r10131;
}
{
add.f16x2 r10132, r10133, r10129;
}
{
add.f16x2 r10135, r10136, r10137;
}
{
add.f16x2 %8, r10132, r10135;
}
{
add.f16x2 r10141, r10142, r10143;
}
{
add.f16x2 r10144, r10145, r10141;
}
{
add.f16x2 r10147, r10148, r10149;
}
{
add.f16x2 %9, r10144, r10147;
}
{
add.f16x2 r10153, r10130, r10131;
}
{
mul.f16x2 r10156, r10153, r10119;
}
{
add.f16x2 r10159, r10133, r10156;
}
{
add.f16x2 r10162, r10136, r10137;
}
{
mul.f16x2 r10165, r10162, r10123;
}
{
add.f16x2 r10168, r10159, r10165;
}
{
sub.f16x2 r10171, r10142, r10143;
}
{
mul.f16x2 r10174, r10171, r10121;
}
{
sub.f16x2 r10177, r10148, r10149;
}
{
mul.f16x2 r10180, r10177, r10125;
}
{
add.f16x2 r10183, r10174, r10180;
}
{
sub.f16x2 %18, r10168, r10183;
}
{
add.f16x2 r10189, r10130, r10131;
}
{
mul.f16x2 r10192, r10189, r10119;
}
{
add.f16x2 r10195, r10133, r10192;
}
{
add.f16x2 r10198, r10136, r10137;
}
{
mul.f16x2 r10201, r10198, r10123;
}
{
add.f16x2 r10204, r10195, r10201;
}
{
sub.f16x2 r10207, r10142, r10143;
}
{
mul.f16x2 r10210, r10207, r10121;
}
{
sub.f16x2 r10213, r10148, r10149;
}
{
mul.f16x2 r10216, r10213, r10125;
}
{
add.f16x2 r10219, r10210, r10216;
}
{
add.f16x2 %48, r10204, r10219;
}
{
add.f16x2 r10225, r10130, r10131;
}
{
mul.f16x2 r10228, r10225, r10123;
}
{
add.f16x2 r10231, r10133, r10228;
}
{
add.f16x2 r10234, r10136, r10137;
}
{
mul.f16x2 r10237, r10234, r10127;
}
{
add.f16x2 r10240, r10231, r10237;
}
{
sub.f16x2 r10243, r10142, r10143;
}
{
mul.f16x2 r10246, r10243, r10125;
}
{
sub.f16x2 r10249, r10148, r10149;
}
{
mul.f16x2 r10252, r10249, r10128;
}
{
add.f16x2 r10255, r10246, r10252;
}
{
sub.f16x2 %28, r10240, r10255;
}
{
add.f16x2 r10261, r10130, r10131;
}
{
mul.f16x2 r10264, r10261, r10123;
}
{
add.f16x2 r10267, r10133, r10264;
}
{
add.f16x2 r10270, r10136, r10137;
}
{
mul.f16x2 r10273, r10270, r10127;
}
{
add.f16x2 r10276, r10267, r10273;
}
{
sub.f16x2 r10279, r10142, r10143;
}
{
mul.f16x2 r10282, r10279, r10125;
}
{
sub.f16x2 r10285, r10148, r10149;
}
{
mul.f16x2 r10288, r10285, r10128;
}
{
add.f16x2 r10291, r10282, r10288;
}
{
add.f16x2 %38, r10276, r10291;
}
{
add.f16x2 r10297, r10142, r10143;
}
{
mul.f16x2 r10300, r10297, r10119;
}
{
add.f16x2 r10303, r10145, r10300;
}
{
add.f16x2 r10306, r10148, r10149;
}
{
mul.f16x2 r10309, r10306, r10123;
}
{
add.f16x2 r10312, r10303, r10309;
}
{
sub.f16x2 r10315, r10130, r10131;
}
{
mul.f16x2 r10318, r10315, r10121;
}
{
sub.f16x2 r10321, r10136, r10137;
}
{
mul.f16x2 r10324, r10321, r10125;
}
{
add.f16x2 r10327, r10318, r10324;
}
{
add.f16x2 %19, r10312, r10327;
}
{
add.f16x2 r10333, r10142, r10143;
}
{
mul.f16x2 r10336, r10333, r10119;
}
{
add.f16x2 r10339, r10145, r10336;
}
{
add.f16x2 r10342, r10148, r10149;
}
{
mul.f16x2 r10345, r10342, r10123;
}
{
add.f16x2 r10348, r10339, r10345;
}
{
sub.f16x2 r10351, r10130, r10131;
}
{
mul.f16x2 r10354, r10351, r10121;
}
{
sub.f16x2 r10357, r10136, r10137;
}
{
mul.f16x2 r10360, r10357, r10125;
}
{
add.f16x2 r10363, r10354, r10360;
}
{
sub.f16x2 %49, r10348, r10363;
}
{
add.f16x2 r10369, r10142, r10143;
}
{
mul.f16x2 r10372, r10369, r10123;
}
{
add.f16x2 r10375, r10145, r10372;
}
{
add.f16x2 r10378, r10148, r10149;
}
{
mul.f16x2 r10381, r10378, r10127;
}
{
add.f16x2 r10384, r10375, r10381;
}
{
sub.f16x2 r10387, r10130, r10131;
}
{
mul.f16x2 r10390, r10387, r10125;
}
{
sub.f16x2 r10393, r10136, r10137;
}
{
mul.f16x2 r10396, r10393, r10128;
}
{
add.f16x2 r10399, r10390, r10396;
}
{
add.f16x2 %29, r10384, r10399;
}
{
add.f16x2 r10405, r10142, r10143;
}
{
mul.f16x2 r10408, r10405, r10123;
}
{
add.f16x2 r10411, r10145, r10408;
}
{
add.f16x2 r10414, r10148, r10149;
}
{
mul.f16x2 r10417, r10414, r10127;
}
{
add.f16x2 r10420, r10411, r10417;
}
{
sub.f16x2 r10423, r10130, r10131;
}
{
mul.f16x2 r10426, r10423, r10125;
}
{
sub.f16x2 r10429, r10136, r10137;
}
{
mul.f16x2 r10432, r10429, r10128;
}
{
add.f16x2 r10435, r10426, r10432;
}
{
sub.f16x2 %39, r10420, r10435;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[13].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1116, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<121>;
.reg .b32 r<2244>;
.reg .b64 rd<10>;
mov.u32 r2215, %tid.x;
mov.f32 f106, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1, {low, high};
}
mov.f32 f108, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r2, {low, high};
}
{
neg.f16x2 r3, r2;
}
mov.f32 f102, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r5, {low, high};
}
mov.f32 f104, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r9, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r10, {low, high};
}
{
add.f16x2 r11, %13, %19;
}
{
add.f16x2 r14, %11, r11;
}
{
add.f16x2 r17, %15, %17;
}
{
add.f16x2 r20, r14, r17;
}
{
add.f16x2 r23, %14, %20;
}
{
add.f16x2 r26, %12, r23;
}
{
add.f16x2 r29, %16, %18;
}
{
add.f16x2 r32, r26, r29;
}
{
add.f16x2 r35, %13, %19;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %11, r38;
}
{
add.f16x2 r44, %15, %17;
}
{
mul.f16x2 r47, r44, r5;
}
{
add.f16x2 r50, r41, r47;
}
{
sub.f16x2 r53, %14, %20;
}
{
mul.f16x2 r56, r53, r3;
}
{
sub.f16x2 r59, %16, %18;
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
add.f16x2 r71, %13, %19;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %11, r74;
}
{
add.f16x2 r80, %15, %17;
}
{
mul.f16x2 r83, r80, r5;
}
{
add.f16x2 r86, r77, r83;
}
{
sub.f16x2 r89, %14, %20;
}
{
mul.f16x2 r92, r89, r3;
}
{
sub.f16x2 r95, %16, %18;
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
add.f16x2 r107, %13, %19;
}
{
mul.f16x2 r110, r107, r5;
}
{
add.f16x2 r113, %11, r110;
}
{
add.f16x2 r116, %15, %17;
}
{
mul.f16x2 r119, r116, r9;
}
{
add.f16x2 r122, r113, r119;
}
{
sub.f16x2 r125, %14, %20;
}
{
mul.f16x2 r128, r125, r7;
}
{
sub.f16x2 r131, %16, %18;
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
add.f16x2 r143, %13, %19;
}
{
mul.f16x2 r146, r143, r5;
}
{
add.f16x2 r149, %11, r146;
}
{
add.f16x2 r152, %15, %17;
}
{
mul.f16x2 r155, r152, r9;
}
{
add.f16x2 r158, r149, r155;
}
{
sub.f16x2 r161, %14, %20;
}
{
mul.f16x2 r164, r161, r7;
}
{
sub.f16x2 r167, %16, %18;
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
add.f16x2 r179, %14, %20;
}
{
mul.f16x2 r182, r179, r1;
}
{
add.f16x2 r185, %12, r182;
}
{
add.f16x2 r188, %16, %18;
}
{
mul.f16x2 r191, r188, r5;
}
{
add.f16x2 r194, r185, r191;
}
{
sub.f16x2 r197, %13, %19;
}
{
mul.f16x2 r200, r197, r3;
}
{
sub.f16x2 r203, %15, %17;
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
add.f16x2 r215, %14, %20;
}
{
mul.f16x2 r218, r215, r1;
}
{
add.f16x2 r221, %12, r218;
}
{
add.f16x2 r224, %16, %18;
}
{
mul.f16x2 r227, r224, r5;
}
{
add.f16x2 r230, r221, r227;
}
{
sub.f16x2 r233, %13, %19;
}
{
mul.f16x2 r236, r233, r3;
}
{
sub.f16x2 r239, %15, %17;
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
add.f16x2 r251, %14, %20;
}
{
mul.f16x2 r254, r251, r5;
}
{
add.f16x2 r257, %12, r254;
}
{
add.f16x2 r260, %16, %18;
}
{
mul.f16x2 r263, r260, r9;
}
{
add.f16x2 r266, r257, r263;
}
{
sub.f16x2 r269, %13, %19;
}
{
mul.f16x2 r272, r269, r7;
}
{
sub.f16x2 r275, %15, %17;
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
add.f16x2 r287, %14, %20;
}
{
mul.f16x2 r290, r287, r5;
}
{
add.f16x2 r293, %12, r290;
}
{
add.f16x2 r296, %16, %18;
}
{
mul.f16x2 r299, r296, r9;
}
{
add.f16x2 r302, r293, r299;
}
{
sub.f16x2 r305, %13, %19;
}
{
mul.f16x2 r308, r305, r7;
}
{
sub.f16x2 r311, %15, %17;
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
mul.wide.u32 rd2, r2215, -776530087;
shr.u64 rd3, rd2, 41;
cvt.u32.u64 r2216, rd3;
mul.lo.s32 r2217, r2216, 625;
sub.s32 r2218, r2215, r2217;
cvt.rn.f32.u32 f109, r2218;
mul.f32 f110, f109, 0f3B03C498;
cos.approx.f32 f13, f110;
sin.approx.f32 f111, f110;
neg.f32 f14, f111;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f13;
cvt.rn.f16.f32 high, f14;
mov.b32 r323, {low, high};
}
mov.u32 r2219, %tid.y;
mov.u32 r2220, %10;
mad.lo.s32 r2221, r2219, 25000, r2220;
{
.reg .f16 low, high;
mov.b32 {low, high}, r323;
mov.b32 r326, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r323;
mov.b32 r328, {high, high};
}
{
mul.f16x2 r330, r212, r328;
}
{
fma.rn.f16x2 r333, r68, r326, r330;
}
{
mul.f16x2 r337, r68, r328;
}
{
neg.f16x2 r340, r337;
}
{
fma.rn.f16x2 r342, r212, r326, r340;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r323;
mov.b32 r346, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r323;
mov.b32 r348, {high, high};
}
mov.f32 f93, 0fBF800000;
mov.f32 f94, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r350, {low, high};
}
{
mul.f16x2 r351, r348, r350;
}
{
mul.f16x2 r354, r323, r346;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r323;
mov.b32 r357, {high, low};
}
{
fma.rn.f16x2 r359, r351, r357, r354;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r359;
mov.b32 r363, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r359;
mov.b32 r365, {high, high};
}
{
mul.f16x2 r367, r284, r365;
}
{
fma.rn.f16x2 r370, r140, r363, r367;
}
{
mul.f16x2 r374, r140, r365;
}
{
neg.f16x2 r377, r374;
}
{
fma.rn.f16x2 r379, r284, r363, r377;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r323;
mov.b32 r383, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r323;
mov.b32 r385, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r387, {low, high};
}
{
mul.f16x2 r388, r385, r387;
}
{
mul.f16x2 r391, r359, r383;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r359;
mov.b32 r394, {high, low};
}
{
fma.rn.f16x2 r396, r388, r394, r391;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r396;
mov.b32 r400, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r396;
mov.b32 r402, {high, high};
}
{
mul.f16x2 r404, r320, r402;
}
{
fma.rn.f16x2 r407, r176, r400, r404;
}
{
mul.f16x2 r411, r176, r402;
}
{
neg.f16x2 r414, r411;
}
{
fma.rn.f16x2 r416, r320, r400, r414;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r323;
mov.b32 r420, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r323;
mov.b32 r422, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r424, {low, high};
}
{
mul.f16x2 r425, r422, r424;
}
{
mul.f16x2 r428, r396, r420;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r396;
mov.b32 r431, {high, low};
}
{
fma.rn.f16x2 r433, r425, r431, r428;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r433;
mov.b32 r437, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r433;
mov.b32 r439, {high, high};
}
{
mul.f16x2 r441, r248, r439;
}
{
fma.rn.f16x2 r444, r104, r437, r441;
}
{
mul.f16x2 r448, r104, r439;
}
{
neg.f16x2 r451, r448;
}
{
fma.rn.f16x2 r453, r248, r437, r451;
}
mad.lo.s32 r2222, r2216, 25000, r2221;
barrier.sync 0;
mad.lo.s32 r2223, r2218, 40, r2222;
st.shared.v2.f32 [r2223], {r20, r32};
st.shared.v2.f32 [r2223+8], {r333, r342};
st.shared.v2.f32 [r2223+16], {r370, r379};
st.shared.v2.f32 [r2223+24], {r407, r416};
st.shared.v2.f32 [r2223+32], {r444, r453};
barrier.sync 0;
shl.b32 r2224, r2218, 5;
sub.s32 r2225, r2223, r2224;
ld.shared.u32 r488, [r2225];
ld.shared.u32 r500, [r2225+4];
ld.shared.u32 r485, [r2225+5000];
ld.shared.u32 r497, [r2225+5004];
ld.shared.u32 r491, [r2225+10000];
ld.shared.u32 r503, [r2225+10004];
ld.shared.u32 r492, [r2225+15000];
ld.shared.u32 r504, [r2225+15004];
ld.shared.u32 r486, [r2225+20000];
ld.shared.u32 r498, [r2225+20004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r474, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r475, {low, high};
}
{
neg.f16x2 r476, r475;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r478, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r479, {low, high};
}
{
neg.f16x2 r480, r479;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r482, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r483, {low, high};
}
{
add.f16x2 r484, r485, r486;
}
{
add.f16x2 r487, r488, r484;
}
{
add.f16x2 r490, r491, r492;
}
{
add.f16x2 r493, r487, r490;
}
{
add.f16x2 r496, r497, r498;
}
{
add.f16x2 r499, r500, r496;
}
{
add.f16x2 r502, r503, r504;
}
{
add.f16x2 r505, r499, r502;
}
{
add.f16x2 r508, r485, r486;
}
{
mul.f16x2 r511, r508, r474;
}
{
add.f16x2 r514, r488, r511;
}
{
add.f16x2 r517, r491, r492;
}
{
mul.f16x2 r520, r517, r478;
}
{
add.f16x2 r523, r514, r520;
}
{
sub.f16x2 r526, r497, r498;
}
{
mul.f16x2 r529, r526, r476;
}
{
sub.f16x2 r532, r503, r504;
}
{
mul.f16x2 r535, r532, r480;
}
{
add.f16x2 r538, r529, r535;
}
{
sub.f16x2 r541, r523, r538;
}
{
add.f16x2 r544, r485, r486;
}
{
mul.f16x2 r547, r544, r474;
}
{
add.f16x2 r550, r488, r547;
}
{
add.f16x2 r553, r491, r492;
}
{
mul.f16x2 r556, r553, r478;
}
{
add.f16x2 r559, r550, r556;
}
{
sub.f16x2 r562, r497, r498;
}
{
mul.f16x2 r565, r562, r476;
}
{
sub.f16x2 r568, r503, r504;
}
{
mul.f16x2 r571, r568, r480;
}
{
add.f16x2 r574, r565, r571;
}
{
add.f16x2 r577, r559, r574;
}
{
add.f16x2 r580, r485, r486;
}
{
mul.f16x2 r583, r580, r478;
}
{
add.f16x2 r586, r488, r583;
}
{
add.f16x2 r589, r491, r492;
}
{
mul.f16x2 r592, r589, r482;
}
{
add.f16x2 r595, r586, r592;
}
{
sub.f16x2 r598, r497, r498;
}
{
mul.f16x2 r601, r598, r480;
}
{
sub.f16x2 r604, r503, r504;
}
{
mul.f16x2 r607, r604, r483;
}
{
add.f16x2 r610, r601, r607;
}
{
sub.f16x2 r613, r595, r610;
}
{
add.f16x2 r616, r485, r486;
}
{
mul.f16x2 r619, r616, r478;
}
{
add.f16x2 r622, r488, r619;
}
{
add.f16x2 r625, r491, r492;
}
{
mul.f16x2 r628, r625, r482;
}
{
add.f16x2 r631, r622, r628;
}
{
sub.f16x2 r634, r497, r498;
}
{
mul.f16x2 r637, r634, r480;
}
{
sub.f16x2 r640, r503, r504;
}
{
mul.f16x2 r643, r640, r483;
}
{
add.f16x2 r646, r637, r643;
}
{
add.f16x2 r649, r631, r646;
}
{
add.f16x2 r652, r497, r498;
}
{
mul.f16x2 r655, r652, r474;
}
{
add.f16x2 r658, r500, r655;
}
{
add.f16x2 r661, r503, r504;
}
{
mul.f16x2 r664, r661, r478;
}
{
add.f16x2 r667, r658, r664;
}
{
sub.f16x2 r670, r485, r486;
}
{
mul.f16x2 r673, r670, r476;
}
{
sub.f16x2 r676, r491, r492;
}
{
mul.f16x2 r679, r676, r480;
}
{
add.f16x2 r682, r673, r679;
}
{
add.f16x2 r685, r667, r682;
}
{
add.f16x2 r688, r497, r498;
}
{
mul.f16x2 r691, r688, r474;
}
{
add.f16x2 r694, r500, r691;
}
{
add.f16x2 r697, r503, r504;
}
{
mul.f16x2 r700, r697, r478;
}
{
add.f16x2 r703, r694, r700;
}
{
sub.f16x2 r706, r485, r486;
}
{
mul.f16x2 r709, r706, r476;
}
{
sub.f16x2 r712, r491, r492;
}
{
mul.f16x2 r715, r712, r480;
}
{
add.f16x2 r718, r709, r715;
}
{
sub.f16x2 r721, r703, r718;
}
{
add.f16x2 r724, r497, r498;
}
{
mul.f16x2 r727, r724, r478;
}
{
add.f16x2 r730, r500, r727;
}
{
add.f16x2 r733, r503, r504;
}
{
mul.f16x2 r736, r733, r482;
}
{
add.f16x2 r739, r730, r736;
}
{
sub.f16x2 r742, r485, r486;
}
{
mul.f16x2 r745, r742, r480;
}
{
sub.f16x2 r748, r491, r492;
}
{
mul.f16x2 r751, r748, r483;
}
{
add.f16x2 r754, r745, r751;
}
{
add.f16x2 r757, r739, r754;
}
{
add.f16x2 r760, r497, r498;
}
{
mul.f16x2 r763, r760, r478;
}
{
add.f16x2 r766, r500, r763;
}
{
add.f16x2 r769, r503, r504;
}
{
mul.f16x2 r772, r769, r482;
}
{
add.f16x2 r775, r766, r772;
}
{
sub.f16x2 r778, r485, r486;
}
{
mul.f16x2 r781, r778, r480;
}
{
sub.f16x2 r784, r491, r492;
}
{
mul.f16x2 r787, r784, r483;
}
{
add.f16x2 r790, r781, r787;
}
{
sub.f16x2 r793, r775, r790;
}
mul.wide.u32 rd4, r2218, -858993459;
shr.u64 rd5, rd4, 34;
cvt.u32.u64 r2226, rd5;
cvt.rn.f32.u32 f112, r2226;
mul.f32 f113, f112, 0f3C24B5BE;
cos.approx.f32 f37, f113;
sin.approx.f32 f114, f113;
neg.f32 f38, f114;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
mov.b32 r796, {low, high};
}
mul.lo.s32 r2227, r2226, 5;
sub.s32 r2228, r2218, r2227;
{
.reg .f16 low, high;
mov.b32 {low, high}, r796;
mov.b32 r799, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r796;
mov.b32 r801, {high, high};
}
{
mul.f16x2 r803, r685, r801;
}
{
fma.rn.f16x2 r806, r541, r799, r803;
}
{
mul.f16x2 r810, r541, r801;
}
{
neg.f16x2 r813, r810;
}
{
fma.rn.f16x2 r815, r685, r799, r813;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r796;
mov.b32 r819, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r796;
mov.b32 r821, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r823, {low, high};
}
{
mul.f16x2 r824, r821, r823;
}
{
mul.f16x2 r827, r796, r819;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r796;
mov.b32 r830, {high, low};
}
{
fma.rn.f16x2 r832, r824, r830, r827;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r832;
mov.b32 r836, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r832;
mov.b32 r838, {high, high};
}
{
mul.f16x2 r840, r757, r838;
}
{
fma.rn.f16x2 r843, r613, r836, r840;
}
{
mul.f16x2 r847, r613, r838;
}
{
neg.f16x2 r850, r847;
}
{
fma.rn.f16x2 r852, r757, r836, r850;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r796;
mov.b32 r856, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r796;
mov.b32 r858, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r860, {low, high};
}
{
mul.f16x2 r861, r858, r860;
}
{
mul.f16x2 r864, r832, r856;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r832;
mov.b32 r867, {high, low};
}
{
fma.rn.f16x2 r869, r861, r867, r864;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r869;
mov.b32 r873, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r869;
mov.b32 r875, {high, high};
}
{
mul.f16x2 r877, r793, r875;
}
{
fma.rn.f16x2 r880, r649, r873, r877;
}
{
mul.f16x2 r884, r649, r875;
}
{
neg.f16x2 r887, r884;
}
{
fma.rn.f16x2 r889, r793, r873, r887;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r796;
mov.b32 r893, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r796;
mov.b32 r895, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r897, {low, high};
}
{
mul.f16x2 r898, r895, r897;
}
{
mul.f16x2 r901, r869, r893;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r869;
mov.b32 r904, {high, low};
}
{
fma.rn.f16x2 r906, r898, r904, r901;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r906;
mov.b32 r910, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r906;
mov.b32 r912, {high, high};
}
{
mul.f16x2 r914, r721, r912;
}
{
fma.rn.f16x2 r917, r577, r910, r914;
}
{
mul.f16x2 r921, r577, r912;
}
{
neg.f16x2 r924, r921;
}
{
fma.rn.f16x2 r926, r721, r910, r924;
}
shl.b32 r2229, r2228, 3;
add.s32 r2230, r2222, r2229;
barrier.sync 0;
mad.lo.s32 r2231, r2226, 200, r2230;
st.shared.u32 [r2231], r493;
st.shared.u32 [r2231+4], r505;
st.shared.u32 [r2231+40], r806;
st.shared.u32 [r2231+44], r815;
st.shared.u32 [r2231+80], r843;
st.shared.u32 [r2231+84], r852;
st.shared.u32 [r2231+120], r880;
st.shared.u32 [r2231+124], r889;
st.shared.u32 [r2231+160], r917;
st.shared.u32 [r2231+164], r926;
barrier.sync 0;
ld.shared.u32 r961, [r2225];
ld.shared.u32 r973, [r2225+4];
ld.shared.u32 r958, [r2225+5000];
ld.shared.u32 r970, [r2225+5004];
ld.shared.u32 r964, [r2225+10000];
ld.shared.u32 r976, [r2225+10004];
ld.shared.u32 r965, [r2225+15000];
ld.shared.u32 r977, [r2225+15004];
ld.shared.u32 r959, [r2225+20000];
ld.shared.u32 r971, [r2225+20004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r947, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r948, {low, high};
}
{
neg.f16x2 r949, r948;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r951, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r952, {low, high};
}
{
neg.f16x2 r953, r952;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r955, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r956, {low, high};
}
{
add.f16x2 r957, r958, r959;
}
{
add.f16x2 r960, r961, r957;
}
{
add.f16x2 r963, r964, r965;
}
{
add.f16x2 r966, r960, r963;
}
{
add.f16x2 r969, r970, r971;
}
{
add.f16x2 r972, r973, r969;
}
{
add.f16x2 r975, r976, r977;
}
{
add.f16x2 r978, r972, r975;
}
{
add.f16x2 r981, r958, r959;
}
{
mul.f16x2 r984, r981, r947;
}
{
add.f16x2 r987, r961, r984;
}
{
add.f16x2 r990, r964, r965;
}
{
mul.f16x2 r993, r990, r951;
}
{
add.f16x2 r996, r987, r993;
}
{
sub.f16x2 r999, r970, r971;
}
{
mul.f16x2 r1002, r999, r949;
}
{
sub.f16x2 r1005, r976, r977;
}
{
mul.f16x2 r1008, r1005, r953;
}
{
add.f16x2 r1011, r1002, r1008;
}
{
sub.f16x2 r1014, r996, r1011;
}
{
add.f16x2 r1017, r958, r959;
}
{
mul.f16x2 r1020, r1017, r947;
}
{
add.f16x2 r1023, r961, r1020;
}
{
add.f16x2 r1026, r964, r965;
}
{
mul.f16x2 r1029, r1026, r951;
}
{
add.f16x2 r1032, r1023, r1029;
}
{
sub.f16x2 r1035, r970, r971;
}
{
mul.f16x2 r1038, r1035, r949;
}
{
sub.f16x2 r1041, r976, r977;
}
{
mul.f16x2 r1044, r1041, r953;
}
{
add.f16x2 r1047, r1038, r1044;
}
{
add.f16x2 r1050, r1032, r1047;
}
{
add.f16x2 r1053, r958, r959;
}
{
mul.f16x2 r1056, r1053, r951;
}
{
add.f16x2 r1059, r961, r1056;
}
{
add.f16x2 r1062, r964, r965;
}
{
mul.f16x2 r1065, r1062, r955;
}
{
add.f16x2 r1068, r1059, r1065;
}
{
sub.f16x2 r1071, r970, r971;
}
{
mul.f16x2 r1074, r1071, r953;
}
{
sub.f16x2 r1077, r976, r977;
}
{
mul.f16x2 r1080, r1077, r956;
}
{
add.f16x2 r1083, r1074, r1080;
}
{
sub.f16x2 r1086, r1068, r1083;
}
{
add.f16x2 r1089, r958, r959;
}
{
mul.f16x2 r1092, r1089, r951;
}
{
add.f16x2 r1095, r961, r1092;
}
{
add.f16x2 r1098, r964, r965;
}
{
mul.f16x2 r1101, r1098, r955;
}
{
add.f16x2 r1104, r1095, r1101;
}
{
sub.f16x2 r1107, r970, r971;
}
{
mul.f16x2 r1110, r1107, r953;
}
{
sub.f16x2 r1113, r976, r977;
}
{
mul.f16x2 r1116, r1113, r956;
}
{
add.f16x2 r1119, r1110, r1116;
}
{
add.f16x2 r1122, r1104, r1119;
}
{
add.f16x2 r1125, r970, r971;
}
{
mul.f16x2 r1128, r1125, r947;
}
{
add.f16x2 r1131, r973, r1128;
}
{
add.f16x2 r1134, r976, r977;
}
{
mul.f16x2 r1137, r1134, r951;
}
{
add.f16x2 r1140, r1131, r1137;
}
{
sub.f16x2 r1143, r958, r959;
}
{
mul.f16x2 r1146, r1143, r949;
}
{
sub.f16x2 r1149, r964, r965;
}
{
mul.f16x2 r1152, r1149, r953;
}
{
add.f16x2 r1155, r1146, r1152;
}
{
add.f16x2 r1158, r1140, r1155;
}
{
add.f16x2 r1161, r970, r971;
}
{
mul.f16x2 r1164, r1161, r947;
}
{
add.f16x2 r1167, r973, r1164;
}
{
add.f16x2 r1170, r976, r977;
}
{
mul.f16x2 r1173, r1170, r951;
}
{
add.f16x2 r1176, r1167, r1173;
}
{
sub.f16x2 r1179, r958, r959;
}
{
mul.f16x2 r1182, r1179, r949;
}
{
sub.f16x2 r1185, r964, r965;
}
{
mul.f16x2 r1188, r1185, r953;
}
{
add.f16x2 r1191, r1182, r1188;
}
{
sub.f16x2 r1194, r1176, r1191;
}
{
add.f16x2 r1197, r970, r971;
}
{
mul.f16x2 r1200, r1197, r951;
}
{
add.f16x2 r1203, r973, r1200;
}
{
add.f16x2 r1206, r976, r977;
}
{
mul.f16x2 r1209, r1206, r955;
}
{
add.f16x2 r1212, r1203, r1209;
}
{
sub.f16x2 r1215, r958, r959;
}
{
mul.f16x2 r1218, r1215, r953;
}
{
sub.f16x2 r1221, r964, r965;
}
{
mul.f16x2 r1224, r1221, r956;
}
{
add.f16x2 r1227, r1218, r1224;
}
{
add.f16x2 r1230, r1212, r1227;
}
{
add.f16x2 r1233, r970, r971;
}
{
mul.f16x2 r1236, r1233, r951;
}
{
add.f16x2 r1239, r973, r1236;
}
{
add.f16x2 r1242, r976, r977;
}
{
mul.f16x2 r1245, r1242, r955;
}
{
add.f16x2 r1248, r1239, r1245;
}
{
sub.f16x2 r1251, r958, r959;
}
{
mul.f16x2 r1254, r1251, r953;
}
{
sub.f16x2 r1257, r964, r965;
}
{
mul.f16x2 r1260, r1257, r956;
}
{
add.f16x2 r1263, r1254, r1260;
}
{
sub.f16x2 r1266, r1248, r1263;
}
mul.wide.u32 rd6, r2218, 1374389535;
shr.u64 rd7, rd6, 35;
cvt.u32.u64 r2232, rd7;
cvt.rn.f32.u32 f115, r2232;
mul.f32 f116, f115, 0f3D4DE32E;
cos.approx.f32 f61, f116;
sin.approx.f32 f117, f116;
neg.f32 f62, f117;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f61;
cvt.rn.f16.f32 high, f62;
mov.b32 r1269, {low, high};
}
mul.lo.s32 r2233, r2232, 25;
sub.s32 r2234, r2218, r2233;
{
.reg .f16 low, high;
mov.b32 {low, high}, r1269;
mov.b32 r1272, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1269;
mov.b32 r1274, {high, high};
}
{
mul.f16x2 r1276, r1158, r1274;
}
{
fma.rn.f16x2 r1279, r1014, r1272, r1276;
}
{
mul.f16x2 r1283, r1014, r1274;
}
{
neg.f16x2 r1286, r1283;
}
{
fma.rn.f16x2 r1288, r1158, r1272, r1286;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1269;
mov.b32 r1292, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1269;
mov.b32 r1294, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1296, {low, high};
}
{
mul.f16x2 r1297, r1294, r1296;
}
{
mul.f16x2 r1300, r1269, r1292;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1269;
mov.b32 r1303, {high, low};
}
{
fma.rn.f16x2 r1305, r1297, r1303, r1300;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1305;
mov.b32 r1309, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1305;
mov.b32 r1311, {high, high};
}
{
mul.f16x2 r1313, r1230, r1311;
}
{
fma.rn.f16x2 r1316, r1086, r1309, r1313;
}
{
mul.f16x2 r1320, r1086, r1311;
}
{
neg.f16x2 r1323, r1320;
}
{
fma.rn.f16x2 r1325, r1230, r1309, r1323;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1269;
mov.b32 r1329, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1269;
mov.b32 r1331, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1333, {low, high};
}
{
mul.f16x2 r1334, r1331, r1333;
}
{
mul.f16x2 r1337, r1305, r1329;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1305;
mov.b32 r1340, {high, low};
}
{
fma.rn.f16x2 r1342, r1334, r1340, r1337;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1342;
mov.b32 r1346, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1342;
mov.b32 r1348, {high, high};
}
{
mul.f16x2 r1350, r1266, r1348;
}
{
fma.rn.f16x2 r1353, r1122, r1346, r1350;
}
{
mul.f16x2 r1357, r1122, r1348;
}
{
neg.f16x2 r1360, r1357;
}
{
fma.rn.f16x2 r1362, r1266, r1346, r1360;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1269;
mov.b32 r1366, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1269;
mov.b32 r1368, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1370, {low, high};
}
{
mul.f16x2 r1371, r1368, r1370;
}
{
mul.f16x2 r1374, r1342, r1366;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1342;
mov.b32 r1377, {high, low};
}
{
fma.rn.f16x2 r1379, r1371, r1377, r1374;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1379;
mov.b32 r1383, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1379;
mov.b32 r1385, {high, high};
}
{
mul.f16x2 r1387, r1194, r1385;
}
{
fma.rn.f16x2 r1390, r1050, r1383, r1387;
}
{
mul.f16x2 r1394, r1050, r1385;
}
{
neg.f16x2 r1397, r1394;
}
{
fma.rn.f16x2 r1399, r1194, r1383, r1397;
}
shl.b32 r2235, r2234, 3;
add.s32 r2236, r2222, r2235;
barrier.sync 0;
mad.lo.s32 r2237, r2232, 1000, r2236;
st.shared.u32 [r2237], r966;
st.shared.u32 [r2237+4], r978;
st.shared.u32 [r2237+200], r1279;
st.shared.u32 [r2237+204], r1288;
st.shared.u32 [r2237+400], r1316;
st.shared.u32 [r2237+404], r1325;
st.shared.u32 [r2237+600], r1353;
st.shared.u32 [r2237+604], r1362;
st.shared.u32 [r2237+800], r1390;
st.shared.u32 [r2237+804], r1399;
barrier.sync 0;
ld.shared.u32 r1434, [r2225];
ld.shared.u32 r1446, [r2225+4];
ld.shared.u32 r1431, [r2225+5000];
ld.shared.u32 r1443, [r2225+5004];
ld.shared.u32 r1437, [r2225+10000];
ld.shared.u32 r1449, [r2225+10004];
ld.shared.u32 r1438, [r2225+15000];
ld.shared.u32 r1450, [r2225+15004];
ld.shared.u32 r1432, [r2225+20000];
ld.shared.u32 r1444, [r2225+20004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1420, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1421, {low, high};
}
{
neg.f16x2 r1422, r1421;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r1424, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r1425, {low, high};
}
{
neg.f16x2 r1426, r1425;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1428, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1429, {low, high};
}
{
add.f16x2 r1430, r1431, r1432;
}
{
add.f16x2 r1433, r1434, r1430;
}
{
add.f16x2 r1436, r1437, r1438;
}
{
add.f16x2 r1439, r1433, r1436;
}
{
add.f16x2 r1442, r1443, r1444;
}
{
add.f16x2 r1445, r1446, r1442;
}
{
add.f16x2 r1448, r1449, r1450;
}
{
add.f16x2 r1451, r1445, r1448;
}
{
add.f16x2 r1454, r1431, r1432;
}
{
mul.f16x2 r1457, r1454, r1420;
}
{
add.f16x2 r1460, r1434, r1457;
}
{
add.f16x2 r1463, r1437, r1438;
}
{
mul.f16x2 r1466, r1463, r1424;
}
{
add.f16x2 r1469, r1460, r1466;
}
{
sub.f16x2 r1472, r1443, r1444;
}
{
mul.f16x2 r1475, r1472, r1422;
}
{
sub.f16x2 r1478, r1449, r1450;
}
{
mul.f16x2 r1481, r1478, r1426;
}
{
add.f16x2 r1484, r1475, r1481;
}
{
sub.f16x2 r1487, r1469, r1484;
}
{
add.f16x2 r1490, r1431, r1432;
}
{
mul.f16x2 r1493, r1490, r1420;
}
{
add.f16x2 r1496, r1434, r1493;
}
{
add.f16x2 r1499, r1437, r1438;
}
{
mul.f16x2 r1502, r1499, r1424;
}
{
add.f16x2 r1505, r1496, r1502;
}
{
sub.f16x2 r1508, r1443, r1444;
}
{
mul.f16x2 r1511, r1508, r1422;
}
{
sub.f16x2 r1514, r1449, r1450;
}
{
mul.f16x2 r1517, r1514, r1426;
}
{
add.f16x2 r1520, r1511, r1517;
}
{
add.f16x2 r1523, r1505, r1520;
}
{
add.f16x2 r1526, r1431, r1432;
}
{
mul.f16x2 r1529, r1526, r1424;
}
{
add.f16x2 r1532, r1434, r1529;
}
{
add.f16x2 r1535, r1437, r1438;
}
{
mul.f16x2 r1538, r1535, r1428;
}
{
add.f16x2 r1541, r1532, r1538;
}
{
sub.f16x2 r1544, r1443, r1444;
}
{
mul.f16x2 r1547, r1544, r1426;
}
{
sub.f16x2 r1550, r1449, r1450;
}
{
mul.f16x2 r1553, r1550, r1429;
}
{
add.f16x2 r1556, r1547, r1553;
}
{
sub.f16x2 r1559, r1541, r1556;
}
{
add.f16x2 r1562, r1431, r1432;
}
{
mul.f16x2 r1565, r1562, r1424;
}
{
add.f16x2 r1568, r1434, r1565;
}
{
add.f16x2 r1571, r1437, r1438;
}
{
mul.f16x2 r1574, r1571, r1428;
}
{
add.f16x2 r1577, r1568, r1574;
}
{
sub.f16x2 r1580, r1443, r1444;
}
{
mul.f16x2 r1583, r1580, r1426;
}
{
sub.f16x2 r1586, r1449, r1450;
}
{
mul.f16x2 r1589, r1586, r1429;
}
{
add.f16x2 r1592, r1583, r1589;
}
{
add.f16x2 r1595, r1577, r1592;
}
{
add.f16x2 r1598, r1443, r1444;
}
{
mul.f16x2 r1601, r1598, r1420;
}
{
add.f16x2 r1604, r1446, r1601;
}
{
add.f16x2 r1607, r1449, r1450;
}
{
mul.f16x2 r1610, r1607, r1424;
}
{
add.f16x2 r1613, r1604, r1610;
}
{
sub.f16x2 r1616, r1431, r1432;
}
{
mul.f16x2 r1619, r1616, r1422;
}
{
sub.f16x2 r1622, r1437, r1438;
}
{
mul.f16x2 r1625, r1622, r1426;
}
{
add.f16x2 r1628, r1619, r1625;
}
{
add.f16x2 r1631, r1613, r1628;
}
{
add.f16x2 r1634, r1443, r1444;
}
{
mul.f16x2 r1637, r1634, r1420;
}
{
add.f16x2 r1640, r1446, r1637;
}
{
add.f16x2 r1643, r1449, r1450;
}
{
mul.f16x2 r1646, r1643, r1424;
}
{
add.f16x2 r1649, r1640, r1646;
}
{
sub.f16x2 r1652, r1431, r1432;
}
{
mul.f16x2 r1655, r1652, r1422;
}
{
sub.f16x2 r1658, r1437, r1438;
}
{
mul.f16x2 r1661, r1658, r1426;
}
{
add.f16x2 r1664, r1655, r1661;
}
{
sub.f16x2 r1667, r1649, r1664;
}
{
add.f16x2 r1670, r1443, r1444;
}
{
mul.f16x2 r1673, r1670, r1424;
}
{
add.f16x2 r1676, r1446, r1673;
}
{
add.f16x2 r1679, r1449, r1450;
}
{
mul.f16x2 r1682, r1679, r1428;
}
{
add.f16x2 r1685, r1676, r1682;
}
{
sub.f16x2 r1688, r1431, r1432;
}
{
mul.f16x2 r1691, r1688, r1426;
}
{
sub.f16x2 r1694, r1437, r1438;
}
{
mul.f16x2 r1697, r1694, r1429;
}
{
add.f16x2 r1700, r1691, r1697;
}
{
add.f16x2 r1703, r1685, r1700;
}
{
add.f16x2 r1706, r1443, r1444;
}
{
mul.f16x2 r1709, r1706, r1424;
}
{
add.f16x2 r1712, r1446, r1709;
}
{
add.f16x2 r1715, r1449, r1450;
}
{
mul.f16x2 r1718, r1715, r1428;
}
{
add.f16x2 r1721, r1712, r1718;
}
{
sub.f16x2 r1724, r1431, r1432;
}
{
mul.f16x2 r1727, r1724, r1426;
}
{
sub.f16x2 r1730, r1437, r1438;
}
{
mul.f16x2 r1733, r1730, r1429;
}
{
add.f16x2 r1736, r1727, r1733;
}
{
sub.f16x2 r1739, r1721, r1736;
}
mul.wide.u32 rd8, r2218, 274877907;
shr.u64 rd9, rd8, 35;
cvt.u32.u64 r2238, rd9;
cvt.rn.f32.u32 f118, r2238;
mul.f32 f119, f118, 0f3E80ADFD;
cos.approx.f32 f85, f119;
sin.approx.f32 f120, f119;
neg.f32 f86, f120;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f85;
cvt.rn.f16.f32 high, f86;
mov.b32 r1742, {low, high};
}
mul.lo.s32 r2239, r2238, 125;
sub.s32 r2240, r2218, r2239;
{
.reg .f16 low, high;
mov.b32 {low, high}, r1742;
mov.b32 r1745, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1742;
mov.b32 r1747, {high, high};
}
{
mul.f16x2 r1749, r1631, r1747;
}
{
fma.rn.f16x2 r1752, r1487, r1745, r1749;
}
{
mul.f16x2 r1756, r1487, r1747;
}
{
neg.f16x2 r1759, r1756;
}
{
fma.rn.f16x2 r1761, r1631, r1745, r1759;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1742;
mov.b32 r1765, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1742;
mov.b32 r1767, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1769, {low, high};
}
{
mul.f16x2 r1770, r1767, r1769;
}
{
mul.f16x2 r1773, r1742, r1765;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1742;
mov.b32 r1776, {high, low};
}
{
fma.rn.f16x2 r1778, r1770, r1776, r1773;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1778;
mov.b32 r1782, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1778;
mov.b32 r1784, {high, high};
}
{
mul.f16x2 r1786, r1703, r1784;
}
{
fma.rn.f16x2 r1789, r1559, r1782, r1786;
}
{
mul.f16x2 r1793, r1559, r1784;
}
{
neg.f16x2 r1796, r1793;
}
{
fma.rn.f16x2 r1798, r1703, r1782, r1796;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1742;
mov.b32 r1802, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1742;
mov.b32 r1804, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1806, {low, high};
}
{
mul.f16x2 r1807, r1804, r1806;
}
{
mul.f16x2 r1810, r1778, r1802;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1778;
mov.b32 r1813, {high, low};
}
{
fma.rn.f16x2 r1815, r1807, r1813, r1810;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1815;
mov.b32 r1819, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1815;
mov.b32 r1821, {high, high};
}
{
mul.f16x2 r1823, r1739, r1821;
}
{
fma.rn.f16x2 r1826, r1595, r1819, r1823;
}
{
mul.f16x2 r1830, r1595, r1821;
}
{
neg.f16x2 r1833, r1830;
}
{
fma.rn.f16x2 r1835, r1739, r1819, r1833;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1742;
mov.b32 r1839, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1742;
mov.b32 r1841, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1843, {low, high};
}
{
mul.f16x2 r1844, r1841, r1843;
}
{
mul.f16x2 r1847, r1815, r1839;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1815;
mov.b32 r1850, {high, low};
}
{
fma.rn.f16x2 r1852, r1844, r1850, r1847;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1852;
mov.b32 r1856, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1852;
mov.b32 r1858, {high, high};
}
{
mul.f16x2 r1860, r1667, r1858;
}
{
fma.rn.f16x2 r1863, r1523, r1856, r1860;
}
{
mul.f16x2 r1867, r1523, r1858;
}
{
neg.f16x2 r1870, r1867;
}
{
fma.rn.f16x2 r1872, r1667, r1856, r1870;
}
shl.b32 r2241, r2240, 3;
add.s32 r2242, r2222, r2241;
barrier.sync 0;
mad.lo.s32 r2243, r2238, 5000, r2242;
st.shared.u32 [r2243], r1439;
st.shared.u32 [r2243+4], r1451;
st.shared.u32 [r2243+1000], r1752;
st.shared.u32 [r2243+1004], r1761;
st.shared.u32 [r2243+2000], r1789;
st.shared.u32 [r2243+2004], r1798;
st.shared.u32 [r2243+3000], r1826;
st.shared.u32 [r2243+3004], r1835;
st.shared.u32 [r2243+4000], r1863;
st.shared.u32 [r2243+4004], r1872;
barrier.sync 0;
ld.shared.u32 r1907, [r2225];
ld.shared.u32 r1919, [r2225+4];
ld.shared.u32 r1904, [r2225+5000];
ld.shared.u32 r1916, [r2225+5004];
ld.shared.u32 r1910, [r2225+10000];
ld.shared.u32 r1922, [r2225+10004];
ld.shared.u32 r1911, [r2225+15000];
ld.shared.u32 r1923, [r2225+15004];
ld.shared.u32 r1905, [r2225+20000];
ld.shared.u32 r1917, [r2225+20004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1893, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1894, {low, high};
}
{
neg.f16x2 r1895, r1894;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r1897, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r1898, {low, high};
}
{
neg.f16x2 r1899, r1898;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1901, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1902, {low, high};
}
{
add.f16x2 r1903, r1904, r1905;
}
{
add.f16x2 r1906, r1907, r1903;
}
{
add.f16x2 r1909, r1910, r1911;
}
{
add.f16x2 %0, r1906, r1909;
}
{
add.f16x2 r1915, r1916, r1917;
}
{
add.f16x2 r1918, r1919, r1915;
}
{
add.f16x2 r1921, r1922, r1923;
}
{
add.f16x2 %1, r1918, r1921;
}
{
add.f16x2 r1927, r1904, r1905;
}
{
mul.f16x2 r1930, r1927, r1893;
}
{
add.f16x2 r1933, r1907, r1930;
}
{
add.f16x2 r1936, r1910, r1911;
}
{
mul.f16x2 r1939, r1936, r1897;
}
{
add.f16x2 r1942, r1933, r1939;
}
{
sub.f16x2 r1945, r1916, r1917;
}
{
mul.f16x2 r1948, r1945, r1895;
}
{
sub.f16x2 r1951, r1922, r1923;
}
{
mul.f16x2 r1954, r1951, r1899;
}
{
add.f16x2 r1957, r1948, r1954;
}
{
sub.f16x2 %2, r1942, r1957;
}
{
add.f16x2 r1963, r1904, r1905;
}
{
mul.f16x2 r1966, r1963, r1893;
}
{
add.f16x2 r1969, r1907, r1966;
}
{
add.f16x2 r1972, r1910, r1911;
}
{
mul.f16x2 r1975, r1972, r1897;
}
{
add.f16x2 r1978, r1969, r1975;
}
{
sub.f16x2 r1981, r1916, r1917;
}
{
mul.f16x2 r1984, r1981, r1895;
}
{
sub.f16x2 r1987, r1922, r1923;
}
{
mul.f16x2 r1990, r1987, r1899;
}
{
add.f16x2 r1993, r1984, r1990;
}
{
add.f16x2 %8, r1978, r1993;
}
{
add.f16x2 r1999, r1904, r1905;
}
{
mul.f16x2 r2002, r1999, r1897;
}
{
add.f16x2 r2005, r1907, r2002;
}
{
add.f16x2 r2008, r1910, r1911;
}
{
mul.f16x2 r2011, r2008, r1901;
}
{
add.f16x2 r2014, r2005, r2011;
}
{
sub.f16x2 r2017, r1916, r1917;
}
{
mul.f16x2 r2020, r2017, r1899;
}
{
sub.f16x2 r2023, r1922, r1923;
}
{
mul.f16x2 r2026, r2023, r1902;
}
{
add.f16x2 r2029, r2020, r2026;
}
{
sub.f16x2 %4, r2014, r2029;
}
{
add.f16x2 r2035, r1904, r1905;
}
{
mul.f16x2 r2038, r2035, r1897;
}
{
add.f16x2 r2041, r1907, r2038;
}
{
add.f16x2 r2044, r1910, r1911;
}
{
mul.f16x2 r2047, r2044, r1901;
}
{
add.f16x2 r2050, r2041, r2047;
}
{
sub.f16x2 r2053, r1916, r1917;
}
{
mul.f16x2 r2056, r2053, r1899;
}
{
sub.f16x2 r2059, r1922, r1923;
}
{
mul.f16x2 r2062, r2059, r1902;
}
{
add.f16x2 r2065, r2056, r2062;
}
{
add.f16x2 %6, r2050, r2065;
}
{
add.f16x2 r2071, r1916, r1917;
}
{
mul.f16x2 r2074, r2071, r1893;
}
{
add.f16x2 r2077, r1919, r2074;
}
{
add.f16x2 r2080, r1922, r1923;
}
{
mul.f16x2 r2083, r2080, r1897;
}
{
add.f16x2 r2086, r2077, r2083;
}
{
sub.f16x2 r2089, r1904, r1905;
}
{
mul.f16x2 r2092, r2089, r1895;
}
{
sub.f16x2 r2095, r1910, r1911;
}
{
mul.f16x2 r2098, r2095, r1899;
}
{
add.f16x2 r2101, r2092, r2098;
}
{
add.f16x2 %3, r2086, r2101;
}
{
add.f16x2 r2107, r1916, r1917;
}
{
mul.f16x2 r2110, r2107, r1893;
}
{
add.f16x2 r2113, r1919, r2110;
}
{
add.f16x2 r2116, r1922, r1923;
}
{
mul.f16x2 r2119, r2116, r1897;
}
{
add.f16x2 r2122, r2113, r2119;
}
{
sub.f16x2 r2125, r1904, r1905;
}
{
mul.f16x2 r2128, r2125, r1895;
}
{
sub.f16x2 r2131, r1910, r1911;
}
{
mul.f16x2 r2134, r2131, r1899;
}
{
add.f16x2 r2137, r2128, r2134;
}
{
sub.f16x2 %9, r2122, r2137;
}
{
add.f16x2 r2143, r1916, r1917;
}
{
mul.f16x2 r2146, r2143, r1897;
}
{
add.f16x2 r2149, r1919, r2146;
}
{
add.f16x2 r2152, r1922, r1923;
}
{
mul.f16x2 r2155, r2152, r1901;
}
{
add.f16x2 r2158, r2149, r2155;
}
{
sub.f16x2 r2161, r1904, r1905;
}
{
mul.f16x2 r2164, r2161, r1899;
}
{
sub.f16x2 r2167, r1910, r1911;
}
{
mul.f16x2 r2170, r2167, r1902;
}
{
add.f16x2 r2173, r2164, r2170;
}
{
add.f16x2 %5, r2158, r2173;
}
{
add.f16x2 r2179, r1916, r1917;
}
{
mul.f16x2 r2182, r2179, r1897;
}
{
add.f16x2 r2185, r1919, r2182;
}
{
add.f16x2 r2188, r1922, r1923;
}
{
mul.f16x2 r2191, r2188, r1901;
}
{
add.f16x2 r2194, r2185, r2191;
}
{
sub.f16x2 r2197, r1904, r1905;
}
{
mul.f16x2 r2200, r2197, r1899;
}
{
sub.f16x2 r2203, r1910, r1911;
}
{
mul.f16x2 r2206, r2203, r1902;
}
{
add.f16x2 r2209, r2200, r2206;
}
{
sub.f16x2 %7, r2194, r2209;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1117, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<121>;
.reg .b32 r<2244>;
.reg .b64 rd<10>;
mov.u32 r2215, %tid.y;
mov.u32 r2216, %10;
mad.lo.s32 r2217, r2215, 12500, r2216;
mov.u32 r2218, %tid.x;
mov.f32 f106, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1, {low, high};
}
mov.f32 f108, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r2, {low, high};
}
{
neg.f16x2 r3, r2;
}
mov.f32 f102, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r5, {low, high};
}
mov.f32 f104, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r9, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r10, {low, high};
}
{
add.f16x2 r11, %13, %19;
}
{
add.f16x2 r14, %11, r11;
}
{
add.f16x2 r17, %15, %17;
}
{
add.f16x2 r20, r14, r17;
}
{
add.f16x2 r23, %14, %20;
}
{
add.f16x2 r26, %12, r23;
}
{
add.f16x2 r29, %16, %18;
}
{
add.f16x2 r32, r26, r29;
}
{
add.f16x2 r35, %13, %19;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %11, r38;
}
{
add.f16x2 r44, %15, %17;
}
{
mul.f16x2 r47, r44, r5;
}
{
add.f16x2 r50, r41, r47;
}
{
sub.f16x2 r53, %14, %20;
}
{
mul.f16x2 r56, r53, r3;
}
{
sub.f16x2 r59, %16, %18;
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
add.f16x2 r71, %13, %19;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %11, r74;
}
{
add.f16x2 r80, %15, %17;
}
{
mul.f16x2 r83, r80, r5;
}
{
add.f16x2 r86, r77, r83;
}
{
sub.f16x2 r89, %14, %20;
}
{
mul.f16x2 r92, r89, r3;
}
{
sub.f16x2 r95, %16, %18;
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
add.f16x2 r107, %13, %19;
}
{
mul.f16x2 r110, r107, r5;
}
{
add.f16x2 r113, %11, r110;
}
{
add.f16x2 r116, %15, %17;
}
{
mul.f16x2 r119, r116, r9;
}
{
add.f16x2 r122, r113, r119;
}
{
sub.f16x2 r125, %14, %20;
}
{
mul.f16x2 r128, r125, r7;
}
{
sub.f16x2 r131, %16, %18;
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
add.f16x2 r143, %13, %19;
}
{
mul.f16x2 r146, r143, r5;
}
{
add.f16x2 r149, %11, r146;
}
{
add.f16x2 r152, %15, %17;
}
{
mul.f16x2 r155, r152, r9;
}
{
add.f16x2 r158, r149, r155;
}
{
sub.f16x2 r161, %14, %20;
}
{
mul.f16x2 r164, r161, r7;
}
{
sub.f16x2 r167, %16, %18;
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
add.f16x2 r179, %14, %20;
}
{
mul.f16x2 r182, r179, r1;
}
{
add.f16x2 r185, %12, r182;
}
{
add.f16x2 r188, %16, %18;
}
{
mul.f16x2 r191, r188, r5;
}
{
add.f16x2 r194, r185, r191;
}
{
sub.f16x2 r197, %13, %19;
}
{
mul.f16x2 r200, r197, r3;
}
{
sub.f16x2 r203, %15, %17;
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
add.f16x2 r215, %14, %20;
}
{
mul.f16x2 r218, r215, r1;
}
{
add.f16x2 r221, %12, r218;
}
{
add.f16x2 r224, %16, %18;
}
{
mul.f16x2 r227, r224, r5;
}
{
add.f16x2 r230, r221, r227;
}
{
sub.f16x2 r233, %13, %19;
}
{
mul.f16x2 r236, r233, r3;
}
{
sub.f16x2 r239, %15, %17;
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
add.f16x2 r251, %14, %20;
}
{
mul.f16x2 r254, r251, r5;
}
{
add.f16x2 r257, %12, r254;
}
{
add.f16x2 r260, %16, %18;
}
{
mul.f16x2 r263, r260, r9;
}
{
add.f16x2 r266, r257, r263;
}
{
sub.f16x2 r269, %13, %19;
}
{
mul.f16x2 r272, r269, r7;
}
{
sub.f16x2 r275, %15, %17;
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
add.f16x2 r287, %14, %20;
}
{
mul.f16x2 r290, r287, r5;
}
{
add.f16x2 r293, %12, r290;
}
{
add.f16x2 r296, %16, %18;
}
{
mul.f16x2 r299, r296, r9;
}
{
add.f16x2 r302, r293, r299;
}
{
sub.f16x2 r305, %13, %19;
}
{
mul.f16x2 r308, r305, r7;
}
{
sub.f16x2 r311, %15, %17;
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
mul.wide.u32 rd2, r2218, -776530087;
shr.u64 rd3, rd2, 41;
cvt.u32.u64 r2219, rd3;
mul.lo.s32 r2220, r2219, 625;
sub.s32 r2221, r2218, r2220;
cvt.rn.f32.u32 f109, r2221;
mul.f32 f110, f109, 0f3B03C498;
cos.approx.f32 f13, f110;
sin.approx.f32 f111, f110;
neg.f32 f14, f111;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f13;
cvt.rn.f16.f32 high, f14;
mov.b32 r323, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r323;
mov.b32 r326, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r323;
mov.b32 r328, {high, high};
}
{
mul.f16x2 r330, r212, r328;
}
{
fma.rn.f16x2 r333, r68, r326, r330;
}
{
mul.f16x2 r337, r68, r328;
}
{
neg.f16x2 r340, r337;
}
{
fma.rn.f16x2 r342, r212, r326, r340;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r323;
mov.b32 r346, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r323;
mov.b32 r348, {high, high};
}
mov.f32 f93, 0fBF800000;
mov.f32 f94, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r350, {low, high};
}
{
mul.f16x2 r351, r348, r350;
}
{
mul.f16x2 r354, r323, r346;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r323;
mov.b32 r357, {high, low};
}
{
fma.rn.f16x2 r359, r351, r357, r354;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r359;
mov.b32 r363, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r359;
mov.b32 r365, {high, high};
}
{
mul.f16x2 r367, r284, r365;
}
{
fma.rn.f16x2 r370, r140, r363, r367;
}
{
mul.f16x2 r374, r140, r365;
}
{
neg.f16x2 r377, r374;
}
{
fma.rn.f16x2 r379, r284, r363, r377;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r323;
mov.b32 r383, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r323;
mov.b32 r385, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r387, {low, high};
}
{
mul.f16x2 r388, r385, r387;
}
{
mul.f16x2 r391, r359, r383;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r359;
mov.b32 r394, {high, low};
}
{
fma.rn.f16x2 r396, r388, r394, r391;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r396;
mov.b32 r400, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r396;
mov.b32 r402, {high, high};
}
{
mul.f16x2 r404, r320, r402;
}
{
fma.rn.f16x2 r407, r176, r400, r404;
}
{
mul.f16x2 r411, r176, r402;
}
{
neg.f16x2 r414, r411;
}
{
fma.rn.f16x2 r416, r320, r400, r414;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r323;
mov.b32 r420, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r323;
mov.b32 r422, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r424, {low, high};
}
{
mul.f16x2 r425, r422, r424;
}
{
mul.f16x2 r428, r396, r420;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r396;
mov.b32 r431, {high, low};
}
{
fma.rn.f16x2 r433, r425, r431, r428;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r433;
mov.b32 r437, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r433;
mov.b32 r439, {high, high};
}
{
mul.f16x2 r441, r248, r439;
}
{
fma.rn.f16x2 r444, r104, r437, r441;
}
{
mul.f16x2 r448, r104, r439;
}
{
neg.f16x2 r451, r448;
}
{
fma.rn.f16x2 r453, r248, r437, r451;
}
mad.lo.s32 r2222, r2219, 12500, r2217;
barrier.sync 0;
mad.lo.s32 r2223, r2221, 20, r2222;
st.shared.u32 [r2223], r20;
st.shared.u32 [r2223+4], r333;
st.shared.u32 [r2223+8], r370;
st.shared.u32 [r2223+12], r407;
st.shared.u32 [r2223+16], r444;
barrier.sync 0;
shl.b32 r2224, r2221, 4;
sub.s32 r2225, r2223, r2224;
ld.shared.u32 r488, [r2225];
ld.shared.u32 r485, [r2225+2500];
ld.shared.u32 r491, [r2225+5000];
ld.shared.u32 r492, [r2225+7500];
ld.shared.u32 r486, [r2225+10000];
barrier.sync 0;
st.shared.u32 [r2223], r32;
st.shared.u32 [r2223+4], r342;
st.shared.u32 [r2223+8], r379;
st.shared.u32 [r2223+12], r416;
st.shared.u32 [r2223+16], r453;
barrier.sync 0;
ld.shared.u32 r500, [r2225];
ld.shared.u32 r497, [r2225+2500];
ld.shared.u32 r503, [r2225+5000];
ld.shared.u32 r504, [r2225+7500];
ld.shared.u32 r498, [r2225+10000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r474, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r475, {low, high};
}
{
neg.f16x2 r476, r475;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r478, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r479, {low, high};
}
{
neg.f16x2 r480, r479;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r482, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r483, {low, high};
}
{
add.f16x2 r484, r485, r486;
}
{
add.f16x2 r487, r488, r484;
}
{
add.f16x2 r490, r491, r492;
}
{
add.f16x2 r493, r487, r490;
}
{
add.f16x2 r496, r497, r498;
}
{
add.f16x2 r499, r500, r496;
}
{
add.f16x2 r502, r503, r504;
}
{
add.f16x2 r505, r499, r502;
}
{
add.f16x2 r508, r485, r486;
}
{
mul.f16x2 r511, r508, r474;
}
{
add.f16x2 r514, r488, r511;
}
{
add.f16x2 r517, r491, r492;
}
{
mul.f16x2 r520, r517, r478;
}
{
add.f16x2 r523, r514, r520;
}
{
sub.f16x2 r526, r497, r498;
}
{
mul.f16x2 r529, r526, r476;
}
{
sub.f16x2 r532, r503, r504;
}
{
mul.f16x2 r535, r532, r480;
}
{
add.f16x2 r538, r529, r535;
}
{
sub.f16x2 r541, r523, r538;
}
{
add.f16x2 r544, r485, r486;
}
{
mul.f16x2 r547, r544, r474;
}
{
add.f16x2 r550, r488, r547;
}
{
add.f16x2 r553, r491, r492;
}
{
mul.f16x2 r556, r553, r478;
}
{
add.f16x2 r559, r550, r556;
}
{
sub.f16x2 r562, r497, r498;
}
{
mul.f16x2 r565, r562, r476;
}
{
sub.f16x2 r568, r503, r504;
}
{
mul.f16x2 r571, r568, r480;
}
{
add.f16x2 r574, r565, r571;
}
{
add.f16x2 r577, r559, r574;
}
{
add.f16x2 r580, r485, r486;
}
{
mul.f16x2 r583, r580, r478;
}
{
add.f16x2 r586, r488, r583;
}
{
add.f16x2 r589, r491, r492;
}
{
mul.f16x2 r592, r589, r482;
}
{
add.f16x2 r595, r586, r592;
}
{
sub.f16x2 r598, r497, r498;
}
{
mul.f16x2 r601, r598, r480;
}
{
sub.f16x2 r604, r503, r504;
}
{
mul.f16x2 r607, r604, r483;
}
{
add.f16x2 r610, r601, r607;
}
{
sub.f16x2 r613, r595, r610;
}
{
add.f16x2 r616, r485, r486;
}
{
mul.f16x2 r619, r616, r478;
}
{
add.f16x2 r622, r488, r619;
}
{
add.f16x2 r625, r491, r492;
}
{
mul.f16x2 r628, r625, r482;
}
{
add.f16x2 r631, r622, r628;
}
{
sub.f16x2 r634, r497, r498;
}
{
mul.f16x2 r637, r634, r480;
}
{
sub.f16x2 r640, r503, r504;
}
{
mul.f16x2 r643, r640, r483;
}
{
add.f16x2 r646, r637, r643;
}
{
add.f16x2 r649, r631, r646;
}
{
add.f16x2 r652, r497, r498;
}
{
mul.f16x2 r655, r652, r474;
}
{
add.f16x2 r658, r500, r655;
}
{
add.f16x2 r661, r503, r504;
}
{
mul.f16x2 r664, r661, r478;
}
{
add.f16x2 r667, r658, r664;
}
{
sub.f16x2 r670, r485, r486;
}
{
mul.f16x2 r673, r670, r476;
}
{
sub.f16x2 r676, r491, r492;
}
{
mul.f16x2 r679, r676, r480;
}
{
add.f16x2 r682, r673, r679;
}
{
add.f16x2 r685, r667, r682;
}
{
add.f16x2 r688, r497, r498;
}
{
mul.f16x2 r691, r688, r474;
}
{
add.f16x2 r694, r500, r691;
}
{
add.f16x2 r697, r503, r504;
}
{
mul.f16x2 r700, r697, r478;
}
{
add.f16x2 r703, r694, r700;
}
{
sub.f16x2 r706, r485, r486;
}
{
mul.f16x2 r709, r706, r476;
}
{
sub.f16x2 r712, r491, r492;
}
{
mul.f16x2 r715, r712, r480;
}
{
add.f16x2 r718, r709, r715;
}
{
sub.f16x2 r721, r703, r718;
}
{
add.f16x2 r724, r497, r498;
}
{
mul.f16x2 r727, r724, r478;
}
{
add.f16x2 r730, r500, r727;
}
{
add.f16x2 r733, r503, r504;
}
{
mul.f16x2 r736, r733, r482;
}
{
add.f16x2 r739, r730, r736;
}
{
sub.f16x2 r742, r485, r486;
}
{
mul.f16x2 r745, r742, r480;
}
{
sub.f16x2 r748, r491, r492;
}
{
mul.f16x2 r751, r748, r483;
}
{
add.f16x2 r754, r745, r751;
}
{
add.f16x2 r757, r739, r754;
}
{
add.f16x2 r760, r497, r498;
}
{
mul.f16x2 r763, r760, r478;
}
{
add.f16x2 r766, r500, r763;
}
{
add.f16x2 r769, r503, r504;
}
{
mul.f16x2 r772, r769, r482;
}
{
add.f16x2 r775, r766, r772;
}
{
sub.f16x2 r778, r485, r486;
}
{
mul.f16x2 r781, r778, r480;
}
{
sub.f16x2 r784, r491, r492;
}
{
mul.f16x2 r787, r784, r483;
}
{
add.f16x2 r790, r781, r787;
}
{
sub.f16x2 r793, r775, r790;
}
mul.wide.u32 rd4, r2221, -858993459;
shr.u64 rd5, rd4, 34;
cvt.u32.u64 r2226, rd5;
cvt.rn.f32.u32 f112, r2226;
mul.f32 f113, f112, 0f3C24B5BE;
cos.approx.f32 f37, f113;
sin.approx.f32 f114, f113;
neg.f32 f38, f114;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
mov.b32 r796, {low, high};
}
mul.lo.s32 r2227, r2226, 5;
sub.s32 r2228, r2221, r2227;
{
.reg .f16 low, high;
mov.b32 {low, high}, r796;
mov.b32 r799, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r796;
mov.b32 r801, {high, high};
}
{
mul.f16x2 r803, r685, r801;
}
{
fma.rn.f16x2 r806, r541, r799, r803;
}
{
mul.f16x2 r810, r541, r801;
}
{
neg.f16x2 r813, r810;
}
{
fma.rn.f16x2 r815, r685, r799, r813;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r796;
mov.b32 r819, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r796;
mov.b32 r821, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r823, {low, high};
}
{
mul.f16x2 r824, r821, r823;
}
{
mul.f16x2 r827, r796, r819;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r796;
mov.b32 r830, {high, low};
}
{
fma.rn.f16x2 r832, r824, r830, r827;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r832;
mov.b32 r836, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r832;
mov.b32 r838, {high, high};
}
{
mul.f16x2 r840, r757, r838;
}
{
fma.rn.f16x2 r843, r613, r836, r840;
}
{
mul.f16x2 r847, r613, r838;
}
{
neg.f16x2 r850, r847;
}
{
fma.rn.f16x2 r852, r757, r836, r850;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r796;
mov.b32 r856, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r796;
mov.b32 r858, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r860, {low, high};
}
{
mul.f16x2 r861, r858, r860;
}
{
mul.f16x2 r864, r832, r856;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r832;
mov.b32 r867, {high, low};
}
{
fma.rn.f16x2 r869, r861, r867, r864;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r869;
mov.b32 r873, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r869;
mov.b32 r875, {high, high};
}
{
mul.f16x2 r877, r793, r875;
}
{
fma.rn.f16x2 r880, r649, r873, r877;
}
{
mul.f16x2 r884, r649, r875;
}
{
neg.f16x2 r887, r884;
}
{
fma.rn.f16x2 r889, r793, r873, r887;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r796;
mov.b32 r893, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r796;
mov.b32 r895, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r897, {low, high};
}
{
mul.f16x2 r898, r895, r897;
}
{
mul.f16x2 r901, r869, r893;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r869;
mov.b32 r904, {high, low};
}
{
fma.rn.f16x2 r906, r898, r904, r901;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r906;
mov.b32 r910, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r906;
mov.b32 r912, {high, high};
}
{
mul.f16x2 r914, r721, r912;
}
{
fma.rn.f16x2 r917, r577, r910, r914;
}
{
mul.f16x2 r921, r577, r912;
}
{
neg.f16x2 r924, r921;
}
{
fma.rn.f16x2 r926, r721, r910, r924;
}
shl.b32 r2229, r2228, 2;
add.s32 r2230, r2222, r2229;
barrier.sync 0;
mad.lo.s32 r2231, r2226, 100, r2230;
st.shared.u32 [r2231], r493;
st.shared.u32 [r2231+20], r806;
st.shared.u32 [r2231+40], r843;
st.shared.u32 [r2231+60], r880;
st.shared.u32 [r2231+80], r917;
barrier.sync 0;
ld.shared.u32 r961, [r2225];
ld.shared.u32 r958, [r2225+2500];
ld.shared.u32 r964, [r2225+5000];
ld.shared.u32 r965, [r2225+7500];
ld.shared.u32 r959, [r2225+10000];
barrier.sync 0;
st.shared.u32 [r2231], r505;
st.shared.u32 [r2231+20], r815;
st.shared.u32 [r2231+40], r852;
st.shared.u32 [r2231+60], r889;
st.shared.u32 [r2231+80], r926;
barrier.sync 0;
ld.shared.u32 r973, [r2225];
ld.shared.u32 r970, [r2225+2500];
ld.shared.u32 r976, [r2225+5000];
ld.shared.u32 r977, [r2225+7500];
ld.shared.u32 r971, [r2225+10000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r947, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r948, {low, high};
}
{
neg.f16x2 r949, r948;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r951, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r952, {low, high};
}
{
neg.f16x2 r953, r952;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r955, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r956, {low, high};
}
{
add.f16x2 r957, r958, r959;
}
{
add.f16x2 r960, r961, r957;
}
{
add.f16x2 r963, r964, r965;
}
{
add.f16x2 r966, r960, r963;
}
{
add.f16x2 r969, r970, r971;
}
{
add.f16x2 r972, r973, r969;
}
{
add.f16x2 r975, r976, r977;
}
{
add.f16x2 r978, r972, r975;
}
{
add.f16x2 r981, r958, r959;
}
{
mul.f16x2 r984, r981, r947;
}
{
add.f16x2 r987, r961, r984;
}
{
add.f16x2 r990, r964, r965;
}
{
mul.f16x2 r993, r990, r951;
}
{
add.f16x2 r996, r987, r993;
}
{
sub.f16x2 r999, r970, r971;
}
{
mul.f16x2 r1002, r999, r949;
}
{
sub.f16x2 r1005, r976, r977;
}
{
mul.f16x2 r1008, r1005, r953;
}
{
add.f16x2 r1011, r1002, r1008;
}
{
sub.f16x2 r1014, r996, r1011;
}
{
add.f16x2 r1017, r958, r959;
}
{
mul.f16x2 r1020, r1017, r947;
}
{
add.f16x2 r1023, r961, r1020;
}
{
add.f16x2 r1026, r964, r965;
}
{
mul.f16x2 r1029, r1026, r951;
}
{
add.f16x2 r1032, r1023, r1029;
}
{
sub.f16x2 r1035, r970, r971;
}
{
mul.f16x2 r1038, r1035, r949;
}
{
sub.f16x2 r1041, r976, r977;
}
{
mul.f16x2 r1044, r1041, r953;
}
{
add.f16x2 r1047, r1038, r1044;
}
{
add.f16x2 r1050, r1032, r1047;
}
{
add.f16x2 r1053, r958, r959;
}
{
mul.f16x2 r1056, r1053, r951;
}
{
add.f16x2 r1059, r961, r1056;
}
{
add.f16x2 r1062, r964, r965;
}
{
mul.f16x2 r1065, r1062, r955;
}
{
add.f16x2 r1068, r1059, r1065;
}
{
sub.f16x2 r1071, r970, r971;
}
{
mul.f16x2 r1074, r1071, r953;
}
{
sub.f16x2 r1077, r976, r977;
}
{
mul.f16x2 r1080, r1077, r956;
}
{
add.f16x2 r1083, r1074, r1080;
}
{
sub.f16x2 r1086, r1068, r1083;
}
{
add.f16x2 r1089, r958, r959;
}
{
mul.f16x2 r1092, r1089, r951;
}
{
add.f16x2 r1095, r961, r1092;
}
{
add.f16x2 r1098, r964, r965;
}
{
mul.f16x2 r1101, r1098, r955;
}
{
add.f16x2 r1104, r1095, r1101;
}
{
sub.f16x2 r1107, r970, r971;
}
{
mul.f16x2 r1110, r1107, r953;
}
{
sub.f16x2 r1113, r976, r977;
}
{
mul.f16x2 r1116, r1113, r956;
}
{
add.f16x2 r1119, r1110, r1116;
}
{
add.f16x2 r1122, r1104, r1119;
}
{
add.f16x2 r1125, r970, r971;
}
{
mul.f16x2 r1128, r1125, r947;
}
{
add.f16x2 r1131, r973, r1128;
}
{
add.f16x2 r1134, r976, r977;
}
{
mul.f16x2 r1137, r1134, r951;
}
{
add.f16x2 r1140, r1131, r1137;
}
{
sub.f16x2 r1143, r958, r959;
}
{
mul.f16x2 r1146, r1143, r949;
}
{
sub.f16x2 r1149, r964, r965;
}
{
mul.f16x2 r1152, r1149, r953;
}
{
add.f16x2 r1155, r1146, r1152;
}
{
add.f16x2 r1158, r1140, r1155;
}
{
add.f16x2 r1161, r970, r971;
}
{
mul.f16x2 r1164, r1161, r947;
}
{
add.f16x2 r1167, r973, r1164;
}
{
add.f16x2 r1170, r976, r977;
}
{
mul.f16x2 r1173, r1170, r951;
}
{
add.f16x2 r1176, r1167, r1173;
}
{
sub.f16x2 r1179, r958, r959;
}
{
mul.f16x2 r1182, r1179, r949;
}
{
sub.f16x2 r1185, r964, r965;
}
{
mul.f16x2 r1188, r1185, r953;
}
{
add.f16x2 r1191, r1182, r1188;
}
{
sub.f16x2 r1194, r1176, r1191;
}
{
add.f16x2 r1197, r970, r971;
}
{
mul.f16x2 r1200, r1197, r951;
}
{
add.f16x2 r1203, r973, r1200;
}
{
add.f16x2 r1206, r976, r977;
}
{
mul.f16x2 r1209, r1206, r955;
}
{
add.f16x2 r1212, r1203, r1209;
}
{
sub.f16x2 r1215, r958, r959;
}
{
mul.f16x2 r1218, r1215, r953;
}
{
sub.f16x2 r1221, r964, r965;
}
{
mul.f16x2 r1224, r1221, r956;
}
{
add.f16x2 r1227, r1218, r1224;
}
{
add.f16x2 r1230, r1212, r1227;
}
{
add.f16x2 r1233, r970, r971;
}
{
mul.f16x2 r1236, r1233, r951;
}
{
add.f16x2 r1239, r973, r1236;
}
{
add.f16x2 r1242, r976, r977;
}
{
mul.f16x2 r1245, r1242, r955;
}
{
add.f16x2 r1248, r1239, r1245;
}
{
sub.f16x2 r1251, r958, r959;
}
{
mul.f16x2 r1254, r1251, r953;
}
{
sub.f16x2 r1257, r964, r965;
}
{
mul.f16x2 r1260, r1257, r956;
}
{
add.f16x2 r1263, r1254, r1260;
}
{
sub.f16x2 r1266, r1248, r1263;
}
mul.wide.u32 rd6, r2221, 1374389535;
shr.u64 rd7, rd6, 35;
cvt.u32.u64 r2232, rd7;
cvt.rn.f32.u32 f115, r2232;
mul.f32 f116, f115, 0f3D4DE32E;
cos.approx.f32 f61, f116;
sin.approx.f32 f117, f116;
neg.f32 f62, f117;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f61;
cvt.rn.f16.f32 high, f62;
mov.b32 r1269, {low, high};
}
mul.lo.s32 r2233, r2232, 25;
sub.s32 r2234, r2221, r2233;
{
.reg .f16 low, high;
mov.b32 {low, high}, r1269;
mov.b32 r1272, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1269;
mov.b32 r1274, {high, high};
}
{
mul.f16x2 r1276, r1158, r1274;
}
{
fma.rn.f16x2 r1279, r1014, r1272, r1276;
}
{
mul.f16x2 r1283, r1014, r1274;
}
{
neg.f16x2 r1286, r1283;
}
{
fma.rn.f16x2 r1288, r1158, r1272, r1286;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1269;
mov.b32 r1292, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1269;
mov.b32 r1294, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1296, {low, high};
}
{
mul.f16x2 r1297, r1294, r1296;
}
{
mul.f16x2 r1300, r1269, r1292;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1269;
mov.b32 r1303, {high, low};
}
{
fma.rn.f16x2 r1305, r1297, r1303, r1300;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1305;
mov.b32 r1309, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1305;
mov.b32 r1311, {high, high};
}
{
mul.f16x2 r1313, r1230, r1311;
}
{
fma.rn.f16x2 r1316, r1086, r1309, r1313;
}
{
mul.f16x2 r1320, r1086, r1311;
}
{
neg.f16x2 r1323, r1320;
}
{
fma.rn.f16x2 r1325, r1230, r1309, r1323;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1269;
mov.b32 r1329, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1269;
mov.b32 r1331, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1333, {low, high};
}
{
mul.f16x2 r1334, r1331, r1333;
}
{
mul.f16x2 r1337, r1305, r1329;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1305;
mov.b32 r1340, {high, low};
}
{
fma.rn.f16x2 r1342, r1334, r1340, r1337;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1342;
mov.b32 r1346, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1342;
mov.b32 r1348, {high, high};
}
{
mul.f16x2 r1350, r1266, r1348;
}
{
fma.rn.f16x2 r1353, r1122, r1346, r1350;
}
{
mul.f16x2 r1357, r1122, r1348;
}
{
neg.f16x2 r1360, r1357;
}
{
fma.rn.f16x2 r1362, r1266, r1346, r1360;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1269;
mov.b32 r1366, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1269;
mov.b32 r1368, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1370, {low, high};
}
{
mul.f16x2 r1371, r1368, r1370;
}
{
mul.f16x2 r1374, r1342, r1366;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1342;
mov.b32 r1377, {high, low};
}
{
fma.rn.f16x2 r1379, r1371, r1377, r1374;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1379;
mov.b32 r1383, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1379;
mov.b32 r1385, {high, high};
}
{
mul.f16x2 r1387, r1194, r1385;
}
{
fma.rn.f16x2 r1390, r1050, r1383, r1387;
}
{
mul.f16x2 r1394, r1050, r1385;
}
{
neg.f16x2 r1397, r1394;
}
{
fma.rn.f16x2 r1399, r1194, r1383, r1397;
}
shl.b32 r2235, r2234, 2;
add.s32 r2236, r2222, r2235;
barrier.sync 0;
mad.lo.s32 r2237, r2232, 500, r2236;
st.shared.u32 [r2237], r966;
st.shared.u32 [r2237+100], r1279;
st.shared.u32 [r2237+200], r1316;
st.shared.u32 [r2237+300], r1353;
st.shared.u32 [r2237+400], r1390;
barrier.sync 0;
ld.shared.u32 r1434, [r2225];
ld.shared.u32 r1431, [r2225+2500];
ld.shared.u32 r1437, [r2225+5000];
ld.shared.u32 r1438, [r2225+7500];
ld.shared.u32 r1432, [r2225+10000];
barrier.sync 0;
st.shared.u32 [r2237], r978;
st.shared.u32 [r2237+100], r1288;
st.shared.u32 [r2237+200], r1325;
st.shared.u32 [r2237+300], r1362;
st.shared.u32 [r2237+400], r1399;
barrier.sync 0;
ld.shared.u32 r1446, [r2225];
ld.shared.u32 r1443, [r2225+2500];
ld.shared.u32 r1449, [r2225+5000];
ld.shared.u32 r1450, [r2225+7500];
ld.shared.u32 r1444, [r2225+10000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1420, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1421, {low, high};
}
{
neg.f16x2 r1422, r1421;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r1424, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r1425, {low, high};
}
{
neg.f16x2 r1426, r1425;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1428, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1429, {low, high};
}
{
add.f16x2 r1430, r1431, r1432;
}
{
add.f16x2 r1433, r1434, r1430;
}
{
add.f16x2 r1436, r1437, r1438;
}
{
add.f16x2 r1439, r1433, r1436;
}
{
add.f16x2 r1442, r1443, r1444;
}
{
add.f16x2 r1445, r1446, r1442;
}
{
add.f16x2 r1448, r1449, r1450;
}
{
add.f16x2 r1451, r1445, r1448;
}
{
add.f16x2 r1454, r1431, r1432;
}
{
mul.f16x2 r1457, r1454, r1420;
}
{
add.f16x2 r1460, r1434, r1457;
}
{
add.f16x2 r1463, r1437, r1438;
}
{
mul.f16x2 r1466, r1463, r1424;
}
{
add.f16x2 r1469, r1460, r1466;
}
{
sub.f16x2 r1472, r1443, r1444;
}
{
mul.f16x2 r1475, r1472, r1422;
}
{
sub.f16x2 r1478, r1449, r1450;
}
{
mul.f16x2 r1481, r1478, r1426;
}
{
add.f16x2 r1484, r1475, r1481;
}
{
sub.f16x2 r1487, r1469, r1484;
}
{
add.f16x2 r1490, r1431, r1432;
}
{
mul.f16x2 r1493, r1490, r1420;
}
{
add.f16x2 r1496, r1434, r1493;
}
{
add.f16x2 r1499, r1437, r1438;
}
{
mul.f16x2 r1502, r1499, r1424;
}
{
add.f16x2 r1505, r1496, r1502;
}
{
sub.f16x2 r1508, r1443, r1444;
}
{
mul.f16x2 r1511, r1508, r1422;
}
{
sub.f16x2 r1514, r1449, r1450;
}
{
mul.f16x2 r1517, r1514, r1426;
}
{
add.f16x2 r1520, r1511, r1517;
}
{
add.f16x2 r1523, r1505, r1520;
}
{
add.f16x2 r1526, r1431, r1432;
}
{
mul.f16x2 r1529, r1526, r1424;
}
{
add.f16x2 r1532, r1434, r1529;
}
{
add.f16x2 r1535, r1437, r1438;
}
{
mul.f16x2 r1538, r1535, r1428;
}
{
add.f16x2 r1541, r1532, r1538;
}
{
sub.f16x2 r1544, r1443, r1444;
}
{
mul.f16x2 r1547, r1544, r1426;
}
{
sub.f16x2 r1550, r1449, r1450;
}
{
mul.f16x2 r1553, r1550, r1429;
}
{
add.f16x2 r1556, r1547, r1553;
}
{
sub.f16x2 r1559, r1541, r1556;
}
{
add.f16x2 r1562, r1431, r1432;
}
{
mul.f16x2 r1565, r1562, r1424;
}
{
add.f16x2 r1568, r1434, r1565;
}
{
add.f16x2 r1571, r1437, r1438;
}
{
mul.f16x2 r1574, r1571, r1428;
}
{
add.f16x2 r1577, r1568, r1574;
}
{
sub.f16x2 r1580, r1443, r1444;
}
{
mul.f16x2 r1583, r1580, r1426;
}
{
sub.f16x2 r1586, r1449, r1450;
}
{
mul.f16x2 r1589, r1586, r1429;
}
{
add.f16x2 r1592, r1583, r1589;
}
{
add.f16x2 r1595, r1577, r1592;
}
{
add.f16x2 r1598, r1443, r1444;
}
{
mul.f16x2 r1601, r1598, r1420;
}
{
add.f16x2 r1604, r1446, r1601;
}
{
add.f16x2 r1607, r1449, r1450;
}
{
mul.f16x2 r1610, r1607, r1424;
}
{
add.f16x2 r1613, r1604, r1610;
}
{
sub.f16x2 r1616, r1431, r1432;
}
{
mul.f16x2 r1619, r1616, r1422;
}
{
sub.f16x2 r1622, r1437, r1438;
}
{
mul.f16x2 r1625, r1622, r1426;
}
{
add.f16x2 r1628, r1619, r1625;
}
{
add.f16x2 r1631, r1613, r1628;
}
{
add.f16x2 r1634, r1443, r1444;
}
{
mul.f16x2 r1637, r1634, r1420;
}
{
add.f16x2 r1640, r1446, r1637;
}
{
add.f16x2 r1643, r1449, r1450;
}
{
mul.f16x2 r1646, r1643, r1424;
}
{
add.f16x2 r1649, r1640, r1646;
}
{
sub.f16x2 r1652, r1431, r1432;
}
{
mul.f16x2 r1655, r1652, r1422;
}
{
sub.f16x2 r1658, r1437, r1438;
}
{
mul.f16x2 r1661, r1658, r1426;
}
{
add.f16x2 r1664, r1655, r1661;
}
{
sub.f16x2 r1667, r1649, r1664;
}
{
add.f16x2 r1670, r1443, r1444;
}
{
mul.f16x2 r1673, r1670, r1424;
}
{
add.f16x2 r1676, r1446, r1673;
}
{
add.f16x2 r1679, r1449, r1450;
}
{
mul.f16x2 r1682, r1679, r1428;
}
{
add.f16x2 r1685, r1676, r1682;
}
{
sub.f16x2 r1688, r1431, r1432;
}
{
mul.f16x2 r1691, r1688, r1426;
}
{
sub.f16x2 r1694, r1437, r1438;
}
{
mul.f16x2 r1697, r1694, r1429;
}
{
add.f16x2 r1700, r1691, r1697;
}
{
add.f16x2 r1703, r1685, r1700;
}
{
add.f16x2 r1706, r1443, r1444;
}
{
mul.f16x2 r1709, r1706, r1424;
}
{
add.f16x2 r1712, r1446, r1709;
}
{
add.f16x2 r1715, r1449, r1450;
}
{
mul.f16x2 r1718, r1715, r1428;
}
{
add.f16x2 r1721, r1712, r1718;
}
{
sub.f16x2 r1724, r1431, r1432;
}
{
mul.f16x2 r1727, r1724, r1426;
}
{
sub.f16x2 r1730, r1437, r1438;
}
{
mul.f16x2 r1733, r1730, r1429;
}
{
add.f16x2 r1736, r1727, r1733;
}
{
sub.f16x2 r1739, r1721, r1736;
}
mul.wide.u32 rd8, r2221, 274877907;
shr.u64 rd9, rd8, 35;
cvt.u32.u64 r2238, rd9;
cvt.rn.f32.u32 f118, r2238;
mul.f32 f119, f118, 0f3E80ADFD;
cos.approx.f32 f85, f119;
sin.approx.f32 f120, f119;
neg.f32 f86, f120;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f85;
cvt.rn.f16.f32 high, f86;
mov.b32 r1742, {low, high};
}
mul.lo.s32 r2239, r2238, 125;
sub.s32 r2240, r2221, r2239;
{
.reg .f16 low, high;
mov.b32 {low, high}, r1742;
mov.b32 r1745, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1742;
mov.b32 r1747, {high, high};
}
{
mul.f16x2 r1749, r1631, r1747;
}
{
fma.rn.f16x2 r1752, r1487, r1745, r1749;
}
{
mul.f16x2 r1756, r1487, r1747;
}
{
neg.f16x2 r1759, r1756;
}
{
fma.rn.f16x2 r1761, r1631, r1745, r1759;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1742;
mov.b32 r1765, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1742;
mov.b32 r1767, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1769, {low, high};
}
{
mul.f16x2 r1770, r1767, r1769;
}
{
mul.f16x2 r1773, r1742, r1765;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1742;
mov.b32 r1776, {high, low};
}
{
fma.rn.f16x2 r1778, r1770, r1776, r1773;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1778;
mov.b32 r1782, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1778;
mov.b32 r1784, {high, high};
}
{
mul.f16x2 r1786, r1703, r1784;
}
{
fma.rn.f16x2 r1789, r1559, r1782, r1786;
}
{
mul.f16x2 r1793, r1559, r1784;
}
{
neg.f16x2 r1796, r1793;
}
{
fma.rn.f16x2 r1798, r1703, r1782, r1796;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1742;
mov.b32 r1802, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1742;
mov.b32 r1804, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1806, {low, high};
}
{
mul.f16x2 r1807, r1804, r1806;
}
{
mul.f16x2 r1810, r1778, r1802;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1778;
mov.b32 r1813, {high, low};
}
{
fma.rn.f16x2 r1815, r1807, r1813, r1810;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1815;
mov.b32 r1819, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1815;
mov.b32 r1821, {high, high};
}
{
mul.f16x2 r1823, r1739, r1821;
}
{
fma.rn.f16x2 r1826, r1595, r1819, r1823;
}
{
mul.f16x2 r1830, r1595, r1821;
}
{
neg.f16x2 r1833, r1830;
}
{
fma.rn.f16x2 r1835, r1739, r1819, r1833;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1742;
mov.b32 r1839, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1742;
mov.b32 r1841, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1843, {low, high};
}
{
mul.f16x2 r1844, r1841, r1843;
}
{
mul.f16x2 r1847, r1815, r1839;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1815;
mov.b32 r1850, {high, low};
}
{
fma.rn.f16x2 r1852, r1844, r1850, r1847;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1852;
mov.b32 r1856, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1852;
mov.b32 r1858, {high, high};
}
{
mul.f16x2 r1860, r1667, r1858;
}
{
fma.rn.f16x2 r1863, r1523, r1856, r1860;
}
{
mul.f16x2 r1867, r1523, r1858;
}
{
neg.f16x2 r1870, r1867;
}
{
fma.rn.f16x2 r1872, r1667, r1856, r1870;
}
shl.b32 r2241, r2240, 2;
add.s32 r2242, r2222, r2241;
barrier.sync 0;
mad.lo.s32 r2243, r2238, 2500, r2242;
st.shared.u32 [r2243], r1439;
st.shared.u32 [r2243+500], r1752;
st.shared.u32 [r2243+1000], r1789;
st.shared.u32 [r2243+1500], r1826;
st.shared.u32 [r2243+2000], r1863;
barrier.sync 0;
ld.shared.u32 r1907, [r2225];
ld.shared.u32 r1904, [r2225+2500];
ld.shared.u32 r1910, [r2225+5000];
ld.shared.u32 r1911, [r2225+7500];
ld.shared.u32 r1905, [r2225+10000];
barrier.sync 0;
st.shared.u32 [r2243], r1451;
st.shared.u32 [r2243+500], r1761;
st.shared.u32 [r2243+1000], r1798;
st.shared.u32 [r2243+1500], r1835;
st.shared.u32 [r2243+2000], r1872;
barrier.sync 0;
ld.shared.u32 r1919, [r2225];
ld.shared.u32 r1916, [r2225+2500];
ld.shared.u32 r1922, [r2225+5000];
ld.shared.u32 r1923, [r2225+7500];
ld.shared.u32 r1917, [r2225+10000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1893, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1894, {low, high};
}
{
neg.f16x2 r1895, r1894;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r1897, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r1898, {low, high};
}
{
neg.f16x2 r1899, r1898;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1901, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1902, {low, high};
}
{
add.f16x2 r1903, r1904, r1905;
}
{
add.f16x2 r1906, r1907, r1903;
}
{
add.f16x2 r1909, r1910, r1911;
}
{
add.f16x2 %0, r1906, r1909;
}
{
add.f16x2 r1915, r1916, r1917;
}
{
add.f16x2 r1918, r1919, r1915;
}
{
add.f16x2 r1921, r1922, r1923;
}
{
add.f16x2 %1, r1918, r1921;
}
{
add.f16x2 r1927, r1904, r1905;
}
{
mul.f16x2 r1930, r1927, r1893;
}
{
add.f16x2 r1933, r1907, r1930;
}
{
add.f16x2 r1936, r1910, r1911;
}
{
mul.f16x2 r1939, r1936, r1897;
}
{
add.f16x2 r1942, r1933, r1939;
}
{
sub.f16x2 r1945, r1916, r1917;
}
{
mul.f16x2 r1948, r1945, r1895;
}
{
sub.f16x2 r1951, r1922, r1923;
}
{
mul.f16x2 r1954, r1951, r1899;
}
{
add.f16x2 r1957, r1948, r1954;
}
{
sub.f16x2 %2, r1942, r1957;
}
{
add.f16x2 r1963, r1904, r1905;
}
{
mul.f16x2 r1966, r1963, r1893;
}
{
add.f16x2 r1969, r1907, r1966;
}
{
add.f16x2 r1972, r1910, r1911;
}
{
mul.f16x2 r1975, r1972, r1897;
}
{
add.f16x2 r1978, r1969, r1975;
}
{
sub.f16x2 r1981, r1916, r1917;
}
{
mul.f16x2 r1984, r1981, r1895;
}
{
sub.f16x2 r1987, r1922, r1923;
}
{
mul.f16x2 r1990, r1987, r1899;
}
{
add.f16x2 r1993, r1984, r1990;
}
{
add.f16x2 %8, r1978, r1993;
}
{
add.f16x2 r1999, r1904, r1905;
}
{
mul.f16x2 r2002, r1999, r1897;
}
{
add.f16x2 r2005, r1907, r2002;
}
{
add.f16x2 r2008, r1910, r1911;
}
{
mul.f16x2 r2011, r2008, r1901;
}
{
add.f16x2 r2014, r2005, r2011;
}
{
sub.f16x2 r2017, r1916, r1917;
}
{
mul.f16x2 r2020, r2017, r1899;
}
{
sub.f16x2 r2023, r1922, r1923;
}
{
mul.f16x2 r2026, r2023, r1902;
}
{
add.f16x2 r2029, r2020, r2026;
}
{
sub.f16x2 %4, r2014, r2029;
}
{
add.f16x2 r2035, r1904, r1905;
}
{
mul.f16x2 r2038, r2035, r1897;
}
{
add.f16x2 r2041, r1907, r2038;
}
{
add.f16x2 r2044, r1910, r1911;
}
{
mul.f16x2 r2047, r2044, r1901;
}
{
add.f16x2 r2050, r2041, r2047;
}
{
sub.f16x2 r2053, r1916, r1917;
}
{
mul.f16x2 r2056, r2053, r1899;
}
{
sub.f16x2 r2059, r1922, r1923;
}
{
mul.f16x2 r2062, r2059, r1902;
}
{
add.f16x2 r2065, r2056, r2062;
}
{
add.f16x2 %6, r2050, r2065;
}
{
add.f16x2 r2071, r1916, r1917;
}
{
mul.f16x2 r2074, r2071, r1893;
}
{
add.f16x2 r2077, r1919, r2074;
}
{
add.f16x2 r2080, r1922, r1923;
}
{
mul.f16x2 r2083, r2080, r1897;
}
{
add.f16x2 r2086, r2077, r2083;
}
{
sub.f16x2 r2089, r1904, r1905;
}
{
mul.f16x2 r2092, r2089, r1895;
}
{
sub.f16x2 r2095, r1910, r1911;
}
{
mul.f16x2 r2098, r2095, r1899;
}
{
add.f16x2 r2101, r2092, r2098;
}
{
add.f16x2 %3, r2086, r2101;
}
{
add.f16x2 r2107, r1916, r1917;
}
{
mul.f16x2 r2110, r2107, r1893;
}
{
add.f16x2 r2113, r1919, r2110;
}
{
add.f16x2 r2116, r1922, r1923;
}
{
mul.f16x2 r2119, r2116, r1897;
}
{
add.f16x2 r2122, r2113, r2119;
}
{
sub.f16x2 r2125, r1904, r1905;
}
{
mul.f16x2 r2128, r2125, r1895;
}
{
sub.f16x2 r2131, r1910, r1911;
}
{
mul.f16x2 r2134, r2131, r1899;
}
{
add.f16x2 r2137, r2128, r2134;
}
{
sub.f16x2 %9, r2122, r2137;
}
{
add.f16x2 r2143, r1916, r1917;
}
{
mul.f16x2 r2146, r2143, r1897;
}
{
add.f16x2 r2149, r1919, r2146;
}
{
add.f16x2 r2152, r1922, r1923;
}
{
mul.f16x2 r2155, r2152, r1901;
}
{
add.f16x2 r2158, r2149, r2155;
}
{
sub.f16x2 r2161, r1904, r1905;
}
{
mul.f16x2 r2164, r2161, r1899;
}
{
sub.f16x2 r2167, r1910, r1911;
}
{
mul.f16x2 r2170, r2167, r1902;
}
{
add.f16x2 r2173, r2164, r2170;
}
{
add.f16x2 %5, r2158, r2173;
}
{
add.f16x2 r2179, r1916, r1917;
}
{
mul.f16x2 r2182, r2179, r1897;
}
{
add.f16x2 r2185, r1919, r2182;
}
{
add.f16x2 r2188, r1922, r1923;
}
{
mul.f16x2 r2191, r2188, r1901;
}
{
add.f16x2 r2194, r2185, r2191;
}
{
sub.f16x2 r2197, r1904, r1905;
}
{
mul.f16x2 r2200, r2197, r1899;
}
{
sub.f16x2 r2203, r1910, r1911;
}
{
mul.f16x2 r2206, r2203, r1902;
}
{
add.f16x2 r2209, r2200, r2206;
}
{
sub.f16x2 %7, r2194, r2209;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)));
};


#endif
