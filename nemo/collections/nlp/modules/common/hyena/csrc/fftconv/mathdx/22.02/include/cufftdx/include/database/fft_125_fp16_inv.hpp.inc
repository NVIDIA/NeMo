#ifndef CUFFTDX_FFT_125_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_125_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1107, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<332>;
.reg .b32 r<6088>;
.reg .b64 rd<4>;
mov.u32 r6086, %tid.y;
mov.u32 r6087, %50;
mad.lo.s32 r6028, r6086, 1000, r6087;
mov.u32 r6029, %tid.x;
mov.f32 f326, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r1, {low, high};
}
mov.f32 f328, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2, {low, high};
}
{
neg.f16x2 r3, r2;
}
mov.f32 f322, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r5, {low, high};
}
mov.f32 f324, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r9, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r10, {low, high};
}
{
add.f16x2 r11, %67, %52;
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
add.f16x2 r35, %67, %52;
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
add.f16x2 r71, %67, %52;
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
add.f16x2 r107, %67, %52;
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
add.f16x2 r143, %67, %52;
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
sub.f16x2 r197, %67, %52;
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
sub.f16x2 r233, %67, %52;
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
sub.f16x2 r269, %67, %52;
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
sub.f16x2 r305, %67, %52;
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
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r323, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r324, {low, high};
}
{
neg.f16x2 r325, r324;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r327, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r328, {low, high};
}
{
neg.f16x2 r329, r328;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r331, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
add.f16x2 r345, %66, %53;
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
sub.f16x2 r375, %66, %53;
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
sub.f16x2 r411, %66, %53;
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
sub.f16x2 r447, %66, %53;
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
sub.f16x2 r483, %66, %53;
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
add.f16x2 r501, %66, %53;
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
add.f16x2 r537, %66, %53;
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
add.f16x2 r573, %66, %53;
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
add.f16x2 r609, %66, %53;
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
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r645, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r646, {low, high};
}
{
neg.f16x2 r647, r646;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r649, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r650, {low, high};
}
{
neg.f16x2 r651, r650;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r653, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
add.f16x2 r673, %54, %86;
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
sub.f16x2 r703, %54, %86;
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
sub.f16x2 r739, %54, %86;
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
sub.f16x2 r775, %54, %86;
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
sub.f16x2 r811, %54, %86;
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
add.f16x2 r832, %54, %86;
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
add.f16x2 r868, %54, %86;
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
add.f16x2 r904, %54, %86;
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
add.f16x2 r940, %54, %86;
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
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r967, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r968, {low, high};
}
{
neg.f16x2 r969, r968;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r971, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r972, {low, high};
}
{
neg.f16x2 r973, r972;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r975, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r1289, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1290, {low, high};
}
{
neg.f16x2 r1291, r1290;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r1293, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r1294, {low, high};
}
{
neg.f16x2 r1295, r1294;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r1297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
mov.f32 f62, 0f3F77F511;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f62;
cvt.rn.f16.f32 high, f62;
mov.b32 r1611, {low, high};
}
mov.f32 f64, 0f3E7EA890;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f64;
cvt.rn.f16.f32 high, f64;
mov.b32 r1612, {low, high};
}
mov.f32 f66, 0f3F6055A2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f66;
cvt.rn.f16.f32 high, f66;
mov.b32 r1613, {low, high};
}
mov.f32 f68, 0f3EF6A86B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f68;
cvt.rn.f16.f32 high, f68;
mov.b32 r1614, {low, high};
}
mov.f32 f70, 0f3F3A9DB0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f70;
cvt.rn.f16.f32 high, f70;
mov.b32 r1615, {low, high};
}
mov.f32 f72, 0f3F2F3E7B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f72;
cvt.rn.f16.f32 high, f72;
mov.b32 r1616, {low, high};
}
mov.f32 f74, 0f3F092BF2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r1617, {low, high};
}
mov.f32 f76, 0f3F5825E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r1618, {low, high};
}
mov.f32 f82, 0f3D809851;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r1621, {low, high};
}
mov.f32 f84, 0f3F7F7EAE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r1622, {low, high};
}
mov.f32 f90, 0fBED9FFBE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r1625, {low, high};
}
mov.f32 f92, 0f3F67A2BF;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r1626, {low, high};
}
mov.f32 f122, 0fBF232E38;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r1627, {low, high};
}
mov.f32 f96, 0f3F45405B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r1628, {low, high};
}
mov.f32 f106, 0fBF7DFB3B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1633, {low, high};
}
mov.f32 f108, 0f3E00575B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1634, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r1641, {low, high};
}
mov.f32 f124, 0fBF45405B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
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
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r1915, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1916, {low, high};
}
{
neg.f16x2 r1917, r1916;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r1919, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r1920, {low, high};
}
{
neg.f16x2 r1921, r1920;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r1923, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2237, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2238, {low, high};
}
{
neg.f16x2 r2239, r2238;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r2241, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r2242, {low, high};
}
{
neg.f16x2 r2243, r2242;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2245, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2559, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2560, {low, high};
}
{
neg.f16x2 r2561, r2560;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r2563, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r2564, {low, high};
}
{
neg.f16x2 r2565, r2564;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2567, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2881, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2882, {low, high};
}
{
neg.f16x2 r2883, r2882;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r2885, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r2886, {low, high};
}
{
neg.f16x2 r2887, r2886;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2889, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r3203, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r3204, {low, high};
}
{
neg.f16x2 r3205, r3204;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r3207, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r3208, {low, high};
}
{
neg.f16x2 r3209, r3208;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r3211, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
mul.wide.u32 rd2, r6029, -858993459;
shr.u64 rd3, rd2, 34;
cvt.u32.u64 r6030, rd3;
mul.lo.s32 r6031, r6030, 5;
sub.s32 r6032, r6029, r6031;
cvt.rn.f32.u32 f329, r6032;
mul.f32 f330, f329, 0f3D4DE32E;
cos.approx.f32 f217, f330;
sin.approx.f32 f331, f330;
neg.f32 f218, f331;
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
mov.f32 f265, 0fBF800000;
mov.f32 f266, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
mad.lo.s32 r6033, r6030, 1000, r6028;
barrier.sync 0;
mad.lo.s32 r6034, r6032, 200, r6033;
st.shared.v2.f32 [r6034], {r1934, r1946};
st.shared.v2.f32 [r6034+8], {r3535, r3544};
st.shared.v2.f32 [r6034+16], {r3572, r3581};
st.shared.v2.f32 [r6034+24], {r3609, r3618};
st.shared.v2.f32 [r6034+32], {r3646, r3655};
st.shared.v2.f32 [r6034+40], {r3683, r3692};
st.shared.v2.f32 [r6034+48], {r3720, r3729};
st.shared.v2.f32 [r6034+56], {r3757, r3766};
st.shared.v2.f32 [r6034+64], {r3794, r3803};
st.shared.v2.f32 [r6034+72], {r3831, r3840};
st.shared.v2.f32 [r6034+80], {r3868, r3877};
st.shared.v2.f32 [r6034+88], {r3905, r3914};
st.shared.v2.f32 [r6034+96], {r3942, r3951};
st.shared.v2.f32 [r6034+104], {r3979, r3988};
st.shared.v2.f32 [r6034+112], {r4016, r4025};
st.shared.v2.f32 [r6034+120], {r4053, r4062};
st.shared.v2.f32 [r6034+128], {r4090, r4099};
st.shared.v2.f32 [r6034+136], {r4127, r4136};
st.shared.v2.f32 [r6034+144], {r4164, r4173};
st.shared.v2.f32 [r6034+152], {r4201, r4210};
st.shared.v2.f32 [r6034+160], {r4238, r4247};
st.shared.v2.f32 [r6034+168], {r4275, r4284};
st.shared.v2.f32 [r6034+176], {r4312, r4321};
st.shared.v2.f32 [r6034+184], {r4349, r4358};
st.shared.v2.f32 [r6034+192], {r4386, r4395};
barrier.sync 0;
mad.lo.s32 r6035, r6032, -192, r6034;
ld.shared.u32 r4430, [r6035];
ld.shared.u32 r4442, [r6035+4];
ld.shared.u32 r4752, [r6035+40];
ld.shared.u32 r4764, [r6035+44];
ld.shared.u32 r5074, [r6035+80];
ld.shared.u32 r5086, [r6035+84];
ld.shared.u32 r5396, [r6035+120];
ld.shared.u32 r5408, [r6035+124];
ld.shared.u32 r5718, [r6035+160];
ld.shared.u32 r5730, [r6035+164];
ld.shared.u32 r4427, [r6035+200];
ld.shared.u32 r4439, [r6035+204];
ld.shared.u32 r4749, [r6035+240];
ld.shared.u32 r4761, [r6035+244];
ld.shared.u32 r5071, [r6035+280];
ld.shared.u32 r5083, [r6035+284];
ld.shared.u32 r5393, [r6035+320];
ld.shared.u32 r5405, [r6035+324];
ld.shared.u32 r5715, [r6035+360];
ld.shared.u32 r5727, [r6035+364];
ld.shared.u32 r4433, [r6035+400];
ld.shared.u32 r4445, [r6035+404];
ld.shared.u32 r4755, [r6035+440];
ld.shared.u32 r4767, [r6035+444];
ld.shared.u32 r5077, [r6035+480];
ld.shared.u32 r5089, [r6035+484];
ld.shared.u32 r5399, [r6035+520];
ld.shared.u32 r5411, [r6035+524];
ld.shared.u32 r5721, [r6035+560];
ld.shared.u32 r5733, [r6035+564];
ld.shared.u32 r4434, [r6035+600];
ld.shared.u32 r4446, [r6035+604];
ld.shared.u32 r4756, [r6035+640];
ld.shared.u32 r4768, [r6035+644];
ld.shared.u32 r5078, [r6035+680];
ld.shared.u32 r5090, [r6035+684];
ld.shared.u32 r5400, [r6035+720];
ld.shared.u32 r5412, [r6035+724];
ld.shared.u32 r5722, [r6035+760];
ld.shared.u32 r5734, [r6035+764];
ld.shared.u32 r4428, [r6035+800];
ld.shared.u32 r4440, [r6035+804];
ld.shared.u32 r4750, [r6035+840];
ld.shared.u32 r4762, [r6035+844];
ld.shared.u32 r5072, [r6035+880];
ld.shared.u32 r5084, [r6035+884];
ld.shared.u32 r5394, [r6035+920];
ld.shared.u32 r5406, [r6035+924];
ld.shared.u32 r5716, [r6035+960];
ld.shared.u32 r5728, [r6035+964];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r4416, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r4417, {low, high};
}
{
neg.f16x2 r4418, r4417;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r4420, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r4421, {low, high};
}
{
neg.f16x2 r4422, r4421;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r4424, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
add.f16x2 %0, r4429, r4432;
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
add.f16x2 %1, r4441, r4444;
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
sub.f16x2 %10, r4465, r4480;
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
add.f16x2 %40, r4501, r4516;
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
sub.f16x2 %20, r4537, r4552;
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
add.f16x2 %30, r4573, r4588;
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
add.f16x2 %11, r4609, r4624;
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
sub.f16x2 %41, r4645, r4660;
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
add.f16x2 %21, r4681, r4696;
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
sub.f16x2 %31, r4717, r4732;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r4738, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r4739, {low, high};
}
{
neg.f16x2 r4740, r4739;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r4742, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r4743, {low, high};
}
{
neg.f16x2 r4744, r4743;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r4746, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
add.f16x2 %2, r4751, r4754;
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
add.f16x2 %3, r4763, r4766;
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
sub.f16x2 %12, r4787, r4802;
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
add.f16x2 %42, r4823, r4838;
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
sub.f16x2 %22, r4859, r4874;
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
add.f16x2 %32, r4895, r4910;
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
add.f16x2 %13, r4931, r4946;
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
sub.f16x2 %43, r4967, r4982;
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
add.f16x2 %23, r5003, r5018;
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
sub.f16x2 %33, r5039, r5054;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r5060, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r5061, {low, high};
}
{
neg.f16x2 r5062, r5061;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r5064, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r5065, {low, high};
}
{
neg.f16x2 r5066, r5065;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r5068, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
add.f16x2 %4, r5073, r5076;
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
add.f16x2 %5, r5085, r5088;
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
sub.f16x2 %14, r5109, r5124;
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
add.f16x2 %44, r5145, r5160;
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
sub.f16x2 %24, r5181, r5196;
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
add.f16x2 %34, r5217, r5232;
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
add.f16x2 %15, r5253, r5268;
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
sub.f16x2 %45, r5289, r5304;
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
add.f16x2 %25, r5325, r5340;
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
sub.f16x2 %35, r5361, r5376;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r5382, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r5383, {low, high};
}
{
neg.f16x2 r5384, r5383;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r5386, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r5387, {low, high};
}
{
neg.f16x2 r5388, r5387;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r5390, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
add.f16x2 %6, r5395, r5398;
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
add.f16x2 %7, r5407, r5410;
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
sub.f16x2 %16, r5431, r5446;
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
add.f16x2 %46, r5467, r5482;
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
sub.f16x2 %26, r5503, r5518;
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
add.f16x2 %36, r5539, r5554;
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
add.f16x2 %17, r5575, r5590;
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
sub.f16x2 %47, r5611, r5626;
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
add.f16x2 %27, r5647, r5662;
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
sub.f16x2 %37, r5683, r5698;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r5704, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r5705, {low, high};
}
{
neg.f16x2 r5706, r5705;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r5708, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r5709, {low, high};
}
{
neg.f16x2 r5710, r5709;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r5712, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
add.f16x2 %8, r5717, r5720;
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
add.f16x2 %9, r5729, r5732;
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
sub.f16x2 %18, r5753, r5768;
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
add.f16x2 %48, r5789, r5804;
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
sub.f16x2 %28, r5825, r5840;
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
add.f16x2 %38, r5861, r5876;
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
add.f16x2 %19, r5897, r5912;
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
sub.f16x2 %49, r5933, r5948;
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
add.f16x2 %29, r5969, r5984;
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
sub.f16x2 %39, r6005, r6020;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[13].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1106, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<332>;
.reg .b32 r<6088>;
.reg .b64 rd<4>;
mov.u32 r6086, %tid.y;
mov.u32 r6087, %50;
mad.lo.s32 r6028, r6086, 500, r6087;
mov.u32 r6029, %tid.x;
mov.f32 f326, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r1, {low, high};
}
mov.f32 f328, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2, {low, high};
}
{
neg.f16x2 r3, r2;
}
mov.f32 f322, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r5, {low, high};
}
mov.f32 f324, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r9, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r10, {low, high};
}
{
add.f16x2 r11, %67, %52;
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
add.f16x2 r35, %67, %52;
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
add.f16x2 r71, %67, %52;
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
add.f16x2 r107, %67, %52;
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
add.f16x2 r143, %67, %52;
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
sub.f16x2 r197, %67, %52;
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
sub.f16x2 r233, %67, %52;
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
sub.f16x2 r269, %67, %52;
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
sub.f16x2 r305, %67, %52;
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
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r323, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r324, {low, high};
}
{
neg.f16x2 r325, r324;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r327, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r328, {low, high};
}
{
neg.f16x2 r329, r328;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r331, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
add.f16x2 r345, %66, %53;
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
sub.f16x2 r375, %66, %53;
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
sub.f16x2 r411, %66, %53;
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
sub.f16x2 r447, %66, %53;
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
sub.f16x2 r483, %66, %53;
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
add.f16x2 r501, %66, %53;
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
add.f16x2 r537, %66, %53;
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
add.f16x2 r573, %66, %53;
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
add.f16x2 r609, %66, %53;
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
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r645, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r646, {low, high};
}
{
neg.f16x2 r647, r646;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r649, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r650, {low, high};
}
{
neg.f16x2 r651, r650;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r653, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
add.f16x2 r673, %54, %86;
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
sub.f16x2 r703, %54, %86;
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
sub.f16x2 r739, %54, %86;
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
sub.f16x2 r775, %54, %86;
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
sub.f16x2 r811, %54, %86;
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
add.f16x2 r832, %54, %86;
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
add.f16x2 r868, %54, %86;
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
add.f16x2 r904, %54, %86;
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
add.f16x2 r940, %54, %86;
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
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r967, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r968, {low, high};
}
{
neg.f16x2 r969, r968;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r971, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r972, {low, high};
}
{
neg.f16x2 r973, r972;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r975, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r1289, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1290, {low, high};
}
{
neg.f16x2 r1291, r1290;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r1293, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r1294, {low, high};
}
{
neg.f16x2 r1295, r1294;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r1297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
mov.f32 f62, 0f3F77F511;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f62;
cvt.rn.f16.f32 high, f62;
mov.b32 r1611, {low, high};
}
mov.f32 f64, 0f3E7EA890;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f64;
cvt.rn.f16.f32 high, f64;
mov.b32 r1612, {low, high};
}
mov.f32 f66, 0f3F6055A2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f66;
cvt.rn.f16.f32 high, f66;
mov.b32 r1613, {low, high};
}
mov.f32 f68, 0f3EF6A86B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f68;
cvt.rn.f16.f32 high, f68;
mov.b32 r1614, {low, high};
}
mov.f32 f70, 0f3F3A9DB0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f70;
cvt.rn.f16.f32 high, f70;
mov.b32 r1615, {low, high};
}
mov.f32 f72, 0f3F2F3E7B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f72;
cvt.rn.f16.f32 high, f72;
mov.b32 r1616, {low, high};
}
mov.f32 f74, 0f3F092BF2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r1617, {low, high};
}
mov.f32 f76, 0f3F5825E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r1618, {low, high};
}
mov.f32 f82, 0f3D809851;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r1621, {low, high};
}
mov.f32 f84, 0f3F7F7EAE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r1622, {low, high};
}
mov.f32 f90, 0fBED9FFBE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r1625, {low, high};
}
mov.f32 f92, 0f3F67A2BF;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r1626, {low, high};
}
mov.f32 f122, 0fBF232E38;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r1627, {low, high};
}
mov.f32 f96, 0f3F45405B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r1628, {low, high};
}
mov.f32 f106, 0fBF7DFB3B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1633, {low, high};
}
mov.f32 f108, 0f3E00575B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1634, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r1641, {low, high};
}
mov.f32 f124, 0fBF45405B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
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
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r1915, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1916, {low, high};
}
{
neg.f16x2 r1917, r1916;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r1919, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r1920, {low, high};
}
{
neg.f16x2 r1921, r1920;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r1923, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2237, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2238, {low, high};
}
{
neg.f16x2 r2239, r2238;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r2241, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r2242, {low, high};
}
{
neg.f16x2 r2243, r2242;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2245, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2559, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2560, {low, high};
}
{
neg.f16x2 r2561, r2560;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r2563, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r2564, {low, high};
}
{
neg.f16x2 r2565, r2564;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2567, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2881, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2882, {low, high};
}
{
neg.f16x2 r2883, r2882;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r2885, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r2886, {low, high};
}
{
neg.f16x2 r2887, r2886;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2889, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r3203, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r3204, {low, high};
}
{
neg.f16x2 r3205, r3204;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r3207, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r3208, {low, high};
}
{
neg.f16x2 r3209, r3208;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r3211, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
mul.wide.u32 rd2, r6029, -858993459;
shr.u64 rd3, rd2, 34;
cvt.u32.u64 r6030, rd3;
mul.lo.s32 r6031, r6030, 5;
sub.s32 r6032, r6029, r6031;
mad.lo.s32 r6033, r6030, 500, r6028;
cvt.rn.f32.u32 f329, r6032;
mul.f32 f330, f329, 0f3D4DE32E;
cos.approx.f32 f217, f330;
sin.approx.f32 f331, f330;
neg.f32 f218, f331;
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
mov.f32 f265, 0fBF800000;
mov.f32 f266, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
mad.lo.s32 r6034, r6032, 100, r6033;
st.shared.u32 [r6034], r1934;
st.shared.u32 [r6034+4], r3535;
st.shared.u32 [r6034+8], r3572;
st.shared.u32 [r6034+12], r3609;
st.shared.u32 [r6034+16], r3646;
st.shared.u32 [r6034+20], r3683;
st.shared.u32 [r6034+24], r3720;
st.shared.u32 [r6034+28], r3757;
st.shared.u32 [r6034+32], r3794;
st.shared.u32 [r6034+36], r3831;
st.shared.u32 [r6034+40], r3868;
st.shared.u32 [r6034+44], r3905;
st.shared.u32 [r6034+48], r3942;
st.shared.u32 [r6034+52], r3979;
st.shared.u32 [r6034+56], r4016;
st.shared.u32 [r6034+60], r4053;
st.shared.u32 [r6034+64], r4090;
st.shared.u32 [r6034+68], r4127;
st.shared.u32 [r6034+72], r4164;
st.shared.u32 [r6034+76], r4201;
st.shared.u32 [r6034+80], r4238;
st.shared.u32 [r6034+84], r4275;
st.shared.u32 [r6034+88], r4312;
st.shared.u32 [r6034+92], r4349;
st.shared.u32 [r6034+96], r4386;
barrier.sync 0;
mad.lo.s32 r6035, r6032, -96, r6034;
ld.shared.u32 r4430, [r6035];
ld.shared.u32 r4752, [r6035+20];
ld.shared.u32 r5074, [r6035+40];
ld.shared.u32 r5396, [r6035+60];
ld.shared.u32 r5718, [r6035+80];
ld.shared.u32 r4427, [r6035+100];
ld.shared.u32 r4749, [r6035+120];
ld.shared.u32 r5071, [r6035+140];
ld.shared.u32 r5393, [r6035+160];
ld.shared.u32 r5715, [r6035+180];
ld.shared.u32 r4433, [r6035+200];
ld.shared.u32 r4755, [r6035+220];
ld.shared.u32 r5077, [r6035+240];
ld.shared.u32 r5399, [r6035+260];
ld.shared.u32 r5721, [r6035+280];
ld.shared.u32 r4434, [r6035+300];
ld.shared.u32 r4756, [r6035+320];
ld.shared.u32 r5078, [r6035+340];
ld.shared.u32 r5400, [r6035+360];
ld.shared.u32 r5722, [r6035+380];
ld.shared.u32 r4428, [r6035+400];
ld.shared.u32 r4750, [r6035+420];
ld.shared.u32 r5072, [r6035+440];
ld.shared.u32 r5394, [r6035+460];
ld.shared.u32 r5716, [r6035+480];
barrier.sync 0;
st.shared.u32 [r6034], r1946;
st.shared.u32 [r6034+4], r3544;
st.shared.u32 [r6034+8], r3581;
st.shared.u32 [r6034+12], r3618;
st.shared.u32 [r6034+16], r3655;
st.shared.u32 [r6034+20], r3692;
st.shared.u32 [r6034+24], r3729;
st.shared.u32 [r6034+28], r3766;
st.shared.u32 [r6034+32], r3803;
st.shared.u32 [r6034+36], r3840;
st.shared.u32 [r6034+40], r3877;
st.shared.u32 [r6034+44], r3914;
st.shared.u32 [r6034+48], r3951;
st.shared.u32 [r6034+52], r3988;
st.shared.u32 [r6034+56], r4025;
st.shared.u32 [r6034+60], r4062;
st.shared.u32 [r6034+64], r4099;
st.shared.u32 [r6034+68], r4136;
st.shared.u32 [r6034+72], r4173;
st.shared.u32 [r6034+76], r4210;
st.shared.u32 [r6034+80], r4247;
st.shared.u32 [r6034+84], r4284;
st.shared.u32 [r6034+88], r4321;
st.shared.u32 [r6034+92], r4358;
st.shared.u32 [r6034+96], r4395;
barrier.sync 0;
ld.shared.u32 r4442, [r6035];
ld.shared.u32 r4764, [r6035+20];
ld.shared.u32 r5086, [r6035+40];
ld.shared.u32 r5408, [r6035+60];
ld.shared.u32 r5730, [r6035+80];
ld.shared.u32 r4439, [r6035+100];
ld.shared.u32 r4761, [r6035+120];
ld.shared.u32 r5083, [r6035+140];
ld.shared.u32 r5405, [r6035+160];
ld.shared.u32 r5727, [r6035+180];
ld.shared.u32 r4445, [r6035+200];
ld.shared.u32 r4767, [r6035+220];
ld.shared.u32 r5089, [r6035+240];
ld.shared.u32 r5411, [r6035+260];
ld.shared.u32 r5733, [r6035+280];
ld.shared.u32 r4446, [r6035+300];
ld.shared.u32 r4768, [r6035+320];
ld.shared.u32 r5090, [r6035+340];
ld.shared.u32 r5412, [r6035+360];
ld.shared.u32 r5734, [r6035+380];
ld.shared.u32 r4440, [r6035+400];
ld.shared.u32 r4762, [r6035+420];
ld.shared.u32 r5084, [r6035+440];
ld.shared.u32 r5406, [r6035+460];
ld.shared.u32 r5728, [r6035+480];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r4416, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r4417, {low, high};
}
{
neg.f16x2 r4418, r4417;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r4420, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r4421, {low, high};
}
{
neg.f16x2 r4422, r4421;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r4424, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
add.f16x2 %0, r4429, r4432;
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
add.f16x2 %1, r4441, r4444;
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
sub.f16x2 %10, r4465, r4480;
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
add.f16x2 %40, r4501, r4516;
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
sub.f16x2 %20, r4537, r4552;
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
add.f16x2 %30, r4573, r4588;
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
add.f16x2 %11, r4609, r4624;
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
sub.f16x2 %41, r4645, r4660;
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
add.f16x2 %21, r4681, r4696;
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
sub.f16x2 %31, r4717, r4732;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r4738, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r4739, {low, high};
}
{
neg.f16x2 r4740, r4739;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r4742, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r4743, {low, high};
}
{
neg.f16x2 r4744, r4743;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r4746, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
add.f16x2 %2, r4751, r4754;
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
add.f16x2 %3, r4763, r4766;
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
sub.f16x2 %12, r4787, r4802;
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
add.f16x2 %42, r4823, r4838;
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
sub.f16x2 %22, r4859, r4874;
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
add.f16x2 %32, r4895, r4910;
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
add.f16x2 %13, r4931, r4946;
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
sub.f16x2 %43, r4967, r4982;
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
add.f16x2 %23, r5003, r5018;
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
sub.f16x2 %33, r5039, r5054;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r5060, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r5061, {low, high};
}
{
neg.f16x2 r5062, r5061;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r5064, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r5065, {low, high};
}
{
neg.f16x2 r5066, r5065;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r5068, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
add.f16x2 %4, r5073, r5076;
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
add.f16x2 %5, r5085, r5088;
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
sub.f16x2 %14, r5109, r5124;
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
add.f16x2 %44, r5145, r5160;
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
sub.f16x2 %24, r5181, r5196;
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
add.f16x2 %34, r5217, r5232;
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
add.f16x2 %15, r5253, r5268;
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
sub.f16x2 %45, r5289, r5304;
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
add.f16x2 %25, r5325, r5340;
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
sub.f16x2 %35, r5361, r5376;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r5382, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r5383, {low, high};
}
{
neg.f16x2 r5384, r5383;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r5386, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r5387, {low, high};
}
{
neg.f16x2 r5388, r5387;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r5390, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
add.f16x2 %6, r5395, r5398;
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
add.f16x2 %7, r5407, r5410;
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
sub.f16x2 %16, r5431, r5446;
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
add.f16x2 %46, r5467, r5482;
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
sub.f16x2 %26, r5503, r5518;
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
add.f16x2 %36, r5539, r5554;
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
add.f16x2 %17, r5575, r5590;
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
sub.f16x2 %47, r5611, r5626;
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
add.f16x2 %27, r5647, r5662;
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
sub.f16x2 %37, r5683, r5698;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r5704, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r5705, {low, high};
}
{
neg.f16x2 r5706, r5705;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f322;
cvt.rn.f16.f32 high, f322;
mov.b32 r5708, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f324;
cvt.rn.f16.f32 high, f324;
mov.b32 r5709, {low, high};
}
{
neg.f16x2 r5710, r5709;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r5712, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
add.f16x2 %8, r5717, r5720;
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
add.f16x2 %9, r5729, r5732;
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
sub.f16x2 %18, r5753, r5768;
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
add.f16x2 %48, r5789, r5804;
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
sub.f16x2 %28, r5825, r5840;
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
add.f16x2 %38, r5861, r5876;
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
add.f16x2 %19, r5897, r5912;
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
sub.f16x2 %49, r5933, r5948;
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
add.f16x2 %29, r5969, r5984;
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
sub.f16x2 %39, r6005, r6020;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[13].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1108, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<67>;
.reg .b32 r<1286>;
.reg .b64 rd<6>;
mov.u32 r1269, %tid.x;
mov.f32 f58, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r1, {low, high};
}
mov.f32 f60, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f60;
cvt.rn.f16.f32 high, f60;
mov.b32 r2, {low, high};
}
{
neg.f16x2 r3, r2;
}
mov.f32 f54, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f54;
cvt.rn.f16.f32 high, f54;
mov.b32 r5, {low, high};
}
mov.f32 f56, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r9, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f60;
cvt.rn.f16.f32 high, f60;
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
mul.wide.u32 rd2, r1269, 1374389535;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r1270, rd3;
mul.lo.s32 r1271, r1270, 25;
sub.s32 r1272, r1269, r1271;
cvt.rn.f32.u32 f61, r1272;
mul.f32 f62, f61, 0f3D4DE32E;
cos.approx.f32 f13, f62;
sin.approx.f32 f63, f62;
neg.f32 f14, f63;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f13;
cvt.rn.f16.f32 high, f14;
mov.b32 r323, {low, high};
}
mov.u32 r1273, %tid.y;
mov.u32 r1274, %10;
mad.lo.s32 r1275, r1273, 1000, r1274;
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
mov.f32 f45, 0fBF800000;
mov.f32 f46, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
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
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
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
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
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
mad.lo.s32 r1276, r1270, 1000, r1275;
barrier.sync 0;
mad.lo.s32 r1277, r1272, 40, r1276;
st.shared.v2.f32 [r1277], {r20, r32};
st.shared.v2.f32 [r1277+8], {r333, r342};
st.shared.v2.f32 [r1277+16], {r370, r379};
st.shared.v2.f32 [r1277+24], {r407, r416};
st.shared.v2.f32 [r1277+32], {r444, r453};
barrier.sync 0;
shl.b32 r1278, r1272, 5;
sub.s32 r1279, r1277, r1278;
ld.shared.u32 r488, [r1279];
ld.shared.u32 r500, [r1279+4];
ld.shared.u32 r485, [r1279+200];
ld.shared.u32 r497, [r1279+204];
ld.shared.u32 r491, [r1279+400];
ld.shared.u32 r503, [r1279+404];
ld.shared.u32 r492, [r1279+600];
ld.shared.u32 r504, [r1279+604];
ld.shared.u32 r486, [r1279+800];
ld.shared.u32 r498, [r1279+804];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r474, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f60;
cvt.rn.f16.f32 high, f60;
mov.b32 r475, {low, high};
}
{
neg.f16x2 r476, r475;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f54;
cvt.rn.f16.f32 high, f54;
mov.b32 r478, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r479, {low, high};
}
{
neg.f16x2 r480, r479;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r482, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f60;
cvt.rn.f16.f32 high, f60;
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
mul.wide.u32 rd4, r1272, -858993459;
shr.u64 rd5, rd4, 34;
cvt.u32.u64 r1280, rd5;
cvt.rn.f32.u32 f64, r1280;
mul.f32 f65, f64, 0f3E80ADFD;
cos.approx.f32 f37, f65;
sin.approx.f32 f66, f65;
neg.f32 f38, f66;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
mov.b32 r796, {low, high};
}
mul.lo.s32 r1281, r1280, 5;
sub.s32 r1282, r1272, r1281;
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
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
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
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
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
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
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
shl.b32 r1283, r1282, 3;
add.s32 r1284, r1276, r1283;
barrier.sync 0;
mad.lo.s32 r1285, r1280, 200, r1284;
st.shared.u32 [r1285], r493;
st.shared.u32 [r1285+4], r505;
st.shared.u32 [r1285+40], r806;
st.shared.u32 [r1285+44], r815;
st.shared.u32 [r1285+80], r843;
st.shared.u32 [r1285+84], r852;
st.shared.u32 [r1285+120], r880;
st.shared.u32 [r1285+124], r889;
st.shared.u32 [r1285+160], r917;
st.shared.u32 [r1285+164], r926;
barrier.sync 0;
ld.shared.u32 r961, [r1279];
ld.shared.u32 r973, [r1279+4];
ld.shared.u32 r958, [r1279+200];
ld.shared.u32 r970, [r1279+204];
ld.shared.u32 r964, [r1279+400];
ld.shared.u32 r976, [r1279+404];
ld.shared.u32 r965, [r1279+600];
ld.shared.u32 r977, [r1279+604];
ld.shared.u32 r959, [r1279+800];
ld.shared.u32 r971, [r1279+804];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r947, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f60;
cvt.rn.f16.f32 high, f60;
mov.b32 r948, {low, high};
}
{
neg.f16x2 r949, r948;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f54;
cvt.rn.f16.f32 high, f54;
mov.b32 r951, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r952, {low, high};
}
{
neg.f16x2 r953, r952;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r955, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f60;
cvt.rn.f16.f32 high, f60;
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
add.f16x2 %0, r960, r963;
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
add.f16x2 %1, r972, r975;
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
sub.f16x2 %2, r996, r1011;
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
add.f16x2 %8, r1032, r1047;
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
sub.f16x2 %4, r1068, r1083;
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
add.f16x2 %6, r1104, r1119;
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
add.f16x2 %3, r1140, r1155;
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
sub.f16x2 %9, r1176, r1191;
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
add.f16x2 %5, r1212, r1227;
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
sub.f16x2 %7, r1248, r1263;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1109, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<67>;
.reg .b32 r<1286>;
.reg .b64 rd<6>;
mov.u32 r1269, %tid.y;
mov.u32 r1270, %10;
mad.lo.s32 r1271, r1269, 500, r1270;
mov.u32 r1272, %tid.x;
mov.f32 f58, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r1, {low, high};
}
mov.f32 f60, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f60;
cvt.rn.f16.f32 high, f60;
mov.b32 r2, {low, high};
}
{
neg.f16x2 r3, r2;
}
mov.f32 f54, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f54;
cvt.rn.f16.f32 high, f54;
mov.b32 r5, {low, high};
}
mov.f32 f56, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r9, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f60;
cvt.rn.f16.f32 high, f60;
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
mul.wide.u32 rd2, r1272, 1374389535;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r1273, rd3;
mul.lo.s32 r1274, r1273, 25;
sub.s32 r1275, r1272, r1274;
mad.lo.s32 r1276, r1273, 500, r1271;
cvt.rn.f32.u32 f61, r1275;
mul.f32 f62, f61, 0f3D4DE32E;
cos.approx.f32 f13, f62;
sin.approx.f32 f63, f62;
neg.f32 f14, f63;
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
mov.f32 f45, 0fBF800000;
mov.f32 f46, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
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
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
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
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
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
barrier.sync 0;
mad.lo.s32 r1277, r1275, 20, r1276;
st.shared.u32 [r1277], r20;
st.shared.u32 [r1277+4], r333;
st.shared.u32 [r1277+8], r370;
st.shared.u32 [r1277+12], r407;
st.shared.u32 [r1277+16], r444;
barrier.sync 0;
shl.b32 r1278, r1275, 4;
sub.s32 r1279, r1277, r1278;
ld.shared.u32 r488, [r1279];
ld.shared.u32 r485, [r1279+100];
ld.shared.u32 r491, [r1279+200];
ld.shared.u32 r492, [r1279+300];
ld.shared.u32 r486, [r1279+400];
barrier.sync 0;
st.shared.u32 [r1277], r32;
st.shared.u32 [r1277+4], r342;
st.shared.u32 [r1277+8], r379;
st.shared.u32 [r1277+12], r416;
st.shared.u32 [r1277+16], r453;
barrier.sync 0;
ld.shared.u32 r500, [r1279];
ld.shared.u32 r497, [r1279+100];
ld.shared.u32 r503, [r1279+200];
ld.shared.u32 r504, [r1279+300];
ld.shared.u32 r498, [r1279+400];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r474, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f60;
cvt.rn.f16.f32 high, f60;
mov.b32 r475, {low, high};
}
{
neg.f16x2 r476, r475;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f54;
cvt.rn.f16.f32 high, f54;
mov.b32 r478, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r479, {low, high};
}
{
neg.f16x2 r480, r479;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r482, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f60;
cvt.rn.f16.f32 high, f60;
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
mul.wide.u32 rd4, r1275, -858993459;
shr.u64 rd5, rd4, 34;
cvt.u32.u64 r1280, rd5;
mul.lo.s32 r1281, r1280, 5;
sub.s32 r1282, r1275, r1281;
shl.b32 r1283, r1282, 2;
add.s32 r1284, r1276, r1283;
cvt.rn.f32.u32 f64, r1280;
mul.f32 f65, f64, 0f3E80ADFD;
cos.approx.f32 f37, f65;
sin.approx.f32 f66, f65;
neg.f32 f38, f66;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
mov.b32 r796, {low, high};
}
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
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
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
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
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
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
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
barrier.sync 0;
mad.lo.s32 r1285, r1280, 100, r1284;
st.shared.u32 [r1285], r493;
st.shared.u32 [r1285+20], r806;
st.shared.u32 [r1285+40], r843;
st.shared.u32 [r1285+60], r880;
st.shared.u32 [r1285+80], r917;
barrier.sync 0;
ld.shared.u32 r961, [r1279];
ld.shared.u32 r958, [r1279+100];
ld.shared.u32 r964, [r1279+200];
ld.shared.u32 r965, [r1279+300];
ld.shared.u32 r959, [r1279+400];
barrier.sync 0;
st.shared.u32 [r1285], r505;
st.shared.u32 [r1285+20], r815;
st.shared.u32 [r1285+40], r852;
st.shared.u32 [r1285+60], r889;
st.shared.u32 [r1285+80], r926;
barrier.sync 0;
ld.shared.u32 r973, [r1279];
ld.shared.u32 r970, [r1279+100];
ld.shared.u32 r976, [r1279+200];
ld.shared.u32 r977, [r1279+300];
ld.shared.u32 r971, [r1279+400];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r947, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f60;
cvt.rn.f16.f32 high, f60;
mov.b32 r948, {low, high};
}
{
neg.f16x2 r949, r948;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f54;
cvt.rn.f16.f32 high, f54;
mov.b32 r951, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r952, {low, high};
}
{
neg.f16x2 r953, r952;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r955, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f60;
cvt.rn.f16.f32 high, f60;
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
add.f16x2 %0, r960, r963;
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
add.f16x2 %1, r972, r975;
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
sub.f16x2 %2, r996, r1011;
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
add.f16x2 %8, r1032, r1047;
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
sub.f16x2 %4, r1068, r1083;
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
add.f16x2 %6, r1104, r1119;
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
add.f16x2 %3, r1140, r1155;
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
sub.f16x2 %9, r1176, r1191;
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
add.f16x2 %5, r1212, r1227;
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
sub.f16x2 %7, r1248, r1263;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)));
};


#endif
