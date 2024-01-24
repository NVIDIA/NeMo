#ifndef CUFFTDX_FFT_16384_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_16384_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1180, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<1085>;
.reg .b32 r<7043>;
.reg .b64 rd<3>;
mov.u32 r6957, %tid.y;
shl.b32 r6958, r6957, 17;
mov.u32 r6959, %64;
add.s32 r6960, r6959, r6958;
mov.u32 r6961, %tid.x;
{
add.f16x2 r1, %119, %111;
}
{
add.f16x2 r4, %91, %81;
}
{
sub.f16x2 r7, %119, %111;
}
{
sub.f16x2 r10, %91, %81;
}
{
add.f16x2 r13, %73, %128;
}
{
add.f16x2 r16, %106, %100;
}
{
sub.f16x2 r19, %73, %128;
}
{
sub.f16x2 r22, %106, %100;
}
{
neg.f16x2 r25, r22;
}
{
add.f16x2 r27, r1, r13;
}
{
add.f16x2 r30, r4, r16;
}
{
sub.f16x2 r33, r1, r13;
}
{
sub.f16x2 r36, r4, r16;
}
{
add.f16x2 r39, r7, r25;
}
{
add.f16x2 r42, r10, r19;
}
{
sub.f16x2 r45, r7, r25;
}
{
sub.f16x2 r48, r10, r19;
}
{
add.f16x2 r51, %105, %96;
}
{
add.f16x2 r54, %77, %67;
}
{
sub.f16x2 r57, %105, %96;
}
{
sub.f16x2 r60, %77, %67;
}
{
add.f16x2 r63, %122, %115;
}
{
add.f16x2 r66, %93, %85;
}
{
sub.f16x2 r69, %122, %115;
}
{
sub.f16x2 r72, %93, %85;
}
{
neg.f16x2 r75, r72;
}
{
add.f16x2 r77, r51, r63;
}
{
add.f16x2 r80, r54, r66;
}
{
sub.f16x2 r83, r51, r63;
}
{
sub.f16x2 r86, r54, r66;
}
{
add.f16x2 r89, r57, r75;
}
{
add.f16x2 r92, r60, r69;
}
{
sub.f16x2 r95, r57, r75;
}
{
sub.f16x2 r98, r60, r69;
}
mov.f32 f1040, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r102, {low, high};
}
mov.f32 f1038, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r106, {low, high};
}
{
mul.f16x2 r115, r89, r101;
}
{
mul.f16x2 r118, r92, r102;
}
{
sub.f16x2 r121, r115, r118;
}
{
mul.f16x2 r124, r89, r102;
}
{
fma.rn.f16x2 r127, r92, r101, r124;
}
{
neg.f16x2 r131, r86;
}
{
mul.f16x2 r133, r95, r105;
}
{
mul.f16x2 r136, r98, r106;
}
{
sub.f16x2 r139, r133, r136;
}
{
mul.f16x2 r142, r95, r106;
}
{
fma.rn.f16x2 r145, r98, r105, r142;
}
{
add.f16x2 r149, r27, r77;
}
{
add.f16x2 r152, r30, r80;
}
{
sub.f16x2 r155, r27, r77;
}
{
sub.f16x2 r158, r30, r80;
}
{
add.f16x2 r161, r39, r121;
}
{
add.f16x2 r164, r42, r127;
}
{
sub.f16x2 r167, r39, r121;
}
{
sub.f16x2 r170, r42, r127;
}
{
add.f16x2 r173, r33, r131;
}
{
add.f16x2 r176, r36, r83;
}
{
sub.f16x2 r179, r33, r131;
}
{
sub.f16x2 r182, r36, r83;
}
{
add.f16x2 r185, r45, r139;
}
{
add.f16x2 r188, r48, r145;
}
{
sub.f16x2 r191, r45, r139;
}
{
sub.f16x2 r194, r48, r145;
}
{
add.f16x2 r197, %94, %87;
}
{
add.f16x2 r200, %66, %123;
}
{
sub.f16x2 r203, %94, %87;
}
{
sub.f16x2 r206, %66, %123;
}
{
add.f16x2 r209, %113, %103;
}
{
add.f16x2 r212, %83, %75;
}
{
sub.f16x2 r215, %113, %103;
}
{
sub.f16x2 r218, %83, %75;
}
{
neg.f16x2 r221, r218;
}
{
add.f16x2 r223, r197, r209;
}
{
add.f16x2 r226, r200, r212;
}
{
sub.f16x2 r229, r197, r209;
}
{
sub.f16x2 r232, r200, r212;
}
{
add.f16x2 r235, r203, r221;
}
{
add.f16x2 r238, r206, r215;
}
{
sub.f16x2 r241, r203, r221;
}
{
sub.f16x2 r244, r206, r215;
}
{
add.f16x2 r247, %78, %72;
}
{
add.f16x2 r250, %117, %108;
}
{
sub.f16x2 r253, %78, %72;
}
{
sub.f16x2 r256, %117, %108;
}
{
add.f16x2 r259, %97, %89;
}
{
add.f16x2 r262, %69, %125;
}
{
sub.f16x2 r265, %97, %89;
}
{
sub.f16x2 r268, %69, %125;
}
{
neg.f16x2 r271, r268;
}
{
add.f16x2 r273, r247, r259;
}
{
add.f16x2 r276, r250, r262;
}
{
sub.f16x2 r279, r247, r259;
}
{
sub.f16x2 r282, r250, r262;
}
{
add.f16x2 r285, r253, r271;
}
{
add.f16x2 r288, r256, r265;
}
{
sub.f16x2 r291, r253, r271;
}
{
sub.f16x2 r294, r256, r265;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r302, {low, high};
}
{
mul.f16x2 r311, r285, r297;
}
{
mul.f16x2 r314, r288, r298;
}
{
sub.f16x2 r317, r311, r314;
}
{
mul.f16x2 r320, r285, r298;
}
{
fma.rn.f16x2 r323, r288, r297, r320;
}
{
neg.f16x2 r327, r282;
}
{
mul.f16x2 r329, r291, r301;
}
{
mul.f16x2 r332, r294, r302;
}
{
sub.f16x2 r335, r329, r332;
}
{
mul.f16x2 r338, r291, r302;
}
{
fma.rn.f16x2 r341, r294, r301, r338;
}
{
add.f16x2 r345, r223, r273;
}
{
add.f16x2 r348, r226, r276;
}
{
sub.f16x2 r351, r223, r273;
}
{
sub.f16x2 r354, r226, r276;
}
{
add.f16x2 r357, r235, r317;
}
{
add.f16x2 r360, r238, r323;
}
{
sub.f16x2 r363, r235, r317;
}
{
sub.f16x2 r366, r238, r323;
}
{
add.f16x2 r369, r229, r327;
}
{
add.f16x2 r372, r232, r279;
}
{
sub.f16x2 r375, r229, r327;
}
{
sub.f16x2 r378, r232, r279;
}
{
add.f16x2 r381, r241, r335;
}
{
add.f16x2 r384, r244, r341;
}
{
sub.f16x2 r387, r241, r335;
}
{
sub.f16x2 r390, r244, r341;
}
mov.f32 f1036, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r393, {low, high};
}
mov.f32 f1044, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r396, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r398, {low, high};
}
mov.f32 f1034, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r404, {low, high};
}
mov.f32 f1042, 0fBF6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1042;
cvt.rn.f16.f32 high, f1042;
mov.b32 r405, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r406, {low, high};
}
{
mul.f16x2 r423, r357, r393;
}
{
mul.f16x2 r426, r360, r394;
}
{
sub.f16x2 r429, r423, r426;
}
{
mul.f16x2 r432, r357, r394;
}
{
fma.rn.f16x2 r435, r360, r393, r432;
}
{
mul.f16x2 r439, r369, r395;
}
{
mul.f16x2 r442, r372, r396;
}
{
sub.f16x2 r445, r439, r442;
}
{
mul.f16x2 r448, r369, r396;
}
{
fma.rn.f16x2 r451, r372, r395, r448;
}
{
mul.f16x2 r455, r381, r397;
}
{
mul.f16x2 r458, r384, r398;
}
{
sub.f16x2 r461, r455, r458;
}
{
mul.f16x2 r464, r381, r398;
}
{
fma.rn.f16x2 r467, r384, r397, r464;
}
{
neg.f16x2 r471, r354;
}
{
mul.f16x2 r473, r363, r401;
}
{
mul.f16x2 r476, r366, r402;
}
{
sub.f16x2 r479, r473, r476;
}
{
mul.f16x2 r482, r363, r402;
}
{
fma.rn.f16x2 r485, r366, r401, r482;
}
{
mul.f16x2 r489, r375, r403;
}
{
mul.f16x2 r492, r378, r404;
}
{
sub.f16x2 r495, r489, r492;
}
{
mul.f16x2 r498, r375, r404;
}
{
fma.rn.f16x2 r501, r378, r403, r498;
}
{
mul.f16x2 r505, r387, r405;
}
{
mul.f16x2 r508, r390, r406;
}
{
sub.f16x2 r511, r505, r508;
}
{
mul.f16x2 r514, r387, r406;
}
{
fma.rn.f16x2 r517, r390, r405, r514;
}
{
add.f16x2 r521, r149, r345;
}
{
add.f16x2 r524, r152, r348;
}
{
sub.f16x2 r527, r149, r345;
}
{
sub.f16x2 r530, r152, r348;
}
{
add.f16x2 r533, r161, r429;
}
{
add.f16x2 r536, r164, r435;
}
{
sub.f16x2 r539, r161, r429;
}
{
sub.f16x2 r542, r164, r435;
}
{
add.f16x2 r545, r173, r445;
}
{
add.f16x2 r548, r176, r451;
}
{
sub.f16x2 r551, r173, r445;
}
{
sub.f16x2 r554, r176, r451;
}
{
add.f16x2 r557, r185, r461;
}
{
add.f16x2 r560, r188, r467;
}
{
sub.f16x2 r563, r185, r461;
}
{
sub.f16x2 r566, r188, r467;
}
{
add.f16x2 r569, r155, r471;
}
{
add.f16x2 r572, r158, r351;
}
{
sub.f16x2 r575, r155, r471;
}
{
sub.f16x2 r578, r158, r351;
}
{
add.f16x2 r581, r167, r479;
}
{
add.f16x2 r584, r170, r485;
}
{
sub.f16x2 r587, r167, r479;
}
{
sub.f16x2 r590, r170, r485;
}
{
add.f16x2 r593, r179, r495;
}
{
add.f16x2 r596, r182, r501;
}
{
sub.f16x2 r599, r179, r495;
}
{
sub.f16x2 r602, r182, r501;
}
{
add.f16x2 r605, r191, r511;
}
{
add.f16x2 r608, r194, r517;
}
{
sub.f16x2 r611, r191, r511;
}
{
sub.f16x2 r614, r194, r517;
}
{
add.f16x2 r617, %68, %124;
}
{
add.f16x2 r620, %104, %95;
}
{
sub.f16x2 r623, %68, %124;
}
{
sub.f16x2 r626, %104, %95;
}
{
add.f16x2 r629, %86, %76;
}
{
add.f16x2 r632, %121, %114;
}
{
sub.f16x2 r635, %86, %76;
}
{
sub.f16x2 r638, %121, %114;
}
{
neg.f16x2 r641, r638;
}
{
add.f16x2 r643, r617, r629;
}
{
add.f16x2 r646, r620, r632;
}
{
sub.f16x2 r649, r617, r629;
}
{
sub.f16x2 r652, r620, r632;
}
{
add.f16x2 r655, r623, r641;
}
{
add.f16x2 r658, r626, r635;
}
{
sub.f16x2 r661, r623, r641;
}
{
sub.f16x2 r664, r626, r635;
}
{
add.f16x2 r667, %118, %110;
}
{
add.f16x2 r670, %90, %80;
}
{
sub.f16x2 r673, %118, %110;
}
{
sub.f16x2 r676, %90, %80;
}
{
add.f16x2 r679, %70, %127;
}
{
add.f16x2 r682, %107, %99;
}
{
sub.f16x2 r685, %70, %127;
}
{
sub.f16x2 r688, %107, %99;
}
{
neg.f16x2 r691, r688;
}
{
add.f16x2 r693, r667, r679;
}
{
add.f16x2 r696, r670, r682;
}
{
sub.f16x2 r699, r667, r679;
}
{
sub.f16x2 r702, r670, r682;
}
{
add.f16x2 r705, r673, r691;
}
{
add.f16x2 r708, r676, r685;
}
{
sub.f16x2 r711, r673, r691;
}
{
sub.f16x2 r714, r676, r685;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r717, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r718, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r721, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r722, {low, high};
}
{
mul.f16x2 r731, r705, r717;
}
{
mul.f16x2 r734, r708, r718;
}
{
sub.f16x2 r737, r731, r734;
}
{
mul.f16x2 r740, r705, r718;
}
{
fma.rn.f16x2 r743, r708, r717, r740;
}
{
neg.f16x2 r747, r702;
}
{
mul.f16x2 r749, r711, r721;
}
{
mul.f16x2 r752, r714, r722;
}
{
sub.f16x2 r755, r749, r752;
}
{
mul.f16x2 r758, r711, r722;
}
{
fma.rn.f16x2 r761, r714, r721, r758;
}
{
add.f16x2 r765, r643, r693;
}
{
add.f16x2 r768, r646, r696;
}
{
sub.f16x2 r771, r643, r693;
}
{
sub.f16x2 r774, r646, r696;
}
{
add.f16x2 r777, r655, r737;
}
{
add.f16x2 r780, r658, r743;
}
{
sub.f16x2 r783, r655, r737;
}
{
sub.f16x2 r786, r658, r743;
}
{
add.f16x2 r789, r649, r747;
}
{
add.f16x2 r792, r652, r699;
}
{
sub.f16x2 r795, r649, r747;
}
{
sub.f16x2 r798, r652, r699;
}
{
add.f16x2 r801, r661, r755;
}
{
add.f16x2 r804, r664, r761;
}
{
sub.f16x2 r807, r661, r755;
}
{
sub.f16x2 r810, r664, r761;
}
{
add.f16x2 r813, %109, %101;
}
{
add.f16x2 r816, %79, %71;
}
{
sub.f16x2 r819, %109, %101;
}
{
sub.f16x2 r822, %79, %71;
}
{
add.f16x2 r825, %126, %116;
}
{
add.f16x2 r828, %98, %88;
}
{
sub.f16x2 r831, %126, %116;
}
{
sub.f16x2 r834, %98, %88;
}
{
neg.f16x2 r837, r834;
}
{
add.f16x2 r839, r813, r825;
}
{
add.f16x2 r842, r816, r828;
}
{
sub.f16x2 r845, r813, r825;
}
{
sub.f16x2 r848, r816, r828;
}
{
add.f16x2 r851, r819, r837;
}
{
add.f16x2 r854, r822, r831;
}
{
sub.f16x2 r857, r819, r837;
}
{
sub.f16x2 r860, r822, r831;
}
{
add.f16x2 r863, %92, %84;
}
{
add.f16x2 r866, %65, %120;
}
{
sub.f16x2 r869, %92, %84;
}
{
sub.f16x2 r872, %65, %120;
}
{
add.f16x2 r875, %112, %102;
}
{
add.f16x2 r878, %82, %74;
}
{
sub.f16x2 r881, %112, %102;
}
{
sub.f16x2 r884, %82, %74;
}
{
neg.f16x2 r887, r884;
}
{
add.f16x2 r889, r863, r875;
}
{
add.f16x2 r892, r866, r878;
}
{
sub.f16x2 r895, r863, r875;
}
{
sub.f16x2 r898, r866, r878;
}
{
add.f16x2 r901, r869, r887;
}
{
add.f16x2 r904, r872, r881;
}
{
sub.f16x2 r907, r869, r887;
}
{
sub.f16x2 r910, r872, r881;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r913, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r914, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r917, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r918, {low, high};
}
{
mul.f16x2 r927, r901, r913;
}
{
mul.f16x2 r930, r904, r914;
}
{
sub.f16x2 r933, r927, r930;
}
{
mul.f16x2 r936, r901, r914;
}
{
fma.rn.f16x2 r939, r904, r913, r936;
}
{
neg.f16x2 r943, r898;
}
{
mul.f16x2 r945, r907, r917;
}
{
mul.f16x2 r948, r910, r918;
}
{
sub.f16x2 r951, r945, r948;
}
{
mul.f16x2 r954, r907, r918;
}
{
fma.rn.f16x2 r957, r910, r917, r954;
}
{
add.f16x2 r961, r839, r889;
}
{
add.f16x2 r964, r842, r892;
}
{
sub.f16x2 r967, r839, r889;
}
{
sub.f16x2 r970, r842, r892;
}
{
add.f16x2 r973, r851, r933;
}
{
add.f16x2 r976, r854, r939;
}
{
sub.f16x2 r979, r851, r933;
}
{
sub.f16x2 r982, r854, r939;
}
{
add.f16x2 r985, r845, r943;
}
{
add.f16x2 r988, r848, r895;
}
{
sub.f16x2 r991, r845, r943;
}
{
sub.f16x2 r994, r848, r895;
}
{
add.f16x2 r997, r857, r951;
}
{
add.f16x2 r1000, r860, r957;
}
{
sub.f16x2 r1003, r857, r951;
}
{
sub.f16x2 r1006, r860, r957;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r1009, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r1010, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r1011, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r1012, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r1013, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r1014, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1017, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r1018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r1019, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r1020, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1042;
cvt.rn.f16.f32 high, f1042;
mov.b32 r1021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r1022, {low, high};
}
{
mul.f16x2 r1039, r973, r1009;
}
{
mul.f16x2 r1042, r976, r1010;
}
{
sub.f16x2 r1045, r1039, r1042;
}
{
mul.f16x2 r1048, r973, r1010;
}
{
fma.rn.f16x2 r1051, r976, r1009, r1048;
}
{
mul.f16x2 r1055, r985, r1011;
}
{
mul.f16x2 r1058, r988, r1012;
}
{
sub.f16x2 r1061, r1055, r1058;
}
{
mul.f16x2 r1064, r985, r1012;
}
{
fma.rn.f16x2 r1067, r988, r1011, r1064;
}
{
mul.f16x2 r1071, r997, r1013;
}
{
mul.f16x2 r1074, r1000, r1014;
}
{
sub.f16x2 r1077, r1071, r1074;
}
{
mul.f16x2 r1080, r997, r1014;
}
{
fma.rn.f16x2 r1083, r1000, r1013, r1080;
}
{
neg.f16x2 r1087, r970;
}
{
mul.f16x2 r1089, r979, r1017;
}
{
mul.f16x2 r1092, r982, r1018;
}
{
sub.f16x2 r1095, r1089, r1092;
}
{
mul.f16x2 r1098, r979, r1018;
}
{
fma.rn.f16x2 r1101, r982, r1017, r1098;
}
{
mul.f16x2 r1105, r991, r1019;
}
{
mul.f16x2 r1108, r994, r1020;
}
{
sub.f16x2 r1111, r1105, r1108;
}
{
mul.f16x2 r1114, r991, r1020;
}
{
fma.rn.f16x2 r1117, r994, r1019, r1114;
}
{
mul.f16x2 r1121, r1003, r1021;
}
{
mul.f16x2 r1124, r1006, r1022;
}
{
sub.f16x2 r1127, r1121, r1124;
}
{
mul.f16x2 r1130, r1003, r1022;
}
{
fma.rn.f16x2 r1133, r1006, r1021, r1130;
}
{
add.f16x2 r1137, r765, r961;
}
{
add.f16x2 r1140, r768, r964;
}
{
sub.f16x2 r1143, r765, r961;
}
{
sub.f16x2 r1146, r768, r964;
}
{
add.f16x2 r1149, r777, r1045;
}
{
add.f16x2 r1152, r780, r1051;
}
{
sub.f16x2 r1155, r777, r1045;
}
{
sub.f16x2 r1158, r780, r1051;
}
{
add.f16x2 r1161, r789, r1061;
}
{
add.f16x2 r1164, r792, r1067;
}
{
sub.f16x2 r1167, r789, r1061;
}
{
sub.f16x2 r1170, r792, r1067;
}
{
add.f16x2 r1173, r801, r1077;
}
{
add.f16x2 r1176, r804, r1083;
}
{
sub.f16x2 r1179, r801, r1077;
}
{
sub.f16x2 r1182, r804, r1083;
}
{
add.f16x2 r1185, r771, r1087;
}
{
add.f16x2 r1188, r774, r967;
}
{
sub.f16x2 r1191, r771, r1087;
}
{
sub.f16x2 r1194, r774, r967;
}
{
add.f16x2 r1197, r783, r1095;
}
{
add.f16x2 r1200, r786, r1101;
}
{
sub.f16x2 r1203, r783, r1095;
}
{
sub.f16x2 r1206, r786, r1101;
}
{
add.f16x2 r1209, r795, r1111;
}
{
add.f16x2 r1212, r798, r1117;
}
{
sub.f16x2 r1215, r795, r1111;
}
{
sub.f16x2 r1218, r798, r1117;
}
{
add.f16x2 r1221, r807, r1127;
}
{
add.f16x2 r1224, r810, r1133;
}
{
sub.f16x2 r1227, r807, r1127;
}
{
sub.f16x2 r1230, r810, r1133;
}
mov.f32 f690, 0f3F7B14BE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f690;
cvt.rn.f16.f32 high, f690;
mov.b32 r1233, {low, high};
}
mov.f32 f714, 0f3E47C5C2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f714;
cvt.rn.f16.f32 high, f714;
mov.b32 r1234, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r1235, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r1236, {low, high};
}
mov.f32 f698, 0f3F54DB31;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f698;
cvt.rn.f16.f32 high, f698;
mov.b32 r1237, {low, high};
}
mov.f32 f706, 0f3F0E39DA;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f706;
cvt.rn.f16.f32 high, f706;
mov.b32 r1238, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r1239, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r1240, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f706;
cvt.rn.f16.f32 high, f706;
mov.b32 r1241, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f698;
cvt.rn.f16.f32 high, f698;
mov.b32 r1242, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r1243, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r1244, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f714;
cvt.rn.f16.f32 high, f714;
mov.b32 r1245, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f690;
cvt.rn.f16.f32 high, f690;
mov.b32 r1246, {low, high};
}
mov.f32 f688, 0fBE47C5C2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f688;
cvt.rn.f16.f32 high, f688;
mov.b32 r1249, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f690;
cvt.rn.f16.f32 high, f690;
mov.b32 r1250, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1251, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r1252, {low, high};
}
mov.f32 f696, 0fBF0E39DA;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f696;
cvt.rn.f16.f32 high, f696;
mov.b32 r1253, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f698;
cvt.rn.f16.f32 high, f698;
mov.b32 r1254, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r1255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r1256, {low, high};
}
mov.f32 f704, 0fBF54DB31;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f704;
cvt.rn.f16.f32 high, f704;
mov.b32 r1257, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f706;
cvt.rn.f16.f32 high, f706;
mov.b32 r1258, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1042;
cvt.rn.f16.f32 high, f1042;
mov.b32 r1259, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r1260, {low, high};
}
mov.f32 f712, 0fBF7B14BE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f712;
cvt.rn.f16.f32 high, f712;
mov.b32 r1261, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f714;
cvt.rn.f16.f32 high, f714;
mov.b32 r1262, {low, high};
}
{
mul.f16x2 r1295, r1149, r1233;
}
{
mul.f16x2 r1298, r1152, r1234;
}
{
sub.f16x2 r1301, r1295, r1298;
}
{
mul.f16x2 r1304, r1149, r1234;
}
{
fma.rn.f16x2 r1307, r1152, r1233, r1304;
}
{
mul.f16x2 r1311, r1161, r1235;
}
{
mul.f16x2 r1314, r1164, r1236;
}
{
sub.f16x2 r1317, r1311, r1314;
}
{
mul.f16x2 r1320, r1161, r1236;
}
{
fma.rn.f16x2 r1323, r1164, r1235, r1320;
}
{
mul.f16x2 r1327, r1173, r1237;
}
{
mul.f16x2 r1330, r1176, r1238;
}
{
sub.f16x2 r1333, r1327, r1330;
}
{
mul.f16x2 r1336, r1173, r1238;
}
{
fma.rn.f16x2 r1339, r1176, r1237, r1336;
}
{
mul.f16x2 r1343, r1185, r1239;
}
{
mul.f16x2 r1346, r1188, r1240;
}
{
sub.f16x2 r1349, r1343, r1346;
}
{
mul.f16x2 r1352, r1185, r1240;
}
{
fma.rn.f16x2 r1355, r1188, r1239, r1352;
}
{
mul.f16x2 r1359, r1197, r1241;
}
{
mul.f16x2 r1362, r1200, r1242;
}
{
sub.f16x2 r1365, r1359, r1362;
}
{
mul.f16x2 r1368, r1197, r1242;
}
{
fma.rn.f16x2 r1371, r1200, r1241, r1368;
}
{
mul.f16x2 r1375, r1209, r1243;
}
{
mul.f16x2 r1378, r1212, r1244;
}
{
sub.f16x2 r1381, r1375, r1378;
}
{
mul.f16x2 r1384, r1209, r1244;
}
{
fma.rn.f16x2 r1387, r1212, r1243, r1384;
}
{
mul.f16x2 r1391, r1221, r1245;
}
{
mul.f16x2 r1394, r1224, r1246;
}
{
sub.f16x2 r1397, r1391, r1394;
}
{
mul.f16x2 r1400, r1221, r1246;
}
{
fma.rn.f16x2 r1403, r1224, r1245, r1400;
}
{
neg.f16x2 r1407, r1146;
}
{
mul.f16x2 r1409, r1155, r1249;
}
{
mul.f16x2 r1412, r1158, r1250;
}
{
sub.f16x2 r1415, r1409, r1412;
}
{
mul.f16x2 r1418, r1155, r1250;
}
{
fma.rn.f16x2 r1421, r1158, r1249, r1418;
}
{
mul.f16x2 r1425, r1167, r1251;
}
{
mul.f16x2 r1428, r1170, r1252;
}
{
sub.f16x2 r1431, r1425, r1428;
}
{
mul.f16x2 r1434, r1167, r1252;
}
{
fma.rn.f16x2 r1437, r1170, r1251, r1434;
}
{
mul.f16x2 r1441, r1179, r1253;
}
{
mul.f16x2 r1444, r1182, r1254;
}
{
sub.f16x2 r1447, r1441, r1444;
}
{
mul.f16x2 r1450, r1179, r1254;
}
{
fma.rn.f16x2 r1453, r1182, r1253, r1450;
}
{
mul.f16x2 r1457, r1191, r1255;
}
{
mul.f16x2 r1460, r1194, r1256;
}
{
sub.f16x2 r1463, r1457, r1460;
}
{
mul.f16x2 r1466, r1191, r1256;
}
{
fma.rn.f16x2 r1469, r1194, r1255, r1466;
}
{
mul.f16x2 r1473, r1203, r1257;
}
{
mul.f16x2 r1476, r1206, r1258;
}
{
sub.f16x2 r1479, r1473, r1476;
}
{
mul.f16x2 r1482, r1203, r1258;
}
{
fma.rn.f16x2 r1485, r1206, r1257, r1482;
}
{
mul.f16x2 r1489, r1215, r1259;
}
{
mul.f16x2 r1492, r1218, r1260;
}
{
sub.f16x2 r1495, r1489, r1492;
}
{
mul.f16x2 r1498, r1215, r1260;
}
{
fma.rn.f16x2 r1501, r1218, r1259, r1498;
}
{
mul.f16x2 r1505, r1227, r1261;
}
{
mul.f16x2 r1508, r1230, r1262;
}
{
sub.f16x2 r1511, r1505, r1508;
}
{
mul.f16x2 r1514, r1227, r1262;
}
{
fma.rn.f16x2 r1517, r1230, r1261, r1514;
}
{
add.f16x2 r1521, r521, r1137;
}
{
add.f16x2 r1524, r524, r1140;
}
{
sub.f16x2 r1527, r521, r1137;
}
{
sub.f16x2 r1530, r524, r1140;
}
{
add.f16x2 r1533, r533, r1301;
}
{
add.f16x2 r1536, r536, r1307;
}
{
sub.f16x2 r1539, r533, r1301;
}
{
sub.f16x2 r1542, r536, r1307;
}
{
add.f16x2 r1545, r545, r1317;
}
{
add.f16x2 r1548, r548, r1323;
}
{
sub.f16x2 r1551, r545, r1317;
}
{
sub.f16x2 r1554, r548, r1323;
}
{
add.f16x2 r1557, r557, r1333;
}
{
add.f16x2 r1560, r560, r1339;
}
{
sub.f16x2 r1563, r557, r1333;
}
{
sub.f16x2 r1566, r560, r1339;
}
{
add.f16x2 r1569, r569, r1349;
}
{
add.f16x2 r1572, r572, r1355;
}
{
sub.f16x2 r1575, r569, r1349;
}
{
sub.f16x2 r1578, r572, r1355;
}
{
add.f16x2 r1581, r581, r1365;
}
{
add.f16x2 r1584, r584, r1371;
}
{
sub.f16x2 r1587, r581, r1365;
}
{
sub.f16x2 r1590, r584, r1371;
}
{
add.f16x2 r1593, r593, r1381;
}
{
add.f16x2 r1596, r596, r1387;
}
{
sub.f16x2 r1599, r593, r1381;
}
{
sub.f16x2 r1602, r596, r1387;
}
{
add.f16x2 r1605, r605, r1397;
}
{
add.f16x2 r1608, r608, r1403;
}
{
sub.f16x2 r1611, r605, r1397;
}
{
sub.f16x2 r1614, r608, r1403;
}
{
add.f16x2 r1617, r527, r1407;
}
{
add.f16x2 r1620, r530, r1143;
}
{
sub.f16x2 r1623, r527, r1407;
}
{
sub.f16x2 r1626, r530, r1143;
}
{
add.f16x2 r1629, r539, r1415;
}
{
add.f16x2 r1632, r542, r1421;
}
{
sub.f16x2 r1635, r539, r1415;
}
{
sub.f16x2 r1638, r542, r1421;
}
{
add.f16x2 r1641, r551, r1431;
}
{
add.f16x2 r1644, r554, r1437;
}
{
sub.f16x2 r1647, r551, r1431;
}
{
sub.f16x2 r1650, r554, r1437;
}
{
add.f16x2 r1653, r563, r1447;
}
{
add.f16x2 r1656, r566, r1453;
}
{
sub.f16x2 r1659, r563, r1447;
}
{
sub.f16x2 r1662, r566, r1453;
}
{
add.f16x2 r1665, r575, r1463;
}
{
add.f16x2 r1668, r578, r1469;
}
{
sub.f16x2 r1671, r575, r1463;
}
{
sub.f16x2 r1674, r578, r1469;
}
{
add.f16x2 r1677, r587, r1479;
}
{
add.f16x2 r1680, r590, r1485;
}
{
sub.f16x2 r1683, r587, r1479;
}
{
sub.f16x2 r1686, r590, r1485;
}
{
add.f16x2 r1689, r599, r1495;
}
{
add.f16x2 r1692, r602, r1501;
}
{
sub.f16x2 r1695, r599, r1495;
}
{
sub.f16x2 r1698, r602, r1501;
}
{
add.f16x2 r1701, r611, r1511;
}
{
add.f16x2 r1704, r614, r1517;
}
{
sub.f16x2 r1707, r611, r1511;
}
{
sub.f16x2 r1710, r614, r1517;
}
shl.b32 r6963, r6961, 8;
and.b32 r6964, r6963, -131072;
add.s32 r6965, r6960, r6964;
and.b32 r6978, r6961, 511;
cvt.rn.f32.u32 f1077, r6978;
mul.f32 f1078, f1077, 0f39C90FDB;
cos.approx.f32 f357, f1078;
sin.approx.f32 f1079, f1078;
neg.f32 f358, f1079;
mov.f32 f1084, 0f3F800000;
mov.f32 f1083, 0fBF800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f357;
cvt.rn.f16.f32 high, f358;
mov.b32 r1713, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1716, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1718, {high, high};
}
{
mul.f16x2 r1720, r1536, r1718;
}
{
fma.rn.f16x2 r1723, r1533, r1716, r1720;
}
{
mul.f16x2 r1727, r1533, r1718;
}
{
neg.f16x2 r1730, r1727;
}
{
fma.rn.f16x2 r1732, r1536, r1716, r1730;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1736, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1738, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r1740, {low, high};
}
{
mul.f16x2 r1741, r1738, r1740;
}
{
mul.f16x2 r1744, r1713, r1736;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1747, {high, low};
}
{
fma.rn.f16x2 r1749, r1741, r1747, r1744;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1749;
mov.b32 r1753, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1749;
mov.b32 r1755, {high, high};
}
{
mul.f16x2 r1757, r1548, r1755;
}
{
fma.rn.f16x2 r1760, r1545, r1753, r1757;
}
{
mul.f16x2 r1764, r1545, r1755;
}
{
neg.f16x2 r1767, r1764;
}
{
fma.rn.f16x2 r1769, r1548, r1753, r1767;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1773, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1775, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r1777, {low, high};
}
{
mul.f16x2 r1778, r1775, r1777;
}
{
mul.f16x2 r1781, r1749, r1773;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1749;
mov.b32 r1784, {high, low};
}
{
fma.rn.f16x2 r1786, r1778, r1784, r1781;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1786;
mov.b32 r1790, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1786;
mov.b32 r1792, {high, high};
}
{
mul.f16x2 r1794, r1560, r1792;
}
{
fma.rn.f16x2 r1797, r1557, r1790, r1794;
}
{
mul.f16x2 r1801, r1557, r1792;
}
{
neg.f16x2 r1804, r1801;
}
{
fma.rn.f16x2 r1806, r1560, r1790, r1804;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1810, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1812, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r1814, {low, high};
}
{
mul.f16x2 r1815, r1812, r1814;
}
{
mul.f16x2 r1818, r1786, r1810;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1786;
mov.b32 r1821, {high, low};
}
{
fma.rn.f16x2 r1823, r1815, r1821, r1818;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1823;
mov.b32 r1827, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1823;
mov.b32 r1829, {high, high};
}
{
mul.f16x2 r1831, r1572, r1829;
}
{
fma.rn.f16x2 r1834, r1569, r1827, r1831;
}
{
mul.f16x2 r1838, r1569, r1829;
}
{
neg.f16x2 r1841, r1838;
}
{
fma.rn.f16x2 r1843, r1572, r1827, r1841;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1847, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1849, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r1851, {low, high};
}
{
mul.f16x2 r1852, r1849, r1851;
}
{
mul.f16x2 r1855, r1823, r1847;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1823;
mov.b32 r1858, {high, low};
}
{
fma.rn.f16x2 r1860, r1852, r1858, r1855;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1860;
mov.b32 r1864, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1860;
mov.b32 r1866, {high, high};
}
{
mul.f16x2 r1868, r1584, r1866;
}
{
fma.rn.f16x2 r1871, r1581, r1864, r1868;
}
{
mul.f16x2 r1875, r1581, r1866;
}
{
neg.f16x2 r1878, r1875;
}
{
fma.rn.f16x2 r1880, r1584, r1864, r1878;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1884, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1886, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r1888, {low, high};
}
{
mul.f16x2 r1889, r1886, r1888;
}
{
mul.f16x2 r1892, r1860, r1884;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1860;
mov.b32 r1895, {high, low};
}
{
fma.rn.f16x2 r1897, r1889, r1895, r1892;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1897;
mov.b32 r1901, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1897;
mov.b32 r1903, {high, high};
}
{
mul.f16x2 r1905, r1596, r1903;
}
{
fma.rn.f16x2 r1908, r1593, r1901, r1905;
}
{
mul.f16x2 r1912, r1593, r1903;
}
{
neg.f16x2 r1915, r1912;
}
{
fma.rn.f16x2 r1917, r1596, r1901, r1915;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1921, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1923, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r1925, {low, high};
}
{
mul.f16x2 r1926, r1923, r1925;
}
{
mul.f16x2 r1929, r1897, r1921;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1897;
mov.b32 r1932, {high, low};
}
{
fma.rn.f16x2 r1934, r1926, r1932, r1929;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1934;
mov.b32 r1938, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1934;
mov.b32 r1940, {high, high};
}
{
mul.f16x2 r1942, r1608, r1940;
}
{
fma.rn.f16x2 r1945, r1605, r1938, r1942;
}
{
mul.f16x2 r1949, r1605, r1940;
}
{
neg.f16x2 r1952, r1949;
}
{
fma.rn.f16x2 r1954, r1608, r1938, r1952;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1958, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1960, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r1962, {low, high};
}
{
mul.f16x2 r1963, r1960, r1962;
}
{
mul.f16x2 r1966, r1934, r1958;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1934;
mov.b32 r1969, {high, low};
}
{
fma.rn.f16x2 r1971, r1963, r1969, r1966;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1971;
mov.b32 r1975, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1971;
mov.b32 r1977, {high, high};
}
{
mul.f16x2 r1979, r1620, r1977;
}
{
fma.rn.f16x2 r1982, r1617, r1975, r1979;
}
{
mul.f16x2 r1986, r1617, r1977;
}
{
neg.f16x2 r1989, r1986;
}
{
fma.rn.f16x2 r1991, r1620, r1975, r1989;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1995, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1997, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r1999, {low, high};
}
{
mul.f16x2 r2000, r1997, r1999;
}
{
mul.f16x2 r2003, r1971, r1995;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1971;
mov.b32 r2006, {high, low};
}
{
fma.rn.f16x2 r2008, r2000, r2006, r2003;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2008;
mov.b32 r2012, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2008;
mov.b32 r2014, {high, high};
}
{
mul.f16x2 r2016, r1632, r2014;
}
{
fma.rn.f16x2 r2019, r1629, r2012, r2016;
}
{
mul.f16x2 r2023, r1629, r2014;
}
{
neg.f16x2 r2026, r2023;
}
{
fma.rn.f16x2 r2028, r1632, r2012, r2026;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2032, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2034, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2036, {low, high};
}
{
mul.f16x2 r2037, r2034, r2036;
}
{
mul.f16x2 r2040, r2008, r2032;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2008;
mov.b32 r2043, {high, low};
}
{
fma.rn.f16x2 r2045, r2037, r2043, r2040;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2045;
mov.b32 r2049, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2045;
mov.b32 r2051, {high, high};
}
{
mul.f16x2 r2053, r1644, r2051;
}
{
fma.rn.f16x2 r2056, r1641, r2049, r2053;
}
{
mul.f16x2 r2060, r1641, r2051;
}
{
neg.f16x2 r2063, r2060;
}
{
fma.rn.f16x2 r2065, r1644, r2049, r2063;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2069, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2071, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2073, {low, high};
}
{
mul.f16x2 r2074, r2071, r2073;
}
{
mul.f16x2 r2077, r2045, r2069;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2045;
mov.b32 r2080, {high, low};
}
{
fma.rn.f16x2 r2082, r2074, r2080, r2077;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2082;
mov.b32 r2086, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2082;
mov.b32 r2088, {high, high};
}
{
mul.f16x2 r2090, r1656, r2088;
}
{
fma.rn.f16x2 r2093, r1653, r2086, r2090;
}
{
mul.f16x2 r2097, r1653, r2088;
}
{
neg.f16x2 r2100, r2097;
}
{
fma.rn.f16x2 r2102, r1656, r2086, r2100;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2106, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2108, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2110, {low, high};
}
{
mul.f16x2 r2111, r2108, r2110;
}
{
mul.f16x2 r2114, r2082, r2106;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2082;
mov.b32 r2117, {high, low};
}
{
fma.rn.f16x2 r2119, r2111, r2117, r2114;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2119;
mov.b32 r2123, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2119;
mov.b32 r2125, {high, high};
}
{
mul.f16x2 r2127, r1668, r2125;
}
{
fma.rn.f16x2 r2130, r1665, r2123, r2127;
}
{
mul.f16x2 r2134, r1665, r2125;
}
{
neg.f16x2 r2137, r2134;
}
{
fma.rn.f16x2 r2139, r1668, r2123, r2137;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2143, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2145, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2147, {low, high};
}
{
mul.f16x2 r2148, r2145, r2147;
}
{
mul.f16x2 r2151, r2119, r2143;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2119;
mov.b32 r2154, {high, low};
}
{
fma.rn.f16x2 r2156, r2148, r2154, r2151;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2156;
mov.b32 r2160, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2156;
mov.b32 r2162, {high, high};
}
{
mul.f16x2 r2164, r1680, r2162;
}
{
fma.rn.f16x2 r2167, r1677, r2160, r2164;
}
{
mul.f16x2 r2171, r1677, r2162;
}
{
neg.f16x2 r2174, r2171;
}
{
fma.rn.f16x2 r2176, r1680, r2160, r2174;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2180, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2182, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2184, {low, high};
}
{
mul.f16x2 r2185, r2182, r2184;
}
{
mul.f16x2 r2188, r2156, r2180;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2156;
mov.b32 r2191, {high, low};
}
{
fma.rn.f16x2 r2193, r2185, r2191, r2188;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2193;
mov.b32 r2197, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2193;
mov.b32 r2199, {high, high};
}
{
mul.f16x2 r2201, r1692, r2199;
}
{
fma.rn.f16x2 r2204, r1689, r2197, r2201;
}
{
mul.f16x2 r2208, r1689, r2199;
}
{
neg.f16x2 r2211, r2208;
}
{
fma.rn.f16x2 r2213, r1692, r2197, r2211;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2217, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2219, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2221, {low, high};
}
{
mul.f16x2 r2222, r2219, r2221;
}
{
mul.f16x2 r2225, r2193, r2217;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2193;
mov.b32 r2228, {high, low};
}
{
fma.rn.f16x2 r2230, r2222, r2228, r2225;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2230;
mov.b32 r2234, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2230;
mov.b32 r2236, {high, high};
}
{
mul.f16x2 r2238, r1704, r2236;
}
{
fma.rn.f16x2 r2241, r1701, r2234, r2238;
}
{
mul.f16x2 r2245, r1701, r2236;
}
{
neg.f16x2 r2248, r2245;
}
{
fma.rn.f16x2 r2250, r1704, r2234, r2248;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2254, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2256, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2258, {low, high};
}
{
mul.f16x2 r2259, r2256, r2258;
}
{
mul.f16x2 r2262, r2230, r2254;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2230;
mov.b32 r2265, {high, low};
}
{
fma.rn.f16x2 r2267, r2259, r2265, r2262;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2267;
mov.b32 r2271, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2267;
mov.b32 r2273, {high, high};
}
{
mul.f16x2 r2275, r1530, r2273;
}
{
fma.rn.f16x2 r2278, r1527, r2271, r2275;
}
{
mul.f16x2 r2282, r1527, r2273;
}
{
neg.f16x2 r2285, r2282;
}
{
fma.rn.f16x2 r2287, r1530, r2271, r2285;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2291, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2293, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2295, {low, high};
}
{
mul.f16x2 r2296, r2293, r2295;
}
{
mul.f16x2 r2299, r2267, r2291;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2267;
mov.b32 r2302, {high, low};
}
{
fma.rn.f16x2 r2304, r2296, r2302, r2299;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2304;
mov.b32 r2308, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2304;
mov.b32 r2310, {high, high};
}
{
mul.f16x2 r2312, r1542, r2310;
}
{
fma.rn.f16x2 r2315, r1539, r2308, r2312;
}
{
mul.f16x2 r2319, r1539, r2310;
}
{
neg.f16x2 r2322, r2319;
}
{
fma.rn.f16x2 r2324, r1542, r2308, r2322;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2328, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2330, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2332, {low, high};
}
{
mul.f16x2 r2333, r2330, r2332;
}
{
mul.f16x2 r2336, r2304, r2328;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2304;
mov.b32 r2339, {high, low};
}
{
fma.rn.f16x2 r2341, r2333, r2339, r2336;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2341;
mov.b32 r2345, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2341;
mov.b32 r2347, {high, high};
}
{
mul.f16x2 r2349, r1554, r2347;
}
{
fma.rn.f16x2 r2352, r1551, r2345, r2349;
}
{
mul.f16x2 r2356, r1551, r2347;
}
{
neg.f16x2 r2359, r2356;
}
{
fma.rn.f16x2 r2361, r1554, r2345, r2359;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2365, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2367, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2369, {low, high};
}
{
mul.f16x2 r2370, r2367, r2369;
}
{
mul.f16x2 r2373, r2341, r2365;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2341;
mov.b32 r2376, {high, low};
}
{
fma.rn.f16x2 r2378, r2370, r2376, r2373;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2378;
mov.b32 r2382, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2378;
mov.b32 r2384, {high, high};
}
{
mul.f16x2 r2386, r1566, r2384;
}
{
fma.rn.f16x2 r2389, r1563, r2382, r2386;
}
{
mul.f16x2 r2393, r1563, r2384;
}
{
neg.f16x2 r2396, r2393;
}
{
fma.rn.f16x2 r2398, r1566, r2382, r2396;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2402, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2404, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2406, {low, high};
}
{
mul.f16x2 r2407, r2404, r2406;
}
{
mul.f16x2 r2410, r2378, r2402;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2378;
mov.b32 r2413, {high, low};
}
{
fma.rn.f16x2 r2415, r2407, r2413, r2410;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2415;
mov.b32 r2419, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2415;
mov.b32 r2421, {high, high};
}
{
mul.f16x2 r2423, r1578, r2421;
}
{
fma.rn.f16x2 r2426, r1575, r2419, r2423;
}
{
mul.f16x2 r2430, r1575, r2421;
}
{
neg.f16x2 r2433, r2430;
}
{
fma.rn.f16x2 r2435, r1578, r2419, r2433;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2439, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2441, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2443, {low, high};
}
{
mul.f16x2 r2444, r2441, r2443;
}
{
mul.f16x2 r2447, r2415, r2439;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2415;
mov.b32 r2450, {high, low};
}
{
fma.rn.f16x2 r2452, r2444, r2450, r2447;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2452;
mov.b32 r2456, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2452;
mov.b32 r2458, {high, high};
}
{
mul.f16x2 r2460, r1590, r2458;
}
{
fma.rn.f16x2 r2463, r1587, r2456, r2460;
}
{
mul.f16x2 r2467, r1587, r2458;
}
{
neg.f16x2 r2470, r2467;
}
{
fma.rn.f16x2 r2472, r1590, r2456, r2470;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2476, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2478, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2480, {low, high};
}
{
mul.f16x2 r2481, r2478, r2480;
}
{
mul.f16x2 r2484, r2452, r2476;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2452;
mov.b32 r2487, {high, low};
}
{
fma.rn.f16x2 r2489, r2481, r2487, r2484;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2489;
mov.b32 r2493, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2489;
mov.b32 r2495, {high, high};
}
{
mul.f16x2 r2497, r1602, r2495;
}
{
fma.rn.f16x2 r2500, r1599, r2493, r2497;
}
{
mul.f16x2 r2504, r1599, r2495;
}
{
neg.f16x2 r2507, r2504;
}
{
fma.rn.f16x2 r2509, r1602, r2493, r2507;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2513, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2515, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2517, {low, high};
}
{
mul.f16x2 r2518, r2515, r2517;
}
{
mul.f16x2 r2521, r2489, r2513;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2489;
mov.b32 r2524, {high, low};
}
{
fma.rn.f16x2 r2526, r2518, r2524, r2521;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2526;
mov.b32 r2530, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2526;
mov.b32 r2532, {high, high};
}
{
mul.f16x2 r2534, r1614, r2532;
}
{
fma.rn.f16x2 r2537, r1611, r2530, r2534;
}
{
mul.f16x2 r2541, r1611, r2532;
}
{
neg.f16x2 r2544, r2541;
}
{
fma.rn.f16x2 r2546, r1614, r2530, r2544;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2550, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2552, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2554, {low, high};
}
{
mul.f16x2 r2555, r2552, r2554;
}
{
mul.f16x2 r2558, r2526, r2550;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2526;
mov.b32 r2561, {high, low};
}
{
fma.rn.f16x2 r2563, r2555, r2561, r2558;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2563;
mov.b32 r2567, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2563;
mov.b32 r2569, {high, high};
}
{
mul.f16x2 r2571, r1626, r2569;
}
{
fma.rn.f16x2 r2574, r1623, r2567, r2571;
}
{
mul.f16x2 r2578, r1623, r2569;
}
{
neg.f16x2 r2581, r2578;
}
{
fma.rn.f16x2 r2583, r1626, r2567, r2581;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2587, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2589, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2591, {low, high};
}
{
mul.f16x2 r2592, r2589, r2591;
}
{
mul.f16x2 r2595, r2563, r2587;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2563;
mov.b32 r2598, {high, low};
}
{
fma.rn.f16x2 r2600, r2592, r2598, r2595;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2600;
mov.b32 r2604, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2600;
mov.b32 r2606, {high, high};
}
{
mul.f16x2 r2608, r1638, r2606;
}
{
fma.rn.f16x2 r2611, r1635, r2604, r2608;
}
{
mul.f16x2 r2615, r1635, r2606;
}
{
neg.f16x2 r2618, r2615;
}
{
fma.rn.f16x2 r2620, r1638, r2604, r2618;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2624, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2626, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2628, {low, high};
}
{
mul.f16x2 r2629, r2626, r2628;
}
{
mul.f16x2 r2632, r2600, r2624;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2600;
mov.b32 r2635, {high, low};
}
{
fma.rn.f16x2 r2637, r2629, r2635, r2632;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2637;
mov.b32 r2641, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2637;
mov.b32 r2643, {high, high};
}
{
mul.f16x2 r2645, r1650, r2643;
}
{
fma.rn.f16x2 r2648, r1647, r2641, r2645;
}
{
mul.f16x2 r2652, r1647, r2643;
}
{
neg.f16x2 r2655, r2652;
}
{
fma.rn.f16x2 r2657, r1650, r2641, r2655;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2661, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2663, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2665, {low, high};
}
{
mul.f16x2 r2666, r2663, r2665;
}
{
mul.f16x2 r2669, r2637, r2661;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2637;
mov.b32 r2672, {high, low};
}
{
fma.rn.f16x2 r2674, r2666, r2672, r2669;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2674;
mov.b32 r2678, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2674;
mov.b32 r2680, {high, high};
}
{
mul.f16x2 r2682, r1662, r2680;
}
{
fma.rn.f16x2 r2685, r1659, r2678, r2682;
}
{
mul.f16x2 r2689, r1659, r2680;
}
{
neg.f16x2 r2692, r2689;
}
{
fma.rn.f16x2 r2694, r1662, r2678, r2692;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2698, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2700, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2702, {low, high};
}
{
mul.f16x2 r2703, r2700, r2702;
}
{
mul.f16x2 r2706, r2674, r2698;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2674;
mov.b32 r2709, {high, low};
}
{
fma.rn.f16x2 r2711, r2703, r2709, r2706;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2711;
mov.b32 r2715, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2711;
mov.b32 r2717, {high, high};
}
{
mul.f16x2 r2719, r1674, r2717;
}
{
fma.rn.f16x2 r2722, r1671, r2715, r2719;
}
{
mul.f16x2 r2726, r1671, r2717;
}
{
neg.f16x2 r2729, r2726;
}
{
fma.rn.f16x2 r2731, r1674, r2715, r2729;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2735, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2737, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2739, {low, high};
}
{
mul.f16x2 r2740, r2737, r2739;
}
{
mul.f16x2 r2743, r2711, r2735;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2711;
mov.b32 r2746, {high, low};
}
{
fma.rn.f16x2 r2748, r2740, r2746, r2743;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2748;
mov.b32 r2752, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2748;
mov.b32 r2754, {high, high};
}
{
mul.f16x2 r2756, r1686, r2754;
}
{
fma.rn.f16x2 r2759, r1683, r2752, r2756;
}
{
mul.f16x2 r2763, r1683, r2754;
}
{
neg.f16x2 r2766, r2763;
}
{
fma.rn.f16x2 r2768, r1686, r2752, r2766;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2772, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2774, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2776, {low, high};
}
{
mul.f16x2 r2777, r2774, r2776;
}
{
mul.f16x2 r2780, r2748, r2772;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2748;
mov.b32 r2783, {high, low};
}
{
fma.rn.f16x2 r2785, r2777, r2783, r2780;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2785;
mov.b32 r2789, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2785;
mov.b32 r2791, {high, high};
}
{
mul.f16x2 r2793, r1698, r2791;
}
{
fma.rn.f16x2 r2796, r1695, r2789, r2793;
}
{
mul.f16x2 r2800, r1695, r2791;
}
{
neg.f16x2 r2803, r2800;
}
{
fma.rn.f16x2 r2805, r1698, r2789, r2803;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2809, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2811, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2813, {low, high};
}
{
mul.f16x2 r2814, r2811, r2813;
}
{
mul.f16x2 r2817, r2785, r2809;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2785;
mov.b32 r2820, {high, low};
}
{
fma.rn.f16x2 r2822, r2814, r2820, r2817;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2822;
mov.b32 r2826, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2822;
mov.b32 r2828, {high, high};
}
{
mul.f16x2 r2830, r1710, r2828;
}
{
fma.rn.f16x2 r2833, r1707, r2826, r2830;
}
{
mul.f16x2 r2837, r1707, r2828;
}
{
neg.f16x2 r2840, r2837;
}
{
fma.rn.f16x2 r2842, r1710, r2826, r2840;
}
barrier.sync 0;
and.b32 r6966, r6963, 130816;
add.s32 r6967, r6965, r6966;
st.shared.v4.f32 [r6967], {r1521, r1524, r1723, r1732};
st.shared.v4.f32 [r6967+16], {r1760, r1769, r1797, r1806};
st.shared.v4.f32 [r6967+32], {r1834, r1843, r1871, r1880};
st.shared.v4.f32 [r6967+48], {r1908, r1917, r1945, r1954};
st.shared.v4.f32 [r6967+64], {r1982, r1991, r2019, r2028};
st.shared.v4.f32 [r6967+80], {r2056, r2065, r2093, r2102};
st.shared.v4.f32 [r6967+96], {r2130, r2139, r2167, r2176};
st.shared.v4.f32 [r6967+112], {r2204, r2213, r2241, r2250};
st.shared.v4.f32 [r6967+128], {r2278, r2287, r2315, r2324};
st.shared.v4.f32 [r6967+144], {r2352, r2361, r2389, r2398};
st.shared.v4.f32 [r6967+160], {r2426, r2435, r2463, r2472};
st.shared.v4.f32 [r6967+176], {r2500, r2509, r2537, r2546};
st.shared.v4.f32 [r6967+192], {r2574, r2583, r2611, r2620};
st.shared.v4.f32 [r6967+208], {r2648, r2657, r2685, r2694};
st.shared.v4.f32 [r6967+224], {r2722, r2731, r2759, r2768};
st.shared.v4.f32 [r6967+240], {r2796, r2805, r2833, r2842};
barrier.sync 0;
mad.lo.s32 r6968, r6978, -248, r6967;
ld.shared.u32 r2864, [r6968];
ld.shared.u32 r2867, [r6968+4];
ld.shared.u32 r3480, [r6968+4096];
ld.shared.u32 r3483, [r6968+4100];
ld.shared.u32 r3060, [r6968+8192];
ld.shared.u32 r3063, [r6968+8196];
ld.shared.u32 r3676, [r6968+12288];
ld.shared.u32 r3679, [r6968+12292];
ld.shared.u32 r2914, [r6968+16384];
ld.shared.u32 r2917, [r6968+16388];
ld.shared.u32 r3530, [r6968+20480];
ld.shared.u32 r3533, [r6968+20484];
ld.shared.u32 r3110, [r6968+24576];
ld.shared.u32 r3113, [r6968+24580];
ld.shared.u32 r3726, [r6968+28672];
ld.shared.u32 r3729, [r6968+28676];
ld.shared.u32 r2876, [r6968+32768];
ld.shared.u32 r2879, [r6968+32772];
ld.shared.u32 r3492, [r6968+36864];
ld.shared.u32 r3495, [r6968+36868];
ld.shared.u32 r3072, [r6968+40960];
ld.shared.u32 r3075, [r6968+40964];
ld.shared.u32 r3688, [r6968+45056];
ld.shared.u32 r3691, [r6968+45060];
ld.shared.u32 r2926, [r6968+49152];
ld.shared.u32 r2929, [r6968+49156];
ld.shared.u32 r3542, [r6968+53248];
ld.shared.u32 r3545, [r6968+53252];
ld.shared.u32 r3122, [r6968+57344];
ld.shared.u32 r3125, [r6968+57348];
ld.shared.u32 r3738, [r6968+61440];
ld.shared.u32 r3741, [r6968+61444];
ld.shared.u32 r2865, [r6968+65536];
ld.shared.u32 r2868, [r6968+65540];
ld.shared.u32 r3481, [r6968+69632];
ld.shared.u32 r3484, [r6968+69636];
ld.shared.u32 r3061, [r6968+73728];
ld.shared.u32 r3064, [r6968+73732];
ld.shared.u32 r3677, [r6968+77824];
ld.shared.u32 r3680, [r6968+77828];
ld.shared.u32 r2915, [r6968+81920];
ld.shared.u32 r2918, [r6968+81924];
ld.shared.u32 r3531, [r6968+86016];
ld.shared.u32 r3534, [r6968+86020];
ld.shared.u32 r3111, [r6968+90112];
ld.shared.u32 r3114, [r6968+90116];
ld.shared.u32 r3727, [r6968+94208];
ld.shared.u32 r3730, [r6968+94212];
ld.shared.u32 r2877, [r6968+98304];
ld.shared.u32 r2880, [r6968+98308];
ld.shared.u32 r3493, [r6968+102400];
ld.shared.u32 r3496, [r6968+102404];
ld.shared.u32 r3073, [r6968+106496];
ld.shared.u32 r3076, [r6968+106500];
ld.shared.u32 r3689, [r6968+110592];
ld.shared.u32 r3692, [r6968+110596];
ld.shared.u32 r2927, [r6968+114688];
ld.shared.u32 r2930, [r6968+114692];
ld.shared.u32 r3543, [r6968+118784];
ld.shared.u32 r3546, [r6968+118788];
ld.shared.u32 r3123, [r6968+122880];
ld.shared.u32 r3126, [r6968+122884];
ld.shared.u32 r3739, [r6968+126976];
ld.shared.u32 r3742, [r6968+126980];
{
add.f16x2 r2863, r2864, r2865;
}
{
add.f16x2 r2866, r2867, r2868;
}
{
sub.f16x2 r2869, r2864, r2865;
}
{
sub.f16x2 r2872, r2867, r2868;
}
{
add.f16x2 r2875, r2876, r2877;
}
{
add.f16x2 r2878, r2879, r2880;
}
{
sub.f16x2 r2881, r2876, r2877;
}
{
sub.f16x2 r2884, r2879, r2880;
}
{
neg.f16x2 r2887, r2884;
}
{
add.f16x2 r2889, r2863, r2875;
}
{
add.f16x2 r2892, r2866, r2878;
}
{
sub.f16x2 r2895, r2863, r2875;
}
{
sub.f16x2 r2898, r2866, r2878;
}
{
add.f16x2 r2901, r2869, r2887;
}
{
add.f16x2 r2904, r2872, r2881;
}
{
sub.f16x2 r2907, r2869, r2887;
}
{
sub.f16x2 r2910, r2872, r2881;
}
{
add.f16x2 r2913, r2914, r2915;
}
{
add.f16x2 r2916, r2917, r2918;
}
{
sub.f16x2 r2919, r2914, r2915;
}
{
sub.f16x2 r2922, r2917, r2918;
}
{
add.f16x2 r2925, r2926, r2927;
}
{
add.f16x2 r2928, r2929, r2930;
}
{
sub.f16x2 r2931, r2926, r2927;
}
{
sub.f16x2 r2934, r2929, r2930;
}
{
neg.f16x2 r2937, r2934;
}
{
add.f16x2 r2939, r2913, r2925;
}
{
add.f16x2 r2942, r2916, r2928;
}
{
sub.f16x2 r2945, r2913, r2925;
}
{
sub.f16x2 r2948, r2916, r2928;
}
{
add.f16x2 r2951, r2919, r2937;
}
{
add.f16x2 r2954, r2922, r2931;
}
{
sub.f16x2 r2957, r2919, r2937;
}
{
sub.f16x2 r2960, r2922, r2931;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r2963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r2964, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r2967, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r2968, {low, high};
}
{
mul.f16x2 r2977, r2951, r2963;
}
{
mul.f16x2 r2980, r2954, r2964;
}
{
sub.f16x2 r2983, r2977, r2980;
}
{
mul.f16x2 r2986, r2951, r2964;
}
{
fma.rn.f16x2 r2989, r2954, r2963, r2986;
}
{
neg.f16x2 r2993, r2948;
}
{
mul.f16x2 r2995, r2957, r2967;
}
{
mul.f16x2 r2998, r2960, r2968;
}
{
sub.f16x2 r3001, r2995, r2998;
}
{
mul.f16x2 r3004, r2957, r2968;
}
{
fma.rn.f16x2 r3007, r2960, r2967, r3004;
}
{
add.f16x2 r3011, r2889, r2939;
}
{
add.f16x2 r3014, r2892, r2942;
}
{
sub.f16x2 r3017, r2889, r2939;
}
{
sub.f16x2 r3020, r2892, r2942;
}
{
add.f16x2 r3023, r2901, r2983;
}
{
add.f16x2 r3026, r2904, r2989;
}
{
sub.f16x2 r3029, r2901, r2983;
}
{
sub.f16x2 r3032, r2904, r2989;
}
{
add.f16x2 r3035, r2895, r2993;
}
{
add.f16x2 r3038, r2898, r2945;
}
{
sub.f16x2 r3041, r2895, r2993;
}
{
sub.f16x2 r3044, r2898, r2945;
}
{
add.f16x2 r3047, r2907, r3001;
}
{
add.f16x2 r3050, r2910, r3007;
}
{
sub.f16x2 r3053, r2907, r3001;
}
{
sub.f16x2 r3056, r2910, r3007;
}
{
add.f16x2 r3059, r3060, r3061;
}
{
add.f16x2 r3062, r3063, r3064;
}
{
sub.f16x2 r3065, r3060, r3061;
}
{
sub.f16x2 r3068, r3063, r3064;
}
{
add.f16x2 r3071, r3072, r3073;
}
{
add.f16x2 r3074, r3075, r3076;
}
{
sub.f16x2 r3077, r3072, r3073;
}
{
sub.f16x2 r3080, r3075, r3076;
}
{
neg.f16x2 r3083, r3080;
}
{
add.f16x2 r3085, r3059, r3071;
}
{
add.f16x2 r3088, r3062, r3074;
}
{
sub.f16x2 r3091, r3059, r3071;
}
{
sub.f16x2 r3094, r3062, r3074;
}
{
add.f16x2 r3097, r3065, r3083;
}
{
add.f16x2 r3100, r3068, r3077;
}
{
sub.f16x2 r3103, r3065, r3083;
}
{
sub.f16x2 r3106, r3068, r3077;
}
{
add.f16x2 r3109, r3110, r3111;
}
{
add.f16x2 r3112, r3113, r3114;
}
{
sub.f16x2 r3115, r3110, r3111;
}
{
sub.f16x2 r3118, r3113, r3114;
}
{
add.f16x2 r3121, r3122, r3123;
}
{
add.f16x2 r3124, r3125, r3126;
}
{
sub.f16x2 r3127, r3122, r3123;
}
{
sub.f16x2 r3130, r3125, r3126;
}
{
neg.f16x2 r3133, r3130;
}
{
add.f16x2 r3135, r3109, r3121;
}
{
add.f16x2 r3138, r3112, r3124;
}
{
sub.f16x2 r3141, r3109, r3121;
}
{
sub.f16x2 r3144, r3112, r3124;
}
{
add.f16x2 r3147, r3115, r3133;
}
{
add.f16x2 r3150, r3118, r3127;
}
{
sub.f16x2 r3153, r3115, r3133;
}
{
sub.f16x2 r3156, r3118, r3127;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3159, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3160, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r3163, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3164, {low, high};
}
{
mul.f16x2 r3173, r3147, r3159;
}
{
mul.f16x2 r3176, r3150, r3160;
}
{
sub.f16x2 r3179, r3173, r3176;
}
{
mul.f16x2 r3182, r3147, r3160;
}
{
fma.rn.f16x2 r3185, r3150, r3159, r3182;
}
{
neg.f16x2 r3189, r3144;
}
{
mul.f16x2 r3191, r3153, r3163;
}
{
mul.f16x2 r3194, r3156, r3164;
}
{
sub.f16x2 r3197, r3191, r3194;
}
{
mul.f16x2 r3200, r3153, r3164;
}
{
fma.rn.f16x2 r3203, r3156, r3163, r3200;
}
{
add.f16x2 r3207, r3085, r3135;
}
{
add.f16x2 r3210, r3088, r3138;
}
{
sub.f16x2 r3213, r3085, r3135;
}
{
sub.f16x2 r3216, r3088, r3138;
}
{
add.f16x2 r3219, r3097, r3179;
}
{
add.f16x2 r3222, r3100, r3185;
}
{
sub.f16x2 r3225, r3097, r3179;
}
{
sub.f16x2 r3228, r3100, r3185;
}
{
add.f16x2 r3231, r3091, r3189;
}
{
add.f16x2 r3234, r3094, r3141;
}
{
sub.f16x2 r3237, r3091, r3189;
}
{
sub.f16x2 r3240, r3094, r3141;
}
{
add.f16x2 r3243, r3103, r3197;
}
{
add.f16x2 r3246, r3106, r3203;
}
{
sub.f16x2 r3249, r3103, r3197;
}
{
sub.f16x2 r3252, r3106, r3203;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r3255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r3256, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3257, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3258, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r3259, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r3260, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r3263, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r3264, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r3265, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3266, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1042;
cvt.rn.f16.f32 high, f1042;
mov.b32 r3267, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r3268, {low, high};
}
{
mul.f16x2 r3285, r3219, r3255;
}
{
mul.f16x2 r3288, r3222, r3256;
}
{
sub.f16x2 r3291, r3285, r3288;
}
{
mul.f16x2 r3294, r3219, r3256;
}
{
fma.rn.f16x2 r3297, r3222, r3255, r3294;
}
{
mul.f16x2 r3301, r3231, r3257;
}
{
mul.f16x2 r3304, r3234, r3258;
}
{
sub.f16x2 r3307, r3301, r3304;
}
{
mul.f16x2 r3310, r3231, r3258;
}
{
fma.rn.f16x2 r3313, r3234, r3257, r3310;
}
{
mul.f16x2 r3317, r3243, r3259;
}
{
mul.f16x2 r3320, r3246, r3260;
}
{
sub.f16x2 r3323, r3317, r3320;
}
{
mul.f16x2 r3326, r3243, r3260;
}
{
fma.rn.f16x2 r3329, r3246, r3259, r3326;
}
{
neg.f16x2 r3333, r3216;
}
{
mul.f16x2 r3335, r3225, r3263;
}
{
mul.f16x2 r3338, r3228, r3264;
}
{
sub.f16x2 r3341, r3335, r3338;
}
{
mul.f16x2 r3344, r3225, r3264;
}
{
fma.rn.f16x2 r3347, r3228, r3263, r3344;
}
{
mul.f16x2 r3351, r3237, r3265;
}
{
mul.f16x2 r3354, r3240, r3266;
}
{
sub.f16x2 r3357, r3351, r3354;
}
{
mul.f16x2 r3360, r3237, r3266;
}
{
fma.rn.f16x2 r3363, r3240, r3265, r3360;
}
{
mul.f16x2 r3367, r3249, r3267;
}
{
mul.f16x2 r3370, r3252, r3268;
}
{
sub.f16x2 r3373, r3367, r3370;
}
{
mul.f16x2 r3376, r3249, r3268;
}
{
fma.rn.f16x2 r3379, r3252, r3267, r3376;
}
{
add.f16x2 r3383, r3011, r3207;
}
{
add.f16x2 r3386, r3014, r3210;
}
{
sub.f16x2 r3389, r3011, r3207;
}
{
sub.f16x2 r3392, r3014, r3210;
}
{
add.f16x2 r3395, r3023, r3291;
}
{
add.f16x2 r3398, r3026, r3297;
}
{
sub.f16x2 r3401, r3023, r3291;
}
{
sub.f16x2 r3404, r3026, r3297;
}
{
add.f16x2 r3407, r3035, r3307;
}
{
add.f16x2 r3410, r3038, r3313;
}
{
sub.f16x2 r3413, r3035, r3307;
}
{
sub.f16x2 r3416, r3038, r3313;
}
{
add.f16x2 r3419, r3047, r3323;
}
{
add.f16x2 r3422, r3050, r3329;
}
{
sub.f16x2 r3425, r3047, r3323;
}
{
sub.f16x2 r3428, r3050, r3329;
}
{
add.f16x2 r3431, r3017, r3333;
}
{
add.f16x2 r3434, r3020, r3213;
}
{
sub.f16x2 r3437, r3017, r3333;
}
{
sub.f16x2 r3440, r3020, r3213;
}
{
add.f16x2 r3443, r3029, r3341;
}
{
add.f16x2 r3446, r3032, r3347;
}
{
sub.f16x2 r3449, r3029, r3341;
}
{
sub.f16x2 r3452, r3032, r3347;
}
{
add.f16x2 r3455, r3041, r3357;
}
{
add.f16x2 r3458, r3044, r3363;
}
{
sub.f16x2 r3461, r3041, r3357;
}
{
sub.f16x2 r3464, r3044, r3363;
}
{
add.f16x2 r3467, r3053, r3373;
}
{
add.f16x2 r3470, r3056, r3379;
}
{
sub.f16x2 r3473, r3053, r3373;
}
{
sub.f16x2 r3476, r3056, r3379;
}
{
add.f16x2 r3479, r3480, r3481;
}
{
add.f16x2 r3482, r3483, r3484;
}
{
sub.f16x2 r3485, r3480, r3481;
}
{
sub.f16x2 r3488, r3483, r3484;
}
{
add.f16x2 r3491, r3492, r3493;
}
{
add.f16x2 r3494, r3495, r3496;
}
{
sub.f16x2 r3497, r3492, r3493;
}
{
sub.f16x2 r3500, r3495, r3496;
}
{
neg.f16x2 r3503, r3500;
}
{
add.f16x2 r3505, r3479, r3491;
}
{
add.f16x2 r3508, r3482, r3494;
}
{
sub.f16x2 r3511, r3479, r3491;
}
{
sub.f16x2 r3514, r3482, r3494;
}
{
add.f16x2 r3517, r3485, r3503;
}
{
add.f16x2 r3520, r3488, r3497;
}
{
sub.f16x2 r3523, r3485, r3503;
}
{
sub.f16x2 r3526, r3488, r3497;
}
{
add.f16x2 r3529, r3530, r3531;
}
{
add.f16x2 r3532, r3533, r3534;
}
{
sub.f16x2 r3535, r3530, r3531;
}
{
sub.f16x2 r3538, r3533, r3534;
}
{
add.f16x2 r3541, r3542, r3543;
}
{
add.f16x2 r3544, r3545, r3546;
}
{
sub.f16x2 r3547, r3542, r3543;
}
{
sub.f16x2 r3550, r3545, r3546;
}
{
neg.f16x2 r3553, r3550;
}
{
add.f16x2 r3555, r3529, r3541;
}
{
add.f16x2 r3558, r3532, r3544;
}
{
sub.f16x2 r3561, r3529, r3541;
}
{
sub.f16x2 r3564, r3532, r3544;
}
{
add.f16x2 r3567, r3535, r3553;
}
{
add.f16x2 r3570, r3538, r3547;
}
{
sub.f16x2 r3573, r3535, r3553;
}
{
sub.f16x2 r3576, r3538, r3547;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3579, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3580, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r3583, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3584, {low, high};
}
{
mul.f16x2 r3593, r3567, r3579;
}
{
mul.f16x2 r3596, r3570, r3580;
}
{
sub.f16x2 r3599, r3593, r3596;
}
{
mul.f16x2 r3602, r3567, r3580;
}
{
fma.rn.f16x2 r3605, r3570, r3579, r3602;
}
{
neg.f16x2 r3609, r3564;
}
{
mul.f16x2 r3611, r3573, r3583;
}
{
mul.f16x2 r3614, r3576, r3584;
}
{
sub.f16x2 r3617, r3611, r3614;
}
{
mul.f16x2 r3620, r3573, r3584;
}
{
fma.rn.f16x2 r3623, r3576, r3583, r3620;
}
{
add.f16x2 r3627, r3505, r3555;
}
{
add.f16x2 r3630, r3508, r3558;
}
{
sub.f16x2 r3633, r3505, r3555;
}
{
sub.f16x2 r3636, r3508, r3558;
}
{
add.f16x2 r3639, r3517, r3599;
}
{
add.f16x2 r3642, r3520, r3605;
}
{
sub.f16x2 r3645, r3517, r3599;
}
{
sub.f16x2 r3648, r3520, r3605;
}
{
add.f16x2 r3651, r3511, r3609;
}
{
add.f16x2 r3654, r3514, r3561;
}
{
sub.f16x2 r3657, r3511, r3609;
}
{
sub.f16x2 r3660, r3514, r3561;
}
{
add.f16x2 r3663, r3523, r3617;
}
{
add.f16x2 r3666, r3526, r3623;
}
{
sub.f16x2 r3669, r3523, r3617;
}
{
sub.f16x2 r3672, r3526, r3623;
}
{
add.f16x2 r3675, r3676, r3677;
}
{
add.f16x2 r3678, r3679, r3680;
}
{
sub.f16x2 r3681, r3676, r3677;
}
{
sub.f16x2 r3684, r3679, r3680;
}
{
add.f16x2 r3687, r3688, r3689;
}
{
add.f16x2 r3690, r3691, r3692;
}
{
sub.f16x2 r3693, r3688, r3689;
}
{
sub.f16x2 r3696, r3691, r3692;
}
{
neg.f16x2 r3699, r3696;
}
{
add.f16x2 r3701, r3675, r3687;
}
{
add.f16x2 r3704, r3678, r3690;
}
{
sub.f16x2 r3707, r3675, r3687;
}
{
sub.f16x2 r3710, r3678, r3690;
}
{
add.f16x2 r3713, r3681, r3699;
}
{
add.f16x2 r3716, r3684, r3693;
}
{
sub.f16x2 r3719, r3681, r3699;
}
{
sub.f16x2 r3722, r3684, r3693;
}
{
add.f16x2 r3725, r3726, r3727;
}
{
add.f16x2 r3728, r3729, r3730;
}
{
sub.f16x2 r3731, r3726, r3727;
}
{
sub.f16x2 r3734, r3729, r3730;
}
{
add.f16x2 r3737, r3738, r3739;
}
{
add.f16x2 r3740, r3741, r3742;
}
{
sub.f16x2 r3743, r3738, r3739;
}
{
sub.f16x2 r3746, r3741, r3742;
}
{
neg.f16x2 r3749, r3746;
}
{
add.f16x2 r3751, r3725, r3737;
}
{
add.f16x2 r3754, r3728, r3740;
}
{
sub.f16x2 r3757, r3725, r3737;
}
{
sub.f16x2 r3760, r3728, r3740;
}
{
add.f16x2 r3763, r3731, r3749;
}
{
add.f16x2 r3766, r3734, r3743;
}
{
sub.f16x2 r3769, r3731, r3749;
}
{
sub.f16x2 r3772, r3734, r3743;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3775, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3776, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r3779, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3780, {low, high};
}
{
mul.f16x2 r3789, r3763, r3775;
}
{
mul.f16x2 r3792, r3766, r3776;
}
{
sub.f16x2 r3795, r3789, r3792;
}
{
mul.f16x2 r3798, r3763, r3776;
}
{
fma.rn.f16x2 r3801, r3766, r3775, r3798;
}
{
neg.f16x2 r3805, r3760;
}
{
mul.f16x2 r3807, r3769, r3779;
}
{
mul.f16x2 r3810, r3772, r3780;
}
{
sub.f16x2 r3813, r3807, r3810;
}
{
mul.f16x2 r3816, r3769, r3780;
}
{
fma.rn.f16x2 r3819, r3772, r3779, r3816;
}
{
add.f16x2 r3823, r3701, r3751;
}
{
add.f16x2 r3826, r3704, r3754;
}
{
sub.f16x2 r3829, r3701, r3751;
}
{
sub.f16x2 r3832, r3704, r3754;
}
{
add.f16x2 r3835, r3713, r3795;
}
{
add.f16x2 r3838, r3716, r3801;
}
{
sub.f16x2 r3841, r3713, r3795;
}
{
sub.f16x2 r3844, r3716, r3801;
}
{
add.f16x2 r3847, r3707, r3805;
}
{
add.f16x2 r3850, r3710, r3757;
}
{
sub.f16x2 r3853, r3707, r3805;
}
{
sub.f16x2 r3856, r3710, r3757;
}
{
add.f16x2 r3859, r3719, r3813;
}
{
add.f16x2 r3862, r3722, r3819;
}
{
sub.f16x2 r3865, r3719, r3813;
}
{
sub.f16x2 r3868, r3722, r3819;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r3871, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r3872, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3873, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3874, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r3875, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r3876, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r3879, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r3880, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r3881, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3882, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1042;
cvt.rn.f16.f32 high, f1042;
mov.b32 r3883, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r3884, {low, high};
}
{
mul.f16x2 r3901, r3835, r3871;
}
{
mul.f16x2 r3904, r3838, r3872;
}
{
sub.f16x2 r3907, r3901, r3904;
}
{
mul.f16x2 r3910, r3835, r3872;
}
{
fma.rn.f16x2 r3913, r3838, r3871, r3910;
}
{
mul.f16x2 r3917, r3847, r3873;
}
{
mul.f16x2 r3920, r3850, r3874;
}
{
sub.f16x2 r3923, r3917, r3920;
}
{
mul.f16x2 r3926, r3847, r3874;
}
{
fma.rn.f16x2 r3929, r3850, r3873, r3926;
}
{
mul.f16x2 r3933, r3859, r3875;
}
{
mul.f16x2 r3936, r3862, r3876;
}
{
sub.f16x2 r3939, r3933, r3936;
}
{
mul.f16x2 r3942, r3859, r3876;
}
{
fma.rn.f16x2 r3945, r3862, r3875, r3942;
}
{
neg.f16x2 r3949, r3832;
}
{
mul.f16x2 r3951, r3841, r3879;
}
{
mul.f16x2 r3954, r3844, r3880;
}
{
sub.f16x2 r3957, r3951, r3954;
}
{
mul.f16x2 r3960, r3841, r3880;
}
{
fma.rn.f16x2 r3963, r3844, r3879, r3960;
}
{
mul.f16x2 r3967, r3853, r3881;
}
{
mul.f16x2 r3970, r3856, r3882;
}
{
sub.f16x2 r3973, r3967, r3970;
}
{
mul.f16x2 r3976, r3853, r3882;
}
{
fma.rn.f16x2 r3979, r3856, r3881, r3976;
}
{
mul.f16x2 r3983, r3865, r3883;
}
{
mul.f16x2 r3986, r3868, r3884;
}
{
sub.f16x2 r3989, r3983, r3986;
}
{
mul.f16x2 r3992, r3865, r3884;
}
{
fma.rn.f16x2 r3995, r3868, r3883, r3992;
}
{
add.f16x2 r3999, r3627, r3823;
}
{
add.f16x2 r4002, r3630, r3826;
}
{
sub.f16x2 r4005, r3627, r3823;
}
{
sub.f16x2 r4008, r3630, r3826;
}
{
add.f16x2 r4011, r3639, r3907;
}
{
add.f16x2 r4014, r3642, r3913;
}
{
sub.f16x2 r4017, r3639, r3907;
}
{
sub.f16x2 r4020, r3642, r3913;
}
{
add.f16x2 r4023, r3651, r3923;
}
{
add.f16x2 r4026, r3654, r3929;
}
{
sub.f16x2 r4029, r3651, r3923;
}
{
sub.f16x2 r4032, r3654, r3929;
}
{
add.f16x2 r4035, r3663, r3939;
}
{
add.f16x2 r4038, r3666, r3945;
}
{
sub.f16x2 r4041, r3663, r3939;
}
{
sub.f16x2 r4044, r3666, r3945;
}
{
add.f16x2 r4047, r3633, r3949;
}
{
add.f16x2 r4050, r3636, r3829;
}
{
sub.f16x2 r4053, r3633, r3949;
}
{
sub.f16x2 r4056, r3636, r3829;
}
{
add.f16x2 r4059, r3645, r3957;
}
{
add.f16x2 r4062, r3648, r3963;
}
{
sub.f16x2 r4065, r3645, r3957;
}
{
sub.f16x2 r4068, r3648, r3963;
}
{
add.f16x2 r4071, r3657, r3973;
}
{
add.f16x2 r4074, r3660, r3979;
}
{
sub.f16x2 r4077, r3657, r3973;
}
{
sub.f16x2 r4080, r3660, r3979;
}
{
add.f16x2 r4083, r3669, r3989;
}
{
add.f16x2 r4086, r3672, r3995;
}
{
sub.f16x2 r4089, r3669, r3989;
}
{
sub.f16x2 r4092, r3672, r3995;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f690;
cvt.rn.f16.f32 high, f690;
mov.b32 r4095, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f714;
cvt.rn.f16.f32 high, f714;
mov.b32 r4096, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r4097, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r4098, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f698;
cvt.rn.f16.f32 high, f698;
mov.b32 r4099, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f706;
cvt.rn.f16.f32 high, f706;
mov.b32 r4100, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r4101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r4102, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f706;
cvt.rn.f16.f32 high, f706;
mov.b32 r4103, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f698;
cvt.rn.f16.f32 high, f698;
mov.b32 r4104, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r4105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r4106, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f714;
cvt.rn.f16.f32 high, f714;
mov.b32 r4107, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f690;
cvt.rn.f16.f32 high, f690;
mov.b32 r4108, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f688;
cvt.rn.f16.f32 high, f688;
mov.b32 r4111, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f690;
cvt.rn.f16.f32 high, f690;
mov.b32 r4112, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r4113, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r4114, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f696;
cvt.rn.f16.f32 high, f696;
mov.b32 r4115, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f698;
cvt.rn.f16.f32 high, f698;
mov.b32 r4116, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r4117, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r4118, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f704;
cvt.rn.f16.f32 high, f704;
mov.b32 r4119, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f706;
cvt.rn.f16.f32 high, f706;
mov.b32 r4120, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1042;
cvt.rn.f16.f32 high, f1042;
mov.b32 r4121, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r4122, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f712;
cvt.rn.f16.f32 high, f712;
mov.b32 r4123, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f714;
cvt.rn.f16.f32 high, f714;
mov.b32 r4124, {low, high};
}
{
mul.f16x2 r4157, r4011, r4095;
}
{
mul.f16x2 r4160, r4014, r4096;
}
{
sub.f16x2 r4163, r4157, r4160;
}
{
mul.f16x2 r4166, r4011, r4096;
}
{
fma.rn.f16x2 r4169, r4014, r4095, r4166;
}
{
mul.f16x2 r4173, r4023, r4097;
}
{
mul.f16x2 r4176, r4026, r4098;
}
{
sub.f16x2 r4179, r4173, r4176;
}
{
mul.f16x2 r4182, r4023, r4098;
}
{
fma.rn.f16x2 r4185, r4026, r4097, r4182;
}
{
mul.f16x2 r4189, r4035, r4099;
}
{
mul.f16x2 r4192, r4038, r4100;
}
{
sub.f16x2 r4195, r4189, r4192;
}
{
mul.f16x2 r4198, r4035, r4100;
}
{
fma.rn.f16x2 r4201, r4038, r4099, r4198;
}
{
mul.f16x2 r4205, r4047, r4101;
}
{
mul.f16x2 r4208, r4050, r4102;
}
{
sub.f16x2 r4211, r4205, r4208;
}
{
mul.f16x2 r4214, r4047, r4102;
}
{
fma.rn.f16x2 r4217, r4050, r4101, r4214;
}
{
mul.f16x2 r4221, r4059, r4103;
}
{
mul.f16x2 r4224, r4062, r4104;
}
{
sub.f16x2 r4227, r4221, r4224;
}
{
mul.f16x2 r4230, r4059, r4104;
}
{
fma.rn.f16x2 r4233, r4062, r4103, r4230;
}
{
mul.f16x2 r4237, r4071, r4105;
}
{
mul.f16x2 r4240, r4074, r4106;
}
{
sub.f16x2 r4243, r4237, r4240;
}
{
mul.f16x2 r4246, r4071, r4106;
}
{
fma.rn.f16x2 r4249, r4074, r4105, r4246;
}
{
mul.f16x2 r4253, r4083, r4107;
}
{
mul.f16x2 r4256, r4086, r4108;
}
{
sub.f16x2 r4259, r4253, r4256;
}
{
mul.f16x2 r4262, r4083, r4108;
}
{
fma.rn.f16x2 r4265, r4086, r4107, r4262;
}
{
neg.f16x2 r4269, r4008;
}
{
mul.f16x2 r4271, r4017, r4111;
}
{
mul.f16x2 r4274, r4020, r4112;
}
{
sub.f16x2 r4277, r4271, r4274;
}
{
mul.f16x2 r4280, r4017, r4112;
}
{
fma.rn.f16x2 r4283, r4020, r4111, r4280;
}
{
mul.f16x2 r4287, r4029, r4113;
}
{
mul.f16x2 r4290, r4032, r4114;
}
{
sub.f16x2 r4293, r4287, r4290;
}
{
mul.f16x2 r4296, r4029, r4114;
}
{
fma.rn.f16x2 r4299, r4032, r4113, r4296;
}
{
mul.f16x2 r4303, r4041, r4115;
}
{
mul.f16x2 r4306, r4044, r4116;
}
{
sub.f16x2 r4309, r4303, r4306;
}
{
mul.f16x2 r4312, r4041, r4116;
}
{
fma.rn.f16x2 r4315, r4044, r4115, r4312;
}
{
mul.f16x2 r4319, r4053, r4117;
}
{
mul.f16x2 r4322, r4056, r4118;
}
{
sub.f16x2 r4325, r4319, r4322;
}
{
mul.f16x2 r4328, r4053, r4118;
}
{
fma.rn.f16x2 r4331, r4056, r4117, r4328;
}
{
mul.f16x2 r4335, r4065, r4119;
}
{
mul.f16x2 r4338, r4068, r4120;
}
{
sub.f16x2 r4341, r4335, r4338;
}
{
mul.f16x2 r4344, r4065, r4120;
}
{
fma.rn.f16x2 r4347, r4068, r4119, r4344;
}
{
mul.f16x2 r4351, r4077, r4121;
}
{
mul.f16x2 r4354, r4080, r4122;
}
{
sub.f16x2 r4357, r4351, r4354;
}
{
mul.f16x2 r4360, r4077, r4122;
}
{
fma.rn.f16x2 r4363, r4080, r4121, r4360;
}
{
mul.f16x2 r4367, r4089, r4123;
}
{
mul.f16x2 r4370, r4092, r4124;
}
{
sub.f16x2 r4373, r4367, r4370;
}
{
mul.f16x2 r4376, r4089, r4124;
}
{
fma.rn.f16x2 r4379, r4092, r4123, r4376;
}
{
add.f16x2 r4383, r3383, r3999;
}
{
add.f16x2 r4386, r3386, r4002;
}
{
sub.f16x2 r4389, r3383, r3999;
}
{
sub.f16x2 r4392, r3386, r4002;
}
{
add.f16x2 r4395, r3395, r4163;
}
{
add.f16x2 r4398, r3398, r4169;
}
{
sub.f16x2 r4401, r3395, r4163;
}
{
sub.f16x2 r4404, r3398, r4169;
}
{
add.f16x2 r4407, r3407, r4179;
}
{
add.f16x2 r4410, r3410, r4185;
}
{
sub.f16x2 r4413, r3407, r4179;
}
{
sub.f16x2 r4416, r3410, r4185;
}
{
add.f16x2 r4419, r3419, r4195;
}
{
add.f16x2 r4422, r3422, r4201;
}
{
sub.f16x2 r4425, r3419, r4195;
}
{
sub.f16x2 r4428, r3422, r4201;
}
{
add.f16x2 r4431, r3431, r4211;
}
{
add.f16x2 r4434, r3434, r4217;
}
{
sub.f16x2 r4437, r3431, r4211;
}
{
sub.f16x2 r4440, r3434, r4217;
}
{
add.f16x2 r4443, r3443, r4227;
}
{
add.f16x2 r4446, r3446, r4233;
}
{
sub.f16x2 r4449, r3443, r4227;
}
{
sub.f16x2 r4452, r3446, r4233;
}
{
add.f16x2 r4455, r3455, r4243;
}
{
add.f16x2 r4458, r3458, r4249;
}
{
sub.f16x2 r4461, r3455, r4243;
}
{
sub.f16x2 r4464, r3458, r4249;
}
{
add.f16x2 r4467, r3467, r4259;
}
{
add.f16x2 r4470, r3470, r4265;
}
{
sub.f16x2 r4473, r3467, r4259;
}
{
sub.f16x2 r4476, r3470, r4265;
}
{
add.f16x2 r4479, r3389, r4269;
}
{
add.f16x2 r4482, r3392, r4005;
}
{
sub.f16x2 r4485, r3389, r4269;
}
{
sub.f16x2 r4488, r3392, r4005;
}
{
add.f16x2 r4491, r3401, r4277;
}
{
add.f16x2 r4494, r3404, r4283;
}
{
sub.f16x2 r4497, r3401, r4277;
}
{
sub.f16x2 r4500, r3404, r4283;
}
{
add.f16x2 r4503, r3413, r4293;
}
{
add.f16x2 r4506, r3416, r4299;
}
{
sub.f16x2 r4509, r3413, r4293;
}
{
sub.f16x2 r4512, r3416, r4299;
}
{
add.f16x2 r4515, r3425, r4309;
}
{
add.f16x2 r4518, r3428, r4315;
}
{
sub.f16x2 r4521, r3425, r4309;
}
{
sub.f16x2 r4524, r3428, r4315;
}
{
add.f16x2 r4527, r3437, r4325;
}
{
add.f16x2 r4530, r3440, r4331;
}
{
sub.f16x2 r4533, r3437, r4325;
}
{
sub.f16x2 r4536, r3440, r4331;
}
{
add.f16x2 r4539, r3449, r4341;
}
{
add.f16x2 r4542, r3452, r4347;
}
{
sub.f16x2 r4545, r3449, r4341;
}
{
sub.f16x2 r4548, r3452, r4347;
}
{
add.f16x2 r4551, r3461, r4357;
}
{
add.f16x2 r4554, r3464, r4363;
}
{
sub.f16x2 r4557, r3461, r4357;
}
{
sub.f16x2 r4560, r3464, r4363;
}
{
add.f16x2 r4563, r3473, r4373;
}
{
add.f16x2 r4566, r3476, r4379;
}
{
sub.f16x2 r4569, r3473, r4373;
}
{
sub.f16x2 r4572, r3476, r4379;
}
bfe.u32 r6970, r6961, 5, 4;
cvt.rn.f32.u32 f1080, r6970;
mul.f32 f1081, f1080, 0f3C490FDB;
cos.approx.f32 f779, f1081;
sin.approx.f32 f1082, f1081;
neg.f32 f780, f1082;
and.b32 r6977, r6961, 480;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f779;
cvt.rn.f16.f32 high, f780;
mov.b32 r4575, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4578, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4580, {high, high};
}
{
mul.f16x2 r4582, r4398, r4580;
}
{
fma.rn.f16x2 r4585, r4395, r4578, r4582;
}
{
mul.f16x2 r4589, r4395, r4580;
}
{
neg.f16x2 r4592, r4589;
}
{
fma.rn.f16x2 r4594, r4398, r4578, r4592;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4598, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4600, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4602, {low, high};
}
{
mul.f16x2 r4603, r4600, r4602;
}
{
mul.f16x2 r4606, r4575, r4598;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4609, {high, low};
}
{
fma.rn.f16x2 r4611, r4603, r4609, r4606;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4611;
mov.b32 r4615, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4611;
mov.b32 r4617, {high, high};
}
{
mul.f16x2 r4619, r4410, r4617;
}
{
fma.rn.f16x2 r4622, r4407, r4615, r4619;
}
{
mul.f16x2 r4626, r4407, r4617;
}
{
neg.f16x2 r4629, r4626;
}
{
fma.rn.f16x2 r4631, r4410, r4615, r4629;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4635, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4637, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4639, {low, high};
}
{
mul.f16x2 r4640, r4637, r4639;
}
{
mul.f16x2 r4643, r4611, r4635;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4611;
mov.b32 r4646, {high, low};
}
{
fma.rn.f16x2 r4648, r4640, r4646, r4643;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4648;
mov.b32 r4652, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4648;
mov.b32 r4654, {high, high};
}
{
mul.f16x2 r4656, r4422, r4654;
}
{
fma.rn.f16x2 r4659, r4419, r4652, r4656;
}
{
mul.f16x2 r4663, r4419, r4654;
}
{
neg.f16x2 r4666, r4663;
}
{
fma.rn.f16x2 r4668, r4422, r4652, r4666;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4672, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4674, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4676, {low, high};
}
{
mul.f16x2 r4677, r4674, r4676;
}
{
mul.f16x2 r4680, r4648, r4672;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4648;
mov.b32 r4683, {high, low};
}
{
fma.rn.f16x2 r4685, r4677, r4683, r4680;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4685;
mov.b32 r4689, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4685;
mov.b32 r4691, {high, high};
}
{
mul.f16x2 r4693, r4434, r4691;
}
{
fma.rn.f16x2 r4696, r4431, r4689, r4693;
}
{
mul.f16x2 r4700, r4431, r4691;
}
{
neg.f16x2 r4703, r4700;
}
{
fma.rn.f16x2 r4705, r4434, r4689, r4703;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4709, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4711, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4713, {low, high};
}
{
mul.f16x2 r4714, r4711, r4713;
}
{
mul.f16x2 r4717, r4685, r4709;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4685;
mov.b32 r4720, {high, low};
}
{
fma.rn.f16x2 r4722, r4714, r4720, r4717;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4722;
mov.b32 r4726, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4722;
mov.b32 r4728, {high, high};
}
{
mul.f16x2 r4730, r4446, r4728;
}
{
fma.rn.f16x2 r4733, r4443, r4726, r4730;
}
{
mul.f16x2 r4737, r4443, r4728;
}
{
neg.f16x2 r4740, r4737;
}
{
fma.rn.f16x2 r4742, r4446, r4726, r4740;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4746, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4748, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4750, {low, high};
}
{
mul.f16x2 r4751, r4748, r4750;
}
{
mul.f16x2 r4754, r4722, r4746;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4722;
mov.b32 r4757, {high, low};
}
{
fma.rn.f16x2 r4759, r4751, r4757, r4754;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4759;
mov.b32 r4763, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4759;
mov.b32 r4765, {high, high};
}
{
mul.f16x2 r4767, r4458, r4765;
}
{
fma.rn.f16x2 r4770, r4455, r4763, r4767;
}
{
mul.f16x2 r4774, r4455, r4765;
}
{
neg.f16x2 r4777, r4774;
}
{
fma.rn.f16x2 r4779, r4458, r4763, r4777;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4783, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4785, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4787, {low, high};
}
{
mul.f16x2 r4788, r4785, r4787;
}
{
mul.f16x2 r4791, r4759, r4783;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4759;
mov.b32 r4794, {high, low};
}
{
fma.rn.f16x2 r4796, r4788, r4794, r4791;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4796;
mov.b32 r4800, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4796;
mov.b32 r4802, {high, high};
}
{
mul.f16x2 r4804, r4470, r4802;
}
{
fma.rn.f16x2 r4807, r4467, r4800, r4804;
}
{
mul.f16x2 r4811, r4467, r4802;
}
{
neg.f16x2 r4814, r4811;
}
{
fma.rn.f16x2 r4816, r4470, r4800, r4814;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4820, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4822, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4824, {low, high};
}
{
mul.f16x2 r4825, r4822, r4824;
}
{
mul.f16x2 r4828, r4796, r4820;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4796;
mov.b32 r4831, {high, low};
}
{
fma.rn.f16x2 r4833, r4825, r4831, r4828;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4833;
mov.b32 r4837, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4833;
mov.b32 r4839, {high, high};
}
{
mul.f16x2 r4841, r4482, r4839;
}
{
fma.rn.f16x2 r4844, r4479, r4837, r4841;
}
{
mul.f16x2 r4848, r4479, r4839;
}
{
neg.f16x2 r4851, r4848;
}
{
fma.rn.f16x2 r4853, r4482, r4837, r4851;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4857, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4859, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4861, {low, high};
}
{
mul.f16x2 r4862, r4859, r4861;
}
{
mul.f16x2 r4865, r4833, r4857;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4833;
mov.b32 r4868, {high, low};
}
{
fma.rn.f16x2 r4870, r4862, r4868, r4865;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4870;
mov.b32 r4874, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4870;
mov.b32 r4876, {high, high};
}
{
mul.f16x2 r4878, r4494, r4876;
}
{
fma.rn.f16x2 r4881, r4491, r4874, r4878;
}
{
mul.f16x2 r4885, r4491, r4876;
}
{
neg.f16x2 r4888, r4885;
}
{
fma.rn.f16x2 r4890, r4494, r4874, r4888;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4894, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4896, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4898, {low, high};
}
{
mul.f16x2 r4899, r4896, r4898;
}
{
mul.f16x2 r4902, r4870, r4894;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4870;
mov.b32 r4905, {high, low};
}
{
fma.rn.f16x2 r4907, r4899, r4905, r4902;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4907;
mov.b32 r4911, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4907;
mov.b32 r4913, {high, high};
}
{
mul.f16x2 r4915, r4506, r4913;
}
{
fma.rn.f16x2 r4918, r4503, r4911, r4915;
}
{
mul.f16x2 r4922, r4503, r4913;
}
{
neg.f16x2 r4925, r4922;
}
{
fma.rn.f16x2 r4927, r4506, r4911, r4925;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4931, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4933, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4935, {low, high};
}
{
mul.f16x2 r4936, r4933, r4935;
}
{
mul.f16x2 r4939, r4907, r4931;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4907;
mov.b32 r4942, {high, low};
}
{
fma.rn.f16x2 r4944, r4936, r4942, r4939;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4944;
mov.b32 r4948, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4944;
mov.b32 r4950, {high, high};
}
{
mul.f16x2 r4952, r4518, r4950;
}
{
fma.rn.f16x2 r4955, r4515, r4948, r4952;
}
{
mul.f16x2 r4959, r4515, r4950;
}
{
neg.f16x2 r4962, r4959;
}
{
fma.rn.f16x2 r4964, r4518, r4948, r4962;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4968, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4970, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4972, {low, high};
}
{
mul.f16x2 r4973, r4970, r4972;
}
{
mul.f16x2 r4976, r4944, r4968;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4944;
mov.b32 r4979, {high, low};
}
{
fma.rn.f16x2 r4981, r4973, r4979, r4976;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4981;
mov.b32 r4985, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4981;
mov.b32 r4987, {high, high};
}
{
mul.f16x2 r4989, r4530, r4987;
}
{
fma.rn.f16x2 r4992, r4527, r4985, r4989;
}
{
mul.f16x2 r4996, r4527, r4987;
}
{
neg.f16x2 r4999, r4996;
}
{
fma.rn.f16x2 r5001, r4530, r4985, r4999;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5005, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5007, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5009, {low, high};
}
{
mul.f16x2 r5010, r5007, r5009;
}
{
mul.f16x2 r5013, r4981, r5005;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4981;
mov.b32 r5016, {high, low};
}
{
fma.rn.f16x2 r5018, r5010, r5016, r5013;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5018;
mov.b32 r5022, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5018;
mov.b32 r5024, {high, high};
}
{
mul.f16x2 r5026, r4542, r5024;
}
{
fma.rn.f16x2 r5029, r4539, r5022, r5026;
}
{
mul.f16x2 r5033, r4539, r5024;
}
{
neg.f16x2 r5036, r5033;
}
{
fma.rn.f16x2 r5038, r4542, r5022, r5036;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5042, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5044, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5046, {low, high};
}
{
mul.f16x2 r5047, r5044, r5046;
}
{
mul.f16x2 r5050, r5018, r5042;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5018;
mov.b32 r5053, {high, low};
}
{
fma.rn.f16x2 r5055, r5047, r5053, r5050;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5055;
mov.b32 r5059, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5055;
mov.b32 r5061, {high, high};
}
{
mul.f16x2 r5063, r4554, r5061;
}
{
fma.rn.f16x2 r5066, r4551, r5059, r5063;
}
{
mul.f16x2 r5070, r4551, r5061;
}
{
neg.f16x2 r5073, r5070;
}
{
fma.rn.f16x2 r5075, r4554, r5059, r5073;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5079, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5081, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5083, {low, high};
}
{
mul.f16x2 r5084, r5081, r5083;
}
{
mul.f16x2 r5087, r5055, r5079;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5055;
mov.b32 r5090, {high, low};
}
{
fma.rn.f16x2 r5092, r5084, r5090, r5087;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5092;
mov.b32 r5096, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5092;
mov.b32 r5098, {high, high};
}
{
mul.f16x2 r5100, r4566, r5098;
}
{
fma.rn.f16x2 r5103, r4563, r5096, r5100;
}
{
mul.f16x2 r5107, r4563, r5098;
}
{
neg.f16x2 r5110, r5107;
}
{
fma.rn.f16x2 r5112, r4566, r5096, r5110;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5116, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5118, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5120, {low, high};
}
{
mul.f16x2 r5121, r5118, r5120;
}
{
mul.f16x2 r5124, r5092, r5116;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5092;
mov.b32 r5127, {high, low};
}
{
fma.rn.f16x2 r5129, r5121, r5127, r5124;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5129;
mov.b32 r5133, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5129;
mov.b32 r5135, {high, high};
}
{
mul.f16x2 r5137, r4392, r5135;
}
{
fma.rn.f16x2 r5140, r4389, r5133, r5137;
}
{
mul.f16x2 r5144, r4389, r5135;
}
{
neg.f16x2 r5147, r5144;
}
{
fma.rn.f16x2 r5149, r4392, r5133, r5147;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5153, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5155, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5157, {low, high};
}
{
mul.f16x2 r5158, r5155, r5157;
}
{
mul.f16x2 r5161, r5129, r5153;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5129;
mov.b32 r5164, {high, low};
}
{
fma.rn.f16x2 r5166, r5158, r5164, r5161;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5166;
mov.b32 r5170, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5166;
mov.b32 r5172, {high, high};
}
{
mul.f16x2 r5174, r4404, r5172;
}
{
fma.rn.f16x2 r5177, r4401, r5170, r5174;
}
{
mul.f16x2 r5181, r4401, r5172;
}
{
neg.f16x2 r5184, r5181;
}
{
fma.rn.f16x2 r5186, r4404, r5170, r5184;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5190, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5192, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5194, {low, high};
}
{
mul.f16x2 r5195, r5192, r5194;
}
{
mul.f16x2 r5198, r5166, r5190;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5166;
mov.b32 r5201, {high, low};
}
{
fma.rn.f16x2 r5203, r5195, r5201, r5198;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5203;
mov.b32 r5207, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5203;
mov.b32 r5209, {high, high};
}
{
mul.f16x2 r5211, r4416, r5209;
}
{
fma.rn.f16x2 r5214, r4413, r5207, r5211;
}
{
mul.f16x2 r5218, r4413, r5209;
}
{
neg.f16x2 r5221, r5218;
}
{
fma.rn.f16x2 r5223, r4416, r5207, r5221;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5227, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5229, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5231, {low, high};
}
{
mul.f16x2 r5232, r5229, r5231;
}
{
mul.f16x2 r5235, r5203, r5227;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5203;
mov.b32 r5238, {high, low};
}
{
fma.rn.f16x2 r5240, r5232, r5238, r5235;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5240;
mov.b32 r5244, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5240;
mov.b32 r5246, {high, high};
}
{
mul.f16x2 r5248, r4428, r5246;
}
{
fma.rn.f16x2 r5251, r4425, r5244, r5248;
}
{
mul.f16x2 r5255, r4425, r5246;
}
{
neg.f16x2 r5258, r5255;
}
{
fma.rn.f16x2 r5260, r4428, r5244, r5258;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5264, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5266, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5268, {low, high};
}
{
mul.f16x2 r5269, r5266, r5268;
}
{
mul.f16x2 r5272, r5240, r5264;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5240;
mov.b32 r5275, {high, low};
}
{
fma.rn.f16x2 r5277, r5269, r5275, r5272;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5277;
mov.b32 r5281, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5277;
mov.b32 r5283, {high, high};
}
{
mul.f16x2 r5285, r4440, r5283;
}
{
fma.rn.f16x2 r5288, r4437, r5281, r5285;
}
{
mul.f16x2 r5292, r4437, r5283;
}
{
neg.f16x2 r5295, r5292;
}
{
fma.rn.f16x2 r5297, r4440, r5281, r5295;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5301, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5303, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5305, {low, high};
}
{
mul.f16x2 r5306, r5303, r5305;
}
{
mul.f16x2 r5309, r5277, r5301;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5277;
mov.b32 r5312, {high, low};
}
{
fma.rn.f16x2 r5314, r5306, r5312, r5309;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5314;
mov.b32 r5318, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5314;
mov.b32 r5320, {high, high};
}
{
mul.f16x2 r5322, r4452, r5320;
}
{
fma.rn.f16x2 r5325, r4449, r5318, r5322;
}
{
mul.f16x2 r5329, r4449, r5320;
}
{
neg.f16x2 r5332, r5329;
}
{
fma.rn.f16x2 r5334, r4452, r5318, r5332;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5338, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5340, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5342, {low, high};
}
{
mul.f16x2 r5343, r5340, r5342;
}
{
mul.f16x2 r5346, r5314, r5338;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5314;
mov.b32 r5349, {high, low};
}
{
fma.rn.f16x2 r5351, r5343, r5349, r5346;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5351;
mov.b32 r5355, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5351;
mov.b32 r5357, {high, high};
}
{
mul.f16x2 r5359, r4464, r5357;
}
{
fma.rn.f16x2 r5362, r4461, r5355, r5359;
}
{
mul.f16x2 r5366, r4461, r5357;
}
{
neg.f16x2 r5369, r5366;
}
{
fma.rn.f16x2 r5371, r4464, r5355, r5369;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5375, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5377, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5379, {low, high};
}
{
mul.f16x2 r5380, r5377, r5379;
}
{
mul.f16x2 r5383, r5351, r5375;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5351;
mov.b32 r5386, {high, low};
}
{
fma.rn.f16x2 r5388, r5380, r5386, r5383;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5388;
mov.b32 r5392, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5388;
mov.b32 r5394, {high, high};
}
{
mul.f16x2 r5396, r4476, r5394;
}
{
fma.rn.f16x2 r5399, r4473, r5392, r5396;
}
{
mul.f16x2 r5403, r4473, r5394;
}
{
neg.f16x2 r5406, r5403;
}
{
fma.rn.f16x2 r5408, r4476, r5392, r5406;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5412, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5414, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5416, {low, high};
}
{
mul.f16x2 r5417, r5414, r5416;
}
{
mul.f16x2 r5420, r5388, r5412;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5388;
mov.b32 r5423, {high, low};
}
{
fma.rn.f16x2 r5425, r5417, r5423, r5420;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5425;
mov.b32 r5429, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5425;
mov.b32 r5431, {high, high};
}
{
mul.f16x2 r5433, r4488, r5431;
}
{
fma.rn.f16x2 r5436, r4485, r5429, r5433;
}
{
mul.f16x2 r5440, r4485, r5431;
}
{
neg.f16x2 r5443, r5440;
}
{
fma.rn.f16x2 r5445, r4488, r5429, r5443;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5449, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5451, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5453, {low, high};
}
{
mul.f16x2 r5454, r5451, r5453;
}
{
mul.f16x2 r5457, r5425, r5449;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5425;
mov.b32 r5460, {high, low};
}
{
fma.rn.f16x2 r5462, r5454, r5460, r5457;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5462;
mov.b32 r5466, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5462;
mov.b32 r5468, {high, high};
}
{
mul.f16x2 r5470, r4500, r5468;
}
{
fma.rn.f16x2 r5473, r4497, r5466, r5470;
}
{
mul.f16x2 r5477, r4497, r5468;
}
{
neg.f16x2 r5480, r5477;
}
{
fma.rn.f16x2 r5482, r4500, r5466, r5480;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5486, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5488, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5490, {low, high};
}
{
mul.f16x2 r5491, r5488, r5490;
}
{
mul.f16x2 r5494, r5462, r5486;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5462;
mov.b32 r5497, {high, low};
}
{
fma.rn.f16x2 r5499, r5491, r5497, r5494;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5499;
mov.b32 r5503, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5499;
mov.b32 r5505, {high, high};
}
{
mul.f16x2 r5507, r4512, r5505;
}
{
fma.rn.f16x2 r5510, r4509, r5503, r5507;
}
{
mul.f16x2 r5514, r4509, r5505;
}
{
neg.f16x2 r5517, r5514;
}
{
fma.rn.f16x2 r5519, r4512, r5503, r5517;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5523, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5525, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5527, {low, high};
}
{
mul.f16x2 r5528, r5525, r5527;
}
{
mul.f16x2 r5531, r5499, r5523;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5499;
mov.b32 r5534, {high, low};
}
{
fma.rn.f16x2 r5536, r5528, r5534, r5531;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5536;
mov.b32 r5540, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5536;
mov.b32 r5542, {high, high};
}
{
mul.f16x2 r5544, r4524, r5542;
}
{
fma.rn.f16x2 r5547, r4521, r5540, r5544;
}
{
mul.f16x2 r5551, r4521, r5542;
}
{
neg.f16x2 r5554, r5551;
}
{
fma.rn.f16x2 r5556, r4524, r5540, r5554;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5560, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5562, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5564, {low, high};
}
{
mul.f16x2 r5565, r5562, r5564;
}
{
mul.f16x2 r5568, r5536, r5560;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5536;
mov.b32 r5571, {high, low};
}
{
fma.rn.f16x2 r5573, r5565, r5571, r5568;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5573;
mov.b32 r5577, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5573;
mov.b32 r5579, {high, high};
}
{
mul.f16x2 r5581, r4536, r5579;
}
{
fma.rn.f16x2 r5584, r4533, r5577, r5581;
}
{
mul.f16x2 r5588, r4533, r5579;
}
{
neg.f16x2 r5591, r5588;
}
{
fma.rn.f16x2 r5593, r4536, r5577, r5591;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5597, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5599, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5601, {low, high};
}
{
mul.f16x2 r5602, r5599, r5601;
}
{
mul.f16x2 r5605, r5573, r5597;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5573;
mov.b32 r5608, {high, low};
}
{
fma.rn.f16x2 r5610, r5602, r5608, r5605;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5610;
mov.b32 r5614, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5610;
mov.b32 r5616, {high, high};
}
{
mul.f16x2 r5618, r4548, r5616;
}
{
fma.rn.f16x2 r5621, r4545, r5614, r5618;
}
{
mul.f16x2 r5625, r4545, r5616;
}
{
neg.f16x2 r5628, r5625;
}
{
fma.rn.f16x2 r5630, r4548, r5614, r5628;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5634, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5636, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5638, {low, high};
}
{
mul.f16x2 r5639, r5636, r5638;
}
{
mul.f16x2 r5642, r5610, r5634;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5610;
mov.b32 r5645, {high, low};
}
{
fma.rn.f16x2 r5647, r5639, r5645, r5642;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5647;
mov.b32 r5651, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5647;
mov.b32 r5653, {high, high};
}
{
mul.f16x2 r5655, r4560, r5653;
}
{
fma.rn.f16x2 r5658, r4557, r5651, r5655;
}
{
mul.f16x2 r5662, r4557, r5653;
}
{
neg.f16x2 r5665, r5662;
}
{
fma.rn.f16x2 r5667, r4560, r5651, r5665;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5671, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5673, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5675, {low, high};
}
{
mul.f16x2 r5676, r5673, r5675;
}
{
mul.f16x2 r5679, r5647, r5671;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5647;
mov.b32 r5682, {high, low};
}
{
fma.rn.f16x2 r5684, r5676, r5682, r5679;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5684;
mov.b32 r5688, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5684;
mov.b32 r5690, {high, high};
}
{
mul.f16x2 r5692, r4572, r5690;
}
{
fma.rn.f16x2 r5695, r4569, r5688, r5692;
}
{
mul.f16x2 r5699, r4569, r5690;
}
{
neg.f16x2 r5702, r5699;
}
{
fma.rn.f16x2 r5704, r4572, r5688, r5702;
}
shl.b32 r6971, r6961, 3;
and.b32 r6972, r6971, 248;
add.s32 r6973, r6965, r6972;
barrier.sync 0;
and.b32 r6974, r6963, 122880;
add.s32 r6975, r6973, r6974;
st.shared.u32 [r6975], r4383;
st.shared.u32 [r6975+4], r4386;
st.shared.u32 [r6975+256], r4585;
st.shared.u32 [r6975+260], r4594;
st.shared.u32 [r6975+512], r4622;
st.shared.u32 [r6975+516], r4631;
st.shared.u32 [r6975+768], r4659;
st.shared.u32 [r6975+772], r4668;
st.shared.u32 [r6975+1024], r4696;
st.shared.u32 [r6975+1028], r4705;
st.shared.u32 [r6975+1280], r4733;
st.shared.u32 [r6975+1284], r4742;
st.shared.u32 [r6975+1536], r4770;
st.shared.u32 [r6975+1540], r4779;
st.shared.u32 [r6975+1792], r4807;
st.shared.u32 [r6975+1796], r4816;
st.shared.u32 [r6975+2048], r4844;
st.shared.u32 [r6975+2052], r4853;
st.shared.u32 [r6975+2304], r4881;
st.shared.u32 [r6975+2308], r4890;
st.shared.u32 [r6975+2560], r4918;
st.shared.u32 [r6975+2564], r4927;
st.shared.u32 [r6975+2816], r4955;
st.shared.u32 [r6975+2820], r4964;
st.shared.u32 [r6975+3072], r4992;
st.shared.u32 [r6975+3076], r5001;
st.shared.u32 [r6975+3328], r5029;
st.shared.u32 [r6975+3332], r5038;
st.shared.u32 [r6975+3584], r5066;
st.shared.u32 [r6975+3588], r5075;
st.shared.u32 [r6975+3840], r5103;
st.shared.u32 [r6975+3844], r5112;
st.shared.u32 [r6975+4096], r5140;
st.shared.u32 [r6975+4100], r5149;
st.shared.u32 [r6975+4352], r5177;
st.shared.u32 [r6975+4356], r5186;
st.shared.u32 [r6975+4608], r5214;
st.shared.u32 [r6975+4612], r5223;
st.shared.u32 [r6975+4864], r5251;
st.shared.u32 [r6975+4868], r5260;
st.shared.u32 [r6975+5120], r5288;
st.shared.u32 [r6975+5124], r5297;
st.shared.u32 [r6975+5376], r5325;
st.shared.u32 [r6975+5380], r5334;
st.shared.u32 [r6975+5632], r5362;
st.shared.u32 [r6975+5636], r5371;
st.shared.u32 [r6975+5888], r5399;
st.shared.u32 [r6975+5892], r5408;
st.shared.u32 [r6975+6144], r5436;
st.shared.u32 [r6975+6148], r5445;
st.shared.u32 [r6975+6400], r5473;
st.shared.u32 [r6975+6404], r5482;
st.shared.u32 [r6975+6656], r5510;
st.shared.u32 [r6975+6660], r5519;
st.shared.u32 [r6975+6912], r5547;
st.shared.u32 [r6975+6916], r5556;
st.shared.u32 [r6975+7168], r5584;
st.shared.u32 [r6975+7172], r5593;
st.shared.u32 [r6975+7424], r5621;
st.shared.u32 [r6975+7428], r5630;
st.shared.u32 [r6975+7680], r5658;
st.shared.u32 [r6975+7684], r5667;
st.shared.u32 [r6975+7936], r5695;
st.shared.u32 [r6975+7940], r5704;
barrier.sync 0;
mad.lo.s32 r6976, r6977, -248, r6975;
ld.shared.u32 r5726, [r6976];
ld.shared.u32 r5729, [r6976+4];
ld.shared.u32 r6342, [r6976+4096];
ld.shared.u32 r6345, [r6976+4100];
ld.shared.u32 r5922, [r6976+8192];
ld.shared.u32 r5925, [r6976+8196];
ld.shared.u32 r6538, [r6976+12288];
ld.shared.u32 r6541, [r6976+12292];
ld.shared.u32 r5776, [r6976+16384];
ld.shared.u32 r5779, [r6976+16388];
ld.shared.u32 r6392, [r6976+20480];
ld.shared.u32 r6395, [r6976+20484];
ld.shared.u32 r5972, [r6976+24576];
ld.shared.u32 r5975, [r6976+24580];
ld.shared.u32 r6588, [r6976+28672];
ld.shared.u32 r6591, [r6976+28676];
ld.shared.u32 r5738, [r6976+32768];
ld.shared.u32 r5741, [r6976+32772];
ld.shared.u32 r6354, [r6976+36864];
ld.shared.u32 r6357, [r6976+36868];
ld.shared.u32 r5934, [r6976+40960];
ld.shared.u32 r5937, [r6976+40964];
ld.shared.u32 r6550, [r6976+45056];
ld.shared.u32 r6553, [r6976+45060];
ld.shared.u32 r5788, [r6976+49152];
ld.shared.u32 r5791, [r6976+49156];
ld.shared.u32 r6404, [r6976+53248];
ld.shared.u32 r6407, [r6976+53252];
ld.shared.u32 r5984, [r6976+57344];
ld.shared.u32 r5987, [r6976+57348];
ld.shared.u32 r6600, [r6976+61440];
ld.shared.u32 r6603, [r6976+61444];
ld.shared.u32 r5727, [r6976+65536];
ld.shared.u32 r5730, [r6976+65540];
ld.shared.u32 r6343, [r6976+69632];
ld.shared.u32 r6346, [r6976+69636];
ld.shared.u32 r5923, [r6976+73728];
ld.shared.u32 r5926, [r6976+73732];
ld.shared.u32 r6539, [r6976+77824];
ld.shared.u32 r6542, [r6976+77828];
ld.shared.u32 r5777, [r6976+81920];
ld.shared.u32 r5780, [r6976+81924];
ld.shared.u32 r6393, [r6976+86016];
ld.shared.u32 r6396, [r6976+86020];
ld.shared.u32 r5973, [r6976+90112];
ld.shared.u32 r5976, [r6976+90116];
ld.shared.u32 r6589, [r6976+94208];
ld.shared.u32 r6592, [r6976+94212];
ld.shared.u32 r5739, [r6976+98304];
ld.shared.u32 r5742, [r6976+98308];
ld.shared.u32 r6355, [r6976+102400];
ld.shared.u32 r6358, [r6976+102404];
ld.shared.u32 r5935, [r6976+106496];
ld.shared.u32 r5938, [r6976+106500];
ld.shared.u32 r6551, [r6976+110592];
ld.shared.u32 r6554, [r6976+110596];
ld.shared.u32 r5789, [r6976+114688];
ld.shared.u32 r5792, [r6976+114692];
ld.shared.u32 r6405, [r6976+118784];
ld.shared.u32 r6408, [r6976+118788];
ld.shared.u32 r5985, [r6976+122880];
ld.shared.u32 r5988, [r6976+122884];
ld.shared.u32 r6601, [r6976+126976];
ld.shared.u32 r6604, [r6976+126980];
{
add.f16x2 r5725, r5726, r5727;
}
{
add.f16x2 r5728, r5729, r5730;
}
{
sub.f16x2 r5731, r5726, r5727;
}
{
sub.f16x2 r5734, r5729, r5730;
}
{
add.f16x2 r5737, r5738, r5739;
}
{
add.f16x2 r5740, r5741, r5742;
}
{
sub.f16x2 r5743, r5738, r5739;
}
{
sub.f16x2 r5746, r5741, r5742;
}
{
neg.f16x2 r5749, r5746;
}
{
add.f16x2 r5751, r5725, r5737;
}
{
add.f16x2 r5754, r5728, r5740;
}
{
sub.f16x2 r5757, r5725, r5737;
}
{
sub.f16x2 r5760, r5728, r5740;
}
{
add.f16x2 r5763, r5731, r5749;
}
{
add.f16x2 r5766, r5734, r5743;
}
{
sub.f16x2 r5769, r5731, r5749;
}
{
sub.f16x2 r5772, r5734, r5743;
}
{
add.f16x2 r5775, r5776, r5777;
}
{
add.f16x2 r5778, r5779, r5780;
}
{
sub.f16x2 r5781, r5776, r5777;
}
{
sub.f16x2 r5784, r5779, r5780;
}
{
add.f16x2 r5787, r5788, r5789;
}
{
add.f16x2 r5790, r5791, r5792;
}
{
sub.f16x2 r5793, r5788, r5789;
}
{
sub.f16x2 r5796, r5791, r5792;
}
{
neg.f16x2 r5799, r5796;
}
{
add.f16x2 r5801, r5775, r5787;
}
{
add.f16x2 r5804, r5778, r5790;
}
{
sub.f16x2 r5807, r5775, r5787;
}
{
sub.f16x2 r5810, r5778, r5790;
}
{
add.f16x2 r5813, r5781, r5799;
}
{
add.f16x2 r5816, r5784, r5793;
}
{
sub.f16x2 r5819, r5781, r5799;
}
{
sub.f16x2 r5822, r5784, r5793;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r5825, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r5826, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r5829, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r5830, {low, high};
}
{
mul.f16x2 r5839, r5813, r5825;
}
{
mul.f16x2 r5842, r5816, r5826;
}
{
sub.f16x2 r5845, r5839, r5842;
}
{
mul.f16x2 r5848, r5813, r5826;
}
{
fma.rn.f16x2 r5851, r5816, r5825, r5848;
}
{
neg.f16x2 r5855, r5810;
}
{
mul.f16x2 r5857, r5819, r5829;
}
{
mul.f16x2 r5860, r5822, r5830;
}
{
sub.f16x2 r5863, r5857, r5860;
}
{
mul.f16x2 r5866, r5819, r5830;
}
{
fma.rn.f16x2 r5869, r5822, r5829, r5866;
}
{
add.f16x2 r5873, r5751, r5801;
}
{
add.f16x2 r5876, r5754, r5804;
}
{
sub.f16x2 r5879, r5751, r5801;
}
{
sub.f16x2 r5882, r5754, r5804;
}
{
add.f16x2 r5885, r5763, r5845;
}
{
add.f16x2 r5888, r5766, r5851;
}
{
sub.f16x2 r5891, r5763, r5845;
}
{
sub.f16x2 r5894, r5766, r5851;
}
{
add.f16x2 r5897, r5757, r5855;
}
{
add.f16x2 r5900, r5760, r5807;
}
{
sub.f16x2 r5903, r5757, r5855;
}
{
sub.f16x2 r5906, r5760, r5807;
}
{
add.f16x2 r5909, r5769, r5863;
}
{
add.f16x2 r5912, r5772, r5869;
}
{
sub.f16x2 r5915, r5769, r5863;
}
{
sub.f16x2 r5918, r5772, r5869;
}
{
add.f16x2 r5921, r5922, r5923;
}
{
add.f16x2 r5924, r5925, r5926;
}
{
sub.f16x2 r5927, r5922, r5923;
}
{
sub.f16x2 r5930, r5925, r5926;
}
{
add.f16x2 r5933, r5934, r5935;
}
{
add.f16x2 r5936, r5937, r5938;
}
{
sub.f16x2 r5939, r5934, r5935;
}
{
sub.f16x2 r5942, r5937, r5938;
}
{
neg.f16x2 r5945, r5942;
}
{
add.f16x2 r5947, r5921, r5933;
}
{
add.f16x2 r5950, r5924, r5936;
}
{
sub.f16x2 r5953, r5921, r5933;
}
{
sub.f16x2 r5956, r5924, r5936;
}
{
add.f16x2 r5959, r5927, r5945;
}
{
add.f16x2 r5962, r5930, r5939;
}
{
sub.f16x2 r5965, r5927, r5945;
}
{
sub.f16x2 r5968, r5930, r5939;
}
{
add.f16x2 r5971, r5972, r5973;
}
{
add.f16x2 r5974, r5975, r5976;
}
{
sub.f16x2 r5977, r5972, r5973;
}
{
sub.f16x2 r5980, r5975, r5976;
}
{
add.f16x2 r5983, r5984, r5985;
}
{
add.f16x2 r5986, r5987, r5988;
}
{
sub.f16x2 r5989, r5984, r5985;
}
{
sub.f16x2 r5992, r5987, r5988;
}
{
neg.f16x2 r5995, r5992;
}
{
add.f16x2 r5997, r5971, r5983;
}
{
add.f16x2 r6000, r5974, r5986;
}
{
sub.f16x2 r6003, r5971, r5983;
}
{
sub.f16x2 r6006, r5974, r5986;
}
{
add.f16x2 r6009, r5977, r5995;
}
{
add.f16x2 r6012, r5980, r5989;
}
{
sub.f16x2 r6015, r5977, r5995;
}
{
sub.f16x2 r6018, r5980, r5989;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6022, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r6025, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6026, {low, high};
}
{
mul.f16x2 r6035, r6009, r6021;
}
{
mul.f16x2 r6038, r6012, r6022;
}
{
sub.f16x2 r6041, r6035, r6038;
}
{
mul.f16x2 r6044, r6009, r6022;
}
{
fma.rn.f16x2 r6047, r6012, r6021, r6044;
}
{
neg.f16x2 r6051, r6006;
}
{
mul.f16x2 r6053, r6015, r6025;
}
{
mul.f16x2 r6056, r6018, r6026;
}
{
sub.f16x2 r6059, r6053, r6056;
}
{
mul.f16x2 r6062, r6015, r6026;
}
{
fma.rn.f16x2 r6065, r6018, r6025, r6062;
}
{
add.f16x2 r6069, r5947, r5997;
}
{
add.f16x2 r6072, r5950, r6000;
}
{
sub.f16x2 r6075, r5947, r5997;
}
{
sub.f16x2 r6078, r5950, r6000;
}
{
add.f16x2 r6081, r5959, r6041;
}
{
add.f16x2 r6084, r5962, r6047;
}
{
sub.f16x2 r6087, r5959, r6041;
}
{
sub.f16x2 r6090, r5962, r6047;
}
{
add.f16x2 r6093, r5953, r6051;
}
{
add.f16x2 r6096, r5956, r6003;
}
{
sub.f16x2 r6099, r5953, r6051;
}
{
sub.f16x2 r6102, r5956, r6003;
}
{
add.f16x2 r6105, r5965, r6059;
}
{
add.f16x2 r6108, r5968, r6065;
}
{
sub.f16x2 r6111, r5965, r6059;
}
{
sub.f16x2 r6114, r5968, r6065;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r6117, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r6118, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6119, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6120, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r6121, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r6122, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6125, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r6126, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r6127, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6128, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1042;
cvt.rn.f16.f32 high, f1042;
mov.b32 r6129, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r6130, {low, high};
}
{
mul.f16x2 r6147, r6081, r6117;
}
{
mul.f16x2 r6150, r6084, r6118;
}
{
sub.f16x2 r6153, r6147, r6150;
}
{
mul.f16x2 r6156, r6081, r6118;
}
{
fma.rn.f16x2 r6159, r6084, r6117, r6156;
}
{
mul.f16x2 r6163, r6093, r6119;
}
{
mul.f16x2 r6166, r6096, r6120;
}
{
sub.f16x2 r6169, r6163, r6166;
}
{
mul.f16x2 r6172, r6093, r6120;
}
{
fma.rn.f16x2 r6175, r6096, r6119, r6172;
}
{
mul.f16x2 r6179, r6105, r6121;
}
{
mul.f16x2 r6182, r6108, r6122;
}
{
sub.f16x2 r6185, r6179, r6182;
}
{
mul.f16x2 r6188, r6105, r6122;
}
{
fma.rn.f16x2 r6191, r6108, r6121, r6188;
}
{
neg.f16x2 r6195, r6078;
}
{
mul.f16x2 r6197, r6087, r6125;
}
{
mul.f16x2 r6200, r6090, r6126;
}
{
sub.f16x2 r6203, r6197, r6200;
}
{
mul.f16x2 r6206, r6087, r6126;
}
{
fma.rn.f16x2 r6209, r6090, r6125, r6206;
}
{
mul.f16x2 r6213, r6099, r6127;
}
{
mul.f16x2 r6216, r6102, r6128;
}
{
sub.f16x2 r6219, r6213, r6216;
}
{
mul.f16x2 r6222, r6099, r6128;
}
{
fma.rn.f16x2 r6225, r6102, r6127, r6222;
}
{
mul.f16x2 r6229, r6111, r6129;
}
{
mul.f16x2 r6232, r6114, r6130;
}
{
sub.f16x2 r6235, r6229, r6232;
}
{
mul.f16x2 r6238, r6111, r6130;
}
{
fma.rn.f16x2 r6241, r6114, r6129, r6238;
}
{
add.f16x2 %0, r5873, r6069;
}
{
add.f16x2 %1, r5876, r6072;
}
{
sub.f16x2 %32, r5873, r6069;
}
{
sub.f16x2 %33, r5876, r6072;
}
{
add.f16x2 %4, r5885, r6153;
}
{
add.f16x2 %5, r5888, r6159;
}
{
sub.f16x2 %36, r5885, r6153;
}
{
sub.f16x2 %37, r5888, r6159;
}
{
add.f16x2 %8, r5897, r6169;
}
{
add.f16x2 %9, r5900, r6175;
}
{
sub.f16x2 %40, r5897, r6169;
}
{
sub.f16x2 %41, r5900, r6175;
}
{
add.f16x2 %12, r5909, r6185;
}
{
add.f16x2 %13, r5912, r6191;
}
{
sub.f16x2 %44, r5909, r6185;
}
{
sub.f16x2 %45, r5912, r6191;
}
{
add.f16x2 %16, r5879, r6195;
}
{
add.f16x2 %17, r5882, r6075;
}
{
sub.f16x2 %48, r5879, r6195;
}
{
sub.f16x2 %49, r5882, r6075;
}
{
add.f16x2 %20, r5891, r6203;
}
{
add.f16x2 %21, r5894, r6209;
}
{
sub.f16x2 %52, r5891, r6203;
}
{
sub.f16x2 %53, r5894, r6209;
}
{
add.f16x2 %24, r5903, r6219;
}
{
add.f16x2 %25, r5906, r6225;
}
{
sub.f16x2 %56, r5903, r6219;
}
{
sub.f16x2 %57, r5906, r6225;
}
{
add.f16x2 %28, r5915, r6235;
}
{
add.f16x2 %29, r5918, r6241;
}
{
sub.f16x2 %60, r5915, r6235;
}
{
sub.f16x2 %61, r5918, r6241;
}
{
add.f16x2 r6341, r6342, r6343;
}
{
add.f16x2 r6344, r6345, r6346;
}
{
sub.f16x2 r6347, r6342, r6343;
}
{
sub.f16x2 r6350, r6345, r6346;
}
{
add.f16x2 r6353, r6354, r6355;
}
{
add.f16x2 r6356, r6357, r6358;
}
{
sub.f16x2 r6359, r6354, r6355;
}
{
sub.f16x2 r6362, r6357, r6358;
}
{
neg.f16x2 r6365, r6362;
}
{
add.f16x2 r6367, r6341, r6353;
}
{
add.f16x2 r6370, r6344, r6356;
}
{
sub.f16x2 r6373, r6341, r6353;
}
{
sub.f16x2 r6376, r6344, r6356;
}
{
add.f16x2 r6379, r6347, r6365;
}
{
add.f16x2 r6382, r6350, r6359;
}
{
sub.f16x2 r6385, r6347, r6365;
}
{
sub.f16x2 r6388, r6350, r6359;
}
{
add.f16x2 r6391, r6392, r6393;
}
{
add.f16x2 r6394, r6395, r6396;
}
{
sub.f16x2 r6397, r6392, r6393;
}
{
sub.f16x2 r6400, r6395, r6396;
}
{
add.f16x2 r6403, r6404, r6405;
}
{
add.f16x2 r6406, r6407, r6408;
}
{
sub.f16x2 r6409, r6404, r6405;
}
{
sub.f16x2 r6412, r6407, r6408;
}
{
neg.f16x2 r6415, r6412;
}
{
add.f16x2 r6417, r6391, r6403;
}
{
add.f16x2 r6420, r6394, r6406;
}
{
sub.f16x2 r6423, r6391, r6403;
}
{
sub.f16x2 r6426, r6394, r6406;
}
{
add.f16x2 r6429, r6397, r6415;
}
{
add.f16x2 r6432, r6400, r6409;
}
{
sub.f16x2 r6435, r6397, r6415;
}
{
sub.f16x2 r6438, r6400, r6409;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6441, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6442, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r6445, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6446, {low, high};
}
{
mul.f16x2 r6455, r6429, r6441;
}
{
mul.f16x2 r6458, r6432, r6442;
}
{
sub.f16x2 r6461, r6455, r6458;
}
{
mul.f16x2 r6464, r6429, r6442;
}
{
fma.rn.f16x2 r6467, r6432, r6441, r6464;
}
{
neg.f16x2 r6471, r6426;
}
{
mul.f16x2 r6473, r6435, r6445;
}
{
mul.f16x2 r6476, r6438, r6446;
}
{
sub.f16x2 r6479, r6473, r6476;
}
{
mul.f16x2 r6482, r6435, r6446;
}
{
fma.rn.f16x2 r6485, r6438, r6445, r6482;
}
{
add.f16x2 r6489, r6367, r6417;
}
{
add.f16x2 r6492, r6370, r6420;
}
{
sub.f16x2 r6495, r6367, r6417;
}
{
sub.f16x2 r6498, r6370, r6420;
}
{
add.f16x2 r6501, r6379, r6461;
}
{
add.f16x2 r6504, r6382, r6467;
}
{
sub.f16x2 r6507, r6379, r6461;
}
{
sub.f16x2 r6510, r6382, r6467;
}
{
add.f16x2 r6513, r6373, r6471;
}
{
add.f16x2 r6516, r6376, r6423;
}
{
sub.f16x2 r6519, r6373, r6471;
}
{
sub.f16x2 r6522, r6376, r6423;
}
{
add.f16x2 r6525, r6385, r6479;
}
{
add.f16x2 r6528, r6388, r6485;
}
{
sub.f16x2 r6531, r6385, r6479;
}
{
sub.f16x2 r6534, r6388, r6485;
}
{
add.f16x2 r6537, r6538, r6539;
}
{
add.f16x2 r6540, r6541, r6542;
}
{
sub.f16x2 r6543, r6538, r6539;
}
{
sub.f16x2 r6546, r6541, r6542;
}
{
add.f16x2 r6549, r6550, r6551;
}
{
add.f16x2 r6552, r6553, r6554;
}
{
sub.f16x2 r6555, r6550, r6551;
}
{
sub.f16x2 r6558, r6553, r6554;
}
{
neg.f16x2 r6561, r6558;
}
{
add.f16x2 r6563, r6537, r6549;
}
{
add.f16x2 r6566, r6540, r6552;
}
{
sub.f16x2 r6569, r6537, r6549;
}
{
sub.f16x2 r6572, r6540, r6552;
}
{
add.f16x2 r6575, r6543, r6561;
}
{
add.f16x2 r6578, r6546, r6555;
}
{
sub.f16x2 r6581, r6543, r6561;
}
{
sub.f16x2 r6584, r6546, r6555;
}
{
add.f16x2 r6587, r6588, r6589;
}
{
add.f16x2 r6590, r6591, r6592;
}
{
sub.f16x2 r6593, r6588, r6589;
}
{
sub.f16x2 r6596, r6591, r6592;
}
{
add.f16x2 r6599, r6600, r6601;
}
{
add.f16x2 r6602, r6603, r6604;
}
{
sub.f16x2 r6605, r6600, r6601;
}
{
sub.f16x2 r6608, r6603, r6604;
}
{
neg.f16x2 r6611, r6608;
}
{
add.f16x2 r6613, r6587, r6599;
}
{
add.f16x2 r6616, r6590, r6602;
}
{
sub.f16x2 r6619, r6587, r6599;
}
{
sub.f16x2 r6622, r6590, r6602;
}
{
add.f16x2 r6625, r6593, r6611;
}
{
add.f16x2 r6628, r6596, r6605;
}
{
sub.f16x2 r6631, r6593, r6611;
}
{
sub.f16x2 r6634, r6596, r6605;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6637, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6638, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r6641, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6642, {low, high};
}
{
mul.f16x2 r6651, r6625, r6637;
}
{
mul.f16x2 r6654, r6628, r6638;
}
{
sub.f16x2 r6657, r6651, r6654;
}
{
mul.f16x2 r6660, r6625, r6638;
}
{
fma.rn.f16x2 r6663, r6628, r6637, r6660;
}
{
neg.f16x2 r6667, r6622;
}
{
mul.f16x2 r6669, r6631, r6641;
}
{
mul.f16x2 r6672, r6634, r6642;
}
{
sub.f16x2 r6675, r6669, r6672;
}
{
mul.f16x2 r6678, r6631, r6642;
}
{
fma.rn.f16x2 r6681, r6634, r6641, r6678;
}
{
add.f16x2 r6685, r6563, r6613;
}
{
add.f16x2 r6688, r6566, r6616;
}
{
sub.f16x2 r6691, r6563, r6613;
}
{
sub.f16x2 r6694, r6566, r6616;
}
{
add.f16x2 r6697, r6575, r6657;
}
{
add.f16x2 r6700, r6578, r6663;
}
{
sub.f16x2 r6703, r6575, r6657;
}
{
sub.f16x2 r6706, r6578, r6663;
}
{
add.f16x2 r6709, r6569, r6667;
}
{
add.f16x2 r6712, r6572, r6619;
}
{
sub.f16x2 r6715, r6569, r6667;
}
{
sub.f16x2 r6718, r6572, r6619;
}
{
add.f16x2 r6721, r6581, r6675;
}
{
add.f16x2 r6724, r6584, r6681;
}
{
sub.f16x2 r6727, r6581, r6675;
}
{
sub.f16x2 r6730, r6584, r6681;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r6733, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r6734, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6735, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6736, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r6737, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r6738, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6741, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r6742, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r6743, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6744, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1042;
cvt.rn.f16.f32 high, f1042;
mov.b32 r6745, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r6746, {low, high};
}
{
mul.f16x2 r6763, r6697, r6733;
}
{
mul.f16x2 r6766, r6700, r6734;
}
{
sub.f16x2 r6769, r6763, r6766;
}
{
mul.f16x2 r6772, r6697, r6734;
}
{
fma.rn.f16x2 r6775, r6700, r6733, r6772;
}
{
mul.f16x2 r6779, r6709, r6735;
}
{
mul.f16x2 r6782, r6712, r6736;
}
{
sub.f16x2 r6785, r6779, r6782;
}
{
mul.f16x2 r6788, r6709, r6736;
}
{
fma.rn.f16x2 r6791, r6712, r6735, r6788;
}
{
mul.f16x2 r6795, r6721, r6737;
}
{
mul.f16x2 r6798, r6724, r6738;
}
{
sub.f16x2 r6801, r6795, r6798;
}
{
mul.f16x2 r6804, r6721, r6738;
}
{
fma.rn.f16x2 r6807, r6724, r6737, r6804;
}
{
neg.f16x2 r6811, r6694;
}
{
mul.f16x2 r6813, r6703, r6741;
}
{
mul.f16x2 r6816, r6706, r6742;
}
{
sub.f16x2 r6819, r6813, r6816;
}
{
mul.f16x2 r6822, r6703, r6742;
}
{
fma.rn.f16x2 r6825, r6706, r6741, r6822;
}
{
mul.f16x2 r6829, r6715, r6743;
}
{
mul.f16x2 r6832, r6718, r6744;
}
{
sub.f16x2 r6835, r6829, r6832;
}
{
mul.f16x2 r6838, r6715, r6744;
}
{
fma.rn.f16x2 r6841, r6718, r6743, r6838;
}
{
mul.f16x2 r6845, r6727, r6745;
}
{
mul.f16x2 r6848, r6730, r6746;
}
{
sub.f16x2 r6851, r6845, r6848;
}
{
mul.f16x2 r6854, r6727, r6746;
}
{
fma.rn.f16x2 r6857, r6730, r6745, r6854;
}
{
add.f16x2 %2, r6489, r6685;
}
{
add.f16x2 %3, r6492, r6688;
}
{
sub.f16x2 %34, r6489, r6685;
}
{
sub.f16x2 %35, r6492, r6688;
}
{
add.f16x2 %6, r6501, r6769;
}
{
add.f16x2 %7, r6504, r6775;
}
{
sub.f16x2 %38, r6501, r6769;
}
{
sub.f16x2 %39, r6504, r6775;
}
{
add.f16x2 %10, r6513, r6785;
}
{
add.f16x2 %11, r6516, r6791;
}
{
sub.f16x2 %42, r6513, r6785;
}
{
sub.f16x2 %43, r6516, r6791;
}
{
add.f16x2 %14, r6525, r6801;
}
{
add.f16x2 %15, r6528, r6807;
}
{
sub.f16x2 %46, r6525, r6801;
}
{
sub.f16x2 %47, r6528, r6807;
}
{
add.f16x2 %18, r6495, r6811;
}
{
add.f16x2 %19, r6498, r6691;
}
{
sub.f16x2 %50, r6495, r6811;
}
{
sub.f16x2 %51, r6498, r6691;
}
{
add.f16x2 %22, r6507, r6819;
}
{
add.f16x2 %23, r6510, r6825;
}
{
sub.f16x2 %54, r6507, r6819;
}
{
sub.f16x2 %55, r6510, r6825;
}
{
add.f16x2 %26, r6519, r6835;
}
{
add.f16x2 %27, r6522, r6841;
}
{
sub.f16x2 %58, r6519, r6835;
}
{
sub.f16x2 %59, r6522, r6841;
}
{
add.f16x2 %30, r6531, r6851;
}
{
add.f16x2 %31, r6534, r6857;
}
{
sub.f16x2 %62, r6531, r6851;
}
{
sub.f16x2 %63, r6534, r6857;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)), "=r"(__HALF2_TO_UI(rmem[27].x)), "=r"(__HALF2_TO_UI(rmem[27].y)), "=r"(__HALF2_TO_UI(rmem[28].x)), "=r"(__HALF2_TO_UI(rmem[28].y)), "=r"(__HALF2_TO_UI(rmem[29].x)), "=r"(__HALF2_TO_UI(rmem[29].y)), "=r"(__HALF2_TO_UI(rmem[30].x)), "=r"(__HALF2_TO_UI(rmem[30].y)), "=r"(__HALF2_TO_UI(rmem[31].x)), "=r"(__HALF2_TO_UI(rmem[31].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[31].y)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[28].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[27].y)), "r"(__HALF2_TO_UI(rmem[30].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[29].y)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[31].x)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[28].x)), "r"(__HALF2_TO_UI(rmem[27].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[30].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[29].x)), "r"(__HALF2_TO_UI(rmem[24].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1063, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<460>;
.reg .b32 r<3750>;
.reg .b64 rd<2>;
mov.u32 r3723, %tid.y;
shl.b32 r3724, r3723, 16;
mov.u32 r3725, %32;
add.s32 r3726, r3725, r3724;
mov.u32 r3727, %tid.x;
{
add.f16x2 r1, %33, %49;
}
{
add.f16x2 r4, %34, %50;
}
{
sub.f16x2 r7, %33, %49;
}
{
sub.f16x2 r10, %34, %50;
}
{
add.f16x2 r13, %41, %57;
}
{
add.f16x2 r16, %42, %58;
}
{
sub.f16x2 r19, %41, %57;
}
{
sub.f16x2 r22, %42, %58;
}
{
neg.f16x2 r25, r22;
}
{
add.f16x2 r27, r1, r13;
}
{
add.f16x2 r30, r4, r16;
}
{
sub.f16x2 r33, r1, r13;
}
{
sub.f16x2 r36, r4, r16;
}
{
add.f16x2 r39, r7, r25;
}
{
add.f16x2 r42, r10, r19;
}
{
sub.f16x2 r45, r7, r25;
}
{
sub.f16x2 r48, r10, r19;
}
{
add.f16x2 r51, %37, %53;
}
{
add.f16x2 r54, %38, %54;
}
{
sub.f16x2 r57, %37, %53;
}
{
sub.f16x2 r60, %38, %54;
}
{
add.f16x2 r63, %45, %61;
}
{
add.f16x2 r66, %46, %62;
}
{
sub.f16x2 r69, %45, %61;
}
{
sub.f16x2 r72, %46, %62;
}
{
neg.f16x2 r75, r72;
}
{
add.f16x2 r77, r51, r63;
}
{
add.f16x2 r80, r54, r66;
}
{
sub.f16x2 r83, r51, r63;
}
{
sub.f16x2 r86, r54, r66;
}
{
add.f16x2 r89, r57, r75;
}
{
add.f16x2 r92, r60, r69;
}
{
sub.f16x2 r95, r57, r75;
}
{
sub.f16x2 r98, r60, r69;
}
mov.f32 f380, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r102, {low, high};
}
mov.f32 f448, 0f3F800000;
mov.f32 f378, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f378;
cvt.rn.f16.f32 high, f378;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r106, {low, high};
}
mov.f32 f447, 0fBF800000;
{
mul.f16x2 r115, r89, r101;
}
{
mul.f16x2 r118, r92, r102;
}
{
sub.f16x2 r121, r115, r118;
}
{
mul.f16x2 r124, r89, r102;
}
{
fma.rn.f16x2 r127, r92, r101, r124;
}
{
neg.f16x2 r131, r86;
}
{
mul.f16x2 r133, r95, r105;
}
{
mul.f16x2 r136, r98, r106;
}
{
sub.f16x2 r139, r133, r136;
}
{
mul.f16x2 r142, r95, r106;
}
{
fma.rn.f16x2 r145, r98, r105, r142;
}
{
add.f16x2 r149, r27, r77;
}
{
add.f16x2 r152, r30, r80;
}
{
sub.f16x2 r155, r27, r77;
}
{
sub.f16x2 r158, r30, r80;
}
{
add.f16x2 r161, r39, r121;
}
{
add.f16x2 r164, r42, r127;
}
{
sub.f16x2 r167, r39, r121;
}
{
sub.f16x2 r170, r42, r127;
}
{
add.f16x2 r173, r33, r131;
}
{
add.f16x2 r176, r36, r83;
}
{
sub.f16x2 r179, r33, r131;
}
{
sub.f16x2 r182, r36, r83;
}
{
add.f16x2 r185, r45, r139;
}
{
add.f16x2 r188, r48, r145;
}
{
sub.f16x2 r191, r45, r139;
}
{
sub.f16x2 r194, r48, r145;
}
{
add.f16x2 r197, %35, %51;
}
{
add.f16x2 r200, %36, %52;
}
{
sub.f16x2 r203, %35, %51;
}
{
sub.f16x2 r206, %36, %52;
}
{
add.f16x2 r209, %43, %59;
}
{
add.f16x2 r212, %44, %60;
}
{
sub.f16x2 r215, %43, %59;
}
{
sub.f16x2 r218, %44, %60;
}
{
neg.f16x2 r221, r218;
}
{
add.f16x2 r223, r197, r209;
}
{
add.f16x2 r226, r200, r212;
}
{
sub.f16x2 r229, r197, r209;
}
{
sub.f16x2 r232, r200, r212;
}
{
add.f16x2 r235, r203, r221;
}
{
add.f16x2 r238, r206, r215;
}
{
sub.f16x2 r241, r203, r221;
}
{
sub.f16x2 r244, r206, r215;
}
{
add.f16x2 r247, %39, %55;
}
{
add.f16x2 r250, %40, %56;
}
{
sub.f16x2 r253, %39, %55;
}
{
sub.f16x2 r256, %40, %56;
}
{
add.f16x2 r259, %47, %63;
}
{
add.f16x2 r262, %48, %64;
}
{
sub.f16x2 r265, %47, %63;
}
{
sub.f16x2 r268, %48, %64;
}
{
neg.f16x2 r271, r268;
}
{
add.f16x2 r273, r247, r259;
}
{
add.f16x2 r276, r250, r262;
}
{
sub.f16x2 r279, r247, r259;
}
{
sub.f16x2 r282, r250, r262;
}
{
add.f16x2 r285, r253, r271;
}
{
add.f16x2 r288, r256, r265;
}
{
sub.f16x2 r291, r253, r271;
}
{
sub.f16x2 r294, r256, r265;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f378;
cvt.rn.f16.f32 high, f378;
mov.b32 r301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r302, {low, high};
}
{
mul.f16x2 r311, r285, r297;
}
{
mul.f16x2 r314, r288, r298;
}
{
sub.f16x2 r317, r311, r314;
}
{
mul.f16x2 r320, r285, r298;
}
{
fma.rn.f16x2 r323, r288, r297, r320;
}
{
neg.f16x2 r327, r282;
}
{
mul.f16x2 r329, r291, r301;
}
{
mul.f16x2 r332, r294, r302;
}
{
sub.f16x2 r335, r329, r332;
}
{
mul.f16x2 r338, r291, r302;
}
{
fma.rn.f16x2 r341, r294, r301, r338;
}
{
add.f16x2 r345, r223, r273;
}
{
add.f16x2 r348, r226, r276;
}
{
sub.f16x2 r351, r223, r273;
}
{
sub.f16x2 r354, r226, r276;
}
{
add.f16x2 r357, r235, r317;
}
{
add.f16x2 r360, r238, r323;
}
{
sub.f16x2 r363, r235, r317;
}
{
sub.f16x2 r366, r238, r323;
}
{
add.f16x2 r369, r229, r327;
}
{
add.f16x2 r372, r232, r279;
}
{
sub.f16x2 r375, r229, r327;
}
{
sub.f16x2 r378, r232, r279;
}
{
add.f16x2 r381, r241, r335;
}
{
add.f16x2 r384, r244, r341;
}
{
sub.f16x2 r387, r241, r335;
}
{
sub.f16x2 r390, r244, r341;
}
mov.f32 f376, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r393, {low, high};
}
mov.f32 f384, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r396, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r398, {low, high};
}
mov.f32 f374, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f374;
cvt.rn.f16.f32 high, f374;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f378;
cvt.rn.f16.f32 high, f378;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r404, {low, high};
}
mov.f32 f382, 0fBF6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f382;
cvt.rn.f16.f32 high, f382;
mov.b32 r405, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r406, {low, high};
}
{
mul.f16x2 r423, r357, r393;
}
{
mul.f16x2 r426, r360, r394;
}
{
sub.f16x2 r429, r423, r426;
}
{
mul.f16x2 r432, r357, r394;
}
{
fma.rn.f16x2 r435, r360, r393, r432;
}
{
mul.f16x2 r439, r369, r395;
}
{
mul.f16x2 r442, r372, r396;
}
{
sub.f16x2 r445, r439, r442;
}
{
mul.f16x2 r448, r369, r396;
}
{
fma.rn.f16x2 r451, r372, r395, r448;
}
{
mul.f16x2 r455, r381, r397;
}
{
mul.f16x2 r458, r384, r398;
}
{
sub.f16x2 r461, r455, r458;
}
{
mul.f16x2 r464, r381, r398;
}
{
fma.rn.f16x2 r467, r384, r397, r464;
}
{
neg.f16x2 r471, r354;
}
{
mul.f16x2 r473, r363, r401;
}
{
mul.f16x2 r476, r366, r402;
}
{
sub.f16x2 r479, r473, r476;
}
{
mul.f16x2 r482, r363, r402;
}
{
fma.rn.f16x2 r485, r366, r401, r482;
}
{
mul.f16x2 r489, r375, r403;
}
{
mul.f16x2 r492, r378, r404;
}
{
sub.f16x2 r495, r489, r492;
}
{
mul.f16x2 r498, r375, r404;
}
{
fma.rn.f16x2 r501, r378, r403, r498;
}
{
mul.f16x2 r505, r387, r405;
}
{
mul.f16x2 r508, r390, r406;
}
{
sub.f16x2 r511, r505, r508;
}
{
mul.f16x2 r514, r387, r406;
}
{
fma.rn.f16x2 r517, r390, r405, r514;
}
{
add.f16x2 r521, r149, r345;
}
{
add.f16x2 r524, r152, r348;
}
{
sub.f16x2 r527, r149, r345;
}
{
sub.f16x2 r530, r152, r348;
}
{
add.f16x2 r533, r161, r429;
}
{
add.f16x2 r536, r164, r435;
}
{
sub.f16x2 r539, r161, r429;
}
{
sub.f16x2 r542, r164, r435;
}
{
add.f16x2 r545, r173, r445;
}
{
add.f16x2 r548, r176, r451;
}
{
sub.f16x2 r551, r173, r445;
}
{
sub.f16x2 r554, r176, r451;
}
{
add.f16x2 r557, r185, r461;
}
{
add.f16x2 r560, r188, r467;
}
{
sub.f16x2 r563, r185, r461;
}
{
sub.f16x2 r566, r188, r467;
}
{
add.f16x2 r569, r155, r471;
}
{
add.f16x2 r572, r158, r351;
}
{
sub.f16x2 r575, r155, r471;
}
{
sub.f16x2 r578, r158, r351;
}
{
add.f16x2 r581, r167, r479;
}
{
add.f16x2 r584, r170, r485;
}
{
sub.f16x2 r587, r167, r479;
}
{
sub.f16x2 r590, r170, r485;
}
{
add.f16x2 r593, r179, r495;
}
{
add.f16x2 r596, r182, r501;
}
{
sub.f16x2 r599, r179, r495;
}
{
sub.f16x2 r602, r182, r501;
}
{
add.f16x2 r605, r191, r511;
}
{
add.f16x2 r608, r194, r517;
}
{
sub.f16x2 r611, r191, r511;
}
{
sub.f16x2 r614, r194, r517;
}
and.b32 r3728, r3727, 1023;
shl.b32 r3729, r3727, 6;
and.b32 r3730, r3729, -65536;
add.s32 r3731, r3726, r3730;
cvt.rn.f32.u32 f451, r3728;
mul.f32 f452, f451, 0f39C90FDB;
cos.approx.f32 f117, f452;
sin.approx.f32 f453, f452;
neg.f32 f118, f453;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f117;
cvt.rn.f16.f32 high, f118;
mov.b32 r617, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r620, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r622, {high, high};
}
{
mul.f16x2 r624, r536, r622;
}
{
fma.rn.f16x2 r627, r533, r620, r624;
}
{
mul.f16x2 r631, r533, r622;
}
{
neg.f16x2 r634, r631;
}
{
fma.rn.f16x2 r636, r536, r620, r634;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r640, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r642, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r644, {low, high};
}
{
mul.f16x2 r645, r642, r644;
}
{
mul.f16x2 r648, r617, r640;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r651, {high, low};
}
{
fma.rn.f16x2 r653, r645, r651, r648;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r653;
mov.b32 r657, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r653;
mov.b32 r659, {high, high};
}
{
mul.f16x2 r661, r548, r659;
}
{
fma.rn.f16x2 r664, r545, r657, r661;
}
{
mul.f16x2 r668, r545, r659;
}
{
neg.f16x2 r671, r668;
}
{
fma.rn.f16x2 r673, r548, r657, r671;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r677, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r679, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r681, {low, high};
}
{
mul.f16x2 r682, r679, r681;
}
{
mul.f16x2 r685, r653, r677;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r653;
mov.b32 r688, {high, low};
}
{
fma.rn.f16x2 r690, r682, r688, r685;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r690;
mov.b32 r694, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r690;
mov.b32 r696, {high, high};
}
{
mul.f16x2 r698, r560, r696;
}
{
fma.rn.f16x2 r701, r557, r694, r698;
}
{
mul.f16x2 r705, r557, r696;
}
{
neg.f16x2 r708, r705;
}
{
fma.rn.f16x2 r710, r560, r694, r708;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r714, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r716, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r718, {low, high};
}
{
mul.f16x2 r719, r716, r718;
}
{
mul.f16x2 r722, r690, r714;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r690;
mov.b32 r725, {high, low};
}
{
fma.rn.f16x2 r727, r719, r725, r722;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r727;
mov.b32 r731, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r727;
mov.b32 r733, {high, high};
}
{
mul.f16x2 r735, r572, r733;
}
{
fma.rn.f16x2 r738, r569, r731, r735;
}
{
mul.f16x2 r742, r569, r733;
}
{
neg.f16x2 r745, r742;
}
{
fma.rn.f16x2 r747, r572, r731, r745;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r751, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r753, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r755, {low, high};
}
{
mul.f16x2 r756, r753, r755;
}
{
mul.f16x2 r759, r727, r751;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r727;
mov.b32 r762, {high, low};
}
{
fma.rn.f16x2 r764, r756, r762, r759;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r764;
mov.b32 r768, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r764;
mov.b32 r770, {high, high};
}
{
mul.f16x2 r772, r584, r770;
}
{
fma.rn.f16x2 r775, r581, r768, r772;
}
{
mul.f16x2 r779, r581, r770;
}
{
neg.f16x2 r782, r779;
}
{
fma.rn.f16x2 r784, r584, r768, r782;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r788, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r790, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r792, {low, high};
}
{
mul.f16x2 r793, r790, r792;
}
{
mul.f16x2 r796, r764, r788;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r764;
mov.b32 r799, {high, low};
}
{
fma.rn.f16x2 r801, r793, r799, r796;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r801;
mov.b32 r805, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r801;
mov.b32 r807, {high, high};
}
{
mul.f16x2 r809, r596, r807;
}
{
fma.rn.f16x2 r812, r593, r805, r809;
}
{
mul.f16x2 r816, r593, r807;
}
{
neg.f16x2 r819, r816;
}
{
fma.rn.f16x2 r821, r596, r805, r819;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r825, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r827, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r829, {low, high};
}
{
mul.f16x2 r830, r827, r829;
}
{
mul.f16x2 r833, r801, r825;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r801;
mov.b32 r836, {high, low};
}
{
fma.rn.f16x2 r838, r830, r836, r833;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r838;
mov.b32 r842, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r838;
mov.b32 r844, {high, high};
}
{
mul.f16x2 r846, r608, r844;
}
{
fma.rn.f16x2 r849, r605, r842, r846;
}
{
mul.f16x2 r853, r605, r844;
}
{
neg.f16x2 r856, r853;
}
{
fma.rn.f16x2 r858, r608, r842, r856;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r862, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r864, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r866, {low, high};
}
{
mul.f16x2 r867, r864, r866;
}
{
mul.f16x2 r870, r838, r862;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r838;
mov.b32 r873, {high, low};
}
{
fma.rn.f16x2 r875, r867, r873, r870;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r875;
mov.b32 r879, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r875;
mov.b32 r881, {high, high};
}
{
mul.f16x2 r883, r530, r881;
}
{
fma.rn.f16x2 r886, r527, r879, r883;
}
{
mul.f16x2 r890, r527, r881;
}
{
neg.f16x2 r893, r890;
}
{
fma.rn.f16x2 r895, r530, r879, r893;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r899, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r901, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r903, {low, high};
}
{
mul.f16x2 r904, r901, r903;
}
{
mul.f16x2 r907, r875, r899;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r875;
mov.b32 r910, {high, low};
}
{
fma.rn.f16x2 r912, r904, r910, r907;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r912;
mov.b32 r916, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r912;
mov.b32 r918, {high, high};
}
{
mul.f16x2 r920, r542, r918;
}
{
fma.rn.f16x2 r923, r539, r916, r920;
}
{
mul.f16x2 r927, r539, r918;
}
{
neg.f16x2 r930, r927;
}
{
fma.rn.f16x2 r932, r542, r916, r930;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r936, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r938, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r940, {low, high};
}
{
mul.f16x2 r941, r938, r940;
}
{
mul.f16x2 r944, r912, r936;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r912;
mov.b32 r947, {high, low};
}
{
fma.rn.f16x2 r949, r941, r947, r944;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r949;
mov.b32 r953, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r949;
mov.b32 r955, {high, high};
}
{
mul.f16x2 r957, r554, r955;
}
{
fma.rn.f16x2 r960, r551, r953, r957;
}
{
mul.f16x2 r964, r551, r955;
}
{
neg.f16x2 r967, r964;
}
{
fma.rn.f16x2 r969, r554, r953, r967;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r973, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r975, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r977, {low, high};
}
{
mul.f16x2 r978, r975, r977;
}
{
mul.f16x2 r981, r949, r973;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r949;
mov.b32 r984, {high, low};
}
{
fma.rn.f16x2 r986, r978, r984, r981;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r986;
mov.b32 r990, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r986;
mov.b32 r992, {high, high};
}
{
mul.f16x2 r994, r566, r992;
}
{
fma.rn.f16x2 r997, r563, r990, r994;
}
{
mul.f16x2 r1001, r563, r992;
}
{
neg.f16x2 r1004, r1001;
}
{
fma.rn.f16x2 r1006, r566, r990, r1004;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1010, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1012, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r1014, {low, high};
}
{
mul.f16x2 r1015, r1012, r1014;
}
{
mul.f16x2 r1018, r986, r1010;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r986;
mov.b32 r1021, {high, low};
}
{
fma.rn.f16x2 r1023, r1015, r1021, r1018;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1023;
mov.b32 r1027, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1023;
mov.b32 r1029, {high, high};
}
{
mul.f16x2 r1031, r578, r1029;
}
{
fma.rn.f16x2 r1034, r575, r1027, r1031;
}
{
mul.f16x2 r1038, r575, r1029;
}
{
neg.f16x2 r1041, r1038;
}
{
fma.rn.f16x2 r1043, r578, r1027, r1041;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1047, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1049, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r1051, {low, high};
}
{
mul.f16x2 r1052, r1049, r1051;
}
{
mul.f16x2 r1055, r1023, r1047;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1023;
mov.b32 r1058, {high, low};
}
{
fma.rn.f16x2 r1060, r1052, r1058, r1055;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1060;
mov.b32 r1064, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1060;
mov.b32 r1066, {high, high};
}
{
mul.f16x2 r1068, r590, r1066;
}
{
fma.rn.f16x2 r1071, r587, r1064, r1068;
}
{
mul.f16x2 r1075, r587, r1066;
}
{
neg.f16x2 r1078, r1075;
}
{
fma.rn.f16x2 r1080, r590, r1064, r1078;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1084, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1086, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r1088, {low, high};
}
{
mul.f16x2 r1089, r1086, r1088;
}
{
mul.f16x2 r1092, r1060, r1084;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1060;
mov.b32 r1095, {high, low};
}
{
fma.rn.f16x2 r1097, r1089, r1095, r1092;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1097;
mov.b32 r1101, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1097;
mov.b32 r1103, {high, high};
}
{
mul.f16x2 r1105, r602, r1103;
}
{
fma.rn.f16x2 r1108, r599, r1101, r1105;
}
{
mul.f16x2 r1112, r599, r1103;
}
{
neg.f16x2 r1115, r1112;
}
{
fma.rn.f16x2 r1117, r602, r1101, r1115;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1121, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1123, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r1125, {low, high};
}
{
mul.f16x2 r1126, r1123, r1125;
}
{
mul.f16x2 r1129, r1097, r1121;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1097;
mov.b32 r1132, {high, low};
}
{
fma.rn.f16x2 r1134, r1126, r1132, r1129;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1134;
mov.b32 r1138, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1134;
mov.b32 r1140, {high, high};
}
{
mul.f16x2 r1142, r614, r1140;
}
{
fma.rn.f16x2 r1145, r611, r1138, r1142;
}
{
mul.f16x2 r1149, r611, r1140;
}
{
neg.f16x2 r1152, r1149;
}
{
fma.rn.f16x2 r1154, r614, r1138, r1152;
}
barrier.sync 0;
and.b32 r3732, r3729, 65472;
add.s32 r3733, r3731, r3732;
st.shared.v4.f32 [r3733], {r521, r627, r664, r701};
st.shared.v4.f32 [r3733+16], {r738, r775, r812, r849};
st.shared.v4.f32 [r3733+32], {r886, r923, r960, r997};
st.shared.v4.f32 [r3733+48], {r1034, r1071, r1108, r1145};
barrier.sync 0;
mad.lo.s32 r3734, r3728, -60, r3733;
ld.shared.u32 r1176, [r3734];
ld.shared.u32 r1372, [r3734+4096];
ld.shared.u32 r1226, [r3734+8192];
ld.shared.u32 r1422, [r3734+12288];
ld.shared.u32 r1188, [r3734+16384];
ld.shared.u32 r1384, [r3734+20480];
ld.shared.u32 r1238, [r3734+24576];
ld.shared.u32 r1434, [r3734+28672];
ld.shared.u32 r1177, [r3734+32768];
ld.shared.u32 r1373, [r3734+36864];
ld.shared.u32 r1227, [r3734+40960];
ld.shared.u32 r1423, [r3734+45056];
ld.shared.u32 r1189, [r3734+49152];
ld.shared.u32 r1385, [r3734+53248];
ld.shared.u32 r1239, [r3734+57344];
ld.shared.u32 r1435, [r3734+61440];
barrier.sync 0;
st.shared.v4.f32 [r3733], {r524, r636, r673, r710};
st.shared.v4.f32 [r3733+16], {r747, r784, r821, r858};
st.shared.v4.f32 [r3733+32], {r895, r932, r969, r1006};
st.shared.v4.f32 [r3733+48], {r1043, r1080, r1117, r1154};
barrier.sync 0;
ld.shared.u32 r1179, [r3734];
ld.shared.u32 r1375, [r3734+4096];
ld.shared.u32 r1229, [r3734+8192];
ld.shared.u32 r1425, [r3734+12288];
ld.shared.u32 r1191, [r3734+16384];
ld.shared.u32 r1387, [r3734+20480];
ld.shared.u32 r1241, [r3734+24576];
ld.shared.u32 r1437, [r3734+28672];
ld.shared.u32 r1180, [r3734+32768];
ld.shared.u32 r1376, [r3734+36864];
ld.shared.u32 r1230, [r3734+40960];
ld.shared.u32 r1426, [r3734+45056];
ld.shared.u32 r1192, [r3734+49152];
ld.shared.u32 r1388, [r3734+53248];
ld.shared.u32 r1242, [r3734+57344];
ld.shared.u32 r1438, [r3734+61440];
{
add.f16x2 r1175, r1176, r1177;
}
{
add.f16x2 r1178, r1179, r1180;
}
{
sub.f16x2 r1181, r1176, r1177;
}
{
sub.f16x2 r1184, r1179, r1180;
}
{
add.f16x2 r1187, r1188, r1189;
}
{
add.f16x2 r1190, r1191, r1192;
}
{
sub.f16x2 r1193, r1188, r1189;
}
{
sub.f16x2 r1196, r1191, r1192;
}
{
neg.f16x2 r1199, r1196;
}
{
add.f16x2 r1201, r1175, r1187;
}
{
add.f16x2 r1204, r1178, r1190;
}
{
sub.f16x2 r1207, r1175, r1187;
}
{
sub.f16x2 r1210, r1178, r1190;
}
{
add.f16x2 r1213, r1181, r1199;
}
{
add.f16x2 r1216, r1184, r1193;
}
{
sub.f16x2 r1219, r1181, r1199;
}
{
sub.f16x2 r1222, r1184, r1193;
}
{
add.f16x2 r1225, r1226, r1227;
}
{
add.f16x2 r1228, r1229, r1230;
}
{
sub.f16x2 r1231, r1226, r1227;
}
{
sub.f16x2 r1234, r1229, r1230;
}
{
add.f16x2 r1237, r1238, r1239;
}
{
add.f16x2 r1240, r1241, r1242;
}
{
sub.f16x2 r1243, r1238, r1239;
}
{
sub.f16x2 r1246, r1241, r1242;
}
{
neg.f16x2 r1249, r1246;
}
{
add.f16x2 r1251, r1225, r1237;
}
{
add.f16x2 r1254, r1228, r1240;
}
{
sub.f16x2 r1257, r1225, r1237;
}
{
sub.f16x2 r1260, r1228, r1240;
}
{
add.f16x2 r1263, r1231, r1249;
}
{
add.f16x2 r1266, r1234, r1243;
}
{
sub.f16x2 r1269, r1231, r1249;
}
{
sub.f16x2 r1272, r1234, r1243;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r1275, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r1276, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f378;
cvt.rn.f16.f32 high, f378;
mov.b32 r1279, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r1280, {low, high};
}
{
mul.f16x2 r1289, r1263, r1275;
}
{
mul.f16x2 r1292, r1266, r1276;
}
{
sub.f16x2 r1295, r1289, r1292;
}
{
mul.f16x2 r1298, r1263, r1276;
}
{
fma.rn.f16x2 r1301, r1266, r1275, r1298;
}
{
neg.f16x2 r1305, r1260;
}
{
mul.f16x2 r1307, r1269, r1279;
}
{
mul.f16x2 r1310, r1272, r1280;
}
{
sub.f16x2 r1313, r1307, r1310;
}
{
mul.f16x2 r1316, r1269, r1280;
}
{
fma.rn.f16x2 r1319, r1272, r1279, r1316;
}
{
add.f16x2 r1323, r1201, r1251;
}
{
add.f16x2 r1326, r1204, r1254;
}
{
sub.f16x2 r1329, r1201, r1251;
}
{
sub.f16x2 r1332, r1204, r1254;
}
{
add.f16x2 r1335, r1213, r1295;
}
{
add.f16x2 r1338, r1216, r1301;
}
{
sub.f16x2 r1341, r1213, r1295;
}
{
sub.f16x2 r1344, r1216, r1301;
}
{
add.f16x2 r1347, r1207, r1305;
}
{
add.f16x2 r1350, r1210, r1257;
}
{
sub.f16x2 r1353, r1207, r1305;
}
{
sub.f16x2 r1356, r1210, r1257;
}
{
add.f16x2 r1359, r1219, r1313;
}
{
add.f16x2 r1362, r1222, r1319;
}
{
sub.f16x2 r1365, r1219, r1313;
}
{
sub.f16x2 r1368, r1222, r1319;
}
{
add.f16x2 r1371, r1372, r1373;
}
{
add.f16x2 r1374, r1375, r1376;
}
{
sub.f16x2 r1377, r1372, r1373;
}
{
sub.f16x2 r1380, r1375, r1376;
}
{
add.f16x2 r1383, r1384, r1385;
}
{
add.f16x2 r1386, r1387, r1388;
}
{
sub.f16x2 r1389, r1384, r1385;
}
{
sub.f16x2 r1392, r1387, r1388;
}
{
neg.f16x2 r1395, r1392;
}
{
add.f16x2 r1397, r1371, r1383;
}
{
add.f16x2 r1400, r1374, r1386;
}
{
sub.f16x2 r1403, r1371, r1383;
}
{
sub.f16x2 r1406, r1374, r1386;
}
{
add.f16x2 r1409, r1377, r1395;
}
{
add.f16x2 r1412, r1380, r1389;
}
{
sub.f16x2 r1415, r1377, r1395;
}
{
sub.f16x2 r1418, r1380, r1389;
}
{
add.f16x2 r1421, r1422, r1423;
}
{
add.f16x2 r1424, r1425, r1426;
}
{
sub.f16x2 r1427, r1422, r1423;
}
{
sub.f16x2 r1430, r1425, r1426;
}
{
add.f16x2 r1433, r1434, r1435;
}
{
add.f16x2 r1436, r1437, r1438;
}
{
sub.f16x2 r1439, r1434, r1435;
}
{
sub.f16x2 r1442, r1437, r1438;
}
{
neg.f16x2 r1445, r1442;
}
{
add.f16x2 r1447, r1421, r1433;
}
{
add.f16x2 r1450, r1424, r1436;
}
{
sub.f16x2 r1453, r1421, r1433;
}
{
sub.f16x2 r1456, r1424, r1436;
}
{
add.f16x2 r1459, r1427, r1445;
}
{
add.f16x2 r1462, r1430, r1439;
}
{
sub.f16x2 r1465, r1427, r1445;
}
{
sub.f16x2 r1468, r1430, r1439;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r1471, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r1472, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f378;
cvt.rn.f16.f32 high, f378;
mov.b32 r1475, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r1476, {low, high};
}
{
mul.f16x2 r1485, r1459, r1471;
}
{
mul.f16x2 r1488, r1462, r1472;
}
{
sub.f16x2 r1491, r1485, r1488;
}
{
mul.f16x2 r1494, r1459, r1472;
}
{
fma.rn.f16x2 r1497, r1462, r1471, r1494;
}
{
neg.f16x2 r1501, r1456;
}
{
mul.f16x2 r1503, r1465, r1475;
}
{
mul.f16x2 r1506, r1468, r1476;
}
{
sub.f16x2 r1509, r1503, r1506;
}
{
mul.f16x2 r1512, r1465, r1476;
}
{
fma.rn.f16x2 r1515, r1468, r1475, r1512;
}
{
add.f16x2 r1519, r1397, r1447;
}
{
add.f16x2 r1522, r1400, r1450;
}
{
sub.f16x2 r1525, r1397, r1447;
}
{
sub.f16x2 r1528, r1400, r1450;
}
{
add.f16x2 r1531, r1409, r1491;
}
{
add.f16x2 r1534, r1412, r1497;
}
{
sub.f16x2 r1537, r1409, r1491;
}
{
sub.f16x2 r1540, r1412, r1497;
}
{
add.f16x2 r1543, r1403, r1501;
}
{
add.f16x2 r1546, r1406, r1453;
}
{
sub.f16x2 r1549, r1403, r1501;
}
{
sub.f16x2 r1552, r1406, r1453;
}
{
add.f16x2 r1555, r1415, r1509;
}
{
add.f16x2 r1558, r1418, r1515;
}
{
sub.f16x2 r1561, r1415, r1509;
}
{
sub.f16x2 r1564, r1418, r1515;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r1567, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r1568, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r1569, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r1570, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r1571, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r1572, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f374;
cvt.rn.f16.f32 high, f374;
mov.b32 r1575, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r1576, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f378;
cvt.rn.f16.f32 high, f378;
mov.b32 r1577, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r1578, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f382;
cvt.rn.f16.f32 high, f382;
mov.b32 r1579, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r1580, {low, high};
}
{
mul.f16x2 r1597, r1531, r1567;
}
{
mul.f16x2 r1600, r1534, r1568;
}
{
sub.f16x2 r1603, r1597, r1600;
}
{
mul.f16x2 r1606, r1531, r1568;
}
{
fma.rn.f16x2 r1609, r1534, r1567, r1606;
}
{
mul.f16x2 r1613, r1543, r1569;
}
{
mul.f16x2 r1616, r1546, r1570;
}
{
sub.f16x2 r1619, r1613, r1616;
}
{
mul.f16x2 r1622, r1543, r1570;
}
{
fma.rn.f16x2 r1625, r1546, r1569, r1622;
}
{
mul.f16x2 r1629, r1555, r1571;
}
{
mul.f16x2 r1632, r1558, r1572;
}
{
sub.f16x2 r1635, r1629, r1632;
}
{
mul.f16x2 r1638, r1555, r1572;
}
{
fma.rn.f16x2 r1641, r1558, r1571, r1638;
}
{
neg.f16x2 r1645, r1528;
}
{
mul.f16x2 r1647, r1537, r1575;
}
{
mul.f16x2 r1650, r1540, r1576;
}
{
sub.f16x2 r1653, r1647, r1650;
}
{
mul.f16x2 r1656, r1537, r1576;
}
{
fma.rn.f16x2 r1659, r1540, r1575, r1656;
}
{
mul.f16x2 r1663, r1549, r1577;
}
{
mul.f16x2 r1666, r1552, r1578;
}
{
sub.f16x2 r1669, r1663, r1666;
}
{
mul.f16x2 r1672, r1549, r1578;
}
{
fma.rn.f16x2 r1675, r1552, r1577, r1672;
}
{
mul.f16x2 r1679, r1561, r1579;
}
{
mul.f16x2 r1682, r1564, r1580;
}
{
sub.f16x2 r1685, r1679, r1682;
}
{
mul.f16x2 r1688, r1561, r1580;
}
{
fma.rn.f16x2 r1691, r1564, r1579, r1688;
}
{
add.f16x2 r1695, r1323, r1519;
}
{
add.f16x2 r1698, r1326, r1522;
}
{
sub.f16x2 r1701, r1323, r1519;
}
{
sub.f16x2 r1704, r1326, r1522;
}
{
add.f16x2 r1707, r1335, r1603;
}
{
add.f16x2 r1710, r1338, r1609;
}
{
sub.f16x2 r1713, r1335, r1603;
}
{
sub.f16x2 r1716, r1338, r1609;
}
{
add.f16x2 r1719, r1347, r1619;
}
{
add.f16x2 r1722, r1350, r1625;
}
{
sub.f16x2 r1725, r1347, r1619;
}
{
sub.f16x2 r1728, r1350, r1625;
}
{
add.f16x2 r1731, r1359, r1635;
}
{
add.f16x2 r1734, r1362, r1641;
}
{
sub.f16x2 r1737, r1359, r1635;
}
{
sub.f16x2 r1740, r1362, r1641;
}
{
add.f16x2 r1743, r1329, r1645;
}
{
add.f16x2 r1746, r1332, r1525;
}
{
sub.f16x2 r1749, r1329, r1645;
}
{
sub.f16x2 r1752, r1332, r1525;
}
{
add.f16x2 r1755, r1341, r1653;
}
{
add.f16x2 r1758, r1344, r1659;
}
{
sub.f16x2 r1761, r1341, r1653;
}
{
sub.f16x2 r1764, r1344, r1659;
}
{
add.f16x2 r1767, r1353, r1669;
}
{
add.f16x2 r1770, r1356, r1675;
}
{
sub.f16x2 r1773, r1353, r1669;
}
{
sub.f16x2 r1776, r1356, r1675;
}
{
add.f16x2 r1779, r1365, r1685;
}
{
add.f16x2 r1782, r1368, r1691;
}
{
sub.f16x2 r1785, r1365, r1685;
}
{
sub.f16x2 r1788, r1368, r1691;
}
and.b32 r3735, r3727, 1008;
bfe.u32 r3736, r3727, 4, 6;
shl.b32 r3737, r3727, 2;
and.b32 r3738, r3737, 60;
add.s32 r3739, r3731, r3738;
cvt.rn.f32.u32 f454, r3736;
mul.f32 f455, f454, 0f3BC90FDB;
cos.approx.f32 f267, f455;
sin.approx.f32 f456, f455;
neg.f32 f268, f456;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f267;
cvt.rn.f16.f32 high, f268;
mov.b32 r1791, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1794, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1796, {high, high};
}
{
mul.f16x2 r1798, r1710, r1796;
}
{
fma.rn.f16x2 r1801, r1707, r1794, r1798;
}
{
mul.f16x2 r1805, r1707, r1796;
}
{
neg.f16x2 r1808, r1805;
}
{
fma.rn.f16x2 r1810, r1710, r1794, r1808;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1814, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1816, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r1818, {low, high};
}
{
mul.f16x2 r1819, r1816, r1818;
}
{
mul.f16x2 r1822, r1791, r1814;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1825, {high, low};
}
{
fma.rn.f16x2 r1827, r1819, r1825, r1822;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1827;
mov.b32 r1831, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1827;
mov.b32 r1833, {high, high};
}
{
mul.f16x2 r1835, r1722, r1833;
}
{
fma.rn.f16x2 r1838, r1719, r1831, r1835;
}
{
mul.f16x2 r1842, r1719, r1833;
}
{
neg.f16x2 r1845, r1842;
}
{
fma.rn.f16x2 r1847, r1722, r1831, r1845;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1851, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1853, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r1855, {low, high};
}
{
mul.f16x2 r1856, r1853, r1855;
}
{
mul.f16x2 r1859, r1827, r1851;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1827;
mov.b32 r1862, {high, low};
}
{
fma.rn.f16x2 r1864, r1856, r1862, r1859;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1864;
mov.b32 r1868, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1864;
mov.b32 r1870, {high, high};
}
{
mul.f16x2 r1872, r1734, r1870;
}
{
fma.rn.f16x2 r1875, r1731, r1868, r1872;
}
{
mul.f16x2 r1879, r1731, r1870;
}
{
neg.f16x2 r1882, r1879;
}
{
fma.rn.f16x2 r1884, r1734, r1868, r1882;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1888, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1890, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r1892, {low, high};
}
{
mul.f16x2 r1893, r1890, r1892;
}
{
mul.f16x2 r1896, r1864, r1888;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1864;
mov.b32 r1899, {high, low};
}
{
fma.rn.f16x2 r1901, r1893, r1899, r1896;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1905, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1907, {high, high};
}
{
mul.f16x2 r1909, r1746, r1907;
}
{
fma.rn.f16x2 r1912, r1743, r1905, r1909;
}
{
mul.f16x2 r1916, r1743, r1907;
}
{
neg.f16x2 r1919, r1916;
}
{
fma.rn.f16x2 r1921, r1746, r1905, r1919;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1925, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1927, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r1929, {low, high};
}
{
mul.f16x2 r1930, r1927, r1929;
}
{
mul.f16x2 r1933, r1901, r1925;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1936, {high, low};
}
{
fma.rn.f16x2 r1938, r1930, r1936, r1933;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1938;
mov.b32 r1942, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1938;
mov.b32 r1944, {high, high};
}
{
mul.f16x2 r1946, r1758, r1944;
}
{
fma.rn.f16x2 r1949, r1755, r1942, r1946;
}
{
mul.f16x2 r1953, r1755, r1944;
}
{
neg.f16x2 r1956, r1953;
}
{
fma.rn.f16x2 r1958, r1758, r1942, r1956;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1962, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1964, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r1966, {low, high};
}
{
mul.f16x2 r1967, r1964, r1966;
}
{
mul.f16x2 r1970, r1938, r1962;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1938;
mov.b32 r1973, {high, low};
}
{
fma.rn.f16x2 r1975, r1967, r1973, r1970;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1975;
mov.b32 r1979, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1975;
mov.b32 r1981, {high, high};
}
{
mul.f16x2 r1983, r1770, r1981;
}
{
fma.rn.f16x2 r1986, r1767, r1979, r1983;
}
{
mul.f16x2 r1990, r1767, r1981;
}
{
neg.f16x2 r1993, r1990;
}
{
fma.rn.f16x2 r1995, r1770, r1979, r1993;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1999, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2001, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r2003, {low, high};
}
{
mul.f16x2 r2004, r2001, r2003;
}
{
mul.f16x2 r2007, r1975, r1999;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1975;
mov.b32 r2010, {high, low};
}
{
fma.rn.f16x2 r2012, r2004, r2010, r2007;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2012;
mov.b32 r2016, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2012;
mov.b32 r2018, {high, high};
}
{
mul.f16x2 r2020, r1782, r2018;
}
{
fma.rn.f16x2 r2023, r1779, r2016, r2020;
}
{
mul.f16x2 r2027, r1779, r2018;
}
{
neg.f16x2 r2030, r2027;
}
{
fma.rn.f16x2 r2032, r1782, r2016, r2030;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2036, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2038, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r2040, {low, high};
}
{
mul.f16x2 r2041, r2038, r2040;
}
{
mul.f16x2 r2044, r2012, r2036;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2012;
mov.b32 r2047, {high, low};
}
{
fma.rn.f16x2 r2049, r2041, r2047, r2044;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2049;
mov.b32 r2053, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2049;
mov.b32 r2055, {high, high};
}
{
mul.f16x2 r2057, r1704, r2055;
}
{
fma.rn.f16x2 r2060, r1701, r2053, r2057;
}
{
mul.f16x2 r2064, r1701, r2055;
}
{
neg.f16x2 r2067, r2064;
}
{
fma.rn.f16x2 r2069, r1704, r2053, r2067;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2073, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2075, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r2077, {low, high};
}
{
mul.f16x2 r2078, r2075, r2077;
}
{
mul.f16x2 r2081, r2049, r2073;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2049;
mov.b32 r2084, {high, low};
}
{
fma.rn.f16x2 r2086, r2078, r2084, r2081;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2086;
mov.b32 r2090, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2086;
mov.b32 r2092, {high, high};
}
{
mul.f16x2 r2094, r1716, r2092;
}
{
fma.rn.f16x2 r2097, r1713, r2090, r2094;
}
{
mul.f16x2 r2101, r1713, r2092;
}
{
neg.f16x2 r2104, r2101;
}
{
fma.rn.f16x2 r2106, r1716, r2090, r2104;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2110, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2112, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r2114, {low, high};
}
{
mul.f16x2 r2115, r2112, r2114;
}
{
mul.f16x2 r2118, r2086, r2110;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2086;
mov.b32 r2121, {high, low};
}
{
fma.rn.f16x2 r2123, r2115, r2121, r2118;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2123;
mov.b32 r2127, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2123;
mov.b32 r2129, {high, high};
}
{
mul.f16x2 r2131, r1728, r2129;
}
{
fma.rn.f16x2 r2134, r1725, r2127, r2131;
}
{
mul.f16x2 r2138, r1725, r2129;
}
{
neg.f16x2 r2141, r2138;
}
{
fma.rn.f16x2 r2143, r1728, r2127, r2141;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2147, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2149, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r2151, {low, high};
}
{
mul.f16x2 r2152, r2149, r2151;
}
{
mul.f16x2 r2155, r2123, r2147;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2123;
mov.b32 r2158, {high, low};
}
{
fma.rn.f16x2 r2160, r2152, r2158, r2155;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2160;
mov.b32 r2164, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2160;
mov.b32 r2166, {high, high};
}
{
mul.f16x2 r2168, r1740, r2166;
}
{
fma.rn.f16x2 r2171, r1737, r2164, r2168;
}
{
mul.f16x2 r2175, r1737, r2166;
}
{
neg.f16x2 r2178, r2175;
}
{
fma.rn.f16x2 r2180, r1740, r2164, r2178;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2184, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2186, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r2188, {low, high};
}
{
mul.f16x2 r2189, r2186, r2188;
}
{
mul.f16x2 r2192, r2160, r2184;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2160;
mov.b32 r2195, {high, low};
}
{
fma.rn.f16x2 r2197, r2189, r2195, r2192;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2197;
mov.b32 r2201, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2197;
mov.b32 r2203, {high, high};
}
{
mul.f16x2 r2205, r1752, r2203;
}
{
fma.rn.f16x2 r2208, r1749, r2201, r2205;
}
{
mul.f16x2 r2212, r1749, r2203;
}
{
neg.f16x2 r2215, r2212;
}
{
fma.rn.f16x2 r2217, r1752, r2201, r2215;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2221, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2223, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r2225, {low, high};
}
{
mul.f16x2 r2226, r2223, r2225;
}
{
mul.f16x2 r2229, r2197, r2221;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2197;
mov.b32 r2232, {high, low};
}
{
fma.rn.f16x2 r2234, r2226, r2232, r2229;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2234;
mov.b32 r2238, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2234;
mov.b32 r2240, {high, high};
}
{
mul.f16x2 r2242, r1764, r2240;
}
{
fma.rn.f16x2 r2245, r1761, r2238, r2242;
}
{
mul.f16x2 r2249, r1761, r2240;
}
{
neg.f16x2 r2252, r2249;
}
{
fma.rn.f16x2 r2254, r1764, r2238, r2252;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2258, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2260, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r2262, {low, high};
}
{
mul.f16x2 r2263, r2260, r2262;
}
{
mul.f16x2 r2266, r2234, r2258;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2234;
mov.b32 r2269, {high, low};
}
{
fma.rn.f16x2 r2271, r2263, r2269, r2266;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2271;
mov.b32 r2275, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2271;
mov.b32 r2277, {high, high};
}
{
mul.f16x2 r2279, r1776, r2277;
}
{
fma.rn.f16x2 r2282, r1773, r2275, r2279;
}
{
mul.f16x2 r2286, r1773, r2277;
}
{
neg.f16x2 r2289, r2286;
}
{
fma.rn.f16x2 r2291, r1776, r2275, r2289;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2295, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2297, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r2299, {low, high};
}
{
mul.f16x2 r2300, r2297, r2299;
}
{
mul.f16x2 r2303, r2271, r2295;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2271;
mov.b32 r2306, {high, low};
}
{
fma.rn.f16x2 r2308, r2300, r2306, r2303;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2308;
mov.b32 r2312, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2308;
mov.b32 r2314, {high, high};
}
{
mul.f16x2 r2316, r1788, r2314;
}
{
fma.rn.f16x2 r2319, r1785, r2312, r2316;
}
{
mul.f16x2 r2323, r1785, r2314;
}
{
neg.f16x2 r2326, r2323;
}
{
fma.rn.f16x2 r2328, r1788, r2312, r2326;
}
barrier.sync 0;
and.b32 r3740, r3729, 64512;
add.s32 r3741, r3739, r3740;
st.shared.u32 [r3741], r1695;
st.shared.u32 [r3741+64], r1801;
st.shared.u32 [r3741+128], r1838;
st.shared.u32 [r3741+192], r1875;
st.shared.u32 [r3741+256], r1912;
st.shared.u32 [r3741+320], r1949;
st.shared.u32 [r3741+384], r1986;
st.shared.u32 [r3741+448], r2023;
st.shared.u32 [r3741+512], r2060;
st.shared.u32 [r3741+576], r2097;
st.shared.u32 [r3741+640], r2134;
st.shared.u32 [r3741+704], r2171;
st.shared.u32 [r3741+768], r2208;
st.shared.u32 [r3741+832], r2245;
st.shared.u32 [r3741+896], r2282;
st.shared.u32 [r3741+960], r2319;
barrier.sync 0;
mad.lo.s32 r3742, r3735, -60, r3741;
ld.shared.u32 r2350, [r3742];
ld.shared.u32 r2546, [r3742+4096];
ld.shared.u32 r2400, [r3742+8192];
ld.shared.u32 r2596, [r3742+12288];
ld.shared.u32 r2362, [r3742+16384];
ld.shared.u32 r2558, [r3742+20480];
ld.shared.u32 r2412, [r3742+24576];
ld.shared.u32 r2608, [r3742+28672];
ld.shared.u32 r2351, [r3742+32768];
ld.shared.u32 r2547, [r3742+36864];
ld.shared.u32 r2401, [r3742+40960];
ld.shared.u32 r2597, [r3742+45056];
ld.shared.u32 r2363, [r3742+49152];
ld.shared.u32 r2559, [r3742+53248];
ld.shared.u32 r2413, [r3742+57344];
ld.shared.u32 r2609, [r3742+61440];
barrier.sync 0;
st.shared.u32 [r3741], r1698;
st.shared.u32 [r3741+64], r1810;
st.shared.u32 [r3741+128], r1847;
st.shared.u32 [r3741+192], r1884;
st.shared.u32 [r3741+256], r1921;
st.shared.u32 [r3741+320], r1958;
st.shared.u32 [r3741+384], r1995;
st.shared.u32 [r3741+448], r2032;
st.shared.u32 [r3741+512], r2069;
st.shared.u32 [r3741+576], r2106;
st.shared.u32 [r3741+640], r2143;
st.shared.u32 [r3741+704], r2180;
st.shared.u32 [r3741+768], r2217;
st.shared.u32 [r3741+832], r2254;
st.shared.u32 [r3741+896], r2291;
st.shared.u32 [r3741+960], r2328;
barrier.sync 0;
ld.shared.u32 r2353, [r3742];
ld.shared.u32 r2549, [r3742+4096];
ld.shared.u32 r2403, [r3742+8192];
ld.shared.u32 r2599, [r3742+12288];
ld.shared.u32 r2365, [r3742+16384];
ld.shared.u32 r2561, [r3742+20480];
ld.shared.u32 r2415, [r3742+24576];
ld.shared.u32 r2611, [r3742+28672];
ld.shared.u32 r2354, [r3742+32768];
ld.shared.u32 r2550, [r3742+36864];
ld.shared.u32 r2404, [r3742+40960];
ld.shared.u32 r2600, [r3742+45056];
ld.shared.u32 r2366, [r3742+49152];
ld.shared.u32 r2562, [r3742+53248];
ld.shared.u32 r2416, [r3742+57344];
ld.shared.u32 r2612, [r3742+61440];
{
add.f16x2 r2349, r2350, r2351;
}
{
add.f16x2 r2352, r2353, r2354;
}
{
sub.f16x2 r2355, r2350, r2351;
}
{
sub.f16x2 r2358, r2353, r2354;
}
{
add.f16x2 r2361, r2362, r2363;
}
{
add.f16x2 r2364, r2365, r2366;
}
{
sub.f16x2 r2367, r2362, r2363;
}
{
sub.f16x2 r2370, r2365, r2366;
}
{
neg.f16x2 r2373, r2370;
}
{
add.f16x2 r2375, r2349, r2361;
}
{
add.f16x2 r2378, r2352, r2364;
}
{
sub.f16x2 r2381, r2349, r2361;
}
{
sub.f16x2 r2384, r2352, r2364;
}
{
add.f16x2 r2387, r2355, r2373;
}
{
add.f16x2 r2390, r2358, r2367;
}
{
sub.f16x2 r2393, r2355, r2373;
}
{
sub.f16x2 r2396, r2358, r2367;
}
{
add.f16x2 r2399, r2400, r2401;
}
{
add.f16x2 r2402, r2403, r2404;
}
{
sub.f16x2 r2405, r2400, r2401;
}
{
sub.f16x2 r2408, r2403, r2404;
}
{
add.f16x2 r2411, r2412, r2413;
}
{
add.f16x2 r2414, r2415, r2416;
}
{
sub.f16x2 r2417, r2412, r2413;
}
{
sub.f16x2 r2420, r2415, r2416;
}
{
neg.f16x2 r2423, r2420;
}
{
add.f16x2 r2425, r2399, r2411;
}
{
add.f16x2 r2428, r2402, r2414;
}
{
sub.f16x2 r2431, r2399, r2411;
}
{
sub.f16x2 r2434, r2402, r2414;
}
{
add.f16x2 r2437, r2405, r2423;
}
{
add.f16x2 r2440, r2408, r2417;
}
{
sub.f16x2 r2443, r2405, r2423;
}
{
sub.f16x2 r2446, r2408, r2417;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r2449, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r2450, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f378;
cvt.rn.f16.f32 high, f378;
mov.b32 r2453, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r2454, {low, high};
}
{
mul.f16x2 r2463, r2437, r2449;
}
{
mul.f16x2 r2466, r2440, r2450;
}
{
sub.f16x2 r2469, r2463, r2466;
}
{
mul.f16x2 r2472, r2437, r2450;
}
{
fma.rn.f16x2 r2475, r2440, r2449, r2472;
}
{
neg.f16x2 r2479, r2434;
}
{
mul.f16x2 r2481, r2443, r2453;
}
{
mul.f16x2 r2484, r2446, r2454;
}
{
sub.f16x2 r2487, r2481, r2484;
}
{
mul.f16x2 r2490, r2443, r2454;
}
{
fma.rn.f16x2 r2493, r2446, r2453, r2490;
}
{
add.f16x2 r2497, r2375, r2425;
}
{
add.f16x2 r2500, r2378, r2428;
}
{
sub.f16x2 r2503, r2375, r2425;
}
{
sub.f16x2 r2506, r2378, r2428;
}
{
add.f16x2 r2509, r2387, r2469;
}
{
add.f16x2 r2512, r2390, r2475;
}
{
sub.f16x2 r2515, r2387, r2469;
}
{
sub.f16x2 r2518, r2390, r2475;
}
{
add.f16x2 r2521, r2381, r2479;
}
{
add.f16x2 r2524, r2384, r2431;
}
{
sub.f16x2 r2527, r2381, r2479;
}
{
sub.f16x2 r2530, r2384, r2431;
}
{
add.f16x2 r2533, r2393, r2487;
}
{
add.f16x2 r2536, r2396, r2493;
}
{
sub.f16x2 r2539, r2393, r2487;
}
{
sub.f16x2 r2542, r2396, r2493;
}
{
add.f16x2 r2545, r2546, r2547;
}
{
add.f16x2 r2548, r2549, r2550;
}
{
sub.f16x2 r2551, r2546, r2547;
}
{
sub.f16x2 r2554, r2549, r2550;
}
{
add.f16x2 r2557, r2558, r2559;
}
{
add.f16x2 r2560, r2561, r2562;
}
{
sub.f16x2 r2563, r2558, r2559;
}
{
sub.f16x2 r2566, r2561, r2562;
}
{
neg.f16x2 r2569, r2566;
}
{
add.f16x2 r2571, r2545, r2557;
}
{
add.f16x2 r2574, r2548, r2560;
}
{
sub.f16x2 r2577, r2545, r2557;
}
{
sub.f16x2 r2580, r2548, r2560;
}
{
add.f16x2 r2583, r2551, r2569;
}
{
add.f16x2 r2586, r2554, r2563;
}
{
sub.f16x2 r2589, r2551, r2569;
}
{
sub.f16x2 r2592, r2554, r2563;
}
{
add.f16x2 r2595, r2596, r2597;
}
{
add.f16x2 r2598, r2599, r2600;
}
{
sub.f16x2 r2601, r2596, r2597;
}
{
sub.f16x2 r2604, r2599, r2600;
}
{
add.f16x2 r2607, r2608, r2609;
}
{
add.f16x2 r2610, r2611, r2612;
}
{
sub.f16x2 r2613, r2608, r2609;
}
{
sub.f16x2 r2616, r2611, r2612;
}
{
neg.f16x2 r2619, r2616;
}
{
add.f16x2 r2621, r2595, r2607;
}
{
add.f16x2 r2624, r2598, r2610;
}
{
sub.f16x2 r2627, r2595, r2607;
}
{
sub.f16x2 r2630, r2598, r2610;
}
{
add.f16x2 r2633, r2601, r2619;
}
{
add.f16x2 r2636, r2604, r2613;
}
{
sub.f16x2 r2639, r2601, r2619;
}
{
sub.f16x2 r2642, r2604, r2613;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r2645, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r2646, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f378;
cvt.rn.f16.f32 high, f378;
mov.b32 r2649, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r2650, {low, high};
}
{
mul.f16x2 r2659, r2633, r2645;
}
{
mul.f16x2 r2662, r2636, r2646;
}
{
sub.f16x2 r2665, r2659, r2662;
}
{
mul.f16x2 r2668, r2633, r2646;
}
{
fma.rn.f16x2 r2671, r2636, r2645, r2668;
}
{
neg.f16x2 r2675, r2630;
}
{
mul.f16x2 r2677, r2639, r2649;
}
{
mul.f16x2 r2680, r2642, r2650;
}
{
sub.f16x2 r2683, r2677, r2680;
}
{
mul.f16x2 r2686, r2639, r2650;
}
{
fma.rn.f16x2 r2689, r2642, r2649, r2686;
}
{
add.f16x2 r2693, r2571, r2621;
}
{
add.f16x2 r2696, r2574, r2624;
}
{
sub.f16x2 r2699, r2571, r2621;
}
{
sub.f16x2 r2702, r2574, r2624;
}
{
add.f16x2 r2705, r2583, r2665;
}
{
add.f16x2 r2708, r2586, r2671;
}
{
sub.f16x2 r2711, r2583, r2665;
}
{
sub.f16x2 r2714, r2586, r2671;
}
{
add.f16x2 r2717, r2577, r2675;
}
{
add.f16x2 r2720, r2580, r2627;
}
{
sub.f16x2 r2723, r2577, r2675;
}
{
sub.f16x2 r2726, r2580, r2627;
}
{
add.f16x2 r2729, r2589, r2683;
}
{
add.f16x2 r2732, r2592, r2689;
}
{
sub.f16x2 r2735, r2589, r2683;
}
{
sub.f16x2 r2738, r2592, r2689;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r2741, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r2742, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r2743, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r2744, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r2745, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r2746, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f374;
cvt.rn.f16.f32 high, f374;
mov.b32 r2749, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r2750, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f378;
cvt.rn.f16.f32 high, f378;
mov.b32 r2751, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r2752, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f382;
cvt.rn.f16.f32 high, f382;
mov.b32 r2753, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r2754, {low, high};
}
{
mul.f16x2 r2771, r2705, r2741;
}
{
mul.f16x2 r2774, r2708, r2742;
}
{
sub.f16x2 r2777, r2771, r2774;
}
{
mul.f16x2 r2780, r2705, r2742;
}
{
fma.rn.f16x2 r2783, r2708, r2741, r2780;
}
{
mul.f16x2 r2787, r2717, r2743;
}
{
mul.f16x2 r2790, r2720, r2744;
}
{
sub.f16x2 r2793, r2787, r2790;
}
{
mul.f16x2 r2796, r2717, r2744;
}
{
fma.rn.f16x2 r2799, r2720, r2743, r2796;
}
{
mul.f16x2 r2803, r2729, r2745;
}
{
mul.f16x2 r2806, r2732, r2746;
}
{
sub.f16x2 r2809, r2803, r2806;
}
{
mul.f16x2 r2812, r2729, r2746;
}
{
fma.rn.f16x2 r2815, r2732, r2745, r2812;
}
{
neg.f16x2 r2819, r2702;
}
{
mul.f16x2 r2821, r2711, r2749;
}
{
mul.f16x2 r2824, r2714, r2750;
}
{
sub.f16x2 r2827, r2821, r2824;
}
{
mul.f16x2 r2830, r2711, r2750;
}
{
fma.rn.f16x2 r2833, r2714, r2749, r2830;
}
{
mul.f16x2 r2837, r2723, r2751;
}
{
mul.f16x2 r2840, r2726, r2752;
}
{
sub.f16x2 r2843, r2837, r2840;
}
{
mul.f16x2 r2846, r2723, r2752;
}
{
fma.rn.f16x2 r2849, r2726, r2751, r2846;
}
{
mul.f16x2 r2853, r2735, r2753;
}
{
mul.f16x2 r2856, r2738, r2754;
}
{
sub.f16x2 r2859, r2853, r2856;
}
{
mul.f16x2 r2862, r2735, r2754;
}
{
fma.rn.f16x2 r2865, r2738, r2753, r2862;
}
{
add.f16x2 r2869, r2497, r2693;
}
{
add.f16x2 r2872, r2500, r2696;
}
{
sub.f16x2 r2875, r2497, r2693;
}
{
sub.f16x2 r2878, r2500, r2696;
}
{
add.f16x2 r2881, r2509, r2777;
}
{
add.f16x2 r2884, r2512, r2783;
}
{
sub.f16x2 r2887, r2509, r2777;
}
{
sub.f16x2 r2890, r2512, r2783;
}
{
add.f16x2 r2893, r2521, r2793;
}
{
add.f16x2 r2896, r2524, r2799;
}
{
sub.f16x2 r2899, r2521, r2793;
}
{
sub.f16x2 r2902, r2524, r2799;
}
{
add.f16x2 r2905, r2533, r2809;
}
{
add.f16x2 r2908, r2536, r2815;
}
{
sub.f16x2 r2911, r2533, r2809;
}
{
sub.f16x2 r2914, r2536, r2815;
}
{
add.f16x2 r2917, r2503, r2819;
}
{
add.f16x2 r2920, r2506, r2699;
}
{
sub.f16x2 r2923, r2503, r2819;
}
{
sub.f16x2 r2926, r2506, r2699;
}
{
add.f16x2 r2929, r2515, r2827;
}
{
add.f16x2 r2932, r2518, r2833;
}
{
sub.f16x2 r2935, r2515, r2827;
}
{
sub.f16x2 r2938, r2518, r2833;
}
{
add.f16x2 r2941, r2527, r2843;
}
{
add.f16x2 r2944, r2530, r2849;
}
{
sub.f16x2 r2947, r2527, r2843;
}
{
sub.f16x2 r2950, r2530, r2849;
}
{
add.f16x2 r2953, r2539, r2859;
}
{
add.f16x2 r2956, r2542, r2865;
}
{
sub.f16x2 r2959, r2539, r2859;
}
{
sub.f16x2 r2962, r2542, r2865;
}
and.b32 r3743, r3727, 768;
bfe.u32 r3744, r3727, 8, 2;
and.b32 r3745, r3737, 1020;
add.s32 r3746, r3731, r3745;
cvt.rn.f32.u32 f457, r3744;
mul.f32 f458, f457, 0f3DC90FDB;
cos.approx.f32 f417, f458;
sin.approx.f32 f459, f458;
neg.f32 f418, f459;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f417;
cvt.rn.f16.f32 high, f418;
mov.b32 r2965, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r2968, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r2970, {high, high};
}
{
mul.f16x2 r2972, r2884, r2970;
}
{
fma.rn.f16x2 r2975, r2881, r2968, r2972;
}
{
mul.f16x2 r2979, r2881, r2970;
}
{
neg.f16x2 r2982, r2979;
}
{
fma.rn.f16x2 r2984, r2884, r2968, r2982;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r2988, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r2990, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r2992, {low, high};
}
{
mul.f16x2 r2993, r2990, r2992;
}
{
mul.f16x2 r2996, r2965, r2988;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r2999, {high, low};
}
{
fma.rn.f16x2 r3001, r2993, r2999, r2996;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3001;
mov.b32 r3005, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3001;
mov.b32 r3007, {high, high};
}
{
mul.f16x2 r3009, r2896, r3007;
}
{
fma.rn.f16x2 r3012, r2893, r3005, r3009;
}
{
mul.f16x2 r3016, r2893, r3007;
}
{
neg.f16x2 r3019, r3016;
}
{
fma.rn.f16x2 r3021, r2896, r3005, r3019;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3025, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3027, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3029, {low, high};
}
{
mul.f16x2 r3030, r3027, r3029;
}
{
mul.f16x2 r3033, r3001, r3025;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3001;
mov.b32 r3036, {high, low};
}
{
fma.rn.f16x2 r3038, r3030, r3036, r3033;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3038;
mov.b32 r3042, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3038;
mov.b32 r3044, {high, high};
}
{
mul.f16x2 r3046, r2908, r3044;
}
{
fma.rn.f16x2 r3049, r2905, r3042, r3046;
}
{
mul.f16x2 r3053, r2905, r3044;
}
{
neg.f16x2 r3056, r3053;
}
{
fma.rn.f16x2 r3058, r2908, r3042, r3056;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3062, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3064, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3066, {low, high};
}
{
mul.f16x2 r3067, r3064, r3066;
}
{
mul.f16x2 r3070, r3038, r3062;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3038;
mov.b32 r3073, {high, low};
}
{
fma.rn.f16x2 r3075, r3067, r3073, r3070;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3075;
mov.b32 r3079, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3075;
mov.b32 r3081, {high, high};
}
{
mul.f16x2 r3083, r2920, r3081;
}
{
fma.rn.f16x2 r3086, r2917, r3079, r3083;
}
{
mul.f16x2 r3090, r2917, r3081;
}
{
neg.f16x2 r3093, r3090;
}
{
fma.rn.f16x2 r3095, r2920, r3079, r3093;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3099, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3101, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3103, {low, high};
}
{
mul.f16x2 r3104, r3101, r3103;
}
{
mul.f16x2 r3107, r3075, r3099;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3075;
mov.b32 r3110, {high, low};
}
{
fma.rn.f16x2 r3112, r3104, r3110, r3107;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3112;
mov.b32 r3116, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3112;
mov.b32 r3118, {high, high};
}
{
mul.f16x2 r3120, r2932, r3118;
}
{
fma.rn.f16x2 r3123, r2929, r3116, r3120;
}
{
mul.f16x2 r3127, r2929, r3118;
}
{
neg.f16x2 r3130, r3127;
}
{
fma.rn.f16x2 r3132, r2932, r3116, r3130;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3136, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3138, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3140, {low, high};
}
{
mul.f16x2 r3141, r3138, r3140;
}
{
mul.f16x2 r3144, r3112, r3136;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3112;
mov.b32 r3147, {high, low};
}
{
fma.rn.f16x2 r3149, r3141, r3147, r3144;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3149;
mov.b32 r3153, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3149;
mov.b32 r3155, {high, high};
}
{
mul.f16x2 r3157, r2944, r3155;
}
{
fma.rn.f16x2 r3160, r2941, r3153, r3157;
}
{
mul.f16x2 r3164, r2941, r3155;
}
{
neg.f16x2 r3167, r3164;
}
{
fma.rn.f16x2 r3169, r2944, r3153, r3167;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3173, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3175, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3177, {low, high};
}
{
mul.f16x2 r3178, r3175, r3177;
}
{
mul.f16x2 r3181, r3149, r3173;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3149;
mov.b32 r3184, {high, low};
}
{
fma.rn.f16x2 r3186, r3178, r3184, r3181;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3186;
mov.b32 r3190, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3186;
mov.b32 r3192, {high, high};
}
{
mul.f16x2 r3194, r2956, r3192;
}
{
fma.rn.f16x2 r3197, r2953, r3190, r3194;
}
{
mul.f16x2 r3201, r2953, r3192;
}
{
neg.f16x2 r3204, r3201;
}
{
fma.rn.f16x2 r3206, r2956, r3190, r3204;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3210, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3212, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3214, {low, high};
}
{
mul.f16x2 r3215, r3212, r3214;
}
{
mul.f16x2 r3218, r3186, r3210;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3186;
mov.b32 r3221, {high, low};
}
{
fma.rn.f16x2 r3223, r3215, r3221, r3218;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3223;
mov.b32 r3227, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3223;
mov.b32 r3229, {high, high};
}
{
mul.f16x2 r3231, r2878, r3229;
}
{
fma.rn.f16x2 r3234, r2875, r3227, r3231;
}
{
mul.f16x2 r3238, r2875, r3229;
}
{
neg.f16x2 r3241, r3238;
}
{
fma.rn.f16x2 r3243, r2878, r3227, r3241;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3247, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3249, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3251, {low, high};
}
{
mul.f16x2 r3252, r3249, r3251;
}
{
mul.f16x2 r3255, r3223, r3247;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3223;
mov.b32 r3258, {high, low};
}
{
fma.rn.f16x2 r3260, r3252, r3258, r3255;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3260;
mov.b32 r3264, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3260;
mov.b32 r3266, {high, high};
}
{
mul.f16x2 r3268, r2890, r3266;
}
{
fma.rn.f16x2 r3271, r2887, r3264, r3268;
}
{
mul.f16x2 r3275, r2887, r3266;
}
{
neg.f16x2 r3278, r3275;
}
{
fma.rn.f16x2 r3280, r2890, r3264, r3278;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3284, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3286, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3288, {low, high};
}
{
mul.f16x2 r3289, r3286, r3288;
}
{
mul.f16x2 r3292, r3260, r3284;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3260;
mov.b32 r3295, {high, low};
}
{
fma.rn.f16x2 r3297, r3289, r3295, r3292;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3297;
mov.b32 r3301, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3297;
mov.b32 r3303, {high, high};
}
{
mul.f16x2 r3305, r2902, r3303;
}
{
fma.rn.f16x2 r3308, r2899, r3301, r3305;
}
{
mul.f16x2 r3312, r2899, r3303;
}
{
neg.f16x2 r3315, r3312;
}
{
fma.rn.f16x2 r3317, r2902, r3301, r3315;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3321, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3323, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3325, {low, high};
}
{
mul.f16x2 r3326, r3323, r3325;
}
{
mul.f16x2 r3329, r3297, r3321;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3297;
mov.b32 r3332, {high, low};
}
{
fma.rn.f16x2 r3334, r3326, r3332, r3329;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3334;
mov.b32 r3338, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3334;
mov.b32 r3340, {high, high};
}
{
mul.f16x2 r3342, r2914, r3340;
}
{
fma.rn.f16x2 r3345, r2911, r3338, r3342;
}
{
mul.f16x2 r3349, r2911, r3340;
}
{
neg.f16x2 r3352, r3349;
}
{
fma.rn.f16x2 r3354, r2914, r3338, r3352;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3358, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3360, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3362, {low, high};
}
{
mul.f16x2 r3363, r3360, r3362;
}
{
mul.f16x2 r3366, r3334, r3358;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3334;
mov.b32 r3369, {high, low};
}
{
fma.rn.f16x2 r3371, r3363, r3369, r3366;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3371;
mov.b32 r3375, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3371;
mov.b32 r3377, {high, high};
}
{
mul.f16x2 r3379, r2926, r3377;
}
{
fma.rn.f16x2 r3382, r2923, r3375, r3379;
}
{
mul.f16x2 r3386, r2923, r3377;
}
{
neg.f16x2 r3389, r3386;
}
{
fma.rn.f16x2 r3391, r2926, r3375, r3389;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3395, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3397, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3399, {low, high};
}
{
mul.f16x2 r3400, r3397, r3399;
}
{
mul.f16x2 r3403, r3371, r3395;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3371;
mov.b32 r3406, {high, low};
}
{
fma.rn.f16x2 r3408, r3400, r3406, r3403;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3408;
mov.b32 r3412, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3408;
mov.b32 r3414, {high, high};
}
{
mul.f16x2 r3416, r2938, r3414;
}
{
fma.rn.f16x2 r3419, r2935, r3412, r3416;
}
{
mul.f16x2 r3423, r2935, r3414;
}
{
neg.f16x2 r3426, r3423;
}
{
fma.rn.f16x2 r3428, r2938, r3412, r3426;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3432, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3434, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3436, {low, high};
}
{
mul.f16x2 r3437, r3434, r3436;
}
{
mul.f16x2 r3440, r3408, r3432;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3408;
mov.b32 r3443, {high, low};
}
{
fma.rn.f16x2 r3445, r3437, r3443, r3440;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3445;
mov.b32 r3449, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3445;
mov.b32 r3451, {high, high};
}
{
mul.f16x2 r3453, r2950, r3451;
}
{
fma.rn.f16x2 r3456, r2947, r3449, r3453;
}
{
mul.f16x2 r3460, r2947, r3451;
}
{
neg.f16x2 r3463, r3460;
}
{
fma.rn.f16x2 r3465, r2950, r3449, r3463;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3469, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3471, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3473, {low, high};
}
{
mul.f16x2 r3474, r3471, r3473;
}
{
mul.f16x2 r3477, r3445, r3469;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3445;
mov.b32 r3480, {high, low};
}
{
fma.rn.f16x2 r3482, r3474, r3480, r3477;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3482;
mov.b32 r3486, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3482;
mov.b32 r3488, {high, high};
}
{
mul.f16x2 r3490, r2962, r3488;
}
{
fma.rn.f16x2 r3493, r2959, r3486, r3490;
}
{
mul.f16x2 r3497, r2959, r3488;
}
{
neg.f16x2 r3500, r3497;
}
{
fma.rn.f16x2 r3502, r2962, r3486, r3500;
}
barrier.sync 0;
and.b32 r3747, r3729, 49152;
add.s32 r3748, r3746, r3747;
st.shared.u32 [r3748], r2869;
st.shared.u32 [r3748+1024], r2975;
st.shared.u32 [r3748+2048], r3012;
st.shared.u32 [r3748+3072], r3049;
st.shared.u32 [r3748+4096], r3086;
st.shared.u32 [r3748+5120], r3123;
st.shared.u32 [r3748+6144], r3160;
st.shared.u32 [r3748+7168], r3197;
st.shared.u32 [r3748+8192], r3234;
st.shared.u32 [r3748+9216], r3271;
st.shared.u32 [r3748+10240], r3308;
st.shared.u32 [r3748+11264], r3345;
st.shared.u32 [r3748+12288], r3382;
st.shared.u32 [r3748+13312], r3419;
st.shared.u32 [r3748+14336], r3456;
st.shared.u32 [r3748+15360], r3493;
barrier.sync 0;
mad.lo.s32 r3749, r3743, -60, r3748;
ld.shared.u32 r3524, [r3749];
ld.shared.u32 r3574, [r3749+4096];
ld.shared.u32 r3624, [r3749+8192];
ld.shared.u32 r3674, [r3749+12288];
ld.shared.u32 r3536, [r3749+16384];
ld.shared.u32 r3586, [r3749+20480];
ld.shared.u32 r3636, [r3749+24576];
ld.shared.u32 r3686, [r3749+28672];
ld.shared.u32 r3525, [r3749+32768];
ld.shared.u32 r3575, [r3749+36864];
ld.shared.u32 r3625, [r3749+40960];
ld.shared.u32 r3675, [r3749+45056];
ld.shared.u32 r3537, [r3749+49152];
ld.shared.u32 r3587, [r3749+53248];
ld.shared.u32 r3637, [r3749+57344];
ld.shared.u32 r3687, [r3749+61440];
barrier.sync 0;
st.shared.u32 [r3748], r2872;
st.shared.u32 [r3748+1024], r2984;
st.shared.u32 [r3748+2048], r3021;
st.shared.u32 [r3748+3072], r3058;
st.shared.u32 [r3748+4096], r3095;
st.shared.u32 [r3748+5120], r3132;
st.shared.u32 [r3748+6144], r3169;
st.shared.u32 [r3748+7168], r3206;
st.shared.u32 [r3748+8192], r3243;
st.shared.u32 [r3748+9216], r3280;
st.shared.u32 [r3748+10240], r3317;
st.shared.u32 [r3748+11264], r3354;
st.shared.u32 [r3748+12288], r3391;
st.shared.u32 [r3748+13312], r3428;
st.shared.u32 [r3748+14336], r3465;
st.shared.u32 [r3748+15360], r3502;
barrier.sync 0;
ld.shared.u32 r3527, [r3749];
ld.shared.u32 r3577, [r3749+4096];
ld.shared.u32 r3627, [r3749+8192];
ld.shared.u32 r3677, [r3749+12288];
ld.shared.u32 r3539, [r3749+16384];
ld.shared.u32 r3589, [r3749+20480];
ld.shared.u32 r3639, [r3749+24576];
ld.shared.u32 r3689, [r3749+28672];
ld.shared.u32 r3528, [r3749+32768];
ld.shared.u32 r3578, [r3749+36864];
ld.shared.u32 r3628, [r3749+40960];
ld.shared.u32 r3678, [r3749+45056];
ld.shared.u32 r3540, [r3749+49152];
ld.shared.u32 r3590, [r3749+53248];
ld.shared.u32 r3640, [r3749+57344];
ld.shared.u32 r3690, [r3749+61440];
{
add.f16x2 r3523, r3524, r3525;
}
{
add.f16x2 r3526, r3527, r3528;
}
{
sub.f16x2 r3529, r3524, r3525;
}
{
sub.f16x2 r3532, r3527, r3528;
}
{
add.f16x2 r3535, r3536, r3537;
}
{
add.f16x2 r3538, r3539, r3540;
}
{
sub.f16x2 r3541, r3536, r3537;
}
{
sub.f16x2 r3544, r3539, r3540;
}
{
neg.f16x2 r3547, r3544;
}
{
add.f16x2 %0, r3523, r3535;
}
{
add.f16x2 %1, r3526, r3538;
}
{
sub.f16x2 %16, r3523, r3535;
}
{
sub.f16x2 %17, r3526, r3538;
}
{
add.f16x2 %8, r3529, r3547;
}
{
add.f16x2 %9, r3532, r3541;
}
{
sub.f16x2 %24, r3529, r3547;
}
{
sub.f16x2 %25, r3532, r3541;
}
{
add.f16x2 r3573, r3574, r3575;
}
{
add.f16x2 r3576, r3577, r3578;
}
{
sub.f16x2 r3579, r3574, r3575;
}
{
sub.f16x2 r3582, r3577, r3578;
}
{
add.f16x2 r3585, r3586, r3587;
}
{
add.f16x2 r3588, r3589, r3590;
}
{
sub.f16x2 r3591, r3586, r3587;
}
{
sub.f16x2 r3594, r3589, r3590;
}
{
neg.f16x2 r3597, r3594;
}
{
add.f16x2 %2, r3573, r3585;
}
{
add.f16x2 %3, r3576, r3588;
}
{
sub.f16x2 %18, r3573, r3585;
}
{
sub.f16x2 %19, r3576, r3588;
}
{
add.f16x2 %10, r3579, r3597;
}
{
add.f16x2 %11, r3582, r3591;
}
{
sub.f16x2 %26, r3579, r3597;
}
{
sub.f16x2 %27, r3582, r3591;
}
{
add.f16x2 r3623, r3624, r3625;
}
{
add.f16x2 r3626, r3627, r3628;
}
{
sub.f16x2 r3629, r3624, r3625;
}
{
sub.f16x2 r3632, r3627, r3628;
}
{
add.f16x2 r3635, r3636, r3637;
}
{
add.f16x2 r3638, r3639, r3640;
}
{
sub.f16x2 r3641, r3636, r3637;
}
{
sub.f16x2 r3644, r3639, r3640;
}
{
neg.f16x2 r3647, r3644;
}
{
add.f16x2 %4, r3623, r3635;
}
{
add.f16x2 %5, r3626, r3638;
}
{
sub.f16x2 %20, r3623, r3635;
}
{
sub.f16x2 %21, r3626, r3638;
}
{
add.f16x2 %12, r3629, r3647;
}
{
add.f16x2 %13, r3632, r3641;
}
{
sub.f16x2 %28, r3629, r3647;
}
{
sub.f16x2 %29, r3632, r3641;
}
{
add.f16x2 r3673, r3674, r3675;
}
{
add.f16x2 r3676, r3677, r3678;
}
{
sub.f16x2 r3679, r3674, r3675;
}
{
sub.f16x2 r3682, r3677, r3678;
}
{
add.f16x2 r3685, r3686, r3687;
}
{
add.f16x2 r3688, r3689, r3690;
}
{
sub.f16x2 r3691, r3686, r3687;
}
{
sub.f16x2 r3694, r3689, r3690;
}
{
neg.f16x2 r3697, r3694;
}
{
add.f16x2 %6, r3673, r3685;
}
{
add.f16x2 %7, r3676, r3688;
}
{
sub.f16x2 %22, r3673, r3685;
}
{
sub.f16x2 %23, r3676, r3688;
}
{
add.f16x2 %14, r3679, r3697;
}
{
add.f16x2 %15, r3682, r3691;
}
{
sub.f16x2 %30, r3679, r3697;
}
{
sub.f16x2 %31, r3682, r3691;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1181, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<460>;
.reg .b32 r<3750>;
.reg .b64 rd<2>;
mov.u32 r3723, %tid.y;
shl.b32 r3724, r3723, 17;
mov.u32 r3725, %32;
add.s32 r3726, r3725, r3724;
mov.u32 r3727, %tid.x;
{
add.f16x2 r1, %33, %49;
}
{
add.f16x2 r4, %34, %50;
}
{
sub.f16x2 r7, %33, %49;
}
{
sub.f16x2 r10, %34, %50;
}
{
add.f16x2 r13, %41, %57;
}
{
add.f16x2 r16, %42, %58;
}
{
sub.f16x2 r19, %41, %57;
}
{
sub.f16x2 r22, %42, %58;
}
{
neg.f16x2 r25, r22;
}
{
add.f16x2 r27, r1, r13;
}
{
add.f16x2 r30, r4, r16;
}
{
sub.f16x2 r33, r1, r13;
}
{
sub.f16x2 r36, r4, r16;
}
{
add.f16x2 r39, r7, r25;
}
{
add.f16x2 r42, r10, r19;
}
{
sub.f16x2 r45, r7, r25;
}
{
sub.f16x2 r48, r10, r19;
}
{
add.f16x2 r51, %37, %53;
}
{
add.f16x2 r54, %38, %54;
}
{
sub.f16x2 r57, %37, %53;
}
{
sub.f16x2 r60, %38, %54;
}
{
add.f16x2 r63, %45, %61;
}
{
add.f16x2 r66, %46, %62;
}
{
sub.f16x2 r69, %45, %61;
}
{
sub.f16x2 r72, %46, %62;
}
{
neg.f16x2 r75, r72;
}
{
add.f16x2 r77, r51, r63;
}
{
add.f16x2 r80, r54, r66;
}
{
sub.f16x2 r83, r51, r63;
}
{
sub.f16x2 r86, r54, r66;
}
{
add.f16x2 r89, r57, r75;
}
{
add.f16x2 r92, r60, r69;
}
{
sub.f16x2 r95, r57, r75;
}
{
sub.f16x2 r98, r60, r69;
}
mov.f32 f380, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r102, {low, high};
}
mov.f32 f448, 0f3F800000;
mov.f32 f378, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f378;
cvt.rn.f16.f32 high, f378;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r106, {low, high};
}
mov.f32 f447, 0fBF800000;
{
mul.f16x2 r115, r89, r101;
}
{
mul.f16x2 r118, r92, r102;
}
{
sub.f16x2 r121, r115, r118;
}
{
mul.f16x2 r124, r89, r102;
}
{
fma.rn.f16x2 r127, r92, r101, r124;
}
{
neg.f16x2 r131, r86;
}
{
mul.f16x2 r133, r95, r105;
}
{
mul.f16x2 r136, r98, r106;
}
{
sub.f16x2 r139, r133, r136;
}
{
mul.f16x2 r142, r95, r106;
}
{
fma.rn.f16x2 r145, r98, r105, r142;
}
{
add.f16x2 r149, r27, r77;
}
{
add.f16x2 r152, r30, r80;
}
{
sub.f16x2 r155, r27, r77;
}
{
sub.f16x2 r158, r30, r80;
}
{
add.f16x2 r161, r39, r121;
}
{
add.f16x2 r164, r42, r127;
}
{
sub.f16x2 r167, r39, r121;
}
{
sub.f16x2 r170, r42, r127;
}
{
add.f16x2 r173, r33, r131;
}
{
add.f16x2 r176, r36, r83;
}
{
sub.f16x2 r179, r33, r131;
}
{
sub.f16x2 r182, r36, r83;
}
{
add.f16x2 r185, r45, r139;
}
{
add.f16x2 r188, r48, r145;
}
{
sub.f16x2 r191, r45, r139;
}
{
sub.f16x2 r194, r48, r145;
}
{
add.f16x2 r197, %35, %51;
}
{
add.f16x2 r200, %36, %52;
}
{
sub.f16x2 r203, %35, %51;
}
{
sub.f16x2 r206, %36, %52;
}
{
add.f16x2 r209, %43, %59;
}
{
add.f16x2 r212, %44, %60;
}
{
sub.f16x2 r215, %43, %59;
}
{
sub.f16x2 r218, %44, %60;
}
{
neg.f16x2 r221, r218;
}
{
add.f16x2 r223, r197, r209;
}
{
add.f16x2 r226, r200, r212;
}
{
sub.f16x2 r229, r197, r209;
}
{
sub.f16x2 r232, r200, r212;
}
{
add.f16x2 r235, r203, r221;
}
{
add.f16x2 r238, r206, r215;
}
{
sub.f16x2 r241, r203, r221;
}
{
sub.f16x2 r244, r206, r215;
}
{
add.f16x2 r247, %39, %55;
}
{
add.f16x2 r250, %40, %56;
}
{
sub.f16x2 r253, %39, %55;
}
{
sub.f16x2 r256, %40, %56;
}
{
add.f16x2 r259, %47, %63;
}
{
add.f16x2 r262, %48, %64;
}
{
sub.f16x2 r265, %47, %63;
}
{
sub.f16x2 r268, %48, %64;
}
{
neg.f16x2 r271, r268;
}
{
add.f16x2 r273, r247, r259;
}
{
add.f16x2 r276, r250, r262;
}
{
sub.f16x2 r279, r247, r259;
}
{
sub.f16x2 r282, r250, r262;
}
{
add.f16x2 r285, r253, r271;
}
{
add.f16x2 r288, r256, r265;
}
{
sub.f16x2 r291, r253, r271;
}
{
sub.f16x2 r294, r256, r265;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f378;
cvt.rn.f16.f32 high, f378;
mov.b32 r301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r302, {low, high};
}
{
mul.f16x2 r311, r285, r297;
}
{
mul.f16x2 r314, r288, r298;
}
{
sub.f16x2 r317, r311, r314;
}
{
mul.f16x2 r320, r285, r298;
}
{
fma.rn.f16x2 r323, r288, r297, r320;
}
{
neg.f16x2 r327, r282;
}
{
mul.f16x2 r329, r291, r301;
}
{
mul.f16x2 r332, r294, r302;
}
{
sub.f16x2 r335, r329, r332;
}
{
mul.f16x2 r338, r291, r302;
}
{
fma.rn.f16x2 r341, r294, r301, r338;
}
{
add.f16x2 r345, r223, r273;
}
{
add.f16x2 r348, r226, r276;
}
{
sub.f16x2 r351, r223, r273;
}
{
sub.f16x2 r354, r226, r276;
}
{
add.f16x2 r357, r235, r317;
}
{
add.f16x2 r360, r238, r323;
}
{
sub.f16x2 r363, r235, r317;
}
{
sub.f16x2 r366, r238, r323;
}
{
add.f16x2 r369, r229, r327;
}
{
add.f16x2 r372, r232, r279;
}
{
sub.f16x2 r375, r229, r327;
}
{
sub.f16x2 r378, r232, r279;
}
{
add.f16x2 r381, r241, r335;
}
{
add.f16x2 r384, r244, r341;
}
{
sub.f16x2 r387, r241, r335;
}
{
sub.f16x2 r390, r244, r341;
}
mov.f32 f376, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r393, {low, high};
}
mov.f32 f384, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r396, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r398, {low, high};
}
mov.f32 f374, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f374;
cvt.rn.f16.f32 high, f374;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f378;
cvt.rn.f16.f32 high, f378;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r404, {low, high};
}
mov.f32 f382, 0fBF6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f382;
cvt.rn.f16.f32 high, f382;
mov.b32 r405, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r406, {low, high};
}
{
mul.f16x2 r423, r357, r393;
}
{
mul.f16x2 r426, r360, r394;
}
{
sub.f16x2 r429, r423, r426;
}
{
mul.f16x2 r432, r357, r394;
}
{
fma.rn.f16x2 r435, r360, r393, r432;
}
{
mul.f16x2 r439, r369, r395;
}
{
mul.f16x2 r442, r372, r396;
}
{
sub.f16x2 r445, r439, r442;
}
{
mul.f16x2 r448, r369, r396;
}
{
fma.rn.f16x2 r451, r372, r395, r448;
}
{
mul.f16x2 r455, r381, r397;
}
{
mul.f16x2 r458, r384, r398;
}
{
sub.f16x2 r461, r455, r458;
}
{
mul.f16x2 r464, r381, r398;
}
{
fma.rn.f16x2 r467, r384, r397, r464;
}
{
neg.f16x2 r471, r354;
}
{
mul.f16x2 r473, r363, r401;
}
{
mul.f16x2 r476, r366, r402;
}
{
sub.f16x2 r479, r473, r476;
}
{
mul.f16x2 r482, r363, r402;
}
{
fma.rn.f16x2 r485, r366, r401, r482;
}
{
mul.f16x2 r489, r375, r403;
}
{
mul.f16x2 r492, r378, r404;
}
{
sub.f16x2 r495, r489, r492;
}
{
mul.f16x2 r498, r375, r404;
}
{
fma.rn.f16x2 r501, r378, r403, r498;
}
{
mul.f16x2 r505, r387, r405;
}
{
mul.f16x2 r508, r390, r406;
}
{
sub.f16x2 r511, r505, r508;
}
{
mul.f16x2 r514, r387, r406;
}
{
fma.rn.f16x2 r517, r390, r405, r514;
}
{
add.f16x2 r521, r149, r345;
}
{
add.f16x2 r524, r152, r348;
}
{
sub.f16x2 r527, r149, r345;
}
{
sub.f16x2 r530, r152, r348;
}
{
add.f16x2 r533, r161, r429;
}
{
add.f16x2 r536, r164, r435;
}
{
sub.f16x2 r539, r161, r429;
}
{
sub.f16x2 r542, r164, r435;
}
{
add.f16x2 r545, r173, r445;
}
{
add.f16x2 r548, r176, r451;
}
{
sub.f16x2 r551, r173, r445;
}
{
sub.f16x2 r554, r176, r451;
}
{
add.f16x2 r557, r185, r461;
}
{
add.f16x2 r560, r188, r467;
}
{
sub.f16x2 r563, r185, r461;
}
{
sub.f16x2 r566, r188, r467;
}
{
add.f16x2 r569, r155, r471;
}
{
add.f16x2 r572, r158, r351;
}
{
sub.f16x2 r575, r155, r471;
}
{
sub.f16x2 r578, r158, r351;
}
{
add.f16x2 r581, r167, r479;
}
{
add.f16x2 r584, r170, r485;
}
{
sub.f16x2 r587, r167, r479;
}
{
sub.f16x2 r590, r170, r485;
}
{
add.f16x2 r593, r179, r495;
}
{
add.f16x2 r596, r182, r501;
}
{
sub.f16x2 r599, r179, r495;
}
{
sub.f16x2 r602, r182, r501;
}
{
add.f16x2 r605, r191, r511;
}
{
add.f16x2 r608, r194, r517;
}
{
sub.f16x2 r611, r191, r511;
}
{
sub.f16x2 r614, r194, r517;
}
and.b32 r3728, r3727, 1023;
shl.b32 r3729, r3727, 7;
and.b32 r3730, r3729, -131072;
add.s32 r3731, r3726, r3730;
cvt.rn.f32.u32 f451, r3728;
mul.f32 f452, f451, 0f39C90FDB;
cos.approx.f32 f117, f452;
sin.approx.f32 f453, f452;
neg.f32 f118, f453;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f117;
cvt.rn.f16.f32 high, f118;
mov.b32 r617, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r620, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r622, {high, high};
}
{
mul.f16x2 r624, r536, r622;
}
{
fma.rn.f16x2 r627, r533, r620, r624;
}
{
mul.f16x2 r631, r533, r622;
}
{
neg.f16x2 r634, r631;
}
{
fma.rn.f16x2 r636, r536, r620, r634;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r640, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r642, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r644, {low, high};
}
{
mul.f16x2 r645, r642, r644;
}
{
mul.f16x2 r648, r617, r640;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r651, {high, low};
}
{
fma.rn.f16x2 r653, r645, r651, r648;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r653;
mov.b32 r657, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r653;
mov.b32 r659, {high, high};
}
{
mul.f16x2 r661, r548, r659;
}
{
fma.rn.f16x2 r664, r545, r657, r661;
}
{
mul.f16x2 r668, r545, r659;
}
{
neg.f16x2 r671, r668;
}
{
fma.rn.f16x2 r673, r548, r657, r671;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r677, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r679, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r681, {low, high};
}
{
mul.f16x2 r682, r679, r681;
}
{
mul.f16x2 r685, r653, r677;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r653;
mov.b32 r688, {high, low};
}
{
fma.rn.f16x2 r690, r682, r688, r685;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r690;
mov.b32 r694, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r690;
mov.b32 r696, {high, high};
}
{
mul.f16x2 r698, r560, r696;
}
{
fma.rn.f16x2 r701, r557, r694, r698;
}
{
mul.f16x2 r705, r557, r696;
}
{
neg.f16x2 r708, r705;
}
{
fma.rn.f16x2 r710, r560, r694, r708;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r714, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r716, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r718, {low, high};
}
{
mul.f16x2 r719, r716, r718;
}
{
mul.f16x2 r722, r690, r714;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r690;
mov.b32 r725, {high, low};
}
{
fma.rn.f16x2 r727, r719, r725, r722;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r727;
mov.b32 r731, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r727;
mov.b32 r733, {high, high};
}
{
mul.f16x2 r735, r572, r733;
}
{
fma.rn.f16x2 r738, r569, r731, r735;
}
{
mul.f16x2 r742, r569, r733;
}
{
neg.f16x2 r745, r742;
}
{
fma.rn.f16x2 r747, r572, r731, r745;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r751, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r753, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r755, {low, high};
}
{
mul.f16x2 r756, r753, r755;
}
{
mul.f16x2 r759, r727, r751;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r727;
mov.b32 r762, {high, low};
}
{
fma.rn.f16x2 r764, r756, r762, r759;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r764;
mov.b32 r768, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r764;
mov.b32 r770, {high, high};
}
{
mul.f16x2 r772, r584, r770;
}
{
fma.rn.f16x2 r775, r581, r768, r772;
}
{
mul.f16x2 r779, r581, r770;
}
{
neg.f16x2 r782, r779;
}
{
fma.rn.f16x2 r784, r584, r768, r782;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r788, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r790, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r792, {low, high};
}
{
mul.f16x2 r793, r790, r792;
}
{
mul.f16x2 r796, r764, r788;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r764;
mov.b32 r799, {high, low};
}
{
fma.rn.f16x2 r801, r793, r799, r796;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r801;
mov.b32 r805, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r801;
mov.b32 r807, {high, high};
}
{
mul.f16x2 r809, r596, r807;
}
{
fma.rn.f16x2 r812, r593, r805, r809;
}
{
mul.f16x2 r816, r593, r807;
}
{
neg.f16x2 r819, r816;
}
{
fma.rn.f16x2 r821, r596, r805, r819;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r825, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r827, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r829, {low, high};
}
{
mul.f16x2 r830, r827, r829;
}
{
mul.f16x2 r833, r801, r825;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r801;
mov.b32 r836, {high, low};
}
{
fma.rn.f16x2 r838, r830, r836, r833;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r838;
mov.b32 r842, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r838;
mov.b32 r844, {high, high};
}
{
mul.f16x2 r846, r608, r844;
}
{
fma.rn.f16x2 r849, r605, r842, r846;
}
{
mul.f16x2 r853, r605, r844;
}
{
neg.f16x2 r856, r853;
}
{
fma.rn.f16x2 r858, r608, r842, r856;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r862, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r864, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r866, {low, high};
}
{
mul.f16x2 r867, r864, r866;
}
{
mul.f16x2 r870, r838, r862;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r838;
mov.b32 r873, {high, low};
}
{
fma.rn.f16x2 r875, r867, r873, r870;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r875;
mov.b32 r879, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r875;
mov.b32 r881, {high, high};
}
{
mul.f16x2 r883, r530, r881;
}
{
fma.rn.f16x2 r886, r527, r879, r883;
}
{
mul.f16x2 r890, r527, r881;
}
{
neg.f16x2 r893, r890;
}
{
fma.rn.f16x2 r895, r530, r879, r893;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r899, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r901, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r903, {low, high};
}
{
mul.f16x2 r904, r901, r903;
}
{
mul.f16x2 r907, r875, r899;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r875;
mov.b32 r910, {high, low};
}
{
fma.rn.f16x2 r912, r904, r910, r907;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r912;
mov.b32 r916, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r912;
mov.b32 r918, {high, high};
}
{
mul.f16x2 r920, r542, r918;
}
{
fma.rn.f16x2 r923, r539, r916, r920;
}
{
mul.f16x2 r927, r539, r918;
}
{
neg.f16x2 r930, r927;
}
{
fma.rn.f16x2 r932, r542, r916, r930;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r936, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r938, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r940, {low, high};
}
{
mul.f16x2 r941, r938, r940;
}
{
mul.f16x2 r944, r912, r936;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r912;
mov.b32 r947, {high, low};
}
{
fma.rn.f16x2 r949, r941, r947, r944;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r949;
mov.b32 r953, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r949;
mov.b32 r955, {high, high};
}
{
mul.f16x2 r957, r554, r955;
}
{
fma.rn.f16x2 r960, r551, r953, r957;
}
{
mul.f16x2 r964, r551, r955;
}
{
neg.f16x2 r967, r964;
}
{
fma.rn.f16x2 r969, r554, r953, r967;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r973, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r975, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r977, {low, high};
}
{
mul.f16x2 r978, r975, r977;
}
{
mul.f16x2 r981, r949, r973;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r949;
mov.b32 r984, {high, low};
}
{
fma.rn.f16x2 r986, r978, r984, r981;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r986;
mov.b32 r990, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r986;
mov.b32 r992, {high, high};
}
{
mul.f16x2 r994, r566, r992;
}
{
fma.rn.f16x2 r997, r563, r990, r994;
}
{
mul.f16x2 r1001, r563, r992;
}
{
neg.f16x2 r1004, r1001;
}
{
fma.rn.f16x2 r1006, r566, r990, r1004;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1010, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1012, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r1014, {low, high};
}
{
mul.f16x2 r1015, r1012, r1014;
}
{
mul.f16x2 r1018, r986, r1010;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r986;
mov.b32 r1021, {high, low};
}
{
fma.rn.f16x2 r1023, r1015, r1021, r1018;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1023;
mov.b32 r1027, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1023;
mov.b32 r1029, {high, high};
}
{
mul.f16x2 r1031, r578, r1029;
}
{
fma.rn.f16x2 r1034, r575, r1027, r1031;
}
{
mul.f16x2 r1038, r575, r1029;
}
{
neg.f16x2 r1041, r1038;
}
{
fma.rn.f16x2 r1043, r578, r1027, r1041;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1047, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1049, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r1051, {low, high};
}
{
mul.f16x2 r1052, r1049, r1051;
}
{
mul.f16x2 r1055, r1023, r1047;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1023;
mov.b32 r1058, {high, low};
}
{
fma.rn.f16x2 r1060, r1052, r1058, r1055;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1060;
mov.b32 r1064, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1060;
mov.b32 r1066, {high, high};
}
{
mul.f16x2 r1068, r590, r1066;
}
{
fma.rn.f16x2 r1071, r587, r1064, r1068;
}
{
mul.f16x2 r1075, r587, r1066;
}
{
neg.f16x2 r1078, r1075;
}
{
fma.rn.f16x2 r1080, r590, r1064, r1078;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1084, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1086, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r1088, {low, high};
}
{
mul.f16x2 r1089, r1086, r1088;
}
{
mul.f16x2 r1092, r1060, r1084;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1060;
mov.b32 r1095, {high, low};
}
{
fma.rn.f16x2 r1097, r1089, r1095, r1092;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1097;
mov.b32 r1101, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1097;
mov.b32 r1103, {high, high};
}
{
mul.f16x2 r1105, r602, r1103;
}
{
fma.rn.f16x2 r1108, r599, r1101, r1105;
}
{
mul.f16x2 r1112, r599, r1103;
}
{
neg.f16x2 r1115, r1112;
}
{
fma.rn.f16x2 r1117, r602, r1101, r1115;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1121, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1123, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r1125, {low, high};
}
{
mul.f16x2 r1126, r1123, r1125;
}
{
mul.f16x2 r1129, r1097, r1121;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1097;
mov.b32 r1132, {high, low};
}
{
fma.rn.f16x2 r1134, r1126, r1132, r1129;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1134;
mov.b32 r1138, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1134;
mov.b32 r1140, {high, high};
}
{
mul.f16x2 r1142, r614, r1140;
}
{
fma.rn.f16x2 r1145, r611, r1138, r1142;
}
{
mul.f16x2 r1149, r611, r1140;
}
{
neg.f16x2 r1152, r1149;
}
{
fma.rn.f16x2 r1154, r614, r1138, r1152;
}
barrier.sync 0;
and.b32 r3732, r3729, 130944;
add.s32 r3733, r3731, r3732;
st.shared.v4.f32 [r3733], {r521, r524, r627, r636};
st.shared.v4.f32 [r3733+16], {r664, r673, r701, r710};
st.shared.v4.f32 [r3733+32], {r738, r747, r775, r784};
st.shared.v4.f32 [r3733+48], {r812, r821, r849, r858};
st.shared.v4.f32 [r3733+64], {r886, r895, r923, r932};
st.shared.v4.f32 [r3733+80], {r960, r969, r997, r1006};
st.shared.v4.f32 [r3733+96], {r1034, r1043, r1071, r1080};
st.shared.v4.f32 [r3733+112], {r1108, r1117, r1145, r1154};
barrier.sync 0;
mad.lo.s32 r3734, r3728, -120, r3733;
ld.shared.u32 r1176, [r3734];
ld.shared.u32 r1179, [r3734+4];
ld.shared.u32 r1372, [r3734+8192];
ld.shared.u32 r1375, [r3734+8196];
ld.shared.u32 r1226, [r3734+16384];
ld.shared.u32 r1229, [r3734+16388];
ld.shared.u32 r1422, [r3734+24576];
ld.shared.u32 r1425, [r3734+24580];
ld.shared.u32 r1188, [r3734+32768];
ld.shared.u32 r1191, [r3734+32772];
ld.shared.u32 r1384, [r3734+40960];
ld.shared.u32 r1387, [r3734+40964];
ld.shared.u32 r1238, [r3734+49152];
ld.shared.u32 r1241, [r3734+49156];
ld.shared.u32 r1434, [r3734+57344];
ld.shared.u32 r1437, [r3734+57348];
ld.shared.u32 r1177, [r3734+65536];
ld.shared.u32 r1180, [r3734+65540];
ld.shared.u32 r1373, [r3734+73728];
ld.shared.u32 r1376, [r3734+73732];
ld.shared.u32 r1227, [r3734+81920];
ld.shared.u32 r1230, [r3734+81924];
ld.shared.u32 r1423, [r3734+90112];
ld.shared.u32 r1426, [r3734+90116];
ld.shared.u32 r1189, [r3734+98304];
ld.shared.u32 r1192, [r3734+98308];
ld.shared.u32 r1385, [r3734+106496];
ld.shared.u32 r1388, [r3734+106500];
ld.shared.u32 r1239, [r3734+114688];
ld.shared.u32 r1242, [r3734+114692];
ld.shared.u32 r1435, [r3734+122880];
ld.shared.u32 r1438, [r3734+122884];
{
add.f16x2 r1175, r1176, r1177;
}
{
add.f16x2 r1178, r1179, r1180;
}
{
sub.f16x2 r1181, r1176, r1177;
}
{
sub.f16x2 r1184, r1179, r1180;
}
{
add.f16x2 r1187, r1188, r1189;
}
{
add.f16x2 r1190, r1191, r1192;
}
{
sub.f16x2 r1193, r1188, r1189;
}
{
sub.f16x2 r1196, r1191, r1192;
}
{
neg.f16x2 r1199, r1196;
}
{
add.f16x2 r1201, r1175, r1187;
}
{
add.f16x2 r1204, r1178, r1190;
}
{
sub.f16x2 r1207, r1175, r1187;
}
{
sub.f16x2 r1210, r1178, r1190;
}
{
add.f16x2 r1213, r1181, r1199;
}
{
add.f16x2 r1216, r1184, r1193;
}
{
sub.f16x2 r1219, r1181, r1199;
}
{
sub.f16x2 r1222, r1184, r1193;
}
{
add.f16x2 r1225, r1226, r1227;
}
{
add.f16x2 r1228, r1229, r1230;
}
{
sub.f16x2 r1231, r1226, r1227;
}
{
sub.f16x2 r1234, r1229, r1230;
}
{
add.f16x2 r1237, r1238, r1239;
}
{
add.f16x2 r1240, r1241, r1242;
}
{
sub.f16x2 r1243, r1238, r1239;
}
{
sub.f16x2 r1246, r1241, r1242;
}
{
neg.f16x2 r1249, r1246;
}
{
add.f16x2 r1251, r1225, r1237;
}
{
add.f16x2 r1254, r1228, r1240;
}
{
sub.f16x2 r1257, r1225, r1237;
}
{
sub.f16x2 r1260, r1228, r1240;
}
{
add.f16x2 r1263, r1231, r1249;
}
{
add.f16x2 r1266, r1234, r1243;
}
{
sub.f16x2 r1269, r1231, r1249;
}
{
sub.f16x2 r1272, r1234, r1243;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r1275, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r1276, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f378;
cvt.rn.f16.f32 high, f378;
mov.b32 r1279, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r1280, {low, high};
}
{
mul.f16x2 r1289, r1263, r1275;
}
{
mul.f16x2 r1292, r1266, r1276;
}
{
sub.f16x2 r1295, r1289, r1292;
}
{
mul.f16x2 r1298, r1263, r1276;
}
{
fma.rn.f16x2 r1301, r1266, r1275, r1298;
}
{
neg.f16x2 r1305, r1260;
}
{
mul.f16x2 r1307, r1269, r1279;
}
{
mul.f16x2 r1310, r1272, r1280;
}
{
sub.f16x2 r1313, r1307, r1310;
}
{
mul.f16x2 r1316, r1269, r1280;
}
{
fma.rn.f16x2 r1319, r1272, r1279, r1316;
}
{
add.f16x2 r1323, r1201, r1251;
}
{
add.f16x2 r1326, r1204, r1254;
}
{
sub.f16x2 r1329, r1201, r1251;
}
{
sub.f16x2 r1332, r1204, r1254;
}
{
add.f16x2 r1335, r1213, r1295;
}
{
add.f16x2 r1338, r1216, r1301;
}
{
sub.f16x2 r1341, r1213, r1295;
}
{
sub.f16x2 r1344, r1216, r1301;
}
{
add.f16x2 r1347, r1207, r1305;
}
{
add.f16x2 r1350, r1210, r1257;
}
{
sub.f16x2 r1353, r1207, r1305;
}
{
sub.f16x2 r1356, r1210, r1257;
}
{
add.f16x2 r1359, r1219, r1313;
}
{
add.f16x2 r1362, r1222, r1319;
}
{
sub.f16x2 r1365, r1219, r1313;
}
{
sub.f16x2 r1368, r1222, r1319;
}
{
add.f16x2 r1371, r1372, r1373;
}
{
add.f16x2 r1374, r1375, r1376;
}
{
sub.f16x2 r1377, r1372, r1373;
}
{
sub.f16x2 r1380, r1375, r1376;
}
{
add.f16x2 r1383, r1384, r1385;
}
{
add.f16x2 r1386, r1387, r1388;
}
{
sub.f16x2 r1389, r1384, r1385;
}
{
sub.f16x2 r1392, r1387, r1388;
}
{
neg.f16x2 r1395, r1392;
}
{
add.f16x2 r1397, r1371, r1383;
}
{
add.f16x2 r1400, r1374, r1386;
}
{
sub.f16x2 r1403, r1371, r1383;
}
{
sub.f16x2 r1406, r1374, r1386;
}
{
add.f16x2 r1409, r1377, r1395;
}
{
add.f16x2 r1412, r1380, r1389;
}
{
sub.f16x2 r1415, r1377, r1395;
}
{
sub.f16x2 r1418, r1380, r1389;
}
{
add.f16x2 r1421, r1422, r1423;
}
{
add.f16x2 r1424, r1425, r1426;
}
{
sub.f16x2 r1427, r1422, r1423;
}
{
sub.f16x2 r1430, r1425, r1426;
}
{
add.f16x2 r1433, r1434, r1435;
}
{
add.f16x2 r1436, r1437, r1438;
}
{
sub.f16x2 r1439, r1434, r1435;
}
{
sub.f16x2 r1442, r1437, r1438;
}
{
neg.f16x2 r1445, r1442;
}
{
add.f16x2 r1447, r1421, r1433;
}
{
add.f16x2 r1450, r1424, r1436;
}
{
sub.f16x2 r1453, r1421, r1433;
}
{
sub.f16x2 r1456, r1424, r1436;
}
{
add.f16x2 r1459, r1427, r1445;
}
{
add.f16x2 r1462, r1430, r1439;
}
{
sub.f16x2 r1465, r1427, r1445;
}
{
sub.f16x2 r1468, r1430, r1439;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r1471, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r1472, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f378;
cvt.rn.f16.f32 high, f378;
mov.b32 r1475, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r1476, {low, high};
}
{
mul.f16x2 r1485, r1459, r1471;
}
{
mul.f16x2 r1488, r1462, r1472;
}
{
sub.f16x2 r1491, r1485, r1488;
}
{
mul.f16x2 r1494, r1459, r1472;
}
{
fma.rn.f16x2 r1497, r1462, r1471, r1494;
}
{
neg.f16x2 r1501, r1456;
}
{
mul.f16x2 r1503, r1465, r1475;
}
{
mul.f16x2 r1506, r1468, r1476;
}
{
sub.f16x2 r1509, r1503, r1506;
}
{
mul.f16x2 r1512, r1465, r1476;
}
{
fma.rn.f16x2 r1515, r1468, r1475, r1512;
}
{
add.f16x2 r1519, r1397, r1447;
}
{
add.f16x2 r1522, r1400, r1450;
}
{
sub.f16x2 r1525, r1397, r1447;
}
{
sub.f16x2 r1528, r1400, r1450;
}
{
add.f16x2 r1531, r1409, r1491;
}
{
add.f16x2 r1534, r1412, r1497;
}
{
sub.f16x2 r1537, r1409, r1491;
}
{
sub.f16x2 r1540, r1412, r1497;
}
{
add.f16x2 r1543, r1403, r1501;
}
{
add.f16x2 r1546, r1406, r1453;
}
{
sub.f16x2 r1549, r1403, r1501;
}
{
sub.f16x2 r1552, r1406, r1453;
}
{
add.f16x2 r1555, r1415, r1509;
}
{
add.f16x2 r1558, r1418, r1515;
}
{
sub.f16x2 r1561, r1415, r1509;
}
{
sub.f16x2 r1564, r1418, r1515;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r1567, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r1568, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r1569, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r1570, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r1571, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r1572, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f374;
cvt.rn.f16.f32 high, f374;
mov.b32 r1575, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r1576, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f378;
cvt.rn.f16.f32 high, f378;
mov.b32 r1577, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r1578, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f382;
cvt.rn.f16.f32 high, f382;
mov.b32 r1579, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r1580, {low, high};
}
{
mul.f16x2 r1597, r1531, r1567;
}
{
mul.f16x2 r1600, r1534, r1568;
}
{
sub.f16x2 r1603, r1597, r1600;
}
{
mul.f16x2 r1606, r1531, r1568;
}
{
fma.rn.f16x2 r1609, r1534, r1567, r1606;
}
{
mul.f16x2 r1613, r1543, r1569;
}
{
mul.f16x2 r1616, r1546, r1570;
}
{
sub.f16x2 r1619, r1613, r1616;
}
{
mul.f16x2 r1622, r1543, r1570;
}
{
fma.rn.f16x2 r1625, r1546, r1569, r1622;
}
{
mul.f16x2 r1629, r1555, r1571;
}
{
mul.f16x2 r1632, r1558, r1572;
}
{
sub.f16x2 r1635, r1629, r1632;
}
{
mul.f16x2 r1638, r1555, r1572;
}
{
fma.rn.f16x2 r1641, r1558, r1571, r1638;
}
{
neg.f16x2 r1645, r1528;
}
{
mul.f16x2 r1647, r1537, r1575;
}
{
mul.f16x2 r1650, r1540, r1576;
}
{
sub.f16x2 r1653, r1647, r1650;
}
{
mul.f16x2 r1656, r1537, r1576;
}
{
fma.rn.f16x2 r1659, r1540, r1575, r1656;
}
{
mul.f16x2 r1663, r1549, r1577;
}
{
mul.f16x2 r1666, r1552, r1578;
}
{
sub.f16x2 r1669, r1663, r1666;
}
{
mul.f16x2 r1672, r1549, r1578;
}
{
fma.rn.f16x2 r1675, r1552, r1577, r1672;
}
{
mul.f16x2 r1679, r1561, r1579;
}
{
mul.f16x2 r1682, r1564, r1580;
}
{
sub.f16x2 r1685, r1679, r1682;
}
{
mul.f16x2 r1688, r1561, r1580;
}
{
fma.rn.f16x2 r1691, r1564, r1579, r1688;
}
{
add.f16x2 r1695, r1323, r1519;
}
{
add.f16x2 r1698, r1326, r1522;
}
{
sub.f16x2 r1701, r1323, r1519;
}
{
sub.f16x2 r1704, r1326, r1522;
}
{
add.f16x2 r1707, r1335, r1603;
}
{
add.f16x2 r1710, r1338, r1609;
}
{
sub.f16x2 r1713, r1335, r1603;
}
{
sub.f16x2 r1716, r1338, r1609;
}
{
add.f16x2 r1719, r1347, r1619;
}
{
add.f16x2 r1722, r1350, r1625;
}
{
sub.f16x2 r1725, r1347, r1619;
}
{
sub.f16x2 r1728, r1350, r1625;
}
{
add.f16x2 r1731, r1359, r1635;
}
{
add.f16x2 r1734, r1362, r1641;
}
{
sub.f16x2 r1737, r1359, r1635;
}
{
sub.f16x2 r1740, r1362, r1641;
}
{
add.f16x2 r1743, r1329, r1645;
}
{
add.f16x2 r1746, r1332, r1525;
}
{
sub.f16x2 r1749, r1329, r1645;
}
{
sub.f16x2 r1752, r1332, r1525;
}
{
add.f16x2 r1755, r1341, r1653;
}
{
add.f16x2 r1758, r1344, r1659;
}
{
sub.f16x2 r1761, r1341, r1653;
}
{
sub.f16x2 r1764, r1344, r1659;
}
{
add.f16x2 r1767, r1353, r1669;
}
{
add.f16x2 r1770, r1356, r1675;
}
{
sub.f16x2 r1773, r1353, r1669;
}
{
sub.f16x2 r1776, r1356, r1675;
}
{
add.f16x2 r1779, r1365, r1685;
}
{
add.f16x2 r1782, r1368, r1691;
}
{
sub.f16x2 r1785, r1365, r1685;
}
{
sub.f16x2 r1788, r1368, r1691;
}
and.b32 r3735, r3727, 1008;
bfe.u32 r3736, r3727, 4, 6;
cvt.rn.f32.u32 f454, r3736;
mul.f32 f455, f454, 0f3BC90FDB;
cos.approx.f32 f267, f455;
sin.approx.f32 f456, f455;
neg.f32 f268, f456;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f267;
cvt.rn.f16.f32 high, f268;
mov.b32 r1791, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1794, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1796, {high, high};
}
{
mul.f16x2 r1798, r1710, r1796;
}
{
fma.rn.f16x2 r1801, r1707, r1794, r1798;
}
{
mul.f16x2 r1805, r1707, r1796;
}
{
neg.f16x2 r1808, r1805;
}
{
fma.rn.f16x2 r1810, r1710, r1794, r1808;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1814, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1816, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r1818, {low, high};
}
{
mul.f16x2 r1819, r1816, r1818;
}
{
mul.f16x2 r1822, r1791, r1814;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1825, {high, low};
}
{
fma.rn.f16x2 r1827, r1819, r1825, r1822;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1827;
mov.b32 r1831, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1827;
mov.b32 r1833, {high, high};
}
{
mul.f16x2 r1835, r1722, r1833;
}
{
fma.rn.f16x2 r1838, r1719, r1831, r1835;
}
{
mul.f16x2 r1842, r1719, r1833;
}
{
neg.f16x2 r1845, r1842;
}
{
fma.rn.f16x2 r1847, r1722, r1831, r1845;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1851, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1853, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r1855, {low, high};
}
{
mul.f16x2 r1856, r1853, r1855;
}
{
mul.f16x2 r1859, r1827, r1851;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1827;
mov.b32 r1862, {high, low};
}
{
fma.rn.f16x2 r1864, r1856, r1862, r1859;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1864;
mov.b32 r1868, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1864;
mov.b32 r1870, {high, high};
}
{
mul.f16x2 r1872, r1734, r1870;
}
{
fma.rn.f16x2 r1875, r1731, r1868, r1872;
}
{
mul.f16x2 r1879, r1731, r1870;
}
{
neg.f16x2 r1882, r1879;
}
{
fma.rn.f16x2 r1884, r1734, r1868, r1882;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1888, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1890, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r1892, {low, high};
}
{
mul.f16x2 r1893, r1890, r1892;
}
{
mul.f16x2 r1896, r1864, r1888;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1864;
mov.b32 r1899, {high, low};
}
{
fma.rn.f16x2 r1901, r1893, r1899, r1896;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1905, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1907, {high, high};
}
{
mul.f16x2 r1909, r1746, r1907;
}
{
fma.rn.f16x2 r1912, r1743, r1905, r1909;
}
{
mul.f16x2 r1916, r1743, r1907;
}
{
neg.f16x2 r1919, r1916;
}
{
fma.rn.f16x2 r1921, r1746, r1905, r1919;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1925, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1927, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r1929, {low, high};
}
{
mul.f16x2 r1930, r1927, r1929;
}
{
mul.f16x2 r1933, r1901, r1925;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1936, {high, low};
}
{
fma.rn.f16x2 r1938, r1930, r1936, r1933;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1938;
mov.b32 r1942, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1938;
mov.b32 r1944, {high, high};
}
{
mul.f16x2 r1946, r1758, r1944;
}
{
fma.rn.f16x2 r1949, r1755, r1942, r1946;
}
{
mul.f16x2 r1953, r1755, r1944;
}
{
neg.f16x2 r1956, r1953;
}
{
fma.rn.f16x2 r1958, r1758, r1942, r1956;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1962, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1964, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r1966, {low, high};
}
{
mul.f16x2 r1967, r1964, r1966;
}
{
mul.f16x2 r1970, r1938, r1962;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1938;
mov.b32 r1973, {high, low};
}
{
fma.rn.f16x2 r1975, r1967, r1973, r1970;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1975;
mov.b32 r1979, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1975;
mov.b32 r1981, {high, high};
}
{
mul.f16x2 r1983, r1770, r1981;
}
{
fma.rn.f16x2 r1986, r1767, r1979, r1983;
}
{
mul.f16x2 r1990, r1767, r1981;
}
{
neg.f16x2 r1993, r1990;
}
{
fma.rn.f16x2 r1995, r1770, r1979, r1993;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r1999, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2001, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r2003, {low, high};
}
{
mul.f16x2 r2004, r2001, r2003;
}
{
mul.f16x2 r2007, r1975, r1999;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1975;
mov.b32 r2010, {high, low};
}
{
fma.rn.f16x2 r2012, r2004, r2010, r2007;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2012;
mov.b32 r2016, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2012;
mov.b32 r2018, {high, high};
}
{
mul.f16x2 r2020, r1782, r2018;
}
{
fma.rn.f16x2 r2023, r1779, r2016, r2020;
}
{
mul.f16x2 r2027, r1779, r2018;
}
{
neg.f16x2 r2030, r2027;
}
{
fma.rn.f16x2 r2032, r1782, r2016, r2030;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2036, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2038, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r2040, {low, high};
}
{
mul.f16x2 r2041, r2038, r2040;
}
{
mul.f16x2 r2044, r2012, r2036;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2012;
mov.b32 r2047, {high, low};
}
{
fma.rn.f16x2 r2049, r2041, r2047, r2044;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2049;
mov.b32 r2053, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2049;
mov.b32 r2055, {high, high};
}
{
mul.f16x2 r2057, r1704, r2055;
}
{
fma.rn.f16x2 r2060, r1701, r2053, r2057;
}
{
mul.f16x2 r2064, r1701, r2055;
}
{
neg.f16x2 r2067, r2064;
}
{
fma.rn.f16x2 r2069, r1704, r2053, r2067;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2073, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2075, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r2077, {low, high};
}
{
mul.f16x2 r2078, r2075, r2077;
}
{
mul.f16x2 r2081, r2049, r2073;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2049;
mov.b32 r2084, {high, low};
}
{
fma.rn.f16x2 r2086, r2078, r2084, r2081;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2086;
mov.b32 r2090, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2086;
mov.b32 r2092, {high, high};
}
{
mul.f16x2 r2094, r1716, r2092;
}
{
fma.rn.f16x2 r2097, r1713, r2090, r2094;
}
{
mul.f16x2 r2101, r1713, r2092;
}
{
neg.f16x2 r2104, r2101;
}
{
fma.rn.f16x2 r2106, r1716, r2090, r2104;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2110, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2112, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r2114, {low, high};
}
{
mul.f16x2 r2115, r2112, r2114;
}
{
mul.f16x2 r2118, r2086, r2110;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2086;
mov.b32 r2121, {high, low};
}
{
fma.rn.f16x2 r2123, r2115, r2121, r2118;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2123;
mov.b32 r2127, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2123;
mov.b32 r2129, {high, high};
}
{
mul.f16x2 r2131, r1728, r2129;
}
{
fma.rn.f16x2 r2134, r1725, r2127, r2131;
}
{
mul.f16x2 r2138, r1725, r2129;
}
{
neg.f16x2 r2141, r2138;
}
{
fma.rn.f16x2 r2143, r1728, r2127, r2141;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2147, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2149, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r2151, {low, high};
}
{
mul.f16x2 r2152, r2149, r2151;
}
{
mul.f16x2 r2155, r2123, r2147;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2123;
mov.b32 r2158, {high, low};
}
{
fma.rn.f16x2 r2160, r2152, r2158, r2155;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2160;
mov.b32 r2164, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2160;
mov.b32 r2166, {high, high};
}
{
mul.f16x2 r2168, r1740, r2166;
}
{
fma.rn.f16x2 r2171, r1737, r2164, r2168;
}
{
mul.f16x2 r2175, r1737, r2166;
}
{
neg.f16x2 r2178, r2175;
}
{
fma.rn.f16x2 r2180, r1740, r2164, r2178;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2184, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2186, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r2188, {low, high};
}
{
mul.f16x2 r2189, r2186, r2188;
}
{
mul.f16x2 r2192, r2160, r2184;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2160;
mov.b32 r2195, {high, low};
}
{
fma.rn.f16x2 r2197, r2189, r2195, r2192;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2197;
mov.b32 r2201, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2197;
mov.b32 r2203, {high, high};
}
{
mul.f16x2 r2205, r1752, r2203;
}
{
fma.rn.f16x2 r2208, r1749, r2201, r2205;
}
{
mul.f16x2 r2212, r1749, r2203;
}
{
neg.f16x2 r2215, r2212;
}
{
fma.rn.f16x2 r2217, r1752, r2201, r2215;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2221, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2223, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r2225, {low, high};
}
{
mul.f16x2 r2226, r2223, r2225;
}
{
mul.f16x2 r2229, r2197, r2221;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2197;
mov.b32 r2232, {high, low};
}
{
fma.rn.f16x2 r2234, r2226, r2232, r2229;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2234;
mov.b32 r2238, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2234;
mov.b32 r2240, {high, high};
}
{
mul.f16x2 r2242, r1764, r2240;
}
{
fma.rn.f16x2 r2245, r1761, r2238, r2242;
}
{
mul.f16x2 r2249, r1761, r2240;
}
{
neg.f16x2 r2252, r2249;
}
{
fma.rn.f16x2 r2254, r1764, r2238, r2252;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2258, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2260, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r2262, {low, high};
}
{
mul.f16x2 r2263, r2260, r2262;
}
{
mul.f16x2 r2266, r2234, r2258;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2234;
mov.b32 r2269, {high, low};
}
{
fma.rn.f16x2 r2271, r2263, r2269, r2266;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2271;
mov.b32 r2275, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2271;
mov.b32 r2277, {high, high};
}
{
mul.f16x2 r2279, r1776, r2277;
}
{
fma.rn.f16x2 r2282, r1773, r2275, r2279;
}
{
mul.f16x2 r2286, r1773, r2277;
}
{
neg.f16x2 r2289, r2286;
}
{
fma.rn.f16x2 r2291, r1776, r2275, r2289;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2295, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1791;
mov.b32 r2297, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r2299, {low, high};
}
{
mul.f16x2 r2300, r2297, r2299;
}
{
mul.f16x2 r2303, r2271, r2295;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2271;
mov.b32 r2306, {high, low};
}
{
fma.rn.f16x2 r2308, r2300, r2306, r2303;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2308;
mov.b32 r2312, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2308;
mov.b32 r2314, {high, high};
}
{
mul.f16x2 r2316, r1788, r2314;
}
{
fma.rn.f16x2 r2319, r1785, r2312, r2316;
}
{
mul.f16x2 r2323, r1785, r2314;
}
{
neg.f16x2 r2326, r2323;
}
{
fma.rn.f16x2 r2328, r1788, r2312, r2326;
}
shl.b32 r3737, r3727, 3;
and.b32 r3738, r3737, 120;
add.s32 r3739, r3731, r3738;
barrier.sync 0;
and.b32 r3740, r3729, 129024;
add.s32 r3741, r3739, r3740;
st.shared.u32 [r3741], r1695;
st.shared.u32 [r3741+4], r1698;
st.shared.u32 [r3741+128], r1801;
st.shared.u32 [r3741+132], r1810;
st.shared.u32 [r3741+256], r1838;
st.shared.u32 [r3741+260], r1847;
st.shared.u32 [r3741+384], r1875;
st.shared.u32 [r3741+388], r1884;
st.shared.u32 [r3741+512], r1912;
st.shared.u32 [r3741+516], r1921;
st.shared.u32 [r3741+640], r1949;
st.shared.u32 [r3741+644], r1958;
st.shared.u32 [r3741+768], r1986;
st.shared.u32 [r3741+772], r1995;
st.shared.u32 [r3741+896], r2023;
st.shared.u32 [r3741+900], r2032;
st.shared.u32 [r3741+1024], r2060;
st.shared.u32 [r3741+1028], r2069;
st.shared.u32 [r3741+1152], r2097;
st.shared.u32 [r3741+1156], r2106;
st.shared.u32 [r3741+1280], r2134;
st.shared.u32 [r3741+1284], r2143;
st.shared.u32 [r3741+1408], r2171;
st.shared.u32 [r3741+1412], r2180;
st.shared.u32 [r3741+1536], r2208;
st.shared.u32 [r3741+1540], r2217;
st.shared.u32 [r3741+1664], r2245;
st.shared.u32 [r3741+1668], r2254;
st.shared.u32 [r3741+1792], r2282;
st.shared.u32 [r3741+1796], r2291;
st.shared.u32 [r3741+1920], r2319;
st.shared.u32 [r3741+1924], r2328;
barrier.sync 0;
mad.lo.s32 r3742, r3735, -120, r3741;
ld.shared.u32 r2350, [r3742];
ld.shared.u32 r2353, [r3742+4];
ld.shared.u32 r2546, [r3742+8192];
ld.shared.u32 r2549, [r3742+8196];
ld.shared.u32 r2400, [r3742+16384];
ld.shared.u32 r2403, [r3742+16388];
ld.shared.u32 r2596, [r3742+24576];
ld.shared.u32 r2599, [r3742+24580];
ld.shared.u32 r2362, [r3742+32768];
ld.shared.u32 r2365, [r3742+32772];
ld.shared.u32 r2558, [r3742+40960];
ld.shared.u32 r2561, [r3742+40964];
ld.shared.u32 r2412, [r3742+49152];
ld.shared.u32 r2415, [r3742+49156];
ld.shared.u32 r2608, [r3742+57344];
ld.shared.u32 r2611, [r3742+57348];
ld.shared.u32 r2351, [r3742+65536];
ld.shared.u32 r2354, [r3742+65540];
ld.shared.u32 r2547, [r3742+73728];
ld.shared.u32 r2550, [r3742+73732];
ld.shared.u32 r2401, [r3742+81920];
ld.shared.u32 r2404, [r3742+81924];
ld.shared.u32 r2597, [r3742+90112];
ld.shared.u32 r2600, [r3742+90116];
ld.shared.u32 r2363, [r3742+98304];
ld.shared.u32 r2366, [r3742+98308];
ld.shared.u32 r2559, [r3742+106496];
ld.shared.u32 r2562, [r3742+106500];
ld.shared.u32 r2413, [r3742+114688];
ld.shared.u32 r2416, [r3742+114692];
ld.shared.u32 r2609, [r3742+122880];
ld.shared.u32 r2612, [r3742+122884];
{
add.f16x2 r2349, r2350, r2351;
}
{
add.f16x2 r2352, r2353, r2354;
}
{
sub.f16x2 r2355, r2350, r2351;
}
{
sub.f16x2 r2358, r2353, r2354;
}
{
add.f16x2 r2361, r2362, r2363;
}
{
add.f16x2 r2364, r2365, r2366;
}
{
sub.f16x2 r2367, r2362, r2363;
}
{
sub.f16x2 r2370, r2365, r2366;
}
{
neg.f16x2 r2373, r2370;
}
{
add.f16x2 r2375, r2349, r2361;
}
{
add.f16x2 r2378, r2352, r2364;
}
{
sub.f16x2 r2381, r2349, r2361;
}
{
sub.f16x2 r2384, r2352, r2364;
}
{
add.f16x2 r2387, r2355, r2373;
}
{
add.f16x2 r2390, r2358, r2367;
}
{
sub.f16x2 r2393, r2355, r2373;
}
{
sub.f16x2 r2396, r2358, r2367;
}
{
add.f16x2 r2399, r2400, r2401;
}
{
add.f16x2 r2402, r2403, r2404;
}
{
sub.f16x2 r2405, r2400, r2401;
}
{
sub.f16x2 r2408, r2403, r2404;
}
{
add.f16x2 r2411, r2412, r2413;
}
{
add.f16x2 r2414, r2415, r2416;
}
{
sub.f16x2 r2417, r2412, r2413;
}
{
sub.f16x2 r2420, r2415, r2416;
}
{
neg.f16x2 r2423, r2420;
}
{
add.f16x2 r2425, r2399, r2411;
}
{
add.f16x2 r2428, r2402, r2414;
}
{
sub.f16x2 r2431, r2399, r2411;
}
{
sub.f16x2 r2434, r2402, r2414;
}
{
add.f16x2 r2437, r2405, r2423;
}
{
add.f16x2 r2440, r2408, r2417;
}
{
sub.f16x2 r2443, r2405, r2423;
}
{
sub.f16x2 r2446, r2408, r2417;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r2449, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r2450, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f378;
cvt.rn.f16.f32 high, f378;
mov.b32 r2453, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r2454, {low, high};
}
{
mul.f16x2 r2463, r2437, r2449;
}
{
mul.f16x2 r2466, r2440, r2450;
}
{
sub.f16x2 r2469, r2463, r2466;
}
{
mul.f16x2 r2472, r2437, r2450;
}
{
fma.rn.f16x2 r2475, r2440, r2449, r2472;
}
{
neg.f16x2 r2479, r2434;
}
{
mul.f16x2 r2481, r2443, r2453;
}
{
mul.f16x2 r2484, r2446, r2454;
}
{
sub.f16x2 r2487, r2481, r2484;
}
{
mul.f16x2 r2490, r2443, r2454;
}
{
fma.rn.f16x2 r2493, r2446, r2453, r2490;
}
{
add.f16x2 r2497, r2375, r2425;
}
{
add.f16x2 r2500, r2378, r2428;
}
{
sub.f16x2 r2503, r2375, r2425;
}
{
sub.f16x2 r2506, r2378, r2428;
}
{
add.f16x2 r2509, r2387, r2469;
}
{
add.f16x2 r2512, r2390, r2475;
}
{
sub.f16x2 r2515, r2387, r2469;
}
{
sub.f16x2 r2518, r2390, r2475;
}
{
add.f16x2 r2521, r2381, r2479;
}
{
add.f16x2 r2524, r2384, r2431;
}
{
sub.f16x2 r2527, r2381, r2479;
}
{
sub.f16x2 r2530, r2384, r2431;
}
{
add.f16x2 r2533, r2393, r2487;
}
{
add.f16x2 r2536, r2396, r2493;
}
{
sub.f16x2 r2539, r2393, r2487;
}
{
sub.f16x2 r2542, r2396, r2493;
}
{
add.f16x2 r2545, r2546, r2547;
}
{
add.f16x2 r2548, r2549, r2550;
}
{
sub.f16x2 r2551, r2546, r2547;
}
{
sub.f16x2 r2554, r2549, r2550;
}
{
add.f16x2 r2557, r2558, r2559;
}
{
add.f16x2 r2560, r2561, r2562;
}
{
sub.f16x2 r2563, r2558, r2559;
}
{
sub.f16x2 r2566, r2561, r2562;
}
{
neg.f16x2 r2569, r2566;
}
{
add.f16x2 r2571, r2545, r2557;
}
{
add.f16x2 r2574, r2548, r2560;
}
{
sub.f16x2 r2577, r2545, r2557;
}
{
sub.f16x2 r2580, r2548, r2560;
}
{
add.f16x2 r2583, r2551, r2569;
}
{
add.f16x2 r2586, r2554, r2563;
}
{
sub.f16x2 r2589, r2551, r2569;
}
{
sub.f16x2 r2592, r2554, r2563;
}
{
add.f16x2 r2595, r2596, r2597;
}
{
add.f16x2 r2598, r2599, r2600;
}
{
sub.f16x2 r2601, r2596, r2597;
}
{
sub.f16x2 r2604, r2599, r2600;
}
{
add.f16x2 r2607, r2608, r2609;
}
{
add.f16x2 r2610, r2611, r2612;
}
{
sub.f16x2 r2613, r2608, r2609;
}
{
sub.f16x2 r2616, r2611, r2612;
}
{
neg.f16x2 r2619, r2616;
}
{
add.f16x2 r2621, r2595, r2607;
}
{
add.f16x2 r2624, r2598, r2610;
}
{
sub.f16x2 r2627, r2595, r2607;
}
{
sub.f16x2 r2630, r2598, r2610;
}
{
add.f16x2 r2633, r2601, r2619;
}
{
add.f16x2 r2636, r2604, r2613;
}
{
sub.f16x2 r2639, r2601, r2619;
}
{
sub.f16x2 r2642, r2604, r2613;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r2645, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r2646, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f378;
cvt.rn.f16.f32 high, f378;
mov.b32 r2649, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r2650, {low, high};
}
{
mul.f16x2 r2659, r2633, r2645;
}
{
mul.f16x2 r2662, r2636, r2646;
}
{
sub.f16x2 r2665, r2659, r2662;
}
{
mul.f16x2 r2668, r2633, r2646;
}
{
fma.rn.f16x2 r2671, r2636, r2645, r2668;
}
{
neg.f16x2 r2675, r2630;
}
{
mul.f16x2 r2677, r2639, r2649;
}
{
mul.f16x2 r2680, r2642, r2650;
}
{
sub.f16x2 r2683, r2677, r2680;
}
{
mul.f16x2 r2686, r2639, r2650;
}
{
fma.rn.f16x2 r2689, r2642, r2649, r2686;
}
{
add.f16x2 r2693, r2571, r2621;
}
{
add.f16x2 r2696, r2574, r2624;
}
{
sub.f16x2 r2699, r2571, r2621;
}
{
sub.f16x2 r2702, r2574, r2624;
}
{
add.f16x2 r2705, r2583, r2665;
}
{
add.f16x2 r2708, r2586, r2671;
}
{
sub.f16x2 r2711, r2583, r2665;
}
{
sub.f16x2 r2714, r2586, r2671;
}
{
add.f16x2 r2717, r2577, r2675;
}
{
add.f16x2 r2720, r2580, r2627;
}
{
sub.f16x2 r2723, r2577, r2675;
}
{
sub.f16x2 r2726, r2580, r2627;
}
{
add.f16x2 r2729, r2589, r2683;
}
{
add.f16x2 r2732, r2592, r2689;
}
{
sub.f16x2 r2735, r2589, r2683;
}
{
sub.f16x2 r2738, r2592, r2689;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r2741, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r2742, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r2743, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r2744, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r2745, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r2746, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f374;
cvt.rn.f16.f32 high, f374;
mov.b32 r2749, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r2750, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f378;
cvt.rn.f16.f32 high, f378;
mov.b32 r2751, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r2752, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f382;
cvt.rn.f16.f32 high, f382;
mov.b32 r2753, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r2754, {low, high};
}
{
mul.f16x2 r2771, r2705, r2741;
}
{
mul.f16x2 r2774, r2708, r2742;
}
{
sub.f16x2 r2777, r2771, r2774;
}
{
mul.f16x2 r2780, r2705, r2742;
}
{
fma.rn.f16x2 r2783, r2708, r2741, r2780;
}
{
mul.f16x2 r2787, r2717, r2743;
}
{
mul.f16x2 r2790, r2720, r2744;
}
{
sub.f16x2 r2793, r2787, r2790;
}
{
mul.f16x2 r2796, r2717, r2744;
}
{
fma.rn.f16x2 r2799, r2720, r2743, r2796;
}
{
mul.f16x2 r2803, r2729, r2745;
}
{
mul.f16x2 r2806, r2732, r2746;
}
{
sub.f16x2 r2809, r2803, r2806;
}
{
mul.f16x2 r2812, r2729, r2746;
}
{
fma.rn.f16x2 r2815, r2732, r2745, r2812;
}
{
neg.f16x2 r2819, r2702;
}
{
mul.f16x2 r2821, r2711, r2749;
}
{
mul.f16x2 r2824, r2714, r2750;
}
{
sub.f16x2 r2827, r2821, r2824;
}
{
mul.f16x2 r2830, r2711, r2750;
}
{
fma.rn.f16x2 r2833, r2714, r2749, r2830;
}
{
mul.f16x2 r2837, r2723, r2751;
}
{
mul.f16x2 r2840, r2726, r2752;
}
{
sub.f16x2 r2843, r2837, r2840;
}
{
mul.f16x2 r2846, r2723, r2752;
}
{
fma.rn.f16x2 r2849, r2726, r2751, r2846;
}
{
mul.f16x2 r2853, r2735, r2753;
}
{
mul.f16x2 r2856, r2738, r2754;
}
{
sub.f16x2 r2859, r2853, r2856;
}
{
mul.f16x2 r2862, r2735, r2754;
}
{
fma.rn.f16x2 r2865, r2738, r2753, r2862;
}
{
add.f16x2 r2869, r2497, r2693;
}
{
add.f16x2 r2872, r2500, r2696;
}
{
sub.f16x2 r2875, r2497, r2693;
}
{
sub.f16x2 r2878, r2500, r2696;
}
{
add.f16x2 r2881, r2509, r2777;
}
{
add.f16x2 r2884, r2512, r2783;
}
{
sub.f16x2 r2887, r2509, r2777;
}
{
sub.f16x2 r2890, r2512, r2783;
}
{
add.f16x2 r2893, r2521, r2793;
}
{
add.f16x2 r2896, r2524, r2799;
}
{
sub.f16x2 r2899, r2521, r2793;
}
{
sub.f16x2 r2902, r2524, r2799;
}
{
add.f16x2 r2905, r2533, r2809;
}
{
add.f16x2 r2908, r2536, r2815;
}
{
sub.f16x2 r2911, r2533, r2809;
}
{
sub.f16x2 r2914, r2536, r2815;
}
{
add.f16x2 r2917, r2503, r2819;
}
{
add.f16x2 r2920, r2506, r2699;
}
{
sub.f16x2 r2923, r2503, r2819;
}
{
sub.f16x2 r2926, r2506, r2699;
}
{
add.f16x2 r2929, r2515, r2827;
}
{
add.f16x2 r2932, r2518, r2833;
}
{
sub.f16x2 r2935, r2515, r2827;
}
{
sub.f16x2 r2938, r2518, r2833;
}
{
add.f16x2 r2941, r2527, r2843;
}
{
add.f16x2 r2944, r2530, r2849;
}
{
sub.f16x2 r2947, r2527, r2843;
}
{
sub.f16x2 r2950, r2530, r2849;
}
{
add.f16x2 r2953, r2539, r2859;
}
{
add.f16x2 r2956, r2542, r2865;
}
{
sub.f16x2 r2959, r2539, r2859;
}
{
sub.f16x2 r2962, r2542, r2865;
}
and.b32 r3743, r3727, 768;
bfe.u32 r3744, r3727, 8, 2;
cvt.rn.f32.u32 f457, r3744;
mul.f32 f458, f457, 0f3DC90FDB;
cos.approx.f32 f417, f458;
sin.approx.f32 f459, f458;
neg.f32 f418, f459;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f417;
cvt.rn.f16.f32 high, f418;
mov.b32 r2965, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r2968, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r2970, {high, high};
}
{
mul.f16x2 r2972, r2884, r2970;
}
{
fma.rn.f16x2 r2975, r2881, r2968, r2972;
}
{
mul.f16x2 r2979, r2881, r2970;
}
{
neg.f16x2 r2982, r2979;
}
{
fma.rn.f16x2 r2984, r2884, r2968, r2982;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r2988, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r2990, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r2992, {low, high};
}
{
mul.f16x2 r2993, r2990, r2992;
}
{
mul.f16x2 r2996, r2965, r2988;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r2999, {high, low};
}
{
fma.rn.f16x2 r3001, r2993, r2999, r2996;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3001;
mov.b32 r3005, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3001;
mov.b32 r3007, {high, high};
}
{
mul.f16x2 r3009, r2896, r3007;
}
{
fma.rn.f16x2 r3012, r2893, r3005, r3009;
}
{
mul.f16x2 r3016, r2893, r3007;
}
{
neg.f16x2 r3019, r3016;
}
{
fma.rn.f16x2 r3021, r2896, r3005, r3019;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3025, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3027, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3029, {low, high};
}
{
mul.f16x2 r3030, r3027, r3029;
}
{
mul.f16x2 r3033, r3001, r3025;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3001;
mov.b32 r3036, {high, low};
}
{
fma.rn.f16x2 r3038, r3030, r3036, r3033;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3038;
mov.b32 r3042, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3038;
mov.b32 r3044, {high, high};
}
{
mul.f16x2 r3046, r2908, r3044;
}
{
fma.rn.f16x2 r3049, r2905, r3042, r3046;
}
{
mul.f16x2 r3053, r2905, r3044;
}
{
neg.f16x2 r3056, r3053;
}
{
fma.rn.f16x2 r3058, r2908, r3042, r3056;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3062, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3064, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3066, {low, high};
}
{
mul.f16x2 r3067, r3064, r3066;
}
{
mul.f16x2 r3070, r3038, r3062;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3038;
mov.b32 r3073, {high, low};
}
{
fma.rn.f16x2 r3075, r3067, r3073, r3070;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3075;
mov.b32 r3079, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3075;
mov.b32 r3081, {high, high};
}
{
mul.f16x2 r3083, r2920, r3081;
}
{
fma.rn.f16x2 r3086, r2917, r3079, r3083;
}
{
mul.f16x2 r3090, r2917, r3081;
}
{
neg.f16x2 r3093, r3090;
}
{
fma.rn.f16x2 r3095, r2920, r3079, r3093;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3099, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3101, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3103, {low, high};
}
{
mul.f16x2 r3104, r3101, r3103;
}
{
mul.f16x2 r3107, r3075, r3099;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3075;
mov.b32 r3110, {high, low};
}
{
fma.rn.f16x2 r3112, r3104, r3110, r3107;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3112;
mov.b32 r3116, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3112;
mov.b32 r3118, {high, high};
}
{
mul.f16x2 r3120, r2932, r3118;
}
{
fma.rn.f16x2 r3123, r2929, r3116, r3120;
}
{
mul.f16x2 r3127, r2929, r3118;
}
{
neg.f16x2 r3130, r3127;
}
{
fma.rn.f16x2 r3132, r2932, r3116, r3130;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3136, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3138, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3140, {low, high};
}
{
mul.f16x2 r3141, r3138, r3140;
}
{
mul.f16x2 r3144, r3112, r3136;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3112;
mov.b32 r3147, {high, low};
}
{
fma.rn.f16x2 r3149, r3141, r3147, r3144;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3149;
mov.b32 r3153, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3149;
mov.b32 r3155, {high, high};
}
{
mul.f16x2 r3157, r2944, r3155;
}
{
fma.rn.f16x2 r3160, r2941, r3153, r3157;
}
{
mul.f16x2 r3164, r2941, r3155;
}
{
neg.f16x2 r3167, r3164;
}
{
fma.rn.f16x2 r3169, r2944, r3153, r3167;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3173, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3175, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3177, {low, high};
}
{
mul.f16x2 r3178, r3175, r3177;
}
{
mul.f16x2 r3181, r3149, r3173;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3149;
mov.b32 r3184, {high, low};
}
{
fma.rn.f16x2 r3186, r3178, r3184, r3181;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3186;
mov.b32 r3190, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3186;
mov.b32 r3192, {high, high};
}
{
mul.f16x2 r3194, r2956, r3192;
}
{
fma.rn.f16x2 r3197, r2953, r3190, r3194;
}
{
mul.f16x2 r3201, r2953, r3192;
}
{
neg.f16x2 r3204, r3201;
}
{
fma.rn.f16x2 r3206, r2956, r3190, r3204;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3210, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3212, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3214, {low, high};
}
{
mul.f16x2 r3215, r3212, r3214;
}
{
mul.f16x2 r3218, r3186, r3210;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3186;
mov.b32 r3221, {high, low};
}
{
fma.rn.f16x2 r3223, r3215, r3221, r3218;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3223;
mov.b32 r3227, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3223;
mov.b32 r3229, {high, high};
}
{
mul.f16x2 r3231, r2878, r3229;
}
{
fma.rn.f16x2 r3234, r2875, r3227, r3231;
}
{
mul.f16x2 r3238, r2875, r3229;
}
{
neg.f16x2 r3241, r3238;
}
{
fma.rn.f16x2 r3243, r2878, r3227, r3241;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3247, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3249, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3251, {low, high};
}
{
mul.f16x2 r3252, r3249, r3251;
}
{
mul.f16x2 r3255, r3223, r3247;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3223;
mov.b32 r3258, {high, low};
}
{
fma.rn.f16x2 r3260, r3252, r3258, r3255;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3260;
mov.b32 r3264, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3260;
mov.b32 r3266, {high, high};
}
{
mul.f16x2 r3268, r2890, r3266;
}
{
fma.rn.f16x2 r3271, r2887, r3264, r3268;
}
{
mul.f16x2 r3275, r2887, r3266;
}
{
neg.f16x2 r3278, r3275;
}
{
fma.rn.f16x2 r3280, r2890, r3264, r3278;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3284, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3286, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3288, {low, high};
}
{
mul.f16x2 r3289, r3286, r3288;
}
{
mul.f16x2 r3292, r3260, r3284;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3260;
mov.b32 r3295, {high, low};
}
{
fma.rn.f16x2 r3297, r3289, r3295, r3292;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3297;
mov.b32 r3301, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3297;
mov.b32 r3303, {high, high};
}
{
mul.f16x2 r3305, r2902, r3303;
}
{
fma.rn.f16x2 r3308, r2899, r3301, r3305;
}
{
mul.f16x2 r3312, r2899, r3303;
}
{
neg.f16x2 r3315, r3312;
}
{
fma.rn.f16x2 r3317, r2902, r3301, r3315;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3321, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3323, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3325, {low, high};
}
{
mul.f16x2 r3326, r3323, r3325;
}
{
mul.f16x2 r3329, r3297, r3321;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3297;
mov.b32 r3332, {high, low};
}
{
fma.rn.f16x2 r3334, r3326, r3332, r3329;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3334;
mov.b32 r3338, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3334;
mov.b32 r3340, {high, high};
}
{
mul.f16x2 r3342, r2914, r3340;
}
{
fma.rn.f16x2 r3345, r2911, r3338, r3342;
}
{
mul.f16x2 r3349, r2911, r3340;
}
{
neg.f16x2 r3352, r3349;
}
{
fma.rn.f16x2 r3354, r2914, r3338, r3352;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3358, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3360, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3362, {low, high};
}
{
mul.f16x2 r3363, r3360, r3362;
}
{
mul.f16x2 r3366, r3334, r3358;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3334;
mov.b32 r3369, {high, low};
}
{
fma.rn.f16x2 r3371, r3363, r3369, r3366;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3371;
mov.b32 r3375, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3371;
mov.b32 r3377, {high, high};
}
{
mul.f16x2 r3379, r2926, r3377;
}
{
fma.rn.f16x2 r3382, r2923, r3375, r3379;
}
{
mul.f16x2 r3386, r2923, r3377;
}
{
neg.f16x2 r3389, r3386;
}
{
fma.rn.f16x2 r3391, r2926, r3375, r3389;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3395, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3397, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3399, {low, high};
}
{
mul.f16x2 r3400, r3397, r3399;
}
{
mul.f16x2 r3403, r3371, r3395;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3371;
mov.b32 r3406, {high, low};
}
{
fma.rn.f16x2 r3408, r3400, r3406, r3403;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3408;
mov.b32 r3412, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3408;
mov.b32 r3414, {high, high};
}
{
mul.f16x2 r3416, r2938, r3414;
}
{
fma.rn.f16x2 r3419, r2935, r3412, r3416;
}
{
mul.f16x2 r3423, r2935, r3414;
}
{
neg.f16x2 r3426, r3423;
}
{
fma.rn.f16x2 r3428, r2938, r3412, r3426;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3432, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3434, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3436, {low, high};
}
{
mul.f16x2 r3437, r3434, r3436;
}
{
mul.f16x2 r3440, r3408, r3432;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3408;
mov.b32 r3443, {high, low};
}
{
fma.rn.f16x2 r3445, r3437, r3443, r3440;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3445;
mov.b32 r3449, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3445;
mov.b32 r3451, {high, high};
}
{
mul.f16x2 r3453, r2950, r3451;
}
{
fma.rn.f16x2 r3456, r2947, r3449, r3453;
}
{
mul.f16x2 r3460, r2947, r3451;
}
{
neg.f16x2 r3463, r3460;
}
{
fma.rn.f16x2 r3465, r2950, r3449, r3463;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3469, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2965;
mov.b32 r3471, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f447;
cvt.rn.f16.f32 high, f448;
mov.b32 r3473, {low, high};
}
{
mul.f16x2 r3474, r3471, r3473;
}
{
mul.f16x2 r3477, r3445, r3469;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3445;
mov.b32 r3480, {high, low};
}
{
fma.rn.f16x2 r3482, r3474, r3480, r3477;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3482;
mov.b32 r3486, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3482;
mov.b32 r3488, {high, high};
}
{
mul.f16x2 r3490, r2962, r3488;
}
{
fma.rn.f16x2 r3493, r2959, r3486, r3490;
}
{
mul.f16x2 r3497, r2959, r3488;
}
{
neg.f16x2 r3500, r3497;
}
{
fma.rn.f16x2 r3502, r2962, r3486, r3500;
}
and.b32 r3745, r3737, 2040;
add.s32 r3746, r3731, r3745;
barrier.sync 0;
and.b32 r3747, r3729, 98304;
add.s32 r3748, r3746, r3747;
st.shared.u32 [r3748], r2869;
st.shared.u32 [r3748+4], r2872;
st.shared.u32 [r3748+2048], r2975;
st.shared.u32 [r3748+2052], r2984;
st.shared.u32 [r3748+4096], r3012;
st.shared.u32 [r3748+4100], r3021;
st.shared.u32 [r3748+6144], r3049;
st.shared.u32 [r3748+6148], r3058;
st.shared.u32 [r3748+8192], r3086;
st.shared.u32 [r3748+8196], r3095;
st.shared.u32 [r3748+10240], r3123;
st.shared.u32 [r3748+10244], r3132;
st.shared.u32 [r3748+12288], r3160;
st.shared.u32 [r3748+12292], r3169;
st.shared.u32 [r3748+14336], r3197;
st.shared.u32 [r3748+14340], r3206;
st.shared.u32 [r3748+16384], r3234;
st.shared.u32 [r3748+16388], r3243;
st.shared.u32 [r3748+18432], r3271;
st.shared.u32 [r3748+18436], r3280;
st.shared.u32 [r3748+20480], r3308;
st.shared.u32 [r3748+20484], r3317;
st.shared.u32 [r3748+22528], r3345;
st.shared.u32 [r3748+22532], r3354;
st.shared.u32 [r3748+24576], r3382;
st.shared.u32 [r3748+24580], r3391;
st.shared.u32 [r3748+26624], r3419;
st.shared.u32 [r3748+26628], r3428;
st.shared.u32 [r3748+28672], r3456;
st.shared.u32 [r3748+28676], r3465;
st.shared.u32 [r3748+30720], r3493;
st.shared.u32 [r3748+30724], r3502;
barrier.sync 0;
mad.lo.s32 r3749, r3743, -120, r3748;
ld.shared.u32 r3524, [r3749];
ld.shared.u32 r3527, [r3749+4];
ld.shared.u32 r3574, [r3749+8192];
ld.shared.u32 r3577, [r3749+8196];
ld.shared.u32 r3624, [r3749+16384];
ld.shared.u32 r3627, [r3749+16388];
ld.shared.u32 r3674, [r3749+24576];
ld.shared.u32 r3677, [r3749+24580];
ld.shared.u32 r3536, [r3749+32768];
ld.shared.u32 r3539, [r3749+32772];
ld.shared.u32 r3586, [r3749+40960];
ld.shared.u32 r3589, [r3749+40964];
ld.shared.u32 r3636, [r3749+49152];
ld.shared.u32 r3639, [r3749+49156];
ld.shared.u32 r3686, [r3749+57344];
ld.shared.u32 r3689, [r3749+57348];
ld.shared.u32 r3525, [r3749+65536];
ld.shared.u32 r3528, [r3749+65540];
ld.shared.u32 r3575, [r3749+73728];
ld.shared.u32 r3578, [r3749+73732];
ld.shared.u32 r3625, [r3749+81920];
ld.shared.u32 r3628, [r3749+81924];
ld.shared.u32 r3675, [r3749+90112];
ld.shared.u32 r3678, [r3749+90116];
ld.shared.u32 r3537, [r3749+98304];
ld.shared.u32 r3540, [r3749+98308];
ld.shared.u32 r3587, [r3749+106496];
ld.shared.u32 r3590, [r3749+106500];
ld.shared.u32 r3637, [r3749+114688];
ld.shared.u32 r3640, [r3749+114692];
ld.shared.u32 r3687, [r3749+122880];
ld.shared.u32 r3690, [r3749+122884];
{
add.f16x2 r3523, r3524, r3525;
}
{
add.f16x2 r3526, r3527, r3528;
}
{
sub.f16x2 r3529, r3524, r3525;
}
{
sub.f16x2 r3532, r3527, r3528;
}
{
add.f16x2 r3535, r3536, r3537;
}
{
add.f16x2 r3538, r3539, r3540;
}
{
sub.f16x2 r3541, r3536, r3537;
}
{
sub.f16x2 r3544, r3539, r3540;
}
{
neg.f16x2 r3547, r3544;
}
{
add.f16x2 %0, r3523, r3535;
}
{
add.f16x2 %1, r3526, r3538;
}
{
sub.f16x2 %16, r3523, r3535;
}
{
sub.f16x2 %17, r3526, r3538;
}
{
add.f16x2 %8, r3529, r3547;
}
{
add.f16x2 %9, r3532, r3541;
}
{
sub.f16x2 %24, r3529, r3547;
}
{
sub.f16x2 %25, r3532, r3541;
}
{
add.f16x2 r3573, r3574, r3575;
}
{
add.f16x2 r3576, r3577, r3578;
}
{
sub.f16x2 r3579, r3574, r3575;
}
{
sub.f16x2 r3582, r3577, r3578;
}
{
add.f16x2 r3585, r3586, r3587;
}
{
add.f16x2 r3588, r3589, r3590;
}
{
sub.f16x2 r3591, r3586, r3587;
}
{
sub.f16x2 r3594, r3589, r3590;
}
{
neg.f16x2 r3597, r3594;
}
{
add.f16x2 %2, r3573, r3585;
}
{
add.f16x2 %3, r3576, r3588;
}
{
sub.f16x2 %18, r3573, r3585;
}
{
sub.f16x2 %19, r3576, r3588;
}
{
add.f16x2 %10, r3579, r3597;
}
{
add.f16x2 %11, r3582, r3591;
}
{
sub.f16x2 %26, r3579, r3597;
}
{
sub.f16x2 %27, r3582, r3591;
}
{
add.f16x2 r3623, r3624, r3625;
}
{
add.f16x2 r3626, r3627, r3628;
}
{
sub.f16x2 r3629, r3624, r3625;
}
{
sub.f16x2 r3632, r3627, r3628;
}
{
add.f16x2 r3635, r3636, r3637;
}
{
add.f16x2 r3638, r3639, r3640;
}
{
sub.f16x2 r3641, r3636, r3637;
}
{
sub.f16x2 r3644, r3639, r3640;
}
{
neg.f16x2 r3647, r3644;
}
{
add.f16x2 %4, r3623, r3635;
}
{
add.f16x2 %5, r3626, r3638;
}
{
sub.f16x2 %20, r3623, r3635;
}
{
sub.f16x2 %21, r3626, r3638;
}
{
add.f16x2 %12, r3629, r3647;
}
{
add.f16x2 %13, r3632, r3641;
}
{
sub.f16x2 %28, r3629, r3647;
}
{
sub.f16x2 %29, r3632, r3641;
}
{
add.f16x2 r3673, r3674, r3675;
}
{
add.f16x2 r3676, r3677, r3678;
}
{
sub.f16x2 r3679, r3674, r3675;
}
{
sub.f16x2 r3682, r3677, r3678;
}
{
add.f16x2 r3685, r3686, r3687;
}
{
add.f16x2 r3688, r3689, r3690;
}
{
sub.f16x2 r3691, r3686, r3687;
}
{
sub.f16x2 r3694, r3689, r3690;
}
{
neg.f16x2 r3697, r3694;
}
{
add.f16x2 %6, r3673, r3685;
}
{
add.f16x2 %7, r3676, r3688;
}
{
sub.f16x2 %22, r3673, r3685;
}
{
sub.f16x2 %23, r3676, r3688;
}
{
add.f16x2 %14, r3679, r3697;
}
{
add.f16x2 %15, r3682, r3691;
}
{
sub.f16x2 %30, r3679, r3697;
}
{
sub.f16x2 %31, r3682, r3691;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1064, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<1085>;
.reg .b32 r<7042>;
.reg .b64 rd<3>;
mov.u32 r6957, %tid.y;
shl.b32 r6958, r6957, 16;
mov.u32 r6959, %64;
add.s32 r6960, r6959, r6958;
mov.u32 r6961, %tid.x;
{
add.f16x2 r1, %119, %111;
}
{
add.f16x2 r4, %91, %81;
}
{
sub.f16x2 r7, %119, %111;
}
{
sub.f16x2 r10, %91, %81;
}
{
add.f16x2 r13, %73, %128;
}
{
add.f16x2 r16, %106, %100;
}
{
sub.f16x2 r19, %73, %128;
}
{
sub.f16x2 r22, %106, %100;
}
{
neg.f16x2 r25, r22;
}
{
add.f16x2 r27, r1, r13;
}
{
add.f16x2 r30, r4, r16;
}
{
sub.f16x2 r33, r1, r13;
}
{
sub.f16x2 r36, r4, r16;
}
{
add.f16x2 r39, r7, r25;
}
{
add.f16x2 r42, r10, r19;
}
{
sub.f16x2 r45, r7, r25;
}
{
sub.f16x2 r48, r10, r19;
}
{
add.f16x2 r51, %105, %96;
}
{
add.f16x2 r54, %77, %67;
}
{
sub.f16x2 r57, %105, %96;
}
{
sub.f16x2 r60, %77, %67;
}
{
add.f16x2 r63, %122, %115;
}
{
add.f16x2 r66, %93, %85;
}
{
sub.f16x2 r69, %122, %115;
}
{
sub.f16x2 r72, %93, %85;
}
{
neg.f16x2 r75, r72;
}
{
add.f16x2 r77, r51, r63;
}
{
add.f16x2 r80, r54, r66;
}
{
sub.f16x2 r83, r51, r63;
}
{
sub.f16x2 r86, r54, r66;
}
{
add.f16x2 r89, r57, r75;
}
{
add.f16x2 r92, r60, r69;
}
{
sub.f16x2 r95, r57, r75;
}
{
sub.f16x2 r98, r60, r69;
}
mov.f32 f1040, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r102, {low, high};
}
mov.f32 f1038, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r106, {low, high};
}
{
mul.f16x2 r115, r89, r101;
}
{
mul.f16x2 r118, r92, r102;
}
{
sub.f16x2 r121, r115, r118;
}
{
mul.f16x2 r124, r89, r102;
}
{
fma.rn.f16x2 r127, r92, r101, r124;
}
{
neg.f16x2 r131, r86;
}
{
mul.f16x2 r133, r95, r105;
}
{
mul.f16x2 r136, r98, r106;
}
{
sub.f16x2 r139, r133, r136;
}
{
mul.f16x2 r142, r95, r106;
}
{
fma.rn.f16x2 r145, r98, r105, r142;
}
{
add.f16x2 r149, r27, r77;
}
{
add.f16x2 r152, r30, r80;
}
{
sub.f16x2 r155, r27, r77;
}
{
sub.f16x2 r158, r30, r80;
}
{
add.f16x2 r161, r39, r121;
}
{
add.f16x2 r164, r42, r127;
}
{
sub.f16x2 r167, r39, r121;
}
{
sub.f16x2 r170, r42, r127;
}
{
add.f16x2 r173, r33, r131;
}
{
add.f16x2 r176, r36, r83;
}
{
sub.f16x2 r179, r33, r131;
}
{
sub.f16x2 r182, r36, r83;
}
{
add.f16x2 r185, r45, r139;
}
{
add.f16x2 r188, r48, r145;
}
{
sub.f16x2 r191, r45, r139;
}
{
sub.f16x2 r194, r48, r145;
}
{
add.f16x2 r197, %94, %87;
}
{
add.f16x2 r200, %66, %123;
}
{
sub.f16x2 r203, %94, %87;
}
{
sub.f16x2 r206, %66, %123;
}
{
add.f16x2 r209, %113, %103;
}
{
add.f16x2 r212, %83, %75;
}
{
sub.f16x2 r215, %113, %103;
}
{
sub.f16x2 r218, %83, %75;
}
{
neg.f16x2 r221, r218;
}
{
add.f16x2 r223, r197, r209;
}
{
add.f16x2 r226, r200, r212;
}
{
sub.f16x2 r229, r197, r209;
}
{
sub.f16x2 r232, r200, r212;
}
{
add.f16x2 r235, r203, r221;
}
{
add.f16x2 r238, r206, r215;
}
{
sub.f16x2 r241, r203, r221;
}
{
sub.f16x2 r244, r206, r215;
}
{
add.f16x2 r247, %78, %72;
}
{
add.f16x2 r250, %117, %108;
}
{
sub.f16x2 r253, %78, %72;
}
{
sub.f16x2 r256, %117, %108;
}
{
add.f16x2 r259, %97, %89;
}
{
add.f16x2 r262, %69, %125;
}
{
sub.f16x2 r265, %97, %89;
}
{
sub.f16x2 r268, %69, %125;
}
{
neg.f16x2 r271, r268;
}
{
add.f16x2 r273, r247, r259;
}
{
add.f16x2 r276, r250, r262;
}
{
sub.f16x2 r279, r247, r259;
}
{
sub.f16x2 r282, r250, r262;
}
{
add.f16x2 r285, r253, r271;
}
{
add.f16x2 r288, r256, r265;
}
{
sub.f16x2 r291, r253, r271;
}
{
sub.f16x2 r294, r256, r265;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r302, {low, high};
}
{
mul.f16x2 r311, r285, r297;
}
{
mul.f16x2 r314, r288, r298;
}
{
sub.f16x2 r317, r311, r314;
}
{
mul.f16x2 r320, r285, r298;
}
{
fma.rn.f16x2 r323, r288, r297, r320;
}
{
neg.f16x2 r327, r282;
}
{
mul.f16x2 r329, r291, r301;
}
{
mul.f16x2 r332, r294, r302;
}
{
sub.f16x2 r335, r329, r332;
}
{
mul.f16x2 r338, r291, r302;
}
{
fma.rn.f16x2 r341, r294, r301, r338;
}
{
add.f16x2 r345, r223, r273;
}
{
add.f16x2 r348, r226, r276;
}
{
sub.f16x2 r351, r223, r273;
}
{
sub.f16x2 r354, r226, r276;
}
{
add.f16x2 r357, r235, r317;
}
{
add.f16x2 r360, r238, r323;
}
{
sub.f16x2 r363, r235, r317;
}
{
sub.f16x2 r366, r238, r323;
}
{
add.f16x2 r369, r229, r327;
}
{
add.f16x2 r372, r232, r279;
}
{
sub.f16x2 r375, r229, r327;
}
{
sub.f16x2 r378, r232, r279;
}
{
add.f16x2 r381, r241, r335;
}
{
add.f16x2 r384, r244, r341;
}
{
sub.f16x2 r387, r241, r335;
}
{
sub.f16x2 r390, r244, r341;
}
mov.f32 f1036, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r393, {low, high};
}
mov.f32 f1044, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r396, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r398, {low, high};
}
mov.f32 f1034, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r404, {low, high};
}
mov.f32 f1042, 0fBF6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1042;
cvt.rn.f16.f32 high, f1042;
mov.b32 r405, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r406, {low, high};
}
{
mul.f16x2 r423, r357, r393;
}
{
mul.f16x2 r426, r360, r394;
}
{
sub.f16x2 r429, r423, r426;
}
{
mul.f16x2 r432, r357, r394;
}
{
fma.rn.f16x2 r435, r360, r393, r432;
}
{
mul.f16x2 r439, r369, r395;
}
{
mul.f16x2 r442, r372, r396;
}
{
sub.f16x2 r445, r439, r442;
}
{
mul.f16x2 r448, r369, r396;
}
{
fma.rn.f16x2 r451, r372, r395, r448;
}
{
mul.f16x2 r455, r381, r397;
}
{
mul.f16x2 r458, r384, r398;
}
{
sub.f16x2 r461, r455, r458;
}
{
mul.f16x2 r464, r381, r398;
}
{
fma.rn.f16x2 r467, r384, r397, r464;
}
{
neg.f16x2 r471, r354;
}
{
mul.f16x2 r473, r363, r401;
}
{
mul.f16x2 r476, r366, r402;
}
{
sub.f16x2 r479, r473, r476;
}
{
mul.f16x2 r482, r363, r402;
}
{
fma.rn.f16x2 r485, r366, r401, r482;
}
{
mul.f16x2 r489, r375, r403;
}
{
mul.f16x2 r492, r378, r404;
}
{
sub.f16x2 r495, r489, r492;
}
{
mul.f16x2 r498, r375, r404;
}
{
fma.rn.f16x2 r501, r378, r403, r498;
}
{
mul.f16x2 r505, r387, r405;
}
{
mul.f16x2 r508, r390, r406;
}
{
sub.f16x2 r511, r505, r508;
}
{
mul.f16x2 r514, r387, r406;
}
{
fma.rn.f16x2 r517, r390, r405, r514;
}
{
add.f16x2 r521, r149, r345;
}
{
add.f16x2 r524, r152, r348;
}
{
sub.f16x2 r527, r149, r345;
}
{
sub.f16x2 r530, r152, r348;
}
{
add.f16x2 r533, r161, r429;
}
{
add.f16x2 r536, r164, r435;
}
{
sub.f16x2 r539, r161, r429;
}
{
sub.f16x2 r542, r164, r435;
}
{
add.f16x2 r545, r173, r445;
}
{
add.f16x2 r548, r176, r451;
}
{
sub.f16x2 r551, r173, r445;
}
{
sub.f16x2 r554, r176, r451;
}
{
add.f16x2 r557, r185, r461;
}
{
add.f16x2 r560, r188, r467;
}
{
sub.f16x2 r563, r185, r461;
}
{
sub.f16x2 r566, r188, r467;
}
{
add.f16x2 r569, r155, r471;
}
{
add.f16x2 r572, r158, r351;
}
{
sub.f16x2 r575, r155, r471;
}
{
sub.f16x2 r578, r158, r351;
}
{
add.f16x2 r581, r167, r479;
}
{
add.f16x2 r584, r170, r485;
}
{
sub.f16x2 r587, r167, r479;
}
{
sub.f16x2 r590, r170, r485;
}
{
add.f16x2 r593, r179, r495;
}
{
add.f16x2 r596, r182, r501;
}
{
sub.f16x2 r599, r179, r495;
}
{
sub.f16x2 r602, r182, r501;
}
{
add.f16x2 r605, r191, r511;
}
{
add.f16x2 r608, r194, r517;
}
{
sub.f16x2 r611, r191, r511;
}
{
sub.f16x2 r614, r194, r517;
}
{
add.f16x2 r617, %68, %124;
}
{
add.f16x2 r620, %104, %95;
}
{
sub.f16x2 r623, %68, %124;
}
{
sub.f16x2 r626, %104, %95;
}
{
add.f16x2 r629, %86, %76;
}
{
add.f16x2 r632, %121, %114;
}
{
sub.f16x2 r635, %86, %76;
}
{
sub.f16x2 r638, %121, %114;
}
{
neg.f16x2 r641, r638;
}
{
add.f16x2 r643, r617, r629;
}
{
add.f16x2 r646, r620, r632;
}
{
sub.f16x2 r649, r617, r629;
}
{
sub.f16x2 r652, r620, r632;
}
{
add.f16x2 r655, r623, r641;
}
{
add.f16x2 r658, r626, r635;
}
{
sub.f16x2 r661, r623, r641;
}
{
sub.f16x2 r664, r626, r635;
}
{
add.f16x2 r667, %118, %110;
}
{
add.f16x2 r670, %90, %80;
}
{
sub.f16x2 r673, %118, %110;
}
{
sub.f16x2 r676, %90, %80;
}
{
add.f16x2 r679, %70, %127;
}
{
add.f16x2 r682, %107, %99;
}
{
sub.f16x2 r685, %70, %127;
}
{
sub.f16x2 r688, %107, %99;
}
{
neg.f16x2 r691, r688;
}
{
add.f16x2 r693, r667, r679;
}
{
add.f16x2 r696, r670, r682;
}
{
sub.f16x2 r699, r667, r679;
}
{
sub.f16x2 r702, r670, r682;
}
{
add.f16x2 r705, r673, r691;
}
{
add.f16x2 r708, r676, r685;
}
{
sub.f16x2 r711, r673, r691;
}
{
sub.f16x2 r714, r676, r685;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r717, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r718, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r721, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r722, {low, high};
}
{
mul.f16x2 r731, r705, r717;
}
{
mul.f16x2 r734, r708, r718;
}
{
sub.f16x2 r737, r731, r734;
}
{
mul.f16x2 r740, r705, r718;
}
{
fma.rn.f16x2 r743, r708, r717, r740;
}
{
neg.f16x2 r747, r702;
}
{
mul.f16x2 r749, r711, r721;
}
{
mul.f16x2 r752, r714, r722;
}
{
sub.f16x2 r755, r749, r752;
}
{
mul.f16x2 r758, r711, r722;
}
{
fma.rn.f16x2 r761, r714, r721, r758;
}
{
add.f16x2 r765, r643, r693;
}
{
add.f16x2 r768, r646, r696;
}
{
sub.f16x2 r771, r643, r693;
}
{
sub.f16x2 r774, r646, r696;
}
{
add.f16x2 r777, r655, r737;
}
{
add.f16x2 r780, r658, r743;
}
{
sub.f16x2 r783, r655, r737;
}
{
sub.f16x2 r786, r658, r743;
}
{
add.f16x2 r789, r649, r747;
}
{
add.f16x2 r792, r652, r699;
}
{
sub.f16x2 r795, r649, r747;
}
{
sub.f16x2 r798, r652, r699;
}
{
add.f16x2 r801, r661, r755;
}
{
add.f16x2 r804, r664, r761;
}
{
sub.f16x2 r807, r661, r755;
}
{
sub.f16x2 r810, r664, r761;
}
{
add.f16x2 r813, %109, %101;
}
{
add.f16x2 r816, %79, %71;
}
{
sub.f16x2 r819, %109, %101;
}
{
sub.f16x2 r822, %79, %71;
}
{
add.f16x2 r825, %126, %116;
}
{
add.f16x2 r828, %98, %88;
}
{
sub.f16x2 r831, %126, %116;
}
{
sub.f16x2 r834, %98, %88;
}
{
neg.f16x2 r837, r834;
}
{
add.f16x2 r839, r813, r825;
}
{
add.f16x2 r842, r816, r828;
}
{
sub.f16x2 r845, r813, r825;
}
{
sub.f16x2 r848, r816, r828;
}
{
add.f16x2 r851, r819, r837;
}
{
add.f16x2 r854, r822, r831;
}
{
sub.f16x2 r857, r819, r837;
}
{
sub.f16x2 r860, r822, r831;
}
{
add.f16x2 r863, %92, %84;
}
{
add.f16x2 r866, %65, %120;
}
{
sub.f16x2 r869, %92, %84;
}
{
sub.f16x2 r872, %65, %120;
}
{
add.f16x2 r875, %112, %102;
}
{
add.f16x2 r878, %82, %74;
}
{
sub.f16x2 r881, %112, %102;
}
{
sub.f16x2 r884, %82, %74;
}
{
neg.f16x2 r887, r884;
}
{
add.f16x2 r889, r863, r875;
}
{
add.f16x2 r892, r866, r878;
}
{
sub.f16x2 r895, r863, r875;
}
{
sub.f16x2 r898, r866, r878;
}
{
add.f16x2 r901, r869, r887;
}
{
add.f16x2 r904, r872, r881;
}
{
sub.f16x2 r907, r869, r887;
}
{
sub.f16x2 r910, r872, r881;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r913, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r914, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r917, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r918, {low, high};
}
{
mul.f16x2 r927, r901, r913;
}
{
mul.f16x2 r930, r904, r914;
}
{
sub.f16x2 r933, r927, r930;
}
{
mul.f16x2 r936, r901, r914;
}
{
fma.rn.f16x2 r939, r904, r913, r936;
}
{
neg.f16x2 r943, r898;
}
{
mul.f16x2 r945, r907, r917;
}
{
mul.f16x2 r948, r910, r918;
}
{
sub.f16x2 r951, r945, r948;
}
{
mul.f16x2 r954, r907, r918;
}
{
fma.rn.f16x2 r957, r910, r917, r954;
}
{
add.f16x2 r961, r839, r889;
}
{
add.f16x2 r964, r842, r892;
}
{
sub.f16x2 r967, r839, r889;
}
{
sub.f16x2 r970, r842, r892;
}
{
add.f16x2 r973, r851, r933;
}
{
add.f16x2 r976, r854, r939;
}
{
sub.f16x2 r979, r851, r933;
}
{
sub.f16x2 r982, r854, r939;
}
{
add.f16x2 r985, r845, r943;
}
{
add.f16x2 r988, r848, r895;
}
{
sub.f16x2 r991, r845, r943;
}
{
sub.f16x2 r994, r848, r895;
}
{
add.f16x2 r997, r857, r951;
}
{
add.f16x2 r1000, r860, r957;
}
{
sub.f16x2 r1003, r857, r951;
}
{
sub.f16x2 r1006, r860, r957;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r1009, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r1010, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r1011, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r1012, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r1013, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r1014, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1017, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r1018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r1019, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r1020, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1042;
cvt.rn.f16.f32 high, f1042;
mov.b32 r1021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r1022, {low, high};
}
{
mul.f16x2 r1039, r973, r1009;
}
{
mul.f16x2 r1042, r976, r1010;
}
{
sub.f16x2 r1045, r1039, r1042;
}
{
mul.f16x2 r1048, r973, r1010;
}
{
fma.rn.f16x2 r1051, r976, r1009, r1048;
}
{
mul.f16x2 r1055, r985, r1011;
}
{
mul.f16x2 r1058, r988, r1012;
}
{
sub.f16x2 r1061, r1055, r1058;
}
{
mul.f16x2 r1064, r985, r1012;
}
{
fma.rn.f16x2 r1067, r988, r1011, r1064;
}
{
mul.f16x2 r1071, r997, r1013;
}
{
mul.f16x2 r1074, r1000, r1014;
}
{
sub.f16x2 r1077, r1071, r1074;
}
{
mul.f16x2 r1080, r997, r1014;
}
{
fma.rn.f16x2 r1083, r1000, r1013, r1080;
}
{
neg.f16x2 r1087, r970;
}
{
mul.f16x2 r1089, r979, r1017;
}
{
mul.f16x2 r1092, r982, r1018;
}
{
sub.f16x2 r1095, r1089, r1092;
}
{
mul.f16x2 r1098, r979, r1018;
}
{
fma.rn.f16x2 r1101, r982, r1017, r1098;
}
{
mul.f16x2 r1105, r991, r1019;
}
{
mul.f16x2 r1108, r994, r1020;
}
{
sub.f16x2 r1111, r1105, r1108;
}
{
mul.f16x2 r1114, r991, r1020;
}
{
fma.rn.f16x2 r1117, r994, r1019, r1114;
}
{
mul.f16x2 r1121, r1003, r1021;
}
{
mul.f16x2 r1124, r1006, r1022;
}
{
sub.f16x2 r1127, r1121, r1124;
}
{
mul.f16x2 r1130, r1003, r1022;
}
{
fma.rn.f16x2 r1133, r1006, r1021, r1130;
}
{
add.f16x2 r1137, r765, r961;
}
{
add.f16x2 r1140, r768, r964;
}
{
sub.f16x2 r1143, r765, r961;
}
{
sub.f16x2 r1146, r768, r964;
}
{
add.f16x2 r1149, r777, r1045;
}
{
add.f16x2 r1152, r780, r1051;
}
{
sub.f16x2 r1155, r777, r1045;
}
{
sub.f16x2 r1158, r780, r1051;
}
{
add.f16x2 r1161, r789, r1061;
}
{
add.f16x2 r1164, r792, r1067;
}
{
sub.f16x2 r1167, r789, r1061;
}
{
sub.f16x2 r1170, r792, r1067;
}
{
add.f16x2 r1173, r801, r1077;
}
{
add.f16x2 r1176, r804, r1083;
}
{
sub.f16x2 r1179, r801, r1077;
}
{
sub.f16x2 r1182, r804, r1083;
}
{
add.f16x2 r1185, r771, r1087;
}
{
add.f16x2 r1188, r774, r967;
}
{
sub.f16x2 r1191, r771, r1087;
}
{
sub.f16x2 r1194, r774, r967;
}
{
add.f16x2 r1197, r783, r1095;
}
{
add.f16x2 r1200, r786, r1101;
}
{
sub.f16x2 r1203, r783, r1095;
}
{
sub.f16x2 r1206, r786, r1101;
}
{
add.f16x2 r1209, r795, r1111;
}
{
add.f16x2 r1212, r798, r1117;
}
{
sub.f16x2 r1215, r795, r1111;
}
{
sub.f16x2 r1218, r798, r1117;
}
{
add.f16x2 r1221, r807, r1127;
}
{
add.f16x2 r1224, r810, r1133;
}
{
sub.f16x2 r1227, r807, r1127;
}
{
sub.f16x2 r1230, r810, r1133;
}
mov.f32 f690, 0f3F7B14BE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f690;
cvt.rn.f16.f32 high, f690;
mov.b32 r1233, {low, high};
}
mov.f32 f714, 0f3E47C5C2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f714;
cvt.rn.f16.f32 high, f714;
mov.b32 r1234, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r1235, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r1236, {low, high};
}
mov.f32 f698, 0f3F54DB31;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f698;
cvt.rn.f16.f32 high, f698;
mov.b32 r1237, {low, high};
}
mov.f32 f706, 0f3F0E39DA;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f706;
cvt.rn.f16.f32 high, f706;
mov.b32 r1238, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r1239, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r1240, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f706;
cvt.rn.f16.f32 high, f706;
mov.b32 r1241, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f698;
cvt.rn.f16.f32 high, f698;
mov.b32 r1242, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r1243, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r1244, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f714;
cvt.rn.f16.f32 high, f714;
mov.b32 r1245, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f690;
cvt.rn.f16.f32 high, f690;
mov.b32 r1246, {low, high};
}
mov.f32 f688, 0fBE47C5C2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f688;
cvt.rn.f16.f32 high, f688;
mov.b32 r1249, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f690;
cvt.rn.f16.f32 high, f690;
mov.b32 r1250, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1251, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r1252, {low, high};
}
mov.f32 f696, 0fBF0E39DA;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f696;
cvt.rn.f16.f32 high, f696;
mov.b32 r1253, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f698;
cvt.rn.f16.f32 high, f698;
mov.b32 r1254, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r1255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r1256, {low, high};
}
mov.f32 f704, 0fBF54DB31;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f704;
cvt.rn.f16.f32 high, f704;
mov.b32 r1257, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f706;
cvt.rn.f16.f32 high, f706;
mov.b32 r1258, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1042;
cvt.rn.f16.f32 high, f1042;
mov.b32 r1259, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r1260, {low, high};
}
mov.f32 f712, 0fBF7B14BE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f712;
cvt.rn.f16.f32 high, f712;
mov.b32 r1261, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f714;
cvt.rn.f16.f32 high, f714;
mov.b32 r1262, {low, high};
}
{
mul.f16x2 r1295, r1149, r1233;
}
{
mul.f16x2 r1298, r1152, r1234;
}
{
sub.f16x2 r1301, r1295, r1298;
}
{
mul.f16x2 r1304, r1149, r1234;
}
{
fma.rn.f16x2 r1307, r1152, r1233, r1304;
}
{
mul.f16x2 r1311, r1161, r1235;
}
{
mul.f16x2 r1314, r1164, r1236;
}
{
sub.f16x2 r1317, r1311, r1314;
}
{
mul.f16x2 r1320, r1161, r1236;
}
{
fma.rn.f16x2 r1323, r1164, r1235, r1320;
}
{
mul.f16x2 r1327, r1173, r1237;
}
{
mul.f16x2 r1330, r1176, r1238;
}
{
sub.f16x2 r1333, r1327, r1330;
}
{
mul.f16x2 r1336, r1173, r1238;
}
{
fma.rn.f16x2 r1339, r1176, r1237, r1336;
}
{
mul.f16x2 r1343, r1185, r1239;
}
{
mul.f16x2 r1346, r1188, r1240;
}
{
sub.f16x2 r1349, r1343, r1346;
}
{
mul.f16x2 r1352, r1185, r1240;
}
{
fma.rn.f16x2 r1355, r1188, r1239, r1352;
}
{
mul.f16x2 r1359, r1197, r1241;
}
{
mul.f16x2 r1362, r1200, r1242;
}
{
sub.f16x2 r1365, r1359, r1362;
}
{
mul.f16x2 r1368, r1197, r1242;
}
{
fma.rn.f16x2 r1371, r1200, r1241, r1368;
}
{
mul.f16x2 r1375, r1209, r1243;
}
{
mul.f16x2 r1378, r1212, r1244;
}
{
sub.f16x2 r1381, r1375, r1378;
}
{
mul.f16x2 r1384, r1209, r1244;
}
{
fma.rn.f16x2 r1387, r1212, r1243, r1384;
}
{
mul.f16x2 r1391, r1221, r1245;
}
{
mul.f16x2 r1394, r1224, r1246;
}
{
sub.f16x2 r1397, r1391, r1394;
}
{
mul.f16x2 r1400, r1221, r1246;
}
{
fma.rn.f16x2 r1403, r1224, r1245, r1400;
}
{
neg.f16x2 r1407, r1146;
}
{
mul.f16x2 r1409, r1155, r1249;
}
{
mul.f16x2 r1412, r1158, r1250;
}
{
sub.f16x2 r1415, r1409, r1412;
}
{
mul.f16x2 r1418, r1155, r1250;
}
{
fma.rn.f16x2 r1421, r1158, r1249, r1418;
}
{
mul.f16x2 r1425, r1167, r1251;
}
{
mul.f16x2 r1428, r1170, r1252;
}
{
sub.f16x2 r1431, r1425, r1428;
}
{
mul.f16x2 r1434, r1167, r1252;
}
{
fma.rn.f16x2 r1437, r1170, r1251, r1434;
}
{
mul.f16x2 r1441, r1179, r1253;
}
{
mul.f16x2 r1444, r1182, r1254;
}
{
sub.f16x2 r1447, r1441, r1444;
}
{
mul.f16x2 r1450, r1179, r1254;
}
{
fma.rn.f16x2 r1453, r1182, r1253, r1450;
}
{
mul.f16x2 r1457, r1191, r1255;
}
{
mul.f16x2 r1460, r1194, r1256;
}
{
sub.f16x2 r1463, r1457, r1460;
}
{
mul.f16x2 r1466, r1191, r1256;
}
{
fma.rn.f16x2 r1469, r1194, r1255, r1466;
}
{
mul.f16x2 r1473, r1203, r1257;
}
{
mul.f16x2 r1476, r1206, r1258;
}
{
sub.f16x2 r1479, r1473, r1476;
}
{
mul.f16x2 r1482, r1203, r1258;
}
{
fma.rn.f16x2 r1485, r1206, r1257, r1482;
}
{
mul.f16x2 r1489, r1215, r1259;
}
{
mul.f16x2 r1492, r1218, r1260;
}
{
sub.f16x2 r1495, r1489, r1492;
}
{
mul.f16x2 r1498, r1215, r1260;
}
{
fma.rn.f16x2 r1501, r1218, r1259, r1498;
}
{
mul.f16x2 r1505, r1227, r1261;
}
{
mul.f16x2 r1508, r1230, r1262;
}
{
sub.f16x2 r1511, r1505, r1508;
}
{
mul.f16x2 r1514, r1227, r1262;
}
{
fma.rn.f16x2 r1517, r1230, r1261, r1514;
}
{
add.f16x2 r1521, r521, r1137;
}
{
add.f16x2 r1524, r524, r1140;
}
{
sub.f16x2 r1527, r521, r1137;
}
{
sub.f16x2 r1530, r524, r1140;
}
{
add.f16x2 r1533, r533, r1301;
}
{
add.f16x2 r1536, r536, r1307;
}
{
sub.f16x2 r1539, r533, r1301;
}
{
sub.f16x2 r1542, r536, r1307;
}
{
add.f16x2 r1545, r545, r1317;
}
{
add.f16x2 r1548, r548, r1323;
}
{
sub.f16x2 r1551, r545, r1317;
}
{
sub.f16x2 r1554, r548, r1323;
}
{
add.f16x2 r1557, r557, r1333;
}
{
add.f16x2 r1560, r560, r1339;
}
{
sub.f16x2 r1563, r557, r1333;
}
{
sub.f16x2 r1566, r560, r1339;
}
{
add.f16x2 r1569, r569, r1349;
}
{
add.f16x2 r1572, r572, r1355;
}
{
sub.f16x2 r1575, r569, r1349;
}
{
sub.f16x2 r1578, r572, r1355;
}
{
add.f16x2 r1581, r581, r1365;
}
{
add.f16x2 r1584, r584, r1371;
}
{
sub.f16x2 r1587, r581, r1365;
}
{
sub.f16x2 r1590, r584, r1371;
}
{
add.f16x2 r1593, r593, r1381;
}
{
add.f16x2 r1596, r596, r1387;
}
{
sub.f16x2 r1599, r593, r1381;
}
{
sub.f16x2 r1602, r596, r1387;
}
{
add.f16x2 r1605, r605, r1397;
}
{
add.f16x2 r1608, r608, r1403;
}
{
sub.f16x2 r1611, r605, r1397;
}
{
sub.f16x2 r1614, r608, r1403;
}
{
add.f16x2 r1617, r527, r1407;
}
{
add.f16x2 r1620, r530, r1143;
}
{
sub.f16x2 r1623, r527, r1407;
}
{
sub.f16x2 r1626, r530, r1143;
}
{
add.f16x2 r1629, r539, r1415;
}
{
add.f16x2 r1632, r542, r1421;
}
{
sub.f16x2 r1635, r539, r1415;
}
{
sub.f16x2 r1638, r542, r1421;
}
{
add.f16x2 r1641, r551, r1431;
}
{
add.f16x2 r1644, r554, r1437;
}
{
sub.f16x2 r1647, r551, r1431;
}
{
sub.f16x2 r1650, r554, r1437;
}
{
add.f16x2 r1653, r563, r1447;
}
{
add.f16x2 r1656, r566, r1453;
}
{
sub.f16x2 r1659, r563, r1447;
}
{
sub.f16x2 r1662, r566, r1453;
}
{
add.f16x2 r1665, r575, r1463;
}
{
add.f16x2 r1668, r578, r1469;
}
{
sub.f16x2 r1671, r575, r1463;
}
{
sub.f16x2 r1674, r578, r1469;
}
{
add.f16x2 r1677, r587, r1479;
}
{
add.f16x2 r1680, r590, r1485;
}
{
sub.f16x2 r1683, r587, r1479;
}
{
sub.f16x2 r1686, r590, r1485;
}
{
add.f16x2 r1689, r599, r1495;
}
{
add.f16x2 r1692, r602, r1501;
}
{
sub.f16x2 r1695, r599, r1495;
}
{
sub.f16x2 r1698, r602, r1501;
}
{
add.f16x2 r1701, r611, r1511;
}
{
add.f16x2 r1704, r614, r1517;
}
{
sub.f16x2 r1707, r611, r1511;
}
{
sub.f16x2 r1710, r614, r1517;
}
shl.b32 r6963, r6961, 7;
and.b32 r6964, r6963, -65536;
add.s32 r6965, r6960, r6964;
and.b32 r6977, r6961, 511;
cvt.rn.f32.u32 f1077, r6977;
mul.f32 f1078, f1077, 0f39C90FDB;
cos.approx.f32 f357, f1078;
sin.approx.f32 f1079, f1078;
neg.f32 f358, f1079;
mov.f32 f1084, 0f3F800000;
mov.f32 f1083, 0fBF800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f357;
cvt.rn.f16.f32 high, f358;
mov.b32 r1713, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1716, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1718, {high, high};
}
{
mul.f16x2 r1720, r1536, r1718;
}
{
fma.rn.f16x2 r1723, r1533, r1716, r1720;
}
{
mul.f16x2 r1727, r1533, r1718;
}
{
neg.f16x2 r1730, r1727;
}
{
fma.rn.f16x2 r1732, r1536, r1716, r1730;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1736, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1738, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r1740, {low, high};
}
{
mul.f16x2 r1741, r1738, r1740;
}
{
mul.f16x2 r1744, r1713, r1736;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1747, {high, low};
}
{
fma.rn.f16x2 r1749, r1741, r1747, r1744;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1749;
mov.b32 r1753, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1749;
mov.b32 r1755, {high, high};
}
{
mul.f16x2 r1757, r1548, r1755;
}
{
fma.rn.f16x2 r1760, r1545, r1753, r1757;
}
{
mul.f16x2 r1764, r1545, r1755;
}
{
neg.f16x2 r1767, r1764;
}
{
fma.rn.f16x2 r1769, r1548, r1753, r1767;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1773, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1775, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r1777, {low, high};
}
{
mul.f16x2 r1778, r1775, r1777;
}
{
mul.f16x2 r1781, r1749, r1773;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1749;
mov.b32 r1784, {high, low};
}
{
fma.rn.f16x2 r1786, r1778, r1784, r1781;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1786;
mov.b32 r1790, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1786;
mov.b32 r1792, {high, high};
}
{
mul.f16x2 r1794, r1560, r1792;
}
{
fma.rn.f16x2 r1797, r1557, r1790, r1794;
}
{
mul.f16x2 r1801, r1557, r1792;
}
{
neg.f16x2 r1804, r1801;
}
{
fma.rn.f16x2 r1806, r1560, r1790, r1804;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1810, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1812, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r1814, {low, high};
}
{
mul.f16x2 r1815, r1812, r1814;
}
{
mul.f16x2 r1818, r1786, r1810;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1786;
mov.b32 r1821, {high, low};
}
{
fma.rn.f16x2 r1823, r1815, r1821, r1818;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1823;
mov.b32 r1827, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1823;
mov.b32 r1829, {high, high};
}
{
mul.f16x2 r1831, r1572, r1829;
}
{
fma.rn.f16x2 r1834, r1569, r1827, r1831;
}
{
mul.f16x2 r1838, r1569, r1829;
}
{
neg.f16x2 r1841, r1838;
}
{
fma.rn.f16x2 r1843, r1572, r1827, r1841;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1847, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1849, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r1851, {low, high};
}
{
mul.f16x2 r1852, r1849, r1851;
}
{
mul.f16x2 r1855, r1823, r1847;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1823;
mov.b32 r1858, {high, low};
}
{
fma.rn.f16x2 r1860, r1852, r1858, r1855;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1860;
mov.b32 r1864, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1860;
mov.b32 r1866, {high, high};
}
{
mul.f16x2 r1868, r1584, r1866;
}
{
fma.rn.f16x2 r1871, r1581, r1864, r1868;
}
{
mul.f16x2 r1875, r1581, r1866;
}
{
neg.f16x2 r1878, r1875;
}
{
fma.rn.f16x2 r1880, r1584, r1864, r1878;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1884, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1886, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r1888, {low, high};
}
{
mul.f16x2 r1889, r1886, r1888;
}
{
mul.f16x2 r1892, r1860, r1884;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1860;
mov.b32 r1895, {high, low};
}
{
fma.rn.f16x2 r1897, r1889, r1895, r1892;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1897;
mov.b32 r1901, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1897;
mov.b32 r1903, {high, high};
}
{
mul.f16x2 r1905, r1596, r1903;
}
{
fma.rn.f16x2 r1908, r1593, r1901, r1905;
}
{
mul.f16x2 r1912, r1593, r1903;
}
{
neg.f16x2 r1915, r1912;
}
{
fma.rn.f16x2 r1917, r1596, r1901, r1915;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1921, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1923, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r1925, {low, high};
}
{
mul.f16x2 r1926, r1923, r1925;
}
{
mul.f16x2 r1929, r1897, r1921;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1897;
mov.b32 r1932, {high, low};
}
{
fma.rn.f16x2 r1934, r1926, r1932, r1929;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1934;
mov.b32 r1938, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1934;
mov.b32 r1940, {high, high};
}
{
mul.f16x2 r1942, r1608, r1940;
}
{
fma.rn.f16x2 r1945, r1605, r1938, r1942;
}
{
mul.f16x2 r1949, r1605, r1940;
}
{
neg.f16x2 r1952, r1949;
}
{
fma.rn.f16x2 r1954, r1608, r1938, r1952;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1958, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1960, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r1962, {low, high};
}
{
mul.f16x2 r1963, r1960, r1962;
}
{
mul.f16x2 r1966, r1934, r1958;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1934;
mov.b32 r1969, {high, low};
}
{
fma.rn.f16x2 r1971, r1963, r1969, r1966;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1971;
mov.b32 r1975, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1971;
mov.b32 r1977, {high, high};
}
{
mul.f16x2 r1979, r1620, r1977;
}
{
fma.rn.f16x2 r1982, r1617, r1975, r1979;
}
{
mul.f16x2 r1986, r1617, r1977;
}
{
neg.f16x2 r1989, r1986;
}
{
fma.rn.f16x2 r1991, r1620, r1975, r1989;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1995, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1997, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r1999, {low, high};
}
{
mul.f16x2 r2000, r1997, r1999;
}
{
mul.f16x2 r2003, r1971, r1995;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1971;
mov.b32 r2006, {high, low};
}
{
fma.rn.f16x2 r2008, r2000, r2006, r2003;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2008;
mov.b32 r2012, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2008;
mov.b32 r2014, {high, high};
}
{
mul.f16x2 r2016, r1632, r2014;
}
{
fma.rn.f16x2 r2019, r1629, r2012, r2016;
}
{
mul.f16x2 r2023, r1629, r2014;
}
{
neg.f16x2 r2026, r2023;
}
{
fma.rn.f16x2 r2028, r1632, r2012, r2026;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2032, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2034, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2036, {low, high};
}
{
mul.f16x2 r2037, r2034, r2036;
}
{
mul.f16x2 r2040, r2008, r2032;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2008;
mov.b32 r2043, {high, low};
}
{
fma.rn.f16x2 r2045, r2037, r2043, r2040;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2045;
mov.b32 r2049, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2045;
mov.b32 r2051, {high, high};
}
{
mul.f16x2 r2053, r1644, r2051;
}
{
fma.rn.f16x2 r2056, r1641, r2049, r2053;
}
{
mul.f16x2 r2060, r1641, r2051;
}
{
neg.f16x2 r2063, r2060;
}
{
fma.rn.f16x2 r2065, r1644, r2049, r2063;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2069, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2071, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2073, {low, high};
}
{
mul.f16x2 r2074, r2071, r2073;
}
{
mul.f16x2 r2077, r2045, r2069;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2045;
mov.b32 r2080, {high, low};
}
{
fma.rn.f16x2 r2082, r2074, r2080, r2077;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2082;
mov.b32 r2086, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2082;
mov.b32 r2088, {high, high};
}
{
mul.f16x2 r2090, r1656, r2088;
}
{
fma.rn.f16x2 r2093, r1653, r2086, r2090;
}
{
mul.f16x2 r2097, r1653, r2088;
}
{
neg.f16x2 r2100, r2097;
}
{
fma.rn.f16x2 r2102, r1656, r2086, r2100;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2106, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2108, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2110, {low, high};
}
{
mul.f16x2 r2111, r2108, r2110;
}
{
mul.f16x2 r2114, r2082, r2106;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2082;
mov.b32 r2117, {high, low};
}
{
fma.rn.f16x2 r2119, r2111, r2117, r2114;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2119;
mov.b32 r2123, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2119;
mov.b32 r2125, {high, high};
}
{
mul.f16x2 r2127, r1668, r2125;
}
{
fma.rn.f16x2 r2130, r1665, r2123, r2127;
}
{
mul.f16x2 r2134, r1665, r2125;
}
{
neg.f16x2 r2137, r2134;
}
{
fma.rn.f16x2 r2139, r1668, r2123, r2137;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2143, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2145, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2147, {low, high};
}
{
mul.f16x2 r2148, r2145, r2147;
}
{
mul.f16x2 r2151, r2119, r2143;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2119;
mov.b32 r2154, {high, low};
}
{
fma.rn.f16x2 r2156, r2148, r2154, r2151;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2156;
mov.b32 r2160, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2156;
mov.b32 r2162, {high, high};
}
{
mul.f16x2 r2164, r1680, r2162;
}
{
fma.rn.f16x2 r2167, r1677, r2160, r2164;
}
{
mul.f16x2 r2171, r1677, r2162;
}
{
neg.f16x2 r2174, r2171;
}
{
fma.rn.f16x2 r2176, r1680, r2160, r2174;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2180, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2182, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2184, {low, high};
}
{
mul.f16x2 r2185, r2182, r2184;
}
{
mul.f16x2 r2188, r2156, r2180;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2156;
mov.b32 r2191, {high, low};
}
{
fma.rn.f16x2 r2193, r2185, r2191, r2188;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2193;
mov.b32 r2197, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2193;
mov.b32 r2199, {high, high};
}
{
mul.f16x2 r2201, r1692, r2199;
}
{
fma.rn.f16x2 r2204, r1689, r2197, r2201;
}
{
mul.f16x2 r2208, r1689, r2199;
}
{
neg.f16x2 r2211, r2208;
}
{
fma.rn.f16x2 r2213, r1692, r2197, r2211;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2217, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2219, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2221, {low, high};
}
{
mul.f16x2 r2222, r2219, r2221;
}
{
mul.f16x2 r2225, r2193, r2217;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2193;
mov.b32 r2228, {high, low};
}
{
fma.rn.f16x2 r2230, r2222, r2228, r2225;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2230;
mov.b32 r2234, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2230;
mov.b32 r2236, {high, high};
}
{
mul.f16x2 r2238, r1704, r2236;
}
{
fma.rn.f16x2 r2241, r1701, r2234, r2238;
}
{
mul.f16x2 r2245, r1701, r2236;
}
{
neg.f16x2 r2248, r2245;
}
{
fma.rn.f16x2 r2250, r1704, r2234, r2248;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2254, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2256, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2258, {low, high};
}
{
mul.f16x2 r2259, r2256, r2258;
}
{
mul.f16x2 r2262, r2230, r2254;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2230;
mov.b32 r2265, {high, low};
}
{
fma.rn.f16x2 r2267, r2259, r2265, r2262;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2267;
mov.b32 r2271, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2267;
mov.b32 r2273, {high, high};
}
{
mul.f16x2 r2275, r1530, r2273;
}
{
fma.rn.f16x2 r2278, r1527, r2271, r2275;
}
{
mul.f16x2 r2282, r1527, r2273;
}
{
neg.f16x2 r2285, r2282;
}
{
fma.rn.f16x2 r2287, r1530, r2271, r2285;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2291, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2293, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2295, {low, high};
}
{
mul.f16x2 r2296, r2293, r2295;
}
{
mul.f16x2 r2299, r2267, r2291;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2267;
mov.b32 r2302, {high, low};
}
{
fma.rn.f16x2 r2304, r2296, r2302, r2299;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2304;
mov.b32 r2308, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2304;
mov.b32 r2310, {high, high};
}
{
mul.f16x2 r2312, r1542, r2310;
}
{
fma.rn.f16x2 r2315, r1539, r2308, r2312;
}
{
mul.f16x2 r2319, r1539, r2310;
}
{
neg.f16x2 r2322, r2319;
}
{
fma.rn.f16x2 r2324, r1542, r2308, r2322;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2328, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2330, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2332, {low, high};
}
{
mul.f16x2 r2333, r2330, r2332;
}
{
mul.f16x2 r2336, r2304, r2328;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2304;
mov.b32 r2339, {high, low};
}
{
fma.rn.f16x2 r2341, r2333, r2339, r2336;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2341;
mov.b32 r2345, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2341;
mov.b32 r2347, {high, high};
}
{
mul.f16x2 r2349, r1554, r2347;
}
{
fma.rn.f16x2 r2352, r1551, r2345, r2349;
}
{
mul.f16x2 r2356, r1551, r2347;
}
{
neg.f16x2 r2359, r2356;
}
{
fma.rn.f16x2 r2361, r1554, r2345, r2359;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2365, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2367, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2369, {low, high};
}
{
mul.f16x2 r2370, r2367, r2369;
}
{
mul.f16x2 r2373, r2341, r2365;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2341;
mov.b32 r2376, {high, low};
}
{
fma.rn.f16x2 r2378, r2370, r2376, r2373;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2378;
mov.b32 r2382, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2378;
mov.b32 r2384, {high, high};
}
{
mul.f16x2 r2386, r1566, r2384;
}
{
fma.rn.f16x2 r2389, r1563, r2382, r2386;
}
{
mul.f16x2 r2393, r1563, r2384;
}
{
neg.f16x2 r2396, r2393;
}
{
fma.rn.f16x2 r2398, r1566, r2382, r2396;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2402, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2404, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2406, {low, high};
}
{
mul.f16x2 r2407, r2404, r2406;
}
{
mul.f16x2 r2410, r2378, r2402;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2378;
mov.b32 r2413, {high, low};
}
{
fma.rn.f16x2 r2415, r2407, r2413, r2410;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2415;
mov.b32 r2419, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2415;
mov.b32 r2421, {high, high};
}
{
mul.f16x2 r2423, r1578, r2421;
}
{
fma.rn.f16x2 r2426, r1575, r2419, r2423;
}
{
mul.f16x2 r2430, r1575, r2421;
}
{
neg.f16x2 r2433, r2430;
}
{
fma.rn.f16x2 r2435, r1578, r2419, r2433;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2439, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2441, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2443, {low, high};
}
{
mul.f16x2 r2444, r2441, r2443;
}
{
mul.f16x2 r2447, r2415, r2439;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2415;
mov.b32 r2450, {high, low};
}
{
fma.rn.f16x2 r2452, r2444, r2450, r2447;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2452;
mov.b32 r2456, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2452;
mov.b32 r2458, {high, high};
}
{
mul.f16x2 r2460, r1590, r2458;
}
{
fma.rn.f16x2 r2463, r1587, r2456, r2460;
}
{
mul.f16x2 r2467, r1587, r2458;
}
{
neg.f16x2 r2470, r2467;
}
{
fma.rn.f16x2 r2472, r1590, r2456, r2470;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2476, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2478, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2480, {low, high};
}
{
mul.f16x2 r2481, r2478, r2480;
}
{
mul.f16x2 r2484, r2452, r2476;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2452;
mov.b32 r2487, {high, low};
}
{
fma.rn.f16x2 r2489, r2481, r2487, r2484;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2489;
mov.b32 r2493, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2489;
mov.b32 r2495, {high, high};
}
{
mul.f16x2 r2497, r1602, r2495;
}
{
fma.rn.f16x2 r2500, r1599, r2493, r2497;
}
{
mul.f16x2 r2504, r1599, r2495;
}
{
neg.f16x2 r2507, r2504;
}
{
fma.rn.f16x2 r2509, r1602, r2493, r2507;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2513, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2515, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2517, {low, high};
}
{
mul.f16x2 r2518, r2515, r2517;
}
{
mul.f16x2 r2521, r2489, r2513;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2489;
mov.b32 r2524, {high, low};
}
{
fma.rn.f16x2 r2526, r2518, r2524, r2521;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2526;
mov.b32 r2530, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2526;
mov.b32 r2532, {high, high};
}
{
mul.f16x2 r2534, r1614, r2532;
}
{
fma.rn.f16x2 r2537, r1611, r2530, r2534;
}
{
mul.f16x2 r2541, r1611, r2532;
}
{
neg.f16x2 r2544, r2541;
}
{
fma.rn.f16x2 r2546, r1614, r2530, r2544;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2550, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2552, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2554, {low, high};
}
{
mul.f16x2 r2555, r2552, r2554;
}
{
mul.f16x2 r2558, r2526, r2550;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2526;
mov.b32 r2561, {high, low};
}
{
fma.rn.f16x2 r2563, r2555, r2561, r2558;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2563;
mov.b32 r2567, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2563;
mov.b32 r2569, {high, high};
}
{
mul.f16x2 r2571, r1626, r2569;
}
{
fma.rn.f16x2 r2574, r1623, r2567, r2571;
}
{
mul.f16x2 r2578, r1623, r2569;
}
{
neg.f16x2 r2581, r2578;
}
{
fma.rn.f16x2 r2583, r1626, r2567, r2581;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2587, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2589, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2591, {low, high};
}
{
mul.f16x2 r2592, r2589, r2591;
}
{
mul.f16x2 r2595, r2563, r2587;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2563;
mov.b32 r2598, {high, low};
}
{
fma.rn.f16x2 r2600, r2592, r2598, r2595;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2600;
mov.b32 r2604, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2600;
mov.b32 r2606, {high, high};
}
{
mul.f16x2 r2608, r1638, r2606;
}
{
fma.rn.f16x2 r2611, r1635, r2604, r2608;
}
{
mul.f16x2 r2615, r1635, r2606;
}
{
neg.f16x2 r2618, r2615;
}
{
fma.rn.f16x2 r2620, r1638, r2604, r2618;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2624, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2626, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2628, {low, high};
}
{
mul.f16x2 r2629, r2626, r2628;
}
{
mul.f16x2 r2632, r2600, r2624;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2600;
mov.b32 r2635, {high, low};
}
{
fma.rn.f16x2 r2637, r2629, r2635, r2632;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2637;
mov.b32 r2641, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2637;
mov.b32 r2643, {high, high};
}
{
mul.f16x2 r2645, r1650, r2643;
}
{
fma.rn.f16x2 r2648, r1647, r2641, r2645;
}
{
mul.f16x2 r2652, r1647, r2643;
}
{
neg.f16x2 r2655, r2652;
}
{
fma.rn.f16x2 r2657, r1650, r2641, r2655;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2661, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2663, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2665, {low, high};
}
{
mul.f16x2 r2666, r2663, r2665;
}
{
mul.f16x2 r2669, r2637, r2661;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2637;
mov.b32 r2672, {high, low};
}
{
fma.rn.f16x2 r2674, r2666, r2672, r2669;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2674;
mov.b32 r2678, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2674;
mov.b32 r2680, {high, high};
}
{
mul.f16x2 r2682, r1662, r2680;
}
{
fma.rn.f16x2 r2685, r1659, r2678, r2682;
}
{
mul.f16x2 r2689, r1659, r2680;
}
{
neg.f16x2 r2692, r2689;
}
{
fma.rn.f16x2 r2694, r1662, r2678, r2692;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2698, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2700, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2702, {low, high};
}
{
mul.f16x2 r2703, r2700, r2702;
}
{
mul.f16x2 r2706, r2674, r2698;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2674;
mov.b32 r2709, {high, low};
}
{
fma.rn.f16x2 r2711, r2703, r2709, r2706;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2711;
mov.b32 r2715, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2711;
mov.b32 r2717, {high, high};
}
{
mul.f16x2 r2719, r1674, r2717;
}
{
fma.rn.f16x2 r2722, r1671, r2715, r2719;
}
{
mul.f16x2 r2726, r1671, r2717;
}
{
neg.f16x2 r2729, r2726;
}
{
fma.rn.f16x2 r2731, r1674, r2715, r2729;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2735, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2737, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2739, {low, high};
}
{
mul.f16x2 r2740, r2737, r2739;
}
{
mul.f16x2 r2743, r2711, r2735;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2711;
mov.b32 r2746, {high, low};
}
{
fma.rn.f16x2 r2748, r2740, r2746, r2743;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2748;
mov.b32 r2752, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2748;
mov.b32 r2754, {high, high};
}
{
mul.f16x2 r2756, r1686, r2754;
}
{
fma.rn.f16x2 r2759, r1683, r2752, r2756;
}
{
mul.f16x2 r2763, r1683, r2754;
}
{
neg.f16x2 r2766, r2763;
}
{
fma.rn.f16x2 r2768, r1686, r2752, r2766;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2772, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2774, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2776, {low, high};
}
{
mul.f16x2 r2777, r2774, r2776;
}
{
mul.f16x2 r2780, r2748, r2772;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2748;
mov.b32 r2783, {high, low};
}
{
fma.rn.f16x2 r2785, r2777, r2783, r2780;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2785;
mov.b32 r2789, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2785;
mov.b32 r2791, {high, high};
}
{
mul.f16x2 r2793, r1698, r2791;
}
{
fma.rn.f16x2 r2796, r1695, r2789, r2793;
}
{
mul.f16x2 r2800, r1695, r2791;
}
{
neg.f16x2 r2803, r2800;
}
{
fma.rn.f16x2 r2805, r1698, r2789, r2803;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2809, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r2811, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r2813, {low, high};
}
{
mul.f16x2 r2814, r2811, r2813;
}
{
mul.f16x2 r2817, r2785, r2809;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2785;
mov.b32 r2820, {high, low};
}
{
fma.rn.f16x2 r2822, r2814, r2820, r2817;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2822;
mov.b32 r2826, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2822;
mov.b32 r2828, {high, high};
}
{
mul.f16x2 r2830, r1710, r2828;
}
{
fma.rn.f16x2 r2833, r1707, r2826, r2830;
}
{
mul.f16x2 r2837, r1707, r2828;
}
{
neg.f16x2 r2840, r2837;
}
{
fma.rn.f16x2 r2842, r1710, r2826, r2840;
}
barrier.sync 0;
and.b32 r6966, r6963, 65408;
add.s32 r6967, r6965, r6966;
st.shared.v4.f32 [r6967], {r1521, r1723, r1760, r1797};
st.shared.v4.f32 [r6967+16], {r1834, r1871, r1908, r1945};
st.shared.v4.f32 [r6967+32], {r1982, r2019, r2056, r2093};
st.shared.v4.f32 [r6967+48], {r2130, r2167, r2204, r2241};
st.shared.v4.f32 [r6967+64], {r2278, r2315, r2352, r2389};
st.shared.v4.f32 [r6967+80], {r2426, r2463, r2500, r2537};
st.shared.v4.f32 [r6967+96], {r2574, r2611, r2648, r2685};
st.shared.v4.f32 [r6967+112], {r2722, r2759, r2796, r2833};
barrier.sync 0;
mad.lo.s32 r6968, r6977, -124, r6967;
ld.shared.u32 r2864, [r6968];
ld.shared.u32 r3480, [r6968+2048];
ld.shared.u32 r3060, [r6968+4096];
ld.shared.u32 r3676, [r6968+6144];
ld.shared.u32 r2914, [r6968+8192];
ld.shared.u32 r3530, [r6968+10240];
ld.shared.u32 r3110, [r6968+12288];
ld.shared.u32 r3726, [r6968+14336];
ld.shared.u32 r2876, [r6968+16384];
ld.shared.u32 r3492, [r6968+18432];
ld.shared.u32 r3072, [r6968+20480];
ld.shared.u32 r3688, [r6968+22528];
ld.shared.u32 r2926, [r6968+24576];
ld.shared.u32 r3542, [r6968+26624];
ld.shared.u32 r3122, [r6968+28672];
ld.shared.u32 r3738, [r6968+30720];
ld.shared.u32 r2865, [r6968+32768];
ld.shared.u32 r3481, [r6968+34816];
ld.shared.u32 r3061, [r6968+36864];
ld.shared.u32 r3677, [r6968+38912];
ld.shared.u32 r2915, [r6968+40960];
ld.shared.u32 r3531, [r6968+43008];
ld.shared.u32 r3111, [r6968+45056];
ld.shared.u32 r3727, [r6968+47104];
ld.shared.u32 r2877, [r6968+49152];
ld.shared.u32 r3493, [r6968+51200];
ld.shared.u32 r3073, [r6968+53248];
ld.shared.u32 r3689, [r6968+55296];
ld.shared.u32 r2927, [r6968+57344];
ld.shared.u32 r3543, [r6968+59392];
ld.shared.u32 r3123, [r6968+61440];
ld.shared.u32 r3739, [r6968+63488];
barrier.sync 0;
st.shared.v4.f32 [r6967], {r1524, r1732, r1769, r1806};
st.shared.v4.f32 [r6967+16], {r1843, r1880, r1917, r1954};
st.shared.v4.f32 [r6967+32], {r1991, r2028, r2065, r2102};
st.shared.v4.f32 [r6967+48], {r2139, r2176, r2213, r2250};
st.shared.v4.f32 [r6967+64], {r2287, r2324, r2361, r2398};
st.shared.v4.f32 [r6967+80], {r2435, r2472, r2509, r2546};
st.shared.v4.f32 [r6967+96], {r2583, r2620, r2657, r2694};
st.shared.v4.f32 [r6967+112], {r2731, r2768, r2805, r2842};
barrier.sync 0;
ld.shared.u32 r2867, [r6968];
ld.shared.u32 r3483, [r6968+2048];
ld.shared.u32 r3063, [r6968+4096];
ld.shared.u32 r3679, [r6968+6144];
ld.shared.u32 r2917, [r6968+8192];
ld.shared.u32 r3533, [r6968+10240];
ld.shared.u32 r3113, [r6968+12288];
ld.shared.u32 r3729, [r6968+14336];
ld.shared.u32 r2879, [r6968+16384];
ld.shared.u32 r3495, [r6968+18432];
ld.shared.u32 r3075, [r6968+20480];
ld.shared.u32 r3691, [r6968+22528];
ld.shared.u32 r2929, [r6968+24576];
ld.shared.u32 r3545, [r6968+26624];
ld.shared.u32 r3125, [r6968+28672];
ld.shared.u32 r3741, [r6968+30720];
ld.shared.u32 r2868, [r6968+32768];
ld.shared.u32 r3484, [r6968+34816];
ld.shared.u32 r3064, [r6968+36864];
ld.shared.u32 r3680, [r6968+38912];
ld.shared.u32 r2918, [r6968+40960];
ld.shared.u32 r3534, [r6968+43008];
ld.shared.u32 r3114, [r6968+45056];
ld.shared.u32 r3730, [r6968+47104];
ld.shared.u32 r2880, [r6968+49152];
ld.shared.u32 r3496, [r6968+51200];
ld.shared.u32 r3076, [r6968+53248];
ld.shared.u32 r3692, [r6968+55296];
ld.shared.u32 r2930, [r6968+57344];
ld.shared.u32 r3546, [r6968+59392];
ld.shared.u32 r3126, [r6968+61440];
ld.shared.u32 r3742, [r6968+63488];
{
add.f16x2 r2863, r2864, r2865;
}
{
add.f16x2 r2866, r2867, r2868;
}
{
sub.f16x2 r2869, r2864, r2865;
}
{
sub.f16x2 r2872, r2867, r2868;
}
{
add.f16x2 r2875, r2876, r2877;
}
{
add.f16x2 r2878, r2879, r2880;
}
{
sub.f16x2 r2881, r2876, r2877;
}
{
sub.f16x2 r2884, r2879, r2880;
}
{
neg.f16x2 r2887, r2884;
}
{
add.f16x2 r2889, r2863, r2875;
}
{
add.f16x2 r2892, r2866, r2878;
}
{
sub.f16x2 r2895, r2863, r2875;
}
{
sub.f16x2 r2898, r2866, r2878;
}
{
add.f16x2 r2901, r2869, r2887;
}
{
add.f16x2 r2904, r2872, r2881;
}
{
sub.f16x2 r2907, r2869, r2887;
}
{
sub.f16x2 r2910, r2872, r2881;
}
{
add.f16x2 r2913, r2914, r2915;
}
{
add.f16x2 r2916, r2917, r2918;
}
{
sub.f16x2 r2919, r2914, r2915;
}
{
sub.f16x2 r2922, r2917, r2918;
}
{
add.f16x2 r2925, r2926, r2927;
}
{
add.f16x2 r2928, r2929, r2930;
}
{
sub.f16x2 r2931, r2926, r2927;
}
{
sub.f16x2 r2934, r2929, r2930;
}
{
neg.f16x2 r2937, r2934;
}
{
add.f16x2 r2939, r2913, r2925;
}
{
add.f16x2 r2942, r2916, r2928;
}
{
sub.f16x2 r2945, r2913, r2925;
}
{
sub.f16x2 r2948, r2916, r2928;
}
{
add.f16x2 r2951, r2919, r2937;
}
{
add.f16x2 r2954, r2922, r2931;
}
{
sub.f16x2 r2957, r2919, r2937;
}
{
sub.f16x2 r2960, r2922, r2931;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r2963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r2964, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r2967, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r2968, {low, high};
}
{
mul.f16x2 r2977, r2951, r2963;
}
{
mul.f16x2 r2980, r2954, r2964;
}
{
sub.f16x2 r2983, r2977, r2980;
}
{
mul.f16x2 r2986, r2951, r2964;
}
{
fma.rn.f16x2 r2989, r2954, r2963, r2986;
}
{
neg.f16x2 r2993, r2948;
}
{
mul.f16x2 r2995, r2957, r2967;
}
{
mul.f16x2 r2998, r2960, r2968;
}
{
sub.f16x2 r3001, r2995, r2998;
}
{
mul.f16x2 r3004, r2957, r2968;
}
{
fma.rn.f16x2 r3007, r2960, r2967, r3004;
}
{
add.f16x2 r3011, r2889, r2939;
}
{
add.f16x2 r3014, r2892, r2942;
}
{
sub.f16x2 r3017, r2889, r2939;
}
{
sub.f16x2 r3020, r2892, r2942;
}
{
add.f16x2 r3023, r2901, r2983;
}
{
add.f16x2 r3026, r2904, r2989;
}
{
sub.f16x2 r3029, r2901, r2983;
}
{
sub.f16x2 r3032, r2904, r2989;
}
{
add.f16x2 r3035, r2895, r2993;
}
{
add.f16x2 r3038, r2898, r2945;
}
{
sub.f16x2 r3041, r2895, r2993;
}
{
sub.f16x2 r3044, r2898, r2945;
}
{
add.f16x2 r3047, r2907, r3001;
}
{
add.f16x2 r3050, r2910, r3007;
}
{
sub.f16x2 r3053, r2907, r3001;
}
{
sub.f16x2 r3056, r2910, r3007;
}
{
add.f16x2 r3059, r3060, r3061;
}
{
add.f16x2 r3062, r3063, r3064;
}
{
sub.f16x2 r3065, r3060, r3061;
}
{
sub.f16x2 r3068, r3063, r3064;
}
{
add.f16x2 r3071, r3072, r3073;
}
{
add.f16x2 r3074, r3075, r3076;
}
{
sub.f16x2 r3077, r3072, r3073;
}
{
sub.f16x2 r3080, r3075, r3076;
}
{
neg.f16x2 r3083, r3080;
}
{
add.f16x2 r3085, r3059, r3071;
}
{
add.f16x2 r3088, r3062, r3074;
}
{
sub.f16x2 r3091, r3059, r3071;
}
{
sub.f16x2 r3094, r3062, r3074;
}
{
add.f16x2 r3097, r3065, r3083;
}
{
add.f16x2 r3100, r3068, r3077;
}
{
sub.f16x2 r3103, r3065, r3083;
}
{
sub.f16x2 r3106, r3068, r3077;
}
{
add.f16x2 r3109, r3110, r3111;
}
{
add.f16x2 r3112, r3113, r3114;
}
{
sub.f16x2 r3115, r3110, r3111;
}
{
sub.f16x2 r3118, r3113, r3114;
}
{
add.f16x2 r3121, r3122, r3123;
}
{
add.f16x2 r3124, r3125, r3126;
}
{
sub.f16x2 r3127, r3122, r3123;
}
{
sub.f16x2 r3130, r3125, r3126;
}
{
neg.f16x2 r3133, r3130;
}
{
add.f16x2 r3135, r3109, r3121;
}
{
add.f16x2 r3138, r3112, r3124;
}
{
sub.f16x2 r3141, r3109, r3121;
}
{
sub.f16x2 r3144, r3112, r3124;
}
{
add.f16x2 r3147, r3115, r3133;
}
{
add.f16x2 r3150, r3118, r3127;
}
{
sub.f16x2 r3153, r3115, r3133;
}
{
sub.f16x2 r3156, r3118, r3127;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3159, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3160, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r3163, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3164, {low, high};
}
{
mul.f16x2 r3173, r3147, r3159;
}
{
mul.f16x2 r3176, r3150, r3160;
}
{
sub.f16x2 r3179, r3173, r3176;
}
{
mul.f16x2 r3182, r3147, r3160;
}
{
fma.rn.f16x2 r3185, r3150, r3159, r3182;
}
{
neg.f16x2 r3189, r3144;
}
{
mul.f16x2 r3191, r3153, r3163;
}
{
mul.f16x2 r3194, r3156, r3164;
}
{
sub.f16x2 r3197, r3191, r3194;
}
{
mul.f16x2 r3200, r3153, r3164;
}
{
fma.rn.f16x2 r3203, r3156, r3163, r3200;
}
{
add.f16x2 r3207, r3085, r3135;
}
{
add.f16x2 r3210, r3088, r3138;
}
{
sub.f16x2 r3213, r3085, r3135;
}
{
sub.f16x2 r3216, r3088, r3138;
}
{
add.f16x2 r3219, r3097, r3179;
}
{
add.f16x2 r3222, r3100, r3185;
}
{
sub.f16x2 r3225, r3097, r3179;
}
{
sub.f16x2 r3228, r3100, r3185;
}
{
add.f16x2 r3231, r3091, r3189;
}
{
add.f16x2 r3234, r3094, r3141;
}
{
sub.f16x2 r3237, r3091, r3189;
}
{
sub.f16x2 r3240, r3094, r3141;
}
{
add.f16x2 r3243, r3103, r3197;
}
{
add.f16x2 r3246, r3106, r3203;
}
{
sub.f16x2 r3249, r3103, r3197;
}
{
sub.f16x2 r3252, r3106, r3203;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r3255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r3256, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3257, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3258, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r3259, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r3260, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r3263, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r3264, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r3265, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3266, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1042;
cvt.rn.f16.f32 high, f1042;
mov.b32 r3267, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r3268, {low, high};
}
{
mul.f16x2 r3285, r3219, r3255;
}
{
mul.f16x2 r3288, r3222, r3256;
}
{
sub.f16x2 r3291, r3285, r3288;
}
{
mul.f16x2 r3294, r3219, r3256;
}
{
fma.rn.f16x2 r3297, r3222, r3255, r3294;
}
{
mul.f16x2 r3301, r3231, r3257;
}
{
mul.f16x2 r3304, r3234, r3258;
}
{
sub.f16x2 r3307, r3301, r3304;
}
{
mul.f16x2 r3310, r3231, r3258;
}
{
fma.rn.f16x2 r3313, r3234, r3257, r3310;
}
{
mul.f16x2 r3317, r3243, r3259;
}
{
mul.f16x2 r3320, r3246, r3260;
}
{
sub.f16x2 r3323, r3317, r3320;
}
{
mul.f16x2 r3326, r3243, r3260;
}
{
fma.rn.f16x2 r3329, r3246, r3259, r3326;
}
{
neg.f16x2 r3333, r3216;
}
{
mul.f16x2 r3335, r3225, r3263;
}
{
mul.f16x2 r3338, r3228, r3264;
}
{
sub.f16x2 r3341, r3335, r3338;
}
{
mul.f16x2 r3344, r3225, r3264;
}
{
fma.rn.f16x2 r3347, r3228, r3263, r3344;
}
{
mul.f16x2 r3351, r3237, r3265;
}
{
mul.f16x2 r3354, r3240, r3266;
}
{
sub.f16x2 r3357, r3351, r3354;
}
{
mul.f16x2 r3360, r3237, r3266;
}
{
fma.rn.f16x2 r3363, r3240, r3265, r3360;
}
{
mul.f16x2 r3367, r3249, r3267;
}
{
mul.f16x2 r3370, r3252, r3268;
}
{
sub.f16x2 r3373, r3367, r3370;
}
{
mul.f16x2 r3376, r3249, r3268;
}
{
fma.rn.f16x2 r3379, r3252, r3267, r3376;
}
{
add.f16x2 r3383, r3011, r3207;
}
{
add.f16x2 r3386, r3014, r3210;
}
{
sub.f16x2 r3389, r3011, r3207;
}
{
sub.f16x2 r3392, r3014, r3210;
}
{
add.f16x2 r3395, r3023, r3291;
}
{
add.f16x2 r3398, r3026, r3297;
}
{
sub.f16x2 r3401, r3023, r3291;
}
{
sub.f16x2 r3404, r3026, r3297;
}
{
add.f16x2 r3407, r3035, r3307;
}
{
add.f16x2 r3410, r3038, r3313;
}
{
sub.f16x2 r3413, r3035, r3307;
}
{
sub.f16x2 r3416, r3038, r3313;
}
{
add.f16x2 r3419, r3047, r3323;
}
{
add.f16x2 r3422, r3050, r3329;
}
{
sub.f16x2 r3425, r3047, r3323;
}
{
sub.f16x2 r3428, r3050, r3329;
}
{
add.f16x2 r3431, r3017, r3333;
}
{
add.f16x2 r3434, r3020, r3213;
}
{
sub.f16x2 r3437, r3017, r3333;
}
{
sub.f16x2 r3440, r3020, r3213;
}
{
add.f16x2 r3443, r3029, r3341;
}
{
add.f16x2 r3446, r3032, r3347;
}
{
sub.f16x2 r3449, r3029, r3341;
}
{
sub.f16x2 r3452, r3032, r3347;
}
{
add.f16x2 r3455, r3041, r3357;
}
{
add.f16x2 r3458, r3044, r3363;
}
{
sub.f16x2 r3461, r3041, r3357;
}
{
sub.f16x2 r3464, r3044, r3363;
}
{
add.f16x2 r3467, r3053, r3373;
}
{
add.f16x2 r3470, r3056, r3379;
}
{
sub.f16x2 r3473, r3053, r3373;
}
{
sub.f16x2 r3476, r3056, r3379;
}
{
add.f16x2 r3479, r3480, r3481;
}
{
add.f16x2 r3482, r3483, r3484;
}
{
sub.f16x2 r3485, r3480, r3481;
}
{
sub.f16x2 r3488, r3483, r3484;
}
{
add.f16x2 r3491, r3492, r3493;
}
{
add.f16x2 r3494, r3495, r3496;
}
{
sub.f16x2 r3497, r3492, r3493;
}
{
sub.f16x2 r3500, r3495, r3496;
}
{
neg.f16x2 r3503, r3500;
}
{
add.f16x2 r3505, r3479, r3491;
}
{
add.f16x2 r3508, r3482, r3494;
}
{
sub.f16x2 r3511, r3479, r3491;
}
{
sub.f16x2 r3514, r3482, r3494;
}
{
add.f16x2 r3517, r3485, r3503;
}
{
add.f16x2 r3520, r3488, r3497;
}
{
sub.f16x2 r3523, r3485, r3503;
}
{
sub.f16x2 r3526, r3488, r3497;
}
{
add.f16x2 r3529, r3530, r3531;
}
{
add.f16x2 r3532, r3533, r3534;
}
{
sub.f16x2 r3535, r3530, r3531;
}
{
sub.f16x2 r3538, r3533, r3534;
}
{
add.f16x2 r3541, r3542, r3543;
}
{
add.f16x2 r3544, r3545, r3546;
}
{
sub.f16x2 r3547, r3542, r3543;
}
{
sub.f16x2 r3550, r3545, r3546;
}
{
neg.f16x2 r3553, r3550;
}
{
add.f16x2 r3555, r3529, r3541;
}
{
add.f16x2 r3558, r3532, r3544;
}
{
sub.f16x2 r3561, r3529, r3541;
}
{
sub.f16x2 r3564, r3532, r3544;
}
{
add.f16x2 r3567, r3535, r3553;
}
{
add.f16x2 r3570, r3538, r3547;
}
{
sub.f16x2 r3573, r3535, r3553;
}
{
sub.f16x2 r3576, r3538, r3547;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3579, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3580, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r3583, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3584, {low, high};
}
{
mul.f16x2 r3593, r3567, r3579;
}
{
mul.f16x2 r3596, r3570, r3580;
}
{
sub.f16x2 r3599, r3593, r3596;
}
{
mul.f16x2 r3602, r3567, r3580;
}
{
fma.rn.f16x2 r3605, r3570, r3579, r3602;
}
{
neg.f16x2 r3609, r3564;
}
{
mul.f16x2 r3611, r3573, r3583;
}
{
mul.f16x2 r3614, r3576, r3584;
}
{
sub.f16x2 r3617, r3611, r3614;
}
{
mul.f16x2 r3620, r3573, r3584;
}
{
fma.rn.f16x2 r3623, r3576, r3583, r3620;
}
{
add.f16x2 r3627, r3505, r3555;
}
{
add.f16x2 r3630, r3508, r3558;
}
{
sub.f16x2 r3633, r3505, r3555;
}
{
sub.f16x2 r3636, r3508, r3558;
}
{
add.f16x2 r3639, r3517, r3599;
}
{
add.f16x2 r3642, r3520, r3605;
}
{
sub.f16x2 r3645, r3517, r3599;
}
{
sub.f16x2 r3648, r3520, r3605;
}
{
add.f16x2 r3651, r3511, r3609;
}
{
add.f16x2 r3654, r3514, r3561;
}
{
sub.f16x2 r3657, r3511, r3609;
}
{
sub.f16x2 r3660, r3514, r3561;
}
{
add.f16x2 r3663, r3523, r3617;
}
{
add.f16x2 r3666, r3526, r3623;
}
{
sub.f16x2 r3669, r3523, r3617;
}
{
sub.f16x2 r3672, r3526, r3623;
}
{
add.f16x2 r3675, r3676, r3677;
}
{
add.f16x2 r3678, r3679, r3680;
}
{
sub.f16x2 r3681, r3676, r3677;
}
{
sub.f16x2 r3684, r3679, r3680;
}
{
add.f16x2 r3687, r3688, r3689;
}
{
add.f16x2 r3690, r3691, r3692;
}
{
sub.f16x2 r3693, r3688, r3689;
}
{
sub.f16x2 r3696, r3691, r3692;
}
{
neg.f16x2 r3699, r3696;
}
{
add.f16x2 r3701, r3675, r3687;
}
{
add.f16x2 r3704, r3678, r3690;
}
{
sub.f16x2 r3707, r3675, r3687;
}
{
sub.f16x2 r3710, r3678, r3690;
}
{
add.f16x2 r3713, r3681, r3699;
}
{
add.f16x2 r3716, r3684, r3693;
}
{
sub.f16x2 r3719, r3681, r3699;
}
{
sub.f16x2 r3722, r3684, r3693;
}
{
add.f16x2 r3725, r3726, r3727;
}
{
add.f16x2 r3728, r3729, r3730;
}
{
sub.f16x2 r3731, r3726, r3727;
}
{
sub.f16x2 r3734, r3729, r3730;
}
{
add.f16x2 r3737, r3738, r3739;
}
{
add.f16x2 r3740, r3741, r3742;
}
{
sub.f16x2 r3743, r3738, r3739;
}
{
sub.f16x2 r3746, r3741, r3742;
}
{
neg.f16x2 r3749, r3746;
}
{
add.f16x2 r3751, r3725, r3737;
}
{
add.f16x2 r3754, r3728, r3740;
}
{
sub.f16x2 r3757, r3725, r3737;
}
{
sub.f16x2 r3760, r3728, r3740;
}
{
add.f16x2 r3763, r3731, r3749;
}
{
add.f16x2 r3766, r3734, r3743;
}
{
sub.f16x2 r3769, r3731, r3749;
}
{
sub.f16x2 r3772, r3734, r3743;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3775, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3776, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r3779, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3780, {low, high};
}
{
mul.f16x2 r3789, r3763, r3775;
}
{
mul.f16x2 r3792, r3766, r3776;
}
{
sub.f16x2 r3795, r3789, r3792;
}
{
mul.f16x2 r3798, r3763, r3776;
}
{
fma.rn.f16x2 r3801, r3766, r3775, r3798;
}
{
neg.f16x2 r3805, r3760;
}
{
mul.f16x2 r3807, r3769, r3779;
}
{
mul.f16x2 r3810, r3772, r3780;
}
{
sub.f16x2 r3813, r3807, r3810;
}
{
mul.f16x2 r3816, r3769, r3780;
}
{
fma.rn.f16x2 r3819, r3772, r3779, r3816;
}
{
add.f16x2 r3823, r3701, r3751;
}
{
add.f16x2 r3826, r3704, r3754;
}
{
sub.f16x2 r3829, r3701, r3751;
}
{
sub.f16x2 r3832, r3704, r3754;
}
{
add.f16x2 r3835, r3713, r3795;
}
{
add.f16x2 r3838, r3716, r3801;
}
{
sub.f16x2 r3841, r3713, r3795;
}
{
sub.f16x2 r3844, r3716, r3801;
}
{
add.f16x2 r3847, r3707, r3805;
}
{
add.f16x2 r3850, r3710, r3757;
}
{
sub.f16x2 r3853, r3707, r3805;
}
{
sub.f16x2 r3856, r3710, r3757;
}
{
add.f16x2 r3859, r3719, r3813;
}
{
add.f16x2 r3862, r3722, r3819;
}
{
sub.f16x2 r3865, r3719, r3813;
}
{
sub.f16x2 r3868, r3722, r3819;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r3871, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r3872, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3873, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3874, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r3875, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r3876, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r3879, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r3880, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r3881, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r3882, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1042;
cvt.rn.f16.f32 high, f1042;
mov.b32 r3883, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r3884, {low, high};
}
{
mul.f16x2 r3901, r3835, r3871;
}
{
mul.f16x2 r3904, r3838, r3872;
}
{
sub.f16x2 r3907, r3901, r3904;
}
{
mul.f16x2 r3910, r3835, r3872;
}
{
fma.rn.f16x2 r3913, r3838, r3871, r3910;
}
{
mul.f16x2 r3917, r3847, r3873;
}
{
mul.f16x2 r3920, r3850, r3874;
}
{
sub.f16x2 r3923, r3917, r3920;
}
{
mul.f16x2 r3926, r3847, r3874;
}
{
fma.rn.f16x2 r3929, r3850, r3873, r3926;
}
{
mul.f16x2 r3933, r3859, r3875;
}
{
mul.f16x2 r3936, r3862, r3876;
}
{
sub.f16x2 r3939, r3933, r3936;
}
{
mul.f16x2 r3942, r3859, r3876;
}
{
fma.rn.f16x2 r3945, r3862, r3875, r3942;
}
{
neg.f16x2 r3949, r3832;
}
{
mul.f16x2 r3951, r3841, r3879;
}
{
mul.f16x2 r3954, r3844, r3880;
}
{
sub.f16x2 r3957, r3951, r3954;
}
{
mul.f16x2 r3960, r3841, r3880;
}
{
fma.rn.f16x2 r3963, r3844, r3879, r3960;
}
{
mul.f16x2 r3967, r3853, r3881;
}
{
mul.f16x2 r3970, r3856, r3882;
}
{
sub.f16x2 r3973, r3967, r3970;
}
{
mul.f16x2 r3976, r3853, r3882;
}
{
fma.rn.f16x2 r3979, r3856, r3881, r3976;
}
{
mul.f16x2 r3983, r3865, r3883;
}
{
mul.f16x2 r3986, r3868, r3884;
}
{
sub.f16x2 r3989, r3983, r3986;
}
{
mul.f16x2 r3992, r3865, r3884;
}
{
fma.rn.f16x2 r3995, r3868, r3883, r3992;
}
{
add.f16x2 r3999, r3627, r3823;
}
{
add.f16x2 r4002, r3630, r3826;
}
{
sub.f16x2 r4005, r3627, r3823;
}
{
sub.f16x2 r4008, r3630, r3826;
}
{
add.f16x2 r4011, r3639, r3907;
}
{
add.f16x2 r4014, r3642, r3913;
}
{
sub.f16x2 r4017, r3639, r3907;
}
{
sub.f16x2 r4020, r3642, r3913;
}
{
add.f16x2 r4023, r3651, r3923;
}
{
add.f16x2 r4026, r3654, r3929;
}
{
sub.f16x2 r4029, r3651, r3923;
}
{
sub.f16x2 r4032, r3654, r3929;
}
{
add.f16x2 r4035, r3663, r3939;
}
{
add.f16x2 r4038, r3666, r3945;
}
{
sub.f16x2 r4041, r3663, r3939;
}
{
sub.f16x2 r4044, r3666, r3945;
}
{
add.f16x2 r4047, r3633, r3949;
}
{
add.f16x2 r4050, r3636, r3829;
}
{
sub.f16x2 r4053, r3633, r3949;
}
{
sub.f16x2 r4056, r3636, r3829;
}
{
add.f16x2 r4059, r3645, r3957;
}
{
add.f16x2 r4062, r3648, r3963;
}
{
sub.f16x2 r4065, r3645, r3957;
}
{
sub.f16x2 r4068, r3648, r3963;
}
{
add.f16x2 r4071, r3657, r3973;
}
{
add.f16x2 r4074, r3660, r3979;
}
{
sub.f16x2 r4077, r3657, r3973;
}
{
sub.f16x2 r4080, r3660, r3979;
}
{
add.f16x2 r4083, r3669, r3989;
}
{
add.f16x2 r4086, r3672, r3995;
}
{
sub.f16x2 r4089, r3669, r3989;
}
{
sub.f16x2 r4092, r3672, r3995;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f690;
cvt.rn.f16.f32 high, f690;
mov.b32 r4095, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f714;
cvt.rn.f16.f32 high, f714;
mov.b32 r4096, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r4097, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r4098, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f698;
cvt.rn.f16.f32 high, f698;
mov.b32 r4099, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f706;
cvt.rn.f16.f32 high, f706;
mov.b32 r4100, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r4101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r4102, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f706;
cvt.rn.f16.f32 high, f706;
mov.b32 r4103, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f698;
cvt.rn.f16.f32 high, f698;
mov.b32 r4104, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r4105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r4106, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f714;
cvt.rn.f16.f32 high, f714;
mov.b32 r4107, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f690;
cvt.rn.f16.f32 high, f690;
mov.b32 r4108, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f688;
cvt.rn.f16.f32 high, f688;
mov.b32 r4111, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f690;
cvt.rn.f16.f32 high, f690;
mov.b32 r4112, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r4113, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r4114, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f696;
cvt.rn.f16.f32 high, f696;
mov.b32 r4115, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f698;
cvt.rn.f16.f32 high, f698;
mov.b32 r4116, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r4117, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r4118, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f704;
cvt.rn.f16.f32 high, f704;
mov.b32 r4119, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f706;
cvt.rn.f16.f32 high, f706;
mov.b32 r4120, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1042;
cvt.rn.f16.f32 high, f1042;
mov.b32 r4121, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r4122, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f712;
cvt.rn.f16.f32 high, f712;
mov.b32 r4123, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f714;
cvt.rn.f16.f32 high, f714;
mov.b32 r4124, {low, high};
}
{
mul.f16x2 r4157, r4011, r4095;
}
{
mul.f16x2 r4160, r4014, r4096;
}
{
sub.f16x2 r4163, r4157, r4160;
}
{
mul.f16x2 r4166, r4011, r4096;
}
{
fma.rn.f16x2 r4169, r4014, r4095, r4166;
}
{
mul.f16x2 r4173, r4023, r4097;
}
{
mul.f16x2 r4176, r4026, r4098;
}
{
sub.f16x2 r4179, r4173, r4176;
}
{
mul.f16x2 r4182, r4023, r4098;
}
{
fma.rn.f16x2 r4185, r4026, r4097, r4182;
}
{
mul.f16x2 r4189, r4035, r4099;
}
{
mul.f16x2 r4192, r4038, r4100;
}
{
sub.f16x2 r4195, r4189, r4192;
}
{
mul.f16x2 r4198, r4035, r4100;
}
{
fma.rn.f16x2 r4201, r4038, r4099, r4198;
}
{
mul.f16x2 r4205, r4047, r4101;
}
{
mul.f16x2 r4208, r4050, r4102;
}
{
sub.f16x2 r4211, r4205, r4208;
}
{
mul.f16x2 r4214, r4047, r4102;
}
{
fma.rn.f16x2 r4217, r4050, r4101, r4214;
}
{
mul.f16x2 r4221, r4059, r4103;
}
{
mul.f16x2 r4224, r4062, r4104;
}
{
sub.f16x2 r4227, r4221, r4224;
}
{
mul.f16x2 r4230, r4059, r4104;
}
{
fma.rn.f16x2 r4233, r4062, r4103, r4230;
}
{
mul.f16x2 r4237, r4071, r4105;
}
{
mul.f16x2 r4240, r4074, r4106;
}
{
sub.f16x2 r4243, r4237, r4240;
}
{
mul.f16x2 r4246, r4071, r4106;
}
{
fma.rn.f16x2 r4249, r4074, r4105, r4246;
}
{
mul.f16x2 r4253, r4083, r4107;
}
{
mul.f16x2 r4256, r4086, r4108;
}
{
sub.f16x2 r4259, r4253, r4256;
}
{
mul.f16x2 r4262, r4083, r4108;
}
{
fma.rn.f16x2 r4265, r4086, r4107, r4262;
}
{
neg.f16x2 r4269, r4008;
}
{
mul.f16x2 r4271, r4017, r4111;
}
{
mul.f16x2 r4274, r4020, r4112;
}
{
sub.f16x2 r4277, r4271, r4274;
}
{
mul.f16x2 r4280, r4017, r4112;
}
{
fma.rn.f16x2 r4283, r4020, r4111, r4280;
}
{
mul.f16x2 r4287, r4029, r4113;
}
{
mul.f16x2 r4290, r4032, r4114;
}
{
sub.f16x2 r4293, r4287, r4290;
}
{
mul.f16x2 r4296, r4029, r4114;
}
{
fma.rn.f16x2 r4299, r4032, r4113, r4296;
}
{
mul.f16x2 r4303, r4041, r4115;
}
{
mul.f16x2 r4306, r4044, r4116;
}
{
sub.f16x2 r4309, r4303, r4306;
}
{
mul.f16x2 r4312, r4041, r4116;
}
{
fma.rn.f16x2 r4315, r4044, r4115, r4312;
}
{
mul.f16x2 r4319, r4053, r4117;
}
{
mul.f16x2 r4322, r4056, r4118;
}
{
sub.f16x2 r4325, r4319, r4322;
}
{
mul.f16x2 r4328, r4053, r4118;
}
{
fma.rn.f16x2 r4331, r4056, r4117, r4328;
}
{
mul.f16x2 r4335, r4065, r4119;
}
{
mul.f16x2 r4338, r4068, r4120;
}
{
sub.f16x2 r4341, r4335, r4338;
}
{
mul.f16x2 r4344, r4065, r4120;
}
{
fma.rn.f16x2 r4347, r4068, r4119, r4344;
}
{
mul.f16x2 r4351, r4077, r4121;
}
{
mul.f16x2 r4354, r4080, r4122;
}
{
sub.f16x2 r4357, r4351, r4354;
}
{
mul.f16x2 r4360, r4077, r4122;
}
{
fma.rn.f16x2 r4363, r4080, r4121, r4360;
}
{
mul.f16x2 r4367, r4089, r4123;
}
{
mul.f16x2 r4370, r4092, r4124;
}
{
sub.f16x2 r4373, r4367, r4370;
}
{
mul.f16x2 r4376, r4089, r4124;
}
{
fma.rn.f16x2 r4379, r4092, r4123, r4376;
}
{
add.f16x2 r4383, r3383, r3999;
}
{
add.f16x2 r4386, r3386, r4002;
}
{
sub.f16x2 r4389, r3383, r3999;
}
{
sub.f16x2 r4392, r3386, r4002;
}
{
add.f16x2 r4395, r3395, r4163;
}
{
add.f16x2 r4398, r3398, r4169;
}
{
sub.f16x2 r4401, r3395, r4163;
}
{
sub.f16x2 r4404, r3398, r4169;
}
{
add.f16x2 r4407, r3407, r4179;
}
{
add.f16x2 r4410, r3410, r4185;
}
{
sub.f16x2 r4413, r3407, r4179;
}
{
sub.f16x2 r4416, r3410, r4185;
}
{
add.f16x2 r4419, r3419, r4195;
}
{
add.f16x2 r4422, r3422, r4201;
}
{
sub.f16x2 r4425, r3419, r4195;
}
{
sub.f16x2 r4428, r3422, r4201;
}
{
add.f16x2 r4431, r3431, r4211;
}
{
add.f16x2 r4434, r3434, r4217;
}
{
sub.f16x2 r4437, r3431, r4211;
}
{
sub.f16x2 r4440, r3434, r4217;
}
{
add.f16x2 r4443, r3443, r4227;
}
{
add.f16x2 r4446, r3446, r4233;
}
{
sub.f16x2 r4449, r3443, r4227;
}
{
sub.f16x2 r4452, r3446, r4233;
}
{
add.f16x2 r4455, r3455, r4243;
}
{
add.f16x2 r4458, r3458, r4249;
}
{
sub.f16x2 r4461, r3455, r4243;
}
{
sub.f16x2 r4464, r3458, r4249;
}
{
add.f16x2 r4467, r3467, r4259;
}
{
add.f16x2 r4470, r3470, r4265;
}
{
sub.f16x2 r4473, r3467, r4259;
}
{
sub.f16x2 r4476, r3470, r4265;
}
{
add.f16x2 r4479, r3389, r4269;
}
{
add.f16x2 r4482, r3392, r4005;
}
{
sub.f16x2 r4485, r3389, r4269;
}
{
sub.f16x2 r4488, r3392, r4005;
}
{
add.f16x2 r4491, r3401, r4277;
}
{
add.f16x2 r4494, r3404, r4283;
}
{
sub.f16x2 r4497, r3401, r4277;
}
{
sub.f16x2 r4500, r3404, r4283;
}
{
add.f16x2 r4503, r3413, r4293;
}
{
add.f16x2 r4506, r3416, r4299;
}
{
sub.f16x2 r4509, r3413, r4293;
}
{
sub.f16x2 r4512, r3416, r4299;
}
{
add.f16x2 r4515, r3425, r4309;
}
{
add.f16x2 r4518, r3428, r4315;
}
{
sub.f16x2 r4521, r3425, r4309;
}
{
sub.f16x2 r4524, r3428, r4315;
}
{
add.f16x2 r4527, r3437, r4325;
}
{
add.f16x2 r4530, r3440, r4331;
}
{
sub.f16x2 r4533, r3437, r4325;
}
{
sub.f16x2 r4536, r3440, r4331;
}
{
add.f16x2 r4539, r3449, r4341;
}
{
add.f16x2 r4542, r3452, r4347;
}
{
sub.f16x2 r4545, r3449, r4341;
}
{
sub.f16x2 r4548, r3452, r4347;
}
{
add.f16x2 r4551, r3461, r4357;
}
{
add.f16x2 r4554, r3464, r4363;
}
{
sub.f16x2 r4557, r3461, r4357;
}
{
sub.f16x2 r4560, r3464, r4363;
}
{
add.f16x2 r4563, r3473, r4373;
}
{
add.f16x2 r4566, r3476, r4379;
}
{
sub.f16x2 r4569, r3473, r4373;
}
{
sub.f16x2 r4572, r3476, r4379;
}
and.b32 r6969, r6961, 480;
bfe.u32 r6970, r6961, 5, 4;
shl.b32 r6971, r6961, 2;
and.b32 r6972, r6971, 124;
add.s32 r6973, r6965, r6972;
cvt.rn.f32.u32 f1080, r6970;
mul.f32 f1081, f1080, 0f3C490FDB;
cos.approx.f32 f779, f1081;
sin.approx.f32 f1082, f1081;
neg.f32 f780, f1082;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f779;
cvt.rn.f16.f32 high, f780;
mov.b32 r4575, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4578, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4580, {high, high};
}
{
mul.f16x2 r4582, r4398, r4580;
}
{
fma.rn.f16x2 r4585, r4395, r4578, r4582;
}
{
mul.f16x2 r4589, r4395, r4580;
}
{
neg.f16x2 r4592, r4589;
}
{
fma.rn.f16x2 r4594, r4398, r4578, r4592;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4598, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4600, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4602, {low, high};
}
{
mul.f16x2 r4603, r4600, r4602;
}
{
mul.f16x2 r4606, r4575, r4598;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4609, {high, low};
}
{
fma.rn.f16x2 r4611, r4603, r4609, r4606;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4611;
mov.b32 r4615, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4611;
mov.b32 r4617, {high, high};
}
{
mul.f16x2 r4619, r4410, r4617;
}
{
fma.rn.f16x2 r4622, r4407, r4615, r4619;
}
{
mul.f16x2 r4626, r4407, r4617;
}
{
neg.f16x2 r4629, r4626;
}
{
fma.rn.f16x2 r4631, r4410, r4615, r4629;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4635, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4637, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4639, {low, high};
}
{
mul.f16x2 r4640, r4637, r4639;
}
{
mul.f16x2 r4643, r4611, r4635;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4611;
mov.b32 r4646, {high, low};
}
{
fma.rn.f16x2 r4648, r4640, r4646, r4643;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4648;
mov.b32 r4652, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4648;
mov.b32 r4654, {high, high};
}
{
mul.f16x2 r4656, r4422, r4654;
}
{
fma.rn.f16x2 r4659, r4419, r4652, r4656;
}
{
mul.f16x2 r4663, r4419, r4654;
}
{
neg.f16x2 r4666, r4663;
}
{
fma.rn.f16x2 r4668, r4422, r4652, r4666;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4672, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4674, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4676, {low, high};
}
{
mul.f16x2 r4677, r4674, r4676;
}
{
mul.f16x2 r4680, r4648, r4672;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4648;
mov.b32 r4683, {high, low};
}
{
fma.rn.f16x2 r4685, r4677, r4683, r4680;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4685;
mov.b32 r4689, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4685;
mov.b32 r4691, {high, high};
}
{
mul.f16x2 r4693, r4434, r4691;
}
{
fma.rn.f16x2 r4696, r4431, r4689, r4693;
}
{
mul.f16x2 r4700, r4431, r4691;
}
{
neg.f16x2 r4703, r4700;
}
{
fma.rn.f16x2 r4705, r4434, r4689, r4703;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4709, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4711, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4713, {low, high};
}
{
mul.f16x2 r4714, r4711, r4713;
}
{
mul.f16x2 r4717, r4685, r4709;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4685;
mov.b32 r4720, {high, low};
}
{
fma.rn.f16x2 r4722, r4714, r4720, r4717;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4722;
mov.b32 r4726, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4722;
mov.b32 r4728, {high, high};
}
{
mul.f16x2 r4730, r4446, r4728;
}
{
fma.rn.f16x2 r4733, r4443, r4726, r4730;
}
{
mul.f16x2 r4737, r4443, r4728;
}
{
neg.f16x2 r4740, r4737;
}
{
fma.rn.f16x2 r4742, r4446, r4726, r4740;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4746, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4748, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4750, {low, high};
}
{
mul.f16x2 r4751, r4748, r4750;
}
{
mul.f16x2 r4754, r4722, r4746;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4722;
mov.b32 r4757, {high, low};
}
{
fma.rn.f16x2 r4759, r4751, r4757, r4754;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4759;
mov.b32 r4763, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4759;
mov.b32 r4765, {high, high};
}
{
mul.f16x2 r4767, r4458, r4765;
}
{
fma.rn.f16x2 r4770, r4455, r4763, r4767;
}
{
mul.f16x2 r4774, r4455, r4765;
}
{
neg.f16x2 r4777, r4774;
}
{
fma.rn.f16x2 r4779, r4458, r4763, r4777;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4783, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4785, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4787, {low, high};
}
{
mul.f16x2 r4788, r4785, r4787;
}
{
mul.f16x2 r4791, r4759, r4783;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4759;
mov.b32 r4794, {high, low};
}
{
fma.rn.f16x2 r4796, r4788, r4794, r4791;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4796;
mov.b32 r4800, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4796;
mov.b32 r4802, {high, high};
}
{
mul.f16x2 r4804, r4470, r4802;
}
{
fma.rn.f16x2 r4807, r4467, r4800, r4804;
}
{
mul.f16x2 r4811, r4467, r4802;
}
{
neg.f16x2 r4814, r4811;
}
{
fma.rn.f16x2 r4816, r4470, r4800, r4814;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4820, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4822, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4824, {low, high};
}
{
mul.f16x2 r4825, r4822, r4824;
}
{
mul.f16x2 r4828, r4796, r4820;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4796;
mov.b32 r4831, {high, low};
}
{
fma.rn.f16x2 r4833, r4825, r4831, r4828;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4833;
mov.b32 r4837, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4833;
mov.b32 r4839, {high, high};
}
{
mul.f16x2 r4841, r4482, r4839;
}
{
fma.rn.f16x2 r4844, r4479, r4837, r4841;
}
{
mul.f16x2 r4848, r4479, r4839;
}
{
neg.f16x2 r4851, r4848;
}
{
fma.rn.f16x2 r4853, r4482, r4837, r4851;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4857, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4859, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4861, {low, high};
}
{
mul.f16x2 r4862, r4859, r4861;
}
{
mul.f16x2 r4865, r4833, r4857;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4833;
mov.b32 r4868, {high, low};
}
{
fma.rn.f16x2 r4870, r4862, r4868, r4865;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4870;
mov.b32 r4874, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4870;
mov.b32 r4876, {high, high};
}
{
mul.f16x2 r4878, r4494, r4876;
}
{
fma.rn.f16x2 r4881, r4491, r4874, r4878;
}
{
mul.f16x2 r4885, r4491, r4876;
}
{
neg.f16x2 r4888, r4885;
}
{
fma.rn.f16x2 r4890, r4494, r4874, r4888;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4894, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4896, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4898, {low, high};
}
{
mul.f16x2 r4899, r4896, r4898;
}
{
mul.f16x2 r4902, r4870, r4894;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4870;
mov.b32 r4905, {high, low};
}
{
fma.rn.f16x2 r4907, r4899, r4905, r4902;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4907;
mov.b32 r4911, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4907;
mov.b32 r4913, {high, high};
}
{
mul.f16x2 r4915, r4506, r4913;
}
{
fma.rn.f16x2 r4918, r4503, r4911, r4915;
}
{
mul.f16x2 r4922, r4503, r4913;
}
{
neg.f16x2 r4925, r4922;
}
{
fma.rn.f16x2 r4927, r4506, r4911, r4925;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4931, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4933, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4935, {low, high};
}
{
mul.f16x2 r4936, r4933, r4935;
}
{
mul.f16x2 r4939, r4907, r4931;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4907;
mov.b32 r4942, {high, low};
}
{
fma.rn.f16x2 r4944, r4936, r4942, r4939;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4944;
mov.b32 r4948, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4944;
mov.b32 r4950, {high, high};
}
{
mul.f16x2 r4952, r4518, r4950;
}
{
fma.rn.f16x2 r4955, r4515, r4948, r4952;
}
{
mul.f16x2 r4959, r4515, r4950;
}
{
neg.f16x2 r4962, r4959;
}
{
fma.rn.f16x2 r4964, r4518, r4948, r4962;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4968, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r4970, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r4972, {low, high};
}
{
mul.f16x2 r4973, r4970, r4972;
}
{
mul.f16x2 r4976, r4944, r4968;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4944;
mov.b32 r4979, {high, low};
}
{
fma.rn.f16x2 r4981, r4973, r4979, r4976;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4981;
mov.b32 r4985, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4981;
mov.b32 r4987, {high, high};
}
{
mul.f16x2 r4989, r4530, r4987;
}
{
fma.rn.f16x2 r4992, r4527, r4985, r4989;
}
{
mul.f16x2 r4996, r4527, r4987;
}
{
neg.f16x2 r4999, r4996;
}
{
fma.rn.f16x2 r5001, r4530, r4985, r4999;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5005, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5007, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5009, {low, high};
}
{
mul.f16x2 r5010, r5007, r5009;
}
{
mul.f16x2 r5013, r4981, r5005;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4981;
mov.b32 r5016, {high, low};
}
{
fma.rn.f16x2 r5018, r5010, r5016, r5013;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5018;
mov.b32 r5022, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5018;
mov.b32 r5024, {high, high};
}
{
mul.f16x2 r5026, r4542, r5024;
}
{
fma.rn.f16x2 r5029, r4539, r5022, r5026;
}
{
mul.f16x2 r5033, r4539, r5024;
}
{
neg.f16x2 r5036, r5033;
}
{
fma.rn.f16x2 r5038, r4542, r5022, r5036;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5042, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5044, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5046, {low, high};
}
{
mul.f16x2 r5047, r5044, r5046;
}
{
mul.f16x2 r5050, r5018, r5042;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5018;
mov.b32 r5053, {high, low};
}
{
fma.rn.f16x2 r5055, r5047, r5053, r5050;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5055;
mov.b32 r5059, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5055;
mov.b32 r5061, {high, high};
}
{
mul.f16x2 r5063, r4554, r5061;
}
{
fma.rn.f16x2 r5066, r4551, r5059, r5063;
}
{
mul.f16x2 r5070, r4551, r5061;
}
{
neg.f16x2 r5073, r5070;
}
{
fma.rn.f16x2 r5075, r4554, r5059, r5073;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5079, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5081, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5083, {low, high};
}
{
mul.f16x2 r5084, r5081, r5083;
}
{
mul.f16x2 r5087, r5055, r5079;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5055;
mov.b32 r5090, {high, low};
}
{
fma.rn.f16x2 r5092, r5084, r5090, r5087;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5092;
mov.b32 r5096, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5092;
mov.b32 r5098, {high, high};
}
{
mul.f16x2 r5100, r4566, r5098;
}
{
fma.rn.f16x2 r5103, r4563, r5096, r5100;
}
{
mul.f16x2 r5107, r4563, r5098;
}
{
neg.f16x2 r5110, r5107;
}
{
fma.rn.f16x2 r5112, r4566, r5096, r5110;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5116, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5118, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5120, {low, high};
}
{
mul.f16x2 r5121, r5118, r5120;
}
{
mul.f16x2 r5124, r5092, r5116;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5092;
mov.b32 r5127, {high, low};
}
{
fma.rn.f16x2 r5129, r5121, r5127, r5124;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5129;
mov.b32 r5133, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5129;
mov.b32 r5135, {high, high};
}
{
mul.f16x2 r5137, r4392, r5135;
}
{
fma.rn.f16x2 r5140, r4389, r5133, r5137;
}
{
mul.f16x2 r5144, r4389, r5135;
}
{
neg.f16x2 r5147, r5144;
}
{
fma.rn.f16x2 r5149, r4392, r5133, r5147;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5153, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5155, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5157, {low, high};
}
{
mul.f16x2 r5158, r5155, r5157;
}
{
mul.f16x2 r5161, r5129, r5153;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5129;
mov.b32 r5164, {high, low};
}
{
fma.rn.f16x2 r5166, r5158, r5164, r5161;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5166;
mov.b32 r5170, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5166;
mov.b32 r5172, {high, high};
}
{
mul.f16x2 r5174, r4404, r5172;
}
{
fma.rn.f16x2 r5177, r4401, r5170, r5174;
}
{
mul.f16x2 r5181, r4401, r5172;
}
{
neg.f16x2 r5184, r5181;
}
{
fma.rn.f16x2 r5186, r4404, r5170, r5184;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5190, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5192, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5194, {low, high};
}
{
mul.f16x2 r5195, r5192, r5194;
}
{
mul.f16x2 r5198, r5166, r5190;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5166;
mov.b32 r5201, {high, low};
}
{
fma.rn.f16x2 r5203, r5195, r5201, r5198;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5203;
mov.b32 r5207, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5203;
mov.b32 r5209, {high, high};
}
{
mul.f16x2 r5211, r4416, r5209;
}
{
fma.rn.f16x2 r5214, r4413, r5207, r5211;
}
{
mul.f16x2 r5218, r4413, r5209;
}
{
neg.f16x2 r5221, r5218;
}
{
fma.rn.f16x2 r5223, r4416, r5207, r5221;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5227, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5229, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5231, {low, high};
}
{
mul.f16x2 r5232, r5229, r5231;
}
{
mul.f16x2 r5235, r5203, r5227;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5203;
mov.b32 r5238, {high, low};
}
{
fma.rn.f16x2 r5240, r5232, r5238, r5235;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5240;
mov.b32 r5244, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5240;
mov.b32 r5246, {high, high};
}
{
mul.f16x2 r5248, r4428, r5246;
}
{
fma.rn.f16x2 r5251, r4425, r5244, r5248;
}
{
mul.f16x2 r5255, r4425, r5246;
}
{
neg.f16x2 r5258, r5255;
}
{
fma.rn.f16x2 r5260, r4428, r5244, r5258;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5264, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5266, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5268, {low, high};
}
{
mul.f16x2 r5269, r5266, r5268;
}
{
mul.f16x2 r5272, r5240, r5264;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5240;
mov.b32 r5275, {high, low};
}
{
fma.rn.f16x2 r5277, r5269, r5275, r5272;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5277;
mov.b32 r5281, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5277;
mov.b32 r5283, {high, high};
}
{
mul.f16x2 r5285, r4440, r5283;
}
{
fma.rn.f16x2 r5288, r4437, r5281, r5285;
}
{
mul.f16x2 r5292, r4437, r5283;
}
{
neg.f16x2 r5295, r5292;
}
{
fma.rn.f16x2 r5297, r4440, r5281, r5295;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5301, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5303, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5305, {low, high};
}
{
mul.f16x2 r5306, r5303, r5305;
}
{
mul.f16x2 r5309, r5277, r5301;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5277;
mov.b32 r5312, {high, low};
}
{
fma.rn.f16x2 r5314, r5306, r5312, r5309;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5314;
mov.b32 r5318, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5314;
mov.b32 r5320, {high, high};
}
{
mul.f16x2 r5322, r4452, r5320;
}
{
fma.rn.f16x2 r5325, r4449, r5318, r5322;
}
{
mul.f16x2 r5329, r4449, r5320;
}
{
neg.f16x2 r5332, r5329;
}
{
fma.rn.f16x2 r5334, r4452, r5318, r5332;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5338, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5340, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5342, {low, high};
}
{
mul.f16x2 r5343, r5340, r5342;
}
{
mul.f16x2 r5346, r5314, r5338;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5314;
mov.b32 r5349, {high, low};
}
{
fma.rn.f16x2 r5351, r5343, r5349, r5346;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5351;
mov.b32 r5355, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5351;
mov.b32 r5357, {high, high};
}
{
mul.f16x2 r5359, r4464, r5357;
}
{
fma.rn.f16x2 r5362, r4461, r5355, r5359;
}
{
mul.f16x2 r5366, r4461, r5357;
}
{
neg.f16x2 r5369, r5366;
}
{
fma.rn.f16x2 r5371, r4464, r5355, r5369;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5375, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5377, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5379, {low, high};
}
{
mul.f16x2 r5380, r5377, r5379;
}
{
mul.f16x2 r5383, r5351, r5375;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5351;
mov.b32 r5386, {high, low};
}
{
fma.rn.f16x2 r5388, r5380, r5386, r5383;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5388;
mov.b32 r5392, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5388;
mov.b32 r5394, {high, high};
}
{
mul.f16x2 r5396, r4476, r5394;
}
{
fma.rn.f16x2 r5399, r4473, r5392, r5396;
}
{
mul.f16x2 r5403, r4473, r5394;
}
{
neg.f16x2 r5406, r5403;
}
{
fma.rn.f16x2 r5408, r4476, r5392, r5406;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5412, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5414, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5416, {low, high};
}
{
mul.f16x2 r5417, r5414, r5416;
}
{
mul.f16x2 r5420, r5388, r5412;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5388;
mov.b32 r5423, {high, low};
}
{
fma.rn.f16x2 r5425, r5417, r5423, r5420;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5425;
mov.b32 r5429, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5425;
mov.b32 r5431, {high, high};
}
{
mul.f16x2 r5433, r4488, r5431;
}
{
fma.rn.f16x2 r5436, r4485, r5429, r5433;
}
{
mul.f16x2 r5440, r4485, r5431;
}
{
neg.f16x2 r5443, r5440;
}
{
fma.rn.f16x2 r5445, r4488, r5429, r5443;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5449, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5451, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5453, {low, high};
}
{
mul.f16x2 r5454, r5451, r5453;
}
{
mul.f16x2 r5457, r5425, r5449;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5425;
mov.b32 r5460, {high, low};
}
{
fma.rn.f16x2 r5462, r5454, r5460, r5457;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5462;
mov.b32 r5466, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5462;
mov.b32 r5468, {high, high};
}
{
mul.f16x2 r5470, r4500, r5468;
}
{
fma.rn.f16x2 r5473, r4497, r5466, r5470;
}
{
mul.f16x2 r5477, r4497, r5468;
}
{
neg.f16x2 r5480, r5477;
}
{
fma.rn.f16x2 r5482, r4500, r5466, r5480;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5486, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5488, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5490, {low, high};
}
{
mul.f16x2 r5491, r5488, r5490;
}
{
mul.f16x2 r5494, r5462, r5486;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5462;
mov.b32 r5497, {high, low};
}
{
fma.rn.f16x2 r5499, r5491, r5497, r5494;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5499;
mov.b32 r5503, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5499;
mov.b32 r5505, {high, high};
}
{
mul.f16x2 r5507, r4512, r5505;
}
{
fma.rn.f16x2 r5510, r4509, r5503, r5507;
}
{
mul.f16x2 r5514, r4509, r5505;
}
{
neg.f16x2 r5517, r5514;
}
{
fma.rn.f16x2 r5519, r4512, r5503, r5517;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5523, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5525, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5527, {low, high};
}
{
mul.f16x2 r5528, r5525, r5527;
}
{
mul.f16x2 r5531, r5499, r5523;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5499;
mov.b32 r5534, {high, low};
}
{
fma.rn.f16x2 r5536, r5528, r5534, r5531;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5536;
mov.b32 r5540, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5536;
mov.b32 r5542, {high, high};
}
{
mul.f16x2 r5544, r4524, r5542;
}
{
fma.rn.f16x2 r5547, r4521, r5540, r5544;
}
{
mul.f16x2 r5551, r4521, r5542;
}
{
neg.f16x2 r5554, r5551;
}
{
fma.rn.f16x2 r5556, r4524, r5540, r5554;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5560, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5562, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5564, {low, high};
}
{
mul.f16x2 r5565, r5562, r5564;
}
{
mul.f16x2 r5568, r5536, r5560;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5536;
mov.b32 r5571, {high, low};
}
{
fma.rn.f16x2 r5573, r5565, r5571, r5568;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5573;
mov.b32 r5577, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5573;
mov.b32 r5579, {high, high};
}
{
mul.f16x2 r5581, r4536, r5579;
}
{
fma.rn.f16x2 r5584, r4533, r5577, r5581;
}
{
mul.f16x2 r5588, r4533, r5579;
}
{
neg.f16x2 r5591, r5588;
}
{
fma.rn.f16x2 r5593, r4536, r5577, r5591;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5597, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5599, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5601, {low, high};
}
{
mul.f16x2 r5602, r5599, r5601;
}
{
mul.f16x2 r5605, r5573, r5597;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5573;
mov.b32 r5608, {high, low};
}
{
fma.rn.f16x2 r5610, r5602, r5608, r5605;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5610;
mov.b32 r5614, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5610;
mov.b32 r5616, {high, high};
}
{
mul.f16x2 r5618, r4548, r5616;
}
{
fma.rn.f16x2 r5621, r4545, r5614, r5618;
}
{
mul.f16x2 r5625, r4545, r5616;
}
{
neg.f16x2 r5628, r5625;
}
{
fma.rn.f16x2 r5630, r4548, r5614, r5628;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5634, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5636, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5638, {low, high};
}
{
mul.f16x2 r5639, r5636, r5638;
}
{
mul.f16x2 r5642, r5610, r5634;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5610;
mov.b32 r5645, {high, low};
}
{
fma.rn.f16x2 r5647, r5639, r5645, r5642;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5647;
mov.b32 r5651, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5647;
mov.b32 r5653, {high, high};
}
{
mul.f16x2 r5655, r4560, r5653;
}
{
fma.rn.f16x2 r5658, r4557, r5651, r5655;
}
{
mul.f16x2 r5662, r4557, r5653;
}
{
neg.f16x2 r5665, r5662;
}
{
fma.rn.f16x2 r5667, r4560, r5651, r5665;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5671, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4575;
mov.b32 r5673, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1083;
cvt.rn.f16.f32 high, f1084;
mov.b32 r5675, {low, high};
}
{
mul.f16x2 r5676, r5673, r5675;
}
{
mul.f16x2 r5679, r5647, r5671;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5647;
mov.b32 r5682, {high, low};
}
{
fma.rn.f16x2 r5684, r5676, r5682, r5679;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5684;
mov.b32 r5688, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r5684;
mov.b32 r5690, {high, high};
}
{
mul.f16x2 r5692, r4572, r5690;
}
{
fma.rn.f16x2 r5695, r4569, r5688, r5692;
}
{
mul.f16x2 r5699, r4569, r5690;
}
{
neg.f16x2 r5702, r5699;
}
{
fma.rn.f16x2 r5704, r4572, r5688, r5702;
}
barrier.sync 0;
and.b32 r6974, r6963, 61440;
add.s32 r6975, r6973, r6974;
st.shared.u32 [r6975], r4383;
st.shared.u32 [r6975+128], r4585;
st.shared.u32 [r6975+256], r4622;
st.shared.u32 [r6975+384], r4659;
st.shared.u32 [r6975+512], r4696;
st.shared.u32 [r6975+640], r4733;
st.shared.u32 [r6975+768], r4770;
st.shared.u32 [r6975+896], r4807;
st.shared.u32 [r6975+1024], r4844;
st.shared.u32 [r6975+1152], r4881;
st.shared.u32 [r6975+1280], r4918;
st.shared.u32 [r6975+1408], r4955;
st.shared.u32 [r6975+1536], r4992;
st.shared.u32 [r6975+1664], r5029;
st.shared.u32 [r6975+1792], r5066;
st.shared.u32 [r6975+1920], r5103;
st.shared.u32 [r6975+2048], r5140;
st.shared.u32 [r6975+2176], r5177;
st.shared.u32 [r6975+2304], r5214;
st.shared.u32 [r6975+2432], r5251;
st.shared.u32 [r6975+2560], r5288;
st.shared.u32 [r6975+2688], r5325;
st.shared.u32 [r6975+2816], r5362;
st.shared.u32 [r6975+2944], r5399;
st.shared.u32 [r6975+3072], r5436;
st.shared.u32 [r6975+3200], r5473;
st.shared.u32 [r6975+3328], r5510;
st.shared.u32 [r6975+3456], r5547;
st.shared.u32 [r6975+3584], r5584;
st.shared.u32 [r6975+3712], r5621;
st.shared.u32 [r6975+3840], r5658;
st.shared.u32 [r6975+3968], r5695;
barrier.sync 0;
mad.lo.s32 r6976, r6969, -124, r6975;
ld.shared.u32 r5726, [r6976];
ld.shared.u32 r6342, [r6976+2048];
ld.shared.u32 r5922, [r6976+4096];
ld.shared.u32 r6538, [r6976+6144];
ld.shared.u32 r5776, [r6976+8192];
ld.shared.u32 r6392, [r6976+10240];
ld.shared.u32 r5972, [r6976+12288];
ld.shared.u32 r6588, [r6976+14336];
ld.shared.u32 r5738, [r6976+16384];
ld.shared.u32 r6354, [r6976+18432];
ld.shared.u32 r5934, [r6976+20480];
ld.shared.u32 r6550, [r6976+22528];
ld.shared.u32 r5788, [r6976+24576];
ld.shared.u32 r6404, [r6976+26624];
ld.shared.u32 r5984, [r6976+28672];
ld.shared.u32 r6600, [r6976+30720];
ld.shared.u32 r5727, [r6976+32768];
ld.shared.u32 r6343, [r6976+34816];
ld.shared.u32 r5923, [r6976+36864];
ld.shared.u32 r6539, [r6976+38912];
ld.shared.u32 r5777, [r6976+40960];
ld.shared.u32 r6393, [r6976+43008];
ld.shared.u32 r5973, [r6976+45056];
ld.shared.u32 r6589, [r6976+47104];
ld.shared.u32 r5739, [r6976+49152];
ld.shared.u32 r6355, [r6976+51200];
ld.shared.u32 r5935, [r6976+53248];
ld.shared.u32 r6551, [r6976+55296];
ld.shared.u32 r5789, [r6976+57344];
ld.shared.u32 r6405, [r6976+59392];
ld.shared.u32 r5985, [r6976+61440];
ld.shared.u32 r6601, [r6976+63488];
barrier.sync 0;
st.shared.u32 [r6975], r4386;
st.shared.u32 [r6975+128], r4594;
st.shared.u32 [r6975+256], r4631;
st.shared.u32 [r6975+384], r4668;
st.shared.u32 [r6975+512], r4705;
st.shared.u32 [r6975+640], r4742;
st.shared.u32 [r6975+768], r4779;
st.shared.u32 [r6975+896], r4816;
st.shared.u32 [r6975+1024], r4853;
st.shared.u32 [r6975+1152], r4890;
st.shared.u32 [r6975+1280], r4927;
st.shared.u32 [r6975+1408], r4964;
st.shared.u32 [r6975+1536], r5001;
st.shared.u32 [r6975+1664], r5038;
st.shared.u32 [r6975+1792], r5075;
st.shared.u32 [r6975+1920], r5112;
st.shared.u32 [r6975+2048], r5149;
st.shared.u32 [r6975+2176], r5186;
st.shared.u32 [r6975+2304], r5223;
st.shared.u32 [r6975+2432], r5260;
st.shared.u32 [r6975+2560], r5297;
st.shared.u32 [r6975+2688], r5334;
st.shared.u32 [r6975+2816], r5371;
st.shared.u32 [r6975+2944], r5408;
st.shared.u32 [r6975+3072], r5445;
st.shared.u32 [r6975+3200], r5482;
st.shared.u32 [r6975+3328], r5519;
st.shared.u32 [r6975+3456], r5556;
st.shared.u32 [r6975+3584], r5593;
st.shared.u32 [r6975+3712], r5630;
st.shared.u32 [r6975+3840], r5667;
st.shared.u32 [r6975+3968], r5704;
barrier.sync 0;
ld.shared.u32 r5729, [r6976];
ld.shared.u32 r6345, [r6976+2048];
ld.shared.u32 r5925, [r6976+4096];
ld.shared.u32 r6541, [r6976+6144];
ld.shared.u32 r5779, [r6976+8192];
ld.shared.u32 r6395, [r6976+10240];
ld.shared.u32 r5975, [r6976+12288];
ld.shared.u32 r6591, [r6976+14336];
ld.shared.u32 r5741, [r6976+16384];
ld.shared.u32 r6357, [r6976+18432];
ld.shared.u32 r5937, [r6976+20480];
ld.shared.u32 r6553, [r6976+22528];
ld.shared.u32 r5791, [r6976+24576];
ld.shared.u32 r6407, [r6976+26624];
ld.shared.u32 r5987, [r6976+28672];
ld.shared.u32 r6603, [r6976+30720];
ld.shared.u32 r5730, [r6976+32768];
ld.shared.u32 r6346, [r6976+34816];
ld.shared.u32 r5926, [r6976+36864];
ld.shared.u32 r6542, [r6976+38912];
ld.shared.u32 r5780, [r6976+40960];
ld.shared.u32 r6396, [r6976+43008];
ld.shared.u32 r5976, [r6976+45056];
ld.shared.u32 r6592, [r6976+47104];
ld.shared.u32 r5742, [r6976+49152];
ld.shared.u32 r6358, [r6976+51200];
ld.shared.u32 r5938, [r6976+53248];
ld.shared.u32 r6554, [r6976+55296];
ld.shared.u32 r5792, [r6976+57344];
ld.shared.u32 r6408, [r6976+59392];
ld.shared.u32 r5988, [r6976+61440];
ld.shared.u32 r6604, [r6976+63488];
{
add.f16x2 r5725, r5726, r5727;
}
{
add.f16x2 r5728, r5729, r5730;
}
{
sub.f16x2 r5731, r5726, r5727;
}
{
sub.f16x2 r5734, r5729, r5730;
}
{
add.f16x2 r5737, r5738, r5739;
}
{
add.f16x2 r5740, r5741, r5742;
}
{
sub.f16x2 r5743, r5738, r5739;
}
{
sub.f16x2 r5746, r5741, r5742;
}
{
neg.f16x2 r5749, r5746;
}
{
add.f16x2 r5751, r5725, r5737;
}
{
add.f16x2 r5754, r5728, r5740;
}
{
sub.f16x2 r5757, r5725, r5737;
}
{
sub.f16x2 r5760, r5728, r5740;
}
{
add.f16x2 r5763, r5731, r5749;
}
{
add.f16x2 r5766, r5734, r5743;
}
{
sub.f16x2 r5769, r5731, r5749;
}
{
sub.f16x2 r5772, r5734, r5743;
}
{
add.f16x2 r5775, r5776, r5777;
}
{
add.f16x2 r5778, r5779, r5780;
}
{
sub.f16x2 r5781, r5776, r5777;
}
{
sub.f16x2 r5784, r5779, r5780;
}
{
add.f16x2 r5787, r5788, r5789;
}
{
add.f16x2 r5790, r5791, r5792;
}
{
sub.f16x2 r5793, r5788, r5789;
}
{
sub.f16x2 r5796, r5791, r5792;
}
{
neg.f16x2 r5799, r5796;
}
{
add.f16x2 r5801, r5775, r5787;
}
{
add.f16x2 r5804, r5778, r5790;
}
{
sub.f16x2 r5807, r5775, r5787;
}
{
sub.f16x2 r5810, r5778, r5790;
}
{
add.f16x2 r5813, r5781, r5799;
}
{
add.f16x2 r5816, r5784, r5793;
}
{
sub.f16x2 r5819, r5781, r5799;
}
{
sub.f16x2 r5822, r5784, r5793;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r5825, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r5826, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r5829, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r5830, {low, high};
}
{
mul.f16x2 r5839, r5813, r5825;
}
{
mul.f16x2 r5842, r5816, r5826;
}
{
sub.f16x2 r5845, r5839, r5842;
}
{
mul.f16x2 r5848, r5813, r5826;
}
{
fma.rn.f16x2 r5851, r5816, r5825, r5848;
}
{
neg.f16x2 r5855, r5810;
}
{
mul.f16x2 r5857, r5819, r5829;
}
{
mul.f16x2 r5860, r5822, r5830;
}
{
sub.f16x2 r5863, r5857, r5860;
}
{
mul.f16x2 r5866, r5819, r5830;
}
{
fma.rn.f16x2 r5869, r5822, r5829, r5866;
}
{
add.f16x2 r5873, r5751, r5801;
}
{
add.f16x2 r5876, r5754, r5804;
}
{
sub.f16x2 r5879, r5751, r5801;
}
{
sub.f16x2 r5882, r5754, r5804;
}
{
add.f16x2 r5885, r5763, r5845;
}
{
add.f16x2 r5888, r5766, r5851;
}
{
sub.f16x2 r5891, r5763, r5845;
}
{
sub.f16x2 r5894, r5766, r5851;
}
{
add.f16x2 r5897, r5757, r5855;
}
{
add.f16x2 r5900, r5760, r5807;
}
{
sub.f16x2 r5903, r5757, r5855;
}
{
sub.f16x2 r5906, r5760, r5807;
}
{
add.f16x2 r5909, r5769, r5863;
}
{
add.f16x2 r5912, r5772, r5869;
}
{
sub.f16x2 r5915, r5769, r5863;
}
{
sub.f16x2 r5918, r5772, r5869;
}
{
add.f16x2 r5921, r5922, r5923;
}
{
add.f16x2 r5924, r5925, r5926;
}
{
sub.f16x2 r5927, r5922, r5923;
}
{
sub.f16x2 r5930, r5925, r5926;
}
{
add.f16x2 r5933, r5934, r5935;
}
{
add.f16x2 r5936, r5937, r5938;
}
{
sub.f16x2 r5939, r5934, r5935;
}
{
sub.f16x2 r5942, r5937, r5938;
}
{
neg.f16x2 r5945, r5942;
}
{
add.f16x2 r5947, r5921, r5933;
}
{
add.f16x2 r5950, r5924, r5936;
}
{
sub.f16x2 r5953, r5921, r5933;
}
{
sub.f16x2 r5956, r5924, r5936;
}
{
add.f16x2 r5959, r5927, r5945;
}
{
add.f16x2 r5962, r5930, r5939;
}
{
sub.f16x2 r5965, r5927, r5945;
}
{
sub.f16x2 r5968, r5930, r5939;
}
{
add.f16x2 r5971, r5972, r5973;
}
{
add.f16x2 r5974, r5975, r5976;
}
{
sub.f16x2 r5977, r5972, r5973;
}
{
sub.f16x2 r5980, r5975, r5976;
}
{
add.f16x2 r5983, r5984, r5985;
}
{
add.f16x2 r5986, r5987, r5988;
}
{
sub.f16x2 r5989, r5984, r5985;
}
{
sub.f16x2 r5992, r5987, r5988;
}
{
neg.f16x2 r5995, r5992;
}
{
add.f16x2 r5997, r5971, r5983;
}
{
add.f16x2 r6000, r5974, r5986;
}
{
sub.f16x2 r6003, r5971, r5983;
}
{
sub.f16x2 r6006, r5974, r5986;
}
{
add.f16x2 r6009, r5977, r5995;
}
{
add.f16x2 r6012, r5980, r5989;
}
{
sub.f16x2 r6015, r5977, r5995;
}
{
sub.f16x2 r6018, r5980, r5989;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6022, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r6025, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6026, {low, high};
}
{
mul.f16x2 r6035, r6009, r6021;
}
{
mul.f16x2 r6038, r6012, r6022;
}
{
sub.f16x2 r6041, r6035, r6038;
}
{
mul.f16x2 r6044, r6009, r6022;
}
{
fma.rn.f16x2 r6047, r6012, r6021, r6044;
}
{
neg.f16x2 r6051, r6006;
}
{
mul.f16x2 r6053, r6015, r6025;
}
{
mul.f16x2 r6056, r6018, r6026;
}
{
sub.f16x2 r6059, r6053, r6056;
}
{
mul.f16x2 r6062, r6015, r6026;
}
{
fma.rn.f16x2 r6065, r6018, r6025, r6062;
}
{
add.f16x2 r6069, r5947, r5997;
}
{
add.f16x2 r6072, r5950, r6000;
}
{
sub.f16x2 r6075, r5947, r5997;
}
{
sub.f16x2 r6078, r5950, r6000;
}
{
add.f16x2 r6081, r5959, r6041;
}
{
add.f16x2 r6084, r5962, r6047;
}
{
sub.f16x2 r6087, r5959, r6041;
}
{
sub.f16x2 r6090, r5962, r6047;
}
{
add.f16x2 r6093, r5953, r6051;
}
{
add.f16x2 r6096, r5956, r6003;
}
{
sub.f16x2 r6099, r5953, r6051;
}
{
sub.f16x2 r6102, r5956, r6003;
}
{
add.f16x2 r6105, r5965, r6059;
}
{
add.f16x2 r6108, r5968, r6065;
}
{
sub.f16x2 r6111, r5965, r6059;
}
{
sub.f16x2 r6114, r5968, r6065;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r6117, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r6118, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6119, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6120, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r6121, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r6122, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6125, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r6126, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r6127, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6128, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1042;
cvt.rn.f16.f32 high, f1042;
mov.b32 r6129, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r6130, {low, high};
}
{
mul.f16x2 r6147, r6081, r6117;
}
{
mul.f16x2 r6150, r6084, r6118;
}
{
sub.f16x2 r6153, r6147, r6150;
}
{
mul.f16x2 r6156, r6081, r6118;
}
{
fma.rn.f16x2 r6159, r6084, r6117, r6156;
}
{
mul.f16x2 r6163, r6093, r6119;
}
{
mul.f16x2 r6166, r6096, r6120;
}
{
sub.f16x2 r6169, r6163, r6166;
}
{
mul.f16x2 r6172, r6093, r6120;
}
{
fma.rn.f16x2 r6175, r6096, r6119, r6172;
}
{
mul.f16x2 r6179, r6105, r6121;
}
{
mul.f16x2 r6182, r6108, r6122;
}
{
sub.f16x2 r6185, r6179, r6182;
}
{
mul.f16x2 r6188, r6105, r6122;
}
{
fma.rn.f16x2 r6191, r6108, r6121, r6188;
}
{
neg.f16x2 r6195, r6078;
}
{
mul.f16x2 r6197, r6087, r6125;
}
{
mul.f16x2 r6200, r6090, r6126;
}
{
sub.f16x2 r6203, r6197, r6200;
}
{
mul.f16x2 r6206, r6087, r6126;
}
{
fma.rn.f16x2 r6209, r6090, r6125, r6206;
}
{
mul.f16x2 r6213, r6099, r6127;
}
{
mul.f16x2 r6216, r6102, r6128;
}
{
sub.f16x2 r6219, r6213, r6216;
}
{
mul.f16x2 r6222, r6099, r6128;
}
{
fma.rn.f16x2 r6225, r6102, r6127, r6222;
}
{
mul.f16x2 r6229, r6111, r6129;
}
{
mul.f16x2 r6232, r6114, r6130;
}
{
sub.f16x2 r6235, r6229, r6232;
}
{
mul.f16x2 r6238, r6111, r6130;
}
{
fma.rn.f16x2 r6241, r6114, r6129, r6238;
}
{
add.f16x2 %0, r5873, r6069;
}
{
add.f16x2 %1, r5876, r6072;
}
{
sub.f16x2 %32, r5873, r6069;
}
{
sub.f16x2 %33, r5876, r6072;
}
{
add.f16x2 %4, r5885, r6153;
}
{
add.f16x2 %5, r5888, r6159;
}
{
sub.f16x2 %36, r5885, r6153;
}
{
sub.f16x2 %37, r5888, r6159;
}
{
add.f16x2 %8, r5897, r6169;
}
{
add.f16x2 %9, r5900, r6175;
}
{
sub.f16x2 %40, r5897, r6169;
}
{
sub.f16x2 %41, r5900, r6175;
}
{
add.f16x2 %12, r5909, r6185;
}
{
add.f16x2 %13, r5912, r6191;
}
{
sub.f16x2 %44, r5909, r6185;
}
{
sub.f16x2 %45, r5912, r6191;
}
{
add.f16x2 %16, r5879, r6195;
}
{
add.f16x2 %17, r5882, r6075;
}
{
sub.f16x2 %48, r5879, r6195;
}
{
sub.f16x2 %49, r5882, r6075;
}
{
add.f16x2 %20, r5891, r6203;
}
{
add.f16x2 %21, r5894, r6209;
}
{
sub.f16x2 %52, r5891, r6203;
}
{
sub.f16x2 %53, r5894, r6209;
}
{
add.f16x2 %24, r5903, r6219;
}
{
add.f16x2 %25, r5906, r6225;
}
{
sub.f16x2 %56, r5903, r6219;
}
{
sub.f16x2 %57, r5906, r6225;
}
{
add.f16x2 %28, r5915, r6235;
}
{
add.f16x2 %29, r5918, r6241;
}
{
sub.f16x2 %60, r5915, r6235;
}
{
sub.f16x2 %61, r5918, r6241;
}
{
add.f16x2 r6341, r6342, r6343;
}
{
add.f16x2 r6344, r6345, r6346;
}
{
sub.f16x2 r6347, r6342, r6343;
}
{
sub.f16x2 r6350, r6345, r6346;
}
{
add.f16x2 r6353, r6354, r6355;
}
{
add.f16x2 r6356, r6357, r6358;
}
{
sub.f16x2 r6359, r6354, r6355;
}
{
sub.f16x2 r6362, r6357, r6358;
}
{
neg.f16x2 r6365, r6362;
}
{
add.f16x2 r6367, r6341, r6353;
}
{
add.f16x2 r6370, r6344, r6356;
}
{
sub.f16x2 r6373, r6341, r6353;
}
{
sub.f16x2 r6376, r6344, r6356;
}
{
add.f16x2 r6379, r6347, r6365;
}
{
add.f16x2 r6382, r6350, r6359;
}
{
sub.f16x2 r6385, r6347, r6365;
}
{
sub.f16x2 r6388, r6350, r6359;
}
{
add.f16x2 r6391, r6392, r6393;
}
{
add.f16x2 r6394, r6395, r6396;
}
{
sub.f16x2 r6397, r6392, r6393;
}
{
sub.f16x2 r6400, r6395, r6396;
}
{
add.f16x2 r6403, r6404, r6405;
}
{
add.f16x2 r6406, r6407, r6408;
}
{
sub.f16x2 r6409, r6404, r6405;
}
{
sub.f16x2 r6412, r6407, r6408;
}
{
neg.f16x2 r6415, r6412;
}
{
add.f16x2 r6417, r6391, r6403;
}
{
add.f16x2 r6420, r6394, r6406;
}
{
sub.f16x2 r6423, r6391, r6403;
}
{
sub.f16x2 r6426, r6394, r6406;
}
{
add.f16x2 r6429, r6397, r6415;
}
{
add.f16x2 r6432, r6400, r6409;
}
{
sub.f16x2 r6435, r6397, r6415;
}
{
sub.f16x2 r6438, r6400, r6409;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6441, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6442, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r6445, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6446, {low, high};
}
{
mul.f16x2 r6455, r6429, r6441;
}
{
mul.f16x2 r6458, r6432, r6442;
}
{
sub.f16x2 r6461, r6455, r6458;
}
{
mul.f16x2 r6464, r6429, r6442;
}
{
fma.rn.f16x2 r6467, r6432, r6441, r6464;
}
{
neg.f16x2 r6471, r6426;
}
{
mul.f16x2 r6473, r6435, r6445;
}
{
mul.f16x2 r6476, r6438, r6446;
}
{
sub.f16x2 r6479, r6473, r6476;
}
{
mul.f16x2 r6482, r6435, r6446;
}
{
fma.rn.f16x2 r6485, r6438, r6445, r6482;
}
{
add.f16x2 r6489, r6367, r6417;
}
{
add.f16x2 r6492, r6370, r6420;
}
{
sub.f16x2 r6495, r6367, r6417;
}
{
sub.f16x2 r6498, r6370, r6420;
}
{
add.f16x2 r6501, r6379, r6461;
}
{
add.f16x2 r6504, r6382, r6467;
}
{
sub.f16x2 r6507, r6379, r6461;
}
{
sub.f16x2 r6510, r6382, r6467;
}
{
add.f16x2 r6513, r6373, r6471;
}
{
add.f16x2 r6516, r6376, r6423;
}
{
sub.f16x2 r6519, r6373, r6471;
}
{
sub.f16x2 r6522, r6376, r6423;
}
{
add.f16x2 r6525, r6385, r6479;
}
{
add.f16x2 r6528, r6388, r6485;
}
{
sub.f16x2 r6531, r6385, r6479;
}
{
sub.f16x2 r6534, r6388, r6485;
}
{
add.f16x2 r6537, r6538, r6539;
}
{
add.f16x2 r6540, r6541, r6542;
}
{
sub.f16x2 r6543, r6538, r6539;
}
{
sub.f16x2 r6546, r6541, r6542;
}
{
add.f16x2 r6549, r6550, r6551;
}
{
add.f16x2 r6552, r6553, r6554;
}
{
sub.f16x2 r6555, r6550, r6551;
}
{
sub.f16x2 r6558, r6553, r6554;
}
{
neg.f16x2 r6561, r6558;
}
{
add.f16x2 r6563, r6537, r6549;
}
{
add.f16x2 r6566, r6540, r6552;
}
{
sub.f16x2 r6569, r6537, r6549;
}
{
sub.f16x2 r6572, r6540, r6552;
}
{
add.f16x2 r6575, r6543, r6561;
}
{
add.f16x2 r6578, r6546, r6555;
}
{
sub.f16x2 r6581, r6543, r6561;
}
{
sub.f16x2 r6584, r6546, r6555;
}
{
add.f16x2 r6587, r6588, r6589;
}
{
add.f16x2 r6590, r6591, r6592;
}
{
sub.f16x2 r6593, r6588, r6589;
}
{
sub.f16x2 r6596, r6591, r6592;
}
{
add.f16x2 r6599, r6600, r6601;
}
{
add.f16x2 r6602, r6603, r6604;
}
{
sub.f16x2 r6605, r6600, r6601;
}
{
sub.f16x2 r6608, r6603, r6604;
}
{
neg.f16x2 r6611, r6608;
}
{
add.f16x2 r6613, r6587, r6599;
}
{
add.f16x2 r6616, r6590, r6602;
}
{
sub.f16x2 r6619, r6587, r6599;
}
{
sub.f16x2 r6622, r6590, r6602;
}
{
add.f16x2 r6625, r6593, r6611;
}
{
add.f16x2 r6628, r6596, r6605;
}
{
sub.f16x2 r6631, r6593, r6611;
}
{
sub.f16x2 r6634, r6596, r6605;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6637, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6638, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r6641, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6642, {low, high};
}
{
mul.f16x2 r6651, r6625, r6637;
}
{
mul.f16x2 r6654, r6628, r6638;
}
{
sub.f16x2 r6657, r6651, r6654;
}
{
mul.f16x2 r6660, r6625, r6638;
}
{
fma.rn.f16x2 r6663, r6628, r6637, r6660;
}
{
neg.f16x2 r6667, r6622;
}
{
mul.f16x2 r6669, r6631, r6641;
}
{
mul.f16x2 r6672, r6634, r6642;
}
{
sub.f16x2 r6675, r6669, r6672;
}
{
mul.f16x2 r6678, r6631, r6642;
}
{
fma.rn.f16x2 r6681, r6634, r6641, r6678;
}
{
add.f16x2 r6685, r6563, r6613;
}
{
add.f16x2 r6688, r6566, r6616;
}
{
sub.f16x2 r6691, r6563, r6613;
}
{
sub.f16x2 r6694, r6566, r6616;
}
{
add.f16x2 r6697, r6575, r6657;
}
{
add.f16x2 r6700, r6578, r6663;
}
{
sub.f16x2 r6703, r6575, r6657;
}
{
sub.f16x2 r6706, r6578, r6663;
}
{
add.f16x2 r6709, r6569, r6667;
}
{
add.f16x2 r6712, r6572, r6619;
}
{
sub.f16x2 r6715, r6569, r6667;
}
{
sub.f16x2 r6718, r6572, r6619;
}
{
add.f16x2 r6721, r6581, r6675;
}
{
add.f16x2 r6724, r6584, r6681;
}
{
sub.f16x2 r6727, r6581, r6675;
}
{
sub.f16x2 r6730, r6584, r6681;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r6733, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r6734, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6735, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6736, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r6737, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r6738, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6741, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r6742, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1038;
cvt.rn.f16.f32 high, f1038;
mov.b32 r6743, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1040;
cvt.rn.f16.f32 high, f1040;
mov.b32 r6744, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1042;
cvt.rn.f16.f32 high, f1042;
mov.b32 r6745, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1044;
cvt.rn.f16.f32 high, f1044;
mov.b32 r6746, {low, high};
}
{
mul.f16x2 r6763, r6697, r6733;
}
{
mul.f16x2 r6766, r6700, r6734;
}
{
sub.f16x2 r6769, r6763, r6766;
}
{
mul.f16x2 r6772, r6697, r6734;
}
{
fma.rn.f16x2 r6775, r6700, r6733, r6772;
}
{
mul.f16x2 r6779, r6709, r6735;
}
{
mul.f16x2 r6782, r6712, r6736;
}
{
sub.f16x2 r6785, r6779, r6782;
}
{
mul.f16x2 r6788, r6709, r6736;
}
{
fma.rn.f16x2 r6791, r6712, r6735, r6788;
}
{
mul.f16x2 r6795, r6721, r6737;
}
{
mul.f16x2 r6798, r6724, r6738;
}
{
sub.f16x2 r6801, r6795, r6798;
}
{
mul.f16x2 r6804, r6721, r6738;
}
{
fma.rn.f16x2 r6807, r6724, r6737, r6804;
}
{
neg.f16x2 r6811, r6694;
}
{
mul.f16x2 r6813, r6703, r6741;
}
{
mul.f16x2 r6816, r6706, r6742;
}
{
sub.f16x2 r6819, r6813, r6816;
}
{
mul.f16x2 r6822, r6703, r6742;
}
{
fma.rn.f16x2 r6825, r6706, r6741, r6822;
}
{
mul.f16x2 r6829, r6715, r6743;
}
{
mul.f16x2 r6832, r6718, r6744;
}
{
sub.f16x2 r6835, r6829, r6832;
}
{
mul.f16x2 r6838, r6715, r6744;
}
{
fma.rn.f16x2 r6841, r6718, r6743, r6838;
}
{
mul.f16x2 r6845, r6727, r6745;
}
{
mul.f16x2 r6848, r6730, r6746;
}
{
sub.f16x2 r6851, r6845, r6848;
}
{
mul.f16x2 r6854, r6727, r6746;
}
{
fma.rn.f16x2 r6857, r6730, r6745, r6854;
}
{
add.f16x2 %2, r6489, r6685;
}
{
add.f16x2 %3, r6492, r6688;
}
{
sub.f16x2 %34, r6489, r6685;
}
{
sub.f16x2 %35, r6492, r6688;
}
{
add.f16x2 %6, r6501, r6769;
}
{
add.f16x2 %7, r6504, r6775;
}
{
sub.f16x2 %38, r6501, r6769;
}
{
sub.f16x2 %39, r6504, r6775;
}
{
add.f16x2 %10, r6513, r6785;
}
{
add.f16x2 %11, r6516, r6791;
}
{
sub.f16x2 %42, r6513, r6785;
}
{
sub.f16x2 %43, r6516, r6791;
}
{
add.f16x2 %14, r6525, r6801;
}
{
add.f16x2 %15, r6528, r6807;
}
{
sub.f16x2 %46, r6525, r6801;
}
{
sub.f16x2 %47, r6528, r6807;
}
{
add.f16x2 %18, r6495, r6811;
}
{
add.f16x2 %19, r6498, r6691;
}
{
sub.f16x2 %50, r6495, r6811;
}
{
sub.f16x2 %51, r6498, r6691;
}
{
add.f16x2 %22, r6507, r6819;
}
{
add.f16x2 %23, r6510, r6825;
}
{
sub.f16x2 %54, r6507, r6819;
}
{
sub.f16x2 %55, r6510, r6825;
}
{
add.f16x2 %26, r6519, r6835;
}
{
add.f16x2 %27, r6522, r6841;
}
{
sub.f16x2 %58, r6519, r6835;
}
{
sub.f16x2 %59, r6522, r6841;
}
{
add.f16x2 %30, r6531, r6851;
}
{
add.f16x2 %31, r6534, r6857;
}
{
sub.f16x2 %62, r6531, r6851;
}
{
sub.f16x2 %63, r6534, r6857;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)), "=r"(__HALF2_TO_UI(rmem[27].x)), "=r"(__HALF2_TO_UI(rmem[27].y)), "=r"(__HALF2_TO_UI(rmem[28].x)), "=r"(__HALF2_TO_UI(rmem[28].y)), "=r"(__HALF2_TO_UI(rmem[29].x)), "=r"(__HALF2_TO_UI(rmem[29].y)), "=r"(__HALF2_TO_UI(rmem[30].x)), "=r"(__HALF2_TO_UI(rmem[30].y)), "=r"(__HALF2_TO_UI(rmem[31].x)), "=r"(__HALF2_TO_UI(rmem[31].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[31].y)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[28].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[27].y)), "r"(__HALF2_TO_UI(rmem[30].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[29].y)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[31].x)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[28].x)), "r"(__HALF2_TO_UI(rmem[27].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[30].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[29].x)), "r"(__HALF2_TO_UI(rmem[24].x)));
};


#endif
