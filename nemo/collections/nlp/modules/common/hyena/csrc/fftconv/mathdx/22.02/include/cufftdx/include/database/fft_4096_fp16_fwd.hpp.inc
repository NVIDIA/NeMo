#ifndef CUFFTDX_FFT_4096_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_4096_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<847, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<423>;
.reg .b32 r<2985>;
.reg .b64 rd<2>;
mov.u32 r2965, %tid.y;
shl.b32 r2966, r2965, 14;
mov.u32 r2967, %32;
add.s32 r2968, r2967, r2966;
mov.u32 r2969, %tid.x;
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
neg.f16x2 r25, r19;
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
add.f16x2 r39, r7, r22;
}
{
add.f16x2 r42, r10, r25;
}
{
sub.f16x2 r45, r7, r22;
}
{
sub.f16x2 r48, r10, r25;
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
neg.f16x2 r75, r69;
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
add.f16x2 r89, r57, r72;
}
{
add.f16x2 r92, r60, r75;
}
{
sub.f16x2 r95, r57, r72;
}
{
sub.f16x2 r98, r60, r75;
}
mov.f32 f362, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f362;
cvt.rn.f16.f32 high, f362;
mov.b32 r101, {low, high};
}
mov.f32 f380, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r102, {low, high};
}
mov.f32 f297, 0fBF800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r106, {low, high};
}
mov.f32 f298, 0f3F800000;
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
neg.f16x2 r131, r83;
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
add.f16x2 r173, r33, r86;
}
{
add.f16x2 r176, r36, r131;
}
{
sub.f16x2 r179, r33, r86;
}
{
sub.f16x2 r182, r36, r131;
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
neg.f16x2 r221, r215;
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
add.f16x2 r235, r203, r218;
}
{
add.f16x2 r238, r206, r221;
}
{
sub.f16x2 r241, r203, r218;
}
{
sub.f16x2 r244, r206, r221;
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
neg.f16x2 r271, r265;
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
add.f16x2 r285, r253, r268;
}
{
add.f16x2 r288, r256, r271;
}
{
sub.f16x2 r291, r253, r268;
}
{
sub.f16x2 r294, r256, r271;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f362;
cvt.rn.f16.f32 high, f362;
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
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
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
neg.f16x2 r327, r279;
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
add.f16x2 r369, r229, r282;
}
{
add.f16x2 r372, r232, r327;
}
{
sub.f16x2 r375, r229, r282;
}
{
sub.f16x2 r378, r232, r327;
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
mov.f32 f358, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f358;
cvt.rn.f16.f32 high, f358;
mov.b32 r393, {low, high};
}
mov.f32 f384, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f362;
cvt.rn.f16.f32 high, f362;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r396, {low, high};
}
mov.f32 f366, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f366;
cvt.rn.f16.f32 high, f366;
mov.b32 r397, {low, high};
}
mov.f32 f382, 0fBF6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f382;
cvt.rn.f16.f32 high, f382;
mov.b32 r398, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f382;
cvt.rn.f16.f32 high, f382;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r404, {low, high};
}
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
neg.f16x2 r471, r351;
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
add.f16x2 r569, r155, r354;
}
{
add.f16x2 r572, r158, r471;
}
{
sub.f16x2 r575, r155, r354;
}
{
sub.f16x2 r578, r158, r471;
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
and.b32 r2970, r2969, 255;
shl.b32 r2971, r2969, 6;
and.b32 r2972, r2971, -16384;
add.s32 r2973, r2968, r2972;
cvt.rn.f32.u32 f417, r2970;
mul.f32 f418, f417, 0f3AC90FDB;
cos.approx.f32 f117, f418;
sin.approx.f32 f419, f418;
neg.f32 f118, f419;
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
neg.f16x2 r627, r624;
}
{
fma.rn.f16x2 r629, r533, r620, r627;
}
{
mul.f16x2 r633, r533, r622;
}
{
fma.rn.f16x2 r636, r536, r620, r633;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r664, r661;
}
{
fma.rn.f16x2 r666, r545, r657, r664;
}
{
mul.f16x2 r670, r545, r659;
}
{
fma.rn.f16x2 r673, r548, r657, r670;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r701, r698;
}
{
fma.rn.f16x2 r703, r557, r694, r701;
}
{
mul.f16x2 r707, r557, r696;
}
{
fma.rn.f16x2 r710, r560, r694, r707;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r738, r735;
}
{
fma.rn.f16x2 r740, r569, r731, r738;
}
{
mul.f16x2 r744, r569, r733;
}
{
fma.rn.f16x2 r747, r572, r731, r744;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r775, r772;
}
{
fma.rn.f16x2 r777, r581, r768, r775;
}
{
mul.f16x2 r781, r581, r770;
}
{
fma.rn.f16x2 r784, r584, r768, r781;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r812, r809;
}
{
fma.rn.f16x2 r814, r593, r805, r812;
}
{
mul.f16x2 r818, r593, r807;
}
{
fma.rn.f16x2 r821, r596, r805, r818;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r849, r846;
}
{
fma.rn.f16x2 r851, r605, r842, r849;
}
{
mul.f16x2 r855, r605, r844;
}
{
fma.rn.f16x2 r858, r608, r842, r855;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r886, r883;
}
{
fma.rn.f16x2 r888, r527, r879, r886;
}
{
mul.f16x2 r892, r527, r881;
}
{
fma.rn.f16x2 r895, r530, r879, r892;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r923, r920;
}
{
fma.rn.f16x2 r925, r539, r916, r923;
}
{
mul.f16x2 r929, r539, r918;
}
{
fma.rn.f16x2 r932, r542, r916, r929;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r960, r957;
}
{
fma.rn.f16x2 r962, r551, r953, r960;
}
{
mul.f16x2 r966, r551, r955;
}
{
fma.rn.f16x2 r969, r554, r953, r966;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r997, r994;
}
{
fma.rn.f16x2 r999, r563, r990, r997;
}
{
mul.f16x2 r1003, r563, r992;
}
{
fma.rn.f16x2 r1006, r566, r990, r1003;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r1034, r1031;
}
{
fma.rn.f16x2 r1036, r575, r1027, r1034;
}
{
mul.f16x2 r1040, r575, r1029;
}
{
fma.rn.f16x2 r1043, r578, r1027, r1040;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r1071, r1068;
}
{
fma.rn.f16x2 r1073, r587, r1064, r1071;
}
{
mul.f16x2 r1077, r587, r1066;
}
{
fma.rn.f16x2 r1080, r590, r1064, r1077;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r1108, r1105;
}
{
fma.rn.f16x2 r1110, r599, r1101, r1108;
}
{
mul.f16x2 r1114, r599, r1103;
}
{
fma.rn.f16x2 r1117, r602, r1101, r1114;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r1145, r1142;
}
{
fma.rn.f16x2 r1147, r611, r1138, r1145;
}
{
mul.f16x2 r1151, r611, r1140;
}
{
fma.rn.f16x2 r1154, r614, r1138, r1151;
}
barrier.sync 0;
and.b32 r2974, r2971, 16320;
add.s32 r2975, r2973, r2974;
st.shared.v4.f32 [r2975], {r521, r629, r666, r703};
st.shared.v4.f32 [r2975+16], {r740, r777, r814, r851};
st.shared.v4.f32 [r2975+32], {r888, r925, r962, r999};
st.shared.v4.f32 [r2975+48], {r1036, r1073, r1110, r1147};
barrier.sync 0;
mad.lo.s32 r2976, r2970, -60, r2975;
ld.shared.u32 r1176, [r2976];
ld.shared.u32 r1372, [r2976+1024];
ld.shared.u32 r1226, [r2976+2048];
ld.shared.u32 r1422, [r2976+3072];
ld.shared.u32 r1188, [r2976+4096];
ld.shared.u32 r1384, [r2976+5120];
ld.shared.u32 r1238, [r2976+6144];
ld.shared.u32 r1434, [r2976+7168];
ld.shared.u32 r1177, [r2976+8192];
ld.shared.u32 r1373, [r2976+9216];
ld.shared.u32 r1227, [r2976+10240];
ld.shared.u32 r1423, [r2976+11264];
ld.shared.u32 r1189, [r2976+12288];
ld.shared.u32 r1385, [r2976+13312];
ld.shared.u32 r1239, [r2976+14336];
ld.shared.u32 r1435, [r2976+15360];
barrier.sync 0;
st.shared.v4.f32 [r2975], {r524, r636, r673, r710};
st.shared.v4.f32 [r2975+16], {r747, r784, r821, r858};
st.shared.v4.f32 [r2975+32], {r895, r932, r969, r1006};
st.shared.v4.f32 [r2975+48], {r1043, r1080, r1117, r1154};
barrier.sync 0;
ld.shared.u32 r1179, [r2976];
ld.shared.u32 r1375, [r2976+1024];
ld.shared.u32 r1229, [r2976+2048];
ld.shared.u32 r1425, [r2976+3072];
ld.shared.u32 r1191, [r2976+4096];
ld.shared.u32 r1387, [r2976+5120];
ld.shared.u32 r1241, [r2976+6144];
ld.shared.u32 r1437, [r2976+7168];
ld.shared.u32 r1180, [r2976+8192];
ld.shared.u32 r1376, [r2976+9216];
ld.shared.u32 r1230, [r2976+10240];
ld.shared.u32 r1426, [r2976+11264];
ld.shared.u32 r1192, [r2976+12288];
ld.shared.u32 r1388, [r2976+13312];
ld.shared.u32 r1242, [r2976+14336];
ld.shared.u32 r1438, [r2976+15360];
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
neg.f16x2 r1199, r1193;
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
add.f16x2 r1213, r1181, r1196;
}
{
add.f16x2 r1216, r1184, r1199;
}
{
sub.f16x2 r1219, r1181, r1196;
}
{
sub.f16x2 r1222, r1184, r1199;
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
neg.f16x2 r1249, r1243;
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
add.f16x2 r1263, r1231, r1246;
}
{
add.f16x2 r1266, r1234, r1249;
}
{
sub.f16x2 r1269, r1231, r1246;
}
{
sub.f16x2 r1272, r1234, r1249;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f362;
cvt.rn.f16.f32 high, f362;
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
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
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
neg.f16x2 r1305, r1257;
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
add.f16x2 r1347, r1207, r1260;
}
{
add.f16x2 r1350, r1210, r1305;
}
{
sub.f16x2 r1353, r1207, r1260;
}
{
sub.f16x2 r1356, r1210, r1305;
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
neg.f16x2 r1395, r1389;
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
add.f16x2 r1409, r1377, r1392;
}
{
add.f16x2 r1412, r1380, r1395;
}
{
sub.f16x2 r1415, r1377, r1392;
}
{
sub.f16x2 r1418, r1380, r1395;
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
neg.f16x2 r1445, r1439;
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
add.f16x2 r1459, r1427, r1442;
}
{
add.f16x2 r1462, r1430, r1445;
}
{
sub.f16x2 r1465, r1427, r1442;
}
{
sub.f16x2 r1468, r1430, r1445;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f362;
cvt.rn.f16.f32 high, f362;
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
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
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
neg.f16x2 r1501, r1453;
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
add.f16x2 r1543, r1403, r1456;
}
{
add.f16x2 r1546, r1406, r1501;
}
{
sub.f16x2 r1549, r1403, r1456;
}
{
sub.f16x2 r1552, r1406, r1501;
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
cvt.rn.f16.f32 low, f358;
cvt.rn.f16.f32 high, f358;
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
cvt.rn.f16.f32 low, f362;
cvt.rn.f16.f32 high, f362;
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
cvt.rn.f16.f32 low, f366;
cvt.rn.f16.f32 high, f366;
mov.b32 r1571, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f382;
cvt.rn.f16.f32 high, f382;
mov.b32 r1572, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r1575, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f382;
cvt.rn.f16.f32 high, f382;
mov.b32 r1576, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
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
neg.f16x2 r1645, r1525;
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
add.f16x2 r1743, r1329, r1528;
}
{
add.f16x2 r1746, r1332, r1645;
}
{
sub.f16x2 r1749, r1329, r1528;
}
{
sub.f16x2 r1752, r1332, r1645;
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
and.b32 r2977, r2969, 240;
bfe.u32 r2978, r2969, 4, 4;
shl.b32 r2979, r2969, 2;
and.b32 r2980, r2979, 60;
add.s32 r2981, r2973, r2980;
cvt.rn.f32.u32 f420, r2978;
mul.f32 f421, f420, 0f3CC90FDB;
cos.approx.f32 f267, f421;
sin.approx.f32 f422, f421;
neg.f32 f268, f422;
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
neg.f16x2 r1801, r1798;
}
{
fma.rn.f16x2 r1803, r1707, r1794, r1801;
}
{
mul.f16x2 r1807, r1707, r1796;
}
{
fma.rn.f16x2 r1810, r1710, r1794, r1807;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r1838, r1835;
}
{
fma.rn.f16x2 r1840, r1719, r1831, r1838;
}
{
mul.f16x2 r1844, r1719, r1833;
}
{
fma.rn.f16x2 r1847, r1722, r1831, r1844;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r1875, r1872;
}
{
fma.rn.f16x2 r1877, r1731, r1868, r1875;
}
{
mul.f16x2 r1881, r1731, r1870;
}
{
fma.rn.f16x2 r1884, r1734, r1868, r1881;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r1912, r1909;
}
{
fma.rn.f16x2 r1914, r1743, r1905, r1912;
}
{
mul.f16x2 r1918, r1743, r1907;
}
{
fma.rn.f16x2 r1921, r1746, r1905, r1918;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r1949, r1946;
}
{
fma.rn.f16x2 r1951, r1755, r1942, r1949;
}
{
mul.f16x2 r1955, r1755, r1944;
}
{
fma.rn.f16x2 r1958, r1758, r1942, r1955;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r1986, r1983;
}
{
fma.rn.f16x2 r1988, r1767, r1979, r1986;
}
{
mul.f16x2 r1992, r1767, r1981;
}
{
fma.rn.f16x2 r1995, r1770, r1979, r1992;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r2023, r2020;
}
{
fma.rn.f16x2 r2025, r1779, r2016, r2023;
}
{
mul.f16x2 r2029, r1779, r2018;
}
{
fma.rn.f16x2 r2032, r1782, r2016, r2029;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r2060, r2057;
}
{
fma.rn.f16x2 r2062, r1701, r2053, r2060;
}
{
mul.f16x2 r2066, r1701, r2055;
}
{
fma.rn.f16x2 r2069, r1704, r2053, r2066;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r2097, r2094;
}
{
fma.rn.f16x2 r2099, r1713, r2090, r2097;
}
{
mul.f16x2 r2103, r1713, r2092;
}
{
fma.rn.f16x2 r2106, r1716, r2090, r2103;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r2134, r2131;
}
{
fma.rn.f16x2 r2136, r1725, r2127, r2134;
}
{
mul.f16x2 r2140, r1725, r2129;
}
{
fma.rn.f16x2 r2143, r1728, r2127, r2140;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r2171, r2168;
}
{
fma.rn.f16x2 r2173, r1737, r2164, r2171;
}
{
mul.f16x2 r2177, r1737, r2166;
}
{
fma.rn.f16x2 r2180, r1740, r2164, r2177;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r2208, r2205;
}
{
fma.rn.f16x2 r2210, r1749, r2201, r2208;
}
{
mul.f16x2 r2214, r1749, r2203;
}
{
fma.rn.f16x2 r2217, r1752, r2201, r2214;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r2245, r2242;
}
{
fma.rn.f16x2 r2247, r1761, r2238, r2245;
}
{
mul.f16x2 r2251, r1761, r2240;
}
{
fma.rn.f16x2 r2254, r1764, r2238, r2251;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r2282, r2279;
}
{
fma.rn.f16x2 r2284, r1773, r2275, r2282;
}
{
mul.f16x2 r2288, r1773, r2277;
}
{
fma.rn.f16x2 r2291, r1776, r2275, r2288;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r2319, r2316;
}
{
fma.rn.f16x2 r2321, r1785, r2312, r2319;
}
{
mul.f16x2 r2325, r1785, r2314;
}
{
fma.rn.f16x2 r2328, r1788, r2312, r2325;
}
barrier.sync 0;
and.b32 r2982, r2971, 15360;
add.s32 r2983, r2981, r2982;
st.shared.u32 [r2983], r1695;
st.shared.u32 [r2983+64], r1803;
st.shared.u32 [r2983+128], r1840;
st.shared.u32 [r2983+192], r1877;
st.shared.u32 [r2983+256], r1914;
st.shared.u32 [r2983+320], r1951;
st.shared.u32 [r2983+384], r1988;
st.shared.u32 [r2983+448], r2025;
st.shared.u32 [r2983+512], r2062;
st.shared.u32 [r2983+576], r2099;
st.shared.u32 [r2983+640], r2136;
st.shared.u32 [r2983+704], r2173;
st.shared.u32 [r2983+768], r2210;
st.shared.u32 [r2983+832], r2247;
st.shared.u32 [r2983+896], r2284;
st.shared.u32 [r2983+960], r2321;
barrier.sync 0;
mad.lo.s32 r2984, r2977, -60, r2983;
ld.shared.u32 r2350, [r2984];
ld.shared.u32 r2546, [r2984+1024];
ld.shared.u32 r2400, [r2984+2048];
ld.shared.u32 r2596, [r2984+3072];
ld.shared.u32 r2362, [r2984+4096];
ld.shared.u32 r2558, [r2984+5120];
ld.shared.u32 r2412, [r2984+6144];
ld.shared.u32 r2608, [r2984+7168];
ld.shared.u32 r2351, [r2984+8192];
ld.shared.u32 r2547, [r2984+9216];
ld.shared.u32 r2401, [r2984+10240];
ld.shared.u32 r2597, [r2984+11264];
ld.shared.u32 r2363, [r2984+12288];
ld.shared.u32 r2559, [r2984+13312];
ld.shared.u32 r2413, [r2984+14336];
ld.shared.u32 r2609, [r2984+15360];
barrier.sync 0;
st.shared.u32 [r2983], r1698;
st.shared.u32 [r2983+64], r1810;
st.shared.u32 [r2983+128], r1847;
st.shared.u32 [r2983+192], r1884;
st.shared.u32 [r2983+256], r1921;
st.shared.u32 [r2983+320], r1958;
st.shared.u32 [r2983+384], r1995;
st.shared.u32 [r2983+448], r2032;
st.shared.u32 [r2983+512], r2069;
st.shared.u32 [r2983+576], r2106;
st.shared.u32 [r2983+640], r2143;
st.shared.u32 [r2983+704], r2180;
st.shared.u32 [r2983+768], r2217;
st.shared.u32 [r2983+832], r2254;
st.shared.u32 [r2983+896], r2291;
st.shared.u32 [r2983+960], r2328;
barrier.sync 0;
ld.shared.u32 r2353, [r2984];
ld.shared.u32 r2549, [r2984+1024];
ld.shared.u32 r2403, [r2984+2048];
ld.shared.u32 r2599, [r2984+3072];
ld.shared.u32 r2365, [r2984+4096];
ld.shared.u32 r2561, [r2984+5120];
ld.shared.u32 r2415, [r2984+6144];
ld.shared.u32 r2611, [r2984+7168];
ld.shared.u32 r2354, [r2984+8192];
ld.shared.u32 r2550, [r2984+9216];
ld.shared.u32 r2404, [r2984+10240];
ld.shared.u32 r2600, [r2984+11264];
ld.shared.u32 r2366, [r2984+12288];
ld.shared.u32 r2562, [r2984+13312];
ld.shared.u32 r2416, [r2984+14336];
ld.shared.u32 r2612, [r2984+15360];
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
neg.f16x2 r2373, r2367;
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
add.f16x2 r2387, r2355, r2370;
}
{
add.f16x2 r2390, r2358, r2373;
}
{
sub.f16x2 r2393, r2355, r2370;
}
{
sub.f16x2 r2396, r2358, r2373;
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
neg.f16x2 r2423, r2417;
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
add.f16x2 r2437, r2405, r2420;
}
{
add.f16x2 r2440, r2408, r2423;
}
{
sub.f16x2 r2443, r2405, r2420;
}
{
sub.f16x2 r2446, r2408, r2423;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f362;
cvt.rn.f16.f32 high, f362;
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
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
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
neg.f16x2 r2479, r2431;
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
add.f16x2 r2521, r2381, r2434;
}
{
add.f16x2 r2524, r2384, r2479;
}
{
sub.f16x2 r2527, r2381, r2434;
}
{
sub.f16x2 r2530, r2384, r2479;
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
neg.f16x2 r2569, r2563;
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
add.f16x2 r2583, r2551, r2566;
}
{
add.f16x2 r2586, r2554, r2569;
}
{
sub.f16x2 r2589, r2551, r2566;
}
{
sub.f16x2 r2592, r2554, r2569;
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
neg.f16x2 r2619, r2613;
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
add.f16x2 r2633, r2601, r2616;
}
{
add.f16x2 r2636, r2604, r2619;
}
{
sub.f16x2 r2639, r2601, r2616;
}
{
sub.f16x2 r2642, r2604, r2619;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f362;
cvt.rn.f16.f32 high, f362;
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
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
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
neg.f16x2 r2675, r2627;
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
add.f16x2 r2717, r2577, r2630;
}
{
add.f16x2 r2720, r2580, r2675;
}
{
sub.f16x2 r2723, r2577, r2630;
}
{
sub.f16x2 r2726, r2580, r2675;
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
cvt.rn.f16.f32 low, f358;
cvt.rn.f16.f32 high, f358;
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
cvt.rn.f16.f32 low, f362;
cvt.rn.f16.f32 high, f362;
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
cvt.rn.f16.f32 low, f366;
cvt.rn.f16.f32 high, f366;
mov.b32 r2745, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f382;
cvt.rn.f16.f32 high, f382;
mov.b32 r2746, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r2749, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f382;
cvt.rn.f16.f32 high, f382;
mov.b32 r2750, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
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
neg.f16x2 r2819, r2699;
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
add.f16x2 %0, r2497, r2693;
}
{
add.f16x2 %1, r2500, r2696;
}
{
sub.f16x2 %16, r2497, r2693;
}
{
sub.f16x2 %17, r2500, r2696;
}
{
add.f16x2 %2, r2509, r2777;
}
{
add.f16x2 %3, r2512, r2783;
}
{
sub.f16x2 %18, r2509, r2777;
}
{
sub.f16x2 %19, r2512, r2783;
}
{
add.f16x2 %4, r2521, r2793;
}
{
add.f16x2 %5, r2524, r2799;
}
{
sub.f16x2 %20, r2521, r2793;
}
{
sub.f16x2 %21, r2524, r2799;
}
{
add.f16x2 %6, r2533, r2809;
}
{
add.f16x2 %7, r2536, r2815;
}
{
sub.f16x2 %22, r2533, r2809;
}
{
sub.f16x2 %23, r2536, r2815;
}
{
add.f16x2 %8, r2503, r2702;
}
{
add.f16x2 %9, r2506, r2819;
}
{
sub.f16x2 %24, r2503, r2702;
}
{
sub.f16x2 %25, r2506, r2819;
}
{
add.f16x2 %10, r2515, r2827;
}
{
add.f16x2 %11, r2518, r2833;
}
{
sub.f16x2 %26, r2515, r2827;
}
{
sub.f16x2 %27, r2518, r2833;
}
{
add.f16x2 %12, r2527, r2843;
}
{
add.f16x2 %13, r2530, r2849;
}
{
sub.f16x2 %28, r2527, r2843;
}
{
sub.f16x2 %29, r2530, r2849;
}
{
add.f16x2 %14, r2539, r2859;
}
{
add.f16x2 %15, r2542, r2865;
}
{
sub.f16x2 %30, r2539, r2859;
}
{
sub.f16x2 %31, r2542, r2865;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<849, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<176>;
.reg .b32 r<1598>;
.reg .b64 rd<2>;
mov.u32 r1571, %tid.y;
shl.b32 r1572, r1571, 14;
mov.u32 r1573, %16;
add.s32 r1574, r1573, r1572;
mov.u32 r1575, %tid.x;
{
add.f16x2 r1, %17, %25;
}
{
add.f16x2 r4, %18, %26;
}
{
sub.f16x2 r7, %17, %25;
}
{
sub.f16x2 r10, %18, %26;
}
{
add.f16x2 r13, %21, %29;
}
{
add.f16x2 r16, %22, %30;
}
{
sub.f16x2 r19, %21, %29;
}
{
sub.f16x2 r22, %22, %30;
}
{
neg.f16x2 r25, r19;
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
add.f16x2 r39, r7, r22;
}
{
add.f16x2 r42, r10, r25;
}
{
sub.f16x2 r45, r7, r22;
}
{
sub.f16x2 r48, r10, r25;
}
{
add.f16x2 r51, %19, %27;
}
{
add.f16x2 r54, %20, %28;
}
{
sub.f16x2 r57, %19, %27;
}
{
sub.f16x2 r60, %20, %28;
}
{
add.f16x2 r63, %23, %31;
}
{
add.f16x2 r66, %24, %32;
}
{
sub.f16x2 r69, %23, %31;
}
{
sub.f16x2 r72, %24, %32;
}
{
neg.f16x2 r75, r69;
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
add.f16x2 r89, r57, r72;
}
{
add.f16x2 r92, r60, r75;
}
{
sub.f16x2 r95, r57, r72;
}
{
sub.f16x2 r98, r60, r75;
}
mov.f32 f140, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r101, {low, high};
}
mov.f32 f150, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r102, {low, high};
}
mov.f32 f135, 0fBF800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r106, {low, high};
}
mov.f32 f136, 0f3F800000;
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
neg.f16x2 r131, r83;
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
add.f16x2 r173, r33, r86;
}
{
add.f16x2 r176, r36, r131;
}
{
sub.f16x2 r179, r33, r86;
}
{
sub.f16x2 r182, r36, r131;
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
and.b32 r1576, r1575, 511;
shl.b32 r1577, r1575, 5;
and.b32 r1578, r1577, -16384;
add.s32 r1579, r1574, r1578;
cvt.rn.f32.u32 f167, r1576;
mul.f32 f168, f167, 0f3AC90FDB;
cos.approx.f32 f29, f168;
sin.approx.f32 f169, f168;
neg.f32 f30, f169;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f29;
cvt.rn.f16.f32 high, f30;
mov.b32 r197, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r200, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r202, {high, high};
}
{
mul.f16x2 r204, r164, r202;
}
{
neg.f16x2 r207, r204;
}
{
fma.rn.f16x2 r209, r161, r200, r207;
}
{
mul.f16x2 r213, r161, r202;
}
{
fma.rn.f16x2 r216, r164, r200, r213;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r220, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r222, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r224, {low, high};
}
{
mul.f16x2 r225, r222, r224;
}
{
mul.f16x2 r228, r197, r220;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r231, {high, low};
}
{
fma.rn.f16x2 r233, r225, r231, r228;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r233;
mov.b32 r237, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r233;
mov.b32 r239, {high, high};
}
{
mul.f16x2 r241, r176, r239;
}
{
neg.f16x2 r244, r241;
}
{
fma.rn.f16x2 r246, r173, r237, r244;
}
{
mul.f16x2 r250, r173, r239;
}
{
fma.rn.f16x2 r253, r176, r237, r250;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r257, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r259, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r261, {low, high};
}
{
mul.f16x2 r262, r259, r261;
}
{
mul.f16x2 r265, r233, r257;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r233;
mov.b32 r268, {high, low};
}
{
fma.rn.f16x2 r270, r262, r268, r265;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r270;
mov.b32 r274, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r270;
mov.b32 r276, {high, high};
}
{
mul.f16x2 r278, r188, r276;
}
{
neg.f16x2 r281, r278;
}
{
fma.rn.f16x2 r283, r185, r274, r281;
}
{
mul.f16x2 r287, r185, r276;
}
{
fma.rn.f16x2 r290, r188, r274, r287;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r294, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r296, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r298, {low, high};
}
{
mul.f16x2 r299, r296, r298;
}
{
mul.f16x2 r302, r270, r294;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r270;
mov.b32 r305, {high, low};
}
{
fma.rn.f16x2 r307, r299, r305, r302;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r307;
mov.b32 r311, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r307;
mov.b32 r313, {high, high};
}
{
mul.f16x2 r315, r158, r313;
}
{
neg.f16x2 r318, r315;
}
{
fma.rn.f16x2 r320, r155, r311, r318;
}
{
mul.f16x2 r324, r155, r313;
}
{
fma.rn.f16x2 r327, r158, r311, r324;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r331, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r333, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r335, {low, high};
}
{
mul.f16x2 r336, r333, r335;
}
{
mul.f16x2 r339, r307, r331;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r307;
mov.b32 r342, {high, low};
}
{
fma.rn.f16x2 r344, r336, r342, r339;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r344;
mov.b32 r348, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r344;
mov.b32 r350, {high, high};
}
{
mul.f16x2 r352, r170, r350;
}
{
neg.f16x2 r355, r352;
}
{
fma.rn.f16x2 r357, r167, r348, r355;
}
{
mul.f16x2 r361, r167, r350;
}
{
fma.rn.f16x2 r364, r170, r348, r361;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r368, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r370, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r372, {low, high};
}
{
mul.f16x2 r373, r370, r372;
}
{
mul.f16x2 r376, r344, r368;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r344;
mov.b32 r379, {high, low};
}
{
fma.rn.f16x2 r381, r373, r379, r376;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r381;
mov.b32 r385, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r381;
mov.b32 r387, {high, high};
}
{
mul.f16x2 r389, r182, r387;
}
{
neg.f16x2 r392, r389;
}
{
fma.rn.f16x2 r394, r179, r385, r392;
}
{
mul.f16x2 r398, r179, r387;
}
{
fma.rn.f16x2 r401, r182, r385, r398;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r405, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r407, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r409, {low, high};
}
{
mul.f16x2 r410, r407, r409;
}
{
mul.f16x2 r413, r381, r405;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r381;
mov.b32 r416, {high, low};
}
{
fma.rn.f16x2 r418, r410, r416, r413;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r418;
mov.b32 r422, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r418;
mov.b32 r424, {high, high};
}
{
mul.f16x2 r426, r194, r424;
}
{
neg.f16x2 r429, r426;
}
{
fma.rn.f16x2 r431, r191, r422, r429;
}
{
mul.f16x2 r435, r191, r424;
}
{
fma.rn.f16x2 r438, r194, r422, r435;
}
barrier.sync 0;
and.b32 r1580, r1577, 16352;
add.s32 r1581, r1579, r1580;
st.shared.v4.f32 [r1581], {r149, r209, r246, r283};
st.shared.v4.f32 [r1581+16], {r320, r357, r394, r431};
barrier.sync 0;
mad.lo.s32 r1582, r1576, -28, r1581;
ld.shared.u32 r460, [r1582];
ld.shared.u32 r510, [r1582+2048];
ld.shared.u32 r472, [r1582+4096];
ld.shared.u32 r522, [r1582+6144];
ld.shared.u32 r461, [r1582+8192];
ld.shared.u32 r511, [r1582+10240];
ld.shared.u32 r473, [r1582+12288];
ld.shared.u32 r523, [r1582+14336];
barrier.sync 0;
st.shared.v4.f32 [r1581], {r152, r216, r253, r290};
st.shared.v4.f32 [r1581+16], {r327, r364, r401, r438};
barrier.sync 0;
ld.shared.u32 r463, [r1582];
ld.shared.u32 r513, [r1582+2048];
ld.shared.u32 r475, [r1582+4096];
ld.shared.u32 r525, [r1582+6144];
ld.shared.u32 r464, [r1582+8192];
ld.shared.u32 r514, [r1582+10240];
ld.shared.u32 r476, [r1582+12288];
ld.shared.u32 r526, [r1582+14336];
{
add.f16x2 r459, r460, r461;
}
{
add.f16x2 r462, r463, r464;
}
{
sub.f16x2 r465, r460, r461;
}
{
sub.f16x2 r468, r463, r464;
}
{
add.f16x2 r471, r472, r473;
}
{
add.f16x2 r474, r475, r476;
}
{
sub.f16x2 r477, r472, r473;
}
{
sub.f16x2 r480, r475, r476;
}
{
neg.f16x2 r483, r477;
}
{
add.f16x2 r485, r459, r471;
}
{
add.f16x2 r488, r462, r474;
}
{
sub.f16x2 r491, r459, r471;
}
{
sub.f16x2 r494, r462, r474;
}
{
add.f16x2 r497, r465, r480;
}
{
add.f16x2 r500, r468, r483;
}
{
sub.f16x2 r503, r465, r480;
}
{
sub.f16x2 r506, r468, r483;
}
{
add.f16x2 r509, r510, r511;
}
{
add.f16x2 r512, r513, r514;
}
{
sub.f16x2 r515, r510, r511;
}
{
sub.f16x2 r518, r513, r514;
}
{
add.f16x2 r521, r522, r523;
}
{
add.f16x2 r524, r525, r526;
}
{
sub.f16x2 r527, r522, r523;
}
{
sub.f16x2 r530, r525, r526;
}
{
neg.f16x2 r533, r527;
}
{
add.f16x2 r535, r509, r521;
}
{
add.f16x2 r538, r512, r524;
}
{
sub.f16x2 r541, r509, r521;
}
{
sub.f16x2 r544, r512, r524;
}
{
add.f16x2 r547, r515, r530;
}
{
add.f16x2 r550, r518, r533;
}
{
sub.f16x2 r553, r515, r530;
}
{
sub.f16x2 r556, r518, r533;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r559, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r560, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r563, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r564, {low, high};
}
{
mul.f16x2 r573, r547, r559;
}
{
mul.f16x2 r576, r550, r560;
}
{
sub.f16x2 r579, r573, r576;
}
{
mul.f16x2 r582, r547, r560;
}
{
fma.rn.f16x2 r585, r550, r559, r582;
}
{
neg.f16x2 r589, r541;
}
{
mul.f16x2 r591, r553, r563;
}
{
mul.f16x2 r594, r556, r564;
}
{
sub.f16x2 r597, r591, r594;
}
{
mul.f16x2 r600, r553, r564;
}
{
fma.rn.f16x2 r603, r556, r563, r600;
}
{
add.f16x2 r607, r485, r535;
}
{
add.f16x2 r610, r488, r538;
}
{
sub.f16x2 r613, r485, r535;
}
{
sub.f16x2 r616, r488, r538;
}
{
add.f16x2 r619, r497, r579;
}
{
add.f16x2 r622, r500, r585;
}
{
sub.f16x2 r625, r497, r579;
}
{
sub.f16x2 r628, r500, r585;
}
{
add.f16x2 r631, r491, r544;
}
{
add.f16x2 r634, r494, r589;
}
{
sub.f16x2 r637, r491, r544;
}
{
sub.f16x2 r640, r494, r589;
}
{
add.f16x2 r643, r503, r597;
}
{
add.f16x2 r646, r506, r603;
}
{
sub.f16x2 r649, r503, r597;
}
{
sub.f16x2 r652, r506, r603;
}
and.b32 r1583, r1575, 504;
bfe.u32 r1584, r1575, 3, 6;
shl.b32 r1585, r1575, 2;
and.b32 r1586, r1585, 28;
add.s32 r1587, r1579, r1586;
cvt.rn.f32.u32 f170, r1584;
mul.f32 f171, f170, 0f3C490FDB;
cos.approx.f32 f75, f171;
sin.approx.f32 f172, f171;
neg.f32 f76, f172;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f75;
cvt.rn.f16.f32 high, f76;
mov.b32 r655, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r658, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r660, {high, high};
}
{
mul.f16x2 r662, r622, r660;
}
{
neg.f16x2 r665, r662;
}
{
fma.rn.f16x2 r667, r619, r658, r665;
}
{
mul.f16x2 r671, r619, r660;
}
{
fma.rn.f16x2 r674, r622, r658, r671;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r678, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r680, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r682, {low, high};
}
{
mul.f16x2 r683, r680, r682;
}
{
mul.f16x2 r686, r655, r678;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r689, {high, low};
}
{
fma.rn.f16x2 r691, r683, r689, r686;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r691;
mov.b32 r695, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r691;
mov.b32 r697, {high, high};
}
{
mul.f16x2 r699, r634, r697;
}
{
neg.f16x2 r702, r699;
}
{
fma.rn.f16x2 r704, r631, r695, r702;
}
{
mul.f16x2 r708, r631, r697;
}
{
fma.rn.f16x2 r711, r634, r695, r708;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r715, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r717, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r719, {low, high};
}
{
mul.f16x2 r720, r717, r719;
}
{
mul.f16x2 r723, r691, r715;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r691;
mov.b32 r726, {high, low};
}
{
fma.rn.f16x2 r728, r720, r726, r723;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r728;
mov.b32 r732, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r728;
mov.b32 r734, {high, high};
}
{
mul.f16x2 r736, r646, r734;
}
{
neg.f16x2 r739, r736;
}
{
fma.rn.f16x2 r741, r643, r732, r739;
}
{
mul.f16x2 r745, r643, r734;
}
{
fma.rn.f16x2 r748, r646, r732, r745;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r752, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r754, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r756, {low, high};
}
{
mul.f16x2 r757, r754, r756;
}
{
mul.f16x2 r760, r728, r752;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r728;
mov.b32 r763, {high, low};
}
{
fma.rn.f16x2 r765, r757, r763, r760;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r765;
mov.b32 r769, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r765;
mov.b32 r771, {high, high};
}
{
mul.f16x2 r773, r616, r771;
}
{
neg.f16x2 r776, r773;
}
{
fma.rn.f16x2 r778, r613, r769, r776;
}
{
mul.f16x2 r782, r613, r771;
}
{
fma.rn.f16x2 r785, r616, r769, r782;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r789, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r791, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r793, {low, high};
}
{
mul.f16x2 r794, r791, r793;
}
{
mul.f16x2 r797, r765, r789;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r765;
mov.b32 r800, {high, low};
}
{
fma.rn.f16x2 r802, r794, r800, r797;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r802;
mov.b32 r806, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r802;
mov.b32 r808, {high, high};
}
{
mul.f16x2 r810, r628, r808;
}
{
neg.f16x2 r813, r810;
}
{
fma.rn.f16x2 r815, r625, r806, r813;
}
{
mul.f16x2 r819, r625, r808;
}
{
fma.rn.f16x2 r822, r628, r806, r819;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r826, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r828, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r830, {low, high};
}
{
mul.f16x2 r831, r828, r830;
}
{
mul.f16x2 r834, r802, r826;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r802;
mov.b32 r837, {high, low};
}
{
fma.rn.f16x2 r839, r831, r837, r834;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r839;
mov.b32 r843, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r839;
mov.b32 r845, {high, high};
}
{
mul.f16x2 r847, r640, r845;
}
{
neg.f16x2 r850, r847;
}
{
fma.rn.f16x2 r852, r637, r843, r850;
}
{
mul.f16x2 r856, r637, r845;
}
{
fma.rn.f16x2 r859, r640, r843, r856;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r863, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r865, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r867, {low, high};
}
{
mul.f16x2 r868, r865, r867;
}
{
mul.f16x2 r871, r839, r863;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r839;
mov.b32 r874, {high, low};
}
{
fma.rn.f16x2 r876, r868, r874, r871;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r876;
mov.b32 r880, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r876;
mov.b32 r882, {high, high};
}
{
mul.f16x2 r884, r652, r882;
}
{
neg.f16x2 r887, r884;
}
{
fma.rn.f16x2 r889, r649, r880, r887;
}
{
mul.f16x2 r893, r649, r882;
}
{
fma.rn.f16x2 r896, r652, r880, r893;
}
barrier.sync 0;
and.b32 r1588, r1577, 16128;
add.s32 r1589, r1587, r1588;
st.shared.u32 [r1589], r607;
st.shared.u32 [r1589+32], r667;
st.shared.u32 [r1589+64], r704;
st.shared.u32 [r1589+96], r741;
st.shared.u32 [r1589+128], r778;
st.shared.u32 [r1589+160], r815;
st.shared.u32 [r1589+192], r852;
st.shared.u32 [r1589+224], r889;
barrier.sync 0;
mad.lo.s32 r1590, r1583, -28, r1589;
ld.shared.u32 r918, [r1590];
ld.shared.u32 r968, [r1590+2048];
ld.shared.u32 r930, [r1590+4096];
ld.shared.u32 r980, [r1590+6144];
ld.shared.u32 r919, [r1590+8192];
ld.shared.u32 r969, [r1590+10240];
ld.shared.u32 r931, [r1590+12288];
ld.shared.u32 r981, [r1590+14336];
barrier.sync 0;
st.shared.u32 [r1589], r610;
st.shared.u32 [r1589+32], r674;
st.shared.u32 [r1589+64], r711;
st.shared.u32 [r1589+96], r748;
st.shared.u32 [r1589+128], r785;
st.shared.u32 [r1589+160], r822;
st.shared.u32 [r1589+192], r859;
st.shared.u32 [r1589+224], r896;
barrier.sync 0;
ld.shared.u32 r921, [r1590];
ld.shared.u32 r971, [r1590+2048];
ld.shared.u32 r933, [r1590+4096];
ld.shared.u32 r983, [r1590+6144];
ld.shared.u32 r922, [r1590+8192];
ld.shared.u32 r972, [r1590+10240];
ld.shared.u32 r934, [r1590+12288];
ld.shared.u32 r984, [r1590+14336];
{
add.f16x2 r917, r918, r919;
}
{
add.f16x2 r920, r921, r922;
}
{
sub.f16x2 r923, r918, r919;
}
{
sub.f16x2 r926, r921, r922;
}
{
add.f16x2 r929, r930, r931;
}
{
add.f16x2 r932, r933, r934;
}
{
sub.f16x2 r935, r930, r931;
}
{
sub.f16x2 r938, r933, r934;
}
{
neg.f16x2 r941, r935;
}
{
add.f16x2 r943, r917, r929;
}
{
add.f16x2 r946, r920, r932;
}
{
sub.f16x2 r949, r917, r929;
}
{
sub.f16x2 r952, r920, r932;
}
{
add.f16x2 r955, r923, r938;
}
{
add.f16x2 r958, r926, r941;
}
{
sub.f16x2 r961, r923, r938;
}
{
sub.f16x2 r964, r926, r941;
}
{
add.f16x2 r967, r968, r969;
}
{
add.f16x2 r970, r971, r972;
}
{
sub.f16x2 r973, r968, r969;
}
{
sub.f16x2 r976, r971, r972;
}
{
add.f16x2 r979, r980, r981;
}
{
add.f16x2 r982, r983, r984;
}
{
sub.f16x2 r985, r980, r981;
}
{
sub.f16x2 r988, r983, r984;
}
{
neg.f16x2 r991, r985;
}
{
add.f16x2 r993, r967, r979;
}
{
add.f16x2 r996, r970, r982;
}
{
sub.f16x2 r999, r967, r979;
}
{
sub.f16x2 r1002, r970, r982;
}
{
add.f16x2 r1005, r973, r988;
}
{
add.f16x2 r1008, r976, r991;
}
{
sub.f16x2 r1011, r973, r988;
}
{
sub.f16x2 r1014, r976, r991;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1017, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r1018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r1021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r1022, {low, high};
}
{
mul.f16x2 r1031, r1005, r1017;
}
{
mul.f16x2 r1034, r1008, r1018;
}
{
sub.f16x2 r1037, r1031, r1034;
}
{
mul.f16x2 r1040, r1005, r1018;
}
{
fma.rn.f16x2 r1043, r1008, r1017, r1040;
}
{
neg.f16x2 r1047, r999;
}
{
mul.f16x2 r1049, r1011, r1021;
}
{
mul.f16x2 r1052, r1014, r1022;
}
{
sub.f16x2 r1055, r1049, r1052;
}
{
mul.f16x2 r1058, r1011, r1022;
}
{
fma.rn.f16x2 r1061, r1014, r1021, r1058;
}
{
add.f16x2 r1065, r943, r993;
}
{
add.f16x2 r1068, r946, r996;
}
{
sub.f16x2 r1071, r943, r993;
}
{
sub.f16x2 r1074, r946, r996;
}
{
add.f16x2 r1077, r955, r1037;
}
{
add.f16x2 r1080, r958, r1043;
}
{
sub.f16x2 r1083, r955, r1037;
}
{
sub.f16x2 r1086, r958, r1043;
}
{
add.f16x2 r1089, r949, r1002;
}
{
add.f16x2 r1092, r952, r1047;
}
{
sub.f16x2 r1095, r949, r1002;
}
{
sub.f16x2 r1098, r952, r1047;
}
{
add.f16x2 r1101, r961, r1055;
}
{
add.f16x2 r1104, r964, r1061;
}
{
sub.f16x2 r1107, r961, r1055;
}
{
sub.f16x2 r1110, r964, r1061;
}
and.b32 r1591, r1575, 448;
bfe.u32 r1592, r1575, 6, 3;
and.b32 r1593, r1585, 252;
add.s32 r1594, r1579, r1593;
cvt.rn.f32.u32 f173, r1592;
mul.f32 f174, f173, 0f3DC90FDB;
cos.approx.f32 f121, f174;
sin.approx.f32 f175, f174;
neg.f32 f122, f175;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f121;
cvt.rn.f16.f32 high, f122;
mov.b32 r1113, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1116, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1118, {high, high};
}
{
mul.f16x2 r1120, r1080, r1118;
}
{
neg.f16x2 r1123, r1120;
}
{
fma.rn.f16x2 r1125, r1077, r1116, r1123;
}
{
mul.f16x2 r1129, r1077, r1118;
}
{
fma.rn.f16x2 r1132, r1080, r1116, r1129;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1136, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1138, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r1140, {low, high};
}
{
mul.f16x2 r1141, r1138, r1140;
}
{
mul.f16x2 r1144, r1113, r1136;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1147, {high, low};
}
{
fma.rn.f16x2 r1149, r1141, r1147, r1144;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1149;
mov.b32 r1153, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1149;
mov.b32 r1155, {high, high};
}
{
mul.f16x2 r1157, r1092, r1155;
}
{
neg.f16x2 r1160, r1157;
}
{
fma.rn.f16x2 r1162, r1089, r1153, r1160;
}
{
mul.f16x2 r1166, r1089, r1155;
}
{
fma.rn.f16x2 r1169, r1092, r1153, r1166;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1173, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1175, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r1177, {low, high};
}
{
mul.f16x2 r1178, r1175, r1177;
}
{
mul.f16x2 r1181, r1149, r1173;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1149;
mov.b32 r1184, {high, low};
}
{
fma.rn.f16x2 r1186, r1178, r1184, r1181;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1186;
mov.b32 r1190, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1186;
mov.b32 r1192, {high, high};
}
{
mul.f16x2 r1194, r1104, r1192;
}
{
neg.f16x2 r1197, r1194;
}
{
fma.rn.f16x2 r1199, r1101, r1190, r1197;
}
{
mul.f16x2 r1203, r1101, r1192;
}
{
fma.rn.f16x2 r1206, r1104, r1190, r1203;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1210, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1212, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r1214, {low, high};
}
{
mul.f16x2 r1215, r1212, r1214;
}
{
mul.f16x2 r1218, r1186, r1210;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1186;
mov.b32 r1221, {high, low};
}
{
fma.rn.f16x2 r1223, r1215, r1221, r1218;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1223;
mov.b32 r1227, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1223;
mov.b32 r1229, {high, high};
}
{
mul.f16x2 r1231, r1074, r1229;
}
{
neg.f16x2 r1234, r1231;
}
{
fma.rn.f16x2 r1236, r1071, r1227, r1234;
}
{
mul.f16x2 r1240, r1071, r1229;
}
{
fma.rn.f16x2 r1243, r1074, r1227, r1240;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1247, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1249, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r1251, {low, high};
}
{
mul.f16x2 r1252, r1249, r1251;
}
{
mul.f16x2 r1255, r1223, r1247;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1223;
mov.b32 r1258, {high, low};
}
{
fma.rn.f16x2 r1260, r1252, r1258, r1255;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1260;
mov.b32 r1264, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1260;
mov.b32 r1266, {high, high};
}
{
mul.f16x2 r1268, r1086, r1266;
}
{
neg.f16x2 r1271, r1268;
}
{
fma.rn.f16x2 r1273, r1083, r1264, r1271;
}
{
mul.f16x2 r1277, r1083, r1266;
}
{
fma.rn.f16x2 r1280, r1086, r1264, r1277;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1284, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1286, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r1288, {low, high};
}
{
mul.f16x2 r1289, r1286, r1288;
}
{
mul.f16x2 r1292, r1260, r1284;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1260;
mov.b32 r1295, {high, low};
}
{
fma.rn.f16x2 r1297, r1289, r1295, r1292;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1297;
mov.b32 r1301, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1297;
mov.b32 r1303, {high, high};
}
{
mul.f16x2 r1305, r1098, r1303;
}
{
neg.f16x2 r1308, r1305;
}
{
fma.rn.f16x2 r1310, r1095, r1301, r1308;
}
{
mul.f16x2 r1314, r1095, r1303;
}
{
fma.rn.f16x2 r1317, r1098, r1301, r1314;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1321, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1323, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r1325, {low, high};
}
{
mul.f16x2 r1326, r1323, r1325;
}
{
mul.f16x2 r1329, r1297, r1321;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1297;
mov.b32 r1332, {high, low};
}
{
fma.rn.f16x2 r1334, r1326, r1332, r1329;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1334;
mov.b32 r1338, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1334;
mov.b32 r1340, {high, high};
}
{
mul.f16x2 r1342, r1110, r1340;
}
{
neg.f16x2 r1345, r1342;
}
{
fma.rn.f16x2 r1347, r1107, r1338, r1345;
}
{
mul.f16x2 r1351, r1107, r1340;
}
{
fma.rn.f16x2 r1354, r1110, r1338, r1351;
}
barrier.sync 0;
and.b32 r1595, r1577, 14336;
add.s32 r1596, r1594, r1595;
st.shared.u32 [r1596], r1065;
st.shared.u32 [r1596+256], r1125;
st.shared.u32 [r1596+512], r1162;
st.shared.u32 [r1596+768], r1199;
st.shared.u32 [r1596+1024], r1236;
st.shared.u32 [r1596+1280], r1273;
st.shared.u32 [r1596+1536], r1310;
st.shared.u32 [r1596+1792], r1347;
barrier.sync 0;
mad.lo.s32 r1597, r1591, -28, r1596;
ld.shared.u32 r1376, [r1597];
ld.shared.u32 r1426, [r1597+2048];
ld.shared.u32 r1388, [r1597+4096];
ld.shared.u32 r1438, [r1597+6144];
ld.shared.u32 r1377, [r1597+8192];
ld.shared.u32 r1427, [r1597+10240];
ld.shared.u32 r1389, [r1597+12288];
ld.shared.u32 r1439, [r1597+14336];
barrier.sync 0;
st.shared.u32 [r1596], r1068;
st.shared.u32 [r1596+256], r1132;
st.shared.u32 [r1596+512], r1169;
st.shared.u32 [r1596+768], r1206;
st.shared.u32 [r1596+1024], r1243;
st.shared.u32 [r1596+1280], r1280;
st.shared.u32 [r1596+1536], r1317;
st.shared.u32 [r1596+1792], r1354;
barrier.sync 0;
ld.shared.u32 r1379, [r1597];
ld.shared.u32 r1429, [r1597+2048];
ld.shared.u32 r1391, [r1597+4096];
ld.shared.u32 r1441, [r1597+6144];
ld.shared.u32 r1380, [r1597+8192];
ld.shared.u32 r1430, [r1597+10240];
ld.shared.u32 r1392, [r1597+12288];
ld.shared.u32 r1442, [r1597+14336];
{
add.f16x2 r1375, r1376, r1377;
}
{
add.f16x2 r1378, r1379, r1380;
}
{
sub.f16x2 r1381, r1376, r1377;
}
{
sub.f16x2 r1384, r1379, r1380;
}
{
add.f16x2 r1387, r1388, r1389;
}
{
add.f16x2 r1390, r1391, r1392;
}
{
sub.f16x2 r1393, r1388, r1389;
}
{
sub.f16x2 r1396, r1391, r1392;
}
{
neg.f16x2 r1399, r1393;
}
{
add.f16x2 r1401, r1375, r1387;
}
{
add.f16x2 r1404, r1378, r1390;
}
{
sub.f16x2 r1407, r1375, r1387;
}
{
sub.f16x2 r1410, r1378, r1390;
}
{
add.f16x2 r1413, r1381, r1396;
}
{
add.f16x2 r1416, r1384, r1399;
}
{
sub.f16x2 r1419, r1381, r1396;
}
{
sub.f16x2 r1422, r1384, r1399;
}
{
add.f16x2 r1425, r1426, r1427;
}
{
add.f16x2 r1428, r1429, r1430;
}
{
sub.f16x2 r1431, r1426, r1427;
}
{
sub.f16x2 r1434, r1429, r1430;
}
{
add.f16x2 r1437, r1438, r1439;
}
{
add.f16x2 r1440, r1441, r1442;
}
{
sub.f16x2 r1443, r1438, r1439;
}
{
sub.f16x2 r1446, r1441, r1442;
}
{
neg.f16x2 r1449, r1443;
}
{
add.f16x2 r1451, r1425, r1437;
}
{
add.f16x2 r1454, r1428, r1440;
}
{
sub.f16x2 r1457, r1425, r1437;
}
{
sub.f16x2 r1460, r1428, r1440;
}
{
add.f16x2 r1463, r1431, r1446;
}
{
add.f16x2 r1466, r1434, r1449;
}
{
sub.f16x2 r1469, r1431, r1446;
}
{
sub.f16x2 r1472, r1434, r1449;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1475, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r1476, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r1479, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r1480, {low, high};
}
{
mul.f16x2 r1489, r1463, r1475;
}
{
mul.f16x2 r1492, r1466, r1476;
}
{
sub.f16x2 r1495, r1489, r1492;
}
{
mul.f16x2 r1498, r1463, r1476;
}
{
fma.rn.f16x2 r1501, r1466, r1475, r1498;
}
{
neg.f16x2 r1505, r1457;
}
{
mul.f16x2 r1507, r1469, r1479;
}
{
mul.f16x2 r1510, r1472, r1480;
}
{
sub.f16x2 r1513, r1507, r1510;
}
{
mul.f16x2 r1516, r1469, r1480;
}
{
fma.rn.f16x2 r1519, r1472, r1479, r1516;
}
{
add.f16x2 %0, r1401, r1451;
}
{
add.f16x2 %1, r1404, r1454;
}
{
sub.f16x2 %8, r1401, r1451;
}
{
sub.f16x2 %9, r1404, r1454;
}
{
add.f16x2 %2, r1413, r1495;
}
{
add.f16x2 %3, r1416, r1501;
}
{
sub.f16x2 %10, r1413, r1495;
}
{
sub.f16x2 %11, r1416, r1501;
}
{
add.f16x2 %4, r1407, r1460;
}
{
add.f16x2 %5, r1410, r1505;
}
{
sub.f16x2 %12, r1407, r1460;
}
{
sub.f16x2 %13, r1410, r1505;
}
{
add.f16x2 %6, r1419, r1513;
}
{
add.f16x2 %7, r1422, r1519;
}
{
sub.f16x2 %14, r1419, r1513;
}
{
sub.f16x2 %15, r1422, r1519;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<850, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<423>;
.reg .b32 r<2985>;
.reg .b64 rd<2>;
mov.u32 r2965, %tid.y;
shl.b32 r2966, r2965, 15;
mov.u32 r2967, %32;
add.s32 r2968, r2967, r2966;
mov.u32 r2969, %tid.x;
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
neg.f16x2 r25, r19;
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
add.f16x2 r39, r7, r22;
}
{
add.f16x2 r42, r10, r25;
}
{
sub.f16x2 r45, r7, r22;
}
{
sub.f16x2 r48, r10, r25;
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
neg.f16x2 r75, r69;
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
add.f16x2 r89, r57, r72;
}
{
add.f16x2 r92, r60, r75;
}
{
sub.f16x2 r95, r57, r72;
}
{
sub.f16x2 r98, r60, r75;
}
mov.f32 f362, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f362;
cvt.rn.f16.f32 high, f362;
mov.b32 r101, {low, high};
}
mov.f32 f380, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r102, {low, high};
}
mov.f32 f297, 0fBF800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r106, {low, high};
}
mov.f32 f298, 0f3F800000;
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
neg.f16x2 r131, r83;
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
add.f16x2 r173, r33, r86;
}
{
add.f16x2 r176, r36, r131;
}
{
sub.f16x2 r179, r33, r86;
}
{
sub.f16x2 r182, r36, r131;
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
neg.f16x2 r221, r215;
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
add.f16x2 r235, r203, r218;
}
{
add.f16x2 r238, r206, r221;
}
{
sub.f16x2 r241, r203, r218;
}
{
sub.f16x2 r244, r206, r221;
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
neg.f16x2 r271, r265;
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
add.f16x2 r285, r253, r268;
}
{
add.f16x2 r288, r256, r271;
}
{
sub.f16x2 r291, r253, r268;
}
{
sub.f16x2 r294, r256, r271;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f362;
cvt.rn.f16.f32 high, f362;
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
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
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
neg.f16x2 r327, r279;
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
add.f16x2 r369, r229, r282;
}
{
add.f16x2 r372, r232, r327;
}
{
sub.f16x2 r375, r229, r282;
}
{
sub.f16x2 r378, r232, r327;
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
mov.f32 f358, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f358;
cvt.rn.f16.f32 high, f358;
mov.b32 r393, {low, high};
}
mov.f32 f384, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f362;
cvt.rn.f16.f32 high, f362;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r396, {low, high};
}
mov.f32 f366, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f366;
cvt.rn.f16.f32 high, f366;
mov.b32 r397, {low, high};
}
mov.f32 f382, 0fBF6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f382;
cvt.rn.f16.f32 high, f382;
mov.b32 r398, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f382;
cvt.rn.f16.f32 high, f382;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
mov.b32 r404, {low, high};
}
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
neg.f16x2 r471, r351;
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
add.f16x2 r569, r155, r354;
}
{
add.f16x2 r572, r158, r471;
}
{
sub.f16x2 r575, r155, r354;
}
{
sub.f16x2 r578, r158, r471;
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
and.b32 r2970, r2969, 255;
shl.b32 r2971, r2969, 7;
and.b32 r2972, r2971, -32768;
add.s32 r2973, r2968, r2972;
cvt.rn.f32.u32 f417, r2970;
mul.f32 f418, f417, 0f3AC90FDB;
cos.approx.f32 f117, f418;
sin.approx.f32 f419, f418;
neg.f32 f118, f419;
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
neg.f16x2 r627, r624;
}
{
fma.rn.f16x2 r629, r533, r620, r627;
}
{
mul.f16x2 r633, r533, r622;
}
{
fma.rn.f16x2 r636, r536, r620, r633;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r664, r661;
}
{
fma.rn.f16x2 r666, r545, r657, r664;
}
{
mul.f16x2 r670, r545, r659;
}
{
fma.rn.f16x2 r673, r548, r657, r670;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r701, r698;
}
{
fma.rn.f16x2 r703, r557, r694, r701;
}
{
mul.f16x2 r707, r557, r696;
}
{
fma.rn.f16x2 r710, r560, r694, r707;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r738, r735;
}
{
fma.rn.f16x2 r740, r569, r731, r738;
}
{
mul.f16x2 r744, r569, r733;
}
{
fma.rn.f16x2 r747, r572, r731, r744;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r775, r772;
}
{
fma.rn.f16x2 r777, r581, r768, r775;
}
{
mul.f16x2 r781, r581, r770;
}
{
fma.rn.f16x2 r784, r584, r768, r781;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r812, r809;
}
{
fma.rn.f16x2 r814, r593, r805, r812;
}
{
mul.f16x2 r818, r593, r807;
}
{
fma.rn.f16x2 r821, r596, r805, r818;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r849, r846;
}
{
fma.rn.f16x2 r851, r605, r842, r849;
}
{
mul.f16x2 r855, r605, r844;
}
{
fma.rn.f16x2 r858, r608, r842, r855;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r886, r883;
}
{
fma.rn.f16x2 r888, r527, r879, r886;
}
{
mul.f16x2 r892, r527, r881;
}
{
fma.rn.f16x2 r895, r530, r879, r892;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r923, r920;
}
{
fma.rn.f16x2 r925, r539, r916, r923;
}
{
mul.f16x2 r929, r539, r918;
}
{
fma.rn.f16x2 r932, r542, r916, r929;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r960, r957;
}
{
fma.rn.f16x2 r962, r551, r953, r960;
}
{
mul.f16x2 r966, r551, r955;
}
{
fma.rn.f16x2 r969, r554, r953, r966;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r997, r994;
}
{
fma.rn.f16x2 r999, r563, r990, r997;
}
{
mul.f16x2 r1003, r563, r992;
}
{
fma.rn.f16x2 r1006, r566, r990, r1003;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r1034, r1031;
}
{
fma.rn.f16x2 r1036, r575, r1027, r1034;
}
{
mul.f16x2 r1040, r575, r1029;
}
{
fma.rn.f16x2 r1043, r578, r1027, r1040;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r1071, r1068;
}
{
fma.rn.f16x2 r1073, r587, r1064, r1071;
}
{
mul.f16x2 r1077, r587, r1066;
}
{
fma.rn.f16x2 r1080, r590, r1064, r1077;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r1108, r1105;
}
{
fma.rn.f16x2 r1110, r599, r1101, r1108;
}
{
mul.f16x2 r1114, r599, r1103;
}
{
fma.rn.f16x2 r1117, r602, r1101, r1114;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r1145, r1142;
}
{
fma.rn.f16x2 r1147, r611, r1138, r1145;
}
{
mul.f16x2 r1151, r611, r1140;
}
{
fma.rn.f16x2 r1154, r614, r1138, r1151;
}
barrier.sync 0;
and.b32 r2974, r2971, 32640;
add.s32 r2975, r2973, r2974;
st.shared.v4.f32 [r2975], {r521, r524, r629, r636};
st.shared.v4.f32 [r2975+16], {r666, r673, r703, r710};
st.shared.v4.f32 [r2975+32], {r740, r747, r777, r784};
st.shared.v4.f32 [r2975+48], {r814, r821, r851, r858};
st.shared.v4.f32 [r2975+64], {r888, r895, r925, r932};
st.shared.v4.f32 [r2975+80], {r962, r969, r999, r1006};
st.shared.v4.f32 [r2975+96], {r1036, r1043, r1073, r1080};
st.shared.v4.f32 [r2975+112], {r1110, r1117, r1147, r1154};
barrier.sync 0;
mad.lo.s32 r2976, r2970, -120, r2975;
ld.shared.u32 r1176, [r2976];
ld.shared.u32 r1179, [r2976+4];
ld.shared.u32 r1372, [r2976+2048];
ld.shared.u32 r1375, [r2976+2052];
ld.shared.u32 r1226, [r2976+4096];
ld.shared.u32 r1229, [r2976+4100];
ld.shared.u32 r1422, [r2976+6144];
ld.shared.u32 r1425, [r2976+6148];
ld.shared.u32 r1188, [r2976+8192];
ld.shared.u32 r1191, [r2976+8196];
ld.shared.u32 r1384, [r2976+10240];
ld.shared.u32 r1387, [r2976+10244];
ld.shared.u32 r1238, [r2976+12288];
ld.shared.u32 r1241, [r2976+12292];
ld.shared.u32 r1434, [r2976+14336];
ld.shared.u32 r1437, [r2976+14340];
ld.shared.u32 r1177, [r2976+16384];
ld.shared.u32 r1180, [r2976+16388];
ld.shared.u32 r1373, [r2976+18432];
ld.shared.u32 r1376, [r2976+18436];
ld.shared.u32 r1227, [r2976+20480];
ld.shared.u32 r1230, [r2976+20484];
ld.shared.u32 r1423, [r2976+22528];
ld.shared.u32 r1426, [r2976+22532];
ld.shared.u32 r1189, [r2976+24576];
ld.shared.u32 r1192, [r2976+24580];
ld.shared.u32 r1385, [r2976+26624];
ld.shared.u32 r1388, [r2976+26628];
ld.shared.u32 r1239, [r2976+28672];
ld.shared.u32 r1242, [r2976+28676];
ld.shared.u32 r1435, [r2976+30720];
ld.shared.u32 r1438, [r2976+30724];
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
neg.f16x2 r1199, r1193;
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
add.f16x2 r1213, r1181, r1196;
}
{
add.f16x2 r1216, r1184, r1199;
}
{
sub.f16x2 r1219, r1181, r1196;
}
{
sub.f16x2 r1222, r1184, r1199;
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
neg.f16x2 r1249, r1243;
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
add.f16x2 r1263, r1231, r1246;
}
{
add.f16x2 r1266, r1234, r1249;
}
{
sub.f16x2 r1269, r1231, r1246;
}
{
sub.f16x2 r1272, r1234, r1249;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f362;
cvt.rn.f16.f32 high, f362;
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
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
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
neg.f16x2 r1305, r1257;
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
add.f16x2 r1347, r1207, r1260;
}
{
add.f16x2 r1350, r1210, r1305;
}
{
sub.f16x2 r1353, r1207, r1260;
}
{
sub.f16x2 r1356, r1210, r1305;
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
neg.f16x2 r1395, r1389;
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
add.f16x2 r1409, r1377, r1392;
}
{
add.f16x2 r1412, r1380, r1395;
}
{
sub.f16x2 r1415, r1377, r1392;
}
{
sub.f16x2 r1418, r1380, r1395;
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
neg.f16x2 r1445, r1439;
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
add.f16x2 r1459, r1427, r1442;
}
{
add.f16x2 r1462, r1430, r1445;
}
{
sub.f16x2 r1465, r1427, r1442;
}
{
sub.f16x2 r1468, r1430, r1445;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f362;
cvt.rn.f16.f32 high, f362;
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
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
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
neg.f16x2 r1501, r1453;
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
add.f16x2 r1543, r1403, r1456;
}
{
add.f16x2 r1546, r1406, r1501;
}
{
sub.f16x2 r1549, r1403, r1456;
}
{
sub.f16x2 r1552, r1406, r1501;
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
cvt.rn.f16.f32 low, f358;
cvt.rn.f16.f32 high, f358;
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
cvt.rn.f16.f32 low, f362;
cvt.rn.f16.f32 high, f362;
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
cvt.rn.f16.f32 low, f366;
cvt.rn.f16.f32 high, f366;
mov.b32 r1571, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f382;
cvt.rn.f16.f32 high, f382;
mov.b32 r1572, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r1575, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f382;
cvt.rn.f16.f32 high, f382;
mov.b32 r1576, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
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
neg.f16x2 r1645, r1525;
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
add.f16x2 r1743, r1329, r1528;
}
{
add.f16x2 r1746, r1332, r1645;
}
{
sub.f16x2 r1749, r1329, r1528;
}
{
sub.f16x2 r1752, r1332, r1645;
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
and.b32 r2977, r2969, 240;
bfe.u32 r2978, r2969, 4, 4;
cvt.rn.f32.u32 f420, r2978;
mul.f32 f421, f420, 0f3CC90FDB;
cos.approx.f32 f267, f421;
sin.approx.f32 f422, f421;
neg.f32 f268, f422;
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
neg.f16x2 r1801, r1798;
}
{
fma.rn.f16x2 r1803, r1707, r1794, r1801;
}
{
mul.f16x2 r1807, r1707, r1796;
}
{
fma.rn.f16x2 r1810, r1710, r1794, r1807;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r1838, r1835;
}
{
fma.rn.f16x2 r1840, r1719, r1831, r1838;
}
{
mul.f16x2 r1844, r1719, r1833;
}
{
fma.rn.f16x2 r1847, r1722, r1831, r1844;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r1875, r1872;
}
{
fma.rn.f16x2 r1877, r1731, r1868, r1875;
}
{
mul.f16x2 r1881, r1731, r1870;
}
{
fma.rn.f16x2 r1884, r1734, r1868, r1881;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r1912, r1909;
}
{
fma.rn.f16x2 r1914, r1743, r1905, r1912;
}
{
mul.f16x2 r1918, r1743, r1907;
}
{
fma.rn.f16x2 r1921, r1746, r1905, r1918;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r1949, r1946;
}
{
fma.rn.f16x2 r1951, r1755, r1942, r1949;
}
{
mul.f16x2 r1955, r1755, r1944;
}
{
fma.rn.f16x2 r1958, r1758, r1942, r1955;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r1986, r1983;
}
{
fma.rn.f16x2 r1988, r1767, r1979, r1986;
}
{
mul.f16x2 r1992, r1767, r1981;
}
{
fma.rn.f16x2 r1995, r1770, r1979, r1992;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r2023, r2020;
}
{
fma.rn.f16x2 r2025, r1779, r2016, r2023;
}
{
mul.f16x2 r2029, r1779, r2018;
}
{
fma.rn.f16x2 r2032, r1782, r2016, r2029;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r2060, r2057;
}
{
fma.rn.f16x2 r2062, r1701, r2053, r2060;
}
{
mul.f16x2 r2066, r1701, r2055;
}
{
fma.rn.f16x2 r2069, r1704, r2053, r2066;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r2097, r2094;
}
{
fma.rn.f16x2 r2099, r1713, r2090, r2097;
}
{
mul.f16x2 r2103, r1713, r2092;
}
{
fma.rn.f16x2 r2106, r1716, r2090, r2103;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r2134, r2131;
}
{
fma.rn.f16x2 r2136, r1725, r2127, r2134;
}
{
mul.f16x2 r2140, r1725, r2129;
}
{
fma.rn.f16x2 r2143, r1728, r2127, r2140;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r2171, r2168;
}
{
fma.rn.f16x2 r2173, r1737, r2164, r2171;
}
{
mul.f16x2 r2177, r1737, r2166;
}
{
fma.rn.f16x2 r2180, r1740, r2164, r2177;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r2208, r2205;
}
{
fma.rn.f16x2 r2210, r1749, r2201, r2208;
}
{
mul.f16x2 r2214, r1749, r2203;
}
{
fma.rn.f16x2 r2217, r1752, r2201, r2214;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r2245, r2242;
}
{
fma.rn.f16x2 r2247, r1761, r2238, r2245;
}
{
mul.f16x2 r2251, r1761, r2240;
}
{
fma.rn.f16x2 r2254, r1764, r2238, r2251;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r2282, r2279;
}
{
fma.rn.f16x2 r2284, r1773, r2275, r2282;
}
{
mul.f16x2 r2288, r1773, r2277;
}
{
fma.rn.f16x2 r2291, r1776, r2275, r2288;
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
cvt.rn.f16.f32 low, f297;
cvt.rn.f16.f32 high, f298;
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
neg.f16x2 r2319, r2316;
}
{
fma.rn.f16x2 r2321, r1785, r2312, r2319;
}
{
mul.f16x2 r2325, r1785, r2314;
}
{
fma.rn.f16x2 r2328, r1788, r2312, r2325;
}
shl.b32 r2979, r2969, 3;
and.b32 r2980, r2979, 120;
add.s32 r2981, r2973, r2980;
barrier.sync 0;
and.b32 r2982, r2971, 30720;
add.s32 r2983, r2981, r2982;
st.shared.u32 [r2983], r1695;
st.shared.u32 [r2983+4], r1698;
st.shared.u32 [r2983+128], r1803;
st.shared.u32 [r2983+132], r1810;
st.shared.u32 [r2983+256], r1840;
st.shared.u32 [r2983+260], r1847;
st.shared.u32 [r2983+384], r1877;
st.shared.u32 [r2983+388], r1884;
st.shared.u32 [r2983+512], r1914;
st.shared.u32 [r2983+516], r1921;
st.shared.u32 [r2983+640], r1951;
st.shared.u32 [r2983+644], r1958;
st.shared.u32 [r2983+768], r1988;
st.shared.u32 [r2983+772], r1995;
st.shared.u32 [r2983+896], r2025;
st.shared.u32 [r2983+900], r2032;
st.shared.u32 [r2983+1024], r2062;
st.shared.u32 [r2983+1028], r2069;
st.shared.u32 [r2983+1152], r2099;
st.shared.u32 [r2983+1156], r2106;
st.shared.u32 [r2983+1280], r2136;
st.shared.u32 [r2983+1284], r2143;
st.shared.u32 [r2983+1408], r2173;
st.shared.u32 [r2983+1412], r2180;
st.shared.u32 [r2983+1536], r2210;
st.shared.u32 [r2983+1540], r2217;
st.shared.u32 [r2983+1664], r2247;
st.shared.u32 [r2983+1668], r2254;
st.shared.u32 [r2983+1792], r2284;
st.shared.u32 [r2983+1796], r2291;
st.shared.u32 [r2983+1920], r2321;
st.shared.u32 [r2983+1924], r2328;
barrier.sync 0;
mad.lo.s32 r2984, r2977, -120, r2983;
ld.shared.u32 r2350, [r2984];
ld.shared.u32 r2353, [r2984+4];
ld.shared.u32 r2546, [r2984+2048];
ld.shared.u32 r2549, [r2984+2052];
ld.shared.u32 r2400, [r2984+4096];
ld.shared.u32 r2403, [r2984+4100];
ld.shared.u32 r2596, [r2984+6144];
ld.shared.u32 r2599, [r2984+6148];
ld.shared.u32 r2362, [r2984+8192];
ld.shared.u32 r2365, [r2984+8196];
ld.shared.u32 r2558, [r2984+10240];
ld.shared.u32 r2561, [r2984+10244];
ld.shared.u32 r2412, [r2984+12288];
ld.shared.u32 r2415, [r2984+12292];
ld.shared.u32 r2608, [r2984+14336];
ld.shared.u32 r2611, [r2984+14340];
ld.shared.u32 r2351, [r2984+16384];
ld.shared.u32 r2354, [r2984+16388];
ld.shared.u32 r2547, [r2984+18432];
ld.shared.u32 r2550, [r2984+18436];
ld.shared.u32 r2401, [r2984+20480];
ld.shared.u32 r2404, [r2984+20484];
ld.shared.u32 r2597, [r2984+22528];
ld.shared.u32 r2600, [r2984+22532];
ld.shared.u32 r2363, [r2984+24576];
ld.shared.u32 r2366, [r2984+24580];
ld.shared.u32 r2559, [r2984+26624];
ld.shared.u32 r2562, [r2984+26628];
ld.shared.u32 r2413, [r2984+28672];
ld.shared.u32 r2416, [r2984+28676];
ld.shared.u32 r2609, [r2984+30720];
ld.shared.u32 r2612, [r2984+30724];
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
neg.f16x2 r2373, r2367;
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
add.f16x2 r2387, r2355, r2370;
}
{
add.f16x2 r2390, r2358, r2373;
}
{
sub.f16x2 r2393, r2355, r2370;
}
{
sub.f16x2 r2396, r2358, r2373;
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
neg.f16x2 r2423, r2417;
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
add.f16x2 r2437, r2405, r2420;
}
{
add.f16x2 r2440, r2408, r2423;
}
{
sub.f16x2 r2443, r2405, r2420;
}
{
sub.f16x2 r2446, r2408, r2423;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f362;
cvt.rn.f16.f32 high, f362;
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
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
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
neg.f16x2 r2479, r2431;
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
add.f16x2 r2521, r2381, r2434;
}
{
add.f16x2 r2524, r2384, r2479;
}
{
sub.f16x2 r2527, r2381, r2434;
}
{
sub.f16x2 r2530, r2384, r2479;
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
neg.f16x2 r2569, r2563;
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
add.f16x2 r2583, r2551, r2566;
}
{
add.f16x2 r2586, r2554, r2569;
}
{
sub.f16x2 r2589, r2551, r2566;
}
{
sub.f16x2 r2592, r2554, r2569;
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
neg.f16x2 r2619, r2613;
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
add.f16x2 r2633, r2601, r2616;
}
{
add.f16x2 r2636, r2604, r2619;
}
{
sub.f16x2 r2639, r2601, r2616;
}
{
sub.f16x2 r2642, r2604, r2619;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f362;
cvt.rn.f16.f32 high, f362;
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
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
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
neg.f16x2 r2675, r2627;
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
add.f16x2 r2717, r2577, r2630;
}
{
add.f16x2 r2720, r2580, r2675;
}
{
sub.f16x2 r2723, r2577, r2630;
}
{
sub.f16x2 r2726, r2580, r2675;
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
cvt.rn.f16.f32 low, f358;
cvt.rn.f16.f32 high, f358;
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
cvt.rn.f16.f32 low, f362;
cvt.rn.f16.f32 high, f362;
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
cvt.rn.f16.f32 low, f366;
cvt.rn.f16.f32 high, f366;
mov.b32 r2745, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f382;
cvt.rn.f16.f32 high, f382;
mov.b32 r2746, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f384;
cvt.rn.f16.f32 high, f384;
mov.b32 r2749, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f382;
cvt.rn.f16.f32 high, f382;
mov.b32 r2750, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f380;
cvt.rn.f16.f32 high, f380;
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
neg.f16x2 r2819, r2699;
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
add.f16x2 %0, r2497, r2693;
}
{
add.f16x2 %1, r2500, r2696;
}
{
sub.f16x2 %16, r2497, r2693;
}
{
sub.f16x2 %17, r2500, r2696;
}
{
add.f16x2 %2, r2509, r2777;
}
{
add.f16x2 %3, r2512, r2783;
}
{
sub.f16x2 %18, r2509, r2777;
}
{
sub.f16x2 %19, r2512, r2783;
}
{
add.f16x2 %4, r2521, r2793;
}
{
add.f16x2 %5, r2524, r2799;
}
{
sub.f16x2 %20, r2521, r2793;
}
{
sub.f16x2 %21, r2524, r2799;
}
{
add.f16x2 %6, r2533, r2809;
}
{
add.f16x2 %7, r2536, r2815;
}
{
sub.f16x2 %22, r2533, r2809;
}
{
sub.f16x2 %23, r2536, r2815;
}
{
add.f16x2 %8, r2503, r2702;
}
{
add.f16x2 %9, r2506, r2819;
}
{
sub.f16x2 %24, r2503, r2702;
}
{
sub.f16x2 %25, r2506, r2819;
}
{
add.f16x2 %10, r2515, r2827;
}
{
add.f16x2 %11, r2518, r2833;
}
{
sub.f16x2 %26, r2515, r2827;
}
{
sub.f16x2 %27, r2518, r2833;
}
{
add.f16x2 %12, r2527, r2843;
}
{
add.f16x2 %13, r2530, r2849;
}
{
sub.f16x2 %28, r2527, r2843;
}
{
sub.f16x2 %29, r2530, r2849;
}
{
add.f16x2 %14, r2539, r2859;
}
{
add.f16x2 %15, r2542, r2865;
}
{
sub.f16x2 %30, r2539, r2859;
}
{
sub.f16x2 %31, r2542, r2865;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<851, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<853>;
.reg .b32 r<6211>;
.reg .b64 rd<3>;
mov.u32 r6125, %tid.y;
shl.b32 r6126, r6125, 15;
mov.u32 r6127, %64;
add.s32 r6128, r6127, r6126;
mov.u32 r6129, %tid.x;
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
add.f16x2 r16, %106, %101;
}
{
sub.f16x2 r19, %73, %128;
}
{
sub.f16x2 r22, %106, %101;
}
{
neg.f16x2 r25, r19;
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
add.f16x2 r39, r7, r22;
}
{
add.f16x2 r42, r10, r25;
}
{
sub.f16x2 r45, r7, r22;
}
{
sub.f16x2 r48, r10, r25;
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
add.f16x2 r63, %121, %115;
}
{
add.f16x2 r66, %93, %85;
}
{
sub.f16x2 r69, %121, %115;
}
{
sub.f16x2 r72, %93, %85;
}
{
neg.f16x2 r75, r69;
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
add.f16x2 r89, r57, r72;
}
{
add.f16x2 r92, r60, r75;
}
{
sub.f16x2 r95, r57, r72;
}
{
sub.f16x2 r98, r60, r75;
}
mov.f32 f668, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r101, {low, high};
}
mov.f32 f702, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r102, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
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
neg.f16x2 r131, r83;
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
add.f16x2 r173, r33, r86;
}
{
add.f16x2 r176, r36, r131;
}
{
sub.f16x2 r179, r33, r86;
}
{
sub.f16x2 r182, r36, r131;
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
neg.f16x2 r221, r215;
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
add.f16x2 r235, r203, r218;
}
{
add.f16x2 r238, r206, r221;
}
{
sub.f16x2 r241, r203, r218;
}
{
sub.f16x2 r244, r206, r221;
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
neg.f16x2 r271, r265;
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
add.f16x2 r285, r253, r268;
}
{
add.f16x2 r288, r256, r271;
}
{
sub.f16x2 r291, r253, r268;
}
{
sub.f16x2 r294, r256, r271;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
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
neg.f16x2 r327, r279;
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
add.f16x2 r369, r229, r282;
}
{
add.f16x2 r372, r232, r327;
}
{
sub.f16x2 r375, r229, r282;
}
{
sub.f16x2 r378, r232, r327;
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
mov.f32 f660, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f660;
cvt.rn.f16.f32 high, f660;
mov.b32 r393, {low, high};
}
mov.f32 f710, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r396, {low, high};
}
mov.f32 f676, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f676;
cvt.rn.f16.f32 high, f676;
mov.b32 r397, {low, high};
}
mov.f32 f708, 0fBF6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r398, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r404, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r405, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
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
neg.f16x2 r471, r351;
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
add.f16x2 r569, r155, r354;
}
{
add.f16x2 r572, r158, r471;
}
{
sub.f16x2 r575, r155, r354;
}
{
sub.f16x2 r578, r158, r471;
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
add.f16x2 r632, %122, %114;
}
{
sub.f16x2 r635, %86, %76;
}
{
sub.f16x2 r638, %122, %114;
}
{
neg.f16x2 r641, r635;
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
add.f16x2 r655, r623, r638;
}
{
add.f16x2 r658, r626, r641;
}
{
sub.f16x2 r661, r623, r638;
}
{
sub.f16x2 r664, r626, r641;
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
neg.f16x2 r691, r685;
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
add.f16x2 r705, r673, r688;
}
{
add.f16x2 r708, r676, r691;
}
{
sub.f16x2 r711, r673, r688;
}
{
sub.f16x2 r714, r676, r691;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r717, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r718, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r721, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
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
neg.f16x2 r747, r699;
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
add.f16x2 r789, r649, r702;
}
{
add.f16x2 r792, r652, r747;
}
{
sub.f16x2 r795, r649, r702;
}
{
sub.f16x2 r798, r652, r747;
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
add.f16x2 r813, %109, %100;
}
{
add.f16x2 r816, %79, %71;
}
{
sub.f16x2 r819, %109, %100;
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
neg.f16x2 r837, r831;
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
add.f16x2 r851, r819, r834;
}
{
add.f16x2 r854, r822, r837;
}
{
sub.f16x2 r857, r819, r834;
}
{
sub.f16x2 r860, r822, r837;
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
neg.f16x2 r887, r881;
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
add.f16x2 r901, r869, r884;
}
{
add.f16x2 r904, r872, r887;
}
{
sub.f16x2 r907, r869, r884;
}
{
sub.f16x2 r910, r872, r887;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r913, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r914, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r917, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
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
neg.f16x2 r943, r895;
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
add.f16x2 r985, r845, r898;
}
{
add.f16x2 r988, r848, r943;
}
{
sub.f16x2 r991, r845, r898;
}
{
sub.f16x2 r994, r848, r943;
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
cvt.rn.f16.f32 low, f660;
cvt.rn.f16.f32 high, f660;
mov.b32 r1009, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r1010, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r1011, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r1012, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f676;
cvt.rn.f16.f32 high, f676;
mov.b32 r1013, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r1014, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r1017, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r1018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r1019, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r1020, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r1021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
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
neg.f16x2 r1087, r967;
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
add.f16x2 r1185, r771, r970;
}
{
add.f16x2 r1188, r774, r1087;
}
{
sub.f16x2 r1191, r771, r970;
}
{
sub.f16x2 r1194, r774, r1087;
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
mov.f32 f656, 0f3F7B14BE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f656;
cvt.rn.f16.f32 high, f656;
mov.b32 r1233, {low, high};
}
mov.f32 f714, 0fBE47C5C2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f714;
cvt.rn.f16.f32 high, f714;
mov.b32 r1234, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f660;
cvt.rn.f16.f32 high, f660;
mov.b32 r1235, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r1236, {low, high};
}
mov.f32 f664, 0f3F54DB31;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f664;
cvt.rn.f16.f32 high, f664;
mov.b32 r1237, {low, high};
}
mov.f32 f706, 0fBF0E39DA;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f706;
cvt.rn.f16.f32 high, f706;
mov.b32 r1238, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r1239, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r1240, {low, high};
}
mov.f32 f672, 0f3F0E39DA;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f672;
cvt.rn.f16.f32 high, f672;
mov.b32 r1241, {low, high};
}
mov.f32 f704, 0fBF54DB31;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f704;
cvt.rn.f16.f32 high, f704;
mov.b32 r1242, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f676;
cvt.rn.f16.f32 high, f676;
mov.b32 r1243, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r1244, {low, high};
}
mov.f32 f680, 0f3E47C5C2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f680;
cvt.rn.f16.f32 high, f680;
mov.b32 r1245, {low, high};
}
mov.f32 f712, 0fBF7B14BE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f712;
cvt.rn.f16.f32 high, f712;
mov.b32 r1246, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f714;
cvt.rn.f16.f32 high, f714;
mov.b32 r1249, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f712;
cvt.rn.f16.f32 high, f712;
mov.b32 r1250, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r1251, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r1252, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f706;
cvt.rn.f16.f32 high, f706;
mov.b32 r1253, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f704;
cvt.rn.f16.f32 high, f704;
mov.b32 r1254, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r1255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r1256, {low, high};
}
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
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r1259, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r1260, {low, high};
}
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
neg.f16x2 r1407, r1143;
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
add.f16x2 r1617, r527, r1146;
}
{
add.f16x2 r1620, r530, r1407;
}
{
sub.f16x2 r1623, r527, r1146;
}
{
sub.f16x2 r1626, r530, r1407;
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
shl.b32 r6131, r6129, 8;
and.b32 r6132, r6131, -32768;
add.s32 r6133, r6128, r6132;
and.b32 r6146, r6129, 127;
cvt.rn.f32.u32 f845, r6146;
mul.f32 f846, f845, 0f3AC90FDB;
cos.approx.f32 f357, f846;
sin.approx.f32 f847, f846;
neg.f32 f358, f847;
mov.f32 f852, 0f3F800000;
mov.f32 f851, 0fBF800000;
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
neg.f16x2 r1723, r1720;
}
{
fma.rn.f16x2 r1725, r1533, r1716, r1723;
}
{
mul.f16x2 r1729, r1533, r1718;
}
{
fma.rn.f16x2 r1732, r1536, r1716, r1729;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r1760, r1757;
}
{
fma.rn.f16x2 r1762, r1545, r1753, r1760;
}
{
mul.f16x2 r1766, r1545, r1755;
}
{
fma.rn.f16x2 r1769, r1548, r1753, r1766;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r1797, r1794;
}
{
fma.rn.f16x2 r1799, r1557, r1790, r1797;
}
{
mul.f16x2 r1803, r1557, r1792;
}
{
fma.rn.f16x2 r1806, r1560, r1790, r1803;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r1834, r1831;
}
{
fma.rn.f16x2 r1836, r1569, r1827, r1834;
}
{
mul.f16x2 r1840, r1569, r1829;
}
{
fma.rn.f16x2 r1843, r1572, r1827, r1840;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r1871, r1868;
}
{
fma.rn.f16x2 r1873, r1581, r1864, r1871;
}
{
mul.f16x2 r1877, r1581, r1866;
}
{
fma.rn.f16x2 r1880, r1584, r1864, r1877;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r1908, r1905;
}
{
fma.rn.f16x2 r1910, r1593, r1901, r1908;
}
{
mul.f16x2 r1914, r1593, r1903;
}
{
fma.rn.f16x2 r1917, r1596, r1901, r1914;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r1945, r1942;
}
{
fma.rn.f16x2 r1947, r1605, r1938, r1945;
}
{
mul.f16x2 r1951, r1605, r1940;
}
{
fma.rn.f16x2 r1954, r1608, r1938, r1951;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r1982, r1979;
}
{
fma.rn.f16x2 r1984, r1617, r1975, r1982;
}
{
mul.f16x2 r1988, r1617, r1977;
}
{
fma.rn.f16x2 r1991, r1620, r1975, r1988;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2019, r2016;
}
{
fma.rn.f16x2 r2021, r1629, r2012, r2019;
}
{
mul.f16x2 r2025, r1629, r2014;
}
{
fma.rn.f16x2 r2028, r1632, r2012, r2025;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2056, r2053;
}
{
fma.rn.f16x2 r2058, r1641, r2049, r2056;
}
{
mul.f16x2 r2062, r1641, r2051;
}
{
fma.rn.f16x2 r2065, r1644, r2049, r2062;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2093, r2090;
}
{
fma.rn.f16x2 r2095, r1653, r2086, r2093;
}
{
mul.f16x2 r2099, r1653, r2088;
}
{
fma.rn.f16x2 r2102, r1656, r2086, r2099;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2130, r2127;
}
{
fma.rn.f16x2 r2132, r1665, r2123, r2130;
}
{
mul.f16x2 r2136, r1665, r2125;
}
{
fma.rn.f16x2 r2139, r1668, r2123, r2136;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2167, r2164;
}
{
fma.rn.f16x2 r2169, r1677, r2160, r2167;
}
{
mul.f16x2 r2173, r1677, r2162;
}
{
fma.rn.f16x2 r2176, r1680, r2160, r2173;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2204, r2201;
}
{
fma.rn.f16x2 r2206, r1689, r2197, r2204;
}
{
mul.f16x2 r2210, r1689, r2199;
}
{
fma.rn.f16x2 r2213, r1692, r2197, r2210;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2241, r2238;
}
{
fma.rn.f16x2 r2243, r1701, r2234, r2241;
}
{
mul.f16x2 r2247, r1701, r2236;
}
{
fma.rn.f16x2 r2250, r1704, r2234, r2247;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2278, r2275;
}
{
fma.rn.f16x2 r2280, r1527, r2271, r2278;
}
{
mul.f16x2 r2284, r1527, r2273;
}
{
fma.rn.f16x2 r2287, r1530, r2271, r2284;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2315, r2312;
}
{
fma.rn.f16x2 r2317, r1539, r2308, r2315;
}
{
mul.f16x2 r2321, r1539, r2310;
}
{
fma.rn.f16x2 r2324, r1542, r2308, r2321;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2352, r2349;
}
{
fma.rn.f16x2 r2354, r1551, r2345, r2352;
}
{
mul.f16x2 r2358, r1551, r2347;
}
{
fma.rn.f16x2 r2361, r1554, r2345, r2358;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2389, r2386;
}
{
fma.rn.f16x2 r2391, r1563, r2382, r2389;
}
{
mul.f16x2 r2395, r1563, r2384;
}
{
fma.rn.f16x2 r2398, r1566, r2382, r2395;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2426, r2423;
}
{
fma.rn.f16x2 r2428, r1575, r2419, r2426;
}
{
mul.f16x2 r2432, r1575, r2421;
}
{
fma.rn.f16x2 r2435, r1578, r2419, r2432;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2463, r2460;
}
{
fma.rn.f16x2 r2465, r1587, r2456, r2463;
}
{
mul.f16x2 r2469, r1587, r2458;
}
{
fma.rn.f16x2 r2472, r1590, r2456, r2469;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2500, r2497;
}
{
fma.rn.f16x2 r2502, r1599, r2493, r2500;
}
{
mul.f16x2 r2506, r1599, r2495;
}
{
fma.rn.f16x2 r2509, r1602, r2493, r2506;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2537, r2534;
}
{
fma.rn.f16x2 r2539, r1611, r2530, r2537;
}
{
mul.f16x2 r2543, r1611, r2532;
}
{
fma.rn.f16x2 r2546, r1614, r2530, r2543;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2574, r2571;
}
{
fma.rn.f16x2 r2576, r1623, r2567, r2574;
}
{
mul.f16x2 r2580, r1623, r2569;
}
{
fma.rn.f16x2 r2583, r1626, r2567, r2580;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2611, r2608;
}
{
fma.rn.f16x2 r2613, r1635, r2604, r2611;
}
{
mul.f16x2 r2617, r1635, r2606;
}
{
fma.rn.f16x2 r2620, r1638, r2604, r2617;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2648, r2645;
}
{
fma.rn.f16x2 r2650, r1647, r2641, r2648;
}
{
mul.f16x2 r2654, r1647, r2643;
}
{
fma.rn.f16x2 r2657, r1650, r2641, r2654;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2685, r2682;
}
{
fma.rn.f16x2 r2687, r1659, r2678, r2685;
}
{
mul.f16x2 r2691, r1659, r2680;
}
{
fma.rn.f16x2 r2694, r1662, r2678, r2691;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2722, r2719;
}
{
fma.rn.f16x2 r2724, r1671, r2715, r2722;
}
{
mul.f16x2 r2728, r1671, r2717;
}
{
fma.rn.f16x2 r2731, r1674, r2715, r2728;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2759, r2756;
}
{
fma.rn.f16x2 r2761, r1683, r2752, r2759;
}
{
mul.f16x2 r2765, r1683, r2754;
}
{
fma.rn.f16x2 r2768, r1686, r2752, r2765;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2796, r2793;
}
{
fma.rn.f16x2 r2798, r1695, r2789, r2796;
}
{
mul.f16x2 r2802, r1695, r2791;
}
{
fma.rn.f16x2 r2805, r1698, r2789, r2802;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2833, r2830;
}
{
fma.rn.f16x2 r2835, r1707, r2826, r2833;
}
{
mul.f16x2 r2839, r1707, r2828;
}
{
fma.rn.f16x2 r2842, r1710, r2826, r2839;
}
barrier.sync 0;
and.b32 r6134, r6131, 32512;
add.s32 r6135, r6133, r6134;
st.shared.v4.f32 [r6135], {r1521, r1524, r1725, r1732};
st.shared.v4.f32 [r6135+16], {r1762, r1769, r1799, r1806};
st.shared.v4.f32 [r6135+32], {r1836, r1843, r1873, r1880};
st.shared.v4.f32 [r6135+48], {r1910, r1917, r1947, r1954};
st.shared.v4.f32 [r6135+64], {r1984, r1991, r2021, r2028};
st.shared.v4.f32 [r6135+80], {r2058, r2065, r2095, r2102};
st.shared.v4.f32 [r6135+96], {r2132, r2139, r2169, r2176};
st.shared.v4.f32 [r6135+112], {r2206, r2213, r2243, r2250};
st.shared.v4.f32 [r6135+128], {r2280, r2287, r2317, r2324};
st.shared.v4.f32 [r6135+144], {r2354, r2361, r2391, r2398};
st.shared.v4.f32 [r6135+160], {r2428, r2435, r2465, r2472};
st.shared.v4.f32 [r6135+176], {r2502, r2509, r2539, r2546};
st.shared.v4.f32 [r6135+192], {r2576, r2583, r2613, r2620};
st.shared.v4.f32 [r6135+208], {r2650, r2657, r2687, r2694};
st.shared.v4.f32 [r6135+224], {r2724, r2731, r2761, r2768};
st.shared.v4.f32 [r6135+240], {r2798, r2805, r2835, r2842};
barrier.sync 0;
mad.lo.s32 r6136, r6146, -248, r6135;
ld.shared.u32 r2864, [r6136];
ld.shared.u32 r2867, [r6136+4];
ld.shared.u32 r3480, [r6136+1024];
ld.shared.u32 r3483, [r6136+1028];
ld.shared.u32 r3060, [r6136+2048];
ld.shared.u32 r3063, [r6136+2052];
ld.shared.u32 r3676, [r6136+3072];
ld.shared.u32 r3679, [r6136+3076];
ld.shared.u32 r2914, [r6136+4096];
ld.shared.u32 r2917, [r6136+4100];
ld.shared.u32 r3530, [r6136+5120];
ld.shared.u32 r3533, [r6136+5124];
ld.shared.u32 r3110, [r6136+6144];
ld.shared.u32 r3113, [r6136+6148];
ld.shared.u32 r3726, [r6136+7168];
ld.shared.u32 r3729, [r6136+7172];
ld.shared.u32 r2876, [r6136+8192];
ld.shared.u32 r2879, [r6136+8196];
ld.shared.u32 r3492, [r6136+9216];
ld.shared.u32 r3495, [r6136+9220];
ld.shared.u32 r3072, [r6136+10240];
ld.shared.u32 r3075, [r6136+10244];
ld.shared.u32 r3688, [r6136+11264];
ld.shared.u32 r3691, [r6136+11268];
ld.shared.u32 r2926, [r6136+12288];
ld.shared.u32 r2929, [r6136+12292];
ld.shared.u32 r3542, [r6136+13312];
ld.shared.u32 r3545, [r6136+13316];
ld.shared.u32 r3122, [r6136+14336];
ld.shared.u32 r3125, [r6136+14340];
ld.shared.u32 r3738, [r6136+15360];
ld.shared.u32 r3741, [r6136+15364];
ld.shared.u32 r2865, [r6136+16384];
ld.shared.u32 r2868, [r6136+16388];
ld.shared.u32 r3481, [r6136+17408];
ld.shared.u32 r3484, [r6136+17412];
ld.shared.u32 r3061, [r6136+18432];
ld.shared.u32 r3064, [r6136+18436];
ld.shared.u32 r3677, [r6136+19456];
ld.shared.u32 r3680, [r6136+19460];
ld.shared.u32 r2915, [r6136+20480];
ld.shared.u32 r2918, [r6136+20484];
ld.shared.u32 r3531, [r6136+21504];
ld.shared.u32 r3534, [r6136+21508];
ld.shared.u32 r3111, [r6136+22528];
ld.shared.u32 r3114, [r6136+22532];
ld.shared.u32 r3727, [r6136+23552];
ld.shared.u32 r3730, [r6136+23556];
ld.shared.u32 r2877, [r6136+24576];
ld.shared.u32 r2880, [r6136+24580];
ld.shared.u32 r3493, [r6136+25600];
ld.shared.u32 r3496, [r6136+25604];
ld.shared.u32 r3073, [r6136+26624];
ld.shared.u32 r3076, [r6136+26628];
ld.shared.u32 r3689, [r6136+27648];
ld.shared.u32 r3692, [r6136+27652];
ld.shared.u32 r2927, [r6136+28672];
ld.shared.u32 r2930, [r6136+28676];
ld.shared.u32 r3543, [r6136+29696];
ld.shared.u32 r3546, [r6136+29700];
ld.shared.u32 r3123, [r6136+30720];
ld.shared.u32 r3126, [r6136+30724];
ld.shared.u32 r3739, [r6136+31744];
ld.shared.u32 r3742, [r6136+31748];
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
neg.f16x2 r2887, r2881;
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
add.f16x2 r2901, r2869, r2884;
}
{
add.f16x2 r2904, r2872, r2887;
}
{
sub.f16x2 r2907, r2869, r2884;
}
{
sub.f16x2 r2910, r2872, r2887;
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
neg.f16x2 r2937, r2931;
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
add.f16x2 r2951, r2919, r2934;
}
{
add.f16x2 r2954, r2922, r2937;
}
{
sub.f16x2 r2957, r2919, r2934;
}
{
sub.f16x2 r2960, r2922, r2937;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r2963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r2964, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r2967, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
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
neg.f16x2 r2993, r2945;
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
add.f16x2 r3035, r2895, r2948;
}
{
add.f16x2 r3038, r2898, r2993;
}
{
sub.f16x2 r3041, r2895, r2948;
}
{
sub.f16x2 r3044, r2898, r2993;
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
neg.f16x2 r3083, r3077;
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
add.f16x2 r3097, r3065, r3080;
}
{
add.f16x2 r3100, r3068, r3083;
}
{
sub.f16x2 r3103, r3065, r3080;
}
{
sub.f16x2 r3106, r3068, r3083;
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
neg.f16x2 r3133, r3127;
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
add.f16x2 r3147, r3115, r3130;
}
{
add.f16x2 r3150, r3118, r3133;
}
{
sub.f16x2 r3153, r3115, r3130;
}
{
sub.f16x2 r3156, r3118, r3133;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r3159, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3160, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3163, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
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
neg.f16x2 r3189, r3141;
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
add.f16x2 r3231, r3091, r3144;
}
{
add.f16x2 r3234, r3094, r3189;
}
{
sub.f16x2 r3237, r3091, r3144;
}
{
sub.f16x2 r3240, r3094, r3189;
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
cvt.rn.f16.f32 low, f660;
cvt.rn.f16.f32 high, f660;
mov.b32 r3255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r3256, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r3257, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3258, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f676;
cvt.rn.f16.f32 high, f676;
mov.b32 r3259, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r3260, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r3263, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r3264, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3265, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3266, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r3267, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
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
neg.f16x2 r3333, r3213;
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
add.f16x2 r3431, r3017, r3216;
}
{
add.f16x2 r3434, r3020, r3333;
}
{
sub.f16x2 r3437, r3017, r3216;
}
{
sub.f16x2 r3440, r3020, r3333;
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
neg.f16x2 r3503, r3497;
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
add.f16x2 r3517, r3485, r3500;
}
{
add.f16x2 r3520, r3488, r3503;
}
{
sub.f16x2 r3523, r3485, r3500;
}
{
sub.f16x2 r3526, r3488, r3503;
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
neg.f16x2 r3553, r3547;
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
add.f16x2 r3567, r3535, r3550;
}
{
add.f16x2 r3570, r3538, r3553;
}
{
sub.f16x2 r3573, r3535, r3550;
}
{
sub.f16x2 r3576, r3538, r3553;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r3579, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3580, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3583, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
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
neg.f16x2 r3609, r3561;
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
add.f16x2 r3651, r3511, r3564;
}
{
add.f16x2 r3654, r3514, r3609;
}
{
sub.f16x2 r3657, r3511, r3564;
}
{
sub.f16x2 r3660, r3514, r3609;
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
neg.f16x2 r3699, r3693;
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
add.f16x2 r3713, r3681, r3696;
}
{
add.f16x2 r3716, r3684, r3699;
}
{
sub.f16x2 r3719, r3681, r3696;
}
{
sub.f16x2 r3722, r3684, r3699;
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
neg.f16x2 r3749, r3743;
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
add.f16x2 r3763, r3731, r3746;
}
{
add.f16x2 r3766, r3734, r3749;
}
{
sub.f16x2 r3769, r3731, r3746;
}
{
sub.f16x2 r3772, r3734, r3749;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r3775, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3776, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3779, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
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
neg.f16x2 r3805, r3757;
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
add.f16x2 r3847, r3707, r3760;
}
{
add.f16x2 r3850, r3710, r3805;
}
{
sub.f16x2 r3853, r3707, r3760;
}
{
sub.f16x2 r3856, r3710, r3805;
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
cvt.rn.f16.f32 low, f660;
cvt.rn.f16.f32 high, f660;
mov.b32 r3871, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r3872, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r3873, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3874, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f676;
cvt.rn.f16.f32 high, f676;
mov.b32 r3875, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r3876, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r3879, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r3880, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3881, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3882, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r3883, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
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
neg.f16x2 r3949, r3829;
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
add.f16x2 r4047, r3633, r3832;
}
{
add.f16x2 r4050, r3636, r3949;
}
{
sub.f16x2 r4053, r3633, r3832;
}
{
sub.f16x2 r4056, r3636, r3949;
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
cvt.rn.f16.f32 low, f656;
cvt.rn.f16.f32 high, f656;
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
cvt.rn.f16.f32 low, f660;
cvt.rn.f16.f32 high, f660;
mov.b32 r4097, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r4098, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f664;
cvt.rn.f16.f32 high, f664;
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
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r4101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r4102, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f672;
cvt.rn.f16.f32 high, f672;
mov.b32 r4103, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f704;
cvt.rn.f16.f32 high, f704;
mov.b32 r4104, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f676;
cvt.rn.f16.f32 high, f676;
mov.b32 r4105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r4106, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f680;
cvt.rn.f16.f32 high, f680;
mov.b32 r4107, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f712;
cvt.rn.f16.f32 high, f712;
mov.b32 r4108, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f714;
cvt.rn.f16.f32 high, f714;
mov.b32 r4111, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f712;
cvt.rn.f16.f32 high, f712;
mov.b32 r4112, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r4113, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r4114, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f706;
cvt.rn.f16.f32 high, f706;
mov.b32 r4115, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f704;
cvt.rn.f16.f32 high, f704;
mov.b32 r4116, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r4117, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
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
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r4121, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
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
neg.f16x2 r4269, r4005;
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
add.f16x2 r4479, r3389, r4008;
}
{
add.f16x2 r4482, r3392, r4269;
}
{
sub.f16x2 r4485, r3389, r4008;
}
{
sub.f16x2 r4488, r3392, r4269;
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
bfe.u32 r6138, r6129, 5, 2;
cvt.rn.f32.u32 f848, r6138;
mul.f32 f849, f848, 0f3D490FDB;
cos.approx.f32 f779, f849;
sin.approx.f32 f850, f849;
neg.f32 f780, f850;
and.b32 r6145, r6129, 96;
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
neg.f16x2 r4585, r4582;
}
{
fma.rn.f16x2 r4587, r4395, r4578, r4585;
}
{
mul.f16x2 r4591, r4395, r4580;
}
{
fma.rn.f16x2 r4594, r4398, r4578, r4591;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4622, r4619;
}
{
fma.rn.f16x2 r4624, r4407, r4615, r4622;
}
{
mul.f16x2 r4628, r4407, r4617;
}
{
fma.rn.f16x2 r4631, r4410, r4615, r4628;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4659, r4656;
}
{
fma.rn.f16x2 r4661, r4419, r4652, r4659;
}
{
mul.f16x2 r4665, r4419, r4654;
}
{
fma.rn.f16x2 r4668, r4422, r4652, r4665;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4696, r4693;
}
{
fma.rn.f16x2 r4698, r4431, r4689, r4696;
}
{
mul.f16x2 r4702, r4431, r4691;
}
{
fma.rn.f16x2 r4705, r4434, r4689, r4702;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4733, r4730;
}
{
fma.rn.f16x2 r4735, r4443, r4726, r4733;
}
{
mul.f16x2 r4739, r4443, r4728;
}
{
fma.rn.f16x2 r4742, r4446, r4726, r4739;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4770, r4767;
}
{
fma.rn.f16x2 r4772, r4455, r4763, r4770;
}
{
mul.f16x2 r4776, r4455, r4765;
}
{
fma.rn.f16x2 r4779, r4458, r4763, r4776;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4807, r4804;
}
{
fma.rn.f16x2 r4809, r4467, r4800, r4807;
}
{
mul.f16x2 r4813, r4467, r4802;
}
{
fma.rn.f16x2 r4816, r4470, r4800, r4813;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4844, r4841;
}
{
fma.rn.f16x2 r4846, r4479, r4837, r4844;
}
{
mul.f16x2 r4850, r4479, r4839;
}
{
fma.rn.f16x2 r4853, r4482, r4837, r4850;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4881, r4878;
}
{
fma.rn.f16x2 r4883, r4491, r4874, r4881;
}
{
mul.f16x2 r4887, r4491, r4876;
}
{
fma.rn.f16x2 r4890, r4494, r4874, r4887;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4918, r4915;
}
{
fma.rn.f16x2 r4920, r4503, r4911, r4918;
}
{
mul.f16x2 r4924, r4503, r4913;
}
{
fma.rn.f16x2 r4927, r4506, r4911, r4924;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4955, r4952;
}
{
fma.rn.f16x2 r4957, r4515, r4948, r4955;
}
{
mul.f16x2 r4961, r4515, r4950;
}
{
fma.rn.f16x2 r4964, r4518, r4948, r4961;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4992, r4989;
}
{
fma.rn.f16x2 r4994, r4527, r4985, r4992;
}
{
mul.f16x2 r4998, r4527, r4987;
}
{
fma.rn.f16x2 r5001, r4530, r4985, r4998;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5029, r5026;
}
{
fma.rn.f16x2 r5031, r4539, r5022, r5029;
}
{
mul.f16x2 r5035, r4539, r5024;
}
{
fma.rn.f16x2 r5038, r4542, r5022, r5035;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5066, r5063;
}
{
fma.rn.f16x2 r5068, r4551, r5059, r5066;
}
{
mul.f16x2 r5072, r4551, r5061;
}
{
fma.rn.f16x2 r5075, r4554, r5059, r5072;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5103, r5100;
}
{
fma.rn.f16x2 r5105, r4563, r5096, r5103;
}
{
mul.f16x2 r5109, r4563, r5098;
}
{
fma.rn.f16x2 r5112, r4566, r5096, r5109;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5140, r5137;
}
{
fma.rn.f16x2 r5142, r4389, r5133, r5140;
}
{
mul.f16x2 r5146, r4389, r5135;
}
{
fma.rn.f16x2 r5149, r4392, r5133, r5146;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5177, r5174;
}
{
fma.rn.f16x2 r5179, r4401, r5170, r5177;
}
{
mul.f16x2 r5183, r4401, r5172;
}
{
fma.rn.f16x2 r5186, r4404, r5170, r5183;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5214, r5211;
}
{
fma.rn.f16x2 r5216, r4413, r5207, r5214;
}
{
mul.f16x2 r5220, r4413, r5209;
}
{
fma.rn.f16x2 r5223, r4416, r5207, r5220;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5251, r5248;
}
{
fma.rn.f16x2 r5253, r4425, r5244, r5251;
}
{
mul.f16x2 r5257, r4425, r5246;
}
{
fma.rn.f16x2 r5260, r4428, r5244, r5257;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5288, r5285;
}
{
fma.rn.f16x2 r5290, r4437, r5281, r5288;
}
{
mul.f16x2 r5294, r4437, r5283;
}
{
fma.rn.f16x2 r5297, r4440, r5281, r5294;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5325, r5322;
}
{
fma.rn.f16x2 r5327, r4449, r5318, r5325;
}
{
mul.f16x2 r5331, r4449, r5320;
}
{
fma.rn.f16x2 r5334, r4452, r5318, r5331;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5362, r5359;
}
{
fma.rn.f16x2 r5364, r4461, r5355, r5362;
}
{
mul.f16x2 r5368, r4461, r5357;
}
{
fma.rn.f16x2 r5371, r4464, r5355, r5368;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5399, r5396;
}
{
fma.rn.f16x2 r5401, r4473, r5392, r5399;
}
{
mul.f16x2 r5405, r4473, r5394;
}
{
fma.rn.f16x2 r5408, r4476, r5392, r5405;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5436, r5433;
}
{
fma.rn.f16x2 r5438, r4485, r5429, r5436;
}
{
mul.f16x2 r5442, r4485, r5431;
}
{
fma.rn.f16x2 r5445, r4488, r5429, r5442;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5473, r5470;
}
{
fma.rn.f16x2 r5475, r4497, r5466, r5473;
}
{
mul.f16x2 r5479, r4497, r5468;
}
{
fma.rn.f16x2 r5482, r4500, r5466, r5479;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5510, r5507;
}
{
fma.rn.f16x2 r5512, r4509, r5503, r5510;
}
{
mul.f16x2 r5516, r4509, r5505;
}
{
fma.rn.f16x2 r5519, r4512, r5503, r5516;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5547, r5544;
}
{
fma.rn.f16x2 r5549, r4521, r5540, r5547;
}
{
mul.f16x2 r5553, r4521, r5542;
}
{
fma.rn.f16x2 r5556, r4524, r5540, r5553;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5584, r5581;
}
{
fma.rn.f16x2 r5586, r4533, r5577, r5584;
}
{
mul.f16x2 r5590, r4533, r5579;
}
{
fma.rn.f16x2 r5593, r4536, r5577, r5590;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5621, r5618;
}
{
fma.rn.f16x2 r5623, r4545, r5614, r5621;
}
{
mul.f16x2 r5627, r4545, r5616;
}
{
fma.rn.f16x2 r5630, r4548, r5614, r5627;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5658, r5655;
}
{
fma.rn.f16x2 r5660, r4557, r5651, r5658;
}
{
mul.f16x2 r5664, r4557, r5653;
}
{
fma.rn.f16x2 r5667, r4560, r5651, r5664;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5695, r5692;
}
{
fma.rn.f16x2 r5697, r4569, r5688, r5695;
}
{
mul.f16x2 r5701, r4569, r5690;
}
{
fma.rn.f16x2 r5704, r4572, r5688, r5701;
}
shl.b32 r6139, r6129, 3;
and.b32 r6140, r6139, 248;
add.s32 r6141, r6133, r6140;
barrier.sync 0;
and.b32 r6142, r6131, 24576;
add.s32 r6143, r6141, r6142;
st.shared.u32 [r6143], r4383;
st.shared.u32 [r6143+4], r4386;
st.shared.u32 [r6143+256], r4587;
st.shared.u32 [r6143+260], r4594;
st.shared.u32 [r6143+512], r4624;
st.shared.u32 [r6143+516], r4631;
st.shared.u32 [r6143+768], r4661;
st.shared.u32 [r6143+772], r4668;
st.shared.u32 [r6143+1024], r4698;
st.shared.u32 [r6143+1028], r4705;
st.shared.u32 [r6143+1280], r4735;
st.shared.u32 [r6143+1284], r4742;
st.shared.u32 [r6143+1536], r4772;
st.shared.u32 [r6143+1540], r4779;
st.shared.u32 [r6143+1792], r4809;
st.shared.u32 [r6143+1796], r4816;
st.shared.u32 [r6143+2048], r4846;
st.shared.u32 [r6143+2052], r4853;
st.shared.u32 [r6143+2304], r4883;
st.shared.u32 [r6143+2308], r4890;
st.shared.u32 [r6143+2560], r4920;
st.shared.u32 [r6143+2564], r4927;
st.shared.u32 [r6143+2816], r4957;
st.shared.u32 [r6143+2820], r4964;
st.shared.u32 [r6143+3072], r4994;
st.shared.u32 [r6143+3076], r5001;
st.shared.u32 [r6143+3328], r5031;
st.shared.u32 [r6143+3332], r5038;
st.shared.u32 [r6143+3584], r5068;
st.shared.u32 [r6143+3588], r5075;
st.shared.u32 [r6143+3840], r5105;
st.shared.u32 [r6143+3844], r5112;
st.shared.u32 [r6143+4096], r5142;
st.shared.u32 [r6143+4100], r5149;
st.shared.u32 [r6143+4352], r5179;
st.shared.u32 [r6143+4356], r5186;
st.shared.u32 [r6143+4608], r5216;
st.shared.u32 [r6143+4612], r5223;
st.shared.u32 [r6143+4864], r5253;
st.shared.u32 [r6143+4868], r5260;
st.shared.u32 [r6143+5120], r5290;
st.shared.u32 [r6143+5124], r5297;
st.shared.u32 [r6143+5376], r5327;
st.shared.u32 [r6143+5380], r5334;
st.shared.u32 [r6143+5632], r5364;
st.shared.u32 [r6143+5636], r5371;
st.shared.u32 [r6143+5888], r5401;
st.shared.u32 [r6143+5892], r5408;
st.shared.u32 [r6143+6144], r5438;
st.shared.u32 [r6143+6148], r5445;
st.shared.u32 [r6143+6400], r5475;
st.shared.u32 [r6143+6404], r5482;
st.shared.u32 [r6143+6656], r5512;
st.shared.u32 [r6143+6660], r5519;
st.shared.u32 [r6143+6912], r5549;
st.shared.u32 [r6143+6916], r5556;
st.shared.u32 [r6143+7168], r5586;
st.shared.u32 [r6143+7172], r5593;
st.shared.u32 [r6143+7424], r5623;
st.shared.u32 [r6143+7428], r5630;
st.shared.u32 [r6143+7680], r5660;
st.shared.u32 [r6143+7684], r5667;
st.shared.u32 [r6143+7936], r5697;
st.shared.u32 [r6143+7940], r5704;
barrier.sync 0;
mad.lo.s32 r6144, r6145, -248, r6143;
ld.shared.u32 r5726, [r6144];
ld.shared.u32 r5729, [r6144+4];
ld.shared.u32 r5776, [r6144+1024];
ld.shared.u32 r5779, [r6144+1028];
ld.shared.u32 r5826, [r6144+2048];
ld.shared.u32 r5829, [r6144+2052];
ld.shared.u32 r5876, [r6144+3072];
ld.shared.u32 r5879, [r6144+3076];
ld.shared.u32 r5926, [r6144+4096];
ld.shared.u32 r5929, [r6144+4100];
ld.shared.u32 r5976, [r6144+5120];
ld.shared.u32 r5979, [r6144+5124];
ld.shared.u32 r6026, [r6144+6144];
ld.shared.u32 r6029, [r6144+6148];
ld.shared.u32 r6076, [r6144+7168];
ld.shared.u32 r6079, [r6144+7172];
ld.shared.u32 r5738, [r6144+8192];
ld.shared.u32 r5741, [r6144+8196];
ld.shared.u32 r5788, [r6144+9216];
ld.shared.u32 r5791, [r6144+9220];
ld.shared.u32 r5838, [r6144+10240];
ld.shared.u32 r5841, [r6144+10244];
ld.shared.u32 r5888, [r6144+11264];
ld.shared.u32 r5891, [r6144+11268];
ld.shared.u32 r5938, [r6144+12288];
ld.shared.u32 r5941, [r6144+12292];
ld.shared.u32 r5988, [r6144+13312];
ld.shared.u32 r5991, [r6144+13316];
ld.shared.u32 r6038, [r6144+14336];
ld.shared.u32 r6041, [r6144+14340];
ld.shared.u32 r6088, [r6144+15360];
ld.shared.u32 r6091, [r6144+15364];
ld.shared.u32 r5727, [r6144+16384];
ld.shared.u32 r5730, [r6144+16388];
ld.shared.u32 r5777, [r6144+17408];
ld.shared.u32 r5780, [r6144+17412];
ld.shared.u32 r5827, [r6144+18432];
ld.shared.u32 r5830, [r6144+18436];
ld.shared.u32 r5877, [r6144+19456];
ld.shared.u32 r5880, [r6144+19460];
ld.shared.u32 r5927, [r6144+20480];
ld.shared.u32 r5930, [r6144+20484];
ld.shared.u32 r5977, [r6144+21504];
ld.shared.u32 r5980, [r6144+21508];
ld.shared.u32 r6027, [r6144+22528];
ld.shared.u32 r6030, [r6144+22532];
ld.shared.u32 r6077, [r6144+23552];
ld.shared.u32 r6080, [r6144+23556];
ld.shared.u32 r5739, [r6144+24576];
ld.shared.u32 r5742, [r6144+24580];
ld.shared.u32 r5789, [r6144+25600];
ld.shared.u32 r5792, [r6144+25604];
ld.shared.u32 r5839, [r6144+26624];
ld.shared.u32 r5842, [r6144+26628];
ld.shared.u32 r5889, [r6144+27648];
ld.shared.u32 r5892, [r6144+27652];
ld.shared.u32 r5939, [r6144+28672];
ld.shared.u32 r5942, [r6144+28676];
ld.shared.u32 r5989, [r6144+29696];
ld.shared.u32 r5992, [r6144+29700];
ld.shared.u32 r6039, [r6144+30720];
ld.shared.u32 r6042, [r6144+30724];
ld.shared.u32 r6089, [r6144+31744];
ld.shared.u32 r6092, [r6144+31748];
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
neg.f16x2 r5749, r5743;
}
{
add.f16x2 %0, r5725, r5737;
}
{
add.f16x2 %1, r5728, r5740;
}
{
sub.f16x2 %32, r5725, r5737;
}
{
sub.f16x2 %33, r5728, r5740;
}
{
add.f16x2 %16, r5731, r5746;
}
{
add.f16x2 %17, r5734, r5749;
}
{
sub.f16x2 %48, r5731, r5746;
}
{
sub.f16x2 %49, r5734, r5749;
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
neg.f16x2 r5799, r5793;
}
{
add.f16x2 %2, r5775, r5787;
}
{
add.f16x2 %3, r5778, r5790;
}
{
sub.f16x2 %34, r5775, r5787;
}
{
sub.f16x2 %35, r5778, r5790;
}
{
add.f16x2 %18, r5781, r5796;
}
{
add.f16x2 %19, r5784, r5799;
}
{
sub.f16x2 %50, r5781, r5796;
}
{
sub.f16x2 %51, r5784, r5799;
}
{
add.f16x2 r5825, r5826, r5827;
}
{
add.f16x2 r5828, r5829, r5830;
}
{
sub.f16x2 r5831, r5826, r5827;
}
{
sub.f16x2 r5834, r5829, r5830;
}
{
add.f16x2 r5837, r5838, r5839;
}
{
add.f16x2 r5840, r5841, r5842;
}
{
sub.f16x2 r5843, r5838, r5839;
}
{
sub.f16x2 r5846, r5841, r5842;
}
{
neg.f16x2 r5849, r5843;
}
{
add.f16x2 %4, r5825, r5837;
}
{
add.f16x2 %5, r5828, r5840;
}
{
sub.f16x2 %36, r5825, r5837;
}
{
sub.f16x2 %37, r5828, r5840;
}
{
add.f16x2 %20, r5831, r5846;
}
{
add.f16x2 %21, r5834, r5849;
}
{
sub.f16x2 %52, r5831, r5846;
}
{
sub.f16x2 %53, r5834, r5849;
}
{
add.f16x2 r5875, r5876, r5877;
}
{
add.f16x2 r5878, r5879, r5880;
}
{
sub.f16x2 r5881, r5876, r5877;
}
{
sub.f16x2 r5884, r5879, r5880;
}
{
add.f16x2 r5887, r5888, r5889;
}
{
add.f16x2 r5890, r5891, r5892;
}
{
sub.f16x2 r5893, r5888, r5889;
}
{
sub.f16x2 r5896, r5891, r5892;
}
{
neg.f16x2 r5899, r5893;
}
{
add.f16x2 %6, r5875, r5887;
}
{
add.f16x2 %7, r5878, r5890;
}
{
sub.f16x2 %38, r5875, r5887;
}
{
sub.f16x2 %39, r5878, r5890;
}
{
add.f16x2 %22, r5881, r5896;
}
{
add.f16x2 %23, r5884, r5899;
}
{
sub.f16x2 %54, r5881, r5896;
}
{
sub.f16x2 %55, r5884, r5899;
}
{
add.f16x2 r5925, r5926, r5927;
}
{
add.f16x2 r5928, r5929, r5930;
}
{
sub.f16x2 r5931, r5926, r5927;
}
{
sub.f16x2 r5934, r5929, r5930;
}
{
add.f16x2 r5937, r5938, r5939;
}
{
add.f16x2 r5940, r5941, r5942;
}
{
sub.f16x2 r5943, r5938, r5939;
}
{
sub.f16x2 r5946, r5941, r5942;
}
{
neg.f16x2 r5949, r5943;
}
{
add.f16x2 %8, r5925, r5937;
}
{
add.f16x2 %9, r5928, r5940;
}
{
sub.f16x2 %40, r5925, r5937;
}
{
sub.f16x2 %41, r5928, r5940;
}
{
add.f16x2 %24, r5931, r5946;
}
{
add.f16x2 %25, r5934, r5949;
}
{
sub.f16x2 %56, r5931, r5946;
}
{
sub.f16x2 %57, r5934, r5949;
}
{
add.f16x2 r5975, r5976, r5977;
}
{
add.f16x2 r5978, r5979, r5980;
}
{
sub.f16x2 r5981, r5976, r5977;
}
{
sub.f16x2 r5984, r5979, r5980;
}
{
add.f16x2 r5987, r5988, r5989;
}
{
add.f16x2 r5990, r5991, r5992;
}
{
sub.f16x2 r5993, r5988, r5989;
}
{
sub.f16x2 r5996, r5991, r5992;
}
{
neg.f16x2 r5999, r5993;
}
{
add.f16x2 %10, r5975, r5987;
}
{
add.f16x2 %11, r5978, r5990;
}
{
sub.f16x2 %42, r5975, r5987;
}
{
sub.f16x2 %43, r5978, r5990;
}
{
add.f16x2 %26, r5981, r5996;
}
{
add.f16x2 %27, r5984, r5999;
}
{
sub.f16x2 %58, r5981, r5996;
}
{
sub.f16x2 %59, r5984, r5999;
}
{
add.f16x2 r6025, r6026, r6027;
}
{
add.f16x2 r6028, r6029, r6030;
}
{
sub.f16x2 r6031, r6026, r6027;
}
{
sub.f16x2 r6034, r6029, r6030;
}
{
add.f16x2 r6037, r6038, r6039;
}
{
add.f16x2 r6040, r6041, r6042;
}
{
sub.f16x2 r6043, r6038, r6039;
}
{
sub.f16x2 r6046, r6041, r6042;
}
{
neg.f16x2 r6049, r6043;
}
{
add.f16x2 %12, r6025, r6037;
}
{
add.f16x2 %13, r6028, r6040;
}
{
sub.f16x2 %44, r6025, r6037;
}
{
sub.f16x2 %45, r6028, r6040;
}
{
add.f16x2 %28, r6031, r6046;
}
{
add.f16x2 %29, r6034, r6049;
}
{
sub.f16x2 %60, r6031, r6046;
}
{
sub.f16x2 %61, r6034, r6049;
}
{
add.f16x2 r6075, r6076, r6077;
}
{
add.f16x2 r6078, r6079, r6080;
}
{
sub.f16x2 r6081, r6076, r6077;
}
{
sub.f16x2 r6084, r6079, r6080;
}
{
add.f16x2 r6087, r6088, r6089;
}
{
add.f16x2 r6090, r6091, r6092;
}
{
sub.f16x2 r6093, r6088, r6089;
}
{
sub.f16x2 r6096, r6091, r6092;
}
{
neg.f16x2 r6099, r6093;
}
{
add.f16x2 %14, r6075, r6087;
}
{
add.f16x2 %15, r6078, r6090;
}
{
sub.f16x2 %46, r6075, r6087;
}
{
sub.f16x2 %47, r6078, r6090;
}
{
add.f16x2 %30, r6081, r6096;
}
{
add.f16x2 %31, r6084, r6099;
}
{
sub.f16x2 %62, r6081, r6096;
}
{
sub.f16x2 %63, r6084, r6099;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)), "=r"(__HALF2_TO_UI(rmem[27].x)), "=r"(__HALF2_TO_UI(rmem[27].y)), "=r"(__HALF2_TO_UI(rmem[28].x)), "=r"(__HALF2_TO_UI(rmem[28].y)), "=r"(__HALF2_TO_UI(rmem[29].x)), "=r"(__HALF2_TO_UI(rmem[29].y)), "=r"(__HALF2_TO_UI(rmem[30].x)), "=r"(__HALF2_TO_UI(rmem[30].y)), "=r"(__HALF2_TO_UI(rmem[31].x)), "=r"(__HALF2_TO_UI(rmem[31].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[31].y)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[28].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[27].y)), "r"(__HALF2_TO_UI(rmem[30].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[29].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[31].x)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[28].x)), "r"(__HALF2_TO_UI(rmem[27].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[30].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[29].x)), "r"(__HALF2_TO_UI(rmem[24].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<848, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<176>;
.reg .b32 r<1598>;
.reg .b64 rd<2>;
mov.u32 r1571, %tid.y;
shl.b32 r1572, r1571, 15;
mov.u32 r1573, %16;
add.s32 r1574, r1573, r1572;
mov.u32 r1575, %tid.x;
{
add.f16x2 r1, %17, %25;
}
{
add.f16x2 r4, %18, %26;
}
{
sub.f16x2 r7, %17, %25;
}
{
sub.f16x2 r10, %18, %26;
}
{
add.f16x2 r13, %21, %29;
}
{
add.f16x2 r16, %22, %30;
}
{
sub.f16x2 r19, %21, %29;
}
{
sub.f16x2 r22, %22, %30;
}
{
neg.f16x2 r25, r19;
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
add.f16x2 r39, r7, r22;
}
{
add.f16x2 r42, r10, r25;
}
{
sub.f16x2 r45, r7, r22;
}
{
sub.f16x2 r48, r10, r25;
}
{
add.f16x2 r51, %19, %27;
}
{
add.f16x2 r54, %20, %28;
}
{
sub.f16x2 r57, %19, %27;
}
{
sub.f16x2 r60, %20, %28;
}
{
add.f16x2 r63, %23, %31;
}
{
add.f16x2 r66, %24, %32;
}
{
sub.f16x2 r69, %23, %31;
}
{
sub.f16x2 r72, %24, %32;
}
{
neg.f16x2 r75, r69;
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
add.f16x2 r89, r57, r72;
}
{
add.f16x2 r92, r60, r75;
}
{
sub.f16x2 r95, r57, r72;
}
{
sub.f16x2 r98, r60, r75;
}
mov.f32 f140, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r101, {low, high};
}
mov.f32 f150, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r102, {low, high};
}
mov.f32 f135, 0fBF800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r106, {low, high};
}
mov.f32 f136, 0f3F800000;
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
neg.f16x2 r131, r83;
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
add.f16x2 r173, r33, r86;
}
{
add.f16x2 r176, r36, r131;
}
{
sub.f16x2 r179, r33, r86;
}
{
sub.f16x2 r182, r36, r131;
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
and.b32 r1576, r1575, 511;
shl.b32 r1577, r1575, 6;
and.b32 r1578, r1577, -32768;
add.s32 r1579, r1574, r1578;
cvt.rn.f32.u32 f167, r1576;
mul.f32 f168, f167, 0f3AC90FDB;
cos.approx.f32 f29, f168;
sin.approx.f32 f169, f168;
neg.f32 f30, f169;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f29;
cvt.rn.f16.f32 high, f30;
mov.b32 r197, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r200, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r202, {high, high};
}
{
mul.f16x2 r204, r164, r202;
}
{
neg.f16x2 r207, r204;
}
{
fma.rn.f16x2 r209, r161, r200, r207;
}
{
mul.f16x2 r213, r161, r202;
}
{
fma.rn.f16x2 r216, r164, r200, r213;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r220, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r222, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r224, {low, high};
}
{
mul.f16x2 r225, r222, r224;
}
{
mul.f16x2 r228, r197, r220;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r231, {high, low};
}
{
fma.rn.f16x2 r233, r225, r231, r228;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r233;
mov.b32 r237, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r233;
mov.b32 r239, {high, high};
}
{
mul.f16x2 r241, r176, r239;
}
{
neg.f16x2 r244, r241;
}
{
fma.rn.f16x2 r246, r173, r237, r244;
}
{
mul.f16x2 r250, r173, r239;
}
{
fma.rn.f16x2 r253, r176, r237, r250;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r257, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r259, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r261, {low, high};
}
{
mul.f16x2 r262, r259, r261;
}
{
mul.f16x2 r265, r233, r257;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r233;
mov.b32 r268, {high, low};
}
{
fma.rn.f16x2 r270, r262, r268, r265;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r270;
mov.b32 r274, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r270;
mov.b32 r276, {high, high};
}
{
mul.f16x2 r278, r188, r276;
}
{
neg.f16x2 r281, r278;
}
{
fma.rn.f16x2 r283, r185, r274, r281;
}
{
mul.f16x2 r287, r185, r276;
}
{
fma.rn.f16x2 r290, r188, r274, r287;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r294, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r296, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r298, {low, high};
}
{
mul.f16x2 r299, r296, r298;
}
{
mul.f16x2 r302, r270, r294;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r270;
mov.b32 r305, {high, low};
}
{
fma.rn.f16x2 r307, r299, r305, r302;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r307;
mov.b32 r311, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r307;
mov.b32 r313, {high, high};
}
{
mul.f16x2 r315, r158, r313;
}
{
neg.f16x2 r318, r315;
}
{
fma.rn.f16x2 r320, r155, r311, r318;
}
{
mul.f16x2 r324, r155, r313;
}
{
fma.rn.f16x2 r327, r158, r311, r324;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r331, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r333, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r335, {low, high};
}
{
mul.f16x2 r336, r333, r335;
}
{
mul.f16x2 r339, r307, r331;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r307;
mov.b32 r342, {high, low};
}
{
fma.rn.f16x2 r344, r336, r342, r339;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r344;
mov.b32 r348, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r344;
mov.b32 r350, {high, high};
}
{
mul.f16x2 r352, r170, r350;
}
{
neg.f16x2 r355, r352;
}
{
fma.rn.f16x2 r357, r167, r348, r355;
}
{
mul.f16x2 r361, r167, r350;
}
{
fma.rn.f16x2 r364, r170, r348, r361;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r368, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r370, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r372, {low, high};
}
{
mul.f16x2 r373, r370, r372;
}
{
mul.f16x2 r376, r344, r368;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r344;
mov.b32 r379, {high, low};
}
{
fma.rn.f16x2 r381, r373, r379, r376;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r381;
mov.b32 r385, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r381;
mov.b32 r387, {high, high};
}
{
mul.f16x2 r389, r182, r387;
}
{
neg.f16x2 r392, r389;
}
{
fma.rn.f16x2 r394, r179, r385, r392;
}
{
mul.f16x2 r398, r179, r387;
}
{
fma.rn.f16x2 r401, r182, r385, r398;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r405, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r407, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r409, {low, high};
}
{
mul.f16x2 r410, r407, r409;
}
{
mul.f16x2 r413, r381, r405;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r381;
mov.b32 r416, {high, low};
}
{
fma.rn.f16x2 r418, r410, r416, r413;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r418;
mov.b32 r422, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r418;
mov.b32 r424, {high, high};
}
{
mul.f16x2 r426, r194, r424;
}
{
neg.f16x2 r429, r426;
}
{
fma.rn.f16x2 r431, r191, r422, r429;
}
{
mul.f16x2 r435, r191, r424;
}
{
fma.rn.f16x2 r438, r194, r422, r435;
}
barrier.sync 0;
and.b32 r1580, r1577, 32704;
add.s32 r1581, r1579, r1580;
st.shared.v4.f32 [r1581], {r149, r152, r209, r216};
st.shared.v4.f32 [r1581+16], {r246, r253, r283, r290};
st.shared.v4.f32 [r1581+32], {r320, r327, r357, r364};
st.shared.v4.f32 [r1581+48], {r394, r401, r431, r438};
barrier.sync 0;
mad.lo.s32 r1582, r1576, -56, r1581;
ld.shared.u32 r460, [r1582];
ld.shared.u32 r463, [r1582+4];
ld.shared.u32 r510, [r1582+4096];
ld.shared.u32 r513, [r1582+4100];
ld.shared.u32 r472, [r1582+8192];
ld.shared.u32 r475, [r1582+8196];
ld.shared.u32 r522, [r1582+12288];
ld.shared.u32 r525, [r1582+12292];
ld.shared.u32 r461, [r1582+16384];
ld.shared.u32 r464, [r1582+16388];
ld.shared.u32 r511, [r1582+20480];
ld.shared.u32 r514, [r1582+20484];
ld.shared.u32 r473, [r1582+24576];
ld.shared.u32 r476, [r1582+24580];
ld.shared.u32 r523, [r1582+28672];
ld.shared.u32 r526, [r1582+28676];
{
add.f16x2 r459, r460, r461;
}
{
add.f16x2 r462, r463, r464;
}
{
sub.f16x2 r465, r460, r461;
}
{
sub.f16x2 r468, r463, r464;
}
{
add.f16x2 r471, r472, r473;
}
{
add.f16x2 r474, r475, r476;
}
{
sub.f16x2 r477, r472, r473;
}
{
sub.f16x2 r480, r475, r476;
}
{
neg.f16x2 r483, r477;
}
{
add.f16x2 r485, r459, r471;
}
{
add.f16x2 r488, r462, r474;
}
{
sub.f16x2 r491, r459, r471;
}
{
sub.f16x2 r494, r462, r474;
}
{
add.f16x2 r497, r465, r480;
}
{
add.f16x2 r500, r468, r483;
}
{
sub.f16x2 r503, r465, r480;
}
{
sub.f16x2 r506, r468, r483;
}
{
add.f16x2 r509, r510, r511;
}
{
add.f16x2 r512, r513, r514;
}
{
sub.f16x2 r515, r510, r511;
}
{
sub.f16x2 r518, r513, r514;
}
{
add.f16x2 r521, r522, r523;
}
{
add.f16x2 r524, r525, r526;
}
{
sub.f16x2 r527, r522, r523;
}
{
sub.f16x2 r530, r525, r526;
}
{
neg.f16x2 r533, r527;
}
{
add.f16x2 r535, r509, r521;
}
{
add.f16x2 r538, r512, r524;
}
{
sub.f16x2 r541, r509, r521;
}
{
sub.f16x2 r544, r512, r524;
}
{
add.f16x2 r547, r515, r530;
}
{
add.f16x2 r550, r518, r533;
}
{
sub.f16x2 r553, r515, r530;
}
{
sub.f16x2 r556, r518, r533;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r559, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r560, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r563, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r564, {low, high};
}
{
mul.f16x2 r573, r547, r559;
}
{
mul.f16x2 r576, r550, r560;
}
{
sub.f16x2 r579, r573, r576;
}
{
mul.f16x2 r582, r547, r560;
}
{
fma.rn.f16x2 r585, r550, r559, r582;
}
{
neg.f16x2 r589, r541;
}
{
mul.f16x2 r591, r553, r563;
}
{
mul.f16x2 r594, r556, r564;
}
{
sub.f16x2 r597, r591, r594;
}
{
mul.f16x2 r600, r553, r564;
}
{
fma.rn.f16x2 r603, r556, r563, r600;
}
{
add.f16x2 r607, r485, r535;
}
{
add.f16x2 r610, r488, r538;
}
{
sub.f16x2 r613, r485, r535;
}
{
sub.f16x2 r616, r488, r538;
}
{
add.f16x2 r619, r497, r579;
}
{
add.f16x2 r622, r500, r585;
}
{
sub.f16x2 r625, r497, r579;
}
{
sub.f16x2 r628, r500, r585;
}
{
add.f16x2 r631, r491, r544;
}
{
add.f16x2 r634, r494, r589;
}
{
sub.f16x2 r637, r491, r544;
}
{
sub.f16x2 r640, r494, r589;
}
{
add.f16x2 r643, r503, r597;
}
{
add.f16x2 r646, r506, r603;
}
{
sub.f16x2 r649, r503, r597;
}
{
sub.f16x2 r652, r506, r603;
}
and.b32 r1583, r1575, 504;
bfe.u32 r1584, r1575, 3, 6;
cvt.rn.f32.u32 f170, r1584;
mul.f32 f171, f170, 0f3C490FDB;
cos.approx.f32 f75, f171;
sin.approx.f32 f172, f171;
neg.f32 f76, f172;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f75;
cvt.rn.f16.f32 high, f76;
mov.b32 r655, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r658, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r660, {high, high};
}
{
mul.f16x2 r662, r622, r660;
}
{
neg.f16x2 r665, r662;
}
{
fma.rn.f16x2 r667, r619, r658, r665;
}
{
mul.f16x2 r671, r619, r660;
}
{
fma.rn.f16x2 r674, r622, r658, r671;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r678, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r680, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r682, {low, high};
}
{
mul.f16x2 r683, r680, r682;
}
{
mul.f16x2 r686, r655, r678;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r689, {high, low};
}
{
fma.rn.f16x2 r691, r683, r689, r686;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r691;
mov.b32 r695, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r691;
mov.b32 r697, {high, high};
}
{
mul.f16x2 r699, r634, r697;
}
{
neg.f16x2 r702, r699;
}
{
fma.rn.f16x2 r704, r631, r695, r702;
}
{
mul.f16x2 r708, r631, r697;
}
{
fma.rn.f16x2 r711, r634, r695, r708;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r715, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r717, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r719, {low, high};
}
{
mul.f16x2 r720, r717, r719;
}
{
mul.f16x2 r723, r691, r715;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r691;
mov.b32 r726, {high, low};
}
{
fma.rn.f16x2 r728, r720, r726, r723;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r728;
mov.b32 r732, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r728;
mov.b32 r734, {high, high};
}
{
mul.f16x2 r736, r646, r734;
}
{
neg.f16x2 r739, r736;
}
{
fma.rn.f16x2 r741, r643, r732, r739;
}
{
mul.f16x2 r745, r643, r734;
}
{
fma.rn.f16x2 r748, r646, r732, r745;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r752, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r754, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r756, {low, high};
}
{
mul.f16x2 r757, r754, r756;
}
{
mul.f16x2 r760, r728, r752;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r728;
mov.b32 r763, {high, low};
}
{
fma.rn.f16x2 r765, r757, r763, r760;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r765;
mov.b32 r769, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r765;
mov.b32 r771, {high, high};
}
{
mul.f16x2 r773, r616, r771;
}
{
neg.f16x2 r776, r773;
}
{
fma.rn.f16x2 r778, r613, r769, r776;
}
{
mul.f16x2 r782, r613, r771;
}
{
fma.rn.f16x2 r785, r616, r769, r782;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r789, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r791, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r793, {low, high};
}
{
mul.f16x2 r794, r791, r793;
}
{
mul.f16x2 r797, r765, r789;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r765;
mov.b32 r800, {high, low};
}
{
fma.rn.f16x2 r802, r794, r800, r797;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r802;
mov.b32 r806, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r802;
mov.b32 r808, {high, high};
}
{
mul.f16x2 r810, r628, r808;
}
{
neg.f16x2 r813, r810;
}
{
fma.rn.f16x2 r815, r625, r806, r813;
}
{
mul.f16x2 r819, r625, r808;
}
{
fma.rn.f16x2 r822, r628, r806, r819;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r826, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r828, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r830, {low, high};
}
{
mul.f16x2 r831, r828, r830;
}
{
mul.f16x2 r834, r802, r826;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r802;
mov.b32 r837, {high, low};
}
{
fma.rn.f16x2 r839, r831, r837, r834;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r839;
mov.b32 r843, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r839;
mov.b32 r845, {high, high};
}
{
mul.f16x2 r847, r640, r845;
}
{
neg.f16x2 r850, r847;
}
{
fma.rn.f16x2 r852, r637, r843, r850;
}
{
mul.f16x2 r856, r637, r845;
}
{
fma.rn.f16x2 r859, r640, r843, r856;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r863, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r655;
mov.b32 r865, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r867, {low, high};
}
{
mul.f16x2 r868, r865, r867;
}
{
mul.f16x2 r871, r839, r863;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r839;
mov.b32 r874, {high, low};
}
{
fma.rn.f16x2 r876, r868, r874, r871;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r876;
mov.b32 r880, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r876;
mov.b32 r882, {high, high};
}
{
mul.f16x2 r884, r652, r882;
}
{
neg.f16x2 r887, r884;
}
{
fma.rn.f16x2 r889, r649, r880, r887;
}
{
mul.f16x2 r893, r649, r882;
}
{
fma.rn.f16x2 r896, r652, r880, r893;
}
shl.b32 r1585, r1575, 3;
and.b32 r1586, r1585, 56;
add.s32 r1587, r1579, r1586;
barrier.sync 0;
and.b32 r1588, r1577, 32256;
add.s32 r1589, r1587, r1588;
st.shared.u32 [r1589], r607;
st.shared.u32 [r1589+4], r610;
st.shared.u32 [r1589+64], r667;
st.shared.u32 [r1589+68], r674;
st.shared.u32 [r1589+128], r704;
st.shared.u32 [r1589+132], r711;
st.shared.u32 [r1589+192], r741;
st.shared.u32 [r1589+196], r748;
st.shared.u32 [r1589+256], r778;
st.shared.u32 [r1589+260], r785;
st.shared.u32 [r1589+320], r815;
st.shared.u32 [r1589+324], r822;
st.shared.u32 [r1589+384], r852;
st.shared.u32 [r1589+388], r859;
st.shared.u32 [r1589+448], r889;
st.shared.u32 [r1589+452], r896;
barrier.sync 0;
mad.lo.s32 r1590, r1583, -56, r1589;
ld.shared.u32 r918, [r1590];
ld.shared.u32 r921, [r1590+4];
ld.shared.u32 r968, [r1590+4096];
ld.shared.u32 r971, [r1590+4100];
ld.shared.u32 r930, [r1590+8192];
ld.shared.u32 r933, [r1590+8196];
ld.shared.u32 r980, [r1590+12288];
ld.shared.u32 r983, [r1590+12292];
ld.shared.u32 r919, [r1590+16384];
ld.shared.u32 r922, [r1590+16388];
ld.shared.u32 r969, [r1590+20480];
ld.shared.u32 r972, [r1590+20484];
ld.shared.u32 r931, [r1590+24576];
ld.shared.u32 r934, [r1590+24580];
ld.shared.u32 r981, [r1590+28672];
ld.shared.u32 r984, [r1590+28676];
{
add.f16x2 r917, r918, r919;
}
{
add.f16x2 r920, r921, r922;
}
{
sub.f16x2 r923, r918, r919;
}
{
sub.f16x2 r926, r921, r922;
}
{
add.f16x2 r929, r930, r931;
}
{
add.f16x2 r932, r933, r934;
}
{
sub.f16x2 r935, r930, r931;
}
{
sub.f16x2 r938, r933, r934;
}
{
neg.f16x2 r941, r935;
}
{
add.f16x2 r943, r917, r929;
}
{
add.f16x2 r946, r920, r932;
}
{
sub.f16x2 r949, r917, r929;
}
{
sub.f16x2 r952, r920, r932;
}
{
add.f16x2 r955, r923, r938;
}
{
add.f16x2 r958, r926, r941;
}
{
sub.f16x2 r961, r923, r938;
}
{
sub.f16x2 r964, r926, r941;
}
{
add.f16x2 r967, r968, r969;
}
{
add.f16x2 r970, r971, r972;
}
{
sub.f16x2 r973, r968, r969;
}
{
sub.f16x2 r976, r971, r972;
}
{
add.f16x2 r979, r980, r981;
}
{
add.f16x2 r982, r983, r984;
}
{
sub.f16x2 r985, r980, r981;
}
{
sub.f16x2 r988, r983, r984;
}
{
neg.f16x2 r991, r985;
}
{
add.f16x2 r993, r967, r979;
}
{
add.f16x2 r996, r970, r982;
}
{
sub.f16x2 r999, r967, r979;
}
{
sub.f16x2 r1002, r970, r982;
}
{
add.f16x2 r1005, r973, r988;
}
{
add.f16x2 r1008, r976, r991;
}
{
sub.f16x2 r1011, r973, r988;
}
{
sub.f16x2 r1014, r976, r991;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1017, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r1018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r1021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r1022, {low, high};
}
{
mul.f16x2 r1031, r1005, r1017;
}
{
mul.f16x2 r1034, r1008, r1018;
}
{
sub.f16x2 r1037, r1031, r1034;
}
{
mul.f16x2 r1040, r1005, r1018;
}
{
fma.rn.f16x2 r1043, r1008, r1017, r1040;
}
{
neg.f16x2 r1047, r999;
}
{
mul.f16x2 r1049, r1011, r1021;
}
{
mul.f16x2 r1052, r1014, r1022;
}
{
sub.f16x2 r1055, r1049, r1052;
}
{
mul.f16x2 r1058, r1011, r1022;
}
{
fma.rn.f16x2 r1061, r1014, r1021, r1058;
}
{
add.f16x2 r1065, r943, r993;
}
{
add.f16x2 r1068, r946, r996;
}
{
sub.f16x2 r1071, r943, r993;
}
{
sub.f16x2 r1074, r946, r996;
}
{
add.f16x2 r1077, r955, r1037;
}
{
add.f16x2 r1080, r958, r1043;
}
{
sub.f16x2 r1083, r955, r1037;
}
{
sub.f16x2 r1086, r958, r1043;
}
{
add.f16x2 r1089, r949, r1002;
}
{
add.f16x2 r1092, r952, r1047;
}
{
sub.f16x2 r1095, r949, r1002;
}
{
sub.f16x2 r1098, r952, r1047;
}
{
add.f16x2 r1101, r961, r1055;
}
{
add.f16x2 r1104, r964, r1061;
}
{
sub.f16x2 r1107, r961, r1055;
}
{
sub.f16x2 r1110, r964, r1061;
}
and.b32 r1591, r1575, 448;
bfe.u32 r1592, r1575, 6, 3;
cvt.rn.f32.u32 f173, r1592;
mul.f32 f174, f173, 0f3DC90FDB;
cos.approx.f32 f121, f174;
sin.approx.f32 f175, f174;
neg.f32 f122, f175;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f121;
cvt.rn.f16.f32 high, f122;
mov.b32 r1113, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1116, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1118, {high, high};
}
{
mul.f16x2 r1120, r1080, r1118;
}
{
neg.f16x2 r1123, r1120;
}
{
fma.rn.f16x2 r1125, r1077, r1116, r1123;
}
{
mul.f16x2 r1129, r1077, r1118;
}
{
fma.rn.f16x2 r1132, r1080, r1116, r1129;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1136, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1138, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r1140, {low, high};
}
{
mul.f16x2 r1141, r1138, r1140;
}
{
mul.f16x2 r1144, r1113, r1136;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1147, {high, low};
}
{
fma.rn.f16x2 r1149, r1141, r1147, r1144;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1149;
mov.b32 r1153, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1149;
mov.b32 r1155, {high, high};
}
{
mul.f16x2 r1157, r1092, r1155;
}
{
neg.f16x2 r1160, r1157;
}
{
fma.rn.f16x2 r1162, r1089, r1153, r1160;
}
{
mul.f16x2 r1166, r1089, r1155;
}
{
fma.rn.f16x2 r1169, r1092, r1153, r1166;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1173, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1175, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r1177, {low, high};
}
{
mul.f16x2 r1178, r1175, r1177;
}
{
mul.f16x2 r1181, r1149, r1173;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1149;
mov.b32 r1184, {high, low};
}
{
fma.rn.f16x2 r1186, r1178, r1184, r1181;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1186;
mov.b32 r1190, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1186;
mov.b32 r1192, {high, high};
}
{
mul.f16x2 r1194, r1104, r1192;
}
{
neg.f16x2 r1197, r1194;
}
{
fma.rn.f16x2 r1199, r1101, r1190, r1197;
}
{
mul.f16x2 r1203, r1101, r1192;
}
{
fma.rn.f16x2 r1206, r1104, r1190, r1203;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1210, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1212, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r1214, {low, high};
}
{
mul.f16x2 r1215, r1212, r1214;
}
{
mul.f16x2 r1218, r1186, r1210;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1186;
mov.b32 r1221, {high, low};
}
{
fma.rn.f16x2 r1223, r1215, r1221, r1218;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1223;
mov.b32 r1227, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1223;
mov.b32 r1229, {high, high};
}
{
mul.f16x2 r1231, r1074, r1229;
}
{
neg.f16x2 r1234, r1231;
}
{
fma.rn.f16x2 r1236, r1071, r1227, r1234;
}
{
mul.f16x2 r1240, r1071, r1229;
}
{
fma.rn.f16x2 r1243, r1074, r1227, r1240;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1247, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1249, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r1251, {low, high};
}
{
mul.f16x2 r1252, r1249, r1251;
}
{
mul.f16x2 r1255, r1223, r1247;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1223;
mov.b32 r1258, {high, low};
}
{
fma.rn.f16x2 r1260, r1252, r1258, r1255;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1260;
mov.b32 r1264, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1260;
mov.b32 r1266, {high, high};
}
{
mul.f16x2 r1268, r1086, r1266;
}
{
neg.f16x2 r1271, r1268;
}
{
fma.rn.f16x2 r1273, r1083, r1264, r1271;
}
{
mul.f16x2 r1277, r1083, r1266;
}
{
fma.rn.f16x2 r1280, r1086, r1264, r1277;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1284, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1286, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r1288, {low, high};
}
{
mul.f16x2 r1289, r1286, r1288;
}
{
mul.f16x2 r1292, r1260, r1284;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1260;
mov.b32 r1295, {high, low};
}
{
fma.rn.f16x2 r1297, r1289, r1295, r1292;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1297;
mov.b32 r1301, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1297;
mov.b32 r1303, {high, high};
}
{
mul.f16x2 r1305, r1098, r1303;
}
{
neg.f16x2 r1308, r1305;
}
{
fma.rn.f16x2 r1310, r1095, r1301, r1308;
}
{
mul.f16x2 r1314, r1095, r1303;
}
{
fma.rn.f16x2 r1317, r1098, r1301, r1314;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1321, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1113;
mov.b32 r1323, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f135;
cvt.rn.f16.f32 high, f136;
mov.b32 r1325, {low, high};
}
{
mul.f16x2 r1326, r1323, r1325;
}
{
mul.f16x2 r1329, r1297, r1321;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1297;
mov.b32 r1332, {high, low};
}
{
fma.rn.f16x2 r1334, r1326, r1332, r1329;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1334;
mov.b32 r1338, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1334;
mov.b32 r1340, {high, high};
}
{
mul.f16x2 r1342, r1110, r1340;
}
{
neg.f16x2 r1345, r1342;
}
{
fma.rn.f16x2 r1347, r1107, r1338, r1345;
}
{
mul.f16x2 r1351, r1107, r1340;
}
{
fma.rn.f16x2 r1354, r1110, r1338, r1351;
}
and.b32 r1593, r1585, 504;
add.s32 r1594, r1579, r1593;
barrier.sync 0;
and.b32 r1595, r1577, 28672;
add.s32 r1596, r1594, r1595;
st.shared.u32 [r1596], r1065;
st.shared.u32 [r1596+4], r1068;
st.shared.u32 [r1596+512], r1125;
st.shared.u32 [r1596+516], r1132;
st.shared.u32 [r1596+1024], r1162;
st.shared.u32 [r1596+1028], r1169;
st.shared.u32 [r1596+1536], r1199;
st.shared.u32 [r1596+1540], r1206;
st.shared.u32 [r1596+2048], r1236;
st.shared.u32 [r1596+2052], r1243;
st.shared.u32 [r1596+2560], r1273;
st.shared.u32 [r1596+2564], r1280;
st.shared.u32 [r1596+3072], r1310;
st.shared.u32 [r1596+3076], r1317;
st.shared.u32 [r1596+3584], r1347;
st.shared.u32 [r1596+3588], r1354;
barrier.sync 0;
mad.lo.s32 r1597, r1591, -56, r1596;
ld.shared.u32 r1376, [r1597];
ld.shared.u32 r1379, [r1597+4];
ld.shared.u32 r1426, [r1597+4096];
ld.shared.u32 r1429, [r1597+4100];
ld.shared.u32 r1388, [r1597+8192];
ld.shared.u32 r1391, [r1597+8196];
ld.shared.u32 r1438, [r1597+12288];
ld.shared.u32 r1441, [r1597+12292];
ld.shared.u32 r1377, [r1597+16384];
ld.shared.u32 r1380, [r1597+16388];
ld.shared.u32 r1427, [r1597+20480];
ld.shared.u32 r1430, [r1597+20484];
ld.shared.u32 r1389, [r1597+24576];
ld.shared.u32 r1392, [r1597+24580];
ld.shared.u32 r1439, [r1597+28672];
ld.shared.u32 r1442, [r1597+28676];
{
add.f16x2 r1375, r1376, r1377;
}
{
add.f16x2 r1378, r1379, r1380;
}
{
sub.f16x2 r1381, r1376, r1377;
}
{
sub.f16x2 r1384, r1379, r1380;
}
{
add.f16x2 r1387, r1388, r1389;
}
{
add.f16x2 r1390, r1391, r1392;
}
{
sub.f16x2 r1393, r1388, r1389;
}
{
sub.f16x2 r1396, r1391, r1392;
}
{
neg.f16x2 r1399, r1393;
}
{
add.f16x2 r1401, r1375, r1387;
}
{
add.f16x2 r1404, r1378, r1390;
}
{
sub.f16x2 r1407, r1375, r1387;
}
{
sub.f16x2 r1410, r1378, r1390;
}
{
add.f16x2 r1413, r1381, r1396;
}
{
add.f16x2 r1416, r1384, r1399;
}
{
sub.f16x2 r1419, r1381, r1396;
}
{
sub.f16x2 r1422, r1384, r1399;
}
{
add.f16x2 r1425, r1426, r1427;
}
{
add.f16x2 r1428, r1429, r1430;
}
{
sub.f16x2 r1431, r1426, r1427;
}
{
sub.f16x2 r1434, r1429, r1430;
}
{
add.f16x2 r1437, r1438, r1439;
}
{
add.f16x2 r1440, r1441, r1442;
}
{
sub.f16x2 r1443, r1438, r1439;
}
{
sub.f16x2 r1446, r1441, r1442;
}
{
neg.f16x2 r1449, r1443;
}
{
add.f16x2 r1451, r1425, r1437;
}
{
add.f16x2 r1454, r1428, r1440;
}
{
sub.f16x2 r1457, r1425, r1437;
}
{
sub.f16x2 r1460, r1428, r1440;
}
{
add.f16x2 r1463, r1431, r1446;
}
{
add.f16x2 r1466, r1434, r1449;
}
{
sub.f16x2 r1469, r1431, r1446;
}
{
sub.f16x2 r1472, r1434, r1449;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1475, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r1476, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r1479, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f150;
cvt.rn.f16.f32 high, f150;
mov.b32 r1480, {low, high};
}
{
mul.f16x2 r1489, r1463, r1475;
}
{
mul.f16x2 r1492, r1466, r1476;
}
{
sub.f16x2 r1495, r1489, r1492;
}
{
mul.f16x2 r1498, r1463, r1476;
}
{
fma.rn.f16x2 r1501, r1466, r1475, r1498;
}
{
neg.f16x2 r1505, r1457;
}
{
mul.f16x2 r1507, r1469, r1479;
}
{
mul.f16x2 r1510, r1472, r1480;
}
{
sub.f16x2 r1513, r1507, r1510;
}
{
mul.f16x2 r1516, r1469, r1480;
}
{
fma.rn.f16x2 r1519, r1472, r1479, r1516;
}
{
add.f16x2 %0, r1401, r1451;
}
{
add.f16x2 %1, r1404, r1454;
}
{
sub.f16x2 %8, r1401, r1451;
}
{
sub.f16x2 %9, r1404, r1454;
}
{
add.f16x2 %2, r1413, r1495;
}
{
add.f16x2 %3, r1416, r1501;
}
{
sub.f16x2 %10, r1413, r1495;
}
{
sub.f16x2 %11, r1416, r1501;
}
{
add.f16x2 %4, r1407, r1460;
}
{
add.f16x2 %5, r1410, r1505;
}
{
sub.f16x2 %12, r1407, r1460;
}
{
sub.f16x2 %13, r1410, r1505;
}
{
add.f16x2 %6, r1419, r1513;
}
{
add.f16x2 %7, r1422, r1519;
}
{
sub.f16x2 %14, r1419, r1513;
}
{
sub.f16x2 %15, r1422, r1519;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<852, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<853>;
.reg .b32 r<6210>;
.reg .b64 rd<3>;
mov.u32 r6125, %tid.y;
shl.b32 r6126, r6125, 14;
mov.u32 r6127, %64;
add.s32 r6128, r6127, r6126;
mov.u32 r6129, %tid.x;
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
add.f16x2 r16, %106, %101;
}
{
sub.f16x2 r19, %73, %128;
}
{
sub.f16x2 r22, %106, %101;
}
{
neg.f16x2 r25, r19;
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
add.f16x2 r39, r7, r22;
}
{
add.f16x2 r42, r10, r25;
}
{
sub.f16x2 r45, r7, r22;
}
{
sub.f16x2 r48, r10, r25;
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
add.f16x2 r63, %121, %115;
}
{
add.f16x2 r66, %93, %85;
}
{
sub.f16x2 r69, %121, %115;
}
{
sub.f16x2 r72, %93, %85;
}
{
neg.f16x2 r75, r69;
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
add.f16x2 r89, r57, r72;
}
{
add.f16x2 r92, r60, r75;
}
{
sub.f16x2 r95, r57, r72;
}
{
sub.f16x2 r98, r60, r75;
}
mov.f32 f668, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r101, {low, high};
}
mov.f32 f702, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r102, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
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
neg.f16x2 r131, r83;
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
add.f16x2 r173, r33, r86;
}
{
add.f16x2 r176, r36, r131;
}
{
sub.f16x2 r179, r33, r86;
}
{
sub.f16x2 r182, r36, r131;
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
neg.f16x2 r221, r215;
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
add.f16x2 r235, r203, r218;
}
{
add.f16x2 r238, r206, r221;
}
{
sub.f16x2 r241, r203, r218;
}
{
sub.f16x2 r244, r206, r221;
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
neg.f16x2 r271, r265;
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
add.f16x2 r285, r253, r268;
}
{
add.f16x2 r288, r256, r271;
}
{
sub.f16x2 r291, r253, r268;
}
{
sub.f16x2 r294, r256, r271;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
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
neg.f16x2 r327, r279;
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
add.f16x2 r369, r229, r282;
}
{
add.f16x2 r372, r232, r327;
}
{
sub.f16x2 r375, r229, r282;
}
{
sub.f16x2 r378, r232, r327;
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
mov.f32 f660, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f660;
cvt.rn.f16.f32 high, f660;
mov.b32 r393, {low, high};
}
mov.f32 f710, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r396, {low, high};
}
mov.f32 f676, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f676;
cvt.rn.f16.f32 high, f676;
mov.b32 r397, {low, high};
}
mov.f32 f708, 0fBF6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r398, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r404, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r405, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
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
neg.f16x2 r471, r351;
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
add.f16x2 r569, r155, r354;
}
{
add.f16x2 r572, r158, r471;
}
{
sub.f16x2 r575, r155, r354;
}
{
sub.f16x2 r578, r158, r471;
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
add.f16x2 r632, %122, %114;
}
{
sub.f16x2 r635, %86, %76;
}
{
sub.f16x2 r638, %122, %114;
}
{
neg.f16x2 r641, r635;
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
add.f16x2 r655, r623, r638;
}
{
add.f16x2 r658, r626, r641;
}
{
sub.f16x2 r661, r623, r638;
}
{
sub.f16x2 r664, r626, r641;
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
neg.f16x2 r691, r685;
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
add.f16x2 r705, r673, r688;
}
{
add.f16x2 r708, r676, r691;
}
{
sub.f16x2 r711, r673, r688;
}
{
sub.f16x2 r714, r676, r691;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r717, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r718, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r721, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
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
neg.f16x2 r747, r699;
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
add.f16x2 r789, r649, r702;
}
{
add.f16x2 r792, r652, r747;
}
{
sub.f16x2 r795, r649, r702;
}
{
sub.f16x2 r798, r652, r747;
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
add.f16x2 r813, %109, %100;
}
{
add.f16x2 r816, %79, %71;
}
{
sub.f16x2 r819, %109, %100;
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
neg.f16x2 r837, r831;
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
add.f16x2 r851, r819, r834;
}
{
add.f16x2 r854, r822, r837;
}
{
sub.f16x2 r857, r819, r834;
}
{
sub.f16x2 r860, r822, r837;
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
neg.f16x2 r887, r881;
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
add.f16x2 r901, r869, r884;
}
{
add.f16x2 r904, r872, r887;
}
{
sub.f16x2 r907, r869, r884;
}
{
sub.f16x2 r910, r872, r887;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r913, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r914, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r917, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
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
neg.f16x2 r943, r895;
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
add.f16x2 r985, r845, r898;
}
{
add.f16x2 r988, r848, r943;
}
{
sub.f16x2 r991, r845, r898;
}
{
sub.f16x2 r994, r848, r943;
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
cvt.rn.f16.f32 low, f660;
cvt.rn.f16.f32 high, f660;
mov.b32 r1009, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r1010, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r1011, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r1012, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f676;
cvt.rn.f16.f32 high, f676;
mov.b32 r1013, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r1014, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r1017, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r1018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r1019, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r1020, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r1021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
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
neg.f16x2 r1087, r967;
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
add.f16x2 r1185, r771, r970;
}
{
add.f16x2 r1188, r774, r1087;
}
{
sub.f16x2 r1191, r771, r970;
}
{
sub.f16x2 r1194, r774, r1087;
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
mov.f32 f656, 0f3F7B14BE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f656;
cvt.rn.f16.f32 high, f656;
mov.b32 r1233, {low, high};
}
mov.f32 f714, 0fBE47C5C2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f714;
cvt.rn.f16.f32 high, f714;
mov.b32 r1234, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f660;
cvt.rn.f16.f32 high, f660;
mov.b32 r1235, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r1236, {low, high};
}
mov.f32 f664, 0f3F54DB31;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f664;
cvt.rn.f16.f32 high, f664;
mov.b32 r1237, {low, high};
}
mov.f32 f706, 0fBF0E39DA;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f706;
cvt.rn.f16.f32 high, f706;
mov.b32 r1238, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r1239, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r1240, {low, high};
}
mov.f32 f672, 0f3F0E39DA;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f672;
cvt.rn.f16.f32 high, f672;
mov.b32 r1241, {low, high};
}
mov.f32 f704, 0fBF54DB31;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f704;
cvt.rn.f16.f32 high, f704;
mov.b32 r1242, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f676;
cvt.rn.f16.f32 high, f676;
mov.b32 r1243, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r1244, {low, high};
}
mov.f32 f680, 0f3E47C5C2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f680;
cvt.rn.f16.f32 high, f680;
mov.b32 r1245, {low, high};
}
mov.f32 f712, 0fBF7B14BE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f712;
cvt.rn.f16.f32 high, f712;
mov.b32 r1246, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f714;
cvt.rn.f16.f32 high, f714;
mov.b32 r1249, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f712;
cvt.rn.f16.f32 high, f712;
mov.b32 r1250, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r1251, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r1252, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f706;
cvt.rn.f16.f32 high, f706;
mov.b32 r1253, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f704;
cvt.rn.f16.f32 high, f704;
mov.b32 r1254, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r1255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r1256, {low, high};
}
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
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r1259, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r1260, {low, high};
}
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
neg.f16x2 r1407, r1143;
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
add.f16x2 r1617, r527, r1146;
}
{
add.f16x2 r1620, r530, r1407;
}
{
sub.f16x2 r1623, r527, r1146;
}
{
sub.f16x2 r1626, r530, r1407;
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
shl.b32 r6131, r6129, 7;
and.b32 r6132, r6131, -16384;
add.s32 r6133, r6128, r6132;
and.b32 r6145, r6129, 127;
cvt.rn.f32.u32 f845, r6145;
mul.f32 f846, f845, 0f3AC90FDB;
cos.approx.f32 f357, f846;
sin.approx.f32 f847, f846;
neg.f32 f358, f847;
mov.f32 f852, 0f3F800000;
mov.f32 f851, 0fBF800000;
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
neg.f16x2 r1723, r1720;
}
{
fma.rn.f16x2 r1725, r1533, r1716, r1723;
}
{
mul.f16x2 r1729, r1533, r1718;
}
{
fma.rn.f16x2 r1732, r1536, r1716, r1729;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r1760, r1757;
}
{
fma.rn.f16x2 r1762, r1545, r1753, r1760;
}
{
mul.f16x2 r1766, r1545, r1755;
}
{
fma.rn.f16x2 r1769, r1548, r1753, r1766;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r1797, r1794;
}
{
fma.rn.f16x2 r1799, r1557, r1790, r1797;
}
{
mul.f16x2 r1803, r1557, r1792;
}
{
fma.rn.f16x2 r1806, r1560, r1790, r1803;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r1834, r1831;
}
{
fma.rn.f16x2 r1836, r1569, r1827, r1834;
}
{
mul.f16x2 r1840, r1569, r1829;
}
{
fma.rn.f16x2 r1843, r1572, r1827, r1840;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r1871, r1868;
}
{
fma.rn.f16x2 r1873, r1581, r1864, r1871;
}
{
mul.f16x2 r1877, r1581, r1866;
}
{
fma.rn.f16x2 r1880, r1584, r1864, r1877;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r1908, r1905;
}
{
fma.rn.f16x2 r1910, r1593, r1901, r1908;
}
{
mul.f16x2 r1914, r1593, r1903;
}
{
fma.rn.f16x2 r1917, r1596, r1901, r1914;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r1945, r1942;
}
{
fma.rn.f16x2 r1947, r1605, r1938, r1945;
}
{
mul.f16x2 r1951, r1605, r1940;
}
{
fma.rn.f16x2 r1954, r1608, r1938, r1951;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r1982, r1979;
}
{
fma.rn.f16x2 r1984, r1617, r1975, r1982;
}
{
mul.f16x2 r1988, r1617, r1977;
}
{
fma.rn.f16x2 r1991, r1620, r1975, r1988;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2019, r2016;
}
{
fma.rn.f16x2 r2021, r1629, r2012, r2019;
}
{
mul.f16x2 r2025, r1629, r2014;
}
{
fma.rn.f16x2 r2028, r1632, r2012, r2025;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2056, r2053;
}
{
fma.rn.f16x2 r2058, r1641, r2049, r2056;
}
{
mul.f16x2 r2062, r1641, r2051;
}
{
fma.rn.f16x2 r2065, r1644, r2049, r2062;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2093, r2090;
}
{
fma.rn.f16x2 r2095, r1653, r2086, r2093;
}
{
mul.f16x2 r2099, r1653, r2088;
}
{
fma.rn.f16x2 r2102, r1656, r2086, r2099;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2130, r2127;
}
{
fma.rn.f16x2 r2132, r1665, r2123, r2130;
}
{
mul.f16x2 r2136, r1665, r2125;
}
{
fma.rn.f16x2 r2139, r1668, r2123, r2136;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2167, r2164;
}
{
fma.rn.f16x2 r2169, r1677, r2160, r2167;
}
{
mul.f16x2 r2173, r1677, r2162;
}
{
fma.rn.f16x2 r2176, r1680, r2160, r2173;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2204, r2201;
}
{
fma.rn.f16x2 r2206, r1689, r2197, r2204;
}
{
mul.f16x2 r2210, r1689, r2199;
}
{
fma.rn.f16x2 r2213, r1692, r2197, r2210;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2241, r2238;
}
{
fma.rn.f16x2 r2243, r1701, r2234, r2241;
}
{
mul.f16x2 r2247, r1701, r2236;
}
{
fma.rn.f16x2 r2250, r1704, r2234, r2247;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2278, r2275;
}
{
fma.rn.f16x2 r2280, r1527, r2271, r2278;
}
{
mul.f16x2 r2284, r1527, r2273;
}
{
fma.rn.f16x2 r2287, r1530, r2271, r2284;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2315, r2312;
}
{
fma.rn.f16x2 r2317, r1539, r2308, r2315;
}
{
mul.f16x2 r2321, r1539, r2310;
}
{
fma.rn.f16x2 r2324, r1542, r2308, r2321;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2352, r2349;
}
{
fma.rn.f16x2 r2354, r1551, r2345, r2352;
}
{
mul.f16x2 r2358, r1551, r2347;
}
{
fma.rn.f16x2 r2361, r1554, r2345, r2358;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2389, r2386;
}
{
fma.rn.f16x2 r2391, r1563, r2382, r2389;
}
{
mul.f16x2 r2395, r1563, r2384;
}
{
fma.rn.f16x2 r2398, r1566, r2382, r2395;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2426, r2423;
}
{
fma.rn.f16x2 r2428, r1575, r2419, r2426;
}
{
mul.f16x2 r2432, r1575, r2421;
}
{
fma.rn.f16x2 r2435, r1578, r2419, r2432;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2463, r2460;
}
{
fma.rn.f16x2 r2465, r1587, r2456, r2463;
}
{
mul.f16x2 r2469, r1587, r2458;
}
{
fma.rn.f16x2 r2472, r1590, r2456, r2469;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2500, r2497;
}
{
fma.rn.f16x2 r2502, r1599, r2493, r2500;
}
{
mul.f16x2 r2506, r1599, r2495;
}
{
fma.rn.f16x2 r2509, r1602, r2493, r2506;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2537, r2534;
}
{
fma.rn.f16x2 r2539, r1611, r2530, r2537;
}
{
mul.f16x2 r2543, r1611, r2532;
}
{
fma.rn.f16x2 r2546, r1614, r2530, r2543;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2574, r2571;
}
{
fma.rn.f16x2 r2576, r1623, r2567, r2574;
}
{
mul.f16x2 r2580, r1623, r2569;
}
{
fma.rn.f16x2 r2583, r1626, r2567, r2580;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2611, r2608;
}
{
fma.rn.f16x2 r2613, r1635, r2604, r2611;
}
{
mul.f16x2 r2617, r1635, r2606;
}
{
fma.rn.f16x2 r2620, r1638, r2604, r2617;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2648, r2645;
}
{
fma.rn.f16x2 r2650, r1647, r2641, r2648;
}
{
mul.f16x2 r2654, r1647, r2643;
}
{
fma.rn.f16x2 r2657, r1650, r2641, r2654;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2685, r2682;
}
{
fma.rn.f16x2 r2687, r1659, r2678, r2685;
}
{
mul.f16x2 r2691, r1659, r2680;
}
{
fma.rn.f16x2 r2694, r1662, r2678, r2691;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2722, r2719;
}
{
fma.rn.f16x2 r2724, r1671, r2715, r2722;
}
{
mul.f16x2 r2728, r1671, r2717;
}
{
fma.rn.f16x2 r2731, r1674, r2715, r2728;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2759, r2756;
}
{
fma.rn.f16x2 r2761, r1683, r2752, r2759;
}
{
mul.f16x2 r2765, r1683, r2754;
}
{
fma.rn.f16x2 r2768, r1686, r2752, r2765;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2796, r2793;
}
{
fma.rn.f16x2 r2798, r1695, r2789, r2796;
}
{
mul.f16x2 r2802, r1695, r2791;
}
{
fma.rn.f16x2 r2805, r1698, r2789, r2802;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r2833, r2830;
}
{
fma.rn.f16x2 r2835, r1707, r2826, r2833;
}
{
mul.f16x2 r2839, r1707, r2828;
}
{
fma.rn.f16x2 r2842, r1710, r2826, r2839;
}
barrier.sync 0;
and.b32 r6134, r6131, 16256;
add.s32 r6135, r6133, r6134;
st.shared.v4.f32 [r6135], {r1521, r1725, r1762, r1799};
st.shared.v4.f32 [r6135+16], {r1836, r1873, r1910, r1947};
st.shared.v4.f32 [r6135+32], {r1984, r2021, r2058, r2095};
st.shared.v4.f32 [r6135+48], {r2132, r2169, r2206, r2243};
st.shared.v4.f32 [r6135+64], {r2280, r2317, r2354, r2391};
st.shared.v4.f32 [r6135+80], {r2428, r2465, r2502, r2539};
st.shared.v4.f32 [r6135+96], {r2576, r2613, r2650, r2687};
st.shared.v4.f32 [r6135+112], {r2724, r2761, r2798, r2835};
barrier.sync 0;
mad.lo.s32 r6136, r6145, -124, r6135;
ld.shared.u32 r2864, [r6136];
ld.shared.u32 r3480, [r6136+512];
ld.shared.u32 r3060, [r6136+1024];
ld.shared.u32 r3676, [r6136+1536];
ld.shared.u32 r2914, [r6136+2048];
ld.shared.u32 r3530, [r6136+2560];
ld.shared.u32 r3110, [r6136+3072];
ld.shared.u32 r3726, [r6136+3584];
ld.shared.u32 r2876, [r6136+4096];
ld.shared.u32 r3492, [r6136+4608];
ld.shared.u32 r3072, [r6136+5120];
ld.shared.u32 r3688, [r6136+5632];
ld.shared.u32 r2926, [r6136+6144];
ld.shared.u32 r3542, [r6136+6656];
ld.shared.u32 r3122, [r6136+7168];
ld.shared.u32 r3738, [r6136+7680];
ld.shared.u32 r2865, [r6136+8192];
ld.shared.u32 r3481, [r6136+8704];
ld.shared.u32 r3061, [r6136+9216];
ld.shared.u32 r3677, [r6136+9728];
ld.shared.u32 r2915, [r6136+10240];
ld.shared.u32 r3531, [r6136+10752];
ld.shared.u32 r3111, [r6136+11264];
ld.shared.u32 r3727, [r6136+11776];
ld.shared.u32 r2877, [r6136+12288];
ld.shared.u32 r3493, [r6136+12800];
ld.shared.u32 r3073, [r6136+13312];
ld.shared.u32 r3689, [r6136+13824];
ld.shared.u32 r2927, [r6136+14336];
ld.shared.u32 r3543, [r6136+14848];
ld.shared.u32 r3123, [r6136+15360];
ld.shared.u32 r3739, [r6136+15872];
barrier.sync 0;
st.shared.v4.f32 [r6135], {r1524, r1732, r1769, r1806};
st.shared.v4.f32 [r6135+16], {r1843, r1880, r1917, r1954};
st.shared.v4.f32 [r6135+32], {r1991, r2028, r2065, r2102};
st.shared.v4.f32 [r6135+48], {r2139, r2176, r2213, r2250};
st.shared.v4.f32 [r6135+64], {r2287, r2324, r2361, r2398};
st.shared.v4.f32 [r6135+80], {r2435, r2472, r2509, r2546};
st.shared.v4.f32 [r6135+96], {r2583, r2620, r2657, r2694};
st.shared.v4.f32 [r6135+112], {r2731, r2768, r2805, r2842};
barrier.sync 0;
ld.shared.u32 r2867, [r6136];
ld.shared.u32 r3483, [r6136+512];
ld.shared.u32 r3063, [r6136+1024];
ld.shared.u32 r3679, [r6136+1536];
ld.shared.u32 r2917, [r6136+2048];
ld.shared.u32 r3533, [r6136+2560];
ld.shared.u32 r3113, [r6136+3072];
ld.shared.u32 r3729, [r6136+3584];
ld.shared.u32 r2879, [r6136+4096];
ld.shared.u32 r3495, [r6136+4608];
ld.shared.u32 r3075, [r6136+5120];
ld.shared.u32 r3691, [r6136+5632];
ld.shared.u32 r2929, [r6136+6144];
ld.shared.u32 r3545, [r6136+6656];
ld.shared.u32 r3125, [r6136+7168];
ld.shared.u32 r3741, [r6136+7680];
ld.shared.u32 r2868, [r6136+8192];
ld.shared.u32 r3484, [r6136+8704];
ld.shared.u32 r3064, [r6136+9216];
ld.shared.u32 r3680, [r6136+9728];
ld.shared.u32 r2918, [r6136+10240];
ld.shared.u32 r3534, [r6136+10752];
ld.shared.u32 r3114, [r6136+11264];
ld.shared.u32 r3730, [r6136+11776];
ld.shared.u32 r2880, [r6136+12288];
ld.shared.u32 r3496, [r6136+12800];
ld.shared.u32 r3076, [r6136+13312];
ld.shared.u32 r3692, [r6136+13824];
ld.shared.u32 r2930, [r6136+14336];
ld.shared.u32 r3546, [r6136+14848];
ld.shared.u32 r3126, [r6136+15360];
ld.shared.u32 r3742, [r6136+15872];
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
neg.f16x2 r2887, r2881;
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
add.f16x2 r2901, r2869, r2884;
}
{
add.f16x2 r2904, r2872, r2887;
}
{
sub.f16x2 r2907, r2869, r2884;
}
{
sub.f16x2 r2910, r2872, r2887;
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
neg.f16x2 r2937, r2931;
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
add.f16x2 r2951, r2919, r2934;
}
{
add.f16x2 r2954, r2922, r2937;
}
{
sub.f16x2 r2957, r2919, r2934;
}
{
sub.f16x2 r2960, r2922, r2937;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r2963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r2964, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r2967, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
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
neg.f16x2 r2993, r2945;
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
add.f16x2 r3035, r2895, r2948;
}
{
add.f16x2 r3038, r2898, r2993;
}
{
sub.f16x2 r3041, r2895, r2948;
}
{
sub.f16x2 r3044, r2898, r2993;
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
neg.f16x2 r3083, r3077;
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
add.f16x2 r3097, r3065, r3080;
}
{
add.f16x2 r3100, r3068, r3083;
}
{
sub.f16x2 r3103, r3065, r3080;
}
{
sub.f16x2 r3106, r3068, r3083;
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
neg.f16x2 r3133, r3127;
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
add.f16x2 r3147, r3115, r3130;
}
{
add.f16x2 r3150, r3118, r3133;
}
{
sub.f16x2 r3153, r3115, r3130;
}
{
sub.f16x2 r3156, r3118, r3133;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r3159, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3160, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3163, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
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
neg.f16x2 r3189, r3141;
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
add.f16x2 r3231, r3091, r3144;
}
{
add.f16x2 r3234, r3094, r3189;
}
{
sub.f16x2 r3237, r3091, r3144;
}
{
sub.f16x2 r3240, r3094, r3189;
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
cvt.rn.f16.f32 low, f660;
cvt.rn.f16.f32 high, f660;
mov.b32 r3255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r3256, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r3257, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3258, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f676;
cvt.rn.f16.f32 high, f676;
mov.b32 r3259, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r3260, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r3263, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r3264, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3265, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3266, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r3267, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
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
neg.f16x2 r3333, r3213;
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
add.f16x2 r3431, r3017, r3216;
}
{
add.f16x2 r3434, r3020, r3333;
}
{
sub.f16x2 r3437, r3017, r3216;
}
{
sub.f16x2 r3440, r3020, r3333;
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
neg.f16x2 r3503, r3497;
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
add.f16x2 r3517, r3485, r3500;
}
{
add.f16x2 r3520, r3488, r3503;
}
{
sub.f16x2 r3523, r3485, r3500;
}
{
sub.f16x2 r3526, r3488, r3503;
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
neg.f16x2 r3553, r3547;
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
add.f16x2 r3567, r3535, r3550;
}
{
add.f16x2 r3570, r3538, r3553;
}
{
sub.f16x2 r3573, r3535, r3550;
}
{
sub.f16x2 r3576, r3538, r3553;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r3579, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3580, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3583, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
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
neg.f16x2 r3609, r3561;
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
add.f16x2 r3651, r3511, r3564;
}
{
add.f16x2 r3654, r3514, r3609;
}
{
sub.f16x2 r3657, r3511, r3564;
}
{
sub.f16x2 r3660, r3514, r3609;
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
neg.f16x2 r3699, r3693;
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
add.f16x2 r3713, r3681, r3696;
}
{
add.f16x2 r3716, r3684, r3699;
}
{
sub.f16x2 r3719, r3681, r3696;
}
{
sub.f16x2 r3722, r3684, r3699;
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
neg.f16x2 r3749, r3743;
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
add.f16x2 r3763, r3731, r3746;
}
{
add.f16x2 r3766, r3734, r3749;
}
{
sub.f16x2 r3769, r3731, r3746;
}
{
sub.f16x2 r3772, r3734, r3749;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r3775, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3776, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3779, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
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
neg.f16x2 r3805, r3757;
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
add.f16x2 r3847, r3707, r3760;
}
{
add.f16x2 r3850, r3710, r3805;
}
{
sub.f16x2 r3853, r3707, r3760;
}
{
sub.f16x2 r3856, r3710, r3805;
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
cvt.rn.f16.f32 low, f660;
cvt.rn.f16.f32 high, f660;
mov.b32 r3871, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r3872, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r3873, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3874, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f676;
cvt.rn.f16.f32 high, f676;
mov.b32 r3875, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r3876, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r3879, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r3880, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3881, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r3882, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r3883, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
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
neg.f16x2 r3949, r3829;
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
add.f16x2 r4047, r3633, r3832;
}
{
add.f16x2 r4050, r3636, r3949;
}
{
sub.f16x2 r4053, r3633, r3832;
}
{
sub.f16x2 r4056, r3636, r3949;
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
cvt.rn.f16.f32 low, f656;
cvt.rn.f16.f32 high, f656;
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
cvt.rn.f16.f32 low, f660;
cvt.rn.f16.f32 high, f660;
mov.b32 r4097, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r4098, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f664;
cvt.rn.f16.f32 high, f664;
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
cvt.rn.f16.f32 low, f668;
cvt.rn.f16.f32 high, f668;
mov.b32 r4101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r4102, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f672;
cvt.rn.f16.f32 high, f672;
mov.b32 r4103, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f704;
cvt.rn.f16.f32 high, f704;
mov.b32 r4104, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f676;
cvt.rn.f16.f32 high, f676;
mov.b32 r4105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r4106, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f680;
cvt.rn.f16.f32 high, f680;
mov.b32 r4107, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f712;
cvt.rn.f16.f32 high, f712;
mov.b32 r4108, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f714;
cvt.rn.f16.f32 high, f714;
mov.b32 r4111, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f712;
cvt.rn.f16.f32 high, f712;
mov.b32 r4112, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
mov.b32 r4113, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r4114, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f706;
cvt.rn.f16.f32 high, f706;
mov.b32 r4115, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f704;
cvt.rn.f16.f32 high, f704;
mov.b32 r4116, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
mov.b32 r4117, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f702;
cvt.rn.f16.f32 high, f702;
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
cvt.rn.f16.f32 low, f708;
cvt.rn.f16.f32 high, f708;
mov.b32 r4121, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f710;
cvt.rn.f16.f32 high, f710;
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
neg.f16x2 r4269, r4005;
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
add.f16x2 r4479, r3389, r4008;
}
{
add.f16x2 r4482, r3392, r4269;
}
{
sub.f16x2 r4485, r3389, r4008;
}
{
sub.f16x2 r4488, r3392, r4269;
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
and.b32 r6137, r6129, 96;
bfe.u32 r6138, r6129, 5, 2;
shl.b32 r6139, r6129, 2;
and.b32 r6140, r6139, 124;
add.s32 r6141, r6133, r6140;
cvt.rn.f32.u32 f848, r6138;
mul.f32 f849, f848, 0f3D490FDB;
cos.approx.f32 f779, f849;
sin.approx.f32 f850, f849;
neg.f32 f780, f850;
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
neg.f16x2 r4585, r4582;
}
{
fma.rn.f16x2 r4587, r4395, r4578, r4585;
}
{
mul.f16x2 r4591, r4395, r4580;
}
{
fma.rn.f16x2 r4594, r4398, r4578, r4591;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4622, r4619;
}
{
fma.rn.f16x2 r4624, r4407, r4615, r4622;
}
{
mul.f16x2 r4628, r4407, r4617;
}
{
fma.rn.f16x2 r4631, r4410, r4615, r4628;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4659, r4656;
}
{
fma.rn.f16x2 r4661, r4419, r4652, r4659;
}
{
mul.f16x2 r4665, r4419, r4654;
}
{
fma.rn.f16x2 r4668, r4422, r4652, r4665;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4696, r4693;
}
{
fma.rn.f16x2 r4698, r4431, r4689, r4696;
}
{
mul.f16x2 r4702, r4431, r4691;
}
{
fma.rn.f16x2 r4705, r4434, r4689, r4702;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4733, r4730;
}
{
fma.rn.f16x2 r4735, r4443, r4726, r4733;
}
{
mul.f16x2 r4739, r4443, r4728;
}
{
fma.rn.f16x2 r4742, r4446, r4726, r4739;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4770, r4767;
}
{
fma.rn.f16x2 r4772, r4455, r4763, r4770;
}
{
mul.f16x2 r4776, r4455, r4765;
}
{
fma.rn.f16x2 r4779, r4458, r4763, r4776;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4807, r4804;
}
{
fma.rn.f16x2 r4809, r4467, r4800, r4807;
}
{
mul.f16x2 r4813, r4467, r4802;
}
{
fma.rn.f16x2 r4816, r4470, r4800, r4813;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4844, r4841;
}
{
fma.rn.f16x2 r4846, r4479, r4837, r4844;
}
{
mul.f16x2 r4850, r4479, r4839;
}
{
fma.rn.f16x2 r4853, r4482, r4837, r4850;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4881, r4878;
}
{
fma.rn.f16x2 r4883, r4491, r4874, r4881;
}
{
mul.f16x2 r4887, r4491, r4876;
}
{
fma.rn.f16x2 r4890, r4494, r4874, r4887;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4918, r4915;
}
{
fma.rn.f16x2 r4920, r4503, r4911, r4918;
}
{
mul.f16x2 r4924, r4503, r4913;
}
{
fma.rn.f16x2 r4927, r4506, r4911, r4924;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4955, r4952;
}
{
fma.rn.f16x2 r4957, r4515, r4948, r4955;
}
{
mul.f16x2 r4961, r4515, r4950;
}
{
fma.rn.f16x2 r4964, r4518, r4948, r4961;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r4992, r4989;
}
{
fma.rn.f16x2 r4994, r4527, r4985, r4992;
}
{
mul.f16x2 r4998, r4527, r4987;
}
{
fma.rn.f16x2 r5001, r4530, r4985, r4998;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5029, r5026;
}
{
fma.rn.f16x2 r5031, r4539, r5022, r5029;
}
{
mul.f16x2 r5035, r4539, r5024;
}
{
fma.rn.f16x2 r5038, r4542, r5022, r5035;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5066, r5063;
}
{
fma.rn.f16x2 r5068, r4551, r5059, r5066;
}
{
mul.f16x2 r5072, r4551, r5061;
}
{
fma.rn.f16x2 r5075, r4554, r5059, r5072;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5103, r5100;
}
{
fma.rn.f16x2 r5105, r4563, r5096, r5103;
}
{
mul.f16x2 r5109, r4563, r5098;
}
{
fma.rn.f16x2 r5112, r4566, r5096, r5109;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5140, r5137;
}
{
fma.rn.f16x2 r5142, r4389, r5133, r5140;
}
{
mul.f16x2 r5146, r4389, r5135;
}
{
fma.rn.f16x2 r5149, r4392, r5133, r5146;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5177, r5174;
}
{
fma.rn.f16x2 r5179, r4401, r5170, r5177;
}
{
mul.f16x2 r5183, r4401, r5172;
}
{
fma.rn.f16x2 r5186, r4404, r5170, r5183;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5214, r5211;
}
{
fma.rn.f16x2 r5216, r4413, r5207, r5214;
}
{
mul.f16x2 r5220, r4413, r5209;
}
{
fma.rn.f16x2 r5223, r4416, r5207, r5220;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5251, r5248;
}
{
fma.rn.f16x2 r5253, r4425, r5244, r5251;
}
{
mul.f16x2 r5257, r4425, r5246;
}
{
fma.rn.f16x2 r5260, r4428, r5244, r5257;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5288, r5285;
}
{
fma.rn.f16x2 r5290, r4437, r5281, r5288;
}
{
mul.f16x2 r5294, r4437, r5283;
}
{
fma.rn.f16x2 r5297, r4440, r5281, r5294;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5325, r5322;
}
{
fma.rn.f16x2 r5327, r4449, r5318, r5325;
}
{
mul.f16x2 r5331, r4449, r5320;
}
{
fma.rn.f16x2 r5334, r4452, r5318, r5331;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5362, r5359;
}
{
fma.rn.f16x2 r5364, r4461, r5355, r5362;
}
{
mul.f16x2 r5368, r4461, r5357;
}
{
fma.rn.f16x2 r5371, r4464, r5355, r5368;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5399, r5396;
}
{
fma.rn.f16x2 r5401, r4473, r5392, r5399;
}
{
mul.f16x2 r5405, r4473, r5394;
}
{
fma.rn.f16x2 r5408, r4476, r5392, r5405;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5436, r5433;
}
{
fma.rn.f16x2 r5438, r4485, r5429, r5436;
}
{
mul.f16x2 r5442, r4485, r5431;
}
{
fma.rn.f16x2 r5445, r4488, r5429, r5442;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5473, r5470;
}
{
fma.rn.f16x2 r5475, r4497, r5466, r5473;
}
{
mul.f16x2 r5479, r4497, r5468;
}
{
fma.rn.f16x2 r5482, r4500, r5466, r5479;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5510, r5507;
}
{
fma.rn.f16x2 r5512, r4509, r5503, r5510;
}
{
mul.f16x2 r5516, r4509, r5505;
}
{
fma.rn.f16x2 r5519, r4512, r5503, r5516;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5547, r5544;
}
{
fma.rn.f16x2 r5549, r4521, r5540, r5547;
}
{
mul.f16x2 r5553, r4521, r5542;
}
{
fma.rn.f16x2 r5556, r4524, r5540, r5553;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5584, r5581;
}
{
fma.rn.f16x2 r5586, r4533, r5577, r5584;
}
{
mul.f16x2 r5590, r4533, r5579;
}
{
fma.rn.f16x2 r5593, r4536, r5577, r5590;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5621, r5618;
}
{
fma.rn.f16x2 r5623, r4545, r5614, r5621;
}
{
mul.f16x2 r5627, r4545, r5616;
}
{
fma.rn.f16x2 r5630, r4548, r5614, r5627;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5658, r5655;
}
{
fma.rn.f16x2 r5660, r4557, r5651, r5658;
}
{
mul.f16x2 r5664, r4557, r5653;
}
{
fma.rn.f16x2 r5667, r4560, r5651, r5664;
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
cvt.rn.f16.f32 low, f851;
cvt.rn.f16.f32 high, f852;
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
neg.f16x2 r5695, r5692;
}
{
fma.rn.f16x2 r5697, r4569, r5688, r5695;
}
{
mul.f16x2 r5701, r4569, r5690;
}
{
fma.rn.f16x2 r5704, r4572, r5688, r5701;
}
barrier.sync 0;
and.b32 r6142, r6131, 12288;
add.s32 r6143, r6141, r6142;
st.shared.u32 [r6143], r4383;
st.shared.u32 [r6143+128], r4587;
st.shared.u32 [r6143+256], r4624;
st.shared.u32 [r6143+384], r4661;
st.shared.u32 [r6143+512], r4698;
st.shared.u32 [r6143+640], r4735;
st.shared.u32 [r6143+768], r4772;
st.shared.u32 [r6143+896], r4809;
st.shared.u32 [r6143+1024], r4846;
st.shared.u32 [r6143+1152], r4883;
st.shared.u32 [r6143+1280], r4920;
st.shared.u32 [r6143+1408], r4957;
st.shared.u32 [r6143+1536], r4994;
st.shared.u32 [r6143+1664], r5031;
st.shared.u32 [r6143+1792], r5068;
st.shared.u32 [r6143+1920], r5105;
st.shared.u32 [r6143+2048], r5142;
st.shared.u32 [r6143+2176], r5179;
st.shared.u32 [r6143+2304], r5216;
st.shared.u32 [r6143+2432], r5253;
st.shared.u32 [r6143+2560], r5290;
st.shared.u32 [r6143+2688], r5327;
st.shared.u32 [r6143+2816], r5364;
st.shared.u32 [r6143+2944], r5401;
st.shared.u32 [r6143+3072], r5438;
st.shared.u32 [r6143+3200], r5475;
st.shared.u32 [r6143+3328], r5512;
st.shared.u32 [r6143+3456], r5549;
st.shared.u32 [r6143+3584], r5586;
st.shared.u32 [r6143+3712], r5623;
st.shared.u32 [r6143+3840], r5660;
st.shared.u32 [r6143+3968], r5697;
barrier.sync 0;
mad.lo.s32 r6144, r6137, -124, r6143;
ld.shared.u32 r5726, [r6144];
ld.shared.u32 r5776, [r6144+512];
ld.shared.u32 r5826, [r6144+1024];
ld.shared.u32 r5876, [r6144+1536];
ld.shared.u32 r5926, [r6144+2048];
ld.shared.u32 r5976, [r6144+2560];
ld.shared.u32 r6026, [r6144+3072];
ld.shared.u32 r6076, [r6144+3584];
ld.shared.u32 r5738, [r6144+4096];
ld.shared.u32 r5788, [r6144+4608];
ld.shared.u32 r5838, [r6144+5120];
ld.shared.u32 r5888, [r6144+5632];
ld.shared.u32 r5938, [r6144+6144];
ld.shared.u32 r5988, [r6144+6656];
ld.shared.u32 r6038, [r6144+7168];
ld.shared.u32 r6088, [r6144+7680];
ld.shared.u32 r5727, [r6144+8192];
ld.shared.u32 r5777, [r6144+8704];
ld.shared.u32 r5827, [r6144+9216];
ld.shared.u32 r5877, [r6144+9728];
ld.shared.u32 r5927, [r6144+10240];
ld.shared.u32 r5977, [r6144+10752];
ld.shared.u32 r6027, [r6144+11264];
ld.shared.u32 r6077, [r6144+11776];
ld.shared.u32 r5739, [r6144+12288];
ld.shared.u32 r5789, [r6144+12800];
ld.shared.u32 r5839, [r6144+13312];
ld.shared.u32 r5889, [r6144+13824];
ld.shared.u32 r5939, [r6144+14336];
ld.shared.u32 r5989, [r6144+14848];
ld.shared.u32 r6039, [r6144+15360];
ld.shared.u32 r6089, [r6144+15872];
barrier.sync 0;
st.shared.u32 [r6143], r4386;
st.shared.u32 [r6143+128], r4594;
st.shared.u32 [r6143+256], r4631;
st.shared.u32 [r6143+384], r4668;
st.shared.u32 [r6143+512], r4705;
st.shared.u32 [r6143+640], r4742;
st.shared.u32 [r6143+768], r4779;
st.shared.u32 [r6143+896], r4816;
st.shared.u32 [r6143+1024], r4853;
st.shared.u32 [r6143+1152], r4890;
st.shared.u32 [r6143+1280], r4927;
st.shared.u32 [r6143+1408], r4964;
st.shared.u32 [r6143+1536], r5001;
st.shared.u32 [r6143+1664], r5038;
st.shared.u32 [r6143+1792], r5075;
st.shared.u32 [r6143+1920], r5112;
st.shared.u32 [r6143+2048], r5149;
st.shared.u32 [r6143+2176], r5186;
st.shared.u32 [r6143+2304], r5223;
st.shared.u32 [r6143+2432], r5260;
st.shared.u32 [r6143+2560], r5297;
st.shared.u32 [r6143+2688], r5334;
st.shared.u32 [r6143+2816], r5371;
st.shared.u32 [r6143+2944], r5408;
st.shared.u32 [r6143+3072], r5445;
st.shared.u32 [r6143+3200], r5482;
st.shared.u32 [r6143+3328], r5519;
st.shared.u32 [r6143+3456], r5556;
st.shared.u32 [r6143+3584], r5593;
st.shared.u32 [r6143+3712], r5630;
st.shared.u32 [r6143+3840], r5667;
st.shared.u32 [r6143+3968], r5704;
barrier.sync 0;
ld.shared.u32 r5729, [r6144];
ld.shared.u32 r5779, [r6144+512];
ld.shared.u32 r5829, [r6144+1024];
ld.shared.u32 r5879, [r6144+1536];
ld.shared.u32 r5929, [r6144+2048];
ld.shared.u32 r5979, [r6144+2560];
ld.shared.u32 r6029, [r6144+3072];
ld.shared.u32 r6079, [r6144+3584];
ld.shared.u32 r5741, [r6144+4096];
ld.shared.u32 r5791, [r6144+4608];
ld.shared.u32 r5841, [r6144+5120];
ld.shared.u32 r5891, [r6144+5632];
ld.shared.u32 r5941, [r6144+6144];
ld.shared.u32 r5991, [r6144+6656];
ld.shared.u32 r6041, [r6144+7168];
ld.shared.u32 r6091, [r6144+7680];
ld.shared.u32 r5730, [r6144+8192];
ld.shared.u32 r5780, [r6144+8704];
ld.shared.u32 r5830, [r6144+9216];
ld.shared.u32 r5880, [r6144+9728];
ld.shared.u32 r5930, [r6144+10240];
ld.shared.u32 r5980, [r6144+10752];
ld.shared.u32 r6030, [r6144+11264];
ld.shared.u32 r6080, [r6144+11776];
ld.shared.u32 r5742, [r6144+12288];
ld.shared.u32 r5792, [r6144+12800];
ld.shared.u32 r5842, [r6144+13312];
ld.shared.u32 r5892, [r6144+13824];
ld.shared.u32 r5942, [r6144+14336];
ld.shared.u32 r5992, [r6144+14848];
ld.shared.u32 r6042, [r6144+15360];
ld.shared.u32 r6092, [r6144+15872];
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
neg.f16x2 r5749, r5743;
}
{
add.f16x2 %0, r5725, r5737;
}
{
add.f16x2 %1, r5728, r5740;
}
{
sub.f16x2 %32, r5725, r5737;
}
{
sub.f16x2 %33, r5728, r5740;
}
{
add.f16x2 %16, r5731, r5746;
}
{
add.f16x2 %17, r5734, r5749;
}
{
sub.f16x2 %48, r5731, r5746;
}
{
sub.f16x2 %49, r5734, r5749;
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
neg.f16x2 r5799, r5793;
}
{
add.f16x2 %2, r5775, r5787;
}
{
add.f16x2 %3, r5778, r5790;
}
{
sub.f16x2 %34, r5775, r5787;
}
{
sub.f16x2 %35, r5778, r5790;
}
{
add.f16x2 %18, r5781, r5796;
}
{
add.f16x2 %19, r5784, r5799;
}
{
sub.f16x2 %50, r5781, r5796;
}
{
sub.f16x2 %51, r5784, r5799;
}
{
add.f16x2 r5825, r5826, r5827;
}
{
add.f16x2 r5828, r5829, r5830;
}
{
sub.f16x2 r5831, r5826, r5827;
}
{
sub.f16x2 r5834, r5829, r5830;
}
{
add.f16x2 r5837, r5838, r5839;
}
{
add.f16x2 r5840, r5841, r5842;
}
{
sub.f16x2 r5843, r5838, r5839;
}
{
sub.f16x2 r5846, r5841, r5842;
}
{
neg.f16x2 r5849, r5843;
}
{
add.f16x2 %4, r5825, r5837;
}
{
add.f16x2 %5, r5828, r5840;
}
{
sub.f16x2 %36, r5825, r5837;
}
{
sub.f16x2 %37, r5828, r5840;
}
{
add.f16x2 %20, r5831, r5846;
}
{
add.f16x2 %21, r5834, r5849;
}
{
sub.f16x2 %52, r5831, r5846;
}
{
sub.f16x2 %53, r5834, r5849;
}
{
add.f16x2 r5875, r5876, r5877;
}
{
add.f16x2 r5878, r5879, r5880;
}
{
sub.f16x2 r5881, r5876, r5877;
}
{
sub.f16x2 r5884, r5879, r5880;
}
{
add.f16x2 r5887, r5888, r5889;
}
{
add.f16x2 r5890, r5891, r5892;
}
{
sub.f16x2 r5893, r5888, r5889;
}
{
sub.f16x2 r5896, r5891, r5892;
}
{
neg.f16x2 r5899, r5893;
}
{
add.f16x2 %6, r5875, r5887;
}
{
add.f16x2 %7, r5878, r5890;
}
{
sub.f16x2 %38, r5875, r5887;
}
{
sub.f16x2 %39, r5878, r5890;
}
{
add.f16x2 %22, r5881, r5896;
}
{
add.f16x2 %23, r5884, r5899;
}
{
sub.f16x2 %54, r5881, r5896;
}
{
sub.f16x2 %55, r5884, r5899;
}
{
add.f16x2 r5925, r5926, r5927;
}
{
add.f16x2 r5928, r5929, r5930;
}
{
sub.f16x2 r5931, r5926, r5927;
}
{
sub.f16x2 r5934, r5929, r5930;
}
{
add.f16x2 r5937, r5938, r5939;
}
{
add.f16x2 r5940, r5941, r5942;
}
{
sub.f16x2 r5943, r5938, r5939;
}
{
sub.f16x2 r5946, r5941, r5942;
}
{
neg.f16x2 r5949, r5943;
}
{
add.f16x2 %8, r5925, r5937;
}
{
add.f16x2 %9, r5928, r5940;
}
{
sub.f16x2 %40, r5925, r5937;
}
{
sub.f16x2 %41, r5928, r5940;
}
{
add.f16x2 %24, r5931, r5946;
}
{
add.f16x2 %25, r5934, r5949;
}
{
sub.f16x2 %56, r5931, r5946;
}
{
sub.f16x2 %57, r5934, r5949;
}
{
add.f16x2 r5975, r5976, r5977;
}
{
add.f16x2 r5978, r5979, r5980;
}
{
sub.f16x2 r5981, r5976, r5977;
}
{
sub.f16x2 r5984, r5979, r5980;
}
{
add.f16x2 r5987, r5988, r5989;
}
{
add.f16x2 r5990, r5991, r5992;
}
{
sub.f16x2 r5993, r5988, r5989;
}
{
sub.f16x2 r5996, r5991, r5992;
}
{
neg.f16x2 r5999, r5993;
}
{
add.f16x2 %10, r5975, r5987;
}
{
add.f16x2 %11, r5978, r5990;
}
{
sub.f16x2 %42, r5975, r5987;
}
{
sub.f16x2 %43, r5978, r5990;
}
{
add.f16x2 %26, r5981, r5996;
}
{
add.f16x2 %27, r5984, r5999;
}
{
sub.f16x2 %58, r5981, r5996;
}
{
sub.f16x2 %59, r5984, r5999;
}
{
add.f16x2 r6025, r6026, r6027;
}
{
add.f16x2 r6028, r6029, r6030;
}
{
sub.f16x2 r6031, r6026, r6027;
}
{
sub.f16x2 r6034, r6029, r6030;
}
{
add.f16x2 r6037, r6038, r6039;
}
{
add.f16x2 r6040, r6041, r6042;
}
{
sub.f16x2 r6043, r6038, r6039;
}
{
sub.f16x2 r6046, r6041, r6042;
}
{
neg.f16x2 r6049, r6043;
}
{
add.f16x2 %12, r6025, r6037;
}
{
add.f16x2 %13, r6028, r6040;
}
{
sub.f16x2 %44, r6025, r6037;
}
{
sub.f16x2 %45, r6028, r6040;
}
{
add.f16x2 %28, r6031, r6046;
}
{
add.f16x2 %29, r6034, r6049;
}
{
sub.f16x2 %60, r6031, r6046;
}
{
sub.f16x2 %61, r6034, r6049;
}
{
add.f16x2 r6075, r6076, r6077;
}
{
add.f16x2 r6078, r6079, r6080;
}
{
sub.f16x2 r6081, r6076, r6077;
}
{
sub.f16x2 r6084, r6079, r6080;
}
{
add.f16x2 r6087, r6088, r6089;
}
{
add.f16x2 r6090, r6091, r6092;
}
{
sub.f16x2 r6093, r6088, r6089;
}
{
sub.f16x2 r6096, r6091, r6092;
}
{
neg.f16x2 r6099, r6093;
}
{
add.f16x2 %14, r6075, r6087;
}
{
add.f16x2 %15, r6078, r6090;
}
{
sub.f16x2 %46, r6075, r6087;
}
{
sub.f16x2 %47, r6078, r6090;
}
{
add.f16x2 %30, r6081, r6096;
}
{
add.f16x2 %31, r6084, r6099;
}
{
sub.f16x2 %62, r6081, r6096;
}
{
sub.f16x2 %63, r6084, r6099;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)), "=r"(__HALF2_TO_UI(rmem[27].x)), "=r"(__HALF2_TO_UI(rmem[27].y)), "=r"(__HALF2_TO_UI(rmem[28].x)), "=r"(__HALF2_TO_UI(rmem[28].y)), "=r"(__HALF2_TO_UI(rmem[29].x)), "=r"(__HALF2_TO_UI(rmem[29].y)), "=r"(__HALF2_TO_UI(rmem[30].x)), "=r"(__HALF2_TO_UI(rmem[30].y)), "=r"(__HALF2_TO_UI(rmem[31].x)), "=r"(__HALF2_TO_UI(rmem[31].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[31].y)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[28].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[27].y)), "r"(__HALF2_TO_UI(rmem[30].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[29].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[31].x)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[28].x)), "r"(__HALF2_TO_UI(rmem[27].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[30].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[29].x)), "r"(__HALF2_TO_UI(rmem[24].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<853, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .pred p<3>;
.reg .f32 f<74>;
.reg .b32 r<930>;
.reg .b64 rd<2>;
{
add.f16x2 r22, %9, %13;
}
{
add.f16x2 r25, %10, %14;
}
{
sub.f16x2 r28, %9, %13;
}
{
sub.f16x2 r31, %10, %14;
}
{
add.f16x2 r34, %11, %15;
}
{
add.f16x2 r37, %12, %16;
}
{
sub.f16x2 r40, %11, %15;
}
{
sub.f16x2 r43, %12, %16;
}
{
neg.f16x2 r46, r40;
}
{
add.f16x2 r48, r22, r34;
}
{
add.f16x2 r51, r25, r37;
}
{
sub.f16x2 r54, r22, r34;
}
{
sub.f16x2 r57, r25, r37;
}
{
add.f16x2 r60, r28, r43;
}
{
add.f16x2 r63, r31, r46;
}
{
sub.f16x2 r66, r28, r43;
}
{
sub.f16x2 r69, r31, r46;
}
mov.u32 r9, %tid.x;
and.b32 r10, r9, 1023;
cvt.rn.f32.u32 f8, r10;
mul.f32 f1, f8, 0f3AC90FDB;
setp.eq.s32 p1, r10, 1020;
mov.f32 f73, 0f3BC90F88;
mov.f32 f72, f73;
@p1 bra LBB6_2;
cos.approx.f32 f72, f1;
LBB6_2:
mov.u32 r236, %tid.y;
shl.b32 r237, r236, 15;
mov.u32 r238, %8;
add.s32 r239, r238, r237;
shl.b32 r240, r9, 5;
and.b32 r241, r240, -32768;
add.s32 r12, r239, r241;
sin.approx.f32 f20, f1;
neg.f32 f10, f20;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f72;
cvt.rn.f16.f32 high, f10;
mov.b32 r72, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r72;
mov.b32 r75, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r72;
mov.b32 r77, {high, high};
}
{
mul.f16x2 r79, r63, r77;
}
{
neg.f16x2 r82, r79;
}
{
fma.rn.f16x2 r84, r60, r75, r82;
}
{
mul.f16x2 r88, r60, r77;
}
{
fma.rn.f16x2 r91, r63, r75, r88;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r72;
mov.b32 r95, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r72;
mov.b32 r97, {high, high};
}
mov.f32 f15, 0fBF800000;
mov.f32 f16, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f15;
cvt.rn.f16.f32 high, f16;
mov.b32 r99, {low, high};
}
{
mul.f16x2 r100, r97, r99;
}
{
mul.f16x2 r103, r72, r95;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r72;
mov.b32 r106, {high, low};
}
{
fma.rn.f16x2 r108, r100, r106, r103;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r108;
mov.b32 r112, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r108;
mov.b32 r114, {high, high};
}
{
mul.f16x2 r116, r57, r114;
}
{
neg.f16x2 r119, r116;
}
{
fma.rn.f16x2 r121, r54, r112, r119;
}
{
mul.f16x2 r125, r54, r114;
}
{
fma.rn.f16x2 r128, r57, r112, r125;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r72;
mov.b32 r132, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r72;
mov.b32 r134, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f15;
cvt.rn.f16.f32 high, f16;
mov.b32 r136, {low, high};
}
{
mul.f16x2 r137, r134, r136;
}
{
mul.f16x2 r140, r108, r132;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r108;
mov.b32 r143, {high, low};
}
{
fma.rn.f16x2 r145, r137, r143, r140;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r145;
mov.b32 r149, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r145;
mov.b32 r151, {high, high};
}
{
mul.f16x2 r153, r69, r151;
}
{
neg.f16x2 r156, r153;
}
{
fma.rn.f16x2 r158, r66, r149, r156;
}
{
mul.f16x2 r162, r66, r151;
}
{
fma.rn.f16x2 r165, r69, r149, r162;
}
barrier.sync 0;
shl.b32 r242, r10, 5;
add.s32 r243, r12, r242;
st.shared.v4.f32 [r243], {r48, r51, r84, r91};
st.shared.v4.f32 [r243+16], {r121, r128, r158, r165};
barrier.sync 0;
mad.lo.s32 r244, r10, -24, r243;
ld.shared.u32 r187, [r244];
ld.shared.u32 r190, [r244+4];
ld.shared.u32 r199, [r244+8192];
ld.shared.u32 r202, [r244+8196];
ld.shared.u32 r188, [r244+16384];
ld.shared.u32 r191, [r244+16388];
ld.shared.u32 r200, [r244+24576];
ld.shared.u32 r203, [r244+24580];
{
add.f16x2 r186, r187, r188;
}
{
add.f16x2 r189, r190, r191;
}
{
sub.f16x2 r192, r187, r188;
}
{
sub.f16x2 r195, r190, r191;
}
{
add.f16x2 r198, r199, r200;
}
{
add.f16x2 r201, r202, r203;
}
{
sub.f16x2 r204, r199, r200;
}
{
sub.f16x2 r207, r202, r203;
}
{
neg.f16x2 r210, r204;
}
{
add.f16x2 r212, r186, r198;
}
{
add.f16x2 r215, r189, r201;
}
{
sub.f16x2 r218, r186, r198;
}
{
sub.f16x2 r221, r189, r201;
}
{
add.f16x2 r224, r192, r207;
}
{
add.f16x2 r227, r195, r210;
}
{
sub.f16x2 r230, r192, r207;
}
{
sub.f16x2 r233, r195, r210;
}
and.b32 r21, r9, 1020;
bfe.u32 r245, r9, 2, 8;
cvt.rn.f32.u32 f21, r245;
mul.f32 f4, f21, 0f3BC90FDB;
setp.eq.s32 p2, r21, 1020;
@p2 bra LBB6_4;
cos.approx.f32 f73, f4;
LBB6_4:
sin.approx.f32 f62, f4;
neg.f32 f23, f62;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f23;
mov.b32 r246, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r246;
mov.b32 r249, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r246;
mov.b32 r251, {high, high};
}
{
mul.f16x2 r253, r227, r251;
}
{
neg.f16x2 r256, r253;
}
{
fma.rn.f16x2 r258, r224, r249, r256;
}
{
mul.f16x2 r262, r224, r251;
}
{
fma.rn.f16x2 r265, r227, r249, r262;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r246;
mov.b32 r269, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r246;
mov.b32 r271, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f15;
cvt.rn.f16.f32 high, f16;
mov.b32 r273, {low, high};
}
{
mul.f16x2 r274, r271, r273;
}
{
mul.f16x2 r277, r246, r269;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r246;
mov.b32 r280, {high, low};
}
{
fma.rn.f16x2 r282, r274, r280, r277;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r282;
mov.b32 r286, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r282;
mov.b32 r288, {high, high};
}
{
mul.f16x2 r290, r221, r288;
}
{
neg.f16x2 r293, r290;
}
{
fma.rn.f16x2 r295, r218, r286, r293;
}
{
mul.f16x2 r299, r218, r288;
}
{
fma.rn.f16x2 r302, r221, r286, r299;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r246;
mov.b32 r306, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r246;
mov.b32 r308, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f15;
cvt.rn.f16.f32 high, f16;
mov.b32 r310, {low, high};
}
{
mul.f16x2 r311, r308, r310;
}
{
mul.f16x2 r314, r282, r306;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r282;
mov.b32 r317, {high, low};
}
{
fma.rn.f16x2 r319, r311, r317, r314;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r319;
mov.b32 r323, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r319;
mov.b32 r325, {high, high};
}
{
mul.f16x2 r327, r233, r325;
}
{
neg.f16x2 r330, r327;
}
{
fma.rn.f16x2 r332, r230, r323, r330;
}
{
mul.f16x2 r336, r230, r325;
}
{
fma.rn.f16x2 r339, r233, r323, r336;
}
barrier.sync 0;
shl.b32 r902, r9, 3;
and.b32 r903, r902, 24;
add.s32 r904, r12, r903;
shl.b32 r905, r21, 5;
add.s32 r906, r904, r905;
st.shared.u32 [r906], r212;
st.shared.u32 [r906+4], r215;
st.shared.u32 [r906+32], r258;
st.shared.u32 [r906+36], r265;
st.shared.u32 [r906+64], r295;
st.shared.u32 [r906+68], r302;
st.shared.u32 [r906+96], r332;
st.shared.u32 [r906+100], r339;
barrier.sync 0;
mad.lo.s32 r907, r21, -24, r906;
ld.shared.u32 r361, [r907];
ld.shared.u32 r364, [r907+4];
ld.shared.u32 r373, [r907+8192];
ld.shared.u32 r376, [r907+8196];
ld.shared.u32 r362, [r907+16384];
ld.shared.u32 r365, [r907+16388];
ld.shared.u32 r374, [r907+24576];
ld.shared.u32 r377, [r907+24580];
{
add.f16x2 r360, r361, r362;
}
{
add.f16x2 r363, r364, r365;
}
{
sub.f16x2 r366, r361, r362;
}
{
sub.f16x2 r369, r364, r365;
}
{
add.f16x2 r372, r373, r374;
}
{
add.f16x2 r375, r376, r377;
}
{
sub.f16x2 r378, r373, r374;
}
{
sub.f16x2 r381, r376, r377;
}
{
neg.f16x2 r384, r378;
}
{
add.f16x2 r386, r360, r372;
}
{
add.f16x2 r389, r363, r375;
}
{
sub.f16x2 r392, r360, r372;
}
{
sub.f16x2 r395, r363, r375;
}
{
add.f16x2 r398, r366, r381;
}
{
add.f16x2 r401, r369, r384;
}
{
sub.f16x2 r404, r366, r381;
}
{
sub.f16x2 r407, r369, r384;
}
and.b32 r908, r9, 1008;
bfe.u32 r909, r9, 4, 6;
cvt.rn.f32.u32 f63, r909;
mul.f32 f64, f63, 0f3CC90FDB;
cos.approx.f32 f32, f64;
sin.approx.f32 f65, f64;
neg.f32 f33, f65;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f32;
cvt.rn.f16.f32 high, f33;
mov.b32 r410, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r410;
mov.b32 r413, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r410;
mov.b32 r415, {high, high};
}
{
mul.f16x2 r417, r401, r415;
}
{
neg.f16x2 r420, r417;
}
{
fma.rn.f16x2 r422, r398, r413, r420;
}
{
mul.f16x2 r426, r398, r415;
}
{
fma.rn.f16x2 r429, r401, r413, r426;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r410;
mov.b32 r433, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r410;
mov.b32 r435, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f15;
cvt.rn.f16.f32 high, f16;
mov.b32 r437, {low, high};
}
{
mul.f16x2 r438, r435, r437;
}
{
mul.f16x2 r441, r410, r433;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r410;
mov.b32 r444, {high, low};
}
{
fma.rn.f16x2 r446, r438, r444, r441;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r446;
mov.b32 r450, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r446;
mov.b32 r452, {high, high};
}
{
mul.f16x2 r454, r395, r452;
}
{
neg.f16x2 r457, r454;
}
{
fma.rn.f16x2 r459, r392, r450, r457;
}
{
mul.f16x2 r463, r392, r452;
}
{
fma.rn.f16x2 r466, r395, r450, r463;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r410;
mov.b32 r470, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r410;
mov.b32 r472, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f15;
cvt.rn.f16.f32 high, f16;
mov.b32 r474, {low, high};
}
{
mul.f16x2 r475, r472, r474;
}
{
mul.f16x2 r478, r446, r470;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r446;
mov.b32 r481, {high, low};
}
{
fma.rn.f16x2 r483, r475, r481, r478;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r483;
mov.b32 r487, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r483;
mov.b32 r489, {high, high};
}
{
mul.f16x2 r491, r407, r489;
}
{
neg.f16x2 r494, r491;
}
{
fma.rn.f16x2 r496, r404, r487, r494;
}
{
mul.f16x2 r500, r404, r489;
}
{
fma.rn.f16x2 r503, r407, r487, r500;
}
and.b32 r910, r902, 120;
add.s32 r911, r12, r910;
barrier.sync 0;
and.b32 r913, r240, 32256;
add.s32 r914, r911, r913;
st.shared.u32 [r914], r386;
st.shared.u32 [r914+4], r389;
st.shared.u32 [r914+128], r422;
st.shared.u32 [r914+132], r429;
st.shared.u32 [r914+256], r459;
st.shared.u32 [r914+260], r466;
st.shared.u32 [r914+384], r496;
st.shared.u32 [r914+388], r503;
barrier.sync 0;
mad.lo.s32 r915, r908, -24, r914;
ld.shared.u32 r525, [r915];
ld.shared.u32 r528, [r915+4];
ld.shared.u32 r537, [r915+8192];
ld.shared.u32 r540, [r915+8196];
ld.shared.u32 r526, [r915+16384];
ld.shared.u32 r529, [r915+16388];
ld.shared.u32 r538, [r915+24576];
ld.shared.u32 r541, [r915+24580];
{
add.f16x2 r524, r525, r526;
}
{
add.f16x2 r527, r528, r529;
}
{
sub.f16x2 r530, r525, r526;
}
{
sub.f16x2 r533, r528, r529;
}
{
add.f16x2 r536, r537, r538;
}
{
add.f16x2 r539, r540, r541;
}
{
sub.f16x2 r542, r537, r538;
}
{
sub.f16x2 r545, r540, r541;
}
{
neg.f16x2 r548, r542;
}
{
add.f16x2 r550, r524, r536;
}
{
add.f16x2 r553, r527, r539;
}
{
sub.f16x2 r556, r524, r536;
}
{
sub.f16x2 r559, r527, r539;
}
{
add.f16x2 r562, r530, r545;
}
{
add.f16x2 r565, r533, r548;
}
{
sub.f16x2 r568, r530, r545;
}
{
sub.f16x2 r571, r533, r548;
}
and.b32 r916, r9, 960;
bfe.u32 r917, r9, 6, 4;
cvt.rn.f32.u32 f66, r917;
mul.f32 f67, f66, 0f3DC90FDB;
cos.approx.f32 f42, f67;
sin.approx.f32 f68, f67;
neg.f32 f43, f68;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f42;
cvt.rn.f16.f32 high, f43;
mov.b32 r574, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r574;
mov.b32 r577, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r574;
mov.b32 r579, {high, high};
}
{
mul.f16x2 r581, r565, r579;
}
{
neg.f16x2 r584, r581;
}
{
fma.rn.f16x2 r586, r562, r577, r584;
}
{
mul.f16x2 r590, r562, r579;
}
{
fma.rn.f16x2 r593, r565, r577, r590;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r574;
mov.b32 r597, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r574;
mov.b32 r599, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f15;
cvt.rn.f16.f32 high, f16;
mov.b32 r601, {low, high};
}
{
mul.f16x2 r602, r599, r601;
}
{
mul.f16x2 r605, r574, r597;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r574;
mov.b32 r608, {high, low};
}
{
fma.rn.f16x2 r610, r602, r608, r605;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r610;
mov.b32 r614, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r610;
mov.b32 r616, {high, high};
}
{
mul.f16x2 r618, r559, r616;
}
{
neg.f16x2 r621, r618;
}
{
fma.rn.f16x2 r623, r556, r614, r621;
}
{
mul.f16x2 r627, r556, r616;
}
{
fma.rn.f16x2 r630, r559, r614, r627;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r574;
mov.b32 r634, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r574;
mov.b32 r636, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f15;
cvt.rn.f16.f32 high, f16;
mov.b32 r638, {low, high};
}
{
mul.f16x2 r639, r636, r638;
}
{
mul.f16x2 r642, r610, r634;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r610;
mov.b32 r645, {high, low};
}
{
fma.rn.f16x2 r647, r639, r645, r642;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r647;
mov.b32 r651, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r647;
mov.b32 r653, {high, high};
}
{
mul.f16x2 r655, r571, r653;
}
{
neg.f16x2 r658, r655;
}
{
fma.rn.f16x2 r660, r568, r651, r658;
}
{
mul.f16x2 r664, r568, r653;
}
{
fma.rn.f16x2 r667, r571, r651, r664;
}
and.b32 r918, r902, 504;
add.s32 r919, r12, r918;
barrier.sync 0;
and.b32 r920, r240, 30720;
add.s32 r921, r919, r920;
st.shared.u32 [r921], r550;
st.shared.u32 [r921+4], r553;
st.shared.u32 [r921+512], r586;
st.shared.u32 [r921+516], r593;
st.shared.u32 [r921+1024], r623;
st.shared.u32 [r921+1028], r630;
st.shared.u32 [r921+1536], r660;
st.shared.u32 [r921+1540], r667;
barrier.sync 0;
mad.lo.s32 r922, r916, -24, r921;
ld.shared.u32 r689, [r922];
ld.shared.u32 r692, [r922+4];
ld.shared.u32 r701, [r922+8192];
ld.shared.u32 r704, [r922+8196];
ld.shared.u32 r690, [r922+16384];
ld.shared.u32 r693, [r922+16388];
ld.shared.u32 r702, [r922+24576];
ld.shared.u32 r705, [r922+24580];
{
add.f16x2 r688, r689, r690;
}
{
add.f16x2 r691, r692, r693;
}
{
sub.f16x2 r694, r689, r690;
}
{
sub.f16x2 r697, r692, r693;
}
{
add.f16x2 r700, r701, r702;
}
{
add.f16x2 r703, r704, r705;
}
{
sub.f16x2 r706, r701, r702;
}
{
sub.f16x2 r709, r704, r705;
}
{
neg.f16x2 r712, r706;
}
{
add.f16x2 r714, r688, r700;
}
{
add.f16x2 r717, r691, r703;
}
{
sub.f16x2 r720, r688, r700;
}
{
sub.f16x2 r723, r691, r703;
}
{
add.f16x2 r726, r694, r709;
}
{
add.f16x2 r729, r697, r712;
}
{
sub.f16x2 r732, r694, r709;
}
{
sub.f16x2 r735, r697, r712;
}
and.b32 r923, r9, 768;
bfe.u32 r924, r9, 8, 2;
cvt.rn.f32.u32 f69, r924;
mul.f32 f70, f69, 0f3EC90FDB;
cos.approx.f32 f52, f70;
sin.approx.f32 f71, f70;
neg.f32 f53, f71;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f53;
mov.b32 r738, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r738;
mov.b32 r741, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r738;
mov.b32 r743, {high, high};
}
{
mul.f16x2 r745, r729, r743;
}
{
neg.f16x2 r748, r745;
}
{
fma.rn.f16x2 r750, r726, r741, r748;
}
{
mul.f16x2 r754, r726, r743;
}
{
fma.rn.f16x2 r757, r729, r741, r754;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r738;
mov.b32 r761, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r738;
mov.b32 r763, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f15;
cvt.rn.f16.f32 high, f16;
mov.b32 r765, {low, high};
}
{
mul.f16x2 r766, r763, r765;
}
{
mul.f16x2 r769, r738, r761;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r738;
mov.b32 r772, {high, low};
}
{
fma.rn.f16x2 r774, r766, r772, r769;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r774;
mov.b32 r778, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r774;
mov.b32 r780, {high, high};
}
{
mul.f16x2 r782, r723, r780;
}
{
neg.f16x2 r785, r782;
}
{
fma.rn.f16x2 r787, r720, r778, r785;
}
{
mul.f16x2 r791, r720, r780;
}
{
fma.rn.f16x2 r794, r723, r778, r791;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r738;
mov.b32 r798, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r738;
mov.b32 r800, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f15;
cvt.rn.f16.f32 high, f16;
mov.b32 r802, {low, high};
}
{
mul.f16x2 r803, r800, r802;
}
{
mul.f16x2 r806, r774, r798;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r774;
mov.b32 r809, {high, low};
}
{
fma.rn.f16x2 r811, r803, r809, r806;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r811;
mov.b32 r815, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r811;
mov.b32 r817, {high, high};
}
{
mul.f16x2 r819, r735, r817;
}
{
neg.f16x2 r822, r819;
}
{
fma.rn.f16x2 r824, r732, r815, r822;
}
{
mul.f16x2 r828, r732, r817;
}
{
fma.rn.f16x2 r831, r735, r815, r828;
}
and.b32 r925, r902, 2040;
add.s32 r926, r12, r925;
barrier.sync 0;
and.b32 r927, r240, 24576;
add.s32 r928, r926, r927;
st.shared.u32 [r928], r714;
st.shared.u32 [r928+4], r717;
st.shared.u32 [r928+2048], r750;
st.shared.u32 [r928+2052], r757;
st.shared.u32 [r928+4096], r787;
st.shared.u32 [r928+4100], r794;
st.shared.u32 [r928+6144], r824;
st.shared.u32 [r928+6148], r831;
barrier.sync 0;
mad.lo.s32 r929, r923, -24, r928;
ld.shared.u32 r853, [r929];
ld.shared.u32 r856, [r929+4];
ld.shared.u32 r865, [r929+8192];
ld.shared.u32 r868, [r929+8196];
ld.shared.u32 r854, [r929+16384];
ld.shared.u32 r857, [r929+16388];
ld.shared.u32 r866, [r929+24576];
ld.shared.u32 r869, [r929+24580];
{
add.f16x2 r852, r853, r854;
}
{
add.f16x2 r855, r856, r857;
}
{
sub.f16x2 r858, r853, r854;
}
{
sub.f16x2 r861, r856, r857;
}
{
add.f16x2 r864, r865, r866;
}
{
add.f16x2 r867, r868, r869;
}
{
sub.f16x2 r870, r865, r866;
}
{
sub.f16x2 r873, r868, r869;
}
{
neg.f16x2 r876, r870;
}
{
add.f16x2 %0, r852, r864;
}
{
add.f16x2 %1, r855, r867;
}
{
sub.f16x2 %4, r852, r864;
}
{
sub.f16x2 %5, r855, r867;
}
{
add.f16x2 %2, r858, r873;
}
{
add.f16x2 %3, r861, r876;
}
{
sub.f16x2 %6, r858, r873;
}
{
sub.f16x2 %7, r861, r876;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<854, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .pred p<3>;
.reg .f32 f<74>;
.reg .b32 r<929>;
.reg .b64 rd<2>;
{
add.f16x2 r21, %9, %13;
}
{
add.f16x2 r24, %10, %14;
}
{
sub.f16x2 r27, %9, %13;
}
{
sub.f16x2 r30, %10, %14;
}
{
add.f16x2 r33, %11, %15;
}
{
add.f16x2 r36, %12, %16;
}
{
sub.f16x2 r39, %11, %15;
}
{
sub.f16x2 r42, %12, %16;
}
{
neg.f16x2 r45, r39;
}
{
add.f16x2 r47, r21, r33;
}
{
add.f16x2 r50, r24, r36;
}
{
sub.f16x2 r53, r21, r33;
}
{
sub.f16x2 r56, r24, r36;
}
{
add.f16x2 r59, r27, r42;
}
{
add.f16x2 r62, r30, r45;
}
{
sub.f16x2 r65, r27, r42;
}
{
sub.f16x2 r68, r30, r45;
}
mov.u32 r9, %tid.x;
and.b32 r10, r9, 1023;
cvt.rn.f32.u32 f8, r10;
mul.f32 f1, f8, 0f3AC90FDB;
setp.eq.s32 p1, r10, 1020;
mov.f32 f73, 0f3BC90F88;
mov.f32 f72, f73;
@p1 bra LBB7_2;
cos.approx.f32 f72, f1;
LBB7_2:
mov.u32 r235, %tid.y;
shl.b32 r236, r235, 14;
mov.u32 r237, %8;
add.s32 r238, r237, r236;
shl.b32 r239, r9, 4;
and.b32 r240, r239, -16384;
add.s32 r11, r238, r240;
sin.approx.f32 f20, f1;
neg.f32 f10, f20;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f72;
cvt.rn.f16.f32 high, f10;
mov.b32 r71, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r71;
mov.b32 r74, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r71;
mov.b32 r76, {high, high};
}
{
mul.f16x2 r78, r62, r76;
}
{
neg.f16x2 r81, r78;
}
{
fma.rn.f16x2 r83, r59, r74, r81;
}
{
mul.f16x2 r87, r59, r76;
}
{
fma.rn.f16x2 r90, r62, r74, r87;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r71;
mov.b32 r94, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r71;
mov.b32 r96, {high, high};
}
mov.f32 f15, 0fBF800000;
mov.f32 f16, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f15;
cvt.rn.f16.f32 high, f16;
mov.b32 r98, {low, high};
}
{
mul.f16x2 r99, r96, r98;
}
{
mul.f16x2 r102, r71, r94;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r71;
mov.b32 r105, {high, low};
}
{
fma.rn.f16x2 r107, r99, r105, r102;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r107;
mov.b32 r111, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r107;
mov.b32 r113, {high, high};
}
{
mul.f16x2 r115, r56, r113;
}
{
neg.f16x2 r118, r115;
}
{
fma.rn.f16x2 r120, r53, r111, r118;
}
{
mul.f16x2 r124, r53, r113;
}
{
fma.rn.f16x2 r127, r56, r111, r124;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r71;
mov.b32 r131, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r71;
mov.b32 r133, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f15;
cvt.rn.f16.f32 high, f16;
mov.b32 r135, {low, high};
}
{
mul.f16x2 r136, r133, r135;
}
{
mul.f16x2 r139, r107, r131;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r107;
mov.b32 r142, {high, low};
}
{
fma.rn.f16x2 r144, r136, r142, r139;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r144;
mov.b32 r148, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r144;
mov.b32 r150, {high, high};
}
{
mul.f16x2 r152, r68, r150;
}
{
neg.f16x2 r155, r152;
}
{
fma.rn.f16x2 r157, r65, r148, r155;
}
{
mul.f16x2 r161, r65, r150;
}
{
fma.rn.f16x2 r164, r68, r148, r161;
}
barrier.sync 0;
shl.b32 r241, r10, 4;
add.s32 r242, r11, r241;
st.shared.v4.f32 [r242], {r47, r83, r120, r157};
barrier.sync 0;
mad.lo.s32 r243, r10, -12, r242;
ld.shared.u32 r186, [r243];
ld.shared.u32 r198, [r243+4096];
ld.shared.u32 r187, [r243+8192];
ld.shared.u32 r199, [r243+12288];
barrier.sync 0;
st.shared.v4.f32 [r242], {r50, r90, r127, r164};
barrier.sync 0;
ld.shared.u32 r189, [r243];
ld.shared.u32 r201, [r243+4096];
ld.shared.u32 r190, [r243+8192];
ld.shared.u32 r202, [r243+12288];
{
add.f16x2 r185, r186, r187;
}
{
add.f16x2 r188, r189, r190;
}
{
sub.f16x2 r191, r186, r187;
}
{
sub.f16x2 r194, r189, r190;
}
{
add.f16x2 r197, r198, r199;
}
{
add.f16x2 r200, r201, r202;
}
{
sub.f16x2 r203, r198, r199;
}
{
sub.f16x2 r206, r201, r202;
}
{
neg.f16x2 r209, r203;
}
{
add.f16x2 r211, r185, r197;
}
{
add.f16x2 r214, r188, r200;
}
{
sub.f16x2 r217, r185, r197;
}
{
sub.f16x2 r220, r188, r200;
}
{
add.f16x2 r223, r191, r206;
}
{
add.f16x2 r226, r194, r209;
}
{
sub.f16x2 r229, r191, r206;
}
{
sub.f16x2 r232, r194, r209;
}
and.b32 r20, r9, 1020;
bfe.u32 r244, r9, 2, 8;
cvt.rn.f32.u32 f21, r244;
mul.f32 f4, f21, 0f3BC90FDB;
setp.eq.s32 p2, r20, 1020;
@p2 bra LBB7_4;
cos.approx.f32 f73, f4;
LBB7_4:
sin.approx.f32 f62, f4;
neg.f32 f23, f62;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f23;
mov.b32 r245, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r245;
mov.b32 r248, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r245;
mov.b32 r250, {high, high};
}
{
mul.f16x2 r252, r226, r250;
}
{
neg.f16x2 r255, r252;
}
{
fma.rn.f16x2 r257, r223, r248, r255;
}
{
mul.f16x2 r261, r223, r250;
}
{
fma.rn.f16x2 r264, r226, r248, r261;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r245;
mov.b32 r268, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r245;
mov.b32 r270, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f15;
cvt.rn.f16.f32 high, f16;
mov.b32 r272, {low, high};
}
{
mul.f16x2 r273, r270, r272;
}
{
mul.f16x2 r276, r245, r268;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r245;
mov.b32 r279, {high, low};
}
{
fma.rn.f16x2 r281, r273, r279, r276;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r281;
mov.b32 r285, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r281;
mov.b32 r287, {high, high};
}
{
mul.f16x2 r289, r220, r287;
}
{
neg.f16x2 r292, r289;
}
{
fma.rn.f16x2 r294, r217, r285, r292;
}
{
mul.f16x2 r298, r217, r287;
}
{
fma.rn.f16x2 r301, r220, r285, r298;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r245;
mov.b32 r305, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r245;
mov.b32 r307, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f15;
cvt.rn.f16.f32 high, f16;
mov.b32 r309, {low, high};
}
{
mul.f16x2 r310, r307, r309;
}
{
mul.f16x2 r313, r281, r305;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r281;
mov.b32 r316, {high, low};
}
{
fma.rn.f16x2 r318, r310, r316, r313;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r318;
mov.b32 r322, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r318;
mov.b32 r324, {high, high};
}
{
mul.f16x2 r326, r232, r324;
}
{
neg.f16x2 r329, r326;
}
{
fma.rn.f16x2 r331, r229, r322, r329;
}
{
mul.f16x2 r335, r229, r324;
}
{
fma.rn.f16x2 r338, r232, r322, r335;
}
barrier.sync 0;
shl.b32 r901, r9, 2;
and.b32 r902, r901, 12;
add.s32 r903, r11, r902;
shl.b32 r904, r20, 4;
add.s32 r905, r903, r904;
st.shared.u32 [r905], r211;
st.shared.u32 [r905+16], r257;
st.shared.u32 [r905+32], r294;
st.shared.u32 [r905+48], r331;
barrier.sync 0;
mad.lo.s32 r906, r20, -12, r905;
ld.shared.u32 r360, [r906];
ld.shared.u32 r372, [r906+4096];
ld.shared.u32 r361, [r906+8192];
ld.shared.u32 r373, [r906+12288];
barrier.sync 0;
st.shared.u32 [r905], r214;
st.shared.u32 [r905+16], r264;
st.shared.u32 [r905+32], r301;
st.shared.u32 [r905+48], r338;
barrier.sync 0;
ld.shared.u32 r363, [r906];
ld.shared.u32 r375, [r906+4096];
ld.shared.u32 r364, [r906+8192];
ld.shared.u32 r376, [r906+12288];
{
add.f16x2 r359, r360, r361;
}
{
add.f16x2 r362, r363, r364;
}
{
sub.f16x2 r365, r360, r361;
}
{
sub.f16x2 r368, r363, r364;
}
{
add.f16x2 r371, r372, r373;
}
{
add.f16x2 r374, r375, r376;
}
{
sub.f16x2 r377, r372, r373;
}
{
sub.f16x2 r380, r375, r376;
}
{
neg.f16x2 r383, r377;
}
{
add.f16x2 r385, r359, r371;
}
{
add.f16x2 r388, r362, r374;
}
{
sub.f16x2 r391, r359, r371;
}
{
sub.f16x2 r394, r362, r374;
}
{
add.f16x2 r397, r365, r380;
}
{
add.f16x2 r400, r368, r383;
}
{
sub.f16x2 r403, r365, r380;
}
{
sub.f16x2 r406, r368, r383;
}
and.b32 r907, r9, 1008;
bfe.u32 r908, r9, 4, 6;
and.b32 r909, r901, 60;
add.s32 r910, r11, r909;
cvt.rn.f32.u32 f63, r908;
mul.f32 f64, f63, 0f3CC90FDB;
cos.approx.f32 f32, f64;
sin.approx.f32 f65, f64;
neg.f32 f33, f65;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f32;
cvt.rn.f16.f32 high, f33;
mov.b32 r409, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r409;
mov.b32 r412, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r409;
mov.b32 r414, {high, high};
}
{
mul.f16x2 r416, r400, r414;
}
{
neg.f16x2 r419, r416;
}
{
fma.rn.f16x2 r421, r397, r412, r419;
}
{
mul.f16x2 r425, r397, r414;
}
{
fma.rn.f16x2 r428, r400, r412, r425;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r409;
mov.b32 r432, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r409;
mov.b32 r434, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f15;
cvt.rn.f16.f32 high, f16;
mov.b32 r436, {low, high};
}
{
mul.f16x2 r437, r434, r436;
}
{
mul.f16x2 r440, r409, r432;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r409;
mov.b32 r443, {high, low};
}
{
fma.rn.f16x2 r445, r437, r443, r440;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r445;
mov.b32 r449, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r445;
mov.b32 r451, {high, high};
}
{
mul.f16x2 r453, r394, r451;
}
{
neg.f16x2 r456, r453;
}
{
fma.rn.f16x2 r458, r391, r449, r456;
}
{
mul.f16x2 r462, r391, r451;
}
{
fma.rn.f16x2 r465, r394, r449, r462;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r409;
mov.b32 r469, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r409;
mov.b32 r471, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f15;
cvt.rn.f16.f32 high, f16;
mov.b32 r473, {low, high};
}
{
mul.f16x2 r474, r471, r473;
}
{
mul.f16x2 r477, r445, r469;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r445;
mov.b32 r480, {high, low};
}
{
fma.rn.f16x2 r482, r474, r480, r477;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r482;
mov.b32 r486, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r482;
mov.b32 r488, {high, high};
}
{
mul.f16x2 r490, r406, r488;
}
{
neg.f16x2 r493, r490;
}
{
fma.rn.f16x2 r495, r403, r486, r493;
}
{
mul.f16x2 r499, r403, r488;
}
{
fma.rn.f16x2 r502, r406, r486, r499;
}
barrier.sync 0;
and.b32 r912, r239, 16128;
add.s32 r913, r910, r912;
st.shared.u32 [r913], r385;
st.shared.u32 [r913+64], r421;
st.shared.u32 [r913+128], r458;
st.shared.u32 [r913+192], r495;
barrier.sync 0;
mad.lo.s32 r914, r907, -12, r913;
ld.shared.u32 r524, [r914];
ld.shared.u32 r536, [r914+4096];
ld.shared.u32 r525, [r914+8192];
ld.shared.u32 r537, [r914+12288];
barrier.sync 0;
st.shared.u32 [r913], r388;
st.shared.u32 [r913+64], r428;
st.shared.u32 [r913+128], r465;
st.shared.u32 [r913+192], r502;
barrier.sync 0;
ld.shared.u32 r527, [r914];
ld.shared.u32 r539, [r914+4096];
ld.shared.u32 r528, [r914+8192];
ld.shared.u32 r540, [r914+12288];
{
add.f16x2 r523, r524, r525;
}
{
add.f16x2 r526, r527, r528;
}
{
sub.f16x2 r529, r524, r525;
}
{
sub.f16x2 r532, r527, r528;
}
{
add.f16x2 r535, r536, r537;
}
{
add.f16x2 r538, r539, r540;
}
{
sub.f16x2 r541, r536, r537;
}
{
sub.f16x2 r544, r539, r540;
}
{
neg.f16x2 r547, r541;
}
{
add.f16x2 r549, r523, r535;
}
{
add.f16x2 r552, r526, r538;
}
{
sub.f16x2 r555, r523, r535;
}
{
sub.f16x2 r558, r526, r538;
}
{
add.f16x2 r561, r529, r544;
}
{
add.f16x2 r564, r532, r547;
}
{
sub.f16x2 r567, r529, r544;
}
{
sub.f16x2 r570, r532, r547;
}
and.b32 r915, r9, 960;
bfe.u32 r916, r9, 6, 4;
and.b32 r917, r901, 252;
add.s32 r918, r11, r917;
cvt.rn.f32.u32 f66, r916;
mul.f32 f67, f66, 0f3DC90FDB;
cos.approx.f32 f42, f67;
sin.approx.f32 f68, f67;
neg.f32 f43, f68;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f42;
cvt.rn.f16.f32 high, f43;
mov.b32 r573, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r573;
mov.b32 r576, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r573;
mov.b32 r578, {high, high};
}
{
mul.f16x2 r580, r564, r578;
}
{
neg.f16x2 r583, r580;
}
{
fma.rn.f16x2 r585, r561, r576, r583;
}
{
mul.f16x2 r589, r561, r578;
}
{
fma.rn.f16x2 r592, r564, r576, r589;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r573;
mov.b32 r596, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r573;
mov.b32 r598, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f15;
cvt.rn.f16.f32 high, f16;
mov.b32 r600, {low, high};
}
{
mul.f16x2 r601, r598, r600;
}
{
mul.f16x2 r604, r573, r596;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r573;
mov.b32 r607, {high, low};
}
{
fma.rn.f16x2 r609, r601, r607, r604;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r613, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r615, {high, high};
}
{
mul.f16x2 r617, r558, r615;
}
{
neg.f16x2 r620, r617;
}
{
fma.rn.f16x2 r622, r555, r613, r620;
}
{
mul.f16x2 r626, r555, r615;
}
{
fma.rn.f16x2 r629, r558, r613, r626;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r573;
mov.b32 r633, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r573;
mov.b32 r635, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f15;
cvt.rn.f16.f32 high, f16;
mov.b32 r637, {low, high};
}
{
mul.f16x2 r638, r635, r637;
}
{
mul.f16x2 r641, r609, r633;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r644, {high, low};
}
{
fma.rn.f16x2 r646, r638, r644, r641;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r646;
mov.b32 r650, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r646;
mov.b32 r652, {high, high};
}
{
mul.f16x2 r654, r570, r652;
}
{
neg.f16x2 r657, r654;
}
{
fma.rn.f16x2 r659, r567, r650, r657;
}
{
mul.f16x2 r663, r567, r652;
}
{
fma.rn.f16x2 r666, r570, r650, r663;
}
barrier.sync 0;
and.b32 r919, r239, 15360;
add.s32 r920, r918, r919;
st.shared.u32 [r920], r549;
st.shared.u32 [r920+256], r585;
st.shared.u32 [r920+512], r622;
st.shared.u32 [r920+768], r659;
barrier.sync 0;
mad.lo.s32 r921, r915, -12, r920;
ld.shared.u32 r688, [r921];
ld.shared.u32 r700, [r921+4096];
ld.shared.u32 r689, [r921+8192];
ld.shared.u32 r701, [r921+12288];
barrier.sync 0;
st.shared.u32 [r920], r552;
st.shared.u32 [r920+256], r592;
st.shared.u32 [r920+512], r629;
st.shared.u32 [r920+768], r666;
barrier.sync 0;
ld.shared.u32 r691, [r921];
ld.shared.u32 r703, [r921+4096];
ld.shared.u32 r692, [r921+8192];
ld.shared.u32 r704, [r921+12288];
{
add.f16x2 r687, r688, r689;
}
{
add.f16x2 r690, r691, r692;
}
{
sub.f16x2 r693, r688, r689;
}
{
sub.f16x2 r696, r691, r692;
}
{
add.f16x2 r699, r700, r701;
}
{
add.f16x2 r702, r703, r704;
}
{
sub.f16x2 r705, r700, r701;
}
{
sub.f16x2 r708, r703, r704;
}
{
neg.f16x2 r711, r705;
}
{
add.f16x2 r713, r687, r699;
}
{
add.f16x2 r716, r690, r702;
}
{
sub.f16x2 r719, r687, r699;
}
{
sub.f16x2 r722, r690, r702;
}
{
add.f16x2 r725, r693, r708;
}
{
add.f16x2 r728, r696, r711;
}
{
sub.f16x2 r731, r693, r708;
}
{
sub.f16x2 r734, r696, r711;
}
and.b32 r922, r9, 768;
bfe.u32 r923, r9, 8, 2;
and.b32 r924, r901, 1020;
add.s32 r925, r11, r924;
cvt.rn.f32.u32 f69, r923;
mul.f32 f70, f69, 0f3EC90FDB;
cos.approx.f32 f52, f70;
sin.approx.f32 f71, f70;
neg.f32 f53, f71;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f53;
mov.b32 r737, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r737;
mov.b32 r740, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r737;
mov.b32 r742, {high, high};
}
{
mul.f16x2 r744, r728, r742;
}
{
neg.f16x2 r747, r744;
}
{
fma.rn.f16x2 r749, r725, r740, r747;
}
{
mul.f16x2 r753, r725, r742;
}
{
fma.rn.f16x2 r756, r728, r740, r753;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r737;
mov.b32 r760, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r737;
mov.b32 r762, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f15;
cvt.rn.f16.f32 high, f16;
mov.b32 r764, {low, high};
}
{
mul.f16x2 r765, r762, r764;
}
{
mul.f16x2 r768, r737, r760;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r737;
mov.b32 r771, {high, low};
}
{
fma.rn.f16x2 r773, r765, r771, r768;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r773;
mov.b32 r777, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r773;
mov.b32 r779, {high, high};
}
{
mul.f16x2 r781, r722, r779;
}
{
neg.f16x2 r784, r781;
}
{
fma.rn.f16x2 r786, r719, r777, r784;
}
{
mul.f16x2 r790, r719, r779;
}
{
fma.rn.f16x2 r793, r722, r777, r790;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r737;
mov.b32 r797, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r737;
mov.b32 r799, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f15;
cvt.rn.f16.f32 high, f16;
mov.b32 r801, {low, high};
}
{
mul.f16x2 r802, r799, r801;
}
{
mul.f16x2 r805, r773, r797;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r773;
mov.b32 r808, {high, low};
}
{
fma.rn.f16x2 r810, r802, r808, r805;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r810;
mov.b32 r814, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r810;
mov.b32 r816, {high, high};
}
{
mul.f16x2 r818, r734, r816;
}
{
neg.f16x2 r821, r818;
}
{
fma.rn.f16x2 r823, r731, r814, r821;
}
{
mul.f16x2 r827, r731, r816;
}
{
fma.rn.f16x2 r830, r734, r814, r827;
}
barrier.sync 0;
and.b32 r926, r239, 12288;
add.s32 r927, r925, r926;
st.shared.u32 [r927], r713;
st.shared.u32 [r927+1024], r749;
st.shared.u32 [r927+2048], r786;
st.shared.u32 [r927+3072], r823;
barrier.sync 0;
mad.lo.s32 r928, r922, -12, r927;
ld.shared.u32 r852, [r928];
ld.shared.u32 r864, [r928+4096];
ld.shared.u32 r853, [r928+8192];
ld.shared.u32 r865, [r928+12288];
barrier.sync 0;
st.shared.u32 [r927], r716;
st.shared.u32 [r927+1024], r756;
st.shared.u32 [r927+2048], r793;
st.shared.u32 [r927+3072], r830;
barrier.sync 0;
ld.shared.u32 r855, [r928];
ld.shared.u32 r867, [r928+4096];
ld.shared.u32 r856, [r928+8192];
ld.shared.u32 r868, [r928+12288];
{
add.f16x2 r851, r852, r853;
}
{
add.f16x2 r854, r855, r856;
}
{
sub.f16x2 r857, r852, r853;
}
{
sub.f16x2 r860, r855, r856;
}
{
add.f16x2 r863, r864, r865;
}
{
add.f16x2 r866, r867, r868;
}
{
sub.f16x2 r869, r864, r865;
}
{
sub.f16x2 r872, r867, r868;
}
{
neg.f16x2 r875, r869;
}
{
add.f16x2 %0, r851, r863;
}
{
add.f16x2 %1, r854, r866;
}
{
sub.f16x2 %4, r851, r863;
}
{
sub.f16x2 %5, r854, r866;
}
{
add.f16x2 %2, r857, r872;
}
{
add.f16x2 %3, r860, r875;
}
{
sub.f16x2 %6, r857, r872;
}
{
sub.f16x2 %7, r860, r875;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)));
};


#endif
