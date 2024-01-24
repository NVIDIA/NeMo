#ifndef CUFFTDX_FFT_2048_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_2048_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<837, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<363>;
.reg .b32 r<2761>;
.reg .b64 rd<2>;
mov.u32 r2741, %tid.y;
shl.b32 r2742, r2741, 13;
mov.u32 r2743, %32;
add.s32 r2744, r2743, r2742;
mov.u32 r2745, %tid.x;
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
mov.f32 f330, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r101, {low, high};
}
mov.f32 f340, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r102, {low, high};
}
mov.f32 f297, 0fBF800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
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
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
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
mov.f32 f208, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f208;
cvt.rn.f16.f32 high, f208;
mov.b32 r393, {low, high};
}
mov.f32 f234, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f234;
cvt.rn.f16.f32 high, f234;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r396, {low, high};
}
mov.f32 f216, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r397, {low, high};
}
mov.f32 f232, 0fBF6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f232;
cvt.rn.f16.f32 high, f232;
mov.b32 r398, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f234;
cvt.rn.f16.f32 high, f234;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f232;
cvt.rn.f16.f32 high, f232;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r404, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f232;
cvt.rn.f16.f32 high, f232;
mov.b32 r405, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f234;
cvt.rn.f16.f32 high, f234;
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
and.b32 r2746, r2745, 127;
shl.b32 r2747, r2745, 6;
and.b32 r2748, r2747, -8192;
add.s32 r2749, r2744, r2748;
cvt.rn.f32.u32 f357, r2746;
mul.f32 f358, f357, 0f3B490FDB;
cos.approx.f32 f117, f358;
sin.approx.f32 f359, f358;
neg.f32 f118, f359;
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
and.b32 r2750, r2747, 8128;
add.s32 r2751, r2749, r2750;
st.shared.v4.f32 [r2751], {r521, r629, r666, r703};
st.shared.v4.f32 [r2751+16], {r740, r777, r814, r851};
st.shared.v4.f32 [r2751+32], {r888, r925, r962, r999};
st.shared.v4.f32 [r2751+48], {r1036, r1073, r1110, r1147};
barrier.sync 0;
mad.lo.s32 r2752, r2746, -60, r2751;
ld.shared.u32 r1176, [r2752];
ld.shared.u32 r1372, [r2752+512];
ld.shared.u32 r1226, [r2752+1024];
ld.shared.u32 r1422, [r2752+1536];
ld.shared.u32 r1188, [r2752+2048];
ld.shared.u32 r1384, [r2752+2560];
ld.shared.u32 r1238, [r2752+3072];
ld.shared.u32 r1434, [r2752+3584];
ld.shared.u32 r1177, [r2752+4096];
ld.shared.u32 r1373, [r2752+4608];
ld.shared.u32 r1227, [r2752+5120];
ld.shared.u32 r1423, [r2752+5632];
ld.shared.u32 r1189, [r2752+6144];
ld.shared.u32 r1385, [r2752+6656];
ld.shared.u32 r1239, [r2752+7168];
ld.shared.u32 r1435, [r2752+7680];
barrier.sync 0;
st.shared.v4.f32 [r2751], {r524, r636, r673, r710};
st.shared.v4.f32 [r2751+16], {r747, r784, r821, r858};
st.shared.v4.f32 [r2751+32], {r895, r932, r969, r1006};
st.shared.v4.f32 [r2751+48], {r1043, r1080, r1117, r1154};
barrier.sync 0;
ld.shared.u32 r1179, [r2752];
ld.shared.u32 r1375, [r2752+512];
ld.shared.u32 r1229, [r2752+1024];
ld.shared.u32 r1425, [r2752+1536];
ld.shared.u32 r1191, [r2752+2048];
ld.shared.u32 r1387, [r2752+2560];
ld.shared.u32 r1241, [r2752+3072];
ld.shared.u32 r1437, [r2752+3584];
ld.shared.u32 r1180, [r2752+4096];
ld.shared.u32 r1376, [r2752+4608];
ld.shared.u32 r1230, [r2752+5120];
ld.shared.u32 r1426, [r2752+5632];
ld.shared.u32 r1192, [r2752+6144];
ld.shared.u32 r1388, [r2752+6656];
ld.shared.u32 r1242, [r2752+7168];
ld.shared.u32 r1438, [r2752+7680];
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
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r1275, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r1276, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r1279, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
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
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r1471, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r1472, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r1475, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
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
cvt.rn.f16.f32 low, f208;
cvt.rn.f16.f32 high, f208;
mov.b32 r1567, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f234;
cvt.rn.f16.f32 high, f234;
mov.b32 r1568, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r1569, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r1570, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1571, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f232;
cvt.rn.f16.f32 high, f232;
mov.b32 r1572, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f234;
cvt.rn.f16.f32 high, f234;
mov.b32 r1575, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f232;
cvt.rn.f16.f32 high, f232;
mov.b32 r1576, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r1577, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r1578, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f232;
cvt.rn.f16.f32 high, f232;
mov.b32 r1579, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f234;
cvt.rn.f16.f32 high, f234;
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
and.b32 r2753, r2745, 112;
bfe.u32 r2754, r2745, 4, 3;
shl.b32 r2755, r2745, 2;
and.b32 r2756, r2755, 60;
add.s32 r2757, r2749, r2756;
cvt.rn.f32.u32 f360, r2754;
mul.f32 f361, f360, 0f3D490FDB;
cos.approx.f32 f267, f361;
sin.approx.f32 f362, f361;
neg.f32 f268, f362;
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
and.b32 r2758, r2747, 7168;
add.s32 r2759, r2757, r2758;
st.shared.u32 [r2759], r1695;
st.shared.u32 [r2759+64], r1803;
st.shared.u32 [r2759+128], r1840;
st.shared.u32 [r2759+192], r1877;
st.shared.u32 [r2759+256], r1914;
st.shared.u32 [r2759+320], r1951;
st.shared.u32 [r2759+384], r1988;
st.shared.u32 [r2759+448], r2025;
st.shared.u32 [r2759+512], r2062;
st.shared.u32 [r2759+576], r2099;
st.shared.u32 [r2759+640], r2136;
st.shared.u32 [r2759+704], r2173;
st.shared.u32 [r2759+768], r2210;
st.shared.u32 [r2759+832], r2247;
st.shared.u32 [r2759+896], r2284;
st.shared.u32 [r2759+960], r2321;
barrier.sync 0;
mad.lo.s32 r2760, r2753, -60, r2759;
ld.shared.u32 r2350, [r2760];
ld.shared.u32 r2546, [r2760+512];
ld.shared.u32 r2400, [r2760+1024];
ld.shared.u32 r2596, [r2760+1536];
ld.shared.u32 r2362, [r2760+2048];
ld.shared.u32 r2558, [r2760+2560];
ld.shared.u32 r2412, [r2760+3072];
ld.shared.u32 r2608, [r2760+3584];
ld.shared.u32 r2351, [r2760+4096];
ld.shared.u32 r2547, [r2760+4608];
ld.shared.u32 r2401, [r2760+5120];
ld.shared.u32 r2597, [r2760+5632];
ld.shared.u32 r2363, [r2760+6144];
ld.shared.u32 r2559, [r2760+6656];
ld.shared.u32 r2413, [r2760+7168];
ld.shared.u32 r2609, [r2760+7680];
barrier.sync 0;
st.shared.u32 [r2759], r1698;
st.shared.u32 [r2759+64], r1810;
st.shared.u32 [r2759+128], r1847;
st.shared.u32 [r2759+192], r1884;
st.shared.u32 [r2759+256], r1921;
st.shared.u32 [r2759+320], r1958;
st.shared.u32 [r2759+384], r1995;
st.shared.u32 [r2759+448], r2032;
st.shared.u32 [r2759+512], r2069;
st.shared.u32 [r2759+576], r2106;
st.shared.u32 [r2759+640], r2143;
st.shared.u32 [r2759+704], r2180;
st.shared.u32 [r2759+768], r2217;
st.shared.u32 [r2759+832], r2254;
st.shared.u32 [r2759+896], r2291;
st.shared.u32 [r2759+960], r2328;
barrier.sync 0;
ld.shared.u32 r2353, [r2760];
ld.shared.u32 r2549, [r2760+512];
ld.shared.u32 r2403, [r2760+1024];
ld.shared.u32 r2599, [r2760+1536];
ld.shared.u32 r2365, [r2760+2048];
ld.shared.u32 r2561, [r2760+2560];
ld.shared.u32 r2415, [r2760+3072];
ld.shared.u32 r2611, [r2760+3584];
ld.shared.u32 r2354, [r2760+4096];
ld.shared.u32 r2550, [r2760+4608];
ld.shared.u32 r2404, [r2760+5120];
ld.shared.u32 r2600, [r2760+5632];
ld.shared.u32 r2366, [r2760+6144];
ld.shared.u32 r2562, [r2760+6656];
ld.shared.u32 r2416, [r2760+7168];
ld.shared.u32 r2612, [r2760+7680];
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
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r2449, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r2450, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r2453, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
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
add.f16x2 %0, r2375, r2425;
}
{
add.f16x2 %1, r2378, r2428;
}
{
sub.f16x2 %16, r2375, r2425;
}
{
sub.f16x2 %17, r2378, r2428;
}
{
add.f16x2 %4, r2387, r2469;
}
{
add.f16x2 %5, r2390, r2475;
}
{
sub.f16x2 %20, r2387, r2469;
}
{
sub.f16x2 %21, r2390, r2475;
}
{
add.f16x2 %8, r2381, r2434;
}
{
add.f16x2 %9, r2384, r2479;
}
{
sub.f16x2 %24, r2381, r2434;
}
{
sub.f16x2 %25, r2384, r2479;
}
{
add.f16x2 %12, r2393, r2487;
}
{
add.f16x2 %13, r2396, r2493;
}
{
sub.f16x2 %28, r2393, r2487;
}
{
sub.f16x2 %29, r2396, r2493;
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
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r2645, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r2646, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r2649, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
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
add.f16x2 %2, r2571, r2621;
}
{
add.f16x2 %3, r2574, r2624;
}
{
sub.f16x2 %18, r2571, r2621;
}
{
sub.f16x2 %19, r2574, r2624;
}
{
add.f16x2 %6, r2583, r2665;
}
{
add.f16x2 %7, r2586, r2671;
}
{
sub.f16x2 %22, r2583, r2665;
}
{
sub.f16x2 %23, r2586, r2671;
}
{
add.f16x2 %10, r2577, r2630;
}
{
add.f16x2 %11, r2580, r2675;
}
{
sub.f16x2 %26, r2577, r2630;
}
{
sub.f16x2 %27, r2580, r2675;
}
{
add.f16x2 %14, r2589, r2683;
}
{
add.f16x2 %15, r2592, r2689;
}
{
sub.f16x2 %30, r2589, r2683;
}
{
sub.f16x2 %31, r2592, r2689;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<839, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<148>;
.reg .b32 r<1502>;
.reg .b64 rd<2>;
mov.u32 r1475, %tid.y;
shl.b32 r1476, r1475, 13;
mov.u32 r1477, %16;
add.s32 r1478, r1477, r1476;
mov.u32 r1479, %tid.x;
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
mov.f32 f94, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f94;
cvt.rn.f16.f32 high, f94;
mov.b32 r101, {low, high};
}
mov.f32 f104, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r102, {low, high};
}
mov.f32 f135, 0fBF800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
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
and.b32 r1480, r1479, 255;
shl.b32 r1481, r1479, 5;
and.b32 r1482, r1481, -8192;
add.s32 r1483, r1478, r1482;
cvt.rn.f32.u32 f139, r1480;
mul.f32 f140, f139, 0f3B490FDB;
cos.approx.f32 f29, f140;
sin.approx.f32 f141, f140;
neg.f32 f30, f141;
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
and.b32 r1484, r1481, 8160;
add.s32 r1485, r1483, r1484;
st.shared.v4.f32 [r1485], {r149, r209, r246, r283};
st.shared.v4.f32 [r1485+16], {r320, r357, r394, r431};
barrier.sync 0;
mad.lo.s32 r1486, r1480, -28, r1485;
ld.shared.u32 r460, [r1486];
ld.shared.u32 r510, [r1486+1024];
ld.shared.u32 r472, [r1486+2048];
ld.shared.u32 r522, [r1486+3072];
ld.shared.u32 r461, [r1486+4096];
ld.shared.u32 r511, [r1486+5120];
ld.shared.u32 r473, [r1486+6144];
ld.shared.u32 r523, [r1486+7168];
barrier.sync 0;
st.shared.v4.f32 [r1485], {r152, r216, r253, r290};
st.shared.v4.f32 [r1485+16], {r327, r364, r401, r438};
barrier.sync 0;
ld.shared.u32 r463, [r1486];
ld.shared.u32 r513, [r1486+1024];
ld.shared.u32 r475, [r1486+2048];
ld.shared.u32 r525, [r1486+3072];
ld.shared.u32 r464, [r1486+4096];
ld.shared.u32 r514, [r1486+5120];
ld.shared.u32 r476, [r1486+6144];
ld.shared.u32 r526, [r1486+7168];
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
cvt.rn.f16.f32 low, f94;
cvt.rn.f16.f32 high, f94;
mov.b32 r559, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r560, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r563, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
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
and.b32 r1487, r1479, 248;
bfe.u32 r1488, r1479, 3, 5;
shl.b32 r1489, r1479, 2;
and.b32 r1490, r1489, 28;
add.s32 r1491, r1483, r1490;
cvt.rn.f32.u32 f142, r1488;
mul.f32 f143, f142, 0f3CC90FDB;
cos.approx.f32 f75, f143;
sin.approx.f32 f144, f143;
neg.f32 f76, f144;
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
and.b32 r1492, r1481, 7936;
add.s32 r1493, r1491, r1492;
st.shared.u32 [r1493], r607;
st.shared.u32 [r1493+32], r667;
st.shared.u32 [r1493+64], r704;
st.shared.u32 [r1493+96], r741;
st.shared.u32 [r1493+128], r778;
st.shared.u32 [r1493+160], r815;
st.shared.u32 [r1493+192], r852;
st.shared.u32 [r1493+224], r889;
barrier.sync 0;
mad.lo.s32 r1494, r1487, -28, r1493;
ld.shared.u32 r918, [r1494];
ld.shared.u32 r968, [r1494+1024];
ld.shared.u32 r930, [r1494+2048];
ld.shared.u32 r980, [r1494+3072];
ld.shared.u32 r919, [r1494+4096];
ld.shared.u32 r969, [r1494+5120];
ld.shared.u32 r931, [r1494+6144];
ld.shared.u32 r981, [r1494+7168];
barrier.sync 0;
st.shared.u32 [r1493], r610;
st.shared.u32 [r1493+32], r674;
st.shared.u32 [r1493+64], r711;
st.shared.u32 [r1493+96], r748;
st.shared.u32 [r1493+128], r785;
st.shared.u32 [r1493+160], r822;
st.shared.u32 [r1493+192], r859;
st.shared.u32 [r1493+224], r896;
barrier.sync 0;
ld.shared.u32 r921, [r1494];
ld.shared.u32 r971, [r1494+1024];
ld.shared.u32 r933, [r1494+2048];
ld.shared.u32 r983, [r1494+3072];
ld.shared.u32 r922, [r1494+4096];
ld.shared.u32 r972, [r1494+5120];
ld.shared.u32 r934, [r1494+6144];
ld.shared.u32 r984, [r1494+7168];
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
cvt.rn.f16.f32 low, f94;
cvt.rn.f16.f32 high, f94;
mov.b32 r1017, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r1018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r1021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
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
and.b32 r1495, r1479, 192;
bfe.u32 r1496, r1479, 6, 2;
and.b32 r1497, r1489, 252;
add.s32 r1498, r1483, r1497;
cvt.rn.f32.u32 f145, r1496;
mul.f32 f146, f145, 0f3E490FDB;
cos.approx.f32 f121, f146;
sin.approx.f32 f147, f146;
neg.f32 f122, f147;
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
and.b32 r1499, r1481, 6144;
add.s32 r1500, r1498, r1499;
st.shared.u32 [r1500], r1065;
st.shared.u32 [r1500+256], r1125;
st.shared.u32 [r1500+512], r1162;
st.shared.u32 [r1500+768], r1199;
st.shared.u32 [r1500+1024], r1236;
st.shared.u32 [r1500+1280], r1273;
st.shared.u32 [r1500+1536], r1310;
st.shared.u32 [r1500+1792], r1347;
barrier.sync 0;
mad.lo.s32 r1501, r1495, -28, r1500;
ld.shared.u32 r1376, [r1501];
ld.shared.u32 r1426, [r1501+1024];
ld.shared.u32 r1388, [r1501+2048];
ld.shared.u32 r1438, [r1501+3072];
ld.shared.u32 r1377, [r1501+4096];
ld.shared.u32 r1427, [r1501+5120];
ld.shared.u32 r1389, [r1501+6144];
ld.shared.u32 r1439, [r1501+7168];
barrier.sync 0;
st.shared.u32 [r1500], r1068;
st.shared.u32 [r1500+256], r1132;
st.shared.u32 [r1500+512], r1169;
st.shared.u32 [r1500+768], r1206;
st.shared.u32 [r1500+1024], r1243;
st.shared.u32 [r1500+1280], r1280;
st.shared.u32 [r1500+1536], r1317;
st.shared.u32 [r1500+1792], r1354;
barrier.sync 0;
ld.shared.u32 r1379, [r1501];
ld.shared.u32 r1429, [r1501+1024];
ld.shared.u32 r1391, [r1501+2048];
ld.shared.u32 r1441, [r1501+3072];
ld.shared.u32 r1380, [r1501+4096];
ld.shared.u32 r1430, [r1501+5120];
ld.shared.u32 r1392, [r1501+6144];
ld.shared.u32 r1442, [r1501+7168];
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
add.f16x2 %0, r1375, r1387;
}
{
add.f16x2 %1, r1378, r1390;
}
{
sub.f16x2 %8, r1375, r1387;
}
{
sub.f16x2 %9, r1378, r1390;
}
{
add.f16x2 %4, r1381, r1396;
}
{
add.f16x2 %5, r1384, r1399;
}
{
sub.f16x2 %12, r1381, r1396;
}
{
sub.f16x2 %13, r1384, r1399;
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
add.f16x2 %2, r1425, r1437;
}
{
add.f16x2 %3, r1428, r1440;
}
{
sub.f16x2 %10, r1425, r1437;
}
{
sub.f16x2 %11, r1428, r1440;
}
{
add.f16x2 %6, r1431, r1446;
}
{
add.f16x2 %7, r1434, r1449;
}
{
sub.f16x2 %14, r1431, r1446;
}
{
sub.f16x2 %15, r1434, r1449;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<838, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<148>;
.reg .b32 r<1502>;
.reg .b64 rd<2>;
mov.u32 r1475, %tid.y;
shl.b32 r1476, r1475, 14;
mov.u32 r1477, %16;
add.s32 r1478, r1477, r1476;
mov.u32 r1479, %tid.x;
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
mov.f32 f94, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f94;
cvt.rn.f16.f32 high, f94;
mov.b32 r101, {low, high};
}
mov.f32 f104, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r102, {low, high};
}
mov.f32 f135, 0fBF800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
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
and.b32 r1480, r1479, 255;
shl.b32 r1481, r1479, 6;
and.b32 r1482, r1481, -16384;
add.s32 r1483, r1478, r1482;
cvt.rn.f32.u32 f139, r1480;
mul.f32 f140, f139, 0f3B490FDB;
cos.approx.f32 f29, f140;
sin.approx.f32 f141, f140;
neg.f32 f30, f141;
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
and.b32 r1484, r1481, 16320;
add.s32 r1485, r1483, r1484;
st.shared.v4.f32 [r1485], {r149, r152, r209, r216};
st.shared.v4.f32 [r1485+16], {r246, r253, r283, r290};
st.shared.v4.f32 [r1485+32], {r320, r327, r357, r364};
st.shared.v4.f32 [r1485+48], {r394, r401, r431, r438};
barrier.sync 0;
mad.lo.s32 r1486, r1480, -56, r1485;
ld.shared.u32 r460, [r1486];
ld.shared.u32 r463, [r1486+4];
ld.shared.u32 r510, [r1486+2048];
ld.shared.u32 r513, [r1486+2052];
ld.shared.u32 r472, [r1486+4096];
ld.shared.u32 r475, [r1486+4100];
ld.shared.u32 r522, [r1486+6144];
ld.shared.u32 r525, [r1486+6148];
ld.shared.u32 r461, [r1486+8192];
ld.shared.u32 r464, [r1486+8196];
ld.shared.u32 r511, [r1486+10240];
ld.shared.u32 r514, [r1486+10244];
ld.shared.u32 r473, [r1486+12288];
ld.shared.u32 r476, [r1486+12292];
ld.shared.u32 r523, [r1486+14336];
ld.shared.u32 r526, [r1486+14340];
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
cvt.rn.f16.f32 low, f94;
cvt.rn.f16.f32 high, f94;
mov.b32 r559, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r560, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r563, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
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
and.b32 r1487, r1479, 248;
bfe.u32 r1488, r1479, 3, 5;
cvt.rn.f32.u32 f142, r1488;
mul.f32 f143, f142, 0f3CC90FDB;
cos.approx.f32 f75, f143;
sin.approx.f32 f144, f143;
neg.f32 f76, f144;
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
shl.b32 r1489, r1479, 3;
and.b32 r1490, r1489, 56;
add.s32 r1491, r1483, r1490;
barrier.sync 0;
and.b32 r1492, r1481, 15872;
add.s32 r1493, r1491, r1492;
st.shared.u32 [r1493], r607;
st.shared.u32 [r1493+4], r610;
st.shared.u32 [r1493+64], r667;
st.shared.u32 [r1493+68], r674;
st.shared.u32 [r1493+128], r704;
st.shared.u32 [r1493+132], r711;
st.shared.u32 [r1493+192], r741;
st.shared.u32 [r1493+196], r748;
st.shared.u32 [r1493+256], r778;
st.shared.u32 [r1493+260], r785;
st.shared.u32 [r1493+320], r815;
st.shared.u32 [r1493+324], r822;
st.shared.u32 [r1493+384], r852;
st.shared.u32 [r1493+388], r859;
st.shared.u32 [r1493+448], r889;
st.shared.u32 [r1493+452], r896;
barrier.sync 0;
mad.lo.s32 r1494, r1487, -56, r1493;
ld.shared.u32 r918, [r1494];
ld.shared.u32 r921, [r1494+4];
ld.shared.u32 r968, [r1494+2048];
ld.shared.u32 r971, [r1494+2052];
ld.shared.u32 r930, [r1494+4096];
ld.shared.u32 r933, [r1494+4100];
ld.shared.u32 r980, [r1494+6144];
ld.shared.u32 r983, [r1494+6148];
ld.shared.u32 r919, [r1494+8192];
ld.shared.u32 r922, [r1494+8196];
ld.shared.u32 r969, [r1494+10240];
ld.shared.u32 r972, [r1494+10244];
ld.shared.u32 r931, [r1494+12288];
ld.shared.u32 r934, [r1494+12292];
ld.shared.u32 r981, [r1494+14336];
ld.shared.u32 r984, [r1494+14340];
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
cvt.rn.f16.f32 low, f94;
cvt.rn.f16.f32 high, f94;
mov.b32 r1017, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r1018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r1021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
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
and.b32 r1495, r1479, 192;
bfe.u32 r1496, r1479, 6, 2;
cvt.rn.f32.u32 f145, r1496;
mul.f32 f146, f145, 0f3E490FDB;
cos.approx.f32 f121, f146;
sin.approx.f32 f147, f146;
neg.f32 f122, f147;
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
and.b32 r1497, r1489, 504;
add.s32 r1498, r1483, r1497;
barrier.sync 0;
and.b32 r1499, r1481, 12288;
add.s32 r1500, r1498, r1499;
st.shared.u32 [r1500], r1065;
st.shared.u32 [r1500+4], r1068;
st.shared.u32 [r1500+512], r1125;
st.shared.u32 [r1500+516], r1132;
st.shared.u32 [r1500+1024], r1162;
st.shared.u32 [r1500+1028], r1169;
st.shared.u32 [r1500+1536], r1199;
st.shared.u32 [r1500+1540], r1206;
st.shared.u32 [r1500+2048], r1236;
st.shared.u32 [r1500+2052], r1243;
st.shared.u32 [r1500+2560], r1273;
st.shared.u32 [r1500+2564], r1280;
st.shared.u32 [r1500+3072], r1310;
st.shared.u32 [r1500+3076], r1317;
st.shared.u32 [r1500+3584], r1347;
st.shared.u32 [r1500+3588], r1354;
barrier.sync 0;
mad.lo.s32 r1501, r1495, -56, r1500;
ld.shared.u32 r1376, [r1501];
ld.shared.u32 r1379, [r1501+4];
ld.shared.u32 r1426, [r1501+2048];
ld.shared.u32 r1429, [r1501+2052];
ld.shared.u32 r1388, [r1501+4096];
ld.shared.u32 r1391, [r1501+4100];
ld.shared.u32 r1438, [r1501+6144];
ld.shared.u32 r1441, [r1501+6148];
ld.shared.u32 r1377, [r1501+8192];
ld.shared.u32 r1380, [r1501+8196];
ld.shared.u32 r1427, [r1501+10240];
ld.shared.u32 r1430, [r1501+10244];
ld.shared.u32 r1389, [r1501+12288];
ld.shared.u32 r1392, [r1501+12292];
ld.shared.u32 r1439, [r1501+14336];
ld.shared.u32 r1442, [r1501+14340];
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
add.f16x2 %0, r1375, r1387;
}
{
add.f16x2 %1, r1378, r1390;
}
{
sub.f16x2 %8, r1375, r1387;
}
{
sub.f16x2 %9, r1378, r1390;
}
{
add.f16x2 %4, r1381, r1396;
}
{
add.f16x2 %5, r1384, r1399;
}
{
sub.f16x2 %12, r1381, r1396;
}
{
sub.f16x2 %13, r1384, r1399;
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
add.f16x2 %2, r1425, r1437;
}
{
add.f16x2 %3, r1428, r1440;
}
{
sub.f16x2 %10, r1425, r1437;
}
{
sub.f16x2 %11, r1428, r1440;
}
{
add.f16x2 %6, r1431, r1446;
}
{
add.f16x2 %7, r1434, r1449;
}
{
sub.f16x2 %14, r1431, r1446;
}
{
sub.f16x2 %15, r1434, r1449;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<840, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<363>;
.reg .b32 r<2761>;
.reg .b64 rd<2>;
mov.u32 r2741, %tid.y;
shl.b32 r2742, r2741, 14;
mov.u32 r2743, %32;
add.s32 r2744, r2743, r2742;
mov.u32 r2745, %tid.x;
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
mov.f32 f330, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r101, {low, high};
}
mov.f32 f340, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r102, {low, high};
}
mov.f32 f297, 0fBF800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
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
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
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
mov.f32 f208, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f208;
cvt.rn.f16.f32 high, f208;
mov.b32 r393, {low, high};
}
mov.f32 f234, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f234;
cvt.rn.f16.f32 high, f234;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r396, {low, high};
}
mov.f32 f216, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r397, {low, high};
}
mov.f32 f232, 0fBF6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f232;
cvt.rn.f16.f32 high, f232;
mov.b32 r398, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f234;
cvt.rn.f16.f32 high, f234;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f232;
cvt.rn.f16.f32 high, f232;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r404, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f232;
cvt.rn.f16.f32 high, f232;
mov.b32 r405, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f234;
cvt.rn.f16.f32 high, f234;
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
and.b32 r2746, r2745, 127;
shl.b32 r2747, r2745, 7;
and.b32 r2748, r2747, -16384;
add.s32 r2749, r2744, r2748;
cvt.rn.f32.u32 f357, r2746;
mul.f32 f358, f357, 0f3B490FDB;
cos.approx.f32 f117, f358;
sin.approx.f32 f359, f358;
neg.f32 f118, f359;
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
and.b32 r2750, r2747, 16256;
add.s32 r2751, r2749, r2750;
st.shared.v4.f32 [r2751], {r521, r524, r629, r636};
st.shared.v4.f32 [r2751+16], {r666, r673, r703, r710};
st.shared.v4.f32 [r2751+32], {r740, r747, r777, r784};
st.shared.v4.f32 [r2751+48], {r814, r821, r851, r858};
st.shared.v4.f32 [r2751+64], {r888, r895, r925, r932};
st.shared.v4.f32 [r2751+80], {r962, r969, r999, r1006};
st.shared.v4.f32 [r2751+96], {r1036, r1043, r1073, r1080};
st.shared.v4.f32 [r2751+112], {r1110, r1117, r1147, r1154};
barrier.sync 0;
mad.lo.s32 r2752, r2746, -120, r2751;
ld.shared.u32 r1176, [r2752];
ld.shared.u32 r1179, [r2752+4];
ld.shared.u32 r1372, [r2752+1024];
ld.shared.u32 r1375, [r2752+1028];
ld.shared.u32 r1226, [r2752+2048];
ld.shared.u32 r1229, [r2752+2052];
ld.shared.u32 r1422, [r2752+3072];
ld.shared.u32 r1425, [r2752+3076];
ld.shared.u32 r1188, [r2752+4096];
ld.shared.u32 r1191, [r2752+4100];
ld.shared.u32 r1384, [r2752+5120];
ld.shared.u32 r1387, [r2752+5124];
ld.shared.u32 r1238, [r2752+6144];
ld.shared.u32 r1241, [r2752+6148];
ld.shared.u32 r1434, [r2752+7168];
ld.shared.u32 r1437, [r2752+7172];
ld.shared.u32 r1177, [r2752+8192];
ld.shared.u32 r1180, [r2752+8196];
ld.shared.u32 r1373, [r2752+9216];
ld.shared.u32 r1376, [r2752+9220];
ld.shared.u32 r1227, [r2752+10240];
ld.shared.u32 r1230, [r2752+10244];
ld.shared.u32 r1423, [r2752+11264];
ld.shared.u32 r1426, [r2752+11268];
ld.shared.u32 r1189, [r2752+12288];
ld.shared.u32 r1192, [r2752+12292];
ld.shared.u32 r1385, [r2752+13312];
ld.shared.u32 r1388, [r2752+13316];
ld.shared.u32 r1239, [r2752+14336];
ld.shared.u32 r1242, [r2752+14340];
ld.shared.u32 r1435, [r2752+15360];
ld.shared.u32 r1438, [r2752+15364];
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
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r1275, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r1276, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r1279, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
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
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r1471, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r1472, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r1475, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
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
cvt.rn.f16.f32 low, f208;
cvt.rn.f16.f32 high, f208;
mov.b32 r1567, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f234;
cvt.rn.f16.f32 high, f234;
mov.b32 r1568, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r1569, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r1570, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1571, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f232;
cvt.rn.f16.f32 high, f232;
mov.b32 r1572, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f234;
cvt.rn.f16.f32 high, f234;
mov.b32 r1575, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f232;
cvt.rn.f16.f32 high, f232;
mov.b32 r1576, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r1577, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r1578, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f232;
cvt.rn.f16.f32 high, f232;
mov.b32 r1579, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f234;
cvt.rn.f16.f32 high, f234;
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
and.b32 r2753, r2745, 112;
bfe.u32 r2754, r2745, 4, 3;
cvt.rn.f32.u32 f360, r2754;
mul.f32 f361, f360, 0f3D490FDB;
cos.approx.f32 f267, f361;
sin.approx.f32 f362, f361;
neg.f32 f268, f362;
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
shl.b32 r2755, r2745, 3;
and.b32 r2756, r2755, 120;
add.s32 r2757, r2749, r2756;
barrier.sync 0;
and.b32 r2758, r2747, 14336;
add.s32 r2759, r2757, r2758;
st.shared.u32 [r2759], r1695;
st.shared.u32 [r2759+4], r1698;
st.shared.u32 [r2759+128], r1803;
st.shared.u32 [r2759+132], r1810;
st.shared.u32 [r2759+256], r1840;
st.shared.u32 [r2759+260], r1847;
st.shared.u32 [r2759+384], r1877;
st.shared.u32 [r2759+388], r1884;
st.shared.u32 [r2759+512], r1914;
st.shared.u32 [r2759+516], r1921;
st.shared.u32 [r2759+640], r1951;
st.shared.u32 [r2759+644], r1958;
st.shared.u32 [r2759+768], r1988;
st.shared.u32 [r2759+772], r1995;
st.shared.u32 [r2759+896], r2025;
st.shared.u32 [r2759+900], r2032;
st.shared.u32 [r2759+1024], r2062;
st.shared.u32 [r2759+1028], r2069;
st.shared.u32 [r2759+1152], r2099;
st.shared.u32 [r2759+1156], r2106;
st.shared.u32 [r2759+1280], r2136;
st.shared.u32 [r2759+1284], r2143;
st.shared.u32 [r2759+1408], r2173;
st.shared.u32 [r2759+1412], r2180;
st.shared.u32 [r2759+1536], r2210;
st.shared.u32 [r2759+1540], r2217;
st.shared.u32 [r2759+1664], r2247;
st.shared.u32 [r2759+1668], r2254;
st.shared.u32 [r2759+1792], r2284;
st.shared.u32 [r2759+1796], r2291;
st.shared.u32 [r2759+1920], r2321;
st.shared.u32 [r2759+1924], r2328;
barrier.sync 0;
mad.lo.s32 r2760, r2753, -120, r2759;
ld.shared.u32 r2350, [r2760];
ld.shared.u32 r2353, [r2760+4];
ld.shared.u32 r2546, [r2760+1024];
ld.shared.u32 r2549, [r2760+1028];
ld.shared.u32 r2400, [r2760+2048];
ld.shared.u32 r2403, [r2760+2052];
ld.shared.u32 r2596, [r2760+3072];
ld.shared.u32 r2599, [r2760+3076];
ld.shared.u32 r2362, [r2760+4096];
ld.shared.u32 r2365, [r2760+4100];
ld.shared.u32 r2558, [r2760+5120];
ld.shared.u32 r2561, [r2760+5124];
ld.shared.u32 r2412, [r2760+6144];
ld.shared.u32 r2415, [r2760+6148];
ld.shared.u32 r2608, [r2760+7168];
ld.shared.u32 r2611, [r2760+7172];
ld.shared.u32 r2351, [r2760+8192];
ld.shared.u32 r2354, [r2760+8196];
ld.shared.u32 r2547, [r2760+9216];
ld.shared.u32 r2550, [r2760+9220];
ld.shared.u32 r2401, [r2760+10240];
ld.shared.u32 r2404, [r2760+10244];
ld.shared.u32 r2597, [r2760+11264];
ld.shared.u32 r2600, [r2760+11268];
ld.shared.u32 r2363, [r2760+12288];
ld.shared.u32 r2366, [r2760+12292];
ld.shared.u32 r2559, [r2760+13312];
ld.shared.u32 r2562, [r2760+13316];
ld.shared.u32 r2413, [r2760+14336];
ld.shared.u32 r2416, [r2760+14340];
ld.shared.u32 r2609, [r2760+15360];
ld.shared.u32 r2612, [r2760+15364];
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
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r2449, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r2450, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r2453, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
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
add.f16x2 %0, r2375, r2425;
}
{
add.f16x2 %1, r2378, r2428;
}
{
sub.f16x2 %16, r2375, r2425;
}
{
sub.f16x2 %17, r2378, r2428;
}
{
add.f16x2 %4, r2387, r2469;
}
{
add.f16x2 %5, r2390, r2475;
}
{
sub.f16x2 %20, r2387, r2469;
}
{
sub.f16x2 %21, r2390, r2475;
}
{
add.f16x2 %8, r2381, r2434;
}
{
add.f16x2 %9, r2384, r2479;
}
{
sub.f16x2 %24, r2381, r2434;
}
{
sub.f16x2 %25, r2384, r2479;
}
{
add.f16x2 %12, r2393, r2487;
}
{
add.f16x2 %13, r2396, r2493;
}
{
sub.f16x2 %28, r2393, r2487;
}
{
sub.f16x2 %29, r2396, r2493;
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
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r2645, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r2646, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r2649, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
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
add.f16x2 %2, r2571, r2621;
}
{
add.f16x2 %3, r2574, r2624;
}
{
sub.f16x2 %18, r2571, r2621;
}
{
sub.f16x2 %19, r2574, r2624;
}
{
add.f16x2 %6, r2583, r2665;
}
{
add.f16x2 %7, r2586, r2671;
}
{
sub.f16x2 %22, r2583, r2665;
}
{
sub.f16x2 %23, r2586, r2671;
}
{
add.f16x2 %10, r2577, r2630;
}
{
add.f16x2 %11, r2580, r2675;
}
{
sub.f16x2 %26, r2577, r2630;
}
{
sub.f16x2 %27, r2580, r2675;
}
{
add.f16x2 %14, r2589, r2683;
}
{
add.f16x2 %15, r2592, r2689;
}
{
sub.f16x2 %30, r2589, r2683;
}
{
sub.f16x2 %31, r2592, r2689;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<841, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<853>;
.reg .b32 r<6003>;
.reg .b64 rd<3>;
mov.u32 r5917, %tid.y;
shl.b32 r5918, r5917, 14;
mov.u32 r5919, %64;
add.s32 r5920, r5919, r5918;
mov.u32 r5921, %tid.x;
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
add.f16x2 r632, %121, %114;
}
{
sub.f16x2 r635, %86, %76;
}
{
sub.f16x2 r638, %121, %114;
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
shl.b32 r5923, r5921, 8;
and.b32 r5924, r5923, -16384;
add.s32 r5925, r5920, r5924;
and.b32 r5938, r5921, 63;
cvt.rn.f32.u32 f845, r5938;
mul.f32 f846, f845, 0f3B490FDB;
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
and.b32 r5926, r5923, 16128;
add.s32 r5927, r5925, r5926;
st.shared.v4.f32 [r5927], {r1521, r1524, r1725, r1732};
st.shared.v4.f32 [r5927+16], {r1762, r1769, r1799, r1806};
st.shared.v4.f32 [r5927+32], {r1836, r1843, r1873, r1880};
st.shared.v4.f32 [r5927+48], {r1910, r1917, r1947, r1954};
st.shared.v4.f32 [r5927+64], {r1984, r1991, r2021, r2028};
st.shared.v4.f32 [r5927+80], {r2058, r2065, r2095, r2102};
st.shared.v4.f32 [r5927+96], {r2132, r2139, r2169, r2176};
st.shared.v4.f32 [r5927+112], {r2206, r2213, r2243, r2250};
st.shared.v4.f32 [r5927+128], {r2280, r2287, r2317, r2324};
st.shared.v4.f32 [r5927+144], {r2354, r2361, r2391, r2398};
st.shared.v4.f32 [r5927+160], {r2428, r2435, r2465, r2472};
st.shared.v4.f32 [r5927+176], {r2502, r2509, r2539, r2546};
st.shared.v4.f32 [r5927+192], {r2576, r2583, r2613, r2620};
st.shared.v4.f32 [r5927+208], {r2650, r2657, r2687, r2694};
st.shared.v4.f32 [r5927+224], {r2724, r2731, r2761, r2768};
st.shared.v4.f32 [r5927+240], {r2798, r2805, r2835, r2842};
barrier.sync 0;
mad.lo.s32 r5928, r5938, -248, r5927;
ld.shared.u32 r2864, [r5928];
ld.shared.u32 r2867, [r5928+4];
ld.shared.u32 r3480, [r5928+512];
ld.shared.u32 r3483, [r5928+516];
ld.shared.u32 r3060, [r5928+1024];
ld.shared.u32 r3063, [r5928+1028];
ld.shared.u32 r3676, [r5928+1536];
ld.shared.u32 r3679, [r5928+1540];
ld.shared.u32 r2914, [r5928+2048];
ld.shared.u32 r2917, [r5928+2052];
ld.shared.u32 r3530, [r5928+2560];
ld.shared.u32 r3533, [r5928+2564];
ld.shared.u32 r3110, [r5928+3072];
ld.shared.u32 r3113, [r5928+3076];
ld.shared.u32 r3726, [r5928+3584];
ld.shared.u32 r3729, [r5928+3588];
ld.shared.u32 r2876, [r5928+4096];
ld.shared.u32 r2879, [r5928+4100];
ld.shared.u32 r3492, [r5928+4608];
ld.shared.u32 r3495, [r5928+4612];
ld.shared.u32 r3072, [r5928+5120];
ld.shared.u32 r3075, [r5928+5124];
ld.shared.u32 r3688, [r5928+5632];
ld.shared.u32 r3691, [r5928+5636];
ld.shared.u32 r2926, [r5928+6144];
ld.shared.u32 r2929, [r5928+6148];
ld.shared.u32 r3542, [r5928+6656];
ld.shared.u32 r3545, [r5928+6660];
ld.shared.u32 r3122, [r5928+7168];
ld.shared.u32 r3125, [r5928+7172];
ld.shared.u32 r3738, [r5928+7680];
ld.shared.u32 r3741, [r5928+7684];
ld.shared.u32 r2865, [r5928+8192];
ld.shared.u32 r2868, [r5928+8196];
ld.shared.u32 r3481, [r5928+8704];
ld.shared.u32 r3484, [r5928+8708];
ld.shared.u32 r3061, [r5928+9216];
ld.shared.u32 r3064, [r5928+9220];
ld.shared.u32 r3677, [r5928+9728];
ld.shared.u32 r3680, [r5928+9732];
ld.shared.u32 r2915, [r5928+10240];
ld.shared.u32 r2918, [r5928+10244];
ld.shared.u32 r3531, [r5928+10752];
ld.shared.u32 r3534, [r5928+10756];
ld.shared.u32 r3111, [r5928+11264];
ld.shared.u32 r3114, [r5928+11268];
ld.shared.u32 r3727, [r5928+11776];
ld.shared.u32 r3730, [r5928+11780];
ld.shared.u32 r2877, [r5928+12288];
ld.shared.u32 r2880, [r5928+12292];
ld.shared.u32 r3493, [r5928+12800];
ld.shared.u32 r3496, [r5928+12804];
ld.shared.u32 r3073, [r5928+13312];
ld.shared.u32 r3076, [r5928+13316];
ld.shared.u32 r3689, [r5928+13824];
ld.shared.u32 r3692, [r5928+13828];
ld.shared.u32 r2927, [r5928+14336];
ld.shared.u32 r2930, [r5928+14340];
ld.shared.u32 r3543, [r5928+14848];
ld.shared.u32 r3546, [r5928+14852];
ld.shared.u32 r3123, [r5928+15360];
ld.shared.u32 r3126, [r5928+15364];
ld.shared.u32 r3739, [r5928+15872];
ld.shared.u32 r3742, [r5928+15876];
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
bfe.u32 r5930, r5921, 5, 1;
cvt.rn.f32.u32 f848, r5930;
mul.f32 f849, f848, 0f3DC90FDB;
cos.approx.f32 f779, f849;
sin.approx.f32 f850, f849;
neg.f32 f780, f850;
and.b32 r5937, r5921, 32;
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
shl.b32 r5931, r5921, 3;
and.b32 r5932, r5931, 248;
add.s32 r5933, r5925, r5932;
barrier.sync 0;
and.b32 r5934, r5923, 8192;
add.s32 r5935, r5933, r5934;
st.shared.u32 [r5935], r4383;
st.shared.u32 [r5935+4], r4386;
st.shared.u32 [r5935+256], r4587;
st.shared.u32 [r5935+260], r4594;
st.shared.u32 [r5935+512], r4624;
st.shared.u32 [r5935+516], r4631;
st.shared.u32 [r5935+768], r4661;
st.shared.u32 [r5935+772], r4668;
st.shared.u32 [r5935+1024], r4698;
st.shared.u32 [r5935+1028], r4705;
st.shared.u32 [r5935+1280], r4735;
st.shared.u32 [r5935+1284], r4742;
st.shared.u32 [r5935+1536], r4772;
st.shared.u32 [r5935+1540], r4779;
st.shared.u32 [r5935+1792], r4809;
st.shared.u32 [r5935+1796], r4816;
st.shared.u32 [r5935+2048], r4846;
st.shared.u32 [r5935+2052], r4853;
st.shared.u32 [r5935+2304], r4883;
st.shared.u32 [r5935+2308], r4890;
st.shared.u32 [r5935+2560], r4920;
st.shared.u32 [r5935+2564], r4927;
st.shared.u32 [r5935+2816], r4957;
st.shared.u32 [r5935+2820], r4964;
st.shared.u32 [r5935+3072], r4994;
st.shared.u32 [r5935+3076], r5001;
st.shared.u32 [r5935+3328], r5031;
st.shared.u32 [r5935+3332], r5038;
st.shared.u32 [r5935+3584], r5068;
st.shared.u32 [r5935+3588], r5075;
st.shared.u32 [r5935+3840], r5105;
st.shared.u32 [r5935+3844], r5112;
st.shared.u32 [r5935+4096], r5142;
st.shared.u32 [r5935+4100], r5149;
st.shared.u32 [r5935+4352], r5179;
st.shared.u32 [r5935+4356], r5186;
st.shared.u32 [r5935+4608], r5216;
st.shared.u32 [r5935+4612], r5223;
st.shared.u32 [r5935+4864], r5253;
st.shared.u32 [r5935+4868], r5260;
st.shared.u32 [r5935+5120], r5290;
st.shared.u32 [r5935+5124], r5297;
st.shared.u32 [r5935+5376], r5327;
st.shared.u32 [r5935+5380], r5334;
st.shared.u32 [r5935+5632], r5364;
st.shared.u32 [r5935+5636], r5371;
st.shared.u32 [r5935+5888], r5401;
st.shared.u32 [r5935+5892], r5408;
st.shared.u32 [r5935+6144], r5438;
st.shared.u32 [r5935+6148], r5445;
st.shared.u32 [r5935+6400], r5475;
st.shared.u32 [r5935+6404], r5482;
st.shared.u32 [r5935+6656], r5512;
st.shared.u32 [r5935+6660], r5519;
st.shared.u32 [r5935+6912], r5549;
st.shared.u32 [r5935+6916], r5556;
st.shared.u32 [r5935+7168], r5586;
st.shared.u32 [r5935+7172], r5593;
st.shared.u32 [r5935+7424], r5623;
st.shared.u32 [r5935+7428], r5630;
st.shared.u32 [r5935+7680], r5660;
st.shared.u32 [r5935+7684], r5667;
st.shared.u32 [r5935+7936], r5697;
st.shared.u32 [r5935+7940], r5704;
barrier.sync 0;
mad.lo.s32 r5936, r5937, -248, r5935;
ld.shared.u32 r5726, [r5936];
ld.shared.u32 r5729, [r5936+4];
ld.shared.u32 r5738, [r5936+512];
ld.shared.u32 r5741, [r5936+516];
ld.shared.u32 r5750, [r5936+1024];
ld.shared.u32 r5753, [r5936+1028];
ld.shared.u32 r5762, [r5936+1536];
ld.shared.u32 r5765, [r5936+1540];
ld.shared.u32 r5774, [r5936+2048];
ld.shared.u32 r5777, [r5936+2052];
ld.shared.u32 r5786, [r5936+2560];
ld.shared.u32 r5789, [r5936+2564];
ld.shared.u32 r5798, [r5936+3072];
ld.shared.u32 r5801, [r5936+3076];
ld.shared.u32 r5810, [r5936+3584];
ld.shared.u32 r5813, [r5936+3588];
ld.shared.u32 r5822, [r5936+4096];
ld.shared.u32 r5825, [r5936+4100];
ld.shared.u32 r5834, [r5936+4608];
ld.shared.u32 r5837, [r5936+4612];
ld.shared.u32 r5846, [r5936+5120];
ld.shared.u32 r5849, [r5936+5124];
ld.shared.u32 r5858, [r5936+5632];
ld.shared.u32 r5861, [r5936+5636];
ld.shared.u32 r5870, [r5936+6144];
ld.shared.u32 r5873, [r5936+6148];
ld.shared.u32 r5882, [r5936+6656];
ld.shared.u32 r5885, [r5936+6660];
ld.shared.u32 r5894, [r5936+7168];
ld.shared.u32 r5897, [r5936+7172];
ld.shared.u32 r5906, [r5936+7680];
ld.shared.u32 r5909, [r5936+7684];
ld.shared.u32 r5727, [r5936+8192];
ld.shared.u32 r5730, [r5936+8196];
ld.shared.u32 r5739, [r5936+8704];
ld.shared.u32 r5742, [r5936+8708];
ld.shared.u32 r5751, [r5936+9216];
ld.shared.u32 r5754, [r5936+9220];
ld.shared.u32 r5763, [r5936+9728];
ld.shared.u32 r5766, [r5936+9732];
ld.shared.u32 r5775, [r5936+10240];
ld.shared.u32 r5778, [r5936+10244];
ld.shared.u32 r5787, [r5936+10752];
ld.shared.u32 r5790, [r5936+10756];
ld.shared.u32 r5799, [r5936+11264];
ld.shared.u32 r5802, [r5936+11268];
ld.shared.u32 r5811, [r5936+11776];
ld.shared.u32 r5814, [r5936+11780];
ld.shared.u32 r5823, [r5936+12288];
ld.shared.u32 r5826, [r5936+12292];
ld.shared.u32 r5835, [r5936+12800];
ld.shared.u32 r5838, [r5936+12804];
ld.shared.u32 r5847, [r5936+13312];
ld.shared.u32 r5850, [r5936+13316];
ld.shared.u32 r5859, [r5936+13824];
ld.shared.u32 r5862, [r5936+13828];
ld.shared.u32 r5871, [r5936+14336];
ld.shared.u32 r5874, [r5936+14340];
ld.shared.u32 r5883, [r5936+14848];
ld.shared.u32 r5886, [r5936+14852];
ld.shared.u32 r5895, [r5936+15360];
ld.shared.u32 r5898, [r5936+15364];
ld.shared.u32 r5907, [r5936+15872];
ld.shared.u32 r5910, [r5936+15876];
{
add.f16x2 %0, r5726, r5727;
}
{
add.f16x2 %1, r5729, r5730;
}
{
sub.f16x2 %32, r5726, r5727;
}
{
sub.f16x2 %33, r5729, r5730;
}
{
add.f16x2 %2, r5738, r5739;
}
{
add.f16x2 %3, r5741, r5742;
}
{
sub.f16x2 %34, r5738, r5739;
}
{
sub.f16x2 %35, r5741, r5742;
}
{
add.f16x2 %4, r5750, r5751;
}
{
add.f16x2 %5, r5753, r5754;
}
{
sub.f16x2 %36, r5750, r5751;
}
{
sub.f16x2 %37, r5753, r5754;
}
{
add.f16x2 %6, r5762, r5763;
}
{
add.f16x2 %7, r5765, r5766;
}
{
sub.f16x2 %38, r5762, r5763;
}
{
sub.f16x2 %39, r5765, r5766;
}
{
add.f16x2 %8, r5774, r5775;
}
{
add.f16x2 %9, r5777, r5778;
}
{
sub.f16x2 %40, r5774, r5775;
}
{
sub.f16x2 %41, r5777, r5778;
}
{
add.f16x2 %10, r5786, r5787;
}
{
add.f16x2 %11, r5789, r5790;
}
{
sub.f16x2 %42, r5786, r5787;
}
{
sub.f16x2 %43, r5789, r5790;
}
{
add.f16x2 %12, r5798, r5799;
}
{
add.f16x2 %13, r5801, r5802;
}
{
sub.f16x2 %44, r5798, r5799;
}
{
sub.f16x2 %45, r5801, r5802;
}
{
add.f16x2 %14, r5810, r5811;
}
{
add.f16x2 %15, r5813, r5814;
}
{
sub.f16x2 %46, r5810, r5811;
}
{
sub.f16x2 %47, r5813, r5814;
}
{
add.f16x2 %16, r5822, r5823;
}
{
add.f16x2 %17, r5825, r5826;
}
{
sub.f16x2 %48, r5822, r5823;
}
{
sub.f16x2 %49, r5825, r5826;
}
{
add.f16x2 %18, r5834, r5835;
}
{
add.f16x2 %19, r5837, r5838;
}
{
sub.f16x2 %50, r5834, r5835;
}
{
sub.f16x2 %51, r5837, r5838;
}
{
add.f16x2 %20, r5846, r5847;
}
{
add.f16x2 %21, r5849, r5850;
}
{
sub.f16x2 %52, r5846, r5847;
}
{
sub.f16x2 %53, r5849, r5850;
}
{
add.f16x2 %22, r5858, r5859;
}
{
add.f16x2 %23, r5861, r5862;
}
{
sub.f16x2 %54, r5858, r5859;
}
{
sub.f16x2 %55, r5861, r5862;
}
{
add.f16x2 %24, r5870, r5871;
}
{
add.f16x2 %25, r5873, r5874;
}
{
sub.f16x2 %56, r5870, r5871;
}
{
sub.f16x2 %57, r5873, r5874;
}
{
add.f16x2 %26, r5882, r5883;
}
{
add.f16x2 %27, r5885, r5886;
}
{
sub.f16x2 %58, r5882, r5883;
}
{
sub.f16x2 %59, r5885, r5886;
}
{
add.f16x2 %28, r5894, r5895;
}
{
add.f16x2 %29, r5897, r5898;
}
{
sub.f16x2 %60, r5894, r5895;
}
{
sub.f16x2 %61, r5897, r5898;
}
{
add.f16x2 %30, r5906, r5907;
}
{
add.f16x2 %31, r5909, r5910;
}
{
sub.f16x2 %62, r5906, r5907;
}
{
sub.f16x2 %63, r5909, r5910;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)), "=r"(__HALF2_TO_UI(rmem[27].x)), "=r"(__HALF2_TO_UI(rmem[27].y)), "=r"(__HALF2_TO_UI(rmem[28].x)), "=r"(__HALF2_TO_UI(rmem[28].y)), "=r"(__HALF2_TO_UI(rmem[29].x)), "=r"(__HALF2_TO_UI(rmem[29].y)), "=r"(__HALF2_TO_UI(rmem[30].x)), "=r"(__HALF2_TO_UI(rmem[30].y)), "=r"(__HALF2_TO_UI(rmem[31].x)), "=r"(__HALF2_TO_UI(rmem[31].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[31].y)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[28].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[27].y)), "r"(__HALF2_TO_UI(rmem[30].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[29].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[31].x)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[28].x)), "r"(__HALF2_TO_UI(rmem[27].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[30].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[29].x)), "r"(__HALF2_TO_UI(rmem[24].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<842, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<853>;
.reg .b32 r<6002>;
.reg .b64 rd<3>;
mov.u32 r5917, %tid.y;
shl.b32 r5918, r5917, 13;
mov.u32 r5919, %64;
add.s32 r5920, r5919, r5918;
mov.u32 r5921, %tid.x;
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
add.f16x2 r632, %121, %114;
}
{
sub.f16x2 r635, %86, %76;
}
{
sub.f16x2 r638, %121, %114;
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
shl.b32 r5923, r5921, 7;
and.b32 r5924, r5923, -8192;
add.s32 r5925, r5920, r5924;
and.b32 r5937, r5921, 63;
cvt.rn.f32.u32 f845, r5937;
mul.f32 f846, f845, 0f3B490FDB;
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
and.b32 r5926, r5923, 8064;
add.s32 r5927, r5925, r5926;
st.shared.v4.f32 [r5927], {r1521, r1725, r1762, r1799};
st.shared.v4.f32 [r5927+16], {r1836, r1873, r1910, r1947};
st.shared.v4.f32 [r5927+32], {r1984, r2021, r2058, r2095};
st.shared.v4.f32 [r5927+48], {r2132, r2169, r2206, r2243};
st.shared.v4.f32 [r5927+64], {r2280, r2317, r2354, r2391};
st.shared.v4.f32 [r5927+80], {r2428, r2465, r2502, r2539};
st.shared.v4.f32 [r5927+96], {r2576, r2613, r2650, r2687};
st.shared.v4.f32 [r5927+112], {r2724, r2761, r2798, r2835};
barrier.sync 0;
mad.lo.s32 r5928, r5937, -124, r5927;
ld.shared.u32 r2864, [r5928];
ld.shared.u32 r3480, [r5928+256];
ld.shared.u32 r3060, [r5928+512];
ld.shared.u32 r3676, [r5928+768];
ld.shared.u32 r2914, [r5928+1024];
ld.shared.u32 r3530, [r5928+1280];
ld.shared.u32 r3110, [r5928+1536];
ld.shared.u32 r3726, [r5928+1792];
ld.shared.u32 r2876, [r5928+2048];
ld.shared.u32 r3492, [r5928+2304];
ld.shared.u32 r3072, [r5928+2560];
ld.shared.u32 r3688, [r5928+2816];
ld.shared.u32 r2926, [r5928+3072];
ld.shared.u32 r3542, [r5928+3328];
ld.shared.u32 r3122, [r5928+3584];
ld.shared.u32 r3738, [r5928+3840];
ld.shared.u32 r2865, [r5928+4096];
ld.shared.u32 r3481, [r5928+4352];
ld.shared.u32 r3061, [r5928+4608];
ld.shared.u32 r3677, [r5928+4864];
ld.shared.u32 r2915, [r5928+5120];
ld.shared.u32 r3531, [r5928+5376];
ld.shared.u32 r3111, [r5928+5632];
ld.shared.u32 r3727, [r5928+5888];
ld.shared.u32 r2877, [r5928+6144];
ld.shared.u32 r3493, [r5928+6400];
ld.shared.u32 r3073, [r5928+6656];
ld.shared.u32 r3689, [r5928+6912];
ld.shared.u32 r2927, [r5928+7168];
ld.shared.u32 r3543, [r5928+7424];
ld.shared.u32 r3123, [r5928+7680];
ld.shared.u32 r3739, [r5928+7936];
barrier.sync 0;
st.shared.v4.f32 [r5927], {r1524, r1732, r1769, r1806};
st.shared.v4.f32 [r5927+16], {r1843, r1880, r1917, r1954};
st.shared.v4.f32 [r5927+32], {r1991, r2028, r2065, r2102};
st.shared.v4.f32 [r5927+48], {r2139, r2176, r2213, r2250};
st.shared.v4.f32 [r5927+64], {r2287, r2324, r2361, r2398};
st.shared.v4.f32 [r5927+80], {r2435, r2472, r2509, r2546};
st.shared.v4.f32 [r5927+96], {r2583, r2620, r2657, r2694};
st.shared.v4.f32 [r5927+112], {r2731, r2768, r2805, r2842};
barrier.sync 0;
ld.shared.u32 r2867, [r5928];
ld.shared.u32 r3483, [r5928+256];
ld.shared.u32 r3063, [r5928+512];
ld.shared.u32 r3679, [r5928+768];
ld.shared.u32 r2917, [r5928+1024];
ld.shared.u32 r3533, [r5928+1280];
ld.shared.u32 r3113, [r5928+1536];
ld.shared.u32 r3729, [r5928+1792];
ld.shared.u32 r2879, [r5928+2048];
ld.shared.u32 r3495, [r5928+2304];
ld.shared.u32 r3075, [r5928+2560];
ld.shared.u32 r3691, [r5928+2816];
ld.shared.u32 r2929, [r5928+3072];
ld.shared.u32 r3545, [r5928+3328];
ld.shared.u32 r3125, [r5928+3584];
ld.shared.u32 r3741, [r5928+3840];
ld.shared.u32 r2868, [r5928+4096];
ld.shared.u32 r3484, [r5928+4352];
ld.shared.u32 r3064, [r5928+4608];
ld.shared.u32 r3680, [r5928+4864];
ld.shared.u32 r2918, [r5928+5120];
ld.shared.u32 r3534, [r5928+5376];
ld.shared.u32 r3114, [r5928+5632];
ld.shared.u32 r3730, [r5928+5888];
ld.shared.u32 r2880, [r5928+6144];
ld.shared.u32 r3496, [r5928+6400];
ld.shared.u32 r3076, [r5928+6656];
ld.shared.u32 r3692, [r5928+6912];
ld.shared.u32 r2930, [r5928+7168];
ld.shared.u32 r3546, [r5928+7424];
ld.shared.u32 r3126, [r5928+7680];
ld.shared.u32 r3742, [r5928+7936];
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
and.b32 r5929, r5921, 32;
bfe.u32 r5930, r5921, 5, 1;
shl.b32 r5931, r5921, 2;
and.b32 r5932, r5931, 124;
add.s32 r5933, r5925, r5932;
cvt.rn.f32.u32 f848, r5930;
mul.f32 f849, f848, 0f3DC90FDB;
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
and.b32 r5934, r5923, 4096;
add.s32 r5935, r5933, r5934;
st.shared.u32 [r5935], r4383;
st.shared.u32 [r5935+128], r4587;
st.shared.u32 [r5935+256], r4624;
st.shared.u32 [r5935+384], r4661;
st.shared.u32 [r5935+512], r4698;
st.shared.u32 [r5935+640], r4735;
st.shared.u32 [r5935+768], r4772;
st.shared.u32 [r5935+896], r4809;
st.shared.u32 [r5935+1024], r4846;
st.shared.u32 [r5935+1152], r4883;
st.shared.u32 [r5935+1280], r4920;
st.shared.u32 [r5935+1408], r4957;
st.shared.u32 [r5935+1536], r4994;
st.shared.u32 [r5935+1664], r5031;
st.shared.u32 [r5935+1792], r5068;
st.shared.u32 [r5935+1920], r5105;
st.shared.u32 [r5935+2048], r5142;
st.shared.u32 [r5935+2176], r5179;
st.shared.u32 [r5935+2304], r5216;
st.shared.u32 [r5935+2432], r5253;
st.shared.u32 [r5935+2560], r5290;
st.shared.u32 [r5935+2688], r5327;
st.shared.u32 [r5935+2816], r5364;
st.shared.u32 [r5935+2944], r5401;
st.shared.u32 [r5935+3072], r5438;
st.shared.u32 [r5935+3200], r5475;
st.shared.u32 [r5935+3328], r5512;
st.shared.u32 [r5935+3456], r5549;
st.shared.u32 [r5935+3584], r5586;
st.shared.u32 [r5935+3712], r5623;
st.shared.u32 [r5935+3840], r5660;
st.shared.u32 [r5935+3968], r5697;
barrier.sync 0;
mad.lo.s32 r5936, r5929, -124, r5935;
ld.shared.u32 r5726, [r5936];
ld.shared.u32 r5738, [r5936+256];
ld.shared.u32 r5750, [r5936+512];
ld.shared.u32 r5762, [r5936+768];
ld.shared.u32 r5774, [r5936+1024];
ld.shared.u32 r5786, [r5936+1280];
ld.shared.u32 r5798, [r5936+1536];
ld.shared.u32 r5810, [r5936+1792];
ld.shared.u32 r5822, [r5936+2048];
ld.shared.u32 r5834, [r5936+2304];
ld.shared.u32 r5846, [r5936+2560];
ld.shared.u32 r5858, [r5936+2816];
ld.shared.u32 r5870, [r5936+3072];
ld.shared.u32 r5882, [r5936+3328];
ld.shared.u32 r5894, [r5936+3584];
ld.shared.u32 r5906, [r5936+3840];
ld.shared.u32 r5727, [r5936+4096];
ld.shared.u32 r5739, [r5936+4352];
ld.shared.u32 r5751, [r5936+4608];
ld.shared.u32 r5763, [r5936+4864];
ld.shared.u32 r5775, [r5936+5120];
ld.shared.u32 r5787, [r5936+5376];
ld.shared.u32 r5799, [r5936+5632];
ld.shared.u32 r5811, [r5936+5888];
ld.shared.u32 r5823, [r5936+6144];
ld.shared.u32 r5835, [r5936+6400];
ld.shared.u32 r5847, [r5936+6656];
ld.shared.u32 r5859, [r5936+6912];
ld.shared.u32 r5871, [r5936+7168];
ld.shared.u32 r5883, [r5936+7424];
ld.shared.u32 r5895, [r5936+7680];
ld.shared.u32 r5907, [r5936+7936];
barrier.sync 0;
st.shared.u32 [r5935], r4386;
st.shared.u32 [r5935+128], r4594;
st.shared.u32 [r5935+256], r4631;
st.shared.u32 [r5935+384], r4668;
st.shared.u32 [r5935+512], r4705;
st.shared.u32 [r5935+640], r4742;
st.shared.u32 [r5935+768], r4779;
st.shared.u32 [r5935+896], r4816;
st.shared.u32 [r5935+1024], r4853;
st.shared.u32 [r5935+1152], r4890;
st.shared.u32 [r5935+1280], r4927;
st.shared.u32 [r5935+1408], r4964;
st.shared.u32 [r5935+1536], r5001;
st.shared.u32 [r5935+1664], r5038;
st.shared.u32 [r5935+1792], r5075;
st.shared.u32 [r5935+1920], r5112;
st.shared.u32 [r5935+2048], r5149;
st.shared.u32 [r5935+2176], r5186;
st.shared.u32 [r5935+2304], r5223;
st.shared.u32 [r5935+2432], r5260;
st.shared.u32 [r5935+2560], r5297;
st.shared.u32 [r5935+2688], r5334;
st.shared.u32 [r5935+2816], r5371;
st.shared.u32 [r5935+2944], r5408;
st.shared.u32 [r5935+3072], r5445;
st.shared.u32 [r5935+3200], r5482;
st.shared.u32 [r5935+3328], r5519;
st.shared.u32 [r5935+3456], r5556;
st.shared.u32 [r5935+3584], r5593;
st.shared.u32 [r5935+3712], r5630;
st.shared.u32 [r5935+3840], r5667;
st.shared.u32 [r5935+3968], r5704;
barrier.sync 0;
ld.shared.u32 r5729, [r5936];
ld.shared.u32 r5741, [r5936+256];
ld.shared.u32 r5753, [r5936+512];
ld.shared.u32 r5765, [r5936+768];
ld.shared.u32 r5777, [r5936+1024];
ld.shared.u32 r5789, [r5936+1280];
ld.shared.u32 r5801, [r5936+1536];
ld.shared.u32 r5813, [r5936+1792];
ld.shared.u32 r5825, [r5936+2048];
ld.shared.u32 r5837, [r5936+2304];
ld.shared.u32 r5849, [r5936+2560];
ld.shared.u32 r5861, [r5936+2816];
ld.shared.u32 r5873, [r5936+3072];
ld.shared.u32 r5885, [r5936+3328];
ld.shared.u32 r5897, [r5936+3584];
ld.shared.u32 r5909, [r5936+3840];
ld.shared.u32 r5730, [r5936+4096];
ld.shared.u32 r5742, [r5936+4352];
ld.shared.u32 r5754, [r5936+4608];
ld.shared.u32 r5766, [r5936+4864];
ld.shared.u32 r5778, [r5936+5120];
ld.shared.u32 r5790, [r5936+5376];
ld.shared.u32 r5802, [r5936+5632];
ld.shared.u32 r5814, [r5936+5888];
ld.shared.u32 r5826, [r5936+6144];
ld.shared.u32 r5838, [r5936+6400];
ld.shared.u32 r5850, [r5936+6656];
ld.shared.u32 r5862, [r5936+6912];
ld.shared.u32 r5874, [r5936+7168];
ld.shared.u32 r5886, [r5936+7424];
ld.shared.u32 r5898, [r5936+7680];
ld.shared.u32 r5910, [r5936+7936];
{
add.f16x2 %0, r5726, r5727;
}
{
add.f16x2 %1, r5729, r5730;
}
{
sub.f16x2 %32, r5726, r5727;
}
{
sub.f16x2 %33, r5729, r5730;
}
{
add.f16x2 %2, r5738, r5739;
}
{
add.f16x2 %3, r5741, r5742;
}
{
sub.f16x2 %34, r5738, r5739;
}
{
sub.f16x2 %35, r5741, r5742;
}
{
add.f16x2 %4, r5750, r5751;
}
{
add.f16x2 %5, r5753, r5754;
}
{
sub.f16x2 %36, r5750, r5751;
}
{
sub.f16x2 %37, r5753, r5754;
}
{
add.f16x2 %6, r5762, r5763;
}
{
add.f16x2 %7, r5765, r5766;
}
{
sub.f16x2 %38, r5762, r5763;
}
{
sub.f16x2 %39, r5765, r5766;
}
{
add.f16x2 %8, r5774, r5775;
}
{
add.f16x2 %9, r5777, r5778;
}
{
sub.f16x2 %40, r5774, r5775;
}
{
sub.f16x2 %41, r5777, r5778;
}
{
add.f16x2 %10, r5786, r5787;
}
{
add.f16x2 %11, r5789, r5790;
}
{
sub.f16x2 %42, r5786, r5787;
}
{
sub.f16x2 %43, r5789, r5790;
}
{
add.f16x2 %12, r5798, r5799;
}
{
add.f16x2 %13, r5801, r5802;
}
{
sub.f16x2 %44, r5798, r5799;
}
{
sub.f16x2 %45, r5801, r5802;
}
{
add.f16x2 %14, r5810, r5811;
}
{
add.f16x2 %15, r5813, r5814;
}
{
sub.f16x2 %46, r5810, r5811;
}
{
sub.f16x2 %47, r5813, r5814;
}
{
add.f16x2 %16, r5822, r5823;
}
{
add.f16x2 %17, r5825, r5826;
}
{
sub.f16x2 %48, r5822, r5823;
}
{
sub.f16x2 %49, r5825, r5826;
}
{
add.f16x2 %18, r5834, r5835;
}
{
add.f16x2 %19, r5837, r5838;
}
{
sub.f16x2 %50, r5834, r5835;
}
{
sub.f16x2 %51, r5837, r5838;
}
{
add.f16x2 %20, r5846, r5847;
}
{
add.f16x2 %21, r5849, r5850;
}
{
sub.f16x2 %52, r5846, r5847;
}
{
sub.f16x2 %53, r5849, r5850;
}
{
add.f16x2 %22, r5858, r5859;
}
{
add.f16x2 %23, r5861, r5862;
}
{
sub.f16x2 %54, r5858, r5859;
}
{
sub.f16x2 %55, r5861, r5862;
}
{
add.f16x2 %24, r5870, r5871;
}
{
add.f16x2 %25, r5873, r5874;
}
{
sub.f16x2 %56, r5870, r5871;
}
{
sub.f16x2 %57, r5873, r5874;
}
{
add.f16x2 %26, r5882, r5883;
}
{
add.f16x2 %27, r5885, r5886;
}
{
sub.f16x2 %58, r5882, r5883;
}
{
sub.f16x2 %59, r5885, r5886;
}
{
add.f16x2 %28, r5894, r5895;
}
{
add.f16x2 %29, r5897, r5898;
}
{
sub.f16x2 %60, r5894, r5895;
}
{
sub.f16x2 %61, r5897, r5898;
}
{
add.f16x2 %30, r5906, r5907;
}
{
add.f16x2 %31, r5909, r5910;
}
{
sub.f16x2 %62, r5906, r5907;
}
{
sub.f16x2 %63, r5909, r5910;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)), "=r"(__HALF2_TO_UI(rmem[27].x)), "=r"(__HALF2_TO_UI(rmem[27].y)), "=r"(__HALF2_TO_UI(rmem[28].x)), "=r"(__HALF2_TO_UI(rmem[28].y)), "=r"(__HALF2_TO_UI(rmem[29].x)), "=r"(__HALF2_TO_UI(rmem[29].y)), "=r"(__HALF2_TO_UI(rmem[30].x)), "=r"(__HALF2_TO_UI(rmem[30].y)), "=r"(__HALF2_TO_UI(rmem[31].x)), "=r"(__HALF2_TO_UI(rmem[31].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[31].y)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[28].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[27].y)), "r"(__HALF2_TO_UI(rmem[30].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[29].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[31].x)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[28].x)), "r"(__HALF2_TO_UI(rmem[27].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[30].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[29].x)), "r"(__HALF2_TO_UI(rmem[24].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<843, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .pred p<3>;
.reg .b16 rs<3>;
.reg .f32 f<71>;
.reg .b32 r<894>;
.reg .b64 rd<2>;
{
add.f16x2 r11, %9, %13;
}
{
add.f16x2 r14, %10, %14;
}
{
sub.f16x2 r17, %9, %13;
}
{
sub.f16x2 r20, %10, %14;
}
{
add.f16x2 r23, %11, %15;
}
{
add.f16x2 r26, %12, %16;
}
{
sub.f16x2 r29, %11, %15;
}
{
sub.f16x2 r32, %12, %16;
}
{
neg.f16x2 r35, r29;
}
{
add.f16x2 r37, r11, r23;
}
{
add.f16x2 r40, r14, r26;
}
{
sub.f16x2 r43, r11, r23;
}
{
sub.f16x2 r46, r14, r26;
}
{
add.f16x2 r49, r17, r32;
}
{
add.f16x2 r52, r20, r35;
}
{
sub.f16x2 r55, r17, r32;
}
{
sub.f16x2 r58, r20, r35;
}
mov.u32 r9, %tid.x;
and.b32 r10, r9, 511;
cvt.rn.f32.u32 f5, r10;
mul.f32 f1, f5, 0f3B490FDB;
cvt.u16.u32 rs2, r9;
and.b16 rs1, rs2, 511;
mov.f32 f70, 0f3BC90F88;
setp.eq.s16 p1, rs1, 510;
@p1 bra LBB6_4;
setp.ne.s16 p2, rs1, 511;
@p2 bra LBB6_3;
mov.f32 f70, 0f3B490FC6;
bra.uni LBB6_4;
LBB6_3:
cos.approx.f32 f70, f1;
LBB6_4:
sin.approx.f32 f57, f1;
neg.f32 f8, f57;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f70;
cvt.rn.f16.f32 high, f8;
mov.b32 r61, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r61;
mov.b32 r64, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r61;
mov.b32 r66, {high, high};
}
{
mul.f16x2 r68, r52, r66;
}
{
neg.f16x2 r71, r68;
}
{
fma.rn.f16x2 r73, r49, r64, r71;
}
{
mul.f16x2 r77, r49, r66;
}
{
fma.rn.f16x2 r80, r52, r64, r77;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r61;
mov.b32 r84, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r61;
mov.b32 r86, {high, high};
}
mov.f32 f53, 0fBF800000;
mov.f32 f54, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r88, {low, high};
}
{
mul.f16x2 r89, r86, r88;
}
{
mul.f16x2 r92, r61, r84;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r61;
mov.b32 r95, {high, low};
}
{
fma.rn.f16x2 r97, r89, r95, r92;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r97;
mov.b32 r101, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r97;
mov.b32 r103, {high, high};
}
{
mul.f16x2 r105, r46, r103;
}
{
neg.f16x2 r108, r105;
}
{
fma.rn.f16x2 r110, r43, r101, r108;
}
{
mul.f16x2 r114, r43, r103;
}
{
fma.rn.f16x2 r117, r46, r101, r114;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r61;
mov.b32 r121, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r61;
mov.b32 r123, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r125, {low, high};
}
{
mul.f16x2 r126, r123, r125;
}
{
mul.f16x2 r129, r97, r121;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r97;
mov.b32 r132, {high, low};
}
{
fma.rn.f16x2 r134, r126, r132, r129;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r134;
mov.b32 r138, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r134;
mov.b32 r140, {high, high};
}
{
mul.f16x2 r142, r58, r140;
}
{
neg.f16x2 r145, r142;
}
{
fma.rn.f16x2 r147, r55, r138, r145;
}
{
mul.f16x2 r151, r55, r140;
}
{
fma.rn.f16x2 r154, r58, r138, r151;
}
barrier.sync 0;
mov.u32 r855, %tid.y;
shl.b32 r856, r855, 14;
mov.u32 r857, %8;
add.s32 r858, r857, r856;
shl.b32 r859, r9, 5;
and.b32 r860, r859, -16384;
add.s32 r861, r858, r860;
shl.b32 r862, r10, 5;
add.s32 r863, r861, r862;
st.shared.v4.f32 [r863], {r37, r40, r73, r80};
st.shared.v4.f32 [r863+16], {r110, r117, r147, r154};
barrier.sync 0;
mad.lo.s32 r864, r10, -24, r863;
ld.shared.u32 r176, [r864];
ld.shared.u32 r179, [r864+4];
ld.shared.u32 r188, [r864+4096];
ld.shared.u32 r191, [r864+4100];
ld.shared.u32 r177, [r864+8192];
ld.shared.u32 r180, [r864+8196];
ld.shared.u32 r189, [r864+12288];
ld.shared.u32 r192, [r864+12292];
{
add.f16x2 r175, r176, r177;
}
{
add.f16x2 r178, r179, r180;
}
{
sub.f16x2 r181, r176, r177;
}
{
sub.f16x2 r184, r179, r180;
}
{
add.f16x2 r187, r188, r189;
}
{
add.f16x2 r190, r191, r192;
}
{
sub.f16x2 r193, r188, r189;
}
{
sub.f16x2 r196, r191, r192;
}
{
neg.f16x2 r199, r193;
}
{
add.f16x2 r201, r175, r187;
}
{
add.f16x2 r204, r178, r190;
}
{
sub.f16x2 r207, r175, r187;
}
{
sub.f16x2 r210, r178, r190;
}
{
add.f16x2 r213, r181, r196;
}
{
add.f16x2 r216, r184, r199;
}
{
sub.f16x2 r219, r181, r196;
}
{
sub.f16x2 r222, r184, r199;
}
and.b32 r865, r9, 508;
bfe.u32 r866, r9, 2, 7;
cvt.rn.f32.u32 f58, r866;
mul.f32 f59, f58, 0f3C490FDB;
cos.approx.f32 f17, f59;
sin.approx.f32 f60, f59;
neg.f32 f18, f60;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f17;
cvt.rn.f16.f32 high, f18;
mov.b32 r225, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r225;
mov.b32 r228, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r225;
mov.b32 r230, {high, high};
}
{
mul.f16x2 r232, r216, r230;
}
{
neg.f16x2 r235, r232;
}
{
fma.rn.f16x2 r237, r213, r228, r235;
}
{
mul.f16x2 r241, r213, r230;
}
{
fma.rn.f16x2 r244, r216, r228, r241;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r225;
mov.b32 r248, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r225;
mov.b32 r250, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r252, {low, high};
}
{
mul.f16x2 r253, r250, r252;
}
{
mul.f16x2 r256, r225, r248;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r225;
mov.b32 r259, {high, low};
}
{
fma.rn.f16x2 r261, r253, r259, r256;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r261;
mov.b32 r265, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r261;
mov.b32 r267, {high, high};
}
{
mul.f16x2 r269, r210, r267;
}
{
neg.f16x2 r272, r269;
}
{
fma.rn.f16x2 r274, r207, r265, r272;
}
{
mul.f16x2 r278, r207, r267;
}
{
fma.rn.f16x2 r281, r210, r265, r278;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r225;
mov.b32 r285, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r225;
mov.b32 r287, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r289, {low, high};
}
{
mul.f16x2 r290, r287, r289;
}
{
mul.f16x2 r293, r261, r285;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r261;
mov.b32 r296, {high, low};
}
{
fma.rn.f16x2 r298, r290, r296, r293;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r298;
mov.b32 r302, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r298;
mov.b32 r304, {high, high};
}
{
mul.f16x2 r306, r222, r304;
}
{
neg.f16x2 r309, r306;
}
{
fma.rn.f16x2 r311, r219, r302, r309;
}
{
mul.f16x2 r315, r219, r304;
}
{
fma.rn.f16x2 r318, r222, r302, r315;
}
shl.b32 r867, r9, 3;
and.b32 r868, r867, 24;
add.s32 r869, r861, r868;
barrier.sync 0;
and.b32 r870, r859, 16256;
add.s32 r871, r869, r870;
st.shared.u32 [r871], r201;
st.shared.u32 [r871+4], r204;
st.shared.u32 [r871+32], r237;
st.shared.u32 [r871+36], r244;
st.shared.u32 [r871+64], r274;
st.shared.u32 [r871+68], r281;
st.shared.u32 [r871+96], r311;
st.shared.u32 [r871+100], r318;
barrier.sync 0;
mad.lo.s32 r872, r865, -24, r871;
ld.shared.u32 r340, [r872];
ld.shared.u32 r343, [r872+4];
ld.shared.u32 r352, [r872+4096];
ld.shared.u32 r355, [r872+4100];
ld.shared.u32 r341, [r872+8192];
ld.shared.u32 r344, [r872+8196];
ld.shared.u32 r353, [r872+12288];
ld.shared.u32 r356, [r872+12292];
{
add.f16x2 r339, r340, r341;
}
{
add.f16x2 r342, r343, r344;
}
{
sub.f16x2 r345, r340, r341;
}
{
sub.f16x2 r348, r343, r344;
}
{
add.f16x2 r351, r352, r353;
}
{
add.f16x2 r354, r355, r356;
}
{
sub.f16x2 r357, r352, r353;
}
{
sub.f16x2 r360, r355, r356;
}
{
neg.f16x2 r363, r357;
}
{
add.f16x2 r365, r339, r351;
}
{
add.f16x2 r368, r342, r354;
}
{
sub.f16x2 r371, r339, r351;
}
{
sub.f16x2 r374, r342, r354;
}
{
add.f16x2 r377, r345, r360;
}
{
add.f16x2 r380, r348, r363;
}
{
sub.f16x2 r383, r345, r360;
}
{
sub.f16x2 r386, r348, r363;
}
and.b32 r873, r9, 496;
bfe.u32 r874, r9, 4, 5;
cvt.rn.f32.u32 f61, r874;
mul.f32 f62, f61, 0f3D490FDB;
cos.approx.f32 f27, f62;
sin.approx.f32 f63, f62;
neg.f32 f28, f63;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f27;
cvt.rn.f16.f32 high, f28;
mov.b32 r389, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r389;
mov.b32 r392, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r389;
mov.b32 r394, {high, high};
}
{
mul.f16x2 r396, r380, r394;
}
{
neg.f16x2 r399, r396;
}
{
fma.rn.f16x2 r401, r377, r392, r399;
}
{
mul.f16x2 r405, r377, r394;
}
{
fma.rn.f16x2 r408, r380, r392, r405;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r389;
mov.b32 r412, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r389;
mov.b32 r414, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r416, {low, high};
}
{
mul.f16x2 r417, r414, r416;
}
{
mul.f16x2 r420, r389, r412;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r389;
mov.b32 r423, {high, low};
}
{
fma.rn.f16x2 r425, r417, r423, r420;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r425;
mov.b32 r429, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r425;
mov.b32 r431, {high, high};
}
{
mul.f16x2 r433, r374, r431;
}
{
neg.f16x2 r436, r433;
}
{
fma.rn.f16x2 r438, r371, r429, r436;
}
{
mul.f16x2 r442, r371, r431;
}
{
fma.rn.f16x2 r445, r374, r429, r442;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r389;
mov.b32 r449, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r389;
mov.b32 r451, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r453, {low, high};
}
{
mul.f16x2 r454, r451, r453;
}
{
mul.f16x2 r457, r425, r449;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r425;
mov.b32 r460, {high, low};
}
{
fma.rn.f16x2 r462, r454, r460, r457;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r462;
mov.b32 r466, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r462;
mov.b32 r468, {high, high};
}
{
mul.f16x2 r470, r386, r468;
}
{
neg.f16x2 r473, r470;
}
{
fma.rn.f16x2 r475, r383, r466, r473;
}
{
mul.f16x2 r479, r383, r468;
}
{
fma.rn.f16x2 r482, r386, r466, r479;
}
and.b32 r875, r867, 120;
add.s32 r876, r861, r875;
barrier.sync 0;
and.b32 r877, r859, 15872;
add.s32 r878, r876, r877;
st.shared.u32 [r878], r365;
st.shared.u32 [r878+4], r368;
st.shared.u32 [r878+128], r401;
st.shared.u32 [r878+132], r408;
st.shared.u32 [r878+256], r438;
st.shared.u32 [r878+260], r445;
st.shared.u32 [r878+384], r475;
st.shared.u32 [r878+388], r482;
barrier.sync 0;
mad.lo.s32 r879, r873, -24, r878;
ld.shared.u32 r504, [r879];
ld.shared.u32 r507, [r879+4];
ld.shared.u32 r516, [r879+4096];
ld.shared.u32 r519, [r879+4100];
ld.shared.u32 r505, [r879+8192];
ld.shared.u32 r508, [r879+8196];
ld.shared.u32 r517, [r879+12288];
ld.shared.u32 r520, [r879+12292];
{
add.f16x2 r503, r504, r505;
}
{
add.f16x2 r506, r507, r508;
}
{
sub.f16x2 r509, r504, r505;
}
{
sub.f16x2 r512, r507, r508;
}
{
add.f16x2 r515, r516, r517;
}
{
add.f16x2 r518, r519, r520;
}
{
sub.f16x2 r521, r516, r517;
}
{
sub.f16x2 r524, r519, r520;
}
{
neg.f16x2 r527, r521;
}
{
add.f16x2 r529, r503, r515;
}
{
add.f16x2 r532, r506, r518;
}
{
sub.f16x2 r535, r503, r515;
}
{
sub.f16x2 r538, r506, r518;
}
{
add.f16x2 r541, r509, r524;
}
{
add.f16x2 r544, r512, r527;
}
{
sub.f16x2 r547, r509, r524;
}
{
sub.f16x2 r550, r512, r527;
}
and.b32 r880, r9, 448;
bfe.u32 r881, r9, 6, 3;
cvt.rn.f32.u32 f64, r881;
mul.f32 f65, f64, 0f3E490FDB;
cos.approx.f32 f37, f65;
sin.approx.f32 f66, f65;
neg.f32 f38, f66;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
mov.b32 r553, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r553;
mov.b32 r556, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r553;
mov.b32 r558, {high, high};
}
{
mul.f16x2 r560, r544, r558;
}
{
neg.f16x2 r563, r560;
}
{
fma.rn.f16x2 r565, r541, r556, r563;
}
{
mul.f16x2 r569, r541, r558;
}
{
fma.rn.f16x2 r572, r544, r556, r569;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r553;
mov.b32 r576, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r553;
mov.b32 r578, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r580, {low, high};
}
{
mul.f16x2 r581, r578, r580;
}
{
mul.f16x2 r584, r553, r576;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r553;
mov.b32 r587, {high, low};
}
{
fma.rn.f16x2 r589, r581, r587, r584;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r589;
mov.b32 r593, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r589;
mov.b32 r595, {high, high};
}
{
mul.f16x2 r597, r538, r595;
}
{
neg.f16x2 r600, r597;
}
{
fma.rn.f16x2 r602, r535, r593, r600;
}
{
mul.f16x2 r606, r535, r595;
}
{
fma.rn.f16x2 r609, r538, r593, r606;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r553;
mov.b32 r613, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r553;
mov.b32 r615, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r617, {low, high};
}
{
mul.f16x2 r618, r615, r617;
}
{
mul.f16x2 r621, r589, r613;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r589;
mov.b32 r624, {high, low};
}
{
fma.rn.f16x2 r626, r618, r624, r621;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r626;
mov.b32 r630, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r626;
mov.b32 r632, {high, high};
}
{
mul.f16x2 r634, r550, r632;
}
{
neg.f16x2 r637, r634;
}
{
fma.rn.f16x2 r639, r547, r630, r637;
}
{
mul.f16x2 r643, r547, r632;
}
{
fma.rn.f16x2 r646, r550, r630, r643;
}
and.b32 r882, r867, 504;
add.s32 r883, r861, r882;
barrier.sync 0;
and.b32 r884, r859, 14336;
add.s32 r885, r883, r884;
st.shared.u32 [r885], r529;
st.shared.u32 [r885+4], r532;
st.shared.u32 [r885+512], r565;
st.shared.u32 [r885+516], r572;
st.shared.u32 [r885+1024], r602;
st.shared.u32 [r885+1028], r609;
st.shared.u32 [r885+1536], r639;
st.shared.u32 [r885+1540], r646;
barrier.sync 0;
mad.lo.s32 r886, r880, -24, r885;
ld.shared.u32 r668, [r886];
ld.shared.u32 r671, [r886+4];
ld.shared.u32 r680, [r886+4096];
ld.shared.u32 r683, [r886+4100];
ld.shared.u32 r669, [r886+8192];
ld.shared.u32 r672, [r886+8196];
ld.shared.u32 r681, [r886+12288];
ld.shared.u32 r684, [r886+12292];
{
add.f16x2 r667, r668, r669;
}
{
add.f16x2 r670, r671, r672;
}
{
sub.f16x2 r673, r668, r669;
}
{
sub.f16x2 r676, r671, r672;
}
{
add.f16x2 r679, r680, r681;
}
{
add.f16x2 r682, r683, r684;
}
{
sub.f16x2 r685, r680, r681;
}
{
sub.f16x2 r688, r683, r684;
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
and.b32 r887, r9, 256;
bfe.u32 r888, r9, 8, 1;
cvt.rn.f32.u32 f67, r888;
mul.f32 f68, f67, 0f3F490FDB;
cos.approx.f32 f47, f68;
sin.approx.f32 f69, f68;
neg.f32 f48, f69;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f47;
cvt.rn.f16.f32 high, f48;
mov.b32 r717, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r717;
mov.b32 r720, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r717;
mov.b32 r722, {high, high};
}
{
mul.f16x2 r724, r708, r722;
}
{
neg.f16x2 r727, r724;
}
{
fma.rn.f16x2 r729, r705, r720, r727;
}
{
mul.f16x2 r733, r705, r722;
}
{
fma.rn.f16x2 r736, r708, r720, r733;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r717;
mov.b32 r740, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r717;
mov.b32 r742, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r744, {low, high};
}
{
mul.f16x2 r745, r742, r744;
}
{
mul.f16x2 r748, r717, r740;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r717;
mov.b32 r751, {high, low};
}
{
fma.rn.f16x2 r753, r745, r751, r748;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r753;
mov.b32 r757, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r753;
mov.b32 r759, {high, high};
}
{
mul.f16x2 r761, r702, r759;
}
{
neg.f16x2 r764, r761;
}
{
fma.rn.f16x2 r766, r699, r757, r764;
}
{
mul.f16x2 r770, r699, r759;
}
{
fma.rn.f16x2 r773, r702, r757, r770;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r717;
mov.b32 r777, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r717;
mov.b32 r779, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r781, {low, high};
}
{
mul.f16x2 r782, r779, r781;
}
{
mul.f16x2 r785, r753, r777;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r753;
mov.b32 r788, {high, low};
}
{
fma.rn.f16x2 r790, r782, r788, r785;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r790;
mov.b32 r794, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r790;
mov.b32 r796, {high, high};
}
{
mul.f16x2 r798, r714, r796;
}
{
neg.f16x2 r801, r798;
}
{
fma.rn.f16x2 r803, r711, r794, r801;
}
{
mul.f16x2 r807, r711, r796;
}
{
fma.rn.f16x2 r810, r714, r794, r807;
}
and.b32 r889, r867, 2040;
add.s32 r890, r861, r889;
barrier.sync 0;
and.b32 r891, r859, 8192;
add.s32 r892, r890, r891;
st.shared.u32 [r892], r693;
st.shared.u32 [r892+4], r696;
st.shared.u32 [r892+2048], r729;
st.shared.u32 [r892+2052], r736;
st.shared.u32 [r892+4096], r766;
st.shared.u32 [r892+4100], r773;
st.shared.u32 [r892+6144], r803;
st.shared.u32 [r892+6148], r810;
barrier.sync 0;
mad.lo.s32 r893, r887, -24, r892;
ld.shared.u32 r832, [r893];
ld.shared.u32 r835, [r893+4];
ld.shared.u32 r844, [r893+4096];
ld.shared.u32 r847, [r893+4100];
ld.shared.u32 r833, [r893+8192];
ld.shared.u32 r836, [r893+8196];
ld.shared.u32 r845, [r893+12288];
ld.shared.u32 r848, [r893+12292];
{
add.f16x2 %0, r832, r833;
}
{
add.f16x2 %1, r835, r836;
}
{
sub.f16x2 %4, r832, r833;
}
{
sub.f16x2 %5, r835, r836;
}
{
add.f16x2 %2, r844, r845;
}
{
add.f16x2 %3, r847, r848;
}
{
sub.f16x2 %6, r844, r845;
}
{
sub.f16x2 %7, r847, r848;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<844, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .pred p<3>;
.reg .b16 rs<3>;
.reg .f32 f<71>;
.reg .b32 r<894>;
.reg .b64 rd<2>;
{
add.f16x2 r11, %9, %13;
}
{
add.f16x2 r14, %10, %14;
}
{
sub.f16x2 r17, %9, %13;
}
{
sub.f16x2 r20, %10, %14;
}
{
add.f16x2 r23, %11, %15;
}
{
add.f16x2 r26, %12, %16;
}
{
sub.f16x2 r29, %11, %15;
}
{
sub.f16x2 r32, %12, %16;
}
{
neg.f16x2 r35, r29;
}
{
add.f16x2 r37, r11, r23;
}
{
add.f16x2 r40, r14, r26;
}
{
sub.f16x2 r43, r11, r23;
}
{
sub.f16x2 r46, r14, r26;
}
{
add.f16x2 r49, r17, r32;
}
{
add.f16x2 r52, r20, r35;
}
{
sub.f16x2 r55, r17, r32;
}
{
sub.f16x2 r58, r20, r35;
}
mov.u32 r9, %tid.x;
and.b32 r10, r9, 511;
cvt.rn.f32.u32 f5, r10;
mul.f32 f1, f5, 0f3B490FDB;
cvt.u16.u32 rs2, r9;
and.b16 rs1, rs2, 511;
mov.f32 f70, 0f3BC90F88;
setp.eq.s16 p1, rs1, 510;
@p1 bra LBB7_4;
setp.ne.s16 p2, rs1, 511;
@p2 bra LBB7_3;
mov.f32 f70, 0f3B490FC6;
bra.uni LBB7_4;
LBB7_3:
cos.approx.f32 f70, f1;
LBB7_4:
mov.u32 r855, %tid.y;
shl.b32 r856, r855, 13;
mov.u32 r857, %8;
add.s32 r858, r857, r856;
shl.b32 r859, r9, 4;
and.b32 r860, r859, -8192;
add.s32 r861, r858, r860;
sin.approx.f32 f57, f1;
neg.f32 f8, f57;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f70;
cvt.rn.f16.f32 high, f8;
mov.b32 r61, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r61;
mov.b32 r64, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r61;
mov.b32 r66, {high, high};
}
{
mul.f16x2 r68, r52, r66;
}
{
neg.f16x2 r71, r68;
}
{
fma.rn.f16x2 r73, r49, r64, r71;
}
{
mul.f16x2 r77, r49, r66;
}
{
fma.rn.f16x2 r80, r52, r64, r77;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r61;
mov.b32 r84, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r61;
mov.b32 r86, {high, high};
}
mov.f32 f53, 0fBF800000;
mov.f32 f54, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r88, {low, high};
}
{
mul.f16x2 r89, r86, r88;
}
{
mul.f16x2 r92, r61, r84;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r61;
mov.b32 r95, {high, low};
}
{
fma.rn.f16x2 r97, r89, r95, r92;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r97;
mov.b32 r101, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r97;
mov.b32 r103, {high, high};
}
{
mul.f16x2 r105, r46, r103;
}
{
neg.f16x2 r108, r105;
}
{
fma.rn.f16x2 r110, r43, r101, r108;
}
{
mul.f16x2 r114, r43, r103;
}
{
fma.rn.f16x2 r117, r46, r101, r114;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r61;
mov.b32 r121, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r61;
mov.b32 r123, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r125, {low, high};
}
{
mul.f16x2 r126, r123, r125;
}
{
mul.f16x2 r129, r97, r121;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r97;
mov.b32 r132, {high, low};
}
{
fma.rn.f16x2 r134, r126, r132, r129;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r134;
mov.b32 r138, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r134;
mov.b32 r140, {high, high};
}
{
mul.f16x2 r142, r58, r140;
}
{
neg.f16x2 r145, r142;
}
{
fma.rn.f16x2 r147, r55, r138, r145;
}
{
mul.f16x2 r151, r55, r140;
}
{
fma.rn.f16x2 r154, r58, r138, r151;
}
barrier.sync 0;
shl.b32 r862, r10, 4;
add.s32 r863, r861, r862;
st.shared.v4.f32 [r863], {r37, r73, r110, r147};
barrier.sync 0;
mad.lo.s32 r864, r10, -12, r863;
ld.shared.u32 r176, [r864];
ld.shared.u32 r188, [r864+2048];
ld.shared.u32 r177, [r864+4096];
ld.shared.u32 r189, [r864+6144];
barrier.sync 0;
st.shared.v4.f32 [r863], {r40, r80, r117, r154};
barrier.sync 0;
ld.shared.u32 r179, [r864];
ld.shared.u32 r191, [r864+2048];
ld.shared.u32 r180, [r864+4096];
ld.shared.u32 r192, [r864+6144];
{
add.f16x2 r175, r176, r177;
}
{
add.f16x2 r178, r179, r180;
}
{
sub.f16x2 r181, r176, r177;
}
{
sub.f16x2 r184, r179, r180;
}
{
add.f16x2 r187, r188, r189;
}
{
add.f16x2 r190, r191, r192;
}
{
sub.f16x2 r193, r188, r189;
}
{
sub.f16x2 r196, r191, r192;
}
{
neg.f16x2 r199, r193;
}
{
add.f16x2 r201, r175, r187;
}
{
add.f16x2 r204, r178, r190;
}
{
sub.f16x2 r207, r175, r187;
}
{
sub.f16x2 r210, r178, r190;
}
{
add.f16x2 r213, r181, r196;
}
{
add.f16x2 r216, r184, r199;
}
{
sub.f16x2 r219, r181, r196;
}
{
sub.f16x2 r222, r184, r199;
}
and.b32 r865, r9, 508;
bfe.u32 r866, r9, 2, 7;
shl.b32 r867, r9, 2;
and.b32 r868, r867, 12;
add.s32 r869, r861, r868;
cvt.rn.f32.u32 f58, r866;
mul.f32 f59, f58, 0f3C490FDB;
cos.approx.f32 f17, f59;
sin.approx.f32 f60, f59;
neg.f32 f18, f60;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f17;
cvt.rn.f16.f32 high, f18;
mov.b32 r225, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r225;
mov.b32 r228, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r225;
mov.b32 r230, {high, high};
}
{
mul.f16x2 r232, r216, r230;
}
{
neg.f16x2 r235, r232;
}
{
fma.rn.f16x2 r237, r213, r228, r235;
}
{
mul.f16x2 r241, r213, r230;
}
{
fma.rn.f16x2 r244, r216, r228, r241;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r225;
mov.b32 r248, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r225;
mov.b32 r250, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r252, {low, high};
}
{
mul.f16x2 r253, r250, r252;
}
{
mul.f16x2 r256, r225, r248;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r225;
mov.b32 r259, {high, low};
}
{
fma.rn.f16x2 r261, r253, r259, r256;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r261;
mov.b32 r265, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r261;
mov.b32 r267, {high, high};
}
{
mul.f16x2 r269, r210, r267;
}
{
neg.f16x2 r272, r269;
}
{
fma.rn.f16x2 r274, r207, r265, r272;
}
{
mul.f16x2 r278, r207, r267;
}
{
fma.rn.f16x2 r281, r210, r265, r278;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r225;
mov.b32 r285, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r225;
mov.b32 r287, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r289, {low, high};
}
{
mul.f16x2 r290, r287, r289;
}
{
mul.f16x2 r293, r261, r285;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r261;
mov.b32 r296, {high, low};
}
{
fma.rn.f16x2 r298, r290, r296, r293;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r298;
mov.b32 r302, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r298;
mov.b32 r304, {high, high};
}
{
mul.f16x2 r306, r222, r304;
}
{
neg.f16x2 r309, r306;
}
{
fma.rn.f16x2 r311, r219, r302, r309;
}
{
mul.f16x2 r315, r219, r304;
}
{
fma.rn.f16x2 r318, r222, r302, r315;
}
barrier.sync 0;
and.b32 r870, r859, 8128;
add.s32 r871, r869, r870;
st.shared.u32 [r871], r201;
st.shared.u32 [r871+16], r237;
st.shared.u32 [r871+32], r274;
st.shared.u32 [r871+48], r311;
barrier.sync 0;
mad.lo.s32 r872, r865, -12, r871;
ld.shared.u32 r340, [r872];
ld.shared.u32 r352, [r872+2048];
ld.shared.u32 r341, [r872+4096];
ld.shared.u32 r353, [r872+6144];
barrier.sync 0;
st.shared.u32 [r871], r204;
st.shared.u32 [r871+16], r244;
st.shared.u32 [r871+32], r281;
st.shared.u32 [r871+48], r318;
barrier.sync 0;
ld.shared.u32 r343, [r872];
ld.shared.u32 r355, [r872+2048];
ld.shared.u32 r344, [r872+4096];
ld.shared.u32 r356, [r872+6144];
{
add.f16x2 r339, r340, r341;
}
{
add.f16x2 r342, r343, r344;
}
{
sub.f16x2 r345, r340, r341;
}
{
sub.f16x2 r348, r343, r344;
}
{
add.f16x2 r351, r352, r353;
}
{
add.f16x2 r354, r355, r356;
}
{
sub.f16x2 r357, r352, r353;
}
{
sub.f16x2 r360, r355, r356;
}
{
neg.f16x2 r363, r357;
}
{
add.f16x2 r365, r339, r351;
}
{
add.f16x2 r368, r342, r354;
}
{
sub.f16x2 r371, r339, r351;
}
{
sub.f16x2 r374, r342, r354;
}
{
add.f16x2 r377, r345, r360;
}
{
add.f16x2 r380, r348, r363;
}
{
sub.f16x2 r383, r345, r360;
}
{
sub.f16x2 r386, r348, r363;
}
and.b32 r873, r9, 496;
bfe.u32 r874, r9, 4, 5;
and.b32 r875, r867, 60;
add.s32 r876, r861, r875;
cvt.rn.f32.u32 f61, r874;
mul.f32 f62, f61, 0f3D490FDB;
cos.approx.f32 f27, f62;
sin.approx.f32 f63, f62;
neg.f32 f28, f63;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f27;
cvt.rn.f16.f32 high, f28;
mov.b32 r389, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r389;
mov.b32 r392, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r389;
mov.b32 r394, {high, high};
}
{
mul.f16x2 r396, r380, r394;
}
{
neg.f16x2 r399, r396;
}
{
fma.rn.f16x2 r401, r377, r392, r399;
}
{
mul.f16x2 r405, r377, r394;
}
{
fma.rn.f16x2 r408, r380, r392, r405;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r389;
mov.b32 r412, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r389;
mov.b32 r414, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r416, {low, high};
}
{
mul.f16x2 r417, r414, r416;
}
{
mul.f16x2 r420, r389, r412;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r389;
mov.b32 r423, {high, low};
}
{
fma.rn.f16x2 r425, r417, r423, r420;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r425;
mov.b32 r429, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r425;
mov.b32 r431, {high, high};
}
{
mul.f16x2 r433, r374, r431;
}
{
neg.f16x2 r436, r433;
}
{
fma.rn.f16x2 r438, r371, r429, r436;
}
{
mul.f16x2 r442, r371, r431;
}
{
fma.rn.f16x2 r445, r374, r429, r442;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r389;
mov.b32 r449, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r389;
mov.b32 r451, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r453, {low, high};
}
{
mul.f16x2 r454, r451, r453;
}
{
mul.f16x2 r457, r425, r449;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r425;
mov.b32 r460, {high, low};
}
{
fma.rn.f16x2 r462, r454, r460, r457;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r462;
mov.b32 r466, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r462;
mov.b32 r468, {high, high};
}
{
mul.f16x2 r470, r386, r468;
}
{
neg.f16x2 r473, r470;
}
{
fma.rn.f16x2 r475, r383, r466, r473;
}
{
mul.f16x2 r479, r383, r468;
}
{
fma.rn.f16x2 r482, r386, r466, r479;
}
barrier.sync 0;
and.b32 r877, r859, 7936;
add.s32 r878, r876, r877;
st.shared.u32 [r878], r365;
st.shared.u32 [r878+64], r401;
st.shared.u32 [r878+128], r438;
st.shared.u32 [r878+192], r475;
barrier.sync 0;
mad.lo.s32 r879, r873, -12, r878;
ld.shared.u32 r504, [r879];
ld.shared.u32 r516, [r879+2048];
ld.shared.u32 r505, [r879+4096];
ld.shared.u32 r517, [r879+6144];
barrier.sync 0;
st.shared.u32 [r878], r368;
st.shared.u32 [r878+64], r408;
st.shared.u32 [r878+128], r445;
st.shared.u32 [r878+192], r482;
barrier.sync 0;
ld.shared.u32 r507, [r879];
ld.shared.u32 r519, [r879+2048];
ld.shared.u32 r508, [r879+4096];
ld.shared.u32 r520, [r879+6144];
{
add.f16x2 r503, r504, r505;
}
{
add.f16x2 r506, r507, r508;
}
{
sub.f16x2 r509, r504, r505;
}
{
sub.f16x2 r512, r507, r508;
}
{
add.f16x2 r515, r516, r517;
}
{
add.f16x2 r518, r519, r520;
}
{
sub.f16x2 r521, r516, r517;
}
{
sub.f16x2 r524, r519, r520;
}
{
neg.f16x2 r527, r521;
}
{
add.f16x2 r529, r503, r515;
}
{
add.f16x2 r532, r506, r518;
}
{
sub.f16x2 r535, r503, r515;
}
{
sub.f16x2 r538, r506, r518;
}
{
add.f16x2 r541, r509, r524;
}
{
add.f16x2 r544, r512, r527;
}
{
sub.f16x2 r547, r509, r524;
}
{
sub.f16x2 r550, r512, r527;
}
and.b32 r880, r9, 448;
bfe.u32 r881, r9, 6, 3;
and.b32 r882, r867, 252;
add.s32 r883, r861, r882;
cvt.rn.f32.u32 f64, r881;
mul.f32 f65, f64, 0f3E490FDB;
cos.approx.f32 f37, f65;
sin.approx.f32 f66, f65;
neg.f32 f38, f66;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
mov.b32 r553, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r553;
mov.b32 r556, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r553;
mov.b32 r558, {high, high};
}
{
mul.f16x2 r560, r544, r558;
}
{
neg.f16x2 r563, r560;
}
{
fma.rn.f16x2 r565, r541, r556, r563;
}
{
mul.f16x2 r569, r541, r558;
}
{
fma.rn.f16x2 r572, r544, r556, r569;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r553;
mov.b32 r576, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r553;
mov.b32 r578, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r580, {low, high};
}
{
mul.f16x2 r581, r578, r580;
}
{
mul.f16x2 r584, r553, r576;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r553;
mov.b32 r587, {high, low};
}
{
fma.rn.f16x2 r589, r581, r587, r584;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r589;
mov.b32 r593, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r589;
mov.b32 r595, {high, high};
}
{
mul.f16x2 r597, r538, r595;
}
{
neg.f16x2 r600, r597;
}
{
fma.rn.f16x2 r602, r535, r593, r600;
}
{
mul.f16x2 r606, r535, r595;
}
{
fma.rn.f16x2 r609, r538, r593, r606;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r553;
mov.b32 r613, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r553;
mov.b32 r615, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r617, {low, high};
}
{
mul.f16x2 r618, r615, r617;
}
{
mul.f16x2 r621, r589, r613;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r589;
mov.b32 r624, {high, low};
}
{
fma.rn.f16x2 r626, r618, r624, r621;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r626;
mov.b32 r630, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r626;
mov.b32 r632, {high, high};
}
{
mul.f16x2 r634, r550, r632;
}
{
neg.f16x2 r637, r634;
}
{
fma.rn.f16x2 r639, r547, r630, r637;
}
{
mul.f16x2 r643, r547, r632;
}
{
fma.rn.f16x2 r646, r550, r630, r643;
}
barrier.sync 0;
and.b32 r884, r859, 7168;
add.s32 r885, r883, r884;
st.shared.u32 [r885], r529;
st.shared.u32 [r885+256], r565;
st.shared.u32 [r885+512], r602;
st.shared.u32 [r885+768], r639;
barrier.sync 0;
mad.lo.s32 r886, r880, -12, r885;
ld.shared.u32 r668, [r886];
ld.shared.u32 r680, [r886+2048];
ld.shared.u32 r669, [r886+4096];
ld.shared.u32 r681, [r886+6144];
barrier.sync 0;
st.shared.u32 [r885], r532;
st.shared.u32 [r885+256], r572;
st.shared.u32 [r885+512], r609;
st.shared.u32 [r885+768], r646;
barrier.sync 0;
ld.shared.u32 r671, [r886];
ld.shared.u32 r683, [r886+2048];
ld.shared.u32 r672, [r886+4096];
ld.shared.u32 r684, [r886+6144];
{
add.f16x2 r667, r668, r669;
}
{
add.f16x2 r670, r671, r672;
}
{
sub.f16x2 r673, r668, r669;
}
{
sub.f16x2 r676, r671, r672;
}
{
add.f16x2 r679, r680, r681;
}
{
add.f16x2 r682, r683, r684;
}
{
sub.f16x2 r685, r680, r681;
}
{
sub.f16x2 r688, r683, r684;
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
and.b32 r887, r9, 256;
bfe.u32 r888, r9, 8, 1;
and.b32 r889, r867, 1020;
add.s32 r890, r861, r889;
cvt.rn.f32.u32 f67, r888;
mul.f32 f68, f67, 0f3F490FDB;
cos.approx.f32 f47, f68;
sin.approx.f32 f69, f68;
neg.f32 f48, f69;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f47;
cvt.rn.f16.f32 high, f48;
mov.b32 r717, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r717;
mov.b32 r720, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r717;
mov.b32 r722, {high, high};
}
{
mul.f16x2 r724, r708, r722;
}
{
neg.f16x2 r727, r724;
}
{
fma.rn.f16x2 r729, r705, r720, r727;
}
{
mul.f16x2 r733, r705, r722;
}
{
fma.rn.f16x2 r736, r708, r720, r733;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r717;
mov.b32 r740, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r717;
mov.b32 r742, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r744, {low, high};
}
{
mul.f16x2 r745, r742, r744;
}
{
mul.f16x2 r748, r717, r740;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r717;
mov.b32 r751, {high, low};
}
{
fma.rn.f16x2 r753, r745, r751, r748;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r753;
mov.b32 r757, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r753;
mov.b32 r759, {high, high};
}
{
mul.f16x2 r761, r702, r759;
}
{
neg.f16x2 r764, r761;
}
{
fma.rn.f16x2 r766, r699, r757, r764;
}
{
mul.f16x2 r770, r699, r759;
}
{
fma.rn.f16x2 r773, r702, r757, r770;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r717;
mov.b32 r777, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r717;
mov.b32 r779, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r781, {low, high};
}
{
mul.f16x2 r782, r779, r781;
}
{
mul.f16x2 r785, r753, r777;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r753;
mov.b32 r788, {high, low};
}
{
fma.rn.f16x2 r790, r782, r788, r785;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r790;
mov.b32 r794, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r790;
mov.b32 r796, {high, high};
}
{
mul.f16x2 r798, r714, r796;
}
{
neg.f16x2 r801, r798;
}
{
fma.rn.f16x2 r803, r711, r794, r801;
}
{
mul.f16x2 r807, r711, r796;
}
{
fma.rn.f16x2 r810, r714, r794, r807;
}
barrier.sync 0;
and.b32 r891, r859, 4096;
add.s32 r892, r890, r891;
st.shared.u32 [r892], r693;
st.shared.u32 [r892+1024], r729;
st.shared.u32 [r892+2048], r766;
st.shared.u32 [r892+3072], r803;
barrier.sync 0;
mad.lo.s32 r893, r887, -12, r892;
ld.shared.u32 r832, [r893];
ld.shared.u32 r844, [r893+2048];
ld.shared.u32 r833, [r893+4096];
ld.shared.u32 r845, [r893+6144];
barrier.sync 0;
st.shared.u32 [r892], r696;
st.shared.u32 [r892+1024], r736;
st.shared.u32 [r892+2048], r773;
st.shared.u32 [r892+3072], r810;
barrier.sync 0;
ld.shared.u32 r835, [r893];
ld.shared.u32 r847, [r893+2048];
ld.shared.u32 r836, [r893+4096];
ld.shared.u32 r848, [r893+6144];
{
add.f16x2 %0, r832, r833;
}
{
add.f16x2 %1, r835, r836;
}
{
sub.f16x2 %4, r832, r833;
}
{
sub.f16x2 %5, r835, r836;
}
{
add.f16x2 %2, r844, r845;
}
{
add.f16x2 %3, r847, r848;
}
{
sub.f16x2 %6, r844, r845;
}
{
sub.f16x2 %7, r847, r848;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<845, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .pred p<5>;
.reg .b16 rs<3>;
.reg .f32 f<103>;
.reg .b32 r<631>;
.reg .b64 rd<2>;
{
add.f16x2 r9, %5, %7;
}
{
add.f16x2 r12, %6, %8;
}
{
sub.f16x2 r15, %5, %7;
}
{
sub.f16x2 r18, %6, %8;
}
mov.u32 r21, %tid.x;
and.b32 r22, r21, 1023;
cvt.rn.f32.u32 f10, r22;
mul.f32 f1, f10, 0f3B490FDB;
cvt.u16.u32 rs2, r21;
and.b16 rs1, rs2, 1023;
mov.f32 f100, 0f3BC90F88;
setp.eq.s16 p1, rs1, 510;
@p1 bra LBB8_4;
setp.ne.s16 p2, rs1, 511;
@p2 bra LBB8_3;
mov.f32 f100, 0f3B490FC6;
bra.uni LBB8_4;
LBB8_3:
cos.approx.f32 f100, f1;
LBB8_4:
setp.eq.s32 p3, r22, 1023;
mov.f32 f101, 0f3B490FC6;
@p3 bra LBB8_6;
sin.approx.f32 f101, f1;
LBB8_6:
mov.u32 r77, %tid.y;
shl.b32 r78, r77, 14;
mov.u32 r79, %4;
add.s32 r80, r79, r78;
shl.b32 r82, r21, 4;
and.b32 r83, r82, -16384;
add.s32 r84, r80, r83;
neg.f32 f14, f101;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f100;
cvt.rn.f16.f32 high, f14;
mov.b32 r25, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r25;
mov.b32 r28, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r25;
mov.b32 r30, {high, high};
}
{
mul.f16x2 r32, r18, r30;
}
{
neg.f16x2 r35, r32;
}
{
fma.rn.f16x2 r37, r15, r28, r35;
}
{
mul.f16x2 r41, r15, r30;
}
{
fma.rn.f16x2 r44, r18, r28, r41;
}
barrier.sync 0;
and.b32 r85, r82, 16368;
add.s32 r86, r84, r85;
st.shared.v2.f32 [r86], {r9, r12};
st.shared.v2.f32 [r86+8], {r37, r44};
barrier.sync 0;
shl.b32 r87, r21, 3;
and.b32 r88, r87, 8184;
sub.s32 r89, r86, r88;
ld.shared.u32 r66, [r89];
ld.shared.u32 r69, [r89+4];
ld.shared.u32 r67, [r89+8192];
ld.shared.u32 r70, [r89+8196];
{
add.f16x2 r65, r66, r67;
}
{
add.f16x2 r68, r69, r70;
}
{
sub.f16x2 r71, r66, r67;
}
{
sub.f16x2 r74, r69, r70;
}
and.b32 r90, r21, 1022;
bfe.u32 r91, r21, 1, 9;
cvt.rn.f32.u32 f20, r91;
mul.f32 f6, f20, 0f3BC90FDB;
setp.eq.s32 p4, r90, 510;
mov.f32 f102, 0f3BC90F88;
@p4 bra LBB8_8;
cos.approx.f32 f102, f6;
LBB8_8:
sin.approx.f32 f75, f6;
neg.f32 f22, f75;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f22;
mov.b32 r92, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r92;
mov.b32 r95, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r92;
mov.b32 r97, {high, high};
}
{
mul.f16x2 r99, r74, r97;
}
{
neg.f16x2 r102, r99;
}
{
fma.rn.f16x2 r104, r71, r95, r102;
}
{
mul.f16x2 r108, r71, r97;
}
{
fma.rn.f16x2 r111, r74, r95, r108;
}
barrier.sync 0;
and.b32 r569, r87, 8;
add.s32 r570, r84, r569;
and.b32 r571, r82, 16352;
add.s32 r572, r570, r571;
st.shared.u32 [r572], r65;
st.shared.u32 [r572+4], r68;
st.shared.u32 [r572+16], r104;
st.shared.u32 [r572+20], r111;
barrier.sync 0;
and.b32 r573, r87, 8176;
sub.s32 r574, r572, r573;
ld.shared.u32 r133, [r574];
ld.shared.u32 r136, [r574+4];
ld.shared.u32 r134, [r574+8192];
ld.shared.u32 r137, [r574+8196];
{
add.f16x2 r132, r133, r134;
}
{
add.f16x2 r135, r136, r137;
}
{
sub.f16x2 r138, r133, r134;
}
{
sub.f16x2 r141, r136, r137;
}
bfe.u32 r575, r21, 2, 8;
cvt.rn.f32.u32 f76, r575;
mul.f32 f77, f76, 0f3C490FDB;
cos.approx.f32 f27, f77;
sin.approx.f32 f78, f77;
neg.f32 f28, f78;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f27;
cvt.rn.f16.f32 high, f28;
mov.b32 r144, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r144;
mov.b32 r147, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r144;
mov.b32 r149, {high, high};
}
{
mul.f16x2 r151, r141, r149;
}
{
neg.f16x2 r154, r151;
}
{
fma.rn.f16x2 r156, r138, r147, r154;
}
{
mul.f16x2 r160, r138, r149;
}
{
fma.rn.f16x2 r163, r141, r147, r160;
}
and.b32 r576, r87, 24;
add.s32 r577, r84, r576;
barrier.sync 0;
and.b32 r578, r82, 16320;
add.s32 r579, r577, r578;
st.shared.u32 [r579], r132;
st.shared.u32 [r579+4], r135;
st.shared.u32 [r579+32], r156;
st.shared.u32 [r579+36], r163;
barrier.sync 0;
and.b32 r580, r87, 8160;
sub.s32 r581, r579, r580;
ld.shared.u32 r185, [r581];
ld.shared.u32 r188, [r581+4];
ld.shared.u32 r186, [r581+8192];
ld.shared.u32 r189, [r581+8196];
{
add.f16x2 r184, r185, r186;
}
{
add.f16x2 r187, r188, r189;
}
{
sub.f16x2 r190, r185, r186;
}
{
sub.f16x2 r193, r188, r189;
}
bfe.u32 r582, r21, 3, 7;
cvt.rn.f32.u32 f79, r582;
mul.f32 f80, f79, 0f3CC90FDB;
cos.approx.f32 f33, f80;
sin.approx.f32 f81, f80;
neg.f32 f34, f81;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
mov.b32 r196, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r196;
mov.b32 r199, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r196;
mov.b32 r201, {high, high};
}
{
mul.f16x2 r203, r193, r201;
}
{
neg.f16x2 r206, r203;
}
{
fma.rn.f16x2 r208, r190, r199, r206;
}
{
mul.f16x2 r212, r190, r201;
}
{
fma.rn.f16x2 r215, r193, r199, r212;
}
and.b32 r583, r87, 56;
add.s32 r584, r84, r583;
barrier.sync 0;
and.b32 r585, r82, 16256;
add.s32 r586, r584, r585;
st.shared.u32 [r586], r184;
st.shared.u32 [r586+4], r187;
st.shared.u32 [r586+64], r208;
st.shared.u32 [r586+68], r215;
barrier.sync 0;
and.b32 r587, r87, 8128;
sub.s32 r588, r586, r587;
ld.shared.u32 r237, [r588];
ld.shared.u32 r240, [r588+4];
ld.shared.u32 r238, [r588+8192];
ld.shared.u32 r241, [r588+8196];
{
add.f16x2 r236, r237, r238;
}
{
add.f16x2 r239, r240, r241;
}
{
sub.f16x2 r242, r237, r238;
}
{
sub.f16x2 r245, r240, r241;
}
bfe.u32 r589, r21, 4, 6;
cvt.rn.f32.u32 f82, r589;
mul.f32 f83, f82, 0f3D490FDB;
cos.approx.f32 f39, f83;
sin.approx.f32 f84, f83;
neg.f32 f40, f84;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f39;
cvt.rn.f16.f32 high, f40;
mov.b32 r248, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r248;
mov.b32 r251, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r248;
mov.b32 r253, {high, high};
}
{
mul.f16x2 r255, r245, r253;
}
{
neg.f16x2 r258, r255;
}
{
fma.rn.f16x2 r260, r242, r251, r258;
}
{
mul.f16x2 r264, r242, r253;
}
{
fma.rn.f16x2 r267, r245, r251, r264;
}
and.b32 r590, r87, 120;
add.s32 r591, r84, r590;
barrier.sync 0;
and.b32 r592, r82, 16128;
add.s32 r593, r591, r592;
st.shared.u32 [r593], r236;
st.shared.u32 [r593+4], r239;
st.shared.u32 [r593+128], r260;
st.shared.u32 [r593+132], r267;
barrier.sync 0;
and.b32 r594, r87, 8064;
sub.s32 r595, r593, r594;
ld.shared.u32 r289, [r595];
ld.shared.u32 r292, [r595+4];
ld.shared.u32 r290, [r595+8192];
ld.shared.u32 r293, [r595+8196];
{
add.f16x2 r288, r289, r290;
}
{
add.f16x2 r291, r292, r293;
}
{
sub.f16x2 r294, r289, r290;
}
{
sub.f16x2 r297, r292, r293;
}
bfe.u32 r596, r21, 5, 5;
cvt.rn.f32.u32 f85, r596;
mul.f32 f86, f85, 0f3DC90FDB;
cos.approx.f32 f45, f86;
sin.approx.f32 f87, f86;
neg.f32 f46, f87;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
mov.b32 r300, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r300;
mov.b32 r303, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r300;
mov.b32 r305, {high, high};
}
{
mul.f16x2 r307, r297, r305;
}
{
neg.f16x2 r310, r307;
}
{
fma.rn.f16x2 r312, r294, r303, r310;
}
{
mul.f16x2 r316, r294, r305;
}
{
fma.rn.f16x2 r319, r297, r303, r316;
}
and.b32 r597, r87, 248;
add.s32 r598, r84, r597;
barrier.sync 0;
and.b32 r599, r82, 15872;
add.s32 r600, r598, r599;
st.shared.u32 [r600], r288;
st.shared.u32 [r600+4], r291;
st.shared.u32 [r600+256], r312;
st.shared.u32 [r600+260], r319;
barrier.sync 0;
and.b32 r601, r87, 7936;
sub.s32 r602, r600, r601;
ld.shared.u32 r341, [r602];
ld.shared.u32 r344, [r602+4];
ld.shared.u32 r342, [r602+8192];
ld.shared.u32 r345, [r602+8196];
{
add.f16x2 r340, r341, r342;
}
{
add.f16x2 r343, r344, r345;
}
{
sub.f16x2 r346, r341, r342;
}
{
sub.f16x2 r349, r344, r345;
}
bfe.u32 r603, r21, 6, 4;
cvt.rn.f32.u32 f88, r603;
mul.f32 f89, f88, 0f3E490FDB;
cos.approx.f32 f51, f89;
sin.approx.f32 f90, f89;
neg.f32 f52, f90;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f51;
cvt.rn.f16.f32 high, f52;
mov.b32 r352, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r352;
mov.b32 r355, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r352;
mov.b32 r357, {high, high};
}
{
mul.f16x2 r359, r349, r357;
}
{
neg.f16x2 r362, r359;
}
{
fma.rn.f16x2 r364, r346, r355, r362;
}
{
mul.f16x2 r368, r346, r357;
}
{
fma.rn.f16x2 r371, r349, r355, r368;
}
and.b32 r604, r87, 504;
add.s32 r605, r84, r604;
barrier.sync 0;
and.b32 r606, r82, 15360;
add.s32 r607, r605, r606;
st.shared.u32 [r607], r340;
st.shared.u32 [r607+4], r343;
st.shared.u32 [r607+512], r364;
st.shared.u32 [r607+516], r371;
barrier.sync 0;
and.b32 r608, r87, 7680;
sub.s32 r609, r607, r608;
ld.shared.u32 r393, [r609];
ld.shared.u32 r396, [r609+4];
ld.shared.u32 r394, [r609+8192];
ld.shared.u32 r397, [r609+8196];
{
add.f16x2 r392, r393, r394;
}
{
add.f16x2 r395, r396, r397;
}
{
sub.f16x2 r398, r393, r394;
}
{
sub.f16x2 r401, r396, r397;
}
bfe.u32 r610, r21, 7, 3;
cvt.rn.f32.u32 f91, r610;
mul.f32 f92, f91, 0f3EC90FDB;
cos.approx.f32 f57, f92;
sin.approx.f32 f93, f92;
neg.f32 f58, f93;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f57;
cvt.rn.f16.f32 high, f58;
mov.b32 r404, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r404;
mov.b32 r407, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r404;
mov.b32 r409, {high, high};
}
{
mul.f16x2 r411, r401, r409;
}
{
neg.f16x2 r414, r411;
}
{
fma.rn.f16x2 r416, r398, r407, r414;
}
{
mul.f16x2 r420, r398, r409;
}
{
fma.rn.f16x2 r423, r401, r407, r420;
}
and.b32 r611, r87, 1016;
add.s32 r612, r84, r611;
barrier.sync 0;
and.b32 r613, r82, 14336;
add.s32 r614, r612, r613;
st.shared.u32 [r614], r392;
st.shared.u32 [r614+4], r395;
st.shared.u32 [r614+1024], r416;
st.shared.u32 [r614+1028], r423;
barrier.sync 0;
and.b32 r615, r87, 7168;
sub.s32 r616, r614, r615;
ld.shared.u32 r445, [r616];
ld.shared.u32 r448, [r616+4];
ld.shared.u32 r446, [r616+8192];
ld.shared.u32 r449, [r616+8196];
{
add.f16x2 r444, r445, r446;
}
{
add.f16x2 r447, r448, r449;
}
{
sub.f16x2 r450, r445, r446;
}
{
sub.f16x2 r453, r448, r449;
}
bfe.u32 r617, r21, 8, 2;
cvt.rn.f32.u32 f94, r617;
mul.f32 f95, f94, 0f3F490FDB;
cos.approx.f32 f63, f95;
sin.approx.f32 f96, f95;
neg.f32 f64, f96;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f63;
cvt.rn.f16.f32 high, f64;
mov.b32 r456, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r456;
mov.b32 r459, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r456;
mov.b32 r461, {high, high};
}
{
mul.f16x2 r463, r453, r461;
}
{
neg.f16x2 r466, r463;
}
{
fma.rn.f16x2 r468, r450, r459, r466;
}
{
mul.f16x2 r472, r450, r461;
}
{
fma.rn.f16x2 r475, r453, r459, r472;
}
and.b32 r618, r87, 2040;
add.s32 r619, r84, r618;
barrier.sync 0;
and.b32 r620, r82, 12288;
add.s32 r621, r619, r620;
st.shared.u32 [r621], r444;
st.shared.u32 [r621+4], r447;
st.shared.u32 [r621+2048], r468;
st.shared.u32 [r621+2052], r475;
barrier.sync 0;
and.b32 r622, r87, 6144;
sub.s32 r623, r621, r622;
ld.shared.u32 r497, [r623];
ld.shared.u32 r500, [r623+4];
ld.shared.u32 r498, [r623+8192];
ld.shared.u32 r501, [r623+8196];
{
add.f16x2 r496, r497, r498;
}
{
add.f16x2 r499, r500, r501;
}
{
sub.f16x2 r502, r497, r498;
}
{
sub.f16x2 r505, r500, r501;
}
bfe.u32 r624, r21, 9, 1;
cvt.rn.f32.u32 f97, r624;
mul.f32 f98, f97, 0f3FC90FDB;
cos.approx.f32 f69, f98;
sin.approx.f32 f99, f98;
neg.f32 f70, f99;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r508, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r508;
mov.b32 r511, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r508;
mov.b32 r513, {high, high};
}
{
mul.f16x2 r515, r505, r513;
}
{
neg.f16x2 r518, r515;
}
{
fma.rn.f16x2 r520, r502, r511, r518;
}
{
mul.f16x2 r524, r502, r513;
}
{
fma.rn.f16x2 r527, r505, r511, r524;
}
and.b32 r625, r87, 4088;
add.s32 r626, r84, r625;
barrier.sync 0;
and.b32 r627, r82, 8192;
add.s32 r628, r626, r627;
st.shared.u32 [r628], r496;
st.shared.u32 [r628+4], r499;
st.shared.u32 [r628+4096], r520;
st.shared.u32 [r628+4100], r527;
barrier.sync 0;
and.b32 r629, r87, 4096;
sub.s32 r630, r628, r629;
ld.shared.u32 r549, [r630];
ld.shared.u32 r552, [r630+4];
ld.shared.u32 r550, [r630+8192];
ld.shared.u32 r553, [r630+8196];
{
add.f16x2 %0, r549, r550;
}
{
add.f16x2 %1, r552, r553;
}
{
sub.f16x2 %2, r549, r550;
}
{
sub.f16x2 %3, r552, r553;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<846, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .pred p<5>;
.reg .b16 rs<3>;
.reg .f32 f<103>;
.reg .b32 r<631>;
.reg .b64 rd<2>;
{
add.f16x2 r9, %5, %7;
}
{
add.f16x2 r12, %6, %8;
}
{
sub.f16x2 r15, %5, %7;
}
{
sub.f16x2 r18, %6, %8;
}
mov.u32 r21, %tid.x;
and.b32 r22, r21, 1023;
cvt.rn.f32.u32 f10, r22;
mul.f32 f1, f10, 0f3B490FDB;
cvt.u16.u32 rs2, r21;
and.b16 rs1, rs2, 1023;
mov.f32 f100, 0f3BC90F88;
setp.eq.s16 p1, rs1, 510;
@p1 bra LBB9_4;
setp.ne.s16 p2, rs1, 511;
@p2 bra LBB9_3;
mov.f32 f100, 0f3B490FC6;
bra.uni LBB9_4;
LBB9_3:
cos.approx.f32 f100, f1;
LBB9_4:
setp.eq.s32 p3, r22, 1023;
mov.f32 f101, 0f3B490FC6;
@p3 bra LBB9_6;
sin.approx.f32 f101, f1;
LBB9_6:
mov.u32 r77, %tid.y;
shl.b32 r78, r77, 13;
mov.u32 r79, %4;
add.s32 r80, r79, r78;
shl.b32 r82, r21, 3;
and.b32 r83, r82, -8192;
add.s32 r84, r80, r83;
neg.f32 f14, f101;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f100;
cvt.rn.f16.f32 high, f14;
mov.b32 r25, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r25;
mov.b32 r28, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r25;
mov.b32 r30, {high, high};
}
{
mul.f16x2 r32, r18, r30;
}
{
neg.f16x2 r35, r32;
}
{
fma.rn.f16x2 r37, r15, r28, r35;
}
{
mul.f16x2 r41, r15, r30;
}
{
fma.rn.f16x2 r44, r18, r28, r41;
}
barrier.sync 0;
and.b32 r85, r82, 8184;
add.s32 r86, r84, r85;
st.shared.v2.f32 [r86], {r9, r37};
barrier.sync 0;
shl.b32 r87, r21, 2;
and.b32 r88, r87, 4092;
sub.s32 r89, r86, r88;
ld.shared.u32 r66, [r89];
ld.shared.u32 r67, [r89+4096];
barrier.sync 0;
st.shared.v2.f32 [r86], {r12, r44};
barrier.sync 0;
ld.shared.u32 r69, [r89];
ld.shared.u32 r70, [r89+4096];
{
add.f16x2 r65, r66, r67;
}
{
add.f16x2 r68, r69, r70;
}
{
sub.f16x2 r71, r66, r67;
}
{
sub.f16x2 r74, r69, r70;
}
and.b32 r90, r21, 1022;
bfe.u32 r91, r21, 1, 9;
cvt.rn.f32.u32 f20, r91;
mul.f32 f6, f20, 0f3BC90FDB;
setp.eq.s32 p4, r90, 510;
mov.f32 f102, 0f3BC90F88;
@p4 bra LBB9_8;
cos.approx.f32 f102, f6;
LBB9_8:
and.b32 r569, r87, 4;
add.s32 r570, r84, r569;
sin.approx.f32 f75, f6;
neg.f32 f22, f75;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f22;
mov.b32 r92, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r92;
mov.b32 r95, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r92;
mov.b32 r97, {high, high};
}
{
mul.f16x2 r99, r74, r97;
}
{
neg.f16x2 r102, r99;
}
{
fma.rn.f16x2 r104, r71, r95, r102;
}
{
mul.f16x2 r108, r71, r97;
}
{
fma.rn.f16x2 r111, r74, r95, r108;
}
barrier.sync 0;
and.b32 r571, r82, 8176;
add.s32 r572, r570, r571;
st.shared.u32 [r572], r65;
st.shared.u32 [r572+8], r104;
barrier.sync 0;
and.b32 r573, r87, 4088;
sub.s32 r574, r572, r573;
ld.shared.u32 r133, [r574];
ld.shared.u32 r134, [r574+4096];
barrier.sync 0;
st.shared.u32 [r572], r68;
st.shared.u32 [r572+8], r111;
barrier.sync 0;
ld.shared.u32 r136, [r574];
ld.shared.u32 r137, [r574+4096];
{
add.f16x2 r132, r133, r134;
}
{
add.f16x2 r135, r136, r137;
}
{
sub.f16x2 r138, r133, r134;
}
{
sub.f16x2 r141, r136, r137;
}
bfe.u32 r575, r21, 2, 8;
and.b32 r576, r87, 12;
add.s32 r577, r84, r576;
cvt.rn.f32.u32 f76, r575;
mul.f32 f77, f76, 0f3C490FDB;
cos.approx.f32 f27, f77;
sin.approx.f32 f78, f77;
neg.f32 f28, f78;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f27;
cvt.rn.f16.f32 high, f28;
mov.b32 r144, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r144;
mov.b32 r147, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r144;
mov.b32 r149, {high, high};
}
{
mul.f16x2 r151, r141, r149;
}
{
neg.f16x2 r154, r151;
}
{
fma.rn.f16x2 r156, r138, r147, r154;
}
{
mul.f16x2 r160, r138, r149;
}
{
fma.rn.f16x2 r163, r141, r147, r160;
}
barrier.sync 0;
and.b32 r578, r82, 8160;
add.s32 r579, r577, r578;
st.shared.u32 [r579], r132;
st.shared.u32 [r579+16], r156;
barrier.sync 0;
and.b32 r580, r87, 4080;
sub.s32 r581, r579, r580;
ld.shared.u32 r185, [r581];
ld.shared.u32 r186, [r581+4096];
barrier.sync 0;
st.shared.u32 [r579], r135;
st.shared.u32 [r579+16], r163;
barrier.sync 0;
ld.shared.u32 r188, [r581];
ld.shared.u32 r189, [r581+4096];
{
add.f16x2 r184, r185, r186;
}
{
add.f16x2 r187, r188, r189;
}
{
sub.f16x2 r190, r185, r186;
}
{
sub.f16x2 r193, r188, r189;
}
bfe.u32 r582, r21, 3, 7;
and.b32 r583, r87, 28;
add.s32 r584, r84, r583;
cvt.rn.f32.u32 f79, r582;
mul.f32 f80, f79, 0f3CC90FDB;
cos.approx.f32 f33, f80;
sin.approx.f32 f81, f80;
neg.f32 f34, f81;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
mov.b32 r196, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r196;
mov.b32 r199, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r196;
mov.b32 r201, {high, high};
}
{
mul.f16x2 r203, r193, r201;
}
{
neg.f16x2 r206, r203;
}
{
fma.rn.f16x2 r208, r190, r199, r206;
}
{
mul.f16x2 r212, r190, r201;
}
{
fma.rn.f16x2 r215, r193, r199, r212;
}
barrier.sync 0;
and.b32 r585, r82, 8128;
add.s32 r586, r584, r585;
st.shared.u32 [r586], r184;
st.shared.u32 [r586+32], r208;
barrier.sync 0;
and.b32 r587, r87, 4064;
sub.s32 r588, r586, r587;
ld.shared.u32 r237, [r588];
ld.shared.u32 r238, [r588+4096];
barrier.sync 0;
st.shared.u32 [r586], r187;
st.shared.u32 [r586+32], r215;
barrier.sync 0;
ld.shared.u32 r240, [r588];
ld.shared.u32 r241, [r588+4096];
{
add.f16x2 r236, r237, r238;
}
{
add.f16x2 r239, r240, r241;
}
{
sub.f16x2 r242, r237, r238;
}
{
sub.f16x2 r245, r240, r241;
}
bfe.u32 r589, r21, 4, 6;
and.b32 r590, r87, 60;
add.s32 r591, r84, r590;
cvt.rn.f32.u32 f82, r589;
mul.f32 f83, f82, 0f3D490FDB;
cos.approx.f32 f39, f83;
sin.approx.f32 f84, f83;
neg.f32 f40, f84;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f39;
cvt.rn.f16.f32 high, f40;
mov.b32 r248, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r248;
mov.b32 r251, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r248;
mov.b32 r253, {high, high};
}
{
mul.f16x2 r255, r245, r253;
}
{
neg.f16x2 r258, r255;
}
{
fma.rn.f16x2 r260, r242, r251, r258;
}
{
mul.f16x2 r264, r242, r253;
}
{
fma.rn.f16x2 r267, r245, r251, r264;
}
barrier.sync 0;
and.b32 r592, r82, 8064;
add.s32 r593, r591, r592;
st.shared.u32 [r593], r236;
st.shared.u32 [r593+64], r260;
barrier.sync 0;
and.b32 r594, r87, 4032;
sub.s32 r595, r593, r594;
ld.shared.u32 r289, [r595];
ld.shared.u32 r290, [r595+4096];
barrier.sync 0;
st.shared.u32 [r593], r239;
st.shared.u32 [r593+64], r267;
barrier.sync 0;
ld.shared.u32 r292, [r595];
ld.shared.u32 r293, [r595+4096];
{
add.f16x2 r288, r289, r290;
}
{
add.f16x2 r291, r292, r293;
}
{
sub.f16x2 r294, r289, r290;
}
{
sub.f16x2 r297, r292, r293;
}
bfe.u32 r596, r21, 5, 5;
and.b32 r597, r87, 124;
add.s32 r598, r84, r597;
cvt.rn.f32.u32 f85, r596;
mul.f32 f86, f85, 0f3DC90FDB;
cos.approx.f32 f45, f86;
sin.approx.f32 f87, f86;
neg.f32 f46, f87;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
mov.b32 r300, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r300;
mov.b32 r303, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r300;
mov.b32 r305, {high, high};
}
{
mul.f16x2 r307, r297, r305;
}
{
neg.f16x2 r310, r307;
}
{
fma.rn.f16x2 r312, r294, r303, r310;
}
{
mul.f16x2 r316, r294, r305;
}
{
fma.rn.f16x2 r319, r297, r303, r316;
}
barrier.sync 0;
and.b32 r599, r82, 7936;
add.s32 r600, r598, r599;
st.shared.u32 [r600], r288;
st.shared.u32 [r600+128], r312;
barrier.sync 0;
and.b32 r601, r87, 3968;
sub.s32 r602, r600, r601;
ld.shared.u32 r341, [r602];
ld.shared.u32 r342, [r602+4096];
barrier.sync 0;
st.shared.u32 [r600], r291;
st.shared.u32 [r600+128], r319;
barrier.sync 0;
ld.shared.u32 r344, [r602];
ld.shared.u32 r345, [r602+4096];
{
add.f16x2 r340, r341, r342;
}
{
add.f16x2 r343, r344, r345;
}
{
sub.f16x2 r346, r341, r342;
}
{
sub.f16x2 r349, r344, r345;
}
bfe.u32 r603, r21, 6, 4;
and.b32 r604, r87, 252;
add.s32 r605, r84, r604;
cvt.rn.f32.u32 f88, r603;
mul.f32 f89, f88, 0f3E490FDB;
cos.approx.f32 f51, f89;
sin.approx.f32 f90, f89;
neg.f32 f52, f90;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f51;
cvt.rn.f16.f32 high, f52;
mov.b32 r352, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r352;
mov.b32 r355, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r352;
mov.b32 r357, {high, high};
}
{
mul.f16x2 r359, r349, r357;
}
{
neg.f16x2 r362, r359;
}
{
fma.rn.f16x2 r364, r346, r355, r362;
}
{
mul.f16x2 r368, r346, r357;
}
{
fma.rn.f16x2 r371, r349, r355, r368;
}
barrier.sync 0;
and.b32 r606, r82, 7680;
add.s32 r607, r605, r606;
st.shared.u32 [r607], r340;
st.shared.u32 [r607+256], r364;
barrier.sync 0;
and.b32 r608, r87, 3840;
sub.s32 r609, r607, r608;
ld.shared.u32 r393, [r609];
ld.shared.u32 r394, [r609+4096];
barrier.sync 0;
st.shared.u32 [r607], r343;
st.shared.u32 [r607+256], r371;
barrier.sync 0;
ld.shared.u32 r396, [r609];
ld.shared.u32 r397, [r609+4096];
{
add.f16x2 r392, r393, r394;
}
{
add.f16x2 r395, r396, r397;
}
{
sub.f16x2 r398, r393, r394;
}
{
sub.f16x2 r401, r396, r397;
}
bfe.u32 r610, r21, 7, 3;
and.b32 r611, r87, 508;
add.s32 r612, r84, r611;
cvt.rn.f32.u32 f91, r610;
mul.f32 f92, f91, 0f3EC90FDB;
cos.approx.f32 f57, f92;
sin.approx.f32 f93, f92;
neg.f32 f58, f93;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f57;
cvt.rn.f16.f32 high, f58;
mov.b32 r404, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r404;
mov.b32 r407, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r404;
mov.b32 r409, {high, high};
}
{
mul.f16x2 r411, r401, r409;
}
{
neg.f16x2 r414, r411;
}
{
fma.rn.f16x2 r416, r398, r407, r414;
}
{
mul.f16x2 r420, r398, r409;
}
{
fma.rn.f16x2 r423, r401, r407, r420;
}
barrier.sync 0;
and.b32 r613, r82, 7168;
add.s32 r614, r612, r613;
st.shared.u32 [r614], r392;
st.shared.u32 [r614+512], r416;
barrier.sync 0;
and.b32 r615, r87, 3584;
sub.s32 r616, r614, r615;
ld.shared.u32 r445, [r616];
ld.shared.u32 r446, [r616+4096];
barrier.sync 0;
st.shared.u32 [r614], r395;
st.shared.u32 [r614+512], r423;
barrier.sync 0;
ld.shared.u32 r448, [r616];
ld.shared.u32 r449, [r616+4096];
{
add.f16x2 r444, r445, r446;
}
{
add.f16x2 r447, r448, r449;
}
{
sub.f16x2 r450, r445, r446;
}
{
sub.f16x2 r453, r448, r449;
}
bfe.u32 r617, r21, 8, 2;
and.b32 r618, r87, 1020;
add.s32 r619, r84, r618;
cvt.rn.f32.u32 f94, r617;
mul.f32 f95, f94, 0f3F490FDB;
cos.approx.f32 f63, f95;
sin.approx.f32 f96, f95;
neg.f32 f64, f96;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f63;
cvt.rn.f16.f32 high, f64;
mov.b32 r456, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r456;
mov.b32 r459, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r456;
mov.b32 r461, {high, high};
}
{
mul.f16x2 r463, r453, r461;
}
{
neg.f16x2 r466, r463;
}
{
fma.rn.f16x2 r468, r450, r459, r466;
}
{
mul.f16x2 r472, r450, r461;
}
{
fma.rn.f16x2 r475, r453, r459, r472;
}
barrier.sync 0;
and.b32 r620, r82, 6144;
add.s32 r621, r619, r620;
st.shared.u32 [r621], r444;
st.shared.u32 [r621+1024], r468;
barrier.sync 0;
and.b32 r622, r87, 3072;
sub.s32 r623, r621, r622;
ld.shared.u32 r497, [r623];
ld.shared.u32 r498, [r623+4096];
barrier.sync 0;
st.shared.u32 [r621], r447;
st.shared.u32 [r621+1024], r475;
barrier.sync 0;
ld.shared.u32 r500, [r623];
ld.shared.u32 r501, [r623+4096];
{
add.f16x2 r496, r497, r498;
}
{
add.f16x2 r499, r500, r501;
}
{
sub.f16x2 r502, r497, r498;
}
{
sub.f16x2 r505, r500, r501;
}
bfe.u32 r624, r21, 9, 1;
and.b32 r625, r87, 2044;
add.s32 r626, r84, r625;
cvt.rn.f32.u32 f97, r624;
mul.f32 f98, f97, 0f3FC90FDB;
cos.approx.f32 f69, f98;
sin.approx.f32 f99, f98;
neg.f32 f70, f99;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r508, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r508;
mov.b32 r511, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r508;
mov.b32 r513, {high, high};
}
{
mul.f16x2 r515, r505, r513;
}
{
neg.f16x2 r518, r515;
}
{
fma.rn.f16x2 r520, r502, r511, r518;
}
{
mul.f16x2 r524, r502, r513;
}
{
fma.rn.f16x2 r527, r505, r511, r524;
}
barrier.sync 0;
and.b32 r627, r82, 4096;
add.s32 r628, r626, r627;
st.shared.u32 [r628], r496;
st.shared.u32 [r628+2048], r520;
barrier.sync 0;
and.b32 r629, r87, 2048;
sub.s32 r630, r628, r629;
ld.shared.u32 r549, [r630];
ld.shared.u32 r550, [r630+4096];
barrier.sync 0;
st.shared.u32 [r628], r499;
st.shared.u32 [r628+2048], r527;
barrier.sync 0;
ld.shared.u32 r552, [r630];
ld.shared.u32 r553, [r630+4096];
{
add.f16x2 %0, r549, r550;
}
{
add.f16x2 %1, r552, r553;
}
{
sub.f16x2 %2, r549, r550;
}
{
sub.f16x2 %3, r552, r553;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)));
};


#endif
