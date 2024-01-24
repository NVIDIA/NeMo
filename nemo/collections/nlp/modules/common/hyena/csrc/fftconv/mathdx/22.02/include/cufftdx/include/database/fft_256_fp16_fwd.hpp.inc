#ifndef CUFFTDX_FFT_256_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_256_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<807, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<270>;
.reg .b32 r<1803>;
.reg .b64 rd<2>;
mov.u32 r1791, %tid.y;
shl.b32 r1792, r1791, 10;
mov.u32 r1793, %32;
add.s32 r1794, r1793, r1792;
mov.u32 r1795, %tid.x;
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
mov.f32 f212, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r101, {low, high};
}
mov.f32 f230, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r102, {low, high};
}
mov.f32 f147, 0fBF800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r106, {low, high};
}
mov.f32 f148, 0f3F800000;
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
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
and.b32 r1796, r1795, 15;
shl.b32 r1797, r1795, 6;
and.b32 r1798, r1797, -1024;
add.s32 r1799, r1794, r1798;
cvt.rn.f32.u32 f267, r1796;
mul.f32 f268, f267, 0f3CC90FDB;
cos.approx.f32 f117, f268;
sin.approx.f32 f269, f268;
neg.f32 f118, f269;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
and.b32 r1800, r1797, 960;
add.s32 r1801, r1799, r1800;
st.shared.v4.f32 [r1801], {r521, r629, r666, r703};
st.shared.v4.f32 [r1801+16], {r740, r777, r814, r851};
st.shared.v4.f32 [r1801+32], {r888, r925, r962, r999};
st.shared.v4.f32 [r1801+48], {r1036, r1073, r1110, r1147};
barrier.sync 0;
mad.lo.s32 r1802, r1796, -60, r1801;
ld.shared.u32 r1176, [r1802];
ld.shared.u32 r1372, [r1802+64];
ld.shared.u32 r1226, [r1802+128];
ld.shared.u32 r1422, [r1802+192];
ld.shared.u32 r1188, [r1802+256];
ld.shared.u32 r1384, [r1802+320];
ld.shared.u32 r1238, [r1802+384];
ld.shared.u32 r1434, [r1802+448];
ld.shared.u32 r1177, [r1802+512];
ld.shared.u32 r1373, [r1802+576];
ld.shared.u32 r1227, [r1802+640];
ld.shared.u32 r1423, [r1802+704];
ld.shared.u32 r1189, [r1802+768];
ld.shared.u32 r1385, [r1802+832];
ld.shared.u32 r1239, [r1802+896];
ld.shared.u32 r1435, [r1802+960];
barrier.sync 0;
st.shared.v4.f32 [r1801], {r524, r636, r673, r710};
st.shared.v4.f32 [r1801+16], {r747, r784, r821, r858};
st.shared.v4.f32 [r1801+32], {r895, r932, r969, r1006};
st.shared.v4.f32 [r1801+48], {r1043, r1080, r1117, r1154};
barrier.sync 0;
ld.shared.u32 r1179, [r1802];
ld.shared.u32 r1375, [r1802+64];
ld.shared.u32 r1229, [r1802+128];
ld.shared.u32 r1425, [r1802+192];
ld.shared.u32 r1191, [r1802+256];
ld.shared.u32 r1387, [r1802+320];
ld.shared.u32 r1241, [r1802+384];
ld.shared.u32 r1437, [r1802+448];
ld.shared.u32 r1180, [r1802+512];
ld.shared.u32 r1376, [r1802+576];
ld.shared.u32 r1230, [r1802+640];
ld.shared.u32 r1426, [r1802+704];
ld.shared.u32 r1192, [r1802+768];
ld.shared.u32 r1388, [r1802+832];
ld.shared.u32 r1242, [r1802+896];
ld.shared.u32 r1438, [r1802+960];
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
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r1275, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r1276, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r1279, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r1471, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r1472, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r1475, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r1569, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r1577, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
add.f16x2 %0, r1323, r1519;
}
{
add.f16x2 %1, r1326, r1522;
}
{
sub.f16x2 %16, r1323, r1519;
}
{
sub.f16x2 %17, r1326, r1522;
}
{
add.f16x2 %2, r1335, r1603;
}
{
add.f16x2 %3, r1338, r1609;
}
{
sub.f16x2 %18, r1335, r1603;
}
{
sub.f16x2 %19, r1338, r1609;
}
{
add.f16x2 %4, r1347, r1619;
}
{
add.f16x2 %5, r1350, r1625;
}
{
sub.f16x2 %20, r1347, r1619;
}
{
sub.f16x2 %21, r1350, r1625;
}
{
add.f16x2 %6, r1359, r1635;
}
{
add.f16x2 %7, r1362, r1641;
}
{
sub.f16x2 %22, r1359, r1635;
}
{
sub.f16x2 %23, r1362, r1641;
}
{
add.f16x2 %8, r1329, r1528;
}
{
add.f16x2 %9, r1332, r1645;
}
{
sub.f16x2 %24, r1329, r1528;
}
{
sub.f16x2 %25, r1332, r1645;
}
{
add.f16x2 %10, r1341, r1653;
}
{
add.f16x2 %11, r1344, r1659;
}
{
sub.f16x2 %26, r1341, r1653;
}
{
sub.f16x2 %27, r1344, r1659;
}
{
add.f16x2 %12, r1353, r1669;
}
{
add.f16x2 %13, r1356, r1675;
}
{
sub.f16x2 %28, r1353, r1669;
}
{
sub.f16x2 %29, r1356, r1675;
}
{
add.f16x2 %14, r1365, r1685;
}
{
add.f16x2 %15, r1368, r1691;
}
{
sub.f16x2 %30, r1365, r1685;
}
{
sub.f16x2 %31, r1368, r1691;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<808, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<99>;
.reg .b32 r<1037>;
.reg .b64 rd<2>;
mov.u32 r1017, %tid.y;
shl.b32 r1018, r1017, 11;
mov.u32 r1019, %16;
add.s32 r1020, r1019, r1018;
mov.u32 r1021, %tid.x;
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
mov.f32 f48, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f48;
cvt.rn.f16.f32 high, f48;
mov.b32 r101, {low, high};
}
mov.f32 f58, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r102, {low, high};
}
mov.f32 f89, 0fBF800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r106, {low, high};
}
mov.f32 f90, 0f3F800000;
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
and.b32 r1022, r1021, 31;
shl.b32 r1023, r1021, 6;
and.b32 r1024, r1023, -2048;
add.s32 r1025, r1020, r1024;
cvt.rn.f32.u32 f93, r1022;
mul.f32 f94, f93, 0f3CC90FDB;
cos.approx.f32 f29, f94;
sin.approx.f32 f95, f94;
neg.f32 f30, f95;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
and.b32 r1026, r1023, 1984;
add.s32 r1027, r1025, r1026;
st.shared.v4.f32 [r1027], {r149, r152, r209, r216};
st.shared.v4.f32 [r1027+16], {r246, r253, r283, r290};
st.shared.v4.f32 [r1027+32], {r320, r327, r357, r364};
st.shared.v4.f32 [r1027+48], {r394, r401, r431, r438};
barrier.sync 0;
mad.lo.s32 r1028, r1022, -56, r1027;
ld.shared.u32 r460, [r1028];
ld.shared.u32 r463, [r1028+4];
ld.shared.u32 r510, [r1028+256];
ld.shared.u32 r513, [r1028+260];
ld.shared.u32 r472, [r1028+512];
ld.shared.u32 r475, [r1028+516];
ld.shared.u32 r522, [r1028+768];
ld.shared.u32 r525, [r1028+772];
ld.shared.u32 r461, [r1028+1024];
ld.shared.u32 r464, [r1028+1028];
ld.shared.u32 r511, [r1028+1280];
ld.shared.u32 r514, [r1028+1284];
ld.shared.u32 r473, [r1028+1536];
ld.shared.u32 r476, [r1028+1540];
ld.shared.u32 r523, [r1028+1792];
ld.shared.u32 r526, [r1028+1796];
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
cvt.rn.f16.f32 low, f48;
cvt.rn.f16.f32 high, f48;
mov.b32 r559, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r560, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r563, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
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
and.b32 r1029, r1021, 24;
bfe.u32 r1030, r1021, 3, 2;
cvt.rn.f32.u32 f96, r1030;
mul.f32 f97, f96, 0f3E490FDB;
cos.approx.f32 f75, f97;
sin.approx.f32 f98, f97;
neg.f32 f76, f98;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
shl.b32 r1031, r1021, 3;
and.b32 r1032, r1031, 56;
add.s32 r1033, r1025, r1032;
barrier.sync 0;
and.b32 r1034, r1023, 1536;
add.s32 r1035, r1033, r1034;
st.shared.u32 [r1035], r607;
st.shared.u32 [r1035+4], r610;
st.shared.u32 [r1035+64], r667;
st.shared.u32 [r1035+68], r674;
st.shared.u32 [r1035+128], r704;
st.shared.u32 [r1035+132], r711;
st.shared.u32 [r1035+192], r741;
st.shared.u32 [r1035+196], r748;
st.shared.u32 [r1035+256], r778;
st.shared.u32 [r1035+260], r785;
st.shared.u32 [r1035+320], r815;
st.shared.u32 [r1035+324], r822;
st.shared.u32 [r1035+384], r852;
st.shared.u32 [r1035+388], r859;
st.shared.u32 [r1035+448], r889;
st.shared.u32 [r1035+452], r896;
barrier.sync 0;
mad.lo.s32 r1036, r1029, -56, r1035;
ld.shared.u32 r918, [r1036];
ld.shared.u32 r921, [r1036+4];
ld.shared.u32 r968, [r1036+256];
ld.shared.u32 r971, [r1036+260];
ld.shared.u32 r930, [r1036+512];
ld.shared.u32 r933, [r1036+516];
ld.shared.u32 r980, [r1036+768];
ld.shared.u32 r983, [r1036+772];
ld.shared.u32 r919, [r1036+1024];
ld.shared.u32 r922, [r1036+1028];
ld.shared.u32 r969, [r1036+1280];
ld.shared.u32 r972, [r1036+1284];
ld.shared.u32 r931, [r1036+1536];
ld.shared.u32 r934, [r1036+1540];
ld.shared.u32 r981, [r1036+1792];
ld.shared.u32 r984, [r1036+1796];
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
add.f16x2 %0, r917, r929;
}
{
add.f16x2 %1, r920, r932;
}
{
sub.f16x2 %8, r917, r929;
}
{
sub.f16x2 %9, r920, r932;
}
{
add.f16x2 %4, r923, r938;
}
{
add.f16x2 %5, r926, r941;
}
{
sub.f16x2 %12, r923, r938;
}
{
sub.f16x2 %13, r926, r941;
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
add.f16x2 %2, r967, r979;
}
{
add.f16x2 %3, r970, r982;
}
{
sub.f16x2 %10, r967, r979;
}
{
sub.f16x2 %11, r970, r982;
}
{
add.f16x2 %6, r973, r988;
}
{
add.f16x2 %7, r976, r991;
}
{
sub.f16x2 %14, r973, r988;
}
{
sub.f16x2 %15, r976, r991;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<809, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<99>;
.reg .b32 r<1037>;
.reg .b64 rd<2>;
mov.u32 r1017, %tid.y;
shl.b32 r1018, r1017, 10;
mov.u32 r1019, %16;
add.s32 r1020, r1019, r1018;
mov.u32 r1021, %tid.x;
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
mov.f32 f48, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f48;
cvt.rn.f16.f32 high, f48;
mov.b32 r101, {low, high};
}
mov.f32 f58, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r102, {low, high};
}
mov.f32 f89, 0fBF800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r106, {low, high};
}
mov.f32 f90, 0f3F800000;
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
and.b32 r1022, r1021, 31;
shl.b32 r1023, r1021, 5;
and.b32 r1024, r1023, -1024;
add.s32 r1025, r1020, r1024;
cvt.rn.f32.u32 f93, r1022;
mul.f32 f94, f93, 0f3CC90FDB;
cos.approx.f32 f29, f94;
sin.approx.f32 f95, f94;
neg.f32 f30, f95;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
and.b32 r1026, r1023, 992;
add.s32 r1027, r1025, r1026;
st.shared.v4.f32 [r1027], {r149, r209, r246, r283};
st.shared.v4.f32 [r1027+16], {r320, r357, r394, r431};
barrier.sync 0;
mad.lo.s32 r1028, r1022, -28, r1027;
ld.shared.u32 r460, [r1028];
ld.shared.u32 r510, [r1028+128];
ld.shared.u32 r472, [r1028+256];
ld.shared.u32 r522, [r1028+384];
ld.shared.u32 r461, [r1028+512];
ld.shared.u32 r511, [r1028+640];
ld.shared.u32 r473, [r1028+768];
ld.shared.u32 r523, [r1028+896];
barrier.sync 0;
st.shared.v4.f32 [r1027], {r152, r216, r253, r290};
st.shared.v4.f32 [r1027+16], {r327, r364, r401, r438};
barrier.sync 0;
ld.shared.u32 r463, [r1028];
ld.shared.u32 r513, [r1028+128];
ld.shared.u32 r475, [r1028+256];
ld.shared.u32 r525, [r1028+384];
ld.shared.u32 r464, [r1028+512];
ld.shared.u32 r514, [r1028+640];
ld.shared.u32 r476, [r1028+768];
ld.shared.u32 r526, [r1028+896];
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
cvt.rn.f16.f32 low, f48;
cvt.rn.f16.f32 high, f48;
mov.b32 r559, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r560, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r563, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
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
and.b32 r1029, r1021, 24;
bfe.u32 r1030, r1021, 3, 2;
shl.b32 r1031, r1021, 2;
and.b32 r1032, r1031, 28;
add.s32 r1033, r1025, r1032;
cvt.rn.f32.u32 f96, r1030;
mul.f32 f97, f96, 0f3E490FDB;
cos.approx.f32 f75, f97;
sin.approx.f32 f98, f97;
neg.f32 f76, f98;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
cvt.rn.f16.f32 low, f89;
cvt.rn.f16.f32 high, f90;
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
and.b32 r1034, r1023, 768;
add.s32 r1035, r1033, r1034;
st.shared.u32 [r1035], r607;
st.shared.u32 [r1035+32], r667;
st.shared.u32 [r1035+64], r704;
st.shared.u32 [r1035+96], r741;
st.shared.u32 [r1035+128], r778;
st.shared.u32 [r1035+160], r815;
st.shared.u32 [r1035+192], r852;
st.shared.u32 [r1035+224], r889;
barrier.sync 0;
mad.lo.s32 r1036, r1029, -28, r1035;
ld.shared.u32 r918, [r1036];
ld.shared.u32 r968, [r1036+128];
ld.shared.u32 r930, [r1036+256];
ld.shared.u32 r980, [r1036+384];
ld.shared.u32 r919, [r1036+512];
ld.shared.u32 r969, [r1036+640];
ld.shared.u32 r931, [r1036+768];
ld.shared.u32 r981, [r1036+896];
barrier.sync 0;
st.shared.u32 [r1035], r610;
st.shared.u32 [r1035+32], r674;
st.shared.u32 [r1035+64], r711;
st.shared.u32 [r1035+96], r748;
st.shared.u32 [r1035+128], r785;
st.shared.u32 [r1035+160], r822;
st.shared.u32 [r1035+192], r859;
st.shared.u32 [r1035+224], r896;
barrier.sync 0;
ld.shared.u32 r921, [r1036];
ld.shared.u32 r971, [r1036+128];
ld.shared.u32 r933, [r1036+256];
ld.shared.u32 r983, [r1036+384];
ld.shared.u32 r922, [r1036+512];
ld.shared.u32 r972, [r1036+640];
ld.shared.u32 r934, [r1036+768];
ld.shared.u32 r984, [r1036+896];
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
add.f16x2 %0, r917, r929;
}
{
add.f16x2 %1, r920, r932;
}
{
sub.f16x2 %8, r917, r929;
}
{
sub.f16x2 %9, r920, r932;
}
{
add.f16x2 %4, r923, r938;
}
{
add.f16x2 %5, r926, r941;
}
{
sub.f16x2 %12, r923, r938;
}
{
sub.f16x2 %13, r926, r941;
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
add.f16x2 %2, r967, r979;
}
{
add.f16x2 %3, r970, r982;
}
{
sub.f16x2 %10, r967, r979;
}
{
sub.f16x2 %11, r970, r982;
}
{
add.f16x2 %6, r973, r988;
}
{
add.f16x2 %7, r976, r991;
}
{
sub.f16x2 %14, r973, r988;
}
{
sub.f16x2 %15, r976, r991;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<810, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<270>;
.reg .b32 r<1803>;
.reg .b64 rd<2>;
mov.u32 r1791, %tid.y;
shl.b32 r1792, r1791, 11;
mov.u32 r1793, %32;
add.s32 r1794, r1793, r1792;
mov.u32 r1795, %tid.x;
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
mov.f32 f212, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r101, {low, high};
}
mov.f32 f230, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r102, {low, high};
}
mov.f32 f147, 0fBF800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r106, {low, high};
}
mov.f32 f148, 0f3F800000;
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
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
and.b32 r1796, r1795, 15;
shl.b32 r1797, r1795, 7;
and.b32 r1798, r1797, -2048;
add.s32 r1799, r1794, r1798;
cvt.rn.f32.u32 f267, r1796;
mul.f32 f268, f267, 0f3CC90FDB;
cos.approx.f32 f117, f268;
sin.approx.f32 f269, f268;
neg.f32 f118, f269;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
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
and.b32 r1800, r1797, 1920;
add.s32 r1801, r1799, r1800;
st.shared.v4.f32 [r1801], {r521, r524, r629, r636};
st.shared.v4.f32 [r1801+16], {r666, r673, r703, r710};
st.shared.v4.f32 [r1801+32], {r740, r747, r777, r784};
st.shared.v4.f32 [r1801+48], {r814, r821, r851, r858};
st.shared.v4.f32 [r1801+64], {r888, r895, r925, r932};
st.shared.v4.f32 [r1801+80], {r962, r969, r999, r1006};
st.shared.v4.f32 [r1801+96], {r1036, r1043, r1073, r1080};
st.shared.v4.f32 [r1801+112], {r1110, r1117, r1147, r1154};
barrier.sync 0;
mad.lo.s32 r1802, r1796, -120, r1801;
ld.shared.u32 r1176, [r1802];
ld.shared.u32 r1179, [r1802+4];
ld.shared.u32 r1372, [r1802+128];
ld.shared.u32 r1375, [r1802+132];
ld.shared.u32 r1226, [r1802+256];
ld.shared.u32 r1229, [r1802+260];
ld.shared.u32 r1422, [r1802+384];
ld.shared.u32 r1425, [r1802+388];
ld.shared.u32 r1188, [r1802+512];
ld.shared.u32 r1191, [r1802+516];
ld.shared.u32 r1384, [r1802+640];
ld.shared.u32 r1387, [r1802+644];
ld.shared.u32 r1238, [r1802+768];
ld.shared.u32 r1241, [r1802+772];
ld.shared.u32 r1434, [r1802+896];
ld.shared.u32 r1437, [r1802+900];
ld.shared.u32 r1177, [r1802+1024];
ld.shared.u32 r1180, [r1802+1028];
ld.shared.u32 r1373, [r1802+1152];
ld.shared.u32 r1376, [r1802+1156];
ld.shared.u32 r1227, [r1802+1280];
ld.shared.u32 r1230, [r1802+1284];
ld.shared.u32 r1423, [r1802+1408];
ld.shared.u32 r1426, [r1802+1412];
ld.shared.u32 r1189, [r1802+1536];
ld.shared.u32 r1192, [r1802+1540];
ld.shared.u32 r1385, [r1802+1664];
ld.shared.u32 r1388, [r1802+1668];
ld.shared.u32 r1239, [r1802+1792];
ld.shared.u32 r1242, [r1802+1796];
ld.shared.u32 r1435, [r1802+1920];
ld.shared.u32 r1438, [r1802+1924];
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
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r1275, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r1276, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r1279, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r1471, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r1472, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r1475, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r1569, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r1577, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
add.f16x2 %0, r1323, r1519;
}
{
add.f16x2 %1, r1326, r1522;
}
{
sub.f16x2 %16, r1323, r1519;
}
{
sub.f16x2 %17, r1326, r1522;
}
{
add.f16x2 %2, r1335, r1603;
}
{
add.f16x2 %3, r1338, r1609;
}
{
sub.f16x2 %18, r1335, r1603;
}
{
sub.f16x2 %19, r1338, r1609;
}
{
add.f16x2 %4, r1347, r1619;
}
{
add.f16x2 %5, r1350, r1625;
}
{
sub.f16x2 %20, r1347, r1619;
}
{
sub.f16x2 %21, r1350, r1625;
}
{
add.f16x2 %6, r1359, r1635;
}
{
add.f16x2 %7, r1362, r1641;
}
{
sub.f16x2 %22, r1359, r1635;
}
{
sub.f16x2 %23, r1362, r1641;
}
{
add.f16x2 %8, r1329, r1528;
}
{
add.f16x2 %9, r1332, r1645;
}
{
sub.f16x2 %24, r1329, r1528;
}
{
sub.f16x2 %25, r1332, r1645;
}
{
add.f16x2 %10, r1341, r1653;
}
{
add.f16x2 %11, r1344, r1659;
}
{
sub.f16x2 %26, r1341, r1653;
}
{
sub.f16x2 %27, r1344, r1659;
}
{
add.f16x2 %12, r1353, r1669;
}
{
add.f16x2 %13, r1356, r1675;
}
{
sub.f16x2 %28, r1353, r1669;
}
{
sub.f16x2 %29, r1356, r1675;
}
{
add.f16x2 %14, r1365, r1685;
}
{
add.f16x2 %15, r1368, r1691;
}
{
sub.f16x2 %30, r1365, r1685;
}
{
sub.f16x2 %31, r1368, r1691;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<811, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<540>;
.reg .b32 r<3723>;
.reg .b64 rd<3>;
mov.u32 r3647, %tid.y;
shl.b32 r3648, r3647, 11;
mov.u32 r3649, %64;
add.s32 r3650, r3649, r3648;
mov.u32 r3651, %tid.x;
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
mov.f32 f508, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r101, {low, high};
}
mov.f32 f518, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r102, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
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
add.f16x2 r247, %79, %72;
}
{
add.f16x2 r250, %117, %108;
}
{
sub.f16x2 r253, %79, %72;
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
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
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
mov.f32 f238, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r393, {low, high};
}
mov.f32 f288, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r396, {low, high};
}
mov.f32 f254, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f254;
cvt.rn.f16.f32 high, f254;
mov.b32 r397, {low, high};
}
mov.f32 f286, 0fBF6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r398, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r404, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r405, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
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
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r717, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r718, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r721, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
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
add.f16x2 r816, %78, %71;
}
{
sub.f16x2 r819, %109, %100;
}
{
sub.f16x2 r822, %78, %71;
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
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r913, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r914, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r917, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1009, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r1010, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r1011, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r1012, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f254;
cvt.rn.f16.f32 high, f254;
mov.b32 r1013, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r1014, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r1017, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r1018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r1019, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r1020, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r1021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
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
mov.f32 f234, 0f3F7B14BE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f234;
cvt.rn.f16.f32 high, f234;
mov.b32 r1233, {low, high};
}
mov.f32 f292, 0fBE47C5C2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f292;
cvt.rn.f16.f32 high, f292;
mov.b32 r1234, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1235, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r1236, {low, high};
}
mov.f32 f242, 0f3F54DB31;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f242;
cvt.rn.f16.f32 high, f242;
mov.b32 r1237, {low, high};
}
mov.f32 f284, 0fBF0E39DA;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r1238, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r1239, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r1240, {low, high};
}
mov.f32 f250, 0f3F0E39DA;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f250;
cvt.rn.f16.f32 high, f250;
mov.b32 r1241, {low, high};
}
mov.f32 f282, 0fBF54DB31;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r1242, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f254;
cvt.rn.f16.f32 high, f254;
mov.b32 r1243, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r1244, {low, high};
}
mov.f32 f258, 0f3E47C5C2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f258;
cvt.rn.f16.f32 high, f258;
mov.b32 r1245, {low, high};
}
mov.f32 f290, 0fBF7B14BE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f290;
cvt.rn.f16.f32 high, f290;
mov.b32 r1246, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f292;
cvt.rn.f16.f32 high, f292;
mov.b32 r1249, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f290;
cvt.rn.f16.f32 high, f290;
mov.b32 r1250, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r1251, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r1252, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r1253, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r1254, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r1255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r1256, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r1257, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r1258, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r1259, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r1260, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f290;
cvt.rn.f16.f32 high, f290;
mov.b32 r1261, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f292;
cvt.rn.f16.f32 high, f292;
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
and.b32 r3652, r3651, 7;
shl.b32 r3653, r3651, 8;
and.b32 r3654, r3653, -2048;
add.s32 r3655, r3650, r3654;
cvt.rn.f32.u32 f535, r3652;
mul.f32 f536, f535, 0f3CC90FDB;
cos.approx.f32 f357, f536;
sin.approx.f32 f537, f536;
neg.f32 f358, f537;
mov.f32 f539, 0fBF800000;
mov.f32 f538, 0f3F800000;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
and.b32 r3656, r3653, 1792;
add.s32 r3657, r3655, r3656;
st.shared.v4.f32 [r3657], {r1521, r1524, r1725, r1732};
st.shared.v4.f32 [r3657+16], {r1762, r1769, r1799, r1806};
st.shared.v4.f32 [r3657+32], {r1836, r1843, r1873, r1880};
st.shared.v4.f32 [r3657+48], {r1910, r1917, r1947, r1954};
st.shared.v4.f32 [r3657+64], {r1984, r1991, r2021, r2028};
st.shared.v4.f32 [r3657+80], {r2058, r2065, r2095, r2102};
st.shared.v4.f32 [r3657+96], {r2132, r2139, r2169, r2176};
st.shared.v4.f32 [r3657+112], {r2206, r2213, r2243, r2250};
st.shared.v4.f32 [r3657+128], {r2280, r2287, r2317, r2324};
st.shared.v4.f32 [r3657+144], {r2354, r2361, r2391, r2398};
st.shared.v4.f32 [r3657+160], {r2428, r2435, r2465, r2472};
st.shared.v4.f32 [r3657+176], {r2502, r2509, r2539, r2546};
st.shared.v4.f32 [r3657+192], {r2576, r2583, r2613, r2620};
st.shared.v4.f32 [r3657+208], {r2650, r2657, r2687, r2694};
st.shared.v4.f32 [r3657+224], {r2724, r2731, r2761, r2768};
st.shared.v4.f32 [r3657+240], {r2798, r2805, r2835, r2842};
barrier.sync 0;
mad.lo.s32 r3658, r3652, -248, r3657;
ld.shared.u32 r2864, [r3658];
ld.shared.u32 r2867, [r3658+4];
ld.shared.u32 r3060, [r3658+64];
ld.shared.u32 r3063, [r3658+68];
ld.shared.u32 r3256, [r3658+128];
ld.shared.u32 r3259, [r3658+132];
ld.shared.u32 r3452, [r3658+192];
ld.shared.u32 r3455, [r3658+196];
ld.shared.u32 r2914, [r3658+256];
ld.shared.u32 r2917, [r3658+260];
ld.shared.u32 r3110, [r3658+320];
ld.shared.u32 r3113, [r3658+324];
ld.shared.u32 r3306, [r3658+384];
ld.shared.u32 r3309, [r3658+388];
ld.shared.u32 r3502, [r3658+448];
ld.shared.u32 r3505, [r3658+452];
ld.shared.u32 r2876, [r3658+512];
ld.shared.u32 r2879, [r3658+516];
ld.shared.u32 r3072, [r3658+576];
ld.shared.u32 r3075, [r3658+580];
ld.shared.u32 r3268, [r3658+640];
ld.shared.u32 r3271, [r3658+644];
ld.shared.u32 r3464, [r3658+704];
ld.shared.u32 r3467, [r3658+708];
ld.shared.u32 r2926, [r3658+768];
ld.shared.u32 r2929, [r3658+772];
ld.shared.u32 r3122, [r3658+832];
ld.shared.u32 r3125, [r3658+836];
ld.shared.u32 r3318, [r3658+896];
ld.shared.u32 r3321, [r3658+900];
ld.shared.u32 r3514, [r3658+960];
ld.shared.u32 r3517, [r3658+964];
ld.shared.u32 r2865, [r3658+1024];
ld.shared.u32 r2868, [r3658+1028];
ld.shared.u32 r3061, [r3658+1088];
ld.shared.u32 r3064, [r3658+1092];
ld.shared.u32 r3257, [r3658+1152];
ld.shared.u32 r3260, [r3658+1156];
ld.shared.u32 r3453, [r3658+1216];
ld.shared.u32 r3456, [r3658+1220];
ld.shared.u32 r2915, [r3658+1280];
ld.shared.u32 r2918, [r3658+1284];
ld.shared.u32 r3111, [r3658+1344];
ld.shared.u32 r3114, [r3658+1348];
ld.shared.u32 r3307, [r3658+1408];
ld.shared.u32 r3310, [r3658+1412];
ld.shared.u32 r3503, [r3658+1472];
ld.shared.u32 r3506, [r3658+1476];
ld.shared.u32 r2877, [r3658+1536];
ld.shared.u32 r2880, [r3658+1540];
ld.shared.u32 r3073, [r3658+1600];
ld.shared.u32 r3076, [r3658+1604];
ld.shared.u32 r3269, [r3658+1664];
ld.shared.u32 r3272, [r3658+1668];
ld.shared.u32 r3465, [r3658+1728];
ld.shared.u32 r3468, [r3658+1732];
ld.shared.u32 r2927, [r3658+1792];
ld.shared.u32 r2930, [r3658+1796];
ld.shared.u32 r3123, [r3658+1856];
ld.shared.u32 r3126, [r3658+1860];
ld.shared.u32 r3319, [r3658+1920];
ld.shared.u32 r3322, [r3658+1924];
ld.shared.u32 r3515, [r3658+1984];
ld.shared.u32 r3518, [r3658+1988];
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
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r2963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r2964, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r2967, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
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
add.f16x2 %0, r2889, r2939;
}
{
add.f16x2 %1, r2892, r2942;
}
{
sub.f16x2 %32, r2889, r2939;
}
{
sub.f16x2 %33, r2892, r2942;
}
{
add.f16x2 %8, r2901, r2983;
}
{
add.f16x2 %9, r2904, r2989;
}
{
sub.f16x2 %40, r2901, r2983;
}
{
sub.f16x2 %41, r2904, r2989;
}
{
add.f16x2 %16, r2895, r2948;
}
{
add.f16x2 %17, r2898, r2993;
}
{
sub.f16x2 %48, r2895, r2948;
}
{
sub.f16x2 %49, r2898, r2993;
}
{
add.f16x2 %24, r2907, r3001;
}
{
add.f16x2 %25, r2910, r3007;
}
{
sub.f16x2 %56, r2907, r3001;
}
{
sub.f16x2 %57, r2910, r3007;
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
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r3159, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r3160, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r3163, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
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
add.f16x2 %2, r3085, r3135;
}
{
add.f16x2 %3, r3088, r3138;
}
{
sub.f16x2 %34, r3085, r3135;
}
{
sub.f16x2 %35, r3088, r3138;
}
{
add.f16x2 %10, r3097, r3179;
}
{
add.f16x2 %11, r3100, r3185;
}
{
sub.f16x2 %42, r3097, r3179;
}
{
sub.f16x2 %43, r3100, r3185;
}
{
add.f16x2 %18, r3091, r3144;
}
{
add.f16x2 %19, r3094, r3189;
}
{
sub.f16x2 %50, r3091, r3144;
}
{
sub.f16x2 %51, r3094, r3189;
}
{
add.f16x2 %26, r3103, r3197;
}
{
add.f16x2 %27, r3106, r3203;
}
{
sub.f16x2 %58, r3103, r3197;
}
{
sub.f16x2 %59, r3106, r3203;
}
{
add.f16x2 r3255, r3256, r3257;
}
{
add.f16x2 r3258, r3259, r3260;
}
{
sub.f16x2 r3261, r3256, r3257;
}
{
sub.f16x2 r3264, r3259, r3260;
}
{
add.f16x2 r3267, r3268, r3269;
}
{
add.f16x2 r3270, r3271, r3272;
}
{
sub.f16x2 r3273, r3268, r3269;
}
{
sub.f16x2 r3276, r3271, r3272;
}
{
neg.f16x2 r3279, r3273;
}
{
add.f16x2 r3281, r3255, r3267;
}
{
add.f16x2 r3284, r3258, r3270;
}
{
sub.f16x2 r3287, r3255, r3267;
}
{
sub.f16x2 r3290, r3258, r3270;
}
{
add.f16x2 r3293, r3261, r3276;
}
{
add.f16x2 r3296, r3264, r3279;
}
{
sub.f16x2 r3299, r3261, r3276;
}
{
sub.f16x2 r3302, r3264, r3279;
}
{
add.f16x2 r3305, r3306, r3307;
}
{
add.f16x2 r3308, r3309, r3310;
}
{
sub.f16x2 r3311, r3306, r3307;
}
{
sub.f16x2 r3314, r3309, r3310;
}
{
add.f16x2 r3317, r3318, r3319;
}
{
add.f16x2 r3320, r3321, r3322;
}
{
sub.f16x2 r3323, r3318, r3319;
}
{
sub.f16x2 r3326, r3321, r3322;
}
{
neg.f16x2 r3329, r3323;
}
{
add.f16x2 r3331, r3305, r3317;
}
{
add.f16x2 r3334, r3308, r3320;
}
{
sub.f16x2 r3337, r3305, r3317;
}
{
sub.f16x2 r3340, r3308, r3320;
}
{
add.f16x2 r3343, r3311, r3326;
}
{
add.f16x2 r3346, r3314, r3329;
}
{
sub.f16x2 r3349, r3311, r3326;
}
{
sub.f16x2 r3352, r3314, r3329;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r3355, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r3356, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r3359, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r3360, {low, high};
}
{
mul.f16x2 r3369, r3343, r3355;
}
{
mul.f16x2 r3372, r3346, r3356;
}
{
sub.f16x2 r3375, r3369, r3372;
}
{
mul.f16x2 r3378, r3343, r3356;
}
{
fma.rn.f16x2 r3381, r3346, r3355, r3378;
}
{
neg.f16x2 r3385, r3337;
}
{
mul.f16x2 r3387, r3349, r3359;
}
{
mul.f16x2 r3390, r3352, r3360;
}
{
sub.f16x2 r3393, r3387, r3390;
}
{
mul.f16x2 r3396, r3349, r3360;
}
{
fma.rn.f16x2 r3399, r3352, r3359, r3396;
}
{
add.f16x2 %4, r3281, r3331;
}
{
add.f16x2 %5, r3284, r3334;
}
{
sub.f16x2 %36, r3281, r3331;
}
{
sub.f16x2 %37, r3284, r3334;
}
{
add.f16x2 %12, r3293, r3375;
}
{
add.f16x2 %13, r3296, r3381;
}
{
sub.f16x2 %44, r3293, r3375;
}
{
sub.f16x2 %45, r3296, r3381;
}
{
add.f16x2 %20, r3287, r3340;
}
{
add.f16x2 %21, r3290, r3385;
}
{
sub.f16x2 %52, r3287, r3340;
}
{
sub.f16x2 %53, r3290, r3385;
}
{
add.f16x2 %28, r3299, r3393;
}
{
add.f16x2 %29, r3302, r3399;
}
{
sub.f16x2 %60, r3299, r3393;
}
{
sub.f16x2 %61, r3302, r3399;
}
{
add.f16x2 r3451, r3452, r3453;
}
{
add.f16x2 r3454, r3455, r3456;
}
{
sub.f16x2 r3457, r3452, r3453;
}
{
sub.f16x2 r3460, r3455, r3456;
}
{
add.f16x2 r3463, r3464, r3465;
}
{
add.f16x2 r3466, r3467, r3468;
}
{
sub.f16x2 r3469, r3464, r3465;
}
{
sub.f16x2 r3472, r3467, r3468;
}
{
neg.f16x2 r3475, r3469;
}
{
add.f16x2 r3477, r3451, r3463;
}
{
add.f16x2 r3480, r3454, r3466;
}
{
sub.f16x2 r3483, r3451, r3463;
}
{
sub.f16x2 r3486, r3454, r3466;
}
{
add.f16x2 r3489, r3457, r3472;
}
{
add.f16x2 r3492, r3460, r3475;
}
{
sub.f16x2 r3495, r3457, r3472;
}
{
sub.f16x2 r3498, r3460, r3475;
}
{
add.f16x2 r3501, r3502, r3503;
}
{
add.f16x2 r3504, r3505, r3506;
}
{
sub.f16x2 r3507, r3502, r3503;
}
{
sub.f16x2 r3510, r3505, r3506;
}
{
add.f16x2 r3513, r3514, r3515;
}
{
add.f16x2 r3516, r3517, r3518;
}
{
sub.f16x2 r3519, r3514, r3515;
}
{
sub.f16x2 r3522, r3517, r3518;
}
{
neg.f16x2 r3525, r3519;
}
{
add.f16x2 r3527, r3501, r3513;
}
{
add.f16x2 r3530, r3504, r3516;
}
{
sub.f16x2 r3533, r3501, r3513;
}
{
sub.f16x2 r3536, r3504, r3516;
}
{
add.f16x2 r3539, r3507, r3522;
}
{
add.f16x2 r3542, r3510, r3525;
}
{
sub.f16x2 r3545, r3507, r3522;
}
{
sub.f16x2 r3548, r3510, r3525;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r3551, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r3552, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r3555, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r3556, {low, high};
}
{
mul.f16x2 r3565, r3539, r3551;
}
{
mul.f16x2 r3568, r3542, r3552;
}
{
sub.f16x2 r3571, r3565, r3568;
}
{
mul.f16x2 r3574, r3539, r3552;
}
{
fma.rn.f16x2 r3577, r3542, r3551, r3574;
}
{
neg.f16x2 r3581, r3533;
}
{
mul.f16x2 r3583, r3545, r3555;
}
{
mul.f16x2 r3586, r3548, r3556;
}
{
sub.f16x2 r3589, r3583, r3586;
}
{
mul.f16x2 r3592, r3545, r3556;
}
{
fma.rn.f16x2 r3595, r3548, r3555, r3592;
}
{
add.f16x2 %6, r3477, r3527;
}
{
add.f16x2 %7, r3480, r3530;
}
{
sub.f16x2 %38, r3477, r3527;
}
{
sub.f16x2 %39, r3480, r3530;
}
{
add.f16x2 %14, r3489, r3571;
}
{
add.f16x2 %15, r3492, r3577;
}
{
sub.f16x2 %46, r3489, r3571;
}
{
sub.f16x2 %47, r3492, r3577;
}
{
add.f16x2 %22, r3483, r3536;
}
{
add.f16x2 %23, r3486, r3581;
}
{
sub.f16x2 %54, r3483, r3536;
}
{
sub.f16x2 %55, r3486, r3581;
}
{
add.f16x2 %30, r3495, r3589;
}
{
add.f16x2 %31, r3498, r3595;
}
{
sub.f16x2 %62, r3495, r3589;
}
{
sub.f16x2 %63, r3498, r3595;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)), "=r"(__HALF2_TO_UI(rmem[27].x)), "=r"(__HALF2_TO_UI(rmem[27].y)), "=r"(__HALF2_TO_UI(rmem[28].x)), "=r"(__HALF2_TO_UI(rmem[28].y)), "=r"(__HALF2_TO_UI(rmem[29].x)), "=r"(__HALF2_TO_UI(rmem[29].y)), "=r"(__HALF2_TO_UI(rmem[30].x)), "=r"(__HALF2_TO_UI(rmem[30].y)), "=r"(__HALF2_TO_UI(rmem[31].x)), "=r"(__HALF2_TO_UI(rmem[31].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[31].y)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[28].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[27].y)), "r"(__HALF2_TO_UI(rmem[30].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[29].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[31].x)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[28].x)), "r"(__HALF2_TO_UI(rmem[27].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[30].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[29].x)), "r"(__HALF2_TO_UI(rmem[24].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<812, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<40>;
.reg .b32 r<570>;
.reg .b64 rd<2>;
mov.u32 r543, %tid.y;
shl.b32 r544, r543, 11;
mov.u32 r545, %8;
add.s32 r546, r545, r544;
mov.u32 r547, %tid.x;
{
add.f16x2 r1, %9, %13;
}
{
add.f16x2 r4, %10, %14;
}
{
sub.f16x2 r7, %9, %13;
}
{
sub.f16x2 r10, %10, %14;
}
{
add.f16x2 r13, %11, %15;
}
{
add.f16x2 r16, %12, %16;
}
{
sub.f16x2 r19, %11, %15;
}
{
sub.f16x2 r22, %12, %16;
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
and.b32 r548, r547, 63;
shl.b32 r549, r547, 5;
and.b32 r550, r549, -2048;
add.s32 r551, r546, r550;
cvt.rn.f32.u32 f31, r548;
mul.f32 f32, f31, 0f3CC90FDB;
cos.approx.f32 f1, f32;
sin.approx.f32 f33, f32;
neg.f32 f2, f33;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1;
cvt.rn.f16.f32 high, f2;
mov.b32 r51, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r54, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r56, {high, high};
}
{
mul.f16x2 r58, r42, r56;
}
{
neg.f16x2 r61, r58;
}
{
fma.rn.f16x2 r63, r39, r54, r61;
}
{
mul.f16x2 r67, r39, r56;
}
{
fma.rn.f16x2 r70, r42, r54, r67;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r74, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r76, {high, high};
}
mov.f32 f27, 0fBF800000;
mov.f32 f28, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f27;
cvt.rn.f16.f32 high, f28;
mov.b32 r78, {low, high};
}
{
mul.f16x2 r79, r76, r78;
}
{
mul.f16x2 r82, r51, r74;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r85, {high, low};
}
{
fma.rn.f16x2 r87, r79, r85, r82;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r91, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r93, {high, high};
}
{
mul.f16x2 r95, r36, r93;
}
{
neg.f16x2 r98, r95;
}
{
fma.rn.f16x2 r100, r33, r91, r98;
}
{
mul.f16x2 r104, r33, r93;
}
{
fma.rn.f16x2 r107, r36, r91, r104;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r111, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r113, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f27;
cvt.rn.f16.f32 high, f28;
mov.b32 r115, {low, high};
}
{
mul.f16x2 r116, r113, r115;
}
{
mul.f16x2 r119, r87, r111;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r122, {high, low};
}
{
fma.rn.f16x2 r124, r116, r122, r119;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r124;
mov.b32 r128, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r124;
mov.b32 r130, {high, high};
}
{
mul.f16x2 r132, r48, r130;
}
{
neg.f16x2 r135, r132;
}
{
fma.rn.f16x2 r137, r45, r128, r135;
}
{
mul.f16x2 r141, r45, r130;
}
{
fma.rn.f16x2 r144, r48, r128, r141;
}
barrier.sync 0;
and.b32 r552, r549, 2016;
add.s32 r553, r551, r552;
st.shared.v4.f32 [r553], {r27, r30, r63, r70};
st.shared.v4.f32 [r553+16], {r100, r107, r137, r144};
barrier.sync 0;
mad.lo.s32 r554, r548, -24, r553;
ld.shared.u32 r166, [r554];
ld.shared.u32 r169, [r554+4];
ld.shared.u32 r178, [r554+512];
ld.shared.u32 r181, [r554+516];
ld.shared.u32 r167, [r554+1024];
ld.shared.u32 r170, [r554+1028];
ld.shared.u32 r179, [r554+1536];
ld.shared.u32 r182, [r554+1540];
{
add.f16x2 r165, r166, r167;
}
{
add.f16x2 r168, r169, r170;
}
{
sub.f16x2 r171, r166, r167;
}
{
sub.f16x2 r174, r169, r170;
}
{
add.f16x2 r177, r178, r179;
}
{
add.f16x2 r180, r181, r182;
}
{
sub.f16x2 r183, r178, r179;
}
{
sub.f16x2 r186, r181, r182;
}
{
neg.f16x2 r189, r183;
}
{
add.f16x2 r191, r165, r177;
}
{
add.f16x2 r194, r168, r180;
}
{
sub.f16x2 r197, r165, r177;
}
{
sub.f16x2 r200, r168, r180;
}
{
add.f16x2 r203, r171, r186;
}
{
add.f16x2 r206, r174, r189;
}
{
sub.f16x2 r209, r171, r186;
}
{
sub.f16x2 r212, r174, r189;
}
and.b32 r555, r547, 60;
bfe.u32 r556, r547, 2, 4;
cvt.rn.f32.u32 f34, r556;
mul.f32 f35, f34, 0f3DC90FDB;
cos.approx.f32 f11, f35;
sin.approx.f32 f36, f35;
neg.f32 f12, f36;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f11;
cvt.rn.f16.f32 high, f12;
mov.b32 r215, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r218, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r220, {high, high};
}
{
mul.f16x2 r222, r206, r220;
}
{
neg.f16x2 r225, r222;
}
{
fma.rn.f16x2 r227, r203, r218, r225;
}
{
mul.f16x2 r231, r203, r220;
}
{
fma.rn.f16x2 r234, r206, r218, r231;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r238, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r240, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f27;
cvt.rn.f16.f32 high, f28;
mov.b32 r242, {low, high};
}
{
mul.f16x2 r243, r240, r242;
}
{
mul.f16x2 r246, r215, r238;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r249, {high, low};
}
{
fma.rn.f16x2 r251, r243, r249, r246;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r255, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r257, {high, high};
}
{
mul.f16x2 r259, r200, r257;
}
{
neg.f16x2 r262, r259;
}
{
fma.rn.f16x2 r264, r197, r255, r262;
}
{
mul.f16x2 r268, r197, r257;
}
{
fma.rn.f16x2 r271, r200, r255, r268;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r275, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r277, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f27;
cvt.rn.f16.f32 high, f28;
mov.b32 r279, {low, high};
}
{
mul.f16x2 r280, r277, r279;
}
{
mul.f16x2 r283, r251, r275;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r286, {high, low};
}
{
fma.rn.f16x2 r288, r280, r286, r283;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r288;
mov.b32 r292, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r288;
mov.b32 r294, {high, high};
}
{
mul.f16x2 r296, r212, r294;
}
{
neg.f16x2 r299, r296;
}
{
fma.rn.f16x2 r301, r209, r292, r299;
}
{
mul.f16x2 r305, r209, r294;
}
{
fma.rn.f16x2 r308, r212, r292, r305;
}
shl.b32 r557, r547, 3;
and.b32 r558, r557, 24;
add.s32 r559, r551, r558;
barrier.sync 0;
and.b32 r560, r549, 1920;
add.s32 r561, r559, r560;
st.shared.u32 [r561], r191;
st.shared.u32 [r561+4], r194;
st.shared.u32 [r561+32], r227;
st.shared.u32 [r561+36], r234;
st.shared.u32 [r561+64], r264;
st.shared.u32 [r561+68], r271;
st.shared.u32 [r561+96], r301;
st.shared.u32 [r561+100], r308;
barrier.sync 0;
mad.lo.s32 r562, r555, -24, r561;
ld.shared.u32 r330, [r562];
ld.shared.u32 r333, [r562+4];
ld.shared.u32 r342, [r562+512];
ld.shared.u32 r345, [r562+516];
ld.shared.u32 r331, [r562+1024];
ld.shared.u32 r334, [r562+1028];
ld.shared.u32 r343, [r562+1536];
ld.shared.u32 r346, [r562+1540];
{
add.f16x2 r329, r330, r331;
}
{
add.f16x2 r332, r333, r334;
}
{
sub.f16x2 r335, r330, r331;
}
{
sub.f16x2 r338, r333, r334;
}
{
add.f16x2 r341, r342, r343;
}
{
add.f16x2 r344, r345, r346;
}
{
sub.f16x2 r347, r342, r343;
}
{
sub.f16x2 r350, r345, r346;
}
{
neg.f16x2 r353, r347;
}
{
add.f16x2 r355, r329, r341;
}
{
add.f16x2 r358, r332, r344;
}
{
sub.f16x2 r361, r329, r341;
}
{
sub.f16x2 r364, r332, r344;
}
{
add.f16x2 r367, r335, r350;
}
{
add.f16x2 r370, r338, r353;
}
{
sub.f16x2 r373, r335, r350;
}
{
sub.f16x2 r376, r338, r353;
}
and.b32 r563, r547, 48;
bfe.u32 r564, r547, 4, 2;
cvt.rn.f32.u32 f37, r564;
mul.f32 f38, f37, 0f3EC90FDB;
cos.approx.f32 f21, f38;
sin.approx.f32 f39, f38;
neg.f32 f22, f39;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f21;
cvt.rn.f16.f32 high, f22;
mov.b32 r379, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r379;
mov.b32 r382, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r379;
mov.b32 r384, {high, high};
}
{
mul.f16x2 r386, r370, r384;
}
{
neg.f16x2 r389, r386;
}
{
fma.rn.f16x2 r391, r367, r382, r389;
}
{
mul.f16x2 r395, r367, r384;
}
{
fma.rn.f16x2 r398, r370, r382, r395;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r379;
mov.b32 r402, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r379;
mov.b32 r404, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f27;
cvt.rn.f16.f32 high, f28;
mov.b32 r406, {low, high};
}
{
mul.f16x2 r407, r404, r406;
}
{
mul.f16x2 r410, r379, r402;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r379;
mov.b32 r413, {high, low};
}
{
fma.rn.f16x2 r415, r407, r413, r410;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r415;
mov.b32 r419, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r415;
mov.b32 r421, {high, high};
}
{
mul.f16x2 r423, r364, r421;
}
{
neg.f16x2 r426, r423;
}
{
fma.rn.f16x2 r428, r361, r419, r426;
}
{
mul.f16x2 r432, r361, r421;
}
{
fma.rn.f16x2 r435, r364, r419, r432;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r379;
mov.b32 r439, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r379;
mov.b32 r441, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f27;
cvt.rn.f16.f32 high, f28;
mov.b32 r443, {low, high};
}
{
mul.f16x2 r444, r441, r443;
}
{
mul.f16x2 r447, r415, r439;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r415;
mov.b32 r450, {high, low};
}
{
fma.rn.f16x2 r452, r444, r450, r447;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r452;
mov.b32 r456, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r452;
mov.b32 r458, {high, high};
}
{
mul.f16x2 r460, r376, r458;
}
{
neg.f16x2 r463, r460;
}
{
fma.rn.f16x2 r465, r373, r456, r463;
}
{
mul.f16x2 r469, r373, r458;
}
{
fma.rn.f16x2 r472, r376, r456, r469;
}
and.b32 r565, r557, 120;
add.s32 r566, r551, r565;
barrier.sync 0;
and.b32 r567, r549, 1536;
add.s32 r568, r566, r567;
st.shared.u32 [r568], r355;
st.shared.u32 [r568+4], r358;
st.shared.u32 [r568+128], r391;
st.shared.u32 [r568+132], r398;
st.shared.u32 [r568+256], r428;
st.shared.u32 [r568+260], r435;
st.shared.u32 [r568+384], r465;
st.shared.u32 [r568+388], r472;
barrier.sync 0;
mad.lo.s32 r569, r563, -24, r568;
ld.shared.u32 r494, [r569];
ld.shared.u32 r497, [r569+4];
ld.shared.u32 r506, [r569+512];
ld.shared.u32 r509, [r569+516];
ld.shared.u32 r495, [r569+1024];
ld.shared.u32 r498, [r569+1028];
ld.shared.u32 r507, [r569+1536];
ld.shared.u32 r510, [r569+1540];
{
add.f16x2 r493, r494, r495;
}
{
add.f16x2 r496, r497, r498;
}
{
sub.f16x2 r499, r494, r495;
}
{
sub.f16x2 r502, r497, r498;
}
{
add.f16x2 r505, r506, r507;
}
{
add.f16x2 r508, r509, r510;
}
{
sub.f16x2 r511, r506, r507;
}
{
sub.f16x2 r514, r509, r510;
}
{
neg.f16x2 r517, r511;
}
{
add.f16x2 %0, r493, r505;
}
{
add.f16x2 %1, r496, r508;
}
{
sub.f16x2 %4, r493, r505;
}
{
sub.f16x2 %5, r496, r508;
}
{
add.f16x2 %2, r499, r514;
}
{
add.f16x2 %3, r502, r517;
}
{
sub.f16x2 %6, r499, r514;
}
{
sub.f16x2 %7, r502, r517;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<813, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<40>;
.reg .b32 r<570>;
.reg .b64 rd<2>;
mov.u32 r543, %tid.y;
shl.b32 r544, r543, 10;
mov.u32 r545, %8;
add.s32 r546, r545, r544;
mov.u32 r547, %tid.x;
{
add.f16x2 r1, %9, %13;
}
{
add.f16x2 r4, %10, %14;
}
{
sub.f16x2 r7, %9, %13;
}
{
sub.f16x2 r10, %10, %14;
}
{
add.f16x2 r13, %11, %15;
}
{
add.f16x2 r16, %12, %16;
}
{
sub.f16x2 r19, %11, %15;
}
{
sub.f16x2 r22, %12, %16;
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
and.b32 r548, r547, 63;
shl.b32 r549, r547, 4;
and.b32 r550, r549, -1024;
add.s32 r551, r546, r550;
cvt.rn.f32.u32 f31, r548;
mul.f32 f32, f31, 0f3CC90FDB;
cos.approx.f32 f1, f32;
sin.approx.f32 f33, f32;
neg.f32 f2, f33;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1;
cvt.rn.f16.f32 high, f2;
mov.b32 r51, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r54, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r56, {high, high};
}
{
mul.f16x2 r58, r42, r56;
}
{
neg.f16x2 r61, r58;
}
{
fma.rn.f16x2 r63, r39, r54, r61;
}
{
mul.f16x2 r67, r39, r56;
}
{
fma.rn.f16x2 r70, r42, r54, r67;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r74, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r76, {high, high};
}
mov.f32 f27, 0fBF800000;
mov.f32 f28, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f27;
cvt.rn.f16.f32 high, f28;
mov.b32 r78, {low, high};
}
{
mul.f16x2 r79, r76, r78;
}
{
mul.f16x2 r82, r51, r74;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r85, {high, low};
}
{
fma.rn.f16x2 r87, r79, r85, r82;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r91, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r93, {high, high};
}
{
mul.f16x2 r95, r36, r93;
}
{
neg.f16x2 r98, r95;
}
{
fma.rn.f16x2 r100, r33, r91, r98;
}
{
mul.f16x2 r104, r33, r93;
}
{
fma.rn.f16x2 r107, r36, r91, r104;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r111, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r113, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f27;
cvt.rn.f16.f32 high, f28;
mov.b32 r115, {low, high};
}
{
mul.f16x2 r116, r113, r115;
}
{
mul.f16x2 r119, r87, r111;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r122, {high, low};
}
{
fma.rn.f16x2 r124, r116, r122, r119;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r124;
mov.b32 r128, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r124;
mov.b32 r130, {high, high};
}
{
mul.f16x2 r132, r48, r130;
}
{
neg.f16x2 r135, r132;
}
{
fma.rn.f16x2 r137, r45, r128, r135;
}
{
mul.f16x2 r141, r45, r130;
}
{
fma.rn.f16x2 r144, r48, r128, r141;
}
barrier.sync 0;
and.b32 r552, r549, 1008;
add.s32 r553, r551, r552;
st.shared.v4.f32 [r553], {r27, r63, r100, r137};
barrier.sync 0;
mad.lo.s32 r554, r548, -12, r553;
ld.shared.u32 r166, [r554];
ld.shared.u32 r178, [r554+256];
ld.shared.u32 r167, [r554+512];
ld.shared.u32 r179, [r554+768];
barrier.sync 0;
st.shared.v4.f32 [r553], {r30, r70, r107, r144};
barrier.sync 0;
ld.shared.u32 r169, [r554];
ld.shared.u32 r181, [r554+256];
ld.shared.u32 r170, [r554+512];
ld.shared.u32 r182, [r554+768];
{
add.f16x2 r165, r166, r167;
}
{
add.f16x2 r168, r169, r170;
}
{
sub.f16x2 r171, r166, r167;
}
{
sub.f16x2 r174, r169, r170;
}
{
add.f16x2 r177, r178, r179;
}
{
add.f16x2 r180, r181, r182;
}
{
sub.f16x2 r183, r178, r179;
}
{
sub.f16x2 r186, r181, r182;
}
{
neg.f16x2 r189, r183;
}
{
add.f16x2 r191, r165, r177;
}
{
add.f16x2 r194, r168, r180;
}
{
sub.f16x2 r197, r165, r177;
}
{
sub.f16x2 r200, r168, r180;
}
{
add.f16x2 r203, r171, r186;
}
{
add.f16x2 r206, r174, r189;
}
{
sub.f16x2 r209, r171, r186;
}
{
sub.f16x2 r212, r174, r189;
}
and.b32 r555, r547, 60;
bfe.u32 r556, r547, 2, 4;
shl.b32 r557, r547, 2;
and.b32 r558, r557, 12;
add.s32 r559, r551, r558;
cvt.rn.f32.u32 f34, r556;
mul.f32 f35, f34, 0f3DC90FDB;
cos.approx.f32 f11, f35;
sin.approx.f32 f36, f35;
neg.f32 f12, f36;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f11;
cvt.rn.f16.f32 high, f12;
mov.b32 r215, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r218, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r220, {high, high};
}
{
mul.f16x2 r222, r206, r220;
}
{
neg.f16x2 r225, r222;
}
{
fma.rn.f16x2 r227, r203, r218, r225;
}
{
mul.f16x2 r231, r203, r220;
}
{
fma.rn.f16x2 r234, r206, r218, r231;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r238, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r240, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f27;
cvt.rn.f16.f32 high, f28;
mov.b32 r242, {low, high};
}
{
mul.f16x2 r243, r240, r242;
}
{
mul.f16x2 r246, r215, r238;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r249, {high, low};
}
{
fma.rn.f16x2 r251, r243, r249, r246;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r255, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r257, {high, high};
}
{
mul.f16x2 r259, r200, r257;
}
{
neg.f16x2 r262, r259;
}
{
fma.rn.f16x2 r264, r197, r255, r262;
}
{
mul.f16x2 r268, r197, r257;
}
{
fma.rn.f16x2 r271, r200, r255, r268;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r275, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r277, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f27;
cvt.rn.f16.f32 high, f28;
mov.b32 r279, {low, high};
}
{
mul.f16x2 r280, r277, r279;
}
{
mul.f16x2 r283, r251, r275;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r286, {high, low};
}
{
fma.rn.f16x2 r288, r280, r286, r283;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r288;
mov.b32 r292, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r288;
mov.b32 r294, {high, high};
}
{
mul.f16x2 r296, r212, r294;
}
{
neg.f16x2 r299, r296;
}
{
fma.rn.f16x2 r301, r209, r292, r299;
}
{
mul.f16x2 r305, r209, r294;
}
{
fma.rn.f16x2 r308, r212, r292, r305;
}
barrier.sync 0;
and.b32 r560, r549, 960;
add.s32 r561, r559, r560;
st.shared.u32 [r561], r191;
st.shared.u32 [r561+16], r227;
st.shared.u32 [r561+32], r264;
st.shared.u32 [r561+48], r301;
barrier.sync 0;
mad.lo.s32 r562, r555, -12, r561;
ld.shared.u32 r330, [r562];
ld.shared.u32 r342, [r562+256];
ld.shared.u32 r331, [r562+512];
ld.shared.u32 r343, [r562+768];
barrier.sync 0;
st.shared.u32 [r561], r194;
st.shared.u32 [r561+16], r234;
st.shared.u32 [r561+32], r271;
st.shared.u32 [r561+48], r308;
barrier.sync 0;
ld.shared.u32 r333, [r562];
ld.shared.u32 r345, [r562+256];
ld.shared.u32 r334, [r562+512];
ld.shared.u32 r346, [r562+768];
{
add.f16x2 r329, r330, r331;
}
{
add.f16x2 r332, r333, r334;
}
{
sub.f16x2 r335, r330, r331;
}
{
sub.f16x2 r338, r333, r334;
}
{
add.f16x2 r341, r342, r343;
}
{
add.f16x2 r344, r345, r346;
}
{
sub.f16x2 r347, r342, r343;
}
{
sub.f16x2 r350, r345, r346;
}
{
neg.f16x2 r353, r347;
}
{
add.f16x2 r355, r329, r341;
}
{
add.f16x2 r358, r332, r344;
}
{
sub.f16x2 r361, r329, r341;
}
{
sub.f16x2 r364, r332, r344;
}
{
add.f16x2 r367, r335, r350;
}
{
add.f16x2 r370, r338, r353;
}
{
sub.f16x2 r373, r335, r350;
}
{
sub.f16x2 r376, r338, r353;
}
and.b32 r563, r547, 48;
bfe.u32 r564, r547, 4, 2;
and.b32 r565, r557, 60;
add.s32 r566, r551, r565;
cvt.rn.f32.u32 f37, r564;
mul.f32 f38, f37, 0f3EC90FDB;
cos.approx.f32 f21, f38;
sin.approx.f32 f39, f38;
neg.f32 f22, f39;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f21;
cvt.rn.f16.f32 high, f22;
mov.b32 r379, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r379;
mov.b32 r382, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r379;
mov.b32 r384, {high, high};
}
{
mul.f16x2 r386, r370, r384;
}
{
neg.f16x2 r389, r386;
}
{
fma.rn.f16x2 r391, r367, r382, r389;
}
{
mul.f16x2 r395, r367, r384;
}
{
fma.rn.f16x2 r398, r370, r382, r395;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r379;
mov.b32 r402, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r379;
mov.b32 r404, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f27;
cvt.rn.f16.f32 high, f28;
mov.b32 r406, {low, high};
}
{
mul.f16x2 r407, r404, r406;
}
{
mul.f16x2 r410, r379, r402;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r379;
mov.b32 r413, {high, low};
}
{
fma.rn.f16x2 r415, r407, r413, r410;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r415;
mov.b32 r419, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r415;
mov.b32 r421, {high, high};
}
{
mul.f16x2 r423, r364, r421;
}
{
neg.f16x2 r426, r423;
}
{
fma.rn.f16x2 r428, r361, r419, r426;
}
{
mul.f16x2 r432, r361, r421;
}
{
fma.rn.f16x2 r435, r364, r419, r432;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r379;
mov.b32 r439, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r379;
mov.b32 r441, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f27;
cvt.rn.f16.f32 high, f28;
mov.b32 r443, {low, high};
}
{
mul.f16x2 r444, r441, r443;
}
{
mul.f16x2 r447, r415, r439;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r415;
mov.b32 r450, {high, low};
}
{
fma.rn.f16x2 r452, r444, r450, r447;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r452;
mov.b32 r456, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r452;
mov.b32 r458, {high, high};
}
{
mul.f16x2 r460, r376, r458;
}
{
neg.f16x2 r463, r460;
}
{
fma.rn.f16x2 r465, r373, r456, r463;
}
{
mul.f16x2 r469, r373, r458;
}
{
fma.rn.f16x2 r472, r376, r456, r469;
}
barrier.sync 0;
and.b32 r567, r549, 768;
add.s32 r568, r566, r567;
st.shared.u32 [r568], r355;
st.shared.u32 [r568+64], r391;
st.shared.u32 [r568+128], r428;
st.shared.u32 [r568+192], r465;
barrier.sync 0;
mad.lo.s32 r569, r563, -12, r568;
ld.shared.u32 r494, [r569];
ld.shared.u32 r506, [r569+256];
ld.shared.u32 r495, [r569+512];
ld.shared.u32 r507, [r569+768];
barrier.sync 0;
st.shared.u32 [r568], r358;
st.shared.u32 [r568+64], r398;
st.shared.u32 [r568+128], r435;
st.shared.u32 [r568+192], r472;
barrier.sync 0;
ld.shared.u32 r497, [r569];
ld.shared.u32 r509, [r569+256];
ld.shared.u32 r498, [r569+512];
ld.shared.u32 r510, [r569+768];
{
add.f16x2 r493, r494, r495;
}
{
add.f16x2 r496, r497, r498;
}
{
sub.f16x2 r499, r494, r495;
}
{
sub.f16x2 r502, r497, r498;
}
{
add.f16x2 r505, r506, r507;
}
{
add.f16x2 r508, r509, r510;
}
{
sub.f16x2 r511, r506, r507;
}
{
sub.f16x2 r514, r509, r510;
}
{
neg.f16x2 r517, r511;
}
{
add.f16x2 %0, r493, r505;
}
{
add.f16x2 %1, r496, r508;
}
{
sub.f16x2 %4, r493, r505;
}
{
sub.f16x2 %5, r496, r508;
}
{
add.f16x2 %2, r499, r514;
}
{
add.f16x2 %3, r502, r517;
}
{
sub.f16x2 %6, r499, r514;
}
{
sub.f16x2 %7, r502, r517;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<814, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<540>;
.reg .b32 r<3723>;
.reg .b64 rd<3>;
mov.u32 r3647, %tid.y;
shl.b32 r3648, r3647, 10;
mov.u32 r3649, %64;
add.s32 r3650, r3649, r3648;
mov.u32 r3651, %tid.x;
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
mov.f32 f508, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r101, {low, high};
}
mov.f32 f518, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r102, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
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
add.f16x2 r247, %79, %72;
}
{
add.f16x2 r250, %117, %108;
}
{
sub.f16x2 r253, %79, %72;
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
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
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
mov.f32 f238, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r393, {low, high};
}
mov.f32 f288, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r396, {low, high};
}
mov.f32 f254, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f254;
cvt.rn.f16.f32 high, f254;
mov.b32 r397, {low, high};
}
mov.f32 f286, 0fBF6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r398, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r404, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r405, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
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
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r717, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r718, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r721, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
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
add.f16x2 r816, %78, %71;
}
{
sub.f16x2 r819, %109, %100;
}
{
sub.f16x2 r822, %78, %71;
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
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r913, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r914, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r917, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1009, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r1010, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r1011, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r1012, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f254;
cvt.rn.f16.f32 high, f254;
mov.b32 r1013, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r1014, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r1017, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r1018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r1019, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r1020, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r1021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
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
mov.f32 f234, 0f3F7B14BE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f234;
cvt.rn.f16.f32 high, f234;
mov.b32 r1233, {low, high};
}
mov.f32 f292, 0fBE47C5C2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f292;
cvt.rn.f16.f32 high, f292;
mov.b32 r1234, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1235, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r1236, {low, high};
}
mov.f32 f242, 0f3F54DB31;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f242;
cvt.rn.f16.f32 high, f242;
mov.b32 r1237, {low, high};
}
mov.f32 f284, 0fBF0E39DA;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r1238, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r1239, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r1240, {low, high};
}
mov.f32 f250, 0f3F0E39DA;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f250;
cvt.rn.f16.f32 high, f250;
mov.b32 r1241, {low, high};
}
mov.f32 f282, 0fBF54DB31;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r1242, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f254;
cvt.rn.f16.f32 high, f254;
mov.b32 r1243, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r1244, {low, high};
}
mov.f32 f258, 0f3E47C5C2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f258;
cvt.rn.f16.f32 high, f258;
mov.b32 r1245, {low, high};
}
mov.f32 f290, 0fBF7B14BE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f290;
cvt.rn.f16.f32 high, f290;
mov.b32 r1246, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f292;
cvt.rn.f16.f32 high, f292;
mov.b32 r1249, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f290;
cvt.rn.f16.f32 high, f290;
mov.b32 r1250, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r1251, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r1252, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r1253, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r1254, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r1255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r1256, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r1257, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r1258, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r1259, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r1260, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f290;
cvt.rn.f16.f32 high, f290;
mov.b32 r1261, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f292;
cvt.rn.f16.f32 high, f292;
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
and.b32 r3652, r3651, 7;
shl.b32 r3653, r3651, 7;
and.b32 r3654, r3653, -1024;
add.s32 r3655, r3650, r3654;
cvt.rn.f32.u32 f535, r3652;
mul.f32 f536, f535, 0f3CC90FDB;
cos.approx.f32 f357, f536;
sin.approx.f32 f537, f536;
neg.f32 f358, f537;
mov.f32 f539, 0fBF800000;
mov.f32 f538, 0f3F800000;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
cvt.rn.f16.f32 low, f539;
cvt.rn.f16.f32 high, f538;
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
and.b32 r3656, r3653, 896;
add.s32 r3657, r3655, r3656;
st.shared.v4.f32 [r3657], {r1521, r1725, r1762, r1799};
st.shared.v4.f32 [r3657+16], {r1836, r1873, r1910, r1947};
st.shared.v4.f32 [r3657+32], {r1984, r2021, r2058, r2095};
st.shared.v4.f32 [r3657+48], {r2132, r2169, r2206, r2243};
st.shared.v4.f32 [r3657+64], {r2280, r2317, r2354, r2391};
st.shared.v4.f32 [r3657+80], {r2428, r2465, r2502, r2539};
st.shared.v4.f32 [r3657+96], {r2576, r2613, r2650, r2687};
st.shared.v4.f32 [r3657+112], {r2724, r2761, r2798, r2835};
barrier.sync 0;
mad.lo.s32 r3658, r3652, -124, r3657;
ld.shared.u32 r2864, [r3658];
ld.shared.u32 r3060, [r3658+32];
ld.shared.u32 r3256, [r3658+64];
ld.shared.u32 r3452, [r3658+96];
ld.shared.u32 r2914, [r3658+128];
ld.shared.u32 r3110, [r3658+160];
ld.shared.u32 r3306, [r3658+192];
ld.shared.u32 r3502, [r3658+224];
ld.shared.u32 r2876, [r3658+256];
ld.shared.u32 r3072, [r3658+288];
ld.shared.u32 r3268, [r3658+320];
ld.shared.u32 r3464, [r3658+352];
ld.shared.u32 r2926, [r3658+384];
ld.shared.u32 r3122, [r3658+416];
ld.shared.u32 r3318, [r3658+448];
ld.shared.u32 r3514, [r3658+480];
ld.shared.u32 r2865, [r3658+512];
ld.shared.u32 r3061, [r3658+544];
ld.shared.u32 r3257, [r3658+576];
ld.shared.u32 r3453, [r3658+608];
ld.shared.u32 r2915, [r3658+640];
ld.shared.u32 r3111, [r3658+672];
ld.shared.u32 r3307, [r3658+704];
ld.shared.u32 r3503, [r3658+736];
ld.shared.u32 r2877, [r3658+768];
ld.shared.u32 r3073, [r3658+800];
ld.shared.u32 r3269, [r3658+832];
ld.shared.u32 r3465, [r3658+864];
ld.shared.u32 r2927, [r3658+896];
ld.shared.u32 r3123, [r3658+928];
ld.shared.u32 r3319, [r3658+960];
ld.shared.u32 r3515, [r3658+992];
barrier.sync 0;
st.shared.v4.f32 [r3657], {r1524, r1732, r1769, r1806};
st.shared.v4.f32 [r3657+16], {r1843, r1880, r1917, r1954};
st.shared.v4.f32 [r3657+32], {r1991, r2028, r2065, r2102};
st.shared.v4.f32 [r3657+48], {r2139, r2176, r2213, r2250};
st.shared.v4.f32 [r3657+64], {r2287, r2324, r2361, r2398};
st.shared.v4.f32 [r3657+80], {r2435, r2472, r2509, r2546};
st.shared.v4.f32 [r3657+96], {r2583, r2620, r2657, r2694};
st.shared.v4.f32 [r3657+112], {r2731, r2768, r2805, r2842};
barrier.sync 0;
ld.shared.u32 r2867, [r3658];
ld.shared.u32 r3063, [r3658+32];
ld.shared.u32 r3259, [r3658+64];
ld.shared.u32 r3455, [r3658+96];
ld.shared.u32 r2917, [r3658+128];
ld.shared.u32 r3113, [r3658+160];
ld.shared.u32 r3309, [r3658+192];
ld.shared.u32 r3505, [r3658+224];
ld.shared.u32 r2879, [r3658+256];
ld.shared.u32 r3075, [r3658+288];
ld.shared.u32 r3271, [r3658+320];
ld.shared.u32 r3467, [r3658+352];
ld.shared.u32 r2929, [r3658+384];
ld.shared.u32 r3125, [r3658+416];
ld.shared.u32 r3321, [r3658+448];
ld.shared.u32 r3517, [r3658+480];
ld.shared.u32 r2868, [r3658+512];
ld.shared.u32 r3064, [r3658+544];
ld.shared.u32 r3260, [r3658+576];
ld.shared.u32 r3456, [r3658+608];
ld.shared.u32 r2918, [r3658+640];
ld.shared.u32 r3114, [r3658+672];
ld.shared.u32 r3310, [r3658+704];
ld.shared.u32 r3506, [r3658+736];
ld.shared.u32 r2880, [r3658+768];
ld.shared.u32 r3076, [r3658+800];
ld.shared.u32 r3272, [r3658+832];
ld.shared.u32 r3468, [r3658+864];
ld.shared.u32 r2930, [r3658+896];
ld.shared.u32 r3126, [r3658+928];
ld.shared.u32 r3322, [r3658+960];
ld.shared.u32 r3518, [r3658+992];
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
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r2963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r2964, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r2967, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
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
add.f16x2 %0, r2889, r2939;
}
{
add.f16x2 %1, r2892, r2942;
}
{
sub.f16x2 %32, r2889, r2939;
}
{
sub.f16x2 %33, r2892, r2942;
}
{
add.f16x2 %8, r2901, r2983;
}
{
add.f16x2 %9, r2904, r2989;
}
{
sub.f16x2 %40, r2901, r2983;
}
{
sub.f16x2 %41, r2904, r2989;
}
{
add.f16x2 %16, r2895, r2948;
}
{
add.f16x2 %17, r2898, r2993;
}
{
sub.f16x2 %48, r2895, r2948;
}
{
sub.f16x2 %49, r2898, r2993;
}
{
add.f16x2 %24, r2907, r3001;
}
{
add.f16x2 %25, r2910, r3007;
}
{
sub.f16x2 %56, r2907, r3001;
}
{
sub.f16x2 %57, r2910, r3007;
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
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r3159, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r3160, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r3163, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
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
add.f16x2 %2, r3085, r3135;
}
{
add.f16x2 %3, r3088, r3138;
}
{
sub.f16x2 %34, r3085, r3135;
}
{
sub.f16x2 %35, r3088, r3138;
}
{
add.f16x2 %10, r3097, r3179;
}
{
add.f16x2 %11, r3100, r3185;
}
{
sub.f16x2 %42, r3097, r3179;
}
{
sub.f16x2 %43, r3100, r3185;
}
{
add.f16x2 %18, r3091, r3144;
}
{
add.f16x2 %19, r3094, r3189;
}
{
sub.f16x2 %50, r3091, r3144;
}
{
sub.f16x2 %51, r3094, r3189;
}
{
add.f16x2 %26, r3103, r3197;
}
{
add.f16x2 %27, r3106, r3203;
}
{
sub.f16x2 %58, r3103, r3197;
}
{
sub.f16x2 %59, r3106, r3203;
}
{
add.f16x2 r3255, r3256, r3257;
}
{
add.f16x2 r3258, r3259, r3260;
}
{
sub.f16x2 r3261, r3256, r3257;
}
{
sub.f16x2 r3264, r3259, r3260;
}
{
add.f16x2 r3267, r3268, r3269;
}
{
add.f16x2 r3270, r3271, r3272;
}
{
sub.f16x2 r3273, r3268, r3269;
}
{
sub.f16x2 r3276, r3271, r3272;
}
{
neg.f16x2 r3279, r3273;
}
{
add.f16x2 r3281, r3255, r3267;
}
{
add.f16x2 r3284, r3258, r3270;
}
{
sub.f16x2 r3287, r3255, r3267;
}
{
sub.f16x2 r3290, r3258, r3270;
}
{
add.f16x2 r3293, r3261, r3276;
}
{
add.f16x2 r3296, r3264, r3279;
}
{
sub.f16x2 r3299, r3261, r3276;
}
{
sub.f16x2 r3302, r3264, r3279;
}
{
add.f16x2 r3305, r3306, r3307;
}
{
add.f16x2 r3308, r3309, r3310;
}
{
sub.f16x2 r3311, r3306, r3307;
}
{
sub.f16x2 r3314, r3309, r3310;
}
{
add.f16x2 r3317, r3318, r3319;
}
{
add.f16x2 r3320, r3321, r3322;
}
{
sub.f16x2 r3323, r3318, r3319;
}
{
sub.f16x2 r3326, r3321, r3322;
}
{
neg.f16x2 r3329, r3323;
}
{
add.f16x2 r3331, r3305, r3317;
}
{
add.f16x2 r3334, r3308, r3320;
}
{
sub.f16x2 r3337, r3305, r3317;
}
{
sub.f16x2 r3340, r3308, r3320;
}
{
add.f16x2 r3343, r3311, r3326;
}
{
add.f16x2 r3346, r3314, r3329;
}
{
sub.f16x2 r3349, r3311, r3326;
}
{
sub.f16x2 r3352, r3314, r3329;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r3355, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r3356, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r3359, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r3360, {low, high};
}
{
mul.f16x2 r3369, r3343, r3355;
}
{
mul.f16x2 r3372, r3346, r3356;
}
{
sub.f16x2 r3375, r3369, r3372;
}
{
mul.f16x2 r3378, r3343, r3356;
}
{
fma.rn.f16x2 r3381, r3346, r3355, r3378;
}
{
neg.f16x2 r3385, r3337;
}
{
mul.f16x2 r3387, r3349, r3359;
}
{
mul.f16x2 r3390, r3352, r3360;
}
{
sub.f16x2 r3393, r3387, r3390;
}
{
mul.f16x2 r3396, r3349, r3360;
}
{
fma.rn.f16x2 r3399, r3352, r3359, r3396;
}
{
add.f16x2 %4, r3281, r3331;
}
{
add.f16x2 %5, r3284, r3334;
}
{
sub.f16x2 %36, r3281, r3331;
}
{
sub.f16x2 %37, r3284, r3334;
}
{
add.f16x2 %12, r3293, r3375;
}
{
add.f16x2 %13, r3296, r3381;
}
{
sub.f16x2 %44, r3293, r3375;
}
{
sub.f16x2 %45, r3296, r3381;
}
{
add.f16x2 %20, r3287, r3340;
}
{
add.f16x2 %21, r3290, r3385;
}
{
sub.f16x2 %52, r3287, r3340;
}
{
sub.f16x2 %53, r3290, r3385;
}
{
add.f16x2 %28, r3299, r3393;
}
{
add.f16x2 %29, r3302, r3399;
}
{
sub.f16x2 %60, r3299, r3393;
}
{
sub.f16x2 %61, r3302, r3399;
}
{
add.f16x2 r3451, r3452, r3453;
}
{
add.f16x2 r3454, r3455, r3456;
}
{
sub.f16x2 r3457, r3452, r3453;
}
{
sub.f16x2 r3460, r3455, r3456;
}
{
add.f16x2 r3463, r3464, r3465;
}
{
add.f16x2 r3466, r3467, r3468;
}
{
sub.f16x2 r3469, r3464, r3465;
}
{
sub.f16x2 r3472, r3467, r3468;
}
{
neg.f16x2 r3475, r3469;
}
{
add.f16x2 r3477, r3451, r3463;
}
{
add.f16x2 r3480, r3454, r3466;
}
{
sub.f16x2 r3483, r3451, r3463;
}
{
sub.f16x2 r3486, r3454, r3466;
}
{
add.f16x2 r3489, r3457, r3472;
}
{
add.f16x2 r3492, r3460, r3475;
}
{
sub.f16x2 r3495, r3457, r3472;
}
{
sub.f16x2 r3498, r3460, r3475;
}
{
add.f16x2 r3501, r3502, r3503;
}
{
add.f16x2 r3504, r3505, r3506;
}
{
sub.f16x2 r3507, r3502, r3503;
}
{
sub.f16x2 r3510, r3505, r3506;
}
{
add.f16x2 r3513, r3514, r3515;
}
{
add.f16x2 r3516, r3517, r3518;
}
{
sub.f16x2 r3519, r3514, r3515;
}
{
sub.f16x2 r3522, r3517, r3518;
}
{
neg.f16x2 r3525, r3519;
}
{
add.f16x2 r3527, r3501, r3513;
}
{
add.f16x2 r3530, r3504, r3516;
}
{
sub.f16x2 r3533, r3501, r3513;
}
{
sub.f16x2 r3536, r3504, r3516;
}
{
add.f16x2 r3539, r3507, r3522;
}
{
add.f16x2 r3542, r3510, r3525;
}
{
sub.f16x2 r3545, r3507, r3522;
}
{
sub.f16x2 r3548, r3510, r3525;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f508;
cvt.rn.f16.f32 high, f508;
mov.b32 r3551, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r3552, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r3555, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f518;
cvt.rn.f16.f32 high, f518;
mov.b32 r3556, {low, high};
}
{
mul.f16x2 r3565, r3539, r3551;
}
{
mul.f16x2 r3568, r3542, r3552;
}
{
sub.f16x2 r3571, r3565, r3568;
}
{
mul.f16x2 r3574, r3539, r3552;
}
{
fma.rn.f16x2 r3577, r3542, r3551, r3574;
}
{
neg.f16x2 r3581, r3533;
}
{
mul.f16x2 r3583, r3545, r3555;
}
{
mul.f16x2 r3586, r3548, r3556;
}
{
sub.f16x2 r3589, r3583, r3586;
}
{
mul.f16x2 r3592, r3545, r3556;
}
{
fma.rn.f16x2 r3595, r3548, r3555, r3592;
}
{
add.f16x2 %6, r3477, r3527;
}
{
add.f16x2 %7, r3480, r3530;
}
{
sub.f16x2 %38, r3477, r3527;
}
{
sub.f16x2 %39, r3480, r3530;
}
{
add.f16x2 %14, r3489, r3571;
}
{
add.f16x2 %15, r3492, r3577;
}
{
sub.f16x2 %46, r3489, r3571;
}
{
sub.f16x2 %47, r3492, r3577;
}
{
add.f16x2 %22, r3483, r3536;
}
{
add.f16x2 %23, r3486, r3581;
}
{
sub.f16x2 %54, r3483, r3536;
}
{
sub.f16x2 %55, r3486, r3581;
}
{
add.f16x2 %30, r3495, r3589;
}
{
add.f16x2 %31, r3498, r3595;
}
{
sub.f16x2 %62, r3495, r3589;
}
{
sub.f16x2 %63, r3498, r3595;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)), "=r"(__HALF2_TO_UI(rmem[27].x)), "=r"(__HALF2_TO_UI(rmem[27].y)), "=r"(__HALF2_TO_UI(rmem[28].x)), "=r"(__HALF2_TO_UI(rmem[28].y)), "=r"(__HALF2_TO_UI(rmem[29].x)), "=r"(__HALF2_TO_UI(rmem[29].y)), "=r"(__HALF2_TO_UI(rmem[30].x)), "=r"(__HALF2_TO_UI(rmem[30].y)), "=r"(__HALF2_TO_UI(rmem[31].x)), "=r"(__HALF2_TO_UI(rmem[31].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[31].y)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[28].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[27].y)), "r"(__HALF2_TO_UI(rmem[30].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[29].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[31].x)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[28].x)), "r"(__HALF2_TO_UI(rmem[27].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[30].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[29].x)), "r"(__HALF2_TO_UI(rmem[24].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<815, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<64>;
.reg .b32 r<433>;
.reg .b64 rd<2>;
mov.u32 r377, %tid.y;
shl.b32 r378, r377, 11;
mov.u32 r379, %4;
add.s32 r380, r379, r378;
mov.u32 r381, %tid.x;
{
add.f16x2 r1, %5, %7;
}
{
add.f16x2 r4, %6, %8;
}
{
sub.f16x2 r7, %5, %7;
}
{
sub.f16x2 r10, %6, %8;
}
and.b32 r382, r381, 127;
shl.b32 r383, r381, 4;
and.b32 r384, r383, -2048;
add.s32 r385, r380, r384;
cvt.rn.f32.u32 f43, r382;
mul.f32 f44, f43, 0f3CC90FDB;
cos.approx.f32 f1, f44;
sin.approx.f32 f45, f44;
neg.f32 f2, f45;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1;
cvt.rn.f16.f32 high, f2;
mov.b32 r13, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r13;
mov.b32 r16, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r13;
mov.b32 r18, {high, high};
}
{
mul.f16x2 r20, r10, r18;
}
{
neg.f16x2 r23, r20;
}
{
fma.rn.f16x2 r25, r7, r16, r23;
}
{
mul.f16x2 r29, r7, r18;
}
{
fma.rn.f16x2 r32, r10, r16, r29;
}
barrier.sync 0;
and.b32 r386, r383, 2032;
add.s32 r387, r385, r386;
st.shared.v2.f32 [r387], {r1, r4};
st.shared.v2.f32 [r387+8], {r25, r32};
barrier.sync 0;
shl.b32 r388, r381, 3;
and.b32 r389, r388, 1016;
sub.s32 r390, r387, r389;
ld.shared.u32 r54, [r390];
ld.shared.u32 r57, [r390+4];
ld.shared.u32 r55, [r390+1024];
ld.shared.u32 r58, [r390+1028];
{
add.f16x2 r53, r54, r55;
}
{
add.f16x2 r56, r57, r58;
}
{
sub.f16x2 r59, r54, r55;
}
{
sub.f16x2 r62, r57, r58;
}
bfe.u32 r391, r381, 1, 6;
cvt.rn.f32.u32 f46, r391;
mul.f32 f47, f46, 0f3D490FDB;
cos.approx.f32 f7, f47;
sin.approx.f32 f48, f47;
neg.f32 f8, f48;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f7;
cvt.rn.f16.f32 high, f8;
mov.b32 r65, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r65;
mov.b32 r68, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r65;
mov.b32 r70, {high, high};
}
{
mul.f16x2 r72, r62, r70;
}
{
neg.f16x2 r75, r72;
}
{
fma.rn.f16x2 r77, r59, r68, r75;
}
{
mul.f16x2 r81, r59, r70;
}
{
fma.rn.f16x2 r84, r62, r68, r81;
}
and.b32 r392, r388, 8;
add.s32 r393, r385, r392;
barrier.sync 0;
and.b32 r394, r383, 2016;
add.s32 r395, r393, r394;
st.shared.u32 [r395], r53;
st.shared.u32 [r395+4], r56;
st.shared.u32 [r395+16], r77;
st.shared.u32 [r395+20], r84;
barrier.sync 0;
and.b32 r396, r388, 1008;
sub.s32 r397, r395, r396;
ld.shared.u32 r106, [r397];
ld.shared.u32 r109, [r397+4];
ld.shared.u32 r107, [r397+1024];
ld.shared.u32 r110, [r397+1028];
{
add.f16x2 r105, r106, r107;
}
{
add.f16x2 r108, r109, r110;
}
{
sub.f16x2 r111, r106, r107;
}
{
sub.f16x2 r114, r109, r110;
}
bfe.u32 r398, r381, 2, 5;
cvt.rn.f32.u32 f49, r398;
mul.f32 f50, f49, 0f3DC90FDB;
cos.approx.f32 f13, f50;
sin.approx.f32 f51, f50;
neg.f32 f14, f51;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f13;
cvt.rn.f16.f32 high, f14;
mov.b32 r117, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r117;
mov.b32 r120, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r117;
mov.b32 r122, {high, high};
}
{
mul.f16x2 r124, r114, r122;
}
{
neg.f16x2 r127, r124;
}
{
fma.rn.f16x2 r129, r111, r120, r127;
}
{
mul.f16x2 r133, r111, r122;
}
{
fma.rn.f16x2 r136, r114, r120, r133;
}
and.b32 r399, r388, 24;
add.s32 r400, r385, r399;
barrier.sync 0;
and.b32 r401, r383, 1984;
add.s32 r402, r400, r401;
st.shared.u32 [r402], r105;
st.shared.u32 [r402+4], r108;
st.shared.u32 [r402+32], r129;
st.shared.u32 [r402+36], r136;
barrier.sync 0;
and.b32 r403, r388, 992;
sub.s32 r404, r402, r403;
ld.shared.u32 r158, [r404];
ld.shared.u32 r161, [r404+4];
ld.shared.u32 r159, [r404+1024];
ld.shared.u32 r162, [r404+1028];
{
add.f16x2 r157, r158, r159;
}
{
add.f16x2 r160, r161, r162;
}
{
sub.f16x2 r163, r158, r159;
}
{
sub.f16x2 r166, r161, r162;
}
bfe.u32 r405, r381, 3, 4;
cvt.rn.f32.u32 f52, r405;
mul.f32 f53, f52, 0f3E490FDB;
cos.approx.f32 f19, f53;
sin.approx.f32 f54, f53;
neg.f32 f20, f54;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f19;
cvt.rn.f16.f32 high, f20;
mov.b32 r169, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r169;
mov.b32 r172, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r169;
mov.b32 r174, {high, high};
}
{
mul.f16x2 r176, r166, r174;
}
{
neg.f16x2 r179, r176;
}
{
fma.rn.f16x2 r181, r163, r172, r179;
}
{
mul.f16x2 r185, r163, r174;
}
{
fma.rn.f16x2 r188, r166, r172, r185;
}
and.b32 r406, r388, 56;
add.s32 r407, r385, r406;
barrier.sync 0;
and.b32 r408, r383, 1920;
add.s32 r409, r407, r408;
st.shared.u32 [r409], r157;
st.shared.u32 [r409+4], r160;
st.shared.u32 [r409+64], r181;
st.shared.u32 [r409+68], r188;
barrier.sync 0;
and.b32 r410, r388, 960;
sub.s32 r411, r409, r410;
ld.shared.u32 r210, [r411];
ld.shared.u32 r213, [r411+4];
ld.shared.u32 r211, [r411+1024];
ld.shared.u32 r214, [r411+1028];
{
add.f16x2 r209, r210, r211;
}
{
add.f16x2 r212, r213, r214;
}
{
sub.f16x2 r215, r210, r211;
}
{
sub.f16x2 r218, r213, r214;
}
bfe.u32 r412, r381, 4, 3;
cvt.rn.f32.u32 f55, r412;
mul.f32 f56, f55, 0f3EC90FDB;
cos.approx.f32 f25, f56;
sin.approx.f32 f57, f56;
neg.f32 f26, f57;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f25;
cvt.rn.f16.f32 high, f26;
mov.b32 r221, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r221;
mov.b32 r224, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r221;
mov.b32 r226, {high, high};
}
{
mul.f16x2 r228, r218, r226;
}
{
neg.f16x2 r231, r228;
}
{
fma.rn.f16x2 r233, r215, r224, r231;
}
{
mul.f16x2 r237, r215, r226;
}
{
fma.rn.f16x2 r240, r218, r224, r237;
}
and.b32 r413, r388, 120;
add.s32 r414, r385, r413;
barrier.sync 0;
and.b32 r415, r383, 1792;
add.s32 r416, r414, r415;
st.shared.u32 [r416], r209;
st.shared.u32 [r416+4], r212;
st.shared.u32 [r416+128], r233;
st.shared.u32 [r416+132], r240;
barrier.sync 0;
and.b32 r417, r388, 896;
sub.s32 r418, r416, r417;
ld.shared.u32 r262, [r418];
ld.shared.u32 r265, [r418+4];
ld.shared.u32 r263, [r418+1024];
ld.shared.u32 r266, [r418+1028];
{
add.f16x2 r261, r262, r263;
}
{
add.f16x2 r264, r265, r266;
}
{
sub.f16x2 r267, r262, r263;
}
{
sub.f16x2 r270, r265, r266;
}
bfe.u32 r419, r381, 5, 2;
cvt.rn.f32.u32 f58, r419;
mul.f32 f59, f58, 0f3F490FDB;
cos.approx.f32 f31, f59;
sin.approx.f32 f60, f59;
neg.f32 f32, f60;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f31;
cvt.rn.f16.f32 high, f32;
mov.b32 r273, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r273;
mov.b32 r276, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r273;
mov.b32 r278, {high, high};
}
{
mul.f16x2 r280, r270, r278;
}
{
neg.f16x2 r283, r280;
}
{
fma.rn.f16x2 r285, r267, r276, r283;
}
{
mul.f16x2 r289, r267, r278;
}
{
fma.rn.f16x2 r292, r270, r276, r289;
}
and.b32 r420, r388, 248;
add.s32 r421, r385, r420;
barrier.sync 0;
and.b32 r422, r383, 1536;
add.s32 r423, r421, r422;
st.shared.u32 [r423], r261;
st.shared.u32 [r423+4], r264;
st.shared.u32 [r423+256], r285;
st.shared.u32 [r423+260], r292;
barrier.sync 0;
and.b32 r424, r388, 768;
sub.s32 r425, r423, r424;
ld.shared.u32 r314, [r425];
ld.shared.u32 r317, [r425+4];
ld.shared.u32 r315, [r425+1024];
ld.shared.u32 r318, [r425+1028];
{
add.f16x2 r313, r314, r315;
}
{
add.f16x2 r316, r317, r318;
}
{
sub.f16x2 r319, r314, r315;
}
{
sub.f16x2 r322, r317, r318;
}
bfe.u32 r426, r381, 6, 1;
cvt.rn.f32.u32 f61, r426;
mul.f32 f62, f61, 0f3FC90FDB;
cos.approx.f32 f37, f62;
sin.approx.f32 f63, f62;
neg.f32 f38, f63;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
mov.b32 r325, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r325;
mov.b32 r328, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r325;
mov.b32 r330, {high, high};
}
{
mul.f16x2 r332, r322, r330;
}
{
neg.f16x2 r335, r332;
}
{
fma.rn.f16x2 r337, r319, r328, r335;
}
{
mul.f16x2 r341, r319, r330;
}
{
fma.rn.f16x2 r344, r322, r328, r341;
}
and.b32 r427, r388, 504;
add.s32 r428, r385, r427;
barrier.sync 0;
and.b32 r429, r383, 1024;
add.s32 r430, r428, r429;
st.shared.u32 [r430], r313;
st.shared.u32 [r430+4], r316;
st.shared.u32 [r430+512], r337;
st.shared.u32 [r430+516], r344;
barrier.sync 0;
and.b32 r431, r388, 512;
sub.s32 r432, r430, r431;
ld.shared.u32 r366, [r432];
ld.shared.u32 r369, [r432+4];
ld.shared.u32 r367, [r432+1024];
ld.shared.u32 r370, [r432+1028];
{
add.f16x2 %0, r366, r367;
}
{
add.f16x2 %1, r369, r370;
}
{
sub.f16x2 %2, r366, r367;
}
{
sub.f16x2 %3, r369, r370;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<816, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<64>;
.reg .b32 r<433>;
.reg .b64 rd<2>;
mov.u32 r377, %tid.y;
shl.b32 r378, r377, 10;
mov.u32 r379, %4;
add.s32 r380, r379, r378;
mov.u32 r381, %tid.x;
{
add.f16x2 r1, %5, %7;
}
{
add.f16x2 r4, %6, %8;
}
{
sub.f16x2 r7, %5, %7;
}
{
sub.f16x2 r10, %6, %8;
}
and.b32 r382, r381, 127;
shl.b32 r383, r381, 3;
and.b32 r384, r383, -1024;
add.s32 r385, r380, r384;
cvt.rn.f32.u32 f43, r382;
mul.f32 f44, f43, 0f3CC90FDB;
cos.approx.f32 f1, f44;
sin.approx.f32 f45, f44;
neg.f32 f2, f45;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1;
cvt.rn.f16.f32 high, f2;
mov.b32 r13, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r13;
mov.b32 r16, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r13;
mov.b32 r18, {high, high};
}
{
mul.f16x2 r20, r10, r18;
}
{
neg.f16x2 r23, r20;
}
{
fma.rn.f16x2 r25, r7, r16, r23;
}
{
mul.f16x2 r29, r7, r18;
}
{
fma.rn.f16x2 r32, r10, r16, r29;
}
barrier.sync 0;
and.b32 r386, r383, 1016;
add.s32 r387, r385, r386;
st.shared.v2.f32 [r387], {r1, r25};
barrier.sync 0;
shl.b32 r388, r381, 2;
and.b32 r389, r388, 508;
sub.s32 r390, r387, r389;
ld.shared.u32 r54, [r390];
ld.shared.u32 r55, [r390+512];
barrier.sync 0;
st.shared.v2.f32 [r387], {r4, r32};
barrier.sync 0;
ld.shared.u32 r57, [r390];
ld.shared.u32 r58, [r390+512];
{
add.f16x2 r53, r54, r55;
}
{
add.f16x2 r56, r57, r58;
}
{
sub.f16x2 r59, r54, r55;
}
{
sub.f16x2 r62, r57, r58;
}
bfe.u32 r391, r381, 1, 6;
and.b32 r392, r388, 4;
add.s32 r393, r385, r392;
cvt.rn.f32.u32 f46, r391;
mul.f32 f47, f46, 0f3D490FDB;
cos.approx.f32 f7, f47;
sin.approx.f32 f48, f47;
neg.f32 f8, f48;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f7;
cvt.rn.f16.f32 high, f8;
mov.b32 r65, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r65;
mov.b32 r68, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r65;
mov.b32 r70, {high, high};
}
{
mul.f16x2 r72, r62, r70;
}
{
neg.f16x2 r75, r72;
}
{
fma.rn.f16x2 r77, r59, r68, r75;
}
{
mul.f16x2 r81, r59, r70;
}
{
fma.rn.f16x2 r84, r62, r68, r81;
}
barrier.sync 0;
and.b32 r394, r383, 1008;
add.s32 r395, r393, r394;
st.shared.u32 [r395], r53;
st.shared.u32 [r395+8], r77;
barrier.sync 0;
and.b32 r396, r388, 504;
sub.s32 r397, r395, r396;
ld.shared.u32 r106, [r397];
ld.shared.u32 r107, [r397+512];
barrier.sync 0;
st.shared.u32 [r395], r56;
st.shared.u32 [r395+8], r84;
barrier.sync 0;
ld.shared.u32 r109, [r397];
ld.shared.u32 r110, [r397+512];
{
add.f16x2 r105, r106, r107;
}
{
add.f16x2 r108, r109, r110;
}
{
sub.f16x2 r111, r106, r107;
}
{
sub.f16x2 r114, r109, r110;
}
bfe.u32 r398, r381, 2, 5;
and.b32 r399, r388, 12;
add.s32 r400, r385, r399;
cvt.rn.f32.u32 f49, r398;
mul.f32 f50, f49, 0f3DC90FDB;
cos.approx.f32 f13, f50;
sin.approx.f32 f51, f50;
neg.f32 f14, f51;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f13;
cvt.rn.f16.f32 high, f14;
mov.b32 r117, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r117;
mov.b32 r120, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r117;
mov.b32 r122, {high, high};
}
{
mul.f16x2 r124, r114, r122;
}
{
neg.f16x2 r127, r124;
}
{
fma.rn.f16x2 r129, r111, r120, r127;
}
{
mul.f16x2 r133, r111, r122;
}
{
fma.rn.f16x2 r136, r114, r120, r133;
}
barrier.sync 0;
and.b32 r401, r383, 992;
add.s32 r402, r400, r401;
st.shared.u32 [r402], r105;
st.shared.u32 [r402+16], r129;
barrier.sync 0;
and.b32 r403, r388, 496;
sub.s32 r404, r402, r403;
ld.shared.u32 r158, [r404];
ld.shared.u32 r159, [r404+512];
barrier.sync 0;
st.shared.u32 [r402], r108;
st.shared.u32 [r402+16], r136;
barrier.sync 0;
ld.shared.u32 r161, [r404];
ld.shared.u32 r162, [r404+512];
{
add.f16x2 r157, r158, r159;
}
{
add.f16x2 r160, r161, r162;
}
{
sub.f16x2 r163, r158, r159;
}
{
sub.f16x2 r166, r161, r162;
}
bfe.u32 r405, r381, 3, 4;
and.b32 r406, r388, 28;
add.s32 r407, r385, r406;
cvt.rn.f32.u32 f52, r405;
mul.f32 f53, f52, 0f3E490FDB;
cos.approx.f32 f19, f53;
sin.approx.f32 f54, f53;
neg.f32 f20, f54;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f19;
cvt.rn.f16.f32 high, f20;
mov.b32 r169, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r169;
mov.b32 r172, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r169;
mov.b32 r174, {high, high};
}
{
mul.f16x2 r176, r166, r174;
}
{
neg.f16x2 r179, r176;
}
{
fma.rn.f16x2 r181, r163, r172, r179;
}
{
mul.f16x2 r185, r163, r174;
}
{
fma.rn.f16x2 r188, r166, r172, r185;
}
barrier.sync 0;
and.b32 r408, r383, 960;
add.s32 r409, r407, r408;
st.shared.u32 [r409], r157;
st.shared.u32 [r409+32], r181;
barrier.sync 0;
and.b32 r410, r388, 480;
sub.s32 r411, r409, r410;
ld.shared.u32 r210, [r411];
ld.shared.u32 r211, [r411+512];
barrier.sync 0;
st.shared.u32 [r409], r160;
st.shared.u32 [r409+32], r188;
barrier.sync 0;
ld.shared.u32 r213, [r411];
ld.shared.u32 r214, [r411+512];
{
add.f16x2 r209, r210, r211;
}
{
add.f16x2 r212, r213, r214;
}
{
sub.f16x2 r215, r210, r211;
}
{
sub.f16x2 r218, r213, r214;
}
bfe.u32 r412, r381, 4, 3;
and.b32 r413, r388, 60;
add.s32 r414, r385, r413;
cvt.rn.f32.u32 f55, r412;
mul.f32 f56, f55, 0f3EC90FDB;
cos.approx.f32 f25, f56;
sin.approx.f32 f57, f56;
neg.f32 f26, f57;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f25;
cvt.rn.f16.f32 high, f26;
mov.b32 r221, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r221;
mov.b32 r224, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r221;
mov.b32 r226, {high, high};
}
{
mul.f16x2 r228, r218, r226;
}
{
neg.f16x2 r231, r228;
}
{
fma.rn.f16x2 r233, r215, r224, r231;
}
{
mul.f16x2 r237, r215, r226;
}
{
fma.rn.f16x2 r240, r218, r224, r237;
}
barrier.sync 0;
and.b32 r415, r383, 896;
add.s32 r416, r414, r415;
st.shared.u32 [r416], r209;
st.shared.u32 [r416+64], r233;
barrier.sync 0;
and.b32 r417, r388, 448;
sub.s32 r418, r416, r417;
ld.shared.u32 r262, [r418];
ld.shared.u32 r263, [r418+512];
barrier.sync 0;
st.shared.u32 [r416], r212;
st.shared.u32 [r416+64], r240;
barrier.sync 0;
ld.shared.u32 r265, [r418];
ld.shared.u32 r266, [r418+512];
{
add.f16x2 r261, r262, r263;
}
{
add.f16x2 r264, r265, r266;
}
{
sub.f16x2 r267, r262, r263;
}
{
sub.f16x2 r270, r265, r266;
}
bfe.u32 r419, r381, 5, 2;
and.b32 r420, r388, 124;
add.s32 r421, r385, r420;
cvt.rn.f32.u32 f58, r419;
mul.f32 f59, f58, 0f3F490FDB;
cos.approx.f32 f31, f59;
sin.approx.f32 f60, f59;
neg.f32 f32, f60;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f31;
cvt.rn.f16.f32 high, f32;
mov.b32 r273, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r273;
mov.b32 r276, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r273;
mov.b32 r278, {high, high};
}
{
mul.f16x2 r280, r270, r278;
}
{
neg.f16x2 r283, r280;
}
{
fma.rn.f16x2 r285, r267, r276, r283;
}
{
mul.f16x2 r289, r267, r278;
}
{
fma.rn.f16x2 r292, r270, r276, r289;
}
barrier.sync 0;
and.b32 r422, r383, 768;
add.s32 r423, r421, r422;
st.shared.u32 [r423], r261;
st.shared.u32 [r423+128], r285;
barrier.sync 0;
and.b32 r424, r388, 384;
sub.s32 r425, r423, r424;
ld.shared.u32 r314, [r425];
ld.shared.u32 r315, [r425+512];
barrier.sync 0;
st.shared.u32 [r423], r264;
st.shared.u32 [r423+128], r292;
barrier.sync 0;
ld.shared.u32 r317, [r425];
ld.shared.u32 r318, [r425+512];
{
add.f16x2 r313, r314, r315;
}
{
add.f16x2 r316, r317, r318;
}
{
sub.f16x2 r319, r314, r315;
}
{
sub.f16x2 r322, r317, r318;
}
bfe.u32 r426, r381, 6, 1;
and.b32 r427, r388, 252;
add.s32 r428, r385, r427;
cvt.rn.f32.u32 f61, r426;
mul.f32 f62, f61, 0f3FC90FDB;
cos.approx.f32 f37, f62;
sin.approx.f32 f63, f62;
neg.f32 f38, f63;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
mov.b32 r325, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r325;
mov.b32 r328, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r325;
mov.b32 r330, {high, high};
}
{
mul.f16x2 r332, r322, r330;
}
{
neg.f16x2 r335, r332;
}
{
fma.rn.f16x2 r337, r319, r328, r335;
}
{
mul.f16x2 r341, r319, r330;
}
{
fma.rn.f16x2 r344, r322, r328, r341;
}
barrier.sync 0;
and.b32 r429, r383, 512;
add.s32 r430, r428, r429;
st.shared.u32 [r430], r313;
st.shared.u32 [r430+256], r337;
barrier.sync 0;
and.b32 r431, r388, 256;
sub.s32 r432, r430, r431;
ld.shared.u32 r366, [r432];
ld.shared.u32 r367, [r432+512];
barrier.sync 0;
st.shared.u32 [r430], r316;
st.shared.u32 [r430+256], r344;
barrier.sync 0;
ld.shared.u32 r369, [r432];
ld.shared.u32 r370, [r432+512];
{
add.f16x2 %0, r366, r367;
}
{
add.f16x2 %1, r369, r370;
}
{
sub.f16x2 %2, r366, r367;
}
{
sub.f16x2 %3, r369, r370;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)));
};


#endif
