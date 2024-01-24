#ifndef CUFFTDX_FFT_128_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_128_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<999, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<210>;
.reg .b32 r<1579>;
.reg .b64 rd<2>;
mov.u32 r1567, %tid.y;
shl.b32 r1568, r1567, 9;
mov.u32 r1569, %32;
add.s32 r1570, r1569, r1568;
mov.u32 r1571, %tid.x;
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
mov.f32 f190, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r102, {low, high};
}
mov.f32 f148, 0f3F800000;
mov.f32 f188, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r106, {low, high};
}
mov.f32 f147, 0fBF800000;
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
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
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
mov.f32 f76, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r393, {low, high};
}
mov.f32 f84, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r396, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r398, {low, high};
}
mov.f32 f74, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r404, {low, high};
}
mov.f32 f82, 0fBF6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r405, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
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
and.b32 r1572, r1571, 7;
shl.b32 r1573, r1571, 6;
and.b32 r1574, r1573, -512;
add.s32 r1575, r1570, r1574;
cvt.rn.f32.u32 f207, r1572;
mul.f32 f208, f207, 0f3D490FDB;
cos.approx.f32 f117, f208;
sin.approx.f32 f209, f208;
neg.f32 f118, f209;
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
and.b32 r1576, r1573, 448;
add.s32 r1577, r1575, r1576;
st.shared.v4.f32 [r1577], {r521, r627, r664, r701};
st.shared.v4.f32 [r1577+16], {r738, r775, r812, r849};
st.shared.v4.f32 [r1577+32], {r886, r923, r960, r997};
st.shared.v4.f32 [r1577+48], {r1034, r1071, r1108, r1145};
barrier.sync 0;
mad.lo.s32 r1578, r1572, -60, r1577;
ld.shared.u32 r1176, [r1578];
ld.shared.u32 r1372, [r1578+32];
ld.shared.u32 r1226, [r1578+64];
ld.shared.u32 r1422, [r1578+96];
ld.shared.u32 r1188, [r1578+128];
ld.shared.u32 r1384, [r1578+160];
ld.shared.u32 r1238, [r1578+192];
ld.shared.u32 r1434, [r1578+224];
ld.shared.u32 r1177, [r1578+256];
ld.shared.u32 r1373, [r1578+288];
ld.shared.u32 r1227, [r1578+320];
ld.shared.u32 r1423, [r1578+352];
ld.shared.u32 r1189, [r1578+384];
ld.shared.u32 r1385, [r1578+416];
ld.shared.u32 r1239, [r1578+448];
ld.shared.u32 r1435, [r1578+480];
barrier.sync 0;
st.shared.v4.f32 [r1577], {r524, r636, r673, r710};
st.shared.v4.f32 [r1577+16], {r747, r784, r821, r858};
st.shared.v4.f32 [r1577+32], {r895, r932, r969, r1006};
st.shared.v4.f32 [r1577+48], {r1043, r1080, r1117, r1154};
barrier.sync 0;
ld.shared.u32 r1179, [r1578];
ld.shared.u32 r1375, [r1578+32];
ld.shared.u32 r1229, [r1578+64];
ld.shared.u32 r1425, [r1578+96];
ld.shared.u32 r1191, [r1578+128];
ld.shared.u32 r1387, [r1578+160];
ld.shared.u32 r1241, [r1578+192];
ld.shared.u32 r1437, [r1578+224];
ld.shared.u32 r1180, [r1578+256];
ld.shared.u32 r1376, [r1578+288];
ld.shared.u32 r1230, [r1578+320];
ld.shared.u32 r1426, [r1578+352];
ld.shared.u32 r1192, [r1578+384];
ld.shared.u32 r1388, [r1578+416];
ld.shared.u32 r1242, [r1578+448];
ld.shared.u32 r1438, [r1578+480];
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
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r1275, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r1276, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r1279, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
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
add.f16x2 %0, r1201, r1251;
}
{
add.f16x2 %1, r1204, r1254;
}
{
sub.f16x2 %16, r1201, r1251;
}
{
sub.f16x2 %17, r1204, r1254;
}
{
add.f16x2 %4, r1213, r1295;
}
{
add.f16x2 %5, r1216, r1301;
}
{
sub.f16x2 %20, r1213, r1295;
}
{
sub.f16x2 %21, r1216, r1301;
}
{
add.f16x2 %8, r1207, r1305;
}
{
add.f16x2 %9, r1210, r1257;
}
{
sub.f16x2 %24, r1207, r1305;
}
{
sub.f16x2 %25, r1210, r1257;
}
{
add.f16x2 %12, r1219, r1313;
}
{
add.f16x2 %13, r1222, r1319;
}
{
sub.f16x2 %28, r1219, r1313;
}
{
sub.f16x2 %29, r1222, r1319;
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
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r1471, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r1472, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r1475, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
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
add.f16x2 %2, r1397, r1447;
}
{
add.f16x2 %3, r1400, r1450;
}
{
sub.f16x2 %18, r1397, r1447;
}
{
sub.f16x2 %19, r1400, r1450;
}
{
add.f16x2 %6, r1409, r1491;
}
{
add.f16x2 %7, r1412, r1497;
}
{
sub.f16x2 %22, r1409, r1491;
}
{
sub.f16x2 %23, r1412, r1497;
}
{
add.f16x2 %10, r1403, r1501;
}
{
add.f16x2 %11, r1406, r1453;
}
{
sub.f16x2 %26, r1403, r1501;
}
{
sub.f16x2 %27, r1406, r1453;
}
{
add.f16x2 %14, r1415, r1509;
}
{
add.f16x2 %15, r1418, r1515;
}
{
sub.f16x2 %30, r1415, r1509;
}
{
sub.f16x2 %31, r1418, r1515;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1000, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<99>;
.reg .b32 r<985>;
.reg .b64 rd<2>;
mov.u32 r965, %tid.y;
shl.b32 r966, r965, 10;
mov.u32 r967, %16;
add.s32 r968, r967, r966;
mov.u32 r969, %tid.x;
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
mov.f32 f58, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r102, {low, high};
}
mov.f32 f90, 0f3F800000;
mov.f32 f56, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r106, {low, high};
}
mov.f32 f89, 0fBF800000;
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
and.b32 r970, r969, 15;
shl.b32 r971, r969, 6;
and.b32 r972, r971, -1024;
add.s32 r973, r968, r972;
cvt.rn.f32.u32 f93, r970;
mul.f32 f94, f93, 0f3D490FDB;
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
fma.rn.f16x2 r207, r161, r200, r204;
}
{
mul.f16x2 r211, r161, r202;
}
{
neg.f16x2 r214, r211;
}
{
fma.rn.f16x2 r216, r164, r200, r214;
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
fma.rn.f16x2 r244, r173, r237, r241;
}
{
mul.f16x2 r248, r173, r239;
}
{
neg.f16x2 r251, r248;
}
{
fma.rn.f16x2 r253, r176, r237, r251;
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
fma.rn.f16x2 r281, r185, r274, r278;
}
{
mul.f16x2 r285, r185, r276;
}
{
neg.f16x2 r288, r285;
}
{
fma.rn.f16x2 r290, r188, r274, r288;
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
fma.rn.f16x2 r318, r155, r311, r315;
}
{
mul.f16x2 r322, r155, r313;
}
{
neg.f16x2 r325, r322;
}
{
fma.rn.f16x2 r327, r158, r311, r325;
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
fma.rn.f16x2 r355, r167, r348, r352;
}
{
mul.f16x2 r359, r167, r350;
}
{
neg.f16x2 r362, r359;
}
{
fma.rn.f16x2 r364, r170, r348, r362;
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
fma.rn.f16x2 r392, r179, r385, r389;
}
{
mul.f16x2 r396, r179, r387;
}
{
neg.f16x2 r399, r396;
}
{
fma.rn.f16x2 r401, r182, r385, r399;
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
fma.rn.f16x2 r429, r191, r422, r426;
}
{
mul.f16x2 r433, r191, r424;
}
{
neg.f16x2 r436, r433;
}
{
fma.rn.f16x2 r438, r194, r422, r436;
}
barrier.sync 0;
and.b32 r974, r971, 960;
add.s32 r975, r973, r974;
st.shared.v4.f32 [r975], {r149, r152, r207, r216};
st.shared.v4.f32 [r975+16], {r244, r253, r281, r290};
st.shared.v4.f32 [r975+32], {r318, r327, r355, r364};
st.shared.v4.f32 [r975+48], {r392, r401, r429, r438};
barrier.sync 0;
mad.lo.s32 r976, r970, -56, r975;
ld.shared.u32 r460, [r976];
ld.shared.u32 r463, [r976+4];
ld.shared.u32 r510, [r976+128];
ld.shared.u32 r513, [r976+132];
ld.shared.u32 r472, [r976+256];
ld.shared.u32 r475, [r976+260];
ld.shared.u32 r522, [r976+384];
ld.shared.u32 r525, [r976+388];
ld.shared.u32 r461, [r976+512];
ld.shared.u32 r464, [r976+516];
ld.shared.u32 r511, [r976+640];
ld.shared.u32 r514, [r976+644];
ld.shared.u32 r473, [r976+768];
ld.shared.u32 r476, [r976+772];
ld.shared.u32 r523, [r976+896];
ld.shared.u32 r526, [r976+900];
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
neg.f16x2 r483, r480;
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
add.f16x2 r497, r465, r483;
}
{
add.f16x2 r500, r468, r477;
}
{
sub.f16x2 r503, r465, r483;
}
{
sub.f16x2 r506, r468, r477;
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
neg.f16x2 r533, r530;
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
add.f16x2 r547, r515, r533;
}
{
add.f16x2 r550, r518, r527;
}
{
sub.f16x2 r553, r515, r533;
}
{
sub.f16x2 r556, r518, r527;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
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
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
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
neg.f16x2 r589, r544;
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
add.f16x2 r631, r491, r589;
}
{
add.f16x2 r634, r494, r541;
}
{
sub.f16x2 r637, r491, r589;
}
{
sub.f16x2 r640, r494, r541;
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
and.b32 r977, r969, 8;
bfe.u32 r978, r969, 3, 1;
cvt.rn.f32.u32 f96, r978;
mul.f32 f97, f96, 0f3EC90FDB;
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
fma.rn.f16x2 r665, r619, r658, r662;
}
{
mul.f16x2 r669, r619, r660;
}
{
neg.f16x2 r672, r669;
}
{
fma.rn.f16x2 r674, r622, r658, r672;
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
fma.rn.f16x2 r702, r631, r695, r699;
}
{
mul.f16x2 r706, r631, r697;
}
{
neg.f16x2 r709, r706;
}
{
fma.rn.f16x2 r711, r634, r695, r709;
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
fma.rn.f16x2 r739, r643, r732, r736;
}
{
mul.f16x2 r743, r643, r734;
}
{
neg.f16x2 r746, r743;
}
{
fma.rn.f16x2 r748, r646, r732, r746;
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
fma.rn.f16x2 r776, r613, r769, r773;
}
{
mul.f16x2 r780, r613, r771;
}
{
neg.f16x2 r783, r780;
}
{
fma.rn.f16x2 r785, r616, r769, r783;
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
fma.rn.f16x2 r813, r625, r806, r810;
}
{
mul.f16x2 r817, r625, r808;
}
{
neg.f16x2 r820, r817;
}
{
fma.rn.f16x2 r822, r628, r806, r820;
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
fma.rn.f16x2 r850, r637, r843, r847;
}
{
mul.f16x2 r854, r637, r845;
}
{
neg.f16x2 r857, r854;
}
{
fma.rn.f16x2 r859, r640, r843, r857;
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
fma.rn.f16x2 r887, r649, r880, r884;
}
{
mul.f16x2 r891, r649, r882;
}
{
neg.f16x2 r894, r891;
}
{
fma.rn.f16x2 r896, r652, r880, r894;
}
shl.b32 r979, r969, 3;
and.b32 r980, r979, 56;
add.s32 r981, r973, r980;
barrier.sync 0;
and.b32 r982, r971, 512;
add.s32 r983, r981, r982;
st.shared.u32 [r983], r607;
st.shared.u32 [r983+4], r610;
st.shared.u32 [r983+64], r665;
st.shared.u32 [r983+68], r674;
st.shared.u32 [r983+128], r702;
st.shared.u32 [r983+132], r711;
st.shared.u32 [r983+192], r739;
st.shared.u32 [r983+196], r748;
st.shared.u32 [r983+256], r776;
st.shared.u32 [r983+260], r785;
st.shared.u32 [r983+320], r813;
st.shared.u32 [r983+324], r822;
st.shared.u32 [r983+384], r850;
st.shared.u32 [r983+388], r859;
st.shared.u32 [r983+448], r887;
st.shared.u32 [r983+452], r896;
barrier.sync 0;
mad.lo.s32 r984, r977, -56, r983;
ld.shared.u32 r918, [r984];
ld.shared.u32 r921, [r984+4];
ld.shared.u32 r930, [r984+128];
ld.shared.u32 r933, [r984+132];
ld.shared.u32 r942, [r984+256];
ld.shared.u32 r945, [r984+260];
ld.shared.u32 r954, [r984+384];
ld.shared.u32 r957, [r984+388];
ld.shared.u32 r919, [r984+512];
ld.shared.u32 r922, [r984+516];
ld.shared.u32 r931, [r984+640];
ld.shared.u32 r934, [r984+644];
ld.shared.u32 r943, [r984+768];
ld.shared.u32 r946, [r984+772];
ld.shared.u32 r955, [r984+896];
ld.shared.u32 r958, [r984+900];
{
add.f16x2 %0, r918, r919;
}
{
add.f16x2 %1, r921, r922;
}
{
sub.f16x2 %8, r918, r919;
}
{
sub.f16x2 %9, r921, r922;
}
{
add.f16x2 %2, r930, r931;
}
{
add.f16x2 %3, r933, r934;
}
{
sub.f16x2 %10, r930, r931;
}
{
sub.f16x2 %11, r933, r934;
}
{
add.f16x2 %4, r942, r943;
}
{
add.f16x2 %5, r945, r946;
}
{
sub.f16x2 %12, r942, r943;
}
{
sub.f16x2 %13, r945, r946;
}
{
add.f16x2 %6, r954, r955;
}
{
add.f16x2 %7, r957, r958;
}
{
sub.f16x2 %14, r954, r955;
}
{
sub.f16x2 %15, r957, r958;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1001, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<210>;
.reg .b32 r<1579>;
.reg .b64 rd<2>;
mov.u32 r1567, %tid.y;
shl.b32 r1568, r1567, 10;
mov.u32 r1569, %32;
add.s32 r1570, r1569, r1568;
mov.u32 r1571, %tid.x;
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
mov.f32 f190, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r102, {low, high};
}
mov.f32 f148, 0f3F800000;
mov.f32 f188, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r106, {low, high};
}
mov.f32 f147, 0fBF800000;
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
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
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
mov.f32 f76, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r393, {low, high};
}
mov.f32 f84, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r396, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r398, {low, high};
}
mov.f32 f74, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r404, {low, high};
}
mov.f32 f82, 0fBF6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r405, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
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
and.b32 r1572, r1571, 7;
shl.b32 r1573, r1571, 7;
and.b32 r1574, r1573, -1024;
add.s32 r1575, r1570, r1574;
cvt.rn.f32.u32 f207, r1572;
mul.f32 f208, f207, 0f3D490FDB;
cos.approx.f32 f117, f208;
sin.approx.f32 f209, f208;
neg.f32 f118, f209;
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
and.b32 r1576, r1573, 896;
add.s32 r1577, r1575, r1576;
st.shared.v4.f32 [r1577], {r521, r524, r627, r636};
st.shared.v4.f32 [r1577+16], {r664, r673, r701, r710};
st.shared.v4.f32 [r1577+32], {r738, r747, r775, r784};
st.shared.v4.f32 [r1577+48], {r812, r821, r849, r858};
st.shared.v4.f32 [r1577+64], {r886, r895, r923, r932};
st.shared.v4.f32 [r1577+80], {r960, r969, r997, r1006};
st.shared.v4.f32 [r1577+96], {r1034, r1043, r1071, r1080};
st.shared.v4.f32 [r1577+112], {r1108, r1117, r1145, r1154};
barrier.sync 0;
mad.lo.s32 r1578, r1572, -120, r1577;
ld.shared.u32 r1176, [r1578];
ld.shared.u32 r1179, [r1578+4];
ld.shared.u32 r1372, [r1578+64];
ld.shared.u32 r1375, [r1578+68];
ld.shared.u32 r1226, [r1578+128];
ld.shared.u32 r1229, [r1578+132];
ld.shared.u32 r1422, [r1578+192];
ld.shared.u32 r1425, [r1578+196];
ld.shared.u32 r1188, [r1578+256];
ld.shared.u32 r1191, [r1578+260];
ld.shared.u32 r1384, [r1578+320];
ld.shared.u32 r1387, [r1578+324];
ld.shared.u32 r1238, [r1578+384];
ld.shared.u32 r1241, [r1578+388];
ld.shared.u32 r1434, [r1578+448];
ld.shared.u32 r1437, [r1578+452];
ld.shared.u32 r1177, [r1578+512];
ld.shared.u32 r1180, [r1578+516];
ld.shared.u32 r1373, [r1578+576];
ld.shared.u32 r1376, [r1578+580];
ld.shared.u32 r1227, [r1578+640];
ld.shared.u32 r1230, [r1578+644];
ld.shared.u32 r1423, [r1578+704];
ld.shared.u32 r1426, [r1578+708];
ld.shared.u32 r1189, [r1578+768];
ld.shared.u32 r1192, [r1578+772];
ld.shared.u32 r1385, [r1578+832];
ld.shared.u32 r1388, [r1578+836];
ld.shared.u32 r1239, [r1578+896];
ld.shared.u32 r1242, [r1578+900];
ld.shared.u32 r1435, [r1578+960];
ld.shared.u32 r1438, [r1578+964];
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
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r1275, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r1276, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r1279, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
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
add.f16x2 %0, r1201, r1251;
}
{
add.f16x2 %1, r1204, r1254;
}
{
sub.f16x2 %16, r1201, r1251;
}
{
sub.f16x2 %17, r1204, r1254;
}
{
add.f16x2 %4, r1213, r1295;
}
{
add.f16x2 %5, r1216, r1301;
}
{
sub.f16x2 %20, r1213, r1295;
}
{
sub.f16x2 %21, r1216, r1301;
}
{
add.f16x2 %8, r1207, r1305;
}
{
add.f16x2 %9, r1210, r1257;
}
{
sub.f16x2 %24, r1207, r1305;
}
{
sub.f16x2 %25, r1210, r1257;
}
{
add.f16x2 %12, r1219, r1313;
}
{
add.f16x2 %13, r1222, r1319;
}
{
sub.f16x2 %28, r1219, r1313;
}
{
sub.f16x2 %29, r1222, r1319;
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
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r1471, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r1472, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r1475, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
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
add.f16x2 %2, r1397, r1447;
}
{
add.f16x2 %3, r1400, r1450;
}
{
sub.f16x2 %18, r1397, r1447;
}
{
sub.f16x2 %19, r1400, r1450;
}
{
add.f16x2 %6, r1409, r1491;
}
{
add.f16x2 %7, r1412, r1497;
}
{
sub.f16x2 %22, r1409, r1491;
}
{
sub.f16x2 %23, r1412, r1497;
}
{
add.f16x2 %10, r1403, r1501;
}
{
add.f16x2 %11, r1406, r1453;
}
{
sub.f16x2 %26, r1403, r1501;
}
{
sub.f16x2 %27, r1406, r1453;
}
{
add.f16x2 %14, r1415, r1509;
}
{
add.f16x2 %15, r1418, r1515;
}
{
sub.f16x2 %30, r1415, r1509;
}
{
sub.f16x2 %31, r1418, r1515;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1002, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<99>;
.reg .b32 r<985>;
.reg .b64 rd<2>;
mov.u32 r965, %tid.y;
shl.b32 r966, r965, 9;
mov.u32 r967, %16;
add.s32 r968, r967, r966;
mov.u32 r969, %tid.x;
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
mov.f32 f58, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r102, {low, high};
}
mov.f32 f90, 0f3F800000;
mov.f32 f56, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r106, {low, high};
}
mov.f32 f89, 0fBF800000;
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
and.b32 r970, r969, 15;
shl.b32 r971, r969, 5;
and.b32 r972, r971, -512;
add.s32 r973, r968, r972;
cvt.rn.f32.u32 f93, r970;
mul.f32 f94, f93, 0f3D490FDB;
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
fma.rn.f16x2 r207, r161, r200, r204;
}
{
mul.f16x2 r211, r161, r202;
}
{
neg.f16x2 r214, r211;
}
{
fma.rn.f16x2 r216, r164, r200, r214;
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
fma.rn.f16x2 r244, r173, r237, r241;
}
{
mul.f16x2 r248, r173, r239;
}
{
neg.f16x2 r251, r248;
}
{
fma.rn.f16x2 r253, r176, r237, r251;
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
fma.rn.f16x2 r281, r185, r274, r278;
}
{
mul.f16x2 r285, r185, r276;
}
{
neg.f16x2 r288, r285;
}
{
fma.rn.f16x2 r290, r188, r274, r288;
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
fma.rn.f16x2 r318, r155, r311, r315;
}
{
mul.f16x2 r322, r155, r313;
}
{
neg.f16x2 r325, r322;
}
{
fma.rn.f16x2 r327, r158, r311, r325;
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
fma.rn.f16x2 r355, r167, r348, r352;
}
{
mul.f16x2 r359, r167, r350;
}
{
neg.f16x2 r362, r359;
}
{
fma.rn.f16x2 r364, r170, r348, r362;
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
fma.rn.f16x2 r392, r179, r385, r389;
}
{
mul.f16x2 r396, r179, r387;
}
{
neg.f16x2 r399, r396;
}
{
fma.rn.f16x2 r401, r182, r385, r399;
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
fma.rn.f16x2 r429, r191, r422, r426;
}
{
mul.f16x2 r433, r191, r424;
}
{
neg.f16x2 r436, r433;
}
{
fma.rn.f16x2 r438, r194, r422, r436;
}
barrier.sync 0;
and.b32 r974, r971, 480;
add.s32 r975, r973, r974;
st.shared.v4.f32 [r975], {r149, r207, r244, r281};
st.shared.v4.f32 [r975+16], {r318, r355, r392, r429};
barrier.sync 0;
mad.lo.s32 r976, r970, -28, r975;
ld.shared.u32 r460, [r976];
ld.shared.u32 r510, [r976+64];
ld.shared.u32 r472, [r976+128];
ld.shared.u32 r522, [r976+192];
ld.shared.u32 r461, [r976+256];
ld.shared.u32 r511, [r976+320];
ld.shared.u32 r473, [r976+384];
ld.shared.u32 r523, [r976+448];
barrier.sync 0;
st.shared.v4.f32 [r975], {r152, r216, r253, r290};
st.shared.v4.f32 [r975+16], {r327, r364, r401, r438};
barrier.sync 0;
ld.shared.u32 r463, [r976];
ld.shared.u32 r513, [r976+64];
ld.shared.u32 r475, [r976+128];
ld.shared.u32 r525, [r976+192];
ld.shared.u32 r464, [r976+256];
ld.shared.u32 r514, [r976+320];
ld.shared.u32 r476, [r976+384];
ld.shared.u32 r526, [r976+448];
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
neg.f16x2 r483, r480;
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
add.f16x2 r497, r465, r483;
}
{
add.f16x2 r500, r468, r477;
}
{
sub.f16x2 r503, r465, r483;
}
{
sub.f16x2 r506, r468, r477;
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
neg.f16x2 r533, r530;
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
add.f16x2 r547, r515, r533;
}
{
add.f16x2 r550, r518, r527;
}
{
sub.f16x2 r553, r515, r533;
}
{
sub.f16x2 r556, r518, r527;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
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
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
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
neg.f16x2 r589, r544;
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
add.f16x2 r631, r491, r589;
}
{
add.f16x2 r634, r494, r541;
}
{
sub.f16x2 r637, r491, r589;
}
{
sub.f16x2 r640, r494, r541;
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
and.b32 r977, r969, 8;
bfe.u32 r978, r969, 3, 1;
shl.b32 r979, r969, 2;
and.b32 r980, r979, 28;
add.s32 r981, r973, r980;
cvt.rn.f32.u32 f96, r978;
mul.f32 f97, f96, 0f3EC90FDB;
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
fma.rn.f16x2 r665, r619, r658, r662;
}
{
mul.f16x2 r669, r619, r660;
}
{
neg.f16x2 r672, r669;
}
{
fma.rn.f16x2 r674, r622, r658, r672;
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
fma.rn.f16x2 r702, r631, r695, r699;
}
{
mul.f16x2 r706, r631, r697;
}
{
neg.f16x2 r709, r706;
}
{
fma.rn.f16x2 r711, r634, r695, r709;
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
fma.rn.f16x2 r739, r643, r732, r736;
}
{
mul.f16x2 r743, r643, r734;
}
{
neg.f16x2 r746, r743;
}
{
fma.rn.f16x2 r748, r646, r732, r746;
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
fma.rn.f16x2 r776, r613, r769, r773;
}
{
mul.f16x2 r780, r613, r771;
}
{
neg.f16x2 r783, r780;
}
{
fma.rn.f16x2 r785, r616, r769, r783;
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
fma.rn.f16x2 r813, r625, r806, r810;
}
{
mul.f16x2 r817, r625, r808;
}
{
neg.f16x2 r820, r817;
}
{
fma.rn.f16x2 r822, r628, r806, r820;
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
fma.rn.f16x2 r850, r637, r843, r847;
}
{
mul.f16x2 r854, r637, r845;
}
{
neg.f16x2 r857, r854;
}
{
fma.rn.f16x2 r859, r640, r843, r857;
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
fma.rn.f16x2 r887, r649, r880, r884;
}
{
mul.f16x2 r891, r649, r882;
}
{
neg.f16x2 r894, r891;
}
{
fma.rn.f16x2 r896, r652, r880, r894;
}
barrier.sync 0;
and.b32 r982, r971, 256;
add.s32 r983, r981, r982;
st.shared.u32 [r983], r607;
st.shared.u32 [r983+32], r665;
st.shared.u32 [r983+64], r702;
st.shared.u32 [r983+96], r739;
st.shared.u32 [r983+128], r776;
st.shared.u32 [r983+160], r813;
st.shared.u32 [r983+192], r850;
st.shared.u32 [r983+224], r887;
barrier.sync 0;
mad.lo.s32 r984, r977, -28, r983;
ld.shared.u32 r918, [r984];
ld.shared.u32 r930, [r984+64];
ld.shared.u32 r942, [r984+128];
ld.shared.u32 r954, [r984+192];
ld.shared.u32 r919, [r984+256];
ld.shared.u32 r931, [r984+320];
ld.shared.u32 r943, [r984+384];
ld.shared.u32 r955, [r984+448];
barrier.sync 0;
st.shared.u32 [r983], r610;
st.shared.u32 [r983+32], r674;
st.shared.u32 [r983+64], r711;
st.shared.u32 [r983+96], r748;
st.shared.u32 [r983+128], r785;
st.shared.u32 [r983+160], r822;
st.shared.u32 [r983+192], r859;
st.shared.u32 [r983+224], r896;
barrier.sync 0;
ld.shared.u32 r921, [r984];
ld.shared.u32 r933, [r984+64];
ld.shared.u32 r945, [r984+128];
ld.shared.u32 r957, [r984+192];
ld.shared.u32 r922, [r984+256];
ld.shared.u32 r934, [r984+320];
ld.shared.u32 r946, [r984+384];
ld.shared.u32 r958, [r984+448];
{
add.f16x2 %0, r918, r919;
}
{
add.f16x2 %1, r921, r922;
}
{
sub.f16x2 %8, r918, r919;
}
{
sub.f16x2 %9, r921, r922;
}
{
add.f16x2 %2, r930, r931;
}
{
add.f16x2 %3, r933, r934;
}
{
sub.f16x2 %10, r930, r931;
}
{
sub.f16x2 %11, r933, r934;
}
{
add.f16x2 %4, r942, r943;
}
{
add.f16x2 %5, r945, r946;
}
{
sub.f16x2 %12, r942, r943;
}
{
sub.f16x2 %13, r945, r946;
}
{
add.f16x2 %6, r954, r955;
}
{
add.f16x2 %7, r957, r958;
}
{
sub.f16x2 %14, r954, r955;
}
{
sub.f16x2 %15, r957, r958;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1003, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<40>;
.reg .b32 r<544>;
.reg .b64 rd<2>;
mov.u32 r517, %tid.y;
shl.b32 r518, r517, 10;
mov.u32 r519, %8;
add.s32 r520, r519, r518;
mov.u32 r521, %tid.x;
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
and.b32 r522, r521, 31;
shl.b32 r523, r521, 5;
and.b32 r524, r523, -1024;
add.s32 r525, r520, r524;
cvt.rn.f32.u32 f31, r522;
mul.f32 f32, f31, 0f3D490FDB;
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
fma.rn.f16x2 r61, r39, r54, r58;
}
{
mul.f16x2 r65, r39, r56;
}
{
neg.f16x2 r68, r65;
}
{
fma.rn.f16x2 r70, r42, r54, r68;
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
fma.rn.f16x2 r98, r33, r91, r95;
}
{
mul.f16x2 r102, r33, r93;
}
{
neg.f16x2 r105, r102;
}
{
fma.rn.f16x2 r107, r36, r91, r105;
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
fma.rn.f16x2 r135, r45, r128, r132;
}
{
mul.f16x2 r139, r45, r130;
}
{
neg.f16x2 r142, r139;
}
{
fma.rn.f16x2 r144, r48, r128, r142;
}
barrier.sync 0;
and.b32 r526, r523, 992;
add.s32 r527, r525, r526;
st.shared.v4.f32 [r527], {r27, r30, r61, r70};
st.shared.v4.f32 [r527+16], {r98, r107, r135, r144};
barrier.sync 0;
mad.lo.s32 r528, r522, -24, r527;
ld.shared.u32 r166, [r528];
ld.shared.u32 r169, [r528+4];
ld.shared.u32 r178, [r528+256];
ld.shared.u32 r181, [r528+260];
ld.shared.u32 r167, [r528+512];
ld.shared.u32 r170, [r528+516];
ld.shared.u32 r179, [r528+768];
ld.shared.u32 r182, [r528+772];
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
neg.f16x2 r189, r186;
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
add.f16x2 r203, r171, r189;
}
{
add.f16x2 r206, r174, r183;
}
{
sub.f16x2 r209, r171, r189;
}
{
sub.f16x2 r212, r174, r183;
}
and.b32 r529, r521, 28;
bfe.u32 r530, r521, 2, 3;
cvt.rn.f32.u32 f34, r530;
mul.f32 f35, f34, 0f3E490FDB;
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
fma.rn.f16x2 r225, r203, r218, r222;
}
{
mul.f16x2 r229, r203, r220;
}
{
neg.f16x2 r232, r229;
}
{
fma.rn.f16x2 r234, r206, r218, r232;
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
fma.rn.f16x2 r262, r197, r255, r259;
}
{
mul.f16x2 r266, r197, r257;
}
{
neg.f16x2 r269, r266;
}
{
fma.rn.f16x2 r271, r200, r255, r269;
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
fma.rn.f16x2 r299, r209, r292, r296;
}
{
mul.f16x2 r303, r209, r294;
}
{
neg.f16x2 r306, r303;
}
{
fma.rn.f16x2 r308, r212, r292, r306;
}
shl.b32 r531, r521, 3;
and.b32 r532, r531, 24;
add.s32 r533, r525, r532;
barrier.sync 0;
and.b32 r534, r523, 896;
add.s32 r535, r533, r534;
st.shared.u32 [r535], r191;
st.shared.u32 [r535+4], r194;
st.shared.u32 [r535+32], r225;
st.shared.u32 [r535+36], r234;
st.shared.u32 [r535+64], r262;
st.shared.u32 [r535+68], r271;
st.shared.u32 [r535+96], r299;
st.shared.u32 [r535+100], r308;
barrier.sync 0;
mad.lo.s32 r536, r529, -24, r535;
ld.shared.u32 r330, [r536];
ld.shared.u32 r333, [r536+4];
ld.shared.u32 r342, [r536+256];
ld.shared.u32 r345, [r536+260];
ld.shared.u32 r331, [r536+512];
ld.shared.u32 r334, [r536+516];
ld.shared.u32 r343, [r536+768];
ld.shared.u32 r346, [r536+772];
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
neg.f16x2 r353, r350;
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
add.f16x2 r367, r335, r353;
}
{
add.f16x2 r370, r338, r347;
}
{
sub.f16x2 r373, r335, r353;
}
{
sub.f16x2 r376, r338, r347;
}
and.b32 r537, r521, 16;
bfe.u32 r538, r521, 4, 1;
cvt.rn.f32.u32 f37, r538;
mul.f32 f38, f37, 0f3F490FDB;
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
fma.rn.f16x2 r389, r367, r382, r386;
}
{
mul.f16x2 r393, r367, r384;
}
{
neg.f16x2 r396, r393;
}
{
fma.rn.f16x2 r398, r370, r382, r396;
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
fma.rn.f16x2 r426, r361, r419, r423;
}
{
mul.f16x2 r430, r361, r421;
}
{
neg.f16x2 r433, r430;
}
{
fma.rn.f16x2 r435, r364, r419, r433;
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
fma.rn.f16x2 r463, r373, r456, r460;
}
{
mul.f16x2 r467, r373, r458;
}
{
neg.f16x2 r470, r467;
}
{
fma.rn.f16x2 r472, r376, r456, r470;
}
and.b32 r539, r531, 120;
add.s32 r540, r525, r539;
barrier.sync 0;
and.b32 r541, r523, 512;
add.s32 r542, r540, r541;
st.shared.u32 [r542], r355;
st.shared.u32 [r542+4], r358;
st.shared.u32 [r542+128], r389;
st.shared.u32 [r542+132], r398;
st.shared.u32 [r542+256], r426;
st.shared.u32 [r542+260], r435;
st.shared.u32 [r542+384], r463;
st.shared.u32 [r542+388], r472;
barrier.sync 0;
mad.lo.s32 r543, r537, -24, r542;
ld.shared.u32 r494, [r543];
ld.shared.u32 r497, [r543+4];
ld.shared.u32 r506, [r543+256];
ld.shared.u32 r509, [r543+260];
ld.shared.u32 r495, [r543+512];
ld.shared.u32 r498, [r543+516];
ld.shared.u32 r507, [r543+768];
ld.shared.u32 r510, [r543+772];
{
add.f16x2 %0, r494, r495;
}
{
add.f16x2 %1, r497, r498;
}
{
sub.f16x2 %4, r494, r495;
}
{
sub.f16x2 %5, r497, r498;
}
{
add.f16x2 %2, r506, r507;
}
{
add.f16x2 %3, r509, r510;
}
{
sub.f16x2 %6, r506, r507;
}
{
sub.f16x2 %7, r509, r510;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1004, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<40>;
.reg .b32 r<544>;
.reg .b64 rd<2>;
mov.u32 r517, %tid.y;
shl.b32 r518, r517, 9;
mov.u32 r519, %8;
add.s32 r520, r519, r518;
mov.u32 r521, %tid.x;
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
and.b32 r522, r521, 31;
shl.b32 r523, r521, 4;
and.b32 r524, r523, -512;
add.s32 r525, r520, r524;
cvt.rn.f32.u32 f31, r522;
mul.f32 f32, f31, 0f3D490FDB;
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
fma.rn.f16x2 r61, r39, r54, r58;
}
{
mul.f16x2 r65, r39, r56;
}
{
neg.f16x2 r68, r65;
}
{
fma.rn.f16x2 r70, r42, r54, r68;
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
fma.rn.f16x2 r98, r33, r91, r95;
}
{
mul.f16x2 r102, r33, r93;
}
{
neg.f16x2 r105, r102;
}
{
fma.rn.f16x2 r107, r36, r91, r105;
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
fma.rn.f16x2 r135, r45, r128, r132;
}
{
mul.f16x2 r139, r45, r130;
}
{
neg.f16x2 r142, r139;
}
{
fma.rn.f16x2 r144, r48, r128, r142;
}
barrier.sync 0;
and.b32 r526, r523, 496;
add.s32 r527, r525, r526;
st.shared.v4.f32 [r527], {r27, r61, r98, r135};
barrier.sync 0;
mad.lo.s32 r528, r522, -12, r527;
ld.shared.u32 r166, [r528];
ld.shared.u32 r178, [r528+128];
ld.shared.u32 r167, [r528+256];
ld.shared.u32 r179, [r528+384];
barrier.sync 0;
st.shared.v4.f32 [r527], {r30, r70, r107, r144};
barrier.sync 0;
ld.shared.u32 r169, [r528];
ld.shared.u32 r181, [r528+128];
ld.shared.u32 r170, [r528+256];
ld.shared.u32 r182, [r528+384];
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
neg.f16x2 r189, r186;
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
add.f16x2 r203, r171, r189;
}
{
add.f16x2 r206, r174, r183;
}
{
sub.f16x2 r209, r171, r189;
}
{
sub.f16x2 r212, r174, r183;
}
and.b32 r529, r521, 28;
bfe.u32 r530, r521, 2, 3;
shl.b32 r531, r521, 2;
and.b32 r532, r531, 12;
add.s32 r533, r525, r532;
cvt.rn.f32.u32 f34, r530;
mul.f32 f35, f34, 0f3E490FDB;
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
fma.rn.f16x2 r225, r203, r218, r222;
}
{
mul.f16x2 r229, r203, r220;
}
{
neg.f16x2 r232, r229;
}
{
fma.rn.f16x2 r234, r206, r218, r232;
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
fma.rn.f16x2 r262, r197, r255, r259;
}
{
mul.f16x2 r266, r197, r257;
}
{
neg.f16x2 r269, r266;
}
{
fma.rn.f16x2 r271, r200, r255, r269;
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
fma.rn.f16x2 r299, r209, r292, r296;
}
{
mul.f16x2 r303, r209, r294;
}
{
neg.f16x2 r306, r303;
}
{
fma.rn.f16x2 r308, r212, r292, r306;
}
barrier.sync 0;
and.b32 r534, r523, 448;
add.s32 r535, r533, r534;
st.shared.u32 [r535], r191;
st.shared.u32 [r535+16], r225;
st.shared.u32 [r535+32], r262;
st.shared.u32 [r535+48], r299;
barrier.sync 0;
mad.lo.s32 r536, r529, -12, r535;
ld.shared.u32 r330, [r536];
ld.shared.u32 r342, [r536+128];
ld.shared.u32 r331, [r536+256];
ld.shared.u32 r343, [r536+384];
barrier.sync 0;
st.shared.u32 [r535], r194;
st.shared.u32 [r535+16], r234;
st.shared.u32 [r535+32], r271;
st.shared.u32 [r535+48], r308;
barrier.sync 0;
ld.shared.u32 r333, [r536];
ld.shared.u32 r345, [r536+128];
ld.shared.u32 r334, [r536+256];
ld.shared.u32 r346, [r536+384];
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
neg.f16x2 r353, r350;
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
add.f16x2 r367, r335, r353;
}
{
add.f16x2 r370, r338, r347;
}
{
sub.f16x2 r373, r335, r353;
}
{
sub.f16x2 r376, r338, r347;
}
and.b32 r537, r521, 16;
bfe.u32 r538, r521, 4, 1;
and.b32 r539, r531, 60;
add.s32 r540, r525, r539;
cvt.rn.f32.u32 f37, r538;
mul.f32 f38, f37, 0f3F490FDB;
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
fma.rn.f16x2 r389, r367, r382, r386;
}
{
mul.f16x2 r393, r367, r384;
}
{
neg.f16x2 r396, r393;
}
{
fma.rn.f16x2 r398, r370, r382, r396;
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
fma.rn.f16x2 r426, r361, r419, r423;
}
{
mul.f16x2 r430, r361, r421;
}
{
neg.f16x2 r433, r430;
}
{
fma.rn.f16x2 r435, r364, r419, r433;
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
fma.rn.f16x2 r463, r373, r456, r460;
}
{
mul.f16x2 r467, r373, r458;
}
{
neg.f16x2 r470, r467;
}
{
fma.rn.f16x2 r472, r376, r456, r470;
}
barrier.sync 0;
and.b32 r541, r523, 256;
add.s32 r542, r540, r541;
st.shared.u32 [r542], r355;
st.shared.u32 [r542+64], r389;
st.shared.u32 [r542+128], r426;
st.shared.u32 [r542+192], r463;
barrier.sync 0;
mad.lo.s32 r543, r537, -12, r542;
ld.shared.u32 r494, [r543];
ld.shared.u32 r506, [r543+128];
ld.shared.u32 r495, [r543+256];
ld.shared.u32 r507, [r543+384];
barrier.sync 0;
st.shared.u32 [r542], r358;
st.shared.u32 [r542+64], r398;
st.shared.u32 [r542+128], r435;
st.shared.u32 [r542+192], r472;
barrier.sync 0;
ld.shared.u32 r497, [r543];
ld.shared.u32 r509, [r543+128];
ld.shared.u32 r498, [r543+256];
ld.shared.u32 r510, [r543+384];
{
add.f16x2 %0, r494, r495;
}
{
add.f16x2 %1, r497, r498;
}
{
sub.f16x2 %4, r494, r495;
}
{
sub.f16x2 %5, r497, r498;
}
{
add.f16x2 %2, r506, r507;
}
{
add.f16x2 %3, r509, r510;
}
{
sub.f16x2 %6, r506, r507;
}
{
sub.f16x2 %7, r509, r510;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1005, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<428>;
.reg .b32 r<3339>;
.reg .b64 rd<3>;
mov.u32 r3263, %tid.y;
shl.b32 r3264, r3263, 10;
mov.u32 r3265, %64;
add.s32 r3266, r3265, r3264;
mov.u32 r3267, %tid.x;
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
mov.f32 f280, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r102, {low, high};
}
mov.f32 f278, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f278;
cvt.rn.f16.f32 high, f278;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
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
add.f16x2 r212, %84, %75;
}
{
sub.f16x2 r215, %113, %103;
}
{
sub.f16x2 r218, %84, %75;
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
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f278;
cvt.rn.f16.f32 high, f278;
mov.b32 r301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
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
mov.f32 f272, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r393, {low, high};
}
mov.f32 f288, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r396, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r398, {low, high};
}
mov.f32 f270, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f270;
cvt.rn.f16.f32 high, f270;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f278;
cvt.rn.f16.f32 high, f278;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r404, {low, high};
}
mov.f32 f286, 0fBF6C835E;
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
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r717, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r718, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f278;
cvt.rn.f16.f32 high, f278;
mov.b32 r721, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
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
add.f16x2 r863, %92, %83;
}
{
add.f16x2 r866, %65, %120;
}
{
sub.f16x2 r869, %92, %83;
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
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r913, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r914, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f278;
cvt.rn.f16.f32 high, f278;
mov.b32 r917, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
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
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
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
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r1011, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r1012, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r1013, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r1014, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f270;
cvt.rn.f16.f32 high, f270;
mov.b32 r1017, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r1018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f278;
cvt.rn.f16.f32 high, f278;
mov.b32 r1019, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
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
mov.f32 f268, 0f3F7B14BE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f268;
cvt.rn.f16.f32 high, f268;
mov.b32 r1233, {low, high};
}
mov.f32 f292, 0f3E47C5C2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f292;
cvt.rn.f16.f32 high, f292;
mov.b32 r1234, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r1235, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r1236, {low, high};
}
mov.f32 f276, 0f3F54DB31;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r1237, {low, high};
}
mov.f32 f284, 0f3F0E39DA;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r1238, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r1239, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r1240, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r1241, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r1242, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r1243, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r1244, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f292;
cvt.rn.f16.f32 high, f292;
mov.b32 r1245, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f268;
cvt.rn.f16.f32 high, f268;
mov.b32 r1246, {low, high};
}
mov.f32 f266, 0fBE47C5C2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f266;
cvt.rn.f16.f32 high, f266;
mov.b32 r1249, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f268;
cvt.rn.f16.f32 high, f268;
mov.b32 r1250, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f270;
cvt.rn.f16.f32 high, f270;
mov.b32 r1251, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r1252, {low, high};
}
mov.f32 f274, 0fBF0E39DA;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f274;
cvt.rn.f16.f32 high, f274;
mov.b32 r1253, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r1254, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f278;
cvt.rn.f16.f32 high, f278;
mov.b32 r1255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r1256, {low, high};
}
mov.f32 f282, 0fBF54DB31;
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
mov.f32 f290, 0fBF7B14BE;
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
and.b32 r3268, r3267, 3;
shl.b32 r3269, r3267, 8;
and.b32 r3270, r3269, -1024;
add.s32 r3271, r3266, r3270;
cvt.rn.f32.u32 f423, r3268;
mul.f32 f424, f423, 0f3D490FDB;
cos.approx.f32 f357, f424;
sin.approx.f32 f425, f424;
neg.f32 f358, f425;
mov.f32 f427, 0fBF800000;
mov.f32 f426, 0f3F800000;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
and.b32 r3272, r3269, 768;
add.s32 r3273, r3271, r3272;
st.shared.v4.f32 [r3273], {r1521, r1524, r1723, r1732};
st.shared.v4.f32 [r3273+16], {r1760, r1769, r1797, r1806};
st.shared.v4.f32 [r3273+32], {r1834, r1843, r1871, r1880};
st.shared.v4.f32 [r3273+48], {r1908, r1917, r1945, r1954};
st.shared.v4.f32 [r3273+64], {r1982, r1991, r2019, r2028};
st.shared.v4.f32 [r3273+80], {r2056, r2065, r2093, r2102};
st.shared.v4.f32 [r3273+96], {r2130, r2139, r2167, r2176};
st.shared.v4.f32 [r3273+112], {r2204, r2213, r2241, r2250};
st.shared.v4.f32 [r3273+128], {r2278, r2287, r2315, r2324};
st.shared.v4.f32 [r3273+144], {r2352, r2361, r2389, r2398};
st.shared.v4.f32 [r3273+160], {r2426, r2435, r2463, r2472};
st.shared.v4.f32 [r3273+176], {r2500, r2509, r2537, r2546};
st.shared.v4.f32 [r3273+192], {r2574, r2583, r2611, r2620};
st.shared.v4.f32 [r3273+208], {r2648, r2657, r2685, r2694};
st.shared.v4.f32 [r3273+224], {r2722, r2731, r2759, r2768};
st.shared.v4.f32 [r3273+240], {r2796, r2805, r2833, r2842};
barrier.sync 0;
mad.lo.s32 r3274, r3268, -248, r3273;
ld.shared.u32 r2864, [r3274];
ld.shared.u32 r2867, [r3274+4];
ld.shared.u32 r2914, [r3274+32];
ld.shared.u32 r2917, [r3274+36];
ld.shared.u32 r2964, [r3274+64];
ld.shared.u32 r2967, [r3274+68];
ld.shared.u32 r3014, [r3274+96];
ld.shared.u32 r3017, [r3274+100];
ld.shared.u32 r3064, [r3274+128];
ld.shared.u32 r3067, [r3274+132];
ld.shared.u32 r3114, [r3274+160];
ld.shared.u32 r3117, [r3274+164];
ld.shared.u32 r3164, [r3274+192];
ld.shared.u32 r3167, [r3274+196];
ld.shared.u32 r3214, [r3274+224];
ld.shared.u32 r3217, [r3274+228];
ld.shared.u32 r2876, [r3274+256];
ld.shared.u32 r2879, [r3274+260];
ld.shared.u32 r2926, [r3274+288];
ld.shared.u32 r2929, [r3274+292];
ld.shared.u32 r2976, [r3274+320];
ld.shared.u32 r2979, [r3274+324];
ld.shared.u32 r3026, [r3274+352];
ld.shared.u32 r3029, [r3274+356];
ld.shared.u32 r3076, [r3274+384];
ld.shared.u32 r3079, [r3274+388];
ld.shared.u32 r3126, [r3274+416];
ld.shared.u32 r3129, [r3274+420];
ld.shared.u32 r3176, [r3274+448];
ld.shared.u32 r3179, [r3274+452];
ld.shared.u32 r3226, [r3274+480];
ld.shared.u32 r3229, [r3274+484];
ld.shared.u32 r2865, [r3274+512];
ld.shared.u32 r2868, [r3274+516];
ld.shared.u32 r2915, [r3274+544];
ld.shared.u32 r2918, [r3274+548];
ld.shared.u32 r2965, [r3274+576];
ld.shared.u32 r2968, [r3274+580];
ld.shared.u32 r3015, [r3274+608];
ld.shared.u32 r3018, [r3274+612];
ld.shared.u32 r3065, [r3274+640];
ld.shared.u32 r3068, [r3274+644];
ld.shared.u32 r3115, [r3274+672];
ld.shared.u32 r3118, [r3274+676];
ld.shared.u32 r3165, [r3274+704];
ld.shared.u32 r3168, [r3274+708];
ld.shared.u32 r3215, [r3274+736];
ld.shared.u32 r3218, [r3274+740];
ld.shared.u32 r2877, [r3274+768];
ld.shared.u32 r2880, [r3274+772];
ld.shared.u32 r2927, [r3274+800];
ld.shared.u32 r2930, [r3274+804];
ld.shared.u32 r2977, [r3274+832];
ld.shared.u32 r2980, [r3274+836];
ld.shared.u32 r3027, [r3274+864];
ld.shared.u32 r3030, [r3274+868];
ld.shared.u32 r3077, [r3274+896];
ld.shared.u32 r3080, [r3274+900];
ld.shared.u32 r3127, [r3274+928];
ld.shared.u32 r3130, [r3274+932];
ld.shared.u32 r3177, [r3274+960];
ld.shared.u32 r3180, [r3274+964];
ld.shared.u32 r3227, [r3274+992];
ld.shared.u32 r3230, [r3274+996];
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
add.f16x2 %0, r2863, r2875;
}
{
add.f16x2 %1, r2866, r2878;
}
{
sub.f16x2 %32, r2863, r2875;
}
{
sub.f16x2 %33, r2866, r2878;
}
{
add.f16x2 %16, r2869, r2887;
}
{
add.f16x2 %17, r2872, r2881;
}
{
sub.f16x2 %48, r2869, r2887;
}
{
sub.f16x2 %49, r2872, r2881;
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
add.f16x2 %2, r2913, r2925;
}
{
add.f16x2 %3, r2916, r2928;
}
{
sub.f16x2 %34, r2913, r2925;
}
{
sub.f16x2 %35, r2916, r2928;
}
{
add.f16x2 %18, r2919, r2937;
}
{
add.f16x2 %19, r2922, r2931;
}
{
sub.f16x2 %50, r2919, r2937;
}
{
sub.f16x2 %51, r2922, r2931;
}
{
add.f16x2 r2963, r2964, r2965;
}
{
add.f16x2 r2966, r2967, r2968;
}
{
sub.f16x2 r2969, r2964, r2965;
}
{
sub.f16x2 r2972, r2967, r2968;
}
{
add.f16x2 r2975, r2976, r2977;
}
{
add.f16x2 r2978, r2979, r2980;
}
{
sub.f16x2 r2981, r2976, r2977;
}
{
sub.f16x2 r2984, r2979, r2980;
}
{
neg.f16x2 r2987, r2984;
}
{
add.f16x2 %4, r2963, r2975;
}
{
add.f16x2 %5, r2966, r2978;
}
{
sub.f16x2 %36, r2963, r2975;
}
{
sub.f16x2 %37, r2966, r2978;
}
{
add.f16x2 %20, r2969, r2987;
}
{
add.f16x2 %21, r2972, r2981;
}
{
sub.f16x2 %52, r2969, r2987;
}
{
sub.f16x2 %53, r2972, r2981;
}
{
add.f16x2 r3013, r3014, r3015;
}
{
add.f16x2 r3016, r3017, r3018;
}
{
sub.f16x2 r3019, r3014, r3015;
}
{
sub.f16x2 r3022, r3017, r3018;
}
{
add.f16x2 r3025, r3026, r3027;
}
{
add.f16x2 r3028, r3029, r3030;
}
{
sub.f16x2 r3031, r3026, r3027;
}
{
sub.f16x2 r3034, r3029, r3030;
}
{
neg.f16x2 r3037, r3034;
}
{
add.f16x2 %6, r3013, r3025;
}
{
add.f16x2 %7, r3016, r3028;
}
{
sub.f16x2 %38, r3013, r3025;
}
{
sub.f16x2 %39, r3016, r3028;
}
{
add.f16x2 %22, r3019, r3037;
}
{
add.f16x2 %23, r3022, r3031;
}
{
sub.f16x2 %54, r3019, r3037;
}
{
sub.f16x2 %55, r3022, r3031;
}
{
add.f16x2 r3063, r3064, r3065;
}
{
add.f16x2 r3066, r3067, r3068;
}
{
sub.f16x2 r3069, r3064, r3065;
}
{
sub.f16x2 r3072, r3067, r3068;
}
{
add.f16x2 r3075, r3076, r3077;
}
{
add.f16x2 r3078, r3079, r3080;
}
{
sub.f16x2 r3081, r3076, r3077;
}
{
sub.f16x2 r3084, r3079, r3080;
}
{
neg.f16x2 r3087, r3084;
}
{
add.f16x2 %8, r3063, r3075;
}
{
add.f16x2 %9, r3066, r3078;
}
{
sub.f16x2 %40, r3063, r3075;
}
{
sub.f16x2 %41, r3066, r3078;
}
{
add.f16x2 %24, r3069, r3087;
}
{
add.f16x2 %25, r3072, r3081;
}
{
sub.f16x2 %56, r3069, r3087;
}
{
sub.f16x2 %57, r3072, r3081;
}
{
add.f16x2 r3113, r3114, r3115;
}
{
add.f16x2 r3116, r3117, r3118;
}
{
sub.f16x2 r3119, r3114, r3115;
}
{
sub.f16x2 r3122, r3117, r3118;
}
{
add.f16x2 r3125, r3126, r3127;
}
{
add.f16x2 r3128, r3129, r3130;
}
{
sub.f16x2 r3131, r3126, r3127;
}
{
sub.f16x2 r3134, r3129, r3130;
}
{
neg.f16x2 r3137, r3134;
}
{
add.f16x2 %10, r3113, r3125;
}
{
add.f16x2 %11, r3116, r3128;
}
{
sub.f16x2 %42, r3113, r3125;
}
{
sub.f16x2 %43, r3116, r3128;
}
{
add.f16x2 %26, r3119, r3137;
}
{
add.f16x2 %27, r3122, r3131;
}
{
sub.f16x2 %58, r3119, r3137;
}
{
sub.f16x2 %59, r3122, r3131;
}
{
add.f16x2 r3163, r3164, r3165;
}
{
add.f16x2 r3166, r3167, r3168;
}
{
sub.f16x2 r3169, r3164, r3165;
}
{
sub.f16x2 r3172, r3167, r3168;
}
{
add.f16x2 r3175, r3176, r3177;
}
{
add.f16x2 r3178, r3179, r3180;
}
{
sub.f16x2 r3181, r3176, r3177;
}
{
sub.f16x2 r3184, r3179, r3180;
}
{
neg.f16x2 r3187, r3184;
}
{
add.f16x2 %12, r3163, r3175;
}
{
add.f16x2 %13, r3166, r3178;
}
{
sub.f16x2 %44, r3163, r3175;
}
{
sub.f16x2 %45, r3166, r3178;
}
{
add.f16x2 %28, r3169, r3187;
}
{
add.f16x2 %29, r3172, r3181;
}
{
sub.f16x2 %60, r3169, r3187;
}
{
sub.f16x2 %61, r3172, r3181;
}
{
add.f16x2 r3213, r3214, r3215;
}
{
add.f16x2 r3216, r3217, r3218;
}
{
sub.f16x2 r3219, r3214, r3215;
}
{
sub.f16x2 r3222, r3217, r3218;
}
{
add.f16x2 r3225, r3226, r3227;
}
{
add.f16x2 r3228, r3229, r3230;
}
{
sub.f16x2 r3231, r3226, r3227;
}
{
sub.f16x2 r3234, r3229, r3230;
}
{
neg.f16x2 r3237, r3234;
}
{
add.f16x2 %14, r3213, r3225;
}
{
add.f16x2 %15, r3216, r3228;
}
{
sub.f16x2 %46, r3213, r3225;
}
{
sub.f16x2 %47, r3216, r3228;
}
{
add.f16x2 %30, r3219, r3237;
}
{
add.f16x2 %31, r3222, r3231;
}
{
sub.f16x2 %62, r3219, r3237;
}
{
sub.f16x2 %63, r3222, r3231;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)), "=r"(__HALF2_TO_UI(rmem[27].x)), "=r"(__HALF2_TO_UI(rmem[27].y)), "=r"(__HALF2_TO_UI(rmem[28].x)), "=r"(__HALF2_TO_UI(rmem[28].y)), "=r"(__HALF2_TO_UI(rmem[29].x)), "=r"(__HALF2_TO_UI(rmem[29].y)), "=r"(__HALF2_TO_UI(rmem[30].x)), "=r"(__HALF2_TO_UI(rmem[30].y)), "=r"(__HALF2_TO_UI(rmem[31].x)), "=r"(__HALF2_TO_UI(rmem[31].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[31].y)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[28].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[27].y)), "r"(__HALF2_TO_UI(rmem[30].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[29].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[31].x)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[28].x)), "r"(__HALF2_TO_UI(rmem[27].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[30].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[29].x)), "r"(__HALF2_TO_UI(rmem[24].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1006, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<428>;
.reg .b32 r<3339>;
.reg .b64 rd<3>;
mov.u32 r3263, %tid.y;
shl.b32 r3264, r3263, 9;
mov.u32 r3265, %64;
add.s32 r3266, r3265, r3264;
mov.u32 r3267, %tid.x;
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
mov.f32 f280, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r102, {low, high};
}
mov.f32 f278, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f278;
cvt.rn.f16.f32 high, f278;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
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
add.f16x2 r212, %84, %75;
}
{
sub.f16x2 r215, %113, %103;
}
{
sub.f16x2 r218, %84, %75;
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
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f278;
cvt.rn.f16.f32 high, f278;
mov.b32 r301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
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
mov.f32 f272, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r393, {low, high};
}
mov.f32 f288, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r396, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r398, {low, high};
}
mov.f32 f270, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f270;
cvt.rn.f16.f32 high, f270;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f278;
cvt.rn.f16.f32 high, f278;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r404, {low, high};
}
mov.f32 f286, 0fBF6C835E;
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
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r717, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r718, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f278;
cvt.rn.f16.f32 high, f278;
mov.b32 r721, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
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
add.f16x2 r863, %92, %83;
}
{
add.f16x2 r866, %65, %120;
}
{
sub.f16x2 r869, %92, %83;
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
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r913, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r914, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f278;
cvt.rn.f16.f32 high, f278;
mov.b32 r917, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
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
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
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
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r1011, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r1012, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r1013, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r1014, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f270;
cvt.rn.f16.f32 high, f270;
mov.b32 r1017, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r1018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f278;
cvt.rn.f16.f32 high, f278;
mov.b32 r1019, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
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
mov.f32 f268, 0f3F7B14BE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f268;
cvt.rn.f16.f32 high, f268;
mov.b32 r1233, {low, high};
}
mov.f32 f292, 0f3E47C5C2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f292;
cvt.rn.f16.f32 high, f292;
mov.b32 r1234, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r1235, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r1236, {low, high};
}
mov.f32 f276, 0f3F54DB31;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r1237, {low, high};
}
mov.f32 f284, 0f3F0E39DA;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r1238, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r1239, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r1240, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r1241, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r1242, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f288;
cvt.rn.f16.f32 high, f288;
mov.b32 r1243, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r1244, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f292;
cvt.rn.f16.f32 high, f292;
mov.b32 r1245, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f268;
cvt.rn.f16.f32 high, f268;
mov.b32 r1246, {low, high};
}
mov.f32 f266, 0fBE47C5C2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f266;
cvt.rn.f16.f32 high, f266;
mov.b32 r1249, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f268;
cvt.rn.f16.f32 high, f268;
mov.b32 r1250, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f270;
cvt.rn.f16.f32 high, f270;
mov.b32 r1251, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r1252, {low, high};
}
mov.f32 f274, 0fBF0E39DA;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f274;
cvt.rn.f16.f32 high, f274;
mov.b32 r1253, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r1254, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f278;
cvt.rn.f16.f32 high, f278;
mov.b32 r1255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r1256, {low, high};
}
mov.f32 f282, 0fBF54DB31;
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
mov.f32 f290, 0fBF7B14BE;
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
and.b32 r3268, r3267, 3;
shl.b32 r3269, r3267, 7;
and.b32 r3270, r3269, -512;
add.s32 r3271, r3266, r3270;
cvt.rn.f32.u32 f423, r3268;
mul.f32 f424, f423, 0f3D490FDB;
cos.approx.f32 f357, f424;
sin.approx.f32 f425, f424;
neg.f32 f358, f425;
mov.f32 f427, 0fBF800000;
mov.f32 f426, 0f3F800000;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
cvt.rn.f16.f32 low, f427;
cvt.rn.f16.f32 high, f426;
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
and.b32 r3272, r3269, 384;
add.s32 r3273, r3271, r3272;
st.shared.v4.f32 [r3273], {r1521, r1723, r1760, r1797};
st.shared.v4.f32 [r3273+16], {r1834, r1871, r1908, r1945};
st.shared.v4.f32 [r3273+32], {r1982, r2019, r2056, r2093};
st.shared.v4.f32 [r3273+48], {r2130, r2167, r2204, r2241};
st.shared.v4.f32 [r3273+64], {r2278, r2315, r2352, r2389};
st.shared.v4.f32 [r3273+80], {r2426, r2463, r2500, r2537};
st.shared.v4.f32 [r3273+96], {r2574, r2611, r2648, r2685};
st.shared.v4.f32 [r3273+112], {r2722, r2759, r2796, r2833};
barrier.sync 0;
mad.lo.s32 r3274, r3268, -124, r3273;
ld.shared.u32 r2864, [r3274];
ld.shared.u32 r2914, [r3274+16];
ld.shared.u32 r2964, [r3274+32];
ld.shared.u32 r3014, [r3274+48];
ld.shared.u32 r3064, [r3274+64];
ld.shared.u32 r3114, [r3274+80];
ld.shared.u32 r3164, [r3274+96];
ld.shared.u32 r3214, [r3274+112];
ld.shared.u32 r2876, [r3274+128];
ld.shared.u32 r2926, [r3274+144];
ld.shared.u32 r2976, [r3274+160];
ld.shared.u32 r3026, [r3274+176];
ld.shared.u32 r3076, [r3274+192];
ld.shared.u32 r3126, [r3274+208];
ld.shared.u32 r3176, [r3274+224];
ld.shared.u32 r3226, [r3274+240];
ld.shared.u32 r2865, [r3274+256];
ld.shared.u32 r2915, [r3274+272];
ld.shared.u32 r2965, [r3274+288];
ld.shared.u32 r3015, [r3274+304];
ld.shared.u32 r3065, [r3274+320];
ld.shared.u32 r3115, [r3274+336];
ld.shared.u32 r3165, [r3274+352];
ld.shared.u32 r3215, [r3274+368];
ld.shared.u32 r2877, [r3274+384];
ld.shared.u32 r2927, [r3274+400];
ld.shared.u32 r2977, [r3274+416];
ld.shared.u32 r3027, [r3274+432];
ld.shared.u32 r3077, [r3274+448];
ld.shared.u32 r3127, [r3274+464];
ld.shared.u32 r3177, [r3274+480];
ld.shared.u32 r3227, [r3274+496];
barrier.sync 0;
st.shared.v4.f32 [r3273], {r1524, r1732, r1769, r1806};
st.shared.v4.f32 [r3273+16], {r1843, r1880, r1917, r1954};
st.shared.v4.f32 [r3273+32], {r1991, r2028, r2065, r2102};
st.shared.v4.f32 [r3273+48], {r2139, r2176, r2213, r2250};
st.shared.v4.f32 [r3273+64], {r2287, r2324, r2361, r2398};
st.shared.v4.f32 [r3273+80], {r2435, r2472, r2509, r2546};
st.shared.v4.f32 [r3273+96], {r2583, r2620, r2657, r2694};
st.shared.v4.f32 [r3273+112], {r2731, r2768, r2805, r2842};
barrier.sync 0;
ld.shared.u32 r2867, [r3274];
ld.shared.u32 r2917, [r3274+16];
ld.shared.u32 r2967, [r3274+32];
ld.shared.u32 r3017, [r3274+48];
ld.shared.u32 r3067, [r3274+64];
ld.shared.u32 r3117, [r3274+80];
ld.shared.u32 r3167, [r3274+96];
ld.shared.u32 r3217, [r3274+112];
ld.shared.u32 r2879, [r3274+128];
ld.shared.u32 r2929, [r3274+144];
ld.shared.u32 r2979, [r3274+160];
ld.shared.u32 r3029, [r3274+176];
ld.shared.u32 r3079, [r3274+192];
ld.shared.u32 r3129, [r3274+208];
ld.shared.u32 r3179, [r3274+224];
ld.shared.u32 r3229, [r3274+240];
ld.shared.u32 r2868, [r3274+256];
ld.shared.u32 r2918, [r3274+272];
ld.shared.u32 r2968, [r3274+288];
ld.shared.u32 r3018, [r3274+304];
ld.shared.u32 r3068, [r3274+320];
ld.shared.u32 r3118, [r3274+336];
ld.shared.u32 r3168, [r3274+352];
ld.shared.u32 r3218, [r3274+368];
ld.shared.u32 r2880, [r3274+384];
ld.shared.u32 r2930, [r3274+400];
ld.shared.u32 r2980, [r3274+416];
ld.shared.u32 r3030, [r3274+432];
ld.shared.u32 r3080, [r3274+448];
ld.shared.u32 r3130, [r3274+464];
ld.shared.u32 r3180, [r3274+480];
ld.shared.u32 r3230, [r3274+496];
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
add.f16x2 %0, r2863, r2875;
}
{
add.f16x2 %1, r2866, r2878;
}
{
sub.f16x2 %32, r2863, r2875;
}
{
sub.f16x2 %33, r2866, r2878;
}
{
add.f16x2 %16, r2869, r2887;
}
{
add.f16x2 %17, r2872, r2881;
}
{
sub.f16x2 %48, r2869, r2887;
}
{
sub.f16x2 %49, r2872, r2881;
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
add.f16x2 %2, r2913, r2925;
}
{
add.f16x2 %3, r2916, r2928;
}
{
sub.f16x2 %34, r2913, r2925;
}
{
sub.f16x2 %35, r2916, r2928;
}
{
add.f16x2 %18, r2919, r2937;
}
{
add.f16x2 %19, r2922, r2931;
}
{
sub.f16x2 %50, r2919, r2937;
}
{
sub.f16x2 %51, r2922, r2931;
}
{
add.f16x2 r2963, r2964, r2965;
}
{
add.f16x2 r2966, r2967, r2968;
}
{
sub.f16x2 r2969, r2964, r2965;
}
{
sub.f16x2 r2972, r2967, r2968;
}
{
add.f16x2 r2975, r2976, r2977;
}
{
add.f16x2 r2978, r2979, r2980;
}
{
sub.f16x2 r2981, r2976, r2977;
}
{
sub.f16x2 r2984, r2979, r2980;
}
{
neg.f16x2 r2987, r2984;
}
{
add.f16x2 %4, r2963, r2975;
}
{
add.f16x2 %5, r2966, r2978;
}
{
sub.f16x2 %36, r2963, r2975;
}
{
sub.f16x2 %37, r2966, r2978;
}
{
add.f16x2 %20, r2969, r2987;
}
{
add.f16x2 %21, r2972, r2981;
}
{
sub.f16x2 %52, r2969, r2987;
}
{
sub.f16x2 %53, r2972, r2981;
}
{
add.f16x2 r3013, r3014, r3015;
}
{
add.f16x2 r3016, r3017, r3018;
}
{
sub.f16x2 r3019, r3014, r3015;
}
{
sub.f16x2 r3022, r3017, r3018;
}
{
add.f16x2 r3025, r3026, r3027;
}
{
add.f16x2 r3028, r3029, r3030;
}
{
sub.f16x2 r3031, r3026, r3027;
}
{
sub.f16x2 r3034, r3029, r3030;
}
{
neg.f16x2 r3037, r3034;
}
{
add.f16x2 %6, r3013, r3025;
}
{
add.f16x2 %7, r3016, r3028;
}
{
sub.f16x2 %38, r3013, r3025;
}
{
sub.f16x2 %39, r3016, r3028;
}
{
add.f16x2 %22, r3019, r3037;
}
{
add.f16x2 %23, r3022, r3031;
}
{
sub.f16x2 %54, r3019, r3037;
}
{
sub.f16x2 %55, r3022, r3031;
}
{
add.f16x2 r3063, r3064, r3065;
}
{
add.f16x2 r3066, r3067, r3068;
}
{
sub.f16x2 r3069, r3064, r3065;
}
{
sub.f16x2 r3072, r3067, r3068;
}
{
add.f16x2 r3075, r3076, r3077;
}
{
add.f16x2 r3078, r3079, r3080;
}
{
sub.f16x2 r3081, r3076, r3077;
}
{
sub.f16x2 r3084, r3079, r3080;
}
{
neg.f16x2 r3087, r3084;
}
{
add.f16x2 %8, r3063, r3075;
}
{
add.f16x2 %9, r3066, r3078;
}
{
sub.f16x2 %40, r3063, r3075;
}
{
sub.f16x2 %41, r3066, r3078;
}
{
add.f16x2 %24, r3069, r3087;
}
{
add.f16x2 %25, r3072, r3081;
}
{
sub.f16x2 %56, r3069, r3087;
}
{
sub.f16x2 %57, r3072, r3081;
}
{
add.f16x2 r3113, r3114, r3115;
}
{
add.f16x2 r3116, r3117, r3118;
}
{
sub.f16x2 r3119, r3114, r3115;
}
{
sub.f16x2 r3122, r3117, r3118;
}
{
add.f16x2 r3125, r3126, r3127;
}
{
add.f16x2 r3128, r3129, r3130;
}
{
sub.f16x2 r3131, r3126, r3127;
}
{
sub.f16x2 r3134, r3129, r3130;
}
{
neg.f16x2 r3137, r3134;
}
{
add.f16x2 %10, r3113, r3125;
}
{
add.f16x2 %11, r3116, r3128;
}
{
sub.f16x2 %42, r3113, r3125;
}
{
sub.f16x2 %43, r3116, r3128;
}
{
add.f16x2 %26, r3119, r3137;
}
{
add.f16x2 %27, r3122, r3131;
}
{
sub.f16x2 %58, r3119, r3137;
}
{
sub.f16x2 %59, r3122, r3131;
}
{
add.f16x2 r3163, r3164, r3165;
}
{
add.f16x2 r3166, r3167, r3168;
}
{
sub.f16x2 r3169, r3164, r3165;
}
{
sub.f16x2 r3172, r3167, r3168;
}
{
add.f16x2 r3175, r3176, r3177;
}
{
add.f16x2 r3178, r3179, r3180;
}
{
sub.f16x2 r3181, r3176, r3177;
}
{
sub.f16x2 r3184, r3179, r3180;
}
{
neg.f16x2 r3187, r3184;
}
{
add.f16x2 %12, r3163, r3175;
}
{
add.f16x2 %13, r3166, r3178;
}
{
sub.f16x2 %44, r3163, r3175;
}
{
sub.f16x2 %45, r3166, r3178;
}
{
add.f16x2 %28, r3169, r3187;
}
{
add.f16x2 %29, r3172, r3181;
}
{
sub.f16x2 %60, r3169, r3187;
}
{
sub.f16x2 %61, r3172, r3181;
}
{
add.f16x2 r3213, r3214, r3215;
}
{
add.f16x2 r3216, r3217, r3218;
}
{
sub.f16x2 r3219, r3214, r3215;
}
{
sub.f16x2 r3222, r3217, r3218;
}
{
add.f16x2 r3225, r3226, r3227;
}
{
add.f16x2 r3228, r3229, r3230;
}
{
sub.f16x2 r3231, r3226, r3227;
}
{
sub.f16x2 r3234, r3229, r3230;
}
{
neg.f16x2 r3237, r3234;
}
{
add.f16x2 %14, r3213, r3225;
}
{
add.f16x2 %15, r3216, r3228;
}
{
sub.f16x2 %46, r3213, r3225;
}
{
sub.f16x2 %47, r3216, r3228;
}
{
add.f16x2 %30, r3219, r3237;
}
{
add.f16x2 %31, r3222, r3231;
}
{
sub.f16x2 %62, r3219, r3237;
}
{
sub.f16x2 %63, r3222, r3231;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)), "=r"(__HALF2_TO_UI(rmem[27].x)), "=r"(__HALF2_TO_UI(rmem[27].y)), "=r"(__HALF2_TO_UI(rmem[28].x)), "=r"(__HALF2_TO_UI(rmem[28].y)), "=r"(__HALF2_TO_UI(rmem[29].x)), "=r"(__HALF2_TO_UI(rmem[29].y)), "=r"(__HALF2_TO_UI(rmem[30].x)), "=r"(__HALF2_TO_UI(rmem[30].y)), "=r"(__HALF2_TO_UI(rmem[31].x)), "=r"(__HALF2_TO_UI(rmem[31].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[31].y)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[28].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[27].y)), "r"(__HALF2_TO_UI(rmem[30].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[29].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[31].x)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[28].x)), "r"(__HALF2_TO_UI(rmem[27].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[30].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[29].x)), "r"(__HALF2_TO_UI(rmem[24].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1007, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<55>;
.reg .b32 r<374>;
.reg .b64 rd<2>;
mov.u32 r325, %tid.y;
shl.b32 r326, r325, 10;
mov.u32 r327, %4;
add.s32 r328, r327, r326;
mov.u32 r329, %tid.x;
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
and.b32 r330, r329, 63;
shl.b32 r331, r329, 4;
and.b32 r332, r331, -1024;
add.s32 r333, r328, r332;
cvt.rn.f32.u32 f37, r330;
mul.f32 f38, f37, 0f3D490FDB;
cos.approx.f32 f1, f38;
sin.approx.f32 f39, f38;
neg.f32 f2, f39;
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
fma.rn.f16x2 r23, r7, r16, r20;
}
{
mul.f16x2 r27, r7, r18;
}
{
neg.f16x2 r30, r27;
}
{
fma.rn.f16x2 r32, r10, r16, r30;
}
barrier.sync 0;
and.b32 r334, r331, 1008;
add.s32 r335, r333, r334;
st.shared.v2.f32 [r335], {r1, r4};
st.shared.v2.f32 [r335+8], {r23, r32};
barrier.sync 0;
shl.b32 r336, r329, 3;
and.b32 r337, r336, 504;
sub.s32 r338, r335, r337;
ld.shared.u32 r54, [r338];
ld.shared.u32 r57, [r338+4];
ld.shared.u32 r55, [r338+512];
ld.shared.u32 r58, [r338+516];
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
bfe.u32 r339, r329, 1, 5;
cvt.rn.f32.u32 f40, r339;
mul.f32 f41, f40, 0f3DC90FDB;
cos.approx.f32 f7, f41;
sin.approx.f32 f42, f41;
neg.f32 f8, f42;
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
fma.rn.f16x2 r75, r59, r68, r72;
}
{
mul.f16x2 r79, r59, r70;
}
{
neg.f16x2 r82, r79;
}
{
fma.rn.f16x2 r84, r62, r68, r82;
}
and.b32 r340, r336, 8;
add.s32 r341, r333, r340;
barrier.sync 0;
and.b32 r342, r331, 992;
add.s32 r343, r341, r342;
st.shared.u32 [r343], r53;
st.shared.u32 [r343+4], r56;
st.shared.u32 [r343+16], r75;
st.shared.u32 [r343+20], r84;
barrier.sync 0;
and.b32 r344, r336, 496;
sub.s32 r345, r343, r344;
ld.shared.u32 r106, [r345];
ld.shared.u32 r109, [r345+4];
ld.shared.u32 r107, [r345+512];
ld.shared.u32 r110, [r345+516];
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
bfe.u32 r346, r329, 2, 4;
cvt.rn.f32.u32 f43, r346;
mul.f32 f44, f43, 0f3E490FDB;
cos.approx.f32 f13, f44;
sin.approx.f32 f45, f44;
neg.f32 f14, f45;
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
fma.rn.f16x2 r127, r111, r120, r124;
}
{
mul.f16x2 r131, r111, r122;
}
{
neg.f16x2 r134, r131;
}
{
fma.rn.f16x2 r136, r114, r120, r134;
}
and.b32 r347, r336, 24;
add.s32 r348, r333, r347;
barrier.sync 0;
and.b32 r349, r331, 960;
add.s32 r350, r348, r349;
st.shared.u32 [r350], r105;
st.shared.u32 [r350+4], r108;
st.shared.u32 [r350+32], r127;
st.shared.u32 [r350+36], r136;
barrier.sync 0;
and.b32 r351, r336, 480;
sub.s32 r352, r350, r351;
ld.shared.u32 r158, [r352];
ld.shared.u32 r161, [r352+4];
ld.shared.u32 r159, [r352+512];
ld.shared.u32 r162, [r352+516];
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
bfe.u32 r353, r329, 3, 3;
cvt.rn.f32.u32 f46, r353;
mul.f32 f47, f46, 0f3EC90FDB;
cos.approx.f32 f19, f47;
sin.approx.f32 f48, f47;
neg.f32 f20, f48;
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
fma.rn.f16x2 r179, r163, r172, r176;
}
{
mul.f16x2 r183, r163, r174;
}
{
neg.f16x2 r186, r183;
}
{
fma.rn.f16x2 r188, r166, r172, r186;
}
and.b32 r354, r336, 56;
add.s32 r355, r333, r354;
barrier.sync 0;
and.b32 r356, r331, 896;
add.s32 r357, r355, r356;
st.shared.u32 [r357], r157;
st.shared.u32 [r357+4], r160;
st.shared.u32 [r357+64], r179;
st.shared.u32 [r357+68], r188;
barrier.sync 0;
and.b32 r358, r336, 448;
sub.s32 r359, r357, r358;
ld.shared.u32 r210, [r359];
ld.shared.u32 r213, [r359+4];
ld.shared.u32 r211, [r359+512];
ld.shared.u32 r214, [r359+516];
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
bfe.u32 r360, r329, 4, 2;
cvt.rn.f32.u32 f49, r360;
mul.f32 f50, f49, 0f3F490FDB;
cos.approx.f32 f25, f50;
sin.approx.f32 f51, f50;
neg.f32 f26, f51;
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
fma.rn.f16x2 r231, r215, r224, r228;
}
{
mul.f16x2 r235, r215, r226;
}
{
neg.f16x2 r238, r235;
}
{
fma.rn.f16x2 r240, r218, r224, r238;
}
and.b32 r361, r336, 120;
add.s32 r362, r333, r361;
barrier.sync 0;
and.b32 r363, r331, 768;
add.s32 r364, r362, r363;
st.shared.u32 [r364], r209;
st.shared.u32 [r364+4], r212;
st.shared.u32 [r364+128], r231;
st.shared.u32 [r364+132], r240;
barrier.sync 0;
and.b32 r365, r336, 384;
sub.s32 r366, r364, r365;
ld.shared.u32 r262, [r366];
ld.shared.u32 r265, [r366+4];
ld.shared.u32 r263, [r366+512];
ld.shared.u32 r266, [r366+516];
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
bfe.u32 r367, r329, 5, 1;
cvt.rn.f32.u32 f52, r367;
mul.f32 f53, f52, 0f3FC90FDB;
cos.approx.f32 f31, f53;
sin.approx.f32 f54, f53;
neg.f32 f32, f54;
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
fma.rn.f16x2 r283, r267, r276, r280;
}
{
mul.f16x2 r287, r267, r278;
}
{
neg.f16x2 r290, r287;
}
{
fma.rn.f16x2 r292, r270, r276, r290;
}
and.b32 r368, r336, 248;
add.s32 r369, r333, r368;
barrier.sync 0;
and.b32 r370, r331, 512;
add.s32 r371, r369, r370;
st.shared.u32 [r371], r261;
st.shared.u32 [r371+4], r264;
st.shared.u32 [r371+256], r283;
st.shared.u32 [r371+260], r292;
barrier.sync 0;
and.b32 r372, r336, 256;
sub.s32 r373, r371, r372;
ld.shared.u32 r314, [r373];
ld.shared.u32 r317, [r373+4];
ld.shared.u32 r315, [r373+512];
ld.shared.u32 r318, [r373+516];
{
add.f16x2 %0, r314, r315;
}
{
add.f16x2 %1, r317, r318;
}
{
sub.f16x2 %2, r314, r315;
}
{
sub.f16x2 %3, r317, r318;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1008, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<55>;
.reg .b32 r<374>;
.reg .b64 rd<2>;
mov.u32 r325, %tid.y;
shl.b32 r326, r325, 9;
mov.u32 r327, %4;
add.s32 r328, r327, r326;
mov.u32 r329, %tid.x;
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
and.b32 r330, r329, 63;
shl.b32 r331, r329, 3;
and.b32 r332, r331, -512;
add.s32 r333, r328, r332;
cvt.rn.f32.u32 f37, r330;
mul.f32 f38, f37, 0f3D490FDB;
cos.approx.f32 f1, f38;
sin.approx.f32 f39, f38;
neg.f32 f2, f39;
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
fma.rn.f16x2 r23, r7, r16, r20;
}
{
mul.f16x2 r27, r7, r18;
}
{
neg.f16x2 r30, r27;
}
{
fma.rn.f16x2 r32, r10, r16, r30;
}
barrier.sync 0;
and.b32 r334, r331, 504;
add.s32 r335, r333, r334;
st.shared.v2.f32 [r335], {r1, r23};
barrier.sync 0;
shl.b32 r336, r329, 2;
and.b32 r337, r336, 252;
sub.s32 r338, r335, r337;
ld.shared.u32 r54, [r338];
ld.shared.u32 r55, [r338+256];
barrier.sync 0;
st.shared.v2.f32 [r335], {r4, r32};
barrier.sync 0;
ld.shared.u32 r57, [r338];
ld.shared.u32 r58, [r338+256];
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
bfe.u32 r339, r329, 1, 5;
and.b32 r340, r336, 4;
add.s32 r341, r333, r340;
cvt.rn.f32.u32 f40, r339;
mul.f32 f41, f40, 0f3DC90FDB;
cos.approx.f32 f7, f41;
sin.approx.f32 f42, f41;
neg.f32 f8, f42;
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
fma.rn.f16x2 r75, r59, r68, r72;
}
{
mul.f16x2 r79, r59, r70;
}
{
neg.f16x2 r82, r79;
}
{
fma.rn.f16x2 r84, r62, r68, r82;
}
barrier.sync 0;
and.b32 r342, r331, 496;
add.s32 r343, r341, r342;
st.shared.u32 [r343], r53;
st.shared.u32 [r343+8], r75;
barrier.sync 0;
and.b32 r344, r336, 248;
sub.s32 r345, r343, r344;
ld.shared.u32 r106, [r345];
ld.shared.u32 r107, [r345+256];
barrier.sync 0;
st.shared.u32 [r343], r56;
st.shared.u32 [r343+8], r84;
barrier.sync 0;
ld.shared.u32 r109, [r345];
ld.shared.u32 r110, [r345+256];
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
bfe.u32 r346, r329, 2, 4;
and.b32 r347, r336, 12;
add.s32 r348, r333, r347;
cvt.rn.f32.u32 f43, r346;
mul.f32 f44, f43, 0f3E490FDB;
cos.approx.f32 f13, f44;
sin.approx.f32 f45, f44;
neg.f32 f14, f45;
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
fma.rn.f16x2 r127, r111, r120, r124;
}
{
mul.f16x2 r131, r111, r122;
}
{
neg.f16x2 r134, r131;
}
{
fma.rn.f16x2 r136, r114, r120, r134;
}
barrier.sync 0;
and.b32 r349, r331, 480;
add.s32 r350, r348, r349;
st.shared.u32 [r350], r105;
st.shared.u32 [r350+16], r127;
barrier.sync 0;
and.b32 r351, r336, 240;
sub.s32 r352, r350, r351;
ld.shared.u32 r158, [r352];
ld.shared.u32 r159, [r352+256];
barrier.sync 0;
st.shared.u32 [r350], r108;
st.shared.u32 [r350+16], r136;
barrier.sync 0;
ld.shared.u32 r161, [r352];
ld.shared.u32 r162, [r352+256];
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
bfe.u32 r353, r329, 3, 3;
and.b32 r354, r336, 28;
add.s32 r355, r333, r354;
cvt.rn.f32.u32 f46, r353;
mul.f32 f47, f46, 0f3EC90FDB;
cos.approx.f32 f19, f47;
sin.approx.f32 f48, f47;
neg.f32 f20, f48;
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
fma.rn.f16x2 r179, r163, r172, r176;
}
{
mul.f16x2 r183, r163, r174;
}
{
neg.f16x2 r186, r183;
}
{
fma.rn.f16x2 r188, r166, r172, r186;
}
barrier.sync 0;
and.b32 r356, r331, 448;
add.s32 r357, r355, r356;
st.shared.u32 [r357], r157;
st.shared.u32 [r357+32], r179;
barrier.sync 0;
and.b32 r358, r336, 224;
sub.s32 r359, r357, r358;
ld.shared.u32 r210, [r359];
ld.shared.u32 r211, [r359+256];
barrier.sync 0;
st.shared.u32 [r357], r160;
st.shared.u32 [r357+32], r188;
barrier.sync 0;
ld.shared.u32 r213, [r359];
ld.shared.u32 r214, [r359+256];
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
bfe.u32 r360, r329, 4, 2;
and.b32 r361, r336, 60;
add.s32 r362, r333, r361;
cvt.rn.f32.u32 f49, r360;
mul.f32 f50, f49, 0f3F490FDB;
cos.approx.f32 f25, f50;
sin.approx.f32 f51, f50;
neg.f32 f26, f51;
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
fma.rn.f16x2 r231, r215, r224, r228;
}
{
mul.f16x2 r235, r215, r226;
}
{
neg.f16x2 r238, r235;
}
{
fma.rn.f16x2 r240, r218, r224, r238;
}
barrier.sync 0;
and.b32 r363, r331, 384;
add.s32 r364, r362, r363;
st.shared.u32 [r364], r209;
st.shared.u32 [r364+64], r231;
barrier.sync 0;
and.b32 r365, r336, 192;
sub.s32 r366, r364, r365;
ld.shared.u32 r262, [r366];
ld.shared.u32 r263, [r366+256];
barrier.sync 0;
st.shared.u32 [r364], r212;
st.shared.u32 [r364+64], r240;
barrier.sync 0;
ld.shared.u32 r265, [r366];
ld.shared.u32 r266, [r366+256];
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
bfe.u32 r367, r329, 5, 1;
and.b32 r368, r336, 124;
add.s32 r369, r333, r368;
cvt.rn.f32.u32 f52, r367;
mul.f32 f53, f52, 0f3FC90FDB;
cos.approx.f32 f31, f53;
sin.approx.f32 f54, f53;
neg.f32 f32, f54;
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
fma.rn.f16x2 r283, r267, r276, r280;
}
{
mul.f16x2 r287, r267, r278;
}
{
neg.f16x2 r290, r287;
}
{
fma.rn.f16x2 r292, r270, r276, r290;
}
barrier.sync 0;
and.b32 r370, r331, 256;
add.s32 r371, r369, r370;
st.shared.u32 [r371], r261;
st.shared.u32 [r371+128], r283;
barrier.sync 0;
and.b32 r372, r336, 128;
sub.s32 r373, r371, r372;
ld.shared.u32 r314, [r373];
ld.shared.u32 r315, [r373+256];
barrier.sync 0;
st.shared.u32 [r371], r264;
st.shared.u32 [r371+128], r292;
barrier.sync 0;
ld.shared.u32 r317, [r373];
ld.shared.u32 r318, [r373+256];
{
add.f16x2 %0, r314, r315;
}
{
add.f16x2 %1, r317, r318;
}
{
sub.f16x2 %2, r314, r315;
}
{
sub.f16x2 %3, r317, r318;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)));
};


#endif
