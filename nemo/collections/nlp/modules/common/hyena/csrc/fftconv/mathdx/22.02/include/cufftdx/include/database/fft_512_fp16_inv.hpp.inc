#ifndef CUFFTDX_FFT_512_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_512_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1019, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<127>;
.reg .b32 r<1133>;
.reg .b64 rd<2>;
mov.u32 r1113, %tid.y;
shl.b32 r1114, r1113, 12;
mov.u32 r1115, %16;
add.s32 r1116, r1115, r1114;
mov.u32 r1117, %tid.x;
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
mov.f32 f104, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r102, {low, high};
}
mov.f32 f90, 0f3F800000;
mov.f32 f102, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
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
and.b32 r1118, r1117, 63;
shl.b32 r1119, r1117, 6;
and.b32 r1120, r1119, -4096;
add.s32 r1121, r1116, r1120;
cvt.rn.f32.u32 f121, r1118;
mul.f32 f122, f121, 0f3C490FDB;
cos.approx.f32 f29, f122;
sin.approx.f32 f123, f122;
neg.f32 f30, f123;
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
and.b32 r1122, r1119, 4032;
add.s32 r1123, r1121, r1122;
st.shared.v4.f32 [r1123], {r149, r152, r207, r216};
st.shared.v4.f32 [r1123+16], {r244, r253, r281, r290};
st.shared.v4.f32 [r1123+32], {r318, r327, r355, r364};
st.shared.v4.f32 [r1123+48], {r392, r401, r429, r438};
barrier.sync 0;
mad.lo.s32 r1124, r1118, -56, r1123;
ld.shared.u32 r460, [r1124];
ld.shared.u32 r463, [r1124+4];
ld.shared.u32 r510, [r1124+512];
ld.shared.u32 r513, [r1124+516];
ld.shared.u32 r472, [r1124+1024];
ld.shared.u32 r475, [r1124+1028];
ld.shared.u32 r522, [r1124+1536];
ld.shared.u32 r525, [r1124+1540];
ld.shared.u32 r461, [r1124+2048];
ld.shared.u32 r464, [r1124+2052];
ld.shared.u32 r511, [r1124+2560];
ld.shared.u32 r514, [r1124+2564];
ld.shared.u32 r473, [r1124+3072];
ld.shared.u32 r476, [r1124+3076];
ld.shared.u32 r523, [r1124+3584];
ld.shared.u32 r526, [r1124+3588];
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
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
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
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
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
and.b32 r1125, r1117, 56;
bfe.u32 r1126, r1117, 3, 3;
cvt.rn.f32.u32 f124, r1126;
mul.f32 f125, f124, 0f3DC90FDB;
cos.approx.f32 f75, f125;
sin.approx.f32 f126, f125;
neg.f32 f76, f126;
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
shl.b32 r1127, r1117, 3;
and.b32 r1128, r1127, 56;
add.s32 r1129, r1121, r1128;
barrier.sync 0;
and.b32 r1130, r1119, 3584;
add.s32 r1131, r1129, r1130;
st.shared.u32 [r1131], r607;
st.shared.u32 [r1131+4], r610;
st.shared.u32 [r1131+64], r665;
st.shared.u32 [r1131+68], r674;
st.shared.u32 [r1131+128], r702;
st.shared.u32 [r1131+132], r711;
st.shared.u32 [r1131+192], r739;
st.shared.u32 [r1131+196], r748;
st.shared.u32 [r1131+256], r776;
st.shared.u32 [r1131+260], r785;
st.shared.u32 [r1131+320], r813;
st.shared.u32 [r1131+324], r822;
st.shared.u32 [r1131+384], r850;
st.shared.u32 [r1131+388], r859;
st.shared.u32 [r1131+448], r887;
st.shared.u32 [r1131+452], r896;
barrier.sync 0;
mad.lo.s32 r1132, r1125, -56, r1131;
ld.shared.u32 r918, [r1132];
ld.shared.u32 r921, [r1132+4];
ld.shared.u32 r968, [r1132+512];
ld.shared.u32 r971, [r1132+516];
ld.shared.u32 r930, [r1132+1024];
ld.shared.u32 r933, [r1132+1028];
ld.shared.u32 r980, [r1132+1536];
ld.shared.u32 r983, [r1132+1540];
ld.shared.u32 r919, [r1132+2048];
ld.shared.u32 r922, [r1132+2052];
ld.shared.u32 r969, [r1132+2560];
ld.shared.u32 r972, [r1132+2564];
ld.shared.u32 r931, [r1132+3072];
ld.shared.u32 r934, [r1132+3076];
ld.shared.u32 r981, [r1132+3584];
ld.shared.u32 r984, [r1132+3588];
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
neg.f16x2 r941, r938;
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
add.f16x2 r955, r923, r941;
}
{
add.f16x2 r958, r926, r935;
}
{
sub.f16x2 r961, r923, r941;
}
{
sub.f16x2 r964, r926, r935;
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
neg.f16x2 r991, r988;
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
add.f16x2 r1005, r973, r991;
}
{
add.f16x2 r1008, r976, r985;
}
{
sub.f16x2 r1011, r973, r991;
}
{
sub.f16x2 r1014, r976, r985;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
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
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
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
neg.f16x2 r1047, r1002;
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
add.f16x2 %0, r943, r993;
}
{
add.f16x2 %1, r946, r996;
}
{
sub.f16x2 %8, r943, r993;
}
{
sub.f16x2 %9, r946, r996;
}
{
add.f16x2 %2, r955, r1037;
}
{
add.f16x2 %3, r958, r1043;
}
{
sub.f16x2 %10, r955, r1037;
}
{
sub.f16x2 %11, r958, r1043;
}
{
add.f16x2 %4, r949, r1047;
}
{
add.f16x2 %5, r952, r999;
}
{
sub.f16x2 %12, r949, r1047;
}
{
sub.f16x2 %13, r952, r999;
}
{
add.f16x2 %6, r961, r1055;
}
{
add.f16x2 %7, r964, r1061;
}
{
sub.f16x2 %14, r961, r1055;
}
{
sub.f16x2 %15, r964, r1061;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1020, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<127>;
.reg .b32 r<1133>;
.reg .b64 rd<2>;
mov.u32 r1113, %tid.y;
shl.b32 r1114, r1113, 11;
mov.u32 r1115, %16;
add.s32 r1116, r1115, r1114;
mov.u32 r1117, %tid.x;
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
mov.f32 f104, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r102, {low, high};
}
mov.f32 f90, 0f3F800000;
mov.f32 f102, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
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
and.b32 r1118, r1117, 63;
shl.b32 r1119, r1117, 5;
and.b32 r1120, r1119, -2048;
add.s32 r1121, r1116, r1120;
cvt.rn.f32.u32 f121, r1118;
mul.f32 f122, f121, 0f3C490FDB;
cos.approx.f32 f29, f122;
sin.approx.f32 f123, f122;
neg.f32 f30, f123;
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
and.b32 r1122, r1119, 2016;
add.s32 r1123, r1121, r1122;
st.shared.v4.f32 [r1123], {r149, r207, r244, r281};
st.shared.v4.f32 [r1123+16], {r318, r355, r392, r429};
barrier.sync 0;
mad.lo.s32 r1124, r1118, -28, r1123;
ld.shared.u32 r460, [r1124];
ld.shared.u32 r510, [r1124+256];
ld.shared.u32 r472, [r1124+512];
ld.shared.u32 r522, [r1124+768];
ld.shared.u32 r461, [r1124+1024];
ld.shared.u32 r511, [r1124+1280];
ld.shared.u32 r473, [r1124+1536];
ld.shared.u32 r523, [r1124+1792];
barrier.sync 0;
st.shared.v4.f32 [r1123], {r152, r216, r253, r290};
st.shared.v4.f32 [r1123+16], {r327, r364, r401, r438};
barrier.sync 0;
ld.shared.u32 r463, [r1124];
ld.shared.u32 r513, [r1124+256];
ld.shared.u32 r475, [r1124+512];
ld.shared.u32 r525, [r1124+768];
ld.shared.u32 r464, [r1124+1024];
ld.shared.u32 r514, [r1124+1280];
ld.shared.u32 r476, [r1124+1536];
ld.shared.u32 r526, [r1124+1792];
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
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
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
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
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
and.b32 r1125, r1117, 56;
bfe.u32 r1126, r1117, 3, 3;
shl.b32 r1127, r1117, 2;
and.b32 r1128, r1127, 28;
add.s32 r1129, r1121, r1128;
cvt.rn.f32.u32 f124, r1126;
mul.f32 f125, f124, 0f3DC90FDB;
cos.approx.f32 f75, f125;
sin.approx.f32 f126, f125;
neg.f32 f76, f126;
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
and.b32 r1130, r1119, 1792;
add.s32 r1131, r1129, r1130;
st.shared.u32 [r1131], r607;
st.shared.u32 [r1131+32], r665;
st.shared.u32 [r1131+64], r702;
st.shared.u32 [r1131+96], r739;
st.shared.u32 [r1131+128], r776;
st.shared.u32 [r1131+160], r813;
st.shared.u32 [r1131+192], r850;
st.shared.u32 [r1131+224], r887;
barrier.sync 0;
mad.lo.s32 r1132, r1125, -28, r1131;
ld.shared.u32 r918, [r1132];
ld.shared.u32 r968, [r1132+256];
ld.shared.u32 r930, [r1132+512];
ld.shared.u32 r980, [r1132+768];
ld.shared.u32 r919, [r1132+1024];
ld.shared.u32 r969, [r1132+1280];
ld.shared.u32 r931, [r1132+1536];
ld.shared.u32 r981, [r1132+1792];
barrier.sync 0;
st.shared.u32 [r1131], r610;
st.shared.u32 [r1131+32], r674;
st.shared.u32 [r1131+64], r711;
st.shared.u32 [r1131+96], r748;
st.shared.u32 [r1131+128], r785;
st.shared.u32 [r1131+160], r822;
st.shared.u32 [r1131+192], r859;
st.shared.u32 [r1131+224], r896;
barrier.sync 0;
ld.shared.u32 r921, [r1132];
ld.shared.u32 r971, [r1132+256];
ld.shared.u32 r933, [r1132+512];
ld.shared.u32 r983, [r1132+768];
ld.shared.u32 r922, [r1132+1024];
ld.shared.u32 r972, [r1132+1280];
ld.shared.u32 r934, [r1132+1536];
ld.shared.u32 r984, [r1132+1792];
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
neg.f16x2 r941, r938;
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
add.f16x2 r955, r923, r941;
}
{
add.f16x2 r958, r926, r935;
}
{
sub.f16x2 r961, r923, r941;
}
{
sub.f16x2 r964, r926, r935;
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
neg.f16x2 r991, r988;
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
add.f16x2 r1005, r973, r991;
}
{
add.f16x2 r1008, r976, r985;
}
{
sub.f16x2 r1011, r973, r991;
}
{
sub.f16x2 r1014, r976, r985;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
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
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
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
neg.f16x2 r1047, r1002;
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
add.f16x2 %0, r943, r993;
}
{
add.f16x2 %1, r946, r996;
}
{
sub.f16x2 %8, r943, r993;
}
{
sub.f16x2 %9, r946, r996;
}
{
add.f16x2 %2, r955, r1037;
}
{
add.f16x2 %3, r958, r1043;
}
{
sub.f16x2 %10, r955, r1037;
}
{
sub.f16x2 %11, r958, r1043;
}
{
add.f16x2 %4, r949, r1047;
}
{
add.f16x2 %5, r952, r999;
}
{
sub.f16x2 %12, r949, r1047;
}
{
sub.f16x2 %13, r952, r999;
}
{
add.f16x2 %6, r961, r1055;
}
{
add.f16x2 %7, r964, r1061;
}
{
sub.f16x2 %14, r961, r1055;
}
{
sub.f16x2 %15, r964, r1061;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1022, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<660>;
.reg .b32 r<4171>;
.reg .b64 rd<3>;
mov.u32 r4095, %tid.y;
shl.b32 r4096, r4095, 12;
mov.u32 r4097, %64;
add.s32 r4098, r4097, r4096;
mov.u32 r4099, %tid.x;
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
mov.f32 f618, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r102, {low, high};
}
mov.f32 f616, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
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
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
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
mov.f32 f614, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r393, {low, high};
}
mov.f32 f622, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r396, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
mov.b32 r397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r398, {low, high};
}
mov.f32 f612, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f612;
cvt.rn.f16.f32 high, f612;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r404, {low, high};
}
mov.f32 f620, 0fBF6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f620;
cvt.rn.f16.f32 high, f620;
mov.b32 r405, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
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
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r717, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r718, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r721, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
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
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r913, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r914, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r917, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
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
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r1009, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
mov.b32 r1010, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r1011, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r1012, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
mov.b32 r1013, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r1014, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f612;
cvt.rn.f16.f32 high, f612;
mov.b32 r1017, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r1018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r1019, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r1020, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f620;
cvt.rn.f16.f32 high, f620;
mov.b32 r1021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
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
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r1235, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
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
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r1239, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
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
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
mov.b32 r1243, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
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
cvt.rn.f16.f32 low, f612;
cvt.rn.f16.f32 high, f612;
mov.b32 r1251, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
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
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r1255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
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
cvt.rn.f16.f32 low, f620;
cvt.rn.f16.f32 high, f620;
mov.b32 r1259, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
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
and.b32 r4100, r4099, 15;
shl.b32 r4101, r4099, 8;
and.b32 r4102, r4101, -4096;
add.s32 r4103, r4098, r4102;
cvt.rn.f32.u32 f655, r4100;
mul.f32 f656, f655, 0f3C490FDB;
cos.approx.f32 f357, f656;
sin.approx.f32 f657, f656;
neg.f32 f358, f657;
mov.f32 f659, 0fBF800000;
mov.f32 f658, 0f3F800000;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
and.b32 r4104, r4101, 3840;
add.s32 r4105, r4103, r4104;
st.shared.v4.f32 [r4105], {r1521, r1524, r1723, r1732};
st.shared.v4.f32 [r4105+16], {r1760, r1769, r1797, r1806};
st.shared.v4.f32 [r4105+32], {r1834, r1843, r1871, r1880};
st.shared.v4.f32 [r4105+48], {r1908, r1917, r1945, r1954};
st.shared.v4.f32 [r4105+64], {r1982, r1991, r2019, r2028};
st.shared.v4.f32 [r4105+80], {r2056, r2065, r2093, r2102};
st.shared.v4.f32 [r4105+96], {r2130, r2139, r2167, r2176};
st.shared.v4.f32 [r4105+112], {r2204, r2213, r2241, r2250};
st.shared.v4.f32 [r4105+128], {r2278, r2287, r2315, r2324};
st.shared.v4.f32 [r4105+144], {r2352, r2361, r2389, r2398};
st.shared.v4.f32 [r4105+160], {r2426, r2435, r2463, r2472};
st.shared.v4.f32 [r4105+176], {r2500, r2509, r2537, r2546};
st.shared.v4.f32 [r4105+192], {r2574, r2583, r2611, r2620};
st.shared.v4.f32 [r4105+208], {r2648, r2657, r2685, r2694};
st.shared.v4.f32 [r4105+224], {r2722, r2731, r2759, r2768};
st.shared.v4.f32 [r4105+240], {r2796, r2805, r2833, r2842};
barrier.sync 0;
mad.lo.s32 r4106, r4100, -248, r4105;
ld.shared.u32 r2864, [r4106];
ld.shared.u32 r2867, [r4106+4];
ld.shared.u32 r3480, [r4106+128];
ld.shared.u32 r3483, [r4106+132];
ld.shared.u32 r3060, [r4106+256];
ld.shared.u32 r3063, [r4106+260];
ld.shared.u32 r3676, [r4106+384];
ld.shared.u32 r3679, [r4106+388];
ld.shared.u32 r2914, [r4106+512];
ld.shared.u32 r2917, [r4106+516];
ld.shared.u32 r3530, [r4106+640];
ld.shared.u32 r3533, [r4106+644];
ld.shared.u32 r3110, [r4106+768];
ld.shared.u32 r3113, [r4106+772];
ld.shared.u32 r3726, [r4106+896];
ld.shared.u32 r3729, [r4106+900];
ld.shared.u32 r2876, [r4106+1024];
ld.shared.u32 r2879, [r4106+1028];
ld.shared.u32 r3492, [r4106+1152];
ld.shared.u32 r3495, [r4106+1156];
ld.shared.u32 r3072, [r4106+1280];
ld.shared.u32 r3075, [r4106+1284];
ld.shared.u32 r3688, [r4106+1408];
ld.shared.u32 r3691, [r4106+1412];
ld.shared.u32 r2926, [r4106+1536];
ld.shared.u32 r2929, [r4106+1540];
ld.shared.u32 r3542, [r4106+1664];
ld.shared.u32 r3545, [r4106+1668];
ld.shared.u32 r3122, [r4106+1792];
ld.shared.u32 r3125, [r4106+1796];
ld.shared.u32 r3738, [r4106+1920];
ld.shared.u32 r3741, [r4106+1924];
ld.shared.u32 r2865, [r4106+2048];
ld.shared.u32 r2868, [r4106+2052];
ld.shared.u32 r3481, [r4106+2176];
ld.shared.u32 r3484, [r4106+2180];
ld.shared.u32 r3061, [r4106+2304];
ld.shared.u32 r3064, [r4106+2308];
ld.shared.u32 r3677, [r4106+2432];
ld.shared.u32 r3680, [r4106+2436];
ld.shared.u32 r2915, [r4106+2560];
ld.shared.u32 r2918, [r4106+2564];
ld.shared.u32 r3531, [r4106+2688];
ld.shared.u32 r3534, [r4106+2692];
ld.shared.u32 r3111, [r4106+2816];
ld.shared.u32 r3114, [r4106+2820];
ld.shared.u32 r3727, [r4106+2944];
ld.shared.u32 r3730, [r4106+2948];
ld.shared.u32 r2877, [r4106+3072];
ld.shared.u32 r2880, [r4106+3076];
ld.shared.u32 r3493, [r4106+3200];
ld.shared.u32 r3496, [r4106+3204];
ld.shared.u32 r3073, [r4106+3328];
ld.shared.u32 r3076, [r4106+3332];
ld.shared.u32 r3689, [r4106+3456];
ld.shared.u32 r3692, [r4106+3460];
ld.shared.u32 r2927, [r4106+3584];
ld.shared.u32 r2930, [r4106+3588];
ld.shared.u32 r3543, [r4106+3712];
ld.shared.u32 r3546, [r4106+3716];
ld.shared.u32 r3123, [r4106+3840];
ld.shared.u32 r3126, [r4106+3844];
ld.shared.u32 r3739, [r4106+3968];
ld.shared.u32 r3742, [r4106+3972];
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
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r2963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r2964, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r2967, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
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
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3159, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3160, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r3163, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
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
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r3255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
mov.b32 r3256, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3257, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3258, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
mov.b32 r3259, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r3260, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f612;
cvt.rn.f16.f32 high, f612;
mov.b32 r3263, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r3264, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r3265, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3266, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f620;
cvt.rn.f16.f32 high, f620;
mov.b32 r3267, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
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
add.f16x2 %0, r3011, r3207;
}
{
add.f16x2 %1, r3014, r3210;
}
{
sub.f16x2 %32, r3011, r3207;
}
{
sub.f16x2 %33, r3014, r3210;
}
{
add.f16x2 %4, r3023, r3291;
}
{
add.f16x2 %5, r3026, r3297;
}
{
sub.f16x2 %36, r3023, r3291;
}
{
sub.f16x2 %37, r3026, r3297;
}
{
add.f16x2 %8, r3035, r3307;
}
{
add.f16x2 %9, r3038, r3313;
}
{
sub.f16x2 %40, r3035, r3307;
}
{
sub.f16x2 %41, r3038, r3313;
}
{
add.f16x2 %12, r3047, r3323;
}
{
add.f16x2 %13, r3050, r3329;
}
{
sub.f16x2 %44, r3047, r3323;
}
{
sub.f16x2 %45, r3050, r3329;
}
{
add.f16x2 %16, r3017, r3333;
}
{
add.f16x2 %17, r3020, r3213;
}
{
sub.f16x2 %48, r3017, r3333;
}
{
sub.f16x2 %49, r3020, r3213;
}
{
add.f16x2 %20, r3029, r3341;
}
{
add.f16x2 %21, r3032, r3347;
}
{
sub.f16x2 %52, r3029, r3341;
}
{
sub.f16x2 %53, r3032, r3347;
}
{
add.f16x2 %24, r3041, r3357;
}
{
add.f16x2 %25, r3044, r3363;
}
{
sub.f16x2 %56, r3041, r3357;
}
{
sub.f16x2 %57, r3044, r3363;
}
{
add.f16x2 %28, r3053, r3373;
}
{
add.f16x2 %29, r3056, r3379;
}
{
sub.f16x2 %60, r3053, r3373;
}
{
sub.f16x2 %61, r3056, r3379;
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
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3579, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3580, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r3583, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
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
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3775, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3776, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r3779, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
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
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r3871, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
mov.b32 r3872, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3873, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3874, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
mov.b32 r3875, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r3876, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f612;
cvt.rn.f16.f32 high, f612;
mov.b32 r3879, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r3880, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r3881, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3882, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f620;
cvt.rn.f16.f32 high, f620;
mov.b32 r3883, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
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
add.f16x2 %2, r3627, r3823;
}
{
add.f16x2 %3, r3630, r3826;
}
{
sub.f16x2 %34, r3627, r3823;
}
{
sub.f16x2 %35, r3630, r3826;
}
{
add.f16x2 %6, r3639, r3907;
}
{
add.f16x2 %7, r3642, r3913;
}
{
sub.f16x2 %38, r3639, r3907;
}
{
sub.f16x2 %39, r3642, r3913;
}
{
add.f16x2 %10, r3651, r3923;
}
{
add.f16x2 %11, r3654, r3929;
}
{
sub.f16x2 %42, r3651, r3923;
}
{
sub.f16x2 %43, r3654, r3929;
}
{
add.f16x2 %14, r3663, r3939;
}
{
add.f16x2 %15, r3666, r3945;
}
{
sub.f16x2 %46, r3663, r3939;
}
{
sub.f16x2 %47, r3666, r3945;
}
{
add.f16x2 %18, r3633, r3949;
}
{
add.f16x2 %19, r3636, r3829;
}
{
sub.f16x2 %50, r3633, r3949;
}
{
sub.f16x2 %51, r3636, r3829;
}
{
add.f16x2 %22, r3645, r3957;
}
{
add.f16x2 %23, r3648, r3963;
}
{
sub.f16x2 %54, r3645, r3957;
}
{
sub.f16x2 %55, r3648, r3963;
}
{
add.f16x2 %26, r3657, r3973;
}
{
add.f16x2 %27, r3660, r3979;
}
{
sub.f16x2 %58, r3657, r3973;
}
{
sub.f16x2 %59, r3660, r3979;
}
{
add.f16x2 %30, r3669, r3989;
}
{
add.f16x2 %31, r3672, r3995;
}
{
sub.f16x2 %62, r3669, r3989;
}
{
sub.f16x2 %63, r3672, r3995;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)), "=r"(__HALF2_TO_UI(rmem[27].x)), "=r"(__HALF2_TO_UI(rmem[27].y)), "=r"(__HALF2_TO_UI(rmem[28].x)), "=r"(__HALF2_TO_UI(rmem[28].y)), "=r"(__HALF2_TO_UI(rmem[29].x)), "=r"(__HALF2_TO_UI(rmem[29].y)), "=r"(__HALF2_TO_UI(rmem[30].x)), "=r"(__HALF2_TO_UI(rmem[30].y)), "=r"(__HALF2_TO_UI(rmem[31].x)), "=r"(__HALF2_TO_UI(rmem[31].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[31].y)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[28].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[27].y)), "r"(__HALF2_TO_UI(rmem[30].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[29].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[31].x)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[28].x)), "r"(__HALF2_TO_UI(rmem[27].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[30].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[29].x)), "r"(__HALF2_TO_UI(rmem[24].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1021, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<307>;
.reg .b32 r<2465>;
.reg .b64 rd<2>;
mov.u32 r2445, %tid.y;
shl.b32 r2446, r2445, 11;
mov.u32 r2447, %32;
add.s32 r2448, r2447, r2446;
mov.u32 r2449, %tid.x;
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
mov.f32 f230, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r102, {low, high};
}
mov.f32 f298, 0f3F800000;
mov.f32 f228, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f228;
cvt.rn.f16.f32 high, f228;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r106, {low, high};
}
mov.f32 f297, 0fBF800000;
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
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
cvt.rn.f16.f32 low, f228;
cvt.rn.f16.f32 high, f228;
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
mov.f32 f226, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f226;
cvt.rn.f16.f32 high, f226;
mov.b32 r393, {low, high};
}
mov.f32 f234, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f234;
cvt.rn.f16.f32 high, f234;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r396, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f234;
cvt.rn.f16.f32 high, f234;
mov.b32 r397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f226;
cvt.rn.f16.f32 high, f226;
mov.b32 r398, {low, high};
}
mov.f32 f224, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f224;
cvt.rn.f16.f32 high, f224;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f226;
cvt.rn.f16.f32 high, f226;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f228;
cvt.rn.f16.f32 high, f228;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r404, {low, high};
}
mov.f32 f232, 0fBF6C835E;
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
and.b32 r2450, r2449, 31;
shl.b32 r2451, r2449, 6;
and.b32 r2452, r2451, -2048;
add.s32 r2453, r2448, r2452;
cvt.rn.f32.u32 f301, r2450;
mul.f32 f302, f301, 0f3C490FDB;
cos.approx.f32 f117, f302;
sin.approx.f32 f303, f302;
neg.f32 f118, f303;
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
and.b32 r2454, r2451, 1984;
add.s32 r2455, r2453, r2454;
st.shared.v4.f32 [r2455], {r521, r627, r664, r701};
st.shared.v4.f32 [r2455+16], {r738, r775, r812, r849};
st.shared.v4.f32 [r2455+32], {r886, r923, r960, r997};
st.shared.v4.f32 [r2455+48], {r1034, r1071, r1108, r1145};
barrier.sync 0;
mad.lo.s32 r2456, r2450, -60, r2455;
ld.shared.u32 r1176, [r2456];
ld.shared.u32 r1372, [r2456+128];
ld.shared.u32 r1226, [r2456+256];
ld.shared.u32 r1422, [r2456+384];
ld.shared.u32 r1188, [r2456+512];
ld.shared.u32 r1384, [r2456+640];
ld.shared.u32 r1238, [r2456+768];
ld.shared.u32 r1434, [r2456+896];
ld.shared.u32 r1177, [r2456+1024];
ld.shared.u32 r1373, [r2456+1152];
ld.shared.u32 r1227, [r2456+1280];
ld.shared.u32 r1423, [r2456+1408];
ld.shared.u32 r1189, [r2456+1536];
ld.shared.u32 r1385, [r2456+1664];
ld.shared.u32 r1239, [r2456+1792];
ld.shared.u32 r1435, [r2456+1920];
barrier.sync 0;
st.shared.v4.f32 [r2455], {r524, r636, r673, r710};
st.shared.v4.f32 [r2455+16], {r747, r784, r821, r858};
st.shared.v4.f32 [r2455+32], {r895, r932, r969, r1006};
st.shared.v4.f32 [r2455+48], {r1043, r1080, r1117, r1154};
barrier.sync 0;
ld.shared.u32 r1179, [r2456];
ld.shared.u32 r1375, [r2456+128];
ld.shared.u32 r1229, [r2456+256];
ld.shared.u32 r1425, [r2456+384];
ld.shared.u32 r1191, [r2456+512];
ld.shared.u32 r1387, [r2456+640];
ld.shared.u32 r1241, [r2456+768];
ld.shared.u32 r1437, [r2456+896];
ld.shared.u32 r1180, [r2456+1024];
ld.shared.u32 r1376, [r2456+1152];
ld.shared.u32 r1230, [r2456+1280];
ld.shared.u32 r1426, [r2456+1408];
ld.shared.u32 r1192, [r2456+1536];
ld.shared.u32 r1388, [r2456+1664];
ld.shared.u32 r1242, [r2456+1792];
ld.shared.u32 r1438, [r2456+1920];
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
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
cvt.rn.f16.f32 low, f228;
cvt.rn.f16.f32 high, f228;
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
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
cvt.rn.f16.f32 low, f228;
cvt.rn.f16.f32 high, f228;
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
cvt.rn.f16.f32 low, f226;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
cvt.rn.f16.f32 low, f234;
cvt.rn.f16.f32 high, f234;
mov.b32 r1571, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f226;
cvt.rn.f16.f32 high, f226;
mov.b32 r1572, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f224;
cvt.rn.f16.f32 high, f224;
mov.b32 r1575, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f226;
cvt.rn.f16.f32 high, f226;
mov.b32 r1576, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f228;
cvt.rn.f16.f32 high, f228;
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
and.b32 r2457, r2449, 16;
bfe.u32 r2458, r2449, 4, 1;
shl.b32 r2459, r2449, 2;
and.b32 r2460, r2459, 60;
add.s32 r2461, r2453, r2460;
cvt.rn.f32.u32 f304, r2458;
mul.f32 f305, f304, 0f3E490FDB;
cos.approx.f32 f267, f305;
sin.approx.f32 f306, f305;
neg.f32 f268, f306;
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
and.b32 r2462, r2451, 1024;
add.s32 r2463, r2461, r2462;
st.shared.u32 [r2463], r1695;
st.shared.u32 [r2463+64], r1801;
st.shared.u32 [r2463+128], r1838;
st.shared.u32 [r2463+192], r1875;
st.shared.u32 [r2463+256], r1912;
st.shared.u32 [r2463+320], r1949;
st.shared.u32 [r2463+384], r1986;
st.shared.u32 [r2463+448], r2023;
st.shared.u32 [r2463+512], r2060;
st.shared.u32 [r2463+576], r2097;
st.shared.u32 [r2463+640], r2134;
st.shared.u32 [r2463+704], r2171;
st.shared.u32 [r2463+768], r2208;
st.shared.u32 [r2463+832], r2245;
st.shared.u32 [r2463+896], r2282;
st.shared.u32 [r2463+960], r2319;
barrier.sync 0;
mad.lo.s32 r2464, r2457, -60, r2463;
ld.shared.u32 r2350, [r2464];
ld.shared.u32 r2362, [r2464+128];
ld.shared.u32 r2374, [r2464+256];
ld.shared.u32 r2386, [r2464+384];
ld.shared.u32 r2398, [r2464+512];
ld.shared.u32 r2410, [r2464+640];
ld.shared.u32 r2422, [r2464+768];
ld.shared.u32 r2434, [r2464+896];
ld.shared.u32 r2351, [r2464+1024];
ld.shared.u32 r2363, [r2464+1152];
ld.shared.u32 r2375, [r2464+1280];
ld.shared.u32 r2387, [r2464+1408];
ld.shared.u32 r2399, [r2464+1536];
ld.shared.u32 r2411, [r2464+1664];
ld.shared.u32 r2423, [r2464+1792];
ld.shared.u32 r2435, [r2464+1920];
barrier.sync 0;
st.shared.u32 [r2463], r1698;
st.shared.u32 [r2463+64], r1810;
st.shared.u32 [r2463+128], r1847;
st.shared.u32 [r2463+192], r1884;
st.shared.u32 [r2463+256], r1921;
st.shared.u32 [r2463+320], r1958;
st.shared.u32 [r2463+384], r1995;
st.shared.u32 [r2463+448], r2032;
st.shared.u32 [r2463+512], r2069;
st.shared.u32 [r2463+576], r2106;
st.shared.u32 [r2463+640], r2143;
st.shared.u32 [r2463+704], r2180;
st.shared.u32 [r2463+768], r2217;
st.shared.u32 [r2463+832], r2254;
st.shared.u32 [r2463+896], r2291;
st.shared.u32 [r2463+960], r2328;
barrier.sync 0;
ld.shared.u32 r2353, [r2464];
ld.shared.u32 r2365, [r2464+128];
ld.shared.u32 r2377, [r2464+256];
ld.shared.u32 r2389, [r2464+384];
ld.shared.u32 r2401, [r2464+512];
ld.shared.u32 r2413, [r2464+640];
ld.shared.u32 r2425, [r2464+768];
ld.shared.u32 r2437, [r2464+896];
ld.shared.u32 r2354, [r2464+1024];
ld.shared.u32 r2366, [r2464+1152];
ld.shared.u32 r2378, [r2464+1280];
ld.shared.u32 r2390, [r2464+1408];
ld.shared.u32 r2402, [r2464+1536];
ld.shared.u32 r2414, [r2464+1664];
ld.shared.u32 r2426, [r2464+1792];
ld.shared.u32 r2438, [r2464+1920];
{
add.f16x2 %0, r2350, r2351;
}
{
add.f16x2 %1, r2353, r2354;
}
{
sub.f16x2 %16, r2350, r2351;
}
{
sub.f16x2 %17, r2353, r2354;
}
{
add.f16x2 %2, r2362, r2363;
}
{
add.f16x2 %3, r2365, r2366;
}
{
sub.f16x2 %18, r2362, r2363;
}
{
sub.f16x2 %19, r2365, r2366;
}
{
add.f16x2 %4, r2374, r2375;
}
{
add.f16x2 %5, r2377, r2378;
}
{
sub.f16x2 %20, r2374, r2375;
}
{
sub.f16x2 %21, r2377, r2378;
}
{
add.f16x2 %6, r2386, r2387;
}
{
add.f16x2 %7, r2389, r2390;
}
{
sub.f16x2 %22, r2386, r2387;
}
{
sub.f16x2 %23, r2389, r2390;
}
{
add.f16x2 %8, r2398, r2399;
}
{
add.f16x2 %9, r2401, r2402;
}
{
sub.f16x2 %24, r2398, r2399;
}
{
sub.f16x2 %25, r2401, r2402;
}
{
add.f16x2 %10, r2410, r2411;
}
{
add.f16x2 %11, r2413, r2414;
}
{
sub.f16x2 %26, r2410, r2411;
}
{
sub.f16x2 %27, r2413, r2414;
}
{
add.f16x2 %12, r2422, r2423;
}
{
add.f16x2 %13, r2425, r2426;
}
{
sub.f16x2 %28, r2422, r2423;
}
{
sub.f16x2 %29, r2425, r2426;
}
{
add.f16x2 %14, r2434, r2435;
}
{
add.f16x2 %15, r2437, r2438;
}
{
sub.f16x2 %30, r2434, r2435;
}
{
sub.f16x2 %31, r2437, r2438;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1024, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<307>;
.reg .b32 r<2465>;
.reg .b64 rd<2>;
mov.u32 r2445, %tid.y;
shl.b32 r2446, r2445, 12;
mov.u32 r2447, %32;
add.s32 r2448, r2447, r2446;
mov.u32 r2449, %tid.x;
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
mov.f32 f230, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r102, {low, high};
}
mov.f32 f298, 0f3F800000;
mov.f32 f228, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f228;
cvt.rn.f16.f32 high, f228;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r106, {low, high};
}
mov.f32 f297, 0fBF800000;
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
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
cvt.rn.f16.f32 low, f228;
cvt.rn.f16.f32 high, f228;
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
mov.f32 f226, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f226;
cvt.rn.f16.f32 high, f226;
mov.b32 r393, {low, high};
}
mov.f32 f234, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f234;
cvt.rn.f16.f32 high, f234;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r396, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f234;
cvt.rn.f16.f32 high, f234;
mov.b32 r397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f226;
cvt.rn.f16.f32 high, f226;
mov.b32 r398, {low, high};
}
mov.f32 f224, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f224;
cvt.rn.f16.f32 high, f224;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f226;
cvt.rn.f16.f32 high, f226;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f228;
cvt.rn.f16.f32 high, f228;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r404, {low, high};
}
mov.f32 f232, 0fBF6C835E;
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
and.b32 r2450, r2449, 31;
shl.b32 r2451, r2449, 7;
and.b32 r2452, r2451, -4096;
add.s32 r2453, r2448, r2452;
cvt.rn.f32.u32 f301, r2450;
mul.f32 f302, f301, 0f3C490FDB;
cos.approx.f32 f117, f302;
sin.approx.f32 f303, f302;
neg.f32 f118, f303;
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
and.b32 r2454, r2451, 3968;
add.s32 r2455, r2453, r2454;
st.shared.v4.f32 [r2455], {r521, r524, r627, r636};
st.shared.v4.f32 [r2455+16], {r664, r673, r701, r710};
st.shared.v4.f32 [r2455+32], {r738, r747, r775, r784};
st.shared.v4.f32 [r2455+48], {r812, r821, r849, r858};
st.shared.v4.f32 [r2455+64], {r886, r895, r923, r932};
st.shared.v4.f32 [r2455+80], {r960, r969, r997, r1006};
st.shared.v4.f32 [r2455+96], {r1034, r1043, r1071, r1080};
st.shared.v4.f32 [r2455+112], {r1108, r1117, r1145, r1154};
barrier.sync 0;
mad.lo.s32 r2456, r2450, -120, r2455;
ld.shared.u32 r1176, [r2456];
ld.shared.u32 r1179, [r2456+4];
ld.shared.u32 r1372, [r2456+256];
ld.shared.u32 r1375, [r2456+260];
ld.shared.u32 r1226, [r2456+512];
ld.shared.u32 r1229, [r2456+516];
ld.shared.u32 r1422, [r2456+768];
ld.shared.u32 r1425, [r2456+772];
ld.shared.u32 r1188, [r2456+1024];
ld.shared.u32 r1191, [r2456+1028];
ld.shared.u32 r1384, [r2456+1280];
ld.shared.u32 r1387, [r2456+1284];
ld.shared.u32 r1238, [r2456+1536];
ld.shared.u32 r1241, [r2456+1540];
ld.shared.u32 r1434, [r2456+1792];
ld.shared.u32 r1437, [r2456+1796];
ld.shared.u32 r1177, [r2456+2048];
ld.shared.u32 r1180, [r2456+2052];
ld.shared.u32 r1373, [r2456+2304];
ld.shared.u32 r1376, [r2456+2308];
ld.shared.u32 r1227, [r2456+2560];
ld.shared.u32 r1230, [r2456+2564];
ld.shared.u32 r1423, [r2456+2816];
ld.shared.u32 r1426, [r2456+2820];
ld.shared.u32 r1189, [r2456+3072];
ld.shared.u32 r1192, [r2456+3076];
ld.shared.u32 r1385, [r2456+3328];
ld.shared.u32 r1388, [r2456+3332];
ld.shared.u32 r1239, [r2456+3584];
ld.shared.u32 r1242, [r2456+3588];
ld.shared.u32 r1435, [r2456+3840];
ld.shared.u32 r1438, [r2456+3844];
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
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
cvt.rn.f16.f32 low, f228;
cvt.rn.f16.f32 high, f228;
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
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
cvt.rn.f16.f32 low, f228;
cvt.rn.f16.f32 high, f228;
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
cvt.rn.f16.f32 low, f226;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
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
cvt.rn.f16.f32 low, f234;
cvt.rn.f16.f32 high, f234;
mov.b32 r1571, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f226;
cvt.rn.f16.f32 high, f226;
mov.b32 r1572, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f224;
cvt.rn.f16.f32 high, f224;
mov.b32 r1575, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f226;
cvt.rn.f16.f32 high, f226;
mov.b32 r1576, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f228;
cvt.rn.f16.f32 high, f228;
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
and.b32 r2457, r2449, 16;
bfe.u32 r2458, r2449, 4, 1;
cvt.rn.f32.u32 f304, r2458;
mul.f32 f305, f304, 0f3E490FDB;
cos.approx.f32 f267, f305;
sin.approx.f32 f306, f305;
neg.f32 f268, f306;
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
shl.b32 r2459, r2449, 3;
and.b32 r2460, r2459, 120;
add.s32 r2461, r2453, r2460;
barrier.sync 0;
and.b32 r2462, r2451, 2048;
add.s32 r2463, r2461, r2462;
st.shared.u32 [r2463], r1695;
st.shared.u32 [r2463+4], r1698;
st.shared.u32 [r2463+128], r1801;
st.shared.u32 [r2463+132], r1810;
st.shared.u32 [r2463+256], r1838;
st.shared.u32 [r2463+260], r1847;
st.shared.u32 [r2463+384], r1875;
st.shared.u32 [r2463+388], r1884;
st.shared.u32 [r2463+512], r1912;
st.shared.u32 [r2463+516], r1921;
st.shared.u32 [r2463+640], r1949;
st.shared.u32 [r2463+644], r1958;
st.shared.u32 [r2463+768], r1986;
st.shared.u32 [r2463+772], r1995;
st.shared.u32 [r2463+896], r2023;
st.shared.u32 [r2463+900], r2032;
st.shared.u32 [r2463+1024], r2060;
st.shared.u32 [r2463+1028], r2069;
st.shared.u32 [r2463+1152], r2097;
st.shared.u32 [r2463+1156], r2106;
st.shared.u32 [r2463+1280], r2134;
st.shared.u32 [r2463+1284], r2143;
st.shared.u32 [r2463+1408], r2171;
st.shared.u32 [r2463+1412], r2180;
st.shared.u32 [r2463+1536], r2208;
st.shared.u32 [r2463+1540], r2217;
st.shared.u32 [r2463+1664], r2245;
st.shared.u32 [r2463+1668], r2254;
st.shared.u32 [r2463+1792], r2282;
st.shared.u32 [r2463+1796], r2291;
st.shared.u32 [r2463+1920], r2319;
st.shared.u32 [r2463+1924], r2328;
barrier.sync 0;
mad.lo.s32 r2464, r2457, -120, r2463;
ld.shared.u32 r2350, [r2464];
ld.shared.u32 r2353, [r2464+4];
ld.shared.u32 r2362, [r2464+256];
ld.shared.u32 r2365, [r2464+260];
ld.shared.u32 r2374, [r2464+512];
ld.shared.u32 r2377, [r2464+516];
ld.shared.u32 r2386, [r2464+768];
ld.shared.u32 r2389, [r2464+772];
ld.shared.u32 r2398, [r2464+1024];
ld.shared.u32 r2401, [r2464+1028];
ld.shared.u32 r2410, [r2464+1280];
ld.shared.u32 r2413, [r2464+1284];
ld.shared.u32 r2422, [r2464+1536];
ld.shared.u32 r2425, [r2464+1540];
ld.shared.u32 r2434, [r2464+1792];
ld.shared.u32 r2437, [r2464+1796];
ld.shared.u32 r2351, [r2464+2048];
ld.shared.u32 r2354, [r2464+2052];
ld.shared.u32 r2363, [r2464+2304];
ld.shared.u32 r2366, [r2464+2308];
ld.shared.u32 r2375, [r2464+2560];
ld.shared.u32 r2378, [r2464+2564];
ld.shared.u32 r2387, [r2464+2816];
ld.shared.u32 r2390, [r2464+2820];
ld.shared.u32 r2399, [r2464+3072];
ld.shared.u32 r2402, [r2464+3076];
ld.shared.u32 r2411, [r2464+3328];
ld.shared.u32 r2414, [r2464+3332];
ld.shared.u32 r2423, [r2464+3584];
ld.shared.u32 r2426, [r2464+3588];
ld.shared.u32 r2435, [r2464+3840];
ld.shared.u32 r2438, [r2464+3844];
{
add.f16x2 %0, r2350, r2351;
}
{
add.f16x2 %1, r2353, r2354;
}
{
sub.f16x2 %16, r2350, r2351;
}
{
sub.f16x2 %17, r2353, r2354;
}
{
add.f16x2 %2, r2362, r2363;
}
{
add.f16x2 %3, r2365, r2366;
}
{
sub.f16x2 %18, r2362, r2363;
}
{
sub.f16x2 %19, r2365, r2366;
}
{
add.f16x2 %4, r2374, r2375;
}
{
add.f16x2 %5, r2377, r2378;
}
{
sub.f16x2 %20, r2374, r2375;
}
{
sub.f16x2 %21, r2377, r2378;
}
{
add.f16x2 %6, r2386, r2387;
}
{
add.f16x2 %7, r2389, r2390;
}
{
sub.f16x2 %22, r2386, r2387;
}
{
sub.f16x2 %23, r2389, r2390;
}
{
add.f16x2 %8, r2398, r2399;
}
{
add.f16x2 %9, r2401, r2402;
}
{
sub.f16x2 %24, r2398, r2399;
}
{
sub.f16x2 %25, r2401, r2402;
}
{
add.f16x2 %10, r2410, r2411;
}
{
add.f16x2 %11, r2413, r2414;
}
{
sub.f16x2 %26, r2410, r2411;
}
{
sub.f16x2 %27, r2413, r2414;
}
{
add.f16x2 %12, r2422, r2423;
}
{
add.f16x2 %13, r2425, r2426;
}
{
sub.f16x2 %28, r2422, r2423;
}
{
sub.f16x2 %29, r2425, r2426;
}
{
add.f16x2 %14, r2434, r2435;
}
{
add.f16x2 %15, r2437, r2438;
}
{
sub.f16x2 %30, r2434, r2435;
}
{
sub.f16x2 %31, r2437, r2438;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1023, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<660>;
.reg .b32 r<4171>;
.reg .b64 rd<3>;
mov.u32 r4095, %tid.y;
shl.b32 r4096, r4095, 11;
mov.u32 r4097, %64;
add.s32 r4098, r4097, r4096;
mov.u32 r4099, %tid.x;
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
mov.f32 f618, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r102, {low, high};
}
mov.f32 f616, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
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
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
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
mov.f32 f614, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r393, {low, high};
}
mov.f32 f622, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r396, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
mov.b32 r397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r398, {low, high};
}
mov.f32 f612, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f612;
cvt.rn.f16.f32 high, f612;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r404, {low, high};
}
mov.f32 f620, 0fBF6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f620;
cvt.rn.f16.f32 high, f620;
mov.b32 r405, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
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
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r717, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r718, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r721, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
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
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r913, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r914, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r917, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
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
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r1009, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
mov.b32 r1010, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r1011, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r1012, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
mov.b32 r1013, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r1014, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f612;
cvt.rn.f16.f32 high, f612;
mov.b32 r1017, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r1018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r1019, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r1020, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f620;
cvt.rn.f16.f32 high, f620;
mov.b32 r1021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
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
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r1235, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
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
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r1239, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
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
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
mov.b32 r1243, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
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
cvt.rn.f16.f32 low, f612;
cvt.rn.f16.f32 high, f612;
mov.b32 r1251, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
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
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r1255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
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
cvt.rn.f16.f32 low, f620;
cvt.rn.f16.f32 high, f620;
mov.b32 r1259, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
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
and.b32 r4100, r4099, 15;
shl.b32 r4101, r4099, 7;
and.b32 r4102, r4101, -2048;
add.s32 r4103, r4098, r4102;
cvt.rn.f32.u32 f655, r4100;
mul.f32 f656, f655, 0f3C490FDB;
cos.approx.f32 f357, f656;
sin.approx.f32 f657, f656;
neg.f32 f358, f657;
mov.f32 f659, 0fBF800000;
mov.f32 f658, 0f3F800000;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
cvt.rn.f16.f32 low, f659;
cvt.rn.f16.f32 high, f658;
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
and.b32 r4104, r4101, 1920;
add.s32 r4105, r4103, r4104;
st.shared.v4.f32 [r4105], {r1521, r1723, r1760, r1797};
st.shared.v4.f32 [r4105+16], {r1834, r1871, r1908, r1945};
st.shared.v4.f32 [r4105+32], {r1982, r2019, r2056, r2093};
st.shared.v4.f32 [r4105+48], {r2130, r2167, r2204, r2241};
st.shared.v4.f32 [r4105+64], {r2278, r2315, r2352, r2389};
st.shared.v4.f32 [r4105+80], {r2426, r2463, r2500, r2537};
st.shared.v4.f32 [r4105+96], {r2574, r2611, r2648, r2685};
st.shared.v4.f32 [r4105+112], {r2722, r2759, r2796, r2833};
barrier.sync 0;
mad.lo.s32 r4106, r4100, -124, r4105;
ld.shared.u32 r2864, [r4106];
ld.shared.u32 r3480, [r4106+64];
ld.shared.u32 r3060, [r4106+128];
ld.shared.u32 r3676, [r4106+192];
ld.shared.u32 r2914, [r4106+256];
ld.shared.u32 r3530, [r4106+320];
ld.shared.u32 r3110, [r4106+384];
ld.shared.u32 r3726, [r4106+448];
ld.shared.u32 r2876, [r4106+512];
ld.shared.u32 r3492, [r4106+576];
ld.shared.u32 r3072, [r4106+640];
ld.shared.u32 r3688, [r4106+704];
ld.shared.u32 r2926, [r4106+768];
ld.shared.u32 r3542, [r4106+832];
ld.shared.u32 r3122, [r4106+896];
ld.shared.u32 r3738, [r4106+960];
ld.shared.u32 r2865, [r4106+1024];
ld.shared.u32 r3481, [r4106+1088];
ld.shared.u32 r3061, [r4106+1152];
ld.shared.u32 r3677, [r4106+1216];
ld.shared.u32 r2915, [r4106+1280];
ld.shared.u32 r3531, [r4106+1344];
ld.shared.u32 r3111, [r4106+1408];
ld.shared.u32 r3727, [r4106+1472];
ld.shared.u32 r2877, [r4106+1536];
ld.shared.u32 r3493, [r4106+1600];
ld.shared.u32 r3073, [r4106+1664];
ld.shared.u32 r3689, [r4106+1728];
ld.shared.u32 r2927, [r4106+1792];
ld.shared.u32 r3543, [r4106+1856];
ld.shared.u32 r3123, [r4106+1920];
ld.shared.u32 r3739, [r4106+1984];
barrier.sync 0;
st.shared.v4.f32 [r4105], {r1524, r1732, r1769, r1806};
st.shared.v4.f32 [r4105+16], {r1843, r1880, r1917, r1954};
st.shared.v4.f32 [r4105+32], {r1991, r2028, r2065, r2102};
st.shared.v4.f32 [r4105+48], {r2139, r2176, r2213, r2250};
st.shared.v4.f32 [r4105+64], {r2287, r2324, r2361, r2398};
st.shared.v4.f32 [r4105+80], {r2435, r2472, r2509, r2546};
st.shared.v4.f32 [r4105+96], {r2583, r2620, r2657, r2694};
st.shared.v4.f32 [r4105+112], {r2731, r2768, r2805, r2842};
barrier.sync 0;
ld.shared.u32 r2867, [r4106];
ld.shared.u32 r3483, [r4106+64];
ld.shared.u32 r3063, [r4106+128];
ld.shared.u32 r3679, [r4106+192];
ld.shared.u32 r2917, [r4106+256];
ld.shared.u32 r3533, [r4106+320];
ld.shared.u32 r3113, [r4106+384];
ld.shared.u32 r3729, [r4106+448];
ld.shared.u32 r2879, [r4106+512];
ld.shared.u32 r3495, [r4106+576];
ld.shared.u32 r3075, [r4106+640];
ld.shared.u32 r3691, [r4106+704];
ld.shared.u32 r2929, [r4106+768];
ld.shared.u32 r3545, [r4106+832];
ld.shared.u32 r3125, [r4106+896];
ld.shared.u32 r3741, [r4106+960];
ld.shared.u32 r2868, [r4106+1024];
ld.shared.u32 r3484, [r4106+1088];
ld.shared.u32 r3064, [r4106+1152];
ld.shared.u32 r3680, [r4106+1216];
ld.shared.u32 r2918, [r4106+1280];
ld.shared.u32 r3534, [r4106+1344];
ld.shared.u32 r3114, [r4106+1408];
ld.shared.u32 r3730, [r4106+1472];
ld.shared.u32 r2880, [r4106+1536];
ld.shared.u32 r3496, [r4106+1600];
ld.shared.u32 r3076, [r4106+1664];
ld.shared.u32 r3692, [r4106+1728];
ld.shared.u32 r2930, [r4106+1792];
ld.shared.u32 r3546, [r4106+1856];
ld.shared.u32 r3126, [r4106+1920];
ld.shared.u32 r3742, [r4106+1984];
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
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r2963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r2964, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r2967, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
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
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3159, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3160, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r3163, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
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
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r3255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
mov.b32 r3256, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3257, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3258, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
mov.b32 r3259, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r3260, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f612;
cvt.rn.f16.f32 high, f612;
mov.b32 r3263, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r3264, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r3265, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3266, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f620;
cvt.rn.f16.f32 high, f620;
mov.b32 r3267, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
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
add.f16x2 %0, r3011, r3207;
}
{
add.f16x2 %1, r3014, r3210;
}
{
sub.f16x2 %32, r3011, r3207;
}
{
sub.f16x2 %33, r3014, r3210;
}
{
add.f16x2 %4, r3023, r3291;
}
{
add.f16x2 %5, r3026, r3297;
}
{
sub.f16x2 %36, r3023, r3291;
}
{
sub.f16x2 %37, r3026, r3297;
}
{
add.f16x2 %8, r3035, r3307;
}
{
add.f16x2 %9, r3038, r3313;
}
{
sub.f16x2 %40, r3035, r3307;
}
{
sub.f16x2 %41, r3038, r3313;
}
{
add.f16x2 %12, r3047, r3323;
}
{
add.f16x2 %13, r3050, r3329;
}
{
sub.f16x2 %44, r3047, r3323;
}
{
sub.f16x2 %45, r3050, r3329;
}
{
add.f16x2 %16, r3017, r3333;
}
{
add.f16x2 %17, r3020, r3213;
}
{
sub.f16x2 %48, r3017, r3333;
}
{
sub.f16x2 %49, r3020, r3213;
}
{
add.f16x2 %20, r3029, r3341;
}
{
add.f16x2 %21, r3032, r3347;
}
{
sub.f16x2 %52, r3029, r3341;
}
{
sub.f16x2 %53, r3032, r3347;
}
{
add.f16x2 %24, r3041, r3357;
}
{
add.f16x2 %25, r3044, r3363;
}
{
sub.f16x2 %56, r3041, r3357;
}
{
sub.f16x2 %57, r3044, r3363;
}
{
add.f16x2 %28, r3053, r3373;
}
{
add.f16x2 %29, r3056, r3379;
}
{
sub.f16x2 %60, r3053, r3373;
}
{
sub.f16x2 %61, r3056, r3379;
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
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3579, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3580, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r3583, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
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
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3775, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3776, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r3779, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
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
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r3871, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
mov.b32 r3872, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3873, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3874, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
mov.b32 r3875, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r3876, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f612;
cvt.rn.f16.f32 high, f612;
mov.b32 r3879, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f614;
cvt.rn.f16.f32 high, f614;
mov.b32 r3880, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f616;
cvt.rn.f16.f32 high, f616;
mov.b32 r3881, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f618;
cvt.rn.f16.f32 high, f618;
mov.b32 r3882, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f620;
cvt.rn.f16.f32 high, f620;
mov.b32 r3883, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f622;
cvt.rn.f16.f32 high, f622;
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
add.f16x2 %2, r3627, r3823;
}
{
add.f16x2 %3, r3630, r3826;
}
{
sub.f16x2 %34, r3627, r3823;
}
{
sub.f16x2 %35, r3630, r3826;
}
{
add.f16x2 %6, r3639, r3907;
}
{
add.f16x2 %7, r3642, r3913;
}
{
sub.f16x2 %38, r3639, r3907;
}
{
sub.f16x2 %39, r3642, r3913;
}
{
add.f16x2 %10, r3651, r3923;
}
{
add.f16x2 %11, r3654, r3929;
}
{
sub.f16x2 %42, r3651, r3923;
}
{
sub.f16x2 %43, r3654, r3929;
}
{
add.f16x2 %14, r3663, r3939;
}
{
add.f16x2 %15, r3666, r3945;
}
{
sub.f16x2 %46, r3663, r3939;
}
{
sub.f16x2 %47, r3666, r3945;
}
{
add.f16x2 %18, r3633, r3949;
}
{
add.f16x2 %19, r3636, r3829;
}
{
sub.f16x2 %50, r3633, r3949;
}
{
sub.f16x2 %51, r3636, r3829;
}
{
add.f16x2 %22, r3645, r3957;
}
{
add.f16x2 %23, r3648, r3963;
}
{
sub.f16x2 %54, r3645, r3957;
}
{
sub.f16x2 %55, r3648, r3963;
}
{
add.f16x2 %26, r3657, r3973;
}
{
add.f16x2 %27, r3660, r3979;
}
{
sub.f16x2 %58, r3657, r3973;
}
{
sub.f16x2 %59, r3660, r3979;
}
{
add.f16x2 %30, r3669, r3989;
}
{
add.f16x2 %31, r3672, r3995;
}
{
sub.f16x2 %62, r3669, r3989;
}
{
sub.f16x2 %63, r3672, r3995;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)), "=r"(__HALF2_TO_UI(rmem[27].x)), "=r"(__HALF2_TO_UI(rmem[27].y)), "=r"(__HALF2_TO_UI(rmem[28].x)), "=r"(__HALF2_TO_UI(rmem[28].y)), "=r"(__HALF2_TO_UI(rmem[29].x)), "=r"(__HALF2_TO_UI(rmem[29].y)), "=r"(__HALF2_TO_UI(rmem[30].x)), "=r"(__HALF2_TO_UI(rmem[30].y)), "=r"(__HALF2_TO_UI(rmem[31].x)), "=r"(__HALF2_TO_UI(rmem[31].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[31].y)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[28].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[27].y)), "r"(__HALF2_TO_UI(rmem[30].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[29].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[31].x)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[28].x)), "r"(__HALF2_TO_UI(rmem[27].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[30].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[29].x)), "r"(__HALF2_TO_UI(rmem[24].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1025, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<53>;
.reg .b32 r<715>;
.reg .b64 rd<2>;
mov.u32 r681, %tid.y;
shl.b32 r682, r681, 12;
mov.u32 r683, %8;
add.s32 r684, r683, r682;
mov.u32 r685, %tid.x;
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
and.b32 r686, r685, 127;
shl.b32 r687, r685, 5;
and.b32 r688, r687, -4096;
add.s32 r689, r684, r688;
cvt.rn.f32.u32 f41, r686;
mul.f32 f42, f41, 0f3C490FDB;
cos.approx.f32 f1, f42;
sin.approx.f32 f43, f42;
neg.f32 f2, f43;
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
mov.f32 f37, 0fBF800000;
mov.f32 f38, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
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
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
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
and.b32 r690, r687, 4064;
add.s32 r691, r689, r690;
st.shared.v4.f32 [r691], {r27, r30, r61, r70};
st.shared.v4.f32 [r691+16], {r98, r107, r135, r144};
barrier.sync 0;
mad.lo.s32 r692, r686, -24, r691;
ld.shared.u32 r166, [r692];
ld.shared.u32 r169, [r692+4];
ld.shared.u32 r178, [r692+1024];
ld.shared.u32 r181, [r692+1028];
ld.shared.u32 r167, [r692+2048];
ld.shared.u32 r170, [r692+2052];
ld.shared.u32 r179, [r692+3072];
ld.shared.u32 r182, [r692+3076];
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
and.b32 r693, r685, 124;
bfe.u32 r694, r685, 2, 5;
cvt.rn.f32.u32 f44, r694;
mul.f32 f45, f44, 0f3D490FDB;
cos.approx.f32 f11, f45;
sin.approx.f32 f46, f45;
neg.f32 f12, f46;
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
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
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
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
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
shl.b32 r695, r685, 3;
and.b32 r696, r695, 24;
add.s32 r697, r689, r696;
barrier.sync 0;
and.b32 r698, r687, 3968;
add.s32 r699, r697, r698;
st.shared.u32 [r699], r191;
st.shared.u32 [r699+4], r194;
st.shared.u32 [r699+32], r225;
st.shared.u32 [r699+36], r234;
st.shared.u32 [r699+64], r262;
st.shared.u32 [r699+68], r271;
st.shared.u32 [r699+96], r299;
st.shared.u32 [r699+100], r308;
barrier.sync 0;
mad.lo.s32 r700, r693, -24, r699;
ld.shared.u32 r330, [r700];
ld.shared.u32 r333, [r700+4];
ld.shared.u32 r342, [r700+1024];
ld.shared.u32 r345, [r700+1028];
ld.shared.u32 r331, [r700+2048];
ld.shared.u32 r334, [r700+2052];
ld.shared.u32 r343, [r700+3072];
ld.shared.u32 r346, [r700+3076];
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
and.b32 r701, r685, 112;
bfe.u32 r702, r685, 4, 3;
cvt.rn.f32.u32 f47, r702;
mul.f32 f48, f47, 0f3E490FDB;
cos.approx.f32 f21, f48;
sin.approx.f32 f49, f48;
neg.f32 f22, f49;
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
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
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
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
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
and.b32 r703, r695, 120;
add.s32 r704, r689, r703;
barrier.sync 0;
and.b32 r705, r687, 3584;
add.s32 r706, r704, r705;
st.shared.u32 [r706], r355;
st.shared.u32 [r706+4], r358;
st.shared.u32 [r706+128], r389;
st.shared.u32 [r706+132], r398;
st.shared.u32 [r706+256], r426;
st.shared.u32 [r706+260], r435;
st.shared.u32 [r706+384], r463;
st.shared.u32 [r706+388], r472;
barrier.sync 0;
mad.lo.s32 r707, r701, -24, r706;
ld.shared.u32 r494, [r707];
ld.shared.u32 r497, [r707+4];
ld.shared.u32 r506, [r707+1024];
ld.shared.u32 r509, [r707+1028];
ld.shared.u32 r495, [r707+2048];
ld.shared.u32 r498, [r707+2052];
ld.shared.u32 r507, [r707+3072];
ld.shared.u32 r510, [r707+3076];
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
neg.f16x2 r517, r514;
}
{
add.f16x2 r519, r493, r505;
}
{
add.f16x2 r522, r496, r508;
}
{
sub.f16x2 r525, r493, r505;
}
{
sub.f16x2 r528, r496, r508;
}
{
add.f16x2 r531, r499, r517;
}
{
add.f16x2 r534, r502, r511;
}
{
sub.f16x2 r537, r499, r517;
}
{
sub.f16x2 r540, r502, r511;
}
and.b32 r708, r685, 64;
bfe.u32 r709, r685, 6, 1;
cvt.rn.f32.u32 f50, r709;
mul.f32 f51, f50, 0f3F490FDB;
cos.approx.f32 f31, f51;
sin.approx.f32 f52, f51;
neg.f32 f32, f52;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f31;
cvt.rn.f16.f32 high, f32;
mov.b32 r543, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r543;
mov.b32 r546, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r543;
mov.b32 r548, {high, high};
}
{
mul.f16x2 r550, r534, r548;
}
{
fma.rn.f16x2 r553, r531, r546, r550;
}
{
mul.f16x2 r557, r531, r548;
}
{
neg.f16x2 r560, r557;
}
{
fma.rn.f16x2 r562, r534, r546, r560;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r543;
mov.b32 r566, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r543;
mov.b32 r568, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
mov.b32 r570, {low, high};
}
{
mul.f16x2 r571, r568, r570;
}
{
mul.f16x2 r574, r543, r566;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r543;
mov.b32 r577, {high, low};
}
{
fma.rn.f16x2 r579, r571, r577, r574;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r579;
mov.b32 r583, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r579;
mov.b32 r585, {high, high};
}
{
mul.f16x2 r587, r528, r585;
}
{
fma.rn.f16x2 r590, r525, r583, r587;
}
{
mul.f16x2 r594, r525, r585;
}
{
neg.f16x2 r597, r594;
}
{
fma.rn.f16x2 r599, r528, r583, r597;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r543;
mov.b32 r603, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r543;
mov.b32 r605, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
mov.b32 r607, {low, high};
}
{
mul.f16x2 r608, r605, r607;
}
{
mul.f16x2 r611, r579, r603;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r579;
mov.b32 r614, {high, low};
}
{
fma.rn.f16x2 r616, r608, r614, r611;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r616;
mov.b32 r620, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r616;
mov.b32 r622, {high, high};
}
{
mul.f16x2 r624, r540, r622;
}
{
fma.rn.f16x2 r627, r537, r620, r624;
}
{
mul.f16x2 r631, r537, r622;
}
{
neg.f16x2 r634, r631;
}
{
fma.rn.f16x2 r636, r540, r620, r634;
}
and.b32 r710, r695, 504;
add.s32 r711, r689, r710;
barrier.sync 0;
and.b32 r712, r687, 2048;
add.s32 r713, r711, r712;
st.shared.u32 [r713], r519;
st.shared.u32 [r713+4], r522;
st.shared.u32 [r713+512], r553;
st.shared.u32 [r713+516], r562;
st.shared.u32 [r713+1024], r590;
st.shared.u32 [r713+1028], r599;
st.shared.u32 [r713+1536], r627;
st.shared.u32 [r713+1540], r636;
barrier.sync 0;
mad.lo.s32 r714, r708, -24, r713;
ld.shared.u32 r658, [r714];
ld.shared.u32 r661, [r714+4];
ld.shared.u32 r670, [r714+1024];
ld.shared.u32 r673, [r714+1028];
ld.shared.u32 r659, [r714+2048];
ld.shared.u32 r662, [r714+2052];
ld.shared.u32 r671, [r714+3072];
ld.shared.u32 r674, [r714+3076];
{
add.f16x2 %0, r658, r659;
}
{
add.f16x2 %1, r661, r662;
}
{
sub.f16x2 %4, r658, r659;
}
{
sub.f16x2 %5, r661, r662;
}
{
add.f16x2 %2, r670, r671;
}
{
add.f16x2 %3, r673, r674;
}
{
sub.f16x2 %6, r670, r671;
}
{
sub.f16x2 %7, r673, r674;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1026, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<53>;
.reg .b32 r<715>;
.reg .b64 rd<2>;
mov.u32 r681, %tid.y;
shl.b32 r682, r681, 11;
mov.u32 r683, %8;
add.s32 r684, r683, r682;
mov.u32 r685, %tid.x;
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
and.b32 r686, r685, 127;
shl.b32 r687, r685, 4;
and.b32 r688, r687, -2048;
add.s32 r689, r684, r688;
cvt.rn.f32.u32 f41, r686;
mul.f32 f42, f41, 0f3C490FDB;
cos.approx.f32 f1, f42;
sin.approx.f32 f43, f42;
neg.f32 f2, f43;
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
mov.f32 f37, 0fBF800000;
mov.f32 f38, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
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
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
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
and.b32 r690, r687, 2032;
add.s32 r691, r689, r690;
st.shared.v4.f32 [r691], {r27, r61, r98, r135};
barrier.sync 0;
mad.lo.s32 r692, r686, -12, r691;
ld.shared.u32 r166, [r692];
ld.shared.u32 r178, [r692+512];
ld.shared.u32 r167, [r692+1024];
ld.shared.u32 r179, [r692+1536];
barrier.sync 0;
st.shared.v4.f32 [r691], {r30, r70, r107, r144};
barrier.sync 0;
ld.shared.u32 r169, [r692];
ld.shared.u32 r181, [r692+512];
ld.shared.u32 r170, [r692+1024];
ld.shared.u32 r182, [r692+1536];
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
and.b32 r693, r685, 124;
bfe.u32 r694, r685, 2, 5;
shl.b32 r695, r685, 2;
and.b32 r696, r695, 12;
add.s32 r697, r689, r696;
cvt.rn.f32.u32 f44, r694;
mul.f32 f45, f44, 0f3D490FDB;
cos.approx.f32 f11, f45;
sin.approx.f32 f46, f45;
neg.f32 f12, f46;
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
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
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
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
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
and.b32 r698, r687, 1984;
add.s32 r699, r697, r698;
st.shared.u32 [r699], r191;
st.shared.u32 [r699+16], r225;
st.shared.u32 [r699+32], r262;
st.shared.u32 [r699+48], r299;
barrier.sync 0;
mad.lo.s32 r700, r693, -12, r699;
ld.shared.u32 r330, [r700];
ld.shared.u32 r342, [r700+512];
ld.shared.u32 r331, [r700+1024];
ld.shared.u32 r343, [r700+1536];
barrier.sync 0;
st.shared.u32 [r699], r194;
st.shared.u32 [r699+16], r234;
st.shared.u32 [r699+32], r271;
st.shared.u32 [r699+48], r308;
barrier.sync 0;
ld.shared.u32 r333, [r700];
ld.shared.u32 r345, [r700+512];
ld.shared.u32 r334, [r700+1024];
ld.shared.u32 r346, [r700+1536];
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
and.b32 r701, r685, 112;
bfe.u32 r702, r685, 4, 3;
and.b32 r703, r695, 60;
add.s32 r704, r689, r703;
cvt.rn.f32.u32 f47, r702;
mul.f32 f48, f47, 0f3E490FDB;
cos.approx.f32 f21, f48;
sin.approx.f32 f49, f48;
neg.f32 f22, f49;
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
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
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
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
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
and.b32 r705, r687, 1792;
add.s32 r706, r704, r705;
st.shared.u32 [r706], r355;
st.shared.u32 [r706+64], r389;
st.shared.u32 [r706+128], r426;
st.shared.u32 [r706+192], r463;
barrier.sync 0;
mad.lo.s32 r707, r701, -12, r706;
ld.shared.u32 r494, [r707];
ld.shared.u32 r506, [r707+512];
ld.shared.u32 r495, [r707+1024];
ld.shared.u32 r507, [r707+1536];
barrier.sync 0;
st.shared.u32 [r706], r358;
st.shared.u32 [r706+64], r398;
st.shared.u32 [r706+128], r435;
st.shared.u32 [r706+192], r472;
barrier.sync 0;
ld.shared.u32 r497, [r707];
ld.shared.u32 r509, [r707+512];
ld.shared.u32 r498, [r707+1024];
ld.shared.u32 r510, [r707+1536];
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
neg.f16x2 r517, r514;
}
{
add.f16x2 r519, r493, r505;
}
{
add.f16x2 r522, r496, r508;
}
{
sub.f16x2 r525, r493, r505;
}
{
sub.f16x2 r528, r496, r508;
}
{
add.f16x2 r531, r499, r517;
}
{
add.f16x2 r534, r502, r511;
}
{
sub.f16x2 r537, r499, r517;
}
{
sub.f16x2 r540, r502, r511;
}
and.b32 r708, r685, 64;
bfe.u32 r709, r685, 6, 1;
and.b32 r710, r695, 252;
add.s32 r711, r689, r710;
cvt.rn.f32.u32 f50, r709;
mul.f32 f51, f50, 0f3F490FDB;
cos.approx.f32 f31, f51;
sin.approx.f32 f52, f51;
neg.f32 f32, f52;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f31;
cvt.rn.f16.f32 high, f32;
mov.b32 r543, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r543;
mov.b32 r546, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r543;
mov.b32 r548, {high, high};
}
{
mul.f16x2 r550, r534, r548;
}
{
fma.rn.f16x2 r553, r531, r546, r550;
}
{
mul.f16x2 r557, r531, r548;
}
{
neg.f16x2 r560, r557;
}
{
fma.rn.f16x2 r562, r534, r546, r560;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r543;
mov.b32 r566, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r543;
mov.b32 r568, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
mov.b32 r570, {low, high};
}
{
mul.f16x2 r571, r568, r570;
}
{
mul.f16x2 r574, r543, r566;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r543;
mov.b32 r577, {high, low};
}
{
fma.rn.f16x2 r579, r571, r577, r574;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r579;
mov.b32 r583, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r579;
mov.b32 r585, {high, high};
}
{
mul.f16x2 r587, r528, r585;
}
{
fma.rn.f16x2 r590, r525, r583, r587;
}
{
mul.f16x2 r594, r525, r585;
}
{
neg.f16x2 r597, r594;
}
{
fma.rn.f16x2 r599, r528, r583, r597;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r543;
mov.b32 r603, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r543;
mov.b32 r605, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
mov.b32 r607, {low, high};
}
{
mul.f16x2 r608, r605, r607;
}
{
mul.f16x2 r611, r579, r603;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r579;
mov.b32 r614, {high, low};
}
{
fma.rn.f16x2 r616, r608, r614, r611;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r616;
mov.b32 r620, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r616;
mov.b32 r622, {high, high};
}
{
mul.f16x2 r624, r540, r622;
}
{
fma.rn.f16x2 r627, r537, r620, r624;
}
{
mul.f16x2 r631, r537, r622;
}
{
neg.f16x2 r634, r631;
}
{
fma.rn.f16x2 r636, r540, r620, r634;
}
barrier.sync 0;
and.b32 r712, r687, 1024;
add.s32 r713, r711, r712;
st.shared.u32 [r713], r519;
st.shared.u32 [r713+256], r553;
st.shared.u32 [r713+512], r590;
st.shared.u32 [r713+768], r627;
barrier.sync 0;
mad.lo.s32 r714, r708, -12, r713;
ld.shared.u32 r658, [r714];
ld.shared.u32 r670, [r714+512];
ld.shared.u32 r659, [r714+1024];
ld.shared.u32 r671, [r714+1536];
barrier.sync 0;
st.shared.u32 [r713], r522;
st.shared.u32 [r713+256], r562;
st.shared.u32 [r713+512], r599;
st.shared.u32 [r713+768], r636;
barrier.sync 0;
ld.shared.u32 r661, [r714];
ld.shared.u32 r673, [r714+512];
ld.shared.u32 r662, [r714+1024];
ld.shared.u32 r674, [r714+1536];
{
add.f16x2 %0, r658, r659;
}
{
add.f16x2 %1, r661, r662;
}
{
sub.f16x2 %4, r658, r659;
}
{
sub.f16x2 %5, r661, r662;
}
{
add.f16x2 %2, r670, r671;
}
{
add.f16x2 %3, r673, r674;
}
{
sub.f16x2 %6, r670, r671;
}
{
sub.f16x2 %7, r673, r674;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1027, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<73>;
.reg .b32 r<492>;
.reg .b64 rd<2>;
mov.u32 r429, %tid.y;
shl.b32 r430, r429, 12;
mov.u32 r431, %4;
add.s32 r432, r431, r430;
mov.u32 r433, %tid.x;
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
and.b32 r434, r433, 255;
shl.b32 r435, r433, 4;
and.b32 r436, r435, -4096;
add.s32 r437, r432, r436;
cvt.rn.f32.u32 f49, r434;
mul.f32 f50, f49, 0f3C490FDB;
cos.approx.f32 f1, f50;
sin.approx.f32 f51, f50;
neg.f32 f2, f51;
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
and.b32 r438, r435, 4080;
add.s32 r439, r437, r438;
st.shared.v2.f32 [r439], {r1, r4};
st.shared.v2.f32 [r439+8], {r23, r32};
barrier.sync 0;
shl.b32 r440, r433, 3;
and.b32 r441, r440, 2040;
sub.s32 r442, r439, r441;
ld.shared.u32 r54, [r442];
ld.shared.u32 r57, [r442+4];
ld.shared.u32 r55, [r442+2048];
ld.shared.u32 r58, [r442+2052];
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
bfe.u32 r443, r433, 1, 7;
cvt.rn.f32.u32 f52, r443;
mul.f32 f53, f52, 0f3CC90FDB;
cos.approx.f32 f7, f53;
sin.approx.f32 f54, f53;
neg.f32 f8, f54;
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
and.b32 r444, r440, 8;
add.s32 r445, r437, r444;
barrier.sync 0;
and.b32 r446, r435, 4064;
add.s32 r447, r445, r446;
st.shared.u32 [r447], r53;
st.shared.u32 [r447+4], r56;
st.shared.u32 [r447+16], r75;
st.shared.u32 [r447+20], r84;
barrier.sync 0;
and.b32 r448, r440, 2032;
sub.s32 r449, r447, r448;
ld.shared.u32 r106, [r449];
ld.shared.u32 r109, [r449+4];
ld.shared.u32 r107, [r449+2048];
ld.shared.u32 r110, [r449+2052];
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
bfe.u32 r450, r433, 2, 6;
cvt.rn.f32.u32 f55, r450;
mul.f32 f56, f55, 0f3D490FDB;
cos.approx.f32 f13, f56;
sin.approx.f32 f57, f56;
neg.f32 f14, f57;
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
and.b32 r451, r440, 24;
add.s32 r452, r437, r451;
barrier.sync 0;
and.b32 r453, r435, 4032;
add.s32 r454, r452, r453;
st.shared.u32 [r454], r105;
st.shared.u32 [r454+4], r108;
st.shared.u32 [r454+32], r127;
st.shared.u32 [r454+36], r136;
barrier.sync 0;
and.b32 r455, r440, 2016;
sub.s32 r456, r454, r455;
ld.shared.u32 r158, [r456];
ld.shared.u32 r161, [r456+4];
ld.shared.u32 r159, [r456+2048];
ld.shared.u32 r162, [r456+2052];
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
bfe.u32 r457, r433, 3, 5;
cvt.rn.f32.u32 f58, r457;
mul.f32 f59, f58, 0f3DC90FDB;
cos.approx.f32 f19, f59;
sin.approx.f32 f60, f59;
neg.f32 f20, f60;
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
and.b32 r458, r440, 56;
add.s32 r459, r437, r458;
barrier.sync 0;
and.b32 r460, r435, 3968;
add.s32 r461, r459, r460;
st.shared.u32 [r461], r157;
st.shared.u32 [r461+4], r160;
st.shared.u32 [r461+64], r179;
st.shared.u32 [r461+68], r188;
barrier.sync 0;
and.b32 r462, r440, 1984;
sub.s32 r463, r461, r462;
ld.shared.u32 r210, [r463];
ld.shared.u32 r213, [r463+4];
ld.shared.u32 r211, [r463+2048];
ld.shared.u32 r214, [r463+2052];
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
bfe.u32 r464, r433, 4, 4;
cvt.rn.f32.u32 f61, r464;
mul.f32 f62, f61, 0f3E490FDB;
cos.approx.f32 f25, f62;
sin.approx.f32 f63, f62;
neg.f32 f26, f63;
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
and.b32 r465, r440, 120;
add.s32 r466, r437, r465;
barrier.sync 0;
and.b32 r467, r435, 3840;
add.s32 r468, r466, r467;
st.shared.u32 [r468], r209;
st.shared.u32 [r468+4], r212;
st.shared.u32 [r468+128], r231;
st.shared.u32 [r468+132], r240;
barrier.sync 0;
and.b32 r469, r440, 1920;
sub.s32 r470, r468, r469;
ld.shared.u32 r262, [r470];
ld.shared.u32 r265, [r470+4];
ld.shared.u32 r263, [r470+2048];
ld.shared.u32 r266, [r470+2052];
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
bfe.u32 r471, r433, 5, 3;
cvt.rn.f32.u32 f64, r471;
mul.f32 f65, f64, 0f3EC90FDB;
cos.approx.f32 f31, f65;
sin.approx.f32 f66, f65;
neg.f32 f32, f66;
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
and.b32 r472, r440, 248;
add.s32 r473, r437, r472;
barrier.sync 0;
and.b32 r474, r435, 3584;
add.s32 r475, r473, r474;
st.shared.u32 [r475], r261;
st.shared.u32 [r475+4], r264;
st.shared.u32 [r475+256], r283;
st.shared.u32 [r475+260], r292;
barrier.sync 0;
and.b32 r476, r440, 1792;
sub.s32 r477, r475, r476;
ld.shared.u32 r314, [r477];
ld.shared.u32 r317, [r477+4];
ld.shared.u32 r315, [r477+2048];
ld.shared.u32 r318, [r477+2052];
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
bfe.u32 r478, r433, 6, 2;
cvt.rn.f32.u32 f67, r478;
mul.f32 f68, f67, 0f3F490FDB;
cos.approx.f32 f37, f68;
sin.approx.f32 f69, f68;
neg.f32 f38, f69;
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
fma.rn.f16x2 r335, r319, r328, r332;
}
{
mul.f16x2 r339, r319, r330;
}
{
neg.f16x2 r342, r339;
}
{
fma.rn.f16x2 r344, r322, r328, r342;
}
and.b32 r479, r440, 504;
add.s32 r480, r437, r479;
barrier.sync 0;
and.b32 r481, r435, 3072;
add.s32 r482, r480, r481;
st.shared.u32 [r482], r313;
st.shared.u32 [r482+4], r316;
st.shared.u32 [r482+512], r335;
st.shared.u32 [r482+516], r344;
barrier.sync 0;
and.b32 r483, r440, 1536;
sub.s32 r484, r482, r483;
ld.shared.u32 r366, [r484];
ld.shared.u32 r369, [r484+4];
ld.shared.u32 r367, [r484+2048];
ld.shared.u32 r370, [r484+2052];
{
add.f16x2 r365, r366, r367;
}
{
add.f16x2 r368, r369, r370;
}
{
sub.f16x2 r371, r366, r367;
}
{
sub.f16x2 r374, r369, r370;
}
bfe.u32 r485, r433, 7, 1;
cvt.rn.f32.u32 f70, r485;
mul.f32 f71, f70, 0f3FC90FDB;
cos.approx.f32 f43, f71;
sin.approx.f32 f72, f71;
neg.f32 f44, f72;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r377, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r377;
mov.b32 r380, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r377;
mov.b32 r382, {high, high};
}
{
mul.f16x2 r384, r374, r382;
}
{
fma.rn.f16x2 r387, r371, r380, r384;
}
{
mul.f16x2 r391, r371, r382;
}
{
neg.f16x2 r394, r391;
}
{
fma.rn.f16x2 r396, r374, r380, r394;
}
and.b32 r486, r440, 1016;
add.s32 r487, r437, r486;
barrier.sync 0;
and.b32 r488, r435, 2048;
add.s32 r489, r487, r488;
st.shared.u32 [r489], r365;
st.shared.u32 [r489+4], r368;
st.shared.u32 [r489+1024], r387;
st.shared.u32 [r489+1028], r396;
barrier.sync 0;
and.b32 r490, r440, 1024;
sub.s32 r491, r489, r490;
ld.shared.u32 r418, [r491];
ld.shared.u32 r421, [r491+4];
ld.shared.u32 r419, [r491+2048];
ld.shared.u32 r422, [r491+2052];
{
add.f16x2 %0, r418, r419;
}
{
add.f16x2 %1, r421, r422;
}
{
sub.f16x2 %2, r418, r419;
}
{
sub.f16x2 %3, r421, r422;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1028, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<73>;
.reg .b32 r<492>;
.reg .b64 rd<2>;
mov.u32 r429, %tid.y;
shl.b32 r430, r429, 11;
mov.u32 r431, %4;
add.s32 r432, r431, r430;
mov.u32 r433, %tid.x;
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
and.b32 r434, r433, 255;
shl.b32 r435, r433, 3;
and.b32 r436, r435, -2048;
add.s32 r437, r432, r436;
cvt.rn.f32.u32 f49, r434;
mul.f32 f50, f49, 0f3C490FDB;
cos.approx.f32 f1, f50;
sin.approx.f32 f51, f50;
neg.f32 f2, f51;
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
and.b32 r438, r435, 2040;
add.s32 r439, r437, r438;
st.shared.v2.f32 [r439], {r1, r23};
barrier.sync 0;
shl.b32 r440, r433, 2;
and.b32 r441, r440, 1020;
sub.s32 r442, r439, r441;
ld.shared.u32 r54, [r442];
ld.shared.u32 r55, [r442+1024];
barrier.sync 0;
st.shared.v2.f32 [r439], {r4, r32};
barrier.sync 0;
ld.shared.u32 r57, [r442];
ld.shared.u32 r58, [r442+1024];
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
bfe.u32 r443, r433, 1, 7;
and.b32 r444, r440, 4;
add.s32 r445, r437, r444;
cvt.rn.f32.u32 f52, r443;
mul.f32 f53, f52, 0f3CC90FDB;
cos.approx.f32 f7, f53;
sin.approx.f32 f54, f53;
neg.f32 f8, f54;
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
and.b32 r446, r435, 2032;
add.s32 r447, r445, r446;
st.shared.u32 [r447], r53;
st.shared.u32 [r447+8], r75;
barrier.sync 0;
and.b32 r448, r440, 1016;
sub.s32 r449, r447, r448;
ld.shared.u32 r106, [r449];
ld.shared.u32 r107, [r449+1024];
barrier.sync 0;
st.shared.u32 [r447], r56;
st.shared.u32 [r447+8], r84;
barrier.sync 0;
ld.shared.u32 r109, [r449];
ld.shared.u32 r110, [r449+1024];
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
bfe.u32 r450, r433, 2, 6;
and.b32 r451, r440, 12;
add.s32 r452, r437, r451;
cvt.rn.f32.u32 f55, r450;
mul.f32 f56, f55, 0f3D490FDB;
cos.approx.f32 f13, f56;
sin.approx.f32 f57, f56;
neg.f32 f14, f57;
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
and.b32 r453, r435, 2016;
add.s32 r454, r452, r453;
st.shared.u32 [r454], r105;
st.shared.u32 [r454+16], r127;
barrier.sync 0;
and.b32 r455, r440, 1008;
sub.s32 r456, r454, r455;
ld.shared.u32 r158, [r456];
ld.shared.u32 r159, [r456+1024];
barrier.sync 0;
st.shared.u32 [r454], r108;
st.shared.u32 [r454+16], r136;
barrier.sync 0;
ld.shared.u32 r161, [r456];
ld.shared.u32 r162, [r456+1024];
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
bfe.u32 r457, r433, 3, 5;
and.b32 r458, r440, 28;
add.s32 r459, r437, r458;
cvt.rn.f32.u32 f58, r457;
mul.f32 f59, f58, 0f3DC90FDB;
cos.approx.f32 f19, f59;
sin.approx.f32 f60, f59;
neg.f32 f20, f60;
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
and.b32 r460, r435, 1984;
add.s32 r461, r459, r460;
st.shared.u32 [r461], r157;
st.shared.u32 [r461+32], r179;
barrier.sync 0;
and.b32 r462, r440, 992;
sub.s32 r463, r461, r462;
ld.shared.u32 r210, [r463];
ld.shared.u32 r211, [r463+1024];
barrier.sync 0;
st.shared.u32 [r461], r160;
st.shared.u32 [r461+32], r188;
barrier.sync 0;
ld.shared.u32 r213, [r463];
ld.shared.u32 r214, [r463+1024];
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
bfe.u32 r464, r433, 4, 4;
and.b32 r465, r440, 60;
add.s32 r466, r437, r465;
cvt.rn.f32.u32 f61, r464;
mul.f32 f62, f61, 0f3E490FDB;
cos.approx.f32 f25, f62;
sin.approx.f32 f63, f62;
neg.f32 f26, f63;
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
and.b32 r467, r435, 1920;
add.s32 r468, r466, r467;
st.shared.u32 [r468], r209;
st.shared.u32 [r468+64], r231;
barrier.sync 0;
and.b32 r469, r440, 960;
sub.s32 r470, r468, r469;
ld.shared.u32 r262, [r470];
ld.shared.u32 r263, [r470+1024];
barrier.sync 0;
st.shared.u32 [r468], r212;
st.shared.u32 [r468+64], r240;
barrier.sync 0;
ld.shared.u32 r265, [r470];
ld.shared.u32 r266, [r470+1024];
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
bfe.u32 r471, r433, 5, 3;
and.b32 r472, r440, 124;
add.s32 r473, r437, r472;
cvt.rn.f32.u32 f64, r471;
mul.f32 f65, f64, 0f3EC90FDB;
cos.approx.f32 f31, f65;
sin.approx.f32 f66, f65;
neg.f32 f32, f66;
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
and.b32 r474, r435, 1792;
add.s32 r475, r473, r474;
st.shared.u32 [r475], r261;
st.shared.u32 [r475+128], r283;
barrier.sync 0;
and.b32 r476, r440, 896;
sub.s32 r477, r475, r476;
ld.shared.u32 r314, [r477];
ld.shared.u32 r315, [r477+1024];
barrier.sync 0;
st.shared.u32 [r475], r264;
st.shared.u32 [r475+128], r292;
barrier.sync 0;
ld.shared.u32 r317, [r477];
ld.shared.u32 r318, [r477+1024];
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
bfe.u32 r478, r433, 6, 2;
and.b32 r479, r440, 252;
add.s32 r480, r437, r479;
cvt.rn.f32.u32 f67, r478;
mul.f32 f68, f67, 0f3F490FDB;
cos.approx.f32 f37, f68;
sin.approx.f32 f69, f68;
neg.f32 f38, f69;
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
fma.rn.f16x2 r335, r319, r328, r332;
}
{
mul.f16x2 r339, r319, r330;
}
{
neg.f16x2 r342, r339;
}
{
fma.rn.f16x2 r344, r322, r328, r342;
}
barrier.sync 0;
and.b32 r481, r435, 1536;
add.s32 r482, r480, r481;
st.shared.u32 [r482], r313;
st.shared.u32 [r482+256], r335;
barrier.sync 0;
and.b32 r483, r440, 768;
sub.s32 r484, r482, r483;
ld.shared.u32 r366, [r484];
ld.shared.u32 r367, [r484+1024];
barrier.sync 0;
st.shared.u32 [r482], r316;
st.shared.u32 [r482+256], r344;
barrier.sync 0;
ld.shared.u32 r369, [r484];
ld.shared.u32 r370, [r484+1024];
{
add.f16x2 r365, r366, r367;
}
{
add.f16x2 r368, r369, r370;
}
{
sub.f16x2 r371, r366, r367;
}
{
sub.f16x2 r374, r369, r370;
}
bfe.u32 r485, r433, 7, 1;
and.b32 r486, r440, 508;
add.s32 r487, r437, r486;
cvt.rn.f32.u32 f70, r485;
mul.f32 f71, f70, 0f3FC90FDB;
cos.approx.f32 f43, f71;
sin.approx.f32 f72, f71;
neg.f32 f44, f72;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r377, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r377;
mov.b32 r380, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r377;
mov.b32 r382, {high, high};
}
{
mul.f16x2 r384, r374, r382;
}
{
fma.rn.f16x2 r387, r371, r380, r384;
}
{
mul.f16x2 r391, r371, r382;
}
{
neg.f16x2 r394, r391;
}
{
fma.rn.f16x2 r396, r374, r380, r394;
}
barrier.sync 0;
and.b32 r488, r435, 1024;
add.s32 r489, r487, r488;
st.shared.u32 [r489], r365;
st.shared.u32 [r489+512], r387;
barrier.sync 0;
and.b32 r490, r440, 512;
sub.s32 r491, r489, r490;
ld.shared.u32 r418, [r491];
ld.shared.u32 r419, [r491+1024];
barrier.sync 0;
st.shared.u32 [r489], r368;
st.shared.u32 [r489+512], r396;
barrier.sync 0;
ld.shared.u32 r421, [r491];
ld.shared.u32 r422, [r491+1024];
{
add.f16x2 %0, r418, r419;
}
{
add.f16x2 %1, r421, r422;
}
{
sub.f16x2 %2, r418, r419;
}
{
sub.f16x2 %3, r421, r422;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)));
};


#endif
