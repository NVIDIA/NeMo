#ifndef CUFFTDX_FFT_1296_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_1296_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1136, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<164>;
.reg .b32 r<1590>;
.reg .b64 rd<8>;
mov.u32 r1565, %tid.y;
shl.b32 r1566, r1565, 1;
mov.u32 r1567, %12;
mad.lo.s32 r1568, r1566, 5184, r1567;
mov.u32 r1569, %tid.x;
mov.f32 f140, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1, {low, high};
}
mov.f32 f134, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f134;
cvt.rn.f16.f32 high, f134;
mov.b32 r2, {low, high};
}
{
add.f16x2 r3, %17, %21;
}
{
add.f16x2 r6, %13, r3;
}
{
add.f16x2 r9, %18, %22;
}
{
add.f16x2 r12, %14, r9;
}
{
add.f16x2 r15, %17, %21;
}
{
mul.f16x2 r18, r15, r1;
}
{
add.f16x2 r21, %13, r18;
}
{
sub.f16x2 r24, %18, %22;
}
{
mul.f16x2 r27, r24, r2;
}
{
add.f16x2 r30, r21, r27;
}
{
add.f16x2 r33, %17, %21;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %13, r36;
}
{
sub.f16x2 r42, %18, %22;
}
{
mul.f16x2 r45, r42, r2;
}
{
sub.f16x2 r48, r39, r45;
}
{
add.f16x2 r51, %18, %22;
}
{
mul.f16x2 r54, r51, r1;
}
{
add.f16x2 r57, %14, r54;
}
{
sub.f16x2 r60, %17, %21;
}
{
mul.f16x2 r63, r60, r2;
}
{
sub.f16x2 r66, r57, r63;
}
{
add.f16x2 r69, %18, %22;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %14, r72;
}
{
sub.f16x2 r78, %17, %21;
}
{
mul.f16x2 r81, r78, r2;
}
{
add.f16x2 r84, r75, r81;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f134;
cvt.rn.f16.f32 high, f134;
mov.b32 r88, {low, high};
}
{
add.f16x2 r89, %19, %23;
}
{
add.f16x2 r92, %15, r89;
}
{
add.f16x2 r95, %20, %24;
}
{
add.f16x2 r98, %16, r95;
}
{
add.f16x2 r101, %19, %23;
}
{
mul.f16x2 r104, r101, r87;
}
{
add.f16x2 r107, %15, r104;
}
{
sub.f16x2 r110, %20, %24;
}
{
mul.f16x2 r113, r110, r88;
}
{
add.f16x2 r116, r107, r113;
}
{
add.f16x2 r119, %19, %23;
}
{
mul.f16x2 r122, r119, r87;
}
{
add.f16x2 r125, %15, r122;
}
{
sub.f16x2 r128, %20, %24;
}
{
mul.f16x2 r131, r128, r88;
}
{
sub.f16x2 r134, r125, r131;
}
{
add.f16x2 r137, %20, %24;
}
{
mul.f16x2 r140, r137, r87;
}
{
add.f16x2 r143, %16, r140;
}
{
sub.f16x2 r146, %19, %23;
}
{
mul.f16x2 r149, r146, r88;
}
{
sub.f16x2 r152, r143, r149;
}
{
add.f16x2 r155, %20, %24;
}
{
mul.f16x2 r158, r155, r87;
}
{
add.f16x2 r161, %16, r158;
}
{
sub.f16x2 r164, %19, %23;
}
{
mul.f16x2 r167, r164, r88;
}
{
add.f16x2 r170, r161, r167;
}
mov.f32 f136, 0f3F000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f136;
cvt.rn.f16.f32 high, f136;
mov.b32 r173, {low, high};
}
mov.f32 f142, 0f3F5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r174, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r175, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r176, {low, high};
}
mov.f32 f123, 0fBF800000;
{
mul.f16x2 r183, r116, r173;
}
{
mul.f16x2 r186, r152, r174;
}
{
sub.f16x2 r189, r183, r186;
}
{
mul.f16x2 r192, r116, r174;
}
{
fma.rn.f16x2 r195, r152, r173, r192;
}
{
mul.f16x2 r199, r134, r175;
}
{
mul.f16x2 r202, r170, r176;
}
{
sub.f16x2 r205, r199, r202;
}
{
mul.f16x2 r208, r134, r176;
}
{
fma.rn.f16x2 r211, r170, r175, r208;
}
{
add.f16x2 r215, r6, r92;
}
{
add.f16x2 r218, r12, r98;
}
{
sub.f16x2 r221, r6, r92;
}
{
sub.f16x2 r224, r12, r98;
}
{
add.f16x2 r227, r30, r189;
}
{
add.f16x2 r230, r66, r195;
}
{
sub.f16x2 r233, r30, r189;
}
{
sub.f16x2 r236, r66, r195;
}
{
add.f16x2 r239, r48, r205;
}
{
add.f16x2 r242, r84, r211;
}
{
sub.f16x2 r245, r48, r205;
}
{
sub.f16x2 r248, r84, r211;
}
shr.u32 r1570, r1569, 3;
mul.wide.u32 rd2, r1570, 159072863;
shr.u64 rd3, rd2, 32;
cvt.u32.u64 r1571, rd3;
mul.lo.s32 r1572, r1571, 216;
sub.s32 r1573, r1569, r1572;
shl.b32 r1574, r1571, 1;
mad.lo.s32 r1575, r1574, 5184, r1568;
cvt.rn.f32.u32 f155, r1573;
mul.f32 f156, f155, 0f3B9EDD1F;
cos.approx.f32 f29, f156;
sin.approx.f32 f157, f156;
neg.f32 f30, f157;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f29;
cvt.rn.f16.f32 high, f30;
mov.b32 r251, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r254, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r256, {high, high};
}
{
mul.f16x2 r258, r230, r256;
}
{
fma.rn.f16x2 r261, r227, r254, r258;
}
{
mul.f16x2 r265, r227, r256;
}
{
neg.f16x2 r268, r265;
}
{
fma.rn.f16x2 r270, r230, r254, r268;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r274, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r276, {high, high};
}
mov.f32 f124, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r278, {low, high};
}
{
mul.f16x2 r279, r276, r278;
}
{
mul.f16x2 r282, r251, r274;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r285, {high, low};
}
{
fma.rn.f16x2 r287, r279, r285, r282;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r287;
mov.b32 r291, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r287;
mov.b32 r293, {high, high};
}
{
mul.f16x2 r295, r242, r293;
}
{
fma.rn.f16x2 r298, r239, r291, r295;
}
{
mul.f16x2 r302, r239, r293;
}
{
neg.f16x2 r305, r302;
}
{
fma.rn.f16x2 r307, r242, r291, r305;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r311, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r313, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r315, {low, high};
}
{
mul.f16x2 r316, r313, r315;
}
{
mul.f16x2 r319, r287, r311;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r287;
mov.b32 r322, {high, low};
}
{
fma.rn.f16x2 r324, r316, r322, r319;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r324;
mov.b32 r328, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r324;
mov.b32 r330, {high, high};
}
{
mul.f16x2 r332, r224, r330;
}
{
fma.rn.f16x2 r335, r221, r328, r332;
}
{
mul.f16x2 r339, r221, r330;
}
{
neg.f16x2 r342, r339;
}
{
fma.rn.f16x2 r344, r224, r328, r342;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r348, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r350, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r352, {low, high};
}
{
mul.f16x2 r353, r350, r352;
}
{
mul.f16x2 r356, r324, r348;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r324;
mov.b32 r359, {high, low};
}
{
fma.rn.f16x2 r361, r353, r359, r356;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r361;
mov.b32 r365, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r361;
mov.b32 r367, {high, high};
}
{
mul.f16x2 r369, r236, r367;
}
{
fma.rn.f16x2 r372, r233, r365, r369;
}
{
mul.f16x2 r376, r233, r367;
}
{
neg.f16x2 r379, r376;
}
{
fma.rn.f16x2 r381, r236, r365, r379;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r385, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r387, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r389, {low, high};
}
{
mul.f16x2 r390, r387, r389;
}
{
mul.f16x2 r393, r361, r385;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r361;
mov.b32 r396, {high, low};
}
{
fma.rn.f16x2 r398, r390, r396, r393;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r398;
mov.b32 r402, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r398;
mov.b32 r404, {high, high};
}
{
mul.f16x2 r406, r248, r404;
}
{
fma.rn.f16x2 r409, r245, r402, r406;
}
{
mul.f16x2 r413, r245, r404;
}
{
neg.f16x2 r416, r413;
}
{
fma.rn.f16x2 r418, r248, r402, r416;
}
barrier.sync 0;
mad.lo.s32 r1576, r1573, 48, r1575;
st.shared.v2.f32 [r1576], {r215, r218};
st.shared.v2.f32 [r1576+8], {r261, r270};
st.shared.v2.f32 [r1576+16], {r298, r307};
st.shared.v2.f32 [r1576+24], {r335, r344};
st.shared.v2.f32 [r1576+32], {r372, r381};
st.shared.v2.f32 [r1576+40], {r409, r418};
barrier.sync 0;
mad.lo.s32 r1577, r1573, -40, r1576;
ld.shared.u32 r445, [r1577];
ld.shared.u32 r451, [r1577+4];
ld.shared.u32 r531, [r1577+1728];
ld.shared.u32 r537, [r1577+1732];
ld.shared.u32 r442, [r1577+3456];
ld.shared.u32 r448, [r1577+3460];
ld.shared.u32 r528, [r1577+5184];
ld.shared.u32 r534, [r1577+5188];
ld.shared.u32 r443, [r1577+6912];
ld.shared.u32 r449, [r1577+6916];
ld.shared.u32 r529, [r1577+8640];
ld.shared.u32 r535, [r1577+8644];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r439, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f134;
cvt.rn.f16.f32 high, f134;
mov.b32 r440, {low, high};
}
{
add.f16x2 r441, r442, r443;
}
{
add.f16x2 r444, r445, r441;
}
{
add.f16x2 r447, r448, r449;
}
{
add.f16x2 r450, r451, r447;
}
{
add.f16x2 r453, r442, r443;
}
{
mul.f16x2 r456, r453, r439;
}
{
add.f16x2 r459, r445, r456;
}
{
sub.f16x2 r462, r448, r449;
}
{
mul.f16x2 r465, r462, r440;
}
{
add.f16x2 r468, r459, r465;
}
{
add.f16x2 r471, r442, r443;
}
{
mul.f16x2 r474, r471, r439;
}
{
add.f16x2 r477, r445, r474;
}
{
sub.f16x2 r480, r448, r449;
}
{
mul.f16x2 r483, r480, r440;
}
{
sub.f16x2 r486, r477, r483;
}
{
add.f16x2 r489, r448, r449;
}
{
mul.f16x2 r492, r489, r439;
}
{
add.f16x2 r495, r451, r492;
}
{
sub.f16x2 r498, r442, r443;
}
{
mul.f16x2 r501, r498, r440;
}
{
sub.f16x2 r504, r495, r501;
}
{
add.f16x2 r507, r448, r449;
}
{
mul.f16x2 r510, r507, r439;
}
{
add.f16x2 r513, r451, r510;
}
{
sub.f16x2 r516, r442, r443;
}
{
mul.f16x2 r519, r516, r440;
}
{
add.f16x2 r522, r513, r519;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r525, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f134;
cvt.rn.f16.f32 high, f134;
mov.b32 r526, {low, high};
}
{
add.f16x2 r527, r528, r529;
}
{
add.f16x2 r530, r531, r527;
}
{
add.f16x2 r533, r534, r535;
}
{
add.f16x2 r536, r537, r533;
}
{
add.f16x2 r539, r528, r529;
}
{
mul.f16x2 r542, r539, r525;
}
{
add.f16x2 r545, r531, r542;
}
{
sub.f16x2 r548, r534, r535;
}
{
mul.f16x2 r551, r548, r526;
}
{
add.f16x2 r554, r545, r551;
}
{
add.f16x2 r557, r528, r529;
}
{
mul.f16x2 r560, r557, r525;
}
{
add.f16x2 r563, r531, r560;
}
{
sub.f16x2 r566, r534, r535;
}
{
mul.f16x2 r569, r566, r526;
}
{
sub.f16x2 r572, r563, r569;
}
{
add.f16x2 r575, r534, r535;
}
{
mul.f16x2 r578, r575, r525;
}
{
add.f16x2 r581, r537, r578;
}
{
sub.f16x2 r584, r528, r529;
}
{
mul.f16x2 r587, r584, r526;
}
{
sub.f16x2 r590, r581, r587;
}
{
add.f16x2 r593, r534, r535;
}
{
mul.f16x2 r596, r593, r525;
}
{
add.f16x2 r599, r537, r596;
}
{
sub.f16x2 r602, r528, r529;
}
{
mul.f16x2 r605, r602, r526;
}
{
add.f16x2 r608, r599, r605;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f136;
cvt.rn.f16.f32 high, f136;
mov.b32 r611, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r612, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r613, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r614, {low, high};
}
{
mul.f16x2 r621, r554, r611;
}
{
mul.f16x2 r624, r590, r612;
}
{
sub.f16x2 r627, r621, r624;
}
{
mul.f16x2 r630, r554, r612;
}
{
fma.rn.f16x2 r633, r590, r611, r630;
}
{
mul.f16x2 r637, r572, r613;
}
{
mul.f16x2 r640, r608, r614;
}
{
sub.f16x2 r643, r637, r640;
}
{
mul.f16x2 r646, r572, r614;
}
{
fma.rn.f16x2 r649, r608, r613, r646;
}
{
add.f16x2 r653, r444, r530;
}
{
add.f16x2 r656, r450, r536;
}
{
sub.f16x2 r659, r444, r530;
}
{
sub.f16x2 r662, r450, r536;
}
{
add.f16x2 r665, r468, r627;
}
{
add.f16x2 r668, r504, r633;
}
{
sub.f16x2 r671, r468, r627;
}
{
sub.f16x2 r674, r504, r633;
}
{
add.f16x2 r677, r486, r643;
}
{
add.f16x2 r680, r522, r649;
}
{
sub.f16x2 r683, r486, r643;
}
{
sub.f16x2 r686, r522, r649;
}
mul.wide.u32 rd4, r1573, -1431655765;
shr.u64 rd5, rd4, 34;
cvt.u32.u64 r1578, rd5;
cvt.rn.f32.u32 f158, r1578;
mul.f32 f159, f158, 0f3CEE4BAE;
cos.approx.f32 f71, f159;
sin.approx.f32 f160, f159;
neg.f32 f72, f160;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f71;
cvt.rn.f16.f32 high, f72;
mov.b32 r689, {low, high};
}
mul.lo.s32 r1579, r1578, 6;
sub.s32 r1580, r1573, r1579;
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r692, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r694, {high, high};
}
{
mul.f16x2 r696, r668, r694;
}
{
fma.rn.f16x2 r699, r665, r692, r696;
}
{
mul.f16x2 r703, r665, r694;
}
{
neg.f16x2 r706, r703;
}
{
fma.rn.f16x2 r708, r668, r692, r706;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r712, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r714, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r716, {low, high};
}
{
mul.f16x2 r717, r714, r716;
}
{
mul.f16x2 r720, r689, r712;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r723, {high, low};
}
{
fma.rn.f16x2 r725, r717, r723, r720;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r725;
mov.b32 r729, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r725;
mov.b32 r731, {high, high};
}
{
mul.f16x2 r733, r680, r731;
}
{
fma.rn.f16x2 r736, r677, r729, r733;
}
{
mul.f16x2 r740, r677, r731;
}
{
neg.f16x2 r743, r740;
}
{
fma.rn.f16x2 r745, r680, r729, r743;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r749, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r751, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r753, {low, high};
}
{
mul.f16x2 r754, r751, r753;
}
{
mul.f16x2 r757, r725, r749;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r725;
mov.b32 r760, {high, low};
}
{
fma.rn.f16x2 r762, r754, r760, r757;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r762;
mov.b32 r766, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r762;
mov.b32 r768, {high, high};
}
{
mul.f16x2 r770, r662, r768;
}
{
fma.rn.f16x2 r773, r659, r766, r770;
}
{
mul.f16x2 r777, r659, r768;
}
{
neg.f16x2 r780, r777;
}
{
fma.rn.f16x2 r782, r662, r766, r780;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r786, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r788, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r790, {low, high};
}
{
mul.f16x2 r791, r788, r790;
}
{
mul.f16x2 r794, r762, r786;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r762;
mov.b32 r797, {high, low};
}
{
fma.rn.f16x2 r799, r791, r797, r794;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r799;
mov.b32 r803, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r799;
mov.b32 r805, {high, high};
}
{
mul.f16x2 r807, r674, r805;
}
{
fma.rn.f16x2 r810, r671, r803, r807;
}
{
mul.f16x2 r814, r671, r805;
}
{
neg.f16x2 r817, r814;
}
{
fma.rn.f16x2 r819, r674, r803, r817;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r823, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r825, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r827, {low, high};
}
{
mul.f16x2 r828, r825, r827;
}
{
mul.f16x2 r831, r799, r823;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r799;
mov.b32 r834, {high, low};
}
{
fma.rn.f16x2 r836, r828, r834, r831;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r836;
mov.b32 r840, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r836;
mov.b32 r842, {high, high};
}
{
mul.f16x2 r844, r686, r842;
}
{
fma.rn.f16x2 r847, r683, r840, r844;
}
{
mul.f16x2 r851, r683, r842;
}
{
neg.f16x2 r854, r851;
}
{
fma.rn.f16x2 r856, r686, r840, r854;
}
shl.b32 r1581, r1580, 3;
add.s32 r1582, r1575, r1581;
barrier.sync 0;
mad.lo.s32 r1583, r1578, 288, r1582;
st.shared.u32 [r1583], r653;
st.shared.u32 [r1583+4], r656;
st.shared.u32 [r1583+48], r699;
st.shared.u32 [r1583+52], r708;
st.shared.u32 [r1583+96], r736;
st.shared.u32 [r1583+100], r745;
st.shared.u32 [r1583+144], r773;
st.shared.u32 [r1583+148], r782;
st.shared.u32 [r1583+192], r810;
st.shared.u32 [r1583+196], r819;
st.shared.u32 [r1583+240], r847;
st.shared.u32 [r1583+244], r856;
barrier.sync 0;
ld.shared.u32 r883, [r1577];
ld.shared.u32 r889, [r1577+4];
ld.shared.u32 r969, [r1577+1728];
ld.shared.u32 r975, [r1577+1732];
ld.shared.u32 r880, [r1577+3456];
ld.shared.u32 r886, [r1577+3460];
ld.shared.u32 r966, [r1577+5184];
ld.shared.u32 r972, [r1577+5188];
ld.shared.u32 r881, [r1577+6912];
ld.shared.u32 r887, [r1577+6916];
ld.shared.u32 r967, [r1577+8640];
ld.shared.u32 r973, [r1577+8644];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r877, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f134;
cvt.rn.f16.f32 high, f134;
mov.b32 r878, {low, high};
}
{
add.f16x2 r879, r880, r881;
}
{
add.f16x2 r882, r883, r879;
}
{
add.f16x2 r885, r886, r887;
}
{
add.f16x2 r888, r889, r885;
}
{
add.f16x2 r891, r880, r881;
}
{
mul.f16x2 r894, r891, r877;
}
{
add.f16x2 r897, r883, r894;
}
{
sub.f16x2 r900, r886, r887;
}
{
mul.f16x2 r903, r900, r878;
}
{
add.f16x2 r906, r897, r903;
}
{
add.f16x2 r909, r880, r881;
}
{
mul.f16x2 r912, r909, r877;
}
{
add.f16x2 r915, r883, r912;
}
{
sub.f16x2 r918, r886, r887;
}
{
mul.f16x2 r921, r918, r878;
}
{
sub.f16x2 r924, r915, r921;
}
{
add.f16x2 r927, r886, r887;
}
{
mul.f16x2 r930, r927, r877;
}
{
add.f16x2 r933, r889, r930;
}
{
sub.f16x2 r936, r880, r881;
}
{
mul.f16x2 r939, r936, r878;
}
{
sub.f16x2 r942, r933, r939;
}
{
add.f16x2 r945, r886, r887;
}
{
mul.f16x2 r948, r945, r877;
}
{
add.f16x2 r951, r889, r948;
}
{
sub.f16x2 r954, r880, r881;
}
{
mul.f16x2 r957, r954, r878;
}
{
add.f16x2 r960, r951, r957;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f134;
cvt.rn.f16.f32 high, f134;
mov.b32 r964, {low, high};
}
{
add.f16x2 r965, r966, r967;
}
{
add.f16x2 r968, r969, r965;
}
{
add.f16x2 r971, r972, r973;
}
{
add.f16x2 r974, r975, r971;
}
{
add.f16x2 r977, r966, r967;
}
{
mul.f16x2 r980, r977, r963;
}
{
add.f16x2 r983, r969, r980;
}
{
sub.f16x2 r986, r972, r973;
}
{
mul.f16x2 r989, r986, r964;
}
{
add.f16x2 r992, r983, r989;
}
{
add.f16x2 r995, r966, r967;
}
{
mul.f16x2 r998, r995, r963;
}
{
add.f16x2 r1001, r969, r998;
}
{
sub.f16x2 r1004, r972, r973;
}
{
mul.f16x2 r1007, r1004, r964;
}
{
sub.f16x2 r1010, r1001, r1007;
}
{
add.f16x2 r1013, r972, r973;
}
{
mul.f16x2 r1016, r1013, r963;
}
{
add.f16x2 r1019, r975, r1016;
}
{
sub.f16x2 r1022, r966, r967;
}
{
mul.f16x2 r1025, r1022, r964;
}
{
sub.f16x2 r1028, r1019, r1025;
}
{
add.f16x2 r1031, r972, r973;
}
{
mul.f16x2 r1034, r1031, r963;
}
{
add.f16x2 r1037, r975, r1034;
}
{
sub.f16x2 r1040, r966, r967;
}
{
mul.f16x2 r1043, r1040, r964;
}
{
add.f16x2 r1046, r1037, r1043;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f136;
cvt.rn.f16.f32 high, f136;
mov.b32 r1049, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r1050, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1051, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r1052, {low, high};
}
{
mul.f16x2 r1059, r992, r1049;
}
{
mul.f16x2 r1062, r1028, r1050;
}
{
sub.f16x2 r1065, r1059, r1062;
}
{
mul.f16x2 r1068, r992, r1050;
}
{
fma.rn.f16x2 r1071, r1028, r1049, r1068;
}
{
mul.f16x2 r1075, r1010, r1051;
}
{
mul.f16x2 r1078, r1046, r1052;
}
{
sub.f16x2 r1081, r1075, r1078;
}
{
mul.f16x2 r1084, r1010, r1052;
}
{
fma.rn.f16x2 r1087, r1046, r1051, r1084;
}
{
add.f16x2 r1091, r882, r968;
}
{
add.f16x2 r1094, r888, r974;
}
{
sub.f16x2 r1097, r882, r968;
}
{
sub.f16x2 r1100, r888, r974;
}
{
add.f16x2 r1103, r906, r1065;
}
{
add.f16x2 r1106, r942, r1071;
}
{
sub.f16x2 r1109, r906, r1065;
}
{
sub.f16x2 r1112, r942, r1071;
}
{
add.f16x2 r1115, r924, r1081;
}
{
add.f16x2 r1118, r960, r1087;
}
{
sub.f16x2 r1121, r924, r1081;
}
{
sub.f16x2 r1124, r960, r1087;
}
mul.wide.u32 rd6, r1573, 954437177;
shr.u64 rd7, rd6, 35;
cvt.u32.u64 r1584, rd7;
cvt.rn.f32.u32 f161, r1584;
mul.f32 f162, f161, 0f3E32B8C2;
cos.approx.f32 f113, f162;
sin.approx.f32 f163, f162;
neg.f32 f114, f163;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f113;
cvt.rn.f16.f32 high, f114;
mov.b32 r1127, {low, high};
}
mul.lo.s32 r1585, r1584, 36;
sub.s32 r1586, r1573, r1585;
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1130, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1132, {high, high};
}
{
mul.f16x2 r1134, r1106, r1132;
}
{
fma.rn.f16x2 r1137, r1103, r1130, r1134;
}
{
mul.f16x2 r1141, r1103, r1132;
}
{
neg.f16x2 r1144, r1141;
}
{
fma.rn.f16x2 r1146, r1106, r1130, r1144;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1150, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1152, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r1154, {low, high};
}
{
mul.f16x2 r1155, r1152, r1154;
}
{
mul.f16x2 r1158, r1127, r1150;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1161, {high, low};
}
{
fma.rn.f16x2 r1163, r1155, r1161, r1158;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1163;
mov.b32 r1167, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1163;
mov.b32 r1169, {high, high};
}
{
mul.f16x2 r1171, r1118, r1169;
}
{
fma.rn.f16x2 r1174, r1115, r1167, r1171;
}
{
mul.f16x2 r1178, r1115, r1169;
}
{
neg.f16x2 r1181, r1178;
}
{
fma.rn.f16x2 r1183, r1118, r1167, r1181;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1187, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1189, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r1191, {low, high};
}
{
mul.f16x2 r1192, r1189, r1191;
}
{
mul.f16x2 r1195, r1163, r1187;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1163;
mov.b32 r1198, {high, low};
}
{
fma.rn.f16x2 r1200, r1192, r1198, r1195;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1200;
mov.b32 r1204, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1200;
mov.b32 r1206, {high, high};
}
{
mul.f16x2 r1208, r1100, r1206;
}
{
fma.rn.f16x2 r1211, r1097, r1204, r1208;
}
{
mul.f16x2 r1215, r1097, r1206;
}
{
neg.f16x2 r1218, r1215;
}
{
fma.rn.f16x2 r1220, r1100, r1204, r1218;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1224, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1226, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r1228, {low, high};
}
{
mul.f16x2 r1229, r1226, r1228;
}
{
mul.f16x2 r1232, r1200, r1224;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1200;
mov.b32 r1235, {high, low};
}
{
fma.rn.f16x2 r1237, r1229, r1235, r1232;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1237;
mov.b32 r1241, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1237;
mov.b32 r1243, {high, high};
}
{
mul.f16x2 r1245, r1112, r1243;
}
{
fma.rn.f16x2 r1248, r1109, r1241, r1245;
}
{
mul.f16x2 r1252, r1109, r1243;
}
{
neg.f16x2 r1255, r1252;
}
{
fma.rn.f16x2 r1257, r1112, r1241, r1255;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1261, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1263, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r1265, {low, high};
}
{
mul.f16x2 r1266, r1263, r1265;
}
{
mul.f16x2 r1269, r1237, r1261;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1237;
mov.b32 r1272, {high, low};
}
{
fma.rn.f16x2 r1274, r1266, r1272, r1269;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1274;
mov.b32 r1278, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1274;
mov.b32 r1280, {high, high};
}
{
mul.f16x2 r1282, r1124, r1280;
}
{
fma.rn.f16x2 r1285, r1121, r1278, r1282;
}
{
mul.f16x2 r1289, r1121, r1280;
}
{
neg.f16x2 r1292, r1289;
}
{
fma.rn.f16x2 r1294, r1124, r1278, r1292;
}
shl.b32 r1587, r1586, 3;
add.s32 r1588, r1575, r1587;
barrier.sync 0;
mad.lo.s32 r1589, r1584, 1728, r1588;
st.shared.u32 [r1589], r1091;
st.shared.u32 [r1589+4], r1094;
st.shared.u32 [r1589+288], r1137;
st.shared.u32 [r1589+292], r1146;
st.shared.u32 [r1589+576], r1174;
st.shared.u32 [r1589+580], r1183;
st.shared.u32 [r1589+864], r1211;
st.shared.u32 [r1589+868], r1220;
st.shared.u32 [r1589+1152], r1248;
st.shared.u32 [r1589+1156], r1257;
st.shared.u32 [r1589+1440], r1285;
st.shared.u32 [r1589+1444], r1294;
barrier.sync 0;
ld.shared.u32 r1321, [r1577];
ld.shared.u32 r1327, [r1577+4];
ld.shared.u32 r1407, [r1577+1728];
ld.shared.u32 r1413, [r1577+1732];
ld.shared.u32 r1318, [r1577+3456];
ld.shared.u32 r1324, [r1577+3460];
ld.shared.u32 r1404, [r1577+5184];
ld.shared.u32 r1410, [r1577+5188];
ld.shared.u32 r1319, [r1577+6912];
ld.shared.u32 r1325, [r1577+6916];
ld.shared.u32 r1405, [r1577+8640];
ld.shared.u32 r1411, [r1577+8644];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1315, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f134;
cvt.rn.f16.f32 high, f134;
mov.b32 r1316, {low, high};
}
{
add.f16x2 r1317, r1318, r1319;
}
{
add.f16x2 r1320, r1321, r1317;
}
{
add.f16x2 r1323, r1324, r1325;
}
{
add.f16x2 r1326, r1327, r1323;
}
{
add.f16x2 r1329, r1318, r1319;
}
{
mul.f16x2 r1332, r1329, r1315;
}
{
add.f16x2 r1335, r1321, r1332;
}
{
sub.f16x2 r1338, r1324, r1325;
}
{
mul.f16x2 r1341, r1338, r1316;
}
{
add.f16x2 r1344, r1335, r1341;
}
{
add.f16x2 r1347, r1318, r1319;
}
{
mul.f16x2 r1350, r1347, r1315;
}
{
add.f16x2 r1353, r1321, r1350;
}
{
sub.f16x2 r1356, r1324, r1325;
}
{
mul.f16x2 r1359, r1356, r1316;
}
{
sub.f16x2 r1362, r1353, r1359;
}
{
add.f16x2 r1365, r1324, r1325;
}
{
mul.f16x2 r1368, r1365, r1315;
}
{
add.f16x2 r1371, r1327, r1368;
}
{
sub.f16x2 r1374, r1318, r1319;
}
{
mul.f16x2 r1377, r1374, r1316;
}
{
sub.f16x2 r1380, r1371, r1377;
}
{
add.f16x2 r1383, r1324, r1325;
}
{
mul.f16x2 r1386, r1383, r1315;
}
{
add.f16x2 r1389, r1327, r1386;
}
{
sub.f16x2 r1392, r1318, r1319;
}
{
mul.f16x2 r1395, r1392, r1316;
}
{
add.f16x2 r1398, r1389, r1395;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f134;
cvt.rn.f16.f32 high, f134;
mov.b32 r1402, {low, high};
}
{
add.f16x2 r1403, r1404, r1405;
}
{
add.f16x2 r1406, r1407, r1403;
}
{
add.f16x2 r1409, r1410, r1411;
}
{
add.f16x2 r1412, r1413, r1409;
}
{
add.f16x2 r1415, r1404, r1405;
}
{
mul.f16x2 r1418, r1415, r1401;
}
{
add.f16x2 r1421, r1407, r1418;
}
{
sub.f16x2 r1424, r1410, r1411;
}
{
mul.f16x2 r1427, r1424, r1402;
}
{
add.f16x2 r1430, r1421, r1427;
}
{
add.f16x2 r1433, r1404, r1405;
}
{
mul.f16x2 r1436, r1433, r1401;
}
{
add.f16x2 r1439, r1407, r1436;
}
{
sub.f16x2 r1442, r1410, r1411;
}
{
mul.f16x2 r1445, r1442, r1402;
}
{
sub.f16x2 r1448, r1439, r1445;
}
{
add.f16x2 r1451, r1410, r1411;
}
{
mul.f16x2 r1454, r1451, r1401;
}
{
add.f16x2 r1457, r1413, r1454;
}
{
sub.f16x2 r1460, r1404, r1405;
}
{
mul.f16x2 r1463, r1460, r1402;
}
{
sub.f16x2 r1466, r1457, r1463;
}
{
add.f16x2 r1469, r1410, r1411;
}
{
mul.f16x2 r1472, r1469, r1401;
}
{
add.f16x2 r1475, r1413, r1472;
}
{
sub.f16x2 r1478, r1404, r1405;
}
{
mul.f16x2 r1481, r1478, r1402;
}
{
add.f16x2 r1484, r1475, r1481;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f136;
cvt.rn.f16.f32 high, f136;
mov.b32 r1487, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r1488, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1489, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r1490, {low, high};
}
{
mul.f16x2 r1497, r1430, r1487;
}
{
mul.f16x2 r1500, r1466, r1488;
}
{
sub.f16x2 r1503, r1497, r1500;
}
{
mul.f16x2 r1506, r1430, r1488;
}
{
fma.rn.f16x2 r1509, r1466, r1487, r1506;
}
{
mul.f16x2 r1513, r1448, r1489;
}
{
mul.f16x2 r1516, r1484, r1490;
}
{
sub.f16x2 r1519, r1513, r1516;
}
{
mul.f16x2 r1522, r1448, r1490;
}
{
fma.rn.f16x2 r1525, r1484, r1489, r1522;
}
{
add.f16x2 %0, r1320, r1406;
}
{
add.f16x2 %1, r1326, r1412;
}
{
sub.f16x2 %6, r1320, r1406;
}
{
sub.f16x2 %7, r1326, r1412;
}
{
add.f16x2 %2, r1344, r1503;
}
{
add.f16x2 %3, r1380, r1509;
}
{
sub.f16x2 %8, r1344, r1503;
}
{
sub.f16x2 %9, r1380, r1509;
}
{
add.f16x2 %4, r1362, r1519;
}
{
add.f16x2 %5, r1398, r1525;
}
{
sub.f16x2 %10, r1362, r1519;
}
{
sub.f16x2 %11, r1398, r1525;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1137, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<164>;
.reg .b32 r<1588>;
.reg .b64 rd<8>;
mov.u32 r1565, %tid.y;
mov.u32 r1566, %12;
mad.lo.s32 r1567, r1565, 5184, r1566;
mov.u32 r1568, %tid.x;
mov.f32 f140, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1, {low, high};
}
mov.f32 f134, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f134;
cvt.rn.f16.f32 high, f134;
mov.b32 r2, {low, high};
}
{
add.f16x2 r3, %17, %21;
}
{
add.f16x2 r6, %13, r3;
}
{
add.f16x2 r9, %18, %22;
}
{
add.f16x2 r12, %14, r9;
}
{
add.f16x2 r15, %17, %21;
}
{
mul.f16x2 r18, r15, r1;
}
{
add.f16x2 r21, %13, r18;
}
{
sub.f16x2 r24, %18, %22;
}
{
mul.f16x2 r27, r24, r2;
}
{
add.f16x2 r30, r21, r27;
}
{
add.f16x2 r33, %17, %21;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %13, r36;
}
{
sub.f16x2 r42, %18, %22;
}
{
mul.f16x2 r45, r42, r2;
}
{
sub.f16x2 r48, r39, r45;
}
{
add.f16x2 r51, %18, %22;
}
{
mul.f16x2 r54, r51, r1;
}
{
add.f16x2 r57, %14, r54;
}
{
sub.f16x2 r60, %17, %21;
}
{
mul.f16x2 r63, r60, r2;
}
{
sub.f16x2 r66, r57, r63;
}
{
add.f16x2 r69, %18, %22;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %14, r72;
}
{
sub.f16x2 r78, %17, %21;
}
{
mul.f16x2 r81, r78, r2;
}
{
add.f16x2 r84, r75, r81;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f134;
cvt.rn.f16.f32 high, f134;
mov.b32 r88, {low, high};
}
{
add.f16x2 r89, %19, %23;
}
{
add.f16x2 r92, %15, r89;
}
{
add.f16x2 r95, %20, %24;
}
{
add.f16x2 r98, %16, r95;
}
{
add.f16x2 r101, %19, %23;
}
{
mul.f16x2 r104, r101, r87;
}
{
add.f16x2 r107, %15, r104;
}
{
sub.f16x2 r110, %20, %24;
}
{
mul.f16x2 r113, r110, r88;
}
{
add.f16x2 r116, r107, r113;
}
{
add.f16x2 r119, %19, %23;
}
{
mul.f16x2 r122, r119, r87;
}
{
add.f16x2 r125, %15, r122;
}
{
sub.f16x2 r128, %20, %24;
}
{
mul.f16x2 r131, r128, r88;
}
{
sub.f16x2 r134, r125, r131;
}
{
add.f16x2 r137, %20, %24;
}
{
mul.f16x2 r140, r137, r87;
}
{
add.f16x2 r143, %16, r140;
}
{
sub.f16x2 r146, %19, %23;
}
{
mul.f16x2 r149, r146, r88;
}
{
sub.f16x2 r152, r143, r149;
}
{
add.f16x2 r155, %20, %24;
}
{
mul.f16x2 r158, r155, r87;
}
{
add.f16x2 r161, %16, r158;
}
{
sub.f16x2 r164, %19, %23;
}
{
mul.f16x2 r167, r164, r88;
}
{
add.f16x2 r170, r161, r167;
}
mov.f32 f136, 0f3F000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f136;
cvt.rn.f16.f32 high, f136;
mov.b32 r173, {low, high};
}
mov.f32 f142, 0f3F5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r174, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r175, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r176, {low, high};
}
mov.f32 f123, 0fBF800000;
{
mul.f16x2 r183, r116, r173;
}
{
mul.f16x2 r186, r152, r174;
}
{
sub.f16x2 r189, r183, r186;
}
{
mul.f16x2 r192, r116, r174;
}
{
fma.rn.f16x2 r195, r152, r173, r192;
}
{
mul.f16x2 r199, r134, r175;
}
{
mul.f16x2 r202, r170, r176;
}
{
sub.f16x2 r205, r199, r202;
}
{
mul.f16x2 r208, r134, r176;
}
{
fma.rn.f16x2 r211, r170, r175, r208;
}
{
add.f16x2 r215, r6, r92;
}
{
add.f16x2 r218, r12, r98;
}
{
sub.f16x2 r221, r6, r92;
}
{
sub.f16x2 r224, r12, r98;
}
{
add.f16x2 r227, r30, r189;
}
{
add.f16x2 r230, r66, r195;
}
{
sub.f16x2 r233, r30, r189;
}
{
sub.f16x2 r236, r66, r195;
}
{
add.f16x2 r239, r48, r205;
}
{
add.f16x2 r242, r84, r211;
}
{
sub.f16x2 r245, r48, r205;
}
{
sub.f16x2 r248, r84, r211;
}
shr.u32 r1569, r1568, 3;
mul.wide.u32 rd2, r1569, 159072863;
shr.u64 rd3, rd2, 32;
cvt.u32.u64 r1570, rd3;
mul.lo.s32 r1571, r1570, 216;
sub.s32 r1572, r1568, r1571;
mad.lo.s32 r1573, r1570, 5184, r1567;
cvt.rn.f32.u32 f155, r1572;
mul.f32 f156, f155, 0f3B9EDD1F;
cos.approx.f32 f29, f156;
sin.approx.f32 f157, f156;
neg.f32 f30, f157;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f29;
cvt.rn.f16.f32 high, f30;
mov.b32 r251, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r254, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r256, {high, high};
}
{
mul.f16x2 r258, r230, r256;
}
{
fma.rn.f16x2 r261, r227, r254, r258;
}
{
mul.f16x2 r265, r227, r256;
}
{
neg.f16x2 r268, r265;
}
{
fma.rn.f16x2 r270, r230, r254, r268;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r274, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r276, {high, high};
}
mov.f32 f124, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r278, {low, high};
}
{
mul.f16x2 r279, r276, r278;
}
{
mul.f16x2 r282, r251, r274;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r285, {high, low};
}
{
fma.rn.f16x2 r287, r279, r285, r282;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r287;
mov.b32 r291, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r287;
mov.b32 r293, {high, high};
}
{
mul.f16x2 r295, r242, r293;
}
{
fma.rn.f16x2 r298, r239, r291, r295;
}
{
mul.f16x2 r302, r239, r293;
}
{
neg.f16x2 r305, r302;
}
{
fma.rn.f16x2 r307, r242, r291, r305;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r311, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r313, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r315, {low, high};
}
{
mul.f16x2 r316, r313, r315;
}
{
mul.f16x2 r319, r287, r311;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r287;
mov.b32 r322, {high, low};
}
{
fma.rn.f16x2 r324, r316, r322, r319;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r324;
mov.b32 r328, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r324;
mov.b32 r330, {high, high};
}
{
mul.f16x2 r332, r224, r330;
}
{
fma.rn.f16x2 r335, r221, r328, r332;
}
{
mul.f16x2 r339, r221, r330;
}
{
neg.f16x2 r342, r339;
}
{
fma.rn.f16x2 r344, r224, r328, r342;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r348, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r350, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r352, {low, high};
}
{
mul.f16x2 r353, r350, r352;
}
{
mul.f16x2 r356, r324, r348;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r324;
mov.b32 r359, {high, low};
}
{
fma.rn.f16x2 r361, r353, r359, r356;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r361;
mov.b32 r365, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r361;
mov.b32 r367, {high, high};
}
{
mul.f16x2 r369, r236, r367;
}
{
fma.rn.f16x2 r372, r233, r365, r369;
}
{
mul.f16x2 r376, r233, r367;
}
{
neg.f16x2 r379, r376;
}
{
fma.rn.f16x2 r381, r236, r365, r379;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r385, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r387, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r389, {low, high};
}
{
mul.f16x2 r390, r387, r389;
}
{
mul.f16x2 r393, r361, r385;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r361;
mov.b32 r396, {high, low};
}
{
fma.rn.f16x2 r398, r390, r396, r393;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r398;
mov.b32 r402, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r398;
mov.b32 r404, {high, high};
}
{
mul.f16x2 r406, r248, r404;
}
{
fma.rn.f16x2 r409, r245, r402, r406;
}
{
mul.f16x2 r413, r245, r404;
}
{
neg.f16x2 r416, r413;
}
{
fma.rn.f16x2 r418, r248, r402, r416;
}
barrier.sync 0;
mad.lo.s32 r1574, r1572, 24, r1573;
st.shared.v2.f32 [r1574], {r215, r261};
st.shared.v2.f32 [r1574+8], {r298, r335};
st.shared.v2.f32 [r1574+16], {r372, r409};
barrier.sync 0;
mad.lo.s32 r1575, r1572, -20, r1574;
ld.shared.u32 r445, [r1575];
ld.shared.u32 r531, [r1575+864];
ld.shared.u32 r442, [r1575+1728];
ld.shared.u32 r528, [r1575+2592];
ld.shared.u32 r443, [r1575+3456];
ld.shared.u32 r529, [r1575+4320];
barrier.sync 0;
st.shared.v2.f32 [r1574], {r218, r270};
st.shared.v2.f32 [r1574+8], {r307, r344};
st.shared.v2.f32 [r1574+16], {r381, r418};
barrier.sync 0;
ld.shared.u32 r451, [r1575];
ld.shared.u32 r537, [r1575+864];
ld.shared.u32 r448, [r1575+1728];
ld.shared.u32 r534, [r1575+2592];
ld.shared.u32 r449, [r1575+3456];
ld.shared.u32 r535, [r1575+4320];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r439, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f134;
cvt.rn.f16.f32 high, f134;
mov.b32 r440, {low, high};
}
{
add.f16x2 r441, r442, r443;
}
{
add.f16x2 r444, r445, r441;
}
{
add.f16x2 r447, r448, r449;
}
{
add.f16x2 r450, r451, r447;
}
{
add.f16x2 r453, r442, r443;
}
{
mul.f16x2 r456, r453, r439;
}
{
add.f16x2 r459, r445, r456;
}
{
sub.f16x2 r462, r448, r449;
}
{
mul.f16x2 r465, r462, r440;
}
{
add.f16x2 r468, r459, r465;
}
{
add.f16x2 r471, r442, r443;
}
{
mul.f16x2 r474, r471, r439;
}
{
add.f16x2 r477, r445, r474;
}
{
sub.f16x2 r480, r448, r449;
}
{
mul.f16x2 r483, r480, r440;
}
{
sub.f16x2 r486, r477, r483;
}
{
add.f16x2 r489, r448, r449;
}
{
mul.f16x2 r492, r489, r439;
}
{
add.f16x2 r495, r451, r492;
}
{
sub.f16x2 r498, r442, r443;
}
{
mul.f16x2 r501, r498, r440;
}
{
sub.f16x2 r504, r495, r501;
}
{
add.f16x2 r507, r448, r449;
}
{
mul.f16x2 r510, r507, r439;
}
{
add.f16x2 r513, r451, r510;
}
{
sub.f16x2 r516, r442, r443;
}
{
mul.f16x2 r519, r516, r440;
}
{
add.f16x2 r522, r513, r519;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r525, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f134;
cvt.rn.f16.f32 high, f134;
mov.b32 r526, {low, high};
}
{
add.f16x2 r527, r528, r529;
}
{
add.f16x2 r530, r531, r527;
}
{
add.f16x2 r533, r534, r535;
}
{
add.f16x2 r536, r537, r533;
}
{
add.f16x2 r539, r528, r529;
}
{
mul.f16x2 r542, r539, r525;
}
{
add.f16x2 r545, r531, r542;
}
{
sub.f16x2 r548, r534, r535;
}
{
mul.f16x2 r551, r548, r526;
}
{
add.f16x2 r554, r545, r551;
}
{
add.f16x2 r557, r528, r529;
}
{
mul.f16x2 r560, r557, r525;
}
{
add.f16x2 r563, r531, r560;
}
{
sub.f16x2 r566, r534, r535;
}
{
mul.f16x2 r569, r566, r526;
}
{
sub.f16x2 r572, r563, r569;
}
{
add.f16x2 r575, r534, r535;
}
{
mul.f16x2 r578, r575, r525;
}
{
add.f16x2 r581, r537, r578;
}
{
sub.f16x2 r584, r528, r529;
}
{
mul.f16x2 r587, r584, r526;
}
{
sub.f16x2 r590, r581, r587;
}
{
add.f16x2 r593, r534, r535;
}
{
mul.f16x2 r596, r593, r525;
}
{
add.f16x2 r599, r537, r596;
}
{
sub.f16x2 r602, r528, r529;
}
{
mul.f16x2 r605, r602, r526;
}
{
add.f16x2 r608, r599, r605;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f136;
cvt.rn.f16.f32 high, f136;
mov.b32 r611, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r612, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r613, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r614, {low, high};
}
{
mul.f16x2 r621, r554, r611;
}
{
mul.f16x2 r624, r590, r612;
}
{
sub.f16x2 r627, r621, r624;
}
{
mul.f16x2 r630, r554, r612;
}
{
fma.rn.f16x2 r633, r590, r611, r630;
}
{
mul.f16x2 r637, r572, r613;
}
{
mul.f16x2 r640, r608, r614;
}
{
sub.f16x2 r643, r637, r640;
}
{
mul.f16x2 r646, r572, r614;
}
{
fma.rn.f16x2 r649, r608, r613, r646;
}
{
add.f16x2 r653, r444, r530;
}
{
add.f16x2 r656, r450, r536;
}
{
sub.f16x2 r659, r444, r530;
}
{
sub.f16x2 r662, r450, r536;
}
{
add.f16x2 r665, r468, r627;
}
{
add.f16x2 r668, r504, r633;
}
{
sub.f16x2 r671, r468, r627;
}
{
sub.f16x2 r674, r504, r633;
}
{
add.f16x2 r677, r486, r643;
}
{
add.f16x2 r680, r522, r649;
}
{
sub.f16x2 r683, r486, r643;
}
{
sub.f16x2 r686, r522, r649;
}
mul.wide.u32 rd4, r1572, -1431655765;
shr.u64 rd5, rd4, 34;
cvt.u32.u64 r1576, rd5;
mul.lo.s32 r1577, r1576, 6;
sub.s32 r1578, r1572, r1577;
shl.b32 r1579, r1578, 2;
add.s32 r1580, r1573, r1579;
cvt.rn.f32.u32 f158, r1576;
mul.f32 f159, f158, 0f3CEE4BAE;
cos.approx.f32 f71, f159;
sin.approx.f32 f160, f159;
neg.f32 f72, f160;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f71;
cvt.rn.f16.f32 high, f72;
mov.b32 r689, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r692, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r694, {high, high};
}
{
mul.f16x2 r696, r668, r694;
}
{
fma.rn.f16x2 r699, r665, r692, r696;
}
{
mul.f16x2 r703, r665, r694;
}
{
neg.f16x2 r706, r703;
}
{
fma.rn.f16x2 r708, r668, r692, r706;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r712, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r714, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r716, {low, high};
}
{
mul.f16x2 r717, r714, r716;
}
{
mul.f16x2 r720, r689, r712;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r723, {high, low};
}
{
fma.rn.f16x2 r725, r717, r723, r720;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r725;
mov.b32 r729, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r725;
mov.b32 r731, {high, high};
}
{
mul.f16x2 r733, r680, r731;
}
{
fma.rn.f16x2 r736, r677, r729, r733;
}
{
mul.f16x2 r740, r677, r731;
}
{
neg.f16x2 r743, r740;
}
{
fma.rn.f16x2 r745, r680, r729, r743;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r749, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r751, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r753, {low, high};
}
{
mul.f16x2 r754, r751, r753;
}
{
mul.f16x2 r757, r725, r749;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r725;
mov.b32 r760, {high, low};
}
{
fma.rn.f16x2 r762, r754, r760, r757;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r762;
mov.b32 r766, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r762;
mov.b32 r768, {high, high};
}
{
mul.f16x2 r770, r662, r768;
}
{
fma.rn.f16x2 r773, r659, r766, r770;
}
{
mul.f16x2 r777, r659, r768;
}
{
neg.f16x2 r780, r777;
}
{
fma.rn.f16x2 r782, r662, r766, r780;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r786, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r788, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r790, {low, high};
}
{
mul.f16x2 r791, r788, r790;
}
{
mul.f16x2 r794, r762, r786;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r762;
mov.b32 r797, {high, low};
}
{
fma.rn.f16x2 r799, r791, r797, r794;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r799;
mov.b32 r803, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r799;
mov.b32 r805, {high, high};
}
{
mul.f16x2 r807, r674, r805;
}
{
fma.rn.f16x2 r810, r671, r803, r807;
}
{
mul.f16x2 r814, r671, r805;
}
{
neg.f16x2 r817, r814;
}
{
fma.rn.f16x2 r819, r674, r803, r817;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r823, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r689;
mov.b32 r825, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r827, {low, high};
}
{
mul.f16x2 r828, r825, r827;
}
{
mul.f16x2 r831, r799, r823;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r799;
mov.b32 r834, {high, low};
}
{
fma.rn.f16x2 r836, r828, r834, r831;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r836;
mov.b32 r840, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r836;
mov.b32 r842, {high, high};
}
{
mul.f16x2 r844, r686, r842;
}
{
fma.rn.f16x2 r847, r683, r840, r844;
}
{
mul.f16x2 r851, r683, r842;
}
{
neg.f16x2 r854, r851;
}
{
fma.rn.f16x2 r856, r686, r840, r854;
}
barrier.sync 0;
mad.lo.s32 r1581, r1576, 144, r1580;
st.shared.u32 [r1581], r653;
st.shared.u32 [r1581+24], r699;
st.shared.u32 [r1581+48], r736;
st.shared.u32 [r1581+72], r773;
st.shared.u32 [r1581+96], r810;
st.shared.u32 [r1581+120], r847;
barrier.sync 0;
ld.shared.u32 r883, [r1575];
ld.shared.u32 r969, [r1575+864];
ld.shared.u32 r880, [r1575+1728];
ld.shared.u32 r966, [r1575+2592];
ld.shared.u32 r881, [r1575+3456];
ld.shared.u32 r967, [r1575+4320];
barrier.sync 0;
st.shared.u32 [r1581], r656;
st.shared.u32 [r1581+24], r708;
st.shared.u32 [r1581+48], r745;
st.shared.u32 [r1581+72], r782;
st.shared.u32 [r1581+96], r819;
st.shared.u32 [r1581+120], r856;
barrier.sync 0;
ld.shared.u32 r889, [r1575];
ld.shared.u32 r975, [r1575+864];
ld.shared.u32 r886, [r1575+1728];
ld.shared.u32 r972, [r1575+2592];
ld.shared.u32 r887, [r1575+3456];
ld.shared.u32 r973, [r1575+4320];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r877, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f134;
cvt.rn.f16.f32 high, f134;
mov.b32 r878, {low, high};
}
{
add.f16x2 r879, r880, r881;
}
{
add.f16x2 r882, r883, r879;
}
{
add.f16x2 r885, r886, r887;
}
{
add.f16x2 r888, r889, r885;
}
{
add.f16x2 r891, r880, r881;
}
{
mul.f16x2 r894, r891, r877;
}
{
add.f16x2 r897, r883, r894;
}
{
sub.f16x2 r900, r886, r887;
}
{
mul.f16x2 r903, r900, r878;
}
{
add.f16x2 r906, r897, r903;
}
{
add.f16x2 r909, r880, r881;
}
{
mul.f16x2 r912, r909, r877;
}
{
add.f16x2 r915, r883, r912;
}
{
sub.f16x2 r918, r886, r887;
}
{
mul.f16x2 r921, r918, r878;
}
{
sub.f16x2 r924, r915, r921;
}
{
add.f16x2 r927, r886, r887;
}
{
mul.f16x2 r930, r927, r877;
}
{
add.f16x2 r933, r889, r930;
}
{
sub.f16x2 r936, r880, r881;
}
{
mul.f16x2 r939, r936, r878;
}
{
sub.f16x2 r942, r933, r939;
}
{
add.f16x2 r945, r886, r887;
}
{
mul.f16x2 r948, r945, r877;
}
{
add.f16x2 r951, r889, r948;
}
{
sub.f16x2 r954, r880, r881;
}
{
mul.f16x2 r957, r954, r878;
}
{
add.f16x2 r960, r951, r957;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f134;
cvt.rn.f16.f32 high, f134;
mov.b32 r964, {low, high};
}
{
add.f16x2 r965, r966, r967;
}
{
add.f16x2 r968, r969, r965;
}
{
add.f16x2 r971, r972, r973;
}
{
add.f16x2 r974, r975, r971;
}
{
add.f16x2 r977, r966, r967;
}
{
mul.f16x2 r980, r977, r963;
}
{
add.f16x2 r983, r969, r980;
}
{
sub.f16x2 r986, r972, r973;
}
{
mul.f16x2 r989, r986, r964;
}
{
add.f16x2 r992, r983, r989;
}
{
add.f16x2 r995, r966, r967;
}
{
mul.f16x2 r998, r995, r963;
}
{
add.f16x2 r1001, r969, r998;
}
{
sub.f16x2 r1004, r972, r973;
}
{
mul.f16x2 r1007, r1004, r964;
}
{
sub.f16x2 r1010, r1001, r1007;
}
{
add.f16x2 r1013, r972, r973;
}
{
mul.f16x2 r1016, r1013, r963;
}
{
add.f16x2 r1019, r975, r1016;
}
{
sub.f16x2 r1022, r966, r967;
}
{
mul.f16x2 r1025, r1022, r964;
}
{
sub.f16x2 r1028, r1019, r1025;
}
{
add.f16x2 r1031, r972, r973;
}
{
mul.f16x2 r1034, r1031, r963;
}
{
add.f16x2 r1037, r975, r1034;
}
{
sub.f16x2 r1040, r966, r967;
}
{
mul.f16x2 r1043, r1040, r964;
}
{
add.f16x2 r1046, r1037, r1043;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f136;
cvt.rn.f16.f32 high, f136;
mov.b32 r1049, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r1050, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1051, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r1052, {low, high};
}
{
mul.f16x2 r1059, r992, r1049;
}
{
mul.f16x2 r1062, r1028, r1050;
}
{
sub.f16x2 r1065, r1059, r1062;
}
{
mul.f16x2 r1068, r992, r1050;
}
{
fma.rn.f16x2 r1071, r1028, r1049, r1068;
}
{
mul.f16x2 r1075, r1010, r1051;
}
{
mul.f16x2 r1078, r1046, r1052;
}
{
sub.f16x2 r1081, r1075, r1078;
}
{
mul.f16x2 r1084, r1010, r1052;
}
{
fma.rn.f16x2 r1087, r1046, r1051, r1084;
}
{
add.f16x2 r1091, r882, r968;
}
{
add.f16x2 r1094, r888, r974;
}
{
sub.f16x2 r1097, r882, r968;
}
{
sub.f16x2 r1100, r888, r974;
}
{
add.f16x2 r1103, r906, r1065;
}
{
add.f16x2 r1106, r942, r1071;
}
{
sub.f16x2 r1109, r906, r1065;
}
{
sub.f16x2 r1112, r942, r1071;
}
{
add.f16x2 r1115, r924, r1081;
}
{
add.f16x2 r1118, r960, r1087;
}
{
sub.f16x2 r1121, r924, r1081;
}
{
sub.f16x2 r1124, r960, r1087;
}
mul.wide.u32 rd6, r1572, 954437177;
shr.u64 rd7, rd6, 35;
cvt.u32.u64 r1582, rd7;
mul.lo.s32 r1583, r1582, 36;
sub.s32 r1584, r1572, r1583;
shl.b32 r1585, r1584, 2;
add.s32 r1586, r1573, r1585;
cvt.rn.f32.u32 f161, r1582;
mul.f32 f162, f161, 0f3E32B8C2;
cos.approx.f32 f113, f162;
sin.approx.f32 f163, f162;
neg.f32 f114, f163;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f113;
cvt.rn.f16.f32 high, f114;
mov.b32 r1127, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1130, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1132, {high, high};
}
{
mul.f16x2 r1134, r1106, r1132;
}
{
fma.rn.f16x2 r1137, r1103, r1130, r1134;
}
{
mul.f16x2 r1141, r1103, r1132;
}
{
neg.f16x2 r1144, r1141;
}
{
fma.rn.f16x2 r1146, r1106, r1130, r1144;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1150, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1152, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r1154, {low, high};
}
{
mul.f16x2 r1155, r1152, r1154;
}
{
mul.f16x2 r1158, r1127, r1150;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1161, {high, low};
}
{
fma.rn.f16x2 r1163, r1155, r1161, r1158;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1163;
mov.b32 r1167, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1163;
mov.b32 r1169, {high, high};
}
{
mul.f16x2 r1171, r1118, r1169;
}
{
fma.rn.f16x2 r1174, r1115, r1167, r1171;
}
{
mul.f16x2 r1178, r1115, r1169;
}
{
neg.f16x2 r1181, r1178;
}
{
fma.rn.f16x2 r1183, r1118, r1167, r1181;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1187, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1189, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r1191, {low, high};
}
{
mul.f16x2 r1192, r1189, r1191;
}
{
mul.f16x2 r1195, r1163, r1187;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1163;
mov.b32 r1198, {high, low};
}
{
fma.rn.f16x2 r1200, r1192, r1198, r1195;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1200;
mov.b32 r1204, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1200;
mov.b32 r1206, {high, high};
}
{
mul.f16x2 r1208, r1100, r1206;
}
{
fma.rn.f16x2 r1211, r1097, r1204, r1208;
}
{
mul.f16x2 r1215, r1097, r1206;
}
{
neg.f16x2 r1218, r1215;
}
{
fma.rn.f16x2 r1220, r1100, r1204, r1218;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1224, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1226, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r1228, {low, high};
}
{
mul.f16x2 r1229, r1226, r1228;
}
{
mul.f16x2 r1232, r1200, r1224;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1200;
mov.b32 r1235, {high, low};
}
{
fma.rn.f16x2 r1237, r1229, r1235, r1232;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1237;
mov.b32 r1241, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1237;
mov.b32 r1243, {high, high};
}
{
mul.f16x2 r1245, r1112, r1243;
}
{
fma.rn.f16x2 r1248, r1109, r1241, r1245;
}
{
mul.f16x2 r1252, r1109, r1243;
}
{
neg.f16x2 r1255, r1252;
}
{
fma.rn.f16x2 r1257, r1112, r1241, r1255;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1261, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1127;
mov.b32 r1263, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r1265, {low, high};
}
{
mul.f16x2 r1266, r1263, r1265;
}
{
mul.f16x2 r1269, r1237, r1261;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1237;
mov.b32 r1272, {high, low};
}
{
fma.rn.f16x2 r1274, r1266, r1272, r1269;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1274;
mov.b32 r1278, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1274;
mov.b32 r1280, {high, high};
}
{
mul.f16x2 r1282, r1124, r1280;
}
{
fma.rn.f16x2 r1285, r1121, r1278, r1282;
}
{
mul.f16x2 r1289, r1121, r1280;
}
{
neg.f16x2 r1292, r1289;
}
{
fma.rn.f16x2 r1294, r1124, r1278, r1292;
}
barrier.sync 0;
mad.lo.s32 r1587, r1582, 864, r1586;
st.shared.u32 [r1587], r1091;
st.shared.u32 [r1587+144], r1137;
st.shared.u32 [r1587+288], r1174;
st.shared.u32 [r1587+432], r1211;
st.shared.u32 [r1587+576], r1248;
st.shared.u32 [r1587+720], r1285;
barrier.sync 0;
ld.shared.u32 r1321, [r1575];
ld.shared.u32 r1407, [r1575+864];
ld.shared.u32 r1318, [r1575+1728];
ld.shared.u32 r1404, [r1575+2592];
ld.shared.u32 r1319, [r1575+3456];
ld.shared.u32 r1405, [r1575+4320];
barrier.sync 0;
st.shared.u32 [r1587], r1094;
st.shared.u32 [r1587+144], r1146;
st.shared.u32 [r1587+288], r1183;
st.shared.u32 [r1587+432], r1220;
st.shared.u32 [r1587+576], r1257;
st.shared.u32 [r1587+720], r1294;
barrier.sync 0;
ld.shared.u32 r1327, [r1575];
ld.shared.u32 r1413, [r1575+864];
ld.shared.u32 r1324, [r1575+1728];
ld.shared.u32 r1410, [r1575+2592];
ld.shared.u32 r1325, [r1575+3456];
ld.shared.u32 r1411, [r1575+4320];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1315, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f134;
cvt.rn.f16.f32 high, f134;
mov.b32 r1316, {low, high};
}
{
add.f16x2 r1317, r1318, r1319;
}
{
add.f16x2 r1320, r1321, r1317;
}
{
add.f16x2 r1323, r1324, r1325;
}
{
add.f16x2 r1326, r1327, r1323;
}
{
add.f16x2 r1329, r1318, r1319;
}
{
mul.f16x2 r1332, r1329, r1315;
}
{
add.f16x2 r1335, r1321, r1332;
}
{
sub.f16x2 r1338, r1324, r1325;
}
{
mul.f16x2 r1341, r1338, r1316;
}
{
add.f16x2 r1344, r1335, r1341;
}
{
add.f16x2 r1347, r1318, r1319;
}
{
mul.f16x2 r1350, r1347, r1315;
}
{
add.f16x2 r1353, r1321, r1350;
}
{
sub.f16x2 r1356, r1324, r1325;
}
{
mul.f16x2 r1359, r1356, r1316;
}
{
sub.f16x2 r1362, r1353, r1359;
}
{
add.f16x2 r1365, r1324, r1325;
}
{
mul.f16x2 r1368, r1365, r1315;
}
{
add.f16x2 r1371, r1327, r1368;
}
{
sub.f16x2 r1374, r1318, r1319;
}
{
mul.f16x2 r1377, r1374, r1316;
}
{
sub.f16x2 r1380, r1371, r1377;
}
{
add.f16x2 r1383, r1324, r1325;
}
{
mul.f16x2 r1386, r1383, r1315;
}
{
add.f16x2 r1389, r1327, r1386;
}
{
sub.f16x2 r1392, r1318, r1319;
}
{
mul.f16x2 r1395, r1392, r1316;
}
{
add.f16x2 r1398, r1389, r1395;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f134;
cvt.rn.f16.f32 high, f134;
mov.b32 r1402, {low, high};
}
{
add.f16x2 r1403, r1404, r1405;
}
{
add.f16x2 r1406, r1407, r1403;
}
{
add.f16x2 r1409, r1410, r1411;
}
{
add.f16x2 r1412, r1413, r1409;
}
{
add.f16x2 r1415, r1404, r1405;
}
{
mul.f16x2 r1418, r1415, r1401;
}
{
add.f16x2 r1421, r1407, r1418;
}
{
sub.f16x2 r1424, r1410, r1411;
}
{
mul.f16x2 r1427, r1424, r1402;
}
{
add.f16x2 r1430, r1421, r1427;
}
{
add.f16x2 r1433, r1404, r1405;
}
{
mul.f16x2 r1436, r1433, r1401;
}
{
add.f16x2 r1439, r1407, r1436;
}
{
sub.f16x2 r1442, r1410, r1411;
}
{
mul.f16x2 r1445, r1442, r1402;
}
{
sub.f16x2 r1448, r1439, r1445;
}
{
add.f16x2 r1451, r1410, r1411;
}
{
mul.f16x2 r1454, r1451, r1401;
}
{
add.f16x2 r1457, r1413, r1454;
}
{
sub.f16x2 r1460, r1404, r1405;
}
{
mul.f16x2 r1463, r1460, r1402;
}
{
sub.f16x2 r1466, r1457, r1463;
}
{
add.f16x2 r1469, r1410, r1411;
}
{
mul.f16x2 r1472, r1469, r1401;
}
{
add.f16x2 r1475, r1413, r1472;
}
{
sub.f16x2 r1478, r1404, r1405;
}
{
mul.f16x2 r1481, r1478, r1402;
}
{
add.f16x2 r1484, r1475, r1481;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f136;
cvt.rn.f16.f32 high, f136;
mov.b32 r1487, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r1488, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f140;
cvt.rn.f16.f32 high, f140;
mov.b32 r1489, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f142;
cvt.rn.f16.f32 high, f142;
mov.b32 r1490, {low, high};
}
{
mul.f16x2 r1497, r1430, r1487;
}
{
mul.f16x2 r1500, r1466, r1488;
}
{
sub.f16x2 r1503, r1497, r1500;
}
{
mul.f16x2 r1506, r1430, r1488;
}
{
fma.rn.f16x2 r1509, r1466, r1487, r1506;
}
{
mul.f16x2 r1513, r1448, r1489;
}
{
mul.f16x2 r1516, r1484, r1490;
}
{
sub.f16x2 r1519, r1513, r1516;
}
{
mul.f16x2 r1522, r1448, r1490;
}
{
fma.rn.f16x2 r1525, r1484, r1489, r1522;
}
{
add.f16x2 %0, r1320, r1406;
}
{
add.f16x2 %1, r1326, r1412;
}
{
sub.f16x2 %6, r1320, r1406;
}
{
sub.f16x2 %7, r1326, r1412;
}
{
add.f16x2 %2, r1344, r1503;
}
{
add.f16x2 %3, r1380, r1509;
}
{
sub.f16x2 %8, r1344, r1503;
}
{
sub.f16x2 %9, r1380, r1509;
}
{
add.f16x2 %4, r1362, r1519;
}
{
add.f16x2 %5, r1398, r1525;
}
{
sub.f16x2 %10, r1362, r1519;
}
{
sub.f16x2 %11, r1398, r1525;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)));
};


#endif
