#ifndef CUFFTDX_FFT_216_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_216_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1134, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<119>;
.reg .b32 r<1146>;
.reg .b64 rd<7>;
mov.u32 r1127, %tid.y;
shl.b32 r1128, r1127, 1;
mov.u32 r1129, %12;
mad.lo.s32 r1130, r1128, 864, r1129;
mov.u32 r1131, %tid.x;
mov.f32 f98, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f98;
cvt.rn.f16.f32 high, f98;
mov.b32 r1, {low, high};
}
mov.f32 f92, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
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
cvt.rn.f16.f32 low, f98;
cvt.rn.f16.f32 high, f98;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
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
mov.f32 f94, 0f3F000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f94;
cvt.rn.f16.f32 high, f94;
mov.b32 r173, {low, high};
}
mov.f32 f100, 0f3F5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f100;
cvt.rn.f16.f32 high, f100;
mov.b32 r174, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f98;
cvt.rn.f16.f32 high, f98;
mov.b32 r175, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f100;
cvt.rn.f16.f32 high, f100;
mov.b32 r176, {low, high};
}
mov.f32 f81, 0fBF800000;
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
mul.wide.u32 rd2, r1131, 954437177;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r1132, rd3;
mul.lo.s32 r1133, r1132, 36;
sub.s32 r1134, r1131, r1133;
shr.u64 rd4, rd2, 34;
cvt.u32.u64 r1135, rd4;
and.b32 r1136, r1135, 1073741822;
mad.lo.s32 r1137, r1136, 864, r1130;
cvt.rn.f32.u32 f113, r1134;
mul.f32 f114, f113, 0f3CEE4BAE;
cos.approx.f32 f29, f114;
sin.approx.f32 f115, f114;
neg.f32 f30, f115;
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
mov.f32 f82, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f81;
cvt.rn.f16.f32 high, f82;
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
cvt.rn.f16.f32 low, f81;
cvt.rn.f16.f32 high, f82;
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
cvt.rn.f16.f32 low, f81;
cvt.rn.f16.f32 high, f82;
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
cvt.rn.f16.f32 low, f81;
cvt.rn.f16.f32 high, f82;
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
mad.lo.s32 r1138, r1134, 48, r1137;
st.shared.v2.f32 [r1138], {r215, r218};
st.shared.v2.f32 [r1138+8], {r261, r270};
st.shared.v2.f32 [r1138+16], {r298, r307};
st.shared.v2.f32 [r1138+24], {r335, r344};
st.shared.v2.f32 [r1138+32], {r372, r381};
st.shared.v2.f32 [r1138+40], {r409, r418};
barrier.sync 0;
mad.lo.s32 r1139, r1134, -40, r1138;
ld.shared.u32 r445, [r1139];
ld.shared.u32 r451, [r1139+4];
ld.shared.u32 r531, [r1139+288];
ld.shared.u32 r537, [r1139+292];
ld.shared.u32 r442, [r1139+576];
ld.shared.u32 r448, [r1139+580];
ld.shared.u32 r528, [r1139+864];
ld.shared.u32 r534, [r1139+868];
ld.shared.u32 r443, [r1139+1152];
ld.shared.u32 r449, [r1139+1156];
ld.shared.u32 r529, [r1139+1440];
ld.shared.u32 r535, [r1139+1444];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f98;
cvt.rn.f16.f32 high, f98;
mov.b32 r439, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
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
cvt.rn.f16.f32 low, f98;
cvt.rn.f16.f32 high, f98;
mov.b32 r525, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
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
cvt.rn.f16.f32 low, f94;
cvt.rn.f16.f32 high, f94;
mov.b32 r611, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f100;
cvt.rn.f16.f32 high, f100;
mov.b32 r612, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f98;
cvt.rn.f16.f32 high, f98;
mov.b32 r613, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f100;
cvt.rn.f16.f32 high, f100;
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
mul.wide.u32 rd5, r1134, -1431655765;
shr.u64 rd6, rd5, 34;
cvt.u32.u64 r1140, rd6;
cvt.rn.f32.u32 f116, r1140;
mul.f32 f117, f116, 0f3E32B8C2;
cos.approx.f32 f71, f117;
sin.approx.f32 f118, f117;
neg.f32 f72, f118;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f71;
cvt.rn.f16.f32 high, f72;
mov.b32 r689, {low, high};
}
mul.lo.s32 r1141, r1140, 6;
sub.s32 r1142, r1134, r1141;
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
cvt.rn.f16.f32 low, f81;
cvt.rn.f16.f32 high, f82;
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
cvt.rn.f16.f32 low, f81;
cvt.rn.f16.f32 high, f82;
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
cvt.rn.f16.f32 low, f81;
cvt.rn.f16.f32 high, f82;
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
cvt.rn.f16.f32 low, f81;
cvt.rn.f16.f32 high, f82;
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
shl.b32 r1143, r1142, 3;
add.s32 r1144, r1137, r1143;
barrier.sync 0;
mad.lo.s32 r1145, r1140, 288, r1144;
st.shared.u32 [r1145], r653;
st.shared.u32 [r1145+4], r656;
st.shared.u32 [r1145+48], r699;
st.shared.u32 [r1145+52], r708;
st.shared.u32 [r1145+96], r736;
st.shared.u32 [r1145+100], r745;
st.shared.u32 [r1145+144], r773;
st.shared.u32 [r1145+148], r782;
st.shared.u32 [r1145+192], r810;
st.shared.u32 [r1145+196], r819;
st.shared.u32 [r1145+240], r847;
st.shared.u32 [r1145+244], r856;
barrier.sync 0;
ld.shared.u32 r883, [r1139];
ld.shared.u32 r889, [r1139+4];
ld.shared.u32 r969, [r1139+288];
ld.shared.u32 r975, [r1139+292];
ld.shared.u32 r880, [r1139+576];
ld.shared.u32 r886, [r1139+580];
ld.shared.u32 r966, [r1139+864];
ld.shared.u32 r972, [r1139+868];
ld.shared.u32 r881, [r1139+1152];
ld.shared.u32 r887, [r1139+1156];
ld.shared.u32 r967, [r1139+1440];
ld.shared.u32 r973, [r1139+1444];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f98;
cvt.rn.f16.f32 high, f98;
mov.b32 r877, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
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
cvt.rn.f16.f32 low, f98;
cvt.rn.f16.f32 high, f98;
mov.b32 r963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
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
cvt.rn.f16.f32 low, f94;
cvt.rn.f16.f32 high, f94;
mov.b32 r1049, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f100;
cvt.rn.f16.f32 high, f100;
mov.b32 r1050, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f98;
cvt.rn.f16.f32 high, f98;
mov.b32 r1051, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f100;
cvt.rn.f16.f32 high, f100;
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
add.f16x2 %0, r882, r968;
}
{
add.f16x2 %1, r888, r974;
}
{
sub.f16x2 %6, r882, r968;
}
{
sub.f16x2 %7, r888, r974;
}
{
add.f16x2 %2, r906, r1065;
}
{
add.f16x2 %3, r942, r1071;
}
{
sub.f16x2 %8, r906, r1065;
}
{
sub.f16x2 %9, r942, r1071;
}
{
add.f16x2 %4, r924, r1081;
}
{
add.f16x2 %5, r960, r1087;
}
{
sub.f16x2 %10, r924, r1081;
}
{
sub.f16x2 %11, r960, r1087;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1135, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<119>;
.reg .b32 r<1143>;
.reg .b64 rd<6>;
mov.u32 r1127, %tid.y;
mov.u32 r1128, %12;
mad.lo.s32 r1129, r1127, 864, r1128;
mov.u32 r1130, %tid.x;
mov.f32 f98, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f98;
cvt.rn.f16.f32 high, f98;
mov.b32 r1, {low, high};
}
mov.f32 f92, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
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
cvt.rn.f16.f32 low, f98;
cvt.rn.f16.f32 high, f98;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
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
mov.f32 f94, 0f3F000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f94;
cvt.rn.f16.f32 high, f94;
mov.b32 r173, {low, high};
}
mov.f32 f100, 0f3F5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f100;
cvt.rn.f16.f32 high, f100;
mov.b32 r174, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f98;
cvt.rn.f16.f32 high, f98;
mov.b32 r175, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f100;
cvt.rn.f16.f32 high, f100;
mov.b32 r176, {low, high};
}
mov.f32 f81, 0fBF800000;
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
mul.wide.u32 rd2, r1130, 954437177;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r1131, rd3;
mul.lo.s32 r1132, r1131, 36;
sub.s32 r1133, r1130, r1132;
mad.lo.s32 r1134, r1131, 864, r1129;
cvt.rn.f32.u32 f113, r1133;
mul.f32 f114, f113, 0f3CEE4BAE;
cos.approx.f32 f29, f114;
sin.approx.f32 f115, f114;
neg.f32 f30, f115;
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
mov.f32 f82, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f81;
cvt.rn.f16.f32 high, f82;
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
cvt.rn.f16.f32 low, f81;
cvt.rn.f16.f32 high, f82;
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
cvt.rn.f16.f32 low, f81;
cvt.rn.f16.f32 high, f82;
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
cvt.rn.f16.f32 low, f81;
cvt.rn.f16.f32 high, f82;
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
mad.lo.s32 r1135, r1133, 24, r1134;
st.shared.v2.f32 [r1135], {r215, r261};
st.shared.v2.f32 [r1135+8], {r298, r335};
st.shared.v2.f32 [r1135+16], {r372, r409};
barrier.sync 0;
mad.lo.s32 r1136, r1133, -20, r1135;
ld.shared.u32 r445, [r1136];
ld.shared.u32 r531, [r1136+144];
ld.shared.u32 r442, [r1136+288];
ld.shared.u32 r528, [r1136+432];
ld.shared.u32 r443, [r1136+576];
ld.shared.u32 r529, [r1136+720];
barrier.sync 0;
st.shared.v2.f32 [r1135], {r218, r270};
st.shared.v2.f32 [r1135+8], {r307, r344};
st.shared.v2.f32 [r1135+16], {r381, r418};
barrier.sync 0;
ld.shared.u32 r451, [r1136];
ld.shared.u32 r537, [r1136+144];
ld.shared.u32 r448, [r1136+288];
ld.shared.u32 r534, [r1136+432];
ld.shared.u32 r449, [r1136+576];
ld.shared.u32 r535, [r1136+720];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f98;
cvt.rn.f16.f32 high, f98;
mov.b32 r439, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
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
cvt.rn.f16.f32 low, f98;
cvt.rn.f16.f32 high, f98;
mov.b32 r525, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
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
cvt.rn.f16.f32 low, f94;
cvt.rn.f16.f32 high, f94;
mov.b32 r611, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f100;
cvt.rn.f16.f32 high, f100;
mov.b32 r612, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f98;
cvt.rn.f16.f32 high, f98;
mov.b32 r613, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f100;
cvt.rn.f16.f32 high, f100;
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
mul.wide.u32 rd4, r1133, -1431655765;
shr.u64 rd5, rd4, 34;
cvt.u32.u64 r1137, rd5;
mul.lo.s32 r1138, r1137, 6;
sub.s32 r1139, r1133, r1138;
shl.b32 r1140, r1139, 2;
add.s32 r1141, r1134, r1140;
cvt.rn.f32.u32 f116, r1137;
mul.f32 f117, f116, 0f3E32B8C2;
cos.approx.f32 f71, f117;
sin.approx.f32 f118, f117;
neg.f32 f72, f118;
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
cvt.rn.f16.f32 low, f81;
cvt.rn.f16.f32 high, f82;
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
cvt.rn.f16.f32 low, f81;
cvt.rn.f16.f32 high, f82;
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
cvt.rn.f16.f32 low, f81;
cvt.rn.f16.f32 high, f82;
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
cvt.rn.f16.f32 low, f81;
cvt.rn.f16.f32 high, f82;
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
mad.lo.s32 r1142, r1137, 144, r1141;
st.shared.u32 [r1142], r653;
st.shared.u32 [r1142+24], r699;
st.shared.u32 [r1142+48], r736;
st.shared.u32 [r1142+72], r773;
st.shared.u32 [r1142+96], r810;
st.shared.u32 [r1142+120], r847;
barrier.sync 0;
ld.shared.u32 r883, [r1136];
ld.shared.u32 r969, [r1136+144];
ld.shared.u32 r880, [r1136+288];
ld.shared.u32 r966, [r1136+432];
ld.shared.u32 r881, [r1136+576];
ld.shared.u32 r967, [r1136+720];
barrier.sync 0;
st.shared.u32 [r1142], r656;
st.shared.u32 [r1142+24], r708;
st.shared.u32 [r1142+48], r745;
st.shared.u32 [r1142+72], r782;
st.shared.u32 [r1142+96], r819;
st.shared.u32 [r1142+120], r856;
barrier.sync 0;
ld.shared.u32 r889, [r1136];
ld.shared.u32 r975, [r1136+144];
ld.shared.u32 r886, [r1136+288];
ld.shared.u32 r972, [r1136+432];
ld.shared.u32 r887, [r1136+576];
ld.shared.u32 r973, [r1136+720];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f98;
cvt.rn.f16.f32 high, f98;
mov.b32 r877, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
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
cvt.rn.f16.f32 low, f98;
cvt.rn.f16.f32 high, f98;
mov.b32 r963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
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
cvt.rn.f16.f32 low, f94;
cvt.rn.f16.f32 high, f94;
mov.b32 r1049, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f100;
cvt.rn.f16.f32 high, f100;
mov.b32 r1050, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f98;
cvt.rn.f16.f32 high, f98;
mov.b32 r1051, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f100;
cvt.rn.f16.f32 high, f100;
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
add.f16x2 %0, r882, r968;
}
{
add.f16x2 %1, r888, r974;
}
{
sub.f16x2 %6, r882, r968;
}
{
sub.f16x2 %7, r888, r974;
}
{
add.f16x2 %2, r906, r1065;
}
{
add.f16x2 %3, r942, r1071;
}
{
sub.f16x2 %8, r906, r1065;
}
{
sub.f16x2 %9, r942, r1071;
}
{
add.f16x2 %4, r924, r1081;
}
{
add.f16x2 %5, r960, r1087;
}
{
sub.f16x2 %10, r924, r1081;
}
{
sub.f16x2 %11, r960, r1087;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)));
};


#endif
