#ifndef CUFFTDX_FFT_36_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_36_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1133, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<74>;
.reg .b32 r<702>;
.reg .b64 rd<5>;
mov.u32 r689, %tid.y;
shl.b32 r690, r689, 1;
mov.u32 r691, %12;
mad.lo.s32 r692, r690, 144, r691;
mov.u32 r693, %tid.x;
mov.f32 f56, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r1, {low, high};
}
mov.f32 f50, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
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
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
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
mov.f32 f52, 0f3F000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r173, {low, high};
}
mov.f32 f58, 0f3F5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r174, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r175, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r176, {low, high};
}
mov.f32 f39, 0fBF800000;
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
mul.wide.u32 rd2, r693, -1431655765;
shr.u64 rd3, rd2, 34;
cvt.u32.u64 r694, rd3;
mul.lo.s32 r695, r694, 6;
sub.s32 r696, r693, r695;
shr.u64 rd4, rd2, 33;
cvt.u32.u64 r697, rd4;
and.b32 r698, r697, 2147483646;
mad.lo.s32 r699, r698, 144, r692;
cvt.rn.f32.u32 f71, r696;
mul.f32 f72, f71, 0f3E32B8C2;
cos.approx.f32 f29, f72;
sin.approx.f32 f73, f72;
neg.f32 f30, f73;
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
mov.f32 f40, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f39;
cvt.rn.f16.f32 high, f40;
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
cvt.rn.f16.f32 low, f39;
cvt.rn.f16.f32 high, f40;
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
cvt.rn.f16.f32 low, f39;
cvt.rn.f16.f32 high, f40;
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
cvt.rn.f16.f32 low, f39;
cvt.rn.f16.f32 high, f40;
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
mad.lo.s32 r700, r696, 48, r699;
st.shared.v2.f32 [r700], {r215, r218};
st.shared.v2.f32 [r700+8], {r261, r270};
st.shared.v2.f32 [r700+16], {r298, r307};
st.shared.v2.f32 [r700+24], {r335, r344};
st.shared.v2.f32 [r700+32], {r372, r381};
st.shared.v2.f32 [r700+40], {r409, r418};
barrier.sync 0;
mad.lo.s32 r701, r696, -40, r700;
ld.shared.u32 r445, [r701];
ld.shared.u32 r451, [r701+4];
ld.shared.u32 r531, [r701+48];
ld.shared.u32 r537, [r701+52];
ld.shared.u32 r442, [r701+96];
ld.shared.u32 r448, [r701+100];
ld.shared.u32 r528, [r701+144];
ld.shared.u32 r534, [r701+148];
ld.shared.u32 r443, [r701+192];
ld.shared.u32 r449, [r701+196];
ld.shared.u32 r529, [r701+240];
ld.shared.u32 r535, [r701+244];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r439, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
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
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r525, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
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
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r611, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r612, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r613, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
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
add.f16x2 %0, r444, r530;
}
{
add.f16x2 %1, r450, r536;
}
{
sub.f16x2 %6, r444, r530;
}
{
sub.f16x2 %7, r450, r536;
}
{
add.f16x2 %2, r468, r627;
}
{
add.f16x2 %3, r504, r633;
}
{
sub.f16x2 %8, r468, r627;
}
{
sub.f16x2 %9, r504, r633;
}
{
add.f16x2 %4, r486, r643;
}
{
add.f16x2 %5, r522, r649;
}
{
sub.f16x2 %10, r486, r643;
}
{
sub.f16x2 %11, r522, r649;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1132, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<74>;
.reg .b32 r<699>;
.reg .b64 rd<4>;
mov.u32 r689, %tid.y;
mov.u32 r690, %12;
mad.lo.s32 r691, r689, 144, r690;
mov.u32 r692, %tid.x;
mov.f32 f56, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r1, {low, high};
}
mov.f32 f50, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
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
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
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
mov.f32 f52, 0f3F000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r173, {low, high};
}
mov.f32 f58, 0f3F5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r174, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r175, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r176, {low, high};
}
mov.f32 f39, 0fBF800000;
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
mul.wide.u32 rd2, r692, -1431655765;
shr.u64 rd3, rd2, 34;
cvt.u32.u64 r693, rd3;
mul.lo.s32 r694, r693, 6;
sub.s32 r695, r692, r694;
mad.lo.s32 r696, r693, 144, r691;
cvt.rn.f32.u32 f71, r695;
mul.f32 f72, f71, 0f3E32B8C2;
cos.approx.f32 f29, f72;
sin.approx.f32 f73, f72;
neg.f32 f30, f73;
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
mov.f32 f40, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f39;
cvt.rn.f16.f32 high, f40;
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
cvt.rn.f16.f32 low, f39;
cvt.rn.f16.f32 high, f40;
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
cvt.rn.f16.f32 low, f39;
cvt.rn.f16.f32 high, f40;
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
cvt.rn.f16.f32 low, f39;
cvt.rn.f16.f32 high, f40;
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
mad.lo.s32 r697, r695, 24, r696;
st.shared.v2.f32 [r697], {r215, r261};
st.shared.v2.f32 [r697+8], {r298, r335};
st.shared.v2.f32 [r697+16], {r372, r409};
barrier.sync 0;
mad.lo.s32 r698, r695, -20, r697;
ld.shared.u32 r445, [r698];
ld.shared.u32 r531, [r698+24];
ld.shared.u32 r442, [r698+48];
ld.shared.u32 r528, [r698+72];
ld.shared.u32 r443, [r698+96];
ld.shared.u32 r529, [r698+120];
barrier.sync 0;
st.shared.v2.f32 [r697], {r218, r270};
st.shared.v2.f32 [r697+8], {r307, r344};
st.shared.v2.f32 [r697+16], {r381, r418};
barrier.sync 0;
ld.shared.u32 r451, [r698];
ld.shared.u32 r537, [r698+24];
ld.shared.u32 r448, [r698+48];
ld.shared.u32 r534, [r698+72];
ld.shared.u32 r449, [r698+96];
ld.shared.u32 r535, [r698+120];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r439, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
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
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r525, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
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
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r611, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
mov.b32 r612, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r613, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f58;
cvt.rn.f16.f32 high, f58;
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
add.f16x2 %0, r444, r530;
}
{
add.f16x2 %1, r450, r536;
}
{
sub.f16x2 %6, r444, r530;
}
{
sub.f16x2 %7, r450, r536;
}
{
add.f16x2 %2, r468, r627;
}
{
add.f16x2 %3, r504, r633;
}
{
sub.f16x2 %8, r468, r627;
}
{
sub.f16x2 %9, r504, r633;
}
{
add.f16x2 %4, r486, r643;
}
{
add.f16x2 %5, r522, r649;
}
{
sub.f16x2 %10, r486, r643;
}
{
sub.f16x2 %11, r522, r649;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)));
};


#endif
