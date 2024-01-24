#ifndef CUFFTDX_FFT_81_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_81_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1074, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<136>;
.reg .b32 r<1503>;
.reg .b64 rd<4>;
mov.u32 r1492, %tid.y;
mov.u32 r1493, %18;
mad.lo.s32 r1494, r1492, 648, r1493;
mov.u32 r1495, %tid.x;
mov.f32 f130, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r1, {low, high};
}
mov.f32 f132, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r2, {low, high};
}
{
add.f16x2 r3, %25, %31;
}
{
add.f16x2 r6, %19, r3;
}
{
add.f16x2 r9, %26, %32;
}
{
add.f16x2 r12, %20, r9;
}
{
add.f16x2 r15, %25, %31;
}
{
mul.f16x2 r18, r15, r1;
}
{
add.f16x2 r21, %19, r18;
}
{
sub.f16x2 r24, %26, %32;
}
{
mul.f16x2 r27, r24, r2;
}
{
add.f16x2 r30, r21, r27;
}
{
add.f16x2 r33, %25, %31;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %19, r36;
}
{
sub.f16x2 r42, %26, %32;
}
{
mul.f16x2 r45, r42, r2;
}
{
sub.f16x2 r48, r39, r45;
}
{
add.f16x2 r51, %26, %32;
}
{
mul.f16x2 r54, r51, r1;
}
{
add.f16x2 r57, %20, r54;
}
{
sub.f16x2 r60, %25, %31;
}
{
mul.f16x2 r63, r60, r2;
}
{
sub.f16x2 r66, r57, r63;
}
{
add.f16x2 r69, %26, %32;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %20, r72;
}
{
sub.f16x2 r78, %25, %31;
}
{
mul.f16x2 r81, r78, r2;
}
{
add.f16x2 r84, r75, r81;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r88, {low, high};
}
{
add.f16x2 r89, %27, %33;
}
{
add.f16x2 r92, %21, r89;
}
{
add.f16x2 r95, %28, %34;
}
{
add.f16x2 r98, %22, r95;
}
{
add.f16x2 r101, %27, %33;
}
{
mul.f16x2 r104, r101, r87;
}
{
add.f16x2 r107, %21, r104;
}
{
sub.f16x2 r110, %28, %34;
}
{
mul.f16x2 r113, r110, r88;
}
{
add.f16x2 r116, r107, r113;
}
{
add.f16x2 r119, %27, %33;
}
{
mul.f16x2 r122, r119, r87;
}
{
add.f16x2 r125, %21, r122;
}
{
sub.f16x2 r128, %28, %34;
}
{
mul.f16x2 r131, r128, r88;
}
{
sub.f16x2 r134, r125, r131;
}
{
add.f16x2 r137, %28, %34;
}
{
mul.f16x2 r140, r137, r87;
}
{
add.f16x2 r143, %22, r140;
}
{
sub.f16x2 r146, %27, %33;
}
{
mul.f16x2 r149, r146, r88;
}
{
sub.f16x2 r152, r143, r149;
}
{
add.f16x2 r155, %28, %34;
}
{
mul.f16x2 r158, r155, r87;
}
{
add.f16x2 r161, %22, r158;
}
{
sub.f16x2 r164, %27, %33;
}
{
mul.f16x2 r167, r164, r88;
}
{
add.f16x2 r170, r161, r167;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r173, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r174, {low, high};
}
{
add.f16x2 r175, %29, %35;
}
{
add.f16x2 r178, %23, r175;
}
{
add.f16x2 r181, %30, %36;
}
{
add.f16x2 r184, %24, r181;
}
{
add.f16x2 r187, %29, %35;
}
{
mul.f16x2 r190, r187, r173;
}
{
add.f16x2 r193, %23, r190;
}
{
sub.f16x2 r196, %30, %36;
}
{
mul.f16x2 r199, r196, r174;
}
{
add.f16x2 r202, r193, r199;
}
{
add.f16x2 r205, %29, %35;
}
{
mul.f16x2 r208, r205, r173;
}
{
add.f16x2 r211, %23, r208;
}
{
sub.f16x2 r214, %30, %36;
}
{
mul.f16x2 r217, r214, r174;
}
{
sub.f16x2 r220, r211, r217;
}
{
add.f16x2 r223, %30, %36;
}
{
mul.f16x2 r226, r223, r173;
}
{
add.f16x2 r229, %24, r226;
}
{
sub.f16x2 r232, %29, %35;
}
{
mul.f16x2 r235, r232, r174;
}
{
sub.f16x2 r238, r229, r235;
}
{
add.f16x2 r241, %30, %36;
}
{
mul.f16x2 r244, r241, r173;
}
{
add.f16x2 r247, %24, r244;
}
{
sub.f16x2 r250, %29, %35;
}
{
mul.f16x2 r253, r250, r174;
}
{
add.f16x2 r256, r247, r253;
}
mov.f32 f90, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r259, {low, high};
}
mov.f32 f92, 0f3F248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r260, {low, high};
}
mov.f32 f94, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f94;
cvt.rn.f16.f32 high, f94;
mov.b32 r261, {low, high};
}
mov.f32 f96, 0f3F7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r262, {low, high};
}
mov.f32 f102, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r265, {low, high};
}
mov.f32 f104, 0f3EAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r266, {low, high};
}
{
mul.f16x2 r275, r116, r259;
}
{
mul.f16x2 r278, r152, r260;
}
{
sub.f16x2 r281, r275, r278;
}
{
mul.f16x2 r284, r116, r260;
}
{
fma.rn.f16x2 r287, r152, r259, r284;
}
{
mul.f16x2 r291, r202, r261;
}
{
mul.f16x2 r294, r238, r262;
}
{
sub.f16x2 r297, r291, r294;
}
{
mul.f16x2 r300, r202, r262;
}
{
fma.rn.f16x2 r303, r238, r261, r300;
}
{
mul.f16x2 r307, r134, r261;
}
{
mul.f16x2 r310, r170, r262;
}
{
sub.f16x2 r313, r307, r310;
}
{
mul.f16x2 r316, r134, r262;
}
{
fma.rn.f16x2 r319, r170, r261, r316;
}
{
mul.f16x2 r323, r220, r265;
}
{
mul.f16x2 r326, r256, r266;
}
{
sub.f16x2 r329, r323, r326;
}
{
mul.f16x2 r332, r220, r266;
}
{
fma.rn.f16x2 r335, r256, r265, r332;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r339, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r340, {low, high};
}
{
add.f16x2 r341, r92, r178;
}
{
add.f16x2 r344, r6, r341;
}
{
add.f16x2 r347, r98, r184;
}
{
add.f16x2 r350, r12, r347;
}
{
add.f16x2 r353, r92, r178;
}
{
mul.f16x2 r356, r353, r339;
}
{
add.f16x2 r359, r6, r356;
}
{
sub.f16x2 r362, r98, r184;
}
{
mul.f16x2 r365, r362, r340;
}
{
add.f16x2 r368, r359, r365;
}
{
add.f16x2 r371, r92, r178;
}
{
mul.f16x2 r374, r371, r339;
}
{
add.f16x2 r377, r6, r374;
}
{
sub.f16x2 r380, r98, r184;
}
{
mul.f16x2 r383, r380, r340;
}
{
sub.f16x2 r386, r377, r383;
}
{
add.f16x2 r389, r98, r184;
}
{
mul.f16x2 r392, r389, r339;
}
{
add.f16x2 r395, r12, r392;
}
{
sub.f16x2 r398, r92, r178;
}
{
mul.f16x2 r401, r398, r340;
}
{
sub.f16x2 r404, r395, r401;
}
{
add.f16x2 r407, r98, r184;
}
{
mul.f16x2 r410, r407, r339;
}
{
add.f16x2 r413, r12, r410;
}
{
sub.f16x2 r416, r92, r178;
}
{
mul.f16x2 r419, r416, r340;
}
{
add.f16x2 r422, r413, r419;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r425, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r426, {low, high};
}
{
add.f16x2 r427, r281, r297;
}
{
add.f16x2 r430, r30, r427;
}
{
add.f16x2 r433, r287, r303;
}
{
add.f16x2 r436, r66, r433;
}
{
add.f16x2 r439, r281, r297;
}
{
mul.f16x2 r442, r439, r425;
}
{
add.f16x2 r445, r30, r442;
}
{
sub.f16x2 r448, r287, r303;
}
{
mul.f16x2 r451, r448, r426;
}
{
add.f16x2 r454, r445, r451;
}
{
add.f16x2 r457, r281, r297;
}
{
mul.f16x2 r460, r457, r425;
}
{
add.f16x2 r463, r30, r460;
}
{
sub.f16x2 r466, r287, r303;
}
{
mul.f16x2 r469, r466, r426;
}
{
sub.f16x2 r472, r463, r469;
}
{
add.f16x2 r475, r287, r303;
}
{
mul.f16x2 r478, r475, r425;
}
{
add.f16x2 r481, r66, r478;
}
{
sub.f16x2 r484, r281, r297;
}
{
mul.f16x2 r487, r484, r426;
}
{
sub.f16x2 r490, r481, r487;
}
{
add.f16x2 r493, r287, r303;
}
{
mul.f16x2 r496, r493, r425;
}
{
add.f16x2 r499, r66, r496;
}
{
sub.f16x2 r502, r281, r297;
}
{
mul.f16x2 r505, r502, r426;
}
{
add.f16x2 r508, r499, r505;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r512, {low, high};
}
{
add.f16x2 r513, r313, r329;
}
{
add.f16x2 r516, r48, r513;
}
{
add.f16x2 r519, r319, r335;
}
{
add.f16x2 r522, r84, r519;
}
{
add.f16x2 r525, r313, r329;
}
{
mul.f16x2 r528, r525, r511;
}
{
add.f16x2 r531, r48, r528;
}
{
sub.f16x2 r534, r319, r335;
}
{
mul.f16x2 r537, r534, r512;
}
{
add.f16x2 r540, r531, r537;
}
{
add.f16x2 r543, r313, r329;
}
{
mul.f16x2 r546, r543, r511;
}
{
add.f16x2 r549, r48, r546;
}
{
sub.f16x2 r552, r319, r335;
}
{
mul.f16x2 r555, r552, r512;
}
{
sub.f16x2 r558, r549, r555;
}
{
add.f16x2 r561, r319, r335;
}
{
mul.f16x2 r564, r561, r511;
}
{
add.f16x2 r567, r84, r564;
}
{
sub.f16x2 r570, r313, r329;
}
{
mul.f16x2 r573, r570, r512;
}
{
sub.f16x2 r576, r567, r573;
}
{
add.f16x2 r579, r319, r335;
}
{
mul.f16x2 r582, r579, r511;
}
{
add.f16x2 r585, r84, r582;
}
{
sub.f16x2 r588, r313, r329;
}
{
mul.f16x2 r591, r588, r512;
}
{
add.f16x2 r594, r585, r591;
}
mul.wide.u32 rd2, r1495, 954437177;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r1496, rd3;
mul.lo.s32 r1497, r1496, 9;
sub.s32 r1498, r1495, r1497;
cvt.rn.f32.u32 f133, r1498;
mul.f32 f134, f133, 0f3D9EDD1F;
cos.approx.f32 f57, f134;
sin.approx.f32 f135, f134;
neg.f32 f58, f135;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f57;
cvt.rn.f16.f32 high, f58;
mov.b32 r597, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r600, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r602, {high, high};
}
{
mul.f16x2 r604, r436, r602;
}
{
fma.rn.f16x2 r607, r430, r600, r604;
}
{
mul.f16x2 r611, r430, r602;
}
{
neg.f16x2 r614, r611;
}
{
fma.rn.f16x2 r616, r436, r600, r614;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r620, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r622, {high, high};
}
mov.f32 f73, 0fBF800000;
mov.f32 f74, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r624, {low, high};
}
{
mul.f16x2 r625, r622, r624;
}
{
mul.f16x2 r628, r597, r620;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r631, {high, low};
}
{
fma.rn.f16x2 r633, r625, r631, r628;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r633;
mov.b32 r637, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r633;
mov.b32 r639, {high, high};
}
{
mul.f16x2 r641, r522, r639;
}
{
fma.rn.f16x2 r644, r516, r637, r641;
}
{
mul.f16x2 r648, r516, r639;
}
{
neg.f16x2 r651, r648;
}
{
fma.rn.f16x2 r653, r522, r637, r651;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r657, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r659, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r661, {low, high};
}
{
mul.f16x2 r662, r659, r661;
}
{
mul.f16x2 r665, r633, r657;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r633;
mov.b32 r668, {high, low};
}
{
fma.rn.f16x2 r670, r662, r668, r665;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r670;
mov.b32 r674, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r670;
mov.b32 r676, {high, high};
}
{
mul.f16x2 r678, r404, r676;
}
{
fma.rn.f16x2 r681, r368, r674, r678;
}
{
mul.f16x2 r685, r368, r676;
}
{
neg.f16x2 r688, r685;
}
{
fma.rn.f16x2 r690, r404, r674, r688;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r694, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r696, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r698, {low, high};
}
{
mul.f16x2 r699, r696, r698;
}
{
mul.f16x2 r702, r670, r694;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r670;
mov.b32 r705, {high, low};
}
{
fma.rn.f16x2 r707, r699, r705, r702;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r707;
mov.b32 r711, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r707;
mov.b32 r713, {high, high};
}
{
mul.f16x2 r715, r490, r713;
}
{
fma.rn.f16x2 r718, r454, r711, r715;
}
{
mul.f16x2 r722, r454, r713;
}
{
neg.f16x2 r725, r722;
}
{
fma.rn.f16x2 r727, r490, r711, r725;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r731, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r733, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r735, {low, high};
}
{
mul.f16x2 r736, r733, r735;
}
{
mul.f16x2 r739, r707, r731;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r707;
mov.b32 r742, {high, low};
}
{
fma.rn.f16x2 r744, r736, r742, r739;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r744;
mov.b32 r748, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r744;
mov.b32 r750, {high, high};
}
{
mul.f16x2 r752, r576, r750;
}
{
fma.rn.f16x2 r755, r540, r748, r752;
}
{
mul.f16x2 r759, r540, r750;
}
{
neg.f16x2 r762, r759;
}
{
fma.rn.f16x2 r764, r576, r748, r762;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r768, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r770, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r772, {low, high};
}
{
mul.f16x2 r773, r770, r772;
}
{
mul.f16x2 r776, r744, r768;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r744;
mov.b32 r779, {high, low};
}
{
fma.rn.f16x2 r781, r773, r779, r776;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r781;
mov.b32 r785, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r781;
mov.b32 r787, {high, high};
}
{
mul.f16x2 r789, r422, r787;
}
{
fma.rn.f16x2 r792, r386, r785, r789;
}
{
mul.f16x2 r796, r386, r787;
}
{
neg.f16x2 r799, r796;
}
{
fma.rn.f16x2 r801, r422, r785, r799;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r805, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r807, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r809, {low, high};
}
{
mul.f16x2 r810, r807, r809;
}
{
mul.f16x2 r813, r781, r805;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r781;
mov.b32 r816, {high, low};
}
{
fma.rn.f16x2 r818, r810, r816, r813;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r818;
mov.b32 r822, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r818;
mov.b32 r824, {high, high};
}
{
mul.f16x2 r826, r508, r824;
}
{
fma.rn.f16x2 r829, r472, r822, r826;
}
{
mul.f16x2 r833, r472, r824;
}
{
neg.f16x2 r836, r833;
}
{
fma.rn.f16x2 r838, r508, r822, r836;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r842, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r844, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r846, {low, high};
}
{
mul.f16x2 r847, r844, r846;
}
{
mul.f16x2 r850, r818, r842;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r818;
mov.b32 r853, {high, low};
}
{
fma.rn.f16x2 r855, r847, r853, r850;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r855;
mov.b32 r859, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r855;
mov.b32 r861, {high, high};
}
{
mul.f16x2 r863, r594, r861;
}
{
fma.rn.f16x2 r866, r558, r859, r863;
}
{
mul.f16x2 r870, r558, r861;
}
{
neg.f16x2 r873, r870;
}
{
fma.rn.f16x2 r875, r594, r859, r873;
}
mad.lo.s32 r1499, r1496, 648, r1494;
barrier.sync 0;
mad.lo.s32 r1500, r1498, 72, r1499;
st.shared.v2.f32 [r1500], {r344, r350};
st.shared.v2.f32 [r1500+8], {r607, r616};
st.shared.v2.f32 [r1500+16], {r644, r653};
st.shared.v2.f32 [r1500+24], {r681, r690};
st.shared.v2.f32 [r1500+32], {r718, r727};
st.shared.v2.f32 [r1500+40], {r755, r764};
st.shared.v2.f32 [r1500+48], {r792, r801};
st.shared.v2.f32 [r1500+56], {r829, r838};
st.shared.v2.f32 [r1500+64], {r866, r875};
barrier.sync 0;
shl.b32 r1501, r1498, 6;
sub.s32 r1502, r1500, r1501;
ld.shared.u32 r902, [r1502];
ld.shared.u32 r908, [r1502+4];
ld.shared.u32 r988, [r1502+72];
ld.shared.u32 r994, [r1502+76];
ld.shared.u32 r1074, [r1502+144];
ld.shared.u32 r1080, [r1502+148];
ld.shared.u32 r899, [r1502+216];
ld.shared.u32 r905, [r1502+220];
ld.shared.u32 r985, [r1502+288];
ld.shared.u32 r991, [r1502+292];
ld.shared.u32 r1071, [r1502+360];
ld.shared.u32 r1077, [r1502+364];
ld.shared.u32 r900, [r1502+432];
ld.shared.u32 r906, [r1502+436];
ld.shared.u32 r986, [r1502+504];
ld.shared.u32 r992, [r1502+508];
ld.shared.u32 r1072, [r1502+576];
ld.shared.u32 r1078, [r1502+580];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r896, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r897, {low, high};
}
{
add.f16x2 r898, r899, r900;
}
{
add.f16x2 r901, r902, r898;
}
{
add.f16x2 r904, r905, r906;
}
{
add.f16x2 r907, r908, r904;
}
{
add.f16x2 r910, r899, r900;
}
{
mul.f16x2 r913, r910, r896;
}
{
add.f16x2 r916, r902, r913;
}
{
sub.f16x2 r919, r905, r906;
}
{
mul.f16x2 r922, r919, r897;
}
{
add.f16x2 r925, r916, r922;
}
{
add.f16x2 r928, r899, r900;
}
{
mul.f16x2 r931, r928, r896;
}
{
add.f16x2 r934, r902, r931;
}
{
sub.f16x2 r937, r905, r906;
}
{
mul.f16x2 r940, r937, r897;
}
{
sub.f16x2 r943, r934, r940;
}
{
add.f16x2 r946, r905, r906;
}
{
mul.f16x2 r949, r946, r896;
}
{
add.f16x2 r952, r908, r949;
}
{
sub.f16x2 r955, r899, r900;
}
{
mul.f16x2 r958, r955, r897;
}
{
sub.f16x2 r961, r952, r958;
}
{
add.f16x2 r964, r905, r906;
}
{
mul.f16x2 r967, r964, r896;
}
{
add.f16x2 r970, r908, r967;
}
{
sub.f16x2 r973, r899, r900;
}
{
mul.f16x2 r976, r973, r897;
}
{
add.f16x2 r979, r970, r976;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r982, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r983, {low, high};
}
{
add.f16x2 r984, r985, r986;
}
{
add.f16x2 r987, r988, r984;
}
{
add.f16x2 r990, r991, r992;
}
{
add.f16x2 r993, r994, r990;
}
{
add.f16x2 r996, r985, r986;
}
{
mul.f16x2 r999, r996, r982;
}
{
add.f16x2 r1002, r988, r999;
}
{
sub.f16x2 r1005, r991, r992;
}
{
mul.f16x2 r1008, r1005, r983;
}
{
add.f16x2 r1011, r1002, r1008;
}
{
add.f16x2 r1014, r985, r986;
}
{
mul.f16x2 r1017, r1014, r982;
}
{
add.f16x2 r1020, r988, r1017;
}
{
sub.f16x2 r1023, r991, r992;
}
{
mul.f16x2 r1026, r1023, r983;
}
{
sub.f16x2 r1029, r1020, r1026;
}
{
add.f16x2 r1032, r991, r992;
}
{
mul.f16x2 r1035, r1032, r982;
}
{
add.f16x2 r1038, r994, r1035;
}
{
sub.f16x2 r1041, r985, r986;
}
{
mul.f16x2 r1044, r1041, r983;
}
{
sub.f16x2 r1047, r1038, r1044;
}
{
add.f16x2 r1050, r991, r992;
}
{
mul.f16x2 r1053, r1050, r982;
}
{
add.f16x2 r1056, r994, r1053;
}
{
sub.f16x2 r1059, r985, r986;
}
{
mul.f16x2 r1062, r1059, r983;
}
{
add.f16x2 r1065, r1056, r1062;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r1068, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r1069, {low, high};
}
{
add.f16x2 r1070, r1071, r1072;
}
{
add.f16x2 r1073, r1074, r1070;
}
{
add.f16x2 r1076, r1077, r1078;
}
{
add.f16x2 r1079, r1080, r1076;
}
{
add.f16x2 r1082, r1071, r1072;
}
{
mul.f16x2 r1085, r1082, r1068;
}
{
add.f16x2 r1088, r1074, r1085;
}
{
sub.f16x2 r1091, r1077, r1078;
}
{
mul.f16x2 r1094, r1091, r1069;
}
{
add.f16x2 r1097, r1088, r1094;
}
{
add.f16x2 r1100, r1071, r1072;
}
{
mul.f16x2 r1103, r1100, r1068;
}
{
add.f16x2 r1106, r1074, r1103;
}
{
sub.f16x2 r1109, r1077, r1078;
}
{
mul.f16x2 r1112, r1109, r1069;
}
{
sub.f16x2 r1115, r1106, r1112;
}
{
add.f16x2 r1118, r1077, r1078;
}
{
mul.f16x2 r1121, r1118, r1068;
}
{
add.f16x2 r1124, r1080, r1121;
}
{
sub.f16x2 r1127, r1071, r1072;
}
{
mul.f16x2 r1130, r1127, r1069;
}
{
sub.f16x2 r1133, r1124, r1130;
}
{
add.f16x2 r1136, r1077, r1078;
}
{
mul.f16x2 r1139, r1136, r1068;
}
{
add.f16x2 r1142, r1080, r1139;
}
{
sub.f16x2 r1145, r1071, r1072;
}
{
mul.f16x2 r1148, r1145, r1069;
}
{
add.f16x2 r1151, r1142, r1148;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r1154, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r1155, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f94;
cvt.rn.f16.f32 high, f94;
mov.b32 r1156, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r1157, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r1160, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r1161, {low, high};
}
{
mul.f16x2 r1170, r1011, r1154;
}
{
mul.f16x2 r1173, r1047, r1155;
}
{
sub.f16x2 r1176, r1170, r1173;
}
{
mul.f16x2 r1179, r1011, r1155;
}
{
fma.rn.f16x2 r1182, r1047, r1154, r1179;
}
{
mul.f16x2 r1186, r1097, r1156;
}
{
mul.f16x2 r1189, r1133, r1157;
}
{
sub.f16x2 r1192, r1186, r1189;
}
{
mul.f16x2 r1195, r1097, r1157;
}
{
fma.rn.f16x2 r1198, r1133, r1156, r1195;
}
{
mul.f16x2 r1202, r1029, r1156;
}
{
mul.f16x2 r1205, r1065, r1157;
}
{
sub.f16x2 r1208, r1202, r1205;
}
{
mul.f16x2 r1211, r1029, r1157;
}
{
fma.rn.f16x2 r1214, r1065, r1156, r1211;
}
{
mul.f16x2 r1218, r1115, r1160;
}
{
mul.f16x2 r1221, r1151, r1161;
}
{
sub.f16x2 r1224, r1218, r1221;
}
{
mul.f16x2 r1227, r1115, r1161;
}
{
fma.rn.f16x2 r1230, r1151, r1160, r1227;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r1234, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r1235, {low, high};
}
{
add.f16x2 r1236, r987, r1073;
}
{
add.f16x2 %0, r901, r1236;
}
{
add.f16x2 r1242, r993, r1079;
}
{
add.f16x2 %1, r907, r1242;
}
{
add.f16x2 r1248, r987, r1073;
}
{
mul.f16x2 r1251, r1248, r1234;
}
{
add.f16x2 r1254, r901, r1251;
}
{
sub.f16x2 r1257, r993, r1079;
}
{
mul.f16x2 r1260, r1257, r1235;
}
{
add.f16x2 %6, r1254, r1260;
}
{
add.f16x2 r1266, r987, r1073;
}
{
mul.f16x2 r1269, r1266, r1234;
}
{
add.f16x2 r1272, r901, r1269;
}
{
sub.f16x2 r1275, r993, r1079;
}
{
mul.f16x2 r1278, r1275, r1235;
}
{
sub.f16x2 %12, r1272, r1278;
}
{
add.f16x2 r1284, r993, r1079;
}
{
mul.f16x2 r1287, r1284, r1234;
}
{
add.f16x2 r1290, r907, r1287;
}
{
sub.f16x2 r1293, r987, r1073;
}
{
mul.f16x2 r1296, r1293, r1235;
}
{
sub.f16x2 %7, r1290, r1296;
}
{
add.f16x2 r1302, r993, r1079;
}
{
mul.f16x2 r1305, r1302, r1234;
}
{
add.f16x2 r1308, r907, r1305;
}
{
sub.f16x2 r1311, r987, r1073;
}
{
mul.f16x2 r1314, r1311, r1235;
}
{
add.f16x2 %13, r1308, r1314;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r1320, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r1321, {low, high};
}
{
add.f16x2 r1322, r1176, r1192;
}
{
add.f16x2 %2, r925, r1322;
}
{
add.f16x2 r1328, r1182, r1198;
}
{
add.f16x2 %3, r961, r1328;
}
{
add.f16x2 r1334, r1176, r1192;
}
{
mul.f16x2 r1337, r1334, r1320;
}
{
add.f16x2 r1340, r925, r1337;
}
{
sub.f16x2 r1343, r1182, r1198;
}
{
mul.f16x2 r1346, r1343, r1321;
}
{
add.f16x2 %8, r1340, r1346;
}
{
add.f16x2 r1352, r1176, r1192;
}
{
mul.f16x2 r1355, r1352, r1320;
}
{
add.f16x2 r1358, r925, r1355;
}
{
sub.f16x2 r1361, r1182, r1198;
}
{
mul.f16x2 r1364, r1361, r1321;
}
{
sub.f16x2 %14, r1358, r1364;
}
{
add.f16x2 r1370, r1182, r1198;
}
{
mul.f16x2 r1373, r1370, r1320;
}
{
add.f16x2 r1376, r961, r1373;
}
{
sub.f16x2 r1379, r1176, r1192;
}
{
mul.f16x2 r1382, r1379, r1321;
}
{
sub.f16x2 %9, r1376, r1382;
}
{
add.f16x2 r1388, r1182, r1198;
}
{
mul.f16x2 r1391, r1388, r1320;
}
{
add.f16x2 r1394, r961, r1391;
}
{
sub.f16x2 r1397, r1176, r1192;
}
{
mul.f16x2 r1400, r1397, r1321;
}
{
add.f16x2 %15, r1394, r1400;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r1406, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r1407, {low, high};
}
{
add.f16x2 r1408, r1208, r1224;
}
{
add.f16x2 %4, r943, r1408;
}
{
add.f16x2 r1414, r1214, r1230;
}
{
add.f16x2 %5, r979, r1414;
}
{
add.f16x2 r1420, r1208, r1224;
}
{
mul.f16x2 r1423, r1420, r1406;
}
{
add.f16x2 r1426, r943, r1423;
}
{
sub.f16x2 r1429, r1214, r1230;
}
{
mul.f16x2 r1432, r1429, r1407;
}
{
add.f16x2 %10, r1426, r1432;
}
{
add.f16x2 r1438, r1208, r1224;
}
{
mul.f16x2 r1441, r1438, r1406;
}
{
add.f16x2 r1444, r943, r1441;
}
{
sub.f16x2 r1447, r1214, r1230;
}
{
mul.f16x2 r1450, r1447, r1407;
}
{
sub.f16x2 %16, r1444, r1450;
}
{
add.f16x2 r1456, r1214, r1230;
}
{
mul.f16x2 r1459, r1456, r1406;
}
{
add.f16x2 r1462, r979, r1459;
}
{
sub.f16x2 r1465, r1208, r1224;
}
{
mul.f16x2 r1468, r1465, r1407;
}
{
sub.f16x2 %11, r1462, r1468;
}
{
add.f16x2 r1474, r1214, r1230;
}
{
mul.f16x2 r1477, r1474, r1406;
}
{
add.f16x2 r1480, r979, r1477;
}
{
sub.f16x2 r1483, r1208, r1224;
}
{
mul.f16x2 r1486, r1483, r1407;
}
{
add.f16x2 %17, r1480, r1486;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1075, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<136>;
.reg .b32 r<1503>;
.reg .b64 rd<4>;
mov.u32 r1492, %tid.y;
mov.u32 r1493, %18;
mad.lo.s32 r1494, r1492, 324, r1493;
mov.u32 r1495, %tid.x;
mov.f32 f130, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r1, {low, high};
}
mov.f32 f132, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r2, {low, high};
}
{
add.f16x2 r3, %25, %31;
}
{
add.f16x2 r6, %19, r3;
}
{
add.f16x2 r9, %26, %32;
}
{
add.f16x2 r12, %20, r9;
}
{
add.f16x2 r15, %25, %31;
}
{
mul.f16x2 r18, r15, r1;
}
{
add.f16x2 r21, %19, r18;
}
{
sub.f16x2 r24, %26, %32;
}
{
mul.f16x2 r27, r24, r2;
}
{
add.f16x2 r30, r21, r27;
}
{
add.f16x2 r33, %25, %31;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %19, r36;
}
{
sub.f16x2 r42, %26, %32;
}
{
mul.f16x2 r45, r42, r2;
}
{
sub.f16x2 r48, r39, r45;
}
{
add.f16x2 r51, %26, %32;
}
{
mul.f16x2 r54, r51, r1;
}
{
add.f16x2 r57, %20, r54;
}
{
sub.f16x2 r60, %25, %31;
}
{
mul.f16x2 r63, r60, r2;
}
{
sub.f16x2 r66, r57, r63;
}
{
add.f16x2 r69, %26, %32;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %20, r72;
}
{
sub.f16x2 r78, %25, %31;
}
{
mul.f16x2 r81, r78, r2;
}
{
add.f16x2 r84, r75, r81;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r88, {low, high};
}
{
add.f16x2 r89, %27, %33;
}
{
add.f16x2 r92, %21, r89;
}
{
add.f16x2 r95, %28, %34;
}
{
add.f16x2 r98, %22, r95;
}
{
add.f16x2 r101, %27, %33;
}
{
mul.f16x2 r104, r101, r87;
}
{
add.f16x2 r107, %21, r104;
}
{
sub.f16x2 r110, %28, %34;
}
{
mul.f16x2 r113, r110, r88;
}
{
add.f16x2 r116, r107, r113;
}
{
add.f16x2 r119, %27, %33;
}
{
mul.f16x2 r122, r119, r87;
}
{
add.f16x2 r125, %21, r122;
}
{
sub.f16x2 r128, %28, %34;
}
{
mul.f16x2 r131, r128, r88;
}
{
sub.f16x2 r134, r125, r131;
}
{
add.f16x2 r137, %28, %34;
}
{
mul.f16x2 r140, r137, r87;
}
{
add.f16x2 r143, %22, r140;
}
{
sub.f16x2 r146, %27, %33;
}
{
mul.f16x2 r149, r146, r88;
}
{
sub.f16x2 r152, r143, r149;
}
{
add.f16x2 r155, %28, %34;
}
{
mul.f16x2 r158, r155, r87;
}
{
add.f16x2 r161, %22, r158;
}
{
sub.f16x2 r164, %27, %33;
}
{
mul.f16x2 r167, r164, r88;
}
{
add.f16x2 r170, r161, r167;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r173, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r174, {low, high};
}
{
add.f16x2 r175, %29, %35;
}
{
add.f16x2 r178, %23, r175;
}
{
add.f16x2 r181, %30, %36;
}
{
add.f16x2 r184, %24, r181;
}
{
add.f16x2 r187, %29, %35;
}
{
mul.f16x2 r190, r187, r173;
}
{
add.f16x2 r193, %23, r190;
}
{
sub.f16x2 r196, %30, %36;
}
{
mul.f16x2 r199, r196, r174;
}
{
add.f16x2 r202, r193, r199;
}
{
add.f16x2 r205, %29, %35;
}
{
mul.f16x2 r208, r205, r173;
}
{
add.f16x2 r211, %23, r208;
}
{
sub.f16x2 r214, %30, %36;
}
{
mul.f16x2 r217, r214, r174;
}
{
sub.f16x2 r220, r211, r217;
}
{
add.f16x2 r223, %30, %36;
}
{
mul.f16x2 r226, r223, r173;
}
{
add.f16x2 r229, %24, r226;
}
{
sub.f16x2 r232, %29, %35;
}
{
mul.f16x2 r235, r232, r174;
}
{
sub.f16x2 r238, r229, r235;
}
{
add.f16x2 r241, %30, %36;
}
{
mul.f16x2 r244, r241, r173;
}
{
add.f16x2 r247, %24, r244;
}
{
sub.f16x2 r250, %29, %35;
}
{
mul.f16x2 r253, r250, r174;
}
{
add.f16x2 r256, r247, r253;
}
mov.f32 f90, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r259, {low, high};
}
mov.f32 f92, 0f3F248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r260, {low, high};
}
mov.f32 f94, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f94;
cvt.rn.f16.f32 high, f94;
mov.b32 r261, {low, high};
}
mov.f32 f96, 0f3F7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r262, {low, high};
}
mov.f32 f102, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r265, {low, high};
}
mov.f32 f104, 0f3EAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r266, {low, high};
}
{
mul.f16x2 r275, r116, r259;
}
{
mul.f16x2 r278, r152, r260;
}
{
sub.f16x2 r281, r275, r278;
}
{
mul.f16x2 r284, r116, r260;
}
{
fma.rn.f16x2 r287, r152, r259, r284;
}
{
mul.f16x2 r291, r202, r261;
}
{
mul.f16x2 r294, r238, r262;
}
{
sub.f16x2 r297, r291, r294;
}
{
mul.f16x2 r300, r202, r262;
}
{
fma.rn.f16x2 r303, r238, r261, r300;
}
{
mul.f16x2 r307, r134, r261;
}
{
mul.f16x2 r310, r170, r262;
}
{
sub.f16x2 r313, r307, r310;
}
{
mul.f16x2 r316, r134, r262;
}
{
fma.rn.f16x2 r319, r170, r261, r316;
}
{
mul.f16x2 r323, r220, r265;
}
{
mul.f16x2 r326, r256, r266;
}
{
sub.f16x2 r329, r323, r326;
}
{
mul.f16x2 r332, r220, r266;
}
{
fma.rn.f16x2 r335, r256, r265, r332;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r339, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r340, {low, high};
}
{
add.f16x2 r341, r92, r178;
}
{
add.f16x2 r344, r6, r341;
}
{
add.f16x2 r347, r98, r184;
}
{
add.f16x2 r350, r12, r347;
}
{
add.f16x2 r353, r92, r178;
}
{
mul.f16x2 r356, r353, r339;
}
{
add.f16x2 r359, r6, r356;
}
{
sub.f16x2 r362, r98, r184;
}
{
mul.f16x2 r365, r362, r340;
}
{
add.f16x2 r368, r359, r365;
}
{
add.f16x2 r371, r92, r178;
}
{
mul.f16x2 r374, r371, r339;
}
{
add.f16x2 r377, r6, r374;
}
{
sub.f16x2 r380, r98, r184;
}
{
mul.f16x2 r383, r380, r340;
}
{
sub.f16x2 r386, r377, r383;
}
{
add.f16x2 r389, r98, r184;
}
{
mul.f16x2 r392, r389, r339;
}
{
add.f16x2 r395, r12, r392;
}
{
sub.f16x2 r398, r92, r178;
}
{
mul.f16x2 r401, r398, r340;
}
{
sub.f16x2 r404, r395, r401;
}
{
add.f16x2 r407, r98, r184;
}
{
mul.f16x2 r410, r407, r339;
}
{
add.f16x2 r413, r12, r410;
}
{
sub.f16x2 r416, r92, r178;
}
{
mul.f16x2 r419, r416, r340;
}
{
add.f16x2 r422, r413, r419;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r425, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r426, {low, high};
}
{
add.f16x2 r427, r281, r297;
}
{
add.f16x2 r430, r30, r427;
}
{
add.f16x2 r433, r287, r303;
}
{
add.f16x2 r436, r66, r433;
}
{
add.f16x2 r439, r281, r297;
}
{
mul.f16x2 r442, r439, r425;
}
{
add.f16x2 r445, r30, r442;
}
{
sub.f16x2 r448, r287, r303;
}
{
mul.f16x2 r451, r448, r426;
}
{
add.f16x2 r454, r445, r451;
}
{
add.f16x2 r457, r281, r297;
}
{
mul.f16x2 r460, r457, r425;
}
{
add.f16x2 r463, r30, r460;
}
{
sub.f16x2 r466, r287, r303;
}
{
mul.f16x2 r469, r466, r426;
}
{
sub.f16x2 r472, r463, r469;
}
{
add.f16x2 r475, r287, r303;
}
{
mul.f16x2 r478, r475, r425;
}
{
add.f16x2 r481, r66, r478;
}
{
sub.f16x2 r484, r281, r297;
}
{
mul.f16x2 r487, r484, r426;
}
{
sub.f16x2 r490, r481, r487;
}
{
add.f16x2 r493, r287, r303;
}
{
mul.f16x2 r496, r493, r425;
}
{
add.f16x2 r499, r66, r496;
}
{
sub.f16x2 r502, r281, r297;
}
{
mul.f16x2 r505, r502, r426;
}
{
add.f16x2 r508, r499, r505;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r512, {low, high};
}
{
add.f16x2 r513, r313, r329;
}
{
add.f16x2 r516, r48, r513;
}
{
add.f16x2 r519, r319, r335;
}
{
add.f16x2 r522, r84, r519;
}
{
add.f16x2 r525, r313, r329;
}
{
mul.f16x2 r528, r525, r511;
}
{
add.f16x2 r531, r48, r528;
}
{
sub.f16x2 r534, r319, r335;
}
{
mul.f16x2 r537, r534, r512;
}
{
add.f16x2 r540, r531, r537;
}
{
add.f16x2 r543, r313, r329;
}
{
mul.f16x2 r546, r543, r511;
}
{
add.f16x2 r549, r48, r546;
}
{
sub.f16x2 r552, r319, r335;
}
{
mul.f16x2 r555, r552, r512;
}
{
sub.f16x2 r558, r549, r555;
}
{
add.f16x2 r561, r319, r335;
}
{
mul.f16x2 r564, r561, r511;
}
{
add.f16x2 r567, r84, r564;
}
{
sub.f16x2 r570, r313, r329;
}
{
mul.f16x2 r573, r570, r512;
}
{
sub.f16x2 r576, r567, r573;
}
{
add.f16x2 r579, r319, r335;
}
{
mul.f16x2 r582, r579, r511;
}
{
add.f16x2 r585, r84, r582;
}
{
sub.f16x2 r588, r313, r329;
}
{
mul.f16x2 r591, r588, r512;
}
{
add.f16x2 r594, r585, r591;
}
mul.wide.u32 rd2, r1495, 954437177;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r1496, rd3;
mul.lo.s32 r1497, r1496, 9;
sub.s32 r1498, r1495, r1497;
mad.lo.s32 r1499, r1496, 324, r1494;
cvt.rn.f32.u32 f133, r1498;
mul.f32 f134, f133, 0f3D9EDD1F;
cos.approx.f32 f57, f134;
sin.approx.f32 f135, f134;
neg.f32 f58, f135;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f57;
cvt.rn.f16.f32 high, f58;
mov.b32 r597, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r600, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r602, {high, high};
}
{
mul.f16x2 r604, r436, r602;
}
{
fma.rn.f16x2 r607, r430, r600, r604;
}
{
mul.f16x2 r611, r430, r602;
}
{
neg.f16x2 r614, r611;
}
{
fma.rn.f16x2 r616, r436, r600, r614;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r620, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r622, {high, high};
}
mov.f32 f73, 0fBF800000;
mov.f32 f74, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r624, {low, high};
}
{
mul.f16x2 r625, r622, r624;
}
{
mul.f16x2 r628, r597, r620;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r631, {high, low};
}
{
fma.rn.f16x2 r633, r625, r631, r628;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r633;
mov.b32 r637, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r633;
mov.b32 r639, {high, high};
}
{
mul.f16x2 r641, r522, r639;
}
{
fma.rn.f16x2 r644, r516, r637, r641;
}
{
mul.f16x2 r648, r516, r639;
}
{
neg.f16x2 r651, r648;
}
{
fma.rn.f16x2 r653, r522, r637, r651;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r657, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r659, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r661, {low, high};
}
{
mul.f16x2 r662, r659, r661;
}
{
mul.f16x2 r665, r633, r657;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r633;
mov.b32 r668, {high, low};
}
{
fma.rn.f16x2 r670, r662, r668, r665;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r670;
mov.b32 r674, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r670;
mov.b32 r676, {high, high};
}
{
mul.f16x2 r678, r404, r676;
}
{
fma.rn.f16x2 r681, r368, r674, r678;
}
{
mul.f16x2 r685, r368, r676;
}
{
neg.f16x2 r688, r685;
}
{
fma.rn.f16x2 r690, r404, r674, r688;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r694, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r696, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r698, {low, high};
}
{
mul.f16x2 r699, r696, r698;
}
{
mul.f16x2 r702, r670, r694;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r670;
mov.b32 r705, {high, low};
}
{
fma.rn.f16x2 r707, r699, r705, r702;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r707;
mov.b32 r711, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r707;
mov.b32 r713, {high, high};
}
{
mul.f16x2 r715, r490, r713;
}
{
fma.rn.f16x2 r718, r454, r711, r715;
}
{
mul.f16x2 r722, r454, r713;
}
{
neg.f16x2 r725, r722;
}
{
fma.rn.f16x2 r727, r490, r711, r725;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r731, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r733, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r735, {low, high};
}
{
mul.f16x2 r736, r733, r735;
}
{
mul.f16x2 r739, r707, r731;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r707;
mov.b32 r742, {high, low};
}
{
fma.rn.f16x2 r744, r736, r742, r739;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r744;
mov.b32 r748, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r744;
mov.b32 r750, {high, high};
}
{
mul.f16x2 r752, r576, r750;
}
{
fma.rn.f16x2 r755, r540, r748, r752;
}
{
mul.f16x2 r759, r540, r750;
}
{
neg.f16x2 r762, r759;
}
{
fma.rn.f16x2 r764, r576, r748, r762;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r768, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r770, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r772, {low, high};
}
{
mul.f16x2 r773, r770, r772;
}
{
mul.f16x2 r776, r744, r768;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r744;
mov.b32 r779, {high, low};
}
{
fma.rn.f16x2 r781, r773, r779, r776;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r781;
mov.b32 r785, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r781;
mov.b32 r787, {high, high};
}
{
mul.f16x2 r789, r422, r787;
}
{
fma.rn.f16x2 r792, r386, r785, r789;
}
{
mul.f16x2 r796, r386, r787;
}
{
neg.f16x2 r799, r796;
}
{
fma.rn.f16x2 r801, r422, r785, r799;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r805, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r807, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r809, {low, high};
}
{
mul.f16x2 r810, r807, r809;
}
{
mul.f16x2 r813, r781, r805;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r781;
mov.b32 r816, {high, low};
}
{
fma.rn.f16x2 r818, r810, r816, r813;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r818;
mov.b32 r822, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r818;
mov.b32 r824, {high, high};
}
{
mul.f16x2 r826, r508, r824;
}
{
fma.rn.f16x2 r829, r472, r822, r826;
}
{
mul.f16x2 r833, r472, r824;
}
{
neg.f16x2 r836, r833;
}
{
fma.rn.f16x2 r838, r508, r822, r836;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r842, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r844, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r846, {low, high};
}
{
mul.f16x2 r847, r844, r846;
}
{
mul.f16x2 r850, r818, r842;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r818;
mov.b32 r853, {high, low};
}
{
fma.rn.f16x2 r855, r847, r853, r850;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r855;
mov.b32 r859, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r855;
mov.b32 r861, {high, high};
}
{
mul.f16x2 r863, r594, r861;
}
{
fma.rn.f16x2 r866, r558, r859, r863;
}
{
mul.f16x2 r870, r558, r861;
}
{
neg.f16x2 r873, r870;
}
{
fma.rn.f16x2 r875, r594, r859, r873;
}
barrier.sync 0;
mad.lo.s32 r1500, r1498, 36, r1499;
st.shared.u32 [r1500], r344;
st.shared.u32 [r1500+4], r607;
st.shared.u32 [r1500+8], r644;
st.shared.u32 [r1500+12], r681;
st.shared.u32 [r1500+16], r718;
st.shared.u32 [r1500+20], r755;
st.shared.u32 [r1500+24], r792;
st.shared.u32 [r1500+28], r829;
st.shared.u32 [r1500+32], r866;
barrier.sync 0;
shl.b32 r1501, r1498, 5;
sub.s32 r1502, r1500, r1501;
ld.shared.u32 r902, [r1502];
ld.shared.u32 r988, [r1502+36];
ld.shared.u32 r1074, [r1502+72];
ld.shared.u32 r899, [r1502+108];
ld.shared.u32 r985, [r1502+144];
ld.shared.u32 r1071, [r1502+180];
ld.shared.u32 r900, [r1502+216];
ld.shared.u32 r986, [r1502+252];
ld.shared.u32 r1072, [r1502+288];
barrier.sync 0;
st.shared.u32 [r1500], r350;
st.shared.u32 [r1500+4], r616;
st.shared.u32 [r1500+8], r653;
st.shared.u32 [r1500+12], r690;
st.shared.u32 [r1500+16], r727;
st.shared.u32 [r1500+20], r764;
st.shared.u32 [r1500+24], r801;
st.shared.u32 [r1500+28], r838;
st.shared.u32 [r1500+32], r875;
barrier.sync 0;
ld.shared.u32 r908, [r1502];
ld.shared.u32 r994, [r1502+36];
ld.shared.u32 r1080, [r1502+72];
ld.shared.u32 r905, [r1502+108];
ld.shared.u32 r991, [r1502+144];
ld.shared.u32 r1077, [r1502+180];
ld.shared.u32 r906, [r1502+216];
ld.shared.u32 r992, [r1502+252];
ld.shared.u32 r1078, [r1502+288];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r896, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r897, {low, high};
}
{
add.f16x2 r898, r899, r900;
}
{
add.f16x2 r901, r902, r898;
}
{
add.f16x2 r904, r905, r906;
}
{
add.f16x2 r907, r908, r904;
}
{
add.f16x2 r910, r899, r900;
}
{
mul.f16x2 r913, r910, r896;
}
{
add.f16x2 r916, r902, r913;
}
{
sub.f16x2 r919, r905, r906;
}
{
mul.f16x2 r922, r919, r897;
}
{
add.f16x2 r925, r916, r922;
}
{
add.f16x2 r928, r899, r900;
}
{
mul.f16x2 r931, r928, r896;
}
{
add.f16x2 r934, r902, r931;
}
{
sub.f16x2 r937, r905, r906;
}
{
mul.f16x2 r940, r937, r897;
}
{
sub.f16x2 r943, r934, r940;
}
{
add.f16x2 r946, r905, r906;
}
{
mul.f16x2 r949, r946, r896;
}
{
add.f16x2 r952, r908, r949;
}
{
sub.f16x2 r955, r899, r900;
}
{
mul.f16x2 r958, r955, r897;
}
{
sub.f16x2 r961, r952, r958;
}
{
add.f16x2 r964, r905, r906;
}
{
mul.f16x2 r967, r964, r896;
}
{
add.f16x2 r970, r908, r967;
}
{
sub.f16x2 r973, r899, r900;
}
{
mul.f16x2 r976, r973, r897;
}
{
add.f16x2 r979, r970, r976;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r982, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r983, {low, high};
}
{
add.f16x2 r984, r985, r986;
}
{
add.f16x2 r987, r988, r984;
}
{
add.f16x2 r990, r991, r992;
}
{
add.f16x2 r993, r994, r990;
}
{
add.f16x2 r996, r985, r986;
}
{
mul.f16x2 r999, r996, r982;
}
{
add.f16x2 r1002, r988, r999;
}
{
sub.f16x2 r1005, r991, r992;
}
{
mul.f16x2 r1008, r1005, r983;
}
{
add.f16x2 r1011, r1002, r1008;
}
{
add.f16x2 r1014, r985, r986;
}
{
mul.f16x2 r1017, r1014, r982;
}
{
add.f16x2 r1020, r988, r1017;
}
{
sub.f16x2 r1023, r991, r992;
}
{
mul.f16x2 r1026, r1023, r983;
}
{
sub.f16x2 r1029, r1020, r1026;
}
{
add.f16x2 r1032, r991, r992;
}
{
mul.f16x2 r1035, r1032, r982;
}
{
add.f16x2 r1038, r994, r1035;
}
{
sub.f16x2 r1041, r985, r986;
}
{
mul.f16x2 r1044, r1041, r983;
}
{
sub.f16x2 r1047, r1038, r1044;
}
{
add.f16x2 r1050, r991, r992;
}
{
mul.f16x2 r1053, r1050, r982;
}
{
add.f16x2 r1056, r994, r1053;
}
{
sub.f16x2 r1059, r985, r986;
}
{
mul.f16x2 r1062, r1059, r983;
}
{
add.f16x2 r1065, r1056, r1062;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r1068, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r1069, {low, high};
}
{
add.f16x2 r1070, r1071, r1072;
}
{
add.f16x2 r1073, r1074, r1070;
}
{
add.f16x2 r1076, r1077, r1078;
}
{
add.f16x2 r1079, r1080, r1076;
}
{
add.f16x2 r1082, r1071, r1072;
}
{
mul.f16x2 r1085, r1082, r1068;
}
{
add.f16x2 r1088, r1074, r1085;
}
{
sub.f16x2 r1091, r1077, r1078;
}
{
mul.f16x2 r1094, r1091, r1069;
}
{
add.f16x2 r1097, r1088, r1094;
}
{
add.f16x2 r1100, r1071, r1072;
}
{
mul.f16x2 r1103, r1100, r1068;
}
{
add.f16x2 r1106, r1074, r1103;
}
{
sub.f16x2 r1109, r1077, r1078;
}
{
mul.f16x2 r1112, r1109, r1069;
}
{
sub.f16x2 r1115, r1106, r1112;
}
{
add.f16x2 r1118, r1077, r1078;
}
{
mul.f16x2 r1121, r1118, r1068;
}
{
add.f16x2 r1124, r1080, r1121;
}
{
sub.f16x2 r1127, r1071, r1072;
}
{
mul.f16x2 r1130, r1127, r1069;
}
{
sub.f16x2 r1133, r1124, r1130;
}
{
add.f16x2 r1136, r1077, r1078;
}
{
mul.f16x2 r1139, r1136, r1068;
}
{
add.f16x2 r1142, r1080, r1139;
}
{
sub.f16x2 r1145, r1071, r1072;
}
{
mul.f16x2 r1148, r1145, r1069;
}
{
add.f16x2 r1151, r1142, r1148;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r1154, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r1155, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f94;
cvt.rn.f16.f32 high, f94;
mov.b32 r1156, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r1157, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r1160, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r1161, {low, high};
}
{
mul.f16x2 r1170, r1011, r1154;
}
{
mul.f16x2 r1173, r1047, r1155;
}
{
sub.f16x2 r1176, r1170, r1173;
}
{
mul.f16x2 r1179, r1011, r1155;
}
{
fma.rn.f16x2 r1182, r1047, r1154, r1179;
}
{
mul.f16x2 r1186, r1097, r1156;
}
{
mul.f16x2 r1189, r1133, r1157;
}
{
sub.f16x2 r1192, r1186, r1189;
}
{
mul.f16x2 r1195, r1097, r1157;
}
{
fma.rn.f16x2 r1198, r1133, r1156, r1195;
}
{
mul.f16x2 r1202, r1029, r1156;
}
{
mul.f16x2 r1205, r1065, r1157;
}
{
sub.f16x2 r1208, r1202, r1205;
}
{
mul.f16x2 r1211, r1029, r1157;
}
{
fma.rn.f16x2 r1214, r1065, r1156, r1211;
}
{
mul.f16x2 r1218, r1115, r1160;
}
{
mul.f16x2 r1221, r1151, r1161;
}
{
sub.f16x2 r1224, r1218, r1221;
}
{
mul.f16x2 r1227, r1115, r1161;
}
{
fma.rn.f16x2 r1230, r1151, r1160, r1227;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r1234, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r1235, {low, high};
}
{
add.f16x2 r1236, r987, r1073;
}
{
add.f16x2 %0, r901, r1236;
}
{
add.f16x2 r1242, r993, r1079;
}
{
add.f16x2 %1, r907, r1242;
}
{
add.f16x2 r1248, r987, r1073;
}
{
mul.f16x2 r1251, r1248, r1234;
}
{
add.f16x2 r1254, r901, r1251;
}
{
sub.f16x2 r1257, r993, r1079;
}
{
mul.f16x2 r1260, r1257, r1235;
}
{
add.f16x2 %6, r1254, r1260;
}
{
add.f16x2 r1266, r987, r1073;
}
{
mul.f16x2 r1269, r1266, r1234;
}
{
add.f16x2 r1272, r901, r1269;
}
{
sub.f16x2 r1275, r993, r1079;
}
{
mul.f16x2 r1278, r1275, r1235;
}
{
sub.f16x2 %12, r1272, r1278;
}
{
add.f16x2 r1284, r993, r1079;
}
{
mul.f16x2 r1287, r1284, r1234;
}
{
add.f16x2 r1290, r907, r1287;
}
{
sub.f16x2 r1293, r987, r1073;
}
{
mul.f16x2 r1296, r1293, r1235;
}
{
sub.f16x2 %7, r1290, r1296;
}
{
add.f16x2 r1302, r993, r1079;
}
{
mul.f16x2 r1305, r1302, r1234;
}
{
add.f16x2 r1308, r907, r1305;
}
{
sub.f16x2 r1311, r987, r1073;
}
{
mul.f16x2 r1314, r1311, r1235;
}
{
add.f16x2 %13, r1308, r1314;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r1320, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r1321, {low, high};
}
{
add.f16x2 r1322, r1176, r1192;
}
{
add.f16x2 %2, r925, r1322;
}
{
add.f16x2 r1328, r1182, r1198;
}
{
add.f16x2 %3, r961, r1328;
}
{
add.f16x2 r1334, r1176, r1192;
}
{
mul.f16x2 r1337, r1334, r1320;
}
{
add.f16x2 r1340, r925, r1337;
}
{
sub.f16x2 r1343, r1182, r1198;
}
{
mul.f16x2 r1346, r1343, r1321;
}
{
add.f16x2 %8, r1340, r1346;
}
{
add.f16x2 r1352, r1176, r1192;
}
{
mul.f16x2 r1355, r1352, r1320;
}
{
add.f16x2 r1358, r925, r1355;
}
{
sub.f16x2 r1361, r1182, r1198;
}
{
mul.f16x2 r1364, r1361, r1321;
}
{
sub.f16x2 %14, r1358, r1364;
}
{
add.f16x2 r1370, r1182, r1198;
}
{
mul.f16x2 r1373, r1370, r1320;
}
{
add.f16x2 r1376, r961, r1373;
}
{
sub.f16x2 r1379, r1176, r1192;
}
{
mul.f16x2 r1382, r1379, r1321;
}
{
sub.f16x2 %9, r1376, r1382;
}
{
add.f16x2 r1388, r1182, r1198;
}
{
mul.f16x2 r1391, r1388, r1320;
}
{
add.f16x2 r1394, r961, r1391;
}
{
sub.f16x2 r1397, r1176, r1192;
}
{
mul.f16x2 r1400, r1397, r1321;
}
{
add.f16x2 %15, r1394, r1400;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r1406, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r1407, {low, high};
}
{
add.f16x2 r1408, r1208, r1224;
}
{
add.f16x2 %4, r943, r1408;
}
{
add.f16x2 r1414, r1214, r1230;
}
{
add.f16x2 %5, r979, r1414;
}
{
add.f16x2 r1420, r1208, r1224;
}
{
mul.f16x2 r1423, r1420, r1406;
}
{
add.f16x2 r1426, r943, r1423;
}
{
sub.f16x2 r1429, r1214, r1230;
}
{
mul.f16x2 r1432, r1429, r1407;
}
{
add.f16x2 %10, r1426, r1432;
}
{
add.f16x2 r1438, r1208, r1224;
}
{
mul.f16x2 r1441, r1438, r1406;
}
{
add.f16x2 r1444, r943, r1441;
}
{
sub.f16x2 r1447, r1214, r1230;
}
{
mul.f16x2 r1450, r1447, r1407;
}
{
sub.f16x2 %16, r1444, r1450;
}
{
add.f16x2 r1456, r1214, r1230;
}
{
mul.f16x2 r1459, r1456, r1406;
}
{
add.f16x2 r1462, r979, r1459;
}
{
sub.f16x2 r1465, r1208, r1224;
}
{
mul.f16x2 r1468, r1465, r1407;
}
{
sub.f16x2 %11, r1462, r1468;
}
{
add.f16x2 r1474, r1214, r1230;
}
{
mul.f16x2 r1477, r1474, r1406;
}
{
add.f16x2 r1480, r979, r1477;
}
{
sub.f16x2 r1483, r1208, r1224;
}
{
mul.f16x2 r1486, r1483, r1407;
}
{
add.f16x2 %17, r1480, r1486;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1077, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<404>;
.reg .b32 r<4676>;
.reg .b64 rd<4>;
mov.u32 r4674, %tid.y;
mov.u32 r4675, %54;
mad.lo.s32 r4612, r4674, 648, r4675;
mov.u32 r4613, %tid.x;
mov.f32 f398, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1, {low, high};
}
mov.f32 f400, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2, {low, high};
}
{
add.f16x2 r3, %106, %91;
}
{
add.f16x2 r6, %61, r3;
}
{
add.f16x2 r9, %70, %104;
}
{
add.f16x2 r12, %76, r9;
}
{
add.f16x2 r15, %106, %91;
}
{
mul.f16x2 r18, r15, r1;
}
{
add.f16x2 r21, %61, r18;
}
{
sub.f16x2 r24, %70, %104;
}
{
mul.f16x2 r27, r24, r2;
}
{
add.f16x2 r30, r21, r27;
}
{
add.f16x2 r33, %106, %91;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %61, r36;
}
{
sub.f16x2 r42, %70, %104;
}
{
mul.f16x2 r45, r42, r2;
}
{
sub.f16x2 r48, r39, r45;
}
{
add.f16x2 r51, %70, %104;
}
{
mul.f16x2 r54, r51, r1;
}
{
add.f16x2 r57, %76, r54;
}
{
sub.f16x2 r60, %106, %91;
}
{
mul.f16x2 r63, r60, r2;
}
{
sub.f16x2 r66, r57, r63;
}
{
add.f16x2 r69, %70, %104;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %76, r72;
}
{
sub.f16x2 r78, %106, %91;
}
{
mul.f16x2 r81, r78, r2;
}
{
add.f16x2 r84, r75, r81;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r88, {low, high};
}
{
add.f16x2 r89, %105, %90;
}
{
add.f16x2 r92, %60, r89;
}
{
add.f16x2 r95, %69, %102;
}
{
add.f16x2 r98, %75, r95;
}
{
add.f16x2 r101, %105, %90;
}
{
mul.f16x2 r104, r101, r87;
}
{
add.f16x2 r107, %60, r104;
}
{
sub.f16x2 r110, %69, %102;
}
{
mul.f16x2 r113, r110, r88;
}
{
add.f16x2 r116, r107, r113;
}
{
add.f16x2 r119, %105, %90;
}
{
mul.f16x2 r122, r119, r87;
}
{
add.f16x2 r125, %60, r122;
}
{
sub.f16x2 r128, %69, %102;
}
{
mul.f16x2 r131, r128, r88;
}
{
sub.f16x2 r134, r125, r131;
}
{
add.f16x2 r137, %69, %102;
}
{
mul.f16x2 r140, r137, r87;
}
{
add.f16x2 r143, %75, r140;
}
{
sub.f16x2 r146, %105, %90;
}
{
mul.f16x2 r149, r146, r88;
}
{
sub.f16x2 r152, r143, r149;
}
{
add.f16x2 r155, %69, %102;
}
{
mul.f16x2 r158, r155, r87;
}
{
add.f16x2 r161, %75, r158;
}
{
sub.f16x2 r164, %105, %90;
}
{
mul.f16x2 r167, r164, r88;
}
{
add.f16x2 r170, r161, r167;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r173, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r174, {low, high};
}
{
add.f16x2 r175, %103, %88;
}
{
add.f16x2 r178, %59, r175;
}
{
add.f16x2 r181, %64, %101;
}
{
add.f16x2 r184, %74, r181;
}
{
add.f16x2 r187, %103, %88;
}
{
mul.f16x2 r190, r187, r173;
}
{
add.f16x2 r193, %59, r190;
}
{
sub.f16x2 r196, %64, %101;
}
{
mul.f16x2 r199, r196, r174;
}
{
add.f16x2 r202, r193, r199;
}
{
add.f16x2 r205, %103, %88;
}
{
mul.f16x2 r208, r205, r173;
}
{
add.f16x2 r211, %59, r208;
}
{
sub.f16x2 r214, %64, %101;
}
{
mul.f16x2 r217, r214, r174;
}
{
sub.f16x2 r220, r211, r217;
}
{
add.f16x2 r223, %64, %101;
}
{
mul.f16x2 r226, r223, r173;
}
{
add.f16x2 r229, %74, r226;
}
{
sub.f16x2 r232, %103, %88;
}
{
mul.f16x2 r235, r232, r174;
}
{
sub.f16x2 r238, r229, r235;
}
{
add.f16x2 r241, %64, %101;
}
{
mul.f16x2 r244, r241, r173;
}
{
add.f16x2 r247, %74, r244;
}
{
sub.f16x2 r250, %103, %88;
}
{
mul.f16x2 r253, r250, r174;
}
{
add.f16x2 r256, r247, r253;
}
mov.f32 f178, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r259, {low, high};
}
mov.f32 f180, 0f3F248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
mov.b32 r260, {low, high};
}
mov.f32 f190, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r261, {low, high};
}
mov.f32 f192, 0f3F7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f192;
cvt.rn.f16.f32 high, f192;
mov.b32 r262, {low, high};
}
mov.f32 f214, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r265, {low, high};
}
mov.f32 f216, 0f3EAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r266, {low, high};
}
{
mul.f16x2 r275, r116, r259;
}
{
mul.f16x2 r278, r152, r260;
}
{
sub.f16x2 r281, r275, r278;
}
{
mul.f16x2 r284, r116, r260;
}
{
fma.rn.f16x2 r287, r152, r259, r284;
}
{
mul.f16x2 r291, r202, r261;
}
{
mul.f16x2 r294, r238, r262;
}
{
sub.f16x2 r297, r291, r294;
}
{
mul.f16x2 r300, r202, r262;
}
{
fma.rn.f16x2 r303, r238, r261, r300;
}
{
mul.f16x2 r307, r134, r261;
}
{
mul.f16x2 r310, r170, r262;
}
{
sub.f16x2 r313, r307, r310;
}
{
mul.f16x2 r316, r134, r262;
}
{
fma.rn.f16x2 r319, r170, r261, r316;
}
{
mul.f16x2 r323, r220, r265;
}
{
mul.f16x2 r326, r256, r266;
}
{
sub.f16x2 r329, r323, r326;
}
{
mul.f16x2 r332, r220, r266;
}
{
fma.rn.f16x2 r335, r256, r265, r332;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r339, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r340, {low, high};
}
{
add.f16x2 r341, r92, r178;
}
{
add.f16x2 r344, r6, r341;
}
{
add.f16x2 r347, r98, r184;
}
{
add.f16x2 r350, r12, r347;
}
{
add.f16x2 r353, r92, r178;
}
{
mul.f16x2 r356, r353, r339;
}
{
add.f16x2 r359, r6, r356;
}
{
sub.f16x2 r362, r98, r184;
}
{
mul.f16x2 r365, r362, r340;
}
{
add.f16x2 r368, r359, r365;
}
{
add.f16x2 r371, r92, r178;
}
{
mul.f16x2 r374, r371, r339;
}
{
add.f16x2 r377, r6, r374;
}
{
sub.f16x2 r380, r98, r184;
}
{
mul.f16x2 r383, r380, r340;
}
{
sub.f16x2 r386, r377, r383;
}
{
add.f16x2 r389, r98, r184;
}
{
mul.f16x2 r392, r389, r339;
}
{
add.f16x2 r395, r12, r392;
}
{
sub.f16x2 r398, r92, r178;
}
{
mul.f16x2 r401, r398, r340;
}
{
sub.f16x2 r404, r395, r401;
}
{
add.f16x2 r407, r98, r184;
}
{
mul.f16x2 r410, r407, r339;
}
{
add.f16x2 r413, r12, r410;
}
{
sub.f16x2 r416, r92, r178;
}
{
mul.f16x2 r419, r416, r340;
}
{
add.f16x2 r422, r413, r419;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r425, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r426, {low, high};
}
{
add.f16x2 r427, r281, r297;
}
{
add.f16x2 r430, r30, r427;
}
{
add.f16x2 r433, r287, r303;
}
{
add.f16x2 r436, r66, r433;
}
{
add.f16x2 r439, r281, r297;
}
{
mul.f16x2 r442, r439, r425;
}
{
add.f16x2 r445, r30, r442;
}
{
sub.f16x2 r448, r287, r303;
}
{
mul.f16x2 r451, r448, r426;
}
{
add.f16x2 r454, r445, r451;
}
{
add.f16x2 r457, r281, r297;
}
{
mul.f16x2 r460, r457, r425;
}
{
add.f16x2 r463, r30, r460;
}
{
sub.f16x2 r466, r287, r303;
}
{
mul.f16x2 r469, r466, r426;
}
{
sub.f16x2 r472, r463, r469;
}
{
add.f16x2 r475, r287, r303;
}
{
mul.f16x2 r478, r475, r425;
}
{
add.f16x2 r481, r66, r478;
}
{
sub.f16x2 r484, r281, r297;
}
{
mul.f16x2 r487, r484, r426;
}
{
sub.f16x2 r490, r481, r487;
}
{
add.f16x2 r493, r287, r303;
}
{
mul.f16x2 r496, r493, r425;
}
{
add.f16x2 r499, r66, r496;
}
{
sub.f16x2 r502, r281, r297;
}
{
mul.f16x2 r505, r502, r426;
}
{
add.f16x2 r508, r499, r505;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r512, {low, high};
}
{
add.f16x2 r513, r313, r329;
}
{
add.f16x2 r516, r48, r513;
}
{
add.f16x2 r519, r319, r335;
}
{
add.f16x2 r522, r84, r519;
}
{
add.f16x2 r525, r313, r329;
}
{
mul.f16x2 r528, r525, r511;
}
{
add.f16x2 r531, r48, r528;
}
{
sub.f16x2 r534, r319, r335;
}
{
mul.f16x2 r537, r534, r512;
}
{
add.f16x2 r540, r531, r537;
}
{
add.f16x2 r543, r313, r329;
}
{
mul.f16x2 r546, r543, r511;
}
{
add.f16x2 r549, r48, r546;
}
{
sub.f16x2 r552, r319, r335;
}
{
mul.f16x2 r555, r552, r512;
}
{
sub.f16x2 r558, r549, r555;
}
{
add.f16x2 r561, r319, r335;
}
{
mul.f16x2 r564, r561, r511;
}
{
add.f16x2 r567, r84, r564;
}
{
sub.f16x2 r570, r313, r329;
}
{
mul.f16x2 r573, r570, r512;
}
{
sub.f16x2 r576, r567, r573;
}
{
add.f16x2 r579, r319, r335;
}
{
mul.f16x2 r582, r579, r511;
}
{
add.f16x2 r585, r84, r582;
}
{
sub.f16x2 r588, r313, r329;
}
{
mul.f16x2 r591, r588, r512;
}
{
add.f16x2 r594, r585, r591;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r597, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r598, {low, high};
}
{
add.f16x2 r599, %85, %68;
}
{
add.f16x2 r602, %89, r599;
}
{
add.f16x2 r605, %97, %83;
}
{
add.f16x2 r608, %100, r605;
}
{
add.f16x2 r611, %85, %68;
}
{
mul.f16x2 r614, r611, r597;
}
{
add.f16x2 r617, %89, r614;
}
{
sub.f16x2 r620, %97, %83;
}
{
mul.f16x2 r623, r620, r598;
}
{
add.f16x2 r626, r617, r623;
}
{
add.f16x2 r629, %85, %68;
}
{
mul.f16x2 r632, r629, r597;
}
{
add.f16x2 r635, %89, r632;
}
{
sub.f16x2 r638, %97, %83;
}
{
mul.f16x2 r641, r638, r598;
}
{
sub.f16x2 r644, r635, r641;
}
{
add.f16x2 r647, %97, %83;
}
{
mul.f16x2 r650, r647, r597;
}
{
add.f16x2 r653, %100, r650;
}
{
sub.f16x2 r656, %85, %68;
}
{
mul.f16x2 r659, r656, r598;
}
{
sub.f16x2 r662, r653, r659;
}
{
add.f16x2 r665, %97, %83;
}
{
mul.f16x2 r668, r665, r597;
}
{
add.f16x2 r671, %100, r668;
}
{
sub.f16x2 r674, %85, %68;
}
{
mul.f16x2 r677, r674, r598;
}
{
add.f16x2 r680, r671, r677;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r683, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r684, {low, high};
}
{
add.f16x2 r685, %84, %67;
}
{
add.f16x2 r688, %87, r685;
}
{
add.f16x2 r691, %96, %81;
}
{
add.f16x2 r694, %99, r691;
}
{
add.f16x2 r697, %84, %67;
}
{
mul.f16x2 r700, r697, r683;
}
{
add.f16x2 r703, %87, r700;
}
{
sub.f16x2 r706, %96, %81;
}
{
mul.f16x2 r709, r706, r684;
}
{
add.f16x2 r712, r703, r709;
}
{
add.f16x2 r715, %84, %67;
}
{
mul.f16x2 r718, r715, r683;
}
{
add.f16x2 r721, %87, r718;
}
{
sub.f16x2 r724, %96, %81;
}
{
mul.f16x2 r727, r724, r684;
}
{
sub.f16x2 r730, r721, r727;
}
{
add.f16x2 r733, %96, %81;
}
{
mul.f16x2 r736, r733, r683;
}
{
add.f16x2 r739, %99, r736;
}
{
sub.f16x2 r742, %84, %67;
}
{
mul.f16x2 r745, r742, r684;
}
{
sub.f16x2 r748, r739, r745;
}
{
add.f16x2 r751, %96, %81;
}
{
mul.f16x2 r754, r751, r683;
}
{
add.f16x2 r757, %99, r754;
}
{
sub.f16x2 r760, %84, %67;
}
{
mul.f16x2 r763, r760, r684;
}
{
add.f16x2 r766, r757, r763;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r769, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r770, {low, high};
}
{
add.f16x2 r771, %82, %66;
}
{
add.f16x2 r774, %86, r771;
}
{
add.f16x2 r777, %94, %80;
}
{
add.f16x2 r780, %98, r777;
}
{
add.f16x2 r783, %82, %66;
}
{
mul.f16x2 r786, r783, r769;
}
{
add.f16x2 r789, %86, r786;
}
{
sub.f16x2 r792, %94, %80;
}
{
mul.f16x2 r795, r792, r770;
}
{
add.f16x2 r798, r789, r795;
}
{
add.f16x2 r801, %82, %66;
}
{
mul.f16x2 r804, r801, r769;
}
{
add.f16x2 r807, %86, r804;
}
{
sub.f16x2 r810, %94, %80;
}
{
mul.f16x2 r813, r810, r770;
}
{
sub.f16x2 r816, r807, r813;
}
{
add.f16x2 r819, %94, %80;
}
{
mul.f16x2 r822, r819, r769;
}
{
add.f16x2 r825, %98, r822;
}
{
sub.f16x2 r828, %82, %66;
}
{
mul.f16x2 r831, r828, r770;
}
{
sub.f16x2 r834, r825, r831;
}
{
add.f16x2 r837, %94, %80;
}
{
mul.f16x2 r840, r837, r769;
}
{
add.f16x2 r843, %98, r840;
}
{
sub.f16x2 r846, %82, %66;
}
{
mul.f16x2 r849, r846, r770;
}
{
add.f16x2 r852, r843, r849;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r855, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
mov.b32 r856, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r857, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f192;
cvt.rn.f16.f32 high, f192;
mov.b32 r858, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r861, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r862, {low, high};
}
{
mul.f16x2 r871, r712, r855;
}
{
mul.f16x2 r874, r748, r856;
}
{
sub.f16x2 r877, r871, r874;
}
{
mul.f16x2 r880, r712, r856;
}
{
fma.rn.f16x2 r883, r748, r855, r880;
}
{
mul.f16x2 r887, r798, r857;
}
{
mul.f16x2 r890, r834, r858;
}
{
sub.f16x2 r893, r887, r890;
}
{
mul.f16x2 r896, r798, r858;
}
{
fma.rn.f16x2 r899, r834, r857, r896;
}
{
mul.f16x2 r903, r730, r857;
}
{
mul.f16x2 r906, r766, r858;
}
{
sub.f16x2 r909, r903, r906;
}
{
mul.f16x2 r912, r730, r858;
}
{
fma.rn.f16x2 r915, r766, r857, r912;
}
{
mul.f16x2 r919, r816, r861;
}
{
mul.f16x2 r922, r852, r862;
}
{
sub.f16x2 r925, r919, r922;
}
{
mul.f16x2 r928, r816, r862;
}
{
fma.rn.f16x2 r931, r852, r861, r928;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r935, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r936, {low, high};
}
{
add.f16x2 r937, r688, r774;
}
{
add.f16x2 r940, r602, r937;
}
{
add.f16x2 r943, r694, r780;
}
{
add.f16x2 r946, r608, r943;
}
{
add.f16x2 r949, r688, r774;
}
{
mul.f16x2 r952, r949, r935;
}
{
add.f16x2 r955, r602, r952;
}
{
sub.f16x2 r958, r694, r780;
}
{
mul.f16x2 r961, r958, r936;
}
{
add.f16x2 r964, r955, r961;
}
{
add.f16x2 r967, r688, r774;
}
{
mul.f16x2 r970, r967, r935;
}
{
add.f16x2 r973, r602, r970;
}
{
sub.f16x2 r976, r694, r780;
}
{
mul.f16x2 r979, r976, r936;
}
{
sub.f16x2 r982, r973, r979;
}
{
add.f16x2 r985, r694, r780;
}
{
mul.f16x2 r988, r985, r935;
}
{
add.f16x2 r991, r608, r988;
}
{
sub.f16x2 r994, r688, r774;
}
{
mul.f16x2 r997, r994, r936;
}
{
sub.f16x2 r1000, r991, r997;
}
{
add.f16x2 r1003, r694, r780;
}
{
mul.f16x2 r1006, r1003, r935;
}
{
add.f16x2 r1009, r608, r1006;
}
{
sub.f16x2 r1012, r688, r774;
}
{
mul.f16x2 r1015, r1012, r936;
}
{
add.f16x2 r1018, r1009, r1015;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1022, {low, high};
}
{
add.f16x2 r1023, r877, r893;
}
{
add.f16x2 r1026, r626, r1023;
}
{
add.f16x2 r1029, r883, r899;
}
{
add.f16x2 r1032, r662, r1029;
}
{
add.f16x2 r1035, r877, r893;
}
{
mul.f16x2 r1038, r1035, r1021;
}
{
add.f16x2 r1041, r626, r1038;
}
{
sub.f16x2 r1044, r883, r899;
}
{
mul.f16x2 r1047, r1044, r1022;
}
{
add.f16x2 r1050, r1041, r1047;
}
{
add.f16x2 r1053, r877, r893;
}
{
mul.f16x2 r1056, r1053, r1021;
}
{
add.f16x2 r1059, r626, r1056;
}
{
sub.f16x2 r1062, r883, r899;
}
{
mul.f16x2 r1065, r1062, r1022;
}
{
sub.f16x2 r1068, r1059, r1065;
}
{
add.f16x2 r1071, r883, r899;
}
{
mul.f16x2 r1074, r1071, r1021;
}
{
add.f16x2 r1077, r662, r1074;
}
{
sub.f16x2 r1080, r877, r893;
}
{
mul.f16x2 r1083, r1080, r1022;
}
{
sub.f16x2 r1086, r1077, r1083;
}
{
add.f16x2 r1089, r883, r899;
}
{
mul.f16x2 r1092, r1089, r1021;
}
{
add.f16x2 r1095, r662, r1092;
}
{
sub.f16x2 r1098, r877, r893;
}
{
mul.f16x2 r1101, r1098, r1022;
}
{
add.f16x2 r1104, r1095, r1101;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1107, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1108, {low, high};
}
{
add.f16x2 r1109, r909, r925;
}
{
add.f16x2 r1112, r644, r1109;
}
{
add.f16x2 r1115, r915, r931;
}
{
add.f16x2 r1118, r680, r1115;
}
{
add.f16x2 r1121, r909, r925;
}
{
mul.f16x2 r1124, r1121, r1107;
}
{
add.f16x2 r1127, r644, r1124;
}
{
sub.f16x2 r1130, r915, r931;
}
{
mul.f16x2 r1133, r1130, r1108;
}
{
add.f16x2 r1136, r1127, r1133;
}
{
add.f16x2 r1139, r909, r925;
}
{
mul.f16x2 r1142, r1139, r1107;
}
{
add.f16x2 r1145, r644, r1142;
}
{
sub.f16x2 r1148, r915, r931;
}
{
mul.f16x2 r1151, r1148, r1108;
}
{
sub.f16x2 r1154, r1145, r1151;
}
{
add.f16x2 r1157, r915, r931;
}
{
mul.f16x2 r1160, r1157, r1107;
}
{
add.f16x2 r1163, r680, r1160;
}
{
sub.f16x2 r1166, r909, r925;
}
{
mul.f16x2 r1169, r1166, r1108;
}
{
sub.f16x2 r1172, r1163, r1169;
}
{
add.f16x2 r1175, r915, r931;
}
{
mul.f16x2 r1178, r1175, r1107;
}
{
add.f16x2 r1181, r680, r1178;
}
{
sub.f16x2 r1184, r909, r925;
}
{
mul.f16x2 r1187, r1184, r1108;
}
{
add.f16x2 r1190, r1181, r1187;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1193, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1194, {low, high};
}
{
add.f16x2 r1195, %58, %95;
}
{
add.f16x2 r1198, %65, r1195;
}
{
add.f16x2 r1201, %73, %56;
}
{
add.f16x2 r1204, %79, r1201;
}
{
add.f16x2 r1207, %58, %95;
}
{
mul.f16x2 r1210, r1207, r1193;
}
{
add.f16x2 r1213, %65, r1210;
}
{
sub.f16x2 r1216, %73, %56;
}
{
mul.f16x2 r1219, r1216, r1194;
}
{
add.f16x2 r1222, r1213, r1219;
}
{
add.f16x2 r1225, %58, %95;
}
{
mul.f16x2 r1228, r1225, r1193;
}
{
add.f16x2 r1231, %65, r1228;
}
{
sub.f16x2 r1234, %73, %56;
}
{
mul.f16x2 r1237, r1234, r1194;
}
{
sub.f16x2 r1240, r1231, r1237;
}
{
add.f16x2 r1243, %73, %56;
}
{
mul.f16x2 r1246, r1243, r1193;
}
{
add.f16x2 r1249, %79, r1246;
}
{
sub.f16x2 r1252, %58, %95;
}
{
mul.f16x2 r1255, r1252, r1194;
}
{
sub.f16x2 r1258, r1249, r1255;
}
{
add.f16x2 r1261, %73, %56;
}
{
mul.f16x2 r1264, r1261, r1193;
}
{
add.f16x2 r1267, %79, r1264;
}
{
sub.f16x2 r1270, %58, %95;
}
{
mul.f16x2 r1273, r1270, r1194;
}
{
add.f16x2 r1276, r1267, r1273;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1279, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1280, {low, high};
}
{
add.f16x2 r1281, %57, %93;
}
{
add.f16x2 r1284, %63, r1281;
}
{
add.f16x2 r1287, %72, %108;
}
{
add.f16x2 r1290, %78, r1287;
}
{
add.f16x2 r1293, %57, %93;
}
{
mul.f16x2 r1296, r1293, r1279;
}
{
add.f16x2 r1299, %63, r1296;
}
{
sub.f16x2 r1302, %72, %108;
}
{
mul.f16x2 r1305, r1302, r1280;
}
{
add.f16x2 r1308, r1299, r1305;
}
{
add.f16x2 r1311, %57, %93;
}
{
mul.f16x2 r1314, r1311, r1279;
}
{
add.f16x2 r1317, %63, r1314;
}
{
sub.f16x2 r1320, %72, %108;
}
{
mul.f16x2 r1323, r1320, r1280;
}
{
sub.f16x2 r1326, r1317, r1323;
}
{
add.f16x2 r1329, %72, %108;
}
{
mul.f16x2 r1332, r1329, r1279;
}
{
add.f16x2 r1335, %78, r1332;
}
{
sub.f16x2 r1338, %57, %93;
}
{
mul.f16x2 r1341, r1338, r1280;
}
{
sub.f16x2 r1344, r1335, r1341;
}
{
add.f16x2 r1347, %72, %108;
}
{
mul.f16x2 r1350, r1347, r1279;
}
{
add.f16x2 r1353, %78, r1350;
}
{
sub.f16x2 r1356, %57, %93;
}
{
mul.f16x2 r1359, r1356, r1280;
}
{
add.f16x2 r1362, r1353, r1359;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1365, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1366, {low, high};
}
{
add.f16x2 r1367, %55, %92;
}
{
add.f16x2 r1370, %62, r1367;
}
{
add.f16x2 r1373, %71, %107;
}
{
add.f16x2 r1376, %77, r1373;
}
{
add.f16x2 r1379, %55, %92;
}
{
mul.f16x2 r1382, r1379, r1365;
}
{
add.f16x2 r1385, %62, r1382;
}
{
sub.f16x2 r1388, %71, %107;
}
{
mul.f16x2 r1391, r1388, r1366;
}
{
add.f16x2 r1394, r1385, r1391;
}
{
add.f16x2 r1397, %55, %92;
}
{
mul.f16x2 r1400, r1397, r1365;
}
{
add.f16x2 r1403, %62, r1400;
}
{
sub.f16x2 r1406, %71, %107;
}
{
mul.f16x2 r1409, r1406, r1366;
}
{
sub.f16x2 r1412, r1403, r1409;
}
{
add.f16x2 r1415, %71, %107;
}
{
mul.f16x2 r1418, r1415, r1365;
}
{
add.f16x2 r1421, %77, r1418;
}
{
sub.f16x2 r1424, %55, %92;
}
{
mul.f16x2 r1427, r1424, r1366;
}
{
sub.f16x2 r1430, r1421, r1427;
}
{
add.f16x2 r1433, %71, %107;
}
{
mul.f16x2 r1436, r1433, r1365;
}
{
add.f16x2 r1439, %77, r1436;
}
{
sub.f16x2 r1442, %55, %92;
}
{
mul.f16x2 r1445, r1442, r1366;
}
{
add.f16x2 r1448, r1439, r1445;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r1451, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
mov.b32 r1452, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r1453, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f192;
cvt.rn.f16.f32 high, f192;
mov.b32 r1454, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1457, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1458, {low, high};
}
{
mul.f16x2 r1467, r1308, r1451;
}
{
mul.f16x2 r1470, r1344, r1452;
}
{
sub.f16x2 r1473, r1467, r1470;
}
{
mul.f16x2 r1476, r1308, r1452;
}
{
fma.rn.f16x2 r1479, r1344, r1451, r1476;
}
{
mul.f16x2 r1483, r1394, r1453;
}
{
mul.f16x2 r1486, r1430, r1454;
}
{
sub.f16x2 r1489, r1483, r1486;
}
{
mul.f16x2 r1492, r1394, r1454;
}
{
fma.rn.f16x2 r1495, r1430, r1453, r1492;
}
{
mul.f16x2 r1499, r1326, r1453;
}
{
mul.f16x2 r1502, r1362, r1454;
}
{
sub.f16x2 r1505, r1499, r1502;
}
{
mul.f16x2 r1508, r1326, r1454;
}
{
fma.rn.f16x2 r1511, r1362, r1453, r1508;
}
{
mul.f16x2 r1515, r1412, r1457;
}
{
mul.f16x2 r1518, r1448, r1458;
}
{
sub.f16x2 r1521, r1515, r1518;
}
{
mul.f16x2 r1524, r1412, r1458;
}
{
fma.rn.f16x2 r1527, r1448, r1457, r1524;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1531, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1532, {low, high};
}
{
add.f16x2 r1533, r1284, r1370;
}
{
add.f16x2 r1536, r1198, r1533;
}
{
add.f16x2 r1539, r1290, r1376;
}
{
add.f16x2 r1542, r1204, r1539;
}
{
add.f16x2 r1545, r1284, r1370;
}
{
mul.f16x2 r1548, r1545, r1531;
}
{
add.f16x2 r1551, r1198, r1548;
}
{
sub.f16x2 r1554, r1290, r1376;
}
{
mul.f16x2 r1557, r1554, r1532;
}
{
add.f16x2 r1560, r1551, r1557;
}
{
add.f16x2 r1563, r1284, r1370;
}
{
mul.f16x2 r1566, r1563, r1531;
}
{
add.f16x2 r1569, r1198, r1566;
}
{
sub.f16x2 r1572, r1290, r1376;
}
{
mul.f16x2 r1575, r1572, r1532;
}
{
sub.f16x2 r1578, r1569, r1575;
}
{
add.f16x2 r1581, r1290, r1376;
}
{
mul.f16x2 r1584, r1581, r1531;
}
{
add.f16x2 r1587, r1204, r1584;
}
{
sub.f16x2 r1590, r1284, r1370;
}
{
mul.f16x2 r1593, r1590, r1532;
}
{
sub.f16x2 r1596, r1587, r1593;
}
{
add.f16x2 r1599, r1290, r1376;
}
{
mul.f16x2 r1602, r1599, r1531;
}
{
add.f16x2 r1605, r1204, r1602;
}
{
sub.f16x2 r1608, r1284, r1370;
}
{
mul.f16x2 r1611, r1608, r1532;
}
{
add.f16x2 r1614, r1605, r1611;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1617, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1618, {low, high};
}
{
add.f16x2 r1619, r1473, r1489;
}
{
add.f16x2 r1622, r1222, r1619;
}
{
add.f16x2 r1625, r1479, r1495;
}
{
add.f16x2 r1628, r1258, r1625;
}
{
add.f16x2 r1631, r1473, r1489;
}
{
mul.f16x2 r1634, r1631, r1617;
}
{
add.f16x2 r1637, r1222, r1634;
}
{
sub.f16x2 r1640, r1479, r1495;
}
{
mul.f16x2 r1643, r1640, r1618;
}
{
add.f16x2 r1646, r1637, r1643;
}
{
add.f16x2 r1649, r1473, r1489;
}
{
mul.f16x2 r1652, r1649, r1617;
}
{
add.f16x2 r1655, r1222, r1652;
}
{
sub.f16x2 r1658, r1479, r1495;
}
{
mul.f16x2 r1661, r1658, r1618;
}
{
sub.f16x2 r1664, r1655, r1661;
}
{
add.f16x2 r1667, r1479, r1495;
}
{
mul.f16x2 r1670, r1667, r1617;
}
{
add.f16x2 r1673, r1258, r1670;
}
{
sub.f16x2 r1676, r1473, r1489;
}
{
mul.f16x2 r1679, r1676, r1618;
}
{
sub.f16x2 r1682, r1673, r1679;
}
{
add.f16x2 r1685, r1479, r1495;
}
{
mul.f16x2 r1688, r1685, r1617;
}
{
add.f16x2 r1691, r1258, r1688;
}
{
sub.f16x2 r1694, r1473, r1489;
}
{
mul.f16x2 r1697, r1694, r1618;
}
{
add.f16x2 r1700, r1691, r1697;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1703, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1704, {low, high};
}
{
add.f16x2 r1705, r1505, r1521;
}
{
add.f16x2 r1708, r1240, r1705;
}
{
add.f16x2 r1711, r1511, r1527;
}
{
add.f16x2 r1714, r1276, r1711;
}
{
add.f16x2 r1717, r1505, r1521;
}
{
mul.f16x2 r1720, r1717, r1703;
}
{
add.f16x2 r1723, r1240, r1720;
}
{
sub.f16x2 r1726, r1511, r1527;
}
{
mul.f16x2 r1729, r1726, r1704;
}
{
add.f16x2 r1732, r1723, r1729;
}
{
add.f16x2 r1735, r1505, r1521;
}
{
mul.f16x2 r1738, r1735, r1703;
}
{
add.f16x2 r1741, r1240, r1738;
}
{
sub.f16x2 r1744, r1511, r1527;
}
{
mul.f16x2 r1747, r1744, r1704;
}
{
sub.f16x2 r1750, r1741, r1747;
}
{
add.f16x2 r1753, r1511, r1527;
}
{
mul.f16x2 r1756, r1753, r1703;
}
{
add.f16x2 r1759, r1276, r1756;
}
{
sub.f16x2 r1762, r1505, r1521;
}
{
mul.f16x2 r1765, r1762, r1704;
}
{
sub.f16x2 r1768, r1759, r1765;
}
{
add.f16x2 r1771, r1511, r1527;
}
{
mul.f16x2 r1774, r1771, r1703;
}
{
add.f16x2 r1777, r1276, r1774;
}
{
sub.f16x2 r1780, r1505, r1521;
}
{
mul.f16x2 r1783, r1780, r1704;
}
{
add.f16x2 r1786, r1777, r1783;
}
mov.f32 f170, 0f3F791978;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f170;
cvt.rn.f16.f32 high, f170;
mov.b32 r1789, {low, high};
}
mov.f32 f172, 0f3E6C2691;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f172;
cvt.rn.f16.f32 high, f172;
mov.b32 r1790, {low, high};
}
mov.f32 f174, 0f3F64C51C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f174;
cvt.rn.f16.f32 high, f174;
mov.b32 r1791, {low, high};
}
mov.f32 f176, 0f3EE5C902;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f176;
cvt.rn.f16.f32 high, f176;
mov.b32 r1792, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r1793, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
mov.b32 r1794, {low, high};
}
mov.f32 f182, 0f3F18DF63;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f182;
cvt.rn.f16.f32 high, f182;
mov.b32 r1795, {low, high};
}
mov.f32 f184, 0f3F4D57F2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f184;
cvt.rn.f16.f32 high, f184;
mov.b32 r1796, {low, high};
}
mov.f32 f186, 0f3ECACAF8;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f186;
cvt.rn.f16.f32 high, f186;
mov.b32 r1797, {low, high};
}
mov.f32 f188, 0f3F6B1036;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r1798, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r1799, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f192;
cvt.rn.f16.f32 high, f192;
mov.b32 r1800, {low, high};
}
mov.f32 f194, 0fBD6E2946;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1801, {low, high};
}
mov.f32 f196, 0f3F7F9120;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r1802, {low, high};
}
mov.f32 f198, 0fBE92D7E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1803, {low, high};
}
mov.f32 f200, 0f3F753ECD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1804, {low, high};
}
mov.f32 f206, 0fBF2FAD88;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f206;
cvt.rn.f16.f32 high, f206;
mov.b32 r1807, {low, high};
}
mov.f32 f208, 0f3F3A3529;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f208;
cvt.rn.f16.f32 high, f208;
mov.b32 r1808, {low, high};
}
mov.f32 f230, 0fBF55E287;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1811, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1812, {low, high};
}
mov.f32 f222, 0fBF7E44DE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f222;
cvt.rn.f16.f32 high, f222;
mov.b32 r1815, {low, high};
}
mov.f32 f224, 0fBDEDC21F;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f224;
cvt.rn.f16.f32 high, f224;
mov.b32 r1816, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r1819, {low, high};
}
mov.f32 f232, 0fBF0CAC9F;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f232;
cvt.rn.f16.f32 high, f232;
mov.b32 r1820, {low, high};
}
{
mul.f16x2 r1841, r1026, r1789;
}
{
mul.f16x2 r1844, r1032, r1790;
}
{
sub.f16x2 r1847, r1841, r1844;
}
{
mul.f16x2 r1850, r1026, r1790;
}
{
fma.rn.f16x2 r1853, r1032, r1789, r1850;
}
{
mul.f16x2 r1857, r1622, r1791;
}
{
mul.f16x2 r1860, r1628, r1792;
}
{
sub.f16x2 r1863, r1857, r1860;
}
{
mul.f16x2 r1866, r1622, r1792;
}
{
fma.rn.f16x2 r1869, r1628, r1791, r1866;
}
{
mul.f16x2 r1873, r1112, r1791;
}
{
mul.f16x2 r1876, r1118, r1792;
}
{
sub.f16x2 r1879, r1873, r1876;
}
{
mul.f16x2 r1882, r1112, r1792;
}
{
fma.rn.f16x2 r1885, r1118, r1791, r1882;
}
{
mul.f16x2 r1889, r1708, r1795;
}
{
mul.f16x2 r1892, r1714, r1796;
}
{
sub.f16x2 r1895, r1889, r1892;
}
{
mul.f16x2 r1898, r1708, r1796;
}
{
fma.rn.f16x2 r1901, r1714, r1795, r1898;
}
{
mul.f16x2 r1905, r964, r1793;
}
{
mul.f16x2 r1908, r1000, r1794;
}
{
sub.f16x2 r1911, r1905, r1908;
}
{
mul.f16x2 r1914, r964, r1794;
}
{
fma.rn.f16x2 r1917, r1000, r1793, r1914;
}
{
mul.f16x2 r1921, r1560, r1799;
}
{
mul.f16x2 r1924, r1596, r1800;
}
{
sub.f16x2 r1927, r1921, r1924;
}
{
mul.f16x2 r1930, r1560, r1800;
}
{
fma.rn.f16x2 r1933, r1596, r1799, r1930;
}
{
mul.f16x2 r1937, r1050, r1795;
}
{
mul.f16x2 r1940, r1086, r1796;
}
{
sub.f16x2 r1943, r1937, r1940;
}
{
mul.f16x2 r1946, r1050, r1796;
}
{
fma.rn.f16x2 r1949, r1086, r1795, r1946;
}
{
mul.f16x2 r1953, r1646, r1803;
}
{
mul.f16x2 r1956, r1682, r1804;
}
{
sub.f16x2 r1959, r1953, r1956;
}
{
mul.f16x2 r1962, r1646, r1804;
}
{
fma.rn.f16x2 r1965, r1682, r1803, r1962;
}
{
mul.f16x2 r1969, r1136, r1797;
}
{
mul.f16x2 r1972, r1172, r1798;
}
{
sub.f16x2 r1975, r1969, r1972;
}
{
mul.f16x2 r1978, r1136, r1798;
}
{
fma.rn.f16x2 r1981, r1172, r1797, r1978;
}
{
mul.f16x2 r1985, r1732, r1807;
}
{
mul.f16x2 r1988, r1768, r1808;
}
{
sub.f16x2 r1991, r1985, r1988;
}
{
mul.f16x2 r1994, r1732, r1808;
}
{
fma.rn.f16x2 r1997, r1768, r1807, r1994;
}
{
mul.f16x2 r2001, r982, r1799;
}
{
mul.f16x2 r2004, r1018, r1800;
}
{
sub.f16x2 r2007, r2001, r2004;
}
{
mul.f16x2 r2010, r982, r1800;
}
{
fma.rn.f16x2 r2013, r1018, r1799, r2010;
}
{
mul.f16x2 r2017, r1578, r1811;
}
{
mul.f16x2 r2020, r1614, r1812;
}
{
sub.f16x2 r2023, r2017, r2020;
}
{
mul.f16x2 r2026, r1578, r1812;
}
{
fma.rn.f16x2 r2029, r1614, r1811, r2026;
}
{
mul.f16x2 r2033, r1068, r1801;
}
{
mul.f16x2 r2036, r1104, r1802;
}
{
sub.f16x2 r2039, r2033, r2036;
}
{
mul.f16x2 r2042, r1068, r1802;
}
{
fma.rn.f16x2 r2045, r1104, r1801, r2042;
}
{
mul.f16x2 r2049, r1664, r1815;
}
{
mul.f16x2 r2052, r1700, r1816;
}
{
sub.f16x2 r2055, r2049, r2052;
}
{
mul.f16x2 r2058, r1664, r1816;
}
{
fma.rn.f16x2 r2061, r1700, r1815, r2058;
}
{
mul.f16x2 r2065, r1154, r1803;
}
{
mul.f16x2 r2068, r1190, r1804;
}
{
sub.f16x2 r2071, r2065, r2068;
}
{
mul.f16x2 r2074, r1154, r1804;
}
{
fma.rn.f16x2 r2077, r1190, r1803, r2074;
}
{
mul.f16x2 r2081, r1750, r1819;
}
{
mul.f16x2 r2084, r1786, r1820;
}
{
sub.f16x2 r2087, r2081, r2084;
}
{
mul.f16x2 r2090, r1750, r1820;
}
{
fma.rn.f16x2 r2093, r1786, r1819, r2090;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2097, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2098, {low, high};
}
{
add.f16x2 r2099, r940, r1536;
}
{
add.f16x2 r2102, r344, r2099;
}
{
add.f16x2 r2105, r946, r1542;
}
{
add.f16x2 r2108, r350, r2105;
}
{
add.f16x2 r2111, r940, r1536;
}
{
mul.f16x2 r2114, r2111, r2097;
}
{
add.f16x2 r2117, r344, r2114;
}
{
sub.f16x2 r2120, r946, r1542;
}
{
mul.f16x2 r2123, r2120, r2098;
}
{
add.f16x2 r2126, r2117, r2123;
}
{
add.f16x2 r2129, r940, r1536;
}
{
mul.f16x2 r2132, r2129, r2097;
}
{
add.f16x2 r2135, r344, r2132;
}
{
sub.f16x2 r2138, r946, r1542;
}
{
mul.f16x2 r2141, r2138, r2098;
}
{
sub.f16x2 r2144, r2135, r2141;
}
{
add.f16x2 r2147, r946, r1542;
}
{
mul.f16x2 r2150, r2147, r2097;
}
{
add.f16x2 r2153, r350, r2150;
}
{
sub.f16x2 r2156, r940, r1536;
}
{
mul.f16x2 r2159, r2156, r2098;
}
{
sub.f16x2 r2162, r2153, r2159;
}
{
add.f16x2 r2165, r946, r1542;
}
{
mul.f16x2 r2168, r2165, r2097;
}
{
add.f16x2 r2171, r350, r2168;
}
{
sub.f16x2 r2174, r940, r1536;
}
{
mul.f16x2 r2177, r2174, r2098;
}
{
add.f16x2 r2180, r2171, r2177;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2183, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2184, {low, high};
}
{
add.f16x2 r2185, r1847, r1863;
}
{
add.f16x2 r2188, r430, r2185;
}
{
add.f16x2 r2191, r1853, r1869;
}
{
add.f16x2 r2194, r436, r2191;
}
{
add.f16x2 r2197, r1847, r1863;
}
{
mul.f16x2 r2200, r2197, r2183;
}
{
add.f16x2 r2203, r430, r2200;
}
{
sub.f16x2 r2206, r1853, r1869;
}
{
mul.f16x2 r2209, r2206, r2184;
}
{
add.f16x2 r2212, r2203, r2209;
}
{
add.f16x2 r2215, r1847, r1863;
}
{
mul.f16x2 r2218, r2215, r2183;
}
{
add.f16x2 r2221, r430, r2218;
}
{
sub.f16x2 r2224, r1853, r1869;
}
{
mul.f16x2 r2227, r2224, r2184;
}
{
sub.f16x2 r2230, r2221, r2227;
}
{
add.f16x2 r2233, r1853, r1869;
}
{
mul.f16x2 r2236, r2233, r2183;
}
{
add.f16x2 r2239, r436, r2236;
}
{
sub.f16x2 r2242, r1847, r1863;
}
{
mul.f16x2 r2245, r2242, r2184;
}
{
sub.f16x2 r2248, r2239, r2245;
}
{
add.f16x2 r2251, r1853, r1869;
}
{
mul.f16x2 r2254, r2251, r2183;
}
{
add.f16x2 r2257, r436, r2254;
}
{
sub.f16x2 r2260, r1847, r1863;
}
{
mul.f16x2 r2263, r2260, r2184;
}
{
add.f16x2 r2266, r2257, r2263;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2269, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2270, {low, high};
}
{
add.f16x2 r2271, r1879, r1895;
}
{
add.f16x2 r2274, r516, r2271;
}
{
add.f16x2 r2277, r1885, r1901;
}
{
add.f16x2 r2280, r522, r2277;
}
{
add.f16x2 r2283, r1879, r1895;
}
{
mul.f16x2 r2286, r2283, r2269;
}
{
add.f16x2 r2289, r516, r2286;
}
{
sub.f16x2 r2292, r1885, r1901;
}
{
mul.f16x2 r2295, r2292, r2270;
}
{
add.f16x2 r2298, r2289, r2295;
}
{
add.f16x2 r2301, r1879, r1895;
}
{
mul.f16x2 r2304, r2301, r2269;
}
{
add.f16x2 r2307, r516, r2304;
}
{
sub.f16x2 r2310, r1885, r1901;
}
{
mul.f16x2 r2313, r2310, r2270;
}
{
sub.f16x2 r2316, r2307, r2313;
}
{
add.f16x2 r2319, r1885, r1901;
}
{
mul.f16x2 r2322, r2319, r2269;
}
{
add.f16x2 r2325, r522, r2322;
}
{
sub.f16x2 r2328, r1879, r1895;
}
{
mul.f16x2 r2331, r2328, r2270;
}
{
sub.f16x2 r2334, r2325, r2331;
}
{
add.f16x2 r2337, r1885, r1901;
}
{
mul.f16x2 r2340, r2337, r2269;
}
{
add.f16x2 r2343, r522, r2340;
}
{
sub.f16x2 r2346, r1879, r1895;
}
{
mul.f16x2 r2349, r2346, r2270;
}
{
add.f16x2 r2352, r2343, r2349;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2355, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2356, {low, high};
}
{
add.f16x2 r2357, r1911, r1927;
}
{
add.f16x2 r2360, r368, r2357;
}
{
add.f16x2 r2363, r1917, r1933;
}
{
add.f16x2 r2366, r404, r2363;
}
{
add.f16x2 r2369, r1911, r1927;
}
{
mul.f16x2 r2372, r2369, r2355;
}
{
add.f16x2 r2375, r368, r2372;
}
{
sub.f16x2 r2378, r1917, r1933;
}
{
mul.f16x2 r2381, r2378, r2356;
}
{
add.f16x2 r2384, r2375, r2381;
}
{
add.f16x2 r2387, r1911, r1927;
}
{
mul.f16x2 r2390, r2387, r2355;
}
{
add.f16x2 r2393, r368, r2390;
}
{
sub.f16x2 r2396, r1917, r1933;
}
{
mul.f16x2 r2399, r2396, r2356;
}
{
sub.f16x2 r2402, r2393, r2399;
}
{
add.f16x2 r2405, r1917, r1933;
}
{
mul.f16x2 r2408, r2405, r2355;
}
{
add.f16x2 r2411, r404, r2408;
}
{
sub.f16x2 r2414, r1911, r1927;
}
{
mul.f16x2 r2417, r2414, r2356;
}
{
sub.f16x2 r2420, r2411, r2417;
}
{
add.f16x2 r2423, r1917, r1933;
}
{
mul.f16x2 r2426, r2423, r2355;
}
{
add.f16x2 r2429, r404, r2426;
}
{
sub.f16x2 r2432, r1911, r1927;
}
{
mul.f16x2 r2435, r2432, r2356;
}
{
add.f16x2 r2438, r2429, r2435;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2441, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2442, {low, high};
}
{
add.f16x2 r2443, r1943, r1959;
}
{
add.f16x2 r2446, r454, r2443;
}
{
add.f16x2 r2449, r1949, r1965;
}
{
add.f16x2 r2452, r490, r2449;
}
{
add.f16x2 r2455, r1943, r1959;
}
{
mul.f16x2 r2458, r2455, r2441;
}
{
add.f16x2 r2461, r454, r2458;
}
{
sub.f16x2 r2464, r1949, r1965;
}
{
mul.f16x2 r2467, r2464, r2442;
}
{
add.f16x2 r2470, r2461, r2467;
}
{
add.f16x2 r2473, r1943, r1959;
}
{
mul.f16x2 r2476, r2473, r2441;
}
{
add.f16x2 r2479, r454, r2476;
}
{
sub.f16x2 r2482, r1949, r1965;
}
{
mul.f16x2 r2485, r2482, r2442;
}
{
sub.f16x2 r2488, r2479, r2485;
}
{
add.f16x2 r2491, r1949, r1965;
}
{
mul.f16x2 r2494, r2491, r2441;
}
{
add.f16x2 r2497, r490, r2494;
}
{
sub.f16x2 r2500, r1943, r1959;
}
{
mul.f16x2 r2503, r2500, r2442;
}
{
sub.f16x2 r2506, r2497, r2503;
}
{
add.f16x2 r2509, r1949, r1965;
}
{
mul.f16x2 r2512, r2509, r2441;
}
{
add.f16x2 r2515, r490, r2512;
}
{
sub.f16x2 r2518, r1943, r1959;
}
{
mul.f16x2 r2521, r2518, r2442;
}
{
add.f16x2 r2524, r2515, r2521;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2527, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2528, {low, high};
}
{
add.f16x2 r2529, r1975, r1991;
}
{
add.f16x2 r2532, r540, r2529;
}
{
add.f16x2 r2535, r1981, r1997;
}
{
add.f16x2 r2538, r576, r2535;
}
{
add.f16x2 r2541, r1975, r1991;
}
{
mul.f16x2 r2544, r2541, r2527;
}
{
add.f16x2 r2547, r540, r2544;
}
{
sub.f16x2 r2550, r1981, r1997;
}
{
mul.f16x2 r2553, r2550, r2528;
}
{
add.f16x2 r2556, r2547, r2553;
}
{
add.f16x2 r2559, r1975, r1991;
}
{
mul.f16x2 r2562, r2559, r2527;
}
{
add.f16x2 r2565, r540, r2562;
}
{
sub.f16x2 r2568, r1981, r1997;
}
{
mul.f16x2 r2571, r2568, r2528;
}
{
sub.f16x2 r2574, r2565, r2571;
}
{
add.f16x2 r2577, r1981, r1997;
}
{
mul.f16x2 r2580, r2577, r2527;
}
{
add.f16x2 r2583, r576, r2580;
}
{
sub.f16x2 r2586, r1975, r1991;
}
{
mul.f16x2 r2589, r2586, r2528;
}
{
sub.f16x2 r2592, r2583, r2589;
}
{
add.f16x2 r2595, r1981, r1997;
}
{
mul.f16x2 r2598, r2595, r2527;
}
{
add.f16x2 r2601, r576, r2598;
}
{
sub.f16x2 r2604, r1975, r1991;
}
{
mul.f16x2 r2607, r2604, r2528;
}
{
add.f16x2 r2610, r2601, r2607;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2613, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2614, {low, high};
}
{
add.f16x2 r2615, r2007, r2023;
}
{
add.f16x2 r2618, r386, r2615;
}
{
add.f16x2 r2621, r2013, r2029;
}
{
add.f16x2 r2624, r422, r2621;
}
{
add.f16x2 r2627, r2007, r2023;
}
{
mul.f16x2 r2630, r2627, r2613;
}
{
add.f16x2 r2633, r386, r2630;
}
{
sub.f16x2 r2636, r2013, r2029;
}
{
mul.f16x2 r2639, r2636, r2614;
}
{
add.f16x2 r2642, r2633, r2639;
}
{
add.f16x2 r2645, r2007, r2023;
}
{
mul.f16x2 r2648, r2645, r2613;
}
{
add.f16x2 r2651, r386, r2648;
}
{
sub.f16x2 r2654, r2013, r2029;
}
{
mul.f16x2 r2657, r2654, r2614;
}
{
sub.f16x2 r2660, r2651, r2657;
}
{
add.f16x2 r2663, r2013, r2029;
}
{
mul.f16x2 r2666, r2663, r2613;
}
{
add.f16x2 r2669, r422, r2666;
}
{
sub.f16x2 r2672, r2007, r2023;
}
{
mul.f16x2 r2675, r2672, r2614;
}
{
sub.f16x2 r2678, r2669, r2675;
}
{
add.f16x2 r2681, r2013, r2029;
}
{
mul.f16x2 r2684, r2681, r2613;
}
{
add.f16x2 r2687, r422, r2684;
}
{
sub.f16x2 r2690, r2007, r2023;
}
{
mul.f16x2 r2693, r2690, r2614;
}
{
add.f16x2 r2696, r2687, r2693;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2699, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2700, {low, high};
}
{
add.f16x2 r2701, r2039, r2055;
}
{
add.f16x2 r2704, r472, r2701;
}
{
add.f16x2 r2707, r2045, r2061;
}
{
add.f16x2 r2710, r508, r2707;
}
{
add.f16x2 r2713, r2039, r2055;
}
{
mul.f16x2 r2716, r2713, r2699;
}
{
add.f16x2 r2719, r472, r2716;
}
{
sub.f16x2 r2722, r2045, r2061;
}
{
mul.f16x2 r2725, r2722, r2700;
}
{
add.f16x2 r2728, r2719, r2725;
}
{
add.f16x2 r2731, r2039, r2055;
}
{
mul.f16x2 r2734, r2731, r2699;
}
{
add.f16x2 r2737, r472, r2734;
}
{
sub.f16x2 r2740, r2045, r2061;
}
{
mul.f16x2 r2743, r2740, r2700;
}
{
sub.f16x2 r2746, r2737, r2743;
}
{
add.f16x2 r2749, r2045, r2061;
}
{
mul.f16x2 r2752, r2749, r2699;
}
{
add.f16x2 r2755, r508, r2752;
}
{
sub.f16x2 r2758, r2039, r2055;
}
{
mul.f16x2 r2761, r2758, r2700;
}
{
sub.f16x2 r2764, r2755, r2761;
}
{
add.f16x2 r2767, r2045, r2061;
}
{
mul.f16x2 r2770, r2767, r2699;
}
{
add.f16x2 r2773, r508, r2770;
}
{
sub.f16x2 r2776, r2039, r2055;
}
{
mul.f16x2 r2779, r2776, r2700;
}
{
add.f16x2 r2782, r2773, r2779;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2785, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2786, {low, high};
}
{
add.f16x2 r2787, r2071, r2087;
}
{
add.f16x2 r2790, r558, r2787;
}
{
add.f16x2 r2793, r2077, r2093;
}
{
add.f16x2 r2796, r594, r2793;
}
{
add.f16x2 r2799, r2071, r2087;
}
{
mul.f16x2 r2802, r2799, r2785;
}
{
add.f16x2 r2805, r558, r2802;
}
{
sub.f16x2 r2808, r2077, r2093;
}
{
mul.f16x2 r2811, r2808, r2786;
}
{
add.f16x2 r2814, r2805, r2811;
}
{
add.f16x2 r2817, r2071, r2087;
}
{
mul.f16x2 r2820, r2817, r2785;
}
{
add.f16x2 r2823, r558, r2820;
}
{
sub.f16x2 r2826, r2077, r2093;
}
{
mul.f16x2 r2829, r2826, r2786;
}
{
sub.f16x2 r2832, r2823, r2829;
}
{
add.f16x2 r2835, r2077, r2093;
}
{
mul.f16x2 r2838, r2835, r2785;
}
{
add.f16x2 r2841, r594, r2838;
}
{
sub.f16x2 r2844, r2071, r2087;
}
{
mul.f16x2 r2847, r2844, r2786;
}
{
sub.f16x2 r2850, r2841, r2847;
}
{
add.f16x2 r2853, r2077, r2093;
}
{
mul.f16x2 r2856, r2853, r2785;
}
{
add.f16x2 r2859, r594, r2856;
}
{
sub.f16x2 r2862, r2071, r2087;
}
{
mul.f16x2 r2865, r2862, r2786;
}
{
add.f16x2 r2868, r2859, r2865;
}
mul.wide.u32 rd2, r4613, -1431655765;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r4614, rd3;
mul.lo.s32 r4615, r4614, 3;
sub.s32 r4616, r4613, r4615;
cvt.rn.f32.u32 f401, r4616;
mul.f32 f402, f401, 0f3D9EDD1F;
cos.approx.f32 f309, f402;
sin.approx.f32 f403, f402;
neg.f32 f310, f403;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f309;
cvt.rn.f16.f32 high, f310;
mov.b32 r2871, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r2874, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r2876, {high, high};
}
{
mul.f16x2 r2878, r2194, r2876;
}
{
fma.rn.f16x2 r2881, r2188, r2874, r2878;
}
{
mul.f16x2 r2885, r2188, r2876;
}
{
neg.f16x2 r2888, r2885;
}
{
fma.rn.f16x2 r2890, r2194, r2874, r2888;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r2894, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r2896, {high, high};
}
mov.f32 f361, 0fBF800000;
mov.f32 f362, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r2898, {low, high};
}
{
mul.f16x2 r2899, r2896, r2898;
}
{
mul.f16x2 r2902, r2871, r2894;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r2905, {high, low};
}
{
fma.rn.f16x2 r2907, r2899, r2905, r2902;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2907;
mov.b32 r2911, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2907;
mov.b32 r2913, {high, high};
}
{
mul.f16x2 r2915, r2280, r2913;
}
{
fma.rn.f16x2 r2918, r2274, r2911, r2915;
}
{
mul.f16x2 r2922, r2274, r2913;
}
{
neg.f16x2 r2925, r2922;
}
{
fma.rn.f16x2 r2927, r2280, r2911, r2925;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r2931, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r2933, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r2935, {low, high};
}
{
mul.f16x2 r2936, r2933, r2935;
}
{
mul.f16x2 r2939, r2907, r2931;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2907;
mov.b32 r2942, {high, low};
}
{
fma.rn.f16x2 r2944, r2936, r2942, r2939;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2944;
mov.b32 r2948, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2944;
mov.b32 r2950, {high, high};
}
{
mul.f16x2 r2952, r2366, r2950;
}
{
fma.rn.f16x2 r2955, r2360, r2948, r2952;
}
{
mul.f16x2 r2959, r2360, r2950;
}
{
neg.f16x2 r2962, r2959;
}
{
fma.rn.f16x2 r2964, r2366, r2948, r2962;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r2968, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r2970, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r2972, {low, high};
}
{
mul.f16x2 r2973, r2970, r2972;
}
{
mul.f16x2 r2976, r2944, r2968;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2944;
mov.b32 r2979, {high, low};
}
{
fma.rn.f16x2 r2981, r2973, r2979, r2976;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2981;
mov.b32 r2985, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2981;
mov.b32 r2987, {high, high};
}
{
mul.f16x2 r2989, r2452, r2987;
}
{
fma.rn.f16x2 r2992, r2446, r2985, r2989;
}
{
mul.f16x2 r2996, r2446, r2987;
}
{
neg.f16x2 r2999, r2996;
}
{
fma.rn.f16x2 r3001, r2452, r2985, r2999;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3005, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3007, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3009, {low, high};
}
{
mul.f16x2 r3010, r3007, r3009;
}
{
mul.f16x2 r3013, r2981, r3005;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2981;
mov.b32 r3016, {high, low};
}
{
fma.rn.f16x2 r3018, r3010, r3016, r3013;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3018;
mov.b32 r3022, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3018;
mov.b32 r3024, {high, high};
}
{
mul.f16x2 r3026, r2538, r3024;
}
{
fma.rn.f16x2 r3029, r2532, r3022, r3026;
}
{
mul.f16x2 r3033, r2532, r3024;
}
{
neg.f16x2 r3036, r3033;
}
{
fma.rn.f16x2 r3038, r2538, r3022, r3036;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3042, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3044, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3046, {low, high};
}
{
mul.f16x2 r3047, r3044, r3046;
}
{
mul.f16x2 r3050, r3018, r3042;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3018;
mov.b32 r3053, {high, low};
}
{
fma.rn.f16x2 r3055, r3047, r3053, r3050;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3055;
mov.b32 r3059, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3055;
mov.b32 r3061, {high, high};
}
{
mul.f16x2 r3063, r2624, r3061;
}
{
fma.rn.f16x2 r3066, r2618, r3059, r3063;
}
{
mul.f16x2 r3070, r2618, r3061;
}
{
neg.f16x2 r3073, r3070;
}
{
fma.rn.f16x2 r3075, r2624, r3059, r3073;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3079, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3081, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3083, {low, high};
}
{
mul.f16x2 r3084, r3081, r3083;
}
{
mul.f16x2 r3087, r3055, r3079;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3055;
mov.b32 r3090, {high, low};
}
{
fma.rn.f16x2 r3092, r3084, r3090, r3087;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3092;
mov.b32 r3096, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3092;
mov.b32 r3098, {high, high};
}
{
mul.f16x2 r3100, r2710, r3098;
}
{
fma.rn.f16x2 r3103, r2704, r3096, r3100;
}
{
mul.f16x2 r3107, r2704, r3098;
}
{
neg.f16x2 r3110, r3107;
}
{
fma.rn.f16x2 r3112, r2710, r3096, r3110;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3116, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3118, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3120, {low, high};
}
{
mul.f16x2 r3121, r3118, r3120;
}
{
mul.f16x2 r3124, r3092, r3116;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3092;
mov.b32 r3127, {high, low};
}
{
fma.rn.f16x2 r3129, r3121, r3127, r3124;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3129;
mov.b32 r3133, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3129;
mov.b32 r3135, {high, high};
}
{
mul.f16x2 r3137, r2796, r3135;
}
{
fma.rn.f16x2 r3140, r2790, r3133, r3137;
}
{
mul.f16x2 r3144, r2790, r3135;
}
{
neg.f16x2 r3147, r3144;
}
{
fma.rn.f16x2 r3149, r2796, r3133, r3147;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3153, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3155, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3157, {low, high};
}
{
mul.f16x2 r3158, r3155, r3157;
}
{
mul.f16x2 r3161, r3129, r3153;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3129;
mov.b32 r3164, {high, low};
}
{
fma.rn.f16x2 r3166, r3158, r3164, r3161;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3166;
mov.b32 r3170, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3166;
mov.b32 r3172, {high, high};
}
{
mul.f16x2 r3174, r2162, r3172;
}
{
fma.rn.f16x2 r3177, r2126, r3170, r3174;
}
{
mul.f16x2 r3181, r2126, r3172;
}
{
neg.f16x2 r3184, r3181;
}
{
fma.rn.f16x2 r3186, r2162, r3170, r3184;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3190, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3192, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3194, {low, high};
}
{
mul.f16x2 r3195, r3192, r3194;
}
{
mul.f16x2 r3198, r3166, r3190;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3166;
mov.b32 r3201, {high, low};
}
{
fma.rn.f16x2 r3203, r3195, r3201, r3198;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3203;
mov.b32 r3207, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3203;
mov.b32 r3209, {high, high};
}
{
mul.f16x2 r3211, r2248, r3209;
}
{
fma.rn.f16x2 r3214, r2212, r3207, r3211;
}
{
mul.f16x2 r3218, r2212, r3209;
}
{
neg.f16x2 r3221, r3218;
}
{
fma.rn.f16x2 r3223, r2248, r3207, r3221;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3227, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3229, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3231, {low, high};
}
{
mul.f16x2 r3232, r3229, r3231;
}
{
mul.f16x2 r3235, r3203, r3227;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3203;
mov.b32 r3238, {high, low};
}
{
fma.rn.f16x2 r3240, r3232, r3238, r3235;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3240;
mov.b32 r3244, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3240;
mov.b32 r3246, {high, high};
}
{
mul.f16x2 r3248, r2334, r3246;
}
{
fma.rn.f16x2 r3251, r2298, r3244, r3248;
}
{
mul.f16x2 r3255, r2298, r3246;
}
{
neg.f16x2 r3258, r3255;
}
{
fma.rn.f16x2 r3260, r2334, r3244, r3258;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3264, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3266, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3268, {low, high};
}
{
mul.f16x2 r3269, r3266, r3268;
}
{
mul.f16x2 r3272, r3240, r3264;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3240;
mov.b32 r3275, {high, low};
}
{
fma.rn.f16x2 r3277, r3269, r3275, r3272;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3277;
mov.b32 r3281, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3277;
mov.b32 r3283, {high, high};
}
{
mul.f16x2 r3285, r2420, r3283;
}
{
fma.rn.f16x2 r3288, r2384, r3281, r3285;
}
{
mul.f16x2 r3292, r2384, r3283;
}
{
neg.f16x2 r3295, r3292;
}
{
fma.rn.f16x2 r3297, r2420, r3281, r3295;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3301, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3303, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3305, {low, high};
}
{
mul.f16x2 r3306, r3303, r3305;
}
{
mul.f16x2 r3309, r3277, r3301;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3277;
mov.b32 r3312, {high, low};
}
{
fma.rn.f16x2 r3314, r3306, r3312, r3309;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3314;
mov.b32 r3318, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3314;
mov.b32 r3320, {high, high};
}
{
mul.f16x2 r3322, r2506, r3320;
}
{
fma.rn.f16x2 r3325, r2470, r3318, r3322;
}
{
mul.f16x2 r3329, r2470, r3320;
}
{
neg.f16x2 r3332, r3329;
}
{
fma.rn.f16x2 r3334, r2506, r3318, r3332;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3338, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3340, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3342, {low, high};
}
{
mul.f16x2 r3343, r3340, r3342;
}
{
mul.f16x2 r3346, r3314, r3338;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3314;
mov.b32 r3349, {high, low};
}
{
fma.rn.f16x2 r3351, r3343, r3349, r3346;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3351;
mov.b32 r3355, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3351;
mov.b32 r3357, {high, high};
}
{
mul.f16x2 r3359, r2592, r3357;
}
{
fma.rn.f16x2 r3362, r2556, r3355, r3359;
}
{
mul.f16x2 r3366, r2556, r3357;
}
{
neg.f16x2 r3369, r3366;
}
{
fma.rn.f16x2 r3371, r2592, r3355, r3369;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3375, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3377, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3379, {low, high};
}
{
mul.f16x2 r3380, r3377, r3379;
}
{
mul.f16x2 r3383, r3351, r3375;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3351;
mov.b32 r3386, {high, low};
}
{
fma.rn.f16x2 r3388, r3380, r3386, r3383;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3388;
mov.b32 r3392, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3388;
mov.b32 r3394, {high, high};
}
{
mul.f16x2 r3396, r2678, r3394;
}
{
fma.rn.f16x2 r3399, r2642, r3392, r3396;
}
{
mul.f16x2 r3403, r2642, r3394;
}
{
neg.f16x2 r3406, r3403;
}
{
fma.rn.f16x2 r3408, r2678, r3392, r3406;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3412, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3414, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3416, {low, high};
}
{
mul.f16x2 r3417, r3414, r3416;
}
{
mul.f16x2 r3420, r3388, r3412;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3388;
mov.b32 r3423, {high, low};
}
{
fma.rn.f16x2 r3425, r3417, r3423, r3420;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3425;
mov.b32 r3429, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3425;
mov.b32 r3431, {high, high};
}
{
mul.f16x2 r3433, r2764, r3431;
}
{
fma.rn.f16x2 r3436, r2728, r3429, r3433;
}
{
mul.f16x2 r3440, r2728, r3431;
}
{
neg.f16x2 r3443, r3440;
}
{
fma.rn.f16x2 r3445, r2764, r3429, r3443;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3449, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3451, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3453, {low, high};
}
{
mul.f16x2 r3454, r3451, r3453;
}
{
mul.f16x2 r3457, r3425, r3449;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3425;
mov.b32 r3460, {high, low};
}
{
fma.rn.f16x2 r3462, r3454, r3460, r3457;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3462;
mov.b32 r3466, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3462;
mov.b32 r3468, {high, high};
}
{
mul.f16x2 r3470, r2850, r3468;
}
{
fma.rn.f16x2 r3473, r2814, r3466, r3470;
}
{
mul.f16x2 r3477, r2814, r3468;
}
{
neg.f16x2 r3480, r3477;
}
{
fma.rn.f16x2 r3482, r2850, r3466, r3480;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3486, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3488, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3490, {low, high};
}
{
mul.f16x2 r3491, r3488, r3490;
}
{
mul.f16x2 r3494, r3462, r3486;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3462;
mov.b32 r3497, {high, low};
}
{
fma.rn.f16x2 r3499, r3491, r3497, r3494;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3499;
mov.b32 r3503, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3499;
mov.b32 r3505, {high, high};
}
{
mul.f16x2 r3507, r2180, r3505;
}
{
fma.rn.f16x2 r3510, r2144, r3503, r3507;
}
{
mul.f16x2 r3514, r2144, r3505;
}
{
neg.f16x2 r3517, r3514;
}
{
fma.rn.f16x2 r3519, r2180, r3503, r3517;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3523, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3525, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3527, {low, high};
}
{
mul.f16x2 r3528, r3525, r3527;
}
{
mul.f16x2 r3531, r3499, r3523;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3499;
mov.b32 r3534, {high, low};
}
{
fma.rn.f16x2 r3536, r3528, r3534, r3531;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3536;
mov.b32 r3540, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3536;
mov.b32 r3542, {high, high};
}
{
mul.f16x2 r3544, r2266, r3542;
}
{
fma.rn.f16x2 r3547, r2230, r3540, r3544;
}
{
mul.f16x2 r3551, r2230, r3542;
}
{
neg.f16x2 r3554, r3551;
}
{
fma.rn.f16x2 r3556, r2266, r3540, r3554;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3560, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3562, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3564, {low, high};
}
{
mul.f16x2 r3565, r3562, r3564;
}
{
mul.f16x2 r3568, r3536, r3560;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3536;
mov.b32 r3571, {high, low};
}
{
fma.rn.f16x2 r3573, r3565, r3571, r3568;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3573;
mov.b32 r3577, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3573;
mov.b32 r3579, {high, high};
}
{
mul.f16x2 r3581, r2352, r3579;
}
{
fma.rn.f16x2 r3584, r2316, r3577, r3581;
}
{
mul.f16x2 r3588, r2316, r3579;
}
{
neg.f16x2 r3591, r3588;
}
{
fma.rn.f16x2 r3593, r2352, r3577, r3591;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3597, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3599, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3601, {low, high};
}
{
mul.f16x2 r3602, r3599, r3601;
}
{
mul.f16x2 r3605, r3573, r3597;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3573;
mov.b32 r3608, {high, low};
}
{
fma.rn.f16x2 r3610, r3602, r3608, r3605;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3610;
mov.b32 r3614, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3610;
mov.b32 r3616, {high, high};
}
{
mul.f16x2 r3618, r2438, r3616;
}
{
fma.rn.f16x2 r3621, r2402, r3614, r3618;
}
{
mul.f16x2 r3625, r2402, r3616;
}
{
neg.f16x2 r3628, r3625;
}
{
fma.rn.f16x2 r3630, r2438, r3614, r3628;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3634, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3636, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3638, {low, high};
}
{
mul.f16x2 r3639, r3636, r3638;
}
{
mul.f16x2 r3642, r3610, r3634;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3610;
mov.b32 r3645, {high, low};
}
{
fma.rn.f16x2 r3647, r3639, r3645, r3642;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3647;
mov.b32 r3651, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3647;
mov.b32 r3653, {high, high};
}
{
mul.f16x2 r3655, r2524, r3653;
}
{
fma.rn.f16x2 r3658, r2488, r3651, r3655;
}
{
mul.f16x2 r3662, r2488, r3653;
}
{
neg.f16x2 r3665, r3662;
}
{
fma.rn.f16x2 r3667, r2524, r3651, r3665;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3671, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3673, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3675, {low, high};
}
{
mul.f16x2 r3676, r3673, r3675;
}
{
mul.f16x2 r3679, r3647, r3671;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3647;
mov.b32 r3682, {high, low};
}
{
fma.rn.f16x2 r3684, r3676, r3682, r3679;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3684;
mov.b32 r3688, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3684;
mov.b32 r3690, {high, high};
}
{
mul.f16x2 r3692, r2610, r3690;
}
{
fma.rn.f16x2 r3695, r2574, r3688, r3692;
}
{
mul.f16x2 r3699, r2574, r3690;
}
{
neg.f16x2 r3702, r3699;
}
{
fma.rn.f16x2 r3704, r2610, r3688, r3702;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3708, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3710, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3712, {low, high};
}
{
mul.f16x2 r3713, r3710, r3712;
}
{
mul.f16x2 r3716, r3684, r3708;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3684;
mov.b32 r3719, {high, low};
}
{
fma.rn.f16x2 r3721, r3713, r3719, r3716;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3721;
mov.b32 r3725, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3721;
mov.b32 r3727, {high, high};
}
{
mul.f16x2 r3729, r2696, r3727;
}
{
fma.rn.f16x2 r3732, r2660, r3725, r3729;
}
{
mul.f16x2 r3736, r2660, r3727;
}
{
neg.f16x2 r3739, r3736;
}
{
fma.rn.f16x2 r3741, r2696, r3725, r3739;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3745, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3747, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3749, {low, high};
}
{
mul.f16x2 r3750, r3747, r3749;
}
{
mul.f16x2 r3753, r3721, r3745;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3721;
mov.b32 r3756, {high, low};
}
{
fma.rn.f16x2 r3758, r3750, r3756, r3753;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3758;
mov.b32 r3762, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3758;
mov.b32 r3764, {high, high};
}
{
mul.f16x2 r3766, r2782, r3764;
}
{
fma.rn.f16x2 r3769, r2746, r3762, r3766;
}
{
mul.f16x2 r3773, r2746, r3764;
}
{
neg.f16x2 r3776, r3773;
}
{
fma.rn.f16x2 r3778, r2782, r3762, r3776;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3782, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3784, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3786, {low, high};
}
{
mul.f16x2 r3787, r3784, r3786;
}
{
mul.f16x2 r3790, r3758, r3782;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3758;
mov.b32 r3793, {high, low};
}
{
fma.rn.f16x2 r3795, r3787, r3793, r3790;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3795;
mov.b32 r3799, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3795;
mov.b32 r3801, {high, high};
}
{
mul.f16x2 r3803, r2868, r3801;
}
{
fma.rn.f16x2 r3806, r2832, r3799, r3803;
}
{
mul.f16x2 r3810, r2832, r3801;
}
{
neg.f16x2 r3813, r3810;
}
{
fma.rn.f16x2 r3815, r2868, r3799, r3813;
}
mad.lo.s32 r4617, r4614, 648, r4612;
barrier.sync 0;
mad.lo.s32 r4618, r4616, 216, r4617;
st.shared.v2.f32 [r4618], {r2102, r2108};
st.shared.v2.f32 [r4618+8], {r2881, r2890};
st.shared.v2.f32 [r4618+16], {r2918, r2927};
st.shared.v2.f32 [r4618+24], {r2955, r2964};
st.shared.v2.f32 [r4618+32], {r2992, r3001};
st.shared.v2.f32 [r4618+40], {r3029, r3038};
st.shared.v2.f32 [r4618+48], {r3066, r3075};
st.shared.v2.f32 [r4618+56], {r3103, r3112};
st.shared.v2.f32 [r4618+64], {r3140, r3149};
st.shared.v2.f32 [r4618+72], {r3177, r3186};
st.shared.v2.f32 [r4618+80], {r3214, r3223};
st.shared.v2.f32 [r4618+88], {r3251, r3260};
st.shared.v2.f32 [r4618+96], {r3288, r3297};
st.shared.v2.f32 [r4618+104], {r3325, r3334};
st.shared.v2.f32 [r4618+112], {r3362, r3371};
st.shared.v2.f32 [r4618+120], {r3399, r3408};
st.shared.v2.f32 [r4618+128], {r3436, r3445};
st.shared.v2.f32 [r4618+136], {r3473, r3482};
st.shared.v2.f32 [r4618+144], {r3510, r3519};
st.shared.v2.f32 [r4618+152], {r3547, r3556};
st.shared.v2.f32 [r4618+160], {r3584, r3593};
st.shared.v2.f32 [r4618+168], {r3621, r3630};
st.shared.v2.f32 [r4618+176], {r3658, r3667};
st.shared.v2.f32 [r4618+184], {r3695, r3704};
st.shared.v2.f32 [r4618+192], {r3732, r3741};
st.shared.v2.f32 [r4618+200], {r3769, r3778};
st.shared.v2.f32 [r4618+208], {r3806, r3815};
barrier.sync 0;
mad.lo.s32 r4619, r4616, -208, r4618;
ld.shared.u32 r3842, [r4619];
ld.shared.u32 r3848, [r4619+4];
ld.shared.u32 r3928, [r4619+24];
ld.shared.u32 r3934, [r4619+28];
ld.shared.u32 r4014, [r4619+48];
ld.shared.u32 r4020, [r4619+52];
ld.shared.u32 r4100, [r4619+72];
ld.shared.u32 r4106, [r4619+76];
ld.shared.u32 r4186, [r4619+96];
ld.shared.u32 r4192, [r4619+100];
ld.shared.u32 r4272, [r4619+120];
ld.shared.u32 r4278, [r4619+124];
ld.shared.u32 r4358, [r4619+144];
ld.shared.u32 r4364, [r4619+148];
ld.shared.u32 r4444, [r4619+168];
ld.shared.u32 r4450, [r4619+172];
ld.shared.u32 r4530, [r4619+192];
ld.shared.u32 r4536, [r4619+196];
ld.shared.u32 r3839, [r4619+216];
ld.shared.u32 r3845, [r4619+220];
ld.shared.u32 r3925, [r4619+240];
ld.shared.u32 r3931, [r4619+244];
ld.shared.u32 r4011, [r4619+264];
ld.shared.u32 r4017, [r4619+268];
ld.shared.u32 r4097, [r4619+288];
ld.shared.u32 r4103, [r4619+292];
ld.shared.u32 r4183, [r4619+312];
ld.shared.u32 r4189, [r4619+316];
ld.shared.u32 r4269, [r4619+336];
ld.shared.u32 r4275, [r4619+340];
ld.shared.u32 r4355, [r4619+360];
ld.shared.u32 r4361, [r4619+364];
ld.shared.u32 r4441, [r4619+384];
ld.shared.u32 r4447, [r4619+388];
ld.shared.u32 r4527, [r4619+408];
ld.shared.u32 r4533, [r4619+412];
ld.shared.u32 r3840, [r4619+432];
ld.shared.u32 r3846, [r4619+436];
ld.shared.u32 r3926, [r4619+456];
ld.shared.u32 r3932, [r4619+460];
ld.shared.u32 r4012, [r4619+480];
ld.shared.u32 r4018, [r4619+484];
ld.shared.u32 r4098, [r4619+504];
ld.shared.u32 r4104, [r4619+508];
ld.shared.u32 r4184, [r4619+528];
ld.shared.u32 r4190, [r4619+532];
ld.shared.u32 r4270, [r4619+552];
ld.shared.u32 r4276, [r4619+556];
ld.shared.u32 r4356, [r4619+576];
ld.shared.u32 r4362, [r4619+580];
ld.shared.u32 r4442, [r4619+600];
ld.shared.u32 r4448, [r4619+604];
ld.shared.u32 r4528, [r4619+624];
ld.shared.u32 r4534, [r4619+628];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r3836, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r3837, {low, high};
}
{
add.f16x2 r3838, r3839, r3840;
}
{
add.f16x2 %0, r3842, r3838;
}
{
add.f16x2 r3844, r3845, r3846;
}
{
add.f16x2 %1, r3848, r3844;
}
{
add.f16x2 r3850, r3839, r3840;
}
{
mul.f16x2 r3853, r3850, r3836;
}
{
add.f16x2 r3856, r3842, r3853;
}
{
sub.f16x2 r3859, r3845, r3846;
}
{
mul.f16x2 r3862, r3859, r3837;
}
{
add.f16x2 %18, r3856, r3862;
}
{
add.f16x2 r3868, r3839, r3840;
}
{
mul.f16x2 r3871, r3868, r3836;
}
{
add.f16x2 r3874, r3842, r3871;
}
{
sub.f16x2 r3877, r3845, r3846;
}
{
mul.f16x2 r3880, r3877, r3837;
}
{
sub.f16x2 %36, r3874, r3880;
}
{
add.f16x2 r3886, r3845, r3846;
}
{
mul.f16x2 r3889, r3886, r3836;
}
{
add.f16x2 r3892, r3848, r3889;
}
{
sub.f16x2 r3895, r3839, r3840;
}
{
mul.f16x2 r3898, r3895, r3837;
}
{
sub.f16x2 %19, r3892, r3898;
}
{
add.f16x2 r3904, r3845, r3846;
}
{
mul.f16x2 r3907, r3904, r3836;
}
{
add.f16x2 r3910, r3848, r3907;
}
{
sub.f16x2 r3913, r3839, r3840;
}
{
mul.f16x2 r3916, r3913, r3837;
}
{
add.f16x2 %37, r3910, r3916;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r3922, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r3923, {low, high};
}
{
add.f16x2 r3924, r3925, r3926;
}
{
add.f16x2 %2, r3928, r3924;
}
{
add.f16x2 r3930, r3931, r3932;
}
{
add.f16x2 %3, r3934, r3930;
}
{
add.f16x2 r3936, r3925, r3926;
}
{
mul.f16x2 r3939, r3936, r3922;
}
{
add.f16x2 r3942, r3928, r3939;
}
{
sub.f16x2 r3945, r3931, r3932;
}
{
mul.f16x2 r3948, r3945, r3923;
}
{
add.f16x2 %20, r3942, r3948;
}
{
add.f16x2 r3954, r3925, r3926;
}
{
mul.f16x2 r3957, r3954, r3922;
}
{
add.f16x2 r3960, r3928, r3957;
}
{
sub.f16x2 r3963, r3931, r3932;
}
{
mul.f16x2 r3966, r3963, r3923;
}
{
sub.f16x2 %38, r3960, r3966;
}
{
add.f16x2 r3972, r3931, r3932;
}
{
mul.f16x2 r3975, r3972, r3922;
}
{
add.f16x2 r3978, r3934, r3975;
}
{
sub.f16x2 r3981, r3925, r3926;
}
{
mul.f16x2 r3984, r3981, r3923;
}
{
sub.f16x2 %21, r3978, r3984;
}
{
add.f16x2 r3990, r3931, r3932;
}
{
mul.f16x2 r3993, r3990, r3922;
}
{
add.f16x2 r3996, r3934, r3993;
}
{
sub.f16x2 r3999, r3925, r3926;
}
{
mul.f16x2 r4002, r3999, r3923;
}
{
add.f16x2 %39, r3996, r4002;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4008, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4009, {low, high};
}
{
add.f16x2 r4010, r4011, r4012;
}
{
add.f16x2 %4, r4014, r4010;
}
{
add.f16x2 r4016, r4017, r4018;
}
{
add.f16x2 %5, r4020, r4016;
}
{
add.f16x2 r4022, r4011, r4012;
}
{
mul.f16x2 r4025, r4022, r4008;
}
{
add.f16x2 r4028, r4014, r4025;
}
{
sub.f16x2 r4031, r4017, r4018;
}
{
mul.f16x2 r4034, r4031, r4009;
}
{
add.f16x2 %22, r4028, r4034;
}
{
add.f16x2 r4040, r4011, r4012;
}
{
mul.f16x2 r4043, r4040, r4008;
}
{
add.f16x2 r4046, r4014, r4043;
}
{
sub.f16x2 r4049, r4017, r4018;
}
{
mul.f16x2 r4052, r4049, r4009;
}
{
sub.f16x2 %40, r4046, r4052;
}
{
add.f16x2 r4058, r4017, r4018;
}
{
mul.f16x2 r4061, r4058, r4008;
}
{
add.f16x2 r4064, r4020, r4061;
}
{
sub.f16x2 r4067, r4011, r4012;
}
{
mul.f16x2 r4070, r4067, r4009;
}
{
sub.f16x2 %23, r4064, r4070;
}
{
add.f16x2 r4076, r4017, r4018;
}
{
mul.f16x2 r4079, r4076, r4008;
}
{
add.f16x2 r4082, r4020, r4079;
}
{
sub.f16x2 r4085, r4011, r4012;
}
{
mul.f16x2 r4088, r4085, r4009;
}
{
add.f16x2 %41, r4082, r4088;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4094, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4095, {low, high};
}
{
add.f16x2 r4096, r4097, r4098;
}
{
add.f16x2 %6, r4100, r4096;
}
{
add.f16x2 r4102, r4103, r4104;
}
{
add.f16x2 %7, r4106, r4102;
}
{
add.f16x2 r4108, r4097, r4098;
}
{
mul.f16x2 r4111, r4108, r4094;
}
{
add.f16x2 r4114, r4100, r4111;
}
{
sub.f16x2 r4117, r4103, r4104;
}
{
mul.f16x2 r4120, r4117, r4095;
}
{
add.f16x2 %24, r4114, r4120;
}
{
add.f16x2 r4126, r4097, r4098;
}
{
mul.f16x2 r4129, r4126, r4094;
}
{
add.f16x2 r4132, r4100, r4129;
}
{
sub.f16x2 r4135, r4103, r4104;
}
{
mul.f16x2 r4138, r4135, r4095;
}
{
sub.f16x2 %42, r4132, r4138;
}
{
add.f16x2 r4144, r4103, r4104;
}
{
mul.f16x2 r4147, r4144, r4094;
}
{
add.f16x2 r4150, r4106, r4147;
}
{
sub.f16x2 r4153, r4097, r4098;
}
{
mul.f16x2 r4156, r4153, r4095;
}
{
sub.f16x2 %25, r4150, r4156;
}
{
add.f16x2 r4162, r4103, r4104;
}
{
mul.f16x2 r4165, r4162, r4094;
}
{
add.f16x2 r4168, r4106, r4165;
}
{
sub.f16x2 r4171, r4097, r4098;
}
{
mul.f16x2 r4174, r4171, r4095;
}
{
add.f16x2 %43, r4168, r4174;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4180, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4181, {low, high};
}
{
add.f16x2 r4182, r4183, r4184;
}
{
add.f16x2 %8, r4186, r4182;
}
{
add.f16x2 r4188, r4189, r4190;
}
{
add.f16x2 %9, r4192, r4188;
}
{
add.f16x2 r4194, r4183, r4184;
}
{
mul.f16x2 r4197, r4194, r4180;
}
{
add.f16x2 r4200, r4186, r4197;
}
{
sub.f16x2 r4203, r4189, r4190;
}
{
mul.f16x2 r4206, r4203, r4181;
}
{
add.f16x2 %26, r4200, r4206;
}
{
add.f16x2 r4212, r4183, r4184;
}
{
mul.f16x2 r4215, r4212, r4180;
}
{
add.f16x2 r4218, r4186, r4215;
}
{
sub.f16x2 r4221, r4189, r4190;
}
{
mul.f16x2 r4224, r4221, r4181;
}
{
sub.f16x2 %44, r4218, r4224;
}
{
add.f16x2 r4230, r4189, r4190;
}
{
mul.f16x2 r4233, r4230, r4180;
}
{
add.f16x2 r4236, r4192, r4233;
}
{
sub.f16x2 r4239, r4183, r4184;
}
{
mul.f16x2 r4242, r4239, r4181;
}
{
sub.f16x2 %27, r4236, r4242;
}
{
add.f16x2 r4248, r4189, r4190;
}
{
mul.f16x2 r4251, r4248, r4180;
}
{
add.f16x2 r4254, r4192, r4251;
}
{
sub.f16x2 r4257, r4183, r4184;
}
{
mul.f16x2 r4260, r4257, r4181;
}
{
add.f16x2 %45, r4254, r4260;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4266, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4267, {low, high};
}
{
add.f16x2 r4268, r4269, r4270;
}
{
add.f16x2 %10, r4272, r4268;
}
{
add.f16x2 r4274, r4275, r4276;
}
{
add.f16x2 %11, r4278, r4274;
}
{
add.f16x2 r4280, r4269, r4270;
}
{
mul.f16x2 r4283, r4280, r4266;
}
{
add.f16x2 r4286, r4272, r4283;
}
{
sub.f16x2 r4289, r4275, r4276;
}
{
mul.f16x2 r4292, r4289, r4267;
}
{
add.f16x2 %28, r4286, r4292;
}
{
add.f16x2 r4298, r4269, r4270;
}
{
mul.f16x2 r4301, r4298, r4266;
}
{
add.f16x2 r4304, r4272, r4301;
}
{
sub.f16x2 r4307, r4275, r4276;
}
{
mul.f16x2 r4310, r4307, r4267;
}
{
sub.f16x2 %46, r4304, r4310;
}
{
add.f16x2 r4316, r4275, r4276;
}
{
mul.f16x2 r4319, r4316, r4266;
}
{
add.f16x2 r4322, r4278, r4319;
}
{
sub.f16x2 r4325, r4269, r4270;
}
{
mul.f16x2 r4328, r4325, r4267;
}
{
sub.f16x2 %29, r4322, r4328;
}
{
add.f16x2 r4334, r4275, r4276;
}
{
mul.f16x2 r4337, r4334, r4266;
}
{
add.f16x2 r4340, r4278, r4337;
}
{
sub.f16x2 r4343, r4269, r4270;
}
{
mul.f16x2 r4346, r4343, r4267;
}
{
add.f16x2 %47, r4340, r4346;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4352, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4353, {low, high};
}
{
add.f16x2 r4354, r4355, r4356;
}
{
add.f16x2 %12, r4358, r4354;
}
{
add.f16x2 r4360, r4361, r4362;
}
{
add.f16x2 %13, r4364, r4360;
}
{
add.f16x2 r4366, r4355, r4356;
}
{
mul.f16x2 r4369, r4366, r4352;
}
{
add.f16x2 r4372, r4358, r4369;
}
{
sub.f16x2 r4375, r4361, r4362;
}
{
mul.f16x2 r4378, r4375, r4353;
}
{
add.f16x2 %30, r4372, r4378;
}
{
add.f16x2 r4384, r4355, r4356;
}
{
mul.f16x2 r4387, r4384, r4352;
}
{
add.f16x2 r4390, r4358, r4387;
}
{
sub.f16x2 r4393, r4361, r4362;
}
{
mul.f16x2 r4396, r4393, r4353;
}
{
sub.f16x2 %48, r4390, r4396;
}
{
add.f16x2 r4402, r4361, r4362;
}
{
mul.f16x2 r4405, r4402, r4352;
}
{
add.f16x2 r4408, r4364, r4405;
}
{
sub.f16x2 r4411, r4355, r4356;
}
{
mul.f16x2 r4414, r4411, r4353;
}
{
sub.f16x2 %31, r4408, r4414;
}
{
add.f16x2 r4420, r4361, r4362;
}
{
mul.f16x2 r4423, r4420, r4352;
}
{
add.f16x2 r4426, r4364, r4423;
}
{
sub.f16x2 r4429, r4355, r4356;
}
{
mul.f16x2 r4432, r4429, r4353;
}
{
add.f16x2 %49, r4426, r4432;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4438, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4439, {low, high};
}
{
add.f16x2 r4440, r4441, r4442;
}
{
add.f16x2 %14, r4444, r4440;
}
{
add.f16x2 r4446, r4447, r4448;
}
{
add.f16x2 %15, r4450, r4446;
}
{
add.f16x2 r4452, r4441, r4442;
}
{
mul.f16x2 r4455, r4452, r4438;
}
{
add.f16x2 r4458, r4444, r4455;
}
{
sub.f16x2 r4461, r4447, r4448;
}
{
mul.f16x2 r4464, r4461, r4439;
}
{
add.f16x2 %32, r4458, r4464;
}
{
add.f16x2 r4470, r4441, r4442;
}
{
mul.f16x2 r4473, r4470, r4438;
}
{
add.f16x2 r4476, r4444, r4473;
}
{
sub.f16x2 r4479, r4447, r4448;
}
{
mul.f16x2 r4482, r4479, r4439;
}
{
sub.f16x2 %50, r4476, r4482;
}
{
add.f16x2 r4488, r4447, r4448;
}
{
mul.f16x2 r4491, r4488, r4438;
}
{
add.f16x2 r4494, r4450, r4491;
}
{
sub.f16x2 r4497, r4441, r4442;
}
{
mul.f16x2 r4500, r4497, r4439;
}
{
sub.f16x2 %33, r4494, r4500;
}
{
add.f16x2 r4506, r4447, r4448;
}
{
mul.f16x2 r4509, r4506, r4438;
}
{
add.f16x2 r4512, r4450, r4509;
}
{
sub.f16x2 r4515, r4441, r4442;
}
{
mul.f16x2 r4518, r4515, r4439;
}
{
add.f16x2 %51, r4512, r4518;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4524, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4525, {low, high};
}
{
add.f16x2 r4526, r4527, r4528;
}
{
add.f16x2 %16, r4530, r4526;
}
{
add.f16x2 r4532, r4533, r4534;
}
{
add.f16x2 %17, r4536, r4532;
}
{
add.f16x2 r4538, r4527, r4528;
}
{
mul.f16x2 r4541, r4538, r4524;
}
{
add.f16x2 r4544, r4530, r4541;
}
{
sub.f16x2 r4547, r4533, r4534;
}
{
mul.f16x2 r4550, r4547, r4525;
}
{
add.f16x2 %34, r4544, r4550;
}
{
add.f16x2 r4556, r4527, r4528;
}
{
mul.f16x2 r4559, r4556, r4524;
}
{
add.f16x2 r4562, r4530, r4559;
}
{
sub.f16x2 r4565, r4533, r4534;
}
{
mul.f16x2 r4568, r4565, r4525;
}
{
sub.f16x2 %52, r4562, r4568;
}
{
add.f16x2 r4574, r4533, r4534;
}
{
mul.f16x2 r4577, r4574, r4524;
}
{
add.f16x2 r4580, r4536, r4577;
}
{
sub.f16x2 r4583, r4527, r4528;
}
{
mul.f16x2 r4586, r4583, r4525;
}
{
sub.f16x2 %35, r4580, r4586;
}
{
add.f16x2 r4592, r4533, r4534;
}
{
mul.f16x2 r4595, r4592, r4524;
}
{
add.f16x2 r4598, r4536, r4595;
}
{
sub.f16x2 r4601, r4527, r4528;
}
{
mul.f16x2 r4604, r4601, r4525;
}
{
add.f16x2 %53, r4598, r4604;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[23].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1076, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<404>;
.reg .b32 r<4676>;
.reg .b64 rd<4>;
mov.u32 r4674, %tid.y;
mov.u32 r4675, %54;
mad.lo.s32 r4612, r4674, 324, r4675;
mov.u32 r4613, %tid.x;
mov.f32 f398, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1, {low, high};
}
mov.f32 f400, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2, {low, high};
}
{
add.f16x2 r3, %106, %91;
}
{
add.f16x2 r6, %61, r3;
}
{
add.f16x2 r9, %70, %104;
}
{
add.f16x2 r12, %76, r9;
}
{
add.f16x2 r15, %106, %91;
}
{
mul.f16x2 r18, r15, r1;
}
{
add.f16x2 r21, %61, r18;
}
{
sub.f16x2 r24, %70, %104;
}
{
mul.f16x2 r27, r24, r2;
}
{
add.f16x2 r30, r21, r27;
}
{
add.f16x2 r33, %106, %91;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %61, r36;
}
{
sub.f16x2 r42, %70, %104;
}
{
mul.f16x2 r45, r42, r2;
}
{
sub.f16x2 r48, r39, r45;
}
{
add.f16x2 r51, %70, %104;
}
{
mul.f16x2 r54, r51, r1;
}
{
add.f16x2 r57, %76, r54;
}
{
sub.f16x2 r60, %106, %91;
}
{
mul.f16x2 r63, r60, r2;
}
{
sub.f16x2 r66, r57, r63;
}
{
add.f16x2 r69, %70, %104;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %76, r72;
}
{
sub.f16x2 r78, %106, %91;
}
{
mul.f16x2 r81, r78, r2;
}
{
add.f16x2 r84, r75, r81;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r88, {low, high};
}
{
add.f16x2 r89, %105, %90;
}
{
add.f16x2 r92, %60, r89;
}
{
add.f16x2 r95, %69, %102;
}
{
add.f16x2 r98, %75, r95;
}
{
add.f16x2 r101, %105, %90;
}
{
mul.f16x2 r104, r101, r87;
}
{
add.f16x2 r107, %60, r104;
}
{
sub.f16x2 r110, %69, %102;
}
{
mul.f16x2 r113, r110, r88;
}
{
add.f16x2 r116, r107, r113;
}
{
add.f16x2 r119, %105, %90;
}
{
mul.f16x2 r122, r119, r87;
}
{
add.f16x2 r125, %60, r122;
}
{
sub.f16x2 r128, %69, %102;
}
{
mul.f16x2 r131, r128, r88;
}
{
sub.f16x2 r134, r125, r131;
}
{
add.f16x2 r137, %69, %102;
}
{
mul.f16x2 r140, r137, r87;
}
{
add.f16x2 r143, %75, r140;
}
{
sub.f16x2 r146, %105, %90;
}
{
mul.f16x2 r149, r146, r88;
}
{
sub.f16x2 r152, r143, r149;
}
{
add.f16x2 r155, %69, %102;
}
{
mul.f16x2 r158, r155, r87;
}
{
add.f16x2 r161, %75, r158;
}
{
sub.f16x2 r164, %105, %90;
}
{
mul.f16x2 r167, r164, r88;
}
{
add.f16x2 r170, r161, r167;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r173, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r174, {low, high};
}
{
add.f16x2 r175, %103, %88;
}
{
add.f16x2 r178, %59, r175;
}
{
add.f16x2 r181, %64, %101;
}
{
add.f16x2 r184, %74, r181;
}
{
add.f16x2 r187, %103, %88;
}
{
mul.f16x2 r190, r187, r173;
}
{
add.f16x2 r193, %59, r190;
}
{
sub.f16x2 r196, %64, %101;
}
{
mul.f16x2 r199, r196, r174;
}
{
add.f16x2 r202, r193, r199;
}
{
add.f16x2 r205, %103, %88;
}
{
mul.f16x2 r208, r205, r173;
}
{
add.f16x2 r211, %59, r208;
}
{
sub.f16x2 r214, %64, %101;
}
{
mul.f16x2 r217, r214, r174;
}
{
sub.f16x2 r220, r211, r217;
}
{
add.f16x2 r223, %64, %101;
}
{
mul.f16x2 r226, r223, r173;
}
{
add.f16x2 r229, %74, r226;
}
{
sub.f16x2 r232, %103, %88;
}
{
mul.f16x2 r235, r232, r174;
}
{
sub.f16x2 r238, r229, r235;
}
{
add.f16x2 r241, %64, %101;
}
{
mul.f16x2 r244, r241, r173;
}
{
add.f16x2 r247, %74, r244;
}
{
sub.f16x2 r250, %103, %88;
}
{
mul.f16x2 r253, r250, r174;
}
{
add.f16x2 r256, r247, r253;
}
mov.f32 f178, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r259, {low, high};
}
mov.f32 f180, 0f3F248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
mov.b32 r260, {low, high};
}
mov.f32 f190, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r261, {low, high};
}
mov.f32 f192, 0f3F7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f192;
cvt.rn.f16.f32 high, f192;
mov.b32 r262, {low, high};
}
mov.f32 f214, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r265, {low, high};
}
mov.f32 f216, 0f3EAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r266, {low, high};
}
{
mul.f16x2 r275, r116, r259;
}
{
mul.f16x2 r278, r152, r260;
}
{
sub.f16x2 r281, r275, r278;
}
{
mul.f16x2 r284, r116, r260;
}
{
fma.rn.f16x2 r287, r152, r259, r284;
}
{
mul.f16x2 r291, r202, r261;
}
{
mul.f16x2 r294, r238, r262;
}
{
sub.f16x2 r297, r291, r294;
}
{
mul.f16x2 r300, r202, r262;
}
{
fma.rn.f16x2 r303, r238, r261, r300;
}
{
mul.f16x2 r307, r134, r261;
}
{
mul.f16x2 r310, r170, r262;
}
{
sub.f16x2 r313, r307, r310;
}
{
mul.f16x2 r316, r134, r262;
}
{
fma.rn.f16x2 r319, r170, r261, r316;
}
{
mul.f16x2 r323, r220, r265;
}
{
mul.f16x2 r326, r256, r266;
}
{
sub.f16x2 r329, r323, r326;
}
{
mul.f16x2 r332, r220, r266;
}
{
fma.rn.f16x2 r335, r256, r265, r332;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r339, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r340, {low, high};
}
{
add.f16x2 r341, r92, r178;
}
{
add.f16x2 r344, r6, r341;
}
{
add.f16x2 r347, r98, r184;
}
{
add.f16x2 r350, r12, r347;
}
{
add.f16x2 r353, r92, r178;
}
{
mul.f16x2 r356, r353, r339;
}
{
add.f16x2 r359, r6, r356;
}
{
sub.f16x2 r362, r98, r184;
}
{
mul.f16x2 r365, r362, r340;
}
{
add.f16x2 r368, r359, r365;
}
{
add.f16x2 r371, r92, r178;
}
{
mul.f16x2 r374, r371, r339;
}
{
add.f16x2 r377, r6, r374;
}
{
sub.f16x2 r380, r98, r184;
}
{
mul.f16x2 r383, r380, r340;
}
{
sub.f16x2 r386, r377, r383;
}
{
add.f16x2 r389, r98, r184;
}
{
mul.f16x2 r392, r389, r339;
}
{
add.f16x2 r395, r12, r392;
}
{
sub.f16x2 r398, r92, r178;
}
{
mul.f16x2 r401, r398, r340;
}
{
sub.f16x2 r404, r395, r401;
}
{
add.f16x2 r407, r98, r184;
}
{
mul.f16x2 r410, r407, r339;
}
{
add.f16x2 r413, r12, r410;
}
{
sub.f16x2 r416, r92, r178;
}
{
mul.f16x2 r419, r416, r340;
}
{
add.f16x2 r422, r413, r419;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r425, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r426, {low, high};
}
{
add.f16x2 r427, r281, r297;
}
{
add.f16x2 r430, r30, r427;
}
{
add.f16x2 r433, r287, r303;
}
{
add.f16x2 r436, r66, r433;
}
{
add.f16x2 r439, r281, r297;
}
{
mul.f16x2 r442, r439, r425;
}
{
add.f16x2 r445, r30, r442;
}
{
sub.f16x2 r448, r287, r303;
}
{
mul.f16x2 r451, r448, r426;
}
{
add.f16x2 r454, r445, r451;
}
{
add.f16x2 r457, r281, r297;
}
{
mul.f16x2 r460, r457, r425;
}
{
add.f16x2 r463, r30, r460;
}
{
sub.f16x2 r466, r287, r303;
}
{
mul.f16x2 r469, r466, r426;
}
{
sub.f16x2 r472, r463, r469;
}
{
add.f16x2 r475, r287, r303;
}
{
mul.f16x2 r478, r475, r425;
}
{
add.f16x2 r481, r66, r478;
}
{
sub.f16x2 r484, r281, r297;
}
{
mul.f16x2 r487, r484, r426;
}
{
sub.f16x2 r490, r481, r487;
}
{
add.f16x2 r493, r287, r303;
}
{
mul.f16x2 r496, r493, r425;
}
{
add.f16x2 r499, r66, r496;
}
{
sub.f16x2 r502, r281, r297;
}
{
mul.f16x2 r505, r502, r426;
}
{
add.f16x2 r508, r499, r505;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r512, {low, high};
}
{
add.f16x2 r513, r313, r329;
}
{
add.f16x2 r516, r48, r513;
}
{
add.f16x2 r519, r319, r335;
}
{
add.f16x2 r522, r84, r519;
}
{
add.f16x2 r525, r313, r329;
}
{
mul.f16x2 r528, r525, r511;
}
{
add.f16x2 r531, r48, r528;
}
{
sub.f16x2 r534, r319, r335;
}
{
mul.f16x2 r537, r534, r512;
}
{
add.f16x2 r540, r531, r537;
}
{
add.f16x2 r543, r313, r329;
}
{
mul.f16x2 r546, r543, r511;
}
{
add.f16x2 r549, r48, r546;
}
{
sub.f16x2 r552, r319, r335;
}
{
mul.f16x2 r555, r552, r512;
}
{
sub.f16x2 r558, r549, r555;
}
{
add.f16x2 r561, r319, r335;
}
{
mul.f16x2 r564, r561, r511;
}
{
add.f16x2 r567, r84, r564;
}
{
sub.f16x2 r570, r313, r329;
}
{
mul.f16x2 r573, r570, r512;
}
{
sub.f16x2 r576, r567, r573;
}
{
add.f16x2 r579, r319, r335;
}
{
mul.f16x2 r582, r579, r511;
}
{
add.f16x2 r585, r84, r582;
}
{
sub.f16x2 r588, r313, r329;
}
{
mul.f16x2 r591, r588, r512;
}
{
add.f16x2 r594, r585, r591;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r597, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r598, {low, high};
}
{
add.f16x2 r599, %85, %68;
}
{
add.f16x2 r602, %89, r599;
}
{
add.f16x2 r605, %97, %83;
}
{
add.f16x2 r608, %100, r605;
}
{
add.f16x2 r611, %85, %68;
}
{
mul.f16x2 r614, r611, r597;
}
{
add.f16x2 r617, %89, r614;
}
{
sub.f16x2 r620, %97, %83;
}
{
mul.f16x2 r623, r620, r598;
}
{
add.f16x2 r626, r617, r623;
}
{
add.f16x2 r629, %85, %68;
}
{
mul.f16x2 r632, r629, r597;
}
{
add.f16x2 r635, %89, r632;
}
{
sub.f16x2 r638, %97, %83;
}
{
mul.f16x2 r641, r638, r598;
}
{
sub.f16x2 r644, r635, r641;
}
{
add.f16x2 r647, %97, %83;
}
{
mul.f16x2 r650, r647, r597;
}
{
add.f16x2 r653, %100, r650;
}
{
sub.f16x2 r656, %85, %68;
}
{
mul.f16x2 r659, r656, r598;
}
{
sub.f16x2 r662, r653, r659;
}
{
add.f16x2 r665, %97, %83;
}
{
mul.f16x2 r668, r665, r597;
}
{
add.f16x2 r671, %100, r668;
}
{
sub.f16x2 r674, %85, %68;
}
{
mul.f16x2 r677, r674, r598;
}
{
add.f16x2 r680, r671, r677;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r683, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r684, {low, high};
}
{
add.f16x2 r685, %84, %67;
}
{
add.f16x2 r688, %87, r685;
}
{
add.f16x2 r691, %96, %81;
}
{
add.f16x2 r694, %99, r691;
}
{
add.f16x2 r697, %84, %67;
}
{
mul.f16x2 r700, r697, r683;
}
{
add.f16x2 r703, %87, r700;
}
{
sub.f16x2 r706, %96, %81;
}
{
mul.f16x2 r709, r706, r684;
}
{
add.f16x2 r712, r703, r709;
}
{
add.f16x2 r715, %84, %67;
}
{
mul.f16x2 r718, r715, r683;
}
{
add.f16x2 r721, %87, r718;
}
{
sub.f16x2 r724, %96, %81;
}
{
mul.f16x2 r727, r724, r684;
}
{
sub.f16x2 r730, r721, r727;
}
{
add.f16x2 r733, %96, %81;
}
{
mul.f16x2 r736, r733, r683;
}
{
add.f16x2 r739, %99, r736;
}
{
sub.f16x2 r742, %84, %67;
}
{
mul.f16x2 r745, r742, r684;
}
{
sub.f16x2 r748, r739, r745;
}
{
add.f16x2 r751, %96, %81;
}
{
mul.f16x2 r754, r751, r683;
}
{
add.f16x2 r757, %99, r754;
}
{
sub.f16x2 r760, %84, %67;
}
{
mul.f16x2 r763, r760, r684;
}
{
add.f16x2 r766, r757, r763;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r769, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r770, {low, high};
}
{
add.f16x2 r771, %82, %66;
}
{
add.f16x2 r774, %86, r771;
}
{
add.f16x2 r777, %94, %80;
}
{
add.f16x2 r780, %98, r777;
}
{
add.f16x2 r783, %82, %66;
}
{
mul.f16x2 r786, r783, r769;
}
{
add.f16x2 r789, %86, r786;
}
{
sub.f16x2 r792, %94, %80;
}
{
mul.f16x2 r795, r792, r770;
}
{
add.f16x2 r798, r789, r795;
}
{
add.f16x2 r801, %82, %66;
}
{
mul.f16x2 r804, r801, r769;
}
{
add.f16x2 r807, %86, r804;
}
{
sub.f16x2 r810, %94, %80;
}
{
mul.f16x2 r813, r810, r770;
}
{
sub.f16x2 r816, r807, r813;
}
{
add.f16x2 r819, %94, %80;
}
{
mul.f16x2 r822, r819, r769;
}
{
add.f16x2 r825, %98, r822;
}
{
sub.f16x2 r828, %82, %66;
}
{
mul.f16x2 r831, r828, r770;
}
{
sub.f16x2 r834, r825, r831;
}
{
add.f16x2 r837, %94, %80;
}
{
mul.f16x2 r840, r837, r769;
}
{
add.f16x2 r843, %98, r840;
}
{
sub.f16x2 r846, %82, %66;
}
{
mul.f16x2 r849, r846, r770;
}
{
add.f16x2 r852, r843, r849;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r855, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
mov.b32 r856, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r857, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f192;
cvt.rn.f16.f32 high, f192;
mov.b32 r858, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r861, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r862, {low, high};
}
{
mul.f16x2 r871, r712, r855;
}
{
mul.f16x2 r874, r748, r856;
}
{
sub.f16x2 r877, r871, r874;
}
{
mul.f16x2 r880, r712, r856;
}
{
fma.rn.f16x2 r883, r748, r855, r880;
}
{
mul.f16x2 r887, r798, r857;
}
{
mul.f16x2 r890, r834, r858;
}
{
sub.f16x2 r893, r887, r890;
}
{
mul.f16x2 r896, r798, r858;
}
{
fma.rn.f16x2 r899, r834, r857, r896;
}
{
mul.f16x2 r903, r730, r857;
}
{
mul.f16x2 r906, r766, r858;
}
{
sub.f16x2 r909, r903, r906;
}
{
mul.f16x2 r912, r730, r858;
}
{
fma.rn.f16x2 r915, r766, r857, r912;
}
{
mul.f16x2 r919, r816, r861;
}
{
mul.f16x2 r922, r852, r862;
}
{
sub.f16x2 r925, r919, r922;
}
{
mul.f16x2 r928, r816, r862;
}
{
fma.rn.f16x2 r931, r852, r861, r928;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r935, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r936, {low, high};
}
{
add.f16x2 r937, r688, r774;
}
{
add.f16x2 r940, r602, r937;
}
{
add.f16x2 r943, r694, r780;
}
{
add.f16x2 r946, r608, r943;
}
{
add.f16x2 r949, r688, r774;
}
{
mul.f16x2 r952, r949, r935;
}
{
add.f16x2 r955, r602, r952;
}
{
sub.f16x2 r958, r694, r780;
}
{
mul.f16x2 r961, r958, r936;
}
{
add.f16x2 r964, r955, r961;
}
{
add.f16x2 r967, r688, r774;
}
{
mul.f16x2 r970, r967, r935;
}
{
add.f16x2 r973, r602, r970;
}
{
sub.f16x2 r976, r694, r780;
}
{
mul.f16x2 r979, r976, r936;
}
{
sub.f16x2 r982, r973, r979;
}
{
add.f16x2 r985, r694, r780;
}
{
mul.f16x2 r988, r985, r935;
}
{
add.f16x2 r991, r608, r988;
}
{
sub.f16x2 r994, r688, r774;
}
{
mul.f16x2 r997, r994, r936;
}
{
sub.f16x2 r1000, r991, r997;
}
{
add.f16x2 r1003, r694, r780;
}
{
mul.f16x2 r1006, r1003, r935;
}
{
add.f16x2 r1009, r608, r1006;
}
{
sub.f16x2 r1012, r688, r774;
}
{
mul.f16x2 r1015, r1012, r936;
}
{
add.f16x2 r1018, r1009, r1015;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1022, {low, high};
}
{
add.f16x2 r1023, r877, r893;
}
{
add.f16x2 r1026, r626, r1023;
}
{
add.f16x2 r1029, r883, r899;
}
{
add.f16x2 r1032, r662, r1029;
}
{
add.f16x2 r1035, r877, r893;
}
{
mul.f16x2 r1038, r1035, r1021;
}
{
add.f16x2 r1041, r626, r1038;
}
{
sub.f16x2 r1044, r883, r899;
}
{
mul.f16x2 r1047, r1044, r1022;
}
{
add.f16x2 r1050, r1041, r1047;
}
{
add.f16x2 r1053, r877, r893;
}
{
mul.f16x2 r1056, r1053, r1021;
}
{
add.f16x2 r1059, r626, r1056;
}
{
sub.f16x2 r1062, r883, r899;
}
{
mul.f16x2 r1065, r1062, r1022;
}
{
sub.f16x2 r1068, r1059, r1065;
}
{
add.f16x2 r1071, r883, r899;
}
{
mul.f16x2 r1074, r1071, r1021;
}
{
add.f16x2 r1077, r662, r1074;
}
{
sub.f16x2 r1080, r877, r893;
}
{
mul.f16x2 r1083, r1080, r1022;
}
{
sub.f16x2 r1086, r1077, r1083;
}
{
add.f16x2 r1089, r883, r899;
}
{
mul.f16x2 r1092, r1089, r1021;
}
{
add.f16x2 r1095, r662, r1092;
}
{
sub.f16x2 r1098, r877, r893;
}
{
mul.f16x2 r1101, r1098, r1022;
}
{
add.f16x2 r1104, r1095, r1101;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1107, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1108, {low, high};
}
{
add.f16x2 r1109, r909, r925;
}
{
add.f16x2 r1112, r644, r1109;
}
{
add.f16x2 r1115, r915, r931;
}
{
add.f16x2 r1118, r680, r1115;
}
{
add.f16x2 r1121, r909, r925;
}
{
mul.f16x2 r1124, r1121, r1107;
}
{
add.f16x2 r1127, r644, r1124;
}
{
sub.f16x2 r1130, r915, r931;
}
{
mul.f16x2 r1133, r1130, r1108;
}
{
add.f16x2 r1136, r1127, r1133;
}
{
add.f16x2 r1139, r909, r925;
}
{
mul.f16x2 r1142, r1139, r1107;
}
{
add.f16x2 r1145, r644, r1142;
}
{
sub.f16x2 r1148, r915, r931;
}
{
mul.f16x2 r1151, r1148, r1108;
}
{
sub.f16x2 r1154, r1145, r1151;
}
{
add.f16x2 r1157, r915, r931;
}
{
mul.f16x2 r1160, r1157, r1107;
}
{
add.f16x2 r1163, r680, r1160;
}
{
sub.f16x2 r1166, r909, r925;
}
{
mul.f16x2 r1169, r1166, r1108;
}
{
sub.f16x2 r1172, r1163, r1169;
}
{
add.f16x2 r1175, r915, r931;
}
{
mul.f16x2 r1178, r1175, r1107;
}
{
add.f16x2 r1181, r680, r1178;
}
{
sub.f16x2 r1184, r909, r925;
}
{
mul.f16x2 r1187, r1184, r1108;
}
{
add.f16x2 r1190, r1181, r1187;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1193, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1194, {low, high};
}
{
add.f16x2 r1195, %58, %95;
}
{
add.f16x2 r1198, %65, r1195;
}
{
add.f16x2 r1201, %73, %56;
}
{
add.f16x2 r1204, %79, r1201;
}
{
add.f16x2 r1207, %58, %95;
}
{
mul.f16x2 r1210, r1207, r1193;
}
{
add.f16x2 r1213, %65, r1210;
}
{
sub.f16x2 r1216, %73, %56;
}
{
mul.f16x2 r1219, r1216, r1194;
}
{
add.f16x2 r1222, r1213, r1219;
}
{
add.f16x2 r1225, %58, %95;
}
{
mul.f16x2 r1228, r1225, r1193;
}
{
add.f16x2 r1231, %65, r1228;
}
{
sub.f16x2 r1234, %73, %56;
}
{
mul.f16x2 r1237, r1234, r1194;
}
{
sub.f16x2 r1240, r1231, r1237;
}
{
add.f16x2 r1243, %73, %56;
}
{
mul.f16x2 r1246, r1243, r1193;
}
{
add.f16x2 r1249, %79, r1246;
}
{
sub.f16x2 r1252, %58, %95;
}
{
mul.f16x2 r1255, r1252, r1194;
}
{
sub.f16x2 r1258, r1249, r1255;
}
{
add.f16x2 r1261, %73, %56;
}
{
mul.f16x2 r1264, r1261, r1193;
}
{
add.f16x2 r1267, %79, r1264;
}
{
sub.f16x2 r1270, %58, %95;
}
{
mul.f16x2 r1273, r1270, r1194;
}
{
add.f16x2 r1276, r1267, r1273;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1279, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1280, {low, high};
}
{
add.f16x2 r1281, %57, %93;
}
{
add.f16x2 r1284, %63, r1281;
}
{
add.f16x2 r1287, %72, %108;
}
{
add.f16x2 r1290, %78, r1287;
}
{
add.f16x2 r1293, %57, %93;
}
{
mul.f16x2 r1296, r1293, r1279;
}
{
add.f16x2 r1299, %63, r1296;
}
{
sub.f16x2 r1302, %72, %108;
}
{
mul.f16x2 r1305, r1302, r1280;
}
{
add.f16x2 r1308, r1299, r1305;
}
{
add.f16x2 r1311, %57, %93;
}
{
mul.f16x2 r1314, r1311, r1279;
}
{
add.f16x2 r1317, %63, r1314;
}
{
sub.f16x2 r1320, %72, %108;
}
{
mul.f16x2 r1323, r1320, r1280;
}
{
sub.f16x2 r1326, r1317, r1323;
}
{
add.f16x2 r1329, %72, %108;
}
{
mul.f16x2 r1332, r1329, r1279;
}
{
add.f16x2 r1335, %78, r1332;
}
{
sub.f16x2 r1338, %57, %93;
}
{
mul.f16x2 r1341, r1338, r1280;
}
{
sub.f16x2 r1344, r1335, r1341;
}
{
add.f16x2 r1347, %72, %108;
}
{
mul.f16x2 r1350, r1347, r1279;
}
{
add.f16x2 r1353, %78, r1350;
}
{
sub.f16x2 r1356, %57, %93;
}
{
mul.f16x2 r1359, r1356, r1280;
}
{
add.f16x2 r1362, r1353, r1359;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1365, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1366, {low, high};
}
{
add.f16x2 r1367, %55, %92;
}
{
add.f16x2 r1370, %62, r1367;
}
{
add.f16x2 r1373, %71, %107;
}
{
add.f16x2 r1376, %77, r1373;
}
{
add.f16x2 r1379, %55, %92;
}
{
mul.f16x2 r1382, r1379, r1365;
}
{
add.f16x2 r1385, %62, r1382;
}
{
sub.f16x2 r1388, %71, %107;
}
{
mul.f16x2 r1391, r1388, r1366;
}
{
add.f16x2 r1394, r1385, r1391;
}
{
add.f16x2 r1397, %55, %92;
}
{
mul.f16x2 r1400, r1397, r1365;
}
{
add.f16x2 r1403, %62, r1400;
}
{
sub.f16x2 r1406, %71, %107;
}
{
mul.f16x2 r1409, r1406, r1366;
}
{
sub.f16x2 r1412, r1403, r1409;
}
{
add.f16x2 r1415, %71, %107;
}
{
mul.f16x2 r1418, r1415, r1365;
}
{
add.f16x2 r1421, %77, r1418;
}
{
sub.f16x2 r1424, %55, %92;
}
{
mul.f16x2 r1427, r1424, r1366;
}
{
sub.f16x2 r1430, r1421, r1427;
}
{
add.f16x2 r1433, %71, %107;
}
{
mul.f16x2 r1436, r1433, r1365;
}
{
add.f16x2 r1439, %77, r1436;
}
{
sub.f16x2 r1442, %55, %92;
}
{
mul.f16x2 r1445, r1442, r1366;
}
{
add.f16x2 r1448, r1439, r1445;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r1451, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
mov.b32 r1452, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r1453, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f192;
cvt.rn.f16.f32 high, f192;
mov.b32 r1454, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1457, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1458, {low, high};
}
{
mul.f16x2 r1467, r1308, r1451;
}
{
mul.f16x2 r1470, r1344, r1452;
}
{
sub.f16x2 r1473, r1467, r1470;
}
{
mul.f16x2 r1476, r1308, r1452;
}
{
fma.rn.f16x2 r1479, r1344, r1451, r1476;
}
{
mul.f16x2 r1483, r1394, r1453;
}
{
mul.f16x2 r1486, r1430, r1454;
}
{
sub.f16x2 r1489, r1483, r1486;
}
{
mul.f16x2 r1492, r1394, r1454;
}
{
fma.rn.f16x2 r1495, r1430, r1453, r1492;
}
{
mul.f16x2 r1499, r1326, r1453;
}
{
mul.f16x2 r1502, r1362, r1454;
}
{
sub.f16x2 r1505, r1499, r1502;
}
{
mul.f16x2 r1508, r1326, r1454;
}
{
fma.rn.f16x2 r1511, r1362, r1453, r1508;
}
{
mul.f16x2 r1515, r1412, r1457;
}
{
mul.f16x2 r1518, r1448, r1458;
}
{
sub.f16x2 r1521, r1515, r1518;
}
{
mul.f16x2 r1524, r1412, r1458;
}
{
fma.rn.f16x2 r1527, r1448, r1457, r1524;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1531, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1532, {low, high};
}
{
add.f16x2 r1533, r1284, r1370;
}
{
add.f16x2 r1536, r1198, r1533;
}
{
add.f16x2 r1539, r1290, r1376;
}
{
add.f16x2 r1542, r1204, r1539;
}
{
add.f16x2 r1545, r1284, r1370;
}
{
mul.f16x2 r1548, r1545, r1531;
}
{
add.f16x2 r1551, r1198, r1548;
}
{
sub.f16x2 r1554, r1290, r1376;
}
{
mul.f16x2 r1557, r1554, r1532;
}
{
add.f16x2 r1560, r1551, r1557;
}
{
add.f16x2 r1563, r1284, r1370;
}
{
mul.f16x2 r1566, r1563, r1531;
}
{
add.f16x2 r1569, r1198, r1566;
}
{
sub.f16x2 r1572, r1290, r1376;
}
{
mul.f16x2 r1575, r1572, r1532;
}
{
sub.f16x2 r1578, r1569, r1575;
}
{
add.f16x2 r1581, r1290, r1376;
}
{
mul.f16x2 r1584, r1581, r1531;
}
{
add.f16x2 r1587, r1204, r1584;
}
{
sub.f16x2 r1590, r1284, r1370;
}
{
mul.f16x2 r1593, r1590, r1532;
}
{
sub.f16x2 r1596, r1587, r1593;
}
{
add.f16x2 r1599, r1290, r1376;
}
{
mul.f16x2 r1602, r1599, r1531;
}
{
add.f16x2 r1605, r1204, r1602;
}
{
sub.f16x2 r1608, r1284, r1370;
}
{
mul.f16x2 r1611, r1608, r1532;
}
{
add.f16x2 r1614, r1605, r1611;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1617, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1618, {low, high};
}
{
add.f16x2 r1619, r1473, r1489;
}
{
add.f16x2 r1622, r1222, r1619;
}
{
add.f16x2 r1625, r1479, r1495;
}
{
add.f16x2 r1628, r1258, r1625;
}
{
add.f16x2 r1631, r1473, r1489;
}
{
mul.f16x2 r1634, r1631, r1617;
}
{
add.f16x2 r1637, r1222, r1634;
}
{
sub.f16x2 r1640, r1479, r1495;
}
{
mul.f16x2 r1643, r1640, r1618;
}
{
add.f16x2 r1646, r1637, r1643;
}
{
add.f16x2 r1649, r1473, r1489;
}
{
mul.f16x2 r1652, r1649, r1617;
}
{
add.f16x2 r1655, r1222, r1652;
}
{
sub.f16x2 r1658, r1479, r1495;
}
{
mul.f16x2 r1661, r1658, r1618;
}
{
sub.f16x2 r1664, r1655, r1661;
}
{
add.f16x2 r1667, r1479, r1495;
}
{
mul.f16x2 r1670, r1667, r1617;
}
{
add.f16x2 r1673, r1258, r1670;
}
{
sub.f16x2 r1676, r1473, r1489;
}
{
mul.f16x2 r1679, r1676, r1618;
}
{
sub.f16x2 r1682, r1673, r1679;
}
{
add.f16x2 r1685, r1479, r1495;
}
{
mul.f16x2 r1688, r1685, r1617;
}
{
add.f16x2 r1691, r1258, r1688;
}
{
sub.f16x2 r1694, r1473, r1489;
}
{
mul.f16x2 r1697, r1694, r1618;
}
{
add.f16x2 r1700, r1691, r1697;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1703, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1704, {low, high};
}
{
add.f16x2 r1705, r1505, r1521;
}
{
add.f16x2 r1708, r1240, r1705;
}
{
add.f16x2 r1711, r1511, r1527;
}
{
add.f16x2 r1714, r1276, r1711;
}
{
add.f16x2 r1717, r1505, r1521;
}
{
mul.f16x2 r1720, r1717, r1703;
}
{
add.f16x2 r1723, r1240, r1720;
}
{
sub.f16x2 r1726, r1511, r1527;
}
{
mul.f16x2 r1729, r1726, r1704;
}
{
add.f16x2 r1732, r1723, r1729;
}
{
add.f16x2 r1735, r1505, r1521;
}
{
mul.f16x2 r1738, r1735, r1703;
}
{
add.f16x2 r1741, r1240, r1738;
}
{
sub.f16x2 r1744, r1511, r1527;
}
{
mul.f16x2 r1747, r1744, r1704;
}
{
sub.f16x2 r1750, r1741, r1747;
}
{
add.f16x2 r1753, r1511, r1527;
}
{
mul.f16x2 r1756, r1753, r1703;
}
{
add.f16x2 r1759, r1276, r1756;
}
{
sub.f16x2 r1762, r1505, r1521;
}
{
mul.f16x2 r1765, r1762, r1704;
}
{
sub.f16x2 r1768, r1759, r1765;
}
{
add.f16x2 r1771, r1511, r1527;
}
{
mul.f16x2 r1774, r1771, r1703;
}
{
add.f16x2 r1777, r1276, r1774;
}
{
sub.f16x2 r1780, r1505, r1521;
}
{
mul.f16x2 r1783, r1780, r1704;
}
{
add.f16x2 r1786, r1777, r1783;
}
mov.f32 f170, 0f3F791978;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f170;
cvt.rn.f16.f32 high, f170;
mov.b32 r1789, {low, high};
}
mov.f32 f172, 0f3E6C2691;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f172;
cvt.rn.f16.f32 high, f172;
mov.b32 r1790, {low, high};
}
mov.f32 f174, 0f3F64C51C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f174;
cvt.rn.f16.f32 high, f174;
mov.b32 r1791, {low, high};
}
mov.f32 f176, 0f3EE5C902;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f176;
cvt.rn.f16.f32 high, f176;
mov.b32 r1792, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r1793, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
mov.b32 r1794, {low, high};
}
mov.f32 f182, 0f3F18DF63;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f182;
cvt.rn.f16.f32 high, f182;
mov.b32 r1795, {low, high};
}
mov.f32 f184, 0f3F4D57F2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f184;
cvt.rn.f16.f32 high, f184;
mov.b32 r1796, {low, high};
}
mov.f32 f186, 0f3ECACAF8;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f186;
cvt.rn.f16.f32 high, f186;
mov.b32 r1797, {low, high};
}
mov.f32 f188, 0f3F6B1036;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r1798, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r1799, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f192;
cvt.rn.f16.f32 high, f192;
mov.b32 r1800, {low, high};
}
mov.f32 f194, 0fBD6E2946;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1801, {low, high};
}
mov.f32 f196, 0f3F7F9120;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r1802, {low, high};
}
mov.f32 f198, 0fBE92D7E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1803, {low, high};
}
mov.f32 f200, 0f3F753ECD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1804, {low, high};
}
mov.f32 f206, 0fBF2FAD88;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f206;
cvt.rn.f16.f32 high, f206;
mov.b32 r1807, {low, high};
}
mov.f32 f208, 0f3F3A3529;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f208;
cvt.rn.f16.f32 high, f208;
mov.b32 r1808, {low, high};
}
mov.f32 f230, 0fBF55E287;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1811, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1812, {low, high};
}
mov.f32 f222, 0fBF7E44DE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f222;
cvt.rn.f16.f32 high, f222;
mov.b32 r1815, {low, high};
}
mov.f32 f224, 0fBDEDC21F;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f224;
cvt.rn.f16.f32 high, f224;
mov.b32 r1816, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r1819, {low, high};
}
mov.f32 f232, 0fBF0CAC9F;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f232;
cvt.rn.f16.f32 high, f232;
mov.b32 r1820, {low, high};
}
{
mul.f16x2 r1841, r1026, r1789;
}
{
mul.f16x2 r1844, r1032, r1790;
}
{
sub.f16x2 r1847, r1841, r1844;
}
{
mul.f16x2 r1850, r1026, r1790;
}
{
fma.rn.f16x2 r1853, r1032, r1789, r1850;
}
{
mul.f16x2 r1857, r1622, r1791;
}
{
mul.f16x2 r1860, r1628, r1792;
}
{
sub.f16x2 r1863, r1857, r1860;
}
{
mul.f16x2 r1866, r1622, r1792;
}
{
fma.rn.f16x2 r1869, r1628, r1791, r1866;
}
{
mul.f16x2 r1873, r1112, r1791;
}
{
mul.f16x2 r1876, r1118, r1792;
}
{
sub.f16x2 r1879, r1873, r1876;
}
{
mul.f16x2 r1882, r1112, r1792;
}
{
fma.rn.f16x2 r1885, r1118, r1791, r1882;
}
{
mul.f16x2 r1889, r1708, r1795;
}
{
mul.f16x2 r1892, r1714, r1796;
}
{
sub.f16x2 r1895, r1889, r1892;
}
{
mul.f16x2 r1898, r1708, r1796;
}
{
fma.rn.f16x2 r1901, r1714, r1795, r1898;
}
{
mul.f16x2 r1905, r964, r1793;
}
{
mul.f16x2 r1908, r1000, r1794;
}
{
sub.f16x2 r1911, r1905, r1908;
}
{
mul.f16x2 r1914, r964, r1794;
}
{
fma.rn.f16x2 r1917, r1000, r1793, r1914;
}
{
mul.f16x2 r1921, r1560, r1799;
}
{
mul.f16x2 r1924, r1596, r1800;
}
{
sub.f16x2 r1927, r1921, r1924;
}
{
mul.f16x2 r1930, r1560, r1800;
}
{
fma.rn.f16x2 r1933, r1596, r1799, r1930;
}
{
mul.f16x2 r1937, r1050, r1795;
}
{
mul.f16x2 r1940, r1086, r1796;
}
{
sub.f16x2 r1943, r1937, r1940;
}
{
mul.f16x2 r1946, r1050, r1796;
}
{
fma.rn.f16x2 r1949, r1086, r1795, r1946;
}
{
mul.f16x2 r1953, r1646, r1803;
}
{
mul.f16x2 r1956, r1682, r1804;
}
{
sub.f16x2 r1959, r1953, r1956;
}
{
mul.f16x2 r1962, r1646, r1804;
}
{
fma.rn.f16x2 r1965, r1682, r1803, r1962;
}
{
mul.f16x2 r1969, r1136, r1797;
}
{
mul.f16x2 r1972, r1172, r1798;
}
{
sub.f16x2 r1975, r1969, r1972;
}
{
mul.f16x2 r1978, r1136, r1798;
}
{
fma.rn.f16x2 r1981, r1172, r1797, r1978;
}
{
mul.f16x2 r1985, r1732, r1807;
}
{
mul.f16x2 r1988, r1768, r1808;
}
{
sub.f16x2 r1991, r1985, r1988;
}
{
mul.f16x2 r1994, r1732, r1808;
}
{
fma.rn.f16x2 r1997, r1768, r1807, r1994;
}
{
mul.f16x2 r2001, r982, r1799;
}
{
mul.f16x2 r2004, r1018, r1800;
}
{
sub.f16x2 r2007, r2001, r2004;
}
{
mul.f16x2 r2010, r982, r1800;
}
{
fma.rn.f16x2 r2013, r1018, r1799, r2010;
}
{
mul.f16x2 r2017, r1578, r1811;
}
{
mul.f16x2 r2020, r1614, r1812;
}
{
sub.f16x2 r2023, r2017, r2020;
}
{
mul.f16x2 r2026, r1578, r1812;
}
{
fma.rn.f16x2 r2029, r1614, r1811, r2026;
}
{
mul.f16x2 r2033, r1068, r1801;
}
{
mul.f16x2 r2036, r1104, r1802;
}
{
sub.f16x2 r2039, r2033, r2036;
}
{
mul.f16x2 r2042, r1068, r1802;
}
{
fma.rn.f16x2 r2045, r1104, r1801, r2042;
}
{
mul.f16x2 r2049, r1664, r1815;
}
{
mul.f16x2 r2052, r1700, r1816;
}
{
sub.f16x2 r2055, r2049, r2052;
}
{
mul.f16x2 r2058, r1664, r1816;
}
{
fma.rn.f16x2 r2061, r1700, r1815, r2058;
}
{
mul.f16x2 r2065, r1154, r1803;
}
{
mul.f16x2 r2068, r1190, r1804;
}
{
sub.f16x2 r2071, r2065, r2068;
}
{
mul.f16x2 r2074, r1154, r1804;
}
{
fma.rn.f16x2 r2077, r1190, r1803, r2074;
}
{
mul.f16x2 r2081, r1750, r1819;
}
{
mul.f16x2 r2084, r1786, r1820;
}
{
sub.f16x2 r2087, r2081, r2084;
}
{
mul.f16x2 r2090, r1750, r1820;
}
{
fma.rn.f16x2 r2093, r1786, r1819, r2090;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2097, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2098, {low, high};
}
{
add.f16x2 r2099, r940, r1536;
}
{
add.f16x2 r2102, r344, r2099;
}
{
add.f16x2 r2105, r946, r1542;
}
{
add.f16x2 r2108, r350, r2105;
}
{
add.f16x2 r2111, r940, r1536;
}
{
mul.f16x2 r2114, r2111, r2097;
}
{
add.f16x2 r2117, r344, r2114;
}
{
sub.f16x2 r2120, r946, r1542;
}
{
mul.f16x2 r2123, r2120, r2098;
}
{
add.f16x2 r2126, r2117, r2123;
}
{
add.f16x2 r2129, r940, r1536;
}
{
mul.f16x2 r2132, r2129, r2097;
}
{
add.f16x2 r2135, r344, r2132;
}
{
sub.f16x2 r2138, r946, r1542;
}
{
mul.f16x2 r2141, r2138, r2098;
}
{
sub.f16x2 r2144, r2135, r2141;
}
{
add.f16x2 r2147, r946, r1542;
}
{
mul.f16x2 r2150, r2147, r2097;
}
{
add.f16x2 r2153, r350, r2150;
}
{
sub.f16x2 r2156, r940, r1536;
}
{
mul.f16x2 r2159, r2156, r2098;
}
{
sub.f16x2 r2162, r2153, r2159;
}
{
add.f16x2 r2165, r946, r1542;
}
{
mul.f16x2 r2168, r2165, r2097;
}
{
add.f16x2 r2171, r350, r2168;
}
{
sub.f16x2 r2174, r940, r1536;
}
{
mul.f16x2 r2177, r2174, r2098;
}
{
add.f16x2 r2180, r2171, r2177;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2183, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2184, {low, high};
}
{
add.f16x2 r2185, r1847, r1863;
}
{
add.f16x2 r2188, r430, r2185;
}
{
add.f16x2 r2191, r1853, r1869;
}
{
add.f16x2 r2194, r436, r2191;
}
{
add.f16x2 r2197, r1847, r1863;
}
{
mul.f16x2 r2200, r2197, r2183;
}
{
add.f16x2 r2203, r430, r2200;
}
{
sub.f16x2 r2206, r1853, r1869;
}
{
mul.f16x2 r2209, r2206, r2184;
}
{
add.f16x2 r2212, r2203, r2209;
}
{
add.f16x2 r2215, r1847, r1863;
}
{
mul.f16x2 r2218, r2215, r2183;
}
{
add.f16x2 r2221, r430, r2218;
}
{
sub.f16x2 r2224, r1853, r1869;
}
{
mul.f16x2 r2227, r2224, r2184;
}
{
sub.f16x2 r2230, r2221, r2227;
}
{
add.f16x2 r2233, r1853, r1869;
}
{
mul.f16x2 r2236, r2233, r2183;
}
{
add.f16x2 r2239, r436, r2236;
}
{
sub.f16x2 r2242, r1847, r1863;
}
{
mul.f16x2 r2245, r2242, r2184;
}
{
sub.f16x2 r2248, r2239, r2245;
}
{
add.f16x2 r2251, r1853, r1869;
}
{
mul.f16x2 r2254, r2251, r2183;
}
{
add.f16x2 r2257, r436, r2254;
}
{
sub.f16x2 r2260, r1847, r1863;
}
{
mul.f16x2 r2263, r2260, r2184;
}
{
add.f16x2 r2266, r2257, r2263;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2269, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2270, {low, high};
}
{
add.f16x2 r2271, r1879, r1895;
}
{
add.f16x2 r2274, r516, r2271;
}
{
add.f16x2 r2277, r1885, r1901;
}
{
add.f16x2 r2280, r522, r2277;
}
{
add.f16x2 r2283, r1879, r1895;
}
{
mul.f16x2 r2286, r2283, r2269;
}
{
add.f16x2 r2289, r516, r2286;
}
{
sub.f16x2 r2292, r1885, r1901;
}
{
mul.f16x2 r2295, r2292, r2270;
}
{
add.f16x2 r2298, r2289, r2295;
}
{
add.f16x2 r2301, r1879, r1895;
}
{
mul.f16x2 r2304, r2301, r2269;
}
{
add.f16x2 r2307, r516, r2304;
}
{
sub.f16x2 r2310, r1885, r1901;
}
{
mul.f16x2 r2313, r2310, r2270;
}
{
sub.f16x2 r2316, r2307, r2313;
}
{
add.f16x2 r2319, r1885, r1901;
}
{
mul.f16x2 r2322, r2319, r2269;
}
{
add.f16x2 r2325, r522, r2322;
}
{
sub.f16x2 r2328, r1879, r1895;
}
{
mul.f16x2 r2331, r2328, r2270;
}
{
sub.f16x2 r2334, r2325, r2331;
}
{
add.f16x2 r2337, r1885, r1901;
}
{
mul.f16x2 r2340, r2337, r2269;
}
{
add.f16x2 r2343, r522, r2340;
}
{
sub.f16x2 r2346, r1879, r1895;
}
{
mul.f16x2 r2349, r2346, r2270;
}
{
add.f16x2 r2352, r2343, r2349;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2355, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2356, {low, high};
}
{
add.f16x2 r2357, r1911, r1927;
}
{
add.f16x2 r2360, r368, r2357;
}
{
add.f16x2 r2363, r1917, r1933;
}
{
add.f16x2 r2366, r404, r2363;
}
{
add.f16x2 r2369, r1911, r1927;
}
{
mul.f16x2 r2372, r2369, r2355;
}
{
add.f16x2 r2375, r368, r2372;
}
{
sub.f16x2 r2378, r1917, r1933;
}
{
mul.f16x2 r2381, r2378, r2356;
}
{
add.f16x2 r2384, r2375, r2381;
}
{
add.f16x2 r2387, r1911, r1927;
}
{
mul.f16x2 r2390, r2387, r2355;
}
{
add.f16x2 r2393, r368, r2390;
}
{
sub.f16x2 r2396, r1917, r1933;
}
{
mul.f16x2 r2399, r2396, r2356;
}
{
sub.f16x2 r2402, r2393, r2399;
}
{
add.f16x2 r2405, r1917, r1933;
}
{
mul.f16x2 r2408, r2405, r2355;
}
{
add.f16x2 r2411, r404, r2408;
}
{
sub.f16x2 r2414, r1911, r1927;
}
{
mul.f16x2 r2417, r2414, r2356;
}
{
sub.f16x2 r2420, r2411, r2417;
}
{
add.f16x2 r2423, r1917, r1933;
}
{
mul.f16x2 r2426, r2423, r2355;
}
{
add.f16x2 r2429, r404, r2426;
}
{
sub.f16x2 r2432, r1911, r1927;
}
{
mul.f16x2 r2435, r2432, r2356;
}
{
add.f16x2 r2438, r2429, r2435;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2441, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2442, {low, high};
}
{
add.f16x2 r2443, r1943, r1959;
}
{
add.f16x2 r2446, r454, r2443;
}
{
add.f16x2 r2449, r1949, r1965;
}
{
add.f16x2 r2452, r490, r2449;
}
{
add.f16x2 r2455, r1943, r1959;
}
{
mul.f16x2 r2458, r2455, r2441;
}
{
add.f16x2 r2461, r454, r2458;
}
{
sub.f16x2 r2464, r1949, r1965;
}
{
mul.f16x2 r2467, r2464, r2442;
}
{
add.f16x2 r2470, r2461, r2467;
}
{
add.f16x2 r2473, r1943, r1959;
}
{
mul.f16x2 r2476, r2473, r2441;
}
{
add.f16x2 r2479, r454, r2476;
}
{
sub.f16x2 r2482, r1949, r1965;
}
{
mul.f16x2 r2485, r2482, r2442;
}
{
sub.f16x2 r2488, r2479, r2485;
}
{
add.f16x2 r2491, r1949, r1965;
}
{
mul.f16x2 r2494, r2491, r2441;
}
{
add.f16x2 r2497, r490, r2494;
}
{
sub.f16x2 r2500, r1943, r1959;
}
{
mul.f16x2 r2503, r2500, r2442;
}
{
sub.f16x2 r2506, r2497, r2503;
}
{
add.f16x2 r2509, r1949, r1965;
}
{
mul.f16x2 r2512, r2509, r2441;
}
{
add.f16x2 r2515, r490, r2512;
}
{
sub.f16x2 r2518, r1943, r1959;
}
{
mul.f16x2 r2521, r2518, r2442;
}
{
add.f16x2 r2524, r2515, r2521;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2527, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2528, {low, high};
}
{
add.f16x2 r2529, r1975, r1991;
}
{
add.f16x2 r2532, r540, r2529;
}
{
add.f16x2 r2535, r1981, r1997;
}
{
add.f16x2 r2538, r576, r2535;
}
{
add.f16x2 r2541, r1975, r1991;
}
{
mul.f16x2 r2544, r2541, r2527;
}
{
add.f16x2 r2547, r540, r2544;
}
{
sub.f16x2 r2550, r1981, r1997;
}
{
mul.f16x2 r2553, r2550, r2528;
}
{
add.f16x2 r2556, r2547, r2553;
}
{
add.f16x2 r2559, r1975, r1991;
}
{
mul.f16x2 r2562, r2559, r2527;
}
{
add.f16x2 r2565, r540, r2562;
}
{
sub.f16x2 r2568, r1981, r1997;
}
{
mul.f16x2 r2571, r2568, r2528;
}
{
sub.f16x2 r2574, r2565, r2571;
}
{
add.f16x2 r2577, r1981, r1997;
}
{
mul.f16x2 r2580, r2577, r2527;
}
{
add.f16x2 r2583, r576, r2580;
}
{
sub.f16x2 r2586, r1975, r1991;
}
{
mul.f16x2 r2589, r2586, r2528;
}
{
sub.f16x2 r2592, r2583, r2589;
}
{
add.f16x2 r2595, r1981, r1997;
}
{
mul.f16x2 r2598, r2595, r2527;
}
{
add.f16x2 r2601, r576, r2598;
}
{
sub.f16x2 r2604, r1975, r1991;
}
{
mul.f16x2 r2607, r2604, r2528;
}
{
add.f16x2 r2610, r2601, r2607;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2613, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2614, {low, high};
}
{
add.f16x2 r2615, r2007, r2023;
}
{
add.f16x2 r2618, r386, r2615;
}
{
add.f16x2 r2621, r2013, r2029;
}
{
add.f16x2 r2624, r422, r2621;
}
{
add.f16x2 r2627, r2007, r2023;
}
{
mul.f16x2 r2630, r2627, r2613;
}
{
add.f16x2 r2633, r386, r2630;
}
{
sub.f16x2 r2636, r2013, r2029;
}
{
mul.f16x2 r2639, r2636, r2614;
}
{
add.f16x2 r2642, r2633, r2639;
}
{
add.f16x2 r2645, r2007, r2023;
}
{
mul.f16x2 r2648, r2645, r2613;
}
{
add.f16x2 r2651, r386, r2648;
}
{
sub.f16x2 r2654, r2013, r2029;
}
{
mul.f16x2 r2657, r2654, r2614;
}
{
sub.f16x2 r2660, r2651, r2657;
}
{
add.f16x2 r2663, r2013, r2029;
}
{
mul.f16x2 r2666, r2663, r2613;
}
{
add.f16x2 r2669, r422, r2666;
}
{
sub.f16x2 r2672, r2007, r2023;
}
{
mul.f16x2 r2675, r2672, r2614;
}
{
sub.f16x2 r2678, r2669, r2675;
}
{
add.f16x2 r2681, r2013, r2029;
}
{
mul.f16x2 r2684, r2681, r2613;
}
{
add.f16x2 r2687, r422, r2684;
}
{
sub.f16x2 r2690, r2007, r2023;
}
{
mul.f16x2 r2693, r2690, r2614;
}
{
add.f16x2 r2696, r2687, r2693;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2699, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2700, {low, high};
}
{
add.f16x2 r2701, r2039, r2055;
}
{
add.f16x2 r2704, r472, r2701;
}
{
add.f16x2 r2707, r2045, r2061;
}
{
add.f16x2 r2710, r508, r2707;
}
{
add.f16x2 r2713, r2039, r2055;
}
{
mul.f16x2 r2716, r2713, r2699;
}
{
add.f16x2 r2719, r472, r2716;
}
{
sub.f16x2 r2722, r2045, r2061;
}
{
mul.f16x2 r2725, r2722, r2700;
}
{
add.f16x2 r2728, r2719, r2725;
}
{
add.f16x2 r2731, r2039, r2055;
}
{
mul.f16x2 r2734, r2731, r2699;
}
{
add.f16x2 r2737, r472, r2734;
}
{
sub.f16x2 r2740, r2045, r2061;
}
{
mul.f16x2 r2743, r2740, r2700;
}
{
sub.f16x2 r2746, r2737, r2743;
}
{
add.f16x2 r2749, r2045, r2061;
}
{
mul.f16x2 r2752, r2749, r2699;
}
{
add.f16x2 r2755, r508, r2752;
}
{
sub.f16x2 r2758, r2039, r2055;
}
{
mul.f16x2 r2761, r2758, r2700;
}
{
sub.f16x2 r2764, r2755, r2761;
}
{
add.f16x2 r2767, r2045, r2061;
}
{
mul.f16x2 r2770, r2767, r2699;
}
{
add.f16x2 r2773, r508, r2770;
}
{
sub.f16x2 r2776, r2039, r2055;
}
{
mul.f16x2 r2779, r2776, r2700;
}
{
add.f16x2 r2782, r2773, r2779;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2785, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2786, {low, high};
}
{
add.f16x2 r2787, r2071, r2087;
}
{
add.f16x2 r2790, r558, r2787;
}
{
add.f16x2 r2793, r2077, r2093;
}
{
add.f16x2 r2796, r594, r2793;
}
{
add.f16x2 r2799, r2071, r2087;
}
{
mul.f16x2 r2802, r2799, r2785;
}
{
add.f16x2 r2805, r558, r2802;
}
{
sub.f16x2 r2808, r2077, r2093;
}
{
mul.f16x2 r2811, r2808, r2786;
}
{
add.f16x2 r2814, r2805, r2811;
}
{
add.f16x2 r2817, r2071, r2087;
}
{
mul.f16x2 r2820, r2817, r2785;
}
{
add.f16x2 r2823, r558, r2820;
}
{
sub.f16x2 r2826, r2077, r2093;
}
{
mul.f16x2 r2829, r2826, r2786;
}
{
sub.f16x2 r2832, r2823, r2829;
}
{
add.f16x2 r2835, r2077, r2093;
}
{
mul.f16x2 r2838, r2835, r2785;
}
{
add.f16x2 r2841, r594, r2838;
}
{
sub.f16x2 r2844, r2071, r2087;
}
{
mul.f16x2 r2847, r2844, r2786;
}
{
sub.f16x2 r2850, r2841, r2847;
}
{
add.f16x2 r2853, r2077, r2093;
}
{
mul.f16x2 r2856, r2853, r2785;
}
{
add.f16x2 r2859, r594, r2856;
}
{
sub.f16x2 r2862, r2071, r2087;
}
{
mul.f16x2 r2865, r2862, r2786;
}
{
add.f16x2 r2868, r2859, r2865;
}
mul.wide.u32 rd2, r4613, -1431655765;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r4614, rd3;
mul.lo.s32 r4615, r4614, 3;
sub.s32 r4616, r4613, r4615;
mad.lo.s32 r4617, r4614, 324, r4612;
cvt.rn.f32.u32 f401, r4616;
mul.f32 f402, f401, 0f3D9EDD1F;
cos.approx.f32 f309, f402;
sin.approx.f32 f403, f402;
neg.f32 f310, f403;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f309;
cvt.rn.f16.f32 high, f310;
mov.b32 r2871, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r2874, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r2876, {high, high};
}
{
mul.f16x2 r2878, r2194, r2876;
}
{
fma.rn.f16x2 r2881, r2188, r2874, r2878;
}
{
mul.f16x2 r2885, r2188, r2876;
}
{
neg.f16x2 r2888, r2885;
}
{
fma.rn.f16x2 r2890, r2194, r2874, r2888;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r2894, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r2896, {high, high};
}
mov.f32 f361, 0fBF800000;
mov.f32 f362, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r2898, {low, high};
}
{
mul.f16x2 r2899, r2896, r2898;
}
{
mul.f16x2 r2902, r2871, r2894;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r2905, {high, low};
}
{
fma.rn.f16x2 r2907, r2899, r2905, r2902;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2907;
mov.b32 r2911, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2907;
mov.b32 r2913, {high, high};
}
{
mul.f16x2 r2915, r2280, r2913;
}
{
fma.rn.f16x2 r2918, r2274, r2911, r2915;
}
{
mul.f16x2 r2922, r2274, r2913;
}
{
neg.f16x2 r2925, r2922;
}
{
fma.rn.f16x2 r2927, r2280, r2911, r2925;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r2931, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r2933, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r2935, {low, high};
}
{
mul.f16x2 r2936, r2933, r2935;
}
{
mul.f16x2 r2939, r2907, r2931;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2907;
mov.b32 r2942, {high, low};
}
{
fma.rn.f16x2 r2944, r2936, r2942, r2939;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2944;
mov.b32 r2948, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2944;
mov.b32 r2950, {high, high};
}
{
mul.f16x2 r2952, r2366, r2950;
}
{
fma.rn.f16x2 r2955, r2360, r2948, r2952;
}
{
mul.f16x2 r2959, r2360, r2950;
}
{
neg.f16x2 r2962, r2959;
}
{
fma.rn.f16x2 r2964, r2366, r2948, r2962;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r2968, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r2970, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r2972, {low, high};
}
{
mul.f16x2 r2973, r2970, r2972;
}
{
mul.f16x2 r2976, r2944, r2968;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2944;
mov.b32 r2979, {high, low};
}
{
fma.rn.f16x2 r2981, r2973, r2979, r2976;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2981;
mov.b32 r2985, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2981;
mov.b32 r2987, {high, high};
}
{
mul.f16x2 r2989, r2452, r2987;
}
{
fma.rn.f16x2 r2992, r2446, r2985, r2989;
}
{
mul.f16x2 r2996, r2446, r2987;
}
{
neg.f16x2 r2999, r2996;
}
{
fma.rn.f16x2 r3001, r2452, r2985, r2999;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3005, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3007, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3009, {low, high};
}
{
mul.f16x2 r3010, r3007, r3009;
}
{
mul.f16x2 r3013, r2981, r3005;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2981;
mov.b32 r3016, {high, low};
}
{
fma.rn.f16x2 r3018, r3010, r3016, r3013;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3018;
mov.b32 r3022, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3018;
mov.b32 r3024, {high, high};
}
{
mul.f16x2 r3026, r2538, r3024;
}
{
fma.rn.f16x2 r3029, r2532, r3022, r3026;
}
{
mul.f16x2 r3033, r2532, r3024;
}
{
neg.f16x2 r3036, r3033;
}
{
fma.rn.f16x2 r3038, r2538, r3022, r3036;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3042, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3044, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3046, {low, high};
}
{
mul.f16x2 r3047, r3044, r3046;
}
{
mul.f16x2 r3050, r3018, r3042;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3018;
mov.b32 r3053, {high, low};
}
{
fma.rn.f16x2 r3055, r3047, r3053, r3050;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3055;
mov.b32 r3059, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3055;
mov.b32 r3061, {high, high};
}
{
mul.f16x2 r3063, r2624, r3061;
}
{
fma.rn.f16x2 r3066, r2618, r3059, r3063;
}
{
mul.f16x2 r3070, r2618, r3061;
}
{
neg.f16x2 r3073, r3070;
}
{
fma.rn.f16x2 r3075, r2624, r3059, r3073;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3079, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3081, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3083, {low, high};
}
{
mul.f16x2 r3084, r3081, r3083;
}
{
mul.f16x2 r3087, r3055, r3079;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3055;
mov.b32 r3090, {high, low};
}
{
fma.rn.f16x2 r3092, r3084, r3090, r3087;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3092;
mov.b32 r3096, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3092;
mov.b32 r3098, {high, high};
}
{
mul.f16x2 r3100, r2710, r3098;
}
{
fma.rn.f16x2 r3103, r2704, r3096, r3100;
}
{
mul.f16x2 r3107, r2704, r3098;
}
{
neg.f16x2 r3110, r3107;
}
{
fma.rn.f16x2 r3112, r2710, r3096, r3110;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3116, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3118, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3120, {low, high};
}
{
mul.f16x2 r3121, r3118, r3120;
}
{
mul.f16x2 r3124, r3092, r3116;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3092;
mov.b32 r3127, {high, low};
}
{
fma.rn.f16x2 r3129, r3121, r3127, r3124;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3129;
mov.b32 r3133, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3129;
mov.b32 r3135, {high, high};
}
{
mul.f16x2 r3137, r2796, r3135;
}
{
fma.rn.f16x2 r3140, r2790, r3133, r3137;
}
{
mul.f16x2 r3144, r2790, r3135;
}
{
neg.f16x2 r3147, r3144;
}
{
fma.rn.f16x2 r3149, r2796, r3133, r3147;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3153, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3155, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3157, {low, high};
}
{
mul.f16x2 r3158, r3155, r3157;
}
{
mul.f16x2 r3161, r3129, r3153;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3129;
mov.b32 r3164, {high, low};
}
{
fma.rn.f16x2 r3166, r3158, r3164, r3161;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3166;
mov.b32 r3170, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3166;
mov.b32 r3172, {high, high};
}
{
mul.f16x2 r3174, r2162, r3172;
}
{
fma.rn.f16x2 r3177, r2126, r3170, r3174;
}
{
mul.f16x2 r3181, r2126, r3172;
}
{
neg.f16x2 r3184, r3181;
}
{
fma.rn.f16x2 r3186, r2162, r3170, r3184;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3190, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3192, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3194, {low, high};
}
{
mul.f16x2 r3195, r3192, r3194;
}
{
mul.f16x2 r3198, r3166, r3190;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3166;
mov.b32 r3201, {high, low};
}
{
fma.rn.f16x2 r3203, r3195, r3201, r3198;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3203;
mov.b32 r3207, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3203;
mov.b32 r3209, {high, high};
}
{
mul.f16x2 r3211, r2248, r3209;
}
{
fma.rn.f16x2 r3214, r2212, r3207, r3211;
}
{
mul.f16x2 r3218, r2212, r3209;
}
{
neg.f16x2 r3221, r3218;
}
{
fma.rn.f16x2 r3223, r2248, r3207, r3221;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3227, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3229, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3231, {low, high};
}
{
mul.f16x2 r3232, r3229, r3231;
}
{
mul.f16x2 r3235, r3203, r3227;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3203;
mov.b32 r3238, {high, low};
}
{
fma.rn.f16x2 r3240, r3232, r3238, r3235;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3240;
mov.b32 r3244, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3240;
mov.b32 r3246, {high, high};
}
{
mul.f16x2 r3248, r2334, r3246;
}
{
fma.rn.f16x2 r3251, r2298, r3244, r3248;
}
{
mul.f16x2 r3255, r2298, r3246;
}
{
neg.f16x2 r3258, r3255;
}
{
fma.rn.f16x2 r3260, r2334, r3244, r3258;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3264, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3266, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3268, {low, high};
}
{
mul.f16x2 r3269, r3266, r3268;
}
{
mul.f16x2 r3272, r3240, r3264;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3240;
mov.b32 r3275, {high, low};
}
{
fma.rn.f16x2 r3277, r3269, r3275, r3272;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3277;
mov.b32 r3281, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3277;
mov.b32 r3283, {high, high};
}
{
mul.f16x2 r3285, r2420, r3283;
}
{
fma.rn.f16x2 r3288, r2384, r3281, r3285;
}
{
mul.f16x2 r3292, r2384, r3283;
}
{
neg.f16x2 r3295, r3292;
}
{
fma.rn.f16x2 r3297, r2420, r3281, r3295;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3301, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3303, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3305, {low, high};
}
{
mul.f16x2 r3306, r3303, r3305;
}
{
mul.f16x2 r3309, r3277, r3301;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3277;
mov.b32 r3312, {high, low};
}
{
fma.rn.f16x2 r3314, r3306, r3312, r3309;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3314;
mov.b32 r3318, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3314;
mov.b32 r3320, {high, high};
}
{
mul.f16x2 r3322, r2506, r3320;
}
{
fma.rn.f16x2 r3325, r2470, r3318, r3322;
}
{
mul.f16x2 r3329, r2470, r3320;
}
{
neg.f16x2 r3332, r3329;
}
{
fma.rn.f16x2 r3334, r2506, r3318, r3332;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3338, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3340, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3342, {low, high};
}
{
mul.f16x2 r3343, r3340, r3342;
}
{
mul.f16x2 r3346, r3314, r3338;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3314;
mov.b32 r3349, {high, low};
}
{
fma.rn.f16x2 r3351, r3343, r3349, r3346;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3351;
mov.b32 r3355, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3351;
mov.b32 r3357, {high, high};
}
{
mul.f16x2 r3359, r2592, r3357;
}
{
fma.rn.f16x2 r3362, r2556, r3355, r3359;
}
{
mul.f16x2 r3366, r2556, r3357;
}
{
neg.f16x2 r3369, r3366;
}
{
fma.rn.f16x2 r3371, r2592, r3355, r3369;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3375, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3377, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3379, {low, high};
}
{
mul.f16x2 r3380, r3377, r3379;
}
{
mul.f16x2 r3383, r3351, r3375;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3351;
mov.b32 r3386, {high, low};
}
{
fma.rn.f16x2 r3388, r3380, r3386, r3383;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3388;
mov.b32 r3392, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3388;
mov.b32 r3394, {high, high};
}
{
mul.f16x2 r3396, r2678, r3394;
}
{
fma.rn.f16x2 r3399, r2642, r3392, r3396;
}
{
mul.f16x2 r3403, r2642, r3394;
}
{
neg.f16x2 r3406, r3403;
}
{
fma.rn.f16x2 r3408, r2678, r3392, r3406;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3412, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3414, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3416, {low, high};
}
{
mul.f16x2 r3417, r3414, r3416;
}
{
mul.f16x2 r3420, r3388, r3412;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3388;
mov.b32 r3423, {high, low};
}
{
fma.rn.f16x2 r3425, r3417, r3423, r3420;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3425;
mov.b32 r3429, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3425;
mov.b32 r3431, {high, high};
}
{
mul.f16x2 r3433, r2764, r3431;
}
{
fma.rn.f16x2 r3436, r2728, r3429, r3433;
}
{
mul.f16x2 r3440, r2728, r3431;
}
{
neg.f16x2 r3443, r3440;
}
{
fma.rn.f16x2 r3445, r2764, r3429, r3443;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3449, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3451, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3453, {low, high};
}
{
mul.f16x2 r3454, r3451, r3453;
}
{
mul.f16x2 r3457, r3425, r3449;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3425;
mov.b32 r3460, {high, low};
}
{
fma.rn.f16x2 r3462, r3454, r3460, r3457;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3462;
mov.b32 r3466, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3462;
mov.b32 r3468, {high, high};
}
{
mul.f16x2 r3470, r2850, r3468;
}
{
fma.rn.f16x2 r3473, r2814, r3466, r3470;
}
{
mul.f16x2 r3477, r2814, r3468;
}
{
neg.f16x2 r3480, r3477;
}
{
fma.rn.f16x2 r3482, r2850, r3466, r3480;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3486, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3488, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3490, {low, high};
}
{
mul.f16x2 r3491, r3488, r3490;
}
{
mul.f16x2 r3494, r3462, r3486;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3462;
mov.b32 r3497, {high, low};
}
{
fma.rn.f16x2 r3499, r3491, r3497, r3494;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3499;
mov.b32 r3503, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3499;
mov.b32 r3505, {high, high};
}
{
mul.f16x2 r3507, r2180, r3505;
}
{
fma.rn.f16x2 r3510, r2144, r3503, r3507;
}
{
mul.f16x2 r3514, r2144, r3505;
}
{
neg.f16x2 r3517, r3514;
}
{
fma.rn.f16x2 r3519, r2180, r3503, r3517;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3523, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3525, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3527, {low, high};
}
{
mul.f16x2 r3528, r3525, r3527;
}
{
mul.f16x2 r3531, r3499, r3523;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3499;
mov.b32 r3534, {high, low};
}
{
fma.rn.f16x2 r3536, r3528, r3534, r3531;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3536;
mov.b32 r3540, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3536;
mov.b32 r3542, {high, high};
}
{
mul.f16x2 r3544, r2266, r3542;
}
{
fma.rn.f16x2 r3547, r2230, r3540, r3544;
}
{
mul.f16x2 r3551, r2230, r3542;
}
{
neg.f16x2 r3554, r3551;
}
{
fma.rn.f16x2 r3556, r2266, r3540, r3554;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3560, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3562, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3564, {low, high};
}
{
mul.f16x2 r3565, r3562, r3564;
}
{
mul.f16x2 r3568, r3536, r3560;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3536;
mov.b32 r3571, {high, low};
}
{
fma.rn.f16x2 r3573, r3565, r3571, r3568;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3573;
mov.b32 r3577, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3573;
mov.b32 r3579, {high, high};
}
{
mul.f16x2 r3581, r2352, r3579;
}
{
fma.rn.f16x2 r3584, r2316, r3577, r3581;
}
{
mul.f16x2 r3588, r2316, r3579;
}
{
neg.f16x2 r3591, r3588;
}
{
fma.rn.f16x2 r3593, r2352, r3577, r3591;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3597, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3599, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3601, {low, high};
}
{
mul.f16x2 r3602, r3599, r3601;
}
{
mul.f16x2 r3605, r3573, r3597;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3573;
mov.b32 r3608, {high, low};
}
{
fma.rn.f16x2 r3610, r3602, r3608, r3605;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3610;
mov.b32 r3614, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3610;
mov.b32 r3616, {high, high};
}
{
mul.f16x2 r3618, r2438, r3616;
}
{
fma.rn.f16x2 r3621, r2402, r3614, r3618;
}
{
mul.f16x2 r3625, r2402, r3616;
}
{
neg.f16x2 r3628, r3625;
}
{
fma.rn.f16x2 r3630, r2438, r3614, r3628;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3634, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3636, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3638, {low, high};
}
{
mul.f16x2 r3639, r3636, r3638;
}
{
mul.f16x2 r3642, r3610, r3634;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3610;
mov.b32 r3645, {high, low};
}
{
fma.rn.f16x2 r3647, r3639, r3645, r3642;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3647;
mov.b32 r3651, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3647;
mov.b32 r3653, {high, high};
}
{
mul.f16x2 r3655, r2524, r3653;
}
{
fma.rn.f16x2 r3658, r2488, r3651, r3655;
}
{
mul.f16x2 r3662, r2488, r3653;
}
{
neg.f16x2 r3665, r3662;
}
{
fma.rn.f16x2 r3667, r2524, r3651, r3665;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3671, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3673, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3675, {low, high};
}
{
mul.f16x2 r3676, r3673, r3675;
}
{
mul.f16x2 r3679, r3647, r3671;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3647;
mov.b32 r3682, {high, low};
}
{
fma.rn.f16x2 r3684, r3676, r3682, r3679;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3684;
mov.b32 r3688, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3684;
mov.b32 r3690, {high, high};
}
{
mul.f16x2 r3692, r2610, r3690;
}
{
fma.rn.f16x2 r3695, r2574, r3688, r3692;
}
{
mul.f16x2 r3699, r2574, r3690;
}
{
neg.f16x2 r3702, r3699;
}
{
fma.rn.f16x2 r3704, r2610, r3688, r3702;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3708, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3710, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3712, {low, high};
}
{
mul.f16x2 r3713, r3710, r3712;
}
{
mul.f16x2 r3716, r3684, r3708;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3684;
mov.b32 r3719, {high, low};
}
{
fma.rn.f16x2 r3721, r3713, r3719, r3716;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3721;
mov.b32 r3725, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3721;
mov.b32 r3727, {high, high};
}
{
mul.f16x2 r3729, r2696, r3727;
}
{
fma.rn.f16x2 r3732, r2660, r3725, r3729;
}
{
mul.f16x2 r3736, r2660, r3727;
}
{
neg.f16x2 r3739, r3736;
}
{
fma.rn.f16x2 r3741, r2696, r3725, r3739;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3745, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3747, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3749, {low, high};
}
{
mul.f16x2 r3750, r3747, r3749;
}
{
mul.f16x2 r3753, r3721, r3745;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3721;
mov.b32 r3756, {high, low};
}
{
fma.rn.f16x2 r3758, r3750, r3756, r3753;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3758;
mov.b32 r3762, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3758;
mov.b32 r3764, {high, high};
}
{
mul.f16x2 r3766, r2782, r3764;
}
{
fma.rn.f16x2 r3769, r2746, r3762, r3766;
}
{
mul.f16x2 r3773, r2746, r3764;
}
{
neg.f16x2 r3776, r3773;
}
{
fma.rn.f16x2 r3778, r2782, r3762, r3776;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3782, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2871;
mov.b32 r3784, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3786, {low, high};
}
{
mul.f16x2 r3787, r3784, r3786;
}
{
mul.f16x2 r3790, r3758, r3782;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3758;
mov.b32 r3793, {high, low};
}
{
fma.rn.f16x2 r3795, r3787, r3793, r3790;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3795;
mov.b32 r3799, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3795;
mov.b32 r3801, {high, high};
}
{
mul.f16x2 r3803, r2868, r3801;
}
{
fma.rn.f16x2 r3806, r2832, r3799, r3803;
}
{
mul.f16x2 r3810, r2832, r3801;
}
{
neg.f16x2 r3813, r3810;
}
{
fma.rn.f16x2 r3815, r2868, r3799, r3813;
}
barrier.sync 0;
mad.lo.s32 r4618, r4616, 108, r4617;
st.shared.u32 [r4618], r2102;
st.shared.u32 [r4618+4], r2881;
st.shared.u32 [r4618+8], r2918;
st.shared.u32 [r4618+12], r2955;
st.shared.u32 [r4618+16], r2992;
st.shared.u32 [r4618+20], r3029;
st.shared.u32 [r4618+24], r3066;
st.shared.u32 [r4618+28], r3103;
st.shared.u32 [r4618+32], r3140;
st.shared.u32 [r4618+36], r3177;
st.shared.u32 [r4618+40], r3214;
st.shared.u32 [r4618+44], r3251;
st.shared.u32 [r4618+48], r3288;
st.shared.u32 [r4618+52], r3325;
st.shared.u32 [r4618+56], r3362;
st.shared.u32 [r4618+60], r3399;
st.shared.u32 [r4618+64], r3436;
st.shared.u32 [r4618+68], r3473;
st.shared.u32 [r4618+72], r3510;
st.shared.u32 [r4618+76], r3547;
st.shared.u32 [r4618+80], r3584;
st.shared.u32 [r4618+84], r3621;
st.shared.u32 [r4618+88], r3658;
st.shared.u32 [r4618+92], r3695;
st.shared.u32 [r4618+96], r3732;
st.shared.u32 [r4618+100], r3769;
st.shared.u32 [r4618+104], r3806;
barrier.sync 0;
mad.lo.s32 r4619, r4616, -104, r4618;
ld.shared.u32 r3842, [r4619];
ld.shared.u32 r3928, [r4619+12];
ld.shared.u32 r4014, [r4619+24];
ld.shared.u32 r4100, [r4619+36];
ld.shared.u32 r4186, [r4619+48];
ld.shared.u32 r4272, [r4619+60];
ld.shared.u32 r4358, [r4619+72];
ld.shared.u32 r4444, [r4619+84];
ld.shared.u32 r4530, [r4619+96];
ld.shared.u32 r3839, [r4619+108];
ld.shared.u32 r3925, [r4619+120];
ld.shared.u32 r4011, [r4619+132];
ld.shared.u32 r4097, [r4619+144];
ld.shared.u32 r4183, [r4619+156];
ld.shared.u32 r4269, [r4619+168];
ld.shared.u32 r4355, [r4619+180];
ld.shared.u32 r4441, [r4619+192];
ld.shared.u32 r4527, [r4619+204];
ld.shared.u32 r3840, [r4619+216];
ld.shared.u32 r3926, [r4619+228];
ld.shared.u32 r4012, [r4619+240];
ld.shared.u32 r4098, [r4619+252];
ld.shared.u32 r4184, [r4619+264];
ld.shared.u32 r4270, [r4619+276];
ld.shared.u32 r4356, [r4619+288];
ld.shared.u32 r4442, [r4619+300];
ld.shared.u32 r4528, [r4619+312];
barrier.sync 0;
st.shared.u32 [r4618], r2108;
st.shared.u32 [r4618+4], r2890;
st.shared.u32 [r4618+8], r2927;
st.shared.u32 [r4618+12], r2964;
st.shared.u32 [r4618+16], r3001;
st.shared.u32 [r4618+20], r3038;
st.shared.u32 [r4618+24], r3075;
st.shared.u32 [r4618+28], r3112;
st.shared.u32 [r4618+32], r3149;
st.shared.u32 [r4618+36], r3186;
st.shared.u32 [r4618+40], r3223;
st.shared.u32 [r4618+44], r3260;
st.shared.u32 [r4618+48], r3297;
st.shared.u32 [r4618+52], r3334;
st.shared.u32 [r4618+56], r3371;
st.shared.u32 [r4618+60], r3408;
st.shared.u32 [r4618+64], r3445;
st.shared.u32 [r4618+68], r3482;
st.shared.u32 [r4618+72], r3519;
st.shared.u32 [r4618+76], r3556;
st.shared.u32 [r4618+80], r3593;
st.shared.u32 [r4618+84], r3630;
st.shared.u32 [r4618+88], r3667;
st.shared.u32 [r4618+92], r3704;
st.shared.u32 [r4618+96], r3741;
st.shared.u32 [r4618+100], r3778;
st.shared.u32 [r4618+104], r3815;
barrier.sync 0;
ld.shared.u32 r3848, [r4619];
ld.shared.u32 r3934, [r4619+12];
ld.shared.u32 r4020, [r4619+24];
ld.shared.u32 r4106, [r4619+36];
ld.shared.u32 r4192, [r4619+48];
ld.shared.u32 r4278, [r4619+60];
ld.shared.u32 r4364, [r4619+72];
ld.shared.u32 r4450, [r4619+84];
ld.shared.u32 r4536, [r4619+96];
ld.shared.u32 r3845, [r4619+108];
ld.shared.u32 r3931, [r4619+120];
ld.shared.u32 r4017, [r4619+132];
ld.shared.u32 r4103, [r4619+144];
ld.shared.u32 r4189, [r4619+156];
ld.shared.u32 r4275, [r4619+168];
ld.shared.u32 r4361, [r4619+180];
ld.shared.u32 r4447, [r4619+192];
ld.shared.u32 r4533, [r4619+204];
ld.shared.u32 r3846, [r4619+216];
ld.shared.u32 r3932, [r4619+228];
ld.shared.u32 r4018, [r4619+240];
ld.shared.u32 r4104, [r4619+252];
ld.shared.u32 r4190, [r4619+264];
ld.shared.u32 r4276, [r4619+276];
ld.shared.u32 r4362, [r4619+288];
ld.shared.u32 r4448, [r4619+300];
ld.shared.u32 r4534, [r4619+312];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r3836, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r3837, {low, high};
}
{
add.f16x2 r3838, r3839, r3840;
}
{
add.f16x2 %0, r3842, r3838;
}
{
add.f16x2 r3844, r3845, r3846;
}
{
add.f16x2 %1, r3848, r3844;
}
{
add.f16x2 r3850, r3839, r3840;
}
{
mul.f16x2 r3853, r3850, r3836;
}
{
add.f16x2 r3856, r3842, r3853;
}
{
sub.f16x2 r3859, r3845, r3846;
}
{
mul.f16x2 r3862, r3859, r3837;
}
{
add.f16x2 %18, r3856, r3862;
}
{
add.f16x2 r3868, r3839, r3840;
}
{
mul.f16x2 r3871, r3868, r3836;
}
{
add.f16x2 r3874, r3842, r3871;
}
{
sub.f16x2 r3877, r3845, r3846;
}
{
mul.f16x2 r3880, r3877, r3837;
}
{
sub.f16x2 %36, r3874, r3880;
}
{
add.f16x2 r3886, r3845, r3846;
}
{
mul.f16x2 r3889, r3886, r3836;
}
{
add.f16x2 r3892, r3848, r3889;
}
{
sub.f16x2 r3895, r3839, r3840;
}
{
mul.f16x2 r3898, r3895, r3837;
}
{
sub.f16x2 %19, r3892, r3898;
}
{
add.f16x2 r3904, r3845, r3846;
}
{
mul.f16x2 r3907, r3904, r3836;
}
{
add.f16x2 r3910, r3848, r3907;
}
{
sub.f16x2 r3913, r3839, r3840;
}
{
mul.f16x2 r3916, r3913, r3837;
}
{
add.f16x2 %37, r3910, r3916;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r3922, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r3923, {low, high};
}
{
add.f16x2 r3924, r3925, r3926;
}
{
add.f16x2 %2, r3928, r3924;
}
{
add.f16x2 r3930, r3931, r3932;
}
{
add.f16x2 %3, r3934, r3930;
}
{
add.f16x2 r3936, r3925, r3926;
}
{
mul.f16x2 r3939, r3936, r3922;
}
{
add.f16x2 r3942, r3928, r3939;
}
{
sub.f16x2 r3945, r3931, r3932;
}
{
mul.f16x2 r3948, r3945, r3923;
}
{
add.f16x2 %20, r3942, r3948;
}
{
add.f16x2 r3954, r3925, r3926;
}
{
mul.f16x2 r3957, r3954, r3922;
}
{
add.f16x2 r3960, r3928, r3957;
}
{
sub.f16x2 r3963, r3931, r3932;
}
{
mul.f16x2 r3966, r3963, r3923;
}
{
sub.f16x2 %38, r3960, r3966;
}
{
add.f16x2 r3972, r3931, r3932;
}
{
mul.f16x2 r3975, r3972, r3922;
}
{
add.f16x2 r3978, r3934, r3975;
}
{
sub.f16x2 r3981, r3925, r3926;
}
{
mul.f16x2 r3984, r3981, r3923;
}
{
sub.f16x2 %21, r3978, r3984;
}
{
add.f16x2 r3990, r3931, r3932;
}
{
mul.f16x2 r3993, r3990, r3922;
}
{
add.f16x2 r3996, r3934, r3993;
}
{
sub.f16x2 r3999, r3925, r3926;
}
{
mul.f16x2 r4002, r3999, r3923;
}
{
add.f16x2 %39, r3996, r4002;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4008, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4009, {low, high};
}
{
add.f16x2 r4010, r4011, r4012;
}
{
add.f16x2 %4, r4014, r4010;
}
{
add.f16x2 r4016, r4017, r4018;
}
{
add.f16x2 %5, r4020, r4016;
}
{
add.f16x2 r4022, r4011, r4012;
}
{
mul.f16x2 r4025, r4022, r4008;
}
{
add.f16x2 r4028, r4014, r4025;
}
{
sub.f16x2 r4031, r4017, r4018;
}
{
mul.f16x2 r4034, r4031, r4009;
}
{
add.f16x2 %22, r4028, r4034;
}
{
add.f16x2 r4040, r4011, r4012;
}
{
mul.f16x2 r4043, r4040, r4008;
}
{
add.f16x2 r4046, r4014, r4043;
}
{
sub.f16x2 r4049, r4017, r4018;
}
{
mul.f16x2 r4052, r4049, r4009;
}
{
sub.f16x2 %40, r4046, r4052;
}
{
add.f16x2 r4058, r4017, r4018;
}
{
mul.f16x2 r4061, r4058, r4008;
}
{
add.f16x2 r4064, r4020, r4061;
}
{
sub.f16x2 r4067, r4011, r4012;
}
{
mul.f16x2 r4070, r4067, r4009;
}
{
sub.f16x2 %23, r4064, r4070;
}
{
add.f16x2 r4076, r4017, r4018;
}
{
mul.f16x2 r4079, r4076, r4008;
}
{
add.f16x2 r4082, r4020, r4079;
}
{
sub.f16x2 r4085, r4011, r4012;
}
{
mul.f16x2 r4088, r4085, r4009;
}
{
add.f16x2 %41, r4082, r4088;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4094, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4095, {low, high};
}
{
add.f16x2 r4096, r4097, r4098;
}
{
add.f16x2 %6, r4100, r4096;
}
{
add.f16x2 r4102, r4103, r4104;
}
{
add.f16x2 %7, r4106, r4102;
}
{
add.f16x2 r4108, r4097, r4098;
}
{
mul.f16x2 r4111, r4108, r4094;
}
{
add.f16x2 r4114, r4100, r4111;
}
{
sub.f16x2 r4117, r4103, r4104;
}
{
mul.f16x2 r4120, r4117, r4095;
}
{
add.f16x2 %24, r4114, r4120;
}
{
add.f16x2 r4126, r4097, r4098;
}
{
mul.f16x2 r4129, r4126, r4094;
}
{
add.f16x2 r4132, r4100, r4129;
}
{
sub.f16x2 r4135, r4103, r4104;
}
{
mul.f16x2 r4138, r4135, r4095;
}
{
sub.f16x2 %42, r4132, r4138;
}
{
add.f16x2 r4144, r4103, r4104;
}
{
mul.f16x2 r4147, r4144, r4094;
}
{
add.f16x2 r4150, r4106, r4147;
}
{
sub.f16x2 r4153, r4097, r4098;
}
{
mul.f16x2 r4156, r4153, r4095;
}
{
sub.f16x2 %25, r4150, r4156;
}
{
add.f16x2 r4162, r4103, r4104;
}
{
mul.f16x2 r4165, r4162, r4094;
}
{
add.f16x2 r4168, r4106, r4165;
}
{
sub.f16x2 r4171, r4097, r4098;
}
{
mul.f16x2 r4174, r4171, r4095;
}
{
add.f16x2 %43, r4168, r4174;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4180, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4181, {low, high};
}
{
add.f16x2 r4182, r4183, r4184;
}
{
add.f16x2 %8, r4186, r4182;
}
{
add.f16x2 r4188, r4189, r4190;
}
{
add.f16x2 %9, r4192, r4188;
}
{
add.f16x2 r4194, r4183, r4184;
}
{
mul.f16x2 r4197, r4194, r4180;
}
{
add.f16x2 r4200, r4186, r4197;
}
{
sub.f16x2 r4203, r4189, r4190;
}
{
mul.f16x2 r4206, r4203, r4181;
}
{
add.f16x2 %26, r4200, r4206;
}
{
add.f16x2 r4212, r4183, r4184;
}
{
mul.f16x2 r4215, r4212, r4180;
}
{
add.f16x2 r4218, r4186, r4215;
}
{
sub.f16x2 r4221, r4189, r4190;
}
{
mul.f16x2 r4224, r4221, r4181;
}
{
sub.f16x2 %44, r4218, r4224;
}
{
add.f16x2 r4230, r4189, r4190;
}
{
mul.f16x2 r4233, r4230, r4180;
}
{
add.f16x2 r4236, r4192, r4233;
}
{
sub.f16x2 r4239, r4183, r4184;
}
{
mul.f16x2 r4242, r4239, r4181;
}
{
sub.f16x2 %27, r4236, r4242;
}
{
add.f16x2 r4248, r4189, r4190;
}
{
mul.f16x2 r4251, r4248, r4180;
}
{
add.f16x2 r4254, r4192, r4251;
}
{
sub.f16x2 r4257, r4183, r4184;
}
{
mul.f16x2 r4260, r4257, r4181;
}
{
add.f16x2 %45, r4254, r4260;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4266, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4267, {low, high};
}
{
add.f16x2 r4268, r4269, r4270;
}
{
add.f16x2 %10, r4272, r4268;
}
{
add.f16x2 r4274, r4275, r4276;
}
{
add.f16x2 %11, r4278, r4274;
}
{
add.f16x2 r4280, r4269, r4270;
}
{
mul.f16x2 r4283, r4280, r4266;
}
{
add.f16x2 r4286, r4272, r4283;
}
{
sub.f16x2 r4289, r4275, r4276;
}
{
mul.f16x2 r4292, r4289, r4267;
}
{
add.f16x2 %28, r4286, r4292;
}
{
add.f16x2 r4298, r4269, r4270;
}
{
mul.f16x2 r4301, r4298, r4266;
}
{
add.f16x2 r4304, r4272, r4301;
}
{
sub.f16x2 r4307, r4275, r4276;
}
{
mul.f16x2 r4310, r4307, r4267;
}
{
sub.f16x2 %46, r4304, r4310;
}
{
add.f16x2 r4316, r4275, r4276;
}
{
mul.f16x2 r4319, r4316, r4266;
}
{
add.f16x2 r4322, r4278, r4319;
}
{
sub.f16x2 r4325, r4269, r4270;
}
{
mul.f16x2 r4328, r4325, r4267;
}
{
sub.f16x2 %29, r4322, r4328;
}
{
add.f16x2 r4334, r4275, r4276;
}
{
mul.f16x2 r4337, r4334, r4266;
}
{
add.f16x2 r4340, r4278, r4337;
}
{
sub.f16x2 r4343, r4269, r4270;
}
{
mul.f16x2 r4346, r4343, r4267;
}
{
add.f16x2 %47, r4340, r4346;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4352, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4353, {low, high};
}
{
add.f16x2 r4354, r4355, r4356;
}
{
add.f16x2 %12, r4358, r4354;
}
{
add.f16x2 r4360, r4361, r4362;
}
{
add.f16x2 %13, r4364, r4360;
}
{
add.f16x2 r4366, r4355, r4356;
}
{
mul.f16x2 r4369, r4366, r4352;
}
{
add.f16x2 r4372, r4358, r4369;
}
{
sub.f16x2 r4375, r4361, r4362;
}
{
mul.f16x2 r4378, r4375, r4353;
}
{
add.f16x2 %30, r4372, r4378;
}
{
add.f16x2 r4384, r4355, r4356;
}
{
mul.f16x2 r4387, r4384, r4352;
}
{
add.f16x2 r4390, r4358, r4387;
}
{
sub.f16x2 r4393, r4361, r4362;
}
{
mul.f16x2 r4396, r4393, r4353;
}
{
sub.f16x2 %48, r4390, r4396;
}
{
add.f16x2 r4402, r4361, r4362;
}
{
mul.f16x2 r4405, r4402, r4352;
}
{
add.f16x2 r4408, r4364, r4405;
}
{
sub.f16x2 r4411, r4355, r4356;
}
{
mul.f16x2 r4414, r4411, r4353;
}
{
sub.f16x2 %31, r4408, r4414;
}
{
add.f16x2 r4420, r4361, r4362;
}
{
mul.f16x2 r4423, r4420, r4352;
}
{
add.f16x2 r4426, r4364, r4423;
}
{
sub.f16x2 r4429, r4355, r4356;
}
{
mul.f16x2 r4432, r4429, r4353;
}
{
add.f16x2 %49, r4426, r4432;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4438, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4439, {low, high};
}
{
add.f16x2 r4440, r4441, r4442;
}
{
add.f16x2 %14, r4444, r4440;
}
{
add.f16x2 r4446, r4447, r4448;
}
{
add.f16x2 %15, r4450, r4446;
}
{
add.f16x2 r4452, r4441, r4442;
}
{
mul.f16x2 r4455, r4452, r4438;
}
{
add.f16x2 r4458, r4444, r4455;
}
{
sub.f16x2 r4461, r4447, r4448;
}
{
mul.f16x2 r4464, r4461, r4439;
}
{
add.f16x2 %32, r4458, r4464;
}
{
add.f16x2 r4470, r4441, r4442;
}
{
mul.f16x2 r4473, r4470, r4438;
}
{
add.f16x2 r4476, r4444, r4473;
}
{
sub.f16x2 r4479, r4447, r4448;
}
{
mul.f16x2 r4482, r4479, r4439;
}
{
sub.f16x2 %50, r4476, r4482;
}
{
add.f16x2 r4488, r4447, r4448;
}
{
mul.f16x2 r4491, r4488, r4438;
}
{
add.f16x2 r4494, r4450, r4491;
}
{
sub.f16x2 r4497, r4441, r4442;
}
{
mul.f16x2 r4500, r4497, r4439;
}
{
sub.f16x2 %33, r4494, r4500;
}
{
add.f16x2 r4506, r4447, r4448;
}
{
mul.f16x2 r4509, r4506, r4438;
}
{
add.f16x2 r4512, r4450, r4509;
}
{
sub.f16x2 r4515, r4441, r4442;
}
{
mul.f16x2 r4518, r4515, r4439;
}
{
add.f16x2 %51, r4512, r4518;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4524, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4525, {low, high};
}
{
add.f16x2 r4526, r4527, r4528;
}
{
add.f16x2 %16, r4530, r4526;
}
{
add.f16x2 r4532, r4533, r4534;
}
{
add.f16x2 %17, r4536, r4532;
}
{
add.f16x2 r4538, r4527, r4528;
}
{
mul.f16x2 r4541, r4538, r4524;
}
{
add.f16x2 r4544, r4530, r4541;
}
{
sub.f16x2 r4547, r4533, r4534;
}
{
mul.f16x2 r4550, r4547, r4525;
}
{
add.f16x2 %34, r4544, r4550;
}
{
add.f16x2 r4556, r4527, r4528;
}
{
mul.f16x2 r4559, r4556, r4524;
}
{
add.f16x2 r4562, r4530, r4559;
}
{
sub.f16x2 r4565, r4533, r4534;
}
{
mul.f16x2 r4568, r4565, r4525;
}
{
sub.f16x2 %52, r4562, r4568;
}
{
add.f16x2 r4574, r4533, r4534;
}
{
mul.f16x2 r4577, r4574, r4524;
}
{
add.f16x2 r4580, r4536, r4577;
}
{
sub.f16x2 r4583, r4527, r4528;
}
{
mul.f16x2 r4586, r4583, r4525;
}
{
sub.f16x2 %35, r4580, r4586;
}
{
add.f16x2 r4592, r4533, r4534;
}
{
mul.f16x2 r4595, r4592, r4524;
}
{
add.f16x2 r4598, r4536, r4595;
}
{
sub.f16x2 r4601, r4527, r4528;
}
{
mul.f16x2 r4604, r4601, r4525;
}
{
add.f16x2 %53, r4598, r4604;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[23].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1078, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<50>;
.reg .b32 r<603>;
.reg .b64 rd<8>;
mov.u32 r576, %tid.y;
mov.u32 r577, %6;
mad.lo.s32 r578, r576, 648, r577;
mov.u32 r579, %tid.x;
mov.f32 f38, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f38;
cvt.rn.f16.f32 high, f38;
mov.b32 r1, {low, high};
}
mov.f32 f40, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f40;
cvt.rn.f16.f32 high, f40;
mov.b32 r2, {low, high};
}
{
add.f16x2 r3, %9, %11;
}
{
add.f16x2 r6, %7, r3;
}
{
add.f16x2 r9, %10, %12;
}
{
add.f16x2 r12, %8, r9;
}
{
add.f16x2 r15, %9, %11;
}
{
mul.f16x2 r18, r15, r1;
}
{
add.f16x2 r21, %7, r18;
}
{
sub.f16x2 r24, %10, %12;
}
{
mul.f16x2 r27, r24, r2;
}
{
add.f16x2 r30, r21, r27;
}
{
add.f16x2 r33, %9, %11;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %7, r36;
}
{
sub.f16x2 r42, %10, %12;
}
{
mul.f16x2 r45, r42, r2;
}
{
sub.f16x2 r48, r39, r45;
}
{
add.f16x2 r51, %10, %12;
}
{
mul.f16x2 r54, r51, r1;
}
{
add.f16x2 r57, %8, r54;
}
{
sub.f16x2 r60, %9, %11;
}
{
mul.f16x2 r63, r60, r2;
}
{
sub.f16x2 r66, r57, r63;
}
{
add.f16x2 r69, %10, %12;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %8, r72;
}
{
sub.f16x2 r78, %9, %11;
}
{
mul.f16x2 r81, r78, r2;
}
{
add.f16x2 r84, r75, r81;
}
mul.wide.u32 rd2, r579, 795364315;
shr.u64 rd3, rd2, 32;
cvt.u32.u64 r580, rd3;
sub.s32 r581, r579, r580;
shr.u32 r582, r581, 1;
add.s32 r583, r582, r580;
shr.u32 r584, r583, 4;
mul.lo.s32 r585, r584, 27;
sub.s32 r586, r579, r585;
mad.lo.s32 r587, r584, 648, r578;
cvt.rn.f32.u32 f41, r586;
mul.f32 f42, f41, 0f3D9EDD1F;
cos.approx.f32 f5, f42;
sin.approx.f32 f43, f42;
neg.f32 f6, f43;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f5;
cvt.rn.f16.f32 high, f6;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r90, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r92, {high, high};
}
{
mul.f16x2 r94, r66, r92;
}
{
fma.rn.f16x2 r97, r30, r90, r94;
}
{
mul.f16x2 r101, r30, r92;
}
{
neg.f16x2 r104, r101;
}
{
fma.rn.f16x2 r106, r66, r90, r104;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r110, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r112, {high, high};
}
mov.f32 f33, 0fBF800000;
mov.f32 f34, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
mov.b32 r114, {low, high};
}
{
mul.f16x2 r115, r112, r114;
}
{
mul.f16x2 r118, r87, r110;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r121, {high, low};
}
{
fma.rn.f16x2 r123, r115, r121, r118;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r123;
mov.b32 r127, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r123;
mov.b32 r129, {high, high};
}
{
mul.f16x2 r131, r84, r129;
}
{
fma.rn.f16x2 r134, r48, r127, r131;
}
{
mul.f16x2 r138, r48, r129;
}
{
neg.f16x2 r141, r138;
}
{
fma.rn.f16x2 r143, r84, r127, r141;
}
barrier.sync 0;
mad.lo.s32 r588, r586, 24, r587;
st.shared.v2.f32 [r588], {r6, r12};
st.shared.v2.f32 [r588+8], {r97, r106};
st.shared.v2.f32 [r588+16], {r134, r143};
barrier.sync 0;
shl.b32 r589, r586, 4;
sub.s32 r590, r588, r589;
ld.shared.u32 r170, [r590];
ld.shared.u32 r176, [r590+4];
ld.shared.u32 r167, [r590+216];
ld.shared.u32 r173, [r590+220];
ld.shared.u32 r168, [r590+432];
ld.shared.u32 r174, [r590+436];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f38;
cvt.rn.f16.f32 high, f38;
mov.b32 r164, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f40;
cvt.rn.f16.f32 high, f40;
mov.b32 r165, {low, high};
}
{
add.f16x2 r166, r167, r168;
}
{
add.f16x2 r169, r170, r166;
}
{
add.f16x2 r172, r173, r174;
}
{
add.f16x2 r175, r176, r172;
}
{
add.f16x2 r178, r167, r168;
}
{
mul.f16x2 r181, r178, r164;
}
{
add.f16x2 r184, r170, r181;
}
{
sub.f16x2 r187, r173, r174;
}
{
mul.f16x2 r190, r187, r165;
}
{
add.f16x2 r193, r184, r190;
}
{
add.f16x2 r196, r167, r168;
}
{
mul.f16x2 r199, r196, r164;
}
{
add.f16x2 r202, r170, r199;
}
{
sub.f16x2 r205, r173, r174;
}
{
mul.f16x2 r208, r205, r165;
}
{
sub.f16x2 r211, r202, r208;
}
{
add.f16x2 r214, r173, r174;
}
{
mul.f16x2 r217, r214, r164;
}
{
add.f16x2 r220, r176, r217;
}
{
sub.f16x2 r223, r167, r168;
}
{
mul.f16x2 r226, r223, r165;
}
{
sub.f16x2 r229, r220, r226;
}
{
add.f16x2 r232, r173, r174;
}
{
mul.f16x2 r235, r232, r164;
}
{
add.f16x2 r238, r176, r235;
}
{
sub.f16x2 r241, r167, r168;
}
{
mul.f16x2 r244, r241, r165;
}
{
add.f16x2 r247, r238, r244;
}
mul.wide.u32 rd4, r586, -1431655765;
shr.u64 rd5, rd4, 33;
cvt.u32.u64 r591, rd5;
mul.lo.s32 r592, r591, 3;
sub.s32 r593, r586, r592;
shl.b32 r594, r593, 3;
add.s32 r595, r587, r594;
cvt.rn.f32.u32 f44, r591;
mul.f32 f45, f44, 0f3E6E4BAE;
cos.approx.f32 f17, f45;
sin.approx.f32 f46, f45;
neg.f32 f18, f46;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f17;
cvt.rn.f16.f32 high, f18;
mov.b32 r250, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r250;
mov.b32 r253, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r250;
mov.b32 r255, {high, high};
}
{
mul.f16x2 r257, r229, r255;
}
{
fma.rn.f16x2 r260, r193, r253, r257;
}
{
mul.f16x2 r264, r193, r255;
}
{
neg.f16x2 r267, r264;
}
{
fma.rn.f16x2 r269, r229, r253, r267;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r250;
mov.b32 r273, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r250;
mov.b32 r275, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
mov.b32 r277, {low, high};
}
{
mul.f16x2 r278, r275, r277;
}
{
mul.f16x2 r281, r250, r273;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r250;
mov.b32 r284, {high, low};
}
{
fma.rn.f16x2 r286, r278, r284, r281;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r286;
mov.b32 r290, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r286;
mov.b32 r292, {high, high};
}
{
mul.f16x2 r294, r247, r292;
}
{
fma.rn.f16x2 r297, r211, r290, r294;
}
{
mul.f16x2 r301, r211, r292;
}
{
neg.f16x2 r304, r301;
}
{
fma.rn.f16x2 r306, r247, r290, r304;
}
barrier.sync 0;
mad.lo.s32 r596, r591, 72, r595;
st.shared.u32 [r596], r169;
st.shared.u32 [r596+4], r175;
st.shared.u32 [r596+24], r260;
st.shared.u32 [r596+28], r269;
st.shared.u32 [r596+48], r297;
st.shared.u32 [r596+52], r306;
barrier.sync 0;
ld.shared.u32 r333, [r590];
ld.shared.u32 r339, [r590+4];
ld.shared.u32 r330, [r590+216];
ld.shared.u32 r336, [r590+220];
ld.shared.u32 r331, [r590+432];
ld.shared.u32 r337, [r590+436];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f38;
cvt.rn.f16.f32 high, f38;
mov.b32 r327, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f40;
cvt.rn.f16.f32 high, f40;
mov.b32 r328, {low, high};
}
{
add.f16x2 r329, r330, r331;
}
{
add.f16x2 r332, r333, r329;
}
{
add.f16x2 r335, r336, r337;
}
{
add.f16x2 r338, r339, r335;
}
{
add.f16x2 r341, r330, r331;
}
{
mul.f16x2 r344, r341, r327;
}
{
add.f16x2 r347, r333, r344;
}
{
sub.f16x2 r350, r336, r337;
}
{
mul.f16x2 r353, r350, r328;
}
{
add.f16x2 r356, r347, r353;
}
{
add.f16x2 r359, r330, r331;
}
{
mul.f16x2 r362, r359, r327;
}
{
add.f16x2 r365, r333, r362;
}
{
sub.f16x2 r368, r336, r337;
}
{
mul.f16x2 r371, r368, r328;
}
{
sub.f16x2 r374, r365, r371;
}
{
add.f16x2 r377, r336, r337;
}
{
mul.f16x2 r380, r377, r327;
}
{
add.f16x2 r383, r339, r380;
}
{
sub.f16x2 r386, r330, r331;
}
{
mul.f16x2 r389, r386, r328;
}
{
sub.f16x2 r392, r383, r389;
}
{
add.f16x2 r395, r336, r337;
}
{
mul.f16x2 r398, r395, r327;
}
{
add.f16x2 r401, r339, r398;
}
{
sub.f16x2 r404, r330, r331;
}
{
mul.f16x2 r407, r404, r328;
}
{
add.f16x2 r410, r401, r407;
}
mul.wide.u32 rd6, r586, 954437177;
shr.u64 rd7, rd6, 33;
cvt.u32.u64 r597, rd7;
mul.lo.s32 r598, r597, 9;
sub.s32 r599, r586, r598;
shl.b32 r600, r599, 3;
add.s32 r601, r587, r600;
cvt.rn.f32.u32 f47, r597;
mul.f32 f48, f47, 0f3F32B8C2;
cos.approx.f32 f29, f48;
sin.approx.f32 f49, f48;
neg.f32 f30, f49;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f29;
cvt.rn.f16.f32 high, f30;
mov.b32 r413, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r413;
mov.b32 r416, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r413;
mov.b32 r418, {high, high};
}
{
mul.f16x2 r420, r392, r418;
}
{
fma.rn.f16x2 r423, r356, r416, r420;
}
{
mul.f16x2 r427, r356, r418;
}
{
neg.f16x2 r430, r427;
}
{
fma.rn.f16x2 r432, r392, r416, r430;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r413;
mov.b32 r436, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r413;
mov.b32 r438, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
mov.b32 r440, {low, high};
}
{
mul.f16x2 r441, r438, r440;
}
{
mul.f16x2 r444, r413, r436;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r413;
mov.b32 r447, {high, low};
}
{
fma.rn.f16x2 r449, r441, r447, r444;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r449;
mov.b32 r453, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r449;
mov.b32 r455, {high, high};
}
{
mul.f16x2 r457, r410, r455;
}
{
fma.rn.f16x2 r460, r374, r453, r457;
}
{
mul.f16x2 r464, r374, r455;
}
{
neg.f16x2 r467, r464;
}
{
fma.rn.f16x2 r469, r410, r453, r467;
}
barrier.sync 0;
mad.lo.s32 r602, r597, 216, r601;
st.shared.u32 [r602], r332;
st.shared.u32 [r602+4], r338;
st.shared.u32 [r602+72], r423;
st.shared.u32 [r602+76], r432;
st.shared.u32 [r602+144], r460;
st.shared.u32 [r602+148], r469;
barrier.sync 0;
ld.shared.u32 r496, [r590];
ld.shared.u32 r502, [r590+4];
ld.shared.u32 r493, [r590+216];
ld.shared.u32 r499, [r590+220];
ld.shared.u32 r494, [r590+432];
ld.shared.u32 r500, [r590+436];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f38;
cvt.rn.f16.f32 high, f38;
mov.b32 r490, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f40;
cvt.rn.f16.f32 high, f40;
mov.b32 r491, {low, high};
}
{
add.f16x2 r492, r493, r494;
}
{
add.f16x2 %0, r496, r492;
}
{
add.f16x2 r498, r499, r500;
}
{
add.f16x2 %1, r502, r498;
}
{
add.f16x2 r504, r493, r494;
}
{
mul.f16x2 r507, r504, r490;
}
{
add.f16x2 r510, r496, r507;
}
{
sub.f16x2 r513, r499, r500;
}
{
mul.f16x2 r516, r513, r491;
}
{
add.f16x2 %2, r510, r516;
}
{
add.f16x2 r522, r493, r494;
}
{
mul.f16x2 r525, r522, r490;
}
{
add.f16x2 r528, r496, r525;
}
{
sub.f16x2 r531, r499, r500;
}
{
mul.f16x2 r534, r531, r491;
}
{
sub.f16x2 %4, r528, r534;
}
{
add.f16x2 r540, r499, r500;
}
{
mul.f16x2 r543, r540, r490;
}
{
add.f16x2 r546, r502, r543;
}
{
sub.f16x2 r549, r493, r494;
}
{
mul.f16x2 r552, r549, r491;
}
{
sub.f16x2 %3, r546, r552;
}
{
add.f16x2 r558, r499, r500;
}
{
mul.f16x2 r561, r558, r490;
}
{
add.f16x2 r564, r502, r561;
}
{
sub.f16x2 r567, r493, r494;
}
{
mul.f16x2 r570, r567, r491;
}
{
add.f16x2 %5, r564, r570;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1079, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<50>;
.reg .b32 r<603>;
.reg .b64 rd<8>;
mov.u32 r576, %tid.y;
mov.u32 r577, %6;
mad.lo.s32 r578, r576, 324, r577;
mov.u32 r579, %tid.x;
mov.f32 f38, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f38;
cvt.rn.f16.f32 high, f38;
mov.b32 r1, {low, high};
}
mov.f32 f40, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f40;
cvt.rn.f16.f32 high, f40;
mov.b32 r2, {low, high};
}
{
add.f16x2 r3, %9, %11;
}
{
add.f16x2 r6, %7, r3;
}
{
add.f16x2 r9, %10, %12;
}
{
add.f16x2 r12, %8, r9;
}
{
add.f16x2 r15, %9, %11;
}
{
mul.f16x2 r18, r15, r1;
}
{
add.f16x2 r21, %7, r18;
}
{
sub.f16x2 r24, %10, %12;
}
{
mul.f16x2 r27, r24, r2;
}
{
add.f16x2 r30, r21, r27;
}
{
add.f16x2 r33, %9, %11;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %7, r36;
}
{
sub.f16x2 r42, %10, %12;
}
{
mul.f16x2 r45, r42, r2;
}
{
sub.f16x2 r48, r39, r45;
}
{
add.f16x2 r51, %10, %12;
}
{
mul.f16x2 r54, r51, r1;
}
{
add.f16x2 r57, %8, r54;
}
{
sub.f16x2 r60, %9, %11;
}
{
mul.f16x2 r63, r60, r2;
}
{
sub.f16x2 r66, r57, r63;
}
{
add.f16x2 r69, %10, %12;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %8, r72;
}
{
sub.f16x2 r78, %9, %11;
}
{
mul.f16x2 r81, r78, r2;
}
{
add.f16x2 r84, r75, r81;
}
mul.wide.u32 rd2, r579, 795364315;
shr.u64 rd3, rd2, 32;
cvt.u32.u64 r580, rd3;
sub.s32 r581, r579, r580;
shr.u32 r582, r581, 1;
add.s32 r583, r582, r580;
shr.u32 r584, r583, 4;
mul.lo.s32 r585, r584, 27;
sub.s32 r586, r579, r585;
mad.lo.s32 r587, r584, 324, r578;
cvt.rn.f32.u32 f41, r586;
mul.f32 f42, f41, 0f3D9EDD1F;
cos.approx.f32 f5, f42;
sin.approx.f32 f43, f42;
neg.f32 f6, f43;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f5;
cvt.rn.f16.f32 high, f6;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r90, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r92, {high, high};
}
{
mul.f16x2 r94, r66, r92;
}
{
fma.rn.f16x2 r97, r30, r90, r94;
}
{
mul.f16x2 r101, r30, r92;
}
{
neg.f16x2 r104, r101;
}
{
fma.rn.f16x2 r106, r66, r90, r104;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r110, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r112, {high, high};
}
mov.f32 f33, 0fBF800000;
mov.f32 f34, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
mov.b32 r114, {low, high};
}
{
mul.f16x2 r115, r112, r114;
}
{
mul.f16x2 r118, r87, r110;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r121, {high, low};
}
{
fma.rn.f16x2 r123, r115, r121, r118;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r123;
mov.b32 r127, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r123;
mov.b32 r129, {high, high};
}
{
mul.f16x2 r131, r84, r129;
}
{
fma.rn.f16x2 r134, r48, r127, r131;
}
{
mul.f16x2 r138, r48, r129;
}
{
neg.f16x2 r141, r138;
}
{
fma.rn.f16x2 r143, r84, r127, r141;
}
barrier.sync 0;
mad.lo.s32 r588, r586, 12, r587;
st.shared.u32 [r588], r6;
st.shared.u32 [r588+4], r97;
st.shared.u32 [r588+8], r134;
barrier.sync 0;
shl.b32 r589, r586, 3;
sub.s32 r590, r588, r589;
ld.shared.u32 r170, [r590];
ld.shared.u32 r167, [r590+108];
ld.shared.u32 r168, [r590+216];
barrier.sync 0;
st.shared.u32 [r588], r12;
st.shared.u32 [r588+4], r106;
st.shared.u32 [r588+8], r143;
barrier.sync 0;
ld.shared.u32 r176, [r590];
ld.shared.u32 r173, [r590+108];
ld.shared.u32 r174, [r590+216];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f38;
cvt.rn.f16.f32 high, f38;
mov.b32 r164, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f40;
cvt.rn.f16.f32 high, f40;
mov.b32 r165, {low, high};
}
{
add.f16x2 r166, r167, r168;
}
{
add.f16x2 r169, r170, r166;
}
{
add.f16x2 r172, r173, r174;
}
{
add.f16x2 r175, r176, r172;
}
{
add.f16x2 r178, r167, r168;
}
{
mul.f16x2 r181, r178, r164;
}
{
add.f16x2 r184, r170, r181;
}
{
sub.f16x2 r187, r173, r174;
}
{
mul.f16x2 r190, r187, r165;
}
{
add.f16x2 r193, r184, r190;
}
{
add.f16x2 r196, r167, r168;
}
{
mul.f16x2 r199, r196, r164;
}
{
add.f16x2 r202, r170, r199;
}
{
sub.f16x2 r205, r173, r174;
}
{
mul.f16x2 r208, r205, r165;
}
{
sub.f16x2 r211, r202, r208;
}
{
add.f16x2 r214, r173, r174;
}
{
mul.f16x2 r217, r214, r164;
}
{
add.f16x2 r220, r176, r217;
}
{
sub.f16x2 r223, r167, r168;
}
{
mul.f16x2 r226, r223, r165;
}
{
sub.f16x2 r229, r220, r226;
}
{
add.f16x2 r232, r173, r174;
}
{
mul.f16x2 r235, r232, r164;
}
{
add.f16x2 r238, r176, r235;
}
{
sub.f16x2 r241, r167, r168;
}
{
mul.f16x2 r244, r241, r165;
}
{
add.f16x2 r247, r238, r244;
}
mul.wide.u32 rd4, r586, -1431655765;
shr.u64 rd5, rd4, 33;
cvt.u32.u64 r591, rd5;
mul.lo.s32 r592, r591, 3;
sub.s32 r593, r586, r592;
shl.b32 r594, r593, 2;
add.s32 r595, r587, r594;
cvt.rn.f32.u32 f44, r591;
mul.f32 f45, f44, 0f3E6E4BAE;
cos.approx.f32 f17, f45;
sin.approx.f32 f46, f45;
neg.f32 f18, f46;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f17;
cvt.rn.f16.f32 high, f18;
mov.b32 r250, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r250;
mov.b32 r253, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r250;
mov.b32 r255, {high, high};
}
{
mul.f16x2 r257, r229, r255;
}
{
fma.rn.f16x2 r260, r193, r253, r257;
}
{
mul.f16x2 r264, r193, r255;
}
{
neg.f16x2 r267, r264;
}
{
fma.rn.f16x2 r269, r229, r253, r267;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r250;
mov.b32 r273, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r250;
mov.b32 r275, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
mov.b32 r277, {low, high};
}
{
mul.f16x2 r278, r275, r277;
}
{
mul.f16x2 r281, r250, r273;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r250;
mov.b32 r284, {high, low};
}
{
fma.rn.f16x2 r286, r278, r284, r281;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r286;
mov.b32 r290, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r286;
mov.b32 r292, {high, high};
}
{
mul.f16x2 r294, r247, r292;
}
{
fma.rn.f16x2 r297, r211, r290, r294;
}
{
mul.f16x2 r301, r211, r292;
}
{
neg.f16x2 r304, r301;
}
{
fma.rn.f16x2 r306, r247, r290, r304;
}
barrier.sync 0;
mad.lo.s32 r596, r591, 36, r595;
st.shared.u32 [r596], r169;
st.shared.u32 [r596+12], r260;
st.shared.u32 [r596+24], r297;
barrier.sync 0;
ld.shared.u32 r333, [r590];
ld.shared.u32 r330, [r590+108];
ld.shared.u32 r331, [r590+216];
barrier.sync 0;
st.shared.u32 [r596], r175;
st.shared.u32 [r596+12], r269;
st.shared.u32 [r596+24], r306;
barrier.sync 0;
ld.shared.u32 r339, [r590];
ld.shared.u32 r336, [r590+108];
ld.shared.u32 r337, [r590+216];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f38;
cvt.rn.f16.f32 high, f38;
mov.b32 r327, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f40;
cvt.rn.f16.f32 high, f40;
mov.b32 r328, {low, high};
}
{
add.f16x2 r329, r330, r331;
}
{
add.f16x2 r332, r333, r329;
}
{
add.f16x2 r335, r336, r337;
}
{
add.f16x2 r338, r339, r335;
}
{
add.f16x2 r341, r330, r331;
}
{
mul.f16x2 r344, r341, r327;
}
{
add.f16x2 r347, r333, r344;
}
{
sub.f16x2 r350, r336, r337;
}
{
mul.f16x2 r353, r350, r328;
}
{
add.f16x2 r356, r347, r353;
}
{
add.f16x2 r359, r330, r331;
}
{
mul.f16x2 r362, r359, r327;
}
{
add.f16x2 r365, r333, r362;
}
{
sub.f16x2 r368, r336, r337;
}
{
mul.f16x2 r371, r368, r328;
}
{
sub.f16x2 r374, r365, r371;
}
{
add.f16x2 r377, r336, r337;
}
{
mul.f16x2 r380, r377, r327;
}
{
add.f16x2 r383, r339, r380;
}
{
sub.f16x2 r386, r330, r331;
}
{
mul.f16x2 r389, r386, r328;
}
{
sub.f16x2 r392, r383, r389;
}
{
add.f16x2 r395, r336, r337;
}
{
mul.f16x2 r398, r395, r327;
}
{
add.f16x2 r401, r339, r398;
}
{
sub.f16x2 r404, r330, r331;
}
{
mul.f16x2 r407, r404, r328;
}
{
add.f16x2 r410, r401, r407;
}
mul.wide.u32 rd6, r586, 954437177;
shr.u64 rd7, rd6, 33;
cvt.u32.u64 r597, rd7;
mul.lo.s32 r598, r597, 9;
sub.s32 r599, r586, r598;
shl.b32 r600, r599, 2;
add.s32 r601, r587, r600;
cvt.rn.f32.u32 f47, r597;
mul.f32 f48, f47, 0f3F32B8C2;
cos.approx.f32 f29, f48;
sin.approx.f32 f49, f48;
neg.f32 f30, f49;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f29;
cvt.rn.f16.f32 high, f30;
mov.b32 r413, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r413;
mov.b32 r416, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r413;
mov.b32 r418, {high, high};
}
{
mul.f16x2 r420, r392, r418;
}
{
fma.rn.f16x2 r423, r356, r416, r420;
}
{
mul.f16x2 r427, r356, r418;
}
{
neg.f16x2 r430, r427;
}
{
fma.rn.f16x2 r432, r392, r416, r430;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r413;
mov.b32 r436, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r413;
mov.b32 r438, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
mov.b32 r440, {low, high};
}
{
mul.f16x2 r441, r438, r440;
}
{
mul.f16x2 r444, r413, r436;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r413;
mov.b32 r447, {high, low};
}
{
fma.rn.f16x2 r449, r441, r447, r444;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r449;
mov.b32 r453, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r449;
mov.b32 r455, {high, high};
}
{
mul.f16x2 r457, r410, r455;
}
{
fma.rn.f16x2 r460, r374, r453, r457;
}
{
mul.f16x2 r464, r374, r455;
}
{
neg.f16x2 r467, r464;
}
{
fma.rn.f16x2 r469, r410, r453, r467;
}
barrier.sync 0;
mad.lo.s32 r602, r597, 108, r601;
st.shared.u32 [r602], r332;
st.shared.u32 [r602+36], r423;
st.shared.u32 [r602+72], r460;
barrier.sync 0;
ld.shared.u32 r496, [r590];
ld.shared.u32 r493, [r590+108];
ld.shared.u32 r494, [r590+216];
barrier.sync 0;
st.shared.u32 [r602], r338;
st.shared.u32 [r602+36], r432;
st.shared.u32 [r602+72], r469;
barrier.sync 0;
ld.shared.u32 r502, [r590];
ld.shared.u32 r499, [r590+108];
ld.shared.u32 r500, [r590+216];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f38;
cvt.rn.f16.f32 high, f38;
mov.b32 r490, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f40;
cvt.rn.f16.f32 high, f40;
mov.b32 r491, {low, high};
}
{
add.f16x2 r492, r493, r494;
}
{
add.f16x2 %0, r496, r492;
}
{
add.f16x2 r498, r499, r500;
}
{
add.f16x2 %1, r502, r498;
}
{
add.f16x2 r504, r493, r494;
}
{
mul.f16x2 r507, r504, r490;
}
{
add.f16x2 r510, r496, r507;
}
{
sub.f16x2 r513, r499, r500;
}
{
mul.f16x2 r516, r513, r491;
}
{
add.f16x2 %2, r510, r516;
}
{
add.f16x2 r522, r493, r494;
}
{
mul.f16x2 r525, r522, r490;
}
{
add.f16x2 r528, r496, r525;
}
{
sub.f16x2 r531, r499, r500;
}
{
mul.f16x2 r534, r531, r491;
}
{
sub.f16x2 %4, r528, r534;
}
{
add.f16x2 r540, r499, r500;
}
{
mul.f16x2 r543, r540, r490;
}
{
add.f16x2 r546, r502, r543;
}
{
sub.f16x2 r549, r493, r494;
}
{
mul.f16x2 r552, r549, r491;
}
{
sub.f16x2 %3, r546, r552;
}
{
add.f16x2 r558, r499, r500;
}
{
mul.f16x2 r561, r558, r490;
}
{
add.f16x2 r564, r502, r561;
}
{
sub.f16x2 r567, r493, r494;
}
{
mul.f16x2 r570, r567, r491;
}
{
add.f16x2 %5, r564, r570;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)));
};


#endif
