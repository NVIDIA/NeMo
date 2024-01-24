#ifndef CUFFTDX_FFT_2187_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_2187_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1093, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<771>;
.reg .b32 r<8520>;
.reg .b64 rd<6>;
mov.u32 r8446, %54;
mov.u32 r8519, %tid.y;
mad.lo.s32 r8447, r8519, 17496, r8446;
mov.u32 r8448, %tid.x;
mov.f32 f762, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1, {low, high};
}
mov.f32 f764, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r2, {low, high};
}
{
add.f16x2 r3, %108, %99;
}
{
add.f16x2 r6, %81, r3;
}
{
add.f16x2 r9, %60, %106;
}
{
add.f16x2 r12, %90, r9;
}
{
add.f16x2 r15, %108, %99;
}
{
mul.f16x2 r18, r15, r1;
}
{
add.f16x2 r21, %81, r18;
}
{
sub.f16x2 r24, %60, %106;
}
{
mul.f16x2 r27, r24, r2;
}
{
add.f16x2 r30, r21, r27;
}
{
add.f16x2 r33, %108, %99;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %81, r36;
}
{
sub.f16x2 r42, %60, %106;
}
{
mul.f16x2 r45, r42, r2;
}
{
sub.f16x2 r48, r39, r45;
}
{
add.f16x2 r51, %60, %106;
}
{
mul.f16x2 r54, r51, r1;
}
{
add.f16x2 r57, %90, r54;
}
{
sub.f16x2 r60, %108, %99;
}
{
mul.f16x2 r63, r60, r2;
}
{
sub.f16x2 r66, r57, r63;
}
{
add.f16x2 r69, %60, %106;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %90, r72;
}
{
sub.f16x2 r78, %108, %99;
}
{
mul.f16x2 r81, r78, r2;
}
{
add.f16x2 r84, r75, r81;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r88, {low, high};
}
{
add.f16x2 r89, %107, %98;
}
{
add.f16x2 r92, %80, r89;
}
{
add.f16x2 r95, %59, %104;
}
{
add.f16x2 r98, %89, r95;
}
{
add.f16x2 r101, %107, %98;
}
{
mul.f16x2 r104, r101, r87;
}
{
add.f16x2 r107, %80, r104;
}
{
sub.f16x2 r110, %59, %104;
}
{
mul.f16x2 r113, r110, r88;
}
{
add.f16x2 r116, r107, r113;
}
{
add.f16x2 r119, %107, %98;
}
{
mul.f16x2 r122, r119, r87;
}
{
add.f16x2 r125, %80, r122;
}
{
sub.f16x2 r128, %59, %104;
}
{
mul.f16x2 r131, r128, r88;
}
{
sub.f16x2 r134, r125, r131;
}
{
add.f16x2 r137, %59, %104;
}
{
mul.f16x2 r140, r137, r87;
}
{
add.f16x2 r143, %89, r140;
}
{
sub.f16x2 r146, %107, %98;
}
{
mul.f16x2 r149, r146, r88;
}
{
sub.f16x2 r152, r143, r149;
}
{
add.f16x2 r155, %59, %104;
}
{
mul.f16x2 r158, r155, r87;
}
{
add.f16x2 r161, %89, r158;
}
{
sub.f16x2 r164, %107, %98;
}
{
mul.f16x2 r167, r164, r88;
}
{
add.f16x2 r170, r161, r167;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r173, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r174, {low, high};
}
{
add.f16x2 r175, %105, %97;
}
{
add.f16x2 r178, %79, r175;
}
{
add.f16x2 r181, %58, %103;
}
{
add.f16x2 r184, %88, r181;
}
{
add.f16x2 r187, %105, %97;
}
{
mul.f16x2 r190, r187, r173;
}
{
add.f16x2 r193, %79, r190;
}
{
sub.f16x2 r196, %58, %103;
}
{
mul.f16x2 r199, r196, r174;
}
{
add.f16x2 r202, r193, r199;
}
{
add.f16x2 r205, %105, %97;
}
{
mul.f16x2 r208, r205, r173;
}
{
add.f16x2 r211, %79, r208;
}
{
sub.f16x2 r214, %58, %103;
}
{
mul.f16x2 r217, r214, r174;
}
{
sub.f16x2 r220, r211, r217;
}
{
add.f16x2 r223, %58, %103;
}
{
mul.f16x2 r226, r223, r173;
}
{
add.f16x2 r229, %88, r226;
}
{
sub.f16x2 r232, %105, %97;
}
{
mul.f16x2 r235, r232, r174;
}
{
sub.f16x2 r238, r229, r235;
}
{
add.f16x2 r241, %58, %103;
}
{
mul.f16x2 r244, r241, r173;
}
{
add.f16x2 r247, %88, r244;
}
{
sub.f16x2 r250, %105, %97;
}
{
mul.f16x2 r253, r250, r174;
}
{
add.f16x2 r256, r247, r253;
}
mov.f32 f542, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r259, {low, high};
}
mov.f32 f544, 0f3F248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r260, {low, high};
}
mov.f32 f554, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r261, {low, high};
}
mov.f32 f556, 0f3F7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r262, {low, high};
}
mov.f32 f578, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r265, {low, high};
}
mov.f32 f580, 0f3EAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r339, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r425, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r597, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r598, {low, high};
}
{
add.f16x2 r599, %96, %84;
}
{
add.f16x2 r602, %66, r599;
}
{
add.f16x2 r605, %102, %94;
}
{
add.f16x2 r608, %72, r605;
}
{
add.f16x2 r611, %96, %84;
}
{
mul.f16x2 r614, r611, r597;
}
{
add.f16x2 r617, %66, r614;
}
{
sub.f16x2 r620, %102, %94;
}
{
mul.f16x2 r623, r620, r598;
}
{
add.f16x2 r626, r617, r623;
}
{
add.f16x2 r629, %96, %84;
}
{
mul.f16x2 r632, r629, r597;
}
{
add.f16x2 r635, %66, r632;
}
{
sub.f16x2 r638, %102, %94;
}
{
mul.f16x2 r641, r638, r598;
}
{
sub.f16x2 r644, r635, r641;
}
{
add.f16x2 r647, %102, %94;
}
{
mul.f16x2 r650, r647, r597;
}
{
add.f16x2 r653, %72, r650;
}
{
sub.f16x2 r656, %96, %84;
}
{
mul.f16x2 r659, r656, r598;
}
{
sub.f16x2 r662, r653, r659;
}
{
add.f16x2 r665, %102, %94;
}
{
mul.f16x2 r668, r665, r597;
}
{
add.f16x2 r671, %72, r668;
}
{
sub.f16x2 r674, %96, %84;
}
{
mul.f16x2 r677, r674, r598;
}
{
add.f16x2 r680, r671, r677;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r683, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r684, {low, high};
}
{
add.f16x2 r685, %95, %83;
}
{
add.f16x2 r688, %65, r685;
}
{
add.f16x2 r691, %101, %92;
}
{
add.f16x2 r694, %71, r691;
}
{
add.f16x2 r697, %95, %83;
}
{
mul.f16x2 r700, r697, r683;
}
{
add.f16x2 r703, %65, r700;
}
{
sub.f16x2 r706, %101, %92;
}
{
mul.f16x2 r709, r706, r684;
}
{
add.f16x2 r712, r703, r709;
}
{
add.f16x2 r715, %95, %83;
}
{
mul.f16x2 r718, r715, r683;
}
{
add.f16x2 r721, %65, r718;
}
{
sub.f16x2 r724, %101, %92;
}
{
mul.f16x2 r727, r724, r684;
}
{
sub.f16x2 r730, r721, r727;
}
{
add.f16x2 r733, %101, %92;
}
{
mul.f16x2 r736, r733, r683;
}
{
add.f16x2 r739, %71, r736;
}
{
sub.f16x2 r742, %95, %83;
}
{
mul.f16x2 r745, r742, r684;
}
{
sub.f16x2 r748, r739, r745;
}
{
add.f16x2 r751, %101, %92;
}
{
mul.f16x2 r754, r751, r683;
}
{
add.f16x2 r757, %71, r754;
}
{
sub.f16x2 r760, %95, %83;
}
{
mul.f16x2 r763, r760, r684;
}
{
add.f16x2 r766, r757, r763;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r769, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r770, {low, high};
}
{
add.f16x2 r771, %93, %82;
}
{
add.f16x2 r774, %64, r771;
}
{
add.f16x2 r777, %100, %91;
}
{
add.f16x2 r780, %70, r777;
}
{
add.f16x2 r783, %93, %82;
}
{
mul.f16x2 r786, r783, r769;
}
{
add.f16x2 r789, %64, r786;
}
{
sub.f16x2 r792, %100, %91;
}
{
mul.f16x2 r795, r792, r770;
}
{
add.f16x2 r798, r789, r795;
}
{
add.f16x2 r801, %93, %82;
}
{
mul.f16x2 r804, r801, r769;
}
{
add.f16x2 r807, %64, r804;
}
{
sub.f16x2 r810, %100, %91;
}
{
mul.f16x2 r813, r810, r770;
}
{
sub.f16x2 r816, r807, r813;
}
{
add.f16x2 r819, %100, %91;
}
{
mul.f16x2 r822, r819, r769;
}
{
add.f16x2 r825, %70, r822;
}
{
sub.f16x2 r828, %93, %82;
}
{
mul.f16x2 r831, r828, r770;
}
{
sub.f16x2 r834, r825, r831;
}
{
add.f16x2 r837, %100, %91;
}
{
mul.f16x2 r840, r837, r769;
}
{
add.f16x2 r843, %70, r840;
}
{
sub.f16x2 r846, %93, %82;
}
{
mul.f16x2 r849, r846, r770;
}
{
add.f16x2 r852, r843, r849;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r855, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r856, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r857, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r858, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r861, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r935, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1107, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1193, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r1194, {low, high};
}
{
add.f16x2 r1195, %78, %69;
}
{
add.f16x2 r1198, %57, r1195;
}
{
add.f16x2 r1201, %87, %76;
}
{
add.f16x2 r1204, %63, r1201;
}
{
add.f16x2 r1207, %78, %69;
}
{
mul.f16x2 r1210, r1207, r1193;
}
{
add.f16x2 r1213, %57, r1210;
}
{
sub.f16x2 r1216, %87, %76;
}
{
mul.f16x2 r1219, r1216, r1194;
}
{
add.f16x2 r1222, r1213, r1219;
}
{
add.f16x2 r1225, %78, %69;
}
{
mul.f16x2 r1228, r1225, r1193;
}
{
add.f16x2 r1231, %57, r1228;
}
{
sub.f16x2 r1234, %87, %76;
}
{
mul.f16x2 r1237, r1234, r1194;
}
{
sub.f16x2 r1240, r1231, r1237;
}
{
add.f16x2 r1243, %87, %76;
}
{
mul.f16x2 r1246, r1243, r1193;
}
{
add.f16x2 r1249, %63, r1246;
}
{
sub.f16x2 r1252, %78, %69;
}
{
mul.f16x2 r1255, r1252, r1194;
}
{
sub.f16x2 r1258, r1249, r1255;
}
{
add.f16x2 r1261, %87, %76;
}
{
mul.f16x2 r1264, r1261, r1193;
}
{
add.f16x2 r1267, %63, r1264;
}
{
sub.f16x2 r1270, %78, %69;
}
{
mul.f16x2 r1273, r1270, r1194;
}
{
add.f16x2 r1276, r1267, r1273;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1279, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r1280, {low, high};
}
{
add.f16x2 r1281, %77, %68;
}
{
add.f16x2 r1284, %56, r1281;
}
{
add.f16x2 r1287, %86, %74;
}
{
add.f16x2 r1290, %62, r1287;
}
{
add.f16x2 r1293, %77, %68;
}
{
mul.f16x2 r1296, r1293, r1279;
}
{
add.f16x2 r1299, %56, r1296;
}
{
sub.f16x2 r1302, %86, %74;
}
{
mul.f16x2 r1305, r1302, r1280;
}
{
add.f16x2 r1308, r1299, r1305;
}
{
add.f16x2 r1311, %77, %68;
}
{
mul.f16x2 r1314, r1311, r1279;
}
{
add.f16x2 r1317, %56, r1314;
}
{
sub.f16x2 r1320, %86, %74;
}
{
mul.f16x2 r1323, r1320, r1280;
}
{
sub.f16x2 r1326, r1317, r1323;
}
{
add.f16x2 r1329, %86, %74;
}
{
mul.f16x2 r1332, r1329, r1279;
}
{
add.f16x2 r1335, %62, r1332;
}
{
sub.f16x2 r1338, %77, %68;
}
{
mul.f16x2 r1341, r1338, r1280;
}
{
sub.f16x2 r1344, r1335, r1341;
}
{
add.f16x2 r1347, %86, %74;
}
{
mul.f16x2 r1350, r1347, r1279;
}
{
add.f16x2 r1353, %62, r1350;
}
{
sub.f16x2 r1356, %77, %68;
}
{
mul.f16x2 r1359, r1356, r1280;
}
{
add.f16x2 r1362, r1353, r1359;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1365, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r1366, {low, high};
}
{
add.f16x2 r1367, %75, %67;
}
{
add.f16x2 r1370, %55, r1367;
}
{
add.f16x2 r1373, %85, %73;
}
{
add.f16x2 r1376, %61, r1373;
}
{
add.f16x2 r1379, %75, %67;
}
{
mul.f16x2 r1382, r1379, r1365;
}
{
add.f16x2 r1385, %55, r1382;
}
{
sub.f16x2 r1388, %85, %73;
}
{
mul.f16x2 r1391, r1388, r1366;
}
{
add.f16x2 r1394, r1385, r1391;
}
{
add.f16x2 r1397, %75, %67;
}
{
mul.f16x2 r1400, r1397, r1365;
}
{
add.f16x2 r1403, %55, r1400;
}
{
sub.f16x2 r1406, %85, %73;
}
{
mul.f16x2 r1409, r1406, r1366;
}
{
sub.f16x2 r1412, r1403, r1409;
}
{
add.f16x2 r1415, %85, %73;
}
{
mul.f16x2 r1418, r1415, r1365;
}
{
add.f16x2 r1421, %61, r1418;
}
{
sub.f16x2 r1424, %75, %67;
}
{
mul.f16x2 r1427, r1424, r1366;
}
{
sub.f16x2 r1430, r1421, r1427;
}
{
add.f16x2 r1433, %85, %73;
}
{
mul.f16x2 r1436, r1433, r1365;
}
{
add.f16x2 r1439, %61, r1436;
}
{
sub.f16x2 r1442, %75, %67;
}
{
mul.f16x2 r1445, r1442, r1366;
}
{
add.f16x2 r1448, r1439, r1445;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r1451, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r1452, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r1453, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r1454, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r1457, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1531, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1617, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1703, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
mov.f32 f534, 0f3F791978;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f534;
cvt.rn.f16.f32 high, f534;
mov.b32 r1789, {low, high};
}
mov.f32 f536, 0f3E6C2691;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f536;
cvt.rn.f16.f32 high, f536;
mov.b32 r1790, {low, high};
}
mov.f32 f538, 0f3F64C51C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f538;
cvt.rn.f16.f32 high, f538;
mov.b32 r1791, {low, high};
}
mov.f32 f540, 0f3EE5C902;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f540;
cvt.rn.f16.f32 high, f540;
mov.b32 r1792, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r1793, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r1794, {low, high};
}
mov.f32 f546, 0f3F18DF63;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f546;
cvt.rn.f16.f32 high, f546;
mov.b32 r1795, {low, high};
}
mov.f32 f548, 0f3F4D57F2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f548;
cvt.rn.f16.f32 high, f548;
mov.b32 r1796, {low, high};
}
mov.f32 f550, 0f3ECACAF8;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f550;
cvt.rn.f16.f32 high, f550;
mov.b32 r1797, {low, high};
}
mov.f32 f552, 0f3F6B1036;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f552;
cvt.rn.f16.f32 high, f552;
mov.b32 r1798, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r1799, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r1800, {low, high};
}
mov.f32 f558, 0fBD6E2946;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f558;
cvt.rn.f16.f32 high, f558;
mov.b32 r1801, {low, high};
}
mov.f32 f560, 0f3F7F9120;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f560;
cvt.rn.f16.f32 high, f560;
mov.b32 r1802, {low, high};
}
mov.f32 f562, 0fBE92D7E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f562;
cvt.rn.f16.f32 high, f562;
mov.b32 r1803, {low, high};
}
mov.f32 f564, 0f3F753ECD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f564;
cvt.rn.f16.f32 high, f564;
mov.b32 r1804, {low, high};
}
mov.f32 f570, 0fBF2FAD88;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f570;
cvt.rn.f16.f32 high, f570;
mov.b32 r1807, {low, high};
}
mov.f32 f572, 0f3F3A3529;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f572;
cvt.rn.f16.f32 high, f572;
mov.b32 r1808, {low, high};
}
mov.f32 f594, 0fBF55E287;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r1811, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
mov.b32 r1812, {low, high};
}
mov.f32 f586, 0fBF7E44DE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f586;
cvt.rn.f16.f32 high, f586;
mov.b32 r1815, {low, high};
}
mov.f32 f588, 0fBDEDC21F;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f588;
cvt.rn.f16.f32 high, f588;
mov.b32 r1816, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1819, {low, high};
}
mov.f32 f596, 0fBF0CAC9F;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2097, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2183, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2269, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2355, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2441, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2527, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2613, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2699, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2785, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
mul.wide.u32 rd2, r8448, -901412889;
shr.u64 rd3, rd2, 38;
cvt.u32.u64 r8449, rd3;
mul.lo.s32 r8450, r8449, 81;
sub.s32 r8451, r8448, r8450;
cvt.rn.f32.u32 f765, r8451;
mul.f32 f766, f765, 0f3B3C4870;
cos.approx.f32 f309, f766;
sin.approx.f32 f767, f766;
neg.f32 f310, f767;
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
mov.f32 f725, 0fBF800000;
mov.f32 f726, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
mad.lo.s32 r8452, r8449, 17496, r8447;
barrier.sync 0;
mad.lo.s32 r8453, r8451, 216, r8452;
st.shared.v2.f32 [r8453], {r2102, r2108};
st.shared.v2.f32 [r8453+8], {r2881, r2890};
st.shared.v2.f32 [r8453+16], {r2918, r2927};
st.shared.v2.f32 [r8453+24], {r2955, r2964};
st.shared.v2.f32 [r8453+32], {r2992, r3001};
st.shared.v2.f32 [r8453+40], {r3029, r3038};
st.shared.v2.f32 [r8453+48], {r3066, r3075};
st.shared.v2.f32 [r8453+56], {r3103, r3112};
st.shared.v2.f32 [r8453+64], {r3140, r3149};
st.shared.v2.f32 [r8453+72], {r3177, r3186};
st.shared.v2.f32 [r8453+80], {r3214, r3223};
st.shared.v2.f32 [r8453+88], {r3251, r3260};
st.shared.v2.f32 [r8453+96], {r3288, r3297};
st.shared.v2.f32 [r8453+104], {r3325, r3334};
st.shared.v2.f32 [r8453+112], {r3362, r3371};
st.shared.v2.f32 [r8453+120], {r3399, r3408};
st.shared.v2.f32 [r8453+128], {r3436, r3445};
st.shared.v2.f32 [r8453+136], {r3473, r3482};
st.shared.v2.f32 [r8453+144], {r3510, r3519};
st.shared.v2.f32 [r8453+152], {r3547, r3556};
st.shared.v2.f32 [r8453+160], {r3584, r3593};
st.shared.v2.f32 [r8453+168], {r3621, r3630};
st.shared.v2.f32 [r8453+176], {r3658, r3667};
st.shared.v2.f32 [r8453+184], {r3695, r3704};
st.shared.v2.f32 [r8453+192], {r3732, r3741};
st.shared.v2.f32 [r8453+200], {r3769, r3778};
st.shared.v2.f32 [r8453+208], {r3806, r3815};
barrier.sync 0;
mad.lo.s32 r8454, r8451, -208, r8453;
ld.shared.u32 r3842, [r8454];
ld.shared.u32 r3848, [r8454+4];
ld.shared.u32 r4438, [r8454+648];
ld.shared.u32 r4444, [r8454+652];
ld.shared.u32 r5034, [r8454+1296];
ld.shared.u32 r5040, [r8454+1300];
ld.shared.u32 r3928, [r8454+1944];
ld.shared.u32 r3934, [r8454+1948];
ld.shared.u32 r4524, [r8454+2592];
ld.shared.u32 r4530, [r8454+2596];
ld.shared.u32 r5120, [r8454+3240];
ld.shared.u32 r5126, [r8454+3244];
ld.shared.u32 r4014, [r8454+3888];
ld.shared.u32 r4020, [r8454+3892];
ld.shared.u32 r4610, [r8454+4536];
ld.shared.u32 r4616, [r8454+4540];
ld.shared.u32 r5206, [r8454+5184];
ld.shared.u32 r5212, [r8454+5188];
ld.shared.u32 r3839, [r8454+5832];
ld.shared.u32 r3845, [r8454+5836];
ld.shared.u32 r4435, [r8454+6480];
ld.shared.u32 r4441, [r8454+6484];
ld.shared.u32 r5031, [r8454+7128];
ld.shared.u32 r5037, [r8454+7132];
ld.shared.u32 r3925, [r8454+7776];
ld.shared.u32 r3931, [r8454+7780];
ld.shared.u32 r4521, [r8454+8424];
ld.shared.u32 r4527, [r8454+8428];
ld.shared.u32 r5117, [r8454+9072];
ld.shared.u32 r5123, [r8454+9076];
ld.shared.u32 r4011, [r8454+9720];
ld.shared.u32 r4017, [r8454+9724];
ld.shared.u32 r4607, [r8454+10368];
ld.shared.u32 r4613, [r8454+10372];
ld.shared.u32 r5203, [r8454+11016];
ld.shared.u32 r5209, [r8454+11020];
ld.shared.u32 r3840, [r8454+11664];
ld.shared.u32 r3846, [r8454+11668];
ld.shared.u32 r4436, [r8454+12312];
ld.shared.u32 r4442, [r8454+12316];
ld.shared.u32 r5032, [r8454+12960];
ld.shared.u32 r5038, [r8454+12964];
ld.shared.u32 r3926, [r8454+13608];
ld.shared.u32 r3932, [r8454+13612];
ld.shared.u32 r4522, [r8454+14256];
ld.shared.u32 r4528, [r8454+14260];
ld.shared.u32 r5118, [r8454+14904];
ld.shared.u32 r5124, [r8454+14908];
ld.shared.u32 r4012, [r8454+15552];
ld.shared.u32 r4018, [r8454+15556];
ld.shared.u32 r4608, [r8454+16200];
ld.shared.u32 r4614, [r8454+16204];
ld.shared.u32 r5204, [r8454+16848];
ld.shared.u32 r5210, [r8454+16852];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r3836, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r3837, {low, high};
}
{
add.f16x2 r3838, r3839, r3840;
}
{
add.f16x2 r3841, r3842, r3838;
}
{
add.f16x2 r3844, r3845, r3846;
}
{
add.f16x2 r3847, r3848, r3844;
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
add.f16x2 r3865, r3856, r3862;
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
sub.f16x2 r3883, r3874, r3880;
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
sub.f16x2 r3901, r3892, r3898;
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
add.f16x2 r3919, r3910, r3916;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r3922, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r3923, {low, high};
}
{
add.f16x2 r3924, r3925, r3926;
}
{
add.f16x2 r3927, r3928, r3924;
}
{
add.f16x2 r3930, r3931, r3932;
}
{
add.f16x2 r3933, r3934, r3930;
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
add.f16x2 r3951, r3942, r3948;
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
sub.f16x2 r3969, r3960, r3966;
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
sub.f16x2 r3987, r3978, r3984;
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
add.f16x2 r4005, r3996, r4002;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4008, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4009, {low, high};
}
{
add.f16x2 r4010, r4011, r4012;
}
{
add.f16x2 r4013, r4014, r4010;
}
{
add.f16x2 r4016, r4017, r4018;
}
{
add.f16x2 r4019, r4020, r4016;
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
add.f16x2 r4037, r4028, r4034;
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
sub.f16x2 r4055, r4046, r4052;
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
sub.f16x2 r4073, r4064, r4070;
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
add.f16x2 r4091, r4082, r4088;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r4094, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r4095, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r4096, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r4097, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r4100, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
mov.b32 r4101, {low, high};
}
{
mul.f16x2 r4110, r3951, r4094;
}
{
mul.f16x2 r4113, r3987, r4095;
}
{
sub.f16x2 r4116, r4110, r4113;
}
{
mul.f16x2 r4119, r3951, r4095;
}
{
fma.rn.f16x2 r4122, r3987, r4094, r4119;
}
{
mul.f16x2 r4126, r4037, r4096;
}
{
mul.f16x2 r4129, r4073, r4097;
}
{
sub.f16x2 r4132, r4126, r4129;
}
{
mul.f16x2 r4135, r4037, r4097;
}
{
fma.rn.f16x2 r4138, r4073, r4096, r4135;
}
{
mul.f16x2 r4142, r3969, r4096;
}
{
mul.f16x2 r4145, r4005, r4097;
}
{
sub.f16x2 r4148, r4142, r4145;
}
{
mul.f16x2 r4151, r3969, r4097;
}
{
fma.rn.f16x2 r4154, r4005, r4096, r4151;
}
{
mul.f16x2 r4158, r4055, r4100;
}
{
mul.f16x2 r4161, r4091, r4101;
}
{
sub.f16x2 r4164, r4158, r4161;
}
{
mul.f16x2 r4167, r4055, r4101;
}
{
fma.rn.f16x2 r4170, r4091, r4100, r4167;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4174, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4175, {low, high};
}
{
add.f16x2 r4176, r3927, r4013;
}
{
add.f16x2 r4179, r3841, r4176;
}
{
add.f16x2 r4182, r3933, r4019;
}
{
add.f16x2 r4185, r3847, r4182;
}
{
add.f16x2 r4188, r3927, r4013;
}
{
mul.f16x2 r4191, r4188, r4174;
}
{
add.f16x2 r4194, r3841, r4191;
}
{
sub.f16x2 r4197, r3933, r4019;
}
{
mul.f16x2 r4200, r4197, r4175;
}
{
add.f16x2 r4203, r4194, r4200;
}
{
add.f16x2 r4206, r3927, r4013;
}
{
mul.f16x2 r4209, r4206, r4174;
}
{
add.f16x2 r4212, r3841, r4209;
}
{
sub.f16x2 r4215, r3933, r4019;
}
{
mul.f16x2 r4218, r4215, r4175;
}
{
sub.f16x2 r4221, r4212, r4218;
}
{
add.f16x2 r4224, r3933, r4019;
}
{
mul.f16x2 r4227, r4224, r4174;
}
{
add.f16x2 r4230, r3847, r4227;
}
{
sub.f16x2 r4233, r3927, r4013;
}
{
mul.f16x2 r4236, r4233, r4175;
}
{
sub.f16x2 r4239, r4230, r4236;
}
{
add.f16x2 r4242, r3933, r4019;
}
{
mul.f16x2 r4245, r4242, r4174;
}
{
add.f16x2 r4248, r3847, r4245;
}
{
sub.f16x2 r4251, r3927, r4013;
}
{
mul.f16x2 r4254, r4251, r4175;
}
{
add.f16x2 r4257, r4248, r4254;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4260, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4261, {low, high};
}
{
add.f16x2 r4262, r4116, r4132;
}
{
add.f16x2 r4265, r3865, r4262;
}
{
add.f16x2 r4268, r4122, r4138;
}
{
add.f16x2 r4271, r3901, r4268;
}
{
add.f16x2 r4274, r4116, r4132;
}
{
mul.f16x2 r4277, r4274, r4260;
}
{
add.f16x2 r4280, r3865, r4277;
}
{
sub.f16x2 r4283, r4122, r4138;
}
{
mul.f16x2 r4286, r4283, r4261;
}
{
add.f16x2 r4289, r4280, r4286;
}
{
add.f16x2 r4292, r4116, r4132;
}
{
mul.f16x2 r4295, r4292, r4260;
}
{
add.f16x2 r4298, r3865, r4295;
}
{
sub.f16x2 r4301, r4122, r4138;
}
{
mul.f16x2 r4304, r4301, r4261;
}
{
sub.f16x2 r4307, r4298, r4304;
}
{
add.f16x2 r4310, r4122, r4138;
}
{
mul.f16x2 r4313, r4310, r4260;
}
{
add.f16x2 r4316, r3901, r4313;
}
{
sub.f16x2 r4319, r4116, r4132;
}
{
mul.f16x2 r4322, r4319, r4261;
}
{
sub.f16x2 r4325, r4316, r4322;
}
{
add.f16x2 r4328, r4122, r4138;
}
{
mul.f16x2 r4331, r4328, r4260;
}
{
add.f16x2 r4334, r3901, r4331;
}
{
sub.f16x2 r4337, r4116, r4132;
}
{
mul.f16x2 r4340, r4337, r4261;
}
{
add.f16x2 r4343, r4334, r4340;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4346, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4347, {low, high};
}
{
add.f16x2 r4348, r4148, r4164;
}
{
add.f16x2 r4351, r3883, r4348;
}
{
add.f16x2 r4354, r4154, r4170;
}
{
add.f16x2 r4357, r3919, r4354;
}
{
add.f16x2 r4360, r4148, r4164;
}
{
mul.f16x2 r4363, r4360, r4346;
}
{
add.f16x2 r4366, r3883, r4363;
}
{
sub.f16x2 r4369, r4154, r4170;
}
{
mul.f16x2 r4372, r4369, r4347;
}
{
add.f16x2 r4375, r4366, r4372;
}
{
add.f16x2 r4378, r4148, r4164;
}
{
mul.f16x2 r4381, r4378, r4346;
}
{
add.f16x2 r4384, r3883, r4381;
}
{
sub.f16x2 r4387, r4154, r4170;
}
{
mul.f16x2 r4390, r4387, r4347;
}
{
sub.f16x2 r4393, r4384, r4390;
}
{
add.f16x2 r4396, r4154, r4170;
}
{
mul.f16x2 r4399, r4396, r4346;
}
{
add.f16x2 r4402, r3919, r4399;
}
{
sub.f16x2 r4405, r4148, r4164;
}
{
mul.f16x2 r4408, r4405, r4347;
}
{
sub.f16x2 r4411, r4402, r4408;
}
{
add.f16x2 r4414, r4154, r4170;
}
{
mul.f16x2 r4417, r4414, r4346;
}
{
add.f16x2 r4420, r3919, r4417;
}
{
sub.f16x2 r4423, r4148, r4164;
}
{
mul.f16x2 r4426, r4423, r4347;
}
{
add.f16x2 r4429, r4420, r4426;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4432, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4433, {low, high};
}
{
add.f16x2 r4434, r4435, r4436;
}
{
add.f16x2 r4437, r4438, r4434;
}
{
add.f16x2 r4440, r4441, r4442;
}
{
add.f16x2 r4443, r4444, r4440;
}
{
add.f16x2 r4446, r4435, r4436;
}
{
mul.f16x2 r4449, r4446, r4432;
}
{
add.f16x2 r4452, r4438, r4449;
}
{
sub.f16x2 r4455, r4441, r4442;
}
{
mul.f16x2 r4458, r4455, r4433;
}
{
add.f16x2 r4461, r4452, r4458;
}
{
add.f16x2 r4464, r4435, r4436;
}
{
mul.f16x2 r4467, r4464, r4432;
}
{
add.f16x2 r4470, r4438, r4467;
}
{
sub.f16x2 r4473, r4441, r4442;
}
{
mul.f16x2 r4476, r4473, r4433;
}
{
sub.f16x2 r4479, r4470, r4476;
}
{
add.f16x2 r4482, r4441, r4442;
}
{
mul.f16x2 r4485, r4482, r4432;
}
{
add.f16x2 r4488, r4444, r4485;
}
{
sub.f16x2 r4491, r4435, r4436;
}
{
mul.f16x2 r4494, r4491, r4433;
}
{
sub.f16x2 r4497, r4488, r4494;
}
{
add.f16x2 r4500, r4441, r4442;
}
{
mul.f16x2 r4503, r4500, r4432;
}
{
add.f16x2 r4506, r4444, r4503;
}
{
sub.f16x2 r4509, r4435, r4436;
}
{
mul.f16x2 r4512, r4509, r4433;
}
{
add.f16x2 r4515, r4506, r4512;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4518, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4519, {low, high};
}
{
add.f16x2 r4520, r4521, r4522;
}
{
add.f16x2 r4523, r4524, r4520;
}
{
add.f16x2 r4526, r4527, r4528;
}
{
add.f16x2 r4529, r4530, r4526;
}
{
add.f16x2 r4532, r4521, r4522;
}
{
mul.f16x2 r4535, r4532, r4518;
}
{
add.f16x2 r4538, r4524, r4535;
}
{
sub.f16x2 r4541, r4527, r4528;
}
{
mul.f16x2 r4544, r4541, r4519;
}
{
add.f16x2 r4547, r4538, r4544;
}
{
add.f16x2 r4550, r4521, r4522;
}
{
mul.f16x2 r4553, r4550, r4518;
}
{
add.f16x2 r4556, r4524, r4553;
}
{
sub.f16x2 r4559, r4527, r4528;
}
{
mul.f16x2 r4562, r4559, r4519;
}
{
sub.f16x2 r4565, r4556, r4562;
}
{
add.f16x2 r4568, r4527, r4528;
}
{
mul.f16x2 r4571, r4568, r4518;
}
{
add.f16x2 r4574, r4530, r4571;
}
{
sub.f16x2 r4577, r4521, r4522;
}
{
mul.f16x2 r4580, r4577, r4519;
}
{
sub.f16x2 r4583, r4574, r4580;
}
{
add.f16x2 r4586, r4527, r4528;
}
{
mul.f16x2 r4589, r4586, r4518;
}
{
add.f16x2 r4592, r4530, r4589;
}
{
sub.f16x2 r4595, r4521, r4522;
}
{
mul.f16x2 r4598, r4595, r4519;
}
{
add.f16x2 r4601, r4592, r4598;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4604, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4605, {low, high};
}
{
add.f16x2 r4606, r4607, r4608;
}
{
add.f16x2 r4609, r4610, r4606;
}
{
add.f16x2 r4612, r4613, r4614;
}
{
add.f16x2 r4615, r4616, r4612;
}
{
add.f16x2 r4618, r4607, r4608;
}
{
mul.f16x2 r4621, r4618, r4604;
}
{
add.f16x2 r4624, r4610, r4621;
}
{
sub.f16x2 r4627, r4613, r4614;
}
{
mul.f16x2 r4630, r4627, r4605;
}
{
add.f16x2 r4633, r4624, r4630;
}
{
add.f16x2 r4636, r4607, r4608;
}
{
mul.f16x2 r4639, r4636, r4604;
}
{
add.f16x2 r4642, r4610, r4639;
}
{
sub.f16x2 r4645, r4613, r4614;
}
{
mul.f16x2 r4648, r4645, r4605;
}
{
sub.f16x2 r4651, r4642, r4648;
}
{
add.f16x2 r4654, r4613, r4614;
}
{
mul.f16x2 r4657, r4654, r4604;
}
{
add.f16x2 r4660, r4616, r4657;
}
{
sub.f16x2 r4663, r4607, r4608;
}
{
mul.f16x2 r4666, r4663, r4605;
}
{
sub.f16x2 r4669, r4660, r4666;
}
{
add.f16x2 r4672, r4613, r4614;
}
{
mul.f16x2 r4675, r4672, r4604;
}
{
add.f16x2 r4678, r4616, r4675;
}
{
sub.f16x2 r4681, r4607, r4608;
}
{
mul.f16x2 r4684, r4681, r4605;
}
{
add.f16x2 r4687, r4678, r4684;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r4690, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r4691, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r4692, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r4693, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r4696, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
mov.b32 r4697, {low, high};
}
{
mul.f16x2 r4706, r4547, r4690;
}
{
mul.f16x2 r4709, r4583, r4691;
}
{
sub.f16x2 r4712, r4706, r4709;
}
{
mul.f16x2 r4715, r4547, r4691;
}
{
fma.rn.f16x2 r4718, r4583, r4690, r4715;
}
{
mul.f16x2 r4722, r4633, r4692;
}
{
mul.f16x2 r4725, r4669, r4693;
}
{
sub.f16x2 r4728, r4722, r4725;
}
{
mul.f16x2 r4731, r4633, r4693;
}
{
fma.rn.f16x2 r4734, r4669, r4692, r4731;
}
{
mul.f16x2 r4738, r4565, r4692;
}
{
mul.f16x2 r4741, r4601, r4693;
}
{
sub.f16x2 r4744, r4738, r4741;
}
{
mul.f16x2 r4747, r4565, r4693;
}
{
fma.rn.f16x2 r4750, r4601, r4692, r4747;
}
{
mul.f16x2 r4754, r4651, r4696;
}
{
mul.f16x2 r4757, r4687, r4697;
}
{
sub.f16x2 r4760, r4754, r4757;
}
{
mul.f16x2 r4763, r4651, r4697;
}
{
fma.rn.f16x2 r4766, r4687, r4696, r4763;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4770, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4771, {low, high};
}
{
add.f16x2 r4772, r4523, r4609;
}
{
add.f16x2 r4775, r4437, r4772;
}
{
add.f16x2 r4778, r4529, r4615;
}
{
add.f16x2 r4781, r4443, r4778;
}
{
add.f16x2 r4784, r4523, r4609;
}
{
mul.f16x2 r4787, r4784, r4770;
}
{
add.f16x2 r4790, r4437, r4787;
}
{
sub.f16x2 r4793, r4529, r4615;
}
{
mul.f16x2 r4796, r4793, r4771;
}
{
add.f16x2 r4799, r4790, r4796;
}
{
add.f16x2 r4802, r4523, r4609;
}
{
mul.f16x2 r4805, r4802, r4770;
}
{
add.f16x2 r4808, r4437, r4805;
}
{
sub.f16x2 r4811, r4529, r4615;
}
{
mul.f16x2 r4814, r4811, r4771;
}
{
sub.f16x2 r4817, r4808, r4814;
}
{
add.f16x2 r4820, r4529, r4615;
}
{
mul.f16x2 r4823, r4820, r4770;
}
{
add.f16x2 r4826, r4443, r4823;
}
{
sub.f16x2 r4829, r4523, r4609;
}
{
mul.f16x2 r4832, r4829, r4771;
}
{
sub.f16x2 r4835, r4826, r4832;
}
{
add.f16x2 r4838, r4529, r4615;
}
{
mul.f16x2 r4841, r4838, r4770;
}
{
add.f16x2 r4844, r4443, r4841;
}
{
sub.f16x2 r4847, r4523, r4609;
}
{
mul.f16x2 r4850, r4847, r4771;
}
{
add.f16x2 r4853, r4844, r4850;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4856, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4857, {low, high};
}
{
add.f16x2 r4858, r4712, r4728;
}
{
add.f16x2 r4861, r4461, r4858;
}
{
add.f16x2 r4864, r4718, r4734;
}
{
add.f16x2 r4867, r4497, r4864;
}
{
add.f16x2 r4870, r4712, r4728;
}
{
mul.f16x2 r4873, r4870, r4856;
}
{
add.f16x2 r4876, r4461, r4873;
}
{
sub.f16x2 r4879, r4718, r4734;
}
{
mul.f16x2 r4882, r4879, r4857;
}
{
add.f16x2 r4885, r4876, r4882;
}
{
add.f16x2 r4888, r4712, r4728;
}
{
mul.f16x2 r4891, r4888, r4856;
}
{
add.f16x2 r4894, r4461, r4891;
}
{
sub.f16x2 r4897, r4718, r4734;
}
{
mul.f16x2 r4900, r4897, r4857;
}
{
sub.f16x2 r4903, r4894, r4900;
}
{
add.f16x2 r4906, r4718, r4734;
}
{
mul.f16x2 r4909, r4906, r4856;
}
{
add.f16x2 r4912, r4497, r4909;
}
{
sub.f16x2 r4915, r4712, r4728;
}
{
mul.f16x2 r4918, r4915, r4857;
}
{
sub.f16x2 r4921, r4912, r4918;
}
{
add.f16x2 r4924, r4718, r4734;
}
{
mul.f16x2 r4927, r4924, r4856;
}
{
add.f16x2 r4930, r4497, r4927;
}
{
sub.f16x2 r4933, r4712, r4728;
}
{
mul.f16x2 r4936, r4933, r4857;
}
{
add.f16x2 r4939, r4930, r4936;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4942, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4943, {low, high};
}
{
add.f16x2 r4944, r4744, r4760;
}
{
add.f16x2 r4947, r4479, r4944;
}
{
add.f16x2 r4950, r4750, r4766;
}
{
add.f16x2 r4953, r4515, r4950;
}
{
add.f16x2 r4956, r4744, r4760;
}
{
mul.f16x2 r4959, r4956, r4942;
}
{
add.f16x2 r4962, r4479, r4959;
}
{
sub.f16x2 r4965, r4750, r4766;
}
{
mul.f16x2 r4968, r4965, r4943;
}
{
add.f16x2 r4971, r4962, r4968;
}
{
add.f16x2 r4974, r4744, r4760;
}
{
mul.f16x2 r4977, r4974, r4942;
}
{
add.f16x2 r4980, r4479, r4977;
}
{
sub.f16x2 r4983, r4750, r4766;
}
{
mul.f16x2 r4986, r4983, r4943;
}
{
sub.f16x2 r4989, r4980, r4986;
}
{
add.f16x2 r4992, r4750, r4766;
}
{
mul.f16x2 r4995, r4992, r4942;
}
{
add.f16x2 r4998, r4515, r4995;
}
{
sub.f16x2 r5001, r4744, r4760;
}
{
mul.f16x2 r5004, r5001, r4943;
}
{
sub.f16x2 r5007, r4998, r5004;
}
{
add.f16x2 r5010, r4750, r4766;
}
{
mul.f16x2 r5013, r5010, r4942;
}
{
add.f16x2 r5016, r4515, r5013;
}
{
sub.f16x2 r5019, r4744, r4760;
}
{
mul.f16x2 r5022, r5019, r4943;
}
{
add.f16x2 r5025, r5016, r5022;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5028, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5029, {low, high};
}
{
add.f16x2 r5030, r5031, r5032;
}
{
add.f16x2 r5033, r5034, r5030;
}
{
add.f16x2 r5036, r5037, r5038;
}
{
add.f16x2 r5039, r5040, r5036;
}
{
add.f16x2 r5042, r5031, r5032;
}
{
mul.f16x2 r5045, r5042, r5028;
}
{
add.f16x2 r5048, r5034, r5045;
}
{
sub.f16x2 r5051, r5037, r5038;
}
{
mul.f16x2 r5054, r5051, r5029;
}
{
add.f16x2 r5057, r5048, r5054;
}
{
add.f16x2 r5060, r5031, r5032;
}
{
mul.f16x2 r5063, r5060, r5028;
}
{
add.f16x2 r5066, r5034, r5063;
}
{
sub.f16x2 r5069, r5037, r5038;
}
{
mul.f16x2 r5072, r5069, r5029;
}
{
sub.f16x2 r5075, r5066, r5072;
}
{
add.f16x2 r5078, r5037, r5038;
}
{
mul.f16x2 r5081, r5078, r5028;
}
{
add.f16x2 r5084, r5040, r5081;
}
{
sub.f16x2 r5087, r5031, r5032;
}
{
mul.f16x2 r5090, r5087, r5029;
}
{
sub.f16x2 r5093, r5084, r5090;
}
{
add.f16x2 r5096, r5037, r5038;
}
{
mul.f16x2 r5099, r5096, r5028;
}
{
add.f16x2 r5102, r5040, r5099;
}
{
sub.f16x2 r5105, r5031, r5032;
}
{
mul.f16x2 r5108, r5105, r5029;
}
{
add.f16x2 r5111, r5102, r5108;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5114, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5115, {low, high};
}
{
add.f16x2 r5116, r5117, r5118;
}
{
add.f16x2 r5119, r5120, r5116;
}
{
add.f16x2 r5122, r5123, r5124;
}
{
add.f16x2 r5125, r5126, r5122;
}
{
add.f16x2 r5128, r5117, r5118;
}
{
mul.f16x2 r5131, r5128, r5114;
}
{
add.f16x2 r5134, r5120, r5131;
}
{
sub.f16x2 r5137, r5123, r5124;
}
{
mul.f16x2 r5140, r5137, r5115;
}
{
add.f16x2 r5143, r5134, r5140;
}
{
add.f16x2 r5146, r5117, r5118;
}
{
mul.f16x2 r5149, r5146, r5114;
}
{
add.f16x2 r5152, r5120, r5149;
}
{
sub.f16x2 r5155, r5123, r5124;
}
{
mul.f16x2 r5158, r5155, r5115;
}
{
sub.f16x2 r5161, r5152, r5158;
}
{
add.f16x2 r5164, r5123, r5124;
}
{
mul.f16x2 r5167, r5164, r5114;
}
{
add.f16x2 r5170, r5126, r5167;
}
{
sub.f16x2 r5173, r5117, r5118;
}
{
mul.f16x2 r5176, r5173, r5115;
}
{
sub.f16x2 r5179, r5170, r5176;
}
{
add.f16x2 r5182, r5123, r5124;
}
{
mul.f16x2 r5185, r5182, r5114;
}
{
add.f16x2 r5188, r5126, r5185;
}
{
sub.f16x2 r5191, r5117, r5118;
}
{
mul.f16x2 r5194, r5191, r5115;
}
{
add.f16x2 r5197, r5188, r5194;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5200, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5201, {low, high};
}
{
add.f16x2 r5202, r5203, r5204;
}
{
add.f16x2 r5205, r5206, r5202;
}
{
add.f16x2 r5208, r5209, r5210;
}
{
add.f16x2 r5211, r5212, r5208;
}
{
add.f16x2 r5214, r5203, r5204;
}
{
mul.f16x2 r5217, r5214, r5200;
}
{
add.f16x2 r5220, r5206, r5217;
}
{
sub.f16x2 r5223, r5209, r5210;
}
{
mul.f16x2 r5226, r5223, r5201;
}
{
add.f16x2 r5229, r5220, r5226;
}
{
add.f16x2 r5232, r5203, r5204;
}
{
mul.f16x2 r5235, r5232, r5200;
}
{
add.f16x2 r5238, r5206, r5235;
}
{
sub.f16x2 r5241, r5209, r5210;
}
{
mul.f16x2 r5244, r5241, r5201;
}
{
sub.f16x2 r5247, r5238, r5244;
}
{
add.f16x2 r5250, r5209, r5210;
}
{
mul.f16x2 r5253, r5250, r5200;
}
{
add.f16x2 r5256, r5212, r5253;
}
{
sub.f16x2 r5259, r5203, r5204;
}
{
mul.f16x2 r5262, r5259, r5201;
}
{
sub.f16x2 r5265, r5256, r5262;
}
{
add.f16x2 r5268, r5209, r5210;
}
{
mul.f16x2 r5271, r5268, r5200;
}
{
add.f16x2 r5274, r5212, r5271;
}
{
sub.f16x2 r5277, r5203, r5204;
}
{
mul.f16x2 r5280, r5277, r5201;
}
{
add.f16x2 r5283, r5274, r5280;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r5286, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r5287, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r5288, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r5289, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r5292, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
mov.b32 r5293, {low, high};
}
{
mul.f16x2 r5302, r5143, r5286;
}
{
mul.f16x2 r5305, r5179, r5287;
}
{
sub.f16x2 r5308, r5302, r5305;
}
{
mul.f16x2 r5311, r5143, r5287;
}
{
fma.rn.f16x2 r5314, r5179, r5286, r5311;
}
{
mul.f16x2 r5318, r5229, r5288;
}
{
mul.f16x2 r5321, r5265, r5289;
}
{
sub.f16x2 r5324, r5318, r5321;
}
{
mul.f16x2 r5327, r5229, r5289;
}
{
fma.rn.f16x2 r5330, r5265, r5288, r5327;
}
{
mul.f16x2 r5334, r5161, r5288;
}
{
mul.f16x2 r5337, r5197, r5289;
}
{
sub.f16x2 r5340, r5334, r5337;
}
{
mul.f16x2 r5343, r5161, r5289;
}
{
fma.rn.f16x2 r5346, r5197, r5288, r5343;
}
{
mul.f16x2 r5350, r5247, r5292;
}
{
mul.f16x2 r5353, r5283, r5293;
}
{
sub.f16x2 r5356, r5350, r5353;
}
{
mul.f16x2 r5359, r5247, r5293;
}
{
fma.rn.f16x2 r5362, r5283, r5292, r5359;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5366, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5367, {low, high};
}
{
add.f16x2 r5368, r5119, r5205;
}
{
add.f16x2 r5371, r5033, r5368;
}
{
add.f16x2 r5374, r5125, r5211;
}
{
add.f16x2 r5377, r5039, r5374;
}
{
add.f16x2 r5380, r5119, r5205;
}
{
mul.f16x2 r5383, r5380, r5366;
}
{
add.f16x2 r5386, r5033, r5383;
}
{
sub.f16x2 r5389, r5125, r5211;
}
{
mul.f16x2 r5392, r5389, r5367;
}
{
add.f16x2 r5395, r5386, r5392;
}
{
add.f16x2 r5398, r5119, r5205;
}
{
mul.f16x2 r5401, r5398, r5366;
}
{
add.f16x2 r5404, r5033, r5401;
}
{
sub.f16x2 r5407, r5125, r5211;
}
{
mul.f16x2 r5410, r5407, r5367;
}
{
sub.f16x2 r5413, r5404, r5410;
}
{
add.f16x2 r5416, r5125, r5211;
}
{
mul.f16x2 r5419, r5416, r5366;
}
{
add.f16x2 r5422, r5039, r5419;
}
{
sub.f16x2 r5425, r5119, r5205;
}
{
mul.f16x2 r5428, r5425, r5367;
}
{
sub.f16x2 r5431, r5422, r5428;
}
{
add.f16x2 r5434, r5125, r5211;
}
{
mul.f16x2 r5437, r5434, r5366;
}
{
add.f16x2 r5440, r5039, r5437;
}
{
sub.f16x2 r5443, r5119, r5205;
}
{
mul.f16x2 r5446, r5443, r5367;
}
{
add.f16x2 r5449, r5440, r5446;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5452, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5453, {low, high};
}
{
add.f16x2 r5454, r5308, r5324;
}
{
add.f16x2 r5457, r5057, r5454;
}
{
add.f16x2 r5460, r5314, r5330;
}
{
add.f16x2 r5463, r5093, r5460;
}
{
add.f16x2 r5466, r5308, r5324;
}
{
mul.f16x2 r5469, r5466, r5452;
}
{
add.f16x2 r5472, r5057, r5469;
}
{
sub.f16x2 r5475, r5314, r5330;
}
{
mul.f16x2 r5478, r5475, r5453;
}
{
add.f16x2 r5481, r5472, r5478;
}
{
add.f16x2 r5484, r5308, r5324;
}
{
mul.f16x2 r5487, r5484, r5452;
}
{
add.f16x2 r5490, r5057, r5487;
}
{
sub.f16x2 r5493, r5314, r5330;
}
{
mul.f16x2 r5496, r5493, r5453;
}
{
sub.f16x2 r5499, r5490, r5496;
}
{
add.f16x2 r5502, r5314, r5330;
}
{
mul.f16x2 r5505, r5502, r5452;
}
{
add.f16x2 r5508, r5093, r5505;
}
{
sub.f16x2 r5511, r5308, r5324;
}
{
mul.f16x2 r5514, r5511, r5453;
}
{
sub.f16x2 r5517, r5508, r5514;
}
{
add.f16x2 r5520, r5314, r5330;
}
{
mul.f16x2 r5523, r5520, r5452;
}
{
add.f16x2 r5526, r5093, r5523;
}
{
sub.f16x2 r5529, r5308, r5324;
}
{
mul.f16x2 r5532, r5529, r5453;
}
{
add.f16x2 r5535, r5526, r5532;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5538, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5539, {low, high};
}
{
add.f16x2 r5540, r5340, r5356;
}
{
add.f16x2 r5543, r5075, r5540;
}
{
add.f16x2 r5546, r5346, r5362;
}
{
add.f16x2 r5549, r5111, r5546;
}
{
add.f16x2 r5552, r5340, r5356;
}
{
mul.f16x2 r5555, r5552, r5538;
}
{
add.f16x2 r5558, r5075, r5555;
}
{
sub.f16x2 r5561, r5346, r5362;
}
{
mul.f16x2 r5564, r5561, r5539;
}
{
add.f16x2 r5567, r5558, r5564;
}
{
add.f16x2 r5570, r5340, r5356;
}
{
mul.f16x2 r5573, r5570, r5538;
}
{
add.f16x2 r5576, r5075, r5573;
}
{
sub.f16x2 r5579, r5346, r5362;
}
{
mul.f16x2 r5582, r5579, r5539;
}
{
sub.f16x2 r5585, r5576, r5582;
}
{
add.f16x2 r5588, r5346, r5362;
}
{
mul.f16x2 r5591, r5588, r5538;
}
{
add.f16x2 r5594, r5111, r5591;
}
{
sub.f16x2 r5597, r5340, r5356;
}
{
mul.f16x2 r5600, r5597, r5539;
}
{
sub.f16x2 r5603, r5594, r5600;
}
{
add.f16x2 r5606, r5346, r5362;
}
{
mul.f16x2 r5609, r5606, r5538;
}
{
add.f16x2 r5612, r5111, r5609;
}
{
sub.f16x2 r5615, r5340, r5356;
}
{
mul.f16x2 r5618, r5615, r5539;
}
{
add.f16x2 r5621, r5612, r5618;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f534;
cvt.rn.f16.f32 high, f534;
mov.b32 r5624, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f536;
cvt.rn.f16.f32 high, f536;
mov.b32 r5625, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f538;
cvt.rn.f16.f32 high, f538;
mov.b32 r5626, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f540;
cvt.rn.f16.f32 high, f540;
mov.b32 r5627, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r5628, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r5629, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f546;
cvt.rn.f16.f32 high, f546;
mov.b32 r5630, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f548;
cvt.rn.f16.f32 high, f548;
mov.b32 r5631, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f550;
cvt.rn.f16.f32 high, f550;
mov.b32 r5632, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f552;
cvt.rn.f16.f32 high, f552;
mov.b32 r5633, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r5634, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r5635, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f558;
cvt.rn.f16.f32 high, f558;
mov.b32 r5636, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f560;
cvt.rn.f16.f32 high, f560;
mov.b32 r5637, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f562;
cvt.rn.f16.f32 high, f562;
mov.b32 r5638, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f564;
cvt.rn.f16.f32 high, f564;
mov.b32 r5639, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f570;
cvt.rn.f16.f32 high, f570;
mov.b32 r5642, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f572;
cvt.rn.f16.f32 high, f572;
mov.b32 r5643, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r5646, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
mov.b32 r5647, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f586;
cvt.rn.f16.f32 high, f586;
mov.b32 r5650, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f588;
cvt.rn.f16.f32 high, f588;
mov.b32 r5651, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5654, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5655, {low, high};
}
{
mul.f16x2 r5676, r4861, r5624;
}
{
mul.f16x2 r5679, r4867, r5625;
}
{
sub.f16x2 r5682, r5676, r5679;
}
{
mul.f16x2 r5685, r4861, r5625;
}
{
fma.rn.f16x2 r5688, r4867, r5624, r5685;
}
{
mul.f16x2 r5692, r5457, r5626;
}
{
mul.f16x2 r5695, r5463, r5627;
}
{
sub.f16x2 r5698, r5692, r5695;
}
{
mul.f16x2 r5701, r5457, r5627;
}
{
fma.rn.f16x2 r5704, r5463, r5626, r5701;
}
{
mul.f16x2 r5708, r4947, r5626;
}
{
mul.f16x2 r5711, r4953, r5627;
}
{
sub.f16x2 r5714, r5708, r5711;
}
{
mul.f16x2 r5717, r4947, r5627;
}
{
fma.rn.f16x2 r5720, r4953, r5626, r5717;
}
{
mul.f16x2 r5724, r5543, r5630;
}
{
mul.f16x2 r5727, r5549, r5631;
}
{
sub.f16x2 r5730, r5724, r5727;
}
{
mul.f16x2 r5733, r5543, r5631;
}
{
fma.rn.f16x2 r5736, r5549, r5630, r5733;
}
{
mul.f16x2 r5740, r4799, r5628;
}
{
mul.f16x2 r5743, r4835, r5629;
}
{
sub.f16x2 r5746, r5740, r5743;
}
{
mul.f16x2 r5749, r4799, r5629;
}
{
fma.rn.f16x2 r5752, r4835, r5628, r5749;
}
{
mul.f16x2 r5756, r5395, r5634;
}
{
mul.f16x2 r5759, r5431, r5635;
}
{
sub.f16x2 r5762, r5756, r5759;
}
{
mul.f16x2 r5765, r5395, r5635;
}
{
fma.rn.f16x2 r5768, r5431, r5634, r5765;
}
{
mul.f16x2 r5772, r4885, r5630;
}
{
mul.f16x2 r5775, r4921, r5631;
}
{
sub.f16x2 r5778, r5772, r5775;
}
{
mul.f16x2 r5781, r4885, r5631;
}
{
fma.rn.f16x2 r5784, r4921, r5630, r5781;
}
{
mul.f16x2 r5788, r5481, r5638;
}
{
mul.f16x2 r5791, r5517, r5639;
}
{
sub.f16x2 r5794, r5788, r5791;
}
{
mul.f16x2 r5797, r5481, r5639;
}
{
fma.rn.f16x2 r5800, r5517, r5638, r5797;
}
{
mul.f16x2 r5804, r4971, r5632;
}
{
mul.f16x2 r5807, r5007, r5633;
}
{
sub.f16x2 r5810, r5804, r5807;
}
{
mul.f16x2 r5813, r4971, r5633;
}
{
fma.rn.f16x2 r5816, r5007, r5632, r5813;
}
{
mul.f16x2 r5820, r5567, r5642;
}
{
mul.f16x2 r5823, r5603, r5643;
}
{
sub.f16x2 r5826, r5820, r5823;
}
{
mul.f16x2 r5829, r5567, r5643;
}
{
fma.rn.f16x2 r5832, r5603, r5642, r5829;
}
{
mul.f16x2 r5836, r4817, r5634;
}
{
mul.f16x2 r5839, r4853, r5635;
}
{
sub.f16x2 r5842, r5836, r5839;
}
{
mul.f16x2 r5845, r4817, r5635;
}
{
fma.rn.f16x2 r5848, r4853, r5634, r5845;
}
{
mul.f16x2 r5852, r5413, r5646;
}
{
mul.f16x2 r5855, r5449, r5647;
}
{
sub.f16x2 r5858, r5852, r5855;
}
{
mul.f16x2 r5861, r5413, r5647;
}
{
fma.rn.f16x2 r5864, r5449, r5646, r5861;
}
{
mul.f16x2 r5868, r4903, r5636;
}
{
mul.f16x2 r5871, r4939, r5637;
}
{
sub.f16x2 r5874, r5868, r5871;
}
{
mul.f16x2 r5877, r4903, r5637;
}
{
fma.rn.f16x2 r5880, r4939, r5636, r5877;
}
{
mul.f16x2 r5884, r5499, r5650;
}
{
mul.f16x2 r5887, r5535, r5651;
}
{
sub.f16x2 r5890, r5884, r5887;
}
{
mul.f16x2 r5893, r5499, r5651;
}
{
fma.rn.f16x2 r5896, r5535, r5650, r5893;
}
{
mul.f16x2 r5900, r4989, r5638;
}
{
mul.f16x2 r5903, r5025, r5639;
}
{
sub.f16x2 r5906, r5900, r5903;
}
{
mul.f16x2 r5909, r4989, r5639;
}
{
fma.rn.f16x2 r5912, r5025, r5638, r5909;
}
{
mul.f16x2 r5916, r5585, r5654;
}
{
mul.f16x2 r5919, r5621, r5655;
}
{
sub.f16x2 r5922, r5916, r5919;
}
{
mul.f16x2 r5925, r5585, r5655;
}
{
fma.rn.f16x2 r5928, r5621, r5654, r5925;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5932, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5933, {low, high};
}
{
add.f16x2 r5934, r4775, r5371;
}
{
add.f16x2 r5937, r4179, r5934;
}
{
add.f16x2 r5940, r4781, r5377;
}
{
add.f16x2 r5943, r4185, r5940;
}
{
add.f16x2 r5946, r4775, r5371;
}
{
mul.f16x2 r5949, r5946, r5932;
}
{
add.f16x2 r5952, r4179, r5949;
}
{
sub.f16x2 r5955, r4781, r5377;
}
{
mul.f16x2 r5958, r5955, r5933;
}
{
add.f16x2 r5961, r5952, r5958;
}
{
add.f16x2 r5964, r4775, r5371;
}
{
mul.f16x2 r5967, r5964, r5932;
}
{
add.f16x2 r5970, r4179, r5967;
}
{
sub.f16x2 r5973, r4781, r5377;
}
{
mul.f16x2 r5976, r5973, r5933;
}
{
sub.f16x2 r5979, r5970, r5976;
}
{
add.f16x2 r5982, r4781, r5377;
}
{
mul.f16x2 r5985, r5982, r5932;
}
{
add.f16x2 r5988, r4185, r5985;
}
{
sub.f16x2 r5991, r4775, r5371;
}
{
mul.f16x2 r5994, r5991, r5933;
}
{
sub.f16x2 r5997, r5988, r5994;
}
{
add.f16x2 r6000, r4781, r5377;
}
{
mul.f16x2 r6003, r6000, r5932;
}
{
add.f16x2 r6006, r4185, r6003;
}
{
sub.f16x2 r6009, r4775, r5371;
}
{
mul.f16x2 r6012, r6009, r5933;
}
{
add.f16x2 r6015, r6006, r6012;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6019, {low, high};
}
{
add.f16x2 r6020, r5682, r5698;
}
{
add.f16x2 r6023, r4265, r6020;
}
{
add.f16x2 r6026, r5688, r5704;
}
{
add.f16x2 r6029, r4271, r6026;
}
{
add.f16x2 r6032, r5682, r5698;
}
{
mul.f16x2 r6035, r6032, r6018;
}
{
add.f16x2 r6038, r4265, r6035;
}
{
sub.f16x2 r6041, r5688, r5704;
}
{
mul.f16x2 r6044, r6041, r6019;
}
{
add.f16x2 r6047, r6038, r6044;
}
{
add.f16x2 r6050, r5682, r5698;
}
{
mul.f16x2 r6053, r6050, r6018;
}
{
add.f16x2 r6056, r4265, r6053;
}
{
sub.f16x2 r6059, r5688, r5704;
}
{
mul.f16x2 r6062, r6059, r6019;
}
{
sub.f16x2 r6065, r6056, r6062;
}
{
add.f16x2 r6068, r5688, r5704;
}
{
mul.f16x2 r6071, r6068, r6018;
}
{
add.f16x2 r6074, r4271, r6071;
}
{
sub.f16x2 r6077, r5682, r5698;
}
{
mul.f16x2 r6080, r6077, r6019;
}
{
sub.f16x2 r6083, r6074, r6080;
}
{
add.f16x2 r6086, r5688, r5704;
}
{
mul.f16x2 r6089, r6086, r6018;
}
{
add.f16x2 r6092, r4271, r6089;
}
{
sub.f16x2 r6095, r5682, r5698;
}
{
mul.f16x2 r6098, r6095, r6019;
}
{
add.f16x2 r6101, r6092, r6098;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6104, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6105, {low, high};
}
{
add.f16x2 r6106, r5714, r5730;
}
{
add.f16x2 r6109, r4351, r6106;
}
{
add.f16x2 r6112, r5720, r5736;
}
{
add.f16x2 r6115, r4357, r6112;
}
{
add.f16x2 r6118, r5714, r5730;
}
{
mul.f16x2 r6121, r6118, r6104;
}
{
add.f16x2 r6124, r4351, r6121;
}
{
sub.f16x2 r6127, r5720, r5736;
}
{
mul.f16x2 r6130, r6127, r6105;
}
{
add.f16x2 r6133, r6124, r6130;
}
{
add.f16x2 r6136, r5714, r5730;
}
{
mul.f16x2 r6139, r6136, r6104;
}
{
add.f16x2 r6142, r4351, r6139;
}
{
sub.f16x2 r6145, r5720, r5736;
}
{
mul.f16x2 r6148, r6145, r6105;
}
{
sub.f16x2 r6151, r6142, r6148;
}
{
add.f16x2 r6154, r5720, r5736;
}
{
mul.f16x2 r6157, r6154, r6104;
}
{
add.f16x2 r6160, r4357, r6157;
}
{
sub.f16x2 r6163, r5714, r5730;
}
{
mul.f16x2 r6166, r6163, r6105;
}
{
sub.f16x2 r6169, r6160, r6166;
}
{
add.f16x2 r6172, r5720, r5736;
}
{
mul.f16x2 r6175, r6172, r6104;
}
{
add.f16x2 r6178, r4357, r6175;
}
{
sub.f16x2 r6181, r5714, r5730;
}
{
mul.f16x2 r6184, r6181, r6105;
}
{
add.f16x2 r6187, r6178, r6184;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6190, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6191, {low, high};
}
{
add.f16x2 r6192, r5746, r5762;
}
{
add.f16x2 r6195, r4203, r6192;
}
{
add.f16x2 r6198, r5752, r5768;
}
{
add.f16x2 r6201, r4239, r6198;
}
{
add.f16x2 r6204, r5746, r5762;
}
{
mul.f16x2 r6207, r6204, r6190;
}
{
add.f16x2 r6210, r4203, r6207;
}
{
sub.f16x2 r6213, r5752, r5768;
}
{
mul.f16x2 r6216, r6213, r6191;
}
{
add.f16x2 r6219, r6210, r6216;
}
{
add.f16x2 r6222, r5746, r5762;
}
{
mul.f16x2 r6225, r6222, r6190;
}
{
add.f16x2 r6228, r4203, r6225;
}
{
sub.f16x2 r6231, r5752, r5768;
}
{
mul.f16x2 r6234, r6231, r6191;
}
{
sub.f16x2 r6237, r6228, r6234;
}
{
add.f16x2 r6240, r5752, r5768;
}
{
mul.f16x2 r6243, r6240, r6190;
}
{
add.f16x2 r6246, r4239, r6243;
}
{
sub.f16x2 r6249, r5746, r5762;
}
{
mul.f16x2 r6252, r6249, r6191;
}
{
sub.f16x2 r6255, r6246, r6252;
}
{
add.f16x2 r6258, r5752, r5768;
}
{
mul.f16x2 r6261, r6258, r6190;
}
{
add.f16x2 r6264, r4239, r6261;
}
{
sub.f16x2 r6267, r5746, r5762;
}
{
mul.f16x2 r6270, r6267, r6191;
}
{
add.f16x2 r6273, r6264, r6270;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6276, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6277, {low, high};
}
{
add.f16x2 r6278, r5778, r5794;
}
{
add.f16x2 r6281, r4289, r6278;
}
{
add.f16x2 r6284, r5784, r5800;
}
{
add.f16x2 r6287, r4325, r6284;
}
{
add.f16x2 r6290, r5778, r5794;
}
{
mul.f16x2 r6293, r6290, r6276;
}
{
add.f16x2 r6296, r4289, r6293;
}
{
sub.f16x2 r6299, r5784, r5800;
}
{
mul.f16x2 r6302, r6299, r6277;
}
{
add.f16x2 r6305, r6296, r6302;
}
{
add.f16x2 r6308, r5778, r5794;
}
{
mul.f16x2 r6311, r6308, r6276;
}
{
add.f16x2 r6314, r4289, r6311;
}
{
sub.f16x2 r6317, r5784, r5800;
}
{
mul.f16x2 r6320, r6317, r6277;
}
{
sub.f16x2 r6323, r6314, r6320;
}
{
add.f16x2 r6326, r5784, r5800;
}
{
mul.f16x2 r6329, r6326, r6276;
}
{
add.f16x2 r6332, r4325, r6329;
}
{
sub.f16x2 r6335, r5778, r5794;
}
{
mul.f16x2 r6338, r6335, r6277;
}
{
sub.f16x2 r6341, r6332, r6338;
}
{
add.f16x2 r6344, r5784, r5800;
}
{
mul.f16x2 r6347, r6344, r6276;
}
{
add.f16x2 r6350, r4325, r6347;
}
{
sub.f16x2 r6353, r5778, r5794;
}
{
mul.f16x2 r6356, r6353, r6277;
}
{
add.f16x2 r6359, r6350, r6356;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6362, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6363, {low, high};
}
{
add.f16x2 r6364, r5810, r5826;
}
{
add.f16x2 r6367, r4375, r6364;
}
{
add.f16x2 r6370, r5816, r5832;
}
{
add.f16x2 r6373, r4411, r6370;
}
{
add.f16x2 r6376, r5810, r5826;
}
{
mul.f16x2 r6379, r6376, r6362;
}
{
add.f16x2 r6382, r4375, r6379;
}
{
sub.f16x2 r6385, r5816, r5832;
}
{
mul.f16x2 r6388, r6385, r6363;
}
{
add.f16x2 r6391, r6382, r6388;
}
{
add.f16x2 r6394, r5810, r5826;
}
{
mul.f16x2 r6397, r6394, r6362;
}
{
add.f16x2 r6400, r4375, r6397;
}
{
sub.f16x2 r6403, r5816, r5832;
}
{
mul.f16x2 r6406, r6403, r6363;
}
{
sub.f16x2 r6409, r6400, r6406;
}
{
add.f16x2 r6412, r5816, r5832;
}
{
mul.f16x2 r6415, r6412, r6362;
}
{
add.f16x2 r6418, r4411, r6415;
}
{
sub.f16x2 r6421, r5810, r5826;
}
{
mul.f16x2 r6424, r6421, r6363;
}
{
sub.f16x2 r6427, r6418, r6424;
}
{
add.f16x2 r6430, r5816, r5832;
}
{
mul.f16x2 r6433, r6430, r6362;
}
{
add.f16x2 r6436, r4411, r6433;
}
{
sub.f16x2 r6439, r5810, r5826;
}
{
mul.f16x2 r6442, r6439, r6363;
}
{
add.f16x2 r6445, r6436, r6442;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6448, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6449, {low, high};
}
{
add.f16x2 r6450, r5842, r5858;
}
{
add.f16x2 r6453, r4221, r6450;
}
{
add.f16x2 r6456, r5848, r5864;
}
{
add.f16x2 r6459, r4257, r6456;
}
{
add.f16x2 r6462, r5842, r5858;
}
{
mul.f16x2 r6465, r6462, r6448;
}
{
add.f16x2 r6468, r4221, r6465;
}
{
sub.f16x2 r6471, r5848, r5864;
}
{
mul.f16x2 r6474, r6471, r6449;
}
{
add.f16x2 r6477, r6468, r6474;
}
{
add.f16x2 r6480, r5842, r5858;
}
{
mul.f16x2 r6483, r6480, r6448;
}
{
add.f16x2 r6486, r4221, r6483;
}
{
sub.f16x2 r6489, r5848, r5864;
}
{
mul.f16x2 r6492, r6489, r6449;
}
{
sub.f16x2 r6495, r6486, r6492;
}
{
add.f16x2 r6498, r5848, r5864;
}
{
mul.f16x2 r6501, r6498, r6448;
}
{
add.f16x2 r6504, r4257, r6501;
}
{
sub.f16x2 r6507, r5842, r5858;
}
{
mul.f16x2 r6510, r6507, r6449;
}
{
sub.f16x2 r6513, r6504, r6510;
}
{
add.f16x2 r6516, r5848, r5864;
}
{
mul.f16x2 r6519, r6516, r6448;
}
{
add.f16x2 r6522, r4257, r6519;
}
{
sub.f16x2 r6525, r5842, r5858;
}
{
mul.f16x2 r6528, r6525, r6449;
}
{
add.f16x2 r6531, r6522, r6528;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6534, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6535, {low, high};
}
{
add.f16x2 r6536, r5874, r5890;
}
{
add.f16x2 r6539, r4307, r6536;
}
{
add.f16x2 r6542, r5880, r5896;
}
{
add.f16x2 r6545, r4343, r6542;
}
{
add.f16x2 r6548, r5874, r5890;
}
{
mul.f16x2 r6551, r6548, r6534;
}
{
add.f16x2 r6554, r4307, r6551;
}
{
sub.f16x2 r6557, r5880, r5896;
}
{
mul.f16x2 r6560, r6557, r6535;
}
{
add.f16x2 r6563, r6554, r6560;
}
{
add.f16x2 r6566, r5874, r5890;
}
{
mul.f16x2 r6569, r6566, r6534;
}
{
add.f16x2 r6572, r4307, r6569;
}
{
sub.f16x2 r6575, r5880, r5896;
}
{
mul.f16x2 r6578, r6575, r6535;
}
{
sub.f16x2 r6581, r6572, r6578;
}
{
add.f16x2 r6584, r5880, r5896;
}
{
mul.f16x2 r6587, r6584, r6534;
}
{
add.f16x2 r6590, r4343, r6587;
}
{
sub.f16x2 r6593, r5874, r5890;
}
{
mul.f16x2 r6596, r6593, r6535;
}
{
sub.f16x2 r6599, r6590, r6596;
}
{
add.f16x2 r6602, r5880, r5896;
}
{
mul.f16x2 r6605, r6602, r6534;
}
{
add.f16x2 r6608, r4343, r6605;
}
{
sub.f16x2 r6611, r5874, r5890;
}
{
mul.f16x2 r6614, r6611, r6535;
}
{
add.f16x2 r6617, r6608, r6614;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6620, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6621, {low, high};
}
{
add.f16x2 r6622, r5906, r5922;
}
{
add.f16x2 r6625, r4393, r6622;
}
{
add.f16x2 r6628, r5912, r5928;
}
{
add.f16x2 r6631, r4429, r6628;
}
{
add.f16x2 r6634, r5906, r5922;
}
{
mul.f16x2 r6637, r6634, r6620;
}
{
add.f16x2 r6640, r4393, r6637;
}
{
sub.f16x2 r6643, r5912, r5928;
}
{
mul.f16x2 r6646, r6643, r6621;
}
{
add.f16x2 r6649, r6640, r6646;
}
{
add.f16x2 r6652, r5906, r5922;
}
{
mul.f16x2 r6655, r6652, r6620;
}
{
add.f16x2 r6658, r4393, r6655;
}
{
sub.f16x2 r6661, r5912, r5928;
}
{
mul.f16x2 r6664, r6661, r6621;
}
{
sub.f16x2 r6667, r6658, r6664;
}
{
add.f16x2 r6670, r5912, r5928;
}
{
mul.f16x2 r6673, r6670, r6620;
}
{
add.f16x2 r6676, r4429, r6673;
}
{
sub.f16x2 r6679, r5906, r5922;
}
{
mul.f16x2 r6682, r6679, r6621;
}
{
sub.f16x2 r6685, r6676, r6682;
}
{
add.f16x2 r6688, r5912, r5928;
}
{
mul.f16x2 r6691, r6688, r6620;
}
{
add.f16x2 r6694, r4429, r6691;
}
{
sub.f16x2 r6697, r5906, r5922;
}
{
mul.f16x2 r6700, r6697, r6621;
}
{
add.f16x2 r6703, r6694, r6700;
}
mul.wide.u32 rd4, r8451, 795364315;
shr.u64 rd5, rd4, 32;
cvt.u32.u64 r8455, rd5;
sub.s32 r8456, r8451, r8455;
shr.u32 r8457, r8456, 1;
add.s32 r8458, r8457, r8455;
shr.u32 r8459, r8458, 4;
cvt.rn.f32.u32 f768, r8459;
mul.f32 f769, f768, 0f3D9EDD1F;
cos.approx.f32 f673, f769;
sin.approx.f32 f770, f769;
neg.f32 f674, f770;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f673;
cvt.rn.f16.f32 high, f674;
mov.b32 r6706, {low, high};
}
mul.lo.s32 r8460, r8459, 27;
sub.s32 r8461, r8451, r8460;
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6709, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6711, {high, high};
}
{
mul.f16x2 r6713, r6029, r6711;
}
{
fma.rn.f16x2 r6716, r6023, r6709, r6713;
}
{
mul.f16x2 r6720, r6023, r6711;
}
{
neg.f16x2 r6723, r6720;
}
{
fma.rn.f16x2 r6725, r6029, r6709, r6723;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6729, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6731, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6733, {low, high};
}
{
mul.f16x2 r6734, r6731, r6733;
}
{
mul.f16x2 r6737, r6706, r6729;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6740, {high, low};
}
{
fma.rn.f16x2 r6742, r6734, r6740, r6737;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6742;
mov.b32 r6746, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6742;
mov.b32 r6748, {high, high};
}
{
mul.f16x2 r6750, r6115, r6748;
}
{
fma.rn.f16x2 r6753, r6109, r6746, r6750;
}
{
mul.f16x2 r6757, r6109, r6748;
}
{
neg.f16x2 r6760, r6757;
}
{
fma.rn.f16x2 r6762, r6115, r6746, r6760;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6766, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6768, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6770, {low, high};
}
{
mul.f16x2 r6771, r6768, r6770;
}
{
mul.f16x2 r6774, r6742, r6766;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6742;
mov.b32 r6777, {high, low};
}
{
fma.rn.f16x2 r6779, r6771, r6777, r6774;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6779;
mov.b32 r6783, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6779;
mov.b32 r6785, {high, high};
}
{
mul.f16x2 r6787, r6201, r6785;
}
{
fma.rn.f16x2 r6790, r6195, r6783, r6787;
}
{
mul.f16x2 r6794, r6195, r6785;
}
{
neg.f16x2 r6797, r6794;
}
{
fma.rn.f16x2 r6799, r6201, r6783, r6797;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6803, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6805, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6807, {low, high};
}
{
mul.f16x2 r6808, r6805, r6807;
}
{
mul.f16x2 r6811, r6779, r6803;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6779;
mov.b32 r6814, {high, low};
}
{
fma.rn.f16x2 r6816, r6808, r6814, r6811;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6816;
mov.b32 r6820, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6816;
mov.b32 r6822, {high, high};
}
{
mul.f16x2 r6824, r6287, r6822;
}
{
fma.rn.f16x2 r6827, r6281, r6820, r6824;
}
{
mul.f16x2 r6831, r6281, r6822;
}
{
neg.f16x2 r6834, r6831;
}
{
fma.rn.f16x2 r6836, r6287, r6820, r6834;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6840, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6842, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6844, {low, high};
}
{
mul.f16x2 r6845, r6842, r6844;
}
{
mul.f16x2 r6848, r6816, r6840;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6816;
mov.b32 r6851, {high, low};
}
{
fma.rn.f16x2 r6853, r6845, r6851, r6848;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6853;
mov.b32 r6857, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6853;
mov.b32 r6859, {high, high};
}
{
mul.f16x2 r6861, r6373, r6859;
}
{
fma.rn.f16x2 r6864, r6367, r6857, r6861;
}
{
mul.f16x2 r6868, r6367, r6859;
}
{
neg.f16x2 r6871, r6868;
}
{
fma.rn.f16x2 r6873, r6373, r6857, r6871;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6877, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6879, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6881, {low, high};
}
{
mul.f16x2 r6882, r6879, r6881;
}
{
mul.f16x2 r6885, r6853, r6877;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6853;
mov.b32 r6888, {high, low};
}
{
fma.rn.f16x2 r6890, r6882, r6888, r6885;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6890;
mov.b32 r6894, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6890;
mov.b32 r6896, {high, high};
}
{
mul.f16x2 r6898, r6459, r6896;
}
{
fma.rn.f16x2 r6901, r6453, r6894, r6898;
}
{
mul.f16x2 r6905, r6453, r6896;
}
{
neg.f16x2 r6908, r6905;
}
{
fma.rn.f16x2 r6910, r6459, r6894, r6908;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6914, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6916, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6918, {low, high};
}
{
mul.f16x2 r6919, r6916, r6918;
}
{
mul.f16x2 r6922, r6890, r6914;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6890;
mov.b32 r6925, {high, low};
}
{
fma.rn.f16x2 r6927, r6919, r6925, r6922;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6927;
mov.b32 r6931, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6927;
mov.b32 r6933, {high, high};
}
{
mul.f16x2 r6935, r6545, r6933;
}
{
fma.rn.f16x2 r6938, r6539, r6931, r6935;
}
{
mul.f16x2 r6942, r6539, r6933;
}
{
neg.f16x2 r6945, r6942;
}
{
fma.rn.f16x2 r6947, r6545, r6931, r6945;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6951, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6953, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6955, {low, high};
}
{
mul.f16x2 r6956, r6953, r6955;
}
{
mul.f16x2 r6959, r6927, r6951;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6927;
mov.b32 r6962, {high, low};
}
{
fma.rn.f16x2 r6964, r6956, r6962, r6959;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6964;
mov.b32 r6968, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6964;
mov.b32 r6970, {high, high};
}
{
mul.f16x2 r6972, r6631, r6970;
}
{
fma.rn.f16x2 r6975, r6625, r6968, r6972;
}
{
mul.f16x2 r6979, r6625, r6970;
}
{
neg.f16x2 r6982, r6979;
}
{
fma.rn.f16x2 r6984, r6631, r6968, r6982;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6988, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6990, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6992, {low, high};
}
{
mul.f16x2 r6993, r6990, r6992;
}
{
mul.f16x2 r6996, r6964, r6988;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6964;
mov.b32 r6999, {high, low};
}
{
fma.rn.f16x2 r7001, r6993, r6999, r6996;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7001;
mov.b32 r7005, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7001;
mov.b32 r7007, {high, high};
}
{
mul.f16x2 r7009, r5997, r7007;
}
{
fma.rn.f16x2 r7012, r5961, r7005, r7009;
}
{
mul.f16x2 r7016, r5961, r7007;
}
{
neg.f16x2 r7019, r7016;
}
{
fma.rn.f16x2 r7021, r5997, r7005, r7019;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7025, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7027, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7029, {low, high};
}
{
mul.f16x2 r7030, r7027, r7029;
}
{
mul.f16x2 r7033, r7001, r7025;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7001;
mov.b32 r7036, {high, low};
}
{
fma.rn.f16x2 r7038, r7030, r7036, r7033;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7038;
mov.b32 r7042, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7038;
mov.b32 r7044, {high, high};
}
{
mul.f16x2 r7046, r6083, r7044;
}
{
fma.rn.f16x2 r7049, r6047, r7042, r7046;
}
{
mul.f16x2 r7053, r6047, r7044;
}
{
neg.f16x2 r7056, r7053;
}
{
fma.rn.f16x2 r7058, r6083, r7042, r7056;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7062, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7064, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7066, {low, high};
}
{
mul.f16x2 r7067, r7064, r7066;
}
{
mul.f16x2 r7070, r7038, r7062;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7038;
mov.b32 r7073, {high, low};
}
{
fma.rn.f16x2 r7075, r7067, r7073, r7070;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7075;
mov.b32 r7079, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7075;
mov.b32 r7081, {high, high};
}
{
mul.f16x2 r7083, r6169, r7081;
}
{
fma.rn.f16x2 r7086, r6133, r7079, r7083;
}
{
mul.f16x2 r7090, r6133, r7081;
}
{
neg.f16x2 r7093, r7090;
}
{
fma.rn.f16x2 r7095, r6169, r7079, r7093;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7099, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7101, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7103, {low, high};
}
{
mul.f16x2 r7104, r7101, r7103;
}
{
mul.f16x2 r7107, r7075, r7099;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7075;
mov.b32 r7110, {high, low};
}
{
fma.rn.f16x2 r7112, r7104, r7110, r7107;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7112;
mov.b32 r7116, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7112;
mov.b32 r7118, {high, high};
}
{
mul.f16x2 r7120, r6255, r7118;
}
{
fma.rn.f16x2 r7123, r6219, r7116, r7120;
}
{
mul.f16x2 r7127, r6219, r7118;
}
{
neg.f16x2 r7130, r7127;
}
{
fma.rn.f16x2 r7132, r6255, r7116, r7130;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7136, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7138, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7140, {low, high};
}
{
mul.f16x2 r7141, r7138, r7140;
}
{
mul.f16x2 r7144, r7112, r7136;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7112;
mov.b32 r7147, {high, low};
}
{
fma.rn.f16x2 r7149, r7141, r7147, r7144;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7149;
mov.b32 r7153, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7149;
mov.b32 r7155, {high, high};
}
{
mul.f16x2 r7157, r6341, r7155;
}
{
fma.rn.f16x2 r7160, r6305, r7153, r7157;
}
{
mul.f16x2 r7164, r6305, r7155;
}
{
neg.f16x2 r7167, r7164;
}
{
fma.rn.f16x2 r7169, r6341, r7153, r7167;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7173, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7175, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7177, {low, high};
}
{
mul.f16x2 r7178, r7175, r7177;
}
{
mul.f16x2 r7181, r7149, r7173;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7149;
mov.b32 r7184, {high, low};
}
{
fma.rn.f16x2 r7186, r7178, r7184, r7181;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7186;
mov.b32 r7190, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7186;
mov.b32 r7192, {high, high};
}
{
mul.f16x2 r7194, r6427, r7192;
}
{
fma.rn.f16x2 r7197, r6391, r7190, r7194;
}
{
mul.f16x2 r7201, r6391, r7192;
}
{
neg.f16x2 r7204, r7201;
}
{
fma.rn.f16x2 r7206, r6427, r7190, r7204;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7210, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7212, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7214, {low, high};
}
{
mul.f16x2 r7215, r7212, r7214;
}
{
mul.f16x2 r7218, r7186, r7210;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7186;
mov.b32 r7221, {high, low};
}
{
fma.rn.f16x2 r7223, r7215, r7221, r7218;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7223;
mov.b32 r7227, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7223;
mov.b32 r7229, {high, high};
}
{
mul.f16x2 r7231, r6513, r7229;
}
{
fma.rn.f16x2 r7234, r6477, r7227, r7231;
}
{
mul.f16x2 r7238, r6477, r7229;
}
{
neg.f16x2 r7241, r7238;
}
{
fma.rn.f16x2 r7243, r6513, r7227, r7241;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7247, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7249, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7251, {low, high};
}
{
mul.f16x2 r7252, r7249, r7251;
}
{
mul.f16x2 r7255, r7223, r7247;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7223;
mov.b32 r7258, {high, low};
}
{
fma.rn.f16x2 r7260, r7252, r7258, r7255;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7260;
mov.b32 r7264, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7260;
mov.b32 r7266, {high, high};
}
{
mul.f16x2 r7268, r6599, r7266;
}
{
fma.rn.f16x2 r7271, r6563, r7264, r7268;
}
{
mul.f16x2 r7275, r6563, r7266;
}
{
neg.f16x2 r7278, r7275;
}
{
fma.rn.f16x2 r7280, r6599, r7264, r7278;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7284, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7286, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7288, {low, high};
}
{
mul.f16x2 r7289, r7286, r7288;
}
{
mul.f16x2 r7292, r7260, r7284;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7260;
mov.b32 r7295, {high, low};
}
{
fma.rn.f16x2 r7297, r7289, r7295, r7292;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7297;
mov.b32 r7301, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7297;
mov.b32 r7303, {high, high};
}
{
mul.f16x2 r7305, r6685, r7303;
}
{
fma.rn.f16x2 r7308, r6649, r7301, r7305;
}
{
mul.f16x2 r7312, r6649, r7303;
}
{
neg.f16x2 r7315, r7312;
}
{
fma.rn.f16x2 r7317, r6685, r7301, r7315;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7321, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7323, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7325, {low, high};
}
{
mul.f16x2 r7326, r7323, r7325;
}
{
mul.f16x2 r7329, r7297, r7321;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7297;
mov.b32 r7332, {high, low};
}
{
fma.rn.f16x2 r7334, r7326, r7332, r7329;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7334;
mov.b32 r7338, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7334;
mov.b32 r7340, {high, high};
}
{
mul.f16x2 r7342, r6015, r7340;
}
{
fma.rn.f16x2 r7345, r5979, r7338, r7342;
}
{
mul.f16x2 r7349, r5979, r7340;
}
{
neg.f16x2 r7352, r7349;
}
{
fma.rn.f16x2 r7354, r6015, r7338, r7352;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7358, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7360, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7362, {low, high};
}
{
mul.f16x2 r7363, r7360, r7362;
}
{
mul.f16x2 r7366, r7334, r7358;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7334;
mov.b32 r7369, {high, low};
}
{
fma.rn.f16x2 r7371, r7363, r7369, r7366;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7371;
mov.b32 r7375, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7371;
mov.b32 r7377, {high, high};
}
{
mul.f16x2 r7379, r6101, r7377;
}
{
fma.rn.f16x2 r7382, r6065, r7375, r7379;
}
{
mul.f16x2 r7386, r6065, r7377;
}
{
neg.f16x2 r7389, r7386;
}
{
fma.rn.f16x2 r7391, r6101, r7375, r7389;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7395, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7397, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7399, {low, high};
}
{
mul.f16x2 r7400, r7397, r7399;
}
{
mul.f16x2 r7403, r7371, r7395;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7371;
mov.b32 r7406, {high, low};
}
{
fma.rn.f16x2 r7408, r7400, r7406, r7403;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7408;
mov.b32 r7412, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7408;
mov.b32 r7414, {high, high};
}
{
mul.f16x2 r7416, r6187, r7414;
}
{
fma.rn.f16x2 r7419, r6151, r7412, r7416;
}
{
mul.f16x2 r7423, r6151, r7414;
}
{
neg.f16x2 r7426, r7423;
}
{
fma.rn.f16x2 r7428, r6187, r7412, r7426;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7432, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7434, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7436, {low, high};
}
{
mul.f16x2 r7437, r7434, r7436;
}
{
mul.f16x2 r7440, r7408, r7432;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7408;
mov.b32 r7443, {high, low};
}
{
fma.rn.f16x2 r7445, r7437, r7443, r7440;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7445;
mov.b32 r7449, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7445;
mov.b32 r7451, {high, high};
}
{
mul.f16x2 r7453, r6273, r7451;
}
{
fma.rn.f16x2 r7456, r6237, r7449, r7453;
}
{
mul.f16x2 r7460, r6237, r7451;
}
{
neg.f16x2 r7463, r7460;
}
{
fma.rn.f16x2 r7465, r6273, r7449, r7463;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7469, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7471, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7473, {low, high};
}
{
mul.f16x2 r7474, r7471, r7473;
}
{
mul.f16x2 r7477, r7445, r7469;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7445;
mov.b32 r7480, {high, low};
}
{
fma.rn.f16x2 r7482, r7474, r7480, r7477;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7482;
mov.b32 r7486, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7482;
mov.b32 r7488, {high, high};
}
{
mul.f16x2 r7490, r6359, r7488;
}
{
fma.rn.f16x2 r7493, r6323, r7486, r7490;
}
{
mul.f16x2 r7497, r6323, r7488;
}
{
neg.f16x2 r7500, r7497;
}
{
fma.rn.f16x2 r7502, r6359, r7486, r7500;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7506, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7508, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7510, {low, high};
}
{
mul.f16x2 r7511, r7508, r7510;
}
{
mul.f16x2 r7514, r7482, r7506;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7482;
mov.b32 r7517, {high, low};
}
{
fma.rn.f16x2 r7519, r7511, r7517, r7514;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7519;
mov.b32 r7523, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7519;
mov.b32 r7525, {high, high};
}
{
mul.f16x2 r7527, r6445, r7525;
}
{
fma.rn.f16x2 r7530, r6409, r7523, r7527;
}
{
mul.f16x2 r7534, r6409, r7525;
}
{
neg.f16x2 r7537, r7534;
}
{
fma.rn.f16x2 r7539, r6445, r7523, r7537;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7543, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7545, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7547, {low, high};
}
{
mul.f16x2 r7548, r7545, r7547;
}
{
mul.f16x2 r7551, r7519, r7543;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7519;
mov.b32 r7554, {high, low};
}
{
fma.rn.f16x2 r7556, r7548, r7554, r7551;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7556;
mov.b32 r7560, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7556;
mov.b32 r7562, {high, high};
}
{
mul.f16x2 r7564, r6531, r7562;
}
{
fma.rn.f16x2 r7567, r6495, r7560, r7564;
}
{
mul.f16x2 r7571, r6495, r7562;
}
{
neg.f16x2 r7574, r7571;
}
{
fma.rn.f16x2 r7576, r6531, r7560, r7574;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7580, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7582, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7584, {low, high};
}
{
mul.f16x2 r7585, r7582, r7584;
}
{
mul.f16x2 r7588, r7556, r7580;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7556;
mov.b32 r7591, {high, low};
}
{
fma.rn.f16x2 r7593, r7585, r7591, r7588;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7593;
mov.b32 r7597, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7593;
mov.b32 r7599, {high, high};
}
{
mul.f16x2 r7601, r6617, r7599;
}
{
fma.rn.f16x2 r7604, r6581, r7597, r7601;
}
{
mul.f16x2 r7608, r6581, r7599;
}
{
neg.f16x2 r7611, r7608;
}
{
fma.rn.f16x2 r7613, r6617, r7597, r7611;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7617, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7619, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7621, {low, high};
}
{
mul.f16x2 r7622, r7619, r7621;
}
{
mul.f16x2 r7625, r7593, r7617;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7593;
mov.b32 r7628, {high, low};
}
{
fma.rn.f16x2 r7630, r7622, r7628, r7625;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7630;
mov.b32 r7634, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7630;
mov.b32 r7636, {high, high};
}
{
mul.f16x2 r7638, r6703, r7636;
}
{
fma.rn.f16x2 r7641, r6667, r7634, r7638;
}
{
mul.f16x2 r7645, r6667, r7636;
}
{
neg.f16x2 r7648, r7645;
}
{
fma.rn.f16x2 r7650, r6703, r7634, r7648;
}
shl.b32 r8462, r8461, 3;
add.s32 r8463, r8452, r8462;
barrier.sync 0;
mad.lo.s32 r8464, r8459, 5832, r8463;
st.shared.u32 [r8464], r5937;
st.shared.u32 [r8464+4], r5943;
st.shared.u32 [r8464+216], r6716;
st.shared.u32 [r8464+220], r6725;
st.shared.u32 [r8464+432], r6753;
st.shared.u32 [r8464+436], r6762;
st.shared.u32 [r8464+648], r6790;
st.shared.u32 [r8464+652], r6799;
st.shared.u32 [r8464+864], r6827;
st.shared.u32 [r8464+868], r6836;
st.shared.u32 [r8464+1080], r6864;
st.shared.u32 [r8464+1084], r6873;
st.shared.u32 [r8464+1296], r6901;
st.shared.u32 [r8464+1300], r6910;
st.shared.u32 [r8464+1512], r6938;
st.shared.u32 [r8464+1516], r6947;
st.shared.u32 [r8464+1728], r6975;
st.shared.u32 [r8464+1732], r6984;
st.shared.u32 [r8464+1944], r7012;
st.shared.u32 [r8464+1948], r7021;
st.shared.u32 [r8464+2160], r7049;
st.shared.u32 [r8464+2164], r7058;
st.shared.u32 [r8464+2376], r7086;
st.shared.u32 [r8464+2380], r7095;
st.shared.u32 [r8464+2592], r7123;
st.shared.u32 [r8464+2596], r7132;
st.shared.u32 [r8464+2808], r7160;
st.shared.u32 [r8464+2812], r7169;
st.shared.u32 [r8464+3024], r7197;
st.shared.u32 [r8464+3028], r7206;
st.shared.u32 [r8464+3240], r7234;
st.shared.u32 [r8464+3244], r7243;
st.shared.u32 [r8464+3456], r7271;
st.shared.u32 [r8464+3460], r7280;
st.shared.u32 [r8464+3672], r7308;
st.shared.u32 [r8464+3676], r7317;
st.shared.u32 [r8464+3888], r7345;
st.shared.u32 [r8464+3892], r7354;
st.shared.u32 [r8464+4104], r7382;
st.shared.u32 [r8464+4108], r7391;
st.shared.u32 [r8464+4320], r7419;
st.shared.u32 [r8464+4324], r7428;
st.shared.u32 [r8464+4536], r7456;
st.shared.u32 [r8464+4540], r7465;
st.shared.u32 [r8464+4752], r7493;
st.shared.u32 [r8464+4756], r7502;
st.shared.u32 [r8464+4968], r7530;
st.shared.u32 [r8464+4972], r7539;
st.shared.u32 [r8464+5184], r7567;
st.shared.u32 [r8464+5188], r7576;
st.shared.u32 [r8464+5400], r7604;
st.shared.u32 [r8464+5404], r7613;
st.shared.u32 [r8464+5616], r7641;
st.shared.u32 [r8464+5620], r7650;
barrier.sync 0;
ld.shared.u32 r7677, [r8454];
ld.shared.u32 r7683, [r8454+4];
ld.shared.u32 r7763, [r8454+648];
ld.shared.u32 r7769, [r8454+652];
ld.shared.u32 r7849, [r8454+1296];
ld.shared.u32 r7855, [r8454+1300];
ld.shared.u32 r7935, [r8454+1944];
ld.shared.u32 r7941, [r8454+1948];
ld.shared.u32 r8021, [r8454+2592];
ld.shared.u32 r8027, [r8454+2596];
ld.shared.u32 r8107, [r8454+3240];
ld.shared.u32 r8113, [r8454+3244];
ld.shared.u32 r8193, [r8454+3888];
ld.shared.u32 r8199, [r8454+3892];
ld.shared.u32 r8279, [r8454+4536];
ld.shared.u32 r8285, [r8454+4540];
ld.shared.u32 r8365, [r8454+5184];
ld.shared.u32 r8371, [r8454+5188];
ld.shared.u32 r7674, [r8454+5832];
ld.shared.u32 r7680, [r8454+5836];
ld.shared.u32 r7760, [r8454+6480];
ld.shared.u32 r7766, [r8454+6484];
ld.shared.u32 r7846, [r8454+7128];
ld.shared.u32 r7852, [r8454+7132];
ld.shared.u32 r7932, [r8454+7776];
ld.shared.u32 r7938, [r8454+7780];
ld.shared.u32 r8018, [r8454+8424];
ld.shared.u32 r8024, [r8454+8428];
ld.shared.u32 r8104, [r8454+9072];
ld.shared.u32 r8110, [r8454+9076];
ld.shared.u32 r8190, [r8454+9720];
ld.shared.u32 r8196, [r8454+9724];
ld.shared.u32 r8276, [r8454+10368];
ld.shared.u32 r8282, [r8454+10372];
ld.shared.u32 r8362, [r8454+11016];
ld.shared.u32 r8368, [r8454+11020];
ld.shared.u32 r7675, [r8454+11664];
ld.shared.u32 r7681, [r8454+11668];
ld.shared.u32 r7761, [r8454+12312];
ld.shared.u32 r7767, [r8454+12316];
ld.shared.u32 r7847, [r8454+12960];
ld.shared.u32 r7853, [r8454+12964];
ld.shared.u32 r7933, [r8454+13608];
ld.shared.u32 r7939, [r8454+13612];
ld.shared.u32 r8019, [r8454+14256];
ld.shared.u32 r8025, [r8454+14260];
ld.shared.u32 r8105, [r8454+14904];
ld.shared.u32 r8111, [r8454+14908];
ld.shared.u32 r8191, [r8454+15552];
ld.shared.u32 r8197, [r8454+15556];
ld.shared.u32 r8277, [r8454+16200];
ld.shared.u32 r8283, [r8454+16204];
ld.shared.u32 r8363, [r8454+16848];
ld.shared.u32 r8369, [r8454+16852];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r7671, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r7672, {low, high};
}
{
add.f16x2 r7673, r7674, r7675;
}
{
add.f16x2 %0, r7677, r7673;
}
{
add.f16x2 r7679, r7680, r7681;
}
{
add.f16x2 %1, r7683, r7679;
}
{
add.f16x2 r7685, r7674, r7675;
}
{
mul.f16x2 r7688, r7685, r7671;
}
{
add.f16x2 r7691, r7677, r7688;
}
{
sub.f16x2 r7694, r7680, r7681;
}
{
mul.f16x2 r7697, r7694, r7672;
}
{
add.f16x2 %18, r7691, r7697;
}
{
add.f16x2 r7703, r7674, r7675;
}
{
mul.f16x2 r7706, r7703, r7671;
}
{
add.f16x2 r7709, r7677, r7706;
}
{
sub.f16x2 r7712, r7680, r7681;
}
{
mul.f16x2 r7715, r7712, r7672;
}
{
sub.f16x2 %36, r7709, r7715;
}
{
add.f16x2 r7721, r7680, r7681;
}
{
mul.f16x2 r7724, r7721, r7671;
}
{
add.f16x2 r7727, r7683, r7724;
}
{
sub.f16x2 r7730, r7674, r7675;
}
{
mul.f16x2 r7733, r7730, r7672;
}
{
sub.f16x2 %19, r7727, r7733;
}
{
add.f16x2 r7739, r7680, r7681;
}
{
mul.f16x2 r7742, r7739, r7671;
}
{
add.f16x2 r7745, r7683, r7742;
}
{
sub.f16x2 r7748, r7674, r7675;
}
{
mul.f16x2 r7751, r7748, r7672;
}
{
add.f16x2 %37, r7745, r7751;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r7757, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r7758, {low, high};
}
{
add.f16x2 r7759, r7760, r7761;
}
{
add.f16x2 %2, r7763, r7759;
}
{
add.f16x2 r7765, r7766, r7767;
}
{
add.f16x2 %3, r7769, r7765;
}
{
add.f16x2 r7771, r7760, r7761;
}
{
mul.f16x2 r7774, r7771, r7757;
}
{
add.f16x2 r7777, r7763, r7774;
}
{
sub.f16x2 r7780, r7766, r7767;
}
{
mul.f16x2 r7783, r7780, r7758;
}
{
add.f16x2 %20, r7777, r7783;
}
{
add.f16x2 r7789, r7760, r7761;
}
{
mul.f16x2 r7792, r7789, r7757;
}
{
add.f16x2 r7795, r7763, r7792;
}
{
sub.f16x2 r7798, r7766, r7767;
}
{
mul.f16x2 r7801, r7798, r7758;
}
{
sub.f16x2 %38, r7795, r7801;
}
{
add.f16x2 r7807, r7766, r7767;
}
{
mul.f16x2 r7810, r7807, r7757;
}
{
add.f16x2 r7813, r7769, r7810;
}
{
sub.f16x2 r7816, r7760, r7761;
}
{
mul.f16x2 r7819, r7816, r7758;
}
{
sub.f16x2 %21, r7813, r7819;
}
{
add.f16x2 r7825, r7766, r7767;
}
{
mul.f16x2 r7828, r7825, r7757;
}
{
add.f16x2 r7831, r7769, r7828;
}
{
sub.f16x2 r7834, r7760, r7761;
}
{
mul.f16x2 r7837, r7834, r7758;
}
{
add.f16x2 %39, r7831, r7837;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r7843, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r7844, {low, high};
}
{
add.f16x2 r7845, r7846, r7847;
}
{
add.f16x2 %4, r7849, r7845;
}
{
add.f16x2 r7851, r7852, r7853;
}
{
add.f16x2 %5, r7855, r7851;
}
{
add.f16x2 r7857, r7846, r7847;
}
{
mul.f16x2 r7860, r7857, r7843;
}
{
add.f16x2 r7863, r7849, r7860;
}
{
sub.f16x2 r7866, r7852, r7853;
}
{
mul.f16x2 r7869, r7866, r7844;
}
{
add.f16x2 %22, r7863, r7869;
}
{
add.f16x2 r7875, r7846, r7847;
}
{
mul.f16x2 r7878, r7875, r7843;
}
{
add.f16x2 r7881, r7849, r7878;
}
{
sub.f16x2 r7884, r7852, r7853;
}
{
mul.f16x2 r7887, r7884, r7844;
}
{
sub.f16x2 %40, r7881, r7887;
}
{
add.f16x2 r7893, r7852, r7853;
}
{
mul.f16x2 r7896, r7893, r7843;
}
{
add.f16x2 r7899, r7855, r7896;
}
{
sub.f16x2 r7902, r7846, r7847;
}
{
mul.f16x2 r7905, r7902, r7844;
}
{
sub.f16x2 %23, r7899, r7905;
}
{
add.f16x2 r7911, r7852, r7853;
}
{
mul.f16x2 r7914, r7911, r7843;
}
{
add.f16x2 r7917, r7855, r7914;
}
{
sub.f16x2 r7920, r7846, r7847;
}
{
mul.f16x2 r7923, r7920, r7844;
}
{
add.f16x2 %41, r7917, r7923;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r7929, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r7930, {low, high};
}
{
add.f16x2 r7931, r7932, r7933;
}
{
add.f16x2 %6, r7935, r7931;
}
{
add.f16x2 r7937, r7938, r7939;
}
{
add.f16x2 %7, r7941, r7937;
}
{
add.f16x2 r7943, r7932, r7933;
}
{
mul.f16x2 r7946, r7943, r7929;
}
{
add.f16x2 r7949, r7935, r7946;
}
{
sub.f16x2 r7952, r7938, r7939;
}
{
mul.f16x2 r7955, r7952, r7930;
}
{
add.f16x2 %24, r7949, r7955;
}
{
add.f16x2 r7961, r7932, r7933;
}
{
mul.f16x2 r7964, r7961, r7929;
}
{
add.f16x2 r7967, r7935, r7964;
}
{
sub.f16x2 r7970, r7938, r7939;
}
{
mul.f16x2 r7973, r7970, r7930;
}
{
sub.f16x2 %42, r7967, r7973;
}
{
add.f16x2 r7979, r7938, r7939;
}
{
mul.f16x2 r7982, r7979, r7929;
}
{
add.f16x2 r7985, r7941, r7982;
}
{
sub.f16x2 r7988, r7932, r7933;
}
{
mul.f16x2 r7991, r7988, r7930;
}
{
sub.f16x2 %25, r7985, r7991;
}
{
add.f16x2 r7997, r7938, r7939;
}
{
mul.f16x2 r8000, r7997, r7929;
}
{
add.f16x2 r8003, r7941, r8000;
}
{
sub.f16x2 r8006, r7932, r7933;
}
{
mul.f16x2 r8009, r8006, r7930;
}
{
add.f16x2 %43, r8003, r8009;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8015, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8016, {low, high};
}
{
add.f16x2 r8017, r8018, r8019;
}
{
add.f16x2 %8, r8021, r8017;
}
{
add.f16x2 r8023, r8024, r8025;
}
{
add.f16x2 %9, r8027, r8023;
}
{
add.f16x2 r8029, r8018, r8019;
}
{
mul.f16x2 r8032, r8029, r8015;
}
{
add.f16x2 r8035, r8021, r8032;
}
{
sub.f16x2 r8038, r8024, r8025;
}
{
mul.f16x2 r8041, r8038, r8016;
}
{
add.f16x2 %26, r8035, r8041;
}
{
add.f16x2 r8047, r8018, r8019;
}
{
mul.f16x2 r8050, r8047, r8015;
}
{
add.f16x2 r8053, r8021, r8050;
}
{
sub.f16x2 r8056, r8024, r8025;
}
{
mul.f16x2 r8059, r8056, r8016;
}
{
sub.f16x2 %44, r8053, r8059;
}
{
add.f16x2 r8065, r8024, r8025;
}
{
mul.f16x2 r8068, r8065, r8015;
}
{
add.f16x2 r8071, r8027, r8068;
}
{
sub.f16x2 r8074, r8018, r8019;
}
{
mul.f16x2 r8077, r8074, r8016;
}
{
sub.f16x2 %27, r8071, r8077;
}
{
add.f16x2 r8083, r8024, r8025;
}
{
mul.f16x2 r8086, r8083, r8015;
}
{
add.f16x2 r8089, r8027, r8086;
}
{
sub.f16x2 r8092, r8018, r8019;
}
{
mul.f16x2 r8095, r8092, r8016;
}
{
add.f16x2 %45, r8089, r8095;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8102, {low, high};
}
{
add.f16x2 r8103, r8104, r8105;
}
{
add.f16x2 %10, r8107, r8103;
}
{
add.f16x2 r8109, r8110, r8111;
}
{
add.f16x2 %11, r8113, r8109;
}
{
add.f16x2 r8115, r8104, r8105;
}
{
mul.f16x2 r8118, r8115, r8101;
}
{
add.f16x2 r8121, r8107, r8118;
}
{
sub.f16x2 r8124, r8110, r8111;
}
{
mul.f16x2 r8127, r8124, r8102;
}
{
add.f16x2 %28, r8121, r8127;
}
{
add.f16x2 r8133, r8104, r8105;
}
{
mul.f16x2 r8136, r8133, r8101;
}
{
add.f16x2 r8139, r8107, r8136;
}
{
sub.f16x2 r8142, r8110, r8111;
}
{
mul.f16x2 r8145, r8142, r8102;
}
{
sub.f16x2 %46, r8139, r8145;
}
{
add.f16x2 r8151, r8110, r8111;
}
{
mul.f16x2 r8154, r8151, r8101;
}
{
add.f16x2 r8157, r8113, r8154;
}
{
sub.f16x2 r8160, r8104, r8105;
}
{
mul.f16x2 r8163, r8160, r8102;
}
{
sub.f16x2 %29, r8157, r8163;
}
{
add.f16x2 r8169, r8110, r8111;
}
{
mul.f16x2 r8172, r8169, r8101;
}
{
add.f16x2 r8175, r8113, r8172;
}
{
sub.f16x2 r8178, r8104, r8105;
}
{
mul.f16x2 r8181, r8178, r8102;
}
{
add.f16x2 %47, r8175, r8181;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8187, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8188, {low, high};
}
{
add.f16x2 r8189, r8190, r8191;
}
{
add.f16x2 %12, r8193, r8189;
}
{
add.f16x2 r8195, r8196, r8197;
}
{
add.f16x2 %13, r8199, r8195;
}
{
add.f16x2 r8201, r8190, r8191;
}
{
mul.f16x2 r8204, r8201, r8187;
}
{
add.f16x2 r8207, r8193, r8204;
}
{
sub.f16x2 r8210, r8196, r8197;
}
{
mul.f16x2 r8213, r8210, r8188;
}
{
add.f16x2 %30, r8207, r8213;
}
{
add.f16x2 r8219, r8190, r8191;
}
{
mul.f16x2 r8222, r8219, r8187;
}
{
add.f16x2 r8225, r8193, r8222;
}
{
sub.f16x2 r8228, r8196, r8197;
}
{
mul.f16x2 r8231, r8228, r8188;
}
{
sub.f16x2 %48, r8225, r8231;
}
{
add.f16x2 r8237, r8196, r8197;
}
{
mul.f16x2 r8240, r8237, r8187;
}
{
add.f16x2 r8243, r8199, r8240;
}
{
sub.f16x2 r8246, r8190, r8191;
}
{
mul.f16x2 r8249, r8246, r8188;
}
{
sub.f16x2 %31, r8243, r8249;
}
{
add.f16x2 r8255, r8196, r8197;
}
{
mul.f16x2 r8258, r8255, r8187;
}
{
add.f16x2 r8261, r8199, r8258;
}
{
sub.f16x2 r8264, r8190, r8191;
}
{
mul.f16x2 r8267, r8264, r8188;
}
{
add.f16x2 %49, r8261, r8267;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8273, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8274, {low, high};
}
{
add.f16x2 r8275, r8276, r8277;
}
{
add.f16x2 %14, r8279, r8275;
}
{
add.f16x2 r8281, r8282, r8283;
}
{
add.f16x2 %15, r8285, r8281;
}
{
add.f16x2 r8287, r8276, r8277;
}
{
mul.f16x2 r8290, r8287, r8273;
}
{
add.f16x2 r8293, r8279, r8290;
}
{
sub.f16x2 r8296, r8282, r8283;
}
{
mul.f16x2 r8299, r8296, r8274;
}
{
add.f16x2 %32, r8293, r8299;
}
{
add.f16x2 r8305, r8276, r8277;
}
{
mul.f16x2 r8308, r8305, r8273;
}
{
add.f16x2 r8311, r8279, r8308;
}
{
sub.f16x2 r8314, r8282, r8283;
}
{
mul.f16x2 r8317, r8314, r8274;
}
{
sub.f16x2 %50, r8311, r8317;
}
{
add.f16x2 r8323, r8282, r8283;
}
{
mul.f16x2 r8326, r8323, r8273;
}
{
add.f16x2 r8329, r8285, r8326;
}
{
sub.f16x2 r8332, r8276, r8277;
}
{
mul.f16x2 r8335, r8332, r8274;
}
{
sub.f16x2 %33, r8329, r8335;
}
{
add.f16x2 r8341, r8282, r8283;
}
{
mul.f16x2 r8344, r8341, r8273;
}
{
add.f16x2 r8347, r8285, r8344;
}
{
sub.f16x2 r8350, r8276, r8277;
}
{
mul.f16x2 r8353, r8350, r8274;
}
{
add.f16x2 %51, r8347, r8353;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8359, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8360, {low, high};
}
{
add.f16x2 r8361, r8362, r8363;
}
{
add.f16x2 %16, r8365, r8361;
}
{
add.f16x2 r8367, r8368, r8369;
}
{
add.f16x2 %17, r8371, r8367;
}
{
add.f16x2 r8373, r8362, r8363;
}
{
mul.f16x2 r8376, r8373, r8359;
}
{
add.f16x2 r8379, r8365, r8376;
}
{
sub.f16x2 r8382, r8368, r8369;
}
{
mul.f16x2 r8385, r8382, r8360;
}
{
add.f16x2 %34, r8379, r8385;
}
{
add.f16x2 r8391, r8362, r8363;
}
{
mul.f16x2 r8394, r8391, r8359;
}
{
add.f16x2 r8397, r8365, r8394;
}
{
sub.f16x2 r8400, r8368, r8369;
}
{
mul.f16x2 r8403, r8400, r8360;
}
{
sub.f16x2 %52, r8397, r8403;
}
{
add.f16x2 r8409, r8368, r8369;
}
{
mul.f16x2 r8412, r8409, r8359;
}
{
add.f16x2 r8415, r8371, r8412;
}
{
sub.f16x2 r8418, r8362, r8363;
}
{
mul.f16x2 r8421, r8418, r8360;
}
{
sub.f16x2 %35, r8415, r8421;
}
{
add.f16x2 r8427, r8368, r8369;
}
{
mul.f16x2 r8430, r8427, r8359;
}
{
add.f16x2 r8433, r8371, r8430;
}
{
sub.f16x2 r8436, r8362, r8363;
}
{
mul.f16x2 r8439, r8436, r8360;
}
{
add.f16x2 %53, r8433, r8439;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[9].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1092, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<250>;
.reg .b32 r<2967>;
.reg .b64 rd<8>;
mov.u32 r2944, %tid.y;
mov.u32 r2945, %18;
mad.lo.s32 r2946, r2944, 17496, r2945;
mov.u32 r2947, %tid.x;
mov.f32 f238, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1, {low, high};
}
mov.f32 f240, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r173, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
mov.f32 f166, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f166;
cvt.rn.f16.f32 high, f166;
mov.b32 r259, {low, high};
}
mov.f32 f168, 0f3F248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f168;
cvt.rn.f16.f32 high, f168;
mov.b32 r260, {low, high};
}
mov.f32 f170, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f170;
cvt.rn.f16.f32 high, f170;
mov.b32 r261, {low, high};
}
mov.f32 f172, 0f3F7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f172;
cvt.rn.f16.f32 high, f172;
mov.b32 r262, {low, high};
}
mov.f32 f178, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r265, {low, high};
}
mov.f32 f180, 0f3EAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r339, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r425, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
mul.wide.u32 rd2, r2947, -2032597691;
shr.u64 rd3, rd2, 39;
cvt.u32.u64 r2948, rd3;
mul.lo.s32 r2949, r2948, 243;
sub.s32 r2950, r2947, r2949;
cvt.rn.f32.u32 f241, r2950;
mul.f32 f242, f241, 0f3B3C4870;
cos.approx.f32 f57, f242;
sin.approx.f32 f243, f242;
neg.f32 f58, f243;
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
mov.f32 f225, 0fBF800000;
mov.f32 f226, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
mad.lo.s32 r2951, r2948, 17496, r2946;
barrier.sync 0;
mad.lo.s32 r2952, r2950, 72, r2951;
st.shared.v2.f32 [r2952], {r344, r350};
st.shared.v2.f32 [r2952+8], {r607, r616};
st.shared.v2.f32 [r2952+16], {r644, r653};
st.shared.v2.f32 [r2952+24], {r681, r690};
st.shared.v2.f32 [r2952+32], {r718, r727};
st.shared.v2.f32 [r2952+40], {r755, r764};
st.shared.v2.f32 [r2952+48], {r792, r801};
st.shared.v2.f32 [r2952+56], {r829, r838};
st.shared.v2.f32 [r2952+64], {r866, r875};
barrier.sync 0;
shl.b32 r2953, r2950, 6;
sub.s32 r2954, r2952, r2953;
ld.shared.u32 r902, [r2954];
ld.shared.u32 r908, [r2954+4];
ld.shared.u32 r988, [r2954+1944];
ld.shared.u32 r994, [r2954+1948];
ld.shared.u32 r1074, [r2954+3888];
ld.shared.u32 r1080, [r2954+3892];
ld.shared.u32 r899, [r2954+5832];
ld.shared.u32 r905, [r2954+5836];
ld.shared.u32 r985, [r2954+7776];
ld.shared.u32 r991, [r2954+7780];
ld.shared.u32 r1071, [r2954+9720];
ld.shared.u32 r1077, [r2954+9724];
ld.shared.u32 r900, [r2954+11664];
ld.shared.u32 r906, [r2954+11668];
ld.shared.u32 r986, [r2954+13608];
ld.shared.u32 r992, [r2954+13612];
ld.shared.u32 r1072, [r2954+15552];
ld.shared.u32 r1078, [r2954+15556];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r896, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r982, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1068, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f166;
cvt.rn.f16.f32 high, f166;
mov.b32 r1154, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f168;
cvt.rn.f16.f32 high, f168;
mov.b32 r1155, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f170;
cvt.rn.f16.f32 high, f170;
mov.b32 r1156, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f172;
cvt.rn.f16.f32 high, f172;
mov.b32 r1157, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r1160, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1234, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1235, {low, high};
}
{
add.f16x2 r1236, r987, r1073;
}
{
add.f16x2 r1239, r901, r1236;
}
{
add.f16x2 r1242, r993, r1079;
}
{
add.f16x2 r1245, r907, r1242;
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
add.f16x2 r1263, r1254, r1260;
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
sub.f16x2 r1281, r1272, r1278;
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
sub.f16x2 r1299, r1290, r1296;
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
add.f16x2 r1317, r1308, r1314;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1320, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1321, {low, high};
}
{
add.f16x2 r1322, r1176, r1192;
}
{
add.f16x2 r1325, r925, r1322;
}
{
add.f16x2 r1328, r1182, r1198;
}
{
add.f16x2 r1331, r961, r1328;
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
add.f16x2 r1349, r1340, r1346;
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
sub.f16x2 r1367, r1358, r1364;
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
sub.f16x2 r1385, r1376, r1382;
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
add.f16x2 r1403, r1394, r1400;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1406, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1407, {low, high};
}
{
add.f16x2 r1408, r1208, r1224;
}
{
add.f16x2 r1411, r943, r1408;
}
{
add.f16x2 r1414, r1214, r1230;
}
{
add.f16x2 r1417, r979, r1414;
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
add.f16x2 r1435, r1426, r1432;
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
sub.f16x2 r1453, r1444, r1450;
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
sub.f16x2 r1471, r1462, r1468;
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
add.f16x2 r1489, r1480, r1486;
}
mul.wide.u32 rd4, r2950, 954437177;
shr.u64 rd5, rd4, 33;
cvt.u32.u64 r2955, rd5;
cvt.rn.f32.u32 f244, r2955;
mul.f32 f245, f244, 0f3CD3D17E;
cos.approx.f32 f133, f245;
sin.approx.f32 f246, f245;
neg.f32 f134, f246;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f133;
cvt.rn.f16.f32 high, f134;
mov.b32 r1492, {low, high};
}
mul.lo.s32 r2956, r2955, 9;
sub.s32 r2957, r2950, r2956;
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1495, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1497, {high, high};
}
{
mul.f16x2 r1499, r1331, r1497;
}
{
fma.rn.f16x2 r1502, r1325, r1495, r1499;
}
{
mul.f16x2 r1506, r1325, r1497;
}
{
neg.f16x2 r1509, r1506;
}
{
fma.rn.f16x2 r1511, r1331, r1495, r1509;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1515, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1517, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1519, {low, high};
}
{
mul.f16x2 r1520, r1517, r1519;
}
{
mul.f16x2 r1523, r1492, r1515;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1526, {high, low};
}
{
fma.rn.f16x2 r1528, r1520, r1526, r1523;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1528;
mov.b32 r1532, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1528;
mov.b32 r1534, {high, high};
}
{
mul.f16x2 r1536, r1417, r1534;
}
{
fma.rn.f16x2 r1539, r1411, r1532, r1536;
}
{
mul.f16x2 r1543, r1411, r1534;
}
{
neg.f16x2 r1546, r1543;
}
{
fma.rn.f16x2 r1548, r1417, r1532, r1546;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1552, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1554, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1556, {low, high};
}
{
mul.f16x2 r1557, r1554, r1556;
}
{
mul.f16x2 r1560, r1528, r1552;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1528;
mov.b32 r1563, {high, low};
}
{
fma.rn.f16x2 r1565, r1557, r1563, r1560;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1565;
mov.b32 r1569, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1565;
mov.b32 r1571, {high, high};
}
{
mul.f16x2 r1573, r1299, r1571;
}
{
fma.rn.f16x2 r1576, r1263, r1569, r1573;
}
{
mul.f16x2 r1580, r1263, r1571;
}
{
neg.f16x2 r1583, r1580;
}
{
fma.rn.f16x2 r1585, r1299, r1569, r1583;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1589, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1591, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1593, {low, high};
}
{
mul.f16x2 r1594, r1591, r1593;
}
{
mul.f16x2 r1597, r1565, r1589;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1565;
mov.b32 r1600, {high, low};
}
{
fma.rn.f16x2 r1602, r1594, r1600, r1597;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1602;
mov.b32 r1606, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1602;
mov.b32 r1608, {high, high};
}
{
mul.f16x2 r1610, r1385, r1608;
}
{
fma.rn.f16x2 r1613, r1349, r1606, r1610;
}
{
mul.f16x2 r1617, r1349, r1608;
}
{
neg.f16x2 r1620, r1617;
}
{
fma.rn.f16x2 r1622, r1385, r1606, r1620;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1626, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1628, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1630, {low, high};
}
{
mul.f16x2 r1631, r1628, r1630;
}
{
mul.f16x2 r1634, r1602, r1626;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1602;
mov.b32 r1637, {high, low};
}
{
fma.rn.f16x2 r1639, r1631, r1637, r1634;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1639;
mov.b32 r1643, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1639;
mov.b32 r1645, {high, high};
}
{
mul.f16x2 r1647, r1471, r1645;
}
{
fma.rn.f16x2 r1650, r1435, r1643, r1647;
}
{
mul.f16x2 r1654, r1435, r1645;
}
{
neg.f16x2 r1657, r1654;
}
{
fma.rn.f16x2 r1659, r1471, r1643, r1657;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1663, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1665, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1667, {low, high};
}
{
mul.f16x2 r1668, r1665, r1667;
}
{
mul.f16x2 r1671, r1639, r1663;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1639;
mov.b32 r1674, {high, low};
}
{
fma.rn.f16x2 r1676, r1668, r1674, r1671;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1676;
mov.b32 r1680, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1676;
mov.b32 r1682, {high, high};
}
{
mul.f16x2 r1684, r1317, r1682;
}
{
fma.rn.f16x2 r1687, r1281, r1680, r1684;
}
{
mul.f16x2 r1691, r1281, r1682;
}
{
neg.f16x2 r1694, r1691;
}
{
fma.rn.f16x2 r1696, r1317, r1680, r1694;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1700, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1702, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1704, {low, high};
}
{
mul.f16x2 r1705, r1702, r1704;
}
{
mul.f16x2 r1708, r1676, r1700;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1676;
mov.b32 r1711, {high, low};
}
{
fma.rn.f16x2 r1713, r1705, r1711, r1708;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1717, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1719, {high, high};
}
{
mul.f16x2 r1721, r1403, r1719;
}
{
fma.rn.f16x2 r1724, r1367, r1717, r1721;
}
{
mul.f16x2 r1728, r1367, r1719;
}
{
neg.f16x2 r1731, r1728;
}
{
fma.rn.f16x2 r1733, r1403, r1717, r1731;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1737, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1739, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1741, {low, high};
}
{
mul.f16x2 r1742, r1739, r1741;
}
{
mul.f16x2 r1745, r1713, r1737;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1748, {high, low};
}
{
fma.rn.f16x2 r1750, r1742, r1748, r1745;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1750;
mov.b32 r1754, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1750;
mov.b32 r1756, {high, high};
}
{
mul.f16x2 r1758, r1489, r1756;
}
{
fma.rn.f16x2 r1761, r1453, r1754, r1758;
}
{
mul.f16x2 r1765, r1453, r1756;
}
{
neg.f16x2 r1768, r1765;
}
{
fma.rn.f16x2 r1770, r1489, r1754, r1768;
}
shl.b32 r2958, r2957, 3;
add.s32 r2959, r2951, r2958;
barrier.sync 0;
mad.lo.s32 r2960, r2955, 648, r2959;
st.shared.u32 [r2960], r1239;
st.shared.u32 [r2960+4], r1245;
st.shared.u32 [r2960+72], r1502;
st.shared.u32 [r2960+76], r1511;
st.shared.u32 [r2960+144], r1539;
st.shared.u32 [r2960+148], r1548;
st.shared.u32 [r2960+216], r1576;
st.shared.u32 [r2960+220], r1585;
st.shared.u32 [r2960+288], r1613;
st.shared.u32 [r2960+292], r1622;
st.shared.u32 [r2960+360], r1650;
st.shared.u32 [r2960+364], r1659;
st.shared.u32 [r2960+432], r1687;
st.shared.u32 [r2960+436], r1696;
st.shared.u32 [r2960+504], r1724;
st.shared.u32 [r2960+508], r1733;
st.shared.u32 [r2960+576], r1761;
st.shared.u32 [r2960+580], r1770;
barrier.sync 0;
ld.shared.u32 r1797, [r2954];
ld.shared.u32 r1803, [r2954+4];
ld.shared.u32 r1883, [r2954+1944];
ld.shared.u32 r1889, [r2954+1948];
ld.shared.u32 r1969, [r2954+3888];
ld.shared.u32 r1975, [r2954+3892];
ld.shared.u32 r1794, [r2954+5832];
ld.shared.u32 r1800, [r2954+5836];
ld.shared.u32 r1880, [r2954+7776];
ld.shared.u32 r1886, [r2954+7780];
ld.shared.u32 r1966, [r2954+9720];
ld.shared.u32 r1972, [r2954+9724];
ld.shared.u32 r1795, [r2954+11664];
ld.shared.u32 r1801, [r2954+11668];
ld.shared.u32 r1881, [r2954+13608];
ld.shared.u32 r1887, [r2954+13612];
ld.shared.u32 r1967, [r2954+15552];
ld.shared.u32 r1973, [r2954+15556];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1791, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1792, {low, high};
}
{
add.f16x2 r1793, r1794, r1795;
}
{
add.f16x2 r1796, r1797, r1793;
}
{
add.f16x2 r1799, r1800, r1801;
}
{
add.f16x2 r1802, r1803, r1799;
}
{
add.f16x2 r1805, r1794, r1795;
}
{
mul.f16x2 r1808, r1805, r1791;
}
{
add.f16x2 r1811, r1797, r1808;
}
{
sub.f16x2 r1814, r1800, r1801;
}
{
mul.f16x2 r1817, r1814, r1792;
}
{
add.f16x2 r1820, r1811, r1817;
}
{
add.f16x2 r1823, r1794, r1795;
}
{
mul.f16x2 r1826, r1823, r1791;
}
{
add.f16x2 r1829, r1797, r1826;
}
{
sub.f16x2 r1832, r1800, r1801;
}
{
mul.f16x2 r1835, r1832, r1792;
}
{
sub.f16x2 r1838, r1829, r1835;
}
{
add.f16x2 r1841, r1800, r1801;
}
{
mul.f16x2 r1844, r1841, r1791;
}
{
add.f16x2 r1847, r1803, r1844;
}
{
sub.f16x2 r1850, r1794, r1795;
}
{
mul.f16x2 r1853, r1850, r1792;
}
{
sub.f16x2 r1856, r1847, r1853;
}
{
add.f16x2 r1859, r1800, r1801;
}
{
mul.f16x2 r1862, r1859, r1791;
}
{
add.f16x2 r1865, r1803, r1862;
}
{
sub.f16x2 r1868, r1794, r1795;
}
{
mul.f16x2 r1871, r1868, r1792;
}
{
add.f16x2 r1874, r1865, r1871;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1877, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1878, {low, high};
}
{
add.f16x2 r1879, r1880, r1881;
}
{
add.f16x2 r1882, r1883, r1879;
}
{
add.f16x2 r1885, r1886, r1887;
}
{
add.f16x2 r1888, r1889, r1885;
}
{
add.f16x2 r1891, r1880, r1881;
}
{
mul.f16x2 r1894, r1891, r1877;
}
{
add.f16x2 r1897, r1883, r1894;
}
{
sub.f16x2 r1900, r1886, r1887;
}
{
mul.f16x2 r1903, r1900, r1878;
}
{
add.f16x2 r1906, r1897, r1903;
}
{
add.f16x2 r1909, r1880, r1881;
}
{
mul.f16x2 r1912, r1909, r1877;
}
{
add.f16x2 r1915, r1883, r1912;
}
{
sub.f16x2 r1918, r1886, r1887;
}
{
mul.f16x2 r1921, r1918, r1878;
}
{
sub.f16x2 r1924, r1915, r1921;
}
{
add.f16x2 r1927, r1886, r1887;
}
{
mul.f16x2 r1930, r1927, r1877;
}
{
add.f16x2 r1933, r1889, r1930;
}
{
sub.f16x2 r1936, r1880, r1881;
}
{
mul.f16x2 r1939, r1936, r1878;
}
{
sub.f16x2 r1942, r1933, r1939;
}
{
add.f16x2 r1945, r1886, r1887;
}
{
mul.f16x2 r1948, r1945, r1877;
}
{
add.f16x2 r1951, r1889, r1948;
}
{
sub.f16x2 r1954, r1880, r1881;
}
{
mul.f16x2 r1957, r1954, r1878;
}
{
add.f16x2 r1960, r1951, r1957;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1964, {low, high};
}
{
add.f16x2 r1965, r1966, r1967;
}
{
add.f16x2 r1968, r1969, r1965;
}
{
add.f16x2 r1971, r1972, r1973;
}
{
add.f16x2 r1974, r1975, r1971;
}
{
add.f16x2 r1977, r1966, r1967;
}
{
mul.f16x2 r1980, r1977, r1963;
}
{
add.f16x2 r1983, r1969, r1980;
}
{
sub.f16x2 r1986, r1972, r1973;
}
{
mul.f16x2 r1989, r1986, r1964;
}
{
add.f16x2 r1992, r1983, r1989;
}
{
add.f16x2 r1995, r1966, r1967;
}
{
mul.f16x2 r1998, r1995, r1963;
}
{
add.f16x2 r2001, r1969, r1998;
}
{
sub.f16x2 r2004, r1972, r1973;
}
{
mul.f16x2 r2007, r2004, r1964;
}
{
sub.f16x2 r2010, r2001, r2007;
}
{
add.f16x2 r2013, r1972, r1973;
}
{
mul.f16x2 r2016, r2013, r1963;
}
{
add.f16x2 r2019, r1975, r2016;
}
{
sub.f16x2 r2022, r1966, r1967;
}
{
mul.f16x2 r2025, r2022, r1964;
}
{
sub.f16x2 r2028, r2019, r2025;
}
{
add.f16x2 r2031, r1972, r1973;
}
{
mul.f16x2 r2034, r2031, r1963;
}
{
add.f16x2 r2037, r1975, r2034;
}
{
sub.f16x2 r2040, r1966, r1967;
}
{
mul.f16x2 r2043, r2040, r1964;
}
{
add.f16x2 r2046, r2037, r2043;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f166;
cvt.rn.f16.f32 high, f166;
mov.b32 r2049, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f168;
cvt.rn.f16.f32 high, f168;
mov.b32 r2050, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f170;
cvt.rn.f16.f32 high, f170;
mov.b32 r2051, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f172;
cvt.rn.f16.f32 high, f172;
mov.b32 r2052, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r2055, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
mov.b32 r2056, {low, high};
}
{
mul.f16x2 r2065, r1906, r2049;
}
{
mul.f16x2 r2068, r1942, r2050;
}
{
sub.f16x2 r2071, r2065, r2068;
}
{
mul.f16x2 r2074, r1906, r2050;
}
{
fma.rn.f16x2 r2077, r1942, r2049, r2074;
}
{
mul.f16x2 r2081, r1992, r2051;
}
{
mul.f16x2 r2084, r2028, r2052;
}
{
sub.f16x2 r2087, r2081, r2084;
}
{
mul.f16x2 r2090, r1992, r2052;
}
{
fma.rn.f16x2 r2093, r2028, r2051, r2090;
}
{
mul.f16x2 r2097, r1924, r2051;
}
{
mul.f16x2 r2100, r1960, r2052;
}
{
sub.f16x2 r2103, r2097, r2100;
}
{
mul.f16x2 r2106, r1924, r2052;
}
{
fma.rn.f16x2 r2109, r1960, r2051, r2106;
}
{
mul.f16x2 r2113, r2010, r2055;
}
{
mul.f16x2 r2116, r2046, r2056;
}
{
sub.f16x2 r2119, r2113, r2116;
}
{
mul.f16x2 r2122, r2010, r2056;
}
{
fma.rn.f16x2 r2125, r2046, r2055, r2122;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2129, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2130, {low, high};
}
{
add.f16x2 r2131, r1882, r1968;
}
{
add.f16x2 r2134, r1796, r2131;
}
{
add.f16x2 r2137, r1888, r1974;
}
{
add.f16x2 r2140, r1802, r2137;
}
{
add.f16x2 r2143, r1882, r1968;
}
{
mul.f16x2 r2146, r2143, r2129;
}
{
add.f16x2 r2149, r1796, r2146;
}
{
sub.f16x2 r2152, r1888, r1974;
}
{
mul.f16x2 r2155, r2152, r2130;
}
{
add.f16x2 r2158, r2149, r2155;
}
{
add.f16x2 r2161, r1882, r1968;
}
{
mul.f16x2 r2164, r2161, r2129;
}
{
add.f16x2 r2167, r1796, r2164;
}
{
sub.f16x2 r2170, r1888, r1974;
}
{
mul.f16x2 r2173, r2170, r2130;
}
{
sub.f16x2 r2176, r2167, r2173;
}
{
add.f16x2 r2179, r1888, r1974;
}
{
mul.f16x2 r2182, r2179, r2129;
}
{
add.f16x2 r2185, r1802, r2182;
}
{
sub.f16x2 r2188, r1882, r1968;
}
{
mul.f16x2 r2191, r2188, r2130;
}
{
sub.f16x2 r2194, r2185, r2191;
}
{
add.f16x2 r2197, r1888, r1974;
}
{
mul.f16x2 r2200, r2197, r2129;
}
{
add.f16x2 r2203, r1802, r2200;
}
{
sub.f16x2 r2206, r1882, r1968;
}
{
mul.f16x2 r2209, r2206, r2130;
}
{
add.f16x2 r2212, r2203, r2209;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2215, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2216, {low, high};
}
{
add.f16x2 r2217, r2071, r2087;
}
{
add.f16x2 r2220, r1820, r2217;
}
{
add.f16x2 r2223, r2077, r2093;
}
{
add.f16x2 r2226, r1856, r2223;
}
{
add.f16x2 r2229, r2071, r2087;
}
{
mul.f16x2 r2232, r2229, r2215;
}
{
add.f16x2 r2235, r1820, r2232;
}
{
sub.f16x2 r2238, r2077, r2093;
}
{
mul.f16x2 r2241, r2238, r2216;
}
{
add.f16x2 r2244, r2235, r2241;
}
{
add.f16x2 r2247, r2071, r2087;
}
{
mul.f16x2 r2250, r2247, r2215;
}
{
add.f16x2 r2253, r1820, r2250;
}
{
sub.f16x2 r2256, r2077, r2093;
}
{
mul.f16x2 r2259, r2256, r2216;
}
{
sub.f16x2 r2262, r2253, r2259;
}
{
add.f16x2 r2265, r2077, r2093;
}
{
mul.f16x2 r2268, r2265, r2215;
}
{
add.f16x2 r2271, r1856, r2268;
}
{
sub.f16x2 r2274, r2071, r2087;
}
{
mul.f16x2 r2277, r2274, r2216;
}
{
sub.f16x2 r2280, r2271, r2277;
}
{
add.f16x2 r2283, r2077, r2093;
}
{
mul.f16x2 r2286, r2283, r2215;
}
{
add.f16x2 r2289, r1856, r2286;
}
{
sub.f16x2 r2292, r2071, r2087;
}
{
mul.f16x2 r2295, r2292, r2216;
}
{
add.f16x2 r2298, r2289, r2295;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2302, {low, high};
}
{
add.f16x2 r2303, r2103, r2119;
}
{
add.f16x2 r2306, r1838, r2303;
}
{
add.f16x2 r2309, r2109, r2125;
}
{
add.f16x2 r2312, r1874, r2309;
}
{
add.f16x2 r2315, r2103, r2119;
}
{
mul.f16x2 r2318, r2315, r2301;
}
{
add.f16x2 r2321, r1838, r2318;
}
{
sub.f16x2 r2324, r2109, r2125;
}
{
mul.f16x2 r2327, r2324, r2302;
}
{
add.f16x2 r2330, r2321, r2327;
}
{
add.f16x2 r2333, r2103, r2119;
}
{
mul.f16x2 r2336, r2333, r2301;
}
{
add.f16x2 r2339, r1838, r2336;
}
{
sub.f16x2 r2342, r2109, r2125;
}
{
mul.f16x2 r2345, r2342, r2302;
}
{
sub.f16x2 r2348, r2339, r2345;
}
{
add.f16x2 r2351, r2109, r2125;
}
{
mul.f16x2 r2354, r2351, r2301;
}
{
add.f16x2 r2357, r1874, r2354;
}
{
sub.f16x2 r2360, r2103, r2119;
}
{
mul.f16x2 r2363, r2360, r2302;
}
{
sub.f16x2 r2366, r2357, r2363;
}
{
add.f16x2 r2369, r2109, r2125;
}
{
mul.f16x2 r2372, r2369, r2301;
}
{
add.f16x2 r2375, r1874, r2372;
}
{
sub.f16x2 r2378, r2103, r2119;
}
{
mul.f16x2 r2381, r2378, r2302;
}
{
add.f16x2 r2384, r2375, r2381;
}
mul.wide.u32 rd6, r2950, -901412889;
shr.u64 rd7, rd6, 38;
cvt.u32.u64 r2961, rd7;
cvt.rn.f32.u32 f247, r2961;
mul.f32 f248, f247, 0f3E6E4BAE;
cos.approx.f32 f209, f248;
sin.approx.f32 f249, f248;
neg.f32 f210, f249;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f209;
cvt.rn.f16.f32 high, f210;
mov.b32 r2387, {low, high};
}
mul.lo.s32 r2962, r2961, 81;
sub.s32 r2963, r2950, r2962;
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2390, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2392, {high, high};
}
{
mul.f16x2 r2394, r2226, r2392;
}
{
fma.rn.f16x2 r2397, r2220, r2390, r2394;
}
{
mul.f16x2 r2401, r2220, r2392;
}
{
neg.f16x2 r2404, r2401;
}
{
fma.rn.f16x2 r2406, r2226, r2390, r2404;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2410, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2412, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2414, {low, high};
}
{
mul.f16x2 r2415, r2412, r2414;
}
{
mul.f16x2 r2418, r2387, r2410;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2421, {high, low};
}
{
fma.rn.f16x2 r2423, r2415, r2421, r2418;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2427, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2429, {high, high};
}
{
mul.f16x2 r2431, r2312, r2429;
}
{
fma.rn.f16x2 r2434, r2306, r2427, r2431;
}
{
mul.f16x2 r2438, r2306, r2429;
}
{
neg.f16x2 r2441, r2438;
}
{
fma.rn.f16x2 r2443, r2312, r2427, r2441;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2447, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2449, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2451, {low, high};
}
{
mul.f16x2 r2452, r2449, r2451;
}
{
mul.f16x2 r2455, r2423, r2447;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2458, {high, low};
}
{
fma.rn.f16x2 r2460, r2452, r2458, r2455;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2460;
mov.b32 r2464, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2460;
mov.b32 r2466, {high, high};
}
{
mul.f16x2 r2468, r2194, r2466;
}
{
fma.rn.f16x2 r2471, r2158, r2464, r2468;
}
{
mul.f16x2 r2475, r2158, r2466;
}
{
neg.f16x2 r2478, r2475;
}
{
fma.rn.f16x2 r2480, r2194, r2464, r2478;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2484, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2486, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2488, {low, high};
}
{
mul.f16x2 r2489, r2486, r2488;
}
{
mul.f16x2 r2492, r2460, r2484;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2460;
mov.b32 r2495, {high, low};
}
{
fma.rn.f16x2 r2497, r2489, r2495, r2492;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2497;
mov.b32 r2501, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2497;
mov.b32 r2503, {high, high};
}
{
mul.f16x2 r2505, r2280, r2503;
}
{
fma.rn.f16x2 r2508, r2244, r2501, r2505;
}
{
mul.f16x2 r2512, r2244, r2503;
}
{
neg.f16x2 r2515, r2512;
}
{
fma.rn.f16x2 r2517, r2280, r2501, r2515;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2521, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2523, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2525, {low, high};
}
{
mul.f16x2 r2526, r2523, r2525;
}
{
mul.f16x2 r2529, r2497, r2521;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2497;
mov.b32 r2532, {high, low};
}
{
fma.rn.f16x2 r2534, r2526, r2532, r2529;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2534;
mov.b32 r2538, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2534;
mov.b32 r2540, {high, high};
}
{
mul.f16x2 r2542, r2366, r2540;
}
{
fma.rn.f16x2 r2545, r2330, r2538, r2542;
}
{
mul.f16x2 r2549, r2330, r2540;
}
{
neg.f16x2 r2552, r2549;
}
{
fma.rn.f16x2 r2554, r2366, r2538, r2552;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2558, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2560, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2562, {low, high};
}
{
mul.f16x2 r2563, r2560, r2562;
}
{
mul.f16x2 r2566, r2534, r2558;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2534;
mov.b32 r2569, {high, low};
}
{
fma.rn.f16x2 r2571, r2563, r2569, r2566;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2571;
mov.b32 r2575, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2571;
mov.b32 r2577, {high, high};
}
{
mul.f16x2 r2579, r2212, r2577;
}
{
fma.rn.f16x2 r2582, r2176, r2575, r2579;
}
{
mul.f16x2 r2586, r2176, r2577;
}
{
neg.f16x2 r2589, r2586;
}
{
fma.rn.f16x2 r2591, r2212, r2575, r2589;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2595, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2597, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2599, {low, high};
}
{
mul.f16x2 r2600, r2597, r2599;
}
{
mul.f16x2 r2603, r2571, r2595;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2571;
mov.b32 r2606, {high, low};
}
{
fma.rn.f16x2 r2608, r2600, r2606, r2603;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2608;
mov.b32 r2612, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2608;
mov.b32 r2614, {high, high};
}
{
mul.f16x2 r2616, r2298, r2614;
}
{
fma.rn.f16x2 r2619, r2262, r2612, r2616;
}
{
mul.f16x2 r2623, r2262, r2614;
}
{
neg.f16x2 r2626, r2623;
}
{
fma.rn.f16x2 r2628, r2298, r2612, r2626;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2632, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2634, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2636, {low, high};
}
{
mul.f16x2 r2637, r2634, r2636;
}
{
mul.f16x2 r2640, r2608, r2632;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2608;
mov.b32 r2643, {high, low};
}
{
fma.rn.f16x2 r2645, r2637, r2643, r2640;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2645;
mov.b32 r2649, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2645;
mov.b32 r2651, {high, high};
}
{
mul.f16x2 r2653, r2384, r2651;
}
{
fma.rn.f16x2 r2656, r2348, r2649, r2653;
}
{
mul.f16x2 r2660, r2348, r2651;
}
{
neg.f16x2 r2663, r2660;
}
{
fma.rn.f16x2 r2665, r2384, r2649, r2663;
}
shl.b32 r2964, r2963, 3;
add.s32 r2965, r2951, r2964;
barrier.sync 0;
mad.lo.s32 r2966, r2961, 5832, r2965;
st.shared.u32 [r2966], r2134;
st.shared.u32 [r2966+4], r2140;
st.shared.u32 [r2966+648], r2397;
st.shared.u32 [r2966+652], r2406;
st.shared.u32 [r2966+1296], r2434;
st.shared.u32 [r2966+1300], r2443;
st.shared.u32 [r2966+1944], r2471;
st.shared.u32 [r2966+1948], r2480;
st.shared.u32 [r2966+2592], r2508;
st.shared.u32 [r2966+2596], r2517;
st.shared.u32 [r2966+3240], r2545;
st.shared.u32 [r2966+3244], r2554;
st.shared.u32 [r2966+3888], r2582;
st.shared.u32 [r2966+3892], r2591;
st.shared.u32 [r2966+4536], r2619;
st.shared.u32 [r2966+4540], r2628;
st.shared.u32 [r2966+5184], r2656;
st.shared.u32 [r2966+5188], r2665;
barrier.sync 0;
ld.shared.u32 r2692, [r2954];
ld.shared.u32 r2698, [r2954+4];
ld.shared.u32 r2778, [r2954+1944];
ld.shared.u32 r2784, [r2954+1948];
ld.shared.u32 r2864, [r2954+3888];
ld.shared.u32 r2870, [r2954+3892];
ld.shared.u32 r2689, [r2954+5832];
ld.shared.u32 r2695, [r2954+5836];
ld.shared.u32 r2775, [r2954+7776];
ld.shared.u32 r2781, [r2954+7780];
ld.shared.u32 r2861, [r2954+9720];
ld.shared.u32 r2867, [r2954+9724];
ld.shared.u32 r2690, [r2954+11664];
ld.shared.u32 r2696, [r2954+11668];
ld.shared.u32 r2776, [r2954+13608];
ld.shared.u32 r2782, [r2954+13612];
ld.shared.u32 r2862, [r2954+15552];
ld.shared.u32 r2868, [r2954+15556];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2686, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2687, {low, high};
}
{
add.f16x2 r2688, r2689, r2690;
}
{
add.f16x2 %0, r2692, r2688;
}
{
add.f16x2 r2694, r2695, r2696;
}
{
add.f16x2 %1, r2698, r2694;
}
{
add.f16x2 r2700, r2689, r2690;
}
{
mul.f16x2 r2703, r2700, r2686;
}
{
add.f16x2 r2706, r2692, r2703;
}
{
sub.f16x2 r2709, r2695, r2696;
}
{
mul.f16x2 r2712, r2709, r2687;
}
{
add.f16x2 %6, r2706, r2712;
}
{
add.f16x2 r2718, r2689, r2690;
}
{
mul.f16x2 r2721, r2718, r2686;
}
{
add.f16x2 r2724, r2692, r2721;
}
{
sub.f16x2 r2727, r2695, r2696;
}
{
mul.f16x2 r2730, r2727, r2687;
}
{
sub.f16x2 %12, r2724, r2730;
}
{
add.f16x2 r2736, r2695, r2696;
}
{
mul.f16x2 r2739, r2736, r2686;
}
{
add.f16x2 r2742, r2698, r2739;
}
{
sub.f16x2 r2745, r2689, r2690;
}
{
mul.f16x2 r2748, r2745, r2687;
}
{
sub.f16x2 %7, r2742, r2748;
}
{
add.f16x2 r2754, r2695, r2696;
}
{
mul.f16x2 r2757, r2754, r2686;
}
{
add.f16x2 r2760, r2698, r2757;
}
{
sub.f16x2 r2763, r2689, r2690;
}
{
mul.f16x2 r2766, r2763, r2687;
}
{
add.f16x2 %13, r2760, r2766;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2772, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2773, {low, high};
}
{
add.f16x2 r2774, r2775, r2776;
}
{
add.f16x2 %2, r2778, r2774;
}
{
add.f16x2 r2780, r2781, r2782;
}
{
add.f16x2 %3, r2784, r2780;
}
{
add.f16x2 r2786, r2775, r2776;
}
{
mul.f16x2 r2789, r2786, r2772;
}
{
add.f16x2 r2792, r2778, r2789;
}
{
sub.f16x2 r2795, r2781, r2782;
}
{
mul.f16x2 r2798, r2795, r2773;
}
{
add.f16x2 %8, r2792, r2798;
}
{
add.f16x2 r2804, r2775, r2776;
}
{
mul.f16x2 r2807, r2804, r2772;
}
{
add.f16x2 r2810, r2778, r2807;
}
{
sub.f16x2 r2813, r2781, r2782;
}
{
mul.f16x2 r2816, r2813, r2773;
}
{
sub.f16x2 %14, r2810, r2816;
}
{
add.f16x2 r2822, r2781, r2782;
}
{
mul.f16x2 r2825, r2822, r2772;
}
{
add.f16x2 r2828, r2784, r2825;
}
{
sub.f16x2 r2831, r2775, r2776;
}
{
mul.f16x2 r2834, r2831, r2773;
}
{
sub.f16x2 %9, r2828, r2834;
}
{
add.f16x2 r2840, r2781, r2782;
}
{
mul.f16x2 r2843, r2840, r2772;
}
{
add.f16x2 r2846, r2784, r2843;
}
{
sub.f16x2 r2849, r2775, r2776;
}
{
mul.f16x2 r2852, r2849, r2773;
}
{
add.f16x2 %15, r2846, r2852;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2858, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2859, {low, high};
}
{
add.f16x2 r2860, r2861, r2862;
}
{
add.f16x2 %4, r2864, r2860;
}
{
add.f16x2 r2866, r2867, r2868;
}
{
add.f16x2 %5, r2870, r2866;
}
{
add.f16x2 r2872, r2861, r2862;
}
{
mul.f16x2 r2875, r2872, r2858;
}
{
add.f16x2 r2878, r2864, r2875;
}
{
sub.f16x2 r2881, r2867, r2868;
}
{
mul.f16x2 r2884, r2881, r2859;
}
{
add.f16x2 %10, r2878, r2884;
}
{
add.f16x2 r2890, r2861, r2862;
}
{
mul.f16x2 r2893, r2890, r2858;
}
{
add.f16x2 r2896, r2864, r2893;
}
{
sub.f16x2 r2899, r2867, r2868;
}
{
mul.f16x2 r2902, r2899, r2859;
}
{
sub.f16x2 %16, r2896, r2902;
}
{
add.f16x2 r2908, r2867, r2868;
}
{
mul.f16x2 r2911, r2908, r2858;
}
{
add.f16x2 r2914, r2870, r2911;
}
{
sub.f16x2 r2917, r2861, r2862;
}
{
mul.f16x2 r2920, r2917, r2859;
}
{
sub.f16x2 %11, r2914, r2920;
}
{
add.f16x2 r2926, r2867, r2868;
}
{
mul.f16x2 r2929, r2926, r2858;
}
{
add.f16x2 r2932, r2870, r2929;
}
{
sub.f16x2 r2935, r2861, r2862;
}
{
mul.f16x2 r2938, r2935, r2859;
}
{
add.f16x2 %17, r2932, r2938;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1094, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<250>;
.reg .b32 r<2967>;
.reg .b64 rd<8>;
mov.u32 r2944, %tid.y;
mov.u32 r2945, %18;
mad.lo.s32 r2946, r2944, 8748, r2945;
mov.u32 r2947, %tid.x;
mov.f32 f238, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1, {low, high};
}
mov.f32 f240, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r173, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
mov.f32 f166, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f166;
cvt.rn.f16.f32 high, f166;
mov.b32 r259, {low, high};
}
mov.f32 f168, 0f3F248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f168;
cvt.rn.f16.f32 high, f168;
mov.b32 r260, {low, high};
}
mov.f32 f170, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f170;
cvt.rn.f16.f32 high, f170;
mov.b32 r261, {low, high};
}
mov.f32 f172, 0f3F7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f172;
cvt.rn.f16.f32 high, f172;
mov.b32 r262, {low, high};
}
mov.f32 f178, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r265, {low, high};
}
mov.f32 f180, 0f3EAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r339, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r425, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
mul.wide.u32 rd2, r2947, -2032597691;
shr.u64 rd3, rd2, 39;
cvt.u32.u64 r2948, rd3;
mul.lo.s32 r2949, r2948, 243;
sub.s32 r2950, r2947, r2949;
mad.lo.s32 r2951, r2948, 8748, r2946;
cvt.rn.f32.u32 f241, r2950;
mul.f32 f242, f241, 0f3B3C4870;
cos.approx.f32 f57, f242;
sin.approx.f32 f243, f242;
neg.f32 f58, f243;
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
mov.f32 f225, 0fBF800000;
mov.f32 f226, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
mad.lo.s32 r2952, r2950, 36, r2951;
st.shared.u32 [r2952], r344;
st.shared.u32 [r2952+4], r607;
st.shared.u32 [r2952+8], r644;
st.shared.u32 [r2952+12], r681;
st.shared.u32 [r2952+16], r718;
st.shared.u32 [r2952+20], r755;
st.shared.u32 [r2952+24], r792;
st.shared.u32 [r2952+28], r829;
st.shared.u32 [r2952+32], r866;
barrier.sync 0;
shl.b32 r2953, r2950, 5;
sub.s32 r2954, r2952, r2953;
ld.shared.u32 r902, [r2954];
ld.shared.u32 r988, [r2954+972];
ld.shared.u32 r1074, [r2954+1944];
ld.shared.u32 r899, [r2954+2916];
ld.shared.u32 r985, [r2954+3888];
ld.shared.u32 r1071, [r2954+4860];
ld.shared.u32 r900, [r2954+5832];
ld.shared.u32 r986, [r2954+6804];
ld.shared.u32 r1072, [r2954+7776];
barrier.sync 0;
st.shared.u32 [r2952], r350;
st.shared.u32 [r2952+4], r616;
st.shared.u32 [r2952+8], r653;
st.shared.u32 [r2952+12], r690;
st.shared.u32 [r2952+16], r727;
st.shared.u32 [r2952+20], r764;
st.shared.u32 [r2952+24], r801;
st.shared.u32 [r2952+28], r838;
st.shared.u32 [r2952+32], r875;
barrier.sync 0;
ld.shared.u32 r908, [r2954];
ld.shared.u32 r994, [r2954+972];
ld.shared.u32 r1080, [r2954+1944];
ld.shared.u32 r905, [r2954+2916];
ld.shared.u32 r991, [r2954+3888];
ld.shared.u32 r1077, [r2954+4860];
ld.shared.u32 r906, [r2954+5832];
ld.shared.u32 r992, [r2954+6804];
ld.shared.u32 r1078, [r2954+7776];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r896, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r982, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1068, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f166;
cvt.rn.f16.f32 high, f166;
mov.b32 r1154, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f168;
cvt.rn.f16.f32 high, f168;
mov.b32 r1155, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f170;
cvt.rn.f16.f32 high, f170;
mov.b32 r1156, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f172;
cvt.rn.f16.f32 high, f172;
mov.b32 r1157, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r1160, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1234, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1235, {low, high};
}
{
add.f16x2 r1236, r987, r1073;
}
{
add.f16x2 r1239, r901, r1236;
}
{
add.f16x2 r1242, r993, r1079;
}
{
add.f16x2 r1245, r907, r1242;
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
add.f16x2 r1263, r1254, r1260;
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
sub.f16x2 r1281, r1272, r1278;
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
sub.f16x2 r1299, r1290, r1296;
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
add.f16x2 r1317, r1308, r1314;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1320, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1321, {low, high};
}
{
add.f16x2 r1322, r1176, r1192;
}
{
add.f16x2 r1325, r925, r1322;
}
{
add.f16x2 r1328, r1182, r1198;
}
{
add.f16x2 r1331, r961, r1328;
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
add.f16x2 r1349, r1340, r1346;
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
sub.f16x2 r1367, r1358, r1364;
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
sub.f16x2 r1385, r1376, r1382;
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
add.f16x2 r1403, r1394, r1400;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1406, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1407, {low, high};
}
{
add.f16x2 r1408, r1208, r1224;
}
{
add.f16x2 r1411, r943, r1408;
}
{
add.f16x2 r1414, r1214, r1230;
}
{
add.f16x2 r1417, r979, r1414;
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
add.f16x2 r1435, r1426, r1432;
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
sub.f16x2 r1453, r1444, r1450;
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
sub.f16x2 r1471, r1462, r1468;
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
add.f16x2 r1489, r1480, r1486;
}
mul.wide.u32 rd4, r2950, 954437177;
shr.u64 rd5, rd4, 33;
cvt.u32.u64 r2955, rd5;
mul.lo.s32 r2956, r2955, 9;
sub.s32 r2957, r2950, r2956;
shl.b32 r2958, r2957, 2;
add.s32 r2959, r2951, r2958;
cvt.rn.f32.u32 f244, r2955;
mul.f32 f245, f244, 0f3CD3D17E;
cos.approx.f32 f133, f245;
sin.approx.f32 f246, f245;
neg.f32 f134, f246;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f133;
cvt.rn.f16.f32 high, f134;
mov.b32 r1492, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1495, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1497, {high, high};
}
{
mul.f16x2 r1499, r1331, r1497;
}
{
fma.rn.f16x2 r1502, r1325, r1495, r1499;
}
{
mul.f16x2 r1506, r1325, r1497;
}
{
neg.f16x2 r1509, r1506;
}
{
fma.rn.f16x2 r1511, r1331, r1495, r1509;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1515, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1517, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1519, {low, high};
}
{
mul.f16x2 r1520, r1517, r1519;
}
{
mul.f16x2 r1523, r1492, r1515;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1526, {high, low};
}
{
fma.rn.f16x2 r1528, r1520, r1526, r1523;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1528;
mov.b32 r1532, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1528;
mov.b32 r1534, {high, high};
}
{
mul.f16x2 r1536, r1417, r1534;
}
{
fma.rn.f16x2 r1539, r1411, r1532, r1536;
}
{
mul.f16x2 r1543, r1411, r1534;
}
{
neg.f16x2 r1546, r1543;
}
{
fma.rn.f16x2 r1548, r1417, r1532, r1546;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1552, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1554, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1556, {low, high};
}
{
mul.f16x2 r1557, r1554, r1556;
}
{
mul.f16x2 r1560, r1528, r1552;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1528;
mov.b32 r1563, {high, low};
}
{
fma.rn.f16x2 r1565, r1557, r1563, r1560;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1565;
mov.b32 r1569, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1565;
mov.b32 r1571, {high, high};
}
{
mul.f16x2 r1573, r1299, r1571;
}
{
fma.rn.f16x2 r1576, r1263, r1569, r1573;
}
{
mul.f16x2 r1580, r1263, r1571;
}
{
neg.f16x2 r1583, r1580;
}
{
fma.rn.f16x2 r1585, r1299, r1569, r1583;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1589, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1591, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1593, {low, high};
}
{
mul.f16x2 r1594, r1591, r1593;
}
{
mul.f16x2 r1597, r1565, r1589;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1565;
mov.b32 r1600, {high, low};
}
{
fma.rn.f16x2 r1602, r1594, r1600, r1597;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1602;
mov.b32 r1606, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1602;
mov.b32 r1608, {high, high};
}
{
mul.f16x2 r1610, r1385, r1608;
}
{
fma.rn.f16x2 r1613, r1349, r1606, r1610;
}
{
mul.f16x2 r1617, r1349, r1608;
}
{
neg.f16x2 r1620, r1617;
}
{
fma.rn.f16x2 r1622, r1385, r1606, r1620;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1626, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1628, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1630, {low, high};
}
{
mul.f16x2 r1631, r1628, r1630;
}
{
mul.f16x2 r1634, r1602, r1626;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1602;
mov.b32 r1637, {high, low};
}
{
fma.rn.f16x2 r1639, r1631, r1637, r1634;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1639;
mov.b32 r1643, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1639;
mov.b32 r1645, {high, high};
}
{
mul.f16x2 r1647, r1471, r1645;
}
{
fma.rn.f16x2 r1650, r1435, r1643, r1647;
}
{
mul.f16x2 r1654, r1435, r1645;
}
{
neg.f16x2 r1657, r1654;
}
{
fma.rn.f16x2 r1659, r1471, r1643, r1657;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1663, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1665, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1667, {low, high};
}
{
mul.f16x2 r1668, r1665, r1667;
}
{
mul.f16x2 r1671, r1639, r1663;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1639;
mov.b32 r1674, {high, low};
}
{
fma.rn.f16x2 r1676, r1668, r1674, r1671;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1676;
mov.b32 r1680, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1676;
mov.b32 r1682, {high, high};
}
{
mul.f16x2 r1684, r1317, r1682;
}
{
fma.rn.f16x2 r1687, r1281, r1680, r1684;
}
{
mul.f16x2 r1691, r1281, r1682;
}
{
neg.f16x2 r1694, r1691;
}
{
fma.rn.f16x2 r1696, r1317, r1680, r1694;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1700, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1702, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1704, {low, high};
}
{
mul.f16x2 r1705, r1702, r1704;
}
{
mul.f16x2 r1708, r1676, r1700;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1676;
mov.b32 r1711, {high, low};
}
{
fma.rn.f16x2 r1713, r1705, r1711, r1708;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1717, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1719, {high, high};
}
{
mul.f16x2 r1721, r1403, r1719;
}
{
fma.rn.f16x2 r1724, r1367, r1717, r1721;
}
{
mul.f16x2 r1728, r1367, r1719;
}
{
neg.f16x2 r1731, r1728;
}
{
fma.rn.f16x2 r1733, r1403, r1717, r1731;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1737, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1492;
mov.b32 r1739, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1741, {low, high};
}
{
mul.f16x2 r1742, r1739, r1741;
}
{
mul.f16x2 r1745, r1713, r1737;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1713;
mov.b32 r1748, {high, low};
}
{
fma.rn.f16x2 r1750, r1742, r1748, r1745;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1750;
mov.b32 r1754, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1750;
mov.b32 r1756, {high, high};
}
{
mul.f16x2 r1758, r1489, r1756;
}
{
fma.rn.f16x2 r1761, r1453, r1754, r1758;
}
{
mul.f16x2 r1765, r1453, r1756;
}
{
neg.f16x2 r1768, r1765;
}
{
fma.rn.f16x2 r1770, r1489, r1754, r1768;
}
barrier.sync 0;
mad.lo.s32 r2960, r2955, 324, r2959;
st.shared.u32 [r2960], r1239;
st.shared.u32 [r2960+36], r1502;
st.shared.u32 [r2960+72], r1539;
st.shared.u32 [r2960+108], r1576;
st.shared.u32 [r2960+144], r1613;
st.shared.u32 [r2960+180], r1650;
st.shared.u32 [r2960+216], r1687;
st.shared.u32 [r2960+252], r1724;
st.shared.u32 [r2960+288], r1761;
barrier.sync 0;
ld.shared.u32 r1797, [r2954];
ld.shared.u32 r1883, [r2954+972];
ld.shared.u32 r1969, [r2954+1944];
ld.shared.u32 r1794, [r2954+2916];
ld.shared.u32 r1880, [r2954+3888];
ld.shared.u32 r1966, [r2954+4860];
ld.shared.u32 r1795, [r2954+5832];
ld.shared.u32 r1881, [r2954+6804];
ld.shared.u32 r1967, [r2954+7776];
barrier.sync 0;
st.shared.u32 [r2960], r1245;
st.shared.u32 [r2960+36], r1511;
st.shared.u32 [r2960+72], r1548;
st.shared.u32 [r2960+108], r1585;
st.shared.u32 [r2960+144], r1622;
st.shared.u32 [r2960+180], r1659;
st.shared.u32 [r2960+216], r1696;
st.shared.u32 [r2960+252], r1733;
st.shared.u32 [r2960+288], r1770;
barrier.sync 0;
ld.shared.u32 r1803, [r2954];
ld.shared.u32 r1889, [r2954+972];
ld.shared.u32 r1975, [r2954+1944];
ld.shared.u32 r1800, [r2954+2916];
ld.shared.u32 r1886, [r2954+3888];
ld.shared.u32 r1972, [r2954+4860];
ld.shared.u32 r1801, [r2954+5832];
ld.shared.u32 r1887, [r2954+6804];
ld.shared.u32 r1973, [r2954+7776];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1791, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1792, {low, high};
}
{
add.f16x2 r1793, r1794, r1795;
}
{
add.f16x2 r1796, r1797, r1793;
}
{
add.f16x2 r1799, r1800, r1801;
}
{
add.f16x2 r1802, r1803, r1799;
}
{
add.f16x2 r1805, r1794, r1795;
}
{
mul.f16x2 r1808, r1805, r1791;
}
{
add.f16x2 r1811, r1797, r1808;
}
{
sub.f16x2 r1814, r1800, r1801;
}
{
mul.f16x2 r1817, r1814, r1792;
}
{
add.f16x2 r1820, r1811, r1817;
}
{
add.f16x2 r1823, r1794, r1795;
}
{
mul.f16x2 r1826, r1823, r1791;
}
{
add.f16x2 r1829, r1797, r1826;
}
{
sub.f16x2 r1832, r1800, r1801;
}
{
mul.f16x2 r1835, r1832, r1792;
}
{
sub.f16x2 r1838, r1829, r1835;
}
{
add.f16x2 r1841, r1800, r1801;
}
{
mul.f16x2 r1844, r1841, r1791;
}
{
add.f16x2 r1847, r1803, r1844;
}
{
sub.f16x2 r1850, r1794, r1795;
}
{
mul.f16x2 r1853, r1850, r1792;
}
{
sub.f16x2 r1856, r1847, r1853;
}
{
add.f16x2 r1859, r1800, r1801;
}
{
mul.f16x2 r1862, r1859, r1791;
}
{
add.f16x2 r1865, r1803, r1862;
}
{
sub.f16x2 r1868, r1794, r1795;
}
{
mul.f16x2 r1871, r1868, r1792;
}
{
add.f16x2 r1874, r1865, r1871;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1877, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1878, {low, high};
}
{
add.f16x2 r1879, r1880, r1881;
}
{
add.f16x2 r1882, r1883, r1879;
}
{
add.f16x2 r1885, r1886, r1887;
}
{
add.f16x2 r1888, r1889, r1885;
}
{
add.f16x2 r1891, r1880, r1881;
}
{
mul.f16x2 r1894, r1891, r1877;
}
{
add.f16x2 r1897, r1883, r1894;
}
{
sub.f16x2 r1900, r1886, r1887;
}
{
mul.f16x2 r1903, r1900, r1878;
}
{
add.f16x2 r1906, r1897, r1903;
}
{
add.f16x2 r1909, r1880, r1881;
}
{
mul.f16x2 r1912, r1909, r1877;
}
{
add.f16x2 r1915, r1883, r1912;
}
{
sub.f16x2 r1918, r1886, r1887;
}
{
mul.f16x2 r1921, r1918, r1878;
}
{
sub.f16x2 r1924, r1915, r1921;
}
{
add.f16x2 r1927, r1886, r1887;
}
{
mul.f16x2 r1930, r1927, r1877;
}
{
add.f16x2 r1933, r1889, r1930;
}
{
sub.f16x2 r1936, r1880, r1881;
}
{
mul.f16x2 r1939, r1936, r1878;
}
{
sub.f16x2 r1942, r1933, r1939;
}
{
add.f16x2 r1945, r1886, r1887;
}
{
mul.f16x2 r1948, r1945, r1877;
}
{
add.f16x2 r1951, r1889, r1948;
}
{
sub.f16x2 r1954, r1880, r1881;
}
{
mul.f16x2 r1957, r1954, r1878;
}
{
add.f16x2 r1960, r1951, r1957;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1964, {low, high};
}
{
add.f16x2 r1965, r1966, r1967;
}
{
add.f16x2 r1968, r1969, r1965;
}
{
add.f16x2 r1971, r1972, r1973;
}
{
add.f16x2 r1974, r1975, r1971;
}
{
add.f16x2 r1977, r1966, r1967;
}
{
mul.f16x2 r1980, r1977, r1963;
}
{
add.f16x2 r1983, r1969, r1980;
}
{
sub.f16x2 r1986, r1972, r1973;
}
{
mul.f16x2 r1989, r1986, r1964;
}
{
add.f16x2 r1992, r1983, r1989;
}
{
add.f16x2 r1995, r1966, r1967;
}
{
mul.f16x2 r1998, r1995, r1963;
}
{
add.f16x2 r2001, r1969, r1998;
}
{
sub.f16x2 r2004, r1972, r1973;
}
{
mul.f16x2 r2007, r2004, r1964;
}
{
sub.f16x2 r2010, r2001, r2007;
}
{
add.f16x2 r2013, r1972, r1973;
}
{
mul.f16x2 r2016, r2013, r1963;
}
{
add.f16x2 r2019, r1975, r2016;
}
{
sub.f16x2 r2022, r1966, r1967;
}
{
mul.f16x2 r2025, r2022, r1964;
}
{
sub.f16x2 r2028, r2019, r2025;
}
{
add.f16x2 r2031, r1972, r1973;
}
{
mul.f16x2 r2034, r2031, r1963;
}
{
add.f16x2 r2037, r1975, r2034;
}
{
sub.f16x2 r2040, r1966, r1967;
}
{
mul.f16x2 r2043, r2040, r1964;
}
{
add.f16x2 r2046, r2037, r2043;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f166;
cvt.rn.f16.f32 high, f166;
mov.b32 r2049, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f168;
cvt.rn.f16.f32 high, f168;
mov.b32 r2050, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f170;
cvt.rn.f16.f32 high, f170;
mov.b32 r2051, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f172;
cvt.rn.f16.f32 high, f172;
mov.b32 r2052, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r2055, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
mov.b32 r2056, {low, high};
}
{
mul.f16x2 r2065, r1906, r2049;
}
{
mul.f16x2 r2068, r1942, r2050;
}
{
sub.f16x2 r2071, r2065, r2068;
}
{
mul.f16x2 r2074, r1906, r2050;
}
{
fma.rn.f16x2 r2077, r1942, r2049, r2074;
}
{
mul.f16x2 r2081, r1992, r2051;
}
{
mul.f16x2 r2084, r2028, r2052;
}
{
sub.f16x2 r2087, r2081, r2084;
}
{
mul.f16x2 r2090, r1992, r2052;
}
{
fma.rn.f16x2 r2093, r2028, r2051, r2090;
}
{
mul.f16x2 r2097, r1924, r2051;
}
{
mul.f16x2 r2100, r1960, r2052;
}
{
sub.f16x2 r2103, r2097, r2100;
}
{
mul.f16x2 r2106, r1924, r2052;
}
{
fma.rn.f16x2 r2109, r1960, r2051, r2106;
}
{
mul.f16x2 r2113, r2010, r2055;
}
{
mul.f16x2 r2116, r2046, r2056;
}
{
sub.f16x2 r2119, r2113, r2116;
}
{
mul.f16x2 r2122, r2010, r2056;
}
{
fma.rn.f16x2 r2125, r2046, r2055, r2122;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2129, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2130, {low, high};
}
{
add.f16x2 r2131, r1882, r1968;
}
{
add.f16x2 r2134, r1796, r2131;
}
{
add.f16x2 r2137, r1888, r1974;
}
{
add.f16x2 r2140, r1802, r2137;
}
{
add.f16x2 r2143, r1882, r1968;
}
{
mul.f16x2 r2146, r2143, r2129;
}
{
add.f16x2 r2149, r1796, r2146;
}
{
sub.f16x2 r2152, r1888, r1974;
}
{
mul.f16x2 r2155, r2152, r2130;
}
{
add.f16x2 r2158, r2149, r2155;
}
{
add.f16x2 r2161, r1882, r1968;
}
{
mul.f16x2 r2164, r2161, r2129;
}
{
add.f16x2 r2167, r1796, r2164;
}
{
sub.f16x2 r2170, r1888, r1974;
}
{
mul.f16x2 r2173, r2170, r2130;
}
{
sub.f16x2 r2176, r2167, r2173;
}
{
add.f16x2 r2179, r1888, r1974;
}
{
mul.f16x2 r2182, r2179, r2129;
}
{
add.f16x2 r2185, r1802, r2182;
}
{
sub.f16x2 r2188, r1882, r1968;
}
{
mul.f16x2 r2191, r2188, r2130;
}
{
sub.f16x2 r2194, r2185, r2191;
}
{
add.f16x2 r2197, r1888, r1974;
}
{
mul.f16x2 r2200, r2197, r2129;
}
{
add.f16x2 r2203, r1802, r2200;
}
{
sub.f16x2 r2206, r1882, r1968;
}
{
mul.f16x2 r2209, r2206, r2130;
}
{
add.f16x2 r2212, r2203, r2209;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2215, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2216, {low, high};
}
{
add.f16x2 r2217, r2071, r2087;
}
{
add.f16x2 r2220, r1820, r2217;
}
{
add.f16x2 r2223, r2077, r2093;
}
{
add.f16x2 r2226, r1856, r2223;
}
{
add.f16x2 r2229, r2071, r2087;
}
{
mul.f16x2 r2232, r2229, r2215;
}
{
add.f16x2 r2235, r1820, r2232;
}
{
sub.f16x2 r2238, r2077, r2093;
}
{
mul.f16x2 r2241, r2238, r2216;
}
{
add.f16x2 r2244, r2235, r2241;
}
{
add.f16x2 r2247, r2071, r2087;
}
{
mul.f16x2 r2250, r2247, r2215;
}
{
add.f16x2 r2253, r1820, r2250;
}
{
sub.f16x2 r2256, r2077, r2093;
}
{
mul.f16x2 r2259, r2256, r2216;
}
{
sub.f16x2 r2262, r2253, r2259;
}
{
add.f16x2 r2265, r2077, r2093;
}
{
mul.f16x2 r2268, r2265, r2215;
}
{
add.f16x2 r2271, r1856, r2268;
}
{
sub.f16x2 r2274, r2071, r2087;
}
{
mul.f16x2 r2277, r2274, r2216;
}
{
sub.f16x2 r2280, r2271, r2277;
}
{
add.f16x2 r2283, r2077, r2093;
}
{
mul.f16x2 r2286, r2283, r2215;
}
{
add.f16x2 r2289, r1856, r2286;
}
{
sub.f16x2 r2292, r2071, r2087;
}
{
mul.f16x2 r2295, r2292, r2216;
}
{
add.f16x2 r2298, r2289, r2295;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2302, {low, high};
}
{
add.f16x2 r2303, r2103, r2119;
}
{
add.f16x2 r2306, r1838, r2303;
}
{
add.f16x2 r2309, r2109, r2125;
}
{
add.f16x2 r2312, r1874, r2309;
}
{
add.f16x2 r2315, r2103, r2119;
}
{
mul.f16x2 r2318, r2315, r2301;
}
{
add.f16x2 r2321, r1838, r2318;
}
{
sub.f16x2 r2324, r2109, r2125;
}
{
mul.f16x2 r2327, r2324, r2302;
}
{
add.f16x2 r2330, r2321, r2327;
}
{
add.f16x2 r2333, r2103, r2119;
}
{
mul.f16x2 r2336, r2333, r2301;
}
{
add.f16x2 r2339, r1838, r2336;
}
{
sub.f16x2 r2342, r2109, r2125;
}
{
mul.f16x2 r2345, r2342, r2302;
}
{
sub.f16x2 r2348, r2339, r2345;
}
{
add.f16x2 r2351, r2109, r2125;
}
{
mul.f16x2 r2354, r2351, r2301;
}
{
add.f16x2 r2357, r1874, r2354;
}
{
sub.f16x2 r2360, r2103, r2119;
}
{
mul.f16x2 r2363, r2360, r2302;
}
{
sub.f16x2 r2366, r2357, r2363;
}
{
add.f16x2 r2369, r2109, r2125;
}
{
mul.f16x2 r2372, r2369, r2301;
}
{
add.f16x2 r2375, r1874, r2372;
}
{
sub.f16x2 r2378, r2103, r2119;
}
{
mul.f16x2 r2381, r2378, r2302;
}
{
add.f16x2 r2384, r2375, r2381;
}
mul.wide.u32 rd6, r2950, -901412889;
shr.u64 rd7, rd6, 38;
cvt.u32.u64 r2961, rd7;
mul.lo.s32 r2962, r2961, 81;
sub.s32 r2963, r2950, r2962;
shl.b32 r2964, r2963, 2;
add.s32 r2965, r2951, r2964;
cvt.rn.f32.u32 f247, r2961;
mul.f32 f248, f247, 0f3E6E4BAE;
cos.approx.f32 f209, f248;
sin.approx.f32 f249, f248;
neg.f32 f210, f249;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f209;
cvt.rn.f16.f32 high, f210;
mov.b32 r2387, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2390, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2392, {high, high};
}
{
mul.f16x2 r2394, r2226, r2392;
}
{
fma.rn.f16x2 r2397, r2220, r2390, r2394;
}
{
mul.f16x2 r2401, r2220, r2392;
}
{
neg.f16x2 r2404, r2401;
}
{
fma.rn.f16x2 r2406, r2226, r2390, r2404;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2410, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2412, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2414, {low, high};
}
{
mul.f16x2 r2415, r2412, r2414;
}
{
mul.f16x2 r2418, r2387, r2410;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2421, {high, low};
}
{
fma.rn.f16x2 r2423, r2415, r2421, r2418;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2427, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2429, {high, high};
}
{
mul.f16x2 r2431, r2312, r2429;
}
{
fma.rn.f16x2 r2434, r2306, r2427, r2431;
}
{
mul.f16x2 r2438, r2306, r2429;
}
{
neg.f16x2 r2441, r2438;
}
{
fma.rn.f16x2 r2443, r2312, r2427, r2441;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2447, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2449, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2451, {low, high};
}
{
mul.f16x2 r2452, r2449, r2451;
}
{
mul.f16x2 r2455, r2423, r2447;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2458, {high, low};
}
{
fma.rn.f16x2 r2460, r2452, r2458, r2455;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2460;
mov.b32 r2464, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2460;
mov.b32 r2466, {high, high};
}
{
mul.f16x2 r2468, r2194, r2466;
}
{
fma.rn.f16x2 r2471, r2158, r2464, r2468;
}
{
mul.f16x2 r2475, r2158, r2466;
}
{
neg.f16x2 r2478, r2475;
}
{
fma.rn.f16x2 r2480, r2194, r2464, r2478;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2484, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2486, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2488, {low, high};
}
{
mul.f16x2 r2489, r2486, r2488;
}
{
mul.f16x2 r2492, r2460, r2484;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2460;
mov.b32 r2495, {high, low};
}
{
fma.rn.f16x2 r2497, r2489, r2495, r2492;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2497;
mov.b32 r2501, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2497;
mov.b32 r2503, {high, high};
}
{
mul.f16x2 r2505, r2280, r2503;
}
{
fma.rn.f16x2 r2508, r2244, r2501, r2505;
}
{
mul.f16x2 r2512, r2244, r2503;
}
{
neg.f16x2 r2515, r2512;
}
{
fma.rn.f16x2 r2517, r2280, r2501, r2515;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2521, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2523, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2525, {low, high};
}
{
mul.f16x2 r2526, r2523, r2525;
}
{
mul.f16x2 r2529, r2497, r2521;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2497;
mov.b32 r2532, {high, low};
}
{
fma.rn.f16x2 r2534, r2526, r2532, r2529;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2534;
mov.b32 r2538, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2534;
mov.b32 r2540, {high, high};
}
{
mul.f16x2 r2542, r2366, r2540;
}
{
fma.rn.f16x2 r2545, r2330, r2538, r2542;
}
{
mul.f16x2 r2549, r2330, r2540;
}
{
neg.f16x2 r2552, r2549;
}
{
fma.rn.f16x2 r2554, r2366, r2538, r2552;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2558, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2560, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2562, {low, high};
}
{
mul.f16x2 r2563, r2560, r2562;
}
{
mul.f16x2 r2566, r2534, r2558;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2534;
mov.b32 r2569, {high, low};
}
{
fma.rn.f16x2 r2571, r2563, r2569, r2566;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2571;
mov.b32 r2575, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2571;
mov.b32 r2577, {high, high};
}
{
mul.f16x2 r2579, r2212, r2577;
}
{
fma.rn.f16x2 r2582, r2176, r2575, r2579;
}
{
mul.f16x2 r2586, r2176, r2577;
}
{
neg.f16x2 r2589, r2586;
}
{
fma.rn.f16x2 r2591, r2212, r2575, r2589;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2595, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2597, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2599, {low, high};
}
{
mul.f16x2 r2600, r2597, r2599;
}
{
mul.f16x2 r2603, r2571, r2595;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2571;
mov.b32 r2606, {high, low};
}
{
fma.rn.f16x2 r2608, r2600, r2606, r2603;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2608;
mov.b32 r2612, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2608;
mov.b32 r2614, {high, high};
}
{
mul.f16x2 r2616, r2298, r2614;
}
{
fma.rn.f16x2 r2619, r2262, r2612, r2616;
}
{
mul.f16x2 r2623, r2262, r2614;
}
{
neg.f16x2 r2626, r2623;
}
{
fma.rn.f16x2 r2628, r2298, r2612, r2626;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2632, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2387;
mov.b32 r2634, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2636, {low, high};
}
{
mul.f16x2 r2637, r2634, r2636;
}
{
mul.f16x2 r2640, r2608, r2632;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2608;
mov.b32 r2643, {high, low};
}
{
fma.rn.f16x2 r2645, r2637, r2643, r2640;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2645;
mov.b32 r2649, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2645;
mov.b32 r2651, {high, high};
}
{
mul.f16x2 r2653, r2384, r2651;
}
{
fma.rn.f16x2 r2656, r2348, r2649, r2653;
}
{
mul.f16x2 r2660, r2348, r2651;
}
{
neg.f16x2 r2663, r2660;
}
{
fma.rn.f16x2 r2665, r2384, r2649, r2663;
}
barrier.sync 0;
mad.lo.s32 r2966, r2961, 2916, r2965;
st.shared.u32 [r2966], r2134;
st.shared.u32 [r2966+324], r2397;
st.shared.u32 [r2966+648], r2434;
st.shared.u32 [r2966+972], r2471;
st.shared.u32 [r2966+1296], r2508;
st.shared.u32 [r2966+1620], r2545;
st.shared.u32 [r2966+1944], r2582;
st.shared.u32 [r2966+2268], r2619;
st.shared.u32 [r2966+2592], r2656;
barrier.sync 0;
ld.shared.u32 r2692, [r2954];
ld.shared.u32 r2778, [r2954+972];
ld.shared.u32 r2864, [r2954+1944];
ld.shared.u32 r2689, [r2954+2916];
ld.shared.u32 r2775, [r2954+3888];
ld.shared.u32 r2861, [r2954+4860];
ld.shared.u32 r2690, [r2954+5832];
ld.shared.u32 r2776, [r2954+6804];
ld.shared.u32 r2862, [r2954+7776];
barrier.sync 0;
st.shared.u32 [r2966], r2140;
st.shared.u32 [r2966+324], r2406;
st.shared.u32 [r2966+648], r2443;
st.shared.u32 [r2966+972], r2480;
st.shared.u32 [r2966+1296], r2517;
st.shared.u32 [r2966+1620], r2554;
st.shared.u32 [r2966+1944], r2591;
st.shared.u32 [r2966+2268], r2628;
st.shared.u32 [r2966+2592], r2665;
barrier.sync 0;
ld.shared.u32 r2698, [r2954];
ld.shared.u32 r2784, [r2954+972];
ld.shared.u32 r2870, [r2954+1944];
ld.shared.u32 r2695, [r2954+2916];
ld.shared.u32 r2781, [r2954+3888];
ld.shared.u32 r2867, [r2954+4860];
ld.shared.u32 r2696, [r2954+5832];
ld.shared.u32 r2782, [r2954+6804];
ld.shared.u32 r2868, [r2954+7776];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2686, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2687, {low, high};
}
{
add.f16x2 r2688, r2689, r2690;
}
{
add.f16x2 %0, r2692, r2688;
}
{
add.f16x2 r2694, r2695, r2696;
}
{
add.f16x2 %1, r2698, r2694;
}
{
add.f16x2 r2700, r2689, r2690;
}
{
mul.f16x2 r2703, r2700, r2686;
}
{
add.f16x2 r2706, r2692, r2703;
}
{
sub.f16x2 r2709, r2695, r2696;
}
{
mul.f16x2 r2712, r2709, r2687;
}
{
add.f16x2 %6, r2706, r2712;
}
{
add.f16x2 r2718, r2689, r2690;
}
{
mul.f16x2 r2721, r2718, r2686;
}
{
add.f16x2 r2724, r2692, r2721;
}
{
sub.f16x2 r2727, r2695, r2696;
}
{
mul.f16x2 r2730, r2727, r2687;
}
{
sub.f16x2 %12, r2724, r2730;
}
{
add.f16x2 r2736, r2695, r2696;
}
{
mul.f16x2 r2739, r2736, r2686;
}
{
add.f16x2 r2742, r2698, r2739;
}
{
sub.f16x2 r2745, r2689, r2690;
}
{
mul.f16x2 r2748, r2745, r2687;
}
{
sub.f16x2 %7, r2742, r2748;
}
{
add.f16x2 r2754, r2695, r2696;
}
{
mul.f16x2 r2757, r2754, r2686;
}
{
add.f16x2 r2760, r2698, r2757;
}
{
sub.f16x2 r2763, r2689, r2690;
}
{
mul.f16x2 r2766, r2763, r2687;
}
{
add.f16x2 %13, r2760, r2766;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2772, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2773, {low, high};
}
{
add.f16x2 r2774, r2775, r2776;
}
{
add.f16x2 %2, r2778, r2774;
}
{
add.f16x2 r2780, r2781, r2782;
}
{
add.f16x2 %3, r2784, r2780;
}
{
add.f16x2 r2786, r2775, r2776;
}
{
mul.f16x2 r2789, r2786, r2772;
}
{
add.f16x2 r2792, r2778, r2789;
}
{
sub.f16x2 r2795, r2781, r2782;
}
{
mul.f16x2 r2798, r2795, r2773;
}
{
add.f16x2 %8, r2792, r2798;
}
{
add.f16x2 r2804, r2775, r2776;
}
{
mul.f16x2 r2807, r2804, r2772;
}
{
add.f16x2 r2810, r2778, r2807;
}
{
sub.f16x2 r2813, r2781, r2782;
}
{
mul.f16x2 r2816, r2813, r2773;
}
{
sub.f16x2 %14, r2810, r2816;
}
{
add.f16x2 r2822, r2781, r2782;
}
{
mul.f16x2 r2825, r2822, r2772;
}
{
add.f16x2 r2828, r2784, r2825;
}
{
sub.f16x2 r2831, r2775, r2776;
}
{
mul.f16x2 r2834, r2831, r2773;
}
{
sub.f16x2 %9, r2828, r2834;
}
{
add.f16x2 r2840, r2781, r2782;
}
{
mul.f16x2 r2843, r2840, r2772;
}
{
add.f16x2 r2846, r2784, r2843;
}
{
sub.f16x2 r2849, r2775, r2776;
}
{
mul.f16x2 r2852, r2849, r2773;
}
{
add.f16x2 %15, r2846, r2852;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2858, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2859, {low, high};
}
{
add.f16x2 r2860, r2861, r2862;
}
{
add.f16x2 %4, r2864, r2860;
}
{
add.f16x2 r2866, r2867, r2868;
}
{
add.f16x2 %5, r2870, r2866;
}
{
add.f16x2 r2872, r2861, r2862;
}
{
mul.f16x2 r2875, r2872, r2858;
}
{
add.f16x2 r2878, r2864, r2875;
}
{
sub.f16x2 r2881, r2867, r2868;
}
{
mul.f16x2 r2884, r2881, r2859;
}
{
add.f16x2 %10, r2878, r2884;
}
{
add.f16x2 r2890, r2861, r2862;
}
{
mul.f16x2 r2893, r2890, r2858;
}
{
add.f16x2 r2896, r2864, r2893;
}
{
sub.f16x2 r2899, r2867, r2868;
}
{
mul.f16x2 r2902, r2899, r2859;
}
{
sub.f16x2 %16, r2896, r2902;
}
{
add.f16x2 r2908, r2867, r2868;
}
{
mul.f16x2 r2911, r2908, r2858;
}
{
add.f16x2 r2914, r2870, r2911;
}
{
sub.f16x2 r2917, r2861, r2862;
}
{
mul.f16x2 r2920, r2917, r2859;
}
{
sub.f16x2 %11, r2914, r2920;
}
{
add.f16x2 r2926, r2867, r2868;
}
{
mul.f16x2 r2929, r2926, r2858;
}
{
add.f16x2 r2932, r2870, r2929;
}
{
sub.f16x2 r2935, r2861, r2862;
}
{
mul.f16x2 r2938, r2935, r2859;
}
{
add.f16x2 %17, r2932, r2938;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1095, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<771>;
.reg .b32 r<8520>;
.reg .b64 rd<6>;
mov.u32 r8446, %54;
mov.u32 r8519, %tid.y;
mad.lo.s32 r8447, r8519, 8748, r8446;
mov.u32 r8448, %tid.x;
mov.f32 f762, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1, {low, high};
}
mov.f32 f764, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r2, {low, high};
}
{
add.f16x2 r3, %108, %99;
}
{
add.f16x2 r6, %81, r3;
}
{
add.f16x2 r9, %60, %106;
}
{
add.f16x2 r12, %90, r9;
}
{
add.f16x2 r15, %108, %99;
}
{
mul.f16x2 r18, r15, r1;
}
{
add.f16x2 r21, %81, r18;
}
{
sub.f16x2 r24, %60, %106;
}
{
mul.f16x2 r27, r24, r2;
}
{
add.f16x2 r30, r21, r27;
}
{
add.f16x2 r33, %108, %99;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %81, r36;
}
{
sub.f16x2 r42, %60, %106;
}
{
mul.f16x2 r45, r42, r2;
}
{
sub.f16x2 r48, r39, r45;
}
{
add.f16x2 r51, %60, %106;
}
{
mul.f16x2 r54, r51, r1;
}
{
add.f16x2 r57, %90, r54;
}
{
sub.f16x2 r60, %108, %99;
}
{
mul.f16x2 r63, r60, r2;
}
{
sub.f16x2 r66, r57, r63;
}
{
add.f16x2 r69, %60, %106;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %90, r72;
}
{
sub.f16x2 r78, %108, %99;
}
{
mul.f16x2 r81, r78, r2;
}
{
add.f16x2 r84, r75, r81;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r88, {low, high};
}
{
add.f16x2 r89, %107, %98;
}
{
add.f16x2 r92, %80, r89;
}
{
add.f16x2 r95, %59, %104;
}
{
add.f16x2 r98, %89, r95;
}
{
add.f16x2 r101, %107, %98;
}
{
mul.f16x2 r104, r101, r87;
}
{
add.f16x2 r107, %80, r104;
}
{
sub.f16x2 r110, %59, %104;
}
{
mul.f16x2 r113, r110, r88;
}
{
add.f16x2 r116, r107, r113;
}
{
add.f16x2 r119, %107, %98;
}
{
mul.f16x2 r122, r119, r87;
}
{
add.f16x2 r125, %80, r122;
}
{
sub.f16x2 r128, %59, %104;
}
{
mul.f16x2 r131, r128, r88;
}
{
sub.f16x2 r134, r125, r131;
}
{
add.f16x2 r137, %59, %104;
}
{
mul.f16x2 r140, r137, r87;
}
{
add.f16x2 r143, %89, r140;
}
{
sub.f16x2 r146, %107, %98;
}
{
mul.f16x2 r149, r146, r88;
}
{
sub.f16x2 r152, r143, r149;
}
{
add.f16x2 r155, %59, %104;
}
{
mul.f16x2 r158, r155, r87;
}
{
add.f16x2 r161, %89, r158;
}
{
sub.f16x2 r164, %107, %98;
}
{
mul.f16x2 r167, r164, r88;
}
{
add.f16x2 r170, r161, r167;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r173, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r174, {low, high};
}
{
add.f16x2 r175, %105, %97;
}
{
add.f16x2 r178, %79, r175;
}
{
add.f16x2 r181, %58, %103;
}
{
add.f16x2 r184, %88, r181;
}
{
add.f16x2 r187, %105, %97;
}
{
mul.f16x2 r190, r187, r173;
}
{
add.f16x2 r193, %79, r190;
}
{
sub.f16x2 r196, %58, %103;
}
{
mul.f16x2 r199, r196, r174;
}
{
add.f16x2 r202, r193, r199;
}
{
add.f16x2 r205, %105, %97;
}
{
mul.f16x2 r208, r205, r173;
}
{
add.f16x2 r211, %79, r208;
}
{
sub.f16x2 r214, %58, %103;
}
{
mul.f16x2 r217, r214, r174;
}
{
sub.f16x2 r220, r211, r217;
}
{
add.f16x2 r223, %58, %103;
}
{
mul.f16x2 r226, r223, r173;
}
{
add.f16x2 r229, %88, r226;
}
{
sub.f16x2 r232, %105, %97;
}
{
mul.f16x2 r235, r232, r174;
}
{
sub.f16x2 r238, r229, r235;
}
{
add.f16x2 r241, %58, %103;
}
{
mul.f16x2 r244, r241, r173;
}
{
add.f16x2 r247, %88, r244;
}
{
sub.f16x2 r250, %105, %97;
}
{
mul.f16x2 r253, r250, r174;
}
{
add.f16x2 r256, r247, r253;
}
mov.f32 f542, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r259, {low, high};
}
mov.f32 f544, 0f3F248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r260, {low, high};
}
mov.f32 f554, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r261, {low, high};
}
mov.f32 f556, 0f3F7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r262, {low, high};
}
mov.f32 f578, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r265, {low, high};
}
mov.f32 f580, 0f3EAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r339, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r425, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r597, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r598, {low, high};
}
{
add.f16x2 r599, %96, %84;
}
{
add.f16x2 r602, %66, r599;
}
{
add.f16x2 r605, %102, %94;
}
{
add.f16x2 r608, %72, r605;
}
{
add.f16x2 r611, %96, %84;
}
{
mul.f16x2 r614, r611, r597;
}
{
add.f16x2 r617, %66, r614;
}
{
sub.f16x2 r620, %102, %94;
}
{
mul.f16x2 r623, r620, r598;
}
{
add.f16x2 r626, r617, r623;
}
{
add.f16x2 r629, %96, %84;
}
{
mul.f16x2 r632, r629, r597;
}
{
add.f16x2 r635, %66, r632;
}
{
sub.f16x2 r638, %102, %94;
}
{
mul.f16x2 r641, r638, r598;
}
{
sub.f16x2 r644, r635, r641;
}
{
add.f16x2 r647, %102, %94;
}
{
mul.f16x2 r650, r647, r597;
}
{
add.f16x2 r653, %72, r650;
}
{
sub.f16x2 r656, %96, %84;
}
{
mul.f16x2 r659, r656, r598;
}
{
sub.f16x2 r662, r653, r659;
}
{
add.f16x2 r665, %102, %94;
}
{
mul.f16x2 r668, r665, r597;
}
{
add.f16x2 r671, %72, r668;
}
{
sub.f16x2 r674, %96, %84;
}
{
mul.f16x2 r677, r674, r598;
}
{
add.f16x2 r680, r671, r677;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r683, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r684, {low, high};
}
{
add.f16x2 r685, %95, %83;
}
{
add.f16x2 r688, %65, r685;
}
{
add.f16x2 r691, %101, %92;
}
{
add.f16x2 r694, %71, r691;
}
{
add.f16x2 r697, %95, %83;
}
{
mul.f16x2 r700, r697, r683;
}
{
add.f16x2 r703, %65, r700;
}
{
sub.f16x2 r706, %101, %92;
}
{
mul.f16x2 r709, r706, r684;
}
{
add.f16x2 r712, r703, r709;
}
{
add.f16x2 r715, %95, %83;
}
{
mul.f16x2 r718, r715, r683;
}
{
add.f16x2 r721, %65, r718;
}
{
sub.f16x2 r724, %101, %92;
}
{
mul.f16x2 r727, r724, r684;
}
{
sub.f16x2 r730, r721, r727;
}
{
add.f16x2 r733, %101, %92;
}
{
mul.f16x2 r736, r733, r683;
}
{
add.f16x2 r739, %71, r736;
}
{
sub.f16x2 r742, %95, %83;
}
{
mul.f16x2 r745, r742, r684;
}
{
sub.f16x2 r748, r739, r745;
}
{
add.f16x2 r751, %101, %92;
}
{
mul.f16x2 r754, r751, r683;
}
{
add.f16x2 r757, %71, r754;
}
{
sub.f16x2 r760, %95, %83;
}
{
mul.f16x2 r763, r760, r684;
}
{
add.f16x2 r766, r757, r763;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r769, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r770, {low, high};
}
{
add.f16x2 r771, %93, %82;
}
{
add.f16x2 r774, %64, r771;
}
{
add.f16x2 r777, %100, %91;
}
{
add.f16x2 r780, %70, r777;
}
{
add.f16x2 r783, %93, %82;
}
{
mul.f16x2 r786, r783, r769;
}
{
add.f16x2 r789, %64, r786;
}
{
sub.f16x2 r792, %100, %91;
}
{
mul.f16x2 r795, r792, r770;
}
{
add.f16x2 r798, r789, r795;
}
{
add.f16x2 r801, %93, %82;
}
{
mul.f16x2 r804, r801, r769;
}
{
add.f16x2 r807, %64, r804;
}
{
sub.f16x2 r810, %100, %91;
}
{
mul.f16x2 r813, r810, r770;
}
{
sub.f16x2 r816, r807, r813;
}
{
add.f16x2 r819, %100, %91;
}
{
mul.f16x2 r822, r819, r769;
}
{
add.f16x2 r825, %70, r822;
}
{
sub.f16x2 r828, %93, %82;
}
{
mul.f16x2 r831, r828, r770;
}
{
sub.f16x2 r834, r825, r831;
}
{
add.f16x2 r837, %100, %91;
}
{
mul.f16x2 r840, r837, r769;
}
{
add.f16x2 r843, %70, r840;
}
{
sub.f16x2 r846, %93, %82;
}
{
mul.f16x2 r849, r846, r770;
}
{
add.f16x2 r852, r843, r849;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r855, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r856, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r857, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r858, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r861, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r935, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1107, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1193, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r1194, {low, high};
}
{
add.f16x2 r1195, %78, %69;
}
{
add.f16x2 r1198, %57, r1195;
}
{
add.f16x2 r1201, %87, %76;
}
{
add.f16x2 r1204, %63, r1201;
}
{
add.f16x2 r1207, %78, %69;
}
{
mul.f16x2 r1210, r1207, r1193;
}
{
add.f16x2 r1213, %57, r1210;
}
{
sub.f16x2 r1216, %87, %76;
}
{
mul.f16x2 r1219, r1216, r1194;
}
{
add.f16x2 r1222, r1213, r1219;
}
{
add.f16x2 r1225, %78, %69;
}
{
mul.f16x2 r1228, r1225, r1193;
}
{
add.f16x2 r1231, %57, r1228;
}
{
sub.f16x2 r1234, %87, %76;
}
{
mul.f16x2 r1237, r1234, r1194;
}
{
sub.f16x2 r1240, r1231, r1237;
}
{
add.f16x2 r1243, %87, %76;
}
{
mul.f16x2 r1246, r1243, r1193;
}
{
add.f16x2 r1249, %63, r1246;
}
{
sub.f16x2 r1252, %78, %69;
}
{
mul.f16x2 r1255, r1252, r1194;
}
{
sub.f16x2 r1258, r1249, r1255;
}
{
add.f16x2 r1261, %87, %76;
}
{
mul.f16x2 r1264, r1261, r1193;
}
{
add.f16x2 r1267, %63, r1264;
}
{
sub.f16x2 r1270, %78, %69;
}
{
mul.f16x2 r1273, r1270, r1194;
}
{
add.f16x2 r1276, r1267, r1273;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1279, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r1280, {low, high};
}
{
add.f16x2 r1281, %77, %68;
}
{
add.f16x2 r1284, %56, r1281;
}
{
add.f16x2 r1287, %86, %74;
}
{
add.f16x2 r1290, %62, r1287;
}
{
add.f16x2 r1293, %77, %68;
}
{
mul.f16x2 r1296, r1293, r1279;
}
{
add.f16x2 r1299, %56, r1296;
}
{
sub.f16x2 r1302, %86, %74;
}
{
mul.f16x2 r1305, r1302, r1280;
}
{
add.f16x2 r1308, r1299, r1305;
}
{
add.f16x2 r1311, %77, %68;
}
{
mul.f16x2 r1314, r1311, r1279;
}
{
add.f16x2 r1317, %56, r1314;
}
{
sub.f16x2 r1320, %86, %74;
}
{
mul.f16x2 r1323, r1320, r1280;
}
{
sub.f16x2 r1326, r1317, r1323;
}
{
add.f16x2 r1329, %86, %74;
}
{
mul.f16x2 r1332, r1329, r1279;
}
{
add.f16x2 r1335, %62, r1332;
}
{
sub.f16x2 r1338, %77, %68;
}
{
mul.f16x2 r1341, r1338, r1280;
}
{
sub.f16x2 r1344, r1335, r1341;
}
{
add.f16x2 r1347, %86, %74;
}
{
mul.f16x2 r1350, r1347, r1279;
}
{
add.f16x2 r1353, %62, r1350;
}
{
sub.f16x2 r1356, %77, %68;
}
{
mul.f16x2 r1359, r1356, r1280;
}
{
add.f16x2 r1362, r1353, r1359;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1365, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r1366, {low, high};
}
{
add.f16x2 r1367, %75, %67;
}
{
add.f16x2 r1370, %55, r1367;
}
{
add.f16x2 r1373, %85, %73;
}
{
add.f16x2 r1376, %61, r1373;
}
{
add.f16x2 r1379, %75, %67;
}
{
mul.f16x2 r1382, r1379, r1365;
}
{
add.f16x2 r1385, %55, r1382;
}
{
sub.f16x2 r1388, %85, %73;
}
{
mul.f16x2 r1391, r1388, r1366;
}
{
add.f16x2 r1394, r1385, r1391;
}
{
add.f16x2 r1397, %75, %67;
}
{
mul.f16x2 r1400, r1397, r1365;
}
{
add.f16x2 r1403, %55, r1400;
}
{
sub.f16x2 r1406, %85, %73;
}
{
mul.f16x2 r1409, r1406, r1366;
}
{
sub.f16x2 r1412, r1403, r1409;
}
{
add.f16x2 r1415, %85, %73;
}
{
mul.f16x2 r1418, r1415, r1365;
}
{
add.f16x2 r1421, %61, r1418;
}
{
sub.f16x2 r1424, %75, %67;
}
{
mul.f16x2 r1427, r1424, r1366;
}
{
sub.f16x2 r1430, r1421, r1427;
}
{
add.f16x2 r1433, %85, %73;
}
{
mul.f16x2 r1436, r1433, r1365;
}
{
add.f16x2 r1439, %61, r1436;
}
{
sub.f16x2 r1442, %75, %67;
}
{
mul.f16x2 r1445, r1442, r1366;
}
{
add.f16x2 r1448, r1439, r1445;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r1451, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r1452, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r1453, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r1454, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r1457, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1531, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1617, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1703, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
mov.f32 f534, 0f3F791978;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f534;
cvt.rn.f16.f32 high, f534;
mov.b32 r1789, {low, high};
}
mov.f32 f536, 0f3E6C2691;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f536;
cvt.rn.f16.f32 high, f536;
mov.b32 r1790, {low, high};
}
mov.f32 f538, 0f3F64C51C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f538;
cvt.rn.f16.f32 high, f538;
mov.b32 r1791, {low, high};
}
mov.f32 f540, 0f3EE5C902;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f540;
cvt.rn.f16.f32 high, f540;
mov.b32 r1792, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r1793, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r1794, {low, high};
}
mov.f32 f546, 0f3F18DF63;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f546;
cvt.rn.f16.f32 high, f546;
mov.b32 r1795, {low, high};
}
mov.f32 f548, 0f3F4D57F2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f548;
cvt.rn.f16.f32 high, f548;
mov.b32 r1796, {low, high};
}
mov.f32 f550, 0f3ECACAF8;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f550;
cvt.rn.f16.f32 high, f550;
mov.b32 r1797, {low, high};
}
mov.f32 f552, 0f3F6B1036;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f552;
cvt.rn.f16.f32 high, f552;
mov.b32 r1798, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r1799, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r1800, {low, high};
}
mov.f32 f558, 0fBD6E2946;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f558;
cvt.rn.f16.f32 high, f558;
mov.b32 r1801, {low, high};
}
mov.f32 f560, 0f3F7F9120;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f560;
cvt.rn.f16.f32 high, f560;
mov.b32 r1802, {low, high};
}
mov.f32 f562, 0fBE92D7E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f562;
cvt.rn.f16.f32 high, f562;
mov.b32 r1803, {low, high};
}
mov.f32 f564, 0f3F753ECD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f564;
cvt.rn.f16.f32 high, f564;
mov.b32 r1804, {low, high};
}
mov.f32 f570, 0fBF2FAD88;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f570;
cvt.rn.f16.f32 high, f570;
mov.b32 r1807, {low, high};
}
mov.f32 f572, 0f3F3A3529;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f572;
cvt.rn.f16.f32 high, f572;
mov.b32 r1808, {low, high};
}
mov.f32 f594, 0fBF55E287;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r1811, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
mov.b32 r1812, {low, high};
}
mov.f32 f586, 0fBF7E44DE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f586;
cvt.rn.f16.f32 high, f586;
mov.b32 r1815, {low, high};
}
mov.f32 f588, 0fBDEDC21F;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f588;
cvt.rn.f16.f32 high, f588;
mov.b32 r1816, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1819, {low, high};
}
mov.f32 f596, 0fBF0CAC9F;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2097, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2183, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2269, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2355, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2441, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2527, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2613, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2699, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2785, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
mul.wide.u32 rd2, r8448, -901412889;
shr.u64 rd3, rd2, 38;
cvt.u32.u64 r8449, rd3;
mul.lo.s32 r8450, r8449, 81;
sub.s32 r8451, r8448, r8450;
mad.lo.s32 r8452, r8449, 8748, r8447;
cvt.rn.f32.u32 f765, r8451;
mul.f32 f766, f765, 0f3B3C4870;
cos.approx.f32 f309, f766;
sin.approx.f32 f767, f766;
neg.f32 f310, f767;
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
mov.f32 f725, 0fBF800000;
mov.f32 f726, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
mad.lo.s32 r8453, r8451, 108, r8452;
st.shared.u32 [r8453], r2102;
st.shared.u32 [r8453+4], r2881;
st.shared.u32 [r8453+8], r2918;
st.shared.u32 [r8453+12], r2955;
st.shared.u32 [r8453+16], r2992;
st.shared.u32 [r8453+20], r3029;
st.shared.u32 [r8453+24], r3066;
st.shared.u32 [r8453+28], r3103;
st.shared.u32 [r8453+32], r3140;
st.shared.u32 [r8453+36], r3177;
st.shared.u32 [r8453+40], r3214;
st.shared.u32 [r8453+44], r3251;
st.shared.u32 [r8453+48], r3288;
st.shared.u32 [r8453+52], r3325;
st.shared.u32 [r8453+56], r3362;
st.shared.u32 [r8453+60], r3399;
st.shared.u32 [r8453+64], r3436;
st.shared.u32 [r8453+68], r3473;
st.shared.u32 [r8453+72], r3510;
st.shared.u32 [r8453+76], r3547;
st.shared.u32 [r8453+80], r3584;
st.shared.u32 [r8453+84], r3621;
st.shared.u32 [r8453+88], r3658;
st.shared.u32 [r8453+92], r3695;
st.shared.u32 [r8453+96], r3732;
st.shared.u32 [r8453+100], r3769;
st.shared.u32 [r8453+104], r3806;
barrier.sync 0;
mad.lo.s32 r8454, r8451, -104, r8453;
ld.shared.u32 r3842, [r8454];
ld.shared.u32 r4438, [r8454+324];
ld.shared.u32 r5034, [r8454+648];
ld.shared.u32 r3928, [r8454+972];
ld.shared.u32 r4524, [r8454+1296];
ld.shared.u32 r5120, [r8454+1620];
ld.shared.u32 r4014, [r8454+1944];
ld.shared.u32 r4610, [r8454+2268];
ld.shared.u32 r5206, [r8454+2592];
ld.shared.u32 r3839, [r8454+2916];
ld.shared.u32 r4435, [r8454+3240];
ld.shared.u32 r5031, [r8454+3564];
ld.shared.u32 r3925, [r8454+3888];
ld.shared.u32 r4521, [r8454+4212];
ld.shared.u32 r5117, [r8454+4536];
ld.shared.u32 r4011, [r8454+4860];
ld.shared.u32 r4607, [r8454+5184];
ld.shared.u32 r5203, [r8454+5508];
ld.shared.u32 r3840, [r8454+5832];
ld.shared.u32 r4436, [r8454+6156];
ld.shared.u32 r5032, [r8454+6480];
ld.shared.u32 r3926, [r8454+6804];
ld.shared.u32 r4522, [r8454+7128];
ld.shared.u32 r5118, [r8454+7452];
ld.shared.u32 r4012, [r8454+7776];
ld.shared.u32 r4608, [r8454+8100];
ld.shared.u32 r5204, [r8454+8424];
barrier.sync 0;
st.shared.u32 [r8453], r2108;
st.shared.u32 [r8453+4], r2890;
st.shared.u32 [r8453+8], r2927;
st.shared.u32 [r8453+12], r2964;
st.shared.u32 [r8453+16], r3001;
st.shared.u32 [r8453+20], r3038;
st.shared.u32 [r8453+24], r3075;
st.shared.u32 [r8453+28], r3112;
st.shared.u32 [r8453+32], r3149;
st.shared.u32 [r8453+36], r3186;
st.shared.u32 [r8453+40], r3223;
st.shared.u32 [r8453+44], r3260;
st.shared.u32 [r8453+48], r3297;
st.shared.u32 [r8453+52], r3334;
st.shared.u32 [r8453+56], r3371;
st.shared.u32 [r8453+60], r3408;
st.shared.u32 [r8453+64], r3445;
st.shared.u32 [r8453+68], r3482;
st.shared.u32 [r8453+72], r3519;
st.shared.u32 [r8453+76], r3556;
st.shared.u32 [r8453+80], r3593;
st.shared.u32 [r8453+84], r3630;
st.shared.u32 [r8453+88], r3667;
st.shared.u32 [r8453+92], r3704;
st.shared.u32 [r8453+96], r3741;
st.shared.u32 [r8453+100], r3778;
st.shared.u32 [r8453+104], r3815;
barrier.sync 0;
ld.shared.u32 r3848, [r8454];
ld.shared.u32 r4444, [r8454+324];
ld.shared.u32 r5040, [r8454+648];
ld.shared.u32 r3934, [r8454+972];
ld.shared.u32 r4530, [r8454+1296];
ld.shared.u32 r5126, [r8454+1620];
ld.shared.u32 r4020, [r8454+1944];
ld.shared.u32 r4616, [r8454+2268];
ld.shared.u32 r5212, [r8454+2592];
ld.shared.u32 r3845, [r8454+2916];
ld.shared.u32 r4441, [r8454+3240];
ld.shared.u32 r5037, [r8454+3564];
ld.shared.u32 r3931, [r8454+3888];
ld.shared.u32 r4527, [r8454+4212];
ld.shared.u32 r5123, [r8454+4536];
ld.shared.u32 r4017, [r8454+4860];
ld.shared.u32 r4613, [r8454+5184];
ld.shared.u32 r5209, [r8454+5508];
ld.shared.u32 r3846, [r8454+5832];
ld.shared.u32 r4442, [r8454+6156];
ld.shared.u32 r5038, [r8454+6480];
ld.shared.u32 r3932, [r8454+6804];
ld.shared.u32 r4528, [r8454+7128];
ld.shared.u32 r5124, [r8454+7452];
ld.shared.u32 r4018, [r8454+7776];
ld.shared.u32 r4614, [r8454+8100];
ld.shared.u32 r5210, [r8454+8424];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r3836, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r3837, {low, high};
}
{
add.f16x2 r3838, r3839, r3840;
}
{
add.f16x2 r3841, r3842, r3838;
}
{
add.f16x2 r3844, r3845, r3846;
}
{
add.f16x2 r3847, r3848, r3844;
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
add.f16x2 r3865, r3856, r3862;
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
sub.f16x2 r3883, r3874, r3880;
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
sub.f16x2 r3901, r3892, r3898;
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
add.f16x2 r3919, r3910, r3916;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r3922, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r3923, {low, high};
}
{
add.f16x2 r3924, r3925, r3926;
}
{
add.f16x2 r3927, r3928, r3924;
}
{
add.f16x2 r3930, r3931, r3932;
}
{
add.f16x2 r3933, r3934, r3930;
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
add.f16x2 r3951, r3942, r3948;
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
sub.f16x2 r3969, r3960, r3966;
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
sub.f16x2 r3987, r3978, r3984;
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
add.f16x2 r4005, r3996, r4002;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4008, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4009, {low, high};
}
{
add.f16x2 r4010, r4011, r4012;
}
{
add.f16x2 r4013, r4014, r4010;
}
{
add.f16x2 r4016, r4017, r4018;
}
{
add.f16x2 r4019, r4020, r4016;
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
add.f16x2 r4037, r4028, r4034;
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
sub.f16x2 r4055, r4046, r4052;
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
sub.f16x2 r4073, r4064, r4070;
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
add.f16x2 r4091, r4082, r4088;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r4094, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r4095, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r4096, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r4097, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r4100, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
mov.b32 r4101, {low, high};
}
{
mul.f16x2 r4110, r3951, r4094;
}
{
mul.f16x2 r4113, r3987, r4095;
}
{
sub.f16x2 r4116, r4110, r4113;
}
{
mul.f16x2 r4119, r3951, r4095;
}
{
fma.rn.f16x2 r4122, r3987, r4094, r4119;
}
{
mul.f16x2 r4126, r4037, r4096;
}
{
mul.f16x2 r4129, r4073, r4097;
}
{
sub.f16x2 r4132, r4126, r4129;
}
{
mul.f16x2 r4135, r4037, r4097;
}
{
fma.rn.f16x2 r4138, r4073, r4096, r4135;
}
{
mul.f16x2 r4142, r3969, r4096;
}
{
mul.f16x2 r4145, r4005, r4097;
}
{
sub.f16x2 r4148, r4142, r4145;
}
{
mul.f16x2 r4151, r3969, r4097;
}
{
fma.rn.f16x2 r4154, r4005, r4096, r4151;
}
{
mul.f16x2 r4158, r4055, r4100;
}
{
mul.f16x2 r4161, r4091, r4101;
}
{
sub.f16x2 r4164, r4158, r4161;
}
{
mul.f16x2 r4167, r4055, r4101;
}
{
fma.rn.f16x2 r4170, r4091, r4100, r4167;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4174, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4175, {low, high};
}
{
add.f16x2 r4176, r3927, r4013;
}
{
add.f16x2 r4179, r3841, r4176;
}
{
add.f16x2 r4182, r3933, r4019;
}
{
add.f16x2 r4185, r3847, r4182;
}
{
add.f16x2 r4188, r3927, r4013;
}
{
mul.f16x2 r4191, r4188, r4174;
}
{
add.f16x2 r4194, r3841, r4191;
}
{
sub.f16x2 r4197, r3933, r4019;
}
{
mul.f16x2 r4200, r4197, r4175;
}
{
add.f16x2 r4203, r4194, r4200;
}
{
add.f16x2 r4206, r3927, r4013;
}
{
mul.f16x2 r4209, r4206, r4174;
}
{
add.f16x2 r4212, r3841, r4209;
}
{
sub.f16x2 r4215, r3933, r4019;
}
{
mul.f16x2 r4218, r4215, r4175;
}
{
sub.f16x2 r4221, r4212, r4218;
}
{
add.f16x2 r4224, r3933, r4019;
}
{
mul.f16x2 r4227, r4224, r4174;
}
{
add.f16x2 r4230, r3847, r4227;
}
{
sub.f16x2 r4233, r3927, r4013;
}
{
mul.f16x2 r4236, r4233, r4175;
}
{
sub.f16x2 r4239, r4230, r4236;
}
{
add.f16x2 r4242, r3933, r4019;
}
{
mul.f16x2 r4245, r4242, r4174;
}
{
add.f16x2 r4248, r3847, r4245;
}
{
sub.f16x2 r4251, r3927, r4013;
}
{
mul.f16x2 r4254, r4251, r4175;
}
{
add.f16x2 r4257, r4248, r4254;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4260, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4261, {low, high};
}
{
add.f16x2 r4262, r4116, r4132;
}
{
add.f16x2 r4265, r3865, r4262;
}
{
add.f16x2 r4268, r4122, r4138;
}
{
add.f16x2 r4271, r3901, r4268;
}
{
add.f16x2 r4274, r4116, r4132;
}
{
mul.f16x2 r4277, r4274, r4260;
}
{
add.f16x2 r4280, r3865, r4277;
}
{
sub.f16x2 r4283, r4122, r4138;
}
{
mul.f16x2 r4286, r4283, r4261;
}
{
add.f16x2 r4289, r4280, r4286;
}
{
add.f16x2 r4292, r4116, r4132;
}
{
mul.f16x2 r4295, r4292, r4260;
}
{
add.f16x2 r4298, r3865, r4295;
}
{
sub.f16x2 r4301, r4122, r4138;
}
{
mul.f16x2 r4304, r4301, r4261;
}
{
sub.f16x2 r4307, r4298, r4304;
}
{
add.f16x2 r4310, r4122, r4138;
}
{
mul.f16x2 r4313, r4310, r4260;
}
{
add.f16x2 r4316, r3901, r4313;
}
{
sub.f16x2 r4319, r4116, r4132;
}
{
mul.f16x2 r4322, r4319, r4261;
}
{
sub.f16x2 r4325, r4316, r4322;
}
{
add.f16x2 r4328, r4122, r4138;
}
{
mul.f16x2 r4331, r4328, r4260;
}
{
add.f16x2 r4334, r3901, r4331;
}
{
sub.f16x2 r4337, r4116, r4132;
}
{
mul.f16x2 r4340, r4337, r4261;
}
{
add.f16x2 r4343, r4334, r4340;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4346, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4347, {low, high};
}
{
add.f16x2 r4348, r4148, r4164;
}
{
add.f16x2 r4351, r3883, r4348;
}
{
add.f16x2 r4354, r4154, r4170;
}
{
add.f16x2 r4357, r3919, r4354;
}
{
add.f16x2 r4360, r4148, r4164;
}
{
mul.f16x2 r4363, r4360, r4346;
}
{
add.f16x2 r4366, r3883, r4363;
}
{
sub.f16x2 r4369, r4154, r4170;
}
{
mul.f16x2 r4372, r4369, r4347;
}
{
add.f16x2 r4375, r4366, r4372;
}
{
add.f16x2 r4378, r4148, r4164;
}
{
mul.f16x2 r4381, r4378, r4346;
}
{
add.f16x2 r4384, r3883, r4381;
}
{
sub.f16x2 r4387, r4154, r4170;
}
{
mul.f16x2 r4390, r4387, r4347;
}
{
sub.f16x2 r4393, r4384, r4390;
}
{
add.f16x2 r4396, r4154, r4170;
}
{
mul.f16x2 r4399, r4396, r4346;
}
{
add.f16x2 r4402, r3919, r4399;
}
{
sub.f16x2 r4405, r4148, r4164;
}
{
mul.f16x2 r4408, r4405, r4347;
}
{
sub.f16x2 r4411, r4402, r4408;
}
{
add.f16x2 r4414, r4154, r4170;
}
{
mul.f16x2 r4417, r4414, r4346;
}
{
add.f16x2 r4420, r3919, r4417;
}
{
sub.f16x2 r4423, r4148, r4164;
}
{
mul.f16x2 r4426, r4423, r4347;
}
{
add.f16x2 r4429, r4420, r4426;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4432, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4433, {low, high};
}
{
add.f16x2 r4434, r4435, r4436;
}
{
add.f16x2 r4437, r4438, r4434;
}
{
add.f16x2 r4440, r4441, r4442;
}
{
add.f16x2 r4443, r4444, r4440;
}
{
add.f16x2 r4446, r4435, r4436;
}
{
mul.f16x2 r4449, r4446, r4432;
}
{
add.f16x2 r4452, r4438, r4449;
}
{
sub.f16x2 r4455, r4441, r4442;
}
{
mul.f16x2 r4458, r4455, r4433;
}
{
add.f16x2 r4461, r4452, r4458;
}
{
add.f16x2 r4464, r4435, r4436;
}
{
mul.f16x2 r4467, r4464, r4432;
}
{
add.f16x2 r4470, r4438, r4467;
}
{
sub.f16x2 r4473, r4441, r4442;
}
{
mul.f16x2 r4476, r4473, r4433;
}
{
sub.f16x2 r4479, r4470, r4476;
}
{
add.f16x2 r4482, r4441, r4442;
}
{
mul.f16x2 r4485, r4482, r4432;
}
{
add.f16x2 r4488, r4444, r4485;
}
{
sub.f16x2 r4491, r4435, r4436;
}
{
mul.f16x2 r4494, r4491, r4433;
}
{
sub.f16x2 r4497, r4488, r4494;
}
{
add.f16x2 r4500, r4441, r4442;
}
{
mul.f16x2 r4503, r4500, r4432;
}
{
add.f16x2 r4506, r4444, r4503;
}
{
sub.f16x2 r4509, r4435, r4436;
}
{
mul.f16x2 r4512, r4509, r4433;
}
{
add.f16x2 r4515, r4506, r4512;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4518, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4519, {low, high};
}
{
add.f16x2 r4520, r4521, r4522;
}
{
add.f16x2 r4523, r4524, r4520;
}
{
add.f16x2 r4526, r4527, r4528;
}
{
add.f16x2 r4529, r4530, r4526;
}
{
add.f16x2 r4532, r4521, r4522;
}
{
mul.f16x2 r4535, r4532, r4518;
}
{
add.f16x2 r4538, r4524, r4535;
}
{
sub.f16x2 r4541, r4527, r4528;
}
{
mul.f16x2 r4544, r4541, r4519;
}
{
add.f16x2 r4547, r4538, r4544;
}
{
add.f16x2 r4550, r4521, r4522;
}
{
mul.f16x2 r4553, r4550, r4518;
}
{
add.f16x2 r4556, r4524, r4553;
}
{
sub.f16x2 r4559, r4527, r4528;
}
{
mul.f16x2 r4562, r4559, r4519;
}
{
sub.f16x2 r4565, r4556, r4562;
}
{
add.f16x2 r4568, r4527, r4528;
}
{
mul.f16x2 r4571, r4568, r4518;
}
{
add.f16x2 r4574, r4530, r4571;
}
{
sub.f16x2 r4577, r4521, r4522;
}
{
mul.f16x2 r4580, r4577, r4519;
}
{
sub.f16x2 r4583, r4574, r4580;
}
{
add.f16x2 r4586, r4527, r4528;
}
{
mul.f16x2 r4589, r4586, r4518;
}
{
add.f16x2 r4592, r4530, r4589;
}
{
sub.f16x2 r4595, r4521, r4522;
}
{
mul.f16x2 r4598, r4595, r4519;
}
{
add.f16x2 r4601, r4592, r4598;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4604, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4605, {low, high};
}
{
add.f16x2 r4606, r4607, r4608;
}
{
add.f16x2 r4609, r4610, r4606;
}
{
add.f16x2 r4612, r4613, r4614;
}
{
add.f16x2 r4615, r4616, r4612;
}
{
add.f16x2 r4618, r4607, r4608;
}
{
mul.f16x2 r4621, r4618, r4604;
}
{
add.f16x2 r4624, r4610, r4621;
}
{
sub.f16x2 r4627, r4613, r4614;
}
{
mul.f16x2 r4630, r4627, r4605;
}
{
add.f16x2 r4633, r4624, r4630;
}
{
add.f16x2 r4636, r4607, r4608;
}
{
mul.f16x2 r4639, r4636, r4604;
}
{
add.f16x2 r4642, r4610, r4639;
}
{
sub.f16x2 r4645, r4613, r4614;
}
{
mul.f16x2 r4648, r4645, r4605;
}
{
sub.f16x2 r4651, r4642, r4648;
}
{
add.f16x2 r4654, r4613, r4614;
}
{
mul.f16x2 r4657, r4654, r4604;
}
{
add.f16x2 r4660, r4616, r4657;
}
{
sub.f16x2 r4663, r4607, r4608;
}
{
mul.f16x2 r4666, r4663, r4605;
}
{
sub.f16x2 r4669, r4660, r4666;
}
{
add.f16x2 r4672, r4613, r4614;
}
{
mul.f16x2 r4675, r4672, r4604;
}
{
add.f16x2 r4678, r4616, r4675;
}
{
sub.f16x2 r4681, r4607, r4608;
}
{
mul.f16x2 r4684, r4681, r4605;
}
{
add.f16x2 r4687, r4678, r4684;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r4690, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r4691, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r4692, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r4693, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r4696, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
mov.b32 r4697, {low, high};
}
{
mul.f16x2 r4706, r4547, r4690;
}
{
mul.f16x2 r4709, r4583, r4691;
}
{
sub.f16x2 r4712, r4706, r4709;
}
{
mul.f16x2 r4715, r4547, r4691;
}
{
fma.rn.f16x2 r4718, r4583, r4690, r4715;
}
{
mul.f16x2 r4722, r4633, r4692;
}
{
mul.f16x2 r4725, r4669, r4693;
}
{
sub.f16x2 r4728, r4722, r4725;
}
{
mul.f16x2 r4731, r4633, r4693;
}
{
fma.rn.f16x2 r4734, r4669, r4692, r4731;
}
{
mul.f16x2 r4738, r4565, r4692;
}
{
mul.f16x2 r4741, r4601, r4693;
}
{
sub.f16x2 r4744, r4738, r4741;
}
{
mul.f16x2 r4747, r4565, r4693;
}
{
fma.rn.f16x2 r4750, r4601, r4692, r4747;
}
{
mul.f16x2 r4754, r4651, r4696;
}
{
mul.f16x2 r4757, r4687, r4697;
}
{
sub.f16x2 r4760, r4754, r4757;
}
{
mul.f16x2 r4763, r4651, r4697;
}
{
fma.rn.f16x2 r4766, r4687, r4696, r4763;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4770, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4771, {low, high};
}
{
add.f16x2 r4772, r4523, r4609;
}
{
add.f16x2 r4775, r4437, r4772;
}
{
add.f16x2 r4778, r4529, r4615;
}
{
add.f16x2 r4781, r4443, r4778;
}
{
add.f16x2 r4784, r4523, r4609;
}
{
mul.f16x2 r4787, r4784, r4770;
}
{
add.f16x2 r4790, r4437, r4787;
}
{
sub.f16x2 r4793, r4529, r4615;
}
{
mul.f16x2 r4796, r4793, r4771;
}
{
add.f16x2 r4799, r4790, r4796;
}
{
add.f16x2 r4802, r4523, r4609;
}
{
mul.f16x2 r4805, r4802, r4770;
}
{
add.f16x2 r4808, r4437, r4805;
}
{
sub.f16x2 r4811, r4529, r4615;
}
{
mul.f16x2 r4814, r4811, r4771;
}
{
sub.f16x2 r4817, r4808, r4814;
}
{
add.f16x2 r4820, r4529, r4615;
}
{
mul.f16x2 r4823, r4820, r4770;
}
{
add.f16x2 r4826, r4443, r4823;
}
{
sub.f16x2 r4829, r4523, r4609;
}
{
mul.f16x2 r4832, r4829, r4771;
}
{
sub.f16x2 r4835, r4826, r4832;
}
{
add.f16x2 r4838, r4529, r4615;
}
{
mul.f16x2 r4841, r4838, r4770;
}
{
add.f16x2 r4844, r4443, r4841;
}
{
sub.f16x2 r4847, r4523, r4609;
}
{
mul.f16x2 r4850, r4847, r4771;
}
{
add.f16x2 r4853, r4844, r4850;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4856, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4857, {low, high};
}
{
add.f16x2 r4858, r4712, r4728;
}
{
add.f16x2 r4861, r4461, r4858;
}
{
add.f16x2 r4864, r4718, r4734;
}
{
add.f16x2 r4867, r4497, r4864;
}
{
add.f16x2 r4870, r4712, r4728;
}
{
mul.f16x2 r4873, r4870, r4856;
}
{
add.f16x2 r4876, r4461, r4873;
}
{
sub.f16x2 r4879, r4718, r4734;
}
{
mul.f16x2 r4882, r4879, r4857;
}
{
add.f16x2 r4885, r4876, r4882;
}
{
add.f16x2 r4888, r4712, r4728;
}
{
mul.f16x2 r4891, r4888, r4856;
}
{
add.f16x2 r4894, r4461, r4891;
}
{
sub.f16x2 r4897, r4718, r4734;
}
{
mul.f16x2 r4900, r4897, r4857;
}
{
sub.f16x2 r4903, r4894, r4900;
}
{
add.f16x2 r4906, r4718, r4734;
}
{
mul.f16x2 r4909, r4906, r4856;
}
{
add.f16x2 r4912, r4497, r4909;
}
{
sub.f16x2 r4915, r4712, r4728;
}
{
mul.f16x2 r4918, r4915, r4857;
}
{
sub.f16x2 r4921, r4912, r4918;
}
{
add.f16x2 r4924, r4718, r4734;
}
{
mul.f16x2 r4927, r4924, r4856;
}
{
add.f16x2 r4930, r4497, r4927;
}
{
sub.f16x2 r4933, r4712, r4728;
}
{
mul.f16x2 r4936, r4933, r4857;
}
{
add.f16x2 r4939, r4930, r4936;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4942, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4943, {low, high};
}
{
add.f16x2 r4944, r4744, r4760;
}
{
add.f16x2 r4947, r4479, r4944;
}
{
add.f16x2 r4950, r4750, r4766;
}
{
add.f16x2 r4953, r4515, r4950;
}
{
add.f16x2 r4956, r4744, r4760;
}
{
mul.f16x2 r4959, r4956, r4942;
}
{
add.f16x2 r4962, r4479, r4959;
}
{
sub.f16x2 r4965, r4750, r4766;
}
{
mul.f16x2 r4968, r4965, r4943;
}
{
add.f16x2 r4971, r4962, r4968;
}
{
add.f16x2 r4974, r4744, r4760;
}
{
mul.f16x2 r4977, r4974, r4942;
}
{
add.f16x2 r4980, r4479, r4977;
}
{
sub.f16x2 r4983, r4750, r4766;
}
{
mul.f16x2 r4986, r4983, r4943;
}
{
sub.f16x2 r4989, r4980, r4986;
}
{
add.f16x2 r4992, r4750, r4766;
}
{
mul.f16x2 r4995, r4992, r4942;
}
{
add.f16x2 r4998, r4515, r4995;
}
{
sub.f16x2 r5001, r4744, r4760;
}
{
mul.f16x2 r5004, r5001, r4943;
}
{
sub.f16x2 r5007, r4998, r5004;
}
{
add.f16x2 r5010, r4750, r4766;
}
{
mul.f16x2 r5013, r5010, r4942;
}
{
add.f16x2 r5016, r4515, r5013;
}
{
sub.f16x2 r5019, r4744, r4760;
}
{
mul.f16x2 r5022, r5019, r4943;
}
{
add.f16x2 r5025, r5016, r5022;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5028, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5029, {low, high};
}
{
add.f16x2 r5030, r5031, r5032;
}
{
add.f16x2 r5033, r5034, r5030;
}
{
add.f16x2 r5036, r5037, r5038;
}
{
add.f16x2 r5039, r5040, r5036;
}
{
add.f16x2 r5042, r5031, r5032;
}
{
mul.f16x2 r5045, r5042, r5028;
}
{
add.f16x2 r5048, r5034, r5045;
}
{
sub.f16x2 r5051, r5037, r5038;
}
{
mul.f16x2 r5054, r5051, r5029;
}
{
add.f16x2 r5057, r5048, r5054;
}
{
add.f16x2 r5060, r5031, r5032;
}
{
mul.f16x2 r5063, r5060, r5028;
}
{
add.f16x2 r5066, r5034, r5063;
}
{
sub.f16x2 r5069, r5037, r5038;
}
{
mul.f16x2 r5072, r5069, r5029;
}
{
sub.f16x2 r5075, r5066, r5072;
}
{
add.f16x2 r5078, r5037, r5038;
}
{
mul.f16x2 r5081, r5078, r5028;
}
{
add.f16x2 r5084, r5040, r5081;
}
{
sub.f16x2 r5087, r5031, r5032;
}
{
mul.f16x2 r5090, r5087, r5029;
}
{
sub.f16x2 r5093, r5084, r5090;
}
{
add.f16x2 r5096, r5037, r5038;
}
{
mul.f16x2 r5099, r5096, r5028;
}
{
add.f16x2 r5102, r5040, r5099;
}
{
sub.f16x2 r5105, r5031, r5032;
}
{
mul.f16x2 r5108, r5105, r5029;
}
{
add.f16x2 r5111, r5102, r5108;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5114, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5115, {low, high};
}
{
add.f16x2 r5116, r5117, r5118;
}
{
add.f16x2 r5119, r5120, r5116;
}
{
add.f16x2 r5122, r5123, r5124;
}
{
add.f16x2 r5125, r5126, r5122;
}
{
add.f16x2 r5128, r5117, r5118;
}
{
mul.f16x2 r5131, r5128, r5114;
}
{
add.f16x2 r5134, r5120, r5131;
}
{
sub.f16x2 r5137, r5123, r5124;
}
{
mul.f16x2 r5140, r5137, r5115;
}
{
add.f16x2 r5143, r5134, r5140;
}
{
add.f16x2 r5146, r5117, r5118;
}
{
mul.f16x2 r5149, r5146, r5114;
}
{
add.f16x2 r5152, r5120, r5149;
}
{
sub.f16x2 r5155, r5123, r5124;
}
{
mul.f16x2 r5158, r5155, r5115;
}
{
sub.f16x2 r5161, r5152, r5158;
}
{
add.f16x2 r5164, r5123, r5124;
}
{
mul.f16x2 r5167, r5164, r5114;
}
{
add.f16x2 r5170, r5126, r5167;
}
{
sub.f16x2 r5173, r5117, r5118;
}
{
mul.f16x2 r5176, r5173, r5115;
}
{
sub.f16x2 r5179, r5170, r5176;
}
{
add.f16x2 r5182, r5123, r5124;
}
{
mul.f16x2 r5185, r5182, r5114;
}
{
add.f16x2 r5188, r5126, r5185;
}
{
sub.f16x2 r5191, r5117, r5118;
}
{
mul.f16x2 r5194, r5191, r5115;
}
{
add.f16x2 r5197, r5188, r5194;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5200, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5201, {low, high};
}
{
add.f16x2 r5202, r5203, r5204;
}
{
add.f16x2 r5205, r5206, r5202;
}
{
add.f16x2 r5208, r5209, r5210;
}
{
add.f16x2 r5211, r5212, r5208;
}
{
add.f16x2 r5214, r5203, r5204;
}
{
mul.f16x2 r5217, r5214, r5200;
}
{
add.f16x2 r5220, r5206, r5217;
}
{
sub.f16x2 r5223, r5209, r5210;
}
{
mul.f16x2 r5226, r5223, r5201;
}
{
add.f16x2 r5229, r5220, r5226;
}
{
add.f16x2 r5232, r5203, r5204;
}
{
mul.f16x2 r5235, r5232, r5200;
}
{
add.f16x2 r5238, r5206, r5235;
}
{
sub.f16x2 r5241, r5209, r5210;
}
{
mul.f16x2 r5244, r5241, r5201;
}
{
sub.f16x2 r5247, r5238, r5244;
}
{
add.f16x2 r5250, r5209, r5210;
}
{
mul.f16x2 r5253, r5250, r5200;
}
{
add.f16x2 r5256, r5212, r5253;
}
{
sub.f16x2 r5259, r5203, r5204;
}
{
mul.f16x2 r5262, r5259, r5201;
}
{
sub.f16x2 r5265, r5256, r5262;
}
{
add.f16x2 r5268, r5209, r5210;
}
{
mul.f16x2 r5271, r5268, r5200;
}
{
add.f16x2 r5274, r5212, r5271;
}
{
sub.f16x2 r5277, r5203, r5204;
}
{
mul.f16x2 r5280, r5277, r5201;
}
{
add.f16x2 r5283, r5274, r5280;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r5286, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r5287, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r5288, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r5289, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r5292, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
mov.b32 r5293, {low, high};
}
{
mul.f16x2 r5302, r5143, r5286;
}
{
mul.f16x2 r5305, r5179, r5287;
}
{
sub.f16x2 r5308, r5302, r5305;
}
{
mul.f16x2 r5311, r5143, r5287;
}
{
fma.rn.f16x2 r5314, r5179, r5286, r5311;
}
{
mul.f16x2 r5318, r5229, r5288;
}
{
mul.f16x2 r5321, r5265, r5289;
}
{
sub.f16x2 r5324, r5318, r5321;
}
{
mul.f16x2 r5327, r5229, r5289;
}
{
fma.rn.f16x2 r5330, r5265, r5288, r5327;
}
{
mul.f16x2 r5334, r5161, r5288;
}
{
mul.f16x2 r5337, r5197, r5289;
}
{
sub.f16x2 r5340, r5334, r5337;
}
{
mul.f16x2 r5343, r5161, r5289;
}
{
fma.rn.f16x2 r5346, r5197, r5288, r5343;
}
{
mul.f16x2 r5350, r5247, r5292;
}
{
mul.f16x2 r5353, r5283, r5293;
}
{
sub.f16x2 r5356, r5350, r5353;
}
{
mul.f16x2 r5359, r5247, r5293;
}
{
fma.rn.f16x2 r5362, r5283, r5292, r5359;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5366, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5367, {low, high};
}
{
add.f16x2 r5368, r5119, r5205;
}
{
add.f16x2 r5371, r5033, r5368;
}
{
add.f16x2 r5374, r5125, r5211;
}
{
add.f16x2 r5377, r5039, r5374;
}
{
add.f16x2 r5380, r5119, r5205;
}
{
mul.f16x2 r5383, r5380, r5366;
}
{
add.f16x2 r5386, r5033, r5383;
}
{
sub.f16x2 r5389, r5125, r5211;
}
{
mul.f16x2 r5392, r5389, r5367;
}
{
add.f16x2 r5395, r5386, r5392;
}
{
add.f16x2 r5398, r5119, r5205;
}
{
mul.f16x2 r5401, r5398, r5366;
}
{
add.f16x2 r5404, r5033, r5401;
}
{
sub.f16x2 r5407, r5125, r5211;
}
{
mul.f16x2 r5410, r5407, r5367;
}
{
sub.f16x2 r5413, r5404, r5410;
}
{
add.f16x2 r5416, r5125, r5211;
}
{
mul.f16x2 r5419, r5416, r5366;
}
{
add.f16x2 r5422, r5039, r5419;
}
{
sub.f16x2 r5425, r5119, r5205;
}
{
mul.f16x2 r5428, r5425, r5367;
}
{
sub.f16x2 r5431, r5422, r5428;
}
{
add.f16x2 r5434, r5125, r5211;
}
{
mul.f16x2 r5437, r5434, r5366;
}
{
add.f16x2 r5440, r5039, r5437;
}
{
sub.f16x2 r5443, r5119, r5205;
}
{
mul.f16x2 r5446, r5443, r5367;
}
{
add.f16x2 r5449, r5440, r5446;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5452, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5453, {low, high};
}
{
add.f16x2 r5454, r5308, r5324;
}
{
add.f16x2 r5457, r5057, r5454;
}
{
add.f16x2 r5460, r5314, r5330;
}
{
add.f16x2 r5463, r5093, r5460;
}
{
add.f16x2 r5466, r5308, r5324;
}
{
mul.f16x2 r5469, r5466, r5452;
}
{
add.f16x2 r5472, r5057, r5469;
}
{
sub.f16x2 r5475, r5314, r5330;
}
{
mul.f16x2 r5478, r5475, r5453;
}
{
add.f16x2 r5481, r5472, r5478;
}
{
add.f16x2 r5484, r5308, r5324;
}
{
mul.f16x2 r5487, r5484, r5452;
}
{
add.f16x2 r5490, r5057, r5487;
}
{
sub.f16x2 r5493, r5314, r5330;
}
{
mul.f16x2 r5496, r5493, r5453;
}
{
sub.f16x2 r5499, r5490, r5496;
}
{
add.f16x2 r5502, r5314, r5330;
}
{
mul.f16x2 r5505, r5502, r5452;
}
{
add.f16x2 r5508, r5093, r5505;
}
{
sub.f16x2 r5511, r5308, r5324;
}
{
mul.f16x2 r5514, r5511, r5453;
}
{
sub.f16x2 r5517, r5508, r5514;
}
{
add.f16x2 r5520, r5314, r5330;
}
{
mul.f16x2 r5523, r5520, r5452;
}
{
add.f16x2 r5526, r5093, r5523;
}
{
sub.f16x2 r5529, r5308, r5324;
}
{
mul.f16x2 r5532, r5529, r5453;
}
{
add.f16x2 r5535, r5526, r5532;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5538, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5539, {low, high};
}
{
add.f16x2 r5540, r5340, r5356;
}
{
add.f16x2 r5543, r5075, r5540;
}
{
add.f16x2 r5546, r5346, r5362;
}
{
add.f16x2 r5549, r5111, r5546;
}
{
add.f16x2 r5552, r5340, r5356;
}
{
mul.f16x2 r5555, r5552, r5538;
}
{
add.f16x2 r5558, r5075, r5555;
}
{
sub.f16x2 r5561, r5346, r5362;
}
{
mul.f16x2 r5564, r5561, r5539;
}
{
add.f16x2 r5567, r5558, r5564;
}
{
add.f16x2 r5570, r5340, r5356;
}
{
mul.f16x2 r5573, r5570, r5538;
}
{
add.f16x2 r5576, r5075, r5573;
}
{
sub.f16x2 r5579, r5346, r5362;
}
{
mul.f16x2 r5582, r5579, r5539;
}
{
sub.f16x2 r5585, r5576, r5582;
}
{
add.f16x2 r5588, r5346, r5362;
}
{
mul.f16x2 r5591, r5588, r5538;
}
{
add.f16x2 r5594, r5111, r5591;
}
{
sub.f16x2 r5597, r5340, r5356;
}
{
mul.f16x2 r5600, r5597, r5539;
}
{
sub.f16x2 r5603, r5594, r5600;
}
{
add.f16x2 r5606, r5346, r5362;
}
{
mul.f16x2 r5609, r5606, r5538;
}
{
add.f16x2 r5612, r5111, r5609;
}
{
sub.f16x2 r5615, r5340, r5356;
}
{
mul.f16x2 r5618, r5615, r5539;
}
{
add.f16x2 r5621, r5612, r5618;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f534;
cvt.rn.f16.f32 high, f534;
mov.b32 r5624, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f536;
cvt.rn.f16.f32 high, f536;
mov.b32 r5625, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f538;
cvt.rn.f16.f32 high, f538;
mov.b32 r5626, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f540;
cvt.rn.f16.f32 high, f540;
mov.b32 r5627, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r5628, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r5629, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f546;
cvt.rn.f16.f32 high, f546;
mov.b32 r5630, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f548;
cvt.rn.f16.f32 high, f548;
mov.b32 r5631, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f550;
cvt.rn.f16.f32 high, f550;
mov.b32 r5632, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f552;
cvt.rn.f16.f32 high, f552;
mov.b32 r5633, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r5634, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r5635, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f558;
cvt.rn.f16.f32 high, f558;
mov.b32 r5636, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f560;
cvt.rn.f16.f32 high, f560;
mov.b32 r5637, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f562;
cvt.rn.f16.f32 high, f562;
mov.b32 r5638, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f564;
cvt.rn.f16.f32 high, f564;
mov.b32 r5639, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f570;
cvt.rn.f16.f32 high, f570;
mov.b32 r5642, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f572;
cvt.rn.f16.f32 high, f572;
mov.b32 r5643, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r5646, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
mov.b32 r5647, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f586;
cvt.rn.f16.f32 high, f586;
mov.b32 r5650, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f588;
cvt.rn.f16.f32 high, f588;
mov.b32 r5651, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5654, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5655, {low, high};
}
{
mul.f16x2 r5676, r4861, r5624;
}
{
mul.f16x2 r5679, r4867, r5625;
}
{
sub.f16x2 r5682, r5676, r5679;
}
{
mul.f16x2 r5685, r4861, r5625;
}
{
fma.rn.f16x2 r5688, r4867, r5624, r5685;
}
{
mul.f16x2 r5692, r5457, r5626;
}
{
mul.f16x2 r5695, r5463, r5627;
}
{
sub.f16x2 r5698, r5692, r5695;
}
{
mul.f16x2 r5701, r5457, r5627;
}
{
fma.rn.f16x2 r5704, r5463, r5626, r5701;
}
{
mul.f16x2 r5708, r4947, r5626;
}
{
mul.f16x2 r5711, r4953, r5627;
}
{
sub.f16x2 r5714, r5708, r5711;
}
{
mul.f16x2 r5717, r4947, r5627;
}
{
fma.rn.f16x2 r5720, r4953, r5626, r5717;
}
{
mul.f16x2 r5724, r5543, r5630;
}
{
mul.f16x2 r5727, r5549, r5631;
}
{
sub.f16x2 r5730, r5724, r5727;
}
{
mul.f16x2 r5733, r5543, r5631;
}
{
fma.rn.f16x2 r5736, r5549, r5630, r5733;
}
{
mul.f16x2 r5740, r4799, r5628;
}
{
mul.f16x2 r5743, r4835, r5629;
}
{
sub.f16x2 r5746, r5740, r5743;
}
{
mul.f16x2 r5749, r4799, r5629;
}
{
fma.rn.f16x2 r5752, r4835, r5628, r5749;
}
{
mul.f16x2 r5756, r5395, r5634;
}
{
mul.f16x2 r5759, r5431, r5635;
}
{
sub.f16x2 r5762, r5756, r5759;
}
{
mul.f16x2 r5765, r5395, r5635;
}
{
fma.rn.f16x2 r5768, r5431, r5634, r5765;
}
{
mul.f16x2 r5772, r4885, r5630;
}
{
mul.f16x2 r5775, r4921, r5631;
}
{
sub.f16x2 r5778, r5772, r5775;
}
{
mul.f16x2 r5781, r4885, r5631;
}
{
fma.rn.f16x2 r5784, r4921, r5630, r5781;
}
{
mul.f16x2 r5788, r5481, r5638;
}
{
mul.f16x2 r5791, r5517, r5639;
}
{
sub.f16x2 r5794, r5788, r5791;
}
{
mul.f16x2 r5797, r5481, r5639;
}
{
fma.rn.f16x2 r5800, r5517, r5638, r5797;
}
{
mul.f16x2 r5804, r4971, r5632;
}
{
mul.f16x2 r5807, r5007, r5633;
}
{
sub.f16x2 r5810, r5804, r5807;
}
{
mul.f16x2 r5813, r4971, r5633;
}
{
fma.rn.f16x2 r5816, r5007, r5632, r5813;
}
{
mul.f16x2 r5820, r5567, r5642;
}
{
mul.f16x2 r5823, r5603, r5643;
}
{
sub.f16x2 r5826, r5820, r5823;
}
{
mul.f16x2 r5829, r5567, r5643;
}
{
fma.rn.f16x2 r5832, r5603, r5642, r5829;
}
{
mul.f16x2 r5836, r4817, r5634;
}
{
mul.f16x2 r5839, r4853, r5635;
}
{
sub.f16x2 r5842, r5836, r5839;
}
{
mul.f16x2 r5845, r4817, r5635;
}
{
fma.rn.f16x2 r5848, r4853, r5634, r5845;
}
{
mul.f16x2 r5852, r5413, r5646;
}
{
mul.f16x2 r5855, r5449, r5647;
}
{
sub.f16x2 r5858, r5852, r5855;
}
{
mul.f16x2 r5861, r5413, r5647;
}
{
fma.rn.f16x2 r5864, r5449, r5646, r5861;
}
{
mul.f16x2 r5868, r4903, r5636;
}
{
mul.f16x2 r5871, r4939, r5637;
}
{
sub.f16x2 r5874, r5868, r5871;
}
{
mul.f16x2 r5877, r4903, r5637;
}
{
fma.rn.f16x2 r5880, r4939, r5636, r5877;
}
{
mul.f16x2 r5884, r5499, r5650;
}
{
mul.f16x2 r5887, r5535, r5651;
}
{
sub.f16x2 r5890, r5884, r5887;
}
{
mul.f16x2 r5893, r5499, r5651;
}
{
fma.rn.f16x2 r5896, r5535, r5650, r5893;
}
{
mul.f16x2 r5900, r4989, r5638;
}
{
mul.f16x2 r5903, r5025, r5639;
}
{
sub.f16x2 r5906, r5900, r5903;
}
{
mul.f16x2 r5909, r4989, r5639;
}
{
fma.rn.f16x2 r5912, r5025, r5638, r5909;
}
{
mul.f16x2 r5916, r5585, r5654;
}
{
mul.f16x2 r5919, r5621, r5655;
}
{
sub.f16x2 r5922, r5916, r5919;
}
{
mul.f16x2 r5925, r5585, r5655;
}
{
fma.rn.f16x2 r5928, r5621, r5654, r5925;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5932, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5933, {low, high};
}
{
add.f16x2 r5934, r4775, r5371;
}
{
add.f16x2 r5937, r4179, r5934;
}
{
add.f16x2 r5940, r4781, r5377;
}
{
add.f16x2 r5943, r4185, r5940;
}
{
add.f16x2 r5946, r4775, r5371;
}
{
mul.f16x2 r5949, r5946, r5932;
}
{
add.f16x2 r5952, r4179, r5949;
}
{
sub.f16x2 r5955, r4781, r5377;
}
{
mul.f16x2 r5958, r5955, r5933;
}
{
add.f16x2 r5961, r5952, r5958;
}
{
add.f16x2 r5964, r4775, r5371;
}
{
mul.f16x2 r5967, r5964, r5932;
}
{
add.f16x2 r5970, r4179, r5967;
}
{
sub.f16x2 r5973, r4781, r5377;
}
{
mul.f16x2 r5976, r5973, r5933;
}
{
sub.f16x2 r5979, r5970, r5976;
}
{
add.f16x2 r5982, r4781, r5377;
}
{
mul.f16x2 r5985, r5982, r5932;
}
{
add.f16x2 r5988, r4185, r5985;
}
{
sub.f16x2 r5991, r4775, r5371;
}
{
mul.f16x2 r5994, r5991, r5933;
}
{
sub.f16x2 r5997, r5988, r5994;
}
{
add.f16x2 r6000, r4781, r5377;
}
{
mul.f16x2 r6003, r6000, r5932;
}
{
add.f16x2 r6006, r4185, r6003;
}
{
sub.f16x2 r6009, r4775, r5371;
}
{
mul.f16x2 r6012, r6009, r5933;
}
{
add.f16x2 r6015, r6006, r6012;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6019, {low, high};
}
{
add.f16x2 r6020, r5682, r5698;
}
{
add.f16x2 r6023, r4265, r6020;
}
{
add.f16x2 r6026, r5688, r5704;
}
{
add.f16x2 r6029, r4271, r6026;
}
{
add.f16x2 r6032, r5682, r5698;
}
{
mul.f16x2 r6035, r6032, r6018;
}
{
add.f16x2 r6038, r4265, r6035;
}
{
sub.f16x2 r6041, r5688, r5704;
}
{
mul.f16x2 r6044, r6041, r6019;
}
{
add.f16x2 r6047, r6038, r6044;
}
{
add.f16x2 r6050, r5682, r5698;
}
{
mul.f16x2 r6053, r6050, r6018;
}
{
add.f16x2 r6056, r4265, r6053;
}
{
sub.f16x2 r6059, r5688, r5704;
}
{
mul.f16x2 r6062, r6059, r6019;
}
{
sub.f16x2 r6065, r6056, r6062;
}
{
add.f16x2 r6068, r5688, r5704;
}
{
mul.f16x2 r6071, r6068, r6018;
}
{
add.f16x2 r6074, r4271, r6071;
}
{
sub.f16x2 r6077, r5682, r5698;
}
{
mul.f16x2 r6080, r6077, r6019;
}
{
sub.f16x2 r6083, r6074, r6080;
}
{
add.f16x2 r6086, r5688, r5704;
}
{
mul.f16x2 r6089, r6086, r6018;
}
{
add.f16x2 r6092, r4271, r6089;
}
{
sub.f16x2 r6095, r5682, r5698;
}
{
mul.f16x2 r6098, r6095, r6019;
}
{
add.f16x2 r6101, r6092, r6098;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6104, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6105, {low, high};
}
{
add.f16x2 r6106, r5714, r5730;
}
{
add.f16x2 r6109, r4351, r6106;
}
{
add.f16x2 r6112, r5720, r5736;
}
{
add.f16x2 r6115, r4357, r6112;
}
{
add.f16x2 r6118, r5714, r5730;
}
{
mul.f16x2 r6121, r6118, r6104;
}
{
add.f16x2 r6124, r4351, r6121;
}
{
sub.f16x2 r6127, r5720, r5736;
}
{
mul.f16x2 r6130, r6127, r6105;
}
{
add.f16x2 r6133, r6124, r6130;
}
{
add.f16x2 r6136, r5714, r5730;
}
{
mul.f16x2 r6139, r6136, r6104;
}
{
add.f16x2 r6142, r4351, r6139;
}
{
sub.f16x2 r6145, r5720, r5736;
}
{
mul.f16x2 r6148, r6145, r6105;
}
{
sub.f16x2 r6151, r6142, r6148;
}
{
add.f16x2 r6154, r5720, r5736;
}
{
mul.f16x2 r6157, r6154, r6104;
}
{
add.f16x2 r6160, r4357, r6157;
}
{
sub.f16x2 r6163, r5714, r5730;
}
{
mul.f16x2 r6166, r6163, r6105;
}
{
sub.f16x2 r6169, r6160, r6166;
}
{
add.f16x2 r6172, r5720, r5736;
}
{
mul.f16x2 r6175, r6172, r6104;
}
{
add.f16x2 r6178, r4357, r6175;
}
{
sub.f16x2 r6181, r5714, r5730;
}
{
mul.f16x2 r6184, r6181, r6105;
}
{
add.f16x2 r6187, r6178, r6184;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6190, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6191, {low, high};
}
{
add.f16x2 r6192, r5746, r5762;
}
{
add.f16x2 r6195, r4203, r6192;
}
{
add.f16x2 r6198, r5752, r5768;
}
{
add.f16x2 r6201, r4239, r6198;
}
{
add.f16x2 r6204, r5746, r5762;
}
{
mul.f16x2 r6207, r6204, r6190;
}
{
add.f16x2 r6210, r4203, r6207;
}
{
sub.f16x2 r6213, r5752, r5768;
}
{
mul.f16x2 r6216, r6213, r6191;
}
{
add.f16x2 r6219, r6210, r6216;
}
{
add.f16x2 r6222, r5746, r5762;
}
{
mul.f16x2 r6225, r6222, r6190;
}
{
add.f16x2 r6228, r4203, r6225;
}
{
sub.f16x2 r6231, r5752, r5768;
}
{
mul.f16x2 r6234, r6231, r6191;
}
{
sub.f16x2 r6237, r6228, r6234;
}
{
add.f16x2 r6240, r5752, r5768;
}
{
mul.f16x2 r6243, r6240, r6190;
}
{
add.f16x2 r6246, r4239, r6243;
}
{
sub.f16x2 r6249, r5746, r5762;
}
{
mul.f16x2 r6252, r6249, r6191;
}
{
sub.f16x2 r6255, r6246, r6252;
}
{
add.f16x2 r6258, r5752, r5768;
}
{
mul.f16x2 r6261, r6258, r6190;
}
{
add.f16x2 r6264, r4239, r6261;
}
{
sub.f16x2 r6267, r5746, r5762;
}
{
mul.f16x2 r6270, r6267, r6191;
}
{
add.f16x2 r6273, r6264, r6270;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6276, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6277, {low, high};
}
{
add.f16x2 r6278, r5778, r5794;
}
{
add.f16x2 r6281, r4289, r6278;
}
{
add.f16x2 r6284, r5784, r5800;
}
{
add.f16x2 r6287, r4325, r6284;
}
{
add.f16x2 r6290, r5778, r5794;
}
{
mul.f16x2 r6293, r6290, r6276;
}
{
add.f16x2 r6296, r4289, r6293;
}
{
sub.f16x2 r6299, r5784, r5800;
}
{
mul.f16x2 r6302, r6299, r6277;
}
{
add.f16x2 r6305, r6296, r6302;
}
{
add.f16x2 r6308, r5778, r5794;
}
{
mul.f16x2 r6311, r6308, r6276;
}
{
add.f16x2 r6314, r4289, r6311;
}
{
sub.f16x2 r6317, r5784, r5800;
}
{
mul.f16x2 r6320, r6317, r6277;
}
{
sub.f16x2 r6323, r6314, r6320;
}
{
add.f16x2 r6326, r5784, r5800;
}
{
mul.f16x2 r6329, r6326, r6276;
}
{
add.f16x2 r6332, r4325, r6329;
}
{
sub.f16x2 r6335, r5778, r5794;
}
{
mul.f16x2 r6338, r6335, r6277;
}
{
sub.f16x2 r6341, r6332, r6338;
}
{
add.f16x2 r6344, r5784, r5800;
}
{
mul.f16x2 r6347, r6344, r6276;
}
{
add.f16x2 r6350, r4325, r6347;
}
{
sub.f16x2 r6353, r5778, r5794;
}
{
mul.f16x2 r6356, r6353, r6277;
}
{
add.f16x2 r6359, r6350, r6356;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6362, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6363, {low, high};
}
{
add.f16x2 r6364, r5810, r5826;
}
{
add.f16x2 r6367, r4375, r6364;
}
{
add.f16x2 r6370, r5816, r5832;
}
{
add.f16x2 r6373, r4411, r6370;
}
{
add.f16x2 r6376, r5810, r5826;
}
{
mul.f16x2 r6379, r6376, r6362;
}
{
add.f16x2 r6382, r4375, r6379;
}
{
sub.f16x2 r6385, r5816, r5832;
}
{
mul.f16x2 r6388, r6385, r6363;
}
{
add.f16x2 r6391, r6382, r6388;
}
{
add.f16x2 r6394, r5810, r5826;
}
{
mul.f16x2 r6397, r6394, r6362;
}
{
add.f16x2 r6400, r4375, r6397;
}
{
sub.f16x2 r6403, r5816, r5832;
}
{
mul.f16x2 r6406, r6403, r6363;
}
{
sub.f16x2 r6409, r6400, r6406;
}
{
add.f16x2 r6412, r5816, r5832;
}
{
mul.f16x2 r6415, r6412, r6362;
}
{
add.f16x2 r6418, r4411, r6415;
}
{
sub.f16x2 r6421, r5810, r5826;
}
{
mul.f16x2 r6424, r6421, r6363;
}
{
sub.f16x2 r6427, r6418, r6424;
}
{
add.f16x2 r6430, r5816, r5832;
}
{
mul.f16x2 r6433, r6430, r6362;
}
{
add.f16x2 r6436, r4411, r6433;
}
{
sub.f16x2 r6439, r5810, r5826;
}
{
mul.f16x2 r6442, r6439, r6363;
}
{
add.f16x2 r6445, r6436, r6442;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6448, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6449, {low, high};
}
{
add.f16x2 r6450, r5842, r5858;
}
{
add.f16x2 r6453, r4221, r6450;
}
{
add.f16x2 r6456, r5848, r5864;
}
{
add.f16x2 r6459, r4257, r6456;
}
{
add.f16x2 r6462, r5842, r5858;
}
{
mul.f16x2 r6465, r6462, r6448;
}
{
add.f16x2 r6468, r4221, r6465;
}
{
sub.f16x2 r6471, r5848, r5864;
}
{
mul.f16x2 r6474, r6471, r6449;
}
{
add.f16x2 r6477, r6468, r6474;
}
{
add.f16x2 r6480, r5842, r5858;
}
{
mul.f16x2 r6483, r6480, r6448;
}
{
add.f16x2 r6486, r4221, r6483;
}
{
sub.f16x2 r6489, r5848, r5864;
}
{
mul.f16x2 r6492, r6489, r6449;
}
{
sub.f16x2 r6495, r6486, r6492;
}
{
add.f16x2 r6498, r5848, r5864;
}
{
mul.f16x2 r6501, r6498, r6448;
}
{
add.f16x2 r6504, r4257, r6501;
}
{
sub.f16x2 r6507, r5842, r5858;
}
{
mul.f16x2 r6510, r6507, r6449;
}
{
sub.f16x2 r6513, r6504, r6510;
}
{
add.f16x2 r6516, r5848, r5864;
}
{
mul.f16x2 r6519, r6516, r6448;
}
{
add.f16x2 r6522, r4257, r6519;
}
{
sub.f16x2 r6525, r5842, r5858;
}
{
mul.f16x2 r6528, r6525, r6449;
}
{
add.f16x2 r6531, r6522, r6528;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6534, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6535, {low, high};
}
{
add.f16x2 r6536, r5874, r5890;
}
{
add.f16x2 r6539, r4307, r6536;
}
{
add.f16x2 r6542, r5880, r5896;
}
{
add.f16x2 r6545, r4343, r6542;
}
{
add.f16x2 r6548, r5874, r5890;
}
{
mul.f16x2 r6551, r6548, r6534;
}
{
add.f16x2 r6554, r4307, r6551;
}
{
sub.f16x2 r6557, r5880, r5896;
}
{
mul.f16x2 r6560, r6557, r6535;
}
{
add.f16x2 r6563, r6554, r6560;
}
{
add.f16x2 r6566, r5874, r5890;
}
{
mul.f16x2 r6569, r6566, r6534;
}
{
add.f16x2 r6572, r4307, r6569;
}
{
sub.f16x2 r6575, r5880, r5896;
}
{
mul.f16x2 r6578, r6575, r6535;
}
{
sub.f16x2 r6581, r6572, r6578;
}
{
add.f16x2 r6584, r5880, r5896;
}
{
mul.f16x2 r6587, r6584, r6534;
}
{
add.f16x2 r6590, r4343, r6587;
}
{
sub.f16x2 r6593, r5874, r5890;
}
{
mul.f16x2 r6596, r6593, r6535;
}
{
sub.f16x2 r6599, r6590, r6596;
}
{
add.f16x2 r6602, r5880, r5896;
}
{
mul.f16x2 r6605, r6602, r6534;
}
{
add.f16x2 r6608, r4343, r6605;
}
{
sub.f16x2 r6611, r5874, r5890;
}
{
mul.f16x2 r6614, r6611, r6535;
}
{
add.f16x2 r6617, r6608, r6614;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6620, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6621, {low, high};
}
{
add.f16x2 r6622, r5906, r5922;
}
{
add.f16x2 r6625, r4393, r6622;
}
{
add.f16x2 r6628, r5912, r5928;
}
{
add.f16x2 r6631, r4429, r6628;
}
{
add.f16x2 r6634, r5906, r5922;
}
{
mul.f16x2 r6637, r6634, r6620;
}
{
add.f16x2 r6640, r4393, r6637;
}
{
sub.f16x2 r6643, r5912, r5928;
}
{
mul.f16x2 r6646, r6643, r6621;
}
{
add.f16x2 r6649, r6640, r6646;
}
{
add.f16x2 r6652, r5906, r5922;
}
{
mul.f16x2 r6655, r6652, r6620;
}
{
add.f16x2 r6658, r4393, r6655;
}
{
sub.f16x2 r6661, r5912, r5928;
}
{
mul.f16x2 r6664, r6661, r6621;
}
{
sub.f16x2 r6667, r6658, r6664;
}
{
add.f16x2 r6670, r5912, r5928;
}
{
mul.f16x2 r6673, r6670, r6620;
}
{
add.f16x2 r6676, r4429, r6673;
}
{
sub.f16x2 r6679, r5906, r5922;
}
{
mul.f16x2 r6682, r6679, r6621;
}
{
sub.f16x2 r6685, r6676, r6682;
}
{
add.f16x2 r6688, r5912, r5928;
}
{
mul.f16x2 r6691, r6688, r6620;
}
{
add.f16x2 r6694, r4429, r6691;
}
{
sub.f16x2 r6697, r5906, r5922;
}
{
mul.f16x2 r6700, r6697, r6621;
}
{
add.f16x2 r6703, r6694, r6700;
}
mul.wide.u32 rd4, r8451, 795364315;
shr.u64 rd5, rd4, 32;
cvt.u32.u64 r8455, rd5;
sub.s32 r8456, r8451, r8455;
shr.u32 r8457, r8456, 1;
add.s32 r8458, r8457, r8455;
shr.u32 r8459, r8458, 4;
mul.lo.s32 r8460, r8459, 27;
sub.s32 r8461, r8451, r8460;
shl.b32 r8462, r8461, 2;
add.s32 r8463, r8452, r8462;
cvt.rn.f32.u32 f768, r8459;
mul.f32 f769, f768, 0f3D9EDD1F;
cos.approx.f32 f673, f769;
sin.approx.f32 f770, f769;
neg.f32 f674, f770;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f673;
cvt.rn.f16.f32 high, f674;
mov.b32 r6706, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6709, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6711, {high, high};
}
{
mul.f16x2 r6713, r6029, r6711;
}
{
fma.rn.f16x2 r6716, r6023, r6709, r6713;
}
{
mul.f16x2 r6720, r6023, r6711;
}
{
neg.f16x2 r6723, r6720;
}
{
fma.rn.f16x2 r6725, r6029, r6709, r6723;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6729, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6731, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6733, {low, high};
}
{
mul.f16x2 r6734, r6731, r6733;
}
{
mul.f16x2 r6737, r6706, r6729;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6740, {high, low};
}
{
fma.rn.f16x2 r6742, r6734, r6740, r6737;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6742;
mov.b32 r6746, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6742;
mov.b32 r6748, {high, high};
}
{
mul.f16x2 r6750, r6115, r6748;
}
{
fma.rn.f16x2 r6753, r6109, r6746, r6750;
}
{
mul.f16x2 r6757, r6109, r6748;
}
{
neg.f16x2 r6760, r6757;
}
{
fma.rn.f16x2 r6762, r6115, r6746, r6760;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6766, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6768, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6770, {low, high};
}
{
mul.f16x2 r6771, r6768, r6770;
}
{
mul.f16x2 r6774, r6742, r6766;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6742;
mov.b32 r6777, {high, low};
}
{
fma.rn.f16x2 r6779, r6771, r6777, r6774;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6779;
mov.b32 r6783, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6779;
mov.b32 r6785, {high, high};
}
{
mul.f16x2 r6787, r6201, r6785;
}
{
fma.rn.f16x2 r6790, r6195, r6783, r6787;
}
{
mul.f16x2 r6794, r6195, r6785;
}
{
neg.f16x2 r6797, r6794;
}
{
fma.rn.f16x2 r6799, r6201, r6783, r6797;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6803, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6805, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6807, {low, high};
}
{
mul.f16x2 r6808, r6805, r6807;
}
{
mul.f16x2 r6811, r6779, r6803;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6779;
mov.b32 r6814, {high, low};
}
{
fma.rn.f16x2 r6816, r6808, r6814, r6811;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6816;
mov.b32 r6820, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6816;
mov.b32 r6822, {high, high};
}
{
mul.f16x2 r6824, r6287, r6822;
}
{
fma.rn.f16x2 r6827, r6281, r6820, r6824;
}
{
mul.f16x2 r6831, r6281, r6822;
}
{
neg.f16x2 r6834, r6831;
}
{
fma.rn.f16x2 r6836, r6287, r6820, r6834;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6840, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6842, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6844, {low, high};
}
{
mul.f16x2 r6845, r6842, r6844;
}
{
mul.f16x2 r6848, r6816, r6840;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6816;
mov.b32 r6851, {high, low};
}
{
fma.rn.f16x2 r6853, r6845, r6851, r6848;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6853;
mov.b32 r6857, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6853;
mov.b32 r6859, {high, high};
}
{
mul.f16x2 r6861, r6373, r6859;
}
{
fma.rn.f16x2 r6864, r6367, r6857, r6861;
}
{
mul.f16x2 r6868, r6367, r6859;
}
{
neg.f16x2 r6871, r6868;
}
{
fma.rn.f16x2 r6873, r6373, r6857, r6871;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6877, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6879, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6881, {low, high};
}
{
mul.f16x2 r6882, r6879, r6881;
}
{
mul.f16x2 r6885, r6853, r6877;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6853;
mov.b32 r6888, {high, low};
}
{
fma.rn.f16x2 r6890, r6882, r6888, r6885;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6890;
mov.b32 r6894, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6890;
mov.b32 r6896, {high, high};
}
{
mul.f16x2 r6898, r6459, r6896;
}
{
fma.rn.f16x2 r6901, r6453, r6894, r6898;
}
{
mul.f16x2 r6905, r6453, r6896;
}
{
neg.f16x2 r6908, r6905;
}
{
fma.rn.f16x2 r6910, r6459, r6894, r6908;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6914, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6916, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6918, {low, high};
}
{
mul.f16x2 r6919, r6916, r6918;
}
{
mul.f16x2 r6922, r6890, r6914;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6890;
mov.b32 r6925, {high, low};
}
{
fma.rn.f16x2 r6927, r6919, r6925, r6922;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6927;
mov.b32 r6931, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6927;
mov.b32 r6933, {high, high};
}
{
mul.f16x2 r6935, r6545, r6933;
}
{
fma.rn.f16x2 r6938, r6539, r6931, r6935;
}
{
mul.f16x2 r6942, r6539, r6933;
}
{
neg.f16x2 r6945, r6942;
}
{
fma.rn.f16x2 r6947, r6545, r6931, r6945;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6951, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6953, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6955, {low, high};
}
{
mul.f16x2 r6956, r6953, r6955;
}
{
mul.f16x2 r6959, r6927, r6951;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6927;
mov.b32 r6962, {high, low};
}
{
fma.rn.f16x2 r6964, r6956, r6962, r6959;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6964;
mov.b32 r6968, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6964;
mov.b32 r6970, {high, high};
}
{
mul.f16x2 r6972, r6631, r6970;
}
{
fma.rn.f16x2 r6975, r6625, r6968, r6972;
}
{
mul.f16x2 r6979, r6625, r6970;
}
{
neg.f16x2 r6982, r6979;
}
{
fma.rn.f16x2 r6984, r6631, r6968, r6982;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6988, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r6990, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6992, {low, high};
}
{
mul.f16x2 r6993, r6990, r6992;
}
{
mul.f16x2 r6996, r6964, r6988;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6964;
mov.b32 r6999, {high, low};
}
{
fma.rn.f16x2 r7001, r6993, r6999, r6996;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7001;
mov.b32 r7005, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7001;
mov.b32 r7007, {high, high};
}
{
mul.f16x2 r7009, r5997, r7007;
}
{
fma.rn.f16x2 r7012, r5961, r7005, r7009;
}
{
mul.f16x2 r7016, r5961, r7007;
}
{
neg.f16x2 r7019, r7016;
}
{
fma.rn.f16x2 r7021, r5997, r7005, r7019;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7025, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7027, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7029, {low, high};
}
{
mul.f16x2 r7030, r7027, r7029;
}
{
mul.f16x2 r7033, r7001, r7025;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7001;
mov.b32 r7036, {high, low};
}
{
fma.rn.f16x2 r7038, r7030, r7036, r7033;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7038;
mov.b32 r7042, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7038;
mov.b32 r7044, {high, high};
}
{
mul.f16x2 r7046, r6083, r7044;
}
{
fma.rn.f16x2 r7049, r6047, r7042, r7046;
}
{
mul.f16x2 r7053, r6047, r7044;
}
{
neg.f16x2 r7056, r7053;
}
{
fma.rn.f16x2 r7058, r6083, r7042, r7056;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7062, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7064, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7066, {low, high};
}
{
mul.f16x2 r7067, r7064, r7066;
}
{
mul.f16x2 r7070, r7038, r7062;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7038;
mov.b32 r7073, {high, low};
}
{
fma.rn.f16x2 r7075, r7067, r7073, r7070;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7075;
mov.b32 r7079, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7075;
mov.b32 r7081, {high, high};
}
{
mul.f16x2 r7083, r6169, r7081;
}
{
fma.rn.f16x2 r7086, r6133, r7079, r7083;
}
{
mul.f16x2 r7090, r6133, r7081;
}
{
neg.f16x2 r7093, r7090;
}
{
fma.rn.f16x2 r7095, r6169, r7079, r7093;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7099, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7101, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7103, {low, high};
}
{
mul.f16x2 r7104, r7101, r7103;
}
{
mul.f16x2 r7107, r7075, r7099;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7075;
mov.b32 r7110, {high, low};
}
{
fma.rn.f16x2 r7112, r7104, r7110, r7107;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7112;
mov.b32 r7116, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7112;
mov.b32 r7118, {high, high};
}
{
mul.f16x2 r7120, r6255, r7118;
}
{
fma.rn.f16x2 r7123, r6219, r7116, r7120;
}
{
mul.f16x2 r7127, r6219, r7118;
}
{
neg.f16x2 r7130, r7127;
}
{
fma.rn.f16x2 r7132, r6255, r7116, r7130;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7136, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7138, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7140, {low, high};
}
{
mul.f16x2 r7141, r7138, r7140;
}
{
mul.f16x2 r7144, r7112, r7136;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7112;
mov.b32 r7147, {high, low};
}
{
fma.rn.f16x2 r7149, r7141, r7147, r7144;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7149;
mov.b32 r7153, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7149;
mov.b32 r7155, {high, high};
}
{
mul.f16x2 r7157, r6341, r7155;
}
{
fma.rn.f16x2 r7160, r6305, r7153, r7157;
}
{
mul.f16x2 r7164, r6305, r7155;
}
{
neg.f16x2 r7167, r7164;
}
{
fma.rn.f16x2 r7169, r6341, r7153, r7167;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7173, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7175, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7177, {low, high};
}
{
mul.f16x2 r7178, r7175, r7177;
}
{
mul.f16x2 r7181, r7149, r7173;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7149;
mov.b32 r7184, {high, low};
}
{
fma.rn.f16x2 r7186, r7178, r7184, r7181;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7186;
mov.b32 r7190, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7186;
mov.b32 r7192, {high, high};
}
{
mul.f16x2 r7194, r6427, r7192;
}
{
fma.rn.f16x2 r7197, r6391, r7190, r7194;
}
{
mul.f16x2 r7201, r6391, r7192;
}
{
neg.f16x2 r7204, r7201;
}
{
fma.rn.f16x2 r7206, r6427, r7190, r7204;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7210, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7212, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7214, {low, high};
}
{
mul.f16x2 r7215, r7212, r7214;
}
{
mul.f16x2 r7218, r7186, r7210;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7186;
mov.b32 r7221, {high, low};
}
{
fma.rn.f16x2 r7223, r7215, r7221, r7218;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7223;
mov.b32 r7227, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7223;
mov.b32 r7229, {high, high};
}
{
mul.f16x2 r7231, r6513, r7229;
}
{
fma.rn.f16x2 r7234, r6477, r7227, r7231;
}
{
mul.f16x2 r7238, r6477, r7229;
}
{
neg.f16x2 r7241, r7238;
}
{
fma.rn.f16x2 r7243, r6513, r7227, r7241;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7247, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7249, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7251, {low, high};
}
{
mul.f16x2 r7252, r7249, r7251;
}
{
mul.f16x2 r7255, r7223, r7247;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7223;
mov.b32 r7258, {high, low};
}
{
fma.rn.f16x2 r7260, r7252, r7258, r7255;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7260;
mov.b32 r7264, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7260;
mov.b32 r7266, {high, high};
}
{
mul.f16x2 r7268, r6599, r7266;
}
{
fma.rn.f16x2 r7271, r6563, r7264, r7268;
}
{
mul.f16x2 r7275, r6563, r7266;
}
{
neg.f16x2 r7278, r7275;
}
{
fma.rn.f16x2 r7280, r6599, r7264, r7278;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7284, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7286, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7288, {low, high};
}
{
mul.f16x2 r7289, r7286, r7288;
}
{
mul.f16x2 r7292, r7260, r7284;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7260;
mov.b32 r7295, {high, low};
}
{
fma.rn.f16x2 r7297, r7289, r7295, r7292;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7297;
mov.b32 r7301, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7297;
mov.b32 r7303, {high, high};
}
{
mul.f16x2 r7305, r6685, r7303;
}
{
fma.rn.f16x2 r7308, r6649, r7301, r7305;
}
{
mul.f16x2 r7312, r6649, r7303;
}
{
neg.f16x2 r7315, r7312;
}
{
fma.rn.f16x2 r7317, r6685, r7301, r7315;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7321, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7323, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7325, {low, high};
}
{
mul.f16x2 r7326, r7323, r7325;
}
{
mul.f16x2 r7329, r7297, r7321;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7297;
mov.b32 r7332, {high, low};
}
{
fma.rn.f16x2 r7334, r7326, r7332, r7329;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7334;
mov.b32 r7338, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7334;
mov.b32 r7340, {high, high};
}
{
mul.f16x2 r7342, r6015, r7340;
}
{
fma.rn.f16x2 r7345, r5979, r7338, r7342;
}
{
mul.f16x2 r7349, r5979, r7340;
}
{
neg.f16x2 r7352, r7349;
}
{
fma.rn.f16x2 r7354, r6015, r7338, r7352;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7358, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7360, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7362, {low, high};
}
{
mul.f16x2 r7363, r7360, r7362;
}
{
mul.f16x2 r7366, r7334, r7358;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7334;
mov.b32 r7369, {high, low};
}
{
fma.rn.f16x2 r7371, r7363, r7369, r7366;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7371;
mov.b32 r7375, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7371;
mov.b32 r7377, {high, high};
}
{
mul.f16x2 r7379, r6101, r7377;
}
{
fma.rn.f16x2 r7382, r6065, r7375, r7379;
}
{
mul.f16x2 r7386, r6065, r7377;
}
{
neg.f16x2 r7389, r7386;
}
{
fma.rn.f16x2 r7391, r6101, r7375, r7389;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7395, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7397, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7399, {low, high};
}
{
mul.f16x2 r7400, r7397, r7399;
}
{
mul.f16x2 r7403, r7371, r7395;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7371;
mov.b32 r7406, {high, low};
}
{
fma.rn.f16x2 r7408, r7400, r7406, r7403;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7408;
mov.b32 r7412, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7408;
mov.b32 r7414, {high, high};
}
{
mul.f16x2 r7416, r6187, r7414;
}
{
fma.rn.f16x2 r7419, r6151, r7412, r7416;
}
{
mul.f16x2 r7423, r6151, r7414;
}
{
neg.f16x2 r7426, r7423;
}
{
fma.rn.f16x2 r7428, r6187, r7412, r7426;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7432, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7434, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7436, {low, high};
}
{
mul.f16x2 r7437, r7434, r7436;
}
{
mul.f16x2 r7440, r7408, r7432;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7408;
mov.b32 r7443, {high, low};
}
{
fma.rn.f16x2 r7445, r7437, r7443, r7440;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7445;
mov.b32 r7449, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7445;
mov.b32 r7451, {high, high};
}
{
mul.f16x2 r7453, r6273, r7451;
}
{
fma.rn.f16x2 r7456, r6237, r7449, r7453;
}
{
mul.f16x2 r7460, r6237, r7451;
}
{
neg.f16x2 r7463, r7460;
}
{
fma.rn.f16x2 r7465, r6273, r7449, r7463;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7469, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7471, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7473, {low, high};
}
{
mul.f16x2 r7474, r7471, r7473;
}
{
mul.f16x2 r7477, r7445, r7469;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7445;
mov.b32 r7480, {high, low};
}
{
fma.rn.f16x2 r7482, r7474, r7480, r7477;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7482;
mov.b32 r7486, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7482;
mov.b32 r7488, {high, high};
}
{
mul.f16x2 r7490, r6359, r7488;
}
{
fma.rn.f16x2 r7493, r6323, r7486, r7490;
}
{
mul.f16x2 r7497, r6323, r7488;
}
{
neg.f16x2 r7500, r7497;
}
{
fma.rn.f16x2 r7502, r6359, r7486, r7500;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7506, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7508, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7510, {low, high};
}
{
mul.f16x2 r7511, r7508, r7510;
}
{
mul.f16x2 r7514, r7482, r7506;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7482;
mov.b32 r7517, {high, low};
}
{
fma.rn.f16x2 r7519, r7511, r7517, r7514;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7519;
mov.b32 r7523, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7519;
mov.b32 r7525, {high, high};
}
{
mul.f16x2 r7527, r6445, r7525;
}
{
fma.rn.f16x2 r7530, r6409, r7523, r7527;
}
{
mul.f16x2 r7534, r6409, r7525;
}
{
neg.f16x2 r7537, r7534;
}
{
fma.rn.f16x2 r7539, r6445, r7523, r7537;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7543, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7545, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7547, {low, high};
}
{
mul.f16x2 r7548, r7545, r7547;
}
{
mul.f16x2 r7551, r7519, r7543;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7519;
mov.b32 r7554, {high, low};
}
{
fma.rn.f16x2 r7556, r7548, r7554, r7551;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7556;
mov.b32 r7560, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7556;
mov.b32 r7562, {high, high};
}
{
mul.f16x2 r7564, r6531, r7562;
}
{
fma.rn.f16x2 r7567, r6495, r7560, r7564;
}
{
mul.f16x2 r7571, r6495, r7562;
}
{
neg.f16x2 r7574, r7571;
}
{
fma.rn.f16x2 r7576, r6531, r7560, r7574;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7580, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7582, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7584, {low, high};
}
{
mul.f16x2 r7585, r7582, r7584;
}
{
mul.f16x2 r7588, r7556, r7580;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7556;
mov.b32 r7591, {high, low};
}
{
fma.rn.f16x2 r7593, r7585, r7591, r7588;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7593;
mov.b32 r7597, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7593;
mov.b32 r7599, {high, high};
}
{
mul.f16x2 r7601, r6617, r7599;
}
{
fma.rn.f16x2 r7604, r6581, r7597, r7601;
}
{
mul.f16x2 r7608, r6581, r7599;
}
{
neg.f16x2 r7611, r7608;
}
{
fma.rn.f16x2 r7613, r6617, r7597, r7611;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7617, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6706;
mov.b32 r7619, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7621, {low, high};
}
{
mul.f16x2 r7622, r7619, r7621;
}
{
mul.f16x2 r7625, r7593, r7617;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7593;
mov.b32 r7628, {high, low};
}
{
fma.rn.f16x2 r7630, r7622, r7628, r7625;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7630;
mov.b32 r7634, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7630;
mov.b32 r7636, {high, high};
}
{
mul.f16x2 r7638, r6703, r7636;
}
{
fma.rn.f16x2 r7641, r6667, r7634, r7638;
}
{
mul.f16x2 r7645, r6667, r7636;
}
{
neg.f16x2 r7648, r7645;
}
{
fma.rn.f16x2 r7650, r6703, r7634, r7648;
}
barrier.sync 0;
mad.lo.s32 r8464, r8459, 2916, r8463;
st.shared.u32 [r8464], r5937;
st.shared.u32 [r8464+108], r6716;
st.shared.u32 [r8464+216], r6753;
st.shared.u32 [r8464+324], r6790;
st.shared.u32 [r8464+432], r6827;
st.shared.u32 [r8464+540], r6864;
st.shared.u32 [r8464+648], r6901;
st.shared.u32 [r8464+756], r6938;
st.shared.u32 [r8464+864], r6975;
st.shared.u32 [r8464+972], r7012;
st.shared.u32 [r8464+1080], r7049;
st.shared.u32 [r8464+1188], r7086;
st.shared.u32 [r8464+1296], r7123;
st.shared.u32 [r8464+1404], r7160;
st.shared.u32 [r8464+1512], r7197;
st.shared.u32 [r8464+1620], r7234;
st.shared.u32 [r8464+1728], r7271;
st.shared.u32 [r8464+1836], r7308;
st.shared.u32 [r8464+1944], r7345;
st.shared.u32 [r8464+2052], r7382;
st.shared.u32 [r8464+2160], r7419;
st.shared.u32 [r8464+2268], r7456;
st.shared.u32 [r8464+2376], r7493;
st.shared.u32 [r8464+2484], r7530;
st.shared.u32 [r8464+2592], r7567;
st.shared.u32 [r8464+2700], r7604;
st.shared.u32 [r8464+2808], r7641;
barrier.sync 0;
ld.shared.u32 r7677, [r8454];
ld.shared.u32 r7763, [r8454+324];
ld.shared.u32 r7849, [r8454+648];
ld.shared.u32 r7935, [r8454+972];
ld.shared.u32 r8021, [r8454+1296];
ld.shared.u32 r8107, [r8454+1620];
ld.shared.u32 r8193, [r8454+1944];
ld.shared.u32 r8279, [r8454+2268];
ld.shared.u32 r8365, [r8454+2592];
ld.shared.u32 r7674, [r8454+2916];
ld.shared.u32 r7760, [r8454+3240];
ld.shared.u32 r7846, [r8454+3564];
ld.shared.u32 r7932, [r8454+3888];
ld.shared.u32 r8018, [r8454+4212];
ld.shared.u32 r8104, [r8454+4536];
ld.shared.u32 r8190, [r8454+4860];
ld.shared.u32 r8276, [r8454+5184];
ld.shared.u32 r8362, [r8454+5508];
ld.shared.u32 r7675, [r8454+5832];
ld.shared.u32 r7761, [r8454+6156];
ld.shared.u32 r7847, [r8454+6480];
ld.shared.u32 r7933, [r8454+6804];
ld.shared.u32 r8019, [r8454+7128];
ld.shared.u32 r8105, [r8454+7452];
ld.shared.u32 r8191, [r8454+7776];
ld.shared.u32 r8277, [r8454+8100];
ld.shared.u32 r8363, [r8454+8424];
barrier.sync 0;
st.shared.u32 [r8464], r5943;
st.shared.u32 [r8464+108], r6725;
st.shared.u32 [r8464+216], r6762;
st.shared.u32 [r8464+324], r6799;
st.shared.u32 [r8464+432], r6836;
st.shared.u32 [r8464+540], r6873;
st.shared.u32 [r8464+648], r6910;
st.shared.u32 [r8464+756], r6947;
st.shared.u32 [r8464+864], r6984;
st.shared.u32 [r8464+972], r7021;
st.shared.u32 [r8464+1080], r7058;
st.shared.u32 [r8464+1188], r7095;
st.shared.u32 [r8464+1296], r7132;
st.shared.u32 [r8464+1404], r7169;
st.shared.u32 [r8464+1512], r7206;
st.shared.u32 [r8464+1620], r7243;
st.shared.u32 [r8464+1728], r7280;
st.shared.u32 [r8464+1836], r7317;
st.shared.u32 [r8464+1944], r7354;
st.shared.u32 [r8464+2052], r7391;
st.shared.u32 [r8464+2160], r7428;
st.shared.u32 [r8464+2268], r7465;
st.shared.u32 [r8464+2376], r7502;
st.shared.u32 [r8464+2484], r7539;
st.shared.u32 [r8464+2592], r7576;
st.shared.u32 [r8464+2700], r7613;
st.shared.u32 [r8464+2808], r7650;
barrier.sync 0;
ld.shared.u32 r7683, [r8454];
ld.shared.u32 r7769, [r8454+324];
ld.shared.u32 r7855, [r8454+648];
ld.shared.u32 r7941, [r8454+972];
ld.shared.u32 r8027, [r8454+1296];
ld.shared.u32 r8113, [r8454+1620];
ld.shared.u32 r8199, [r8454+1944];
ld.shared.u32 r8285, [r8454+2268];
ld.shared.u32 r8371, [r8454+2592];
ld.shared.u32 r7680, [r8454+2916];
ld.shared.u32 r7766, [r8454+3240];
ld.shared.u32 r7852, [r8454+3564];
ld.shared.u32 r7938, [r8454+3888];
ld.shared.u32 r8024, [r8454+4212];
ld.shared.u32 r8110, [r8454+4536];
ld.shared.u32 r8196, [r8454+4860];
ld.shared.u32 r8282, [r8454+5184];
ld.shared.u32 r8368, [r8454+5508];
ld.shared.u32 r7681, [r8454+5832];
ld.shared.u32 r7767, [r8454+6156];
ld.shared.u32 r7853, [r8454+6480];
ld.shared.u32 r7939, [r8454+6804];
ld.shared.u32 r8025, [r8454+7128];
ld.shared.u32 r8111, [r8454+7452];
ld.shared.u32 r8197, [r8454+7776];
ld.shared.u32 r8283, [r8454+8100];
ld.shared.u32 r8369, [r8454+8424];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r7671, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r7672, {low, high};
}
{
add.f16x2 r7673, r7674, r7675;
}
{
add.f16x2 %0, r7677, r7673;
}
{
add.f16x2 r7679, r7680, r7681;
}
{
add.f16x2 %1, r7683, r7679;
}
{
add.f16x2 r7685, r7674, r7675;
}
{
mul.f16x2 r7688, r7685, r7671;
}
{
add.f16x2 r7691, r7677, r7688;
}
{
sub.f16x2 r7694, r7680, r7681;
}
{
mul.f16x2 r7697, r7694, r7672;
}
{
add.f16x2 %18, r7691, r7697;
}
{
add.f16x2 r7703, r7674, r7675;
}
{
mul.f16x2 r7706, r7703, r7671;
}
{
add.f16x2 r7709, r7677, r7706;
}
{
sub.f16x2 r7712, r7680, r7681;
}
{
mul.f16x2 r7715, r7712, r7672;
}
{
sub.f16x2 %36, r7709, r7715;
}
{
add.f16x2 r7721, r7680, r7681;
}
{
mul.f16x2 r7724, r7721, r7671;
}
{
add.f16x2 r7727, r7683, r7724;
}
{
sub.f16x2 r7730, r7674, r7675;
}
{
mul.f16x2 r7733, r7730, r7672;
}
{
sub.f16x2 %19, r7727, r7733;
}
{
add.f16x2 r7739, r7680, r7681;
}
{
mul.f16x2 r7742, r7739, r7671;
}
{
add.f16x2 r7745, r7683, r7742;
}
{
sub.f16x2 r7748, r7674, r7675;
}
{
mul.f16x2 r7751, r7748, r7672;
}
{
add.f16x2 %37, r7745, r7751;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r7757, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r7758, {low, high};
}
{
add.f16x2 r7759, r7760, r7761;
}
{
add.f16x2 %2, r7763, r7759;
}
{
add.f16x2 r7765, r7766, r7767;
}
{
add.f16x2 %3, r7769, r7765;
}
{
add.f16x2 r7771, r7760, r7761;
}
{
mul.f16x2 r7774, r7771, r7757;
}
{
add.f16x2 r7777, r7763, r7774;
}
{
sub.f16x2 r7780, r7766, r7767;
}
{
mul.f16x2 r7783, r7780, r7758;
}
{
add.f16x2 %20, r7777, r7783;
}
{
add.f16x2 r7789, r7760, r7761;
}
{
mul.f16x2 r7792, r7789, r7757;
}
{
add.f16x2 r7795, r7763, r7792;
}
{
sub.f16x2 r7798, r7766, r7767;
}
{
mul.f16x2 r7801, r7798, r7758;
}
{
sub.f16x2 %38, r7795, r7801;
}
{
add.f16x2 r7807, r7766, r7767;
}
{
mul.f16x2 r7810, r7807, r7757;
}
{
add.f16x2 r7813, r7769, r7810;
}
{
sub.f16x2 r7816, r7760, r7761;
}
{
mul.f16x2 r7819, r7816, r7758;
}
{
sub.f16x2 %21, r7813, r7819;
}
{
add.f16x2 r7825, r7766, r7767;
}
{
mul.f16x2 r7828, r7825, r7757;
}
{
add.f16x2 r7831, r7769, r7828;
}
{
sub.f16x2 r7834, r7760, r7761;
}
{
mul.f16x2 r7837, r7834, r7758;
}
{
add.f16x2 %39, r7831, r7837;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r7843, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r7844, {low, high};
}
{
add.f16x2 r7845, r7846, r7847;
}
{
add.f16x2 %4, r7849, r7845;
}
{
add.f16x2 r7851, r7852, r7853;
}
{
add.f16x2 %5, r7855, r7851;
}
{
add.f16x2 r7857, r7846, r7847;
}
{
mul.f16x2 r7860, r7857, r7843;
}
{
add.f16x2 r7863, r7849, r7860;
}
{
sub.f16x2 r7866, r7852, r7853;
}
{
mul.f16x2 r7869, r7866, r7844;
}
{
add.f16x2 %22, r7863, r7869;
}
{
add.f16x2 r7875, r7846, r7847;
}
{
mul.f16x2 r7878, r7875, r7843;
}
{
add.f16x2 r7881, r7849, r7878;
}
{
sub.f16x2 r7884, r7852, r7853;
}
{
mul.f16x2 r7887, r7884, r7844;
}
{
sub.f16x2 %40, r7881, r7887;
}
{
add.f16x2 r7893, r7852, r7853;
}
{
mul.f16x2 r7896, r7893, r7843;
}
{
add.f16x2 r7899, r7855, r7896;
}
{
sub.f16x2 r7902, r7846, r7847;
}
{
mul.f16x2 r7905, r7902, r7844;
}
{
sub.f16x2 %23, r7899, r7905;
}
{
add.f16x2 r7911, r7852, r7853;
}
{
mul.f16x2 r7914, r7911, r7843;
}
{
add.f16x2 r7917, r7855, r7914;
}
{
sub.f16x2 r7920, r7846, r7847;
}
{
mul.f16x2 r7923, r7920, r7844;
}
{
add.f16x2 %41, r7917, r7923;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r7929, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r7930, {low, high};
}
{
add.f16x2 r7931, r7932, r7933;
}
{
add.f16x2 %6, r7935, r7931;
}
{
add.f16x2 r7937, r7938, r7939;
}
{
add.f16x2 %7, r7941, r7937;
}
{
add.f16x2 r7943, r7932, r7933;
}
{
mul.f16x2 r7946, r7943, r7929;
}
{
add.f16x2 r7949, r7935, r7946;
}
{
sub.f16x2 r7952, r7938, r7939;
}
{
mul.f16x2 r7955, r7952, r7930;
}
{
add.f16x2 %24, r7949, r7955;
}
{
add.f16x2 r7961, r7932, r7933;
}
{
mul.f16x2 r7964, r7961, r7929;
}
{
add.f16x2 r7967, r7935, r7964;
}
{
sub.f16x2 r7970, r7938, r7939;
}
{
mul.f16x2 r7973, r7970, r7930;
}
{
sub.f16x2 %42, r7967, r7973;
}
{
add.f16x2 r7979, r7938, r7939;
}
{
mul.f16x2 r7982, r7979, r7929;
}
{
add.f16x2 r7985, r7941, r7982;
}
{
sub.f16x2 r7988, r7932, r7933;
}
{
mul.f16x2 r7991, r7988, r7930;
}
{
sub.f16x2 %25, r7985, r7991;
}
{
add.f16x2 r7997, r7938, r7939;
}
{
mul.f16x2 r8000, r7997, r7929;
}
{
add.f16x2 r8003, r7941, r8000;
}
{
sub.f16x2 r8006, r7932, r7933;
}
{
mul.f16x2 r8009, r8006, r7930;
}
{
add.f16x2 %43, r8003, r8009;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8015, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8016, {low, high};
}
{
add.f16x2 r8017, r8018, r8019;
}
{
add.f16x2 %8, r8021, r8017;
}
{
add.f16x2 r8023, r8024, r8025;
}
{
add.f16x2 %9, r8027, r8023;
}
{
add.f16x2 r8029, r8018, r8019;
}
{
mul.f16x2 r8032, r8029, r8015;
}
{
add.f16x2 r8035, r8021, r8032;
}
{
sub.f16x2 r8038, r8024, r8025;
}
{
mul.f16x2 r8041, r8038, r8016;
}
{
add.f16x2 %26, r8035, r8041;
}
{
add.f16x2 r8047, r8018, r8019;
}
{
mul.f16x2 r8050, r8047, r8015;
}
{
add.f16x2 r8053, r8021, r8050;
}
{
sub.f16x2 r8056, r8024, r8025;
}
{
mul.f16x2 r8059, r8056, r8016;
}
{
sub.f16x2 %44, r8053, r8059;
}
{
add.f16x2 r8065, r8024, r8025;
}
{
mul.f16x2 r8068, r8065, r8015;
}
{
add.f16x2 r8071, r8027, r8068;
}
{
sub.f16x2 r8074, r8018, r8019;
}
{
mul.f16x2 r8077, r8074, r8016;
}
{
sub.f16x2 %27, r8071, r8077;
}
{
add.f16x2 r8083, r8024, r8025;
}
{
mul.f16x2 r8086, r8083, r8015;
}
{
add.f16x2 r8089, r8027, r8086;
}
{
sub.f16x2 r8092, r8018, r8019;
}
{
mul.f16x2 r8095, r8092, r8016;
}
{
add.f16x2 %45, r8089, r8095;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8102, {low, high};
}
{
add.f16x2 r8103, r8104, r8105;
}
{
add.f16x2 %10, r8107, r8103;
}
{
add.f16x2 r8109, r8110, r8111;
}
{
add.f16x2 %11, r8113, r8109;
}
{
add.f16x2 r8115, r8104, r8105;
}
{
mul.f16x2 r8118, r8115, r8101;
}
{
add.f16x2 r8121, r8107, r8118;
}
{
sub.f16x2 r8124, r8110, r8111;
}
{
mul.f16x2 r8127, r8124, r8102;
}
{
add.f16x2 %28, r8121, r8127;
}
{
add.f16x2 r8133, r8104, r8105;
}
{
mul.f16x2 r8136, r8133, r8101;
}
{
add.f16x2 r8139, r8107, r8136;
}
{
sub.f16x2 r8142, r8110, r8111;
}
{
mul.f16x2 r8145, r8142, r8102;
}
{
sub.f16x2 %46, r8139, r8145;
}
{
add.f16x2 r8151, r8110, r8111;
}
{
mul.f16x2 r8154, r8151, r8101;
}
{
add.f16x2 r8157, r8113, r8154;
}
{
sub.f16x2 r8160, r8104, r8105;
}
{
mul.f16x2 r8163, r8160, r8102;
}
{
sub.f16x2 %29, r8157, r8163;
}
{
add.f16x2 r8169, r8110, r8111;
}
{
mul.f16x2 r8172, r8169, r8101;
}
{
add.f16x2 r8175, r8113, r8172;
}
{
sub.f16x2 r8178, r8104, r8105;
}
{
mul.f16x2 r8181, r8178, r8102;
}
{
add.f16x2 %47, r8175, r8181;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8187, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8188, {low, high};
}
{
add.f16x2 r8189, r8190, r8191;
}
{
add.f16x2 %12, r8193, r8189;
}
{
add.f16x2 r8195, r8196, r8197;
}
{
add.f16x2 %13, r8199, r8195;
}
{
add.f16x2 r8201, r8190, r8191;
}
{
mul.f16x2 r8204, r8201, r8187;
}
{
add.f16x2 r8207, r8193, r8204;
}
{
sub.f16x2 r8210, r8196, r8197;
}
{
mul.f16x2 r8213, r8210, r8188;
}
{
add.f16x2 %30, r8207, r8213;
}
{
add.f16x2 r8219, r8190, r8191;
}
{
mul.f16x2 r8222, r8219, r8187;
}
{
add.f16x2 r8225, r8193, r8222;
}
{
sub.f16x2 r8228, r8196, r8197;
}
{
mul.f16x2 r8231, r8228, r8188;
}
{
sub.f16x2 %48, r8225, r8231;
}
{
add.f16x2 r8237, r8196, r8197;
}
{
mul.f16x2 r8240, r8237, r8187;
}
{
add.f16x2 r8243, r8199, r8240;
}
{
sub.f16x2 r8246, r8190, r8191;
}
{
mul.f16x2 r8249, r8246, r8188;
}
{
sub.f16x2 %31, r8243, r8249;
}
{
add.f16x2 r8255, r8196, r8197;
}
{
mul.f16x2 r8258, r8255, r8187;
}
{
add.f16x2 r8261, r8199, r8258;
}
{
sub.f16x2 r8264, r8190, r8191;
}
{
mul.f16x2 r8267, r8264, r8188;
}
{
add.f16x2 %49, r8261, r8267;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8273, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8274, {low, high};
}
{
add.f16x2 r8275, r8276, r8277;
}
{
add.f16x2 %14, r8279, r8275;
}
{
add.f16x2 r8281, r8282, r8283;
}
{
add.f16x2 %15, r8285, r8281;
}
{
add.f16x2 r8287, r8276, r8277;
}
{
mul.f16x2 r8290, r8287, r8273;
}
{
add.f16x2 r8293, r8279, r8290;
}
{
sub.f16x2 r8296, r8282, r8283;
}
{
mul.f16x2 r8299, r8296, r8274;
}
{
add.f16x2 %32, r8293, r8299;
}
{
add.f16x2 r8305, r8276, r8277;
}
{
mul.f16x2 r8308, r8305, r8273;
}
{
add.f16x2 r8311, r8279, r8308;
}
{
sub.f16x2 r8314, r8282, r8283;
}
{
mul.f16x2 r8317, r8314, r8274;
}
{
sub.f16x2 %50, r8311, r8317;
}
{
add.f16x2 r8323, r8282, r8283;
}
{
mul.f16x2 r8326, r8323, r8273;
}
{
add.f16x2 r8329, r8285, r8326;
}
{
sub.f16x2 r8332, r8276, r8277;
}
{
mul.f16x2 r8335, r8332, r8274;
}
{
sub.f16x2 %33, r8329, r8335;
}
{
add.f16x2 r8341, r8282, r8283;
}
{
mul.f16x2 r8344, r8341, r8273;
}
{
add.f16x2 r8347, r8285, r8344;
}
{
sub.f16x2 r8350, r8276, r8277;
}
{
mul.f16x2 r8353, r8350, r8274;
}
{
add.f16x2 %51, r8347, r8353;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8359, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8360, {low, high};
}
{
add.f16x2 r8361, r8362, r8363;
}
{
add.f16x2 %16, r8365, r8361;
}
{
add.f16x2 r8367, r8368, r8369;
}
{
add.f16x2 %17, r8371, r8367;
}
{
add.f16x2 r8373, r8362, r8363;
}
{
mul.f16x2 r8376, r8373, r8359;
}
{
add.f16x2 r8379, r8365, r8376;
}
{
sub.f16x2 r8382, r8368, r8369;
}
{
mul.f16x2 r8385, r8382, r8360;
}
{
add.f16x2 %34, r8379, r8385;
}
{
add.f16x2 r8391, r8362, r8363;
}
{
mul.f16x2 r8394, r8391, r8359;
}
{
add.f16x2 r8397, r8365, r8394;
}
{
sub.f16x2 r8400, r8368, r8369;
}
{
mul.f16x2 r8403, r8400, r8360;
}
{
sub.f16x2 %52, r8397, r8403;
}
{
add.f16x2 r8409, r8368, r8369;
}
{
mul.f16x2 r8412, r8409, r8359;
}
{
add.f16x2 r8415, r8371, r8412;
}
{
sub.f16x2 r8418, r8362, r8363;
}
{
mul.f16x2 r8421, r8418, r8360;
}
{
sub.f16x2 %35, r8415, r8421;
}
{
add.f16x2 r8427, r8368, r8369;
}
{
mul.f16x2 r8430, r8427, r8359;
}
{
add.f16x2 r8433, r8371, r8430;
}
{
sub.f16x2 r8436, r8362, r8363;
}
{
mul.f16x2 r8439, r8436, r8360;
}
{
add.f16x2 %53, r8433, r8439;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[9].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1096, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<95>;
.reg .b32 r<1110>;
.reg .b64 rd<14>;
mov.u32 r1065, %tid.y;
mov.u32 r1066, %6;
mad.lo.s32 r1067, r1065, 17496, r1066;
mov.u32 r1068, %tid.x;
mov.f32 f74, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r1, {low, high};
}
mov.f32 f76, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
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
mul.wide.u32 rd2, r1068, 1508246403;
shr.u64 rd3, rd2, 40;
cvt.u32.u64 r1069, rd3;
mul.lo.s32 r1070, r1069, 729;
sub.s32 r1071, r1068, r1070;
mad.lo.s32 r1072, r1069, 17496, r1067;
cvt.rn.f32.u32 f77, r1071;
mul.f32 f78, f77, 0f3B3C4870;
cos.approx.f32 f5, f78;
sin.approx.f32 f79, f78;
neg.f32 f6, f79;
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
mov.f32 f69, 0fBF800000;
mov.f32 f70, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
mad.lo.s32 r1073, r1071, 24, r1072;
st.shared.v2.f32 [r1073], {r6, r12};
st.shared.v2.f32 [r1073+8], {r97, r106};
st.shared.v2.f32 [r1073+16], {r134, r143};
barrier.sync 0;
shl.b32 r1074, r1071, 4;
sub.s32 r1075, r1073, r1074;
ld.shared.u32 r170, [r1075];
ld.shared.u32 r176, [r1075+4];
ld.shared.u32 r167, [r1075+5832];
ld.shared.u32 r173, [r1075+5836];
ld.shared.u32 r168, [r1075+11664];
ld.shared.u32 r174, [r1075+11668];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r164, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
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
mul.wide.u32 rd4, r1071, -1431655765;
shr.u64 rd5, rd4, 33;
cvt.u32.u64 r1076, rd5;
mul.lo.s32 r1077, r1076, 3;
sub.s32 r1078, r1071, r1077;
shl.b32 r1079, r1078, 3;
add.s32 r1080, r1072, r1079;
cvt.rn.f32.u32 f80, r1076;
mul.f32 f81, f80, 0f3C0D3654;
cos.approx.f32 f17, f81;
sin.approx.f32 f82, f81;
neg.f32 f18, f82;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
mad.lo.s32 r1081, r1076, 72, r1080;
st.shared.u32 [r1081], r169;
st.shared.u32 [r1081+4], r175;
st.shared.u32 [r1081+24], r260;
st.shared.u32 [r1081+28], r269;
st.shared.u32 [r1081+48], r297;
st.shared.u32 [r1081+52], r306;
barrier.sync 0;
ld.shared.u32 r333, [r1075];
ld.shared.u32 r339, [r1075+4];
ld.shared.u32 r330, [r1075+5832];
ld.shared.u32 r336, [r1075+5836];
ld.shared.u32 r331, [r1075+11664];
ld.shared.u32 r337, [r1075+11668];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r327, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
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
mul.wide.u32 rd6, r1071, 954437177;
shr.u64 rd7, rd6, 33;
cvt.u32.u64 r1082, rd7;
mul.lo.s32 r1083, r1082, 9;
sub.s32 r1084, r1071, r1083;
shl.b32 r1085, r1084, 3;
add.s32 r1086, r1072, r1085;
cvt.rn.f32.u32 f83, r1082;
mul.f32 f84, f83, 0f3CD3D17E;
cos.approx.f32 f29, f84;
sin.approx.f32 f85, f84;
neg.f32 f30, f85;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
mad.lo.s32 r1087, r1082, 216, r1086;
st.shared.u32 [r1087], r332;
st.shared.u32 [r1087+4], r338;
st.shared.u32 [r1087+72], r423;
st.shared.u32 [r1087+76], r432;
st.shared.u32 [r1087+144], r460;
st.shared.u32 [r1087+148], r469;
barrier.sync 0;
ld.shared.u32 r496, [r1075];
ld.shared.u32 r502, [r1075+4];
ld.shared.u32 r493, [r1075+5832];
ld.shared.u32 r499, [r1075+5836];
ld.shared.u32 r494, [r1075+11664];
ld.shared.u32 r500, [r1075+11668];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r490, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r491, {low, high};
}
{
add.f16x2 r492, r493, r494;
}
{
add.f16x2 r495, r496, r492;
}
{
add.f16x2 r498, r499, r500;
}
{
add.f16x2 r501, r502, r498;
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
add.f16x2 r519, r510, r516;
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
sub.f16x2 r537, r528, r534;
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
sub.f16x2 r555, r546, r552;
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
add.f16x2 r573, r564, r570;
}
mul.wide.u32 rd8, r1071, 795364315;
shr.u64 rd9, rd8, 32;
cvt.u32.u64 r1088, rd9;
sub.s32 r1089, r1071, r1088;
shr.u32 r1090, r1089, 1;
add.s32 r1091, r1090, r1088;
shr.u32 r1092, r1091, 4;
mul.lo.s32 r1093, r1092, 27;
sub.s32 r1094, r1071, r1093;
shl.b32 r1095, r1094, 3;
add.s32 r1096, r1072, r1095;
cvt.rn.f32.u32 f86, r1092;
mul.f32 f87, f86, 0f3D9EDD1F;
cos.approx.f32 f41, f87;
sin.approx.f32 f88, f87;
neg.f32 f42, f88;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f41;
cvt.rn.f16.f32 high, f42;
mov.b32 r576, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r576;
mov.b32 r579, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r576;
mov.b32 r581, {high, high};
}
{
mul.f16x2 r583, r555, r581;
}
{
fma.rn.f16x2 r586, r519, r579, r583;
}
{
mul.f16x2 r590, r519, r581;
}
{
neg.f16x2 r593, r590;
}
{
fma.rn.f16x2 r595, r555, r579, r593;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r576;
mov.b32 r599, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r576;
mov.b32 r601, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r603, {low, high};
}
{
mul.f16x2 r604, r601, r603;
}
{
mul.f16x2 r607, r576, r599;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r576;
mov.b32 r610, {high, low};
}
{
fma.rn.f16x2 r612, r604, r610, r607;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r612;
mov.b32 r616, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r612;
mov.b32 r618, {high, high};
}
{
mul.f16x2 r620, r573, r618;
}
{
fma.rn.f16x2 r623, r537, r616, r620;
}
{
mul.f16x2 r627, r537, r618;
}
{
neg.f16x2 r630, r627;
}
{
fma.rn.f16x2 r632, r573, r616, r630;
}
barrier.sync 0;
mad.lo.s32 r1097, r1092, 648, r1096;
st.shared.u32 [r1097], r495;
st.shared.u32 [r1097+4], r501;
st.shared.u32 [r1097+216], r586;
st.shared.u32 [r1097+220], r595;
st.shared.u32 [r1097+432], r623;
st.shared.u32 [r1097+436], r632;
barrier.sync 0;
ld.shared.u32 r659, [r1075];
ld.shared.u32 r665, [r1075+4];
ld.shared.u32 r656, [r1075+5832];
ld.shared.u32 r662, [r1075+5836];
ld.shared.u32 r657, [r1075+11664];
ld.shared.u32 r663, [r1075+11668];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r653, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r654, {low, high};
}
{
add.f16x2 r655, r656, r657;
}
{
add.f16x2 r658, r659, r655;
}
{
add.f16x2 r661, r662, r663;
}
{
add.f16x2 r664, r665, r661;
}
{
add.f16x2 r667, r656, r657;
}
{
mul.f16x2 r670, r667, r653;
}
{
add.f16x2 r673, r659, r670;
}
{
sub.f16x2 r676, r662, r663;
}
{
mul.f16x2 r679, r676, r654;
}
{
add.f16x2 r682, r673, r679;
}
{
add.f16x2 r685, r656, r657;
}
{
mul.f16x2 r688, r685, r653;
}
{
add.f16x2 r691, r659, r688;
}
{
sub.f16x2 r694, r662, r663;
}
{
mul.f16x2 r697, r694, r654;
}
{
sub.f16x2 r700, r691, r697;
}
{
add.f16x2 r703, r662, r663;
}
{
mul.f16x2 r706, r703, r653;
}
{
add.f16x2 r709, r665, r706;
}
{
sub.f16x2 r712, r656, r657;
}
{
mul.f16x2 r715, r712, r654;
}
{
sub.f16x2 r718, r709, r715;
}
{
add.f16x2 r721, r662, r663;
}
{
mul.f16x2 r724, r721, r653;
}
{
add.f16x2 r727, r665, r724;
}
{
sub.f16x2 r730, r656, r657;
}
{
mul.f16x2 r733, r730, r654;
}
{
add.f16x2 r736, r727, r733;
}
mul.wide.u32 rd10, r1071, -901412889;
shr.u64 rd11, rd10, 38;
cvt.u32.u64 r1098, rd11;
mul.lo.s32 r1099, r1098, 81;
sub.s32 r1100, r1071, r1099;
shl.b32 r1101, r1100, 3;
add.s32 r1102, r1072, r1101;
cvt.rn.f32.u32 f89, r1098;
mul.f32 f90, f89, 0f3E6E4BAE;
cos.approx.f32 f53, f90;
sin.approx.f32 f91, f90;
neg.f32 f54, f91;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r739, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r739;
mov.b32 r742, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r739;
mov.b32 r744, {high, high};
}
{
mul.f16x2 r746, r718, r744;
}
{
fma.rn.f16x2 r749, r682, r742, r746;
}
{
mul.f16x2 r753, r682, r744;
}
{
neg.f16x2 r756, r753;
}
{
fma.rn.f16x2 r758, r718, r742, r756;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r739;
mov.b32 r762, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r739;
mov.b32 r764, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r766, {low, high};
}
{
mul.f16x2 r767, r764, r766;
}
{
mul.f16x2 r770, r739, r762;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r739;
mov.b32 r773, {high, low};
}
{
fma.rn.f16x2 r775, r767, r773, r770;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r775;
mov.b32 r779, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r775;
mov.b32 r781, {high, high};
}
{
mul.f16x2 r783, r736, r781;
}
{
fma.rn.f16x2 r786, r700, r779, r783;
}
{
mul.f16x2 r790, r700, r781;
}
{
neg.f16x2 r793, r790;
}
{
fma.rn.f16x2 r795, r736, r779, r793;
}
barrier.sync 0;
mad.lo.s32 r1103, r1098, 1944, r1102;
st.shared.u32 [r1103], r658;
st.shared.u32 [r1103+4], r664;
st.shared.u32 [r1103+648], r749;
st.shared.u32 [r1103+652], r758;
st.shared.u32 [r1103+1296], r786;
st.shared.u32 [r1103+1300], r795;
barrier.sync 0;
ld.shared.u32 r822, [r1075];
ld.shared.u32 r828, [r1075+4];
ld.shared.u32 r819, [r1075+5832];
ld.shared.u32 r825, [r1075+5836];
ld.shared.u32 r820, [r1075+11664];
ld.shared.u32 r826, [r1075+11668];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r816, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r817, {low, high};
}
{
add.f16x2 r818, r819, r820;
}
{
add.f16x2 r821, r822, r818;
}
{
add.f16x2 r824, r825, r826;
}
{
add.f16x2 r827, r828, r824;
}
{
add.f16x2 r830, r819, r820;
}
{
mul.f16x2 r833, r830, r816;
}
{
add.f16x2 r836, r822, r833;
}
{
sub.f16x2 r839, r825, r826;
}
{
mul.f16x2 r842, r839, r817;
}
{
add.f16x2 r845, r836, r842;
}
{
add.f16x2 r848, r819, r820;
}
{
mul.f16x2 r851, r848, r816;
}
{
add.f16x2 r854, r822, r851;
}
{
sub.f16x2 r857, r825, r826;
}
{
mul.f16x2 r860, r857, r817;
}
{
sub.f16x2 r863, r854, r860;
}
{
add.f16x2 r866, r825, r826;
}
{
mul.f16x2 r869, r866, r816;
}
{
add.f16x2 r872, r828, r869;
}
{
sub.f16x2 r875, r819, r820;
}
{
mul.f16x2 r878, r875, r817;
}
{
sub.f16x2 r881, r872, r878;
}
{
add.f16x2 r884, r825, r826;
}
{
mul.f16x2 r887, r884, r816;
}
{
add.f16x2 r890, r828, r887;
}
{
sub.f16x2 r893, r819, r820;
}
{
mul.f16x2 r896, r893, r817;
}
{
add.f16x2 r899, r890, r896;
}
mul.wide.u32 rd12, r1071, -2032597691;
shr.u64 rd13, rd12, 39;
cvt.u32.u64 r1104, rd13;
mul.lo.s32 r1105, r1104, 243;
sub.s32 r1106, r1071, r1105;
shl.b32 r1107, r1106, 3;
add.s32 r1108, r1072, r1107;
cvt.rn.f32.u32 f92, r1104;
mul.f32 f93, f92, 0f3F32B8C2;
cos.approx.f32 f65, f93;
sin.approx.f32 f94, f93;
neg.f32 f66, f94;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f65;
cvt.rn.f16.f32 high, f66;
mov.b32 r902, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r902;
mov.b32 r905, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r902;
mov.b32 r907, {high, high};
}
{
mul.f16x2 r909, r881, r907;
}
{
fma.rn.f16x2 r912, r845, r905, r909;
}
{
mul.f16x2 r916, r845, r907;
}
{
neg.f16x2 r919, r916;
}
{
fma.rn.f16x2 r921, r881, r905, r919;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r902;
mov.b32 r925, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r902;
mov.b32 r927, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r929, {low, high};
}
{
mul.f16x2 r930, r927, r929;
}
{
mul.f16x2 r933, r902, r925;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r902;
mov.b32 r936, {high, low};
}
{
fma.rn.f16x2 r938, r930, r936, r933;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r938;
mov.b32 r942, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r938;
mov.b32 r944, {high, high};
}
{
mul.f16x2 r946, r899, r944;
}
{
fma.rn.f16x2 r949, r863, r942, r946;
}
{
mul.f16x2 r953, r863, r944;
}
{
neg.f16x2 r956, r953;
}
{
fma.rn.f16x2 r958, r899, r942, r956;
}
barrier.sync 0;
mad.lo.s32 r1109, r1104, 5832, r1108;
st.shared.u32 [r1109], r821;
st.shared.u32 [r1109+4], r827;
st.shared.u32 [r1109+1944], r912;
st.shared.u32 [r1109+1948], r921;
st.shared.u32 [r1109+3888], r949;
st.shared.u32 [r1109+3892], r958;
barrier.sync 0;
ld.shared.u32 r985, [r1075];
ld.shared.u32 r991, [r1075+4];
ld.shared.u32 r982, [r1075+5832];
ld.shared.u32 r988, [r1075+5836];
ld.shared.u32 r983, [r1075+11664];
ld.shared.u32 r989, [r1075+11668];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r979, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r980, {low, high};
}
{
add.f16x2 r981, r982, r983;
}
{
add.f16x2 %0, r985, r981;
}
{
add.f16x2 r987, r988, r989;
}
{
add.f16x2 %1, r991, r987;
}
{
add.f16x2 r993, r982, r983;
}
{
mul.f16x2 r996, r993, r979;
}
{
add.f16x2 r999, r985, r996;
}
{
sub.f16x2 r1002, r988, r989;
}
{
mul.f16x2 r1005, r1002, r980;
}
{
add.f16x2 %2, r999, r1005;
}
{
add.f16x2 r1011, r982, r983;
}
{
mul.f16x2 r1014, r1011, r979;
}
{
add.f16x2 r1017, r985, r1014;
}
{
sub.f16x2 r1020, r988, r989;
}
{
mul.f16x2 r1023, r1020, r980;
}
{
sub.f16x2 %4, r1017, r1023;
}
{
add.f16x2 r1029, r988, r989;
}
{
mul.f16x2 r1032, r1029, r979;
}
{
add.f16x2 r1035, r991, r1032;
}
{
sub.f16x2 r1038, r982, r983;
}
{
mul.f16x2 r1041, r1038, r980;
}
{
sub.f16x2 %3, r1035, r1041;
}
{
add.f16x2 r1047, r988, r989;
}
{
mul.f16x2 r1050, r1047, r979;
}
{
add.f16x2 r1053, r991, r1050;
}
{
sub.f16x2 r1056, r982, r983;
}
{
mul.f16x2 r1059, r1056, r980;
}
{
add.f16x2 %5, r1053, r1059;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1097, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<95>;
.reg .b32 r<1110>;
.reg .b64 rd<14>;
mov.u32 r1065, %tid.y;
mov.u32 r1066, %6;
mad.lo.s32 r1067, r1065, 8748, r1066;
mov.u32 r1068, %tid.x;
mov.f32 f74, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r1, {low, high};
}
mov.f32 f76, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
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
mul.wide.u32 rd2, r1068, 1508246403;
shr.u64 rd3, rd2, 40;
cvt.u32.u64 r1069, rd3;
mul.lo.s32 r1070, r1069, 729;
sub.s32 r1071, r1068, r1070;
mad.lo.s32 r1072, r1069, 8748, r1067;
cvt.rn.f32.u32 f77, r1071;
mul.f32 f78, f77, 0f3B3C4870;
cos.approx.f32 f5, f78;
sin.approx.f32 f79, f78;
neg.f32 f6, f79;
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
mov.f32 f69, 0fBF800000;
mov.f32 f70, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
mad.lo.s32 r1073, r1071, 12, r1072;
st.shared.u32 [r1073], r6;
st.shared.u32 [r1073+4], r97;
st.shared.u32 [r1073+8], r134;
barrier.sync 0;
shl.b32 r1074, r1071, 3;
sub.s32 r1075, r1073, r1074;
ld.shared.u32 r170, [r1075];
ld.shared.u32 r167, [r1075+2916];
ld.shared.u32 r168, [r1075+5832];
barrier.sync 0;
st.shared.u32 [r1073], r12;
st.shared.u32 [r1073+4], r106;
st.shared.u32 [r1073+8], r143;
barrier.sync 0;
ld.shared.u32 r176, [r1075];
ld.shared.u32 r173, [r1075+2916];
ld.shared.u32 r174, [r1075+5832];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r164, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
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
mul.wide.u32 rd4, r1071, -1431655765;
shr.u64 rd5, rd4, 33;
cvt.u32.u64 r1076, rd5;
mul.lo.s32 r1077, r1076, 3;
sub.s32 r1078, r1071, r1077;
shl.b32 r1079, r1078, 2;
add.s32 r1080, r1072, r1079;
cvt.rn.f32.u32 f80, r1076;
mul.f32 f81, f80, 0f3C0D3654;
cos.approx.f32 f17, f81;
sin.approx.f32 f82, f81;
neg.f32 f18, f82;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
mad.lo.s32 r1081, r1076, 36, r1080;
st.shared.u32 [r1081], r169;
st.shared.u32 [r1081+12], r260;
st.shared.u32 [r1081+24], r297;
barrier.sync 0;
ld.shared.u32 r333, [r1075];
ld.shared.u32 r330, [r1075+2916];
ld.shared.u32 r331, [r1075+5832];
barrier.sync 0;
st.shared.u32 [r1081], r175;
st.shared.u32 [r1081+12], r269;
st.shared.u32 [r1081+24], r306;
barrier.sync 0;
ld.shared.u32 r339, [r1075];
ld.shared.u32 r336, [r1075+2916];
ld.shared.u32 r337, [r1075+5832];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r327, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
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
mul.wide.u32 rd6, r1071, 954437177;
shr.u64 rd7, rd6, 33;
cvt.u32.u64 r1082, rd7;
mul.lo.s32 r1083, r1082, 9;
sub.s32 r1084, r1071, r1083;
shl.b32 r1085, r1084, 2;
add.s32 r1086, r1072, r1085;
cvt.rn.f32.u32 f83, r1082;
mul.f32 f84, f83, 0f3CD3D17E;
cos.approx.f32 f29, f84;
sin.approx.f32 f85, f84;
neg.f32 f30, f85;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
mad.lo.s32 r1087, r1082, 108, r1086;
st.shared.u32 [r1087], r332;
st.shared.u32 [r1087+36], r423;
st.shared.u32 [r1087+72], r460;
barrier.sync 0;
ld.shared.u32 r496, [r1075];
ld.shared.u32 r493, [r1075+2916];
ld.shared.u32 r494, [r1075+5832];
barrier.sync 0;
st.shared.u32 [r1087], r338;
st.shared.u32 [r1087+36], r432;
st.shared.u32 [r1087+72], r469;
barrier.sync 0;
ld.shared.u32 r502, [r1075];
ld.shared.u32 r499, [r1075+2916];
ld.shared.u32 r500, [r1075+5832];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r490, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r491, {low, high};
}
{
add.f16x2 r492, r493, r494;
}
{
add.f16x2 r495, r496, r492;
}
{
add.f16x2 r498, r499, r500;
}
{
add.f16x2 r501, r502, r498;
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
add.f16x2 r519, r510, r516;
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
sub.f16x2 r537, r528, r534;
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
sub.f16x2 r555, r546, r552;
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
add.f16x2 r573, r564, r570;
}
mul.wide.u32 rd8, r1071, 795364315;
shr.u64 rd9, rd8, 32;
cvt.u32.u64 r1088, rd9;
sub.s32 r1089, r1071, r1088;
shr.u32 r1090, r1089, 1;
add.s32 r1091, r1090, r1088;
shr.u32 r1092, r1091, 4;
mul.lo.s32 r1093, r1092, 27;
sub.s32 r1094, r1071, r1093;
shl.b32 r1095, r1094, 2;
add.s32 r1096, r1072, r1095;
cvt.rn.f32.u32 f86, r1092;
mul.f32 f87, f86, 0f3D9EDD1F;
cos.approx.f32 f41, f87;
sin.approx.f32 f88, f87;
neg.f32 f42, f88;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f41;
cvt.rn.f16.f32 high, f42;
mov.b32 r576, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r576;
mov.b32 r579, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r576;
mov.b32 r581, {high, high};
}
{
mul.f16x2 r583, r555, r581;
}
{
fma.rn.f16x2 r586, r519, r579, r583;
}
{
mul.f16x2 r590, r519, r581;
}
{
neg.f16x2 r593, r590;
}
{
fma.rn.f16x2 r595, r555, r579, r593;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r576;
mov.b32 r599, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r576;
mov.b32 r601, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r603, {low, high};
}
{
mul.f16x2 r604, r601, r603;
}
{
mul.f16x2 r607, r576, r599;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r576;
mov.b32 r610, {high, low};
}
{
fma.rn.f16x2 r612, r604, r610, r607;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r612;
mov.b32 r616, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r612;
mov.b32 r618, {high, high};
}
{
mul.f16x2 r620, r573, r618;
}
{
fma.rn.f16x2 r623, r537, r616, r620;
}
{
mul.f16x2 r627, r537, r618;
}
{
neg.f16x2 r630, r627;
}
{
fma.rn.f16x2 r632, r573, r616, r630;
}
barrier.sync 0;
mad.lo.s32 r1097, r1092, 324, r1096;
st.shared.u32 [r1097], r495;
st.shared.u32 [r1097+108], r586;
st.shared.u32 [r1097+216], r623;
barrier.sync 0;
ld.shared.u32 r659, [r1075];
ld.shared.u32 r656, [r1075+2916];
ld.shared.u32 r657, [r1075+5832];
barrier.sync 0;
st.shared.u32 [r1097], r501;
st.shared.u32 [r1097+108], r595;
st.shared.u32 [r1097+216], r632;
barrier.sync 0;
ld.shared.u32 r665, [r1075];
ld.shared.u32 r662, [r1075+2916];
ld.shared.u32 r663, [r1075+5832];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r653, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r654, {low, high};
}
{
add.f16x2 r655, r656, r657;
}
{
add.f16x2 r658, r659, r655;
}
{
add.f16x2 r661, r662, r663;
}
{
add.f16x2 r664, r665, r661;
}
{
add.f16x2 r667, r656, r657;
}
{
mul.f16x2 r670, r667, r653;
}
{
add.f16x2 r673, r659, r670;
}
{
sub.f16x2 r676, r662, r663;
}
{
mul.f16x2 r679, r676, r654;
}
{
add.f16x2 r682, r673, r679;
}
{
add.f16x2 r685, r656, r657;
}
{
mul.f16x2 r688, r685, r653;
}
{
add.f16x2 r691, r659, r688;
}
{
sub.f16x2 r694, r662, r663;
}
{
mul.f16x2 r697, r694, r654;
}
{
sub.f16x2 r700, r691, r697;
}
{
add.f16x2 r703, r662, r663;
}
{
mul.f16x2 r706, r703, r653;
}
{
add.f16x2 r709, r665, r706;
}
{
sub.f16x2 r712, r656, r657;
}
{
mul.f16x2 r715, r712, r654;
}
{
sub.f16x2 r718, r709, r715;
}
{
add.f16x2 r721, r662, r663;
}
{
mul.f16x2 r724, r721, r653;
}
{
add.f16x2 r727, r665, r724;
}
{
sub.f16x2 r730, r656, r657;
}
{
mul.f16x2 r733, r730, r654;
}
{
add.f16x2 r736, r727, r733;
}
mul.wide.u32 rd10, r1071, -901412889;
shr.u64 rd11, rd10, 38;
cvt.u32.u64 r1098, rd11;
mul.lo.s32 r1099, r1098, 81;
sub.s32 r1100, r1071, r1099;
shl.b32 r1101, r1100, 2;
add.s32 r1102, r1072, r1101;
cvt.rn.f32.u32 f89, r1098;
mul.f32 f90, f89, 0f3E6E4BAE;
cos.approx.f32 f53, f90;
sin.approx.f32 f91, f90;
neg.f32 f54, f91;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r739, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r739;
mov.b32 r742, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r739;
mov.b32 r744, {high, high};
}
{
mul.f16x2 r746, r718, r744;
}
{
fma.rn.f16x2 r749, r682, r742, r746;
}
{
mul.f16x2 r753, r682, r744;
}
{
neg.f16x2 r756, r753;
}
{
fma.rn.f16x2 r758, r718, r742, r756;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r739;
mov.b32 r762, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r739;
mov.b32 r764, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r766, {low, high};
}
{
mul.f16x2 r767, r764, r766;
}
{
mul.f16x2 r770, r739, r762;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r739;
mov.b32 r773, {high, low};
}
{
fma.rn.f16x2 r775, r767, r773, r770;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r775;
mov.b32 r779, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r775;
mov.b32 r781, {high, high};
}
{
mul.f16x2 r783, r736, r781;
}
{
fma.rn.f16x2 r786, r700, r779, r783;
}
{
mul.f16x2 r790, r700, r781;
}
{
neg.f16x2 r793, r790;
}
{
fma.rn.f16x2 r795, r736, r779, r793;
}
barrier.sync 0;
mad.lo.s32 r1103, r1098, 972, r1102;
st.shared.u32 [r1103], r658;
st.shared.u32 [r1103+324], r749;
st.shared.u32 [r1103+648], r786;
barrier.sync 0;
ld.shared.u32 r822, [r1075];
ld.shared.u32 r819, [r1075+2916];
ld.shared.u32 r820, [r1075+5832];
barrier.sync 0;
st.shared.u32 [r1103], r664;
st.shared.u32 [r1103+324], r758;
st.shared.u32 [r1103+648], r795;
barrier.sync 0;
ld.shared.u32 r828, [r1075];
ld.shared.u32 r825, [r1075+2916];
ld.shared.u32 r826, [r1075+5832];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r816, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r817, {low, high};
}
{
add.f16x2 r818, r819, r820;
}
{
add.f16x2 r821, r822, r818;
}
{
add.f16x2 r824, r825, r826;
}
{
add.f16x2 r827, r828, r824;
}
{
add.f16x2 r830, r819, r820;
}
{
mul.f16x2 r833, r830, r816;
}
{
add.f16x2 r836, r822, r833;
}
{
sub.f16x2 r839, r825, r826;
}
{
mul.f16x2 r842, r839, r817;
}
{
add.f16x2 r845, r836, r842;
}
{
add.f16x2 r848, r819, r820;
}
{
mul.f16x2 r851, r848, r816;
}
{
add.f16x2 r854, r822, r851;
}
{
sub.f16x2 r857, r825, r826;
}
{
mul.f16x2 r860, r857, r817;
}
{
sub.f16x2 r863, r854, r860;
}
{
add.f16x2 r866, r825, r826;
}
{
mul.f16x2 r869, r866, r816;
}
{
add.f16x2 r872, r828, r869;
}
{
sub.f16x2 r875, r819, r820;
}
{
mul.f16x2 r878, r875, r817;
}
{
sub.f16x2 r881, r872, r878;
}
{
add.f16x2 r884, r825, r826;
}
{
mul.f16x2 r887, r884, r816;
}
{
add.f16x2 r890, r828, r887;
}
{
sub.f16x2 r893, r819, r820;
}
{
mul.f16x2 r896, r893, r817;
}
{
add.f16x2 r899, r890, r896;
}
mul.wide.u32 rd12, r1071, -2032597691;
shr.u64 rd13, rd12, 39;
cvt.u32.u64 r1104, rd13;
mul.lo.s32 r1105, r1104, 243;
sub.s32 r1106, r1071, r1105;
shl.b32 r1107, r1106, 2;
add.s32 r1108, r1072, r1107;
cvt.rn.f32.u32 f92, r1104;
mul.f32 f93, f92, 0f3F32B8C2;
cos.approx.f32 f65, f93;
sin.approx.f32 f94, f93;
neg.f32 f66, f94;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f65;
cvt.rn.f16.f32 high, f66;
mov.b32 r902, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r902;
mov.b32 r905, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r902;
mov.b32 r907, {high, high};
}
{
mul.f16x2 r909, r881, r907;
}
{
fma.rn.f16x2 r912, r845, r905, r909;
}
{
mul.f16x2 r916, r845, r907;
}
{
neg.f16x2 r919, r916;
}
{
fma.rn.f16x2 r921, r881, r905, r919;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r902;
mov.b32 r925, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r902;
mov.b32 r927, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r929, {low, high};
}
{
mul.f16x2 r930, r927, r929;
}
{
mul.f16x2 r933, r902, r925;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r902;
mov.b32 r936, {high, low};
}
{
fma.rn.f16x2 r938, r930, r936, r933;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r938;
mov.b32 r942, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r938;
mov.b32 r944, {high, high};
}
{
mul.f16x2 r946, r899, r944;
}
{
fma.rn.f16x2 r949, r863, r942, r946;
}
{
mul.f16x2 r953, r863, r944;
}
{
neg.f16x2 r956, r953;
}
{
fma.rn.f16x2 r958, r899, r942, r956;
}
barrier.sync 0;
mad.lo.s32 r1109, r1104, 2916, r1108;
st.shared.u32 [r1109], r821;
st.shared.u32 [r1109+972], r912;
st.shared.u32 [r1109+1944], r949;
barrier.sync 0;
ld.shared.u32 r985, [r1075];
ld.shared.u32 r982, [r1075+2916];
ld.shared.u32 r983, [r1075+5832];
barrier.sync 0;
st.shared.u32 [r1109], r827;
st.shared.u32 [r1109+972], r921;
st.shared.u32 [r1109+1944], r958;
barrier.sync 0;
ld.shared.u32 r991, [r1075];
ld.shared.u32 r988, [r1075+2916];
ld.shared.u32 r989, [r1075+5832];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r979, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r980, {low, high};
}
{
add.f16x2 r981, r982, r983;
}
{
add.f16x2 %0, r985, r981;
}
{
add.f16x2 r987, r988, r989;
}
{
add.f16x2 %1, r991, r987;
}
{
add.f16x2 r993, r982, r983;
}
{
mul.f16x2 r996, r993, r979;
}
{
add.f16x2 r999, r985, r996;
}
{
sub.f16x2 r1002, r988, r989;
}
{
mul.f16x2 r1005, r1002, r980;
}
{
add.f16x2 %2, r999, r1005;
}
{
add.f16x2 r1011, r982, r983;
}
{
mul.f16x2 r1014, r1011, r979;
}
{
add.f16x2 r1017, r985, r1014;
}
{
sub.f16x2 r1020, r988, r989;
}
{
mul.f16x2 r1023, r1020, r980;
}
{
sub.f16x2 %4, r1017, r1023;
}
{
add.f16x2 r1029, r988, r989;
}
{
mul.f16x2 r1032, r1029, r979;
}
{
add.f16x2 r1035, r991, r1032;
}
{
sub.f16x2 r1038, r982, r983;
}
{
mul.f16x2 r1041, r1038, r980;
}
{
sub.f16x2 %3, r1035, r1041;
}
{
add.f16x2 r1047, r988, r989;
}
{
mul.f16x2 r1050, r1047, r979;
}
{
add.f16x2 r1053, r991, r1050;
}
{
sub.f16x2 r1056, r982, r983;
}
{
mul.f16x2 r1059, r1056, r980;
}
{
add.f16x2 %5, r1053, r1059;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)));
};


#endif
