#ifndef CUFFTDX_FFT_243_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_243_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1080, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<536>;
.reg .b32 r<5689>;
.reg .b64 rd<4>;
mov.u32 r5625, %54;
mov.u32 r5688, %tid.y;
mad.lo.s32 r5626, r5688, 1944, r5625;
mov.u32 r5627, %tid.x;
mov.f32 f530, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r1, {low, high};
}
mov.f32 f532, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r2, {low, high};
}
{
add.f16x2 r3, %106, %91;
}
{
add.f16x2 r6, %61, r3;
}
{
add.f16x2 r9, %70, %102;
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
sub.f16x2 r24, %70, %102;
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
sub.f16x2 r42, %70, %102;
}
{
mul.f16x2 r45, r42, r2;
}
{
sub.f16x2 r48, r39, r45;
}
{
add.f16x2 r51, %70, %102;
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
add.f16x2 r69, %70, %102;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r88, {low, high};
}
{
add.f16x2 r89, %105, %90;
}
{
add.f16x2 r92, %60, r89;
}
{
add.f16x2 r95, %69, %103;
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
sub.f16x2 r110, %69, %103;
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
sub.f16x2 r128, %69, %103;
}
{
mul.f16x2 r131, r128, r88;
}
{
sub.f16x2 r134, r125, r131;
}
{
add.f16x2 r137, %69, %103;
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
add.f16x2 r155, %69, %103;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r173, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r174, {low, high};
}
{
add.f16x2 r175, %104, %89;
}
{
add.f16x2 r178, %59, r175;
}
{
add.f16x2 r181, %67, %101;
}
{
add.f16x2 r184, %74, r181;
}
{
add.f16x2 r187, %104, %89;
}
{
mul.f16x2 r190, r187, r173;
}
{
add.f16x2 r193, %59, r190;
}
{
sub.f16x2 r196, %67, %101;
}
{
mul.f16x2 r199, r196, r174;
}
{
add.f16x2 r202, r193, r199;
}
{
add.f16x2 r205, %104, %89;
}
{
mul.f16x2 r208, r205, r173;
}
{
add.f16x2 r211, %59, r208;
}
{
sub.f16x2 r214, %67, %101;
}
{
mul.f16x2 r217, r214, r174;
}
{
sub.f16x2 r220, r211, r217;
}
{
add.f16x2 r223, %67, %101;
}
{
mul.f16x2 r226, r223, r173;
}
{
add.f16x2 r229, %74, r226;
}
{
sub.f16x2 r232, %104, %89;
}
{
mul.f16x2 r235, r232, r174;
}
{
sub.f16x2 r238, r229, r235;
}
{
add.f16x2 r241, %67, %101;
}
{
mul.f16x2 r244, r241, r173;
}
{
add.f16x2 r247, %74, r244;
}
{
sub.f16x2 r250, %104, %89;
}
{
mul.f16x2 r253, r250, r174;
}
{
add.f16x2 r256, r247, r253;
}
mov.f32 f490, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f490;
mov.b32 r259, {low, high};
}
mov.f32 f492, 0f3F248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f492;
cvt.rn.f16.f32 high, f492;
mov.b32 r260, {low, high};
}
mov.f32 f494, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f494;
cvt.rn.f16.f32 high, f494;
mov.b32 r261, {low, high};
}
mov.f32 f496, 0f3F7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f496;
cvt.rn.f16.f32 high, f496;
mov.b32 r262, {low, high};
}
mov.f32 f502, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f502;
cvt.rn.f16.f32 high, f502;
mov.b32 r265, {low, high};
}
mov.f32 f504, 0f3EAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f504;
cvt.rn.f16.f32 high, f504;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r339, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r425, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r597, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r598, {low, high};
}
{
add.f16x2 r599, %85, %68;
}
{
add.f16x2 r602, %88, r599;
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
add.f16x2 r617, %88, r614;
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
add.f16x2 r635, %88, r632;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r683, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r684, {low, high};
}
{
add.f16x2 r685, %84, %66;
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
add.f16x2 r697, %84, %66;
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
add.f16x2 r715, %84, %66;
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
sub.f16x2 r742, %84, %66;
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
sub.f16x2 r760, %84, %66;
}
{
mul.f16x2 r763, r760, r684;
}
{
add.f16x2 r766, r757, r763;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r769, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r770, {low, high};
}
{
add.f16x2 r771, %82, %65;
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
add.f16x2 r783, %82, %65;
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
add.f16x2 r801, %82, %65;
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
sub.f16x2 r828, %82, %65;
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
sub.f16x2 r846, %82, %65;
}
{
mul.f16x2 r849, r846, r770;
}
{
add.f16x2 r852, r843, r849;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f490;
mov.b32 r855, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f492;
cvt.rn.f16.f32 high, f492;
mov.b32 r856, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f494;
cvt.rn.f16.f32 high, f494;
mov.b32 r857, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f496;
cvt.rn.f16.f32 high, f496;
mov.b32 r858, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f502;
cvt.rn.f16.f32 high, f502;
mov.b32 r861, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f504;
cvt.rn.f16.f32 high, f504;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r935, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r1021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r1107, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r1193, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r1194, {low, high};
}
{
add.f16x2 r1195, %58, %95;
}
{
add.f16x2 r1198, %64, r1195;
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
add.f16x2 r1213, %64, r1210;
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
add.f16x2 r1231, %64, r1228;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r1279, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r1365, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f490;
mov.b32 r1451, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f492;
cvt.rn.f16.f32 high, f492;
mov.b32 r1452, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f494;
cvt.rn.f16.f32 high, f494;
mov.b32 r1453, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f496;
cvt.rn.f16.f32 high, f496;
mov.b32 r1454, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f502;
cvt.rn.f16.f32 high, f502;
mov.b32 r1457, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f504;
cvt.rn.f16.f32 high, f504;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r1531, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r1617, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r1703, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f490;
mov.b32 r1793, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f492;
cvt.rn.f16.f32 high, f492;
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
cvt.rn.f16.f32 low, f494;
cvt.rn.f16.f32 high, f494;
mov.b32 r1799, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f496;
cvt.rn.f16.f32 high, f496;
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
cvt.rn.f16.f32 low, f502;
cvt.rn.f16.f32 high, f502;
mov.b32 r1811, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f504;
cvt.rn.f16.f32 high, f504;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r2097, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r2183, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r2269, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r2355, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r2441, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r2527, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r2613, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r2699, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r2785, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
mul.wide.u32 rd2, r5627, 954437177;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r5628, rd3;
mul.lo.s32 r5629, r5628, 9;
sub.s32 r5630, r5627, r5629;
cvt.rn.f32.u32 f533, r5630;
mul.f32 f534, f533, 0f3CD3D17E;
cos.approx.f32 f309, f534;
sin.approx.f32 f535, f534;
neg.f32 f310, f535;
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
mad.lo.s32 r5631, r5628, 1944, r5626;
barrier.sync 0;
mad.lo.s32 r5632, r5630, 216, r5631;
st.shared.v2.f32 [r5632], {r2102, r2108};
st.shared.v2.f32 [r5632+8], {r2881, r2890};
st.shared.v2.f32 [r5632+16], {r2918, r2927};
st.shared.v2.f32 [r5632+24], {r2955, r2964};
st.shared.v2.f32 [r5632+32], {r2992, r3001};
st.shared.v2.f32 [r5632+40], {r3029, r3038};
st.shared.v2.f32 [r5632+48], {r3066, r3075};
st.shared.v2.f32 [r5632+56], {r3103, r3112};
st.shared.v2.f32 [r5632+64], {r3140, r3149};
st.shared.v2.f32 [r5632+72], {r3177, r3186};
st.shared.v2.f32 [r5632+80], {r3214, r3223};
st.shared.v2.f32 [r5632+88], {r3251, r3260};
st.shared.v2.f32 [r5632+96], {r3288, r3297};
st.shared.v2.f32 [r5632+104], {r3325, r3334};
st.shared.v2.f32 [r5632+112], {r3362, r3371};
st.shared.v2.f32 [r5632+120], {r3399, r3408};
st.shared.v2.f32 [r5632+128], {r3436, r3445};
st.shared.v2.f32 [r5632+136], {r3473, r3482};
st.shared.v2.f32 [r5632+144], {r3510, r3519};
st.shared.v2.f32 [r5632+152], {r3547, r3556};
st.shared.v2.f32 [r5632+160], {r3584, r3593};
st.shared.v2.f32 [r5632+168], {r3621, r3630};
st.shared.v2.f32 [r5632+176], {r3658, r3667};
st.shared.v2.f32 [r5632+184], {r3695, r3704};
st.shared.v2.f32 [r5632+192], {r3732, r3741};
st.shared.v2.f32 [r5632+200], {r3769, r3778};
st.shared.v2.f32 [r5632+208], {r3806, r3815};
barrier.sync 0;
mad.lo.s32 r5633, r5630, -208, r5632;
ld.shared.u32 r3842, [r5633];
ld.shared.u32 r3848, [r5633+4];
ld.shared.u32 r4438, [r5633+72];
ld.shared.u32 r4444, [r5633+76];
ld.shared.u32 r5034, [r5633+144];
ld.shared.u32 r5040, [r5633+148];
ld.shared.u32 r3928, [r5633+216];
ld.shared.u32 r3934, [r5633+220];
ld.shared.u32 r4524, [r5633+288];
ld.shared.u32 r4530, [r5633+292];
ld.shared.u32 r5120, [r5633+360];
ld.shared.u32 r5126, [r5633+364];
ld.shared.u32 r4014, [r5633+432];
ld.shared.u32 r4020, [r5633+436];
ld.shared.u32 r4610, [r5633+504];
ld.shared.u32 r4616, [r5633+508];
ld.shared.u32 r5206, [r5633+576];
ld.shared.u32 r5212, [r5633+580];
ld.shared.u32 r3839, [r5633+648];
ld.shared.u32 r3845, [r5633+652];
ld.shared.u32 r4435, [r5633+720];
ld.shared.u32 r4441, [r5633+724];
ld.shared.u32 r5031, [r5633+792];
ld.shared.u32 r5037, [r5633+796];
ld.shared.u32 r3925, [r5633+864];
ld.shared.u32 r3931, [r5633+868];
ld.shared.u32 r4521, [r5633+936];
ld.shared.u32 r4527, [r5633+940];
ld.shared.u32 r5117, [r5633+1008];
ld.shared.u32 r5123, [r5633+1012];
ld.shared.u32 r4011, [r5633+1080];
ld.shared.u32 r4017, [r5633+1084];
ld.shared.u32 r4607, [r5633+1152];
ld.shared.u32 r4613, [r5633+1156];
ld.shared.u32 r5203, [r5633+1224];
ld.shared.u32 r5209, [r5633+1228];
ld.shared.u32 r3840, [r5633+1296];
ld.shared.u32 r3846, [r5633+1300];
ld.shared.u32 r4436, [r5633+1368];
ld.shared.u32 r4442, [r5633+1372];
ld.shared.u32 r5032, [r5633+1440];
ld.shared.u32 r5038, [r5633+1444];
ld.shared.u32 r3926, [r5633+1512];
ld.shared.u32 r3932, [r5633+1516];
ld.shared.u32 r4522, [r5633+1584];
ld.shared.u32 r4528, [r5633+1588];
ld.shared.u32 r5118, [r5633+1656];
ld.shared.u32 r5124, [r5633+1660];
ld.shared.u32 r4012, [r5633+1728];
ld.shared.u32 r4018, [r5633+1732];
ld.shared.u32 r4608, [r5633+1800];
ld.shared.u32 r4614, [r5633+1804];
ld.shared.u32 r5204, [r5633+1872];
ld.shared.u32 r5210, [r5633+1876];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r3836, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r3922, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r4008, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f490;
mov.b32 r4094, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f492;
cvt.rn.f16.f32 high, f492;
mov.b32 r4095, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f494;
cvt.rn.f16.f32 high, f494;
mov.b32 r4096, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f496;
cvt.rn.f16.f32 high, f496;
mov.b32 r4097, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f502;
cvt.rn.f16.f32 high, f502;
mov.b32 r4100, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f504;
cvt.rn.f16.f32 high, f504;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r4174, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r4175, {low, high};
}
{
add.f16x2 r4176, r3927, r4013;
}
{
add.f16x2 %0, r3841, r4176;
}
{
add.f16x2 r4182, r3933, r4019;
}
{
add.f16x2 %1, r3847, r4182;
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
add.f16x2 %18, r4194, r4200;
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
sub.f16x2 %36, r4212, r4218;
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
sub.f16x2 %19, r4230, r4236;
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
add.f16x2 %37, r4248, r4254;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r4260, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r4261, {low, high};
}
{
add.f16x2 r4262, r4116, r4132;
}
{
add.f16x2 %6, r3865, r4262;
}
{
add.f16x2 r4268, r4122, r4138;
}
{
add.f16x2 %7, r3901, r4268;
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
add.f16x2 %24, r4280, r4286;
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
sub.f16x2 %42, r4298, r4304;
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
sub.f16x2 %25, r4316, r4322;
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
add.f16x2 %43, r4334, r4340;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r4346, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r4347, {low, high};
}
{
add.f16x2 r4348, r4148, r4164;
}
{
add.f16x2 %12, r3883, r4348;
}
{
add.f16x2 r4354, r4154, r4170;
}
{
add.f16x2 %13, r3919, r4354;
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
add.f16x2 %30, r4366, r4372;
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
sub.f16x2 %48, r4384, r4390;
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
sub.f16x2 %31, r4402, r4408;
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
add.f16x2 %49, r4420, r4426;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r4432, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r4518, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r4604, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f490;
mov.b32 r4690, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f492;
cvt.rn.f16.f32 high, f492;
mov.b32 r4691, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f494;
cvt.rn.f16.f32 high, f494;
mov.b32 r4692, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f496;
cvt.rn.f16.f32 high, f496;
mov.b32 r4693, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f502;
cvt.rn.f16.f32 high, f502;
mov.b32 r4696, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f504;
cvt.rn.f16.f32 high, f504;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r4770, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r4771, {low, high};
}
{
add.f16x2 r4772, r4523, r4609;
}
{
add.f16x2 %2, r4437, r4772;
}
{
add.f16x2 r4778, r4529, r4615;
}
{
add.f16x2 %3, r4443, r4778;
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
add.f16x2 %20, r4790, r4796;
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
sub.f16x2 %38, r4808, r4814;
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
sub.f16x2 %21, r4826, r4832;
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
add.f16x2 %39, r4844, r4850;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r4856, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r4857, {low, high};
}
{
add.f16x2 r4858, r4712, r4728;
}
{
add.f16x2 %8, r4461, r4858;
}
{
add.f16x2 r4864, r4718, r4734;
}
{
add.f16x2 %9, r4497, r4864;
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
add.f16x2 %26, r4876, r4882;
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
sub.f16x2 %44, r4894, r4900;
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
sub.f16x2 %27, r4912, r4918;
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
add.f16x2 %45, r4930, r4936;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r4942, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r4943, {low, high};
}
{
add.f16x2 r4944, r4744, r4760;
}
{
add.f16x2 %14, r4479, r4944;
}
{
add.f16x2 r4950, r4750, r4766;
}
{
add.f16x2 %15, r4515, r4950;
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
add.f16x2 %32, r4962, r4968;
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
sub.f16x2 %50, r4980, r4986;
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
sub.f16x2 %33, r4998, r5004;
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
add.f16x2 %51, r5016, r5022;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r5028, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r5114, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r5200, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f490;
mov.b32 r5286, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f492;
cvt.rn.f16.f32 high, f492;
mov.b32 r5287, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f494;
cvt.rn.f16.f32 high, f494;
mov.b32 r5288, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f496;
cvt.rn.f16.f32 high, f496;
mov.b32 r5289, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f502;
cvt.rn.f16.f32 high, f502;
mov.b32 r5292, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f504;
cvt.rn.f16.f32 high, f504;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r5366, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r5367, {low, high};
}
{
add.f16x2 r5368, r5119, r5205;
}
{
add.f16x2 %4, r5033, r5368;
}
{
add.f16x2 r5374, r5125, r5211;
}
{
add.f16x2 %5, r5039, r5374;
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
add.f16x2 %22, r5386, r5392;
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
sub.f16x2 %40, r5404, r5410;
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
sub.f16x2 %23, r5422, r5428;
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
add.f16x2 %41, r5440, r5446;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r5452, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r5453, {low, high};
}
{
add.f16x2 r5454, r5308, r5324;
}
{
add.f16x2 %10, r5057, r5454;
}
{
add.f16x2 r5460, r5314, r5330;
}
{
add.f16x2 %11, r5093, r5460;
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
add.f16x2 %28, r5472, r5478;
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
sub.f16x2 %46, r5490, r5496;
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
sub.f16x2 %29, r5508, r5514;
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
add.f16x2 %47, r5526, r5532;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r5538, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r5539, {low, high};
}
{
add.f16x2 r5540, r5340, r5356;
}
{
add.f16x2 %16, r5075, r5540;
}
{
add.f16x2 r5546, r5346, r5362;
}
{
add.f16x2 %17, r5111, r5546;
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
add.f16x2 %34, r5558, r5564;
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
sub.f16x2 %52, r5576, r5582;
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
sub.f16x2 %35, r5594, r5600;
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
add.f16x2 %53, r5612, r5618;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[23].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1081, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<536>;
.reg .b32 r<5689>;
.reg .b64 rd<4>;
mov.u32 r5625, %54;
mov.u32 r5688, %tid.y;
mad.lo.s32 r5626, r5688, 972, r5625;
mov.u32 r5627, %tid.x;
mov.f32 f530, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r1, {low, high};
}
mov.f32 f532, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r2, {low, high};
}
{
add.f16x2 r3, %106, %91;
}
{
add.f16x2 r6, %61, r3;
}
{
add.f16x2 r9, %70, %102;
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
sub.f16x2 r24, %70, %102;
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
sub.f16x2 r42, %70, %102;
}
{
mul.f16x2 r45, r42, r2;
}
{
sub.f16x2 r48, r39, r45;
}
{
add.f16x2 r51, %70, %102;
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
add.f16x2 r69, %70, %102;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r88, {low, high};
}
{
add.f16x2 r89, %105, %90;
}
{
add.f16x2 r92, %60, r89;
}
{
add.f16x2 r95, %69, %103;
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
sub.f16x2 r110, %69, %103;
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
sub.f16x2 r128, %69, %103;
}
{
mul.f16x2 r131, r128, r88;
}
{
sub.f16x2 r134, r125, r131;
}
{
add.f16x2 r137, %69, %103;
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
add.f16x2 r155, %69, %103;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r173, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r174, {low, high};
}
{
add.f16x2 r175, %104, %89;
}
{
add.f16x2 r178, %59, r175;
}
{
add.f16x2 r181, %67, %101;
}
{
add.f16x2 r184, %74, r181;
}
{
add.f16x2 r187, %104, %89;
}
{
mul.f16x2 r190, r187, r173;
}
{
add.f16x2 r193, %59, r190;
}
{
sub.f16x2 r196, %67, %101;
}
{
mul.f16x2 r199, r196, r174;
}
{
add.f16x2 r202, r193, r199;
}
{
add.f16x2 r205, %104, %89;
}
{
mul.f16x2 r208, r205, r173;
}
{
add.f16x2 r211, %59, r208;
}
{
sub.f16x2 r214, %67, %101;
}
{
mul.f16x2 r217, r214, r174;
}
{
sub.f16x2 r220, r211, r217;
}
{
add.f16x2 r223, %67, %101;
}
{
mul.f16x2 r226, r223, r173;
}
{
add.f16x2 r229, %74, r226;
}
{
sub.f16x2 r232, %104, %89;
}
{
mul.f16x2 r235, r232, r174;
}
{
sub.f16x2 r238, r229, r235;
}
{
add.f16x2 r241, %67, %101;
}
{
mul.f16x2 r244, r241, r173;
}
{
add.f16x2 r247, %74, r244;
}
{
sub.f16x2 r250, %104, %89;
}
{
mul.f16x2 r253, r250, r174;
}
{
add.f16x2 r256, r247, r253;
}
mov.f32 f490, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f490;
mov.b32 r259, {low, high};
}
mov.f32 f492, 0f3F248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f492;
cvt.rn.f16.f32 high, f492;
mov.b32 r260, {low, high};
}
mov.f32 f494, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f494;
cvt.rn.f16.f32 high, f494;
mov.b32 r261, {low, high};
}
mov.f32 f496, 0f3F7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f496;
cvt.rn.f16.f32 high, f496;
mov.b32 r262, {low, high};
}
mov.f32 f502, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f502;
cvt.rn.f16.f32 high, f502;
mov.b32 r265, {low, high};
}
mov.f32 f504, 0f3EAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f504;
cvt.rn.f16.f32 high, f504;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r339, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r425, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r597, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r598, {low, high};
}
{
add.f16x2 r599, %85, %68;
}
{
add.f16x2 r602, %88, r599;
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
add.f16x2 r617, %88, r614;
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
add.f16x2 r635, %88, r632;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r683, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r684, {low, high};
}
{
add.f16x2 r685, %84, %66;
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
add.f16x2 r697, %84, %66;
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
add.f16x2 r715, %84, %66;
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
sub.f16x2 r742, %84, %66;
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
sub.f16x2 r760, %84, %66;
}
{
mul.f16x2 r763, r760, r684;
}
{
add.f16x2 r766, r757, r763;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r769, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r770, {low, high};
}
{
add.f16x2 r771, %82, %65;
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
add.f16x2 r783, %82, %65;
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
add.f16x2 r801, %82, %65;
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
sub.f16x2 r828, %82, %65;
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
sub.f16x2 r846, %82, %65;
}
{
mul.f16x2 r849, r846, r770;
}
{
add.f16x2 r852, r843, r849;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f490;
mov.b32 r855, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f492;
cvt.rn.f16.f32 high, f492;
mov.b32 r856, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f494;
cvt.rn.f16.f32 high, f494;
mov.b32 r857, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f496;
cvt.rn.f16.f32 high, f496;
mov.b32 r858, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f502;
cvt.rn.f16.f32 high, f502;
mov.b32 r861, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f504;
cvt.rn.f16.f32 high, f504;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r935, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r1021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r1107, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r1193, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r1194, {low, high};
}
{
add.f16x2 r1195, %58, %95;
}
{
add.f16x2 r1198, %64, r1195;
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
add.f16x2 r1213, %64, r1210;
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
add.f16x2 r1231, %64, r1228;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r1279, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r1365, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f490;
mov.b32 r1451, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f492;
cvt.rn.f16.f32 high, f492;
mov.b32 r1452, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f494;
cvt.rn.f16.f32 high, f494;
mov.b32 r1453, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f496;
cvt.rn.f16.f32 high, f496;
mov.b32 r1454, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f502;
cvt.rn.f16.f32 high, f502;
mov.b32 r1457, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f504;
cvt.rn.f16.f32 high, f504;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r1531, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r1617, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r1703, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f490;
mov.b32 r1793, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f492;
cvt.rn.f16.f32 high, f492;
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
cvt.rn.f16.f32 low, f494;
cvt.rn.f16.f32 high, f494;
mov.b32 r1799, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f496;
cvt.rn.f16.f32 high, f496;
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
cvt.rn.f16.f32 low, f502;
cvt.rn.f16.f32 high, f502;
mov.b32 r1811, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f504;
cvt.rn.f16.f32 high, f504;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r2097, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r2183, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r2269, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r2355, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r2441, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r2527, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r2613, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r2699, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r2785, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
mul.wide.u32 rd2, r5627, 954437177;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r5628, rd3;
mul.lo.s32 r5629, r5628, 9;
sub.s32 r5630, r5627, r5629;
mad.lo.s32 r5631, r5628, 972, r5626;
cvt.rn.f32.u32 f533, r5630;
mul.f32 f534, f533, 0f3CD3D17E;
cos.approx.f32 f309, f534;
sin.approx.f32 f535, f534;
neg.f32 f310, f535;
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
mad.lo.s32 r5632, r5630, 108, r5631;
st.shared.u32 [r5632], r2102;
st.shared.u32 [r5632+4], r2881;
st.shared.u32 [r5632+8], r2918;
st.shared.u32 [r5632+12], r2955;
st.shared.u32 [r5632+16], r2992;
st.shared.u32 [r5632+20], r3029;
st.shared.u32 [r5632+24], r3066;
st.shared.u32 [r5632+28], r3103;
st.shared.u32 [r5632+32], r3140;
st.shared.u32 [r5632+36], r3177;
st.shared.u32 [r5632+40], r3214;
st.shared.u32 [r5632+44], r3251;
st.shared.u32 [r5632+48], r3288;
st.shared.u32 [r5632+52], r3325;
st.shared.u32 [r5632+56], r3362;
st.shared.u32 [r5632+60], r3399;
st.shared.u32 [r5632+64], r3436;
st.shared.u32 [r5632+68], r3473;
st.shared.u32 [r5632+72], r3510;
st.shared.u32 [r5632+76], r3547;
st.shared.u32 [r5632+80], r3584;
st.shared.u32 [r5632+84], r3621;
st.shared.u32 [r5632+88], r3658;
st.shared.u32 [r5632+92], r3695;
st.shared.u32 [r5632+96], r3732;
st.shared.u32 [r5632+100], r3769;
st.shared.u32 [r5632+104], r3806;
barrier.sync 0;
mad.lo.s32 r5633, r5630, -104, r5632;
ld.shared.u32 r3842, [r5633];
ld.shared.u32 r4438, [r5633+36];
ld.shared.u32 r5034, [r5633+72];
ld.shared.u32 r3928, [r5633+108];
ld.shared.u32 r4524, [r5633+144];
ld.shared.u32 r5120, [r5633+180];
ld.shared.u32 r4014, [r5633+216];
ld.shared.u32 r4610, [r5633+252];
ld.shared.u32 r5206, [r5633+288];
ld.shared.u32 r3839, [r5633+324];
ld.shared.u32 r4435, [r5633+360];
ld.shared.u32 r5031, [r5633+396];
ld.shared.u32 r3925, [r5633+432];
ld.shared.u32 r4521, [r5633+468];
ld.shared.u32 r5117, [r5633+504];
ld.shared.u32 r4011, [r5633+540];
ld.shared.u32 r4607, [r5633+576];
ld.shared.u32 r5203, [r5633+612];
ld.shared.u32 r3840, [r5633+648];
ld.shared.u32 r4436, [r5633+684];
ld.shared.u32 r5032, [r5633+720];
ld.shared.u32 r3926, [r5633+756];
ld.shared.u32 r4522, [r5633+792];
ld.shared.u32 r5118, [r5633+828];
ld.shared.u32 r4012, [r5633+864];
ld.shared.u32 r4608, [r5633+900];
ld.shared.u32 r5204, [r5633+936];
barrier.sync 0;
st.shared.u32 [r5632], r2108;
st.shared.u32 [r5632+4], r2890;
st.shared.u32 [r5632+8], r2927;
st.shared.u32 [r5632+12], r2964;
st.shared.u32 [r5632+16], r3001;
st.shared.u32 [r5632+20], r3038;
st.shared.u32 [r5632+24], r3075;
st.shared.u32 [r5632+28], r3112;
st.shared.u32 [r5632+32], r3149;
st.shared.u32 [r5632+36], r3186;
st.shared.u32 [r5632+40], r3223;
st.shared.u32 [r5632+44], r3260;
st.shared.u32 [r5632+48], r3297;
st.shared.u32 [r5632+52], r3334;
st.shared.u32 [r5632+56], r3371;
st.shared.u32 [r5632+60], r3408;
st.shared.u32 [r5632+64], r3445;
st.shared.u32 [r5632+68], r3482;
st.shared.u32 [r5632+72], r3519;
st.shared.u32 [r5632+76], r3556;
st.shared.u32 [r5632+80], r3593;
st.shared.u32 [r5632+84], r3630;
st.shared.u32 [r5632+88], r3667;
st.shared.u32 [r5632+92], r3704;
st.shared.u32 [r5632+96], r3741;
st.shared.u32 [r5632+100], r3778;
st.shared.u32 [r5632+104], r3815;
barrier.sync 0;
ld.shared.u32 r3848, [r5633];
ld.shared.u32 r4444, [r5633+36];
ld.shared.u32 r5040, [r5633+72];
ld.shared.u32 r3934, [r5633+108];
ld.shared.u32 r4530, [r5633+144];
ld.shared.u32 r5126, [r5633+180];
ld.shared.u32 r4020, [r5633+216];
ld.shared.u32 r4616, [r5633+252];
ld.shared.u32 r5212, [r5633+288];
ld.shared.u32 r3845, [r5633+324];
ld.shared.u32 r4441, [r5633+360];
ld.shared.u32 r5037, [r5633+396];
ld.shared.u32 r3931, [r5633+432];
ld.shared.u32 r4527, [r5633+468];
ld.shared.u32 r5123, [r5633+504];
ld.shared.u32 r4017, [r5633+540];
ld.shared.u32 r4613, [r5633+576];
ld.shared.u32 r5209, [r5633+612];
ld.shared.u32 r3846, [r5633+648];
ld.shared.u32 r4442, [r5633+684];
ld.shared.u32 r5038, [r5633+720];
ld.shared.u32 r3932, [r5633+756];
ld.shared.u32 r4528, [r5633+792];
ld.shared.u32 r5124, [r5633+828];
ld.shared.u32 r4018, [r5633+864];
ld.shared.u32 r4614, [r5633+900];
ld.shared.u32 r5210, [r5633+936];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r3836, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r3922, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r4008, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f490;
mov.b32 r4094, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f492;
cvt.rn.f16.f32 high, f492;
mov.b32 r4095, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f494;
cvt.rn.f16.f32 high, f494;
mov.b32 r4096, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f496;
cvt.rn.f16.f32 high, f496;
mov.b32 r4097, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f502;
cvt.rn.f16.f32 high, f502;
mov.b32 r4100, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f504;
cvt.rn.f16.f32 high, f504;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r4174, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r4175, {low, high};
}
{
add.f16x2 r4176, r3927, r4013;
}
{
add.f16x2 %0, r3841, r4176;
}
{
add.f16x2 r4182, r3933, r4019;
}
{
add.f16x2 %1, r3847, r4182;
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
add.f16x2 %18, r4194, r4200;
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
sub.f16x2 %36, r4212, r4218;
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
sub.f16x2 %19, r4230, r4236;
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
add.f16x2 %37, r4248, r4254;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r4260, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r4261, {low, high};
}
{
add.f16x2 r4262, r4116, r4132;
}
{
add.f16x2 %6, r3865, r4262;
}
{
add.f16x2 r4268, r4122, r4138;
}
{
add.f16x2 %7, r3901, r4268;
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
add.f16x2 %24, r4280, r4286;
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
sub.f16x2 %42, r4298, r4304;
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
sub.f16x2 %25, r4316, r4322;
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
add.f16x2 %43, r4334, r4340;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r4346, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r4347, {low, high};
}
{
add.f16x2 r4348, r4148, r4164;
}
{
add.f16x2 %12, r3883, r4348;
}
{
add.f16x2 r4354, r4154, r4170;
}
{
add.f16x2 %13, r3919, r4354;
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
add.f16x2 %30, r4366, r4372;
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
sub.f16x2 %48, r4384, r4390;
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
sub.f16x2 %31, r4402, r4408;
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
add.f16x2 %49, r4420, r4426;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r4432, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r4518, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r4604, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f490;
mov.b32 r4690, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f492;
cvt.rn.f16.f32 high, f492;
mov.b32 r4691, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f494;
cvt.rn.f16.f32 high, f494;
mov.b32 r4692, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f496;
cvt.rn.f16.f32 high, f496;
mov.b32 r4693, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f502;
cvt.rn.f16.f32 high, f502;
mov.b32 r4696, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f504;
cvt.rn.f16.f32 high, f504;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r4770, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r4771, {low, high};
}
{
add.f16x2 r4772, r4523, r4609;
}
{
add.f16x2 %2, r4437, r4772;
}
{
add.f16x2 r4778, r4529, r4615;
}
{
add.f16x2 %3, r4443, r4778;
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
add.f16x2 %20, r4790, r4796;
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
sub.f16x2 %38, r4808, r4814;
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
sub.f16x2 %21, r4826, r4832;
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
add.f16x2 %39, r4844, r4850;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r4856, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r4857, {low, high};
}
{
add.f16x2 r4858, r4712, r4728;
}
{
add.f16x2 %8, r4461, r4858;
}
{
add.f16x2 r4864, r4718, r4734;
}
{
add.f16x2 %9, r4497, r4864;
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
add.f16x2 %26, r4876, r4882;
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
sub.f16x2 %44, r4894, r4900;
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
sub.f16x2 %27, r4912, r4918;
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
add.f16x2 %45, r4930, r4936;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r4942, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r4943, {low, high};
}
{
add.f16x2 r4944, r4744, r4760;
}
{
add.f16x2 %14, r4479, r4944;
}
{
add.f16x2 r4950, r4750, r4766;
}
{
add.f16x2 %15, r4515, r4950;
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
add.f16x2 %32, r4962, r4968;
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
sub.f16x2 %50, r4980, r4986;
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
sub.f16x2 %33, r4998, r5004;
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
add.f16x2 %51, r5016, r5022;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r5028, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r5114, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r5200, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
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
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f490;
mov.b32 r5286, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f492;
cvt.rn.f16.f32 high, f492;
mov.b32 r5287, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f494;
cvt.rn.f16.f32 high, f494;
mov.b32 r5288, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f496;
cvt.rn.f16.f32 high, f496;
mov.b32 r5289, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f502;
cvt.rn.f16.f32 high, f502;
mov.b32 r5292, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f504;
cvt.rn.f16.f32 high, f504;
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
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r5366, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r5367, {low, high};
}
{
add.f16x2 r5368, r5119, r5205;
}
{
add.f16x2 %4, r5033, r5368;
}
{
add.f16x2 r5374, r5125, r5211;
}
{
add.f16x2 %5, r5039, r5374;
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
add.f16x2 %22, r5386, r5392;
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
sub.f16x2 %40, r5404, r5410;
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
sub.f16x2 %23, r5422, r5428;
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
add.f16x2 %41, r5440, r5446;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r5452, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r5453, {low, high};
}
{
add.f16x2 r5454, r5308, r5324;
}
{
add.f16x2 %10, r5057, r5454;
}
{
add.f16x2 r5460, r5314, r5330;
}
{
add.f16x2 %11, r5093, r5460;
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
add.f16x2 %28, r5472, r5478;
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
sub.f16x2 %46, r5490, r5496;
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
sub.f16x2 %29, r5508, r5514;
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
add.f16x2 %47, r5526, r5532;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f530;
cvt.rn.f16.f32 high, f530;
mov.b32 r5538, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f532;
cvt.rn.f16.f32 high, f532;
mov.b32 r5539, {low, high};
}
{
add.f16x2 r5540, r5340, r5356;
}
{
add.f16x2 %16, r5075, r5540;
}
{
add.f16x2 r5546, r5346, r5362;
}
{
add.f16x2 %17, r5111, r5546;
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
add.f16x2 %34, r5558, r5564;
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
sub.f16x2 %52, r5576, r5582;
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
sub.f16x2 %35, r5594, r5600;
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
add.f16x2 %53, r5612, r5618;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[23].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1082, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<171>;
.reg .b32 r<2070>;
.reg .b64 rd<6>;
mov.u32 r2049, %tid.y;
mov.u32 r2050, %18;
mad.lo.s32 r2051, r2049, 1944, r2050;
mov.u32 r2052, %tid.x;
mov.f32 f162, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r1, {low, high};
}
mov.f32 f164, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r173, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r339, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r425, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
mul.wide.u32 rd2, r2052, 795364315;
shr.u64 rd3, rd2, 32;
cvt.u32.u64 r2053, rd3;
sub.s32 r2054, r2052, r2053;
shr.u32 r2055, r2054, 1;
add.s32 r2056, r2055, r2053;
shr.u32 r2057, r2056, 4;
mul.lo.s32 r2058, r2057, 27;
sub.s32 r2059, r2052, r2058;
cvt.rn.f32.u32 f165, r2059;
mul.f32 f166, f165, 0f3CD3D17E;
cos.approx.f32 f57, f166;
sin.approx.f32 f167, f166;
neg.f32 f58, f167;
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
mov.f32 f149, 0fBF800000;
mov.f32 f150, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
mad.lo.s32 r2060, r2057, 1944, r2051;
barrier.sync 0;
mad.lo.s32 r2061, r2059, 72, r2060;
st.shared.v2.f32 [r2061], {r344, r350};
st.shared.v2.f32 [r2061+8], {r607, r616};
st.shared.v2.f32 [r2061+16], {r644, r653};
st.shared.v2.f32 [r2061+24], {r681, r690};
st.shared.v2.f32 [r2061+32], {r718, r727};
st.shared.v2.f32 [r2061+40], {r755, r764};
st.shared.v2.f32 [r2061+48], {r792, r801};
st.shared.v2.f32 [r2061+56], {r829, r838};
st.shared.v2.f32 [r2061+64], {r866, r875};
barrier.sync 0;
shl.b32 r2062, r2059, 6;
sub.s32 r2063, r2061, r2062;
ld.shared.u32 r902, [r2063];
ld.shared.u32 r908, [r2063+4];
ld.shared.u32 r988, [r2063+216];
ld.shared.u32 r994, [r2063+220];
ld.shared.u32 r1074, [r2063+432];
ld.shared.u32 r1080, [r2063+436];
ld.shared.u32 r899, [r2063+648];
ld.shared.u32 r905, [r2063+652];
ld.shared.u32 r985, [r2063+864];
ld.shared.u32 r991, [r2063+868];
ld.shared.u32 r1071, [r2063+1080];
ld.shared.u32 r1077, [r2063+1084];
ld.shared.u32 r900, [r2063+1296];
ld.shared.u32 r906, [r2063+1300];
ld.shared.u32 r986, [r2063+1512];
ld.shared.u32 r992, [r2063+1516];
ld.shared.u32 r1072, [r2063+1728];
ld.shared.u32 r1078, [r2063+1732];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r896, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r982, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r1068, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r1234, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r1320, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r1406, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
mul.wide.u32 rd4, r2059, 954437177;
shr.u64 rd5, rd4, 33;
cvt.u32.u64 r2064, rd5;
cvt.rn.f32.u32 f168, r2064;
mul.f32 f169, f168, 0f3E6E4BAE;
cos.approx.f32 f133, f169;
sin.approx.f32 f170, f169;
neg.f32 f134, f170;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f133;
cvt.rn.f16.f32 high, f134;
mov.b32 r1492, {low, high};
}
mul.lo.s32 r2065, r2064, 9;
sub.s32 r2066, r2059, r2065;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
shl.b32 r2067, r2066, 3;
add.s32 r2068, r2060, r2067;
barrier.sync 0;
mad.lo.s32 r2069, r2064, 648, r2068;
st.shared.u32 [r2069], r1239;
st.shared.u32 [r2069+4], r1245;
st.shared.u32 [r2069+72], r1502;
st.shared.u32 [r2069+76], r1511;
st.shared.u32 [r2069+144], r1539;
st.shared.u32 [r2069+148], r1548;
st.shared.u32 [r2069+216], r1576;
st.shared.u32 [r2069+220], r1585;
st.shared.u32 [r2069+288], r1613;
st.shared.u32 [r2069+292], r1622;
st.shared.u32 [r2069+360], r1650;
st.shared.u32 [r2069+364], r1659;
st.shared.u32 [r2069+432], r1687;
st.shared.u32 [r2069+436], r1696;
st.shared.u32 [r2069+504], r1724;
st.shared.u32 [r2069+508], r1733;
st.shared.u32 [r2069+576], r1761;
st.shared.u32 [r2069+580], r1770;
barrier.sync 0;
ld.shared.u32 r1797, [r2063];
ld.shared.u32 r1803, [r2063+4];
ld.shared.u32 r1883, [r2063+216];
ld.shared.u32 r1889, [r2063+220];
ld.shared.u32 r1969, [r2063+432];
ld.shared.u32 r1975, [r2063+436];
ld.shared.u32 r1794, [r2063+648];
ld.shared.u32 r1800, [r2063+652];
ld.shared.u32 r1880, [r2063+864];
ld.shared.u32 r1886, [r2063+868];
ld.shared.u32 r1966, [r2063+1080];
ld.shared.u32 r1972, [r2063+1084];
ld.shared.u32 r1795, [r2063+1296];
ld.shared.u32 r1801, [r2063+1300];
ld.shared.u32 r1881, [r2063+1512];
ld.shared.u32 r1887, [r2063+1516];
ld.shared.u32 r1967, [r2063+1728];
ld.shared.u32 r1973, [r2063+1732];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r1791, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
mov.b32 r1792, {low, high};
}
{
add.f16x2 r1793, r1794, r1795;
}
{
add.f16x2 %0, r1797, r1793;
}
{
add.f16x2 r1799, r1800, r1801;
}
{
add.f16x2 %1, r1803, r1799;
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
add.f16x2 %6, r1811, r1817;
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
sub.f16x2 %12, r1829, r1835;
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
sub.f16x2 %7, r1847, r1853;
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
add.f16x2 %13, r1865, r1871;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r1877, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
mov.b32 r1878, {low, high};
}
{
add.f16x2 r1879, r1880, r1881;
}
{
add.f16x2 %2, r1883, r1879;
}
{
add.f16x2 r1885, r1886, r1887;
}
{
add.f16x2 %3, r1889, r1885;
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
add.f16x2 %8, r1897, r1903;
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
sub.f16x2 %14, r1915, r1921;
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
sub.f16x2 %9, r1933, r1939;
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
add.f16x2 %15, r1951, r1957;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r1963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
mov.b32 r1964, {low, high};
}
{
add.f16x2 r1965, r1966, r1967;
}
{
add.f16x2 %4, r1969, r1965;
}
{
add.f16x2 r1971, r1972, r1973;
}
{
add.f16x2 %5, r1975, r1971;
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
add.f16x2 %10, r1983, r1989;
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
sub.f16x2 %16, r2001, r2007;
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
sub.f16x2 %11, r2019, r2025;
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
add.f16x2 %17, r2037, r2043;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1083, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<171>;
.reg .b32 r<2070>;
.reg .b64 rd<6>;
mov.u32 r2049, %tid.y;
mov.u32 r2050, %18;
mad.lo.s32 r2051, r2049, 972, r2050;
mov.u32 r2052, %tid.x;
mov.f32 f162, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r1, {low, high};
}
mov.f32 f164, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r173, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r339, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r425, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
mul.wide.u32 rd2, r2052, 795364315;
shr.u64 rd3, rd2, 32;
cvt.u32.u64 r2053, rd3;
sub.s32 r2054, r2052, r2053;
shr.u32 r2055, r2054, 1;
add.s32 r2056, r2055, r2053;
shr.u32 r2057, r2056, 4;
mul.lo.s32 r2058, r2057, 27;
sub.s32 r2059, r2052, r2058;
mad.lo.s32 r2060, r2057, 972, r2051;
cvt.rn.f32.u32 f165, r2059;
mul.f32 f166, f165, 0f3CD3D17E;
cos.approx.f32 f57, f166;
sin.approx.f32 f167, f166;
neg.f32 f58, f167;
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
mov.f32 f149, 0fBF800000;
mov.f32 f150, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
mad.lo.s32 r2061, r2059, 36, r2060;
st.shared.u32 [r2061], r344;
st.shared.u32 [r2061+4], r607;
st.shared.u32 [r2061+8], r644;
st.shared.u32 [r2061+12], r681;
st.shared.u32 [r2061+16], r718;
st.shared.u32 [r2061+20], r755;
st.shared.u32 [r2061+24], r792;
st.shared.u32 [r2061+28], r829;
st.shared.u32 [r2061+32], r866;
barrier.sync 0;
shl.b32 r2062, r2059, 5;
sub.s32 r2063, r2061, r2062;
ld.shared.u32 r902, [r2063];
ld.shared.u32 r988, [r2063+108];
ld.shared.u32 r1074, [r2063+216];
ld.shared.u32 r899, [r2063+324];
ld.shared.u32 r985, [r2063+432];
ld.shared.u32 r1071, [r2063+540];
ld.shared.u32 r900, [r2063+648];
ld.shared.u32 r986, [r2063+756];
ld.shared.u32 r1072, [r2063+864];
barrier.sync 0;
st.shared.u32 [r2061], r350;
st.shared.u32 [r2061+4], r616;
st.shared.u32 [r2061+8], r653;
st.shared.u32 [r2061+12], r690;
st.shared.u32 [r2061+16], r727;
st.shared.u32 [r2061+20], r764;
st.shared.u32 [r2061+24], r801;
st.shared.u32 [r2061+28], r838;
st.shared.u32 [r2061+32], r875;
barrier.sync 0;
ld.shared.u32 r908, [r2063];
ld.shared.u32 r994, [r2063+108];
ld.shared.u32 r1080, [r2063+216];
ld.shared.u32 r905, [r2063+324];
ld.shared.u32 r991, [r2063+432];
ld.shared.u32 r1077, [r2063+540];
ld.shared.u32 r906, [r2063+648];
ld.shared.u32 r992, [r2063+756];
ld.shared.u32 r1078, [r2063+864];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r896, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r982, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r1068, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r1234, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r1320, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r1406, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
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
mul.wide.u32 rd4, r2059, 954437177;
shr.u64 rd5, rd4, 33;
cvt.u32.u64 r2064, rd5;
mul.lo.s32 r2065, r2064, 9;
sub.s32 r2066, r2059, r2065;
shl.b32 r2067, r2066, 2;
add.s32 r2068, r2060, r2067;
cvt.rn.f32.u32 f168, r2064;
mul.f32 f169, f168, 0f3E6E4BAE;
cos.approx.f32 f133, f169;
sin.approx.f32 f170, f169;
neg.f32 f134, f170;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
cvt.rn.f16.f32 low, f149;
cvt.rn.f16.f32 high, f150;
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
mad.lo.s32 r2069, r2064, 324, r2068;
st.shared.u32 [r2069], r1239;
st.shared.u32 [r2069+36], r1502;
st.shared.u32 [r2069+72], r1539;
st.shared.u32 [r2069+108], r1576;
st.shared.u32 [r2069+144], r1613;
st.shared.u32 [r2069+180], r1650;
st.shared.u32 [r2069+216], r1687;
st.shared.u32 [r2069+252], r1724;
st.shared.u32 [r2069+288], r1761;
barrier.sync 0;
ld.shared.u32 r1797, [r2063];
ld.shared.u32 r1883, [r2063+108];
ld.shared.u32 r1969, [r2063+216];
ld.shared.u32 r1794, [r2063+324];
ld.shared.u32 r1880, [r2063+432];
ld.shared.u32 r1966, [r2063+540];
ld.shared.u32 r1795, [r2063+648];
ld.shared.u32 r1881, [r2063+756];
ld.shared.u32 r1967, [r2063+864];
barrier.sync 0;
st.shared.u32 [r2069], r1245;
st.shared.u32 [r2069+36], r1511;
st.shared.u32 [r2069+72], r1548;
st.shared.u32 [r2069+108], r1585;
st.shared.u32 [r2069+144], r1622;
st.shared.u32 [r2069+180], r1659;
st.shared.u32 [r2069+216], r1696;
st.shared.u32 [r2069+252], r1733;
st.shared.u32 [r2069+288], r1770;
barrier.sync 0;
ld.shared.u32 r1803, [r2063];
ld.shared.u32 r1889, [r2063+108];
ld.shared.u32 r1975, [r2063+216];
ld.shared.u32 r1800, [r2063+324];
ld.shared.u32 r1886, [r2063+432];
ld.shared.u32 r1972, [r2063+540];
ld.shared.u32 r1801, [r2063+648];
ld.shared.u32 r1887, [r2063+756];
ld.shared.u32 r1973, [r2063+864];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r1791, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
mov.b32 r1792, {low, high};
}
{
add.f16x2 r1793, r1794, r1795;
}
{
add.f16x2 %0, r1797, r1793;
}
{
add.f16x2 r1799, r1800, r1801;
}
{
add.f16x2 %1, r1803, r1799;
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
add.f16x2 %6, r1811, r1817;
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
sub.f16x2 %12, r1829, r1835;
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
sub.f16x2 %7, r1847, r1853;
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
add.f16x2 %13, r1865, r1871;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r1877, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
mov.b32 r1878, {low, high};
}
{
add.f16x2 r1879, r1880, r1881;
}
{
add.f16x2 %2, r1883, r1879;
}
{
add.f16x2 r1885, r1886, r1887;
}
{
add.f16x2 %3, r1889, r1885;
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
add.f16x2 %8, r1897, r1903;
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
sub.f16x2 %14, r1915, r1921;
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
sub.f16x2 %9, r1933, r1939;
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
add.f16x2 %15, r1951, r1957;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f162;
cvt.rn.f16.f32 high, f162;
mov.b32 r1963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f164;
cvt.rn.f16.f32 high, f164;
mov.b32 r1964, {low, high};
}
{
add.f16x2 r1965, r1966, r1967;
}
{
add.f16x2 %4, r1969, r1965;
}
{
add.f16x2 r1971, r1972, r1973;
}
{
add.f16x2 %5, r1975, r1971;
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
add.f16x2 %10, r1983, r1989;
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
sub.f16x2 %16, r2001, r2007;
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
sub.f16x2 %11, r2019, r2025;
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
add.f16x2 %17, r2037, r2043;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1084, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<65>;
.reg .b32 r<772>;
.reg .b64 rd<10>;
mov.u32 r739, %tid.y;
mov.u32 r740, %6;
mad.lo.s32 r741, r739, 1944, r740;
mov.u32 r742, %tid.x;
mov.f32 f50, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r1, {low, high};
}
mov.f32 f52, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
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
mul.wide.u32 rd2, r742, -901412889;
shr.u64 rd3, rd2, 38;
cvt.u32.u64 r743, rd3;
mul.lo.s32 r744, r743, 81;
sub.s32 r745, r742, r744;
mad.lo.s32 r746, r743, 1944, r741;
cvt.rn.f32.u32 f53, r745;
mul.f32 f54, f53, 0f3CD3D17E;
cos.approx.f32 f5, f54;
sin.approx.f32 f55, f54;
neg.f32 f6, f55;
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
mov.f32 f45, 0fBF800000;
mov.f32 f46, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
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
mad.lo.s32 r747, r745, 24, r746;
st.shared.v2.f32 [r747], {r6, r12};
st.shared.v2.f32 [r747+8], {r97, r106};
st.shared.v2.f32 [r747+16], {r134, r143};
barrier.sync 0;
shl.b32 r748, r745, 4;
sub.s32 r749, r747, r748;
ld.shared.u32 r170, [r749];
ld.shared.u32 r176, [r749+4];
ld.shared.u32 r167, [r749+648];
ld.shared.u32 r173, [r749+652];
ld.shared.u32 r168, [r749+1296];
ld.shared.u32 r174, [r749+1300];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r164, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
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
mul.wide.u32 rd4, r745, -1431655765;
shr.u64 rd5, rd4, 33;
cvt.u32.u64 r750, rd5;
mul.lo.s32 r751, r750, 3;
sub.s32 r752, r745, r751;
shl.b32 r753, r752, 3;
add.s32 r754, r746, r753;
cvt.rn.f32.u32 f56, r750;
mul.f32 f57, f56, 0f3D9EDD1F;
cos.approx.f32 f17, f57;
sin.approx.f32 f58, f57;
neg.f32 f18, f58;
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
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
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
mad.lo.s32 r755, r750, 72, r754;
st.shared.u32 [r755], r169;
st.shared.u32 [r755+4], r175;
st.shared.u32 [r755+24], r260;
st.shared.u32 [r755+28], r269;
st.shared.u32 [r755+48], r297;
st.shared.u32 [r755+52], r306;
barrier.sync 0;
ld.shared.u32 r333, [r749];
ld.shared.u32 r339, [r749+4];
ld.shared.u32 r330, [r749+648];
ld.shared.u32 r336, [r749+652];
ld.shared.u32 r331, [r749+1296];
ld.shared.u32 r337, [r749+1300];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r327, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
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
mul.wide.u32 rd6, r745, 954437177;
shr.u64 rd7, rd6, 33;
cvt.u32.u64 r756, rd7;
mul.lo.s32 r757, r756, 9;
sub.s32 r758, r745, r757;
shl.b32 r759, r758, 3;
add.s32 r760, r746, r759;
cvt.rn.f32.u32 f59, r756;
mul.f32 f60, f59, 0f3E6E4BAE;
cos.approx.f32 f29, f60;
sin.approx.f32 f61, f60;
neg.f32 f30, f61;
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
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
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
mad.lo.s32 r761, r756, 216, r760;
st.shared.u32 [r761], r332;
st.shared.u32 [r761+4], r338;
st.shared.u32 [r761+72], r423;
st.shared.u32 [r761+76], r432;
st.shared.u32 [r761+144], r460;
st.shared.u32 [r761+148], r469;
barrier.sync 0;
ld.shared.u32 r496, [r749];
ld.shared.u32 r502, [r749+4];
ld.shared.u32 r493, [r749+648];
ld.shared.u32 r499, [r749+652];
ld.shared.u32 r494, [r749+1296];
ld.shared.u32 r500, [r749+1300];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r490, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
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
mul.wide.u32 rd8, r745, 795364315;
shr.u64 rd9, rd8, 32;
cvt.u32.u64 r762, rd9;
sub.s32 r763, r745, r762;
shr.u32 r764, r763, 1;
add.s32 r765, r764, r762;
shr.u32 r766, r765, 4;
mul.lo.s32 r767, r766, 27;
sub.s32 r768, r745, r767;
shl.b32 r769, r768, 3;
add.s32 r770, r746, r769;
cvt.rn.f32.u32 f62, r766;
mul.f32 f63, f62, 0f3F32B8C2;
cos.approx.f32 f41, f63;
sin.approx.f32 f64, f63;
neg.f32 f42, f64;
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
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
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
mad.lo.s32 r771, r766, 648, r770;
st.shared.u32 [r771], r495;
st.shared.u32 [r771+4], r501;
st.shared.u32 [r771+216], r586;
st.shared.u32 [r771+220], r595;
st.shared.u32 [r771+432], r623;
st.shared.u32 [r771+436], r632;
barrier.sync 0;
ld.shared.u32 r659, [r749];
ld.shared.u32 r665, [r749+4];
ld.shared.u32 r656, [r749+648];
ld.shared.u32 r662, [r749+652];
ld.shared.u32 r657, [r749+1296];
ld.shared.u32 r663, [r749+1300];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r653, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r654, {low, high};
}
{
add.f16x2 r655, r656, r657;
}
{
add.f16x2 %0, r659, r655;
}
{
add.f16x2 r661, r662, r663;
}
{
add.f16x2 %1, r665, r661;
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
add.f16x2 %2, r673, r679;
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
sub.f16x2 %4, r691, r697;
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
sub.f16x2 %3, r709, r715;
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
add.f16x2 %5, r727, r733;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1085, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<65>;
.reg .b32 r<772>;
.reg .b64 rd<10>;
mov.u32 r739, %tid.y;
mov.u32 r740, %6;
mad.lo.s32 r741, r739, 972, r740;
mov.u32 r742, %tid.x;
mov.f32 f50, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r1, {low, high};
}
mov.f32 f52, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
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
mul.wide.u32 rd2, r742, -901412889;
shr.u64 rd3, rd2, 38;
cvt.u32.u64 r743, rd3;
mul.lo.s32 r744, r743, 81;
sub.s32 r745, r742, r744;
mad.lo.s32 r746, r743, 972, r741;
cvt.rn.f32.u32 f53, r745;
mul.f32 f54, f53, 0f3CD3D17E;
cos.approx.f32 f5, f54;
sin.approx.f32 f55, f54;
neg.f32 f6, f55;
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
mov.f32 f45, 0fBF800000;
mov.f32 f46, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
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
mad.lo.s32 r747, r745, 12, r746;
st.shared.u32 [r747], r6;
st.shared.u32 [r747+4], r97;
st.shared.u32 [r747+8], r134;
barrier.sync 0;
shl.b32 r748, r745, 3;
sub.s32 r749, r747, r748;
ld.shared.u32 r170, [r749];
ld.shared.u32 r167, [r749+324];
ld.shared.u32 r168, [r749+648];
barrier.sync 0;
st.shared.u32 [r747], r12;
st.shared.u32 [r747+4], r106;
st.shared.u32 [r747+8], r143;
barrier.sync 0;
ld.shared.u32 r176, [r749];
ld.shared.u32 r173, [r749+324];
ld.shared.u32 r174, [r749+648];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r164, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
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
mul.wide.u32 rd4, r745, -1431655765;
shr.u64 rd5, rd4, 33;
cvt.u32.u64 r750, rd5;
mul.lo.s32 r751, r750, 3;
sub.s32 r752, r745, r751;
shl.b32 r753, r752, 2;
add.s32 r754, r746, r753;
cvt.rn.f32.u32 f56, r750;
mul.f32 f57, f56, 0f3D9EDD1F;
cos.approx.f32 f17, f57;
sin.approx.f32 f58, f57;
neg.f32 f18, f58;
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
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
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
mad.lo.s32 r755, r750, 36, r754;
st.shared.u32 [r755], r169;
st.shared.u32 [r755+12], r260;
st.shared.u32 [r755+24], r297;
barrier.sync 0;
ld.shared.u32 r333, [r749];
ld.shared.u32 r330, [r749+324];
ld.shared.u32 r331, [r749+648];
barrier.sync 0;
st.shared.u32 [r755], r175;
st.shared.u32 [r755+12], r269;
st.shared.u32 [r755+24], r306;
barrier.sync 0;
ld.shared.u32 r339, [r749];
ld.shared.u32 r336, [r749+324];
ld.shared.u32 r337, [r749+648];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r327, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
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
mul.wide.u32 rd6, r745, 954437177;
shr.u64 rd7, rd6, 33;
cvt.u32.u64 r756, rd7;
mul.lo.s32 r757, r756, 9;
sub.s32 r758, r745, r757;
shl.b32 r759, r758, 2;
add.s32 r760, r746, r759;
cvt.rn.f32.u32 f59, r756;
mul.f32 f60, f59, 0f3E6E4BAE;
cos.approx.f32 f29, f60;
sin.approx.f32 f61, f60;
neg.f32 f30, f61;
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
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
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
mad.lo.s32 r761, r756, 108, r760;
st.shared.u32 [r761], r332;
st.shared.u32 [r761+36], r423;
st.shared.u32 [r761+72], r460;
barrier.sync 0;
ld.shared.u32 r496, [r749];
ld.shared.u32 r493, [r749+324];
ld.shared.u32 r494, [r749+648];
barrier.sync 0;
st.shared.u32 [r761], r338;
st.shared.u32 [r761+36], r432;
st.shared.u32 [r761+72], r469;
barrier.sync 0;
ld.shared.u32 r502, [r749];
ld.shared.u32 r499, [r749+324];
ld.shared.u32 r500, [r749+648];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r490, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
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
mul.wide.u32 rd8, r745, 795364315;
shr.u64 rd9, rd8, 32;
cvt.u32.u64 r762, rd9;
sub.s32 r763, r745, r762;
shr.u32 r764, r763, 1;
add.s32 r765, r764, r762;
shr.u32 r766, r765, 4;
mul.lo.s32 r767, r766, 27;
sub.s32 r768, r745, r767;
shl.b32 r769, r768, 2;
add.s32 r770, r746, r769;
cvt.rn.f32.u32 f62, r766;
mul.f32 f63, f62, 0f3F32B8C2;
cos.approx.f32 f41, f63;
sin.approx.f32 f64, f63;
neg.f32 f42, f64;
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
cvt.rn.f16.f32 low, f45;
cvt.rn.f16.f32 high, f46;
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
mad.lo.s32 r771, r766, 324, r770;
st.shared.u32 [r771], r495;
st.shared.u32 [r771+108], r586;
st.shared.u32 [r771+216], r623;
barrier.sync 0;
ld.shared.u32 r659, [r749];
ld.shared.u32 r656, [r749+324];
ld.shared.u32 r657, [r749+648];
barrier.sync 0;
st.shared.u32 [r771], r501;
st.shared.u32 [r771+108], r595;
st.shared.u32 [r771+216], r632;
barrier.sync 0;
ld.shared.u32 r665, [r749];
ld.shared.u32 r662, [r749+324];
ld.shared.u32 r663, [r749+648];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r653, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r654, {low, high};
}
{
add.f16x2 r655, r656, r657;
}
{
add.f16x2 %0, r659, r655;
}
{
add.f16x2 r661, r662, r663;
}
{
add.f16x2 %1, r665, r661;
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
add.f16x2 %2, r673, r679;
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
sub.f16x2 %4, r691, r697;
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
sub.f16x2 %3, r709, r715;
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
add.f16x2 %5, r727, r733;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)));
};


#endif
