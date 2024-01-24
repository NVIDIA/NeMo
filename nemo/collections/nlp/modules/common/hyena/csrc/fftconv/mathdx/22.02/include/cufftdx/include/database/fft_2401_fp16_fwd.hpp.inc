#ifndef CUFFTDX_FFT_2401_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_2401_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<922, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<138>;
.reg .b32 r<3494>;
.reg .b64 rd<8>;
mov.u32 r3468, %tid.y;
mov.u32 r3469, %14;
mad.lo.s32 r3470, r3468, 19208, r3469;
mov.u32 r3471, %tid.x;
mov.f32 f126, 0f3F1F9D07;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f126;
cvt.rn.f16.f32 high, f126;
mov.b32 r1, {low, high};
}
mov.f32 f128, 0fBF48261C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f128;
cvt.rn.f16.f32 high, f128;
mov.b32 r2, {low, high};
}
mov.f32 f114, 0fBE63DC87;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f114;
cvt.rn.f16.f32 high, f114;
mov.b32 r3, {low, high};
}
mov.f32 f116, 0fBF7994E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f116;
cvt.rn.f16.f32 high, f116;
mov.b32 r4, {low, high};
}
mov.f32 f122, 0fBF66A5E5;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r5, {low, high};
}
mov.f32 f124, 0fBEDE2602;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r6, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r7, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r8, {low, high};
}
{
neg.f16x2 r9, r8;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f126;
cvt.rn.f16.f32 high, f126;
mov.b32 r11, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f128;
cvt.rn.f16.f32 high, f128;
mov.b32 r12, {low, high};
}
{
neg.f16x2 r13, r12;
}
{
add.f16x2 r15, %17, %27;
}
{
add.f16x2 r18, %15, r15;
}
{
add.f16x2 r21, %19, %25;
}
{
add.f16x2 r24, r18, r21;
}
{
add.f16x2 r27, %21, %23;
}
{
add.f16x2 r30, r24, r27;
}
{
add.f16x2 r33, %18, %28;
}
{
add.f16x2 r36, %16, r33;
}
{
add.f16x2 r39, %20, %26;
}
{
add.f16x2 r42, r36, r39;
}
{
add.f16x2 r45, %22, %24;
}
{
add.f16x2 r48, r42, r45;
}
{
add.f16x2 r51, %17, %27;
}
{
mul.f16x2 r54, r51, r1;
}
{
add.f16x2 r57, %15, r54;
}
{
add.f16x2 r60, %19, %25;
}
{
mul.f16x2 r63, r60, r3;
}
{
add.f16x2 r66, r57, r63;
}
{
add.f16x2 r69, %21, %23;
}
{
mul.f16x2 r72, r69, r5;
}
{
add.f16x2 r75, r66, r72;
}
{
sub.f16x2 r78, %18, %28;
}
{
mul.f16x2 r81, r78, r2;
}
{
sub.f16x2 r84, %20, %26;
}
{
mul.f16x2 r87, r84, r4;
}
{
add.f16x2 r90, r81, r87;
}
{
sub.f16x2 r93, %22, %24;
}
{
mul.f16x2 r96, r93, r6;
}
{
add.f16x2 r99, r90, r96;
}
{
sub.f16x2 r102, r75, r99;
}
{
add.f16x2 r105, %17, %27;
}
{
mul.f16x2 r108, r105, r1;
}
{
add.f16x2 r111, %15, r108;
}
{
add.f16x2 r114, %19, %25;
}
{
mul.f16x2 r117, r114, r3;
}
{
add.f16x2 r120, r111, r117;
}
{
add.f16x2 r123, %21, %23;
}
{
mul.f16x2 r126, r123, r5;
}
{
add.f16x2 r129, r120, r126;
}
{
sub.f16x2 r132, %18, %28;
}
{
mul.f16x2 r135, r132, r2;
}
{
sub.f16x2 r138, %20, %26;
}
{
mul.f16x2 r141, r138, r4;
}
{
add.f16x2 r144, r135, r141;
}
{
sub.f16x2 r147, %22, %24;
}
{
mul.f16x2 r150, r147, r6;
}
{
add.f16x2 r153, r144, r150;
}
{
add.f16x2 r156, r129, r153;
}
{
add.f16x2 r159, %17, %27;
}
{
mul.f16x2 r162, r159, r3;
}
{
add.f16x2 r165, %15, r162;
}
{
add.f16x2 r168, %19, %25;
}
{
mul.f16x2 r171, r168, r7;
}
{
add.f16x2 r174, r165, r171;
}
{
add.f16x2 r177, %21, %23;
}
{
mul.f16x2 r180, r177, r11;
}
{
add.f16x2 r183, r174, r180;
}
{
sub.f16x2 r186, %18, %28;
}
{
mul.f16x2 r189, r186, r4;
}
{
sub.f16x2 r192, %20, %26;
}
{
mul.f16x2 r195, r192, r9;
}
{
add.f16x2 r198, r189, r195;
}
{
sub.f16x2 r201, %22, %24;
}
{
mul.f16x2 r204, r201, r13;
}
{
add.f16x2 r207, r198, r204;
}
{
sub.f16x2 r210, r183, r207;
}
{
add.f16x2 r213, %17, %27;
}
{
mul.f16x2 r216, r213, r3;
}
{
add.f16x2 r219, %15, r216;
}
{
add.f16x2 r222, %19, %25;
}
{
mul.f16x2 r225, r222, r7;
}
{
add.f16x2 r228, r219, r225;
}
{
add.f16x2 r231, %21, %23;
}
{
mul.f16x2 r234, r231, r11;
}
{
add.f16x2 r237, r228, r234;
}
{
sub.f16x2 r240, %18, %28;
}
{
mul.f16x2 r243, r240, r4;
}
{
sub.f16x2 r246, %20, %26;
}
{
mul.f16x2 r249, r246, r9;
}
{
add.f16x2 r252, r243, r249;
}
{
sub.f16x2 r255, %22, %24;
}
{
mul.f16x2 r258, r255, r13;
}
{
add.f16x2 r261, r252, r258;
}
{
add.f16x2 r264, r237, r261;
}
{
add.f16x2 r267, %17, %27;
}
{
mul.f16x2 r270, r267, r5;
}
{
add.f16x2 r273, %15, r270;
}
{
add.f16x2 r276, %19, %25;
}
{
mul.f16x2 r279, r276, r11;
}
{
add.f16x2 r282, r273, r279;
}
{
add.f16x2 r285, %21, %23;
}
{
mul.f16x2 r288, r285, r3;
}
{
add.f16x2 r291, r282, r288;
}
{
sub.f16x2 r294, %18, %28;
}
{
mul.f16x2 r297, r294, r6;
}
{
sub.f16x2 r300, %20, %26;
}
{
mul.f16x2 r303, r300, r13;
}
{
add.f16x2 r306, r297, r303;
}
{
sub.f16x2 r309, %22, %24;
}
{
mul.f16x2 r312, r309, r4;
}
{
add.f16x2 r315, r306, r312;
}
{
sub.f16x2 r318, r291, r315;
}
{
add.f16x2 r321, %17, %27;
}
{
mul.f16x2 r324, r321, r5;
}
{
add.f16x2 r327, %15, r324;
}
{
add.f16x2 r330, %19, %25;
}
{
mul.f16x2 r333, r330, r11;
}
{
add.f16x2 r336, r327, r333;
}
{
add.f16x2 r339, %21, %23;
}
{
mul.f16x2 r342, r339, r3;
}
{
add.f16x2 r345, r336, r342;
}
{
sub.f16x2 r348, %18, %28;
}
{
mul.f16x2 r351, r348, r6;
}
{
sub.f16x2 r354, %20, %26;
}
{
mul.f16x2 r357, r354, r13;
}
{
add.f16x2 r360, r351, r357;
}
{
sub.f16x2 r363, %22, %24;
}
{
mul.f16x2 r366, r363, r4;
}
{
add.f16x2 r369, r360, r366;
}
{
add.f16x2 r372, r345, r369;
}
{
add.f16x2 r375, %18, %28;
}
{
mul.f16x2 r378, r375, r1;
}
{
add.f16x2 r381, %16, r378;
}
{
add.f16x2 r384, %20, %26;
}
{
mul.f16x2 r387, r384, r3;
}
{
add.f16x2 r390, r381, r387;
}
{
add.f16x2 r393, %22, %24;
}
{
mul.f16x2 r396, r393, r5;
}
{
add.f16x2 r399, r390, r396;
}
{
sub.f16x2 r402, %17, %27;
}
{
mul.f16x2 r405, r402, r2;
}
{
sub.f16x2 r408, %19, %25;
}
{
mul.f16x2 r411, r408, r4;
}
{
add.f16x2 r414, r405, r411;
}
{
sub.f16x2 r417, %21, %23;
}
{
mul.f16x2 r420, r417, r6;
}
{
add.f16x2 r423, r414, r420;
}
{
add.f16x2 r426, r399, r423;
}
{
add.f16x2 r429, %18, %28;
}
{
mul.f16x2 r432, r429, r1;
}
{
add.f16x2 r435, %16, r432;
}
{
add.f16x2 r438, %20, %26;
}
{
mul.f16x2 r441, r438, r3;
}
{
add.f16x2 r444, r435, r441;
}
{
add.f16x2 r447, %22, %24;
}
{
mul.f16x2 r450, r447, r5;
}
{
add.f16x2 r453, r444, r450;
}
{
sub.f16x2 r456, %17, %27;
}
{
mul.f16x2 r459, r456, r2;
}
{
sub.f16x2 r462, %19, %25;
}
{
mul.f16x2 r465, r462, r4;
}
{
add.f16x2 r468, r459, r465;
}
{
sub.f16x2 r471, %21, %23;
}
{
mul.f16x2 r474, r471, r6;
}
{
add.f16x2 r477, r468, r474;
}
{
sub.f16x2 r480, r453, r477;
}
{
add.f16x2 r483, %18, %28;
}
{
mul.f16x2 r486, r483, r3;
}
{
add.f16x2 r489, %16, r486;
}
{
add.f16x2 r492, %20, %26;
}
{
mul.f16x2 r495, r492, r7;
}
{
add.f16x2 r498, r489, r495;
}
{
add.f16x2 r501, %22, %24;
}
{
mul.f16x2 r504, r501, r11;
}
{
add.f16x2 r507, r498, r504;
}
{
sub.f16x2 r510, %17, %27;
}
{
mul.f16x2 r513, r510, r4;
}
{
sub.f16x2 r516, %19, %25;
}
{
mul.f16x2 r519, r516, r9;
}
{
add.f16x2 r522, r513, r519;
}
{
sub.f16x2 r525, %21, %23;
}
{
mul.f16x2 r528, r525, r13;
}
{
add.f16x2 r531, r522, r528;
}
{
add.f16x2 r534, r507, r531;
}
{
add.f16x2 r537, %18, %28;
}
{
mul.f16x2 r540, r537, r3;
}
{
add.f16x2 r543, %16, r540;
}
{
add.f16x2 r546, %20, %26;
}
{
mul.f16x2 r549, r546, r7;
}
{
add.f16x2 r552, r543, r549;
}
{
add.f16x2 r555, %22, %24;
}
{
mul.f16x2 r558, r555, r11;
}
{
add.f16x2 r561, r552, r558;
}
{
sub.f16x2 r564, %17, %27;
}
{
mul.f16x2 r567, r564, r4;
}
{
sub.f16x2 r570, %19, %25;
}
{
mul.f16x2 r573, r570, r9;
}
{
add.f16x2 r576, r567, r573;
}
{
sub.f16x2 r579, %21, %23;
}
{
mul.f16x2 r582, r579, r13;
}
{
add.f16x2 r585, r576, r582;
}
{
sub.f16x2 r588, r561, r585;
}
{
add.f16x2 r591, %18, %28;
}
{
mul.f16x2 r594, r591, r5;
}
{
add.f16x2 r597, %16, r594;
}
{
add.f16x2 r600, %20, %26;
}
{
mul.f16x2 r603, r600, r11;
}
{
add.f16x2 r606, r597, r603;
}
{
add.f16x2 r609, %22, %24;
}
{
mul.f16x2 r612, r609, r3;
}
{
add.f16x2 r615, r606, r612;
}
{
sub.f16x2 r618, %17, %27;
}
{
mul.f16x2 r621, r618, r6;
}
{
sub.f16x2 r624, %19, %25;
}
{
mul.f16x2 r627, r624, r13;
}
{
add.f16x2 r630, r621, r627;
}
{
sub.f16x2 r633, %21, %23;
}
{
mul.f16x2 r636, r633, r4;
}
{
add.f16x2 r639, r630, r636;
}
{
add.f16x2 r642, r615, r639;
}
{
add.f16x2 r645, %18, %28;
}
{
mul.f16x2 r648, r645, r5;
}
{
add.f16x2 r651, %16, r648;
}
{
add.f16x2 r654, %20, %26;
}
{
mul.f16x2 r657, r654, r11;
}
{
add.f16x2 r660, r651, r657;
}
{
add.f16x2 r663, %22, %24;
}
{
mul.f16x2 r666, r663, r3;
}
{
add.f16x2 r669, r660, r666;
}
{
sub.f16x2 r672, %17, %27;
}
{
mul.f16x2 r675, r672, r6;
}
{
sub.f16x2 r678, %19, %25;
}
{
mul.f16x2 r681, r678, r13;
}
{
add.f16x2 r684, r675, r681;
}
{
sub.f16x2 r687, %21, %23;
}
{
mul.f16x2 r690, r687, r4;
}
{
add.f16x2 r693, r684, r690;
}
{
sub.f16x2 r696, r669, r693;
}
mul.wide.u32 rd2, r3471, -1089394037;
shr.u64 rd3, rd2, 40;
cvt.u32.u64 r3472, rd3;
mul.lo.s32 r3473, r3472, 343;
sub.s32 r3474, r3471, r3473;
cvt.rn.f32.u32 f129, r3474;
mul.f32 f130, f129, 0f3B2B805B;
cos.approx.f32 f21, f130;
sin.approx.f32 f131, f130;
neg.f32 f22, f131;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f21;
cvt.rn.f16.f32 high, f22;
mov.b32 r699, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r702, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r704, {high, high};
}
{
mul.f16x2 r706, r426, r704;
}
{
neg.f16x2 r709, r706;
}
{
fma.rn.f16x2 r711, r102, r702, r709;
}
{
mul.f16x2 r715, r102, r704;
}
{
fma.rn.f16x2 r718, r426, r702, r715;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r722, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r724, {high, high};
}
mov.f32 f105, 0fBF800000;
mov.f32 f106, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r726, {low, high};
}
{
mul.f16x2 r727, r724, r726;
}
{
mul.f16x2 r730, r699, r722;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r733, {high, low};
}
{
fma.rn.f16x2 r735, r727, r733, r730;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r735;
mov.b32 r739, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r735;
mov.b32 r741, {high, high};
}
{
mul.f16x2 r743, r534, r741;
}
{
neg.f16x2 r746, r743;
}
{
fma.rn.f16x2 r748, r210, r739, r746;
}
{
mul.f16x2 r752, r210, r741;
}
{
fma.rn.f16x2 r755, r534, r739, r752;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r759, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r761, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r763, {low, high};
}
{
mul.f16x2 r764, r761, r763;
}
{
mul.f16x2 r767, r735, r759;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r735;
mov.b32 r770, {high, low};
}
{
fma.rn.f16x2 r772, r764, r770, r767;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r772;
mov.b32 r776, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r772;
mov.b32 r778, {high, high};
}
{
mul.f16x2 r780, r642, r778;
}
{
neg.f16x2 r783, r780;
}
{
fma.rn.f16x2 r785, r318, r776, r783;
}
{
mul.f16x2 r789, r318, r778;
}
{
fma.rn.f16x2 r792, r642, r776, r789;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r796, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r798, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r800, {low, high};
}
{
mul.f16x2 r801, r798, r800;
}
{
mul.f16x2 r804, r772, r796;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r772;
mov.b32 r807, {high, low};
}
{
fma.rn.f16x2 r809, r801, r807, r804;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r809;
mov.b32 r813, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r809;
mov.b32 r815, {high, high};
}
{
mul.f16x2 r817, r696, r815;
}
{
neg.f16x2 r820, r817;
}
{
fma.rn.f16x2 r822, r372, r813, r820;
}
{
mul.f16x2 r826, r372, r815;
}
{
fma.rn.f16x2 r829, r696, r813, r826;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r833, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r835, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r837, {low, high};
}
{
mul.f16x2 r838, r835, r837;
}
{
mul.f16x2 r841, r809, r833;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r809;
mov.b32 r844, {high, low};
}
{
fma.rn.f16x2 r846, r838, r844, r841;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r846;
mov.b32 r850, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r846;
mov.b32 r852, {high, high};
}
{
mul.f16x2 r854, r588, r852;
}
{
neg.f16x2 r857, r854;
}
{
fma.rn.f16x2 r859, r264, r850, r857;
}
{
mul.f16x2 r863, r264, r852;
}
{
fma.rn.f16x2 r866, r588, r850, r863;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r870, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r872, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r874, {low, high};
}
{
mul.f16x2 r875, r872, r874;
}
{
mul.f16x2 r878, r846, r870;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r846;
mov.b32 r881, {high, low};
}
{
fma.rn.f16x2 r883, r875, r881, r878;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r883;
mov.b32 r887, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r883;
mov.b32 r889, {high, high};
}
{
mul.f16x2 r891, r480, r889;
}
{
neg.f16x2 r894, r891;
}
{
fma.rn.f16x2 r896, r156, r887, r894;
}
{
mul.f16x2 r900, r156, r889;
}
{
fma.rn.f16x2 r903, r480, r887, r900;
}
mad.lo.s32 r3475, r3472, 19208, r3470;
barrier.sync 0;
mad.lo.s32 r3476, r3474, 56, r3475;
st.shared.v2.f32 [r3476], {r30, r48};
st.shared.v2.f32 [r3476+8], {r711, r718};
st.shared.v2.f32 [r3476+16], {r748, r755};
st.shared.v2.f32 [r3476+24], {r785, r792};
st.shared.v2.f32 [r3476+32], {r822, r829};
st.shared.v2.f32 [r3476+40], {r859, r866};
st.shared.v2.f32 [r3476+48], {r896, r903};
barrier.sync 0;
mad.lo.s32 r3477, r3474, -48, r3476;
ld.shared.u32 r942, [r3477];
ld.shared.u32 r960, [r3477+4];
ld.shared.u32 r939, [r3477+2744];
ld.shared.u32 r957, [r3477+2748];
ld.shared.u32 r945, [r3477+5488];
ld.shared.u32 r963, [r3477+5492];
ld.shared.u32 r951, [r3477+8232];
ld.shared.u32 r969, [r3477+8236];
ld.shared.u32 r952, [r3477+10976];
ld.shared.u32 r970, [r3477+10980];
ld.shared.u32 r946, [r3477+13720];
ld.shared.u32 r964, [r3477+13724];
ld.shared.u32 r940, [r3477+16464];
ld.shared.u32 r958, [r3477+16468];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f126;
cvt.rn.f16.f32 high, f126;
mov.b32 r924, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f128;
cvt.rn.f16.f32 high, f128;
mov.b32 r925, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f114;
cvt.rn.f16.f32 high, f114;
mov.b32 r926, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f116;
cvt.rn.f16.f32 high, f116;
mov.b32 r927, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r928, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r929, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r930, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r931, {low, high};
}
{
neg.f16x2 r932, r931;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f126;
cvt.rn.f16.f32 high, f126;
mov.b32 r934, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f128;
cvt.rn.f16.f32 high, f128;
mov.b32 r935, {low, high};
}
{
neg.f16x2 r936, r935;
}
{
add.f16x2 r938, r939, r940;
}
{
add.f16x2 r941, r942, r938;
}
{
add.f16x2 r944, r945, r946;
}
{
add.f16x2 r947, r941, r944;
}
{
add.f16x2 r950, r951, r952;
}
{
add.f16x2 r953, r947, r950;
}
{
add.f16x2 r956, r957, r958;
}
{
add.f16x2 r959, r960, r956;
}
{
add.f16x2 r962, r963, r964;
}
{
add.f16x2 r965, r959, r962;
}
{
add.f16x2 r968, r969, r970;
}
{
add.f16x2 r971, r965, r968;
}
{
add.f16x2 r974, r939, r940;
}
{
mul.f16x2 r977, r974, r924;
}
{
add.f16x2 r980, r942, r977;
}
{
add.f16x2 r983, r945, r946;
}
{
mul.f16x2 r986, r983, r926;
}
{
add.f16x2 r989, r980, r986;
}
{
add.f16x2 r992, r951, r952;
}
{
mul.f16x2 r995, r992, r928;
}
{
add.f16x2 r998, r989, r995;
}
{
sub.f16x2 r1001, r957, r958;
}
{
mul.f16x2 r1004, r1001, r925;
}
{
sub.f16x2 r1007, r963, r964;
}
{
mul.f16x2 r1010, r1007, r927;
}
{
add.f16x2 r1013, r1004, r1010;
}
{
sub.f16x2 r1016, r969, r970;
}
{
mul.f16x2 r1019, r1016, r929;
}
{
add.f16x2 r1022, r1013, r1019;
}
{
sub.f16x2 r1025, r998, r1022;
}
{
add.f16x2 r1028, r939, r940;
}
{
mul.f16x2 r1031, r1028, r924;
}
{
add.f16x2 r1034, r942, r1031;
}
{
add.f16x2 r1037, r945, r946;
}
{
mul.f16x2 r1040, r1037, r926;
}
{
add.f16x2 r1043, r1034, r1040;
}
{
add.f16x2 r1046, r951, r952;
}
{
mul.f16x2 r1049, r1046, r928;
}
{
add.f16x2 r1052, r1043, r1049;
}
{
sub.f16x2 r1055, r957, r958;
}
{
mul.f16x2 r1058, r1055, r925;
}
{
sub.f16x2 r1061, r963, r964;
}
{
mul.f16x2 r1064, r1061, r927;
}
{
add.f16x2 r1067, r1058, r1064;
}
{
sub.f16x2 r1070, r969, r970;
}
{
mul.f16x2 r1073, r1070, r929;
}
{
add.f16x2 r1076, r1067, r1073;
}
{
add.f16x2 r1079, r1052, r1076;
}
{
add.f16x2 r1082, r939, r940;
}
{
mul.f16x2 r1085, r1082, r926;
}
{
add.f16x2 r1088, r942, r1085;
}
{
add.f16x2 r1091, r945, r946;
}
{
mul.f16x2 r1094, r1091, r930;
}
{
add.f16x2 r1097, r1088, r1094;
}
{
add.f16x2 r1100, r951, r952;
}
{
mul.f16x2 r1103, r1100, r934;
}
{
add.f16x2 r1106, r1097, r1103;
}
{
sub.f16x2 r1109, r957, r958;
}
{
mul.f16x2 r1112, r1109, r927;
}
{
sub.f16x2 r1115, r963, r964;
}
{
mul.f16x2 r1118, r1115, r932;
}
{
add.f16x2 r1121, r1112, r1118;
}
{
sub.f16x2 r1124, r969, r970;
}
{
mul.f16x2 r1127, r1124, r936;
}
{
add.f16x2 r1130, r1121, r1127;
}
{
sub.f16x2 r1133, r1106, r1130;
}
{
add.f16x2 r1136, r939, r940;
}
{
mul.f16x2 r1139, r1136, r926;
}
{
add.f16x2 r1142, r942, r1139;
}
{
add.f16x2 r1145, r945, r946;
}
{
mul.f16x2 r1148, r1145, r930;
}
{
add.f16x2 r1151, r1142, r1148;
}
{
add.f16x2 r1154, r951, r952;
}
{
mul.f16x2 r1157, r1154, r934;
}
{
add.f16x2 r1160, r1151, r1157;
}
{
sub.f16x2 r1163, r957, r958;
}
{
mul.f16x2 r1166, r1163, r927;
}
{
sub.f16x2 r1169, r963, r964;
}
{
mul.f16x2 r1172, r1169, r932;
}
{
add.f16x2 r1175, r1166, r1172;
}
{
sub.f16x2 r1178, r969, r970;
}
{
mul.f16x2 r1181, r1178, r936;
}
{
add.f16x2 r1184, r1175, r1181;
}
{
add.f16x2 r1187, r1160, r1184;
}
{
add.f16x2 r1190, r939, r940;
}
{
mul.f16x2 r1193, r1190, r928;
}
{
add.f16x2 r1196, r942, r1193;
}
{
add.f16x2 r1199, r945, r946;
}
{
mul.f16x2 r1202, r1199, r934;
}
{
add.f16x2 r1205, r1196, r1202;
}
{
add.f16x2 r1208, r951, r952;
}
{
mul.f16x2 r1211, r1208, r926;
}
{
add.f16x2 r1214, r1205, r1211;
}
{
sub.f16x2 r1217, r957, r958;
}
{
mul.f16x2 r1220, r1217, r929;
}
{
sub.f16x2 r1223, r963, r964;
}
{
mul.f16x2 r1226, r1223, r936;
}
{
add.f16x2 r1229, r1220, r1226;
}
{
sub.f16x2 r1232, r969, r970;
}
{
mul.f16x2 r1235, r1232, r927;
}
{
add.f16x2 r1238, r1229, r1235;
}
{
sub.f16x2 r1241, r1214, r1238;
}
{
add.f16x2 r1244, r939, r940;
}
{
mul.f16x2 r1247, r1244, r928;
}
{
add.f16x2 r1250, r942, r1247;
}
{
add.f16x2 r1253, r945, r946;
}
{
mul.f16x2 r1256, r1253, r934;
}
{
add.f16x2 r1259, r1250, r1256;
}
{
add.f16x2 r1262, r951, r952;
}
{
mul.f16x2 r1265, r1262, r926;
}
{
add.f16x2 r1268, r1259, r1265;
}
{
sub.f16x2 r1271, r957, r958;
}
{
mul.f16x2 r1274, r1271, r929;
}
{
sub.f16x2 r1277, r963, r964;
}
{
mul.f16x2 r1280, r1277, r936;
}
{
add.f16x2 r1283, r1274, r1280;
}
{
sub.f16x2 r1286, r969, r970;
}
{
mul.f16x2 r1289, r1286, r927;
}
{
add.f16x2 r1292, r1283, r1289;
}
{
add.f16x2 r1295, r1268, r1292;
}
{
add.f16x2 r1298, r957, r958;
}
{
mul.f16x2 r1301, r1298, r924;
}
{
add.f16x2 r1304, r960, r1301;
}
{
add.f16x2 r1307, r963, r964;
}
{
mul.f16x2 r1310, r1307, r926;
}
{
add.f16x2 r1313, r1304, r1310;
}
{
add.f16x2 r1316, r969, r970;
}
{
mul.f16x2 r1319, r1316, r928;
}
{
add.f16x2 r1322, r1313, r1319;
}
{
sub.f16x2 r1325, r939, r940;
}
{
mul.f16x2 r1328, r1325, r925;
}
{
sub.f16x2 r1331, r945, r946;
}
{
mul.f16x2 r1334, r1331, r927;
}
{
add.f16x2 r1337, r1328, r1334;
}
{
sub.f16x2 r1340, r951, r952;
}
{
mul.f16x2 r1343, r1340, r929;
}
{
add.f16x2 r1346, r1337, r1343;
}
{
add.f16x2 r1349, r1322, r1346;
}
{
add.f16x2 r1352, r957, r958;
}
{
mul.f16x2 r1355, r1352, r924;
}
{
add.f16x2 r1358, r960, r1355;
}
{
add.f16x2 r1361, r963, r964;
}
{
mul.f16x2 r1364, r1361, r926;
}
{
add.f16x2 r1367, r1358, r1364;
}
{
add.f16x2 r1370, r969, r970;
}
{
mul.f16x2 r1373, r1370, r928;
}
{
add.f16x2 r1376, r1367, r1373;
}
{
sub.f16x2 r1379, r939, r940;
}
{
mul.f16x2 r1382, r1379, r925;
}
{
sub.f16x2 r1385, r945, r946;
}
{
mul.f16x2 r1388, r1385, r927;
}
{
add.f16x2 r1391, r1382, r1388;
}
{
sub.f16x2 r1394, r951, r952;
}
{
mul.f16x2 r1397, r1394, r929;
}
{
add.f16x2 r1400, r1391, r1397;
}
{
sub.f16x2 r1403, r1376, r1400;
}
{
add.f16x2 r1406, r957, r958;
}
{
mul.f16x2 r1409, r1406, r926;
}
{
add.f16x2 r1412, r960, r1409;
}
{
add.f16x2 r1415, r963, r964;
}
{
mul.f16x2 r1418, r1415, r930;
}
{
add.f16x2 r1421, r1412, r1418;
}
{
add.f16x2 r1424, r969, r970;
}
{
mul.f16x2 r1427, r1424, r934;
}
{
add.f16x2 r1430, r1421, r1427;
}
{
sub.f16x2 r1433, r939, r940;
}
{
mul.f16x2 r1436, r1433, r927;
}
{
sub.f16x2 r1439, r945, r946;
}
{
mul.f16x2 r1442, r1439, r932;
}
{
add.f16x2 r1445, r1436, r1442;
}
{
sub.f16x2 r1448, r951, r952;
}
{
mul.f16x2 r1451, r1448, r936;
}
{
add.f16x2 r1454, r1445, r1451;
}
{
add.f16x2 r1457, r1430, r1454;
}
{
add.f16x2 r1460, r957, r958;
}
{
mul.f16x2 r1463, r1460, r926;
}
{
add.f16x2 r1466, r960, r1463;
}
{
add.f16x2 r1469, r963, r964;
}
{
mul.f16x2 r1472, r1469, r930;
}
{
add.f16x2 r1475, r1466, r1472;
}
{
add.f16x2 r1478, r969, r970;
}
{
mul.f16x2 r1481, r1478, r934;
}
{
add.f16x2 r1484, r1475, r1481;
}
{
sub.f16x2 r1487, r939, r940;
}
{
mul.f16x2 r1490, r1487, r927;
}
{
sub.f16x2 r1493, r945, r946;
}
{
mul.f16x2 r1496, r1493, r932;
}
{
add.f16x2 r1499, r1490, r1496;
}
{
sub.f16x2 r1502, r951, r952;
}
{
mul.f16x2 r1505, r1502, r936;
}
{
add.f16x2 r1508, r1499, r1505;
}
{
sub.f16x2 r1511, r1484, r1508;
}
{
add.f16x2 r1514, r957, r958;
}
{
mul.f16x2 r1517, r1514, r928;
}
{
add.f16x2 r1520, r960, r1517;
}
{
add.f16x2 r1523, r963, r964;
}
{
mul.f16x2 r1526, r1523, r934;
}
{
add.f16x2 r1529, r1520, r1526;
}
{
add.f16x2 r1532, r969, r970;
}
{
mul.f16x2 r1535, r1532, r926;
}
{
add.f16x2 r1538, r1529, r1535;
}
{
sub.f16x2 r1541, r939, r940;
}
{
mul.f16x2 r1544, r1541, r929;
}
{
sub.f16x2 r1547, r945, r946;
}
{
mul.f16x2 r1550, r1547, r936;
}
{
add.f16x2 r1553, r1544, r1550;
}
{
sub.f16x2 r1556, r951, r952;
}
{
mul.f16x2 r1559, r1556, r927;
}
{
add.f16x2 r1562, r1553, r1559;
}
{
add.f16x2 r1565, r1538, r1562;
}
{
add.f16x2 r1568, r957, r958;
}
{
mul.f16x2 r1571, r1568, r928;
}
{
add.f16x2 r1574, r960, r1571;
}
{
add.f16x2 r1577, r963, r964;
}
{
mul.f16x2 r1580, r1577, r934;
}
{
add.f16x2 r1583, r1574, r1580;
}
{
add.f16x2 r1586, r969, r970;
}
{
mul.f16x2 r1589, r1586, r926;
}
{
add.f16x2 r1592, r1583, r1589;
}
{
sub.f16x2 r1595, r939, r940;
}
{
mul.f16x2 r1598, r1595, r929;
}
{
sub.f16x2 r1601, r945, r946;
}
{
mul.f16x2 r1604, r1601, r936;
}
{
add.f16x2 r1607, r1598, r1604;
}
{
sub.f16x2 r1610, r951, r952;
}
{
mul.f16x2 r1613, r1610, r927;
}
{
add.f16x2 r1616, r1607, r1613;
}
{
sub.f16x2 r1619, r1592, r1616;
}
mul.wide.u32 rd4, r3474, 613566757;
shr.u64 rd5, rd4, 32;
cvt.u32.u64 r3478, rd5;
sub.s32 r3479, r3474, r3478;
shr.u32 r3480, r3479, 1;
add.s32 r3481, r3480, r3478;
shr.u32 r3482, r3481, 2;
cvt.rn.f32.u32 f132, r3482;
mul.f32 f133, f132, 0f3C961050;
cos.approx.f32 f57, f133;
sin.approx.f32 f134, f133;
neg.f32 f58, f134;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f57;
cvt.rn.f16.f32 high, f58;
mov.b32 r1622, {low, high};
}
mul.lo.s32 r3483, r3482, 7;
sub.s32 r3484, r3474, r3483;
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1625, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1627, {high, high};
}
{
mul.f16x2 r1629, r1349, r1627;
}
{
neg.f16x2 r1632, r1629;
}
{
fma.rn.f16x2 r1634, r1025, r1625, r1632;
}
{
mul.f16x2 r1638, r1025, r1627;
}
{
fma.rn.f16x2 r1641, r1349, r1625, r1638;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1645, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1647, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r1649, {low, high};
}
{
mul.f16x2 r1650, r1647, r1649;
}
{
mul.f16x2 r1653, r1622, r1645;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1656, {high, low};
}
{
fma.rn.f16x2 r1658, r1650, r1656, r1653;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1658;
mov.b32 r1662, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1658;
mov.b32 r1664, {high, high};
}
{
mul.f16x2 r1666, r1457, r1664;
}
{
neg.f16x2 r1669, r1666;
}
{
fma.rn.f16x2 r1671, r1133, r1662, r1669;
}
{
mul.f16x2 r1675, r1133, r1664;
}
{
fma.rn.f16x2 r1678, r1457, r1662, r1675;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1682, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1684, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r1686, {low, high};
}
{
mul.f16x2 r1687, r1684, r1686;
}
{
mul.f16x2 r1690, r1658, r1682;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1658;
mov.b32 r1693, {high, low};
}
{
fma.rn.f16x2 r1695, r1687, r1693, r1690;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1695;
mov.b32 r1699, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1695;
mov.b32 r1701, {high, high};
}
{
mul.f16x2 r1703, r1565, r1701;
}
{
neg.f16x2 r1706, r1703;
}
{
fma.rn.f16x2 r1708, r1241, r1699, r1706;
}
{
mul.f16x2 r1712, r1241, r1701;
}
{
fma.rn.f16x2 r1715, r1565, r1699, r1712;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1719, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1721, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r1723, {low, high};
}
{
mul.f16x2 r1724, r1721, r1723;
}
{
mul.f16x2 r1727, r1695, r1719;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1695;
mov.b32 r1730, {high, low};
}
{
fma.rn.f16x2 r1732, r1724, r1730, r1727;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1732;
mov.b32 r1736, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1732;
mov.b32 r1738, {high, high};
}
{
mul.f16x2 r1740, r1619, r1738;
}
{
neg.f16x2 r1743, r1740;
}
{
fma.rn.f16x2 r1745, r1295, r1736, r1743;
}
{
mul.f16x2 r1749, r1295, r1738;
}
{
fma.rn.f16x2 r1752, r1619, r1736, r1749;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1756, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1758, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r1760, {low, high};
}
{
mul.f16x2 r1761, r1758, r1760;
}
{
mul.f16x2 r1764, r1732, r1756;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1732;
mov.b32 r1767, {high, low};
}
{
fma.rn.f16x2 r1769, r1761, r1767, r1764;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1769;
mov.b32 r1773, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1769;
mov.b32 r1775, {high, high};
}
{
mul.f16x2 r1777, r1511, r1775;
}
{
neg.f16x2 r1780, r1777;
}
{
fma.rn.f16x2 r1782, r1187, r1773, r1780;
}
{
mul.f16x2 r1786, r1187, r1775;
}
{
fma.rn.f16x2 r1789, r1511, r1773, r1786;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1793, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1795, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r1797, {low, high};
}
{
mul.f16x2 r1798, r1795, r1797;
}
{
mul.f16x2 r1801, r1769, r1793;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1769;
mov.b32 r1804, {high, low};
}
{
fma.rn.f16x2 r1806, r1798, r1804, r1801;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1806;
mov.b32 r1810, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1806;
mov.b32 r1812, {high, high};
}
{
mul.f16x2 r1814, r1403, r1812;
}
{
neg.f16x2 r1817, r1814;
}
{
fma.rn.f16x2 r1819, r1079, r1810, r1817;
}
{
mul.f16x2 r1823, r1079, r1812;
}
{
fma.rn.f16x2 r1826, r1403, r1810, r1823;
}
shl.b32 r3485, r3484, 3;
add.s32 r3486, r3475, r3485;
barrier.sync 0;
mad.lo.s32 r3487, r3482, 392, r3486;
st.shared.u32 [r3487], r953;
st.shared.u32 [r3487+4], r971;
st.shared.u32 [r3487+56], r1634;
st.shared.u32 [r3487+60], r1641;
st.shared.u32 [r3487+112], r1671;
st.shared.u32 [r3487+116], r1678;
st.shared.u32 [r3487+168], r1708;
st.shared.u32 [r3487+172], r1715;
st.shared.u32 [r3487+224], r1745;
st.shared.u32 [r3487+228], r1752;
st.shared.u32 [r3487+280], r1782;
st.shared.u32 [r3487+284], r1789;
st.shared.u32 [r3487+336], r1819;
st.shared.u32 [r3487+340], r1826;
barrier.sync 0;
ld.shared.u32 r1865, [r3477];
ld.shared.u32 r1883, [r3477+4];
ld.shared.u32 r1862, [r3477+2744];
ld.shared.u32 r1880, [r3477+2748];
ld.shared.u32 r1868, [r3477+5488];
ld.shared.u32 r1886, [r3477+5492];
ld.shared.u32 r1874, [r3477+8232];
ld.shared.u32 r1892, [r3477+8236];
ld.shared.u32 r1875, [r3477+10976];
ld.shared.u32 r1893, [r3477+10980];
ld.shared.u32 r1869, [r3477+13720];
ld.shared.u32 r1887, [r3477+13724];
ld.shared.u32 r1863, [r3477+16464];
ld.shared.u32 r1881, [r3477+16468];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f126;
cvt.rn.f16.f32 high, f126;
mov.b32 r1847, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f128;
cvt.rn.f16.f32 high, f128;
mov.b32 r1848, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f114;
cvt.rn.f16.f32 high, f114;
mov.b32 r1849, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f116;
cvt.rn.f16.f32 high, f116;
mov.b32 r1850, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r1851, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r1852, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r1853, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r1854, {low, high};
}
{
neg.f16x2 r1855, r1854;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f126;
cvt.rn.f16.f32 high, f126;
mov.b32 r1857, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f128;
cvt.rn.f16.f32 high, f128;
mov.b32 r1858, {low, high};
}
{
neg.f16x2 r1859, r1858;
}
{
add.f16x2 r1861, r1862, r1863;
}
{
add.f16x2 r1864, r1865, r1861;
}
{
add.f16x2 r1867, r1868, r1869;
}
{
add.f16x2 r1870, r1864, r1867;
}
{
add.f16x2 r1873, r1874, r1875;
}
{
add.f16x2 r1876, r1870, r1873;
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
add.f16x2 r1888, r1882, r1885;
}
{
add.f16x2 r1891, r1892, r1893;
}
{
add.f16x2 r1894, r1888, r1891;
}
{
add.f16x2 r1897, r1862, r1863;
}
{
mul.f16x2 r1900, r1897, r1847;
}
{
add.f16x2 r1903, r1865, r1900;
}
{
add.f16x2 r1906, r1868, r1869;
}
{
mul.f16x2 r1909, r1906, r1849;
}
{
add.f16x2 r1912, r1903, r1909;
}
{
add.f16x2 r1915, r1874, r1875;
}
{
mul.f16x2 r1918, r1915, r1851;
}
{
add.f16x2 r1921, r1912, r1918;
}
{
sub.f16x2 r1924, r1880, r1881;
}
{
mul.f16x2 r1927, r1924, r1848;
}
{
sub.f16x2 r1930, r1886, r1887;
}
{
mul.f16x2 r1933, r1930, r1850;
}
{
add.f16x2 r1936, r1927, r1933;
}
{
sub.f16x2 r1939, r1892, r1893;
}
{
mul.f16x2 r1942, r1939, r1852;
}
{
add.f16x2 r1945, r1936, r1942;
}
{
sub.f16x2 r1948, r1921, r1945;
}
{
add.f16x2 r1951, r1862, r1863;
}
{
mul.f16x2 r1954, r1951, r1847;
}
{
add.f16x2 r1957, r1865, r1954;
}
{
add.f16x2 r1960, r1868, r1869;
}
{
mul.f16x2 r1963, r1960, r1849;
}
{
add.f16x2 r1966, r1957, r1963;
}
{
add.f16x2 r1969, r1874, r1875;
}
{
mul.f16x2 r1972, r1969, r1851;
}
{
add.f16x2 r1975, r1966, r1972;
}
{
sub.f16x2 r1978, r1880, r1881;
}
{
mul.f16x2 r1981, r1978, r1848;
}
{
sub.f16x2 r1984, r1886, r1887;
}
{
mul.f16x2 r1987, r1984, r1850;
}
{
add.f16x2 r1990, r1981, r1987;
}
{
sub.f16x2 r1993, r1892, r1893;
}
{
mul.f16x2 r1996, r1993, r1852;
}
{
add.f16x2 r1999, r1990, r1996;
}
{
add.f16x2 r2002, r1975, r1999;
}
{
add.f16x2 r2005, r1862, r1863;
}
{
mul.f16x2 r2008, r2005, r1849;
}
{
add.f16x2 r2011, r1865, r2008;
}
{
add.f16x2 r2014, r1868, r1869;
}
{
mul.f16x2 r2017, r2014, r1853;
}
{
add.f16x2 r2020, r2011, r2017;
}
{
add.f16x2 r2023, r1874, r1875;
}
{
mul.f16x2 r2026, r2023, r1857;
}
{
add.f16x2 r2029, r2020, r2026;
}
{
sub.f16x2 r2032, r1880, r1881;
}
{
mul.f16x2 r2035, r2032, r1850;
}
{
sub.f16x2 r2038, r1886, r1887;
}
{
mul.f16x2 r2041, r2038, r1855;
}
{
add.f16x2 r2044, r2035, r2041;
}
{
sub.f16x2 r2047, r1892, r1893;
}
{
mul.f16x2 r2050, r2047, r1859;
}
{
add.f16x2 r2053, r2044, r2050;
}
{
sub.f16x2 r2056, r2029, r2053;
}
{
add.f16x2 r2059, r1862, r1863;
}
{
mul.f16x2 r2062, r2059, r1849;
}
{
add.f16x2 r2065, r1865, r2062;
}
{
add.f16x2 r2068, r1868, r1869;
}
{
mul.f16x2 r2071, r2068, r1853;
}
{
add.f16x2 r2074, r2065, r2071;
}
{
add.f16x2 r2077, r1874, r1875;
}
{
mul.f16x2 r2080, r2077, r1857;
}
{
add.f16x2 r2083, r2074, r2080;
}
{
sub.f16x2 r2086, r1880, r1881;
}
{
mul.f16x2 r2089, r2086, r1850;
}
{
sub.f16x2 r2092, r1886, r1887;
}
{
mul.f16x2 r2095, r2092, r1855;
}
{
add.f16x2 r2098, r2089, r2095;
}
{
sub.f16x2 r2101, r1892, r1893;
}
{
mul.f16x2 r2104, r2101, r1859;
}
{
add.f16x2 r2107, r2098, r2104;
}
{
add.f16x2 r2110, r2083, r2107;
}
{
add.f16x2 r2113, r1862, r1863;
}
{
mul.f16x2 r2116, r2113, r1851;
}
{
add.f16x2 r2119, r1865, r2116;
}
{
add.f16x2 r2122, r1868, r1869;
}
{
mul.f16x2 r2125, r2122, r1857;
}
{
add.f16x2 r2128, r2119, r2125;
}
{
add.f16x2 r2131, r1874, r1875;
}
{
mul.f16x2 r2134, r2131, r1849;
}
{
add.f16x2 r2137, r2128, r2134;
}
{
sub.f16x2 r2140, r1880, r1881;
}
{
mul.f16x2 r2143, r2140, r1852;
}
{
sub.f16x2 r2146, r1886, r1887;
}
{
mul.f16x2 r2149, r2146, r1859;
}
{
add.f16x2 r2152, r2143, r2149;
}
{
sub.f16x2 r2155, r1892, r1893;
}
{
mul.f16x2 r2158, r2155, r1850;
}
{
add.f16x2 r2161, r2152, r2158;
}
{
sub.f16x2 r2164, r2137, r2161;
}
{
add.f16x2 r2167, r1862, r1863;
}
{
mul.f16x2 r2170, r2167, r1851;
}
{
add.f16x2 r2173, r1865, r2170;
}
{
add.f16x2 r2176, r1868, r1869;
}
{
mul.f16x2 r2179, r2176, r1857;
}
{
add.f16x2 r2182, r2173, r2179;
}
{
add.f16x2 r2185, r1874, r1875;
}
{
mul.f16x2 r2188, r2185, r1849;
}
{
add.f16x2 r2191, r2182, r2188;
}
{
sub.f16x2 r2194, r1880, r1881;
}
{
mul.f16x2 r2197, r2194, r1852;
}
{
sub.f16x2 r2200, r1886, r1887;
}
{
mul.f16x2 r2203, r2200, r1859;
}
{
add.f16x2 r2206, r2197, r2203;
}
{
sub.f16x2 r2209, r1892, r1893;
}
{
mul.f16x2 r2212, r2209, r1850;
}
{
add.f16x2 r2215, r2206, r2212;
}
{
add.f16x2 r2218, r2191, r2215;
}
{
add.f16x2 r2221, r1880, r1881;
}
{
mul.f16x2 r2224, r2221, r1847;
}
{
add.f16x2 r2227, r1883, r2224;
}
{
add.f16x2 r2230, r1886, r1887;
}
{
mul.f16x2 r2233, r2230, r1849;
}
{
add.f16x2 r2236, r2227, r2233;
}
{
add.f16x2 r2239, r1892, r1893;
}
{
mul.f16x2 r2242, r2239, r1851;
}
{
add.f16x2 r2245, r2236, r2242;
}
{
sub.f16x2 r2248, r1862, r1863;
}
{
mul.f16x2 r2251, r2248, r1848;
}
{
sub.f16x2 r2254, r1868, r1869;
}
{
mul.f16x2 r2257, r2254, r1850;
}
{
add.f16x2 r2260, r2251, r2257;
}
{
sub.f16x2 r2263, r1874, r1875;
}
{
mul.f16x2 r2266, r2263, r1852;
}
{
add.f16x2 r2269, r2260, r2266;
}
{
add.f16x2 r2272, r2245, r2269;
}
{
add.f16x2 r2275, r1880, r1881;
}
{
mul.f16x2 r2278, r2275, r1847;
}
{
add.f16x2 r2281, r1883, r2278;
}
{
add.f16x2 r2284, r1886, r1887;
}
{
mul.f16x2 r2287, r2284, r1849;
}
{
add.f16x2 r2290, r2281, r2287;
}
{
add.f16x2 r2293, r1892, r1893;
}
{
mul.f16x2 r2296, r2293, r1851;
}
{
add.f16x2 r2299, r2290, r2296;
}
{
sub.f16x2 r2302, r1862, r1863;
}
{
mul.f16x2 r2305, r2302, r1848;
}
{
sub.f16x2 r2308, r1868, r1869;
}
{
mul.f16x2 r2311, r2308, r1850;
}
{
add.f16x2 r2314, r2305, r2311;
}
{
sub.f16x2 r2317, r1874, r1875;
}
{
mul.f16x2 r2320, r2317, r1852;
}
{
add.f16x2 r2323, r2314, r2320;
}
{
sub.f16x2 r2326, r2299, r2323;
}
{
add.f16x2 r2329, r1880, r1881;
}
{
mul.f16x2 r2332, r2329, r1849;
}
{
add.f16x2 r2335, r1883, r2332;
}
{
add.f16x2 r2338, r1886, r1887;
}
{
mul.f16x2 r2341, r2338, r1853;
}
{
add.f16x2 r2344, r2335, r2341;
}
{
add.f16x2 r2347, r1892, r1893;
}
{
mul.f16x2 r2350, r2347, r1857;
}
{
add.f16x2 r2353, r2344, r2350;
}
{
sub.f16x2 r2356, r1862, r1863;
}
{
mul.f16x2 r2359, r2356, r1850;
}
{
sub.f16x2 r2362, r1868, r1869;
}
{
mul.f16x2 r2365, r2362, r1855;
}
{
add.f16x2 r2368, r2359, r2365;
}
{
sub.f16x2 r2371, r1874, r1875;
}
{
mul.f16x2 r2374, r2371, r1859;
}
{
add.f16x2 r2377, r2368, r2374;
}
{
add.f16x2 r2380, r2353, r2377;
}
{
add.f16x2 r2383, r1880, r1881;
}
{
mul.f16x2 r2386, r2383, r1849;
}
{
add.f16x2 r2389, r1883, r2386;
}
{
add.f16x2 r2392, r1886, r1887;
}
{
mul.f16x2 r2395, r2392, r1853;
}
{
add.f16x2 r2398, r2389, r2395;
}
{
add.f16x2 r2401, r1892, r1893;
}
{
mul.f16x2 r2404, r2401, r1857;
}
{
add.f16x2 r2407, r2398, r2404;
}
{
sub.f16x2 r2410, r1862, r1863;
}
{
mul.f16x2 r2413, r2410, r1850;
}
{
sub.f16x2 r2416, r1868, r1869;
}
{
mul.f16x2 r2419, r2416, r1855;
}
{
add.f16x2 r2422, r2413, r2419;
}
{
sub.f16x2 r2425, r1874, r1875;
}
{
mul.f16x2 r2428, r2425, r1859;
}
{
add.f16x2 r2431, r2422, r2428;
}
{
sub.f16x2 r2434, r2407, r2431;
}
{
add.f16x2 r2437, r1880, r1881;
}
{
mul.f16x2 r2440, r2437, r1851;
}
{
add.f16x2 r2443, r1883, r2440;
}
{
add.f16x2 r2446, r1886, r1887;
}
{
mul.f16x2 r2449, r2446, r1857;
}
{
add.f16x2 r2452, r2443, r2449;
}
{
add.f16x2 r2455, r1892, r1893;
}
{
mul.f16x2 r2458, r2455, r1849;
}
{
add.f16x2 r2461, r2452, r2458;
}
{
sub.f16x2 r2464, r1862, r1863;
}
{
mul.f16x2 r2467, r2464, r1852;
}
{
sub.f16x2 r2470, r1868, r1869;
}
{
mul.f16x2 r2473, r2470, r1859;
}
{
add.f16x2 r2476, r2467, r2473;
}
{
sub.f16x2 r2479, r1874, r1875;
}
{
mul.f16x2 r2482, r2479, r1850;
}
{
add.f16x2 r2485, r2476, r2482;
}
{
add.f16x2 r2488, r2461, r2485;
}
{
add.f16x2 r2491, r1880, r1881;
}
{
mul.f16x2 r2494, r2491, r1851;
}
{
add.f16x2 r2497, r1883, r2494;
}
{
add.f16x2 r2500, r1886, r1887;
}
{
mul.f16x2 r2503, r2500, r1857;
}
{
add.f16x2 r2506, r2497, r2503;
}
{
add.f16x2 r2509, r1892, r1893;
}
{
mul.f16x2 r2512, r2509, r1849;
}
{
add.f16x2 r2515, r2506, r2512;
}
{
sub.f16x2 r2518, r1862, r1863;
}
{
mul.f16x2 r2521, r2518, r1852;
}
{
sub.f16x2 r2524, r1868, r1869;
}
{
mul.f16x2 r2527, r2524, r1859;
}
{
add.f16x2 r2530, r2521, r2527;
}
{
sub.f16x2 r2533, r1874, r1875;
}
{
mul.f16x2 r2536, r2533, r1850;
}
{
add.f16x2 r2539, r2530, r2536;
}
{
sub.f16x2 r2542, r2515, r2539;
}
mul.wide.u32 rd6, r3474, 1402438301;
shr.u64 rd7, rd6, 36;
cvt.u32.u64 r3488, rd7;
cvt.rn.f32.u32 f135, r3488;
mul.f32 f136, f135, 0f3E034E46;
cos.approx.f32 f93, f136;
sin.approx.f32 f137, f136;
neg.f32 f94, f137;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r2545, {low, high};
}
mul.lo.s32 r3489, r3488, 49;
sub.s32 r3490, r3474, r3489;
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2548, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2550, {high, high};
}
{
mul.f16x2 r2552, r2272, r2550;
}
{
neg.f16x2 r2555, r2552;
}
{
fma.rn.f16x2 r2557, r1948, r2548, r2555;
}
{
mul.f16x2 r2561, r1948, r2550;
}
{
fma.rn.f16x2 r2564, r2272, r2548, r2561;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2568, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2570, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r2572, {low, high};
}
{
mul.f16x2 r2573, r2570, r2572;
}
{
mul.f16x2 r2576, r2545, r2568;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2579, {high, low};
}
{
fma.rn.f16x2 r2581, r2573, r2579, r2576;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2581;
mov.b32 r2585, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2581;
mov.b32 r2587, {high, high};
}
{
mul.f16x2 r2589, r2380, r2587;
}
{
neg.f16x2 r2592, r2589;
}
{
fma.rn.f16x2 r2594, r2056, r2585, r2592;
}
{
mul.f16x2 r2598, r2056, r2587;
}
{
fma.rn.f16x2 r2601, r2380, r2585, r2598;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2605, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2607, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r2609, {low, high};
}
{
mul.f16x2 r2610, r2607, r2609;
}
{
mul.f16x2 r2613, r2581, r2605;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2581;
mov.b32 r2616, {high, low};
}
{
fma.rn.f16x2 r2618, r2610, r2616, r2613;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2618;
mov.b32 r2622, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2618;
mov.b32 r2624, {high, high};
}
{
mul.f16x2 r2626, r2488, r2624;
}
{
neg.f16x2 r2629, r2626;
}
{
fma.rn.f16x2 r2631, r2164, r2622, r2629;
}
{
mul.f16x2 r2635, r2164, r2624;
}
{
fma.rn.f16x2 r2638, r2488, r2622, r2635;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2642, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2644, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r2646, {low, high};
}
{
mul.f16x2 r2647, r2644, r2646;
}
{
mul.f16x2 r2650, r2618, r2642;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2618;
mov.b32 r2653, {high, low};
}
{
fma.rn.f16x2 r2655, r2647, r2653, r2650;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2655;
mov.b32 r2659, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2655;
mov.b32 r2661, {high, high};
}
{
mul.f16x2 r2663, r2542, r2661;
}
{
neg.f16x2 r2666, r2663;
}
{
fma.rn.f16x2 r2668, r2218, r2659, r2666;
}
{
mul.f16x2 r2672, r2218, r2661;
}
{
fma.rn.f16x2 r2675, r2542, r2659, r2672;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2679, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2681, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r2683, {low, high};
}
{
mul.f16x2 r2684, r2681, r2683;
}
{
mul.f16x2 r2687, r2655, r2679;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2655;
mov.b32 r2690, {high, low};
}
{
fma.rn.f16x2 r2692, r2684, r2690, r2687;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2692;
mov.b32 r2696, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2692;
mov.b32 r2698, {high, high};
}
{
mul.f16x2 r2700, r2434, r2698;
}
{
neg.f16x2 r2703, r2700;
}
{
fma.rn.f16x2 r2705, r2110, r2696, r2703;
}
{
mul.f16x2 r2709, r2110, r2698;
}
{
fma.rn.f16x2 r2712, r2434, r2696, r2709;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2716, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2718, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r2720, {low, high};
}
{
mul.f16x2 r2721, r2718, r2720;
}
{
mul.f16x2 r2724, r2692, r2716;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2692;
mov.b32 r2727, {high, low};
}
{
fma.rn.f16x2 r2729, r2721, r2727, r2724;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2729;
mov.b32 r2733, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2729;
mov.b32 r2735, {high, high};
}
{
mul.f16x2 r2737, r2326, r2735;
}
{
neg.f16x2 r2740, r2737;
}
{
fma.rn.f16x2 r2742, r2002, r2733, r2740;
}
{
mul.f16x2 r2746, r2002, r2735;
}
{
fma.rn.f16x2 r2749, r2326, r2733, r2746;
}
shl.b32 r3491, r3490, 3;
add.s32 r3492, r3475, r3491;
barrier.sync 0;
mad.lo.s32 r3493, r3488, 2744, r3492;
st.shared.u32 [r3493], r1876;
st.shared.u32 [r3493+4], r1894;
st.shared.u32 [r3493+392], r2557;
st.shared.u32 [r3493+396], r2564;
st.shared.u32 [r3493+784], r2594;
st.shared.u32 [r3493+788], r2601;
st.shared.u32 [r3493+1176], r2631;
st.shared.u32 [r3493+1180], r2638;
st.shared.u32 [r3493+1568], r2668;
st.shared.u32 [r3493+1572], r2675;
st.shared.u32 [r3493+1960], r2705;
st.shared.u32 [r3493+1964], r2712;
st.shared.u32 [r3493+2352], r2742;
st.shared.u32 [r3493+2356], r2749;
barrier.sync 0;
ld.shared.u32 r2788, [r3477];
ld.shared.u32 r2806, [r3477+4];
ld.shared.u32 r2785, [r3477+2744];
ld.shared.u32 r2803, [r3477+2748];
ld.shared.u32 r2791, [r3477+5488];
ld.shared.u32 r2809, [r3477+5492];
ld.shared.u32 r2797, [r3477+8232];
ld.shared.u32 r2815, [r3477+8236];
ld.shared.u32 r2798, [r3477+10976];
ld.shared.u32 r2816, [r3477+10980];
ld.shared.u32 r2792, [r3477+13720];
ld.shared.u32 r2810, [r3477+13724];
ld.shared.u32 r2786, [r3477+16464];
ld.shared.u32 r2804, [r3477+16468];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f126;
cvt.rn.f16.f32 high, f126;
mov.b32 r2770, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f128;
cvt.rn.f16.f32 high, f128;
mov.b32 r2771, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f114;
cvt.rn.f16.f32 high, f114;
mov.b32 r2772, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f116;
cvt.rn.f16.f32 high, f116;
mov.b32 r2773, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r2774, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r2775, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r2776, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r2777, {low, high};
}
{
neg.f16x2 r2778, r2777;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f126;
cvt.rn.f16.f32 high, f126;
mov.b32 r2780, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f128;
cvt.rn.f16.f32 high, f128;
mov.b32 r2781, {low, high};
}
{
neg.f16x2 r2782, r2781;
}
{
add.f16x2 r2784, r2785, r2786;
}
{
add.f16x2 r2787, r2788, r2784;
}
{
add.f16x2 r2790, r2791, r2792;
}
{
add.f16x2 r2793, r2787, r2790;
}
{
add.f16x2 r2796, r2797, r2798;
}
{
add.f16x2 %0, r2793, r2796;
}
{
add.f16x2 r2802, r2803, r2804;
}
{
add.f16x2 r2805, r2806, r2802;
}
{
add.f16x2 r2808, r2809, r2810;
}
{
add.f16x2 r2811, r2805, r2808;
}
{
add.f16x2 r2814, r2815, r2816;
}
{
add.f16x2 %1, r2811, r2814;
}
{
add.f16x2 r2820, r2785, r2786;
}
{
mul.f16x2 r2823, r2820, r2770;
}
{
add.f16x2 r2826, r2788, r2823;
}
{
add.f16x2 r2829, r2791, r2792;
}
{
mul.f16x2 r2832, r2829, r2772;
}
{
add.f16x2 r2835, r2826, r2832;
}
{
add.f16x2 r2838, r2797, r2798;
}
{
mul.f16x2 r2841, r2838, r2774;
}
{
add.f16x2 r2844, r2835, r2841;
}
{
sub.f16x2 r2847, r2803, r2804;
}
{
mul.f16x2 r2850, r2847, r2771;
}
{
sub.f16x2 r2853, r2809, r2810;
}
{
mul.f16x2 r2856, r2853, r2773;
}
{
add.f16x2 r2859, r2850, r2856;
}
{
sub.f16x2 r2862, r2815, r2816;
}
{
mul.f16x2 r2865, r2862, r2775;
}
{
add.f16x2 r2868, r2859, r2865;
}
{
sub.f16x2 %2, r2844, r2868;
}
{
add.f16x2 r2874, r2785, r2786;
}
{
mul.f16x2 r2877, r2874, r2770;
}
{
add.f16x2 r2880, r2788, r2877;
}
{
add.f16x2 r2883, r2791, r2792;
}
{
mul.f16x2 r2886, r2883, r2772;
}
{
add.f16x2 r2889, r2880, r2886;
}
{
add.f16x2 r2892, r2797, r2798;
}
{
mul.f16x2 r2895, r2892, r2774;
}
{
add.f16x2 r2898, r2889, r2895;
}
{
sub.f16x2 r2901, r2803, r2804;
}
{
mul.f16x2 r2904, r2901, r2771;
}
{
sub.f16x2 r2907, r2809, r2810;
}
{
mul.f16x2 r2910, r2907, r2773;
}
{
add.f16x2 r2913, r2904, r2910;
}
{
sub.f16x2 r2916, r2815, r2816;
}
{
mul.f16x2 r2919, r2916, r2775;
}
{
add.f16x2 r2922, r2913, r2919;
}
{
add.f16x2 %12, r2898, r2922;
}
{
add.f16x2 r2928, r2785, r2786;
}
{
mul.f16x2 r2931, r2928, r2772;
}
{
add.f16x2 r2934, r2788, r2931;
}
{
add.f16x2 r2937, r2791, r2792;
}
{
mul.f16x2 r2940, r2937, r2776;
}
{
add.f16x2 r2943, r2934, r2940;
}
{
add.f16x2 r2946, r2797, r2798;
}
{
mul.f16x2 r2949, r2946, r2780;
}
{
add.f16x2 r2952, r2943, r2949;
}
{
sub.f16x2 r2955, r2803, r2804;
}
{
mul.f16x2 r2958, r2955, r2773;
}
{
sub.f16x2 r2961, r2809, r2810;
}
{
mul.f16x2 r2964, r2961, r2778;
}
{
add.f16x2 r2967, r2958, r2964;
}
{
sub.f16x2 r2970, r2815, r2816;
}
{
mul.f16x2 r2973, r2970, r2782;
}
{
add.f16x2 r2976, r2967, r2973;
}
{
sub.f16x2 %4, r2952, r2976;
}
{
add.f16x2 r2982, r2785, r2786;
}
{
mul.f16x2 r2985, r2982, r2772;
}
{
add.f16x2 r2988, r2788, r2985;
}
{
add.f16x2 r2991, r2791, r2792;
}
{
mul.f16x2 r2994, r2991, r2776;
}
{
add.f16x2 r2997, r2988, r2994;
}
{
add.f16x2 r3000, r2797, r2798;
}
{
mul.f16x2 r3003, r3000, r2780;
}
{
add.f16x2 r3006, r2997, r3003;
}
{
sub.f16x2 r3009, r2803, r2804;
}
{
mul.f16x2 r3012, r3009, r2773;
}
{
sub.f16x2 r3015, r2809, r2810;
}
{
mul.f16x2 r3018, r3015, r2778;
}
{
add.f16x2 r3021, r3012, r3018;
}
{
sub.f16x2 r3024, r2815, r2816;
}
{
mul.f16x2 r3027, r3024, r2782;
}
{
add.f16x2 r3030, r3021, r3027;
}
{
add.f16x2 %10, r3006, r3030;
}
{
add.f16x2 r3036, r2785, r2786;
}
{
mul.f16x2 r3039, r3036, r2774;
}
{
add.f16x2 r3042, r2788, r3039;
}
{
add.f16x2 r3045, r2791, r2792;
}
{
mul.f16x2 r3048, r3045, r2780;
}
{
add.f16x2 r3051, r3042, r3048;
}
{
add.f16x2 r3054, r2797, r2798;
}
{
mul.f16x2 r3057, r3054, r2772;
}
{
add.f16x2 r3060, r3051, r3057;
}
{
sub.f16x2 r3063, r2803, r2804;
}
{
mul.f16x2 r3066, r3063, r2775;
}
{
sub.f16x2 r3069, r2809, r2810;
}
{
mul.f16x2 r3072, r3069, r2782;
}
{
add.f16x2 r3075, r3066, r3072;
}
{
sub.f16x2 r3078, r2815, r2816;
}
{
mul.f16x2 r3081, r3078, r2773;
}
{
add.f16x2 r3084, r3075, r3081;
}
{
sub.f16x2 %6, r3060, r3084;
}
{
add.f16x2 r3090, r2785, r2786;
}
{
mul.f16x2 r3093, r3090, r2774;
}
{
add.f16x2 r3096, r2788, r3093;
}
{
add.f16x2 r3099, r2791, r2792;
}
{
mul.f16x2 r3102, r3099, r2780;
}
{
add.f16x2 r3105, r3096, r3102;
}
{
add.f16x2 r3108, r2797, r2798;
}
{
mul.f16x2 r3111, r3108, r2772;
}
{
add.f16x2 r3114, r3105, r3111;
}
{
sub.f16x2 r3117, r2803, r2804;
}
{
mul.f16x2 r3120, r3117, r2775;
}
{
sub.f16x2 r3123, r2809, r2810;
}
{
mul.f16x2 r3126, r3123, r2782;
}
{
add.f16x2 r3129, r3120, r3126;
}
{
sub.f16x2 r3132, r2815, r2816;
}
{
mul.f16x2 r3135, r3132, r2773;
}
{
add.f16x2 r3138, r3129, r3135;
}
{
add.f16x2 %8, r3114, r3138;
}
{
add.f16x2 r3144, r2803, r2804;
}
{
mul.f16x2 r3147, r3144, r2770;
}
{
add.f16x2 r3150, r2806, r3147;
}
{
add.f16x2 r3153, r2809, r2810;
}
{
mul.f16x2 r3156, r3153, r2772;
}
{
add.f16x2 r3159, r3150, r3156;
}
{
add.f16x2 r3162, r2815, r2816;
}
{
mul.f16x2 r3165, r3162, r2774;
}
{
add.f16x2 r3168, r3159, r3165;
}
{
sub.f16x2 r3171, r2785, r2786;
}
{
mul.f16x2 r3174, r3171, r2771;
}
{
sub.f16x2 r3177, r2791, r2792;
}
{
mul.f16x2 r3180, r3177, r2773;
}
{
add.f16x2 r3183, r3174, r3180;
}
{
sub.f16x2 r3186, r2797, r2798;
}
{
mul.f16x2 r3189, r3186, r2775;
}
{
add.f16x2 r3192, r3183, r3189;
}
{
add.f16x2 %3, r3168, r3192;
}
{
add.f16x2 r3198, r2803, r2804;
}
{
mul.f16x2 r3201, r3198, r2770;
}
{
add.f16x2 r3204, r2806, r3201;
}
{
add.f16x2 r3207, r2809, r2810;
}
{
mul.f16x2 r3210, r3207, r2772;
}
{
add.f16x2 r3213, r3204, r3210;
}
{
add.f16x2 r3216, r2815, r2816;
}
{
mul.f16x2 r3219, r3216, r2774;
}
{
add.f16x2 r3222, r3213, r3219;
}
{
sub.f16x2 r3225, r2785, r2786;
}
{
mul.f16x2 r3228, r3225, r2771;
}
{
sub.f16x2 r3231, r2791, r2792;
}
{
mul.f16x2 r3234, r3231, r2773;
}
{
add.f16x2 r3237, r3228, r3234;
}
{
sub.f16x2 r3240, r2797, r2798;
}
{
mul.f16x2 r3243, r3240, r2775;
}
{
add.f16x2 r3246, r3237, r3243;
}
{
sub.f16x2 %13, r3222, r3246;
}
{
add.f16x2 r3252, r2803, r2804;
}
{
mul.f16x2 r3255, r3252, r2772;
}
{
add.f16x2 r3258, r2806, r3255;
}
{
add.f16x2 r3261, r2809, r2810;
}
{
mul.f16x2 r3264, r3261, r2776;
}
{
add.f16x2 r3267, r3258, r3264;
}
{
add.f16x2 r3270, r2815, r2816;
}
{
mul.f16x2 r3273, r3270, r2780;
}
{
add.f16x2 r3276, r3267, r3273;
}
{
sub.f16x2 r3279, r2785, r2786;
}
{
mul.f16x2 r3282, r3279, r2773;
}
{
sub.f16x2 r3285, r2791, r2792;
}
{
mul.f16x2 r3288, r3285, r2778;
}
{
add.f16x2 r3291, r3282, r3288;
}
{
sub.f16x2 r3294, r2797, r2798;
}
{
mul.f16x2 r3297, r3294, r2782;
}
{
add.f16x2 r3300, r3291, r3297;
}
{
add.f16x2 %5, r3276, r3300;
}
{
add.f16x2 r3306, r2803, r2804;
}
{
mul.f16x2 r3309, r3306, r2772;
}
{
add.f16x2 r3312, r2806, r3309;
}
{
add.f16x2 r3315, r2809, r2810;
}
{
mul.f16x2 r3318, r3315, r2776;
}
{
add.f16x2 r3321, r3312, r3318;
}
{
add.f16x2 r3324, r2815, r2816;
}
{
mul.f16x2 r3327, r3324, r2780;
}
{
add.f16x2 r3330, r3321, r3327;
}
{
sub.f16x2 r3333, r2785, r2786;
}
{
mul.f16x2 r3336, r3333, r2773;
}
{
sub.f16x2 r3339, r2791, r2792;
}
{
mul.f16x2 r3342, r3339, r2778;
}
{
add.f16x2 r3345, r3336, r3342;
}
{
sub.f16x2 r3348, r2797, r2798;
}
{
mul.f16x2 r3351, r3348, r2782;
}
{
add.f16x2 r3354, r3345, r3351;
}
{
sub.f16x2 %11, r3330, r3354;
}
{
add.f16x2 r3360, r2803, r2804;
}
{
mul.f16x2 r3363, r3360, r2774;
}
{
add.f16x2 r3366, r2806, r3363;
}
{
add.f16x2 r3369, r2809, r2810;
}
{
mul.f16x2 r3372, r3369, r2780;
}
{
add.f16x2 r3375, r3366, r3372;
}
{
add.f16x2 r3378, r2815, r2816;
}
{
mul.f16x2 r3381, r3378, r2772;
}
{
add.f16x2 r3384, r3375, r3381;
}
{
sub.f16x2 r3387, r2785, r2786;
}
{
mul.f16x2 r3390, r3387, r2775;
}
{
sub.f16x2 r3393, r2791, r2792;
}
{
mul.f16x2 r3396, r3393, r2782;
}
{
add.f16x2 r3399, r3390, r3396;
}
{
sub.f16x2 r3402, r2797, r2798;
}
{
mul.f16x2 r3405, r3402, r2773;
}
{
add.f16x2 r3408, r3399, r3405;
}
{
add.f16x2 %7, r3384, r3408;
}
{
add.f16x2 r3414, r2803, r2804;
}
{
mul.f16x2 r3417, r3414, r2774;
}
{
add.f16x2 r3420, r2806, r3417;
}
{
add.f16x2 r3423, r2809, r2810;
}
{
mul.f16x2 r3426, r3423, r2780;
}
{
add.f16x2 r3429, r3420, r3426;
}
{
add.f16x2 r3432, r2815, r2816;
}
{
mul.f16x2 r3435, r3432, r2772;
}
{
add.f16x2 r3438, r3429, r3435;
}
{
sub.f16x2 r3441, r2785, r2786;
}
{
mul.f16x2 r3444, r3441, r2775;
}
{
sub.f16x2 r3447, r2791, r2792;
}
{
mul.f16x2 r3450, r3447, r2782;
}
{
add.f16x2 r3453, r3444, r3450;
}
{
sub.f16x2 r3456, r2797, r2798;
}
{
mul.f16x2 r3459, r3456, r2773;
}
{
add.f16x2 r3462, r3453, r3459;
}
{
sub.f16x2 %9, r3438, r3462;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<923, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<138>;
.reg .b32 r<3494>;
.reg .b64 rd<8>;
mov.u32 r3468, %tid.y;
mov.u32 r3469, %14;
mad.lo.s32 r3470, r3468, 9604, r3469;
mov.u32 r3471, %tid.x;
mov.f32 f126, 0f3F1F9D07;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f126;
cvt.rn.f16.f32 high, f126;
mov.b32 r1, {low, high};
}
mov.f32 f128, 0fBF48261C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f128;
cvt.rn.f16.f32 high, f128;
mov.b32 r2, {low, high};
}
mov.f32 f114, 0fBE63DC87;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f114;
cvt.rn.f16.f32 high, f114;
mov.b32 r3, {low, high};
}
mov.f32 f116, 0fBF7994E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f116;
cvt.rn.f16.f32 high, f116;
mov.b32 r4, {low, high};
}
mov.f32 f122, 0fBF66A5E5;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r5, {low, high};
}
mov.f32 f124, 0fBEDE2602;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r6, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r7, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r8, {low, high};
}
{
neg.f16x2 r9, r8;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f126;
cvt.rn.f16.f32 high, f126;
mov.b32 r11, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f128;
cvt.rn.f16.f32 high, f128;
mov.b32 r12, {low, high};
}
{
neg.f16x2 r13, r12;
}
{
add.f16x2 r15, %17, %27;
}
{
add.f16x2 r18, %15, r15;
}
{
add.f16x2 r21, %19, %25;
}
{
add.f16x2 r24, r18, r21;
}
{
add.f16x2 r27, %21, %23;
}
{
add.f16x2 r30, r24, r27;
}
{
add.f16x2 r33, %18, %28;
}
{
add.f16x2 r36, %16, r33;
}
{
add.f16x2 r39, %20, %26;
}
{
add.f16x2 r42, r36, r39;
}
{
add.f16x2 r45, %22, %24;
}
{
add.f16x2 r48, r42, r45;
}
{
add.f16x2 r51, %17, %27;
}
{
mul.f16x2 r54, r51, r1;
}
{
add.f16x2 r57, %15, r54;
}
{
add.f16x2 r60, %19, %25;
}
{
mul.f16x2 r63, r60, r3;
}
{
add.f16x2 r66, r57, r63;
}
{
add.f16x2 r69, %21, %23;
}
{
mul.f16x2 r72, r69, r5;
}
{
add.f16x2 r75, r66, r72;
}
{
sub.f16x2 r78, %18, %28;
}
{
mul.f16x2 r81, r78, r2;
}
{
sub.f16x2 r84, %20, %26;
}
{
mul.f16x2 r87, r84, r4;
}
{
add.f16x2 r90, r81, r87;
}
{
sub.f16x2 r93, %22, %24;
}
{
mul.f16x2 r96, r93, r6;
}
{
add.f16x2 r99, r90, r96;
}
{
sub.f16x2 r102, r75, r99;
}
{
add.f16x2 r105, %17, %27;
}
{
mul.f16x2 r108, r105, r1;
}
{
add.f16x2 r111, %15, r108;
}
{
add.f16x2 r114, %19, %25;
}
{
mul.f16x2 r117, r114, r3;
}
{
add.f16x2 r120, r111, r117;
}
{
add.f16x2 r123, %21, %23;
}
{
mul.f16x2 r126, r123, r5;
}
{
add.f16x2 r129, r120, r126;
}
{
sub.f16x2 r132, %18, %28;
}
{
mul.f16x2 r135, r132, r2;
}
{
sub.f16x2 r138, %20, %26;
}
{
mul.f16x2 r141, r138, r4;
}
{
add.f16x2 r144, r135, r141;
}
{
sub.f16x2 r147, %22, %24;
}
{
mul.f16x2 r150, r147, r6;
}
{
add.f16x2 r153, r144, r150;
}
{
add.f16x2 r156, r129, r153;
}
{
add.f16x2 r159, %17, %27;
}
{
mul.f16x2 r162, r159, r3;
}
{
add.f16x2 r165, %15, r162;
}
{
add.f16x2 r168, %19, %25;
}
{
mul.f16x2 r171, r168, r7;
}
{
add.f16x2 r174, r165, r171;
}
{
add.f16x2 r177, %21, %23;
}
{
mul.f16x2 r180, r177, r11;
}
{
add.f16x2 r183, r174, r180;
}
{
sub.f16x2 r186, %18, %28;
}
{
mul.f16x2 r189, r186, r4;
}
{
sub.f16x2 r192, %20, %26;
}
{
mul.f16x2 r195, r192, r9;
}
{
add.f16x2 r198, r189, r195;
}
{
sub.f16x2 r201, %22, %24;
}
{
mul.f16x2 r204, r201, r13;
}
{
add.f16x2 r207, r198, r204;
}
{
sub.f16x2 r210, r183, r207;
}
{
add.f16x2 r213, %17, %27;
}
{
mul.f16x2 r216, r213, r3;
}
{
add.f16x2 r219, %15, r216;
}
{
add.f16x2 r222, %19, %25;
}
{
mul.f16x2 r225, r222, r7;
}
{
add.f16x2 r228, r219, r225;
}
{
add.f16x2 r231, %21, %23;
}
{
mul.f16x2 r234, r231, r11;
}
{
add.f16x2 r237, r228, r234;
}
{
sub.f16x2 r240, %18, %28;
}
{
mul.f16x2 r243, r240, r4;
}
{
sub.f16x2 r246, %20, %26;
}
{
mul.f16x2 r249, r246, r9;
}
{
add.f16x2 r252, r243, r249;
}
{
sub.f16x2 r255, %22, %24;
}
{
mul.f16x2 r258, r255, r13;
}
{
add.f16x2 r261, r252, r258;
}
{
add.f16x2 r264, r237, r261;
}
{
add.f16x2 r267, %17, %27;
}
{
mul.f16x2 r270, r267, r5;
}
{
add.f16x2 r273, %15, r270;
}
{
add.f16x2 r276, %19, %25;
}
{
mul.f16x2 r279, r276, r11;
}
{
add.f16x2 r282, r273, r279;
}
{
add.f16x2 r285, %21, %23;
}
{
mul.f16x2 r288, r285, r3;
}
{
add.f16x2 r291, r282, r288;
}
{
sub.f16x2 r294, %18, %28;
}
{
mul.f16x2 r297, r294, r6;
}
{
sub.f16x2 r300, %20, %26;
}
{
mul.f16x2 r303, r300, r13;
}
{
add.f16x2 r306, r297, r303;
}
{
sub.f16x2 r309, %22, %24;
}
{
mul.f16x2 r312, r309, r4;
}
{
add.f16x2 r315, r306, r312;
}
{
sub.f16x2 r318, r291, r315;
}
{
add.f16x2 r321, %17, %27;
}
{
mul.f16x2 r324, r321, r5;
}
{
add.f16x2 r327, %15, r324;
}
{
add.f16x2 r330, %19, %25;
}
{
mul.f16x2 r333, r330, r11;
}
{
add.f16x2 r336, r327, r333;
}
{
add.f16x2 r339, %21, %23;
}
{
mul.f16x2 r342, r339, r3;
}
{
add.f16x2 r345, r336, r342;
}
{
sub.f16x2 r348, %18, %28;
}
{
mul.f16x2 r351, r348, r6;
}
{
sub.f16x2 r354, %20, %26;
}
{
mul.f16x2 r357, r354, r13;
}
{
add.f16x2 r360, r351, r357;
}
{
sub.f16x2 r363, %22, %24;
}
{
mul.f16x2 r366, r363, r4;
}
{
add.f16x2 r369, r360, r366;
}
{
add.f16x2 r372, r345, r369;
}
{
add.f16x2 r375, %18, %28;
}
{
mul.f16x2 r378, r375, r1;
}
{
add.f16x2 r381, %16, r378;
}
{
add.f16x2 r384, %20, %26;
}
{
mul.f16x2 r387, r384, r3;
}
{
add.f16x2 r390, r381, r387;
}
{
add.f16x2 r393, %22, %24;
}
{
mul.f16x2 r396, r393, r5;
}
{
add.f16x2 r399, r390, r396;
}
{
sub.f16x2 r402, %17, %27;
}
{
mul.f16x2 r405, r402, r2;
}
{
sub.f16x2 r408, %19, %25;
}
{
mul.f16x2 r411, r408, r4;
}
{
add.f16x2 r414, r405, r411;
}
{
sub.f16x2 r417, %21, %23;
}
{
mul.f16x2 r420, r417, r6;
}
{
add.f16x2 r423, r414, r420;
}
{
add.f16x2 r426, r399, r423;
}
{
add.f16x2 r429, %18, %28;
}
{
mul.f16x2 r432, r429, r1;
}
{
add.f16x2 r435, %16, r432;
}
{
add.f16x2 r438, %20, %26;
}
{
mul.f16x2 r441, r438, r3;
}
{
add.f16x2 r444, r435, r441;
}
{
add.f16x2 r447, %22, %24;
}
{
mul.f16x2 r450, r447, r5;
}
{
add.f16x2 r453, r444, r450;
}
{
sub.f16x2 r456, %17, %27;
}
{
mul.f16x2 r459, r456, r2;
}
{
sub.f16x2 r462, %19, %25;
}
{
mul.f16x2 r465, r462, r4;
}
{
add.f16x2 r468, r459, r465;
}
{
sub.f16x2 r471, %21, %23;
}
{
mul.f16x2 r474, r471, r6;
}
{
add.f16x2 r477, r468, r474;
}
{
sub.f16x2 r480, r453, r477;
}
{
add.f16x2 r483, %18, %28;
}
{
mul.f16x2 r486, r483, r3;
}
{
add.f16x2 r489, %16, r486;
}
{
add.f16x2 r492, %20, %26;
}
{
mul.f16x2 r495, r492, r7;
}
{
add.f16x2 r498, r489, r495;
}
{
add.f16x2 r501, %22, %24;
}
{
mul.f16x2 r504, r501, r11;
}
{
add.f16x2 r507, r498, r504;
}
{
sub.f16x2 r510, %17, %27;
}
{
mul.f16x2 r513, r510, r4;
}
{
sub.f16x2 r516, %19, %25;
}
{
mul.f16x2 r519, r516, r9;
}
{
add.f16x2 r522, r513, r519;
}
{
sub.f16x2 r525, %21, %23;
}
{
mul.f16x2 r528, r525, r13;
}
{
add.f16x2 r531, r522, r528;
}
{
add.f16x2 r534, r507, r531;
}
{
add.f16x2 r537, %18, %28;
}
{
mul.f16x2 r540, r537, r3;
}
{
add.f16x2 r543, %16, r540;
}
{
add.f16x2 r546, %20, %26;
}
{
mul.f16x2 r549, r546, r7;
}
{
add.f16x2 r552, r543, r549;
}
{
add.f16x2 r555, %22, %24;
}
{
mul.f16x2 r558, r555, r11;
}
{
add.f16x2 r561, r552, r558;
}
{
sub.f16x2 r564, %17, %27;
}
{
mul.f16x2 r567, r564, r4;
}
{
sub.f16x2 r570, %19, %25;
}
{
mul.f16x2 r573, r570, r9;
}
{
add.f16x2 r576, r567, r573;
}
{
sub.f16x2 r579, %21, %23;
}
{
mul.f16x2 r582, r579, r13;
}
{
add.f16x2 r585, r576, r582;
}
{
sub.f16x2 r588, r561, r585;
}
{
add.f16x2 r591, %18, %28;
}
{
mul.f16x2 r594, r591, r5;
}
{
add.f16x2 r597, %16, r594;
}
{
add.f16x2 r600, %20, %26;
}
{
mul.f16x2 r603, r600, r11;
}
{
add.f16x2 r606, r597, r603;
}
{
add.f16x2 r609, %22, %24;
}
{
mul.f16x2 r612, r609, r3;
}
{
add.f16x2 r615, r606, r612;
}
{
sub.f16x2 r618, %17, %27;
}
{
mul.f16x2 r621, r618, r6;
}
{
sub.f16x2 r624, %19, %25;
}
{
mul.f16x2 r627, r624, r13;
}
{
add.f16x2 r630, r621, r627;
}
{
sub.f16x2 r633, %21, %23;
}
{
mul.f16x2 r636, r633, r4;
}
{
add.f16x2 r639, r630, r636;
}
{
add.f16x2 r642, r615, r639;
}
{
add.f16x2 r645, %18, %28;
}
{
mul.f16x2 r648, r645, r5;
}
{
add.f16x2 r651, %16, r648;
}
{
add.f16x2 r654, %20, %26;
}
{
mul.f16x2 r657, r654, r11;
}
{
add.f16x2 r660, r651, r657;
}
{
add.f16x2 r663, %22, %24;
}
{
mul.f16x2 r666, r663, r3;
}
{
add.f16x2 r669, r660, r666;
}
{
sub.f16x2 r672, %17, %27;
}
{
mul.f16x2 r675, r672, r6;
}
{
sub.f16x2 r678, %19, %25;
}
{
mul.f16x2 r681, r678, r13;
}
{
add.f16x2 r684, r675, r681;
}
{
sub.f16x2 r687, %21, %23;
}
{
mul.f16x2 r690, r687, r4;
}
{
add.f16x2 r693, r684, r690;
}
{
sub.f16x2 r696, r669, r693;
}
mul.wide.u32 rd2, r3471, -1089394037;
shr.u64 rd3, rd2, 40;
cvt.u32.u64 r3472, rd3;
mul.lo.s32 r3473, r3472, 343;
sub.s32 r3474, r3471, r3473;
cvt.rn.f32.u32 f129, r3474;
mul.f32 f130, f129, 0f3B2B805B;
cos.approx.f32 f21, f130;
sin.approx.f32 f131, f130;
neg.f32 f22, f131;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f21;
cvt.rn.f16.f32 high, f22;
mov.b32 r699, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r702, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r704, {high, high};
}
{
mul.f16x2 r706, r426, r704;
}
{
neg.f16x2 r709, r706;
}
{
fma.rn.f16x2 r711, r102, r702, r709;
}
{
mul.f16x2 r715, r102, r704;
}
{
fma.rn.f16x2 r718, r426, r702, r715;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r722, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r724, {high, high};
}
mov.f32 f105, 0fBF800000;
mov.f32 f106, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r726, {low, high};
}
{
mul.f16x2 r727, r724, r726;
}
{
mul.f16x2 r730, r699, r722;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r733, {high, low};
}
{
fma.rn.f16x2 r735, r727, r733, r730;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r735;
mov.b32 r739, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r735;
mov.b32 r741, {high, high};
}
{
mul.f16x2 r743, r534, r741;
}
{
neg.f16x2 r746, r743;
}
{
fma.rn.f16x2 r748, r210, r739, r746;
}
{
mul.f16x2 r752, r210, r741;
}
{
fma.rn.f16x2 r755, r534, r739, r752;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r759, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r761, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r763, {low, high};
}
{
mul.f16x2 r764, r761, r763;
}
{
mul.f16x2 r767, r735, r759;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r735;
mov.b32 r770, {high, low};
}
{
fma.rn.f16x2 r772, r764, r770, r767;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r772;
mov.b32 r776, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r772;
mov.b32 r778, {high, high};
}
{
mul.f16x2 r780, r642, r778;
}
{
neg.f16x2 r783, r780;
}
{
fma.rn.f16x2 r785, r318, r776, r783;
}
{
mul.f16x2 r789, r318, r778;
}
{
fma.rn.f16x2 r792, r642, r776, r789;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r796, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r798, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r800, {low, high};
}
{
mul.f16x2 r801, r798, r800;
}
{
mul.f16x2 r804, r772, r796;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r772;
mov.b32 r807, {high, low};
}
{
fma.rn.f16x2 r809, r801, r807, r804;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r809;
mov.b32 r813, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r809;
mov.b32 r815, {high, high};
}
{
mul.f16x2 r817, r696, r815;
}
{
neg.f16x2 r820, r817;
}
{
fma.rn.f16x2 r822, r372, r813, r820;
}
{
mul.f16x2 r826, r372, r815;
}
{
fma.rn.f16x2 r829, r696, r813, r826;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r833, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r835, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r837, {low, high};
}
{
mul.f16x2 r838, r835, r837;
}
{
mul.f16x2 r841, r809, r833;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r809;
mov.b32 r844, {high, low};
}
{
fma.rn.f16x2 r846, r838, r844, r841;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r846;
mov.b32 r850, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r846;
mov.b32 r852, {high, high};
}
{
mul.f16x2 r854, r588, r852;
}
{
neg.f16x2 r857, r854;
}
{
fma.rn.f16x2 r859, r264, r850, r857;
}
{
mul.f16x2 r863, r264, r852;
}
{
fma.rn.f16x2 r866, r588, r850, r863;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r870, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r699;
mov.b32 r872, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r874, {low, high};
}
{
mul.f16x2 r875, r872, r874;
}
{
mul.f16x2 r878, r846, r870;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r846;
mov.b32 r881, {high, low};
}
{
fma.rn.f16x2 r883, r875, r881, r878;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r883;
mov.b32 r887, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r883;
mov.b32 r889, {high, high};
}
{
mul.f16x2 r891, r480, r889;
}
{
neg.f16x2 r894, r891;
}
{
fma.rn.f16x2 r896, r156, r887, r894;
}
{
mul.f16x2 r900, r156, r889;
}
{
fma.rn.f16x2 r903, r480, r887, r900;
}
mad.lo.s32 r3475, r3472, 9604, r3470;
barrier.sync 0;
mad.lo.s32 r3476, r3474, 28, r3475;
st.shared.u32 [r3476], r30;
st.shared.u32 [r3476+4], r711;
st.shared.u32 [r3476+8], r748;
st.shared.u32 [r3476+12], r785;
st.shared.u32 [r3476+16], r822;
st.shared.u32 [r3476+20], r859;
st.shared.u32 [r3476+24], r896;
barrier.sync 0;
mad.lo.s32 r3477, r3474, -24, r3476;
ld.shared.u32 r942, [r3477];
ld.shared.u32 r939, [r3477+1372];
ld.shared.u32 r945, [r3477+2744];
ld.shared.u32 r951, [r3477+4116];
ld.shared.u32 r952, [r3477+5488];
ld.shared.u32 r946, [r3477+6860];
ld.shared.u32 r940, [r3477+8232];
barrier.sync 0;
st.shared.u32 [r3476], r48;
st.shared.u32 [r3476+4], r718;
st.shared.u32 [r3476+8], r755;
st.shared.u32 [r3476+12], r792;
st.shared.u32 [r3476+16], r829;
st.shared.u32 [r3476+20], r866;
st.shared.u32 [r3476+24], r903;
barrier.sync 0;
ld.shared.u32 r960, [r3477];
ld.shared.u32 r957, [r3477+1372];
ld.shared.u32 r963, [r3477+2744];
ld.shared.u32 r969, [r3477+4116];
ld.shared.u32 r970, [r3477+5488];
ld.shared.u32 r964, [r3477+6860];
ld.shared.u32 r958, [r3477+8232];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f126;
cvt.rn.f16.f32 high, f126;
mov.b32 r924, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f128;
cvt.rn.f16.f32 high, f128;
mov.b32 r925, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f114;
cvt.rn.f16.f32 high, f114;
mov.b32 r926, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f116;
cvt.rn.f16.f32 high, f116;
mov.b32 r927, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r928, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r929, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r930, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r931, {low, high};
}
{
neg.f16x2 r932, r931;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f126;
cvt.rn.f16.f32 high, f126;
mov.b32 r934, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f128;
cvt.rn.f16.f32 high, f128;
mov.b32 r935, {low, high};
}
{
neg.f16x2 r936, r935;
}
{
add.f16x2 r938, r939, r940;
}
{
add.f16x2 r941, r942, r938;
}
{
add.f16x2 r944, r945, r946;
}
{
add.f16x2 r947, r941, r944;
}
{
add.f16x2 r950, r951, r952;
}
{
add.f16x2 r953, r947, r950;
}
{
add.f16x2 r956, r957, r958;
}
{
add.f16x2 r959, r960, r956;
}
{
add.f16x2 r962, r963, r964;
}
{
add.f16x2 r965, r959, r962;
}
{
add.f16x2 r968, r969, r970;
}
{
add.f16x2 r971, r965, r968;
}
{
add.f16x2 r974, r939, r940;
}
{
mul.f16x2 r977, r974, r924;
}
{
add.f16x2 r980, r942, r977;
}
{
add.f16x2 r983, r945, r946;
}
{
mul.f16x2 r986, r983, r926;
}
{
add.f16x2 r989, r980, r986;
}
{
add.f16x2 r992, r951, r952;
}
{
mul.f16x2 r995, r992, r928;
}
{
add.f16x2 r998, r989, r995;
}
{
sub.f16x2 r1001, r957, r958;
}
{
mul.f16x2 r1004, r1001, r925;
}
{
sub.f16x2 r1007, r963, r964;
}
{
mul.f16x2 r1010, r1007, r927;
}
{
add.f16x2 r1013, r1004, r1010;
}
{
sub.f16x2 r1016, r969, r970;
}
{
mul.f16x2 r1019, r1016, r929;
}
{
add.f16x2 r1022, r1013, r1019;
}
{
sub.f16x2 r1025, r998, r1022;
}
{
add.f16x2 r1028, r939, r940;
}
{
mul.f16x2 r1031, r1028, r924;
}
{
add.f16x2 r1034, r942, r1031;
}
{
add.f16x2 r1037, r945, r946;
}
{
mul.f16x2 r1040, r1037, r926;
}
{
add.f16x2 r1043, r1034, r1040;
}
{
add.f16x2 r1046, r951, r952;
}
{
mul.f16x2 r1049, r1046, r928;
}
{
add.f16x2 r1052, r1043, r1049;
}
{
sub.f16x2 r1055, r957, r958;
}
{
mul.f16x2 r1058, r1055, r925;
}
{
sub.f16x2 r1061, r963, r964;
}
{
mul.f16x2 r1064, r1061, r927;
}
{
add.f16x2 r1067, r1058, r1064;
}
{
sub.f16x2 r1070, r969, r970;
}
{
mul.f16x2 r1073, r1070, r929;
}
{
add.f16x2 r1076, r1067, r1073;
}
{
add.f16x2 r1079, r1052, r1076;
}
{
add.f16x2 r1082, r939, r940;
}
{
mul.f16x2 r1085, r1082, r926;
}
{
add.f16x2 r1088, r942, r1085;
}
{
add.f16x2 r1091, r945, r946;
}
{
mul.f16x2 r1094, r1091, r930;
}
{
add.f16x2 r1097, r1088, r1094;
}
{
add.f16x2 r1100, r951, r952;
}
{
mul.f16x2 r1103, r1100, r934;
}
{
add.f16x2 r1106, r1097, r1103;
}
{
sub.f16x2 r1109, r957, r958;
}
{
mul.f16x2 r1112, r1109, r927;
}
{
sub.f16x2 r1115, r963, r964;
}
{
mul.f16x2 r1118, r1115, r932;
}
{
add.f16x2 r1121, r1112, r1118;
}
{
sub.f16x2 r1124, r969, r970;
}
{
mul.f16x2 r1127, r1124, r936;
}
{
add.f16x2 r1130, r1121, r1127;
}
{
sub.f16x2 r1133, r1106, r1130;
}
{
add.f16x2 r1136, r939, r940;
}
{
mul.f16x2 r1139, r1136, r926;
}
{
add.f16x2 r1142, r942, r1139;
}
{
add.f16x2 r1145, r945, r946;
}
{
mul.f16x2 r1148, r1145, r930;
}
{
add.f16x2 r1151, r1142, r1148;
}
{
add.f16x2 r1154, r951, r952;
}
{
mul.f16x2 r1157, r1154, r934;
}
{
add.f16x2 r1160, r1151, r1157;
}
{
sub.f16x2 r1163, r957, r958;
}
{
mul.f16x2 r1166, r1163, r927;
}
{
sub.f16x2 r1169, r963, r964;
}
{
mul.f16x2 r1172, r1169, r932;
}
{
add.f16x2 r1175, r1166, r1172;
}
{
sub.f16x2 r1178, r969, r970;
}
{
mul.f16x2 r1181, r1178, r936;
}
{
add.f16x2 r1184, r1175, r1181;
}
{
add.f16x2 r1187, r1160, r1184;
}
{
add.f16x2 r1190, r939, r940;
}
{
mul.f16x2 r1193, r1190, r928;
}
{
add.f16x2 r1196, r942, r1193;
}
{
add.f16x2 r1199, r945, r946;
}
{
mul.f16x2 r1202, r1199, r934;
}
{
add.f16x2 r1205, r1196, r1202;
}
{
add.f16x2 r1208, r951, r952;
}
{
mul.f16x2 r1211, r1208, r926;
}
{
add.f16x2 r1214, r1205, r1211;
}
{
sub.f16x2 r1217, r957, r958;
}
{
mul.f16x2 r1220, r1217, r929;
}
{
sub.f16x2 r1223, r963, r964;
}
{
mul.f16x2 r1226, r1223, r936;
}
{
add.f16x2 r1229, r1220, r1226;
}
{
sub.f16x2 r1232, r969, r970;
}
{
mul.f16x2 r1235, r1232, r927;
}
{
add.f16x2 r1238, r1229, r1235;
}
{
sub.f16x2 r1241, r1214, r1238;
}
{
add.f16x2 r1244, r939, r940;
}
{
mul.f16x2 r1247, r1244, r928;
}
{
add.f16x2 r1250, r942, r1247;
}
{
add.f16x2 r1253, r945, r946;
}
{
mul.f16x2 r1256, r1253, r934;
}
{
add.f16x2 r1259, r1250, r1256;
}
{
add.f16x2 r1262, r951, r952;
}
{
mul.f16x2 r1265, r1262, r926;
}
{
add.f16x2 r1268, r1259, r1265;
}
{
sub.f16x2 r1271, r957, r958;
}
{
mul.f16x2 r1274, r1271, r929;
}
{
sub.f16x2 r1277, r963, r964;
}
{
mul.f16x2 r1280, r1277, r936;
}
{
add.f16x2 r1283, r1274, r1280;
}
{
sub.f16x2 r1286, r969, r970;
}
{
mul.f16x2 r1289, r1286, r927;
}
{
add.f16x2 r1292, r1283, r1289;
}
{
add.f16x2 r1295, r1268, r1292;
}
{
add.f16x2 r1298, r957, r958;
}
{
mul.f16x2 r1301, r1298, r924;
}
{
add.f16x2 r1304, r960, r1301;
}
{
add.f16x2 r1307, r963, r964;
}
{
mul.f16x2 r1310, r1307, r926;
}
{
add.f16x2 r1313, r1304, r1310;
}
{
add.f16x2 r1316, r969, r970;
}
{
mul.f16x2 r1319, r1316, r928;
}
{
add.f16x2 r1322, r1313, r1319;
}
{
sub.f16x2 r1325, r939, r940;
}
{
mul.f16x2 r1328, r1325, r925;
}
{
sub.f16x2 r1331, r945, r946;
}
{
mul.f16x2 r1334, r1331, r927;
}
{
add.f16x2 r1337, r1328, r1334;
}
{
sub.f16x2 r1340, r951, r952;
}
{
mul.f16x2 r1343, r1340, r929;
}
{
add.f16x2 r1346, r1337, r1343;
}
{
add.f16x2 r1349, r1322, r1346;
}
{
add.f16x2 r1352, r957, r958;
}
{
mul.f16x2 r1355, r1352, r924;
}
{
add.f16x2 r1358, r960, r1355;
}
{
add.f16x2 r1361, r963, r964;
}
{
mul.f16x2 r1364, r1361, r926;
}
{
add.f16x2 r1367, r1358, r1364;
}
{
add.f16x2 r1370, r969, r970;
}
{
mul.f16x2 r1373, r1370, r928;
}
{
add.f16x2 r1376, r1367, r1373;
}
{
sub.f16x2 r1379, r939, r940;
}
{
mul.f16x2 r1382, r1379, r925;
}
{
sub.f16x2 r1385, r945, r946;
}
{
mul.f16x2 r1388, r1385, r927;
}
{
add.f16x2 r1391, r1382, r1388;
}
{
sub.f16x2 r1394, r951, r952;
}
{
mul.f16x2 r1397, r1394, r929;
}
{
add.f16x2 r1400, r1391, r1397;
}
{
sub.f16x2 r1403, r1376, r1400;
}
{
add.f16x2 r1406, r957, r958;
}
{
mul.f16x2 r1409, r1406, r926;
}
{
add.f16x2 r1412, r960, r1409;
}
{
add.f16x2 r1415, r963, r964;
}
{
mul.f16x2 r1418, r1415, r930;
}
{
add.f16x2 r1421, r1412, r1418;
}
{
add.f16x2 r1424, r969, r970;
}
{
mul.f16x2 r1427, r1424, r934;
}
{
add.f16x2 r1430, r1421, r1427;
}
{
sub.f16x2 r1433, r939, r940;
}
{
mul.f16x2 r1436, r1433, r927;
}
{
sub.f16x2 r1439, r945, r946;
}
{
mul.f16x2 r1442, r1439, r932;
}
{
add.f16x2 r1445, r1436, r1442;
}
{
sub.f16x2 r1448, r951, r952;
}
{
mul.f16x2 r1451, r1448, r936;
}
{
add.f16x2 r1454, r1445, r1451;
}
{
add.f16x2 r1457, r1430, r1454;
}
{
add.f16x2 r1460, r957, r958;
}
{
mul.f16x2 r1463, r1460, r926;
}
{
add.f16x2 r1466, r960, r1463;
}
{
add.f16x2 r1469, r963, r964;
}
{
mul.f16x2 r1472, r1469, r930;
}
{
add.f16x2 r1475, r1466, r1472;
}
{
add.f16x2 r1478, r969, r970;
}
{
mul.f16x2 r1481, r1478, r934;
}
{
add.f16x2 r1484, r1475, r1481;
}
{
sub.f16x2 r1487, r939, r940;
}
{
mul.f16x2 r1490, r1487, r927;
}
{
sub.f16x2 r1493, r945, r946;
}
{
mul.f16x2 r1496, r1493, r932;
}
{
add.f16x2 r1499, r1490, r1496;
}
{
sub.f16x2 r1502, r951, r952;
}
{
mul.f16x2 r1505, r1502, r936;
}
{
add.f16x2 r1508, r1499, r1505;
}
{
sub.f16x2 r1511, r1484, r1508;
}
{
add.f16x2 r1514, r957, r958;
}
{
mul.f16x2 r1517, r1514, r928;
}
{
add.f16x2 r1520, r960, r1517;
}
{
add.f16x2 r1523, r963, r964;
}
{
mul.f16x2 r1526, r1523, r934;
}
{
add.f16x2 r1529, r1520, r1526;
}
{
add.f16x2 r1532, r969, r970;
}
{
mul.f16x2 r1535, r1532, r926;
}
{
add.f16x2 r1538, r1529, r1535;
}
{
sub.f16x2 r1541, r939, r940;
}
{
mul.f16x2 r1544, r1541, r929;
}
{
sub.f16x2 r1547, r945, r946;
}
{
mul.f16x2 r1550, r1547, r936;
}
{
add.f16x2 r1553, r1544, r1550;
}
{
sub.f16x2 r1556, r951, r952;
}
{
mul.f16x2 r1559, r1556, r927;
}
{
add.f16x2 r1562, r1553, r1559;
}
{
add.f16x2 r1565, r1538, r1562;
}
{
add.f16x2 r1568, r957, r958;
}
{
mul.f16x2 r1571, r1568, r928;
}
{
add.f16x2 r1574, r960, r1571;
}
{
add.f16x2 r1577, r963, r964;
}
{
mul.f16x2 r1580, r1577, r934;
}
{
add.f16x2 r1583, r1574, r1580;
}
{
add.f16x2 r1586, r969, r970;
}
{
mul.f16x2 r1589, r1586, r926;
}
{
add.f16x2 r1592, r1583, r1589;
}
{
sub.f16x2 r1595, r939, r940;
}
{
mul.f16x2 r1598, r1595, r929;
}
{
sub.f16x2 r1601, r945, r946;
}
{
mul.f16x2 r1604, r1601, r936;
}
{
add.f16x2 r1607, r1598, r1604;
}
{
sub.f16x2 r1610, r951, r952;
}
{
mul.f16x2 r1613, r1610, r927;
}
{
add.f16x2 r1616, r1607, r1613;
}
{
sub.f16x2 r1619, r1592, r1616;
}
mul.wide.u32 rd4, r3474, 613566757;
shr.u64 rd5, rd4, 32;
cvt.u32.u64 r3478, rd5;
sub.s32 r3479, r3474, r3478;
shr.u32 r3480, r3479, 1;
add.s32 r3481, r3480, r3478;
shr.u32 r3482, r3481, 2;
cvt.rn.f32.u32 f132, r3482;
mul.f32 f133, f132, 0f3C961050;
cos.approx.f32 f57, f133;
sin.approx.f32 f134, f133;
neg.f32 f58, f134;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f57;
cvt.rn.f16.f32 high, f58;
mov.b32 r1622, {low, high};
}
mul.lo.s32 r3483, r3482, 7;
sub.s32 r3484, r3474, r3483;
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1625, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1627, {high, high};
}
{
mul.f16x2 r1629, r1349, r1627;
}
{
neg.f16x2 r1632, r1629;
}
{
fma.rn.f16x2 r1634, r1025, r1625, r1632;
}
{
mul.f16x2 r1638, r1025, r1627;
}
{
fma.rn.f16x2 r1641, r1349, r1625, r1638;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1645, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1647, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r1649, {low, high};
}
{
mul.f16x2 r1650, r1647, r1649;
}
{
mul.f16x2 r1653, r1622, r1645;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1656, {high, low};
}
{
fma.rn.f16x2 r1658, r1650, r1656, r1653;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1658;
mov.b32 r1662, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1658;
mov.b32 r1664, {high, high};
}
{
mul.f16x2 r1666, r1457, r1664;
}
{
neg.f16x2 r1669, r1666;
}
{
fma.rn.f16x2 r1671, r1133, r1662, r1669;
}
{
mul.f16x2 r1675, r1133, r1664;
}
{
fma.rn.f16x2 r1678, r1457, r1662, r1675;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1682, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1684, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r1686, {low, high};
}
{
mul.f16x2 r1687, r1684, r1686;
}
{
mul.f16x2 r1690, r1658, r1682;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1658;
mov.b32 r1693, {high, low};
}
{
fma.rn.f16x2 r1695, r1687, r1693, r1690;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1695;
mov.b32 r1699, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1695;
mov.b32 r1701, {high, high};
}
{
mul.f16x2 r1703, r1565, r1701;
}
{
neg.f16x2 r1706, r1703;
}
{
fma.rn.f16x2 r1708, r1241, r1699, r1706;
}
{
mul.f16x2 r1712, r1241, r1701;
}
{
fma.rn.f16x2 r1715, r1565, r1699, r1712;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1719, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1721, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r1723, {low, high};
}
{
mul.f16x2 r1724, r1721, r1723;
}
{
mul.f16x2 r1727, r1695, r1719;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1695;
mov.b32 r1730, {high, low};
}
{
fma.rn.f16x2 r1732, r1724, r1730, r1727;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1732;
mov.b32 r1736, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1732;
mov.b32 r1738, {high, high};
}
{
mul.f16x2 r1740, r1619, r1738;
}
{
neg.f16x2 r1743, r1740;
}
{
fma.rn.f16x2 r1745, r1295, r1736, r1743;
}
{
mul.f16x2 r1749, r1295, r1738;
}
{
fma.rn.f16x2 r1752, r1619, r1736, r1749;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1756, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1758, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r1760, {low, high};
}
{
mul.f16x2 r1761, r1758, r1760;
}
{
mul.f16x2 r1764, r1732, r1756;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1732;
mov.b32 r1767, {high, low};
}
{
fma.rn.f16x2 r1769, r1761, r1767, r1764;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1769;
mov.b32 r1773, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1769;
mov.b32 r1775, {high, high};
}
{
mul.f16x2 r1777, r1511, r1775;
}
{
neg.f16x2 r1780, r1777;
}
{
fma.rn.f16x2 r1782, r1187, r1773, r1780;
}
{
mul.f16x2 r1786, r1187, r1775;
}
{
fma.rn.f16x2 r1789, r1511, r1773, r1786;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1793, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1622;
mov.b32 r1795, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r1797, {low, high};
}
{
mul.f16x2 r1798, r1795, r1797;
}
{
mul.f16x2 r1801, r1769, r1793;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1769;
mov.b32 r1804, {high, low};
}
{
fma.rn.f16x2 r1806, r1798, r1804, r1801;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1806;
mov.b32 r1810, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1806;
mov.b32 r1812, {high, high};
}
{
mul.f16x2 r1814, r1403, r1812;
}
{
neg.f16x2 r1817, r1814;
}
{
fma.rn.f16x2 r1819, r1079, r1810, r1817;
}
{
mul.f16x2 r1823, r1079, r1812;
}
{
fma.rn.f16x2 r1826, r1403, r1810, r1823;
}
shl.b32 r3485, r3484, 2;
add.s32 r3486, r3475, r3485;
barrier.sync 0;
mad.lo.s32 r3487, r3482, 196, r3486;
st.shared.u32 [r3487], r953;
st.shared.u32 [r3487+28], r1634;
st.shared.u32 [r3487+56], r1671;
st.shared.u32 [r3487+84], r1708;
st.shared.u32 [r3487+112], r1745;
st.shared.u32 [r3487+140], r1782;
st.shared.u32 [r3487+168], r1819;
barrier.sync 0;
ld.shared.u32 r1865, [r3477];
ld.shared.u32 r1862, [r3477+1372];
ld.shared.u32 r1868, [r3477+2744];
ld.shared.u32 r1874, [r3477+4116];
ld.shared.u32 r1875, [r3477+5488];
ld.shared.u32 r1869, [r3477+6860];
ld.shared.u32 r1863, [r3477+8232];
barrier.sync 0;
st.shared.u32 [r3487], r971;
st.shared.u32 [r3487+28], r1641;
st.shared.u32 [r3487+56], r1678;
st.shared.u32 [r3487+84], r1715;
st.shared.u32 [r3487+112], r1752;
st.shared.u32 [r3487+140], r1789;
st.shared.u32 [r3487+168], r1826;
barrier.sync 0;
ld.shared.u32 r1883, [r3477];
ld.shared.u32 r1880, [r3477+1372];
ld.shared.u32 r1886, [r3477+2744];
ld.shared.u32 r1892, [r3477+4116];
ld.shared.u32 r1893, [r3477+5488];
ld.shared.u32 r1887, [r3477+6860];
ld.shared.u32 r1881, [r3477+8232];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f126;
cvt.rn.f16.f32 high, f126;
mov.b32 r1847, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f128;
cvt.rn.f16.f32 high, f128;
mov.b32 r1848, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f114;
cvt.rn.f16.f32 high, f114;
mov.b32 r1849, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f116;
cvt.rn.f16.f32 high, f116;
mov.b32 r1850, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r1851, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r1852, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r1853, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r1854, {low, high};
}
{
neg.f16x2 r1855, r1854;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f126;
cvt.rn.f16.f32 high, f126;
mov.b32 r1857, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f128;
cvt.rn.f16.f32 high, f128;
mov.b32 r1858, {low, high};
}
{
neg.f16x2 r1859, r1858;
}
{
add.f16x2 r1861, r1862, r1863;
}
{
add.f16x2 r1864, r1865, r1861;
}
{
add.f16x2 r1867, r1868, r1869;
}
{
add.f16x2 r1870, r1864, r1867;
}
{
add.f16x2 r1873, r1874, r1875;
}
{
add.f16x2 r1876, r1870, r1873;
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
add.f16x2 r1888, r1882, r1885;
}
{
add.f16x2 r1891, r1892, r1893;
}
{
add.f16x2 r1894, r1888, r1891;
}
{
add.f16x2 r1897, r1862, r1863;
}
{
mul.f16x2 r1900, r1897, r1847;
}
{
add.f16x2 r1903, r1865, r1900;
}
{
add.f16x2 r1906, r1868, r1869;
}
{
mul.f16x2 r1909, r1906, r1849;
}
{
add.f16x2 r1912, r1903, r1909;
}
{
add.f16x2 r1915, r1874, r1875;
}
{
mul.f16x2 r1918, r1915, r1851;
}
{
add.f16x2 r1921, r1912, r1918;
}
{
sub.f16x2 r1924, r1880, r1881;
}
{
mul.f16x2 r1927, r1924, r1848;
}
{
sub.f16x2 r1930, r1886, r1887;
}
{
mul.f16x2 r1933, r1930, r1850;
}
{
add.f16x2 r1936, r1927, r1933;
}
{
sub.f16x2 r1939, r1892, r1893;
}
{
mul.f16x2 r1942, r1939, r1852;
}
{
add.f16x2 r1945, r1936, r1942;
}
{
sub.f16x2 r1948, r1921, r1945;
}
{
add.f16x2 r1951, r1862, r1863;
}
{
mul.f16x2 r1954, r1951, r1847;
}
{
add.f16x2 r1957, r1865, r1954;
}
{
add.f16x2 r1960, r1868, r1869;
}
{
mul.f16x2 r1963, r1960, r1849;
}
{
add.f16x2 r1966, r1957, r1963;
}
{
add.f16x2 r1969, r1874, r1875;
}
{
mul.f16x2 r1972, r1969, r1851;
}
{
add.f16x2 r1975, r1966, r1972;
}
{
sub.f16x2 r1978, r1880, r1881;
}
{
mul.f16x2 r1981, r1978, r1848;
}
{
sub.f16x2 r1984, r1886, r1887;
}
{
mul.f16x2 r1987, r1984, r1850;
}
{
add.f16x2 r1990, r1981, r1987;
}
{
sub.f16x2 r1993, r1892, r1893;
}
{
mul.f16x2 r1996, r1993, r1852;
}
{
add.f16x2 r1999, r1990, r1996;
}
{
add.f16x2 r2002, r1975, r1999;
}
{
add.f16x2 r2005, r1862, r1863;
}
{
mul.f16x2 r2008, r2005, r1849;
}
{
add.f16x2 r2011, r1865, r2008;
}
{
add.f16x2 r2014, r1868, r1869;
}
{
mul.f16x2 r2017, r2014, r1853;
}
{
add.f16x2 r2020, r2011, r2017;
}
{
add.f16x2 r2023, r1874, r1875;
}
{
mul.f16x2 r2026, r2023, r1857;
}
{
add.f16x2 r2029, r2020, r2026;
}
{
sub.f16x2 r2032, r1880, r1881;
}
{
mul.f16x2 r2035, r2032, r1850;
}
{
sub.f16x2 r2038, r1886, r1887;
}
{
mul.f16x2 r2041, r2038, r1855;
}
{
add.f16x2 r2044, r2035, r2041;
}
{
sub.f16x2 r2047, r1892, r1893;
}
{
mul.f16x2 r2050, r2047, r1859;
}
{
add.f16x2 r2053, r2044, r2050;
}
{
sub.f16x2 r2056, r2029, r2053;
}
{
add.f16x2 r2059, r1862, r1863;
}
{
mul.f16x2 r2062, r2059, r1849;
}
{
add.f16x2 r2065, r1865, r2062;
}
{
add.f16x2 r2068, r1868, r1869;
}
{
mul.f16x2 r2071, r2068, r1853;
}
{
add.f16x2 r2074, r2065, r2071;
}
{
add.f16x2 r2077, r1874, r1875;
}
{
mul.f16x2 r2080, r2077, r1857;
}
{
add.f16x2 r2083, r2074, r2080;
}
{
sub.f16x2 r2086, r1880, r1881;
}
{
mul.f16x2 r2089, r2086, r1850;
}
{
sub.f16x2 r2092, r1886, r1887;
}
{
mul.f16x2 r2095, r2092, r1855;
}
{
add.f16x2 r2098, r2089, r2095;
}
{
sub.f16x2 r2101, r1892, r1893;
}
{
mul.f16x2 r2104, r2101, r1859;
}
{
add.f16x2 r2107, r2098, r2104;
}
{
add.f16x2 r2110, r2083, r2107;
}
{
add.f16x2 r2113, r1862, r1863;
}
{
mul.f16x2 r2116, r2113, r1851;
}
{
add.f16x2 r2119, r1865, r2116;
}
{
add.f16x2 r2122, r1868, r1869;
}
{
mul.f16x2 r2125, r2122, r1857;
}
{
add.f16x2 r2128, r2119, r2125;
}
{
add.f16x2 r2131, r1874, r1875;
}
{
mul.f16x2 r2134, r2131, r1849;
}
{
add.f16x2 r2137, r2128, r2134;
}
{
sub.f16x2 r2140, r1880, r1881;
}
{
mul.f16x2 r2143, r2140, r1852;
}
{
sub.f16x2 r2146, r1886, r1887;
}
{
mul.f16x2 r2149, r2146, r1859;
}
{
add.f16x2 r2152, r2143, r2149;
}
{
sub.f16x2 r2155, r1892, r1893;
}
{
mul.f16x2 r2158, r2155, r1850;
}
{
add.f16x2 r2161, r2152, r2158;
}
{
sub.f16x2 r2164, r2137, r2161;
}
{
add.f16x2 r2167, r1862, r1863;
}
{
mul.f16x2 r2170, r2167, r1851;
}
{
add.f16x2 r2173, r1865, r2170;
}
{
add.f16x2 r2176, r1868, r1869;
}
{
mul.f16x2 r2179, r2176, r1857;
}
{
add.f16x2 r2182, r2173, r2179;
}
{
add.f16x2 r2185, r1874, r1875;
}
{
mul.f16x2 r2188, r2185, r1849;
}
{
add.f16x2 r2191, r2182, r2188;
}
{
sub.f16x2 r2194, r1880, r1881;
}
{
mul.f16x2 r2197, r2194, r1852;
}
{
sub.f16x2 r2200, r1886, r1887;
}
{
mul.f16x2 r2203, r2200, r1859;
}
{
add.f16x2 r2206, r2197, r2203;
}
{
sub.f16x2 r2209, r1892, r1893;
}
{
mul.f16x2 r2212, r2209, r1850;
}
{
add.f16x2 r2215, r2206, r2212;
}
{
add.f16x2 r2218, r2191, r2215;
}
{
add.f16x2 r2221, r1880, r1881;
}
{
mul.f16x2 r2224, r2221, r1847;
}
{
add.f16x2 r2227, r1883, r2224;
}
{
add.f16x2 r2230, r1886, r1887;
}
{
mul.f16x2 r2233, r2230, r1849;
}
{
add.f16x2 r2236, r2227, r2233;
}
{
add.f16x2 r2239, r1892, r1893;
}
{
mul.f16x2 r2242, r2239, r1851;
}
{
add.f16x2 r2245, r2236, r2242;
}
{
sub.f16x2 r2248, r1862, r1863;
}
{
mul.f16x2 r2251, r2248, r1848;
}
{
sub.f16x2 r2254, r1868, r1869;
}
{
mul.f16x2 r2257, r2254, r1850;
}
{
add.f16x2 r2260, r2251, r2257;
}
{
sub.f16x2 r2263, r1874, r1875;
}
{
mul.f16x2 r2266, r2263, r1852;
}
{
add.f16x2 r2269, r2260, r2266;
}
{
add.f16x2 r2272, r2245, r2269;
}
{
add.f16x2 r2275, r1880, r1881;
}
{
mul.f16x2 r2278, r2275, r1847;
}
{
add.f16x2 r2281, r1883, r2278;
}
{
add.f16x2 r2284, r1886, r1887;
}
{
mul.f16x2 r2287, r2284, r1849;
}
{
add.f16x2 r2290, r2281, r2287;
}
{
add.f16x2 r2293, r1892, r1893;
}
{
mul.f16x2 r2296, r2293, r1851;
}
{
add.f16x2 r2299, r2290, r2296;
}
{
sub.f16x2 r2302, r1862, r1863;
}
{
mul.f16x2 r2305, r2302, r1848;
}
{
sub.f16x2 r2308, r1868, r1869;
}
{
mul.f16x2 r2311, r2308, r1850;
}
{
add.f16x2 r2314, r2305, r2311;
}
{
sub.f16x2 r2317, r1874, r1875;
}
{
mul.f16x2 r2320, r2317, r1852;
}
{
add.f16x2 r2323, r2314, r2320;
}
{
sub.f16x2 r2326, r2299, r2323;
}
{
add.f16x2 r2329, r1880, r1881;
}
{
mul.f16x2 r2332, r2329, r1849;
}
{
add.f16x2 r2335, r1883, r2332;
}
{
add.f16x2 r2338, r1886, r1887;
}
{
mul.f16x2 r2341, r2338, r1853;
}
{
add.f16x2 r2344, r2335, r2341;
}
{
add.f16x2 r2347, r1892, r1893;
}
{
mul.f16x2 r2350, r2347, r1857;
}
{
add.f16x2 r2353, r2344, r2350;
}
{
sub.f16x2 r2356, r1862, r1863;
}
{
mul.f16x2 r2359, r2356, r1850;
}
{
sub.f16x2 r2362, r1868, r1869;
}
{
mul.f16x2 r2365, r2362, r1855;
}
{
add.f16x2 r2368, r2359, r2365;
}
{
sub.f16x2 r2371, r1874, r1875;
}
{
mul.f16x2 r2374, r2371, r1859;
}
{
add.f16x2 r2377, r2368, r2374;
}
{
add.f16x2 r2380, r2353, r2377;
}
{
add.f16x2 r2383, r1880, r1881;
}
{
mul.f16x2 r2386, r2383, r1849;
}
{
add.f16x2 r2389, r1883, r2386;
}
{
add.f16x2 r2392, r1886, r1887;
}
{
mul.f16x2 r2395, r2392, r1853;
}
{
add.f16x2 r2398, r2389, r2395;
}
{
add.f16x2 r2401, r1892, r1893;
}
{
mul.f16x2 r2404, r2401, r1857;
}
{
add.f16x2 r2407, r2398, r2404;
}
{
sub.f16x2 r2410, r1862, r1863;
}
{
mul.f16x2 r2413, r2410, r1850;
}
{
sub.f16x2 r2416, r1868, r1869;
}
{
mul.f16x2 r2419, r2416, r1855;
}
{
add.f16x2 r2422, r2413, r2419;
}
{
sub.f16x2 r2425, r1874, r1875;
}
{
mul.f16x2 r2428, r2425, r1859;
}
{
add.f16x2 r2431, r2422, r2428;
}
{
sub.f16x2 r2434, r2407, r2431;
}
{
add.f16x2 r2437, r1880, r1881;
}
{
mul.f16x2 r2440, r2437, r1851;
}
{
add.f16x2 r2443, r1883, r2440;
}
{
add.f16x2 r2446, r1886, r1887;
}
{
mul.f16x2 r2449, r2446, r1857;
}
{
add.f16x2 r2452, r2443, r2449;
}
{
add.f16x2 r2455, r1892, r1893;
}
{
mul.f16x2 r2458, r2455, r1849;
}
{
add.f16x2 r2461, r2452, r2458;
}
{
sub.f16x2 r2464, r1862, r1863;
}
{
mul.f16x2 r2467, r2464, r1852;
}
{
sub.f16x2 r2470, r1868, r1869;
}
{
mul.f16x2 r2473, r2470, r1859;
}
{
add.f16x2 r2476, r2467, r2473;
}
{
sub.f16x2 r2479, r1874, r1875;
}
{
mul.f16x2 r2482, r2479, r1850;
}
{
add.f16x2 r2485, r2476, r2482;
}
{
add.f16x2 r2488, r2461, r2485;
}
{
add.f16x2 r2491, r1880, r1881;
}
{
mul.f16x2 r2494, r2491, r1851;
}
{
add.f16x2 r2497, r1883, r2494;
}
{
add.f16x2 r2500, r1886, r1887;
}
{
mul.f16x2 r2503, r2500, r1857;
}
{
add.f16x2 r2506, r2497, r2503;
}
{
add.f16x2 r2509, r1892, r1893;
}
{
mul.f16x2 r2512, r2509, r1849;
}
{
add.f16x2 r2515, r2506, r2512;
}
{
sub.f16x2 r2518, r1862, r1863;
}
{
mul.f16x2 r2521, r2518, r1852;
}
{
sub.f16x2 r2524, r1868, r1869;
}
{
mul.f16x2 r2527, r2524, r1859;
}
{
add.f16x2 r2530, r2521, r2527;
}
{
sub.f16x2 r2533, r1874, r1875;
}
{
mul.f16x2 r2536, r2533, r1850;
}
{
add.f16x2 r2539, r2530, r2536;
}
{
sub.f16x2 r2542, r2515, r2539;
}
mul.wide.u32 rd6, r3474, 1402438301;
shr.u64 rd7, rd6, 36;
cvt.u32.u64 r3488, rd7;
cvt.rn.f32.u32 f135, r3488;
mul.f32 f136, f135, 0f3E034E46;
cos.approx.f32 f93, f136;
sin.approx.f32 f137, f136;
neg.f32 f94, f137;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r2545, {low, high};
}
mul.lo.s32 r3489, r3488, 49;
sub.s32 r3490, r3474, r3489;
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2548, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2550, {high, high};
}
{
mul.f16x2 r2552, r2272, r2550;
}
{
neg.f16x2 r2555, r2552;
}
{
fma.rn.f16x2 r2557, r1948, r2548, r2555;
}
{
mul.f16x2 r2561, r1948, r2550;
}
{
fma.rn.f16x2 r2564, r2272, r2548, r2561;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2568, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2570, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r2572, {low, high};
}
{
mul.f16x2 r2573, r2570, r2572;
}
{
mul.f16x2 r2576, r2545, r2568;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2579, {high, low};
}
{
fma.rn.f16x2 r2581, r2573, r2579, r2576;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2581;
mov.b32 r2585, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2581;
mov.b32 r2587, {high, high};
}
{
mul.f16x2 r2589, r2380, r2587;
}
{
neg.f16x2 r2592, r2589;
}
{
fma.rn.f16x2 r2594, r2056, r2585, r2592;
}
{
mul.f16x2 r2598, r2056, r2587;
}
{
fma.rn.f16x2 r2601, r2380, r2585, r2598;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2605, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2607, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r2609, {low, high};
}
{
mul.f16x2 r2610, r2607, r2609;
}
{
mul.f16x2 r2613, r2581, r2605;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2581;
mov.b32 r2616, {high, low};
}
{
fma.rn.f16x2 r2618, r2610, r2616, r2613;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2618;
mov.b32 r2622, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2618;
mov.b32 r2624, {high, high};
}
{
mul.f16x2 r2626, r2488, r2624;
}
{
neg.f16x2 r2629, r2626;
}
{
fma.rn.f16x2 r2631, r2164, r2622, r2629;
}
{
mul.f16x2 r2635, r2164, r2624;
}
{
fma.rn.f16x2 r2638, r2488, r2622, r2635;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2642, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2644, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r2646, {low, high};
}
{
mul.f16x2 r2647, r2644, r2646;
}
{
mul.f16x2 r2650, r2618, r2642;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2618;
mov.b32 r2653, {high, low};
}
{
fma.rn.f16x2 r2655, r2647, r2653, r2650;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2655;
mov.b32 r2659, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2655;
mov.b32 r2661, {high, high};
}
{
mul.f16x2 r2663, r2542, r2661;
}
{
neg.f16x2 r2666, r2663;
}
{
fma.rn.f16x2 r2668, r2218, r2659, r2666;
}
{
mul.f16x2 r2672, r2218, r2661;
}
{
fma.rn.f16x2 r2675, r2542, r2659, r2672;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2679, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2681, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r2683, {low, high};
}
{
mul.f16x2 r2684, r2681, r2683;
}
{
mul.f16x2 r2687, r2655, r2679;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2655;
mov.b32 r2690, {high, low};
}
{
fma.rn.f16x2 r2692, r2684, r2690, r2687;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2692;
mov.b32 r2696, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2692;
mov.b32 r2698, {high, high};
}
{
mul.f16x2 r2700, r2434, r2698;
}
{
neg.f16x2 r2703, r2700;
}
{
fma.rn.f16x2 r2705, r2110, r2696, r2703;
}
{
mul.f16x2 r2709, r2110, r2698;
}
{
fma.rn.f16x2 r2712, r2434, r2696, r2709;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2716, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2545;
mov.b32 r2718, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f105;
cvt.rn.f16.f32 high, f106;
mov.b32 r2720, {low, high};
}
{
mul.f16x2 r2721, r2718, r2720;
}
{
mul.f16x2 r2724, r2692, r2716;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2692;
mov.b32 r2727, {high, low};
}
{
fma.rn.f16x2 r2729, r2721, r2727, r2724;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2729;
mov.b32 r2733, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2729;
mov.b32 r2735, {high, high};
}
{
mul.f16x2 r2737, r2326, r2735;
}
{
neg.f16x2 r2740, r2737;
}
{
fma.rn.f16x2 r2742, r2002, r2733, r2740;
}
{
mul.f16x2 r2746, r2002, r2735;
}
{
fma.rn.f16x2 r2749, r2326, r2733, r2746;
}
shl.b32 r3491, r3490, 2;
add.s32 r3492, r3475, r3491;
barrier.sync 0;
mad.lo.s32 r3493, r3488, 1372, r3492;
st.shared.u32 [r3493], r1876;
st.shared.u32 [r3493+196], r2557;
st.shared.u32 [r3493+392], r2594;
st.shared.u32 [r3493+588], r2631;
st.shared.u32 [r3493+784], r2668;
st.shared.u32 [r3493+980], r2705;
st.shared.u32 [r3493+1176], r2742;
barrier.sync 0;
ld.shared.u32 r2788, [r3477];
ld.shared.u32 r2785, [r3477+1372];
ld.shared.u32 r2791, [r3477+2744];
ld.shared.u32 r2797, [r3477+4116];
ld.shared.u32 r2798, [r3477+5488];
ld.shared.u32 r2792, [r3477+6860];
ld.shared.u32 r2786, [r3477+8232];
barrier.sync 0;
st.shared.u32 [r3493], r1894;
st.shared.u32 [r3493+196], r2564;
st.shared.u32 [r3493+392], r2601;
st.shared.u32 [r3493+588], r2638;
st.shared.u32 [r3493+784], r2675;
st.shared.u32 [r3493+980], r2712;
st.shared.u32 [r3493+1176], r2749;
barrier.sync 0;
ld.shared.u32 r2806, [r3477];
ld.shared.u32 r2803, [r3477+1372];
ld.shared.u32 r2809, [r3477+2744];
ld.shared.u32 r2815, [r3477+4116];
ld.shared.u32 r2816, [r3477+5488];
ld.shared.u32 r2810, [r3477+6860];
ld.shared.u32 r2804, [r3477+8232];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f126;
cvt.rn.f16.f32 high, f126;
mov.b32 r2770, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f128;
cvt.rn.f16.f32 high, f128;
mov.b32 r2771, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f114;
cvt.rn.f16.f32 high, f114;
mov.b32 r2772, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f116;
cvt.rn.f16.f32 high, f116;
mov.b32 r2773, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r2774, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r2775, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r2776, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r2777, {low, high};
}
{
neg.f16x2 r2778, r2777;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f126;
cvt.rn.f16.f32 high, f126;
mov.b32 r2780, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f128;
cvt.rn.f16.f32 high, f128;
mov.b32 r2781, {low, high};
}
{
neg.f16x2 r2782, r2781;
}
{
add.f16x2 r2784, r2785, r2786;
}
{
add.f16x2 r2787, r2788, r2784;
}
{
add.f16x2 r2790, r2791, r2792;
}
{
add.f16x2 r2793, r2787, r2790;
}
{
add.f16x2 r2796, r2797, r2798;
}
{
add.f16x2 %0, r2793, r2796;
}
{
add.f16x2 r2802, r2803, r2804;
}
{
add.f16x2 r2805, r2806, r2802;
}
{
add.f16x2 r2808, r2809, r2810;
}
{
add.f16x2 r2811, r2805, r2808;
}
{
add.f16x2 r2814, r2815, r2816;
}
{
add.f16x2 %1, r2811, r2814;
}
{
add.f16x2 r2820, r2785, r2786;
}
{
mul.f16x2 r2823, r2820, r2770;
}
{
add.f16x2 r2826, r2788, r2823;
}
{
add.f16x2 r2829, r2791, r2792;
}
{
mul.f16x2 r2832, r2829, r2772;
}
{
add.f16x2 r2835, r2826, r2832;
}
{
add.f16x2 r2838, r2797, r2798;
}
{
mul.f16x2 r2841, r2838, r2774;
}
{
add.f16x2 r2844, r2835, r2841;
}
{
sub.f16x2 r2847, r2803, r2804;
}
{
mul.f16x2 r2850, r2847, r2771;
}
{
sub.f16x2 r2853, r2809, r2810;
}
{
mul.f16x2 r2856, r2853, r2773;
}
{
add.f16x2 r2859, r2850, r2856;
}
{
sub.f16x2 r2862, r2815, r2816;
}
{
mul.f16x2 r2865, r2862, r2775;
}
{
add.f16x2 r2868, r2859, r2865;
}
{
sub.f16x2 %2, r2844, r2868;
}
{
add.f16x2 r2874, r2785, r2786;
}
{
mul.f16x2 r2877, r2874, r2770;
}
{
add.f16x2 r2880, r2788, r2877;
}
{
add.f16x2 r2883, r2791, r2792;
}
{
mul.f16x2 r2886, r2883, r2772;
}
{
add.f16x2 r2889, r2880, r2886;
}
{
add.f16x2 r2892, r2797, r2798;
}
{
mul.f16x2 r2895, r2892, r2774;
}
{
add.f16x2 r2898, r2889, r2895;
}
{
sub.f16x2 r2901, r2803, r2804;
}
{
mul.f16x2 r2904, r2901, r2771;
}
{
sub.f16x2 r2907, r2809, r2810;
}
{
mul.f16x2 r2910, r2907, r2773;
}
{
add.f16x2 r2913, r2904, r2910;
}
{
sub.f16x2 r2916, r2815, r2816;
}
{
mul.f16x2 r2919, r2916, r2775;
}
{
add.f16x2 r2922, r2913, r2919;
}
{
add.f16x2 %12, r2898, r2922;
}
{
add.f16x2 r2928, r2785, r2786;
}
{
mul.f16x2 r2931, r2928, r2772;
}
{
add.f16x2 r2934, r2788, r2931;
}
{
add.f16x2 r2937, r2791, r2792;
}
{
mul.f16x2 r2940, r2937, r2776;
}
{
add.f16x2 r2943, r2934, r2940;
}
{
add.f16x2 r2946, r2797, r2798;
}
{
mul.f16x2 r2949, r2946, r2780;
}
{
add.f16x2 r2952, r2943, r2949;
}
{
sub.f16x2 r2955, r2803, r2804;
}
{
mul.f16x2 r2958, r2955, r2773;
}
{
sub.f16x2 r2961, r2809, r2810;
}
{
mul.f16x2 r2964, r2961, r2778;
}
{
add.f16x2 r2967, r2958, r2964;
}
{
sub.f16x2 r2970, r2815, r2816;
}
{
mul.f16x2 r2973, r2970, r2782;
}
{
add.f16x2 r2976, r2967, r2973;
}
{
sub.f16x2 %4, r2952, r2976;
}
{
add.f16x2 r2982, r2785, r2786;
}
{
mul.f16x2 r2985, r2982, r2772;
}
{
add.f16x2 r2988, r2788, r2985;
}
{
add.f16x2 r2991, r2791, r2792;
}
{
mul.f16x2 r2994, r2991, r2776;
}
{
add.f16x2 r2997, r2988, r2994;
}
{
add.f16x2 r3000, r2797, r2798;
}
{
mul.f16x2 r3003, r3000, r2780;
}
{
add.f16x2 r3006, r2997, r3003;
}
{
sub.f16x2 r3009, r2803, r2804;
}
{
mul.f16x2 r3012, r3009, r2773;
}
{
sub.f16x2 r3015, r2809, r2810;
}
{
mul.f16x2 r3018, r3015, r2778;
}
{
add.f16x2 r3021, r3012, r3018;
}
{
sub.f16x2 r3024, r2815, r2816;
}
{
mul.f16x2 r3027, r3024, r2782;
}
{
add.f16x2 r3030, r3021, r3027;
}
{
add.f16x2 %10, r3006, r3030;
}
{
add.f16x2 r3036, r2785, r2786;
}
{
mul.f16x2 r3039, r3036, r2774;
}
{
add.f16x2 r3042, r2788, r3039;
}
{
add.f16x2 r3045, r2791, r2792;
}
{
mul.f16x2 r3048, r3045, r2780;
}
{
add.f16x2 r3051, r3042, r3048;
}
{
add.f16x2 r3054, r2797, r2798;
}
{
mul.f16x2 r3057, r3054, r2772;
}
{
add.f16x2 r3060, r3051, r3057;
}
{
sub.f16x2 r3063, r2803, r2804;
}
{
mul.f16x2 r3066, r3063, r2775;
}
{
sub.f16x2 r3069, r2809, r2810;
}
{
mul.f16x2 r3072, r3069, r2782;
}
{
add.f16x2 r3075, r3066, r3072;
}
{
sub.f16x2 r3078, r2815, r2816;
}
{
mul.f16x2 r3081, r3078, r2773;
}
{
add.f16x2 r3084, r3075, r3081;
}
{
sub.f16x2 %6, r3060, r3084;
}
{
add.f16x2 r3090, r2785, r2786;
}
{
mul.f16x2 r3093, r3090, r2774;
}
{
add.f16x2 r3096, r2788, r3093;
}
{
add.f16x2 r3099, r2791, r2792;
}
{
mul.f16x2 r3102, r3099, r2780;
}
{
add.f16x2 r3105, r3096, r3102;
}
{
add.f16x2 r3108, r2797, r2798;
}
{
mul.f16x2 r3111, r3108, r2772;
}
{
add.f16x2 r3114, r3105, r3111;
}
{
sub.f16x2 r3117, r2803, r2804;
}
{
mul.f16x2 r3120, r3117, r2775;
}
{
sub.f16x2 r3123, r2809, r2810;
}
{
mul.f16x2 r3126, r3123, r2782;
}
{
add.f16x2 r3129, r3120, r3126;
}
{
sub.f16x2 r3132, r2815, r2816;
}
{
mul.f16x2 r3135, r3132, r2773;
}
{
add.f16x2 r3138, r3129, r3135;
}
{
add.f16x2 %8, r3114, r3138;
}
{
add.f16x2 r3144, r2803, r2804;
}
{
mul.f16x2 r3147, r3144, r2770;
}
{
add.f16x2 r3150, r2806, r3147;
}
{
add.f16x2 r3153, r2809, r2810;
}
{
mul.f16x2 r3156, r3153, r2772;
}
{
add.f16x2 r3159, r3150, r3156;
}
{
add.f16x2 r3162, r2815, r2816;
}
{
mul.f16x2 r3165, r3162, r2774;
}
{
add.f16x2 r3168, r3159, r3165;
}
{
sub.f16x2 r3171, r2785, r2786;
}
{
mul.f16x2 r3174, r3171, r2771;
}
{
sub.f16x2 r3177, r2791, r2792;
}
{
mul.f16x2 r3180, r3177, r2773;
}
{
add.f16x2 r3183, r3174, r3180;
}
{
sub.f16x2 r3186, r2797, r2798;
}
{
mul.f16x2 r3189, r3186, r2775;
}
{
add.f16x2 r3192, r3183, r3189;
}
{
add.f16x2 %3, r3168, r3192;
}
{
add.f16x2 r3198, r2803, r2804;
}
{
mul.f16x2 r3201, r3198, r2770;
}
{
add.f16x2 r3204, r2806, r3201;
}
{
add.f16x2 r3207, r2809, r2810;
}
{
mul.f16x2 r3210, r3207, r2772;
}
{
add.f16x2 r3213, r3204, r3210;
}
{
add.f16x2 r3216, r2815, r2816;
}
{
mul.f16x2 r3219, r3216, r2774;
}
{
add.f16x2 r3222, r3213, r3219;
}
{
sub.f16x2 r3225, r2785, r2786;
}
{
mul.f16x2 r3228, r3225, r2771;
}
{
sub.f16x2 r3231, r2791, r2792;
}
{
mul.f16x2 r3234, r3231, r2773;
}
{
add.f16x2 r3237, r3228, r3234;
}
{
sub.f16x2 r3240, r2797, r2798;
}
{
mul.f16x2 r3243, r3240, r2775;
}
{
add.f16x2 r3246, r3237, r3243;
}
{
sub.f16x2 %13, r3222, r3246;
}
{
add.f16x2 r3252, r2803, r2804;
}
{
mul.f16x2 r3255, r3252, r2772;
}
{
add.f16x2 r3258, r2806, r3255;
}
{
add.f16x2 r3261, r2809, r2810;
}
{
mul.f16x2 r3264, r3261, r2776;
}
{
add.f16x2 r3267, r3258, r3264;
}
{
add.f16x2 r3270, r2815, r2816;
}
{
mul.f16x2 r3273, r3270, r2780;
}
{
add.f16x2 r3276, r3267, r3273;
}
{
sub.f16x2 r3279, r2785, r2786;
}
{
mul.f16x2 r3282, r3279, r2773;
}
{
sub.f16x2 r3285, r2791, r2792;
}
{
mul.f16x2 r3288, r3285, r2778;
}
{
add.f16x2 r3291, r3282, r3288;
}
{
sub.f16x2 r3294, r2797, r2798;
}
{
mul.f16x2 r3297, r3294, r2782;
}
{
add.f16x2 r3300, r3291, r3297;
}
{
add.f16x2 %5, r3276, r3300;
}
{
add.f16x2 r3306, r2803, r2804;
}
{
mul.f16x2 r3309, r3306, r2772;
}
{
add.f16x2 r3312, r2806, r3309;
}
{
add.f16x2 r3315, r2809, r2810;
}
{
mul.f16x2 r3318, r3315, r2776;
}
{
add.f16x2 r3321, r3312, r3318;
}
{
add.f16x2 r3324, r2815, r2816;
}
{
mul.f16x2 r3327, r3324, r2780;
}
{
add.f16x2 r3330, r3321, r3327;
}
{
sub.f16x2 r3333, r2785, r2786;
}
{
mul.f16x2 r3336, r3333, r2773;
}
{
sub.f16x2 r3339, r2791, r2792;
}
{
mul.f16x2 r3342, r3339, r2778;
}
{
add.f16x2 r3345, r3336, r3342;
}
{
sub.f16x2 r3348, r2797, r2798;
}
{
mul.f16x2 r3351, r3348, r2782;
}
{
add.f16x2 r3354, r3345, r3351;
}
{
sub.f16x2 %11, r3330, r3354;
}
{
add.f16x2 r3360, r2803, r2804;
}
{
mul.f16x2 r3363, r3360, r2774;
}
{
add.f16x2 r3366, r2806, r3363;
}
{
add.f16x2 r3369, r2809, r2810;
}
{
mul.f16x2 r3372, r3369, r2780;
}
{
add.f16x2 r3375, r3366, r3372;
}
{
add.f16x2 r3378, r2815, r2816;
}
{
mul.f16x2 r3381, r3378, r2772;
}
{
add.f16x2 r3384, r3375, r3381;
}
{
sub.f16x2 r3387, r2785, r2786;
}
{
mul.f16x2 r3390, r3387, r2775;
}
{
sub.f16x2 r3393, r2791, r2792;
}
{
mul.f16x2 r3396, r3393, r2782;
}
{
add.f16x2 r3399, r3390, r3396;
}
{
sub.f16x2 r3402, r2797, r2798;
}
{
mul.f16x2 r3405, r3402, r2773;
}
{
add.f16x2 r3408, r3399, r3405;
}
{
add.f16x2 %7, r3384, r3408;
}
{
add.f16x2 r3414, r2803, r2804;
}
{
mul.f16x2 r3417, r3414, r2774;
}
{
add.f16x2 r3420, r2806, r3417;
}
{
add.f16x2 r3423, r2809, r2810;
}
{
mul.f16x2 r3426, r3423, r2780;
}
{
add.f16x2 r3429, r3420, r3426;
}
{
add.f16x2 r3432, r2815, r2816;
}
{
mul.f16x2 r3435, r3432, r2772;
}
{
add.f16x2 r3438, r3429, r3435;
}
{
sub.f16x2 r3441, r2785, r2786;
}
{
mul.f16x2 r3444, r3441, r2775;
}
{
sub.f16x2 r3447, r2791, r2792;
}
{
mul.f16x2 r3450, r3447, r2782;
}
{
add.f16x2 r3453, r3444, r3450;
}
{
sub.f16x2 r3456, r2797, r2798;
}
{
mul.f16x2 r3459, r3456, r2773;
}
{
add.f16x2 r3462, r3453, r3459;
}
{
sub.f16x2 %9, r3438, r3462;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)));
};


#endif
