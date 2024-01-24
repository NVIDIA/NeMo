#ifndef CUFFTDX_FFT_49_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_49_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<918, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<60>;
.reg .b32 r<1636>;
.reg .b64 rd<4>;
mov.u32 r1622, %tid.x;
mov.f32 f54, 0f3F1F9D07;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f54;
cvt.rn.f16.f32 high, f54;
mov.b32 r1, {low, high};
}
mov.f32 f56, 0fBF48261C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r2, {low, high};
}
mov.f32 f42, 0fBE63DC87;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f42;
cvt.rn.f16.f32 high, f42;
mov.b32 r3, {low, high};
}
mov.f32 f44, 0fBF7994E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f44;
cvt.rn.f16.f32 high, f44;
mov.b32 r4, {low, high};
}
mov.f32 f50, 0fBF66A5E5;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r5, {low, high};
}
mov.f32 f52, 0fBEDE2602;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r6, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r7, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r8, {low, high};
}
{
neg.f16x2 r9, r8;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f54;
cvt.rn.f16.f32 high, f54;
mov.b32 r11, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
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
mul.wide.u32 rd2, r1622, 613566757;
shr.u64 rd3, rd2, 32;
cvt.u32.u64 r1623, rd3;
sub.s32 r1624, r1622, r1623;
shr.u32 r1625, r1624, 1;
add.s32 r1626, r1625, r1623;
shr.u32 r1627, r1626, 2;
mul.lo.s32 r1628, r1627, 7;
sub.s32 r1629, r1622, r1628;
cvt.rn.f32.u32 f57, r1629;
mul.f32 f58, f57, 0f3E034E46;
cos.approx.f32 f21, f58;
sin.approx.f32 f59, f58;
neg.f32 f22, f59;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f21;
cvt.rn.f16.f32 high, f22;
mov.b32 r699, {low, high};
}
mov.u32 r1630, %tid.y;
mov.u32 r1631, %14;
mad.lo.s32 r1632, r1630, 392, r1631;
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
mov.f32 f33, 0fBF800000;
mov.f32 f34, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
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
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
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
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
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
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
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
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
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
mad.lo.s32 r1633, r1627, 392, r1632;
barrier.sync 0;
mad.lo.s32 r1634, r1629, 56, r1633;
st.shared.v2.f32 [r1634], {r30, r48};
st.shared.v2.f32 [r1634+8], {r711, r718};
st.shared.v2.f32 [r1634+16], {r748, r755};
st.shared.v2.f32 [r1634+24], {r785, r792};
st.shared.v2.f32 [r1634+32], {r822, r829};
st.shared.v2.f32 [r1634+40], {r859, r866};
st.shared.v2.f32 [r1634+48], {r896, r903};
barrier.sync 0;
mad.lo.s32 r1635, r1629, -48, r1634;
ld.shared.u32 r942, [r1635];
ld.shared.u32 r960, [r1635+4];
ld.shared.u32 r939, [r1635+56];
ld.shared.u32 r957, [r1635+60];
ld.shared.u32 r945, [r1635+112];
ld.shared.u32 r963, [r1635+116];
ld.shared.u32 r951, [r1635+168];
ld.shared.u32 r969, [r1635+172];
ld.shared.u32 r952, [r1635+224];
ld.shared.u32 r970, [r1635+228];
ld.shared.u32 r946, [r1635+280];
ld.shared.u32 r964, [r1635+284];
ld.shared.u32 r940, [r1635+336];
ld.shared.u32 r958, [r1635+340];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f54;
cvt.rn.f16.f32 high, f54;
mov.b32 r924, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r925, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f42;
cvt.rn.f16.f32 high, f42;
mov.b32 r926, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f44;
cvt.rn.f16.f32 high, f44;
mov.b32 r927, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r928, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r929, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r930, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r931, {low, high};
}
{
neg.f16x2 r932, r931;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f54;
cvt.rn.f16.f32 high, f54;
mov.b32 r934, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
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
add.f16x2 %0, r947, r950;
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
add.f16x2 %1, r965, r968;
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
sub.f16x2 %2, r998, r1022;
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
add.f16x2 %12, r1052, r1076;
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
sub.f16x2 %4, r1106, r1130;
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
add.f16x2 %10, r1160, r1184;
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
sub.f16x2 %6, r1214, r1238;
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
add.f16x2 %8, r1268, r1292;
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
add.f16x2 %3, r1322, r1346;
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
sub.f16x2 %13, r1376, r1400;
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
add.f16x2 %5, r1430, r1454;
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
sub.f16x2 %11, r1484, r1508;
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
add.f16x2 %7, r1538, r1562;
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
sub.f16x2 %9, r1592, r1616;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<919, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<60>;
.reg .b32 r<1636>;
.reg .b64 rd<4>;
mov.u32 r1622, %tid.y;
mov.u32 r1623, %14;
mad.lo.s32 r1624, r1622, 196, r1623;
mov.u32 r1625, %tid.x;
mov.f32 f54, 0f3F1F9D07;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f54;
cvt.rn.f16.f32 high, f54;
mov.b32 r1, {low, high};
}
mov.f32 f56, 0fBF48261C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r2, {low, high};
}
mov.f32 f42, 0fBE63DC87;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f42;
cvt.rn.f16.f32 high, f42;
mov.b32 r3, {low, high};
}
mov.f32 f44, 0fBF7994E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f44;
cvt.rn.f16.f32 high, f44;
mov.b32 r4, {low, high};
}
mov.f32 f50, 0fBF66A5E5;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r5, {low, high};
}
mov.f32 f52, 0fBEDE2602;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r6, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r7, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r8, {low, high};
}
{
neg.f16x2 r9, r8;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f54;
cvt.rn.f16.f32 high, f54;
mov.b32 r11, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
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
mul.wide.u32 rd2, r1625, 613566757;
shr.u64 rd3, rd2, 32;
cvt.u32.u64 r1626, rd3;
sub.s32 r1627, r1625, r1626;
shr.u32 r1628, r1627, 1;
add.s32 r1629, r1628, r1626;
shr.u32 r1630, r1629, 2;
mul.lo.s32 r1631, r1630, 7;
sub.s32 r1632, r1625, r1631;
mad.lo.s32 r1633, r1630, 196, r1624;
cvt.rn.f32.u32 f57, r1632;
mul.f32 f58, f57, 0f3E034E46;
cos.approx.f32 f21, f58;
sin.approx.f32 f59, f58;
neg.f32 f22, f59;
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
mov.f32 f33, 0fBF800000;
mov.f32 f34, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
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
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
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
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
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
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
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
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
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
barrier.sync 0;
mad.lo.s32 r1634, r1632, 28, r1633;
st.shared.u32 [r1634], r30;
st.shared.u32 [r1634+4], r711;
st.shared.u32 [r1634+8], r748;
st.shared.u32 [r1634+12], r785;
st.shared.u32 [r1634+16], r822;
st.shared.u32 [r1634+20], r859;
st.shared.u32 [r1634+24], r896;
barrier.sync 0;
mad.lo.s32 r1635, r1632, -24, r1634;
ld.shared.u32 r942, [r1635];
ld.shared.u32 r939, [r1635+28];
ld.shared.u32 r945, [r1635+56];
ld.shared.u32 r951, [r1635+84];
ld.shared.u32 r952, [r1635+112];
ld.shared.u32 r946, [r1635+140];
ld.shared.u32 r940, [r1635+168];
barrier.sync 0;
st.shared.u32 [r1634], r48;
st.shared.u32 [r1634+4], r718;
st.shared.u32 [r1634+8], r755;
st.shared.u32 [r1634+12], r792;
st.shared.u32 [r1634+16], r829;
st.shared.u32 [r1634+20], r866;
st.shared.u32 [r1634+24], r903;
barrier.sync 0;
ld.shared.u32 r960, [r1635];
ld.shared.u32 r957, [r1635+28];
ld.shared.u32 r963, [r1635+56];
ld.shared.u32 r969, [r1635+84];
ld.shared.u32 r970, [r1635+112];
ld.shared.u32 r964, [r1635+140];
ld.shared.u32 r958, [r1635+168];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f54;
cvt.rn.f16.f32 high, f54;
mov.b32 r924, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r925, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f42;
cvt.rn.f16.f32 high, f42;
mov.b32 r926, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f44;
cvt.rn.f16.f32 high, f44;
mov.b32 r927, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r928, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r929, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r930, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r931, {low, high};
}
{
neg.f16x2 r932, r931;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f54;
cvt.rn.f16.f32 high, f54;
mov.b32 r934, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
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
add.f16x2 %0, r947, r950;
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
add.f16x2 %1, r965, r968;
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
sub.f16x2 %2, r998, r1022;
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
add.f16x2 %12, r1052, r1076;
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
sub.f16x2 %4, r1106, r1130;
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
add.f16x2 %10, r1160, r1184;
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
sub.f16x2 %6, r1214, r1238;
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
add.f16x2 %8, r1268, r1292;
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
add.f16x2 %3, r1322, r1346;
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
sub.f16x2 %13, r1376, r1400;
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
add.f16x2 %5, r1430, r1454;
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
sub.f16x2 %11, r1484, r1508;
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
add.f16x2 %7, r1538, r1562;
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
sub.f16x2 %9, r1592, r1616;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)));
};


#endif
