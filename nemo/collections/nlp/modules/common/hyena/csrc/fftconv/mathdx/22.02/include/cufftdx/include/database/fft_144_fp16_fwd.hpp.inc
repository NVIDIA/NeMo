#ifndef CUFFTDX_FFT_144_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_144_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<944, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<230>;
.reg .b32 r<1760>;
.reg .b64 rd<5>;
mov.u32 r1747, %tid.y;
shl.b32 r1748, r1747, 1;
mov.u32 r1749, %24;
mad.lo.s32 r1750, r1748, 576, r1749;
mov.u32 r1751, %tid.x;
mov.f32 f202, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1, {low, high};
}
mov.f32 f200, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2, {low, high};
}
{
neg.f16x2 r3, r2;
}
{
add.f16x2 r5, %33, %41;
}
{
add.f16x2 r8, %25, r5;
}
{
add.f16x2 r11, %34, %42;
}
{
add.f16x2 r14, %26, r11;
}
{
add.f16x2 r17, %33, %41;
}
{
mul.f16x2 r20, r17, r1;
}
{
add.f16x2 r23, %25, r20;
}
{
sub.f16x2 r26, %34, %42;
}
{
mul.f16x2 r29, r26, r3;
}
{
add.f16x2 r32, r23, r29;
}
{
add.f16x2 r35, %33, %41;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %25, r38;
}
{
sub.f16x2 r44, %34, %42;
}
{
mul.f16x2 r47, r44, r3;
}
{
sub.f16x2 r50, r41, r47;
}
{
add.f16x2 r53, %34, %42;
}
{
mul.f16x2 r56, r53, r1;
}
{
add.f16x2 r59, %26, r56;
}
{
sub.f16x2 r62, %33, %41;
}
{
mul.f16x2 r65, r62, r3;
}
{
sub.f16x2 r68, r59, r65;
}
{
add.f16x2 r71, %34, %42;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %26, r74;
}
{
sub.f16x2 r80, %33, %41;
}
{
mul.f16x2 r83, r80, r3;
}
{
add.f16x2 r86, r77, r83;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r90, {low, high};
}
{
neg.f16x2 r91, r90;
}
{
add.f16x2 r93, %37, %45;
}
{
add.f16x2 r96, %29, r93;
}
{
add.f16x2 r99, %38, %46;
}
{
add.f16x2 r102, %30, r99;
}
{
add.f16x2 r105, %37, %45;
}
{
mul.f16x2 r108, r105, r89;
}
{
add.f16x2 r111, %29, r108;
}
{
sub.f16x2 r114, %38, %46;
}
{
mul.f16x2 r117, r114, r91;
}
{
add.f16x2 r120, r111, r117;
}
{
add.f16x2 r123, %37, %45;
}
{
mul.f16x2 r126, r123, r89;
}
{
add.f16x2 r129, %29, r126;
}
{
sub.f16x2 r132, %38, %46;
}
{
mul.f16x2 r135, r132, r91;
}
{
sub.f16x2 r138, r129, r135;
}
{
add.f16x2 r141, %38, %46;
}
{
mul.f16x2 r144, r141, r89;
}
{
add.f16x2 r147, %30, r144;
}
{
sub.f16x2 r150, %37, %45;
}
{
mul.f16x2 r153, r150, r91;
}
{
sub.f16x2 r156, r147, r153;
}
{
add.f16x2 r159, %38, %46;
}
{
mul.f16x2 r162, r159, r89;
}
{
add.f16x2 r165, %30, r162;
}
{
sub.f16x2 r168, %37, %45;
}
{
mul.f16x2 r171, r168, r91;
}
{
add.f16x2 r174, r165, r171;
}
mov.f32 f188, 0f3F000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r177, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r178, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r179, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r180, {low, high};
}
mov.f32 f123, 0fBF800000;
mov.f32 f184, 0f3F5DB3D7;
{
mul.f16x2 r187, r120, r177;
}
{
mul.f16x2 r190, r156, r178;
}
{
sub.f16x2 r193, r187, r190;
}
{
mul.f16x2 r196, r120, r178;
}
{
fma.rn.f16x2 r199, r156, r177, r196;
}
{
mul.f16x2 r203, r138, r179;
}
{
mul.f16x2 r206, r174, r180;
}
{
sub.f16x2 r209, r203, r206;
}
{
mul.f16x2 r212, r138, r180;
}
{
fma.rn.f16x2 r215, r174, r179, r212;
}
{
add.f16x2 r219, r8, r96;
}
{
add.f16x2 r222, r14, r102;
}
{
sub.f16x2 r225, r8, r96;
}
{
sub.f16x2 r228, r14, r102;
}
{
add.f16x2 r231, r32, r193;
}
{
add.f16x2 r234, r68, r199;
}
{
sub.f16x2 r237, r32, r193;
}
{
sub.f16x2 r240, r68, r199;
}
{
add.f16x2 r243, r50, r209;
}
{
add.f16x2 r246, r86, r215;
}
{
sub.f16x2 r249, r50, r209;
}
{
sub.f16x2 r252, r86, r215;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r256, {low, high};
}
{
neg.f16x2 r257, r256;
}
{
add.f16x2 r259, %35, %43;
}
{
add.f16x2 r262, %27, r259;
}
{
add.f16x2 r265, %36, %44;
}
{
add.f16x2 r268, %28, r265;
}
{
add.f16x2 r271, %35, %43;
}
{
mul.f16x2 r274, r271, r255;
}
{
add.f16x2 r277, %27, r274;
}
{
sub.f16x2 r280, %36, %44;
}
{
mul.f16x2 r283, r280, r257;
}
{
add.f16x2 r286, r277, r283;
}
{
add.f16x2 r289, %35, %43;
}
{
mul.f16x2 r292, r289, r255;
}
{
add.f16x2 r295, %27, r292;
}
{
sub.f16x2 r298, %36, %44;
}
{
mul.f16x2 r301, r298, r257;
}
{
sub.f16x2 r304, r295, r301;
}
{
add.f16x2 r307, %36, %44;
}
{
mul.f16x2 r310, r307, r255;
}
{
add.f16x2 r313, %28, r310;
}
{
sub.f16x2 r316, %35, %43;
}
{
mul.f16x2 r319, r316, r257;
}
{
sub.f16x2 r322, r313, r319;
}
{
add.f16x2 r325, %36, %44;
}
{
mul.f16x2 r328, r325, r255;
}
{
add.f16x2 r331, %28, r328;
}
{
sub.f16x2 r334, %35, %43;
}
{
mul.f16x2 r337, r334, r257;
}
{
add.f16x2 r340, r331, r337;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r343, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r344, {low, high};
}
{
neg.f16x2 r345, r344;
}
{
add.f16x2 r347, %39, %47;
}
{
add.f16x2 r350, %31, r347;
}
{
add.f16x2 r353, %40, %48;
}
{
add.f16x2 r356, %32, r353;
}
{
add.f16x2 r359, %39, %47;
}
{
mul.f16x2 r362, r359, r343;
}
{
add.f16x2 r365, %31, r362;
}
{
sub.f16x2 r368, %40, %48;
}
{
mul.f16x2 r371, r368, r345;
}
{
add.f16x2 r374, r365, r371;
}
{
add.f16x2 r377, %39, %47;
}
{
mul.f16x2 r380, r377, r343;
}
{
add.f16x2 r383, %31, r380;
}
{
sub.f16x2 r386, %40, %48;
}
{
mul.f16x2 r389, r386, r345;
}
{
sub.f16x2 r392, r383, r389;
}
{
add.f16x2 r395, %40, %48;
}
{
mul.f16x2 r398, r395, r343;
}
{
add.f16x2 r401, %32, r398;
}
{
sub.f16x2 r404, %39, %47;
}
{
mul.f16x2 r407, r404, r345;
}
{
sub.f16x2 r410, r401, r407;
}
{
add.f16x2 r413, %40, %48;
}
{
mul.f16x2 r416, r413, r343;
}
{
add.f16x2 r419, %32, r416;
}
{
sub.f16x2 r422, %39, %47;
}
{
mul.f16x2 r425, r422, r345;
}
{
add.f16x2 r428, r419, r425;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r431, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r432, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r433, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r434, {low, high};
}
{
mul.f16x2 r441, r374, r431;
}
{
mul.f16x2 r444, r410, r432;
}
{
sub.f16x2 r447, r441, r444;
}
{
mul.f16x2 r450, r374, r432;
}
{
fma.rn.f16x2 r453, r410, r431, r450;
}
{
mul.f16x2 r457, r392, r433;
}
{
mul.f16x2 r460, r428, r434;
}
{
sub.f16x2 r463, r457, r460;
}
{
mul.f16x2 r466, r392, r434;
}
{
fma.rn.f16x2 r469, r428, r433, r466;
}
{
add.f16x2 r473, r262, r350;
}
{
add.f16x2 r476, r268, r356;
}
{
sub.f16x2 r479, r262, r350;
}
{
sub.f16x2 r482, r268, r356;
}
{
add.f16x2 r485, r286, r447;
}
{
add.f16x2 r488, r322, r453;
}
{
sub.f16x2 r491, r286, r447;
}
{
sub.f16x2 r494, r322, r453;
}
{
add.f16x2 r497, r304, r463;
}
{
add.f16x2 r500, r340, r469;
}
{
sub.f16x2 r503, r304, r463;
}
{
sub.f16x2 r506, r340, r469;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f184;
cvt.rn.f16.f32 high, f184;
mov.b32 r509, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r510, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r512, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r515, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r516, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r517, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r518, {low, high};
}
mov.f32 f124, 0f3F800000;
{
mul.f16x2 r531, r485, r509;
}
{
mul.f16x2 r534, r488, r510;
}
{
sub.f16x2 r537, r531, r534;
}
{
mul.f16x2 r540, r485, r510;
}
{
fma.rn.f16x2 r543, r488, r509, r540;
}
{
mul.f16x2 r547, r497, r511;
}
{
mul.f16x2 r550, r500, r512;
}
{
sub.f16x2 r553, r547, r550;
}
{
mul.f16x2 r556, r497, r512;
}
{
fma.rn.f16x2 r559, r500, r511, r556;
}
{
neg.f16x2 r563, r479;
}
{
mul.f16x2 r565, r491, r515;
}
{
mul.f16x2 r568, r494, r516;
}
{
sub.f16x2 r571, r565, r568;
}
{
mul.f16x2 r574, r491, r516;
}
{
fma.rn.f16x2 r577, r494, r515, r574;
}
{
mul.f16x2 r581, r503, r517;
}
{
mul.f16x2 r584, r506, r518;
}
{
sub.f16x2 r587, r581, r584;
}
{
mul.f16x2 r590, r503, r518;
}
{
fma.rn.f16x2 r593, r506, r517, r590;
}
{
add.f16x2 r597, r219, r473;
}
{
add.f16x2 r600, r222, r476;
}
{
sub.f16x2 r603, r219, r473;
}
{
sub.f16x2 r606, r222, r476;
}
{
add.f16x2 r609, r231, r537;
}
{
add.f16x2 r612, r234, r543;
}
{
sub.f16x2 r615, r231, r537;
}
{
sub.f16x2 r618, r234, r543;
}
{
add.f16x2 r621, r243, r553;
}
{
add.f16x2 r624, r246, r559;
}
{
sub.f16x2 r627, r243, r553;
}
{
sub.f16x2 r630, r246, r559;
}
{
add.f16x2 r633, r225, r482;
}
{
add.f16x2 r636, r228, r563;
}
{
sub.f16x2 r639, r225, r482;
}
{
sub.f16x2 r642, r228, r563;
}
{
add.f16x2 r645, r237, r571;
}
{
add.f16x2 r648, r240, r577;
}
{
sub.f16x2 r651, r237, r571;
}
{
sub.f16x2 r654, r240, r577;
}
{
add.f16x2 r657, r249, r587;
}
{
add.f16x2 r660, r252, r593;
}
{
sub.f16x2 r663, r249, r587;
}
{
sub.f16x2 r666, r252, r593;
}
mul.wide.u32 rd2, r1751, -1431655765;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r1752, rd3;
mul.lo.s32 r1753, r1752, 12;
sub.s32 r1754, r1751, r1753;
shr.u64 rd4, rd2, 34;
cvt.u32.u64 r1755, rd4;
and.b32 r1756, r1755, 1073741822;
mad.lo.s32 r1757, r1756, 576, r1750;
cvt.rn.f32.u32 f227, r1754;
mul.f32 f228, f227, 0f3D32B8C2;
cos.approx.f32 f101, f228;
sin.approx.f32 f229, f228;
neg.f32 f102, f229;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f101;
cvt.rn.f16.f32 high, f102;
mov.b32 r669, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r672, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r674, {high, high};
}
{
mul.f16x2 r676, r612, r674;
}
{
neg.f16x2 r679, r676;
}
{
fma.rn.f16x2 r681, r609, r672, r679;
}
{
mul.f16x2 r685, r609, r674;
}
{
fma.rn.f16x2 r688, r612, r672, r685;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r692, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r694, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r696, {low, high};
}
{
mul.f16x2 r697, r694, r696;
}
{
mul.f16x2 r700, r669, r692;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r703, {high, low};
}
{
fma.rn.f16x2 r705, r697, r703, r700;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r705;
mov.b32 r709, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r705;
mov.b32 r711, {high, high};
}
{
mul.f16x2 r713, r624, r711;
}
{
neg.f16x2 r716, r713;
}
{
fma.rn.f16x2 r718, r621, r709, r716;
}
{
mul.f16x2 r722, r621, r711;
}
{
fma.rn.f16x2 r725, r624, r709, r722;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r729, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r731, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r733, {low, high};
}
{
mul.f16x2 r734, r731, r733;
}
{
mul.f16x2 r737, r705, r729;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r705;
mov.b32 r740, {high, low};
}
{
fma.rn.f16x2 r742, r734, r740, r737;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r742;
mov.b32 r746, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r742;
mov.b32 r748, {high, high};
}
{
mul.f16x2 r750, r636, r748;
}
{
neg.f16x2 r753, r750;
}
{
fma.rn.f16x2 r755, r633, r746, r753;
}
{
mul.f16x2 r759, r633, r748;
}
{
fma.rn.f16x2 r762, r636, r746, r759;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r766, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r768, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r770, {low, high};
}
{
mul.f16x2 r771, r768, r770;
}
{
mul.f16x2 r774, r742, r766;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r742;
mov.b32 r777, {high, low};
}
{
fma.rn.f16x2 r779, r771, r777, r774;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r779;
mov.b32 r783, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r779;
mov.b32 r785, {high, high};
}
{
mul.f16x2 r787, r648, r785;
}
{
neg.f16x2 r790, r787;
}
{
fma.rn.f16x2 r792, r645, r783, r790;
}
{
mul.f16x2 r796, r645, r785;
}
{
fma.rn.f16x2 r799, r648, r783, r796;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r803, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r805, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r807, {low, high};
}
{
mul.f16x2 r808, r805, r807;
}
{
mul.f16x2 r811, r779, r803;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r779;
mov.b32 r814, {high, low};
}
{
fma.rn.f16x2 r816, r808, r814, r811;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r816;
mov.b32 r820, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r816;
mov.b32 r822, {high, high};
}
{
mul.f16x2 r824, r660, r822;
}
{
neg.f16x2 r827, r824;
}
{
fma.rn.f16x2 r829, r657, r820, r827;
}
{
mul.f16x2 r833, r657, r822;
}
{
fma.rn.f16x2 r836, r660, r820, r833;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r840, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r842, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r844, {low, high};
}
{
mul.f16x2 r845, r842, r844;
}
{
mul.f16x2 r848, r816, r840;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r816;
mov.b32 r851, {high, low};
}
{
fma.rn.f16x2 r853, r845, r851, r848;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r853;
mov.b32 r857, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r853;
mov.b32 r859, {high, high};
}
{
mul.f16x2 r861, r606, r859;
}
{
neg.f16x2 r864, r861;
}
{
fma.rn.f16x2 r866, r603, r857, r864;
}
{
mul.f16x2 r870, r603, r859;
}
{
fma.rn.f16x2 r873, r606, r857, r870;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r877, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r879, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r881, {low, high};
}
{
mul.f16x2 r882, r879, r881;
}
{
mul.f16x2 r885, r853, r877;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r853;
mov.b32 r888, {high, low};
}
{
fma.rn.f16x2 r890, r882, r888, r885;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r890;
mov.b32 r894, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r890;
mov.b32 r896, {high, high};
}
{
mul.f16x2 r898, r618, r896;
}
{
neg.f16x2 r901, r898;
}
{
fma.rn.f16x2 r903, r615, r894, r901;
}
{
mul.f16x2 r907, r615, r896;
}
{
fma.rn.f16x2 r910, r618, r894, r907;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r914, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r916, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r918, {low, high};
}
{
mul.f16x2 r919, r916, r918;
}
{
mul.f16x2 r922, r890, r914;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r890;
mov.b32 r925, {high, low};
}
{
fma.rn.f16x2 r927, r919, r925, r922;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r927;
mov.b32 r931, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r927;
mov.b32 r933, {high, high};
}
{
mul.f16x2 r935, r630, r933;
}
{
neg.f16x2 r938, r935;
}
{
fma.rn.f16x2 r940, r627, r931, r938;
}
{
mul.f16x2 r944, r627, r933;
}
{
fma.rn.f16x2 r947, r630, r931, r944;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r951, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r953, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r955, {low, high};
}
{
mul.f16x2 r956, r953, r955;
}
{
mul.f16x2 r959, r927, r951;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r927;
mov.b32 r962, {high, low};
}
{
fma.rn.f16x2 r964, r956, r962, r959;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r964;
mov.b32 r968, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r964;
mov.b32 r970, {high, high};
}
{
mul.f16x2 r972, r642, r970;
}
{
neg.f16x2 r975, r972;
}
{
fma.rn.f16x2 r977, r639, r968, r975;
}
{
mul.f16x2 r981, r639, r970;
}
{
fma.rn.f16x2 r984, r642, r968, r981;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r988, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r990, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r992, {low, high};
}
{
mul.f16x2 r993, r990, r992;
}
{
mul.f16x2 r996, r964, r988;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r964;
mov.b32 r999, {high, low};
}
{
fma.rn.f16x2 r1001, r993, r999, r996;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1001;
mov.b32 r1005, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1001;
mov.b32 r1007, {high, high};
}
{
mul.f16x2 r1009, r654, r1007;
}
{
neg.f16x2 r1012, r1009;
}
{
fma.rn.f16x2 r1014, r651, r1005, r1012;
}
{
mul.f16x2 r1018, r651, r1007;
}
{
fma.rn.f16x2 r1021, r654, r1005, r1018;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r1025, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r1027, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r1029, {low, high};
}
{
mul.f16x2 r1030, r1027, r1029;
}
{
mul.f16x2 r1033, r1001, r1025;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1001;
mov.b32 r1036, {high, low};
}
{
fma.rn.f16x2 r1038, r1030, r1036, r1033;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1038;
mov.b32 r1042, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1038;
mov.b32 r1044, {high, high};
}
{
mul.f16x2 r1046, r666, r1044;
}
{
neg.f16x2 r1049, r1046;
}
{
fma.rn.f16x2 r1051, r663, r1042, r1049;
}
{
mul.f16x2 r1055, r663, r1044;
}
{
fma.rn.f16x2 r1058, r666, r1042, r1055;
}
barrier.sync 0;
mad.lo.s32 r1758, r1754, 96, r1757;
st.shared.v4.f32 [r1758], {r597, r600, r681, r688};
st.shared.v4.f32 [r1758+16], {r718, r725, r755, r762};
st.shared.v4.f32 [r1758+32], {r792, r799, r829, r836};
st.shared.v4.f32 [r1758+48], {r866, r873, r903, r910};
st.shared.v4.f32 [r1758+64], {r940, r947, r977, r984};
st.shared.v4.f32 [r1758+80], {r1014, r1021, r1051, r1058};
barrier.sync 0;
mad.lo.s32 r1759, r1754, -88, r1758;
ld.shared.u32 r1087, [r1759];
ld.shared.u32 r1093, [r1759+4];
ld.shared.u32 r1341, [r1759+96];
ld.shared.u32 r1347, [r1759+100];
ld.shared.u32 r1175, [r1759+192];
ld.shared.u32 r1181, [r1759+196];
ld.shared.u32 r1429, [r1759+288];
ld.shared.u32 r1435, [r1759+292];
ld.shared.u32 r1084, [r1759+384];
ld.shared.u32 r1090, [r1759+388];
ld.shared.u32 r1338, [r1759+480];
ld.shared.u32 r1344, [r1759+484];
ld.shared.u32 r1172, [r1759+576];
ld.shared.u32 r1178, [r1759+580];
ld.shared.u32 r1426, [r1759+672];
ld.shared.u32 r1432, [r1759+676];
ld.shared.u32 r1085, [r1759+768];
ld.shared.u32 r1091, [r1759+772];
ld.shared.u32 r1339, [r1759+864];
ld.shared.u32 r1345, [r1759+868];
ld.shared.u32 r1173, [r1759+960];
ld.shared.u32 r1179, [r1759+964];
ld.shared.u32 r1427, [r1759+1056];
ld.shared.u32 r1433, [r1759+1060];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1079, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1080, {low, high};
}
{
neg.f16x2 r1081, r1080;
}
{
add.f16x2 r1083, r1084, r1085;
}
{
add.f16x2 r1086, r1087, r1083;
}
{
add.f16x2 r1089, r1090, r1091;
}
{
add.f16x2 r1092, r1093, r1089;
}
{
add.f16x2 r1095, r1084, r1085;
}
{
mul.f16x2 r1098, r1095, r1079;
}
{
add.f16x2 r1101, r1087, r1098;
}
{
sub.f16x2 r1104, r1090, r1091;
}
{
mul.f16x2 r1107, r1104, r1081;
}
{
add.f16x2 r1110, r1101, r1107;
}
{
add.f16x2 r1113, r1084, r1085;
}
{
mul.f16x2 r1116, r1113, r1079;
}
{
add.f16x2 r1119, r1087, r1116;
}
{
sub.f16x2 r1122, r1090, r1091;
}
{
mul.f16x2 r1125, r1122, r1081;
}
{
sub.f16x2 r1128, r1119, r1125;
}
{
add.f16x2 r1131, r1090, r1091;
}
{
mul.f16x2 r1134, r1131, r1079;
}
{
add.f16x2 r1137, r1093, r1134;
}
{
sub.f16x2 r1140, r1084, r1085;
}
{
mul.f16x2 r1143, r1140, r1081;
}
{
sub.f16x2 r1146, r1137, r1143;
}
{
add.f16x2 r1149, r1090, r1091;
}
{
mul.f16x2 r1152, r1149, r1079;
}
{
add.f16x2 r1155, r1093, r1152;
}
{
sub.f16x2 r1158, r1084, r1085;
}
{
mul.f16x2 r1161, r1158, r1081;
}
{
add.f16x2 r1164, r1155, r1161;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1167, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1168, {low, high};
}
{
neg.f16x2 r1169, r1168;
}
{
add.f16x2 r1171, r1172, r1173;
}
{
add.f16x2 r1174, r1175, r1171;
}
{
add.f16x2 r1177, r1178, r1179;
}
{
add.f16x2 r1180, r1181, r1177;
}
{
add.f16x2 r1183, r1172, r1173;
}
{
mul.f16x2 r1186, r1183, r1167;
}
{
add.f16x2 r1189, r1175, r1186;
}
{
sub.f16x2 r1192, r1178, r1179;
}
{
mul.f16x2 r1195, r1192, r1169;
}
{
add.f16x2 r1198, r1189, r1195;
}
{
add.f16x2 r1201, r1172, r1173;
}
{
mul.f16x2 r1204, r1201, r1167;
}
{
add.f16x2 r1207, r1175, r1204;
}
{
sub.f16x2 r1210, r1178, r1179;
}
{
mul.f16x2 r1213, r1210, r1169;
}
{
sub.f16x2 r1216, r1207, r1213;
}
{
add.f16x2 r1219, r1178, r1179;
}
{
mul.f16x2 r1222, r1219, r1167;
}
{
add.f16x2 r1225, r1181, r1222;
}
{
sub.f16x2 r1228, r1172, r1173;
}
{
mul.f16x2 r1231, r1228, r1169;
}
{
sub.f16x2 r1234, r1225, r1231;
}
{
add.f16x2 r1237, r1178, r1179;
}
{
mul.f16x2 r1240, r1237, r1167;
}
{
add.f16x2 r1243, r1181, r1240;
}
{
sub.f16x2 r1246, r1172, r1173;
}
{
mul.f16x2 r1249, r1246, r1169;
}
{
add.f16x2 r1252, r1243, r1249;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r1255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1256, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1257, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1258, {low, high};
}
{
mul.f16x2 r1265, r1198, r1255;
}
{
mul.f16x2 r1268, r1234, r1256;
}
{
sub.f16x2 r1271, r1265, r1268;
}
{
mul.f16x2 r1274, r1198, r1256;
}
{
fma.rn.f16x2 r1277, r1234, r1255, r1274;
}
{
mul.f16x2 r1281, r1216, r1257;
}
{
mul.f16x2 r1284, r1252, r1258;
}
{
sub.f16x2 r1287, r1281, r1284;
}
{
mul.f16x2 r1290, r1216, r1258;
}
{
fma.rn.f16x2 r1293, r1252, r1257, r1290;
}
{
add.f16x2 r1297, r1086, r1174;
}
{
add.f16x2 r1300, r1092, r1180;
}
{
sub.f16x2 r1303, r1086, r1174;
}
{
sub.f16x2 r1306, r1092, r1180;
}
{
add.f16x2 r1309, r1110, r1271;
}
{
add.f16x2 r1312, r1146, r1277;
}
{
sub.f16x2 r1315, r1110, r1271;
}
{
sub.f16x2 r1318, r1146, r1277;
}
{
add.f16x2 r1321, r1128, r1287;
}
{
add.f16x2 r1324, r1164, r1293;
}
{
sub.f16x2 r1327, r1128, r1287;
}
{
sub.f16x2 r1330, r1164, r1293;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1333, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1334, {low, high};
}
{
neg.f16x2 r1335, r1334;
}
{
add.f16x2 r1337, r1338, r1339;
}
{
add.f16x2 r1340, r1341, r1337;
}
{
add.f16x2 r1343, r1344, r1345;
}
{
add.f16x2 r1346, r1347, r1343;
}
{
add.f16x2 r1349, r1338, r1339;
}
{
mul.f16x2 r1352, r1349, r1333;
}
{
add.f16x2 r1355, r1341, r1352;
}
{
sub.f16x2 r1358, r1344, r1345;
}
{
mul.f16x2 r1361, r1358, r1335;
}
{
add.f16x2 r1364, r1355, r1361;
}
{
add.f16x2 r1367, r1338, r1339;
}
{
mul.f16x2 r1370, r1367, r1333;
}
{
add.f16x2 r1373, r1341, r1370;
}
{
sub.f16x2 r1376, r1344, r1345;
}
{
mul.f16x2 r1379, r1376, r1335;
}
{
sub.f16x2 r1382, r1373, r1379;
}
{
add.f16x2 r1385, r1344, r1345;
}
{
mul.f16x2 r1388, r1385, r1333;
}
{
add.f16x2 r1391, r1347, r1388;
}
{
sub.f16x2 r1394, r1338, r1339;
}
{
mul.f16x2 r1397, r1394, r1335;
}
{
sub.f16x2 r1400, r1391, r1397;
}
{
add.f16x2 r1403, r1344, r1345;
}
{
mul.f16x2 r1406, r1403, r1333;
}
{
add.f16x2 r1409, r1347, r1406;
}
{
sub.f16x2 r1412, r1338, r1339;
}
{
mul.f16x2 r1415, r1412, r1335;
}
{
add.f16x2 r1418, r1409, r1415;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1421, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1422, {low, high};
}
{
neg.f16x2 r1423, r1422;
}
{
add.f16x2 r1425, r1426, r1427;
}
{
add.f16x2 r1428, r1429, r1425;
}
{
add.f16x2 r1431, r1432, r1433;
}
{
add.f16x2 r1434, r1435, r1431;
}
{
add.f16x2 r1437, r1426, r1427;
}
{
mul.f16x2 r1440, r1437, r1421;
}
{
add.f16x2 r1443, r1429, r1440;
}
{
sub.f16x2 r1446, r1432, r1433;
}
{
mul.f16x2 r1449, r1446, r1423;
}
{
add.f16x2 r1452, r1443, r1449;
}
{
add.f16x2 r1455, r1426, r1427;
}
{
mul.f16x2 r1458, r1455, r1421;
}
{
add.f16x2 r1461, r1429, r1458;
}
{
sub.f16x2 r1464, r1432, r1433;
}
{
mul.f16x2 r1467, r1464, r1423;
}
{
sub.f16x2 r1470, r1461, r1467;
}
{
add.f16x2 r1473, r1432, r1433;
}
{
mul.f16x2 r1476, r1473, r1421;
}
{
add.f16x2 r1479, r1435, r1476;
}
{
sub.f16x2 r1482, r1426, r1427;
}
{
mul.f16x2 r1485, r1482, r1423;
}
{
sub.f16x2 r1488, r1479, r1485;
}
{
add.f16x2 r1491, r1432, r1433;
}
{
mul.f16x2 r1494, r1491, r1421;
}
{
add.f16x2 r1497, r1435, r1494;
}
{
sub.f16x2 r1500, r1426, r1427;
}
{
mul.f16x2 r1503, r1500, r1423;
}
{
add.f16x2 r1506, r1497, r1503;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r1509, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1510, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1512, {low, high};
}
{
mul.f16x2 r1519, r1452, r1509;
}
{
mul.f16x2 r1522, r1488, r1510;
}
{
sub.f16x2 r1525, r1519, r1522;
}
{
mul.f16x2 r1528, r1452, r1510;
}
{
fma.rn.f16x2 r1531, r1488, r1509, r1528;
}
{
mul.f16x2 r1535, r1470, r1511;
}
{
mul.f16x2 r1538, r1506, r1512;
}
{
sub.f16x2 r1541, r1535, r1538;
}
{
mul.f16x2 r1544, r1470, r1512;
}
{
fma.rn.f16x2 r1547, r1506, r1511, r1544;
}
{
add.f16x2 r1551, r1340, r1428;
}
{
add.f16x2 r1554, r1346, r1434;
}
{
sub.f16x2 r1557, r1340, r1428;
}
{
sub.f16x2 r1560, r1346, r1434;
}
{
add.f16x2 r1563, r1364, r1525;
}
{
add.f16x2 r1566, r1400, r1531;
}
{
sub.f16x2 r1569, r1364, r1525;
}
{
sub.f16x2 r1572, r1400, r1531;
}
{
add.f16x2 r1575, r1382, r1541;
}
{
add.f16x2 r1578, r1418, r1547;
}
{
sub.f16x2 r1581, r1382, r1541;
}
{
sub.f16x2 r1584, r1418, r1547;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f184;
cvt.rn.f16.f32 high, f184;
mov.b32 r1587, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1588, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r1589, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1590, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1593, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1594, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1595, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1596, {low, high};
}
{
mul.f16x2 r1609, r1563, r1587;
}
{
mul.f16x2 r1612, r1566, r1588;
}
{
sub.f16x2 r1615, r1609, r1612;
}
{
mul.f16x2 r1618, r1563, r1588;
}
{
fma.rn.f16x2 r1621, r1566, r1587, r1618;
}
{
mul.f16x2 r1625, r1575, r1589;
}
{
mul.f16x2 r1628, r1578, r1590;
}
{
sub.f16x2 r1631, r1625, r1628;
}
{
mul.f16x2 r1634, r1575, r1590;
}
{
fma.rn.f16x2 r1637, r1578, r1589, r1634;
}
{
neg.f16x2 r1641, r1557;
}
{
mul.f16x2 r1643, r1569, r1593;
}
{
mul.f16x2 r1646, r1572, r1594;
}
{
sub.f16x2 r1649, r1643, r1646;
}
{
mul.f16x2 r1652, r1569, r1594;
}
{
fma.rn.f16x2 r1655, r1572, r1593, r1652;
}
{
mul.f16x2 r1659, r1581, r1595;
}
{
mul.f16x2 r1662, r1584, r1596;
}
{
sub.f16x2 r1665, r1659, r1662;
}
{
mul.f16x2 r1668, r1581, r1596;
}
{
fma.rn.f16x2 r1671, r1584, r1595, r1668;
}
{
add.f16x2 %0, r1297, r1551;
}
{
add.f16x2 %1, r1300, r1554;
}
{
sub.f16x2 %12, r1297, r1551;
}
{
sub.f16x2 %13, r1300, r1554;
}
{
add.f16x2 %2, r1309, r1615;
}
{
add.f16x2 %3, r1312, r1621;
}
{
sub.f16x2 %14, r1309, r1615;
}
{
sub.f16x2 %15, r1312, r1621;
}
{
add.f16x2 %4, r1321, r1631;
}
{
add.f16x2 %5, r1324, r1637;
}
{
sub.f16x2 %16, r1321, r1631;
}
{
sub.f16x2 %17, r1324, r1637;
}
{
add.f16x2 %6, r1303, r1560;
}
{
add.f16x2 %7, r1306, r1641;
}
{
sub.f16x2 %18, r1303, r1560;
}
{
sub.f16x2 %19, r1306, r1641;
}
{
add.f16x2 %8, r1315, r1649;
}
{
add.f16x2 %9, r1318, r1655;
}
{
sub.f16x2 %20, r1315, r1649;
}
{
sub.f16x2 %21, r1318, r1655;
}
{
add.f16x2 %10, r1327, r1665;
}
{
add.f16x2 %11, r1330, r1671;
}
{
sub.f16x2 %22, r1327, r1665;
}
{
sub.f16x2 %23, r1330, r1671;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<945, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<230>;
.reg .b32 r<1757>;
.reg .b64 rd<4>;
mov.u32 r1747, %tid.y;
mov.u32 r1748, %24;
mad.lo.s32 r1749, r1747, 576, r1748;
mov.u32 r1750, %tid.x;
mov.f32 f202, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1, {low, high};
}
mov.f32 f200, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2, {low, high};
}
{
neg.f16x2 r3, r2;
}
{
add.f16x2 r5, %33, %41;
}
{
add.f16x2 r8, %25, r5;
}
{
add.f16x2 r11, %34, %42;
}
{
add.f16x2 r14, %26, r11;
}
{
add.f16x2 r17, %33, %41;
}
{
mul.f16x2 r20, r17, r1;
}
{
add.f16x2 r23, %25, r20;
}
{
sub.f16x2 r26, %34, %42;
}
{
mul.f16x2 r29, r26, r3;
}
{
add.f16x2 r32, r23, r29;
}
{
add.f16x2 r35, %33, %41;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %25, r38;
}
{
sub.f16x2 r44, %34, %42;
}
{
mul.f16x2 r47, r44, r3;
}
{
sub.f16x2 r50, r41, r47;
}
{
add.f16x2 r53, %34, %42;
}
{
mul.f16x2 r56, r53, r1;
}
{
add.f16x2 r59, %26, r56;
}
{
sub.f16x2 r62, %33, %41;
}
{
mul.f16x2 r65, r62, r3;
}
{
sub.f16x2 r68, r59, r65;
}
{
add.f16x2 r71, %34, %42;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %26, r74;
}
{
sub.f16x2 r80, %33, %41;
}
{
mul.f16x2 r83, r80, r3;
}
{
add.f16x2 r86, r77, r83;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r90, {low, high};
}
{
neg.f16x2 r91, r90;
}
{
add.f16x2 r93, %37, %45;
}
{
add.f16x2 r96, %29, r93;
}
{
add.f16x2 r99, %38, %46;
}
{
add.f16x2 r102, %30, r99;
}
{
add.f16x2 r105, %37, %45;
}
{
mul.f16x2 r108, r105, r89;
}
{
add.f16x2 r111, %29, r108;
}
{
sub.f16x2 r114, %38, %46;
}
{
mul.f16x2 r117, r114, r91;
}
{
add.f16x2 r120, r111, r117;
}
{
add.f16x2 r123, %37, %45;
}
{
mul.f16x2 r126, r123, r89;
}
{
add.f16x2 r129, %29, r126;
}
{
sub.f16x2 r132, %38, %46;
}
{
mul.f16x2 r135, r132, r91;
}
{
sub.f16x2 r138, r129, r135;
}
{
add.f16x2 r141, %38, %46;
}
{
mul.f16x2 r144, r141, r89;
}
{
add.f16x2 r147, %30, r144;
}
{
sub.f16x2 r150, %37, %45;
}
{
mul.f16x2 r153, r150, r91;
}
{
sub.f16x2 r156, r147, r153;
}
{
add.f16x2 r159, %38, %46;
}
{
mul.f16x2 r162, r159, r89;
}
{
add.f16x2 r165, %30, r162;
}
{
sub.f16x2 r168, %37, %45;
}
{
mul.f16x2 r171, r168, r91;
}
{
add.f16x2 r174, r165, r171;
}
mov.f32 f188, 0f3F000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r177, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r178, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r179, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r180, {low, high};
}
mov.f32 f123, 0fBF800000;
mov.f32 f184, 0f3F5DB3D7;
{
mul.f16x2 r187, r120, r177;
}
{
mul.f16x2 r190, r156, r178;
}
{
sub.f16x2 r193, r187, r190;
}
{
mul.f16x2 r196, r120, r178;
}
{
fma.rn.f16x2 r199, r156, r177, r196;
}
{
mul.f16x2 r203, r138, r179;
}
{
mul.f16x2 r206, r174, r180;
}
{
sub.f16x2 r209, r203, r206;
}
{
mul.f16x2 r212, r138, r180;
}
{
fma.rn.f16x2 r215, r174, r179, r212;
}
{
add.f16x2 r219, r8, r96;
}
{
add.f16x2 r222, r14, r102;
}
{
sub.f16x2 r225, r8, r96;
}
{
sub.f16x2 r228, r14, r102;
}
{
add.f16x2 r231, r32, r193;
}
{
add.f16x2 r234, r68, r199;
}
{
sub.f16x2 r237, r32, r193;
}
{
sub.f16x2 r240, r68, r199;
}
{
add.f16x2 r243, r50, r209;
}
{
add.f16x2 r246, r86, r215;
}
{
sub.f16x2 r249, r50, r209;
}
{
sub.f16x2 r252, r86, r215;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r256, {low, high};
}
{
neg.f16x2 r257, r256;
}
{
add.f16x2 r259, %35, %43;
}
{
add.f16x2 r262, %27, r259;
}
{
add.f16x2 r265, %36, %44;
}
{
add.f16x2 r268, %28, r265;
}
{
add.f16x2 r271, %35, %43;
}
{
mul.f16x2 r274, r271, r255;
}
{
add.f16x2 r277, %27, r274;
}
{
sub.f16x2 r280, %36, %44;
}
{
mul.f16x2 r283, r280, r257;
}
{
add.f16x2 r286, r277, r283;
}
{
add.f16x2 r289, %35, %43;
}
{
mul.f16x2 r292, r289, r255;
}
{
add.f16x2 r295, %27, r292;
}
{
sub.f16x2 r298, %36, %44;
}
{
mul.f16x2 r301, r298, r257;
}
{
sub.f16x2 r304, r295, r301;
}
{
add.f16x2 r307, %36, %44;
}
{
mul.f16x2 r310, r307, r255;
}
{
add.f16x2 r313, %28, r310;
}
{
sub.f16x2 r316, %35, %43;
}
{
mul.f16x2 r319, r316, r257;
}
{
sub.f16x2 r322, r313, r319;
}
{
add.f16x2 r325, %36, %44;
}
{
mul.f16x2 r328, r325, r255;
}
{
add.f16x2 r331, %28, r328;
}
{
sub.f16x2 r334, %35, %43;
}
{
mul.f16x2 r337, r334, r257;
}
{
add.f16x2 r340, r331, r337;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r343, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r344, {low, high};
}
{
neg.f16x2 r345, r344;
}
{
add.f16x2 r347, %39, %47;
}
{
add.f16x2 r350, %31, r347;
}
{
add.f16x2 r353, %40, %48;
}
{
add.f16x2 r356, %32, r353;
}
{
add.f16x2 r359, %39, %47;
}
{
mul.f16x2 r362, r359, r343;
}
{
add.f16x2 r365, %31, r362;
}
{
sub.f16x2 r368, %40, %48;
}
{
mul.f16x2 r371, r368, r345;
}
{
add.f16x2 r374, r365, r371;
}
{
add.f16x2 r377, %39, %47;
}
{
mul.f16x2 r380, r377, r343;
}
{
add.f16x2 r383, %31, r380;
}
{
sub.f16x2 r386, %40, %48;
}
{
mul.f16x2 r389, r386, r345;
}
{
sub.f16x2 r392, r383, r389;
}
{
add.f16x2 r395, %40, %48;
}
{
mul.f16x2 r398, r395, r343;
}
{
add.f16x2 r401, %32, r398;
}
{
sub.f16x2 r404, %39, %47;
}
{
mul.f16x2 r407, r404, r345;
}
{
sub.f16x2 r410, r401, r407;
}
{
add.f16x2 r413, %40, %48;
}
{
mul.f16x2 r416, r413, r343;
}
{
add.f16x2 r419, %32, r416;
}
{
sub.f16x2 r422, %39, %47;
}
{
mul.f16x2 r425, r422, r345;
}
{
add.f16x2 r428, r419, r425;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r431, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r432, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r433, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r434, {low, high};
}
{
mul.f16x2 r441, r374, r431;
}
{
mul.f16x2 r444, r410, r432;
}
{
sub.f16x2 r447, r441, r444;
}
{
mul.f16x2 r450, r374, r432;
}
{
fma.rn.f16x2 r453, r410, r431, r450;
}
{
mul.f16x2 r457, r392, r433;
}
{
mul.f16x2 r460, r428, r434;
}
{
sub.f16x2 r463, r457, r460;
}
{
mul.f16x2 r466, r392, r434;
}
{
fma.rn.f16x2 r469, r428, r433, r466;
}
{
add.f16x2 r473, r262, r350;
}
{
add.f16x2 r476, r268, r356;
}
{
sub.f16x2 r479, r262, r350;
}
{
sub.f16x2 r482, r268, r356;
}
{
add.f16x2 r485, r286, r447;
}
{
add.f16x2 r488, r322, r453;
}
{
sub.f16x2 r491, r286, r447;
}
{
sub.f16x2 r494, r322, r453;
}
{
add.f16x2 r497, r304, r463;
}
{
add.f16x2 r500, r340, r469;
}
{
sub.f16x2 r503, r304, r463;
}
{
sub.f16x2 r506, r340, r469;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f184;
cvt.rn.f16.f32 high, f184;
mov.b32 r509, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r510, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r512, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r515, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r516, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r517, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r518, {low, high};
}
mov.f32 f124, 0f3F800000;
{
mul.f16x2 r531, r485, r509;
}
{
mul.f16x2 r534, r488, r510;
}
{
sub.f16x2 r537, r531, r534;
}
{
mul.f16x2 r540, r485, r510;
}
{
fma.rn.f16x2 r543, r488, r509, r540;
}
{
mul.f16x2 r547, r497, r511;
}
{
mul.f16x2 r550, r500, r512;
}
{
sub.f16x2 r553, r547, r550;
}
{
mul.f16x2 r556, r497, r512;
}
{
fma.rn.f16x2 r559, r500, r511, r556;
}
{
neg.f16x2 r563, r479;
}
{
mul.f16x2 r565, r491, r515;
}
{
mul.f16x2 r568, r494, r516;
}
{
sub.f16x2 r571, r565, r568;
}
{
mul.f16x2 r574, r491, r516;
}
{
fma.rn.f16x2 r577, r494, r515, r574;
}
{
mul.f16x2 r581, r503, r517;
}
{
mul.f16x2 r584, r506, r518;
}
{
sub.f16x2 r587, r581, r584;
}
{
mul.f16x2 r590, r503, r518;
}
{
fma.rn.f16x2 r593, r506, r517, r590;
}
{
add.f16x2 r597, r219, r473;
}
{
add.f16x2 r600, r222, r476;
}
{
sub.f16x2 r603, r219, r473;
}
{
sub.f16x2 r606, r222, r476;
}
{
add.f16x2 r609, r231, r537;
}
{
add.f16x2 r612, r234, r543;
}
{
sub.f16x2 r615, r231, r537;
}
{
sub.f16x2 r618, r234, r543;
}
{
add.f16x2 r621, r243, r553;
}
{
add.f16x2 r624, r246, r559;
}
{
sub.f16x2 r627, r243, r553;
}
{
sub.f16x2 r630, r246, r559;
}
{
add.f16x2 r633, r225, r482;
}
{
add.f16x2 r636, r228, r563;
}
{
sub.f16x2 r639, r225, r482;
}
{
sub.f16x2 r642, r228, r563;
}
{
add.f16x2 r645, r237, r571;
}
{
add.f16x2 r648, r240, r577;
}
{
sub.f16x2 r651, r237, r571;
}
{
sub.f16x2 r654, r240, r577;
}
{
add.f16x2 r657, r249, r587;
}
{
add.f16x2 r660, r252, r593;
}
{
sub.f16x2 r663, r249, r587;
}
{
sub.f16x2 r666, r252, r593;
}
mul.wide.u32 rd2, r1750, -1431655765;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r1751, rd3;
mul.lo.s32 r1752, r1751, 12;
sub.s32 r1753, r1750, r1752;
mad.lo.s32 r1754, r1751, 576, r1749;
cvt.rn.f32.u32 f227, r1753;
mul.f32 f228, f227, 0f3D32B8C2;
cos.approx.f32 f101, f228;
sin.approx.f32 f229, f228;
neg.f32 f102, f229;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f101;
cvt.rn.f16.f32 high, f102;
mov.b32 r669, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r672, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r674, {high, high};
}
{
mul.f16x2 r676, r612, r674;
}
{
neg.f16x2 r679, r676;
}
{
fma.rn.f16x2 r681, r609, r672, r679;
}
{
mul.f16x2 r685, r609, r674;
}
{
fma.rn.f16x2 r688, r612, r672, r685;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r692, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r694, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r696, {low, high};
}
{
mul.f16x2 r697, r694, r696;
}
{
mul.f16x2 r700, r669, r692;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r703, {high, low};
}
{
fma.rn.f16x2 r705, r697, r703, r700;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r705;
mov.b32 r709, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r705;
mov.b32 r711, {high, high};
}
{
mul.f16x2 r713, r624, r711;
}
{
neg.f16x2 r716, r713;
}
{
fma.rn.f16x2 r718, r621, r709, r716;
}
{
mul.f16x2 r722, r621, r711;
}
{
fma.rn.f16x2 r725, r624, r709, r722;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r729, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r731, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r733, {low, high};
}
{
mul.f16x2 r734, r731, r733;
}
{
mul.f16x2 r737, r705, r729;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r705;
mov.b32 r740, {high, low};
}
{
fma.rn.f16x2 r742, r734, r740, r737;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r742;
mov.b32 r746, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r742;
mov.b32 r748, {high, high};
}
{
mul.f16x2 r750, r636, r748;
}
{
neg.f16x2 r753, r750;
}
{
fma.rn.f16x2 r755, r633, r746, r753;
}
{
mul.f16x2 r759, r633, r748;
}
{
fma.rn.f16x2 r762, r636, r746, r759;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r766, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r768, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r770, {low, high};
}
{
mul.f16x2 r771, r768, r770;
}
{
mul.f16x2 r774, r742, r766;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r742;
mov.b32 r777, {high, low};
}
{
fma.rn.f16x2 r779, r771, r777, r774;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r779;
mov.b32 r783, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r779;
mov.b32 r785, {high, high};
}
{
mul.f16x2 r787, r648, r785;
}
{
neg.f16x2 r790, r787;
}
{
fma.rn.f16x2 r792, r645, r783, r790;
}
{
mul.f16x2 r796, r645, r785;
}
{
fma.rn.f16x2 r799, r648, r783, r796;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r803, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r805, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r807, {low, high};
}
{
mul.f16x2 r808, r805, r807;
}
{
mul.f16x2 r811, r779, r803;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r779;
mov.b32 r814, {high, low};
}
{
fma.rn.f16x2 r816, r808, r814, r811;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r816;
mov.b32 r820, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r816;
mov.b32 r822, {high, high};
}
{
mul.f16x2 r824, r660, r822;
}
{
neg.f16x2 r827, r824;
}
{
fma.rn.f16x2 r829, r657, r820, r827;
}
{
mul.f16x2 r833, r657, r822;
}
{
fma.rn.f16x2 r836, r660, r820, r833;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r840, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r842, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r844, {low, high};
}
{
mul.f16x2 r845, r842, r844;
}
{
mul.f16x2 r848, r816, r840;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r816;
mov.b32 r851, {high, low};
}
{
fma.rn.f16x2 r853, r845, r851, r848;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r853;
mov.b32 r857, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r853;
mov.b32 r859, {high, high};
}
{
mul.f16x2 r861, r606, r859;
}
{
neg.f16x2 r864, r861;
}
{
fma.rn.f16x2 r866, r603, r857, r864;
}
{
mul.f16x2 r870, r603, r859;
}
{
fma.rn.f16x2 r873, r606, r857, r870;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r877, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r879, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r881, {low, high};
}
{
mul.f16x2 r882, r879, r881;
}
{
mul.f16x2 r885, r853, r877;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r853;
mov.b32 r888, {high, low};
}
{
fma.rn.f16x2 r890, r882, r888, r885;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r890;
mov.b32 r894, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r890;
mov.b32 r896, {high, high};
}
{
mul.f16x2 r898, r618, r896;
}
{
neg.f16x2 r901, r898;
}
{
fma.rn.f16x2 r903, r615, r894, r901;
}
{
mul.f16x2 r907, r615, r896;
}
{
fma.rn.f16x2 r910, r618, r894, r907;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r914, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r916, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r918, {low, high};
}
{
mul.f16x2 r919, r916, r918;
}
{
mul.f16x2 r922, r890, r914;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r890;
mov.b32 r925, {high, low};
}
{
fma.rn.f16x2 r927, r919, r925, r922;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r927;
mov.b32 r931, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r927;
mov.b32 r933, {high, high};
}
{
mul.f16x2 r935, r630, r933;
}
{
neg.f16x2 r938, r935;
}
{
fma.rn.f16x2 r940, r627, r931, r938;
}
{
mul.f16x2 r944, r627, r933;
}
{
fma.rn.f16x2 r947, r630, r931, r944;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r951, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r953, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r955, {low, high};
}
{
mul.f16x2 r956, r953, r955;
}
{
mul.f16x2 r959, r927, r951;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r927;
mov.b32 r962, {high, low};
}
{
fma.rn.f16x2 r964, r956, r962, r959;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r964;
mov.b32 r968, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r964;
mov.b32 r970, {high, high};
}
{
mul.f16x2 r972, r642, r970;
}
{
neg.f16x2 r975, r972;
}
{
fma.rn.f16x2 r977, r639, r968, r975;
}
{
mul.f16x2 r981, r639, r970;
}
{
fma.rn.f16x2 r984, r642, r968, r981;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r988, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r990, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r992, {low, high};
}
{
mul.f16x2 r993, r990, r992;
}
{
mul.f16x2 r996, r964, r988;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r964;
mov.b32 r999, {high, low};
}
{
fma.rn.f16x2 r1001, r993, r999, r996;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1001;
mov.b32 r1005, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1001;
mov.b32 r1007, {high, high};
}
{
mul.f16x2 r1009, r654, r1007;
}
{
neg.f16x2 r1012, r1009;
}
{
fma.rn.f16x2 r1014, r651, r1005, r1012;
}
{
mul.f16x2 r1018, r651, r1007;
}
{
fma.rn.f16x2 r1021, r654, r1005, r1018;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r1025, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r669;
mov.b32 r1027, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r1029, {low, high};
}
{
mul.f16x2 r1030, r1027, r1029;
}
{
mul.f16x2 r1033, r1001, r1025;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1001;
mov.b32 r1036, {high, low};
}
{
fma.rn.f16x2 r1038, r1030, r1036, r1033;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1038;
mov.b32 r1042, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1038;
mov.b32 r1044, {high, high};
}
{
mul.f16x2 r1046, r666, r1044;
}
{
neg.f16x2 r1049, r1046;
}
{
fma.rn.f16x2 r1051, r663, r1042, r1049;
}
{
mul.f16x2 r1055, r663, r1044;
}
{
fma.rn.f16x2 r1058, r666, r1042, r1055;
}
barrier.sync 0;
mad.lo.s32 r1755, r1753, 48, r1754;
st.shared.v4.f32 [r1755], {r597, r681, r718, r755};
st.shared.v4.f32 [r1755+16], {r792, r829, r866, r903};
st.shared.v4.f32 [r1755+32], {r940, r977, r1014, r1051};
barrier.sync 0;
mad.lo.s32 r1756, r1753, -44, r1755;
ld.shared.u32 r1087, [r1756];
ld.shared.u32 r1341, [r1756+48];
ld.shared.u32 r1175, [r1756+96];
ld.shared.u32 r1429, [r1756+144];
ld.shared.u32 r1084, [r1756+192];
ld.shared.u32 r1338, [r1756+240];
ld.shared.u32 r1172, [r1756+288];
ld.shared.u32 r1426, [r1756+336];
ld.shared.u32 r1085, [r1756+384];
ld.shared.u32 r1339, [r1756+432];
ld.shared.u32 r1173, [r1756+480];
ld.shared.u32 r1427, [r1756+528];
barrier.sync 0;
st.shared.v4.f32 [r1755], {r600, r688, r725, r762};
st.shared.v4.f32 [r1755+16], {r799, r836, r873, r910};
st.shared.v4.f32 [r1755+32], {r947, r984, r1021, r1058};
barrier.sync 0;
ld.shared.u32 r1093, [r1756];
ld.shared.u32 r1347, [r1756+48];
ld.shared.u32 r1181, [r1756+96];
ld.shared.u32 r1435, [r1756+144];
ld.shared.u32 r1090, [r1756+192];
ld.shared.u32 r1344, [r1756+240];
ld.shared.u32 r1178, [r1756+288];
ld.shared.u32 r1432, [r1756+336];
ld.shared.u32 r1091, [r1756+384];
ld.shared.u32 r1345, [r1756+432];
ld.shared.u32 r1179, [r1756+480];
ld.shared.u32 r1433, [r1756+528];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1079, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1080, {low, high};
}
{
neg.f16x2 r1081, r1080;
}
{
add.f16x2 r1083, r1084, r1085;
}
{
add.f16x2 r1086, r1087, r1083;
}
{
add.f16x2 r1089, r1090, r1091;
}
{
add.f16x2 r1092, r1093, r1089;
}
{
add.f16x2 r1095, r1084, r1085;
}
{
mul.f16x2 r1098, r1095, r1079;
}
{
add.f16x2 r1101, r1087, r1098;
}
{
sub.f16x2 r1104, r1090, r1091;
}
{
mul.f16x2 r1107, r1104, r1081;
}
{
add.f16x2 r1110, r1101, r1107;
}
{
add.f16x2 r1113, r1084, r1085;
}
{
mul.f16x2 r1116, r1113, r1079;
}
{
add.f16x2 r1119, r1087, r1116;
}
{
sub.f16x2 r1122, r1090, r1091;
}
{
mul.f16x2 r1125, r1122, r1081;
}
{
sub.f16x2 r1128, r1119, r1125;
}
{
add.f16x2 r1131, r1090, r1091;
}
{
mul.f16x2 r1134, r1131, r1079;
}
{
add.f16x2 r1137, r1093, r1134;
}
{
sub.f16x2 r1140, r1084, r1085;
}
{
mul.f16x2 r1143, r1140, r1081;
}
{
sub.f16x2 r1146, r1137, r1143;
}
{
add.f16x2 r1149, r1090, r1091;
}
{
mul.f16x2 r1152, r1149, r1079;
}
{
add.f16x2 r1155, r1093, r1152;
}
{
sub.f16x2 r1158, r1084, r1085;
}
{
mul.f16x2 r1161, r1158, r1081;
}
{
add.f16x2 r1164, r1155, r1161;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1167, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1168, {low, high};
}
{
neg.f16x2 r1169, r1168;
}
{
add.f16x2 r1171, r1172, r1173;
}
{
add.f16x2 r1174, r1175, r1171;
}
{
add.f16x2 r1177, r1178, r1179;
}
{
add.f16x2 r1180, r1181, r1177;
}
{
add.f16x2 r1183, r1172, r1173;
}
{
mul.f16x2 r1186, r1183, r1167;
}
{
add.f16x2 r1189, r1175, r1186;
}
{
sub.f16x2 r1192, r1178, r1179;
}
{
mul.f16x2 r1195, r1192, r1169;
}
{
add.f16x2 r1198, r1189, r1195;
}
{
add.f16x2 r1201, r1172, r1173;
}
{
mul.f16x2 r1204, r1201, r1167;
}
{
add.f16x2 r1207, r1175, r1204;
}
{
sub.f16x2 r1210, r1178, r1179;
}
{
mul.f16x2 r1213, r1210, r1169;
}
{
sub.f16x2 r1216, r1207, r1213;
}
{
add.f16x2 r1219, r1178, r1179;
}
{
mul.f16x2 r1222, r1219, r1167;
}
{
add.f16x2 r1225, r1181, r1222;
}
{
sub.f16x2 r1228, r1172, r1173;
}
{
mul.f16x2 r1231, r1228, r1169;
}
{
sub.f16x2 r1234, r1225, r1231;
}
{
add.f16x2 r1237, r1178, r1179;
}
{
mul.f16x2 r1240, r1237, r1167;
}
{
add.f16x2 r1243, r1181, r1240;
}
{
sub.f16x2 r1246, r1172, r1173;
}
{
mul.f16x2 r1249, r1246, r1169;
}
{
add.f16x2 r1252, r1243, r1249;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r1255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1256, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1257, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1258, {low, high};
}
{
mul.f16x2 r1265, r1198, r1255;
}
{
mul.f16x2 r1268, r1234, r1256;
}
{
sub.f16x2 r1271, r1265, r1268;
}
{
mul.f16x2 r1274, r1198, r1256;
}
{
fma.rn.f16x2 r1277, r1234, r1255, r1274;
}
{
mul.f16x2 r1281, r1216, r1257;
}
{
mul.f16x2 r1284, r1252, r1258;
}
{
sub.f16x2 r1287, r1281, r1284;
}
{
mul.f16x2 r1290, r1216, r1258;
}
{
fma.rn.f16x2 r1293, r1252, r1257, r1290;
}
{
add.f16x2 r1297, r1086, r1174;
}
{
add.f16x2 r1300, r1092, r1180;
}
{
sub.f16x2 r1303, r1086, r1174;
}
{
sub.f16x2 r1306, r1092, r1180;
}
{
add.f16x2 r1309, r1110, r1271;
}
{
add.f16x2 r1312, r1146, r1277;
}
{
sub.f16x2 r1315, r1110, r1271;
}
{
sub.f16x2 r1318, r1146, r1277;
}
{
add.f16x2 r1321, r1128, r1287;
}
{
add.f16x2 r1324, r1164, r1293;
}
{
sub.f16x2 r1327, r1128, r1287;
}
{
sub.f16x2 r1330, r1164, r1293;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1333, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1334, {low, high};
}
{
neg.f16x2 r1335, r1334;
}
{
add.f16x2 r1337, r1338, r1339;
}
{
add.f16x2 r1340, r1341, r1337;
}
{
add.f16x2 r1343, r1344, r1345;
}
{
add.f16x2 r1346, r1347, r1343;
}
{
add.f16x2 r1349, r1338, r1339;
}
{
mul.f16x2 r1352, r1349, r1333;
}
{
add.f16x2 r1355, r1341, r1352;
}
{
sub.f16x2 r1358, r1344, r1345;
}
{
mul.f16x2 r1361, r1358, r1335;
}
{
add.f16x2 r1364, r1355, r1361;
}
{
add.f16x2 r1367, r1338, r1339;
}
{
mul.f16x2 r1370, r1367, r1333;
}
{
add.f16x2 r1373, r1341, r1370;
}
{
sub.f16x2 r1376, r1344, r1345;
}
{
mul.f16x2 r1379, r1376, r1335;
}
{
sub.f16x2 r1382, r1373, r1379;
}
{
add.f16x2 r1385, r1344, r1345;
}
{
mul.f16x2 r1388, r1385, r1333;
}
{
add.f16x2 r1391, r1347, r1388;
}
{
sub.f16x2 r1394, r1338, r1339;
}
{
mul.f16x2 r1397, r1394, r1335;
}
{
sub.f16x2 r1400, r1391, r1397;
}
{
add.f16x2 r1403, r1344, r1345;
}
{
mul.f16x2 r1406, r1403, r1333;
}
{
add.f16x2 r1409, r1347, r1406;
}
{
sub.f16x2 r1412, r1338, r1339;
}
{
mul.f16x2 r1415, r1412, r1335;
}
{
add.f16x2 r1418, r1409, r1415;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1421, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1422, {low, high};
}
{
neg.f16x2 r1423, r1422;
}
{
add.f16x2 r1425, r1426, r1427;
}
{
add.f16x2 r1428, r1429, r1425;
}
{
add.f16x2 r1431, r1432, r1433;
}
{
add.f16x2 r1434, r1435, r1431;
}
{
add.f16x2 r1437, r1426, r1427;
}
{
mul.f16x2 r1440, r1437, r1421;
}
{
add.f16x2 r1443, r1429, r1440;
}
{
sub.f16x2 r1446, r1432, r1433;
}
{
mul.f16x2 r1449, r1446, r1423;
}
{
add.f16x2 r1452, r1443, r1449;
}
{
add.f16x2 r1455, r1426, r1427;
}
{
mul.f16x2 r1458, r1455, r1421;
}
{
add.f16x2 r1461, r1429, r1458;
}
{
sub.f16x2 r1464, r1432, r1433;
}
{
mul.f16x2 r1467, r1464, r1423;
}
{
sub.f16x2 r1470, r1461, r1467;
}
{
add.f16x2 r1473, r1432, r1433;
}
{
mul.f16x2 r1476, r1473, r1421;
}
{
add.f16x2 r1479, r1435, r1476;
}
{
sub.f16x2 r1482, r1426, r1427;
}
{
mul.f16x2 r1485, r1482, r1423;
}
{
sub.f16x2 r1488, r1479, r1485;
}
{
add.f16x2 r1491, r1432, r1433;
}
{
mul.f16x2 r1494, r1491, r1421;
}
{
add.f16x2 r1497, r1435, r1494;
}
{
sub.f16x2 r1500, r1426, r1427;
}
{
mul.f16x2 r1503, r1500, r1423;
}
{
add.f16x2 r1506, r1497, r1503;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r1509, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1510, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1512, {low, high};
}
{
mul.f16x2 r1519, r1452, r1509;
}
{
mul.f16x2 r1522, r1488, r1510;
}
{
sub.f16x2 r1525, r1519, r1522;
}
{
mul.f16x2 r1528, r1452, r1510;
}
{
fma.rn.f16x2 r1531, r1488, r1509, r1528;
}
{
mul.f16x2 r1535, r1470, r1511;
}
{
mul.f16x2 r1538, r1506, r1512;
}
{
sub.f16x2 r1541, r1535, r1538;
}
{
mul.f16x2 r1544, r1470, r1512;
}
{
fma.rn.f16x2 r1547, r1506, r1511, r1544;
}
{
add.f16x2 r1551, r1340, r1428;
}
{
add.f16x2 r1554, r1346, r1434;
}
{
sub.f16x2 r1557, r1340, r1428;
}
{
sub.f16x2 r1560, r1346, r1434;
}
{
add.f16x2 r1563, r1364, r1525;
}
{
add.f16x2 r1566, r1400, r1531;
}
{
sub.f16x2 r1569, r1364, r1525;
}
{
sub.f16x2 r1572, r1400, r1531;
}
{
add.f16x2 r1575, r1382, r1541;
}
{
add.f16x2 r1578, r1418, r1547;
}
{
sub.f16x2 r1581, r1382, r1541;
}
{
sub.f16x2 r1584, r1418, r1547;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f184;
cvt.rn.f16.f32 high, f184;
mov.b32 r1587, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1588, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r1589, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1590, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1593, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1594, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1595, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1596, {low, high};
}
{
mul.f16x2 r1609, r1563, r1587;
}
{
mul.f16x2 r1612, r1566, r1588;
}
{
sub.f16x2 r1615, r1609, r1612;
}
{
mul.f16x2 r1618, r1563, r1588;
}
{
fma.rn.f16x2 r1621, r1566, r1587, r1618;
}
{
mul.f16x2 r1625, r1575, r1589;
}
{
mul.f16x2 r1628, r1578, r1590;
}
{
sub.f16x2 r1631, r1625, r1628;
}
{
mul.f16x2 r1634, r1575, r1590;
}
{
fma.rn.f16x2 r1637, r1578, r1589, r1634;
}
{
neg.f16x2 r1641, r1557;
}
{
mul.f16x2 r1643, r1569, r1593;
}
{
mul.f16x2 r1646, r1572, r1594;
}
{
sub.f16x2 r1649, r1643, r1646;
}
{
mul.f16x2 r1652, r1569, r1594;
}
{
fma.rn.f16x2 r1655, r1572, r1593, r1652;
}
{
mul.f16x2 r1659, r1581, r1595;
}
{
mul.f16x2 r1662, r1584, r1596;
}
{
sub.f16x2 r1665, r1659, r1662;
}
{
mul.f16x2 r1668, r1581, r1596;
}
{
fma.rn.f16x2 r1671, r1584, r1595, r1668;
}
{
add.f16x2 %0, r1297, r1551;
}
{
add.f16x2 %1, r1300, r1554;
}
{
sub.f16x2 %12, r1297, r1551;
}
{
sub.f16x2 %13, r1300, r1554;
}
{
add.f16x2 %2, r1309, r1615;
}
{
add.f16x2 %3, r1312, r1621;
}
{
sub.f16x2 %14, r1309, r1615;
}
{
sub.f16x2 %15, r1312, r1621;
}
{
add.f16x2 %4, r1321, r1631;
}
{
add.f16x2 %5, r1324, r1637;
}
{
sub.f16x2 %16, r1321, r1631;
}
{
sub.f16x2 %17, r1324, r1637;
}
{
add.f16x2 %6, r1303, r1560;
}
{
add.f16x2 %7, r1306, r1641;
}
{
sub.f16x2 %18, r1303, r1560;
}
{
sub.f16x2 %19, r1306, r1641;
}
{
add.f16x2 %8, r1315, r1649;
}
{
add.f16x2 %9, r1318, r1655;
}
{
sub.f16x2 %20, r1315, r1649;
}
{
sub.f16x2 %21, r1318, r1655;
}
{
add.f16x2 %10, r1327, r1665;
}
{
add.f16x2 %11, r1330, r1671;
}
{
sub.f16x2 %22, r1327, r1665;
}
{
sub.f16x2 %23, r1330, r1671;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)));
};


#endif
