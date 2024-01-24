#ifndef CUFFTDX_FFT_1728_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_1728_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<946, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<359>;
.reg .b32 r<2844>;
.reg .b64 rd<7>;
mov.u32 r2825, %tid.y;
shl.b32 r2826, r2825, 1;
mov.u32 r2827, %24;
mad.lo.s32 r2828, r2826, 6912, r2827;
mov.u32 r2829, %tid.x;
mov.f32 f328, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1, {low, high};
}
mov.f32 f326, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
mov.f32 f314, 0f3F000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f314;
cvt.rn.f16.f32 high, f314;
mov.b32 r177, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r178, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r179, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r180, {low, high};
}
mov.f32 f249, 0fBF800000;
mov.f32 f310, 0f3F5DB3D7;
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
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r343, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
cvt.rn.f16.f32 low, f314;
cvt.rn.f16.f32 high, f314;
mov.b32 r431, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r432, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r433, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
cvt.rn.f16.f32 low, f310;
cvt.rn.f16.f32 high, f310;
mov.b32 r509, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r510, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f314;
cvt.rn.f16.f32 high, f314;
mov.b32 r511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r512, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r515, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r516, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r517, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r518, {low, high};
}
mov.f32 f250, 0f3F800000;
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
mul.wide.u32 rd2, r2829, 954437177;
shr.u64 rd3, rd2, 37;
cvt.u32.u64 r2830, rd3;
mul.lo.s32 r2831, r2830, 144;
sub.s32 r2832, r2829, r2831;
shr.u64 rd4, rd2, 36;
cvt.u32.u64 r2833, rd4;
and.b32 r2834, r2833, 268435454;
mad.lo.s32 r2835, r2834, 6912, r2828;
cvt.rn.f32.u32 f353, r2832;
mul.f32 f354, f353, 0f3B6E4BAE;
cos.approx.f32 f101, f354;
sin.approx.f32 f355, f354;
neg.f32 f102, f355;
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
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
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
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
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
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
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
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
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
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
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
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
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
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
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
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
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
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
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
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
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
mad.lo.s32 r2836, r2832, 96, r2835;
st.shared.v4.f32 [r2836], {r597, r600, r681, r688};
st.shared.v4.f32 [r2836+16], {r718, r725, r755, r762};
st.shared.v4.f32 [r2836+32], {r792, r799, r829, r836};
st.shared.v4.f32 [r2836+48], {r866, r873, r903, r910};
st.shared.v4.f32 [r2836+64], {r940, r947, r977, r984};
st.shared.v4.f32 [r2836+80], {r1014, r1021, r1051, r1058};
barrier.sync 0;
mad.lo.s32 r2837, r2832, -88, r2836;
ld.shared.u32 r1087, [r2837];
ld.shared.u32 r1093, [r2837+4];
ld.shared.u32 r1341, [r2837+1152];
ld.shared.u32 r1347, [r2837+1156];
ld.shared.u32 r1175, [r2837+2304];
ld.shared.u32 r1181, [r2837+2308];
ld.shared.u32 r1429, [r2837+3456];
ld.shared.u32 r1435, [r2837+3460];
ld.shared.u32 r1084, [r2837+4608];
ld.shared.u32 r1090, [r2837+4612];
ld.shared.u32 r1338, [r2837+5760];
ld.shared.u32 r1344, [r2837+5764];
ld.shared.u32 r1172, [r2837+6912];
ld.shared.u32 r1178, [r2837+6916];
ld.shared.u32 r1426, [r2837+8064];
ld.shared.u32 r1432, [r2837+8068];
ld.shared.u32 r1085, [r2837+9216];
ld.shared.u32 r1091, [r2837+9220];
ld.shared.u32 r1339, [r2837+10368];
ld.shared.u32 r1345, [r2837+10372];
ld.shared.u32 r1173, [r2837+11520];
ld.shared.u32 r1179, [r2837+11524];
ld.shared.u32 r1427, [r2837+12672];
ld.shared.u32 r1433, [r2837+12676];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1079, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1167, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
cvt.rn.f16.f32 low, f314;
cvt.rn.f16.f32 high, f314;
mov.b32 r1255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r1256, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1257, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1333, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1421, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
cvt.rn.f16.f32 low, f314;
cvt.rn.f16.f32 high, f314;
mov.b32 r1509, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r1510, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
cvt.rn.f16.f32 low, f310;
cvt.rn.f16.f32 high, f310;
mov.b32 r1587, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1588, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f314;
cvt.rn.f16.f32 high, f314;
mov.b32 r1589, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r1590, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1593, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r1594, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r1595, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
add.f16x2 r1675, r1297, r1551;
}
{
add.f16x2 r1678, r1300, r1554;
}
{
sub.f16x2 r1681, r1297, r1551;
}
{
sub.f16x2 r1684, r1300, r1554;
}
{
add.f16x2 r1687, r1309, r1615;
}
{
add.f16x2 r1690, r1312, r1621;
}
{
sub.f16x2 r1693, r1309, r1615;
}
{
sub.f16x2 r1696, r1312, r1621;
}
{
add.f16x2 r1699, r1321, r1631;
}
{
add.f16x2 r1702, r1324, r1637;
}
{
sub.f16x2 r1705, r1321, r1631;
}
{
sub.f16x2 r1708, r1324, r1637;
}
{
add.f16x2 r1711, r1303, r1560;
}
{
add.f16x2 r1714, r1306, r1641;
}
{
sub.f16x2 r1717, r1303, r1560;
}
{
sub.f16x2 r1720, r1306, r1641;
}
{
add.f16x2 r1723, r1315, r1649;
}
{
add.f16x2 r1726, r1318, r1655;
}
{
sub.f16x2 r1729, r1315, r1649;
}
{
sub.f16x2 r1732, r1318, r1655;
}
{
add.f16x2 r1735, r1327, r1665;
}
{
add.f16x2 r1738, r1330, r1671;
}
{
sub.f16x2 r1741, r1327, r1665;
}
{
sub.f16x2 r1744, r1330, r1671;
}
mul.wide.u32 rd5, r2832, -1431655765;
shr.u64 rd6, rd5, 35;
cvt.u32.u64 r2838, rd6;
cvt.rn.f32.u32 f356, r2838;
mul.f32 f357, f356, 0f3D32B8C2;
cos.approx.f32 f227, f357;
sin.approx.f32 f358, f357;
neg.f32 f228, f358;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f227;
cvt.rn.f16.f32 high, f228;
mov.b32 r1747, {low, high};
}
mul.lo.s32 r2839, r2838, 12;
sub.s32 r2840, r2832, r2839;
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1750, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1752, {high, high};
}
{
mul.f16x2 r1754, r1690, r1752;
}
{
neg.f16x2 r1757, r1754;
}
{
fma.rn.f16x2 r1759, r1687, r1750, r1757;
}
{
mul.f16x2 r1763, r1687, r1752;
}
{
fma.rn.f16x2 r1766, r1690, r1750, r1763;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1770, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1772, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
mov.b32 r1774, {low, high};
}
{
mul.f16x2 r1775, r1772, r1774;
}
{
mul.f16x2 r1778, r1747, r1770;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1781, {high, low};
}
{
fma.rn.f16x2 r1783, r1775, r1781, r1778;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1783;
mov.b32 r1787, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1783;
mov.b32 r1789, {high, high};
}
{
mul.f16x2 r1791, r1702, r1789;
}
{
neg.f16x2 r1794, r1791;
}
{
fma.rn.f16x2 r1796, r1699, r1787, r1794;
}
{
mul.f16x2 r1800, r1699, r1789;
}
{
fma.rn.f16x2 r1803, r1702, r1787, r1800;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1807, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1809, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
mov.b32 r1811, {low, high};
}
{
mul.f16x2 r1812, r1809, r1811;
}
{
mul.f16x2 r1815, r1783, r1807;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1783;
mov.b32 r1818, {high, low};
}
{
fma.rn.f16x2 r1820, r1812, r1818, r1815;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1820;
mov.b32 r1824, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1820;
mov.b32 r1826, {high, high};
}
{
mul.f16x2 r1828, r1714, r1826;
}
{
neg.f16x2 r1831, r1828;
}
{
fma.rn.f16x2 r1833, r1711, r1824, r1831;
}
{
mul.f16x2 r1837, r1711, r1826;
}
{
fma.rn.f16x2 r1840, r1714, r1824, r1837;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1844, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1846, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
mov.b32 r1848, {low, high};
}
{
mul.f16x2 r1849, r1846, r1848;
}
{
mul.f16x2 r1852, r1820, r1844;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1820;
mov.b32 r1855, {high, low};
}
{
fma.rn.f16x2 r1857, r1849, r1855, r1852;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1857;
mov.b32 r1861, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1857;
mov.b32 r1863, {high, high};
}
{
mul.f16x2 r1865, r1726, r1863;
}
{
neg.f16x2 r1868, r1865;
}
{
fma.rn.f16x2 r1870, r1723, r1861, r1868;
}
{
mul.f16x2 r1874, r1723, r1863;
}
{
fma.rn.f16x2 r1877, r1726, r1861, r1874;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1881, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1883, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
mov.b32 r1885, {low, high};
}
{
mul.f16x2 r1886, r1883, r1885;
}
{
mul.f16x2 r1889, r1857, r1881;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1857;
mov.b32 r1892, {high, low};
}
{
fma.rn.f16x2 r1894, r1886, r1892, r1889;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1894;
mov.b32 r1898, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1894;
mov.b32 r1900, {high, high};
}
{
mul.f16x2 r1902, r1738, r1900;
}
{
neg.f16x2 r1905, r1902;
}
{
fma.rn.f16x2 r1907, r1735, r1898, r1905;
}
{
mul.f16x2 r1911, r1735, r1900;
}
{
fma.rn.f16x2 r1914, r1738, r1898, r1911;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1918, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1920, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
mov.b32 r1922, {low, high};
}
{
mul.f16x2 r1923, r1920, r1922;
}
{
mul.f16x2 r1926, r1894, r1918;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1894;
mov.b32 r1929, {high, low};
}
{
fma.rn.f16x2 r1931, r1923, r1929, r1926;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1931;
mov.b32 r1935, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1931;
mov.b32 r1937, {high, high};
}
{
mul.f16x2 r1939, r1684, r1937;
}
{
neg.f16x2 r1942, r1939;
}
{
fma.rn.f16x2 r1944, r1681, r1935, r1942;
}
{
mul.f16x2 r1948, r1681, r1937;
}
{
fma.rn.f16x2 r1951, r1684, r1935, r1948;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1955, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1957, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
mov.b32 r1959, {low, high};
}
{
mul.f16x2 r1960, r1957, r1959;
}
{
mul.f16x2 r1963, r1931, r1955;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1931;
mov.b32 r1966, {high, low};
}
{
fma.rn.f16x2 r1968, r1960, r1966, r1963;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1968;
mov.b32 r1972, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1968;
mov.b32 r1974, {high, high};
}
{
mul.f16x2 r1976, r1696, r1974;
}
{
neg.f16x2 r1979, r1976;
}
{
fma.rn.f16x2 r1981, r1693, r1972, r1979;
}
{
mul.f16x2 r1985, r1693, r1974;
}
{
fma.rn.f16x2 r1988, r1696, r1972, r1985;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1992, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1994, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
mov.b32 r1996, {low, high};
}
{
mul.f16x2 r1997, r1994, r1996;
}
{
mul.f16x2 r2000, r1968, r1992;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1968;
mov.b32 r2003, {high, low};
}
{
fma.rn.f16x2 r2005, r1997, r2003, r2000;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2005;
mov.b32 r2009, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2005;
mov.b32 r2011, {high, high};
}
{
mul.f16x2 r2013, r1708, r2011;
}
{
neg.f16x2 r2016, r2013;
}
{
fma.rn.f16x2 r2018, r1705, r2009, r2016;
}
{
mul.f16x2 r2022, r1705, r2011;
}
{
fma.rn.f16x2 r2025, r1708, r2009, r2022;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r2029, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r2031, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
mov.b32 r2033, {low, high};
}
{
mul.f16x2 r2034, r2031, r2033;
}
{
mul.f16x2 r2037, r2005, r2029;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2005;
mov.b32 r2040, {high, low};
}
{
fma.rn.f16x2 r2042, r2034, r2040, r2037;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2042;
mov.b32 r2046, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2042;
mov.b32 r2048, {high, high};
}
{
mul.f16x2 r2050, r1720, r2048;
}
{
neg.f16x2 r2053, r2050;
}
{
fma.rn.f16x2 r2055, r1717, r2046, r2053;
}
{
mul.f16x2 r2059, r1717, r2048;
}
{
fma.rn.f16x2 r2062, r1720, r2046, r2059;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r2066, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r2068, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
mov.b32 r2070, {low, high};
}
{
mul.f16x2 r2071, r2068, r2070;
}
{
mul.f16x2 r2074, r2042, r2066;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2042;
mov.b32 r2077, {high, low};
}
{
fma.rn.f16x2 r2079, r2071, r2077, r2074;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2079;
mov.b32 r2083, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2079;
mov.b32 r2085, {high, high};
}
{
mul.f16x2 r2087, r1732, r2085;
}
{
neg.f16x2 r2090, r2087;
}
{
fma.rn.f16x2 r2092, r1729, r2083, r2090;
}
{
mul.f16x2 r2096, r1729, r2085;
}
{
fma.rn.f16x2 r2099, r1732, r2083, r2096;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r2103, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r2105, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
mov.b32 r2107, {low, high};
}
{
mul.f16x2 r2108, r2105, r2107;
}
{
mul.f16x2 r2111, r2079, r2103;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2079;
mov.b32 r2114, {high, low};
}
{
fma.rn.f16x2 r2116, r2108, r2114, r2111;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2116;
mov.b32 r2120, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2116;
mov.b32 r2122, {high, high};
}
{
mul.f16x2 r2124, r1744, r2122;
}
{
neg.f16x2 r2127, r2124;
}
{
fma.rn.f16x2 r2129, r1741, r2120, r2127;
}
{
mul.f16x2 r2133, r1741, r2122;
}
{
fma.rn.f16x2 r2136, r1744, r2120, r2133;
}
shl.b32 r2841, r2840, 3;
add.s32 r2842, r2835, r2841;
barrier.sync 0;
mad.lo.s32 r2843, r2838, 1152, r2842;
st.shared.u32 [r2843], r1675;
st.shared.u32 [r2843+4], r1678;
st.shared.u32 [r2843+96], r1759;
st.shared.u32 [r2843+100], r1766;
st.shared.u32 [r2843+192], r1796;
st.shared.u32 [r2843+196], r1803;
st.shared.u32 [r2843+288], r1833;
st.shared.u32 [r2843+292], r1840;
st.shared.u32 [r2843+384], r1870;
st.shared.u32 [r2843+388], r1877;
st.shared.u32 [r2843+480], r1907;
st.shared.u32 [r2843+484], r1914;
st.shared.u32 [r2843+576], r1944;
st.shared.u32 [r2843+580], r1951;
st.shared.u32 [r2843+672], r1981;
st.shared.u32 [r2843+676], r1988;
st.shared.u32 [r2843+768], r2018;
st.shared.u32 [r2843+772], r2025;
st.shared.u32 [r2843+864], r2055;
st.shared.u32 [r2843+868], r2062;
st.shared.u32 [r2843+960], r2092;
st.shared.u32 [r2843+964], r2099;
st.shared.u32 [r2843+1056], r2129;
st.shared.u32 [r2843+1060], r2136;
barrier.sync 0;
ld.shared.u32 r2165, [r2837];
ld.shared.u32 r2171, [r2837+4];
ld.shared.u32 r2419, [r2837+1152];
ld.shared.u32 r2425, [r2837+1156];
ld.shared.u32 r2253, [r2837+2304];
ld.shared.u32 r2259, [r2837+2308];
ld.shared.u32 r2507, [r2837+3456];
ld.shared.u32 r2513, [r2837+3460];
ld.shared.u32 r2162, [r2837+4608];
ld.shared.u32 r2168, [r2837+4612];
ld.shared.u32 r2416, [r2837+5760];
ld.shared.u32 r2422, [r2837+5764];
ld.shared.u32 r2250, [r2837+6912];
ld.shared.u32 r2256, [r2837+6916];
ld.shared.u32 r2504, [r2837+8064];
ld.shared.u32 r2510, [r2837+8068];
ld.shared.u32 r2163, [r2837+9216];
ld.shared.u32 r2169, [r2837+9220];
ld.shared.u32 r2417, [r2837+10368];
ld.shared.u32 r2423, [r2837+10372];
ld.shared.u32 r2251, [r2837+11520];
ld.shared.u32 r2257, [r2837+11524];
ld.shared.u32 r2505, [r2837+12672];
ld.shared.u32 r2511, [r2837+12676];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2157, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2158, {low, high};
}
{
neg.f16x2 r2159, r2158;
}
{
add.f16x2 r2161, r2162, r2163;
}
{
add.f16x2 r2164, r2165, r2161;
}
{
add.f16x2 r2167, r2168, r2169;
}
{
add.f16x2 r2170, r2171, r2167;
}
{
add.f16x2 r2173, r2162, r2163;
}
{
mul.f16x2 r2176, r2173, r2157;
}
{
add.f16x2 r2179, r2165, r2176;
}
{
sub.f16x2 r2182, r2168, r2169;
}
{
mul.f16x2 r2185, r2182, r2159;
}
{
add.f16x2 r2188, r2179, r2185;
}
{
add.f16x2 r2191, r2162, r2163;
}
{
mul.f16x2 r2194, r2191, r2157;
}
{
add.f16x2 r2197, r2165, r2194;
}
{
sub.f16x2 r2200, r2168, r2169;
}
{
mul.f16x2 r2203, r2200, r2159;
}
{
sub.f16x2 r2206, r2197, r2203;
}
{
add.f16x2 r2209, r2168, r2169;
}
{
mul.f16x2 r2212, r2209, r2157;
}
{
add.f16x2 r2215, r2171, r2212;
}
{
sub.f16x2 r2218, r2162, r2163;
}
{
mul.f16x2 r2221, r2218, r2159;
}
{
sub.f16x2 r2224, r2215, r2221;
}
{
add.f16x2 r2227, r2168, r2169;
}
{
mul.f16x2 r2230, r2227, r2157;
}
{
add.f16x2 r2233, r2171, r2230;
}
{
sub.f16x2 r2236, r2162, r2163;
}
{
mul.f16x2 r2239, r2236, r2159;
}
{
add.f16x2 r2242, r2233, r2239;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2245, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2246, {low, high};
}
{
neg.f16x2 r2247, r2246;
}
{
add.f16x2 r2249, r2250, r2251;
}
{
add.f16x2 r2252, r2253, r2249;
}
{
add.f16x2 r2255, r2256, r2257;
}
{
add.f16x2 r2258, r2259, r2255;
}
{
add.f16x2 r2261, r2250, r2251;
}
{
mul.f16x2 r2264, r2261, r2245;
}
{
add.f16x2 r2267, r2253, r2264;
}
{
sub.f16x2 r2270, r2256, r2257;
}
{
mul.f16x2 r2273, r2270, r2247;
}
{
add.f16x2 r2276, r2267, r2273;
}
{
add.f16x2 r2279, r2250, r2251;
}
{
mul.f16x2 r2282, r2279, r2245;
}
{
add.f16x2 r2285, r2253, r2282;
}
{
sub.f16x2 r2288, r2256, r2257;
}
{
mul.f16x2 r2291, r2288, r2247;
}
{
sub.f16x2 r2294, r2285, r2291;
}
{
add.f16x2 r2297, r2256, r2257;
}
{
mul.f16x2 r2300, r2297, r2245;
}
{
add.f16x2 r2303, r2259, r2300;
}
{
sub.f16x2 r2306, r2250, r2251;
}
{
mul.f16x2 r2309, r2306, r2247;
}
{
sub.f16x2 r2312, r2303, r2309;
}
{
add.f16x2 r2315, r2256, r2257;
}
{
mul.f16x2 r2318, r2315, r2245;
}
{
add.f16x2 r2321, r2259, r2318;
}
{
sub.f16x2 r2324, r2250, r2251;
}
{
mul.f16x2 r2327, r2324, r2247;
}
{
add.f16x2 r2330, r2321, r2327;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f314;
cvt.rn.f16.f32 high, f314;
mov.b32 r2333, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2334, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2335, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2336, {low, high};
}
{
mul.f16x2 r2343, r2276, r2333;
}
{
mul.f16x2 r2346, r2312, r2334;
}
{
sub.f16x2 r2349, r2343, r2346;
}
{
mul.f16x2 r2352, r2276, r2334;
}
{
fma.rn.f16x2 r2355, r2312, r2333, r2352;
}
{
mul.f16x2 r2359, r2294, r2335;
}
{
mul.f16x2 r2362, r2330, r2336;
}
{
sub.f16x2 r2365, r2359, r2362;
}
{
mul.f16x2 r2368, r2294, r2336;
}
{
fma.rn.f16x2 r2371, r2330, r2335, r2368;
}
{
add.f16x2 r2375, r2164, r2252;
}
{
add.f16x2 r2378, r2170, r2258;
}
{
sub.f16x2 r2381, r2164, r2252;
}
{
sub.f16x2 r2384, r2170, r2258;
}
{
add.f16x2 r2387, r2188, r2349;
}
{
add.f16x2 r2390, r2224, r2355;
}
{
sub.f16x2 r2393, r2188, r2349;
}
{
sub.f16x2 r2396, r2224, r2355;
}
{
add.f16x2 r2399, r2206, r2365;
}
{
add.f16x2 r2402, r2242, r2371;
}
{
sub.f16x2 r2405, r2206, r2365;
}
{
sub.f16x2 r2408, r2242, r2371;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2411, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2412, {low, high};
}
{
neg.f16x2 r2413, r2412;
}
{
add.f16x2 r2415, r2416, r2417;
}
{
add.f16x2 r2418, r2419, r2415;
}
{
add.f16x2 r2421, r2422, r2423;
}
{
add.f16x2 r2424, r2425, r2421;
}
{
add.f16x2 r2427, r2416, r2417;
}
{
mul.f16x2 r2430, r2427, r2411;
}
{
add.f16x2 r2433, r2419, r2430;
}
{
sub.f16x2 r2436, r2422, r2423;
}
{
mul.f16x2 r2439, r2436, r2413;
}
{
add.f16x2 r2442, r2433, r2439;
}
{
add.f16x2 r2445, r2416, r2417;
}
{
mul.f16x2 r2448, r2445, r2411;
}
{
add.f16x2 r2451, r2419, r2448;
}
{
sub.f16x2 r2454, r2422, r2423;
}
{
mul.f16x2 r2457, r2454, r2413;
}
{
sub.f16x2 r2460, r2451, r2457;
}
{
add.f16x2 r2463, r2422, r2423;
}
{
mul.f16x2 r2466, r2463, r2411;
}
{
add.f16x2 r2469, r2425, r2466;
}
{
sub.f16x2 r2472, r2416, r2417;
}
{
mul.f16x2 r2475, r2472, r2413;
}
{
sub.f16x2 r2478, r2469, r2475;
}
{
add.f16x2 r2481, r2422, r2423;
}
{
mul.f16x2 r2484, r2481, r2411;
}
{
add.f16x2 r2487, r2425, r2484;
}
{
sub.f16x2 r2490, r2416, r2417;
}
{
mul.f16x2 r2493, r2490, r2413;
}
{
add.f16x2 r2496, r2487, r2493;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2499, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2500, {low, high};
}
{
neg.f16x2 r2501, r2500;
}
{
add.f16x2 r2503, r2504, r2505;
}
{
add.f16x2 r2506, r2507, r2503;
}
{
add.f16x2 r2509, r2510, r2511;
}
{
add.f16x2 r2512, r2513, r2509;
}
{
add.f16x2 r2515, r2504, r2505;
}
{
mul.f16x2 r2518, r2515, r2499;
}
{
add.f16x2 r2521, r2507, r2518;
}
{
sub.f16x2 r2524, r2510, r2511;
}
{
mul.f16x2 r2527, r2524, r2501;
}
{
add.f16x2 r2530, r2521, r2527;
}
{
add.f16x2 r2533, r2504, r2505;
}
{
mul.f16x2 r2536, r2533, r2499;
}
{
add.f16x2 r2539, r2507, r2536;
}
{
sub.f16x2 r2542, r2510, r2511;
}
{
mul.f16x2 r2545, r2542, r2501;
}
{
sub.f16x2 r2548, r2539, r2545;
}
{
add.f16x2 r2551, r2510, r2511;
}
{
mul.f16x2 r2554, r2551, r2499;
}
{
add.f16x2 r2557, r2513, r2554;
}
{
sub.f16x2 r2560, r2504, r2505;
}
{
mul.f16x2 r2563, r2560, r2501;
}
{
sub.f16x2 r2566, r2557, r2563;
}
{
add.f16x2 r2569, r2510, r2511;
}
{
mul.f16x2 r2572, r2569, r2499;
}
{
add.f16x2 r2575, r2513, r2572;
}
{
sub.f16x2 r2578, r2504, r2505;
}
{
mul.f16x2 r2581, r2578, r2501;
}
{
add.f16x2 r2584, r2575, r2581;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f314;
cvt.rn.f16.f32 high, f314;
mov.b32 r2587, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2588, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2589, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2590, {low, high};
}
{
mul.f16x2 r2597, r2530, r2587;
}
{
mul.f16x2 r2600, r2566, r2588;
}
{
sub.f16x2 r2603, r2597, r2600;
}
{
mul.f16x2 r2606, r2530, r2588;
}
{
fma.rn.f16x2 r2609, r2566, r2587, r2606;
}
{
mul.f16x2 r2613, r2548, r2589;
}
{
mul.f16x2 r2616, r2584, r2590;
}
{
sub.f16x2 r2619, r2613, r2616;
}
{
mul.f16x2 r2622, r2548, r2590;
}
{
fma.rn.f16x2 r2625, r2584, r2589, r2622;
}
{
add.f16x2 r2629, r2418, r2506;
}
{
add.f16x2 r2632, r2424, r2512;
}
{
sub.f16x2 r2635, r2418, r2506;
}
{
sub.f16x2 r2638, r2424, r2512;
}
{
add.f16x2 r2641, r2442, r2603;
}
{
add.f16x2 r2644, r2478, r2609;
}
{
sub.f16x2 r2647, r2442, r2603;
}
{
sub.f16x2 r2650, r2478, r2609;
}
{
add.f16x2 r2653, r2460, r2619;
}
{
add.f16x2 r2656, r2496, r2625;
}
{
sub.f16x2 r2659, r2460, r2619;
}
{
sub.f16x2 r2662, r2496, r2625;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f310;
cvt.rn.f16.f32 high, f310;
mov.b32 r2665, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2666, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f314;
cvt.rn.f16.f32 high, f314;
mov.b32 r2667, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2668, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2671, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2672, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2673, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2674, {low, high};
}
{
mul.f16x2 r2687, r2641, r2665;
}
{
mul.f16x2 r2690, r2644, r2666;
}
{
sub.f16x2 r2693, r2687, r2690;
}
{
mul.f16x2 r2696, r2641, r2666;
}
{
fma.rn.f16x2 r2699, r2644, r2665, r2696;
}
{
mul.f16x2 r2703, r2653, r2667;
}
{
mul.f16x2 r2706, r2656, r2668;
}
{
sub.f16x2 r2709, r2703, r2706;
}
{
mul.f16x2 r2712, r2653, r2668;
}
{
fma.rn.f16x2 r2715, r2656, r2667, r2712;
}
{
neg.f16x2 r2719, r2635;
}
{
mul.f16x2 r2721, r2647, r2671;
}
{
mul.f16x2 r2724, r2650, r2672;
}
{
sub.f16x2 r2727, r2721, r2724;
}
{
mul.f16x2 r2730, r2647, r2672;
}
{
fma.rn.f16x2 r2733, r2650, r2671, r2730;
}
{
mul.f16x2 r2737, r2659, r2673;
}
{
mul.f16x2 r2740, r2662, r2674;
}
{
sub.f16x2 r2743, r2737, r2740;
}
{
mul.f16x2 r2746, r2659, r2674;
}
{
fma.rn.f16x2 r2749, r2662, r2673, r2746;
}
{
add.f16x2 %0, r2375, r2629;
}
{
add.f16x2 %1, r2378, r2632;
}
{
sub.f16x2 %12, r2375, r2629;
}
{
sub.f16x2 %13, r2378, r2632;
}
{
add.f16x2 %2, r2387, r2693;
}
{
add.f16x2 %3, r2390, r2699;
}
{
sub.f16x2 %14, r2387, r2693;
}
{
sub.f16x2 %15, r2390, r2699;
}
{
add.f16x2 %4, r2399, r2709;
}
{
add.f16x2 %5, r2402, r2715;
}
{
sub.f16x2 %16, r2399, r2709;
}
{
sub.f16x2 %17, r2402, r2715;
}
{
add.f16x2 %6, r2381, r2638;
}
{
add.f16x2 %7, r2384, r2719;
}
{
sub.f16x2 %18, r2381, r2638;
}
{
sub.f16x2 %19, r2384, r2719;
}
{
add.f16x2 %8, r2393, r2727;
}
{
add.f16x2 %9, r2396, r2733;
}
{
sub.f16x2 %20, r2393, r2727;
}
{
sub.f16x2 %21, r2396, r2733;
}
{
add.f16x2 %10, r2405, r2743;
}
{
add.f16x2 %11, r2408, r2749;
}
{
sub.f16x2 %22, r2405, r2743;
}
{
sub.f16x2 %23, r2408, r2749;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<947, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<359>;
.reg .b32 r<2841>;
.reg .b64 rd<6>;
mov.u32 r2825, %tid.y;
mov.u32 r2826, %24;
mad.lo.s32 r2827, r2825, 6912, r2826;
mov.u32 r2828, %tid.x;
mov.f32 f328, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1, {low, high};
}
mov.f32 f326, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
mov.f32 f314, 0f3F000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f314;
cvt.rn.f16.f32 high, f314;
mov.b32 r177, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r178, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r179, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r180, {low, high};
}
mov.f32 f249, 0fBF800000;
mov.f32 f310, 0f3F5DB3D7;
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
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r343, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
cvt.rn.f16.f32 low, f314;
cvt.rn.f16.f32 high, f314;
mov.b32 r431, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r432, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r433, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
cvt.rn.f16.f32 low, f310;
cvt.rn.f16.f32 high, f310;
mov.b32 r509, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r510, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f314;
cvt.rn.f16.f32 high, f314;
mov.b32 r511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r512, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r515, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r516, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r517, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r518, {low, high};
}
mov.f32 f250, 0f3F800000;
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
mul.wide.u32 rd2, r2828, 954437177;
shr.u64 rd3, rd2, 37;
cvt.u32.u64 r2829, rd3;
mul.lo.s32 r2830, r2829, 144;
sub.s32 r2831, r2828, r2830;
mad.lo.s32 r2832, r2829, 6912, r2827;
cvt.rn.f32.u32 f353, r2831;
mul.f32 f354, f353, 0f3B6E4BAE;
cos.approx.f32 f101, f354;
sin.approx.f32 f355, f354;
neg.f32 f102, f355;
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
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
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
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
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
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
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
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
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
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
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
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
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
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
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
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
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
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
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
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
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
mad.lo.s32 r2833, r2831, 48, r2832;
st.shared.v4.f32 [r2833], {r597, r681, r718, r755};
st.shared.v4.f32 [r2833+16], {r792, r829, r866, r903};
st.shared.v4.f32 [r2833+32], {r940, r977, r1014, r1051};
barrier.sync 0;
mad.lo.s32 r2834, r2831, -44, r2833;
ld.shared.u32 r1087, [r2834];
ld.shared.u32 r1341, [r2834+576];
ld.shared.u32 r1175, [r2834+1152];
ld.shared.u32 r1429, [r2834+1728];
ld.shared.u32 r1084, [r2834+2304];
ld.shared.u32 r1338, [r2834+2880];
ld.shared.u32 r1172, [r2834+3456];
ld.shared.u32 r1426, [r2834+4032];
ld.shared.u32 r1085, [r2834+4608];
ld.shared.u32 r1339, [r2834+5184];
ld.shared.u32 r1173, [r2834+5760];
ld.shared.u32 r1427, [r2834+6336];
barrier.sync 0;
st.shared.v4.f32 [r2833], {r600, r688, r725, r762};
st.shared.v4.f32 [r2833+16], {r799, r836, r873, r910};
st.shared.v4.f32 [r2833+32], {r947, r984, r1021, r1058};
barrier.sync 0;
ld.shared.u32 r1093, [r2834];
ld.shared.u32 r1347, [r2834+576];
ld.shared.u32 r1181, [r2834+1152];
ld.shared.u32 r1435, [r2834+1728];
ld.shared.u32 r1090, [r2834+2304];
ld.shared.u32 r1344, [r2834+2880];
ld.shared.u32 r1178, [r2834+3456];
ld.shared.u32 r1432, [r2834+4032];
ld.shared.u32 r1091, [r2834+4608];
ld.shared.u32 r1345, [r2834+5184];
ld.shared.u32 r1179, [r2834+5760];
ld.shared.u32 r1433, [r2834+6336];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1079, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1167, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
cvt.rn.f16.f32 low, f314;
cvt.rn.f16.f32 high, f314;
mov.b32 r1255, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r1256, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1257, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1333, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1421, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
cvt.rn.f16.f32 low, f314;
cvt.rn.f16.f32 high, f314;
mov.b32 r1509, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r1510, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
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
cvt.rn.f16.f32 low, f310;
cvt.rn.f16.f32 high, f310;
mov.b32 r1587, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1588, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f314;
cvt.rn.f16.f32 high, f314;
mov.b32 r1589, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r1590, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r1593, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r1594, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r1595, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
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
add.f16x2 r1675, r1297, r1551;
}
{
add.f16x2 r1678, r1300, r1554;
}
{
sub.f16x2 r1681, r1297, r1551;
}
{
sub.f16x2 r1684, r1300, r1554;
}
{
add.f16x2 r1687, r1309, r1615;
}
{
add.f16x2 r1690, r1312, r1621;
}
{
sub.f16x2 r1693, r1309, r1615;
}
{
sub.f16x2 r1696, r1312, r1621;
}
{
add.f16x2 r1699, r1321, r1631;
}
{
add.f16x2 r1702, r1324, r1637;
}
{
sub.f16x2 r1705, r1321, r1631;
}
{
sub.f16x2 r1708, r1324, r1637;
}
{
add.f16x2 r1711, r1303, r1560;
}
{
add.f16x2 r1714, r1306, r1641;
}
{
sub.f16x2 r1717, r1303, r1560;
}
{
sub.f16x2 r1720, r1306, r1641;
}
{
add.f16x2 r1723, r1315, r1649;
}
{
add.f16x2 r1726, r1318, r1655;
}
{
sub.f16x2 r1729, r1315, r1649;
}
{
sub.f16x2 r1732, r1318, r1655;
}
{
add.f16x2 r1735, r1327, r1665;
}
{
add.f16x2 r1738, r1330, r1671;
}
{
sub.f16x2 r1741, r1327, r1665;
}
{
sub.f16x2 r1744, r1330, r1671;
}
mul.wide.u32 rd4, r2831, -1431655765;
shr.u64 rd5, rd4, 35;
cvt.u32.u64 r2835, rd5;
mul.lo.s32 r2836, r2835, 12;
sub.s32 r2837, r2831, r2836;
shl.b32 r2838, r2837, 2;
add.s32 r2839, r2832, r2838;
cvt.rn.f32.u32 f356, r2835;
mul.f32 f357, f356, 0f3D32B8C2;
cos.approx.f32 f227, f357;
sin.approx.f32 f358, f357;
neg.f32 f228, f358;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f227;
cvt.rn.f16.f32 high, f228;
mov.b32 r1747, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1750, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1752, {high, high};
}
{
mul.f16x2 r1754, r1690, r1752;
}
{
neg.f16x2 r1757, r1754;
}
{
fma.rn.f16x2 r1759, r1687, r1750, r1757;
}
{
mul.f16x2 r1763, r1687, r1752;
}
{
fma.rn.f16x2 r1766, r1690, r1750, r1763;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1770, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1772, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
mov.b32 r1774, {low, high};
}
{
mul.f16x2 r1775, r1772, r1774;
}
{
mul.f16x2 r1778, r1747, r1770;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1781, {high, low};
}
{
fma.rn.f16x2 r1783, r1775, r1781, r1778;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1783;
mov.b32 r1787, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1783;
mov.b32 r1789, {high, high};
}
{
mul.f16x2 r1791, r1702, r1789;
}
{
neg.f16x2 r1794, r1791;
}
{
fma.rn.f16x2 r1796, r1699, r1787, r1794;
}
{
mul.f16x2 r1800, r1699, r1789;
}
{
fma.rn.f16x2 r1803, r1702, r1787, r1800;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1807, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1809, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
mov.b32 r1811, {low, high};
}
{
mul.f16x2 r1812, r1809, r1811;
}
{
mul.f16x2 r1815, r1783, r1807;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1783;
mov.b32 r1818, {high, low};
}
{
fma.rn.f16x2 r1820, r1812, r1818, r1815;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1820;
mov.b32 r1824, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1820;
mov.b32 r1826, {high, high};
}
{
mul.f16x2 r1828, r1714, r1826;
}
{
neg.f16x2 r1831, r1828;
}
{
fma.rn.f16x2 r1833, r1711, r1824, r1831;
}
{
mul.f16x2 r1837, r1711, r1826;
}
{
fma.rn.f16x2 r1840, r1714, r1824, r1837;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1844, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1846, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
mov.b32 r1848, {low, high};
}
{
mul.f16x2 r1849, r1846, r1848;
}
{
mul.f16x2 r1852, r1820, r1844;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1820;
mov.b32 r1855, {high, low};
}
{
fma.rn.f16x2 r1857, r1849, r1855, r1852;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1857;
mov.b32 r1861, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1857;
mov.b32 r1863, {high, high};
}
{
mul.f16x2 r1865, r1726, r1863;
}
{
neg.f16x2 r1868, r1865;
}
{
fma.rn.f16x2 r1870, r1723, r1861, r1868;
}
{
mul.f16x2 r1874, r1723, r1863;
}
{
fma.rn.f16x2 r1877, r1726, r1861, r1874;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1881, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1883, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
mov.b32 r1885, {low, high};
}
{
mul.f16x2 r1886, r1883, r1885;
}
{
mul.f16x2 r1889, r1857, r1881;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1857;
mov.b32 r1892, {high, low};
}
{
fma.rn.f16x2 r1894, r1886, r1892, r1889;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1894;
mov.b32 r1898, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1894;
mov.b32 r1900, {high, high};
}
{
mul.f16x2 r1902, r1738, r1900;
}
{
neg.f16x2 r1905, r1902;
}
{
fma.rn.f16x2 r1907, r1735, r1898, r1905;
}
{
mul.f16x2 r1911, r1735, r1900;
}
{
fma.rn.f16x2 r1914, r1738, r1898, r1911;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1918, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1920, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
mov.b32 r1922, {low, high};
}
{
mul.f16x2 r1923, r1920, r1922;
}
{
mul.f16x2 r1926, r1894, r1918;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1894;
mov.b32 r1929, {high, low};
}
{
fma.rn.f16x2 r1931, r1923, r1929, r1926;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1931;
mov.b32 r1935, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1931;
mov.b32 r1937, {high, high};
}
{
mul.f16x2 r1939, r1684, r1937;
}
{
neg.f16x2 r1942, r1939;
}
{
fma.rn.f16x2 r1944, r1681, r1935, r1942;
}
{
mul.f16x2 r1948, r1681, r1937;
}
{
fma.rn.f16x2 r1951, r1684, r1935, r1948;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1955, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1957, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
mov.b32 r1959, {low, high};
}
{
mul.f16x2 r1960, r1957, r1959;
}
{
mul.f16x2 r1963, r1931, r1955;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1931;
mov.b32 r1966, {high, low};
}
{
fma.rn.f16x2 r1968, r1960, r1966, r1963;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1968;
mov.b32 r1972, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1968;
mov.b32 r1974, {high, high};
}
{
mul.f16x2 r1976, r1696, r1974;
}
{
neg.f16x2 r1979, r1976;
}
{
fma.rn.f16x2 r1981, r1693, r1972, r1979;
}
{
mul.f16x2 r1985, r1693, r1974;
}
{
fma.rn.f16x2 r1988, r1696, r1972, r1985;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1992, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r1994, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
mov.b32 r1996, {low, high};
}
{
mul.f16x2 r1997, r1994, r1996;
}
{
mul.f16x2 r2000, r1968, r1992;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1968;
mov.b32 r2003, {high, low};
}
{
fma.rn.f16x2 r2005, r1997, r2003, r2000;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2005;
mov.b32 r2009, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2005;
mov.b32 r2011, {high, high};
}
{
mul.f16x2 r2013, r1708, r2011;
}
{
neg.f16x2 r2016, r2013;
}
{
fma.rn.f16x2 r2018, r1705, r2009, r2016;
}
{
mul.f16x2 r2022, r1705, r2011;
}
{
fma.rn.f16x2 r2025, r1708, r2009, r2022;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r2029, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r2031, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
mov.b32 r2033, {low, high};
}
{
mul.f16x2 r2034, r2031, r2033;
}
{
mul.f16x2 r2037, r2005, r2029;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2005;
mov.b32 r2040, {high, low};
}
{
fma.rn.f16x2 r2042, r2034, r2040, r2037;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2042;
mov.b32 r2046, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2042;
mov.b32 r2048, {high, high};
}
{
mul.f16x2 r2050, r1720, r2048;
}
{
neg.f16x2 r2053, r2050;
}
{
fma.rn.f16x2 r2055, r1717, r2046, r2053;
}
{
mul.f16x2 r2059, r1717, r2048;
}
{
fma.rn.f16x2 r2062, r1720, r2046, r2059;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r2066, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r2068, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
mov.b32 r2070, {low, high};
}
{
mul.f16x2 r2071, r2068, r2070;
}
{
mul.f16x2 r2074, r2042, r2066;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2042;
mov.b32 r2077, {high, low};
}
{
fma.rn.f16x2 r2079, r2071, r2077, r2074;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2079;
mov.b32 r2083, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2079;
mov.b32 r2085, {high, high};
}
{
mul.f16x2 r2087, r1732, r2085;
}
{
neg.f16x2 r2090, r2087;
}
{
fma.rn.f16x2 r2092, r1729, r2083, r2090;
}
{
mul.f16x2 r2096, r1729, r2085;
}
{
fma.rn.f16x2 r2099, r1732, r2083, r2096;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r2103, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1747;
mov.b32 r2105, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f249;
cvt.rn.f16.f32 high, f250;
mov.b32 r2107, {low, high};
}
{
mul.f16x2 r2108, r2105, r2107;
}
{
mul.f16x2 r2111, r2079, r2103;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2079;
mov.b32 r2114, {high, low};
}
{
fma.rn.f16x2 r2116, r2108, r2114, r2111;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2116;
mov.b32 r2120, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2116;
mov.b32 r2122, {high, high};
}
{
mul.f16x2 r2124, r1744, r2122;
}
{
neg.f16x2 r2127, r2124;
}
{
fma.rn.f16x2 r2129, r1741, r2120, r2127;
}
{
mul.f16x2 r2133, r1741, r2122;
}
{
fma.rn.f16x2 r2136, r1744, r2120, r2133;
}
barrier.sync 0;
mad.lo.s32 r2840, r2835, 576, r2839;
st.shared.u32 [r2840], r1675;
st.shared.u32 [r2840+48], r1759;
st.shared.u32 [r2840+96], r1796;
st.shared.u32 [r2840+144], r1833;
st.shared.u32 [r2840+192], r1870;
st.shared.u32 [r2840+240], r1907;
st.shared.u32 [r2840+288], r1944;
st.shared.u32 [r2840+336], r1981;
st.shared.u32 [r2840+384], r2018;
st.shared.u32 [r2840+432], r2055;
st.shared.u32 [r2840+480], r2092;
st.shared.u32 [r2840+528], r2129;
barrier.sync 0;
ld.shared.u32 r2165, [r2834];
ld.shared.u32 r2419, [r2834+576];
ld.shared.u32 r2253, [r2834+1152];
ld.shared.u32 r2507, [r2834+1728];
ld.shared.u32 r2162, [r2834+2304];
ld.shared.u32 r2416, [r2834+2880];
ld.shared.u32 r2250, [r2834+3456];
ld.shared.u32 r2504, [r2834+4032];
ld.shared.u32 r2163, [r2834+4608];
ld.shared.u32 r2417, [r2834+5184];
ld.shared.u32 r2251, [r2834+5760];
ld.shared.u32 r2505, [r2834+6336];
barrier.sync 0;
st.shared.u32 [r2840], r1678;
st.shared.u32 [r2840+48], r1766;
st.shared.u32 [r2840+96], r1803;
st.shared.u32 [r2840+144], r1840;
st.shared.u32 [r2840+192], r1877;
st.shared.u32 [r2840+240], r1914;
st.shared.u32 [r2840+288], r1951;
st.shared.u32 [r2840+336], r1988;
st.shared.u32 [r2840+384], r2025;
st.shared.u32 [r2840+432], r2062;
st.shared.u32 [r2840+480], r2099;
st.shared.u32 [r2840+528], r2136;
barrier.sync 0;
ld.shared.u32 r2171, [r2834];
ld.shared.u32 r2425, [r2834+576];
ld.shared.u32 r2259, [r2834+1152];
ld.shared.u32 r2513, [r2834+1728];
ld.shared.u32 r2168, [r2834+2304];
ld.shared.u32 r2422, [r2834+2880];
ld.shared.u32 r2256, [r2834+3456];
ld.shared.u32 r2510, [r2834+4032];
ld.shared.u32 r2169, [r2834+4608];
ld.shared.u32 r2423, [r2834+5184];
ld.shared.u32 r2257, [r2834+5760];
ld.shared.u32 r2511, [r2834+6336];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2157, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2158, {low, high};
}
{
neg.f16x2 r2159, r2158;
}
{
add.f16x2 r2161, r2162, r2163;
}
{
add.f16x2 r2164, r2165, r2161;
}
{
add.f16x2 r2167, r2168, r2169;
}
{
add.f16x2 r2170, r2171, r2167;
}
{
add.f16x2 r2173, r2162, r2163;
}
{
mul.f16x2 r2176, r2173, r2157;
}
{
add.f16x2 r2179, r2165, r2176;
}
{
sub.f16x2 r2182, r2168, r2169;
}
{
mul.f16x2 r2185, r2182, r2159;
}
{
add.f16x2 r2188, r2179, r2185;
}
{
add.f16x2 r2191, r2162, r2163;
}
{
mul.f16x2 r2194, r2191, r2157;
}
{
add.f16x2 r2197, r2165, r2194;
}
{
sub.f16x2 r2200, r2168, r2169;
}
{
mul.f16x2 r2203, r2200, r2159;
}
{
sub.f16x2 r2206, r2197, r2203;
}
{
add.f16x2 r2209, r2168, r2169;
}
{
mul.f16x2 r2212, r2209, r2157;
}
{
add.f16x2 r2215, r2171, r2212;
}
{
sub.f16x2 r2218, r2162, r2163;
}
{
mul.f16x2 r2221, r2218, r2159;
}
{
sub.f16x2 r2224, r2215, r2221;
}
{
add.f16x2 r2227, r2168, r2169;
}
{
mul.f16x2 r2230, r2227, r2157;
}
{
add.f16x2 r2233, r2171, r2230;
}
{
sub.f16x2 r2236, r2162, r2163;
}
{
mul.f16x2 r2239, r2236, r2159;
}
{
add.f16x2 r2242, r2233, r2239;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2245, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2246, {low, high};
}
{
neg.f16x2 r2247, r2246;
}
{
add.f16x2 r2249, r2250, r2251;
}
{
add.f16x2 r2252, r2253, r2249;
}
{
add.f16x2 r2255, r2256, r2257;
}
{
add.f16x2 r2258, r2259, r2255;
}
{
add.f16x2 r2261, r2250, r2251;
}
{
mul.f16x2 r2264, r2261, r2245;
}
{
add.f16x2 r2267, r2253, r2264;
}
{
sub.f16x2 r2270, r2256, r2257;
}
{
mul.f16x2 r2273, r2270, r2247;
}
{
add.f16x2 r2276, r2267, r2273;
}
{
add.f16x2 r2279, r2250, r2251;
}
{
mul.f16x2 r2282, r2279, r2245;
}
{
add.f16x2 r2285, r2253, r2282;
}
{
sub.f16x2 r2288, r2256, r2257;
}
{
mul.f16x2 r2291, r2288, r2247;
}
{
sub.f16x2 r2294, r2285, r2291;
}
{
add.f16x2 r2297, r2256, r2257;
}
{
mul.f16x2 r2300, r2297, r2245;
}
{
add.f16x2 r2303, r2259, r2300;
}
{
sub.f16x2 r2306, r2250, r2251;
}
{
mul.f16x2 r2309, r2306, r2247;
}
{
sub.f16x2 r2312, r2303, r2309;
}
{
add.f16x2 r2315, r2256, r2257;
}
{
mul.f16x2 r2318, r2315, r2245;
}
{
add.f16x2 r2321, r2259, r2318;
}
{
sub.f16x2 r2324, r2250, r2251;
}
{
mul.f16x2 r2327, r2324, r2247;
}
{
add.f16x2 r2330, r2321, r2327;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f314;
cvt.rn.f16.f32 high, f314;
mov.b32 r2333, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2334, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2335, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2336, {low, high};
}
{
mul.f16x2 r2343, r2276, r2333;
}
{
mul.f16x2 r2346, r2312, r2334;
}
{
sub.f16x2 r2349, r2343, r2346;
}
{
mul.f16x2 r2352, r2276, r2334;
}
{
fma.rn.f16x2 r2355, r2312, r2333, r2352;
}
{
mul.f16x2 r2359, r2294, r2335;
}
{
mul.f16x2 r2362, r2330, r2336;
}
{
sub.f16x2 r2365, r2359, r2362;
}
{
mul.f16x2 r2368, r2294, r2336;
}
{
fma.rn.f16x2 r2371, r2330, r2335, r2368;
}
{
add.f16x2 r2375, r2164, r2252;
}
{
add.f16x2 r2378, r2170, r2258;
}
{
sub.f16x2 r2381, r2164, r2252;
}
{
sub.f16x2 r2384, r2170, r2258;
}
{
add.f16x2 r2387, r2188, r2349;
}
{
add.f16x2 r2390, r2224, r2355;
}
{
sub.f16x2 r2393, r2188, r2349;
}
{
sub.f16x2 r2396, r2224, r2355;
}
{
add.f16x2 r2399, r2206, r2365;
}
{
add.f16x2 r2402, r2242, r2371;
}
{
sub.f16x2 r2405, r2206, r2365;
}
{
sub.f16x2 r2408, r2242, r2371;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2411, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2412, {low, high};
}
{
neg.f16x2 r2413, r2412;
}
{
add.f16x2 r2415, r2416, r2417;
}
{
add.f16x2 r2418, r2419, r2415;
}
{
add.f16x2 r2421, r2422, r2423;
}
{
add.f16x2 r2424, r2425, r2421;
}
{
add.f16x2 r2427, r2416, r2417;
}
{
mul.f16x2 r2430, r2427, r2411;
}
{
add.f16x2 r2433, r2419, r2430;
}
{
sub.f16x2 r2436, r2422, r2423;
}
{
mul.f16x2 r2439, r2436, r2413;
}
{
add.f16x2 r2442, r2433, r2439;
}
{
add.f16x2 r2445, r2416, r2417;
}
{
mul.f16x2 r2448, r2445, r2411;
}
{
add.f16x2 r2451, r2419, r2448;
}
{
sub.f16x2 r2454, r2422, r2423;
}
{
mul.f16x2 r2457, r2454, r2413;
}
{
sub.f16x2 r2460, r2451, r2457;
}
{
add.f16x2 r2463, r2422, r2423;
}
{
mul.f16x2 r2466, r2463, r2411;
}
{
add.f16x2 r2469, r2425, r2466;
}
{
sub.f16x2 r2472, r2416, r2417;
}
{
mul.f16x2 r2475, r2472, r2413;
}
{
sub.f16x2 r2478, r2469, r2475;
}
{
add.f16x2 r2481, r2422, r2423;
}
{
mul.f16x2 r2484, r2481, r2411;
}
{
add.f16x2 r2487, r2425, r2484;
}
{
sub.f16x2 r2490, r2416, r2417;
}
{
mul.f16x2 r2493, r2490, r2413;
}
{
add.f16x2 r2496, r2487, r2493;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2499, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2500, {low, high};
}
{
neg.f16x2 r2501, r2500;
}
{
add.f16x2 r2503, r2504, r2505;
}
{
add.f16x2 r2506, r2507, r2503;
}
{
add.f16x2 r2509, r2510, r2511;
}
{
add.f16x2 r2512, r2513, r2509;
}
{
add.f16x2 r2515, r2504, r2505;
}
{
mul.f16x2 r2518, r2515, r2499;
}
{
add.f16x2 r2521, r2507, r2518;
}
{
sub.f16x2 r2524, r2510, r2511;
}
{
mul.f16x2 r2527, r2524, r2501;
}
{
add.f16x2 r2530, r2521, r2527;
}
{
add.f16x2 r2533, r2504, r2505;
}
{
mul.f16x2 r2536, r2533, r2499;
}
{
add.f16x2 r2539, r2507, r2536;
}
{
sub.f16x2 r2542, r2510, r2511;
}
{
mul.f16x2 r2545, r2542, r2501;
}
{
sub.f16x2 r2548, r2539, r2545;
}
{
add.f16x2 r2551, r2510, r2511;
}
{
mul.f16x2 r2554, r2551, r2499;
}
{
add.f16x2 r2557, r2513, r2554;
}
{
sub.f16x2 r2560, r2504, r2505;
}
{
mul.f16x2 r2563, r2560, r2501;
}
{
sub.f16x2 r2566, r2557, r2563;
}
{
add.f16x2 r2569, r2510, r2511;
}
{
mul.f16x2 r2572, r2569, r2499;
}
{
add.f16x2 r2575, r2513, r2572;
}
{
sub.f16x2 r2578, r2504, r2505;
}
{
mul.f16x2 r2581, r2578, r2501;
}
{
add.f16x2 r2584, r2575, r2581;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f314;
cvt.rn.f16.f32 high, f314;
mov.b32 r2587, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2588, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2589, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2590, {low, high};
}
{
mul.f16x2 r2597, r2530, r2587;
}
{
mul.f16x2 r2600, r2566, r2588;
}
{
sub.f16x2 r2603, r2597, r2600;
}
{
mul.f16x2 r2606, r2530, r2588;
}
{
fma.rn.f16x2 r2609, r2566, r2587, r2606;
}
{
mul.f16x2 r2613, r2548, r2589;
}
{
mul.f16x2 r2616, r2584, r2590;
}
{
sub.f16x2 r2619, r2613, r2616;
}
{
mul.f16x2 r2622, r2548, r2590;
}
{
fma.rn.f16x2 r2625, r2584, r2589, r2622;
}
{
add.f16x2 r2629, r2418, r2506;
}
{
add.f16x2 r2632, r2424, r2512;
}
{
sub.f16x2 r2635, r2418, r2506;
}
{
sub.f16x2 r2638, r2424, r2512;
}
{
add.f16x2 r2641, r2442, r2603;
}
{
add.f16x2 r2644, r2478, r2609;
}
{
sub.f16x2 r2647, r2442, r2603;
}
{
sub.f16x2 r2650, r2478, r2609;
}
{
add.f16x2 r2653, r2460, r2619;
}
{
add.f16x2 r2656, r2496, r2625;
}
{
sub.f16x2 r2659, r2460, r2619;
}
{
sub.f16x2 r2662, r2496, r2625;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f310;
cvt.rn.f16.f32 high, f310;
mov.b32 r2665, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2666, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f314;
cvt.rn.f16.f32 high, f314;
mov.b32 r2667, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2668, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2671, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2672, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f326;
cvt.rn.f16.f32 high, f326;
mov.b32 r2673, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f328;
cvt.rn.f16.f32 high, f328;
mov.b32 r2674, {low, high};
}
{
mul.f16x2 r2687, r2641, r2665;
}
{
mul.f16x2 r2690, r2644, r2666;
}
{
sub.f16x2 r2693, r2687, r2690;
}
{
mul.f16x2 r2696, r2641, r2666;
}
{
fma.rn.f16x2 r2699, r2644, r2665, r2696;
}
{
mul.f16x2 r2703, r2653, r2667;
}
{
mul.f16x2 r2706, r2656, r2668;
}
{
sub.f16x2 r2709, r2703, r2706;
}
{
mul.f16x2 r2712, r2653, r2668;
}
{
fma.rn.f16x2 r2715, r2656, r2667, r2712;
}
{
neg.f16x2 r2719, r2635;
}
{
mul.f16x2 r2721, r2647, r2671;
}
{
mul.f16x2 r2724, r2650, r2672;
}
{
sub.f16x2 r2727, r2721, r2724;
}
{
mul.f16x2 r2730, r2647, r2672;
}
{
fma.rn.f16x2 r2733, r2650, r2671, r2730;
}
{
mul.f16x2 r2737, r2659, r2673;
}
{
mul.f16x2 r2740, r2662, r2674;
}
{
sub.f16x2 r2743, r2737, r2740;
}
{
mul.f16x2 r2746, r2659, r2674;
}
{
fma.rn.f16x2 r2749, r2662, r2673, r2746;
}
{
add.f16x2 %0, r2375, r2629;
}
{
add.f16x2 %1, r2378, r2632;
}
{
sub.f16x2 %12, r2375, r2629;
}
{
sub.f16x2 %13, r2378, r2632;
}
{
add.f16x2 %2, r2387, r2693;
}
{
add.f16x2 %3, r2390, r2699;
}
{
sub.f16x2 %14, r2387, r2693;
}
{
sub.f16x2 %15, r2390, r2699;
}
{
add.f16x2 %4, r2399, r2709;
}
{
add.f16x2 %5, r2402, r2715;
}
{
sub.f16x2 %16, r2399, r2709;
}
{
sub.f16x2 %17, r2402, r2715;
}
{
add.f16x2 %6, r2381, r2638;
}
{
add.f16x2 %7, r2384, r2719;
}
{
sub.f16x2 %18, r2381, r2638;
}
{
sub.f16x2 %19, r2384, r2719;
}
{
add.f16x2 %8, r2393, r2727;
}
{
add.f16x2 %9, r2396, r2733;
}
{
sub.f16x2 %20, r2393, r2727;
}
{
sub.f16x2 %21, r2396, r2733;
}
{
add.f16x2 %10, r2405, r2743;
}
{
add.f16x2 %11, r2408, r2749;
}
{
sub.f16x2 %22, r2405, r2743;
}
{
sub.f16x2 %23, r2408, r2749;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)));
};


#endif
