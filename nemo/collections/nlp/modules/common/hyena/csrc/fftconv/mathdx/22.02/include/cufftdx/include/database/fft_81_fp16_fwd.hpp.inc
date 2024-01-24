#ifndef CUFFTDX_FFT_81_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_81_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<872, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<136>;
.reg .b32 r<1527>;
.reg .b64 rd<4>;
mov.u32 r1516, %tid.y;
mov.u32 r1517, %18;
mad.lo.s32 r1518, r1516, 648, r1517;
mov.u32 r1519, %tid.x;
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
neg.f16x2 r3, r2;
}
{
add.f16x2 r5, %25, %31;
}
{
add.f16x2 r8, %19, r5;
}
{
add.f16x2 r11, %26, %32;
}
{
add.f16x2 r14, %20, r11;
}
{
add.f16x2 r17, %25, %31;
}
{
mul.f16x2 r20, r17, r1;
}
{
add.f16x2 r23, %19, r20;
}
{
sub.f16x2 r26, %26, %32;
}
{
mul.f16x2 r29, r26, r3;
}
{
add.f16x2 r32, r23, r29;
}
{
add.f16x2 r35, %25, %31;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %19, r38;
}
{
sub.f16x2 r44, %26, %32;
}
{
mul.f16x2 r47, r44, r3;
}
{
sub.f16x2 r50, r41, r47;
}
{
add.f16x2 r53, %26, %32;
}
{
mul.f16x2 r56, r53, r1;
}
{
add.f16x2 r59, %20, r56;
}
{
sub.f16x2 r62, %25, %31;
}
{
mul.f16x2 r65, r62, r3;
}
{
sub.f16x2 r68, r59, r65;
}
{
add.f16x2 r71, %26, %32;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %20, r74;
}
{
sub.f16x2 r80, %25, %31;
}
{
mul.f16x2 r83, r80, r3;
}
{
add.f16x2 r86, r77, r83;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r90, {low, high};
}
{
neg.f16x2 r91, r90;
}
{
add.f16x2 r93, %27, %33;
}
{
add.f16x2 r96, %21, r93;
}
{
add.f16x2 r99, %28, %34;
}
{
add.f16x2 r102, %22, r99;
}
{
add.f16x2 r105, %27, %33;
}
{
mul.f16x2 r108, r105, r89;
}
{
add.f16x2 r111, %21, r108;
}
{
sub.f16x2 r114, %28, %34;
}
{
mul.f16x2 r117, r114, r91;
}
{
add.f16x2 r120, r111, r117;
}
{
add.f16x2 r123, %27, %33;
}
{
mul.f16x2 r126, r123, r89;
}
{
add.f16x2 r129, %21, r126;
}
{
sub.f16x2 r132, %28, %34;
}
{
mul.f16x2 r135, r132, r91;
}
{
sub.f16x2 r138, r129, r135;
}
{
add.f16x2 r141, %28, %34;
}
{
mul.f16x2 r144, r141, r89;
}
{
add.f16x2 r147, %22, r144;
}
{
sub.f16x2 r150, %27, %33;
}
{
mul.f16x2 r153, r150, r91;
}
{
sub.f16x2 r156, r147, r153;
}
{
add.f16x2 r159, %28, %34;
}
{
mul.f16x2 r162, r159, r89;
}
{
add.f16x2 r165, %22, r162;
}
{
sub.f16x2 r168, %27, %33;
}
{
mul.f16x2 r171, r168, r91;
}
{
add.f16x2 r174, r165, r171;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r177, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r178, {low, high};
}
{
neg.f16x2 r179, r178;
}
{
add.f16x2 r181, %29, %35;
}
{
add.f16x2 r184, %23, r181;
}
{
add.f16x2 r187, %30, %36;
}
{
add.f16x2 r190, %24, r187;
}
{
add.f16x2 r193, %29, %35;
}
{
mul.f16x2 r196, r193, r177;
}
{
add.f16x2 r199, %23, r196;
}
{
sub.f16x2 r202, %30, %36;
}
{
mul.f16x2 r205, r202, r179;
}
{
add.f16x2 r208, r199, r205;
}
{
add.f16x2 r211, %29, %35;
}
{
mul.f16x2 r214, r211, r177;
}
{
add.f16x2 r217, %23, r214;
}
{
sub.f16x2 r220, %30, %36;
}
{
mul.f16x2 r223, r220, r179;
}
{
sub.f16x2 r226, r217, r223;
}
{
add.f16x2 r229, %30, %36;
}
{
mul.f16x2 r232, r229, r177;
}
{
add.f16x2 r235, %24, r232;
}
{
sub.f16x2 r238, %29, %35;
}
{
mul.f16x2 r241, r238, r179;
}
{
sub.f16x2 r244, r235, r241;
}
{
add.f16x2 r247, %30, %36;
}
{
mul.f16x2 r250, r247, r177;
}
{
add.f16x2 r253, %24, r250;
}
{
sub.f16x2 r256, %29, %35;
}
{
mul.f16x2 r259, r256, r179;
}
{
add.f16x2 r262, r253, r259;
}
mov.f32 f90, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r265, {low, high};
}
mov.f32 f92, 0fBF248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r266, {low, high};
}
mov.f32 f94, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f94;
cvt.rn.f16.f32 high, f94;
mov.b32 r267, {low, high};
}
mov.f32 f96, 0fBF7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r268, {low, high};
}
mov.f32 f102, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r271, {low, high};
}
mov.f32 f104, 0fBEAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r272, {low, high};
}
{
mul.f16x2 r281, r120, r265;
}
{
mul.f16x2 r284, r156, r266;
}
{
sub.f16x2 r287, r281, r284;
}
{
mul.f16x2 r290, r120, r266;
}
{
fma.rn.f16x2 r293, r156, r265, r290;
}
{
mul.f16x2 r297, r208, r267;
}
{
mul.f16x2 r300, r244, r268;
}
{
sub.f16x2 r303, r297, r300;
}
{
mul.f16x2 r306, r208, r268;
}
{
fma.rn.f16x2 r309, r244, r267, r306;
}
{
mul.f16x2 r313, r138, r267;
}
{
mul.f16x2 r316, r174, r268;
}
{
sub.f16x2 r319, r313, r316;
}
{
mul.f16x2 r322, r138, r268;
}
{
fma.rn.f16x2 r325, r174, r267, r322;
}
{
mul.f16x2 r329, r226, r271;
}
{
mul.f16x2 r332, r262, r272;
}
{
sub.f16x2 r335, r329, r332;
}
{
mul.f16x2 r338, r226, r272;
}
{
fma.rn.f16x2 r341, r262, r271, r338;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r345, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r346, {low, high};
}
{
neg.f16x2 r347, r346;
}
{
add.f16x2 r349, r96, r184;
}
{
add.f16x2 r352, r8, r349;
}
{
add.f16x2 r355, r102, r190;
}
{
add.f16x2 r358, r14, r355;
}
{
add.f16x2 r361, r96, r184;
}
{
mul.f16x2 r364, r361, r345;
}
{
add.f16x2 r367, r8, r364;
}
{
sub.f16x2 r370, r102, r190;
}
{
mul.f16x2 r373, r370, r347;
}
{
add.f16x2 r376, r367, r373;
}
{
add.f16x2 r379, r96, r184;
}
{
mul.f16x2 r382, r379, r345;
}
{
add.f16x2 r385, r8, r382;
}
{
sub.f16x2 r388, r102, r190;
}
{
mul.f16x2 r391, r388, r347;
}
{
sub.f16x2 r394, r385, r391;
}
{
add.f16x2 r397, r102, r190;
}
{
mul.f16x2 r400, r397, r345;
}
{
add.f16x2 r403, r14, r400;
}
{
sub.f16x2 r406, r96, r184;
}
{
mul.f16x2 r409, r406, r347;
}
{
sub.f16x2 r412, r403, r409;
}
{
add.f16x2 r415, r102, r190;
}
{
mul.f16x2 r418, r415, r345;
}
{
add.f16x2 r421, r14, r418;
}
{
sub.f16x2 r424, r96, r184;
}
{
mul.f16x2 r427, r424, r347;
}
{
add.f16x2 r430, r421, r427;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r433, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r434, {low, high};
}
{
neg.f16x2 r435, r434;
}
{
add.f16x2 r437, r287, r303;
}
{
add.f16x2 r440, r32, r437;
}
{
add.f16x2 r443, r293, r309;
}
{
add.f16x2 r446, r68, r443;
}
{
add.f16x2 r449, r287, r303;
}
{
mul.f16x2 r452, r449, r433;
}
{
add.f16x2 r455, r32, r452;
}
{
sub.f16x2 r458, r293, r309;
}
{
mul.f16x2 r461, r458, r435;
}
{
add.f16x2 r464, r455, r461;
}
{
add.f16x2 r467, r287, r303;
}
{
mul.f16x2 r470, r467, r433;
}
{
add.f16x2 r473, r32, r470;
}
{
sub.f16x2 r476, r293, r309;
}
{
mul.f16x2 r479, r476, r435;
}
{
sub.f16x2 r482, r473, r479;
}
{
add.f16x2 r485, r293, r309;
}
{
mul.f16x2 r488, r485, r433;
}
{
add.f16x2 r491, r68, r488;
}
{
sub.f16x2 r494, r287, r303;
}
{
mul.f16x2 r497, r494, r435;
}
{
sub.f16x2 r500, r491, r497;
}
{
add.f16x2 r503, r293, r309;
}
{
mul.f16x2 r506, r503, r433;
}
{
add.f16x2 r509, r68, r506;
}
{
sub.f16x2 r512, r287, r303;
}
{
mul.f16x2 r515, r512, r435;
}
{
add.f16x2 r518, r509, r515;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r521, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r522, {low, high};
}
{
neg.f16x2 r523, r522;
}
{
add.f16x2 r525, r319, r335;
}
{
add.f16x2 r528, r50, r525;
}
{
add.f16x2 r531, r325, r341;
}
{
add.f16x2 r534, r86, r531;
}
{
add.f16x2 r537, r319, r335;
}
{
mul.f16x2 r540, r537, r521;
}
{
add.f16x2 r543, r50, r540;
}
{
sub.f16x2 r546, r325, r341;
}
{
mul.f16x2 r549, r546, r523;
}
{
add.f16x2 r552, r543, r549;
}
{
add.f16x2 r555, r319, r335;
}
{
mul.f16x2 r558, r555, r521;
}
{
add.f16x2 r561, r50, r558;
}
{
sub.f16x2 r564, r325, r341;
}
{
mul.f16x2 r567, r564, r523;
}
{
sub.f16x2 r570, r561, r567;
}
{
add.f16x2 r573, r325, r341;
}
{
mul.f16x2 r576, r573, r521;
}
{
add.f16x2 r579, r86, r576;
}
{
sub.f16x2 r582, r319, r335;
}
{
mul.f16x2 r585, r582, r523;
}
{
sub.f16x2 r588, r579, r585;
}
{
add.f16x2 r591, r325, r341;
}
{
mul.f16x2 r594, r591, r521;
}
{
add.f16x2 r597, r86, r594;
}
{
sub.f16x2 r600, r319, r335;
}
{
mul.f16x2 r603, r600, r523;
}
{
add.f16x2 r606, r597, r603;
}
mul.wide.u32 rd2, r1519, 954437177;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r1520, rd3;
mul.lo.s32 r1521, r1520, 9;
sub.s32 r1522, r1519, r1521;
cvt.rn.f32.u32 f133, r1522;
mul.f32 f134, f133, 0f3D9EDD1F;
cos.approx.f32 f57, f134;
sin.approx.f32 f135, f134;
neg.f32 f58, f135;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f57;
cvt.rn.f16.f32 high, f58;
mov.b32 r609, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r612, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r614, {high, high};
}
{
mul.f16x2 r616, r446, r614;
}
{
neg.f16x2 r619, r616;
}
{
fma.rn.f16x2 r621, r440, r612, r619;
}
{
mul.f16x2 r625, r440, r614;
}
{
fma.rn.f16x2 r628, r446, r612, r625;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r632, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r634, {high, high};
}
mov.f32 f73, 0fBF800000;
mov.f32 f74, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r636, {low, high};
}
{
mul.f16x2 r637, r634, r636;
}
{
mul.f16x2 r640, r609, r632;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r643, {high, low};
}
{
fma.rn.f16x2 r645, r637, r643, r640;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r645;
mov.b32 r649, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r645;
mov.b32 r651, {high, high};
}
{
mul.f16x2 r653, r534, r651;
}
{
neg.f16x2 r656, r653;
}
{
fma.rn.f16x2 r658, r528, r649, r656;
}
{
mul.f16x2 r662, r528, r651;
}
{
fma.rn.f16x2 r665, r534, r649, r662;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r669, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r671, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r673, {low, high};
}
{
mul.f16x2 r674, r671, r673;
}
{
mul.f16x2 r677, r645, r669;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r645;
mov.b32 r680, {high, low};
}
{
fma.rn.f16x2 r682, r674, r680, r677;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r682;
mov.b32 r686, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r682;
mov.b32 r688, {high, high};
}
{
mul.f16x2 r690, r412, r688;
}
{
neg.f16x2 r693, r690;
}
{
fma.rn.f16x2 r695, r376, r686, r693;
}
{
mul.f16x2 r699, r376, r688;
}
{
fma.rn.f16x2 r702, r412, r686, r699;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r706, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r708, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r710, {low, high};
}
{
mul.f16x2 r711, r708, r710;
}
{
mul.f16x2 r714, r682, r706;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r682;
mov.b32 r717, {high, low};
}
{
fma.rn.f16x2 r719, r711, r717, r714;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r719;
mov.b32 r723, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r719;
mov.b32 r725, {high, high};
}
{
mul.f16x2 r727, r500, r725;
}
{
neg.f16x2 r730, r727;
}
{
fma.rn.f16x2 r732, r464, r723, r730;
}
{
mul.f16x2 r736, r464, r725;
}
{
fma.rn.f16x2 r739, r500, r723, r736;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r743, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r745, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r747, {low, high};
}
{
mul.f16x2 r748, r745, r747;
}
{
mul.f16x2 r751, r719, r743;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r719;
mov.b32 r754, {high, low};
}
{
fma.rn.f16x2 r756, r748, r754, r751;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r756;
mov.b32 r760, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r756;
mov.b32 r762, {high, high};
}
{
mul.f16x2 r764, r588, r762;
}
{
neg.f16x2 r767, r764;
}
{
fma.rn.f16x2 r769, r552, r760, r767;
}
{
mul.f16x2 r773, r552, r762;
}
{
fma.rn.f16x2 r776, r588, r760, r773;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r780, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r782, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r784, {low, high};
}
{
mul.f16x2 r785, r782, r784;
}
{
mul.f16x2 r788, r756, r780;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r756;
mov.b32 r791, {high, low};
}
{
fma.rn.f16x2 r793, r785, r791, r788;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r793;
mov.b32 r797, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r793;
mov.b32 r799, {high, high};
}
{
mul.f16x2 r801, r430, r799;
}
{
neg.f16x2 r804, r801;
}
{
fma.rn.f16x2 r806, r394, r797, r804;
}
{
mul.f16x2 r810, r394, r799;
}
{
fma.rn.f16x2 r813, r430, r797, r810;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r817, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r819, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r821, {low, high};
}
{
mul.f16x2 r822, r819, r821;
}
{
mul.f16x2 r825, r793, r817;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r793;
mov.b32 r828, {high, low};
}
{
fma.rn.f16x2 r830, r822, r828, r825;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r830;
mov.b32 r834, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r830;
mov.b32 r836, {high, high};
}
{
mul.f16x2 r838, r518, r836;
}
{
neg.f16x2 r841, r838;
}
{
fma.rn.f16x2 r843, r482, r834, r841;
}
{
mul.f16x2 r847, r482, r836;
}
{
fma.rn.f16x2 r850, r518, r834, r847;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r854, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r856, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r858, {low, high};
}
{
mul.f16x2 r859, r856, r858;
}
{
mul.f16x2 r862, r830, r854;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r830;
mov.b32 r865, {high, low};
}
{
fma.rn.f16x2 r867, r859, r865, r862;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r867;
mov.b32 r871, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r867;
mov.b32 r873, {high, high};
}
{
mul.f16x2 r875, r606, r873;
}
{
neg.f16x2 r878, r875;
}
{
fma.rn.f16x2 r880, r570, r871, r878;
}
{
mul.f16x2 r884, r570, r873;
}
{
fma.rn.f16x2 r887, r606, r871, r884;
}
mad.lo.s32 r1523, r1520, 648, r1518;
barrier.sync 0;
mad.lo.s32 r1524, r1522, 72, r1523;
st.shared.v2.f32 [r1524], {r352, r358};
st.shared.v2.f32 [r1524+8], {r621, r628};
st.shared.v2.f32 [r1524+16], {r658, r665};
st.shared.v2.f32 [r1524+24], {r695, r702};
st.shared.v2.f32 [r1524+32], {r732, r739};
st.shared.v2.f32 [r1524+40], {r769, r776};
st.shared.v2.f32 [r1524+48], {r806, r813};
st.shared.v2.f32 [r1524+56], {r843, r850};
st.shared.v2.f32 [r1524+64], {r880, r887};
barrier.sync 0;
shl.b32 r1525, r1522, 6;
sub.s32 r1526, r1524, r1525;
ld.shared.u32 r916, [r1526];
ld.shared.u32 r922, [r1526+4];
ld.shared.u32 r1004, [r1526+72];
ld.shared.u32 r1010, [r1526+76];
ld.shared.u32 r1092, [r1526+144];
ld.shared.u32 r1098, [r1526+148];
ld.shared.u32 r913, [r1526+216];
ld.shared.u32 r919, [r1526+220];
ld.shared.u32 r1001, [r1526+288];
ld.shared.u32 r1007, [r1526+292];
ld.shared.u32 r1089, [r1526+360];
ld.shared.u32 r1095, [r1526+364];
ld.shared.u32 r914, [r1526+432];
ld.shared.u32 r920, [r1526+436];
ld.shared.u32 r1002, [r1526+504];
ld.shared.u32 r1008, [r1526+508];
ld.shared.u32 r1090, [r1526+576];
ld.shared.u32 r1096, [r1526+580];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r908, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r909, {low, high};
}
{
neg.f16x2 r910, r909;
}
{
add.f16x2 r912, r913, r914;
}
{
add.f16x2 r915, r916, r912;
}
{
add.f16x2 r918, r919, r920;
}
{
add.f16x2 r921, r922, r918;
}
{
add.f16x2 r924, r913, r914;
}
{
mul.f16x2 r927, r924, r908;
}
{
add.f16x2 r930, r916, r927;
}
{
sub.f16x2 r933, r919, r920;
}
{
mul.f16x2 r936, r933, r910;
}
{
add.f16x2 r939, r930, r936;
}
{
add.f16x2 r942, r913, r914;
}
{
mul.f16x2 r945, r942, r908;
}
{
add.f16x2 r948, r916, r945;
}
{
sub.f16x2 r951, r919, r920;
}
{
mul.f16x2 r954, r951, r910;
}
{
sub.f16x2 r957, r948, r954;
}
{
add.f16x2 r960, r919, r920;
}
{
mul.f16x2 r963, r960, r908;
}
{
add.f16x2 r966, r922, r963;
}
{
sub.f16x2 r969, r913, r914;
}
{
mul.f16x2 r972, r969, r910;
}
{
sub.f16x2 r975, r966, r972;
}
{
add.f16x2 r978, r919, r920;
}
{
mul.f16x2 r981, r978, r908;
}
{
add.f16x2 r984, r922, r981;
}
{
sub.f16x2 r987, r913, r914;
}
{
mul.f16x2 r990, r987, r910;
}
{
add.f16x2 r993, r984, r990;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r996, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r997, {low, high};
}
{
neg.f16x2 r998, r997;
}
{
add.f16x2 r1000, r1001, r1002;
}
{
add.f16x2 r1003, r1004, r1000;
}
{
add.f16x2 r1006, r1007, r1008;
}
{
add.f16x2 r1009, r1010, r1006;
}
{
add.f16x2 r1012, r1001, r1002;
}
{
mul.f16x2 r1015, r1012, r996;
}
{
add.f16x2 r1018, r1004, r1015;
}
{
sub.f16x2 r1021, r1007, r1008;
}
{
mul.f16x2 r1024, r1021, r998;
}
{
add.f16x2 r1027, r1018, r1024;
}
{
add.f16x2 r1030, r1001, r1002;
}
{
mul.f16x2 r1033, r1030, r996;
}
{
add.f16x2 r1036, r1004, r1033;
}
{
sub.f16x2 r1039, r1007, r1008;
}
{
mul.f16x2 r1042, r1039, r998;
}
{
sub.f16x2 r1045, r1036, r1042;
}
{
add.f16x2 r1048, r1007, r1008;
}
{
mul.f16x2 r1051, r1048, r996;
}
{
add.f16x2 r1054, r1010, r1051;
}
{
sub.f16x2 r1057, r1001, r1002;
}
{
mul.f16x2 r1060, r1057, r998;
}
{
sub.f16x2 r1063, r1054, r1060;
}
{
add.f16x2 r1066, r1007, r1008;
}
{
mul.f16x2 r1069, r1066, r996;
}
{
add.f16x2 r1072, r1010, r1069;
}
{
sub.f16x2 r1075, r1001, r1002;
}
{
mul.f16x2 r1078, r1075, r998;
}
{
add.f16x2 r1081, r1072, r1078;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r1084, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r1085, {low, high};
}
{
neg.f16x2 r1086, r1085;
}
{
add.f16x2 r1088, r1089, r1090;
}
{
add.f16x2 r1091, r1092, r1088;
}
{
add.f16x2 r1094, r1095, r1096;
}
{
add.f16x2 r1097, r1098, r1094;
}
{
add.f16x2 r1100, r1089, r1090;
}
{
mul.f16x2 r1103, r1100, r1084;
}
{
add.f16x2 r1106, r1092, r1103;
}
{
sub.f16x2 r1109, r1095, r1096;
}
{
mul.f16x2 r1112, r1109, r1086;
}
{
add.f16x2 r1115, r1106, r1112;
}
{
add.f16x2 r1118, r1089, r1090;
}
{
mul.f16x2 r1121, r1118, r1084;
}
{
add.f16x2 r1124, r1092, r1121;
}
{
sub.f16x2 r1127, r1095, r1096;
}
{
mul.f16x2 r1130, r1127, r1086;
}
{
sub.f16x2 r1133, r1124, r1130;
}
{
add.f16x2 r1136, r1095, r1096;
}
{
mul.f16x2 r1139, r1136, r1084;
}
{
add.f16x2 r1142, r1098, r1139;
}
{
sub.f16x2 r1145, r1089, r1090;
}
{
mul.f16x2 r1148, r1145, r1086;
}
{
sub.f16x2 r1151, r1142, r1148;
}
{
add.f16x2 r1154, r1095, r1096;
}
{
mul.f16x2 r1157, r1154, r1084;
}
{
add.f16x2 r1160, r1098, r1157;
}
{
sub.f16x2 r1163, r1089, r1090;
}
{
mul.f16x2 r1166, r1163, r1086;
}
{
add.f16x2 r1169, r1160, r1166;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r1172, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r1173, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f94;
cvt.rn.f16.f32 high, f94;
mov.b32 r1174, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r1175, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r1178, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r1179, {low, high};
}
{
mul.f16x2 r1188, r1027, r1172;
}
{
mul.f16x2 r1191, r1063, r1173;
}
{
sub.f16x2 r1194, r1188, r1191;
}
{
mul.f16x2 r1197, r1027, r1173;
}
{
fma.rn.f16x2 r1200, r1063, r1172, r1197;
}
{
mul.f16x2 r1204, r1115, r1174;
}
{
mul.f16x2 r1207, r1151, r1175;
}
{
sub.f16x2 r1210, r1204, r1207;
}
{
mul.f16x2 r1213, r1115, r1175;
}
{
fma.rn.f16x2 r1216, r1151, r1174, r1213;
}
{
mul.f16x2 r1220, r1045, r1174;
}
{
mul.f16x2 r1223, r1081, r1175;
}
{
sub.f16x2 r1226, r1220, r1223;
}
{
mul.f16x2 r1229, r1045, r1175;
}
{
fma.rn.f16x2 r1232, r1081, r1174, r1229;
}
{
mul.f16x2 r1236, r1133, r1178;
}
{
mul.f16x2 r1239, r1169, r1179;
}
{
sub.f16x2 r1242, r1236, r1239;
}
{
mul.f16x2 r1245, r1133, r1179;
}
{
fma.rn.f16x2 r1248, r1169, r1178, r1245;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r1252, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r1253, {low, high};
}
{
neg.f16x2 r1254, r1253;
}
{
add.f16x2 r1256, r1003, r1091;
}
{
add.f16x2 %0, r915, r1256;
}
{
add.f16x2 r1262, r1009, r1097;
}
{
add.f16x2 %1, r921, r1262;
}
{
add.f16x2 r1268, r1003, r1091;
}
{
mul.f16x2 r1271, r1268, r1252;
}
{
add.f16x2 r1274, r915, r1271;
}
{
sub.f16x2 r1277, r1009, r1097;
}
{
mul.f16x2 r1280, r1277, r1254;
}
{
add.f16x2 %6, r1274, r1280;
}
{
add.f16x2 r1286, r1003, r1091;
}
{
mul.f16x2 r1289, r1286, r1252;
}
{
add.f16x2 r1292, r915, r1289;
}
{
sub.f16x2 r1295, r1009, r1097;
}
{
mul.f16x2 r1298, r1295, r1254;
}
{
sub.f16x2 %12, r1292, r1298;
}
{
add.f16x2 r1304, r1009, r1097;
}
{
mul.f16x2 r1307, r1304, r1252;
}
{
add.f16x2 r1310, r921, r1307;
}
{
sub.f16x2 r1313, r1003, r1091;
}
{
mul.f16x2 r1316, r1313, r1254;
}
{
sub.f16x2 %7, r1310, r1316;
}
{
add.f16x2 r1322, r1009, r1097;
}
{
mul.f16x2 r1325, r1322, r1252;
}
{
add.f16x2 r1328, r921, r1325;
}
{
sub.f16x2 r1331, r1003, r1091;
}
{
mul.f16x2 r1334, r1331, r1254;
}
{
add.f16x2 %13, r1328, r1334;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r1340, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r1341, {low, high};
}
{
neg.f16x2 r1342, r1341;
}
{
add.f16x2 r1344, r1194, r1210;
}
{
add.f16x2 %2, r939, r1344;
}
{
add.f16x2 r1350, r1200, r1216;
}
{
add.f16x2 %3, r975, r1350;
}
{
add.f16x2 r1356, r1194, r1210;
}
{
mul.f16x2 r1359, r1356, r1340;
}
{
add.f16x2 r1362, r939, r1359;
}
{
sub.f16x2 r1365, r1200, r1216;
}
{
mul.f16x2 r1368, r1365, r1342;
}
{
add.f16x2 %8, r1362, r1368;
}
{
add.f16x2 r1374, r1194, r1210;
}
{
mul.f16x2 r1377, r1374, r1340;
}
{
add.f16x2 r1380, r939, r1377;
}
{
sub.f16x2 r1383, r1200, r1216;
}
{
mul.f16x2 r1386, r1383, r1342;
}
{
sub.f16x2 %14, r1380, r1386;
}
{
add.f16x2 r1392, r1200, r1216;
}
{
mul.f16x2 r1395, r1392, r1340;
}
{
add.f16x2 r1398, r975, r1395;
}
{
sub.f16x2 r1401, r1194, r1210;
}
{
mul.f16x2 r1404, r1401, r1342;
}
{
sub.f16x2 %9, r1398, r1404;
}
{
add.f16x2 r1410, r1200, r1216;
}
{
mul.f16x2 r1413, r1410, r1340;
}
{
add.f16x2 r1416, r975, r1413;
}
{
sub.f16x2 r1419, r1194, r1210;
}
{
mul.f16x2 r1422, r1419, r1342;
}
{
add.f16x2 %15, r1416, r1422;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r1428, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r1429, {low, high};
}
{
neg.f16x2 r1430, r1429;
}
{
add.f16x2 r1432, r1226, r1242;
}
{
add.f16x2 %4, r957, r1432;
}
{
add.f16x2 r1438, r1232, r1248;
}
{
add.f16x2 %5, r993, r1438;
}
{
add.f16x2 r1444, r1226, r1242;
}
{
mul.f16x2 r1447, r1444, r1428;
}
{
add.f16x2 r1450, r957, r1447;
}
{
sub.f16x2 r1453, r1232, r1248;
}
{
mul.f16x2 r1456, r1453, r1430;
}
{
add.f16x2 %10, r1450, r1456;
}
{
add.f16x2 r1462, r1226, r1242;
}
{
mul.f16x2 r1465, r1462, r1428;
}
{
add.f16x2 r1468, r957, r1465;
}
{
sub.f16x2 r1471, r1232, r1248;
}
{
mul.f16x2 r1474, r1471, r1430;
}
{
sub.f16x2 %16, r1468, r1474;
}
{
add.f16x2 r1480, r1232, r1248;
}
{
mul.f16x2 r1483, r1480, r1428;
}
{
add.f16x2 r1486, r993, r1483;
}
{
sub.f16x2 r1489, r1226, r1242;
}
{
mul.f16x2 r1492, r1489, r1430;
}
{
sub.f16x2 %11, r1486, r1492;
}
{
add.f16x2 r1498, r1232, r1248;
}
{
mul.f16x2 r1501, r1498, r1428;
}
{
add.f16x2 r1504, r993, r1501;
}
{
sub.f16x2 r1507, r1226, r1242;
}
{
mul.f16x2 r1510, r1507, r1430;
}
{
add.f16x2 %17, r1504, r1510;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<873, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<136>;
.reg .b32 r<1527>;
.reg .b64 rd<4>;
mov.u32 r1516, %tid.y;
mov.u32 r1517, %18;
mad.lo.s32 r1518, r1516, 324, r1517;
mov.u32 r1519, %tid.x;
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
neg.f16x2 r3, r2;
}
{
add.f16x2 r5, %25, %31;
}
{
add.f16x2 r8, %19, r5;
}
{
add.f16x2 r11, %26, %32;
}
{
add.f16x2 r14, %20, r11;
}
{
add.f16x2 r17, %25, %31;
}
{
mul.f16x2 r20, r17, r1;
}
{
add.f16x2 r23, %19, r20;
}
{
sub.f16x2 r26, %26, %32;
}
{
mul.f16x2 r29, r26, r3;
}
{
add.f16x2 r32, r23, r29;
}
{
add.f16x2 r35, %25, %31;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %19, r38;
}
{
sub.f16x2 r44, %26, %32;
}
{
mul.f16x2 r47, r44, r3;
}
{
sub.f16x2 r50, r41, r47;
}
{
add.f16x2 r53, %26, %32;
}
{
mul.f16x2 r56, r53, r1;
}
{
add.f16x2 r59, %20, r56;
}
{
sub.f16x2 r62, %25, %31;
}
{
mul.f16x2 r65, r62, r3;
}
{
sub.f16x2 r68, r59, r65;
}
{
add.f16x2 r71, %26, %32;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %20, r74;
}
{
sub.f16x2 r80, %25, %31;
}
{
mul.f16x2 r83, r80, r3;
}
{
add.f16x2 r86, r77, r83;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r90, {low, high};
}
{
neg.f16x2 r91, r90;
}
{
add.f16x2 r93, %27, %33;
}
{
add.f16x2 r96, %21, r93;
}
{
add.f16x2 r99, %28, %34;
}
{
add.f16x2 r102, %22, r99;
}
{
add.f16x2 r105, %27, %33;
}
{
mul.f16x2 r108, r105, r89;
}
{
add.f16x2 r111, %21, r108;
}
{
sub.f16x2 r114, %28, %34;
}
{
mul.f16x2 r117, r114, r91;
}
{
add.f16x2 r120, r111, r117;
}
{
add.f16x2 r123, %27, %33;
}
{
mul.f16x2 r126, r123, r89;
}
{
add.f16x2 r129, %21, r126;
}
{
sub.f16x2 r132, %28, %34;
}
{
mul.f16x2 r135, r132, r91;
}
{
sub.f16x2 r138, r129, r135;
}
{
add.f16x2 r141, %28, %34;
}
{
mul.f16x2 r144, r141, r89;
}
{
add.f16x2 r147, %22, r144;
}
{
sub.f16x2 r150, %27, %33;
}
{
mul.f16x2 r153, r150, r91;
}
{
sub.f16x2 r156, r147, r153;
}
{
add.f16x2 r159, %28, %34;
}
{
mul.f16x2 r162, r159, r89;
}
{
add.f16x2 r165, %22, r162;
}
{
sub.f16x2 r168, %27, %33;
}
{
mul.f16x2 r171, r168, r91;
}
{
add.f16x2 r174, r165, r171;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r177, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r178, {low, high};
}
{
neg.f16x2 r179, r178;
}
{
add.f16x2 r181, %29, %35;
}
{
add.f16x2 r184, %23, r181;
}
{
add.f16x2 r187, %30, %36;
}
{
add.f16x2 r190, %24, r187;
}
{
add.f16x2 r193, %29, %35;
}
{
mul.f16x2 r196, r193, r177;
}
{
add.f16x2 r199, %23, r196;
}
{
sub.f16x2 r202, %30, %36;
}
{
mul.f16x2 r205, r202, r179;
}
{
add.f16x2 r208, r199, r205;
}
{
add.f16x2 r211, %29, %35;
}
{
mul.f16x2 r214, r211, r177;
}
{
add.f16x2 r217, %23, r214;
}
{
sub.f16x2 r220, %30, %36;
}
{
mul.f16x2 r223, r220, r179;
}
{
sub.f16x2 r226, r217, r223;
}
{
add.f16x2 r229, %30, %36;
}
{
mul.f16x2 r232, r229, r177;
}
{
add.f16x2 r235, %24, r232;
}
{
sub.f16x2 r238, %29, %35;
}
{
mul.f16x2 r241, r238, r179;
}
{
sub.f16x2 r244, r235, r241;
}
{
add.f16x2 r247, %30, %36;
}
{
mul.f16x2 r250, r247, r177;
}
{
add.f16x2 r253, %24, r250;
}
{
sub.f16x2 r256, %29, %35;
}
{
mul.f16x2 r259, r256, r179;
}
{
add.f16x2 r262, r253, r259;
}
mov.f32 f90, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r265, {low, high};
}
mov.f32 f92, 0fBF248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r266, {low, high};
}
mov.f32 f94, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f94;
cvt.rn.f16.f32 high, f94;
mov.b32 r267, {low, high};
}
mov.f32 f96, 0fBF7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r268, {low, high};
}
mov.f32 f102, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r271, {low, high};
}
mov.f32 f104, 0fBEAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r272, {low, high};
}
{
mul.f16x2 r281, r120, r265;
}
{
mul.f16x2 r284, r156, r266;
}
{
sub.f16x2 r287, r281, r284;
}
{
mul.f16x2 r290, r120, r266;
}
{
fma.rn.f16x2 r293, r156, r265, r290;
}
{
mul.f16x2 r297, r208, r267;
}
{
mul.f16x2 r300, r244, r268;
}
{
sub.f16x2 r303, r297, r300;
}
{
mul.f16x2 r306, r208, r268;
}
{
fma.rn.f16x2 r309, r244, r267, r306;
}
{
mul.f16x2 r313, r138, r267;
}
{
mul.f16x2 r316, r174, r268;
}
{
sub.f16x2 r319, r313, r316;
}
{
mul.f16x2 r322, r138, r268;
}
{
fma.rn.f16x2 r325, r174, r267, r322;
}
{
mul.f16x2 r329, r226, r271;
}
{
mul.f16x2 r332, r262, r272;
}
{
sub.f16x2 r335, r329, r332;
}
{
mul.f16x2 r338, r226, r272;
}
{
fma.rn.f16x2 r341, r262, r271, r338;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r345, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r346, {low, high};
}
{
neg.f16x2 r347, r346;
}
{
add.f16x2 r349, r96, r184;
}
{
add.f16x2 r352, r8, r349;
}
{
add.f16x2 r355, r102, r190;
}
{
add.f16x2 r358, r14, r355;
}
{
add.f16x2 r361, r96, r184;
}
{
mul.f16x2 r364, r361, r345;
}
{
add.f16x2 r367, r8, r364;
}
{
sub.f16x2 r370, r102, r190;
}
{
mul.f16x2 r373, r370, r347;
}
{
add.f16x2 r376, r367, r373;
}
{
add.f16x2 r379, r96, r184;
}
{
mul.f16x2 r382, r379, r345;
}
{
add.f16x2 r385, r8, r382;
}
{
sub.f16x2 r388, r102, r190;
}
{
mul.f16x2 r391, r388, r347;
}
{
sub.f16x2 r394, r385, r391;
}
{
add.f16x2 r397, r102, r190;
}
{
mul.f16x2 r400, r397, r345;
}
{
add.f16x2 r403, r14, r400;
}
{
sub.f16x2 r406, r96, r184;
}
{
mul.f16x2 r409, r406, r347;
}
{
sub.f16x2 r412, r403, r409;
}
{
add.f16x2 r415, r102, r190;
}
{
mul.f16x2 r418, r415, r345;
}
{
add.f16x2 r421, r14, r418;
}
{
sub.f16x2 r424, r96, r184;
}
{
mul.f16x2 r427, r424, r347;
}
{
add.f16x2 r430, r421, r427;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r433, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r434, {low, high};
}
{
neg.f16x2 r435, r434;
}
{
add.f16x2 r437, r287, r303;
}
{
add.f16x2 r440, r32, r437;
}
{
add.f16x2 r443, r293, r309;
}
{
add.f16x2 r446, r68, r443;
}
{
add.f16x2 r449, r287, r303;
}
{
mul.f16x2 r452, r449, r433;
}
{
add.f16x2 r455, r32, r452;
}
{
sub.f16x2 r458, r293, r309;
}
{
mul.f16x2 r461, r458, r435;
}
{
add.f16x2 r464, r455, r461;
}
{
add.f16x2 r467, r287, r303;
}
{
mul.f16x2 r470, r467, r433;
}
{
add.f16x2 r473, r32, r470;
}
{
sub.f16x2 r476, r293, r309;
}
{
mul.f16x2 r479, r476, r435;
}
{
sub.f16x2 r482, r473, r479;
}
{
add.f16x2 r485, r293, r309;
}
{
mul.f16x2 r488, r485, r433;
}
{
add.f16x2 r491, r68, r488;
}
{
sub.f16x2 r494, r287, r303;
}
{
mul.f16x2 r497, r494, r435;
}
{
sub.f16x2 r500, r491, r497;
}
{
add.f16x2 r503, r293, r309;
}
{
mul.f16x2 r506, r503, r433;
}
{
add.f16x2 r509, r68, r506;
}
{
sub.f16x2 r512, r287, r303;
}
{
mul.f16x2 r515, r512, r435;
}
{
add.f16x2 r518, r509, r515;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r521, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r522, {low, high};
}
{
neg.f16x2 r523, r522;
}
{
add.f16x2 r525, r319, r335;
}
{
add.f16x2 r528, r50, r525;
}
{
add.f16x2 r531, r325, r341;
}
{
add.f16x2 r534, r86, r531;
}
{
add.f16x2 r537, r319, r335;
}
{
mul.f16x2 r540, r537, r521;
}
{
add.f16x2 r543, r50, r540;
}
{
sub.f16x2 r546, r325, r341;
}
{
mul.f16x2 r549, r546, r523;
}
{
add.f16x2 r552, r543, r549;
}
{
add.f16x2 r555, r319, r335;
}
{
mul.f16x2 r558, r555, r521;
}
{
add.f16x2 r561, r50, r558;
}
{
sub.f16x2 r564, r325, r341;
}
{
mul.f16x2 r567, r564, r523;
}
{
sub.f16x2 r570, r561, r567;
}
{
add.f16x2 r573, r325, r341;
}
{
mul.f16x2 r576, r573, r521;
}
{
add.f16x2 r579, r86, r576;
}
{
sub.f16x2 r582, r319, r335;
}
{
mul.f16x2 r585, r582, r523;
}
{
sub.f16x2 r588, r579, r585;
}
{
add.f16x2 r591, r325, r341;
}
{
mul.f16x2 r594, r591, r521;
}
{
add.f16x2 r597, r86, r594;
}
{
sub.f16x2 r600, r319, r335;
}
{
mul.f16x2 r603, r600, r523;
}
{
add.f16x2 r606, r597, r603;
}
mul.wide.u32 rd2, r1519, 954437177;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r1520, rd3;
mul.lo.s32 r1521, r1520, 9;
sub.s32 r1522, r1519, r1521;
mad.lo.s32 r1523, r1520, 324, r1518;
cvt.rn.f32.u32 f133, r1522;
mul.f32 f134, f133, 0f3D9EDD1F;
cos.approx.f32 f57, f134;
sin.approx.f32 f135, f134;
neg.f32 f58, f135;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f57;
cvt.rn.f16.f32 high, f58;
mov.b32 r609, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r612, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r614, {high, high};
}
{
mul.f16x2 r616, r446, r614;
}
{
neg.f16x2 r619, r616;
}
{
fma.rn.f16x2 r621, r440, r612, r619;
}
{
mul.f16x2 r625, r440, r614;
}
{
fma.rn.f16x2 r628, r446, r612, r625;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r632, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r634, {high, high};
}
mov.f32 f73, 0fBF800000;
mov.f32 f74, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r636, {low, high};
}
{
mul.f16x2 r637, r634, r636;
}
{
mul.f16x2 r640, r609, r632;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r643, {high, low};
}
{
fma.rn.f16x2 r645, r637, r643, r640;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r645;
mov.b32 r649, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r645;
mov.b32 r651, {high, high};
}
{
mul.f16x2 r653, r534, r651;
}
{
neg.f16x2 r656, r653;
}
{
fma.rn.f16x2 r658, r528, r649, r656;
}
{
mul.f16x2 r662, r528, r651;
}
{
fma.rn.f16x2 r665, r534, r649, r662;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r669, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r671, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r673, {low, high};
}
{
mul.f16x2 r674, r671, r673;
}
{
mul.f16x2 r677, r645, r669;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r645;
mov.b32 r680, {high, low};
}
{
fma.rn.f16x2 r682, r674, r680, r677;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r682;
mov.b32 r686, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r682;
mov.b32 r688, {high, high};
}
{
mul.f16x2 r690, r412, r688;
}
{
neg.f16x2 r693, r690;
}
{
fma.rn.f16x2 r695, r376, r686, r693;
}
{
mul.f16x2 r699, r376, r688;
}
{
fma.rn.f16x2 r702, r412, r686, r699;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r706, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r708, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r710, {low, high};
}
{
mul.f16x2 r711, r708, r710;
}
{
mul.f16x2 r714, r682, r706;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r682;
mov.b32 r717, {high, low};
}
{
fma.rn.f16x2 r719, r711, r717, r714;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r719;
mov.b32 r723, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r719;
mov.b32 r725, {high, high};
}
{
mul.f16x2 r727, r500, r725;
}
{
neg.f16x2 r730, r727;
}
{
fma.rn.f16x2 r732, r464, r723, r730;
}
{
mul.f16x2 r736, r464, r725;
}
{
fma.rn.f16x2 r739, r500, r723, r736;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r743, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r745, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r747, {low, high};
}
{
mul.f16x2 r748, r745, r747;
}
{
mul.f16x2 r751, r719, r743;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r719;
mov.b32 r754, {high, low};
}
{
fma.rn.f16x2 r756, r748, r754, r751;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r756;
mov.b32 r760, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r756;
mov.b32 r762, {high, high};
}
{
mul.f16x2 r764, r588, r762;
}
{
neg.f16x2 r767, r764;
}
{
fma.rn.f16x2 r769, r552, r760, r767;
}
{
mul.f16x2 r773, r552, r762;
}
{
fma.rn.f16x2 r776, r588, r760, r773;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r780, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r782, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r784, {low, high};
}
{
mul.f16x2 r785, r782, r784;
}
{
mul.f16x2 r788, r756, r780;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r756;
mov.b32 r791, {high, low};
}
{
fma.rn.f16x2 r793, r785, r791, r788;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r793;
mov.b32 r797, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r793;
mov.b32 r799, {high, high};
}
{
mul.f16x2 r801, r430, r799;
}
{
neg.f16x2 r804, r801;
}
{
fma.rn.f16x2 r806, r394, r797, r804;
}
{
mul.f16x2 r810, r394, r799;
}
{
fma.rn.f16x2 r813, r430, r797, r810;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r817, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r819, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r821, {low, high};
}
{
mul.f16x2 r822, r819, r821;
}
{
mul.f16x2 r825, r793, r817;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r793;
mov.b32 r828, {high, low};
}
{
fma.rn.f16x2 r830, r822, r828, r825;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r830;
mov.b32 r834, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r830;
mov.b32 r836, {high, high};
}
{
mul.f16x2 r838, r518, r836;
}
{
neg.f16x2 r841, r838;
}
{
fma.rn.f16x2 r843, r482, r834, r841;
}
{
mul.f16x2 r847, r482, r836;
}
{
fma.rn.f16x2 r850, r518, r834, r847;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r854, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r856, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r858, {low, high};
}
{
mul.f16x2 r859, r856, r858;
}
{
mul.f16x2 r862, r830, r854;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r830;
mov.b32 r865, {high, low};
}
{
fma.rn.f16x2 r867, r859, r865, r862;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r867;
mov.b32 r871, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r867;
mov.b32 r873, {high, high};
}
{
mul.f16x2 r875, r606, r873;
}
{
neg.f16x2 r878, r875;
}
{
fma.rn.f16x2 r880, r570, r871, r878;
}
{
mul.f16x2 r884, r570, r873;
}
{
fma.rn.f16x2 r887, r606, r871, r884;
}
barrier.sync 0;
mad.lo.s32 r1524, r1522, 36, r1523;
st.shared.u32 [r1524], r352;
st.shared.u32 [r1524+4], r621;
st.shared.u32 [r1524+8], r658;
st.shared.u32 [r1524+12], r695;
st.shared.u32 [r1524+16], r732;
st.shared.u32 [r1524+20], r769;
st.shared.u32 [r1524+24], r806;
st.shared.u32 [r1524+28], r843;
st.shared.u32 [r1524+32], r880;
barrier.sync 0;
shl.b32 r1525, r1522, 5;
sub.s32 r1526, r1524, r1525;
ld.shared.u32 r916, [r1526];
ld.shared.u32 r1004, [r1526+36];
ld.shared.u32 r1092, [r1526+72];
ld.shared.u32 r913, [r1526+108];
ld.shared.u32 r1001, [r1526+144];
ld.shared.u32 r1089, [r1526+180];
ld.shared.u32 r914, [r1526+216];
ld.shared.u32 r1002, [r1526+252];
ld.shared.u32 r1090, [r1526+288];
barrier.sync 0;
st.shared.u32 [r1524], r358;
st.shared.u32 [r1524+4], r628;
st.shared.u32 [r1524+8], r665;
st.shared.u32 [r1524+12], r702;
st.shared.u32 [r1524+16], r739;
st.shared.u32 [r1524+20], r776;
st.shared.u32 [r1524+24], r813;
st.shared.u32 [r1524+28], r850;
st.shared.u32 [r1524+32], r887;
barrier.sync 0;
ld.shared.u32 r922, [r1526];
ld.shared.u32 r1010, [r1526+36];
ld.shared.u32 r1098, [r1526+72];
ld.shared.u32 r919, [r1526+108];
ld.shared.u32 r1007, [r1526+144];
ld.shared.u32 r1095, [r1526+180];
ld.shared.u32 r920, [r1526+216];
ld.shared.u32 r1008, [r1526+252];
ld.shared.u32 r1096, [r1526+288];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r908, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r909, {low, high};
}
{
neg.f16x2 r910, r909;
}
{
add.f16x2 r912, r913, r914;
}
{
add.f16x2 r915, r916, r912;
}
{
add.f16x2 r918, r919, r920;
}
{
add.f16x2 r921, r922, r918;
}
{
add.f16x2 r924, r913, r914;
}
{
mul.f16x2 r927, r924, r908;
}
{
add.f16x2 r930, r916, r927;
}
{
sub.f16x2 r933, r919, r920;
}
{
mul.f16x2 r936, r933, r910;
}
{
add.f16x2 r939, r930, r936;
}
{
add.f16x2 r942, r913, r914;
}
{
mul.f16x2 r945, r942, r908;
}
{
add.f16x2 r948, r916, r945;
}
{
sub.f16x2 r951, r919, r920;
}
{
mul.f16x2 r954, r951, r910;
}
{
sub.f16x2 r957, r948, r954;
}
{
add.f16x2 r960, r919, r920;
}
{
mul.f16x2 r963, r960, r908;
}
{
add.f16x2 r966, r922, r963;
}
{
sub.f16x2 r969, r913, r914;
}
{
mul.f16x2 r972, r969, r910;
}
{
sub.f16x2 r975, r966, r972;
}
{
add.f16x2 r978, r919, r920;
}
{
mul.f16x2 r981, r978, r908;
}
{
add.f16x2 r984, r922, r981;
}
{
sub.f16x2 r987, r913, r914;
}
{
mul.f16x2 r990, r987, r910;
}
{
add.f16x2 r993, r984, r990;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r996, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r997, {low, high};
}
{
neg.f16x2 r998, r997;
}
{
add.f16x2 r1000, r1001, r1002;
}
{
add.f16x2 r1003, r1004, r1000;
}
{
add.f16x2 r1006, r1007, r1008;
}
{
add.f16x2 r1009, r1010, r1006;
}
{
add.f16x2 r1012, r1001, r1002;
}
{
mul.f16x2 r1015, r1012, r996;
}
{
add.f16x2 r1018, r1004, r1015;
}
{
sub.f16x2 r1021, r1007, r1008;
}
{
mul.f16x2 r1024, r1021, r998;
}
{
add.f16x2 r1027, r1018, r1024;
}
{
add.f16x2 r1030, r1001, r1002;
}
{
mul.f16x2 r1033, r1030, r996;
}
{
add.f16x2 r1036, r1004, r1033;
}
{
sub.f16x2 r1039, r1007, r1008;
}
{
mul.f16x2 r1042, r1039, r998;
}
{
sub.f16x2 r1045, r1036, r1042;
}
{
add.f16x2 r1048, r1007, r1008;
}
{
mul.f16x2 r1051, r1048, r996;
}
{
add.f16x2 r1054, r1010, r1051;
}
{
sub.f16x2 r1057, r1001, r1002;
}
{
mul.f16x2 r1060, r1057, r998;
}
{
sub.f16x2 r1063, r1054, r1060;
}
{
add.f16x2 r1066, r1007, r1008;
}
{
mul.f16x2 r1069, r1066, r996;
}
{
add.f16x2 r1072, r1010, r1069;
}
{
sub.f16x2 r1075, r1001, r1002;
}
{
mul.f16x2 r1078, r1075, r998;
}
{
add.f16x2 r1081, r1072, r1078;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r1084, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r1085, {low, high};
}
{
neg.f16x2 r1086, r1085;
}
{
add.f16x2 r1088, r1089, r1090;
}
{
add.f16x2 r1091, r1092, r1088;
}
{
add.f16x2 r1094, r1095, r1096;
}
{
add.f16x2 r1097, r1098, r1094;
}
{
add.f16x2 r1100, r1089, r1090;
}
{
mul.f16x2 r1103, r1100, r1084;
}
{
add.f16x2 r1106, r1092, r1103;
}
{
sub.f16x2 r1109, r1095, r1096;
}
{
mul.f16x2 r1112, r1109, r1086;
}
{
add.f16x2 r1115, r1106, r1112;
}
{
add.f16x2 r1118, r1089, r1090;
}
{
mul.f16x2 r1121, r1118, r1084;
}
{
add.f16x2 r1124, r1092, r1121;
}
{
sub.f16x2 r1127, r1095, r1096;
}
{
mul.f16x2 r1130, r1127, r1086;
}
{
sub.f16x2 r1133, r1124, r1130;
}
{
add.f16x2 r1136, r1095, r1096;
}
{
mul.f16x2 r1139, r1136, r1084;
}
{
add.f16x2 r1142, r1098, r1139;
}
{
sub.f16x2 r1145, r1089, r1090;
}
{
mul.f16x2 r1148, r1145, r1086;
}
{
sub.f16x2 r1151, r1142, r1148;
}
{
add.f16x2 r1154, r1095, r1096;
}
{
mul.f16x2 r1157, r1154, r1084;
}
{
add.f16x2 r1160, r1098, r1157;
}
{
sub.f16x2 r1163, r1089, r1090;
}
{
mul.f16x2 r1166, r1163, r1086;
}
{
add.f16x2 r1169, r1160, r1166;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r1172, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r1173, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f94;
cvt.rn.f16.f32 high, f94;
mov.b32 r1174, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r1175, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r1178, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r1179, {low, high};
}
{
mul.f16x2 r1188, r1027, r1172;
}
{
mul.f16x2 r1191, r1063, r1173;
}
{
sub.f16x2 r1194, r1188, r1191;
}
{
mul.f16x2 r1197, r1027, r1173;
}
{
fma.rn.f16x2 r1200, r1063, r1172, r1197;
}
{
mul.f16x2 r1204, r1115, r1174;
}
{
mul.f16x2 r1207, r1151, r1175;
}
{
sub.f16x2 r1210, r1204, r1207;
}
{
mul.f16x2 r1213, r1115, r1175;
}
{
fma.rn.f16x2 r1216, r1151, r1174, r1213;
}
{
mul.f16x2 r1220, r1045, r1174;
}
{
mul.f16x2 r1223, r1081, r1175;
}
{
sub.f16x2 r1226, r1220, r1223;
}
{
mul.f16x2 r1229, r1045, r1175;
}
{
fma.rn.f16x2 r1232, r1081, r1174, r1229;
}
{
mul.f16x2 r1236, r1133, r1178;
}
{
mul.f16x2 r1239, r1169, r1179;
}
{
sub.f16x2 r1242, r1236, r1239;
}
{
mul.f16x2 r1245, r1133, r1179;
}
{
fma.rn.f16x2 r1248, r1169, r1178, r1245;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r1252, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r1253, {low, high};
}
{
neg.f16x2 r1254, r1253;
}
{
add.f16x2 r1256, r1003, r1091;
}
{
add.f16x2 %0, r915, r1256;
}
{
add.f16x2 r1262, r1009, r1097;
}
{
add.f16x2 %1, r921, r1262;
}
{
add.f16x2 r1268, r1003, r1091;
}
{
mul.f16x2 r1271, r1268, r1252;
}
{
add.f16x2 r1274, r915, r1271;
}
{
sub.f16x2 r1277, r1009, r1097;
}
{
mul.f16x2 r1280, r1277, r1254;
}
{
add.f16x2 %6, r1274, r1280;
}
{
add.f16x2 r1286, r1003, r1091;
}
{
mul.f16x2 r1289, r1286, r1252;
}
{
add.f16x2 r1292, r915, r1289;
}
{
sub.f16x2 r1295, r1009, r1097;
}
{
mul.f16x2 r1298, r1295, r1254;
}
{
sub.f16x2 %12, r1292, r1298;
}
{
add.f16x2 r1304, r1009, r1097;
}
{
mul.f16x2 r1307, r1304, r1252;
}
{
add.f16x2 r1310, r921, r1307;
}
{
sub.f16x2 r1313, r1003, r1091;
}
{
mul.f16x2 r1316, r1313, r1254;
}
{
sub.f16x2 %7, r1310, r1316;
}
{
add.f16x2 r1322, r1009, r1097;
}
{
mul.f16x2 r1325, r1322, r1252;
}
{
add.f16x2 r1328, r921, r1325;
}
{
sub.f16x2 r1331, r1003, r1091;
}
{
mul.f16x2 r1334, r1331, r1254;
}
{
add.f16x2 %13, r1328, r1334;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r1340, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r1341, {low, high};
}
{
neg.f16x2 r1342, r1341;
}
{
add.f16x2 r1344, r1194, r1210;
}
{
add.f16x2 %2, r939, r1344;
}
{
add.f16x2 r1350, r1200, r1216;
}
{
add.f16x2 %3, r975, r1350;
}
{
add.f16x2 r1356, r1194, r1210;
}
{
mul.f16x2 r1359, r1356, r1340;
}
{
add.f16x2 r1362, r939, r1359;
}
{
sub.f16x2 r1365, r1200, r1216;
}
{
mul.f16x2 r1368, r1365, r1342;
}
{
add.f16x2 %8, r1362, r1368;
}
{
add.f16x2 r1374, r1194, r1210;
}
{
mul.f16x2 r1377, r1374, r1340;
}
{
add.f16x2 r1380, r939, r1377;
}
{
sub.f16x2 r1383, r1200, r1216;
}
{
mul.f16x2 r1386, r1383, r1342;
}
{
sub.f16x2 %14, r1380, r1386;
}
{
add.f16x2 r1392, r1200, r1216;
}
{
mul.f16x2 r1395, r1392, r1340;
}
{
add.f16x2 r1398, r975, r1395;
}
{
sub.f16x2 r1401, r1194, r1210;
}
{
mul.f16x2 r1404, r1401, r1342;
}
{
sub.f16x2 %9, r1398, r1404;
}
{
add.f16x2 r1410, r1200, r1216;
}
{
mul.f16x2 r1413, r1410, r1340;
}
{
add.f16x2 r1416, r975, r1413;
}
{
sub.f16x2 r1419, r1194, r1210;
}
{
mul.f16x2 r1422, r1419, r1342;
}
{
add.f16x2 %15, r1416, r1422;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f130;
cvt.rn.f16.f32 high, f130;
mov.b32 r1428, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f132;
cvt.rn.f16.f32 high, f132;
mov.b32 r1429, {low, high};
}
{
neg.f16x2 r1430, r1429;
}
{
add.f16x2 r1432, r1226, r1242;
}
{
add.f16x2 %4, r957, r1432;
}
{
add.f16x2 r1438, r1232, r1248;
}
{
add.f16x2 %5, r993, r1438;
}
{
add.f16x2 r1444, r1226, r1242;
}
{
mul.f16x2 r1447, r1444, r1428;
}
{
add.f16x2 r1450, r957, r1447;
}
{
sub.f16x2 r1453, r1232, r1248;
}
{
mul.f16x2 r1456, r1453, r1430;
}
{
add.f16x2 %10, r1450, r1456;
}
{
add.f16x2 r1462, r1226, r1242;
}
{
mul.f16x2 r1465, r1462, r1428;
}
{
add.f16x2 r1468, r957, r1465;
}
{
sub.f16x2 r1471, r1232, r1248;
}
{
mul.f16x2 r1474, r1471, r1430;
}
{
sub.f16x2 %16, r1468, r1474;
}
{
add.f16x2 r1480, r1232, r1248;
}
{
mul.f16x2 r1483, r1480, r1428;
}
{
add.f16x2 r1486, r993, r1483;
}
{
sub.f16x2 r1489, r1226, r1242;
}
{
mul.f16x2 r1492, r1489, r1430;
}
{
sub.f16x2 %11, r1486, r1492;
}
{
add.f16x2 r1498, r1232, r1248;
}
{
mul.f16x2 r1501, r1498, r1428;
}
{
add.f16x2 r1504, r993, r1501;
}
{
sub.f16x2 r1507, r1226, r1242;
}
{
mul.f16x2 r1510, r1507, r1430;
}
{
add.f16x2 %17, r1504, r1510;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<875, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<404>;
.reg .b32 r<4748>;
.reg .b64 rd<4>;
mov.u32 r4746, %tid.y;
mov.u32 r4747, %54;
mad.lo.s32 r4684, r4746, 648, r4747;
mov.u32 r4685, %tid.x;
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
neg.f16x2 r3, r2;
}
{
add.f16x2 r5, %74, %58;
}
{
add.f16x2 r8, %83, r5;
}
{
add.f16x2 r11, %91, %73;
}
{
add.f16x2 r14, %97, r11;
}
{
add.f16x2 r17, %74, %58;
}
{
mul.f16x2 r20, r17, r1;
}
{
add.f16x2 r23, %83, r20;
}
{
sub.f16x2 r26, %91, %73;
}
{
mul.f16x2 r29, r26, r3;
}
{
add.f16x2 r32, r23, r29;
}
{
add.f16x2 r35, %74, %58;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %83, r38;
}
{
sub.f16x2 r44, %91, %73;
}
{
mul.f16x2 r47, r44, r3;
}
{
sub.f16x2 r50, r41, r47;
}
{
add.f16x2 r53, %91, %73;
}
{
mul.f16x2 r56, r53, r1;
}
{
add.f16x2 r59, %97, r56;
}
{
sub.f16x2 r62, %74, %58;
}
{
mul.f16x2 r65, r62, r3;
}
{
sub.f16x2 r68, r59, r65;
}
{
add.f16x2 r71, %91, %73;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %97, r74;
}
{
sub.f16x2 r80, %74, %58;
}
{
mul.f16x2 r83, r80, r3;
}
{
add.f16x2 r86, r77, r83;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r90, {low, high};
}
{
neg.f16x2 r91, r90;
}
{
add.f16x2 r93, %98, %81;
}
{
add.f16x2 r96, %106, r93;
}
{
add.f16x2 r99, %57, %95;
}
{
add.f16x2 r102, %63, r99;
}
{
add.f16x2 r105, %98, %81;
}
{
mul.f16x2 r108, r105, r89;
}
{
add.f16x2 r111, %106, r108;
}
{
sub.f16x2 r114, %57, %95;
}
{
mul.f16x2 r117, r114, r91;
}
{
add.f16x2 r120, r111, r117;
}
{
add.f16x2 r123, %98, %81;
}
{
mul.f16x2 r126, r123, r89;
}
{
add.f16x2 r129, %106, r126;
}
{
sub.f16x2 r132, %57, %95;
}
{
mul.f16x2 r135, r132, r91;
}
{
sub.f16x2 r138, r129, r135;
}
{
add.f16x2 r141, %57, %95;
}
{
mul.f16x2 r144, r141, r89;
}
{
add.f16x2 r147, %63, r144;
}
{
sub.f16x2 r150, %98, %81;
}
{
mul.f16x2 r153, r150, r91;
}
{
sub.f16x2 r156, r147, r153;
}
{
add.f16x2 r159, %57, %95;
}
{
mul.f16x2 r162, r159, r89;
}
{
add.f16x2 r165, %63, r162;
}
{
sub.f16x2 r168, %98, %81;
}
{
mul.f16x2 r171, r168, r91;
}
{
add.f16x2 r174, r165, r171;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r177, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r178, {low, high};
}
{
neg.f16x2 r179, r178;
}
{
add.f16x2 r181, %64, %103;
}
{
add.f16x2 r184, %72, r181;
}
{
add.f16x2 r187, %77, %61;
}
{
add.f16x2 r190, %88, r187;
}
{
add.f16x2 r193, %64, %103;
}
{
mul.f16x2 r196, r193, r177;
}
{
add.f16x2 r199, %72, r196;
}
{
sub.f16x2 r202, %77, %61;
}
{
mul.f16x2 r205, r202, r179;
}
{
add.f16x2 r208, r199, r205;
}
{
add.f16x2 r211, %64, %103;
}
{
mul.f16x2 r214, r211, r177;
}
{
add.f16x2 r217, %72, r214;
}
{
sub.f16x2 r220, %77, %61;
}
{
mul.f16x2 r223, r220, r179;
}
{
sub.f16x2 r226, r217, r223;
}
{
add.f16x2 r229, %77, %61;
}
{
mul.f16x2 r232, r229, r177;
}
{
add.f16x2 r235, %88, r232;
}
{
sub.f16x2 r238, %64, %103;
}
{
mul.f16x2 r241, r238, r179;
}
{
sub.f16x2 r244, r235, r241;
}
{
add.f16x2 r247, %77, %61;
}
{
mul.f16x2 r250, r247, r177;
}
{
add.f16x2 r253, %88, r250;
}
{
sub.f16x2 r256, %64, %103;
}
{
mul.f16x2 r259, r256, r179;
}
{
add.f16x2 r262, r253, r259;
}
mov.f32 f178, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r265, {low, high};
}
mov.f32 f180, 0fBF248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
mov.b32 r266, {low, high};
}
mov.f32 f190, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r267, {low, high};
}
mov.f32 f192, 0fBF7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f192;
cvt.rn.f16.f32 high, f192;
mov.b32 r268, {low, high};
}
mov.f32 f214, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r271, {low, high};
}
mov.f32 f216, 0fBEAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r272, {low, high};
}
{
mul.f16x2 r281, r120, r265;
}
{
mul.f16x2 r284, r156, r266;
}
{
sub.f16x2 r287, r281, r284;
}
{
mul.f16x2 r290, r120, r266;
}
{
fma.rn.f16x2 r293, r156, r265, r290;
}
{
mul.f16x2 r297, r208, r267;
}
{
mul.f16x2 r300, r244, r268;
}
{
sub.f16x2 r303, r297, r300;
}
{
mul.f16x2 r306, r208, r268;
}
{
fma.rn.f16x2 r309, r244, r267, r306;
}
{
mul.f16x2 r313, r138, r267;
}
{
mul.f16x2 r316, r174, r268;
}
{
sub.f16x2 r319, r313, r316;
}
{
mul.f16x2 r322, r138, r268;
}
{
fma.rn.f16x2 r325, r174, r267, r322;
}
{
mul.f16x2 r329, r226, r271;
}
{
mul.f16x2 r332, r262, r272;
}
{
sub.f16x2 r335, r329, r332;
}
{
mul.f16x2 r338, r226, r272;
}
{
fma.rn.f16x2 r341, r262, r271, r338;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r345, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r346, {low, high};
}
{
neg.f16x2 r347, r346;
}
{
add.f16x2 r349, r96, r184;
}
{
add.f16x2 r352, r8, r349;
}
{
add.f16x2 r355, r102, r190;
}
{
add.f16x2 r358, r14, r355;
}
{
add.f16x2 r361, r96, r184;
}
{
mul.f16x2 r364, r361, r345;
}
{
add.f16x2 r367, r8, r364;
}
{
sub.f16x2 r370, r102, r190;
}
{
mul.f16x2 r373, r370, r347;
}
{
add.f16x2 r376, r367, r373;
}
{
add.f16x2 r379, r96, r184;
}
{
mul.f16x2 r382, r379, r345;
}
{
add.f16x2 r385, r8, r382;
}
{
sub.f16x2 r388, r102, r190;
}
{
mul.f16x2 r391, r388, r347;
}
{
sub.f16x2 r394, r385, r391;
}
{
add.f16x2 r397, r102, r190;
}
{
mul.f16x2 r400, r397, r345;
}
{
add.f16x2 r403, r14, r400;
}
{
sub.f16x2 r406, r96, r184;
}
{
mul.f16x2 r409, r406, r347;
}
{
sub.f16x2 r412, r403, r409;
}
{
add.f16x2 r415, r102, r190;
}
{
mul.f16x2 r418, r415, r345;
}
{
add.f16x2 r421, r14, r418;
}
{
sub.f16x2 r424, r96, r184;
}
{
mul.f16x2 r427, r424, r347;
}
{
add.f16x2 r430, r421, r427;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r433, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r434, {low, high};
}
{
neg.f16x2 r435, r434;
}
{
add.f16x2 r437, r287, r303;
}
{
add.f16x2 r440, r32, r437;
}
{
add.f16x2 r443, r293, r309;
}
{
add.f16x2 r446, r68, r443;
}
{
add.f16x2 r449, r287, r303;
}
{
mul.f16x2 r452, r449, r433;
}
{
add.f16x2 r455, r32, r452;
}
{
sub.f16x2 r458, r293, r309;
}
{
mul.f16x2 r461, r458, r435;
}
{
add.f16x2 r464, r455, r461;
}
{
add.f16x2 r467, r287, r303;
}
{
mul.f16x2 r470, r467, r433;
}
{
add.f16x2 r473, r32, r470;
}
{
sub.f16x2 r476, r293, r309;
}
{
mul.f16x2 r479, r476, r435;
}
{
sub.f16x2 r482, r473, r479;
}
{
add.f16x2 r485, r293, r309;
}
{
mul.f16x2 r488, r485, r433;
}
{
add.f16x2 r491, r68, r488;
}
{
sub.f16x2 r494, r287, r303;
}
{
mul.f16x2 r497, r494, r435;
}
{
sub.f16x2 r500, r491, r497;
}
{
add.f16x2 r503, r293, r309;
}
{
mul.f16x2 r506, r503, r433;
}
{
add.f16x2 r509, r68, r506;
}
{
sub.f16x2 r512, r287, r303;
}
{
mul.f16x2 r515, r512, r435;
}
{
add.f16x2 r518, r509, r515;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r521, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r522, {low, high};
}
{
neg.f16x2 r523, r522;
}
{
add.f16x2 r525, r319, r335;
}
{
add.f16x2 r528, r50, r525;
}
{
add.f16x2 r531, r325, r341;
}
{
add.f16x2 r534, r86, r531;
}
{
add.f16x2 r537, r319, r335;
}
{
mul.f16x2 r540, r537, r521;
}
{
add.f16x2 r543, r50, r540;
}
{
sub.f16x2 r546, r325, r341;
}
{
mul.f16x2 r549, r546, r523;
}
{
add.f16x2 r552, r543, r549;
}
{
add.f16x2 r555, r319, r335;
}
{
mul.f16x2 r558, r555, r521;
}
{
add.f16x2 r561, r50, r558;
}
{
sub.f16x2 r564, r325, r341;
}
{
mul.f16x2 r567, r564, r523;
}
{
sub.f16x2 r570, r561, r567;
}
{
add.f16x2 r573, r325, r341;
}
{
mul.f16x2 r576, r573, r521;
}
{
add.f16x2 r579, r86, r576;
}
{
sub.f16x2 r582, r319, r335;
}
{
mul.f16x2 r585, r582, r523;
}
{
sub.f16x2 r588, r579, r585;
}
{
add.f16x2 r591, r325, r341;
}
{
mul.f16x2 r594, r591, r521;
}
{
add.f16x2 r597, r86, r594;
}
{
sub.f16x2 r600, r319, r335;
}
{
mul.f16x2 r603, r600, r523;
}
{
add.f16x2 r606, r597, r603;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r609, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r610, {low, high};
}
{
neg.f16x2 r611, r610;
}
{
add.f16x2 r613, %78, %60;
}
{
add.f16x2 r616, %86, r613;
}
{
add.f16x2 r619, %93, %76;
}
{
add.f16x2 r622, %100, r619;
}
{
add.f16x2 r625, %78, %60;
}
{
mul.f16x2 r628, r625, r609;
}
{
add.f16x2 r631, %86, r628;
}
{
sub.f16x2 r634, %93, %76;
}
{
mul.f16x2 r637, r634, r611;
}
{
add.f16x2 r640, r631, r637;
}
{
add.f16x2 r643, %78, %60;
}
{
mul.f16x2 r646, r643, r609;
}
{
add.f16x2 r649, %86, r646;
}
{
sub.f16x2 r652, %93, %76;
}
{
mul.f16x2 r655, r652, r611;
}
{
sub.f16x2 r658, r649, r655;
}
{
add.f16x2 r661, %93, %76;
}
{
mul.f16x2 r664, r661, r609;
}
{
add.f16x2 r667, %100, r664;
}
{
sub.f16x2 r670, %78, %60;
}
{
mul.f16x2 r673, r670, r611;
}
{
sub.f16x2 r676, r667, r673;
}
{
add.f16x2 r679, %93, %76;
}
{
mul.f16x2 r682, r679, r609;
}
{
add.f16x2 r685, %100, r682;
}
{
sub.f16x2 r688, %78, %60;
}
{
mul.f16x2 r691, r688, r611;
}
{
add.f16x2 r694, r685, r691;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r697, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r698, {low, high};
}
{
neg.f16x2 r699, r698;
}
{
add.f16x2 r701, %101, %85;
}
{
add.f16x2 r704, %108, r701;
}
{
add.f16x2 r707, %59, %99;
}
{
add.f16x2 r710, %68, r707;
}
{
add.f16x2 r713, %101, %85;
}
{
mul.f16x2 r716, r713, r697;
}
{
add.f16x2 r719, %108, r716;
}
{
sub.f16x2 r722, %59, %99;
}
{
mul.f16x2 r725, r722, r699;
}
{
add.f16x2 r728, r719, r725;
}
{
add.f16x2 r731, %101, %85;
}
{
mul.f16x2 r734, r731, r697;
}
{
add.f16x2 r737, %108, r734;
}
{
sub.f16x2 r740, %59, %99;
}
{
mul.f16x2 r743, r740, r699;
}
{
sub.f16x2 r746, r737, r743;
}
{
add.f16x2 r749, %59, %99;
}
{
mul.f16x2 r752, r749, r697;
}
{
add.f16x2 r755, %68, r752;
}
{
sub.f16x2 r758, %101, %85;
}
{
mul.f16x2 r761, r758, r699;
}
{
sub.f16x2 r764, r755, r761;
}
{
add.f16x2 r767, %59, %99;
}
{
mul.f16x2 r770, r767, r697;
}
{
add.f16x2 r773, %68, r770;
}
{
sub.f16x2 r776, %101, %85;
}
{
mul.f16x2 r779, r776, r699;
}
{
add.f16x2 r782, r773, r779;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r785, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r786, {low, high};
}
{
neg.f16x2 r787, r786;
}
{
add.f16x2 r789, %67, %107;
}
{
add.f16x2 r792, %75, r789;
}
{
add.f16x2 r795, %84, %66;
}
{
add.f16x2 r798, %92, r795;
}
{
add.f16x2 r801, %67, %107;
}
{
mul.f16x2 r804, r801, r785;
}
{
add.f16x2 r807, %75, r804;
}
{
sub.f16x2 r810, %84, %66;
}
{
mul.f16x2 r813, r810, r787;
}
{
add.f16x2 r816, r807, r813;
}
{
add.f16x2 r819, %67, %107;
}
{
mul.f16x2 r822, r819, r785;
}
{
add.f16x2 r825, %75, r822;
}
{
sub.f16x2 r828, %84, %66;
}
{
mul.f16x2 r831, r828, r787;
}
{
sub.f16x2 r834, r825, r831;
}
{
add.f16x2 r837, %84, %66;
}
{
mul.f16x2 r840, r837, r785;
}
{
add.f16x2 r843, %92, r840;
}
{
sub.f16x2 r846, %67, %107;
}
{
mul.f16x2 r849, r846, r787;
}
{
sub.f16x2 r852, r843, r849;
}
{
add.f16x2 r855, %84, %66;
}
{
mul.f16x2 r858, r855, r785;
}
{
add.f16x2 r861, %92, r858;
}
{
sub.f16x2 r864, %67, %107;
}
{
mul.f16x2 r867, r864, r787;
}
{
add.f16x2 r870, r861, r867;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r873, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
mov.b32 r874, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r875, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f192;
cvt.rn.f16.f32 high, f192;
mov.b32 r876, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r879, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r880, {low, high};
}
{
mul.f16x2 r889, r728, r873;
}
{
mul.f16x2 r892, r764, r874;
}
{
sub.f16x2 r895, r889, r892;
}
{
mul.f16x2 r898, r728, r874;
}
{
fma.rn.f16x2 r901, r764, r873, r898;
}
{
mul.f16x2 r905, r816, r875;
}
{
mul.f16x2 r908, r852, r876;
}
{
sub.f16x2 r911, r905, r908;
}
{
mul.f16x2 r914, r816, r876;
}
{
fma.rn.f16x2 r917, r852, r875, r914;
}
{
mul.f16x2 r921, r746, r875;
}
{
mul.f16x2 r924, r782, r876;
}
{
sub.f16x2 r927, r921, r924;
}
{
mul.f16x2 r930, r746, r876;
}
{
fma.rn.f16x2 r933, r782, r875, r930;
}
{
mul.f16x2 r937, r834, r879;
}
{
mul.f16x2 r940, r870, r880;
}
{
sub.f16x2 r943, r937, r940;
}
{
mul.f16x2 r946, r834, r880;
}
{
fma.rn.f16x2 r949, r870, r879, r946;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r953, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r954, {low, high};
}
{
neg.f16x2 r955, r954;
}
{
add.f16x2 r957, r704, r792;
}
{
add.f16x2 r960, r616, r957;
}
{
add.f16x2 r963, r710, r798;
}
{
add.f16x2 r966, r622, r963;
}
{
add.f16x2 r969, r704, r792;
}
{
mul.f16x2 r972, r969, r953;
}
{
add.f16x2 r975, r616, r972;
}
{
sub.f16x2 r978, r710, r798;
}
{
mul.f16x2 r981, r978, r955;
}
{
add.f16x2 r984, r975, r981;
}
{
add.f16x2 r987, r704, r792;
}
{
mul.f16x2 r990, r987, r953;
}
{
add.f16x2 r993, r616, r990;
}
{
sub.f16x2 r996, r710, r798;
}
{
mul.f16x2 r999, r996, r955;
}
{
sub.f16x2 r1002, r993, r999;
}
{
add.f16x2 r1005, r710, r798;
}
{
mul.f16x2 r1008, r1005, r953;
}
{
add.f16x2 r1011, r622, r1008;
}
{
sub.f16x2 r1014, r704, r792;
}
{
mul.f16x2 r1017, r1014, r955;
}
{
sub.f16x2 r1020, r1011, r1017;
}
{
add.f16x2 r1023, r710, r798;
}
{
mul.f16x2 r1026, r1023, r953;
}
{
add.f16x2 r1029, r622, r1026;
}
{
sub.f16x2 r1032, r704, r792;
}
{
mul.f16x2 r1035, r1032, r955;
}
{
add.f16x2 r1038, r1029, r1035;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1041, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1042, {low, high};
}
{
neg.f16x2 r1043, r1042;
}
{
add.f16x2 r1045, r895, r911;
}
{
add.f16x2 r1048, r640, r1045;
}
{
add.f16x2 r1051, r901, r917;
}
{
add.f16x2 r1054, r676, r1051;
}
{
add.f16x2 r1057, r895, r911;
}
{
mul.f16x2 r1060, r1057, r1041;
}
{
add.f16x2 r1063, r640, r1060;
}
{
sub.f16x2 r1066, r901, r917;
}
{
mul.f16x2 r1069, r1066, r1043;
}
{
add.f16x2 r1072, r1063, r1069;
}
{
add.f16x2 r1075, r895, r911;
}
{
mul.f16x2 r1078, r1075, r1041;
}
{
add.f16x2 r1081, r640, r1078;
}
{
sub.f16x2 r1084, r901, r917;
}
{
mul.f16x2 r1087, r1084, r1043;
}
{
sub.f16x2 r1090, r1081, r1087;
}
{
add.f16x2 r1093, r901, r917;
}
{
mul.f16x2 r1096, r1093, r1041;
}
{
add.f16x2 r1099, r676, r1096;
}
{
sub.f16x2 r1102, r895, r911;
}
{
mul.f16x2 r1105, r1102, r1043;
}
{
sub.f16x2 r1108, r1099, r1105;
}
{
add.f16x2 r1111, r901, r917;
}
{
mul.f16x2 r1114, r1111, r1041;
}
{
add.f16x2 r1117, r676, r1114;
}
{
sub.f16x2 r1120, r895, r911;
}
{
mul.f16x2 r1123, r1120, r1043;
}
{
add.f16x2 r1126, r1117, r1123;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1129, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1130, {low, high};
}
{
neg.f16x2 r1131, r1130;
}
{
add.f16x2 r1133, r927, r943;
}
{
add.f16x2 r1136, r658, r1133;
}
{
add.f16x2 r1139, r933, r949;
}
{
add.f16x2 r1142, r694, r1139;
}
{
add.f16x2 r1145, r927, r943;
}
{
mul.f16x2 r1148, r1145, r1129;
}
{
add.f16x2 r1151, r658, r1148;
}
{
sub.f16x2 r1154, r933, r949;
}
{
mul.f16x2 r1157, r1154, r1131;
}
{
add.f16x2 r1160, r1151, r1157;
}
{
add.f16x2 r1163, r927, r943;
}
{
mul.f16x2 r1166, r1163, r1129;
}
{
add.f16x2 r1169, r658, r1166;
}
{
sub.f16x2 r1172, r933, r949;
}
{
mul.f16x2 r1175, r1172, r1131;
}
{
sub.f16x2 r1178, r1169, r1175;
}
{
add.f16x2 r1181, r933, r949;
}
{
mul.f16x2 r1184, r1181, r1129;
}
{
add.f16x2 r1187, r694, r1184;
}
{
sub.f16x2 r1190, r927, r943;
}
{
mul.f16x2 r1193, r1190, r1131;
}
{
sub.f16x2 r1196, r1187, r1193;
}
{
add.f16x2 r1199, r933, r949;
}
{
mul.f16x2 r1202, r1199, r1129;
}
{
add.f16x2 r1205, r694, r1202;
}
{
sub.f16x2 r1208, r927, r943;
}
{
mul.f16x2 r1211, r1208, r1131;
}
{
add.f16x2 r1214, r1205, r1211;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1217, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1218, {low, high};
}
{
neg.f16x2 r1219, r1218;
}
{
add.f16x2 r1221, %82, %65;
}
{
add.f16x2 r1224, %90, r1221;
}
{
add.f16x2 r1227, %96, %80;
}
{
add.f16x2 r1230, %104, r1227;
}
{
add.f16x2 r1233, %82, %65;
}
{
mul.f16x2 r1236, r1233, r1217;
}
{
add.f16x2 r1239, %90, r1236;
}
{
sub.f16x2 r1242, %96, %80;
}
{
mul.f16x2 r1245, r1242, r1219;
}
{
add.f16x2 r1248, r1239, r1245;
}
{
add.f16x2 r1251, %82, %65;
}
{
mul.f16x2 r1254, r1251, r1217;
}
{
add.f16x2 r1257, %90, r1254;
}
{
sub.f16x2 r1260, %96, %80;
}
{
mul.f16x2 r1263, r1260, r1219;
}
{
sub.f16x2 r1266, r1257, r1263;
}
{
add.f16x2 r1269, %96, %80;
}
{
mul.f16x2 r1272, r1269, r1217;
}
{
add.f16x2 r1275, %104, r1272;
}
{
sub.f16x2 r1278, %82, %65;
}
{
mul.f16x2 r1281, r1278, r1219;
}
{
sub.f16x2 r1284, r1275, r1281;
}
{
add.f16x2 r1287, %96, %80;
}
{
mul.f16x2 r1290, r1287, r1217;
}
{
add.f16x2 r1293, %104, r1290;
}
{
sub.f16x2 r1296, %82, %65;
}
{
mul.f16x2 r1299, r1296, r1219;
}
{
add.f16x2 r1302, r1293, r1299;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1305, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1306, {low, high};
}
{
neg.f16x2 r1307, r1306;
}
{
add.f16x2 r1309, %105, %89;
}
{
add.f16x2 r1312, %56, r1309;
}
{
add.f16x2 r1315, %62, %102;
}
{
add.f16x2 r1318, %70, r1315;
}
{
add.f16x2 r1321, %105, %89;
}
{
mul.f16x2 r1324, r1321, r1305;
}
{
add.f16x2 r1327, %56, r1324;
}
{
sub.f16x2 r1330, %62, %102;
}
{
mul.f16x2 r1333, r1330, r1307;
}
{
add.f16x2 r1336, r1327, r1333;
}
{
add.f16x2 r1339, %105, %89;
}
{
mul.f16x2 r1342, r1339, r1305;
}
{
add.f16x2 r1345, %56, r1342;
}
{
sub.f16x2 r1348, %62, %102;
}
{
mul.f16x2 r1351, r1348, r1307;
}
{
sub.f16x2 r1354, r1345, r1351;
}
{
add.f16x2 r1357, %62, %102;
}
{
mul.f16x2 r1360, r1357, r1305;
}
{
add.f16x2 r1363, %70, r1360;
}
{
sub.f16x2 r1366, %105, %89;
}
{
mul.f16x2 r1369, r1366, r1307;
}
{
sub.f16x2 r1372, r1363, r1369;
}
{
add.f16x2 r1375, %62, %102;
}
{
mul.f16x2 r1378, r1375, r1305;
}
{
add.f16x2 r1381, %70, r1378;
}
{
sub.f16x2 r1384, %105, %89;
}
{
mul.f16x2 r1387, r1384, r1307;
}
{
add.f16x2 r1390, r1381, r1387;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1393, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1394, {low, high};
}
{
neg.f16x2 r1395, r1394;
}
{
add.f16x2 r1397, %71, %55;
}
{
add.f16x2 r1400, %79, r1397;
}
{
add.f16x2 r1403, %87, %69;
}
{
add.f16x2 r1406, %94, r1403;
}
{
add.f16x2 r1409, %71, %55;
}
{
mul.f16x2 r1412, r1409, r1393;
}
{
add.f16x2 r1415, %79, r1412;
}
{
sub.f16x2 r1418, %87, %69;
}
{
mul.f16x2 r1421, r1418, r1395;
}
{
add.f16x2 r1424, r1415, r1421;
}
{
add.f16x2 r1427, %71, %55;
}
{
mul.f16x2 r1430, r1427, r1393;
}
{
add.f16x2 r1433, %79, r1430;
}
{
sub.f16x2 r1436, %87, %69;
}
{
mul.f16x2 r1439, r1436, r1395;
}
{
sub.f16x2 r1442, r1433, r1439;
}
{
add.f16x2 r1445, %87, %69;
}
{
mul.f16x2 r1448, r1445, r1393;
}
{
add.f16x2 r1451, %94, r1448;
}
{
sub.f16x2 r1454, %71, %55;
}
{
mul.f16x2 r1457, r1454, r1395;
}
{
sub.f16x2 r1460, r1451, r1457;
}
{
add.f16x2 r1463, %87, %69;
}
{
mul.f16x2 r1466, r1463, r1393;
}
{
add.f16x2 r1469, %94, r1466;
}
{
sub.f16x2 r1472, %71, %55;
}
{
mul.f16x2 r1475, r1472, r1395;
}
{
add.f16x2 r1478, r1469, r1475;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r1481, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
mov.b32 r1482, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r1483, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f192;
cvt.rn.f16.f32 high, f192;
mov.b32 r1484, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1487, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1488, {low, high};
}
{
mul.f16x2 r1497, r1336, r1481;
}
{
mul.f16x2 r1500, r1372, r1482;
}
{
sub.f16x2 r1503, r1497, r1500;
}
{
mul.f16x2 r1506, r1336, r1482;
}
{
fma.rn.f16x2 r1509, r1372, r1481, r1506;
}
{
mul.f16x2 r1513, r1424, r1483;
}
{
mul.f16x2 r1516, r1460, r1484;
}
{
sub.f16x2 r1519, r1513, r1516;
}
{
mul.f16x2 r1522, r1424, r1484;
}
{
fma.rn.f16x2 r1525, r1460, r1483, r1522;
}
{
mul.f16x2 r1529, r1354, r1483;
}
{
mul.f16x2 r1532, r1390, r1484;
}
{
sub.f16x2 r1535, r1529, r1532;
}
{
mul.f16x2 r1538, r1354, r1484;
}
{
fma.rn.f16x2 r1541, r1390, r1483, r1538;
}
{
mul.f16x2 r1545, r1442, r1487;
}
{
mul.f16x2 r1548, r1478, r1488;
}
{
sub.f16x2 r1551, r1545, r1548;
}
{
mul.f16x2 r1554, r1442, r1488;
}
{
fma.rn.f16x2 r1557, r1478, r1487, r1554;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1561, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1562, {low, high};
}
{
neg.f16x2 r1563, r1562;
}
{
add.f16x2 r1565, r1312, r1400;
}
{
add.f16x2 r1568, r1224, r1565;
}
{
add.f16x2 r1571, r1318, r1406;
}
{
add.f16x2 r1574, r1230, r1571;
}
{
add.f16x2 r1577, r1312, r1400;
}
{
mul.f16x2 r1580, r1577, r1561;
}
{
add.f16x2 r1583, r1224, r1580;
}
{
sub.f16x2 r1586, r1318, r1406;
}
{
mul.f16x2 r1589, r1586, r1563;
}
{
add.f16x2 r1592, r1583, r1589;
}
{
add.f16x2 r1595, r1312, r1400;
}
{
mul.f16x2 r1598, r1595, r1561;
}
{
add.f16x2 r1601, r1224, r1598;
}
{
sub.f16x2 r1604, r1318, r1406;
}
{
mul.f16x2 r1607, r1604, r1563;
}
{
sub.f16x2 r1610, r1601, r1607;
}
{
add.f16x2 r1613, r1318, r1406;
}
{
mul.f16x2 r1616, r1613, r1561;
}
{
add.f16x2 r1619, r1230, r1616;
}
{
sub.f16x2 r1622, r1312, r1400;
}
{
mul.f16x2 r1625, r1622, r1563;
}
{
sub.f16x2 r1628, r1619, r1625;
}
{
add.f16x2 r1631, r1318, r1406;
}
{
mul.f16x2 r1634, r1631, r1561;
}
{
add.f16x2 r1637, r1230, r1634;
}
{
sub.f16x2 r1640, r1312, r1400;
}
{
mul.f16x2 r1643, r1640, r1563;
}
{
add.f16x2 r1646, r1637, r1643;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1649, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1650, {low, high};
}
{
neg.f16x2 r1651, r1650;
}
{
add.f16x2 r1653, r1503, r1519;
}
{
add.f16x2 r1656, r1248, r1653;
}
{
add.f16x2 r1659, r1509, r1525;
}
{
add.f16x2 r1662, r1284, r1659;
}
{
add.f16x2 r1665, r1503, r1519;
}
{
mul.f16x2 r1668, r1665, r1649;
}
{
add.f16x2 r1671, r1248, r1668;
}
{
sub.f16x2 r1674, r1509, r1525;
}
{
mul.f16x2 r1677, r1674, r1651;
}
{
add.f16x2 r1680, r1671, r1677;
}
{
add.f16x2 r1683, r1503, r1519;
}
{
mul.f16x2 r1686, r1683, r1649;
}
{
add.f16x2 r1689, r1248, r1686;
}
{
sub.f16x2 r1692, r1509, r1525;
}
{
mul.f16x2 r1695, r1692, r1651;
}
{
sub.f16x2 r1698, r1689, r1695;
}
{
add.f16x2 r1701, r1509, r1525;
}
{
mul.f16x2 r1704, r1701, r1649;
}
{
add.f16x2 r1707, r1284, r1704;
}
{
sub.f16x2 r1710, r1503, r1519;
}
{
mul.f16x2 r1713, r1710, r1651;
}
{
sub.f16x2 r1716, r1707, r1713;
}
{
add.f16x2 r1719, r1509, r1525;
}
{
mul.f16x2 r1722, r1719, r1649;
}
{
add.f16x2 r1725, r1284, r1722;
}
{
sub.f16x2 r1728, r1503, r1519;
}
{
mul.f16x2 r1731, r1728, r1651;
}
{
add.f16x2 r1734, r1725, r1731;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1737, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1738, {low, high};
}
{
neg.f16x2 r1739, r1738;
}
{
add.f16x2 r1741, r1535, r1551;
}
{
add.f16x2 r1744, r1266, r1741;
}
{
add.f16x2 r1747, r1541, r1557;
}
{
add.f16x2 r1750, r1302, r1747;
}
{
add.f16x2 r1753, r1535, r1551;
}
{
mul.f16x2 r1756, r1753, r1737;
}
{
add.f16x2 r1759, r1266, r1756;
}
{
sub.f16x2 r1762, r1541, r1557;
}
{
mul.f16x2 r1765, r1762, r1739;
}
{
add.f16x2 r1768, r1759, r1765;
}
{
add.f16x2 r1771, r1535, r1551;
}
{
mul.f16x2 r1774, r1771, r1737;
}
{
add.f16x2 r1777, r1266, r1774;
}
{
sub.f16x2 r1780, r1541, r1557;
}
{
mul.f16x2 r1783, r1780, r1739;
}
{
sub.f16x2 r1786, r1777, r1783;
}
{
add.f16x2 r1789, r1541, r1557;
}
{
mul.f16x2 r1792, r1789, r1737;
}
{
add.f16x2 r1795, r1302, r1792;
}
{
sub.f16x2 r1798, r1535, r1551;
}
{
mul.f16x2 r1801, r1798, r1739;
}
{
sub.f16x2 r1804, r1795, r1801;
}
{
add.f16x2 r1807, r1541, r1557;
}
{
mul.f16x2 r1810, r1807, r1737;
}
{
add.f16x2 r1813, r1302, r1810;
}
{
sub.f16x2 r1816, r1535, r1551;
}
{
mul.f16x2 r1819, r1816, r1739;
}
{
add.f16x2 r1822, r1813, r1819;
}
mov.f32 f170, 0f3F791978;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f170;
cvt.rn.f16.f32 high, f170;
mov.b32 r1825, {low, high};
}
mov.f32 f172, 0fBE6C2691;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f172;
cvt.rn.f16.f32 high, f172;
mov.b32 r1826, {low, high};
}
mov.f32 f174, 0f3F64C51C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f174;
cvt.rn.f16.f32 high, f174;
mov.b32 r1827, {low, high};
}
mov.f32 f176, 0fBEE5C902;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f176;
cvt.rn.f16.f32 high, f176;
mov.b32 r1828, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r1829, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
mov.b32 r1830, {low, high};
}
mov.f32 f182, 0f3F18DF63;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f182;
cvt.rn.f16.f32 high, f182;
mov.b32 r1831, {low, high};
}
mov.f32 f184, 0fBF4D57F2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f184;
cvt.rn.f16.f32 high, f184;
mov.b32 r1832, {low, high};
}
mov.f32 f186, 0f3ECACAF8;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f186;
cvt.rn.f16.f32 high, f186;
mov.b32 r1833, {low, high};
}
mov.f32 f188, 0fBF6B1036;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r1834, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r1835, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f192;
cvt.rn.f16.f32 high, f192;
mov.b32 r1836, {low, high};
}
mov.f32 f194, 0fBD6E2946;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1837, {low, high};
}
mov.f32 f196, 0fBF7F9120;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r1838, {low, high};
}
mov.f32 f198, 0fBE92D7E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1839, {low, high};
}
mov.f32 f200, 0fBF753ECD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1840, {low, high};
}
mov.f32 f206, 0fBF2FAD88;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f206;
cvt.rn.f16.f32 high, f206;
mov.b32 r1843, {low, high};
}
mov.f32 f208, 0fBF3A3529;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f208;
cvt.rn.f16.f32 high, f208;
mov.b32 r1844, {low, high};
}
mov.f32 f230, 0fBF55E287;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1847, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1848, {low, high};
}
mov.f32 f222, 0fBF7E44DE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f222;
cvt.rn.f16.f32 high, f222;
mov.b32 r1851, {low, high};
}
mov.f32 f224, 0f3DEDC21F;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f224;
cvt.rn.f16.f32 high, f224;
mov.b32 r1852, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r1855, {low, high};
}
mov.f32 f232, 0f3F0CAC9F;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f232;
cvt.rn.f16.f32 high, f232;
mov.b32 r1856, {low, high};
}
{
mul.f16x2 r1877, r1048, r1825;
}
{
mul.f16x2 r1880, r1054, r1826;
}
{
sub.f16x2 r1883, r1877, r1880;
}
{
mul.f16x2 r1886, r1048, r1826;
}
{
fma.rn.f16x2 r1889, r1054, r1825, r1886;
}
{
mul.f16x2 r1893, r1656, r1827;
}
{
mul.f16x2 r1896, r1662, r1828;
}
{
sub.f16x2 r1899, r1893, r1896;
}
{
mul.f16x2 r1902, r1656, r1828;
}
{
fma.rn.f16x2 r1905, r1662, r1827, r1902;
}
{
mul.f16x2 r1909, r1136, r1827;
}
{
mul.f16x2 r1912, r1142, r1828;
}
{
sub.f16x2 r1915, r1909, r1912;
}
{
mul.f16x2 r1918, r1136, r1828;
}
{
fma.rn.f16x2 r1921, r1142, r1827, r1918;
}
{
mul.f16x2 r1925, r1744, r1831;
}
{
mul.f16x2 r1928, r1750, r1832;
}
{
sub.f16x2 r1931, r1925, r1928;
}
{
mul.f16x2 r1934, r1744, r1832;
}
{
fma.rn.f16x2 r1937, r1750, r1831, r1934;
}
{
mul.f16x2 r1941, r984, r1829;
}
{
mul.f16x2 r1944, r1020, r1830;
}
{
sub.f16x2 r1947, r1941, r1944;
}
{
mul.f16x2 r1950, r984, r1830;
}
{
fma.rn.f16x2 r1953, r1020, r1829, r1950;
}
{
mul.f16x2 r1957, r1592, r1835;
}
{
mul.f16x2 r1960, r1628, r1836;
}
{
sub.f16x2 r1963, r1957, r1960;
}
{
mul.f16x2 r1966, r1592, r1836;
}
{
fma.rn.f16x2 r1969, r1628, r1835, r1966;
}
{
mul.f16x2 r1973, r1072, r1831;
}
{
mul.f16x2 r1976, r1108, r1832;
}
{
sub.f16x2 r1979, r1973, r1976;
}
{
mul.f16x2 r1982, r1072, r1832;
}
{
fma.rn.f16x2 r1985, r1108, r1831, r1982;
}
{
mul.f16x2 r1989, r1680, r1839;
}
{
mul.f16x2 r1992, r1716, r1840;
}
{
sub.f16x2 r1995, r1989, r1992;
}
{
mul.f16x2 r1998, r1680, r1840;
}
{
fma.rn.f16x2 r2001, r1716, r1839, r1998;
}
{
mul.f16x2 r2005, r1160, r1833;
}
{
mul.f16x2 r2008, r1196, r1834;
}
{
sub.f16x2 r2011, r2005, r2008;
}
{
mul.f16x2 r2014, r1160, r1834;
}
{
fma.rn.f16x2 r2017, r1196, r1833, r2014;
}
{
mul.f16x2 r2021, r1768, r1843;
}
{
mul.f16x2 r2024, r1804, r1844;
}
{
sub.f16x2 r2027, r2021, r2024;
}
{
mul.f16x2 r2030, r1768, r1844;
}
{
fma.rn.f16x2 r2033, r1804, r1843, r2030;
}
{
mul.f16x2 r2037, r1002, r1835;
}
{
mul.f16x2 r2040, r1038, r1836;
}
{
sub.f16x2 r2043, r2037, r2040;
}
{
mul.f16x2 r2046, r1002, r1836;
}
{
fma.rn.f16x2 r2049, r1038, r1835, r2046;
}
{
mul.f16x2 r2053, r1610, r1847;
}
{
mul.f16x2 r2056, r1646, r1848;
}
{
sub.f16x2 r2059, r2053, r2056;
}
{
mul.f16x2 r2062, r1610, r1848;
}
{
fma.rn.f16x2 r2065, r1646, r1847, r2062;
}
{
mul.f16x2 r2069, r1090, r1837;
}
{
mul.f16x2 r2072, r1126, r1838;
}
{
sub.f16x2 r2075, r2069, r2072;
}
{
mul.f16x2 r2078, r1090, r1838;
}
{
fma.rn.f16x2 r2081, r1126, r1837, r2078;
}
{
mul.f16x2 r2085, r1698, r1851;
}
{
mul.f16x2 r2088, r1734, r1852;
}
{
sub.f16x2 r2091, r2085, r2088;
}
{
mul.f16x2 r2094, r1698, r1852;
}
{
fma.rn.f16x2 r2097, r1734, r1851, r2094;
}
{
mul.f16x2 r2101, r1178, r1839;
}
{
mul.f16x2 r2104, r1214, r1840;
}
{
sub.f16x2 r2107, r2101, r2104;
}
{
mul.f16x2 r2110, r1178, r1840;
}
{
fma.rn.f16x2 r2113, r1214, r1839, r2110;
}
{
mul.f16x2 r2117, r1786, r1855;
}
{
mul.f16x2 r2120, r1822, r1856;
}
{
sub.f16x2 r2123, r2117, r2120;
}
{
mul.f16x2 r2126, r1786, r1856;
}
{
fma.rn.f16x2 r2129, r1822, r1855, r2126;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2133, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2134, {low, high};
}
{
neg.f16x2 r2135, r2134;
}
{
add.f16x2 r2137, r960, r1568;
}
{
add.f16x2 r2140, r352, r2137;
}
{
add.f16x2 r2143, r966, r1574;
}
{
add.f16x2 r2146, r358, r2143;
}
{
add.f16x2 r2149, r960, r1568;
}
{
mul.f16x2 r2152, r2149, r2133;
}
{
add.f16x2 r2155, r352, r2152;
}
{
sub.f16x2 r2158, r966, r1574;
}
{
mul.f16x2 r2161, r2158, r2135;
}
{
add.f16x2 r2164, r2155, r2161;
}
{
add.f16x2 r2167, r960, r1568;
}
{
mul.f16x2 r2170, r2167, r2133;
}
{
add.f16x2 r2173, r352, r2170;
}
{
sub.f16x2 r2176, r966, r1574;
}
{
mul.f16x2 r2179, r2176, r2135;
}
{
sub.f16x2 r2182, r2173, r2179;
}
{
add.f16x2 r2185, r966, r1574;
}
{
mul.f16x2 r2188, r2185, r2133;
}
{
add.f16x2 r2191, r358, r2188;
}
{
sub.f16x2 r2194, r960, r1568;
}
{
mul.f16x2 r2197, r2194, r2135;
}
{
sub.f16x2 r2200, r2191, r2197;
}
{
add.f16x2 r2203, r966, r1574;
}
{
mul.f16x2 r2206, r2203, r2133;
}
{
add.f16x2 r2209, r358, r2206;
}
{
sub.f16x2 r2212, r960, r1568;
}
{
mul.f16x2 r2215, r2212, r2135;
}
{
add.f16x2 r2218, r2209, r2215;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2221, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2222, {low, high};
}
{
neg.f16x2 r2223, r2222;
}
{
add.f16x2 r2225, r1883, r1899;
}
{
add.f16x2 r2228, r440, r2225;
}
{
add.f16x2 r2231, r1889, r1905;
}
{
add.f16x2 r2234, r446, r2231;
}
{
add.f16x2 r2237, r1883, r1899;
}
{
mul.f16x2 r2240, r2237, r2221;
}
{
add.f16x2 r2243, r440, r2240;
}
{
sub.f16x2 r2246, r1889, r1905;
}
{
mul.f16x2 r2249, r2246, r2223;
}
{
add.f16x2 r2252, r2243, r2249;
}
{
add.f16x2 r2255, r1883, r1899;
}
{
mul.f16x2 r2258, r2255, r2221;
}
{
add.f16x2 r2261, r440, r2258;
}
{
sub.f16x2 r2264, r1889, r1905;
}
{
mul.f16x2 r2267, r2264, r2223;
}
{
sub.f16x2 r2270, r2261, r2267;
}
{
add.f16x2 r2273, r1889, r1905;
}
{
mul.f16x2 r2276, r2273, r2221;
}
{
add.f16x2 r2279, r446, r2276;
}
{
sub.f16x2 r2282, r1883, r1899;
}
{
mul.f16x2 r2285, r2282, r2223;
}
{
sub.f16x2 r2288, r2279, r2285;
}
{
add.f16x2 r2291, r1889, r1905;
}
{
mul.f16x2 r2294, r2291, r2221;
}
{
add.f16x2 r2297, r446, r2294;
}
{
sub.f16x2 r2300, r1883, r1899;
}
{
mul.f16x2 r2303, r2300, r2223;
}
{
add.f16x2 r2306, r2297, r2303;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2309, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2310, {low, high};
}
{
neg.f16x2 r2311, r2310;
}
{
add.f16x2 r2313, r1915, r1931;
}
{
add.f16x2 r2316, r528, r2313;
}
{
add.f16x2 r2319, r1921, r1937;
}
{
add.f16x2 r2322, r534, r2319;
}
{
add.f16x2 r2325, r1915, r1931;
}
{
mul.f16x2 r2328, r2325, r2309;
}
{
add.f16x2 r2331, r528, r2328;
}
{
sub.f16x2 r2334, r1921, r1937;
}
{
mul.f16x2 r2337, r2334, r2311;
}
{
add.f16x2 r2340, r2331, r2337;
}
{
add.f16x2 r2343, r1915, r1931;
}
{
mul.f16x2 r2346, r2343, r2309;
}
{
add.f16x2 r2349, r528, r2346;
}
{
sub.f16x2 r2352, r1921, r1937;
}
{
mul.f16x2 r2355, r2352, r2311;
}
{
sub.f16x2 r2358, r2349, r2355;
}
{
add.f16x2 r2361, r1921, r1937;
}
{
mul.f16x2 r2364, r2361, r2309;
}
{
add.f16x2 r2367, r534, r2364;
}
{
sub.f16x2 r2370, r1915, r1931;
}
{
mul.f16x2 r2373, r2370, r2311;
}
{
sub.f16x2 r2376, r2367, r2373;
}
{
add.f16x2 r2379, r1921, r1937;
}
{
mul.f16x2 r2382, r2379, r2309;
}
{
add.f16x2 r2385, r534, r2382;
}
{
sub.f16x2 r2388, r1915, r1931;
}
{
mul.f16x2 r2391, r2388, r2311;
}
{
add.f16x2 r2394, r2385, r2391;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2398, {low, high};
}
{
neg.f16x2 r2399, r2398;
}
{
add.f16x2 r2401, r1947, r1963;
}
{
add.f16x2 r2404, r376, r2401;
}
{
add.f16x2 r2407, r1953, r1969;
}
{
add.f16x2 r2410, r412, r2407;
}
{
add.f16x2 r2413, r1947, r1963;
}
{
mul.f16x2 r2416, r2413, r2397;
}
{
add.f16x2 r2419, r376, r2416;
}
{
sub.f16x2 r2422, r1953, r1969;
}
{
mul.f16x2 r2425, r2422, r2399;
}
{
add.f16x2 r2428, r2419, r2425;
}
{
add.f16x2 r2431, r1947, r1963;
}
{
mul.f16x2 r2434, r2431, r2397;
}
{
add.f16x2 r2437, r376, r2434;
}
{
sub.f16x2 r2440, r1953, r1969;
}
{
mul.f16x2 r2443, r2440, r2399;
}
{
sub.f16x2 r2446, r2437, r2443;
}
{
add.f16x2 r2449, r1953, r1969;
}
{
mul.f16x2 r2452, r2449, r2397;
}
{
add.f16x2 r2455, r412, r2452;
}
{
sub.f16x2 r2458, r1947, r1963;
}
{
mul.f16x2 r2461, r2458, r2399;
}
{
sub.f16x2 r2464, r2455, r2461;
}
{
add.f16x2 r2467, r1953, r1969;
}
{
mul.f16x2 r2470, r2467, r2397;
}
{
add.f16x2 r2473, r412, r2470;
}
{
sub.f16x2 r2476, r1947, r1963;
}
{
mul.f16x2 r2479, r2476, r2399;
}
{
add.f16x2 r2482, r2473, r2479;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2485, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2486, {low, high};
}
{
neg.f16x2 r2487, r2486;
}
{
add.f16x2 r2489, r1979, r1995;
}
{
add.f16x2 r2492, r464, r2489;
}
{
add.f16x2 r2495, r1985, r2001;
}
{
add.f16x2 r2498, r500, r2495;
}
{
add.f16x2 r2501, r1979, r1995;
}
{
mul.f16x2 r2504, r2501, r2485;
}
{
add.f16x2 r2507, r464, r2504;
}
{
sub.f16x2 r2510, r1985, r2001;
}
{
mul.f16x2 r2513, r2510, r2487;
}
{
add.f16x2 r2516, r2507, r2513;
}
{
add.f16x2 r2519, r1979, r1995;
}
{
mul.f16x2 r2522, r2519, r2485;
}
{
add.f16x2 r2525, r464, r2522;
}
{
sub.f16x2 r2528, r1985, r2001;
}
{
mul.f16x2 r2531, r2528, r2487;
}
{
sub.f16x2 r2534, r2525, r2531;
}
{
add.f16x2 r2537, r1985, r2001;
}
{
mul.f16x2 r2540, r2537, r2485;
}
{
add.f16x2 r2543, r500, r2540;
}
{
sub.f16x2 r2546, r1979, r1995;
}
{
mul.f16x2 r2549, r2546, r2487;
}
{
sub.f16x2 r2552, r2543, r2549;
}
{
add.f16x2 r2555, r1985, r2001;
}
{
mul.f16x2 r2558, r2555, r2485;
}
{
add.f16x2 r2561, r500, r2558;
}
{
sub.f16x2 r2564, r1979, r1995;
}
{
mul.f16x2 r2567, r2564, r2487;
}
{
add.f16x2 r2570, r2561, r2567;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2573, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2574, {low, high};
}
{
neg.f16x2 r2575, r2574;
}
{
add.f16x2 r2577, r2011, r2027;
}
{
add.f16x2 r2580, r552, r2577;
}
{
add.f16x2 r2583, r2017, r2033;
}
{
add.f16x2 r2586, r588, r2583;
}
{
add.f16x2 r2589, r2011, r2027;
}
{
mul.f16x2 r2592, r2589, r2573;
}
{
add.f16x2 r2595, r552, r2592;
}
{
sub.f16x2 r2598, r2017, r2033;
}
{
mul.f16x2 r2601, r2598, r2575;
}
{
add.f16x2 r2604, r2595, r2601;
}
{
add.f16x2 r2607, r2011, r2027;
}
{
mul.f16x2 r2610, r2607, r2573;
}
{
add.f16x2 r2613, r552, r2610;
}
{
sub.f16x2 r2616, r2017, r2033;
}
{
mul.f16x2 r2619, r2616, r2575;
}
{
sub.f16x2 r2622, r2613, r2619;
}
{
add.f16x2 r2625, r2017, r2033;
}
{
mul.f16x2 r2628, r2625, r2573;
}
{
add.f16x2 r2631, r588, r2628;
}
{
sub.f16x2 r2634, r2011, r2027;
}
{
mul.f16x2 r2637, r2634, r2575;
}
{
sub.f16x2 r2640, r2631, r2637;
}
{
add.f16x2 r2643, r2017, r2033;
}
{
mul.f16x2 r2646, r2643, r2573;
}
{
add.f16x2 r2649, r588, r2646;
}
{
sub.f16x2 r2652, r2011, r2027;
}
{
mul.f16x2 r2655, r2652, r2575;
}
{
add.f16x2 r2658, r2649, r2655;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2661, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2662, {low, high};
}
{
neg.f16x2 r2663, r2662;
}
{
add.f16x2 r2665, r2043, r2059;
}
{
add.f16x2 r2668, r394, r2665;
}
{
add.f16x2 r2671, r2049, r2065;
}
{
add.f16x2 r2674, r430, r2671;
}
{
add.f16x2 r2677, r2043, r2059;
}
{
mul.f16x2 r2680, r2677, r2661;
}
{
add.f16x2 r2683, r394, r2680;
}
{
sub.f16x2 r2686, r2049, r2065;
}
{
mul.f16x2 r2689, r2686, r2663;
}
{
add.f16x2 r2692, r2683, r2689;
}
{
add.f16x2 r2695, r2043, r2059;
}
{
mul.f16x2 r2698, r2695, r2661;
}
{
add.f16x2 r2701, r394, r2698;
}
{
sub.f16x2 r2704, r2049, r2065;
}
{
mul.f16x2 r2707, r2704, r2663;
}
{
sub.f16x2 r2710, r2701, r2707;
}
{
add.f16x2 r2713, r2049, r2065;
}
{
mul.f16x2 r2716, r2713, r2661;
}
{
add.f16x2 r2719, r430, r2716;
}
{
sub.f16x2 r2722, r2043, r2059;
}
{
mul.f16x2 r2725, r2722, r2663;
}
{
sub.f16x2 r2728, r2719, r2725;
}
{
add.f16x2 r2731, r2049, r2065;
}
{
mul.f16x2 r2734, r2731, r2661;
}
{
add.f16x2 r2737, r430, r2734;
}
{
sub.f16x2 r2740, r2043, r2059;
}
{
mul.f16x2 r2743, r2740, r2663;
}
{
add.f16x2 r2746, r2737, r2743;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2749, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2750, {low, high};
}
{
neg.f16x2 r2751, r2750;
}
{
add.f16x2 r2753, r2075, r2091;
}
{
add.f16x2 r2756, r482, r2753;
}
{
add.f16x2 r2759, r2081, r2097;
}
{
add.f16x2 r2762, r518, r2759;
}
{
add.f16x2 r2765, r2075, r2091;
}
{
mul.f16x2 r2768, r2765, r2749;
}
{
add.f16x2 r2771, r482, r2768;
}
{
sub.f16x2 r2774, r2081, r2097;
}
{
mul.f16x2 r2777, r2774, r2751;
}
{
add.f16x2 r2780, r2771, r2777;
}
{
add.f16x2 r2783, r2075, r2091;
}
{
mul.f16x2 r2786, r2783, r2749;
}
{
add.f16x2 r2789, r482, r2786;
}
{
sub.f16x2 r2792, r2081, r2097;
}
{
mul.f16x2 r2795, r2792, r2751;
}
{
sub.f16x2 r2798, r2789, r2795;
}
{
add.f16x2 r2801, r2081, r2097;
}
{
mul.f16x2 r2804, r2801, r2749;
}
{
add.f16x2 r2807, r518, r2804;
}
{
sub.f16x2 r2810, r2075, r2091;
}
{
mul.f16x2 r2813, r2810, r2751;
}
{
sub.f16x2 r2816, r2807, r2813;
}
{
add.f16x2 r2819, r2081, r2097;
}
{
mul.f16x2 r2822, r2819, r2749;
}
{
add.f16x2 r2825, r518, r2822;
}
{
sub.f16x2 r2828, r2075, r2091;
}
{
mul.f16x2 r2831, r2828, r2751;
}
{
add.f16x2 r2834, r2825, r2831;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2837, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2838, {low, high};
}
{
neg.f16x2 r2839, r2838;
}
{
add.f16x2 r2841, r2107, r2123;
}
{
add.f16x2 r2844, r570, r2841;
}
{
add.f16x2 r2847, r2113, r2129;
}
{
add.f16x2 r2850, r606, r2847;
}
{
add.f16x2 r2853, r2107, r2123;
}
{
mul.f16x2 r2856, r2853, r2837;
}
{
add.f16x2 r2859, r570, r2856;
}
{
sub.f16x2 r2862, r2113, r2129;
}
{
mul.f16x2 r2865, r2862, r2839;
}
{
add.f16x2 r2868, r2859, r2865;
}
{
add.f16x2 r2871, r2107, r2123;
}
{
mul.f16x2 r2874, r2871, r2837;
}
{
add.f16x2 r2877, r570, r2874;
}
{
sub.f16x2 r2880, r2113, r2129;
}
{
mul.f16x2 r2883, r2880, r2839;
}
{
sub.f16x2 r2886, r2877, r2883;
}
{
add.f16x2 r2889, r2113, r2129;
}
{
mul.f16x2 r2892, r2889, r2837;
}
{
add.f16x2 r2895, r606, r2892;
}
{
sub.f16x2 r2898, r2107, r2123;
}
{
mul.f16x2 r2901, r2898, r2839;
}
{
sub.f16x2 r2904, r2895, r2901;
}
{
add.f16x2 r2907, r2113, r2129;
}
{
mul.f16x2 r2910, r2907, r2837;
}
{
add.f16x2 r2913, r606, r2910;
}
{
sub.f16x2 r2916, r2107, r2123;
}
{
mul.f16x2 r2919, r2916, r2839;
}
{
add.f16x2 r2922, r2913, r2919;
}
mul.wide.u32 rd2, r4685, -1431655765;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r4686, rd3;
mul.lo.s32 r4687, r4686, 3;
sub.s32 r4688, r4685, r4687;
cvt.rn.f32.u32 f401, r4688;
mul.f32 f402, f401, 0f3D9EDD1F;
cos.approx.f32 f309, f402;
sin.approx.f32 f403, f402;
neg.f32 f310, f403;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f309;
cvt.rn.f16.f32 high, f310;
mov.b32 r2925, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r2928, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r2930, {high, high};
}
{
mul.f16x2 r2932, r2234, r2930;
}
{
neg.f16x2 r2935, r2932;
}
{
fma.rn.f16x2 r2937, r2228, r2928, r2935;
}
{
mul.f16x2 r2941, r2228, r2930;
}
{
fma.rn.f16x2 r2944, r2234, r2928, r2941;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r2948, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r2950, {high, high};
}
mov.f32 f361, 0fBF800000;
mov.f32 f362, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r2952, {low, high};
}
{
mul.f16x2 r2953, r2950, r2952;
}
{
mul.f16x2 r2956, r2925, r2948;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r2959, {high, low};
}
{
fma.rn.f16x2 r2961, r2953, r2959, r2956;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2961;
mov.b32 r2965, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2961;
mov.b32 r2967, {high, high};
}
{
mul.f16x2 r2969, r2322, r2967;
}
{
neg.f16x2 r2972, r2969;
}
{
fma.rn.f16x2 r2974, r2316, r2965, r2972;
}
{
mul.f16x2 r2978, r2316, r2967;
}
{
fma.rn.f16x2 r2981, r2322, r2965, r2978;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r2985, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r2987, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r2989, {low, high};
}
{
mul.f16x2 r2990, r2987, r2989;
}
{
mul.f16x2 r2993, r2961, r2985;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2961;
mov.b32 r2996, {high, low};
}
{
fma.rn.f16x2 r2998, r2990, r2996, r2993;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2998;
mov.b32 r3002, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2998;
mov.b32 r3004, {high, high};
}
{
mul.f16x2 r3006, r2410, r3004;
}
{
neg.f16x2 r3009, r3006;
}
{
fma.rn.f16x2 r3011, r2404, r3002, r3009;
}
{
mul.f16x2 r3015, r2404, r3004;
}
{
fma.rn.f16x2 r3018, r2410, r3002, r3015;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3022, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3024, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3026, {low, high};
}
{
mul.f16x2 r3027, r3024, r3026;
}
{
mul.f16x2 r3030, r2998, r3022;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2998;
mov.b32 r3033, {high, low};
}
{
fma.rn.f16x2 r3035, r3027, r3033, r3030;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3035;
mov.b32 r3039, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3035;
mov.b32 r3041, {high, high};
}
{
mul.f16x2 r3043, r2498, r3041;
}
{
neg.f16x2 r3046, r3043;
}
{
fma.rn.f16x2 r3048, r2492, r3039, r3046;
}
{
mul.f16x2 r3052, r2492, r3041;
}
{
fma.rn.f16x2 r3055, r2498, r3039, r3052;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3059, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3061, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3063, {low, high};
}
{
mul.f16x2 r3064, r3061, r3063;
}
{
mul.f16x2 r3067, r3035, r3059;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3035;
mov.b32 r3070, {high, low};
}
{
fma.rn.f16x2 r3072, r3064, r3070, r3067;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3072;
mov.b32 r3076, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3072;
mov.b32 r3078, {high, high};
}
{
mul.f16x2 r3080, r2586, r3078;
}
{
neg.f16x2 r3083, r3080;
}
{
fma.rn.f16x2 r3085, r2580, r3076, r3083;
}
{
mul.f16x2 r3089, r2580, r3078;
}
{
fma.rn.f16x2 r3092, r2586, r3076, r3089;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3096, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3098, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3100, {low, high};
}
{
mul.f16x2 r3101, r3098, r3100;
}
{
mul.f16x2 r3104, r3072, r3096;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3072;
mov.b32 r3107, {high, low};
}
{
fma.rn.f16x2 r3109, r3101, r3107, r3104;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3109;
mov.b32 r3113, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3109;
mov.b32 r3115, {high, high};
}
{
mul.f16x2 r3117, r2674, r3115;
}
{
neg.f16x2 r3120, r3117;
}
{
fma.rn.f16x2 r3122, r2668, r3113, r3120;
}
{
mul.f16x2 r3126, r2668, r3115;
}
{
fma.rn.f16x2 r3129, r2674, r3113, r3126;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3133, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3135, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3137, {low, high};
}
{
mul.f16x2 r3138, r3135, r3137;
}
{
mul.f16x2 r3141, r3109, r3133;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3109;
mov.b32 r3144, {high, low};
}
{
fma.rn.f16x2 r3146, r3138, r3144, r3141;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3146;
mov.b32 r3150, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3146;
mov.b32 r3152, {high, high};
}
{
mul.f16x2 r3154, r2762, r3152;
}
{
neg.f16x2 r3157, r3154;
}
{
fma.rn.f16x2 r3159, r2756, r3150, r3157;
}
{
mul.f16x2 r3163, r2756, r3152;
}
{
fma.rn.f16x2 r3166, r2762, r3150, r3163;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3170, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3172, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3174, {low, high};
}
{
mul.f16x2 r3175, r3172, r3174;
}
{
mul.f16x2 r3178, r3146, r3170;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3146;
mov.b32 r3181, {high, low};
}
{
fma.rn.f16x2 r3183, r3175, r3181, r3178;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3183;
mov.b32 r3187, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3183;
mov.b32 r3189, {high, high};
}
{
mul.f16x2 r3191, r2850, r3189;
}
{
neg.f16x2 r3194, r3191;
}
{
fma.rn.f16x2 r3196, r2844, r3187, r3194;
}
{
mul.f16x2 r3200, r2844, r3189;
}
{
fma.rn.f16x2 r3203, r2850, r3187, r3200;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3207, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3209, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3211, {low, high};
}
{
mul.f16x2 r3212, r3209, r3211;
}
{
mul.f16x2 r3215, r3183, r3207;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3183;
mov.b32 r3218, {high, low};
}
{
fma.rn.f16x2 r3220, r3212, r3218, r3215;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3220;
mov.b32 r3224, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3220;
mov.b32 r3226, {high, high};
}
{
mul.f16x2 r3228, r2200, r3226;
}
{
neg.f16x2 r3231, r3228;
}
{
fma.rn.f16x2 r3233, r2164, r3224, r3231;
}
{
mul.f16x2 r3237, r2164, r3226;
}
{
fma.rn.f16x2 r3240, r2200, r3224, r3237;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3244, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3246, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3248, {low, high};
}
{
mul.f16x2 r3249, r3246, r3248;
}
{
mul.f16x2 r3252, r3220, r3244;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3220;
mov.b32 r3255, {high, low};
}
{
fma.rn.f16x2 r3257, r3249, r3255, r3252;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3257;
mov.b32 r3261, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3257;
mov.b32 r3263, {high, high};
}
{
mul.f16x2 r3265, r2288, r3263;
}
{
neg.f16x2 r3268, r3265;
}
{
fma.rn.f16x2 r3270, r2252, r3261, r3268;
}
{
mul.f16x2 r3274, r2252, r3263;
}
{
fma.rn.f16x2 r3277, r2288, r3261, r3274;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3281, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3283, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3285, {low, high};
}
{
mul.f16x2 r3286, r3283, r3285;
}
{
mul.f16x2 r3289, r3257, r3281;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3257;
mov.b32 r3292, {high, low};
}
{
fma.rn.f16x2 r3294, r3286, r3292, r3289;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3294;
mov.b32 r3298, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3294;
mov.b32 r3300, {high, high};
}
{
mul.f16x2 r3302, r2376, r3300;
}
{
neg.f16x2 r3305, r3302;
}
{
fma.rn.f16x2 r3307, r2340, r3298, r3305;
}
{
mul.f16x2 r3311, r2340, r3300;
}
{
fma.rn.f16x2 r3314, r2376, r3298, r3311;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3318, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3320, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3322, {low, high};
}
{
mul.f16x2 r3323, r3320, r3322;
}
{
mul.f16x2 r3326, r3294, r3318;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3294;
mov.b32 r3329, {high, low};
}
{
fma.rn.f16x2 r3331, r3323, r3329, r3326;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3331;
mov.b32 r3335, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3331;
mov.b32 r3337, {high, high};
}
{
mul.f16x2 r3339, r2464, r3337;
}
{
neg.f16x2 r3342, r3339;
}
{
fma.rn.f16x2 r3344, r2428, r3335, r3342;
}
{
mul.f16x2 r3348, r2428, r3337;
}
{
fma.rn.f16x2 r3351, r2464, r3335, r3348;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3355, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3357, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3359, {low, high};
}
{
mul.f16x2 r3360, r3357, r3359;
}
{
mul.f16x2 r3363, r3331, r3355;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3331;
mov.b32 r3366, {high, low};
}
{
fma.rn.f16x2 r3368, r3360, r3366, r3363;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3368;
mov.b32 r3372, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3368;
mov.b32 r3374, {high, high};
}
{
mul.f16x2 r3376, r2552, r3374;
}
{
neg.f16x2 r3379, r3376;
}
{
fma.rn.f16x2 r3381, r2516, r3372, r3379;
}
{
mul.f16x2 r3385, r2516, r3374;
}
{
fma.rn.f16x2 r3388, r2552, r3372, r3385;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3392, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3394, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3396, {low, high};
}
{
mul.f16x2 r3397, r3394, r3396;
}
{
mul.f16x2 r3400, r3368, r3392;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3368;
mov.b32 r3403, {high, low};
}
{
fma.rn.f16x2 r3405, r3397, r3403, r3400;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3405;
mov.b32 r3409, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3405;
mov.b32 r3411, {high, high};
}
{
mul.f16x2 r3413, r2640, r3411;
}
{
neg.f16x2 r3416, r3413;
}
{
fma.rn.f16x2 r3418, r2604, r3409, r3416;
}
{
mul.f16x2 r3422, r2604, r3411;
}
{
fma.rn.f16x2 r3425, r2640, r3409, r3422;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3429, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3431, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3433, {low, high};
}
{
mul.f16x2 r3434, r3431, r3433;
}
{
mul.f16x2 r3437, r3405, r3429;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3405;
mov.b32 r3440, {high, low};
}
{
fma.rn.f16x2 r3442, r3434, r3440, r3437;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3442;
mov.b32 r3446, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3442;
mov.b32 r3448, {high, high};
}
{
mul.f16x2 r3450, r2728, r3448;
}
{
neg.f16x2 r3453, r3450;
}
{
fma.rn.f16x2 r3455, r2692, r3446, r3453;
}
{
mul.f16x2 r3459, r2692, r3448;
}
{
fma.rn.f16x2 r3462, r2728, r3446, r3459;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3466, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3468, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3470, {low, high};
}
{
mul.f16x2 r3471, r3468, r3470;
}
{
mul.f16x2 r3474, r3442, r3466;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3442;
mov.b32 r3477, {high, low};
}
{
fma.rn.f16x2 r3479, r3471, r3477, r3474;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3479;
mov.b32 r3483, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3479;
mov.b32 r3485, {high, high};
}
{
mul.f16x2 r3487, r2816, r3485;
}
{
neg.f16x2 r3490, r3487;
}
{
fma.rn.f16x2 r3492, r2780, r3483, r3490;
}
{
mul.f16x2 r3496, r2780, r3485;
}
{
fma.rn.f16x2 r3499, r2816, r3483, r3496;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3503, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3505, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3507, {low, high};
}
{
mul.f16x2 r3508, r3505, r3507;
}
{
mul.f16x2 r3511, r3479, r3503;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3479;
mov.b32 r3514, {high, low};
}
{
fma.rn.f16x2 r3516, r3508, r3514, r3511;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3516;
mov.b32 r3520, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3516;
mov.b32 r3522, {high, high};
}
{
mul.f16x2 r3524, r2904, r3522;
}
{
neg.f16x2 r3527, r3524;
}
{
fma.rn.f16x2 r3529, r2868, r3520, r3527;
}
{
mul.f16x2 r3533, r2868, r3522;
}
{
fma.rn.f16x2 r3536, r2904, r3520, r3533;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3540, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3542, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3544, {low, high};
}
{
mul.f16x2 r3545, r3542, r3544;
}
{
mul.f16x2 r3548, r3516, r3540;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3516;
mov.b32 r3551, {high, low};
}
{
fma.rn.f16x2 r3553, r3545, r3551, r3548;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3553;
mov.b32 r3557, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3553;
mov.b32 r3559, {high, high};
}
{
mul.f16x2 r3561, r2218, r3559;
}
{
neg.f16x2 r3564, r3561;
}
{
fma.rn.f16x2 r3566, r2182, r3557, r3564;
}
{
mul.f16x2 r3570, r2182, r3559;
}
{
fma.rn.f16x2 r3573, r2218, r3557, r3570;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3577, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3579, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3581, {low, high};
}
{
mul.f16x2 r3582, r3579, r3581;
}
{
mul.f16x2 r3585, r3553, r3577;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3553;
mov.b32 r3588, {high, low};
}
{
fma.rn.f16x2 r3590, r3582, r3588, r3585;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3590;
mov.b32 r3594, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3590;
mov.b32 r3596, {high, high};
}
{
mul.f16x2 r3598, r2306, r3596;
}
{
neg.f16x2 r3601, r3598;
}
{
fma.rn.f16x2 r3603, r2270, r3594, r3601;
}
{
mul.f16x2 r3607, r2270, r3596;
}
{
fma.rn.f16x2 r3610, r2306, r3594, r3607;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3614, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3616, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3618, {low, high};
}
{
mul.f16x2 r3619, r3616, r3618;
}
{
mul.f16x2 r3622, r3590, r3614;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3590;
mov.b32 r3625, {high, low};
}
{
fma.rn.f16x2 r3627, r3619, r3625, r3622;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3627;
mov.b32 r3631, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3627;
mov.b32 r3633, {high, high};
}
{
mul.f16x2 r3635, r2394, r3633;
}
{
neg.f16x2 r3638, r3635;
}
{
fma.rn.f16x2 r3640, r2358, r3631, r3638;
}
{
mul.f16x2 r3644, r2358, r3633;
}
{
fma.rn.f16x2 r3647, r2394, r3631, r3644;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3651, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3653, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3655, {low, high};
}
{
mul.f16x2 r3656, r3653, r3655;
}
{
mul.f16x2 r3659, r3627, r3651;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3627;
mov.b32 r3662, {high, low};
}
{
fma.rn.f16x2 r3664, r3656, r3662, r3659;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3664;
mov.b32 r3668, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3664;
mov.b32 r3670, {high, high};
}
{
mul.f16x2 r3672, r2482, r3670;
}
{
neg.f16x2 r3675, r3672;
}
{
fma.rn.f16x2 r3677, r2446, r3668, r3675;
}
{
mul.f16x2 r3681, r2446, r3670;
}
{
fma.rn.f16x2 r3684, r2482, r3668, r3681;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3688, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3690, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3692, {low, high};
}
{
mul.f16x2 r3693, r3690, r3692;
}
{
mul.f16x2 r3696, r3664, r3688;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3664;
mov.b32 r3699, {high, low};
}
{
fma.rn.f16x2 r3701, r3693, r3699, r3696;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3701;
mov.b32 r3705, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3701;
mov.b32 r3707, {high, high};
}
{
mul.f16x2 r3709, r2570, r3707;
}
{
neg.f16x2 r3712, r3709;
}
{
fma.rn.f16x2 r3714, r2534, r3705, r3712;
}
{
mul.f16x2 r3718, r2534, r3707;
}
{
fma.rn.f16x2 r3721, r2570, r3705, r3718;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3725, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3727, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3729, {low, high};
}
{
mul.f16x2 r3730, r3727, r3729;
}
{
mul.f16x2 r3733, r3701, r3725;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3701;
mov.b32 r3736, {high, low};
}
{
fma.rn.f16x2 r3738, r3730, r3736, r3733;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3738;
mov.b32 r3742, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3738;
mov.b32 r3744, {high, high};
}
{
mul.f16x2 r3746, r2658, r3744;
}
{
neg.f16x2 r3749, r3746;
}
{
fma.rn.f16x2 r3751, r2622, r3742, r3749;
}
{
mul.f16x2 r3755, r2622, r3744;
}
{
fma.rn.f16x2 r3758, r2658, r3742, r3755;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3762, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3764, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3766, {low, high};
}
{
mul.f16x2 r3767, r3764, r3766;
}
{
mul.f16x2 r3770, r3738, r3762;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3738;
mov.b32 r3773, {high, low};
}
{
fma.rn.f16x2 r3775, r3767, r3773, r3770;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3775;
mov.b32 r3779, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3775;
mov.b32 r3781, {high, high};
}
{
mul.f16x2 r3783, r2746, r3781;
}
{
neg.f16x2 r3786, r3783;
}
{
fma.rn.f16x2 r3788, r2710, r3779, r3786;
}
{
mul.f16x2 r3792, r2710, r3781;
}
{
fma.rn.f16x2 r3795, r2746, r3779, r3792;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3799, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3801, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3803, {low, high};
}
{
mul.f16x2 r3804, r3801, r3803;
}
{
mul.f16x2 r3807, r3775, r3799;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3775;
mov.b32 r3810, {high, low};
}
{
fma.rn.f16x2 r3812, r3804, r3810, r3807;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3812;
mov.b32 r3816, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3812;
mov.b32 r3818, {high, high};
}
{
mul.f16x2 r3820, r2834, r3818;
}
{
neg.f16x2 r3823, r3820;
}
{
fma.rn.f16x2 r3825, r2798, r3816, r3823;
}
{
mul.f16x2 r3829, r2798, r3818;
}
{
fma.rn.f16x2 r3832, r2834, r3816, r3829;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3836, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3838, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3840, {low, high};
}
{
mul.f16x2 r3841, r3838, r3840;
}
{
mul.f16x2 r3844, r3812, r3836;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3812;
mov.b32 r3847, {high, low};
}
{
fma.rn.f16x2 r3849, r3841, r3847, r3844;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3849;
mov.b32 r3853, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3849;
mov.b32 r3855, {high, high};
}
{
mul.f16x2 r3857, r2922, r3855;
}
{
neg.f16x2 r3860, r3857;
}
{
fma.rn.f16x2 r3862, r2886, r3853, r3860;
}
{
mul.f16x2 r3866, r2886, r3855;
}
{
fma.rn.f16x2 r3869, r2922, r3853, r3866;
}
mad.lo.s32 r4689, r4686, 648, r4684;
barrier.sync 0;
mad.lo.s32 r4690, r4688, 216, r4689;
st.shared.v2.f32 [r4690], {r2140, r2146};
st.shared.v2.f32 [r4690+8], {r2937, r2944};
st.shared.v2.f32 [r4690+16], {r2974, r2981};
st.shared.v2.f32 [r4690+24], {r3011, r3018};
st.shared.v2.f32 [r4690+32], {r3048, r3055};
st.shared.v2.f32 [r4690+40], {r3085, r3092};
st.shared.v2.f32 [r4690+48], {r3122, r3129};
st.shared.v2.f32 [r4690+56], {r3159, r3166};
st.shared.v2.f32 [r4690+64], {r3196, r3203};
st.shared.v2.f32 [r4690+72], {r3233, r3240};
st.shared.v2.f32 [r4690+80], {r3270, r3277};
st.shared.v2.f32 [r4690+88], {r3307, r3314};
st.shared.v2.f32 [r4690+96], {r3344, r3351};
st.shared.v2.f32 [r4690+104], {r3381, r3388};
st.shared.v2.f32 [r4690+112], {r3418, r3425};
st.shared.v2.f32 [r4690+120], {r3455, r3462};
st.shared.v2.f32 [r4690+128], {r3492, r3499};
st.shared.v2.f32 [r4690+136], {r3529, r3536};
st.shared.v2.f32 [r4690+144], {r3566, r3573};
st.shared.v2.f32 [r4690+152], {r3603, r3610};
st.shared.v2.f32 [r4690+160], {r3640, r3647};
st.shared.v2.f32 [r4690+168], {r3677, r3684};
st.shared.v2.f32 [r4690+176], {r3714, r3721};
st.shared.v2.f32 [r4690+184], {r3751, r3758};
st.shared.v2.f32 [r4690+192], {r3788, r3795};
st.shared.v2.f32 [r4690+200], {r3825, r3832};
st.shared.v2.f32 [r4690+208], {r3862, r3869};
barrier.sync 0;
mad.lo.s32 r4691, r4688, -208, r4690;
ld.shared.u32 r3898, [r4691];
ld.shared.u32 r3904, [r4691+4];
ld.shared.u32 r3986, [r4691+24];
ld.shared.u32 r3992, [r4691+28];
ld.shared.u32 r4074, [r4691+48];
ld.shared.u32 r4080, [r4691+52];
ld.shared.u32 r4162, [r4691+72];
ld.shared.u32 r4168, [r4691+76];
ld.shared.u32 r4250, [r4691+96];
ld.shared.u32 r4256, [r4691+100];
ld.shared.u32 r4338, [r4691+120];
ld.shared.u32 r4344, [r4691+124];
ld.shared.u32 r4426, [r4691+144];
ld.shared.u32 r4432, [r4691+148];
ld.shared.u32 r4514, [r4691+168];
ld.shared.u32 r4520, [r4691+172];
ld.shared.u32 r4602, [r4691+192];
ld.shared.u32 r4608, [r4691+196];
ld.shared.u32 r3895, [r4691+216];
ld.shared.u32 r3901, [r4691+220];
ld.shared.u32 r3983, [r4691+240];
ld.shared.u32 r3989, [r4691+244];
ld.shared.u32 r4071, [r4691+264];
ld.shared.u32 r4077, [r4691+268];
ld.shared.u32 r4159, [r4691+288];
ld.shared.u32 r4165, [r4691+292];
ld.shared.u32 r4247, [r4691+312];
ld.shared.u32 r4253, [r4691+316];
ld.shared.u32 r4335, [r4691+336];
ld.shared.u32 r4341, [r4691+340];
ld.shared.u32 r4423, [r4691+360];
ld.shared.u32 r4429, [r4691+364];
ld.shared.u32 r4511, [r4691+384];
ld.shared.u32 r4517, [r4691+388];
ld.shared.u32 r4599, [r4691+408];
ld.shared.u32 r4605, [r4691+412];
ld.shared.u32 r3896, [r4691+432];
ld.shared.u32 r3902, [r4691+436];
ld.shared.u32 r3984, [r4691+456];
ld.shared.u32 r3990, [r4691+460];
ld.shared.u32 r4072, [r4691+480];
ld.shared.u32 r4078, [r4691+484];
ld.shared.u32 r4160, [r4691+504];
ld.shared.u32 r4166, [r4691+508];
ld.shared.u32 r4248, [r4691+528];
ld.shared.u32 r4254, [r4691+532];
ld.shared.u32 r4336, [r4691+552];
ld.shared.u32 r4342, [r4691+556];
ld.shared.u32 r4424, [r4691+576];
ld.shared.u32 r4430, [r4691+580];
ld.shared.u32 r4512, [r4691+600];
ld.shared.u32 r4518, [r4691+604];
ld.shared.u32 r4600, [r4691+624];
ld.shared.u32 r4606, [r4691+628];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r3890, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r3891, {low, high};
}
{
neg.f16x2 r3892, r3891;
}
{
add.f16x2 r3894, r3895, r3896;
}
{
add.f16x2 %0, r3898, r3894;
}
{
add.f16x2 r3900, r3901, r3902;
}
{
add.f16x2 %1, r3904, r3900;
}
{
add.f16x2 r3906, r3895, r3896;
}
{
mul.f16x2 r3909, r3906, r3890;
}
{
add.f16x2 r3912, r3898, r3909;
}
{
sub.f16x2 r3915, r3901, r3902;
}
{
mul.f16x2 r3918, r3915, r3892;
}
{
add.f16x2 %18, r3912, r3918;
}
{
add.f16x2 r3924, r3895, r3896;
}
{
mul.f16x2 r3927, r3924, r3890;
}
{
add.f16x2 r3930, r3898, r3927;
}
{
sub.f16x2 r3933, r3901, r3902;
}
{
mul.f16x2 r3936, r3933, r3892;
}
{
sub.f16x2 %36, r3930, r3936;
}
{
add.f16x2 r3942, r3901, r3902;
}
{
mul.f16x2 r3945, r3942, r3890;
}
{
add.f16x2 r3948, r3904, r3945;
}
{
sub.f16x2 r3951, r3895, r3896;
}
{
mul.f16x2 r3954, r3951, r3892;
}
{
sub.f16x2 %19, r3948, r3954;
}
{
add.f16x2 r3960, r3901, r3902;
}
{
mul.f16x2 r3963, r3960, r3890;
}
{
add.f16x2 r3966, r3904, r3963;
}
{
sub.f16x2 r3969, r3895, r3896;
}
{
mul.f16x2 r3972, r3969, r3892;
}
{
add.f16x2 %37, r3966, r3972;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r3978, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r3979, {low, high};
}
{
neg.f16x2 r3980, r3979;
}
{
add.f16x2 r3982, r3983, r3984;
}
{
add.f16x2 %2, r3986, r3982;
}
{
add.f16x2 r3988, r3989, r3990;
}
{
add.f16x2 %3, r3992, r3988;
}
{
add.f16x2 r3994, r3983, r3984;
}
{
mul.f16x2 r3997, r3994, r3978;
}
{
add.f16x2 r4000, r3986, r3997;
}
{
sub.f16x2 r4003, r3989, r3990;
}
{
mul.f16x2 r4006, r4003, r3980;
}
{
add.f16x2 %20, r4000, r4006;
}
{
add.f16x2 r4012, r3983, r3984;
}
{
mul.f16x2 r4015, r4012, r3978;
}
{
add.f16x2 r4018, r3986, r4015;
}
{
sub.f16x2 r4021, r3989, r3990;
}
{
mul.f16x2 r4024, r4021, r3980;
}
{
sub.f16x2 %38, r4018, r4024;
}
{
add.f16x2 r4030, r3989, r3990;
}
{
mul.f16x2 r4033, r4030, r3978;
}
{
add.f16x2 r4036, r3992, r4033;
}
{
sub.f16x2 r4039, r3983, r3984;
}
{
mul.f16x2 r4042, r4039, r3980;
}
{
sub.f16x2 %21, r4036, r4042;
}
{
add.f16x2 r4048, r3989, r3990;
}
{
mul.f16x2 r4051, r4048, r3978;
}
{
add.f16x2 r4054, r3992, r4051;
}
{
sub.f16x2 r4057, r3983, r3984;
}
{
mul.f16x2 r4060, r4057, r3980;
}
{
add.f16x2 %39, r4054, r4060;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4066, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4067, {low, high};
}
{
neg.f16x2 r4068, r4067;
}
{
add.f16x2 r4070, r4071, r4072;
}
{
add.f16x2 %4, r4074, r4070;
}
{
add.f16x2 r4076, r4077, r4078;
}
{
add.f16x2 %5, r4080, r4076;
}
{
add.f16x2 r4082, r4071, r4072;
}
{
mul.f16x2 r4085, r4082, r4066;
}
{
add.f16x2 r4088, r4074, r4085;
}
{
sub.f16x2 r4091, r4077, r4078;
}
{
mul.f16x2 r4094, r4091, r4068;
}
{
add.f16x2 %22, r4088, r4094;
}
{
add.f16x2 r4100, r4071, r4072;
}
{
mul.f16x2 r4103, r4100, r4066;
}
{
add.f16x2 r4106, r4074, r4103;
}
{
sub.f16x2 r4109, r4077, r4078;
}
{
mul.f16x2 r4112, r4109, r4068;
}
{
sub.f16x2 %40, r4106, r4112;
}
{
add.f16x2 r4118, r4077, r4078;
}
{
mul.f16x2 r4121, r4118, r4066;
}
{
add.f16x2 r4124, r4080, r4121;
}
{
sub.f16x2 r4127, r4071, r4072;
}
{
mul.f16x2 r4130, r4127, r4068;
}
{
sub.f16x2 %23, r4124, r4130;
}
{
add.f16x2 r4136, r4077, r4078;
}
{
mul.f16x2 r4139, r4136, r4066;
}
{
add.f16x2 r4142, r4080, r4139;
}
{
sub.f16x2 r4145, r4071, r4072;
}
{
mul.f16x2 r4148, r4145, r4068;
}
{
add.f16x2 %41, r4142, r4148;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4154, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4155, {low, high};
}
{
neg.f16x2 r4156, r4155;
}
{
add.f16x2 r4158, r4159, r4160;
}
{
add.f16x2 %6, r4162, r4158;
}
{
add.f16x2 r4164, r4165, r4166;
}
{
add.f16x2 %7, r4168, r4164;
}
{
add.f16x2 r4170, r4159, r4160;
}
{
mul.f16x2 r4173, r4170, r4154;
}
{
add.f16x2 r4176, r4162, r4173;
}
{
sub.f16x2 r4179, r4165, r4166;
}
{
mul.f16x2 r4182, r4179, r4156;
}
{
add.f16x2 %24, r4176, r4182;
}
{
add.f16x2 r4188, r4159, r4160;
}
{
mul.f16x2 r4191, r4188, r4154;
}
{
add.f16x2 r4194, r4162, r4191;
}
{
sub.f16x2 r4197, r4165, r4166;
}
{
mul.f16x2 r4200, r4197, r4156;
}
{
sub.f16x2 %42, r4194, r4200;
}
{
add.f16x2 r4206, r4165, r4166;
}
{
mul.f16x2 r4209, r4206, r4154;
}
{
add.f16x2 r4212, r4168, r4209;
}
{
sub.f16x2 r4215, r4159, r4160;
}
{
mul.f16x2 r4218, r4215, r4156;
}
{
sub.f16x2 %25, r4212, r4218;
}
{
add.f16x2 r4224, r4165, r4166;
}
{
mul.f16x2 r4227, r4224, r4154;
}
{
add.f16x2 r4230, r4168, r4227;
}
{
sub.f16x2 r4233, r4159, r4160;
}
{
mul.f16x2 r4236, r4233, r4156;
}
{
add.f16x2 %43, r4230, r4236;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4242, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4243, {low, high};
}
{
neg.f16x2 r4244, r4243;
}
{
add.f16x2 r4246, r4247, r4248;
}
{
add.f16x2 %8, r4250, r4246;
}
{
add.f16x2 r4252, r4253, r4254;
}
{
add.f16x2 %9, r4256, r4252;
}
{
add.f16x2 r4258, r4247, r4248;
}
{
mul.f16x2 r4261, r4258, r4242;
}
{
add.f16x2 r4264, r4250, r4261;
}
{
sub.f16x2 r4267, r4253, r4254;
}
{
mul.f16x2 r4270, r4267, r4244;
}
{
add.f16x2 %26, r4264, r4270;
}
{
add.f16x2 r4276, r4247, r4248;
}
{
mul.f16x2 r4279, r4276, r4242;
}
{
add.f16x2 r4282, r4250, r4279;
}
{
sub.f16x2 r4285, r4253, r4254;
}
{
mul.f16x2 r4288, r4285, r4244;
}
{
sub.f16x2 %44, r4282, r4288;
}
{
add.f16x2 r4294, r4253, r4254;
}
{
mul.f16x2 r4297, r4294, r4242;
}
{
add.f16x2 r4300, r4256, r4297;
}
{
sub.f16x2 r4303, r4247, r4248;
}
{
mul.f16x2 r4306, r4303, r4244;
}
{
sub.f16x2 %27, r4300, r4306;
}
{
add.f16x2 r4312, r4253, r4254;
}
{
mul.f16x2 r4315, r4312, r4242;
}
{
add.f16x2 r4318, r4256, r4315;
}
{
sub.f16x2 r4321, r4247, r4248;
}
{
mul.f16x2 r4324, r4321, r4244;
}
{
add.f16x2 %45, r4318, r4324;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4330, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4331, {low, high};
}
{
neg.f16x2 r4332, r4331;
}
{
add.f16x2 r4334, r4335, r4336;
}
{
add.f16x2 %10, r4338, r4334;
}
{
add.f16x2 r4340, r4341, r4342;
}
{
add.f16x2 %11, r4344, r4340;
}
{
add.f16x2 r4346, r4335, r4336;
}
{
mul.f16x2 r4349, r4346, r4330;
}
{
add.f16x2 r4352, r4338, r4349;
}
{
sub.f16x2 r4355, r4341, r4342;
}
{
mul.f16x2 r4358, r4355, r4332;
}
{
add.f16x2 %28, r4352, r4358;
}
{
add.f16x2 r4364, r4335, r4336;
}
{
mul.f16x2 r4367, r4364, r4330;
}
{
add.f16x2 r4370, r4338, r4367;
}
{
sub.f16x2 r4373, r4341, r4342;
}
{
mul.f16x2 r4376, r4373, r4332;
}
{
sub.f16x2 %46, r4370, r4376;
}
{
add.f16x2 r4382, r4341, r4342;
}
{
mul.f16x2 r4385, r4382, r4330;
}
{
add.f16x2 r4388, r4344, r4385;
}
{
sub.f16x2 r4391, r4335, r4336;
}
{
mul.f16x2 r4394, r4391, r4332;
}
{
sub.f16x2 %29, r4388, r4394;
}
{
add.f16x2 r4400, r4341, r4342;
}
{
mul.f16x2 r4403, r4400, r4330;
}
{
add.f16x2 r4406, r4344, r4403;
}
{
sub.f16x2 r4409, r4335, r4336;
}
{
mul.f16x2 r4412, r4409, r4332;
}
{
add.f16x2 %47, r4406, r4412;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4418, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4419, {low, high};
}
{
neg.f16x2 r4420, r4419;
}
{
add.f16x2 r4422, r4423, r4424;
}
{
add.f16x2 %12, r4426, r4422;
}
{
add.f16x2 r4428, r4429, r4430;
}
{
add.f16x2 %13, r4432, r4428;
}
{
add.f16x2 r4434, r4423, r4424;
}
{
mul.f16x2 r4437, r4434, r4418;
}
{
add.f16x2 r4440, r4426, r4437;
}
{
sub.f16x2 r4443, r4429, r4430;
}
{
mul.f16x2 r4446, r4443, r4420;
}
{
add.f16x2 %30, r4440, r4446;
}
{
add.f16x2 r4452, r4423, r4424;
}
{
mul.f16x2 r4455, r4452, r4418;
}
{
add.f16x2 r4458, r4426, r4455;
}
{
sub.f16x2 r4461, r4429, r4430;
}
{
mul.f16x2 r4464, r4461, r4420;
}
{
sub.f16x2 %48, r4458, r4464;
}
{
add.f16x2 r4470, r4429, r4430;
}
{
mul.f16x2 r4473, r4470, r4418;
}
{
add.f16x2 r4476, r4432, r4473;
}
{
sub.f16x2 r4479, r4423, r4424;
}
{
mul.f16x2 r4482, r4479, r4420;
}
{
sub.f16x2 %31, r4476, r4482;
}
{
add.f16x2 r4488, r4429, r4430;
}
{
mul.f16x2 r4491, r4488, r4418;
}
{
add.f16x2 r4494, r4432, r4491;
}
{
sub.f16x2 r4497, r4423, r4424;
}
{
mul.f16x2 r4500, r4497, r4420;
}
{
add.f16x2 %49, r4494, r4500;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4506, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4507, {low, high};
}
{
neg.f16x2 r4508, r4507;
}
{
add.f16x2 r4510, r4511, r4512;
}
{
add.f16x2 %14, r4514, r4510;
}
{
add.f16x2 r4516, r4517, r4518;
}
{
add.f16x2 %15, r4520, r4516;
}
{
add.f16x2 r4522, r4511, r4512;
}
{
mul.f16x2 r4525, r4522, r4506;
}
{
add.f16x2 r4528, r4514, r4525;
}
{
sub.f16x2 r4531, r4517, r4518;
}
{
mul.f16x2 r4534, r4531, r4508;
}
{
add.f16x2 %32, r4528, r4534;
}
{
add.f16x2 r4540, r4511, r4512;
}
{
mul.f16x2 r4543, r4540, r4506;
}
{
add.f16x2 r4546, r4514, r4543;
}
{
sub.f16x2 r4549, r4517, r4518;
}
{
mul.f16x2 r4552, r4549, r4508;
}
{
sub.f16x2 %50, r4546, r4552;
}
{
add.f16x2 r4558, r4517, r4518;
}
{
mul.f16x2 r4561, r4558, r4506;
}
{
add.f16x2 r4564, r4520, r4561;
}
{
sub.f16x2 r4567, r4511, r4512;
}
{
mul.f16x2 r4570, r4567, r4508;
}
{
sub.f16x2 %33, r4564, r4570;
}
{
add.f16x2 r4576, r4517, r4518;
}
{
mul.f16x2 r4579, r4576, r4506;
}
{
add.f16x2 r4582, r4520, r4579;
}
{
sub.f16x2 r4585, r4511, r4512;
}
{
mul.f16x2 r4588, r4585, r4508;
}
{
add.f16x2 %51, r4582, r4588;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4594, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4595, {low, high};
}
{
neg.f16x2 r4596, r4595;
}
{
add.f16x2 r4598, r4599, r4600;
}
{
add.f16x2 %16, r4602, r4598;
}
{
add.f16x2 r4604, r4605, r4606;
}
{
add.f16x2 %17, r4608, r4604;
}
{
add.f16x2 r4610, r4599, r4600;
}
{
mul.f16x2 r4613, r4610, r4594;
}
{
add.f16x2 r4616, r4602, r4613;
}
{
sub.f16x2 r4619, r4605, r4606;
}
{
mul.f16x2 r4622, r4619, r4596;
}
{
add.f16x2 %34, r4616, r4622;
}
{
add.f16x2 r4628, r4599, r4600;
}
{
mul.f16x2 r4631, r4628, r4594;
}
{
add.f16x2 r4634, r4602, r4631;
}
{
sub.f16x2 r4637, r4605, r4606;
}
{
mul.f16x2 r4640, r4637, r4596;
}
{
sub.f16x2 %52, r4634, r4640;
}
{
add.f16x2 r4646, r4605, r4606;
}
{
mul.f16x2 r4649, r4646, r4594;
}
{
add.f16x2 r4652, r4608, r4649;
}
{
sub.f16x2 r4655, r4599, r4600;
}
{
mul.f16x2 r4658, r4655, r4596;
}
{
sub.f16x2 %35, r4652, r4658;
}
{
add.f16x2 r4664, r4605, r4606;
}
{
mul.f16x2 r4667, r4664, r4594;
}
{
add.f16x2 r4670, r4608, r4667;
}
{
sub.f16x2 r4673, r4599, r4600;
}
{
mul.f16x2 r4676, r4673, r4596;
}
{
add.f16x2 %53, r4670, r4676;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[4].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<874, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<404>;
.reg .b32 r<4748>;
.reg .b64 rd<4>;
mov.u32 r4746, %tid.y;
mov.u32 r4747, %54;
mad.lo.s32 r4684, r4746, 324, r4747;
mov.u32 r4685, %tid.x;
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
neg.f16x2 r3, r2;
}
{
add.f16x2 r5, %74, %58;
}
{
add.f16x2 r8, %83, r5;
}
{
add.f16x2 r11, %91, %73;
}
{
add.f16x2 r14, %97, r11;
}
{
add.f16x2 r17, %74, %58;
}
{
mul.f16x2 r20, r17, r1;
}
{
add.f16x2 r23, %83, r20;
}
{
sub.f16x2 r26, %91, %73;
}
{
mul.f16x2 r29, r26, r3;
}
{
add.f16x2 r32, r23, r29;
}
{
add.f16x2 r35, %74, %58;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %83, r38;
}
{
sub.f16x2 r44, %91, %73;
}
{
mul.f16x2 r47, r44, r3;
}
{
sub.f16x2 r50, r41, r47;
}
{
add.f16x2 r53, %91, %73;
}
{
mul.f16x2 r56, r53, r1;
}
{
add.f16x2 r59, %97, r56;
}
{
sub.f16x2 r62, %74, %58;
}
{
mul.f16x2 r65, r62, r3;
}
{
sub.f16x2 r68, r59, r65;
}
{
add.f16x2 r71, %91, %73;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %97, r74;
}
{
sub.f16x2 r80, %74, %58;
}
{
mul.f16x2 r83, r80, r3;
}
{
add.f16x2 r86, r77, r83;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r90, {low, high};
}
{
neg.f16x2 r91, r90;
}
{
add.f16x2 r93, %98, %81;
}
{
add.f16x2 r96, %106, r93;
}
{
add.f16x2 r99, %57, %95;
}
{
add.f16x2 r102, %63, r99;
}
{
add.f16x2 r105, %98, %81;
}
{
mul.f16x2 r108, r105, r89;
}
{
add.f16x2 r111, %106, r108;
}
{
sub.f16x2 r114, %57, %95;
}
{
mul.f16x2 r117, r114, r91;
}
{
add.f16x2 r120, r111, r117;
}
{
add.f16x2 r123, %98, %81;
}
{
mul.f16x2 r126, r123, r89;
}
{
add.f16x2 r129, %106, r126;
}
{
sub.f16x2 r132, %57, %95;
}
{
mul.f16x2 r135, r132, r91;
}
{
sub.f16x2 r138, r129, r135;
}
{
add.f16x2 r141, %57, %95;
}
{
mul.f16x2 r144, r141, r89;
}
{
add.f16x2 r147, %63, r144;
}
{
sub.f16x2 r150, %98, %81;
}
{
mul.f16x2 r153, r150, r91;
}
{
sub.f16x2 r156, r147, r153;
}
{
add.f16x2 r159, %57, %95;
}
{
mul.f16x2 r162, r159, r89;
}
{
add.f16x2 r165, %63, r162;
}
{
sub.f16x2 r168, %98, %81;
}
{
mul.f16x2 r171, r168, r91;
}
{
add.f16x2 r174, r165, r171;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r177, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r178, {low, high};
}
{
neg.f16x2 r179, r178;
}
{
add.f16x2 r181, %64, %103;
}
{
add.f16x2 r184, %72, r181;
}
{
add.f16x2 r187, %77, %61;
}
{
add.f16x2 r190, %88, r187;
}
{
add.f16x2 r193, %64, %103;
}
{
mul.f16x2 r196, r193, r177;
}
{
add.f16x2 r199, %72, r196;
}
{
sub.f16x2 r202, %77, %61;
}
{
mul.f16x2 r205, r202, r179;
}
{
add.f16x2 r208, r199, r205;
}
{
add.f16x2 r211, %64, %103;
}
{
mul.f16x2 r214, r211, r177;
}
{
add.f16x2 r217, %72, r214;
}
{
sub.f16x2 r220, %77, %61;
}
{
mul.f16x2 r223, r220, r179;
}
{
sub.f16x2 r226, r217, r223;
}
{
add.f16x2 r229, %77, %61;
}
{
mul.f16x2 r232, r229, r177;
}
{
add.f16x2 r235, %88, r232;
}
{
sub.f16x2 r238, %64, %103;
}
{
mul.f16x2 r241, r238, r179;
}
{
sub.f16x2 r244, r235, r241;
}
{
add.f16x2 r247, %77, %61;
}
{
mul.f16x2 r250, r247, r177;
}
{
add.f16x2 r253, %88, r250;
}
{
sub.f16x2 r256, %64, %103;
}
{
mul.f16x2 r259, r256, r179;
}
{
add.f16x2 r262, r253, r259;
}
mov.f32 f178, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r265, {low, high};
}
mov.f32 f180, 0fBF248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
mov.b32 r266, {low, high};
}
mov.f32 f190, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r267, {low, high};
}
mov.f32 f192, 0fBF7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f192;
cvt.rn.f16.f32 high, f192;
mov.b32 r268, {low, high};
}
mov.f32 f214, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r271, {low, high};
}
mov.f32 f216, 0fBEAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r272, {low, high};
}
{
mul.f16x2 r281, r120, r265;
}
{
mul.f16x2 r284, r156, r266;
}
{
sub.f16x2 r287, r281, r284;
}
{
mul.f16x2 r290, r120, r266;
}
{
fma.rn.f16x2 r293, r156, r265, r290;
}
{
mul.f16x2 r297, r208, r267;
}
{
mul.f16x2 r300, r244, r268;
}
{
sub.f16x2 r303, r297, r300;
}
{
mul.f16x2 r306, r208, r268;
}
{
fma.rn.f16x2 r309, r244, r267, r306;
}
{
mul.f16x2 r313, r138, r267;
}
{
mul.f16x2 r316, r174, r268;
}
{
sub.f16x2 r319, r313, r316;
}
{
mul.f16x2 r322, r138, r268;
}
{
fma.rn.f16x2 r325, r174, r267, r322;
}
{
mul.f16x2 r329, r226, r271;
}
{
mul.f16x2 r332, r262, r272;
}
{
sub.f16x2 r335, r329, r332;
}
{
mul.f16x2 r338, r226, r272;
}
{
fma.rn.f16x2 r341, r262, r271, r338;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r345, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r346, {low, high};
}
{
neg.f16x2 r347, r346;
}
{
add.f16x2 r349, r96, r184;
}
{
add.f16x2 r352, r8, r349;
}
{
add.f16x2 r355, r102, r190;
}
{
add.f16x2 r358, r14, r355;
}
{
add.f16x2 r361, r96, r184;
}
{
mul.f16x2 r364, r361, r345;
}
{
add.f16x2 r367, r8, r364;
}
{
sub.f16x2 r370, r102, r190;
}
{
mul.f16x2 r373, r370, r347;
}
{
add.f16x2 r376, r367, r373;
}
{
add.f16x2 r379, r96, r184;
}
{
mul.f16x2 r382, r379, r345;
}
{
add.f16x2 r385, r8, r382;
}
{
sub.f16x2 r388, r102, r190;
}
{
mul.f16x2 r391, r388, r347;
}
{
sub.f16x2 r394, r385, r391;
}
{
add.f16x2 r397, r102, r190;
}
{
mul.f16x2 r400, r397, r345;
}
{
add.f16x2 r403, r14, r400;
}
{
sub.f16x2 r406, r96, r184;
}
{
mul.f16x2 r409, r406, r347;
}
{
sub.f16x2 r412, r403, r409;
}
{
add.f16x2 r415, r102, r190;
}
{
mul.f16x2 r418, r415, r345;
}
{
add.f16x2 r421, r14, r418;
}
{
sub.f16x2 r424, r96, r184;
}
{
mul.f16x2 r427, r424, r347;
}
{
add.f16x2 r430, r421, r427;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r433, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r434, {low, high};
}
{
neg.f16x2 r435, r434;
}
{
add.f16x2 r437, r287, r303;
}
{
add.f16x2 r440, r32, r437;
}
{
add.f16x2 r443, r293, r309;
}
{
add.f16x2 r446, r68, r443;
}
{
add.f16x2 r449, r287, r303;
}
{
mul.f16x2 r452, r449, r433;
}
{
add.f16x2 r455, r32, r452;
}
{
sub.f16x2 r458, r293, r309;
}
{
mul.f16x2 r461, r458, r435;
}
{
add.f16x2 r464, r455, r461;
}
{
add.f16x2 r467, r287, r303;
}
{
mul.f16x2 r470, r467, r433;
}
{
add.f16x2 r473, r32, r470;
}
{
sub.f16x2 r476, r293, r309;
}
{
mul.f16x2 r479, r476, r435;
}
{
sub.f16x2 r482, r473, r479;
}
{
add.f16x2 r485, r293, r309;
}
{
mul.f16x2 r488, r485, r433;
}
{
add.f16x2 r491, r68, r488;
}
{
sub.f16x2 r494, r287, r303;
}
{
mul.f16x2 r497, r494, r435;
}
{
sub.f16x2 r500, r491, r497;
}
{
add.f16x2 r503, r293, r309;
}
{
mul.f16x2 r506, r503, r433;
}
{
add.f16x2 r509, r68, r506;
}
{
sub.f16x2 r512, r287, r303;
}
{
mul.f16x2 r515, r512, r435;
}
{
add.f16x2 r518, r509, r515;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r521, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r522, {low, high};
}
{
neg.f16x2 r523, r522;
}
{
add.f16x2 r525, r319, r335;
}
{
add.f16x2 r528, r50, r525;
}
{
add.f16x2 r531, r325, r341;
}
{
add.f16x2 r534, r86, r531;
}
{
add.f16x2 r537, r319, r335;
}
{
mul.f16x2 r540, r537, r521;
}
{
add.f16x2 r543, r50, r540;
}
{
sub.f16x2 r546, r325, r341;
}
{
mul.f16x2 r549, r546, r523;
}
{
add.f16x2 r552, r543, r549;
}
{
add.f16x2 r555, r319, r335;
}
{
mul.f16x2 r558, r555, r521;
}
{
add.f16x2 r561, r50, r558;
}
{
sub.f16x2 r564, r325, r341;
}
{
mul.f16x2 r567, r564, r523;
}
{
sub.f16x2 r570, r561, r567;
}
{
add.f16x2 r573, r325, r341;
}
{
mul.f16x2 r576, r573, r521;
}
{
add.f16x2 r579, r86, r576;
}
{
sub.f16x2 r582, r319, r335;
}
{
mul.f16x2 r585, r582, r523;
}
{
sub.f16x2 r588, r579, r585;
}
{
add.f16x2 r591, r325, r341;
}
{
mul.f16x2 r594, r591, r521;
}
{
add.f16x2 r597, r86, r594;
}
{
sub.f16x2 r600, r319, r335;
}
{
mul.f16x2 r603, r600, r523;
}
{
add.f16x2 r606, r597, r603;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r609, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r610, {low, high};
}
{
neg.f16x2 r611, r610;
}
{
add.f16x2 r613, %78, %60;
}
{
add.f16x2 r616, %86, r613;
}
{
add.f16x2 r619, %93, %76;
}
{
add.f16x2 r622, %100, r619;
}
{
add.f16x2 r625, %78, %60;
}
{
mul.f16x2 r628, r625, r609;
}
{
add.f16x2 r631, %86, r628;
}
{
sub.f16x2 r634, %93, %76;
}
{
mul.f16x2 r637, r634, r611;
}
{
add.f16x2 r640, r631, r637;
}
{
add.f16x2 r643, %78, %60;
}
{
mul.f16x2 r646, r643, r609;
}
{
add.f16x2 r649, %86, r646;
}
{
sub.f16x2 r652, %93, %76;
}
{
mul.f16x2 r655, r652, r611;
}
{
sub.f16x2 r658, r649, r655;
}
{
add.f16x2 r661, %93, %76;
}
{
mul.f16x2 r664, r661, r609;
}
{
add.f16x2 r667, %100, r664;
}
{
sub.f16x2 r670, %78, %60;
}
{
mul.f16x2 r673, r670, r611;
}
{
sub.f16x2 r676, r667, r673;
}
{
add.f16x2 r679, %93, %76;
}
{
mul.f16x2 r682, r679, r609;
}
{
add.f16x2 r685, %100, r682;
}
{
sub.f16x2 r688, %78, %60;
}
{
mul.f16x2 r691, r688, r611;
}
{
add.f16x2 r694, r685, r691;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r697, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r698, {low, high};
}
{
neg.f16x2 r699, r698;
}
{
add.f16x2 r701, %101, %85;
}
{
add.f16x2 r704, %108, r701;
}
{
add.f16x2 r707, %59, %99;
}
{
add.f16x2 r710, %68, r707;
}
{
add.f16x2 r713, %101, %85;
}
{
mul.f16x2 r716, r713, r697;
}
{
add.f16x2 r719, %108, r716;
}
{
sub.f16x2 r722, %59, %99;
}
{
mul.f16x2 r725, r722, r699;
}
{
add.f16x2 r728, r719, r725;
}
{
add.f16x2 r731, %101, %85;
}
{
mul.f16x2 r734, r731, r697;
}
{
add.f16x2 r737, %108, r734;
}
{
sub.f16x2 r740, %59, %99;
}
{
mul.f16x2 r743, r740, r699;
}
{
sub.f16x2 r746, r737, r743;
}
{
add.f16x2 r749, %59, %99;
}
{
mul.f16x2 r752, r749, r697;
}
{
add.f16x2 r755, %68, r752;
}
{
sub.f16x2 r758, %101, %85;
}
{
mul.f16x2 r761, r758, r699;
}
{
sub.f16x2 r764, r755, r761;
}
{
add.f16x2 r767, %59, %99;
}
{
mul.f16x2 r770, r767, r697;
}
{
add.f16x2 r773, %68, r770;
}
{
sub.f16x2 r776, %101, %85;
}
{
mul.f16x2 r779, r776, r699;
}
{
add.f16x2 r782, r773, r779;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r785, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r786, {low, high};
}
{
neg.f16x2 r787, r786;
}
{
add.f16x2 r789, %67, %107;
}
{
add.f16x2 r792, %75, r789;
}
{
add.f16x2 r795, %84, %66;
}
{
add.f16x2 r798, %92, r795;
}
{
add.f16x2 r801, %67, %107;
}
{
mul.f16x2 r804, r801, r785;
}
{
add.f16x2 r807, %75, r804;
}
{
sub.f16x2 r810, %84, %66;
}
{
mul.f16x2 r813, r810, r787;
}
{
add.f16x2 r816, r807, r813;
}
{
add.f16x2 r819, %67, %107;
}
{
mul.f16x2 r822, r819, r785;
}
{
add.f16x2 r825, %75, r822;
}
{
sub.f16x2 r828, %84, %66;
}
{
mul.f16x2 r831, r828, r787;
}
{
sub.f16x2 r834, r825, r831;
}
{
add.f16x2 r837, %84, %66;
}
{
mul.f16x2 r840, r837, r785;
}
{
add.f16x2 r843, %92, r840;
}
{
sub.f16x2 r846, %67, %107;
}
{
mul.f16x2 r849, r846, r787;
}
{
sub.f16x2 r852, r843, r849;
}
{
add.f16x2 r855, %84, %66;
}
{
mul.f16x2 r858, r855, r785;
}
{
add.f16x2 r861, %92, r858;
}
{
sub.f16x2 r864, %67, %107;
}
{
mul.f16x2 r867, r864, r787;
}
{
add.f16x2 r870, r861, r867;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r873, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
mov.b32 r874, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r875, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f192;
cvt.rn.f16.f32 high, f192;
mov.b32 r876, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r879, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r880, {low, high};
}
{
mul.f16x2 r889, r728, r873;
}
{
mul.f16x2 r892, r764, r874;
}
{
sub.f16x2 r895, r889, r892;
}
{
mul.f16x2 r898, r728, r874;
}
{
fma.rn.f16x2 r901, r764, r873, r898;
}
{
mul.f16x2 r905, r816, r875;
}
{
mul.f16x2 r908, r852, r876;
}
{
sub.f16x2 r911, r905, r908;
}
{
mul.f16x2 r914, r816, r876;
}
{
fma.rn.f16x2 r917, r852, r875, r914;
}
{
mul.f16x2 r921, r746, r875;
}
{
mul.f16x2 r924, r782, r876;
}
{
sub.f16x2 r927, r921, r924;
}
{
mul.f16x2 r930, r746, r876;
}
{
fma.rn.f16x2 r933, r782, r875, r930;
}
{
mul.f16x2 r937, r834, r879;
}
{
mul.f16x2 r940, r870, r880;
}
{
sub.f16x2 r943, r937, r940;
}
{
mul.f16x2 r946, r834, r880;
}
{
fma.rn.f16x2 r949, r870, r879, r946;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r953, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r954, {low, high};
}
{
neg.f16x2 r955, r954;
}
{
add.f16x2 r957, r704, r792;
}
{
add.f16x2 r960, r616, r957;
}
{
add.f16x2 r963, r710, r798;
}
{
add.f16x2 r966, r622, r963;
}
{
add.f16x2 r969, r704, r792;
}
{
mul.f16x2 r972, r969, r953;
}
{
add.f16x2 r975, r616, r972;
}
{
sub.f16x2 r978, r710, r798;
}
{
mul.f16x2 r981, r978, r955;
}
{
add.f16x2 r984, r975, r981;
}
{
add.f16x2 r987, r704, r792;
}
{
mul.f16x2 r990, r987, r953;
}
{
add.f16x2 r993, r616, r990;
}
{
sub.f16x2 r996, r710, r798;
}
{
mul.f16x2 r999, r996, r955;
}
{
sub.f16x2 r1002, r993, r999;
}
{
add.f16x2 r1005, r710, r798;
}
{
mul.f16x2 r1008, r1005, r953;
}
{
add.f16x2 r1011, r622, r1008;
}
{
sub.f16x2 r1014, r704, r792;
}
{
mul.f16x2 r1017, r1014, r955;
}
{
sub.f16x2 r1020, r1011, r1017;
}
{
add.f16x2 r1023, r710, r798;
}
{
mul.f16x2 r1026, r1023, r953;
}
{
add.f16x2 r1029, r622, r1026;
}
{
sub.f16x2 r1032, r704, r792;
}
{
mul.f16x2 r1035, r1032, r955;
}
{
add.f16x2 r1038, r1029, r1035;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1041, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1042, {low, high};
}
{
neg.f16x2 r1043, r1042;
}
{
add.f16x2 r1045, r895, r911;
}
{
add.f16x2 r1048, r640, r1045;
}
{
add.f16x2 r1051, r901, r917;
}
{
add.f16x2 r1054, r676, r1051;
}
{
add.f16x2 r1057, r895, r911;
}
{
mul.f16x2 r1060, r1057, r1041;
}
{
add.f16x2 r1063, r640, r1060;
}
{
sub.f16x2 r1066, r901, r917;
}
{
mul.f16x2 r1069, r1066, r1043;
}
{
add.f16x2 r1072, r1063, r1069;
}
{
add.f16x2 r1075, r895, r911;
}
{
mul.f16x2 r1078, r1075, r1041;
}
{
add.f16x2 r1081, r640, r1078;
}
{
sub.f16x2 r1084, r901, r917;
}
{
mul.f16x2 r1087, r1084, r1043;
}
{
sub.f16x2 r1090, r1081, r1087;
}
{
add.f16x2 r1093, r901, r917;
}
{
mul.f16x2 r1096, r1093, r1041;
}
{
add.f16x2 r1099, r676, r1096;
}
{
sub.f16x2 r1102, r895, r911;
}
{
mul.f16x2 r1105, r1102, r1043;
}
{
sub.f16x2 r1108, r1099, r1105;
}
{
add.f16x2 r1111, r901, r917;
}
{
mul.f16x2 r1114, r1111, r1041;
}
{
add.f16x2 r1117, r676, r1114;
}
{
sub.f16x2 r1120, r895, r911;
}
{
mul.f16x2 r1123, r1120, r1043;
}
{
add.f16x2 r1126, r1117, r1123;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1129, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1130, {low, high};
}
{
neg.f16x2 r1131, r1130;
}
{
add.f16x2 r1133, r927, r943;
}
{
add.f16x2 r1136, r658, r1133;
}
{
add.f16x2 r1139, r933, r949;
}
{
add.f16x2 r1142, r694, r1139;
}
{
add.f16x2 r1145, r927, r943;
}
{
mul.f16x2 r1148, r1145, r1129;
}
{
add.f16x2 r1151, r658, r1148;
}
{
sub.f16x2 r1154, r933, r949;
}
{
mul.f16x2 r1157, r1154, r1131;
}
{
add.f16x2 r1160, r1151, r1157;
}
{
add.f16x2 r1163, r927, r943;
}
{
mul.f16x2 r1166, r1163, r1129;
}
{
add.f16x2 r1169, r658, r1166;
}
{
sub.f16x2 r1172, r933, r949;
}
{
mul.f16x2 r1175, r1172, r1131;
}
{
sub.f16x2 r1178, r1169, r1175;
}
{
add.f16x2 r1181, r933, r949;
}
{
mul.f16x2 r1184, r1181, r1129;
}
{
add.f16x2 r1187, r694, r1184;
}
{
sub.f16x2 r1190, r927, r943;
}
{
mul.f16x2 r1193, r1190, r1131;
}
{
sub.f16x2 r1196, r1187, r1193;
}
{
add.f16x2 r1199, r933, r949;
}
{
mul.f16x2 r1202, r1199, r1129;
}
{
add.f16x2 r1205, r694, r1202;
}
{
sub.f16x2 r1208, r927, r943;
}
{
mul.f16x2 r1211, r1208, r1131;
}
{
add.f16x2 r1214, r1205, r1211;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1217, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1218, {low, high};
}
{
neg.f16x2 r1219, r1218;
}
{
add.f16x2 r1221, %82, %65;
}
{
add.f16x2 r1224, %90, r1221;
}
{
add.f16x2 r1227, %96, %80;
}
{
add.f16x2 r1230, %104, r1227;
}
{
add.f16x2 r1233, %82, %65;
}
{
mul.f16x2 r1236, r1233, r1217;
}
{
add.f16x2 r1239, %90, r1236;
}
{
sub.f16x2 r1242, %96, %80;
}
{
mul.f16x2 r1245, r1242, r1219;
}
{
add.f16x2 r1248, r1239, r1245;
}
{
add.f16x2 r1251, %82, %65;
}
{
mul.f16x2 r1254, r1251, r1217;
}
{
add.f16x2 r1257, %90, r1254;
}
{
sub.f16x2 r1260, %96, %80;
}
{
mul.f16x2 r1263, r1260, r1219;
}
{
sub.f16x2 r1266, r1257, r1263;
}
{
add.f16x2 r1269, %96, %80;
}
{
mul.f16x2 r1272, r1269, r1217;
}
{
add.f16x2 r1275, %104, r1272;
}
{
sub.f16x2 r1278, %82, %65;
}
{
mul.f16x2 r1281, r1278, r1219;
}
{
sub.f16x2 r1284, r1275, r1281;
}
{
add.f16x2 r1287, %96, %80;
}
{
mul.f16x2 r1290, r1287, r1217;
}
{
add.f16x2 r1293, %104, r1290;
}
{
sub.f16x2 r1296, %82, %65;
}
{
mul.f16x2 r1299, r1296, r1219;
}
{
add.f16x2 r1302, r1293, r1299;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1305, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1306, {low, high};
}
{
neg.f16x2 r1307, r1306;
}
{
add.f16x2 r1309, %105, %89;
}
{
add.f16x2 r1312, %56, r1309;
}
{
add.f16x2 r1315, %62, %102;
}
{
add.f16x2 r1318, %70, r1315;
}
{
add.f16x2 r1321, %105, %89;
}
{
mul.f16x2 r1324, r1321, r1305;
}
{
add.f16x2 r1327, %56, r1324;
}
{
sub.f16x2 r1330, %62, %102;
}
{
mul.f16x2 r1333, r1330, r1307;
}
{
add.f16x2 r1336, r1327, r1333;
}
{
add.f16x2 r1339, %105, %89;
}
{
mul.f16x2 r1342, r1339, r1305;
}
{
add.f16x2 r1345, %56, r1342;
}
{
sub.f16x2 r1348, %62, %102;
}
{
mul.f16x2 r1351, r1348, r1307;
}
{
sub.f16x2 r1354, r1345, r1351;
}
{
add.f16x2 r1357, %62, %102;
}
{
mul.f16x2 r1360, r1357, r1305;
}
{
add.f16x2 r1363, %70, r1360;
}
{
sub.f16x2 r1366, %105, %89;
}
{
mul.f16x2 r1369, r1366, r1307;
}
{
sub.f16x2 r1372, r1363, r1369;
}
{
add.f16x2 r1375, %62, %102;
}
{
mul.f16x2 r1378, r1375, r1305;
}
{
add.f16x2 r1381, %70, r1378;
}
{
sub.f16x2 r1384, %105, %89;
}
{
mul.f16x2 r1387, r1384, r1307;
}
{
add.f16x2 r1390, r1381, r1387;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1393, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1394, {low, high};
}
{
neg.f16x2 r1395, r1394;
}
{
add.f16x2 r1397, %71, %55;
}
{
add.f16x2 r1400, %79, r1397;
}
{
add.f16x2 r1403, %87, %69;
}
{
add.f16x2 r1406, %94, r1403;
}
{
add.f16x2 r1409, %71, %55;
}
{
mul.f16x2 r1412, r1409, r1393;
}
{
add.f16x2 r1415, %79, r1412;
}
{
sub.f16x2 r1418, %87, %69;
}
{
mul.f16x2 r1421, r1418, r1395;
}
{
add.f16x2 r1424, r1415, r1421;
}
{
add.f16x2 r1427, %71, %55;
}
{
mul.f16x2 r1430, r1427, r1393;
}
{
add.f16x2 r1433, %79, r1430;
}
{
sub.f16x2 r1436, %87, %69;
}
{
mul.f16x2 r1439, r1436, r1395;
}
{
sub.f16x2 r1442, r1433, r1439;
}
{
add.f16x2 r1445, %87, %69;
}
{
mul.f16x2 r1448, r1445, r1393;
}
{
add.f16x2 r1451, %94, r1448;
}
{
sub.f16x2 r1454, %71, %55;
}
{
mul.f16x2 r1457, r1454, r1395;
}
{
sub.f16x2 r1460, r1451, r1457;
}
{
add.f16x2 r1463, %87, %69;
}
{
mul.f16x2 r1466, r1463, r1393;
}
{
add.f16x2 r1469, %94, r1466;
}
{
sub.f16x2 r1472, %71, %55;
}
{
mul.f16x2 r1475, r1472, r1395;
}
{
add.f16x2 r1478, r1469, r1475;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r1481, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
mov.b32 r1482, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r1483, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f192;
cvt.rn.f16.f32 high, f192;
mov.b32 r1484, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1487, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1488, {low, high};
}
{
mul.f16x2 r1497, r1336, r1481;
}
{
mul.f16x2 r1500, r1372, r1482;
}
{
sub.f16x2 r1503, r1497, r1500;
}
{
mul.f16x2 r1506, r1336, r1482;
}
{
fma.rn.f16x2 r1509, r1372, r1481, r1506;
}
{
mul.f16x2 r1513, r1424, r1483;
}
{
mul.f16x2 r1516, r1460, r1484;
}
{
sub.f16x2 r1519, r1513, r1516;
}
{
mul.f16x2 r1522, r1424, r1484;
}
{
fma.rn.f16x2 r1525, r1460, r1483, r1522;
}
{
mul.f16x2 r1529, r1354, r1483;
}
{
mul.f16x2 r1532, r1390, r1484;
}
{
sub.f16x2 r1535, r1529, r1532;
}
{
mul.f16x2 r1538, r1354, r1484;
}
{
fma.rn.f16x2 r1541, r1390, r1483, r1538;
}
{
mul.f16x2 r1545, r1442, r1487;
}
{
mul.f16x2 r1548, r1478, r1488;
}
{
sub.f16x2 r1551, r1545, r1548;
}
{
mul.f16x2 r1554, r1442, r1488;
}
{
fma.rn.f16x2 r1557, r1478, r1487, r1554;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1561, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1562, {low, high};
}
{
neg.f16x2 r1563, r1562;
}
{
add.f16x2 r1565, r1312, r1400;
}
{
add.f16x2 r1568, r1224, r1565;
}
{
add.f16x2 r1571, r1318, r1406;
}
{
add.f16x2 r1574, r1230, r1571;
}
{
add.f16x2 r1577, r1312, r1400;
}
{
mul.f16x2 r1580, r1577, r1561;
}
{
add.f16x2 r1583, r1224, r1580;
}
{
sub.f16x2 r1586, r1318, r1406;
}
{
mul.f16x2 r1589, r1586, r1563;
}
{
add.f16x2 r1592, r1583, r1589;
}
{
add.f16x2 r1595, r1312, r1400;
}
{
mul.f16x2 r1598, r1595, r1561;
}
{
add.f16x2 r1601, r1224, r1598;
}
{
sub.f16x2 r1604, r1318, r1406;
}
{
mul.f16x2 r1607, r1604, r1563;
}
{
sub.f16x2 r1610, r1601, r1607;
}
{
add.f16x2 r1613, r1318, r1406;
}
{
mul.f16x2 r1616, r1613, r1561;
}
{
add.f16x2 r1619, r1230, r1616;
}
{
sub.f16x2 r1622, r1312, r1400;
}
{
mul.f16x2 r1625, r1622, r1563;
}
{
sub.f16x2 r1628, r1619, r1625;
}
{
add.f16x2 r1631, r1318, r1406;
}
{
mul.f16x2 r1634, r1631, r1561;
}
{
add.f16x2 r1637, r1230, r1634;
}
{
sub.f16x2 r1640, r1312, r1400;
}
{
mul.f16x2 r1643, r1640, r1563;
}
{
add.f16x2 r1646, r1637, r1643;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1649, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1650, {low, high};
}
{
neg.f16x2 r1651, r1650;
}
{
add.f16x2 r1653, r1503, r1519;
}
{
add.f16x2 r1656, r1248, r1653;
}
{
add.f16x2 r1659, r1509, r1525;
}
{
add.f16x2 r1662, r1284, r1659;
}
{
add.f16x2 r1665, r1503, r1519;
}
{
mul.f16x2 r1668, r1665, r1649;
}
{
add.f16x2 r1671, r1248, r1668;
}
{
sub.f16x2 r1674, r1509, r1525;
}
{
mul.f16x2 r1677, r1674, r1651;
}
{
add.f16x2 r1680, r1671, r1677;
}
{
add.f16x2 r1683, r1503, r1519;
}
{
mul.f16x2 r1686, r1683, r1649;
}
{
add.f16x2 r1689, r1248, r1686;
}
{
sub.f16x2 r1692, r1509, r1525;
}
{
mul.f16x2 r1695, r1692, r1651;
}
{
sub.f16x2 r1698, r1689, r1695;
}
{
add.f16x2 r1701, r1509, r1525;
}
{
mul.f16x2 r1704, r1701, r1649;
}
{
add.f16x2 r1707, r1284, r1704;
}
{
sub.f16x2 r1710, r1503, r1519;
}
{
mul.f16x2 r1713, r1710, r1651;
}
{
sub.f16x2 r1716, r1707, r1713;
}
{
add.f16x2 r1719, r1509, r1525;
}
{
mul.f16x2 r1722, r1719, r1649;
}
{
add.f16x2 r1725, r1284, r1722;
}
{
sub.f16x2 r1728, r1503, r1519;
}
{
mul.f16x2 r1731, r1728, r1651;
}
{
add.f16x2 r1734, r1725, r1731;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r1737, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r1738, {low, high};
}
{
neg.f16x2 r1739, r1738;
}
{
add.f16x2 r1741, r1535, r1551;
}
{
add.f16x2 r1744, r1266, r1741;
}
{
add.f16x2 r1747, r1541, r1557;
}
{
add.f16x2 r1750, r1302, r1747;
}
{
add.f16x2 r1753, r1535, r1551;
}
{
mul.f16x2 r1756, r1753, r1737;
}
{
add.f16x2 r1759, r1266, r1756;
}
{
sub.f16x2 r1762, r1541, r1557;
}
{
mul.f16x2 r1765, r1762, r1739;
}
{
add.f16x2 r1768, r1759, r1765;
}
{
add.f16x2 r1771, r1535, r1551;
}
{
mul.f16x2 r1774, r1771, r1737;
}
{
add.f16x2 r1777, r1266, r1774;
}
{
sub.f16x2 r1780, r1541, r1557;
}
{
mul.f16x2 r1783, r1780, r1739;
}
{
sub.f16x2 r1786, r1777, r1783;
}
{
add.f16x2 r1789, r1541, r1557;
}
{
mul.f16x2 r1792, r1789, r1737;
}
{
add.f16x2 r1795, r1302, r1792;
}
{
sub.f16x2 r1798, r1535, r1551;
}
{
mul.f16x2 r1801, r1798, r1739;
}
{
sub.f16x2 r1804, r1795, r1801;
}
{
add.f16x2 r1807, r1541, r1557;
}
{
mul.f16x2 r1810, r1807, r1737;
}
{
add.f16x2 r1813, r1302, r1810;
}
{
sub.f16x2 r1816, r1535, r1551;
}
{
mul.f16x2 r1819, r1816, r1739;
}
{
add.f16x2 r1822, r1813, r1819;
}
mov.f32 f170, 0f3F791978;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f170;
cvt.rn.f16.f32 high, f170;
mov.b32 r1825, {low, high};
}
mov.f32 f172, 0fBE6C2691;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f172;
cvt.rn.f16.f32 high, f172;
mov.b32 r1826, {low, high};
}
mov.f32 f174, 0f3F64C51C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f174;
cvt.rn.f16.f32 high, f174;
mov.b32 r1827, {low, high};
}
mov.f32 f176, 0fBEE5C902;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f176;
cvt.rn.f16.f32 high, f176;
mov.b32 r1828, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r1829, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
mov.b32 r1830, {low, high};
}
mov.f32 f182, 0f3F18DF63;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f182;
cvt.rn.f16.f32 high, f182;
mov.b32 r1831, {low, high};
}
mov.f32 f184, 0fBF4D57F2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f184;
cvt.rn.f16.f32 high, f184;
mov.b32 r1832, {low, high};
}
mov.f32 f186, 0f3ECACAF8;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f186;
cvt.rn.f16.f32 high, f186;
mov.b32 r1833, {low, high};
}
mov.f32 f188, 0fBF6B1036;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f188;
cvt.rn.f16.f32 high, f188;
mov.b32 r1834, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r1835, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f192;
cvt.rn.f16.f32 high, f192;
mov.b32 r1836, {low, high};
}
mov.f32 f194, 0fBD6E2946;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1837, {low, high};
}
mov.f32 f196, 0fBF7F9120;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r1838, {low, high};
}
mov.f32 f198, 0fBE92D7E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1839, {low, high};
}
mov.f32 f200, 0fBF753ECD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1840, {low, high};
}
mov.f32 f206, 0fBF2FAD88;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f206;
cvt.rn.f16.f32 high, f206;
mov.b32 r1843, {low, high};
}
mov.f32 f208, 0fBF3A3529;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f208;
cvt.rn.f16.f32 high, f208;
mov.b32 r1844, {low, high};
}
mov.f32 f230, 0fBF55E287;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1847, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1848, {low, high};
}
mov.f32 f222, 0fBF7E44DE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f222;
cvt.rn.f16.f32 high, f222;
mov.b32 r1851, {low, high};
}
mov.f32 f224, 0f3DEDC21F;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f224;
cvt.rn.f16.f32 high, f224;
mov.b32 r1852, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f230;
cvt.rn.f16.f32 high, f230;
mov.b32 r1855, {low, high};
}
mov.f32 f232, 0f3F0CAC9F;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f232;
cvt.rn.f16.f32 high, f232;
mov.b32 r1856, {low, high};
}
{
mul.f16x2 r1877, r1048, r1825;
}
{
mul.f16x2 r1880, r1054, r1826;
}
{
sub.f16x2 r1883, r1877, r1880;
}
{
mul.f16x2 r1886, r1048, r1826;
}
{
fma.rn.f16x2 r1889, r1054, r1825, r1886;
}
{
mul.f16x2 r1893, r1656, r1827;
}
{
mul.f16x2 r1896, r1662, r1828;
}
{
sub.f16x2 r1899, r1893, r1896;
}
{
mul.f16x2 r1902, r1656, r1828;
}
{
fma.rn.f16x2 r1905, r1662, r1827, r1902;
}
{
mul.f16x2 r1909, r1136, r1827;
}
{
mul.f16x2 r1912, r1142, r1828;
}
{
sub.f16x2 r1915, r1909, r1912;
}
{
mul.f16x2 r1918, r1136, r1828;
}
{
fma.rn.f16x2 r1921, r1142, r1827, r1918;
}
{
mul.f16x2 r1925, r1744, r1831;
}
{
mul.f16x2 r1928, r1750, r1832;
}
{
sub.f16x2 r1931, r1925, r1928;
}
{
mul.f16x2 r1934, r1744, r1832;
}
{
fma.rn.f16x2 r1937, r1750, r1831, r1934;
}
{
mul.f16x2 r1941, r984, r1829;
}
{
mul.f16x2 r1944, r1020, r1830;
}
{
sub.f16x2 r1947, r1941, r1944;
}
{
mul.f16x2 r1950, r984, r1830;
}
{
fma.rn.f16x2 r1953, r1020, r1829, r1950;
}
{
mul.f16x2 r1957, r1592, r1835;
}
{
mul.f16x2 r1960, r1628, r1836;
}
{
sub.f16x2 r1963, r1957, r1960;
}
{
mul.f16x2 r1966, r1592, r1836;
}
{
fma.rn.f16x2 r1969, r1628, r1835, r1966;
}
{
mul.f16x2 r1973, r1072, r1831;
}
{
mul.f16x2 r1976, r1108, r1832;
}
{
sub.f16x2 r1979, r1973, r1976;
}
{
mul.f16x2 r1982, r1072, r1832;
}
{
fma.rn.f16x2 r1985, r1108, r1831, r1982;
}
{
mul.f16x2 r1989, r1680, r1839;
}
{
mul.f16x2 r1992, r1716, r1840;
}
{
sub.f16x2 r1995, r1989, r1992;
}
{
mul.f16x2 r1998, r1680, r1840;
}
{
fma.rn.f16x2 r2001, r1716, r1839, r1998;
}
{
mul.f16x2 r2005, r1160, r1833;
}
{
mul.f16x2 r2008, r1196, r1834;
}
{
sub.f16x2 r2011, r2005, r2008;
}
{
mul.f16x2 r2014, r1160, r1834;
}
{
fma.rn.f16x2 r2017, r1196, r1833, r2014;
}
{
mul.f16x2 r2021, r1768, r1843;
}
{
mul.f16x2 r2024, r1804, r1844;
}
{
sub.f16x2 r2027, r2021, r2024;
}
{
mul.f16x2 r2030, r1768, r1844;
}
{
fma.rn.f16x2 r2033, r1804, r1843, r2030;
}
{
mul.f16x2 r2037, r1002, r1835;
}
{
mul.f16x2 r2040, r1038, r1836;
}
{
sub.f16x2 r2043, r2037, r2040;
}
{
mul.f16x2 r2046, r1002, r1836;
}
{
fma.rn.f16x2 r2049, r1038, r1835, r2046;
}
{
mul.f16x2 r2053, r1610, r1847;
}
{
mul.f16x2 r2056, r1646, r1848;
}
{
sub.f16x2 r2059, r2053, r2056;
}
{
mul.f16x2 r2062, r1610, r1848;
}
{
fma.rn.f16x2 r2065, r1646, r1847, r2062;
}
{
mul.f16x2 r2069, r1090, r1837;
}
{
mul.f16x2 r2072, r1126, r1838;
}
{
sub.f16x2 r2075, r2069, r2072;
}
{
mul.f16x2 r2078, r1090, r1838;
}
{
fma.rn.f16x2 r2081, r1126, r1837, r2078;
}
{
mul.f16x2 r2085, r1698, r1851;
}
{
mul.f16x2 r2088, r1734, r1852;
}
{
sub.f16x2 r2091, r2085, r2088;
}
{
mul.f16x2 r2094, r1698, r1852;
}
{
fma.rn.f16x2 r2097, r1734, r1851, r2094;
}
{
mul.f16x2 r2101, r1178, r1839;
}
{
mul.f16x2 r2104, r1214, r1840;
}
{
sub.f16x2 r2107, r2101, r2104;
}
{
mul.f16x2 r2110, r1178, r1840;
}
{
fma.rn.f16x2 r2113, r1214, r1839, r2110;
}
{
mul.f16x2 r2117, r1786, r1855;
}
{
mul.f16x2 r2120, r1822, r1856;
}
{
sub.f16x2 r2123, r2117, r2120;
}
{
mul.f16x2 r2126, r1786, r1856;
}
{
fma.rn.f16x2 r2129, r1822, r1855, r2126;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2133, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2134, {low, high};
}
{
neg.f16x2 r2135, r2134;
}
{
add.f16x2 r2137, r960, r1568;
}
{
add.f16x2 r2140, r352, r2137;
}
{
add.f16x2 r2143, r966, r1574;
}
{
add.f16x2 r2146, r358, r2143;
}
{
add.f16x2 r2149, r960, r1568;
}
{
mul.f16x2 r2152, r2149, r2133;
}
{
add.f16x2 r2155, r352, r2152;
}
{
sub.f16x2 r2158, r966, r1574;
}
{
mul.f16x2 r2161, r2158, r2135;
}
{
add.f16x2 r2164, r2155, r2161;
}
{
add.f16x2 r2167, r960, r1568;
}
{
mul.f16x2 r2170, r2167, r2133;
}
{
add.f16x2 r2173, r352, r2170;
}
{
sub.f16x2 r2176, r966, r1574;
}
{
mul.f16x2 r2179, r2176, r2135;
}
{
sub.f16x2 r2182, r2173, r2179;
}
{
add.f16x2 r2185, r966, r1574;
}
{
mul.f16x2 r2188, r2185, r2133;
}
{
add.f16x2 r2191, r358, r2188;
}
{
sub.f16x2 r2194, r960, r1568;
}
{
mul.f16x2 r2197, r2194, r2135;
}
{
sub.f16x2 r2200, r2191, r2197;
}
{
add.f16x2 r2203, r966, r1574;
}
{
mul.f16x2 r2206, r2203, r2133;
}
{
add.f16x2 r2209, r358, r2206;
}
{
sub.f16x2 r2212, r960, r1568;
}
{
mul.f16x2 r2215, r2212, r2135;
}
{
add.f16x2 r2218, r2209, r2215;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2221, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2222, {low, high};
}
{
neg.f16x2 r2223, r2222;
}
{
add.f16x2 r2225, r1883, r1899;
}
{
add.f16x2 r2228, r440, r2225;
}
{
add.f16x2 r2231, r1889, r1905;
}
{
add.f16x2 r2234, r446, r2231;
}
{
add.f16x2 r2237, r1883, r1899;
}
{
mul.f16x2 r2240, r2237, r2221;
}
{
add.f16x2 r2243, r440, r2240;
}
{
sub.f16x2 r2246, r1889, r1905;
}
{
mul.f16x2 r2249, r2246, r2223;
}
{
add.f16x2 r2252, r2243, r2249;
}
{
add.f16x2 r2255, r1883, r1899;
}
{
mul.f16x2 r2258, r2255, r2221;
}
{
add.f16x2 r2261, r440, r2258;
}
{
sub.f16x2 r2264, r1889, r1905;
}
{
mul.f16x2 r2267, r2264, r2223;
}
{
sub.f16x2 r2270, r2261, r2267;
}
{
add.f16x2 r2273, r1889, r1905;
}
{
mul.f16x2 r2276, r2273, r2221;
}
{
add.f16x2 r2279, r446, r2276;
}
{
sub.f16x2 r2282, r1883, r1899;
}
{
mul.f16x2 r2285, r2282, r2223;
}
{
sub.f16x2 r2288, r2279, r2285;
}
{
add.f16x2 r2291, r1889, r1905;
}
{
mul.f16x2 r2294, r2291, r2221;
}
{
add.f16x2 r2297, r446, r2294;
}
{
sub.f16x2 r2300, r1883, r1899;
}
{
mul.f16x2 r2303, r2300, r2223;
}
{
add.f16x2 r2306, r2297, r2303;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2309, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2310, {low, high};
}
{
neg.f16x2 r2311, r2310;
}
{
add.f16x2 r2313, r1915, r1931;
}
{
add.f16x2 r2316, r528, r2313;
}
{
add.f16x2 r2319, r1921, r1937;
}
{
add.f16x2 r2322, r534, r2319;
}
{
add.f16x2 r2325, r1915, r1931;
}
{
mul.f16x2 r2328, r2325, r2309;
}
{
add.f16x2 r2331, r528, r2328;
}
{
sub.f16x2 r2334, r1921, r1937;
}
{
mul.f16x2 r2337, r2334, r2311;
}
{
add.f16x2 r2340, r2331, r2337;
}
{
add.f16x2 r2343, r1915, r1931;
}
{
mul.f16x2 r2346, r2343, r2309;
}
{
add.f16x2 r2349, r528, r2346;
}
{
sub.f16x2 r2352, r1921, r1937;
}
{
mul.f16x2 r2355, r2352, r2311;
}
{
sub.f16x2 r2358, r2349, r2355;
}
{
add.f16x2 r2361, r1921, r1937;
}
{
mul.f16x2 r2364, r2361, r2309;
}
{
add.f16x2 r2367, r534, r2364;
}
{
sub.f16x2 r2370, r1915, r1931;
}
{
mul.f16x2 r2373, r2370, r2311;
}
{
sub.f16x2 r2376, r2367, r2373;
}
{
add.f16x2 r2379, r1921, r1937;
}
{
mul.f16x2 r2382, r2379, r2309;
}
{
add.f16x2 r2385, r534, r2382;
}
{
sub.f16x2 r2388, r1915, r1931;
}
{
mul.f16x2 r2391, r2388, r2311;
}
{
add.f16x2 r2394, r2385, r2391;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2398, {low, high};
}
{
neg.f16x2 r2399, r2398;
}
{
add.f16x2 r2401, r1947, r1963;
}
{
add.f16x2 r2404, r376, r2401;
}
{
add.f16x2 r2407, r1953, r1969;
}
{
add.f16x2 r2410, r412, r2407;
}
{
add.f16x2 r2413, r1947, r1963;
}
{
mul.f16x2 r2416, r2413, r2397;
}
{
add.f16x2 r2419, r376, r2416;
}
{
sub.f16x2 r2422, r1953, r1969;
}
{
mul.f16x2 r2425, r2422, r2399;
}
{
add.f16x2 r2428, r2419, r2425;
}
{
add.f16x2 r2431, r1947, r1963;
}
{
mul.f16x2 r2434, r2431, r2397;
}
{
add.f16x2 r2437, r376, r2434;
}
{
sub.f16x2 r2440, r1953, r1969;
}
{
mul.f16x2 r2443, r2440, r2399;
}
{
sub.f16x2 r2446, r2437, r2443;
}
{
add.f16x2 r2449, r1953, r1969;
}
{
mul.f16x2 r2452, r2449, r2397;
}
{
add.f16x2 r2455, r412, r2452;
}
{
sub.f16x2 r2458, r1947, r1963;
}
{
mul.f16x2 r2461, r2458, r2399;
}
{
sub.f16x2 r2464, r2455, r2461;
}
{
add.f16x2 r2467, r1953, r1969;
}
{
mul.f16x2 r2470, r2467, r2397;
}
{
add.f16x2 r2473, r412, r2470;
}
{
sub.f16x2 r2476, r1947, r1963;
}
{
mul.f16x2 r2479, r2476, r2399;
}
{
add.f16x2 r2482, r2473, r2479;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2485, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2486, {low, high};
}
{
neg.f16x2 r2487, r2486;
}
{
add.f16x2 r2489, r1979, r1995;
}
{
add.f16x2 r2492, r464, r2489;
}
{
add.f16x2 r2495, r1985, r2001;
}
{
add.f16x2 r2498, r500, r2495;
}
{
add.f16x2 r2501, r1979, r1995;
}
{
mul.f16x2 r2504, r2501, r2485;
}
{
add.f16x2 r2507, r464, r2504;
}
{
sub.f16x2 r2510, r1985, r2001;
}
{
mul.f16x2 r2513, r2510, r2487;
}
{
add.f16x2 r2516, r2507, r2513;
}
{
add.f16x2 r2519, r1979, r1995;
}
{
mul.f16x2 r2522, r2519, r2485;
}
{
add.f16x2 r2525, r464, r2522;
}
{
sub.f16x2 r2528, r1985, r2001;
}
{
mul.f16x2 r2531, r2528, r2487;
}
{
sub.f16x2 r2534, r2525, r2531;
}
{
add.f16x2 r2537, r1985, r2001;
}
{
mul.f16x2 r2540, r2537, r2485;
}
{
add.f16x2 r2543, r500, r2540;
}
{
sub.f16x2 r2546, r1979, r1995;
}
{
mul.f16x2 r2549, r2546, r2487;
}
{
sub.f16x2 r2552, r2543, r2549;
}
{
add.f16x2 r2555, r1985, r2001;
}
{
mul.f16x2 r2558, r2555, r2485;
}
{
add.f16x2 r2561, r500, r2558;
}
{
sub.f16x2 r2564, r1979, r1995;
}
{
mul.f16x2 r2567, r2564, r2487;
}
{
add.f16x2 r2570, r2561, r2567;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2573, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2574, {low, high};
}
{
neg.f16x2 r2575, r2574;
}
{
add.f16x2 r2577, r2011, r2027;
}
{
add.f16x2 r2580, r552, r2577;
}
{
add.f16x2 r2583, r2017, r2033;
}
{
add.f16x2 r2586, r588, r2583;
}
{
add.f16x2 r2589, r2011, r2027;
}
{
mul.f16x2 r2592, r2589, r2573;
}
{
add.f16x2 r2595, r552, r2592;
}
{
sub.f16x2 r2598, r2017, r2033;
}
{
mul.f16x2 r2601, r2598, r2575;
}
{
add.f16x2 r2604, r2595, r2601;
}
{
add.f16x2 r2607, r2011, r2027;
}
{
mul.f16x2 r2610, r2607, r2573;
}
{
add.f16x2 r2613, r552, r2610;
}
{
sub.f16x2 r2616, r2017, r2033;
}
{
mul.f16x2 r2619, r2616, r2575;
}
{
sub.f16x2 r2622, r2613, r2619;
}
{
add.f16x2 r2625, r2017, r2033;
}
{
mul.f16x2 r2628, r2625, r2573;
}
{
add.f16x2 r2631, r588, r2628;
}
{
sub.f16x2 r2634, r2011, r2027;
}
{
mul.f16x2 r2637, r2634, r2575;
}
{
sub.f16x2 r2640, r2631, r2637;
}
{
add.f16x2 r2643, r2017, r2033;
}
{
mul.f16x2 r2646, r2643, r2573;
}
{
add.f16x2 r2649, r588, r2646;
}
{
sub.f16x2 r2652, r2011, r2027;
}
{
mul.f16x2 r2655, r2652, r2575;
}
{
add.f16x2 r2658, r2649, r2655;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2661, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2662, {low, high};
}
{
neg.f16x2 r2663, r2662;
}
{
add.f16x2 r2665, r2043, r2059;
}
{
add.f16x2 r2668, r394, r2665;
}
{
add.f16x2 r2671, r2049, r2065;
}
{
add.f16x2 r2674, r430, r2671;
}
{
add.f16x2 r2677, r2043, r2059;
}
{
mul.f16x2 r2680, r2677, r2661;
}
{
add.f16x2 r2683, r394, r2680;
}
{
sub.f16x2 r2686, r2049, r2065;
}
{
mul.f16x2 r2689, r2686, r2663;
}
{
add.f16x2 r2692, r2683, r2689;
}
{
add.f16x2 r2695, r2043, r2059;
}
{
mul.f16x2 r2698, r2695, r2661;
}
{
add.f16x2 r2701, r394, r2698;
}
{
sub.f16x2 r2704, r2049, r2065;
}
{
mul.f16x2 r2707, r2704, r2663;
}
{
sub.f16x2 r2710, r2701, r2707;
}
{
add.f16x2 r2713, r2049, r2065;
}
{
mul.f16x2 r2716, r2713, r2661;
}
{
add.f16x2 r2719, r430, r2716;
}
{
sub.f16x2 r2722, r2043, r2059;
}
{
mul.f16x2 r2725, r2722, r2663;
}
{
sub.f16x2 r2728, r2719, r2725;
}
{
add.f16x2 r2731, r2049, r2065;
}
{
mul.f16x2 r2734, r2731, r2661;
}
{
add.f16x2 r2737, r430, r2734;
}
{
sub.f16x2 r2740, r2043, r2059;
}
{
mul.f16x2 r2743, r2740, r2663;
}
{
add.f16x2 r2746, r2737, r2743;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2749, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2750, {low, high};
}
{
neg.f16x2 r2751, r2750;
}
{
add.f16x2 r2753, r2075, r2091;
}
{
add.f16x2 r2756, r482, r2753;
}
{
add.f16x2 r2759, r2081, r2097;
}
{
add.f16x2 r2762, r518, r2759;
}
{
add.f16x2 r2765, r2075, r2091;
}
{
mul.f16x2 r2768, r2765, r2749;
}
{
add.f16x2 r2771, r482, r2768;
}
{
sub.f16x2 r2774, r2081, r2097;
}
{
mul.f16x2 r2777, r2774, r2751;
}
{
add.f16x2 r2780, r2771, r2777;
}
{
add.f16x2 r2783, r2075, r2091;
}
{
mul.f16x2 r2786, r2783, r2749;
}
{
add.f16x2 r2789, r482, r2786;
}
{
sub.f16x2 r2792, r2081, r2097;
}
{
mul.f16x2 r2795, r2792, r2751;
}
{
sub.f16x2 r2798, r2789, r2795;
}
{
add.f16x2 r2801, r2081, r2097;
}
{
mul.f16x2 r2804, r2801, r2749;
}
{
add.f16x2 r2807, r518, r2804;
}
{
sub.f16x2 r2810, r2075, r2091;
}
{
mul.f16x2 r2813, r2810, r2751;
}
{
sub.f16x2 r2816, r2807, r2813;
}
{
add.f16x2 r2819, r2081, r2097;
}
{
mul.f16x2 r2822, r2819, r2749;
}
{
add.f16x2 r2825, r518, r2822;
}
{
sub.f16x2 r2828, r2075, r2091;
}
{
mul.f16x2 r2831, r2828, r2751;
}
{
add.f16x2 r2834, r2825, r2831;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r2837, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r2838, {low, high};
}
{
neg.f16x2 r2839, r2838;
}
{
add.f16x2 r2841, r2107, r2123;
}
{
add.f16x2 r2844, r570, r2841;
}
{
add.f16x2 r2847, r2113, r2129;
}
{
add.f16x2 r2850, r606, r2847;
}
{
add.f16x2 r2853, r2107, r2123;
}
{
mul.f16x2 r2856, r2853, r2837;
}
{
add.f16x2 r2859, r570, r2856;
}
{
sub.f16x2 r2862, r2113, r2129;
}
{
mul.f16x2 r2865, r2862, r2839;
}
{
add.f16x2 r2868, r2859, r2865;
}
{
add.f16x2 r2871, r2107, r2123;
}
{
mul.f16x2 r2874, r2871, r2837;
}
{
add.f16x2 r2877, r570, r2874;
}
{
sub.f16x2 r2880, r2113, r2129;
}
{
mul.f16x2 r2883, r2880, r2839;
}
{
sub.f16x2 r2886, r2877, r2883;
}
{
add.f16x2 r2889, r2113, r2129;
}
{
mul.f16x2 r2892, r2889, r2837;
}
{
add.f16x2 r2895, r606, r2892;
}
{
sub.f16x2 r2898, r2107, r2123;
}
{
mul.f16x2 r2901, r2898, r2839;
}
{
sub.f16x2 r2904, r2895, r2901;
}
{
add.f16x2 r2907, r2113, r2129;
}
{
mul.f16x2 r2910, r2907, r2837;
}
{
add.f16x2 r2913, r606, r2910;
}
{
sub.f16x2 r2916, r2107, r2123;
}
{
mul.f16x2 r2919, r2916, r2839;
}
{
add.f16x2 r2922, r2913, r2919;
}
mul.wide.u32 rd2, r4685, -1431655765;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r4686, rd3;
mul.lo.s32 r4687, r4686, 3;
sub.s32 r4688, r4685, r4687;
mad.lo.s32 r4689, r4686, 324, r4684;
cvt.rn.f32.u32 f401, r4688;
mul.f32 f402, f401, 0f3D9EDD1F;
cos.approx.f32 f309, f402;
sin.approx.f32 f403, f402;
neg.f32 f310, f403;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f309;
cvt.rn.f16.f32 high, f310;
mov.b32 r2925, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r2928, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r2930, {high, high};
}
{
mul.f16x2 r2932, r2234, r2930;
}
{
neg.f16x2 r2935, r2932;
}
{
fma.rn.f16x2 r2937, r2228, r2928, r2935;
}
{
mul.f16x2 r2941, r2228, r2930;
}
{
fma.rn.f16x2 r2944, r2234, r2928, r2941;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r2948, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r2950, {high, high};
}
mov.f32 f361, 0fBF800000;
mov.f32 f362, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r2952, {low, high};
}
{
mul.f16x2 r2953, r2950, r2952;
}
{
mul.f16x2 r2956, r2925, r2948;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r2959, {high, low};
}
{
fma.rn.f16x2 r2961, r2953, r2959, r2956;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2961;
mov.b32 r2965, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2961;
mov.b32 r2967, {high, high};
}
{
mul.f16x2 r2969, r2322, r2967;
}
{
neg.f16x2 r2972, r2969;
}
{
fma.rn.f16x2 r2974, r2316, r2965, r2972;
}
{
mul.f16x2 r2978, r2316, r2967;
}
{
fma.rn.f16x2 r2981, r2322, r2965, r2978;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r2985, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r2987, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r2989, {low, high};
}
{
mul.f16x2 r2990, r2987, r2989;
}
{
mul.f16x2 r2993, r2961, r2985;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2961;
mov.b32 r2996, {high, low};
}
{
fma.rn.f16x2 r2998, r2990, r2996, r2993;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2998;
mov.b32 r3002, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2998;
mov.b32 r3004, {high, high};
}
{
mul.f16x2 r3006, r2410, r3004;
}
{
neg.f16x2 r3009, r3006;
}
{
fma.rn.f16x2 r3011, r2404, r3002, r3009;
}
{
mul.f16x2 r3015, r2404, r3004;
}
{
fma.rn.f16x2 r3018, r2410, r3002, r3015;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3022, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3024, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3026, {low, high};
}
{
mul.f16x2 r3027, r3024, r3026;
}
{
mul.f16x2 r3030, r2998, r3022;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2998;
mov.b32 r3033, {high, low};
}
{
fma.rn.f16x2 r3035, r3027, r3033, r3030;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3035;
mov.b32 r3039, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3035;
mov.b32 r3041, {high, high};
}
{
mul.f16x2 r3043, r2498, r3041;
}
{
neg.f16x2 r3046, r3043;
}
{
fma.rn.f16x2 r3048, r2492, r3039, r3046;
}
{
mul.f16x2 r3052, r2492, r3041;
}
{
fma.rn.f16x2 r3055, r2498, r3039, r3052;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3059, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3061, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3063, {low, high};
}
{
mul.f16x2 r3064, r3061, r3063;
}
{
mul.f16x2 r3067, r3035, r3059;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3035;
mov.b32 r3070, {high, low};
}
{
fma.rn.f16x2 r3072, r3064, r3070, r3067;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3072;
mov.b32 r3076, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3072;
mov.b32 r3078, {high, high};
}
{
mul.f16x2 r3080, r2586, r3078;
}
{
neg.f16x2 r3083, r3080;
}
{
fma.rn.f16x2 r3085, r2580, r3076, r3083;
}
{
mul.f16x2 r3089, r2580, r3078;
}
{
fma.rn.f16x2 r3092, r2586, r3076, r3089;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3096, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3098, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3100, {low, high};
}
{
mul.f16x2 r3101, r3098, r3100;
}
{
mul.f16x2 r3104, r3072, r3096;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3072;
mov.b32 r3107, {high, low};
}
{
fma.rn.f16x2 r3109, r3101, r3107, r3104;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3109;
mov.b32 r3113, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3109;
mov.b32 r3115, {high, high};
}
{
mul.f16x2 r3117, r2674, r3115;
}
{
neg.f16x2 r3120, r3117;
}
{
fma.rn.f16x2 r3122, r2668, r3113, r3120;
}
{
mul.f16x2 r3126, r2668, r3115;
}
{
fma.rn.f16x2 r3129, r2674, r3113, r3126;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3133, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3135, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3137, {low, high};
}
{
mul.f16x2 r3138, r3135, r3137;
}
{
mul.f16x2 r3141, r3109, r3133;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3109;
mov.b32 r3144, {high, low};
}
{
fma.rn.f16x2 r3146, r3138, r3144, r3141;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3146;
mov.b32 r3150, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3146;
mov.b32 r3152, {high, high};
}
{
mul.f16x2 r3154, r2762, r3152;
}
{
neg.f16x2 r3157, r3154;
}
{
fma.rn.f16x2 r3159, r2756, r3150, r3157;
}
{
mul.f16x2 r3163, r2756, r3152;
}
{
fma.rn.f16x2 r3166, r2762, r3150, r3163;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3170, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3172, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3174, {low, high};
}
{
mul.f16x2 r3175, r3172, r3174;
}
{
mul.f16x2 r3178, r3146, r3170;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3146;
mov.b32 r3181, {high, low};
}
{
fma.rn.f16x2 r3183, r3175, r3181, r3178;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3183;
mov.b32 r3187, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3183;
mov.b32 r3189, {high, high};
}
{
mul.f16x2 r3191, r2850, r3189;
}
{
neg.f16x2 r3194, r3191;
}
{
fma.rn.f16x2 r3196, r2844, r3187, r3194;
}
{
mul.f16x2 r3200, r2844, r3189;
}
{
fma.rn.f16x2 r3203, r2850, r3187, r3200;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3207, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3209, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3211, {low, high};
}
{
mul.f16x2 r3212, r3209, r3211;
}
{
mul.f16x2 r3215, r3183, r3207;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3183;
mov.b32 r3218, {high, low};
}
{
fma.rn.f16x2 r3220, r3212, r3218, r3215;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3220;
mov.b32 r3224, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3220;
mov.b32 r3226, {high, high};
}
{
mul.f16x2 r3228, r2200, r3226;
}
{
neg.f16x2 r3231, r3228;
}
{
fma.rn.f16x2 r3233, r2164, r3224, r3231;
}
{
mul.f16x2 r3237, r2164, r3226;
}
{
fma.rn.f16x2 r3240, r2200, r3224, r3237;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3244, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3246, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3248, {low, high};
}
{
mul.f16x2 r3249, r3246, r3248;
}
{
mul.f16x2 r3252, r3220, r3244;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3220;
mov.b32 r3255, {high, low};
}
{
fma.rn.f16x2 r3257, r3249, r3255, r3252;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3257;
mov.b32 r3261, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3257;
mov.b32 r3263, {high, high};
}
{
mul.f16x2 r3265, r2288, r3263;
}
{
neg.f16x2 r3268, r3265;
}
{
fma.rn.f16x2 r3270, r2252, r3261, r3268;
}
{
mul.f16x2 r3274, r2252, r3263;
}
{
fma.rn.f16x2 r3277, r2288, r3261, r3274;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3281, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3283, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3285, {low, high};
}
{
mul.f16x2 r3286, r3283, r3285;
}
{
mul.f16x2 r3289, r3257, r3281;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3257;
mov.b32 r3292, {high, low};
}
{
fma.rn.f16x2 r3294, r3286, r3292, r3289;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3294;
mov.b32 r3298, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3294;
mov.b32 r3300, {high, high};
}
{
mul.f16x2 r3302, r2376, r3300;
}
{
neg.f16x2 r3305, r3302;
}
{
fma.rn.f16x2 r3307, r2340, r3298, r3305;
}
{
mul.f16x2 r3311, r2340, r3300;
}
{
fma.rn.f16x2 r3314, r2376, r3298, r3311;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3318, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3320, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3322, {low, high};
}
{
mul.f16x2 r3323, r3320, r3322;
}
{
mul.f16x2 r3326, r3294, r3318;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3294;
mov.b32 r3329, {high, low};
}
{
fma.rn.f16x2 r3331, r3323, r3329, r3326;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3331;
mov.b32 r3335, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3331;
mov.b32 r3337, {high, high};
}
{
mul.f16x2 r3339, r2464, r3337;
}
{
neg.f16x2 r3342, r3339;
}
{
fma.rn.f16x2 r3344, r2428, r3335, r3342;
}
{
mul.f16x2 r3348, r2428, r3337;
}
{
fma.rn.f16x2 r3351, r2464, r3335, r3348;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3355, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3357, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3359, {low, high};
}
{
mul.f16x2 r3360, r3357, r3359;
}
{
mul.f16x2 r3363, r3331, r3355;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3331;
mov.b32 r3366, {high, low};
}
{
fma.rn.f16x2 r3368, r3360, r3366, r3363;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3368;
mov.b32 r3372, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3368;
mov.b32 r3374, {high, high};
}
{
mul.f16x2 r3376, r2552, r3374;
}
{
neg.f16x2 r3379, r3376;
}
{
fma.rn.f16x2 r3381, r2516, r3372, r3379;
}
{
mul.f16x2 r3385, r2516, r3374;
}
{
fma.rn.f16x2 r3388, r2552, r3372, r3385;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3392, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3394, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3396, {low, high};
}
{
mul.f16x2 r3397, r3394, r3396;
}
{
mul.f16x2 r3400, r3368, r3392;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3368;
mov.b32 r3403, {high, low};
}
{
fma.rn.f16x2 r3405, r3397, r3403, r3400;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3405;
mov.b32 r3409, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3405;
mov.b32 r3411, {high, high};
}
{
mul.f16x2 r3413, r2640, r3411;
}
{
neg.f16x2 r3416, r3413;
}
{
fma.rn.f16x2 r3418, r2604, r3409, r3416;
}
{
mul.f16x2 r3422, r2604, r3411;
}
{
fma.rn.f16x2 r3425, r2640, r3409, r3422;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3429, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3431, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3433, {low, high};
}
{
mul.f16x2 r3434, r3431, r3433;
}
{
mul.f16x2 r3437, r3405, r3429;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3405;
mov.b32 r3440, {high, low};
}
{
fma.rn.f16x2 r3442, r3434, r3440, r3437;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3442;
mov.b32 r3446, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3442;
mov.b32 r3448, {high, high};
}
{
mul.f16x2 r3450, r2728, r3448;
}
{
neg.f16x2 r3453, r3450;
}
{
fma.rn.f16x2 r3455, r2692, r3446, r3453;
}
{
mul.f16x2 r3459, r2692, r3448;
}
{
fma.rn.f16x2 r3462, r2728, r3446, r3459;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3466, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3468, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3470, {low, high};
}
{
mul.f16x2 r3471, r3468, r3470;
}
{
mul.f16x2 r3474, r3442, r3466;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3442;
mov.b32 r3477, {high, low};
}
{
fma.rn.f16x2 r3479, r3471, r3477, r3474;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3479;
mov.b32 r3483, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3479;
mov.b32 r3485, {high, high};
}
{
mul.f16x2 r3487, r2816, r3485;
}
{
neg.f16x2 r3490, r3487;
}
{
fma.rn.f16x2 r3492, r2780, r3483, r3490;
}
{
mul.f16x2 r3496, r2780, r3485;
}
{
fma.rn.f16x2 r3499, r2816, r3483, r3496;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3503, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3505, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3507, {low, high};
}
{
mul.f16x2 r3508, r3505, r3507;
}
{
mul.f16x2 r3511, r3479, r3503;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3479;
mov.b32 r3514, {high, low};
}
{
fma.rn.f16x2 r3516, r3508, r3514, r3511;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3516;
mov.b32 r3520, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3516;
mov.b32 r3522, {high, high};
}
{
mul.f16x2 r3524, r2904, r3522;
}
{
neg.f16x2 r3527, r3524;
}
{
fma.rn.f16x2 r3529, r2868, r3520, r3527;
}
{
mul.f16x2 r3533, r2868, r3522;
}
{
fma.rn.f16x2 r3536, r2904, r3520, r3533;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3540, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3542, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3544, {low, high};
}
{
mul.f16x2 r3545, r3542, r3544;
}
{
mul.f16x2 r3548, r3516, r3540;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3516;
mov.b32 r3551, {high, low};
}
{
fma.rn.f16x2 r3553, r3545, r3551, r3548;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3553;
mov.b32 r3557, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3553;
mov.b32 r3559, {high, high};
}
{
mul.f16x2 r3561, r2218, r3559;
}
{
neg.f16x2 r3564, r3561;
}
{
fma.rn.f16x2 r3566, r2182, r3557, r3564;
}
{
mul.f16x2 r3570, r2182, r3559;
}
{
fma.rn.f16x2 r3573, r2218, r3557, r3570;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3577, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3579, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3581, {low, high};
}
{
mul.f16x2 r3582, r3579, r3581;
}
{
mul.f16x2 r3585, r3553, r3577;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3553;
mov.b32 r3588, {high, low};
}
{
fma.rn.f16x2 r3590, r3582, r3588, r3585;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3590;
mov.b32 r3594, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3590;
mov.b32 r3596, {high, high};
}
{
mul.f16x2 r3598, r2306, r3596;
}
{
neg.f16x2 r3601, r3598;
}
{
fma.rn.f16x2 r3603, r2270, r3594, r3601;
}
{
mul.f16x2 r3607, r2270, r3596;
}
{
fma.rn.f16x2 r3610, r2306, r3594, r3607;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3614, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3616, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3618, {low, high};
}
{
mul.f16x2 r3619, r3616, r3618;
}
{
mul.f16x2 r3622, r3590, r3614;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3590;
mov.b32 r3625, {high, low};
}
{
fma.rn.f16x2 r3627, r3619, r3625, r3622;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3627;
mov.b32 r3631, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3627;
mov.b32 r3633, {high, high};
}
{
mul.f16x2 r3635, r2394, r3633;
}
{
neg.f16x2 r3638, r3635;
}
{
fma.rn.f16x2 r3640, r2358, r3631, r3638;
}
{
mul.f16x2 r3644, r2358, r3633;
}
{
fma.rn.f16x2 r3647, r2394, r3631, r3644;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3651, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3653, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3655, {low, high};
}
{
mul.f16x2 r3656, r3653, r3655;
}
{
mul.f16x2 r3659, r3627, r3651;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3627;
mov.b32 r3662, {high, low};
}
{
fma.rn.f16x2 r3664, r3656, r3662, r3659;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3664;
mov.b32 r3668, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3664;
mov.b32 r3670, {high, high};
}
{
mul.f16x2 r3672, r2482, r3670;
}
{
neg.f16x2 r3675, r3672;
}
{
fma.rn.f16x2 r3677, r2446, r3668, r3675;
}
{
mul.f16x2 r3681, r2446, r3670;
}
{
fma.rn.f16x2 r3684, r2482, r3668, r3681;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3688, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3690, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3692, {low, high};
}
{
mul.f16x2 r3693, r3690, r3692;
}
{
mul.f16x2 r3696, r3664, r3688;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3664;
mov.b32 r3699, {high, low};
}
{
fma.rn.f16x2 r3701, r3693, r3699, r3696;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3701;
mov.b32 r3705, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3701;
mov.b32 r3707, {high, high};
}
{
mul.f16x2 r3709, r2570, r3707;
}
{
neg.f16x2 r3712, r3709;
}
{
fma.rn.f16x2 r3714, r2534, r3705, r3712;
}
{
mul.f16x2 r3718, r2534, r3707;
}
{
fma.rn.f16x2 r3721, r2570, r3705, r3718;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3725, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3727, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3729, {low, high};
}
{
mul.f16x2 r3730, r3727, r3729;
}
{
mul.f16x2 r3733, r3701, r3725;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3701;
mov.b32 r3736, {high, low};
}
{
fma.rn.f16x2 r3738, r3730, r3736, r3733;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3738;
mov.b32 r3742, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3738;
mov.b32 r3744, {high, high};
}
{
mul.f16x2 r3746, r2658, r3744;
}
{
neg.f16x2 r3749, r3746;
}
{
fma.rn.f16x2 r3751, r2622, r3742, r3749;
}
{
mul.f16x2 r3755, r2622, r3744;
}
{
fma.rn.f16x2 r3758, r2658, r3742, r3755;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3762, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3764, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3766, {low, high};
}
{
mul.f16x2 r3767, r3764, r3766;
}
{
mul.f16x2 r3770, r3738, r3762;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3738;
mov.b32 r3773, {high, low};
}
{
fma.rn.f16x2 r3775, r3767, r3773, r3770;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3775;
mov.b32 r3779, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3775;
mov.b32 r3781, {high, high};
}
{
mul.f16x2 r3783, r2746, r3781;
}
{
neg.f16x2 r3786, r3783;
}
{
fma.rn.f16x2 r3788, r2710, r3779, r3786;
}
{
mul.f16x2 r3792, r2710, r3781;
}
{
fma.rn.f16x2 r3795, r2746, r3779, r3792;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3799, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3801, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3803, {low, high};
}
{
mul.f16x2 r3804, r3801, r3803;
}
{
mul.f16x2 r3807, r3775, r3799;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3775;
mov.b32 r3810, {high, low};
}
{
fma.rn.f16x2 r3812, r3804, r3810, r3807;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3812;
mov.b32 r3816, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3812;
mov.b32 r3818, {high, high};
}
{
mul.f16x2 r3820, r2834, r3818;
}
{
neg.f16x2 r3823, r3820;
}
{
fma.rn.f16x2 r3825, r2798, r3816, r3823;
}
{
mul.f16x2 r3829, r2798, r3818;
}
{
fma.rn.f16x2 r3832, r2834, r3816, r3829;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3836, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2925;
mov.b32 r3838, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f361;
cvt.rn.f16.f32 high, f362;
mov.b32 r3840, {low, high};
}
{
mul.f16x2 r3841, r3838, r3840;
}
{
mul.f16x2 r3844, r3812, r3836;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3812;
mov.b32 r3847, {high, low};
}
{
fma.rn.f16x2 r3849, r3841, r3847, r3844;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3849;
mov.b32 r3853, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3849;
mov.b32 r3855, {high, high};
}
{
mul.f16x2 r3857, r2922, r3855;
}
{
neg.f16x2 r3860, r3857;
}
{
fma.rn.f16x2 r3862, r2886, r3853, r3860;
}
{
mul.f16x2 r3866, r2886, r3855;
}
{
fma.rn.f16x2 r3869, r2922, r3853, r3866;
}
barrier.sync 0;
mad.lo.s32 r4690, r4688, 108, r4689;
st.shared.u32 [r4690], r2140;
st.shared.u32 [r4690+4], r2937;
st.shared.u32 [r4690+8], r2974;
st.shared.u32 [r4690+12], r3011;
st.shared.u32 [r4690+16], r3048;
st.shared.u32 [r4690+20], r3085;
st.shared.u32 [r4690+24], r3122;
st.shared.u32 [r4690+28], r3159;
st.shared.u32 [r4690+32], r3196;
st.shared.u32 [r4690+36], r3233;
st.shared.u32 [r4690+40], r3270;
st.shared.u32 [r4690+44], r3307;
st.shared.u32 [r4690+48], r3344;
st.shared.u32 [r4690+52], r3381;
st.shared.u32 [r4690+56], r3418;
st.shared.u32 [r4690+60], r3455;
st.shared.u32 [r4690+64], r3492;
st.shared.u32 [r4690+68], r3529;
st.shared.u32 [r4690+72], r3566;
st.shared.u32 [r4690+76], r3603;
st.shared.u32 [r4690+80], r3640;
st.shared.u32 [r4690+84], r3677;
st.shared.u32 [r4690+88], r3714;
st.shared.u32 [r4690+92], r3751;
st.shared.u32 [r4690+96], r3788;
st.shared.u32 [r4690+100], r3825;
st.shared.u32 [r4690+104], r3862;
barrier.sync 0;
mad.lo.s32 r4691, r4688, -104, r4690;
ld.shared.u32 r3898, [r4691];
ld.shared.u32 r3986, [r4691+12];
ld.shared.u32 r4074, [r4691+24];
ld.shared.u32 r4162, [r4691+36];
ld.shared.u32 r4250, [r4691+48];
ld.shared.u32 r4338, [r4691+60];
ld.shared.u32 r4426, [r4691+72];
ld.shared.u32 r4514, [r4691+84];
ld.shared.u32 r4602, [r4691+96];
ld.shared.u32 r3895, [r4691+108];
ld.shared.u32 r3983, [r4691+120];
ld.shared.u32 r4071, [r4691+132];
ld.shared.u32 r4159, [r4691+144];
ld.shared.u32 r4247, [r4691+156];
ld.shared.u32 r4335, [r4691+168];
ld.shared.u32 r4423, [r4691+180];
ld.shared.u32 r4511, [r4691+192];
ld.shared.u32 r4599, [r4691+204];
ld.shared.u32 r3896, [r4691+216];
ld.shared.u32 r3984, [r4691+228];
ld.shared.u32 r4072, [r4691+240];
ld.shared.u32 r4160, [r4691+252];
ld.shared.u32 r4248, [r4691+264];
ld.shared.u32 r4336, [r4691+276];
ld.shared.u32 r4424, [r4691+288];
ld.shared.u32 r4512, [r4691+300];
ld.shared.u32 r4600, [r4691+312];
barrier.sync 0;
st.shared.u32 [r4690], r2146;
st.shared.u32 [r4690+4], r2944;
st.shared.u32 [r4690+8], r2981;
st.shared.u32 [r4690+12], r3018;
st.shared.u32 [r4690+16], r3055;
st.shared.u32 [r4690+20], r3092;
st.shared.u32 [r4690+24], r3129;
st.shared.u32 [r4690+28], r3166;
st.shared.u32 [r4690+32], r3203;
st.shared.u32 [r4690+36], r3240;
st.shared.u32 [r4690+40], r3277;
st.shared.u32 [r4690+44], r3314;
st.shared.u32 [r4690+48], r3351;
st.shared.u32 [r4690+52], r3388;
st.shared.u32 [r4690+56], r3425;
st.shared.u32 [r4690+60], r3462;
st.shared.u32 [r4690+64], r3499;
st.shared.u32 [r4690+68], r3536;
st.shared.u32 [r4690+72], r3573;
st.shared.u32 [r4690+76], r3610;
st.shared.u32 [r4690+80], r3647;
st.shared.u32 [r4690+84], r3684;
st.shared.u32 [r4690+88], r3721;
st.shared.u32 [r4690+92], r3758;
st.shared.u32 [r4690+96], r3795;
st.shared.u32 [r4690+100], r3832;
st.shared.u32 [r4690+104], r3869;
barrier.sync 0;
ld.shared.u32 r3904, [r4691];
ld.shared.u32 r3992, [r4691+12];
ld.shared.u32 r4080, [r4691+24];
ld.shared.u32 r4168, [r4691+36];
ld.shared.u32 r4256, [r4691+48];
ld.shared.u32 r4344, [r4691+60];
ld.shared.u32 r4432, [r4691+72];
ld.shared.u32 r4520, [r4691+84];
ld.shared.u32 r4608, [r4691+96];
ld.shared.u32 r3901, [r4691+108];
ld.shared.u32 r3989, [r4691+120];
ld.shared.u32 r4077, [r4691+132];
ld.shared.u32 r4165, [r4691+144];
ld.shared.u32 r4253, [r4691+156];
ld.shared.u32 r4341, [r4691+168];
ld.shared.u32 r4429, [r4691+180];
ld.shared.u32 r4517, [r4691+192];
ld.shared.u32 r4605, [r4691+204];
ld.shared.u32 r3902, [r4691+216];
ld.shared.u32 r3990, [r4691+228];
ld.shared.u32 r4078, [r4691+240];
ld.shared.u32 r4166, [r4691+252];
ld.shared.u32 r4254, [r4691+264];
ld.shared.u32 r4342, [r4691+276];
ld.shared.u32 r4430, [r4691+288];
ld.shared.u32 r4518, [r4691+300];
ld.shared.u32 r4606, [r4691+312];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r3890, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r3891, {low, high};
}
{
neg.f16x2 r3892, r3891;
}
{
add.f16x2 r3894, r3895, r3896;
}
{
add.f16x2 %0, r3898, r3894;
}
{
add.f16x2 r3900, r3901, r3902;
}
{
add.f16x2 %1, r3904, r3900;
}
{
add.f16x2 r3906, r3895, r3896;
}
{
mul.f16x2 r3909, r3906, r3890;
}
{
add.f16x2 r3912, r3898, r3909;
}
{
sub.f16x2 r3915, r3901, r3902;
}
{
mul.f16x2 r3918, r3915, r3892;
}
{
add.f16x2 %18, r3912, r3918;
}
{
add.f16x2 r3924, r3895, r3896;
}
{
mul.f16x2 r3927, r3924, r3890;
}
{
add.f16x2 r3930, r3898, r3927;
}
{
sub.f16x2 r3933, r3901, r3902;
}
{
mul.f16x2 r3936, r3933, r3892;
}
{
sub.f16x2 %36, r3930, r3936;
}
{
add.f16x2 r3942, r3901, r3902;
}
{
mul.f16x2 r3945, r3942, r3890;
}
{
add.f16x2 r3948, r3904, r3945;
}
{
sub.f16x2 r3951, r3895, r3896;
}
{
mul.f16x2 r3954, r3951, r3892;
}
{
sub.f16x2 %19, r3948, r3954;
}
{
add.f16x2 r3960, r3901, r3902;
}
{
mul.f16x2 r3963, r3960, r3890;
}
{
add.f16x2 r3966, r3904, r3963;
}
{
sub.f16x2 r3969, r3895, r3896;
}
{
mul.f16x2 r3972, r3969, r3892;
}
{
add.f16x2 %37, r3966, r3972;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r3978, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r3979, {low, high};
}
{
neg.f16x2 r3980, r3979;
}
{
add.f16x2 r3982, r3983, r3984;
}
{
add.f16x2 %2, r3986, r3982;
}
{
add.f16x2 r3988, r3989, r3990;
}
{
add.f16x2 %3, r3992, r3988;
}
{
add.f16x2 r3994, r3983, r3984;
}
{
mul.f16x2 r3997, r3994, r3978;
}
{
add.f16x2 r4000, r3986, r3997;
}
{
sub.f16x2 r4003, r3989, r3990;
}
{
mul.f16x2 r4006, r4003, r3980;
}
{
add.f16x2 %20, r4000, r4006;
}
{
add.f16x2 r4012, r3983, r3984;
}
{
mul.f16x2 r4015, r4012, r3978;
}
{
add.f16x2 r4018, r3986, r4015;
}
{
sub.f16x2 r4021, r3989, r3990;
}
{
mul.f16x2 r4024, r4021, r3980;
}
{
sub.f16x2 %38, r4018, r4024;
}
{
add.f16x2 r4030, r3989, r3990;
}
{
mul.f16x2 r4033, r4030, r3978;
}
{
add.f16x2 r4036, r3992, r4033;
}
{
sub.f16x2 r4039, r3983, r3984;
}
{
mul.f16x2 r4042, r4039, r3980;
}
{
sub.f16x2 %21, r4036, r4042;
}
{
add.f16x2 r4048, r3989, r3990;
}
{
mul.f16x2 r4051, r4048, r3978;
}
{
add.f16x2 r4054, r3992, r4051;
}
{
sub.f16x2 r4057, r3983, r3984;
}
{
mul.f16x2 r4060, r4057, r3980;
}
{
add.f16x2 %39, r4054, r4060;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4066, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4067, {low, high};
}
{
neg.f16x2 r4068, r4067;
}
{
add.f16x2 r4070, r4071, r4072;
}
{
add.f16x2 %4, r4074, r4070;
}
{
add.f16x2 r4076, r4077, r4078;
}
{
add.f16x2 %5, r4080, r4076;
}
{
add.f16x2 r4082, r4071, r4072;
}
{
mul.f16x2 r4085, r4082, r4066;
}
{
add.f16x2 r4088, r4074, r4085;
}
{
sub.f16x2 r4091, r4077, r4078;
}
{
mul.f16x2 r4094, r4091, r4068;
}
{
add.f16x2 %22, r4088, r4094;
}
{
add.f16x2 r4100, r4071, r4072;
}
{
mul.f16x2 r4103, r4100, r4066;
}
{
add.f16x2 r4106, r4074, r4103;
}
{
sub.f16x2 r4109, r4077, r4078;
}
{
mul.f16x2 r4112, r4109, r4068;
}
{
sub.f16x2 %40, r4106, r4112;
}
{
add.f16x2 r4118, r4077, r4078;
}
{
mul.f16x2 r4121, r4118, r4066;
}
{
add.f16x2 r4124, r4080, r4121;
}
{
sub.f16x2 r4127, r4071, r4072;
}
{
mul.f16x2 r4130, r4127, r4068;
}
{
sub.f16x2 %23, r4124, r4130;
}
{
add.f16x2 r4136, r4077, r4078;
}
{
mul.f16x2 r4139, r4136, r4066;
}
{
add.f16x2 r4142, r4080, r4139;
}
{
sub.f16x2 r4145, r4071, r4072;
}
{
mul.f16x2 r4148, r4145, r4068;
}
{
add.f16x2 %41, r4142, r4148;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4154, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4155, {low, high};
}
{
neg.f16x2 r4156, r4155;
}
{
add.f16x2 r4158, r4159, r4160;
}
{
add.f16x2 %6, r4162, r4158;
}
{
add.f16x2 r4164, r4165, r4166;
}
{
add.f16x2 %7, r4168, r4164;
}
{
add.f16x2 r4170, r4159, r4160;
}
{
mul.f16x2 r4173, r4170, r4154;
}
{
add.f16x2 r4176, r4162, r4173;
}
{
sub.f16x2 r4179, r4165, r4166;
}
{
mul.f16x2 r4182, r4179, r4156;
}
{
add.f16x2 %24, r4176, r4182;
}
{
add.f16x2 r4188, r4159, r4160;
}
{
mul.f16x2 r4191, r4188, r4154;
}
{
add.f16x2 r4194, r4162, r4191;
}
{
sub.f16x2 r4197, r4165, r4166;
}
{
mul.f16x2 r4200, r4197, r4156;
}
{
sub.f16x2 %42, r4194, r4200;
}
{
add.f16x2 r4206, r4165, r4166;
}
{
mul.f16x2 r4209, r4206, r4154;
}
{
add.f16x2 r4212, r4168, r4209;
}
{
sub.f16x2 r4215, r4159, r4160;
}
{
mul.f16x2 r4218, r4215, r4156;
}
{
sub.f16x2 %25, r4212, r4218;
}
{
add.f16x2 r4224, r4165, r4166;
}
{
mul.f16x2 r4227, r4224, r4154;
}
{
add.f16x2 r4230, r4168, r4227;
}
{
sub.f16x2 r4233, r4159, r4160;
}
{
mul.f16x2 r4236, r4233, r4156;
}
{
add.f16x2 %43, r4230, r4236;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4242, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4243, {low, high};
}
{
neg.f16x2 r4244, r4243;
}
{
add.f16x2 r4246, r4247, r4248;
}
{
add.f16x2 %8, r4250, r4246;
}
{
add.f16x2 r4252, r4253, r4254;
}
{
add.f16x2 %9, r4256, r4252;
}
{
add.f16x2 r4258, r4247, r4248;
}
{
mul.f16x2 r4261, r4258, r4242;
}
{
add.f16x2 r4264, r4250, r4261;
}
{
sub.f16x2 r4267, r4253, r4254;
}
{
mul.f16x2 r4270, r4267, r4244;
}
{
add.f16x2 %26, r4264, r4270;
}
{
add.f16x2 r4276, r4247, r4248;
}
{
mul.f16x2 r4279, r4276, r4242;
}
{
add.f16x2 r4282, r4250, r4279;
}
{
sub.f16x2 r4285, r4253, r4254;
}
{
mul.f16x2 r4288, r4285, r4244;
}
{
sub.f16x2 %44, r4282, r4288;
}
{
add.f16x2 r4294, r4253, r4254;
}
{
mul.f16x2 r4297, r4294, r4242;
}
{
add.f16x2 r4300, r4256, r4297;
}
{
sub.f16x2 r4303, r4247, r4248;
}
{
mul.f16x2 r4306, r4303, r4244;
}
{
sub.f16x2 %27, r4300, r4306;
}
{
add.f16x2 r4312, r4253, r4254;
}
{
mul.f16x2 r4315, r4312, r4242;
}
{
add.f16x2 r4318, r4256, r4315;
}
{
sub.f16x2 r4321, r4247, r4248;
}
{
mul.f16x2 r4324, r4321, r4244;
}
{
add.f16x2 %45, r4318, r4324;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4330, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4331, {low, high};
}
{
neg.f16x2 r4332, r4331;
}
{
add.f16x2 r4334, r4335, r4336;
}
{
add.f16x2 %10, r4338, r4334;
}
{
add.f16x2 r4340, r4341, r4342;
}
{
add.f16x2 %11, r4344, r4340;
}
{
add.f16x2 r4346, r4335, r4336;
}
{
mul.f16x2 r4349, r4346, r4330;
}
{
add.f16x2 r4352, r4338, r4349;
}
{
sub.f16x2 r4355, r4341, r4342;
}
{
mul.f16x2 r4358, r4355, r4332;
}
{
add.f16x2 %28, r4352, r4358;
}
{
add.f16x2 r4364, r4335, r4336;
}
{
mul.f16x2 r4367, r4364, r4330;
}
{
add.f16x2 r4370, r4338, r4367;
}
{
sub.f16x2 r4373, r4341, r4342;
}
{
mul.f16x2 r4376, r4373, r4332;
}
{
sub.f16x2 %46, r4370, r4376;
}
{
add.f16x2 r4382, r4341, r4342;
}
{
mul.f16x2 r4385, r4382, r4330;
}
{
add.f16x2 r4388, r4344, r4385;
}
{
sub.f16x2 r4391, r4335, r4336;
}
{
mul.f16x2 r4394, r4391, r4332;
}
{
sub.f16x2 %29, r4388, r4394;
}
{
add.f16x2 r4400, r4341, r4342;
}
{
mul.f16x2 r4403, r4400, r4330;
}
{
add.f16x2 r4406, r4344, r4403;
}
{
sub.f16x2 r4409, r4335, r4336;
}
{
mul.f16x2 r4412, r4409, r4332;
}
{
add.f16x2 %47, r4406, r4412;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4418, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4419, {low, high};
}
{
neg.f16x2 r4420, r4419;
}
{
add.f16x2 r4422, r4423, r4424;
}
{
add.f16x2 %12, r4426, r4422;
}
{
add.f16x2 r4428, r4429, r4430;
}
{
add.f16x2 %13, r4432, r4428;
}
{
add.f16x2 r4434, r4423, r4424;
}
{
mul.f16x2 r4437, r4434, r4418;
}
{
add.f16x2 r4440, r4426, r4437;
}
{
sub.f16x2 r4443, r4429, r4430;
}
{
mul.f16x2 r4446, r4443, r4420;
}
{
add.f16x2 %30, r4440, r4446;
}
{
add.f16x2 r4452, r4423, r4424;
}
{
mul.f16x2 r4455, r4452, r4418;
}
{
add.f16x2 r4458, r4426, r4455;
}
{
sub.f16x2 r4461, r4429, r4430;
}
{
mul.f16x2 r4464, r4461, r4420;
}
{
sub.f16x2 %48, r4458, r4464;
}
{
add.f16x2 r4470, r4429, r4430;
}
{
mul.f16x2 r4473, r4470, r4418;
}
{
add.f16x2 r4476, r4432, r4473;
}
{
sub.f16x2 r4479, r4423, r4424;
}
{
mul.f16x2 r4482, r4479, r4420;
}
{
sub.f16x2 %31, r4476, r4482;
}
{
add.f16x2 r4488, r4429, r4430;
}
{
mul.f16x2 r4491, r4488, r4418;
}
{
add.f16x2 r4494, r4432, r4491;
}
{
sub.f16x2 r4497, r4423, r4424;
}
{
mul.f16x2 r4500, r4497, r4420;
}
{
add.f16x2 %49, r4494, r4500;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4506, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4507, {low, high};
}
{
neg.f16x2 r4508, r4507;
}
{
add.f16x2 r4510, r4511, r4512;
}
{
add.f16x2 %14, r4514, r4510;
}
{
add.f16x2 r4516, r4517, r4518;
}
{
add.f16x2 %15, r4520, r4516;
}
{
add.f16x2 r4522, r4511, r4512;
}
{
mul.f16x2 r4525, r4522, r4506;
}
{
add.f16x2 r4528, r4514, r4525;
}
{
sub.f16x2 r4531, r4517, r4518;
}
{
mul.f16x2 r4534, r4531, r4508;
}
{
add.f16x2 %32, r4528, r4534;
}
{
add.f16x2 r4540, r4511, r4512;
}
{
mul.f16x2 r4543, r4540, r4506;
}
{
add.f16x2 r4546, r4514, r4543;
}
{
sub.f16x2 r4549, r4517, r4518;
}
{
mul.f16x2 r4552, r4549, r4508;
}
{
sub.f16x2 %50, r4546, r4552;
}
{
add.f16x2 r4558, r4517, r4518;
}
{
mul.f16x2 r4561, r4558, r4506;
}
{
add.f16x2 r4564, r4520, r4561;
}
{
sub.f16x2 r4567, r4511, r4512;
}
{
mul.f16x2 r4570, r4567, r4508;
}
{
sub.f16x2 %33, r4564, r4570;
}
{
add.f16x2 r4576, r4517, r4518;
}
{
mul.f16x2 r4579, r4576, r4506;
}
{
add.f16x2 r4582, r4520, r4579;
}
{
sub.f16x2 r4585, r4511, r4512;
}
{
mul.f16x2 r4588, r4585, r4508;
}
{
add.f16x2 %51, r4582, r4588;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f398;
cvt.rn.f16.f32 high, f398;
mov.b32 r4594, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f400;
cvt.rn.f16.f32 high, f400;
mov.b32 r4595, {low, high};
}
{
neg.f16x2 r4596, r4595;
}
{
add.f16x2 r4598, r4599, r4600;
}
{
add.f16x2 %16, r4602, r4598;
}
{
add.f16x2 r4604, r4605, r4606;
}
{
add.f16x2 %17, r4608, r4604;
}
{
add.f16x2 r4610, r4599, r4600;
}
{
mul.f16x2 r4613, r4610, r4594;
}
{
add.f16x2 r4616, r4602, r4613;
}
{
sub.f16x2 r4619, r4605, r4606;
}
{
mul.f16x2 r4622, r4619, r4596;
}
{
add.f16x2 %34, r4616, r4622;
}
{
add.f16x2 r4628, r4599, r4600;
}
{
mul.f16x2 r4631, r4628, r4594;
}
{
add.f16x2 r4634, r4602, r4631;
}
{
sub.f16x2 r4637, r4605, r4606;
}
{
mul.f16x2 r4640, r4637, r4596;
}
{
sub.f16x2 %52, r4634, r4640;
}
{
add.f16x2 r4646, r4605, r4606;
}
{
mul.f16x2 r4649, r4646, r4594;
}
{
add.f16x2 r4652, r4608, r4649;
}
{
sub.f16x2 r4655, r4599, r4600;
}
{
mul.f16x2 r4658, r4655, r4596;
}
{
sub.f16x2 %35, r4652, r4658;
}
{
add.f16x2 r4664, r4605, r4606;
}
{
mul.f16x2 r4667, r4664, r4594;
}
{
add.f16x2 r4670, r4608, r4667;
}
{
sub.f16x2 r4673, r4599, r4600;
}
{
mul.f16x2 r4676, r4673, r4596;
}
{
add.f16x2 %53, r4670, r4676;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[4].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<876, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<50>;
.reg .b32 r<611>;
.reg .b64 rd<8>;
mov.u32 r584, %tid.y;
mov.u32 r585, %6;
mad.lo.s32 r586, r584, 648, r585;
mov.u32 r587, %tid.x;
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
neg.f16x2 r3, r2;
}
{
add.f16x2 r5, %9, %11;
}
{
add.f16x2 r8, %7, r5;
}
{
add.f16x2 r11, %10, %12;
}
{
add.f16x2 r14, %8, r11;
}
{
add.f16x2 r17, %9, %11;
}
{
mul.f16x2 r20, r17, r1;
}
{
add.f16x2 r23, %7, r20;
}
{
sub.f16x2 r26, %10, %12;
}
{
mul.f16x2 r29, r26, r3;
}
{
add.f16x2 r32, r23, r29;
}
{
add.f16x2 r35, %9, %11;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %7, r38;
}
{
sub.f16x2 r44, %10, %12;
}
{
mul.f16x2 r47, r44, r3;
}
{
sub.f16x2 r50, r41, r47;
}
{
add.f16x2 r53, %10, %12;
}
{
mul.f16x2 r56, r53, r1;
}
{
add.f16x2 r59, %8, r56;
}
{
sub.f16x2 r62, %9, %11;
}
{
mul.f16x2 r65, r62, r3;
}
{
sub.f16x2 r68, r59, r65;
}
{
add.f16x2 r71, %10, %12;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %8, r74;
}
{
sub.f16x2 r80, %9, %11;
}
{
mul.f16x2 r83, r80, r3;
}
{
add.f16x2 r86, r77, r83;
}
mul.wide.u32 rd2, r587, 795364315;
shr.u64 rd3, rd2, 32;
cvt.u32.u64 r588, rd3;
sub.s32 r589, r587, r588;
shr.u32 r590, r589, 1;
add.s32 r591, r590, r588;
shr.u32 r592, r591, 4;
mul.lo.s32 r593, r592, 27;
sub.s32 r594, r587, r593;
mad.lo.s32 r595, r592, 648, r586;
cvt.rn.f32.u32 f41, r594;
mul.f32 f42, f41, 0f3D9EDD1F;
cos.approx.f32 f5, f42;
sin.approx.f32 f43, f42;
neg.f32 f6, f43;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f5;
cvt.rn.f16.f32 high, f6;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r92, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r94, {high, high};
}
{
mul.f16x2 r96, r68, r94;
}
{
neg.f16x2 r99, r96;
}
{
fma.rn.f16x2 r101, r32, r92, r99;
}
{
mul.f16x2 r105, r32, r94;
}
{
fma.rn.f16x2 r108, r68, r92, r105;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r112, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r114, {high, high};
}
mov.f32 f33, 0fBF800000;
mov.f32 f34, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
mov.b32 r116, {low, high};
}
{
mul.f16x2 r117, r114, r116;
}
{
mul.f16x2 r120, r89, r112;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r123, {high, low};
}
{
fma.rn.f16x2 r125, r117, r123, r120;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r125;
mov.b32 r129, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r125;
mov.b32 r131, {high, high};
}
{
mul.f16x2 r133, r86, r131;
}
{
neg.f16x2 r136, r133;
}
{
fma.rn.f16x2 r138, r50, r129, r136;
}
{
mul.f16x2 r142, r50, r131;
}
{
fma.rn.f16x2 r145, r86, r129, r142;
}
barrier.sync 0;
mad.lo.s32 r596, r594, 24, r595;
st.shared.v2.f32 [r596], {r8, r14};
st.shared.v2.f32 [r596+8], {r101, r108};
st.shared.v2.f32 [r596+16], {r138, r145};
barrier.sync 0;
shl.b32 r597, r594, 4;
sub.s32 r598, r596, r597;
ld.shared.u32 r174, [r598];
ld.shared.u32 r180, [r598+4];
ld.shared.u32 r171, [r598+216];
ld.shared.u32 r177, [r598+220];
ld.shared.u32 r172, [r598+432];
ld.shared.u32 r178, [r598+436];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f38;
cvt.rn.f16.f32 high, f38;
mov.b32 r166, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f40;
cvt.rn.f16.f32 high, f40;
mov.b32 r167, {low, high};
}
{
neg.f16x2 r168, r167;
}
{
add.f16x2 r170, r171, r172;
}
{
add.f16x2 r173, r174, r170;
}
{
add.f16x2 r176, r177, r178;
}
{
add.f16x2 r179, r180, r176;
}
{
add.f16x2 r182, r171, r172;
}
{
mul.f16x2 r185, r182, r166;
}
{
add.f16x2 r188, r174, r185;
}
{
sub.f16x2 r191, r177, r178;
}
{
mul.f16x2 r194, r191, r168;
}
{
add.f16x2 r197, r188, r194;
}
{
add.f16x2 r200, r171, r172;
}
{
mul.f16x2 r203, r200, r166;
}
{
add.f16x2 r206, r174, r203;
}
{
sub.f16x2 r209, r177, r178;
}
{
mul.f16x2 r212, r209, r168;
}
{
sub.f16x2 r215, r206, r212;
}
{
add.f16x2 r218, r177, r178;
}
{
mul.f16x2 r221, r218, r166;
}
{
add.f16x2 r224, r180, r221;
}
{
sub.f16x2 r227, r171, r172;
}
{
mul.f16x2 r230, r227, r168;
}
{
sub.f16x2 r233, r224, r230;
}
{
add.f16x2 r236, r177, r178;
}
{
mul.f16x2 r239, r236, r166;
}
{
add.f16x2 r242, r180, r239;
}
{
sub.f16x2 r245, r171, r172;
}
{
mul.f16x2 r248, r245, r168;
}
{
add.f16x2 r251, r242, r248;
}
mul.wide.u32 rd4, r594, -1431655765;
shr.u64 rd5, rd4, 33;
cvt.u32.u64 r599, rd5;
mul.lo.s32 r600, r599, 3;
sub.s32 r601, r594, r600;
shl.b32 r602, r601, 3;
add.s32 r603, r595, r602;
cvt.rn.f32.u32 f44, r599;
mul.f32 f45, f44, 0f3E6E4BAE;
cos.approx.f32 f17, f45;
sin.approx.f32 f46, f45;
neg.f32 f18, f46;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f17;
cvt.rn.f16.f32 high, f18;
mov.b32 r254, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r254;
mov.b32 r257, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r254;
mov.b32 r259, {high, high};
}
{
mul.f16x2 r261, r233, r259;
}
{
neg.f16x2 r264, r261;
}
{
fma.rn.f16x2 r266, r197, r257, r264;
}
{
mul.f16x2 r270, r197, r259;
}
{
fma.rn.f16x2 r273, r233, r257, r270;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r254;
mov.b32 r277, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r254;
mov.b32 r279, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
mov.b32 r281, {low, high};
}
{
mul.f16x2 r282, r279, r281;
}
{
mul.f16x2 r285, r254, r277;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r254;
mov.b32 r288, {high, low};
}
{
fma.rn.f16x2 r290, r282, r288, r285;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r290;
mov.b32 r294, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r290;
mov.b32 r296, {high, high};
}
{
mul.f16x2 r298, r251, r296;
}
{
neg.f16x2 r301, r298;
}
{
fma.rn.f16x2 r303, r215, r294, r301;
}
{
mul.f16x2 r307, r215, r296;
}
{
fma.rn.f16x2 r310, r251, r294, r307;
}
barrier.sync 0;
mad.lo.s32 r604, r599, 72, r603;
st.shared.u32 [r604], r173;
st.shared.u32 [r604+4], r179;
st.shared.u32 [r604+24], r266;
st.shared.u32 [r604+28], r273;
st.shared.u32 [r604+48], r303;
st.shared.u32 [r604+52], r310;
barrier.sync 0;
ld.shared.u32 r339, [r598];
ld.shared.u32 r345, [r598+4];
ld.shared.u32 r336, [r598+216];
ld.shared.u32 r342, [r598+220];
ld.shared.u32 r337, [r598+432];
ld.shared.u32 r343, [r598+436];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f38;
cvt.rn.f16.f32 high, f38;
mov.b32 r331, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f40;
cvt.rn.f16.f32 high, f40;
mov.b32 r332, {low, high};
}
{
neg.f16x2 r333, r332;
}
{
add.f16x2 r335, r336, r337;
}
{
add.f16x2 r338, r339, r335;
}
{
add.f16x2 r341, r342, r343;
}
{
add.f16x2 r344, r345, r341;
}
{
add.f16x2 r347, r336, r337;
}
{
mul.f16x2 r350, r347, r331;
}
{
add.f16x2 r353, r339, r350;
}
{
sub.f16x2 r356, r342, r343;
}
{
mul.f16x2 r359, r356, r333;
}
{
add.f16x2 r362, r353, r359;
}
{
add.f16x2 r365, r336, r337;
}
{
mul.f16x2 r368, r365, r331;
}
{
add.f16x2 r371, r339, r368;
}
{
sub.f16x2 r374, r342, r343;
}
{
mul.f16x2 r377, r374, r333;
}
{
sub.f16x2 r380, r371, r377;
}
{
add.f16x2 r383, r342, r343;
}
{
mul.f16x2 r386, r383, r331;
}
{
add.f16x2 r389, r345, r386;
}
{
sub.f16x2 r392, r336, r337;
}
{
mul.f16x2 r395, r392, r333;
}
{
sub.f16x2 r398, r389, r395;
}
{
add.f16x2 r401, r342, r343;
}
{
mul.f16x2 r404, r401, r331;
}
{
add.f16x2 r407, r345, r404;
}
{
sub.f16x2 r410, r336, r337;
}
{
mul.f16x2 r413, r410, r333;
}
{
add.f16x2 r416, r407, r413;
}
mul.wide.u32 rd6, r594, 954437177;
shr.u64 rd7, rd6, 33;
cvt.u32.u64 r605, rd7;
mul.lo.s32 r606, r605, 9;
sub.s32 r607, r594, r606;
shl.b32 r608, r607, 3;
add.s32 r609, r595, r608;
cvt.rn.f32.u32 f47, r605;
mul.f32 f48, f47, 0f3F32B8C2;
cos.approx.f32 f29, f48;
sin.approx.f32 f49, f48;
neg.f32 f30, f49;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f29;
cvt.rn.f16.f32 high, f30;
mov.b32 r419, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r419;
mov.b32 r422, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r419;
mov.b32 r424, {high, high};
}
{
mul.f16x2 r426, r398, r424;
}
{
neg.f16x2 r429, r426;
}
{
fma.rn.f16x2 r431, r362, r422, r429;
}
{
mul.f16x2 r435, r362, r424;
}
{
fma.rn.f16x2 r438, r398, r422, r435;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r419;
mov.b32 r442, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r419;
mov.b32 r444, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
mov.b32 r446, {low, high};
}
{
mul.f16x2 r447, r444, r446;
}
{
mul.f16x2 r450, r419, r442;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r419;
mov.b32 r453, {high, low};
}
{
fma.rn.f16x2 r455, r447, r453, r450;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r455;
mov.b32 r459, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r455;
mov.b32 r461, {high, high};
}
{
mul.f16x2 r463, r416, r461;
}
{
neg.f16x2 r466, r463;
}
{
fma.rn.f16x2 r468, r380, r459, r466;
}
{
mul.f16x2 r472, r380, r461;
}
{
fma.rn.f16x2 r475, r416, r459, r472;
}
barrier.sync 0;
mad.lo.s32 r610, r605, 216, r609;
st.shared.u32 [r610], r338;
st.shared.u32 [r610+4], r344;
st.shared.u32 [r610+72], r431;
st.shared.u32 [r610+76], r438;
st.shared.u32 [r610+144], r468;
st.shared.u32 [r610+148], r475;
barrier.sync 0;
ld.shared.u32 r504, [r598];
ld.shared.u32 r510, [r598+4];
ld.shared.u32 r501, [r598+216];
ld.shared.u32 r507, [r598+220];
ld.shared.u32 r502, [r598+432];
ld.shared.u32 r508, [r598+436];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f38;
cvt.rn.f16.f32 high, f38;
mov.b32 r496, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f40;
cvt.rn.f16.f32 high, f40;
mov.b32 r497, {low, high};
}
{
neg.f16x2 r498, r497;
}
{
add.f16x2 r500, r501, r502;
}
{
add.f16x2 %0, r504, r500;
}
{
add.f16x2 r506, r507, r508;
}
{
add.f16x2 %1, r510, r506;
}
{
add.f16x2 r512, r501, r502;
}
{
mul.f16x2 r515, r512, r496;
}
{
add.f16x2 r518, r504, r515;
}
{
sub.f16x2 r521, r507, r508;
}
{
mul.f16x2 r524, r521, r498;
}
{
add.f16x2 %2, r518, r524;
}
{
add.f16x2 r530, r501, r502;
}
{
mul.f16x2 r533, r530, r496;
}
{
add.f16x2 r536, r504, r533;
}
{
sub.f16x2 r539, r507, r508;
}
{
mul.f16x2 r542, r539, r498;
}
{
sub.f16x2 %4, r536, r542;
}
{
add.f16x2 r548, r507, r508;
}
{
mul.f16x2 r551, r548, r496;
}
{
add.f16x2 r554, r510, r551;
}
{
sub.f16x2 r557, r501, r502;
}
{
mul.f16x2 r560, r557, r498;
}
{
sub.f16x2 %3, r554, r560;
}
{
add.f16x2 r566, r507, r508;
}
{
mul.f16x2 r569, r566, r496;
}
{
add.f16x2 r572, r510, r569;
}
{
sub.f16x2 r575, r501, r502;
}
{
mul.f16x2 r578, r575, r498;
}
{
add.f16x2 %5, r572, r578;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<877, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<50>;
.reg .b32 r<611>;
.reg .b64 rd<8>;
mov.u32 r584, %tid.y;
mov.u32 r585, %6;
mad.lo.s32 r586, r584, 324, r585;
mov.u32 r587, %tid.x;
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
neg.f16x2 r3, r2;
}
{
add.f16x2 r5, %9, %11;
}
{
add.f16x2 r8, %7, r5;
}
{
add.f16x2 r11, %10, %12;
}
{
add.f16x2 r14, %8, r11;
}
{
add.f16x2 r17, %9, %11;
}
{
mul.f16x2 r20, r17, r1;
}
{
add.f16x2 r23, %7, r20;
}
{
sub.f16x2 r26, %10, %12;
}
{
mul.f16x2 r29, r26, r3;
}
{
add.f16x2 r32, r23, r29;
}
{
add.f16x2 r35, %9, %11;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %7, r38;
}
{
sub.f16x2 r44, %10, %12;
}
{
mul.f16x2 r47, r44, r3;
}
{
sub.f16x2 r50, r41, r47;
}
{
add.f16x2 r53, %10, %12;
}
{
mul.f16x2 r56, r53, r1;
}
{
add.f16x2 r59, %8, r56;
}
{
sub.f16x2 r62, %9, %11;
}
{
mul.f16x2 r65, r62, r3;
}
{
sub.f16x2 r68, r59, r65;
}
{
add.f16x2 r71, %10, %12;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %8, r74;
}
{
sub.f16x2 r80, %9, %11;
}
{
mul.f16x2 r83, r80, r3;
}
{
add.f16x2 r86, r77, r83;
}
mul.wide.u32 rd2, r587, 795364315;
shr.u64 rd3, rd2, 32;
cvt.u32.u64 r588, rd3;
sub.s32 r589, r587, r588;
shr.u32 r590, r589, 1;
add.s32 r591, r590, r588;
shr.u32 r592, r591, 4;
mul.lo.s32 r593, r592, 27;
sub.s32 r594, r587, r593;
mad.lo.s32 r595, r592, 324, r586;
cvt.rn.f32.u32 f41, r594;
mul.f32 f42, f41, 0f3D9EDD1F;
cos.approx.f32 f5, f42;
sin.approx.f32 f43, f42;
neg.f32 f6, f43;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f5;
cvt.rn.f16.f32 high, f6;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r92, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r94, {high, high};
}
{
mul.f16x2 r96, r68, r94;
}
{
neg.f16x2 r99, r96;
}
{
fma.rn.f16x2 r101, r32, r92, r99;
}
{
mul.f16x2 r105, r32, r94;
}
{
fma.rn.f16x2 r108, r68, r92, r105;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r112, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r114, {high, high};
}
mov.f32 f33, 0fBF800000;
mov.f32 f34, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
mov.b32 r116, {low, high};
}
{
mul.f16x2 r117, r114, r116;
}
{
mul.f16x2 r120, r89, r112;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r123, {high, low};
}
{
fma.rn.f16x2 r125, r117, r123, r120;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r125;
mov.b32 r129, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r125;
mov.b32 r131, {high, high};
}
{
mul.f16x2 r133, r86, r131;
}
{
neg.f16x2 r136, r133;
}
{
fma.rn.f16x2 r138, r50, r129, r136;
}
{
mul.f16x2 r142, r50, r131;
}
{
fma.rn.f16x2 r145, r86, r129, r142;
}
barrier.sync 0;
mad.lo.s32 r596, r594, 12, r595;
st.shared.u32 [r596], r8;
st.shared.u32 [r596+4], r101;
st.shared.u32 [r596+8], r138;
barrier.sync 0;
shl.b32 r597, r594, 3;
sub.s32 r598, r596, r597;
ld.shared.u32 r174, [r598];
ld.shared.u32 r171, [r598+108];
ld.shared.u32 r172, [r598+216];
barrier.sync 0;
st.shared.u32 [r596], r14;
st.shared.u32 [r596+4], r108;
st.shared.u32 [r596+8], r145;
barrier.sync 0;
ld.shared.u32 r180, [r598];
ld.shared.u32 r177, [r598+108];
ld.shared.u32 r178, [r598+216];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f38;
cvt.rn.f16.f32 high, f38;
mov.b32 r166, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f40;
cvt.rn.f16.f32 high, f40;
mov.b32 r167, {low, high};
}
{
neg.f16x2 r168, r167;
}
{
add.f16x2 r170, r171, r172;
}
{
add.f16x2 r173, r174, r170;
}
{
add.f16x2 r176, r177, r178;
}
{
add.f16x2 r179, r180, r176;
}
{
add.f16x2 r182, r171, r172;
}
{
mul.f16x2 r185, r182, r166;
}
{
add.f16x2 r188, r174, r185;
}
{
sub.f16x2 r191, r177, r178;
}
{
mul.f16x2 r194, r191, r168;
}
{
add.f16x2 r197, r188, r194;
}
{
add.f16x2 r200, r171, r172;
}
{
mul.f16x2 r203, r200, r166;
}
{
add.f16x2 r206, r174, r203;
}
{
sub.f16x2 r209, r177, r178;
}
{
mul.f16x2 r212, r209, r168;
}
{
sub.f16x2 r215, r206, r212;
}
{
add.f16x2 r218, r177, r178;
}
{
mul.f16x2 r221, r218, r166;
}
{
add.f16x2 r224, r180, r221;
}
{
sub.f16x2 r227, r171, r172;
}
{
mul.f16x2 r230, r227, r168;
}
{
sub.f16x2 r233, r224, r230;
}
{
add.f16x2 r236, r177, r178;
}
{
mul.f16x2 r239, r236, r166;
}
{
add.f16x2 r242, r180, r239;
}
{
sub.f16x2 r245, r171, r172;
}
{
mul.f16x2 r248, r245, r168;
}
{
add.f16x2 r251, r242, r248;
}
mul.wide.u32 rd4, r594, -1431655765;
shr.u64 rd5, rd4, 33;
cvt.u32.u64 r599, rd5;
mul.lo.s32 r600, r599, 3;
sub.s32 r601, r594, r600;
shl.b32 r602, r601, 2;
add.s32 r603, r595, r602;
cvt.rn.f32.u32 f44, r599;
mul.f32 f45, f44, 0f3E6E4BAE;
cos.approx.f32 f17, f45;
sin.approx.f32 f46, f45;
neg.f32 f18, f46;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f17;
cvt.rn.f16.f32 high, f18;
mov.b32 r254, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r254;
mov.b32 r257, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r254;
mov.b32 r259, {high, high};
}
{
mul.f16x2 r261, r233, r259;
}
{
neg.f16x2 r264, r261;
}
{
fma.rn.f16x2 r266, r197, r257, r264;
}
{
mul.f16x2 r270, r197, r259;
}
{
fma.rn.f16x2 r273, r233, r257, r270;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r254;
mov.b32 r277, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r254;
mov.b32 r279, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
mov.b32 r281, {low, high};
}
{
mul.f16x2 r282, r279, r281;
}
{
mul.f16x2 r285, r254, r277;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r254;
mov.b32 r288, {high, low};
}
{
fma.rn.f16x2 r290, r282, r288, r285;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r290;
mov.b32 r294, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r290;
mov.b32 r296, {high, high};
}
{
mul.f16x2 r298, r251, r296;
}
{
neg.f16x2 r301, r298;
}
{
fma.rn.f16x2 r303, r215, r294, r301;
}
{
mul.f16x2 r307, r215, r296;
}
{
fma.rn.f16x2 r310, r251, r294, r307;
}
barrier.sync 0;
mad.lo.s32 r604, r599, 36, r603;
st.shared.u32 [r604], r173;
st.shared.u32 [r604+12], r266;
st.shared.u32 [r604+24], r303;
barrier.sync 0;
ld.shared.u32 r339, [r598];
ld.shared.u32 r336, [r598+108];
ld.shared.u32 r337, [r598+216];
barrier.sync 0;
st.shared.u32 [r604], r179;
st.shared.u32 [r604+12], r273;
st.shared.u32 [r604+24], r310;
barrier.sync 0;
ld.shared.u32 r345, [r598];
ld.shared.u32 r342, [r598+108];
ld.shared.u32 r343, [r598+216];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f38;
cvt.rn.f16.f32 high, f38;
mov.b32 r331, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f40;
cvt.rn.f16.f32 high, f40;
mov.b32 r332, {low, high};
}
{
neg.f16x2 r333, r332;
}
{
add.f16x2 r335, r336, r337;
}
{
add.f16x2 r338, r339, r335;
}
{
add.f16x2 r341, r342, r343;
}
{
add.f16x2 r344, r345, r341;
}
{
add.f16x2 r347, r336, r337;
}
{
mul.f16x2 r350, r347, r331;
}
{
add.f16x2 r353, r339, r350;
}
{
sub.f16x2 r356, r342, r343;
}
{
mul.f16x2 r359, r356, r333;
}
{
add.f16x2 r362, r353, r359;
}
{
add.f16x2 r365, r336, r337;
}
{
mul.f16x2 r368, r365, r331;
}
{
add.f16x2 r371, r339, r368;
}
{
sub.f16x2 r374, r342, r343;
}
{
mul.f16x2 r377, r374, r333;
}
{
sub.f16x2 r380, r371, r377;
}
{
add.f16x2 r383, r342, r343;
}
{
mul.f16x2 r386, r383, r331;
}
{
add.f16x2 r389, r345, r386;
}
{
sub.f16x2 r392, r336, r337;
}
{
mul.f16x2 r395, r392, r333;
}
{
sub.f16x2 r398, r389, r395;
}
{
add.f16x2 r401, r342, r343;
}
{
mul.f16x2 r404, r401, r331;
}
{
add.f16x2 r407, r345, r404;
}
{
sub.f16x2 r410, r336, r337;
}
{
mul.f16x2 r413, r410, r333;
}
{
add.f16x2 r416, r407, r413;
}
mul.wide.u32 rd6, r594, 954437177;
shr.u64 rd7, rd6, 33;
cvt.u32.u64 r605, rd7;
mul.lo.s32 r606, r605, 9;
sub.s32 r607, r594, r606;
shl.b32 r608, r607, 2;
add.s32 r609, r595, r608;
cvt.rn.f32.u32 f47, r605;
mul.f32 f48, f47, 0f3F32B8C2;
cos.approx.f32 f29, f48;
sin.approx.f32 f49, f48;
neg.f32 f30, f49;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f29;
cvt.rn.f16.f32 high, f30;
mov.b32 r419, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r419;
mov.b32 r422, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r419;
mov.b32 r424, {high, high};
}
{
mul.f16x2 r426, r398, r424;
}
{
neg.f16x2 r429, r426;
}
{
fma.rn.f16x2 r431, r362, r422, r429;
}
{
mul.f16x2 r435, r362, r424;
}
{
fma.rn.f16x2 r438, r398, r422, r435;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r419;
mov.b32 r442, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r419;
mov.b32 r444, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
mov.b32 r446, {low, high};
}
{
mul.f16x2 r447, r444, r446;
}
{
mul.f16x2 r450, r419, r442;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r419;
mov.b32 r453, {high, low};
}
{
fma.rn.f16x2 r455, r447, r453, r450;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r455;
mov.b32 r459, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r455;
mov.b32 r461, {high, high};
}
{
mul.f16x2 r463, r416, r461;
}
{
neg.f16x2 r466, r463;
}
{
fma.rn.f16x2 r468, r380, r459, r466;
}
{
mul.f16x2 r472, r380, r461;
}
{
fma.rn.f16x2 r475, r416, r459, r472;
}
barrier.sync 0;
mad.lo.s32 r610, r605, 108, r609;
st.shared.u32 [r610], r338;
st.shared.u32 [r610+36], r431;
st.shared.u32 [r610+72], r468;
barrier.sync 0;
ld.shared.u32 r504, [r598];
ld.shared.u32 r501, [r598+108];
ld.shared.u32 r502, [r598+216];
barrier.sync 0;
st.shared.u32 [r610], r344;
st.shared.u32 [r610+36], r438;
st.shared.u32 [r610+72], r475;
barrier.sync 0;
ld.shared.u32 r510, [r598];
ld.shared.u32 r507, [r598+108];
ld.shared.u32 r508, [r598+216];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f38;
cvt.rn.f16.f32 high, f38;
mov.b32 r496, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f40;
cvt.rn.f16.f32 high, f40;
mov.b32 r497, {low, high};
}
{
neg.f16x2 r498, r497;
}
{
add.f16x2 r500, r501, r502;
}
{
add.f16x2 %0, r504, r500;
}
{
add.f16x2 r506, r507, r508;
}
{
add.f16x2 %1, r510, r506;
}
{
add.f16x2 r512, r501, r502;
}
{
mul.f16x2 r515, r512, r496;
}
{
add.f16x2 r518, r504, r515;
}
{
sub.f16x2 r521, r507, r508;
}
{
mul.f16x2 r524, r521, r498;
}
{
add.f16x2 %2, r518, r524;
}
{
add.f16x2 r530, r501, r502;
}
{
mul.f16x2 r533, r530, r496;
}
{
add.f16x2 r536, r504, r533;
}
{
sub.f16x2 r539, r507, r508;
}
{
mul.f16x2 r542, r539, r498;
}
{
sub.f16x2 %4, r536, r542;
}
{
add.f16x2 r548, r507, r508;
}
{
mul.f16x2 r551, r548, r496;
}
{
add.f16x2 r554, r510, r551;
}
{
sub.f16x2 r557, r501, r502;
}
{
mul.f16x2 r560, r557, r498;
}
{
sub.f16x2 %3, r554, r560;
}
{
add.f16x2 r566, r507, r508;
}
{
mul.f16x2 r569, r566, r496;
}
{
add.f16x2 r572, r510, r569;
}
{
sub.f16x2 r575, r501, r502;
}
{
mul.f16x2 r578, r575, r498;
}
{
add.f16x2 %5, r572, r578;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)));
};


#endif
