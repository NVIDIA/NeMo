#ifndef CUFFTDX_FFT_2187_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_2187_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<891, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<771>;
.reg .b32 r<8646>;
.reg .b64 rd<6>;
mov.u32 r8572, %54;
mov.u32 r8645, %tid.y;
mad.lo.s32 r8573, r8645, 17496, r8572;
mov.u32 r8574, %tid.x;
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
neg.f16x2 r3, r2;
}
{
add.f16x2 r5, %92, %85;
}
{
add.f16x2 r8, %68, r5;
}
{
add.f16x2 r11, %100, %91;
}
{
add.f16x2 r14, %76, r11;
}
{
add.f16x2 r17, %92, %85;
}
{
mul.f16x2 r20, r17, r1;
}
{
add.f16x2 r23, %68, r20;
}
{
sub.f16x2 r26, %100, %91;
}
{
mul.f16x2 r29, r26, r3;
}
{
add.f16x2 r32, r23, r29;
}
{
add.f16x2 r35, %92, %85;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %68, r38;
}
{
sub.f16x2 r44, %100, %91;
}
{
mul.f16x2 r47, r44, r3;
}
{
sub.f16x2 r50, r41, r47;
}
{
add.f16x2 r53, %100, %91;
}
{
mul.f16x2 r56, r53, r1;
}
{
add.f16x2 r59, %76, r56;
}
{
sub.f16x2 r62, %92, %85;
}
{
mul.f16x2 r65, r62, r3;
}
{
sub.f16x2 r68, r59, r65;
}
{
add.f16x2 r71, %100, %91;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %76, r74;
}
{
sub.f16x2 r80, %92, %85;
}
{
mul.f16x2 r83, r80, r3;
}
{
add.f16x2 r86, r77, r83;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r90, {low, high};
}
{
neg.f16x2 r91, r90;
}
{
add.f16x2 r93, %77, %67;
}
{
add.f16x2 r96, %106, r93;
}
{
add.f16x2 r99, %84, %75;
}
{
add.f16x2 r102, %58, r99;
}
{
add.f16x2 r105, %77, %67;
}
{
mul.f16x2 r108, r105, r89;
}
{
add.f16x2 r111, %106, r108;
}
{
sub.f16x2 r114, %84, %75;
}
{
mul.f16x2 r117, r114, r91;
}
{
add.f16x2 r120, r111, r117;
}
{
add.f16x2 r123, %77, %67;
}
{
mul.f16x2 r126, r123, r89;
}
{
add.f16x2 r129, %106, r126;
}
{
sub.f16x2 r132, %84, %75;
}
{
mul.f16x2 r135, r132, r91;
}
{
sub.f16x2 r138, r129, r135;
}
{
add.f16x2 r141, %84, %75;
}
{
mul.f16x2 r144, r141, r89;
}
{
add.f16x2 r147, %58, r144;
}
{
sub.f16x2 r150, %77, %67;
}
{
mul.f16x2 r153, r150, r91;
}
{
sub.f16x2 r156, r147, r153;
}
{
add.f16x2 r159, %84, %75;
}
{
mul.f16x2 r162, r159, r89;
}
{
add.f16x2 r165, %58, r162;
}
{
sub.f16x2 r168, %77, %67;
}
{
mul.f16x2 r171, r168, r91;
}
{
add.f16x2 r174, r165, r171;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r177, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r178, {low, high};
}
{
neg.f16x2 r179, r178;
}
{
add.f16x2 r181, %59, %105;
}
{
add.f16x2 r184, %90, r181;
}
{
add.f16x2 r187, %66, %57;
}
{
add.f16x2 r190, %99, r187;
}
{
add.f16x2 r193, %59, %105;
}
{
mul.f16x2 r196, r193, r177;
}
{
add.f16x2 r199, %90, r196;
}
{
sub.f16x2 r202, %66, %57;
}
{
mul.f16x2 r205, r202, r179;
}
{
add.f16x2 r208, r199, r205;
}
{
add.f16x2 r211, %59, %105;
}
{
mul.f16x2 r214, r211, r177;
}
{
add.f16x2 r217, %90, r214;
}
{
sub.f16x2 r220, %66, %57;
}
{
mul.f16x2 r223, r220, r179;
}
{
sub.f16x2 r226, r217, r223;
}
{
add.f16x2 r229, %66, %57;
}
{
mul.f16x2 r232, r229, r177;
}
{
add.f16x2 r235, %99, r232;
}
{
sub.f16x2 r238, %59, %105;
}
{
mul.f16x2 r241, r238, r179;
}
{
sub.f16x2 r244, r235, r241;
}
{
add.f16x2 r247, %66, %57;
}
{
mul.f16x2 r250, r247, r177;
}
{
add.f16x2 r253, %99, r250;
}
{
sub.f16x2 r256, %59, %105;
}
{
mul.f16x2 r259, r256, r179;
}
{
add.f16x2 r262, r253, r259;
}
mov.f32 f542, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r265, {low, high};
}
mov.f32 f544, 0fBF248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r266, {low, high};
}
mov.f32 f554, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r267, {low, high};
}
mov.f32 f556, 0fBF7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r268, {low, high};
}
mov.f32 f578, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r271, {low, high};
}
mov.f32 f580, 0fBEAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r345, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r433, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r521, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r609, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r610, {low, high};
}
{
neg.f16x2 r611, r610;
}
{
add.f16x2 r613, %95, %87;
}
{
add.f16x2 r616, %71, r613;
}
{
add.f16x2 r619, %102, %94;
}
{
add.f16x2 r622, %79, r619;
}
{
add.f16x2 r625, %95, %87;
}
{
mul.f16x2 r628, r625, r609;
}
{
add.f16x2 r631, %71, r628;
}
{
sub.f16x2 r634, %102, %94;
}
{
mul.f16x2 r637, r634, r611;
}
{
add.f16x2 r640, r631, r637;
}
{
add.f16x2 r643, %95, %87;
}
{
mul.f16x2 r646, r643, r609;
}
{
add.f16x2 r649, %71, r646;
}
{
sub.f16x2 r652, %102, %94;
}
{
mul.f16x2 r655, r652, r611;
}
{
sub.f16x2 r658, r649, r655;
}
{
add.f16x2 r661, %102, %94;
}
{
mul.f16x2 r664, r661, r609;
}
{
add.f16x2 r667, %79, r664;
}
{
sub.f16x2 r670, %95, %87;
}
{
mul.f16x2 r673, r670, r611;
}
{
sub.f16x2 r676, r667, r673;
}
{
add.f16x2 r679, %102, %94;
}
{
mul.f16x2 r682, r679, r609;
}
{
add.f16x2 r685, %79, r682;
}
{
sub.f16x2 r688, %95, %87;
}
{
mul.f16x2 r691, r688, r611;
}
{
add.f16x2 r694, r685, r691;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r697, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r698, {low, high};
}
{
neg.f16x2 r699, r698;
}
{
add.f16x2 r701, %80, %70;
}
{
add.f16x2 r704, %108, r701;
}
{
add.f16x2 r707, %86, %78;
}
{
add.f16x2 r710, %61, r707;
}
{
add.f16x2 r713, %80, %70;
}
{
mul.f16x2 r716, r713, r697;
}
{
add.f16x2 r719, %108, r716;
}
{
sub.f16x2 r722, %86, %78;
}
{
mul.f16x2 r725, r722, r699;
}
{
add.f16x2 r728, r719, r725;
}
{
add.f16x2 r731, %80, %70;
}
{
mul.f16x2 r734, r731, r697;
}
{
add.f16x2 r737, %108, r734;
}
{
sub.f16x2 r740, %86, %78;
}
{
mul.f16x2 r743, r740, r699;
}
{
sub.f16x2 r746, r737, r743;
}
{
add.f16x2 r749, %86, %78;
}
{
mul.f16x2 r752, r749, r697;
}
{
add.f16x2 r755, %61, r752;
}
{
sub.f16x2 r758, %80, %70;
}
{
mul.f16x2 r761, r758, r699;
}
{
sub.f16x2 r764, r755, r761;
}
{
add.f16x2 r767, %86, %78;
}
{
mul.f16x2 r770, r767, r697;
}
{
add.f16x2 r773, %61, r770;
}
{
sub.f16x2 r776, %80, %70;
}
{
mul.f16x2 r779, r776, r699;
}
{
add.f16x2 r782, r773, r779;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r785, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r786, {low, high};
}
{
neg.f16x2 r787, r786;
}
{
add.f16x2 r789, %62, %107;
}
{
add.f16x2 r792, %93, r789;
}
{
add.f16x2 r795, %69, %60;
}
{
add.f16x2 r798, %101, r795;
}
{
add.f16x2 r801, %62, %107;
}
{
mul.f16x2 r804, r801, r785;
}
{
add.f16x2 r807, %93, r804;
}
{
sub.f16x2 r810, %69, %60;
}
{
mul.f16x2 r813, r810, r787;
}
{
add.f16x2 r816, r807, r813;
}
{
add.f16x2 r819, %62, %107;
}
{
mul.f16x2 r822, r819, r785;
}
{
add.f16x2 r825, %93, r822;
}
{
sub.f16x2 r828, %69, %60;
}
{
mul.f16x2 r831, r828, r787;
}
{
sub.f16x2 r834, r825, r831;
}
{
add.f16x2 r837, %69, %60;
}
{
mul.f16x2 r840, r837, r785;
}
{
add.f16x2 r843, %101, r840;
}
{
sub.f16x2 r846, %62, %107;
}
{
mul.f16x2 r849, r846, r787;
}
{
sub.f16x2 r852, r843, r849;
}
{
add.f16x2 r855, %69, %60;
}
{
mul.f16x2 r858, r855, r785;
}
{
add.f16x2 r861, %101, r858;
}
{
sub.f16x2 r864, %62, %107;
}
{
mul.f16x2 r867, r864, r787;
}
{
add.f16x2 r870, r861, r867;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r873, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r874, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r875, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r876, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r879, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r953, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1041, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1129, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1217, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r1218, {low, high};
}
{
neg.f16x2 r1219, r1218;
}
{
add.f16x2 r1221, %98, %89;
}
{
add.f16x2 r1224, %74, r1221;
}
{
add.f16x2 r1227, %104, %97;
}
{
add.f16x2 r1230, %82, r1227;
}
{
add.f16x2 r1233, %98, %89;
}
{
mul.f16x2 r1236, r1233, r1217;
}
{
add.f16x2 r1239, %74, r1236;
}
{
sub.f16x2 r1242, %104, %97;
}
{
mul.f16x2 r1245, r1242, r1219;
}
{
add.f16x2 r1248, r1239, r1245;
}
{
add.f16x2 r1251, %98, %89;
}
{
mul.f16x2 r1254, r1251, r1217;
}
{
add.f16x2 r1257, %74, r1254;
}
{
sub.f16x2 r1260, %104, %97;
}
{
mul.f16x2 r1263, r1260, r1219;
}
{
sub.f16x2 r1266, r1257, r1263;
}
{
add.f16x2 r1269, %104, %97;
}
{
mul.f16x2 r1272, r1269, r1217;
}
{
add.f16x2 r1275, %82, r1272;
}
{
sub.f16x2 r1278, %98, %89;
}
{
mul.f16x2 r1281, r1278, r1219;
}
{
sub.f16x2 r1284, r1275, r1281;
}
{
add.f16x2 r1287, %104, %97;
}
{
mul.f16x2 r1290, r1287, r1217;
}
{
add.f16x2 r1293, %82, r1290;
}
{
sub.f16x2 r1296, %98, %89;
}
{
mul.f16x2 r1299, r1296, r1219;
}
{
add.f16x2 r1302, r1293, r1299;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1305, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r1306, {low, high};
}
{
neg.f16x2 r1307, r1306;
}
{
add.f16x2 r1309, %83, %73;
}
{
add.f16x2 r1312, %56, r1309;
}
{
add.f16x2 r1315, %88, %81;
}
{
add.f16x2 r1318, %64, r1315;
}
{
add.f16x2 r1321, %83, %73;
}
{
mul.f16x2 r1324, r1321, r1305;
}
{
add.f16x2 r1327, %56, r1324;
}
{
sub.f16x2 r1330, %88, %81;
}
{
mul.f16x2 r1333, r1330, r1307;
}
{
add.f16x2 r1336, r1327, r1333;
}
{
add.f16x2 r1339, %83, %73;
}
{
mul.f16x2 r1342, r1339, r1305;
}
{
add.f16x2 r1345, %56, r1342;
}
{
sub.f16x2 r1348, %88, %81;
}
{
mul.f16x2 r1351, r1348, r1307;
}
{
sub.f16x2 r1354, r1345, r1351;
}
{
add.f16x2 r1357, %88, %81;
}
{
mul.f16x2 r1360, r1357, r1305;
}
{
add.f16x2 r1363, %64, r1360;
}
{
sub.f16x2 r1366, %83, %73;
}
{
mul.f16x2 r1369, r1366, r1307;
}
{
sub.f16x2 r1372, r1363, r1369;
}
{
add.f16x2 r1375, %88, %81;
}
{
mul.f16x2 r1378, r1375, r1305;
}
{
add.f16x2 r1381, %64, r1378;
}
{
sub.f16x2 r1384, %83, %73;
}
{
mul.f16x2 r1387, r1384, r1307;
}
{
add.f16x2 r1390, r1381, r1387;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1393, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r1394, {low, high};
}
{
neg.f16x2 r1395, r1394;
}
{
add.f16x2 r1397, %65, %55;
}
{
add.f16x2 r1400, %96, r1397;
}
{
add.f16x2 r1403, %72, %63;
}
{
add.f16x2 r1406, %103, r1403;
}
{
add.f16x2 r1409, %65, %55;
}
{
mul.f16x2 r1412, r1409, r1393;
}
{
add.f16x2 r1415, %96, r1412;
}
{
sub.f16x2 r1418, %72, %63;
}
{
mul.f16x2 r1421, r1418, r1395;
}
{
add.f16x2 r1424, r1415, r1421;
}
{
add.f16x2 r1427, %65, %55;
}
{
mul.f16x2 r1430, r1427, r1393;
}
{
add.f16x2 r1433, %96, r1430;
}
{
sub.f16x2 r1436, %72, %63;
}
{
mul.f16x2 r1439, r1436, r1395;
}
{
sub.f16x2 r1442, r1433, r1439;
}
{
add.f16x2 r1445, %72, %63;
}
{
mul.f16x2 r1448, r1445, r1393;
}
{
add.f16x2 r1451, %103, r1448;
}
{
sub.f16x2 r1454, %65, %55;
}
{
mul.f16x2 r1457, r1454, r1395;
}
{
sub.f16x2 r1460, r1451, r1457;
}
{
add.f16x2 r1463, %72, %63;
}
{
mul.f16x2 r1466, r1463, r1393;
}
{
add.f16x2 r1469, %103, r1466;
}
{
sub.f16x2 r1472, %65, %55;
}
{
mul.f16x2 r1475, r1472, r1395;
}
{
add.f16x2 r1478, r1469, r1475;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r1481, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r1482, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r1483, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r1484, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r1487, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1561, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1649, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1737, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
mov.f32 f534, 0f3F791978;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f534;
cvt.rn.f16.f32 high, f534;
mov.b32 r1825, {low, high};
}
mov.f32 f536, 0fBE6C2691;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f536;
cvt.rn.f16.f32 high, f536;
mov.b32 r1826, {low, high};
}
mov.f32 f538, 0f3F64C51C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f538;
cvt.rn.f16.f32 high, f538;
mov.b32 r1827, {low, high};
}
mov.f32 f540, 0fBEE5C902;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f540;
cvt.rn.f16.f32 high, f540;
mov.b32 r1828, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r1829, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r1830, {low, high};
}
mov.f32 f546, 0f3F18DF63;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f546;
cvt.rn.f16.f32 high, f546;
mov.b32 r1831, {low, high};
}
mov.f32 f548, 0fBF4D57F2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f548;
cvt.rn.f16.f32 high, f548;
mov.b32 r1832, {low, high};
}
mov.f32 f550, 0f3ECACAF8;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f550;
cvt.rn.f16.f32 high, f550;
mov.b32 r1833, {low, high};
}
mov.f32 f552, 0fBF6B1036;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f552;
cvt.rn.f16.f32 high, f552;
mov.b32 r1834, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r1835, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r1836, {low, high};
}
mov.f32 f558, 0fBD6E2946;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f558;
cvt.rn.f16.f32 high, f558;
mov.b32 r1837, {low, high};
}
mov.f32 f560, 0fBF7F9120;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f560;
cvt.rn.f16.f32 high, f560;
mov.b32 r1838, {low, high};
}
mov.f32 f562, 0fBE92D7E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f562;
cvt.rn.f16.f32 high, f562;
mov.b32 r1839, {low, high};
}
mov.f32 f564, 0fBF753ECD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f564;
cvt.rn.f16.f32 high, f564;
mov.b32 r1840, {low, high};
}
mov.f32 f570, 0fBF2FAD88;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f570;
cvt.rn.f16.f32 high, f570;
mov.b32 r1843, {low, high};
}
mov.f32 f572, 0fBF3A3529;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f572;
cvt.rn.f16.f32 high, f572;
mov.b32 r1844, {low, high};
}
mov.f32 f594, 0fBF55E287;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r1847, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
mov.b32 r1848, {low, high};
}
mov.f32 f586, 0fBF7E44DE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f586;
cvt.rn.f16.f32 high, f586;
mov.b32 r1851, {low, high};
}
mov.f32 f588, 0f3DEDC21F;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f588;
cvt.rn.f16.f32 high, f588;
mov.b32 r1852, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1855, {low, high};
}
mov.f32 f596, 0f3F0CAC9F;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2133, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2221, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2309, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2485, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2573, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2661, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2749, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2837, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
mul.wide.u32 rd2, r8574, -901412889;
shr.u64 rd3, rd2, 38;
cvt.u32.u64 r8575, rd3;
mul.lo.s32 r8576, r8575, 81;
sub.s32 r8577, r8574, r8576;
cvt.rn.f32.u32 f765, r8577;
mul.f32 f766, f765, 0f3B3C4870;
cos.approx.f32 f309, f766;
sin.approx.f32 f767, f766;
neg.f32 f310, f767;
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
mov.f32 f725, 0fBF800000;
mov.f32 f726, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
mad.lo.s32 r8578, r8575, 17496, r8573;
barrier.sync 0;
mad.lo.s32 r8579, r8577, 216, r8578;
st.shared.v2.f32 [r8579], {r2140, r2146};
st.shared.v2.f32 [r8579+8], {r2937, r2944};
st.shared.v2.f32 [r8579+16], {r2974, r2981};
st.shared.v2.f32 [r8579+24], {r3011, r3018};
st.shared.v2.f32 [r8579+32], {r3048, r3055};
st.shared.v2.f32 [r8579+40], {r3085, r3092};
st.shared.v2.f32 [r8579+48], {r3122, r3129};
st.shared.v2.f32 [r8579+56], {r3159, r3166};
st.shared.v2.f32 [r8579+64], {r3196, r3203};
st.shared.v2.f32 [r8579+72], {r3233, r3240};
st.shared.v2.f32 [r8579+80], {r3270, r3277};
st.shared.v2.f32 [r8579+88], {r3307, r3314};
st.shared.v2.f32 [r8579+96], {r3344, r3351};
st.shared.v2.f32 [r8579+104], {r3381, r3388};
st.shared.v2.f32 [r8579+112], {r3418, r3425};
st.shared.v2.f32 [r8579+120], {r3455, r3462};
st.shared.v2.f32 [r8579+128], {r3492, r3499};
st.shared.v2.f32 [r8579+136], {r3529, r3536};
st.shared.v2.f32 [r8579+144], {r3566, r3573};
st.shared.v2.f32 [r8579+152], {r3603, r3610};
st.shared.v2.f32 [r8579+160], {r3640, r3647};
st.shared.v2.f32 [r8579+168], {r3677, r3684};
st.shared.v2.f32 [r8579+176], {r3714, r3721};
st.shared.v2.f32 [r8579+184], {r3751, r3758};
st.shared.v2.f32 [r8579+192], {r3788, r3795};
st.shared.v2.f32 [r8579+200], {r3825, r3832};
st.shared.v2.f32 [r8579+208], {r3862, r3869};
barrier.sync 0;
mad.lo.s32 r8580, r8577, -208, r8579;
ld.shared.u32 r3898, [r8580];
ld.shared.u32 r3904, [r8580+4];
ld.shared.u32 r4506, [r8580+648];
ld.shared.u32 r4512, [r8580+652];
ld.shared.u32 r5114, [r8580+1296];
ld.shared.u32 r5120, [r8580+1300];
ld.shared.u32 r3986, [r8580+1944];
ld.shared.u32 r3992, [r8580+1948];
ld.shared.u32 r4594, [r8580+2592];
ld.shared.u32 r4600, [r8580+2596];
ld.shared.u32 r5202, [r8580+3240];
ld.shared.u32 r5208, [r8580+3244];
ld.shared.u32 r4074, [r8580+3888];
ld.shared.u32 r4080, [r8580+3892];
ld.shared.u32 r4682, [r8580+4536];
ld.shared.u32 r4688, [r8580+4540];
ld.shared.u32 r5290, [r8580+5184];
ld.shared.u32 r5296, [r8580+5188];
ld.shared.u32 r3895, [r8580+5832];
ld.shared.u32 r3901, [r8580+5836];
ld.shared.u32 r4503, [r8580+6480];
ld.shared.u32 r4509, [r8580+6484];
ld.shared.u32 r5111, [r8580+7128];
ld.shared.u32 r5117, [r8580+7132];
ld.shared.u32 r3983, [r8580+7776];
ld.shared.u32 r3989, [r8580+7780];
ld.shared.u32 r4591, [r8580+8424];
ld.shared.u32 r4597, [r8580+8428];
ld.shared.u32 r5199, [r8580+9072];
ld.shared.u32 r5205, [r8580+9076];
ld.shared.u32 r4071, [r8580+9720];
ld.shared.u32 r4077, [r8580+9724];
ld.shared.u32 r4679, [r8580+10368];
ld.shared.u32 r4685, [r8580+10372];
ld.shared.u32 r5287, [r8580+11016];
ld.shared.u32 r5293, [r8580+11020];
ld.shared.u32 r3896, [r8580+11664];
ld.shared.u32 r3902, [r8580+11668];
ld.shared.u32 r4504, [r8580+12312];
ld.shared.u32 r4510, [r8580+12316];
ld.shared.u32 r5112, [r8580+12960];
ld.shared.u32 r5118, [r8580+12964];
ld.shared.u32 r3984, [r8580+13608];
ld.shared.u32 r3990, [r8580+13612];
ld.shared.u32 r4592, [r8580+14256];
ld.shared.u32 r4598, [r8580+14260];
ld.shared.u32 r5200, [r8580+14904];
ld.shared.u32 r5206, [r8580+14908];
ld.shared.u32 r4072, [r8580+15552];
ld.shared.u32 r4078, [r8580+15556];
ld.shared.u32 r4680, [r8580+16200];
ld.shared.u32 r4686, [r8580+16204];
ld.shared.u32 r5288, [r8580+16848];
ld.shared.u32 r5294, [r8580+16852];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r3890, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r3891, {low, high};
}
{
neg.f16x2 r3892, r3891;
}
{
add.f16x2 r3894, r3895, r3896;
}
{
add.f16x2 r3897, r3898, r3894;
}
{
add.f16x2 r3900, r3901, r3902;
}
{
add.f16x2 r3903, r3904, r3900;
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
add.f16x2 r3921, r3912, r3918;
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
sub.f16x2 r3939, r3930, r3936;
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
sub.f16x2 r3957, r3948, r3954;
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
add.f16x2 r3975, r3966, r3972;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r3978, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r3979, {low, high};
}
{
neg.f16x2 r3980, r3979;
}
{
add.f16x2 r3982, r3983, r3984;
}
{
add.f16x2 r3985, r3986, r3982;
}
{
add.f16x2 r3988, r3989, r3990;
}
{
add.f16x2 r3991, r3992, r3988;
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
add.f16x2 r4009, r4000, r4006;
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
sub.f16x2 r4027, r4018, r4024;
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
sub.f16x2 r4045, r4036, r4042;
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
add.f16x2 r4063, r4054, r4060;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4066, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4067, {low, high};
}
{
neg.f16x2 r4068, r4067;
}
{
add.f16x2 r4070, r4071, r4072;
}
{
add.f16x2 r4073, r4074, r4070;
}
{
add.f16x2 r4076, r4077, r4078;
}
{
add.f16x2 r4079, r4080, r4076;
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
add.f16x2 r4097, r4088, r4094;
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
sub.f16x2 r4115, r4106, r4112;
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
sub.f16x2 r4133, r4124, r4130;
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
add.f16x2 r4151, r4142, r4148;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r4154, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r4155, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r4156, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r4157, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r4160, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
mov.b32 r4161, {low, high};
}
{
mul.f16x2 r4170, r4009, r4154;
}
{
mul.f16x2 r4173, r4045, r4155;
}
{
sub.f16x2 r4176, r4170, r4173;
}
{
mul.f16x2 r4179, r4009, r4155;
}
{
fma.rn.f16x2 r4182, r4045, r4154, r4179;
}
{
mul.f16x2 r4186, r4097, r4156;
}
{
mul.f16x2 r4189, r4133, r4157;
}
{
sub.f16x2 r4192, r4186, r4189;
}
{
mul.f16x2 r4195, r4097, r4157;
}
{
fma.rn.f16x2 r4198, r4133, r4156, r4195;
}
{
mul.f16x2 r4202, r4027, r4156;
}
{
mul.f16x2 r4205, r4063, r4157;
}
{
sub.f16x2 r4208, r4202, r4205;
}
{
mul.f16x2 r4211, r4027, r4157;
}
{
fma.rn.f16x2 r4214, r4063, r4156, r4211;
}
{
mul.f16x2 r4218, r4115, r4160;
}
{
mul.f16x2 r4221, r4151, r4161;
}
{
sub.f16x2 r4224, r4218, r4221;
}
{
mul.f16x2 r4227, r4115, r4161;
}
{
fma.rn.f16x2 r4230, r4151, r4160, r4227;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4234, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4235, {low, high};
}
{
neg.f16x2 r4236, r4235;
}
{
add.f16x2 r4238, r3985, r4073;
}
{
add.f16x2 r4241, r3897, r4238;
}
{
add.f16x2 r4244, r3991, r4079;
}
{
add.f16x2 r4247, r3903, r4244;
}
{
add.f16x2 r4250, r3985, r4073;
}
{
mul.f16x2 r4253, r4250, r4234;
}
{
add.f16x2 r4256, r3897, r4253;
}
{
sub.f16x2 r4259, r3991, r4079;
}
{
mul.f16x2 r4262, r4259, r4236;
}
{
add.f16x2 r4265, r4256, r4262;
}
{
add.f16x2 r4268, r3985, r4073;
}
{
mul.f16x2 r4271, r4268, r4234;
}
{
add.f16x2 r4274, r3897, r4271;
}
{
sub.f16x2 r4277, r3991, r4079;
}
{
mul.f16x2 r4280, r4277, r4236;
}
{
sub.f16x2 r4283, r4274, r4280;
}
{
add.f16x2 r4286, r3991, r4079;
}
{
mul.f16x2 r4289, r4286, r4234;
}
{
add.f16x2 r4292, r3903, r4289;
}
{
sub.f16x2 r4295, r3985, r4073;
}
{
mul.f16x2 r4298, r4295, r4236;
}
{
sub.f16x2 r4301, r4292, r4298;
}
{
add.f16x2 r4304, r3991, r4079;
}
{
mul.f16x2 r4307, r4304, r4234;
}
{
add.f16x2 r4310, r3903, r4307;
}
{
sub.f16x2 r4313, r3985, r4073;
}
{
mul.f16x2 r4316, r4313, r4236;
}
{
add.f16x2 r4319, r4310, r4316;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4322, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4323, {low, high};
}
{
neg.f16x2 r4324, r4323;
}
{
add.f16x2 r4326, r4176, r4192;
}
{
add.f16x2 r4329, r3921, r4326;
}
{
add.f16x2 r4332, r4182, r4198;
}
{
add.f16x2 r4335, r3957, r4332;
}
{
add.f16x2 r4338, r4176, r4192;
}
{
mul.f16x2 r4341, r4338, r4322;
}
{
add.f16x2 r4344, r3921, r4341;
}
{
sub.f16x2 r4347, r4182, r4198;
}
{
mul.f16x2 r4350, r4347, r4324;
}
{
add.f16x2 r4353, r4344, r4350;
}
{
add.f16x2 r4356, r4176, r4192;
}
{
mul.f16x2 r4359, r4356, r4322;
}
{
add.f16x2 r4362, r3921, r4359;
}
{
sub.f16x2 r4365, r4182, r4198;
}
{
mul.f16x2 r4368, r4365, r4324;
}
{
sub.f16x2 r4371, r4362, r4368;
}
{
add.f16x2 r4374, r4182, r4198;
}
{
mul.f16x2 r4377, r4374, r4322;
}
{
add.f16x2 r4380, r3957, r4377;
}
{
sub.f16x2 r4383, r4176, r4192;
}
{
mul.f16x2 r4386, r4383, r4324;
}
{
sub.f16x2 r4389, r4380, r4386;
}
{
add.f16x2 r4392, r4182, r4198;
}
{
mul.f16x2 r4395, r4392, r4322;
}
{
add.f16x2 r4398, r3957, r4395;
}
{
sub.f16x2 r4401, r4176, r4192;
}
{
mul.f16x2 r4404, r4401, r4324;
}
{
add.f16x2 r4407, r4398, r4404;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4410, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4411, {low, high};
}
{
neg.f16x2 r4412, r4411;
}
{
add.f16x2 r4414, r4208, r4224;
}
{
add.f16x2 r4417, r3939, r4414;
}
{
add.f16x2 r4420, r4214, r4230;
}
{
add.f16x2 r4423, r3975, r4420;
}
{
add.f16x2 r4426, r4208, r4224;
}
{
mul.f16x2 r4429, r4426, r4410;
}
{
add.f16x2 r4432, r3939, r4429;
}
{
sub.f16x2 r4435, r4214, r4230;
}
{
mul.f16x2 r4438, r4435, r4412;
}
{
add.f16x2 r4441, r4432, r4438;
}
{
add.f16x2 r4444, r4208, r4224;
}
{
mul.f16x2 r4447, r4444, r4410;
}
{
add.f16x2 r4450, r3939, r4447;
}
{
sub.f16x2 r4453, r4214, r4230;
}
{
mul.f16x2 r4456, r4453, r4412;
}
{
sub.f16x2 r4459, r4450, r4456;
}
{
add.f16x2 r4462, r4214, r4230;
}
{
mul.f16x2 r4465, r4462, r4410;
}
{
add.f16x2 r4468, r3975, r4465;
}
{
sub.f16x2 r4471, r4208, r4224;
}
{
mul.f16x2 r4474, r4471, r4412;
}
{
sub.f16x2 r4477, r4468, r4474;
}
{
add.f16x2 r4480, r4214, r4230;
}
{
mul.f16x2 r4483, r4480, r4410;
}
{
add.f16x2 r4486, r3975, r4483;
}
{
sub.f16x2 r4489, r4208, r4224;
}
{
mul.f16x2 r4492, r4489, r4412;
}
{
add.f16x2 r4495, r4486, r4492;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4498, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4499, {low, high};
}
{
neg.f16x2 r4500, r4499;
}
{
add.f16x2 r4502, r4503, r4504;
}
{
add.f16x2 r4505, r4506, r4502;
}
{
add.f16x2 r4508, r4509, r4510;
}
{
add.f16x2 r4511, r4512, r4508;
}
{
add.f16x2 r4514, r4503, r4504;
}
{
mul.f16x2 r4517, r4514, r4498;
}
{
add.f16x2 r4520, r4506, r4517;
}
{
sub.f16x2 r4523, r4509, r4510;
}
{
mul.f16x2 r4526, r4523, r4500;
}
{
add.f16x2 r4529, r4520, r4526;
}
{
add.f16x2 r4532, r4503, r4504;
}
{
mul.f16x2 r4535, r4532, r4498;
}
{
add.f16x2 r4538, r4506, r4535;
}
{
sub.f16x2 r4541, r4509, r4510;
}
{
mul.f16x2 r4544, r4541, r4500;
}
{
sub.f16x2 r4547, r4538, r4544;
}
{
add.f16x2 r4550, r4509, r4510;
}
{
mul.f16x2 r4553, r4550, r4498;
}
{
add.f16x2 r4556, r4512, r4553;
}
{
sub.f16x2 r4559, r4503, r4504;
}
{
mul.f16x2 r4562, r4559, r4500;
}
{
sub.f16x2 r4565, r4556, r4562;
}
{
add.f16x2 r4568, r4509, r4510;
}
{
mul.f16x2 r4571, r4568, r4498;
}
{
add.f16x2 r4574, r4512, r4571;
}
{
sub.f16x2 r4577, r4503, r4504;
}
{
mul.f16x2 r4580, r4577, r4500;
}
{
add.f16x2 r4583, r4574, r4580;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4586, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4587, {low, high};
}
{
neg.f16x2 r4588, r4587;
}
{
add.f16x2 r4590, r4591, r4592;
}
{
add.f16x2 r4593, r4594, r4590;
}
{
add.f16x2 r4596, r4597, r4598;
}
{
add.f16x2 r4599, r4600, r4596;
}
{
add.f16x2 r4602, r4591, r4592;
}
{
mul.f16x2 r4605, r4602, r4586;
}
{
add.f16x2 r4608, r4594, r4605;
}
{
sub.f16x2 r4611, r4597, r4598;
}
{
mul.f16x2 r4614, r4611, r4588;
}
{
add.f16x2 r4617, r4608, r4614;
}
{
add.f16x2 r4620, r4591, r4592;
}
{
mul.f16x2 r4623, r4620, r4586;
}
{
add.f16x2 r4626, r4594, r4623;
}
{
sub.f16x2 r4629, r4597, r4598;
}
{
mul.f16x2 r4632, r4629, r4588;
}
{
sub.f16x2 r4635, r4626, r4632;
}
{
add.f16x2 r4638, r4597, r4598;
}
{
mul.f16x2 r4641, r4638, r4586;
}
{
add.f16x2 r4644, r4600, r4641;
}
{
sub.f16x2 r4647, r4591, r4592;
}
{
mul.f16x2 r4650, r4647, r4588;
}
{
sub.f16x2 r4653, r4644, r4650;
}
{
add.f16x2 r4656, r4597, r4598;
}
{
mul.f16x2 r4659, r4656, r4586;
}
{
add.f16x2 r4662, r4600, r4659;
}
{
sub.f16x2 r4665, r4591, r4592;
}
{
mul.f16x2 r4668, r4665, r4588;
}
{
add.f16x2 r4671, r4662, r4668;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4674, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4675, {low, high};
}
{
neg.f16x2 r4676, r4675;
}
{
add.f16x2 r4678, r4679, r4680;
}
{
add.f16x2 r4681, r4682, r4678;
}
{
add.f16x2 r4684, r4685, r4686;
}
{
add.f16x2 r4687, r4688, r4684;
}
{
add.f16x2 r4690, r4679, r4680;
}
{
mul.f16x2 r4693, r4690, r4674;
}
{
add.f16x2 r4696, r4682, r4693;
}
{
sub.f16x2 r4699, r4685, r4686;
}
{
mul.f16x2 r4702, r4699, r4676;
}
{
add.f16x2 r4705, r4696, r4702;
}
{
add.f16x2 r4708, r4679, r4680;
}
{
mul.f16x2 r4711, r4708, r4674;
}
{
add.f16x2 r4714, r4682, r4711;
}
{
sub.f16x2 r4717, r4685, r4686;
}
{
mul.f16x2 r4720, r4717, r4676;
}
{
sub.f16x2 r4723, r4714, r4720;
}
{
add.f16x2 r4726, r4685, r4686;
}
{
mul.f16x2 r4729, r4726, r4674;
}
{
add.f16x2 r4732, r4688, r4729;
}
{
sub.f16x2 r4735, r4679, r4680;
}
{
mul.f16x2 r4738, r4735, r4676;
}
{
sub.f16x2 r4741, r4732, r4738;
}
{
add.f16x2 r4744, r4685, r4686;
}
{
mul.f16x2 r4747, r4744, r4674;
}
{
add.f16x2 r4750, r4688, r4747;
}
{
sub.f16x2 r4753, r4679, r4680;
}
{
mul.f16x2 r4756, r4753, r4676;
}
{
add.f16x2 r4759, r4750, r4756;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r4762, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r4763, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r4764, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r4765, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r4768, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
mov.b32 r4769, {low, high};
}
{
mul.f16x2 r4778, r4617, r4762;
}
{
mul.f16x2 r4781, r4653, r4763;
}
{
sub.f16x2 r4784, r4778, r4781;
}
{
mul.f16x2 r4787, r4617, r4763;
}
{
fma.rn.f16x2 r4790, r4653, r4762, r4787;
}
{
mul.f16x2 r4794, r4705, r4764;
}
{
mul.f16x2 r4797, r4741, r4765;
}
{
sub.f16x2 r4800, r4794, r4797;
}
{
mul.f16x2 r4803, r4705, r4765;
}
{
fma.rn.f16x2 r4806, r4741, r4764, r4803;
}
{
mul.f16x2 r4810, r4635, r4764;
}
{
mul.f16x2 r4813, r4671, r4765;
}
{
sub.f16x2 r4816, r4810, r4813;
}
{
mul.f16x2 r4819, r4635, r4765;
}
{
fma.rn.f16x2 r4822, r4671, r4764, r4819;
}
{
mul.f16x2 r4826, r4723, r4768;
}
{
mul.f16x2 r4829, r4759, r4769;
}
{
sub.f16x2 r4832, r4826, r4829;
}
{
mul.f16x2 r4835, r4723, r4769;
}
{
fma.rn.f16x2 r4838, r4759, r4768, r4835;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4842, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4843, {low, high};
}
{
neg.f16x2 r4844, r4843;
}
{
add.f16x2 r4846, r4593, r4681;
}
{
add.f16x2 r4849, r4505, r4846;
}
{
add.f16x2 r4852, r4599, r4687;
}
{
add.f16x2 r4855, r4511, r4852;
}
{
add.f16x2 r4858, r4593, r4681;
}
{
mul.f16x2 r4861, r4858, r4842;
}
{
add.f16x2 r4864, r4505, r4861;
}
{
sub.f16x2 r4867, r4599, r4687;
}
{
mul.f16x2 r4870, r4867, r4844;
}
{
add.f16x2 r4873, r4864, r4870;
}
{
add.f16x2 r4876, r4593, r4681;
}
{
mul.f16x2 r4879, r4876, r4842;
}
{
add.f16x2 r4882, r4505, r4879;
}
{
sub.f16x2 r4885, r4599, r4687;
}
{
mul.f16x2 r4888, r4885, r4844;
}
{
sub.f16x2 r4891, r4882, r4888;
}
{
add.f16x2 r4894, r4599, r4687;
}
{
mul.f16x2 r4897, r4894, r4842;
}
{
add.f16x2 r4900, r4511, r4897;
}
{
sub.f16x2 r4903, r4593, r4681;
}
{
mul.f16x2 r4906, r4903, r4844;
}
{
sub.f16x2 r4909, r4900, r4906;
}
{
add.f16x2 r4912, r4599, r4687;
}
{
mul.f16x2 r4915, r4912, r4842;
}
{
add.f16x2 r4918, r4511, r4915;
}
{
sub.f16x2 r4921, r4593, r4681;
}
{
mul.f16x2 r4924, r4921, r4844;
}
{
add.f16x2 r4927, r4918, r4924;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4930, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4931, {low, high};
}
{
neg.f16x2 r4932, r4931;
}
{
add.f16x2 r4934, r4784, r4800;
}
{
add.f16x2 r4937, r4529, r4934;
}
{
add.f16x2 r4940, r4790, r4806;
}
{
add.f16x2 r4943, r4565, r4940;
}
{
add.f16x2 r4946, r4784, r4800;
}
{
mul.f16x2 r4949, r4946, r4930;
}
{
add.f16x2 r4952, r4529, r4949;
}
{
sub.f16x2 r4955, r4790, r4806;
}
{
mul.f16x2 r4958, r4955, r4932;
}
{
add.f16x2 r4961, r4952, r4958;
}
{
add.f16x2 r4964, r4784, r4800;
}
{
mul.f16x2 r4967, r4964, r4930;
}
{
add.f16x2 r4970, r4529, r4967;
}
{
sub.f16x2 r4973, r4790, r4806;
}
{
mul.f16x2 r4976, r4973, r4932;
}
{
sub.f16x2 r4979, r4970, r4976;
}
{
add.f16x2 r4982, r4790, r4806;
}
{
mul.f16x2 r4985, r4982, r4930;
}
{
add.f16x2 r4988, r4565, r4985;
}
{
sub.f16x2 r4991, r4784, r4800;
}
{
mul.f16x2 r4994, r4991, r4932;
}
{
sub.f16x2 r4997, r4988, r4994;
}
{
add.f16x2 r5000, r4790, r4806;
}
{
mul.f16x2 r5003, r5000, r4930;
}
{
add.f16x2 r5006, r4565, r5003;
}
{
sub.f16x2 r5009, r4784, r4800;
}
{
mul.f16x2 r5012, r5009, r4932;
}
{
add.f16x2 r5015, r5006, r5012;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5019, {low, high};
}
{
neg.f16x2 r5020, r5019;
}
{
add.f16x2 r5022, r4816, r4832;
}
{
add.f16x2 r5025, r4547, r5022;
}
{
add.f16x2 r5028, r4822, r4838;
}
{
add.f16x2 r5031, r4583, r5028;
}
{
add.f16x2 r5034, r4816, r4832;
}
{
mul.f16x2 r5037, r5034, r5018;
}
{
add.f16x2 r5040, r4547, r5037;
}
{
sub.f16x2 r5043, r4822, r4838;
}
{
mul.f16x2 r5046, r5043, r5020;
}
{
add.f16x2 r5049, r5040, r5046;
}
{
add.f16x2 r5052, r4816, r4832;
}
{
mul.f16x2 r5055, r5052, r5018;
}
{
add.f16x2 r5058, r4547, r5055;
}
{
sub.f16x2 r5061, r4822, r4838;
}
{
mul.f16x2 r5064, r5061, r5020;
}
{
sub.f16x2 r5067, r5058, r5064;
}
{
add.f16x2 r5070, r4822, r4838;
}
{
mul.f16x2 r5073, r5070, r5018;
}
{
add.f16x2 r5076, r4583, r5073;
}
{
sub.f16x2 r5079, r4816, r4832;
}
{
mul.f16x2 r5082, r5079, r5020;
}
{
sub.f16x2 r5085, r5076, r5082;
}
{
add.f16x2 r5088, r4822, r4838;
}
{
mul.f16x2 r5091, r5088, r5018;
}
{
add.f16x2 r5094, r4583, r5091;
}
{
sub.f16x2 r5097, r4816, r4832;
}
{
mul.f16x2 r5100, r5097, r5020;
}
{
add.f16x2 r5103, r5094, r5100;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5106, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5107, {low, high};
}
{
neg.f16x2 r5108, r5107;
}
{
add.f16x2 r5110, r5111, r5112;
}
{
add.f16x2 r5113, r5114, r5110;
}
{
add.f16x2 r5116, r5117, r5118;
}
{
add.f16x2 r5119, r5120, r5116;
}
{
add.f16x2 r5122, r5111, r5112;
}
{
mul.f16x2 r5125, r5122, r5106;
}
{
add.f16x2 r5128, r5114, r5125;
}
{
sub.f16x2 r5131, r5117, r5118;
}
{
mul.f16x2 r5134, r5131, r5108;
}
{
add.f16x2 r5137, r5128, r5134;
}
{
add.f16x2 r5140, r5111, r5112;
}
{
mul.f16x2 r5143, r5140, r5106;
}
{
add.f16x2 r5146, r5114, r5143;
}
{
sub.f16x2 r5149, r5117, r5118;
}
{
mul.f16x2 r5152, r5149, r5108;
}
{
sub.f16x2 r5155, r5146, r5152;
}
{
add.f16x2 r5158, r5117, r5118;
}
{
mul.f16x2 r5161, r5158, r5106;
}
{
add.f16x2 r5164, r5120, r5161;
}
{
sub.f16x2 r5167, r5111, r5112;
}
{
mul.f16x2 r5170, r5167, r5108;
}
{
sub.f16x2 r5173, r5164, r5170;
}
{
add.f16x2 r5176, r5117, r5118;
}
{
mul.f16x2 r5179, r5176, r5106;
}
{
add.f16x2 r5182, r5120, r5179;
}
{
sub.f16x2 r5185, r5111, r5112;
}
{
mul.f16x2 r5188, r5185, r5108;
}
{
add.f16x2 r5191, r5182, r5188;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5194, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5195, {low, high};
}
{
neg.f16x2 r5196, r5195;
}
{
add.f16x2 r5198, r5199, r5200;
}
{
add.f16x2 r5201, r5202, r5198;
}
{
add.f16x2 r5204, r5205, r5206;
}
{
add.f16x2 r5207, r5208, r5204;
}
{
add.f16x2 r5210, r5199, r5200;
}
{
mul.f16x2 r5213, r5210, r5194;
}
{
add.f16x2 r5216, r5202, r5213;
}
{
sub.f16x2 r5219, r5205, r5206;
}
{
mul.f16x2 r5222, r5219, r5196;
}
{
add.f16x2 r5225, r5216, r5222;
}
{
add.f16x2 r5228, r5199, r5200;
}
{
mul.f16x2 r5231, r5228, r5194;
}
{
add.f16x2 r5234, r5202, r5231;
}
{
sub.f16x2 r5237, r5205, r5206;
}
{
mul.f16x2 r5240, r5237, r5196;
}
{
sub.f16x2 r5243, r5234, r5240;
}
{
add.f16x2 r5246, r5205, r5206;
}
{
mul.f16x2 r5249, r5246, r5194;
}
{
add.f16x2 r5252, r5208, r5249;
}
{
sub.f16x2 r5255, r5199, r5200;
}
{
mul.f16x2 r5258, r5255, r5196;
}
{
sub.f16x2 r5261, r5252, r5258;
}
{
add.f16x2 r5264, r5205, r5206;
}
{
mul.f16x2 r5267, r5264, r5194;
}
{
add.f16x2 r5270, r5208, r5267;
}
{
sub.f16x2 r5273, r5199, r5200;
}
{
mul.f16x2 r5276, r5273, r5196;
}
{
add.f16x2 r5279, r5270, r5276;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5282, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5283, {low, high};
}
{
neg.f16x2 r5284, r5283;
}
{
add.f16x2 r5286, r5287, r5288;
}
{
add.f16x2 r5289, r5290, r5286;
}
{
add.f16x2 r5292, r5293, r5294;
}
{
add.f16x2 r5295, r5296, r5292;
}
{
add.f16x2 r5298, r5287, r5288;
}
{
mul.f16x2 r5301, r5298, r5282;
}
{
add.f16x2 r5304, r5290, r5301;
}
{
sub.f16x2 r5307, r5293, r5294;
}
{
mul.f16x2 r5310, r5307, r5284;
}
{
add.f16x2 r5313, r5304, r5310;
}
{
add.f16x2 r5316, r5287, r5288;
}
{
mul.f16x2 r5319, r5316, r5282;
}
{
add.f16x2 r5322, r5290, r5319;
}
{
sub.f16x2 r5325, r5293, r5294;
}
{
mul.f16x2 r5328, r5325, r5284;
}
{
sub.f16x2 r5331, r5322, r5328;
}
{
add.f16x2 r5334, r5293, r5294;
}
{
mul.f16x2 r5337, r5334, r5282;
}
{
add.f16x2 r5340, r5296, r5337;
}
{
sub.f16x2 r5343, r5287, r5288;
}
{
mul.f16x2 r5346, r5343, r5284;
}
{
sub.f16x2 r5349, r5340, r5346;
}
{
add.f16x2 r5352, r5293, r5294;
}
{
mul.f16x2 r5355, r5352, r5282;
}
{
add.f16x2 r5358, r5296, r5355;
}
{
sub.f16x2 r5361, r5287, r5288;
}
{
mul.f16x2 r5364, r5361, r5284;
}
{
add.f16x2 r5367, r5358, r5364;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r5370, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r5371, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r5372, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r5373, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r5376, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
mov.b32 r5377, {low, high};
}
{
mul.f16x2 r5386, r5225, r5370;
}
{
mul.f16x2 r5389, r5261, r5371;
}
{
sub.f16x2 r5392, r5386, r5389;
}
{
mul.f16x2 r5395, r5225, r5371;
}
{
fma.rn.f16x2 r5398, r5261, r5370, r5395;
}
{
mul.f16x2 r5402, r5313, r5372;
}
{
mul.f16x2 r5405, r5349, r5373;
}
{
sub.f16x2 r5408, r5402, r5405;
}
{
mul.f16x2 r5411, r5313, r5373;
}
{
fma.rn.f16x2 r5414, r5349, r5372, r5411;
}
{
mul.f16x2 r5418, r5243, r5372;
}
{
mul.f16x2 r5421, r5279, r5373;
}
{
sub.f16x2 r5424, r5418, r5421;
}
{
mul.f16x2 r5427, r5243, r5373;
}
{
fma.rn.f16x2 r5430, r5279, r5372, r5427;
}
{
mul.f16x2 r5434, r5331, r5376;
}
{
mul.f16x2 r5437, r5367, r5377;
}
{
sub.f16x2 r5440, r5434, r5437;
}
{
mul.f16x2 r5443, r5331, r5377;
}
{
fma.rn.f16x2 r5446, r5367, r5376, r5443;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5450, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5451, {low, high};
}
{
neg.f16x2 r5452, r5451;
}
{
add.f16x2 r5454, r5201, r5289;
}
{
add.f16x2 r5457, r5113, r5454;
}
{
add.f16x2 r5460, r5207, r5295;
}
{
add.f16x2 r5463, r5119, r5460;
}
{
add.f16x2 r5466, r5201, r5289;
}
{
mul.f16x2 r5469, r5466, r5450;
}
{
add.f16x2 r5472, r5113, r5469;
}
{
sub.f16x2 r5475, r5207, r5295;
}
{
mul.f16x2 r5478, r5475, r5452;
}
{
add.f16x2 r5481, r5472, r5478;
}
{
add.f16x2 r5484, r5201, r5289;
}
{
mul.f16x2 r5487, r5484, r5450;
}
{
add.f16x2 r5490, r5113, r5487;
}
{
sub.f16x2 r5493, r5207, r5295;
}
{
mul.f16x2 r5496, r5493, r5452;
}
{
sub.f16x2 r5499, r5490, r5496;
}
{
add.f16x2 r5502, r5207, r5295;
}
{
mul.f16x2 r5505, r5502, r5450;
}
{
add.f16x2 r5508, r5119, r5505;
}
{
sub.f16x2 r5511, r5201, r5289;
}
{
mul.f16x2 r5514, r5511, r5452;
}
{
sub.f16x2 r5517, r5508, r5514;
}
{
add.f16x2 r5520, r5207, r5295;
}
{
mul.f16x2 r5523, r5520, r5450;
}
{
add.f16x2 r5526, r5119, r5523;
}
{
sub.f16x2 r5529, r5201, r5289;
}
{
mul.f16x2 r5532, r5529, r5452;
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
neg.f16x2 r5540, r5539;
}
{
add.f16x2 r5542, r5392, r5408;
}
{
add.f16x2 r5545, r5137, r5542;
}
{
add.f16x2 r5548, r5398, r5414;
}
{
add.f16x2 r5551, r5173, r5548;
}
{
add.f16x2 r5554, r5392, r5408;
}
{
mul.f16x2 r5557, r5554, r5538;
}
{
add.f16x2 r5560, r5137, r5557;
}
{
sub.f16x2 r5563, r5398, r5414;
}
{
mul.f16x2 r5566, r5563, r5540;
}
{
add.f16x2 r5569, r5560, r5566;
}
{
add.f16x2 r5572, r5392, r5408;
}
{
mul.f16x2 r5575, r5572, r5538;
}
{
add.f16x2 r5578, r5137, r5575;
}
{
sub.f16x2 r5581, r5398, r5414;
}
{
mul.f16x2 r5584, r5581, r5540;
}
{
sub.f16x2 r5587, r5578, r5584;
}
{
add.f16x2 r5590, r5398, r5414;
}
{
mul.f16x2 r5593, r5590, r5538;
}
{
add.f16x2 r5596, r5173, r5593;
}
{
sub.f16x2 r5599, r5392, r5408;
}
{
mul.f16x2 r5602, r5599, r5540;
}
{
sub.f16x2 r5605, r5596, r5602;
}
{
add.f16x2 r5608, r5398, r5414;
}
{
mul.f16x2 r5611, r5608, r5538;
}
{
add.f16x2 r5614, r5173, r5611;
}
{
sub.f16x2 r5617, r5392, r5408;
}
{
mul.f16x2 r5620, r5617, r5540;
}
{
add.f16x2 r5623, r5614, r5620;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5626, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5627, {low, high};
}
{
neg.f16x2 r5628, r5627;
}
{
add.f16x2 r5630, r5424, r5440;
}
{
add.f16x2 r5633, r5155, r5630;
}
{
add.f16x2 r5636, r5430, r5446;
}
{
add.f16x2 r5639, r5191, r5636;
}
{
add.f16x2 r5642, r5424, r5440;
}
{
mul.f16x2 r5645, r5642, r5626;
}
{
add.f16x2 r5648, r5155, r5645;
}
{
sub.f16x2 r5651, r5430, r5446;
}
{
mul.f16x2 r5654, r5651, r5628;
}
{
add.f16x2 r5657, r5648, r5654;
}
{
add.f16x2 r5660, r5424, r5440;
}
{
mul.f16x2 r5663, r5660, r5626;
}
{
add.f16x2 r5666, r5155, r5663;
}
{
sub.f16x2 r5669, r5430, r5446;
}
{
mul.f16x2 r5672, r5669, r5628;
}
{
sub.f16x2 r5675, r5666, r5672;
}
{
add.f16x2 r5678, r5430, r5446;
}
{
mul.f16x2 r5681, r5678, r5626;
}
{
add.f16x2 r5684, r5191, r5681;
}
{
sub.f16x2 r5687, r5424, r5440;
}
{
mul.f16x2 r5690, r5687, r5628;
}
{
sub.f16x2 r5693, r5684, r5690;
}
{
add.f16x2 r5696, r5430, r5446;
}
{
mul.f16x2 r5699, r5696, r5626;
}
{
add.f16x2 r5702, r5191, r5699;
}
{
sub.f16x2 r5705, r5424, r5440;
}
{
mul.f16x2 r5708, r5705, r5628;
}
{
add.f16x2 r5711, r5702, r5708;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f534;
cvt.rn.f16.f32 high, f534;
mov.b32 r5714, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f536;
cvt.rn.f16.f32 high, f536;
mov.b32 r5715, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f538;
cvt.rn.f16.f32 high, f538;
mov.b32 r5716, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f540;
cvt.rn.f16.f32 high, f540;
mov.b32 r5717, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r5718, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r5719, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f546;
cvt.rn.f16.f32 high, f546;
mov.b32 r5720, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f548;
cvt.rn.f16.f32 high, f548;
mov.b32 r5721, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f550;
cvt.rn.f16.f32 high, f550;
mov.b32 r5722, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f552;
cvt.rn.f16.f32 high, f552;
mov.b32 r5723, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r5724, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r5725, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f558;
cvt.rn.f16.f32 high, f558;
mov.b32 r5726, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f560;
cvt.rn.f16.f32 high, f560;
mov.b32 r5727, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f562;
cvt.rn.f16.f32 high, f562;
mov.b32 r5728, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f564;
cvt.rn.f16.f32 high, f564;
mov.b32 r5729, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f570;
cvt.rn.f16.f32 high, f570;
mov.b32 r5732, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f572;
cvt.rn.f16.f32 high, f572;
mov.b32 r5733, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r5736, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
mov.b32 r5737, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f586;
cvt.rn.f16.f32 high, f586;
mov.b32 r5740, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f588;
cvt.rn.f16.f32 high, f588;
mov.b32 r5741, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5744, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5745, {low, high};
}
{
mul.f16x2 r5766, r4937, r5714;
}
{
mul.f16x2 r5769, r4943, r5715;
}
{
sub.f16x2 r5772, r5766, r5769;
}
{
mul.f16x2 r5775, r4937, r5715;
}
{
fma.rn.f16x2 r5778, r4943, r5714, r5775;
}
{
mul.f16x2 r5782, r5545, r5716;
}
{
mul.f16x2 r5785, r5551, r5717;
}
{
sub.f16x2 r5788, r5782, r5785;
}
{
mul.f16x2 r5791, r5545, r5717;
}
{
fma.rn.f16x2 r5794, r5551, r5716, r5791;
}
{
mul.f16x2 r5798, r5025, r5716;
}
{
mul.f16x2 r5801, r5031, r5717;
}
{
sub.f16x2 r5804, r5798, r5801;
}
{
mul.f16x2 r5807, r5025, r5717;
}
{
fma.rn.f16x2 r5810, r5031, r5716, r5807;
}
{
mul.f16x2 r5814, r5633, r5720;
}
{
mul.f16x2 r5817, r5639, r5721;
}
{
sub.f16x2 r5820, r5814, r5817;
}
{
mul.f16x2 r5823, r5633, r5721;
}
{
fma.rn.f16x2 r5826, r5639, r5720, r5823;
}
{
mul.f16x2 r5830, r4873, r5718;
}
{
mul.f16x2 r5833, r4909, r5719;
}
{
sub.f16x2 r5836, r5830, r5833;
}
{
mul.f16x2 r5839, r4873, r5719;
}
{
fma.rn.f16x2 r5842, r4909, r5718, r5839;
}
{
mul.f16x2 r5846, r5481, r5724;
}
{
mul.f16x2 r5849, r5517, r5725;
}
{
sub.f16x2 r5852, r5846, r5849;
}
{
mul.f16x2 r5855, r5481, r5725;
}
{
fma.rn.f16x2 r5858, r5517, r5724, r5855;
}
{
mul.f16x2 r5862, r4961, r5720;
}
{
mul.f16x2 r5865, r4997, r5721;
}
{
sub.f16x2 r5868, r5862, r5865;
}
{
mul.f16x2 r5871, r4961, r5721;
}
{
fma.rn.f16x2 r5874, r4997, r5720, r5871;
}
{
mul.f16x2 r5878, r5569, r5728;
}
{
mul.f16x2 r5881, r5605, r5729;
}
{
sub.f16x2 r5884, r5878, r5881;
}
{
mul.f16x2 r5887, r5569, r5729;
}
{
fma.rn.f16x2 r5890, r5605, r5728, r5887;
}
{
mul.f16x2 r5894, r5049, r5722;
}
{
mul.f16x2 r5897, r5085, r5723;
}
{
sub.f16x2 r5900, r5894, r5897;
}
{
mul.f16x2 r5903, r5049, r5723;
}
{
fma.rn.f16x2 r5906, r5085, r5722, r5903;
}
{
mul.f16x2 r5910, r5657, r5732;
}
{
mul.f16x2 r5913, r5693, r5733;
}
{
sub.f16x2 r5916, r5910, r5913;
}
{
mul.f16x2 r5919, r5657, r5733;
}
{
fma.rn.f16x2 r5922, r5693, r5732, r5919;
}
{
mul.f16x2 r5926, r4891, r5724;
}
{
mul.f16x2 r5929, r4927, r5725;
}
{
sub.f16x2 r5932, r5926, r5929;
}
{
mul.f16x2 r5935, r4891, r5725;
}
{
fma.rn.f16x2 r5938, r4927, r5724, r5935;
}
{
mul.f16x2 r5942, r5499, r5736;
}
{
mul.f16x2 r5945, r5535, r5737;
}
{
sub.f16x2 r5948, r5942, r5945;
}
{
mul.f16x2 r5951, r5499, r5737;
}
{
fma.rn.f16x2 r5954, r5535, r5736, r5951;
}
{
mul.f16x2 r5958, r4979, r5726;
}
{
mul.f16x2 r5961, r5015, r5727;
}
{
sub.f16x2 r5964, r5958, r5961;
}
{
mul.f16x2 r5967, r4979, r5727;
}
{
fma.rn.f16x2 r5970, r5015, r5726, r5967;
}
{
mul.f16x2 r5974, r5587, r5740;
}
{
mul.f16x2 r5977, r5623, r5741;
}
{
sub.f16x2 r5980, r5974, r5977;
}
{
mul.f16x2 r5983, r5587, r5741;
}
{
fma.rn.f16x2 r5986, r5623, r5740, r5983;
}
{
mul.f16x2 r5990, r5067, r5728;
}
{
mul.f16x2 r5993, r5103, r5729;
}
{
sub.f16x2 r5996, r5990, r5993;
}
{
mul.f16x2 r5999, r5067, r5729;
}
{
fma.rn.f16x2 r6002, r5103, r5728, r5999;
}
{
mul.f16x2 r6006, r5675, r5744;
}
{
mul.f16x2 r6009, r5711, r5745;
}
{
sub.f16x2 r6012, r6006, r6009;
}
{
mul.f16x2 r6015, r5675, r5745;
}
{
fma.rn.f16x2 r6018, r5711, r5744, r6015;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6022, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6023, {low, high};
}
{
neg.f16x2 r6024, r6023;
}
{
add.f16x2 r6026, r4849, r5457;
}
{
add.f16x2 r6029, r4241, r6026;
}
{
add.f16x2 r6032, r4855, r5463;
}
{
add.f16x2 r6035, r4247, r6032;
}
{
add.f16x2 r6038, r4849, r5457;
}
{
mul.f16x2 r6041, r6038, r6022;
}
{
add.f16x2 r6044, r4241, r6041;
}
{
sub.f16x2 r6047, r4855, r5463;
}
{
mul.f16x2 r6050, r6047, r6024;
}
{
add.f16x2 r6053, r6044, r6050;
}
{
add.f16x2 r6056, r4849, r5457;
}
{
mul.f16x2 r6059, r6056, r6022;
}
{
add.f16x2 r6062, r4241, r6059;
}
{
sub.f16x2 r6065, r4855, r5463;
}
{
mul.f16x2 r6068, r6065, r6024;
}
{
sub.f16x2 r6071, r6062, r6068;
}
{
add.f16x2 r6074, r4855, r5463;
}
{
mul.f16x2 r6077, r6074, r6022;
}
{
add.f16x2 r6080, r4247, r6077;
}
{
sub.f16x2 r6083, r4849, r5457;
}
{
mul.f16x2 r6086, r6083, r6024;
}
{
sub.f16x2 r6089, r6080, r6086;
}
{
add.f16x2 r6092, r4855, r5463;
}
{
mul.f16x2 r6095, r6092, r6022;
}
{
add.f16x2 r6098, r4247, r6095;
}
{
sub.f16x2 r6101, r4849, r5457;
}
{
mul.f16x2 r6104, r6101, r6024;
}
{
add.f16x2 r6107, r6098, r6104;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6110, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6111, {low, high};
}
{
neg.f16x2 r6112, r6111;
}
{
add.f16x2 r6114, r5772, r5788;
}
{
add.f16x2 r6117, r4329, r6114;
}
{
add.f16x2 r6120, r5778, r5794;
}
{
add.f16x2 r6123, r4335, r6120;
}
{
add.f16x2 r6126, r5772, r5788;
}
{
mul.f16x2 r6129, r6126, r6110;
}
{
add.f16x2 r6132, r4329, r6129;
}
{
sub.f16x2 r6135, r5778, r5794;
}
{
mul.f16x2 r6138, r6135, r6112;
}
{
add.f16x2 r6141, r6132, r6138;
}
{
add.f16x2 r6144, r5772, r5788;
}
{
mul.f16x2 r6147, r6144, r6110;
}
{
add.f16x2 r6150, r4329, r6147;
}
{
sub.f16x2 r6153, r5778, r5794;
}
{
mul.f16x2 r6156, r6153, r6112;
}
{
sub.f16x2 r6159, r6150, r6156;
}
{
add.f16x2 r6162, r5778, r5794;
}
{
mul.f16x2 r6165, r6162, r6110;
}
{
add.f16x2 r6168, r4335, r6165;
}
{
sub.f16x2 r6171, r5772, r5788;
}
{
mul.f16x2 r6174, r6171, r6112;
}
{
sub.f16x2 r6177, r6168, r6174;
}
{
add.f16x2 r6180, r5778, r5794;
}
{
mul.f16x2 r6183, r6180, r6110;
}
{
add.f16x2 r6186, r4335, r6183;
}
{
sub.f16x2 r6189, r5772, r5788;
}
{
mul.f16x2 r6192, r6189, r6112;
}
{
add.f16x2 r6195, r6186, r6192;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6198, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6199, {low, high};
}
{
neg.f16x2 r6200, r6199;
}
{
add.f16x2 r6202, r5804, r5820;
}
{
add.f16x2 r6205, r4417, r6202;
}
{
add.f16x2 r6208, r5810, r5826;
}
{
add.f16x2 r6211, r4423, r6208;
}
{
add.f16x2 r6214, r5804, r5820;
}
{
mul.f16x2 r6217, r6214, r6198;
}
{
add.f16x2 r6220, r4417, r6217;
}
{
sub.f16x2 r6223, r5810, r5826;
}
{
mul.f16x2 r6226, r6223, r6200;
}
{
add.f16x2 r6229, r6220, r6226;
}
{
add.f16x2 r6232, r5804, r5820;
}
{
mul.f16x2 r6235, r6232, r6198;
}
{
add.f16x2 r6238, r4417, r6235;
}
{
sub.f16x2 r6241, r5810, r5826;
}
{
mul.f16x2 r6244, r6241, r6200;
}
{
sub.f16x2 r6247, r6238, r6244;
}
{
add.f16x2 r6250, r5810, r5826;
}
{
mul.f16x2 r6253, r6250, r6198;
}
{
add.f16x2 r6256, r4423, r6253;
}
{
sub.f16x2 r6259, r5804, r5820;
}
{
mul.f16x2 r6262, r6259, r6200;
}
{
sub.f16x2 r6265, r6256, r6262;
}
{
add.f16x2 r6268, r5810, r5826;
}
{
mul.f16x2 r6271, r6268, r6198;
}
{
add.f16x2 r6274, r4423, r6271;
}
{
sub.f16x2 r6277, r5804, r5820;
}
{
mul.f16x2 r6280, r6277, r6200;
}
{
add.f16x2 r6283, r6274, r6280;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6286, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6287, {low, high};
}
{
neg.f16x2 r6288, r6287;
}
{
add.f16x2 r6290, r5836, r5852;
}
{
add.f16x2 r6293, r4265, r6290;
}
{
add.f16x2 r6296, r5842, r5858;
}
{
add.f16x2 r6299, r4301, r6296;
}
{
add.f16x2 r6302, r5836, r5852;
}
{
mul.f16x2 r6305, r6302, r6286;
}
{
add.f16x2 r6308, r4265, r6305;
}
{
sub.f16x2 r6311, r5842, r5858;
}
{
mul.f16x2 r6314, r6311, r6288;
}
{
add.f16x2 r6317, r6308, r6314;
}
{
add.f16x2 r6320, r5836, r5852;
}
{
mul.f16x2 r6323, r6320, r6286;
}
{
add.f16x2 r6326, r4265, r6323;
}
{
sub.f16x2 r6329, r5842, r5858;
}
{
mul.f16x2 r6332, r6329, r6288;
}
{
sub.f16x2 r6335, r6326, r6332;
}
{
add.f16x2 r6338, r5842, r5858;
}
{
mul.f16x2 r6341, r6338, r6286;
}
{
add.f16x2 r6344, r4301, r6341;
}
{
sub.f16x2 r6347, r5836, r5852;
}
{
mul.f16x2 r6350, r6347, r6288;
}
{
sub.f16x2 r6353, r6344, r6350;
}
{
add.f16x2 r6356, r5842, r5858;
}
{
mul.f16x2 r6359, r6356, r6286;
}
{
add.f16x2 r6362, r4301, r6359;
}
{
sub.f16x2 r6365, r5836, r5852;
}
{
mul.f16x2 r6368, r6365, r6288;
}
{
add.f16x2 r6371, r6362, r6368;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6374, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6375, {low, high};
}
{
neg.f16x2 r6376, r6375;
}
{
add.f16x2 r6378, r5868, r5884;
}
{
add.f16x2 r6381, r4353, r6378;
}
{
add.f16x2 r6384, r5874, r5890;
}
{
add.f16x2 r6387, r4389, r6384;
}
{
add.f16x2 r6390, r5868, r5884;
}
{
mul.f16x2 r6393, r6390, r6374;
}
{
add.f16x2 r6396, r4353, r6393;
}
{
sub.f16x2 r6399, r5874, r5890;
}
{
mul.f16x2 r6402, r6399, r6376;
}
{
add.f16x2 r6405, r6396, r6402;
}
{
add.f16x2 r6408, r5868, r5884;
}
{
mul.f16x2 r6411, r6408, r6374;
}
{
add.f16x2 r6414, r4353, r6411;
}
{
sub.f16x2 r6417, r5874, r5890;
}
{
mul.f16x2 r6420, r6417, r6376;
}
{
sub.f16x2 r6423, r6414, r6420;
}
{
add.f16x2 r6426, r5874, r5890;
}
{
mul.f16x2 r6429, r6426, r6374;
}
{
add.f16x2 r6432, r4389, r6429;
}
{
sub.f16x2 r6435, r5868, r5884;
}
{
mul.f16x2 r6438, r6435, r6376;
}
{
sub.f16x2 r6441, r6432, r6438;
}
{
add.f16x2 r6444, r5874, r5890;
}
{
mul.f16x2 r6447, r6444, r6374;
}
{
add.f16x2 r6450, r4389, r6447;
}
{
sub.f16x2 r6453, r5868, r5884;
}
{
mul.f16x2 r6456, r6453, r6376;
}
{
add.f16x2 r6459, r6450, r6456;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6462, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6463, {low, high};
}
{
neg.f16x2 r6464, r6463;
}
{
add.f16x2 r6466, r5900, r5916;
}
{
add.f16x2 r6469, r4441, r6466;
}
{
add.f16x2 r6472, r5906, r5922;
}
{
add.f16x2 r6475, r4477, r6472;
}
{
add.f16x2 r6478, r5900, r5916;
}
{
mul.f16x2 r6481, r6478, r6462;
}
{
add.f16x2 r6484, r4441, r6481;
}
{
sub.f16x2 r6487, r5906, r5922;
}
{
mul.f16x2 r6490, r6487, r6464;
}
{
add.f16x2 r6493, r6484, r6490;
}
{
add.f16x2 r6496, r5900, r5916;
}
{
mul.f16x2 r6499, r6496, r6462;
}
{
add.f16x2 r6502, r4441, r6499;
}
{
sub.f16x2 r6505, r5906, r5922;
}
{
mul.f16x2 r6508, r6505, r6464;
}
{
sub.f16x2 r6511, r6502, r6508;
}
{
add.f16x2 r6514, r5906, r5922;
}
{
mul.f16x2 r6517, r6514, r6462;
}
{
add.f16x2 r6520, r4477, r6517;
}
{
sub.f16x2 r6523, r5900, r5916;
}
{
mul.f16x2 r6526, r6523, r6464;
}
{
sub.f16x2 r6529, r6520, r6526;
}
{
add.f16x2 r6532, r5906, r5922;
}
{
mul.f16x2 r6535, r6532, r6462;
}
{
add.f16x2 r6538, r4477, r6535;
}
{
sub.f16x2 r6541, r5900, r5916;
}
{
mul.f16x2 r6544, r6541, r6464;
}
{
add.f16x2 r6547, r6538, r6544;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6550, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6551, {low, high};
}
{
neg.f16x2 r6552, r6551;
}
{
add.f16x2 r6554, r5932, r5948;
}
{
add.f16x2 r6557, r4283, r6554;
}
{
add.f16x2 r6560, r5938, r5954;
}
{
add.f16x2 r6563, r4319, r6560;
}
{
add.f16x2 r6566, r5932, r5948;
}
{
mul.f16x2 r6569, r6566, r6550;
}
{
add.f16x2 r6572, r4283, r6569;
}
{
sub.f16x2 r6575, r5938, r5954;
}
{
mul.f16x2 r6578, r6575, r6552;
}
{
add.f16x2 r6581, r6572, r6578;
}
{
add.f16x2 r6584, r5932, r5948;
}
{
mul.f16x2 r6587, r6584, r6550;
}
{
add.f16x2 r6590, r4283, r6587;
}
{
sub.f16x2 r6593, r5938, r5954;
}
{
mul.f16x2 r6596, r6593, r6552;
}
{
sub.f16x2 r6599, r6590, r6596;
}
{
add.f16x2 r6602, r5938, r5954;
}
{
mul.f16x2 r6605, r6602, r6550;
}
{
add.f16x2 r6608, r4319, r6605;
}
{
sub.f16x2 r6611, r5932, r5948;
}
{
mul.f16x2 r6614, r6611, r6552;
}
{
sub.f16x2 r6617, r6608, r6614;
}
{
add.f16x2 r6620, r5938, r5954;
}
{
mul.f16x2 r6623, r6620, r6550;
}
{
add.f16x2 r6626, r4319, r6623;
}
{
sub.f16x2 r6629, r5932, r5948;
}
{
mul.f16x2 r6632, r6629, r6552;
}
{
add.f16x2 r6635, r6626, r6632;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6638, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6639, {low, high};
}
{
neg.f16x2 r6640, r6639;
}
{
add.f16x2 r6642, r5964, r5980;
}
{
add.f16x2 r6645, r4371, r6642;
}
{
add.f16x2 r6648, r5970, r5986;
}
{
add.f16x2 r6651, r4407, r6648;
}
{
add.f16x2 r6654, r5964, r5980;
}
{
mul.f16x2 r6657, r6654, r6638;
}
{
add.f16x2 r6660, r4371, r6657;
}
{
sub.f16x2 r6663, r5970, r5986;
}
{
mul.f16x2 r6666, r6663, r6640;
}
{
add.f16x2 r6669, r6660, r6666;
}
{
add.f16x2 r6672, r5964, r5980;
}
{
mul.f16x2 r6675, r6672, r6638;
}
{
add.f16x2 r6678, r4371, r6675;
}
{
sub.f16x2 r6681, r5970, r5986;
}
{
mul.f16x2 r6684, r6681, r6640;
}
{
sub.f16x2 r6687, r6678, r6684;
}
{
add.f16x2 r6690, r5970, r5986;
}
{
mul.f16x2 r6693, r6690, r6638;
}
{
add.f16x2 r6696, r4407, r6693;
}
{
sub.f16x2 r6699, r5964, r5980;
}
{
mul.f16x2 r6702, r6699, r6640;
}
{
sub.f16x2 r6705, r6696, r6702;
}
{
add.f16x2 r6708, r5970, r5986;
}
{
mul.f16x2 r6711, r6708, r6638;
}
{
add.f16x2 r6714, r4407, r6711;
}
{
sub.f16x2 r6717, r5964, r5980;
}
{
mul.f16x2 r6720, r6717, r6640;
}
{
add.f16x2 r6723, r6714, r6720;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6726, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6727, {low, high};
}
{
neg.f16x2 r6728, r6727;
}
{
add.f16x2 r6730, r5996, r6012;
}
{
add.f16x2 r6733, r4459, r6730;
}
{
add.f16x2 r6736, r6002, r6018;
}
{
add.f16x2 r6739, r4495, r6736;
}
{
add.f16x2 r6742, r5996, r6012;
}
{
mul.f16x2 r6745, r6742, r6726;
}
{
add.f16x2 r6748, r4459, r6745;
}
{
sub.f16x2 r6751, r6002, r6018;
}
{
mul.f16x2 r6754, r6751, r6728;
}
{
add.f16x2 r6757, r6748, r6754;
}
{
add.f16x2 r6760, r5996, r6012;
}
{
mul.f16x2 r6763, r6760, r6726;
}
{
add.f16x2 r6766, r4459, r6763;
}
{
sub.f16x2 r6769, r6002, r6018;
}
{
mul.f16x2 r6772, r6769, r6728;
}
{
sub.f16x2 r6775, r6766, r6772;
}
{
add.f16x2 r6778, r6002, r6018;
}
{
mul.f16x2 r6781, r6778, r6726;
}
{
add.f16x2 r6784, r4495, r6781;
}
{
sub.f16x2 r6787, r5996, r6012;
}
{
mul.f16x2 r6790, r6787, r6728;
}
{
sub.f16x2 r6793, r6784, r6790;
}
{
add.f16x2 r6796, r6002, r6018;
}
{
mul.f16x2 r6799, r6796, r6726;
}
{
add.f16x2 r6802, r4495, r6799;
}
{
sub.f16x2 r6805, r5996, r6012;
}
{
mul.f16x2 r6808, r6805, r6728;
}
{
add.f16x2 r6811, r6802, r6808;
}
mul.wide.u32 rd4, r8577, 795364315;
shr.u64 rd5, rd4, 32;
cvt.u32.u64 r8581, rd5;
sub.s32 r8582, r8577, r8581;
shr.u32 r8583, r8582, 1;
add.s32 r8584, r8583, r8581;
shr.u32 r8585, r8584, 4;
cvt.rn.f32.u32 f768, r8585;
mul.f32 f769, f768, 0f3D9EDD1F;
cos.approx.f32 f673, f769;
sin.approx.f32 f770, f769;
neg.f32 f674, f770;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f673;
cvt.rn.f16.f32 high, f674;
mov.b32 r6814, {low, high};
}
mul.lo.s32 r8586, r8585, 27;
sub.s32 r8587, r8577, r8586;
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6817, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6819, {high, high};
}
{
mul.f16x2 r6821, r6123, r6819;
}
{
neg.f16x2 r6824, r6821;
}
{
fma.rn.f16x2 r6826, r6117, r6817, r6824;
}
{
mul.f16x2 r6830, r6117, r6819;
}
{
fma.rn.f16x2 r6833, r6123, r6817, r6830;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6837, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6839, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6841, {low, high};
}
{
mul.f16x2 r6842, r6839, r6841;
}
{
mul.f16x2 r6845, r6814, r6837;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6848, {high, low};
}
{
fma.rn.f16x2 r6850, r6842, r6848, r6845;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6850;
mov.b32 r6854, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6850;
mov.b32 r6856, {high, high};
}
{
mul.f16x2 r6858, r6211, r6856;
}
{
neg.f16x2 r6861, r6858;
}
{
fma.rn.f16x2 r6863, r6205, r6854, r6861;
}
{
mul.f16x2 r6867, r6205, r6856;
}
{
fma.rn.f16x2 r6870, r6211, r6854, r6867;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6874, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6876, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6878, {low, high};
}
{
mul.f16x2 r6879, r6876, r6878;
}
{
mul.f16x2 r6882, r6850, r6874;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6850;
mov.b32 r6885, {high, low};
}
{
fma.rn.f16x2 r6887, r6879, r6885, r6882;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6887;
mov.b32 r6891, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6887;
mov.b32 r6893, {high, high};
}
{
mul.f16x2 r6895, r6299, r6893;
}
{
neg.f16x2 r6898, r6895;
}
{
fma.rn.f16x2 r6900, r6293, r6891, r6898;
}
{
mul.f16x2 r6904, r6293, r6893;
}
{
fma.rn.f16x2 r6907, r6299, r6891, r6904;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6911, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6913, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6915, {low, high};
}
{
mul.f16x2 r6916, r6913, r6915;
}
{
mul.f16x2 r6919, r6887, r6911;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6887;
mov.b32 r6922, {high, low};
}
{
fma.rn.f16x2 r6924, r6916, r6922, r6919;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6924;
mov.b32 r6928, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6924;
mov.b32 r6930, {high, high};
}
{
mul.f16x2 r6932, r6387, r6930;
}
{
neg.f16x2 r6935, r6932;
}
{
fma.rn.f16x2 r6937, r6381, r6928, r6935;
}
{
mul.f16x2 r6941, r6381, r6930;
}
{
fma.rn.f16x2 r6944, r6387, r6928, r6941;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6948, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6950, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6952, {low, high};
}
{
mul.f16x2 r6953, r6950, r6952;
}
{
mul.f16x2 r6956, r6924, r6948;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6924;
mov.b32 r6959, {high, low};
}
{
fma.rn.f16x2 r6961, r6953, r6959, r6956;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6961;
mov.b32 r6965, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6961;
mov.b32 r6967, {high, high};
}
{
mul.f16x2 r6969, r6475, r6967;
}
{
neg.f16x2 r6972, r6969;
}
{
fma.rn.f16x2 r6974, r6469, r6965, r6972;
}
{
mul.f16x2 r6978, r6469, r6967;
}
{
fma.rn.f16x2 r6981, r6475, r6965, r6978;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6985, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6987, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6989, {low, high};
}
{
mul.f16x2 r6990, r6987, r6989;
}
{
mul.f16x2 r6993, r6961, r6985;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6961;
mov.b32 r6996, {high, low};
}
{
fma.rn.f16x2 r6998, r6990, r6996, r6993;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6998;
mov.b32 r7002, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6998;
mov.b32 r7004, {high, high};
}
{
mul.f16x2 r7006, r6563, r7004;
}
{
neg.f16x2 r7009, r7006;
}
{
fma.rn.f16x2 r7011, r6557, r7002, r7009;
}
{
mul.f16x2 r7015, r6557, r7004;
}
{
fma.rn.f16x2 r7018, r6563, r7002, r7015;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7022, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7024, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7026, {low, high};
}
{
mul.f16x2 r7027, r7024, r7026;
}
{
mul.f16x2 r7030, r6998, r7022;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6998;
mov.b32 r7033, {high, low};
}
{
fma.rn.f16x2 r7035, r7027, r7033, r7030;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7035;
mov.b32 r7039, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7035;
mov.b32 r7041, {high, high};
}
{
mul.f16x2 r7043, r6651, r7041;
}
{
neg.f16x2 r7046, r7043;
}
{
fma.rn.f16x2 r7048, r6645, r7039, r7046;
}
{
mul.f16x2 r7052, r6645, r7041;
}
{
fma.rn.f16x2 r7055, r6651, r7039, r7052;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7059, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7061, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7063, {low, high};
}
{
mul.f16x2 r7064, r7061, r7063;
}
{
mul.f16x2 r7067, r7035, r7059;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7035;
mov.b32 r7070, {high, low};
}
{
fma.rn.f16x2 r7072, r7064, r7070, r7067;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7072;
mov.b32 r7076, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7072;
mov.b32 r7078, {high, high};
}
{
mul.f16x2 r7080, r6739, r7078;
}
{
neg.f16x2 r7083, r7080;
}
{
fma.rn.f16x2 r7085, r6733, r7076, r7083;
}
{
mul.f16x2 r7089, r6733, r7078;
}
{
fma.rn.f16x2 r7092, r6739, r7076, r7089;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7096, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7098, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7100, {low, high};
}
{
mul.f16x2 r7101, r7098, r7100;
}
{
mul.f16x2 r7104, r7072, r7096;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7072;
mov.b32 r7107, {high, low};
}
{
fma.rn.f16x2 r7109, r7101, r7107, r7104;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7109;
mov.b32 r7113, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7109;
mov.b32 r7115, {high, high};
}
{
mul.f16x2 r7117, r6089, r7115;
}
{
neg.f16x2 r7120, r7117;
}
{
fma.rn.f16x2 r7122, r6053, r7113, r7120;
}
{
mul.f16x2 r7126, r6053, r7115;
}
{
fma.rn.f16x2 r7129, r6089, r7113, r7126;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7133, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7135, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7137, {low, high};
}
{
mul.f16x2 r7138, r7135, r7137;
}
{
mul.f16x2 r7141, r7109, r7133;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7109;
mov.b32 r7144, {high, low};
}
{
fma.rn.f16x2 r7146, r7138, r7144, r7141;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7146;
mov.b32 r7150, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7146;
mov.b32 r7152, {high, high};
}
{
mul.f16x2 r7154, r6177, r7152;
}
{
neg.f16x2 r7157, r7154;
}
{
fma.rn.f16x2 r7159, r6141, r7150, r7157;
}
{
mul.f16x2 r7163, r6141, r7152;
}
{
fma.rn.f16x2 r7166, r6177, r7150, r7163;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7170, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7172, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7174, {low, high};
}
{
mul.f16x2 r7175, r7172, r7174;
}
{
mul.f16x2 r7178, r7146, r7170;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7146;
mov.b32 r7181, {high, low};
}
{
fma.rn.f16x2 r7183, r7175, r7181, r7178;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7183;
mov.b32 r7187, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7183;
mov.b32 r7189, {high, high};
}
{
mul.f16x2 r7191, r6265, r7189;
}
{
neg.f16x2 r7194, r7191;
}
{
fma.rn.f16x2 r7196, r6229, r7187, r7194;
}
{
mul.f16x2 r7200, r6229, r7189;
}
{
fma.rn.f16x2 r7203, r6265, r7187, r7200;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7207, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7209, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7211, {low, high};
}
{
mul.f16x2 r7212, r7209, r7211;
}
{
mul.f16x2 r7215, r7183, r7207;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7183;
mov.b32 r7218, {high, low};
}
{
fma.rn.f16x2 r7220, r7212, r7218, r7215;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7220;
mov.b32 r7224, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7220;
mov.b32 r7226, {high, high};
}
{
mul.f16x2 r7228, r6353, r7226;
}
{
neg.f16x2 r7231, r7228;
}
{
fma.rn.f16x2 r7233, r6317, r7224, r7231;
}
{
mul.f16x2 r7237, r6317, r7226;
}
{
fma.rn.f16x2 r7240, r6353, r7224, r7237;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7244, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7246, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7248, {low, high};
}
{
mul.f16x2 r7249, r7246, r7248;
}
{
mul.f16x2 r7252, r7220, r7244;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7220;
mov.b32 r7255, {high, low};
}
{
fma.rn.f16x2 r7257, r7249, r7255, r7252;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7257;
mov.b32 r7261, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7257;
mov.b32 r7263, {high, high};
}
{
mul.f16x2 r7265, r6441, r7263;
}
{
neg.f16x2 r7268, r7265;
}
{
fma.rn.f16x2 r7270, r6405, r7261, r7268;
}
{
mul.f16x2 r7274, r6405, r7263;
}
{
fma.rn.f16x2 r7277, r6441, r7261, r7274;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7281, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7283, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7285, {low, high};
}
{
mul.f16x2 r7286, r7283, r7285;
}
{
mul.f16x2 r7289, r7257, r7281;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7257;
mov.b32 r7292, {high, low};
}
{
fma.rn.f16x2 r7294, r7286, r7292, r7289;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7294;
mov.b32 r7298, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7294;
mov.b32 r7300, {high, high};
}
{
mul.f16x2 r7302, r6529, r7300;
}
{
neg.f16x2 r7305, r7302;
}
{
fma.rn.f16x2 r7307, r6493, r7298, r7305;
}
{
mul.f16x2 r7311, r6493, r7300;
}
{
fma.rn.f16x2 r7314, r6529, r7298, r7311;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7318, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7320, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7322, {low, high};
}
{
mul.f16x2 r7323, r7320, r7322;
}
{
mul.f16x2 r7326, r7294, r7318;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7294;
mov.b32 r7329, {high, low};
}
{
fma.rn.f16x2 r7331, r7323, r7329, r7326;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7331;
mov.b32 r7335, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7331;
mov.b32 r7337, {high, high};
}
{
mul.f16x2 r7339, r6617, r7337;
}
{
neg.f16x2 r7342, r7339;
}
{
fma.rn.f16x2 r7344, r6581, r7335, r7342;
}
{
mul.f16x2 r7348, r6581, r7337;
}
{
fma.rn.f16x2 r7351, r6617, r7335, r7348;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7355, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7357, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7359, {low, high};
}
{
mul.f16x2 r7360, r7357, r7359;
}
{
mul.f16x2 r7363, r7331, r7355;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7331;
mov.b32 r7366, {high, low};
}
{
fma.rn.f16x2 r7368, r7360, r7366, r7363;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7368;
mov.b32 r7372, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7368;
mov.b32 r7374, {high, high};
}
{
mul.f16x2 r7376, r6705, r7374;
}
{
neg.f16x2 r7379, r7376;
}
{
fma.rn.f16x2 r7381, r6669, r7372, r7379;
}
{
mul.f16x2 r7385, r6669, r7374;
}
{
fma.rn.f16x2 r7388, r6705, r7372, r7385;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7392, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7394, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7396, {low, high};
}
{
mul.f16x2 r7397, r7394, r7396;
}
{
mul.f16x2 r7400, r7368, r7392;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7368;
mov.b32 r7403, {high, low};
}
{
fma.rn.f16x2 r7405, r7397, r7403, r7400;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7405;
mov.b32 r7409, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7405;
mov.b32 r7411, {high, high};
}
{
mul.f16x2 r7413, r6793, r7411;
}
{
neg.f16x2 r7416, r7413;
}
{
fma.rn.f16x2 r7418, r6757, r7409, r7416;
}
{
mul.f16x2 r7422, r6757, r7411;
}
{
fma.rn.f16x2 r7425, r6793, r7409, r7422;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7429, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7431, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7433, {low, high};
}
{
mul.f16x2 r7434, r7431, r7433;
}
{
mul.f16x2 r7437, r7405, r7429;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7405;
mov.b32 r7440, {high, low};
}
{
fma.rn.f16x2 r7442, r7434, r7440, r7437;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7442;
mov.b32 r7446, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7442;
mov.b32 r7448, {high, high};
}
{
mul.f16x2 r7450, r6107, r7448;
}
{
neg.f16x2 r7453, r7450;
}
{
fma.rn.f16x2 r7455, r6071, r7446, r7453;
}
{
mul.f16x2 r7459, r6071, r7448;
}
{
fma.rn.f16x2 r7462, r6107, r7446, r7459;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7466, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7468, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7470, {low, high};
}
{
mul.f16x2 r7471, r7468, r7470;
}
{
mul.f16x2 r7474, r7442, r7466;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7442;
mov.b32 r7477, {high, low};
}
{
fma.rn.f16x2 r7479, r7471, r7477, r7474;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7479;
mov.b32 r7483, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7479;
mov.b32 r7485, {high, high};
}
{
mul.f16x2 r7487, r6195, r7485;
}
{
neg.f16x2 r7490, r7487;
}
{
fma.rn.f16x2 r7492, r6159, r7483, r7490;
}
{
mul.f16x2 r7496, r6159, r7485;
}
{
fma.rn.f16x2 r7499, r6195, r7483, r7496;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7503, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7505, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7507, {low, high};
}
{
mul.f16x2 r7508, r7505, r7507;
}
{
mul.f16x2 r7511, r7479, r7503;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7479;
mov.b32 r7514, {high, low};
}
{
fma.rn.f16x2 r7516, r7508, r7514, r7511;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7516;
mov.b32 r7520, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7516;
mov.b32 r7522, {high, high};
}
{
mul.f16x2 r7524, r6283, r7522;
}
{
neg.f16x2 r7527, r7524;
}
{
fma.rn.f16x2 r7529, r6247, r7520, r7527;
}
{
mul.f16x2 r7533, r6247, r7522;
}
{
fma.rn.f16x2 r7536, r6283, r7520, r7533;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7540, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7542, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7544, {low, high};
}
{
mul.f16x2 r7545, r7542, r7544;
}
{
mul.f16x2 r7548, r7516, r7540;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7516;
mov.b32 r7551, {high, low};
}
{
fma.rn.f16x2 r7553, r7545, r7551, r7548;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7553;
mov.b32 r7557, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7553;
mov.b32 r7559, {high, high};
}
{
mul.f16x2 r7561, r6371, r7559;
}
{
neg.f16x2 r7564, r7561;
}
{
fma.rn.f16x2 r7566, r6335, r7557, r7564;
}
{
mul.f16x2 r7570, r6335, r7559;
}
{
fma.rn.f16x2 r7573, r6371, r7557, r7570;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7577, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7579, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7581, {low, high};
}
{
mul.f16x2 r7582, r7579, r7581;
}
{
mul.f16x2 r7585, r7553, r7577;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7553;
mov.b32 r7588, {high, low};
}
{
fma.rn.f16x2 r7590, r7582, r7588, r7585;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7590;
mov.b32 r7594, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7590;
mov.b32 r7596, {high, high};
}
{
mul.f16x2 r7598, r6459, r7596;
}
{
neg.f16x2 r7601, r7598;
}
{
fma.rn.f16x2 r7603, r6423, r7594, r7601;
}
{
mul.f16x2 r7607, r6423, r7596;
}
{
fma.rn.f16x2 r7610, r6459, r7594, r7607;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7614, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7616, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7618, {low, high};
}
{
mul.f16x2 r7619, r7616, r7618;
}
{
mul.f16x2 r7622, r7590, r7614;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7590;
mov.b32 r7625, {high, low};
}
{
fma.rn.f16x2 r7627, r7619, r7625, r7622;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7627;
mov.b32 r7631, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7627;
mov.b32 r7633, {high, high};
}
{
mul.f16x2 r7635, r6547, r7633;
}
{
neg.f16x2 r7638, r7635;
}
{
fma.rn.f16x2 r7640, r6511, r7631, r7638;
}
{
mul.f16x2 r7644, r6511, r7633;
}
{
fma.rn.f16x2 r7647, r6547, r7631, r7644;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7651, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7653, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7655, {low, high};
}
{
mul.f16x2 r7656, r7653, r7655;
}
{
mul.f16x2 r7659, r7627, r7651;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7627;
mov.b32 r7662, {high, low};
}
{
fma.rn.f16x2 r7664, r7656, r7662, r7659;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7664;
mov.b32 r7668, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7664;
mov.b32 r7670, {high, high};
}
{
mul.f16x2 r7672, r6635, r7670;
}
{
neg.f16x2 r7675, r7672;
}
{
fma.rn.f16x2 r7677, r6599, r7668, r7675;
}
{
mul.f16x2 r7681, r6599, r7670;
}
{
fma.rn.f16x2 r7684, r6635, r7668, r7681;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7688, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7690, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7692, {low, high};
}
{
mul.f16x2 r7693, r7690, r7692;
}
{
mul.f16x2 r7696, r7664, r7688;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7664;
mov.b32 r7699, {high, low};
}
{
fma.rn.f16x2 r7701, r7693, r7699, r7696;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7701;
mov.b32 r7705, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7701;
mov.b32 r7707, {high, high};
}
{
mul.f16x2 r7709, r6723, r7707;
}
{
neg.f16x2 r7712, r7709;
}
{
fma.rn.f16x2 r7714, r6687, r7705, r7712;
}
{
mul.f16x2 r7718, r6687, r7707;
}
{
fma.rn.f16x2 r7721, r6723, r7705, r7718;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7725, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7727, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7729, {low, high};
}
{
mul.f16x2 r7730, r7727, r7729;
}
{
mul.f16x2 r7733, r7701, r7725;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7701;
mov.b32 r7736, {high, low};
}
{
fma.rn.f16x2 r7738, r7730, r7736, r7733;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7738;
mov.b32 r7742, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7738;
mov.b32 r7744, {high, high};
}
{
mul.f16x2 r7746, r6811, r7744;
}
{
neg.f16x2 r7749, r7746;
}
{
fma.rn.f16x2 r7751, r6775, r7742, r7749;
}
{
mul.f16x2 r7755, r6775, r7744;
}
{
fma.rn.f16x2 r7758, r6811, r7742, r7755;
}
shl.b32 r8588, r8587, 3;
add.s32 r8589, r8578, r8588;
barrier.sync 0;
mad.lo.s32 r8590, r8585, 5832, r8589;
st.shared.u32 [r8590], r6029;
st.shared.u32 [r8590+4], r6035;
st.shared.u32 [r8590+216], r6826;
st.shared.u32 [r8590+220], r6833;
st.shared.u32 [r8590+432], r6863;
st.shared.u32 [r8590+436], r6870;
st.shared.u32 [r8590+648], r6900;
st.shared.u32 [r8590+652], r6907;
st.shared.u32 [r8590+864], r6937;
st.shared.u32 [r8590+868], r6944;
st.shared.u32 [r8590+1080], r6974;
st.shared.u32 [r8590+1084], r6981;
st.shared.u32 [r8590+1296], r7011;
st.shared.u32 [r8590+1300], r7018;
st.shared.u32 [r8590+1512], r7048;
st.shared.u32 [r8590+1516], r7055;
st.shared.u32 [r8590+1728], r7085;
st.shared.u32 [r8590+1732], r7092;
st.shared.u32 [r8590+1944], r7122;
st.shared.u32 [r8590+1948], r7129;
st.shared.u32 [r8590+2160], r7159;
st.shared.u32 [r8590+2164], r7166;
st.shared.u32 [r8590+2376], r7196;
st.shared.u32 [r8590+2380], r7203;
st.shared.u32 [r8590+2592], r7233;
st.shared.u32 [r8590+2596], r7240;
st.shared.u32 [r8590+2808], r7270;
st.shared.u32 [r8590+2812], r7277;
st.shared.u32 [r8590+3024], r7307;
st.shared.u32 [r8590+3028], r7314;
st.shared.u32 [r8590+3240], r7344;
st.shared.u32 [r8590+3244], r7351;
st.shared.u32 [r8590+3456], r7381;
st.shared.u32 [r8590+3460], r7388;
st.shared.u32 [r8590+3672], r7418;
st.shared.u32 [r8590+3676], r7425;
st.shared.u32 [r8590+3888], r7455;
st.shared.u32 [r8590+3892], r7462;
st.shared.u32 [r8590+4104], r7492;
st.shared.u32 [r8590+4108], r7499;
st.shared.u32 [r8590+4320], r7529;
st.shared.u32 [r8590+4324], r7536;
st.shared.u32 [r8590+4536], r7566;
st.shared.u32 [r8590+4540], r7573;
st.shared.u32 [r8590+4752], r7603;
st.shared.u32 [r8590+4756], r7610;
st.shared.u32 [r8590+4968], r7640;
st.shared.u32 [r8590+4972], r7647;
st.shared.u32 [r8590+5184], r7677;
st.shared.u32 [r8590+5188], r7684;
st.shared.u32 [r8590+5400], r7714;
st.shared.u32 [r8590+5404], r7721;
st.shared.u32 [r8590+5616], r7751;
st.shared.u32 [r8590+5620], r7758;
barrier.sync 0;
ld.shared.u32 r7787, [r8580];
ld.shared.u32 r7793, [r8580+4];
ld.shared.u32 r7875, [r8580+648];
ld.shared.u32 r7881, [r8580+652];
ld.shared.u32 r7963, [r8580+1296];
ld.shared.u32 r7969, [r8580+1300];
ld.shared.u32 r8051, [r8580+1944];
ld.shared.u32 r8057, [r8580+1948];
ld.shared.u32 r8139, [r8580+2592];
ld.shared.u32 r8145, [r8580+2596];
ld.shared.u32 r8227, [r8580+3240];
ld.shared.u32 r8233, [r8580+3244];
ld.shared.u32 r8315, [r8580+3888];
ld.shared.u32 r8321, [r8580+3892];
ld.shared.u32 r8403, [r8580+4536];
ld.shared.u32 r8409, [r8580+4540];
ld.shared.u32 r8491, [r8580+5184];
ld.shared.u32 r8497, [r8580+5188];
ld.shared.u32 r7784, [r8580+5832];
ld.shared.u32 r7790, [r8580+5836];
ld.shared.u32 r7872, [r8580+6480];
ld.shared.u32 r7878, [r8580+6484];
ld.shared.u32 r7960, [r8580+7128];
ld.shared.u32 r7966, [r8580+7132];
ld.shared.u32 r8048, [r8580+7776];
ld.shared.u32 r8054, [r8580+7780];
ld.shared.u32 r8136, [r8580+8424];
ld.shared.u32 r8142, [r8580+8428];
ld.shared.u32 r8224, [r8580+9072];
ld.shared.u32 r8230, [r8580+9076];
ld.shared.u32 r8312, [r8580+9720];
ld.shared.u32 r8318, [r8580+9724];
ld.shared.u32 r8400, [r8580+10368];
ld.shared.u32 r8406, [r8580+10372];
ld.shared.u32 r8488, [r8580+11016];
ld.shared.u32 r8494, [r8580+11020];
ld.shared.u32 r7785, [r8580+11664];
ld.shared.u32 r7791, [r8580+11668];
ld.shared.u32 r7873, [r8580+12312];
ld.shared.u32 r7879, [r8580+12316];
ld.shared.u32 r7961, [r8580+12960];
ld.shared.u32 r7967, [r8580+12964];
ld.shared.u32 r8049, [r8580+13608];
ld.shared.u32 r8055, [r8580+13612];
ld.shared.u32 r8137, [r8580+14256];
ld.shared.u32 r8143, [r8580+14260];
ld.shared.u32 r8225, [r8580+14904];
ld.shared.u32 r8231, [r8580+14908];
ld.shared.u32 r8313, [r8580+15552];
ld.shared.u32 r8319, [r8580+15556];
ld.shared.u32 r8401, [r8580+16200];
ld.shared.u32 r8407, [r8580+16204];
ld.shared.u32 r8489, [r8580+16848];
ld.shared.u32 r8495, [r8580+16852];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r7779, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r7780, {low, high};
}
{
neg.f16x2 r7781, r7780;
}
{
add.f16x2 r7783, r7784, r7785;
}
{
add.f16x2 %0, r7787, r7783;
}
{
add.f16x2 r7789, r7790, r7791;
}
{
add.f16x2 %1, r7793, r7789;
}
{
add.f16x2 r7795, r7784, r7785;
}
{
mul.f16x2 r7798, r7795, r7779;
}
{
add.f16x2 r7801, r7787, r7798;
}
{
sub.f16x2 r7804, r7790, r7791;
}
{
mul.f16x2 r7807, r7804, r7781;
}
{
add.f16x2 %18, r7801, r7807;
}
{
add.f16x2 r7813, r7784, r7785;
}
{
mul.f16x2 r7816, r7813, r7779;
}
{
add.f16x2 r7819, r7787, r7816;
}
{
sub.f16x2 r7822, r7790, r7791;
}
{
mul.f16x2 r7825, r7822, r7781;
}
{
sub.f16x2 %36, r7819, r7825;
}
{
add.f16x2 r7831, r7790, r7791;
}
{
mul.f16x2 r7834, r7831, r7779;
}
{
add.f16x2 r7837, r7793, r7834;
}
{
sub.f16x2 r7840, r7784, r7785;
}
{
mul.f16x2 r7843, r7840, r7781;
}
{
sub.f16x2 %19, r7837, r7843;
}
{
add.f16x2 r7849, r7790, r7791;
}
{
mul.f16x2 r7852, r7849, r7779;
}
{
add.f16x2 r7855, r7793, r7852;
}
{
sub.f16x2 r7858, r7784, r7785;
}
{
mul.f16x2 r7861, r7858, r7781;
}
{
add.f16x2 %37, r7855, r7861;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r7867, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r7868, {low, high};
}
{
neg.f16x2 r7869, r7868;
}
{
add.f16x2 r7871, r7872, r7873;
}
{
add.f16x2 %2, r7875, r7871;
}
{
add.f16x2 r7877, r7878, r7879;
}
{
add.f16x2 %3, r7881, r7877;
}
{
add.f16x2 r7883, r7872, r7873;
}
{
mul.f16x2 r7886, r7883, r7867;
}
{
add.f16x2 r7889, r7875, r7886;
}
{
sub.f16x2 r7892, r7878, r7879;
}
{
mul.f16x2 r7895, r7892, r7869;
}
{
add.f16x2 %20, r7889, r7895;
}
{
add.f16x2 r7901, r7872, r7873;
}
{
mul.f16x2 r7904, r7901, r7867;
}
{
add.f16x2 r7907, r7875, r7904;
}
{
sub.f16x2 r7910, r7878, r7879;
}
{
mul.f16x2 r7913, r7910, r7869;
}
{
sub.f16x2 %38, r7907, r7913;
}
{
add.f16x2 r7919, r7878, r7879;
}
{
mul.f16x2 r7922, r7919, r7867;
}
{
add.f16x2 r7925, r7881, r7922;
}
{
sub.f16x2 r7928, r7872, r7873;
}
{
mul.f16x2 r7931, r7928, r7869;
}
{
sub.f16x2 %21, r7925, r7931;
}
{
add.f16x2 r7937, r7878, r7879;
}
{
mul.f16x2 r7940, r7937, r7867;
}
{
add.f16x2 r7943, r7881, r7940;
}
{
sub.f16x2 r7946, r7872, r7873;
}
{
mul.f16x2 r7949, r7946, r7869;
}
{
add.f16x2 %39, r7943, r7949;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r7955, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r7956, {low, high};
}
{
neg.f16x2 r7957, r7956;
}
{
add.f16x2 r7959, r7960, r7961;
}
{
add.f16x2 %4, r7963, r7959;
}
{
add.f16x2 r7965, r7966, r7967;
}
{
add.f16x2 %5, r7969, r7965;
}
{
add.f16x2 r7971, r7960, r7961;
}
{
mul.f16x2 r7974, r7971, r7955;
}
{
add.f16x2 r7977, r7963, r7974;
}
{
sub.f16x2 r7980, r7966, r7967;
}
{
mul.f16x2 r7983, r7980, r7957;
}
{
add.f16x2 %22, r7977, r7983;
}
{
add.f16x2 r7989, r7960, r7961;
}
{
mul.f16x2 r7992, r7989, r7955;
}
{
add.f16x2 r7995, r7963, r7992;
}
{
sub.f16x2 r7998, r7966, r7967;
}
{
mul.f16x2 r8001, r7998, r7957;
}
{
sub.f16x2 %40, r7995, r8001;
}
{
add.f16x2 r8007, r7966, r7967;
}
{
mul.f16x2 r8010, r8007, r7955;
}
{
add.f16x2 r8013, r7969, r8010;
}
{
sub.f16x2 r8016, r7960, r7961;
}
{
mul.f16x2 r8019, r8016, r7957;
}
{
sub.f16x2 %23, r8013, r8019;
}
{
add.f16x2 r8025, r7966, r7967;
}
{
mul.f16x2 r8028, r8025, r7955;
}
{
add.f16x2 r8031, r7969, r8028;
}
{
sub.f16x2 r8034, r7960, r7961;
}
{
mul.f16x2 r8037, r8034, r7957;
}
{
add.f16x2 %41, r8031, r8037;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8043, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8044, {low, high};
}
{
neg.f16x2 r8045, r8044;
}
{
add.f16x2 r8047, r8048, r8049;
}
{
add.f16x2 %6, r8051, r8047;
}
{
add.f16x2 r8053, r8054, r8055;
}
{
add.f16x2 %7, r8057, r8053;
}
{
add.f16x2 r8059, r8048, r8049;
}
{
mul.f16x2 r8062, r8059, r8043;
}
{
add.f16x2 r8065, r8051, r8062;
}
{
sub.f16x2 r8068, r8054, r8055;
}
{
mul.f16x2 r8071, r8068, r8045;
}
{
add.f16x2 %24, r8065, r8071;
}
{
add.f16x2 r8077, r8048, r8049;
}
{
mul.f16x2 r8080, r8077, r8043;
}
{
add.f16x2 r8083, r8051, r8080;
}
{
sub.f16x2 r8086, r8054, r8055;
}
{
mul.f16x2 r8089, r8086, r8045;
}
{
sub.f16x2 %42, r8083, r8089;
}
{
add.f16x2 r8095, r8054, r8055;
}
{
mul.f16x2 r8098, r8095, r8043;
}
{
add.f16x2 r8101, r8057, r8098;
}
{
sub.f16x2 r8104, r8048, r8049;
}
{
mul.f16x2 r8107, r8104, r8045;
}
{
sub.f16x2 %25, r8101, r8107;
}
{
add.f16x2 r8113, r8054, r8055;
}
{
mul.f16x2 r8116, r8113, r8043;
}
{
add.f16x2 r8119, r8057, r8116;
}
{
sub.f16x2 r8122, r8048, r8049;
}
{
mul.f16x2 r8125, r8122, r8045;
}
{
add.f16x2 %43, r8119, r8125;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8131, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8132, {low, high};
}
{
neg.f16x2 r8133, r8132;
}
{
add.f16x2 r8135, r8136, r8137;
}
{
add.f16x2 %8, r8139, r8135;
}
{
add.f16x2 r8141, r8142, r8143;
}
{
add.f16x2 %9, r8145, r8141;
}
{
add.f16x2 r8147, r8136, r8137;
}
{
mul.f16x2 r8150, r8147, r8131;
}
{
add.f16x2 r8153, r8139, r8150;
}
{
sub.f16x2 r8156, r8142, r8143;
}
{
mul.f16x2 r8159, r8156, r8133;
}
{
add.f16x2 %26, r8153, r8159;
}
{
add.f16x2 r8165, r8136, r8137;
}
{
mul.f16x2 r8168, r8165, r8131;
}
{
add.f16x2 r8171, r8139, r8168;
}
{
sub.f16x2 r8174, r8142, r8143;
}
{
mul.f16x2 r8177, r8174, r8133;
}
{
sub.f16x2 %44, r8171, r8177;
}
{
add.f16x2 r8183, r8142, r8143;
}
{
mul.f16x2 r8186, r8183, r8131;
}
{
add.f16x2 r8189, r8145, r8186;
}
{
sub.f16x2 r8192, r8136, r8137;
}
{
mul.f16x2 r8195, r8192, r8133;
}
{
sub.f16x2 %27, r8189, r8195;
}
{
add.f16x2 r8201, r8142, r8143;
}
{
mul.f16x2 r8204, r8201, r8131;
}
{
add.f16x2 r8207, r8145, r8204;
}
{
sub.f16x2 r8210, r8136, r8137;
}
{
mul.f16x2 r8213, r8210, r8133;
}
{
add.f16x2 %45, r8207, r8213;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8219, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8220, {low, high};
}
{
neg.f16x2 r8221, r8220;
}
{
add.f16x2 r8223, r8224, r8225;
}
{
add.f16x2 %10, r8227, r8223;
}
{
add.f16x2 r8229, r8230, r8231;
}
{
add.f16x2 %11, r8233, r8229;
}
{
add.f16x2 r8235, r8224, r8225;
}
{
mul.f16x2 r8238, r8235, r8219;
}
{
add.f16x2 r8241, r8227, r8238;
}
{
sub.f16x2 r8244, r8230, r8231;
}
{
mul.f16x2 r8247, r8244, r8221;
}
{
add.f16x2 %28, r8241, r8247;
}
{
add.f16x2 r8253, r8224, r8225;
}
{
mul.f16x2 r8256, r8253, r8219;
}
{
add.f16x2 r8259, r8227, r8256;
}
{
sub.f16x2 r8262, r8230, r8231;
}
{
mul.f16x2 r8265, r8262, r8221;
}
{
sub.f16x2 %46, r8259, r8265;
}
{
add.f16x2 r8271, r8230, r8231;
}
{
mul.f16x2 r8274, r8271, r8219;
}
{
add.f16x2 r8277, r8233, r8274;
}
{
sub.f16x2 r8280, r8224, r8225;
}
{
mul.f16x2 r8283, r8280, r8221;
}
{
sub.f16x2 %29, r8277, r8283;
}
{
add.f16x2 r8289, r8230, r8231;
}
{
mul.f16x2 r8292, r8289, r8219;
}
{
add.f16x2 r8295, r8233, r8292;
}
{
sub.f16x2 r8298, r8224, r8225;
}
{
mul.f16x2 r8301, r8298, r8221;
}
{
add.f16x2 %47, r8295, r8301;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8307, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8308, {low, high};
}
{
neg.f16x2 r8309, r8308;
}
{
add.f16x2 r8311, r8312, r8313;
}
{
add.f16x2 %12, r8315, r8311;
}
{
add.f16x2 r8317, r8318, r8319;
}
{
add.f16x2 %13, r8321, r8317;
}
{
add.f16x2 r8323, r8312, r8313;
}
{
mul.f16x2 r8326, r8323, r8307;
}
{
add.f16x2 r8329, r8315, r8326;
}
{
sub.f16x2 r8332, r8318, r8319;
}
{
mul.f16x2 r8335, r8332, r8309;
}
{
add.f16x2 %30, r8329, r8335;
}
{
add.f16x2 r8341, r8312, r8313;
}
{
mul.f16x2 r8344, r8341, r8307;
}
{
add.f16x2 r8347, r8315, r8344;
}
{
sub.f16x2 r8350, r8318, r8319;
}
{
mul.f16x2 r8353, r8350, r8309;
}
{
sub.f16x2 %48, r8347, r8353;
}
{
add.f16x2 r8359, r8318, r8319;
}
{
mul.f16x2 r8362, r8359, r8307;
}
{
add.f16x2 r8365, r8321, r8362;
}
{
sub.f16x2 r8368, r8312, r8313;
}
{
mul.f16x2 r8371, r8368, r8309;
}
{
sub.f16x2 %31, r8365, r8371;
}
{
add.f16x2 r8377, r8318, r8319;
}
{
mul.f16x2 r8380, r8377, r8307;
}
{
add.f16x2 r8383, r8321, r8380;
}
{
sub.f16x2 r8386, r8312, r8313;
}
{
mul.f16x2 r8389, r8386, r8309;
}
{
add.f16x2 %49, r8383, r8389;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8396, {low, high};
}
{
neg.f16x2 r8397, r8396;
}
{
add.f16x2 r8399, r8400, r8401;
}
{
add.f16x2 %14, r8403, r8399;
}
{
add.f16x2 r8405, r8406, r8407;
}
{
add.f16x2 %15, r8409, r8405;
}
{
add.f16x2 r8411, r8400, r8401;
}
{
mul.f16x2 r8414, r8411, r8395;
}
{
add.f16x2 r8417, r8403, r8414;
}
{
sub.f16x2 r8420, r8406, r8407;
}
{
mul.f16x2 r8423, r8420, r8397;
}
{
add.f16x2 %32, r8417, r8423;
}
{
add.f16x2 r8429, r8400, r8401;
}
{
mul.f16x2 r8432, r8429, r8395;
}
{
add.f16x2 r8435, r8403, r8432;
}
{
sub.f16x2 r8438, r8406, r8407;
}
{
mul.f16x2 r8441, r8438, r8397;
}
{
sub.f16x2 %50, r8435, r8441;
}
{
add.f16x2 r8447, r8406, r8407;
}
{
mul.f16x2 r8450, r8447, r8395;
}
{
add.f16x2 r8453, r8409, r8450;
}
{
sub.f16x2 r8456, r8400, r8401;
}
{
mul.f16x2 r8459, r8456, r8397;
}
{
sub.f16x2 %33, r8453, r8459;
}
{
add.f16x2 r8465, r8406, r8407;
}
{
mul.f16x2 r8468, r8465, r8395;
}
{
add.f16x2 r8471, r8409, r8468;
}
{
sub.f16x2 r8474, r8400, r8401;
}
{
mul.f16x2 r8477, r8474, r8397;
}
{
add.f16x2 %51, r8471, r8477;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8483, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8484, {low, high};
}
{
neg.f16x2 r8485, r8484;
}
{
add.f16x2 r8487, r8488, r8489;
}
{
add.f16x2 %16, r8491, r8487;
}
{
add.f16x2 r8493, r8494, r8495;
}
{
add.f16x2 %17, r8497, r8493;
}
{
add.f16x2 r8499, r8488, r8489;
}
{
mul.f16x2 r8502, r8499, r8483;
}
{
add.f16x2 r8505, r8491, r8502;
}
{
sub.f16x2 r8508, r8494, r8495;
}
{
mul.f16x2 r8511, r8508, r8485;
}
{
add.f16x2 %34, r8505, r8511;
}
{
add.f16x2 r8517, r8488, r8489;
}
{
mul.f16x2 r8520, r8517, r8483;
}
{
add.f16x2 r8523, r8491, r8520;
}
{
sub.f16x2 r8526, r8494, r8495;
}
{
mul.f16x2 r8529, r8526, r8485;
}
{
sub.f16x2 %52, r8523, r8529;
}
{
add.f16x2 r8535, r8494, r8495;
}
{
mul.f16x2 r8538, r8535, r8483;
}
{
add.f16x2 r8541, r8497, r8538;
}
{
sub.f16x2 r8544, r8488, r8489;
}
{
mul.f16x2 r8547, r8544, r8485;
}
{
sub.f16x2 %35, r8541, r8547;
}
{
add.f16x2 r8553, r8494, r8495;
}
{
mul.f16x2 r8556, r8553, r8483;
}
{
add.f16x2 r8559, r8497, r8556;
}
{
sub.f16x2 r8562, r8488, r8489;
}
{
mul.f16x2 r8565, r8562, r8485;
}
{
add.f16x2 %53, r8559, r8565;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[4].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<890, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<250>;
.reg .b32 r<3009>;
.reg .b64 rd<8>;
mov.u32 r2986, %tid.y;
mov.u32 r2987, %18;
mad.lo.s32 r2988, r2986, 17496, r2987;
mov.u32 r2989, %tid.x;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r177, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
mov.f32 f166, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f166;
cvt.rn.f16.f32 high, f166;
mov.b32 r265, {low, high};
}
mov.f32 f168, 0fBF248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f168;
cvt.rn.f16.f32 high, f168;
mov.b32 r266, {low, high};
}
mov.f32 f170, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f170;
cvt.rn.f16.f32 high, f170;
mov.b32 r267, {low, high};
}
mov.f32 f172, 0fBF7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f172;
cvt.rn.f16.f32 high, f172;
mov.b32 r268, {low, high};
}
mov.f32 f178, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r271, {low, high};
}
mov.f32 f180, 0fBEAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r345, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r433, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r521, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
mul.wide.u32 rd2, r2989, -2032597691;
shr.u64 rd3, rd2, 39;
cvt.u32.u64 r2990, rd3;
mul.lo.s32 r2991, r2990, 243;
sub.s32 r2992, r2989, r2991;
cvt.rn.f32.u32 f241, r2992;
mul.f32 f242, f241, 0f3B3C4870;
cos.approx.f32 f57, f242;
sin.approx.f32 f243, f242;
neg.f32 f58, f243;
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
mov.f32 f225, 0fBF800000;
mov.f32 f226, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
mad.lo.s32 r2993, r2990, 17496, r2988;
barrier.sync 0;
mad.lo.s32 r2994, r2992, 72, r2993;
st.shared.v2.f32 [r2994], {r352, r358};
st.shared.v2.f32 [r2994+8], {r621, r628};
st.shared.v2.f32 [r2994+16], {r658, r665};
st.shared.v2.f32 [r2994+24], {r695, r702};
st.shared.v2.f32 [r2994+32], {r732, r739};
st.shared.v2.f32 [r2994+40], {r769, r776};
st.shared.v2.f32 [r2994+48], {r806, r813};
st.shared.v2.f32 [r2994+56], {r843, r850};
st.shared.v2.f32 [r2994+64], {r880, r887};
barrier.sync 0;
shl.b32 r2995, r2992, 6;
sub.s32 r2996, r2994, r2995;
ld.shared.u32 r916, [r2996];
ld.shared.u32 r922, [r2996+4];
ld.shared.u32 r1004, [r2996+1944];
ld.shared.u32 r1010, [r2996+1948];
ld.shared.u32 r1092, [r2996+3888];
ld.shared.u32 r1098, [r2996+3892];
ld.shared.u32 r913, [r2996+5832];
ld.shared.u32 r919, [r2996+5836];
ld.shared.u32 r1001, [r2996+7776];
ld.shared.u32 r1007, [r2996+7780];
ld.shared.u32 r1089, [r2996+9720];
ld.shared.u32 r1095, [r2996+9724];
ld.shared.u32 r914, [r2996+11664];
ld.shared.u32 r920, [r2996+11668];
ld.shared.u32 r1002, [r2996+13608];
ld.shared.u32 r1008, [r2996+13612];
ld.shared.u32 r1090, [r2996+15552];
ld.shared.u32 r1096, [r2996+15556];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r908, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r996, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1084, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f166;
cvt.rn.f16.f32 high, f166;
mov.b32 r1172, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f168;
cvt.rn.f16.f32 high, f168;
mov.b32 r1173, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f170;
cvt.rn.f16.f32 high, f170;
mov.b32 r1174, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f172;
cvt.rn.f16.f32 high, f172;
mov.b32 r1175, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r1178, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1252, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1253, {low, high};
}
{
neg.f16x2 r1254, r1253;
}
{
add.f16x2 r1256, r1003, r1091;
}
{
add.f16x2 r1259, r915, r1256;
}
{
add.f16x2 r1262, r1009, r1097;
}
{
add.f16x2 r1265, r921, r1262;
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
add.f16x2 r1283, r1274, r1280;
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
sub.f16x2 r1301, r1292, r1298;
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
sub.f16x2 r1319, r1310, r1316;
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
add.f16x2 r1337, r1328, r1334;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1340, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1341, {low, high};
}
{
neg.f16x2 r1342, r1341;
}
{
add.f16x2 r1344, r1194, r1210;
}
{
add.f16x2 r1347, r939, r1344;
}
{
add.f16x2 r1350, r1200, r1216;
}
{
add.f16x2 r1353, r975, r1350;
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
add.f16x2 r1371, r1362, r1368;
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
sub.f16x2 r1389, r1380, r1386;
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
sub.f16x2 r1407, r1398, r1404;
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
add.f16x2 r1425, r1416, r1422;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1428, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1429, {low, high};
}
{
neg.f16x2 r1430, r1429;
}
{
add.f16x2 r1432, r1226, r1242;
}
{
add.f16x2 r1435, r957, r1432;
}
{
add.f16x2 r1438, r1232, r1248;
}
{
add.f16x2 r1441, r993, r1438;
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
add.f16x2 r1459, r1450, r1456;
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
sub.f16x2 r1477, r1468, r1474;
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
sub.f16x2 r1495, r1486, r1492;
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
add.f16x2 r1513, r1504, r1510;
}
mul.wide.u32 rd4, r2992, 954437177;
shr.u64 rd5, rd4, 33;
cvt.u32.u64 r2997, rd5;
cvt.rn.f32.u32 f244, r2997;
mul.f32 f245, f244, 0f3CD3D17E;
cos.approx.f32 f133, f245;
sin.approx.f32 f246, f245;
neg.f32 f134, f246;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f133;
cvt.rn.f16.f32 high, f134;
mov.b32 r1516, {low, high};
}
mul.lo.s32 r2998, r2997, 9;
sub.s32 r2999, r2992, r2998;
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1519, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1521, {high, high};
}
{
mul.f16x2 r1523, r1353, r1521;
}
{
neg.f16x2 r1526, r1523;
}
{
fma.rn.f16x2 r1528, r1347, r1519, r1526;
}
{
mul.f16x2 r1532, r1347, r1521;
}
{
fma.rn.f16x2 r1535, r1353, r1519, r1532;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1539, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1541, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1543, {low, high};
}
{
mul.f16x2 r1544, r1541, r1543;
}
{
mul.f16x2 r1547, r1516, r1539;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1550, {high, low};
}
{
fma.rn.f16x2 r1552, r1544, r1550, r1547;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1552;
mov.b32 r1556, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1552;
mov.b32 r1558, {high, high};
}
{
mul.f16x2 r1560, r1441, r1558;
}
{
neg.f16x2 r1563, r1560;
}
{
fma.rn.f16x2 r1565, r1435, r1556, r1563;
}
{
mul.f16x2 r1569, r1435, r1558;
}
{
fma.rn.f16x2 r1572, r1441, r1556, r1569;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1576, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1578, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1580, {low, high};
}
{
mul.f16x2 r1581, r1578, r1580;
}
{
mul.f16x2 r1584, r1552, r1576;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1552;
mov.b32 r1587, {high, low};
}
{
fma.rn.f16x2 r1589, r1581, r1587, r1584;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1589;
mov.b32 r1593, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1589;
mov.b32 r1595, {high, high};
}
{
mul.f16x2 r1597, r1319, r1595;
}
{
neg.f16x2 r1600, r1597;
}
{
fma.rn.f16x2 r1602, r1283, r1593, r1600;
}
{
mul.f16x2 r1606, r1283, r1595;
}
{
fma.rn.f16x2 r1609, r1319, r1593, r1606;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1613, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1615, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1617, {low, high};
}
{
mul.f16x2 r1618, r1615, r1617;
}
{
mul.f16x2 r1621, r1589, r1613;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1589;
mov.b32 r1624, {high, low};
}
{
fma.rn.f16x2 r1626, r1618, r1624, r1621;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1630, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1632, {high, high};
}
{
mul.f16x2 r1634, r1407, r1632;
}
{
neg.f16x2 r1637, r1634;
}
{
fma.rn.f16x2 r1639, r1371, r1630, r1637;
}
{
mul.f16x2 r1643, r1371, r1632;
}
{
fma.rn.f16x2 r1646, r1407, r1630, r1643;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1650, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1652, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1654, {low, high};
}
{
mul.f16x2 r1655, r1652, r1654;
}
{
mul.f16x2 r1658, r1626, r1650;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1661, {high, low};
}
{
fma.rn.f16x2 r1663, r1655, r1661, r1658;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1663;
mov.b32 r1667, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1663;
mov.b32 r1669, {high, high};
}
{
mul.f16x2 r1671, r1495, r1669;
}
{
neg.f16x2 r1674, r1671;
}
{
fma.rn.f16x2 r1676, r1459, r1667, r1674;
}
{
mul.f16x2 r1680, r1459, r1669;
}
{
fma.rn.f16x2 r1683, r1495, r1667, r1680;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1687, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1689, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1691, {low, high};
}
{
mul.f16x2 r1692, r1689, r1691;
}
{
mul.f16x2 r1695, r1663, r1687;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1663;
mov.b32 r1698, {high, low};
}
{
fma.rn.f16x2 r1700, r1692, r1698, r1695;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1700;
mov.b32 r1704, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1700;
mov.b32 r1706, {high, high};
}
{
mul.f16x2 r1708, r1337, r1706;
}
{
neg.f16x2 r1711, r1708;
}
{
fma.rn.f16x2 r1713, r1301, r1704, r1711;
}
{
mul.f16x2 r1717, r1301, r1706;
}
{
fma.rn.f16x2 r1720, r1337, r1704, r1717;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1724, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1726, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1728, {low, high};
}
{
mul.f16x2 r1729, r1726, r1728;
}
{
mul.f16x2 r1732, r1700, r1724;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1700;
mov.b32 r1735, {high, low};
}
{
fma.rn.f16x2 r1737, r1729, r1735, r1732;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1737;
mov.b32 r1741, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1737;
mov.b32 r1743, {high, high};
}
{
mul.f16x2 r1745, r1425, r1743;
}
{
neg.f16x2 r1748, r1745;
}
{
fma.rn.f16x2 r1750, r1389, r1741, r1748;
}
{
mul.f16x2 r1754, r1389, r1743;
}
{
fma.rn.f16x2 r1757, r1425, r1741, r1754;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1761, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1763, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1765, {low, high};
}
{
mul.f16x2 r1766, r1763, r1765;
}
{
mul.f16x2 r1769, r1737, r1761;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1737;
mov.b32 r1772, {high, low};
}
{
fma.rn.f16x2 r1774, r1766, r1772, r1769;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1774;
mov.b32 r1778, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1774;
mov.b32 r1780, {high, high};
}
{
mul.f16x2 r1782, r1513, r1780;
}
{
neg.f16x2 r1785, r1782;
}
{
fma.rn.f16x2 r1787, r1477, r1778, r1785;
}
{
mul.f16x2 r1791, r1477, r1780;
}
{
fma.rn.f16x2 r1794, r1513, r1778, r1791;
}
shl.b32 r3000, r2999, 3;
add.s32 r3001, r2993, r3000;
barrier.sync 0;
mad.lo.s32 r3002, r2997, 648, r3001;
st.shared.u32 [r3002], r1259;
st.shared.u32 [r3002+4], r1265;
st.shared.u32 [r3002+72], r1528;
st.shared.u32 [r3002+76], r1535;
st.shared.u32 [r3002+144], r1565;
st.shared.u32 [r3002+148], r1572;
st.shared.u32 [r3002+216], r1602;
st.shared.u32 [r3002+220], r1609;
st.shared.u32 [r3002+288], r1639;
st.shared.u32 [r3002+292], r1646;
st.shared.u32 [r3002+360], r1676;
st.shared.u32 [r3002+364], r1683;
st.shared.u32 [r3002+432], r1713;
st.shared.u32 [r3002+436], r1720;
st.shared.u32 [r3002+504], r1750;
st.shared.u32 [r3002+508], r1757;
st.shared.u32 [r3002+576], r1787;
st.shared.u32 [r3002+580], r1794;
barrier.sync 0;
ld.shared.u32 r1823, [r2996];
ld.shared.u32 r1829, [r2996+4];
ld.shared.u32 r1911, [r2996+1944];
ld.shared.u32 r1917, [r2996+1948];
ld.shared.u32 r1999, [r2996+3888];
ld.shared.u32 r2005, [r2996+3892];
ld.shared.u32 r1820, [r2996+5832];
ld.shared.u32 r1826, [r2996+5836];
ld.shared.u32 r1908, [r2996+7776];
ld.shared.u32 r1914, [r2996+7780];
ld.shared.u32 r1996, [r2996+9720];
ld.shared.u32 r2002, [r2996+9724];
ld.shared.u32 r1821, [r2996+11664];
ld.shared.u32 r1827, [r2996+11668];
ld.shared.u32 r1909, [r2996+13608];
ld.shared.u32 r1915, [r2996+13612];
ld.shared.u32 r1997, [r2996+15552];
ld.shared.u32 r2003, [r2996+15556];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1815, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1816, {low, high};
}
{
neg.f16x2 r1817, r1816;
}
{
add.f16x2 r1819, r1820, r1821;
}
{
add.f16x2 r1822, r1823, r1819;
}
{
add.f16x2 r1825, r1826, r1827;
}
{
add.f16x2 r1828, r1829, r1825;
}
{
add.f16x2 r1831, r1820, r1821;
}
{
mul.f16x2 r1834, r1831, r1815;
}
{
add.f16x2 r1837, r1823, r1834;
}
{
sub.f16x2 r1840, r1826, r1827;
}
{
mul.f16x2 r1843, r1840, r1817;
}
{
add.f16x2 r1846, r1837, r1843;
}
{
add.f16x2 r1849, r1820, r1821;
}
{
mul.f16x2 r1852, r1849, r1815;
}
{
add.f16x2 r1855, r1823, r1852;
}
{
sub.f16x2 r1858, r1826, r1827;
}
{
mul.f16x2 r1861, r1858, r1817;
}
{
sub.f16x2 r1864, r1855, r1861;
}
{
add.f16x2 r1867, r1826, r1827;
}
{
mul.f16x2 r1870, r1867, r1815;
}
{
add.f16x2 r1873, r1829, r1870;
}
{
sub.f16x2 r1876, r1820, r1821;
}
{
mul.f16x2 r1879, r1876, r1817;
}
{
sub.f16x2 r1882, r1873, r1879;
}
{
add.f16x2 r1885, r1826, r1827;
}
{
mul.f16x2 r1888, r1885, r1815;
}
{
add.f16x2 r1891, r1829, r1888;
}
{
sub.f16x2 r1894, r1820, r1821;
}
{
mul.f16x2 r1897, r1894, r1817;
}
{
add.f16x2 r1900, r1891, r1897;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1903, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1904, {low, high};
}
{
neg.f16x2 r1905, r1904;
}
{
add.f16x2 r1907, r1908, r1909;
}
{
add.f16x2 r1910, r1911, r1907;
}
{
add.f16x2 r1913, r1914, r1915;
}
{
add.f16x2 r1916, r1917, r1913;
}
{
add.f16x2 r1919, r1908, r1909;
}
{
mul.f16x2 r1922, r1919, r1903;
}
{
add.f16x2 r1925, r1911, r1922;
}
{
sub.f16x2 r1928, r1914, r1915;
}
{
mul.f16x2 r1931, r1928, r1905;
}
{
add.f16x2 r1934, r1925, r1931;
}
{
add.f16x2 r1937, r1908, r1909;
}
{
mul.f16x2 r1940, r1937, r1903;
}
{
add.f16x2 r1943, r1911, r1940;
}
{
sub.f16x2 r1946, r1914, r1915;
}
{
mul.f16x2 r1949, r1946, r1905;
}
{
sub.f16x2 r1952, r1943, r1949;
}
{
add.f16x2 r1955, r1914, r1915;
}
{
mul.f16x2 r1958, r1955, r1903;
}
{
add.f16x2 r1961, r1917, r1958;
}
{
sub.f16x2 r1964, r1908, r1909;
}
{
mul.f16x2 r1967, r1964, r1905;
}
{
sub.f16x2 r1970, r1961, r1967;
}
{
add.f16x2 r1973, r1914, r1915;
}
{
mul.f16x2 r1976, r1973, r1903;
}
{
add.f16x2 r1979, r1917, r1976;
}
{
sub.f16x2 r1982, r1908, r1909;
}
{
mul.f16x2 r1985, r1982, r1905;
}
{
add.f16x2 r1988, r1979, r1985;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1991, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1992, {low, high};
}
{
neg.f16x2 r1993, r1992;
}
{
add.f16x2 r1995, r1996, r1997;
}
{
add.f16x2 r1998, r1999, r1995;
}
{
add.f16x2 r2001, r2002, r2003;
}
{
add.f16x2 r2004, r2005, r2001;
}
{
add.f16x2 r2007, r1996, r1997;
}
{
mul.f16x2 r2010, r2007, r1991;
}
{
add.f16x2 r2013, r1999, r2010;
}
{
sub.f16x2 r2016, r2002, r2003;
}
{
mul.f16x2 r2019, r2016, r1993;
}
{
add.f16x2 r2022, r2013, r2019;
}
{
add.f16x2 r2025, r1996, r1997;
}
{
mul.f16x2 r2028, r2025, r1991;
}
{
add.f16x2 r2031, r1999, r2028;
}
{
sub.f16x2 r2034, r2002, r2003;
}
{
mul.f16x2 r2037, r2034, r1993;
}
{
sub.f16x2 r2040, r2031, r2037;
}
{
add.f16x2 r2043, r2002, r2003;
}
{
mul.f16x2 r2046, r2043, r1991;
}
{
add.f16x2 r2049, r2005, r2046;
}
{
sub.f16x2 r2052, r1996, r1997;
}
{
mul.f16x2 r2055, r2052, r1993;
}
{
sub.f16x2 r2058, r2049, r2055;
}
{
add.f16x2 r2061, r2002, r2003;
}
{
mul.f16x2 r2064, r2061, r1991;
}
{
add.f16x2 r2067, r2005, r2064;
}
{
sub.f16x2 r2070, r1996, r1997;
}
{
mul.f16x2 r2073, r2070, r1993;
}
{
add.f16x2 r2076, r2067, r2073;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f166;
cvt.rn.f16.f32 high, f166;
mov.b32 r2079, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f168;
cvt.rn.f16.f32 high, f168;
mov.b32 r2080, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f170;
cvt.rn.f16.f32 high, f170;
mov.b32 r2081, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f172;
cvt.rn.f16.f32 high, f172;
mov.b32 r2082, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r2085, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
mov.b32 r2086, {low, high};
}
{
mul.f16x2 r2095, r1934, r2079;
}
{
mul.f16x2 r2098, r1970, r2080;
}
{
sub.f16x2 r2101, r2095, r2098;
}
{
mul.f16x2 r2104, r1934, r2080;
}
{
fma.rn.f16x2 r2107, r1970, r2079, r2104;
}
{
mul.f16x2 r2111, r2022, r2081;
}
{
mul.f16x2 r2114, r2058, r2082;
}
{
sub.f16x2 r2117, r2111, r2114;
}
{
mul.f16x2 r2120, r2022, r2082;
}
{
fma.rn.f16x2 r2123, r2058, r2081, r2120;
}
{
mul.f16x2 r2127, r1952, r2081;
}
{
mul.f16x2 r2130, r1988, r2082;
}
{
sub.f16x2 r2133, r2127, r2130;
}
{
mul.f16x2 r2136, r1952, r2082;
}
{
fma.rn.f16x2 r2139, r1988, r2081, r2136;
}
{
mul.f16x2 r2143, r2040, r2085;
}
{
mul.f16x2 r2146, r2076, r2086;
}
{
sub.f16x2 r2149, r2143, r2146;
}
{
mul.f16x2 r2152, r2040, r2086;
}
{
fma.rn.f16x2 r2155, r2076, r2085, r2152;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2159, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2160, {low, high};
}
{
neg.f16x2 r2161, r2160;
}
{
add.f16x2 r2163, r1910, r1998;
}
{
add.f16x2 r2166, r1822, r2163;
}
{
add.f16x2 r2169, r1916, r2004;
}
{
add.f16x2 r2172, r1828, r2169;
}
{
add.f16x2 r2175, r1910, r1998;
}
{
mul.f16x2 r2178, r2175, r2159;
}
{
add.f16x2 r2181, r1822, r2178;
}
{
sub.f16x2 r2184, r1916, r2004;
}
{
mul.f16x2 r2187, r2184, r2161;
}
{
add.f16x2 r2190, r2181, r2187;
}
{
add.f16x2 r2193, r1910, r1998;
}
{
mul.f16x2 r2196, r2193, r2159;
}
{
add.f16x2 r2199, r1822, r2196;
}
{
sub.f16x2 r2202, r1916, r2004;
}
{
mul.f16x2 r2205, r2202, r2161;
}
{
sub.f16x2 r2208, r2199, r2205;
}
{
add.f16x2 r2211, r1916, r2004;
}
{
mul.f16x2 r2214, r2211, r2159;
}
{
add.f16x2 r2217, r1828, r2214;
}
{
sub.f16x2 r2220, r1910, r1998;
}
{
mul.f16x2 r2223, r2220, r2161;
}
{
sub.f16x2 r2226, r2217, r2223;
}
{
add.f16x2 r2229, r1916, r2004;
}
{
mul.f16x2 r2232, r2229, r2159;
}
{
add.f16x2 r2235, r1828, r2232;
}
{
sub.f16x2 r2238, r1910, r1998;
}
{
mul.f16x2 r2241, r2238, r2161;
}
{
add.f16x2 r2244, r2235, r2241;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2247, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2248, {low, high};
}
{
neg.f16x2 r2249, r2248;
}
{
add.f16x2 r2251, r2101, r2117;
}
{
add.f16x2 r2254, r1846, r2251;
}
{
add.f16x2 r2257, r2107, r2123;
}
{
add.f16x2 r2260, r1882, r2257;
}
{
add.f16x2 r2263, r2101, r2117;
}
{
mul.f16x2 r2266, r2263, r2247;
}
{
add.f16x2 r2269, r1846, r2266;
}
{
sub.f16x2 r2272, r2107, r2123;
}
{
mul.f16x2 r2275, r2272, r2249;
}
{
add.f16x2 r2278, r2269, r2275;
}
{
add.f16x2 r2281, r2101, r2117;
}
{
mul.f16x2 r2284, r2281, r2247;
}
{
add.f16x2 r2287, r1846, r2284;
}
{
sub.f16x2 r2290, r2107, r2123;
}
{
mul.f16x2 r2293, r2290, r2249;
}
{
sub.f16x2 r2296, r2287, r2293;
}
{
add.f16x2 r2299, r2107, r2123;
}
{
mul.f16x2 r2302, r2299, r2247;
}
{
add.f16x2 r2305, r1882, r2302;
}
{
sub.f16x2 r2308, r2101, r2117;
}
{
mul.f16x2 r2311, r2308, r2249;
}
{
sub.f16x2 r2314, r2305, r2311;
}
{
add.f16x2 r2317, r2107, r2123;
}
{
mul.f16x2 r2320, r2317, r2247;
}
{
add.f16x2 r2323, r1882, r2320;
}
{
sub.f16x2 r2326, r2101, r2117;
}
{
mul.f16x2 r2329, r2326, r2249;
}
{
add.f16x2 r2332, r2323, r2329;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2335, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2336, {low, high};
}
{
neg.f16x2 r2337, r2336;
}
{
add.f16x2 r2339, r2133, r2149;
}
{
add.f16x2 r2342, r1864, r2339;
}
{
add.f16x2 r2345, r2139, r2155;
}
{
add.f16x2 r2348, r1900, r2345;
}
{
add.f16x2 r2351, r2133, r2149;
}
{
mul.f16x2 r2354, r2351, r2335;
}
{
add.f16x2 r2357, r1864, r2354;
}
{
sub.f16x2 r2360, r2139, r2155;
}
{
mul.f16x2 r2363, r2360, r2337;
}
{
add.f16x2 r2366, r2357, r2363;
}
{
add.f16x2 r2369, r2133, r2149;
}
{
mul.f16x2 r2372, r2369, r2335;
}
{
add.f16x2 r2375, r1864, r2372;
}
{
sub.f16x2 r2378, r2139, r2155;
}
{
mul.f16x2 r2381, r2378, r2337;
}
{
sub.f16x2 r2384, r2375, r2381;
}
{
add.f16x2 r2387, r2139, r2155;
}
{
mul.f16x2 r2390, r2387, r2335;
}
{
add.f16x2 r2393, r1900, r2390;
}
{
sub.f16x2 r2396, r2133, r2149;
}
{
mul.f16x2 r2399, r2396, r2337;
}
{
sub.f16x2 r2402, r2393, r2399;
}
{
add.f16x2 r2405, r2139, r2155;
}
{
mul.f16x2 r2408, r2405, r2335;
}
{
add.f16x2 r2411, r1900, r2408;
}
{
sub.f16x2 r2414, r2133, r2149;
}
{
mul.f16x2 r2417, r2414, r2337;
}
{
add.f16x2 r2420, r2411, r2417;
}
mul.wide.u32 rd6, r2992, -901412889;
shr.u64 rd7, rd6, 38;
cvt.u32.u64 r3003, rd7;
cvt.rn.f32.u32 f247, r3003;
mul.f32 f248, f247, 0f3E6E4BAE;
cos.approx.f32 f209, f248;
sin.approx.f32 f249, f248;
neg.f32 f210, f249;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f209;
cvt.rn.f16.f32 high, f210;
mov.b32 r2423, {low, high};
}
mul.lo.s32 r3004, r3003, 81;
sub.s32 r3005, r2992, r3004;
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2426, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2428, {high, high};
}
{
mul.f16x2 r2430, r2260, r2428;
}
{
neg.f16x2 r2433, r2430;
}
{
fma.rn.f16x2 r2435, r2254, r2426, r2433;
}
{
mul.f16x2 r2439, r2254, r2428;
}
{
fma.rn.f16x2 r2442, r2260, r2426, r2439;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2446, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2448, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2450, {low, high};
}
{
mul.f16x2 r2451, r2448, r2450;
}
{
mul.f16x2 r2454, r2423, r2446;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2457, {high, low};
}
{
fma.rn.f16x2 r2459, r2451, r2457, r2454;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2459;
mov.b32 r2463, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2459;
mov.b32 r2465, {high, high};
}
{
mul.f16x2 r2467, r2348, r2465;
}
{
neg.f16x2 r2470, r2467;
}
{
fma.rn.f16x2 r2472, r2342, r2463, r2470;
}
{
mul.f16x2 r2476, r2342, r2465;
}
{
fma.rn.f16x2 r2479, r2348, r2463, r2476;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2483, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2485, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2487, {low, high};
}
{
mul.f16x2 r2488, r2485, r2487;
}
{
mul.f16x2 r2491, r2459, r2483;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2459;
mov.b32 r2494, {high, low};
}
{
fma.rn.f16x2 r2496, r2488, r2494, r2491;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2496;
mov.b32 r2500, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2496;
mov.b32 r2502, {high, high};
}
{
mul.f16x2 r2504, r2226, r2502;
}
{
neg.f16x2 r2507, r2504;
}
{
fma.rn.f16x2 r2509, r2190, r2500, r2507;
}
{
mul.f16x2 r2513, r2190, r2502;
}
{
fma.rn.f16x2 r2516, r2226, r2500, r2513;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2520, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2522, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2524, {low, high};
}
{
mul.f16x2 r2525, r2522, r2524;
}
{
mul.f16x2 r2528, r2496, r2520;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2496;
mov.b32 r2531, {high, low};
}
{
fma.rn.f16x2 r2533, r2525, r2531, r2528;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2533;
mov.b32 r2537, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2533;
mov.b32 r2539, {high, high};
}
{
mul.f16x2 r2541, r2314, r2539;
}
{
neg.f16x2 r2544, r2541;
}
{
fma.rn.f16x2 r2546, r2278, r2537, r2544;
}
{
mul.f16x2 r2550, r2278, r2539;
}
{
fma.rn.f16x2 r2553, r2314, r2537, r2550;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2557, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2559, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2561, {low, high};
}
{
mul.f16x2 r2562, r2559, r2561;
}
{
mul.f16x2 r2565, r2533, r2557;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2533;
mov.b32 r2568, {high, low};
}
{
fma.rn.f16x2 r2570, r2562, r2568, r2565;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2570;
mov.b32 r2574, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2570;
mov.b32 r2576, {high, high};
}
{
mul.f16x2 r2578, r2402, r2576;
}
{
neg.f16x2 r2581, r2578;
}
{
fma.rn.f16x2 r2583, r2366, r2574, r2581;
}
{
mul.f16x2 r2587, r2366, r2576;
}
{
fma.rn.f16x2 r2590, r2402, r2574, r2587;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2594, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2596, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2598, {low, high};
}
{
mul.f16x2 r2599, r2596, r2598;
}
{
mul.f16x2 r2602, r2570, r2594;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2570;
mov.b32 r2605, {high, low};
}
{
fma.rn.f16x2 r2607, r2599, r2605, r2602;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2607;
mov.b32 r2611, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2607;
mov.b32 r2613, {high, high};
}
{
mul.f16x2 r2615, r2244, r2613;
}
{
neg.f16x2 r2618, r2615;
}
{
fma.rn.f16x2 r2620, r2208, r2611, r2618;
}
{
mul.f16x2 r2624, r2208, r2613;
}
{
fma.rn.f16x2 r2627, r2244, r2611, r2624;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2631, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2633, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2635, {low, high};
}
{
mul.f16x2 r2636, r2633, r2635;
}
{
mul.f16x2 r2639, r2607, r2631;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2607;
mov.b32 r2642, {high, low};
}
{
fma.rn.f16x2 r2644, r2636, r2642, r2639;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2644;
mov.b32 r2648, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2644;
mov.b32 r2650, {high, high};
}
{
mul.f16x2 r2652, r2332, r2650;
}
{
neg.f16x2 r2655, r2652;
}
{
fma.rn.f16x2 r2657, r2296, r2648, r2655;
}
{
mul.f16x2 r2661, r2296, r2650;
}
{
fma.rn.f16x2 r2664, r2332, r2648, r2661;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2668, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2670, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2672, {low, high};
}
{
mul.f16x2 r2673, r2670, r2672;
}
{
mul.f16x2 r2676, r2644, r2668;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2644;
mov.b32 r2679, {high, low};
}
{
fma.rn.f16x2 r2681, r2673, r2679, r2676;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2681;
mov.b32 r2685, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2681;
mov.b32 r2687, {high, high};
}
{
mul.f16x2 r2689, r2420, r2687;
}
{
neg.f16x2 r2692, r2689;
}
{
fma.rn.f16x2 r2694, r2384, r2685, r2692;
}
{
mul.f16x2 r2698, r2384, r2687;
}
{
fma.rn.f16x2 r2701, r2420, r2685, r2698;
}
shl.b32 r3006, r3005, 3;
add.s32 r3007, r2993, r3006;
barrier.sync 0;
mad.lo.s32 r3008, r3003, 5832, r3007;
st.shared.u32 [r3008], r2166;
st.shared.u32 [r3008+4], r2172;
st.shared.u32 [r3008+648], r2435;
st.shared.u32 [r3008+652], r2442;
st.shared.u32 [r3008+1296], r2472;
st.shared.u32 [r3008+1300], r2479;
st.shared.u32 [r3008+1944], r2509;
st.shared.u32 [r3008+1948], r2516;
st.shared.u32 [r3008+2592], r2546;
st.shared.u32 [r3008+2596], r2553;
st.shared.u32 [r3008+3240], r2583;
st.shared.u32 [r3008+3244], r2590;
st.shared.u32 [r3008+3888], r2620;
st.shared.u32 [r3008+3892], r2627;
st.shared.u32 [r3008+4536], r2657;
st.shared.u32 [r3008+4540], r2664;
st.shared.u32 [r3008+5184], r2694;
st.shared.u32 [r3008+5188], r2701;
barrier.sync 0;
ld.shared.u32 r2730, [r2996];
ld.shared.u32 r2736, [r2996+4];
ld.shared.u32 r2818, [r2996+1944];
ld.shared.u32 r2824, [r2996+1948];
ld.shared.u32 r2906, [r2996+3888];
ld.shared.u32 r2912, [r2996+3892];
ld.shared.u32 r2727, [r2996+5832];
ld.shared.u32 r2733, [r2996+5836];
ld.shared.u32 r2815, [r2996+7776];
ld.shared.u32 r2821, [r2996+7780];
ld.shared.u32 r2903, [r2996+9720];
ld.shared.u32 r2909, [r2996+9724];
ld.shared.u32 r2728, [r2996+11664];
ld.shared.u32 r2734, [r2996+11668];
ld.shared.u32 r2816, [r2996+13608];
ld.shared.u32 r2822, [r2996+13612];
ld.shared.u32 r2904, [r2996+15552];
ld.shared.u32 r2910, [r2996+15556];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2722, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2723, {low, high};
}
{
neg.f16x2 r2724, r2723;
}
{
add.f16x2 r2726, r2727, r2728;
}
{
add.f16x2 %0, r2730, r2726;
}
{
add.f16x2 r2732, r2733, r2734;
}
{
add.f16x2 %1, r2736, r2732;
}
{
add.f16x2 r2738, r2727, r2728;
}
{
mul.f16x2 r2741, r2738, r2722;
}
{
add.f16x2 r2744, r2730, r2741;
}
{
sub.f16x2 r2747, r2733, r2734;
}
{
mul.f16x2 r2750, r2747, r2724;
}
{
add.f16x2 %6, r2744, r2750;
}
{
add.f16x2 r2756, r2727, r2728;
}
{
mul.f16x2 r2759, r2756, r2722;
}
{
add.f16x2 r2762, r2730, r2759;
}
{
sub.f16x2 r2765, r2733, r2734;
}
{
mul.f16x2 r2768, r2765, r2724;
}
{
sub.f16x2 %12, r2762, r2768;
}
{
add.f16x2 r2774, r2733, r2734;
}
{
mul.f16x2 r2777, r2774, r2722;
}
{
add.f16x2 r2780, r2736, r2777;
}
{
sub.f16x2 r2783, r2727, r2728;
}
{
mul.f16x2 r2786, r2783, r2724;
}
{
sub.f16x2 %7, r2780, r2786;
}
{
add.f16x2 r2792, r2733, r2734;
}
{
mul.f16x2 r2795, r2792, r2722;
}
{
add.f16x2 r2798, r2736, r2795;
}
{
sub.f16x2 r2801, r2727, r2728;
}
{
mul.f16x2 r2804, r2801, r2724;
}
{
add.f16x2 %13, r2798, r2804;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2810, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2811, {low, high};
}
{
neg.f16x2 r2812, r2811;
}
{
add.f16x2 r2814, r2815, r2816;
}
{
add.f16x2 %2, r2818, r2814;
}
{
add.f16x2 r2820, r2821, r2822;
}
{
add.f16x2 %3, r2824, r2820;
}
{
add.f16x2 r2826, r2815, r2816;
}
{
mul.f16x2 r2829, r2826, r2810;
}
{
add.f16x2 r2832, r2818, r2829;
}
{
sub.f16x2 r2835, r2821, r2822;
}
{
mul.f16x2 r2838, r2835, r2812;
}
{
add.f16x2 %8, r2832, r2838;
}
{
add.f16x2 r2844, r2815, r2816;
}
{
mul.f16x2 r2847, r2844, r2810;
}
{
add.f16x2 r2850, r2818, r2847;
}
{
sub.f16x2 r2853, r2821, r2822;
}
{
mul.f16x2 r2856, r2853, r2812;
}
{
sub.f16x2 %14, r2850, r2856;
}
{
add.f16x2 r2862, r2821, r2822;
}
{
mul.f16x2 r2865, r2862, r2810;
}
{
add.f16x2 r2868, r2824, r2865;
}
{
sub.f16x2 r2871, r2815, r2816;
}
{
mul.f16x2 r2874, r2871, r2812;
}
{
sub.f16x2 %9, r2868, r2874;
}
{
add.f16x2 r2880, r2821, r2822;
}
{
mul.f16x2 r2883, r2880, r2810;
}
{
add.f16x2 r2886, r2824, r2883;
}
{
sub.f16x2 r2889, r2815, r2816;
}
{
mul.f16x2 r2892, r2889, r2812;
}
{
add.f16x2 %15, r2886, r2892;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2898, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2899, {low, high};
}
{
neg.f16x2 r2900, r2899;
}
{
add.f16x2 r2902, r2903, r2904;
}
{
add.f16x2 %4, r2906, r2902;
}
{
add.f16x2 r2908, r2909, r2910;
}
{
add.f16x2 %5, r2912, r2908;
}
{
add.f16x2 r2914, r2903, r2904;
}
{
mul.f16x2 r2917, r2914, r2898;
}
{
add.f16x2 r2920, r2906, r2917;
}
{
sub.f16x2 r2923, r2909, r2910;
}
{
mul.f16x2 r2926, r2923, r2900;
}
{
add.f16x2 %10, r2920, r2926;
}
{
add.f16x2 r2932, r2903, r2904;
}
{
mul.f16x2 r2935, r2932, r2898;
}
{
add.f16x2 r2938, r2906, r2935;
}
{
sub.f16x2 r2941, r2909, r2910;
}
{
mul.f16x2 r2944, r2941, r2900;
}
{
sub.f16x2 %16, r2938, r2944;
}
{
add.f16x2 r2950, r2909, r2910;
}
{
mul.f16x2 r2953, r2950, r2898;
}
{
add.f16x2 r2956, r2912, r2953;
}
{
sub.f16x2 r2959, r2903, r2904;
}
{
mul.f16x2 r2962, r2959, r2900;
}
{
sub.f16x2 %11, r2956, r2962;
}
{
add.f16x2 r2968, r2909, r2910;
}
{
mul.f16x2 r2971, r2968, r2898;
}
{
add.f16x2 r2974, r2912, r2971;
}
{
sub.f16x2 r2977, r2903, r2904;
}
{
mul.f16x2 r2980, r2977, r2900;
}
{
add.f16x2 %17, r2974, r2980;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<892, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<250>;
.reg .b32 r<3009>;
.reg .b64 rd<8>;
mov.u32 r2986, %tid.y;
mov.u32 r2987, %18;
mad.lo.s32 r2988, r2986, 8748, r2987;
mov.u32 r2989, %tid.x;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r177, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
mov.f32 f166, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f166;
cvt.rn.f16.f32 high, f166;
mov.b32 r265, {low, high};
}
mov.f32 f168, 0fBF248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f168;
cvt.rn.f16.f32 high, f168;
mov.b32 r266, {low, high};
}
mov.f32 f170, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f170;
cvt.rn.f16.f32 high, f170;
mov.b32 r267, {low, high};
}
mov.f32 f172, 0fBF7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f172;
cvt.rn.f16.f32 high, f172;
mov.b32 r268, {low, high};
}
mov.f32 f178, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r271, {low, high};
}
mov.f32 f180, 0fBEAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r345, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r433, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r521, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
mul.wide.u32 rd2, r2989, -2032597691;
shr.u64 rd3, rd2, 39;
cvt.u32.u64 r2990, rd3;
mul.lo.s32 r2991, r2990, 243;
sub.s32 r2992, r2989, r2991;
mad.lo.s32 r2993, r2990, 8748, r2988;
cvt.rn.f32.u32 f241, r2992;
mul.f32 f242, f241, 0f3B3C4870;
cos.approx.f32 f57, f242;
sin.approx.f32 f243, f242;
neg.f32 f58, f243;
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
mov.f32 f225, 0fBF800000;
mov.f32 f226, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
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
mad.lo.s32 r2994, r2992, 36, r2993;
st.shared.u32 [r2994], r352;
st.shared.u32 [r2994+4], r621;
st.shared.u32 [r2994+8], r658;
st.shared.u32 [r2994+12], r695;
st.shared.u32 [r2994+16], r732;
st.shared.u32 [r2994+20], r769;
st.shared.u32 [r2994+24], r806;
st.shared.u32 [r2994+28], r843;
st.shared.u32 [r2994+32], r880;
barrier.sync 0;
shl.b32 r2995, r2992, 5;
sub.s32 r2996, r2994, r2995;
ld.shared.u32 r916, [r2996];
ld.shared.u32 r1004, [r2996+972];
ld.shared.u32 r1092, [r2996+1944];
ld.shared.u32 r913, [r2996+2916];
ld.shared.u32 r1001, [r2996+3888];
ld.shared.u32 r1089, [r2996+4860];
ld.shared.u32 r914, [r2996+5832];
ld.shared.u32 r1002, [r2996+6804];
ld.shared.u32 r1090, [r2996+7776];
barrier.sync 0;
st.shared.u32 [r2994], r358;
st.shared.u32 [r2994+4], r628;
st.shared.u32 [r2994+8], r665;
st.shared.u32 [r2994+12], r702;
st.shared.u32 [r2994+16], r739;
st.shared.u32 [r2994+20], r776;
st.shared.u32 [r2994+24], r813;
st.shared.u32 [r2994+28], r850;
st.shared.u32 [r2994+32], r887;
barrier.sync 0;
ld.shared.u32 r922, [r2996];
ld.shared.u32 r1010, [r2996+972];
ld.shared.u32 r1098, [r2996+1944];
ld.shared.u32 r919, [r2996+2916];
ld.shared.u32 r1007, [r2996+3888];
ld.shared.u32 r1095, [r2996+4860];
ld.shared.u32 r920, [r2996+5832];
ld.shared.u32 r1008, [r2996+6804];
ld.shared.u32 r1096, [r2996+7776];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r908, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r996, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1084, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
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
cvt.rn.f16.f32 low, f166;
cvt.rn.f16.f32 high, f166;
mov.b32 r1172, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f168;
cvt.rn.f16.f32 high, f168;
mov.b32 r1173, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f170;
cvt.rn.f16.f32 high, f170;
mov.b32 r1174, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f172;
cvt.rn.f16.f32 high, f172;
mov.b32 r1175, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r1178, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
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
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1252, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1253, {low, high};
}
{
neg.f16x2 r1254, r1253;
}
{
add.f16x2 r1256, r1003, r1091;
}
{
add.f16x2 r1259, r915, r1256;
}
{
add.f16x2 r1262, r1009, r1097;
}
{
add.f16x2 r1265, r921, r1262;
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
add.f16x2 r1283, r1274, r1280;
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
sub.f16x2 r1301, r1292, r1298;
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
sub.f16x2 r1319, r1310, r1316;
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
add.f16x2 r1337, r1328, r1334;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1340, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1341, {low, high};
}
{
neg.f16x2 r1342, r1341;
}
{
add.f16x2 r1344, r1194, r1210;
}
{
add.f16x2 r1347, r939, r1344;
}
{
add.f16x2 r1350, r1200, r1216;
}
{
add.f16x2 r1353, r975, r1350;
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
add.f16x2 r1371, r1362, r1368;
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
sub.f16x2 r1389, r1380, r1386;
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
sub.f16x2 r1407, r1398, r1404;
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
add.f16x2 r1425, r1416, r1422;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1428, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1429, {low, high};
}
{
neg.f16x2 r1430, r1429;
}
{
add.f16x2 r1432, r1226, r1242;
}
{
add.f16x2 r1435, r957, r1432;
}
{
add.f16x2 r1438, r1232, r1248;
}
{
add.f16x2 r1441, r993, r1438;
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
add.f16x2 r1459, r1450, r1456;
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
sub.f16x2 r1477, r1468, r1474;
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
sub.f16x2 r1495, r1486, r1492;
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
add.f16x2 r1513, r1504, r1510;
}
mul.wide.u32 rd4, r2992, 954437177;
shr.u64 rd5, rd4, 33;
cvt.u32.u64 r2997, rd5;
mul.lo.s32 r2998, r2997, 9;
sub.s32 r2999, r2992, r2998;
shl.b32 r3000, r2999, 2;
add.s32 r3001, r2993, r3000;
cvt.rn.f32.u32 f244, r2997;
mul.f32 f245, f244, 0f3CD3D17E;
cos.approx.f32 f133, f245;
sin.approx.f32 f246, f245;
neg.f32 f134, f246;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f133;
cvt.rn.f16.f32 high, f134;
mov.b32 r1516, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1519, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1521, {high, high};
}
{
mul.f16x2 r1523, r1353, r1521;
}
{
neg.f16x2 r1526, r1523;
}
{
fma.rn.f16x2 r1528, r1347, r1519, r1526;
}
{
mul.f16x2 r1532, r1347, r1521;
}
{
fma.rn.f16x2 r1535, r1353, r1519, r1532;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1539, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1541, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1543, {low, high};
}
{
mul.f16x2 r1544, r1541, r1543;
}
{
mul.f16x2 r1547, r1516, r1539;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1550, {high, low};
}
{
fma.rn.f16x2 r1552, r1544, r1550, r1547;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1552;
mov.b32 r1556, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1552;
mov.b32 r1558, {high, high};
}
{
mul.f16x2 r1560, r1441, r1558;
}
{
neg.f16x2 r1563, r1560;
}
{
fma.rn.f16x2 r1565, r1435, r1556, r1563;
}
{
mul.f16x2 r1569, r1435, r1558;
}
{
fma.rn.f16x2 r1572, r1441, r1556, r1569;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1576, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1578, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1580, {low, high};
}
{
mul.f16x2 r1581, r1578, r1580;
}
{
mul.f16x2 r1584, r1552, r1576;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1552;
mov.b32 r1587, {high, low};
}
{
fma.rn.f16x2 r1589, r1581, r1587, r1584;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1589;
mov.b32 r1593, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1589;
mov.b32 r1595, {high, high};
}
{
mul.f16x2 r1597, r1319, r1595;
}
{
neg.f16x2 r1600, r1597;
}
{
fma.rn.f16x2 r1602, r1283, r1593, r1600;
}
{
mul.f16x2 r1606, r1283, r1595;
}
{
fma.rn.f16x2 r1609, r1319, r1593, r1606;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1613, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1615, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1617, {low, high};
}
{
mul.f16x2 r1618, r1615, r1617;
}
{
mul.f16x2 r1621, r1589, r1613;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1589;
mov.b32 r1624, {high, low};
}
{
fma.rn.f16x2 r1626, r1618, r1624, r1621;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1630, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1632, {high, high};
}
{
mul.f16x2 r1634, r1407, r1632;
}
{
neg.f16x2 r1637, r1634;
}
{
fma.rn.f16x2 r1639, r1371, r1630, r1637;
}
{
mul.f16x2 r1643, r1371, r1632;
}
{
fma.rn.f16x2 r1646, r1407, r1630, r1643;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1650, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1652, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1654, {low, high};
}
{
mul.f16x2 r1655, r1652, r1654;
}
{
mul.f16x2 r1658, r1626, r1650;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1661, {high, low};
}
{
fma.rn.f16x2 r1663, r1655, r1661, r1658;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1663;
mov.b32 r1667, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1663;
mov.b32 r1669, {high, high};
}
{
mul.f16x2 r1671, r1495, r1669;
}
{
neg.f16x2 r1674, r1671;
}
{
fma.rn.f16x2 r1676, r1459, r1667, r1674;
}
{
mul.f16x2 r1680, r1459, r1669;
}
{
fma.rn.f16x2 r1683, r1495, r1667, r1680;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1687, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1689, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1691, {low, high};
}
{
mul.f16x2 r1692, r1689, r1691;
}
{
mul.f16x2 r1695, r1663, r1687;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1663;
mov.b32 r1698, {high, low};
}
{
fma.rn.f16x2 r1700, r1692, r1698, r1695;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1700;
mov.b32 r1704, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1700;
mov.b32 r1706, {high, high};
}
{
mul.f16x2 r1708, r1337, r1706;
}
{
neg.f16x2 r1711, r1708;
}
{
fma.rn.f16x2 r1713, r1301, r1704, r1711;
}
{
mul.f16x2 r1717, r1301, r1706;
}
{
fma.rn.f16x2 r1720, r1337, r1704, r1717;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1724, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1726, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1728, {low, high};
}
{
mul.f16x2 r1729, r1726, r1728;
}
{
mul.f16x2 r1732, r1700, r1724;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1700;
mov.b32 r1735, {high, low};
}
{
fma.rn.f16x2 r1737, r1729, r1735, r1732;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1737;
mov.b32 r1741, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1737;
mov.b32 r1743, {high, high};
}
{
mul.f16x2 r1745, r1425, r1743;
}
{
neg.f16x2 r1748, r1745;
}
{
fma.rn.f16x2 r1750, r1389, r1741, r1748;
}
{
mul.f16x2 r1754, r1389, r1743;
}
{
fma.rn.f16x2 r1757, r1425, r1741, r1754;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1761, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1516;
mov.b32 r1763, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r1765, {low, high};
}
{
mul.f16x2 r1766, r1763, r1765;
}
{
mul.f16x2 r1769, r1737, r1761;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1737;
mov.b32 r1772, {high, low};
}
{
fma.rn.f16x2 r1774, r1766, r1772, r1769;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1774;
mov.b32 r1778, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1774;
mov.b32 r1780, {high, high};
}
{
mul.f16x2 r1782, r1513, r1780;
}
{
neg.f16x2 r1785, r1782;
}
{
fma.rn.f16x2 r1787, r1477, r1778, r1785;
}
{
mul.f16x2 r1791, r1477, r1780;
}
{
fma.rn.f16x2 r1794, r1513, r1778, r1791;
}
barrier.sync 0;
mad.lo.s32 r3002, r2997, 324, r3001;
st.shared.u32 [r3002], r1259;
st.shared.u32 [r3002+36], r1528;
st.shared.u32 [r3002+72], r1565;
st.shared.u32 [r3002+108], r1602;
st.shared.u32 [r3002+144], r1639;
st.shared.u32 [r3002+180], r1676;
st.shared.u32 [r3002+216], r1713;
st.shared.u32 [r3002+252], r1750;
st.shared.u32 [r3002+288], r1787;
barrier.sync 0;
ld.shared.u32 r1823, [r2996];
ld.shared.u32 r1911, [r2996+972];
ld.shared.u32 r1999, [r2996+1944];
ld.shared.u32 r1820, [r2996+2916];
ld.shared.u32 r1908, [r2996+3888];
ld.shared.u32 r1996, [r2996+4860];
ld.shared.u32 r1821, [r2996+5832];
ld.shared.u32 r1909, [r2996+6804];
ld.shared.u32 r1997, [r2996+7776];
barrier.sync 0;
st.shared.u32 [r3002], r1265;
st.shared.u32 [r3002+36], r1535;
st.shared.u32 [r3002+72], r1572;
st.shared.u32 [r3002+108], r1609;
st.shared.u32 [r3002+144], r1646;
st.shared.u32 [r3002+180], r1683;
st.shared.u32 [r3002+216], r1720;
st.shared.u32 [r3002+252], r1757;
st.shared.u32 [r3002+288], r1794;
barrier.sync 0;
ld.shared.u32 r1829, [r2996];
ld.shared.u32 r1917, [r2996+972];
ld.shared.u32 r2005, [r2996+1944];
ld.shared.u32 r1826, [r2996+2916];
ld.shared.u32 r1914, [r2996+3888];
ld.shared.u32 r2002, [r2996+4860];
ld.shared.u32 r1827, [r2996+5832];
ld.shared.u32 r1915, [r2996+6804];
ld.shared.u32 r2003, [r2996+7776];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1815, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1816, {low, high};
}
{
neg.f16x2 r1817, r1816;
}
{
add.f16x2 r1819, r1820, r1821;
}
{
add.f16x2 r1822, r1823, r1819;
}
{
add.f16x2 r1825, r1826, r1827;
}
{
add.f16x2 r1828, r1829, r1825;
}
{
add.f16x2 r1831, r1820, r1821;
}
{
mul.f16x2 r1834, r1831, r1815;
}
{
add.f16x2 r1837, r1823, r1834;
}
{
sub.f16x2 r1840, r1826, r1827;
}
{
mul.f16x2 r1843, r1840, r1817;
}
{
add.f16x2 r1846, r1837, r1843;
}
{
add.f16x2 r1849, r1820, r1821;
}
{
mul.f16x2 r1852, r1849, r1815;
}
{
add.f16x2 r1855, r1823, r1852;
}
{
sub.f16x2 r1858, r1826, r1827;
}
{
mul.f16x2 r1861, r1858, r1817;
}
{
sub.f16x2 r1864, r1855, r1861;
}
{
add.f16x2 r1867, r1826, r1827;
}
{
mul.f16x2 r1870, r1867, r1815;
}
{
add.f16x2 r1873, r1829, r1870;
}
{
sub.f16x2 r1876, r1820, r1821;
}
{
mul.f16x2 r1879, r1876, r1817;
}
{
sub.f16x2 r1882, r1873, r1879;
}
{
add.f16x2 r1885, r1826, r1827;
}
{
mul.f16x2 r1888, r1885, r1815;
}
{
add.f16x2 r1891, r1829, r1888;
}
{
sub.f16x2 r1894, r1820, r1821;
}
{
mul.f16x2 r1897, r1894, r1817;
}
{
add.f16x2 r1900, r1891, r1897;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1903, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1904, {low, high};
}
{
neg.f16x2 r1905, r1904;
}
{
add.f16x2 r1907, r1908, r1909;
}
{
add.f16x2 r1910, r1911, r1907;
}
{
add.f16x2 r1913, r1914, r1915;
}
{
add.f16x2 r1916, r1917, r1913;
}
{
add.f16x2 r1919, r1908, r1909;
}
{
mul.f16x2 r1922, r1919, r1903;
}
{
add.f16x2 r1925, r1911, r1922;
}
{
sub.f16x2 r1928, r1914, r1915;
}
{
mul.f16x2 r1931, r1928, r1905;
}
{
add.f16x2 r1934, r1925, r1931;
}
{
add.f16x2 r1937, r1908, r1909;
}
{
mul.f16x2 r1940, r1937, r1903;
}
{
add.f16x2 r1943, r1911, r1940;
}
{
sub.f16x2 r1946, r1914, r1915;
}
{
mul.f16x2 r1949, r1946, r1905;
}
{
sub.f16x2 r1952, r1943, r1949;
}
{
add.f16x2 r1955, r1914, r1915;
}
{
mul.f16x2 r1958, r1955, r1903;
}
{
add.f16x2 r1961, r1917, r1958;
}
{
sub.f16x2 r1964, r1908, r1909;
}
{
mul.f16x2 r1967, r1964, r1905;
}
{
sub.f16x2 r1970, r1961, r1967;
}
{
add.f16x2 r1973, r1914, r1915;
}
{
mul.f16x2 r1976, r1973, r1903;
}
{
add.f16x2 r1979, r1917, r1976;
}
{
sub.f16x2 r1982, r1908, r1909;
}
{
mul.f16x2 r1985, r1982, r1905;
}
{
add.f16x2 r1988, r1979, r1985;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r1991, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r1992, {low, high};
}
{
neg.f16x2 r1993, r1992;
}
{
add.f16x2 r1995, r1996, r1997;
}
{
add.f16x2 r1998, r1999, r1995;
}
{
add.f16x2 r2001, r2002, r2003;
}
{
add.f16x2 r2004, r2005, r2001;
}
{
add.f16x2 r2007, r1996, r1997;
}
{
mul.f16x2 r2010, r2007, r1991;
}
{
add.f16x2 r2013, r1999, r2010;
}
{
sub.f16x2 r2016, r2002, r2003;
}
{
mul.f16x2 r2019, r2016, r1993;
}
{
add.f16x2 r2022, r2013, r2019;
}
{
add.f16x2 r2025, r1996, r1997;
}
{
mul.f16x2 r2028, r2025, r1991;
}
{
add.f16x2 r2031, r1999, r2028;
}
{
sub.f16x2 r2034, r2002, r2003;
}
{
mul.f16x2 r2037, r2034, r1993;
}
{
sub.f16x2 r2040, r2031, r2037;
}
{
add.f16x2 r2043, r2002, r2003;
}
{
mul.f16x2 r2046, r2043, r1991;
}
{
add.f16x2 r2049, r2005, r2046;
}
{
sub.f16x2 r2052, r1996, r1997;
}
{
mul.f16x2 r2055, r2052, r1993;
}
{
sub.f16x2 r2058, r2049, r2055;
}
{
add.f16x2 r2061, r2002, r2003;
}
{
mul.f16x2 r2064, r2061, r1991;
}
{
add.f16x2 r2067, r2005, r2064;
}
{
sub.f16x2 r2070, r1996, r1997;
}
{
mul.f16x2 r2073, r2070, r1993;
}
{
add.f16x2 r2076, r2067, r2073;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f166;
cvt.rn.f16.f32 high, f166;
mov.b32 r2079, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f168;
cvt.rn.f16.f32 high, f168;
mov.b32 r2080, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f170;
cvt.rn.f16.f32 high, f170;
mov.b32 r2081, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f172;
cvt.rn.f16.f32 high, f172;
mov.b32 r2082, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f178;
cvt.rn.f16.f32 high, f178;
mov.b32 r2085, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f180;
cvt.rn.f16.f32 high, f180;
mov.b32 r2086, {low, high};
}
{
mul.f16x2 r2095, r1934, r2079;
}
{
mul.f16x2 r2098, r1970, r2080;
}
{
sub.f16x2 r2101, r2095, r2098;
}
{
mul.f16x2 r2104, r1934, r2080;
}
{
fma.rn.f16x2 r2107, r1970, r2079, r2104;
}
{
mul.f16x2 r2111, r2022, r2081;
}
{
mul.f16x2 r2114, r2058, r2082;
}
{
sub.f16x2 r2117, r2111, r2114;
}
{
mul.f16x2 r2120, r2022, r2082;
}
{
fma.rn.f16x2 r2123, r2058, r2081, r2120;
}
{
mul.f16x2 r2127, r1952, r2081;
}
{
mul.f16x2 r2130, r1988, r2082;
}
{
sub.f16x2 r2133, r2127, r2130;
}
{
mul.f16x2 r2136, r1952, r2082;
}
{
fma.rn.f16x2 r2139, r1988, r2081, r2136;
}
{
mul.f16x2 r2143, r2040, r2085;
}
{
mul.f16x2 r2146, r2076, r2086;
}
{
sub.f16x2 r2149, r2143, r2146;
}
{
mul.f16x2 r2152, r2040, r2086;
}
{
fma.rn.f16x2 r2155, r2076, r2085, r2152;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2159, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2160, {low, high};
}
{
neg.f16x2 r2161, r2160;
}
{
add.f16x2 r2163, r1910, r1998;
}
{
add.f16x2 r2166, r1822, r2163;
}
{
add.f16x2 r2169, r1916, r2004;
}
{
add.f16x2 r2172, r1828, r2169;
}
{
add.f16x2 r2175, r1910, r1998;
}
{
mul.f16x2 r2178, r2175, r2159;
}
{
add.f16x2 r2181, r1822, r2178;
}
{
sub.f16x2 r2184, r1916, r2004;
}
{
mul.f16x2 r2187, r2184, r2161;
}
{
add.f16x2 r2190, r2181, r2187;
}
{
add.f16x2 r2193, r1910, r1998;
}
{
mul.f16x2 r2196, r2193, r2159;
}
{
add.f16x2 r2199, r1822, r2196;
}
{
sub.f16x2 r2202, r1916, r2004;
}
{
mul.f16x2 r2205, r2202, r2161;
}
{
sub.f16x2 r2208, r2199, r2205;
}
{
add.f16x2 r2211, r1916, r2004;
}
{
mul.f16x2 r2214, r2211, r2159;
}
{
add.f16x2 r2217, r1828, r2214;
}
{
sub.f16x2 r2220, r1910, r1998;
}
{
mul.f16x2 r2223, r2220, r2161;
}
{
sub.f16x2 r2226, r2217, r2223;
}
{
add.f16x2 r2229, r1916, r2004;
}
{
mul.f16x2 r2232, r2229, r2159;
}
{
add.f16x2 r2235, r1828, r2232;
}
{
sub.f16x2 r2238, r1910, r1998;
}
{
mul.f16x2 r2241, r2238, r2161;
}
{
add.f16x2 r2244, r2235, r2241;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2247, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2248, {low, high};
}
{
neg.f16x2 r2249, r2248;
}
{
add.f16x2 r2251, r2101, r2117;
}
{
add.f16x2 r2254, r1846, r2251;
}
{
add.f16x2 r2257, r2107, r2123;
}
{
add.f16x2 r2260, r1882, r2257;
}
{
add.f16x2 r2263, r2101, r2117;
}
{
mul.f16x2 r2266, r2263, r2247;
}
{
add.f16x2 r2269, r1846, r2266;
}
{
sub.f16x2 r2272, r2107, r2123;
}
{
mul.f16x2 r2275, r2272, r2249;
}
{
add.f16x2 r2278, r2269, r2275;
}
{
add.f16x2 r2281, r2101, r2117;
}
{
mul.f16x2 r2284, r2281, r2247;
}
{
add.f16x2 r2287, r1846, r2284;
}
{
sub.f16x2 r2290, r2107, r2123;
}
{
mul.f16x2 r2293, r2290, r2249;
}
{
sub.f16x2 r2296, r2287, r2293;
}
{
add.f16x2 r2299, r2107, r2123;
}
{
mul.f16x2 r2302, r2299, r2247;
}
{
add.f16x2 r2305, r1882, r2302;
}
{
sub.f16x2 r2308, r2101, r2117;
}
{
mul.f16x2 r2311, r2308, r2249;
}
{
sub.f16x2 r2314, r2305, r2311;
}
{
add.f16x2 r2317, r2107, r2123;
}
{
mul.f16x2 r2320, r2317, r2247;
}
{
add.f16x2 r2323, r1882, r2320;
}
{
sub.f16x2 r2326, r2101, r2117;
}
{
mul.f16x2 r2329, r2326, r2249;
}
{
add.f16x2 r2332, r2323, r2329;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2335, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2336, {low, high};
}
{
neg.f16x2 r2337, r2336;
}
{
add.f16x2 r2339, r2133, r2149;
}
{
add.f16x2 r2342, r1864, r2339;
}
{
add.f16x2 r2345, r2139, r2155;
}
{
add.f16x2 r2348, r1900, r2345;
}
{
add.f16x2 r2351, r2133, r2149;
}
{
mul.f16x2 r2354, r2351, r2335;
}
{
add.f16x2 r2357, r1864, r2354;
}
{
sub.f16x2 r2360, r2139, r2155;
}
{
mul.f16x2 r2363, r2360, r2337;
}
{
add.f16x2 r2366, r2357, r2363;
}
{
add.f16x2 r2369, r2133, r2149;
}
{
mul.f16x2 r2372, r2369, r2335;
}
{
add.f16x2 r2375, r1864, r2372;
}
{
sub.f16x2 r2378, r2139, r2155;
}
{
mul.f16x2 r2381, r2378, r2337;
}
{
sub.f16x2 r2384, r2375, r2381;
}
{
add.f16x2 r2387, r2139, r2155;
}
{
mul.f16x2 r2390, r2387, r2335;
}
{
add.f16x2 r2393, r1900, r2390;
}
{
sub.f16x2 r2396, r2133, r2149;
}
{
mul.f16x2 r2399, r2396, r2337;
}
{
sub.f16x2 r2402, r2393, r2399;
}
{
add.f16x2 r2405, r2139, r2155;
}
{
mul.f16x2 r2408, r2405, r2335;
}
{
add.f16x2 r2411, r1900, r2408;
}
{
sub.f16x2 r2414, r2133, r2149;
}
{
mul.f16x2 r2417, r2414, r2337;
}
{
add.f16x2 r2420, r2411, r2417;
}
mul.wide.u32 rd6, r2992, -901412889;
shr.u64 rd7, rd6, 38;
cvt.u32.u64 r3003, rd7;
mul.lo.s32 r3004, r3003, 81;
sub.s32 r3005, r2992, r3004;
shl.b32 r3006, r3005, 2;
add.s32 r3007, r2993, r3006;
cvt.rn.f32.u32 f247, r3003;
mul.f32 f248, f247, 0f3E6E4BAE;
cos.approx.f32 f209, f248;
sin.approx.f32 f249, f248;
neg.f32 f210, f249;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f209;
cvt.rn.f16.f32 high, f210;
mov.b32 r2423, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2426, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2428, {high, high};
}
{
mul.f16x2 r2430, r2260, r2428;
}
{
neg.f16x2 r2433, r2430;
}
{
fma.rn.f16x2 r2435, r2254, r2426, r2433;
}
{
mul.f16x2 r2439, r2254, r2428;
}
{
fma.rn.f16x2 r2442, r2260, r2426, r2439;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2446, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2448, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2450, {low, high};
}
{
mul.f16x2 r2451, r2448, r2450;
}
{
mul.f16x2 r2454, r2423, r2446;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2457, {high, low};
}
{
fma.rn.f16x2 r2459, r2451, r2457, r2454;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2459;
mov.b32 r2463, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2459;
mov.b32 r2465, {high, high};
}
{
mul.f16x2 r2467, r2348, r2465;
}
{
neg.f16x2 r2470, r2467;
}
{
fma.rn.f16x2 r2472, r2342, r2463, r2470;
}
{
mul.f16x2 r2476, r2342, r2465;
}
{
fma.rn.f16x2 r2479, r2348, r2463, r2476;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2483, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2485, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2487, {low, high};
}
{
mul.f16x2 r2488, r2485, r2487;
}
{
mul.f16x2 r2491, r2459, r2483;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2459;
mov.b32 r2494, {high, low};
}
{
fma.rn.f16x2 r2496, r2488, r2494, r2491;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2496;
mov.b32 r2500, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2496;
mov.b32 r2502, {high, high};
}
{
mul.f16x2 r2504, r2226, r2502;
}
{
neg.f16x2 r2507, r2504;
}
{
fma.rn.f16x2 r2509, r2190, r2500, r2507;
}
{
mul.f16x2 r2513, r2190, r2502;
}
{
fma.rn.f16x2 r2516, r2226, r2500, r2513;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2520, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2522, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2524, {low, high};
}
{
mul.f16x2 r2525, r2522, r2524;
}
{
mul.f16x2 r2528, r2496, r2520;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2496;
mov.b32 r2531, {high, low};
}
{
fma.rn.f16x2 r2533, r2525, r2531, r2528;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2533;
mov.b32 r2537, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2533;
mov.b32 r2539, {high, high};
}
{
mul.f16x2 r2541, r2314, r2539;
}
{
neg.f16x2 r2544, r2541;
}
{
fma.rn.f16x2 r2546, r2278, r2537, r2544;
}
{
mul.f16x2 r2550, r2278, r2539;
}
{
fma.rn.f16x2 r2553, r2314, r2537, r2550;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2557, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2559, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2561, {low, high};
}
{
mul.f16x2 r2562, r2559, r2561;
}
{
mul.f16x2 r2565, r2533, r2557;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2533;
mov.b32 r2568, {high, low};
}
{
fma.rn.f16x2 r2570, r2562, r2568, r2565;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2570;
mov.b32 r2574, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2570;
mov.b32 r2576, {high, high};
}
{
mul.f16x2 r2578, r2402, r2576;
}
{
neg.f16x2 r2581, r2578;
}
{
fma.rn.f16x2 r2583, r2366, r2574, r2581;
}
{
mul.f16x2 r2587, r2366, r2576;
}
{
fma.rn.f16x2 r2590, r2402, r2574, r2587;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2594, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2596, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2598, {low, high};
}
{
mul.f16x2 r2599, r2596, r2598;
}
{
mul.f16x2 r2602, r2570, r2594;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2570;
mov.b32 r2605, {high, low};
}
{
fma.rn.f16x2 r2607, r2599, r2605, r2602;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2607;
mov.b32 r2611, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2607;
mov.b32 r2613, {high, high};
}
{
mul.f16x2 r2615, r2244, r2613;
}
{
neg.f16x2 r2618, r2615;
}
{
fma.rn.f16x2 r2620, r2208, r2611, r2618;
}
{
mul.f16x2 r2624, r2208, r2613;
}
{
fma.rn.f16x2 r2627, r2244, r2611, r2624;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2631, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2633, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2635, {low, high};
}
{
mul.f16x2 r2636, r2633, r2635;
}
{
mul.f16x2 r2639, r2607, r2631;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2607;
mov.b32 r2642, {high, low};
}
{
fma.rn.f16x2 r2644, r2636, r2642, r2639;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2644;
mov.b32 r2648, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2644;
mov.b32 r2650, {high, high};
}
{
mul.f16x2 r2652, r2332, r2650;
}
{
neg.f16x2 r2655, r2652;
}
{
fma.rn.f16x2 r2657, r2296, r2648, r2655;
}
{
mul.f16x2 r2661, r2296, r2650;
}
{
fma.rn.f16x2 r2664, r2332, r2648, r2661;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2668, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2423;
mov.b32 r2670, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r2672, {low, high};
}
{
mul.f16x2 r2673, r2670, r2672;
}
{
mul.f16x2 r2676, r2644, r2668;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2644;
mov.b32 r2679, {high, low};
}
{
fma.rn.f16x2 r2681, r2673, r2679, r2676;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2681;
mov.b32 r2685, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2681;
mov.b32 r2687, {high, high};
}
{
mul.f16x2 r2689, r2420, r2687;
}
{
neg.f16x2 r2692, r2689;
}
{
fma.rn.f16x2 r2694, r2384, r2685, r2692;
}
{
mul.f16x2 r2698, r2384, r2687;
}
{
fma.rn.f16x2 r2701, r2420, r2685, r2698;
}
barrier.sync 0;
mad.lo.s32 r3008, r3003, 2916, r3007;
st.shared.u32 [r3008], r2166;
st.shared.u32 [r3008+324], r2435;
st.shared.u32 [r3008+648], r2472;
st.shared.u32 [r3008+972], r2509;
st.shared.u32 [r3008+1296], r2546;
st.shared.u32 [r3008+1620], r2583;
st.shared.u32 [r3008+1944], r2620;
st.shared.u32 [r3008+2268], r2657;
st.shared.u32 [r3008+2592], r2694;
barrier.sync 0;
ld.shared.u32 r2730, [r2996];
ld.shared.u32 r2818, [r2996+972];
ld.shared.u32 r2906, [r2996+1944];
ld.shared.u32 r2727, [r2996+2916];
ld.shared.u32 r2815, [r2996+3888];
ld.shared.u32 r2903, [r2996+4860];
ld.shared.u32 r2728, [r2996+5832];
ld.shared.u32 r2816, [r2996+6804];
ld.shared.u32 r2904, [r2996+7776];
barrier.sync 0;
st.shared.u32 [r3008], r2172;
st.shared.u32 [r3008+324], r2442;
st.shared.u32 [r3008+648], r2479;
st.shared.u32 [r3008+972], r2516;
st.shared.u32 [r3008+1296], r2553;
st.shared.u32 [r3008+1620], r2590;
st.shared.u32 [r3008+1944], r2627;
st.shared.u32 [r3008+2268], r2664;
st.shared.u32 [r3008+2592], r2701;
barrier.sync 0;
ld.shared.u32 r2736, [r2996];
ld.shared.u32 r2824, [r2996+972];
ld.shared.u32 r2912, [r2996+1944];
ld.shared.u32 r2733, [r2996+2916];
ld.shared.u32 r2821, [r2996+3888];
ld.shared.u32 r2909, [r2996+4860];
ld.shared.u32 r2734, [r2996+5832];
ld.shared.u32 r2822, [r2996+6804];
ld.shared.u32 r2910, [r2996+7776];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2722, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2723, {low, high};
}
{
neg.f16x2 r2724, r2723;
}
{
add.f16x2 r2726, r2727, r2728;
}
{
add.f16x2 %0, r2730, r2726;
}
{
add.f16x2 r2732, r2733, r2734;
}
{
add.f16x2 %1, r2736, r2732;
}
{
add.f16x2 r2738, r2727, r2728;
}
{
mul.f16x2 r2741, r2738, r2722;
}
{
add.f16x2 r2744, r2730, r2741;
}
{
sub.f16x2 r2747, r2733, r2734;
}
{
mul.f16x2 r2750, r2747, r2724;
}
{
add.f16x2 %6, r2744, r2750;
}
{
add.f16x2 r2756, r2727, r2728;
}
{
mul.f16x2 r2759, r2756, r2722;
}
{
add.f16x2 r2762, r2730, r2759;
}
{
sub.f16x2 r2765, r2733, r2734;
}
{
mul.f16x2 r2768, r2765, r2724;
}
{
sub.f16x2 %12, r2762, r2768;
}
{
add.f16x2 r2774, r2733, r2734;
}
{
mul.f16x2 r2777, r2774, r2722;
}
{
add.f16x2 r2780, r2736, r2777;
}
{
sub.f16x2 r2783, r2727, r2728;
}
{
mul.f16x2 r2786, r2783, r2724;
}
{
sub.f16x2 %7, r2780, r2786;
}
{
add.f16x2 r2792, r2733, r2734;
}
{
mul.f16x2 r2795, r2792, r2722;
}
{
add.f16x2 r2798, r2736, r2795;
}
{
sub.f16x2 r2801, r2727, r2728;
}
{
mul.f16x2 r2804, r2801, r2724;
}
{
add.f16x2 %13, r2798, r2804;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2810, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2811, {low, high};
}
{
neg.f16x2 r2812, r2811;
}
{
add.f16x2 r2814, r2815, r2816;
}
{
add.f16x2 %2, r2818, r2814;
}
{
add.f16x2 r2820, r2821, r2822;
}
{
add.f16x2 %3, r2824, r2820;
}
{
add.f16x2 r2826, r2815, r2816;
}
{
mul.f16x2 r2829, r2826, r2810;
}
{
add.f16x2 r2832, r2818, r2829;
}
{
sub.f16x2 r2835, r2821, r2822;
}
{
mul.f16x2 r2838, r2835, r2812;
}
{
add.f16x2 %8, r2832, r2838;
}
{
add.f16x2 r2844, r2815, r2816;
}
{
mul.f16x2 r2847, r2844, r2810;
}
{
add.f16x2 r2850, r2818, r2847;
}
{
sub.f16x2 r2853, r2821, r2822;
}
{
mul.f16x2 r2856, r2853, r2812;
}
{
sub.f16x2 %14, r2850, r2856;
}
{
add.f16x2 r2862, r2821, r2822;
}
{
mul.f16x2 r2865, r2862, r2810;
}
{
add.f16x2 r2868, r2824, r2865;
}
{
sub.f16x2 r2871, r2815, r2816;
}
{
mul.f16x2 r2874, r2871, r2812;
}
{
sub.f16x2 %9, r2868, r2874;
}
{
add.f16x2 r2880, r2821, r2822;
}
{
mul.f16x2 r2883, r2880, r2810;
}
{
add.f16x2 r2886, r2824, r2883;
}
{
sub.f16x2 r2889, r2815, r2816;
}
{
mul.f16x2 r2892, r2889, r2812;
}
{
add.f16x2 %15, r2886, r2892;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f238;
mov.b32 r2898, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f240;
cvt.rn.f16.f32 high, f240;
mov.b32 r2899, {low, high};
}
{
neg.f16x2 r2900, r2899;
}
{
add.f16x2 r2902, r2903, r2904;
}
{
add.f16x2 %4, r2906, r2902;
}
{
add.f16x2 r2908, r2909, r2910;
}
{
add.f16x2 %5, r2912, r2908;
}
{
add.f16x2 r2914, r2903, r2904;
}
{
mul.f16x2 r2917, r2914, r2898;
}
{
add.f16x2 r2920, r2906, r2917;
}
{
sub.f16x2 r2923, r2909, r2910;
}
{
mul.f16x2 r2926, r2923, r2900;
}
{
add.f16x2 %10, r2920, r2926;
}
{
add.f16x2 r2932, r2903, r2904;
}
{
mul.f16x2 r2935, r2932, r2898;
}
{
add.f16x2 r2938, r2906, r2935;
}
{
sub.f16x2 r2941, r2909, r2910;
}
{
mul.f16x2 r2944, r2941, r2900;
}
{
sub.f16x2 %16, r2938, r2944;
}
{
add.f16x2 r2950, r2909, r2910;
}
{
mul.f16x2 r2953, r2950, r2898;
}
{
add.f16x2 r2956, r2912, r2953;
}
{
sub.f16x2 r2959, r2903, r2904;
}
{
mul.f16x2 r2962, r2959, r2900;
}
{
sub.f16x2 %11, r2956, r2962;
}
{
add.f16x2 r2968, r2909, r2910;
}
{
mul.f16x2 r2971, r2968, r2898;
}
{
add.f16x2 r2974, r2912, r2971;
}
{
sub.f16x2 r2977, r2903, r2904;
}
{
mul.f16x2 r2980, r2977, r2900;
}
{
add.f16x2 %17, r2974, r2980;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<893, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<771>;
.reg .b32 r<8646>;
.reg .b64 rd<6>;
mov.u32 r8572, %54;
mov.u32 r8645, %tid.y;
mad.lo.s32 r8573, r8645, 8748, r8572;
mov.u32 r8574, %tid.x;
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
neg.f16x2 r3, r2;
}
{
add.f16x2 r5, %92, %85;
}
{
add.f16x2 r8, %68, r5;
}
{
add.f16x2 r11, %100, %91;
}
{
add.f16x2 r14, %76, r11;
}
{
add.f16x2 r17, %92, %85;
}
{
mul.f16x2 r20, r17, r1;
}
{
add.f16x2 r23, %68, r20;
}
{
sub.f16x2 r26, %100, %91;
}
{
mul.f16x2 r29, r26, r3;
}
{
add.f16x2 r32, r23, r29;
}
{
add.f16x2 r35, %92, %85;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %68, r38;
}
{
sub.f16x2 r44, %100, %91;
}
{
mul.f16x2 r47, r44, r3;
}
{
sub.f16x2 r50, r41, r47;
}
{
add.f16x2 r53, %100, %91;
}
{
mul.f16x2 r56, r53, r1;
}
{
add.f16x2 r59, %76, r56;
}
{
sub.f16x2 r62, %92, %85;
}
{
mul.f16x2 r65, r62, r3;
}
{
sub.f16x2 r68, r59, r65;
}
{
add.f16x2 r71, %100, %91;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %76, r74;
}
{
sub.f16x2 r80, %92, %85;
}
{
mul.f16x2 r83, r80, r3;
}
{
add.f16x2 r86, r77, r83;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r90, {low, high};
}
{
neg.f16x2 r91, r90;
}
{
add.f16x2 r93, %77, %67;
}
{
add.f16x2 r96, %106, r93;
}
{
add.f16x2 r99, %84, %75;
}
{
add.f16x2 r102, %58, r99;
}
{
add.f16x2 r105, %77, %67;
}
{
mul.f16x2 r108, r105, r89;
}
{
add.f16x2 r111, %106, r108;
}
{
sub.f16x2 r114, %84, %75;
}
{
mul.f16x2 r117, r114, r91;
}
{
add.f16x2 r120, r111, r117;
}
{
add.f16x2 r123, %77, %67;
}
{
mul.f16x2 r126, r123, r89;
}
{
add.f16x2 r129, %106, r126;
}
{
sub.f16x2 r132, %84, %75;
}
{
mul.f16x2 r135, r132, r91;
}
{
sub.f16x2 r138, r129, r135;
}
{
add.f16x2 r141, %84, %75;
}
{
mul.f16x2 r144, r141, r89;
}
{
add.f16x2 r147, %58, r144;
}
{
sub.f16x2 r150, %77, %67;
}
{
mul.f16x2 r153, r150, r91;
}
{
sub.f16x2 r156, r147, r153;
}
{
add.f16x2 r159, %84, %75;
}
{
mul.f16x2 r162, r159, r89;
}
{
add.f16x2 r165, %58, r162;
}
{
sub.f16x2 r168, %77, %67;
}
{
mul.f16x2 r171, r168, r91;
}
{
add.f16x2 r174, r165, r171;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r177, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r178, {low, high};
}
{
neg.f16x2 r179, r178;
}
{
add.f16x2 r181, %59, %105;
}
{
add.f16x2 r184, %90, r181;
}
{
add.f16x2 r187, %66, %57;
}
{
add.f16x2 r190, %99, r187;
}
{
add.f16x2 r193, %59, %105;
}
{
mul.f16x2 r196, r193, r177;
}
{
add.f16x2 r199, %90, r196;
}
{
sub.f16x2 r202, %66, %57;
}
{
mul.f16x2 r205, r202, r179;
}
{
add.f16x2 r208, r199, r205;
}
{
add.f16x2 r211, %59, %105;
}
{
mul.f16x2 r214, r211, r177;
}
{
add.f16x2 r217, %90, r214;
}
{
sub.f16x2 r220, %66, %57;
}
{
mul.f16x2 r223, r220, r179;
}
{
sub.f16x2 r226, r217, r223;
}
{
add.f16x2 r229, %66, %57;
}
{
mul.f16x2 r232, r229, r177;
}
{
add.f16x2 r235, %99, r232;
}
{
sub.f16x2 r238, %59, %105;
}
{
mul.f16x2 r241, r238, r179;
}
{
sub.f16x2 r244, r235, r241;
}
{
add.f16x2 r247, %66, %57;
}
{
mul.f16x2 r250, r247, r177;
}
{
add.f16x2 r253, %99, r250;
}
{
sub.f16x2 r256, %59, %105;
}
{
mul.f16x2 r259, r256, r179;
}
{
add.f16x2 r262, r253, r259;
}
mov.f32 f542, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r265, {low, high};
}
mov.f32 f544, 0fBF248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r266, {low, high};
}
mov.f32 f554, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r267, {low, high};
}
mov.f32 f556, 0fBF7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r268, {low, high};
}
mov.f32 f578, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r271, {low, high};
}
mov.f32 f580, 0fBEAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r345, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r433, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r521, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r609, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r610, {low, high};
}
{
neg.f16x2 r611, r610;
}
{
add.f16x2 r613, %95, %87;
}
{
add.f16x2 r616, %71, r613;
}
{
add.f16x2 r619, %102, %94;
}
{
add.f16x2 r622, %79, r619;
}
{
add.f16x2 r625, %95, %87;
}
{
mul.f16x2 r628, r625, r609;
}
{
add.f16x2 r631, %71, r628;
}
{
sub.f16x2 r634, %102, %94;
}
{
mul.f16x2 r637, r634, r611;
}
{
add.f16x2 r640, r631, r637;
}
{
add.f16x2 r643, %95, %87;
}
{
mul.f16x2 r646, r643, r609;
}
{
add.f16x2 r649, %71, r646;
}
{
sub.f16x2 r652, %102, %94;
}
{
mul.f16x2 r655, r652, r611;
}
{
sub.f16x2 r658, r649, r655;
}
{
add.f16x2 r661, %102, %94;
}
{
mul.f16x2 r664, r661, r609;
}
{
add.f16x2 r667, %79, r664;
}
{
sub.f16x2 r670, %95, %87;
}
{
mul.f16x2 r673, r670, r611;
}
{
sub.f16x2 r676, r667, r673;
}
{
add.f16x2 r679, %102, %94;
}
{
mul.f16x2 r682, r679, r609;
}
{
add.f16x2 r685, %79, r682;
}
{
sub.f16x2 r688, %95, %87;
}
{
mul.f16x2 r691, r688, r611;
}
{
add.f16x2 r694, r685, r691;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r697, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r698, {low, high};
}
{
neg.f16x2 r699, r698;
}
{
add.f16x2 r701, %80, %70;
}
{
add.f16x2 r704, %108, r701;
}
{
add.f16x2 r707, %86, %78;
}
{
add.f16x2 r710, %61, r707;
}
{
add.f16x2 r713, %80, %70;
}
{
mul.f16x2 r716, r713, r697;
}
{
add.f16x2 r719, %108, r716;
}
{
sub.f16x2 r722, %86, %78;
}
{
mul.f16x2 r725, r722, r699;
}
{
add.f16x2 r728, r719, r725;
}
{
add.f16x2 r731, %80, %70;
}
{
mul.f16x2 r734, r731, r697;
}
{
add.f16x2 r737, %108, r734;
}
{
sub.f16x2 r740, %86, %78;
}
{
mul.f16x2 r743, r740, r699;
}
{
sub.f16x2 r746, r737, r743;
}
{
add.f16x2 r749, %86, %78;
}
{
mul.f16x2 r752, r749, r697;
}
{
add.f16x2 r755, %61, r752;
}
{
sub.f16x2 r758, %80, %70;
}
{
mul.f16x2 r761, r758, r699;
}
{
sub.f16x2 r764, r755, r761;
}
{
add.f16x2 r767, %86, %78;
}
{
mul.f16x2 r770, r767, r697;
}
{
add.f16x2 r773, %61, r770;
}
{
sub.f16x2 r776, %80, %70;
}
{
mul.f16x2 r779, r776, r699;
}
{
add.f16x2 r782, r773, r779;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r785, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r786, {low, high};
}
{
neg.f16x2 r787, r786;
}
{
add.f16x2 r789, %62, %107;
}
{
add.f16x2 r792, %93, r789;
}
{
add.f16x2 r795, %69, %60;
}
{
add.f16x2 r798, %101, r795;
}
{
add.f16x2 r801, %62, %107;
}
{
mul.f16x2 r804, r801, r785;
}
{
add.f16x2 r807, %93, r804;
}
{
sub.f16x2 r810, %69, %60;
}
{
mul.f16x2 r813, r810, r787;
}
{
add.f16x2 r816, r807, r813;
}
{
add.f16x2 r819, %62, %107;
}
{
mul.f16x2 r822, r819, r785;
}
{
add.f16x2 r825, %93, r822;
}
{
sub.f16x2 r828, %69, %60;
}
{
mul.f16x2 r831, r828, r787;
}
{
sub.f16x2 r834, r825, r831;
}
{
add.f16x2 r837, %69, %60;
}
{
mul.f16x2 r840, r837, r785;
}
{
add.f16x2 r843, %101, r840;
}
{
sub.f16x2 r846, %62, %107;
}
{
mul.f16x2 r849, r846, r787;
}
{
sub.f16x2 r852, r843, r849;
}
{
add.f16x2 r855, %69, %60;
}
{
mul.f16x2 r858, r855, r785;
}
{
add.f16x2 r861, %101, r858;
}
{
sub.f16x2 r864, %62, %107;
}
{
mul.f16x2 r867, r864, r787;
}
{
add.f16x2 r870, r861, r867;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r873, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r874, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r875, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r876, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r879, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r953, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1041, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1129, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1217, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r1218, {low, high};
}
{
neg.f16x2 r1219, r1218;
}
{
add.f16x2 r1221, %98, %89;
}
{
add.f16x2 r1224, %74, r1221;
}
{
add.f16x2 r1227, %104, %97;
}
{
add.f16x2 r1230, %82, r1227;
}
{
add.f16x2 r1233, %98, %89;
}
{
mul.f16x2 r1236, r1233, r1217;
}
{
add.f16x2 r1239, %74, r1236;
}
{
sub.f16x2 r1242, %104, %97;
}
{
mul.f16x2 r1245, r1242, r1219;
}
{
add.f16x2 r1248, r1239, r1245;
}
{
add.f16x2 r1251, %98, %89;
}
{
mul.f16x2 r1254, r1251, r1217;
}
{
add.f16x2 r1257, %74, r1254;
}
{
sub.f16x2 r1260, %104, %97;
}
{
mul.f16x2 r1263, r1260, r1219;
}
{
sub.f16x2 r1266, r1257, r1263;
}
{
add.f16x2 r1269, %104, %97;
}
{
mul.f16x2 r1272, r1269, r1217;
}
{
add.f16x2 r1275, %82, r1272;
}
{
sub.f16x2 r1278, %98, %89;
}
{
mul.f16x2 r1281, r1278, r1219;
}
{
sub.f16x2 r1284, r1275, r1281;
}
{
add.f16x2 r1287, %104, %97;
}
{
mul.f16x2 r1290, r1287, r1217;
}
{
add.f16x2 r1293, %82, r1290;
}
{
sub.f16x2 r1296, %98, %89;
}
{
mul.f16x2 r1299, r1296, r1219;
}
{
add.f16x2 r1302, r1293, r1299;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1305, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r1306, {low, high};
}
{
neg.f16x2 r1307, r1306;
}
{
add.f16x2 r1309, %83, %73;
}
{
add.f16x2 r1312, %56, r1309;
}
{
add.f16x2 r1315, %88, %81;
}
{
add.f16x2 r1318, %64, r1315;
}
{
add.f16x2 r1321, %83, %73;
}
{
mul.f16x2 r1324, r1321, r1305;
}
{
add.f16x2 r1327, %56, r1324;
}
{
sub.f16x2 r1330, %88, %81;
}
{
mul.f16x2 r1333, r1330, r1307;
}
{
add.f16x2 r1336, r1327, r1333;
}
{
add.f16x2 r1339, %83, %73;
}
{
mul.f16x2 r1342, r1339, r1305;
}
{
add.f16x2 r1345, %56, r1342;
}
{
sub.f16x2 r1348, %88, %81;
}
{
mul.f16x2 r1351, r1348, r1307;
}
{
sub.f16x2 r1354, r1345, r1351;
}
{
add.f16x2 r1357, %88, %81;
}
{
mul.f16x2 r1360, r1357, r1305;
}
{
add.f16x2 r1363, %64, r1360;
}
{
sub.f16x2 r1366, %83, %73;
}
{
mul.f16x2 r1369, r1366, r1307;
}
{
sub.f16x2 r1372, r1363, r1369;
}
{
add.f16x2 r1375, %88, %81;
}
{
mul.f16x2 r1378, r1375, r1305;
}
{
add.f16x2 r1381, %64, r1378;
}
{
sub.f16x2 r1384, %83, %73;
}
{
mul.f16x2 r1387, r1384, r1307;
}
{
add.f16x2 r1390, r1381, r1387;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1393, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r1394, {low, high};
}
{
neg.f16x2 r1395, r1394;
}
{
add.f16x2 r1397, %65, %55;
}
{
add.f16x2 r1400, %96, r1397;
}
{
add.f16x2 r1403, %72, %63;
}
{
add.f16x2 r1406, %103, r1403;
}
{
add.f16x2 r1409, %65, %55;
}
{
mul.f16x2 r1412, r1409, r1393;
}
{
add.f16x2 r1415, %96, r1412;
}
{
sub.f16x2 r1418, %72, %63;
}
{
mul.f16x2 r1421, r1418, r1395;
}
{
add.f16x2 r1424, r1415, r1421;
}
{
add.f16x2 r1427, %65, %55;
}
{
mul.f16x2 r1430, r1427, r1393;
}
{
add.f16x2 r1433, %96, r1430;
}
{
sub.f16x2 r1436, %72, %63;
}
{
mul.f16x2 r1439, r1436, r1395;
}
{
sub.f16x2 r1442, r1433, r1439;
}
{
add.f16x2 r1445, %72, %63;
}
{
mul.f16x2 r1448, r1445, r1393;
}
{
add.f16x2 r1451, %103, r1448;
}
{
sub.f16x2 r1454, %65, %55;
}
{
mul.f16x2 r1457, r1454, r1395;
}
{
sub.f16x2 r1460, r1451, r1457;
}
{
add.f16x2 r1463, %72, %63;
}
{
mul.f16x2 r1466, r1463, r1393;
}
{
add.f16x2 r1469, %103, r1466;
}
{
sub.f16x2 r1472, %65, %55;
}
{
mul.f16x2 r1475, r1472, r1395;
}
{
add.f16x2 r1478, r1469, r1475;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r1481, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r1482, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r1483, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r1484, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r1487, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1561, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1649, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r1737, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
mov.f32 f534, 0f3F791978;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f534;
cvt.rn.f16.f32 high, f534;
mov.b32 r1825, {low, high};
}
mov.f32 f536, 0fBE6C2691;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f536;
cvt.rn.f16.f32 high, f536;
mov.b32 r1826, {low, high};
}
mov.f32 f538, 0f3F64C51C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f538;
cvt.rn.f16.f32 high, f538;
mov.b32 r1827, {low, high};
}
mov.f32 f540, 0fBEE5C902;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f540;
cvt.rn.f16.f32 high, f540;
mov.b32 r1828, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r1829, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r1830, {low, high};
}
mov.f32 f546, 0f3F18DF63;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f546;
cvt.rn.f16.f32 high, f546;
mov.b32 r1831, {low, high};
}
mov.f32 f548, 0fBF4D57F2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f548;
cvt.rn.f16.f32 high, f548;
mov.b32 r1832, {low, high};
}
mov.f32 f550, 0f3ECACAF8;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f550;
cvt.rn.f16.f32 high, f550;
mov.b32 r1833, {low, high};
}
mov.f32 f552, 0fBF6B1036;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f552;
cvt.rn.f16.f32 high, f552;
mov.b32 r1834, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r1835, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r1836, {low, high};
}
mov.f32 f558, 0fBD6E2946;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f558;
cvt.rn.f16.f32 high, f558;
mov.b32 r1837, {low, high};
}
mov.f32 f560, 0fBF7F9120;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f560;
cvt.rn.f16.f32 high, f560;
mov.b32 r1838, {low, high};
}
mov.f32 f562, 0fBE92D7E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f562;
cvt.rn.f16.f32 high, f562;
mov.b32 r1839, {low, high};
}
mov.f32 f564, 0fBF753ECD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f564;
cvt.rn.f16.f32 high, f564;
mov.b32 r1840, {low, high};
}
mov.f32 f570, 0fBF2FAD88;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f570;
cvt.rn.f16.f32 high, f570;
mov.b32 r1843, {low, high};
}
mov.f32 f572, 0fBF3A3529;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f572;
cvt.rn.f16.f32 high, f572;
mov.b32 r1844, {low, high};
}
mov.f32 f594, 0fBF55E287;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r1847, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
mov.b32 r1848, {low, high};
}
mov.f32 f586, 0fBF7E44DE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f586;
cvt.rn.f16.f32 high, f586;
mov.b32 r1851, {low, high};
}
mov.f32 f588, 0f3DEDC21F;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f588;
cvt.rn.f16.f32 high, f588;
mov.b32 r1852, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1855, {low, high};
}
mov.f32 f596, 0f3F0CAC9F;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2133, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2221, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2309, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2485, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2573, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2661, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2749, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r2837, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
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
mul.wide.u32 rd2, r8574, -901412889;
shr.u64 rd3, rd2, 38;
cvt.u32.u64 r8575, rd3;
mul.lo.s32 r8576, r8575, 81;
sub.s32 r8577, r8574, r8576;
mad.lo.s32 r8578, r8575, 8748, r8573;
cvt.rn.f32.u32 f765, r8577;
mul.f32 f766, f765, 0f3B3C4870;
cos.approx.f32 f309, f766;
sin.approx.f32 f767, f766;
neg.f32 f310, f767;
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
mov.f32 f725, 0fBF800000;
mov.f32 f726, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
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
mad.lo.s32 r8579, r8577, 108, r8578;
st.shared.u32 [r8579], r2140;
st.shared.u32 [r8579+4], r2937;
st.shared.u32 [r8579+8], r2974;
st.shared.u32 [r8579+12], r3011;
st.shared.u32 [r8579+16], r3048;
st.shared.u32 [r8579+20], r3085;
st.shared.u32 [r8579+24], r3122;
st.shared.u32 [r8579+28], r3159;
st.shared.u32 [r8579+32], r3196;
st.shared.u32 [r8579+36], r3233;
st.shared.u32 [r8579+40], r3270;
st.shared.u32 [r8579+44], r3307;
st.shared.u32 [r8579+48], r3344;
st.shared.u32 [r8579+52], r3381;
st.shared.u32 [r8579+56], r3418;
st.shared.u32 [r8579+60], r3455;
st.shared.u32 [r8579+64], r3492;
st.shared.u32 [r8579+68], r3529;
st.shared.u32 [r8579+72], r3566;
st.shared.u32 [r8579+76], r3603;
st.shared.u32 [r8579+80], r3640;
st.shared.u32 [r8579+84], r3677;
st.shared.u32 [r8579+88], r3714;
st.shared.u32 [r8579+92], r3751;
st.shared.u32 [r8579+96], r3788;
st.shared.u32 [r8579+100], r3825;
st.shared.u32 [r8579+104], r3862;
barrier.sync 0;
mad.lo.s32 r8580, r8577, -104, r8579;
ld.shared.u32 r3898, [r8580];
ld.shared.u32 r4506, [r8580+324];
ld.shared.u32 r5114, [r8580+648];
ld.shared.u32 r3986, [r8580+972];
ld.shared.u32 r4594, [r8580+1296];
ld.shared.u32 r5202, [r8580+1620];
ld.shared.u32 r4074, [r8580+1944];
ld.shared.u32 r4682, [r8580+2268];
ld.shared.u32 r5290, [r8580+2592];
ld.shared.u32 r3895, [r8580+2916];
ld.shared.u32 r4503, [r8580+3240];
ld.shared.u32 r5111, [r8580+3564];
ld.shared.u32 r3983, [r8580+3888];
ld.shared.u32 r4591, [r8580+4212];
ld.shared.u32 r5199, [r8580+4536];
ld.shared.u32 r4071, [r8580+4860];
ld.shared.u32 r4679, [r8580+5184];
ld.shared.u32 r5287, [r8580+5508];
ld.shared.u32 r3896, [r8580+5832];
ld.shared.u32 r4504, [r8580+6156];
ld.shared.u32 r5112, [r8580+6480];
ld.shared.u32 r3984, [r8580+6804];
ld.shared.u32 r4592, [r8580+7128];
ld.shared.u32 r5200, [r8580+7452];
ld.shared.u32 r4072, [r8580+7776];
ld.shared.u32 r4680, [r8580+8100];
ld.shared.u32 r5288, [r8580+8424];
barrier.sync 0;
st.shared.u32 [r8579], r2146;
st.shared.u32 [r8579+4], r2944;
st.shared.u32 [r8579+8], r2981;
st.shared.u32 [r8579+12], r3018;
st.shared.u32 [r8579+16], r3055;
st.shared.u32 [r8579+20], r3092;
st.shared.u32 [r8579+24], r3129;
st.shared.u32 [r8579+28], r3166;
st.shared.u32 [r8579+32], r3203;
st.shared.u32 [r8579+36], r3240;
st.shared.u32 [r8579+40], r3277;
st.shared.u32 [r8579+44], r3314;
st.shared.u32 [r8579+48], r3351;
st.shared.u32 [r8579+52], r3388;
st.shared.u32 [r8579+56], r3425;
st.shared.u32 [r8579+60], r3462;
st.shared.u32 [r8579+64], r3499;
st.shared.u32 [r8579+68], r3536;
st.shared.u32 [r8579+72], r3573;
st.shared.u32 [r8579+76], r3610;
st.shared.u32 [r8579+80], r3647;
st.shared.u32 [r8579+84], r3684;
st.shared.u32 [r8579+88], r3721;
st.shared.u32 [r8579+92], r3758;
st.shared.u32 [r8579+96], r3795;
st.shared.u32 [r8579+100], r3832;
st.shared.u32 [r8579+104], r3869;
barrier.sync 0;
ld.shared.u32 r3904, [r8580];
ld.shared.u32 r4512, [r8580+324];
ld.shared.u32 r5120, [r8580+648];
ld.shared.u32 r3992, [r8580+972];
ld.shared.u32 r4600, [r8580+1296];
ld.shared.u32 r5208, [r8580+1620];
ld.shared.u32 r4080, [r8580+1944];
ld.shared.u32 r4688, [r8580+2268];
ld.shared.u32 r5296, [r8580+2592];
ld.shared.u32 r3901, [r8580+2916];
ld.shared.u32 r4509, [r8580+3240];
ld.shared.u32 r5117, [r8580+3564];
ld.shared.u32 r3989, [r8580+3888];
ld.shared.u32 r4597, [r8580+4212];
ld.shared.u32 r5205, [r8580+4536];
ld.shared.u32 r4077, [r8580+4860];
ld.shared.u32 r4685, [r8580+5184];
ld.shared.u32 r5293, [r8580+5508];
ld.shared.u32 r3902, [r8580+5832];
ld.shared.u32 r4510, [r8580+6156];
ld.shared.u32 r5118, [r8580+6480];
ld.shared.u32 r3990, [r8580+6804];
ld.shared.u32 r4598, [r8580+7128];
ld.shared.u32 r5206, [r8580+7452];
ld.shared.u32 r4078, [r8580+7776];
ld.shared.u32 r4686, [r8580+8100];
ld.shared.u32 r5294, [r8580+8424];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r3890, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r3891, {low, high};
}
{
neg.f16x2 r3892, r3891;
}
{
add.f16x2 r3894, r3895, r3896;
}
{
add.f16x2 r3897, r3898, r3894;
}
{
add.f16x2 r3900, r3901, r3902;
}
{
add.f16x2 r3903, r3904, r3900;
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
add.f16x2 r3921, r3912, r3918;
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
sub.f16x2 r3939, r3930, r3936;
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
sub.f16x2 r3957, r3948, r3954;
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
add.f16x2 r3975, r3966, r3972;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r3978, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r3979, {low, high};
}
{
neg.f16x2 r3980, r3979;
}
{
add.f16x2 r3982, r3983, r3984;
}
{
add.f16x2 r3985, r3986, r3982;
}
{
add.f16x2 r3988, r3989, r3990;
}
{
add.f16x2 r3991, r3992, r3988;
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
add.f16x2 r4009, r4000, r4006;
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
sub.f16x2 r4027, r4018, r4024;
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
sub.f16x2 r4045, r4036, r4042;
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
add.f16x2 r4063, r4054, r4060;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4066, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4067, {low, high};
}
{
neg.f16x2 r4068, r4067;
}
{
add.f16x2 r4070, r4071, r4072;
}
{
add.f16x2 r4073, r4074, r4070;
}
{
add.f16x2 r4076, r4077, r4078;
}
{
add.f16x2 r4079, r4080, r4076;
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
add.f16x2 r4097, r4088, r4094;
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
sub.f16x2 r4115, r4106, r4112;
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
sub.f16x2 r4133, r4124, r4130;
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
add.f16x2 r4151, r4142, r4148;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r4154, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r4155, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r4156, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r4157, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r4160, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
mov.b32 r4161, {low, high};
}
{
mul.f16x2 r4170, r4009, r4154;
}
{
mul.f16x2 r4173, r4045, r4155;
}
{
sub.f16x2 r4176, r4170, r4173;
}
{
mul.f16x2 r4179, r4009, r4155;
}
{
fma.rn.f16x2 r4182, r4045, r4154, r4179;
}
{
mul.f16x2 r4186, r4097, r4156;
}
{
mul.f16x2 r4189, r4133, r4157;
}
{
sub.f16x2 r4192, r4186, r4189;
}
{
mul.f16x2 r4195, r4097, r4157;
}
{
fma.rn.f16x2 r4198, r4133, r4156, r4195;
}
{
mul.f16x2 r4202, r4027, r4156;
}
{
mul.f16x2 r4205, r4063, r4157;
}
{
sub.f16x2 r4208, r4202, r4205;
}
{
mul.f16x2 r4211, r4027, r4157;
}
{
fma.rn.f16x2 r4214, r4063, r4156, r4211;
}
{
mul.f16x2 r4218, r4115, r4160;
}
{
mul.f16x2 r4221, r4151, r4161;
}
{
sub.f16x2 r4224, r4218, r4221;
}
{
mul.f16x2 r4227, r4115, r4161;
}
{
fma.rn.f16x2 r4230, r4151, r4160, r4227;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4234, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4235, {low, high};
}
{
neg.f16x2 r4236, r4235;
}
{
add.f16x2 r4238, r3985, r4073;
}
{
add.f16x2 r4241, r3897, r4238;
}
{
add.f16x2 r4244, r3991, r4079;
}
{
add.f16x2 r4247, r3903, r4244;
}
{
add.f16x2 r4250, r3985, r4073;
}
{
mul.f16x2 r4253, r4250, r4234;
}
{
add.f16x2 r4256, r3897, r4253;
}
{
sub.f16x2 r4259, r3991, r4079;
}
{
mul.f16x2 r4262, r4259, r4236;
}
{
add.f16x2 r4265, r4256, r4262;
}
{
add.f16x2 r4268, r3985, r4073;
}
{
mul.f16x2 r4271, r4268, r4234;
}
{
add.f16x2 r4274, r3897, r4271;
}
{
sub.f16x2 r4277, r3991, r4079;
}
{
mul.f16x2 r4280, r4277, r4236;
}
{
sub.f16x2 r4283, r4274, r4280;
}
{
add.f16x2 r4286, r3991, r4079;
}
{
mul.f16x2 r4289, r4286, r4234;
}
{
add.f16x2 r4292, r3903, r4289;
}
{
sub.f16x2 r4295, r3985, r4073;
}
{
mul.f16x2 r4298, r4295, r4236;
}
{
sub.f16x2 r4301, r4292, r4298;
}
{
add.f16x2 r4304, r3991, r4079;
}
{
mul.f16x2 r4307, r4304, r4234;
}
{
add.f16x2 r4310, r3903, r4307;
}
{
sub.f16x2 r4313, r3985, r4073;
}
{
mul.f16x2 r4316, r4313, r4236;
}
{
add.f16x2 r4319, r4310, r4316;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4322, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4323, {low, high};
}
{
neg.f16x2 r4324, r4323;
}
{
add.f16x2 r4326, r4176, r4192;
}
{
add.f16x2 r4329, r3921, r4326;
}
{
add.f16x2 r4332, r4182, r4198;
}
{
add.f16x2 r4335, r3957, r4332;
}
{
add.f16x2 r4338, r4176, r4192;
}
{
mul.f16x2 r4341, r4338, r4322;
}
{
add.f16x2 r4344, r3921, r4341;
}
{
sub.f16x2 r4347, r4182, r4198;
}
{
mul.f16x2 r4350, r4347, r4324;
}
{
add.f16x2 r4353, r4344, r4350;
}
{
add.f16x2 r4356, r4176, r4192;
}
{
mul.f16x2 r4359, r4356, r4322;
}
{
add.f16x2 r4362, r3921, r4359;
}
{
sub.f16x2 r4365, r4182, r4198;
}
{
mul.f16x2 r4368, r4365, r4324;
}
{
sub.f16x2 r4371, r4362, r4368;
}
{
add.f16x2 r4374, r4182, r4198;
}
{
mul.f16x2 r4377, r4374, r4322;
}
{
add.f16x2 r4380, r3957, r4377;
}
{
sub.f16x2 r4383, r4176, r4192;
}
{
mul.f16x2 r4386, r4383, r4324;
}
{
sub.f16x2 r4389, r4380, r4386;
}
{
add.f16x2 r4392, r4182, r4198;
}
{
mul.f16x2 r4395, r4392, r4322;
}
{
add.f16x2 r4398, r3957, r4395;
}
{
sub.f16x2 r4401, r4176, r4192;
}
{
mul.f16x2 r4404, r4401, r4324;
}
{
add.f16x2 r4407, r4398, r4404;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4410, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4411, {low, high};
}
{
neg.f16x2 r4412, r4411;
}
{
add.f16x2 r4414, r4208, r4224;
}
{
add.f16x2 r4417, r3939, r4414;
}
{
add.f16x2 r4420, r4214, r4230;
}
{
add.f16x2 r4423, r3975, r4420;
}
{
add.f16x2 r4426, r4208, r4224;
}
{
mul.f16x2 r4429, r4426, r4410;
}
{
add.f16x2 r4432, r3939, r4429;
}
{
sub.f16x2 r4435, r4214, r4230;
}
{
mul.f16x2 r4438, r4435, r4412;
}
{
add.f16x2 r4441, r4432, r4438;
}
{
add.f16x2 r4444, r4208, r4224;
}
{
mul.f16x2 r4447, r4444, r4410;
}
{
add.f16x2 r4450, r3939, r4447;
}
{
sub.f16x2 r4453, r4214, r4230;
}
{
mul.f16x2 r4456, r4453, r4412;
}
{
sub.f16x2 r4459, r4450, r4456;
}
{
add.f16x2 r4462, r4214, r4230;
}
{
mul.f16x2 r4465, r4462, r4410;
}
{
add.f16x2 r4468, r3975, r4465;
}
{
sub.f16x2 r4471, r4208, r4224;
}
{
mul.f16x2 r4474, r4471, r4412;
}
{
sub.f16x2 r4477, r4468, r4474;
}
{
add.f16x2 r4480, r4214, r4230;
}
{
mul.f16x2 r4483, r4480, r4410;
}
{
add.f16x2 r4486, r3975, r4483;
}
{
sub.f16x2 r4489, r4208, r4224;
}
{
mul.f16x2 r4492, r4489, r4412;
}
{
add.f16x2 r4495, r4486, r4492;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4498, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4499, {low, high};
}
{
neg.f16x2 r4500, r4499;
}
{
add.f16x2 r4502, r4503, r4504;
}
{
add.f16x2 r4505, r4506, r4502;
}
{
add.f16x2 r4508, r4509, r4510;
}
{
add.f16x2 r4511, r4512, r4508;
}
{
add.f16x2 r4514, r4503, r4504;
}
{
mul.f16x2 r4517, r4514, r4498;
}
{
add.f16x2 r4520, r4506, r4517;
}
{
sub.f16x2 r4523, r4509, r4510;
}
{
mul.f16x2 r4526, r4523, r4500;
}
{
add.f16x2 r4529, r4520, r4526;
}
{
add.f16x2 r4532, r4503, r4504;
}
{
mul.f16x2 r4535, r4532, r4498;
}
{
add.f16x2 r4538, r4506, r4535;
}
{
sub.f16x2 r4541, r4509, r4510;
}
{
mul.f16x2 r4544, r4541, r4500;
}
{
sub.f16x2 r4547, r4538, r4544;
}
{
add.f16x2 r4550, r4509, r4510;
}
{
mul.f16x2 r4553, r4550, r4498;
}
{
add.f16x2 r4556, r4512, r4553;
}
{
sub.f16x2 r4559, r4503, r4504;
}
{
mul.f16x2 r4562, r4559, r4500;
}
{
sub.f16x2 r4565, r4556, r4562;
}
{
add.f16x2 r4568, r4509, r4510;
}
{
mul.f16x2 r4571, r4568, r4498;
}
{
add.f16x2 r4574, r4512, r4571;
}
{
sub.f16x2 r4577, r4503, r4504;
}
{
mul.f16x2 r4580, r4577, r4500;
}
{
add.f16x2 r4583, r4574, r4580;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4586, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4587, {low, high};
}
{
neg.f16x2 r4588, r4587;
}
{
add.f16x2 r4590, r4591, r4592;
}
{
add.f16x2 r4593, r4594, r4590;
}
{
add.f16x2 r4596, r4597, r4598;
}
{
add.f16x2 r4599, r4600, r4596;
}
{
add.f16x2 r4602, r4591, r4592;
}
{
mul.f16x2 r4605, r4602, r4586;
}
{
add.f16x2 r4608, r4594, r4605;
}
{
sub.f16x2 r4611, r4597, r4598;
}
{
mul.f16x2 r4614, r4611, r4588;
}
{
add.f16x2 r4617, r4608, r4614;
}
{
add.f16x2 r4620, r4591, r4592;
}
{
mul.f16x2 r4623, r4620, r4586;
}
{
add.f16x2 r4626, r4594, r4623;
}
{
sub.f16x2 r4629, r4597, r4598;
}
{
mul.f16x2 r4632, r4629, r4588;
}
{
sub.f16x2 r4635, r4626, r4632;
}
{
add.f16x2 r4638, r4597, r4598;
}
{
mul.f16x2 r4641, r4638, r4586;
}
{
add.f16x2 r4644, r4600, r4641;
}
{
sub.f16x2 r4647, r4591, r4592;
}
{
mul.f16x2 r4650, r4647, r4588;
}
{
sub.f16x2 r4653, r4644, r4650;
}
{
add.f16x2 r4656, r4597, r4598;
}
{
mul.f16x2 r4659, r4656, r4586;
}
{
add.f16x2 r4662, r4600, r4659;
}
{
sub.f16x2 r4665, r4591, r4592;
}
{
mul.f16x2 r4668, r4665, r4588;
}
{
add.f16x2 r4671, r4662, r4668;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4674, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4675, {low, high};
}
{
neg.f16x2 r4676, r4675;
}
{
add.f16x2 r4678, r4679, r4680;
}
{
add.f16x2 r4681, r4682, r4678;
}
{
add.f16x2 r4684, r4685, r4686;
}
{
add.f16x2 r4687, r4688, r4684;
}
{
add.f16x2 r4690, r4679, r4680;
}
{
mul.f16x2 r4693, r4690, r4674;
}
{
add.f16x2 r4696, r4682, r4693;
}
{
sub.f16x2 r4699, r4685, r4686;
}
{
mul.f16x2 r4702, r4699, r4676;
}
{
add.f16x2 r4705, r4696, r4702;
}
{
add.f16x2 r4708, r4679, r4680;
}
{
mul.f16x2 r4711, r4708, r4674;
}
{
add.f16x2 r4714, r4682, r4711;
}
{
sub.f16x2 r4717, r4685, r4686;
}
{
mul.f16x2 r4720, r4717, r4676;
}
{
sub.f16x2 r4723, r4714, r4720;
}
{
add.f16x2 r4726, r4685, r4686;
}
{
mul.f16x2 r4729, r4726, r4674;
}
{
add.f16x2 r4732, r4688, r4729;
}
{
sub.f16x2 r4735, r4679, r4680;
}
{
mul.f16x2 r4738, r4735, r4676;
}
{
sub.f16x2 r4741, r4732, r4738;
}
{
add.f16x2 r4744, r4685, r4686;
}
{
mul.f16x2 r4747, r4744, r4674;
}
{
add.f16x2 r4750, r4688, r4747;
}
{
sub.f16x2 r4753, r4679, r4680;
}
{
mul.f16x2 r4756, r4753, r4676;
}
{
add.f16x2 r4759, r4750, r4756;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r4762, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r4763, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r4764, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r4765, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r4768, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
mov.b32 r4769, {low, high};
}
{
mul.f16x2 r4778, r4617, r4762;
}
{
mul.f16x2 r4781, r4653, r4763;
}
{
sub.f16x2 r4784, r4778, r4781;
}
{
mul.f16x2 r4787, r4617, r4763;
}
{
fma.rn.f16x2 r4790, r4653, r4762, r4787;
}
{
mul.f16x2 r4794, r4705, r4764;
}
{
mul.f16x2 r4797, r4741, r4765;
}
{
sub.f16x2 r4800, r4794, r4797;
}
{
mul.f16x2 r4803, r4705, r4765;
}
{
fma.rn.f16x2 r4806, r4741, r4764, r4803;
}
{
mul.f16x2 r4810, r4635, r4764;
}
{
mul.f16x2 r4813, r4671, r4765;
}
{
sub.f16x2 r4816, r4810, r4813;
}
{
mul.f16x2 r4819, r4635, r4765;
}
{
fma.rn.f16x2 r4822, r4671, r4764, r4819;
}
{
mul.f16x2 r4826, r4723, r4768;
}
{
mul.f16x2 r4829, r4759, r4769;
}
{
sub.f16x2 r4832, r4826, r4829;
}
{
mul.f16x2 r4835, r4723, r4769;
}
{
fma.rn.f16x2 r4838, r4759, r4768, r4835;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4842, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4843, {low, high};
}
{
neg.f16x2 r4844, r4843;
}
{
add.f16x2 r4846, r4593, r4681;
}
{
add.f16x2 r4849, r4505, r4846;
}
{
add.f16x2 r4852, r4599, r4687;
}
{
add.f16x2 r4855, r4511, r4852;
}
{
add.f16x2 r4858, r4593, r4681;
}
{
mul.f16x2 r4861, r4858, r4842;
}
{
add.f16x2 r4864, r4505, r4861;
}
{
sub.f16x2 r4867, r4599, r4687;
}
{
mul.f16x2 r4870, r4867, r4844;
}
{
add.f16x2 r4873, r4864, r4870;
}
{
add.f16x2 r4876, r4593, r4681;
}
{
mul.f16x2 r4879, r4876, r4842;
}
{
add.f16x2 r4882, r4505, r4879;
}
{
sub.f16x2 r4885, r4599, r4687;
}
{
mul.f16x2 r4888, r4885, r4844;
}
{
sub.f16x2 r4891, r4882, r4888;
}
{
add.f16x2 r4894, r4599, r4687;
}
{
mul.f16x2 r4897, r4894, r4842;
}
{
add.f16x2 r4900, r4511, r4897;
}
{
sub.f16x2 r4903, r4593, r4681;
}
{
mul.f16x2 r4906, r4903, r4844;
}
{
sub.f16x2 r4909, r4900, r4906;
}
{
add.f16x2 r4912, r4599, r4687;
}
{
mul.f16x2 r4915, r4912, r4842;
}
{
add.f16x2 r4918, r4511, r4915;
}
{
sub.f16x2 r4921, r4593, r4681;
}
{
mul.f16x2 r4924, r4921, r4844;
}
{
add.f16x2 r4927, r4918, r4924;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r4930, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r4931, {low, high};
}
{
neg.f16x2 r4932, r4931;
}
{
add.f16x2 r4934, r4784, r4800;
}
{
add.f16x2 r4937, r4529, r4934;
}
{
add.f16x2 r4940, r4790, r4806;
}
{
add.f16x2 r4943, r4565, r4940;
}
{
add.f16x2 r4946, r4784, r4800;
}
{
mul.f16x2 r4949, r4946, r4930;
}
{
add.f16x2 r4952, r4529, r4949;
}
{
sub.f16x2 r4955, r4790, r4806;
}
{
mul.f16x2 r4958, r4955, r4932;
}
{
add.f16x2 r4961, r4952, r4958;
}
{
add.f16x2 r4964, r4784, r4800;
}
{
mul.f16x2 r4967, r4964, r4930;
}
{
add.f16x2 r4970, r4529, r4967;
}
{
sub.f16x2 r4973, r4790, r4806;
}
{
mul.f16x2 r4976, r4973, r4932;
}
{
sub.f16x2 r4979, r4970, r4976;
}
{
add.f16x2 r4982, r4790, r4806;
}
{
mul.f16x2 r4985, r4982, r4930;
}
{
add.f16x2 r4988, r4565, r4985;
}
{
sub.f16x2 r4991, r4784, r4800;
}
{
mul.f16x2 r4994, r4991, r4932;
}
{
sub.f16x2 r4997, r4988, r4994;
}
{
add.f16x2 r5000, r4790, r4806;
}
{
mul.f16x2 r5003, r5000, r4930;
}
{
add.f16x2 r5006, r4565, r5003;
}
{
sub.f16x2 r5009, r4784, r4800;
}
{
mul.f16x2 r5012, r5009, r4932;
}
{
add.f16x2 r5015, r5006, r5012;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5019, {low, high};
}
{
neg.f16x2 r5020, r5019;
}
{
add.f16x2 r5022, r4816, r4832;
}
{
add.f16x2 r5025, r4547, r5022;
}
{
add.f16x2 r5028, r4822, r4838;
}
{
add.f16x2 r5031, r4583, r5028;
}
{
add.f16x2 r5034, r4816, r4832;
}
{
mul.f16x2 r5037, r5034, r5018;
}
{
add.f16x2 r5040, r4547, r5037;
}
{
sub.f16x2 r5043, r4822, r4838;
}
{
mul.f16x2 r5046, r5043, r5020;
}
{
add.f16x2 r5049, r5040, r5046;
}
{
add.f16x2 r5052, r4816, r4832;
}
{
mul.f16x2 r5055, r5052, r5018;
}
{
add.f16x2 r5058, r4547, r5055;
}
{
sub.f16x2 r5061, r4822, r4838;
}
{
mul.f16x2 r5064, r5061, r5020;
}
{
sub.f16x2 r5067, r5058, r5064;
}
{
add.f16x2 r5070, r4822, r4838;
}
{
mul.f16x2 r5073, r5070, r5018;
}
{
add.f16x2 r5076, r4583, r5073;
}
{
sub.f16x2 r5079, r4816, r4832;
}
{
mul.f16x2 r5082, r5079, r5020;
}
{
sub.f16x2 r5085, r5076, r5082;
}
{
add.f16x2 r5088, r4822, r4838;
}
{
mul.f16x2 r5091, r5088, r5018;
}
{
add.f16x2 r5094, r4583, r5091;
}
{
sub.f16x2 r5097, r4816, r4832;
}
{
mul.f16x2 r5100, r5097, r5020;
}
{
add.f16x2 r5103, r5094, r5100;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5106, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5107, {low, high};
}
{
neg.f16x2 r5108, r5107;
}
{
add.f16x2 r5110, r5111, r5112;
}
{
add.f16x2 r5113, r5114, r5110;
}
{
add.f16x2 r5116, r5117, r5118;
}
{
add.f16x2 r5119, r5120, r5116;
}
{
add.f16x2 r5122, r5111, r5112;
}
{
mul.f16x2 r5125, r5122, r5106;
}
{
add.f16x2 r5128, r5114, r5125;
}
{
sub.f16x2 r5131, r5117, r5118;
}
{
mul.f16x2 r5134, r5131, r5108;
}
{
add.f16x2 r5137, r5128, r5134;
}
{
add.f16x2 r5140, r5111, r5112;
}
{
mul.f16x2 r5143, r5140, r5106;
}
{
add.f16x2 r5146, r5114, r5143;
}
{
sub.f16x2 r5149, r5117, r5118;
}
{
mul.f16x2 r5152, r5149, r5108;
}
{
sub.f16x2 r5155, r5146, r5152;
}
{
add.f16x2 r5158, r5117, r5118;
}
{
mul.f16x2 r5161, r5158, r5106;
}
{
add.f16x2 r5164, r5120, r5161;
}
{
sub.f16x2 r5167, r5111, r5112;
}
{
mul.f16x2 r5170, r5167, r5108;
}
{
sub.f16x2 r5173, r5164, r5170;
}
{
add.f16x2 r5176, r5117, r5118;
}
{
mul.f16x2 r5179, r5176, r5106;
}
{
add.f16x2 r5182, r5120, r5179;
}
{
sub.f16x2 r5185, r5111, r5112;
}
{
mul.f16x2 r5188, r5185, r5108;
}
{
add.f16x2 r5191, r5182, r5188;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5194, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5195, {low, high};
}
{
neg.f16x2 r5196, r5195;
}
{
add.f16x2 r5198, r5199, r5200;
}
{
add.f16x2 r5201, r5202, r5198;
}
{
add.f16x2 r5204, r5205, r5206;
}
{
add.f16x2 r5207, r5208, r5204;
}
{
add.f16x2 r5210, r5199, r5200;
}
{
mul.f16x2 r5213, r5210, r5194;
}
{
add.f16x2 r5216, r5202, r5213;
}
{
sub.f16x2 r5219, r5205, r5206;
}
{
mul.f16x2 r5222, r5219, r5196;
}
{
add.f16x2 r5225, r5216, r5222;
}
{
add.f16x2 r5228, r5199, r5200;
}
{
mul.f16x2 r5231, r5228, r5194;
}
{
add.f16x2 r5234, r5202, r5231;
}
{
sub.f16x2 r5237, r5205, r5206;
}
{
mul.f16x2 r5240, r5237, r5196;
}
{
sub.f16x2 r5243, r5234, r5240;
}
{
add.f16x2 r5246, r5205, r5206;
}
{
mul.f16x2 r5249, r5246, r5194;
}
{
add.f16x2 r5252, r5208, r5249;
}
{
sub.f16x2 r5255, r5199, r5200;
}
{
mul.f16x2 r5258, r5255, r5196;
}
{
sub.f16x2 r5261, r5252, r5258;
}
{
add.f16x2 r5264, r5205, r5206;
}
{
mul.f16x2 r5267, r5264, r5194;
}
{
add.f16x2 r5270, r5208, r5267;
}
{
sub.f16x2 r5273, r5199, r5200;
}
{
mul.f16x2 r5276, r5273, r5196;
}
{
add.f16x2 r5279, r5270, r5276;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5282, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5283, {low, high};
}
{
neg.f16x2 r5284, r5283;
}
{
add.f16x2 r5286, r5287, r5288;
}
{
add.f16x2 r5289, r5290, r5286;
}
{
add.f16x2 r5292, r5293, r5294;
}
{
add.f16x2 r5295, r5296, r5292;
}
{
add.f16x2 r5298, r5287, r5288;
}
{
mul.f16x2 r5301, r5298, r5282;
}
{
add.f16x2 r5304, r5290, r5301;
}
{
sub.f16x2 r5307, r5293, r5294;
}
{
mul.f16x2 r5310, r5307, r5284;
}
{
add.f16x2 r5313, r5304, r5310;
}
{
add.f16x2 r5316, r5287, r5288;
}
{
mul.f16x2 r5319, r5316, r5282;
}
{
add.f16x2 r5322, r5290, r5319;
}
{
sub.f16x2 r5325, r5293, r5294;
}
{
mul.f16x2 r5328, r5325, r5284;
}
{
sub.f16x2 r5331, r5322, r5328;
}
{
add.f16x2 r5334, r5293, r5294;
}
{
mul.f16x2 r5337, r5334, r5282;
}
{
add.f16x2 r5340, r5296, r5337;
}
{
sub.f16x2 r5343, r5287, r5288;
}
{
mul.f16x2 r5346, r5343, r5284;
}
{
sub.f16x2 r5349, r5340, r5346;
}
{
add.f16x2 r5352, r5293, r5294;
}
{
mul.f16x2 r5355, r5352, r5282;
}
{
add.f16x2 r5358, r5296, r5355;
}
{
sub.f16x2 r5361, r5287, r5288;
}
{
mul.f16x2 r5364, r5361, r5284;
}
{
add.f16x2 r5367, r5358, r5364;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r5370, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r5371, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r5372, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r5373, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r5376, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
mov.b32 r5377, {low, high};
}
{
mul.f16x2 r5386, r5225, r5370;
}
{
mul.f16x2 r5389, r5261, r5371;
}
{
sub.f16x2 r5392, r5386, r5389;
}
{
mul.f16x2 r5395, r5225, r5371;
}
{
fma.rn.f16x2 r5398, r5261, r5370, r5395;
}
{
mul.f16x2 r5402, r5313, r5372;
}
{
mul.f16x2 r5405, r5349, r5373;
}
{
sub.f16x2 r5408, r5402, r5405;
}
{
mul.f16x2 r5411, r5313, r5373;
}
{
fma.rn.f16x2 r5414, r5349, r5372, r5411;
}
{
mul.f16x2 r5418, r5243, r5372;
}
{
mul.f16x2 r5421, r5279, r5373;
}
{
sub.f16x2 r5424, r5418, r5421;
}
{
mul.f16x2 r5427, r5243, r5373;
}
{
fma.rn.f16x2 r5430, r5279, r5372, r5427;
}
{
mul.f16x2 r5434, r5331, r5376;
}
{
mul.f16x2 r5437, r5367, r5377;
}
{
sub.f16x2 r5440, r5434, r5437;
}
{
mul.f16x2 r5443, r5331, r5377;
}
{
fma.rn.f16x2 r5446, r5367, r5376, r5443;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5450, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5451, {low, high};
}
{
neg.f16x2 r5452, r5451;
}
{
add.f16x2 r5454, r5201, r5289;
}
{
add.f16x2 r5457, r5113, r5454;
}
{
add.f16x2 r5460, r5207, r5295;
}
{
add.f16x2 r5463, r5119, r5460;
}
{
add.f16x2 r5466, r5201, r5289;
}
{
mul.f16x2 r5469, r5466, r5450;
}
{
add.f16x2 r5472, r5113, r5469;
}
{
sub.f16x2 r5475, r5207, r5295;
}
{
mul.f16x2 r5478, r5475, r5452;
}
{
add.f16x2 r5481, r5472, r5478;
}
{
add.f16x2 r5484, r5201, r5289;
}
{
mul.f16x2 r5487, r5484, r5450;
}
{
add.f16x2 r5490, r5113, r5487;
}
{
sub.f16x2 r5493, r5207, r5295;
}
{
mul.f16x2 r5496, r5493, r5452;
}
{
sub.f16x2 r5499, r5490, r5496;
}
{
add.f16x2 r5502, r5207, r5295;
}
{
mul.f16x2 r5505, r5502, r5450;
}
{
add.f16x2 r5508, r5119, r5505;
}
{
sub.f16x2 r5511, r5201, r5289;
}
{
mul.f16x2 r5514, r5511, r5452;
}
{
sub.f16x2 r5517, r5508, r5514;
}
{
add.f16x2 r5520, r5207, r5295;
}
{
mul.f16x2 r5523, r5520, r5450;
}
{
add.f16x2 r5526, r5119, r5523;
}
{
sub.f16x2 r5529, r5201, r5289;
}
{
mul.f16x2 r5532, r5529, r5452;
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
neg.f16x2 r5540, r5539;
}
{
add.f16x2 r5542, r5392, r5408;
}
{
add.f16x2 r5545, r5137, r5542;
}
{
add.f16x2 r5548, r5398, r5414;
}
{
add.f16x2 r5551, r5173, r5548;
}
{
add.f16x2 r5554, r5392, r5408;
}
{
mul.f16x2 r5557, r5554, r5538;
}
{
add.f16x2 r5560, r5137, r5557;
}
{
sub.f16x2 r5563, r5398, r5414;
}
{
mul.f16x2 r5566, r5563, r5540;
}
{
add.f16x2 r5569, r5560, r5566;
}
{
add.f16x2 r5572, r5392, r5408;
}
{
mul.f16x2 r5575, r5572, r5538;
}
{
add.f16x2 r5578, r5137, r5575;
}
{
sub.f16x2 r5581, r5398, r5414;
}
{
mul.f16x2 r5584, r5581, r5540;
}
{
sub.f16x2 r5587, r5578, r5584;
}
{
add.f16x2 r5590, r5398, r5414;
}
{
mul.f16x2 r5593, r5590, r5538;
}
{
add.f16x2 r5596, r5173, r5593;
}
{
sub.f16x2 r5599, r5392, r5408;
}
{
mul.f16x2 r5602, r5599, r5540;
}
{
sub.f16x2 r5605, r5596, r5602;
}
{
add.f16x2 r5608, r5398, r5414;
}
{
mul.f16x2 r5611, r5608, r5538;
}
{
add.f16x2 r5614, r5173, r5611;
}
{
sub.f16x2 r5617, r5392, r5408;
}
{
mul.f16x2 r5620, r5617, r5540;
}
{
add.f16x2 r5623, r5614, r5620;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r5626, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r5627, {low, high};
}
{
neg.f16x2 r5628, r5627;
}
{
add.f16x2 r5630, r5424, r5440;
}
{
add.f16x2 r5633, r5155, r5630;
}
{
add.f16x2 r5636, r5430, r5446;
}
{
add.f16x2 r5639, r5191, r5636;
}
{
add.f16x2 r5642, r5424, r5440;
}
{
mul.f16x2 r5645, r5642, r5626;
}
{
add.f16x2 r5648, r5155, r5645;
}
{
sub.f16x2 r5651, r5430, r5446;
}
{
mul.f16x2 r5654, r5651, r5628;
}
{
add.f16x2 r5657, r5648, r5654;
}
{
add.f16x2 r5660, r5424, r5440;
}
{
mul.f16x2 r5663, r5660, r5626;
}
{
add.f16x2 r5666, r5155, r5663;
}
{
sub.f16x2 r5669, r5430, r5446;
}
{
mul.f16x2 r5672, r5669, r5628;
}
{
sub.f16x2 r5675, r5666, r5672;
}
{
add.f16x2 r5678, r5430, r5446;
}
{
mul.f16x2 r5681, r5678, r5626;
}
{
add.f16x2 r5684, r5191, r5681;
}
{
sub.f16x2 r5687, r5424, r5440;
}
{
mul.f16x2 r5690, r5687, r5628;
}
{
sub.f16x2 r5693, r5684, r5690;
}
{
add.f16x2 r5696, r5430, r5446;
}
{
mul.f16x2 r5699, r5696, r5626;
}
{
add.f16x2 r5702, r5191, r5699;
}
{
sub.f16x2 r5705, r5424, r5440;
}
{
mul.f16x2 r5708, r5705, r5628;
}
{
add.f16x2 r5711, r5702, r5708;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f534;
cvt.rn.f16.f32 high, f534;
mov.b32 r5714, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f536;
cvt.rn.f16.f32 high, f536;
mov.b32 r5715, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f538;
cvt.rn.f16.f32 high, f538;
mov.b32 r5716, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f540;
cvt.rn.f16.f32 high, f540;
mov.b32 r5717, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f542;
cvt.rn.f16.f32 high, f542;
mov.b32 r5718, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f544;
cvt.rn.f16.f32 high, f544;
mov.b32 r5719, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f546;
cvt.rn.f16.f32 high, f546;
mov.b32 r5720, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f548;
cvt.rn.f16.f32 high, f548;
mov.b32 r5721, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f550;
cvt.rn.f16.f32 high, f550;
mov.b32 r5722, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f552;
cvt.rn.f16.f32 high, f552;
mov.b32 r5723, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f554;
cvt.rn.f16.f32 high, f554;
mov.b32 r5724, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f556;
cvt.rn.f16.f32 high, f556;
mov.b32 r5725, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f558;
cvt.rn.f16.f32 high, f558;
mov.b32 r5726, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f560;
cvt.rn.f16.f32 high, f560;
mov.b32 r5727, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f562;
cvt.rn.f16.f32 high, f562;
mov.b32 r5728, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f564;
cvt.rn.f16.f32 high, f564;
mov.b32 r5729, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f570;
cvt.rn.f16.f32 high, f570;
mov.b32 r5732, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f572;
cvt.rn.f16.f32 high, f572;
mov.b32 r5733, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f578;
cvt.rn.f16.f32 high, f578;
mov.b32 r5736, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f580;
cvt.rn.f16.f32 high, f580;
mov.b32 r5737, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f586;
cvt.rn.f16.f32 high, f586;
mov.b32 r5740, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f588;
cvt.rn.f16.f32 high, f588;
mov.b32 r5741, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5744, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5745, {low, high};
}
{
mul.f16x2 r5766, r4937, r5714;
}
{
mul.f16x2 r5769, r4943, r5715;
}
{
sub.f16x2 r5772, r5766, r5769;
}
{
mul.f16x2 r5775, r4937, r5715;
}
{
fma.rn.f16x2 r5778, r4943, r5714, r5775;
}
{
mul.f16x2 r5782, r5545, r5716;
}
{
mul.f16x2 r5785, r5551, r5717;
}
{
sub.f16x2 r5788, r5782, r5785;
}
{
mul.f16x2 r5791, r5545, r5717;
}
{
fma.rn.f16x2 r5794, r5551, r5716, r5791;
}
{
mul.f16x2 r5798, r5025, r5716;
}
{
mul.f16x2 r5801, r5031, r5717;
}
{
sub.f16x2 r5804, r5798, r5801;
}
{
mul.f16x2 r5807, r5025, r5717;
}
{
fma.rn.f16x2 r5810, r5031, r5716, r5807;
}
{
mul.f16x2 r5814, r5633, r5720;
}
{
mul.f16x2 r5817, r5639, r5721;
}
{
sub.f16x2 r5820, r5814, r5817;
}
{
mul.f16x2 r5823, r5633, r5721;
}
{
fma.rn.f16x2 r5826, r5639, r5720, r5823;
}
{
mul.f16x2 r5830, r4873, r5718;
}
{
mul.f16x2 r5833, r4909, r5719;
}
{
sub.f16x2 r5836, r5830, r5833;
}
{
mul.f16x2 r5839, r4873, r5719;
}
{
fma.rn.f16x2 r5842, r4909, r5718, r5839;
}
{
mul.f16x2 r5846, r5481, r5724;
}
{
mul.f16x2 r5849, r5517, r5725;
}
{
sub.f16x2 r5852, r5846, r5849;
}
{
mul.f16x2 r5855, r5481, r5725;
}
{
fma.rn.f16x2 r5858, r5517, r5724, r5855;
}
{
mul.f16x2 r5862, r4961, r5720;
}
{
mul.f16x2 r5865, r4997, r5721;
}
{
sub.f16x2 r5868, r5862, r5865;
}
{
mul.f16x2 r5871, r4961, r5721;
}
{
fma.rn.f16x2 r5874, r4997, r5720, r5871;
}
{
mul.f16x2 r5878, r5569, r5728;
}
{
mul.f16x2 r5881, r5605, r5729;
}
{
sub.f16x2 r5884, r5878, r5881;
}
{
mul.f16x2 r5887, r5569, r5729;
}
{
fma.rn.f16x2 r5890, r5605, r5728, r5887;
}
{
mul.f16x2 r5894, r5049, r5722;
}
{
mul.f16x2 r5897, r5085, r5723;
}
{
sub.f16x2 r5900, r5894, r5897;
}
{
mul.f16x2 r5903, r5049, r5723;
}
{
fma.rn.f16x2 r5906, r5085, r5722, r5903;
}
{
mul.f16x2 r5910, r5657, r5732;
}
{
mul.f16x2 r5913, r5693, r5733;
}
{
sub.f16x2 r5916, r5910, r5913;
}
{
mul.f16x2 r5919, r5657, r5733;
}
{
fma.rn.f16x2 r5922, r5693, r5732, r5919;
}
{
mul.f16x2 r5926, r4891, r5724;
}
{
mul.f16x2 r5929, r4927, r5725;
}
{
sub.f16x2 r5932, r5926, r5929;
}
{
mul.f16x2 r5935, r4891, r5725;
}
{
fma.rn.f16x2 r5938, r4927, r5724, r5935;
}
{
mul.f16x2 r5942, r5499, r5736;
}
{
mul.f16x2 r5945, r5535, r5737;
}
{
sub.f16x2 r5948, r5942, r5945;
}
{
mul.f16x2 r5951, r5499, r5737;
}
{
fma.rn.f16x2 r5954, r5535, r5736, r5951;
}
{
mul.f16x2 r5958, r4979, r5726;
}
{
mul.f16x2 r5961, r5015, r5727;
}
{
sub.f16x2 r5964, r5958, r5961;
}
{
mul.f16x2 r5967, r4979, r5727;
}
{
fma.rn.f16x2 r5970, r5015, r5726, r5967;
}
{
mul.f16x2 r5974, r5587, r5740;
}
{
mul.f16x2 r5977, r5623, r5741;
}
{
sub.f16x2 r5980, r5974, r5977;
}
{
mul.f16x2 r5983, r5587, r5741;
}
{
fma.rn.f16x2 r5986, r5623, r5740, r5983;
}
{
mul.f16x2 r5990, r5067, r5728;
}
{
mul.f16x2 r5993, r5103, r5729;
}
{
sub.f16x2 r5996, r5990, r5993;
}
{
mul.f16x2 r5999, r5067, r5729;
}
{
fma.rn.f16x2 r6002, r5103, r5728, r5999;
}
{
mul.f16x2 r6006, r5675, r5744;
}
{
mul.f16x2 r6009, r5711, r5745;
}
{
sub.f16x2 r6012, r6006, r6009;
}
{
mul.f16x2 r6015, r5675, r5745;
}
{
fma.rn.f16x2 r6018, r5711, r5744, r6015;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6022, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6023, {low, high};
}
{
neg.f16x2 r6024, r6023;
}
{
add.f16x2 r6026, r4849, r5457;
}
{
add.f16x2 r6029, r4241, r6026;
}
{
add.f16x2 r6032, r4855, r5463;
}
{
add.f16x2 r6035, r4247, r6032;
}
{
add.f16x2 r6038, r4849, r5457;
}
{
mul.f16x2 r6041, r6038, r6022;
}
{
add.f16x2 r6044, r4241, r6041;
}
{
sub.f16x2 r6047, r4855, r5463;
}
{
mul.f16x2 r6050, r6047, r6024;
}
{
add.f16x2 r6053, r6044, r6050;
}
{
add.f16x2 r6056, r4849, r5457;
}
{
mul.f16x2 r6059, r6056, r6022;
}
{
add.f16x2 r6062, r4241, r6059;
}
{
sub.f16x2 r6065, r4855, r5463;
}
{
mul.f16x2 r6068, r6065, r6024;
}
{
sub.f16x2 r6071, r6062, r6068;
}
{
add.f16x2 r6074, r4855, r5463;
}
{
mul.f16x2 r6077, r6074, r6022;
}
{
add.f16x2 r6080, r4247, r6077;
}
{
sub.f16x2 r6083, r4849, r5457;
}
{
mul.f16x2 r6086, r6083, r6024;
}
{
sub.f16x2 r6089, r6080, r6086;
}
{
add.f16x2 r6092, r4855, r5463;
}
{
mul.f16x2 r6095, r6092, r6022;
}
{
add.f16x2 r6098, r4247, r6095;
}
{
sub.f16x2 r6101, r4849, r5457;
}
{
mul.f16x2 r6104, r6101, r6024;
}
{
add.f16x2 r6107, r6098, r6104;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6110, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6111, {low, high};
}
{
neg.f16x2 r6112, r6111;
}
{
add.f16x2 r6114, r5772, r5788;
}
{
add.f16x2 r6117, r4329, r6114;
}
{
add.f16x2 r6120, r5778, r5794;
}
{
add.f16x2 r6123, r4335, r6120;
}
{
add.f16x2 r6126, r5772, r5788;
}
{
mul.f16x2 r6129, r6126, r6110;
}
{
add.f16x2 r6132, r4329, r6129;
}
{
sub.f16x2 r6135, r5778, r5794;
}
{
mul.f16x2 r6138, r6135, r6112;
}
{
add.f16x2 r6141, r6132, r6138;
}
{
add.f16x2 r6144, r5772, r5788;
}
{
mul.f16x2 r6147, r6144, r6110;
}
{
add.f16x2 r6150, r4329, r6147;
}
{
sub.f16x2 r6153, r5778, r5794;
}
{
mul.f16x2 r6156, r6153, r6112;
}
{
sub.f16x2 r6159, r6150, r6156;
}
{
add.f16x2 r6162, r5778, r5794;
}
{
mul.f16x2 r6165, r6162, r6110;
}
{
add.f16x2 r6168, r4335, r6165;
}
{
sub.f16x2 r6171, r5772, r5788;
}
{
mul.f16x2 r6174, r6171, r6112;
}
{
sub.f16x2 r6177, r6168, r6174;
}
{
add.f16x2 r6180, r5778, r5794;
}
{
mul.f16x2 r6183, r6180, r6110;
}
{
add.f16x2 r6186, r4335, r6183;
}
{
sub.f16x2 r6189, r5772, r5788;
}
{
mul.f16x2 r6192, r6189, r6112;
}
{
add.f16x2 r6195, r6186, r6192;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6198, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6199, {low, high};
}
{
neg.f16x2 r6200, r6199;
}
{
add.f16x2 r6202, r5804, r5820;
}
{
add.f16x2 r6205, r4417, r6202;
}
{
add.f16x2 r6208, r5810, r5826;
}
{
add.f16x2 r6211, r4423, r6208;
}
{
add.f16x2 r6214, r5804, r5820;
}
{
mul.f16x2 r6217, r6214, r6198;
}
{
add.f16x2 r6220, r4417, r6217;
}
{
sub.f16x2 r6223, r5810, r5826;
}
{
mul.f16x2 r6226, r6223, r6200;
}
{
add.f16x2 r6229, r6220, r6226;
}
{
add.f16x2 r6232, r5804, r5820;
}
{
mul.f16x2 r6235, r6232, r6198;
}
{
add.f16x2 r6238, r4417, r6235;
}
{
sub.f16x2 r6241, r5810, r5826;
}
{
mul.f16x2 r6244, r6241, r6200;
}
{
sub.f16x2 r6247, r6238, r6244;
}
{
add.f16x2 r6250, r5810, r5826;
}
{
mul.f16x2 r6253, r6250, r6198;
}
{
add.f16x2 r6256, r4423, r6253;
}
{
sub.f16x2 r6259, r5804, r5820;
}
{
mul.f16x2 r6262, r6259, r6200;
}
{
sub.f16x2 r6265, r6256, r6262;
}
{
add.f16x2 r6268, r5810, r5826;
}
{
mul.f16x2 r6271, r6268, r6198;
}
{
add.f16x2 r6274, r4423, r6271;
}
{
sub.f16x2 r6277, r5804, r5820;
}
{
mul.f16x2 r6280, r6277, r6200;
}
{
add.f16x2 r6283, r6274, r6280;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6286, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6287, {low, high};
}
{
neg.f16x2 r6288, r6287;
}
{
add.f16x2 r6290, r5836, r5852;
}
{
add.f16x2 r6293, r4265, r6290;
}
{
add.f16x2 r6296, r5842, r5858;
}
{
add.f16x2 r6299, r4301, r6296;
}
{
add.f16x2 r6302, r5836, r5852;
}
{
mul.f16x2 r6305, r6302, r6286;
}
{
add.f16x2 r6308, r4265, r6305;
}
{
sub.f16x2 r6311, r5842, r5858;
}
{
mul.f16x2 r6314, r6311, r6288;
}
{
add.f16x2 r6317, r6308, r6314;
}
{
add.f16x2 r6320, r5836, r5852;
}
{
mul.f16x2 r6323, r6320, r6286;
}
{
add.f16x2 r6326, r4265, r6323;
}
{
sub.f16x2 r6329, r5842, r5858;
}
{
mul.f16x2 r6332, r6329, r6288;
}
{
sub.f16x2 r6335, r6326, r6332;
}
{
add.f16x2 r6338, r5842, r5858;
}
{
mul.f16x2 r6341, r6338, r6286;
}
{
add.f16x2 r6344, r4301, r6341;
}
{
sub.f16x2 r6347, r5836, r5852;
}
{
mul.f16x2 r6350, r6347, r6288;
}
{
sub.f16x2 r6353, r6344, r6350;
}
{
add.f16x2 r6356, r5842, r5858;
}
{
mul.f16x2 r6359, r6356, r6286;
}
{
add.f16x2 r6362, r4301, r6359;
}
{
sub.f16x2 r6365, r5836, r5852;
}
{
mul.f16x2 r6368, r6365, r6288;
}
{
add.f16x2 r6371, r6362, r6368;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6374, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6375, {low, high};
}
{
neg.f16x2 r6376, r6375;
}
{
add.f16x2 r6378, r5868, r5884;
}
{
add.f16x2 r6381, r4353, r6378;
}
{
add.f16x2 r6384, r5874, r5890;
}
{
add.f16x2 r6387, r4389, r6384;
}
{
add.f16x2 r6390, r5868, r5884;
}
{
mul.f16x2 r6393, r6390, r6374;
}
{
add.f16x2 r6396, r4353, r6393;
}
{
sub.f16x2 r6399, r5874, r5890;
}
{
mul.f16x2 r6402, r6399, r6376;
}
{
add.f16x2 r6405, r6396, r6402;
}
{
add.f16x2 r6408, r5868, r5884;
}
{
mul.f16x2 r6411, r6408, r6374;
}
{
add.f16x2 r6414, r4353, r6411;
}
{
sub.f16x2 r6417, r5874, r5890;
}
{
mul.f16x2 r6420, r6417, r6376;
}
{
sub.f16x2 r6423, r6414, r6420;
}
{
add.f16x2 r6426, r5874, r5890;
}
{
mul.f16x2 r6429, r6426, r6374;
}
{
add.f16x2 r6432, r4389, r6429;
}
{
sub.f16x2 r6435, r5868, r5884;
}
{
mul.f16x2 r6438, r6435, r6376;
}
{
sub.f16x2 r6441, r6432, r6438;
}
{
add.f16x2 r6444, r5874, r5890;
}
{
mul.f16x2 r6447, r6444, r6374;
}
{
add.f16x2 r6450, r4389, r6447;
}
{
sub.f16x2 r6453, r5868, r5884;
}
{
mul.f16x2 r6456, r6453, r6376;
}
{
add.f16x2 r6459, r6450, r6456;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6462, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6463, {low, high};
}
{
neg.f16x2 r6464, r6463;
}
{
add.f16x2 r6466, r5900, r5916;
}
{
add.f16x2 r6469, r4441, r6466;
}
{
add.f16x2 r6472, r5906, r5922;
}
{
add.f16x2 r6475, r4477, r6472;
}
{
add.f16x2 r6478, r5900, r5916;
}
{
mul.f16x2 r6481, r6478, r6462;
}
{
add.f16x2 r6484, r4441, r6481;
}
{
sub.f16x2 r6487, r5906, r5922;
}
{
mul.f16x2 r6490, r6487, r6464;
}
{
add.f16x2 r6493, r6484, r6490;
}
{
add.f16x2 r6496, r5900, r5916;
}
{
mul.f16x2 r6499, r6496, r6462;
}
{
add.f16x2 r6502, r4441, r6499;
}
{
sub.f16x2 r6505, r5906, r5922;
}
{
mul.f16x2 r6508, r6505, r6464;
}
{
sub.f16x2 r6511, r6502, r6508;
}
{
add.f16x2 r6514, r5906, r5922;
}
{
mul.f16x2 r6517, r6514, r6462;
}
{
add.f16x2 r6520, r4477, r6517;
}
{
sub.f16x2 r6523, r5900, r5916;
}
{
mul.f16x2 r6526, r6523, r6464;
}
{
sub.f16x2 r6529, r6520, r6526;
}
{
add.f16x2 r6532, r5906, r5922;
}
{
mul.f16x2 r6535, r6532, r6462;
}
{
add.f16x2 r6538, r4477, r6535;
}
{
sub.f16x2 r6541, r5900, r5916;
}
{
mul.f16x2 r6544, r6541, r6464;
}
{
add.f16x2 r6547, r6538, r6544;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6550, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6551, {low, high};
}
{
neg.f16x2 r6552, r6551;
}
{
add.f16x2 r6554, r5932, r5948;
}
{
add.f16x2 r6557, r4283, r6554;
}
{
add.f16x2 r6560, r5938, r5954;
}
{
add.f16x2 r6563, r4319, r6560;
}
{
add.f16x2 r6566, r5932, r5948;
}
{
mul.f16x2 r6569, r6566, r6550;
}
{
add.f16x2 r6572, r4283, r6569;
}
{
sub.f16x2 r6575, r5938, r5954;
}
{
mul.f16x2 r6578, r6575, r6552;
}
{
add.f16x2 r6581, r6572, r6578;
}
{
add.f16x2 r6584, r5932, r5948;
}
{
mul.f16x2 r6587, r6584, r6550;
}
{
add.f16x2 r6590, r4283, r6587;
}
{
sub.f16x2 r6593, r5938, r5954;
}
{
mul.f16x2 r6596, r6593, r6552;
}
{
sub.f16x2 r6599, r6590, r6596;
}
{
add.f16x2 r6602, r5938, r5954;
}
{
mul.f16x2 r6605, r6602, r6550;
}
{
add.f16x2 r6608, r4319, r6605;
}
{
sub.f16x2 r6611, r5932, r5948;
}
{
mul.f16x2 r6614, r6611, r6552;
}
{
sub.f16x2 r6617, r6608, r6614;
}
{
add.f16x2 r6620, r5938, r5954;
}
{
mul.f16x2 r6623, r6620, r6550;
}
{
add.f16x2 r6626, r4319, r6623;
}
{
sub.f16x2 r6629, r5932, r5948;
}
{
mul.f16x2 r6632, r6629, r6552;
}
{
add.f16x2 r6635, r6626, r6632;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6638, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6639, {low, high};
}
{
neg.f16x2 r6640, r6639;
}
{
add.f16x2 r6642, r5964, r5980;
}
{
add.f16x2 r6645, r4371, r6642;
}
{
add.f16x2 r6648, r5970, r5986;
}
{
add.f16x2 r6651, r4407, r6648;
}
{
add.f16x2 r6654, r5964, r5980;
}
{
mul.f16x2 r6657, r6654, r6638;
}
{
add.f16x2 r6660, r4371, r6657;
}
{
sub.f16x2 r6663, r5970, r5986;
}
{
mul.f16x2 r6666, r6663, r6640;
}
{
add.f16x2 r6669, r6660, r6666;
}
{
add.f16x2 r6672, r5964, r5980;
}
{
mul.f16x2 r6675, r6672, r6638;
}
{
add.f16x2 r6678, r4371, r6675;
}
{
sub.f16x2 r6681, r5970, r5986;
}
{
mul.f16x2 r6684, r6681, r6640;
}
{
sub.f16x2 r6687, r6678, r6684;
}
{
add.f16x2 r6690, r5970, r5986;
}
{
mul.f16x2 r6693, r6690, r6638;
}
{
add.f16x2 r6696, r4407, r6693;
}
{
sub.f16x2 r6699, r5964, r5980;
}
{
mul.f16x2 r6702, r6699, r6640;
}
{
sub.f16x2 r6705, r6696, r6702;
}
{
add.f16x2 r6708, r5970, r5986;
}
{
mul.f16x2 r6711, r6708, r6638;
}
{
add.f16x2 r6714, r4407, r6711;
}
{
sub.f16x2 r6717, r5964, r5980;
}
{
mul.f16x2 r6720, r6717, r6640;
}
{
add.f16x2 r6723, r6714, r6720;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r6726, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r6727, {low, high};
}
{
neg.f16x2 r6728, r6727;
}
{
add.f16x2 r6730, r5996, r6012;
}
{
add.f16x2 r6733, r4459, r6730;
}
{
add.f16x2 r6736, r6002, r6018;
}
{
add.f16x2 r6739, r4495, r6736;
}
{
add.f16x2 r6742, r5996, r6012;
}
{
mul.f16x2 r6745, r6742, r6726;
}
{
add.f16x2 r6748, r4459, r6745;
}
{
sub.f16x2 r6751, r6002, r6018;
}
{
mul.f16x2 r6754, r6751, r6728;
}
{
add.f16x2 r6757, r6748, r6754;
}
{
add.f16x2 r6760, r5996, r6012;
}
{
mul.f16x2 r6763, r6760, r6726;
}
{
add.f16x2 r6766, r4459, r6763;
}
{
sub.f16x2 r6769, r6002, r6018;
}
{
mul.f16x2 r6772, r6769, r6728;
}
{
sub.f16x2 r6775, r6766, r6772;
}
{
add.f16x2 r6778, r6002, r6018;
}
{
mul.f16x2 r6781, r6778, r6726;
}
{
add.f16x2 r6784, r4495, r6781;
}
{
sub.f16x2 r6787, r5996, r6012;
}
{
mul.f16x2 r6790, r6787, r6728;
}
{
sub.f16x2 r6793, r6784, r6790;
}
{
add.f16x2 r6796, r6002, r6018;
}
{
mul.f16x2 r6799, r6796, r6726;
}
{
add.f16x2 r6802, r4495, r6799;
}
{
sub.f16x2 r6805, r5996, r6012;
}
{
mul.f16x2 r6808, r6805, r6728;
}
{
add.f16x2 r6811, r6802, r6808;
}
mul.wide.u32 rd4, r8577, 795364315;
shr.u64 rd5, rd4, 32;
cvt.u32.u64 r8581, rd5;
sub.s32 r8582, r8577, r8581;
shr.u32 r8583, r8582, 1;
add.s32 r8584, r8583, r8581;
shr.u32 r8585, r8584, 4;
mul.lo.s32 r8586, r8585, 27;
sub.s32 r8587, r8577, r8586;
shl.b32 r8588, r8587, 2;
add.s32 r8589, r8578, r8588;
cvt.rn.f32.u32 f768, r8585;
mul.f32 f769, f768, 0f3D9EDD1F;
cos.approx.f32 f673, f769;
sin.approx.f32 f770, f769;
neg.f32 f674, f770;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f673;
cvt.rn.f16.f32 high, f674;
mov.b32 r6814, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6817, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6819, {high, high};
}
{
mul.f16x2 r6821, r6123, r6819;
}
{
neg.f16x2 r6824, r6821;
}
{
fma.rn.f16x2 r6826, r6117, r6817, r6824;
}
{
mul.f16x2 r6830, r6117, r6819;
}
{
fma.rn.f16x2 r6833, r6123, r6817, r6830;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6837, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6839, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6841, {low, high};
}
{
mul.f16x2 r6842, r6839, r6841;
}
{
mul.f16x2 r6845, r6814, r6837;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6848, {high, low};
}
{
fma.rn.f16x2 r6850, r6842, r6848, r6845;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6850;
mov.b32 r6854, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6850;
mov.b32 r6856, {high, high};
}
{
mul.f16x2 r6858, r6211, r6856;
}
{
neg.f16x2 r6861, r6858;
}
{
fma.rn.f16x2 r6863, r6205, r6854, r6861;
}
{
mul.f16x2 r6867, r6205, r6856;
}
{
fma.rn.f16x2 r6870, r6211, r6854, r6867;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6874, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6876, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6878, {low, high};
}
{
mul.f16x2 r6879, r6876, r6878;
}
{
mul.f16x2 r6882, r6850, r6874;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6850;
mov.b32 r6885, {high, low};
}
{
fma.rn.f16x2 r6887, r6879, r6885, r6882;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6887;
mov.b32 r6891, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6887;
mov.b32 r6893, {high, high};
}
{
mul.f16x2 r6895, r6299, r6893;
}
{
neg.f16x2 r6898, r6895;
}
{
fma.rn.f16x2 r6900, r6293, r6891, r6898;
}
{
mul.f16x2 r6904, r6293, r6893;
}
{
fma.rn.f16x2 r6907, r6299, r6891, r6904;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6911, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6913, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6915, {low, high};
}
{
mul.f16x2 r6916, r6913, r6915;
}
{
mul.f16x2 r6919, r6887, r6911;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6887;
mov.b32 r6922, {high, low};
}
{
fma.rn.f16x2 r6924, r6916, r6922, r6919;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6924;
mov.b32 r6928, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6924;
mov.b32 r6930, {high, high};
}
{
mul.f16x2 r6932, r6387, r6930;
}
{
neg.f16x2 r6935, r6932;
}
{
fma.rn.f16x2 r6937, r6381, r6928, r6935;
}
{
mul.f16x2 r6941, r6381, r6930;
}
{
fma.rn.f16x2 r6944, r6387, r6928, r6941;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6948, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6950, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6952, {low, high};
}
{
mul.f16x2 r6953, r6950, r6952;
}
{
mul.f16x2 r6956, r6924, r6948;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6924;
mov.b32 r6959, {high, low};
}
{
fma.rn.f16x2 r6961, r6953, r6959, r6956;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6961;
mov.b32 r6965, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6961;
mov.b32 r6967, {high, high};
}
{
mul.f16x2 r6969, r6475, r6967;
}
{
neg.f16x2 r6972, r6969;
}
{
fma.rn.f16x2 r6974, r6469, r6965, r6972;
}
{
mul.f16x2 r6978, r6469, r6967;
}
{
fma.rn.f16x2 r6981, r6475, r6965, r6978;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6985, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r6987, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r6989, {low, high};
}
{
mul.f16x2 r6990, r6987, r6989;
}
{
mul.f16x2 r6993, r6961, r6985;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6961;
mov.b32 r6996, {high, low};
}
{
fma.rn.f16x2 r6998, r6990, r6996, r6993;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6998;
mov.b32 r7002, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6998;
mov.b32 r7004, {high, high};
}
{
mul.f16x2 r7006, r6563, r7004;
}
{
neg.f16x2 r7009, r7006;
}
{
fma.rn.f16x2 r7011, r6557, r7002, r7009;
}
{
mul.f16x2 r7015, r6557, r7004;
}
{
fma.rn.f16x2 r7018, r6563, r7002, r7015;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7022, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7024, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7026, {low, high};
}
{
mul.f16x2 r7027, r7024, r7026;
}
{
mul.f16x2 r7030, r6998, r7022;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6998;
mov.b32 r7033, {high, low};
}
{
fma.rn.f16x2 r7035, r7027, r7033, r7030;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7035;
mov.b32 r7039, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7035;
mov.b32 r7041, {high, high};
}
{
mul.f16x2 r7043, r6651, r7041;
}
{
neg.f16x2 r7046, r7043;
}
{
fma.rn.f16x2 r7048, r6645, r7039, r7046;
}
{
mul.f16x2 r7052, r6645, r7041;
}
{
fma.rn.f16x2 r7055, r6651, r7039, r7052;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7059, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7061, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7063, {low, high};
}
{
mul.f16x2 r7064, r7061, r7063;
}
{
mul.f16x2 r7067, r7035, r7059;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7035;
mov.b32 r7070, {high, low};
}
{
fma.rn.f16x2 r7072, r7064, r7070, r7067;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7072;
mov.b32 r7076, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7072;
mov.b32 r7078, {high, high};
}
{
mul.f16x2 r7080, r6739, r7078;
}
{
neg.f16x2 r7083, r7080;
}
{
fma.rn.f16x2 r7085, r6733, r7076, r7083;
}
{
mul.f16x2 r7089, r6733, r7078;
}
{
fma.rn.f16x2 r7092, r6739, r7076, r7089;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7096, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7098, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7100, {low, high};
}
{
mul.f16x2 r7101, r7098, r7100;
}
{
mul.f16x2 r7104, r7072, r7096;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7072;
mov.b32 r7107, {high, low};
}
{
fma.rn.f16x2 r7109, r7101, r7107, r7104;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7109;
mov.b32 r7113, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7109;
mov.b32 r7115, {high, high};
}
{
mul.f16x2 r7117, r6089, r7115;
}
{
neg.f16x2 r7120, r7117;
}
{
fma.rn.f16x2 r7122, r6053, r7113, r7120;
}
{
mul.f16x2 r7126, r6053, r7115;
}
{
fma.rn.f16x2 r7129, r6089, r7113, r7126;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7133, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7135, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7137, {low, high};
}
{
mul.f16x2 r7138, r7135, r7137;
}
{
mul.f16x2 r7141, r7109, r7133;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7109;
mov.b32 r7144, {high, low};
}
{
fma.rn.f16x2 r7146, r7138, r7144, r7141;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7146;
mov.b32 r7150, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7146;
mov.b32 r7152, {high, high};
}
{
mul.f16x2 r7154, r6177, r7152;
}
{
neg.f16x2 r7157, r7154;
}
{
fma.rn.f16x2 r7159, r6141, r7150, r7157;
}
{
mul.f16x2 r7163, r6141, r7152;
}
{
fma.rn.f16x2 r7166, r6177, r7150, r7163;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7170, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7172, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7174, {low, high};
}
{
mul.f16x2 r7175, r7172, r7174;
}
{
mul.f16x2 r7178, r7146, r7170;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7146;
mov.b32 r7181, {high, low};
}
{
fma.rn.f16x2 r7183, r7175, r7181, r7178;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7183;
mov.b32 r7187, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7183;
mov.b32 r7189, {high, high};
}
{
mul.f16x2 r7191, r6265, r7189;
}
{
neg.f16x2 r7194, r7191;
}
{
fma.rn.f16x2 r7196, r6229, r7187, r7194;
}
{
mul.f16x2 r7200, r6229, r7189;
}
{
fma.rn.f16x2 r7203, r6265, r7187, r7200;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7207, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7209, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7211, {low, high};
}
{
mul.f16x2 r7212, r7209, r7211;
}
{
mul.f16x2 r7215, r7183, r7207;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7183;
mov.b32 r7218, {high, low};
}
{
fma.rn.f16x2 r7220, r7212, r7218, r7215;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7220;
mov.b32 r7224, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7220;
mov.b32 r7226, {high, high};
}
{
mul.f16x2 r7228, r6353, r7226;
}
{
neg.f16x2 r7231, r7228;
}
{
fma.rn.f16x2 r7233, r6317, r7224, r7231;
}
{
mul.f16x2 r7237, r6317, r7226;
}
{
fma.rn.f16x2 r7240, r6353, r7224, r7237;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7244, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7246, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7248, {low, high};
}
{
mul.f16x2 r7249, r7246, r7248;
}
{
mul.f16x2 r7252, r7220, r7244;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7220;
mov.b32 r7255, {high, low};
}
{
fma.rn.f16x2 r7257, r7249, r7255, r7252;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7257;
mov.b32 r7261, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7257;
mov.b32 r7263, {high, high};
}
{
mul.f16x2 r7265, r6441, r7263;
}
{
neg.f16x2 r7268, r7265;
}
{
fma.rn.f16x2 r7270, r6405, r7261, r7268;
}
{
mul.f16x2 r7274, r6405, r7263;
}
{
fma.rn.f16x2 r7277, r6441, r7261, r7274;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7281, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7283, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7285, {low, high};
}
{
mul.f16x2 r7286, r7283, r7285;
}
{
mul.f16x2 r7289, r7257, r7281;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7257;
mov.b32 r7292, {high, low};
}
{
fma.rn.f16x2 r7294, r7286, r7292, r7289;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7294;
mov.b32 r7298, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7294;
mov.b32 r7300, {high, high};
}
{
mul.f16x2 r7302, r6529, r7300;
}
{
neg.f16x2 r7305, r7302;
}
{
fma.rn.f16x2 r7307, r6493, r7298, r7305;
}
{
mul.f16x2 r7311, r6493, r7300;
}
{
fma.rn.f16x2 r7314, r6529, r7298, r7311;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7318, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7320, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7322, {low, high};
}
{
mul.f16x2 r7323, r7320, r7322;
}
{
mul.f16x2 r7326, r7294, r7318;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7294;
mov.b32 r7329, {high, low};
}
{
fma.rn.f16x2 r7331, r7323, r7329, r7326;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7331;
mov.b32 r7335, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7331;
mov.b32 r7337, {high, high};
}
{
mul.f16x2 r7339, r6617, r7337;
}
{
neg.f16x2 r7342, r7339;
}
{
fma.rn.f16x2 r7344, r6581, r7335, r7342;
}
{
mul.f16x2 r7348, r6581, r7337;
}
{
fma.rn.f16x2 r7351, r6617, r7335, r7348;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7355, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7357, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7359, {low, high};
}
{
mul.f16x2 r7360, r7357, r7359;
}
{
mul.f16x2 r7363, r7331, r7355;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7331;
mov.b32 r7366, {high, low};
}
{
fma.rn.f16x2 r7368, r7360, r7366, r7363;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7368;
mov.b32 r7372, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7368;
mov.b32 r7374, {high, high};
}
{
mul.f16x2 r7376, r6705, r7374;
}
{
neg.f16x2 r7379, r7376;
}
{
fma.rn.f16x2 r7381, r6669, r7372, r7379;
}
{
mul.f16x2 r7385, r6669, r7374;
}
{
fma.rn.f16x2 r7388, r6705, r7372, r7385;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7392, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7394, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7396, {low, high};
}
{
mul.f16x2 r7397, r7394, r7396;
}
{
mul.f16x2 r7400, r7368, r7392;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7368;
mov.b32 r7403, {high, low};
}
{
fma.rn.f16x2 r7405, r7397, r7403, r7400;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7405;
mov.b32 r7409, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7405;
mov.b32 r7411, {high, high};
}
{
mul.f16x2 r7413, r6793, r7411;
}
{
neg.f16x2 r7416, r7413;
}
{
fma.rn.f16x2 r7418, r6757, r7409, r7416;
}
{
mul.f16x2 r7422, r6757, r7411;
}
{
fma.rn.f16x2 r7425, r6793, r7409, r7422;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7429, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7431, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7433, {low, high};
}
{
mul.f16x2 r7434, r7431, r7433;
}
{
mul.f16x2 r7437, r7405, r7429;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7405;
mov.b32 r7440, {high, low};
}
{
fma.rn.f16x2 r7442, r7434, r7440, r7437;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7442;
mov.b32 r7446, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7442;
mov.b32 r7448, {high, high};
}
{
mul.f16x2 r7450, r6107, r7448;
}
{
neg.f16x2 r7453, r7450;
}
{
fma.rn.f16x2 r7455, r6071, r7446, r7453;
}
{
mul.f16x2 r7459, r6071, r7448;
}
{
fma.rn.f16x2 r7462, r6107, r7446, r7459;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7466, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7468, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7470, {low, high};
}
{
mul.f16x2 r7471, r7468, r7470;
}
{
mul.f16x2 r7474, r7442, r7466;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7442;
mov.b32 r7477, {high, low};
}
{
fma.rn.f16x2 r7479, r7471, r7477, r7474;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7479;
mov.b32 r7483, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7479;
mov.b32 r7485, {high, high};
}
{
mul.f16x2 r7487, r6195, r7485;
}
{
neg.f16x2 r7490, r7487;
}
{
fma.rn.f16x2 r7492, r6159, r7483, r7490;
}
{
mul.f16x2 r7496, r6159, r7485;
}
{
fma.rn.f16x2 r7499, r6195, r7483, r7496;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7503, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7505, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7507, {low, high};
}
{
mul.f16x2 r7508, r7505, r7507;
}
{
mul.f16x2 r7511, r7479, r7503;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7479;
mov.b32 r7514, {high, low};
}
{
fma.rn.f16x2 r7516, r7508, r7514, r7511;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7516;
mov.b32 r7520, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7516;
mov.b32 r7522, {high, high};
}
{
mul.f16x2 r7524, r6283, r7522;
}
{
neg.f16x2 r7527, r7524;
}
{
fma.rn.f16x2 r7529, r6247, r7520, r7527;
}
{
mul.f16x2 r7533, r6247, r7522;
}
{
fma.rn.f16x2 r7536, r6283, r7520, r7533;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7540, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7542, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7544, {low, high};
}
{
mul.f16x2 r7545, r7542, r7544;
}
{
mul.f16x2 r7548, r7516, r7540;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7516;
mov.b32 r7551, {high, low};
}
{
fma.rn.f16x2 r7553, r7545, r7551, r7548;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7553;
mov.b32 r7557, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7553;
mov.b32 r7559, {high, high};
}
{
mul.f16x2 r7561, r6371, r7559;
}
{
neg.f16x2 r7564, r7561;
}
{
fma.rn.f16x2 r7566, r6335, r7557, r7564;
}
{
mul.f16x2 r7570, r6335, r7559;
}
{
fma.rn.f16x2 r7573, r6371, r7557, r7570;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7577, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7579, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7581, {low, high};
}
{
mul.f16x2 r7582, r7579, r7581;
}
{
mul.f16x2 r7585, r7553, r7577;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7553;
mov.b32 r7588, {high, low};
}
{
fma.rn.f16x2 r7590, r7582, r7588, r7585;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7590;
mov.b32 r7594, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7590;
mov.b32 r7596, {high, high};
}
{
mul.f16x2 r7598, r6459, r7596;
}
{
neg.f16x2 r7601, r7598;
}
{
fma.rn.f16x2 r7603, r6423, r7594, r7601;
}
{
mul.f16x2 r7607, r6423, r7596;
}
{
fma.rn.f16x2 r7610, r6459, r7594, r7607;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7614, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7616, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7618, {low, high};
}
{
mul.f16x2 r7619, r7616, r7618;
}
{
mul.f16x2 r7622, r7590, r7614;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7590;
mov.b32 r7625, {high, low};
}
{
fma.rn.f16x2 r7627, r7619, r7625, r7622;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7627;
mov.b32 r7631, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7627;
mov.b32 r7633, {high, high};
}
{
mul.f16x2 r7635, r6547, r7633;
}
{
neg.f16x2 r7638, r7635;
}
{
fma.rn.f16x2 r7640, r6511, r7631, r7638;
}
{
mul.f16x2 r7644, r6511, r7633;
}
{
fma.rn.f16x2 r7647, r6547, r7631, r7644;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7651, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7653, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7655, {low, high};
}
{
mul.f16x2 r7656, r7653, r7655;
}
{
mul.f16x2 r7659, r7627, r7651;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7627;
mov.b32 r7662, {high, low};
}
{
fma.rn.f16x2 r7664, r7656, r7662, r7659;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7664;
mov.b32 r7668, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7664;
mov.b32 r7670, {high, high};
}
{
mul.f16x2 r7672, r6635, r7670;
}
{
neg.f16x2 r7675, r7672;
}
{
fma.rn.f16x2 r7677, r6599, r7668, r7675;
}
{
mul.f16x2 r7681, r6599, r7670;
}
{
fma.rn.f16x2 r7684, r6635, r7668, r7681;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7688, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7690, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7692, {low, high};
}
{
mul.f16x2 r7693, r7690, r7692;
}
{
mul.f16x2 r7696, r7664, r7688;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7664;
mov.b32 r7699, {high, low};
}
{
fma.rn.f16x2 r7701, r7693, r7699, r7696;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7701;
mov.b32 r7705, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7701;
mov.b32 r7707, {high, high};
}
{
mul.f16x2 r7709, r6723, r7707;
}
{
neg.f16x2 r7712, r7709;
}
{
fma.rn.f16x2 r7714, r6687, r7705, r7712;
}
{
mul.f16x2 r7718, r6687, r7707;
}
{
fma.rn.f16x2 r7721, r6723, r7705, r7718;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7725, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r6814;
mov.b32 r7727, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f725;
cvt.rn.f16.f32 high, f726;
mov.b32 r7729, {low, high};
}
{
mul.f16x2 r7730, r7727, r7729;
}
{
mul.f16x2 r7733, r7701, r7725;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7701;
mov.b32 r7736, {high, low};
}
{
fma.rn.f16x2 r7738, r7730, r7736, r7733;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7738;
mov.b32 r7742, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7738;
mov.b32 r7744, {high, high};
}
{
mul.f16x2 r7746, r6811, r7744;
}
{
neg.f16x2 r7749, r7746;
}
{
fma.rn.f16x2 r7751, r6775, r7742, r7749;
}
{
mul.f16x2 r7755, r6775, r7744;
}
{
fma.rn.f16x2 r7758, r6811, r7742, r7755;
}
barrier.sync 0;
mad.lo.s32 r8590, r8585, 2916, r8589;
st.shared.u32 [r8590], r6029;
st.shared.u32 [r8590+108], r6826;
st.shared.u32 [r8590+216], r6863;
st.shared.u32 [r8590+324], r6900;
st.shared.u32 [r8590+432], r6937;
st.shared.u32 [r8590+540], r6974;
st.shared.u32 [r8590+648], r7011;
st.shared.u32 [r8590+756], r7048;
st.shared.u32 [r8590+864], r7085;
st.shared.u32 [r8590+972], r7122;
st.shared.u32 [r8590+1080], r7159;
st.shared.u32 [r8590+1188], r7196;
st.shared.u32 [r8590+1296], r7233;
st.shared.u32 [r8590+1404], r7270;
st.shared.u32 [r8590+1512], r7307;
st.shared.u32 [r8590+1620], r7344;
st.shared.u32 [r8590+1728], r7381;
st.shared.u32 [r8590+1836], r7418;
st.shared.u32 [r8590+1944], r7455;
st.shared.u32 [r8590+2052], r7492;
st.shared.u32 [r8590+2160], r7529;
st.shared.u32 [r8590+2268], r7566;
st.shared.u32 [r8590+2376], r7603;
st.shared.u32 [r8590+2484], r7640;
st.shared.u32 [r8590+2592], r7677;
st.shared.u32 [r8590+2700], r7714;
st.shared.u32 [r8590+2808], r7751;
barrier.sync 0;
ld.shared.u32 r7787, [r8580];
ld.shared.u32 r7875, [r8580+324];
ld.shared.u32 r7963, [r8580+648];
ld.shared.u32 r8051, [r8580+972];
ld.shared.u32 r8139, [r8580+1296];
ld.shared.u32 r8227, [r8580+1620];
ld.shared.u32 r8315, [r8580+1944];
ld.shared.u32 r8403, [r8580+2268];
ld.shared.u32 r8491, [r8580+2592];
ld.shared.u32 r7784, [r8580+2916];
ld.shared.u32 r7872, [r8580+3240];
ld.shared.u32 r7960, [r8580+3564];
ld.shared.u32 r8048, [r8580+3888];
ld.shared.u32 r8136, [r8580+4212];
ld.shared.u32 r8224, [r8580+4536];
ld.shared.u32 r8312, [r8580+4860];
ld.shared.u32 r8400, [r8580+5184];
ld.shared.u32 r8488, [r8580+5508];
ld.shared.u32 r7785, [r8580+5832];
ld.shared.u32 r7873, [r8580+6156];
ld.shared.u32 r7961, [r8580+6480];
ld.shared.u32 r8049, [r8580+6804];
ld.shared.u32 r8137, [r8580+7128];
ld.shared.u32 r8225, [r8580+7452];
ld.shared.u32 r8313, [r8580+7776];
ld.shared.u32 r8401, [r8580+8100];
ld.shared.u32 r8489, [r8580+8424];
barrier.sync 0;
st.shared.u32 [r8590], r6035;
st.shared.u32 [r8590+108], r6833;
st.shared.u32 [r8590+216], r6870;
st.shared.u32 [r8590+324], r6907;
st.shared.u32 [r8590+432], r6944;
st.shared.u32 [r8590+540], r6981;
st.shared.u32 [r8590+648], r7018;
st.shared.u32 [r8590+756], r7055;
st.shared.u32 [r8590+864], r7092;
st.shared.u32 [r8590+972], r7129;
st.shared.u32 [r8590+1080], r7166;
st.shared.u32 [r8590+1188], r7203;
st.shared.u32 [r8590+1296], r7240;
st.shared.u32 [r8590+1404], r7277;
st.shared.u32 [r8590+1512], r7314;
st.shared.u32 [r8590+1620], r7351;
st.shared.u32 [r8590+1728], r7388;
st.shared.u32 [r8590+1836], r7425;
st.shared.u32 [r8590+1944], r7462;
st.shared.u32 [r8590+2052], r7499;
st.shared.u32 [r8590+2160], r7536;
st.shared.u32 [r8590+2268], r7573;
st.shared.u32 [r8590+2376], r7610;
st.shared.u32 [r8590+2484], r7647;
st.shared.u32 [r8590+2592], r7684;
st.shared.u32 [r8590+2700], r7721;
st.shared.u32 [r8590+2808], r7758;
barrier.sync 0;
ld.shared.u32 r7793, [r8580];
ld.shared.u32 r7881, [r8580+324];
ld.shared.u32 r7969, [r8580+648];
ld.shared.u32 r8057, [r8580+972];
ld.shared.u32 r8145, [r8580+1296];
ld.shared.u32 r8233, [r8580+1620];
ld.shared.u32 r8321, [r8580+1944];
ld.shared.u32 r8409, [r8580+2268];
ld.shared.u32 r8497, [r8580+2592];
ld.shared.u32 r7790, [r8580+2916];
ld.shared.u32 r7878, [r8580+3240];
ld.shared.u32 r7966, [r8580+3564];
ld.shared.u32 r8054, [r8580+3888];
ld.shared.u32 r8142, [r8580+4212];
ld.shared.u32 r8230, [r8580+4536];
ld.shared.u32 r8318, [r8580+4860];
ld.shared.u32 r8406, [r8580+5184];
ld.shared.u32 r8494, [r8580+5508];
ld.shared.u32 r7791, [r8580+5832];
ld.shared.u32 r7879, [r8580+6156];
ld.shared.u32 r7967, [r8580+6480];
ld.shared.u32 r8055, [r8580+6804];
ld.shared.u32 r8143, [r8580+7128];
ld.shared.u32 r8231, [r8580+7452];
ld.shared.u32 r8319, [r8580+7776];
ld.shared.u32 r8407, [r8580+8100];
ld.shared.u32 r8495, [r8580+8424];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r7779, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r7780, {low, high};
}
{
neg.f16x2 r7781, r7780;
}
{
add.f16x2 r7783, r7784, r7785;
}
{
add.f16x2 %0, r7787, r7783;
}
{
add.f16x2 r7789, r7790, r7791;
}
{
add.f16x2 %1, r7793, r7789;
}
{
add.f16x2 r7795, r7784, r7785;
}
{
mul.f16x2 r7798, r7795, r7779;
}
{
add.f16x2 r7801, r7787, r7798;
}
{
sub.f16x2 r7804, r7790, r7791;
}
{
mul.f16x2 r7807, r7804, r7781;
}
{
add.f16x2 %18, r7801, r7807;
}
{
add.f16x2 r7813, r7784, r7785;
}
{
mul.f16x2 r7816, r7813, r7779;
}
{
add.f16x2 r7819, r7787, r7816;
}
{
sub.f16x2 r7822, r7790, r7791;
}
{
mul.f16x2 r7825, r7822, r7781;
}
{
sub.f16x2 %36, r7819, r7825;
}
{
add.f16x2 r7831, r7790, r7791;
}
{
mul.f16x2 r7834, r7831, r7779;
}
{
add.f16x2 r7837, r7793, r7834;
}
{
sub.f16x2 r7840, r7784, r7785;
}
{
mul.f16x2 r7843, r7840, r7781;
}
{
sub.f16x2 %19, r7837, r7843;
}
{
add.f16x2 r7849, r7790, r7791;
}
{
mul.f16x2 r7852, r7849, r7779;
}
{
add.f16x2 r7855, r7793, r7852;
}
{
sub.f16x2 r7858, r7784, r7785;
}
{
mul.f16x2 r7861, r7858, r7781;
}
{
add.f16x2 %37, r7855, r7861;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r7867, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r7868, {low, high};
}
{
neg.f16x2 r7869, r7868;
}
{
add.f16x2 r7871, r7872, r7873;
}
{
add.f16x2 %2, r7875, r7871;
}
{
add.f16x2 r7877, r7878, r7879;
}
{
add.f16x2 %3, r7881, r7877;
}
{
add.f16x2 r7883, r7872, r7873;
}
{
mul.f16x2 r7886, r7883, r7867;
}
{
add.f16x2 r7889, r7875, r7886;
}
{
sub.f16x2 r7892, r7878, r7879;
}
{
mul.f16x2 r7895, r7892, r7869;
}
{
add.f16x2 %20, r7889, r7895;
}
{
add.f16x2 r7901, r7872, r7873;
}
{
mul.f16x2 r7904, r7901, r7867;
}
{
add.f16x2 r7907, r7875, r7904;
}
{
sub.f16x2 r7910, r7878, r7879;
}
{
mul.f16x2 r7913, r7910, r7869;
}
{
sub.f16x2 %38, r7907, r7913;
}
{
add.f16x2 r7919, r7878, r7879;
}
{
mul.f16x2 r7922, r7919, r7867;
}
{
add.f16x2 r7925, r7881, r7922;
}
{
sub.f16x2 r7928, r7872, r7873;
}
{
mul.f16x2 r7931, r7928, r7869;
}
{
sub.f16x2 %21, r7925, r7931;
}
{
add.f16x2 r7937, r7878, r7879;
}
{
mul.f16x2 r7940, r7937, r7867;
}
{
add.f16x2 r7943, r7881, r7940;
}
{
sub.f16x2 r7946, r7872, r7873;
}
{
mul.f16x2 r7949, r7946, r7869;
}
{
add.f16x2 %39, r7943, r7949;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r7955, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r7956, {low, high};
}
{
neg.f16x2 r7957, r7956;
}
{
add.f16x2 r7959, r7960, r7961;
}
{
add.f16x2 %4, r7963, r7959;
}
{
add.f16x2 r7965, r7966, r7967;
}
{
add.f16x2 %5, r7969, r7965;
}
{
add.f16x2 r7971, r7960, r7961;
}
{
mul.f16x2 r7974, r7971, r7955;
}
{
add.f16x2 r7977, r7963, r7974;
}
{
sub.f16x2 r7980, r7966, r7967;
}
{
mul.f16x2 r7983, r7980, r7957;
}
{
add.f16x2 %22, r7977, r7983;
}
{
add.f16x2 r7989, r7960, r7961;
}
{
mul.f16x2 r7992, r7989, r7955;
}
{
add.f16x2 r7995, r7963, r7992;
}
{
sub.f16x2 r7998, r7966, r7967;
}
{
mul.f16x2 r8001, r7998, r7957;
}
{
sub.f16x2 %40, r7995, r8001;
}
{
add.f16x2 r8007, r7966, r7967;
}
{
mul.f16x2 r8010, r8007, r7955;
}
{
add.f16x2 r8013, r7969, r8010;
}
{
sub.f16x2 r8016, r7960, r7961;
}
{
mul.f16x2 r8019, r8016, r7957;
}
{
sub.f16x2 %23, r8013, r8019;
}
{
add.f16x2 r8025, r7966, r7967;
}
{
mul.f16x2 r8028, r8025, r7955;
}
{
add.f16x2 r8031, r7969, r8028;
}
{
sub.f16x2 r8034, r7960, r7961;
}
{
mul.f16x2 r8037, r8034, r7957;
}
{
add.f16x2 %41, r8031, r8037;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8043, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8044, {low, high};
}
{
neg.f16x2 r8045, r8044;
}
{
add.f16x2 r8047, r8048, r8049;
}
{
add.f16x2 %6, r8051, r8047;
}
{
add.f16x2 r8053, r8054, r8055;
}
{
add.f16x2 %7, r8057, r8053;
}
{
add.f16x2 r8059, r8048, r8049;
}
{
mul.f16x2 r8062, r8059, r8043;
}
{
add.f16x2 r8065, r8051, r8062;
}
{
sub.f16x2 r8068, r8054, r8055;
}
{
mul.f16x2 r8071, r8068, r8045;
}
{
add.f16x2 %24, r8065, r8071;
}
{
add.f16x2 r8077, r8048, r8049;
}
{
mul.f16x2 r8080, r8077, r8043;
}
{
add.f16x2 r8083, r8051, r8080;
}
{
sub.f16x2 r8086, r8054, r8055;
}
{
mul.f16x2 r8089, r8086, r8045;
}
{
sub.f16x2 %42, r8083, r8089;
}
{
add.f16x2 r8095, r8054, r8055;
}
{
mul.f16x2 r8098, r8095, r8043;
}
{
add.f16x2 r8101, r8057, r8098;
}
{
sub.f16x2 r8104, r8048, r8049;
}
{
mul.f16x2 r8107, r8104, r8045;
}
{
sub.f16x2 %25, r8101, r8107;
}
{
add.f16x2 r8113, r8054, r8055;
}
{
mul.f16x2 r8116, r8113, r8043;
}
{
add.f16x2 r8119, r8057, r8116;
}
{
sub.f16x2 r8122, r8048, r8049;
}
{
mul.f16x2 r8125, r8122, r8045;
}
{
add.f16x2 %43, r8119, r8125;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8131, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8132, {low, high};
}
{
neg.f16x2 r8133, r8132;
}
{
add.f16x2 r8135, r8136, r8137;
}
{
add.f16x2 %8, r8139, r8135;
}
{
add.f16x2 r8141, r8142, r8143;
}
{
add.f16x2 %9, r8145, r8141;
}
{
add.f16x2 r8147, r8136, r8137;
}
{
mul.f16x2 r8150, r8147, r8131;
}
{
add.f16x2 r8153, r8139, r8150;
}
{
sub.f16x2 r8156, r8142, r8143;
}
{
mul.f16x2 r8159, r8156, r8133;
}
{
add.f16x2 %26, r8153, r8159;
}
{
add.f16x2 r8165, r8136, r8137;
}
{
mul.f16x2 r8168, r8165, r8131;
}
{
add.f16x2 r8171, r8139, r8168;
}
{
sub.f16x2 r8174, r8142, r8143;
}
{
mul.f16x2 r8177, r8174, r8133;
}
{
sub.f16x2 %44, r8171, r8177;
}
{
add.f16x2 r8183, r8142, r8143;
}
{
mul.f16x2 r8186, r8183, r8131;
}
{
add.f16x2 r8189, r8145, r8186;
}
{
sub.f16x2 r8192, r8136, r8137;
}
{
mul.f16x2 r8195, r8192, r8133;
}
{
sub.f16x2 %27, r8189, r8195;
}
{
add.f16x2 r8201, r8142, r8143;
}
{
mul.f16x2 r8204, r8201, r8131;
}
{
add.f16x2 r8207, r8145, r8204;
}
{
sub.f16x2 r8210, r8136, r8137;
}
{
mul.f16x2 r8213, r8210, r8133;
}
{
add.f16x2 %45, r8207, r8213;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8219, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8220, {low, high};
}
{
neg.f16x2 r8221, r8220;
}
{
add.f16x2 r8223, r8224, r8225;
}
{
add.f16x2 %10, r8227, r8223;
}
{
add.f16x2 r8229, r8230, r8231;
}
{
add.f16x2 %11, r8233, r8229;
}
{
add.f16x2 r8235, r8224, r8225;
}
{
mul.f16x2 r8238, r8235, r8219;
}
{
add.f16x2 r8241, r8227, r8238;
}
{
sub.f16x2 r8244, r8230, r8231;
}
{
mul.f16x2 r8247, r8244, r8221;
}
{
add.f16x2 %28, r8241, r8247;
}
{
add.f16x2 r8253, r8224, r8225;
}
{
mul.f16x2 r8256, r8253, r8219;
}
{
add.f16x2 r8259, r8227, r8256;
}
{
sub.f16x2 r8262, r8230, r8231;
}
{
mul.f16x2 r8265, r8262, r8221;
}
{
sub.f16x2 %46, r8259, r8265;
}
{
add.f16x2 r8271, r8230, r8231;
}
{
mul.f16x2 r8274, r8271, r8219;
}
{
add.f16x2 r8277, r8233, r8274;
}
{
sub.f16x2 r8280, r8224, r8225;
}
{
mul.f16x2 r8283, r8280, r8221;
}
{
sub.f16x2 %29, r8277, r8283;
}
{
add.f16x2 r8289, r8230, r8231;
}
{
mul.f16x2 r8292, r8289, r8219;
}
{
add.f16x2 r8295, r8233, r8292;
}
{
sub.f16x2 r8298, r8224, r8225;
}
{
mul.f16x2 r8301, r8298, r8221;
}
{
add.f16x2 %47, r8295, r8301;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8307, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8308, {low, high};
}
{
neg.f16x2 r8309, r8308;
}
{
add.f16x2 r8311, r8312, r8313;
}
{
add.f16x2 %12, r8315, r8311;
}
{
add.f16x2 r8317, r8318, r8319;
}
{
add.f16x2 %13, r8321, r8317;
}
{
add.f16x2 r8323, r8312, r8313;
}
{
mul.f16x2 r8326, r8323, r8307;
}
{
add.f16x2 r8329, r8315, r8326;
}
{
sub.f16x2 r8332, r8318, r8319;
}
{
mul.f16x2 r8335, r8332, r8309;
}
{
add.f16x2 %30, r8329, r8335;
}
{
add.f16x2 r8341, r8312, r8313;
}
{
mul.f16x2 r8344, r8341, r8307;
}
{
add.f16x2 r8347, r8315, r8344;
}
{
sub.f16x2 r8350, r8318, r8319;
}
{
mul.f16x2 r8353, r8350, r8309;
}
{
sub.f16x2 %48, r8347, r8353;
}
{
add.f16x2 r8359, r8318, r8319;
}
{
mul.f16x2 r8362, r8359, r8307;
}
{
add.f16x2 r8365, r8321, r8362;
}
{
sub.f16x2 r8368, r8312, r8313;
}
{
mul.f16x2 r8371, r8368, r8309;
}
{
sub.f16x2 %31, r8365, r8371;
}
{
add.f16x2 r8377, r8318, r8319;
}
{
mul.f16x2 r8380, r8377, r8307;
}
{
add.f16x2 r8383, r8321, r8380;
}
{
sub.f16x2 r8386, r8312, r8313;
}
{
mul.f16x2 r8389, r8386, r8309;
}
{
add.f16x2 %49, r8383, r8389;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8396, {low, high};
}
{
neg.f16x2 r8397, r8396;
}
{
add.f16x2 r8399, r8400, r8401;
}
{
add.f16x2 %14, r8403, r8399;
}
{
add.f16x2 r8405, r8406, r8407;
}
{
add.f16x2 %15, r8409, r8405;
}
{
add.f16x2 r8411, r8400, r8401;
}
{
mul.f16x2 r8414, r8411, r8395;
}
{
add.f16x2 r8417, r8403, r8414;
}
{
sub.f16x2 r8420, r8406, r8407;
}
{
mul.f16x2 r8423, r8420, r8397;
}
{
add.f16x2 %32, r8417, r8423;
}
{
add.f16x2 r8429, r8400, r8401;
}
{
mul.f16x2 r8432, r8429, r8395;
}
{
add.f16x2 r8435, r8403, r8432;
}
{
sub.f16x2 r8438, r8406, r8407;
}
{
mul.f16x2 r8441, r8438, r8397;
}
{
sub.f16x2 %50, r8435, r8441;
}
{
add.f16x2 r8447, r8406, r8407;
}
{
mul.f16x2 r8450, r8447, r8395;
}
{
add.f16x2 r8453, r8409, r8450;
}
{
sub.f16x2 r8456, r8400, r8401;
}
{
mul.f16x2 r8459, r8456, r8397;
}
{
sub.f16x2 %33, r8453, r8459;
}
{
add.f16x2 r8465, r8406, r8407;
}
{
mul.f16x2 r8468, r8465, r8395;
}
{
add.f16x2 r8471, r8409, r8468;
}
{
sub.f16x2 r8474, r8400, r8401;
}
{
mul.f16x2 r8477, r8474, r8397;
}
{
add.f16x2 %51, r8471, r8477;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f762;
cvt.rn.f16.f32 high, f762;
mov.b32 r8483, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f764;
cvt.rn.f16.f32 high, f764;
mov.b32 r8484, {low, high};
}
{
neg.f16x2 r8485, r8484;
}
{
add.f16x2 r8487, r8488, r8489;
}
{
add.f16x2 %16, r8491, r8487;
}
{
add.f16x2 r8493, r8494, r8495;
}
{
add.f16x2 %17, r8497, r8493;
}
{
add.f16x2 r8499, r8488, r8489;
}
{
mul.f16x2 r8502, r8499, r8483;
}
{
add.f16x2 r8505, r8491, r8502;
}
{
sub.f16x2 r8508, r8494, r8495;
}
{
mul.f16x2 r8511, r8508, r8485;
}
{
add.f16x2 %34, r8505, r8511;
}
{
add.f16x2 r8517, r8488, r8489;
}
{
mul.f16x2 r8520, r8517, r8483;
}
{
add.f16x2 r8523, r8491, r8520;
}
{
sub.f16x2 r8526, r8494, r8495;
}
{
mul.f16x2 r8529, r8526, r8485;
}
{
sub.f16x2 %52, r8523, r8529;
}
{
add.f16x2 r8535, r8494, r8495;
}
{
mul.f16x2 r8538, r8535, r8483;
}
{
add.f16x2 r8541, r8497, r8538;
}
{
sub.f16x2 r8544, r8488, r8489;
}
{
mul.f16x2 r8547, r8544, r8485;
}
{
sub.f16x2 %35, r8541, r8547;
}
{
add.f16x2 r8553, r8494, r8495;
}
{
mul.f16x2 r8556, r8553, r8483;
}
{
add.f16x2 r8559, r8497, r8556;
}
{
sub.f16x2 r8562, r8488, r8489;
}
{
mul.f16x2 r8565, r8562, r8485;
}
{
add.f16x2 %53, r8559, r8565;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[4].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<894, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<95>;
.reg .b32 r<1124>;
.reg .b64 rd<14>;
mov.u32 r1079, %tid.y;
mov.u32 r1080, %6;
mad.lo.s32 r1081, r1079, 17496, r1080;
mov.u32 r1082, %tid.x;
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
mul.wide.u32 rd2, r1082, 1508246403;
shr.u64 rd3, rd2, 40;
cvt.u32.u64 r1083, rd3;
mul.lo.s32 r1084, r1083, 729;
sub.s32 r1085, r1082, r1084;
mad.lo.s32 r1086, r1083, 17496, r1081;
cvt.rn.f32.u32 f77, r1085;
mul.f32 f78, f77, 0f3B3C4870;
cos.approx.f32 f5, f78;
sin.approx.f32 f79, f78;
neg.f32 f6, f79;
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
mov.f32 f69, 0fBF800000;
mov.f32 f70, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
mad.lo.s32 r1087, r1085, 24, r1086;
st.shared.v2.f32 [r1087], {r8, r14};
st.shared.v2.f32 [r1087+8], {r101, r108};
st.shared.v2.f32 [r1087+16], {r138, r145};
barrier.sync 0;
shl.b32 r1088, r1085, 4;
sub.s32 r1089, r1087, r1088;
ld.shared.u32 r174, [r1089];
ld.shared.u32 r180, [r1089+4];
ld.shared.u32 r171, [r1089+5832];
ld.shared.u32 r177, [r1089+5836];
ld.shared.u32 r172, [r1089+11664];
ld.shared.u32 r178, [r1089+11668];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r166, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
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
mul.wide.u32 rd4, r1085, -1431655765;
shr.u64 rd5, rd4, 33;
cvt.u32.u64 r1090, rd5;
mul.lo.s32 r1091, r1090, 3;
sub.s32 r1092, r1085, r1091;
shl.b32 r1093, r1092, 3;
add.s32 r1094, r1086, r1093;
cvt.rn.f32.u32 f80, r1090;
mul.f32 f81, f80, 0f3C0D3654;
cos.approx.f32 f17, f81;
sin.approx.f32 f82, f81;
neg.f32 f18, f82;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
mad.lo.s32 r1095, r1090, 72, r1094;
st.shared.u32 [r1095], r173;
st.shared.u32 [r1095+4], r179;
st.shared.u32 [r1095+24], r266;
st.shared.u32 [r1095+28], r273;
st.shared.u32 [r1095+48], r303;
st.shared.u32 [r1095+52], r310;
barrier.sync 0;
ld.shared.u32 r339, [r1089];
ld.shared.u32 r345, [r1089+4];
ld.shared.u32 r336, [r1089+5832];
ld.shared.u32 r342, [r1089+5836];
ld.shared.u32 r337, [r1089+11664];
ld.shared.u32 r343, [r1089+11668];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r331, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
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
mul.wide.u32 rd6, r1085, 954437177;
shr.u64 rd7, rd6, 33;
cvt.u32.u64 r1096, rd7;
mul.lo.s32 r1097, r1096, 9;
sub.s32 r1098, r1085, r1097;
shl.b32 r1099, r1098, 3;
add.s32 r1100, r1086, r1099;
cvt.rn.f32.u32 f83, r1096;
mul.f32 f84, f83, 0f3CD3D17E;
cos.approx.f32 f29, f84;
sin.approx.f32 f85, f84;
neg.f32 f30, f85;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
mad.lo.s32 r1101, r1096, 216, r1100;
st.shared.u32 [r1101], r338;
st.shared.u32 [r1101+4], r344;
st.shared.u32 [r1101+72], r431;
st.shared.u32 [r1101+76], r438;
st.shared.u32 [r1101+144], r468;
st.shared.u32 [r1101+148], r475;
barrier.sync 0;
ld.shared.u32 r504, [r1089];
ld.shared.u32 r510, [r1089+4];
ld.shared.u32 r501, [r1089+5832];
ld.shared.u32 r507, [r1089+5836];
ld.shared.u32 r502, [r1089+11664];
ld.shared.u32 r508, [r1089+11668];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r496, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r497, {low, high};
}
{
neg.f16x2 r498, r497;
}
{
add.f16x2 r500, r501, r502;
}
{
add.f16x2 r503, r504, r500;
}
{
add.f16x2 r506, r507, r508;
}
{
add.f16x2 r509, r510, r506;
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
add.f16x2 r527, r518, r524;
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
sub.f16x2 r545, r536, r542;
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
sub.f16x2 r563, r554, r560;
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
add.f16x2 r581, r572, r578;
}
mul.wide.u32 rd8, r1085, 795364315;
shr.u64 rd9, rd8, 32;
cvt.u32.u64 r1102, rd9;
sub.s32 r1103, r1085, r1102;
shr.u32 r1104, r1103, 1;
add.s32 r1105, r1104, r1102;
shr.u32 r1106, r1105, 4;
mul.lo.s32 r1107, r1106, 27;
sub.s32 r1108, r1085, r1107;
shl.b32 r1109, r1108, 3;
add.s32 r1110, r1086, r1109;
cvt.rn.f32.u32 f86, r1106;
mul.f32 f87, f86, 0f3D9EDD1F;
cos.approx.f32 f41, f87;
sin.approx.f32 f88, f87;
neg.f32 f42, f88;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f41;
cvt.rn.f16.f32 high, f42;
mov.b32 r584, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r584;
mov.b32 r587, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r584;
mov.b32 r589, {high, high};
}
{
mul.f16x2 r591, r563, r589;
}
{
neg.f16x2 r594, r591;
}
{
fma.rn.f16x2 r596, r527, r587, r594;
}
{
mul.f16x2 r600, r527, r589;
}
{
fma.rn.f16x2 r603, r563, r587, r600;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r584;
mov.b32 r607, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r584;
mov.b32 r609, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r611, {low, high};
}
{
mul.f16x2 r612, r609, r611;
}
{
mul.f16x2 r615, r584, r607;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r584;
mov.b32 r618, {high, low};
}
{
fma.rn.f16x2 r620, r612, r618, r615;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r620;
mov.b32 r624, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r620;
mov.b32 r626, {high, high};
}
{
mul.f16x2 r628, r581, r626;
}
{
neg.f16x2 r631, r628;
}
{
fma.rn.f16x2 r633, r545, r624, r631;
}
{
mul.f16x2 r637, r545, r626;
}
{
fma.rn.f16x2 r640, r581, r624, r637;
}
barrier.sync 0;
mad.lo.s32 r1111, r1106, 648, r1110;
st.shared.u32 [r1111], r503;
st.shared.u32 [r1111+4], r509;
st.shared.u32 [r1111+216], r596;
st.shared.u32 [r1111+220], r603;
st.shared.u32 [r1111+432], r633;
st.shared.u32 [r1111+436], r640;
barrier.sync 0;
ld.shared.u32 r669, [r1089];
ld.shared.u32 r675, [r1089+4];
ld.shared.u32 r666, [r1089+5832];
ld.shared.u32 r672, [r1089+5836];
ld.shared.u32 r667, [r1089+11664];
ld.shared.u32 r673, [r1089+11668];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r661, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r662, {low, high};
}
{
neg.f16x2 r663, r662;
}
{
add.f16x2 r665, r666, r667;
}
{
add.f16x2 r668, r669, r665;
}
{
add.f16x2 r671, r672, r673;
}
{
add.f16x2 r674, r675, r671;
}
{
add.f16x2 r677, r666, r667;
}
{
mul.f16x2 r680, r677, r661;
}
{
add.f16x2 r683, r669, r680;
}
{
sub.f16x2 r686, r672, r673;
}
{
mul.f16x2 r689, r686, r663;
}
{
add.f16x2 r692, r683, r689;
}
{
add.f16x2 r695, r666, r667;
}
{
mul.f16x2 r698, r695, r661;
}
{
add.f16x2 r701, r669, r698;
}
{
sub.f16x2 r704, r672, r673;
}
{
mul.f16x2 r707, r704, r663;
}
{
sub.f16x2 r710, r701, r707;
}
{
add.f16x2 r713, r672, r673;
}
{
mul.f16x2 r716, r713, r661;
}
{
add.f16x2 r719, r675, r716;
}
{
sub.f16x2 r722, r666, r667;
}
{
mul.f16x2 r725, r722, r663;
}
{
sub.f16x2 r728, r719, r725;
}
{
add.f16x2 r731, r672, r673;
}
{
mul.f16x2 r734, r731, r661;
}
{
add.f16x2 r737, r675, r734;
}
{
sub.f16x2 r740, r666, r667;
}
{
mul.f16x2 r743, r740, r663;
}
{
add.f16x2 r746, r737, r743;
}
mul.wide.u32 rd10, r1085, -901412889;
shr.u64 rd11, rd10, 38;
cvt.u32.u64 r1112, rd11;
mul.lo.s32 r1113, r1112, 81;
sub.s32 r1114, r1085, r1113;
shl.b32 r1115, r1114, 3;
add.s32 r1116, r1086, r1115;
cvt.rn.f32.u32 f89, r1112;
mul.f32 f90, f89, 0f3E6E4BAE;
cos.approx.f32 f53, f90;
sin.approx.f32 f91, f90;
neg.f32 f54, f91;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r749, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r749;
mov.b32 r752, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r749;
mov.b32 r754, {high, high};
}
{
mul.f16x2 r756, r728, r754;
}
{
neg.f16x2 r759, r756;
}
{
fma.rn.f16x2 r761, r692, r752, r759;
}
{
mul.f16x2 r765, r692, r754;
}
{
fma.rn.f16x2 r768, r728, r752, r765;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r749;
mov.b32 r772, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r749;
mov.b32 r774, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r776, {low, high};
}
{
mul.f16x2 r777, r774, r776;
}
{
mul.f16x2 r780, r749, r772;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r749;
mov.b32 r783, {high, low};
}
{
fma.rn.f16x2 r785, r777, r783, r780;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r785;
mov.b32 r789, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r785;
mov.b32 r791, {high, high};
}
{
mul.f16x2 r793, r746, r791;
}
{
neg.f16x2 r796, r793;
}
{
fma.rn.f16x2 r798, r710, r789, r796;
}
{
mul.f16x2 r802, r710, r791;
}
{
fma.rn.f16x2 r805, r746, r789, r802;
}
barrier.sync 0;
mad.lo.s32 r1117, r1112, 1944, r1116;
st.shared.u32 [r1117], r668;
st.shared.u32 [r1117+4], r674;
st.shared.u32 [r1117+648], r761;
st.shared.u32 [r1117+652], r768;
st.shared.u32 [r1117+1296], r798;
st.shared.u32 [r1117+1300], r805;
barrier.sync 0;
ld.shared.u32 r834, [r1089];
ld.shared.u32 r840, [r1089+4];
ld.shared.u32 r831, [r1089+5832];
ld.shared.u32 r837, [r1089+5836];
ld.shared.u32 r832, [r1089+11664];
ld.shared.u32 r838, [r1089+11668];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r826, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r827, {low, high};
}
{
neg.f16x2 r828, r827;
}
{
add.f16x2 r830, r831, r832;
}
{
add.f16x2 r833, r834, r830;
}
{
add.f16x2 r836, r837, r838;
}
{
add.f16x2 r839, r840, r836;
}
{
add.f16x2 r842, r831, r832;
}
{
mul.f16x2 r845, r842, r826;
}
{
add.f16x2 r848, r834, r845;
}
{
sub.f16x2 r851, r837, r838;
}
{
mul.f16x2 r854, r851, r828;
}
{
add.f16x2 r857, r848, r854;
}
{
add.f16x2 r860, r831, r832;
}
{
mul.f16x2 r863, r860, r826;
}
{
add.f16x2 r866, r834, r863;
}
{
sub.f16x2 r869, r837, r838;
}
{
mul.f16x2 r872, r869, r828;
}
{
sub.f16x2 r875, r866, r872;
}
{
add.f16x2 r878, r837, r838;
}
{
mul.f16x2 r881, r878, r826;
}
{
add.f16x2 r884, r840, r881;
}
{
sub.f16x2 r887, r831, r832;
}
{
mul.f16x2 r890, r887, r828;
}
{
sub.f16x2 r893, r884, r890;
}
{
add.f16x2 r896, r837, r838;
}
{
mul.f16x2 r899, r896, r826;
}
{
add.f16x2 r902, r840, r899;
}
{
sub.f16x2 r905, r831, r832;
}
{
mul.f16x2 r908, r905, r828;
}
{
add.f16x2 r911, r902, r908;
}
mul.wide.u32 rd12, r1085, -2032597691;
shr.u64 rd13, rd12, 39;
cvt.u32.u64 r1118, rd13;
mul.lo.s32 r1119, r1118, 243;
sub.s32 r1120, r1085, r1119;
shl.b32 r1121, r1120, 3;
add.s32 r1122, r1086, r1121;
cvt.rn.f32.u32 f92, r1118;
mul.f32 f93, f92, 0f3F32B8C2;
cos.approx.f32 f65, f93;
sin.approx.f32 f94, f93;
neg.f32 f66, f94;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f65;
cvt.rn.f16.f32 high, f66;
mov.b32 r914, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r914;
mov.b32 r917, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r914;
mov.b32 r919, {high, high};
}
{
mul.f16x2 r921, r893, r919;
}
{
neg.f16x2 r924, r921;
}
{
fma.rn.f16x2 r926, r857, r917, r924;
}
{
mul.f16x2 r930, r857, r919;
}
{
fma.rn.f16x2 r933, r893, r917, r930;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r914;
mov.b32 r937, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r914;
mov.b32 r939, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r941, {low, high};
}
{
mul.f16x2 r942, r939, r941;
}
{
mul.f16x2 r945, r914, r937;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r914;
mov.b32 r948, {high, low};
}
{
fma.rn.f16x2 r950, r942, r948, r945;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r950;
mov.b32 r954, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r950;
mov.b32 r956, {high, high};
}
{
mul.f16x2 r958, r911, r956;
}
{
neg.f16x2 r961, r958;
}
{
fma.rn.f16x2 r963, r875, r954, r961;
}
{
mul.f16x2 r967, r875, r956;
}
{
fma.rn.f16x2 r970, r911, r954, r967;
}
barrier.sync 0;
mad.lo.s32 r1123, r1118, 5832, r1122;
st.shared.u32 [r1123], r833;
st.shared.u32 [r1123+4], r839;
st.shared.u32 [r1123+1944], r926;
st.shared.u32 [r1123+1948], r933;
st.shared.u32 [r1123+3888], r963;
st.shared.u32 [r1123+3892], r970;
barrier.sync 0;
ld.shared.u32 r999, [r1089];
ld.shared.u32 r1005, [r1089+4];
ld.shared.u32 r996, [r1089+5832];
ld.shared.u32 r1002, [r1089+5836];
ld.shared.u32 r997, [r1089+11664];
ld.shared.u32 r1003, [r1089+11668];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r991, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r992, {low, high};
}
{
neg.f16x2 r993, r992;
}
{
add.f16x2 r995, r996, r997;
}
{
add.f16x2 %0, r999, r995;
}
{
add.f16x2 r1001, r1002, r1003;
}
{
add.f16x2 %1, r1005, r1001;
}
{
add.f16x2 r1007, r996, r997;
}
{
mul.f16x2 r1010, r1007, r991;
}
{
add.f16x2 r1013, r999, r1010;
}
{
sub.f16x2 r1016, r1002, r1003;
}
{
mul.f16x2 r1019, r1016, r993;
}
{
add.f16x2 %2, r1013, r1019;
}
{
add.f16x2 r1025, r996, r997;
}
{
mul.f16x2 r1028, r1025, r991;
}
{
add.f16x2 r1031, r999, r1028;
}
{
sub.f16x2 r1034, r1002, r1003;
}
{
mul.f16x2 r1037, r1034, r993;
}
{
sub.f16x2 %4, r1031, r1037;
}
{
add.f16x2 r1043, r1002, r1003;
}
{
mul.f16x2 r1046, r1043, r991;
}
{
add.f16x2 r1049, r1005, r1046;
}
{
sub.f16x2 r1052, r996, r997;
}
{
mul.f16x2 r1055, r1052, r993;
}
{
sub.f16x2 %3, r1049, r1055;
}
{
add.f16x2 r1061, r1002, r1003;
}
{
mul.f16x2 r1064, r1061, r991;
}
{
add.f16x2 r1067, r1005, r1064;
}
{
sub.f16x2 r1070, r996, r997;
}
{
mul.f16x2 r1073, r1070, r993;
}
{
add.f16x2 %5, r1067, r1073;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<895, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<95>;
.reg .b32 r<1124>;
.reg .b64 rd<14>;
mov.u32 r1079, %tid.y;
mov.u32 r1080, %6;
mad.lo.s32 r1081, r1079, 8748, r1080;
mov.u32 r1082, %tid.x;
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
mul.wide.u32 rd2, r1082, 1508246403;
shr.u64 rd3, rd2, 40;
cvt.u32.u64 r1083, rd3;
mul.lo.s32 r1084, r1083, 729;
sub.s32 r1085, r1082, r1084;
mad.lo.s32 r1086, r1083, 8748, r1081;
cvt.rn.f32.u32 f77, r1085;
mul.f32 f78, f77, 0f3B3C4870;
cos.approx.f32 f5, f78;
sin.approx.f32 f79, f78;
neg.f32 f6, f79;
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
mov.f32 f69, 0fBF800000;
mov.f32 f70, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
mad.lo.s32 r1087, r1085, 12, r1086;
st.shared.u32 [r1087], r8;
st.shared.u32 [r1087+4], r101;
st.shared.u32 [r1087+8], r138;
barrier.sync 0;
shl.b32 r1088, r1085, 3;
sub.s32 r1089, r1087, r1088;
ld.shared.u32 r174, [r1089];
ld.shared.u32 r171, [r1089+2916];
ld.shared.u32 r172, [r1089+5832];
barrier.sync 0;
st.shared.u32 [r1087], r14;
st.shared.u32 [r1087+4], r108;
st.shared.u32 [r1087+8], r145;
barrier.sync 0;
ld.shared.u32 r180, [r1089];
ld.shared.u32 r177, [r1089+2916];
ld.shared.u32 r178, [r1089+5832];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r166, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
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
mul.wide.u32 rd4, r1085, -1431655765;
shr.u64 rd5, rd4, 33;
cvt.u32.u64 r1090, rd5;
mul.lo.s32 r1091, r1090, 3;
sub.s32 r1092, r1085, r1091;
shl.b32 r1093, r1092, 2;
add.s32 r1094, r1086, r1093;
cvt.rn.f32.u32 f80, r1090;
mul.f32 f81, f80, 0f3C0D3654;
cos.approx.f32 f17, f81;
sin.approx.f32 f82, f81;
neg.f32 f18, f82;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
mad.lo.s32 r1095, r1090, 36, r1094;
st.shared.u32 [r1095], r173;
st.shared.u32 [r1095+12], r266;
st.shared.u32 [r1095+24], r303;
barrier.sync 0;
ld.shared.u32 r339, [r1089];
ld.shared.u32 r336, [r1089+2916];
ld.shared.u32 r337, [r1089+5832];
barrier.sync 0;
st.shared.u32 [r1095], r179;
st.shared.u32 [r1095+12], r273;
st.shared.u32 [r1095+24], r310;
barrier.sync 0;
ld.shared.u32 r345, [r1089];
ld.shared.u32 r342, [r1089+2916];
ld.shared.u32 r343, [r1089+5832];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r331, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
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
mul.wide.u32 rd6, r1085, 954437177;
shr.u64 rd7, rd6, 33;
cvt.u32.u64 r1096, rd7;
mul.lo.s32 r1097, r1096, 9;
sub.s32 r1098, r1085, r1097;
shl.b32 r1099, r1098, 2;
add.s32 r1100, r1086, r1099;
cvt.rn.f32.u32 f83, r1096;
mul.f32 f84, f83, 0f3CD3D17E;
cos.approx.f32 f29, f84;
sin.approx.f32 f85, f84;
neg.f32 f30, f85;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
mad.lo.s32 r1101, r1096, 108, r1100;
st.shared.u32 [r1101], r338;
st.shared.u32 [r1101+36], r431;
st.shared.u32 [r1101+72], r468;
barrier.sync 0;
ld.shared.u32 r504, [r1089];
ld.shared.u32 r501, [r1089+2916];
ld.shared.u32 r502, [r1089+5832];
barrier.sync 0;
st.shared.u32 [r1101], r344;
st.shared.u32 [r1101+36], r438;
st.shared.u32 [r1101+72], r475;
barrier.sync 0;
ld.shared.u32 r510, [r1089];
ld.shared.u32 r507, [r1089+2916];
ld.shared.u32 r508, [r1089+5832];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r496, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r497, {low, high};
}
{
neg.f16x2 r498, r497;
}
{
add.f16x2 r500, r501, r502;
}
{
add.f16x2 r503, r504, r500;
}
{
add.f16x2 r506, r507, r508;
}
{
add.f16x2 r509, r510, r506;
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
add.f16x2 r527, r518, r524;
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
sub.f16x2 r545, r536, r542;
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
sub.f16x2 r563, r554, r560;
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
add.f16x2 r581, r572, r578;
}
mul.wide.u32 rd8, r1085, 795364315;
shr.u64 rd9, rd8, 32;
cvt.u32.u64 r1102, rd9;
sub.s32 r1103, r1085, r1102;
shr.u32 r1104, r1103, 1;
add.s32 r1105, r1104, r1102;
shr.u32 r1106, r1105, 4;
mul.lo.s32 r1107, r1106, 27;
sub.s32 r1108, r1085, r1107;
shl.b32 r1109, r1108, 2;
add.s32 r1110, r1086, r1109;
cvt.rn.f32.u32 f86, r1106;
mul.f32 f87, f86, 0f3D9EDD1F;
cos.approx.f32 f41, f87;
sin.approx.f32 f88, f87;
neg.f32 f42, f88;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f41;
cvt.rn.f16.f32 high, f42;
mov.b32 r584, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r584;
mov.b32 r587, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r584;
mov.b32 r589, {high, high};
}
{
mul.f16x2 r591, r563, r589;
}
{
neg.f16x2 r594, r591;
}
{
fma.rn.f16x2 r596, r527, r587, r594;
}
{
mul.f16x2 r600, r527, r589;
}
{
fma.rn.f16x2 r603, r563, r587, r600;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r584;
mov.b32 r607, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r584;
mov.b32 r609, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r611, {low, high};
}
{
mul.f16x2 r612, r609, r611;
}
{
mul.f16x2 r615, r584, r607;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r584;
mov.b32 r618, {high, low};
}
{
fma.rn.f16x2 r620, r612, r618, r615;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r620;
mov.b32 r624, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r620;
mov.b32 r626, {high, high};
}
{
mul.f16x2 r628, r581, r626;
}
{
neg.f16x2 r631, r628;
}
{
fma.rn.f16x2 r633, r545, r624, r631;
}
{
mul.f16x2 r637, r545, r626;
}
{
fma.rn.f16x2 r640, r581, r624, r637;
}
barrier.sync 0;
mad.lo.s32 r1111, r1106, 324, r1110;
st.shared.u32 [r1111], r503;
st.shared.u32 [r1111+108], r596;
st.shared.u32 [r1111+216], r633;
barrier.sync 0;
ld.shared.u32 r669, [r1089];
ld.shared.u32 r666, [r1089+2916];
ld.shared.u32 r667, [r1089+5832];
barrier.sync 0;
st.shared.u32 [r1111], r509;
st.shared.u32 [r1111+108], r603;
st.shared.u32 [r1111+216], r640;
barrier.sync 0;
ld.shared.u32 r675, [r1089];
ld.shared.u32 r672, [r1089+2916];
ld.shared.u32 r673, [r1089+5832];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r661, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r662, {low, high};
}
{
neg.f16x2 r663, r662;
}
{
add.f16x2 r665, r666, r667;
}
{
add.f16x2 r668, r669, r665;
}
{
add.f16x2 r671, r672, r673;
}
{
add.f16x2 r674, r675, r671;
}
{
add.f16x2 r677, r666, r667;
}
{
mul.f16x2 r680, r677, r661;
}
{
add.f16x2 r683, r669, r680;
}
{
sub.f16x2 r686, r672, r673;
}
{
mul.f16x2 r689, r686, r663;
}
{
add.f16x2 r692, r683, r689;
}
{
add.f16x2 r695, r666, r667;
}
{
mul.f16x2 r698, r695, r661;
}
{
add.f16x2 r701, r669, r698;
}
{
sub.f16x2 r704, r672, r673;
}
{
mul.f16x2 r707, r704, r663;
}
{
sub.f16x2 r710, r701, r707;
}
{
add.f16x2 r713, r672, r673;
}
{
mul.f16x2 r716, r713, r661;
}
{
add.f16x2 r719, r675, r716;
}
{
sub.f16x2 r722, r666, r667;
}
{
mul.f16x2 r725, r722, r663;
}
{
sub.f16x2 r728, r719, r725;
}
{
add.f16x2 r731, r672, r673;
}
{
mul.f16x2 r734, r731, r661;
}
{
add.f16x2 r737, r675, r734;
}
{
sub.f16x2 r740, r666, r667;
}
{
mul.f16x2 r743, r740, r663;
}
{
add.f16x2 r746, r737, r743;
}
mul.wide.u32 rd10, r1085, -901412889;
shr.u64 rd11, rd10, 38;
cvt.u32.u64 r1112, rd11;
mul.lo.s32 r1113, r1112, 81;
sub.s32 r1114, r1085, r1113;
shl.b32 r1115, r1114, 2;
add.s32 r1116, r1086, r1115;
cvt.rn.f32.u32 f89, r1112;
mul.f32 f90, f89, 0f3E6E4BAE;
cos.approx.f32 f53, f90;
sin.approx.f32 f91, f90;
neg.f32 f54, f91;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f53;
cvt.rn.f16.f32 high, f54;
mov.b32 r749, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r749;
mov.b32 r752, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r749;
mov.b32 r754, {high, high};
}
{
mul.f16x2 r756, r728, r754;
}
{
neg.f16x2 r759, r756;
}
{
fma.rn.f16x2 r761, r692, r752, r759;
}
{
mul.f16x2 r765, r692, r754;
}
{
fma.rn.f16x2 r768, r728, r752, r765;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r749;
mov.b32 r772, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r749;
mov.b32 r774, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r776, {low, high};
}
{
mul.f16x2 r777, r774, r776;
}
{
mul.f16x2 r780, r749, r772;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r749;
mov.b32 r783, {high, low};
}
{
fma.rn.f16x2 r785, r777, r783, r780;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r785;
mov.b32 r789, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r785;
mov.b32 r791, {high, high};
}
{
mul.f16x2 r793, r746, r791;
}
{
neg.f16x2 r796, r793;
}
{
fma.rn.f16x2 r798, r710, r789, r796;
}
{
mul.f16x2 r802, r710, r791;
}
{
fma.rn.f16x2 r805, r746, r789, r802;
}
barrier.sync 0;
mad.lo.s32 r1117, r1112, 972, r1116;
st.shared.u32 [r1117], r668;
st.shared.u32 [r1117+324], r761;
st.shared.u32 [r1117+648], r798;
barrier.sync 0;
ld.shared.u32 r834, [r1089];
ld.shared.u32 r831, [r1089+2916];
ld.shared.u32 r832, [r1089+5832];
barrier.sync 0;
st.shared.u32 [r1117], r674;
st.shared.u32 [r1117+324], r768;
st.shared.u32 [r1117+648], r805;
barrier.sync 0;
ld.shared.u32 r840, [r1089];
ld.shared.u32 r837, [r1089+2916];
ld.shared.u32 r838, [r1089+5832];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r826, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r827, {low, high};
}
{
neg.f16x2 r828, r827;
}
{
add.f16x2 r830, r831, r832;
}
{
add.f16x2 r833, r834, r830;
}
{
add.f16x2 r836, r837, r838;
}
{
add.f16x2 r839, r840, r836;
}
{
add.f16x2 r842, r831, r832;
}
{
mul.f16x2 r845, r842, r826;
}
{
add.f16x2 r848, r834, r845;
}
{
sub.f16x2 r851, r837, r838;
}
{
mul.f16x2 r854, r851, r828;
}
{
add.f16x2 r857, r848, r854;
}
{
add.f16x2 r860, r831, r832;
}
{
mul.f16x2 r863, r860, r826;
}
{
add.f16x2 r866, r834, r863;
}
{
sub.f16x2 r869, r837, r838;
}
{
mul.f16x2 r872, r869, r828;
}
{
sub.f16x2 r875, r866, r872;
}
{
add.f16x2 r878, r837, r838;
}
{
mul.f16x2 r881, r878, r826;
}
{
add.f16x2 r884, r840, r881;
}
{
sub.f16x2 r887, r831, r832;
}
{
mul.f16x2 r890, r887, r828;
}
{
sub.f16x2 r893, r884, r890;
}
{
add.f16x2 r896, r837, r838;
}
{
mul.f16x2 r899, r896, r826;
}
{
add.f16x2 r902, r840, r899;
}
{
sub.f16x2 r905, r831, r832;
}
{
mul.f16x2 r908, r905, r828;
}
{
add.f16x2 r911, r902, r908;
}
mul.wide.u32 rd12, r1085, -2032597691;
shr.u64 rd13, rd12, 39;
cvt.u32.u64 r1118, rd13;
mul.lo.s32 r1119, r1118, 243;
sub.s32 r1120, r1085, r1119;
shl.b32 r1121, r1120, 2;
add.s32 r1122, r1086, r1121;
cvt.rn.f32.u32 f92, r1118;
mul.f32 f93, f92, 0f3F32B8C2;
cos.approx.f32 f65, f93;
sin.approx.f32 f94, f93;
neg.f32 f66, f94;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f65;
cvt.rn.f16.f32 high, f66;
mov.b32 r914, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r914;
mov.b32 r917, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r914;
mov.b32 r919, {high, high};
}
{
mul.f16x2 r921, r893, r919;
}
{
neg.f16x2 r924, r921;
}
{
fma.rn.f16x2 r926, r857, r917, r924;
}
{
mul.f16x2 r930, r857, r919;
}
{
fma.rn.f16x2 r933, r893, r917, r930;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r914;
mov.b32 r937, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r914;
mov.b32 r939, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r941, {low, high};
}
{
mul.f16x2 r942, r939, r941;
}
{
mul.f16x2 r945, r914, r937;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r914;
mov.b32 r948, {high, low};
}
{
fma.rn.f16x2 r950, r942, r948, r945;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r950;
mov.b32 r954, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r950;
mov.b32 r956, {high, high};
}
{
mul.f16x2 r958, r911, r956;
}
{
neg.f16x2 r961, r958;
}
{
fma.rn.f16x2 r963, r875, r954, r961;
}
{
mul.f16x2 r967, r875, r956;
}
{
fma.rn.f16x2 r970, r911, r954, r967;
}
barrier.sync 0;
mad.lo.s32 r1123, r1118, 2916, r1122;
st.shared.u32 [r1123], r833;
st.shared.u32 [r1123+972], r926;
st.shared.u32 [r1123+1944], r963;
barrier.sync 0;
ld.shared.u32 r999, [r1089];
ld.shared.u32 r996, [r1089+2916];
ld.shared.u32 r997, [r1089+5832];
barrier.sync 0;
st.shared.u32 [r1123], r839;
st.shared.u32 [r1123+972], r933;
st.shared.u32 [r1123+1944], r970;
barrier.sync 0;
ld.shared.u32 r1005, [r1089];
ld.shared.u32 r1002, [r1089+2916];
ld.shared.u32 r1003, [r1089+5832];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r991, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r992, {low, high};
}
{
neg.f16x2 r993, r992;
}
{
add.f16x2 r995, r996, r997;
}
{
add.f16x2 %0, r999, r995;
}
{
add.f16x2 r1001, r1002, r1003;
}
{
add.f16x2 %1, r1005, r1001;
}
{
add.f16x2 r1007, r996, r997;
}
{
mul.f16x2 r1010, r1007, r991;
}
{
add.f16x2 r1013, r999, r1010;
}
{
sub.f16x2 r1016, r1002, r1003;
}
{
mul.f16x2 r1019, r1016, r993;
}
{
add.f16x2 %2, r1013, r1019;
}
{
add.f16x2 r1025, r996, r997;
}
{
mul.f16x2 r1028, r1025, r991;
}
{
add.f16x2 r1031, r999, r1028;
}
{
sub.f16x2 r1034, r1002, r1003;
}
{
mul.f16x2 r1037, r1034, r993;
}
{
sub.f16x2 %4, r1031, r1037;
}
{
add.f16x2 r1043, r1002, r1003;
}
{
mul.f16x2 r1046, r1043, r991;
}
{
add.f16x2 r1049, r1005, r1046;
}
{
sub.f16x2 r1052, r996, r997;
}
{
mul.f16x2 r1055, r1052, r993;
}
{
sub.f16x2 %3, r1049, r1055;
}
{
add.f16x2 r1061, r1002, r1003;
}
{
mul.f16x2 r1064, r1061, r991;
}
{
add.f16x2 r1067, r1005, r1064;
}
{
sub.f16x2 r1070, r996, r997;
}
{
mul.f16x2 r1073, r1070, r993;
}
{
add.f16x2 %5, r1067, r1073;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)));
};


#endif
