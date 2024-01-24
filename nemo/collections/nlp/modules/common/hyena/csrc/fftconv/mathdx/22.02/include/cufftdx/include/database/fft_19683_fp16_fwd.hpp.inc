#ifndef CUFFTDX_FFT_19683_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_19683_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1177, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<1043>;
.reg .b32 r<10778>;
.reg .b64 rd<6>;
mov.u32 r10704, %54;
mov.u32 r10777, %tid.y;
mad.lo.s32 r10705, r10777, 157464, r10704;
mov.u32 r10706, %tid.x;
mov.f32 f1034, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1, {low, high};
}
mov.f32 f1036, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r177, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
mov.f32 f906, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r265, {low, high};
}
mov.f32 f908, 0fBF248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r266, {low, high};
}
mov.f32 f918, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r267, {low, high};
}
mov.f32 f920, 0fBF7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r268, {low, high};
}
mov.f32 f942, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r271, {low, high};
}
mov.f32 f944, 0fBEAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r345, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r433, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r521, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r609, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r697, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r785, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r873, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r874, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r875, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r876, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r879, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r953, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1041, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1129, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1217, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
add.f16x2 r1230, %83, r1227;
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
add.f16x2 r1275, %83, r1272;
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
add.f16x2 r1293, %83, r1290;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1305, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r1306, {low, high};
}
{
neg.f16x2 r1307, r1306;
}
{
add.f16x2 r1309, %82, %73;
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
add.f16x2 r1321, %82, %73;
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
add.f16x2 r1339, %82, %73;
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
sub.f16x2 r1366, %82, %73;
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
sub.f16x2 r1384, %82, %73;
}
{
mul.f16x2 r1387, r1384, r1307;
}
{
add.f16x2 r1390, r1381, r1387;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1393, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r1481, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r1482, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r1483, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r1484, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r1487, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1561, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1649, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1737, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
mov.f32 f898, 0f3F791978;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f898;
cvt.rn.f16.f32 high, f898;
mov.b32 r1825, {low, high};
}
mov.f32 f900, 0fBE6C2691;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f900;
cvt.rn.f16.f32 high, f900;
mov.b32 r1826, {low, high};
}
mov.f32 f902, 0f3F64C51C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f902;
cvt.rn.f16.f32 high, f902;
mov.b32 r1827, {low, high};
}
mov.f32 f904, 0fBEE5C902;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f904;
cvt.rn.f16.f32 high, f904;
mov.b32 r1828, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r1829, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r1830, {low, high};
}
mov.f32 f910, 0f3F18DF63;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f910;
cvt.rn.f16.f32 high, f910;
mov.b32 r1831, {low, high};
}
mov.f32 f912, 0fBF4D57F2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f912;
cvt.rn.f16.f32 high, f912;
mov.b32 r1832, {low, high};
}
mov.f32 f914, 0f3ECACAF8;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f914;
cvt.rn.f16.f32 high, f914;
mov.b32 r1833, {low, high};
}
mov.f32 f916, 0fBF6B1036;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f916;
cvt.rn.f16.f32 high, f916;
mov.b32 r1834, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r1835, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r1836, {low, high};
}
mov.f32 f922, 0fBD6E2946;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f922;
cvt.rn.f16.f32 high, f922;
mov.b32 r1837, {low, high};
}
mov.f32 f924, 0fBF7F9120;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f924;
cvt.rn.f16.f32 high, f924;
mov.b32 r1838, {low, high};
}
mov.f32 f926, 0fBE92D7E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f926;
cvt.rn.f16.f32 high, f926;
mov.b32 r1839, {low, high};
}
mov.f32 f928, 0fBF753ECD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f928;
cvt.rn.f16.f32 high, f928;
mov.b32 r1840, {low, high};
}
mov.f32 f934, 0fBF2FAD88;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f934;
cvt.rn.f16.f32 high, f934;
mov.b32 r1843, {low, high};
}
mov.f32 f936, 0fBF3A3529;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f936;
cvt.rn.f16.f32 high, f936;
mov.b32 r1844, {low, high};
}
mov.f32 f958, 0fBF55E287;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r1847, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
mov.b32 r1848, {low, high};
}
mov.f32 f950, 0fBF7E44DE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f950;
cvt.rn.f16.f32 high, f950;
mov.b32 r1851, {low, high};
}
mov.f32 f952, 0f3DEDC21F;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f952;
cvt.rn.f16.f32 high, f952;
mov.b32 r1852, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f958;
cvt.rn.f16.f32 high, f958;
mov.b32 r1855, {low, high};
}
mov.f32 f960, 0f3F0CAC9F;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f960;
cvt.rn.f16.f32 high, f960;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r2133, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r2221, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r2309, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r2397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r2485, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r2573, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r2661, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r2749, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r2837, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
mul.wide.u32 rd2, r10706, 1508246403;
shr.u64 rd3, rd2, 40;
cvt.u32.u64 r10707, rd3;
mul.lo.s32 r10708, r10707, 729;
sub.s32 r10709, r10706, r10708;
cvt.rn.f32.u32 f1037, r10709;
mul.f32 f1038, f1037, 0f39A75CD5;
cos.approx.f32 f309, f1038;
sin.approx.f32 f1039, f1038;
neg.f32 f310, f1039;
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
mad.lo.s32 r10710, r10707, 157464, r10705;
barrier.sync 0;
mad.lo.s32 r10711, r10709, 216, r10710;
st.shared.v2.f32 [r10711], {r2140, r2146};
st.shared.v2.f32 [r10711+8], {r2937, r2944};
st.shared.v2.f32 [r10711+16], {r2974, r2981};
st.shared.v2.f32 [r10711+24], {r3011, r3018};
st.shared.v2.f32 [r10711+32], {r3048, r3055};
st.shared.v2.f32 [r10711+40], {r3085, r3092};
st.shared.v2.f32 [r10711+48], {r3122, r3129};
st.shared.v2.f32 [r10711+56], {r3159, r3166};
st.shared.v2.f32 [r10711+64], {r3196, r3203};
st.shared.v2.f32 [r10711+72], {r3233, r3240};
st.shared.v2.f32 [r10711+80], {r3270, r3277};
st.shared.v2.f32 [r10711+88], {r3307, r3314};
st.shared.v2.f32 [r10711+96], {r3344, r3351};
st.shared.v2.f32 [r10711+104], {r3381, r3388};
st.shared.v2.f32 [r10711+112], {r3418, r3425};
st.shared.v2.f32 [r10711+120], {r3455, r3462};
st.shared.v2.f32 [r10711+128], {r3492, r3499};
st.shared.v2.f32 [r10711+136], {r3529, r3536};
st.shared.v2.f32 [r10711+144], {r3566, r3573};
st.shared.v2.f32 [r10711+152], {r3603, r3610};
st.shared.v2.f32 [r10711+160], {r3640, r3647};
st.shared.v2.f32 [r10711+168], {r3677, r3684};
st.shared.v2.f32 [r10711+176], {r3714, r3721};
st.shared.v2.f32 [r10711+184], {r3751, r3758};
st.shared.v2.f32 [r10711+192], {r3788, r3795};
st.shared.v2.f32 [r10711+200], {r3825, r3832};
st.shared.v2.f32 [r10711+208], {r3862, r3869};
barrier.sync 0;
mad.lo.s32 r10712, r10709, -208, r10711;
ld.shared.u32 r3898, [r10712];
ld.shared.u32 r3904, [r10712+4];
ld.shared.u32 r4506, [r10712+5832];
ld.shared.u32 r4512, [r10712+5836];
ld.shared.u32 r5114, [r10712+11664];
ld.shared.u32 r5120, [r10712+11668];
ld.shared.u32 r3986, [r10712+17496];
ld.shared.u32 r3992, [r10712+17500];
ld.shared.u32 r4594, [r10712+23328];
ld.shared.u32 r4600, [r10712+23332];
ld.shared.u32 r5202, [r10712+29160];
ld.shared.u32 r5208, [r10712+29164];
ld.shared.u32 r4074, [r10712+34992];
ld.shared.u32 r4080, [r10712+34996];
ld.shared.u32 r4682, [r10712+40824];
ld.shared.u32 r4688, [r10712+40828];
ld.shared.u32 r5290, [r10712+46656];
ld.shared.u32 r5296, [r10712+46660];
ld.shared.u32 r3895, [r10712+52488];
ld.shared.u32 r3901, [r10712+52492];
ld.shared.u32 r4503, [r10712+58320];
ld.shared.u32 r4509, [r10712+58324];
ld.shared.u32 r5111, [r10712+64152];
ld.shared.u32 r5117, [r10712+64156];
ld.shared.u32 r3983, [r10712+69984];
ld.shared.u32 r3989, [r10712+69988];
ld.shared.u32 r4591, [r10712+75816];
ld.shared.u32 r4597, [r10712+75820];
ld.shared.u32 r5199, [r10712+81648];
ld.shared.u32 r5205, [r10712+81652];
ld.shared.u32 r4071, [r10712+87480];
ld.shared.u32 r4077, [r10712+87484];
ld.shared.u32 r4679, [r10712+93312];
ld.shared.u32 r4685, [r10712+93316];
ld.shared.u32 r5287, [r10712+99144];
ld.shared.u32 r5293, [r10712+99148];
ld.shared.u32 r3896, [r10712+104976];
ld.shared.u32 r3902, [r10712+104980];
ld.shared.u32 r4504, [r10712+110808];
ld.shared.u32 r4510, [r10712+110812];
ld.shared.u32 r5112, [r10712+116640];
ld.shared.u32 r5118, [r10712+116644];
ld.shared.u32 r3984, [r10712+122472];
ld.shared.u32 r3990, [r10712+122476];
ld.shared.u32 r4592, [r10712+128304];
ld.shared.u32 r4598, [r10712+128308];
ld.shared.u32 r5200, [r10712+134136];
ld.shared.u32 r5206, [r10712+134140];
ld.shared.u32 r4072, [r10712+139968];
ld.shared.u32 r4078, [r10712+139972];
ld.shared.u32 r4680, [r10712+145800];
ld.shared.u32 r4686, [r10712+145804];
ld.shared.u32 r5288, [r10712+151632];
ld.shared.u32 r5294, [r10712+151636];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r3890, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r3978, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r4066, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r4154, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r4155, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r4156, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r4157, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r4160, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r4234, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r4322, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r4410, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r4498, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r4586, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r4674, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r4762, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r4763, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r4764, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r4765, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r4768, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r4842, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r4930, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r5018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r5106, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r5194, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r5282, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r5370, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r5371, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r5372, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r5373, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r5376, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r5450, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r5538, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r5626, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f898;
cvt.rn.f16.f32 high, f898;
mov.b32 r5714, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f900;
cvt.rn.f16.f32 high, f900;
mov.b32 r5715, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f902;
cvt.rn.f16.f32 high, f902;
mov.b32 r5716, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f904;
cvt.rn.f16.f32 high, f904;
mov.b32 r5717, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r5718, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r5719, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f910;
cvt.rn.f16.f32 high, f910;
mov.b32 r5720, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f912;
cvt.rn.f16.f32 high, f912;
mov.b32 r5721, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f914;
cvt.rn.f16.f32 high, f914;
mov.b32 r5722, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f916;
cvt.rn.f16.f32 high, f916;
mov.b32 r5723, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r5724, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r5725, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f922;
cvt.rn.f16.f32 high, f922;
mov.b32 r5726, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f924;
cvt.rn.f16.f32 high, f924;
mov.b32 r5727, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f926;
cvt.rn.f16.f32 high, f926;
mov.b32 r5728, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f928;
cvt.rn.f16.f32 high, f928;
mov.b32 r5729, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f934;
cvt.rn.f16.f32 high, f934;
mov.b32 r5732, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f936;
cvt.rn.f16.f32 high, f936;
mov.b32 r5733, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r5736, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
mov.b32 r5737, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f950;
cvt.rn.f16.f32 high, f950;
mov.b32 r5740, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f952;
cvt.rn.f16.f32 high, f952;
mov.b32 r5741, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f958;
cvt.rn.f16.f32 high, f958;
mov.b32 r5744, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f960;
cvt.rn.f16.f32 high, f960;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6022, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6110, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6198, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6286, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6374, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6462, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6550, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6638, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6726, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
mul.wide.u32 rd4, r10709, 795364315;
shr.u64 rd5, rd4, 32;
cvt.u32.u64 r10713, rd5;
sub.s32 r10714, r10709, r10713;
shr.u32 r10715, r10714, 1;
add.s32 r10716, r10715, r10713;
shr.u32 r10717, r10716, 4;
cvt.rn.f32.u32 f1040, r10717;
mul.f32 f1041, f1040, 0f3C0D3654;
cos.approx.f32 f673, f1041;
sin.approx.f32 f1042, f1041;
neg.f32 f674, f1042;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f673;
cvt.rn.f16.f32 high, f674;
mov.b32 r6814, {low, high};
}
mul.lo.s32 r10718, r10717, 27;
sub.s32 r10719, r10709, r10718;
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
shl.b32 r10720, r10719, 3;
add.s32 r10721, r10710, r10720;
barrier.sync 0;
mad.lo.s32 r10722, r10717, 5832, r10721;
st.shared.u32 [r10722], r6029;
st.shared.u32 [r10722+4], r6035;
st.shared.u32 [r10722+216], r6826;
st.shared.u32 [r10722+220], r6833;
st.shared.u32 [r10722+432], r6863;
st.shared.u32 [r10722+436], r6870;
st.shared.u32 [r10722+648], r6900;
st.shared.u32 [r10722+652], r6907;
st.shared.u32 [r10722+864], r6937;
st.shared.u32 [r10722+868], r6944;
st.shared.u32 [r10722+1080], r6974;
st.shared.u32 [r10722+1084], r6981;
st.shared.u32 [r10722+1296], r7011;
st.shared.u32 [r10722+1300], r7018;
st.shared.u32 [r10722+1512], r7048;
st.shared.u32 [r10722+1516], r7055;
st.shared.u32 [r10722+1728], r7085;
st.shared.u32 [r10722+1732], r7092;
st.shared.u32 [r10722+1944], r7122;
st.shared.u32 [r10722+1948], r7129;
st.shared.u32 [r10722+2160], r7159;
st.shared.u32 [r10722+2164], r7166;
st.shared.u32 [r10722+2376], r7196;
st.shared.u32 [r10722+2380], r7203;
st.shared.u32 [r10722+2592], r7233;
st.shared.u32 [r10722+2596], r7240;
st.shared.u32 [r10722+2808], r7270;
st.shared.u32 [r10722+2812], r7277;
st.shared.u32 [r10722+3024], r7307;
st.shared.u32 [r10722+3028], r7314;
st.shared.u32 [r10722+3240], r7344;
st.shared.u32 [r10722+3244], r7351;
st.shared.u32 [r10722+3456], r7381;
st.shared.u32 [r10722+3460], r7388;
st.shared.u32 [r10722+3672], r7418;
st.shared.u32 [r10722+3676], r7425;
st.shared.u32 [r10722+3888], r7455;
st.shared.u32 [r10722+3892], r7462;
st.shared.u32 [r10722+4104], r7492;
st.shared.u32 [r10722+4108], r7499;
st.shared.u32 [r10722+4320], r7529;
st.shared.u32 [r10722+4324], r7536;
st.shared.u32 [r10722+4536], r7566;
st.shared.u32 [r10722+4540], r7573;
st.shared.u32 [r10722+4752], r7603;
st.shared.u32 [r10722+4756], r7610;
st.shared.u32 [r10722+4968], r7640;
st.shared.u32 [r10722+4972], r7647;
st.shared.u32 [r10722+5184], r7677;
st.shared.u32 [r10722+5188], r7684;
st.shared.u32 [r10722+5400], r7714;
st.shared.u32 [r10722+5404], r7721;
st.shared.u32 [r10722+5616], r7751;
st.shared.u32 [r10722+5620], r7758;
barrier.sync 0;
ld.shared.u32 r7787, [r10712];
ld.shared.u32 r7793, [r10712+4];
ld.shared.u32 r8395, [r10712+5832];
ld.shared.u32 r8401, [r10712+5836];
ld.shared.u32 r9003, [r10712+11664];
ld.shared.u32 r9009, [r10712+11668];
ld.shared.u32 r7875, [r10712+17496];
ld.shared.u32 r7881, [r10712+17500];
ld.shared.u32 r8483, [r10712+23328];
ld.shared.u32 r8489, [r10712+23332];
ld.shared.u32 r9091, [r10712+29160];
ld.shared.u32 r9097, [r10712+29164];
ld.shared.u32 r7963, [r10712+34992];
ld.shared.u32 r7969, [r10712+34996];
ld.shared.u32 r8571, [r10712+40824];
ld.shared.u32 r8577, [r10712+40828];
ld.shared.u32 r9179, [r10712+46656];
ld.shared.u32 r9185, [r10712+46660];
ld.shared.u32 r7784, [r10712+52488];
ld.shared.u32 r7790, [r10712+52492];
ld.shared.u32 r8392, [r10712+58320];
ld.shared.u32 r8398, [r10712+58324];
ld.shared.u32 r9000, [r10712+64152];
ld.shared.u32 r9006, [r10712+64156];
ld.shared.u32 r7872, [r10712+69984];
ld.shared.u32 r7878, [r10712+69988];
ld.shared.u32 r8480, [r10712+75816];
ld.shared.u32 r8486, [r10712+75820];
ld.shared.u32 r9088, [r10712+81648];
ld.shared.u32 r9094, [r10712+81652];
ld.shared.u32 r7960, [r10712+87480];
ld.shared.u32 r7966, [r10712+87484];
ld.shared.u32 r8568, [r10712+93312];
ld.shared.u32 r8574, [r10712+93316];
ld.shared.u32 r9176, [r10712+99144];
ld.shared.u32 r9182, [r10712+99148];
ld.shared.u32 r7785, [r10712+104976];
ld.shared.u32 r7791, [r10712+104980];
ld.shared.u32 r8393, [r10712+110808];
ld.shared.u32 r8399, [r10712+110812];
ld.shared.u32 r9001, [r10712+116640];
ld.shared.u32 r9007, [r10712+116644];
ld.shared.u32 r7873, [r10712+122472];
ld.shared.u32 r7879, [r10712+122476];
ld.shared.u32 r8481, [r10712+128304];
ld.shared.u32 r8487, [r10712+128308];
ld.shared.u32 r9089, [r10712+134136];
ld.shared.u32 r9095, [r10712+134140];
ld.shared.u32 r7961, [r10712+139968];
ld.shared.u32 r7967, [r10712+139972];
ld.shared.u32 r8569, [r10712+145800];
ld.shared.u32 r8575, [r10712+145804];
ld.shared.u32 r9177, [r10712+151632];
ld.shared.u32 r9183, [r10712+151636];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r7779, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r7780, {low, high};
}
{
neg.f16x2 r7781, r7780;
}
{
add.f16x2 r7783, r7784, r7785;
}
{
add.f16x2 r7786, r7787, r7783;
}
{
add.f16x2 r7789, r7790, r7791;
}
{
add.f16x2 r7792, r7793, r7789;
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
add.f16x2 r7810, r7801, r7807;
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
sub.f16x2 r7828, r7819, r7825;
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
sub.f16x2 r7846, r7837, r7843;
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
add.f16x2 r7864, r7855, r7861;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r7867, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r7868, {low, high};
}
{
neg.f16x2 r7869, r7868;
}
{
add.f16x2 r7871, r7872, r7873;
}
{
add.f16x2 r7874, r7875, r7871;
}
{
add.f16x2 r7877, r7878, r7879;
}
{
add.f16x2 r7880, r7881, r7877;
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
add.f16x2 r7898, r7889, r7895;
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
sub.f16x2 r7916, r7907, r7913;
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
sub.f16x2 r7934, r7925, r7931;
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
add.f16x2 r7952, r7943, r7949;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r7955, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r7956, {low, high};
}
{
neg.f16x2 r7957, r7956;
}
{
add.f16x2 r7959, r7960, r7961;
}
{
add.f16x2 r7962, r7963, r7959;
}
{
add.f16x2 r7965, r7966, r7967;
}
{
add.f16x2 r7968, r7969, r7965;
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
add.f16x2 r7986, r7977, r7983;
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
sub.f16x2 r8004, r7995, r8001;
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
sub.f16x2 r8022, r8013, r8019;
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
add.f16x2 r8040, r8031, r8037;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r8043, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r8044, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r8045, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r8046, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r8049, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
mov.b32 r8050, {low, high};
}
{
mul.f16x2 r8059, r7898, r8043;
}
{
mul.f16x2 r8062, r7934, r8044;
}
{
sub.f16x2 r8065, r8059, r8062;
}
{
mul.f16x2 r8068, r7898, r8044;
}
{
fma.rn.f16x2 r8071, r7934, r8043, r8068;
}
{
mul.f16x2 r8075, r7986, r8045;
}
{
mul.f16x2 r8078, r8022, r8046;
}
{
sub.f16x2 r8081, r8075, r8078;
}
{
mul.f16x2 r8084, r7986, r8046;
}
{
fma.rn.f16x2 r8087, r8022, r8045, r8084;
}
{
mul.f16x2 r8091, r7916, r8045;
}
{
mul.f16x2 r8094, r7952, r8046;
}
{
sub.f16x2 r8097, r8091, r8094;
}
{
mul.f16x2 r8100, r7916, r8046;
}
{
fma.rn.f16x2 r8103, r7952, r8045, r8100;
}
{
mul.f16x2 r8107, r8004, r8049;
}
{
mul.f16x2 r8110, r8040, r8050;
}
{
sub.f16x2 r8113, r8107, r8110;
}
{
mul.f16x2 r8116, r8004, r8050;
}
{
fma.rn.f16x2 r8119, r8040, r8049, r8116;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r8123, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r8124, {low, high};
}
{
neg.f16x2 r8125, r8124;
}
{
add.f16x2 r8127, r7874, r7962;
}
{
add.f16x2 r8130, r7786, r8127;
}
{
add.f16x2 r8133, r7880, r7968;
}
{
add.f16x2 r8136, r7792, r8133;
}
{
add.f16x2 r8139, r7874, r7962;
}
{
mul.f16x2 r8142, r8139, r8123;
}
{
add.f16x2 r8145, r7786, r8142;
}
{
sub.f16x2 r8148, r7880, r7968;
}
{
mul.f16x2 r8151, r8148, r8125;
}
{
add.f16x2 r8154, r8145, r8151;
}
{
add.f16x2 r8157, r7874, r7962;
}
{
mul.f16x2 r8160, r8157, r8123;
}
{
add.f16x2 r8163, r7786, r8160;
}
{
sub.f16x2 r8166, r7880, r7968;
}
{
mul.f16x2 r8169, r8166, r8125;
}
{
sub.f16x2 r8172, r8163, r8169;
}
{
add.f16x2 r8175, r7880, r7968;
}
{
mul.f16x2 r8178, r8175, r8123;
}
{
add.f16x2 r8181, r7792, r8178;
}
{
sub.f16x2 r8184, r7874, r7962;
}
{
mul.f16x2 r8187, r8184, r8125;
}
{
sub.f16x2 r8190, r8181, r8187;
}
{
add.f16x2 r8193, r7880, r7968;
}
{
mul.f16x2 r8196, r8193, r8123;
}
{
add.f16x2 r8199, r7792, r8196;
}
{
sub.f16x2 r8202, r7874, r7962;
}
{
mul.f16x2 r8205, r8202, r8125;
}
{
add.f16x2 r8208, r8199, r8205;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r8211, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r8212, {low, high};
}
{
neg.f16x2 r8213, r8212;
}
{
add.f16x2 r8215, r8065, r8081;
}
{
add.f16x2 r8218, r7810, r8215;
}
{
add.f16x2 r8221, r8071, r8087;
}
{
add.f16x2 r8224, r7846, r8221;
}
{
add.f16x2 r8227, r8065, r8081;
}
{
mul.f16x2 r8230, r8227, r8211;
}
{
add.f16x2 r8233, r7810, r8230;
}
{
sub.f16x2 r8236, r8071, r8087;
}
{
mul.f16x2 r8239, r8236, r8213;
}
{
add.f16x2 r8242, r8233, r8239;
}
{
add.f16x2 r8245, r8065, r8081;
}
{
mul.f16x2 r8248, r8245, r8211;
}
{
add.f16x2 r8251, r7810, r8248;
}
{
sub.f16x2 r8254, r8071, r8087;
}
{
mul.f16x2 r8257, r8254, r8213;
}
{
sub.f16x2 r8260, r8251, r8257;
}
{
add.f16x2 r8263, r8071, r8087;
}
{
mul.f16x2 r8266, r8263, r8211;
}
{
add.f16x2 r8269, r7846, r8266;
}
{
sub.f16x2 r8272, r8065, r8081;
}
{
mul.f16x2 r8275, r8272, r8213;
}
{
sub.f16x2 r8278, r8269, r8275;
}
{
add.f16x2 r8281, r8071, r8087;
}
{
mul.f16x2 r8284, r8281, r8211;
}
{
add.f16x2 r8287, r7846, r8284;
}
{
sub.f16x2 r8290, r8065, r8081;
}
{
mul.f16x2 r8293, r8290, r8213;
}
{
add.f16x2 r8296, r8287, r8293;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r8299, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r8300, {low, high};
}
{
neg.f16x2 r8301, r8300;
}
{
add.f16x2 r8303, r8097, r8113;
}
{
add.f16x2 r8306, r7828, r8303;
}
{
add.f16x2 r8309, r8103, r8119;
}
{
add.f16x2 r8312, r7864, r8309;
}
{
add.f16x2 r8315, r8097, r8113;
}
{
mul.f16x2 r8318, r8315, r8299;
}
{
add.f16x2 r8321, r7828, r8318;
}
{
sub.f16x2 r8324, r8103, r8119;
}
{
mul.f16x2 r8327, r8324, r8301;
}
{
add.f16x2 r8330, r8321, r8327;
}
{
add.f16x2 r8333, r8097, r8113;
}
{
mul.f16x2 r8336, r8333, r8299;
}
{
add.f16x2 r8339, r7828, r8336;
}
{
sub.f16x2 r8342, r8103, r8119;
}
{
mul.f16x2 r8345, r8342, r8301;
}
{
sub.f16x2 r8348, r8339, r8345;
}
{
add.f16x2 r8351, r8103, r8119;
}
{
mul.f16x2 r8354, r8351, r8299;
}
{
add.f16x2 r8357, r7864, r8354;
}
{
sub.f16x2 r8360, r8097, r8113;
}
{
mul.f16x2 r8363, r8360, r8301;
}
{
sub.f16x2 r8366, r8357, r8363;
}
{
add.f16x2 r8369, r8103, r8119;
}
{
mul.f16x2 r8372, r8369, r8299;
}
{
add.f16x2 r8375, r7864, r8372;
}
{
sub.f16x2 r8378, r8097, r8113;
}
{
mul.f16x2 r8381, r8378, r8301;
}
{
add.f16x2 r8384, r8375, r8381;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r8387, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r8388, {low, high};
}
{
neg.f16x2 r8389, r8388;
}
{
add.f16x2 r8391, r8392, r8393;
}
{
add.f16x2 r8394, r8395, r8391;
}
{
add.f16x2 r8397, r8398, r8399;
}
{
add.f16x2 r8400, r8401, r8397;
}
{
add.f16x2 r8403, r8392, r8393;
}
{
mul.f16x2 r8406, r8403, r8387;
}
{
add.f16x2 r8409, r8395, r8406;
}
{
sub.f16x2 r8412, r8398, r8399;
}
{
mul.f16x2 r8415, r8412, r8389;
}
{
add.f16x2 r8418, r8409, r8415;
}
{
add.f16x2 r8421, r8392, r8393;
}
{
mul.f16x2 r8424, r8421, r8387;
}
{
add.f16x2 r8427, r8395, r8424;
}
{
sub.f16x2 r8430, r8398, r8399;
}
{
mul.f16x2 r8433, r8430, r8389;
}
{
sub.f16x2 r8436, r8427, r8433;
}
{
add.f16x2 r8439, r8398, r8399;
}
{
mul.f16x2 r8442, r8439, r8387;
}
{
add.f16x2 r8445, r8401, r8442;
}
{
sub.f16x2 r8448, r8392, r8393;
}
{
mul.f16x2 r8451, r8448, r8389;
}
{
sub.f16x2 r8454, r8445, r8451;
}
{
add.f16x2 r8457, r8398, r8399;
}
{
mul.f16x2 r8460, r8457, r8387;
}
{
add.f16x2 r8463, r8401, r8460;
}
{
sub.f16x2 r8466, r8392, r8393;
}
{
mul.f16x2 r8469, r8466, r8389;
}
{
add.f16x2 r8472, r8463, r8469;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r8475, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r8476, {low, high};
}
{
neg.f16x2 r8477, r8476;
}
{
add.f16x2 r8479, r8480, r8481;
}
{
add.f16x2 r8482, r8483, r8479;
}
{
add.f16x2 r8485, r8486, r8487;
}
{
add.f16x2 r8488, r8489, r8485;
}
{
add.f16x2 r8491, r8480, r8481;
}
{
mul.f16x2 r8494, r8491, r8475;
}
{
add.f16x2 r8497, r8483, r8494;
}
{
sub.f16x2 r8500, r8486, r8487;
}
{
mul.f16x2 r8503, r8500, r8477;
}
{
add.f16x2 r8506, r8497, r8503;
}
{
add.f16x2 r8509, r8480, r8481;
}
{
mul.f16x2 r8512, r8509, r8475;
}
{
add.f16x2 r8515, r8483, r8512;
}
{
sub.f16x2 r8518, r8486, r8487;
}
{
mul.f16x2 r8521, r8518, r8477;
}
{
sub.f16x2 r8524, r8515, r8521;
}
{
add.f16x2 r8527, r8486, r8487;
}
{
mul.f16x2 r8530, r8527, r8475;
}
{
add.f16x2 r8533, r8489, r8530;
}
{
sub.f16x2 r8536, r8480, r8481;
}
{
mul.f16x2 r8539, r8536, r8477;
}
{
sub.f16x2 r8542, r8533, r8539;
}
{
add.f16x2 r8545, r8486, r8487;
}
{
mul.f16x2 r8548, r8545, r8475;
}
{
add.f16x2 r8551, r8489, r8548;
}
{
sub.f16x2 r8554, r8480, r8481;
}
{
mul.f16x2 r8557, r8554, r8477;
}
{
add.f16x2 r8560, r8551, r8557;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r8563, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r8564, {low, high};
}
{
neg.f16x2 r8565, r8564;
}
{
add.f16x2 r8567, r8568, r8569;
}
{
add.f16x2 r8570, r8571, r8567;
}
{
add.f16x2 r8573, r8574, r8575;
}
{
add.f16x2 r8576, r8577, r8573;
}
{
add.f16x2 r8579, r8568, r8569;
}
{
mul.f16x2 r8582, r8579, r8563;
}
{
add.f16x2 r8585, r8571, r8582;
}
{
sub.f16x2 r8588, r8574, r8575;
}
{
mul.f16x2 r8591, r8588, r8565;
}
{
add.f16x2 r8594, r8585, r8591;
}
{
add.f16x2 r8597, r8568, r8569;
}
{
mul.f16x2 r8600, r8597, r8563;
}
{
add.f16x2 r8603, r8571, r8600;
}
{
sub.f16x2 r8606, r8574, r8575;
}
{
mul.f16x2 r8609, r8606, r8565;
}
{
sub.f16x2 r8612, r8603, r8609;
}
{
add.f16x2 r8615, r8574, r8575;
}
{
mul.f16x2 r8618, r8615, r8563;
}
{
add.f16x2 r8621, r8577, r8618;
}
{
sub.f16x2 r8624, r8568, r8569;
}
{
mul.f16x2 r8627, r8624, r8565;
}
{
sub.f16x2 r8630, r8621, r8627;
}
{
add.f16x2 r8633, r8574, r8575;
}
{
mul.f16x2 r8636, r8633, r8563;
}
{
add.f16x2 r8639, r8577, r8636;
}
{
sub.f16x2 r8642, r8568, r8569;
}
{
mul.f16x2 r8645, r8642, r8565;
}
{
add.f16x2 r8648, r8639, r8645;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r8651, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r8652, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r8653, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r8654, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r8657, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
mov.b32 r8658, {low, high};
}
{
mul.f16x2 r8667, r8506, r8651;
}
{
mul.f16x2 r8670, r8542, r8652;
}
{
sub.f16x2 r8673, r8667, r8670;
}
{
mul.f16x2 r8676, r8506, r8652;
}
{
fma.rn.f16x2 r8679, r8542, r8651, r8676;
}
{
mul.f16x2 r8683, r8594, r8653;
}
{
mul.f16x2 r8686, r8630, r8654;
}
{
sub.f16x2 r8689, r8683, r8686;
}
{
mul.f16x2 r8692, r8594, r8654;
}
{
fma.rn.f16x2 r8695, r8630, r8653, r8692;
}
{
mul.f16x2 r8699, r8524, r8653;
}
{
mul.f16x2 r8702, r8560, r8654;
}
{
sub.f16x2 r8705, r8699, r8702;
}
{
mul.f16x2 r8708, r8524, r8654;
}
{
fma.rn.f16x2 r8711, r8560, r8653, r8708;
}
{
mul.f16x2 r8715, r8612, r8657;
}
{
mul.f16x2 r8718, r8648, r8658;
}
{
sub.f16x2 r8721, r8715, r8718;
}
{
mul.f16x2 r8724, r8612, r8658;
}
{
fma.rn.f16x2 r8727, r8648, r8657, r8724;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r8731, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r8732, {low, high};
}
{
neg.f16x2 r8733, r8732;
}
{
add.f16x2 r8735, r8482, r8570;
}
{
add.f16x2 r8738, r8394, r8735;
}
{
add.f16x2 r8741, r8488, r8576;
}
{
add.f16x2 r8744, r8400, r8741;
}
{
add.f16x2 r8747, r8482, r8570;
}
{
mul.f16x2 r8750, r8747, r8731;
}
{
add.f16x2 r8753, r8394, r8750;
}
{
sub.f16x2 r8756, r8488, r8576;
}
{
mul.f16x2 r8759, r8756, r8733;
}
{
add.f16x2 r8762, r8753, r8759;
}
{
add.f16x2 r8765, r8482, r8570;
}
{
mul.f16x2 r8768, r8765, r8731;
}
{
add.f16x2 r8771, r8394, r8768;
}
{
sub.f16x2 r8774, r8488, r8576;
}
{
mul.f16x2 r8777, r8774, r8733;
}
{
sub.f16x2 r8780, r8771, r8777;
}
{
add.f16x2 r8783, r8488, r8576;
}
{
mul.f16x2 r8786, r8783, r8731;
}
{
add.f16x2 r8789, r8400, r8786;
}
{
sub.f16x2 r8792, r8482, r8570;
}
{
mul.f16x2 r8795, r8792, r8733;
}
{
sub.f16x2 r8798, r8789, r8795;
}
{
add.f16x2 r8801, r8488, r8576;
}
{
mul.f16x2 r8804, r8801, r8731;
}
{
add.f16x2 r8807, r8400, r8804;
}
{
sub.f16x2 r8810, r8482, r8570;
}
{
mul.f16x2 r8813, r8810, r8733;
}
{
add.f16x2 r8816, r8807, r8813;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r8819, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r8820, {low, high};
}
{
neg.f16x2 r8821, r8820;
}
{
add.f16x2 r8823, r8673, r8689;
}
{
add.f16x2 r8826, r8418, r8823;
}
{
add.f16x2 r8829, r8679, r8695;
}
{
add.f16x2 r8832, r8454, r8829;
}
{
add.f16x2 r8835, r8673, r8689;
}
{
mul.f16x2 r8838, r8835, r8819;
}
{
add.f16x2 r8841, r8418, r8838;
}
{
sub.f16x2 r8844, r8679, r8695;
}
{
mul.f16x2 r8847, r8844, r8821;
}
{
add.f16x2 r8850, r8841, r8847;
}
{
add.f16x2 r8853, r8673, r8689;
}
{
mul.f16x2 r8856, r8853, r8819;
}
{
add.f16x2 r8859, r8418, r8856;
}
{
sub.f16x2 r8862, r8679, r8695;
}
{
mul.f16x2 r8865, r8862, r8821;
}
{
sub.f16x2 r8868, r8859, r8865;
}
{
add.f16x2 r8871, r8679, r8695;
}
{
mul.f16x2 r8874, r8871, r8819;
}
{
add.f16x2 r8877, r8454, r8874;
}
{
sub.f16x2 r8880, r8673, r8689;
}
{
mul.f16x2 r8883, r8880, r8821;
}
{
sub.f16x2 r8886, r8877, r8883;
}
{
add.f16x2 r8889, r8679, r8695;
}
{
mul.f16x2 r8892, r8889, r8819;
}
{
add.f16x2 r8895, r8454, r8892;
}
{
sub.f16x2 r8898, r8673, r8689;
}
{
mul.f16x2 r8901, r8898, r8821;
}
{
add.f16x2 r8904, r8895, r8901;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r8907, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r8908, {low, high};
}
{
neg.f16x2 r8909, r8908;
}
{
add.f16x2 r8911, r8705, r8721;
}
{
add.f16x2 r8914, r8436, r8911;
}
{
add.f16x2 r8917, r8711, r8727;
}
{
add.f16x2 r8920, r8472, r8917;
}
{
add.f16x2 r8923, r8705, r8721;
}
{
mul.f16x2 r8926, r8923, r8907;
}
{
add.f16x2 r8929, r8436, r8926;
}
{
sub.f16x2 r8932, r8711, r8727;
}
{
mul.f16x2 r8935, r8932, r8909;
}
{
add.f16x2 r8938, r8929, r8935;
}
{
add.f16x2 r8941, r8705, r8721;
}
{
mul.f16x2 r8944, r8941, r8907;
}
{
add.f16x2 r8947, r8436, r8944;
}
{
sub.f16x2 r8950, r8711, r8727;
}
{
mul.f16x2 r8953, r8950, r8909;
}
{
sub.f16x2 r8956, r8947, r8953;
}
{
add.f16x2 r8959, r8711, r8727;
}
{
mul.f16x2 r8962, r8959, r8907;
}
{
add.f16x2 r8965, r8472, r8962;
}
{
sub.f16x2 r8968, r8705, r8721;
}
{
mul.f16x2 r8971, r8968, r8909;
}
{
sub.f16x2 r8974, r8965, r8971;
}
{
add.f16x2 r8977, r8711, r8727;
}
{
mul.f16x2 r8980, r8977, r8907;
}
{
add.f16x2 r8983, r8472, r8980;
}
{
sub.f16x2 r8986, r8705, r8721;
}
{
mul.f16x2 r8989, r8986, r8909;
}
{
add.f16x2 r8992, r8983, r8989;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r8995, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r8996, {low, high};
}
{
neg.f16x2 r8997, r8996;
}
{
add.f16x2 r8999, r9000, r9001;
}
{
add.f16x2 r9002, r9003, r8999;
}
{
add.f16x2 r9005, r9006, r9007;
}
{
add.f16x2 r9008, r9009, r9005;
}
{
add.f16x2 r9011, r9000, r9001;
}
{
mul.f16x2 r9014, r9011, r8995;
}
{
add.f16x2 r9017, r9003, r9014;
}
{
sub.f16x2 r9020, r9006, r9007;
}
{
mul.f16x2 r9023, r9020, r8997;
}
{
add.f16x2 r9026, r9017, r9023;
}
{
add.f16x2 r9029, r9000, r9001;
}
{
mul.f16x2 r9032, r9029, r8995;
}
{
add.f16x2 r9035, r9003, r9032;
}
{
sub.f16x2 r9038, r9006, r9007;
}
{
mul.f16x2 r9041, r9038, r8997;
}
{
sub.f16x2 r9044, r9035, r9041;
}
{
add.f16x2 r9047, r9006, r9007;
}
{
mul.f16x2 r9050, r9047, r8995;
}
{
add.f16x2 r9053, r9009, r9050;
}
{
sub.f16x2 r9056, r9000, r9001;
}
{
mul.f16x2 r9059, r9056, r8997;
}
{
sub.f16x2 r9062, r9053, r9059;
}
{
add.f16x2 r9065, r9006, r9007;
}
{
mul.f16x2 r9068, r9065, r8995;
}
{
add.f16x2 r9071, r9009, r9068;
}
{
sub.f16x2 r9074, r9000, r9001;
}
{
mul.f16x2 r9077, r9074, r8997;
}
{
add.f16x2 r9080, r9071, r9077;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r9083, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r9084, {low, high};
}
{
neg.f16x2 r9085, r9084;
}
{
add.f16x2 r9087, r9088, r9089;
}
{
add.f16x2 r9090, r9091, r9087;
}
{
add.f16x2 r9093, r9094, r9095;
}
{
add.f16x2 r9096, r9097, r9093;
}
{
add.f16x2 r9099, r9088, r9089;
}
{
mul.f16x2 r9102, r9099, r9083;
}
{
add.f16x2 r9105, r9091, r9102;
}
{
sub.f16x2 r9108, r9094, r9095;
}
{
mul.f16x2 r9111, r9108, r9085;
}
{
add.f16x2 r9114, r9105, r9111;
}
{
add.f16x2 r9117, r9088, r9089;
}
{
mul.f16x2 r9120, r9117, r9083;
}
{
add.f16x2 r9123, r9091, r9120;
}
{
sub.f16x2 r9126, r9094, r9095;
}
{
mul.f16x2 r9129, r9126, r9085;
}
{
sub.f16x2 r9132, r9123, r9129;
}
{
add.f16x2 r9135, r9094, r9095;
}
{
mul.f16x2 r9138, r9135, r9083;
}
{
add.f16x2 r9141, r9097, r9138;
}
{
sub.f16x2 r9144, r9088, r9089;
}
{
mul.f16x2 r9147, r9144, r9085;
}
{
sub.f16x2 r9150, r9141, r9147;
}
{
add.f16x2 r9153, r9094, r9095;
}
{
mul.f16x2 r9156, r9153, r9083;
}
{
add.f16x2 r9159, r9097, r9156;
}
{
sub.f16x2 r9162, r9088, r9089;
}
{
mul.f16x2 r9165, r9162, r9085;
}
{
add.f16x2 r9168, r9159, r9165;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r9171, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r9172, {low, high};
}
{
neg.f16x2 r9173, r9172;
}
{
add.f16x2 r9175, r9176, r9177;
}
{
add.f16x2 r9178, r9179, r9175;
}
{
add.f16x2 r9181, r9182, r9183;
}
{
add.f16x2 r9184, r9185, r9181;
}
{
add.f16x2 r9187, r9176, r9177;
}
{
mul.f16x2 r9190, r9187, r9171;
}
{
add.f16x2 r9193, r9179, r9190;
}
{
sub.f16x2 r9196, r9182, r9183;
}
{
mul.f16x2 r9199, r9196, r9173;
}
{
add.f16x2 r9202, r9193, r9199;
}
{
add.f16x2 r9205, r9176, r9177;
}
{
mul.f16x2 r9208, r9205, r9171;
}
{
add.f16x2 r9211, r9179, r9208;
}
{
sub.f16x2 r9214, r9182, r9183;
}
{
mul.f16x2 r9217, r9214, r9173;
}
{
sub.f16x2 r9220, r9211, r9217;
}
{
add.f16x2 r9223, r9182, r9183;
}
{
mul.f16x2 r9226, r9223, r9171;
}
{
add.f16x2 r9229, r9185, r9226;
}
{
sub.f16x2 r9232, r9176, r9177;
}
{
mul.f16x2 r9235, r9232, r9173;
}
{
sub.f16x2 r9238, r9229, r9235;
}
{
add.f16x2 r9241, r9182, r9183;
}
{
mul.f16x2 r9244, r9241, r9171;
}
{
add.f16x2 r9247, r9185, r9244;
}
{
sub.f16x2 r9250, r9176, r9177;
}
{
mul.f16x2 r9253, r9250, r9173;
}
{
add.f16x2 r9256, r9247, r9253;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r9259, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r9260, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r9261, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r9262, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r9265, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
mov.b32 r9266, {low, high};
}
{
mul.f16x2 r9275, r9114, r9259;
}
{
mul.f16x2 r9278, r9150, r9260;
}
{
sub.f16x2 r9281, r9275, r9278;
}
{
mul.f16x2 r9284, r9114, r9260;
}
{
fma.rn.f16x2 r9287, r9150, r9259, r9284;
}
{
mul.f16x2 r9291, r9202, r9261;
}
{
mul.f16x2 r9294, r9238, r9262;
}
{
sub.f16x2 r9297, r9291, r9294;
}
{
mul.f16x2 r9300, r9202, r9262;
}
{
fma.rn.f16x2 r9303, r9238, r9261, r9300;
}
{
mul.f16x2 r9307, r9132, r9261;
}
{
mul.f16x2 r9310, r9168, r9262;
}
{
sub.f16x2 r9313, r9307, r9310;
}
{
mul.f16x2 r9316, r9132, r9262;
}
{
fma.rn.f16x2 r9319, r9168, r9261, r9316;
}
{
mul.f16x2 r9323, r9220, r9265;
}
{
mul.f16x2 r9326, r9256, r9266;
}
{
sub.f16x2 r9329, r9323, r9326;
}
{
mul.f16x2 r9332, r9220, r9266;
}
{
fma.rn.f16x2 r9335, r9256, r9265, r9332;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r9339, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r9340, {low, high};
}
{
neg.f16x2 r9341, r9340;
}
{
add.f16x2 r9343, r9090, r9178;
}
{
add.f16x2 r9346, r9002, r9343;
}
{
add.f16x2 r9349, r9096, r9184;
}
{
add.f16x2 r9352, r9008, r9349;
}
{
add.f16x2 r9355, r9090, r9178;
}
{
mul.f16x2 r9358, r9355, r9339;
}
{
add.f16x2 r9361, r9002, r9358;
}
{
sub.f16x2 r9364, r9096, r9184;
}
{
mul.f16x2 r9367, r9364, r9341;
}
{
add.f16x2 r9370, r9361, r9367;
}
{
add.f16x2 r9373, r9090, r9178;
}
{
mul.f16x2 r9376, r9373, r9339;
}
{
add.f16x2 r9379, r9002, r9376;
}
{
sub.f16x2 r9382, r9096, r9184;
}
{
mul.f16x2 r9385, r9382, r9341;
}
{
sub.f16x2 r9388, r9379, r9385;
}
{
add.f16x2 r9391, r9096, r9184;
}
{
mul.f16x2 r9394, r9391, r9339;
}
{
add.f16x2 r9397, r9008, r9394;
}
{
sub.f16x2 r9400, r9090, r9178;
}
{
mul.f16x2 r9403, r9400, r9341;
}
{
sub.f16x2 r9406, r9397, r9403;
}
{
add.f16x2 r9409, r9096, r9184;
}
{
mul.f16x2 r9412, r9409, r9339;
}
{
add.f16x2 r9415, r9008, r9412;
}
{
sub.f16x2 r9418, r9090, r9178;
}
{
mul.f16x2 r9421, r9418, r9341;
}
{
add.f16x2 r9424, r9415, r9421;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r9427, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r9428, {low, high};
}
{
neg.f16x2 r9429, r9428;
}
{
add.f16x2 r9431, r9281, r9297;
}
{
add.f16x2 r9434, r9026, r9431;
}
{
add.f16x2 r9437, r9287, r9303;
}
{
add.f16x2 r9440, r9062, r9437;
}
{
add.f16x2 r9443, r9281, r9297;
}
{
mul.f16x2 r9446, r9443, r9427;
}
{
add.f16x2 r9449, r9026, r9446;
}
{
sub.f16x2 r9452, r9287, r9303;
}
{
mul.f16x2 r9455, r9452, r9429;
}
{
add.f16x2 r9458, r9449, r9455;
}
{
add.f16x2 r9461, r9281, r9297;
}
{
mul.f16x2 r9464, r9461, r9427;
}
{
add.f16x2 r9467, r9026, r9464;
}
{
sub.f16x2 r9470, r9287, r9303;
}
{
mul.f16x2 r9473, r9470, r9429;
}
{
sub.f16x2 r9476, r9467, r9473;
}
{
add.f16x2 r9479, r9287, r9303;
}
{
mul.f16x2 r9482, r9479, r9427;
}
{
add.f16x2 r9485, r9062, r9482;
}
{
sub.f16x2 r9488, r9281, r9297;
}
{
mul.f16x2 r9491, r9488, r9429;
}
{
sub.f16x2 r9494, r9485, r9491;
}
{
add.f16x2 r9497, r9287, r9303;
}
{
mul.f16x2 r9500, r9497, r9427;
}
{
add.f16x2 r9503, r9062, r9500;
}
{
sub.f16x2 r9506, r9281, r9297;
}
{
mul.f16x2 r9509, r9506, r9429;
}
{
add.f16x2 r9512, r9503, r9509;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r9515, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r9516, {low, high};
}
{
neg.f16x2 r9517, r9516;
}
{
add.f16x2 r9519, r9313, r9329;
}
{
add.f16x2 r9522, r9044, r9519;
}
{
add.f16x2 r9525, r9319, r9335;
}
{
add.f16x2 r9528, r9080, r9525;
}
{
add.f16x2 r9531, r9313, r9329;
}
{
mul.f16x2 r9534, r9531, r9515;
}
{
add.f16x2 r9537, r9044, r9534;
}
{
sub.f16x2 r9540, r9319, r9335;
}
{
mul.f16x2 r9543, r9540, r9517;
}
{
add.f16x2 r9546, r9537, r9543;
}
{
add.f16x2 r9549, r9313, r9329;
}
{
mul.f16x2 r9552, r9549, r9515;
}
{
add.f16x2 r9555, r9044, r9552;
}
{
sub.f16x2 r9558, r9319, r9335;
}
{
mul.f16x2 r9561, r9558, r9517;
}
{
sub.f16x2 r9564, r9555, r9561;
}
{
add.f16x2 r9567, r9319, r9335;
}
{
mul.f16x2 r9570, r9567, r9515;
}
{
add.f16x2 r9573, r9080, r9570;
}
{
sub.f16x2 r9576, r9313, r9329;
}
{
mul.f16x2 r9579, r9576, r9517;
}
{
sub.f16x2 r9582, r9573, r9579;
}
{
add.f16x2 r9585, r9319, r9335;
}
{
mul.f16x2 r9588, r9585, r9515;
}
{
add.f16x2 r9591, r9080, r9588;
}
{
sub.f16x2 r9594, r9313, r9329;
}
{
mul.f16x2 r9597, r9594, r9517;
}
{
add.f16x2 r9600, r9591, r9597;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f898;
cvt.rn.f16.f32 high, f898;
mov.b32 r9603, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f900;
cvt.rn.f16.f32 high, f900;
mov.b32 r9604, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f902;
cvt.rn.f16.f32 high, f902;
mov.b32 r9605, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f904;
cvt.rn.f16.f32 high, f904;
mov.b32 r9606, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r9607, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r9608, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f910;
cvt.rn.f16.f32 high, f910;
mov.b32 r9609, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f912;
cvt.rn.f16.f32 high, f912;
mov.b32 r9610, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f914;
cvt.rn.f16.f32 high, f914;
mov.b32 r9611, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f916;
cvt.rn.f16.f32 high, f916;
mov.b32 r9612, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r9613, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r9614, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f922;
cvt.rn.f16.f32 high, f922;
mov.b32 r9615, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f924;
cvt.rn.f16.f32 high, f924;
mov.b32 r9616, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f926;
cvt.rn.f16.f32 high, f926;
mov.b32 r9617, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f928;
cvt.rn.f16.f32 high, f928;
mov.b32 r9618, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f934;
cvt.rn.f16.f32 high, f934;
mov.b32 r9621, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f936;
cvt.rn.f16.f32 high, f936;
mov.b32 r9622, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r9625, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
mov.b32 r9626, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f950;
cvt.rn.f16.f32 high, f950;
mov.b32 r9629, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f952;
cvt.rn.f16.f32 high, f952;
mov.b32 r9630, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f958;
cvt.rn.f16.f32 high, f958;
mov.b32 r9633, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f960;
cvt.rn.f16.f32 high, f960;
mov.b32 r9634, {low, high};
}
{
mul.f16x2 r9655, r8826, r9603;
}
{
mul.f16x2 r9658, r8832, r9604;
}
{
sub.f16x2 r9661, r9655, r9658;
}
{
mul.f16x2 r9664, r8826, r9604;
}
{
fma.rn.f16x2 r9667, r8832, r9603, r9664;
}
{
mul.f16x2 r9671, r9434, r9605;
}
{
mul.f16x2 r9674, r9440, r9606;
}
{
sub.f16x2 r9677, r9671, r9674;
}
{
mul.f16x2 r9680, r9434, r9606;
}
{
fma.rn.f16x2 r9683, r9440, r9605, r9680;
}
{
mul.f16x2 r9687, r8914, r9605;
}
{
mul.f16x2 r9690, r8920, r9606;
}
{
sub.f16x2 r9693, r9687, r9690;
}
{
mul.f16x2 r9696, r8914, r9606;
}
{
fma.rn.f16x2 r9699, r8920, r9605, r9696;
}
{
mul.f16x2 r9703, r9522, r9609;
}
{
mul.f16x2 r9706, r9528, r9610;
}
{
sub.f16x2 r9709, r9703, r9706;
}
{
mul.f16x2 r9712, r9522, r9610;
}
{
fma.rn.f16x2 r9715, r9528, r9609, r9712;
}
{
mul.f16x2 r9719, r8762, r9607;
}
{
mul.f16x2 r9722, r8798, r9608;
}
{
sub.f16x2 r9725, r9719, r9722;
}
{
mul.f16x2 r9728, r8762, r9608;
}
{
fma.rn.f16x2 r9731, r8798, r9607, r9728;
}
{
mul.f16x2 r9735, r9370, r9613;
}
{
mul.f16x2 r9738, r9406, r9614;
}
{
sub.f16x2 r9741, r9735, r9738;
}
{
mul.f16x2 r9744, r9370, r9614;
}
{
fma.rn.f16x2 r9747, r9406, r9613, r9744;
}
{
mul.f16x2 r9751, r8850, r9609;
}
{
mul.f16x2 r9754, r8886, r9610;
}
{
sub.f16x2 r9757, r9751, r9754;
}
{
mul.f16x2 r9760, r8850, r9610;
}
{
fma.rn.f16x2 r9763, r8886, r9609, r9760;
}
{
mul.f16x2 r9767, r9458, r9617;
}
{
mul.f16x2 r9770, r9494, r9618;
}
{
sub.f16x2 r9773, r9767, r9770;
}
{
mul.f16x2 r9776, r9458, r9618;
}
{
fma.rn.f16x2 r9779, r9494, r9617, r9776;
}
{
mul.f16x2 r9783, r8938, r9611;
}
{
mul.f16x2 r9786, r8974, r9612;
}
{
sub.f16x2 r9789, r9783, r9786;
}
{
mul.f16x2 r9792, r8938, r9612;
}
{
fma.rn.f16x2 r9795, r8974, r9611, r9792;
}
{
mul.f16x2 r9799, r9546, r9621;
}
{
mul.f16x2 r9802, r9582, r9622;
}
{
sub.f16x2 r9805, r9799, r9802;
}
{
mul.f16x2 r9808, r9546, r9622;
}
{
fma.rn.f16x2 r9811, r9582, r9621, r9808;
}
{
mul.f16x2 r9815, r8780, r9613;
}
{
mul.f16x2 r9818, r8816, r9614;
}
{
sub.f16x2 r9821, r9815, r9818;
}
{
mul.f16x2 r9824, r8780, r9614;
}
{
fma.rn.f16x2 r9827, r8816, r9613, r9824;
}
{
mul.f16x2 r9831, r9388, r9625;
}
{
mul.f16x2 r9834, r9424, r9626;
}
{
sub.f16x2 r9837, r9831, r9834;
}
{
mul.f16x2 r9840, r9388, r9626;
}
{
fma.rn.f16x2 r9843, r9424, r9625, r9840;
}
{
mul.f16x2 r9847, r8868, r9615;
}
{
mul.f16x2 r9850, r8904, r9616;
}
{
sub.f16x2 r9853, r9847, r9850;
}
{
mul.f16x2 r9856, r8868, r9616;
}
{
fma.rn.f16x2 r9859, r8904, r9615, r9856;
}
{
mul.f16x2 r9863, r9476, r9629;
}
{
mul.f16x2 r9866, r9512, r9630;
}
{
sub.f16x2 r9869, r9863, r9866;
}
{
mul.f16x2 r9872, r9476, r9630;
}
{
fma.rn.f16x2 r9875, r9512, r9629, r9872;
}
{
mul.f16x2 r9879, r8956, r9617;
}
{
mul.f16x2 r9882, r8992, r9618;
}
{
sub.f16x2 r9885, r9879, r9882;
}
{
mul.f16x2 r9888, r8956, r9618;
}
{
fma.rn.f16x2 r9891, r8992, r9617, r9888;
}
{
mul.f16x2 r9895, r9564, r9633;
}
{
mul.f16x2 r9898, r9600, r9634;
}
{
sub.f16x2 r9901, r9895, r9898;
}
{
mul.f16x2 r9904, r9564, r9634;
}
{
fma.rn.f16x2 r9907, r9600, r9633, r9904;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r9911, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r9912, {low, high};
}
{
neg.f16x2 r9913, r9912;
}
{
add.f16x2 r9915, r8738, r9346;
}
{
add.f16x2 %0, r8130, r9915;
}
{
add.f16x2 r9921, r8744, r9352;
}
{
add.f16x2 %1, r8136, r9921;
}
{
add.f16x2 r9927, r8738, r9346;
}
{
mul.f16x2 r9930, r9927, r9911;
}
{
add.f16x2 r9933, r8130, r9930;
}
{
sub.f16x2 r9936, r8744, r9352;
}
{
mul.f16x2 r9939, r9936, r9913;
}
{
add.f16x2 %18, r9933, r9939;
}
{
add.f16x2 r9945, r8738, r9346;
}
{
mul.f16x2 r9948, r9945, r9911;
}
{
add.f16x2 r9951, r8130, r9948;
}
{
sub.f16x2 r9954, r8744, r9352;
}
{
mul.f16x2 r9957, r9954, r9913;
}
{
sub.f16x2 %36, r9951, r9957;
}
{
add.f16x2 r9963, r8744, r9352;
}
{
mul.f16x2 r9966, r9963, r9911;
}
{
add.f16x2 r9969, r8136, r9966;
}
{
sub.f16x2 r9972, r8738, r9346;
}
{
mul.f16x2 r9975, r9972, r9913;
}
{
sub.f16x2 %19, r9969, r9975;
}
{
add.f16x2 r9981, r8744, r9352;
}
{
mul.f16x2 r9984, r9981, r9911;
}
{
add.f16x2 r9987, r8136, r9984;
}
{
sub.f16x2 r9990, r8738, r9346;
}
{
mul.f16x2 r9993, r9990, r9913;
}
{
add.f16x2 %37, r9987, r9993;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r9999, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r10000, {low, high};
}
{
neg.f16x2 r10001, r10000;
}
{
add.f16x2 r10003, r9661, r9677;
}
{
add.f16x2 %2, r8218, r10003;
}
{
add.f16x2 r10009, r9667, r9683;
}
{
add.f16x2 %3, r8224, r10009;
}
{
add.f16x2 r10015, r9661, r9677;
}
{
mul.f16x2 r10018, r10015, r9999;
}
{
add.f16x2 r10021, r8218, r10018;
}
{
sub.f16x2 r10024, r9667, r9683;
}
{
mul.f16x2 r10027, r10024, r10001;
}
{
add.f16x2 %20, r10021, r10027;
}
{
add.f16x2 r10033, r9661, r9677;
}
{
mul.f16x2 r10036, r10033, r9999;
}
{
add.f16x2 r10039, r8218, r10036;
}
{
sub.f16x2 r10042, r9667, r9683;
}
{
mul.f16x2 r10045, r10042, r10001;
}
{
sub.f16x2 %38, r10039, r10045;
}
{
add.f16x2 r10051, r9667, r9683;
}
{
mul.f16x2 r10054, r10051, r9999;
}
{
add.f16x2 r10057, r8224, r10054;
}
{
sub.f16x2 r10060, r9661, r9677;
}
{
mul.f16x2 r10063, r10060, r10001;
}
{
sub.f16x2 %21, r10057, r10063;
}
{
add.f16x2 r10069, r9667, r9683;
}
{
mul.f16x2 r10072, r10069, r9999;
}
{
add.f16x2 r10075, r8224, r10072;
}
{
sub.f16x2 r10078, r9661, r9677;
}
{
mul.f16x2 r10081, r10078, r10001;
}
{
add.f16x2 %39, r10075, r10081;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r10087, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r10088, {low, high};
}
{
neg.f16x2 r10089, r10088;
}
{
add.f16x2 r10091, r9693, r9709;
}
{
add.f16x2 %4, r8306, r10091;
}
{
add.f16x2 r10097, r9699, r9715;
}
{
add.f16x2 %5, r8312, r10097;
}
{
add.f16x2 r10103, r9693, r9709;
}
{
mul.f16x2 r10106, r10103, r10087;
}
{
add.f16x2 r10109, r8306, r10106;
}
{
sub.f16x2 r10112, r9699, r9715;
}
{
mul.f16x2 r10115, r10112, r10089;
}
{
add.f16x2 %22, r10109, r10115;
}
{
add.f16x2 r10121, r9693, r9709;
}
{
mul.f16x2 r10124, r10121, r10087;
}
{
add.f16x2 r10127, r8306, r10124;
}
{
sub.f16x2 r10130, r9699, r9715;
}
{
mul.f16x2 r10133, r10130, r10089;
}
{
sub.f16x2 %40, r10127, r10133;
}
{
add.f16x2 r10139, r9699, r9715;
}
{
mul.f16x2 r10142, r10139, r10087;
}
{
add.f16x2 r10145, r8312, r10142;
}
{
sub.f16x2 r10148, r9693, r9709;
}
{
mul.f16x2 r10151, r10148, r10089;
}
{
sub.f16x2 %23, r10145, r10151;
}
{
add.f16x2 r10157, r9699, r9715;
}
{
mul.f16x2 r10160, r10157, r10087;
}
{
add.f16x2 r10163, r8312, r10160;
}
{
sub.f16x2 r10166, r9693, r9709;
}
{
mul.f16x2 r10169, r10166, r10089;
}
{
add.f16x2 %41, r10163, r10169;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r10175, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r10176, {low, high};
}
{
neg.f16x2 r10177, r10176;
}
{
add.f16x2 r10179, r9725, r9741;
}
{
add.f16x2 %6, r8154, r10179;
}
{
add.f16x2 r10185, r9731, r9747;
}
{
add.f16x2 %7, r8190, r10185;
}
{
add.f16x2 r10191, r9725, r9741;
}
{
mul.f16x2 r10194, r10191, r10175;
}
{
add.f16x2 r10197, r8154, r10194;
}
{
sub.f16x2 r10200, r9731, r9747;
}
{
mul.f16x2 r10203, r10200, r10177;
}
{
add.f16x2 %24, r10197, r10203;
}
{
add.f16x2 r10209, r9725, r9741;
}
{
mul.f16x2 r10212, r10209, r10175;
}
{
add.f16x2 r10215, r8154, r10212;
}
{
sub.f16x2 r10218, r9731, r9747;
}
{
mul.f16x2 r10221, r10218, r10177;
}
{
sub.f16x2 %42, r10215, r10221;
}
{
add.f16x2 r10227, r9731, r9747;
}
{
mul.f16x2 r10230, r10227, r10175;
}
{
add.f16x2 r10233, r8190, r10230;
}
{
sub.f16x2 r10236, r9725, r9741;
}
{
mul.f16x2 r10239, r10236, r10177;
}
{
sub.f16x2 %25, r10233, r10239;
}
{
add.f16x2 r10245, r9731, r9747;
}
{
mul.f16x2 r10248, r10245, r10175;
}
{
add.f16x2 r10251, r8190, r10248;
}
{
sub.f16x2 r10254, r9725, r9741;
}
{
mul.f16x2 r10257, r10254, r10177;
}
{
add.f16x2 %43, r10251, r10257;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r10263, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r10264, {low, high};
}
{
neg.f16x2 r10265, r10264;
}
{
add.f16x2 r10267, r9757, r9773;
}
{
add.f16x2 %8, r8242, r10267;
}
{
add.f16x2 r10273, r9763, r9779;
}
{
add.f16x2 %9, r8278, r10273;
}
{
add.f16x2 r10279, r9757, r9773;
}
{
mul.f16x2 r10282, r10279, r10263;
}
{
add.f16x2 r10285, r8242, r10282;
}
{
sub.f16x2 r10288, r9763, r9779;
}
{
mul.f16x2 r10291, r10288, r10265;
}
{
add.f16x2 %26, r10285, r10291;
}
{
add.f16x2 r10297, r9757, r9773;
}
{
mul.f16x2 r10300, r10297, r10263;
}
{
add.f16x2 r10303, r8242, r10300;
}
{
sub.f16x2 r10306, r9763, r9779;
}
{
mul.f16x2 r10309, r10306, r10265;
}
{
sub.f16x2 %44, r10303, r10309;
}
{
add.f16x2 r10315, r9763, r9779;
}
{
mul.f16x2 r10318, r10315, r10263;
}
{
add.f16x2 r10321, r8278, r10318;
}
{
sub.f16x2 r10324, r9757, r9773;
}
{
mul.f16x2 r10327, r10324, r10265;
}
{
sub.f16x2 %27, r10321, r10327;
}
{
add.f16x2 r10333, r9763, r9779;
}
{
mul.f16x2 r10336, r10333, r10263;
}
{
add.f16x2 r10339, r8278, r10336;
}
{
sub.f16x2 r10342, r9757, r9773;
}
{
mul.f16x2 r10345, r10342, r10265;
}
{
add.f16x2 %45, r10339, r10345;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r10351, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r10352, {low, high};
}
{
neg.f16x2 r10353, r10352;
}
{
add.f16x2 r10355, r9789, r9805;
}
{
add.f16x2 %10, r8330, r10355;
}
{
add.f16x2 r10361, r9795, r9811;
}
{
add.f16x2 %11, r8366, r10361;
}
{
add.f16x2 r10367, r9789, r9805;
}
{
mul.f16x2 r10370, r10367, r10351;
}
{
add.f16x2 r10373, r8330, r10370;
}
{
sub.f16x2 r10376, r9795, r9811;
}
{
mul.f16x2 r10379, r10376, r10353;
}
{
add.f16x2 %28, r10373, r10379;
}
{
add.f16x2 r10385, r9789, r9805;
}
{
mul.f16x2 r10388, r10385, r10351;
}
{
add.f16x2 r10391, r8330, r10388;
}
{
sub.f16x2 r10394, r9795, r9811;
}
{
mul.f16x2 r10397, r10394, r10353;
}
{
sub.f16x2 %46, r10391, r10397;
}
{
add.f16x2 r10403, r9795, r9811;
}
{
mul.f16x2 r10406, r10403, r10351;
}
{
add.f16x2 r10409, r8366, r10406;
}
{
sub.f16x2 r10412, r9789, r9805;
}
{
mul.f16x2 r10415, r10412, r10353;
}
{
sub.f16x2 %29, r10409, r10415;
}
{
add.f16x2 r10421, r9795, r9811;
}
{
mul.f16x2 r10424, r10421, r10351;
}
{
add.f16x2 r10427, r8366, r10424;
}
{
sub.f16x2 r10430, r9789, r9805;
}
{
mul.f16x2 r10433, r10430, r10353;
}
{
add.f16x2 %47, r10427, r10433;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r10439, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r10440, {low, high};
}
{
neg.f16x2 r10441, r10440;
}
{
add.f16x2 r10443, r9821, r9837;
}
{
add.f16x2 %12, r8172, r10443;
}
{
add.f16x2 r10449, r9827, r9843;
}
{
add.f16x2 %13, r8208, r10449;
}
{
add.f16x2 r10455, r9821, r9837;
}
{
mul.f16x2 r10458, r10455, r10439;
}
{
add.f16x2 r10461, r8172, r10458;
}
{
sub.f16x2 r10464, r9827, r9843;
}
{
mul.f16x2 r10467, r10464, r10441;
}
{
add.f16x2 %30, r10461, r10467;
}
{
add.f16x2 r10473, r9821, r9837;
}
{
mul.f16x2 r10476, r10473, r10439;
}
{
add.f16x2 r10479, r8172, r10476;
}
{
sub.f16x2 r10482, r9827, r9843;
}
{
mul.f16x2 r10485, r10482, r10441;
}
{
sub.f16x2 %48, r10479, r10485;
}
{
add.f16x2 r10491, r9827, r9843;
}
{
mul.f16x2 r10494, r10491, r10439;
}
{
add.f16x2 r10497, r8208, r10494;
}
{
sub.f16x2 r10500, r9821, r9837;
}
{
mul.f16x2 r10503, r10500, r10441;
}
{
sub.f16x2 %31, r10497, r10503;
}
{
add.f16x2 r10509, r9827, r9843;
}
{
mul.f16x2 r10512, r10509, r10439;
}
{
add.f16x2 r10515, r8208, r10512;
}
{
sub.f16x2 r10518, r9821, r9837;
}
{
mul.f16x2 r10521, r10518, r10441;
}
{
add.f16x2 %49, r10515, r10521;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r10527, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r10528, {low, high};
}
{
neg.f16x2 r10529, r10528;
}
{
add.f16x2 r10531, r9853, r9869;
}
{
add.f16x2 %14, r8260, r10531;
}
{
add.f16x2 r10537, r9859, r9875;
}
{
add.f16x2 %15, r8296, r10537;
}
{
add.f16x2 r10543, r9853, r9869;
}
{
mul.f16x2 r10546, r10543, r10527;
}
{
add.f16x2 r10549, r8260, r10546;
}
{
sub.f16x2 r10552, r9859, r9875;
}
{
mul.f16x2 r10555, r10552, r10529;
}
{
add.f16x2 %32, r10549, r10555;
}
{
add.f16x2 r10561, r9853, r9869;
}
{
mul.f16x2 r10564, r10561, r10527;
}
{
add.f16x2 r10567, r8260, r10564;
}
{
sub.f16x2 r10570, r9859, r9875;
}
{
mul.f16x2 r10573, r10570, r10529;
}
{
sub.f16x2 %50, r10567, r10573;
}
{
add.f16x2 r10579, r9859, r9875;
}
{
mul.f16x2 r10582, r10579, r10527;
}
{
add.f16x2 r10585, r8296, r10582;
}
{
sub.f16x2 r10588, r9853, r9869;
}
{
mul.f16x2 r10591, r10588, r10529;
}
{
sub.f16x2 %33, r10585, r10591;
}
{
add.f16x2 r10597, r9859, r9875;
}
{
mul.f16x2 r10600, r10597, r10527;
}
{
add.f16x2 r10603, r8296, r10600;
}
{
sub.f16x2 r10606, r9853, r9869;
}
{
mul.f16x2 r10609, r10606, r10529;
}
{
add.f16x2 %51, r10603, r10609;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r10615, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r10616, {low, high};
}
{
neg.f16x2 r10617, r10616;
}
{
add.f16x2 r10619, r9885, r9901;
}
{
add.f16x2 %16, r8348, r10619;
}
{
add.f16x2 r10625, r9891, r9907;
}
{
add.f16x2 %17, r8384, r10625;
}
{
add.f16x2 r10631, r9885, r9901;
}
{
mul.f16x2 r10634, r10631, r10615;
}
{
add.f16x2 r10637, r8348, r10634;
}
{
sub.f16x2 r10640, r9891, r9907;
}
{
mul.f16x2 r10643, r10640, r10617;
}
{
add.f16x2 %34, r10637, r10643;
}
{
add.f16x2 r10649, r9885, r9901;
}
{
mul.f16x2 r10652, r10649, r10615;
}
{
add.f16x2 r10655, r8348, r10652;
}
{
sub.f16x2 r10658, r9891, r9907;
}
{
mul.f16x2 r10661, r10658, r10617;
}
{
sub.f16x2 %52, r10655, r10661;
}
{
add.f16x2 r10667, r9891, r9907;
}
{
mul.f16x2 r10670, r10667, r10615;
}
{
add.f16x2 r10673, r8384, r10670;
}
{
sub.f16x2 r10676, r9885, r9901;
}
{
mul.f16x2 r10679, r10676, r10617;
}
{
sub.f16x2 %35, r10673, r10679;
}
{
add.f16x2 r10685, r9891, r9907;
}
{
mul.f16x2 r10688, r10685, r10615;
}
{
add.f16x2 r10691, r8384, r10688;
}
{
sub.f16x2 r10694, r9885, r9901;
}
{
mul.f16x2 r10697, r10694, r10617;
}
{
add.f16x2 %53, r10691, r10697;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[4].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1178, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<1043>;
.reg .b32 r<10778>;
.reg .b64 rd<6>;
mov.u32 r10704, %54;
mov.u32 r10777, %tid.y;
mad.lo.s32 r10705, r10777, 78732, r10704;
mov.u32 r10706, %tid.x;
mov.f32 f1034, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1, {low, high};
}
mov.f32 f1036, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r177, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
mov.f32 f906, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r265, {low, high};
}
mov.f32 f908, 0fBF248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r266, {low, high};
}
mov.f32 f918, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r267, {low, high};
}
mov.f32 f920, 0fBF7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r268, {low, high};
}
mov.f32 f942, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r271, {low, high};
}
mov.f32 f944, 0fBEAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r345, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r433, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r521, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r609, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r697, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r785, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r873, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r874, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r875, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r876, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r879, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r953, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1041, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1129, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1217, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
add.f16x2 r1230, %83, r1227;
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
add.f16x2 r1275, %83, r1272;
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
add.f16x2 r1293, %83, r1290;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1305, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r1306, {low, high};
}
{
neg.f16x2 r1307, r1306;
}
{
add.f16x2 r1309, %82, %73;
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
add.f16x2 r1321, %82, %73;
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
add.f16x2 r1339, %82, %73;
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
sub.f16x2 r1366, %82, %73;
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
sub.f16x2 r1384, %82, %73;
}
{
mul.f16x2 r1387, r1384, r1307;
}
{
add.f16x2 r1390, r1381, r1387;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1393, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r1481, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r1482, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r1483, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r1484, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r1487, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1561, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1649, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r1737, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
mov.f32 f898, 0f3F791978;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f898;
cvt.rn.f16.f32 high, f898;
mov.b32 r1825, {low, high};
}
mov.f32 f900, 0fBE6C2691;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f900;
cvt.rn.f16.f32 high, f900;
mov.b32 r1826, {low, high};
}
mov.f32 f902, 0f3F64C51C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f902;
cvt.rn.f16.f32 high, f902;
mov.b32 r1827, {low, high};
}
mov.f32 f904, 0fBEE5C902;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f904;
cvt.rn.f16.f32 high, f904;
mov.b32 r1828, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r1829, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r1830, {low, high};
}
mov.f32 f910, 0f3F18DF63;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f910;
cvt.rn.f16.f32 high, f910;
mov.b32 r1831, {low, high};
}
mov.f32 f912, 0fBF4D57F2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f912;
cvt.rn.f16.f32 high, f912;
mov.b32 r1832, {low, high};
}
mov.f32 f914, 0f3ECACAF8;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f914;
cvt.rn.f16.f32 high, f914;
mov.b32 r1833, {low, high};
}
mov.f32 f916, 0fBF6B1036;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f916;
cvt.rn.f16.f32 high, f916;
mov.b32 r1834, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r1835, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r1836, {low, high};
}
mov.f32 f922, 0fBD6E2946;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f922;
cvt.rn.f16.f32 high, f922;
mov.b32 r1837, {low, high};
}
mov.f32 f924, 0fBF7F9120;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f924;
cvt.rn.f16.f32 high, f924;
mov.b32 r1838, {low, high};
}
mov.f32 f926, 0fBE92D7E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f926;
cvt.rn.f16.f32 high, f926;
mov.b32 r1839, {low, high};
}
mov.f32 f928, 0fBF753ECD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f928;
cvt.rn.f16.f32 high, f928;
mov.b32 r1840, {low, high};
}
mov.f32 f934, 0fBF2FAD88;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f934;
cvt.rn.f16.f32 high, f934;
mov.b32 r1843, {low, high};
}
mov.f32 f936, 0fBF3A3529;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f936;
cvt.rn.f16.f32 high, f936;
mov.b32 r1844, {low, high};
}
mov.f32 f958, 0fBF55E287;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r1847, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
mov.b32 r1848, {low, high};
}
mov.f32 f950, 0fBF7E44DE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f950;
cvt.rn.f16.f32 high, f950;
mov.b32 r1851, {low, high};
}
mov.f32 f952, 0f3DEDC21F;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f952;
cvt.rn.f16.f32 high, f952;
mov.b32 r1852, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f958;
cvt.rn.f16.f32 high, f958;
mov.b32 r1855, {low, high};
}
mov.f32 f960, 0f3F0CAC9F;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f960;
cvt.rn.f16.f32 high, f960;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r2133, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r2221, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r2309, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r2397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r2485, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r2573, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r2661, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r2749, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r2837, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
mul.wide.u32 rd2, r10706, 1508246403;
shr.u64 rd3, rd2, 40;
cvt.u32.u64 r10707, rd3;
mul.lo.s32 r10708, r10707, 729;
sub.s32 r10709, r10706, r10708;
mad.lo.s32 r10710, r10707, 78732, r10705;
cvt.rn.f32.u32 f1037, r10709;
mul.f32 f1038, f1037, 0f39A75CD5;
cos.approx.f32 f309, f1038;
sin.approx.f32 f1039, f1038;
neg.f32 f310, f1039;
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
mad.lo.s32 r10711, r10709, 108, r10710;
st.shared.u32 [r10711], r2140;
st.shared.u32 [r10711+4], r2937;
st.shared.u32 [r10711+8], r2974;
st.shared.u32 [r10711+12], r3011;
st.shared.u32 [r10711+16], r3048;
st.shared.u32 [r10711+20], r3085;
st.shared.u32 [r10711+24], r3122;
st.shared.u32 [r10711+28], r3159;
st.shared.u32 [r10711+32], r3196;
st.shared.u32 [r10711+36], r3233;
st.shared.u32 [r10711+40], r3270;
st.shared.u32 [r10711+44], r3307;
st.shared.u32 [r10711+48], r3344;
st.shared.u32 [r10711+52], r3381;
st.shared.u32 [r10711+56], r3418;
st.shared.u32 [r10711+60], r3455;
st.shared.u32 [r10711+64], r3492;
st.shared.u32 [r10711+68], r3529;
st.shared.u32 [r10711+72], r3566;
st.shared.u32 [r10711+76], r3603;
st.shared.u32 [r10711+80], r3640;
st.shared.u32 [r10711+84], r3677;
st.shared.u32 [r10711+88], r3714;
st.shared.u32 [r10711+92], r3751;
st.shared.u32 [r10711+96], r3788;
st.shared.u32 [r10711+100], r3825;
st.shared.u32 [r10711+104], r3862;
barrier.sync 0;
mad.lo.s32 r10712, r10709, -104, r10711;
ld.shared.u32 r3898, [r10712];
ld.shared.u32 r4506, [r10712+2916];
ld.shared.u32 r5114, [r10712+5832];
ld.shared.u32 r3986, [r10712+8748];
ld.shared.u32 r4594, [r10712+11664];
ld.shared.u32 r5202, [r10712+14580];
ld.shared.u32 r4074, [r10712+17496];
ld.shared.u32 r4682, [r10712+20412];
ld.shared.u32 r5290, [r10712+23328];
ld.shared.u32 r3895, [r10712+26244];
ld.shared.u32 r4503, [r10712+29160];
ld.shared.u32 r5111, [r10712+32076];
ld.shared.u32 r3983, [r10712+34992];
ld.shared.u32 r4591, [r10712+37908];
ld.shared.u32 r5199, [r10712+40824];
ld.shared.u32 r4071, [r10712+43740];
ld.shared.u32 r4679, [r10712+46656];
ld.shared.u32 r5287, [r10712+49572];
ld.shared.u32 r3896, [r10712+52488];
ld.shared.u32 r4504, [r10712+55404];
ld.shared.u32 r5112, [r10712+58320];
ld.shared.u32 r3984, [r10712+61236];
ld.shared.u32 r4592, [r10712+64152];
ld.shared.u32 r5200, [r10712+67068];
ld.shared.u32 r4072, [r10712+69984];
ld.shared.u32 r4680, [r10712+72900];
ld.shared.u32 r5288, [r10712+75816];
barrier.sync 0;
st.shared.u32 [r10711], r2146;
st.shared.u32 [r10711+4], r2944;
st.shared.u32 [r10711+8], r2981;
st.shared.u32 [r10711+12], r3018;
st.shared.u32 [r10711+16], r3055;
st.shared.u32 [r10711+20], r3092;
st.shared.u32 [r10711+24], r3129;
st.shared.u32 [r10711+28], r3166;
st.shared.u32 [r10711+32], r3203;
st.shared.u32 [r10711+36], r3240;
st.shared.u32 [r10711+40], r3277;
st.shared.u32 [r10711+44], r3314;
st.shared.u32 [r10711+48], r3351;
st.shared.u32 [r10711+52], r3388;
st.shared.u32 [r10711+56], r3425;
st.shared.u32 [r10711+60], r3462;
st.shared.u32 [r10711+64], r3499;
st.shared.u32 [r10711+68], r3536;
st.shared.u32 [r10711+72], r3573;
st.shared.u32 [r10711+76], r3610;
st.shared.u32 [r10711+80], r3647;
st.shared.u32 [r10711+84], r3684;
st.shared.u32 [r10711+88], r3721;
st.shared.u32 [r10711+92], r3758;
st.shared.u32 [r10711+96], r3795;
st.shared.u32 [r10711+100], r3832;
st.shared.u32 [r10711+104], r3869;
barrier.sync 0;
ld.shared.u32 r3904, [r10712];
ld.shared.u32 r4512, [r10712+2916];
ld.shared.u32 r5120, [r10712+5832];
ld.shared.u32 r3992, [r10712+8748];
ld.shared.u32 r4600, [r10712+11664];
ld.shared.u32 r5208, [r10712+14580];
ld.shared.u32 r4080, [r10712+17496];
ld.shared.u32 r4688, [r10712+20412];
ld.shared.u32 r5296, [r10712+23328];
ld.shared.u32 r3901, [r10712+26244];
ld.shared.u32 r4509, [r10712+29160];
ld.shared.u32 r5117, [r10712+32076];
ld.shared.u32 r3989, [r10712+34992];
ld.shared.u32 r4597, [r10712+37908];
ld.shared.u32 r5205, [r10712+40824];
ld.shared.u32 r4077, [r10712+43740];
ld.shared.u32 r4685, [r10712+46656];
ld.shared.u32 r5293, [r10712+49572];
ld.shared.u32 r3902, [r10712+52488];
ld.shared.u32 r4510, [r10712+55404];
ld.shared.u32 r5118, [r10712+58320];
ld.shared.u32 r3990, [r10712+61236];
ld.shared.u32 r4598, [r10712+64152];
ld.shared.u32 r5206, [r10712+67068];
ld.shared.u32 r4078, [r10712+69984];
ld.shared.u32 r4686, [r10712+72900];
ld.shared.u32 r5294, [r10712+75816];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r3890, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r3978, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r4066, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r4154, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r4155, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r4156, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r4157, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r4160, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r4234, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r4322, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r4410, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r4498, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r4586, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r4674, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r4762, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r4763, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r4764, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r4765, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r4768, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r4842, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r4930, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r5018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r5106, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r5194, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r5282, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r5370, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r5371, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r5372, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r5373, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r5376, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r5450, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r5538, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r5626, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f898;
cvt.rn.f16.f32 high, f898;
mov.b32 r5714, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f900;
cvt.rn.f16.f32 high, f900;
mov.b32 r5715, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f902;
cvt.rn.f16.f32 high, f902;
mov.b32 r5716, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f904;
cvt.rn.f16.f32 high, f904;
mov.b32 r5717, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r5718, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r5719, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f910;
cvt.rn.f16.f32 high, f910;
mov.b32 r5720, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f912;
cvt.rn.f16.f32 high, f912;
mov.b32 r5721, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f914;
cvt.rn.f16.f32 high, f914;
mov.b32 r5722, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f916;
cvt.rn.f16.f32 high, f916;
mov.b32 r5723, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r5724, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r5725, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f922;
cvt.rn.f16.f32 high, f922;
mov.b32 r5726, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f924;
cvt.rn.f16.f32 high, f924;
mov.b32 r5727, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f926;
cvt.rn.f16.f32 high, f926;
mov.b32 r5728, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f928;
cvt.rn.f16.f32 high, f928;
mov.b32 r5729, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f934;
cvt.rn.f16.f32 high, f934;
mov.b32 r5732, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f936;
cvt.rn.f16.f32 high, f936;
mov.b32 r5733, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r5736, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
mov.b32 r5737, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f950;
cvt.rn.f16.f32 high, f950;
mov.b32 r5740, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f952;
cvt.rn.f16.f32 high, f952;
mov.b32 r5741, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f958;
cvt.rn.f16.f32 high, f958;
mov.b32 r5744, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f960;
cvt.rn.f16.f32 high, f960;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6022, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6110, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6198, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6286, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6374, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6462, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6550, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6638, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r6726, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
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
mul.wide.u32 rd4, r10709, 795364315;
shr.u64 rd5, rd4, 32;
cvt.u32.u64 r10713, rd5;
sub.s32 r10714, r10709, r10713;
shr.u32 r10715, r10714, 1;
add.s32 r10716, r10715, r10713;
shr.u32 r10717, r10716, 4;
mul.lo.s32 r10718, r10717, 27;
sub.s32 r10719, r10709, r10718;
shl.b32 r10720, r10719, 2;
add.s32 r10721, r10710, r10720;
cvt.rn.f32.u32 f1040, r10717;
mul.f32 f1041, f1040, 0f3C0D3654;
cos.approx.f32 f673, f1041;
sin.approx.f32 f1042, f1041;
neg.f32 f674, f1042;
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
mad.lo.s32 r10722, r10717, 2916, r10721;
st.shared.u32 [r10722], r6029;
st.shared.u32 [r10722+108], r6826;
st.shared.u32 [r10722+216], r6863;
st.shared.u32 [r10722+324], r6900;
st.shared.u32 [r10722+432], r6937;
st.shared.u32 [r10722+540], r6974;
st.shared.u32 [r10722+648], r7011;
st.shared.u32 [r10722+756], r7048;
st.shared.u32 [r10722+864], r7085;
st.shared.u32 [r10722+972], r7122;
st.shared.u32 [r10722+1080], r7159;
st.shared.u32 [r10722+1188], r7196;
st.shared.u32 [r10722+1296], r7233;
st.shared.u32 [r10722+1404], r7270;
st.shared.u32 [r10722+1512], r7307;
st.shared.u32 [r10722+1620], r7344;
st.shared.u32 [r10722+1728], r7381;
st.shared.u32 [r10722+1836], r7418;
st.shared.u32 [r10722+1944], r7455;
st.shared.u32 [r10722+2052], r7492;
st.shared.u32 [r10722+2160], r7529;
st.shared.u32 [r10722+2268], r7566;
st.shared.u32 [r10722+2376], r7603;
st.shared.u32 [r10722+2484], r7640;
st.shared.u32 [r10722+2592], r7677;
st.shared.u32 [r10722+2700], r7714;
st.shared.u32 [r10722+2808], r7751;
barrier.sync 0;
ld.shared.u32 r7787, [r10712];
ld.shared.u32 r8395, [r10712+2916];
ld.shared.u32 r9003, [r10712+5832];
ld.shared.u32 r7875, [r10712+8748];
ld.shared.u32 r8483, [r10712+11664];
ld.shared.u32 r9091, [r10712+14580];
ld.shared.u32 r7963, [r10712+17496];
ld.shared.u32 r8571, [r10712+20412];
ld.shared.u32 r9179, [r10712+23328];
ld.shared.u32 r7784, [r10712+26244];
ld.shared.u32 r8392, [r10712+29160];
ld.shared.u32 r9000, [r10712+32076];
ld.shared.u32 r7872, [r10712+34992];
ld.shared.u32 r8480, [r10712+37908];
ld.shared.u32 r9088, [r10712+40824];
ld.shared.u32 r7960, [r10712+43740];
ld.shared.u32 r8568, [r10712+46656];
ld.shared.u32 r9176, [r10712+49572];
ld.shared.u32 r7785, [r10712+52488];
ld.shared.u32 r8393, [r10712+55404];
ld.shared.u32 r9001, [r10712+58320];
ld.shared.u32 r7873, [r10712+61236];
ld.shared.u32 r8481, [r10712+64152];
ld.shared.u32 r9089, [r10712+67068];
ld.shared.u32 r7961, [r10712+69984];
ld.shared.u32 r8569, [r10712+72900];
ld.shared.u32 r9177, [r10712+75816];
barrier.sync 0;
st.shared.u32 [r10722], r6035;
st.shared.u32 [r10722+108], r6833;
st.shared.u32 [r10722+216], r6870;
st.shared.u32 [r10722+324], r6907;
st.shared.u32 [r10722+432], r6944;
st.shared.u32 [r10722+540], r6981;
st.shared.u32 [r10722+648], r7018;
st.shared.u32 [r10722+756], r7055;
st.shared.u32 [r10722+864], r7092;
st.shared.u32 [r10722+972], r7129;
st.shared.u32 [r10722+1080], r7166;
st.shared.u32 [r10722+1188], r7203;
st.shared.u32 [r10722+1296], r7240;
st.shared.u32 [r10722+1404], r7277;
st.shared.u32 [r10722+1512], r7314;
st.shared.u32 [r10722+1620], r7351;
st.shared.u32 [r10722+1728], r7388;
st.shared.u32 [r10722+1836], r7425;
st.shared.u32 [r10722+1944], r7462;
st.shared.u32 [r10722+2052], r7499;
st.shared.u32 [r10722+2160], r7536;
st.shared.u32 [r10722+2268], r7573;
st.shared.u32 [r10722+2376], r7610;
st.shared.u32 [r10722+2484], r7647;
st.shared.u32 [r10722+2592], r7684;
st.shared.u32 [r10722+2700], r7721;
st.shared.u32 [r10722+2808], r7758;
barrier.sync 0;
ld.shared.u32 r7793, [r10712];
ld.shared.u32 r8401, [r10712+2916];
ld.shared.u32 r9009, [r10712+5832];
ld.shared.u32 r7881, [r10712+8748];
ld.shared.u32 r8489, [r10712+11664];
ld.shared.u32 r9097, [r10712+14580];
ld.shared.u32 r7969, [r10712+17496];
ld.shared.u32 r8577, [r10712+20412];
ld.shared.u32 r9185, [r10712+23328];
ld.shared.u32 r7790, [r10712+26244];
ld.shared.u32 r8398, [r10712+29160];
ld.shared.u32 r9006, [r10712+32076];
ld.shared.u32 r7878, [r10712+34992];
ld.shared.u32 r8486, [r10712+37908];
ld.shared.u32 r9094, [r10712+40824];
ld.shared.u32 r7966, [r10712+43740];
ld.shared.u32 r8574, [r10712+46656];
ld.shared.u32 r9182, [r10712+49572];
ld.shared.u32 r7791, [r10712+52488];
ld.shared.u32 r8399, [r10712+55404];
ld.shared.u32 r9007, [r10712+58320];
ld.shared.u32 r7879, [r10712+61236];
ld.shared.u32 r8487, [r10712+64152];
ld.shared.u32 r9095, [r10712+67068];
ld.shared.u32 r7967, [r10712+69984];
ld.shared.u32 r8575, [r10712+72900];
ld.shared.u32 r9183, [r10712+75816];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r7779, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r7780, {low, high};
}
{
neg.f16x2 r7781, r7780;
}
{
add.f16x2 r7783, r7784, r7785;
}
{
add.f16x2 r7786, r7787, r7783;
}
{
add.f16x2 r7789, r7790, r7791;
}
{
add.f16x2 r7792, r7793, r7789;
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
add.f16x2 r7810, r7801, r7807;
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
sub.f16x2 r7828, r7819, r7825;
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
sub.f16x2 r7846, r7837, r7843;
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
add.f16x2 r7864, r7855, r7861;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r7867, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r7868, {low, high};
}
{
neg.f16x2 r7869, r7868;
}
{
add.f16x2 r7871, r7872, r7873;
}
{
add.f16x2 r7874, r7875, r7871;
}
{
add.f16x2 r7877, r7878, r7879;
}
{
add.f16x2 r7880, r7881, r7877;
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
add.f16x2 r7898, r7889, r7895;
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
sub.f16x2 r7916, r7907, r7913;
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
sub.f16x2 r7934, r7925, r7931;
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
add.f16x2 r7952, r7943, r7949;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r7955, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r7956, {low, high};
}
{
neg.f16x2 r7957, r7956;
}
{
add.f16x2 r7959, r7960, r7961;
}
{
add.f16x2 r7962, r7963, r7959;
}
{
add.f16x2 r7965, r7966, r7967;
}
{
add.f16x2 r7968, r7969, r7965;
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
add.f16x2 r7986, r7977, r7983;
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
sub.f16x2 r8004, r7995, r8001;
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
sub.f16x2 r8022, r8013, r8019;
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
add.f16x2 r8040, r8031, r8037;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r8043, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r8044, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r8045, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r8046, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r8049, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
mov.b32 r8050, {low, high};
}
{
mul.f16x2 r8059, r7898, r8043;
}
{
mul.f16x2 r8062, r7934, r8044;
}
{
sub.f16x2 r8065, r8059, r8062;
}
{
mul.f16x2 r8068, r7898, r8044;
}
{
fma.rn.f16x2 r8071, r7934, r8043, r8068;
}
{
mul.f16x2 r8075, r7986, r8045;
}
{
mul.f16x2 r8078, r8022, r8046;
}
{
sub.f16x2 r8081, r8075, r8078;
}
{
mul.f16x2 r8084, r7986, r8046;
}
{
fma.rn.f16x2 r8087, r8022, r8045, r8084;
}
{
mul.f16x2 r8091, r7916, r8045;
}
{
mul.f16x2 r8094, r7952, r8046;
}
{
sub.f16x2 r8097, r8091, r8094;
}
{
mul.f16x2 r8100, r7916, r8046;
}
{
fma.rn.f16x2 r8103, r7952, r8045, r8100;
}
{
mul.f16x2 r8107, r8004, r8049;
}
{
mul.f16x2 r8110, r8040, r8050;
}
{
sub.f16x2 r8113, r8107, r8110;
}
{
mul.f16x2 r8116, r8004, r8050;
}
{
fma.rn.f16x2 r8119, r8040, r8049, r8116;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r8123, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r8124, {low, high};
}
{
neg.f16x2 r8125, r8124;
}
{
add.f16x2 r8127, r7874, r7962;
}
{
add.f16x2 r8130, r7786, r8127;
}
{
add.f16x2 r8133, r7880, r7968;
}
{
add.f16x2 r8136, r7792, r8133;
}
{
add.f16x2 r8139, r7874, r7962;
}
{
mul.f16x2 r8142, r8139, r8123;
}
{
add.f16x2 r8145, r7786, r8142;
}
{
sub.f16x2 r8148, r7880, r7968;
}
{
mul.f16x2 r8151, r8148, r8125;
}
{
add.f16x2 r8154, r8145, r8151;
}
{
add.f16x2 r8157, r7874, r7962;
}
{
mul.f16x2 r8160, r8157, r8123;
}
{
add.f16x2 r8163, r7786, r8160;
}
{
sub.f16x2 r8166, r7880, r7968;
}
{
mul.f16x2 r8169, r8166, r8125;
}
{
sub.f16x2 r8172, r8163, r8169;
}
{
add.f16x2 r8175, r7880, r7968;
}
{
mul.f16x2 r8178, r8175, r8123;
}
{
add.f16x2 r8181, r7792, r8178;
}
{
sub.f16x2 r8184, r7874, r7962;
}
{
mul.f16x2 r8187, r8184, r8125;
}
{
sub.f16x2 r8190, r8181, r8187;
}
{
add.f16x2 r8193, r7880, r7968;
}
{
mul.f16x2 r8196, r8193, r8123;
}
{
add.f16x2 r8199, r7792, r8196;
}
{
sub.f16x2 r8202, r7874, r7962;
}
{
mul.f16x2 r8205, r8202, r8125;
}
{
add.f16x2 r8208, r8199, r8205;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r8211, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r8212, {low, high};
}
{
neg.f16x2 r8213, r8212;
}
{
add.f16x2 r8215, r8065, r8081;
}
{
add.f16x2 r8218, r7810, r8215;
}
{
add.f16x2 r8221, r8071, r8087;
}
{
add.f16x2 r8224, r7846, r8221;
}
{
add.f16x2 r8227, r8065, r8081;
}
{
mul.f16x2 r8230, r8227, r8211;
}
{
add.f16x2 r8233, r7810, r8230;
}
{
sub.f16x2 r8236, r8071, r8087;
}
{
mul.f16x2 r8239, r8236, r8213;
}
{
add.f16x2 r8242, r8233, r8239;
}
{
add.f16x2 r8245, r8065, r8081;
}
{
mul.f16x2 r8248, r8245, r8211;
}
{
add.f16x2 r8251, r7810, r8248;
}
{
sub.f16x2 r8254, r8071, r8087;
}
{
mul.f16x2 r8257, r8254, r8213;
}
{
sub.f16x2 r8260, r8251, r8257;
}
{
add.f16x2 r8263, r8071, r8087;
}
{
mul.f16x2 r8266, r8263, r8211;
}
{
add.f16x2 r8269, r7846, r8266;
}
{
sub.f16x2 r8272, r8065, r8081;
}
{
mul.f16x2 r8275, r8272, r8213;
}
{
sub.f16x2 r8278, r8269, r8275;
}
{
add.f16x2 r8281, r8071, r8087;
}
{
mul.f16x2 r8284, r8281, r8211;
}
{
add.f16x2 r8287, r7846, r8284;
}
{
sub.f16x2 r8290, r8065, r8081;
}
{
mul.f16x2 r8293, r8290, r8213;
}
{
add.f16x2 r8296, r8287, r8293;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r8299, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r8300, {low, high};
}
{
neg.f16x2 r8301, r8300;
}
{
add.f16x2 r8303, r8097, r8113;
}
{
add.f16x2 r8306, r7828, r8303;
}
{
add.f16x2 r8309, r8103, r8119;
}
{
add.f16x2 r8312, r7864, r8309;
}
{
add.f16x2 r8315, r8097, r8113;
}
{
mul.f16x2 r8318, r8315, r8299;
}
{
add.f16x2 r8321, r7828, r8318;
}
{
sub.f16x2 r8324, r8103, r8119;
}
{
mul.f16x2 r8327, r8324, r8301;
}
{
add.f16x2 r8330, r8321, r8327;
}
{
add.f16x2 r8333, r8097, r8113;
}
{
mul.f16x2 r8336, r8333, r8299;
}
{
add.f16x2 r8339, r7828, r8336;
}
{
sub.f16x2 r8342, r8103, r8119;
}
{
mul.f16x2 r8345, r8342, r8301;
}
{
sub.f16x2 r8348, r8339, r8345;
}
{
add.f16x2 r8351, r8103, r8119;
}
{
mul.f16x2 r8354, r8351, r8299;
}
{
add.f16x2 r8357, r7864, r8354;
}
{
sub.f16x2 r8360, r8097, r8113;
}
{
mul.f16x2 r8363, r8360, r8301;
}
{
sub.f16x2 r8366, r8357, r8363;
}
{
add.f16x2 r8369, r8103, r8119;
}
{
mul.f16x2 r8372, r8369, r8299;
}
{
add.f16x2 r8375, r7864, r8372;
}
{
sub.f16x2 r8378, r8097, r8113;
}
{
mul.f16x2 r8381, r8378, r8301;
}
{
add.f16x2 r8384, r8375, r8381;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r8387, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r8388, {low, high};
}
{
neg.f16x2 r8389, r8388;
}
{
add.f16x2 r8391, r8392, r8393;
}
{
add.f16x2 r8394, r8395, r8391;
}
{
add.f16x2 r8397, r8398, r8399;
}
{
add.f16x2 r8400, r8401, r8397;
}
{
add.f16x2 r8403, r8392, r8393;
}
{
mul.f16x2 r8406, r8403, r8387;
}
{
add.f16x2 r8409, r8395, r8406;
}
{
sub.f16x2 r8412, r8398, r8399;
}
{
mul.f16x2 r8415, r8412, r8389;
}
{
add.f16x2 r8418, r8409, r8415;
}
{
add.f16x2 r8421, r8392, r8393;
}
{
mul.f16x2 r8424, r8421, r8387;
}
{
add.f16x2 r8427, r8395, r8424;
}
{
sub.f16x2 r8430, r8398, r8399;
}
{
mul.f16x2 r8433, r8430, r8389;
}
{
sub.f16x2 r8436, r8427, r8433;
}
{
add.f16x2 r8439, r8398, r8399;
}
{
mul.f16x2 r8442, r8439, r8387;
}
{
add.f16x2 r8445, r8401, r8442;
}
{
sub.f16x2 r8448, r8392, r8393;
}
{
mul.f16x2 r8451, r8448, r8389;
}
{
sub.f16x2 r8454, r8445, r8451;
}
{
add.f16x2 r8457, r8398, r8399;
}
{
mul.f16x2 r8460, r8457, r8387;
}
{
add.f16x2 r8463, r8401, r8460;
}
{
sub.f16x2 r8466, r8392, r8393;
}
{
mul.f16x2 r8469, r8466, r8389;
}
{
add.f16x2 r8472, r8463, r8469;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r8475, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r8476, {low, high};
}
{
neg.f16x2 r8477, r8476;
}
{
add.f16x2 r8479, r8480, r8481;
}
{
add.f16x2 r8482, r8483, r8479;
}
{
add.f16x2 r8485, r8486, r8487;
}
{
add.f16x2 r8488, r8489, r8485;
}
{
add.f16x2 r8491, r8480, r8481;
}
{
mul.f16x2 r8494, r8491, r8475;
}
{
add.f16x2 r8497, r8483, r8494;
}
{
sub.f16x2 r8500, r8486, r8487;
}
{
mul.f16x2 r8503, r8500, r8477;
}
{
add.f16x2 r8506, r8497, r8503;
}
{
add.f16x2 r8509, r8480, r8481;
}
{
mul.f16x2 r8512, r8509, r8475;
}
{
add.f16x2 r8515, r8483, r8512;
}
{
sub.f16x2 r8518, r8486, r8487;
}
{
mul.f16x2 r8521, r8518, r8477;
}
{
sub.f16x2 r8524, r8515, r8521;
}
{
add.f16x2 r8527, r8486, r8487;
}
{
mul.f16x2 r8530, r8527, r8475;
}
{
add.f16x2 r8533, r8489, r8530;
}
{
sub.f16x2 r8536, r8480, r8481;
}
{
mul.f16x2 r8539, r8536, r8477;
}
{
sub.f16x2 r8542, r8533, r8539;
}
{
add.f16x2 r8545, r8486, r8487;
}
{
mul.f16x2 r8548, r8545, r8475;
}
{
add.f16x2 r8551, r8489, r8548;
}
{
sub.f16x2 r8554, r8480, r8481;
}
{
mul.f16x2 r8557, r8554, r8477;
}
{
add.f16x2 r8560, r8551, r8557;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r8563, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r8564, {low, high};
}
{
neg.f16x2 r8565, r8564;
}
{
add.f16x2 r8567, r8568, r8569;
}
{
add.f16x2 r8570, r8571, r8567;
}
{
add.f16x2 r8573, r8574, r8575;
}
{
add.f16x2 r8576, r8577, r8573;
}
{
add.f16x2 r8579, r8568, r8569;
}
{
mul.f16x2 r8582, r8579, r8563;
}
{
add.f16x2 r8585, r8571, r8582;
}
{
sub.f16x2 r8588, r8574, r8575;
}
{
mul.f16x2 r8591, r8588, r8565;
}
{
add.f16x2 r8594, r8585, r8591;
}
{
add.f16x2 r8597, r8568, r8569;
}
{
mul.f16x2 r8600, r8597, r8563;
}
{
add.f16x2 r8603, r8571, r8600;
}
{
sub.f16x2 r8606, r8574, r8575;
}
{
mul.f16x2 r8609, r8606, r8565;
}
{
sub.f16x2 r8612, r8603, r8609;
}
{
add.f16x2 r8615, r8574, r8575;
}
{
mul.f16x2 r8618, r8615, r8563;
}
{
add.f16x2 r8621, r8577, r8618;
}
{
sub.f16x2 r8624, r8568, r8569;
}
{
mul.f16x2 r8627, r8624, r8565;
}
{
sub.f16x2 r8630, r8621, r8627;
}
{
add.f16x2 r8633, r8574, r8575;
}
{
mul.f16x2 r8636, r8633, r8563;
}
{
add.f16x2 r8639, r8577, r8636;
}
{
sub.f16x2 r8642, r8568, r8569;
}
{
mul.f16x2 r8645, r8642, r8565;
}
{
add.f16x2 r8648, r8639, r8645;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r8651, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r8652, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r8653, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r8654, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r8657, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
mov.b32 r8658, {low, high};
}
{
mul.f16x2 r8667, r8506, r8651;
}
{
mul.f16x2 r8670, r8542, r8652;
}
{
sub.f16x2 r8673, r8667, r8670;
}
{
mul.f16x2 r8676, r8506, r8652;
}
{
fma.rn.f16x2 r8679, r8542, r8651, r8676;
}
{
mul.f16x2 r8683, r8594, r8653;
}
{
mul.f16x2 r8686, r8630, r8654;
}
{
sub.f16x2 r8689, r8683, r8686;
}
{
mul.f16x2 r8692, r8594, r8654;
}
{
fma.rn.f16x2 r8695, r8630, r8653, r8692;
}
{
mul.f16x2 r8699, r8524, r8653;
}
{
mul.f16x2 r8702, r8560, r8654;
}
{
sub.f16x2 r8705, r8699, r8702;
}
{
mul.f16x2 r8708, r8524, r8654;
}
{
fma.rn.f16x2 r8711, r8560, r8653, r8708;
}
{
mul.f16x2 r8715, r8612, r8657;
}
{
mul.f16x2 r8718, r8648, r8658;
}
{
sub.f16x2 r8721, r8715, r8718;
}
{
mul.f16x2 r8724, r8612, r8658;
}
{
fma.rn.f16x2 r8727, r8648, r8657, r8724;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r8731, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r8732, {low, high};
}
{
neg.f16x2 r8733, r8732;
}
{
add.f16x2 r8735, r8482, r8570;
}
{
add.f16x2 r8738, r8394, r8735;
}
{
add.f16x2 r8741, r8488, r8576;
}
{
add.f16x2 r8744, r8400, r8741;
}
{
add.f16x2 r8747, r8482, r8570;
}
{
mul.f16x2 r8750, r8747, r8731;
}
{
add.f16x2 r8753, r8394, r8750;
}
{
sub.f16x2 r8756, r8488, r8576;
}
{
mul.f16x2 r8759, r8756, r8733;
}
{
add.f16x2 r8762, r8753, r8759;
}
{
add.f16x2 r8765, r8482, r8570;
}
{
mul.f16x2 r8768, r8765, r8731;
}
{
add.f16x2 r8771, r8394, r8768;
}
{
sub.f16x2 r8774, r8488, r8576;
}
{
mul.f16x2 r8777, r8774, r8733;
}
{
sub.f16x2 r8780, r8771, r8777;
}
{
add.f16x2 r8783, r8488, r8576;
}
{
mul.f16x2 r8786, r8783, r8731;
}
{
add.f16x2 r8789, r8400, r8786;
}
{
sub.f16x2 r8792, r8482, r8570;
}
{
mul.f16x2 r8795, r8792, r8733;
}
{
sub.f16x2 r8798, r8789, r8795;
}
{
add.f16x2 r8801, r8488, r8576;
}
{
mul.f16x2 r8804, r8801, r8731;
}
{
add.f16x2 r8807, r8400, r8804;
}
{
sub.f16x2 r8810, r8482, r8570;
}
{
mul.f16x2 r8813, r8810, r8733;
}
{
add.f16x2 r8816, r8807, r8813;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r8819, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r8820, {low, high};
}
{
neg.f16x2 r8821, r8820;
}
{
add.f16x2 r8823, r8673, r8689;
}
{
add.f16x2 r8826, r8418, r8823;
}
{
add.f16x2 r8829, r8679, r8695;
}
{
add.f16x2 r8832, r8454, r8829;
}
{
add.f16x2 r8835, r8673, r8689;
}
{
mul.f16x2 r8838, r8835, r8819;
}
{
add.f16x2 r8841, r8418, r8838;
}
{
sub.f16x2 r8844, r8679, r8695;
}
{
mul.f16x2 r8847, r8844, r8821;
}
{
add.f16x2 r8850, r8841, r8847;
}
{
add.f16x2 r8853, r8673, r8689;
}
{
mul.f16x2 r8856, r8853, r8819;
}
{
add.f16x2 r8859, r8418, r8856;
}
{
sub.f16x2 r8862, r8679, r8695;
}
{
mul.f16x2 r8865, r8862, r8821;
}
{
sub.f16x2 r8868, r8859, r8865;
}
{
add.f16x2 r8871, r8679, r8695;
}
{
mul.f16x2 r8874, r8871, r8819;
}
{
add.f16x2 r8877, r8454, r8874;
}
{
sub.f16x2 r8880, r8673, r8689;
}
{
mul.f16x2 r8883, r8880, r8821;
}
{
sub.f16x2 r8886, r8877, r8883;
}
{
add.f16x2 r8889, r8679, r8695;
}
{
mul.f16x2 r8892, r8889, r8819;
}
{
add.f16x2 r8895, r8454, r8892;
}
{
sub.f16x2 r8898, r8673, r8689;
}
{
mul.f16x2 r8901, r8898, r8821;
}
{
add.f16x2 r8904, r8895, r8901;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r8907, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r8908, {low, high};
}
{
neg.f16x2 r8909, r8908;
}
{
add.f16x2 r8911, r8705, r8721;
}
{
add.f16x2 r8914, r8436, r8911;
}
{
add.f16x2 r8917, r8711, r8727;
}
{
add.f16x2 r8920, r8472, r8917;
}
{
add.f16x2 r8923, r8705, r8721;
}
{
mul.f16x2 r8926, r8923, r8907;
}
{
add.f16x2 r8929, r8436, r8926;
}
{
sub.f16x2 r8932, r8711, r8727;
}
{
mul.f16x2 r8935, r8932, r8909;
}
{
add.f16x2 r8938, r8929, r8935;
}
{
add.f16x2 r8941, r8705, r8721;
}
{
mul.f16x2 r8944, r8941, r8907;
}
{
add.f16x2 r8947, r8436, r8944;
}
{
sub.f16x2 r8950, r8711, r8727;
}
{
mul.f16x2 r8953, r8950, r8909;
}
{
sub.f16x2 r8956, r8947, r8953;
}
{
add.f16x2 r8959, r8711, r8727;
}
{
mul.f16x2 r8962, r8959, r8907;
}
{
add.f16x2 r8965, r8472, r8962;
}
{
sub.f16x2 r8968, r8705, r8721;
}
{
mul.f16x2 r8971, r8968, r8909;
}
{
sub.f16x2 r8974, r8965, r8971;
}
{
add.f16x2 r8977, r8711, r8727;
}
{
mul.f16x2 r8980, r8977, r8907;
}
{
add.f16x2 r8983, r8472, r8980;
}
{
sub.f16x2 r8986, r8705, r8721;
}
{
mul.f16x2 r8989, r8986, r8909;
}
{
add.f16x2 r8992, r8983, r8989;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r8995, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r8996, {low, high};
}
{
neg.f16x2 r8997, r8996;
}
{
add.f16x2 r8999, r9000, r9001;
}
{
add.f16x2 r9002, r9003, r8999;
}
{
add.f16x2 r9005, r9006, r9007;
}
{
add.f16x2 r9008, r9009, r9005;
}
{
add.f16x2 r9011, r9000, r9001;
}
{
mul.f16x2 r9014, r9011, r8995;
}
{
add.f16x2 r9017, r9003, r9014;
}
{
sub.f16x2 r9020, r9006, r9007;
}
{
mul.f16x2 r9023, r9020, r8997;
}
{
add.f16x2 r9026, r9017, r9023;
}
{
add.f16x2 r9029, r9000, r9001;
}
{
mul.f16x2 r9032, r9029, r8995;
}
{
add.f16x2 r9035, r9003, r9032;
}
{
sub.f16x2 r9038, r9006, r9007;
}
{
mul.f16x2 r9041, r9038, r8997;
}
{
sub.f16x2 r9044, r9035, r9041;
}
{
add.f16x2 r9047, r9006, r9007;
}
{
mul.f16x2 r9050, r9047, r8995;
}
{
add.f16x2 r9053, r9009, r9050;
}
{
sub.f16x2 r9056, r9000, r9001;
}
{
mul.f16x2 r9059, r9056, r8997;
}
{
sub.f16x2 r9062, r9053, r9059;
}
{
add.f16x2 r9065, r9006, r9007;
}
{
mul.f16x2 r9068, r9065, r8995;
}
{
add.f16x2 r9071, r9009, r9068;
}
{
sub.f16x2 r9074, r9000, r9001;
}
{
mul.f16x2 r9077, r9074, r8997;
}
{
add.f16x2 r9080, r9071, r9077;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r9083, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r9084, {low, high};
}
{
neg.f16x2 r9085, r9084;
}
{
add.f16x2 r9087, r9088, r9089;
}
{
add.f16x2 r9090, r9091, r9087;
}
{
add.f16x2 r9093, r9094, r9095;
}
{
add.f16x2 r9096, r9097, r9093;
}
{
add.f16x2 r9099, r9088, r9089;
}
{
mul.f16x2 r9102, r9099, r9083;
}
{
add.f16x2 r9105, r9091, r9102;
}
{
sub.f16x2 r9108, r9094, r9095;
}
{
mul.f16x2 r9111, r9108, r9085;
}
{
add.f16x2 r9114, r9105, r9111;
}
{
add.f16x2 r9117, r9088, r9089;
}
{
mul.f16x2 r9120, r9117, r9083;
}
{
add.f16x2 r9123, r9091, r9120;
}
{
sub.f16x2 r9126, r9094, r9095;
}
{
mul.f16x2 r9129, r9126, r9085;
}
{
sub.f16x2 r9132, r9123, r9129;
}
{
add.f16x2 r9135, r9094, r9095;
}
{
mul.f16x2 r9138, r9135, r9083;
}
{
add.f16x2 r9141, r9097, r9138;
}
{
sub.f16x2 r9144, r9088, r9089;
}
{
mul.f16x2 r9147, r9144, r9085;
}
{
sub.f16x2 r9150, r9141, r9147;
}
{
add.f16x2 r9153, r9094, r9095;
}
{
mul.f16x2 r9156, r9153, r9083;
}
{
add.f16x2 r9159, r9097, r9156;
}
{
sub.f16x2 r9162, r9088, r9089;
}
{
mul.f16x2 r9165, r9162, r9085;
}
{
add.f16x2 r9168, r9159, r9165;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r9171, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r9172, {low, high};
}
{
neg.f16x2 r9173, r9172;
}
{
add.f16x2 r9175, r9176, r9177;
}
{
add.f16x2 r9178, r9179, r9175;
}
{
add.f16x2 r9181, r9182, r9183;
}
{
add.f16x2 r9184, r9185, r9181;
}
{
add.f16x2 r9187, r9176, r9177;
}
{
mul.f16x2 r9190, r9187, r9171;
}
{
add.f16x2 r9193, r9179, r9190;
}
{
sub.f16x2 r9196, r9182, r9183;
}
{
mul.f16x2 r9199, r9196, r9173;
}
{
add.f16x2 r9202, r9193, r9199;
}
{
add.f16x2 r9205, r9176, r9177;
}
{
mul.f16x2 r9208, r9205, r9171;
}
{
add.f16x2 r9211, r9179, r9208;
}
{
sub.f16x2 r9214, r9182, r9183;
}
{
mul.f16x2 r9217, r9214, r9173;
}
{
sub.f16x2 r9220, r9211, r9217;
}
{
add.f16x2 r9223, r9182, r9183;
}
{
mul.f16x2 r9226, r9223, r9171;
}
{
add.f16x2 r9229, r9185, r9226;
}
{
sub.f16x2 r9232, r9176, r9177;
}
{
mul.f16x2 r9235, r9232, r9173;
}
{
sub.f16x2 r9238, r9229, r9235;
}
{
add.f16x2 r9241, r9182, r9183;
}
{
mul.f16x2 r9244, r9241, r9171;
}
{
add.f16x2 r9247, r9185, r9244;
}
{
sub.f16x2 r9250, r9176, r9177;
}
{
mul.f16x2 r9253, r9250, r9173;
}
{
add.f16x2 r9256, r9247, r9253;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r9259, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r9260, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r9261, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r9262, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r9265, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
mov.b32 r9266, {low, high};
}
{
mul.f16x2 r9275, r9114, r9259;
}
{
mul.f16x2 r9278, r9150, r9260;
}
{
sub.f16x2 r9281, r9275, r9278;
}
{
mul.f16x2 r9284, r9114, r9260;
}
{
fma.rn.f16x2 r9287, r9150, r9259, r9284;
}
{
mul.f16x2 r9291, r9202, r9261;
}
{
mul.f16x2 r9294, r9238, r9262;
}
{
sub.f16x2 r9297, r9291, r9294;
}
{
mul.f16x2 r9300, r9202, r9262;
}
{
fma.rn.f16x2 r9303, r9238, r9261, r9300;
}
{
mul.f16x2 r9307, r9132, r9261;
}
{
mul.f16x2 r9310, r9168, r9262;
}
{
sub.f16x2 r9313, r9307, r9310;
}
{
mul.f16x2 r9316, r9132, r9262;
}
{
fma.rn.f16x2 r9319, r9168, r9261, r9316;
}
{
mul.f16x2 r9323, r9220, r9265;
}
{
mul.f16x2 r9326, r9256, r9266;
}
{
sub.f16x2 r9329, r9323, r9326;
}
{
mul.f16x2 r9332, r9220, r9266;
}
{
fma.rn.f16x2 r9335, r9256, r9265, r9332;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r9339, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r9340, {low, high};
}
{
neg.f16x2 r9341, r9340;
}
{
add.f16x2 r9343, r9090, r9178;
}
{
add.f16x2 r9346, r9002, r9343;
}
{
add.f16x2 r9349, r9096, r9184;
}
{
add.f16x2 r9352, r9008, r9349;
}
{
add.f16x2 r9355, r9090, r9178;
}
{
mul.f16x2 r9358, r9355, r9339;
}
{
add.f16x2 r9361, r9002, r9358;
}
{
sub.f16x2 r9364, r9096, r9184;
}
{
mul.f16x2 r9367, r9364, r9341;
}
{
add.f16x2 r9370, r9361, r9367;
}
{
add.f16x2 r9373, r9090, r9178;
}
{
mul.f16x2 r9376, r9373, r9339;
}
{
add.f16x2 r9379, r9002, r9376;
}
{
sub.f16x2 r9382, r9096, r9184;
}
{
mul.f16x2 r9385, r9382, r9341;
}
{
sub.f16x2 r9388, r9379, r9385;
}
{
add.f16x2 r9391, r9096, r9184;
}
{
mul.f16x2 r9394, r9391, r9339;
}
{
add.f16x2 r9397, r9008, r9394;
}
{
sub.f16x2 r9400, r9090, r9178;
}
{
mul.f16x2 r9403, r9400, r9341;
}
{
sub.f16x2 r9406, r9397, r9403;
}
{
add.f16x2 r9409, r9096, r9184;
}
{
mul.f16x2 r9412, r9409, r9339;
}
{
add.f16x2 r9415, r9008, r9412;
}
{
sub.f16x2 r9418, r9090, r9178;
}
{
mul.f16x2 r9421, r9418, r9341;
}
{
add.f16x2 r9424, r9415, r9421;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r9427, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r9428, {low, high};
}
{
neg.f16x2 r9429, r9428;
}
{
add.f16x2 r9431, r9281, r9297;
}
{
add.f16x2 r9434, r9026, r9431;
}
{
add.f16x2 r9437, r9287, r9303;
}
{
add.f16x2 r9440, r9062, r9437;
}
{
add.f16x2 r9443, r9281, r9297;
}
{
mul.f16x2 r9446, r9443, r9427;
}
{
add.f16x2 r9449, r9026, r9446;
}
{
sub.f16x2 r9452, r9287, r9303;
}
{
mul.f16x2 r9455, r9452, r9429;
}
{
add.f16x2 r9458, r9449, r9455;
}
{
add.f16x2 r9461, r9281, r9297;
}
{
mul.f16x2 r9464, r9461, r9427;
}
{
add.f16x2 r9467, r9026, r9464;
}
{
sub.f16x2 r9470, r9287, r9303;
}
{
mul.f16x2 r9473, r9470, r9429;
}
{
sub.f16x2 r9476, r9467, r9473;
}
{
add.f16x2 r9479, r9287, r9303;
}
{
mul.f16x2 r9482, r9479, r9427;
}
{
add.f16x2 r9485, r9062, r9482;
}
{
sub.f16x2 r9488, r9281, r9297;
}
{
mul.f16x2 r9491, r9488, r9429;
}
{
sub.f16x2 r9494, r9485, r9491;
}
{
add.f16x2 r9497, r9287, r9303;
}
{
mul.f16x2 r9500, r9497, r9427;
}
{
add.f16x2 r9503, r9062, r9500;
}
{
sub.f16x2 r9506, r9281, r9297;
}
{
mul.f16x2 r9509, r9506, r9429;
}
{
add.f16x2 r9512, r9503, r9509;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r9515, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r9516, {low, high};
}
{
neg.f16x2 r9517, r9516;
}
{
add.f16x2 r9519, r9313, r9329;
}
{
add.f16x2 r9522, r9044, r9519;
}
{
add.f16x2 r9525, r9319, r9335;
}
{
add.f16x2 r9528, r9080, r9525;
}
{
add.f16x2 r9531, r9313, r9329;
}
{
mul.f16x2 r9534, r9531, r9515;
}
{
add.f16x2 r9537, r9044, r9534;
}
{
sub.f16x2 r9540, r9319, r9335;
}
{
mul.f16x2 r9543, r9540, r9517;
}
{
add.f16x2 r9546, r9537, r9543;
}
{
add.f16x2 r9549, r9313, r9329;
}
{
mul.f16x2 r9552, r9549, r9515;
}
{
add.f16x2 r9555, r9044, r9552;
}
{
sub.f16x2 r9558, r9319, r9335;
}
{
mul.f16x2 r9561, r9558, r9517;
}
{
sub.f16x2 r9564, r9555, r9561;
}
{
add.f16x2 r9567, r9319, r9335;
}
{
mul.f16x2 r9570, r9567, r9515;
}
{
add.f16x2 r9573, r9080, r9570;
}
{
sub.f16x2 r9576, r9313, r9329;
}
{
mul.f16x2 r9579, r9576, r9517;
}
{
sub.f16x2 r9582, r9573, r9579;
}
{
add.f16x2 r9585, r9319, r9335;
}
{
mul.f16x2 r9588, r9585, r9515;
}
{
add.f16x2 r9591, r9080, r9588;
}
{
sub.f16x2 r9594, r9313, r9329;
}
{
mul.f16x2 r9597, r9594, r9517;
}
{
add.f16x2 r9600, r9591, r9597;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f898;
cvt.rn.f16.f32 high, f898;
mov.b32 r9603, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f900;
cvt.rn.f16.f32 high, f900;
mov.b32 r9604, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f902;
cvt.rn.f16.f32 high, f902;
mov.b32 r9605, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f904;
cvt.rn.f16.f32 high, f904;
mov.b32 r9606, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f906;
cvt.rn.f16.f32 high, f906;
mov.b32 r9607, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f908;
cvt.rn.f16.f32 high, f908;
mov.b32 r9608, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f910;
cvt.rn.f16.f32 high, f910;
mov.b32 r9609, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f912;
cvt.rn.f16.f32 high, f912;
mov.b32 r9610, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f914;
cvt.rn.f16.f32 high, f914;
mov.b32 r9611, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f916;
cvt.rn.f16.f32 high, f916;
mov.b32 r9612, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f918;
cvt.rn.f16.f32 high, f918;
mov.b32 r9613, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f920;
cvt.rn.f16.f32 high, f920;
mov.b32 r9614, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f922;
cvt.rn.f16.f32 high, f922;
mov.b32 r9615, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f924;
cvt.rn.f16.f32 high, f924;
mov.b32 r9616, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f926;
cvt.rn.f16.f32 high, f926;
mov.b32 r9617, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f928;
cvt.rn.f16.f32 high, f928;
mov.b32 r9618, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f934;
cvt.rn.f16.f32 high, f934;
mov.b32 r9621, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f936;
cvt.rn.f16.f32 high, f936;
mov.b32 r9622, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f942;
cvt.rn.f16.f32 high, f942;
mov.b32 r9625, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f944;
cvt.rn.f16.f32 high, f944;
mov.b32 r9626, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f950;
cvt.rn.f16.f32 high, f950;
mov.b32 r9629, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f952;
cvt.rn.f16.f32 high, f952;
mov.b32 r9630, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f958;
cvt.rn.f16.f32 high, f958;
mov.b32 r9633, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f960;
cvt.rn.f16.f32 high, f960;
mov.b32 r9634, {low, high};
}
{
mul.f16x2 r9655, r8826, r9603;
}
{
mul.f16x2 r9658, r8832, r9604;
}
{
sub.f16x2 r9661, r9655, r9658;
}
{
mul.f16x2 r9664, r8826, r9604;
}
{
fma.rn.f16x2 r9667, r8832, r9603, r9664;
}
{
mul.f16x2 r9671, r9434, r9605;
}
{
mul.f16x2 r9674, r9440, r9606;
}
{
sub.f16x2 r9677, r9671, r9674;
}
{
mul.f16x2 r9680, r9434, r9606;
}
{
fma.rn.f16x2 r9683, r9440, r9605, r9680;
}
{
mul.f16x2 r9687, r8914, r9605;
}
{
mul.f16x2 r9690, r8920, r9606;
}
{
sub.f16x2 r9693, r9687, r9690;
}
{
mul.f16x2 r9696, r8914, r9606;
}
{
fma.rn.f16x2 r9699, r8920, r9605, r9696;
}
{
mul.f16x2 r9703, r9522, r9609;
}
{
mul.f16x2 r9706, r9528, r9610;
}
{
sub.f16x2 r9709, r9703, r9706;
}
{
mul.f16x2 r9712, r9522, r9610;
}
{
fma.rn.f16x2 r9715, r9528, r9609, r9712;
}
{
mul.f16x2 r9719, r8762, r9607;
}
{
mul.f16x2 r9722, r8798, r9608;
}
{
sub.f16x2 r9725, r9719, r9722;
}
{
mul.f16x2 r9728, r8762, r9608;
}
{
fma.rn.f16x2 r9731, r8798, r9607, r9728;
}
{
mul.f16x2 r9735, r9370, r9613;
}
{
mul.f16x2 r9738, r9406, r9614;
}
{
sub.f16x2 r9741, r9735, r9738;
}
{
mul.f16x2 r9744, r9370, r9614;
}
{
fma.rn.f16x2 r9747, r9406, r9613, r9744;
}
{
mul.f16x2 r9751, r8850, r9609;
}
{
mul.f16x2 r9754, r8886, r9610;
}
{
sub.f16x2 r9757, r9751, r9754;
}
{
mul.f16x2 r9760, r8850, r9610;
}
{
fma.rn.f16x2 r9763, r8886, r9609, r9760;
}
{
mul.f16x2 r9767, r9458, r9617;
}
{
mul.f16x2 r9770, r9494, r9618;
}
{
sub.f16x2 r9773, r9767, r9770;
}
{
mul.f16x2 r9776, r9458, r9618;
}
{
fma.rn.f16x2 r9779, r9494, r9617, r9776;
}
{
mul.f16x2 r9783, r8938, r9611;
}
{
mul.f16x2 r9786, r8974, r9612;
}
{
sub.f16x2 r9789, r9783, r9786;
}
{
mul.f16x2 r9792, r8938, r9612;
}
{
fma.rn.f16x2 r9795, r8974, r9611, r9792;
}
{
mul.f16x2 r9799, r9546, r9621;
}
{
mul.f16x2 r9802, r9582, r9622;
}
{
sub.f16x2 r9805, r9799, r9802;
}
{
mul.f16x2 r9808, r9546, r9622;
}
{
fma.rn.f16x2 r9811, r9582, r9621, r9808;
}
{
mul.f16x2 r9815, r8780, r9613;
}
{
mul.f16x2 r9818, r8816, r9614;
}
{
sub.f16x2 r9821, r9815, r9818;
}
{
mul.f16x2 r9824, r8780, r9614;
}
{
fma.rn.f16x2 r9827, r8816, r9613, r9824;
}
{
mul.f16x2 r9831, r9388, r9625;
}
{
mul.f16x2 r9834, r9424, r9626;
}
{
sub.f16x2 r9837, r9831, r9834;
}
{
mul.f16x2 r9840, r9388, r9626;
}
{
fma.rn.f16x2 r9843, r9424, r9625, r9840;
}
{
mul.f16x2 r9847, r8868, r9615;
}
{
mul.f16x2 r9850, r8904, r9616;
}
{
sub.f16x2 r9853, r9847, r9850;
}
{
mul.f16x2 r9856, r8868, r9616;
}
{
fma.rn.f16x2 r9859, r8904, r9615, r9856;
}
{
mul.f16x2 r9863, r9476, r9629;
}
{
mul.f16x2 r9866, r9512, r9630;
}
{
sub.f16x2 r9869, r9863, r9866;
}
{
mul.f16x2 r9872, r9476, r9630;
}
{
fma.rn.f16x2 r9875, r9512, r9629, r9872;
}
{
mul.f16x2 r9879, r8956, r9617;
}
{
mul.f16x2 r9882, r8992, r9618;
}
{
sub.f16x2 r9885, r9879, r9882;
}
{
mul.f16x2 r9888, r8956, r9618;
}
{
fma.rn.f16x2 r9891, r8992, r9617, r9888;
}
{
mul.f16x2 r9895, r9564, r9633;
}
{
mul.f16x2 r9898, r9600, r9634;
}
{
sub.f16x2 r9901, r9895, r9898;
}
{
mul.f16x2 r9904, r9564, r9634;
}
{
fma.rn.f16x2 r9907, r9600, r9633, r9904;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r9911, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r9912, {low, high};
}
{
neg.f16x2 r9913, r9912;
}
{
add.f16x2 r9915, r8738, r9346;
}
{
add.f16x2 %0, r8130, r9915;
}
{
add.f16x2 r9921, r8744, r9352;
}
{
add.f16x2 %1, r8136, r9921;
}
{
add.f16x2 r9927, r8738, r9346;
}
{
mul.f16x2 r9930, r9927, r9911;
}
{
add.f16x2 r9933, r8130, r9930;
}
{
sub.f16x2 r9936, r8744, r9352;
}
{
mul.f16x2 r9939, r9936, r9913;
}
{
add.f16x2 %18, r9933, r9939;
}
{
add.f16x2 r9945, r8738, r9346;
}
{
mul.f16x2 r9948, r9945, r9911;
}
{
add.f16x2 r9951, r8130, r9948;
}
{
sub.f16x2 r9954, r8744, r9352;
}
{
mul.f16x2 r9957, r9954, r9913;
}
{
sub.f16x2 %36, r9951, r9957;
}
{
add.f16x2 r9963, r8744, r9352;
}
{
mul.f16x2 r9966, r9963, r9911;
}
{
add.f16x2 r9969, r8136, r9966;
}
{
sub.f16x2 r9972, r8738, r9346;
}
{
mul.f16x2 r9975, r9972, r9913;
}
{
sub.f16x2 %19, r9969, r9975;
}
{
add.f16x2 r9981, r8744, r9352;
}
{
mul.f16x2 r9984, r9981, r9911;
}
{
add.f16x2 r9987, r8136, r9984;
}
{
sub.f16x2 r9990, r8738, r9346;
}
{
mul.f16x2 r9993, r9990, r9913;
}
{
add.f16x2 %37, r9987, r9993;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r9999, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r10000, {low, high};
}
{
neg.f16x2 r10001, r10000;
}
{
add.f16x2 r10003, r9661, r9677;
}
{
add.f16x2 %2, r8218, r10003;
}
{
add.f16x2 r10009, r9667, r9683;
}
{
add.f16x2 %3, r8224, r10009;
}
{
add.f16x2 r10015, r9661, r9677;
}
{
mul.f16x2 r10018, r10015, r9999;
}
{
add.f16x2 r10021, r8218, r10018;
}
{
sub.f16x2 r10024, r9667, r9683;
}
{
mul.f16x2 r10027, r10024, r10001;
}
{
add.f16x2 %20, r10021, r10027;
}
{
add.f16x2 r10033, r9661, r9677;
}
{
mul.f16x2 r10036, r10033, r9999;
}
{
add.f16x2 r10039, r8218, r10036;
}
{
sub.f16x2 r10042, r9667, r9683;
}
{
mul.f16x2 r10045, r10042, r10001;
}
{
sub.f16x2 %38, r10039, r10045;
}
{
add.f16x2 r10051, r9667, r9683;
}
{
mul.f16x2 r10054, r10051, r9999;
}
{
add.f16x2 r10057, r8224, r10054;
}
{
sub.f16x2 r10060, r9661, r9677;
}
{
mul.f16x2 r10063, r10060, r10001;
}
{
sub.f16x2 %21, r10057, r10063;
}
{
add.f16x2 r10069, r9667, r9683;
}
{
mul.f16x2 r10072, r10069, r9999;
}
{
add.f16x2 r10075, r8224, r10072;
}
{
sub.f16x2 r10078, r9661, r9677;
}
{
mul.f16x2 r10081, r10078, r10001;
}
{
add.f16x2 %39, r10075, r10081;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r10087, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r10088, {low, high};
}
{
neg.f16x2 r10089, r10088;
}
{
add.f16x2 r10091, r9693, r9709;
}
{
add.f16x2 %4, r8306, r10091;
}
{
add.f16x2 r10097, r9699, r9715;
}
{
add.f16x2 %5, r8312, r10097;
}
{
add.f16x2 r10103, r9693, r9709;
}
{
mul.f16x2 r10106, r10103, r10087;
}
{
add.f16x2 r10109, r8306, r10106;
}
{
sub.f16x2 r10112, r9699, r9715;
}
{
mul.f16x2 r10115, r10112, r10089;
}
{
add.f16x2 %22, r10109, r10115;
}
{
add.f16x2 r10121, r9693, r9709;
}
{
mul.f16x2 r10124, r10121, r10087;
}
{
add.f16x2 r10127, r8306, r10124;
}
{
sub.f16x2 r10130, r9699, r9715;
}
{
mul.f16x2 r10133, r10130, r10089;
}
{
sub.f16x2 %40, r10127, r10133;
}
{
add.f16x2 r10139, r9699, r9715;
}
{
mul.f16x2 r10142, r10139, r10087;
}
{
add.f16x2 r10145, r8312, r10142;
}
{
sub.f16x2 r10148, r9693, r9709;
}
{
mul.f16x2 r10151, r10148, r10089;
}
{
sub.f16x2 %23, r10145, r10151;
}
{
add.f16x2 r10157, r9699, r9715;
}
{
mul.f16x2 r10160, r10157, r10087;
}
{
add.f16x2 r10163, r8312, r10160;
}
{
sub.f16x2 r10166, r9693, r9709;
}
{
mul.f16x2 r10169, r10166, r10089;
}
{
add.f16x2 %41, r10163, r10169;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r10175, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r10176, {low, high};
}
{
neg.f16x2 r10177, r10176;
}
{
add.f16x2 r10179, r9725, r9741;
}
{
add.f16x2 %6, r8154, r10179;
}
{
add.f16x2 r10185, r9731, r9747;
}
{
add.f16x2 %7, r8190, r10185;
}
{
add.f16x2 r10191, r9725, r9741;
}
{
mul.f16x2 r10194, r10191, r10175;
}
{
add.f16x2 r10197, r8154, r10194;
}
{
sub.f16x2 r10200, r9731, r9747;
}
{
mul.f16x2 r10203, r10200, r10177;
}
{
add.f16x2 %24, r10197, r10203;
}
{
add.f16x2 r10209, r9725, r9741;
}
{
mul.f16x2 r10212, r10209, r10175;
}
{
add.f16x2 r10215, r8154, r10212;
}
{
sub.f16x2 r10218, r9731, r9747;
}
{
mul.f16x2 r10221, r10218, r10177;
}
{
sub.f16x2 %42, r10215, r10221;
}
{
add.f16x2 r10227, r9731, r9747;
}
{
mul.f16x2 r10230, r10227, r10175;
}
{
add.f16x2 r10233, r8190, r10230;
}
{
sub.f16x2 r10236, r9725, r9741;
}
{
mul.f16x2 r10239, r10236, r10177;
}
{
sub.f16x2 %25, r10233, r10239;
}
{
add.f16x2 r10245, r9731, r9747;
}
{
mul.f16x2 r10248, r10245, r10175;
}
{
add.f16x2 r10251, r8190, r10248;
}
{
sub.f16x2 r10254, r9725, r9741;
}
{
mul.f16x2 r10257, r10254, r10177;
}
{
add.f16x2 %43, r10251, r10257;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r10263, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r10264, {low, high};
}
{
neg.f16x2 r10265, r10264;
}
{
add.f16x2 r10267, r9757, r9773;
}
{
add.f16x2 %8, r8242, r10267;
}
{
add.f16x2 r10273, r9763, r9779;
}
{
add.f16x2 %9, r8278, r10273;
}
{
add.f16x2 r10279, r9757, r9773;
}
{
mul.f16x2 r10282, r10279, r10263;
}
{
add.f16x2 r10285, r8242, r10282;
}
{
sub.f16x2 r10288, r9763, r9779;
}
{
mul.f16x2 r10291, r10288, r10265;
}
{
add.f16x2 %26, r10285, r10291;
}
{
add.f16x2 r10297, r9757, r9773;
}
{
mul.f16x2 r10300, r10297, r10263;
}
{
add.f16x2 r10303, r8242, r10300;
}
{
sub.f16x2 r10306, r9763, r9779;
}
{
mul.f16x2 r10309, r10306, r10265;
}
{
sub.f16x2 %44, r10303, r10309;
}
{
add.f16x2 r10315, r9763, r9779;
}
{
mul.f16x2 r10318, r10315, r10263;
}
{
add.f16x2 r10321, r8278, r10318;
}
{
sub.f16x2 r10324, r9757, r9773;
}
{
mul.f16x2 r10327, r10324, r10265;
}
{
sub.f16x2 %27, r10321, r10327;
}
{
add.f16x2 r10333, r9763, r9779;
}
{
mul.f16x2 r10336, r10333, r10263;
}
{
add.f16x2 r10339, r8278, r10336;
}
{
sub.f16x2 r10342, r9757, r9773;
}
{
mul.f16x2 r10345, r10342, r10265;
}
{
add.f16x2 %45, r10339, r10345;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r10351, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r10352, {low, high};
}
{
neg.f16x2 r10353, r10352;
}
{
add.f16x2 r10355, r9789, r9805;
}
{
add.f16x2 %10, r8330, r10355;
}
{
add.f16x2 r10361, r9795, r9811;
}
{
add.f16x2 %11, r8366, r10361;
}
{
add.f16x2 r10367, r9789, r9805;
}
{
mul.f16x2 r10370, r10367, r10351;
}
{
add.f16x2 r10373, r8330, r10370;
}
{
sub.f16x2 r10376, r9795, r9811;
}
{
mul.f16x2 r10379, r10376, r10353;
}
{
add.f16x2 %28, r10373, r10379;
}
{
add.f16x2 r10385, r9789, r9805;
}
{
mul.f16x2 r10388, r10385, r10351;
}
{
add.f16x2 r10391, r8330, r10388;
}
{
sub.f16x2 r10394, r9795, r9811;
}
{
mul.f16x2 r10397, r10394, r10353;
}
{
sub.f16x2 %46, r10391, r10397;
}
{
add.f16x2 r10403, r9795, r9811;
}
{
mul.f16x2 r10406, r10403, r10351;
}
{
add.f16x2 r10409, r8366, r10406;
}
{
sub.f16x2 r10412, r9789, r9805;
}
{
mul.f16x2 r10415, r10412, r10353;
}
{
sub.f16x2 %29, r10409, r10415;
}
{
add.f16x2 r10421, r9795, r9811;
}
{
mul.f16x2 r10424, r10421, r10351;
}
{
add.f16x2 r10427, r8366, r10424;
}
{
sub.f16x2 r10430, r9789, r9805;
}
{
mul.f16x2 r10433, r10430, r10353;
}
{
add.f16x2 %47, r10427, r10433;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r10439, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r10440, {low, high};
}
{
neg.f16x2 r10441, r10440;
}
{
add.f16x2 r10443, r9821, r9837;
}
{
add.f16x2 %12, r8172, r10443;
}
{
add.f16x2 r10449, r9827, r9843;
}
{
add.f16x2 %13, r8208, r10449;
}
{
add.f16x2 r10455, r9821, r9837;
}
{
mul.f16x2 r10458, r10455, r10439;
}
{
add.f16x2 r10461, r8172, r10458;
}
{
sub.f16x2 r10464, r9827, r9843;
}
{
mul.f16x2 r10467, r10464, r10441;
}
{
add.f16x2 %30, r10461, r10467;
}
{
add.f16x2 r10473, r9821, r9837;
}
{
mul.f16x2 r10476, r10473, r10439;
}
{
add.f16x2 r10479, r8172, r10476;
}
{
sub.f16x2 r10482, r9827, r9843;
}
{
mul.f16x2 r10485, r10482, r10441;
}
{
sub.f16x2 %48, r10479, r10485;
}
{
add.f16x2 r10491, r9827, r9843;
}
{
mul.f16x2 r10494, r10491, r10439;
}
{
add.f16x2 r10497, r8208, r10494;
}
{
sub.f16x2 r10500, r9821, r9837;
}
{
mul.f16x2 r10503, r10500, r10441;
}
{
sub.f16x2 %31, r10497, r10503;
}
{
add.f16x2 r10509, r9827, r9843;
}
{
mul.f16x2 r10512, r10509, r10439;
}
{
add.f16x2 r10515, r8208, r10512;
}
{
sub.f16x2 r10518, r9821, r9837;
}
{
mul.f16x2 r10521, r10518, r10441;
}
{
add.f16x2 %49, r10515, r10521;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r10527, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r10528, {low, high};
}
{
neg.f16x2 r10529, r10528;
}
{
add.f16x2 r10531, r9853, r9869;
}
{
add.f16x2 %14, r8260, r10531;
}
{
add.f16x2 r10537, r9859, r9875;
}
{
add.f16x2 %15, r8296, r10537;
}
{
add.f16x2 r10543, r9853, r9869;
}
{
mul.f16x2 r10546, r10543, r10527;
}
{
add.f16x2 r10549, r8260, r10546;
}
{
sub.f16x2 r10552, r9859, r9875;
}
{
mul.f16x2 r10555, r10552, r10529;
}
{
add.f16x2 %32, r10549, r10555;
}
{
add.f16x2 r10561, r9853, r9869;
}
{
mul.f16x2 r10564, r10561, r10527;
}
{
add.f16x2 r10567, r8260, r10564;
}
{
sub.f16x2 r10570, r9859, r9875;
}
{
mul.f16x2 r10573, r10570, r10529;
}
{
sub.f16x2 %50, r10567, r10573;
}
{
add.f16x2 r10579, r9859, r9875;
}
{
mul.f16x2 r10582, r10579, r10527;
}
{
add.f16x2 r10585, r8296, r10582;
}
{
sub.f16x2 r10588, r9853, r9869;
}
{
mul.f16x2 r10591, r10588, r10529;
}
{
sub.f16x2 %33, r10585, r10591;
}
{
add.f16x2 r10597, r9859, r9875;
}
{
mul.f16x2 r10600, r10597, r10527;
}
{
add.f16x2 r10603, r8296, r10600;
}
{
sub.f16x2 r10606, r9853, r9869;
}
{
mul.f16x2 r10609, r10606, r10529;
}
{
add.f16x2 %51, r10603, r10609;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1034;
cvt.rn.f16.f32 high, f1034;
mov.b32 r10615, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1036;
cvt.rn.f16.f32 high, f1036;
mov.b32 r10616, {low, high};
}
{
neg.f16x2 r10617, r10616;
}
{
add.f16x2 r10619, r9885, r9901;
}
{
add.f16x2 %16, r8348, r10619;
}
{
add.f16x2 r10625, r9891, r9907;
}
{
add.f16x2 %17, r8384, r10625;
}
{
add.f16x2 r10631, r9885, r9901;
}
{
mul.f16x2 r10634, r10631, r10615;
}
{
add.f16x2 r10637, r8348, r10634;
}
{
sub.f16x2 r10640, r9891, r9907;
}
{
mul.f16x2 r10643, r10640, r10617;
}
{
add.f16x2 %34, r10637, r10643;
}
{
add.f16x2 r10649, r9885, r9901;
}
{
mul.f16x2 r10652, r10649, r10615;
}
{
add.f16x2 r10655, r8348, r10652;
}
{
sub.f16x2 r10658, r9891, r9907;
}
{
mul.f16x2 r10661, r10658, r10617;
}
{
sub.f16x2 %52, r10655, r10661;
}
{
add.f16x2 r10667, r9891, r9907;
}
{
mul.f16x2 r10670, r10667, r10615;
}
{
add.f16x2 r10673, r8384, r10670;
}
{
sub.f16x2 r10676, r9885, r9901;
}
{
mul.f16x2 r10679, r10676, r10617;
}
{
sub.f16x2 %35, r10673, r10679;
}
{
add.f16x2 r10685, r9891, r9907;
}
{
mul.f16x2 r10688, r10685, r10615;
}
{
add.f16x2 r10691, r8384, r10688;
}
{
sub.f16x2 r10694, r9885, r9901;
}
{
mul.f16x2 r10697, r10694, r10617;
}
{
add.f16x2 %53, r10691, r10697;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[4].x)));
};


#endif
