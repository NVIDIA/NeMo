#ifndef CUFFTDX_FFT_1000_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_1000_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<939, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<231>;
.reg .b32 r<3038>;
.reg .b64 rd<7>;
mov.u32 r3019, %tid.y;
shl.b32 r3020, r3019, 1;
mov.u32 r3021, %20;
mad.lo.s32 r3022, r3020, 4000, r3021;
mov.u32 r3023, %tid.x;
mov.f32 f194, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1, {low, high};
}
mov.f32 f200, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2, {low, high};
}
mov.f32 f202, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r3, {low, high};
}
mov.f32 f204, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r4, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r5, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
{
add.f16x2 r9, %25, %37;
}
{
add.f16x2 r12, %21, r9;
}
{
add.f16x2 r15, %29, %33;
}
{
add.f16x2 r18, r12, r15;
}
{
add.f16x2 r21, %26, %38;
}
{
add.f16x2 r24, %22, r21;
}
{
add.f16x2 r27, %30, %34;
}
{
add.f16x2 r30, r24, r27;
}
{
add.f16x2 r33, %25, %37;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %21, r36;
}
{
add.f16x2 r42, %29, %33;
}
{
mul.f16x2 r45, r42, r3;
}
{
add.f16x2 r48, r39, r45;
}
{
sub.f16x2 r51, %26, %38;
}
{
mul.f16x2 r54, r51, r2;
}
{
sub.f16x2 r57, %30, %34;
}
{
mul.f16x2 r60, r57, r4;
}
{
add.f16x2 r63, r54, r60;
}
{
sub.f16x2 r66, r48, r63;
}
{
add.f16x2 r69, %25, %37;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %21, r72;
}
{
add.f16x2 r78, %29, %33;
}
{
mul.f16x2 r81, r78, r3;
}
{
add.f16x2 r84, r75, r81;
}
{
sub.f16x2 r87, %26, %38;
}
{
mul.f16x2 r90, r87, r2;
}
{
sub.f16x2 r93, %30, %34;
}
{
mul.f16x2 r96, r93, r4;
}
{
add.f16x2 r99, r90, r96;
}
{
add.f16x2 r102, r84, r99;
}
{
add.f16x2 r105, %25, %37;
}
{
mul.f16x2 r108, r105, r3;
}
{
add.f16x2 r111, %21, r108;
}
{
add.f16x2 r114, %29, %33;
}
{
mul.f16x2 r117, r114, r5;
}
{
add.f16x2 r120, r111, r117;
}
{
sub.f16x2 r123, %26, %38;
}
{
mul.f16x2 r126, r123, r4;
}
{
sub.f16x2 r129, %30, %34;
}
{
mul.f16x2 r132, r129, r7;
}
{
add.f16x2 r135, r126, r132;
}
{
sub.f16x2 r138, r120, r135;
}
{
add.f16x2 r141, %25, %37;
}
{
mul.f16x2 r144, r141, r3;
}
{
add.f16x2 r147, %21, r144;
}
{
add.f16x2 r150, %29, %33;
}
{
mul.f16x2 r153, r150, r5;
}
{
add.f16x2 r156, r147, r153;
}
{
sub.f16x2 r159, %26, %38;
}
{
mul.f16x2 r162, r159, r4;
}
{
sub.f16x2 r165, %30, %34;
}
{
mul.f16x2 r168, r165, r7;
}
{
add.f16x2 r171, r162, r168;
}
{
add.f16x2 r174, r156, r171;
}
{
add.f16x2 r177, %26, %38;
}
{
mul.f16x2 r180, r177, r1;
}
{
add.f16x2 r183, %22, r180;
}
{
add.f16x2 r186, %30, %34;
}
{
mul.f16x2 r189, r186, r3;
}
{
add.f16x2 r192, r183, r189;
}
{
sub.f16x2 r195, %25, %37;
}
{
mul.f16x2 r198, r195, r2;
}
{
sub.f16x2 r201, %29, %33;
}
{
mul.f16x2 r204, r201, r4;
}
{
add.f16x2 r207, r198, r204;
}
{
add.f16x2 r210, r192, r207;
}
{
add.f16x2 r213, %26, %38;
}
{
mul.f16x2 r216, r213, r1;
}
{
add.f16x2 r219, %22, r216;
}
{
add.f16x2 r222, %30, %34;
}
{
mul.f16x2 r225, r222, r3;
}
{
add.f16x2 r228, r219, r225;
}
{
sub.f16x2 r231, %25, %37;
}
{
mul.f16x2 r234, r231, r2;
}
{
sub.f16x2 r237, %29, %33;
}
{
mul.f16x2 r240, r237, r4;
}
{
add.f16x2 r243, r234, r240;
}
{
sub.f16x2 r246, r228, r243;
}
{
add.f16x2 r249, %26, %38;
}
{
mul.f16x2 r252, r249, r3;
}
{
add.f16x2 r255, %22, r252;
}
{
add.f16x2 r258, %30, %34;
}
{
mul.f16x2 r261, r258, r5;
}
{
add.f16x2 r264, r255, r261;
}
{
sub.f16x2 r267, %25, %37;
}
{
mul.f16x2 r270, r267, r4;
}
{
sub.f16x2 r273, %29, %33;
}
{
mul.f16x2 r276, r273, r7;
}
{
add.f16x2 r279, r270, r276;
}
{
add.f16x2 r282, r264, r279;
}
{
add.f16x2 r285, %26, %38;
}
{
mul.f16x2 r288, r285, r3;
}
{
add.f16x2 r291, %22, r288;
}
{
add.f16x2 r294, %30, %34;
}
{
mul.f16x2 r297, r294, r5;
}
{
add.f16x2 r300, r291, r297;
}
{
sub.f16x2 r303, %25, %37;
}
{
mul.f16x2 r306, r303, r4;
}
{
sub.f16x2 r309, %29, %33;
}
{
mul.f16x2 r312, r309, r7;
}
{
add.f16x2 r315, r306, r312;
}
{
sub.f16x2 r318, r300, r315;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r321, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r322, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r323, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r324, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r325, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r326, {low, high};
}
{
neg.f16x2 r327, r326;
}
{
add.f16x2 r329, %27, %39;
}
{
add.f16x2 r332, %23, r329;
}
{
add.f16x2 r335, %31, %35;
}
{
add.f16x2 r338, r332, r335;
}
{
add.f16x2 r341, %28, %40;
}
{
add.f16x2 r344, %24, r341;
}
{
add.f16x2 r347, %32, %36;
}
{
add.f16x2 r350, r344, r347;
}
{
add.f16x2 r353, %27, %39;
}
{
mul.f16x2 r356, r353, r321;
}
{
add.f16x2 r359, %23, r356;
}
{
add.f16x2 r362, %31, %35;
}
{
mul.f16x2 r365, r362, r323;
}
{
add.f16x2 r368, r359, r365;
}
{
sub.f16x2 r371, %28, %40;
}
{
mul.f16x2 r374, r371, r322;
}
{
sub.f16x2 r377, %32, %36;
}
{
mul.f16x2 r380, r377, r324;
}
{
add.f16x2 r383, r374, r380;
}
{
sub.f16x2 r386, r368, r383;
}
{
add.f16x2 r389, %27, %39;
}
{
mul.f16x2 r392, r389, r321;
}
{
add.f16x2 r395, %23, r392;
}
{
add.f16x2 r398, %31, %35;
}
{
mul.f16x2 r401, r398, r323;
}
{
add.f16x2 r404, r395, r401;
}
{
sub.f16x2 r407, %28, %40;
}
{
mul.f16x2 r410, r407, r322;
}
{
sub.f16x2 r413, %32, %36;
}
{
mul.f16x2 r416, r413, r324;
}
{
add.f16x2 r419, r410, r416;
}
{
add.f16x2 r422, r404, r419;
}
{
add.f16x2 r425, %27, %39;
}
{
mul.f16x2 r428, r425, r323;
}
{
add.f16x2 r431, %23, r428;
}
{
add.f16x2 r434, %31, %35;
}
{
mul.f16x2 r437, r434, r325;
}
{
add.f16x2 r440, r431, r437;
}
{
sub.f16x2 r443, %28, %40;
}
{
mul.f16x2 r446, r443, r324;
}
{
sub.f16x2 r449, %32, %36;
}
{
mul.f16x2 r452, r449, r327;
}
{
add.f16x2 r455, r446, r452;
}
{
sub.f16x2 r458, r440, r455;
}
{
add.f16x2 r461, %27, %39;
}
{
mul.f16x2 r464, r461, r323;
}
{
add.f16x2 r467, %23, r464;
}
{
add.f16x2 r470, %31, %35;
}
{
mul.f16x2 r473, r470, r325;
}
{
add.f16x2 r476, r467, r473;
}
{
sub.f16x2 r479, %28, %40;
}
{
mul.f16x2 r482, r479, r324;
}
{
sub.f16x2 r485, %32, %36;
}
{
mul.f16x2 r488, r485, r327;
}
{
add.f16x2 r491, r482, r488;
}
{
add.f16x2 r494, r476, r491;
}
{
add.f16x2 r497, %28, %40;
}
{
mul.f16x2 r500, r497, r321;
}
{
add.f16x2 r503, %24, r500;
}
{
add.f16x2 r506, %32, %36;
}
{
mul.f16x2 r509, r506, r323;
}
{
add.f16x2 r512, r503, r509;
}
{
sub.f16x2 r515, %27, %39;
}
{
mul.f16x2 r518, r515, r322;
}
{
sub.f16x2 r521, %31, %35;
}
{
mul.f16x2 r524, r521, r324;
}
{
add.f16x2 r527, r518, r524;
}
{
add.f16x2 r530, r512, r527;
}
{
add.f16x2 r533, %28, %40;
}
{
mul.f16x2 r536, r533, r321;
}
{
add.f16x2 r539, %24, r536;
}
{
add.f16x2 r542, %32, %36;
}
{
mul.f16x2 r545, r542, r323;
}
{
add.f16x2 r548, r539, r545;
}
{
sub.f16x2 r551, %27, %39;
}
{
mul.f16x2 r554, r551, r322;
}
{
sub.f16x2 r557, %31, %35;
}
{
mul.f16x2 r560, r557, r324;
}
{
add.f16x2 r563, r554, r560;
}
{
sub.f16x2 r566, r548, r563;
}
{
add.f16x2 r569, %28, %40;
}
{
mul.f16x2 r572, r569, r323;
}
{
add.f16x2 r575, %24, r572;
}
{
add.f16x2 r578, %32, %36;
}
{
mul.f16x2 r581, r578, r325;
}
{
add.f16x2 r584, r575, r581;
}
{
sub.f16x2 r587, %27, %39;
}
{
mul.f16x2 r590, r587, r324;
}
{
sub.f16x2 r593, %31, %35;
}
{
mul.f16x2 r596, r593, r327;
}
{
add.f16x2 r599, r590, r596;
}
{
add.f16x2 r602, r584, r599;
}
{
add.f16x2 r605, %28, %40;
}
{
mul.f16x2 r608, r605, r323;
}
{
add.f16x2 r611, %24, r608;
}
{
add.f16x2 r614, %32, %36;
}
{
mul.f16x2 r617, r614, r325;
}
{
add.f16x2 r620, r611, r617;
}
{
sub.f16x2 r623, %27, %39;
}
{
mul.f16x2 r626, r623, r324;
}
{
sub.f16x2 r629, %31, %35;
}
{
mul.f16x2 r632, r629, r327;
}
{
add.f16x2 r635, r626, r632;
}
{
sub.f16x2 r638, r620, r635;
}
mov.f32 f190, 0f3F4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r641, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r642, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r643, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r644, {low, high};
}
mov.f32 f198, 0fBE9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r645, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r646, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r647, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r648, {low, high};
}
mov.f32 f161, 0fBF800000;
{
mul.f16x2 r659, r386, r641;
}
{
mul.f16x2 r662, r530, r642;
}
{
sub.f16x2 r665, r659, r662;
}
{
mul.f16x2 r668, r386, r642;
}
{
fma.rn.f16x2 r671, r530, r641, r668;
}
{
mul.f16x2 r675, r458, r643;
}
{
mul.f16x2 r678, r602, r644;
}
{
sub.f16x2 r681, r675, r678;
}
{
mul.f16x2 r684, r458, r644;
}
{
fma.rn.f16x2 r687, r602, r643, r684;
}
{
mul.f16x2 r691, r494, r645;
}
{
mul.f16x2 r694, r638, r646;
}
{
sub.f16x2 r697, r691, r694;
}
{
mul.f16x2 r700, r494, r646;
}
{
fma.rn.f16x2 r703, r638, r645, r700;
}
{
mul.f16x2 r707, r422, r647;
}
{
mul.f16x2 r710, r566, r648;
}
{
sub.f16x2 r713, r707, r710;
}
{
mul.f16x2 r716, r422, r648;
}
{
fma.rn.f16x2 r719, r566, r647, r716;
}
{
add.f16x2 r723, r18, r338;
}
{
add.f16x2 r726, r30, r350;
}
{
sub.f16x2 r729, r18, r338;
}
{
sub.f16x2 r732, r30, r350;
}
{
add.f16x2 r735, r66, r665;
}
{
add.f16x2 r738, r210, r671;
}
{
sub.f16x2 r741, r66, r665;
}
{
sub.f16x2 r744, r210, r671;
}
{
add.f16x2 r747, r138, r681;
}
{
add.f16x2 r750, r282, r687;
}
{
sub.f16x2 r753, r138, r681;
}
{
sub.f16x2 r756, r282, r687;
}
{
add.f16x2 r759, r174, r697;
}
{
add.f16x2 r762, r318, r703;
}
{
sub.f16x2 r765, r174, r697;
}
{
sub.f16x2 r768, r318, r703;
}
{
add.f16x2 r771, r102, r713;
}
{
add.f16x2 r774, r246, r719;
}
{
sub.f16x2 r777, r102, r713;
}
{
sub.f16x2 r780, r246, r719;
}
mul.wide.u32 rd2, r3023, 1374389535;
shr.u64 rd3, rd2, 37;
cvt.u32.u64 r3024, rd3;
mul.lo.s32 r3025, r3024, 100;
sub.s32 r3026, r3023, r3025;
shr.u64 rd4, rd2, 36;
cvt.u32.u64 r3027, rd4;
and.b32 r3028, r3027, 268435454;
mad.lo.s32 r3029, r3028, 4000, r3022;
cvt.rn.f32.u32 f225, r3026;
mul.f32 f226, f225, 0f3BCDE32E;
cos.approx.f32 f61, f226;
sin.approx.f32 f227, f226;
neg.f32 f62, f227;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f61;
cvt.rn.f16.f32 high, f62;
mov.b32 r783, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r786, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r788, {high, high};
}
{
mul.f16x2 r790, r738, r788;
}
{
neg.f16x2 r793, r790;
}
{
fma.rn.f16x2 r795, r735, r786, r793;
}
{
mul.f16x2 r799, r735, r788;
}
{
fma.rn.f16x2 r802, r738, r786, r799;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r806, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r808, {high, high};
}
mov.f32 f162, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r810, {low, high};
}
{
mul.f16x2 r811, r808, r810;
}
{
mul.f16x2 r814, r783, r806;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r817, {high, low};
}
{
fma.rn.f16x2 r819, r811, r817, r814;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r819;
mov.b32 r823, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r819;
mov.b32 r825, {high, high};
}
{
mul.f16x2 r827, r750, r825;
}
{
neg.f16x2 r830, r827;
}
{
fma.rn.f16x2 r832, r747, r823, r830;
}
{
mul.f16x2 r836, r747, r825;
}
{
fma.rn.f16x2 r839, r750, r823, r836;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r843, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r845, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r847, {low, high};
}
{
mul.f16x2 r848, r845, r847;
}
{
mul.f16x2 r851, r819, r843;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r819;
mov.b32 r854, {high, low};
}
{
fma.rn.f16x2 r856, r848, r854, r851;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r856;
mov.b32 r860, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r856;
mov.b32 r862, {high, high};
}
{
mul.f16x2 r864, r762, r862;
}
{
neg.f16x2 r867, r864;
}
{
fma.rn.f16x2 r869, r759, r860, r867;
}
{
mul.f16x2 r873, r759, r862;
}
{
fma.rn.f16x2 r876, r762, r860, r873;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r880, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r882, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r884, {low, high};
}
{
mul.f16x2 r885, r882, r884;
}
{
mul.f16x2 r888, r856, r880;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r856;
mov.b32 r891, {high, low};
}
{
fma.rn.f16x2 r893, r885, r891, r888;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r893;
mov.b32 r897, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r893;
mov.b32 r899, {high, high};
}
{
mul.f16x2 r901, r774, r899;
}
{
neg.f16x2 r904, r901;
}
{
fma.rn.f16x2 r906, r771, r897, r904;
}
{
mul.f16x2 r910, r771, r899;
}
{
fma.rn.f16x2 r913, r774, r897, r910;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r917, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r919, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r921, {low, high};
}
{
mul.f16x2 r922, r919, r921;
}
{
mul.f16x2 r925, r893, r917;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r893;
mov.b32 r928, {high, low};
}
{
fma.rn.f16x2 r930, r922, r928, r925;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r930;
mov.b32 r934, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r930;
mov.b32 r936, {high, high};
}
{
mul.f16x2 r938, r732, r936;
}
{
neg.f16x2 r941, r938;
}
{
fma.rn.f16x2 r943, r729, r934, r941;
}
{
mul.f16x2 r947, r729, r936;
}
{
fma.rn.f16x2 r950, r732, r934, r947;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r954, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r956, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r958, {low, high};
}
{
mul.f16x2 r959, r956, r958;
}
{
mul.f16x2 r962, r930, r954;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r930;
mov.b32 r965, {high, low};
}
{
fma.rn.f16x2 r967, r959, r965, r962;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r967;
mov.b32 r971, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r967;
mov.b32 r973, {high, high};
}
{
mul.f16x2 r975, r744, r973;
}
{
neg.f16x2 r978, r975;
}
{
fma.rn.f16x2 r980, r741, r971, r978;
}
{
mul.f16x2 r984, r741, r973;
}
{
fma.rn.f16x2 r987, r744, r971, r984;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r991, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r993, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r995, {low, high};
}
{
mul.f16x2 r996, r993, r995;
}
{
mul.f16x2 r999, r967, r991;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r967;
mov.b32 r1002, {high, low};
}
{
fma.rn.f16x2 r1004, r996, r1002, r999;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1004;
mov.b32 r1008, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1004;
mov.b32 r1010, {high, high};
}
{
mul.f16x2 r1012, r756, r1010;
}
{
neg.f16x2 r1015, r1012;
}
{
fma.rn.f16x2 r1017, r753, r1008, r1015;
}
{
mul.f16x2 r1021, r753, r1010;
}
{
fma.rn.f16x2 r1024, r756, r1008, r1021;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r1028, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r1030, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r1032, {low, high};
}
{
mul.f16x2 r1033, r1030, r1032;
}
{
mul.f16x2 r1036, r1004, r1028;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1004;
mov.b32 r1039, {high, low};
}
{
fma.rn.f16x2 r1041, r1033, r1039, r1036;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1041;
mov.b32 r1045, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1041;
mov.b32 r1047, {high, high};
}
{
mul.f16x2 r1049, r768, r1047;
}
{
neg.f16x2 r1052, r1049;
}
{
fma.rn.f16x2 r1054, r765, r1045, r1052;
}
{
mul.f16x2 r1058, r765, r1047;
}
{
fma.rn.f16x2 r1061, r768, r1045, r1058;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r1065, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r1067, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r1069, {low, high};
}
{
mul.f16x2 r1070, r1067, r1069;
}
{
mul.f16x2 r1073, r1041, r1065;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1041;
mov.b32 r1076, {high, low};
}
{
fma.rn.f16x2 r1078, r1070, r1076, r1073;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1078;
mov.b32 r1082, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1078;
mov.b32 r1084, {high, high};
}
{
mul.f16x2 r1086, r780, r1084;
}
{
neg.f16x2 r1089, r1086;
}
{
fma.rn.f16x2 r1091, r777, r1082, r1089;
}
{
mul.f16x2 r1095, r777, r1084;
}
{
fma.rn.f16x2 r1098, r780, r1082, r1095;
}
barrier.sync 0;
mad.lo.s32 r3030, r3026, 80, r3029;
st.shared.v2.f32 [r3030], {r723, r726};
st.shared.v2.f32 [r3030+8], {r795, r802};
st.shared.v2.f32 [r3030+16], {r832, r839};
st.shared.v2.f32 [r3030+24], {r869, r876};
st.shared.v2.f32 [r3030+32], {r906, r913};
st.shared.v2.f32 [r3030+40], {r943, r950};
st.shared.v2.f32 [r3030+48], {r980, r987};
st.shared.v2.f32 [r3030+56], {r1017, r1024};
st.shared.v2.f32 [r3030+64], {r1054, r1061};
st.shared.v2.f32 [r3030+72], {r1091, r1098};
barrier.sync 0;
mad.lo.s32 r3031, r3026, -72, r3030;
ld.shared.u32 r1131, [r3031];
ld.shared.u32 r1143, [r3031+4];
ld.shared.u32 r1451, [r3031+800];
ld.shared.u32 r1463, [r3031+804];
ld.shared.u32 r1128, [r3031+1600];
ld.shared.u32 r1140, [r3031+1604];
ld.shared.u32 r1448, [r3031+2400];
ld.shared.u32 r1460, [r3031+2404];
ld.shared.u32 r1134, [r3031+3200];
ld.shared.u32 r1146, [r3031+3204];
ld.shared.u32 r1454, [r3031+4000];
ld.shared.u32 r1466, [r3031+4004];
ld.shared.u32 r1135, [r3031+4800];
ld.shared.u32 r1147, [r3031+4804];
ld.shared.u32 r1455, [r3031+5600];
ld.shared.u32 r1467, [r3031+5604];
ld.shared.u32 r1129, [r3031+6400];
ld.shared.u32 r1141, [r3031+6404];
ld.shared.u32 r1449, [r3031+7200];
ld.shared.u32 r1461, [r3031+7204];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1119, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1120, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1121, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r1122, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1123, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1124, {low, high};
}
{
neg.f16x2 r1125, r1124;
}
{
add.f16x2 r1127, r1128, r1129;
}
{
add.f16x2 r1130, r1131, r1127;
}
{
add.f16x2 r1133, r1134, r1135;
}
{
add.f16x2 r1136, r1130, r1133;
}
{
add.f16x2 r1139, r1140, r1141;
}
{
add.f16x2 r1142, r1143, r1139;
}
{
add.f16x2 r1145, r1146, r1147;
}
{
add.f16x2 r1148, r1142, r1145;
}
{
add.f16x2 r1151, r1128, r1129;
}
{
mul.f16x2 r1154, r1151, r1119;
}
{
add.f16x2 r1157, r1131, r1154;
}
{
add.f16x2 r1160, r1134, r1135;
}
{
mul.f16x2 r1163, r1160, r1121;
}
{
add.f16x2 r1166, r1157, r1163;
}
{
sub.f16x2 r1169, r1140, r1141;
}
{
mul.f16x2 r1172, r1169, r1120;
}
{
sub.f16x2 r1175, r1146, r1147;
}
{
mul.f16x2 r1178, r1175, r1122;
}
{
add.f16x2 r1181, r1172, r1178;
}
{
sub.f16x2 r1184, r1166, r1181;
}
{
add.f16x2 r1187, r1128, r1129;
}
{
mul.f16x2 r1190, r1187, r1119;
}
{
add.f16x2 r1193, r1131, r1190;
}
{
add.f16x2 r1196, r1134, r1135;
}
{
mul.f16x2 r1199, r1196, r1121;
}
{
add.f16x2 r1202, r1193, r1199;
}
{
sub.f16x2 r1205, r1140, r1141;
}
{
mul.f16x2 r1208, r1205, r1120;
}
{
sub.f16x2 r1211, r1146, r1147;
}
{
mul.f16x2 r1214, r1211, r1122;
}
{
add.f16x2 r1217, r1208, r1214;
}
{
add.f16x2 r1220, r1202, r1217;
}
{
add.f16x2 r1223, r1128, r1129;
}
{
mul.f16x2 r1226, r1223, r1121;
}
{
add.f16x2 r1229, r1131, r1226;
}
{
add.f16x2 r1232, r1134, r1135;
}
{
mul.f16x2 r1235, r1232, r1123;
}
{
add.f16x2 r1238, r1229, r1235;
}
{
sub.f16x2 r1241, r1140, r1141;
}
{
mul.f16x2 r1244, r1241, r1122;
}
{
sub.f16x2 r1247, r1146, r1147;
}
{
mul.f16x2 r1250, r1247, r1125;
}
{
add.f16x2 r1253, r1244, r1250;
}
{
sub.f16x2 r1256, r1238, r1253;
}
{
add.f16x2 r1259, r1128, r1129;
}
{
mul.f16x2 r1262, r1259, r1121;
}
{
add.f16x2 r1265, r1131, r1262;
}
{
add.f16x2 r1268, r1134, r1135;
}
{
mul.f16x2 r1271, r1268, r1123;
}
{
add.f16x2 r1274, r1265, r1271;
}
{
sub.f16x2 r1277, r1140, r1141;
}
{
mul.f16x2 r1280, r1277, r1122;
}
{
sub.f16x2 r1283, r1146, r1147;
}
{
mul.f16x2 r1286, r1283, r1125;
}
{
add.f16x2 r1289, r1280, r1286;
}
{
add.f16x2 r1292, r1274, r1289;
}
{
add.f16x2 r1295, r1140, r1141;
}
{
mul.f16x2 r1298, r1295, r1119;
}
{
add.f16x2 r1301, r1143, r1298;
}
{
add.f16x2 r1304, r1146, r1147;
}
{
mul.f16x2 r1307, r1304, r1121;
}
{
add.f16x2 r1310, r1301, r1307;
}
{
sub.f16x2 r1313, r1128, r1129;
}
{
mul.f16x2 r1316, r1313, r1120;
}
{
sub.f16x2 r1319, r1134, r1135;
}
{
mul.f16x2 r1322, r1319, r1122;
}
{
add.f16x2 r1325, r1316, r1322;
}
{
add.f16x2 r1328, r1310, r1325;
}
{
add.f16x2 r1331, r1140, r1141;
}
{
mul.f16x2 r1334, r1331, r1119;
}
{
add.f16x2 r1337, r1143, r1334;
}
{
add.f16x2 r1340, r1146, r1147;
}
{
mul.f16x2 r1343, r1340, r1121;
}
{
add.f16x2 r1346, r1337, r1343;
}
{
sub.f16x2 r1349, r1128, r1129;
}
{
mul.f16x2 r1352, r1349, r1120;
}
{
sub.f16x2 r1355, r1134, r1135;
}
{
mul.f16x2 r1358, r1355, r1122;
}
{
add.f16x2 r1361, r1352, r1358;
}
{
sub.f16x2 r1364, r1346, r1361;
}
{
add.f16x2 r1367, r1140, r1141;
}
{
mul.f16x2 r1370, r1367, r1121;
}
{
add.f16x2 r1373, r1143, r1370;
}
{
add.f16x2 r1376, r1146, r1147;
}
{
mul.f16x2 r1379, r1376, r1123;
}
{
add.f16x2 r1382, r1373, r1379;
}
{
sub.f16x2 r1385, r1128, r1129;
}
{
mul.f16x2 r1388, r1385, r1122;
}
{
sub.f16x2 r1391, r1134, r1135;
}
{
mul.f16x2 r1394, r1391, r1125;
}
{
add.f16x2 r1397, r1388, r1394;
}
{
add.f16x2 r1400, r1382, r1397;
}
{
add.f16x2 r1403, r1140, r1141;
}
{
mul.f16x2 r1406, r1403, r1121;
}
{
add.f16x2 r1409, r1143, r1406;
}
{
add.f16x2 r1412, r1146, r1147;
}
{
mul.f16x2 r1415, r1412, r1123;
}
{
add.f16x2 r1418, r1409, r1415;
}
{
sub.f16x2 r1421, r1128, r1129;
}
{
mul.f16x2 r1424, r1421, r1122;
}
{
sub.f16x2 r1427, r1134, r1135;
}
{
mul.f16x2 r1430, r1427, r1125;
}
{
add.f16x2 r1433, r1424, r1430;
}
{
sub.f16x2 r1436, r1418, r1433;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1439, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1440, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1441, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r1442, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1443, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1444, {low, high};
}
{
neg.f16x2 r1445, r1444;
}
{
add.f16x2 r1447, r1448, r1449;
}
{
add.f16x2 r1450, r1451, r1447;
}
{
add.f16x2 r1453, r1454, r1455;
}
{
add.f16x2 r1456, r1450, r1453;
}
{
add.f16x2 r1459, r1460, r1461;
}
{
add.f16x2 r1462, r1463, r1459;
}
{
add.f16x2 r1465, r1466, r1467;
}
{
add.f16x2 r1468, r1462, r1465;
}
{
add.f16x2 r1471, r1448, r1449;
}
{
mul.f16x2 r1474, r1471, r1439;
}
{
add.f16x2 r1477, r1451, r1474;
}
{
add.f16x2 r1480, r1454, r1455;
}
{
mul.f16x2 r1483, r1480, r1441;
}
{
add.f16x2 r1486, r1477, r1483;
}
{
sub.f16x2 r1489, r1460, r1461;
}
{
mul.f16x2 r1492, r1489, r1440;
}
{
sub.f16x2 r1495, r1466, r1467;
}
{
mul.f16x2 r1498, r1495, r1442;
}
{
add.f16x2 r1501, r1492, r1498;
}
{
sub.f16x2 r1504, r1486, r1501;
}
{
add.f16x2 r1507, r1448, r1449;
}
{
mul.f16x2 r1510, r1507, r1439;
}
{
add.f16x2 r1513, r1451, r1510;
}
{
add.f16x2 r1516, r1454, r1455;
}
{
mul.f16x2 r1519, r1516, r1441;
}
{
add.f16x2 r1522, r1513, r1519;
}
{
sub.f16x2 r1525, r1460, r1461;
}
{
mul.f16x2 r1528, r1525, r1440;
}
{
sub.f16x2 r1531, r1466, r1467;
}
{
mul.f16x2 r1534, r1531, r1442;
}
{
add.f16x2 r1537, r1528, r1534;
}
{
add.f16x2 r1540, r1522, r1537;
}
{
add.f16x2 r1543, r1448, r1449;
}
{
mul.f16x2 r1546, r1543, r1441;
}
{
add.f16x2 r1549, r1451, r1546;
}
{
add.f16x2 r1552, r1454, r1455;
}
{
mul.f16x2 r1555, r1552, r1443;
}
{
add.f16x2 r1558, r1549, r1555;
}
{
sub.f16x2 r1561, r1460, r1461;
}
{
mul.f16x2 r1564, r1561, r1442;
}
{
sub.f16x2 r1567, r1466, r1467;
}
{
mul.f16x2 r1570, r1567, r1445;
}
{
add.f16x2 r1573, r1564, r1570;
}
{
sub.f16x2 r1576, r1558, r1573;
}
{
add.f16x2 r1579, r1448, r1449;
}
{
mul.f16x2 r1582, r1579, r1441;
}
{
add.f16x2 r1585, r1451, r1582;
}
{
add.f16x2 r1588, r1454, r1455;
}
{
mul.f16x2 r1591, r1588, r1443;
}
{
add.f16x2 r1594, r1585, r1591;
}
{
sub.f16x2 r1597, r1460, r1461;
}
{
mul.f16x2 r1600, r1597, r1442;
}
{
sub.f16x2 r1603, r1466, r1467;
}
{
mul.f16x2 r1606, r1603, r1445;
}
{
add.f16x2 r1609, r1600, r1606;
}
{
add.f16x2 r1612, r1594, r1609;
}
{
add.f16x2 r1615, r1460, r1461;
}
{
mul.f16x2 r1618, r1615, r1439;
}
{
add.f16x2 r1621, r1463, r1618;
}
{
add.f16x2 r1624, r1466, r1467;
}
{
mul.f16x2 r1627, r1624, r1441;
}
{
add.f16x2 r1630, r1621, r1627;
}
{
sub.f16x2 r1633, r1448, r1449;
}
{
mul.f16x2 r1636, r1633, r1440;
}
{
sub.f16x2 r1639, r1454, r1455;
}
{
mul.f16x2 r1642, r1639, r1442;
}
{
add.f16x2 r1645, r1636, r1642;
}
{
add.f16x2 r1648, r1630, r1645;
}
{
add.f16x2 r1651, r1460, r1461;
}
{
mul.f16x2 r1654, r1651, r1439;
}
{
add.f16x2 r1657, r1463, r1654;
}
{
add.f16x2 r1660, r1466, r1467;
}
{
mul.f16x2 r1663, r1660, r1441;
}
{
add.f16x2 r1666, r1657, r1663;
}
{
sub.f16x2 r1669, r1448, r1449;
}
{
mul.f16x2 r1672, r1669, r1440;
}
{
sub.f16x2 r1675, r1454, r1455;
}
{
mul.f16x2 r1678, r1675, r1442;
}
{
add.f16x2 r1681, r1672, r1678;
}
{
sub.f16x2 r1684, r1666, r1681;
}
{
add.f16x2 r1687, r1460, r1461;
}
{
mul.f16x2 r1690, r1687, r1441;
}
{
add.f16x2 r1693, r1463, r1690;
}
{
add.f16x2 r1696, r1466, r1467;
}
{
mul.f16x2 r1699, r1696, r1443;
}
{
add.f16x2 r1702, r1693, r1699;
}
{
sub.f16x2 r1705, r1448, r1449;
}
{
mul.f16x2 r1708, r1705, r1442;
}
{
sub.f16x2 r1711, r1454, r1455;
}
{
mul.f16x2 r1714, r1711, r1445;
}
{
add.f16x2 r1717, r1708, r1714;
}
{
add.f16x2 r1720, r1702, r1717;
}
{
add.f16x2 r1723, r1460, r1461;
}
{
mul.f16x2 r1726, r1723, r1441;
}
{
add.f16x2 r1729, r1463, r1726;
}
{
add.f16x2 r1732, r1466, r1467;
}
{
mul.f16x2 r1735, r1732, r1443;
}
{
add.f16x2 r1738, r1729, r1735;
}
{
sub.f16x2 r1741, r1448, r1449;
}
{
mul.f16x2 r1744, r1741, r1442;
}
{
sub.f16x2 r1747, r1454, r1455;
}
{
mul.f16x2 r1750, r1747, r1445;
}
{
add.f16x2 r1753, r1744, r1750;
}
{
sub.f16x2 r1756, r1738, r1753;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r1759, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r1760, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1761, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1762, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1763, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1764, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1765, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r1766, {low, high};
}
{
mul.f16x2 r1777, r1504, r1759;
}
{
mul.f16x2 r1780, r1648, r1760;
}
{
sub.f16x2 r1783, r1777, r1780;
}
{
mul.f16x2 r1786, r1504, r1760;
}
{
fma.rn.f16x2 r1789, r1648, r1759, r1786;
}
{
mul.f16x2 r1793, r1576, r1761;
}
{
mul.f16x2 r1796, r1720, r1762;
}
{
sub.f16x2 r1799, r1793, r1796;
}
{
mul.f16x2 r1802, r1576, r1762;
}
{
fma.rn.f16x2 r1805, r1720, r1761, r1802;
}
{
mul.f16x2 r1809, r1612, r1763;
}
{
mul.f16x2 r1812, r1756, r1764;
}
{
sub.f16x2 r1815, r1809, r1812;
}
{
mul.f16x2 r1818, r1612, r1764;
}
{
fma.rn.f16x2 r1821, r1756, r1763, r1818;
}
{
mul.f16x2 r1825, r1540, r1765;
}
{
mul.f16x2 r1828, r1684, r1766;
}
{
sub.f16x2 r1831, r1825, r1828;
}
{
mul.f16x2 r1834, r1540, r1766;
}
{
fma.rn.f16x2 r1837, r1684, r1765, r1834;
}
{
add.f16x2 r1841, r1136, r1456;
}
{
add.f16x2 r1844, r1148, r1468;
}
{
sub.f16x2 r1847, r1136, r1456;
}
{
sub.f16x2 r1850, r1148, r1468;
}
{
add.f16x2 r1853, r1184, r1783;
}
{
add.f16x2 r1856, r1328, r1789;
}
{
sub.f16x2 r1859, r1184, r1783;
}
{
sub.f16x2 r1862, r1328, r1789;
}
{
add.f16x2 r1865, r1256, r1799;
}
{
add.f16x2 r1868, r1400, r1805;
}
{
sub.f16x2 r1871, r1256, r1799;
}
{
sub.f16x2 r1874, r1400, r1805;
}
{
add.f16x2 r1877, r1292, r1815;
}
{
add.f16x2 r1880, r1436, r1821;
}
{
sub.f16x2 r1883, r1292, r1815;
}
{
sub.f16x2 r1886, r1436, r1821;
}
{
add.f16x2 r1889, r1220, r1831;
}
{
add.f16x2 r1892, r1364, r1837;
}
{
sub.f16x2 r1895, r1220, r1831;
}
{
sub.f16x2 r1898, r1364, r1837;
}
mul.wide.u32 rd5, r3026, -858993459;
shr.u64 rd6, rd5, 35;
cvt.u32.u64 r3032, rd6;
cvt.rn.f32.u32 f228, r3032;
mul.f32 f229, f228, 0f3D80ADFD;
cos.approx.f32 f143, f229;
sin.approx.f32 f230, f229;
neg.f32 f144, f230;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f143;
cvt.rn.f16.f32 high, f144;
mov.b32 r1901, {low, high};
}
mul.lo.s32 r3033, r3032, 10;
sub.s32 r3034, r3026, r3033;
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1904, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1906, {high, high};
}
{
mul.f16x2 r1908, r1856, r1906;
}
{
neg.f16x2 r1911, r1908;
}
{
fma.rn.f16x2 r1913, r1853, r1904, r1911;
}
{
mul.f16x2 r1917, r1853, r1906;
}
{
fma.rn.f16x2 r1920, r1856, r1904, r1917;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1924, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1926, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r1928, {low, high};
}
{
mul.f16x2 r1929, r1926, r1928;
}
{
mul.f16x2 r1932, r1901, r1924;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1935, {high, low};
}
{
fma.rn.f16x2 r1937, r1929, r1935, r1932;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1937;
mov.b32 r1941, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1937;
mov.b32 r1943, {high, high};
}
{
mul.f16x2 r1945, r1868, r1943;
}
{
neg.f16x2 r1948, r1945;
}
{
fma.rn.f16x2 r1950, r1865, r1941, r1948;
}
{
mul.f16x2 r1954, r1865, r1943;
}
{
fma.rn.f16x2 r1957, r1868, r1941, r1954;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1961, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1963, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r1965, {low, high};
}
{
mul.f16x2 r1966, r1963, r1965;
}
{
mul.f16x2 r1969, r1937, r1961;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1937;
mov.b32 r1972, {high, low};
}
{
fma.rn.f16x2 r1974, r1966, r1972, r1969;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1974;
mov.b32 r1978, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1974;
mov.b32 r1980, {high, high};
}
{
mul.f16x2 r1982, r1880, r1980;
}
{
neg.f16x2 r1985, r1982;
}
{
fma.rn.f16x2 r1987, r1877, r1978, r1985;
}
{
mul.f16x2 r1991, r1877, r1980;
}
{
fma.rn.f16x2 r1994, r1880, r1978, r1991;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1998, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2000, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2002, {low, high};
}
{
mul.f16x2 r2003, r2000, r2002;
}
{
mul.f16x2 r2006, r1974, r1998;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1974;
mov.b32 r2009, {high, low};
}
{
fma.rn.f16x2 r2011, r2003, r2009, r2006;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2011;
mov.b32 r2015, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2011;
mov.b32 r2017, {high, high};
}
{
mul.f16x2 r2019, r1892, r2017;
}
{
neg.f16x2 r2022, r2019;
}
{
fma.rn.f16x2 r2024, r1889, r2015, r2022;
}
{
mul.f16x2 r2028, r1889, r2017;
}
{
fma.rn.f16x2 r2031, r1892, r2015, r2028;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2035, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2037, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2039, {low, high};
}
{
mul.f16x2 r2040, r2037, r2039;
}
{
mul.f16x2 r2043, r2011, r2035;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2011;
mov.b32 r2046, {high, low};
}
{
fma.rn.f16x2 r2048, r2040, r2046, r2043;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2048;
mov.b32 r2052, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2048;
mov.b32 r2054, {high, high};
}
{
mul.f16x2 r2056, r1850, r2054;
}
{
neg.f16x2 r2059, r2056;
}
{
fma.rn.f16x2 r2061, r1847, r2052, r2059;
}
{
mul.f16x2 r2065, r1847, r2054;
}
{
fma.rn.f16x2 r2068, r1850, r2052, r2065;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2072, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2074, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2076, {low, high};
}
{
mul.f16x2 r2077, r2074, r2076;
}
{
mul.f16x2 r2080, r2048, r2072;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2048;
mov.b32 r2083, {high, low};
}
{
fma.rn.f16x2 r2085, r2077, r2083, r2080;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2085;
mov.b32 r2089, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2085;
mov.b32 r2091, {high, high};
}
{
mul.f16x2 r2093, r1862, r2091;
}
{
neg.f16x2 r2096, r2093;
}
{
fma.rn.f16x2 r2098, r1859, r2089, r2096;
}
{
mul.f16x2 r2102, r1859, r2091;
}
{
fma.rn.f16x2 r2105, r1862, r2089, r2102;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2109, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2111, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2113, {low, high};
}
{
mul.f16x2 r2114, r2111, r2113;
}
{
mul.f16x2 r2117, r2085, r2109;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2085;
mov.b32 r2120, {high, low};
}
{
fma.rn.f16x2 r2122, r2114, r2120, r2117;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2122;
mov.b32 r2126, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2122;
mov.b32 r2128, {high, high};
}
{
mul.f16x2 r2130, r1874, r2128;
}
{
neg.f16x2 r2133, r2130;
}
{
fma.rn.f16x2 r2135, r1871, r2126, r2133;
}
{
mul.f16x2 r2139, r1871, r2128;
}
{
fma.rn.f16x2 r2142, r1874, r2126, r2139;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2146, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2148, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2150, {low, high};
}
{
mul.f16x2 r2151, r2148, r2150;
}
{
mul.f16x2 r2154, r2122, r2146;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2122;
mov.b32 r2157, {high, low};
}
{
fma.rn.f16x2 r2159, r2151, r2157, r2154;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2159;
mov.b32 r2163, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2159;
mov.b32 r2165, {high, high};
}
{
mul.f16x2 r2167, r1886, r2165;
}
{
neg.f16x2 r2170, r2167;
}
{
fma.rn.f16x2 r2172, r1883, r2163, r2170;
}
{
mul.f16x2 r2176, r1883, r2165;
}
{
fma.rn.f16x2 r2179, r1886, r2163, r2176;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2183, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2185, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2187, {low, high};
}
{
mul.f16x2 r2188, r2185, r2187;
}
{
mul.f16x2 r2191, r2159, r2183;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2159;
mov.b32 r2194, {high, low};
}
{
fma.rn.f16x2 r2196, r2188, r2194, r2191;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2196;
mov.b32 r2200, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2196;
mov.b32 r2202, {high, high};
}
{
mul.f16x2 r2204, r1898, r2202;
}
{
neg.f16x2 r2207, r2204;
}
{
fma.rn.f16x2 r2209, r1895, r2200, r2207;
}
{
mul.f16x2 r2213, r1895, r2202;
}
{
fma.rn.f16x2 r2216, r1898, r2200, r2213;
}
shl.b32 r3035, r3034, 3;
add.s32 r3036, r3029, r3035;
barrier.sync 0;
mad.lo.s32 r3037, r3032, 800, r3036;
st.shared.u32 [r3037], r1841;
st.shared.u32 [r3037+4], r1844;
st.shared.u32 [r3037+80], r1913;
st.shared.u32 [r3037+84], r1920;
st.shared.u32 [r3037+160], r1950;
st.shared.u32 [r3037+164], r1957;
st.shared.u32 [r3037+240], r1987;
st.shared.u32 [r3037+244], r1994;
st.shared.u32 [r3037+320], r2024;
st.shared.u32 [r3037+324], r2031;
st.shared.u32 [r3037+400], r2061;
st.shared.u32 [r3037+404], r2068;
st.shared.u32 [r3037+480], r2098;
st.shared.u32 [r3037+484], r2105;
st.shared.u32 [r3037+560], r2135;
st.shared.u32 [r3037+564], r2142;
st.shared.u32 [r3037+640], r2172;
st.shared.u32 [r3037+644], r2179;
st.shared.u32 [r3037+720], r2209;
st.shared.u32 [r3037+724], r2216;
barrier.sync 0;
ld.shared.u32 r2249, [r3031];
ld.shared.u32 r2261, [r3031+4];
ld.shared.u32 r2569, [r3031+800];
ld.shared.u32 r2581, [r3031+804];
ld.shared.u32 r2246, [r3031+1600];
ld.shared.u32 r2258, [r3031+1604];
ld.shared.u32 r2566, [r3031+2400];
ld.shared.u32 r2578, [r3031+2404];
ld.shared.u32 r2252, [r3031+3200];
ld.shared.u32 r2264, [r3031+3204];
ld.shared.u32 r2572, [r3031+4000];
ld.shared.u32 r2584, [r3031+4004];
ld.shared.u32 r2253, [r3031+4800];
ld.shared.u32 r2265, [r3031+4804];
ld.shared.u32 r2573, [r3031+5600];
ld.shared.u32 r2585, [r3031+5604];
ld.shared.u32 r2247, [r3031+6400];
ld.shared.u32 r2259, [r3031+6404];
ld.shared.u32 r2567, [r3031+7200];
ld.shared.u32 r2579, [r3031+7204];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r2237, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2238, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r2239, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r2240, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r2241, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2242, {low, high};
}
{
neg.f16x2 r2243, r2242;
}
{
add.f16x2 r2245, r2246, r2247;
}
{
add.f16x2 r2248, r2249, r2245;
}
{
add.f16x2 r2251, r2252, r2253;
}
{
add.f16x2 r2254, r2248, r2251;
}
{
add.f16x2 r2257, r2258, r2259;
}
{
add.f16x2 r2260, r2261, r2257;
}
{
add.f16x2 r2263, r2264, r2265;
}
{
add.f16x2 r2266, r2260, r2263;
}
{
add.f16x2 r2269, r2246, r2247;
}
{
mul.f16x2 r2272, r2269, r2237;
}
{
add.f16x2 r2275, r2249, r2272;
}
{
add.f16x2 r2278, r2252, r2253;
}
{
mul.f16x2 r2281, r2278, r2239;
}
{
add.f16x2 r2284, r2275, r2281;
}
{
sub.f16x2 r2287, r2258, r2259;
}
{
mul.f16x2 r2290, r2287, r2238;
}
{
sub.f16x2 r2293, r2264, r2265;
}
{
mul.f16x2 r2296, r2293, r2240;
}
{
add.f16x2 r2299, r2290, r2296;
}
{
sub.f16x2 r2302, r2284, r2299;
}
{
add.f16x2 r2305, r2246, r2247;
}
{
mul.f16x2 r2308, r2305, r2237;
}
{
add.f16x2 r2311, r2249, r2308;
}
{
add.f16x2 r2314, r2252, r2253;
}
{
mul.f16x2 r2317, r2314, r2239;
}
{
add.f16x2 r2320, r2311, r2317;
}
{
sub.f16x2 r2323, r2258, r2259;
}
{
mul.f16x2 r2326, r2323, r2238;
}
{
sub.f16x2 r2329, r2264, r2265;
}
{
mul.f16x2 r2332, r2329, r2240;
}
{
add.f16x2 r2335, r2326, r2332;
}
{
add.f16x2 r2338, r2320, r2335;
}
{
add.f16x2 r2341, r2246, r2247;
}
{
mul.f16x2 r2344, r2341, r2239;
}
{
add.f16x2 r2347, r2249, r2344;
}
{
add.f16x2 r2350, r2252, r2253;
}
{
mul.f16x2 r2353, r2350, r2241;
}
{
add.f16x2 r2356, r2347, r2353;
}
{
sub.f16x2 r2359, r2258, r2259;
}
{
mul.f16x2 r2362, r2359, r2240;
}
{
sub.f16x2 r2365, r2264, r2265;
}
{
mul.f16x2 r2368, r2365, r2243;
}
{
add.f16x2 r2371, r2362, r2368;
}
{
sub.f16x2 r2374, r2356, r2371;
}
{
add.f16x2 r2377, r2246, r2247;
}
{
mul.f16x2 r2380, r2377, r2239;
}
{
add.f16x2 r2383, r2249, r2380;
}
{
add.f16x2 r2386, r2252, r2253;
}
{
mul.f16x2 r2389, r2386, r2241;
}
{
add.f16x2 r2392, r2383, r2389;
}
{
sub.f16x2 r2395, r2258, r2259;
}
{
mul.f16x2 r2398, r2395, r2240;
}
{
sub.f16x2 r2401, r2264, r2265;
}
{
mul.f16x2 r2404, r2401, r2243;
}
{
add.f16x2 r2407, r2398, r2404;
}
{
add.f16x2 r2410, r2392, r2407;
}
{
add.f16x2 r2413, r2258, r2259;
}
{
mul.f16x2 r2416, r2413, r2237;
}
{
add.f16x2 r2419, r2261, r2416;
}
{
add.f16x2 r2422, r2264, r2265;
}
{
mul.f16x2 r2425, r2422, r2239;
}
{
add.f16x2 r2428, r2419, r2425;
}
{
sub.f16x2 r2431, r2246, r2247;
}
{
mul.f16x2 r2434, r2431, r2238;
}
{
sub.f16x2 r2437, r2252, r2253;
}
{
mul.f16x2 r2440, r2437, r2240;
}
{
add.f16x2 r2443, r2434, r2440;
}
{
add.f16x2 r2446, r2428, r2443;
}
{
add.f16x2 r2449, r2258, r2259;
}
{
mul.f16x2 r2452, r2449, r2237;
}
{
add.f16x2 r2455, r2261, r2452;
}
{
add.f16x2 r2458, r2264, r2265;
}
{
mul.f16x2 r2461, r2458, r2239;
}
{
add.f16x2 r2464, r2455, r2461;
}
{
sub.f16x2 r2467, r2246, r2247;
}
{
mul.f16x2 r2470, r2467, r2238;
}
{
sub.f16x2 r2473, r2252, r2253;
}
{
mul.f16x2 r2476, r2473, r2240;
}
{
add.f16x2 r2479, r2470, r2476;
}
{
sub.f16x2 r2482, r2464, r2479;
}
{
add.f16x2 r2485, r2258, r2259;
}
{
mul.f16x2 r2488, r2485, r2239;
}
{
add.f16x2 r2491, r2261, r2488;
}
{
add.f16x2 r2494, r2264, r2265;
}
{
mul.f16x2 r2497, r2494, r2241;
}
{
add.f16x2 r2500, r2491, r2497;
}
{
sub.f16x2 r2503, r2246, r2247;
}
{
mul.f16x2 r2506, r2503, r2240;
}
{
sub.f16x2 r2509, r2252, r2253;
}
{
mul.f16x2 r2512, r2509, r2243;
}
{
add.f16x2 r2515, r2506, r2512;
}
{
add.f16x2 r2518, r2500, r2515;
}
{
add.f16x2 r2521, r2258, r2259;
}
{
mul.f16x2 r2524, r2521, r2239;
}
{
add.f16x2 r2527, r2261, r2524;
}
{
add.f16x2 r2530, r2264, r2265;
}
{
mul.f16x2 r2533, r2530, r2241;
}
{
add.f16x2 r2536, r2527, r2533;
}
{
sub.f16x2 r2539, r2246, r2247;
}
{
mul.f16x2 r2542, r2539, r2240;
}
{
sub.f16x2 r2545, r2252, r2253;
}
{
mul.f16x2 r2548, r2545, r2243;
}
{
add.f16x2 r2551, r2542, r2548;
}
{
sub.f16x2 r2554, r2536, r2551;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r2557, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2558, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r2559, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r2560, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r2561, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2562, {low, high};
}
{
neg.f16x2 r2563, r2562;
}
{
add.f16x2 r2565, r2566, r2567;
}
{
add.f16x2 r2568, r2569, r2565;
}
{
add.f16x2 r2571, r2572, r2573;
}
{
add.f16x2 r2574, r2568, r2571;
}
{
add.f16x2 r2577, r2578, r2579;
}
{
add.f16x2 r2580, r2581, r2577;
}
{
add.f16x2 r2583, r2584, r2585;
}
{
add.f16x2 r2586, r2580, r2583;
}
{
add.f16x2 r2589, r2566, r2567;
}
{
mul.f16x2 r2592, r2589, r2557;
}
{
add.f16x2 r2595, r2569, r2592;
}
{
add.f16x2 r2598, r2572, r2573;
}
{
mul.f16x2 r2601, r2598, r2559;
}
{
add.f16x2 r2604, r2595, r2601;
}
{
sub.f16x2 r2607, r2578, r2579;
}
{
mul.f16x2 r2610, r2607, r2558;
}
{
sub.f16x2 r2613, r2584, r2585;
}
{
mul.f16x2 r2616, r2613, r2560;
}
{
add.f16x2 r2619, r2610, r2616;
}
{
sub.f16x2 r2622, r2604, r2619;
}
{
add.f16x2 r2625, r2566, r2567;
}
{
mul.f16x2 r2628, r2625, r2557;
}
{
add.f16x2 r2631, r2569, r2628;
}
{
add.f16x2 r2634, r2572, r2573;
}
{
mul.f16x2 r2637, r2634, r2559;
}
{
add.f16x2 r2640, r2631, r2637;
}
{
sub.f16x2 r2643, r2578, r2579;
}
{
mul.f16x2 r2646, r2643, r2558;
}
{
sub.f16x2 r2649, r2584, r2585;
}
{
mul.f16x2 r2652, r2649, r2560;
}
{
add.f16x2 r2655, r2646, r2652;
}
{
add.f16x2 r2658, r2640, r2655;
}
{
add.f16x2 r2661, r2566, r2567;
}
{
mul.f16x2 r2664, r2661, r2559;
}
{
add.f16x2 r2667, r2569, r2664;
}
{
add.f16x2 r2670, r2572, r2573;
}
{
mul.f16x2 r2673, r2670, r2561;
}
{
add.f16x2 r2676, r2667, r2673;
}
{
sub.f16x2 r2679, r2578, r2579;
}
{
mul.f16x2 r2682, r2679, r2560;
}
{
sub.f16x2 r2685, r2584, r2585;
}
{
mul.f16x2 r2688, r2685, r2563;
}
{
add.f16x2 r2691, r2682, r2688;
}
{
sub.f16x2 r2694, r2676, r2691;
}
{
add.f16x2 r2697, r2566, r2567;
}
{
mul.f16x2 r2700, r2697, r2559;
}
{
add.f16x2 r2703, r2569, r2700;
}
{
add.f16x2 r2706, r2572, r2573;
}
{
mul.f16x2 r2709, r2706, r2561;
}
{
add.f16x2 r2712, r2703, r2709;
}
{
sub.f16x2 r2715, r2578, r2579;
}
{
mul.f16x2 r2718, r2715, r2560;
}
{
sub.f16x2 r2721, r2584, r2585;
}
{
mul.f16x2 r2724, r2721, r2563;
}
{
add.f16x2 r2727, r2718, r2724;
}
{
add.f16x2 r2730, r2712, r2727;
}
{
add.f16x2 r2733, r2578, r2579;
}
{
mul.f16x2 r2736, r2733, r2557;
}
{
add.f16x2 r2739, r2581, r2736;
}
{
add.f16x2 r2742, r2584, r2585;
}
{
mul.f16x2 r2745, r2742, r2559;
}
{
add.f16x2 r2748, r2739, r2745;
}
{
sub.f16x2 r2751, r2566, r2567;
}
{
mul.f16x2 r2754, r2751, r2558;
}
{
sub.f16x2 r2757, r2572, r2573;
}
{
mul.f16x2 r2760, r2757, r2560;
}
{
add.f16x2 r2763, r2754, r2760;
}
{
add.f16x2 r2766, r2748, r2763;
}
{
add.f16x2 r2769, r2578, r2579;
}
{
mul.f16x2 r2772, r2769, r2557;
}
{
add.f16x2 r2775, r2581, r2772;
}
{
add.f16x2 r2778, r2584, r2585;
}
{
mul.f16x2 r2781, r2778, r2559;
}
{
add.f16x2 r2784, r2775, r2781;
}
{
sub.f16x2 r2787, r2566, r2567;
}
{
mul.f16x2 r2790, r2787, r2558;
}
{
sub.f16x2 r2793, r2572, r2573;
}
{
mul.f16x2 r2796, r2793, r2560;
}
{
add.f16x2 r2799, r2790, r2796;
}
{
sub.f16x2 r2802, r2784, r2799;
}
{
add.f16x2 r2805, r2578, r2579;
}
{
mul.f16x2 r2808, r2805, r2559;
}
{
add.f16x2 r2811, r2581, r2808;
}
{
add.f16x2 r2814, r2584, r2585;
}
{
mul.f16x2 r2817, r2814, r2561;
}
{
add.f16x2 r2820, r2811, r2817;
}
{
sub.f16x2 r2823, r2566, r2567;
}
{
mul.f16x2 r2826, r2823, r2560;
}
{
sub.f16x2 r2829, r2572, r2573;
}
{
mul.f16x2 r2832, r2829, r2563;
}
{
add.f16x2 r2835, r2826, r2832;
}
{
add.f16x2 r2838, r2820, r2835;
}
{
add.f16x2 r2841, r2578, r2579;
}
{
mul.f16x2 r2844, r2841, r2559;
}
{
add.f16x2 r2847, r2581, r2844;
}
{
add.f16x2 r2850, r2584, r2585;
}
{
mul.f16x2 r2853, r2850, r2561;
}
{
add.f16x2 r2856, r2847, r2853;
}
{
sub.f16x2 r2859, r2566, r2567;
}
{
mul.f16x2 r2862, r2859, r2560;
}
{
sub.f16x2 r2865, r2572, r2573;
}
{
mul.f16x2 r2868, r2865, r2563;
}
{
add.f16x2 r2871, r2862, r2868;
}
{
sub.f16x2 r2874, r2856, r2871;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r2877, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r2878, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r2879, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2880, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r2881, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2882, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r2883, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r2884, {low, high};
}
{
mul.f16x2 r2895, r2622, r2877;
}
{
mul.f16x2 r2898, r2766, r2878;
}
{
sub.f16x2 r2901, r2895, r2898;
}
{
mul.f16x2 r2904, r2622, r2878;
}
{
fma.rn.f16x2 r2907, r2766, r2877, r2904;
}
{
mul.f16x2 r2911, r2694, r2879;
}
{
mul.f16x2 r2914, r2838, r2880;
}
{
sub.f16x2 r2917, r2911, r2914;
}
{
mul.f16x2 r2920, r2694, r2880;
}
{
fma.rn.f16x2 r2923, r2838, r2879, r2920;
}
{
mul.f16x2 r2927, r2730, r2881;
}
{
mul.f16x2 r2930, r2874, r2882;
}
{
sub.f16x2 r2933, r2927, r2930;
}
{
mul.f16x2 r2936, r2730, r2882;
}
{
fma.rn.f16x2 r2939, r2874, r2881, r2936;
}
{
mul.f16x2 r2943, r2658, r2883;
}
{
mul.f16x2 r2946, r2802, r2884;
}
{
sub.f16x2 r2949, r2943, r2946;
}
{
mul.f16x2 r2952, r2658, r2884;
}
{
fma.rn.f16x2 r2955, r2802, r2883, r2952;
}
{
add.f16x2 %0, r2254, r2574;
}
{
add.f16x2 %1, r2266, r2586;
}
{
sub.f16x2 %10, r2254, r2574;
}
{
sub.f16x2 %11, r2266, r2586;
}
{
add.f16x2 %2, r2302, r2901;
}
{
add.f16x2 %3, r2446, r2907;
}
{
sub.f16x2 %12, r2302, r2901;
}
{
sub.f16x2 %13, r2446, r2907;
}
{
add.f16x2 %4, r2374, r2917;
}
{
add.f16x2 %5, r2518, r2923;
}
{
sub.f16x2 %14, r2374, r2917;
}
{
sub.f16x2 %15, r2518, r2923;
}
{
add.f16x2 %6, r2410, r2933;
}
{
add.f16x2 %7, r2554, r2939;
}
{
sub.f16x2 %16, r2410, r2933;
}
{
sub.f16x2 %17, r2554, r2939;
}
{
add.f16x2 %8, r2338, r2949;
}
{
add.f16x2 %9, r2482, r2955;
}
{
sub.f16x2 %18, r2338, r2949;
}
{
sub.f16x2 %19, r2482, r2955;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<940, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<231>;
.reg .b32 r<3035>;
.reg .b64 rd<6>;
mov.u32 r3019, %tid.y;
mov.u32 r3020, %20;
mad.lo.s32 r3021, r3019, 4000, r3020;
mov.u32 r3022, %tid.x;
mov.f32 f194, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1, {low, high};
}
mov.f32 f200, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2, {low, high};
}
mov.f32 f202, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r3, {low, high};
}
mov.f32 f204, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r4, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r5, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
{
add.f16x2 r9, %25, %37;
}
{
add.f16x2 r12, %21, r9;
}
{
add.f16x2 r15, %29, %33;
}
{
add.f16x2 r18, r12, r15;
}
{
add.f16x2 r21, %26, %38;
}
{
add.f16x2 r24, %22, r21;
}
{
add.f16x2 r27, %30, %34;
}
{
add.f16x2 r30, r24, r27;
}
{
add.f16x2 r33, %25, %37;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %21, r36;
}
{
add.f16x2 r42, %29, %33;
}
{
mul.f16x2 r45, r42, r3;
}
{
add.f16x2 r48, r39, r45;
}
{
sub.f16x2 r51, %26, %38;
}
{
mul.f16x2 r54, r51, r2;
}
{
sub.f16x2 r57, %30, %34;
}
{
mul.f16x2 r60, r57, r4;
}
{
add.f16x2 r63, r54, r60;
}
{
sub.f16x2 r66, r48, r63;
}
{
add.f16x2 r69, %25, %37;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %21, r72;
}
{
add.f16x2 r78, %29, %33;
}
{
mul.f16x2 r81, r78, r3;
}
{
add.f16x2 r84, r75, r81;
}
{
sub.f16x2 r87, %26, %38;
}
{
mul.f16x2 r90, r87, r2;
}
{
sub.f16x2 r93, %30, %34;
}
{
mul.f16x2 r96, r93, r4;
}
{
add.f16x2 r99, r90, r96;
}
{
add.f16x2 r102, r84, r99;
}
{
add.f16x2 r105, %25, %37;
}
{
mul.f16x2 r108, r105, r3;
}
{
add.f16x2 r111, %21, r108;
}
{
add.f16x2 r114, %29, %33;
}
{
mul.f16x2 r117, r114, r5;
}
{
add.f16x2 r120, r111, r117;
}
{
sub.f16x2 r123, %26, %38;
}
{
mul.f16x2 r126, r123, r4;
}
{
sub.f16x2 r129, %30, %34;
}
{
mul.f16x2 r132, r129, r7;
}
{
add.f16x2 r135, r126, r132;
}
{
sub.f16x2 r138, r120, r135;
}
{
add.f16x2 r141, %25, %37;
}
{
mul.f16x2 r144, r141, r3;
}
{
add.f16x2 r147, %21, r144;
}
{
add.f16x2 r150, %29, %33;
}
{
mul.f16x2 r153, r150, r5;
}
{
add.f16x2 r156, r147, r153;
}
{
sub.f16x2 r159, %26, %38;
}
{
mul.f16x2 r162, r159, r4;
}
{
sub.f16x2 r165, %30, %34;
}
{
mul.f16x2 r168, r165, r7;
}
{
add.f16x2 r171, r162, r168;
}
{
add.f16x2 r174, r156, r171;
}
{
add.f16x2 r177, %26, %38;
}
{
mul.f16x2 r180, r177, r1;
}
{
add.f16x2 r183, %22, r180;
}
{
add.f16x2 r186, %30, %34;
}
{
mul.f16x2 r189, r186, r3;
}
{
add.f16x2 r192, r183, r189;
}
{
sub.f16x2 r195, %25, %37;
}
{
mul.f16x2 r198, r195, r2;
}
{
sub.f16x2 r201, %29, %33;
}
{
mul.f16x2 r204, r201, r4;
}
{
add.f16x2 r207, r198, r204;
}
{
add.f16x2 r210, r192, r207;
}
{
add.f16x2 r213, %26, %38;
}
{
mul.f16x2 r216, r213, r1;
}
{
add.f16x2 r219, %22, r216;
}
{
add.f16x2 r222, %30, %34;
}
{
mul.f16x2 r225, r222, r3;
}
{
add.f16x2 r228, r219, r225;
}
{
sub.f16x2 r231, %25, %37;
}
{
mul.f16x2 r234, r231, r2;
}
{
sub.f16x2 r237, %29, %33;
}
{
mul.f16x2 r240, r237, r4;
}
{
add.f16x2 r243, r234, r240;
}
{
sub.f16x2 r246, r228, r243;
}
{
add.f16x2 r249, %26, %38;
}
{
mul.f16x2 r252, r249, r3;
}
{
add.f16x2 r255, %22, r252;
}
{
add.f16x2 r258, %30, %34;
}
{
mul.f16x2 r261, r258, r5;
}
{
add.f16x2 r264, r255, r261;
}
{
sub.f16x2 r267, %25, %37;
}
{
mul.f16x2 r270, r267, r4;
}
{
sub.f16x2 r273, %29, %33;
}
{
mul.f16x2 r276, r273, r7;
}
{
add.f16x2 r279, r270, r276;
}
{
add.f16x2 r282, r264, r279;
}
{
add.f16x2 r285, %26, %38;
}
{
mul.f16x2 r288, r285, r3;
}
{
add.f16x2 r291, %22, r288;
}
{
add.f16x2 r294, %30, %34;
}
{
mul.f16x2 r297, r294, r5;
}
{
add.f16x2 r300, r291, r297;
}
{
sub.f16x2 r303, %25, %37;
}
{
mul.f16x2 r306, r303, r4;
}
{
sub.f16x2 r309, %29, %33;
}
{
mul.f16x2 r312, r309, r7;
}
{
add.f16x2 r315, r306, r312;
}
{
sub.f16x2 r318, r300, r315;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r321, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r322, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r323, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r324, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r325, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r326, {low, high};
}
{
neg.f16x2 r327, r326;
}
{
add.f16x2 r329, %27, %39;
}
{
add.f16x2 r332, %23, r329;
}
{
add.f16x2 r335, %31, %35;
}
{
add.f16x2 r338, r332, r335;
}
{
add.f16x2 r341, %28, %40;
}
{
add.f16x2 r344, %24, r341;
}
{
add.f16x2 r347, %32, %36;
}
{
add.f16x2 r350, r344, r347;
}
{
add.f16x2 r353, %27, %39;
}
{
mul.f16x2 r356, r353, r321;
}
{
add.f16x2 r359, %23, r356;
}
{
add.f16x2 r362, %31, %35;
}
{
mul.f16x2 r365, r362, r323;
}
{
add.f16x2 r368, r359, r365;
}
{
sub.f16x2 r371, %28, %40;
}
{
mul.f16x2 r374, r371, r322;
}
{
sub.f16x2 r377, %32, %36;
}
{
mul.f16x2 r380, r377, r324;
}
{
add.f16x2 r383, r374, r380;
}
{
sub.f16x2 r386, r368, r383;
}
{
add.f16x2 r389, %27, %39;
}
{
mul.f16x2 r392, r389, r321;
}
{
add.f16x2 r395, %23, r392;
}
{
add.f16x2 r398, %31, %35;
}
{
mul.f16x2 r401, r398, r323;
}
{
add.f16x2 r404, r395, r401;
}
{
sub.f16x2 r407, %28, %40;
}
{
mul.f16x2 r410, r407, r322;
}
{
sub.f16x2 r413, %32, %36;
}
{
mul.f16x2 r416, r413, r324;
}
{
add.f16x2 r419, r410, r416;
}
{
add.f16x2 r422, r404, r419;
}
{
add.f16x2 r425, %27, %39;
}
{
mul.f16x2 r428, r425, r323;
}
{
add.f16x2 r431, %23, r428;
}
{
add.f16x2 r434, %31, %35;
}
{
mul.f16x2 r437, r434, r325;
}
{
add.f16x2 r440, r431, r437;
}
{
sub.f16x2 r443, %28, %40;
}
{
mul.f16x2 r446, r443, r324;
}
{
sub.f16x2 r449, %32, %36;
}
{
mul.f16x2 r452, r449, r327;
}
{
add.f16x2 r455, r446, r452;
}
{
sub.f16x2 r458, r440, r455;
}
{
add.f16x2 r461, %27, %39;
}
{
mul.f16x2 r464, r461, r323;
}
{
add.f16x2 r467, %23, r464;
}
{
add.f16x2 r470, %31, %35;
}
{
mul.f16x2 r473, r470, r325;
}
{
add.f16x2 r476, r467, r473;
}
{
sub.f16x2 r479, %28, %40;
}
{
mul.f16x2 r482, r479, r324;
}
{
sub.f16x2 r485, %32, %36;
}
{
mul.f16x2 r488, r485, r327;
}
{
add.f16x2 r491, r482, r488;
}
{
add.f16x2 r494, r476, r491;
}
{
add.f16x2 r497, %28, %40;
}
{
mul.f16x2 r500, r497, r321;
}
{
add.f16x2 r503, %24, r500;
}
{
add.f16x2 r506, %32, %36;
}
{
mul.f16x2 r509, r506, r323;
}
{
add.f16x2 r512, r503, r509;
}
{
sub.f16x2 r515, %27, %39;
}
{
mul.f16x2 r518, r515, r322;
}
{
sub.f16x2 r521, %31, %35;
}
{
mul.f16x2 r524, r521, r324;
}
{
add.f16x2 r527, r518, r524;
}
{
add.f16x2 r530, r512, r527;
}
{
add.f16x2 r533, %28, %40;
}
{
mul.f16x2 r536, r533, r321;
}
{
add.f16x2 r539, %24, r536;
}
{
add.f16x2 r542, %32, %36;
}
{
mul.f16x2 r545, r542, r323;
}
{
add.f16x2 r548, r539, r545;
}
{
sub.f16x2 r551, %27, %39;
}
{
mul.f16x2 r554, r551, r322;
}
{
sub.f16x2 r557, %31, %35;
}
{
mul.f16x2 r560, r557, r324;
}
{
add.f16x2 r563, r554, r560;
}
{
sub.f16x2 r566, r548, r563;
}
{
add.f16x2 r569, %28, %40;
}
{
mul.f16x2 r572, r569, r323;
}
{
add.f16x2 r575, %24, r572;
}
{
add.f16x2 r578, %32, %36;
}
{
mul.f16x2 r581, r578, r325;
}
{
add.f16x2 r584, r575, r581;
}
{
sub.f16x2 r587, %27, %39;
}
{
mul.f16x2 r590, r587, r324;
}
{
sub.f16x2 r593, %31, %35;
}
{
mul.f16x2 r596, r593, r327;
}
{
add.f16x2 r599, r590, r596;
}
{
add.f16x2 r602, r584, r599;
}
{
add.f16x2 r605, %28, %40;
}
{
mul.f16x2 r608, r605, r323;
}
{
add.f16x2 r611, %24, r608;
}
{
add.f16x2 r614, %32, %36;
}
{
mul.f16x2 r617, r614, r325;
}
{
add.f16x2 r620, r611, r617;
}
{
sub.f16x2 r623, %27, %39;
}
{
mul.f16x2 r626, r623, r324;
}
{
sub.f16x2 r629, %31, %35;
}
{
mul.f16x2 r632, r629, r327;
}
{
add.f16x2 r635, r626, r632;
}
{
sub.f16x2 r638, r620, r635;
}
mov.f32 f190, 0f3F4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r641, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r642, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r643, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r644, {low, high};
}
mov.f32 f198, 0fBE9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r645, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r646, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r647, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r648, {low, high};
}
mov.f32 f161, 0fBF800000;
{
mul.f16x2 r659, r386, r641;
}
{
mul.f16x2 r662, r530, r642;
}
{
sub.f16x2 r665, r659, r662;
}
{
mul.f16x2 r668, r386, r642;
}
{
fma.rn.f16x2 r671, r530, r641, r668;
}
{
mul.f16x2 r675, r458, r643;
}
{
mul.f16x2 r678, r602, r644;
}
{
sub.f16x2 r681, r675, r678;
}
{
mul.f16x2 r684, r458, r644;
}
{
fma.rn.f16x2 r687, r602, r643, r684;
}
{
mul.f16x2 r691, r494, r645;
}
{
mul.f16x2 r694, r638, r646;
}
{
sub.f16x2 r697, r691, r694;
}
{
mul.f16x2 r700, r494, r646;
}
{
fma.rn.f16x2 r703, r638, r645, r700;
}
{
mul.f16x2 r707, r422, r647;
}
{
mul.f16x2 r710, r566, r648;
}
{
sub.f16x2 r713, r707, r710;
}
{
mul.f16x2 r716, r422, r648;
}
{
fma.rn.f16x2 r719, r566, r647, r716;
}
{
add.f16x2 r723, r18, r338;
}
{
add.f16x2 r726, r30, r350;
}
{
sub.f16x2 r729, r18, r338;
}
{
sub.f16x2 r732, r30, r350;
}
{
add.f16x2 r735, r66, r665;
}
{
add.f16x2 r738, r210, r671;
}
{
sub.f16x2 r741, r66, r665;
}
{
sub.f16x2 r744, r210, r671;
}
{
add.f16x2 r747, r138, r681;
}
{
add.f16x2 r750, r282, r687;
}
{
sub.f16x2 r753, r138, r681;
}
{
sub.f16x2 r756, r282, r687;
}
{
add.f16x2 r759, r174, r697;
}
{
add.f16x2 r762, r318, r703;
}
{
sub.f16x2 r765, r174, r697;
}
{
sub.f16x2 r768, r318, r703;
}
{
add.f16x2 r771, r102, r713;
}
{
add.f16x2 r774, r246, r719;
}
{
sub.f16x2 r777, r102, r713;
}
{
sub.f16x2 r780, r246, r719;
}
mul.wide.u32 rd2, r3022, 1374389535;
shr.u64 rd3, rd2, 37;
cvt.u32.u64 r3023, rd3;
mul.lo.s32 r3024, r3023, 100;
sub.s32 r3025, r3022, r3024;
mad.lo.s32 r3026, r3023, 4000, r3021;
cvt.rn.f32.u32 f225, r3025;
mul.f32 f226, f225, 0f3BCDE32E;
cos.approx.f32 f61, f226;
sin.approx.f32 f227, f226;
neg.f32 f62, f227;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f61;
cvt.rn.f16.f32 high, f62;
mov.b32 r783, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r786, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r788, {high, high};
}
{
mul.f16x2 r790, r738, r788;
}
{
neg.f16x2 r793, r790;
}
{
fma.rn.f16x2 r795, r735, r786, r793;
}
{
mul.f16x2 r799, r735, r788;
}
{
fma.rn.f16x2 r802, r738, r786, r799;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r806, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r808, {high, high};
}
mov.f32 f162, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r810, {low, high};
}
{
mul.f16x2 r811, r808, r810;
}
{
mul.f16x2 r814, r783, r806;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r817, {high, low};
}
{
fma.rn.f16x2 r819, r811, r817, r814;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r819;
mov.b32 r823, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r819;
mov.b32 r825, {high, high};
}
{
mul.f16x2 r827, r750, r825;
}
{
neg.f16x2 r830, r827;
}
{
fma.rn.f16x2 r832, r747, r823, r830;
}
{
mul.f16x2 r836, r747, r825;
}
{
fma.rn.f16x2 r839, r750, r823, r836;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r843, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r845, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r847, {low, high};
}
{
mul.f16x2 r848, r845, r847;
}
{
mul.f16x2 r851, r819, r843;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r819;
mov.b32 r854, {high, low};
}
{
fma.rn.f16x2 r856, r848, r854, r851;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r856;
mov.b32 r860, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r856;
mov.b32 r862, {high, high};
}
{
mul.f16x2 r864, r762, r862;
}
{
neg.f16x2 r867, r864;
}
{
fma.rn.f16x2 r869, r759, r860, r867;
}
{
mul.f16x2 r873, r759, r862;
}
{
fma.rn.f16x2 r876, r762, r860, r873;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r880, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r882, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r884, {low, high};
}
{
mul.f16x2 r885, r882, r884;
}
{
mul.f16x2 r888, r856, r880;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r856;
mov.b32 r891, {high, low};
}
{
fma.rn.f16x2 r893, r885, r891, r888;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r893;
mov.b32 r897, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r893;
mov.b32 r899, {high, high};
}
{
mul.f16x2 r901, r774, r899;
}
{
neg.f16x2 r904, r901;
}
{
fma.rn.f16x2 r906, r771, r897, r904;
}
{
mul.f16x2 r910, r771, r899;
}
{
fma.rn.f16x2 r913, r774, r897, r910;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r917, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r919, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r921, {low, high};
}
{
mul.f16x2 r922, r919, r921;
}
{
mul.f16x2 r925, r893, r917;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r893;
mov.b32 r928, {high, low};
}
{
fma.rn.f16x2 r930, r922, r928, r925;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r930;
mov.b32 r934, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r930;
mov.b32 r936, {high, high};
}
{
mul.f16x2 r938, r732, r936;
}
{
neg.f16x2 r941, r938;
}
{
fma.rn.f16x2 r943, r729, r934, r941;
}
{
mul.f16x2 r947, r729, r936;
}
{
fma.rn.f16x2 r950, r732, r934, r947;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r954, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r956, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r958, {low, high};
}
{
mul.f16x2 r959, r956, r958;
}
{
mul.f16x2 r962, r930, r954;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r930;
mov.b32 r965, {high, low};
}
{
fma.rn.f16x2 r967, r959, r965, r962;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r967;
mov.b32 r971, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r967;
mov.b32 r973, {high, high};
}
{
mul.f16x2 r975, r744, r973;
}
{
neg.f16x2 r978, r975;
}
{
fma.rn.f16x2 r980, r741, r971, r978;
}
{
mul.f16x2 r984, r741, r973;
}
{
fma.rn.f16x2 r987, r744, r971, r984;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r991, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r993, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r995, {low, high};
}
{
mul.f16x2 r996, r993, r995;
}
{
mul.f16x2 r999, r967, r991;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r967;
mov.b32 r1002, {high, low};
}
{
fma.rn.f16x2 r1004, r996, r1002, r999;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1004;
mov.b32 r1008, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1004;
mov.b32 r1010, {high, high};
}
{
mul.f16x2 r1012, r756, r1010;
}
{
neg.f16x2 r1015, r1012;
}
{
fma.rn.f16x2 r1017, r753, r1008, r1015;
}
{
mul.f16x2 r1021, r753, r1010;
}
{
fma.rn.f16x2 r1024, r756, r1008, r1021;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r1028, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r1030, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r1032, {low, high};
}
{
mul.f16x2 r1033, r1030, r1032;
}
{
mul.f16x2 r1036, r1004, r1028;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1004;
mov.b32 r1039, {high, low};
}
{
fma.rn.f16x2 r1041, r1033, r1039, r1036;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1041;
mov.b32 r1045, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1041;
mov.b32 r1047, {high, high};
}
{
mul.f16x2 r1049, r768, r1047;
}
{
neg.f16x2 r1052, r1049;
}
{
fma.rn.f16x2 r1054, r765, r1045, r1052;
}
{
mul.f16x2 r1058, r765, r1047;
}
{
fma.rn.f16x2 r1061, r768, r1045, r1058;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r1065, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r783;
mov.b32 r1067, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r1069, {low, high};
}
{
mul.f16x2 r1070, r1067, r1069;
}
{
mul.f16x2 r1073, r1041, r1065;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1041;
mov.b32 r1076, {high, low};
}
{
fma.rn.f16x2 r1078, r1070, r1076, r1073;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1078;
mov.b32 r1082, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1078;
mov.b32 r1084, {high, high};
}
{
mul.f16x2 r1086, r780, r1084;
}
{
neg.f16x2 r1089, r1086;
}
{
fma.rn.f16x2 r1091, r777, r1082, r1089;
}
{
mul.f16x2 r1095, r777, r1084;
}
{
fma.rn.f16x2 r1098, r780, r1082, r1095;
}
barrier.sync 0;
mad.lo.s32 r3027, r3025, 40, r3026;
st.shared.v2.f32 [r3027], {r723, r795};
st.shared.v2.f32 [r3027+8], {r832, r869};
st.shared.v2.f32 [r3027+16], {r906, r943};
st.shared.v2.f32 [r3027+24], {r980, r1017};
st.shared.v2.f32 [r3027+32], {r1054, r1091};
barrier.sync 0;
mad.lo.s32 r3028, r3025, -36, r3027;
ld.shared.u32 r1131, [r3028];
ld.shared.u32 r1451, [r3028+400];
ld.shared.u32 r1128, [r3028+800];
ld.shared.u32 r1448, [r3028+1200];
ld.shared.u32 r1134, [r3028+1600];
ld.shared.u32 r1454, [r3028+2000];
ld.shared.u32 r1135, [r3028+2400];
ld.shared.u32 r1455, [r3028+2800];
ld.shared.u32 r1129, [r3028+3200];
ld.shared.u32 r1449, [r3028+3600];
barrier.sync 0;
st.shared.v2.f32 [r3027], {r726, r802};
st.shared.v2.f32 [r3027+8], {r839, r876};
st.shared.v2.f32 [r3027+16], {r913, r950};
st.shared.v2.f32 [r3027+24], {r987, r1024};
st.shared.v2.f32 [r3027+32], {r1061, r1098};
barrier.sync 0;
ld.shared.u32 r1143, [r3028];
ld.shared.u32 r1463, [r3028+400];
ld.shared.u32 r1140, [r3028+800];
ld.shared.u32 r1460, [r3028+1200];
ld.shared.u32 r1146, [r3028+1600];
ld.shared.u32 r1466, [r3028+2000];
ld.shared.u32 r1147, [r3028+2400];
ld.shared.u32 r1467, [r3028+2800];
ld.shared.u32 r1141, [r3028+3200];
ld.shared.u32 r1461, [r3028+3600];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1119, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1120, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1121, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r1122, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1123, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1124, {low, high};
}
{
neg.f16x2 r1125, r1124;
}
{
add.f16x2 r1127, r1128, r1129;
}
{
add.f16x2 r1130, r1131, r1127;
}
{
add.f16x2 r1133, r1134, r1135;
}
{
add.f16x2 r1136, r1130, r1133;
}
{
add.f16x2 r1139, r1140, r1141;
}
{
add.f16x2 r1142, r1143, r1139;
}
{
add.f16x2 r1145, r1146, r1147;
}
{
add.f16x2 r1148, r1142, r1145;
}
{
add.f16x2 r1151, r1128, r1129;
}
{
mul.f16x2 r1154, r1151, r1119;
}
{
add.f16x2 r1157, r1131, r1154;
}
{
add.f16x2 r1160, r1134, r1135;
}
{
mul.f16x2 r1163, r1160, r1121;
}
{
add.f16x2 r1166, r1157, r1163;
}
{
sub.f16x2 r1169, r1140, r1141;
}
{
mul.f16x2 r1172, r1169, r1120;
}
{
sub.f16x2 r1175, r1146, r1147;
}
{
mul.f16x2 r1178, r1175, r1122;
}
{
add.f16x2 r1181, r1172, r1178;
}
{
sub.f16x2 r1184, r1166, r1181;
}
{
add.f16x2 r1187, r1128, r1129;
}
{
mul.f16x2 r1190, r1187, r1119;
}
{
add.f16x2 r1193, r1131, r1190;
}
{
add.f16x2 r1196, r1134, r1135;
}
{
mul.f16x2 r1199, r1196, r1121;
}
{
add.f16x2 r1202, r1193, r1199;
}
{
sub.f16x2 r1205, r1140, r1141;
}
{
mul.f16x2 r1208, r1205, r1120;
}
{
sub.f16x2 r1211, r1146, r1147;
}
{
mul.f16x2 r1214, r1211, r1122;
}
{
add.f16x2 r1217, r1208, r1214;
}
{
add.f16x2 r1220, r1202, r1217;
}
{
add.f16x2 r1223, r1128, r1129;
}
{
mul.f16x2 r1226, r1223, r1121;
}
{
add.f16x2 r1229, r1131, r1226;
}
{
add.f16x2 r1232, r1134, r1135;
}
{
mul.f16x2 r1235, r1232, r1123;
}
{
add.f16x2 r1238, r1229, r1235;
}
{
sub.f16x2 r1241, r1140, r1141;
}
{
mul.f16x2 r1244, r1241, r1122;
}
{
sub.f16x2 r1247, r1146, r1147;
}
{
mul.f16x2 r1250, r1247, r1125;
}
{
add.f16x2 r1253, r1244, r1250;
}
{
sub.f16x2 r1256, r1238, r1253;
}
{
add.f16x2 r1259, r1128, r1129;
}
{
mul.f16x2 r1262, r1259, r1121;
}
{
add.f16x2 r1265, r1131, r1262;
}
{
add.f16x2 r1268, r1134, r1135;
}
{
mul.f16x2 r1271, r1268, r1123;
}
{
add.f16x2 r1274, r1265, r1271;
}
{
sub.f16x2 r1277, r1140, r1141;
}
{
mul.f16x2 r1280, r1277, r1122;
}
{
sub.f16x2 r1283, r1146, r1147;
}
{
mul.f16x2 r1286, r1283, r1125;
}
{
add.f16x2 r1289, r1280, r1286;
}
{
add.f16x2 r1292, r1274, r1289;
}
{
add.f16x2 r1295, r1140, r1141;
}
{
mul.f16x2 r1298, r1295, r1119;
}
{
add.f16x2 r1301, r1143, r1298;
}
{
add.f16x2 r1304, r1146, r1147;
}
{
mul.f16x2 r1307, r1304, r1121;
}
{
add.f16x2 r1310, r1301, r1307;
}
{
sub.f16x2 r1313, r1128, r1129;
}
{
mul.f16x2 r1316, r1313, r1120;
}
{
sub.f16x2 r1319, r1134, r1135;
}
{
mul.f16x2 r1322, r1319, r1122;
}
{
add.f16x2 r1325, r1316, r1322;
}
{
add.f16x2 r1328, r1310, r1325;
}
{
add.f16x2 r1331, r1140, r1141;
}
{
mul.f16x2 r1334, r1331, r1119;
}
{
add.f16x2 r1337, r1143, r1334;
}
{
add.f16x2 r1340, r1146, r1147;
}
{
mul.f16x2 r1343, r1340, r1121;
}
{
add.f16x2 r1346, r1337, r1343;
}
{
sub.f16x2 r1349, r1128, r1129;
}
{
mul.f16x2 r1352, r1349, r1120;
}
{
sub.f16x2 r1355, r1134, r1135;
}
{
mul.f16x2 r1358, r1355, r1122;
}
{
add.f16x2 r1361, r1352, r1358;
}
{
sub.f16x2 r1364, r1346, r1361;
}
{
add.f16x2 r1367, r1140, r1141;
}
{
mul.f16x2 r1370, r1367, r1121;
}
{
add.f16x2 r1373, r1143, r1370;
}
{
add.f16x2 r1376, r1146, r1147;
}
{
mul.f16x2 r1379, r1376, r1123;
}
{
add.f16x2 r1382, r1373, r1379;
}
{
sub.f16x2 r1385, r1128, r1129;
}
{
mul.f16x2 r1388, r1385, r1122;
}
{
sub.f16x2 r1391, r1134, r1135;
}
{
mul.f16x2 r1394, r1391, r1125;
}
{
add.f16x2 r1397, r1388, r1394;
}
{
add.f16x2 r1400, r1382, r1397;
}
{
add.f16x2 r1403, r1140, r1141;
}
{
mul.f16x2 r1406, r1403, r1121;
}
{
add.f16x2 r1409, r1143, r1406;
}
{
add.f16x2 r1412, r1146, r1147;
}
{
mul.f16x2 r1415, r1412, r1123;
}
{
add.f16x2 r1418, r1409, r1415;
}
{
sub.f16x2 r1421, r1128, r1129;
}
{
mul.f16x2 r1424, r1421, r1122;
}
{
sub.f16x2 r1427, r1134, r1135;
}
{
mul.f16x2 r1430, r1427, r1125;
}
{
add.f16x2 r1433, r1424, r1430;
}
{
sub.f16x2 r1436, r1418, r1433;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1439, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1440, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1441, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r1442, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1443, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1444, {low, high};
}
{
neg.f16x2 r1445, r1444;
}
{
add.f16x2 r1447, r1448, r1449;
}
{
add.f16x2 r1450, r1451, r1447;
}
{
add.f16x2 r1453, r1454, r1455;
}
{
add.f16x2 r1456, r1450, r1453;
}
{
add.f16x2 r1459, r1460, r1461;
}
{
add.f16x2 r1462, r1463, r1459;
}
{
add.f16x2 r1465, r1466, r1467;
}
{
add.f16x2 r1468, r1462, r1465;
}
{
add.f16x2 r1471, r1448, r1449;
}
{
mul.f16x2 r1474, r1471, r1439;
}
{
add.f16x2 r1477, r1451, r1474;
}
{
add.f16x2 r1480, r1454, r1455;
}
{
mul.f16x2 r1483, r1480, r1441;
}
{
add.f16x2 r1486, r1477, r1483;
}
{
sub.f16x2 r1489, r1460, r1461;
}
{
mul.f16x2 r1492, r1489, r1440;
}
{
sub.f16x2 r1495, r1466, r1467;
}
{
mul.f16x2 r1498, r1495, r1442;
}
{
add.f16x2 r1501, r1492, r1498;
}
{
sub.f16x2 r1504, r1486, r1501;
}
{
add.f16x2 r1507, r1448, r1449;
}
{
mul.f16x2 r1510, r1507, r1439;
}
{
add.f16x2 r1513, r1451, r1510;
}
{
add.f16x2 r1516, r1454, r1455;
}
{
mul.f16x2 r1519, r1516, r1441;
}
{
add.f16x2 r1522, r1513, r1519;
}
{
sub.f16x2 r1525, r1460, r1461;
}
{
mul.f16x2 r1528, r1525, r1440;
}
{
sub.f16x2 r1531, r1466, r1467;
}
{
mul.f16x2 r1534, r1531, r1442;
}
{
add.f16x2 r1537, r1528, r1534;
}
{
add.f16x2 r1540, r1522, r1537;
}
{
add.f16x2 r1543, r1448, r1449;
}
{
mul.f16x2 r1546, r1543, r1441;
}
{
add.f16x2 r1549, r1451, r1546;
}
{
add.f16x2 r1552, r1454, r1455;
}
{
mul.f16x2 r1555, r1552, r1443;
}
{
add.f16x2 r1558, r1549, r1555;
}
{
sub.f16x2 r1561, r1460, r1461;
}
{
mul.f16x2 r1564, r1561, r1442;
}
{
sub.f16x2 r1567, r1466, r1467;
}
{
mul.f16x2 r1570, r1567, r1445;
}
{
add.f16x2 r1573, r1564, r1570;
}
{
sub.f16x2 r1576, r1558, r1573;
}
{
add.f16x2 r1579, r1448, r1449;
}
{
mul.f16x2 r1582, r1579, r1441;
}
{
add.f16x2 r1585, r1451, r1582;
}
{
add.f16x2 r1588, r1454, r1455;
}
{
mul.f16x2 r1591, r1588, r1443;
}
{
add.f16x2 r1594, r1585, r1591;
}
{
sub.f16x2 r1597, r1460, r1461;
}
{
mul.f16x2 r1600, r1597, r1442;
}
{
sub.f16x2 r1603, r1466, r1467;
}
{
mul.f16x2 r1606, r1603, r1445;
}
{
add.f16x2 r1609, r1600, r1606;
}
{
add.f16x2 r1612, r1594, r1609;
}
{
add.f16x2 r1615, r1460, r1461;
}
{
mul.f16x2 r1618, r1615, r1439;
}
{
add.f16x2 r1621, r1463, r1618;
}
{
add.f16x2 r1624, r1466, r1467;
}
{
mul.f16x2 r1627, r1624, r1441;
}
{
add.f16x2 r1630, r1621, r1627;
}
{
sub.f16x2 r1633, r1448, r1449;
}
{
mul.f16x2 r1636, r1633, r1440;
}
{
sub.f16x2 r1639, r1454, r1455;
}
{
mul.f16x2 r1642, r1639, r1442;
}
{
add.f16x2 r1645, r1636, r1642;
}
{
add.f16x2 r1648, r1630, r1645;
}
{
add.f16x2 r1651, r1460, r1461;
}
{
mul.f16x2 r1654, r1651, r1439;
}
{
add.f16x2 r1657, r1463, r1654;
}
{
add.f16x2 r1660, r1466, r1467;
}
{
mul.f16x2 r1663, r1660, r1441;
}
{
add.f16x2 r1666, r1657, r1663;
}
{
sub.f16x2 r1669, r1448, r1449;
}
{
mul.f16x2 r1672, r1669, r1440;
}
{
sub.f16x2 r1675, r1454, r1455;
}
{
mul.f16x2 r1678, r1675, r1442;
}
{
add.f16x2 r1681, r1672, r1678;
}
{
sub.f16x2 r1684, r1666, r1681;
}
{
add.f16x2 r1687, r1460, r1461;
}
{
mul.f16x2 r1690, r1687, r1441;
}
{
add.f16x2 r1693, r1463, r1690;
}
{
add.f16x2 r1696, r1466, r1467;
}
{
mul.f16x2 r1699, r1696, r1443;
}
{
add.f16x2 r1702, r1693, r1699;
}
{
sub.f16x2 r1705, r1448, r1449;
}
{
mul.f16x2 r1708, r1705, r1442;
}
{
sub.f16x2 r1711, r1454, r1455;
}
{
mul.f16x2 r1714, r1711, r1445;
}
{
add.f16x2 r1717, r1708, r1714;
}
{
add.f16x2 r1720, r1702, r1717;
}
{
add.f16x2 r1723, r1460, r1461;
}
{
mul.f16x2 r1726, r1723, r1441;
}
{
add.f16x2 r1729, r1463, r1726;
}
{
add.f16x2 r1732, r1466, r1467;
}
{
mul.f16x2 r1735, r1732, r1443;
}
{
add.f16x2 r1738, r1729, r1735;
}
{
sub.f16x2 r1741, r1448, r1449;
}
{
mul.f16x2 r1744, r1741, r1442;
}
{
sub.f16x2 r1747, r1454, r1455;
}
{
mul.f16x2 r1750, r1747, r1445;
}
{
add.f16x2 r1753, r1744, r1750;
}
{
sub.f16x2 r1756, r1738, r1753;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r1759, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r1760, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r1761, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1762, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1763, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1764, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1765, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r1766, {low, high};
}
{
mul.f16x2 r1777, r1504, r1759;
}
{
mul.f16x2 r1780, r1648, r1760;
}
{
sub.f16x2 r1783, r1777, r1780;
}
{
mul.f16x2 r1786, r1504, r1760;
}
{
fma.rn.f16x2 r1789, r1648, r1759, r1786;
}
{
mul.f16x2 r1793, r1576, r1761;
}
{
mul.f16x2 r1796, r1720, r1762;
}
{
sub.f16x2 r1799, r1793, r1796;
}
{
mul.f16x2 r1802, r1576, r1762;
}
{
fma.rn.f16x2 r1805, r1720, r1761, r1802;
}
{
mul.f16x2 r1809, r1612, r1763;
}
{
mul.f16x2 r1812, r1756, r1764;
}
{
sub.f16x2 r1815, r1809, r1812;
}
{
mul.f16x2 r1818, r1612, r1764;
}
{
fma.rn.f16x2 r1821, r1756, r1763, r1818;
}
{
mul.f16x2 r1825, r1540, r1765;
}
{
mul.f16x2 r1828, r1684, r1766;
}
{
sub.f16x2 r1831, r1825, r1828;
}
{
mul.f16x2 r1834, r1540, r1766;
}
{
fma.rn.f16x2 r1837, r1684, r1765, r1834;
}
{
add.f16x2 r1841, r1136, r1456;
}
{
add.f16x2 r1844, r1148, r1468;
}
{
sub.f16x2 r1847, r1136, r1456;
}
{
sub.f16x2 r1850, r1148, r1468;
}
{
add.f16x2 r1853, r1184, r1783;
}
{
add.f16x2 r1856, r1328, r1789;
}
{
sub.f16x2 r1859, r1184, r1783;
}
{
sub.f16x2 r1862, r1328, r1789;
}
{
add.f16x2 r1865, r1256, r1799;
}
{
add.f16x2 r1868, r1400, r1805;
}
{
sub.f16x2 r1871, r1256, r1799;
}
{
sub.f16x2 r1874, r1400, r1805;
}
{
add.f16x2 r1877, r1292, r1815;
}
{
add.f16x2 r1880, r1436, r1821;
}
{
sub.f16x2 r1883, r1292, r1815;
}
{
sub.f16x2 r1886, r1436, r1821;
}
{
add.f16x2 r1889, r1220, r1831;
}
{
add.f16x2 r1892, r1364, r1837;
}
{
sub.f16x2 r1895, r1220, r1831;
}
{
sub.f16x2 r1898, r1364, r1837;
}
mul.wide.u32 rd4, r3025, -858993459;
shr.u64 rd5, rd4, 35;
cvt.u32.u64 r3029, rd5;
mul.lo.s32 r3030, r3029, 10;
sub.s32 r3031, r3025, r3030;
shl.b32 r3032, r3031, 2;
add.s32 r3033, r3026, r3032;
cvt.rn.f32.u32 f228, r3029;
mul.f32 f229, f228, 0f3D80ADFD;
cos.approx.f32 f143, f229;
sin.approx.f32 f230, f229;
neg.f32 f144, f230;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f143;
cvt.rn.f16.f32 high, f144;
mov.b32 r1901, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1904, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1906, {high, high};
}
{
mul.f16x2 r1908, r1856, r1906;
}
{
neg.f16x2 r1911, r1908;
}
{
fma.rn.f16x2 r1913, r1853, r1904, r1911;
}
{
mul.f16x2 r1917, r1853, r1906;
}
{
fma.rn.f16x2 r1920, r1856, r1904, r1917;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1924, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1926, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r1928, {low, high};
}
{
mul.f16x2 r1929, r1926, r1928;
}
{
mul.f16x2 r1932, r1901, r1924;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1935, {high, low};
}
{
fma.rn.f16x2 r1937, r1929, r1935, r1932;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1937;
mov.b32 r1941, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1937;
mov.b32 r1943, {high, high};
}
{
mul.f16x2 r1945, r1868, r1943;
}
{
neg.f16x2 r1948, r1945;
}
{
fma.rn.f16x2 r1950, r1865, r1941, r1948;
}
{
mul.f16x2 r1954, r1865, r1943;
}
{
fma.rn.f16x2 r1957, r1868, r1941, r1954;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1961, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1963, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r1965, {low, high};
}
{
mul.f16x2 r1966, r1963, r1965;
}
{
mul.f16x2 r1969, r1937, r1961;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1937;
mov.b32 r1972, {high, low};
}
{
fma.rn.f16x2 r1974, r1966, r1972, r1969;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1974;
mov.b32 r1978, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1974;
mov.b32 r1980, {high, high};
}
{
mul.f16x2 r1982, r1880, r1980;
}
{
neg.f16x2 r1985, r1982;
}
{
fma.rn.f16x2 r1987, r1877, r1978, r1985;
}
{
mul.f16x2 r1991, r1877, r1980;
}
{
fma.rn.f16x2 r1994, r1880, r1978, r1991;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r1998, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2000, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2002, {low, high};
}
{
mul.f16x2 r2003, r2000, r2002;
}
{
mul.f16x2 r2006, r1974, r1998;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1974;
mov.b32 r2009, {high, low};
}
{
fma.rn.f16x2 r2011, r2003, r2009, r2006;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2011;
mov.b32 r2015, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2011;
mov.b32 r2017, {high, high};
}
{
mul.f16x2 r2019, r1892, r2017;
}
{
neg.f16x2 r2022, r2019;
}
{
fma.rn.f16x2 r2024, r1889, r2015, r2022;
}
{
mul.f16x2 r2028, r1889, r2017;
}
{
fma.rn.f16x2 r2031, r1892, r2015, r2028;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2035, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2037, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2039, {low, high};
}
{
mul.f16x2 r2040, r2037, r2039;
}
{
mul.f16x2 r2043, r2011, r2035;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2011;
mov.b32 r2046, {high, low};
}
{
fma.rn.f16x2 r2048, r2040, r2046, r2043;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2048;
mov.b32 r2052, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2048;
mov.b32 r2054, {high, high};
}
{
mul.f16x2 r2056, r1850, r2054;
}
{
neg.f16x2 r2059, r2056;
}
{
fma.rn.f16x2 r2061, r1847, r2052, r2059;
}
{
mul.f16x2 r2065, r1847, r2054;
}
{
fma.rn.f16x2 r2068, r1850, r2052, r2065;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2072, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2074, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2076, {low, high};
}
{
mul.f16x2 r2077, r2074, r2076;
}
{
mul.f16x2 r2080, r2048, r2072;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2048;
mov.b32 r2083, {high, low};
}
{
fma.rn.f16x2 r2085, r2077, r2083, r2080;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2085;
mov.b32 r2089, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2085;
mov.b32 r2091, {high, high};
}
{
mul.f16x2 r2093, r1862, r2091;
}
{
neg.f16x2 r2096, r2093;
}
{
fma.rn.f16x2 r2098, r1859, r2089, r2096;
}
{
mul.f16x2 r2102, r1859, r2091;
}
{
fma.rn.f16x2 r2105, r1862, r2089, r2102;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2109, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2111, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2113, {low, high};
}
{
mul.f16x2 r2114, r2111, r2113;
}
{
mul.f16x2 r2117, r2085, r2109;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2085;
mov.b32 r2120, {high, low};
}
{
fma.rn.f16x2 r2122, r2114, r2120, r2117;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2122;
mov.b32 r2126, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2122;
mov.b32 r2128, {high, high};
}
{
mul.f16x2 r2130, r1874, r2128;
}
{
neg.f16x2 r2133, r2130;
}
{
fma.rn.f16x2 r2135, r1871, r2126, r2133;
}
{
mul.f16x2 r2139, r1871, r2128;
}
{
fma.rn.f16x2 r2142, r1874, r2126, r2139;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2146, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2148, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2150, {low, high};
}
{
mul.f16x2 r2151, r2148, r2150;
}
{
mul.f16x2 r2154, r2122, r2146;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2122;
mov.b32 r2157, {high, low};
}
{
fma.rn.f16x2 r2159, r2151, r2157, r2154;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2159;
mov.b32 r2163, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2159;
mov.b32 r2165, {high, high};
}
{
mul.f16x2 r2167, r1886, r2165;
}
{
neg.f16x2 r2170, r2167;
}
{
fma.rn.f16x2 r2172, r1883, r2163, r2170;
}
{
mul.f16x2 r2176, r1883, r2165;
}
{
fma.rn.f16x2 r2179, r1886, r2163, r2176;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2183, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1901;
mov.b32 r2185, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f161;
cvt.rn.f16.f32 high, f162;
mov.b32 r2187, {low, high};
}
{
mul.f16x2 r2188, r2185, r2187;
}
{
mul.f16x2 r2191, r2159, r2183;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2159;
mov.b32 r2194, {high, low};
}
{
fma.rn.f16x2 r2196, r2188, r2194, r2191;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2196;
mov.b32 r2200, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2196;
mov.b32 r2202, {high, high};
}
{
mul.f16x2 r2204, r1898, r2202;
}
{
neg.f16x2 r2207, r2204;
}
{
fma.rn.f16x2 r2209, r1895, r2200, r2207;
}
{
mul.f16x2 r2213, r1895, r2202;
}
{
fma.rn.f16x2 r2216, r1898, r2200, r2213;
}
barrier.sync 0;
mad.lo.s32 r3034, r3029, 400, r3033;
st.shared.u32 [r3034], r1841;
st.shared.u32 [r3034+40], r1913;
st.shared.u32 [r3034+80], r1950;
st.shared.u32 [r3034+120], r1987;
st.shared.u32 [r3034+160], r2024;
st.shared.u32 [r3034+200], r2061;
st.shared.u32 [r3034+240], r2098;
st.shared.u32 [r3034+280], r2135;
st.shared.u32 [r3034+320], r2172;
st.shared.u32 [r3034+360], r2209;
barrier.sync 0;
ld.shared.u32 r2249, [r3028];
ld.shared.u32 r2569, [r3028+400];
ld.shared.u32 r2246, [r3028+800];
ld.shared.u32 r2566, [r3028+1200];
ld.shared.u32 r2252, [r3028+1600];
ld.shared.u32 r2572, [r3028+2000];
ld.shared.u32 r2253, [r3028+2400];
ld.shared.u32 r2573, [r3028+2800];
ld.shared.u32 r2247, [r3028+3200];
ld.shared.u32 r2567, [r3028+3600];
barrier.sync 0;
st.shared.u32 [r3034], r1844;
st.shared.u32 [r3034+40], r1920;
st.shared.u32 [r3034+80], r1957;
st.shared.u32 [r3034+120], r1994;
st.shared.u32 [r3034+160], r2031;
st.shared.u32 [r3034+200], r2068;
st.shared.u32 [r3034+240], r2105;
st.shared.u32 [r3034+280], r2142;
st.shared.u32 [r3034+320], r2179;
st.shared.u32 [r3034+360], r2216;
barrier.sync 0;
ld.shared.u32 r2261, [r3028];
ld.shared.u32 r2581, [r3028+400];
ld.shared.u32 r2258, [r3028+800];
ld.shared.u32 r2578, [r3028+1200];
ld.shared.u32 r2264, [r3028+1600];
ld.shared.u32 r2584, [r3028+2000];
ld.shared.u32 r2265, [r3028+2400];
ld.shared.u32 r2585, [r3028+2800];
ld.shared.u32 r2259, [r3028+3200];
ld.shared.u32 r2579, [r3028+3600];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r2237, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2238, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r2239, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r2240, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r2241, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2242, {low, high};
}
{
neg.f16x2 r2243, r2242;
}
{
add.f16x2 r2245, r2246, r2247;
}
{
add.f16x2 r2248, r2249, r2245;
}
{
add.f16x2 r2251, r2252, r2253;
}
{
add.f16x2 r2254, r2248, r2251;
}
{
add.f16x2 r2257, r2258, r2259;
}
{
add.f16x2 r2260, r2261, r2257;
}
{
add.f16x2 r2263, r2264, r2265;
}
{
add.f16x2 r2266, r2260, r2263;
}
{
add.f16x2 r2269, r2246, r2247;
}
{
mul.f16x2 r2272, r2269, r2237;
}
{
add.f16x2 r2275, r2249, r2272;
}
{
add.f16x2 r2278, r2252, r2253;
}
{
mul.f16x2 r2281, r2278, r2239;
}
{
add.f16x2 r2284, r2275, r2281;
}
{
sub.f16x2 r2287, r2258, r2259;
}
{
mul.f16x2 r2290, r2287, r2238;
}
{
sub.f16x2 r2293, r2264, r2265;
}
{
mul.f16x2 r2296, r2293, r2240;
}
{
add.f16x2 r2299, r2290, r2296;
}
{
sub.f16x2 r2302, r2284, r2299;
}
{
add.f16x2 r2305, r2246, r2247;
}
{
mul.f16x2 r2308, r2305, r2237;
}
{
add.f16x2 r2311, r2249, r2308;
}
{
add.f16x2 r2314, r2252, r2253;
}
{
mul.f16x2 r2317, r2314, r2239;
}
{
add.f16x2 r2320, r2311, r2317;
}
{
sub.f16x2 r2323, r2258, r2259;
}
{
mul.f16x2 r2326, r2323, r2238;
}
{
sub.f16x2 r2329, r2264, r2265;
}
{
mul.f16x2 r2332, r2329, r2240;
}
{
add.f16x2 r2335, r2326, r2332;
}
{
add.f16x2 r2338, r2320, r2335;
}
{
add.f16x2 r2341, r2246, r2247;
}
{
mul.f16x2 r2344, r2341, r2239;
}
{
add.f16x2 r2347, r2249, r2344;
}
{
add.f16x2 r2350, r2252, r2253;
}
{
mul.f16x2 r2353, r2350, r2241;
}
{
add.f16x2 r2356, r2347, r2353;
}
{
sub.f16x2 r2359, r2258, r2259;
}
{
mul.f16x2 r2362, r2359, r2240;
}
{
sub.f16x2 r2365, r2264, r2265;
}
{
mul.f16x2 r2368, r2365, r2243;
}
{
add.f16x2 r2371, r2362, r2368;
}
{
sub.f16x2 r2374, r2356, r2371;
}
{
add.f16x2 r2377, r2246, r2247;
}
{
mul.f16x2 r2380, r2377, r2239;
}
{
add.f16x2 r2383, r2249, r2380;
}
{
add.f16x2 r2386, r2252, r2253;
}
{
mul.f16x2 r2389, r2386, r2241;
}
{
add.f16x2 r2392, r2383, r2389;
}
{
sub.f16x2 r2395, r2258, r2259;
}
{
mul.f16x2 r2398, r2395, r2240;
}
{
sub.f16x2 r2401, r2264, r2265;
}
{
mul.f16x2 r2404, r2401, r2243;
}
{
add.f16x2 r2407, r2398, r2404;
}
{
add.f16x2 r2410, r2392, r2407;
}
{
add.f16x2 r2413, r2258, r2259;
}
{
mul.f16x2 r2416, r2413, r2237;
}
{
add.f16x2 r2419, r2261, r2416;
}
{
add.f16x2 r2422, r2264, r2265;
}
{
mul.f16x2 r2425, r2422, r2239;
}
{
add.f16x2 r2428, r2419, r2425;
}
{
sub.f16x2 r2431, r2246, r2247;
}
{
mul.f16x2 r2434, r2431, r2238;
}
{
sub.f16x2 r2437, r2252, r2253;
}
{
mul.f16x2 r2440, r2437, r2240;
}
{
add.f16x2 r2443, r2434, r2440;
}
{
add.f16x2 r2446, r2428, r2443;
}
{
add.f16x2 r2449, r2258, r2259;
}
{
mul.f16x2 r2452, r2449, r2237;
}
{
add.f16x2 r2455, r2261, r2452;
}
{
add.f16x2 r2458, r2264, r2265;
}
{
mul.f16x2 r2461, r2458, r2239;
}
{
add.f16x2 r2464, r2455, r2461;
}
{
sub.f16x2 r2467, r2246, r2247;
}
{
mul.f16x2 r2470, r2467, r2238;
}
{
sub.f16x2 r2473, r2252, r2253;
}
{
mul.f16x2 r2476, r2473, r2240;
}
{
add.f16x2 r2479, r2470, r2476;
}
{
sub.f16x2 r2482, r2464, r2479;
}
{
add.f16x2 r2485, r2258, r2259;
}
{
mul.f16x2 r2488, r2485, r2239;
}
{
add.f16x2 r2491, r2261, r2488;
}
{
add.f16x2 r2494, r2264, r2265;
}
{
mul.f16x2 r2497, r2494, r2241;
}
{
add.f16x2 r2500, r2491, r2497;
}
{
sub.f16x2 r2503, r2246, r2247;
}
{
mul.f16x2 r2506, r2503, r2240;
}
{
sub.f16x2 r2509, r2252, r2253;
}
{
mul.f16x2 r2512, r2509, r2243;
}
{
add.f16x2 r2515, r2506, r2512;
}
{
add.f16x2 r2518, r2500, r2515;
}
{
add.f16x2 r2521, r2258, r2259;
}
{
mul.f16x2 r2524, r2521, r2239;
}
{
add.f16x2 r2527, r2261, r2524;
}
{
add.f16x2 r2530, r2264, r2265;
}
{
mul.f16x2 r2533, r2530, r2241;
}
{
add.f16x2 r2536, r2527, r2533;
}
{
sub.f16x2 r2539, r2246, r2247;
}
{
mul.f16x2 r2542, r2539, r2240;
}
{
sub.f16x2 r2545, r2252, r2253;
}
{
mul.f16x2 r2548, r2545, r2243;
}
{
add.f16x2 r2551, r2542, r2548;
}
{
sub.f16x2 r2554, r2536, r2551;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r2557, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2558, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r2559, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r2560, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r2561, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2562, {low, high};
}
{
neg.f16x2 r2563, r2562;
}
{
add.f16x2 r2565, r2566, r2567;
}
{
add.f16x2 r2568, r2569, r2565;
}
{
add.f16x2 r2571, r2572, r2573;
}
{
add.f16x2 r2574, r2568, r2571;
}
{
add.f16x2 r2577, r2578, r2579;
}
{
add.f16x2 r2580, r2581, r2577;
}
{
add.f16x2 r2583, r2584, r2585;
}
{
add.f16x2 r2586, r2580, r2583;
}
{
add.f16x2 r2589, r2566, r2567;
}
{
mul.f16x2 r2592, r2589, r2557;
}
{
add.f16x2 r2595, r2569, r2592;
}
{
add.f16x2 r2598, r2572, r2573;
}
{
mul.f16x2 r2601, r2598, r2559;
}
{
add.f16x2 r2604, r2595, r2601;
}
{
sub.f16x2 r2607, r2578, r2579;
}
{
mul.f16x2 r2610, r2607, r2558;
}
{
sub.f16x2 r2613, r2584, r2585;
}
{
mul.f16x2 r2616, r2613, r2560;
}
{
add.f16x2 r2619, r2610, r2616;
}
{
sub.f16x2 r2622, r2604, r2619;
}
{
add.f16x2 r2625, r2566, r2567;
}
{
mul.f16x2 r2628, r2625, r2557;
}
{
add.f16x2 r2631, r2569, r2628;
}
{
add.f16x2 r2634, r2572, r2573;
}
{
mul.f16x2 r2637, r2634, r2559;
}
{
add.f16x2 r2640, r2631, r2637;
}
{
sub.f16x2 r2643, r2578, r2579;
}
{
mul.f16x2 r2646, r2643, r2558;
}
{
sub.f16x2 r2649, r2584, r2585;
}
{
mul.f16x2 r2652, r2649, r2560;
}
{
add.f16x2 r2655, r2646, r2652;
}
{
add.f16x2 r2658, r2640, r2655;
}
{
add.f16x2 r2661, r2566, r2567;
}
{
mul.f16x2 r2664, r2661, r2559;
}
{
add.f16x2 r2667, r2569, r2664;
}
{
add.f16x2 r2670, r2572, r2573;
}
{
mul.f16x2 r2673, r2670, r2561;
}
{
add.f16x2 r2676, r2667, r2673;
}
{
sub.f16x2 r2679, r2578, r2579;
}
{
mul.f16x2 r2682, r2679, r2560;
}
{
sub.f16x2 r2685, r2584, r2585;
}
{
mul.f16x2 r2688, r2685, r2563;
}
{
add.f16x2 r2691, r2682, r2688;
}
{
sub.f16x2 r2694, r2676, r2691;
}
{
add.f16x2 r2697, r2566, r2567;
}
{
mul.f16x2 r2700, r2697, r2559;
}
{
add.f16x2 r2703, r2569, r2700;
}
{
add.f16x2 r2706, r2572, r2573;
}
{
mul.f16x2 r2709, r2706, r2561;
}
{
add.f16x2 r2712, r2703, r2709;
}
{
sub.f16x2 r2715, r2578, r2579;
}
{
mul.f16x2 r2718, r2715, r2560;
}
{
sub.f16x2 r2721, r2584, r2585;
}
{
mul.f16x2 r2724, r2721, r2563;
}
{
add.f16x2 r2727, r2718, r2724;
}
{
add.f16x2 r2730, r2712, r2727;
}
{
add.f16x2 r2733, r2578, r2579;
}
{
mul.f16x2 r2736, r2733, r2557;
}
{
add.f16x2 r2739, r2581, r2736;
}
{
add.f16x2 r2742, r2584, r2585;
}
{
mul.f16x2 r2745, r2742, r2559;
}
{
add.f16x2 r2748, r2739, r2745;
}
{
sub.f16x2 r2751, r2566, r2567;
}
{
mul.f16x2 r2754, r2751, r2558;
}
{
sub.f16x2 r2757, r2572, r2573;
}
{
mul.f16x2 r2760, r2757, r2560;
}
{
add.f16x2 r2763, r2754, r2760;
}
{
add.f16x2 r2766, r2748, r2763;
}
{
add.f16x2 r2769, r2578, r2579;
}
{
mul.f16x2 r2772, r2769, r2557;
}
{
add.f16x2 r2775, r2581, r2772;
}
{
add.f16x2 r2778, r2584, r2585;
}
{
mul.f16x2 r2781, r2778, r2559;
}
{
add.f16x2 r2784, r2775, r2781;
}
{
sub.f16x2 r2787, r2566, r2567;
}
{
mul.f16x2 r2790, r2787, r2558;
}
{
sub.f16x2 r2793, r2572, r2573;
}
{
mul.f16x2 r2796, r2793, r2560;
}
{
add.f16x2 r2799, r2790, r2796;
}
{
sub.f16x2 r2802, r2784, r2799;
}
{
add.f16x2 r2805, r2578, r2579;
}
{
mul.f16x2 r2808, r2805, r2559;
}
{
add.f16x2 r2811, r2581, r2808;
}
{
add.f16x2 r2814, r2584, r2585;
}
{
mul.f16x2 r2817, r2814, r2561;
}
{
add.f16x2 r2820, r2811, r2817;
}
{
sub.f16x2 r2823, r2566, r2567;
}
{
mul.f16x2 r2826, r2823, r2560;
}
{
sub.f16x2 r2829, r2572, r2573;
}
{
mul.f16x2 r2832, r2829, r2563;
}
{
add.f16x2 r2835, r2826, r2832;
}
{
add.f16x2 r2838, r2820, r2835;
}
{
add.f16x2 r2841, r2578, r2579;
}
{
mul.f16x2 r2844, r2841, r2559;
}
{
add.f16x2 r2847, r2581, r2844;
}
{
add.f16x2 r2850, r2584, r2585;
}
{
mul.f16x2 r2853, r2850, r2561;
}
{
add.f16x2 r2856, r2847, r2853;
}
{
sub.f16x2 r2859, r2566, r2567;
}
{
mul.f16x2 r2862, r2859, r2560;
}
{
sub.f16x2 r2865, r2572, r2573;
}
{
mul.f16x2 r2868, r2865, r2563;
}
{
add.f16x2 r2871, r2862, r2868;
}
{
sub.f16x2 r2874, r2856, r2871;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f190;
cvt.rn.f16.f32 high, f190;
mov.b32 r2877, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r2878, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f194;
cvt.rn.f16.f32 high, f194;
mov.b32 r2879, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2880, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r2881, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2882, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r2883, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f204;
cvt.rn.f16.f32 high, f204;
mov.b32 r2884, {low, high};
}
{
mul.f16x2 r2895, r2622, r2877;
}
{
mul.f16x2 r2898, r2766, r2878;
}
{
sub.f16x2 r2901, r2895, r2898;
}
{
mul.f16x2 r2904, r2622, r2878;
}
{
fma.rn.f16x2 r2907, r2766, r2877, r2904;
}
{
mul.f16x2 r2911, r2694, r2879;
}
{
mul.f16x2 r2914, r2838, r2880;
}
{
sub.f16x2 r2917, r2911, r2914;
}
{
mul.f16x2 r2920, r2694, r2880;
}
{
fma.rn.f16x2 r2923, r2838, r2879, r2920;
}
{
mul.f16x2 r2927, r2730, r2881;
}
{
mul.f16x2 r2930, r2874, r2882;
}
{
sub.f16x2 r2933, r2927, r2930;
}
{
mul.f16x2 r2936, r2730, r2882;
}
{
fma.rn.f16x2 r2939, r2874, r2881, r2936;
}
{
mul.f16x2 r2943, r2658, r2883;
}
{
mul.f16x2 r2946, r2802, r2884;
}
{
sub.f16x2 r2949, r2943, r2946;
}
{
mul.f16x2 r2952, r2658, r2884;
}
{
fma.rn.f16x2 r2955, r2802, r2883, r2952;
}
{
add.f16x2 %0, r2254, r2574;
}
{
add.f16x2 %1, r2266, r2586;
}
{
sub.f16x2 %10, r2254, r2574;
}
{
sub.f16x2 %11, r2266, r2586;
}
{
add.f16x2 %2, r2302, r2901;
}
{
add.f16x2 %3, r2446, r2907;
}
{
sub.f16x2 %12, r2302, r2901;
}
{
sub.f16x2 %13, r2446, r2907;
}
{
add.f16x2 %4, r2374, r2917;
}
{
add.f16x2 %5, r2518, r2923;
}
{
sub.f16x2 %14, r2374, r2917;
}
{
sub.f16x2 %15, r2518, r2923;
}
{
add.f16x2 %6, r2410, r2933;
}
{
add.f16x2 %7, r2554, r2939;
}
{
sub.f16x2 %16, r2410, r2933;
}
{
sub.f16x2 %17, r2554, r2939;
}
{
add.f16x2 %8, r2338, r2949;
}
{
add.f16x2 %9, r2482, r2955;
}
{
sub.f16x2 %18, r2338, r2949;
}
{
sub.f16x2 %19, r2482, r2955;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)));
};


#endif
