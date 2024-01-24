#ifndef CUFFTDX_FFT_10000_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_10000_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<941, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<316>;
.reg .b32 r<4162>;
.reg .b64 rd<9>;
mov.u32 r4137, %tid.y;
shl.b32 r4138, r4137, 1;
mov.u32 r4139, %20;
mad.lo.s32 r4140, r4138, 40000, r4139;
mov.u32 r4141, %tid.x;
mov.f32 f276, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r1, {low, high};
}
mov.f32 f282, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r2, {low, high};
}
mov.f32 f284, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r3, {low, high};
}
mov.f32 f286, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r4, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r5, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
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
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r321, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r322, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r323, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r324, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r325, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
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
mov.f32 f272, 0f3F4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r641, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r642, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r643, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r644, {low, high};
}
mov.f32 f280, 0fBE9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r645, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r646, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r647, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r648, {low, high};
}
mov.f32 f243, 0fBF800000;
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
mul.wide.u32 rd2, r4141, 274877907;
shr.u64 rd3, rd2, 38;
cvt.u32.u64 r4142, rd3;
mul.lo.s32 r4143, r4142, 1000;
sub.s32 r4144, r4141, r4143;
shr.u64 rd4, rd2, 37;
cvt.u32.u64 r4145, rd4;
and.b32 r4146, r4145, 134217726;
mad.lo.s32 r4147, r4146, 40000, r4140;
cvt.rn.f32.u32 f307, r4144;
mul.f32 f308, f307, 0f3A24B5BE;
cos.approx.f32 f61, f308;
sin.approx.f32 f309, f308;
neg.f32 f62, f309;
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
mov.f32 f244, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
mad.lo.s32 r4148, r4144, 80, r4147;
st.shared.v2.f32 [r4148], {r723, r726};
st.shared.v2.f32 [r4148+8], {r795, r802};
st.shared.v2.f32 [r4148+16], {r832, r839};
st.shared.v2.f32 [r4148+24], {r869, r876};
st.shared.v2.f32 [r4148+32], {r906, r913};
st.shared.v2.f32 [r4148+40], {r943, r950};
st.shared.v2.f32 [r4148+48], {r980, r987};
st.shared.v2.f32 [r4148+56], {r1017, r1024};
st.shared.v2.f32 [r4148+64], {r1054, r1061};
st.shared.v2.f32 [r4148+72], {r1091, r1098};
barrier.sync 0;
mad.lo.s32 r4149, r4144, -72, r4148;
ld.shared.u32 r1131, [r4149];
ld.shared.u32 r1143, [r4149+4];
ld.shared.u32 r1451, [r4149+8000];
ld.shared.u32 r1463, [r4149+8004];
ld.shared.u32 r1128, [r4149+16000];
ld.shared.u32 r1140, [r4149+16004];
ld.shared.u32 r1448, [r4149+24000];
ld.shared.u32 r1460, [r4149+24004];
ld.shared.u32 r1134, [r4149+32000];
ld.shared.u32 r1146, [r4149+32004];
ld.shared.u32 r1454, [r4149+40000];
ld.shared.u32 r1466, [r4149+40004];
ld.shared.u32 r1135, [r4149+48000];
ld.shared.u32 r1147, [r4149+48004];
ld.shared.u32 r1455, [r4149+56000];
ld.shared.u32 r1467, [r4149+56004];
ld.shared.u32 r1129, [r4149+64000];
ld.shared.u32 r1141, [r4149+64004];
ld.shared.u32 r1449, [r4149+72000];
ld.shared.u32 r1461, [r4149+72004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r1119, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r1120, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r1121, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r1122, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r1123, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
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
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r1439, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r1440, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r1441, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r1442, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r1443, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
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
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r1759, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r1760, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r1761, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r1762, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r1763, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r1764, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r1765, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
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
mul.wide.u32 rd5, r4144, -858993459;
shr.u64 rd6, rd5, 35;
cvt.u32.u64 r4150, rd6;
cvt.rn.f32.u32 f310, r4150;
mul.f32 f311, f310, 0f3BCDE32E;
cos.approx.f32 f143, f311;
sin.approx.f32 f312, f311;
neg.f32 f144, f312;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f143;
cvt.rn.f16.f32 high, f144;
mov.b32 r1901, {low, high};
}
mul.lo.s32 r4151, r4150, 10;
sub.s32 r4152, r4144, r4151;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
shl.b32 r4153, r4152, 3;
add.s32 r4154, r4147, r4153;
barrier.sync 0;
mad.lo.s32 r4155, r4150, 800, r4154;
st.shared.u32 [r4155], r1841;
st.shared.u32 [r4155+4], r1844;
st.shared.u32 [r4155+80], r1913;
st.shared.u32 [r4155+84], r1920;
st.shared.u32 [r4155+160], r1950;
st.shared.u32 [r4155+164], r1957;
st.shared.u32 [r4155+240], r1987;
st.shared.u32 [r4155+244], r1994;
st.shared.u32 [r4155+320], r2024;
st.shared.u32 [r4155+324], r2031;
st.shared.u32 [r4155+400], r2061;
st.shared.u32 [r4155+404], r2068;
st.shared.u32 [r4155+480], r2098;
st.shared.u32 [r4155+484], r2105;
st.shared.u32 [r4155+560], r2135;
st.shared.u32 [r4155+564], r2142;
st.shared.u32 [r4155+640], r2172;
st.shared.u32 [r4155+644], r2179;
st.shared.u32 [r4155+720], r2209;
st.shared.u32 [r4155+724], r2216;
barrier.sync 0;
ld.shared.u32 r2249, [r4149];
ld.shared.u32 r2261, [r4149+4];
ld.shared.u32 r2569, [r4149+8000];
ld.shared.u32 r2581, [r4149+8004];
ld.shared.u32 r2246, [r4149+16000];
ld.shared.u32 r2258, [r4149+16004];
ld.shared.u32 r2566, [r4149+24000];
ld.shared.u32 r2578, [r4149+24004];
ld.shared.u32 r2252, [r4149+32000];
ld.shared.u32 r2264, [r4149+32004];
ld.shared.u32 r2572, [r4149+40000];
ld.shared.u32 r2584, [r4149+40004];
ld.shared.u32 r2253, [r4149+48000];
ld.shared.u32 r2265, [r4149+48004];
ld.shared.u32 r2573, [r4149+56000];
ld.shared.u32 r2585, [r4149+56004];
ld.shared.u32 r2247, [r4149+64000];
ld.shared.u32 r2259, [r4149+64004];
ld.shared.u32 r2567, [r4149+72000];
ld.shared.u32 r2579, [r4149+72004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r2237, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r2238, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r2239, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r2240, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r2241, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
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
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r2557, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r2558, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r2559, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r2560, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r2561, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
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
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r2877, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r2878, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r2879, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r2880, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r2881, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r2882, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r2883, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
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
add.f16x2 r2959, r2254, r2574;
}
{
add.f16x2 r2962, r2266, r2586;
}
{
sub.f16x2 r2965, r2254, r2574;
}
{
sub.f16x2 r2968, r2266, r2586;
}
{
add.f16x2 r2971, r2302, r2901;
}
{
add.f16x2 r2974, r2446, r2907;
}
{
sub.f16x2 r2977, r2302, r2901;
}
{
sub.f16x2 r2980, r2446, r2907;
}
{
add.f16x2 r2983, r2374, r2917;
}
{
add.f16x2 r2986, r2518, r2923;
}
{
sub.f16x2 r2989, r2374, r2917;
}
{
sub.f16x2 r2992, r2518, r2923;
}
{
add.f16x2 r2995, r2410, r2933;
}
{
add.f16x2 r2998, r2554, r2939;
}
{
sub.f16x2 r3001, r2410, r2933;
}
{
sub.f16x2 r3004, r2554, r2939;
}
{
add.f16x2 r3007, r2338, r2949;
}
{
add.f16x2 r3010, r2482, r2955;
}
{
sub.f16x2 r3013, r2338, r2949;
}
{
sub.f16x2 r3016, r2482, r2955;
}
mul.wide.u32 rd7, r4144, 1374389535;
shr.u64 rd8, rd7, 37;
cvt.u32.u64 r4156, rd8;
cvt.rn.f32.u32 f313, r4156;
mul.f32 f314, f313, 0f3D80ADFD;
cos.approx.f32 f225, f314;
sin.approx.f32 f315, f314;
neg.f32 f226, f315;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r3019, {low, high};
}
mul.lo.s32 r4157, r4156, 100;
sub.s32 r4158, r4144, r4157;
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3022, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3024, {high, high};
}
{
mul.f16x2 r3026, r2974, r3024;
}
{
neg.f16x2 r3029, r3026;
}
{
fma.rn.f16x2 r3031, r2971, r3022, r3029;
}
{
mul.f16x2 r3035, r2971, r3024;
}
{
fma.rn.f16x2 r3038, r2974, r3022, r3035;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3042, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3044, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
mov.b32 r3046, {low, high};
}
{
mul.f16x2 r3047, r3044, r3046;
}
{
mul.f16x2 r3050, r3019, r3042;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
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
mul.f16x2 r3063, r2986, r3061;
}
{
neg.f16x2 r3066, r3063;
}
{
fma.rn.f16x2 r3068, r2983, r3059, r3066;
}
{
mul.f16x2 r3072, r2983, r3061;
}
{
fma.rn.f16x2 r3075, r2986, r3059, r3072;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3079, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3081, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
mul.f16x2 r3100, r2998, r3098;
}
{
neg.f16x2 r3103, r3100;
}
{
fma.rn.f16x2 r3105, r2995, r3096, r3103;
}
{
mul.f16x2 r3109, r2995, r3098;
}
{
fma.rn.f16x2 r3112, r2998, r3096, r3109;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3116, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3118, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
mul.f16x2 r3137, r3010, r3135;
}
{
neg.f16x2 r3140, r3137;
}
{
fma.rn.f16x2 r3142, r3007, r3133, r3140;
}
{
mul.f16x2 r3146, r3007, r3135;
}
{
fma.rn.f16x2 r3149, r3010, r3133, r3146;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3153, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3155, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
mul.f16x2 r3174, r2968, r3172;
}
{
neg.f16x2 r3177, r3174;
}
{
fma.rn.f16x2 r3179, r2965, r3170, r3177;
}
{
mul.f16x2 r3183, r2965, r3172;
}
{
fma.rn.f16x2 r3186, r2968, r3170, r3183;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3190, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3192, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
mul.f16x2 r3211, r2980, r3209;
}
{
neg.f16x2 r3214, r3211;
}
{
fma.rn.f16x2 r3216, r2977, r3207, r3214;
}
{
mul.f16x2 r3220, r2977, r3209;
}
{
fma.rn.f16x2 r3223, r2980, r3207, r3220;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3227, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3229, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
mul.f16x2 r3248, r2992, r3246;
}
{
neg.f16x2 r3251, r3248;
}
{
fma.rn.f16x2 r3253, r2989, r3244, r3251;
}
{
mul.f16x2 r3257, r2989, r3246;
}
{
fma.rn.f16x2 r3260, r2992, r3244, r3257;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3264, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3266, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
mul.f16x2 r3285, r3004, r3283;
}
{
neg.f16x2 r3288, r3285;
}
{
fma.rn.f16x2 r3290, r3001, r3281, r3288;
}
{
mul.f16x2 r3294, r3001, r3283;
}
{
fma.rn.f16x2 r3297, r3004, r3281, r3294;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3301, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3303, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
mul.f16x2 r3322, r3016, r3320;
}
{
neg.f16x2 r3325, r3322;
}
{
fma.rn.f16x2 r3327, r3013, r3318, r3325;
}
{
mul.f16x2 r3331, r3013, r3320;
}
{
fma.rn.f16x2 r3334, r3016, r3318, r3331;
}
shl.b32 r4159, r4158, 3;
add.s32 r4160, r4147, r4159;
barrier.sync 0;
mad.lo.s32 r4161, r4156, 8000, r4160;
st.shared.u32 [r4161], r2959;
st.shared.u32 [r4161+4], r2962;
st.shared.u32 [r4161+800], r3031;
st.shared.u32 [r4161+804], r3038;
st.shared.u32 [r4161+1600], r3068;
st.shared.u32 [r4161+1604], r3075;
st.shared.u32 [r4161+2400], r3105;
st.shared.u32 [r4161+2404], r3112;
st.shared.u32 [r4161+3200], r3142;
st.shared.u32 [r4161+3204], r3149;
st.shared.u32 [r4161+4000], r3179;
st.shared.u32 [r4161+4004], r3186;
st.shared.u32 [r4161+4800], r3216;
st.shared.u32 [r4161+4804], r3223;
st.shared.u32 [r4161+5600], r3253;
st.shared.u32 [r4161+5604], r3260;
st.shared.u32 [r4161+6400], r3290;
st.shared.u32 [r4161+6404], r3297;
st.shared.u32 [r4161+7200], r3327;
st.shared.u32 [r4161+7204], r3334;
barrier.sync 0;
ld.shared.u32 r3367, [r4149];
ld.shared.u32 r3379, [r4149+4];
ld.shared.u32 r3687, [r4149+8000];
ld.shared.u32 r3699, [r4149+8004];
ld.shared.u32 r3364, [r4149+16000];
ld.shared.u32 r3376, [r4149+16004];
ld.shared.u32 r3684, [r4149+24000];
ld.shared.u32 r3696, [r4149+24004];
ld.shared.u32 r3370, [r4149+32000];
ld.shared.u32 r3382, [r4149+32004];
ld.shared.u32 r3690, [r4149+40000];
ld.shared.u32 r3702, [r4149+40004];
ld.shared.u32 r3371, [r4149+48000];
ld.shared.u32 r3383, [r4149+48004];
ld.shared.u32 r3691, [r4149+56000];
ld.shared.u32 r3703, [r4149+56004];
ld.shared.u32 r3365, [r4149+64000];
ld.shared.u32 r3377, [r4149+64004];
ld.shared.u32 r3685, [r4149+72000];
ld.shared.u32 r3697, [r4149+72004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r3355, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r3356, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r3357, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r3358, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r3359, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r3360, {low, high};
}
{
neg.f16x2 r3361, r3360;
}
{
add.f16x2 r3363, r3364, r3365;
}
{
add.f16x2 r3366, r3367, r3363;
}
{
add.f16x2 r3369, r3370, r3371;
}
{
add.f16x2 r3372, r3366, r3369;
}
{
add.f16x2 r3375, r3376, r3377;
}
{
add.f16x2 r3378, r3379, r3375;
}
{
add.f16x2 r3381, r3382, r3383;
}
{
add.f16x2 r3384, r3378, r3381;
}
{
add.f16x2 r3387, r3364, r3365;
}
{
mul.f16x2 r3390, r3387, r3355;
}
{
add.f16x2 r3393, r3367, r3390;
}
{
add.f16x2 r3396, r3370, r3371;
}
{
mul.f16x2 r3399, r3396, r3357;
}
{
add.f16x2 r3402, r3393, r3399;
}
{
sub.f16x2 r3405, r3376, r3377;
}
{
mul.f16x2 r3408, r3405, r3356;
}
{
sub.f16x2 r3411, r3382, r3383;
}
{
mul.f16x2 r3414, r3411, r3358;
}
{
add.f16x2 r3417, r3408, r3414;
}
{
sub.f16x2 r3420, r3402, r3417;
}
{
add.f16x2 r3423, r3364, r3365;
}
{
mul.f16x2 r3426, r3423, r3355;
}
{
add.f16x2 r3429, r3367, r3426;
}
{
add.f16x2 r3432, r3370, r3371;
}
{
mul.f16x2 r3435, r3432, r3357;
}
{
add.f16x2 r3438, r3429, r3435;
}
{
sub.f16x2 r3441, r3376, r3377;
}
{
mul.f16x2 r3444, r3441, r3356;
}
{
sub.f16x2 r3447, r3382, r3383;
}
{
mul.f16x2 r3450, r3447, r3358;
}
{
add.f16x2 r3453, r3444, r3450;
}
{
add.f16x2 r3456, r3438, r3453;
}
{
add.f16x2 r3459, r3364, r3365;
}
{
mul.f16x2 r3462, r3459, r3357;
}
{
add.f16x2 r3465, r3367, r3462;
}
{
add.f16x2 r3468, r3370, r3371;
}
{
mul.f16x2 r3471, r3468, r3359;
}
{
add.f16x2 r3474, r3465, r3471;
}
{
sub.f16x2 r3477, r3376, r3377;
}
{
mul.f16x2 r3480, r3477, r3358;
}
{
sub.f16x2 r3483, r3382, r3383;
}
{
mul.f16x2 r3486, r3483, r3361;
}
{
add.f16x2 r3489, r3480, r3486;
}
{
sub.f16x2 r3492, r3474, r3489;
}
{
add.f16x2 r3495, r3364, r3365;
}
{
mul.f16x2 r3498, r3495, r3357;
}
{
add.f16x2 r3501, r3367, r3498;
}
{
add.f16x2 r3504, r3370, r3371;
}
{
mul.f16x2 r3507, r3504, r3359;
}
{
add.f16x2 r3510, r3501, r3507;
}
{
sub.f16x2 r3513, r3376, r3377;
}
{
mul.f16x2 r3516, r3513, r3358;
}
{
sub.f16x2 r3519, r3382, r3383;
}
{
mul.f16x2 r3522, r3519, r3361;
}
{
add.f16x2 r3525, r3516, r3522;
}
{
add.f16x2 r3528, r3510, r3525;
}
{
add.f16x2 r3531, r3376, r3377;
}
{
mul.f16x2 r3534, r3531, r3355;
}
{
add.f16x2 r3537, r3379, r3534;
}
{
add.f16x2 r3540, r3382, r3383;
}
{
mul.f16x2 r3543, r3540, r3357;
}
{
add.f16x2 r3546, r3537, r3543;
}
{
sub.f16x2 r3549, r3364, r3365;
}
{
mul.f16x2 r3552, r3549, r3356;
}
{
sub.f16x2 r3555, r3370, r3371;
}
{
mul.f16x2 r3558, r3555, r3358;
}
{
add.f16x2 r3561, r3552, r3558;
}
{
add.f16x2 r3564, r3546, r3561;
}
{
add.f16x2 r3567, r3376, r3377;
}
{
mul.f16x2 r3570, r3567, r3355;
}
{
add.f16x2 r3573, r3379, r3570;
}
{
add.f16x2 r3576, r3382, r3383;
}
{
mul.f16x2 r3579, r3576, r3357;
}
{
add.f16x2 r3582, r3573, r3579;
}
{
sub.f16x2 r3585, r3364, r3365;
}
{
mul.f16x2 r3588, r3585, r3356;
}
{
sub.f16x2 r3591, r3370, r3371;
}
{
mul.f16x2 r3594, r3591, r3358;
}
{
add.f16x2 r3597, r3588, r3594;
}
{
sub.f16x2 r3600, r3582, r3597;
}
{
add.f16x2 r3603, r3376, r3377;
}
{
mul.f16x2 r3606, r3603, r3357;
}
{
add.f16x2 r3609, r3379, r3606;
}
{
add.f16x2 r3612, r3382, r3383;
}
{
mul.f16x2 r3615, r3612, r3359;
}
{
add.f16x2 r3618, r3609, r3615;
}
{
sub.f16x2 r3621, r3364, r3365;
}
{
mul.f16x2 r3624, r3621, r3358;
}
{
sub.f16x2 r3627, r3370, r3371;
}
{
mul.f16x2 r3630, r3627, r3361;
}
{
add.f16x2 r3633, r3624, r3630;
}
{
add.f16x2 r3636, r3618, r3633;
}
{
add.f16x2 r3639, r3376, r3377;
}
{
mul.f16x2 r3642, r3639, r3357;
}
{
add.f16x2 r3645, r3379, r3642;
}
{
add.f16x2 r3648, r3382, r3383;
}
{
mul.f16x2 r3651, r3648, r3359;
}
{
add.f16x2 r3654, r3645, r3651;
}
{
sub.f16x2 r3657, r3364, r3365;
}
{
mul.f16x2 r3660, r3657, r3358;
}
{
sub.f16x2 r3663, r3370, r3371;
}
{
mul.f16x2 r3666, r3663, r3361;
}
{
add.f16x2 r3669, r3660, r3666;
}
{
sub.f16x2 r3672, r3654, r3669;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r3675, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r3676, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r3677, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r3678, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r3679, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r3680, {low, high};
}
{
neg.f16x2 r3681, r3680;
}
{
add.f16x2 r3683, r3684, r3685;
}
{
add.f16x2 r3686, r3687, r3683;
}
{
add.f16x2 r3689, r3690, r3691;
}
{
add.f16x2 r3692, r3686, r3689;
}
{
add.f16x2 r3695, r3696, r3697;
}
{
add.f16x2 r3698, r3699, r3695;
}
{
add.f16x2 r3701, r3702, r3703;
}
{
add.f16x2 r3704, r3698, r3701;
}
{
add.f16x2 r3707, r3684, r3685;
}
{
mul.f16x2 r3710, r3707, r3675;
}
{
add.f16x2 r3713, r3687, r3710;
}
{
add.f16x2 r3716, r3690, r3691;
}
{
mul.f16x2 r3719, r3716, r3677;
}
{
add.f16x2 r3722, r3713, r3719;
}
{
sub.f16x2 r3725, r3696, r3697;
}
{
mul.f16x2 r3728, r3725, r3676;
}
{
sub.f16x2 r3731, r3702, r3703;
}
{
mul.f16x2 r3734, r3731, r3678;
}
{
add.f16x2 r3737, r3728, r3734;
}
{
sub.f16x2 r3740, r3722, r3737;
}
{
add.f16x2 r3743, r3684, r3685;
}
{
mul.f16x2 r3746, r3743, r3675;
}
{
add.f16x2 r3749, r3687, r3746;
}
{
add.f16x2 r3752, r3690, r3691;
}
{
mul.f16x2 r3755, r3752, r3677;
}
{
add.f16x2 r3758, r3749, r3755;
}
{
sub.f16x2 r3761, r3696, r3697;
}
{
mul.f16x2 r3764, r3761, r3676;
}
{
sub.f16x2 r3767, r3702, r3703;
}
{
mul.f16x2 r3770, r3767, r3678;
}
{
add.f16x2 r3773, r3764, r3770;
}
{
add.f16x2 r3776, r3758, r3773;
}
{
add.f16x2 r3779, r3684, r3685;
}
{
mul.f16x2 r3782, r3779, r3677;
}
{
add.f16x2 r3785, r3687, r3782;
}
{
add.f16x2 r3788, r3690, r3691;
}
{
mul.f16x2 r3791, r3788, r3679;
}
{
add.f16x2 r3794, r3785, r3791;
}
{
sub.f16x2 r3797, r3696, r3697;
}
{
mul.f16x2 r3800, r3797, r3678;
}
{
sub.f16x2 r3803, r3702, r3703;
}
{
mul.f16x2 r3806, r3803, r3681;
}
{
add.f16x2 r3809, r3800, r3806;
}
{
sub.f16x2 r3812, r3794, r3809;
}
{
add.f16x2 r3815, r3684, r3685;
}
{
mul.f16x2 r3818, r3815, r3677;
}
{
add.f16x2 r3821, r3687, r3818;
}
{
add.f16x2 r3824, r3690, r3691;
}
{
mul.f16x2 r3827, r3824, r3679;
}
{
add.f16x2 r3830, r3821, r3827;
}
{
sub.f16x2 r3833, r3696, r3697;
}
{
mul.f16x2 r3836, r3833, r3678;
}
{
sub.f16x2 r3839, r3702, r3703;
}
{
mul.f16x2 r3842, r3839, r3681;
}
{
add.f16x2 r3845, r3836, r3842;
}
{
add.f16x2 r3848, r3830, r3845;
}
{
add.f16x2 r3851, r3696, r3697;
}
{
mul.f16x2 r3854, r3851, r3675;
}
{
add.f16x2 r3857, r3699, r3854;
}
{
add.f16x2 r3860, r3702, r3703;
}
{
mul.f16x2 r3863, r3860, r3677;
}
{
add.f16x2 r3866, r3857, r3863;
}
{
sub.f16x2 r3869, r3684, r3685;
}
{
mul.f16x2 r3872, r3869, r3676;
}
{
sub.f16x2 r3875, r3690, r3691;
}
{
mul.f16x2 r3878, r3875, r3678;
}
{
add.f16x2 r3881, r3872, r3878;
}
{
add.f16x2 r3884, r3866, r3881;
}
{
add.f16x2 r3887, r3696, r3697;
}
{
mul.f16x2 r3890, r3887, r3675;
}
{
add.f16x2 r3893, r3699, r3890;
}
{
add.f16x2 r3896, r3702, r3703;
}
{
mul.f16x2 r3899, r3896, r3677;
}
{
add.f16x2 r3902, r3893, r3899;
}
{
sub.f16x2 r3905, r3684, r3685;
}
{
mul.f16x2 r3908, r3905, r3676;
}
{
sub.f16x2 r3911, r3690, r3691;
}
{
mul.f16x2 r3914, r3911, r3678;
}
{
add.f16x2 r3917, r3908, r3914;
}
{
sub.f16x2 r3920, r3902, r3917;
}
{
add.f16x2 r3923, r3696, r3697;
}
{
mul.f16x2 r3926, r3923, r3677;
}
{
add.f16x2 r3929, r3699, r3926;
}
{
add.f16x2 r3932, r3702, r3703;
}
{
mul.f16x2 r3935, r3932, r3679;
}
{
add.f16x2 r3938, r3929, r3935;
}
{
sub.f16x2 r3941, r3684, r3685;
}
{
mul.f16x2 r3944, r3941, r3678;
}
{
sub.f16x2 r3947, r3690, r3691;
}
{
mul.f16x2 r3950, r3947, r3681;
}
{
add.f16x2 r3953, r3944, r3950;
}
{
add.f16x2 r3956, r3938, r3953;
}
{
add.f16x2 r3959, r3696, r3697;
}
{
mul.f16x2 r3962, r3959, r3677;
}
{
add.f16x2 r3965, r3699, r3962;
}
{
add.f16x2 r3968, r3702, r3703;
}
{
mul.f16x2 r3971, r3968, r3679;
}
{
add.f16x2 r3974, r3965, r3971;
}
{
sub.f16x2 r3977, r3684, r3685;
}
{
mul.f16x2 r3980, r3977, r3678;
}
{
sub.f16x2 r3983, r3690, r3691;
}
{
mul.f16x2 r3986, r3983, r3681;
}
{
add.f16x2 r3989, r3980, r3986;
}
{
sub.f16x2 r3992, r3974, r3989;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r3995, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r3996, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r3997, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r3998, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r3999, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r4000, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r4001, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r4002, {low, high};
}
{
mul.f16x2 r4013, r3740, r3995;
}
{
mul.f16x2 r4016, r3884, r3996;
}
{
sub.f16x2 r4019, r4013, r4016;
}
{
mul.f16x2 r4022, r3740, r3996;
}
{
fma.rn.f16x2 r4025, r3884, r3995, r4022;
}
{
mul.f16x2 r4029, r3812, r3997;
}
{
mul.f16x2 r4032, r3956, r3998;
}
{
sub.f16x2 r4035, r4029, r4032;
}
{
mul.f16x2 r4038, r3812, r3998;
}
{
fma.rn.f16x2 r4041, r3956, r3997, r4038;
}
{
mul.f16x2 r4045, r3848, r3999;
}
{
mul.f16x2 r4048, r3992, r4000;
}
{
sub.f16x2 r4051, r4045, r4048;
}
{
mul.f16x2 r4054, r3848, r4000;
}
{
fma.rn.f16x2 r4057, r3992, r3999, r4054;
}
{
mul.f16x2 r4061, r3776, r4001;
}
{
mul.f16x2 r4064, r3920, r4002;
}
{
sub.f16x2 r4067, r4061, r4064;
}
{
mul.f16x2 r4070, r3776, r4002;
}
{
fma.rn.f16x2 r4073, r3920, r4001, r4070;
}
{
add.f16x2 %0, r3372, r3692;
}
{
add.f16x2 %1, r3384, r3704;
}
{
sub.f16x2 %10, r3372, r3692;
}
{
sub.f16x2 %11, r3384, r3704;
}
{
add.f16x2 %2, r3420, r4019;
}
{
add.f16x2 %3, r3564, r4025;
}
{
sub.f16x2 %12, r3420, r4019;
}
{
sub.f16x2 %13, r3564, r4025;
}
{
add.f16x2 %4, r3492, r4035;
}
{
add.f16x2 %5, r3636, r4041;
}
{
sub.f16x2 %14, r3492, r4035;
}
{
sub.f16x2 %15, r3636, r4041;
}
{
add.f16x2 %6, r3528, r4051;
}
{
add.f16x2 %7, r3672, r4057;
}
{
sub.f16x2 %16, r3528, r4051;
}
{
sub.f16x2 %17, r3672, r4057;
}
{
add.f16x2 %8, r3456, r4067;
}
{
add.f16x2 %9, r3600, r4073;
}
{
sub.f16x2 %18, r3456, r4067;
}
{
sub.f16x2 %19, r3600, r4073;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<942, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<316>;
.reg .b32 r<4159>;
.reg .b64 rd<8>;
mov.u32 r4137, %tid.y;
mov.u32 r4138, %20;
mad.lo.s32 r4139, r4137, 40000, r4138;
mov.u32 r4140, %tid.x;
mov.f32 f276, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r1, {low, high};
}
mov.f32 f282, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r2, {low, high};
}
mov.f32 f284, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r3, {low, high};
}
mov.f32 f286, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r4, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r5, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
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
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r321, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r322, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r323, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r324, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r325, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
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
mov.f32 f272, 0f3F4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r641, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r642, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r643, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r644, {low, high};
}
mov.f32 f280, 0fBE9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r645, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r646, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r647, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r648, {low, high};
}
mov.f32 f243, 0fBF800000;
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
mul.wide.u32 rd2, r4140, 274877907;
shr.u64 rd3, rd2, 38;
cvt.u32.u64 r4141, rd3;
mul.lo.s32 r4142, r4141, 1000;
sub.s32 r4143, r4140, r4142;
mad.lo.s32 r4144, r4141, 40000, r4139;
cvt.rn.f32.u32 f307, r4143;
mul.f32 f308, f307, 0f3A24B5BE;
cos.approx.f32 f61, f308;
sin.approx.f32 f309, f308;
neg.f32 f62, f309;
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
mov.f32 f244, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
mad.lo.s32 r4145, r4143, 40, r4144;
st.shared.v2.f32 [r4145], {r723, r795};
st.shared.v2.f32 [r4145+8], {r832, r869};
st.shared.v2.f32 [r4145+16], {r906, r943};
st.shared.v2.f32 [r4145+24], {r980, r1017};
st.shared.v2.f32 [r4145+32], {r1054, r1091};
barrier.sync 0;
mad.lo.s32 r4146, r4143, -36, r4145;
ld.shared.u32 r1131, [r4146];
ld.shared.u32 r1451, [r4146+4000];
ld.shared.u32 r1128, [r4146+8000];
ld.shared.u32 r1448, [r4146+12000];
ld.shared.u32 r1134, [r4146+16000];
ld.shared.u32 r1454, [r4146+20000];
ld.shared.u32 r1135, [r4146+24000];
ld.shared.u32 r1455, [r4146+28000];
ld.shared.u32 r1129, [r4146+32000];
ld.shared.u32 r1449, [r4146+36000];
barrier.sync 0;
st.shared.v2.f32 [r4145], {r726, r802};
st.shared.v2.f32 [r4145+8], {r839, r876};
st.shared.v2.f32 [r4145+16], {r913, r950};
st.shared.v2.f32 [r4145+24], {r987, r1024};
st.shared.v2.f32 [r4145+32], {r1061, r1098};
barrier.sync 0;
ld.shared.u32 r1143, [r4146];
ld.shared.u32 r1463, [r4146+4000];
ld.shared.u32 r1140, [r4146+8000];
ld.shared.u32 r1460, [r4146+12000];
ld.shared.u32 r1146, [r4146+16000];
ld.shared.u32 r1466, [r4146+20000];
ld.shared.u32 r1147, [r4146+24000];
ld.shared.u32 r1467, [r4146+28000];
ld.shared.u32 r1141, [r4146+32000];
ld.shared.u32 r1461, [r4146+36000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r1119, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r1120, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r1121, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r1122, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r1123, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
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
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r1439, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r1440, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r1441, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r1442, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r1443, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
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
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r1759, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r1760, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r1761, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r1762, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r1763, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r1764, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r1765, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
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
mul.wide.u32 rd4, r4143, -858993459;
shr.u64 rd5, rd4, 35;
cvt.u32.u64 r4147, rd5;
mul.lo.s32 r4148, r4147, 10;
sub.s32 r4149, r4143, r4148;
shl.b32 r4150, r4149, 2;
add.s32 r4151, r4144, r4150;
cvt.rn.f32.u32 f310, r4147;
mul.f32 f311, f310, 0f3BCDE32E;
cos.approx.f32 f143, f311;
sin.approx.f32 f312, f311;
neg.f32 f144, f312;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
mad.lo.s32 r4152, r4147, 400, r4151;
st.shared.u32 [r4152], r1841;
st.shared.u32 [r4152+40], r1913;
st.shared.u32 [r4152+80], r1950;
st.shared.u32 [r4152+120], r1987;
st.shared.u32 [r4152+160], r2024;
st.shared.u32 [r4152+200], r2061;
st.shared.u32 [r4152+240], r2098;
st.shared.u32 [r4152+280], r2135;
st.shared.u32 [r4152+320], r2172;
st.shared.u32 [r4152+360], r2209;
barrier.sync 0;
ld.shared.u32 r2249, [r4146];
ld.shared.u32 r2569, [r4146+4000];
ld.shared.u32 r2246, [r4146+8000];
ld.shared.u32 r2566, [r4146+12000];
ld.shared.u32 r2252, [r4146+16000];
ld.shared.u32 r2572, [r4146+20000];
ld.shared.u32 r2253, [r4146+24000];
ld.shared.u32 r2573, [r4146+28000];
ld.shared.u32 r2247, [r4146+32000];
ld.shared.u32 r2567, [r4146+36000];
barrier.sync 0;
st.shared.u32 [r4152], r1844;
st.shared.u32 [r4152+40], r1920;
st.shared.u32 [r4152+80], r1957;
st.shared.u32 [r4152+120], r1994;
st.shared.u32 [r4152+160], r2031;
st.shared.u32 [r4152+200], r2068;
st.shared.u32 [r4152+240], r2105;
st.shared.u32 [r4152+280], r2142;
st.shared.u32 [r4152+320], r2179;
st.shared.u32 [r4152+360], r2216;
barrier.sync 0;
ld.shared.u32 r2261, [r4146];
ld.shared.u32 r2581, [r4146+4000];
ld.shared.u32 r2258, [r4146+8000];
ld.shared.u32 r2578, [r4146+12000];
ld.shared.u32 r2264, [r4146+16000];
ld.shared.u32 r2584, [r4146+20000];
ld.shared.u32 r2265, [r4146+24000];
ld.shared.u32 r2585, [r4146+28000];
ld.shared.u32 r2259, [r4146+32000];
ld.shared.u32 r2579, [r4146+36000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r2237, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r2238, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r2239, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r2240, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r2241, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
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
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r2557, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r2558, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r2559, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r2560, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r2561, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
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
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r2877, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r2878, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r2879, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r2880, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r2881, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r2882, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r2883, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
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
add.f16x2 r2959, r2254, r2574;
}
{
add.f16x2 r2962, r2266, r2586;
}
{
sub.f16x2 r2965, r2254, r2574;
}
{
sub.f16x2 r2968, r2266, r2586;
}
{
add.f16x2 r2971, r2302, r2901;
}
{
add.f16x2 r2974, r2446, r2907;
}
{
sub.f16x2 r2977, r2302, r2901;
}
{
sub.f16x2 r2980, r2446, r2907;
}
{
add.f16x2 r2983, r2374, r2917;
}
{
add.f16x2 r2986, r2518, r2923;
}
{
sub.f16x2 r2989, r2374, r2917;
}
{
sub.f16x2 r2992, r2518, r2923;
}
{
add.f16x2 r2995, r2410, r2933;
}
{
add.f16x2 r2998, r2554, r2939;
}
{
sub.f16x2 r3001, r2410, r2933;
}
{
sub.f16x2 r3004, r2554, r2939;
}
{
add.f16x2 r3007, r2338, r2949;
}
{
add.f16x2 r3010, r2482, r2955;
}
{
sub.f16x2 r3013, r2338, r2949;
}
{
sub.f16x2 r3016, r2482, r2955;
}
mul.wide.u32 rd6, r4143, 1374389535;
shr.u64 rd7, rd6, 37;
cvt.u32.u64 r4153, rd7;
mul.lo.s32 r4154, r4153, 100;
sub.s32 r4155, r4143, r4154;
shl.b32 r4156, r4155, 2;
add.s32 r4157, r4144, r4156;
cvt.rn.f32.u32 f313, r4153;
mul.f32 f314, f313, 0f3D80ADFD;
cos.approx.f32 f225, f314;
sin.approx.f32 f315, f314;
neg.f32 f226, f315;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f225;
cvt.rn.f16.f32 high, f226;
mov.b32 r3019, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3022, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3024, {high, high};
}
{
mul.f16x2 r3026, r2974, r3024;
}
{
neg.f16x2 r3029, r3026;
}
{
fma.rn.f16x2 r3031, r2971, r3022, r3029;
}
{
mul.f16x2 r3035, r2971, r3024;
}
{
fma.rn.f16x2 r3038, r2974, r3022, r3035;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3042, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3044, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
mov.b32 r3046, {low, high};
}
{
mul.f16x2 r3047, r3044, r3046;
}
{
mul.f16x2 r3050, r3019, r3042;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
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
mul.f16x2 r3063, r2986, r3061;
}
{
neg.f16x2 r3066, r3063;
}
{
fma.rn.f16x2 r3068, r2983, r3059, r3066;
}
{
mul.f16x2 r3072, r2983, r3061;
}
{
fma.rn.f16x2 r3075, r2986, r3059, r3072;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3079, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3081, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
mul.f16x2 r3100, r2998, r3098;
}
{
neg.f16x2 r3103, r3100;
}
{
fma.rn.f16x2 r3105, r2995, r3096, r3103;
}
{
mul.f16x2 r3109, r2995, r3098;
}
{
fma.rn.f16x2 r3112, r2998, r3096, r3109;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3116, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3118, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
mul.f16x2 r3137, r3010, r3135;
}
{
neg.f16x2 r3140, r3137;
}
{
fma.rn.f16x2 r3142, r3007, r3133, r3140;
}
{
mul.f16x2 r3146, r3007, r3135;
}
{
fma.rn.f16x2 r3149, r3010, r3133, r3146;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3153, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3155, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
mul.f16x2 r3174, r2968, r3172;
}
{
neg.f16x2 r3177, r3174;
}
{
fma.rn.f16x2 r3179, r2965, r3170, r3177;
}
{
mul.f16x2 r3183, r2965, r3172;
}
{
fma.rn.f16x2 r3186, r2968, r3170, r3183;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3190, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3192, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
mul.f16x2 r3211, r2980, r3209;
}
{
neg.f16x2 r3214, r3211;
}
{
fma.rn.f16x2 r3216, r2977, r3207, r3214;
}
{
mul.f16x2 r3220, r2977, r3209;
}
{
fma.rn.f16x2 r3223, r2980, r3207, r3220;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3227, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3229, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
mul.f16x2 r3248, r2992, r3246;
}
{
neg.f16x2 r3251, r3248;
}
{
fma.rn.f16x2 r3253, r2989, r3244, r3251;
}
{
mul.f16x2 r3257, r2989, r3246;
}
{
fma.rn.f16x2 r3260, r2992, r3244, r3257;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3264, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3266, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
mul.f16x2 r3285, r3004, r3283;
}
{
neg.f16x2 r3288, r3285;
}
{
fma.rn.f16x2 r3290, r3001, r3281, r3288;
}
{
mul.f16x2 r3294, r3001, r3283;
}
{
fma.rn.f16x2 r3297, r3004, r3281, r3294;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3301, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3019;
mov.b32 r3303, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f243;
cvt.rn.f16.f32 high, f244;
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
mul.f16x2 r3322, r3016, r3320;
}
{
neg.f16x2 r3325, r3322;
}
{
fma.rn.f16x2 r3327, r3013, r3318, r3325;
}
{
mul.f16x2 r3331, r3013, r3320;
}
{
fma.rn.f16x2 r3334, r3016, r3318, r3331;
}
barrier.sync 0;
mad.lo.s32 r4158, r4153, 4000, r4157;
st.shared.u32 [r4158], r2959;
st.shared.u32 [r4158+400], r3031;
st.shared.u32 [r4158+800], r3068;
st.shared.u32 [r4158+1200], r3105;
st.shared.u32 [r4158+1600], r3142;
st.shared.u32 [r4158+2000], r3179;
st.shared.u32 [r4158+2400], r3216;
st.shared.u32 [r4158+2800], r3253;
st.shared.u32 [r4158+3200], r3290;
st.shared.u32 [r4158+3600], r3327;
barrier.sync 0;
ld.shared.u32 r3367, [r4146];
ld.shared.u32 r3687, [r4146+4000];
ld.shared.u32 r3364, [r4146+8000];
ld.shared.u32 r3684, [r4146+12000];
ld.shared.u32 r3370, [r4146+16000];
ld.shared.u32 r3690, [r4146+20000];
ld.shared.u32 r3371, [r4146+24000];
ld.shared.u32 r3691, [r4146+28000];
ld.shared.u32 r3365, [r4146+32000];
ld.shared.u32 r3685, [r4146+36000];
barrier.sync 0;
st.shared.u32 [r4158], r2962;
st.shared.u32 [r4158+400], r3038;
st.shared.u32 [r4158+800], r3075;
st.shared.u32 [r4158+1200], r3112;
st.shared.u32 [r4158+1600], r3149;
st.shared.u32 [r4158+2000], r3186;
st.shared.u32 [r4158+2400], r3223;
st.shared.u32 [r4158+2800], r3260;
st.shared.u32 [r4158+3200], r3297;
st.shared.u32 [r4158+3600], r3334;
barrier.sync 0;
ld.shared.u32 r3379, [r4146];
ld.shared.u32 r3699, [r4146+4000];
ld.shared.u32 r3376, [r4146+8000];
ld.shared.u32 r3696, [r4146+12000];
ld.shared.u32 r3382, [r4146+16000];
ld.shared.u32 r3702, [r4146+20000];
ld.shared.u32 r3383, [r4146+24000];
ld.shared.u32 r3703, [r4146+28000];
ld.shared.u32 r3377, [r4146+32000];
ld.shared.u32 r3697, [r4146+36000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r3355, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r3356, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r3357, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r3358, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r3359, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r3360, {low, high};
}
{
neg.f16x2 r3361, r3360;
}
{
add.f16x2 r3363, r3364, r3365;
}
{
add.f16x2 r3366, r3367, r3363;
}
{
add.f16x2 r3369, r3370, r3371;
}
{
add.f16x2 r3372, r3366, r3369;
}
{
add.f16x2 r3375, r3376, r3377;
}
{
add.f16x2 r3378, r3379, r3375;
}
{
add.f16x2 r3381, r3382, r3383;
}
{
add.f16x2 r3384, r3378, r3381;
}
{
add.f16x2 r3387, r3364, r3365;
}
{
mul.f16x2 r3390, r3387, r3355;
}
{
add.f16x2 r3393, r3367, r3390;
}
{
add.f16x2 r3396, r3370, r3371;
}
{
mul.f16x2 r3399, r3396, r3357;
}
{
add.f16x2 r3402, r3393, r3399;
}
{
sub.f16x2 r3405, r3376, r3377;
}
{
mul.f16x2 r3408, r3405, r3356;
}
{
sub.f16x2 r3411, r3382, r3383;
}
{
mul.f16x2 r3414, r3411, r3358;
}
{
add.f16x2 r3417, r3408, r3414;
}
{
sub.f16x2 r3420, r3402, r3417;
}
{
add.f16x2 r3423, r3364, r3365;
}
{
mul.f16x2 r3426, r3423, r3355;
}
{
add.f16x2 r3429, r3367, r3426;
}
{
add.f16x2 r3432, r3370, r3371;
}
{
mul.f16x2 r3435, r3432, r3357;
}
{
add.f16x2 r3438, r3429, r3435;
}
{
sub.f16x2 r3441, r3376, r3377;
}
{
mul.f16x2 r3444, r3441, r3356;
}
{
sub.f16x2 r3447, r3382, r3383;
}
{
mul.f16x2 r3450, r3447, r3358;
}
{
add.f16x2 r3453, r3444, r3450;
}
{
add.f16x2 r3456, r3438, r3453;
}
{
add.f16x2 r3459, r3364, r3365;
}
{
mul.f16x2 r3462, r3459, r3357;
}
{
add.f16x2 r3465, r3367, r3462;
}
{
add.f16x2 r3468, r3370, r3371;
}
{
mul.f16x2 r3471, r3468, r3359;
}
{
add.f16x2 r3474, r3465, r3471;
}
{
sub.f16x2 r3477, r3376, r3377;
}
{
mul.f16x2 r3480, r3477, r3358;
}
{
sub.f16x2 r3483, r3382, r3383;
}
{
mul.f16x2 r3486, r3483, r3361;
}
{
add.f16x2 r3489, r3480, r3486;
}
{
sub.f16x2 r3492, r3474, r3489;
}
{
add.f16x2 r3495, r3364, r3365;
}
{
mul.f16x2 r3498, r3495, r3357;
}
{
add.f16x2 r3501, r3367, r3498;
}
{
add.f16x2 r3504, r3370, r3371;
}
{
mul.f16x2 r3507, r3504, r3359;
}
{
add.f16x2 r3510, r3501, r3507;
}
{
sub.f16x2 r3513, r3376, r3377;
}
{
mul.f16x2 r3516, r3513, r3358;
}
{
sub.f16x2 r3519, r3382, r3383;
}
{
mul.f16x2 r3522, r3519, r3361;
}
{
add.f16x2 r3525, r3516, r3522;
}
{
add.f16x2 r3528, r3510, r3525;
}
{
add.f16x2 r3531, r3376, r3377;
}
{
mul.f16x2 r3534, r3531, r3355;
}
{
add.f16x2 r3537, r3379, r3534;
}
{
add.f16x2 r3540, r3382, r3383;
}
{
mul.f16x2 r3543, r3540, r3357;
}
{
add.f16x2 r3546, r3537, r3543;
}
{
sub.f16x2 r3549, r3364, r3365;
}
{
mul.f16x2 r3552, r3549, r3356;
}
{
sub.f16x2 r3555, r3370, r3371;
}
{
mul.f16x2 r3558, r3555, r3358;
}
{
add.f16x2 r3561, r3552, r3558;
}
{
add.f16x2 r3564, r3546, r3561;
}
{
add.f16x2 r3567, r3376, r3377;
}
{
mul.f16x2 r3570, r3567, r3355;
}
{
add.f16x2 r3573, r3379, r3570;
}
{
add.f16x2 r3576, r3382, r3383;
}
{
mul.f16x2 r3579, r3576, r3357;
}
{
add.f16x2 r3582, r3573, r3579;
}
{
sub.f16x2 r3585, r3364, r3365;
}
{
mul.f16x2 r3588, r3585, r3356;
}
{
sub.f16x2 r3591, r3370, r3371;
}
{
mul.f16x2 r3594, r3591, r3358;
}
{
add.f16x2 r3597, r3588, r3594;
}
{
sub.f16x2 r3600, r3582, r3597;
}
{
add.f16x2 r3603, r3376, r3377;
}
{
mul.f16x2 r3606, r3603, r3357;
}
{
add.f16x2 r3609, r3379, r3606;
}
{
add.f16x2 r3612, r3382, r3383;
}
{
mul.f16x2 r3615, r3612, r3359;
}
{
add.f16x2 r3618, r3609, r3615;
}
{
sub.f16x2 r3621, r3364, r3365;
}
{
mul.f16x2 r3624, r3621, r3358;
}
{
sub.f16x2 r3627, r3370, r3371;
}
{
mul.f16x2 r3630, r3627, r3361;
}
{
add.f16x2 r3633, r3624, r3630;
}
{
add.f16x2 r3636, r3618, r3633;
}
{
add.f16x2 r3639, r3376, r3377;
}
{
mul.f16x2 r3642, r3639, r3357;
}
{
add.f16x2 r3645, r3379, r3642;
}
{
add.f16x2 r3648, r3382, r3383;
}
{
mul.f16x2 r3651, r3648, r3359;
}
{
add.f16x2 r3654, r3645, r3651;
}
{
sub.f16x2 r3657, r3364, r3365;
}
{
mul.f16x2 r3660, r3657, r3358;
}
{
sub.f16x2 r3663, r3370, r3371;
}
{
mul.f16x2 r3666, r3663, r3361;
}
{
add.f16x2 r3669, r3660, r3666;
}
{
sub.f16x2 r3672, r3654, r3669;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r3675, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r3676, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r3677, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r3678, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r3679, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r3680, {low, high};
}
{
neg.f16x2 r3681, r3680;
}
{
add.f16x2 r3683, r3684, r3685;
}
{
add.f16x2 r3686, r3687, r3683;
}
{
add.f16x2 r3689, r3690, r3691;
}
{
add.f16x2 r3692, r3686, r3689;
}
{
add.f16x2 r3695, r3696, r3697;
}
{
add.f16x2 r3698, r3699, r3695;
}
{
add.f16x2 r3701, r3702, r3703;
}
{
add.f16x2 r3704, r3698, r3701;
}
{
add.f16x2 r3707, r3684, r3685;
}
{
mul.f16x2 r3710, r3707, r3675;
}
{
add.f16x2 r3713, r3687, r3710;
}
{
add.f16x2 r3716, r3690, r3691;
}
{
mul.f16x2 r3719, r3716, r3677;
}
{
add.f16x2 r3722, r3713, r3719;
}
{
sub.f16x2 r3725, r3696, r3697;
}
{
mul.f16x2 r3728, r3725, r3676;
}
{
sub.f16x2 r3731, r3702, r3703;
}
{
mul.f16x2 r3734, r3731, r3678;
}
{
add.f16x2 r3737, r3728, r3734;
}
{
sub.f16x2 r3740, r3722, r3737;
}
{
add.f16x2 r3743, r3684, r3685;
}
{
mul.f16x2 r3746, r3743, r3675;
}
{
add.f16x2 r3749, r3687, r3746;
}
{
add.f16x2 r3752, r3690, r3691;
}
{
mul.f16x2 r3755, r3752, r3677;
}
{
add.f16x2 r3758, r3749, r3755;
}
{
sub.f16x2 r3761, r3696, r3697;
}
{
mul.f16x2 r3764, r3761, r3676;
}
{
sub.f16x2 r3767, r3702, r3703;
}
{
mul.f16x2 r3770, r3767, r3678;
}
{
add.f16x2 r3773, r3764, r3770;
}
{
add.f16x2 r3776, r3758, r3773;
}
{
add.f16x2 r3779, r3684, r3685;
}
{
mul.f16x2 r3782, r3779, r3677;
}
{
add.f16x2 r3785, r3687, r3782;
}
{
add.f16x2 r3788, r3690, r3691;
}
{
mul.f16x2 r3791, r3788, r3679;
}
{
add.f16x2 r3794, r3785, r3791;
}
{
sub.f16x2 r3797, r3696, r3697;
}
{
mul.f16x2 r3800, r3797, r3678;
}
{
sub.f16x2 r3803, r3702, r3703;
}
{
mul.f16x2 r3806, r3803, r3681;
}
{
add.f16x2 r3809, r3800, r3806;
}
{
sub.f16x2 r3812, r3794, r3809;
}
{
add.f16x2 r3815, r3684, r3685;
}
{
mul.f16x2 r3818, r3815, r3677;
}
{
add.f16x2 r3821, r3687, r3818;
}
{
add.f16x2 r3824, r3690, r3691;
}
{
mul.f16x2 r3827, r3824, r3679;
}
{
add.f16x2 r3830, r3821, r3827;
}
{
sub.f16x2 r3833, r3696, r3697;
}
{
mul.f16x2 r3836, r3833, r3678;
}
{
sub.f16x2 r3839, r3702, r3703;
}
{
mul.f16x2 r3842, r3839, r3681;
}
{
add.f16x2 r3845, r3836, r3842;
}
{
add.f16x2 r3848, r3830, r3845;
}
{
add.f16x2 r3851, r3696, r3697;
}
{
mul.f16x2 r3854, r3851, r3675;
}
{
add.f16x2 r3857, r3699, r3854;
}
{
add.f16x2 r3860, r3702, r3703;
}
{
mul.f16x2 r3863, r3860, r3677;
}
{
add.f16x2 r3866, r3857, r3863;
}
{
sub.f16x2 r3869, r3684, r3685;
}
{
mul.f16x2 r3872, r3869, r3676;
}
{
sub.f16x2 r3875, r3690, r3691;
}
{
mul.f16x2 r3878, r3875, r3678;
}
{
add.f16x2 r3881, r3872, r3878;
}
{
add.f16x2 r3884, r3866, r3881;
}
{
add.f16x2 r3887, r3696, r3697;
}
{
mul.f16x2 r3890, r3887, r3675;
}
{
add.f16x2 r3893, r3699, r3890;
}
{
add.f16x2 r3896, r3702, r3703;
}
{
mul.f16x2 r3899, r3896, r3677;
}
{
add.f16x2 r3902, r3893, r3899;
}
{
sub.f16x2 r3905, r3684, r3685;
}
{
mul.f16x2 r3908, r3905, r3676;
}
{
sub.f16x2 r3911, r3690, r3691;
}
{
mul.f16x2 r3914, r3911, r3678;
}
{
add.f16x2 r3917, r3908, r3914;
}
{
sub.f16x2 r3920, r3902, r3917;
}
{
add.f16x2 r3923, r3696, r3697;
}
{
mul.f16x2 r3926, r3923, r3677;
}
{
add.f16x2 r3929, r3699, r3926;
}
{
add.f16x2 r3932, r3702, r3703;
}
{
mul.f16x2 r3935, r3932, r3679;
}
{
add.f16x2 r3938, r3929, r3935;
}
{
sub.f16x2 r3941, r3684, r3685;
}
{
mul.f16x2 r3944, r3941, r3678;
}
{
sub.f16x2 r3947, r3690, r3691;
}
{
mul.f16x2 r3950, r3947, r3681;
}
{
add.f16x2 r3953, r3944, r3950;
}
{
add.f16x2 r3956, r3938, r3953;
}
{
add.f16x2 r3959, r3696, r3697;
}
{
mul.f16x2 r3962, r3959, r3677;
}
{
add.f16x2 r3965, r3699, r3962;
}
{
add.f16x2 r3968, r3702, r3703;
}
{
mul.f16x2 r3971, r3968, r3679;
}
{
add.f16x2 r3974, r3965, r3971;
}
{
sub.f16x2 r3977, r3684, r3685;
}
{
mul.f16x2 r3980, r3977, r3678;
}
{
sub.f16x2 r3983, r3690, r3691;
}
{
mul.f16x2 r3986, r3983, r3681;
}
{
add.f16x2 r3989, r3980, r3986;
}
{
sub.f16x2 r3992, r3974, r3989;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f272;
cvt.rn.f16.f32 high, f272;
mov.b32 r3995, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r3996, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f276;
cvt.rn.f16.f32 high, f276;
mov.b32 r3997, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r3998, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f280;
cvt.rn.f16.f32 high, f280;
mov.b32 r3999, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f282;
cvt.rn.f16.f32 high, f282;
mov.b32 r4000, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f284;
cvt.rn.f16.f32 high, f284;
mov.b32 r4001, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f286;
cvt.rn.f16.f32 high, f286;
mov.b32 r4002, {low, high};
}
{
mul.f16x2 r4013, r3740, r3995;
}
{
mul.f16x2 r4016, r3884, r3996;
}
{
sub.f16x2 r4019, r4013, r4016;
}
{
mul.f16x2 r4022, r3740, r3996;
}
{
fma.rn.f16x2 r4025, r3884, r3995, r4022;
}
{
mul.f16x2 r4029, r3812, r3997;
}
{
mul.f16x2 r4032, r3956, r3998;
}
{
sub.f16x2 r4035, r4029, r4032;
}
{
mul.f16x2 r4038, r3812, r3998;
}
{
fma.rn.f16x2 r4041, r3956, r3997, r4038;
}
{
mul.f16x2 r4045, r3848, r3999;
}
{
mul.f16x2 r4048, r3992, r4000;
}
{
sub.f16x2 r4051, r4045, r4048;
}
{
mul.f16x2 r4054, r3848, r4000;
}
{
fma.rn.f16x2 r4057, r3992, r3999, r4054;
}
{
mul.f16x2 r4061, r3776, r4001;
}
{
mul.f16x2 r4064, r3920, r4002;
}
{
sub.f16x2 r4067, r4061, r4064;
}
{
mul.f16x2 r4070, r3776, r4002;
}
{
fma.rn.f16x2 r4073, r3920, r4001, r4070;
}
{
add.f16x2 %0, r3372, r3692;
}
{
add.f16x2 %1, r3384, r3704;
}
{
sub.f16x2 %10, r3372, r3692;
}
{
sub.f16x2 %11, r3384, r3704;
}
{
add.f16x2 %2, r3420, r4019;
}
{
add.f16x2 %3, r3564, r4025;
}
{
sub.f16x2 %12, r3420, r4019;
}
{
sub.f16x2 %13, r3564, r4025;
}
{
add.f16x2 %4, r3492, r4035;
}
{
add.f16x2 %5, r3636, r4041;
}
{
sub.f16x2 %14, r3492, r4035;
}
{
sub.f16x2 %15, r3636, r4041;
}
{
add.f16x2 %6, r3528, r4051;
}
{
add.f16x2 %7, r3672, r4057;
}
{
sub.f16x2 %16, r3528, r4051;
}
{
sub.f16x2 %17, r3672, r4057;
}
{
add.f16x2 %8, r3456, r4067;
}
{
add.f16x2 %9, r3600, r4073;
}
{
sub.f16x2 %18, r3456, r4067;
}
{
sub.f16x2 %19, r3600, r4073;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)));
};


#endif
