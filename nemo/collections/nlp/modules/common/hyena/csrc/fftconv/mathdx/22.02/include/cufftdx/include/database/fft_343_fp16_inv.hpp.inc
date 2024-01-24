#ifndef CUFFTDX_FFT_343_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_343_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1122, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<99>;
.reg .b32 r<2571>;
.reg .b64 rd<6>;
mov.u32 r2551, %tid.y;
mov.u32 r2552, %14;
mad.lo.s32 r2553, r2551, 2744, r2552;
mov.u32 r2554, %tid.x;
mov.f32 f90, 0f3F1F9D07;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r1, {low, high};
}
mov.f32 f92, 0fBF48261C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r2, {low, high};
}
{
neg.f16x2 r3, r2;
}
mov.f32 f78, 0fBE63DC87;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f78;
cvt.rn.f16.f32 high, f78;
mov.b32 r5, {low, high};
}
mov.f32 f80, 0fBF7994E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
mov.f32 f86, 0fBF66A5E5;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r9, {low, high};
}
mov.f32 f88, 0fBEDE2602;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r10, {low, high};
}
{
neg.f16x2 r11, r10;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r13, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r14, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r15, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r16, {low, high};
}
{
add.f16x2 r17, %17, %27;
}
{
add.f16x2 r20, %15, r17;
}
{
add.f16x2 r23, %19, %25;
}
{
add.f16x2 r26, r20, r23;
}
{
add.f16x2 r29, %21, %23;
}
{
add.f16x2 r32, r26, r29;
}
{
add.f16x2 r35, %18, %28;
}
{
add.f16x2 r38, %16, r35;
}
{
add.f16x2 r41, %20, %26;
}
{
add.f16x2 r44, r38, r41;
}
{
add.f16x2 r47, %22, %24;
}
{
add.f16x2 r50, r44, r47;
}
{
add.f16x2 r53, %17, %27;
}
{
mul.f16x2 r56, r53, r1;
}
{
add.f16x2 r59, %15, r56;
}
{
add.f16x2 r62, %19, %25;
}
{
mul.f16x2 r65, r62, r5;
}
{
add.f16x2 r68, r59, r65;
}
{
add.f16x2 r71, %21, %23;
}
{
mul.f16x2 r74, r71, r9;
}
{
add.f16x2 r77, r68, r74;
}
{
sub.f16x2 r80, %18, %28;
}
{
mul.f16x2 r83, r80, r3;
}
{
sub.f16x2 r86, %20, %26;
}
{
mul.f16x2 r89, r86, r7;
}
{
add.f16x2 r92, r83, r89;
}
{
sub.f16x2 r95, %22, %24;
}
{
mul.f16x2 r98, r95, r11;
}
{
add.f16x2 r101, r92, r98;
}
{
sub.f16x2 r104, r77, r101;
}
{
add.f16x2 r107, %17, %27;
}
{
mul.f16x2 r110, r107, r1;
}
{
add.f16x2 r113, %15, r110;
}
{
add.f16x2 r116, %19, %25;
}
{
mul.f16x2 r119, r116, r5;
}
{
add.f16x2 r122, r113, r119;
}
{
add.f16x2 r125, %21, %23;
}
{
mul.f16x2 r128, r125, r9;
}
{
add.f16x2 r131, r122, r128;
}
{
sub.f16x2 r134, %18, %28;
}
{
mul.f16x2 r137, r134, r3;
}
{
sub.f16x2 r140, %20, %26;
}
{
mul.f16x2 r143, r140, r7;
}
{
add.f16x2 r146, r137, r143;
}
{
sub.f16x2 r149, %22, %24;
}
{
mul.f16x2 r152, r149, r11;
}
{
add.f16x2 r155, r146, r152;
}
{
add.f16x2 r158, r131, r155;
}
{
add.f16x2 r161, %17, %27;
}
{
mul.f16x2 r164, r161, r5;
}
{
add.f16x2 r167, %15, r164;
}
{
add.f16x2 r170, %19, %25;
}
{
mul.f16x2 r173, r170, r13;
}
{
add.f16x2 r176, r167, r173;
}
{
add.f16x2 r179, %21, %23;
}
{
mul.f16x2 r182, r179, r15;
}
{
add.f16x2 r185, r176, r182;
}
{
sub.f16x2 r188, %18, %28;
}
{
mul.f16x2 r191, r188, r7;
}
{
sub.f16x2 r194, %20, %26;
}
{
mul.f16x2 r197, r194, r14;
}
{
add.f16x2 r200, r191, r197;
}
{
sub.f16x2 r203, %22, %24;
}
{
mul.f16x2 r206, r203, r16;
}
{
add.f16x2 r209, r200, r206;
}
{
sub.f16x2 r212, r185, r209;
}
{
add.f16x2 r215, %17, %27;
}
{
mul.f16x2 r218, r215, r5;
}
{
add.f16x2 r221, %15, r218;
}
{
add.f16x2 r224, %19, %25;
}
{
mul.f16x2 r227, r224, r13;
}
{
add.f16x2 r230, r221, r227;
}
{
add.f16x2 r233, %21, %23;
}
{
mul.f16x2 r236, r233, r15;
}
{
add.f16x2 r239, r230, r236;
}
{
sub.f16x2 r242, %18, %28;
}
{
mul.f16x2 r245, r242, r7;
}
{
sub.f16x2 r248, %20, %26;
}
{
mul.f16x2 r251, r248, r14;
}
{
add.f16x2 r254, r245, r251;
}
{
sub.f16x2 r257, %22, %24;
}
{
mul.f16x2 r260, r257, r16;
}
{
add.f16x2 r263, r254, r260;
}
{
add.f16x2 r266, r239, r263;
}
{
add.f16x2 r269, %17, %27;
}
{
mul.f16x2 r272, r269, r9;
}
{
add.f16x2 r275, %15, r272;
}
{
add.f16x2 r278, %19, %25;
}
{
mul.f16x2 r281, r278, r15;
}
{
add.f16x2 r284, r275, r281;
}
{
add.f16x2 r287, %21, %23;
}
{
mul.f16x2 r290, r287, r5;
}
{
add.f16x2 r293, r284, r290;
}
{
sub.f16x2 r296, %18, %28;
}
{
mul.f16x2 r299, r296, r11;
}
{
sub.f16x2 r302, %20, %26;
}
{
mul.f16x2 r305, r302, r16;
}
{
add.f16x2 r308, r299, r305;
}
{
sub.f16x2 r311, %22, %24;
}
{
mul.f16x2 r314, r311, r7;
}
{
add.f16x2 r317, r308, r314;
}
{
sub.f16x2 r320, r293, r317;
}
{
add.f16x2 r323, %17, %27;
}
{
mul.f16x2 r326, r323, r9;
}
{
add.f16x2 r329, %15, r326;
}
{
add.f16x2 r332, %19, %25;
}
{
mul.f16x2 r335, r332, r15;
}
{
add.f16x2 r338, r329, r335;
}
{
add.f16x2 r341, %21, %23;
}
{
mul.f16x2 r344, r341, r5;
}
{
add.f16x2 r347, r338, r344;
}
{
sub.f16x2 r350, %18, %28;
}
{
mul.f16x2 r353, r350, r11;
}
{
sub.f16x2 r356, %20, %26;
}
{
mul.f16x2 r359, r356, r16;
}
{
add.f16x2 r362, r353, r359;
}
{
sub.f16x2 r365, %22, %24;
}
{
mul.f16x2 r368, r365, r7;
}
{
add.f16x2 r371, r362, r368;
}
{
add.f16x2 r374, r347, r371;
}
{
add.f16x2 r377, %18, %28;
}
{
mul.f16x2 r380, r377, r1;
}
{
add.f16x2 r383, %16, r380;
}
{
add.f16x2 r386, %20, %26;
}
{
mul.f16x2 r389, r386, r5;
}
{
add.f16x2 r392, r383, r389;
}
{
add.f16x2 r395, %22, %24;
}
{
mul.f16x2 r398, r395, r9;
}
{
add.f16x2 r401, r392, r398;
}
{
sub.f16x2 r404, %17, %27;
}
{
mul.f16x2 r407, r404, r3;
}
{
sub.f16x2 r410, %19, %25;
}
{
mul.f16x2 r413, r410, r7;
}
{
add.f16x2 r416, r407, r413;
}
{
sub.f16x2 r419, %21, %23;
}
{
mul.f16x2 r422, r419, r11;
}
{
add.f16x2 r425, r416, r422;
}
{
add.f16x2 r428, r401, r425;
}
{
add.f16x2 r431, %18, %28;
}
{
mul.f16x2 r434, r431, r1;
}
{
add.f16x2 r437, %16, r434;
}
{
add.f16x2 r440, %20, %26;
}
{
mul.f16x2 r443, r440, r5;
}
{
add.f16x2 r446, r437, r443;
}
{
add.f16x2 r449, %22, %24;
}
{
mul.f16x2 r452, r449, r9;
}
{
add.f16x2 r455, r446, r452;
}
{
sub.f16x2 r458, %17, %27;
}
{
mul.f16x2 r461, r458, r3;
}
{
sub.f16x2 r464, %19, %25;
}
{
mul.f16x2 r467, r464, r7;
}
{
add.f16x2 r470, r461, r467;
}
{
sub.f16x2 r473, %21, %23;
}
{
mul.f16x2 r476, r473, r11;
}
{
add.f16x2 r479, r470, r476;
}
{
sub.f16x2 r482, r455, r479;
}
{
add.f16x2 r485, %18, %28;
}
{
mul.f16x2 r488, r485, r5;
}
{
add.f16x2 r491, %16, r488;
}
{
add.f16x2 r494, %20, %26;
}
{
mul.f16x2 r497, r494, r13;
}
{
add.f16x2 r500, r491, r497;
}
{
add.f16x2 r503, %22, %24;
}
{
mul.f16x2 r506, r503, r15;
}
{
add.f16x2 r509, r500, r506;
}
{
sub.f16x2 r512, %17, %27;
}
{
mul.f16x2 r515, r512, r7;
}
{
sub.f16x2 r518, %19, %25;
}
{
mul.f16x2 r521, r518, r14;
}
{
add.f16x2 r524, r515, r521;
}
{
sub.f16x2 r527, %21, %23;
}
{
mul.f16x2 r530, r527, r16;
}
{
add.f16x2 r533, r524, r530;
}
{
add.f16x2 r536, r509, r533;
}
{
add.f16x2 r539, %18, %28;
}
{
mul.f16x2 r542, r539, r5;
}
{
add.f16x2 r545, %16, r542;
}
{
add.f16x2 r548, %20, %26;
}
{
mul.f16x2 r551, r548, r13;
}
{
add.f16x2 r554, r545, r551;
}
{
add.f16x2 r557, %22, %24;
}
{
mul.f16x2 r560, r557, r15;
}
{
add.f16x2 r563, r554, r560;
}
{
sub.f16x2 r566, %17, %27;
}
{
mul.f16x2 r569, r566, r7;
}
{
sub.f16x2 r572, %19, %25;
}
{
mul.f16x2 r575, r572, r14;
}
{
add.f16x2 r578, r569, r575;
}
{
sub.f16x2 r581, %21, %23;
}
{
mul.f16x2 r584, r581, r16;
}
{
add.f16x2 r587, r578, r584;
}
{
sub.f16x2 r590, r563, r587;
}
{
add.f16x2 r593, %18, %28;
}
{
mul.f16x2 r596, r593, r9;
}
{
add.f16x2 r599, %16, r596;
}
{
add.f16x2 r602, %20, %26;
}
{
mul.f16x2 r605, r602, r15;
}
{
add.f16x2 r608, r599, r605;
}
{
add.f16x2 r611, %22, %24;
}
{
mul.f16x2 r614, r611, r5;
}
{
add.f16x2 r617, r608, r614;
}
{
sub.f16x2 r620, %17, %27;
}
{
mul.f16x2 r623, r620, r11;
}
{
sub.f16x2 r626, %19, %25;
}
{
mul.f16x2 r629, r626, r16;
}
{
add.f16x2 r632, r623, r629;
}
{
sub.f16x2 r635, %21, %23;
}
{
mul.f16x2 r638, r635, r7;
}
{
add.f16x2 r641, r632, r638;
}
{
add.f16x2 r644, r617, r641;
}
{
add.f16x2 r647, %18, %28;
}
{
mul.f16x2 r650, r647, r9;
}
{
add.f16x2 r653, %16, r650;
}
{
add.f16x2 r656, %20, %26;
}
{
mul.f16x2 r659, r656, r15;
}
{
add.f16x2 r662, r653, r659;
}
{
add.f16x2 r665, %22, %24;
}
{
mul.f16x2 r668, r665, r5;
}
{
add.f16x2 r671, r662, r668;
}
{
sub.f16x2 r674, %17, %27;
}
{
mul.f16x2 r677, r674, r11;
}
{
sub.f16x2 r680, %19, %25;
}
{
mul.f16x2 r683, r680, r16;
}
{
add.f16x2 r686, r677, r683;
}
{
sub.f16x2 r689, %21, %23;
}
{
mul.f16x2 r692, r689, r7;
}
{
add.f16x2 r695, r686, r692;
}
{
sub.f16x2 r698, r671, r695;
}
mul.wide.u32 rd2, r2554, 1402438301;
shr.u64 rd3, rd2, 36;
cvt.u32.u64 r2555, rd3;
mul.lo.s32 r2556, r2555, 49;
sub.s32 r2557, r2554, r2556;
cvt.rn.f32.u32 f93, r2557;
mul.f32 f94, f93, 0f3C961050;
cos.approx.f32 f21, f94;
sin.approx.f32 f95, f94;
neg.f32 f22, f95;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f21;
cvt.rn.f16.f32 high, f22;
mov.b32 r701, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r704, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r706, {high, high};
}
{
mul.f16x2 r708, r428, r706;
}
{
fma.rn.f16x2 r711, r104, r704, r708;
}
{
mul.f16x2 r715, r104, r706;
}
{
neg.f16x2 r718, r715;
}
{
fma.rn.f16x2 r720, r428, r704, r718;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r724, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r726, {high, high};
}
mov.f32 f69, 0fBF800000;
mov.f32 f70, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r728, {low, high};
}
{
mul.f16x2 r729, r726, r728;
}
{
mul.f16x2 r732, r701, r724;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r735, {high, low};
}
{
fma.rn.f16x2 r737, r729, r735, r732;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r737;
mov.b32 r741, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r737;
mov.b32 r743, {high, high};
}
{
mul.f16x2 r745, r536, r743;
}
{
fma.rn.f16x2 r748, r212, r741, r745;
}
{
mul.f16x2 r752, r212, r743;
}
{
neg.f16x2 r755, r752;
}
{
fma.rn.f16x2 r757, r536, r741, r755;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r761, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r763, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r765, {low, high};
}
{
mul.f16x2 r766, r763, r765;
}
{
mul.f16x2 r769, r737, r761;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r737;
mov.b32 r772, {high, low};
}
{
fma.rn.f16x2 r774, r766, r772, r769;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r774;
mov.b32 r778, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r774;
mov.b32 r780, {high, high};
}
{
mul.f16x2 r782, r644, r780;
}
{
fma.rn.f16x2 r785, r320, r778, r782;
}
{
mul.f16x2 r789, r320, r780;
}
{
neg.f16x2 r792, r789;
}
{
fma.rn.f16x2 r794, r644, r778, r792;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r798, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r800, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r802, {low, high};
}
{
mul.f16x2 r803, r800, r802;
}
{
mul.f16x2 r806, r774, r798;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r774;
mov.b32 r809, {high, low};
}
{
fma.rn.f16x2 r811, r803, r809, r806;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r811;
mov.b32 r815, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r811;
mov.b32 r817, {high, high};
}
{
mul.f16x2 r819, r698, r817;
}
{
fma.rn.f16x2 r822, r374, r815, r819;
}
{
mul.f16x2 r826, r374, r817;
}
{
neg.f16x2 r829, r826;
}
{
fma.rn.f16x2 r831, r698, r815, r829;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r835, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r837, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r839, {low, high};
}
{
mul.f16x2 r840, r837, r839;
}
{
mul.f16x2 r843, r811, r835;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r811;
mov.b32 r846, {high, low};
}
{
fma.rn.f16x2 r848, r840, r846, r843;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r848;
mov.b32 r852, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r848;
mov.b32 r854, {high, high};
}
{
mul.f16x2 r856, r590, r854;
}
{
fma.rn.f16x2 r859, r266, r852, r856;
}
{
mul.f16x2 r863, r266, r854;
}
{
neg.f16x2 r866, r863;
}
{
fma.rn.f16x2 r868, r590, r852, r866;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r872, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r874, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r876, {low, high};
}
{
mul.f16x2 r877, r874, r876;
}
{
mul.f16x2 r880, r848, r872;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r848;
mov.b32 r883, {high, low};
}
{
fma.rn.f16x2 r885, r877, r883, r880;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r885;
mov.b32 r889, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r885;
mov.b32 r891, {high, high};
}
{
mul.f16x2 r893, r482, r891;
}
{
fma.rn.f16x2 r896, r158, r889, r893;
}
{
mul.f16x2 r900, r158, r891;
}
{
neg.f16x2 r903, r900;
}
{
fma.rn.f16x2 r905, r482, r889, r903;
}
mad.lo.s32 r2558, r2555, 2744, r2553;
barrier.sync 0;
mad.lo.s32 r2559, r2557, 56, r2558;
st.shared.v2.f32 [r2559], {r32, r50};
st.shared.v2.f32 [r2559+8], {r711, r720};
st.shared.v2.f32 [r2559+16], {r748, r757};
st.shared.v2.f32 [r2559+24], {r785, r794};
st.shared.v2.f32 [r2559+32], {r822, r831};
st.shared.v2.f32 [r2559+40], {r859, r868};
st.shared.v2.f32 [r2559+48], {r896, r905};
barrier.sync 0;
mad.lo.s32 r2560, r2557, -48, r2559;
ld.shared.u32 r946, [r2560];
ld.shared.u32 r964, [r2560+4];
ld.shared.u32 r943, [r2560+392];
ld.shared.u32 r961, [r2560+396];
ld.shared.u32 r949, [r2560+784];
ld.shared.u32 r967, [r2560+788];
ld.shared.u32 r955, [r2560+1176];
ld.shared.u32 r973, [r2560+1180];
ld.shared.u32 r956, [r2560+1568];
ld.shared.u32 r974, [r2560+1572];
ld.shared.u32 r950, [r2560+1960];
ld.shared.u32 r968, [r2560+1964];
ld.shared.u32 r944, [r2560+2352];
ld.shared.u32 r962, [r2560+2356];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r926, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r927, {low, high};
}
{
neg.f16x2 r928, r927;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f78;
cvt.rn.f16.f32 high, f78;
mov.b32 r930, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r931, {low, high};
}
{
neg.f16x2 r932, r931;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r934, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r935, {low, high};
}
{
neg.f16x2 r936, r935;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r938, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r939, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r940, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r941, {low, high};
}
{
add.f16x2 r942, r943, r944;
}
{
add.f16x2 r945, r946, r942;
}
{
add.f16x2 r948, r949, r950;
}
{
add.f16x2 r951, r945, r948;
}
{
add.f16x2 r954, r955, r956;
}
{
add.f16x2 r957, r951, r954;
}
{
add.f16x2 r960, r961, r962;
}
{
add.f16x2 r963, r964, r960;
}
{
add.f16x2 r966, r967, r968;
}
{
add.f16x2 r969, r963, r966;
}
{
add.f16x2 r972, r973, r974;
}
{
add.f16x2 r975, r969, r972;
}
{
add.f16x2 r978, r943, r944;
}
{
mul.f16x2 r981, r978, r926;
}
{
add.f16x2 r984, r946, r981;
}
{
add.f16x2 r987, r949, r950;
}
{
mul.f16x2 r990, r987, r930;
}
{
add.f16x2 r993, r984, r990;
}
{
add.f16x2 r996, r955, r956;
}
{
mul.f16x2 r999, r996, r934;
}
{
add.f16x2 r1002, r993, r999;
}
{
sub.f16x2 r1005, r961, r962;
}
{
mul.f16x2 r1008, r1005, r928;
}
{
sub.f16x2 r1011, r967, r968;
}
{
mul.f16x2 r1014, r1011, r932;
}
{
add.f16x2 r1017, r1008, r1014;
}
{
sub.f16x2 r1020, r973, r974;
}
{
mul.f16x2 r1023, r1020, r936;
}
{
add.f16x2 r1026, r1017, r1023;
}
{
sub.f16x2 r1029, r1002, r1026;
}
{
add.f16x2 r1032, r943, r944;
}
{
mul.f16x2 r1035, r1032, r926;
}
{
add.f16x2 r1038, r946, r1035;
}
{
add.f16x2 r1041, r949, r950;
}
{
mul.f16x2 r1044, r1041, r930;
}
{
add.f16x2 r1047, r1038, r1044;
}
{
add.f16x2 r1050, r955, r956;
}
{
mul.f16x2 r1053, r1050, r934;
}
{
add.f16x2 r1056, r1047, r1053;
}
{
sub.f16x2 r1059, r961, r962;
}
{
mul.f16x2 r1062, r1059, r928;
}
{
sub.f16x2 r1065, r967, r968;
}
{
mul.f16x2 r1068, r1065, r932;
}
{
add.f16x2 r1071, r1062, r1068;
}
{
sub.f16x2 r1074, r973, r974;
}
{
mul.f16x2 r1077, r1074, r936;
}
{
add.f16x2 r1080, r1071, r1077;
}
{
add.f16x2 r1083, r1056, r1080;
}
{
add.f16x2 r1086, r943, r944;
}
{
mul.f16x2 r1089, r1086, r930;
}
{
add.f16x2 r1092, r946, r1089;
}
{
add.f16x2 r1095, r949, r950;
}
{
mul.f16x2 r1098, r1095, r938;
}
{
add.f16x2 r1101, r1092, r1098;
}
{
add.f16x2 r1104, r955, r956;
}
{
mul.f16x2 r1107, r1104, r940;
}
{
add.f16x2 r1110, r1101, r1107;
}
{
sub.f16x2 r1113, r961, r962;
}
{
mul.f16x2 r1116, r1113, r932;
}
{
sub.f16x2 r1119, r967, r968;
}
{
mul.f16x2 r1122, r1119, r939;
}
{
add.f16x2 r1125, r1116, r1122;
}
{
sub.f16x2 r1128, r973, r974;
}
{
mul.f16x2 r1131, r1128, r941;
}
{
add.f16x2 r1134, r1125, r1131;
}
{
sub.f16x2 r1137, r1110, r1134;
}
{
add.f16x2 r1140, r943, r944;
}
{
mul.f16x2 r1143, r1140, r930;
}
{
add.f16x2 r1146, r946, r1143;
}
{
add.f16x2 r1149, r949, r950;
}
{
mul.f16x2 r1152, r1149, r938;
}
{
add.f16x2 r1155, r1146, r1152;
}
{
add.f16x2 r1158, r955, r956;
}
{
mul.f16x2 r1161, r1158, r940;
}
{
add.f16x2 r1164, r1155, r1161;
}
{
sub.f16x2 r1167, r961, r962;
}
{
mul.f16x2 r1170, r1167, r932;
}
{
sub.f16x2 r1173, r967, r968;
}
{
mul.f16x2 r1176, r1173, r939;
}
{
add.f16x2 r1179, r1170, r1176;
}
{
sub.f16x2 r1182, r973, r974;
}
{
mul.f16x2 r1185, r1182, r941;
}
{
add.f16x2 r1188, r1179, r1185;
}
{
add.f16x2 r1191, r1164, r1188;
}
{
add.f16x2 r1194, r943, r944;
}
{
mul.f16x2 r1197, r1194, r934;
}
{
add.f16x2 r1200, r946, r1197;
}
{
add.f16x2 r1203, r949, r950;
}
{
mul.f16x2 r1206, r1203, r940;
}
{
add.f16x2 r1209, r1200, r1206;
}
{
add.f16x2 r1212, r955, r956;
}
{
mul.f16x2 r1215, r1212, r930;
}
{
add.f16x2 r1218, r1209, r1215;
}
{
sub.f16x2 r1221, r961, r962;
}
{
mul.f16x2 r1224, r1221, r936;
}
{
sub.f16x2 r1227, r967, r968;
}
{
mul.f16x2 r1230, r1227, r941;
}
{
add.f16x2 r1233, r1224, r1230;
}
{
sub.f16x2 r1236, r973, r974;
}
{
mul.f16x2 r1239, r1236, r932;
}
{
add.f16x2 r1242, r1233, r1239;
}
{
sub.f16x2 r1245, r1218, r1242;
}
{
add.f16x2 r1248, r943, r944;
}
{
mul.f16x2 r1251, r1248, r934;
}
{
add.f16x2 r1254, r946, r1251;
}
{
add.f16x2 r1257, r949, r950;
}
{
mul.f16x2 r1260, r1257, r940;
}
{
add.f16x2 r1263, r1254, r1260;
}
{
add.f16x2 r1266, r955, r956;
}
{
mul.f16x2 r1269, r1266, r930;
}
{
add.f16x2 r1272, r1263, r1269;
}
{
sub.f16x2 r1275, r961, r962;
}
{
mul.f16x2 r1278, r1275, r936;
}
{
sub.f16x2 r1281, r967, r968;
}
{
mul.f16x2 r1284, r1281, r941;
}
{
add.f16x2 r1287, r1278, r1284;
}
{
sub.f16x2 r1290, r973, r974;
}
{
mul.f16x2 r1293, r1290, r932;
}
{
add.f16x2 r1296, r1287, r1293;
}
{
add.f16x2 r1299, r1272, r1296;
}
{
add.f16x2 r1302, r961, r962;
}
{
mul.f16x2 r1305, r1302, r926;
}
{
add.f16x2 r1308, r964, r1305;
}
{
add.f16x2 r1311, r967, r968;
}
{
mul.f16x2 r1314, r1311, r930;
}
{
add.f16x2 r1317, r1308, r1314;
}
{
add.f16x2 r1320, r973, r974;
}
{
mul.f16x2 r1323, r1320, r934;
}
{
add.f16x2 r1326, r1317, r1323;
}
{
sub.f16x2 r1329, r943, r944;
}
{
mul.f16x2 r1332, r1329, r928;
}
{
sub.f16x2 r1335, r949, r950;
}
{
mul.f16x2 r1338, r1335, r932;
}
{
add.f16x2 r1341, r1332, r1338;
}
{
sub.f16x2 r1344, r955, r956;
}
{
mul.f16x2 r1347, r1344, r936;
}
{
add.f16x2 r1350, r1341, r1347;
}
{
add.f16x2 r1353, r1326, r1350;
}
{
add.f16x2 r1356, r961, r962;
}
{
mul.f16x2 r1359, r1356, r926;
}
{
add.f16x2 r1362, r964, r1359;
}
{
add.f16x2 r1365, r967, r968;
}
{
mul.f16x2 r1368, r1365, r930;
}
{
add.f16x2 r1371, r1362, r1368;
}
{
add.f16x2 r1374, r973, r974;
}
{
mul.f16x2 r1377, r1374, r934;
}
{
add.f16x2 r1380, r1371, r1377;
}
{
sub.f16x2 r1383, r943, r944;
}
{
mul.f16x2 r1386, r1383, r928;
}
{
sub.f16x2 r1389, r949, r950;
}
{
mul.f16x2 r1392, r1389, r932;
}
{
add.f16x2 r1395, r1386, r1392;
}
{
sub.f16x2 r1398, r955, r956;
}
{
mul.f16x2 r1401, r1398, r936;
}
{
add.f16x2 r1404, r1395, r1401;
}
{
sub.f16x2 r1407, r1380, r1404;
}
{
add.f16x2 r1410, r961, r962;
}
{
mul.f16x2 r1413, r1410, r930;
}
{
add.f16x2 r1416, r964, r1413;
}
{
add.f16x2 r1419, r967, r968;
}
{
mul.f16x2 r1422, r1419, r938;
}
{
add.f16x2 r1425, r1416, r1422;
}
{
add.f16x2 r1428, r973, r974;
}
{
mul.f16x2 r1431, r1428, r940;
}
{
add.f16x2 r1434, r1425, r1431;
}
{
sub.f16x2 r1437, r943, r944;
}
{
mul.f16x2 r1440, r1437, r932;
}
{
sub.f16x2 r1443, r949, r950;
}
{
mul.f16x2 r1446, r1443, r939;
}
{
add.f16x2 r1449, r1440, r1446;
}
{
sub.f16x2 r1452, r955, r956;
}
{
mul.f16x2 r1455, r1452, r941;
}
{
add.f16x2 r1458, r1449, r1455;
}
{
add.f16x2 r1461, r1434, r1458;
}
{
add.f16x2 r1464, r961, r962;
}
{
mul.f16x2 r1467, r1464, r930;
}
{
add.f16x2 r1470, r964, r1467;
}
{
add.f16x2 r1473, r967, r968;
}
{
mul.f16x2 r1476, r1473, r938;
}
{
add.f16x2 r1479, r1470, r1476;
}
{
add.f16x2 r1482, r973, r974;
}
{
mul.f16x2 r1485, r1482, r940;
}
{
add.f16x2 r1488, r1479, r1485;
}
{
sub.f16x2 r1491, r943, r944;
}
{
mul.f16x2 r1494, r1491, r932;
}
{
sub.f16x2 r1497, r949, r950;
}
{
mul.f16x2 r1500, r1497, r939;
}
{
add.f16x2 r1503, r1494, r1500;
}
{
sub.f16x2 r1506, r955, r956;
}
{
mul.f16x2 r1509, r1506, r941;
}
{
add.f16x2 r1512, r1503, r1509;
}
{
sub.f16x2 r1515, r1488, r1512;
}
{
add.f16x2 r1518, r961, r962;
}
{
mul.f16x2 r1521, r1518, r934;
}
{
add.f16x2 r1524, r964, r1521;
}
{
add.f16x2 r1527, r967, r968;
}
{
mul.f16x2 r1530, r1527, r940;
}
{
add.f16x2 r1533, r1524, r1530;
}
{
add.f16x2 r1536, r973, r974;
}
{
mul.f16x2 r1539, r1536, r930;
}
{
add.f16x2 r1542, r1533, r1539;
}
{
sub.f16x2 r1545, r943, r944;
}
{
mul.f16x2 r1548, r1545, r936;
}
{
sub.f16x2 r1551, r949, r950;
}
{
mul.f16x2 r1554, r1551, r941;
}
{
add.f16x2 r1557, r1548, r1554;
}
{
sub.f16x2 r1560, r955, r956;
}
{
mul.f16x2 r1563, r1560, r932;
}
{
add.f16x2 r1566, r1557, r1563;
}
{
add.f16x2 r1569, r1542, r1566;
}
{
add.f16x2 r1572, r961, r962;
}
{
mul.f16x2 r1575, r1572, r934;
}
{
add.f16x2 r1578, r964, r1575;
}
{
add.f16x2 r1581, r967, r968;
}
{
mul.f16x2 r1584, r1581, r940;
}
{
add.f16x2 r1587, r1578, r1584;
}
{
add.f16x2 r1590, r973, r974;
}
{
mul.f16x2 r1593, r1590, r930;
}
{
add.f16x2 r1596, r1587, r1593;
}
{
sub.f16x2 r1599, r943, r944;
}
{
mul.f16x2 r1602, r1599, r936;
}
{
sub.f16x2 r1605, r949, r950;
}
{
mul.f16x2 r1608, r1605, r941;
}
{
add.f16x2 r1611, r1602, r1608;
}
{
sub.f16x2 r1614, r955, r956;
}
{
mul.f16x2 r1617, r1614, r932;
}
{
add.f16x2 r1620, r1611, r1617;
}
{
sub.f16x2 r1623, r1596, r1620;
}
mul.wide.u32 rd4, r2557, 613566757;
shr.u64 rd5, rd4, 32;
cvt.u32.u64 r2561, rd5;
sub.s32 r2562, r2557, r2561;
shr.u32 r2563, r2562, 1;
add.s32 r2564, r2563, r2561;
shr.u32 r2565, r2564, 2;
cvt.rn.f32.u32 f96, r2565;
mul.f32 f97, f96, 0f3E034E46;
cos.approx.f32 f57, f97;
sin.approx.f32 f98, f97;
neg.f32 f58, f98;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f57;
cvt.rn.f16.f32 high, f58;
mov.b32 r1626, {low, high};
}
mul.lo.s32 r2566, r2565, 7;
sub.s32 r2567, r2557, r2566;
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1629, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1631, {high, high};
}
{
mul.f16x2 r1633, r1353, r1631;
}
{
fma.rn.f16x2 r1636, r1029, r1629, r1633;
}
{
mul.f16x2 r1640, r1029, r1631;
}
{
neg.f16x2 r1643, r1640;
}
{
fma.rn.f16x2 r1645, r1353, r1629, r1643;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1649, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1651, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r1653, {low, high};
}
{
mul.f16x2 r1654, r1651, r1653;
}
{
mul.f16x2 r1657, r1626, r1649;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1660, {high, low};
}
{
fma.rn.f16x2 r1662, r1654, r1660, r1657;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1662;
mov.b32 r1666, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1662;
mov.b32 r1668, {high, high};
}
{
mul.f16x2 r1670, r1461, r1668;
}
{
fma.rn.f16x2 r1673, r1137, r1666, r1670;
}
{
mul.f16x2 r1677, r1137, r1668;
}
{
neg.f16x2 r1680, r1677;
}
{
fma.rn.f16x2 r1682, r1461, r1666, r1680;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1686, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1688, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r1690, {low, high};
}
{
mul.f16x2 r1691, r1688, r1690;
}
{
mul.f16x2 r1694, r1662, r1686;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1662;
mov.b32 r1697, {high, low};
}
{
fma.rn.f16x2 r1699, r1691, r1697, r1694;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1699;
mov.b32 r1703, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1699;
mov.b32 r1705, {high, high};
}
{
mul.f16x2 r1707, r1569, r1705;
}
{
fma.rn.f16x2 r1710, r1245, r1703, r1707;
}
{
mul.f16x2 r1714, r1245, r1705;
}
{
neg.f16x2 r1717, r1714;
}
{
fma.rn.f16x2 r1719, r1569, r1703, r1717;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1723, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1725, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r1727, {low, high};
}
{
mul.f16x2 r1728, r1725, r1727;
}
{
mul.f16x2 r1731, r1699, r1723;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1699;
mov.b32 r1734, {high, low};
}
{
fma.rn.f16x2 r1736, r1728, r1734, r1731;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1736;
mov.b32 r1740, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1736;
mov.b32 r1742, {high, high};
}
{
mul.f16x2 r1744, r1623, r1742;
}
{
fma.rn.f16x2 r1747, r1299, r1740, r1744;
}
{
mul.f16x2 r1751, r1299, r1742;
}
{
neg.f16x2 r1754, r1751;
}
{
fma.rn.f16x2 r1756, r1623, r1740, r1754;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1760, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1762, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r1764, {low, high};
}
{
mul.f16x2 r1765, r1762, r1764;
}
{
mul.f16x2 r1768, r1736, r1760;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1736;
mov.b32 r1771, {high, low};
}
{
fma.rn.f16x2 r1773, r1765, r1771, r1768;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1773;
mov.b32 r1777, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1773;
mov.b32 r1779, {high, high};
}
{
mul.f16x2 r1781, r1515, r1779;
}
{
fma.rn.f16x2 r1784, r1191, r1777, r1781;
}
{
mul.f16x2 r1788, r1191, r1779;
}
{
neg.f16x2 r1791, r1788;
}
{
fma.rn.f16x2 r1793, r1515, r1777, r1791;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1797, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1799, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r1801, {low, high};
}
{
mul.f16x2 r1802, r1799, r1801;
}
{
mul.f16x2 r1805, r1773, r1797;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1773;
mov.b32 r1808, {high, low};
}
{
fma.rn.f16x2 r1810, r1802, r1808, r1805;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1810;
mov.b32 r1814, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1810;
mov.b32 r1816, {high, high};
}
{
mul.f16x2 r1818, r1407, r1816;
}
{
fma.rn.f16x2 r1821, r1083, r1814, r1818;
}
{
mul.f16x2 r1825, r1083, r1816;
}
{
neg.f16x2 r1828, r1825;
}
{
fma.rn.f16x2 r1830, r1407, r1814, r1828;
}
shl.b32 r2568, r2567, 3;
add.s32 r2569, r2558, r2568;
barrier.sync 0;
mad.lo.s32 r2570, r2565, 392, r2569;
st.shared.u32 [r2570], r957;
st.shared.u32 [r2570+4], r975;
st.shared.u32 [r2570+56], r1636;
st.shared.u32 [r2570+60], r1645;
st.shared.u32 [r2570+112], r1673;
st.shared.u32 [r2570+116], r1682;
st.shared.u32 [r2570+168], r1710;
st.shared.u32 [r2570+172], r1719;
st.shared.u32 [r2570+224], r1747;
st.shared.u32 [r2570+228], r1756;
st.shared.u32 [r2570+280], r1784;
st.shared.u32 [r2570+284], r1793;
st.shared.u32 [r2570+336], r1821;
st.shared.u32 [r2570+340], r1830;
barrier.sync 0;
ld.shared.u32 r1871, [r2560];
ld.shared.u32 r1889, [r2560+4];
ld.shared.u32 r1868, [r2560+392];
ld.shared.u32 r1886, [r2560+396];
ld.shared.u32 r1874, [r2560+784];
ld.shared.u32 r1892, [r2560+788];
ld.shared.u32 r1880, [r2560+1176];
ld.shared.u32 r1898, [r2560+1180];
ld.shared.u32 r1881, [r2560+1568];
ld.shared.u32 r1899, [r2560+1572];
ld.shared.u32 r1875, [r2560+1960];
ld.shared.u32 r1893, [r2560+1964];
ld.shared.u32 r1869, [r2560+2352];
ld.shared.u32 r1887, [r2560+2356];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r1851, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r1852, {low, high};
}
{
neg.f16x2 r1853, r1852;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f78;
cvt.rn.f16.f32 high, f78;
mov.b32 r1855, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r1856, {low, high};
}
{
neg.f16x2 r1857, r1856;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r1859, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r1860, {low, high};
}
{
neg.f16x2 r1861, r1860;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r1863, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r1864, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r1865, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r1866, {low, high};
}
{
add.f16x2 r1867, r1868, r1869;
}
{
add.f16x2 r1870, r1871, r1867;
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
add.f16x2 %0, r1876, r1879;
}
{
add.f16x2 r1885, r1886, r1887;
}
{
add.f16x2 r1888, r1889, r1885;
}
{
add.f16x2 r1891, r1892, r1893;
}
{
add.f16x2 r1894, r1888, r1891;
}
{
add.f16x2 r1897, r1898, r1899;
}
{
add.f16x2 %1, r1894, r1897;
}
{
add.f16x2 r1903, r1868, r1869;
}
{
mul.f16x2 r1906, r1903, r1851;
}
{
add.f16x2 r1909, r1871, r1906;
}
{
add.f16x2 r1912, r1874, r1875;
}
{
mul.f16x2 r1915, r1912, r1855;
}
{
add.f16x2 r1918, r1909, r1915;
}
{
add.f16x2 r1921, r1880, r1881;
}
{
mul.f16x2 r1924, r1921, r1859;
}
{
add.f16x2 r1927, r1918, r1924;
}
{
sub.f16x2 r1930, r1886, r1887;
}
{
mul.f16x2 r1933, r1930, r1853;
}
{
sub.f16x2 r1936, r1892, r1893;
}
{
mul.f16x2 r1939, r1936, r1857;
}
{
add.f16x2 r1942, r1933, r1939;
}
{
sub.f16x2 r1945, r1898, r1899;
}
{
mul.f16x2 r1948, r1945, r1861;
}
{
add.f16x2 r1951, r1942, r1948;
}
{
sub.f16x2 %2, r1927, r1951;
}
{
add.f16x2 r1957, r1868, r1869;
}
{
mul.f16x2 r1960, r1957, r1851;
}
{
add.f16x2 r1963, r1871, r1960;
}
{
add.f16x2 r1966, r1874, r1875;
}
{
mul.f16x2 r1969, r1966, r1855;
}
{
add.f16x2 r1972, r1963, r1969;
}
{
add.f16x2 r1975, r1880, r1881;
}
{
mul.f16x2 r1978, r1975, r1859;
}
{
add.f16x2 r1981, r1972, r1978;
}
{
sub.f16x2 r1984, r1886, r1887;
}
{
mul.f16x2 r1987, r1984, r1853;
}
{
sub.f16x2 r1990, r1892, r1893;
}
{
mul.f16x2 r1993, r1990, r1857;
}
{
add.f16x2 r1996, r1987, r1993;
}
{
sub.f16x2 r1999, r1898, r1899;
}
{
mul.f16x2 r2002, r1999, r1861;
}
{
add.f16x2 r2005, r1996, r2002;
}
{
add.f16x2 %12, r1981, r2005;
}
{
add.f16x2 r2011, r1868, r1869;
}
{
mul.f16x2 r2014, r2011, r1855;
}
{
add.f16x2 r2017, r1871, r2014;
}
{
add.f16x2 r2020, r1874, r1875;
}
{
mul.f16x2 r2023, r2020, r1863;
}
{
add.f16x2 r2026, r2017, r2023;
}
{
add.f16x2 r2029, r1880, r1881;
}
{
mul.f16x2 r2032, r2029, r1865;
}
{
add.f16x2 r2035, r2026, r2032;
}
{
sub.f16x2 r2038, r1886, r1887;
}
{
mul.f16x2 r2041, r2038, r1857;
}
{
sub.f16x2 r2044, r1892, r1893;
}
{
mul.f16x2 r2047, r2044, r1864;
}
{
add.f16x2 r2050, r2041, r2047;
}
{
sub.f16x2 r2053, r1898, r1899;
}
{
mul.f16x2 r2056, r2053, r1866;
}
{
add.f16x2 r2059, r2050, r2056;
}
{
sub.f16x2 %4, r2035, r2059;
}
{
add.f16x2 r2065, r1868, r1869;
}
{
mul.f16x2 r2068, r2065, r1855;
}
{
add.f16x2 r2071, r1871, r2068;
}
{
add.f16x2 r2074, r1874, r1875;
}
{
mul.f16x2 r2077, r2074, r1863;
}
{
add.f16x2 r2080, r2071, r2077;
}
{
add.f16x2 r2083, r1880, r1881;
}
{
mul.f16x2 r2086, r2083, r1865;
}
{
add.f16x2 r2089, r2080, r2086;
}
{
sub.f16x2 r2092, r1886, r1887;
}
{
mul.f16x2 r2095, r2092, r1857;
}
{
sub.f16x2 r2098, r1892, r1893;
}
{
mul.f16x2 r2101, r2098, r1864;
}
{
add.f16x2 r2104, r2095, r2101;
}
{
sub.f16x2 r2107, r1898, r1899;
}
{
mul.f16x2 r2110, r2107, r1866;
}
{
add.f16x2 r2113, r2104, r2110;
}
{
add.f16x2 %10, r2089, r2113;
}
{
add.f16x2 r2119, r1868, r1869;
}
{
mul.f16x2 r2122, r2119, r1859;
}
{
add.f16x2 r2125, r1871, r2122;
}
{
add.f16x2 r2128, r1874, r1875;
}
{
mul.f16x2 r2131, r2128, r1865;
}
{
add.f16x2 r2134, r2125, r2131;
}
{
add.f16x2 r2137, r1880, r1881;
}
{
mul.f16x2 r2140, r2137, r1855;
}
{
add.f16x2 r2143, r2134, r2140;
}
{
sub.f16x2 r2146, r1886, r1887;
}
{
mul.f16x2 r2149, r2146, r1861;
}
{
sub.f16x2 r2152, r1892, r1893;
}
{
mul.f16x2 r2155, r2152, r1866;
}
{
add.f16x2 r2158, r2149, r2155;
}
{
sub.f16x2 r2161, r1898, r1899;
}
{
mul.f16x2 r2164, r2161, r1857;
}
{
add.f16x2 r2167, r2158, r2164;
}
{
sub.f16x2 %6, r2143, r2167;
}
{
add.f16x2 r2173, r1868, r1869;
}
{
mul.f16x2 r2176, r2173, r1859;
}
{
add.f16x2 r2179, r1871, r2176;
}
{
add.f16x2 r2182, r1874, r1875;
}
{
mul.f16x2 r2185, r2182, r1865;
}
{
add.f16x2 r2188, r2179, r2185;
}
{
add.f16x2 r2191, r1880, r1881;
}
{
mul.f16x2 r2194, r2191, r1855;
}
{
add.f16x2 r2197, r2188, r2194;
}
{
sub.f16x2 r2200, r1886, r1887;
}
{
mul.f16x2 r2203, r2200, r1861;
}
{
sub.f16x2 r2206, r1892, r1893;
}
{
mul.f16x2 r2209, r2206, r1866;
}
{
add.f16x2 r2212, r2203, r2209;
}
{
sub.f16x2 r2215, r1898, r1899;
}
{
mul.f16x2 r2218, r2215, r1857;
}
{
add.f16x2 r2221, r2212, r2218;
}
{
add.f16x2 %8, r2197, r2221;
}
{
add.f16x2 r2227, r1886, r1887;
}
{
mul.f16x2 r2230, r2227, r1851;
}
{
add.f16x2 r2233, r1889, r2230;
}
{
add.f16x2 r2236, r1892, r1893;
}
{
mul.f16x2 r2239, r2236, r1855;
}
{
add.f16x2 r2242, r2233, r2239;
}
{
add.f16x2 r2245, r1898, r1899;
}
{
mul.f16x2 r2248, r2245, r1859;
}
{
add.f16x2 r2251, r2242, r2248;
}
{
sub.f16x2 r2254, r1868, r1869;
}
{
mul.f16x2 r2257, r2254, r1853;
}
{
sub.f16x2 r2260, r1874, r1875;
}
{
mul.f16x2 r2263, r2260, r1857;
}
{
add.f16x2 r2266, r2257, r2263;
}
{
sub.f16x2 r2269, r1880, r1881;
}
{
mul.f16x2 r2272, r2269, r1861;
}
{
add.f16x2 r2275, r2266, r2272;
}
{
add.f16x2 %3, r2251, r2275;
}
{
add.f16x2 r2281, r1886, r1887;
}
{
mul.f16x2 r2284, r2281, r1851;
}
{
add.f16x2 r2287, r1889, r2284;
}
{
add.f16x2 r2290, r1892, r1893;
}
{
mul.f16x2 r2293, r2290, r1855;
}
{
add.f16x2 r2296, r2287, r2293;
}
{
add.f16x2 r2299, r1898, r1899;
}
{
mul.f16x2 r2302, r2299, r1859;
}
{
add.f16x2 r2305, r2296, r2302;
}
{
sub.f16x2 r2308, r1868, r1869;
}
{
mul.f16x2 r2311, r2308, r1853;
}
{
sub.f16x2 r2314, r1874, r1875;
}
{
mul.f16x2 r2317, r2314, r1857;
}
{
add.f16x2 r2320, r2311, r2317;
}
{
sub.f16x2 r2323, r1880, r1881;
}
{
mul.f16x2 r2326, r2323, r1861;
}
{
add.f16x2 r2329, r2320, r2326;
}
{
sub.f16x2 %13, r2305, r2329;
}
{
add.f16x2 r2335, r1886, r1887;
}
{
mul.f16x2 r2338, r2335, r1855;
}
{
add.f16x2 r2341, r1889, r2338;
}
{
add.f16x2 r2344, r1892, r1893;
}
{
mul.f16x2 r2347, r2344, r1863;
}
{
add.f16x2 r2350, r2341, r2347;
}
{
add.f16x2 r2353, r1898, r1899;
}
{
mul.f16x2 r2356, r2353, r1865;
}
{
add.f16x2 r2359, r2350, r2356;
}
{
sub.f16x2 r2362, r1868, r1869;
}
{
mul.f16x2 r2365, r2362, r1857;
}
{
sub.f16x2 r2368, r1874, r1875;
}
{
mul.f16x2 r2371, r2368, r1864;
}
{
add.f16x2 r2374, r2365, r2371;
}
{
sub.f16x2 r2377, r1880, r1881;
}
{
mul.f16x2 r2380, r2377, r1866;
}
{
add.f16x2 r2383, r2374, r2380;
}
{
add.f16x2 %5, r2359, r2383;
}
{
add.f16x2 r2389, r1886, r1887;
}
{
mul.f16x2 r2392, r2389, r1855;
}
{
add.f16x2 r2395, r1889, r2392;
}
{
add.f16x2 r2398, r1892, r1893;
}
{
mul.f16x2 r2401, r2398, r1863;
}
{
add.f16x2 r2404, r2395, r2401;
}
{
add.f16x2 r2407, r1898, r1899;
}
{
mul.f16x2 r2410, r2407, r1865;
}
{
add.f16x2 r2413, r2404, r2410;
}
{
sub.f16x2 r2416, r1868, r1869;
}
{
mul.f16x2 r2419, r2416, r1857;
}
{
sub.f16x2 r2422, r1874, r1875;
}
{
mul.f16x2 r2425, r2422, r1864;
}
{
add.f16x2 r2428, r2419, r2425;
}
{
sub.f16x2 r2431, r1880, r1881;
}
{
mul.f16x2 r2434, r2431, r1866;
}
{
add.f16x2 r2437, r2428, r2434;
}
{
sub.f16x2 %11, r2413, r2437;
}
{
add.f16x2 r2443, r1886, r1887;
}
{
mul.f16x2 r2446, r2443, r1859;
}
{
add.f16x2 r2449, r1889, r2446;
}
{
add.f16x2 r2452, r1892, r1893;
}
{
mul.f16x2 r2455, r2452, r1865;
}
{
add.f16x2 r2458, r2449, r2455;
}
{
add.f16x2 r2461, r1898, r1899;
}
{
mul.f16x2 r2464, r2461, r1855;
}
{
add.f16x2 r2467, r2458, r2464;
}
{
sub.f16x2 r2470, r1868, r1869;
}
{
mul.f16x2 r2473, r2470, r1861;
}
{
sub.f16x2 r2476, r1874, r1875;
}
{
mul.f16x2 r2479, r2476, r1866;
}
{
add.f16x2 r2482, r2473, r2479;
}
{
sub.f16x2 r2485, r1880, r1881;
}
{
mul.f16x2 r2488, r2485, r1857;
}
{
add.f16x2 r2491, r2482, r2488;
}
{
add.f16x2 %7, r2467, r2491;
}
{
add.f16x2 r2497, r1886, r1887;
}
{
mul.f16x2 r2500, r2497, r1859;
}
{
add.f16x2 r2503, r1889, r2500;
}
{
add.f16x2 r2506, r1892, r1893;
}
{
mul.f16x2 r2509, r2506, r1865;
}
{
add.f16x2 r2512, r2503, r2509;
}
{
add.f16x2 r2515, r1898, r1899;
}
{
mul.f16x2 r2518, r2515, r1855;
}
{
add.f16x2 r2521, r2512, r2518;
}
{
sub.f16x2 r2524, r1868, r1869;
}
{
mul.f16x2 r2527, r2524, r1861;
}
{
sub.f16x2 r2530, r1874, r1875;
}
{
mul.f16x2 r2533, r2530, r1866;
}
{
add.f16x2 r2536, r2527, r2533;
}
{
sub.f16x2 r2539, r1880, r1881;
}
{
mul.f16x2 r2542, r2539, r1857;
}
{
add.f16x2 r2545, r2536, r2542;
}
{
sub.f16x2 %9, r2521, r2545;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1123, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<99>;
.reg .b32 r<2571>;
.reg .b64 rd<6>;
mov.u32 r2551, %tid.y;
mov.u32 r2552, %14;
mad.lo.s32 r2553, r2551, 1372, r2552;
mov.u32 r2554, %tid.x;
mov.f32 f90, 0f3F1F9D07;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r1, {low, high};
}
mov.f32 f92, 0fBF48261C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r2, {low, high};
}
{
neg.f16x2 r3, r2;
}
mov.f32 f78, 0fBE63DC87;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f78;
cvt.rn.f16.f32 high, f78;
mov.b32 r5, {low, high};
}
mov.f32 f80, 0fBF7994E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
mov.f32 f86, 0fBF66A5E5;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r9, {low, high};
}
mov.f32 f88, 0fBEDE2602;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r10, {low, high};
}
{
neg.f16x2 r11, r10;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r13, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r14, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r15, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r16, {low, high};
}
{
add.f16x2 r17, %17, %27;
}
{
add.f16x2 r20, %15, r17;
}
{
add.f16x2 r23, %19, %25;
}
{
add.f16x2 r26, r20, r23;
}
{
add.f16x2 r29, %21, %23;
}
{
add.f16x2 r32, r26, r29;
}
{
add.f16x2 r35, %18, %28;
}
{
add.f16x2 r38, %16, r35;
}
{
add.f16x2 r41, %20, %26;
}
{
add.f16x2 r44, r38, r41;
}
{
add.f16x2 r47, %22, %24;
}
{
add.f16x2 r50, r44, r47;
}
{
add.f16x2 r53, %17, %27;
}
{
mul.f16x2 r56, r53, r1;
}
{
add.f16x2 r59, %15, r56;
}
{
add.f16x2 r62, %19, %25;
}
{
mul.f16x2 r65, r62, r5;
}
{
add.f16x2 r68, r59, r65;
}
{
add.f16x2 r71, %21, %23;
}
{
mul.f16x2 r74, r71, r9;
}
{
add.f16x2 r77, r68, r74;
}
{
sub.f16x2 r80, %18, %28;
}
{
mul.f16x2 r83, r80, r3;
}
{
sub.f16x2 r86, %20, %26;
}
{
mul.f16x2 r89, r86, r7;
}
{
add.f16x2 r92, r83, r89;
}
{
sub.f16x2 r95, %22, %24;
}
{
mul.f16x2 r98, r95, r11;
}
{
add.f16x2 r101, r92, r98;
}
{
sub.f16x2 r104, r77, r101;
}
{
add.f16x2 r107, %17, %27;
}
{
mul.f16x2 r110, r107, r1;
}
{
add.f16x2 r113, %15, r110;
}
{
add.f16x2 r116, %19, %25;
}
{
mul.f16x2 r119, r116, r5;
}
{
add.f16x2 r122, r113, r119;
}
{
add.f16x2 r125, %21, %23;
}
{
mul.f16x2 r128, r125, r9;
}
{
add.f16x2 r131, r122, r128;
}
{
sub.f16x2 r134, %18, %28;
}
{
mul.f16x2 r137, r134, r3;
}
{
sub.f16x2 r140, %20, %26;
}
{
mul.f16x2 r143, r140, r7;
}
{
add.f16x2 r146, r137, r143;
}
{
sub.f16x2 r149, %22, %24;
}
{
mul.f16x2 r152, r149, r11;
}
{
add.f16x2 r155, r146, r152;
}
{
add.f16x2 r158, r131, r155;
}
{
add.f16x2 r161, %17, %27;
}
{
mul.f16x2 r164, r161, r5;
}
{
add.f16x2 r167, %15, r164;
}
{
add.f16x2 r170, %19, %25;
}
{
mul.f16x2 r173, r170, r13;
}
{
add.f16x2 r176, r167, r173;
}
{
add.f16x2 r179, %21, %23;
}
{
mul.f16x2 r182, r179, r15;
}
{
add.f16x2 r185, r176, r182;
}
{
sub.f16x2 r188, %18, %28;
}
{
mul.f16x2 r191, r188, r7;
}
{
sub.f16x2 r194, %20, %26;
}
{
mul.f16x2 r197, r194, r14;
}
{
add.f16x2 r200, r191, r197;
}
{
sub.f16x2 r203, %22, %24;
}
{
mul.f16x2 r206, r203, r16;
}
{
add.f16x2 r209, r200, r206;
}
{
sub.f16x2 r212, r185, r209;
}
{
add.f16x2 r215, %17, %27;
}
{
mul.f16x2 r218, r215, r5;
}
{
add.f16x2 r221, %15, r218;
}
{
add.f16x2 r224, %19, %25;
}
{
mul.f16x2 r227, r224, r13;
}
{
add.f16x2 r230, r221, r227;
}
{
add.f16x2 r233, %21, %23;
}
{
mul.f16x2 r236, r233, r15;
}
{
add.f16x2 r239, r230, r236;
}
{
sub.f16x2 r242, %18, %28;
}
{
mul.f16x2 r245, r242, r7;
}
{
sub.f16x2 r248, %20, %26;
}
{
mul.f16x2 r251, r248, r14;
}
{
add.f16x2 r254, r245, r251;
}
{
sub.f16x2 r257, %22, %24;
}
{
mul.f16x2 r260, r257, r16;
}
{
add.f16x2 r263, r254, r260;
}
{
add.f16x2 r266, r239, r263;
}
{
add.f16x2 r269, %17, %27;
}
{
mul.f16x2 r272, r269, r9;
}
{
add.f16x2 r275, %15, r272;
}
{
add.f16x2 r278, %19, %25;
}
{
mul.f16x2 r281, r278, r15;
}
{
add.f16x2 r284, r275, r281;
}
{
add.f16x2 r287, %21, %23;
}
{
mul.f16x2 r290, r287, r5;
}
{
add.f16x2 r293, r284, r290;
}
{
sub.f16x2 r296, %18, %28;
}
{
mul.f16x2 r299, r296, r11;
}
{
sub.f16x2 r302, %20, %26;
}
{
mul.f16x2 r305, r302, r16;
}
{
add.f16x2 r308, r299, r305;
}
{
sub.f16x2 r311, %22, %24;
}
{
mul.f16x2 r314, r311, r7;
}
{
add.f16x2 r317, r308, r314;
}
{
sub.f16x2 r320, r293, r317;
}
{
add.f16x2 r323, %17, %27;
}
{
mul.f16x2 r326, r323, r9;
}
{
add.f16x2 r329, %15, r326;
}
{
add.f16x2 r332, %19, %25;
}
{
mul.f16x2 r335, r332, r15;
}
{
add.f16x2 r338, r329, r335;
}
{
add.f16x2 r341, %21, %23;
}
{
mul.f16x2 r344, r341, r5;
}
{
add.f16x2 r347, r338, r344;
}
{
sub.f16x2 r350, %18, %28;
}
{
mul.f16x2 r353, r350, r11;
}
{
sub.f16x2 r356, %20, %26;
}
{
mul.f16x2 r359, r356, r16;
}
{
add.f16x2 r362, r353, r359;
}
{
sub.f16x2 r365, %22, %24;
}
{
mul.f16x2 r368, r365, r7;
}
{
add.f16x2 r371, r362, r368;
}
{
add.f16x2 r374, r347, r371;
}
{
add.f16x2 r377, %18, %28;
}
{
mul.f16x2 r380, r377, r1;
}
{
add.f16x2 r383, %16, r380;
}
{
add.f16x2 r386, %20, %26;
}
{
mul.f16x2 r389, r386, r5;
}
{
add.f16x2 r392, r383, r389;
}
{
add.f16x2 r395, %22, %24;
}
{
mul.f16x2 r398, r395, r9;
}
{
add.f16x2 r401, r392, r398;
}
{
sub.f16x2 r404, %17, %27;
}
{
mul.f16x2 r407, r404, r3;
}
{
sub.f16x2 r410, %19, %25;
}
{
mul.f16x2 r413, r410, r7;
}
{
add.f16x2 r416, r407, r413;
}
{
sub.f16x2 r419, %21, %23;
}
{
mul.f16x2 r422, r419, r11;
}
{
add.f16x2 r425, r416, r422;
}
{
add.f16x2 r428, r401, r425;
}
{
add.f16x2 r431, %18, %28;
}
{
mul.f16x2 r434, r431, r1;
}
{
add.f16x2 r437, %16, r434;
}
{
add.f16x2 r440, %20, %26;
}
{
mul.f16x2 r443, r440, r5;
}
{
add.f16x2 r446, r437, r443;
}
{
add.f16x2 r449, %22, %24;
}
{
mul.f16x2 r452, r449, r9;
}
{
add.f16x2 r455, r446, r452;
}
{
sub.f16x2 r458, %17, %27;
}
{
mul.f16x2 r461, r458, r3;
}
{
sub.f16x2 r464, %19, %25;
}
{
mul.f16x2 r467, r464, r7;
}
{
add.f16x2 r470, r461, r467;
}
{
sub.f16x2 r473, %21, %23;
}
{
mul.f16x2 r476, r473, r11;
}
{
add.f16x2 r479, r470, r476;
}
{
sub.f16x2 r482, r455, r479;
}
{
add.f16x2 r485, %18, %28;
}
{
mul.f16x2 r488, r485, r5;
}
{
add.f16x2 r491, %16, r488;
}
{
add.f16x2 r494, %20, %26;
}
{
mul.f16x2 r497, r494, r13;
}
{
add.f16x2 r500, r491, r497;
}
{
add.f16x2 r503, %22, %24;
}
{
mul.f16x2 r506, r503, r15;
}
{
add.f16x2 r509, r500, r506;
}
{
sub.f16x2 r512, %17, %27;
}
{
mul.f16x2 r515, r512, r7;
}
{
sub.f16x2 r518, %19, %25;
}
{
mul.f16x2 r521, r518, r14;
}
{
add.f16x2 r524, r515, r521;
}
{
sub.f16x2 r527, %21, %23;
}
{
mul.f16x2 r530, r527, r16;
}
{
add.f16x2 r533, r524, r530;
}
{
add.f16x2 r536, r509, r533;
}
{
add.f16x2 r539, %18, %28;
}
{
mul.f16x2 r542, r539, r5;
}
{
add.f16x2 r545, %16, r542;
}
{
add.f16x2 r548, %20, %26;
}
{
mul.f16x2 r551, r548, r13;
}
{
add.f16x2 r554, r545, r551;
}
{
add.f16x2 r557, %22, %24;
}
{
mul.f16x2 r560, r557, r15;
}
{
add.f16x2 r563, r554, r560;
}
{
sub.f16x2 r566, %17, %27;
}
{
mul.f16x2 r569, r566, r7;
}
{
sub.f16x2 r572, %19, %25;
}
{
mul.f16x2 r575, r572, r14;
}
{
add.f16x2 r578, r569, r575;
}
{
sub.f16x2 r581, %21, %23;
}
{
mul.f16x2 r584, r581, r16;
}
{
add.f16x2 r587, r578, r584;
}
{
sub.f16x2 r590, r563, r587;
}
{
add.f16x2 r593, %18, %28;
}
{
mul.f16x2 r596, r593, r9;
}
{
add.f16x2 r599, %16, r596;
}
{
add.f16x2 r602, %20, %26;
}
{
mul.f16x2 r605, r602, r15;
}
{
add.f16x2 r608, r599, r605;
}
{
add.f16x2 r611, %22, %24;
}
{
mul.f16x2 r614, r611, r5;
}
{
add.f16x2 r617, r608, r614;
}
{
sub.f16x2 r620, %17, %27;
}
{
mul.f16x2 r623, r620, r11;
}
{
sub.f16x2 r626, %19, %25;
}
{
mul.f16x2 r629, r626, r16;
}
{
add.f16x2 r632, r623, r629;
}
{
sub.f16x2 r635, %21, %23;
}
{
mul.f16x2 r638, r635, r7;
}
{
add.f16x2 r641, r632, r638;
}
{
add.f16x2 r644, r617, r641;
}
{
add.f16x2 r647, %18, %28;
}
{
mul.f16x2 r650, r647, r9;
}
{
add.f16x2 r653, %16, r650;
}
{
add.f16x2 r656, %20, %26;
}
{
mul.f16x2 r659, r656, r15;
}
{
add.f16x2 r662, r653, r659;
}
{
add.f16x2 r665, %22, %24;
}
{
mul.f16x2 r668, r665, r5;
}
{
add.f16x2 r671, r662, r668;
}
{
sub.f16x2 r674, %17, %27;
}
{
mul.f16x2 r677, r674, r11;
}
{
sub.f16x2 r680, %19, %25;
}
{
mul.f16x2 r683, r680, r16;
}
{
add.f16x2 r686, r677, r683;
}
{
sub.f16x2 r689, %21, %23;
}
{
mul.f16x2 r692, r689, r7;
}
{
add.f16x2 r695, r686, r692;
}
{
sub.f16x2 r698, r671, r695;
}
mul.wide.u32 rd2, r2554, 1402438301;
shr.u64 rd3, rd2, 36;
cvt.u32.u64 r2555, rd3;
mul.lo.s32 r2556, r2555, 49;
sub.s32 r2557, r2554, r2556;
cvt.rn.f32.u32 f93, r2557;
mul.f32 f94, f93, 0f3C961050;
cos.approx.f32 f21, f94;
sin.approx.f32 f95, f94;
neg.f32 f22, f95;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f21;
cvt.rn.f16.f32 high, f22;
mov.b32 r701, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r704, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r706, {high, high};
}
{
mul.f16x2 r708, r428, r706;
}
{
fma.rn.f16x2 r711, r104, r704, r708;
}
{
mul.f16x2 r715, r104, r706;
}
{
neg.f16x2 r718, r715;
}
{
fma.rn.f16x2 r720, r428, r704, r718;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r724, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r726, {high, high};
}
mov.f32 f69, 0fBF800000;
mov.f32 f70, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r728, {low, high};
}
{
mul.f16x2 r729, r726, r728;
}
{
mul.f16x2 r732, r701, r724;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r735, {high, low};
}
{
fma.rn.f16x2 r737, r729, r735, r732;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r737;
mov.b32 r741, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r737;
mov.b32 r743, {high, high};
}
{
mul.f16x2 r745, r536, r743;
}
{
fma.rn.f16x2 r748, r212, r741, r745;
}
{
mul.f16x2 r752, r212, r743;
}
{
neg.f16x2 r755, r752;
}
{
fma.rn.f16x2 r757, r536, r741, r755;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r761, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r763, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r765, {low, high};
}
{
mul.f16x2 r766, r763, r765;
}
{
mul.f16x2 r769, r737, r761;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r737;
mov.b32 r772, {high, low};
}
{
fma.rn.f16x2 r774, r766, r772, r769;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r774;
mov.b32 r778, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r774;
mov.b32 r780, {high, high};
}
{
mul.f16x2 r782, r644, r780;
}
{
fma.rn.f16x2 r785, r320, r778, r782;
}
{
mul.f16x2 r789, r320, r780;
}
{
neg.f16x2 r792, r789;
}
{
fma.rn.f16x2 r794, r644, r778, r792;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r798, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r800, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r802, {low, high};
}
{
mul.f16x2 r803, r800, r802;
}
{
mul.f16x2 r806, r774, r798;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r774;
mov.b32 r809, {high, low};
}
{
fma.rn.f16x2 r811, r803, r809, r806;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r811;
mov.b32 r815, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r811;
mov.b32 r817, {high, high};
}
{
mul.f16x2 r819, r698, r817;
}
{
fma.rn.f16x2 r822, r374, r815, r819;
}
{
mul.f16x2 r826, r374, r817;
}
{
neg.f16x2 r829, r826;
}
{
fma.rn.f16x2 r831, r698, r815, r829;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r835, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r837, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r839, {low, high};
}
{
mul.f16x2 r840, r837, r839;
}
{
mul.f16x2 r843, r811, r835;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r811;
mov.b32 r846, {high, low};
}
{
fma.rn.f16x2 r848, r840, r846, r843;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r848;
mov.b32 r852, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r848;
mov.b32 r854, {high, high};
}
{
mul.f16x2 r856, r590, r854;
}
{
fma.rn.f16x2 r859, r266, r852, r856;
}
{
mul.f16x2 r863, r266, r854;
}
{
neg.f16x2 r866, r863;
}
{
fma.rn.f16x2 r868, r590, r852, r866;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r872, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r701;
mov.b32 r874, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r876, {low, high};
}
{
mul.f16x2 r877, r874, r876;
}
{
mul.f16x2 r880, r848, r872;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r848;
mov.b32 r883, {high, low};
}
{
fma.rn.f16x2 r885, r877, r883, r880;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r885;
mov.b32 r889, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r885;
mov.b32 r891, {high, high};
}
{
mul.f16x2 r893, r482, r891;
}
{
fma.rn.f16x2 r896, r158, r889, r893;
}
{
mul.f16x2 r900, r158, r891;
}
{
neg.f16x2 r903, r900;
}
{
fma.rn.f16x2 r905, r482, r889, r903;
}
mad.lo.s32 r2558, r2555, 1372, r2553;
barrier.sync 0;
mad.lo.s32 r2559, r2557, 28, r2558;
st.shared.u32 [r2559], r32;
st.shared.u32 [r2559+4], r711;
st.shared.u32 [r2559+8], r748;
st.shared.u32 [r2559+12], r785;
st.shared.u32 [r2559+16], r822;
st.shared.u32 [r2559+20], r859;
st.shared.u32 [r2559+24], r896;
barrier.sync 0;
mad.lo.s32 r2560, r2557, -24, r2559;
ld.shared.u32 r946, [r2560];
ld.shared.u32 r943, [r2560+196];
ld.shared.u32 r949, [r2560+392];
ld.shared.u32 r955, [r2560+588];
ld.shared.u32 r956, [r2560+784];
ld.shared.u32 r950, [r2560+980];
ld.shared.u32 r944, [r2560+1176];
barrier.sync 0;
st.shared.u32 [r2559], r50;
st.shared.u32 [r2559+4], r720;
st.shared.u32 [r2559+8], r757;
st.shared.u32 [r2559+12], r794;
st.shared.u32 [r2559+16], r831;
st.shared.u32 [r2559+20], r868;
st.shared.u32 [r2559+24], r905;
barrier.sync 0;
ld.shared.u32 r964, [r2560];
ld.shared.u32 r961, [r2560+196];
ld.shared.u32 r967, [r2560+392];
ld.shared.u32 r973, [r2560+588];
ld.shared.u32 r974, [r2560+784];
ld.shared.u32 r968, [r2560+980];
ld.shared.u32 r962, [r2560+1176];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r926, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r927, {low, high};
}
{
neg.f16x2 r928, r927;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f78;
cvt.rn.f16.f32 high, f78;
mov.b32 r930, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r931, {low, high};
}
{
neg.f16x2 r932, r931;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r934, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r935, {low, high};
}
{
neg.f16x2 r936, r935;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r938, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r939, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r940, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r941, {low, high};
}
{
add.f16x2 r942, r943, r944;
}
{
add.f16x2 r945, r946, r942;
}
{
add.f16x2 r948, r949, r950;
}
{
add.f16x2 r951, r945, r948;
}
{
add.f16x2 r954, r955, r956;
}
{
add.f16x2 r957, r951, r954;
}
{
add.f16x2 r960, r961, r962;
}
{
add.f16x2 r963, r964, r960;
}
{
add.f16x2 r966, r967, r968;
}
{
add.f16x2 r969, r963, r966;
}
{
add.f16x2 r972, r973, r974;
}
{
add.f16x2 r975, r969, r972;
}
{
add.f16x2 r978, r943, r944;
}
{
mul.f16x2 r981, r978, r926;
}
{
add.f16x2 r984, r946, r981;
}
{
add.f16x2 r987, r949, r950;
}
{
mul.f16x2 r990, r987, r930;
}
{
add.f16x2 r993, r984, r990;
}
{
add.f16x2 r996, r955, r956;
}
{
mul.f16x2 r999, r996, r934;
}
{
add.f16x2 r1002, r993, r999;
}
{
sub.f16x2 r1005, r961, r962;
}
{
mul.f16x2 r1008, r1005, r928;
}
{
sub.f16x2 r1011, r967, r968;
}
{
mul.f16x2 r1014, r1011, r932;
}
{
add.f16x2 r1017, r1008, r1014;
}
{
sub.f16x2 r1020, r973, r974;
}
{
mul.f16x2 r1023, r1020, r936;
}
{
add.f16x2 r1026, r1017, r1023;
}
{
sub.f16x2 r1029, r1002, r1026;
}
{
add.f16x2 r1032, r943, r944;
}
{
mul.f16x2 r1035, r1032, r926;
}
{
add.f16x2 r1038, r946, r1035;
}
{
add.f16x2 r1041, r949, r950;
}
{
mul.f16x2 r1044, r1041, r930;
}
{
add.f16x2 r1047, r1038, r1044;
}
{
add.f16x2 r1050, r955, r956;
}
{
mul.f16x2 r1053, r1050, r934;
}
{
add.f16x2 r1056, r1047, r1053;
}
{
sub.f16x2 r1059, r961, r962;
}
{
mul.f16x2 r1062, r1059, r928;
}
{
sub.f16x2 r1065, r967, r968;
}
{
mul.f16x2 r1068, r1065, r932;
}
{
add.f16x2 r1071, r1062, r1068;
}
{
sub.f16x2 r1074, r973, r974;
}
{
mul.f16x2 r1077, r1074, r936;
}
{
add.f16x2 r1080, r1071, r1077;
}
{
add.f16x2 r1083, r1056, r1080;
}
{
add.f16x2 r1086, r943, r944;
}
{
mul.f16x2 r1089, r1086, r930;
}
{
add.f16x2 r1092, r946, r1089;
}
{
add.f16x2 r1095, r949, r950;
}
{
mul.f16x2 r1098, r1095, r938;
}
{
add.f16x2 r1101, r1092, r1098;
}
{
add.f16x2 r1104, r955, r956;
}
{
mul.f16x2 r1107, r1104, r940;
}
{
add.f16x2 r1110, r1101, r1107;
}
{
sub.f16x2 r1113, r961, r962;
}
{
mul.f16x2 r1116, r1113, r932;
}
{
sub.f16x2 r1119, r967, r968;
}
{
mul.f16x2 r1122, r1119, r939;
}
{
add.f16x2 r1125, r1116, r1122;
}
{
sub.f16x2 r1128, r973, r974;
}
{
mul.f16x2 r1131, r1128, r941;
}
{
add.f16x2 r1134, r1125, r1131;
}
{
sub.f16x2 r1137, r1110, r1134;
}
{
add.f16x2 r1140, r943, r944;
}
{
mul.f16x2 r1143, r1140, r930;
}
{
add.f16x2 r1146, r946, r1143;
}
{
add.f16x2 r1149, r949, r950;
}
{
mul.f16x2 r1152, r1149, r938;
}
{
add.f16x2 r1155, r1146, r1152;
}
{
add.f16x2 r1158, r955, r956;
}
{
mul.f16x2 r1161, r1158, r940;
}
{
add.f16x2 r1164, r1155, r1161;
}
{
sub.f16x2 r1167, r961, r962;
}
{
mul.f16x2 r1170, r1167, r932;
}
{
sub.f16x2 r1173, r967, r968;
}
{
mul.f16x2 r1176, r1173, r939;
}
{
add.f16x2 r1179, r1170, r1176;
}
{
sub.f16x2 r1182, r973, r974;
}
{
mul.f16x2 r1185, r1182, r941;
}
{
add.f16x2 r1188, r1179, r1185;
}
{
add.f16x2 r1191, r1164, r1188;
}
{
add.f16x2 r1194, r943, r944;
}
{
mul.f16x2 r1197, r1194, r934;
}
{
add.f16x2 r1200, r946, r1197;
}
{
add.f16x2 r1203, r949, r950;
}
{
mul.f16x2 r1206, r1203, r940;
}
{
add.f16x2 r1209, r1200, r1206;
}
{
add.f16x2 r1212, r955, r956;
}
{
mul.f16x2 r1215, r1212, r930;
}
{
add.f16x2 r1218, r1209, r1215;
}
{
sub.f16x2 r1221, r961, r962;
}
{
mul.f16x2 r1224, r1221, r936;
}
{
sub.f16x2 r1227, r967, r968;
}
{
mul.f16x2 r1230, r1227, r941;
}
{
add.f16x2 r1233, r1224, r1230;
}
{
sub.f16x2 r1236, r973, r974;
}
{
mul.f16x2 r1239, r1236, r932;
}
{
add.f16x2 r1242, r1233, r1239;
}
{
sub.f16x2 r1245, r1218, r1242;
}
{
add.f16x2 r1248, r943, r944;
}
{
mul.f16x2 r1251, r1248, r934;
}
{
add.f16x2 r1254, r946, r1251;
}
{
add.f16x2 r1257, r949, r950;
}
{
mul.f16x2 r1260, r1257, r940;
}
{
add.f16x2 r1263, r1254, r1260;
}
{
add.f16x2 r1266, r955, r956;
}
{
mul.f16x2 r1269, r1266, r930;
}
{
add.f16x2 r1272, r1263, r1269;
}
{
sub.f16x2 r1275, r961, r962;
}
{
mul.f16x2 r1278, r1275, r936;
}
{
sub.f16x2 r1281, r967, r968;
}
{
mul.f16x2 r1284, r1281, r941;
}
{
add.f16x2 r1287, r1278, r1284;
}
{
sub.f16x2 r1290, r973, r974;
}
{
mul.f16x2 r1293, r1290, r932;
}
{
add.f16x2 r1296, r1287, r1293;
}
{
add.f16x2 r1299, r1272, r1296;
}
{
add.f16x2 r1302, r961, r962;
}
{
mul.f16x2 r1305, r1302, r926;
}
{
add.f16x2 r1308, r964, r1305;
}
{
add.f16x2 r1311, r967, r968;
}
{
mul.f16x2 r1314, r1311, r930;
}
{
add.f16x2 r1317, r1308, r1314;
}
{
add.f16x2 r1320, r973, r974;
}
{
mul.f16x2 r1323, r1320, r934;
}
{
add.f16x2 r1326, r1317, r1323;
}
{
sub.f16x2 r1329, r943, r944;
}
{
mul.f16x2 r1332, r1329, r928;
}
{
sub.f16x2 r1335, r949, r950;
}
{
mul.f16x2 r1338, r1335, r932;
}
{
add.f16x2 r1341, r1332, r1338;
}
{
sub.f16x2 r1344, r955, r956;
}
{
mul.f16x2 r1347, r1344, r936;
}
{
add.f16x2 r1350, r1341, r1347;
}
{
add.f16x2 r1353, r1326, r1350;
}
{
add.f16x2 r1356, r961, r962;
}
{
mul.f16x2 r1359, r1356, r926;
}
{
add.f16x2 r1362, r964, r1359;
}
{
add.f16x2 r1365, r967, r968;
}
{
mul.f16x2 r1368, r1365, r930;
}
{
add.f16x2 r1371, r1362, r1368;
}
{
add.f16x2 r1374, r973, r974;
}
{
mul.f16x2 r1377, r1374, r934;
}
{
add.f16x2 r1380, r1371, r1377;
}
{
sub.f16x2 r1383, r943, r944;
}
{
mul.f16x2 r1386, r1383, r928;
}
{
sub.f16x2 r1389, r949, r950;
}
{
mul.f16x2 r1392, r1389, r932;
}
{
add.f16x2 r1395, r1386, r1392;
}
{
sub.f16x2 r1398, r955, r956;
}
{
mul.f16x2 r1401, r1398, r936;
}
{
add.f16x2 r1404, r1395, r1401;
}
{
sub.f16x2 r1407, r1380, r1404;
}
{
add.f16x2 r1410, r961, r962;
}
{
mul.f16x2 r1413, r1410, r930;
}
{
add.f16x2 r1416, r964, r1413;
}
{
add.f16x2 r1419, r967, r968;
}
{
mul.f16x2 r1422, r1419, r938;
}
{
add.f16x2 r1425, r1416, r1422;
}
{
add.f16x2 r1428, r973, r974;
}
{
mul.f16x2 r1431, r1428, r940;
}
{
add.f16x2 r1434, r1425, r1431;
}
{
sub.f16x2 r1437, r943, r944;
}
{
mul.f16x2 r1440, r1437, r932;
}
{
sub.f16x2 r1443, r949, r950;
}
{
mul.f16x2 r1446, r1443, r939;
}
{
add.f16x2 r1449, r1440, r1446;
}
{
sub.f16x2 r1452, r955, r956;
}
{
mul.f16x2 r1455, r1452, r941;
}
{
add.f16x2 r1458, r1449, r1455;
}
{
add.f16x2 r1461, r1434, r1458;
}
{
add.f16x2 r1464, r961, r962;
}
{
mul.f16x2 r1467, r1464, r930;
}
{
add.f16x2 r1470, r964, r1467;
}
{
add.f16x2 r1473, r967, r968;
}
{
mul.f16x2 r1476, r1473, r938;
}
{
add.f16x2 r1479, r1470, r1476;
}
{
add.f16x2 r1482, r973, r974;
}
{
mul.f16x2 r1485, r1482, r940;
}
{
add.f16x2 r1488, r1479, r1485;
}
{
sub.f16x2 r1491, r943, r944;
}
{
mul.f16x2 r1494, r1491, r932;
}
{
sub.f16x2 r1497, r949, r950;
}
{
mul.f16x2 r1500, r1497, r939;
}
{
add.f16x2 r1503, r1494, r1500;
}
{
sub.f16x2 r1506, r955, r956;
}
{
mul.f16x2 r1509, r1506, r941;
}
{
add.f16x2 r1512, r1503, r1509;
}
{
sub.f16x2 r1515, r1488, r1512;
}
{
add.f16x2 r1518, r961, r962;
}
{
mul.f16x2 r1521, r1518, r934;
}
{
add.f16x2 r1524, r964, r1521;
}
{
add.f16x2 r1527, r967, r968;
}
{
mul.f16x2 r1530, r1527, r940;
}
{
add.f16x2 r1533, r1524, r1530;
}
{
add.f16x2 r1536, r973, r974;
}
{
mul.f16x2 r1539, r1536, r930;
}
{
add.f16x2 r1542, r1533, r1539;
}
{
sub.f16x2 r1545, r943, r944;
}
{
mul.f16x2 r1548, r1545, r936;
}
{
sub.f16x2 r1551, r949, r950;
}
{
mul.f16x2 r1554, r1551, r941;
}
{
add.f16x2 r1557, r1548, r1554;
}
{
sub.f16x2 r1560, r955, r956;
}
{
mul.f16x2 r1563, r1560, r932;
}
{
add.f16x2 r1566, r1557, r1563;
}
{
add.f16x2 r1569, r1542, r1566;
}
{
add.f16x2 r1572, r961, r962;
}
{
mul.f16x2 r1575, r1572, r934;
}
{
add.f16x2 r1578, r964, r1575;
}
{
add.f16x2 r1581, r967, r968;
}
{
mul.f16x2 r1584, r1581, r940;
}
{
add.f16x2 r1587, r1578, r1584;
}
{
add.f16x2 r1590, r973, r974;
}
{
mul.f16x2 r1593, r1590, r930;
}
{
add.f16x2 r1596, r1587, r1593;
}
{
sub.f16x2 r1599, r943, r944;
}
{
mul.f16x2 r1602, r1599, r936;
}
{
sub.f16x2 r1605, r949, r950;
}
{
mul.f16x2 r1608, r1605, r941;
}
{
add.f16x2 r1611, r1602, r1608;
}
{
sub.f16x2 r1614, r955, r956;
}
{
mul.f16x2 r1617, r1614, r932;
}
{
add.f16x2 r1620, r1611, r1617;
}
{
sub.f16x2 r1623, r1596, r1620;
}
mul.wide.u32 rd4, r2557, 613566757;
shr.u64 rd5, rd4, 32;
cvt.u32.u64 r2561, rd5;
sub.s32 r2562, r2557, r2561;
shr.u32 r2563, r2562, 1;
add.s32 r2564, r2563, r2561;
shr.u32 r2565, r2564, 2;
cvt.rn.f32.u32 f96, r2565;
mul.f32 f97, f96, 0f3E034E46;
cos.approx.f32 f57, f97;
sin.approx.f32 f98, f97;
neg.f32 f58, f98;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f57;
cvt.rn.f16.f32 high, f58;
mov.b32 r1626, {low, high};
}
mul.lo.s32 r2566, r2565, 7;
sub.s32 r2567, r2557, r2566;
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1629, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1631, {high, high};
}
{
mul.f16x2 r1633, r1353, r1631;
}
{
fma.rn.f16x2 r1636, r1029, r1629, r1633;
}
{
mul.f16x2 r1640, r1029, r1631;
}
{
neg.f16x2 r1643, r1640;
}
{
fma.rn.f16x2 r1645, r1353, r1629, r1643;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1649, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1651, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r1653, {low, high};
}
{
mul.f16x2 r1654, r1651, r1653;
}
{
mul.f16x2 r1657, r1626, r1649;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1660, {high, low};
}
{
fma.rn.f16x2 r1662, r1654, r1660, r1657;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1662;
mov.b32 r1666, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1662;
mov.b32 r1668, {high, high};
}
{
mul.f16x2 r1670, r1461, r1668;
}
{
fma.rn.f16x2 r1673, r1137, r1666, r1670;
}
{
mul.f16x2 r1677, r1137, r1668;
}
{
neg.f16x2 r1680, r1677;
}
{
fma.rn.f16x2 r1682, r1461, r1666, r1680;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1686, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1688, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r1690, {low, high};
}
{
mul.f16x2 r1691, r1688, r1690;
}
{
mul.f16x2 r1694, r1662, r1686;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1662;
mov.b32 r1697, {high, low};
}
{
fma.rn.f16x2 r1699, r1691, r1697, r1694;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1699;
mov.b32 r1703, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1699;
mov.b32 r1705, {high, high};
}
{
mul.f16x2 r1707, r1569, r1705;
}
{
fma.rn.f16x2 r1710, r1245, r1703, r1707;
}
{
mul.f16x2 r1714, r1245, r1705;
}
{
neg.f16x2 r1717, r1714;
}
{
fma.rn.f16x2 r1719, r1569, r1703, r1717;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1723, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1725, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r1727, {low, high};
}
{
mul.f16x2 r1728, r1725, r1727;
}
{
mul.f16x2 r1731, r1699, r1723;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1699;
mov.b32 r1734, {high, low};
}
{
fma.rn.f16x2 r1736, r1728, r1734, r1731;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1736;
mov.b32 r1740, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1736;
mov.b32 r1742, {high, high};
}
{
mul.f16x2 r1744, r1623, r1742;
}
{
fma.rn.f16x2 r1747, r1299, r1740, r1744;
}
{
mul.f16x2 r1751, r1299, r1742;
}
{
neg.f16x2 r1754, r1751;
}
{
fma.rn.f16x2 r1756, r1623, r1740, r1754;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1760, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1762, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r1764, {low, high};
}
{
mul.f16x2 r1765, r1762, r1764;
}
{
mul.f16x2 r1768, r1736, r1760;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1736;
mov.b32 r1771, {high, low};
}
{
fma.rn.f16x2 r1773, r1765, r1771, r1768;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1773;
mov.b32 r1777, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1773;
mov.b32 r1779, {high, high};
}
{
mul.f16x2 r1781, r1515, r1779;
}
{
fma.rn.f16x2 r1784, r1191, r1777, r1781;
}
{
mul.f16x2 r1788, r1191, r1779;
}
{
neg.f16x2 r1791, r1788;
}
{
fma.rn.f16x2 r1793, r1515, r1777, r1791;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1797, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1626;
mov.b32 r1799, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
mov.b32 r1801, {low, high};
}
{
mul.f16x2 r1802, r1799, r1801;
}
{
mul.f16x2 r1805, r1773, r1797;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1773;
mov.b32 r1808, {high, low};
}
{
fma.rn.f16x2 r1810, r1802, r1808, r1805;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1810;
mov.b32 r1814, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1810;
mov.b32 r1816, {high, high};
}
{
mul.f16x2 r1818, r1407, r1816;
}
{
fma.rn.f16x2 r1821, r1083, r1814, r1818;
}
{
mul.f16x2 r1825, r1083, r1816;
}
{
neg.f16x2 r1828, r1825;
}
{
fma.rn.f16x2 r1830, r1407, r1814, r1828;
}
shl.b32 r2568, r2567, 2;
add.s32 r2569, r2558, r2568;
barrier.sync 0;
mad.lo.s32 r2570, r2565, 196, r2569;
st.shared.u32 [r2570], r957;
st.shared.u32 [r2570+28], r1636;
st.shared.u32 [r2570+56], r1673;
st.shared.u32 [r2570+84], r1710;
st.shared.u32 [r2570+112], r1747;
st.shared.u32 [r2570+140], r1784;
st.shared.u32 [r2570+168], r1821;
barrier.sync 0;
ld.shared.u32 r1871, [r2560];
ld.shared.u32 r1868, [r2560+196];
ld.shared.u32 r1874, [r2560+392];
ld.shared.u32 r1880, [r2560+588];
ld.shared.u32 r1881, [r2560+784];
ld.shared.u32 r1875, [r2560+980];
ld.shared.u32 r1869, [r2560+1176];
barrier.sync 0;
st.shared.u32 [r2570], r975;
st.shared.u32 [r2570+28], r1645;
st.shared.u32 [r2570+56], r1682;
st.shared.u32 [r2570+84], r1719;
st.shared.u32 [r2570+112], r1756;
st.shared.u32 [r2570+140], r1793;
st.shared.u32 [r2570+168], r1830;
barrier.sync 0;
ld.shared.u32 r1889, [r2560];
ld.shared.u32 r1886, [r2560+196];
ld.shared.u32 r1892, [r2560+392];
ld.shared.u32 r1898, [r2560+588];
ld.shared.u32 r1899, [r2560+784];
ld.shared.u32 r1893, [r2560+980];
ld.shared.u32 r1887, [r2560+1176];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r1851, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r1852, {low, high};
}
{
neg.f16x2 r1853, r1852;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f78;
cvt.rn.f16.f32 high, f78;
mov.b32 r1855, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r1856, {low, high};
}
{
neg.f16x2 r1857, r1856;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r1859, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r1860, {low, high};
}
{
neg.f16x2 r1861, r1860;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r1863, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r1864, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r1865, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r1866, {low, high};
}
{
add.f16x2 r1867, r1868, r1869;
}
{
add.f16x2 r1870, r1871, r1867;
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
add.f16x2 %0, r1876, r1879;
}
{
add.f16x2 r1885, r1886, r1887;
}
{
add.f16x2 r1888, r1889, r1885;
}
{
add.f16x2 r1891, r1892, r1893;
}
{
add.f16x2 r1894, r1888, r1891;
}
{
add.f16x2 r1897, r1898, r1899;
}
{
add.f16x2 %1, r1894, r1897;
}
{
add.f16x2 r1903, r1868, r1869;
}
{
mul.f16x2 r1906, r1903, r1851;
}
{
add.f16x2 r1909, r1871, r1906;
}
{
add.f16x2 r1912, r1874, r1875;
}
{
mul.f16x2 r1915, r1912, r1855;
}
{
add.f16x2 r1918, r1909, r1915;
}
{
add.f16x2 r1921, r1880, r1881;
}
{
mul.f16x2 r1924, r1921, r1859;
}
{
add.f16x2 r1927, r1918, r1924;
}
{
sub.f16x2 r1930, r1886, r1887;
}
{
mul.f16x2 r1933, r1930, r1853;
}
{
sub.f16x2 r1936, r1892, r1893;
}
{
mul.f16x2 r1939, r1936, r1857;
}
{
add.f16x2 r1942, r1933, r1939;
}
{
sub.f16x2 r1945, r1898, r1899;
}
{
mul.f16x2 r1948, r1945, r1861;
}
{
add.f16x2 r1951, r1942, r1948;
}
{
sub.f16x2 %2, r1927, r1951;
}
{
add.f16x2 r1957, r1868, r1869;
}
{
mul.f16x2 r1960, r1957, r1851;
}
{
add.f16x2 r1963, r1871, r1960;
}
{
add.f16x2 r1966, r1874, r1875;
}
{
mul.f16x2 r1969, r1966, r1855;
}
{
add.f16x2 r1972, r1963, r1969;
}
{
add.f16x2 r1975, r1880, r1881;
}
{
mul.f16x2 r1978, r1975, r1859;
}
{
add.f16x2 r1981, r1972, r1978;
}
{
sub.f16x2 r1984, r1886, r1887;
}
{
mul.f16x2 r1987, r1984, r1853;
}
{
sub.f16x2 r1990, r1892, r1893;
}
{
mul.f16x2 r1993, r1990, r1857;
}
{
add.f16x2 r1996, r1987, r1993;
}
{
sub.f16x2 r1999, r1898, r1899;
}
{
mul.f16x2 r2002, r1999, r1861;
}
{
add.f16x2 r2005, r1996, r2002;
}
{
add.f16x2 %12, r1981, r2005;
}
{
add.f16x2 r2011, r1868, r1869;
}
{
mul.f16x2 r2014, r2011, r1855;
}
{
add.f16x2 r2017, r1871, r2014;
}
{
add.f16x2 r2020, r1874, r1875;
}
{
mul.f16x2 r2023, r2020, r1863;
}
{
add.f16x2 r2026, r2017, r2023;
}
{
add.f16x2 r2029, r1880, r1881;
}
{
mul.f16x2 r2032, r2029, r1865;
}
{
add.f16x2 r2035, r2026, r2032;
}
{
sub.f16x2 r2038, r1886, r1887;
}
{
mul.f16x2 r2041, r2038, r1857;
}
{
sub.f16x2 r2044, r1892, r1893;
}
{
mul.f16x2 r2047, r2044, r1864;
}
{
add.f16x2 r2050, r2041, r2047;
}
{
sub.f16x2 r2053, r1898, r1899;
}
{
mul.f16x2 r2056, r2053, r1866;
}
{
add.f16x2 r2059, r2050, r2056;
}
{
sub.f16x2 %4, r2035, r2059;
}
{
add.f16x2 r2065, r1868, r1869;
}
{
mul.f16x2 r2068, r2065, r1855;
}
{
add.f16x2 r2071, r1871, r2068;
}
{
add.f16x2 r2074, r1874, r1875;
}
{
mul.f16x2 r2077, r2074, r1863;
}
{
add.f16x2 r2080, r2071, r2077;
}
{
add.f16x2 r2083, r1880, r1881;
}
{
mul.f16x2 r2086, r2083, r1865;
}
{
add.f16x2 r2089, r2080, r2086;
}
{
sub.f16x2 r2092, r1886, r1887;
}
{
mul.f16x2 r2095, r2092, r1857;
}
{
sub.f16x2 r2098, r1892, r1893;
}
{
mul.f16x2 r2101, r2098, r1864;
}
{
add.f16x2 r2104, r2095, r2101;
}
{
sub.f16x2 r2107, r1898, r1899;
}
{
mul.f16x2 r2110, r2107, r1866;
}
{
add.f16x2 r2113, r2104, r2110;
}
{
add.f16x2 %10, r2089, r2113;
}
{
add.f16x2 r2119, r1868, r1869;
}
{
mul.f16x2 r2122, r2119, r1859;
}
{
add.f16x2 r2125, r1871, r2122;
}
{
add.f16x2 r2128, r1874, r1875;
}
{
mul.f16x2 r2131, r2128, r1865;
}
{
add.f16x2 r2134, r2125, r2131;
}
{
add.f16x2 r2137, r1880, r1881;
}
{
mul.f16x2 r2140, r2137, r1855;
}
{
add.f16x2 r2143, r2134, r2140;
}
{
sub.f16x2 r2146, r1886, r1887;
}
{
mul.f16x2 r2149, r2146, r1861;
}
{
sub.f16x2 r2152, r1892, r1893;
}
{
mul.f16x2 r2155, r2152, r1866;
}
{
add.f16x2 r2158, r2149, r2155;
}
{
sub.f16x2 r2161, r1898, r1899;
}
{
mul.f16x2 r2164, r2161, r1857;
}
{
add.f16x2 r2167, r2158, r2164;
}
{
sub.f16x2 %6, r2143, r2167;
}
{
add.f16x2 r2173, r1868, r1869;
}
{
mul.f16x2 r2176, r2173, r1859;
}
{
add.f16x2 r2179, r1871, r2176;
}
{
add.f16x2 r2182, r1874, r1875;
}
{
mul.f16x2 r2185, r2182, r1865;
}
{
add.f16x2 r2188, r2179, r2185;
}
{
add.f16x2 r2191, r1880, r1881;
}
{
mul.f16x2 r2194, r2191, r1855;
}
{
add.f16x2 r2197, r2188, r2194;
}
{
sub.f16x2 r2200, r1886, r1887;
}
{
mul.f16x2 r2203, r2200, r1861;
}
{
sub.f16x2 r2206, r1892, r1893;
}
{
mul.f16x2 r2209, r2206, r1866;
}
{
add.f16x2 r2212, r2203, r2209;
}
{
sub.f16x2 r2215, r1898, r1899;
}
{
mul.f16x2 r2218, r2215, r1857;
}
{
add.f16x2 r2221, r2212, r2218;
}
{
add.f16x2 %8, r2197, r2221;
}
{
add.f16x2 r2227, r1886, r1887;
}
{
mul.f16x2 r2230, r2227, r1851;
}
{
add.f16x2 r2233, r1889, r2230;
}
{
add.f16x2 r2236, r1892, r1893;
}
{
mul.f16x2 r2239, r2236, r1855;
}
{
add.f16x2 r2242, r2233, r2239;
}
{
add.f16x2 r2245, r1898, r1899;
}
{
mul.f16x2 r2248, r2245, r1859;
}
{
add.f16x2 r2251, r2242, r2248;
}
{
sub.f16x2 r2254, r1868, r1869;
}
{
mul.f16x2 r2257, r2254, r1853;
}
{
sub.f16x2 r2260, r1874, r1875;
}
{
mul.f16x2 r2263, r2260, r1857;
}
{
add.f16x2 r2266, r2257, r2263;
}
{
sub.f16x2 r2269, r1880, r1881;
}
{
mul.f16x2 r2272, r2269, r1861;
}
{
add.f16x2 r2275, r2266, r2272;
}
{
add.f16x2 %3, r2251, r2275;
}
{
add.f16x2 r2281, r1886, r1887;
}
{
mul.f16x2 r2284, r2281, r1851;
}
{
add.f16x2 r2287, r1889, r2284;
}
{
add.f16x2 r2290, r1892, r1893;
}
{
mul.f16x2 r2293, r2290, r1855;
}
{
add.f16x2 r2296, r2287, r2293;
}
{
add.f16x2 r2299, r1898, r1899;
}
{
mul.f16x2 r2302, r2299, r1859;
}
{
add.f16x2 r2305, r2296, r2302;
}
{
sub.f16x2 r2308, r1868, r1869;
}
{
mul.f16x2 r2311, r2308, r1853;
}
{
sub.f16x2 r2314, r1874, r1875;
}
{
mul.f16x2 r2317, r2314, r1857;
}
{
add.f16x2 r2320, r2311, r2317;
}
{
sub.f16x2 r2323, r1880, r1881;
}
{
mul.f16x2 r2326, r2323, r1861;
}
{
add.f16x2 r2329, r2320, r2326;
}
{
sub.f16x2 %13, r2305, r2329;
}
{
add.f16x2 r2335, r1886, r1887;
}
{
mul.f16x2 r2338, r2335, r1855;
}
{
add.f16x2 r2341, r1889, r2338;
}
{
add.f16x2 r2344, r1892, r1893;
}
{
mul.f16x2 r2347, r2344, r1863;
}
{
add.f16x2 r2350, r2341, r2347;
}
{
add.f16x2 r2353, r1898, r1899;
}
{
mul.f16x2 r2356, r2353, r1865;
}
{
add.f16x2 r2359, r2350, r2356;
}
{
sub.f16x2 r2362, r1868, r1869;
}
{
mul.f16x2 r2365, r2362, r1857;
}
{
sub.f16x2 r2368, r1874, r1875;
}
{
mul.f16x2 r2371, r2368, r1864;
}
{
add.f16x2 r2374, r2365, r2371;
}
{
sub.f16x2 r2377, r1880, r1881;
}
{
mul.f16x2 r2380, r2377, r1866;
}
{
add.f16x2 r2383, r2374, r2380;
}
{
add.f16x2 %5, r2359, r2383;
}
{
add.f16x2 r2389, r1886, r1887;
}
{
mul.f16x2 r2392, r2389, r1855;
}
{
add.f16x2 r2395, r1889, r2392;
}
{
add.f16x2 r2398, r1892, r1893;
}
{
mul.f16x2 r2401, r2398, r1863;
}
{
add.f16x2 r2404, r2395, r2401;
}
{
add.f16x2 r2407, r1898, r1899;
}
{
mul.f16x2 r2410, r2407, r1865;
}
{
add.f16x2 r2413, r2404, r2410;
}
{
sub.f16x2 r2416, r1868, r1869;
}
{
mul.f16x2 r2419, r2416, r1857;
}
{
sub.f16x2 r2422, r1874, r1875;
}
{
mul.f16x2 r2425, r2422, r1864;
}
{
add.f16x2 r2428, r2419, r2425;
}
{
sub.f16x2 r2431, r1880, r1881;
}
{
mul.f16x2 r2434, r2431, r1866;
}
{
add.f16x2 r2437, r2428, r2434;
}
{
sub.f16x2 %11, r2413, r2437;
}
{
add.f16x2 r2443, r1886, r1887;
}
{
mul.f16x2 r2446, r2443, r1859;
}
{
add.f16x2 r2449, r1889, r2446;
}
{
add.f16x2 r2452, r1892, r1893;
}
{
mul.f16x2 r2455, r2452, r1865;
}
{
add.f16x2 r2458, r2449, r2455;
}
{
add.f16x2 r2461, r1898, r1899;
}
{
mul.f16x2 r2464, r2461, r1855;
}
{
add.f16x2 r2467, r2458, r2464;
}
{
sub.f16x2 r2470, r1868, r1869;
}
{
mul.f16x2 r2473, r2470, r1861;
}
{
sub.f16x2 r2476, r1874, r1875;
}
{
mul.f16x2 r2479, r2476, r1866;
}
{
add.f16x2 r2482, r2473, r2479;
}
{
sub.f16x2 r2485, r1880, r1881;
}
{
mul.f16x2 r2488, r2485, r1857;
}
{
add.f16x2 r2491, r2482, r2488;
}
{
add.f16x2 %7, r2467, r2491;
}
{
add.f16x2 r2497, r1886, r1887;
}
{
mul.f16x2 r2500, r2497, r1859;
}
{
add.f16x2 r2503, r1889, r2500;
}
{
add.f16x2 r2506, r1892, r1893;
}
{
mul.f16x2 r2509, r2506, r1865;
}
{
add.f16x2 r2512, r2503, r2509;
}
{
add.f16x2 r2515, r1898, r1899;
}
{
mul.f16x2 r2518, r2515, r1855;
}
{
add.f16x2 r2521, r2512, r2518;
}
{
sub.f16x2 r2524, r1868, r1869;
}
{
mul.f16x2 r2527, r2524, r1861;
}
{
sub.f16x2 r2530, r1874, r1875;
}
{
mul.f16x2 r2533, r2530, r1866;
}
{
add.f16x2 r2536, r2527, r2533;
}
{
sub.f16x2 r2539, r1880, r1881;
}
{
mul.f16x2 r2542, r2539, r1857;
}
{
add.f16x2 r2545, r2536, r2542;
}
{
sub.f16x2 %9, r2521, r2545;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)));
};


#endif
