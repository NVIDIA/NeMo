#ifndef CUFFTDX_FFT_49_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_49_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1120, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<60>;
.reg .b32 r<1640>;
.reg .b64 rd<4>;
mov.u32 r1626, %tid.x;
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
{
neg.f16x2 r3, r2;
}
mov.f32 f42, 0fBE63DC87;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f42;
cvt.rn.f16.f32 high, f42;
mov.b32 r5, {low, high};
}
mov.f32 f44, 0fBF7994E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f44;
cvt.rn.f16.f32 high, f44;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
mov.f32 f50, 0fBF66A5E5;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r9, {low, high};
}
mov.f32 f52, 0fBEDE2602;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r10, {low, high};
}
{
neg.f16x2 r11, r10;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r13, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r14, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f54;
cvt.rn.f16.f32 high, f54;
mov.b32 r15, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
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
mul.wide.u32 rd2, r1626, 613566757;
shr.u64 rd3, rd2, 32;
cvt.u32.u64 r1627, rd3;
sub.s32 r1628, r1626, r1627;
shr.u32 r1629, r1628, 1;
add.s32 r1630, r1629, r1627;
shr.u32 r1631, r1630, 2;
mul.lo.s32 r1632, r1631, 7;
sub.s32 r1633, r1626, r1632;
cvt.rn.f32.u32 f57, r1633;
mul.f32 f58, f57, 0f3E034E46;
cos.approx.f32 f21, f58;
sin.approx.f32 f59, f58;
neg.f32 f22, f59;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f21;
cvt.rn.f16.f32 high, f22;
mov.b32 r701, {low, high};
}
mov.u32 r1634, %tid.y;
mov.u32 r1635, %14;
mad.lo.s32 r1636, r1634, 392, r1635;
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
mov.f32 f33, 0fBF800000;
mov.f32 f34, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
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
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
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
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
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
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
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
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
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
mad.lo.s32 r1637, r1631, 392, r1636;
barrier.sync 0;
mad.lo.s32 r1638, r1633, 56, r1637;
st.shared.v2.f32 [r1638], {r32, r50};
st.shared.v2.f32 [r1638+8], {r711, r720};
st.shared.v2.f32 [r1638+16], {r748, r757};
st.shared.v2.f32 [r1638+24], {r785, r794};
st.shared.v2.f32 [r1638+32], {r822, r831};
st.shared.v2.f32 [r1638+40], {r859, r868};
st.shared.v2.f32 [r1638+48], {r896, r905};
barrier.sync 0;
mad.lo.s32 r1639, r1633, -48, r1638;
ld.shared.u32 r946, [r1639];
ld.shared.u32 r964, [r1639+4];
ld.shared.u32 r943, [r1639+56];
ld.shared.u32 r961, [r1639+60];
ld.shared.u32 r949, [r1639+112];
ld.shared.u32 r967, [r1639+116];
ld.shared.u32 r955, [r1639+168];
ld.shared.u32 r973, [r1639+172];
ld.shared.u32 r956, [r1639+224];
ld.shared.u32 r974, [r1639+228];
ld.shared.u32 r950, [r1639+280];
ld.shared.u32 r968, [r1639+284];
ld.shared.u32 r944, [r1639+336];
ld.shared.u32 r962, [r1639+340];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f54;
cvt.rn.f16.f32 high, f54;
mov.b32 r926, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r927, {low, high};
}
{
neg.f16x2 r928, r927;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f42;
cvt.rn.f16.f32 high, f42;
mov.b32 r930, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f44;
cvt.rn.f16.f32 high, f44;
mov.b32 r931, {low, high};
}
{
neg.f16x2 r932, r931;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r934, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r935, {low, high};
}
{
neg.f16x2 r936, r935;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r938, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r939, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f54;
cvt.rn.f16.f32 high, f54;
mov.b32 r940, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
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
add.f16x2 %0, r951, r954;
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
add.f16x2 %1, r969, r972;
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
sub.f16x2 %2, r1002, r1026;
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
add.f16x2 %12, r1056, r1080;
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
sub.f16x2 %4, r1110, r1134;
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
add.f16x2 %10, r1164, r1188;
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
sub.f16x2 %6, r1218, r1242;
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
add.f16x2 %8, r1272, r1296;
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
add.f16x2 %3, r1326, r1350;
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
sub.f16x2 %13, r1380, r1404;
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
add.f16x2 %5, r1434, r1458;
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
sub.f16x2 %11, r1488, r1512;
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
add.f16x2 %7, r1542, r1566;
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
sub.f16x2 %9, r1596, r1620;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1121, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<60>;
.reg .b32 r<1640>;
.reg .b64 rd<4>;
mov.u32 r1626, %tid.y;
mov.u32 r1627, %14;
mad.lo.s32 r1628, r1626, 196, r1627;
mov.u32 r1629, %tid.x;
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
{
neg.f16x2 r3, r2;
}
mov.f32 f42, 0fBE63DC87;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f42;
cvt.rn.f16.f32 high, f42;
mov.b32 r5, {low, high};
}
mov.f32 f44, 0fBF7994E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f44;
cvt.rn.f16.f32 high, f44;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
mov.f32 f50, 0fBF66A5E5;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r9, {low, high};
}
mov.f32 f52, 0fBEDE2602;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r10, {low, high};
}
{
neg.f16x2 r11, r10;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r13, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r14, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f54;
cvt.rn.f16.f32 high, f54;
mov.b32 r15, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
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
mul.wide.u32 rd2, r1629, 613566757;
shr.u64 rd3, rd2, 32;
cvt.u32.u64 r1630, rd3;
sub.s32 r1631, r1629, r1630;
shr.u32 r1632, r1631, 1;
add.s32 r1633, r1632, r1630;
shr.u32 r1634, r1633, 2;
mul.lo.s32 r1635, r1634, 7;
sub.s32 r1636, r1629, r1635;
mad.lo.s32 r1637, r1634, 196, r1628;
cvt.rn.f32.u32 f57, r1636;
mul.f32 f58, f57, 0f3E034E46;
cos.approx.f32 f21, f58;
sin.approx.f32 f59, f58;
neg.f32 f22, f59;
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
mov.f32 f33, 0fBF800000;
mov.f32 f34, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
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
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
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
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
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
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
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
cvt.rn.f16.f32 low, f33;
cvt.rn.f16.f32 high, f34;
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
barrier.sync 0;
mad.lo.s32 r1638, r1636, 28, r1637;
st.shared.u32 [r1638], r32;
st.shared.u32 [r1638+4], r711;
st.shared.u32 [r1638+8], r748;
st.shared.u32 [r1638+12], r785;
st.shared.u32 [r1638+16], r822;
st.shared.u32 [r1638+20], r859;
st.shared.u32 [r1638+24], r896;
barrier.sync 0;
mad.lo.s32 r1639, r1636, -24, r1638;
ld.shared.u32 r946, [r1639];
ld.shared.u32 r943, [r1639+28];
ld.shared.u32 r949, [r1639+56];
ld.shared.u32 r955, [r1639+84];
ld.shared.u32 r956, [r1639+112];
ld.shared.u32 r950, [r1639+140];
ld.shared.u32 r944, [r1639+168];
barrier.sync 0;
st.shared.u32 [r1638], r50;
st.shared.u32 [r1638+4], r720;
st.shared.u32 [r1638+8], r757;
st.shared.u32 [r1638+12], r794;
st.shared.u32 [r1638+16], r831;
st.shared.u32 [r1638+20], r868;
st.shared.u32 [r1638+24], r905;
barrier.sync 0;
ld.shared.u32 r964, [r1639];
ld.shared.u32 r961, [r1639+28];
ld.shared.u32 r967, [r1639+56];
ld.shared.u32 r973, [r1639+84];
ld.shared.u32 r974, [r1639+112];
ld.shared.u32 r968, [r1639+140];
ld.shared.u32 r962, [r1639+168];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f54;
cvt.rn.f16.f32 high, f54;
mov.b32 r926, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
mov.b32 r927, {low, high};
}
{
neg.f16x2 r928, r927;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f42;
cvt.rn.f16.f32 high, f42;
mov.b32 r930, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f44;
cvt.rn.f16.f32 high, f44;
mov.b32 r931, {low, high};
}
{
neg.f16x2 r932, r931;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r934, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r935, {low, high};
}
{
neg.f16x2 r936, r935;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f50;
cvt.rn.f16.f32 high, f50;
mov.b32 r938, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f52;
cvt.rn.f16.f32 high, f52;
mov.b32 r939, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f54;
cvt.rn.f16.f32 high, f54;
mov.b32 r940, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f56;
cvt.rn.f16.f32 high, f56;
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
add.f16x2 %0, r951, r954;
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
add.f16x2 %1, r969, r972;
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
sub.f16x2 %2, r1002, r1026;
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
add.f16x2 %12, r1056, r1080;
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
sub.f16x2 %4, r1110, r1134;
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
add.f16x2 %10, r1164, r1188;
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
sub.f16x2 %6, r1218, r1242;
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
add.f16x2 %8, r1272, r1296;
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
add.f16x2 %3, r1326, r1350;
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
sub.f16x2 %13, r1380, r1404;
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
add.f16x2 %5, r1434, r1458;
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
sub.f16x2 %11, r1488, r1512;
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
add.f16x2 %7, r1542, r1566;
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
sub.f16x2 %9, r1596, r1620;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)));
};


#endif
