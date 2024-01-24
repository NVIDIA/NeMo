#ifndef CUFFTDX_FFT_1331_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_1331_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1130, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .pred p<2>;
.reg .f32 f<718>;
.reg .b32 r<3374>;
.reg .b64 rd<8>;
{
add.f16x2 r25, %25, %43;
}
{
add.f16x2 r28, %26, %44;
}
{
sub.f16x2 r31, %25, %43;
}
{
sub.f16x2 r34, %26, %44;
}
{
add.f16x2 r37, %27, %41;
}
{
add.f16x2 r40, %28, %42;
}
{
sub.f16x2 r43, %27, %41;
}
{
sub.f16x2 r46, %28, %42;
}
{
add.f16x2 r49, %29, %39;
}
{
add.f16x2 r52, %30, %40;
}
{
sub.f16x2 r55, %29, %39;
}
{
sub.f16x2 r58, %30, %40;
}
{
add.f16x2 r61, %31, %37;
}
{
add.f16x2 r64, %32, %38;
}
{
sub.f16x2 r67, %31, %37;
}
{
sub.f16x2 r70, %32, %38;
}
{
add.f16x2 r73, %33, %35;
}
{
add.f16x2 r76, %34, %36;
}
{
sub.f16x2 r79, %33, %35;
}
{
sub.f16x2 r82, %34, %36;
}
{
add.f16x2 r85, %23, r25;
}
{
add.f16x2 r88, %24, r28;
}
{
add.f16x2 r91, r85, r37;
}
{
add.f16x2 r94, r88, r40;
}
{
add.f16x2 r97, r91, r49;
}
{
add.f16x2 r100, r94, r52;
}
{
add.f16x2 r103, r97, r61;
}
{
add.f16x2 r106, r100, r64;
}
{
add.f16x2 r109, r103, r73;
}
{
add.f16x2 r112, r106, r76;
}
mov.f32 f183, 0f00000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r115, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r116, {low, high};
}
mov.f32 f197, 0f3F575C64;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r117, {low, high};
}
{
mul.f16x2 r118, r25, r117;
}
{
add.f16x2 r121, %23, r118;
}
mov.f32 f163, 0f3F0A6770;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r124, {low, high};
}
{
mul.f16x2 r125, r34, r124;
}
{
add.f16x2 r128, r115, r125;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r131, {low, high};
}
{
mul.f16x2 r132, r28, r131;
}
{
add.f16x2 r135, %24, r132;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r138, {low, high};
}
{
mul.f16x2 r139, r31, r138;
}
{
add.f16x2 r142, r116, r139;
}
mov.f32 f213, 0f3ED4B147;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r145, {low, high};
}
{
mul.f16x2 r146, r37, r145;
}
{
add.f16x2 r149, r121, r146;
}
mov.f32 f59, 0f3F68DDA4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r152, {low, high};
}
{
mul.f16x2 r153, r46, r152;
}
{
add.f16x2 r156, r128, r153;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r159, {low, high};
}
{
mul.f16x2 r160, r40, r159;
}
{
add.f16x2 r163, r135, r160;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r166, {low, high};
}
{
mul.f16x2 r167, r43, r166;
}
{
add.f16x2 r170, r142, r167;
}
mov.f32 f221, 0fBE11BAFB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r173, {low, high};
}
{
mul.f16x2 r174, r49, r173;
}
{
add.f16x2 r177, r149, r174;
}
mov.f32 f223, 0f3F7D64F0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r180, {low, high};
}
{
mul.f16x2 r181, r58, r180;
}
{
add.f16x2 r184, r156, r181;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r187, {low, high};
}
{
mul.f16x2 r188, r52, r187;
}
{
add.f16x2 r191, r163, r188;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r194, {low, high};
}
{
mul.f16x2 r195, r55, r194;
}
{
add.f16x2 r198, r170, r195;
}
mov.f32 f205, 0fBF27A4F4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r201, {low, high};
}
{
mul.f16x2 r202, r61, r201;
}
{
add.f16x2 r205, r177, r202;
}
mov.f32 f207, 0f3F4178CE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r208, {low, high};
}
{
mul.f16x2 r209, r70, r208;
}
{
add.f16x2 r212, r184, r209;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r215, {low, high};
}
{
mul.f16x2 r216, r64, r215;
}
{
add.f16x2 r219, r191, r216;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r222, {low, high};
}
{
mul.f16x2 r223, r67, r222;
}
{
add.f16x2 r226, r198, r223;
}
mov.f32 f189, 0fBF75A155;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r229, {low, high};
}
{
mul.f16x2 r230, r73, r229;
}
{
add.f16x2 r233, r205, r230;
}
mov.f32 f191, 0f3E903F40;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r236, {low, high};
}
{
mul.f16x2 r237, r82, r236;
}
{
add.f16x2 r240, r212, r237;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r243, {low, high};
}
{
mul.f16x2 r244, r76, r243;
}
{
add.f16x2 r247, r219, r244;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r250, {low, high};
}
{
mul.f16x2 r251, r79, r250;
}
{
add.f16x2 r254, r226, r251;
}
{
sub.f16x2 r257, r233, r240;
}
{
add.f16x2 r260, r247, r254;
}
{
add.f16x2 r263, r233, r240;
}
{
sub.f16x2 r266, r247, r254;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r269, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r270, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r271, {low, high};
}
{
mul.f16x2 r272, r25, r271;
}
{
add.f16x2 r275, %23, r272;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r278, {low, high};
}
{
mul.f16x2 r279, r34, r278;
}
{
add.f16x2 r282, r269, r279;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r285, {low, high};
}
{
mul.f16x2 r286, r28, r285;
}
{
add.f16x2 r289, %24, r286;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r292, {low, high};
}
{
mul.f16x2 r293, r31, r292;
}
{
add.f16x2 r296, r270, r293;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r299, {low, high};
}
{
mul.f16x2 r300, r37, r299;
}
{
add.f16x2 r303, r275, r300;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r306, {low, high};
}
{
mul.f16x2 r307, r46, r306;
}
{
add.f16x2 r310, r282, r307;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r313, {low, high};
}
{
mul.f16x2 r314, r40, r313;
}
{
add.f16x2 r317, r289, r314;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r320, {low, high};
}
{
mul.f16x2 r321, r43, r320;
}
{
add.f16x2 r324, r296, r321;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r327, {low, high};
}
{
mul.f16x2 r328, r49, r327;
}
{
add.f16x2 r331, r303, r328;
}
mov.f32 f111, 0fBE903F40;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r334, {low, high};
}
{
mul.f16x2 r335, r58, r334;
}
{
add.f16x2 r338, r310, r335;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r341, {low, high};
}
{
mul.f16x2 r342, r52, r341;
}
{
add.f16x2 r345, r317, r342;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r348, {low, high};
}
{
mul.f16x2 r349, r55, r348;
}
{
add.f16x2 r352, r324, r349;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r355, {low, high};
}
{
mul.f16x2 r356, r61, r355;
}
{
add.f16x2 r359, r331, r356;
}
mov.f32 f155, 0fBF7D64F0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r362, {low, high};
}
{
mul.f16x2 r363, r70, r362;
}
{
add.f16x2 r366, r338, r363;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r369, {low, high};
}
{
mul.f16x2 r370, r64, r369;
}
{
add.f16x2 r373, r345, r370;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r376, {low, high};
}
{
mul.f16x2 r377, r67, r376;
}
{
add.f16x2 r380, r352, r377;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r383, {low, high};
}
{
mul.f16x2 r384, r73, r383;
}
{
add.f16x2 r387, r359, r384;
}
mov.f32 f199, 0fBF0A6770;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r390, {low, high};
}
{
mul.f16x2 r391, r82, r390;
}
{
add.f16x2 r394, r366, r391;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r397, {low, high};
}
{
mul.f16x2 r398, r76, r397;
}
{
add.f16x2 r401, r373, r398;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r404, {low, high};
}
{
mul.f16x2 r405, r79, r404;
}
{
add.f16x2 r408, r380, r405;
}
{
sub.f16x2 r411, r387, r394;
}
{
add.f16x2 r414, r401, r408;
}
{
add.f16x2 r417, r387, r394;
}
{
sub.f16x2 r420, r401, r408;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r423, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r424, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r425, {low, high};
}
{
mul.f16x2 r426, r25, r425;
}
{
add.f16x2 r429, %23, r426;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r432, {low, high};
}
{
mul.f16x2 r433, r34, r432;
}
{
add.f16x2 r436, r423, r433;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r439, {low, high};
}
{
mul.f16x2 r440, r28, r439;
}
{
add.f16x2 r443, %24, r440;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r446, {low, high};
}
{
mul.f16x2 r447, r31, r446;
}
{
add.f16x2 r450, r424, r447;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r453, {low, high};
}
{
mul.f16x2 r454, r37, r453;
}
{
add.f16x2 r457, r429, r454;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r460, {low, high};
}
{
mul.f16x2 r461, r46, r460;
}
{
add.f16x2 r464, r436, r461;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r467, {low, high};
}
{
mul.f16x2 r468, r40, r467;
}
{
add.f16x2 r471, r443, r468;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r474, {low, high};
}
{
mul.f16x2 r475, r43, r474;
}
{
add.f16x2 r478, r450, r475;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r481, {low, high};
}
{
mul.f16x2 r482, r49, r481;
}
{
add.f16x2 r485, r457, r482;
}
mov.f32 f215, 0fBF68DDA4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r488, {low, high};
}
{
mul.f16x2 r489, r58, r488;
}
{
add.f16x2 r492, r464, r489;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r495, {low, high};
}
{
mul.f16x2 r496, r52, r495;
}
{
add.f16x2 r499, r471, r496;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r502, {low, high};
}
{
mul.f16x2 r503, r55, r502;
}
{
add.f16x2 r506, r478, r503;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r509, {low, high};
}
{
mul.f16x2 r510, r61, r509;
}
{
add.f16x2 r513, r485, r510;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r516, {low, high};
}
{
mul.f16x2 r517, r70, r516;
}
{
add.f16x2 r520, r492, r517;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r523, {low, high};
}
{
mul.f16x2 r524, r64, r523;
}
{
add.f16x2 r527, r499, r524;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r530, {low, high};
}
{
mul.f16x2 r531, r67, r530;
}
{
add.f16x2 r534, r506, r531;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r537, {low, high};
}
{
mul.f16x2 r538, r73, r537;
}
{
add.f16x2 r541, r513, r538;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r544, {low, high};
}
{
mul.f16x2 r545, r82, r544;
}
{
add.f16x2 r548, r520, r545;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r551, {low, high};
}
{
mul.f16x2 r552, r76, r551;
}
{
add.f16x2 r555, r527, r552;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r558, {low, high};
}
{
mul.f16x2 r559, r79, r558;
}
{
add.f16x2 r562, r534, r559;
}
{
sub.f16x2 r565, r541, r548;
}
{
add.f16x2 r568, r555, r562;
}
{
add.f16x2 r571, r541, r548;
}
{
sub.f16x2 r574, r555, r562;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r577, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r578, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r579, {low, high};
}
{
mul.f16x2 r580, r25, r579;
}
{
add.f16x2 r583, %23, r580;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r586, {low, high};
}
{
mul.f16x2 r587, r34, r586;
}
{
add.f16x2 r590, r577, r587;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r593, {low, high};
}
{
mul.f16x2 r594, r28, r593;
}
{
add.f16x2 r597, %24, r594;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r600, {low, high};
}
{
mul.f16x2 r601, r31, r600;
}
{
add.f16x2 r604, r578, r601;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r607, {low, high};
}
{
mul.f16x2 r608, r37, r607;
}
{
add.f16x2 r611, r583, r608;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r614, {low, high};
}
{
mul.f16x2 r615, r46, r614;
}
{
add.f16x2 r618, r590, r615;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r621, {low, high};
}
{
mul.f16x2 r622, r40, r621;
}
{
add.f16x2 r625, r597, r622;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r628, {low, high};
}
{
mul.f16x2 r629, r43, r628;
}
{
add.f16x2 r632, r604, r629;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r635, {low, high};
}
{
mul.f16x2 r636, r49, r635;
}
{
add.f16x2 r639, r611, r636;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r642, {low, high};
}
{
mul.f16x2 r643, r58, r642;
}
{
add.f16x2 r646, r618, r643;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r649, {low, high};
}
{
mul.f16x2 r650, r52, r649;
}
{
add.f16x2 r653, r625, r650;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r656, {low, high};
}
{
mul.f16x2 r657, r55, r656;
}
{
add.f16x2 r660, r632, r657;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r663, {low, high};
}
{
mul.f16x2 r664, r61, r663;
}
{
add.f16x2 r667, r639, r664;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r670, {low, high};
}
{
mul.f16x2 r671, r70, r670;
}
{
add.f16x2 r674, r646, r671;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r677, {low, high};
}
{
mul.f16x2 r678, r64, r677;
}
{
add.f16x2 r681, r653, r678;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r684, {low, high};
}
{
mul.f16x2 r685, r67, r684;
}
{
add.f16x2 r688, r660, r685;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r691, {low, high};
}
{
mul.f16x2 r692, r73, r691;
}
{
add.f16x2 r695, r667, r692;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r698, {low, high};
}
{
mul.f16x2 r699, r82, r698;
}
{
add.f16x2 r702, r674, r699;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r705, {low, high};
}
{
mul.f16x2 r706, r76, r705;
}
{
add.f16x2 r709, r681, r706;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r712, {low, high};
}
{
mul.f16x2 r713, r79, r712;
}
{
add.f16x2 r716, r688, r713;
}
{
sub.f16x2 r719, r695, r702;
}
{
add.f16x2 r722, r709, r716;
}
{
add.f16x2 r725, r695, r702;
}
{
sub.f16x2 r728, r709, r716;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r731, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r732, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r733, {low, high};
}
{
mul.f16x2 r734, r25, r733;
}
{
add.f16x2 r737, %23, r734;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r740, {low, high};
}
{
mul.f16x2 r741, r34, r740;
}
{
add.f16x2 r744, r731, r741;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r747, {low, high};
}
{
mul.f16x2 r748, r28, r747;
}
{
add.f16x2 r751, %24, r748;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r754, {low, high};
}
{
mul.f16x2 r755, r31, r754;
}
{
add.f16x2 r758, r732, r755;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r761, {low, high};
}
{
mul.f16x2 r762, r37, r761;
}
{
add.f16x2 r765, r737, r762;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r768, {low, high};
}
{
mul.f16x2 r769, r46, r768;
}
{
add.f16x2 r772, r744, r769;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r775, {low, high};
}
{
mul.f16x2 r776, r40, r775;
}
{
add.f16x2 r779, r751, r776;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r782, {low, high};
}
{
mul.f16x2 r783, r43, r782;
}
{
add.f16x2 r786, r758, r783;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r789, {low, high};
}
{
mul.f16x2 r790, r49, r789;
}
{
add.f16x2 r793, r765, r790;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r796, {low, high};
}
{
mul.f16x2 r797, r58, r796;
}
{
add.f16x2 r800, r772, r797;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r803, {low, high};
}
{
mul.f16x2 r804, r52, r803;
}
{
add.f16x2 r807, r779, r804;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r810, {low, high};
}
{
mul.f16x2 r811, r55, r810;
}
{
add.f16x2 r814, r786, r811;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r817, {low, high};
}
{
mul.f16x2 r818, r61, r817;
}
{
add.f16x2 r821, r793, r818;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r824, {low, high};
}
{
mul.f16x2 r825, r70, r824;
}
{
add.f16x2 r828, r800, r825;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r831, {low, high};
}
{
mul.f16x2 r832, r64, r831;
}
{
add.f16x2 r835, r807, r832;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r838, {low, high};
}
{
mul.f16x2 r839, r67, r838;
}
{
add.f16x2 r842, r814, r839;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r845, {low, high};
}
{
mul.f16x2 r846, r73, r845;
}
{
add.f16x2 r849, r821, r846;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r852, {low, high};
}
{
mul.f16x2 r853, r82, r852;
}
{
add.f16x2 r856, r828, r853;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r859, {low, high};
}
{
mul.f16x2 r860, r76, r859;
}
{
add.f16x2 r863, r835, r860;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r866, {low, high};
}
{
mul.f16x2 r867, r79, r866;
}
{
add.f16x2 r870, r842, r867;
}
{
sub.f16x2 r873, r849, r856;
}
{
add.f16x2 r876, r863, r870;
}
{
add.f16x2 r879, r849, r856;
}
{
sub.f16x2 r882, r863, r870;
}
mov.u32 r23, %tid.x;
mul.wide.u32 rd2, r23, 248469183;
shr.u64 rd3, rd2, 32;
cvt.u32.u64 r885, rd3;
sub.s32 r886, r23, r885;
shr.u32 r887, r886, 1;
add.s32 r888, r887, r885;
shr.u32 r889, r888, 6;
mul.lo.s32 r890, r889, 121;
sub.s32 r24, r23, r890;
cvt.rn.f32.u32 f225, r24;
mul.f32 f1, f225, 0f3B9AAFAF;
setp.eq.s32 p1, r24, 18;
mov.f32 f717, 0f3DADD00E;
@p1 bra LBB0_2;
sin.approx.f32 f717, f1;
LBB0_2:
mov.u32 r3362, %tid.y;
mov.u32 r3363, %22;
mad.lo.s32 r3364, r3362, 10648, r3363;
mad.lo.s32 r3365, r889, 10648, r3364;
neg.f32 f227, f717;
cos.approx.f32 f226, f1;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f226;
cvt.rn.f16.f32 high, f227;
mov.b32 r891, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r894, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r896, {high, high};
}
{
mul.f16x2 r898, r260, r896;
}
{
fma.rn.f16x2 r901, r257, r894, r898;
}
{
mul.f16x2 r905, r257, r896;
}
{
neg.f16x2 r908, r905;
}
{
fma.rn.f16x2 r910, r260, r894, r908;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r914, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r916, {high, high};
}
mov.f32 f490, 0fBF800000;
mov.f32 f491, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r918, {low, high};
}
{
mul.f16x2 r919, r916, r918;
}
{
mul.f16x2 r922, r891, r914;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
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
mul.f16x2 r935, r414, r933;
}
{
fma.rn.f16x2 r938, r411, r931, r935;
}
{
mul.f16x2 r942, r411, r933;
}
{
neg.f16x2 r945, r942;
}
{
fma.rn.f16x2 r947, r414, r931, r945;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r951, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r953, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
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
mul.f16x2 r972, r568, r970;
}
{
fma.rn.f16x2 r975, r565, r968, r972;
}
{
mul.f16x2 r979, r565, r970;
}
{
neg.f16x2 r982, r979;
}
{
fma.rn.f16x2 r984, r568, r968, r982;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r988, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r990, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
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
mul.f16x2 r1009, r722, r1007;
}
{
fma.rn.f16x2 r1012, r719, r1005, r1009;
}
{
mul.f16x2 r1016, r719, r1007;
}
{
neg.f16x2 r1019, r1016;
}
{
fma.rn.f16x2 r1021, r722, r1005, r1019;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1025, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1027, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
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
mul.f16x2 r1046, r876, r1044;
}
{
fma.rn.f16x2 r1049, r873, r1042, r1046;
}
{
mul.f16x2 r1053, r873, r1044;
}
{
neg.f16x2 r1056, r1053;
}
{
fma.rn.f16x2 r1058, r876, r1042, r1056;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1062, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1064, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r1066, {low, high};
}
{
mul.f16x2 r1067, r1064, r1066;
}
{
mul.f16x2 r1070, r1038, r1062;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1038;
mov.b32 r1073, {high, low};
}
{
fma.rn.f16x2 r1075, r1067, r1073, r1070;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1075;
mov.b32 r1079, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1075;
mov.b32 r1081, {high, high};
}
{
mul.f16x2 r1083, r882, r1081;
}
{
fma.rn.f16x2 r1086, r879, r1079, r1083;
}
{
mul.f16x2 r1090, r879, r1081;
}
{
neg.f16x2 r1093, r1090;
}
{
fma.rn.f16x2 r1095, r882, r1079, r1093;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1099, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1101, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r1103, {low, high};
}
{
mul.f16x2 r1104, r1101, r1103;
}
{
mul.f16x2 r1107, r1075, r1099;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1075;
mov.b32 r1110, {high, low};
}
{
fma.rn.f16x2 r1112, r1104, r1110, r1107;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1112;
mov.b32 r1116, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1112;
mov.b32 r1118, {high, high};
}
{
mul.f16x2 r1120, r728, r1118;
}
{
fma.rn.f16x2 r1123, r725, r1116, r1120;
}
{
mul.f16x2 r1127, r725, r1118;
}
{
neg.f16x2 r1130, r1127;
}
{
fma.rn.f16x2 r1132, r728, r1116, r1130;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1136, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1138, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r1140, {low, high};
}
{
mul.f16x2 r1141, r1138, r1140;
}
{
mul.f16x2 r1144, r1112, r1136;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1112;
mov.b32 r1147, {high, low};
}
{
fma.rn.f16x2 r1149, r1141, r1147, r1144;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1149;
mov.b32 r1153, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1149;
mov.b32 r1155, {high, high};
}
{
mul.f16x2 r1157, r574, r1155;
}
{
fma.rn.f16x2 r1160, r571, r1153, r1157;
}
{
mul.f16x2 r1164, r571, r1155;
}
{
neg.f16x2 r1167, r1164;
}
{
fma.rn.f16x2 r1169, r574, r1153, r1167;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1173, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1175, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r1177, {low, high};
}
{
mul.f16x2 r1178, r1175, r1177;
}
{
mul.f16x2 r1181, r1149, r1173;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1149;
mov.b32 r1184, {high, low};
}
{
fma.rn.f16x2 r1186, r1178, r1184, r1181;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1186;
mov.b32 r1190, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1186;
mov.b32 r1192, {high, high};
}
{
mul.f16x2 r1194, r420, r1192;
}
{
fma.rn.f16x2 r1197, r417, r1190, r1194;
}
{
mul.f16x2 r1201, r417, r1192;
}
{
neg.f16x2 r1204, r1201;
}
{
fma.rn.f16x2 r1206, r420, r1190, r1204;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1210, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1212, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r1214, {low, high};
}
{
mul.f16x2 r1215, r1212, r1214;
}
{
mul.f16x2 r1218, r1186, r1210;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1186;
mov.b32 r1221, {high, low};
}
{
fma.rn.f16x2 r1223, r1215, r1221, r1218;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1223;
mov.b32 r1227, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1223;
mov.b32 r1229, {high, high};
}
{
mul.f16x2 r1231, r266, r1229;
}
{
fma.rn.f16x2 r1234, r263, r1227, r1231;
}
{
mul.f16x2 r1238, r263, r1229;
}
{
neg.f16x2 r1241, r1238;
}
{
fma.rn.f16x2 r1243, r266, r1227, r1241;
}
barrier.sync 0;
mad.lo.s32 r3366, r24, 88, r3365;
st.shared.v2.f32 [r3366], {r109, r112};
st.shared.v2.f32 [r3366+8], {r901, r910};
st.shared.v2.f32 [r3366+16], {r938, r947};
st.shared.v2.f32 [r3366+24], {r975, r984};
st.shared.v2.f32 [r3366+32], {r1012, r1021};
st.shared.v2.f32 [r3366+40], {r1049, r1058};
st.shared.v2.f32 [r3366+48], {r1086, r1095};
st.shared.v2.f32 [r3366+56], {r1123, r1132};
st.shared.v2.f32 [r3366+64], {r1160, r1169};
st.shared.v2.f32 [r3366+72], {r1197, r1206};
st.shared.v2.f32 [r3366+80], {r1234, r1243};
barrier.sync 0;
mad.lo.s32 r3367, r24, -80, r3366;
ld.shared.u32 r1325, [r3367];
ld.shared.u32 r1328, [r3367+4];
ld.shared.u32 r1265, [r3367+968];
ld.shared.u32 r1268, [r3367+972];
ld.shared.u32 r1277, [r3367+1936];
ld.shared.u32 r1280, [r3367+1940];
ld.shared.u32 r1289, [r3367+2904];
ld.shared.u32 r1292, [r3367+2908];
ld.shared.u32 r1301, [r3367+3872];
ld.shared.u32 r1304, [r3367+3876];
ld.shared.u32 r1313, [r3367+4840];
ld.shared.u32 r1316, [r3367+4844];
ld.shared.u32 r1314, [r3367+5808];
ld.shared.u32 r1317, [r3367+5812];
ld.shared.u32 r1302, [r3367+6776];
ld.shared.u32 r1305, [r3367+6780];
ld.shared.u32 r1290, [r3367+7744];
ld.shared.u32 r1293, [r3367+7748];
ld.shared.u32 r1278, [r3367+8712];
ld.shared.u32 r1281, [r3367+8716];
ld.shared.u32 r1266, [r3367+9680];
ld.shared.u32 r1269, [r3367+9684];
{
add.f16x2 r1264, r1265, r1266;
}
{
add.f16x2 r1267, r1268, r1269;
}
{
sub.f16x2 r1270, r1265, r1266;
}
{
sub.f16x2 r1273, r1268, r1269;
}
{
add.f16x2 r1276, r1277, r1278;
}
{
add.f16x2 r1279, r1280, r1281;
}
{
sub.f16x2 r1282, r1277, r1278;
}
{
sub.f16x2 r1285, r1280, r1281;
}
{
add.f16x2 r1288, r1289, r1290;
}
{
add.f16x2 r1291, r1292, r1293;
}
{
sub.f16x2 r1294, r1289, r1290;
}
{
sub.f16x2 r1297, r1292, r1293;
}
{
add.f16x2 r1300, r1301, r1302;
}
{
add.f16x2 r1303, r1304, r1305;
}
{
sub.f16x2 r1306, r1301, r1302;
}
{
sub.f16x2 r1309, r1304, r1305;
}
{
add.f16x2 r1312, r1313, r1314;
}
{
add.f16x2 r1315, r1316, r1317;
}
{
sub.f16x2 r1318, r1313, r1314;
}
{
sub.f16x2 r1321, r1316, r1317;
}
{
add.f16x2 r1324, r1325, r1264;
}
{
add.f16x2 r1327, r1328, r1267;
}
{
add.f16x2 r1330, r1324, r1276;
}
{
add.f16x2 r1333, r1327, r1279;
}
{
add.f16x2 r1336, r1330, r1288;
}
{
add.f16x2 r1339, r1333, r1291;
}
{
add.f16x2 r1342, r1336, r1300;
}
{
add.f16x2 r1345, r1339, r1303;
}
{
add.f16x2 r1348, r1342, r1312;
}
{
add.f16x2 r1351, r1345, r1315;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r1354, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r1355, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r1356, {low, high};
}
{
mul.f16x2 r1357, r1264, r1356;
}
{
add.f16x2 r1360, r1325, r1357;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r1363, {low, high};
}
{
mul.f16x2 r1364, r1273, r1363;
}
{
add.f16x2 r1367, r1354, r1364;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r1370, {low, high};
}
{
mul.f16x2 r1371, r1267, r1370;
}
{
add.f16x2 r1374, r1328, r1371;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r1377, {low, high};
}
{
mul.f16x2 r1378, r1270, r1377;
}
{
add.f16x2 r1381, r1355, r1378;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r1384, {low, high};
}
{
mul.f16x2 r1385, r1276, r1384;
}
{
add.f16x2 r1388, r1360, r1385;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r1391, {low, high};
}
{
mul.f16x2 r1392, r1285, r1391;
}
{
add.f16x2 r1395, r1367, r1392;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r1398, {low, high};
}
{
mul.f16x2 r1399, r1279, r1398;
}
{
add.f16x2 r1402, r1374, r1399;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r1405, {low, high};
}
{
mul.f16x2 r1406, r1282, r1405;
}
{
add.f16x2 r1409, r1381, r1406;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r1412, {low, high};
}
{
mul.f16x2 r1413, r1288, r1412;
}
{
add.f16x2 r1416, r1388, r1413;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r1419, {low, high};
}
{
mul.f16x2 r1420, r1297, r1419;
}
{
add.f16x2 r1423, r1395, r1420;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r1426, {low, high};
}
{
mul.f16x2 r1427, r1291, r1426;
}
{
add.f16x2 r1430, r1402, r1427;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r1433, {low, high};
}
{
mul.f16x2 r1434, r1294, r1433;
}
{
add.f16x2 r1437, r1409, r1434;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r1440, {low, high};
}
{
mul.f16x2 r1441, r1300, r1440;
}
{
add.f16x2 r1444, r1416, r1441;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r1447, {low, high};
}
{
mul.f16x2 r1448, r1309, r1447;
}
{
add.f16x2 r1451, r1423, r1448;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r1454, {low, high};
}
{
mul.f16x2 r1455, r1303, r1454;
}
{
add.f16x2 r1458, r1430, r1455;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r1461, {low, high};
}
{
mul.f16x2 r1462, r1306, r1461;
}
{
add.f16x2 r1465, r1437, r1462;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r1468, {low, high};
}
{
mul.f16x2 r1469, r1312, r1468;
}
{
add.f16x2 r1472, r1444, r1469;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r1475, {low, high};
}
{
mul.f16x2 r1476, r1321, r1475;
}
{
add.f16x2 r1479, r1451, r1476;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r1482, {low, high};
}
{
mul.f16x2 r1483, r1315, r1482;
}
{
add.f16x2 r1486, r1458, r1483;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r1489, {low, high};
}
{
mul.f16x2 r1490, r1318, r1489;
}
{
add.f16x2 r1493, r1465, r1490;
}
{
sub.f16x2 r1496, r1472, r1479;
}
{
add.f16x2 r1499, r1486, r1493;
}
{
add.f16x2 r1502, r1472, r1479;
}
{
sub.f16x2 r1505, r1486, r1493;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r1508, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r1509, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r1510, {low, high};
}
{
mul.f16x2 r1511, r1264, r1510;
}
{
add.f16x2 r1514, r1325, r1511;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r1517, {low, high};
}
{
mul.f16x2 r1518, r1273, r1517;
}
{
add.f16x2 r1521, r1508, r1518;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r1524, {low, high};
}
{
mul.f16x2 r1525, r1267, r1524;
}
{
add.f16x2 r1528, r1328, r1525;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r1531, {low, high};
}
{
mul.f16x2 r1532, r1270, r1531;
}
{
add.f16x2 r1535, r1509, r1532;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r1538, {low, high};
}
{
mul.f16x2 r1539, r1276, r1538;
}
{
add.f16x2 r1542, r1514, r1539;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r1545, {low, high};
}
{
mul.f16x2 r1546, r1285, r1545;
}
{
add.f16x2 r1549, r1521, r1546;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r1552, {low, high};
}
{
mul.f16x2 r1553, r1279, r1552;
}
{
add.f16x2 r1556, r1528, r1553;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r1559, {low, high};
}
{
mul.f16x2 r1560, r1282, r1559;
}
{
add.f16x2 r1563, r1535, r1560;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r1566, {low, high};
}
{
mul.f16x2 r1567, r1288, r1566;
}
{
add.f16x2 r1570, r1542, r1567;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r1573, {low, high};
}
{
mul.f16x2 r1574, r1297, r1573;
}
{
add.f16x2 r1577, r1549, r1574;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r1580, {low, high};
}
{
mul.f16x2 r1581, r1291, r1580;
}
{
add.f16x2 r1584, r1556, r1581;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r1587, {low, high};
}
{
mul.f16x2 r1588, r1294, r1587;
}
{
add.f16x2 r1591, r1563, r1588;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r1594, {low, high};
}
{
mul.f16x2 r1595, r1300, r1594;
}
{
add.f16x2 r1598, r1570, r1595;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r1601, {low, high};
}
{
mul.f16x2 r1602, r1309, r1601;
}
{
add.f16x2 r1605, r1577, r1602;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r1608, {low, high};
}
{
mul.f16x2 r1609, r1303, r1608;
}
{
add.f16x2 r1612, r1584, r1609;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r1615, {low, high};
}
{
mul.f16x2 r1616, r1306, r1615;
}
{
add.f16x2 r1619, r1591, r1616;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r1622, {low, high};
}
{
mul.f16x2 r1623, r1312, r1622;
}
{
add.f16x2 r1626, r1598, r1623;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r1629, {low, high};
}
{
mul.f16x2 r1630, r1321, r1629;
}
{
add.f16x2 r1633, r1605, r1630;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r1636, {low, high};
}
{
mul.f16x2 r1637, r1315, r1636;
}
{
add.f16x2 r1640, r1612, r1637;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r1643, {low, high};
}
{
mul.f16x2 r1644, r1318, r1643;
}
{
add.f16x2 r1647, r1619, r1644;
}
{
sub.f16x2 r1650, r1626, r1633;
}
{
add.f16x2 r1653, r1640, r1647;
}
{
add.f16x2 r1656, r1626, r1633;
}
{
sub.f16x2 r1659, r1640, r1647;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r1662, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r1663, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r1664, {low, high};
}
{
mul.f16x2 r1665, r1264, r1664;
}
{
add.f16x2 r1668, r1325, r1665;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r1671, {low, high};
}
{
mul.f16x2 r1672, r1273, r1671;
}
{
add.f16x2 r1675, r1662, r1672;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r1678, {low, high};
}
{
mul.f16x2 r1679, r1267, r1678;
}
{
add.f16x2 r1682, r1328, r1679;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r1685, {low, high};
}
{
mul.f16x2 r1686, r1270, r1685;
}
{
add.f16x2 r1689, r1663, r1686;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r1692, {low, high};
}
{
mul.f16x2 r1693, r1276, r1692;
}
{
add.f16x2 r1696, r1668, r1693;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r1699, {low, high};
}
{
mul.f16x2 r1700, r1285, r1699;
}
{
add.f16x2 r1703, r1675, r1700;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r1706, {low, high};
}
{
mul.f16x2 r1707, r1279, r1706;
}
{
add.f16x2 r1710, r1682, r1707;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r1713, {low, high};
}
{
mul.f16x2 r1714, r1282, r1713;
}
{
add.f16x2 r1717, r1689, r1714;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r1720, {low, high};
}
{
mul.f16x2 r1721, r1288, r1720;
}
{
add.f16x2 r1724, r1696, r1721;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r1727, {low, high};
}
{
mul.f16x2 r1728, r1297, r1727;
}
{
add.f16x2 r1731, r1703, r1728;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r1734, {low, high};
}
{
mul.f16x2 r1735, r1291, r1734;
}
{
add.f16x2 r1738, r1710, r1735;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r1741, {low, high};
}
{
mul.f16x2 r1742, r1294, r1741;
}
{
add.f16x2 r1745, r1717, r1742;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r1748, {low, high};
}
{
mul.f16x2 r1749, r1300, r1748;
}
{
add.f16x2 r1752, r1724, r1749;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r1755, {low, high};
}
{
mul.f16x2 r1756, r1309, r1755;
}
{
add.f16x2 r1759, r1731, r1756;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r1762, {low, high};
}
{
mul.f16x2 r1763, r1303, r1762;
}
{
add.f16x2 r1766, r1738, r1763;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r1769, {low, high};
}
{
mul.f16x2 r1770, r1306, r1769;
}
{
add.f16x2 r1773, r1745, r1770;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r1776, {low, high};
}
{
mul.f16x2 r1777, r1312, r1776;
}
{
add.f16x2 r1780, r1752, r1777;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r1783, {low, high};
}
{
mul.f16x2 r1784, r1321, r1783;
}
{
add.f16x2 r1787, r1759, r1784;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r1790, {low, high};
}
{
mul.f16x2 r1791, r1315, r1790;
}
{
add.f16x2 r1794, r1766, r1791;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r1797, {low, high};
}
{
mul.f16x2 r1798, r1318, r1797;
}
{
add.f16x2 r1801, r1773, r1798;
}
{
sub.f16x2 r1804, r1780, r1787;
}
{
add.f16x2 r1807, r1794, r1801;
}
{
add.f16x2 r1810, r1780, r1787;
}
{
sub.f16x2 r1813, r1794, r1801;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r1816, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r1817, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r1818, {low, high};
}
{
mul.f16x2 r1819, r1264, r1818;
}
{
add.f16x2 r1822, r1325, r1819;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r1825, {low, high};
}
{
mul.f16x2 r1826, r1273, r1825;
}
{
add.f16x2 r1829, r1816, r1826;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r1832, {low, high};
}
{
mul.f16x2 r1833, r1267, r1832;
}
{
add.f16x2 r1836, r1328, r1833;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r1839, {low, high};
}
{
mul.f16x2 r1840, r1270, r1839;
}
{
add.f16x2 r1843, r1817, r1840;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r1846, {low, high};
}
{
mul.f16x2 r1847, r1276, r1846;
}
{
add.f16x2 r1850, r1822, r1847;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r1853, {low, high};
}
{
mul.f16x2 r1854, r1285, r1853;
}
{
add.f16x2 r1857, r1829, r1854;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r1860, {low, high};
}
{
mul.f16x2 r1861, r1279, r1860;
}
{
add.f16x2 r1864, r1836, r1861;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r1867, {low, high};
}
{
mul.f16x2 r1868, r1282, r1867;
}
{
add.f16x2 r1871, r1843, r1868;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r1874, {low, high};
}
{
mul.f16x2 r1875, r1288, r1874;
}
{
add.f16x2 r1878, r1850, r1875;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r1881, {low, high};
}
{
mul.f16x2 r1882, r1297, r1881;
}
{
add.f16x2 r1885, r1857, r1882;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r1888, {low, high};
}
{
mul.f16x2 r1889, r1291, r1888;
}
{
add.f16x2 r1892, r1864, r1889;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r1895, {low, high};
}
{
mul.f16x2 r1896, r1294, r1895;
}
{
add.f16x2 r1899, r1871, r1896;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r1902, {low, high};
}
{
mul.f16x2 r1903, r1300, r1902;
}
{
add.f16x2 r1906, r1878, r1903;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r1909, {low, high};
}
{
mul.f16x2 r1910, r1309, r1909;
}
{
add.f16x2 r1913, r1885, r1910;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r1916, {low, high};
}
{
mul.f16x2 r1917, r1303, r1916;
}
{
add.f16x2 r1920, r1892, r1917;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r1923, {low, high};
}
{
mul.f16x2 r1924, r1306, r1923;
}
{
add.f16x2 r1927, r1899, r1924;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r1930, {low, high};
}
{
mul.f16x2 r1931, r1312, r1930;
}
{
add.f16x2 r1934, r1906, r1931;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r1937, {low, high};
}
{
mul.f16x2 r1938, r1321, r1937;
}
{
add.f16x2 r1941, r1913, r1938;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r1944, {low, high};
}
{
mul.f16x2 r1945, r1315, r1944;
}
{
add.f16x2 r1948, r1920, r1945;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r1951, {low, high};
}
{
mul.f16x2 r1952, r1318, r1951;
}
{
add.f16x2 r1955, r1927, r1952;
}
{
sub.f16x2 r1958, r1934, r1941;
}
{
add.f16x2 r1961, r1948, r1955;
}
{
add.f16x2 r1964, r1934, r1941;
}
{
sub.f16x2 r1967, r1948, r1955;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r1970, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r1971, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r1972, {low, high};
}
{
mul.f16x2 r1973, r1264, r1972;
}
{
add.f16x2 r1976, r1325, r1973;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r1979, {low, high};
}
{
mul.f16x2 r1980, r1273, r1979;
}
{
add.f16x2 r1983, r1970, r1980;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r1986, {low, high};
}
{
mul.f16x2 r1987, r1267, r1986;
}
{
add.f16x2 r1990, r1328, r1987;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r1993, {low, high};
}
{
mul.f16x2 r1994, r1270, r1993;
}
{
add.f16x2 r1997, r1971, r1994;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r2000, {low, high};
}
{
mul.f16x2 r2001, r1276, r2000;
}
{
add.f16x2 r2004, r1976, r2001;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r2007, {low, high};
}
{
mul.f16x2 r2008, r1285, r2007;
}
{
add.f16x2 r2011, r1983, r2008;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r2014, {low, high};
}
{
mul.f16x2 r2015, r1279, r2014;
}
{
add.f16x2 r2018, r1990, r2015;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r2021, {low, high};
}
{
mul.f16x2 r2022, r1282, r2021;
}
{
add.f16x2 r2025, r1997, r2022;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r2028, {low, high};
}
{
mul.f16x2 r2029, r1288, r2028;
}
{
add.f16x2 r2032, r2004, r2029;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r2035, {low, high};
}
{
mul.f16x2 r2036, r1297, r2035;
}
{
add.f16x2 r2039, r2011, r2036;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r2042, {low, high};
}
{
mul.f16x2 r2043, r1291, r2042;
}
{
add.f16x2 r2046, r2018, r2043;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r2049, {low, high};
}
{
mul.f16x2 r2050, r1294, r2049;
}
{
add.f16x2 r2053, r2025, r2050;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r2056, {low, high};
}
{
mul.f16x2 r2057, r1300, r2056;
}
{
add.f16x2 r2060, r2032, r2057;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r2063, {low, high};
}
{
mul.f16x2 r2064, r1309, r2063;
}
{
add.f16x2 r2067, r2039, r2064;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r2070, {low, high};
}
{
mul.f16x2 r2071, r1303, r2070;
}
{
add.f16x2 r2074, r2046, r2071;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r2077, {low, high};
}
{
mul.f16x2 r2078, r1306, r2077;
}
{
add.f16x2 r2081, r2053, r2078;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r2084, {low, high};
}
{
mul.f16x2 r2085, r1312, r2084;
}
{
add.f16x2 r2088, r2060, r2085;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r2091, {low, high};
}
{
mul.f16x2 r2092, r1321, r2091;
}
{
add.f16x2 r2095, r2067, r2092;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r2098, {low, high};
}
{
mul.f16x2 r2099, r1315, r2098;
}
{
add.f16x2 r2102, r2074, r2099;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r2105, {low, high};
}
{
mul.f16x2 r2106, r1318, r2105;
}
{
add.f16x2 r2109, r2081, r2106;
}
{
sub.f16x2 r2112, r2088, r2095;
}
{
add.f16x2 r2115, r2102, r2109;
}
{
add.f16x2 r2118, r2088, r2095;
}
{
sub.f16x2 r2121, r2102, r2109;
}
mul.wide.u32 rd6, r24, -1171354717;
shr.u64 rd7, rd6, 35;
cvt.u32.u64 r3368, rd7;
cvt.rn.f32.u32 f714, r3368;
mul.f32 f715, f714, 0f3D54B191;
cos.approx.f32 f470, f715;
sin.approx.f32 f716, f715;
neg.f32 f471, f716;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f470;
cvt.rn.f16.f32 high, f471;
mov.b32 r2124, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2127, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2129, {high, high};
}
{
mul.f16x2 r2131, r1499, r2129;
}
{
fma.rn.f16x2 r2134, r1496, r2127, r2131;
}
{
mul.f16x2 r2138, r1496, r2129;
}
{
neg.f16x2 r2141, r2138;
}
{
fma.rn.f16x2 r2143, r1499, r2127, r2141;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2147, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2149, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r2151, {low, high};
}
{
mul.f16x2 r2152, r2149, r2151;
}
{
mul.f16x2 r2155, r2124, r2147;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2158, {high, low};
}
{
fma.rn.f16x2 r2160, r2152, r2158, r2155;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2160;
mov.b32 r2164, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2160;
mov.b32 r2166, {high, high};
}
{
mul.f16x2 r2168, r1653, r2166;
}
{
fma.rn.f16x2 r2171, r1650, r2164, r2168;
}
{
mul.f16x2 r2175, r1650, r2166;
}
{
neg.f16x2 r2178, r2175;
}
{
fma.rn.f16x2 r2180, r1653, r2164, r2178;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2184, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2186, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r2188, {low, high};
}
{
mul.f16x2 r2189, r2186, r2188;
}
{
mul.f16x2 r2192, r2160, r2184;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2160;
mov.b32 r2195, {high, low};
}
{
fma.rn.f16x2 r2197, r2189, r2195, r2192;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2197;
mov.b32 r2201, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2197;
mov.b32 r2203, {high, high};
}
{
mul.f16x2 r2205, r1807, r2203;
}
{
fma.rn.f16x2 r2208, r1804, r2201, r2205;
}
{
mul.f16x2 r2212, r1804, r2203;
}
{
neg.f16x2 r2215, r2212;
}
{
fma.rn.f16x2 r2217, r1807, r2201, r2215;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2221, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2223, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r2225, {low, high};
}
{
mul.f16x2 r2226, r2223, r2225;
}
{
mul.f16x2 r2229, r2197, r2221;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2197;
mov.b32 r2232, {high, low};
}
{
fma.rn.f16x2 r2234, r2226, r2232, r2229;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2234;
mov.b32 r2238, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2234;
mov.b32 r2240, {high, high};
}
{
mul.f16x2 r2242, r1961, r2240;
}
{
fma.rn.f16x2 r2245, r1958, r2238, r2242;
}
{
mul.f16x2 r2249, r1958, r2240;
}
{
neg.f16x2 r2252, r2249;
}
{
fma.rn.f16x2 r2254, r1961, r2238, r2252;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2258, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2260, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r2262, {low, high};
}
{
mul.f16x2 r2263, r2260, r2262;
}
{
mul.f16x2 r2266, r2234, r2258;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2234;
mov.b32 r2269, {high, low};
}
{
fma.rn.f16x2 r2271, r2263, r2269, r2266;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2271;
mov.b32 r2275, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2271;
mov.b32 r2277, {high, high};
}
{
mul.f16x2 r2279, r2115, r2277;
}
{
fma.rn.f16x2 r2282, r2112, r2275, r2279;
}
{
mul.f16x2 r2286, r2112, r2277;
}
{
neg.f16x2 r2289, r2286;
}
{
fma.rn.f16x2 r2291, r2115, r2275, r2289;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2295, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2297, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r2299, {low, high};
}
{
mul.f16x2 r2300, r2297, r2299;
}
{
mul.f16x2 r2303, r2271, r2295;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2271;
mov.b32 r2306, {high, low};
}
{
fma.rn.f16x2 r2308, r2300, r2306, r2303;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2308;
mov.b32 r2312, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2308;
mov.b32 r2314, {high, high};
}
{
mul.f16x2 r2316, r2121, r2314;
}
{
fma.rn.f16x2 r2319, r2118, r2312, r2316;
}
{
mul.f16x2 r2323, r2118, r2314;
}
{
neg.f16x2 r2326, r2323;
}
{
fma.rn.f16x2 r2328, r2121, r2312, r2326;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2332, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2334, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r2336, {low, high};
}
{
mul.f16x2 r2337, r2334, r2336;
}
{
mul.f16x2 r2340, r2308, r2332;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2308;
mov.b32 r2343, {high, low};
}
{
fma.rn.f16x2 r2345, r2337, r2343, r2340;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2345;
mov.b32 r2349, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2345;
mov.b32 r2351, {high, high};
}
{
mul.f16x2 r2353, r1967, r2351;
}
{
fma.rn.f16x2 r2356, r1964, r2349, r2353;
}
{
mul.f16x2 r2360, r1964, r2351;
}
{
neg.f16x2 r2363, r2360;
}
{
fma.rn.f16x2 r2365, r1967, r2349, r2363;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2369, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2371, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r2373, {low, high};
}
{
mul.f16x2 r2374, r2371, r2373;
}
{
mul.f16x2 r2377, r2345, r2369;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2345;
mov.b32 r2380, {high, low};
}
{
fma.rn.f16x2 r2382, r2374, r2380, r2377;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2382;
mov.b32 r2386, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2382;
mov.b32 r2388, {high, high};
}
{
mul.f16x2 r2390, r1813, r2388;
}
{
fma.rn.f16x2 r2393, r1810, r2386, r2390;
}
{
mul.f16x2 r2397, r1810, r2388;
}
{
neg.f16x2 r2400, r2397;
}
{
fma.rn.f16x2 r2402, r1813, r2386, r2400;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2406, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2408, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r2410, {low, high};
}
{
mul.f16x2 r2411, r2408, r2410;
}
{
mul.f16x2 r2414, r2382, r2406;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2382;
mov.b32 r2417, {high, low};
}
{
fma.rn.f16x2 r2419, r2411, r2417, r2414;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2419;
mov.b32 r2423, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2419;
mov.b32 r2425, {high, high};
}
{
mul.f16x2 r2427, r1659, r2425;
}
{
fma.rn.f16x2 r2430, r1656, r2423, r2427;
}
{
mul.f16x2 r2434, r1656, r2425;
}
{
neg.f16x2 r2437, r2434;
}
{
fma.rn.f16x2 r2439, r1659, r2423, r2437;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2443, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2445, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r2447, {low, high};
}
{
mul.f16x2 r2448, r2445, r2447;
}
{
mul.f16x2 r2451, r2419, r2443;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2419;
mov.b32 r2454, {high, low};
}
{
fma.rn.f16x2 r2456, r2448, r2454, r2451;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2456;
mov.b32 r2460, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2456;
mov.b32 r2462, {high, high};
}
{
mul.f16x2 r2464, r1505, r2462;
}
{
fma.rn.f16x2 r2467, r1502, r2460, r2464;
}
{
mul.f16x2 r2471, r1502, r2462;
}
{
neg.f16x2 r2474, r2471;
}
{
fma.rn.f16x2 r2476, r1505, r2460, r2474;
}
mul.lo.s32 r3369, r3368, 11;
sub.s32 r3370, r24, r3369;
shl.b32 r3371, r3370, 3;
add.s32 r3372, r3365, r3371;
barrier.sync 0;
mad.lo.s32 r3373, r3368, 968, r3372;
st.shared.u32 [r3373], r1348;
st.shared.u32 [r3373+4], r1351;
st.shared.u32 [r3373+88], r2134;
st.shared.u32 [r3373+92], r2143;
st.shared.u32 [r3373+176], r2171;
st.shared.u32 [r3373+180], r2180;
st.shared.u32 [r3373+264], r2208;
st.shared.u32 [r3373+268], r2217;
st.shared.u32 [r3373+352], r2245;
st.shared.u32 [r3373+356], r2254;
st.shared.u32 [r3373+440], r2282;
st.shared.u32 [r3373+444], r2291;
st.shared.u32 [r3373+528], r2319;
st.shared.u32 [r3373+532], r2328;
st.shared.u32 [r3373+616], r2356;
st.shared.u32 [r3373+620], r2365;
st.shared.u32 [r3373+704], r2393;
st.shared.u32 [r3373+708], r2402;
st.shared.u32 [r3373+792], r2430;
st.shared.u32 [r3373+796], r2439;
st.shared.u32 [r3373+880], r2467;
st.shared.u32 [r3373+884], r2476;
barrier.sync 0;
ld.shared.u32 r2558, [r3367];
ld.shared.u32 r2561, [r3367+4];
ld.shared.u32 r2498, [r3367+968];
ld.shared.u32 r2501, [r3367+972];
ld.shared.u32 r2510, [r3367+1936];
ld.shared.u32 r2513, [r3367+1940];
ld.shared.u32 r2522, [r3367+2904];
ld.shared.u32 r2525, [r3367+2908];
ld.shared.u32 r2534, [r3367+3872];
ld.shared.u32 r2537, [r3367+3876];
ld.shared.u32 r2546, [r3367+4840];
ld.shared.u32 r2549, [r3367+4844];
ld.shared.u32 r2547, [r3367+5808];
ld.shared.u32 r2550, [r3367+5812];
ld.shared.u32 r2535, [r3367+6776];
ld.shared.u32 r2538, [r3367+6780];
ld.shared.u32 r2523, [r3367+7744];
ld.shared.u32 r2526, [r3367+7748];
ld.shared.u32 r2511, [r3367+8712];
ld.shared.u32 r2514, [r3367+8716];
ld.shared.u32 r2499, [r3367+9680];
ld.shared.u32 r2502, [r3367+9684];
{
add.f16x2 r2497, r2498, r2499;
}
{
add.f16x2 r2500, r2501, r2502;
}
{
sub.f16x2 r2503, r2498, r2499;
}
{
sub.f16x2 r2506, r2501, r2502;
}
{
add.f16x2 r2509, r2510, r2511;
}
{
add.f16x2 r2512, r2513, r2514;
}
{
sub.f16x2 r2515, r2510, r2511;
}
{
sub.f16x2 r2518, r2513, r2514;
}
{
add.f16x2 r2521, r2522, r2523;
}
{
add.f16x2 r2524, r2525, r2526;
}
{
sub.f16x2 r2527, r2522, r2523;
}
{
sub.f16x2 r2530, r2525, r2526;
}
{
add.f16x2 r2533, r2534, r2535;
}
{
add.f16x2 r2536, r2537, r2538;
}
{
sub.f16x2 r2539, r2534, r2535;
}
{
sub.f16x2 r2542, r2537, r2538;
}
{
add.f16x2 r2545, r2546, r2547;
}
{
add.f16x2 r2548, r2549, r2550;
}
{
sub.f16x2 r2551, r2546, r2547;
}
{
sub.f16x2 r2554, r2549, r2550;
}
{
add.f16x2 r2557, r2558, r2497;
}
{
add.f16x2 r2560, r2561, r2500;
}
{
add.f16x2 r2563, r2557, r2509;
}
{
add.f16x2 r2566, r2560, r2512;
}
{
add.f16x2 r2569, r2563, r2521;
}
{
add.f16x2 r2572, r2566, r2524;
}
{
add.f16x2 r2575, r2569, r2533;
}
{
add.f16x2 r2578, r2572, r2536;
}
{
add.f16x2 %0, r2575, r2545;
}
{
add.f16x2 %1, r2578, r2548;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r2587, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r2588, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r2589, {low, high};
}
{
mul.f16x2 r2590, r2497, r2589;
}
{
add.f16x2 r2593, r2558, r2590;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r2596, {low, high};
}
{
mul.f16x2 r2597, r2506, r2596;
}
{
add.f16x2 r2600, r2587, r2597;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r2603, {low, high};
}
{
mul.f16x2 r2604, r2500, r2603;
}
{
add.f16x2 r2607, r2561, r2604;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r2610, {low, high};
}
{
mul.f16x2 r2611, r2503, r2610;
}
{
add.f16x2 r2614, r2588, r2611;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r2617, {low, high};
}
{
mul.f16x2 r2618, r2509, r2617;
}
{
add.f16x2 r2621, r2593, r2618;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r2624, {low, high};
}
{
mul.f16x2 r2625, r2518, r2624;
}
{
add.f16x2 r2628, r2600, r2625;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r2631, {low, high};
}
{
mul.f16x2 r2632, r2512, r2631;
}
{
add.f16x2 r2635, r2607, r2632;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r2638, {low, high};
}
{
mul.f16x2 r2639, r2515, r2638;
}
{
add.f16x2 r2642, r2614, r2639;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r2645, {low, high};
}
{
mul.f16x2 r2646, r2521, r2645;
}
{
add.f16x2 r2649, r2621, r2646;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r2652, {low, high};
}
{
mul.f16x2 r2653, r2530, r2652;
}
{
add.f16x2 r2656, r2628, r2653;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r2659, {low, high};
}
{
mul.f16x2 r2660, r2524, r2659;
}
{
add.f16x2 r2663, r2635, r2660;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r2666, {low, high};
}
{
mul.f16x2 r2667, r2527, r2666;
}
{
add.f16x2 r2670, r2642, r2667;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r2673, {low, high};
}
{
mul.f16x2 r2674, r2533, r2673;
}
{
add.f16x2 r2677, r2649, r2674;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r2680, {low, high};
}
{
mul.f16x2 r2681, r2542, r2680;
}
{
add.f16x2 r2684, r2656, r2681;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r2687, {low, high};
}
{
mul.f16x2 r2688, r2536, r2687;
}
{
add.f16x2 r2691, r2663, r2688;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r2694, {low, high};
}
{
mul.f16x2 r2695, r2539, r2694;
}
{
add.f16x2 r2698, r2670, r2695;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r2701, {low, high};
}
{
mul.f16x2 r2702, r2545, r2701;
}
{
add.f16x2 r2705, r2677, r2702;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r2708, {low, high};
}
{
mul.f16x2 r2709, r2554, r2708;
}
{
add.f16x2 r2712, r2684, r2709;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r2715, {low, high};
}
{
mul.f16x2 r2716, r2548, r2715;
}
{
add.f16x2 r2719, r2691, r2716;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r2722, {low, high};
}
{
mul.f16x2 r2723, r2551, r2722;
}
{
add.f16x2 r2726, r2698, r2723;
}
{
sub.f16x2 %2, r2705, r2712;
}
{
add.f16x2 %3, r2719, r2726;
}
{
add.f16x2 %20, r2705, r2712;
}
{
sub.f16x2 %21, r2719, r2726;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r2741, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r2742, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r2743, {low, high};
}
{
mul.f16x2 r2744, r2497, r2743;
}
{
add.f16x2 r2747, r2558, r2744;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r2750, {low, high};
}
{
mul.f16x2 r2751, r2506, r2750;
}
{
add.f16x2 r2754, r2741, r2751;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r2757, {low, high};
}
{
mul.f16x2 r2758, r2500, r2757;
}
{
add.f16x2 r2761, r2561, r2758;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r2764, {low, high};
}
{
mul.f16x2 r2765, r2503, r2764;
}
{
add.f16x2 r2768, r2742, r2765;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r2771, {low, high};
}
{
mul.f16x2 r2772, r2509, r2771;
}
{
add.f16x2 r2775, r2747, r2772;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r2778, {low, high};
}
{
mul.f16x2 r2779, r2518, r2778;
}
{
add.f16x2 r2782, r2754, r2779;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r2785, {low, high};
}
{
mul.f16x2 r2786, r2512, r2785;
}
{
add.f16x2 r2789, r2761, r2786;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r2792, {low, high};
}
{
mul.f16x2 r2793, r2515, r2792;
}
{
add.f16x2 r2796, r2768, r2793;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r2799, {low, high};
}
{
mul.f16x2 r2800, r2521, r2799;
}
{
add.f16x2 r2803, r2775, r2800;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r2806, {low, high};
}
{
mul.f16x2 r2807, r2530, r2806;
}
{
add.f16x2 r2810, r2782, r2807;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r2813, {low, high};
}
{
mul.f16x2 r2814, r2524, r2813;
}
{
add.f16x2 r2817, r2789, r2814;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r2820, {low, high};
}
{
mul.f16x2 r2821, r2527, r2820;
}
{
add.f16x2 r2824, r2796, r2821;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r2827, {low, high};
}
{
mul.f16x2 r2828, r2533, r2827;
}
{
add.f16x2 r2831, r2803, r2828;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r2834, {low, high};
}
{
mul.f16x2 r2835, r2542, r2834;
}
{
add.f16x2 r2838, r2810, r2835;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r2841, {low, high};
}
{
mul.f16x2 r2842, r2536, r2841;
}
{
add.f16x2 r2845, r2817, r2842;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r2848, {low, high};
}
{
mul.f16x2 r2849, r2539, r2848;
}
{
add.f16x2 r2852, r2824, r2849;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r2855, {low, high};
}
{
mul.f16x2 r2856, r2545, r2855;
}
{
add.f16x2 r2859, r2831, r2856;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r2862, {low, high};
}
{
mul.f16x2 r2863, r2554, r2862;
}
{
add.f16x2 r2866, r2838, r2863;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r2869, {low, high};
}
{
mul.f16x2 r2870, r2548, r2869;
}
{
add.f16x2 r2873, r2845, r2870;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r2876, {low, high};
}
{
mul.f16x2 r2877, r2551, r2876;
}
{
add.f16x2 r2880, r2852, r2877;
}
{
sub.f16x2 %4, r2859, r2866;
}
{
add.f16x2 %5, r2873, r2880;
}
{
add.f16x2 %18, r2859, r2866;
}
{
sub.f16x2 %19, r2873, r2880;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r2895, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r2896, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r2897, {low, high};
}
{
mul.f16x2 r2898, r2497, r2897;
}
{
add.f16x2 r2901, r2558, r2898;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r2904, {low, high};
}
{
mul.f16x2 r2905, r2506, r2904;
}
{
add.f16x2 r2908, r2895, r2905;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r2911, {low, high};
}
{
mul.f16x2 r2912, r2500, r2911;
}
{
add.f16x2 r2915, r2561, r2912;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r2918, {low, high};
}
{
mul.f16x2 r2919, r2503, r2918;
}
{
add.f16x2 r2922, r2896, r2919;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r2925, {low, high};
}
{
mul.f16x2 r2926, r2509, r2925;
}
{
add.f16x2 r2929, r2901, r2926;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r2932, {low, high};
}
{
mul.f16x2 r2933, r2518, r2932;
}
{
add.f16x2 r2936, r2908, r2933;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r2939, {low, high};
}
{
mul.f16x2 r2940, r2512, r2939;
}
{
add.f16x2 r2943, r2915, r2940;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r2946, {low, high};
}
{
mul.f16x2 r2947, r2515, r2946;
}
{
add.f16x2 r2950, r2922, r2947;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r2953, {low, high};
}
{
mul.f16x2 r2954, r2521, r2953;
}
{
add.f16x2 r2957, r2929, r2954;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r2960, {low, high};
}
{
mul.f16x2 r2961, r2530, r2960;
}
{
add.f16x2 r2964, r2936, r2961;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r2967, {low, high};
}
{
mul.f16x2 r2968, r2524, r2967;
}
{
add.f16x2 r2971, r2943, r2968;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r2974, {low, high};
}
{
mul.f16x2 r2975, r2527, r2974;
}
{
add.f16x2 r2978, r2950, r2975;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r2981, {low, high};
}
{
mul.f16x2 r2982, r2533, r2981;
}
{
add.f16x2 r2985, r2957, r2982;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r2988, {low, high};
}
{
mul.f16x2 r2989, r2542, r2988;
}
{
add.f16x2 r2992, r2964, r2989;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r2995, {low, high};
}
{
mul.f16x2 r2996, r2536, r2995;
}
{
add.f16x2 r2999, r2971, r2996;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r3002, {low, high};
}
{
mul.f16x2 r3003, r2539, r3002;
}
{
add.f16x2 r3006, r2978, r3003;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r3009, {low, high};
}
{
mul.f16x2 r3010, r2545, r3009;
}
{
add.f16x2 r3013, r2985, r3010;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r3016, {low, high};
}
{
mul.f16x2 r3017, r2554, r3016;
}
{
add.f16x2 r3020, r2992, r3017;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r3023, {low, high};
}
{
mul.f16x2 r3024, r2548, r3023;
}
{
add.f16x2 r3027, r2999, r3024;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r3030, {low, high};
}
{
mul.f16x2 r3031, r2551, r3030;
}
{
add.f16x2 r3034, r3006, r3031;
}
{
sub.f16x2 %6, r3013, r3020;
}
{
add.f16x2 %7, r3027, r3034;
}
{
add.f16x2 %16, r3013, r3020;
}
{
sub.f16x2 %17, r3027, r3034;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r3049, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r3050, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r3051, {low, high};
}
{
mul.f16x2 r3052, r2497, r3051;
}
{
add.f16x2 r3055, r2558, r3052;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r3058, {low, high};
}
{
mul.f16x2 r3059, r2506, r3058;
}
{
add.f16x2 r3062, r3049, r3059;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r3065, {low, high};
}
{
mul.f16x2 r3066, r2500, r3065;
}
{
add.f16x2 r3069, r2561, r3066;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r3072, {low, high};
}
{
mul.f16x2 r3073, r2503, r3072;
}
{
add.f16x2 r3076, r3050, r3073;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r3079, {low, high};
}
{
mul.f16x2 r3080, r2509, r3079;
}
{
add.f16x2 r3083, r3055, r3080;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r3086, {low, high};
}
{
mul.f16x2 r3087, r2518, r3086;
}
{
add.f16x2 r3090, r3062, r3087;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r3093, {low, high};
}
{
mul.f16x2 r3094, r2512, r3093;
}
{
add.f16x2 r3097, r3069, r3094;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r3100, {low, high};
}
{
mul.f16x2 r3101, r2515, r3100;
}
{
add.f16x2 r3104, r3076, r3101;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r3107, {low, high};
}
{
mul.f16x2 r3108, r2521, r3107;
}
{
add.f16x2 r3111, r3083, r3108;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r3114, {low, high};
}
{
mul.f16x2 r3115, r2530, r3114;
}
{
add.f16x2 r3118, r3090, r3115;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r3121, {low, high};
}
{
mul.f16x2 r3122, r2524, r3121;
}
{
add.f16x2 r3125, r3097, r3122;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r3128, {low, high};
}
{
mul.f16x2 r3129, r2527, r3128;
}
{
add.f16x2 r3132, r3104, r3129;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r3135, {low, high};
}
{
mul.f16x2 r3136, r2533, r3135;
}
{
add.f16x2 r3139, r3111, r3136;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r3142, {low, high};
}
{
mul.f16x2 r3143, r2542, r3142;
}
{
add.f16x2 r3146, r3118, r3143;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r3149, {low, high};
}
{
mul.f16x2 r3150, r2536, r3149;
}
{
add.f16x2 r3153, r3125, r3150;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r3156, {low, high};
}
{
mul.f16x2 r3157, r2539, r3156;
}
{
add.f16x2 r3160, r3132, r3157;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r3163, {low, high};
}
{
mul.f16x2 r3164, r2545, r3163;
}
{
add.f16x2 r3167, r3139, r3164;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r3170, {low, high};
}
{
mul.f16x2 r3171, r2554, r3170;
}
{
add.f16x2 r3174, r3146, r3171;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r3177, {low, high};
}
{
mul.f16x2 r3178, r2548, r3177;
}
{
add.f16x2 r3181, r3153, r3178;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r3184, {low, high};
}
{
mul.f16x2 r3185, r2551, r3184;
}
{
add.f16x2 r3188, r3160, r3185;
}
{
sub.f16x2 %8, r3167, r3174;
}
{
add.f16x2 %9, r3181, r3188;
}
{
add.f16x2 %14, r3167, r3174;
}
{
sub.f16x2 %15, r3181, r3188;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r3203, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r3204, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r3205, {low, high};
}
{
mul.f16x2 r3206, r2497, r3205;
}
{
add.f16x2 r3209, r2558, r3206;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r3212, {low, high};
}
{
mul.f16x2 r3213, r2506, r3212;
}
{
add.f16x2 r3216, r3203, r3213;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r3219, {low, high};
}
{
mul.f16x2 r3220, r2500, r3219;
}
{
add.f16x2 r3223, r2561, r3220;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r3226, {low, high};
}
{
mul.f16x2 r3227, r2503, r3226;
}
{
add.f16x2 r3230, r3204, r3227;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r3233, {low, high};
}
{
mul.f16x2 r3234, r2509, r3233;
}
{
add.f16x2 r3237, r3209, r3234;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r3240, {low, high};
}
{
mul.f16x2 r3241, r2518, r3240;
}
{
add.f16x2 r3244, r3216, r3241;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r3247, {low, high};
}
{
mul.f16x2 r3248, r2512, r3247;
}
{
add.f16x2 r3251, r3223, r3248;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r3254, {low, high};
}
{
mul.f16x2 r3255, r2515, r3254;
}
{
add.f16x2 r3258, r3230, r3255;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r3261, {low, high};
}
{
mul.f16x2 r3262, r2521, r3261;
}
{
add.f16x2 r3265, r3237, r3262;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r3268, {low, high};
}
{
mul.f16x2 r3269, r2530, r3268;
}
{
add.f16x2 r3272, r3244, r3269;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r3275, {low, high};
}
{
mul.f16x2 r3276, r2524, r3275;
}
{
add.f16x2 r3279, r3251, r3276;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r3282, {low, high};
}
{
mul.f16x2 r3283, r2527, r3282;
}
{
add.f16x2 r3286, r3258, r3283;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r3289, {low, high};
}
{
mul.f16x2 r3290, r2533, r3289;
}
{
add.f16x2 r3293, r3265, r3290;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r3296, {low, high};
}
{
mul.f16x2 r3297, r2542, r3296;
}
{
add.f16x2 r3300, r3272, r3297;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r3303, {low, high};
}
{
mul.f16x2 r3304, r2536, r3303;
}
{
add.f16x2 r3307, r3279, r3304;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r3310, {low, high};
}
{
mul.f16x2 r3311, r2539, r3310;
}
{
add.f16x2 r3314, r3286, r3311;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r3317, {low, high};
}
{
mul.f16x2 r3318, r2545, r3317;
}
{
add.f16x2 r3321, r3293, r3318;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r3324, {low, high};
}
{
mul.f16x2 r3325, r2554, r3324;
}
{
add.f16x2 r3328, r3300, r3325;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r3331, {low, high};
}
{
mul.f16x2 r3332, r2548, r3331;
}
{
add.f16x2 r3335, r3307, r3332;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r3338, {low, high};
}
{
mul.f16x2 r3339, r2551, r3338;
}
{
add.f16x2 r3342, r3314, r3339;
}
{
sub.f16x2 %10, r3321, r3328;
}
{
add.f16x2 %11, r3335, r3342;
}
{
add.f16x2 %12, r3321, r3328;
}
{
sub.f16x2 %13, r3335, r3342;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1129, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .pred p<2>;
.reg .f32 f<718>;
.reg .b32 r<3374>;
.reg .b64 rd<8>;
{
add.f16x2 r25, %25, %43;
}
{
add.f16x2 r28, %26, %44;
}
{
sub.f16x2 r31, %25, %43;
}
{
sub.f16x2 r34, %26, %44;
}
{
add.f16x2 r37, %27, %41;
}
{
add.f16x2 r40, %28, %42;
}
{
sub.f16x2 r43, %27, %41;
}
{
sub.f16x2 r46, %28, %42;
}
{
add.f16x2 r49, %29, %39;
}
{
add.f16x2 r52, %30, %40;
}
{
sub.f16x2 r55, %29, %39;
}
{
sub.f16x2 r58, %30, %40;
}
{
add.f16x2 r61, %31, %37;
}
{
add.f16x2 r64, %32, %38;
}
{
sub.f16x2 r67, %31, %37;
}
{
sub.f16x2 r70, %32, %38;
}
{
add.f16x2 r73, %33, %35;
}
{
add.f16x2 r76, %34, %36;
}
{
sub.f16x2 r79, %33, %35;
}
{
sub.f16x2 r82, %34, %36;
}
{
add.f16x2 r85, %23, r25;
}
{
add.f16x2 r88, %24, r28;
}
{
add.f16x2 r91, r85, r37;
}
{
add.f16x2 r94, r88, r40;
}
{
add.f16x2 r97, r91, r49;
}
{
add.f16x2 r100, r94, r52;
}
{
add.f16x2 r103, r97, r61;
}
{
add.f16x2 r106, r100, r64;
}
{
add.f16x2 r109, r103, r73;
}
{
add.f16x2 r112, r106, r76;
}
mov.f32 f183, 0f00000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r115, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r116, {low, high};
}
mov.f32 f197, 0f3F575C64;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r117, {low, high};
}
{
mul.f16x2 r118, r25, r117;
}
{
add.f16x2 r121, %23, r118;
}
mov.f32 f163, 0f3F0A6770;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r124, {low, high};
}
{
mul.f16x2 r125, r34, r124;
}
{
add.f16x2 r128, r115, r125;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r131, {low, high};
}
{
mul.f16x2 r132, r28, r131;
}
{
add.f16x2 r135, %24, r132;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r138, {low, high};
}
{
mul.f16x2 r139, r31, r138;
}
{
add.f16x2 r142, r116, r139;
}
mov.f32 f213, 0f3ED4B147;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r145, {low, high};
}
{
mul.f16x2 r146, r37, r145;
}
{
add.f16x2 r149, r121, r146;
}
mov.f32 f59, 0f3F68DDA4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r152, {low, high};
}
{
mul.f16x2 r153, r46, r152;
}
{
add.f16x2 r156, r128, r153;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r159, {low, high};
}
{
mul.f16x2 r160, r40, r159;
}
{
add.f16x2 r163, r135, r160;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r166, {low, high};
}
{
mul.f16x2 r167, r43, r166;
}
{
add.f16x2 r170, r142, r167;
}
mov.f32 f221, 0fBE11BAFB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r173, {low, high};
}
{
mul.f16x2 r174, r49, r173;
}
{
add.f16x2 r177, r149, r174;
}
mov.f32 f223, 0f3F7D64F0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r180, {low, high};
}
{
mul.f16x2 r181, r58, r180;
}
{
add.f16x2 r184, r156, r181;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r187, {low, high};
}
{
mul.f16x2 r188, r52, r187;
}
{
add.f16x2 r191, r163, r188;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r194, {low, high};
}
{
mul.f16x2 r195, r55, r194;
}
{
add.f16x2 r198, r170, r195;
}
mov.f32 f205, 0fBF27A4F4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r201, {low, high};
}
{
mul.f16x2 r202, r61, r201;
}
{
add.f16x2 r205, r177, r202;
}
mov.f32 f207, 0f3F4178CE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r208, {low, high};
}
{
mul.f16x2 r209, r70, r208;
}
{
add.f16x2 r212, r184, r209;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r215, {low, high};
}
{
mul.f16x2 r216, r64, r215;
}
{
add.f16x2 r219, r191, r216;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r222, {low, high};
}
{
mul.f16x2 r223, r67, r222;
}
{
add.f16x2 r226, r198, r223;
}
mov.f32 f189, 0fBF75A155;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r229, {low, high};
}
{
mul.f16x2 r230, r73, r229;
}
{
add.f16x2 r233, r205, r230;
}
mov.f32 f191, 0f3E903F40;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r236, {low, high};
}
{
mul.f16x2 r237, r82, r236;
}
{
add.f16x2 r240, r212, r237;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r243, {low, high};
}
{
mul.f16x2 r244, r76, r243;
}
{
add.f16x2 r247, r219, r244;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r250, {low, high};
}
{
mul.f16x2 r251, r79, r250;
}
{
add.f16x2 r254, r226, r251;
}
{
sub.f16x2 r257, r233, r240;
}
{
add.f16x2 r260, r247, r254;
}
{
add.f16x2 r263, r233, r240;
}
{
sub.f16x2 r266, r247, r254;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r269, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r270, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r271, {low, high};
}
{
mul.f16x2 r272, r25, r271;
}
{
add.f16x2 r275, %23, r272;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r278, {low, high};
}
{
mul.f16x2 r279, r34, r278;
}
{
add.f16x2 r282, r269, r279;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r285, {low, high};
}
{
mul.f16x2 r286, r28, r285;
}
{
add.f16x2 r289, %24, r286;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r292, {low, high};
}
{
mul.f16x2 r293, r31, r292;
}
{
add.f16x2 r296, r270, r293;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r299, {low, high};
}
{
mul.f16x2 r300, r37, r299;
}
{
add.f16x2 r303, r275, r300;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r306, {low, high};
}
{
mul.f16x2 r307, r46, r306;
}
{
add.f16x2 r310, r282, r307;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r313, {low, high};
}
{
mul.f16x2 r314, r40, r313;
}
{
add.f16x2 r317, r289, r314;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r320, {low, high};
}
{
mul.f16x2 r321, r43, r320;
}
{
add.f16x2 r324, r296, r321;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r327, {low, high};
}
{
mul.f16x2 r328, r49, r327;
}
{
add.f16x2 r331, r303, r328;
}
mov.f32 f111, 0fBE903F40;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r334, {low, high};
}
{
mul.f16x2 r335, r58, r334;
}
{
add.f16x2 r338, r310, r335;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r341, {low, high};
}
{
mul.f16x2 r342, r52, r341;
}
{
add.f16x2 r345, r317, r342;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r348, {low, high};
}
{
mul.f16x2 r349, r55, r348;
}
{
add.f16x2 r352, r324, r349;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r355, {low, high};
}
{
mul.f16x2 r356, r61, r355;
}
{
add.f16x2 r359, r331, r356;
}
mov.f32 f155, 0fBF7D64F0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r362, {low, high};
}
{
mul.f16x2 r363, r70, r362;
}
{
add.f16x2 r366, r338, r363;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r369, {low, high};
}
{
mul.f16x2 r370, r64, r369;
}
{
add.f16x2 r373, r345, r370;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r376, {low, high};
}
{
mul.f16x2 r377, r67, r376;
}
{
add.f16x2 r380, r352, r377;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r383, {low, high};
}
{
mul.f16x2 r384, r73, r383;
}
{
add.f16x2 r387, r359, r384;
}
mov.f32 f199, 0fBF0A6770;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r390, {low, high};
}
{
mul.f16x2 r391, r82, r390;
}
{
add.f16x2 r394, r366, r391;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r397, {low, high};
}
{
mul.f16x2 r398, r76, r397;
}
{
add.f16x2 r401, r373, r398;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r404, {low, high};
}
{
mul.f16x2 r405, r79, r404;
}
{
add.f16x2 r408, r380, r405;
}
{
sub.f16x2 r411, r387, r394;
}
{
add.f16x2 r414, r401, r408;
}
{
add.f16x2 r417, r387, r394;
}
{
sub.f16x2 r420, r401, r408;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r423, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r424, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r425, {low, high};
}
{
mul.f16x2 r426, r25, r425;
}
{
add.f16x2 r429, %23, r426;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r432, {low, high};
}
{
mul.f16x2 r433, r34, r432;
}
{
add.f16x2 r436, r423, r433;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r439, {low, high};
}
{
mul.f16x2 r440, r28, r439;
}
{
add.f16x2 r443, %24, r440;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r446, {low, high};
}
{
mul.f16x2 r447, r31, r446;
}
{
add.f16x2 r450, r424, r447;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r453, {low, high};
}
{
mul.f16x2 r454, r37, r453;
}
{
add.f16x2 r457, r429, r454;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r460, {low, high};
}
{
mul.f16x2 r461, r46, r460;
}
{
add.f16x2 r464, r436, r461;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r467, {low, high};
}
{
mul.f16x2 r468, r40, r467;
}
{
add.f16x2 r471, r443, r468;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r474, {low, high};
}
{
mul.f16x2 r475, r43, r474;
}
{
add.f16x2 r478, r450, r475;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r481, {low, high};
}
{
mul.f16x2 r482, r49, r481;
}
{
add.f16x2 r485, r457, r482;
}
mov.f32 f215, 0fBF68DDA4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r488, {low, high};
}
{
mul.f16x2 r489, r58, r488;
}
{
add.f16x2 r492, r464, r489;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r495, {low, high};
}
{
mul.f16x2 r496, r52, r495;
}
{
add.f16x2 r499, r471, r496;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r502, {low, high};
}
{
mul.f16x2 r503, r55, r502;
}
{
add.f16x2 r506, r478, r503;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r509, {low, high};
}
{
mul.f16x2 r510, r61, r509;
}
{
add.f16x2 r513, r485, r510;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r516, {low, high};
}
{
mul.f16x2 r517, r70, r516;
}
{
add.f16x2 r520, r492, r517;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r523, {low, high};
}
{
mul.f16x2 r524, r64, r523;
}
{
add.f16x2 r527, r499, r524;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r530, {low, high};
}
{
mul.f16x2 r531, r67, r530;
}
{
add.f16x2 r534, r506, r531;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r537, {low, high};
}
{
mul.f16x2 r538, r73, r537;
}
{
add.f16x2 r541, r513, r538;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r544, {low, high};
}
{
mul.f16x2 r545, r82, r544;
}
{
add.f16x2 r548, r520, r545;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r551, {low, high};
}
{
mul.f16x2 r552, r76, r551;
}
{
add.f16x2 r555, r527, r552;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r558, {low, high};
}
{
mul.f16x2 r559, r79, r558;
}
{
add.f16x2 r562, r534, r559;
}
{
sub.f16x2 r565, r541, r548;
}
{
add.f16x2 r568, r555, r562;
}
{
add.f16x2 r571, r541, r548;
}
{
sub.f16x2 r574, r555, r562;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r577, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r578, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r579, {low, high};
}
{
mul.f16x2 r580, r25, r579;
}
{
add.f16x2 r583, %23, r580;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r586, {low, high};
}
{
mul.f16x2 r587, r34, r586;
}
{
add.f16x2 r590, r577, r587;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r593, {low, high};
}
{
mul.f16x2 r594, r28, r593;
}
{
add.f16x2 r597, %24, r594;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r600, {low, high};
}
{
mul.f16x2 r601, r31, r600;
}
{
add.f16x2 r604, r578, r601;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r607, {low, high};
}
{
mul.f16x2 r608, r37, r607;
}
{
add.f16x2 r611, r583, r608;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r614, {low, high};
}
{
mul.f16x2 r615, r46, r614;
}
{
add.f16x2 r618, r590, r615;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r621, {low, high};
}
{
mul.f16x2 r622, r40, r621;
}
{
add.f16x2 r625, r597, r622;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r628, {low, high};
}
{
mul.f16x2 r629, r43, r628;
}
{
add.f16x2 r632, r604, r629;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r635, {low, high};
}
{
mul.f16x2 r636, r49, r635;
}
{
add.f16x2 r639, r611, r636;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r642, {low, high};
}
{
mul.f16x2 r643, r58, r642;
}
{
add.f16x2 r646, r618, r643;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r649, {low, high};
}
{
mul.f16x2 r650, r52, r649;
}
{
add.f16x2 r653, r625, r650;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r656, {low, high};
}
{
mul.f16x2 r657, r55, r656;
}
{
add.f16x2 r660, r632, r657;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r663, {low, high};
}
{
mul.f16x2 r664, r61, r663;
}
{
add.f16x2 r667, r639, r664;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r670, {low, high};
}
{
mul.f16x2 r671, r70, r670;
}
{
add.f16x2 r674, r646, r671;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r677, {low, high};
}
{
mul.f16x2 r678, r64, r677;
}
{
add.f16x2 r681, r653, r678;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r684, {low, high};
}
{
mul.f16x2 r685, r67, r684;
}
{
add.f16x2 r688, r660, r685;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r691, {low, high};
}
{
mul.f16x2 r692, r73, r691;
}
{
add.f16x2 r695, r667, r692;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r698, {low, high};
}
{
mul.f16x2 r699, r82, r698;
}
{
add.f16x2 r702, r674, r699;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r705, {low, high};
}
{
mul.f16x2 r706, r76, r705;
}
{
add.f16x2 r709, r681, r706;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r712, {low, high};
}
{
mul.f16x2 r713, r79, r712;
}
{
add.f16x2 r716, r688, r713;
}
{
sub.f16x2 r719, r695, r702;
}
{
add.f16x2 r722, r709, r716;
}
{
add.f16x2 r725, r695, r702;
}
{
sub.f16x2 r728, r709, r716;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r731, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r732, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r733, {low, high};
}
{
mul.f16x2 r734, r25, r733;
}
{
add.f16x2 r737, %23, r734;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r740, {low, high};
}
{
mul.f16x2 r741, r34, r740;
}
{
add.f16x2 r744, r731, r741;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r747, {low, high};
}
{
mul.f16x2 r748, r28, r747;
}
{
add.f16x2 r751, %24, r748;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r754, {low, high};
}
{
mul.f16x2 r755, r31, r754;
}
{
add.f16x2 r758, r732, r755;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r761, {low, high};
}
{
mul.f16x2 r762, r37, r761;
}
{
add.f16x2 r765, r737, r762;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r768, {low, high};
}
{
mul.f16x2 r769, r46, r768;
}
{
add.f16x2 r772, r744, r769;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r775, {low, high};
}
{
mul.f16x2 r776, r40, r775;
}
{
add.f16x2 r779, r751, r776;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r782, {low, high};
}
{
mul.f16x2 r783, r43, r782;
}
{
add.f16x2 r786, r758, r783;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r789, {low, high};
}
{
mul.f16x2 r790, r49, r789;
}
{
add.f16x2 r793, r765, r790;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r796, {low, high};
}
{
mul.f16x2 r797, r58, r796;
}
{
add.f16x2 r800, r772, r797;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r803, {low, high};
}
{
mul.f16x2 r804, r52, r803;
}
{
add.f16x2 r807, r779, r804;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r810, {low, high};
}
{
mul.f16x2 r811, r55, r810;
}
{
add.f16x2 r814, r786, r811;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r817, {low, high};
}
{
mul.f16x2 r818, r61, r817;
}
{
add.f16x2 r821, r793, r818;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r824, {low, high};
}
{
mul.f16x2 r825, r70, r824;
}
{
add.f16x2 r828, r800, r825;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r831, {low, high};
}
{
mul.f16x2 r832, r64, r831;
}
{
add.f16x2 r835, r807, r832;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r838, {low, high};
}
{
mul.f16x2 r839, r67, r838;
}
{
add.f16x2 r842, r814, r839;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r845, {low, high};
}
{
mul.f16x2 r846, r73, r845;
}
{
add.f16x2 r849, r821, r846;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r852, {low, high};
}
{
mul.f16x2 r853, r82, r852;
}
{
add.f16x2 r856, r828, r853;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r859, {low, high};
}
{
mul.f16x2 r860, r76, r859;
}
{
add.f16x2 r863, r835, r860;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r866, {low, high};
}
{
mul.f16x2 r867, r79, r866;
}
{
add.f16x2 r870, r842, r867;
}
{
sub.f16x2 r873, r849, r856;
}
{
add.f16x2 r876, r863, r870;
}
{
add.f16x2 r879, r849, r856;
}
{
sub.f16x2 r882, r863, r870;
}
mov.u32 r23, %tid.x;
mul.wide.u32 rd2, r23, 248469183;
shr.u64 rd3, rd2, 32;
cvt.u32.u64 r885, rd3;
sub.s32 r886, r23, r885;
shr.u32 r887, r886, 1;
add.s32 r888, r887, r885;
shr.u32 r889, r888, 6;
mul.lo.s32 r890, r889, 121;
sub.s32 r24, r23, r890;
cvt.rn.f32.u32 f225, r24;
mul.f32 f1, f225, 0f3B9AAFAF;
setp.eq.s32 p1, r24, 18;
mov.f32 f717, 0f3DADD00E;
@p1 bra LBB1_2;
sin.approx.f32 f717, f1;
LBB1_2:
mov.u32 r3362, %tid.y;
mov.u32 r3363, %22;
mad.lo.s32 r3364, r3362, 5324, r3363;
mad.lo.s32 r3365, r889, 5324, r3364;
neg.f32 f227, f717;
cos.approx.f32 f226, f1;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f226;
cvt.rn.f16.f32 high, f227;
mov.b32 r891, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r894, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r896, {high, high};
}
{
mul.f16x2 r898, r260, r896;
}
{
fma.rn.f16x2 r901, r257, r894, r898;
}
{
mul.f16x2 r905, r257, r896;
}
{
neg.f16x2 r908, r905;
}
{
fma.rn.f16x2 r910, r260, r894, r908;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r914, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r916, {high, high};
}
mov.f32 f490, 0fBF800000;
mov.f32 f491, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r918, {low, high};
}
{
mul.f16x2 r919, r916, r918;
}
{
mul.f16x2 r922, r891, r914;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
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
mul.f16x2 r935, r414, r933;
}
{
fma.rn.f16x2 r938, r411, r931, r935;
}
{
mul.f16x2 r942, r411, r933;
}
{
neg.f16x2 r945, r942;
}
{
fma.rn.f16x2 r947, r414, r931, r945;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r951, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r953, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
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
mul.f16x2 r972, r568, r970;
}
{
fma.rn.f16x2 r975, r565, r968, r972;
}
{
mul.f16x2 r979, r565, r970;
}
{
neg.f16x2 r982, r979;
}
{
fma.rn.f16x2 r984, r568, r968, r982;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r988, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r990, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
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
mul.f16x2 r1009, r722, r1007;
}
{
fma.rn.f16x2 r1012, r719, r1005, r1009;
}
{
mul.f16x2 r1016, r719, r1007;
}
{
neg.f16x2 r1019, r1016;
}
{
fma.rn.f16x2 r1021, r722, r1005, r1019;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1025, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1027, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
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
mul.f16x2 r1046, r876, r1044;
}
{
fma.rn.f16x2 r1049, r873, r1042, r1046;
}
{
mul.f16x2 r1053, r873, r1044;
}
{
neg.f16x2 r1056, r1053;
}
{
fma.rn.f16x2 r1058, r876, r1042, r1056;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1062, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1064, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r1066, {low, high};
}
{
mul.f16x2 r1067, r1064, r1066;
}
{
mul.f16x2 r1070, r1038, r1062;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1038;
mov.b32 r1073, {high, low};
}
{
fma.rn.f16x2 r1075, r1067, r1073, r1070;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1075;
mov.b32 r1079, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1075;
mov.b32 r1081, {high, high};
}
{
mul.f16x2 r1083, r882, r1081;
}
{
fma.rn.f16x2 r1086, r879, r1079, r1083;
}
{
mul.f16x2 r1090, r879, r1081;
}
{
neg.f16x2 r1093, r1090;
}
{
fma.rn.f16x2 r1095, r882, r1079, r1093;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1099, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1101, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r1103, {low, high};
}
{
mul.f16x2 r1104, r1101, r1103;
}
{
mul.f16x2 r1107, r1075, r1099;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1075;
mov.b32 r1110, {high, low};
}
{
fma.rn.f16x2 r1112, r1104, r1110, r1107;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1112;
mov.b32 r1116, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1112;
mov.b32 r1118, {high, high};
}
{
mul.f16x2 r1120, r728, r1118;
}
{
fma.rn.f16x2 r1123, r725, r1116, r1120;
}
{
mul.f16x2 r1127, r725, r1118;
}
{
neg.f16x2 r1130, r1127;
}
{
fma.rn.f16x2 r1132, r728, r1116, r1130;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1136, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1138, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r1140, {low, high};
}
{
mul.f16x2 r1141, r1138, r1140;
}
{
mul.f16x2 r1144, r1112, r1136;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1112;
mov.b32 r1147, {high, low};
}
{
fma.rn.f16x2 r1149, r1141, r1147, r1144;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1149;
mov.b32 r1153, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1149;
mov.b32 r1155, {high, high};
}
{
mul.f16x2 r1157, r574, r1155;
}
{
fma.rn.f16x2 r1160, r571, r1153, r1157;
}
{
mul.f16x2 r1164, r571, r1155;
}
{
neg.f16x2 r1167, r1164;
}
{
fma.rn.f16x2 r1169, r574, r1153, r1167;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1173, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1175, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r1177, {low, high};
}
{
mul.f16x2 r1178, r1175, r1177;
}
{
mul.f16x2 r1181, r1149, r1173;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1149;
mov.b32 r1184, {high, low};
}
{
fma.rn.f16x2 r1186, r1178, r1184, r1181;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1186;
mov.b32 r1190, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1186;
mov.b32 r1192, {high, high};
}
{
mul.f16x2 r1194, r420, r1192;
}
{
fma.rn.f16x2 r1197, r417, r1190, r1194;
}
{
mul.f16x2 r1201, r417, r1192;
}
{
neg.f16x2 r1204, r1201;
}
{
fma.rn.f16x2 r1206, r420, r1190, r1204;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1210, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r891;
mov.b32 r1212, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r1214, {low, high};
}
{
mul.f16x2 r1215, r1212, r1214;
}
{
mul.f16x2 r1218, r1186, r1210;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1186;
mov.b32 r1221, {high, low};
}
{
fma.rn.f16x2 r1223, r1215, r1221, r1218;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1223;
mov.b32 r1227, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1223;
mov.b32 r1229, {high, high};
}
{
mul.f16x2 r1231, r266, r1229;
}
{
fma.rn.f16x2 r1234, r263, r1227, r1231;
}
{
mul.f16x2 r1238, r263, r1229;
}
{
neg.f16x2 r1241, r1238;
}
{
fma.rn.f16x2 r1243, r266, r1227, r1241;
}
barrier.sync 0;
mad.lo.s32 r3366, r24, 44, r3365;
st.shared.u32 [r3366], r109;
st.shared.u32 [r3366+4], r901;
st.shared.u32 [r3366+8], r938;
st.shared.u32 [r3366+12], r975;
st.shared.u32 [r3366+16], r1012;
st.shared.u32 [r3366+20], r1049;
st.shared.u32 [r3366+24], r1086;
st.shared.u32 [r3366+28], r1123;
st.shared.u32 [r3366+32], r1160;
st.shared.u32 [r3366+36], r1197;
st.shared.u32 [r3366+40], r1234;
barrier.sync 0;
mad.lo.s32 r3367, r24, -40, r3366;
ld.shared.u32 r1325, [r3367];
ld.shared.u32 r1265, [r3367+484];
ld.shared.u32 r1277, [r3367+968];
ld.shared.u32 r1289, [r3367+1452];
ld.shared.u32 r1301, [r3367+1936];
ld.shared.u32 r1313, [r3367+2420];
ld.shared.u32 r1314, [r3367+2904];
ld.shared.u32 r1302, [r3367+3388];
ld.shared.u32 r1290, [r3367+3872];
ld.shared.u32 r1278, [r3367+4356];
ld.shared.u32 r1266, [r3367+4840];
barrier.sync 0;
st.shared.u32 [r3366], r112;
st.shared.u32 [r3366+4], r910;
st.shared.u32 [r3366+8], r947;
st.shared.u32 [r3366+12], r984;
st.shared.u32 [r3366+16], r1021;
st.shared.u32 [r3366+20], r1058;
st.shared.u32 [r3366+24], r1095;
st.shared.u32 [r3366+28], r1132;
st.shared.u32 [r3366+32], r1169;
st.shared.u32 [r3366+36], r1206;
st.shared.u32 [r3366+40], r1243;
barrier.sync 0;
ld.shared.u32 r1328, [r3367];
ld.shared.u32 r1268, [r3367+484];
ld.shared.u32 r1280, [r3367+968];
ld.shared.u32 r1292, [r3367+1452];
ld.shared.u32 r1304, [r3367+1936];
ld.shared.u32 r1316, [r3367+2420];
ld.shared.u32 r1317, [r3367+2904];
ld.shared.u32 r1305, [r3367+3388];
ld.shared.u32 r1293, [r3367+3872];
ld.shared.u32 r1281, [r3367+4356];
ld.shared.u32 r1269, [r3367+4840];
{
add.f16x2 r1264, r1265, r1266;
}
{
add.f16x2 r1267, r1268, r1269;
}
{
sub.f16x2 r1270, r1265, r1266;
}
{
sub.f16x2 r1273, r1268, r1269;
}
{
add.f16x2 r1276, r1277, r1278;
}
{
add.f16x2 r1279, r1280, r1281;
}
{
sub.f16x2 r1282, r1277, r1278;
}
{
sub.f16x2 r1285, r1280, r1281;
}
{
add.f16x2 r1288, r1289, r1290;
}
{
add.f16x2 r1291, r1292, r1293;
}
{
sub.f16x2 r1294, r1289, r1290;
}
{
sub.f16x2 r1297, r1292, r1293;
}
{
add.f16x2 r1300, r1301, r1302;
}
{
add.f16x2 r1303, r1304, r1305;
}
{
sub.f16x2 r1306, r1301, r1302;
}
{
sub.f16x2 r1309, r1304, r1305;
}
{
add.f16x2 r1312, r1313, r1314;
}
{
add.f16x2 r1315, r1316, r1317;
}
{
sub.f16x2 r1318, r1313, r1314;
}
{
sub.f16x2 r1321, r1316, r1317;
}
{
add.f16x2 r1324, r1325, r1264;
}
{
add.f16x2 r1327, r1328, r1267;
}
{
add.f16x2 r1330, r1324, r1276;
}
{
add.f16x2 r1333, r1327, r1279;
}
{
add.f16x2 r1336, r1330, r1288;
}
{
add.f16x2 r1339, r1333, r1291;
}
{
add.f16x2 r1342, r1336, r1300;
}
{
add.f16x2 r1345, r1339, r1303;
}
{
add.f16x2 r1348, r1342, r1312;
}
{
add.f16x2 r1351, r1345, r1315;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r1354, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r1355, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r1356, {low, high};
}
{
mul.f16x2 r1357, r1264, r1356;
}
{
add.f16x2 r1360, r1325, r1357;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r1363, {low, high};
}
{
mul.f16x2 r1364, r1273, r1363;
}
{
add.f16x2 r1367, r1354, r1364;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r1370, {low, high};
}
{
mul.f16x2 r1371, r1267, r1370;
}
{
add.f16x2 r1374, r1328, r1371;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r1377, {low, high};
}
{
mul.f16x2 r1378, r1270, r1377;
}
{
add.f16x2 r1381, r1355, r1378;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r1384, {low, high};
}
{
mul.f16x2 r1385, r1276, r1384;
}
{
add.f16x2 r1388, r1360, r1385;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r1391, {low, high};
}
{
mul.f16x2 r1392, r1285, r1391;
}
{
add.f16x2 r1395, r1367, r1392;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r1398, {low, high};
}
{
mul.f16x2 r1399, r1279, r1398;
}
{
add.f16x2 r1402, r1374, r1399;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r1405, {low, high};
}
{
mul.f16x2 r1406, r1282, r1405;
}
{
add.f16x2 r1409, r1381, r1406;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r1412, {low, high};
}
{
mul.f16x2 r1413, r1288, r1412;
}
{
add.f16x2 r1416, r1388, r1413;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r1419, {low, high};
}
{
mul.f16x2 r1420, r1297, r1419;
}
{
add.f16x2 r1423, r1395, r1420;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r1426, {low, high};
}
{
mul.f16x2 r1427, r1291, r1426;
}
{
add.f16x2 r1430, r1402, r1427;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r1433, {low, high};
}
{
mul.f16x2 r1434, r1294, r1433;
}
{
add.f16x2 r1437, r1409, r1434;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r1440, {low, high};
}
{
mul.f16x2 r1441, r1300, r1440;
}
{
add.f16x2 r1444, r1416, r1441;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r1447, {low, high};
}
{
mul.f16x2 r1448, r1309, r1447;
}
{
add.f16x2 r1451, r1423, r1448;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r1454, {low, high};
}
{
mul.f16x2 r1455, r1303, r1454;
}
{
add.f16x2 r1458, r1430, r1455;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r1461, {low, high};
}
{
mul.f16x2 r1462, r1306, r1461;
}
{
add.f16x2 r1465, r1437, r1462;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r1468, {low, high};
}
{
mul.f16x2 r1469, r1312, r1468;
}
{
add.f16x2 r1472, r1444, r1469;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r1475, {low, high};
}
{
mul.f16x2 r1476, r1321, r1475;
}
{
add.f16x2 r1479, r1451, r1476;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r1482, {low, high};
}
{
mul.f16x2 r1483, r1315, r1482;
}
{
add.f16x2 r1486, r1458, r1483;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r1489, {low, high};
}
{
mul.f16x2 r1490, r1318, r1489;
}
{
add.f16x2 r1493, r1465, r1490;
}
{
sub.f16x2 r1496, r1472, r1479;
}
{
add.f16x2 r1499, r1486, r1493;
}
{
add.f16x2 r1502, r1472, r1479;
}
{
sub.f16x2 r1505, r1486, r1493;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r1508, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r1509, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r1510, {low, high};
}
{
mul.f16x2 r1511, r1264, r1510;
}
{
add.f16x2 r1514, r1325, r1511;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r1517, {low, high};
}
{
mul.f16x2 r1518, r1273, r1517;
}
{
add.f16x2 r1521, r1508, r1518;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r1524, {low, high};
}
{
mul.f16x2 r1525, r1267, r1524;
}
{
add.f16x2 r1528, r1328, r1525;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r1531, {low, high};
}
{
mul.f16x2 r1532, r1270, r1531;
}
{
add.f16x2 r1535, r1509, r1532;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r1538, {low, high};
}
{
mul.f16x2 r1539, r1276, r1538;
}
{
add.f16x2 r1542, r1514, r1539;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r1545, {low, high};
}
{
mul.f16x2 r1546, r1285, r1545;
}
{
add.f16x2 r1549, r1521, r1546;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r1552, {low, high};
}
{
mul.f16x2 r1553, r1279, r1552;
}
{
add.f16x2 r1556, r1528, r1553;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r1559, {low, high};
}
{
mul.f16x2 r1560, r1282, r1559;
}
{
add.f16x2 r1563, r1535, r1560;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r1566, {low, high};
}
{
mul.f16x2 r1567, r1288, r1566;
}
{
add.f16x2 r1570, r1542, r1567;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r1573, {low, high};
}
{
mul.f16x2 r1574, r1297, r1573;
}
{
add.f16x2 r1577, r1549, r1574;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r1580, {low, high};
}
{
mul.f16x2 r1581, r1291, r1580;
}
{
add.f16x2 r1584, r1556, r1581;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r1587, {low, high};
}
{
mul.f16x2 r1588, r1294, r1587;
}
{
add.f16x2 r1591, r1563, r1588;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r1594, {low, high};
}
{
mul.f16x2 r1595, r1300, r1594;
}
{
add.f16x2 r1598, r1570, r1595;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r1601, {low, high};
}
{
mul.f16x2 r1602, r1309, r1601;
}
{
add.f16x2 r1605, r1577, r1602;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r1608, {low, high};
}
{
mul.f16x2 r1609, r1303, r1608;
}
{
add.f16x2 r1612, r1584, r1609;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r1615, {low, high};
}
{
mul.f16x2 r1616, r1306, r1615;
}
{
add.f16x2 r1619, r1591, r1616;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r1622, {low, high};
}
{
mul.f16x2 r1623, r1312, r1622;
}
{
add.f16x2 r1626, r1598, r1623;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r1629, {low, high};
}
{
mul.f16x2 r1630, r1321, r1629;
}
{
add.f16x2 r1633, r1605, r1630;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r1636, {low, high};
}
{
mul.f16x2 r1637, r1315, r1636;
}
{
add.f16x2 r1640, r1612, r1637;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r1643, {low, high};
}
{
mul.f16x2 r1644, r1318, r1643;
}
{
add.f16x2 r1647, r1619, r1644;
}
{
sub.f16x2 r1650, r1626, r1633;
}
{
add.f16x2 r1653, r1640, r1647;
}
{
add.f16x2 r1656, r1626, r1633;
}
{
sub.f16x2 r1659, r1640, r1647;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r1662, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r1663, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r1664, {low, high};
}
{
mul.f16x2 r1665, r1264, r1664;
}
{
add.f16x2 r1668, r1325, r1665;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r1671, {low, high};
}
{
mul.f16x2 r1672, r1273, r1671;
}
{
add.f16x2 r1675, r1662, r1672;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r1678, {low, high};
}
{
mul.f16x2 r1679, r1267, r1678;
}
{
add.f16x2 r1682, r1328, r1679;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r1685, {low, high};
}
{
mul.f16x2 r1686, r1270, r1685;
}
{
add.f16x2 r1689, r1663, r1686;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r1692, {low, high};
}
{
mul.f16x2 r1693, r1276, r1692;
}
{
add.f16x2 r1696, r1668, r1693;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r1699, {low, high};
}
{
mul.f16x2 r1700, r1285, r1699;
}
{
add.f16x2 r1703, r1675, r1700;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r1706, {low, high};
}
{
mul.f16x2 r1707, r1279, r1706;
}
{
add.f16x2 r1710, r1682, r1707;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r1713, {low, high};
}
{
mul.f16x2 r1714, r1282, r1713;
}
{
add.f16x2 r1717, r1689, r1714;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r1720, {low, high};
}
{
mul.f16x2 r1721, r1288, r1720;
}
{
add.f16x2 r1724, r1696, r1721;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r1727, {low, high};
}
{
mul.f16x2 r1728, r1297, r1727;
}
{
add.f16x2 r1731, r1703, r1728;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r1734, {low, high};
}
{
mul.f16x2 r1735, r1291, r1734;
}
{
add.f16x2 r1738, r1710, r1735;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r1741, {low, high};
}
{
mul.f16x2 r1742, r1294, r1741;
}
{
add.f16x2 r1745, r1717, r1742;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r1748, {low, high};
}
{
mul.f16x2 r1749, r1300, r1748;
}
{
add.f16x2 r1752, r1724, r1749;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r1755, {low, high};
}
{
mul.f16x2 r1756, r1309, r1755;
}
{
add.f16x2 r1759, r1731, r1756;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r1762, {low, high};
}
{
mul.f16x2 r1763, r1303, r1762;
}
{
add.f16x2 r1766, r1738, r1763;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r1769, {low, high};
}
{
mul.f16x2 r1770, r1306, r1769;
}
{
add.f16x2 r1773, r1745, r1770;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r1776, {low, high};
}
{
mul.f16x2 r1777, r1312, r1776;
}
{
add.f16x2 r1780, r1752, r1777;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r1783, {low, high};
}
{
mul.f16x2 r1784, r1321, r1783;
}
{
add.f16x2 r1787, r1759, r1784;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r1790, {low, high};
}
{
mul.f16x2 r1791, r1315, r1790;
}
{
add.f16x2 r1794, r1766, r1791;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r1797, {low, high};
}
{
mul.f16x2 r1798, r1318, r1797;
}
{
add.f16x2 r1801, r1773, r1798;
}
{
sub.f16x2 r1804, r1780, r1787;
}
{
add.f16x2 r1807, r1794, r1801;
}
{
add.f16x2 r1810, r1780, r1787;
}
{
sub.f16x2 r1813, r1794, r1801;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r1816, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r1817, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r1818, {low, high};
}
{
mul.f16x2 r1819, r1264, r1818;
}
{
add.f16x2 r1822, r1325, r1819;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r1825, {low, high};
}
{
mul.f16x2 r1826, r1273, r1825;
}
{
add.f16x2 r1829, r1816, r1826;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r1832, {low, high};
}
{
mul.f16x2 r1833, r1267, r1832;
}
{
add.f16x2 r1836, r1328, r1833;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r1839, {low, high};
}
{
mul.f16x2 r1840, r1270, r1839;
}
{
add.f16x2 r1843, r1817, r1840;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r1846, {low, high};
}
{
mul.f16x2 r1847, r1276, r1846;
}
{
add.f16x2 r1850, r1822, r1847;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r1853, {low, high};
}
{
mul.f16x2 r1854, r1285, r1853;
}
{
add.f16x2 r1857, r1829, r1854;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r1860, {low, high};
}
{
mul.f16x2 r1861, r1279, r1860;
}
{
add.f16x2 r1864, r1836, r1861;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r1867, {low, high};
}
{
mul.f16x2 r1868, r1282, r1867;
}
{
add.f16x2 r1871, r1843, r1868;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r1874, {low, high};
}
{
mul.f16x2 r1875, r1288, r1874;
}
{
add.f16x2 r1878, r1850, r1875;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r1881, {low, high};
}
{
mul.f16x2 r1882, r1297, r1881;
}
{
add.f16x2 r1885, r1857, r1882;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r1888, {low, high};
}
{
mul.f16x2 r1889, r1291, r1888;
}
{
add.f16x2 r1892, r1864, r1889;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r1895, {low, high};
}
{
mul.f16x2 r1896, r1294, r1895;
}
{
add.f16x2 r1899, r1871, r1896;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r1902, {low, high};
}
{
mul.f16x2 r1903, r1300, r1902;
}
{
add.f16x2 r1906, r1878, r1903;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r1909, {low, high};
}
{
mul.f16x2 r1910, r1309, r1909;
}
{
add.f16x2 r1913, r1885, r1910;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r1916, {low, high};
}
{
mul.f16x2 r1917, r1303, r1916;
}
{
add.f16x2 r1920, r1892, r1917;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r1923, {low, high};
}
{
mul.f16x2 r1924, r1306, r1923;
}
{
add.f16x2 r1927, r1899, r1924;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r1930, {low, high};
}
{
mul.f16x2 r1931, r1312, r1930;
}
{
add.f16x2 r1934, r1906, r1931;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r1937, {low, high};
}
{
mul.f16x2 r1938, r1321, r1937;
}
{
add.f16x2 r1941, r1913, r1938;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r1944, {low, high};
}
{
mul.f16x2 r1945, r1315, r1944;
}
{
add.f16x2 r1948, r1920, r1945;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r1951, {low, high};
}
{
mul.f16x2 r1952, r1318, r1951;
}
{
add.f16x2 r1955, r1927, r1952;
}
{
sub.f16x2 r1958, r1934, r1941;
}
{
add.f16x2 r1961, r1948, r1955;
}
{
add.f16x2 r1964, r1934, r1941;
}
{
sub.f16x2 r1967, r1948, r1955;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r1970, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r1971, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r1972, {low, high};
}
{
mul.f16x2 r1973, r1264, r1972;
}
{
add.f16x2 r1976, r1325, r1973;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r1979, {low, high};
}
{
mul.f16x2 r1980, r1273, r1979;
}
{
add.f16x2 r1983, r1970, r1980;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r1986, {low, high};
}
{
mul.f16x2 r1987, r1267, r1986;
}
{
add.f16x2 r1990, r1328, r1987;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r1993, {low, high};
}
{
mul.f16x2 r1994, r1270, r1993;
}
{
add.f16x2 r1997, r1971, r1994;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r2000, {low, high};
}
{
mul.f16x2 r2001, r1276, r2000;
}
{
add.f16x2 r2004, r1976, r2001;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r2007, {low, high};
}
{
mul.f16x2 r2008, r1285, r2007;
}
{
add.f16x2 r2011, r1983, r2008;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r2014, {low, high};
}
{
mul.f16x2 r2015, r1279, r2014;
}
{
add.f16x2 r2018, r1990, r2015;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r2021, {low, high};
}
{
mul.f16x2 r2022, r1282, r2021;
}
{
add.f16x2 r2025, r1997, r2022;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r2028, {low, high};
}
{
mul.f16x2 r2029, r1288, r2028;
}
{
add.f16x2 r2032, r2004, r2029;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r2035, {low, high};
}
{
mul.f16x2 r2036, r1297, r2035;
}
{
add.f16x2 r2039, r2011, r2036;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r2042, {low, high};
}
{
mul.f16x2 r2043, r1291, r2042;
}
{
add.f16x2 r2046, r2018, r2043;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r2049, {low, high};
}
{
mul.f16x2 r2050, r1294, r2049;
}
{
add.f16x2 r2053, r2025, r2050;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r2056, {low, high};
}
{
mul.f16x2 r2057, r1300, r2056;
}
{
add.f16x2 r2060, r2032, r2057;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r2063, {low, high};
}
{
mul.f16x2 r2064, r1309, r2063;
}
{
add.f16x2 r2067, r2039, r2064;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r2070, {low, high};
}
{
mul.f16x2 r2071, r1303, r2070;
}
{
add.f16x2 r2074, r2046, r2071;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r2077, {low, high};
}
{
mul.f16x2 r2078, r1306, r2077;
}
{
add.f16x2 r2081, r2053, r2078;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r2084, {low, high};
}
{
mul.f16x2 r2085, r1312, r2084;
}
{
add.f16x2 r2088, r2060, r2085;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r2091, {low, high};
}
{
mul.f16x2 r2092, r1321, r2091;
}
{
add.f16x2 r2095, r2067, r2092;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r2098, {low, high};
}
{
mul.f16x2 r2099, r1315, r2098;
}
{
add.f16x2 r2102, r2074, r2099;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r2105, {low, high};
}
{
mul.f16x2 r2106, r1318, r2105;
}
{
add.f16x2 r2109, r2081, r2106;
}
{
sub.f16x2 r2112, r2088, r2095;
}
{
add.f16x2 r2115, r2102, r2109;
}
{
add.f16x2 r2118, r2088, r2095;
}
{
sub.f16x2 r2121, r2102, r2109;
}
mul.wide.u32 rd6, r24, -1171354717;
shr.u64 rd7, rd6, 35;
cvt.u32.u64 r3368, rd7;
mul.lo.s32 r3369, r3368, 11;
sub.s32 r3370, r24, r3369;
shl.b32 r3371, r3370, 2;
add.s32 r3372, r3365, r3371;
cvt.rn.f32.u32 f714, r3368;
mul.f32 f715, f714, 0f3D54B191;
cos.approx.f32 f470, f715;
sin.approx.f32 f716, f715;
neg.f32 f471, f716;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f470;
cvt.rn.f16.f32 high, f471;
mov.b32 r2124, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2127, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2129, {high, high};
}
{
mul.f16x2 r2131, r1499, r2129;
}
{
fma.rn.f16x2 r2134, r1496, r2127, r2131;
}
{
mul.f16x2 r2138, r1496, r2129;
}
{
neg.f16x2 r2141, r2138;
}
{
fma.rn.f16x2 r2143, r1499, r2127, r2141;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2147, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2149, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r2151, {low, high};
}
{
mul.f16x2 r2152, r2149, r2151;
}
{
mul.f16x2 r2155, r2124, r2147;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2158, {high, low};
}
{
fma.rn.f16x2 r2160, r2152, r2158, r2155;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2160;
mov.b32 r2164, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2160;
mov.b32 r2166, {high, high};
}
{
mul.f16x2 r2168, r1653, r2166;
}
{
fma.rn.f16x2 r2171, r1650, r2164, r2168;
}
{
mul.f16x2 r2175, r1650, r2166;
}
{
neg.f16x2 r2178, r2175;
}
{
fma.rn.f16x2 r2180, r1653, r2164, r2178;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2184, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2186, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r2188, {low, high};
}
{
mul.f16x2 r2189, r2186, r2188;
}
{
mul.f16x2 r2192, r2160, r2184;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2160;
mov.b32 r2195, {high, low};
}
{
fma.rn.f16x2 r2197, r2189, r2195, r2192;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2197;
mov.b32 r2201, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2197;
mov.b32 r2203, {high, high};
}
{
mul.f16x2 r2205, r1807, r2203;
}
{
fma.rn.f16x2 r2208, r1804, r2201, r2205;
}
{
mul.f16x2 r2212, r1804, r2203;
}
{
neg.f16x2 r2215, r2212;
}
{
fma.rn.f16x2 r2217, r1807, r2201, r2215;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2221, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2223, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r2225, {low, high};
}
{
mul.f16x2 r2226, r2223, r2225;
}
{
mul.f16x2 r2229, r2197, r2221;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2197;
mov.b32 r2232, {high, low};
}
{
fma.rn.f16x2 r2234, r2226, r2232, r2229;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2234;
mov.b32 r2238, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2234;
mov.b32 r2240, {high, high};
}
{
mul.f16x2 r2242, r1961, r2240;
}
{
fma.rn.f16x2 r2245, r1958, r2238, r2242;
}
{
mul.f16x2 r2249, r1958, r2240;
}
{
neg.f16x2 r2252, r2249;
}
{
fma.rn.f16x2 r2254, r1961, r2238, r2252;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2258, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2260, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r2262, {low, high};
}
{
mul.f16x2 r2263, r2260, r2262;
}
{
mul.f16x2 r2266, r2234, r2258;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2234;
mov.b32 r2269, {high, low};
}
{
fma.rn.f16x2 r2271, r2263, r2269, r2266;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2271;
mov.b32 r2275, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2271;
mov.b32 r2277, {high, high};
}
{
mul.f16x2 r2279, r2115, r2277;
}
{
fma.rn.f16x2 r2282, r2112, r2275, r2279;
}
{
mul.f16x2 r2286, r2112, r2277;
}
{
neg.f16x2 r2289, r2286;
}
{
fma.rn.f16x2 r2291, r2115, r2275, r2289;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2295, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2297, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r2299, {low, high};
}
{
mul.f16x2 r2300, r2297, r2299;
}
{
mul.f16x2 r2303, r2271, r2295;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2271;
mov.b32 r2306, {high, low};
}
{
fma.rn.f16x2 r2308, r2300, r2306, r2303;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2308;
mov.b32 r2312, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2308;
mov.b32 r2314, {high, high};
}
{
mul.f16x2 r2316, r2121, r2314;
}
{
fma.rn.f16x2 r2319, r2118, r2312, r2316;
}
{
mul.f16x2 r2323, r2118, r2314;
}
{
neg.f16x2 r2326, r2323;
}
{
fma.rn.f16x2 r2328, r2121, r2312, r2326;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2332, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2334, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r2336, {low, high};
}
{
mul.f16x2 r2337, r2334, r2336;
}
{
mul.f16x2 r2340, r2308, r2332;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2308;
mov.b32 r2343, {high, low};
}
{
fma.rn.f16x2 r2345, r2337, r2343, r2340;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2345;
mov.b32 r2349, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2345;
mov.b32 r2351, {high, high};
}
{
mul.f16x2 r2353, r1967, r2351;
}
{
fma.rn.f16x2 r2356, r1964, r2349, r2353;
}
{
mul.f16x2 r2360, r1964, r2351;
}
{
neg.f16x2 r2363, r2360;
}
{
fma.rn.f16x2 r2365, r1967, r2349, r2363;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2369, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2371, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r2373, {low, high};
}
{
mul.f16x2 r2374, r2371, r2373;
}
{
mul.f16x2 r2377, r2345, r2369;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2345;
mov.b32 r2380, {high, low};
}
{
fma.rn.f16x2 r2382, r2374, r2380, r2377;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2382;
mov.b32 r2386, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2382;
mov.b32 r2388, {high, high};
}
{
mul.f16x2 r2390, r1813, r2388;
}
{
fma.rn.f16x2 r2393, r1810, r2386, r2390;
}
{
mul.f16x2 r2397, r1810, r2388;
}
{
neg.f16x2 r2400, r2397;
}
{
fma.rn.f16x2 r2402, r1813, r2386, r2400;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2406, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2408, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r2410, {low, high};
}
{
mul.f16x2 r2411, r2408, r2410;
}
{
mul.f16x2 r2414, r2382, r2406;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2382;
mov.b32 r2417, {high, low};
}
{
fma.rn.f16x2 r2419, r2411, r2417, r2414;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2419;
mov.b32 r2423, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2419;
mov.b32 r2425, {high, high};
}
{
mul.f16x2 r2427, r1659, r2425;
}
{
fma.rn.f16x2 r2430, r1656, r2423, r2427;
}
{
mul.f16x2 r2434, r1656, r2425;
}
{
neg.f16x2 r2437, r2434;
}
{
fma.rn.f16x2 r2439, r1659, r2423, r2437;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2443, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2124;
mov.b32 r2445, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f490;
cvt.rn.f16.f32 high, f491;
mov.b32 r2447, {low, high};
}
{
mul.f16x2 r2448, r2445, r2447;
}
{
mul.f16x2 r2451, r2419, r2443;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2419;
mov.b32 r2454, {high, low};
}
{
fma.rn.f16x2 r2456, r2448, r2454, r2451;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2456;
mov.b32 r2460, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r2456;
mov.b32 r2462, {high, high};
}
{
mul.f16x2 r2464, r1505, r2462;
}
{
fma.rn.f16x2 r2467, r1502, r2460, r2464;
}
{
mul.f16x2 r2471, r1502, r2462;
}
{
neg.f16x2 r2474, r2471;
}
{
fma.rn.f16x2 r2476, r1505, r2460, r2474;
}
barrier.sync 0;
mad.lo.s32 r3373, r3368, 484, r3372;
st.shared.u32 [r3373], r1348;
st.shared.u32 [r3373+44], r2134;
st.shared.u32 [r3373+88], r2171;
st.shared.u32 [r3373+132], r2208;
st.shared.u32 [r3373+176], r2245;
st.shared.u32 [r3373+220], r2282;
st.shared.u32 [r3373+264], r2319;
st.shared.u32 [r3373+308], r2356;
st.shared.u32 [r3373+352], r2393;
st.shared.u32 [r3373+396], r2430;
st.shared.u32 [r3373+440], r2467;
barrier.sync 0;
ld.shared.u32 r2558, [r3367];
ld.shared.u32 r2498, [r3367+484];
ld.shared.u32 r2510, [r3367+968];
ld.shared.u32 r2522, [r3367+1452];
ld.shared.u32 r2534, [r3367+1936];
ld.shared.u32 r2546, [r3367+2420];
ld.shared.u32 r2547, [r3367+2904];
ld.shared.u32 r2535, [r3367+3388];
ld.shared.u32 r2523, [r3367+3872];
ld.shared.u32 r2511, [r3367+4356];
ld.shared.u32 r2499, [r3367+4840];
barrier.sync 0;
st.shared.u32 [r3373], r1351;
st.shared.u32 [r3373+44], r2143;
st.shared.u32 [r3373+88], r2180;
st.shared.u32 [r3373+132], r2217;
st.shared.u32 [r3373+176], r2254;
st.shared.u32 [r3373+220], r2291;
st.shared.u32 [r3373+264], r2328;
st.shared.u32 [r3373+308], r2365;
st.shared.u32 [r3373+352], r2402;
st.shared.u32 [r3373+396], r2439;
st.shared.u32 [r3373+440], r2476;
barrier.sync 0;
ld.shared.u32 r2561, [r3367];
ld.shared.u32 r2501, [r3367+484];
ld.shared.u32 r2513, [r3367+968];
ld.shared.u32 r2525, [r3367+1452];
ld.shared.u32 r2537, [r3367+1936];
ld.shared.u32 r2549, [r3367+2420];
ld.shared.u32 r2550, [r3367+2904];
ld.shared.u32 r2538, [r3367+3388];
ld.shared.u32 r2526, [r3367+3872];
ld.shared.u32 r2514, [r3367+4356];
ld.shared.u32 r2502, [r3367+4840];
{
add.f16x2 r2497, r2498, r2499;
}
{
add.f16x2 r2500, r2501, r2502;
}
{
sub.f16x2 r2503, r2498, r2499;
}
{
sub.f16x2 r2506, r2501, r2502;
}
{
add.f16x2 r2509, r2510, r2511;
}
{
add.f16x2 r2512, r2513, r2514;
}
{
sub.f16x2 r2515, r2510, r2511;
}
{
sub.f16x2 r2518, r2513, r2514;
}
{
add.f16x2 r2521, r2522, r2523;
}
{
add.f16x2 r2524, r2525, r2526;
}
{
sub.f16x2 r2527, r2522, r2523;
}
{
sub.f16x2 r2530, r2525, r2526;
}
{
add.f16x2 r2533, r2534, r2535;
}
{
add.f16x2 r2536, r2537, r2538;
}
{
sub.f16x2 r2539, r2534, r2535;
}
{
sub.f16x2 r2542, r2537, r2538;
}
{
add.f16x2 r2545, r2546, r2547;
}
{
add.f16x2 r2548, r2549, r2550;
}
{
sub.f16x2 r2551, r2546, r2547;
}
{
sub.f16x2 r2554, r2549, r2550;
}
{
add.f16x2 r2557, r2558, r2497;
}
{
add.f16x2 r2560, r2561, r2500;
}
{
add.f16x2 r2563, r2557, r2509;
}
{
add.f16x2 r2566, r2560, r2512;
}
{
add.f16x2 r2569, r2563, r2521;
}
{
add.f16x2 r2572, r2566, r2524;
}
{
add.f16x2 r2575, r2569, r2533;
}
{
add.f16x2 r2578, r2572, r2536;
}
{
add.f16x2 %0, r2575, r2545;
}
{
add.f16x2 %1, r2578, r2548;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r2587, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r2588, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r2589, {low, high};
}
{
mul.f16x2 r2590, r2497, r2589;
}
{
add.f16x2 r2593, r2558, r2590;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r2596, {low, high};
}
{
mul.f16x2 r2597, r2506, r2596;
}
{
add.f16x2 r2600, r2587, r2597;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r2603, {low, high};
}
{
mul.f16x2 r2604, r2500, r2603;
}
{
add.f16x2 r2607, r2561, r2604;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r2610, {low, high};
}
{
mul.f16x2 r2611, r2503, r2610;
}
{
add.f16x2 r2614, r2588, r2611;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r2617, {low, high};
}
{
mul.f16x2 r2618, r2509, r2617;
}
{
add.f16x2 r2621, r2593, r2618;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r2624, {low, high};
}
{
mul.f16x2 r2625, r2518, r2624;
}
{
add.f16x2 r2628, r2600, r2625;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r2631, {low, high};
}
{
mul.f16x2 r2632, r2512, r2631;
}
{
add.f16x2 r2635, r2607, r2632;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r2638, {low, high};
}
{
mul.f16x2 r2639, r2515, r2638;
}
{
add.f16x2 r2642, r2614, r2639;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r2645, {low, high};
}
{
mul.f16x2 r2646, r2521, r2645;
}
{
add.f16x2 r2649, r2621, r2646;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r2652, {low, high};
}
{
mul.f16x2 r2653, r2530, r2652;
}
{
add.f16x2 r2656, r2628, r2653;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r2659, {low, high};
}
{
mul.f16x2 r2660, r2524, r2659;
}
{
add.f16x2 r2663, r2635, r2660;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r2666, {low, high};
}
{
mul.f16x2 r2667, r2527, r2666;
}
{
add.f16x2 r2670, r2642, r2667;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r2673, {low, high};
}
{
mul.f16x2 r2674, r2533, r2673;
}
{
add.f16x2 r2677, r2649, r2674;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r2680, {low, high};
}
{
mul.f16x2 r2681, r2542, r2680;
}
{
add.f16x2 r2684, r2656, r2681;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r2687, {low, high};
}
{
mul.f16x2 r2688, r2536, r2687;
}
{
add.f16x2 r2691, r2663, r2688;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r2694, {low, high};
}
{
mul.f16x2 r2695, r2539, r2694;
}
{
add.f16x2 r2698, r2670, r2695;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r2701, {low, high};
}
{
mul.f16x2 r2702, r2545, r2701;
}
{
add.f16x2 r2705, r2677, r2702;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r2708, {low, high};
}
{
mul.f16x2 r2709, r2554, r2708;
}
{
add.f16x2 r2712, r2684, r2709;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r2715, {low, high};
}
{
mul.f16x2 r2716, r2548, r2715;
}
{
add.f16x2 r2719, r2691, r2716;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r2722, {low, high};
}
{
mul.f16x2 r2723, r2551, r2722;
}
{
add.f16x2 r2726, r2698, r2723;
}
{
sub.f16x2 %2, r2705, r2712;
}
{
add.f16x2 %3, r2719, r2726;
}
{
add.f16x2 %20, r2705, r2712;
}
{
sub.f16x2 %21, r2719, r2726;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r2741, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r2742, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r2743, {low, high};
}
{
mul.f16x2 r2744, r2497, r2743;
}
{
add.f16x2 r2747, r2558, r2744;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r2750, {low, high};
}
{
mul.f16x2 r2751, r2506, r2750;
}
{
add.f16x2 r2754, r2741, r2751;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r2757, {low, high};
}
{
mul.f16x2 r2758, r2500, r2757;
}
{
add.f16x2 r2761, r2561, r2758;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f59;
cvt.rn.f16.f32 high, f59;
mov.b32 r2764, {low, high};
}
{
mul.f16x2 r2765, r2503, r2764;
}
{
add.f16x2 r2768, r2742, r2765;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r2771, {low, high};
}
{
mul.f16x2 r2772, r2509, r2771;
}
{
add.f16x2 r2775, r2747, r2772;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r2778, {low, high};
}
{
mul.f16x2 r2779, r2518, r2778;
}
{
add.f16x2 r2782, r2754, r2779;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r2785, {low, high};
}
{
mul.f16x2 r2786, r2512, r2785;
}
{
add.f16x2 r2789, r2761, r2786;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r2792, {low, high};
}
{
mul.f16x2 r2793, r2515, r2792;
}
{
add.f16x2 r2796, r2768, r2793;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r2799, {low, high};
}
{
mul.f16x2 r2800, r2521, r2799;
}
{
add.f16x2 r2803, r2775, r2800;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r2806, {low, high};
}
{
mul.f16x2 r2807, r2530, r2806;
}
{
add.f16x2 r2810, r2782, r2807;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r2813, {low, high};
}
{
mul.f16x2 r2814, r2524, r2813;
}
{
add.f16x2 r2817, r2789, r2814;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r2820, {low, high};
}
{
mul.f16x2 r2821, r2527, r2820;
}
{
add.f16x2 r2824, r2796, r2821;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r2827, {low, high};
}
{
mul.f16x2 r2828, r2533, r2827;
}
{
add.f16x2 r2831, r2803, r2828;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r2834, {low, high};
}
{
mul.f16x2 r2835, r2542, r2834;
}
{
add.f16x2 r2838, r2810, r2835;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r2841, {low, high};
}
{
mul.f16x2 r2842, r2536, r2841;
}
{
add.f16x2 r2845, r2817, r2842;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r2848, {low, high};
}
{
mul.f16x2 r2849, r2539, r2848;
}
{
add.f16x2 r2852, r2824, r2849;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r2855, {low, high};
}
{
mul.f16x2 r2856, r2545, r2855;
}
{
add.f16x2 r2859, r2831, r2856;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r2862, {low, high};
}
{
mul.f16x2 r2863, r2554, r2862;
}
{
add.f16x2 r2866, r2838, r2863;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r2869, {low, high};
}
{
mul.f16x2 r2870, r2548, r2869;
}
{
add.f16x2 r2873, r2845, r2870;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r2876, {low, high};
}
{
mul.f16x2 r2877, r2551, r2876;
}
{
add.f16x2 r2880, r2852, r2877;
}
{
sub.f16x2 %4, r2859, r2866;
}
{
add.f16x2 %5, r2873, r2880;
}
{
add.f16x2 %18, r2859, r2866;
}
{
sub.f16x2 %19, r2873, r2880;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r2895, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r2896, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r2897, {low, high};
}
{
mul.f16x2 r2898, r2497, r2897;
}
{
add.f16x2 r2901, r2558, r2898;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r2904, {low, high};
}
{
mul.f16x2 r2905, r2506, r2904;
}
{
add.f16x2 r2908, r2895, r2905;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r2911, {low, high};
}
{
mul.f16x2 r2912, r2500, r2911;
}
{
add.f16x2 r2915, r2561, r2912;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r2918, {low, high};
}
{
mul.f16x2 r2919, r2503, r2918;
}
{
add.f16x2 r2922, r2896, r2919;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r2925, {low, high};
}
{
mul.f16x2 r2926, r2509, r2925;
}
{
add.f16x2 r2929, r2901, r2926;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r2932, {low, high};
}
{
mul.f16x2 r2933, r2518, r2932;
}
{
add.f16x2 r2936, r2908, r2933;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r2939, {low, high};
}
{
mul.f16x2 r2940, r2512, r2939;
}
{
add.f16x2 r2943, r2915, r2940;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f111;
cvt.rn.f16.f32 high, f111;
mov.b32 r2946, {low, high};
}
{
mul.f16x2 r2947, r2515, r2946;
}
{
add.f16x2 r2950, r2922, r2947;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r2953, {low, high};
}
{
mul.f16x2 r2954, r2521, r2953;
}
{
add.f16x2 r2957, r2929, r2954;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r2960, {low, high};
}
{
mul.f16x2 r2961, r2530, r2960;
}
{
add.f16x2 r2964, r2936, r2961;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r2967, {low, high};
}
{
mul.f16x2 r2968, r2524, r2967;
}
{
add.f16x2 r2971, r2943, r2968;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r2974, {low, high};
}
{
mul.f16x2 r2975, r2527, r2974;
}
{
add.f16x2 r2978, r2950, r2975;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r2981, {low, high};
}
{
mul.f16x2 r2982, r2533, r2981;
}
{
add.f16x2 r2985, r2957, r2982;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r2988, {low, high};
}
{
mul.f16x2 r2989, r2542, r2988;
}
{
add.f16x2 r2992, r2964, r2989;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r2995, {low, high};
}
{
mul.f16x2 r2996, r2536, r2995;
}
{
add.f16x2 r2999, r2971, r2996;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r3002, {low, high};
}
{
mul.f16x2 r3003, r2539, r3002;
}
{
add.f16x2 r3006, r2978, r3003;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r3009, {low, high};
}
{
mul.f16x2 r3010, r2545, r3009;
}
{
add.f16x2 r3013, r2985, r3010;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r3016, {low, high};
}
{
mul.f16x2 r3017, r2554, r3016;
}
{
add.f16x2 r3020, r2992, r3017;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r3023, {low, high};
}
{
mul.f16x2 r3024, r2548, r3023;
}
{
add.f16x2 r3027, r2999, r3024;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r3030, {low, high};
}
{
mul.f16x2 r3031, r2551, r3030;
}
{
add.f16x2 r3034, r3006, r3031;
}
{
sub.f16x2 %6, r3013, r3020;
}
{
add.f16x2 %7, r3027, r3034;
}
{
add.f16x2 %16, r3013, r3020;
}
{
sub.f16x2 %17, r3027, r3034;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r3049, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r3050, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r3051, {low, high};
}
{
mul.f16x2 r3052, r2497, r3051;
}
{
add.f16x2 r3055, r2558, r3052;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r3058, {low, high};
}
{
mul.f16x2 r3059, r2506, r3058;
}
{
add.f16x2 r3062, r3049, r3059;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r3065, {low, high};
}
{
mul.f16x2 r3066, r2500, r3065;
}
{
add.f16x2 r3069, r2561, r3066;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r3072, {low, high};
}
{
mul.f16x2 r3073, r2503, r3072;
}
{
add.f16x2 r3076, r3050, r3073;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r3079, {low, high};
}
{
mul.f16x2 r3080, r2509, r3079;
}
{
add.f16x2 r3083, r3055, r3080;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r3086, {low, high};
}
{
mul.f16x2 r3087, r2518, r3086;
}
{
add.f16x2 r3090, r3062, r3087;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r3093, {low, high};
}
{
mul.f16x2 r3094, r2512, r3093;
}
{
add.f16x2 r3097, r3069, r3094;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f155;
cvt.rn.f16.f32 high, f155;
mov.b32 r3100, {low, high};
}
{
mul.f16x2 r3101, r2515, r3100;
}
{
add.f16x2 r3104, r3076, r3101;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r3107, {low, high};
}
{
mul.f16x2 r3108, r2521, r3107;
}
{
add.f16x2 r3111, r3083, r3108;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r3114, {low, high};
}
{
mul.f16x2 r3115, r2530, r3114;
}
{
add.f16x2 r3118, r3090, r3115;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r3121, {low, high};
}
{
mul.f16x2 r3122, r2524, r3121;
}
{
add.f16x2 r3125, r3097, r3122;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f163;
cvt.rn.f16.f32 high, f163;
mov.b32 r3128, {low, high};
}
{
mul.f16x2 r3129, r2527, r3128;
}
{
add.f16x2 r3132, r3104, r3129;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r3135, {low, high};
}
{
mul.f16x2 r3136, r2533, r3135;
}
{
add.f16x2 r3139, r3111, r3136;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r3142, {low, high};
}
{
mul.f16x2 r3143, r2542, r3142;
}
{
add.f16x2 r3146, r3118, r3143;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r3149, {low, high};
}
{
mul.f16x2 r3150, r2536, r3149;
}
{
add.f16x2 r3153, r3125, r3150;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r3156, {low, high};
}
{
mul.f16x2 r3157, r2539, r3156;
}
{
add.f16x2 r3160, r3132, r3157;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r3163, {low, high};
}
{
mul.f16x2 r3164, r2545, r3163;
}
{
add.f16x2 r3167, r3139, r3164;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r3170, {low, high};
}
{
mul.f16x2 r3171, r2554, r3170;
}
{
add.f16x2 r3174, r3146, r3171;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r3177, {low, high};
}
{
mul.f16x2 r3178, r2548, r3177;
}
{
add.f16x2 r3181, r3153, r3178;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r3184, {low, high};
}
{
mul.f16x2 r3185, r2551, r3184;
}
{
add.f16x2 r3188, r3160, r3185;
}
{
sub.f16x2 %8, r3167, r3174;
}
{
add.f16x2 %9, r3181, r3188;
}
{
add.f16x2 %14, r3167, r3174;
}
{
sub.f16x2 %15, r3181, r3188;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r3203, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f183;
cvt.rn.f16.f32 high, f183;
mov.b32 r3204, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r3205, {low, high};
}
{
mul.f16x2 r3206, r2497, r3205;
}
{
add.f16x2 r3209, r2558, r3206;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r3212, {low, high};
}
{
mul.f16x2 r3213, r2506, r3212;
}
{
add.f16x2 r3216, r3203, r3213;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f189;
cvt.rn.f16.f32 high, f189;
mov.b32 r3219, {low, high};
}
{
mul.f16x2 r3220, r2500, r3219;
}
{
add.f16x2 r3223, r2561, r3220;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f191;
cvt.rn.f16.f32 high, f191;
mov.b32 r3226, {low, high};
}
{
mul.f16x2 r3227, r2503, r3226;
}
{
add.f16x2 r3230, r3204, r3227;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r3233, {low, high};
}
{
mul.f16x2 r3234, r2509, r3233;
}
{
add.f16x2 r3237, r3209, r3234;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r3240, {low, high};
}
{
mul.f16x2 r3241, r2518, r3240;
}
{
add.f16x2 r3244, r3216, r3241;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f197;
cvt.rn.f16.f32 high, f197;
mov.b32 r3247, {low, high};
}
{
mul.f16x2 r3248, r2512, r3247;
}
{
add.f16x2 r3251, r3223, r3248;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f199;
cvt.rn.f16.f32 high, f199;
mov.b32 r3254, {low, high};
}
{
mul.f16x2 r3255, r2515, r3254;
}
{
add.f16x2 r3258, r3230, r3255;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r3261, {low, high};
}
{
mul.f16x2 r3262, r2521, r3261;
}
{
add.f16x2 r3265, r3237, r3262;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r3268, {low, high};
}
{
mul.f16x2 r3269, r2530, r3268;
}
{
add.f16x2 r3272, r3244, r3269;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f205;
cvt.rn.f16.f32 high, f205;
mov.b32 r3275, {low, high};
}
{
mul.f16x2 r3276, r2524, r3275;
}
{
add.f16x2 r3279, r3251, r3276;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f207;
cvt.rn.f16.f32 high, f207;
mov.b32 r3282, {low, high};
}
{
mul.f16x2 r3283, r2527, r3282;
}
{
add.f16x2 r3286, r3258, r3283;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r3289, {low, high};
}
{
mul.f16x2 r3290, r2533, r3289;
}
{
add.f16x2 r3293, r3265, r3290;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r3296, {low, high};
}
{
mul.f16x2 r3297, r2542, r3296;
}
{
add.f16x2 r3300, r3272, r3297;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f213;
cvt.rn.f16.f32 high, f213;
mov.b32 r3303, {low, high};
}
{
mul.f16x2 r3304, r2536, r3303;
}
{
add.f16x2 r3307, r3279, r3304;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f215;
cvt.rn.f16.f32 high, f215;
mov.b32 r3310, {low, high};
}
{
mul.f16x2 r3311, r2539, r3310;
}
{
add.f16x2 r3314, r3286, r3311;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r3317, {low, high};
}
{
mul.f16x2 r3318, r2545, r3317;
}
{
add.f16x2 r3321, r3293, r3318;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r3324, {low, high};
}
{
mul.f16x2 r3325, r2554, r3324;
}
{
add.f16x2 r3328, r3300, r3325;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f221;
mov.b32 r3331, {low, high};
}
{
mul.f16x2 r3332, r2548, r3331;
}
{
add.f16x2 r3335, r3307, r3332;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f223;
cvt.rn.f16.f32 high, f223;
mov.b32 r3338, {low, high};
}
{
mul.f16x2 r3339, r2551, r3338;
}
{
add.f16x2 r3342, r3314, r3339;
}
{
sub.f16x2 %10, r3321, r3328;
}
{
add.f16x2 %11, r3335, r3342;
}
{
add.f16x2 %12, r3321, r3328;
}
{
sub.f16x2 %13, r3335, r3342;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)));
};


#endif
