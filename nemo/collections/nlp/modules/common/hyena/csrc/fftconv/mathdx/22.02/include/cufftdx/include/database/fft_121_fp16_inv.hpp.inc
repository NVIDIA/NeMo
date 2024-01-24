#ifndef CUFFTDX_FFT_121_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_121_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1127, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<468>;
.reg .b32 r<2104>;
.reg .b64 rd<4>;
mov.u32 r2094, %tid.y;
mov.u32 r2095, %22;
mad.lo.s32 r2096, r2094, 968, r2095;
mov.u32 r2097, %tid.x;
{
add.f16x2 r1, %25, %43;
}
{
add.f16x2 r4, %26, %44;
}
{
sub.f16x2 r7, %25, %43;
}
{
sub.f16x2 r10, %26, %44;
}
{
add.f16x2 r13, %27, %41;
}
{
add.f16x2 r16, %28, %42;
}
{
sub.f16x2 r19, %27, %41;
}
{
sub.f16x2 r22, %28, %42;
}
{
add.f16x2 r25, %29, %39;
}
{
add.f16x2 r28, %30, %40;
}
{
sub.f16x2 r31, %29, %39;
}
{
sub.f16x2 r34, %30, %40;
}
{
add.f16x2 r37, %31, %37;
}
{
add.f16x2 r40, %32, %38;
}
{
sub.f16x2 r43, %31, %37;
}
{
sub.f16x2 r46, %32, %38;
}
{
add.f16x2 r49, %33, %35;
}
{
add.f16x2 r52, %34, %36;
}
{
sub.f16x2 r55, %33, %35;
}
{
sub.f16x2 r58, %34, %36;
}
{
add.f16x2 r61, %23, r1;
}
{
add.f16x2 r64, %24, r4;
}
{
add.f16x2 r67, r61, r13;
}
{
add.f16x2 r70, r64, r16;
}
{
add.f16x2 r73, r67, r25;
}
{
add.f16x2 r76, r70, r28;
}
{
add.f16x2 r79, r73, r37;
}
{
add.f16x2 r82, r76, r40;
}
{
add.f16x2 r85, r79, r49;
}
{
add.f16x2 r88, r82, r52;
}
mov.f32 f424, 0f00000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r91, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r92, {low, high};
}
mov.f32 f438, 0f3F575C64;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r93, {low, high};
}
{
mul.f16x2 r94, r1, r93;
}
{
add.f16x2 r97, %23, r94;
}
mov.f32 f404, 0f3F0A6770;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r100, {low, high};
}
{
mul.f16x2 r101, r10, r100;
}
{
add.f16x2 r104, r91, r101;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r107, {low, high};
}
{
mul.f16x2 r108, r4, r107;
}
{
add.f16x2 r111, %24, r108;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r114, {low, high};
}
{
mul.f16x2 r115, r7, r114;
}
{
add.f16x2 r118, r92, r115;
}
mov.f32 f454, 0f3ED4B147;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r121, {low, high};
}
{
mul.f16x2 r122, r13, r121;
}
{
add.f16x2 r125, r97, r122;
}
mov.f32 f300, 0f3F68DDA4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f300;
cvt.rn.f16.f32 high, f300;
mov.b32 r128, {low, high};
}
{
mul.f16x2 r129, r22, r128;
}
{
add.f16x2 r132, r104, r129;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r135, {low, high};
}
{
mul.f16x2 r136, r16, r135;
}
{
add.f16x2 r139, r111, r136;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f300;
cvt.rn.f16.f32 high, f300;
mov.b32 r142, {low, high};
}
{
mul.f16x2 r143, r19, r142;
}
{
add.f16x2 r146, r118, r143;
}
mov.f32 f462, 0fBE11BAFB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r149, {low, high};
}
{
mul.f16x2 r150, r25, r149;
}
{
add.f16x2 r153, r125, r150;
}
mov.f32 f464, 0f3F7D64F0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r156, {low, high};
}
{
mul.f16x2 r157, r34, r156;
}
{
add.f16x2 r160, r132, r157;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r163, {low, high};
}
{
mul.f16x2 r164, r28, r163;
}
{
add.f16x2 r167, r139, r164;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r170, {low, high};
}
{
mul.f16x2 r171, r31, r170;
}
{
add.f16x2 r174, r146, r171;
}
mov.f32 f446, 0fBF27A4F4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r177, {low, high};
}
{
mul.f16x2 r178, r37, r177;
}
{
add.f16x2 r181, r153, r178;
}
mov.f32 f448, 0f3F4178CE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r184, {low, high};
}
{
mul.f16x2 r185, r46, r184;
}
{
add.f16x2 r188, r160, r185;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r191, {low, high};
}
{
mul.f16x2 r192, r40, r191;
}
{
add.f16x2 r195, r167, r192;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r198, {low, high};
}
{
mul.f16x2 r199, r43, r198;
}
{
add.f16x2 r202, r174, r199;
}
mov.f32 f430, 0fBF75A155;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r205, {low, high};
}
{
mul.f16x2 r206, r49, r205;
}
{
add.f16x2 r209, r181, r206;
}
mov.f32 f432, 0f3E903F40;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r212, {low, high};
}
{
mul.f16x2 r213, r58, r212;
}
{
add.f16x2 r216, r188, r213;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r219, {low, high};
}
{
mul.f16x2 r220, r52, r219;
}
{
add.f16x2 r223, r195, r220;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r226, {low, high};
}
{
mul.f16x2 r227, r55, r226;
}
{
add.f16x2 r230, r202, r227;
}
{
sub.f16x2 r233, r209, r216;
}
{
add.f16x2 r236, r223, r230;
}
{
add.f16x2 r239, r209, r216;
}
{
sub.f16x2 r242, r223, r230;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r245, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r246, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r247, {low, high};
}
{
mul.f16x2 r248, r1, r247;
}
{
add.f16x2 r251, %23, r248;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f300;
cvt.rn.f16.f32 high, f300;
mov.b32 r254, {low, high};
}
{
mul.f16x2 r255, r10, r254;
}
{
add.f16x2 r258, r245, r255;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r261, {low, high};
}
{
mul.f16x2 r262, r4, r261;
}
{
add.f16x2 r265, %24, r262;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f300;
cvt.rn.f16.f32 high, f300;
mov.b32 r268, {low, high};
}
{
mul.f16x2 r269, r7, r268;
}
{
add.f16x2 r272, r246, r269;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r275, {low, high};
}
{
mul.f16x2 r276, r13, r275;
}
{
add.f16x2 r279, r251, r276;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r282, {low, high};
}
{
mul.f16x2 r283, r22, r282;
}
{
add.f16x2 r286, r258, r283;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r289, {low, high};
}
{
mul.f16x2 r290, r16, r289;
}
{
add.f16x2 r293, r265, r290;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r296, {low, high};
}
{
mul.f16x2 r297, r19, r296;
}
{
add.f16x2 r300, r272, r297;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r303, {low, high};
}
{
mul.f16x2 r304, r25, r303;
}
{
add.f16x2 r307, r279, r304;
}
mov.f32 f352, 0fBE903F40;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r310, {low, high};
}
{
mul.f16x2 r311, r34, r310;
}
{
add.f16x2 r314, r286, r311;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r317, {low, high};
}
{
mul.f16x2 r318, r28, r317;
}
{
add.f16x2 r321, r293, r318;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r324, {low, high};
}
{
mul.f16x2 r325, r31, r324;
}
{
add.f16x2 r328, r300, r325;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r331, {low, high};
}
{
mul.f16x2 r332, r37, r331;
}
{
add.f16x2 r335, r307, r332;
}
mov.f32 f396, 0fBF7D64F0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f396;
cvt.rn.f16.f32 high, f396;
mov.b32 r338, {low, high};
}
{
mul.f16x2 r339, r46, r338;
}
{
add.f16x2 r342, r314, r339;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r345, {low, high};
}
{
mul.f16x2 r346, r40, r345;
}
{
add.f16x2 r349, r321, r346;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f396;
cvt.rn.f16.f32 high, f396;
mov.b32 r352, {low, high};
}
{
mul.f16x2 r353, r43, r352;
}
{
add.f16x2 r356, r328, r353;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r359, {low, high};
}
{
mul.f16x2 r360, r49, r359;
}
{
add.f16x2 r363, r335, r360;
}
mov.f32 f440, 0fBF0A6770;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f440;
cvt.rn.f16.f32 high, f440;
mov.b32 r366, {low, high};
}
{
mul.f16x2 r367, r58, r366;
}
{
add.f16x2 r370, r342, r367;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r373, {low, high};
}
{
mul.f16x2 r374, r52, r373;
}
{
add.f16x2 r377, r349, r374;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f440;
cvt.rn.f16.f32 high, f440;
mov.b32 r380, {low, high};
}
{
mul.f16x2 r381, r55, r380;
}
{
add.f16x2 r384, r356, r381;
}
{
sub.f16x2 r387, r363, r370;
}
{
add.f16x2 r390, r377, r384;
}
{
add.f16x2 r393, r363, r370;
}
{
sub.f16x2 r396, r377, r384;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r399, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r400, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r401, {low, high};
}
{
mul.f16x2 r402, r1, r401;
}
{
add.f16x2 r405, %23, r402;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r408, {low, high};
}
{
mul.f16x2 r409, r10, r408;
}
{
add.f16x2 r412, r399, r409;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r415, {low, high};
}
{
mul.f16x2 r416, r4, r415;
}
{
add.f16x2 r419, %24, r416;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r422, {low, high};
}
{
mul.f16x2 r423, r7, r422;
}
{
add.f16x2 r426, r400, r423;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r429, {low, high};
}
{
mul.f16x2 r430, r13, r429;
}
{
add.f16x2 r433, r405, r430;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r436, {low, high};
}
{
mul.f16x2 r437, r22, r436;
}
{
add.f16x2 r440, r412, r437;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r443, {low, high};
}
{
mul.f16x2 r444, r16, r443;
}
{
add.f16x2 r447, r419, r444;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r450, {low, high};
}
{
mul.f16x2 r451, r19, r450;
}
{
add.f16x2 r454, r426, r451;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r457, {low, high};
}
{
mul.f16x2 r458, r25, r457;
}
{
add.f16x2 r461, r433, r458;
}
mov.f32 f456, 0fBF68DDA4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r464, {low, high};
}
{
mul.f16x2 r465, r34, r464;
}
{
add.f16x2 r468, r440, r465;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r471, {low, high};
}
{
mul.f16x2 r472, r28, r471;
}
{
add.f16x2 r475, r447, r472;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r478, {low, high};
}
{
mul.f16x2 r479, r31, r478;
}
{
add.f16x2 r482, r454, r479;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r485, {low, high};
}
{
mul.f16x2 r486, r37, r485;
}
{
add.f16x2 r489, r461, r486;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r492, {low, high};
}
{
mul.f16x2 r493, r46, r492;
}
{
add.f16x2 r496, r468, r493;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r499, {low, high};
}
{
mul.f16x2 r500, r40, r499;
}
{
add.f16x2 r503, r475, r500;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r506, {low, high};
}
{
mul.f16x2 r507, r43, r506;
}
{
add.f16x2 r510, r482, r507;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r513, {low, high};
}
{
mul.f16x2 r514, r49, r513;
}
{
add.f16x2 r517, r489, r514;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r520, {low, high};
}
{
mul.f16x2 r521, r58, r520;
}
{
add.f16x2 r524, r496, r521;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r527, {low, high};
}
{
mul.f16x2 r528, r52, r527;
}
{
add.f16x2 r531, r503, r528;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r534, {low, high};
}
{
mul.f16x2 r535, r55, r534;
}
{
add.f16x2 r538, r510, r535;
}
{
sub.f16x2 r541, r517, r524;
}
{
add.f16x2 r544, r531, r538;
}
{
add.f16x2 r547, r517, r524;
}
{
sub.f16x2 r550, r531, r538;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r553, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r554, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r555, {low, high};
}
{
mul.f16x2 r556, r1, r555;
}
{
add.f16x2 r559, %23, r556;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r562, {low, high};
}
{
mul.f16x2 r563, r10, r562;
}
{
add.f16x2 r566, r553, r563;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r569, {low, high};
}
{
mul.f16x2 r570, r4, r569;
}
{
add.f16x2 r573, %24, r570;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r576, {low, high};
}
{
mul.f16x2 r577, r7, r576;
}
{
add.f16x2 r580, r554, r577;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r583, {low, high};
}
{
mul.f16x2 r584, r13, r583;
}
{
add.f16x2 r587, r559, r584;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f396;
cvt.rn.f16.f32 high, f396;
mov.b32 r590, {low, high};
}
{
mul.f16x2 r591, r22, r590;
}
{
add.f16x2 r594, r566, r591;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r597, {low, high};
}
{
mul.f16x2 r598, r16, r597;
}
{
add.f16x2 r601, r573, r598;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f396;
cvt.rn.f16.f32 high, f396;
mov.b32 r604, {low, high};
}
{
mul.f16x2 r605, r19, r604;
}
{
add.f16x2 r608, r580, r605;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r611, {low, high};
}
{
mul.f16x2 r612, r25, r611;
}
{
add.f16x2 r615, r587, r612;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r618, {low, high};
}
{
mul.f16x2 r619, r34, r618;
}
{
add.f16x2 r622, r594, r619;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r625, {low, high};
}
{
mul.f16x2 r626, r28, r625;
}
{
add.f16x2 r629, r601, r626;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r632, {low, high};
}
{
mul.f16x2 r633, r31, r632;
}
{
add.f16x2 r636, r608, r633;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r639, {low, high};
}
{
mul.f16x2 r640, r37, r639;
}
{
add.f16x2 r643, r615, r640;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r646, {low, high};
}
{
mul.f16x2 r647, r46, r646;
}
{
add.f16x2 r650, r622, r647;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r653, {low, high};
}
{
mul.f16x2 r654, r40, r653;
}
{
add.f16x2 r657, r629, r654;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r660, {low, high};
}
{
mul.f16x2 r661, r43, r660;
}
{
add.f16x2 r664, r636, r661;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r667, {low, high};
}
{
mul.f16x2 r668, r49, r667;
}
{
add.f16x2 r671, r643, r668;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r674, {low, high};
}
{
mul.f16x2 r675, r58, r674;
}
{
add.f16x2 r678, r650, r675;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r681, {low, high};
}
{
mul.f16x2 r682, r52, r681;
}
{
add.f16x2 r685, r657, r682;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r688, {low, high};
}
{
mul.f16x2 r689, r55, r688;
}
{
add.f16x2 r692, r664, r689;
}
{
sub.f16x2 r695, r671, r678;
}
{
add.f16x2 r698, r685, r692;
}
{
add.f16x2 r701, r671, r678;
}
{
sub.f16x2 r704, r685, r692;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r707, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r708, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r709, {low, high};
}
{
mul.f16x2 r710, r1, r709;
}
{
add.f16x2 r713, %23, r710;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r716, {low, high};
}
{
mul.f16x2 r717, r10, r716;
}
{
add.f16x2 r720, r707, r717;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r723, {low, high};
}
{
mul.f16x2 r724, r4, r723;
}
{
add.f16x2 r727, %24, r724;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r730, {low, high};
}
{
mul.f16x2 r731, r7, r730;
}
{
add.f16x2 r734, r708, r731;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r737, {low, high};
}
{
mul.f16x2 r738, r13, r737;
}
{
add.f16x2 r741, r713, r738;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f440;
cvt.rn.f16.f32 high, f440;
mov.b32 r744, {low, high};
}
{
mul.f16x2 r745, r22, r744;
}
{
add.f16x2 r748, r720, r745;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r751, {low, high};
}
{
mul.f16x2 r752, r16, r751;
}
{
add.f16x2 r755, r727, r752;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f440;
cvt.rn.f16.f32 high, f440;
mov.b32 r758, {low, high};
}
{
mul.f16x2 r759, r19, r758;
}
{
add.f16x2 r762, r734, r759;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r765, {low, high};
}
{
mul.f16x2 r766, r25, r765;
}
{
add.f16x2 r769, r741, r766;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r772, {low, high};
}
{
mul.f16x2 r773, r34, r772;
}
{
add.f16x2 r776, r748, r773;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r779, {low, high};
}
{
mul.f16x2 r780, r28, r779;
}
{
add.f16x2 r783, r755, r780;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r786, {low, high};
}
{
mul.f16x2 r787, r31, r786;
}
{
add.f16x2 r790, r762, r787;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r793, {low, high};
}
{
mul.f16x2 r794, r37, r793;
}
{
add.f16x2 r797, r769, r794;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r800, {low, high};
}
{
mul.f16x2 r801, r46, r800;
}
{
add.f16x2 r804, r776, r801;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r807, {low, high};
}
{
mul.f16x2 r808, r40, r807;
}
{
add.f16x2 r811, r783, r808;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r814, {low, high};
}
{
mul.f16x2 r815, r43, r814;
}
{
add.f16x2 r818, r790, r815;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r821, {low, high};
}
{
mul.f16x2 r822, r49, r821;
}
{
add.f16x2 r825, r797, r822;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r828, {low, high};
}
{
mul.f16x2 r829, r58, r828;
}
{
add.f16x2 r832, r804, r829;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r835, {low, high};
}
{
mul.f16x2 r836, r52, r835;
}
{
add.f16x2 r839, r811, r836;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r842, {low, high};
}
{
mul.f16x2 r843, r55, r842;
}
{
add.f16x2 r846, r818, r843;
}
{
sub.f16x2 r849, r825, r832;
}
{
add.f16x2 r852, r839, r846;
}
{
add.f16x2 r855, r825, r832;
}
{
sub.f16x2 r858, r839, r846;
}
mul.wide.u32 rd2, r2097, -1171354717;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r2098, rd3;
mul.lo.s32 r2099, r2098, 11;
sub.s32 r2100, r2097, r2099;
cvt.rn.f32.u32 f465, r2100;
mul.f32 f466, f465, 0f3D54B191;
cos.approx.f32 f221, f466;
sin.approx.f32 f467, f466;
neg.f32 f222, f467;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f222;
mov.b32 r861, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r864, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r866, {high, high};
}
{
mul.f16x2 r868, r236, r866;
}
{
fma.rn.f16x2 r871, r233, r864, r868;
}
{
mul.f16x2 r875, r233, r866;
}
{
neg.f16x2 r878, r875;
}
{
fma.rn.f16x2 r880, r236, r864, r878;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r884, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r886, {high, high};
}
mov.f32 f241, 0fBF800000;
mov.f32 f242, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f241;
cvt.rn.f16.f32 high, f242;
mov.b32 r888, {low, high};
}
{
mul.f16x2 r889, r886, r888;
}
{
mul.f16x2 r892, r861, r884;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r895, {high, low};
}
{
fma.rn.f16x2 r897, r889, r895, r892;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r897;
mov.b32 r901, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r897;
mov.b32 r903, {high, high};
}
{
mul.f16x2 r905, r390, r903;
}
{
fma.rn.f16x2 r908, r387, r901, r905;
}
{
mul.f16x2 r912, r387, r903;
}
{
neg.f16x2 r915, r912;
}
{
fma.rn.f16x2 r917, r390, r901, r915;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r921, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r923, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f241;
cvt.rn.f16.f32 high, f242;
mov.b32 r925, {low, high};
}
{
mul.f16x2 r926, r923, r925;
}
{
mul.f16x2 r929, r897, r921;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r897;
mov.b32 r932, {high, low};
}
{
fma.rn.f16x2 r934, r926, r932, r929;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r934;
mov.b32 r938, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r934;
mov.b32 r940, {high, high};
}
{
mul.f16x2 r942, r544, r940;
}
{
fma.rn.f16x2 r945, r541, r938, r942;
}
{
mul.f16x2 r949, r541, r940;
}
{
neg.f16x2 r952, r949;
}
{
fma.rn.f16x2 r954, r544, r938, r952;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r958, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r960, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f241;
cvt.rn.f16.f32 high, f242;
mov.b32 r962, {low, high};
}
{
mul.f16x2 r963, r960, r962;
}
{
mul.f16x2 r966, r934, r958;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r934;
mov.b32 r969, {high, low};
}
{
fma.rn.f16x2 r971, r963, r969, r966;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r971;
mov.b32 r975, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r971;
mov.b32 r977, {high, high};
}
{
mul.f16x2 r979, r698, r977;
}
{
fma.rn.f16x2 r982, r695, r975, r979;
}
{
mul.f16x2 r986, r695, r977;
}
{
neg.f16x2 r989, r986;
}
{
fma.rn.f16x2 r991, r698, r975, r989;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r995, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r997, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f241;
cvt.rn.f16.f32 high, f242;
mov.b32 r999, {low, high};
}
{
mul.f16x2 r1000, r997, r999;
}
{
mul.f16x2 r1003, r971, r995;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r971;
mov.b32 r1006, {high, low};
}
{
fma.rn.f16x2 r1008, r1000, r1006, r1003;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1008;
mov.b32 r1012, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1008;
mov.b32 r1014, {high, high};
}
{
mul.f16x2 r1016, r852, r1014;
}
{
fma.rn.f16x2 r1019, r849, r1012, r1016;
}
{
mul.f16x2 r1023, r849, r1014;
}
{
neg.f16x2 r1026, r1023;
}
{
fma.rn.f16x2 r1028, r852, r1012, r1026;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r1032, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r1034, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f241;
cvt.rn.f16.f32 high, f242;
mov.b32 r1036, {low, high};
}
{
mul.f16x2 r1037, r1034, r1036;
}
{
mul.f16x2 r1040, r1008, r1032;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1008;
mov.b32 r1043, {high, low};
}
{
fma.rn.f16x2 r1045, r1037, r1043, r1040;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1045;
mov.b32 r1049, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1045;
mov.b32 r1051, {high, high};
}
{
mul.f16x2 r1053, r858, r1051;
}
{
fma.rn.f16x2 r1056, r855, r1049, r1053;
}
{
mul.f16x2 r1060, r855, r1051;
}
{
neg.f16x2 r1063, r1060;
}
{
fma.rn.f16x2 r1065, r858, r1049, r1063;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r1069, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r1071, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f241;
cvt.rn.f16.f32 high, f242;
mov.b32 r1073, {low, high};
}
{
mul.f16x2 r1074, r1071, r1073;
}
{
mul.f16x2 r1077, r1045, r1069;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1045;
mov.b32 r1080, {high, low};
}
{
fma.rn.f16x2 r1082, r1074, r1080, r1077;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1082;
mov.b32 r1086, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1082;
mov.b32 r1088, {high, high};
}
{
mul.f16x2 r1090, r704, r1088;
}
{
fma.rn.f16x2 r1093, r701, r1086, r1090;
}
{
mul.f16x2 r1097, r701, r1088;
}
{
neg.f16x2 r1100, r1097;
}
{
fma.rn.f16x2 r1102, r704, r1086, r1100;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r1106, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r1108, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f241;
cvt.rn.f16.f32 high, f242;
mov.b32 r1110, {low, high};
}
{
mul.f16x2 r1111, r1108, r1110;
}
{
mul.f16x2 r1114, r1082, r1106;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1082;
mov.b32 r1117, {high, low};
}
{
fma.rn.f16x2 r1119, r1111, r1117, r1114;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1119;
mov.b32 r1123, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1119;
mov.b32 r1125, {high, high};
}
{
mul.f16x2 r1127, r550, r1125;
}
{
fma.rn.f16x2 r1130, r547, r1123, r1127;
}
{
mul.f16x2 r1134, r547, r1125;
}
{
neg.f16x2 r1137, r1134;
}
{
fma.rn.f16x2 r1139, r550, r1123, r1137;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r1143, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r1145, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f241;
cvt.rn.f16.f32 high, f242;
mov.b32 r1147, {low, high};
}
{
mul.f16x2 r1148, r1145, r1147;
}
{
mul.f16x2 r1151, r1119, r1143;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1119;
mov.b32 r1154, {high, low};
}
{
fma.rn.f16x2 r1156, r1148, r1154, r1151;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1156;
mov.b32 r1160, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1156;
mov.b32 r1162, {high, high};
}
{
mul.f16x2 r1164, r396, r1162;
}
{
fma.rn.f16x2 r1167, r393, r1160, r1164;
}
{
mul.f16x2 r1171, r393, r1162;
}
{
neg.f16x2 r1174, r1171;
}
{
fma.rn.f16x2 r1176, r396, r1160, r1174;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r1180, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r1182, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f241;
cvt.rn.f16.f32 high, f242;
mov.b32 r1184, {low, high};
}
{
mul.f16x2 r1185, r1182, r1184;
}
{
mul.f16x2 r1188, r1156, r1180;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1156;
mov.b32 r1191, {high, low};
}
{
fma.rn.f16x2 r1193, r1185, r1191, r1188;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1193;
mov.b32 r1197, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1193;
mov.b32 r1199, {high, high};
}
{
mul.f16x2 r1201, r242, r1199;
}
{
fma.rn.f16x2 r1204, r239, r1197, r1201;
}
{
mul.f16x2 r1208, r239, r1199;
}
{
neg.f16x2 r1211, r1208;
}
{
fma.rn.f16x2 r1213, r242, r1197, r1211;
}
mad.lo.s32 r2101, r2098, 968, r2096;
barrier.sync 0;
mad.lo.s32 r2102, r2100, 88, r2101;
st.shared.v2.f32 [r2102], {r85, r88};
st.shared.v2.f32 [r2102+8], {r871, r880};
st.shared.v2.f32 [r2102+16], {r908, r917};
st.shared.v2.f32 [r2102+24], {r945, r954};
st.shared.v2.f32 [r2102+32], {r982, r991};
st.shared.v2.f32 [r2102+40], {r1019, r1028};
st.shared.v2.f32 [r2102+48], {r1056, r1065};
st.shared.v2.f32 [r2102+56], {r1093, r1102};
st.shared.v2.f32 [r2102+64], {r1130, r1139};
st.shared.v2.f32 [r2102+72], {r1167, r1176};
st.shared.v2.f32 [r2102+80], {r1204, r1213};
barrier.sync 0;
mad.lo.s32 r2103, r2100, -80, r2102;
ld.shared.u32 r1295, [r2103];
ld.shared.u32 r1298, [r2103+4];
ld.shared.u32 r1235, [r2103+88];
ld.shared.u32 r1238, [r2103+92];
ld.shared.u32 r1247, [r2103+176];
ld.shared.u32 r1250, [r2103+180];
ld.shared.u32 r1259, [r2103+264];
ld.shared.u32 r1262, [r2103+268];
ld.shared.u32 r1271, [r2103+352];
ld.shared.u32 r1274, [r2103+356];
ld.shared.u32 r1283, [r2103+440];
ld.shared.u32 r1286, [r2103+444];
ld.shared.u32 r1284, [r2103+528];
ld.shared.u32 r1287, [r2103+532];
ld.shared.u32 r1272, [r2103+616];
ld.shared.u32 r1275, [r2103+620];
ld.shared.u32 r1260, [r2103+704];
ld.shared.u32 r1263, [r2103+708];
ld.shared.u32 r1248, [r2103+792];
ld.shared.u32 r1251, [r2103+796];
ld.shared.u32 r1236, [r2103+880];
ld.shared.u32 r1239, [r2103+884];
{
add.f16x2 r1234, r1235, r1236;
}
{
add.f16x2 r1237, r1238, r1239;
}
{
sub.f16x2 r1240, r1235, r1236;
}
{
sub.f16x2 r1243, r1238, r1239;
}
{
add.f16x2 r1246, r1247, r1248;
}
{
add.f16x2 r1249, r1250, r1251;
}
{
sub.f16x2 r1252, r1247, r1248;
}
{
sub.f16x2 r1255, r1250, r1251;
}
{
add.f16x2 r1258, r1259, r1260;
}
{
add.f16x2 r1261, r1262, r1263;
}
{
sub.f16x2 r1264, r1259, r1260;
}
{
sub.f16x2 r1267, r1262, r1263;
}
{
add.f16x2 r1270, r1271, r1272;
}
{
add.f16x2 r1273, r1274, r1275;
}
{
sub.f16x2 r1276, r1271, r1272;
}
{
sub.f16x2 r1279, r1274, r1275;
}
{
add.f16x2 r1282, r1283, r1284;
}
{
add.f16x2 r1285, r1286, r1287;
}
{
sub.f16x2 r1288, r1283, r1284;
}
{
sub.f16x2 r1291, r1286, r1287;
}
{
add.f16x2 r1294, r1295, r1234;
}
{
add.f16x2 r1297, r1298, r1237;
}
{
add.f16x2 r1300, r1294, r1246;
}
{
add.f16x2 r1303, r1297, r1249;
}
{
add.f16x2 r1306, r1300, r1258;
}
{
add.f16x2 r1309, r1303, r1261;
}
{
add.f16x2 r1312, r1306, r1270;
}
{
add.f16x2 r1315, r1309, r1273;
}
{
add.f16x2 %0, r1312, r1282;
}
{
add.f16x2 %1, r1315, r1285;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r1324, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r1325, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r1326, {low, high};
}
{
mul.f16x2 r1327, r1234, r1326;
}
{
add.f16x2 r1330, r1295, r1327;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r1333, {low, high};
}
{
mul.f16x2 r1334, r1243, r1333;
}
{
add.f16x2 r1337, r1324, r1334;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r1340, {low, high};
}
{
mul.f16x2 r1341, r1237, r1340;
}
{
add.f16x2 r1344, r1298, r1341;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r1347, {low, high};
}
{
mul.f16x2 r1348, r1240, r1347;
}
{
add.f16x2 r1351, r1325, r1348;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r1354, {low, high};
}
{
mul.f16x2 r1355, r1246, r1354;
}
{
add.f16x2 r1358, r1330, r1355;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f300;
cvt.rn.f16.f32 high, f300;
mov.b32 r1361, {low, high};
}
{
mul.f16x2 r1362, r1255, r1361;
}
{
add.f16x2 r1365, r1337, r1362;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r1368, {low, high};
}
{
mul.f16x2 r1369, r1249, r1368;
}
{
add.f16x2 r1372, r1344, r1369;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f300;
cvt.rn.f16.f32 high, f300;
mov.b32 r1375, {low, high};
}
{
mul.f16x2 r1376, r1252, r1375;
}
{
add.f16x2 r1379, r1351, r1376;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r1382, {low, high};
}
{
mul.f16x2 r1383, r1258, r1382;
}
{
add.f16x2 r1386, r1358, r1383;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r1389, {low, high};
}
{
mul.f16x2 r1390, r1267, r1389;
}
{
add.f16x2 r1393, r1365, r1390;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r1396, {low, high};
}
{
mul.f16x2 r1397, r1261, r1396;
}
{
add.f16x2 r1400, r1372, r1397;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r1403, {low, high};
}
{
mul.f16x2 r1404, r1264, r1403;
}
{
add.f16x2 r1407, r1379, r1404;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r1410, {low, high};
}
{
mul.f16x2 r1411, r1270, r1410;
}
{
add.f16x2 r1414, r1386, r1411;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r1417, {low, high};
}
{
mul.f16x2 r1418, r1279, r1417;
}
{
add.f16x2 r1421, r1393, r1418;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r1424, {low, high};
}
{
mul.f16x2 r1425, r1273, r1424;
}
{
add.f16x2 r1428, r1400, r1425;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r1431, {low, high};
}
{
mul.f16x2 r1432, r1276, r1431;
}
{
add.f16x2 r1435, r1407, r1432;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r1438, {low, high};
}
{
mul.f16x2 r1439, r1282, r1438;
}
{
add.f16x2 r1442, r1414, r1439;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r1445, {low, high};
}
{
mul.f16x2 r1446, r1291, r1445;
}
{
add.f16x2 r1449, r1421, r1446;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r1452, {low, high};
}
{
mul.f16x2 r1453, r1285, r1452;
}
{
add.f16x2 r1456, r1428, r1453;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r1459, {low, high};
}
{
mul.f16x2 r1460, r1288, r1459;
}
{
add.f16x2 r1463, r1435, r1460;
}
{
sub.f16x2 %2, r1442, r1449;
}
{
add.f16x2 %3, r1456, r1463;
}
{
add.f16x2 %20, r1442, r1449;
}
{
sub.f16x2 %21, r1456, r1463;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r1478, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r1479, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r1480, {low, high};
}
{
mul.f16x2 r1481, r1234, r1480;
}
{
add.f16x2 r1484, r1295, r1481;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f300;
cvt.rn.f16.f32 high, f300;
mov.b32 r1487, {low, high};
}
{
mul.f16x2 r1488, r1243, r1487;
}
{
add.f16x2 r1491, r1478, r1488;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r1494, {low, high};
}
{
mul.f16x2 r1495, r1237, r1494;
}
{
add.f16x2 r1498, r1298, r1495;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f300;
cvt.rn.f16.f32 high, f300;
mov.b32 r1501, {low, high};
}
{
mul.f16x2 r1502, r1240, r1501;
}
{
add.f16x2 r1505, r1479, r1502;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r1508, {low, high};
}
{
mul.f16x2 r1509, r1246, r1508;
}
{
add.f16x2 r1512, r1484, r1509;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r1515, {low, high};
}
{
mul.f16x2 r1516, r1255, r1515;
}
{
add.f16x2 r1519, r1491, r1516;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r1522, {low, high};
}
{
mul.f16x2 r1523, r1249, r1522;
}
{
add.f16x2 r1526, r1498, r1523;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r1529, {low, high};
}
{
mul.f16x2 r1530, r1252, r1529;
}
{
add.f16x2 r1533, r1505, r1530;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r1536, {low, high};
}
{
mul.f16x2 r1537, r1258, r1536;
}
{
add.f16x2 r1540, r1512, r1537;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r1543, {low, high};
}
{
mul.f16x2 r1544, r1267, r1543;
}
{
add.f16x2 r1547, r1519, r1544;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r1550, {low, high};
}
{
mul.f16x2 r1551, r1261, r1550;
}
{
add.f16x2 r1554, r1526, r1551;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r1557, {low, high};
}
{
mul.f16x2 r1558, r1264, r1557;
}
{
add.f16x2 r1561, r1533, r1558;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r1564, {low, high};
}
{
mul.f16x2 r1565, r1270, r1564;
}
{
add.f16x2 r1568, r1540, r1565;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f396;
cvt.rn.f16.f32 high, f396;
mov.b32 r1571, {low, high};
}
{
mul.f16x2 r1572, r1279, r1571;
}
{
add.f16x2 r1575, r1547, r1572;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r1578, {low, high};
}
{
mul.f16x2 r1579, r1273, r1578;
}
{
add.f16x2 r1582, r1554, r1579;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f396;
cvt.rn.f16.f32 high, f396;
mov.b32 r1585, {low, high};
}
{
mul.f16x2 r1586, r1276, r1585;
}
{
add.f16x2 r1589, r1561, r1586;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r1592, {low, high};
}
{
mul.f16x2 r1593, r1282, r1592;
}
{
add.f16x2 r1596, r1568, r1593;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f440;
cvt.rn.f16.f32 high, f440;
mov.b32 r1599, {low, high};
}
{
mul.f16x2 r1600, r1291, r1599;
}
{
add.f16x2 r1603, r1575, r1600;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r1606, {low, high};
}
{
mul.f16x2 r1607, r1285, r1606;
}
{
add.f16x2 r1610, r1582, r1607;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f440;
cvt.rn.f16.f32 high, f440;
mov.b32 r1613, {low, high};
}
{
mul.f16x2 r1614, r1288, r1613;
}
{
add.f16x2 r1617, r1589, r1614;
}
{
sub.f16x2 %4, r1596, r1603;
}
{
add.f16x2 %5, r1610, r1617;
}
{
add.f16x2 %18, r1596, r1603;
}
{
sub.f16x2 %19, r1610, r1617;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r1632, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r1633, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r1634, {low, high};
}
{
mul.f16x2 r1635, r1234, r1634;
}
{
add.f16x2 r1638, r1295, r1635;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r1641, {low, high};
}
{
mul.f16x2 r1642, r1243, r1641;
}
{
add.f16x2 r1645, r1632, r1642;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r1648, {low, high};
}
{
mul.f16x2 r1649, r1237, r1648;
}
{
add.f16x2 r1652, r1298, r1649;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r1655, {low, high};
}
{
mul.f16x2 r1656, r1240, r1655;
}
{
add.f16x2 r1659, r1633, r1656;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r1662, {low, high};
}
{
mul.f16x2 r1663, r1246, r1662;
}
{
add.f16x2 r1666, r1638, r1663;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r1669, {low, high};
}
{
mul.f16x2 r1670, r1255, r1669;
}
{
add.f16x2 r1673, r1645, r1670;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r1676, {low, high};
}
{
mul.f16x2 r1677, r1249, r1676;
}
{
add.f16x2 r1680, r1652, r1677;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r1683, {low, high};
}
{
mul.f16x2 r1684, r1252, r1683;
}
{
add.f16x2 r1687, r1659, r1684;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r1690, {low, high};
}
{
mul.f16x2 r1691, r1258, r1690;
}
{
add.f16x2 r1694, r1666, r1691;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r1697, {low, high};
}
{
mul.f16x2 r1698, r1267, r1697;
}
{
add.f16x2 r1701, r1673, r1698;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r1704, {low, high};
}
{
mul.f16x2 r1705, r1261, r1704;
}
{
add.f16x2 r1708, r1680, r1705;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r1711, {low, high};
}
{
mul.f16x2 r1712, r1264, r1711;
}
{
add.f16x2 r1715, r1687, r1712;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r1718, {low, high};
}
{
mul.f16x2 r1719, r1270, r1718;
}
{
add.f16x2 r1722, r1694, r1719;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r1725, {low, high};
}
{
mul.f16x2 r1726, r1279, r1725;
}
{
add.f16x2 r1729, r1701, r1726;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r1732, {low, high};
}
{
mul.f16x2 r1733, r1273, r1732;
}
{
add.f16x2 r1736, r1708, r1733;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r1739, {low, high};
}
{
mul.f16x2 r1740, r1276, r1739;
}
{
add.f16x2 r1743, r1715, r1740;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r1746, {low, high};
}
{
mul.f16x2 r1747, r1282, r1746;
}
{
add.f16x2 r1750, r1722, r1747;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r1753, {low, high};
}
{
mul.f16x2 r1754, r1291, r1753;
}
{
add.f16x2 r1757, r1729, r1754;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r1760, {low, high};
}
{
mul.f16x2 r1761, r1285, r1760;
}
{
add.f16x2 r1764, r1736, r1761;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r1767, {low, high};
}
{
mul.f16x2 r1768, r1288, r1767;
}
{
add.f16x2 r1771, r1743, r1768;
}
{
sub.f16x2 %6, r1750, r1757;
}
{
add.f16x2 %7, r1764, r1771;
}
{
add.f16x2 %16, r1750, r1757;
}
{
sub.f16x2 %17, r1764, r1771;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r1786, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r1787, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r1788, {low, high};
}
{
mul.f16x2 r1789, r1234, r1788;
}
{
add.f16x2 r1792, r1295, r1789;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r1795, {low, high};
}
{
mul.f16x2 r1796, r1243, r1795;
}
{
add.f16x2 r1799, r1786, r1796;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r1802, {low, high};
}
{
mul.f16x2 r1803, r1237, r1802;
}
{
add.f16x2 r1806, r1298, r1803;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r1809, {low, high};
}
{
mul.f16x2 r1810, r1240, r1809;
}
{
add.f16x2 r1813, r1787, r1810;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r1816, {low, high};
}
{
mul.f16x2 r1817, r1246, r1816;
}
{
add.f16x2 r1820, r1792, r1817;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f396;
cvt.rn.f16.f32 high, f396;
mov.b32 r1823, {low, high};
}
{
mul.f16x2 r1824, r1255, r1823;
}
{
add.f16x2 r1827, r1799, r1824;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r1830, {low, high};
}
{
mul.f16x2 r1831, r1249, r1830;
}
{
add.f16x2 r1834, r1806, r1831;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f396;
cvt.rn.f16.f32 high, f396;
mov.b32 r1837, {low, high};
}
{
mul.f16x2 r1838, r1252, r1837;
}
{
add.f16x2 r1841, r1813, r1838;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r1844, {low, high};
}
{
mul.f16x2 r1845, r1258, r1844;
}
{
add.f16x2 r1848, r1820, r1845;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r1851, {low, high};
}
{
mul.f16x2 r1852, r1267, r1851;
}
{
add.f16x2 r1855, r1827, r1852;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r1858, {low, high};
}
{
mul.f16x2 r1859, r1261, r1858;
}
{
add.f16x2 r1862, r1834, r1859;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r1865, {low, high};
}
{
mul.f16x2 r1866, r1264, r1865;
}
{
add.f16x2 r1869, r1841, r1866;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r1872, {low, high};
}
{
mul.f16x2 r1873, r1270, r1872;
}
{
add.f16x2 r1876, r1848, r1873;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r1879, {low, high};
}
{
mul.f16x2 r1880, r1279, r1879;
}
{
add.f16x2 r1883, r1855, r1880;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r1886, {low, high};
}
{
mul.f16x2 r1887, r1273, r1886;
}
{
add.f16x2 r1890, r1862, r1887;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r1893, {low, high};
}
{
mul.f16x2 r1894, r1276, r1893;
}
{
add.f16x2 r1897, r1869, r1894;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r1900, {low, high};
}
{
mul.f16x2 r1901, r1282, r1900;
}
{
add.f16x2 r1904, r1876, r1901;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r1907, {low, high};
}
{
mul.f16x2 r1908, r1291, r1907;
}
{
add.f16x2 r1911, r1883, r1908;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r1914, {low, high};
}
{
mul.f16x2 r1915, r1285, r1914;
}
{
add.f16x2 r1918, r1890, r1915;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r1921, {low, high};
}
{
mul.f16x2 r1922, r1288, r1921;
}
{
add.f16x2 r1925, r1897, r1922;
}
{
sub.f16x2 %8, r1904, r1911;
}
{
add.f16x2 %9, r1918, r1925;
}
{
add.f16x2 %14, r1904, r1911;
}
{
sub.f16x2 %15, r1918, r1925;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r1940, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r1941, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r1942, {low, high};
}
{
mul.f16x2 r1943, r1234, r1942;
}
{
add.f16x2 r1946, r1295, r1943;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r1949, {low, high};
}
{
mul.f16x2 r1950, r1243, r1949;
}
{
add.f16x2 r1953, r1940, r1950;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r1956, {low, high};
}
{
mul.f16x2 r1957, r1237, r1956;
}
{
add.f16x2 r1960, r1298, r1957;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r1963, {low, high};
}
{
mul.f16x2 r1964, r1240, r1963;
}
{
add.f16x2 r1967, r1941, r1964;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r1970, {low, high};
}
{
mul.f16x2 r1971, r1246, r1970;
}
{
add.f16x2 r1974, r1946, r1971;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f440;
cvt.rn.f16.f32 high, f440;
mov.b32 r1977, {low, high};
}
{
mul.f16x2 r1978, r1255, r1977;
}
{
add.f16x2 r1981, r1953, r1978;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r1984, {low, high};
}
{
mul.f16x2 r1985, r1249, r1984;
}
{
add.f16x2 r1988, r1960, r1985;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f440;
cvt.rn.f16.f32 high, f440;
mov.b32 r1991, {low, high};
}
{
mul.f16x2 r1992, r1252, r1991;
}
{
add.f16x2 r1995, r1967, r1992;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r1998, {low, high};
}
{
mul.f16x2 r1999, r1258, r1998;
}
{
add.f16x2 r2002, r1974, r1999;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r2005, {low, high};
}
{
mul.f16x2 r2006, r1267, r2005;
}
{
add.f16x2 r2009, r1981, r2006;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r2012, {low, high};
}
{
mul.f16x2 r2013, r1261, r2012;
}
{
add.f16x2 r2016, r1988, r2013;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r2019, {low, high};
}
{
mul.f16x2 r2020, r1264, r2019;
}
{
add.f16x2 r2023, r1995, r2020;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r2026, {low, high};
}
{
mul.f16x2 r2027, r1270, r2026;
}
{
add.f16x2 r2030, r2002, r2027;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r2033, {low, high};
}
{
mul.f16x2 r2034, r1279, r2033;
}
{
add.f16x2 r2037, r2009, r2034;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r2040, {low, high};
}
{
mul.f16x2 r2041, r1273, r2040;
}
{
add.f16x2 r2044, r2016, r2041;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r2047, {low, high};
}
{
mul.f16x2 r2048, r1276, r2047;
}
{
add.f16x2 r2051, r2023, r2048;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r2054, {low, high};
}
{
mul.f16x2 r2055, r1282, r2054;
}
{
add.f16x2 r2058, r2030, r2055;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r2061, {low, high};
}
{
mul.f16x2 r2062, r1291, r2061;
}
{
add.f16x2 r2065, r2037, r2062;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r2068, {low, high};
}
{
mul.f16x2 r2069, r1285, r2068;
}
{
add.f16x2 r2072, r2044, r2069;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r2075, {low, high};
}
{
mul.f16x2 r2076, r1288, r2075;
}
{
add.f16x2 r2079, r2051, r2076;
}
{
sub.f16x2 %10, r2058, r2065;
}
{
add.f16x2 %11, r2072, r2079;
}
{
add.f16x2 %12, r2058, r2065;
}
{
sub.f16x2 %13, r2072, r2079;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1128, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<468>;
.reg .b32 r<2104>;
.reg .b64 rd<4>;
mov.u32 r2094, %tid.y;
mov.u32 r2095, %22;
mad.lo.s32 r2096, r2094, 484, r2095;
mov.u32 r2097, %tid.x;
{
add.f16x2 r1, %25, %43;
}
{
add.f16x2 r4, %26, %44;
}
{
sub.f16x2 r7, %25, %43;
}
{
sub.f16x2 r10, %26, %44;
}
{
add.f16x2 r13, %27, %41;
}
{
add.f16x2 r16, %28, %42;
}
{
sub.f16x2 r19, %27, %41;
}
{
sub.f16x2 r22, %28, %42;
}
{
add.f16x2 r25, %29, %39;
}
{
add.f16x2 r28, %30, %40;
}
{
sub.f16x2 r31, %29, %39;
}
{
sub.f16x2 r34, %30, %40;
}
{
add.f16x2 r37, %31, %37;
}
{
add.f16x2 r40, %32, %38;
}
{
sub.f16x2 r43, %31, %37;
}
{
sub.f16x2 r46, %32, %38;
}
{
add.f16x2 r49, %33, %35;
}
{
add.f16x2 r52, %34, %36;
}
{
sub.f16x2 r55, %33, %35;
}
{
sub.f16x2 r58, %34, %36;
}
{
add.f16x2 r61, %23, r1;
}
{
add.f16x2 r64, %24, r4;
}
{
add.f16x2 r67, r61, r13;
}
{
add.f16x2 r70, r64, r16;
}
{
add.f16x2 r73, r67, r25;
}
{
add.f16x2 r76, r70, r28;
}
{
add.f16x2 r79, r73, r37;
}
{
add.f16x2 r82, r76, r40;
}
{
add.f16x2 r85, r79, r49;
}
{
add.f16x2 r88, r82, r52;
}
mov.f32 f424, 0f00000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r91, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r92, {low, high};
}
mov.f32 f438, 0f3F575C64;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r93, {low, high};
}
{
mul.f16x2 r94, r1, r93;
}
{
add.f16x2 r97, %23, r94;
}
mov.f32 f404, 0f3F0A6770;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r100, {low, high};
}
{
mul.f16x2 r101, r10, r100;
}
{
add.f16x2 r104, r91, r101;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r107, {low, high};
}
{
mul.f16x2 r108, r4, r107;
}
{
add.f16x2 r111, %24, r108;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r114, {low, high};
}
{
mul.f16x2 r115, r7, r114;
}
{
add.f16x2 r118, r92, r115;
}
mov.f32 f454, 0f3ED4B147;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r121, {low, high};
}
{
mul.f16x2 r122, r13, r121;
}
{
add.f16x2 r125, r97, r122;
}
mov.f32 f300, 0f3F68DDA4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f300;
cvt.rn.f16.f32 high, f300;
mov.b32 r128, {low, high};
}
{
mul.f16x2 r129, r22, r128;
}
{
add.f16x2 r132, r104, r129;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r135, {low, high};
}
{
mul.f16x2 r136, r16, r135;
}
{
add.f16x2 r139, r111, r136;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f300;
cvt.rn.f16.f32 high, f300;
mov.b32 r142, {low, high};
}
{
mul.f16x2 r143, r19, r142;
}
{
add.f16x2 r146, r118, r143;
}
mov.f32 f462, 0fBE11BAFB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r149, {low, high};
}
{
mul.f16x2 r150, r25, r149;
}
{
add.f16x2 r153, r125, r150;
}
mov.f32 f464, 0f3F7D64F0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r156, {low, high};
}
{
mul.f16x2 r157, r34, r156;
}
{
add.f16x2 r160, r132, r157;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r163, {low, high};
}
{
mul.f16x2 r164, r28, r163;
}
{
add.f16x2 r167, r139, r164;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r170, {low, high};
}
{
mul.f16x2 r171, r31, r170;
}
{
add.f16x2 r174, r146, r171;
}
mov.f32 f446, 0fBF27A4F4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r177, {low, high};
}
{
mul.f16x2 r178, r37, r177;
}
{
add.f16x2 r181, r153, r178;
}
mov.f32 f448, 0f3F4178CE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r184, {low, high};
}
{
mul.f16x2 r185, r46, r184;
}
{
add.f16x2 r188, r160, r185;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r191, {low, high};
}
{
mul.f16x2 r192, r40, r191;
}
{
add.f16x2 r195, r167, r192;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r198, {low, high};
}
{
mul.f16x2 r199, r43, r198;
}
{
add.f16x2 r202, r174, r199;
}
mov.f32 f430, 0fBF75A155;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r205, {low, high};
}
{
mul.f16x2 r206, r49, r205;
}
{
add.f16x2 r209, r181, r206;
}
mov.f32 f432, 0f3E903F40;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r212, {low, high};
}
{
mul.f16x2 r213, r58, r212;
}
{
add.f16x2 r216, r188, r213;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r219, {low, high};
}
{
mul.f16x2 r220, r52, r219;
}
{
add.f16x2 r223, r195, r220;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r226, {low, high};
}
{
mul.f16x2 r227, r55, r226;
}
{
add.f16x2 r230, r202, r227;
}
{
sub.f16x2 r233, r209, r216;
}
{
add.f16x2 r236, r223, r230;
}
{
add.f16x2 r239, r209, r216;
}
{
sub.f16x2 r242, r223, r230;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r245, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r246, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r247, {low, high};
}
{
mul.f16x2 r248, r1, r247;
}
{
add.f16x2 r251, %23, r248;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f300;
cvt.rn.f16.f32 high, f300;
mov.b32 r254, {low, high};
}
{
mul.f16x2 r255, r10, r254;
}
{
add.f16x2 r258, r245, r255;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r261, {low, high};
}
{
mul.f16x2 r262, r4, r261;
}
{
add.f16x2 r265, %24, r262;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f300;
cvt.rn.f16.f32 high, f300;
mov.b32 r268, {low, high};
}
{
mul.f16x2 r269, r7, r268;
}
{
add.f16x2 r272, r246, r269;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r275, {low, high};
}
{
mul.f16x2 r276, r13, r275;
}
{
add.f16x2 r279, r251, r276;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r282, {low, high};
}
{
mul.f16x2 r283, r22, r282;
}
{
add.f16x2 r286, r258, r283;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r289, {low, high};
}
{
mul.f16x2 r290, r16, r289;
}
{
add.f16x2 r293, r265, r290;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r296, {low, high};
}
{
mul.f16x2 r297, r19, r296;
}
{
add.f16x2 r300, r272, r297;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r303, {low, high};
}
{
mul.f16x2 r304, r25, r303;
}
{
add.f16x2 r307, r279, r304;
}
mov.f32 f352, 0fBE903F40;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r310, {low, high};
}
{
mul.f16x2 r311, r34, r310;
}
{
add.f16x2 r314, r286, r311;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r317, {low, high};
}
{
mul.f16x2 r318, r28, r317;
}
{
add.f16x2 r321, r293, r318;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r324, {low, high};
}
{
mul.f16x2 r325, r31, r324;
}
{
add.f16x2 r328, r300, r325;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r331, {low, high};
}
{
mul.f16x2 r332, r37, r331;
}
{
add.f16x2 r335, r307, r332;
}
mov.f32 f396, 0fBF7D64F0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f396;
cvt.rn.f16.f32 high, f396;
mov.b32 r338, {low, high};
}
{
mul.f16x2 r339, r46, r338;
}
{
add.f16x2 r342, r314, r339;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r345, {low, high};
}
{
mul.f16x2 r346, r40, r345;
}
{
add.f16x2 r349, r321, r346;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f396;
cvt.rn.f16.f32 high, f396;
mov.b32 r352, {low, high};
}
{
mul.f16x2 r353, r43, r352;
}
{
add.f16x2 r356, r328, r353;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r359, {low, high};
}
{
mul.f16x2 r360, r49, r359;
}
{
add.f16x2 r363, r335, r360;
}
mov.f32 f440, 0fBF0A6770;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f440;
cvt.rn.f16.f32 high, f440;
mov.b32 r366, {low, high};
}
{
mul.f16x2 r367, r58, r366;
}
{
add.f16x2 r370, r342, r367;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r373, {low, high};
}
{
mul.f16x2 r374, r52, r373;
}
{
add.f16x2 r377, r349, r374;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f440;
cvt.rn.f16.f32 high, f440;
mov.b32 r380, {low, high};
}
{
mul.f16x2 r381, r55, r380;
}
{
add.f16x2 r384, r356, r381;
}
{
sub.f16x2 r387, r363, r370;
}
{
add.f16x2 r390, r377, r384;
}
{
add.f16x2 r393, r363, r370;
}
{
sub.f16x2 r396, r377, r384;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r399, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r400, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r401, {low, high};
}
{
mul.f16x2 r402, r1, r401;
}
{
add.f16x2 r405, %23, r402;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r408, {low, high};
}
{
mul.f16x2 r409, r10, r408;
}
{
add.f16x2 r412, r399, r409;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r415, {low, high};
}
{
mul.f16x2 r416, r4, r415;
}
{
add.f16x2 r419, %24, r416;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r422, {low, high};
}
{
mul.f16x2 r423, r7, r422;
}
{
add.f16x2 r426, r400, r423;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r429, {low, high};
}
{
mul.f16x2 r430, r13, r429;
}
{
add.f16x2 r433, r405, r430;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r436, {low, high};
}
{
mul.f16x2 r437, r22, r436;
}
{
add.f16x2 r440, r412, r437;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r443, {low, high};
}
{
mul.f16x2 r444, r16, r443;
}
{
add.f16x2 r447, r419, r444;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r450, {low, high};
}
{
mul.f16x2 r451, r19, r450;
}
{
add.f16x2 r454, r426, r451;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r457, {low, high};
}
{
mul.f16x2 r458, r25, r457;
}
{
add.f16x2 r461, r433, r458;
}
mov.f32 f456, 0fBF68DDA4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r464, {low, high};
}
{
mul.f16x2 r465, r34, r464;
}
{
add.f16x2 r468, r440, r465;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r471, {low, high};
}
{
mul.f16x2 r472, r28, r471;
}
{
add.f16x2 r475, r447, r472;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r478, {low, high};
}
{
mul.f16x2 r479, r31, r478;
}
{
add.f16x2 r482, r454, r479;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r485, {low, high};
}
{
mul.f16x2 r486, r37, r485;
}
{
add.f16x2 r489, r461, r486;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r492, {low, high};
}
{
mul.f16x2 r493, r46, r492;
}
{
add.f16x2 r496, r468, r493;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r499, {low, high};
}
{
mul.f16x2 r500, r40, r499;
}
{
add.f16x2 r503, r475, r500;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r506, {low, high};
}
{
mul.f16x2 r507, r43, r506;
}
{
add.f16x2 r510, r482, r507;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r513, {low, high};
}
{
mul.f16x2 r514, r49, r513;
}
{
add.f16x2 r517, r489, r514;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r520, {low, high};
}
{
mul.f16x2 r521, r58, r520;
}
{
add.f16x2 r524, r496, r521;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r527, {low, high};
}
{
mul.f16x2 r528, r52, r527;
}
{
add.f16x2 r531, r503, r528;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r534, {low, high};
}
{
mul.f16x2 r535, r55, r534;
}
{
add.f16x2 r538, r510, r535;
}
{
sub.f16x2 r541, r517, r524;
}
{
add.f16x2 r544, r531, r538;
}
{
add.f16x2 r547, r517, r524;
}
{
sub.f16x2 r550, r531, r538;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r553, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r554, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r555, {low, high};
}
{
mul.f16x2 r556, r1, r555;
}
{
add.f16x2 r559, %23, r556;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r562, {low, high};
}
{
mul.f16x2 r563, r10, r562;
}
{
add.f16x2 r566, r553, r563;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r569, {low, high};
}
{
mul.f16x2 r570, r4, r569;
}
{
add.f16x2 r573, %24, r570;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r576, {low, high};
}
{
mul.f16x2 r577, r7, r576;
}
{
add.f16x2 r580, r554, r577;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r583, {low, high};
}
{
mul.f16x2 r584, r13, r583;
}
{
add.f16x2 r587, r559, r584;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f396;
cvt.rn.f16.f32 high, f396;
mov.b32 r590, {low, high};
}
{
mul.f16x2 r591, r22, r590;
}
{
add.f16x2 r594, r566, r591;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r597, {low, high};
}
{
mul.f16x2 r598, r16, r597;
}
{
add.f16x2 r601, r573, r598;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f396;
cvt.rn.f16.f32 high, f396;
mov.b32 r604, {low, high};
}
{
mul.f16x2 r605, r19, r604;
}
{
add.f16x2 r608, r580, r605;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r611, {low, high};
}
{
mul.f16x2 r612, r25, r611;
}
{
add.f16x2 r615, r587, r612;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r618, {low, high};
}
{
mul.f16x2 r619, r34, r618;
}
{
add.f16x2 r622, r594, r619;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r625, {low, high};
}
{
mul.f16x2 r626, r28, r625;
}
{
add.f16x2 r629, r601, r626;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r632, {low, high};
}
{
mul.f16x2 r633, r31, r632;
}
{
add.f16x2 r636, r608, r633;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r639, {low, high};
}
{
mul.f16x2 r640, r37, r639;
}
{
add.f16x2 r643, r615, r640;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r646, {low, high};
}
{
mul.f16x2 r647, r46, r646;
}
{
add.f16x2 r650, r622, r647;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r653, {low, high};
}
{
mul.f16x2 r654, r40, r653;
}
{
add.f16x2 r657, r629, r654;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r660, {low, high};
}
{
mul.f16x2 r661, r43, r660;
}
{
add.f16x2 r664, r636, r661;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r667, {low, high};
}
{
mul.f16x2 r668, r49, r667;
}
{
add.f16x2 r671, r643, r668;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r674, {low, high};
}
{
mul.f16x2 r675, r58, r674;
}
{
add.f16x2 r678, r650, r675;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r681, {low, high};
}
{
mul.f16x2 r682, r52, r681;
}
{
add.f16x2 r685, r657, r682;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r688, {low, high};
}
{
mul.f16x2 r689, r55, r688;
}
{
add.f16x2 r692, r664, r689;
}
{
sub.f16x2 r695, r671, r678;
}
{
add.f16x2 r698, r685, r692;
}
{
add.f16x2 r701, r671, r678;
}
{
sub.f16x2 r704, r685, r692;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r707, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r708, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r709, {low, high};
}
{
mul.f16x2 r710, r1, r709;
}
{
add.f16x2 r713, %23, r710;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r716, {low, high};
}
{
mul.f16x2 r717, r10, r716;
}
{
add.f16x2 r720, r707, r717;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r723, {low, high};
}
{
mul.f16x2 r724, r4, r723;
}
{
add.f16x2 r727, %24, r724;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r730, {low, high};
}
{
mul.f16x2 r731, r7, r730;
}
{
add.f16x2 r734, r708, r731;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r737, {low, high};
}
{
mul.f16x2 r738, r13, r737;
}
{
add.f16x2 r741, r713, r738;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f440;
cvt.rn.f16.f32 high, f440;
mov.b32 r744, {low, high};
}
{
mul.f16x2 r745, r22, r744;
}
{
add.f16x2 r748, r720, r745;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r751, {low, high};
}
{
mul.f16x2 r752, r16, r751;
}
{
add.f16x2 r755, r727, r752;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f440;
cvt.rn.f16.f32 high, f440;
mov.b32 r758, {low, high};
}
{
mul.f16x2 r759, r19, r758;
}
{
add.f16x2 r762, r734, r759;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r765, {low, high};
}
{
mul.f16x2 r766, r25, r765;
}
{
add.f16x2 r769, r741, r766;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r772, {low, high};
}
{
mul.f16x2 r773, r34, r772;
}
{
add.f16x2 r776, r748, r773;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r779, {low, high};
}
{
mul.f16x2 r780, r28, r779;
}
{
add.f16x2 r783, r755, r780;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r786, {low, high};
}
{
mul.f16x2 r787, r31, r786;
}
{
add.f16x2 r790, r762, r787;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r793, {low, high};
}
{
mul.f16x2 r794, r37, r793;
}
{
add.f16x2 r797, r769, r794;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r800, {low, high};
}
{
mul.f16x2 r801, r46, r800;
}
{
add.f16x2 r804, r776, r801;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r807, {low, high};
}
{
mul.f16x2 r808, r40, r807;
}
{
add.f16x2 r811, r783, r808;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r814, {low, high};
}
{
mul.f16x2 r815, r43, r814;
}
{
add.f16x2 r818, r790, r815;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r821, {low, high};
}
{
mul.f16x2 r822, r49, r821;
}
{
add.f16x2 r825, r797, r822;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r828, {low, high};
}
{
mul.f16x2 r829, r58, r828;
}
{
add.f16x2 r832, r804, r829;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r835, {low, high};
}
{
mul.f16x2 r836, r52, r835;
}
{
add.f16x2 r839, r811, r836;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r842, {low, high};
}
{
mul.f16x2 r843, r55, r842;
}
{
add.f16x2 r846, r818, r843;
}
{
sub.f16x2 r849, r825, r832;
}
{
add.f16x2 r852, r839, r846;
}
{
add.f16x2 r855, r825, r832;
}
{
sub.f16x2 r858, r839, r846;
}
mul.wide.u32 rd2, r2097, -1171354717;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r2098, rd3;
mul.lo.s32 r2099, r2098, 11;
sub.s32 r2100, r2097, r2099;
mad.lo.s32 r2101, r2098, 484, r2096;
cvt.rn.f32.u32 f465, r2100;
mul.f32 f466, f465, 0f3D54B191;
cos.approx.f32 f221, f466;
sin.approx.f32 f467, f466;
neg.f32 f222, f467;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f221;
cvt.rn.f16.f32 high, f222;
mov.b32 r861, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r864, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r866, {high, high};
}
{
mul.f16x2 r868, r236, r866;
}
{
fma.rn.f16x2 r871, r233, r864, r868;
}
{
mul.f16x2 r875, r233, r866;
}
{
neg.f16x2 r878, r875;
}
{
fma.rn.f16x2 r880, r236, r864, r878;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r884, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r886, {high, high};
}
mov.f32 f241, 0fBF800000;
mov.f32 f242, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f241;
cvt.rn.f16.f32 high, f242;
mov.b32 r888, {low, high};
}
{
mul.f16x2 r889, r886, r888;
}
{
mul.f16x2 r892, r861, r884;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r895, {high, low};
}
{
fma.rn.f16x2 r897, r889, r895, r892;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r897;
mov.b32 r901, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r897;
mov.b32 r903, {high, high};
}
{
mul.f16x2 r905, r390, r903;
}
{
fma.rn.f16x2 r908, r387, r901, r905;
}
{
mul.f16x2 r912, r387, r903;
}
{
neg.f16x2 r915, r912;
}
{
fma.rn.f16x2 r917, r390, r901, r915;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r921, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r923, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f241;
cvt.rn.f16.f32 high, f242;
mov.b32 r925, {low, high};
}
{
mul.f16x2 r926, r923, r925;
}
{
mul.f16x2 r929, r897, r921;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r897;
mov.b32 r932, {high, low};
}
{
fma.rn.f16x2 r934, r926, r932, r929;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r934;
mov.b32 r938, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r934;
mov.b32 r940, {high, high};
}
{
mul.f16x2 r942, r544, r940;
}
{
fma.rn.f16x2 r945, r541, r938, r942;
}
{
mul.f16x2 r949, r541, r940;
}
{
neg.f16x2 r952, r949;
}
{
fma.rn.f16x2 r954, r544, r938, r952;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r958, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r960, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f241;
cvt.rn.f16.f32 high, f242;
mov.b32 r962, {low, high};
}
{
mul.f16x2 r963, r960, r962;
}
{
mul.f16x2 r966, r934, r958;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r934;
mov.b32 r969, {high, low};
}
{
fma.rn.f16x2 r971, r963, r969, r966;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r971;
mov.b32 r975, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r971;
mov.b32 r977, {high, high};
}
{
mul.f16x2 r979, r698, r977;
}
{
fma.rn.f16x2 r982, r695, r975, r979;
}
{
mul.f16x2 r986, r695, r977;
}
{
neg.f16x2 r989, r986;
}
{
fma.rn.f16x2 r991, r698, r975, r989;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r995, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r997, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f241;
cvt.rn.f16.f32 high, f242;
mov.b32 r999, {low, high};
}
{
mul.f16x2 r1000, r997, r999;
}
{
mul.f16x2 r1003, r971, r995;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r971;
mov.b32 r1006, {high, low};
}
{
fma.rn.f16x2 r1008, r1000, r1006, r1003;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1008;
mov.b32 r1012, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1008;
mov.b32 r1014, {high, high};
}
{
mul.f16x2 r1016, r852, r1014;
}
{
fma.rn.f16x2 r1019, r849, r1012, r1016;
}
{
mul.f16x2 r1023, r849, r1014;
}
{
neg.f16x2 r1026, r1023;
}
{
fma.rn.f16x2 r1028, r852, r1012, r1026;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r1032, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r1034, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f241;
cvt.rn.f16.f32 high, f242;
mov.b32 r1036, {low, high};
}
{
mul.f16x2 r1037, r1034, r1036;
}
{
mul.f16x2 r1040, r1008, r1032;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1008;
mov.b32 r1043, {high, low};
}
{
fma.rn.f16x2 r1045, r1037, r1043, r1040;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1045;
mov.b32 r1049, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1045;
mov.b32 r1051, {high, high};
}
{
mul.f16x2 r1053, r858, r1051;
}
{
fma.rn.f16x2 r1056, r855, r1049, r1053;
}
{
mul.f16x2 r1060, r855, r1051;
}
{
neg.f16x2 r1063, r1060;
}
{
fma.rn.f16x2 r1065, r858, r1049, r1063;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r1069, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r1071, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f241;
cvt.rn.f16.f32 high, f242;
mov.b32 r1073, {low, high};
}
{
mul.f16x2 r1074, r1071, r1073;
}
{
mul.f16x2 r1077, r1045, r1069;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1045;
mov.b32 r1080, {high, low};
}
{
fma.rn.f16x2 r1082, r1074, r1080, r1077;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1082;
mov.b32 r1086, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1082;
mov.b32 r1088, {high, high};
}
{
mul.f16x2 r1090, r704, r1088;
}
{
fma.rn.f16x2 r1093, r701, r1086, r1090;
}
{
mul.f16x2 r1097, r701, r1088;
}
{
neg.f16x2 r1100, r1097;
}
{
fma.rn.f16x2 r1102, r704, r1086, r1100;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r1106, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r1108, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f241;
cvt.rn.f16.f32 high, f242;
mov.b32 r1110, {low, high};
}
{
mul.f16x2 r1111, r1108, r1110;
}
{
mul.f16x2 r1114, r1082, r1106;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1082;
mov.b32 r1117, {high, low};
}
{
fma.rn.f16x2 r1119, r1111, r1117, r1114;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1119;
mov.b32 r1123, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1119;
mov.b32 r1125, {high, high};
}
{
mul.f16x2 r1127, r550, r1125;
}
{
fma.rn.f16x2 r1130, r547, r1123, r1127;
}
{
mul.f16x2 r1134, r547, r1125;
}
{
neg.f16x2 r1137, r1134;
}
{
fma.rn.f16x2 r1139, r550, r1123, r1137;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r1143, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r1145, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f241;
cvt.rn.f16.f32 high, f242;
mov.b32 r1147, {low, high};
}
{
mul.f16x2 r1148, r1145, r1147;
}
{
mul.f16x2 r1151, r1119, r1143;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1119;
mov.b32 r1154, {high, low};
}
{
fma.rn.f16x2 r1156, r1148, r1154, r1151;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1156;
mov.b32 r1160, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1156;
mov.b32 r1162, {high, high};
}
{
mul.f16x2 r1164, r396, r1162;
}
{
fma.rn.f16x2 r1167, r393, r1160, r1164;
}
{
mul.f16x2 r1171, r393, r1162;
}
{
neg.f16x2 r1174, r1171;
}
{
fma.rn.f16x2 r1176, r396, r1160, r1174;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r1180, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r861;
mov.b32 r1182, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f241;
cvt.rn.f16.f32 high, f242;
mov.b32 r1184, {low, high};
}
{
mul.f16x2 r1185, r1182, r1184;
}
{
mul.f16x2 r1188, r1156, r1180;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1156;
mov.b32 r1191, {high, low};
}
{
fma.rn.f16x2 r1193, r1185, r1191, r1188;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1193;
mov.b32 r1197, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1193;
mov.b32 r1199, {high, high};
}
{
mul.f16x2 r1201, r242, r1199;
}
{
fma.rn.f16x2 r1204, r239, r1197, r1201;
}
{
mul.f16x2 r1208, r239, r1199;
}
{
neg.f16x2 r1211, r1208;
}
{
fma.rn.f16x2 r1213, r242, r1197, r1211;
}
barrier.sync 0;
mad.lo.s32 r2102, r2100, 44, r2101;
st.shared.u32 [r2102], r85;
st.shared.u32 [r2102+4], r871;
st.shared.u32 [r2102+8], r908;
st.shared.u32 [r2102+12], r945;
st.shared.u32 [r2102+16], r982;
st.shared.u32 [r2102+20], r1019;
st.shared.u32 [r2102+24], r1056;
st.shared.u32 [r2102+28], r1093;
st.shared.u32 [r2102+32], r1130;
st.shared.u32 [r2102+36], r1167;
st.shared.u32 [r2102+40], r1204;
barrier.sync 0;
mad.lo.s32 r2103, r2100, -40, r2102;
ld.shared.u32 r1295, [r2103];
ld.shared.u32 r1235, [r2103+44];
ld.shared.u32 r1247, [r2103+88];
ld.shared.u32 r1259, [r2103+132];
ld.shared.u32 r1271, [r2103+176];
ld.shared.u32 r1283, [r2103+220];
ld.shared.u32 r1284, [r2103+264];
ld.shared.u32 r1272, [r2103+308];
ld.shared.u32 r1260, [r2103+352];
ld.shared.u32 r1248, [r2103+396];
ld.shared.u32 r1236, [r2103+440];
barrier.sync 0;
st.shared.u32 [r2102], r88;
st.shared.u32 [r2102+4], r880;
st.shared.u32 [r2102+8], r917;
st.shared.u32 [r2102+12], r954;
st.shared.u32 [r2102+16], r991;
st.shared.u32 [r2102+20], r1028;
st.shared.u32 [r2102+24], r1065;
st.shared.u32 [r2102+28], r1102;
st.shared.u32 [r2102+32], r1139;
st.shared.u32 [r2102+36], r1176;
st.shared.u32 [r2102+40], r1213;
barrier.sync 0;
ld.shared.u32 r1298, [r2103];
ld.shared.u32 r1238, [r2103+44];
ld.shared.u32 r1250, [r2103+88];
ld.shared.u32 r1262, [r2103+132];
ld.shared.u32 r1274, [r2103+176];
ld.shared.u32 r1286, [r2103+220];
ld.shared.u32 r1287, [r2103+264];
ld.shared.u32 r1275, [r2103+308];
ld.shared.u32 r1263, [r2103+352];
ld.shared.u32 r1251, [r2103+396];
ld.shared.u32 r1239, [r2103+440];
{
add.f16x2 r1234, r1235, r1236;
}
{
add.f16x2 r1237, r1238, r1239;
}
{
sub.f16x2 r1240, r1235, r1236;
}
{
sub.f16x2 r1243, r1238, r1239;
}
{
add.f16x2 r1246, r1247, r1248;
}
{
add.f16x2 r1249, r1250, r1251;
}
{
sub.f16x2 r1252, r1247, r1248;
}
{
sub.f16x2 r1255, r1250, r1251;
}
{
add.f16x2 r1258, r1259, r1260;
}
{
add.f16x2 r1261, r1262, r1263;
}
{
sub.f16x2 r1264, r1259, r1260;
}
{
sub.f16x2 r1267, r1262, r1263;
}
{
add.f16x2 r1270, r1271, r1272;
}
{
add.f16x2 r1273, r1274, r1275;
}
{
sub.f16x2 r1276, r1271, r1272;
}
{
sub.f16x2 r1279, r1274, r1275;
}
{
add.f16x2 r1282, r1283, r1284;
}
{
add.f16x2 r1285, r1286, r1287;
}
{
sub.f16x2 r1288, r1283, r1284;
}
{
sub.f16x2 r1291, r1286, r1287;
}
{
add.f16x2 r1294, r1295, r1234;
}
{
add.f16x2 r1297, r1298, r1237;
}
{
add.f16x2 r1300, r1294, r1246;
}
{
add.f16x2 r1303, r1297, r1249;
}
{
add.f16x2 r1306, r1300, r1258;
}
{
add.f16x2 r1309, r1303, r1261;
}
{
add.f16x2 r1312, r1306, r1270;
}
{
add.f16x2 r1315, r1309, r1273;
}
{
add.f16x2 %0, r1312, r1282;
}
{
add.f16x2 %1, r1315, r1285;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r1324, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r1325, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r1326, {low, high};
}
{
mul.f16x2 r1327, r1234, r1326;
}
{
add.f16x2 r1330, r1295, r1327;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r1333, {low, high};
}
{
mul.f16x2 r1334, r1243, r1333;
}
{
add.f16x2 r1337, r1324, r1334;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r1340, {low, high};
}
{
mul.f16x2 r1341, r1237, r1340;
}
{
add.f16x2 r1344, r1298, r1341;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r1347, {low, high};
}
{
mul.f16x2 r1348, r1240, r1347;
}
{
add.f16x2 r1351, r1325, r1348;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r1354, {low, high};
}
{
mul.f16x2 r1355, r1246, r1354;
}
{
add.f16x2 r1358, r1330, r1355;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f300;
cvt.rn.f16.f32 high, f300;
mov.b32 r1361, {low, high};
}
{
mul.f16x2 r1362, r1255, r1361;
}
{
add.f16x2 r1365, r1337, r1362;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r1368, {low, high};
}
{
mul.f16x2 r1369, r1249, r1368;
}
{
add.f16x2 r1372, r1344, r1369;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f300;
cvt.rn.f16.f32 high, f300;
mov.b32 r1375, {low, high};
}
{
mul.f16x2 r1376, r1252, r1375;
}
{
add.f16x2 r1379, r1351, r1376;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r1382, {low, high};
}
{
mul.f16x2 r1383, r1258, r1382;
}
{
add.f16x2 r1386, r1358, r1383;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r1389, {low, high};
}
{
mul.f16x2 r1390, r1267, r1389;
}
{
add.f16x2 r1393, r1365, r1390;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r1396, {low, high};
}
{
mul.f16x2 r1397, r1261, r1396;
}
{
add.f16x2 r1400, r1372, r1397;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r1403, {low, high};
}
{
mul.f16x2 r1404, r1264, r1403;
}
{
add.f16x2 r1407, r1379, r1404;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r1410, {low, high};
}
{
mul.f16x2 r1411, r1270, r1410;
}
{
add.f16x2 r1414, r1386, r1411;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r1417, {low, high};
}
{
mul.f16x2 r1418, r1279, r1417;
}
{
add.f16x2 r1421, r1393, r1418;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r1424, {low, high};
}
{
mul.f16x2 r1425, r1273, r1424;
}
{
add.f16x2 r1428, r1400, r1425;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r1431, {low, high};
}
{
mul.f16x2 r1432, r1276, r1431;
}
{
add.f16x2 r1435, r1407, r1432;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r1438, {low, high};
}
{
mul.f16x2 r1439, r1282, r1438;
}
{
add.f16x2 r1442, r1414, r1439;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r1445, {low, high};
}
{
mul.f16x2 r1446, r1291, r1445;
}
{
add.f16x2 r1449, r1421, r1446;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r1452, {low, high};
}
{
mul.f16x2 r1453, r1285, r1452;
}
{
add.f16x2 r1456, r1428, r1453;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r1459, {low, high};
}
{
mul.f16x2 r1460, r1288, r1459;
}
{
add.f16x2 r1463, r1435, r1460;
}
{
sub.f16x2 %2, r1442, r1449;
}
{
add.f16x2 %3, r1456, r1463;
}
{
add.f16x2 %20, r1442, r1449;
}
{
sub.f16x2 %21, r1456, r1463;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r1478, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r1479, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r1480, {low, high};
}
{
mul.f16x2 r1481, r1234, r1480;
}
{
add.f16x2 r1484, r1295, r1481;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f300;
cvt.rn.f16.f32 high, f300;
mov.b32 r1487, {low, high};
}
{
mul.f16x2 r1488, r1243, r1487;
}
{
add.f16x2 r1491, r1478, r1488;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r1494, {low, high};
}
{
mul.f16x2 r1495, r1237, r1494;
}
{
add.f16x2 r1498, r1298, r1495;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f300;
cvt.rn.f16.f32 high, f300;
mov.b32 r1501, {low, high};
}
{
mul.f16x2 r1502, r1240, r1501;
}
{
add.f16x2 r1505, r1479, r1502;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r1508, {low, high};
}
{
mul.f16x2 r1509, r1246, r1508;
}
{
add.f16x2 r1512, r1484, r1509;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r1515, {low, high};
}
{
mul.f16x2 r1516, r1255, r1515;
}
{
add.f16x2 r1519, r1491, r1516;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r1522, {low, high};
}
{
mul.f16x2 r1523, r1249, r1522;
}
{
add.f16x2 r1526, r1498, r1523;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r1529, {low, high};
}
{
mul.f16x2 r1530, r1252, r1529;
}
{
add.f16x2 r1533, r1505, r1530;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r1536, {low, high};
}
{
mul.f16x2 r1537, r1258, r1536;
}
{
add.f16x2 r1540, r1512, r1537;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r1543, {low, high};
}
{
mul.f16x2 r1544, r1267, r1543;
}
{
add.f16x2 r1547, r1519, r1544;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r1550, {low, high};
}
{
mul.f16x2 r1551, r1261, r1550;
}
{
add.f16x2 r1554, r1526, r1551;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r1557, {low, high};
}
{
mul.f16x2 r1558, r1264, r1557;
}
{
add.f16x2 r1561, r1533, r1558;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r1564, {low, high};
}
{
mul.f16x2 r1565, r1270, r1564;
}
{
add.f16x2 r1568, r1540, r1565;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f396;
cvt.rn.f16.f32 high, f396;
mov.b32 r1571, {low, high};
}
{
mul.f16x2 r1572, r1279, r1571;
}
{
add.f16x2 r1575, r1547, r1572;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r1578, {low, high};
}
{
mul.f16x2 r1579, r1273, r1578;
}
{
add.f16x2 r1582, r1554, r1579;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f396;
cvt.rn.f16.f32 high, f396;
mov.b32 r1585, {low, high};
}
{
mul.f16x2 r1586, r1276, r1585;
}
{
add.f16x2 r1589, r1561, r1586;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r1592, {low, high};
}
{
mul.f16x2 r1593, r1282, r1592;
}
{
add.f16x2 r1596, r1568, r1593;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f440;
cvt.rn.f16.f32 high, f440;
mov.b32 r1599, {low, high};
}
{
mul.f16x2 r1600, r1291, r1599;
}
{
add.f16x2 r1603, r1575, r1600;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r1606, {low, high};
}
{
mul.f16x2 r1607, r1285, r1606;
}
{
add.f16x2 r1610, r1582, r1607;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f440;
cvt.rn.f16.f32 high, f440;
mov.b32 r1613, {low, high};
}
{
mul.f16x2 r1614, r1288, r1613;
}
{
add.f16x2 r1617, r1589, r1614;
}
{
sub.f16x2 %4, r1596, r1603;
}
{
add.f16x2 %5, r1610, r1617;
}
{
add.f16x2 %18, r1596, r1603;
}
{
sub.f16x2 %19, r1610, r1617;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r1632, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r1633, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r1634, {low, high};
}
{
mul.f16x2 r1635, r1234, r1634;
}
{
add.f16x2 r1638, r1295, r1635;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r1641, {low, high};
}
{
mul.f16x2 r1642, r1243, r1641;
}
{
add.f16x2 r1645, r1632, r1642;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r1648, {low, high};
}
{
mul.f16x2 r1649, r1237, r1648;
}
{
add.f16x2 r1652, r1298, r1649;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r1655, {low, high};
}
{
mul.f16x2 r1656, r1240, r1655;
}
{
add.f16x2 r1659, r1633, r1656;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r1662, {low, high};
}
{
mul.f16x2 r1663, r1246, r1662;
}
{
add.f16x2 r1666, r1638, r1663;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r1669, {low, high};
}
{
mul.f16x2 r1670, r1255, r1669;
}
{
add.f16x2 r1673, r1645, r1670;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r1676, {low, high};
}
{
mul.f16x2 r1677, r1249, r1676;
}
{
add.f16x2 r1680, r1652, r1677;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r1683, {low, high};
}
{
mul.f16x2 r1684, r1252, r1683;
}
{
add.f16x2 r1687, r1659, r1684;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r1690, {low, high};
}
{
mul.f16x2 r1691, r1258, r1690;
}
{
add.f16x2 r1694, r1666, r1691;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r1697, {low, high};
}
{
mul.f16x2 r1698, r1267, r1697;
}
{
add.f16x2 r1701, r1673, r1698;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r1704, {low, high};
}
{
mul.f16x2 r1705, r1261, r1704;
}
{
add.f16x2 r1708, r1680, r1705;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r1711, {low, high};
}
{
mul.f16x2 r1712, r1264, r1711;
}
{
add.f16x2 r1715, r1687, r1712;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r1718, {low, high};
}
{
mul.f16x2 r1719, r1270, r1718;
}
{
add.f16x2 r1722, r1694, r1719;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r1725, {low, high};
}
{
mul.f16x2 r1726, r1279, r1725;
}
{
add.f16x2 r1729, r1701, r1726;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r1732, {low, high};
}
{
mul.f16x2 r1733, r1273, r1732;
}
{
add.f16x2 r1736, r1708, r1733;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r1739, {low, high};
}
{
mul.f16x2 r1740, r1276, r1739;
}
{
add.f16x2 r1743, r1715, r1740;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r1746, {low, high};
}
{
mul.f16x2 r1747, r1282, r1746;
}
{
add.f16x2 r1750, r1722, r1747;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r1753, {low, high};
}
{
mul.f16x2 r1754, r1291, r1753;
}
{
add.f16x2 r1757, r1729, r1754;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r1760, {low, high};
}
{
mul.f16x2 r1761, r1285, r1760;
}
{
add.f16x2 r1764, r1736, r1761;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r1767, {low, high};
}
{
mul.f16x2 r1768, r1288, r1767;
}
{
add.f16x2 r1771, r1743, r1768;
}
{
sub.f16x2 %6, r1750, r1757;
}
{
add.f16x2 %7, r1764, r1771;
}
{
add.f16x2 %16, r1750, r1757;
}
{
sub.f16x2 %17, r1764, r1771;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r1786, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r1787, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r1788, {low, high};
}
{
mul.f16x2 r1789, r1234, r1788;
}
{
add.f16x2 r1792, r1295, r1789;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r1795, {low, high};
}
{
mul.f16x2 r1796, r1243, r1795;
}
{
add.f16x2 r1799, r1786, r1796;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r1802, {low, high};
}
{
mul.f16x2 r1803, r1237, r1802;
}
{
add.f16x2 r1806, r1298, r1803;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r1809, {low, high};
}
{
mul.f16x2 r1810, r1240, r1809;
}
{
add.f16x2 r1813, r1787, r1810;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r1816, {low, high};
}
{
mul.f16x2 r1817, r1246, r1816;
}
{
add.f16x2 r1820, r1792, r1817;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f396;
cvt.rn.f16.f32 high, f396;
mov.b32 r1823, {low, high};
}
{
mul.f16x2 r1824, r1255, r1823;
}
{
add.f16x2 r1827, r1799, r1824;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r1830, {low, high};
}
{
mul.f16x2 r1831, r1249, r1830;
}
{
add.f16x2 r1834, r1806, r1831;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f396;
cvt.rn.f16.f32 high, f396;
mov.b32 r1837, {low, high};
}
{
mul.f16x2 r1838, r1252, r1837;
}
{
add.f16x2 r1841, r1813, r1838;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r1844, {low, high};
}
{
mul.f16x2 r1845, r1258, r1844;
}
{
add.f16x2 r1848, r1820, r1845;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r1851, {low, high};
}
{
mul.f16x2 r1852, r1267, r1851;
}
{
add.f16x2 r1855, r1827, r1852;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r1858, {low, high};
}
{
mul.f16x2 r1859, r1261, r1858;
}
{
add.f16x2 r1862, r1834, r1859;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f404;
cvt.rn.f16.f32 high, f404;
mov.b32 r1865, {low, high};
}
{
mul.f16x2 r1866, r1264, r1865;
}
{
add.f16x2 r1869, r1841, r1866;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r1872, {low, high};
}
{
mul.f16x2 r1873, r1270, r1872;
}
{
add.f16x2 r1876, r1848, r1873;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r1879, {low, high};
}
{
mul.f16x2 r1880, r1279, r1879;
}
{
add.f16x2 r1883, r1855, r1880;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r1886, {low, high};
}
{
mul.f16x2 r1887, r1273, r1886;
}
{
add.f16x2 r1890, r1862, r1887;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r1893, {low, high};
}
{
mul.f16x2 r1894, r1276, r1893;
}
{
add.f16x2 r1897, r1869, r1894;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r1900, {low, high};
}
{
mul.f16x2 r1901, r1282, r1900;
}
{
add.f16x2 r1904, r1876, r1901;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r1907, {low, high};
}
{
mul.f16x2 r1908, r1291, r1907;
}
{
add.f16x2 r1911, r1883, r1908;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r1914, {low, high};
}
{
mul.f16x2 r1915, r1285, r1914;
}
{
add.f16x2 r1918, r1890, r1915;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r1921, {low, high};
}
{
mul.f16x2 r1922, r1288, r1921;
}
{
add.f16x2 r1925, r1897, r1922;
}
{
sub.f16x2 %8, r1904, r1911;
}
{
add.f16x2 %9, r1918, r1925;
}
{
add.f16x2 %14, r1904, r1911;
}
{
sub.f16x2 %15, r1918, r1925;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r1940, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f424;
cvt.rn.f16.f32 high, f424;
mov.b32 r1941, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r1942, {low, high};
}
{
mul.f16x2 r1943, r1234, r1942;
}
{
add.f16x2 r1946, r1295, r1943;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r1949, {low, high};
}
{
mul.f16x2 r1950, r1243, r1949;
}
{
add.f16x2 r1953, r1940, r1950;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f430;
cvt.rn.f16.f32 high, f430;
mov.b32 r1956, {low, high};
}
{
mul.f16x2 r1957, r1237, r1956;
}
{
add.f16x2 r1960, r1298, r1957;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f432;
cvt.rn.f16.f32 high, f432;
mov.b32 r1963, {low, high};
}
{
mul.f16x2 r1964, r1240, r1963;
}
{
add.f16x2 r1967, r1941, r1964;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r1970, {low, high};
}
{
mul.f16x2 r1971, r1246, r1970;
}
{
add.f16x2 r1974, r1946, r1971;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f440;
cvt.rn.f16.f32 high, f440;
mov.b32 r1977, {low, high};
}
{
mul.f16x2 r1978, r1255, r1977;
}
{
add.f16x2 r1981, r1953, r1978;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f438;
cvt.rn.f16.f32 high, f438;
mov.b32 r1984, {low, high};
}
{
mul.f16x2 r1985, r1249, r1984;
}
{
add.f16x2 r1988, r1960, r1985;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f440;
cvt.rn.f16.f32 high, f440;
mov.b32 r1991, {low, high};
}
{
mul.f16x2 r1992, r1252, r1991;
}
{
add.f16x2 r1995, r1967, r1992;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r1998, {low, high};
}
{
mul.f16x2 r1999, r1258, r1998;
}
{
add.f16x2 r2002, r1974, r1999;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r2005, {low, high};
}
{
mul.f16x2 r2006, r1267, r2005;
}
{
add.f16x2 r2009, r1981, r2006;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f446;
cvt.rn.f16.f32 high, f446;
mov.b32 r2012, {low, high};
}
{
mul.f16x2 r2013, r1261, r2012;
}
{
add.f16x2 r2016, r1988, r2013;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f448;
cvt.rn.f16.f32 high, f448;
mov.b32 r2019, {low, high};
}
{
mul.f16x2 r2020, r1264, r2019;
}
{
add.f16x2 r2023, r1995, r2020;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r2026, {low, high};
}
{
mul.f16x2 r2027, r1270, r2026;
}
{
add.f16x2 r2030, r2002, r2027;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r2033, {low, high};
}
{
mul.f16x2 r2034, r1279, r2033;
}
{
add.f16x2 r2037, r2009, r2034;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f454;
cvt.rn.f16.f32 high, f454;
mov.b32 r2040, {low, high};
}
{
mul.f16x2 r2041, r1273, r2040;
}
{
add.f16x2 r2044, r2016, r2041;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f456;
mov.b32 r2047, {low, high};
}
{
mul.f16x2 r2048, r1276, r2047;
}
{
add.f16x2 r2051, r2023, r2048;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r2054, {low, high};
}
{
mul.f16x2 r2055, r1282, r2054;
}
{
add.f16x2 r2058, r2030, r2055;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r2061, {low, high};
}
{
mul.f16x2 r2062, r1291, r2061;
}
{
add.f16x2 r2065, r2037, r2062;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f462;
cvt.rn.f16.f32 high, f462;
mov.b32 r2068, {low, high};
}
{
mul.f16x2 r2069, r1285, r2068;
}
{
add.f16x2 r2072, r2044, r2069;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f464;
cvt.rn.f16.f32 high, f464;
mov.b32 r2075, {low, high};
}
{
mul.f16x2 r2076, r1288, r2075;
}
{
add.f16x2 r2079, r2051, r2076;
}
{
sub.f16x2 %10, r2058, r2065;
}
{
add.f16x2 %11, r2072, r2079;
}
{
add.f16x2 %12, r2058, r2065;
}
{
sub.f16x2 %13, r2072, r2079;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)));
};


#endif
