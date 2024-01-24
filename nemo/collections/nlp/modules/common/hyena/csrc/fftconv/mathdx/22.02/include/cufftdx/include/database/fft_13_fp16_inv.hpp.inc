#ifndef CUFFTDX_FFT_13_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_13_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<948, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<157>;
.reg .b32 r<1057>;
.reg .f64 fd<145>;
.reg .b64 rd<2>;
{
add.f16x2 r1, %28, %50;
}
{
add.f16x2 r4, %29, %51;
}
{
sub.f16x2 r7, %28, %50;
}
{
sub.f16x2 r10, %29, %51;
}
{
add.f16x2 r13, %30, %48;
}
{
add.f16x2 r16, %31, %49;
}
{
sub.f16x2 r19, %30, %48;
}
{
sub.f16x2 r22, %31, %49;
}
{
add.f16x2 r25, %32, %46;
}
{
add.f16x2 r28, %33, %47;
}
{
sub.f16x2 r31, %32, %46;
}
{
sub.f16x2 r34, %33, %47;
}
{
add.f16x2 r37, %34, %44;
}
{
add.f16x2 r40, %35, %45;
}
{
sub.f16x2 r43, %34, %44;
}
{
sub.f16x2 r46, %35, %45;
}
{
add.f16x2 r49, %36, %42;
}
{
add.f16x2 r52, %37, %43;
}
{
sub.f16x2 r55, %36, %42;
}
{
sub.f16x2 r58, %37, %43;
}
{
add.f16x2 r61, %38, %40;
}
{
add.f16x2 r64, %39, %41;
}
{
sub.f16x2 r67, %38, %40;
}
{
sub.f16x2 r70, %39, %41;
}
{
add.f16x2 r73, %26, r1;
}
{
add.f16x2 r76, %27, r4;
}
{
add.f16x2 r79, r73, r13;
}
{
add.f16x2 r82, r76, r16;
}
{
add.f16x2 r85, r79, r25;
}
{
add.f16x2 r88, r82, r28;
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
add.f16x2 %0, r97, r61;
}
{
add.f16x2 %1, r100, r64;
}
mov.u32 r900, 0;
cvt.rn.f16.s32 rs1, r900;
mov.b32 r121, {rs1, rs1};
cvt.rn.f16.s32 rs2, r900;
mov.b32 r133, {rs2, rs2};
mov.f64 fd127, 0d3FEC55A7E00740E9;
{
cvt.rn.f16.f64 rs3, fd127;
}
mov.b32 r113, {rs3, rs3};
{
mul.f16x2 r111, r1, r113;
}
{
add.f16x2 r114, %26, r111;
}
mov.f64 fd4, 0d3FDDBE064267C47C;
{
cvt.rn.f16.f64 rs4, fd4;
}
mov.b32 r119, {rs4, rs4};
{
mul.f16x2 r117, r10, r119;
}
{
add.f16x2 r120, r121, r117;
}
{
cvt.rn.f16.f64 rs5, fd127;
}
mov.b32 r125, {rs5, rs5};
{
mul.f16x2 r123, r4, r125;
}
{
add.f16x2 r126, %27, r123;
}
{
cvt.rn.f16.f64 rs6, fd4;
}
mov.b32 r131, {rs6, rs6};
{
mul.f16x2 r129, r7, r131;
}
{
add.f16x2 r132, r133, r129;
}
mov.f64 fd135, 0d3FE22D961EA71119;
{
cvt.rn.f16.f64 rs7, fd135;
}
mov.b32 r137, {rs7, rs7};
{
mul.f16x2 r135, r13, r137;
}
{
add.f16x2 r138, r114, r135;
}
mov.f64 fd108, 0d3FEA55E242A4C3D2;
{
cvt.rn.f16.f64 rs8, fd108;
}
mov.b32 r143, {rs8, rs8};
{
mul.f16x2 r141, r22, r143;
}
{
add.f16x2 r144, r120, r141;
}
{
cvt.rn.f16.f64 rs9, fd135;
}
mov.b32 r149, {rs9, rs9};
{
mul.f16x2 r147, r16, r149;
}
{
add.f16x2 r150, r126, r147;
}
{
cvt.rn.f16.f64 rs10, fd108;
}
mov.b32 r155, {rs10, rs10};
{
mul.f16x2 r153, r19, r155;
}
{
add.f16x2 r156, r132, r153;
}
mov.f64 fd143, 0d3FBEDB7DEBAA3ED8;
{
cvt.rn.f16.f64 rs11, fd143;
}
mov.b32 r161, {rs11, rs11};
{
mul.f16x2 r159, r25, r161;
}
{
add.f16x2 r162, r138, r159;
}
mov.f64 fd88, 0d3FEFC44566966769;
{
cvt.rn.f16.f64 rs12, fd88;
}
mov.b32 r167, {rs12, rs12};
{
mul.f16x2 r165, r34, r167;
}
{
add.f16x2 r168, r144, r165;
}
{
cvt.rn.f16.f64 rs13, fd143;
}
mov.b32 r173, {rs13, rs13};
{
mul.f16x2 r171, r28, r173;
}
{
add.f16x2 r174, r150, r171;
}
{
cvt.rn.f16.f64 rs14, fd88;
}
mov.b32 r179, {rs14, rs14};
{
mul.f16x2 r177, r31, r179;
}
{
add.f16x2 r180, r156, r177;
}
mov.f64 fd139, 0dBFD6B1D8B2365DA1;
{
cvt.rn.f16.f64 rs15, fd139;
}
mov.b32 r185, {rs15, rs15};
{
mul.f16x2 r183, r37, r185;
}
{
add.f16x2 r186, r162, r183;
}
mov.f64 fd140, 0d3FEDEBA72EF20147;
{
cvt.rn.f16.f64 rs16, fd140;
}
mov.b32 r191, {rs16, rs16};
{
mul.f16x2 r189, r46, r191;
}
{
add.f16x2 r192, r168, r189;
}
{
cvt.rn.f16.f64 rs17, fd139;
}
mov.b32 r197, {rs17, rs17};
{
mul.f16x2 r195, r40, r197;
}
{
add.f16x2 r198, r174, r195;
}
{
cvt.rn.f16.f64 rs18, fd140;
}
mov.b32 r203, {rs18, rs18};
{
mul.f16x2 r201, r43, r203;
}
{
add.f16x2 r204, r180, r201;
}
mov.f64 fd131, 0dBFE7F3CCD0032E0C;
{
cvt.rn.f16.f64 rs19, fd131;
}
mov.b32 r209, {rs19, rs19};
{
mul.f16x2 r207, r49, r209;
}
{
add.f16x2 r210, r186, r207;
}
mov.f64 fd132, 0d3FE5384D024C2F84;
{
cvt.rn.f16.f64 rs20, fd132;
}
mov.b32 r215, {rs20, rs20};
{
mul.f16x2 r213, r58, r215;
}
{
add.f16x2 r216, r192, r213;
}
{
cvt.rn.f16.f64 rs21, fd131;
}
mov.b32 r221, {rs21, rs21};
{
mul.f16x2 r219, r52, r221;
}
{
add.f16x2 r222, r198, r219;
}
{
cvt.rn.f16.f64 rs22, fd132;
}
mov.b32 r227, {rs22, rs22};
{
mul.f16x2 r225, r55, r227;
}
{
add.f16x2 r228, r204, r225;
}
mov.f64 fd123, 0dBFEF11F493053D00;
{
cvt.rn.f16.f64 rs23, fd123;
}
mov.b32 r233, {rs23, rs23};
{
mul.f16x2 r231, r61, r233;
}
{
add.f16x2 r234, r210, r231;
}
mov.f64 fd124, 0d3FCEA1E54BC48DBF;
{
cvt.rn.f16.f64 rs24, fd124;
}
mov.b32 r239, {rs24, rs24};
{
mul.f16x2 r237, r70, r239;
}
{
add.f16x2 r240, r216, r237;
}
{
cvt.rn.f16.f64 rs25, fd123;
}
mov.b32 r245, {rs25, rs25};
{
mul.f16x2 r243, r64, r245;
}
{
add.f16x2 r246, r222, r243;
}
{
cvt.rn.f16.f64 rs26, fd124;
}
mov.b32 r251, {rs26, rs26};
{
mul.f16x2 r249, r67, r251;
}
{
add.f16x2 r252, r228, r249;
}
{
sub.f16x2 %2, r234, r240;
}
{
add.f16x2 %3, r246, r252;
}
{
add.f16x2 %24, r234, r240;
}
{
sub.f16x2 %25, r246, r252;
}
cvt.rn.f16.s32 rs27, r900;
mov.b32 r279, {rs27, rs27};
cvt.rn.f16.s32 rs28, r900;
mov.b32 r291, {rs28, rs28};
{
cvt.rn.f16.f64 rs29, fd135;
}
mov.b32 r271, {rs29, rs29};
{
mul.f16x2 r269, r1, r271;
}
{
add.f16x2 r272, %26, r269;
}
{
cvt.rn.f16.f64 rs30, fd108;
}
mov.b32 r277, {rs30, rs30};
{
mul.f16x2 r275, r10, r277;
}
{
add.f16x2 r278, r279, r275;
}
{
cvt.rn.f16.f64 rs31, fd135;
}
mov.b32 r283, {rs31, rs31};
{
mul.f16x2 r281, r4, r283;
}
{
add.f16x2 r284, %27, r281;
}
{
cvt.rn.f16.f64 rs32, fd108;
}
mov.b32 r289, {rs32, rs32};
{
mul.f16x2 r287, r7, r289;
}
{
add.f16x2 r290, r291, r287;
}
{
cvt.rn.f16.f64 rs33, fd139;
}
mov.b32 r295, {rs33, rs33};
{
mul.f16x2 r293, r13, r295;
}
{
add.f16x2 r296, r272, r293;
}
{
cvt.rn.f16.f64 rs34, fd140;
}
mov.b32 r301, {rs34, rs34};
{
mul.f16x2 r299, r22, r301;
}
{
add.f16x2 r302, r278, r299;
}
{
cvt.rn.f16.f64 rs35, fd139;
}
mov.b32 r307, {rs35, rs35};
{
mul.f16x2 r305, r16, r307;
}
{
add.f16x2 r308, r284, r305;
}
{
cvt.rn.f16.f64 rs36, fd140;
}
mov.b32 r313, {rs36, rs36};
{
mul.f16x2 r311, r19, r313;
}
{
add.f16x2 r314, r290, r311;
}
{
cvt.rn.f16.f64 rs37, fd123;
}
mov.b32 r319, {rs37, rs37};
{
mul.f16x2 r317, r25, r319;
}
{
add.f16x2 r320, r296, r317;
}
{
cvt.rn.f16.f64 rs38, fd124;
}
mov.b32 r325, {rs38, rs38};
{
mul.f16x2 r323, r34, r325;
}
{
add.f16x2 r326, r302, r323;
}
{
cvt.rn.f16.f64 rs39, fd123;
}
mov.b32 r331, {rs39, rs39};
{
mul.f16x2 r329, r28, r331;
}
{
add.f16x2 r332, r308, r329;
}
{
cvt.rn.f16.f64 rs40, fd124;
}
mov.b32 r337, {rs40, rs40};
{
mul.f16x2 r335, r31, r337;
}
{
add.f16x2 r338, r314, r335;
}
{
cvt.rn.f16.f64 rs41, fd131;
}
mov.b32 r343, {rs41, rs41};
{
mul.f16x2 r341, r37, r343;
}
{
add.f16x2 r344, r320, r341;
}
mov.f64 fd80, 0dBFE5384D024C2F84;
{
cvt.rn.f16.f64 rs42, fd80;
}
mov.b32 r349, {rs42, rs42};
{
mul.f16x2 r347, r46, r349;
}
{
add.f16x2 r350, r326, r347;
}
{
cvt.rn.f16.f64 rs43, fd131;
}
mov.b32 r355, {rs43, rs43};
{
mul.f16x2 r353, r40, r355;
}
{
add.f16x2 r356, r332, r353;
}
{
cvt.rn.f16.f64 rs44, fd80;
}
mov.b32 r361, {rs44, rs44};
{
mul.f16x2 r359, r43, r361;
}
{
add.f16x2 r362, r338, r359;
}
{
cvt.rn.f16.f64 rs45, fd143;
}
mov.b32 r367, {rs45, rs45};
{
mul.f16x2 r365, r49, r367;
}
{
add.f16x2 r368, r344, r365;
}
mov.f64 fd144, 0dBFEFC44566966769;
{
cvt.rn.f16.f64 rs46, fd144;
}
mov.b32 r373, {rs46, rs46};
{
mul.f16x2 r371, r58, r373;
}
{
add.f16x2 r374, r350, r371;
}
{
cvt.rn.f16.f64 rs47, fd143;
}
mov.b32 r379, {rs47, rs47};
{
mul.f16x2 r377, r52, r379;
}
{
add.f16x2 r380, r356, r377;
}
{
cvt.rn.f16.f64 rs48, fd144;
}
mov.b32 r385, {rs48, rs48};
{
mul.f16x2 r383, r55, r385;
}
{
add.f16x2 r386, r362, r383;
}
{
cvt.rn.f16.f64 rs49, fd127;
}
mov.b32 r391, {rs49, rs49};
{
mul.f16x2 r389, r61, r391;
}
{
add.f16x2 r392, r368, r389;
}
mov.f64 fd128, 0dBFDDBE064267C47C;
{
cvt.rn.f16.f64 rs50, fd128;
}
mov.b32 r397, {rs50, rs50};
{
mul.f16x2 r395, r70, r397;
}
{
add.f16x2 r398, r374, r395;
}
{
cvt.rn.f16.f64 rs51, fd127;
}
mov.b32 r403, {rs51, rs51};
{
mul.f16x2 r401, r64, r403;
}
{
add.f16x2 r404, r380, r401;
}
{
cvt.rn.f16.f64 rs52, fd128;
}
mov.b32 r409, {rs52, rs52};
{
mul.f16x2 r407, r67, r409;
}
{
add.f16x2 r410, r386, r407;
}
{
sub.f16x2 %4, r392, r398;
}
{
add.f16x2 %5, r404, r410;
}
{
add.f16x2 %22, r392, r398;
}
{
sub.f16x2 %23, r404, r410;
}
cvt.rn.f16.s32 rs53, r900;
mov.b32 r437, {rs53, rs53};
cvt.rn.f16.s32 rs54, r900;
mov.b32 r449, {rs54, rs54};
{
cvt.rn.f16.f64 rs55, fd143;
}
mov.b32 r429, {rs55, rs55};
{
mul.f16x2 r427, r1, r429;
}
{
add.f16x2 r430, %26, r427;
}
{
cvt.rn.f16.f64 rs56, fd88;
}
mov.b32 r435, {rs56, rs56};
{
mul.f16x2 r433, r10, r435;
}
{
add.f16x2 r436, r437, r433;
}
{
cvt.rn.f16.f64 rs57, fd143;
}
mov.b32 r441, {rs57, rs57};
{
mul.f16x2 r439, r4, r441;
}
{
add.f16x2 r442, %27, r439;
}
{
cvt.rn.f16.f64 rs58, fd88;
}
mov.b32 r447, {rs58, rs58};
{
mul.f16x2 r445, r7, r447;
}
{
add.f16x2 r448, r449, r445;
}
{
cvt.rn.f16.f64 rs59, fd123;
}
mov.b32 r453, {rs59, rs59};
{
mul.f16x2 r451, r13, r453;
}
{
add.f16x2 r454, r430, r451;
}
{
cvt.rn.f16.f64 rs60, fd124;
}
mov.b32 r459, {rs60, rs60};
{
mul.f16x2 r457, r22, r459;
}
{
add.f16x2 r460, r436, r457;
}
{
cvt.rn.f16.f64 rs61, fd123;
}
mov.b32 r465, {rs61, rs61};
{
mul.f16x2 r463, r16, r465;
}
{
add.f16x2 r466, r442, r463;
}
{
cvt.rn.f16.f64 rs62, fd124;
}
mov.b32 r471, {rs62, rs62};
{
mul.f16x2 r469, r19, r471;
}
{
add.f16x2 r472, r448, r469;
}
{
cvt.rn.f16.f64 rs63, fd139;
}
mov.b32 r477, {rs63, rs63};
{
mul.f16x2 r475, r25, r477;
}
{
add.f16x2 r478, r454, r475;
}
mov.f64 fd60, 0dBFEDEBA72EF20147;
{
cvt.rn.f16.f64 rs64, fd60;
}
mov.b32 r483, {rs64, rs64};
{
mul.f16x2 r481, r34, r483;
}
{
add.f16x2 r484, r460, r481;
}
{
cvt.rn.f16.f64 rs65, fd139;
}
mov.b32 r489, {rs65, rs65};
{
mul.f16x2 r487, r28, r489;
}
{
add.f16x2 r490, r466, r487;
}
{
cvt.rn.f16.f64 rs66, fd60;
}
mov.b32 r495, {rs66, rs66};
{
mul.f16x2 r493, r31, r495;
}
{
add.f16x2 r496, r472, r493;
}
{
cvt.rn.f16.f64 rs67, fd127;
}
mov.b32 r501, {rs67, rs67};
{
mul.f16x2 r499, r37, r501;
}
{
add.f16x2 r502, r478, r499;
}
{
cvt.rn.f16.f64 rs68, fd128;
}
mov.b32 r507, {rs68, rs68};
{
mul.f16x2 r505, r46, r507;
}
{
add.f16x2 r508, r484, r505;
}
{
cvt.rn.f16.f64 rs69, fd127;
}
mov.b32 r513, {rs69, rs69};
{
mul.f16x2 r511, r40, r513;
}
{
add.f16x2 r514, r490, r511;
}
{
cvt.rn.f16.f64 rs70, fd128;
}
mov.b32 r519, {rs70, rs70};
{
mul.f16x2 r517, r43, r519;
}
{
add.f16x2 r520, r496, r517;
}
{
cvt.rn.f16.f64 rs71, fd135;
}
mov.b32 r525, {rs71, rs71};
{
mul.f16x2 r523, r49, r525;
}
{
add.f16x2 r526, r502, r523;
}
{
cvt.rn.f16.f64 rs72, fd108;
}
mov.b32 r531, {rs72, rs72};
{
mul.f16x2 r529, r58, r531;
}
{
add.f16x2 r532, r508, r529;
}
{
cvt.rn.f16.f64 rs73, fd135;
}
mov.b32 r537, {rs73, rs73};
{
mul.f16x2 r535, r52, r537;
}
{
add.f16x2 r538, r514, r535;
}
{
cvt.rn.f16.f64 rs74, fd108;
}
mov.b32 r543, {rs74, rs74};
{
mul.f16x2 r541, r55, r543;
}
{
add.f16x2 r544, r520, r541;
}
{
cvt.rn.f16.f64 rs75, fd131;
}
mov.b32 r549, {rs75, rs75};
{
mul.f16x2 r547, r61, r549;
}
{
add.f16x2 r550, r526, r547;
}
{
cvt.rn.f16.f64 rs76, fd132;
}
mov.b32 r555, {rs76, rs76};
{
mul.f16x2 r553, r70, r555;
}
{
add.f16x2 r556, r532, r553;
}
{
cvt.rn.f16.f64 rs77, fd131;
}
mov.b32 r561, {rs77, rs77};
{
mul.f16x2 r559, r64, r561;
}
{
add.f16x2 r562, r538, r559;
}
{
cvt.rn.f16.f64 rs78, fd132;
}
mov.b32 r567, {rs78, rs78};
{
mul.f16x2 r565, r67, r567;
}
{
add.f16x2 r568, r544, r565;
}
{
sub.f16x2 %6, r550, r556;
}
{
add.f16x2 %7, r562, r568;
}
{
add.f16x2 %20, r550, r556;
}
{
sub.f16x2 %21, r562, r568;
}
cvt.rn.f16.s32 rs79, r900;
mov.b32 r595, {rs79, rs79};
cvt.rn.f16.s32 rs80, r900;
mov.b32 r607, {rs80, rs80};
{
cvt.rn.f16.f64 rs81, fd139;
}
mov.b32 r587, {rs81, rs81};
{
mul.f16x2 r585, r1, r587;
}
{
add.f16x2 r588, %26, r585;
}
{
cvt.rn.f16.f64 rs82, fd140;
}
mov.b32 r593, {rs82, rs82};
{
mul.f16x2 r591, r10, r593;
}
{
add.f16x2 r594, r595, r591;
}
{
cvt.rn.f16.f64 rs83, fd139;
}
mov.b32 r599, {rs83, rs83};
{
mul.f16x2 r597, r4, r599;
}
{
add.f16x2 r600, %27, r597;
}
{
cvt.rn.f16.f64 rs84, fd140;
}
mov.b32 r605, {rs84, rs84};
{
mul.f16x2 r603, r7, r605;
}
{
add.f16x2 r606, r607, r603;
}
{
cvt.rn.f16.f64 rs85, fd131;
}
mov.b32 r611, {rs85, rs85};
{
mul.f16x2 r609, r13, r611;
}
{
add.f16x2 r612, r588, r609;
}
{
cvt.rn.f16.f64 rs86, fd80;
}
mov.b32 r617, {rs86, rs86};
{
mul.f16x2 r615, r22, r617;
}
{
add.f16x2 r618, r594, r615;
}
{
cvt.rn.f16.f64 rs87, fd131;
}
mov.b32 r623, {rs87, rs87};
{
mul.f16x2 r621, r16, r623;
}
{
add.f16x2 r624, r600, r621;
}
{
cvt.rn.f16.f64 rs88, fd80;
}
mov.b32 r629, {rs88, rs88};
{
mul.f16x2 r627, r19, r629;
}
{
add.f16x2 r630, r606, r627;
}
{
cvt.rn.f16.f64 rs89, fd127;
}
mov.b32 r635, {rs89, rs89};
{
mul.f16x2 r633, r25, r635;
}
{
add.f16x2 r636, r612, r633;
}
{
cvt.rn.f16.f64 rs90, fd128;
}
mov.b32 r641, {rs90, rs90};
{
mul.f16x2 r639, r34, r641;
}
{
add.f16x2 r642, r618, r639;
}
{
cvt.rn.f16.f64 rs91, fd127;
}
mov.b32 r647, {rs91, rs91};
{
mul.f16x2 r645, r28, r647;
}
{
add.f16x2 r648, r624, r645;
}
{
cvt.rn.f16.f64 rs92, fd128;
}
mov.b32 r653, {rs92, rs92};
{
mul.f16x2 r651, r31, r653;
}
{
add.f16x2 r654, r630, r651;
}
{
cvt.rn.f16.f64 rs93, fd143;
}
mov.b32 r659, {rs93, rs93};
{
mul.f16x2 r657, r37, r659;
}
{
add.f16x2 r660, r636, r657;
}
{
cvt.rn.f16.f64 rs94, fd88;
}
mov.b32 r665, {rs94, rs94};
{
mul.f16x2 r663, r46, r665;
}
{
add.f16x2 r666, r642, r663;
}
{
cvt.rn.f16.f64 rs95, fd143;
}
mov.b32 r671, {rs95, rs95};
{
mul.f16x2 r669, r40, r671;
}
{
add.f16x2 r672, r648, r669;
}
{
cvt.rn.f16.f64 rs96, fd88;
}
mov.b32 r677, {rs96, rs96};
{
mul.f16x2 r675, r43, r677;
}
{
add.f16x2 r678, r654, r675;
}
{
cvt.rn.f16.f64 rs97, fd123;
}
mov.b32 r683, {rs97, rs97};
{
mul.f16x2 r681, r49, r683;
}
{
add.f16x2 r684, r660, r681;
}
mov.f64 fd112, 0dBFCEA1E54BC48DBF;
{
cvt.rn.f16.f64 rs98, fd112;
}
mov.b32 r689, {rs98, rs98};
{
mul.f16x2 r687, r58, r689;
}
{
add.f16x2 r690, r666, r687;
}
{
cvt.rn.f16.f64 rs99, fd123;
}
mov.b32 r695, {rs99, rs99};
{
mul.f16x2 r693, r52, r695;
}
{
add.f16x2 r696, r672, r693;
}
{
cvt.rn.f16.f64 rs100, fd112;
}
mov.b32 r701, {rs100, rs100};
{
mul.f16x2 r699, r55, r701;
}
{
add.f16x2 r702, r678, r699;
}
{
cvt.rn.f16.f64 rs101, fd135;
}
mov.b32 r707, {rs101, rs101};
{
mul.f16x2 r705, r61, r707;
}
{
add.f16x2 r708, r684, r705;
}
mov.f64 fd136, 0dBFEA55E242A4C3D2;
{
cvt.rn.f16.f64 rs102, fd136;
}
mov.b32 r713, {rs102, rs102};
{
mul.f16x2 r711, r70, r713;
}
{
add.f16x2 r714, r690, r711;
}
{
cvt.rn.f16.f64 rs103, fd135;
}
mov.b32 r719, {rs103, rs103};
{
mul.f16x2 r717, r64, r719;
}
{
add.f16x2 r720, r696, r717;
}
{
cvt.rn.f16.f64 rs104, fd136;
}
mov.b32 r725, {rs104, rs104};
{
mul.f16x2 r723, r67, r725;
}
{
add.f16x2 r726, r702, r723;
}
{
sub.f16x2 %8, r708, r714;
}
{
add.f16x2 %9, r720, r726;
}
{
add.f16x2 %18, r708, r714;
}
{
sub.f16x2 %19, r720, r726;
}
cvt.rn.f16.s32 rs105, r900;
mov.b32 r753, {rs105, rs105};
cvt.rn.f16.s32 rs106, r900;
mov.b32 r765, {rs106, rs106};
{
cvt.rn.f16.f64 rs107, fd131;
}
mov.b32 r745, {rs107, rs107};
{
mul.f16x2 r743, r1, r745;
}
{
add.f16x2 r746, %26, r743;
}
{
cvt.rn.f16.f64 rs108, fd132;
}
mov.b32 r751, {rs108, rs108};
{
mul.f16x2 r749, r10, r751;
}
{
add.f16x2 r752, r753, r749;
}
{
cvt.rn.f16.f64 rs109, fd131;
}
mov.b32 r757, {rs109, rs109};
{
mul.f16x2 r755, r4, r757;
}
{
add.f16x2 r758, %27, r755;
}
{
cvt.rn.f16.f64 rs110, fd132;
}
mov.b32 r763, {rs110, rs110};
{
mul.f16x2 r761, r7, r763;
}
{
add.f16x2 r764, r765, r761;
}
{
cvt.rn.f16.f64 rs111, fd143;
}
mov.b32 r769, {rs111, rs111};
{
mul.f16x2 r767, r13, r769;
}
{
add.f16x2 r770, r746, r767;
}
{
cvt.rn.f16.f64 rs112, fd144;
}
mov.b32 r775, {rs112, rs112};
{
mul.f16x2 r773, r22, r775;
}
{
add.f16x2 r776, r752, r773;
}
{
cvt.rn.f16.f64 rs113, fd143;
}
mov.b32 r781, {rs113, rs113};
{
mul.f16x2 r779, r16, r781;
}
{
add.f16x2 r782, r758, r779;
}
{
cvt.rn.f16.f64 rs114, fd144;
}
mov.b32 r787, {rs114, rs114};
{
mul.f16x2 r785, r19, r787;
}
{
add.f16x2 r788, r764, r785;
}
{
cvt.rn.f16.f64 rs115, fd135;
}
mov.b32 r793, {rs115, rs115};
{
mul.f16x2 r791, r25, r793;
}
{
add.f16x2 r794, r770, r791;
}
{
cvt.rn.f16.f64 rs116, fd108;
}
mov.b32 r799, {rs116, rs116};
{
mul.f16x2 r797, r34, r799;
}
{
add.f16x2 r800, r776, r797;
}
{
cvt.rn.f16.f64 rs117, fd135;
}
mov.b32 r805, {rs117, rs117};
{
mul.f16x2 r803, r28, r805;
}
{
add.f16x2 r806, r782, r803;
}
{
cvt.rn.f16.f64 rs118, fd108;
}
mov.b32 r811, {rs118, rs118};
{
mul.f16x2 r809, r31, r811;
}
{
add.f16x2 r812, r788, r809;
}
{
cvt.rn.f16.f64 rs119, fd123;
}
mov.b32 r817, {rs119, rs119};
{
mul.f16x2 r815, r37, r817;
}
{
add.f16x2 r818, r794, r815;
}
{
cvt.rn.f16.f64 rs120, fd112;
}
mov.b32 r823, {rs120, rs120};
{
mul.f16x2 r821, r46, r823;
}
{
add.f16x2 r824, r800, r821;
}
{
cvt.rn.f16.f64 rs121, fd123;
}
mov.b32 r829, {rs121, rs121};
{
mul.f16x2 r827, r40, r829;
}
{
add.f16x2 r830, r806, r827;
}
{
cvt.rn.f16.f64 rs122, fd112;
}
mov.b32 r835, {rs122, rs122};
{
mul.f16x2 r833, r43, r835;
}
{
add.f16x2 r836, r812, r833;
}
{
cvt.rn.f16.f64 rs123, fd127;
}
mov.b32 r841, {rs123, rs123};
{
mul.f16x2 r839, r49, r841;
}
{
add.f16x2 r842, r818, r839;
}
{
cvt.rn.f16.f64 rs124, fd128;
}
mov.b32 r847, {rs124, rs124};
{
mul.f16x2 r845, r58, r847;
}
{
add.f16x2 r848, r824, r845;
}
{
cvt.rn.f16.f64 rs125, fd127;
}
mov.b32 r853, {rs125, rs125};
{
mul.f16x2 r851, r52, r853;
}
{
add.f16x2 r854, r830, r851;
}
{
cvt.rn.f16.f64 rs126, fd128;
}
mov.b32 r859, {rs126, rs126};
{
mul.f16x2 r857, r55, r859;
}
{
add.f16x2 r860, r836, r857;
}
{
cvt.rn.f16.f64 rs127, fd139;
}
mov.b32 r865, {rs127, rs127};
{
mul.f16x2 r863, r61, r865;
}
{
add.f16x2 r866, r842, r863;
}
{
cvt.rn.f16.f64 rs128, fd140;
}
mov.b32 r871, {rs128, rs128};
{
mul.f16x2 r869, r70, r871;
}
{
add.f16x2 r872, r848, r869;
}
{
cvt.rn.f16.f64 rs129, fd139;
}
mov.b32 r877, {rs129, rs129};
{
mul.f16x2 r875, r64, r877;
}
{
add.f16x2 r878, r854, r875;
}
{
cvt.rn.f16.f64 rs130, fd140;
}
mov.b32 r883, {rs130, rs130};
{
mul.f16x2 r881, r67, r883;
}
{
add.f16x2 r884, r860, r881;
}
{
sub.f16x2 %10, r866, r872;
}
{
add.f16x2 %11, r878, r884;
}
{
add.f16x2 %16, r866, r872;
}
{
sub.f16x2 %17, r878, r884;
}
cvt.rn.f16.s32 rs131, r900;
mov.b32 r911, {rs131, rs131};
cvt.rn.f16.s32 rs132, r900;
mov.b32 r923, {rs132, rs132};
{
cvt.rn.f16.f64 rs133, fd123;
}
mov.b32 r903, {rs133, rs133};
{
mul.f16x2 r901, r1, r903;
}
{
add.f16x2 r904, %26, r901;
}
{
cvt.rn.f16.f64 rs134, fd124;
}
mov.b32 r909, {rs134, rs134};
{
mul.f16x2 r907, r10, r909;
}
{
add.f16x2 r910, r911, r907;
}
{
cvt.rn.f16.f64 rs135, fd123;
}
mov.b32 r915, {rs135, rs135};
{
mul.f16x2 r913, r4, r915;
}
{
add.f16x2 r916, %27, r913;
}
{
cvt.rn.f16.f64 rs136, fd124;
}
mov.b32 r921, {rs136, rs136};
{
mul.f16x2 r919, r7, r921;
}
{
add.f16x2 r922, r923, r919;
}
{
cvt.rn.f16.f64 rs137, fd127;
}
mov.b32 r927, {rs137, rs137};
{
mul.f16x2 r925, r13, r927;
}
{
add.f16x2 r928, r904, r925;
}
{
cvt.rn.f16.f64 rs138, fd128;
}
mov.b32 r933, {rs138, rs138};
{
mul.f16x2 r931, r22, r933;
}
{
add.f16x2 r934, r910, r931;
}
{
cvt.rn.f16.f64 rs139, fd127;
}
mov.b32 r939, {rs139, rs139};
{
mul.f16x2 r937, r16, r939;
}
{
add.f16x2 r940, r916, r937;
}
{
cvt.rn.f16.f64 rs140, fd128;
}
mov.b32 r945, {rs140, rs140};
{
mul.f16x2 r943, r19, r945;
}
{
add.f16x2 r946, r922, r943;
}
{
cvt.rn.f16.f64 rs141, fd131;
}
mov.b32 r951, {rs141, rs141};
{
mul.f16x2 r949, r25, r951;
}
{
add.f16x2 r952, r928, r949;
}
{
cvt.rn.f16.f64 rs142, fd132;
}
mov.b32 r957, {rs142, rs142};
{
mul.f16x2 r955, r34, r957;
}
{
add.f16x2 r958, r934, r955;
}
{
cvt.rn.f16.f64 rs143, fd131;
}
mov.b32 r963, {rs143, rs143};
{
mul.f16x2 r961, r28, r963;
}
{
add.f16x2 r964, r940, r961;
}
{
cvt.rn.f16.f64 rs144, fd132;
}
mov.b32 r969, {rs144, rs144};
{
mul.f16x2 r967, r31, r969;
}
{
add.f16x2 r970, r946, r967;
}
{
cvt.rn.f16.f64 rs145, fd135;
}
mov.b32 r975, {rs145, rs145};
{
mul.f16x2 r973, r37, r975;
}
{
add.f16x2 r976, r952, r973;
}
{
cvt.rn.f16.f64 rs146, fd136;
}
mov.b32 r981, {rs146, rs146};
{
mul.f16x2 r979, r46, r981;
}
{
add.f16x2 r982, r958, r979;
}
{
cvt.rn.f16.f64 rs147, fd135;
}
mov.b32 r987, {rs147, rs147};
{
mul.f16x2 r985, r40, r987;
}
{
add.f16x2 r988, r964, r985;
}
{
cvt.rn.f16.f64 rs148, fd136;
}
mov.b32 r993, {rs148, rs148};
{
mul.f16x2 r991, r43, r993;
}
{
add.f16x2 r994, r970, r991;
}
{
cvt.rn.f16.f64 rs149, fd139;
}
mov.b32 r999, {rs149, rs149};
{
mul.f16x2 r997, r49, r999;
}
{
add.f16x2 r1000, r976, r997;
}
{
cvt.rn.f16.f64 rs150, fd140;
}
mov.b32 r1005, {rs150, rs150};
{
mul.f16x2 r1003, r58, r1005;
}
{
add.f16x2 r1006, r982, r1003;
}
{
cvt.rn.f16.f64 rs151, fd139;
}
mov.b32 r1011, {rs151, rs151};
{
mul.f16x2 r1009, r52, r1011;
}
{
add.f16x2 r1012, r988, r1009;
}
{
cvt.rn.f16.f64 rs152, fd140;
}
mov.b32 r1017, {rs152, rs152};
{
mul.f16x2 r1015, r55, r1017;
}
{
add.f16x2 r1018, r994, r1015;
}
{
cvt.rn.f16.f64 rs153, fd143;
}
mov.b32 r1023, {rs153, rs153};
{
mul.f16x2 r1021, r61, r1023;
}
{
add.f16x2 r1024, r1000, r1021;
}
{
cvt.rn.f16.f64 rs154, fd144;
}
mov.b32 r1029, {rs154, rs154};
{
mul.f16x2 r1027, r70, r1029;
}
{
add.f16x2 r1030, r1006, r1027;
}
{
cvt.rn.f16.f64 rs155, fd143;
}
mov.b32 r1035, {rs155, rs155};
{
mul.f16x2 r1033, r64, r1035;
}
{
add.f16x2 r1036, r1012, r1033;
}
{
cvt.rn.f16.f64 rs156, fd144;
}
mov.b32 r1041, {rs156, rs156};
{
mul.f16x2 r1039, r67, r1041;
}
{
add.f16x2 r1042, r1018, r1039;
}
{
sub.f16x2 %12, r1024, r1030;
}
{
add.f16x2 %13, r1036, r1042;
}
{
add.f16x2 %14, r1024, r1030;
}
{
sub.f16x2 %15, r1036, r1042;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)));
};


#endif
