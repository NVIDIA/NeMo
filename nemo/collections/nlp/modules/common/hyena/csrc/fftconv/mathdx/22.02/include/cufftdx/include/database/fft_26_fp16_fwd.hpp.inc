#ifndef CUFFTDX_FFT_26_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_26_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<757, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<363>;
.reg .b32 r<2487>;
.reg .f64 fd<339>;
.reg .b64 rd<2>;
{
add.f16x2 r1, %54, %76;
}
{
add.f16x2 r4, %55, %77;
}
{
sub.f16x2 r7, %54, %76;
}
{
sub.f16x2 r10, %55, %77;
}
{
add.f16x2 r13, %56, %74;
}
{
add.f16x2 r16, %57, %75;
}
{
sub.f16x2 r19, %56, %74;
}
{
sub.f16x2 r22, %57, %75;
}
{
add.f16x2 r25, %58, %72;
}
{
add.f16x2 r28, %59, %73;
}
{
sub.f16x2 r31, %58, %72;
}
{
sub.f16x2 r34, %59, %73;
}
{
add.f16x2 r37, %60, %70;
}
{
add.f16x2 r40, %61, %71;
}
{
sub.f16x2 r43, %60, %70;
}
{
sub.f16x2 r46, %61, %71;
}
{
add.f16x2 r49, %62, %68;
}
{
add.f16x2 r52, %63, %69;
}
{
sub.f16x2 r55, %62, %68;
}
{
sub.f16x2 r58, %63, %69;
}
{
add.f16x2 r61, %64, %66;
}
{
add.f16x2 r64, %65, %67;
}
{
sub.f16x2 r67, %64, %66;
}
{
sub.f16x2 r70, %65, %67;
}
{
add.f16x2 r73, %52, r1;
}
{
add.f16x2 r76, %53, r4;
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
add.f16x2 r103, r97, r61;
}
{
add.f16x2 r106, r100, r64;
}
mov.u32 r1956, 0;
cvt.rn.f16.s32 rs1, r1956;
mov.b32 r121, {rs1, rs1};
cvt.rn.f16.s32 rs2, r1956;
mov.b32 r133, {rs2, rs2};
mov.f64 fd291, 0d3FEC55A7E00740E9;
{
cvt.rn.f16.f64 rs3, fd291;
}
mov.b32 r113, {rs3, rs3};
{
mul.f16x2 r111, r1, r113;
}
{
add.f16x2 r114, %52, r111;
}
mov.f64 fd310, 0dBFDDBE064267C47C;
{
cvt.rn.f16.f64 rs4, fd310;
}
mov.b32 r119, {rs4, rs4};
{
mul.f16x2 r117, r10, r119;
}
{
add.f16x2 r120, r121, r117;
}
{
cvt.rn.f16.f64 rs5, fd291;
}
mov.b32 r125, {rs5, rs5};
{
mul.f16x2 r123, r4, r125;
}
{
add.f16x2 r126, %53, r123;
}
{
cvt.rn.f16.f64 rs6, fd310;
}
mov.b32 r131, {rs6, rs6};
{
mul.f16x2 r129, r7, r131;
}
{
add.f16x2 r132, r133, r129;
}
mov.f64 fd295, 0d3FE22D961EA71119;
{
cvt.rn.f16.f64 rs7, fd295;
}
mov.b32 r137, {rs7, rs7};
{
mul.f16x2 r135, r13, r137;
}
{
add.f16x2 r138, r114, r135;
}
mov.f64 fd306, 0dBFEA55E242A4C3D2;
{
cvt.rn.f16.f64 rs8, fd306;
}
mov.b32 r143, {rs8, rs8};
{
mul.f16x2 r141, r22, r143;
}
{
add.f16x2 r144, r120, r141;
}
{
cvt.rn.f16.f64 rs9, fd295;
}
mov.b32 r149, {rs9, rs9};
{
mul.f16x2 r147, r16, r149;
}
{
add.f16x2 r150, r126, r147;
}
{
cvt.rn.f16.f64 rs10, fd306;
}
mov.b32 r155, {rs10, rs10};
{
mul.f16x2 r153, r19, r155;
}
{
add.f16x2 r156, r132, r153;
}
mov.f64 fd299, 0d3FBEDB7DEBAA3ED8;
{
cvt.rn.f16.f64 rs11, fd299;
}
mov.b32 r161, {rs11, rs11};
{
mul.f16x2 r159, r25, r161;
}
{
add.f16x2 r162, r138, r159;
}
mov.f64 fd302, 0dBFEFC44566966769;
{
cvt.rn.f16.f64 rs12, fd302;
}
mov.b32 r167, {rs12, rs12};
{
mul.f16x2 r165, r34, r167;
}
{
add.f16x2 r168, r144, r165;
}
{
cvt.rn.f16.f64 rs13, fd299;
}
mov.b32 r173, {rs13, rs13};
{
mul.f16x2 r171, r28, r173;
}
{
add.f16x2 r174, r150, r171;
}
{
cvt.rn.f16.f64 rs14, fd302;
}
mov.b32 r179, {rs14, rs14};
{
mul.f16x2 r177, r31, r179;
}
{
add.f16x2 r180, r156, r177;
}
mov.f64 fd303, 0dBFD6B1D8B2365DA1;
{
cvt.rn.f16.f64 rs15, fd303;
}
mov.b32 r185, {rs15, rs15};
{
mul.f16x2 r183, r37, r185;
}
{
add.f16x2 r186, r162, r183;
}
mov.f64 fd304, 0dBFEDEBA72EF20147;
{
cvt.rn.f16.f64 rs16, fd304;
}
mov.b32 r191, {rs16, rs16};
{
mul.f16x2 r189, r46, r191;
}
{
add.f16x2 r192, r168, r189;
}
{
cvt.rn.f16.f64 rs17, fd303;
}
mov.b32 r197, {rs17, rs17};
{
mul.f16x2 r195, r40, r197;
}
{
add.f16x2 r198, r174, r195;
}
{
cvt.rn.f16.f64 rs18, fd304;
}
mov.b32 r203, {rs18, rs18};
{
mul.f16x2 r201, r43, r203;
}
{
add.f16x2 r204, r180, r201;
}
mov.f64 fd307, 0dBFE7F3CCD0032E0C;
{
cvt.rn.f16.f64 rs19, fd307;
}
mov.b32 r209, {rs19, rs19};
{
mul.f16x2 r207, r49, r209;
}
{
add.f16x2 r210, r186, r207;
}
mov.f64 fd308, 0dBFE5384D024C2F84;
{
cvt.rn.f16.f64 rs20, fd308;
}
mov.b32 r215, {rs20, rs20};
{
mul.f16x2 r213, r58, r215;
}
{
add.f16x2 r216, r192, r213;
}
{
cvt.rn.f16.f64 rs21, fd307;
}
mov.b32 r221, {rs21, rs21};
{
mul.f16x2 r219, r52, r221;
}
{
add.f16x2 r222, r198, r219;
}
{
cvt.rn.f16.f64 rs22, fd308;
}
mov.b32 r227, {rs22, rs22};
{
mul.f16x2 r225, r55, r227;
}
{
add.f16x2 r228, r204, r225;
}
mov.f64 fd311, 0dBFEF11F493053D00;
{
cvt.rn.f16.f64 rs23, fd311;
}
mov.b32 r233, {rs23, rs23};
{
mul.f16x2 r231, r61, r233;
}
{
add.f16x2 r234, r210, r231;
}
mov.f64 fd312, 0dBFCEA1E54BC48DBF;
{
cvt.rn.f16.f64 rs24, fd312;
}
mov.b32 r239, {rs24, rs24};
{
mul.f16x2 r237, r70, r239;
}
{
add.f16x2 r240, r216, r237;
}
{
cvt.rn.f16.f64 rs25, fd311;
}
mov.b32 r245, {rs25, rs25};
{
mul.f16x2 r243, r64, r245;
}
{
add.f16x2 r246, r222, r243;
}
{
cvt.rn.f16.f64 rs26, fd312;
}
mov.b32 r251, {rs26, rs26};
{
mul.f16x2 r249, r67, r251;
}
{
add.f16x2 r252, r228, r249;
}
{
sub.f16x2 r255, r234, r240;
}
{
add.f16x2 r258, r246, r252;
}
{
add.f16x2 r261, r234, r240;
}
{
sub.f16x2 r264, r246, r252;
}
cvt.rn.f16.s32 rs27, r1956;
mov.b32 r279, {rs27, rs27};
cvt.rn.f16.s32 rs28, r1956;
mov.b32 r291, {rs28, rs28};
{
cvt.rn.f16.f64 rs29, fd295;
}
mov.b32 r271, {rs29, rs29};
{
mul.f16x2 r269, r1, r271;
}
{
add.f16x2 r272, %52, r269;
}
{
cvt.rn.f16.f64 rs30, fd306;
}
mov.b32 r277, {rs30, rs30};
{
mul.f16x2 r275, r10, r277;
}
{
add.f16x2 r278, r279, r275;
}
{
cvt.rn.f16.f64 rs31, fd295;
}
mov.b32 r283, {rs31, rs31};
{
mul.f16x2 r281, r4, r283;
}
{
add.f16x2 r284, %53, r281;
}
{
cvt.rn.f16.f64 rs32, fd306;
}
mov.b32 r289, {rs32, rs32};
{
mul.f16x2 r287, r7, r289;
}
{
add.f16x2 r290, r291, r287;
}
{
cvt.rn.f16.f64 rs33, fd303;
}
mov.b32 r295, {rs33, rs33};
{
mul.f16x2 r293, r13, r295;
}
{
add.f16x2 r296, r272, r293;
}
{
cvt.rn.f16.f64 rs34, fd304;
}
mov.b32 r301, {rs34, rs34};
{
mul.f16x2 r299, r22, r301;
}
{
add.f16x2 r302, r278, r299;
}
{
cvt.rn.f16.f64 rs35, fd303;
}
mov.b32 r307, {rs35, rs35};
{
mul.f16x2 r305, r16, r307;
}
{
add.f16x2 r308, r284, r305;
}
{
cvt.rn.f16.f64 rs36, fd304;
}
mov.b32 r313, {rs36, rs36};
{
mul.f16x2 r311, r19, r313;
}
{
add.f16x2 r314, r290, r311;
}
{
cvt.rn.f16.f64 rs37, fd311;
}
mov.b32 r319, {rs37, rs37};
{
mul.f16x2 r317, r25, r319;
}
{
add.f16x2 r320, r296, r317;
}
{
cvt.rn.f16.f64 rs38, fd312;
}
mov.b32 r325, {rs38, rs38};
{
mul.f16x2 r323, r34, r325;
}
{
add.f16x2 r326, r302, r323;
}
{
cvt.rn.f16.f64 rs39, fd311;
}
mov.b32 r331, {rs39, rs39};
{
mul.f16x2 r329, r28, r331;
}
{
add.f16x2 r332, r308, r329;
}
{
cvt.rn.f16.f64 rs40, fd312;
}
mov.b32 r337, {rs40, rs40};
{
mul.f16x2 r335, r31, r337;
}
{
add.f16x2 r338, r314, r335;
}
{
cvt.rn.f16.f64 rs41, fd307;
}
mov.b32 r343, {rs41, rs41};
{
mul.f16x2 r341, r37, r343;
}
{
add.f16x2 r344, r320, r341;
}
mov.f64 fd224, 0d3FE5384D024C2F84;
{
cvt.rn.f16.f64 rs42, fd224;
}
mov.b32 r349, {rs42, rs42};
{
mul.f16x2 r347, r46, r349;
}
{
add.f16x2 r350, r326, r347;
}
{
cvt.rn.f16.f64 rs43, fd307;
}
mov.b32 r355, {rs43, rs43};
{
mul.f16x2 r353, r40, r355;
}
{
add.f16x2 r356, r332, r353;
}
{
cvt.rn.f16.f64 rs44, fd224;
}
mov.b32 r361, {rs44, rs44};
{
mul.f16x2 r359, r43, r361;
}
{
add.f16x2 r362, r338, r359;
}
{
cvt.rn.f16.f64 rs45, fd299;
}
mov.b32 r367, {rs45, rs45};
{
mul.f16x2 r365, r49, r367;
}
{
add.f16x2 r368, r344, r365;
}
mov.f64 fd288, 0d3FEFC44566966769;
{
cvt.rn.f16.f64 rs46, fd288;
}
mov.b32 r373, {rs46, rs46};
{
mul.f16x2 r371, r58, r373;
}
{
add.f16x2 r374, r350, r371;
}
{
cvt.rn.f16.f64 rs47, fd299;
}
mov.b32 r379, {rs47, rs47};
{
mul.f16x2 r377, r52, r379;
}
{
add.f16x2 r380, r356, r377;
}
{
cvt.rn.f16.f64 rs48, fd288;
}
mov.b32 r385, {rs48, rs48};
{
mul.f16x2 r383, r55, r385;
}
{
add.f16x2 r386, r362, r383;
}
{
cvt.rn.f16.f64 rs49, fd291;
}
mov.b32 r391, {rs49, rs49};
{
mul.f16x2 r389, r61, r391;
}
{
add.f16x2 r392, r368, r389;
}
mov.f64 fd272, 0d3FDDBE064267C47C;
{
cvt.rn.f16.f64 rs50, fd272;
}
mov.b32 r397, {rs50, rs50};
{
mul.f16x2 r395, r70, r397;
}
{
add.f16x2 r398, r374, r395;
}
{
cvt.rn.f16.f64 rs51, fd291;
}
mov.b32 r403, {rs51, rs51};
{
mul.f16x2 r401, r64, r403;
}
{
add.f16x2 r404, r380, r401;
}
{
cvt.rn.f16.f64 rs52, fd272;
}
mov.b32 r409, {rs52, rs52};
{
mul.f16x2 r407, r67, r409;
}
{
add.f16x2 r410, r386, r407;
}
{
sub.f16x2 r413, r392, r398;
}
{
add.f16x2 r416, r404, r410;
}
{
add.f16x2 r419, r392, r398;
}
{
sub.f16x2 r422, r404, r410;
}
cvt.rn.f16.s32 rs53, r1956;
mov.b32 r437, {rs53, rs53};
cvt.rn.f16.s32 rs54, r1956;
mov.b32 r449, {rs54, rs54};
{
cvt.rn.f16.f64 rs55, fd299;
}
mov.b32 r429, {rs55, rs55};
{
mul.f16x2 r427, r1, r429;
}
{
add.f16x2 r430, %52, r427;
}
{
cvt.rn.f16.f64 rs56, fd302;
}
mov.b32 r435, {rs56, rs56};
{
mul.f16x2 r433, r10, r435;
}
{
add.f16x2 r436, r437, r433;
}
{
cvt.rn.f16.f64 rs57, fd299;
}
mov.b32 r441, {rs57, rs57};
{
mul.f16x2 r439, r4, r441;
}
{
add.f16x2 r442, %53, r439;
}
{
cvt.rn.f16.f64 rs58, fd302;
}
mov.b32 r447, {rs58, rs58};
{
mul.f16x2 r445, r7, r447;
}
{
add.f16x2 r448, r449, r445;
}
{
cvt.rn.f16.f64 rs59, fd311;
}
mov.b32 r453, {rs59, rs59};
{
mul.f16x2 r451, r13, r453;
}
{
add.f16x2 r454, r430, r451;
}
{
cvt.rn.f16.f64 rs60, fd312;
}
mov.b32 r459, {rs60, rs60};
{
mul.f16x2 r457, r22, r459;
}
{
add.f16x2 r460, r436, r457;
}
{
cvt.rn.f16.f64 rs61, fd311;
}
mov.b32 r465, {rs61, rs61};
{
mul.f16x2 r463, r16, r465;
}
{
add.f16x2 r466, r442, r463;
}
{
cvt.rn.f16.f64 rs62, fd312;
}
mov.b32 r471, {rs62, rs62};
{
mul.f16x2 r469, r19, r471;
}
{
add.f16x2 r472, r448, r469;
}
{
cvt.rn.f16.f64 rs63, fd303;
}
mov.b32 r477, {rs63, rs63};
{
mul.f16x2 r475, r25, r477;
}
{
add.f16x2 r478, r454, r475;
}
mov.f64 fd204, 0d3FEDEBA72EF20147;
{
cvt.rn.f16.f64 rs64, fd204;
}
mov.b32 r483, {rs64, rs64};
{
mul.f16x2 r481, r34, r483;
}
{
add.f16x2 r484, r460, r481;
}
{
cvt.rn.f16.f64 rs65, fd303;
}
mov.b32 r489, {rs65, rs65};
{
mul.f16x2 r487, r28, r489;
}
{
add.f16x2 r490, r466, r487;
}
{
cvt.rn.f16.f64 rs66, fd204;
}
mov.b32 r495, {rs66, rs66};
{
mul.f16x2 r493, r31, r495;
}
{
add.f16x2 r496, r472, r493;
}
{
cvt.rn.f16.f64 rs67, fd291;
}
mov.b32 r501, {rs67, rs67};
{
mul.f16x2 r499, r37, r501;
}
{
add.f16x2 r502, r478, r499;
}
{
cvt.rn.f16.f64 rs68, fd272;
}
mov.b32 r507, {rs68, rs68};
{
mul.f16x2 r505, r46, r507;
}
{
add.f16x2 r508, r484, r505;
}
{
cvt.rn.f16.f64 rs69, fd291;
}
mov.b32 r513, {rs69, rs69};
{
mul.f16x2 r511, r40, r513;
}
{
add.f16x2 r514, r490, r511;
}
{
cvt.rn.f16.f64 rs70, fd272;
}
mov.b32 r519, {rs70, rs70};
{
mul.f16x2 r517, r43, r519;
}
{
add.f16x2 r520, r496, r517;
}
{
cvt.rn.f16.f64 rs71, fd295;
}
mov.b32 r525, {rs71, rs71};
{
mul.f16x2 r523, r49, r525;
}
{
add.f16x2 r526, r502, r523;
}
{
cvt.rn.f16.f64 rs72, fd306;
}
mov.b32 r531, {rs72, rs72};
{
mul.f16x2 r529, r58, r531;
}
{
add.f16x2 r532, r508, r529;
}
{
cvt.rn.f16.f64 rs73, fd295;
}
mov.b32 r537, {rs73, rs73};
{
mul.f16x2 r535, r52, r537;
}
{
add.f16x2 r538, r514, r535;
}
{
cvt.rn.f16.f64 rs74, fd306;
}
mov.b32 r543, {rs74, rs74};
{
mul.f16x2 r541, r55, r543;
}
{
add.f16x2 r544, r520, r541;
}
{
cvt.rn.f16.f64 rs75, fd307;
}
mov.b32 r549, {rs75, rs75};
{
mul.f16x2 r547, r61, r549;
}
{
add.f16x2 r550, r526, r547;
}
{
cvt.rn.f16.f64 rs76, fd308;
}
mov.b32 r555, {rs76, rs76};
{
mul.f16x2 r553, r70, r555;
}
{
add.f16x2 r556, r532, r553;
}
{
cvt.rn.f16.f64 rs77, fd307;
}
mov.b32 r561, {rs77, rs77};
{
mul.f16x2 r559, r64, r561;
}
{
add.f16x2 r562, r538, r559;
}
{
cvt.rn.f16.f64 rs78, fd308;
}
mov.b32 r567, {rs78, rs78};
{
mul.f16x2 r565, r67, r567;
}
{
add.f16x2 r568, r544, r565;
}
{
sub.f16x2 r571, r550, r556;
}
{
add.f16x2 r574, r562, r568;
}
{
add.f16x2 r577, r550, r556;
}
{
sub.f16x2 r580, r562, r568;
}
cvt.rn.f16.s32 rs79, r1956;
mov.b32 r595, {rs79, rs79};
cvt.rn.f16.s32 rs80, r1956;
mov.b32 r607, {rs80, rs80};
{
cvt.rn.f16.f64 rs81, fd303;
}
mov.b32 r587, {rs81, rs81};
{
mul.f16x2 r585, r1, r587;
}
{
add.f16x2 r588, %52, r585;
}
{
cvt.rn.f16.f64 rs82, fd304;
}
mov.b32 r593, {rs82, rs82};
{
mul.f16x2 r591, r10, r593;
}
{
add.f16x2 r594, r595, r591;
}
{
cvt.rn.f16.f64 rs83, fd303;
}
mov.b32 r599, {rs83, rs83};
{
mul.f16x2 r597, r4, r599;
}
{
add.f16x2 r600, %53, r597;
}
{
cvt.rn.f16.f64 rs84, fd304;
}
mov.b32 r605, {rs84, rs84};
{
mul.f16x2 r603, r7, r605;
}
{
add.f16x2 r606, r607, r603;
}
{
cvt.rn.f16.f64 rs85, fd307;
}
mov.b32 r611, {rs85, rs85};
{
mul.f16x2 r609, r13, r611;
}
{
add.f16x2 r612, r588, r609;
}
{
cvt.rn.f16.f64 rs86, fd224;
}
mov.b32 r617, {rs86, rs86};
{
mul.f16x2 r615, r22, r617;
}
{
add.f16x2 r618, r594, r615;
}
{
cvt.rn.f16.f64 rs87, fd307;
}
mov.b32 r623, {rs87, rs87};
{
mul.f16x2 r621, r16, r623;
}
{
add.f16x2 r624, r600, r621;
}
{
cvt.rn.f16.f64 rs88, fd224;
}
mov.b32 r629, {rs88, rs88};
{
mul.f16x2 r627, r19, r629;
}
{
add.f16x2 r630, r606, r627;
}
{
cvt.rn.f16.f64 rs89, fd291;
}
mov.b32 r635, {rs89, rs89};
{
mul.f16x2 r633, r25, r635;
}
{
add.f16x2 r636, r612, r633;
}
{
cvt.rn.f16.f64 rs90, fd272;
}
mov.b32 r641, {rs90, rs90};
{
mul.f16x2 r639, r34, r641;
}
{
add.f16x2 r642, r618, r639;
}
{
cvt.rn.f16.f64 rs91, fd291;
}
mov.b32 r647, {rs91, rs91};
{
mul.f16x2 r645, r28, r647;
}
{
add.f16x2 r648, r624, r645;
}
{
cvt.rn.f16.f64 rs92, fd272;
}
mov.b32 r653, {rs92, rs92};
{
mul.f16x2 r651, r31, r653;
}
{
add.f16x2 r654, r630, r651;
}
{
cvt.rn.f16.f64 rs93, fd299;
}
mov.b32 r659, {rs93, rs93};
{
mul.f16x2 r657, r37, r659;
}
{
add.f16x2 r660, r636, r657;
}
{
cvt.rn.f16.f64 rs94, fd302;
}
mov.b32 r665, {rs94, rs94};
{
mul.f16x2 r663, r46, r665;
}
{
add.f16x2 r666, r642, r663;
}
{
cvt.rn.f16.f64 rs95, fd299;
}
mov.b32 r671, {rs95, rs95};
{
mul.f16x2 r669, r40, r671;
}
{
add.f16x2 r672, r648, r669;
}
{
cvt.rn.f16.f64 rs96, fd302;
}
mov.b32 r677, {rs96, rs96};
{
mul.f16x2 r675, r43, r677;
}
{
add.f16x2 r678, r654, r675;
}
{
cvt.rn.f16.f64 rs97, fd311;
}
mov.b32 r683, {rs97, rs97};
{
mul.f16x2 r681, r49, r683;
}
{
add.f16x2 r684, r660, r681;
}
mov.f64 fd256, 0d3FCEA1E54BC48DBF;
{
cvt.rn.f16.f64 rs98, fd256;
}
mov.b32 r689, {rs98, rs98};
{
mul.f16x2 r687, r58, r689;
}
{
add.f16x2 r690, r666, r687;
}
{
cvt.rn.f16.f64 rs99, fd311;
}
mov.b32 r695, {rs99, rs99};
{
mul.f16x2 r693, r52, r695;
}
{
add.f16x2 r696, r672, r693;
}
{
cvt.rn.f16.f64 rs100, fd256;
}
mov.b32 r701, {rs100, rs100};
{
mul.f16x2 r699, r55, r701;
}
{
add.f16x2 r702, r678, r699;
}
{
cvt.rn.f16.f64 rs101, fd295;
}
mov.b32 r707, {rs101, rs101};
{
mul.f16x2 r705, r61, r707;
}
{
add.f16x2 r708, r684, r705;
}
mov.f64 fd280, 0d3FEA55E242A4C3D2;
{
cvt.rn.f16.f64 rs102, fd280;
}
mov.b32 r713, {rs102, rs102};
{
mul.f16x2 r711, r70, r713;
}
{
add.f16x2 r714, r690, r711;
}
{
cvt.rn.f16.f64 rs103, fd295;
}
mov.b32 r719, {rs103, rs103};
{
mul.f16x2 r717, r64, r719;
}
{
add.f16x2 r720, r696, r717;
}
{
cvt.rn.f16.f64 rs104, fd280;
}
mov.b32 r725, {rs104, rs104};
{
mul.f16x2 r723, r67, r725;
}
{
add.f16x2 r726, r702, r723;
}
{
sub.f16x2 r729, r708, r714;
}
{
add.f16x2 r732, r720, r726;
}
{
add.f16x2 r735, r708, r714;
}
{
sub.f16x2 r738, r720, r726;
}
cvt.rn.f16.s32 rs105, r1956;
mov.b32 r753, {rs105, rs105};
cvt.rn.f16.s32 rs106, r1956;
mov.b32 r765, {rs106, rs106};
{
cvt.rn.f16.f64 rs107, fd307;
}
mov.b32 r745, {rs107, rs107};
{
mul.f16x2 r743, r1, r745;
}
{
add.f16x2 r746, %52, r743;
}
{
cvt.rn.f16.f64 rs108, fd308;
}
mov.b32 r751, {rs108, rs108};
{
mul.f16x2 r749, r10, r751;
}
{
add.f16x2 r752, r753, r749;
}
{
cvt.rn.f16.f64 rs109, fd307;
}
mov.b32 r757, {rs109, rs109};
{
mul.f16x2 r755, r4, r757;
}
{
add.f16x2 r758, %53, r755;
}
{
cvt.rn.f16.f64 rs110, fd308;
}
mov.b32 r763, {rs110, rs110};
{
mul.f16x2 r761, r7, r763;
}
{
add.f16x2 r764, r765, r761;
}
{
cvt.rn.f16.f64 rs111, fd299;
}
mov.b32 r769, {rs111, rs111};
{
mul.f16x2 r767, r13, r769;
}
{
add.f16x2 r770, r746, r767;
}
{
cvt.rn.f16.f64 rs112, fd288;
}
mov.b32 r775, {rs112, rs112};
{
mul.f16x2 r773, r22, r775;
}
{
add.f16x2 r776, r752, r773;
}
{
cvt.rn.f16.f64 rs113, fd299;
}
mov.b32 r781, {rs113, rs113};
{
mul.f16x2 r779, r16, r781;
}
{
add.f16x2 r782, r758, r779;
}
{
cvt.rn.f16.f64 rs114, fd288;
}
mov.b32 r787, {rs114, rs114};
{
mul.f16x2 r785, r19, r787;
}
{
add.f16x2 r788, r764, r785;
}
{
cvt.rn.f16.f64 rs115, fd295;
}
mov.b32 r793, {rs115, rs115};
{
mul.f16x2 r791, r25, r793;
}
{
add.f16x2 r794, r770, r791;
}
{
cvt.rn.f16.f64 rs116, fd306;
}
mov.b32 r799, {rs116, rs116};
{
mul.f16x2 r797, r34, r799;
}
{
add.f16x2 r800, r776, r797;
}
{
cvt.rn.f16.f64 rs117, fd295;
}
mov.b32 r805, {rs117, rs117};
{
mul.f16x2 r803, r28, r805;
}
{
add.f16x2 r806, r782, r803;
}
{
cvt.rn.f16.f64 rs118, fd306;
}
mov.b32 r811, {rs118, rs118};
{
mul.f16x2 r809, r31, r811;
}
{
add.f16x2 r812, r788, r809;
}
{
cvt.rn.f16.f64 rs119, fd311;
}
mov.b32 r817, {rs119, rs119};
{
mul.f16x2 r815, r37, r817;
}
{
add.f16x2 r818, r794, r815;
}
{
cvt.rn.f16.f64 rs120, fd256;
}
mov.b32 r823, {rs120, rs120};
{
mul.f16x2 r821, r46, r823;
}
{
add.f16x2 r824, r800, r821;
}
{
cvt.rn.f16.f64 rs121, fd311;
}
mov.b32 r829, {rs121, rs121};
{
mul.f16x2 r827, r40, r829;
}
{
add.f16x2 r830, r806, r827;
}
{
cvt.rn.f16.f64 rs122, fd256;
}
mov.b32 r835, {rs122, rs122};
{
mul.f16x2 r833, r43, r835;
}
{
add.f16x2 r836, r812, r833;
}
{
cvt.rn.f16.f64 rs123, fd291;
}
mov.b32 r841, {rs123, rs123};
{
mul.f16x2 r839, r49, r841;
}
{
add.f16x2 r842, r818, r839;
}
{
cvt.rn.f16.f64 rs124, fd272;
}
mov.b32 r847, {rs124, rs124};
{
mul.f16x2 r845, r58, r847;
}
{
add.f16x2 r848, r824, r845;
}
{
cvt.rn.f16.f64 rs125, fd291;
}
mov.b32 r853, {rs125, rs125};
{
mul.f16x2 r851, r52, r853;
}
{
add.f16x2 r854, r830, r851;
}
{
cvt.rn.f16.f64 rs126, fd272;
}
mov.b32 r859, {rs126, rs126};
{
mul.f16x2 r857, r55, r859;
}
{
add.f16x2 r860, r836, r857;
}
{
cvt.rn.f16.f64 rs127, fd303;
}
mov.b32 r865, {rs127, rs127};
{
mul.f16x2 r863, r61, r865;
}
{
add.f16x2 r866, r842, r863;
}
{
cvt.rn.f16.f64 rs128, fd304;
}
mov.b32 r871, {rs128, rs128};
{
mul.f16x2 r869, r70, r871;
}
{
add.f16x2 r872, r848, r869;
}
{
cvt.rn.f16.f64 rs129, fd303;
}
mov.b32 r877, {rs129, rs129};
{
mul.f16x2 r875, r64, r877;
}
{
add.f16x2 r878, r854, r875;
}
{
cvt.rn.f16.f64 rs130, fd304;
}
mov.b32 r883, {rs130, rs130};
{
mul.f16x2 r881, r67, r883;
}
{
add.f16x2 r884, r860, r881;
}
{
sub.f16x2 r887, r866, r872;
}
{
add.f16x2 r890, r878, r884;
}
{
add.f16x2 r893, r866, r872;
}
{
sub.f16x2 r896, r878, r884;
}
cvt.rn.f16.s32 rs131, r1956;
mov.b32 r911, {rs131, rs131};
cvt.rn.f16.s32 rs132, r1956;
mov.b32 r923, {rs132, rs132};
{
cvt.rn.f16.f64 rs133, fd311;
}
mov.b32 r903, {rs133, rs133};
{
mul.f16x2 r901, r1, r903;
}
{
add.f16x2 r904, %52, r901;
}
{
cvt.rn.f16.f64 rs134, fd312;
}
mov.b32 r909, {rs134, rs134};
{
mul.f16x2 r907, r10, r909;
}
{
add.f16x2 r910, r911, r907;
}
{
cvt.rn.f16.f64 rs135, fd311;
}
mov.b32 r915, {rs135, rs135};
{
mul.f16x2 r913, r4, r915;
}
{
add.f16x2 r916, %53, r913;
}
{
cvt.rn.f16.f64 rs136, fd312;
}
mov.b32 r921, {rs136, rs136};
{
mul.f16x2 r919, r7, r921;
}
{
add.f16x2 r922, r923, r919;
}
{
cvt.rn.f16.f64 rs137, fd291;
}
mov.b32 r927, {rs137, rs137};
{
mul.f16x2 r925, r13, r927;
}
{
add.f16x2 r928, r904, r925;
}
{
cvt.rn.f16.f64 rs138, fd272;
}
mov.b32 r933, {rs138, rs138};
{
mul.f16x2 r931, r22, r933;
}
{
add.f16x2 r934, r910, r931;
}
{
cvt.rn.f16.f64 rs139, fd291;
}
mov.b32 r939, {rs139, rs139};
{
mul.f16x2 r937, r16, r939;
}
{
add.f16x2 r940, r916, r937;
}
{
cvt.rn.f16.f64 rs140, fd272;
}
mov.b32 r945, {rs140, rs140};
{
mul.f16x2 r943, r19, r945;
}
{
add.f16x2 r946, r922, r943;
}
{
cvt.rn.f16.f64 rs141, fd307;
}
mov.b32 r951, {rs141, rs141};
{
mul.f16x2 r949, r25, r951;
}
{
add.f16x2 r952, r928, r949;
}
{
cvt.rn.f16.f64 rs142, fd308;
}
mov.b32 r957, {rs142, rs142};
{
mul.f16x2 r955, r34, r957;
}
{
add.f16x2 r958, r934, r955;
}
{
cvt.rn.f16.f64 rs143, fd307;
}
mov.b32 r963, {rs143, rs143};
{
mul.f16x2 r961, r28, r963;
}
{
add.f16x2 r964, r940, r961;
}
{
cvt.rn.f16.f64 rs144, fd308;
}
mov.b32 r969, {rs144, rs144};
{
mul.f16x2 r967, r31, r969;
}
{
add.f16x2 r970, r946, r967;
}
{
cvt.rn.f16.f64 rs145, fd295;
}
mov.b32 r975, {rs145, rs145};
{
mul.f16x2 r973, r37, r975;
}
{
add.f16x2 r976, r952, r973;
}
{
cvt.rn.f16.f64 rs146, fd280;
}
mov.b32 r981, {rs146, rs146};
{
mul.f16x2 r979, r46, r981;
}
{
add.f16x2 r982, r958, r979;
}
{
cvt.rn.f16.f64 rs147, fd295;
}
mov.b32 r987, {rs147, rs147};
{
mul.f16x2 r985, r40, r987;
}
{
add.f16x2 r988, r964, r985;
}
{
cvt.rn.f16.f64 rs148, fd280;
}
mov.b32 r993, {rs148, rs148};
{
mul.f16x2 r991, r43, r993;
}
{
add.f16x2 r994, r970, r991;
}
{
cvt.rn.f16.f64 rs149, fd303;
}
mov.b32 r999, {rs149, rs149};
{
mul.f16x2 r997, r49, r999;
}
{
add.f16x2 r1000, r976, r997;
}
{
cvt.rn.f16.f64 rs150, fd304;
}
mov.b32 r1005, {rs150, rs150};
{
mul.f16x2 r1003, r58, r1005;
}
{
add.f16x2 r1006, r982, r1003;
}
{
cvt.rn.f16.f64 rs151, fd303;
}
mov.b32 r1011, {rs151, rs151};
{
mul.f16x2 r1009, r52, r1011;
}
{
add.f16x2 r1012, r988, r1009;
}
{
cvt.rn.f16.f64 rs152, fd304;
}
mov.b32 r1017, {rs152, rs152};
{
mul.f16x2 r1015, r55, r1017;
}
{
add.f16x2 r1018, r994, r1015;
}
{
cvt.rn.f16.f64 rs153, fd299;
}
mov.b32 r1023, {rs153, rs153};
{
mul.f16x2 r1021, r61, r1023;
}
{
add.f16x2 r1024, r1000, r1021;
}
{
cvt.rn.f16.f64 rs154, fd288;
}
mov.b32 r1029, {rs154, rs154};
{
mul.f16x2 r1027, r70, r1029;
}
{
add.f16x2 r1030, r1006, r1027;
}
{
cvt.rn.f16.f64 rs155, fd299;
}
mov.b32 r1035, {rs155, rs155};
{
mul.f16x2 r1033, r64, r1035;
}
{
add.f16x2 r1036, r1012, r1033;
}
{
cvt.rn.f16.f64 rs156, fd288;
}
mov.b32 r1041, {rs156, rs156};
{
mul.f16x2 r1039, r67, r1041;
}
{
add.f16x2 r1042, r1018, r1039;
}
{
sub.f16x2 r1045, r1024, r1030;
}
{
add.f16x2 r1048, r1036, r1042;
}
{
add.f16x2 r1051, r1024, r1030;
}
{
sub.f16x2 r1054, r1036, r1042;
}
{
add.f16x2 r1057, %96, %88;
}
{
add.f16x2 r1060, %99, %93;
}
{
sub.f16x2 r1063, %96, %88;
}
{
sub.f16x2 r1066, %99, %93;
}
{
add.f16x2 r1069, %83, %103;
}
{
add.f16x2 r1072, %87, %81;
}
{
sub.f16x2 r1075, %83, %103;
}
{
sub.f16x2 r1078, %87, %81;
}
{
add.f16x2 r1081, %97, %91;
}
{
add.f16x2 r1084, %101, %94;
}
{
sub.f16x2 r1087, %97, %91;
}
{
sub.f16x2 r1090, %101, %94;
}
{
add.f16x2 r1093, %85, %78;
}
{
add.f16x2 r1096, %89, %82;
}
{
sub.f16x2 r1099, %85, %78;
}
{
sub.f16x2 r1102, %89, %82;
}
{
add.f16x2 r1105, %98, %92;
}
{
add.f16x2 r1108, %102, %95;
}
{
sub.f16x2 r1111, %98, %92;
}
{
sub.f16x2 r1114, %102, %95;
}
{
add.f16x2 r1117, %86, %80;
}
{
add.f16x2 r1120, %90, %84;
}
{
sub.f16x2 r1123, %86, %80;
}
{
sub.f16x2 r1126, %90, %84;
}
{
add.f16x2 r1129, %100, r1057;
}
{
add.f16x2 r1132, %79, r1060;
}
{
add.f16x2 r1135, r1129, r1069;
}
{
add.f16x2 r1138, r1132, r1072;
}
{
add.f16x2 r1141, r1135, r1081;
}
{
add.f16x2 r1144, r1138, r1084;
}
{
add.f16x2 r1147, r1141, r1093;
}
{
add.f16x2 r1150, r1144, r1096;
}
{
add.f16x2 r1153, r1147, r1105;
}
{
add.f16x2 r1156, r1150, r1108;
}
{
add.f16x2 r1159, r1153, r1117;
}
{
add.f16x2 r1162, r1156, r1120;
}
cvt.rn.f16.s32 rs157, r1956;
mov.b32 r1177, {rs157, rs157};
cvt.rn.f16.s32 rs158, r1956;
mov.b32 r1189, {rs158, rs158};
{
cvt.rn.f16.f64 rs159, fd291;
}
mov.b32 r1169, {rs159, rs159};
{
mul.f16x2 r1167, r1057, r1169;
}
{
add.f16x2 r1170, %100, r1167;
}
{
cvt.rn.f16.f64 rs160, fd310;
}
mov.b32 r1175, {rs160, rs160};
{
mul.f16x2 r1173, r1066, r1175;
}
{
add.f16x2 r1176, r1177, r1173;
}
{
cvt.rn.f16.f64 rs161, fd291;
}
mov.b32 r1181, {rs161, rs161};
{
mul.f16x2 r1179, r1060, r1181;
}
{
add.f16x2 r1182, %79, r1179;
}
{
cvt.rn.f16.f64 rs162, fd310;
}
mov.b32 r1187, {rs162, rs162};
{
mul.f16x2 r1185, r1063, r1187;
}
{
add.f16x2 r1188, r1189, r1185;
}
{
cvt.rn.f16.f64 rs163, fd295;
}
mov.b32 r1193, {rs163, rs163};
{
mul.f16x2 r1191, r1069, r1193;
}
{
add.f16x2 r1194, r1170, r1191;
}
{
cvt.rn.f16.f64 rs164, fd306;
}
mov.b32 r1199, {rs164, rs164};
{
mul.f16x2 r1197, r1078, r1199;
}
{
add.f16x2 r1200, r1176, r1197;
}
{
cvt.rn.f16.f64 rs165, fd295;
}
mov.b32 r1205, {rs165, rs165};
{
mul.f16x2 r1203, r1072, r1205;
}
{
add.f16x2 r1206, r1182, r1203;
}
{
cvt.rn.f16.f64 rs166, fd306;
}
mov.b32 r1211, {rs166, rs166};
{
mul.f16x2 r1209, r1075, r1211;
}
{
add.f16x2 r1212, r1188, r1209;
}
{
cvt.rn.f16.f64 rs167, fd299;
}
mov.b32 r1217, {rs167, rs167};
{
mul.f16x2 r1215, r1081, r1217;
}
{
add.f16x2 r1218, r1194, r1215;
}
{
cvt.rn.f16.f64 rs168, fd302;
}
mov.b32 r1223, {rs168, rs168};
{
mul.f16x2 r1221, r1090, r1223;
}
{
add.f16x2 r1224, r1200, r1221;
}
{
cvt.rn.f16.f64 rs169, fd299;
}
mov.b32 r1229, {rs169, rs169};
{
mul.f16x2 r1227, r1084, r1229;
}
{
add.f16x2 r1230, r1206, r1227;
}
{
cvt.rn.f16.f64 rs170, fd302;
}
mov.b32 r1235, {rs170, rs170};
{
mul.f16x2 r1233, r1087, r1235;
}
{
add.f16x2 r1236, r1212, r1233;
}
{
cvt.rn.f16.f64 rs171, fd303;
}
mov.b32 r1241, {rs171, rs171};
{
mul.f16x2 r1239, r1093, r1241;
}
{
add.f16x2 r1242, r1218, r1239;
}
{
cvt.rn.f16.f64 rs172, fd304;
}
mov.b32 r1247, {rs172, rs172};
{
mul.f16x2 r1245, r1102, r1247;
}
{
add.f16x2 r1248, r1224, r1245;
}
{
cvt.rn.f16.f64 rs173, fd303;
}
mov.b32 r1253, {rs173, rs173};
{
mul.f16x2 r1251, r1096, r1253;
}
{
add.f16x2 r1254, r1230, r1251;
}
{
cvt.rn.f16.f64 rs174, fd304;
}
mov.b32 r1259, {rs174, rs174};
{
mul.f16x2 r1257, r1099, r1259;
}
{
add.f16x2 r1260, r1236, r1257;
}
{
cvt.rn.f16.f64 rs175, fd307;
}
mov.b32 r1265, {rs175, rs175};
{
mul.f16x2 r1263, r1105, r1265;
}
{
add.f16x2 r1266, r1242, r1263;
}
{
cvt.rn.f16.f64 rs176, fd308;
}
mov.b32 r1271, {rs176, rs176};
{
mul.f16x2 r1269, r1114, r1271;
}
{
add.f16x2 r1272, r1248, r1269;
}
{
cvt.rn.f16.f64 rs177, fd307;
}
mov.b32 r1277, {rs177, rs177};
{
mul.f16x2 r1275, r1108, r1277;
}
{
add.f16x2 r1278, r1254, r1275;
}
{
cvt.rn.f16.f64 rs178, fd308;
}
mov.b32 r1283, {rs178, rs178};
{
mul.f16x2 r1281, r1111, r1283;
}
{
add.f16x2 r1284, r1260, r1281;
}
{
cvt.rn.f16.f64 rs179, fd311;
}
mov.b32 r1289, {rs179, rs179};
{
mul.f16x2 r1287, r1117, r1289;
}
{
add.f16x2 r1290, r1266, r1287;
}
{
cvt.rn.f16.f64 rs180, fd312;
}
mov.b32 r1295, {rs180, rs180};
{
mul.f16x2 r1293, r1126, r1295;
}
{
add.f16x2 r1296, r1272, r1293;
}
{
cvt.rn.f16.f64 rs181, fd311;
}
mov.b32 r1301, {rs181, rs181};
{
mul.f16x2 r1299, r1120, r1301;
}
{
add.f16x2 r1302, r1278, r1299;
}
{
cvt.rn.f16.f64 rs182, fd312;
}
mov.b32 r1307, {rs182, rs182};
{
mul.f16x2 r1305, r1123, r1307;
}
{
add.f16x2 r1308, r1284, r1305;
}
{
sub.f16x2 r1311, r1290, r1296;
}
{
add.f16x2 r1314, r1302, r1308;
}
{
add.f16x2 r1317, r1290, r1296;
}
{
sub.f16x2 r1320, r1302, r1308;
}
cvt.rn.f16.s32 rs183, r1956;
mov.b32 r1335, {rs183, rs183};
cvt.rn.f16.s32 rs184, r1956;
mov.b32 r1347, {rs184, rs184};
{
cvt.rn.f16.f64 rs185, fd295;
}
mov.b32 r1327, {rs185, rs185};
{
mul.f16x2 r1325, r1057, r1327;
}
{
add.f16x2 r1328, %100, r1325;
}
{
cvt.rn.f16.f64 rs186, fd306;
}
mov.b32 r1333, {rs186, rs186};
{
mul.f16x2 r1331, r1066, r1333;
}
{
add.f16x2 r1334, r1335, r1331;
}
{
cvt.rn.f16.f64 rs187, fd295;
}
mov.b32 r1339, {rs187, rs187};
{
mul.f16x2 r1337, r1060, r1339;
}
{
add.f16x2 r1340, %79, r1337;
}
{
cvt.rn.f16.f64 rs188, fd306;
}
mov.b32 r1345, {rs188, rs188};
{
mul.f16x2 r1343, r1063, r1345;
}
{
add.f16x2 r1346, r1347, r1343;
}
{
cvt.rn.f16.f64 rs189, fd303;
}
mov.b32 r1351, {rs189, rs189};
{
mul.f16x2 r1349, r1069, r1351;
}
{
add.f16x2 r1352, r1328, r1349;
}
{
cvt.rn.f16.f64 rs190, fd304;
}
mov.b32 r1357, {rs190, rs190};
{
mul.f16x2 r1355, r1078, r1357;
}
{
add.f16x2 r1358, r1334, r1355;
}
{
cvt.rn.f16.f64 rs191, fd303;
}
mov.b32 r1363, {rs191, rs191};
{
mul.f16x2 r1361, r1072, r1363;
}
{
add.f16x2 r1364, r1340, r1361;
}
{
cvt.rn.f16.f64 rs192, fd304;
}
mov.b32 r1369, {rs192, rs192};
{
mul.f16x2 r1367, r1075, r1369;
}
{
add.f16x2 r1370, r1346, r1367;
}
{
cvt.rn.f16.f64 rs193, fd311;
}
mov.b32 r1375, {rs193, rs193};
{
mul.f16x2 r1373, r1081, r1375;
}
{
add.f16x2 r1376, r1352, r1373;
}
{
cvt.rn.f16.f64 rs194, fd312;
}
mov.b32 r1381, {rs194, rs194};
{
mul.f16x2 r1379, r1090, r1381;
}
{
add.f16x2 r1382, r1358, r1379;
}
{
cvt.rn.f16.f64 rs195, fd311;
}
mov.b32 r1387, {rs195, rs195};
{
mul.f16x2 r1385, r1084, r1387;
}
{
add.f16x2 r1388, r1364, r1385;
}
{
cvt.rn.f16.f64 rs196, fd312;
}
mov.b32 r1393, {rs196, rs196};
{
mul.f16x2 r1391, r1087, r1393;
}
{
add.f16x2 r1394, r1370, r1391;
}
{
cvt.rn.f16.f64 rs197, fd307;
}
mov.b32 r1399, {rs197, rs197};
{
mul.f16x2 r1397, r1093, r1399;
}
{
add.f16x2 r1400, r1376, r1397;
}
{
cvt.rn.f16.f64 rs198, fd224;
}
mov.b32 r1405, {rs198, rs198};
{
mul.f16x2 r1403, r1102, r1405;
}
{
add.f16x2 r1406, r1382, r1403;
}
{
cvt.rn.f16.f64 rs199, fd307;
}
mov.b32 r1411, {rs199, rs199};
{
mul.f16x2 r1409, r1096, r1411;
}
{
add.f16x2 r1412, r1388, r1409;
}
{
cvt.rn.f16.f64 rs200, fd224;
}
mov.b32 r1417, {rs200, rs200};
{
mul.f16x2 r1415, r1099, r1417;
}
{
add.f16x2 r1418, r1394, r1415;
}
{
cvt.rn.f16.f64 rs201, fd299;
}
mov.b32 r1423, {rs201, rs201};
{
mul.f16x2 r1421, r1105, r1423;
}
{
add.f16x2 r1424, r1400, r1421;
}
{
cvt.rn.f16.f64 rs202, fd288;
}
mov.b32 r1429, {rs202, rs202};
{
mul.f16x2 r1427, r1114, r1429;
}
{
add.f16x2 r1430, r1406, r1427;
}
{
cvt.rn.f16.f64 rs203, fd299;
}
mov.b32 r1435, {rs203, rs203};
{
mul.f16x2 r1433, r1108, r1435;
}
{
add.f16x2 r1436, r1412, r1433;
}
{
cvt.rn.f16.f64 rs204, fd288;
}
mov.b32 r1441, {rs204, rs204};
{
mul.f16x2 r1439, r1111, r1441;
}
{
add.f16x2 r1442, r1418, r1439;
}
{
cvt.rn.f16.f64 rs205, fd291;
}
mov.b32 r1447, {rs205, rs205};
{
mul.f16x2 r1445, r1117, r1447;
}
{
add.f16x2 r1448, r1424, r1445;
}
{
cvt.rn.f16.f64 rs206, fd272;
}
mov.b32 r1453, {rs206, rs206};
{
mul.f16x2 r1451, r1126, r1453;
}
{
add.f16x2 r1454, r1430, r1451;
}
{
cvt.rn.f16.f64 rs207, fd291;
}
mov.b32 r1459, {rs207, rs207};
{
mul.f16x2 r1457, r1120, r1459;
}
{
add.f16x2 r1460, r1436, r1457;
}
{
cvt.rn.f16.f64 rs208, fd272;
}
mov.b32 r1465, {rs208, rs208};
{
mul.f16x2 r1463, r1123, r1465;
}
{
add.f16x2 r1466, r1442, r1463;
}
{
sub.f16x2 r1469, r1448, r1454;
}
{
add.f16x2 r1472, r1460, r1466;
}
{
add.f16x2 r1475, r1448, r1454;
}
{
sub.f16x2 r1478, r1460, r1466;
}
cvt.rn.f16.s32 rs209, r1956;
mov.b32 r1493, {rs209, rs209};
cvt.rn.f16.s32 rs210, r1956;
mov.b32 r1505, {rs210, rs210};
{
cvt.rn.f16.f64 rs211, fd299;
}
mov.b32 r1485, {rs211, rs211};
{
mul.f16x2 r1483, r1057, r1485;
}
{
add.f16x2 r1486, %100, r1483;
}
{
cvt.rn.f16.f64 rs212, fd302;
}
mov.b32 r1491, {rs212, rs212};
{
mul.f16x2 r1489, r1066, r1491;
}
{
add.f16x2 r1492, r1493, r1489;
}
{
cvt.rn.f16.f64 rs213, fd299;
}
mov.b32 r1497, {rs213, rs213};
{
mul.f16x2 r1495, r1060, r1497;
}
{
add.f16x2 r1498, %79, r1495;
}
{
cvt.rn.f16.f64 rs214, fd302;
}
mov.b32 r1503, {rs214, rs214};
{
mul.f16x2 r1501, r1063, r1503;
}
{
add.f16x2 r1504, r1505, r1501;
}
{
cvt.rn.f16.f64 rs215, fd311;
}
mov.b32 r1509, {rs215, rs215};
{
mul.f16x2 r1507, r1069, r1509;
}
{
add.f16x2 r1510, r1486, r1507;
}
{
cvt.rn.f16.f64 rs216, fd312;
}
mov.b32 r1515, {rs216, rs216};
{
mul.f16x2 r1513, r1078, r1515;
}
{
add.f16x2 r1516, r1492, r1513;
}
{
cvt.rn.f16.f64 rs217, fd311;
}
mov.b32 r1521, {rs217, rs217};
{
mul.f16x2 r1519, r1072, r1521;
}
{
add.f16x2 r1522, r1498, r1519;
}
{
cvt.rn.f16.f64 rs218, fd312;
}
mov.b32 r1527, {rs218, rs218};
{
mul.f16x2 r1525, r1075, r1527;
}
{
add.f16x2 r1528, r1504, r1525;
}
{
cvt.rn.f16.f64 rs219, fd303;
}
mov.b32 r1533, {rs219, rs219};
{
mul.f16x2 r1531, r1081, r1533;
}
{
add.f16x2 r1534, r1510, r1531;
}
{
cvt.rn.f16.f64 rs220, fd204;
}
mov.b32 r1539, {rs220, rs220};
{
mul.f16x2 r1537, r1090, r1539;
}
{
add.f16x2 r1540, r1516, r1537;
}
{
cvt.rn.f16.f64 rs221, fd303;
}
mov.b32 r1545, {rs221, rs221};
{
mul.f16x2 r1543, r1084, r1545;
}
{
add.f16x2 r1546, r1522, r1543;
}
{
cvt.rn.f16.f64 rs222, fd204;
}
mov.b32 r1551, {rs222, rs222};
{
mul.f16x2 r1549, r1087, r1551;
}
{
add.f16x2 r1552, r1528, r1549;
}
{
cvt.rn.f16.f64 rs223, fd291;
}
mov.b32 r1557, {rs223, rs223};
{
mul.f16x2 r1555, r1093, r1557;
}
{
add.f16x2 r1558, r1534, r1555;
}
{
cvt.rn.f16.f64 rs224, fd272;
}
mov.b32 r1563, {rs224, rs224};
{
mul.f16x2 r1561, r1102, r1563;
}
{
add.f16x2 r1564, r1540, r1561;
}
{
cvt.rn.f16.f64 rs225, fd291;
}
mov.b32 r1569, {rs225, rs225};
{
mul.f16x2 r1567, r1096, r1569;
}
{
add.f16x2 r1570, r1546, r1567;
}
{
cvt.rn.f16.f64 rs226, fd272;
}
mov.b32 r1575, {rs226, rs226};
{
mul.f16x2 r1573, r1099, r1575;
}
{
add.f16x2 r1576, r1552, r1573;
}
{
cvt.rn.f16.f64 rs227, fd295;
}
mov.b32 r1581, {rs227, rs227};
{
mul.f16x2 r1579, r1105, r1581;
}
{
add.f16x2 r1582, r1558, r1579;
}
{
cvt.rn.f16.f64 rs228, fd306;
}
mov.b32 r1587, {rs228, rs228};
{
mul.f16x2 r1585, r1114, r1587;
}
{
add.f16x2 r1588, r1564, r1585;
}
{
cvt.rn.f16.f64 rs229, fd295;
}
mov.b32 r1593, {rs229, rs229};
{
mul.f16x2 r1591, r1108, r1593;
}
{
add.f16x2 r1594, r1570, r1591;
}
{
cvt.rn.f16.f64 rs230, fd306;
}
mov.b32 r1599, {rs230, rs230};
{
mul.f16x2 r1597, r1111, r1599;
}
{
add.f16x2 r1600, r1576, r1597;
}
{
cvt.rn.f16.f64 rs231, fd307;
}
mov.b32 r1605, {rs231, rs231};
{
mul.f16x2 r1603, r1117, r1605;
}
{
add.f16x2 r1606, r1582, r1603;
}
{
cvt.rn.f16.f64 rs232, fd308;
}
mov.b32 r1611, {rs232, rs232};
{
mul.f16x2 r1609, r1126, r1611;
}
{
add.f16x2 r1612, r1588, r1609;
}
{
cvt.rn.f16.f64 rs233, fd307;
}
mov.b32 r1617, {rs233, rs233};
{
mul.f16x2 r1615, r1120, r1617;
}
{
add.f16x2 r1618, r1594, r1615;
}
{
cvt.rn.f16.f64 rs234, fd308;
}
mov.b32 r1623, {rs234, rs234};
{
mul.f16x2 r1621, r1123, r1623;
}
{
add.f16x2 r1624, r1600, r1621;
}
{
sub.f16x2 r1627, r1606, r1612;
}
{
add.f16x2 r1630, r1618, r1624;
}
{
add.f16x2 r1633, r1606, r1612;
}
{
sub.f16x2 r1636, r1618, r1624;
}
cvt.rn.f16.s32 rs235, r1956;
mov.b32 r1651, {rs235, rs235};
cvt.rn.f16.s32 rs236, r1956;
mov.b32 r1663, {rs236, rs236};
{
cvt.rn.f16.f64 rs237, fd303;
}
mov.b32 r1643, {rs237, rs237};
{
mul.f16x2 r1641, r1057, r1643;
}
{
add.f16x2 r1644, %100, r1641;
}
{
cvt.rn.f16.f64 rs238, fd304;
}
mov.b32 r1649, {rs238, rs238};
{
mul.f16x2 r1647, r1066, r1649;
}
{
add.f16x2 r1650, r1651, r1647;
}
{
cvt.rn.f16.f64 rs239, fd303;
}
mov.b32 r1655, {rs239, rs239};
{
mul.f16x2 r1653, r1060, r1655;
}
{
add.f16x2 r1656, %79, r1653;
}
{
cvt.rn.f16.f64 rs240, fd304;
}
mov.b32 r1661, {rs240, rs240};
{
mul.f16x2 r1659, r1063, r1661;
}
{
add.f16x2 r1662, r1663, r1659;
}
{
cvt.rn.f16.f64 rs241, fd307;
}
mov.b32 r1667, {rs241, rs241};
{
mul.f16x2 r1665, r1069, r1667;
}
{
add.f16x2 r1668, r1644, r1665;
}
{
cvt.rn.f16.f64 rs242, fd224;
}
mov.b32 r1673, {rs242, rs242};
{
mul.f16x2 r1671, r1078, r1673;
}
{
add.f16x2 r1674, r1650, r1671;
}
{
cvt.rn.f16.f64 rs243, fd307;
}
mov.b32 r1679, {rs243, rs243};
{
mul.f16x2 r1677, r1072, r1679;
}
{
add.f16x2 r1680, r1656, r1677;
}
{
cvt.rn.f16.f64 rs244, fd224;
}
mov.b32 r1685, {rs244, rs244};
{
mul.f16x2 r1683, r1075, r1685;
}
{
add.f16x2 r1686, r1662, r1683;
}
{
cvt.rn.f16.f64 rs245, fd291;
}
mov.b32 r1691, {rs245, rs245};
{
mul.f16x2 r1689, r1081, r1691;
}
{
add.f16x2 r1692, r1668, r1689;
}
{
cvt.rn.f16.f64 rs246, fd272;
}
mov.b32 r1697, {rs246, rs246};
{
mul.f16x2 r1695, r1090, r1697;
}
{
add.f16x2 r1698, r1674, r1695;
}
{
cvt.rn.f16.f64 rs247, fd291;
}
mov.b32 r1703, {rs247, rs247};
{
mul.f16x2 r1701, r1084, r1703;
}
{
add.f16x2 r1704, r1680, r1701;
}
{
cvt.rn.f16.f64 rs248, fd272;
}
mov.b32 r1709, {rs248, rs248};
{
mul.f16x2 r1707, r1087, r1709;
}
{
add.f16x2 r1710, r1686, r1707;
}
{
cvt.rn.f16.f64 rs249, fd299;
}
mov.b32 r1715, {rs249, rs249};
{
mul.f16x2 r1713, r1093, r1715;
}
{
add.f16x2 r1716, r1692, r1713;
}
{
cvt.rn.f16.f64 rs250, fd302;
}
mov.b32 r1721, {rs250, rs250};
{
mul.f16x2 r1719, r1102, r1721;
}
{
add.f16x2 r1722, r1698, r1719;
}
{
cvt.rn.f16.f64 rs251, fd299;
}
mov.b32 r1727, {rs251, rs251};
{
mul.f16x2 r1725, r1096, r1727;
}
{
add.f16x2 r1728, r1704, r1725;
}
{
cvt.rn.f16.f64 rs252, fd302;
}
mov.b32 r1733, {rs252, rs252};
{
mul.f16x2 r1731, r1099, r1733;
}
{
add.f16x2 r1734, r1710, r1731;
}
{
cvt.rn.f16.f64 rs253, fd311;
}
mov.b32 r1739, {rs253, rs253};
{
mul.f16x2 r1737, r1105, r1739;
}
{
add.f16x2 r1740, r1716, r1737;
}
{
cvt.rn.f16.f64 rs254, fd256;
}
mov.b32 r1745, {rs254, rs254};
{
mul.f16x2 r1743, r1114, r1745;
}
{
add.f16x2 r1746, r1722, r1743;
}
{
cvt.rn.f16.f64 rs255, fd311;
}
mov.b32 r1751, {rs255, rs255};
{
mul.f16x2 r1749, r1108, r1751;
}
{
add.f16x2 r1752, r1728, r1749;
}
{
cvt.rn.f16.f64 rs256, fd256;
}
mov.b32 r1757, {rs256, rs256};
{
mul.f16x2 r1755, r1111, r1757;
}
{
add.f16x2 r1758, r1734, r1755;
}
{
cvt.rn.f16.f64 rs257, fd295;
}
mov.b32 r1763, {rs257, rs257};
{
mul.f16x2 r1761, r1117, r1763;
}
{
add.f16x2 r1764, r1740, r1761;
}
{
cvt.rn.f16.f64 rs258, fd280;
}
mov.b32 r1769, {rs258, rs258};
{
mul.f16x2 r1767, r1126, r1769;
}
{
add.f16x2 r1770, r1746, r1767;
}
{
cvt.rn.f16.f64 rs259, fd295;
}
mov.b32 r1775, {rs259, rs259};
{
mul.f16x2 r1773, r1120, r1775;
}
{
add.f16x2 r1776, r1752, r1773;
}
{
cvt.rn.f16.f64 rs260, fd280;
}
mov.b32 r1781, {rs260, rs260};
{
mul.f16x2 r1779, r1123, r1781;
}
{
add.f16x2 r1782, r1758, r1779;
}
{
sub.f16x2 r1785, r1764, r1770;
}
{
add.f16x2 r1788, r1776, r1782;
}
{
add.f16x2 r1791, r1764, r1770;
}
{
sub.f16x2 r1794, r1776, r1782;
}
cvt.rn.f16.s32 rs261, r1956;
mov.b32 r1809, {rs261, rs261};
cvt.rn.f16.s32 rs262, r1956;
mov.b32 r1821, {rs262, rs262};
{
cvt.rn.f16.f64 rs263, fd307;
}
mov.b32 r1801, {rs263, rs263};
{
mul.f16x2 r1799, r1057, r1801;
}
{
add.f16x2 r1802, %100, r1799;
}
{
cvt.rn.f16.f64 rs264, fd308;
}
mov.b32 r1807, {rs264, rs264};
{
mul.f16x2 r1805, r1066, r1807;
}
{
add.f16x2 r1808, r1809, r1805;
}
{
cvt.rn.f16.f64 rs265, fd307;
}
mov.b32 r1813, {rs265, rs265};
{
mul.f16x2 r1811, r1060, r1813;
}
{
add.f16x2 r1814, %79, r1811;
}
{
cvt.rn.f16.f64 rs266, fd308;
}
mov.b32 r1819, {rs266, rs266};
{
mul.f16x2 r1817, r1063, r1819;
}
{
add.f16x2 r1820, r1821, r1817;
}
{
cvt.rn.f16.f64 rs267, fd299;
}
mov.b32 r1825, {rs267, rs267};
{
mul.f16x2 r1823, r1069, r1825;
}
{
add.f16x2 r1826, r1802, r1823;
}
{
cvt.rn.f16.f64 rs268, fd288;
}
mov.b32 r1831, {rs268, rs268};
{
mul.f16x2 r1829, r1078, r1831;
}
{
add.f16x2 r1832, r1808, r1829;
}
{
cvt.rn.f16.f64 rs269, fd299;
}
mov.b32 r1837, {rs269, rs269};
{
mul.f16x2 r1835, r1072, r1837;
}
{
add.f16x2 r1838, r1814, r1835;
}
{
cvt.rn.f16.f64 rs270, fd288;
}
mov.b32 r1843, {rs270, rs270};
{
mul.f16x2 r1841, r1075, r1843;
}
{
add.f16x2 r1844, r1820, r1841;
}
{
cvt.rn.f16.f64 rs271, fd295;
}
mov.b32 r1849, {rs271, rs271};
{
mul.f16x2 r1847, r1081, r1849;
}
{
add.f16x2 r1850, r1826, r1847;
}
{
cvt.rn.f16.f64 rs272, fd306;
}
mov.b32 r1855, {rs272, rs272};
{
mul.f16x2 r1853, r1090, r1855;
}
{
add.f16x2 r1856, r1832, r1853;
}
{
cvt.rn.f16.f64 rs273, fd295;
}
mov.b32 r1861, {rs273, rs273};
{
mul.f16x2 r1859, r1084, r1861;
}
{
add.f16x2 r1862, r1838, r1859;
}
{
cvt.rn.f16.f64 rs274, fd306;
}
mov.b32 r1867, {rs274, rs274};
{
mul.f16x2 r1865, r1087, r1867;
}
{
add.f16x2 r1868, r1844, r1865;
}
{
cvt.rn.f16.f64 rs275, fd311;
}
mov.b32 r1873, {rs275, rs275};
{
mul.f16x2 r1871, r1093, r1873;
}
{
add.f16x2 r1874, r1850, r1871;
}
{
cvt.rn.f16.f64 rs276, fd256;
}
mov.b32 r1879, {rs276, rs276};
{
mul.f16x2 r1877, r1102, r1879;
}
{
add.f16x2 r1880, r1856, r1877;
}
{
cvt.rn.f16.f64 rs277, fd311;
}
mov.b32 r1885, {rs277, rs277};
{
mul.f16x2 r1883, r1096, r1885;
}
{
add.f16x2 r1886, r1862, r1883;
}
{
cvt.rn.f16.f64 rs278, fd256;
}
mov.b32 r1891, {rs278, rs278};
{
mul.f16x2 r1889, r1099, r1891;
}
{
add.f16x2 r1892, r1868, r1889;
}
{
cvt.rn.f16.f64 rs279, fd291;
}
mov.b32 r1897, {rs279, rs279};
{
mul.f16x2 r1895, r1105, r1897;
}
{
add.f16x2 r1898, r1874, r1895;
}
{
cvt.rn.f16.f64 rs280, fd272;
}
mov.b32 r1903, {rs280, rs280};
{
mul.f16x2 r1901, r1114, r1903;
}
{
add.f16x2 r1904, r1880, r1901;
}
{
cvt.rn.f16.f64 rs281, fd291;
}
mov.b32 r1909, {rs281, rs281};
{
mul.f16x2 r1907, r1108, r1909;
}
{
add.f16x2 r1910, r1886, r1907;
}
{
cvt.rn.f16.f64 rs282, fd272;
}
mov.b32 r1915, {rs282, rs282};
{
mul.f16x2 r1913, r1111, r1915;
}
{
add.f16x2 r1916, r1892, r1913;
}
{
cvt.rn.f16.f64 rs283, fd303;
}
mov.b32 r1921, {rs283, rs283};
{
mul.f16x2 r1919, r1117, r1921;
}
{
add.f16x2 r1922, r1898, r1919;
}
{
cvt.rn.f16.f64 rs284, fd304;
}
mov.b32 r1927, {rs284, rs284};
{
mul.f16x2 r1925, r1126, r1927;
}
{
add.f16x2 r1928, r1904, r1925;
}
{
cvt.rn.f16.f64 rs285, fd303;
}
mov.b32 r1933, {rs285, rs285};
{
mul.f16x2 r1931, r1120, r1933;
}
{
add.f16x2 r1934, r1910, r1931;
}
{
cvt.rn.f16.f64 rs286, fd304;
}
mov.b32 r1939, {rs286, rs286};
{
mul.f16x2 r1937, r1123, r1939;
}
{
add.f16x2 r1940, r1916, r1937;
}
{
sub.f16x2 r1943, r1922, r1928;
}
{
add.f16x2 r1946, r1934, r1940;
}
{
add.f16x2 r1949, r1922, r1928;
}
{
sub.f16x2 r1952, r1934, r1940;
}
cvt.rn.f16.s32 rs287, r1956;
mov.b32 r1967, {rs287, rs287};
cvt.rn.f16.s32 rs288, r1956;
mov.b32 r1979, {rs288, rs288};
{
cvt.rn.f16.f64 rs289, fd311;
}
mov.b32 r1959, {rs289, rs289};
{
mul.f16x2 r1957, r1057, r1959;
}
{
add.f16x2 r1960, %100, r1957;
}
{
cvt.rn.f16.f64 rs290, fd312;
}
mov.b32 r1965, {rs290, rs290};
{
mul.f16x2 r1963, r1066, r1965;
}
{
add.f16x2 r1966, r1967, r1963;
}
{
cvt.rn.f16.f64 rs291, fd311;
}
mov.b32 r1971, {rs291, rs291};
{
mul.f16x2 r1969, r1060, r1971;
}
{
add.f16x2 r1972, %79, r1969;
}
{
cvt.rn.f16.f64 rs292, fd312;
}
mov.b32 r1977, {rs292, rs292};
{
mul.f16x2 r1975, r1063, r1977;
}
{
add.f16x2 r1978, r1979, r1975;
}
{
cvt.rn.f16.f64 rs293, fd291;
}
mov.b32 r1983, {rs293, rs293};
{
mul.f16x2 r1981, r1069, r1983;
}
{
add.f16x2 r1984, r1960, r1981;
}
{
cvt.rn.f16.f64 rs294, fd272;
}
mov.b32 r1989, {rs294, rs294};
{
mul.f16x2 r1987, r1078, r1989;
}
{
add.f16x2 r1990, r1966, r1987;
}
{
cvt.rn.f16.f64 rs295, fd291;
}
mov.b32 r1995, {rs295, rs295};
{
mul.f16x2 r1993, r1072, r1995;
}
{
add.f16x2 r1996, r1972, r1993;
}
{
cvt.rn.f16.f64 rs296, fd272;
}
mov.b32 r2001, {rs296, rs296};
{
mul.f16x2 r1999, r1075, r2001;
}
{
add.f16x2 r2002, r1978, r1999;
}
{
cvt.rn.f16.f64 rs297, fd307;
}
mov.b32 r2007, {rs297, rs297};
{
mul.f16x2 r2005, r1081, r2007;
}
{
add.f16x2 r2008, r1984, r2005;
}
{
cvt.rn.f16.f64 rs298, fd308;
}
mov.b32 r2013, {rs298, rs298};
{
mul.f16x2 r2011, r1090, r2013;
}
{
add.f16x2 r2014, r1990, r2011;
}
{
cvt.rn.f16.f64 rs299, fd307;
}
mov.b32 r2019, {rs299, rs299};
{
mul.f16x2 r2017, r1084, r2019;
}
{
add.f16x2 r2020, r1996, r2017;
}
{
cvt.rn.f16.f64 rs300, fd308;
}
mov.b32 r2025, {rs300, rs300};
{
mul.f16x2 r2023, r1087, r2025;
}
{
add.f16x2 r2026, r2002, r2023;
}
{
cvt.rn.f16.f64 rs301, fd295;
}
mov.b32 r2031, {rs301, rs301};
{
mul.f16x2 r2029, r1093, r2031;
}
{
add.f16x2 r2032, r2008, r2029;
}
{
cvt.rn.f16.f64 rs302, fd280;
}
mov.b32 r2037, {rs302, rs302};
{
mul.f16x2 r2035, r1102, r2037;
}
{
add.f16x2 r2038, r2014, r2035;
}
{
cvt.rn.f16.f64 rs303, fd295;
}
mov.b32 r2043, {rs303, rs303};
{
mul.f16x2 r2041, r1096, r2043;
}
{
add.f16x2 r2044, r2020, r2041;
}
{
cvt.rn.f16.f64 rs304, fd280;
}
mov.b32 r2049, {rs304, rs304};
{
mul.f16x2 r2047, r1099, r2049;
}
{
add.f16x2 r2050, r2026, r2047;
}
{
cvt.rn.f16.f64 rs305, fd303;
}
mov.b32 r2055, {rs305, rs305};
{
mul.f16x2 r2053, r1105, r2055;
}
{
add.f16x2 r2056, r2032, r2053;
}
{
cvt.rn.f16.f64 rs306, fd304;
}
mov.b32 r2061, {rs306, rs306};
{
mul.f16x2 r2059, r1114, r2061;
}
{
add.f16x2 r2062, r2038, r2059;
}
{
cvt.rn.f16.f64 rs307, fd303;
}
mov.b32 r2067, {rs307, rs307};
{
mul.f16x2 r2065, r1108, r2067;
}
{
add.f16x2 r2068, r2044, r2065;
}
{
cvt.rn.f16.f64 rs308, fd304;
}
mov.b32 r2073, {rs308, rs308};
{
mul.f16x2 r2071, r1111, r2073;
}
{
add.f16x2 r2074, r2050, r2071;
}
{
cvt.rn.f16.f64 rs309, fd299;
}
mov.b32 r2079, {rs309, rs309};
{
mul.f16x2 r2077, r1117, r2079;
}
{
add.f16x2 r2080, r2056, r2077;
}
{
cvt.rn.f16.f64 rs310, fd288;
}
mov.b32 r2085, {rs310, rs310};
{
mul.f16x2 r2083, r1126, r2085;
}
{
add.f16x2 r2086, r2062, r2083;
}
{
cvt.rn.f16.f64 rs311, fd299;
}
mov.b32 r2091, {rs311, rs311};
{
mul.f16x2 r2089, r1120, r2091;
}
{
add.f16x2 r2092, r2068, r2089;
}
{
cvt.rn.f16.f64 rs312, fd288;
}
mov.b32 r2097, {rs312, rs312};
{
mul.f16x2 r2095, r1123, r2097;
}
{
add.f16x2 r2098, r2074, r2095;
}
{
sub.f16x2 r2101, r2080, r2086;
}
{
add.f16x2 r2104, r2092, r2098;
}
{
add.f16x2 r2107, r2080, r2086;
}
{
sub.f16x2 r2110, r2092, r2098;
}
mov.f64 fd289, 0d3FEF11F493053D00;
{
cvt.rn.f16.f64 rs313, fd289;
}
{
cvt.rn.f16.f64 rs314, fd312;
}
{
cvt.rn.f16.f64 rs315, fd291;
}
{
cvt.rn.f16.f64 rs316, fd310;
}
mov.f64 fd293, 0d3FE7F3CCD0032E0C;
{
cvt.rn.f16.f64 rs317, fd293;
}
{
cvt.rn.f16.f64 rs318, fd308;
}
{
cvt.rn.f16.f64 rs319, fd295;
}
{
cvt.rn.f16.f64 rs320, fd306;
}
mov.f64 fd297, 0d3FD6B1D8B2365DA1;
{
cvt.rn.f16.f64 rs321, fd297;
}
{
cvt.rn.f16.f64 rs322, fd304;
}
{
cvt.rn.f16.f64 rs323, fd299;
}
{
cvt.rn.f16.f64 rs324, fd302;
}
mov.f64 fd301, 0dBFBEDB7DEBAA3ED8;
{
cvt.rn.f16.f64 rs325, fd301;
}
{
cvt.rn.f16.f64 rs326, fd302;
}
{
cvt.rn.f16.f64 rs327, fd303;
}
{
cvt.rn.f16.f64 rs328, fd304;
}
mov.f64 fd305, 0dBFE22D961EA71119;
{
cvt.rn.f16.f64 rs329, fd305;
}
{
cvt.rn.f16.f64 rs330, fd306;
}
{
cvt.rn.f16.f64 rs331, fd307;
}
{
cvt.rn.f16.f64 rs332, fd308;
}
mov.f64 fd309, 0dBFEC55A7E00740E9;
{
cvt.rn.f16.f64 rs333, fd309;
}
{
cvt.rn.f16.f64 rs334, fd310;
}
{
cvt.rn.f16.f64 rs335, fd311;
}
{
cvt.rn.f16.f64 rs336, fd312;
}
mov.b32 r2127, {rs313, rs313};
{
mul.f16x2 r2113, r1311, r2127;
}
mov.b32 r2124, {rs314, rs314};
{
mul.f16x2 r2116, r1314, r2124;
}
{
sub.f16x2 r2119, r2113, r2116;
}
{
mul.f16x2 r2122, r1311, r2124;
}
{
fma.rn.f16x2 r2125, r1314, r2127, r2122;
}
mov.b32 r2143, {rs315, rs315};
{
mul.f16x2 r2129, r1469, r2143;
}
mov.b32 r2140, {rs316, rs316};
{
mul.f16x2 r2132, r1472, r2140;
}
{
sub.f16x2 r2135, r2129, r2132;
}
{
mul.f16x2 r2138, r1469, r2140;
}
{
fma.rn.f16x2 r2141, r1472, r2143, r2138;
}
mov.b32 r2159, {rs317, rs317};
{
mul.f16x2 r2145, r1627, r2159;
}
mov.b32 r2156, {rs318, rs318};
{
mul.f16x2 r2148, r1630, r2156;
}
{
sub.f16x2 r2151, r2145, r2148;
}
{
mul.f16x2 r2154, r1627, r2156;
}
{
fma.rn.f16x2 r2157, r1630, r2159, r2154;
}
mov.b32 r2175, {rs319, rs319};
{
mul.f16x2 r2161, r1785, r2175;
}
mov.b32 r2172, {rs320, rs320};
{
mul.f16x2 r2164, r1788, r2172;
}
{
sub.f16x2 r2167, r2161, r2164;
}
{
mul.f16x2 r2170, r1785, r2172;
}
{
fma.rn.f16x2 r2173, r1788, r2175, r2170;
}
mov.b32 r2191, {rs321, rs321};
{
mul.f16x2 r2177, r1943, r2191;
}
mov.b32 r2188, {rs322, rs322};
{
mul.f16x2 r2180, r1946, r2188;
}
{
sub.f16x2 r2183, r2177, r2180;
}
{
mul.f16x2 r2186, r1943, r2188;
}
{
fma.rn.f16x2 r2189, r1946, r2191, r2186;
}
mov.b32 r2207, {rs323, rs323};
{
mul.f16x2 r2193, r2101, r2207;
}
mov.b32 r2204, {rs324, rs324};
{
mul.f16x2 r2196, r2104, r2204;
}
{
sub.f16x2 r2199, r2193, r2196;
}
{
mul.f16x2 r2202, r2101, r2204;
}
{
fma.rn.f16x2 r2205, r2104, r2207, r2202;
}
mov.b32 r2223, {rs325, rs325};
{
mul.f16x2 r2209, r2107, r2223;
}
mov.b32 r2220, {rs326, rs326};
{
mul.f16x2 r2212, r2110, r2220;
}
{
sub.f16x2 r2215, r2209, r2212;
}
{
mul.f16x2 r2218, r2107, r2220;
}
{
fma.rn.f16x2 r2221, r2110, r2223, r2218;
}
mov.b32 r2239, {rs327, rs327};
{
mul.f16x2 r2225, r1949, r2239;
}
mov.b32 r2236, {rs328, rs328};
{
mul.f16x2 r2228, r1952, r2236;
}
{
sub.f16x2 r2231, r2225, r2228;
}
{
mul.f16x2 r2234, r1949, r2236;
}
{
fma.rn.f16x2 r2237, r1952, r2239, r2234;
}
mov.b32 r2255, {rs329, rs329};
{
mul.f16x2 r2241, r1791, r2255;
}
mov.b32 r2252, {rs330, rs330};
{
mul.f16x2 r2244, r1794, r2252;
}
{
sub.f16x2 r2247, r2241, r2244;
}
{
mul.f16x2 r2250, r1791, r2252;
}
{
fma.rn.f16x2 r2253, r1794, r2255, r2250;
}
mov.b32 r2271, {rs331, rs331};
{
mul.f16x2 r2257, r1633, r2271;
}
mov.b32 r2268, {rs332, rs332};
{
mul.f16x2 r2260, r1636, r2268;
}
{
sub.f16x2 r2263, r2257, r2260;
}
{
mul.f16x2 r2266, r1633, r2268;
}
{
fma.rn.f16x2 r2269, r1636, r2271, r2266;
}
mov.b32 r2287, {rs333, rs333};
{
mul.f16x2 r2273, r1475, r2287;
}
mov.b32 r2284, {rs334, rs334};
{
mul.f16x2 r2276, r1478, r2284;
}
{
sub.f16x2 r2279, r2273, r2276;
}
{
mul.f16x2 r2282, r1475, r2284;
}
{
fma.rn.f16x2 r2285, r1478, r2287, r2282;
}
mov.b32 r2303, {rs335, rs335};
{
mul.f16x2 r2289, r1317, r2303;
}
mov.b32 r2300, {rs336, rs336};
{
mul.f16x2 r2292, r1320, r2300;
}
{
sub.f16x2 r2295, r2289, r2292;
}
{
mul.f16x2 r2298, r1317, r2300;
}
{
fma.rn.f16x2 r2301, r1320, r2303, r2298;
}
{
add.f16x2 %0, r103, r1159;
}
{
add.f16x2 %1, r106, r1162;
}
{
sub.f16x2 %26, r103, r1159;
}
{
sub.f16x2 %27, r106, r1162;
}
{
add.f16x2 %2, r255, r2119;
}
{
add.f16x2 %3, r258, r2125;
}
{
sub.f16x2 %28, r255, r2119;
}
{
sub.f16x2 %29, r258, r2125;
}
{
add.f16x2 %4, r413, r2135;
}
{
add.f16x2 %5, r416, r2141;
}
{
sub.f16x2 %30, r413, r2135;
}
{
sub.f16x2 %31, r416, r2141;
}
{
add.f16x2 %6, r571, r2151;
}
{
add.f16x2 %7, r574, r2157;
}
{
sub.f16x2 %32, r571, r2151;
}
{
sub.f16x2 %33, r574, r2157;
}
{
add.f16x2 %8, r729, r2167;
}
{
add.f16x2 %9, r732, r2173;
}
{
sub.f16x2 %34, r729, r2167;
}
{
sub.f16x2 %35, r732, r2173;
}
{
add.f16x2 %10, r887, r2183;
}
{
add.f16x2 %11, r890, r2189;
}
{
sub.f16x2 %36, r887, r2183;
}
{
sub.f16x2 %37, r890, r2189;
}
{
add.f16x2 %12, r1045, r2199;
}
{
add.f16x2 %13, r1048, r2205;
}
{
sub.f16x2 %38, r1045, r2199;
}
{
sub.f16x2 %39, r1048, r2205;
}
{
add.f16x2 %14, r1051, r2215;
}
{
add.f16x2 %15, r1054, r2221;
}
{
sub.f16x2 %40, r1051, r2215;
}
{
sub.f16x2 %41, r1054, r2221;
}
{
add.f16x2 %16, r893, r2231;
}
{
add.f16x2 %17, r896, r2237;
}
{
sub.f16x2 %42, r893, r2231;
}
{
sub.f16x2 %43, r896, r2237;
}
{
add.f16x2 %18, r735, r2247;
}
{
add.f16x2 %19, r738, r2253;
}
{
sub.f16x2 %44, r735, r2247;
}
{
sub.f16x2 %45, r738, r2253;
}
{
add.f16x2 %20, r577, r2263;
}
{
add.f16x2 %21, r580, r2269;
}
{
sub.f16x2 %46, r577, r2263;
}
{
sub.f16x2 %47, r580, r2269;
}
{
add.f16x2 %22, r419, r2279;
}
{
add.f16x2 %23, r422, r2285;
}
{
sub.f16x2 %48, r419, r2279;
}
{
sub.f16x2 %49, r422, r2285;
}
{
add.f16x2 %24, r261, r2295;
}
{
add.f16x2 %25, r264, r2301;
}
{
sub.f16x2 %50, r261, r2295;
}
{
sub.f16x2 %51, r264, r2301;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[23].x)));
};


#endif
