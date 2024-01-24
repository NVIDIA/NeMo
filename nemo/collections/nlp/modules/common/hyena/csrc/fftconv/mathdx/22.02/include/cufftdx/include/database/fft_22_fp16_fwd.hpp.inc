#ifndef CUFFTDX_FFT_22_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_22_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<754, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<263>;
.reg .b32 r<1835>;
.reg .f64 fd<243>;
.reg .b64 rd<2>;
{
add.f16x2 r1, %46, %64;
}
{
add.f16x2 r4, %47, %65;
}
{
sub.f16x2 r7, %46, %64;
}
{
sub.f16x2 r10, %47, %65;
}
{
add.f16x2 r13, %48, %62;
}
{
add.f16x2 r16, %49, %63;
}
{
sub.f16x2 r19, %48, %62;
}
{
sub.f16x2 r22, %49, %63;
}
{
add.f16x2 r25, %50, %60;
}
{
add.f16x2 r28, %51, %61;
}
{
sub.f16x2 r31, %50, %60;
}
{
sub.f16x2 r34, %51, %61;
}
{
add.f16x2 r37, %52, %58;
}
{
add.f16x2 r40, %53, %59;
}
{
sub.f16x2 r43, %52, %58;
}
{
sub.f16x2 r46, %53, %59;
}
{
add.f16x2 r49, %54, %56;
}
{
add.f16x2 r52, %55, %57;
}
{
sub.f16x2 r55, %54, %56;
}
{
sub.f16x2 r58, %55, %57;
}
{
add.f16x2 r61, %44, r1;
}
{
add.f16x2 r64, %45, r4;
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
mov.u32 r1388, 0;
cvt.rn.f16.s32 rs1, r1388;
mov.b32 r103, {rs1, rs1};
cvt.rn.f16.s32 rs2, r1388;
mov.b32 r115, {rs2, rs2};
mov.f64 fd203, 0d3FEAEB8C8764F0BA;
{
cvt.rn.f16.f64 rs3, fd203;
}
mov.b32 r95, {rs3, rs3};
{
mul.f16x2 r93, r1, r95;
}
{
add.f16x2 r96, %44, r93;
}
mov.f64 fd218, 0dBFE14CEDF8BB580B;
{
cvt.rn.f16.f64 rs4, fd218;
}
mov.b32 r101, {rs4, rs4};
{
mul.f16x2 r99, r10, r101;
}
{
add.f16x2 r102, r103, r99;
}
{
cvt.rn.f16.f64 rs5, fd203;
}
mov.b32 r107, {rs5, rs5};
{
mul.f16x2 r105, r4, r107;
}
{
add.f16x2 r108, %45, r105;
}
{
cvt.rn.f16.f64 rs6, fd218;
}
mov.b32 r113, {rs6, rs6};
{
mul.f16x2 r111, r7, r113;
}
{
add.f16x2 r114, r115, r111;
}
mov.f64 fd207, 0d3FDA9628D9C712B6;
{
cvt.rn.f16.f64 rs7, fd207;
}
mov.b32 r119, {rs7, rs7};
{
mul.f16x2 r117, r13, r119;
}
{
add.f16x2 r120, r96, r117;
}
mov.f64 fd214, 0dBFED1BB48EEE2C13;
{
cvt.rn.f16.f64 rs8, fd214;
}
mov.b32 r125, {rs8, rs8};
{
mul.f16x2 r123, r22, r125;
}
{
add.f16x2 r126, r102, r123;
}
{
cvt.rn.f16.f64 rs9, fd207;
}
mov.b32 r131, {rs9, rs9};
{
mul.f16x2 r129, r16, r131;
}
{
add.f16x2 r132, r108, r129;
}
{
cvt.rn.f16.f64 rs10, fd214;
}
mov.b32 r137, {rs10, rs10};
{
mul.f16x2 r135, r19, r137;
}
{
add.f16x2 r138, r114, r135;
}
mov.f64 fd211, 0dBFC2375F640F44DB;
{
cvt.rn.f16.f64 rs11, fd211;
}
mov.b32 r143, {rs11, rs11};
{
mul.f16x2 r141, r25, r143;
}
{
add.f16x2 r144, r120, r141;
}
mov.f64 fd212, 0dBFEFAC9E043842EF;
{
cvt.rn.f16.f64 rs12, fd212;
}
mov.b32 r149, {rs12, rs12};
{
mul.f16x2 r147, r34, r149;
}
{
add.f16x2 r150, r126, r147;
}
{
cvt.rn.f16.f64 rs13, fd211;
}
mov.b32 r155, {rs13, rs13};
{
mul.f16x2 r153, r28, r155;
}
{
add.f16x2 r156, r132, r153;
}
{
cvt.rn.f16.f64 rs14, fd212;
}
mov.b32 r161, {rs14, rs14};
{
mul.f16x2 r159, r31, r161;
}
{
add.f16x2 r162, r138, r159;
}
mov.f64 fd215, 0dBFE4F49E7F775887;
{
cvt.rn.f16.f64 rs15, fd215;
}
mov.b32 r167, {rs15, rs15};
{
mul.f16x2 r165, r37, r167;
}
{
add.f16x2 r168, r144, r165;
}
mov.f64 fd216, 0dBFE82F19BB3A28A1;
{
cvt.rn.f16.f64 rs16, fd216;
}
mov.b32 r173, {rs16, rs16};
{
mul.f16x2 r171, r46, r173;
}
{
add.f16x2 r174, r150, r171;
}
{
cvt.rn.f16.f64 rs17, fd215;
}
mov.b32 r179, {rs17, rs17};
{
mul.f16x2 r177, r40, r179;
}
{
add.f16x2 r180, r156, r177;
}
{
cvt.rn.f16.f64 rs18, fd216;
}
mov.b32 r185, {rs18, rs18};
{
mul.f16x2 r183, r43, r185;
}
{
add.f16x2 r186, r162, r183;
}
mov.f64 fd219, 0dBFEEB42A9BCD5057;
{
cvt.rn.f16.f64 rs19, fd219;
}
mov.b32 r191, {rs19, rs19};
{
mul.f16x2 r189, r49, r191;
}
{
add.f16x2 r192, r168, r189;
}
mov.f64 fd220, 0dBFD207E7FD768DBF;
{
cvt.rn.f16.f64 rs20, fd220;
}
mov.b32 r197, {rs20, rs20};
{
mul.f16x2 r195, r58, r197;
}
{
add.f16x2 r198, r174, r195;
}
{
cvt.rn.f16.f64 rs21, fd219;
}
mov.b32 r203, {rs21, rs21};
{
mul.f16x2 r201, r52, r203;
}
{
add.f16x2 r204, r180, r201;
}
{
cvt.rn.f16.f64 rs22, fd220;
}
mov.b32 r209, {rs22, rs22};
{
mul.f16x2 r207, r55, r209;
}
{
add.f16x2 r210, r186, r207;
}
{
sub.f16x2 r213, r192, r198;
}
{
add.f16x2 r216, r204, r210;
}
{
add.f16x2 r219, r192, r198;
}
{
sub.f16x2 r222, r204, r210;
}
cvt.rn.f16.s32 rs23, r1388;
mov.b32 r237, {rs23, rs23};
cvt.rn.f16.s32 rs24, r1388;
mov.b32 r249, {rs24, rs24};
{
cvt.rn.f16.f64 rs25, fd207;
}
mov.b32 r229, {rs25, rs25};
{
mul.f16x2 r227, r1, r229;
}
{
add.f16x2 r230, %44, r227;
}
{
cvt.rn.f16.f64 rs26, fd214;
}
mov.b32 r235, {rs26, rs26};
{
mul.f16x2 r233, r10, r235;
}
{
add.f16x2 r236, r237, r233;
}
{
cvt.rn.f16.f64 rs27, fd207;
}
mov.b32 r241, {rs27, rs27};
{
mul.f16x2 r239, r4, r241;
}
{
add.f16x2 r242, %45, r239;
}
{
cvt.rn.f16.f64 rs28, fd214;
}
mov.b32 r247, {rs28, rs28};
{
mul.f16x2 r245, r7, r247;
}
{
add.f16x2 r248, r249, r245;
}
{
cvt.rn.f16.f64 rs29, fd215;
}
mov.b32 r253, {rs29, rs29};
{
mul.f16x2 r251, r13, r253;
}
{
add.f16x2 r254, r230, r251;
}
{
cvt.rn.f16.f64 rs30, fd216;
}
mov.b32 r259, {rs30, rs30};
{
mul.f16x2 r257, r22, r259;
}
{
add.f16x2 r260, r236, r257;
}
{
cvt.rn.f16.f64 rs31, fd215;
}
mov.b32 r265, {rs31, rs31};
{
mul.f16x2 r263, r16, r265;
}
{
add.f16x2 r266, r242, r263;
}
{
cvt.rn.f16.f64 rs32, fd216;
}
mov.b32 r271, {rs32, rs32};
{
mul.f16x2 r269, r19, r271;
}
{
add.f16x2 r272, r248, r269;
}
{
cvt.rn.f16.f64 rs33, fd219;
}
mov.b32 r277, {rs33, rs33};
{
mul.f16x2 r275, r25, r277;
}
{
add.f16x2 r278, r254, r275;
}
mov.f64 fd148, 0d3FD207E7FD768DBF;
{
cvt.rn.f16.f64 rs34, fd148;
}
mov.b32 r283, {rs34, rs34};
{
mul.f16x2 r281, r34, r283;
}
{
add.f16x2 r284, r260, r281;
}
{
cvt.rn.f16.f64 rs35, fd219;
}
mov.b32 r289, {rs35, rs35};
{
mul.f16x2 r287, r28, r289;
}
{
add.f16x2 r290, r266, r287;
}
{
cvt.rn.f16.f64 rs36, fd148;
}
mov.b32 r295, {rs36, rs36};
{
mul.f16x2 r293, r31, r295;
}
{
add.f16x2 r296, r272, r293;
}
{
cvt.rn.f16.f64 rs37, fd211;
}
mov.b32 r301, {rs37, rs37};
{
mul.f16x2 r299, r37, r301;
}
{
add.f16x2 r302, r278, r299;
}
mov.f64 fd168, 0d3FEFAC9E043842EF;
{
cvt.rn.f16.f64 rs38, fd168;
}
mov.b32 r307, {rs38, rs38};
{
mul.f16x2 r305, r46, r307;
}
{
add.f16x2 r308, r284, r305;
}
{
cvt.rn.f16.f64 rs39, fd211;
}
mov.b32 r313, {rs39, rs39};
{
mul.f16x2 r311, r40, r313;
}
{
add.f16x2 r314, r290, r311;
}
{
cvt.rn.f16.f64 rs40, fd168;
}
mov.b32 r319, {rs40, rs40};
{
mul.f16x2 r317, r43, r319;
}
{
add.f16x2 r320, r296, r317;
}
{
cvt.rn.f16.f64 rs41, fd203;
}
mov.b32 r325, {rs41, rs41};
{
mul.f16x2 r323, r49, r325;
}
{
add.f16x2 r326, r302, r323;
}
mov.f64 fd188, 0d3FE14CEDF8BB580B;
{
cvt.rn.f16.f64 rs42, fd188;
}
mov.b32 r331, {rs42, rs42};
{
mul.f16x2 r329, r58, r331;
}
{
add.f16x2 r332, r308, r329;
}
{
cvt.rn.f16.f64 rs43, fd203;
}
mov.b32 r337, {rs43, rs43};
{
mul.f16x2 r335, r52, r337;
}
{
add.f16x2 r338, r314, r335;
}
{
cvt.rn.f16.f64 rs44, fd188;
}
mov.b32 r343, {rs44, rs44};
{
mul.f16x2 r341, r55, r343;
}
{
add.f16x2 r344, r320, r341;
}
{
sub.f16x2 r347, r326, r332;
}
{
add.f16x2 r350, r338, r344;
}
{
add.f16x2 r353, r326, r332;
}
{
sub.f16x2 r356, r338, r344;
}
cvt.rn.f16.s32 rs45, r1388;
mov.b32 r371, {rs45, rs45};
cvt.rn.f16.s32 rs46, r1388;
mov.b32 r383, {rs46, rs46};
{
cvt.rn.f16.f64 rs47, fd211;
}
mov.b32 r363, {rs47, rs47};
{
mul.f16x2 r361, r1, r363;
}
{
add.f16x2 r364, %44, r361;
}
{
cvt.rn.f16.f64 rs48, fd212;
}
mov.b32 r369, {rs48, rs48};
{
mul.f16x2 r367, r10, r369;
}
{
add.f16x2 r370, r371, r367;
}
{
cvt.rn.f16.f64 rs49, fd211;
}
mov.b32 r375, {rs49, rs49};
{
mul.f16x2 r373, r4, r375;
}
{
add.f16x2 r376, %45, r373;
}
{
cvt.rn.f16.f64 rs50, fd212;
}
mov.b32 r381, {rs50, rs50};
{
mul.f16x2 r379, r7, r381;
}
{
add.f16x2 r382, r383, r379;
}
{
cvt.rn.f16.f64 rs51, fd219;
}
mov.b32 r387, {rs51, rs51};
{
mul.f16x2 r385, r13, r387;
}
{
add.f16x2 r388, r364, r385;
}
{
cvt.rn.f16.f64 rs52, fd148;
}
mov.b32 r393, {rs52, rs52};
{
mul.f16x2 r391, r22, r393;
}
{
add.f16x2 r394, r370, r391;
}
{
cvt.rn.f16.f64 rs53, fd219;
}
mov.b32 r399, {rs53, rs53};
{
mul.f16x2 r397, r16, r399;
}
{
add.f16x2 r400, r376, r397;
}
{
cvt.rn.f16.f64 rs54, fd148;
}
mov.b32 r405, {rs54, rs54};
{
mul.f16x2 r403, r19, r405;
}
{
add.f16x2 r406, r382, r403;
}
{
cvt.rn.f16.f64 rs55, fd207;
}
mov.b32 r411, {rs55, rs55};
{
mul.f16x2 r409, r25, r411;
}
{
add.f16x2 r412, r388, r409;
}
mov.f64 fd196, 0d3FED1BB48EEE2C13;
{
cvt.rn.f16.f64 rs56, fd196;
}
mov.b32 r417, {rs56, rs56};
{
mul.f16x2 r415, r34, r417;
}
{
add.f16x2 r418, r394, r415;
}
{
cvt.rn.f16.f64 rs57, fd207;
}
mov.b32 r423, {rs57, rs57};
{
mul.f16x2 r421, r28, r423;
}
{
add.f16x2 r424, r400, r421;
}
{
cvt.rn.f16.f64 rs58, fd196;
}
mov.b32 r429, {rs58, rs58};
{
mul.f16x2 r427, r31, r429;
}
{
add.f16x2 r430, r406, r427;
}
{
cvt.rn.f16.f64 rs59, fd203;
}
mov.b32 r435, {rs59, rs59};
{
mul.f16x2 r433, r37, r435;
}
{
add.f16x2 r436, r412, r433;
}
{
cvt.rn.f16.f64 rs60, fd218;
}
mov.b32 r441, {rs60, rs60};
{
mul.f16x2 r439, r46, r441;
}
{
add.f16x2 r442, r418, r439;
}
{
cvt.rn.f16.f64 rs61, fd203;
}
mov.b32 r447, {rs61, rs61};
{
mul.f16x2 r445, r40, r447;
}
{
add.f16x2 r448, r424, r445;
}
{
cvt.rn.f16.f64 rs62, fd218;
}
mov.b32 r453, {rs62, rs62};
{
mul.f16x2 r451, r43, r453;
}
{
add.f16x2 r454, r430, r451;
}
{
cvt.rn.f16.f64 rs63, fd215;
}
mov.b32 r459, {rs63, rs63};
{
mul.f16x2 r457, r49, r459;
}
{
add.f16x2 r460, r436, r457;
}
{
cvt.rn.f16.f64 rs64, fd216;
}
mov.b32 r465, {rs64, rs64};
{
mul.f16x2 r463, r58, r465;
}
{
add.f16x2 r466, r442, r463;
}
{
cvt.rn.f16.f64 rs65, fd215;
}
mov.b32 r471, {rs65, rs65};
{
mul.f16x2 r469, r52, r471;
}
{
add.f16x2 r472, r448, r469;
}
{
cvt.rn.f16.f64 rs66, fd216;
}
mov.b32 r477, {rs66, rs66};
{
mul.f16x2 r475, r55, r477;
}
{
add.f16x2 r478, r454, r475;
}
{
sub.f16x2 r481, r460, r466;
}
{
add.f16x2 r484, r472, r478;
}
{
add.f16x2 r487, r460, r466;
}
{
sub.f16x2 r490, r472, r478;
}
cvt.rn.f16.s32 rs67, r1388;
mov.b32 r505, {rs67, rs67};
cvt.rn.f16.s32 rs68, r1388;
mov.b32 r517, {rs68, rs68};
{
cvt.rn.f16.f64 rs69, fd215;
}
mov.b32 r497, {rs69, rs69};
{
mul.f16x2 r495, r1, r497;
}
{
add.f16x2 r498, %44, r495;
}
{
cvt.rn.f16.f64 rs70, fd216;
}
mov.b32 r503, {rs70, rs70};
{
mul.f16x2 r501, r10, r503;
}
{
add.f16x2 r504, r505, r501;
}
{
cvt.rn.f16.f64 rs71, fd215;
}
mov.b32 r509, {rs71, rs71};
{
mul.f16x2 r507, r4, r509;
}
{
add.f16x2 r510, %45, r507;
}
{
cvt.rn.f16.f64 rs72, fd216;
}
mov.b32 r515, {rs72, rs72};
{
mul.f16x2 r513, r7, r515;
}
{
add.f16x2 r516, r517, r513;
}
{
cvt.rn.f16.f64 rs73, fd211;
}
mov.b32 r521, {rs73, rs73};
{
mul.f16x2 r519, r13, r521;
}
{
add.f16x2 r522, r498, r519;
}
{
cvt.rn.f16.f64 rs74, fd168;
}
mov.b32 r527, {rs74, rs74};
{
mul.f16x2 r525, r22, r527;
}
{
add.f16x2 r528, r504, r525;
}
{
cvt.rn.f16.f64 rs75, fd211;
}
mov.b32 r533, {rs75, rs75};
{
mul.f16x2 r531, r16, r533;
}
{
add.f16x2 r534, r510, r531;
}
{
cvt.rn.f16.f64 rs76, fd168;
}
mov.b32 r539, {rs76, rs76};
{
mul.f16x2 r537, r19, r539;
}
{
add.f16x2 r540, r516, r537;
}
{
cvt.rn.f16.f64 rs77, fd203;
}
mov.b32 r545, {rs77, rs77};
{
mul.f16x2 r543, r25, r545;
}
{
add.f16x2 r546, r522, r543;
}
{
cvt.rn.f16.f64 rs78, fd218;
}
mov.b32 r551, {rs78, rs78};
{
mul.f16x2 r549, r34, r551;
}
{
add.f16x2 r552, r528, r549;
}
{
cvt.rn.f16.f64 rs79, fd203;
}
mov.b32 r557, {rs79, rs79};
{
mul.f16x2 r555, r28, r557;
}
{
add.f16x2 r558, r534, r555;
}
{
cvt.rn.f16.f64 rs80, fd218;
}
mov.b32 r563, {rs80, rs80};
{
mul.f16x2 r561, r31, r563;
}
{
add.f16x2 r564, r540, r561;
}
{
cvt.rn.f16.f64 rs81, fd219;
}
mov.b32 r569, {rs81, rs81};
{
mul.f16x2 r567, r37, r569;
}
{
add.f16x2 r570, r546, r567;
}
{
cvt.rn.f16.f64 rs82, fd220;
}
mov.b32 r575, {rs82, rs82};
{
mul.f16x2 r573, r46, r575;
}
{
add.f16x2 r576, r552, r573;
}
{
cvt.rn.f16.f64 rs83, fd219;
}
mov.b32 r581, {rs83, rs83};
{
mul.f16x2 r579, r40, r581;
}
{
add.f16x2 r582, r558, r579;
}
{
cvt.rn.f16.f64 rs84, fd220;
}
mov.b32 r587, {rs84, rs84};
{
mul.f16x2 r585, r43, r587;
}
{
add.f16x2 r588, r564, r585;
}
{
cvt.rn.f16.f64 rs85, fd207;
}
mov.b32 r593, {rs85, rs85};
{
mul.f16x2 r591, r49, r593;
}
{
add.f16x2 r594, r570, r591;
}
{
cvt.rn.f16.f64 rs86, fd196;
}
mov.b32 r599, {rs86, rs86};
{
mul.f16x2 r597, r58, r599;
}
{
add.f16x2 r600, r576, r597;
}
{
cvt.rn.f16.f64 rs87, fd207;
}
mov.b32 r605, {rs87, rs87};
{
mul.f16x2 r603, r52, r605;
}
{
add.f16x2 r606, r582, r603;
}
{
cvt.rn.f16.f64 rs88, fd196;
}
mov.b32 r611, {rs88, rs88};
{
mul.f16x2 r609, r55, r611;
}
{
add.f16x2 r612, r588, r609;
}
{
sub.f16x2 r615, r594, r600;
}
{
add.f16x2 r618, r606, r612;
}
{
add.f16x2 r621, r594, r600;
}
{
sub.f16x2 r624, r606, r612;
}
cvt.rn.f16.s32 rs89, r1388;
mov.b32 r639, {rs89, rs89};
cvt.rn.f16.s32 rs90, r1388;
mov.b32 r651, {rs90, rs90};
{
cvt.rn.f16.f64 rs91, fd219;
}
mov.b32 r631, {rs91, rs91};
{
mul.f16x2 r629, r1, r631;
}
{
add.f16x2 r632, %44, r629;
}
{
cvt.rn.f16.f64 rs92, fd220;
}
mov.b32 r637, {rs92, rs92};
{
mul.f16x2 r635, r10, r637;
}
{
add.f16x2 r638, r639, r635;
}
{
cvt.rn.f16.f64 rs93, fd219;
}
mov.b32 r643, {rs93, rs93};
{
mul.f16x2 r641, r4, r643;
}
{
add.f16x2 r644, %45, r641;
}
{
cvt.rn.f16.f64 rs94, fd220;
}
mov.b32 r649, {rs94, rs94};
{
mul.f16x2 r647, r7, r649;
}
{
add.f16x2 r650, r651, r647;
}
{
cvt.rn.f16.f64 rs95, fd203;
}
mov.b32 r655, {rs95, rs95};
{
mul.f16x2 r653, r13, r655;
}
{
add.f16x2 r656, r632, r653;
}
{
cvt.rn.f16.f64 rs96, fd188;
}
mov.b32 r661, {rs96, rs96};
{
mul.f16x2 r659, r22, r661;
}
{
add.f16x2 r662, r638, r659;
}
{
cvt.rn.f16.f64 rs97, fd203;
}
mov.b32 r667, {rs97, rs97};
{
mul.f16x2 r665, r16, r667;
}
{
add.f16x2 r668, r644, r665;
}
{
cvt.rn.f16.f64 rs98, fd188;
}
mov.b32 r673, {rs98, rs98};
{
mul.f16x2 r671, r19, r673;
}
{
add.f16x2 r674, r650, r671;
}
{
cvt.rn.f16.f64 rs99, fd215;
}
mov.b32 r679, {rs99, rs99};
{
mul.f16x2 r677, r25, r679;
}
{
add.f16x2 r680, r656, r677;
}
{
cvt.rn.f16.f64 rs100, fd216;
}
mov.b32 r685, {rs100, rs100};
{
mul.f16x2 r683, r34, r685;
}
{
add.f16x2 r686, r662, r683;
}
{
cvt.rn.f16.f64 rs101, fd215;
}
mov.b32 r691, {rs101, rs101};
{
mul.f16x2 r689, r28, r691;
}
{
add.f16x2 r692, r668, r689;
}
{
cvt.rn.f16.f64 rs102, fd216;
}
mov.b32 r697, {rs102, rs102};
{
mul.f16x2 r695, r31, r697;
}
{
add.f16x2 r698, r674, r695;
}
{
cvt.rn.f16.f64 rs103, fd207;
}
mov.b32 r703, {rs103, rs103};
{
mul.f16x2 r701, r37, r703;
}
{
add.f16x2 r704, r680, r701;
}
{
cvt.rn.f16.f64 rs104, fd196;
}
mov.b32 r709, {rs104, rs104};
{
mul.f16x2 r707, r46, r709;
}
{
add.f16x2 r710, r686, r707;
}
{
cvt.rn.f16.f64 rs105, fd207;
}
mov.b32 r715, {rs105, rs105};
{
mul.f16x2 r713, r40, r715;
}
{
add.f16x2 r716, r692, r713;
}
{
cvt.rn.f16.f64 rs106, fd196;
}
mov.b32 r721, {rs106, rs106};
{
mul.f16x2 r719, r43, r721;
}
{
add.f16x2 r722, r698, r719;
}
{
cvt.rn.f16.f64 rs107, fd211;
}
mov.b32 r727, {rs107, rs107};
{
mul.f16x2 r725, r49, r727;
}
{
add.f16x2 r728, r704, r725;
}
{
cvt.rn.f16.f64 rs108, fd212;
}
mov.b32 r733, {rs108, rs108};
{
mul.f16x2 r731, r58, r733;
}
{
add.f16x2 r734, r710, r731;
}
{
cvt.rn.f16.f64 rs109, fd211;
}
mov.b32 r739, {rs109, rs109};
{
mul.f16x2 r737, r52, r739;
}
{
add.f16x2 r740, r716, r737;
}
{
cvt.rn.f16.f64 rs110, fd212;
}
mov.b32 r745, {rs110, rs110};
{
mul.f16x2 r743, r55, r745;
}
{
add.f16x2 r746, r722, r743;
}
{
sub.f16x2 r749, r728, r734;
}
{
add.f16x2 r752, r740, r746;
}
{
add.f16x2 r755, r728, r734;
}
{
sub.f16x2 r758, r740, r746;
}
{
add.f16x2 r761, %74, %66;
}
{
add.f16x2 r764, %77, %71;
}
{
sub.f16x2 r767, %74, %66;
}
{
sub.f16x2 r770, %77, %71;
}
{
add.f16x2 r773, %84, %80;
}
{
add.f16x2 r776, %87, %82;
}
{
sub.f16x2 r779, %84, %80;
}
{
sub.f16x2 r782, %87, %82;
}
{
add.f16x2 r785, %75, %68;
}
{
add.f16x2 r788, %78, %72;
}
{
sub.f16x2 r791, %75, %68;
}
{
sub.f16x2 r794, %78, %72;
}
{
add.f16x2 r797, %86, %81;
}
{
add.f16x2 r800, %67, %83;
}
{
sub.f16x2 r803, %86, %81;
}
{
sub.f16x2 r806, %67, %83;
}
{
add.f16x2 r809, %76, %70;
}
{
add.f16x2 r812, %79, %73;
}
{
sub.f16x2 r815, %76, %70;
}
{
sub.f16x2 r818, %79, %73;
}
{
add.f16x2 r821, %85, r761;
}
{
add.f16x2 r824, %69, r764;
}
{
add.f16x2 r827, r821, r773;
}
{
add.f16x2 r830, r824, r776;
}
{
add.f16x2 r833, r827, r785;
}
{
add.f16x2 r836, r830, r788;
}
{
add.f16x2 r839, r833, r797;
}
{
add.f16x2 r842, r836, r800;
}
{
add.f16x2 r845, r839, r809;
}
{
add.f16x2 r848, r842, r812;
}
cvt.rn.f16.s32 rs111, r1388;
mov.b32 r863, {rs111, rs111};
cvt.rn.f16.s32 rs112, r1388;
mov.b32 r875, {rs112, rs112};
{
cvt.rn.f16.f64 rs113, fd203;
}
mov.b32 r855, {rs113, rs113};
{
mul.f16x2 r853, r761, r855;
}
{
add.f16x2 r856, %85, r853;
}
{
cvt.rn.f16.f64 rs114, fd218;
}
mov.b32 r861, {rs114, rs114};
{
mul.f16x2 r859, r770, r861;
}
{
add.f16x2 r862, r863, r859;
}
{
cvt.rn.f16.f64 rs115, fd203;
}
mov.b32 r867, {rs115, rs115};
{
mul.f16x2 r865, r764, r867;
}
{
add.f16x2 r868, %69, r865;
}
{
cvt.rn.f16.f64 rs116, fd218;
}
mov.b32 r873, {rs116, rs116};
{
mul.f16x2 r871, r767, r873;
}
{
add.f16x2 r874, r875, r871;
}
{
cvt.rn.f16.f64 rs117, fd207;
}
mov.b32 r879, {rs117, rs117};
{
mul.f16x2 r877, r773, r879;
}
{
add.f16x2 r880, r856, r877;
}
{
cvt.rn.f16.f64 rs118, fd214;
}
mov.b32 r885, {rs118, rs118};
{
mul.f16x2 r883, r782, r885;
}
{
add.f16x2 r886, r862, r883;
}
{
cvt.rn.f16.f64 rs119, fd207;
}
mov.b32 r891, {rs119, rs119};
{
mul.f16x2 r889, r776, r891;
}
{
add.f16x2 r892, r868, r889;
}
{
cvt.rn.f16.f64 rs120, fd214;
}
mov.b32 r897, {rs120, rs120};
{
mul.f16x2 r895, r779, r897;
}
{
add.f16x2 r898, r874, r895;
}
{
cvt.rn.f16.f64 rs121, fd211;
}
mov.b32 r903, {rs121, rs121};
{
mul.f16x2 r901, r785, r903;
}
{
add.f16x2 r904, r880, r901;
}
{
cvt.rn.f16.f64 rs122, fd212;
}
mov.b32 r909, {rs122, rs122};
{
mul.f16x2 r907, r794, r909;
}
{
add.f16x2 r910, r886, r907;
}
{
cvt.rn.f16.f64 rs123, fd211;
}
mov.b32 r915, {rs123, rs123};
{
mul.f16x2 r913, r788, r915;
}
{
add.f16x2 r916, r892, r913;
}
{
cvt.rn.f16.f64 rs124, fd212;
}
mov.b32 r921, {rs124, rs124};
{
mul.f16x2 r919, r791, r921;
}
{
add.f16x2 r922, r898, r919;
}
{
cvt.rn.f16.f64 rs125, fd215;
}
mov.b32 r927, {rs125, rs125};
{
mul.f16x2 r925, r797, r927;
}
{
add.f16x2 r928, r904, r925;
}
{
cvt.rn.f16.f64 rs126, fd216;
}
mov.b32 r933, {rs126, rs126};
{
mul.f16x2 r931, r806, r933;
}
{
add.f16x2 r934, r910, r931;
}
{
cvt.rn.f16.f64 rs127, fd215;
}
mov.b32 r939, {rs127, rs127};
{
mul.f16x2 r937, r800, r939;
}
{
add.f16x2 r940, r916, r937;
}
{
cvt.rn.f16.f64 rs128, fd216;
}
mov.b32 r945, {rs128, rs128};
{
mul.f16x2 r943, r803, r945;
}
{
add.f16x2 r946, r922, r943;
}
{
cvt.rn.f16.f64 rs129, fd219;
}
mov.b32 r951, {rs129, rs129};
{
mul.f16x2 r949, r809, r951;
}
{
add.f16x2 r952, r928, r949;
}
{
cvt.rn.f16.f64 rs130, fd220;
}
mov.b32 r957, {rs130, rs130};
{
mul.f16x2 r955, r818, r957;
}
{
add.f16x2 r958, r934, r955;
}
{
cvt.rn.f16.f64 rs131, fd219;
}
mov.b32 r963, {rs131, rs131};
{
mul.f16x2 r961, r812, r963;
}
{
add.f16x2 r964, r940, r961;
}
{
cvt.rn.f16.f64 rs132, fd220;
}
mov.b32 r969, {rs132, rs132};
{
mul.f16x2 r967, r815, r969;
}
{
add.f16x2 r970, r946, r967;
}
{
sub.f16x2 r973, r952, r958;
}
{
add.f16x2 r976, r964, r970;
}
{
add.f16x2 r979, r952, r958;
}
{
sub.f16x2 r982, r964, r970;
}
cvt.rn.f16.s32 rs133, r1388;
mov.b32 r997, {rs133, rs133};
cvt.rn.f16.s32 rs134, r1388;
mov.b32 r1009, {rs134, rs134};
{
cvt.rn.f16.f64 rs135, fd207;
}
mov.b32 r989, {rs135, rs135};
{
mul.f16x2 r987, r761, r989;
}
{
add.f16x2 r990, %85, r987;
}
{
cvt.rn.f16.f64 rs136, fd214;
}
mov.b32 r995, {rs136, rs136};
{
mul.f16x2 r993, r770, r995;
}
{
add.f16x2 r996, r997, r993;
}
{
cvt.rn.f16.f64 rs137, fd207;
}
mov.b32 r1001, {rs137, rs137};
{
mul.f16x2 r999, r764, r1001;
}
{
add.f16x2 r1002, %69, r999;
}
{
cvt.rn.f16.f64 rs138, fd214;
}
mov.b32 r1007, {rs138, rs138};
{
mul.f16x2 r1005, r767, r1007;
}
{
add.f16x2 r1008, r1009, r1005;
}
{
cvt.rn.f16.f64 rs139, fd215;
}
mov.b32 r1013, {rs139, rs139};
{
mul.f16x2 r1011, r773, r1013;
}
{
add.f16x2 r1014, r990, r1011;
}
{
cvt.rn.f16.f64 rs140, fd216;
}
mov.b32 r1019, {rs140, rs140};
{
mul.f16x2 r1017, r782, r1019;
}
{
add.f16x2 r1020, r996, r1017;
}
{
cvt.rn.f16.f64 rs141, fd215;
}
mov.b32 r1025, {rs141, rs141};
{
mul.f16x2 r1023, r776, r1025;
}
{
add.f16x2 r1026, r1002, r1023;
}
{
cvt.rn.f16.f64 rs142, fd216;
}
mov.b32 r1031, {rs142, rs142};
{
mul.f16x2 r1029, r779, r1031;
}
{
add.f16x2 r1032, r1008, r1029;
}
{
cvt.rn.f16.f64 rs143, fd219;
}
mov.b32 r1037, {rs143, rs143};
{
mul.f16x2 r1035, r785, r1037;
}
{
add.f16x2 r1038, r1014, r1035;
}
{
cvt.rn.f16.f64 rs144, fd148;
}
mov.b32 r1043, {rs144, rs144};
{
mul.f16x2 r1041, r794, r1043;
}
{
add.f16x2 r1044, r1020, r1041;
}
{
cvt.rn.f16.f64 rs145, fd219;
}
mov.b32 r1049, {rs145, rs145};
{
mul.f16x2 r1047, r788, r1049;
}
{
add.f16x2 r1050, r1026, r1047;
}
{
cvt.rn.f16.f64 rs146, fd148;
}
mov.b32 r1055, {rs146, rs146};
{
mul.f16x2 r1053, r791, r1055;
}
{
add.f16x2 r1056, r1032, r1053;
}
{
cvt.rn.f16.f64 rs147, fd211;
}
mov.b32 r1061, {rs147, rs147};
{
mul.f16x2 r1059, r797, r1061;
}
{
add.f16x2 r1062, r1038, r1059;
}
{
cvt.rn.f16.f64 rs148, fd168;
}
mov.b32 r1067, {rs148, rs148};
{
mul.f16x2 r1065, r806, r1067;
}
{
add.f16x2 r1068, r1044, r1065;
}
{
cvt.rn.f16.f64 rs149, fd211;
}
mov.b32 r1073, {rs149, rs149};
{
mul.f16x2 r1071, r800, r1073;
}
{
add.f16x2 r1074, r1050, r1071;
}
{
cvt.rn.f16.f64 rs150, fd168;
}
mov.b32 r1079, {rs150, rs150};
{
mul.f16x2 r1077, r803, r1079;
}
{
add.f16x2 r1080, r1056, r1077;
}
{
cvt.rn.f16.f64 rs151, fd203;
}
mov.b32 r1085, {rs151, rs151};
{
mul.f16x2 r1083, r809, r1085;
}
{
add.f16x2 r1086, r1062, r1083;
}
{
cvt.rn.f16.f64 rs152, fd188;
}
mov.b32 r1091, {rs152, rs152};
{
mul.f16x2 r1089, r818, r1091;
}
{
add.f16x2 r1092, r1068, r1089;
}
{
cvt.rn.f16.f64 rs153, fd203;
}
mov.b32 r1097, {rs153, rs153};
{
mul.f16x2 r1095, r812, r1097;
}
{
add.f16x2 r1098, r1074, r1095;
}
{
cvt.rn.f16.f64 rs154, fd188;
}
mov.b32 r1103, {rs154, rs154};
{
mul.f16x2 r1101, r815, r1103;
}
{
add.f16x2 r1104, r1080, r1101;
}
{
sub.f16x2 r1107, r1086, r1092;
}
{
add.f16x2 r1110, r1098, r1104;
}
{
add.f16x2 r1113, r1086, r1092;
}
{
sub.f16x2 r1116, r1098, r1104;
}
cvt.rn.f16.s32 rs155, r1388;
mov.b32 r1131, {rs155, rs155};
cvt.rn.f16.s32 rs156, r1388;
mov.b32 r1143, {rs156, rs156};
{
cvt.rn.f16.f64 rs157, fd211;
}
mov.b32 r1123, {rs157, rs157};
{
mul.f16x2 r1121, r761, r1123;
}
{
add.f16x2 r1124, %85, r1121;
}
{
cvt.rn.f16.f64 rs158, fd212;
}
mov.b32 r1129, {rs158, rs158};
{
mul.f16x2 r1127, r770, r1129;
}
{
add.f16x2 r1130, r1131, r1127;
}
{
cvt.rn.f16.f64 rs159, fd211;
}
mov.b32 r1135, {rs159, rs159};
{
mul.f16x2 r1133, r764, r1135;
}
{
add.f16x2 r1136, %69, r1133;
}
{
cvt.rn.f16.f64 rs160, fd212;
}
mov.b32 r1141, {rs160, rs160};
{
mul.f16x2 r1139, r767, r1141;
}
{
add.f16x2 r1142, r1143, r1139;
}
{
cvt.rn.f16.f64 rs161, fd219;
}
mov.b32 r1147, {rs161, rs161};
{
mul.f16x2 r1145, r773, r1147;
}
{
add.f16x2 r1148, r1124, r1145;
}
{
cvt.rn.f16.f64 rs162, fd148;
}
mov.b32 r1153, {rs162, rs162};
{
mul.f16x2 r1151, r782, r1153;
}
{
add.f16x2 r1154, r1130, r1151;
}
{
cvt.rn.f16.f64 rs163, fd219;
}
mov.b32 r1159, {rs163, rs163};
{
mul.f16x2 r1157, r776, r1159;
}
{
add.f16x2 r1160, r1136, r1157;
}
{
cvt.rn.f16.f64 rs164, fd148;
}
mov.b32 r1165, {rs164, rs164};
{
mul.f16x2 r1163, r779, r1165;
}
{
add.f16x2 r1166, r1142, r1163;
}
{
cvt.rn.f16.f64 rs165, fd207;
}
mov.b32 r1171, {rs165, rs165};
{
mul.f16x2 r1169, r785, r1171;
}
{
add.f16x2 r1172, r1148, r1169;
}
{
cvt.rn.f16.f64 rs166, fd196;
}
mov.b32 r1177, {rs166, rs166};
{
mul.f16x2 r1175, r794, r1177;
}
{
add.f16x2 r1178, r1154, r1175;
}
{
cvt.rn.f16.f64 rs167, fd207;
}
mov.b32 r1183, {rs167, rs167};
{
mul.f16x2 r1181, r788, r1183;
}
{
add.f16x2 r1184, r1160, r1181;
}
{
cvt.rn.f16.f64 rs168, fd196;
}
mov.b32 r1189, {rs168, rs168};
{
mul.f16x2 r1187, r791, r1189;
}
{
add.f16x2 r1190, r1166, r1187;
}
{
cvt.rn.f16.f64 rs169, fd203;
}
mov.b32 r1195, {rs169, rs169};
{
mul.f16x2 r1193, r797, r1195;
}
{
add.f16x2 r1196, r1172, r1193;
}
{
cvt.rn.f16.f64 rs170, fd218;
}
mov.b32 r1201, {rs170, rs170};
{
mul.f16x2 r1199, r806, r1201;
}
{
add.f16x2 r1202, r1178, r1199;
}
{
cvt.rn.f16.f64 rs171, fd203;
}
mov.b32 r1207, {rs171, rs171};
{
mul.f16x2 r1205, r800, r1207;
}
{
add.f16x2 r1208, r1184, r1205;
}
{
cvt.rn.f16.f64 rs172, fd218;
}
mov.b32 r1213, {rs172, rs172};
{
mul.f16x2 r1211, r803, r1213;
}
{
add.f16x2 r1214, r1190, r1211;
}
{
cvt.rn.f16.f64 rs173, fd215;
}
mov.b32 r1219, {rs173, rs173};
{
mul.f16x2 r1217, r809, r1219;
}
{
add.f16x2 r1220, r1196, r1217;
}
{
cvt.rn.f16.f64 rs174, fd216;
}
mov.b32 r1225, {rs174, rs174};
{
mul.f16x2 r1223, r818, r1225;
}
{
add.f16x2 r1226, r1202, r1223;
}
{
cvt.rn.f16.f64 rs175, fd215;
}
mov.b32 r1231, {rs175, rs175};
{
mul.f16x2 r1229, r812, r1231;
}
{
add.f16x2 r1232, r1208, r1229;
}
{
cvt.rn.f16.f64 rs176, fd216;
}
mov.b32 r1237, {rs176, rs176};
{
mul.f16x2 r1235, r815, r1237;
}
{
add.f16x2 r1238, r1214, r1235;
}
{
sub.f16x2 r1241, r1220, r1226;
}
{
add.f16x2 r1244, r1232, r1238;
}
{
add.f16x2 r1247, r1220, r1226;
}
{
sub.f16x2 r1250, r1232, r1238;
}
cvt.rn.f16.s32 rs177, r1388;
mov.b32 r1265, {rs177, rs177};
cvt.rn.f16.s32 rs178, r1388;
mov.b32 r1277, {rs178, rs178};
{
cvt.rn.f16.f64 rs179, fd215;
}
mov.b32 r1257, {rs179, rs179};
{
mul.f16x2 r1255, r761, r1257;
}
{
add.f16x2 r1258, %85, r1255;
}
{
cvt.rn.f16.f64 rs180, fd216;
}
mov.b32 r1263, {rs180, rs180};
{
mul.f16x2 r1261, r770, r1263;
}
{
add.f16x2 r1264, r1265, r1261;
}
{
cvt.rn.f16.f64 rs181, fd215;
}
mov.b32 r1269, {rs181, rs181};
{
mul.f16x2 r1267, r764, r1269;
}
{
add.f16x2 r1270, %69, r1267;
}
{
cvt.rn.f16.f64 rs182, fd216;
}
mov.b32 r1275, {rs182, rs182};
{
mul.f16x2 r1273, r767, r1275;
}
{
add.f16x2 r1276, r1277, r1273;
}
{
cvt.rn.f16.f64 rs183, fd211;
}
mov.b32 r1281, {rs183, rs183};
{
mul.f16x2 r1279, r773, r1281;
}
{
add.f16x2 r1282, r1258, r1279;
}
{
cvt.rn.f16.f64 rs184, fd168;
}
mov.b32 r1287, {rs184, rs184};
{
mul.f16x2 r1285, r782, r1287;
}
{
add.f16x2 r1288, r1264, r1285;
}
{
cvt.rn.f16.f64 rs185, fd211;
}
mov.b32 r1293, {rs185, rs185};
{
mul.f16x2 r1291, r776, r1293;
}
{
add.f16x2 r1294, r1270, r1291;
}
{
cvt.rn.f16.f64 rs186, fd168;
}
mov.b32 r1299, {rs186, rs186};
{
mul.f16x2 r1297, r779, r1299;
}
{
add.f16x2 r1300, r1276, r1297;
}
{
cvt.rn.f16.f64 rs187, fd203;
}
mov.b32 r1305, {rs187, rs187};
{
mul.f16x2 r1303, r785, r1305;
}
{
add.f16x2 r1306, r1282, r1303;
}
{
cvt.rn.f16.f64 rs188, fd218;
}
mov.b32 r1311, {rs188, rs188};
{
mul.f16x2 r1309, r794, r1311;
}
{
add.f16x2 r1312, r1288, r1309;
}
{
cvt.rn.f16.f64 rs189, fd203;
}
mov.b32 r1317, {rs189, rs189};
{
mul.f16x2 r1315, r788, r1317;
}
{
add.f16x2 r1318, r1294, r1315;
}
{
cvt.rn.f16.f64 rs190, fd218;
}
mov.b32 r1323, {rs190, rs190};
{
mul.f16x2 r1321, r791, r1323;
}
{
add.f16x2 r1324, r1300, r1321;
}
{
cvt.rn.f16.f64 rs191, fd219;
}
mov.b32 r1329, {rs191, rs191};
{
mul.f16x2 r1327, r797, r1329;
}
{
add.f16x2 r1330, r1306, r1327;
}
{
cvt.rn.f16.f64 rs192, fd220;
}
mov.b32 r1335, {rs192, rs192};
{
mul.f16x2 r1333, r806, r1335;
}
{
add.f16x2 r1336, r1312, r1333;
}
{
cvt.rn.f16.f64 rs193, fd219;
}
mov.b32 r1341, {rs193, rs193};
{
mul.f16x2 r1339, r800, r1341;
}
{
add.f16x2 r1342, r1318, r1339;
}
{
cvt.rn.f16.f64 rs194, fd220;
}
mov.b32 r1347, {rs194, rs194};
{
mul.f16x2 r1345, r803, r1347;
}
{
add.f16x2 r1348, r1324, r1345;
}
{
cvt.rn.f16.f64 rs195, fd207;
}
mov.b32 r1353, {rs195, rs195};
{
mul.f16x2 r1351, r809, r1353;
}
{
add.f16x2 r1354, r1330, r1351;
}
{
cvt.rn.f16.f64 rs196, fd196;
}
mov.b32 r1359, {rs196, rs196};
{
mul.f16x2 r1357, r818, r1359;
}
{
add.f16x2 r1360, r1336, r1357;
}
{
cvt.rn.f16.f64 rs197, fd207;
}
mov.b32 r1365, {rs197, rs197};
{
mul.f16x2 r1363, r812, r1365;
}
{
add.f16x2 r1366, r1342, r1363;
}
{
cvt.rn.f16.f64 rs198, fd196;
}
mov.b32 r1371, {rs198, rs198};
{
mul.f16x2 r1369, r815, r1371;
}
{
add.f16x2 r1372, r1348, r1369;
}
{
sub.f16x2 r1375, r1354, r1360;
}
{
add.f16x2 r1378, r1366, r1372;
}
{
add.f16x2 r1381, r1354, r1360;
}
{
sub.f16x2 r1384, r1366, r1372;
}
cvt.rn.f16.s32 rs199, r1388;
mov.b32 r1399, {rs199, rs199};
cvt.rn.f16.s32 rs200, r1388;
mov.b32 r1411, {rs200, rs200};
{
cvt.rn.f16.f64 rs201, fd219;
}
mov.b32 r1391, {rs201, rs201};
{
mul.f16x2 r1389, r761, r1391;
}
{
add.f16x2 r1392, %85, r1389;
}
{
cvt.rn.f16.f64 rs202, fd220;
}
mov.b32 r1397, {rs202, rs202};
{
mul.f16x2 r1395, r770, r1397;
}
{
add.f16x2 r1398, r1399, r1395;
}
{
cvt.rn.f16.f64 rs203, fd219;
}
mov.b32 r1403, {rs203, rs203};
{
mul.f16x2 r1401, r764, r1403;
}
{
add.f16x2 r1404, %69, r1401;
}
{
cvt.rn.f16.f64 rs204, fd220;
}
mov.b32 r1409, {rs204, rs204};
{
mul.f16x2 r1407, r767, r1409;
}
{
add.f16x2 r1410, r1411, r1407;
}
{
cvt.rn.f16.f64 rs205, fd203;
}
mov.b32 r1415, {rs205, rs205};
{
mul.f16x2 r1413, r773, r1415;
}
{
add.f16x2 r1416, r1392, r1413;
}
{
cvt.rn.f16.f64 rs206, fd188;
}
mov.b32 r1421, {rs206, rs206};
{
mul.f16x2 r1419, r782, r1421;
}
{
add.f16x2 r1422, r1398, r1419;
}
{
cvt.rn.f16.f64 rs207, fd203;
}
mov.b32 r1427, {rs207, rs207};
{
mul.f16x2 r1425, r776, r1427;
}
{
add.f16x2 r1428, r1404, r1425;
}
{
cvt.rn.f16.f64 rs208, fd188;
}
mov.b32 r1433, {rs208, rs208};
{
mul.f16x2 r1431, r779, r1433;
}
{
add.f16x2 r1434, r1410, r1431;
}
{
cvt.rn.f16.f64 rs209, fd215;
}
mov.b32 r1439, {rs209, rs209};
{
mul.f16x2 r1437, r785, r1439;
}
{
add.f16x2 r1440, r1416, r1437;
}
{
cvt.rn.f16.f64 rs210, fd216;
}
mov.b32 r1445, {rs210, rs210};
{
mul.f16x2 r1443, r794, r1445;
}
{
add.f16x2 r1446, r1422, r1443;
}
{
cvt.rn.f16.f64 rs211, fd215;
}
mov.b32 r1451, {rs211, rs211};
{
mul.f16x2 r1449, r788, r1451;
}
{
add.f16x2 r1452, r1428, r1449;
}
{
cvt.rn.f16.f64 rs212, fd216;
}
mov.b32 r1457, {rs212, rs212};
{
mul.f16x2 r1455, r791, r1457;
}
{
add.f16x2 r1458, r1434, r1455;
}
{
cvt.rn.f16.f64 rs213, fd207;
}
mov.b32 r1463, {rs213, rs213};
{
mul.f16x2 r1461, r797, r1463;
}
{
add.f16x2 r1464, r1440, r1461;
}
{
cvt.rn.f16.f64 rs214, fd196;
}
mov.b32 r1469, {rs214, rs214};
{
mul.f16x2 r1467, r806, r1469;
}
{
add.f16x2 r1470, r1446, r1467;
}
{
cvt.rn.f16.f64 rs215, fd207;
}
mov.b32 r1475, {rs215, rs215};
{
mul.f16x2 r1473, r800, r1475;
}
{
add.f16x2 r1476, r1452, r1473;
}
{
cvt.rn.f16.f64 rs216, fd196;
}
mov.b32 r1481, {rs216, rs216};
{
mul.f16x2 r1479, r803, r1481;
}
{
add.f16x2 r1482, r1458, r1479;
}
{
cvt.rn.f16.f64 rs217, fd211;
}
mov.b32 r1487, {rs217, rs217};
{
mul.f16x2 r1485, r809, r1487;
}
{
add.f16x2 r1488, r1464, r1485;
}
{
cvt.rn.f16.f64 rs218, fd212;
}
mov.b32 r1493, {rs218, rs218};
{
mul.f16x2 r1491, r818, r1493;
}
{
add.f16x2 r1494, r1470, r1491;
}
{
cvt.rn.f16.f64 rs219, fd211;
}
mov.b32 r1499, {rs219, rs219};
{
mul.f16x2 r1497, r812, r1499;
}
{
add.f16x2 r1500, r1476, r1497;
}
{
cvt.rn.f16.f64 rs220, fd212;
}
mov.b32 r1505, {rs220, rs220};
{
mul.f16x2 r1503, r815, r1505;
}
{
add.f16x2 r1506, r1482, r1503;
}
{
sub.f16x2 r1509, r1488, r1494;
}
{
add.f16x2 r1512, r1500, r1506;
}
{
add.f16x2 r1515, r1488, r1494;
}
{
sub.f16x2 r1518, r1500, r1506;
}
mov.f64 fd201, 0d3FEEB42A9BCD5057;
{
cvt.rn.f16.f64 rs221, fd201;
}
{
cvt.rn.f16.f64 rs222, fd220;
}
{
cvt.rn.f16.f64 rs223, fd203;
}
{
cvt.rn.f16.f64 rs224, fd218;
}
mov.f64 fd205, 0d3FE4F49E7F775887;
{
cvt.rn.f16.f64 rs225, fd205;
}
{
cvt.rn.f16.f64 rs226, fd216;
}
{
cvt.rn.f16.f64 rs227, fd207;
}
{
cvt.rn.f16.f64 rs228, fd214;
}
mov.f64 fd209, 0d3FC2375F640F44DB;
{
cvt.rn.f16.f64 rs229, fd209;
}
{
cvt.rn.f16.f64 rs230, fd212;
}
{
cvt.rn.f16.f64 rs231, fd211;
}
{
cvt.rn.f16.f64 rs232, fd212;
}
mov.f64 fd213, 0dBFDA9628D9C712B6;
{
cvt.rn.f16.f64 rs233, fd213;
}
{
cvt.rn.f16.f64 rs234, fd214;
}
{
cvt.rn.f16.f64 rs235, fd215;
}
{
cvt.rn.f16.f64 rs236, fd216;
}
mov.f64 fd217, 0dBFEAEB8C8764F0BA;
{
cvt.rn.f16.f64 rs237, fd217;
}
{
cvt.rn.f16.f64 rs238, fd218;
}
{
cvt.rn.f16.f64 rs239, fd219;
}
{
cvt.rn.f16.f64 rs240, fd220;
}
mov.b32 r1535, {rs221, rs221};
{
mul.f16x2 r1521, r973, r1535;
}
mov.b32 r1532, {rs222, rs222};
{
mul.f16x2 r1524, r976, r1532;
}
{
sub.f16x2 r1527, r1521, r1524;
}
{
mul.f16x2 r1530, r973, r1532;
}
{
fma.rn.f16x2 r1533, r976, r1535, r1530;
}
mov.b32 r1551, {rs223, rs223};
{
mul.f16x2 r1537, r1107, r1551;
}
mov.b32 r1548, {rs224, rs224};
{
mul.f16x2 r1540, r1110, r1548;
}
{
sub.f16x2 r1543, r1537, r1540;
}
{
mul.f16x2 r1546, r1107, r1548;
}
{
fma.rn.f16x2 r1549, r1110, r1551, r1546;
}
mov.b32 r1567, {rs225, rs225};
{
mul.f16x2 r1553, r1241, r1567;
}
mov.b32 r1564, {rs226, rs226};
{
mul.f16x2 r1556, r1244, r1564;
}
{
sub.f16x2 r1559, r1553, r1556;
}
{
mul.f16x2 r1562, r1241, r1564;
}
{
fma.rn.f16x2 r1565, r1244, r1567, r1562;
}
mov.b32 r1583, {rs227, rs227};
{
mul.f16x2 r1569, r1375, r1583;
}
mov.b32 r1580, {rs228, rs228};
{
mul.f16x2 r1572, r1378, r1580;
}
{
sub.f16x2 r1575, r1569, r1572;
}
{
mul.f16x2 r1578, r1375, r1580;
}
{
fma.rn.f16x2 r1581, r1378, r1583, r1578;
}
mov.b32 r1599, {rs229, rs229};
{
mul.f16x2 r1585, r1509, r1599;
}
mov.b32 r1596, {rs230, rs230};
{
mul.f16x2 r1588, r1512, r1596;
}
{
sub.f16x2 r1591, r1585, r1588;
}
{
mul.f16x2 r1594, r1509, r1596;
}
{
fma.rn.f16x2 r1597, r1512, r1599, r1594;
}
mov.b32 r1615, {rs231, rs231};
{
mul.f16x2 r1601, r1515, r1615;
}
mov.b32 r1612, {rs232, rs232};
{
mul.f16x2 r1604, r1518, r1612;
}
{
sub.f16x2 r1607, r1601, r1604;
}
{
mul.f16x2 r1610, r1515, r1612;
}
{
fma.rn.f16x2 r1613, r1518, r1615, r1610;
}
mov.b32 r1631, {rs233, rs233};
{
mul.f16x2 r1617, r1381, r1631;
}
mov.b32 r1628, {rs234, rs234};
{
mul.f16x2 r1620, r1384, r1628;
}
{
sub.f16x2 r1623, r1617, r1620;
}
{
mul.f16x2 r1626, r1381, r1628;
}
{
fma.rn.f16x2 r1629, r1384, r1631, r1626;
}
mov.b32 r1647, {rs235, rs235};
{
mul.f16x2 r1633, r1247, r1647;
}
mov.b32 r1644, {rs236, rs236};
{
mul.f16x2 r1636, r1250, r1644;
}
{
sub.f16x2 r1639, r1633, r1636;
}
{
mul.f16x2 r1642, r1247, r1644;
}
{
fma.rn.f16x2 r1645, r1250, r1647, r1642;
}
mov.b32 r1663, {rs237, rs237};
{
mul.f16x2 r1649, r1113, r1663;
}
mov.b32 r1660, {rs238, rs238};
{
mul.f16x2 r1652, r1116, r1660;
}
{
sub.f16x2 r1655, r1649, r1652;
}
{
mul.f16x2 r1658, r1113, r1660;
}
{
fma.rn.f16x2 r1661, r1116, r1663, r1658;
}
mov.b32 r1679, {rs239, rs239};
{
mul.f16x2 r1665, r979, r1679;
}
mov.b32 r1676, {rs240, rs240};
{
mul.f16x2 r1668, r982, r1676;
}
{
sub.f16x2 r1671, r1665, r1668;
}
{
mul.f16x2 r1674, r979, r1676;
}
{
fma.rn.f16x2 r1677, r982, r1679, r1674;
}
{
add.f16x2 %0, r85, r845;
}
{
add.f16x2 %1, r88, r848;
}
{
sub.f16x2 %22, r85, r845;
}
{
sub.f16x2 %23, r88, r848;
}
{
add.f16x2 %2, r213, r1527;
}
{
add.f16x2 %3, r216, r1533;
}
{
sub.f16x2 %24, r213, r1527;
}
{
sub.f16x2 %25, r216, r1533;
}
{
add.f16x2 %4, r347, r1543;
}
{
add.f16x2 %5, r350, r1549;
}
{
sub.f16x2 %26, r347, r1543;
}
{
sub.f16x2 %27, r350, r1549;
}
{
add.f16x2 %6, r481, r1559;
}
{
add.f16x2 %7, r484, r1565;
}
{
sub.f16x2 %28, r481, r1559;
}
{
sub.f16x2 %29, r484, r1565;
}
{
add.f16x2 %8, r615, r1575;
}
{
add.f16x2 %9, r618, r1581;
}
{
sub.f16x2 %30, r615, r1575;
}
{
sub.f16x2 %31, r618, r1581;
}
{
add.f16x2 %10, r749, r1591;
}
{
add.f16x2 %11, r752, r1597;
}
{
sub.f16x2 %32, r749, r1591;
}
{
sub.f16x2 %33, r752, r1597;
}
{
add.f16x2 %12, r755, r1607;
}
{
add.f16x2 %13, r758, r1613;
}
{
sub.f16x2 %34, r755, r1607;
}
{
sub.f16x2 %35, r758, r1613;
}
{
add.f16x2 %14, r621, r1623;
}
{
add.f16x2 %15, r624, r1629;
}
{
sub.f16x2 %36, r621, r1623;
}
{
sub.f16x2 %37, r624, r1629;
}
{
add.f16x2 %16, r487, r1639;
}
{
add.f16x2 %17, r490, r1645;
}
{
sub.f16x2 %38, r487, r1639;
}
{
sub.f16x2 %39, r490, r1645;
}
{
add.f16x2 %18, r353, r1655;
}
{
add.f16x2 %19, r356, r1661;
}
{
sub.f16x2 %40, r353, r1655;
}
{
sub.f16x2 %41, r356, r1661;
}
{
add.f16x2 %20, r219, r1671;
}
{
add.f16x2 %21, r222, r1677;
}
{
sub.f16x2 %42, r219, r1671;
}
{
sub.f16x2 %43, r222, r1677;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[5].y)));
};


#endif
