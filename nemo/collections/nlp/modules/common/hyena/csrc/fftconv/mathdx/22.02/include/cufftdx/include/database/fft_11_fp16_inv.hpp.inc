#ifndef CUFFTDX_FFT_11_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_11_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1126, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<111>;
.reg .b32 r<761>;
.reg .f64 fd<101>;
.reg .b64 rd<2>;
{
add.f16x2 r1, %24, %42;
}
{
add.f16x2 r4, %25, %43;
}
{
sub.f16x2 r7, %24, %42;
}
{
sub.f16x2 r10, %25, %43;
}
{
add.f16x2 r13, %26, %40;
}
{
add.f16x2 r16, %27, %41;
}
{
sub.f16x2 r19, %26, %40;
}
{
sub.f16x2 r22, %27, %41;
}
{
add.f16x2 r25, %28, %38;
}
{
add.f16x2 r28, %29, %39;
}
{
sub.f16x2 r31, %28, %38;
}
{
sub.f16x2 r34, %29, %39;
}
{
add.f16x2 r37, %30, %36;
}
{
add.f16x2 r40, %31, %37;
}
{
sub.f16x2 r43, %30, %36;
}
{
sub.f16x2 r46, %31, %37;
}
{
add.f16x2 r49, %32, %34;
}
{
add.f16x2 r52, %33, %35;
}
{
sub.f16x2 r55, %32, %34;
}
{
sub.f16x2 r58, %33, %35;
}
{
add.f16x2 r61, %22, r1;
}
{
add.f16x2 r64, %23, r4;
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
add.f16x2 %0, r79, r49;
}
{
add.f16x2 %1, r82, r52;
}
mov.u32 r628, 0;
cvt.rn.f16.s32 rs1, r628;
mov.b32 r103, {rs1, rs1};
cvt.rn.f16.s32 rs2, r628;
mov.b32 r115, {rs2, rs2};
mov.f64 fd87, 0d3FEAEB8C8764F0BA;
{
cvt.rn.f16.f64 rs3, fd87;
}
mov.b32 r95, {rs3, rs3};
{
mul.f16x2 r93, r1, r95;
}
{
add.f16x2 r96, %22, r93;
}
mov.f64 fd72, 0d3FE14CEDF8BB580B;
{
cvt.rn.f16.f64 rs4, fd72;
}
mov.b32 r101, {rs4, rs4};
{
mul.f16x2 r99, r10, r101;
}
{
add.f16x2 r102, r103, r99;
}
{
cvt.rn.f16.f64 rs5, fd87;
}
mov.b32 r107, {rs5, rs5};
{
mul.f16x2 r105, r4, r107;
}
{
add.f16x2 r108, %23, r105;
}
{
cvt.rn.f16.f64 rs6, fd72;
}
mov.b32 r113, {rs6, rs6};
{
mul.f16x2 r111, r7, r113;
}
{
add.f16x2 r114, r115, r111;
}
mov.f64 fd95, 0d3FDA9628D9C712B6;
{
cvt.rn.f16.f64 rs7, fd95;
}
mov.b32 r119, {rs7, rs7};
{
mul.f16x2 r117, r13, r119;
}
{
add.f16x2 r120, r96, r117;
}
mov.f64 fd24, 0d3FED1BB48EEE2C13;
{
cvt.rn.f16.f64 rs8, fd24;
}
mov.b32 r125, {rs8, rs8};
{
mul.f16x2 r123, r22, r125;
}
{
add.f16x2 r126, r102, r123;
}
{
cvt.rn.f16.f64 rs9, fd95;
}
mov.b32 r131, {rs9, rs9};
{
mul.f16x2 r129, r16, r131;
}
{
add.f16x2 r132, r108, r129;
}
{
cvt.rn.f16.f64 rs10, fd24;
}
mov.b32 r137, {rs10, rs10};
{
mul.f16x2 r135, r19, r137;
}
{
add.f16x2 r138, r114, r135;
}
mov.f64 fd99, 0dBFC2375F640F44DB;
{
cvt.rn.f16.f64 rs11, fd99;
}
mov.b32 r143, {rs11, rs11};
{
mul.f16x2 r141, r25, r143;
}
{
add.f16x2 r144, r120, r141;
}
mov.f64 fd100, 0d3FEFAC9E043842EF;
{
cvt.rn.f16.f64 rs12, fd100;
}
mov.b32 r149, {rs12, rs12};
{
mul.f16x2 r147, r34, r149;
}
{
add.f16x2 r150, r126, r147;
}
{
cvt.rn.f16.f64 rs13, fd99;
}
mov.b32 r155, {rs13, rs13};
{
mul.f16x2 r153, r28, r155;
}
{
add.f16x2 r156, r132, r153;
}
{
cvt.rn.f16.f64 rs14, fd100;
}
mov.b32 r161, {rs14, rs14};
{
mul.f16x2 r159, r31, r161;
}
{
add.f16x2 r162, r138, r159;
}
mov.f64 fd91, 0dBFE4F49E7F775887;
{
cvt.rn.f16.f64 rs15, fd91;
}
mov.b32 r167, {rs15, rs15};
{
mul.f16x2 r165, r37, r167;
}
{
add.f16x2 r168, r144, r165;
}
mov.f64 fd92, 0d3FE82F19BB3A28A1;
{
cvt.rn.f16.f64 rs16, fd92;
}
mov.b32 r173, {rs16, rs16};
{
mul.f16x2 r171, r46, r173;
}
{
add.f16x2 r174, r150, r171;
}
{
cvt.rn.f16.f64 rs17, fd91;
}
mov.b32 r179, {rs17, rs17};
{
mul.f16x2 r177, r40, r179;
}
{
add.f16x2 r180, r156, r177;
}
{
cvt.rn.f16.f64 rs18, fd92;
}
mov.b32 r185, {rs18, rs18};
{
mul.f16x2 r183, r43, r185;
}
{
add.f16x2 r186, r162, r183;
}
mov.f64 fd83, 0dBFEEB42A9BCD5057;
{
cvt.rn.f16.f64 rs19, fd83;
}
mov.b32 r191, {rs19, rs19};
{
mul.f16x2 r189, r49, r191;
}
{
add.f16x2 r192, r168, r189;
}
mov.f64 fd84, 0d3FD207E7FD768DBF;
{
cvt.rn.f16.f64 rs20, fd84;
}
mov.b32 r197, {rs20, rs20};
{
mul.f16x2 r195, r58, r197;
}
{
add.f16x2 r198, r174, r195;
}
{
cvt.rn.f16.f64 rs21, fd83;
}
mov.b32 r203, {rs21, rs21};
{
mul.f16x2 r201, r52, r203;
}
{
add.f16x2 r204, r180, r201;
}
{
cvt.rn.f16.f64 rs22, fd84;
}
mov.b32 r209, {rs22, rs22};
{
mul.f16x2 r207, r55, r209;
}
{
add.f16x2 r210, r186, r207;
}
{
sub.f16x2 %2, r192, r198;
}
{
add.f16x2 %3, r204, r210;
}
{
add.f16x2 %20, r192, r198;
}
{
sub.f16x2 %21, r204, r210;
}
cvt.rn.f16.s32 rs23, r628;
mov.b32 r237, {rs23, rs23};
cvt.rn.f16.s32 rs24, r628;
mov.b32 r249, {rs24, rs24};
{
cvt.rn.f16.f64 rs25, fd95;
}
mov.b32 r229, {rs25, rs25};
{
mul.f16x2 r227, r1, r229;
}
{
add.f16x2 r230, %22, r227;
}
{
cvt.rn.f16.f64 rs26, fd24;
}
mov.b32 r235, {rs26, rs26};
{
mul.f16x2 r233, r10, r235;
}
{
add.f16x2 r236, r237, r233;
}
{
cvt.rn.f16.f64 rs27, fd95;
}
mov.b32 r241, {rs27, rs27};
{
mul.f16x2 r239, r4, r241;
}
{
add.f16x2 r242, %23, r239;
}
{
cvt.rn.f16.f64 rs28, fd24;
}
mov.b32 r247, {rs28, rs28};
{
mul.f16x2 r245, r7, r247;
}
{
add.f16x2 r248, r249, r245;
}
{
cvt.rn.f16.f64 rs29, fd91;
}
mov.b32 r253, {rs29, rs29};
{
mul.f16x2 r251, r13, r253;
}
{
add.f16x2 r254, r230, r251;
}
{
cvt.rn.f16.f64 rs30, fd92;
}
mov.b32 r259, {rs30, rs30};
{
mul.f16x2 r257, r22, r259;
}
{
add.f16x2 r260, r236, r257;
}
{
cvt.rn.f16.f64 rs31, fd91;
}
mov.b32 r265, {rs31, rs31};
{
mul.f16x2 r263, r16, r265;
}
{
add.f16x2 r266, r242, r263;
}
{
cvt.rn.f16.f64 rs32, fd92;
}
mov.b32 r271, {rs32, rs32};
{
mul.f16x2 r269, r19, r271;
}
{
add.f16x2 r272, r248, r269;
}
{
cvt.rn.f16.f64 rs33, fd83;
}
mov.b32 r277, {rs33, rs33};
{
mul.f16x2 r275, r25, r277;
}
{
add.f16x2 r278, r254, r275;
}
mov.f64 fd48, 0dBFD207E7FD768DBF;
{
cvt.rn.f16.f64 rs34, fd48;
}
mov.b32 r283, {rs34, rs34};
{
mul.f16x2 r281, r34, r283;
}
{
add.f16x2 r284, r260, r281;
}
{
cvt.rn.f16.f64 rs35, fd83;
}
mov.b32 r289, {rs35, rs35};
{
mul.f16x2 r287, r28, r289;
}
{
add.f16x2 r290, r266, r287;
}
{
cvt.rn.f16.f64 rs36, fd48;
}
mov.b32 r295, {rs36, rs36};
{
mul.f16x2 r293, r31, r295;
}
{
add.f16x2 r296, r272, r293;
}
{
cvt.rn.f16.f64 rs37, fd99;
}
mov.b32 r301, {rs37, rs37};
{
mul.f16x2 r299, r37, r301;
}
{
add.f16x2 r302, r278, r299;
}
mov.f64 fd68, 0dBFEFAC9E043842EF;
{
cvt.rn.f16.f64 rs38, fd68;
}
mov.b32 r307, {rs38, rs38};
{
mul.f16x2 r305, r46, r307;
}
{
add.f16x2 r308, r284, r305;
}
{
cvt.rn.f16.f64 rs39, fd99;
}
mov.b32 r313, {rs39, rs39};
{
mul.f16x2 r311, r40, r313;
}
{
add.f16x2 r314, r290, r311;
}
{
cvt.rn.f16.f64 rs40, fd68;
}
mov.b32 r319, {rs40, rs40};
{
mul.f16x2 r317, r43, r319;
}
{
add.f16x2 r320, r296, r317;
}
{
cvt.rn.f16.f64 rs41, fd87;
}
mov.b32 r325, {rs41, rs41};
{
mul.f16x2 r323, r49, r325;
}
{
add.f16x2 r326, r302, r323;
}
mov.f64 fd88, 0dBFE14CEDF8BB580B;
{
cvt.rn.f16.f64 rs42, fd88;
}
mov.b32 r331, {rs42, rs42};
{
mul.f16x2 r329, r58, r331;
}
{
add.f16x2 r332, r308, r329;
}
{
cvt.rn.f16.f64 rs43, fd87;
}
mov.b32 r337, {rs43, rs43};
{
mul.f16x2 r335, r52, r337;
}
{
add.f16x2 r338, r314, r335;
}
{
cvt.rn.f16.f64 rs44, fd88;
}
mov.b32 r343, {rs44, rs44};
{
mul.f16x2 r341, r55, r343;
}
{
add.f16x2 r344, r320, r341;
}
{
sub.f16x2 %4, r326, r332;
}
{
add.f16x2 %5, r338, r344;
}
{
add.f16x2 %18, r326, r332;
}
{
sub.f16x2 %19, r338, r344;
}
cvt.rn.f16.s32 rs45, r628;
mov.b32 r371, {rs45, rs45};
cvt.rn.f16.s32 rs46, r628;
mov.b32 r383, {rs46, rs46};
{
cvt.rn.f16.f64 rs47, fd99;
}
mov.b32 r363, {rs47, rs47};
{
mul.f16x2 r361, r1, r363;
}
{
add.f16x2 r364, %22, r361;
}
{
cvt.rn.f16.f64 rs48, fd100;
}
mov.b32 r369, {rs48, rs48};
{
mul.f16x2 r367, r10, r369;
}
{
add.f16x2 r370, r371, r367;
}
{
cvt.rn.f16.f64 rs49, fd99;
}
mov.b32 r375, {rs49, rs49};
{
mul.f16x2 r373, r4, r375;
}
{
add.f16x2 r376, %23, r373;
}
{
cvt.rn.f16.f64 rs50, fd100;
}
mov.b32 r381, {rs50, rs50};
{
mul.f16x2 r379, r7, r381;
}
{
add.f16x2 r382, r383, r379;
}
{
cvt.rn.f16.f64 rs51, fd83;
}
mov.b32 r387, {rs51, rs51};
{
mul.f16x2 r385, r13, r387;
}
{
add.f16x2 r388, r364, r385;
}
{
cvt.rn.f16.f64 rs52, fd48;
}
mov.b32 r393, {rs52, rs52};
{
mul.f16x2 r391, r22, r393;
}
{
add.f16x2 r394, r370, r391;
}
{
cvt.rn.f16.f64 rs53, fd83;
}
mov.b32 r399, {rs53, rs53};
{
mul.f16x2 r397, r16, r399;
}
{
add.f16x2 r400, r376, r397;
}
{
cvt.rn.f16.f64 rs54, fd48;
}
mov.b32 r405, {rs54, rs54};
{
mul.f16x2 r403, r19, r405;
}
{
add.f16x2 r406, r382, r403;
}
{
cvt.rn.f16.f64 rs55, fd95;
}
mov.b32 r411, {rs55, rs55};
{
mul.f16x2 r409, r25, r411;
}
{
add.f16x2 r412, r388, r409;
}
mov.f64 fd96, 0dBFED1BB48EEE2C13;
{
cvt.rn.f16.f64 rs56, fd96;
}
mov.b32 r417, {rs56, rs56};
{
mul.f16x2 r415, r34, r417;
}
{
add.f16x2 r418, r394, r415;
}
{
cvt.rn.f16.f64 rs57, fd95;
}
mov.b32 r423, {rs57, rs57};
{
mul.f16x2 r421, r28, r423;
}
{
add.f16x2 r424, r400, r421;
}
{
cvt.rn.f16.f64 rs58, fd96;
}
mov.b32 r429, {rs58, rs58};
{
mul.f16x2 r427, r31, r429;
}
{
add.f16x2 r430, r406, r427;
}
{
cvt.rn.f16.f64 rs59, fd87;
}
mov.b32 r435, {rs59, rs59};
{
mul.f16x2 r433, r37, r435;
}
{
add.f16x2 r436, r412, r433;
}
{
cvt.rn.f16.f64 rs60, fd72;
}
mov.b32 r441, {rs60, rs60};
{
mul.f16x2 r439, r46, r441;
}
{
add.f16x2 r442, r418, r439;
}
{
cvt.rn.f16.f64 rs61, fd87;
}
mov.b32 r447, {rs61, rs61};
{
mul.f16x2 r445, r40, r447;
}
{
add.f16x2 r448, r424, r445;
}
{
cvt.rn.f16.f64 rs62, fd72;
}
mov.b32 r453, {rs62, rs62};
{
mul.f16x2 r451, r43, r453;
}
{
add.f16x2 r454, r430, r451;
}
{
cvt.rn.f16.f64 rs63, fd91;
}
mov.b32 r459, {rs63, rs63};
{
mul.f16x2 r457, r49, r459;
}
{
add.f16x2 r460, r436, r457;
}
{
cvt.rn.f16.f64 rs64, fd92;
}
mov.b32 r465, {rs64, rs64};
{
mul.f16x2 r463, r58, r465;
}
{
add.f16x2 r466, r442, r463;
}
{
cvt.rn.f16.f64 rs65, fd91;
}
mov.b32 r471, {rs65, rs65};
{
mul.f16x2 r469, r52, r471;
}
{
add.f16x2 r472, r448, r469;
}
{
cvt.rn.f16.f64 rs66, fd92;
}
mov.b32 r477, {rs66, rs66};
{
mul.f16x2 r475, r55, r477;
}
{
add.f16x2 r478, r454, r475;
}
{
sub.f16x2 %6, r460, r466;
}
{
add.f16x2 %7, r472, r478;
}
{
add.f16x2 %16, r460, r466;
}
{
sub.f16x2 %17, r472, r478;
}
cvt.rn.f16.s32 rs67, r628;
mov.b32 r505, {rs67, rs67};
cvt.rn.f16.s32 rs68, r628;
mov.b32 r517, {rs68, rs68};
{
cvt.rn.f16.f64 rs69, fd91;
}
mov.b32 r497, {rs69, rs69};
{
mul.f16x2 r495, r1, r497;
}
{
add.f16x2 r498, %22, r495;
}
{
cvt.rn.f16.f64 rs70, fd92;
}
mov.b32 r503, {rs70, rs70};
{
mul.f16x2 r501, r10, r503;
}
{
add.f16x2 r504, r505, r501;
}
{
cvt.rn.f16.f64 rs71, fd91;
}
mov.b32 r509, {rs71, rs71};
{
mul.f16x2 r507, r4, r509;
}
{
add.f16x2 r510, %23, r507;
}
{
cvt.rn.f16.f64 rs72, fd92;
}
mov.b32 r515, {rs72, rs72};
{
mul.f16x2 r513, r7, r515;
}
{
add.f16x2 r516, r517, r513;
}
{
cvt.rn.f16.f64 rs73, fd99;
}
mov.b32 r521, {rs73, rs73};
{
mul.f16x2 r519, r13, r521;
}
{
add.f16x2 r522, r498, r519;
}
{
cvt.rn.f16.f64 rs74, fd68;
}
mov.b32 r527, {rs74, rs74};
{
mul.f16x2 r525, r22, r527;
}
{
add.f16x2 r528, r504, r525;
}
{
cvt.rn.f16.f64 rs75, fd99;
}
mov.b32 r533, {rs75, rs75};
{
mul.f16x2 r531, r16, r533;
}
{
add.f16x2 r534, r510, r531;
}
{
cvt.rn.f16.f64 rs76, fd68;
}
mov.b32 r539, {rs76, rs76};
{
mul.f16x2 r537, r19, r539;
}
{
add.f16x2 r540, r516, r537;
}
{
cvt.rn.f16.f64 rs77, fd87;
}
mov.b32 r545, {rs77, rs77};
{
mul.f16x2 r543, r25, r545;
}
{
add.f16x2 r546, r522, r543;
}
{
cvt.rn.f16.f64 rs78, fd72;
}
mov.b32 r551, {rs78, rs78};
{
mul.f16x2 r549, r34, r551;
}
{
add.f16x2 r552, r528, r549;
}
{
cvt.rn.f16.f64 rs79, fd87;
}
mov.b32 r557, {rs79, rs79};
{
mul.f16x2 r555, r28, r557;
}
{
add.f16x2 r558, r534, r555;
}
{
cvt.rn.f16.f64 rs80, fd72;
}
mov.b32 r563, {rs80, rs80};
{
mul.f16x2 r561, r31, r563;
}
{
add.f16x2 r564, r540, r561;
}
{
cvt.rn.f16.f64 rs81, fd83;
}
mov.b32 r569, {rs81, rs81};
{
mul.f16x2 r567, r37, r569;
}
{
add.f16x2 r570, r546, r567;
}
{
cvt.rn.f16.f64 rs82, fd84;
}
mov.b32 r575, {rs82, rs82};
{
mul.f16x2 r573, r46, r575;
}
{
add.f16x2 r576, r552, r573;
}
{
cvt.rn.f16.f64 rs83, fd83;
}
mov.b32 r581, {rs83, rs83};
{
mul.f16x2 r579, r40, r581;
}
{
add.f16x2 r582, r558, r579;
}
{
cvt.rn.f16.f64 rs84, fd84;
}
mov.b32 r587, {rs84, rs84};
{
mul.f16x2 r585, r43, r587;
}
{
add.f16x2 r588, r564, r585;
}
{
cvt.rn.f16.f64 rs85, fd95;
}
mov.b32 r593, {rs85, rs85};
{
mul.f16x2 r591, r49, r593;
}
{
add.f16x2 r594, r570, r591;
}
{
cvt.rn.f16.f64 rs86, fd96;
}
mov.b32 r599, {rs86, rs86};
{
mul.f16x2 r597, r58, r599;
}
{
add.f16x2 r600, r576, r597;
}
{
cvt.rn.f16.f64 rs87, fd95;
}
mov.b32 r605, {rs87, rs87};
{
mul.f16x2 r603, r52, r605;
}
{
add.f16x2 r606, r582, r603;
}
{
cvt.rn.f16.f64 rs88, fd96;
}
mov.b32 r611, {rs88, rs88};
{
mul.f16x2 r609, r55, r611;
}
{
add.f16x2 r612, r588, r609;
}
{
sub.f16x2 %8, r594, r600;
}
{
add.f16x2 %9, r606, r612;
}
{
add.f16x2 %14, r594, r600;
}
{
sub.f16x2 %15, r606, r612;
}
cvt.rn.f16.s32 rs89, r628;
mov.b32 r639, {rs89, rs89};
cvt.rn.f16.s32 rs90, r628;
mov.b32 r651, {rs90, rs90};
{
cvt.rn.f16.f64 rs91, fd83;
}
mov.b32 r631, {rs91, rs91};
{
mul.f16x2 r629, r1, r631;
}
{
add.f16x2 r632, %22, r629;
}
{
cvt.rn.f16.f64 rs92, fd84;
}
mov.b32 r637, {rs92, rs92};
{
mul.f16x2 r635, r10, r637;
}
{
add.f16x2 r638, r639, r635;
}
{
cvt.rn.f16.f64 rs93, fd83;
}
mov.b32 r643, {rs93, rs93};
{
mul.f16x2 r641, r4, r643;
}
{
add.f16x2 r644, %23, r641;
}
{
cvt.rn.f16.f64 rs94, fd84;
}
mov.b32 r649, {rs94, rs94};
{
mul.f16x2 r647, r7, r649;
}
{
add.f16x2 r650, r651, r647;
}
{
cvt.rn.f16.f64 rs95, fd87;
}
mov.b32 r655, {rs95, rs95};
{
mul.f16x2 r653, r13, r655;
}
{
add.f16x2 r656, r632, r653;
}
{
cvt.rn.f16.f64 rs96, fd88;
}
mov.b32 r661, {rs96, rs96};
{
mul.f16x2 r659, r22, r661;
}
{
add.f16x2 r662, r638, r659;
}
{
cvt.rn.f16.f64 rs97, fd87;
}
mov.b32 r667, {rs97, rs97};
{
mul.f16x2 r665, r16, r667;
}
{
add.f16x2 r668, r644, r665;
}
{
cvt.rn.f16.f64 rs98, fd88;
}
mov.b32 r673, {rs98, rs98};
{
mul.f16x2 r671, r19, r673;
}
{
add.f16x2 r674, r650, r671;
}
{
cvt.rn.f16.f64 rs99, fd91;
}
mov.b32 r679, {rs99, rs99};
{
mul.f16x2 r677, r25, r679;
}
{
add.f16x2 r680, r656, r677;
}
{
cvt.rn.f16.f64 rs100, fd92;
}
mov.b32 r685, {rs100, rs100};
{
mul.f16x2 r683, r34, r685;
}
{
add.f16x2 r686, r662, r683;
}
{
cvt.rn.f16.f64 rs101, fd91;
}
mov.b32 r691, {rs101, rs101};
{
mul.f16x2 r689, r28, r691;
}
{
add.f16x2 r692, r668, r689;
}
{
cvt.rn.f16.f64 rs102, fd92;
}
mov.b32 r697, {rs102, rs102};
{
mul.f16x2 r695, r31, r697;
}
{
add.f16x2 r698, r674, r695;
}
{
cvt.rn.f16.f64 rs103, fd95;
}
mov.b32 r703, {rs103, rs103};
{
mul.f16x2 r701, r37, r703;
}
{
add.f16x2 r704, r680, r701;
}
{
cvt.rn.f16.f64 rs104, fd96;
}
mov.b32 r709, {rs104, rs104};
{
mul.f16x2 r707, r46, r709;
}
{
add.f16x2 r710, r686, r707;
}
{
cvt.rn.f16.f64 rs105, fd95;
}
mov.b32 r715, {rs105, rs105};
{
mul.f16x2 r713, r40, r715;
}
{
add.f16x2 r716, r692, r713;
}
{
cvt.rn.f16.f64 rs106, fd96;
}
mov.b32 r721, {rs106, rs106};
{
mul.f16x2 r719, r43, r721;
}
{
add.f16x2 r722, r698, r719;
}
{
cvt.rn.f16.f64 rs107, fd99;
}
mov.b32 r727, {rs107, rs107};
{
mul.f16x2 r725, r49, r727;
}
{
add.f16x2 r728, r704, r725;
}
{
cvt.rn.f16.f64 rs108, fd100;
}
mov.b32 r733, {rs108, rs108};
{
mul.f16x2 r731, r58, r733;
}
{
add.f16x2 r734, r710, r731;
}
{
cvt.rn.f16.f64 rs109, fd99;
}
mov.b32 r739, {rs109, rs109};
{
mul.f16x2 r737, r52, r739;
}
{
add.f16x2 r740, r716, r737;
}
{
cvt.rn.f16.f64 rs110, fd100;
}
mov.b32 r745, {rs110, rs110};
{
mul.f16x2 r743, r55, r745;
}
{
add.f16x2 r746, r722, r743;
}
{
sub.f16x2 %10, r728, r734;
}
{
add.f16x2 %11, r740, r746;
}
{
add.f16x2 %12, r728, r734;
}
{
sub.f16x2 %13, r740, r746;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)));
};


#endif
