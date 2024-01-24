#ifndef CUFFTDX_FFT_17_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_17_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<951, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<273>;
.reg .b32 r<1793>;
.reg .f64 fd<257>;
.reg .b64 rd<2>;
{
add.f16x2 r1, %36, %66;
}
{
add.f16x2 r4, %37, %67;
}
{
sub.f16x2 r7, %36, %66;
}
{
sub.f16x2 r10, %37, %67;
}
{
add.f16x2 r13, %38, %64;
}
{
add.f16x2 r16, %39, %65;
}
{
sub.f16x2 r19, %38, %64;
}
{
sub.f16x2 r22, %39, %65;
}
{
add.f16x2 r25, %40, %62;
}
{
add.f16x2 r28, %41, %63;
}
{
sub.f16x2 r31, %40, %62;
}
{
sub.f16x2 r34, %41, %63;
}
{
add.f16x2 r37, %42, %60;
}
{
add.f16x2 r40, %43, %61;
}
{
sub.f16x2 r43, %42, %60;
}
{
sub.f16x2 r46, %43, %61;
}
{
add.f16x2 r49, %44, %58;
}
{
add.f16x2 r52, %45, %59;
}
{
sub.f16x2 r55, %44, %58;
}
{
sub.f16x2 r58, %45, %59;
}
{
add.f16x2 r61, %46, %56;
}
{
add.f16x2 r64, %47, %57;
}
{
sub.f16x2 r67, %46, %56;
}
{
sub.f16x2 r70, %47, %57;
}
{
add.f16x2 r73, %48, %54;
}
{
add.f16x2 r76, %49, %55;
}
{
sub.f16x2 r79, %48, %54;
}
{
sub.f16x2 r82, %49, %55;
}
{
add.f16x2 r85, %50, %52;
}
{
add.f16x2 r88, %51, %53;
}
{
sub.f16x2 r91, %50, %52;
}
{
sub.f16x2 r94, %51, %53;
}
{
add.f16x2 r97, %34, r1;
}
{
add.f16x2 r100, %35, r4;
}
{
add.f16x2 r103, r97, r13;
}
{
add.f16x2 r106, r100, r16;
}
{
add.f16x2 r109, r103, r25;
}
{
add.f16x2 r112, r106, r28;
}
{
add.f16x2 r115, r109, r37;
}
{
add.f16x2 r118, r112, r40;
}
{
add.f16x2 r121, r115, r49;
}
{
add.f16x2 r124, r118, r52;
}
{
add.f16x2 r127, r121, r61;
}
{
add.f16x2 r130, r124, r64;
}
{
add.f16x2 r133, r127, r73;
}
{
add.f16x2 r136, r130, r76;
}
{
add.f16x2 %0, r133, r85;
}
{
add.f16x2 %1, r136, r88;
}
mov.u32 r1588, 0;
cvt.rn.f16.s32 rs1, r1588;
mov.b32 r157, {rs1, rs1};
cvt.rn.f16.s32 rs2, r1588;
mov.b32 r169, {rs2, rs2};
mov.f64 fd231, 0d3FEDD6D000370991;
{
cvt.rn.f16.f64 rs3, fd231;
}
mov.b32 r149, {rs3, rs3};
{
mul.f16x2 r147, r1, r149;
}
{
add.f16x2 r150, %34, r147;
}
mov.f64 fd212, 0d3FD71E955D8E7CDC;
{
cvt.rn.f16.f64 rs4, fd212;
}
mov.b32 r155, {rs4, rs4};
{
mul.f16x2 r153, r10, r155;
}
{
add.f16x2 r156, r157, r153;
}
{
cvt.rn.f16.f64 rs5, fd231;
}
mov.b32 r161, {rs5, rs5};
{
mul.f16x2 r159, r4, r161;
}
{
add.f16x2 r162, %35, r159;
}
{
cvt.rn.f16.f64 rs6, fd212;
}
mov.b32 r167, {rs6, rs6};
{
mul.f16x2 r165, r7, r167;
}
{
add.f16x2 r168, r169, r165;
}
mov.f64 fd239, 0d3FE7A5F6075D4884;
{
cvt.rn.f16.f64 rs7, fd239;
}
mov.b32 r173, {rs7, rs7};
{
mul.f16x2 r171, r13, r173;
}
{
add.f16x2 r174, r150, r171;
}
mov.f64 fd184, 0d3FE58EEA2A9D6DA3;
{
cvt.rn.f16.f64 rs8, fd184;
}
mov.b32 r179, {rs8, rs8};
{
mul.f16x2 r177, r22, r179;
}
{
add.f16x2 r180, r156, r177;
}
{
cvt.rn.f16.f64 rs9, fd239;
}
mov.b32 r185, {rs9, rs9};
{
mul.f16x2 r183, r16, r185;
}
{
add.f16x2 r186, r162, r183;
}
{
cvt.rn.f16.f64 rs10, fd184;
}
mov.b32 r191, {rs10, rs10};
{
mul.f16x2 r189, r19, r191;
}
{
add.f16x2 r192, r168, r189;
}
mov.f64 fd247, 0d3FDC86FA2B2883CD;
{
cvt.rn.f16.f64 rs11, fd247;
}
mov.b32 r197, {rs11, rs11};
{
mul.f16x2 r195, r25, r197;
}
{
add.f16x2 r198, r174, r195;
}
mov.f64 fd144, 0d3FECA52D7C9E640B;
{
cvt.rn.f16.f64 rs12, fd144;
}
mov.b32 r203, {rs12, rs12};
{
mul.f16x2 r201, r34, r203;
}
{
add.f16x2 r204, r180, r201;
}
{
cvt.rn.f16.f64 rs13, fd247;
}
mov.b32 r209, {rs13, rs13};
{
mul.f16x2 r207, r28, r209;
}
{
add.f16x2 r210, r186, r207;
}
{
cvt.rn.f16.f64 rs14, fd144;
}
mov.b32 r215, {rs14, rs14};
{
mul.f16x2 r213, r31, r215;
}
{
add.f16x2 r216, r192, r213;
}
mov.f64 fd255, 0d3FB79EE63259B75E;
{
cvt.rn.f16.f64 rs15, fd255;
}
mov.b32 r221, {rs15, rs15};
{
mul.f16x2 r219, r37, r221;
}
{
add.f16x2 r222, r198, r219;
}
mov.f64 fd204, 0d3FEFDD0DEB564B22;
{
cvt.rn.f16.f64 rs16, fd204;
}
mov.b32 r227, {rs16, rs16};
{
mul.f16x2 r225, r46, r227;
}
{
add.f16x2 r228, r204, r225;
}
{
cvt.rn.f16.f64 rs17, fd255;
}
mov.b32 r233, {rs17, rs17};
{
mul.f16x2 r231, r40, r233;
}
{
add.f16x2 r234, r210, r231;
}
{
cvt.rn.f16.f64 rs18, fd204;
}
mov.b32 r239, {rs18, rs18};
{
mul.f16x2 r237, r43, r239;
}
{
add.f16x2 r240, r216, r237;
}
mov.f64 fd251, 0dBFD183B1C61F0D01;
{
cvt.rn.f16.f64 rs19, fd251;
}
mov.b32 r245, {rs19, rs19};
{
mul.f16x2 r243, r49, r245;
}
{
add.f16x2 r246, r222, r243;
}
mov.f64 fd252, 0d3FEEC746923C349F;
{
cvt.rn.f16.f64 rs20, fd252;
}
mov.b32 r251, {rs20, rs20};
{
mul.f16x2 r249, r58, r251;
}
{
add.f16x2 r252, r228, r249;
}
{
cvt.rn.f16.f64 rs21, fd251;
}
mov.b32 r257, {rs21, rs21};
{
mul.f16x2 r255, r52, r257;
}
{
add.f16x2 r258, r234, r255;
}
{
cvt.rn.f16.f64 rs22, fd252;
}
mov.b32 r263, {rs22, rs22};
{
mul.f16x2 r261, r55, r263;
}
{
add.f16x2 r264, r240, r261;
}
mov.f64 fd243, 0dBFE348C86ED5F1BB;
{
cvt.rn.f16.f64 rs23, fd243;
}
mov.b32 r269, {rs23, rs23};
{
mul.f16x2 r267, r61, r269;
}
{
add.f16x2 r270, r246, r267;
}
mov.f64 fd244, 0d3FE9895B6C9A05F6;
{
cvt.rn.f16.f64 rs24, fd244;
}
mov.b32 r275, {rs24, rs24};
{
mul.f16x2 r273, r70, r275;
}
{
add.f16x2 r276, r252, r273;
}
{
cvt.rn.f16.f64 rs25, fd243;
}
mov.b32 r281, {rs25, rs25};
{
mul.f16x2 r279, r64, r281;
}
{
add.f16x2 r282, r258, r279;
}
{
cvt.rn.f16.f64 rs26, fd244;
}
mov.b32 r287, {rs26, rs26};
{
mul.f16x2 r285, r67, r287;
}
{
add.f16x2 r288, r264, r285;
}
mov.f64 fd235, 0dBFEB34FA910EA3B9;
{
cvt.rn.f16.f64 rs27, fd235;
}
mov.b32 r293, {rs27, rs27};
{
mul.f16x2 r291, r73, r293;
}
{
add.f16x2 r294, r270, r291;
}
mov.f64 fd236, 0d3FE0D8884363DD80;
{
cvt.rn.f16.f64 rs28, fd236;
}
mov.b32 r299, {rs28, rs28};
{
mul.f16x2 r297, r82, r299;
}
{
add.f16x2 r300, r276, r297;
}
{
cvt.rn.f16.f64 rs29, fd235;
}
mov.b32 r305, {rs29, rs29};
{
mul.f16x2 r303, r76, r305;
}
{
add.f16x2 r306, r282, r303;
}
{
cvt.rn.f16.f64 rs30, fd236;
}
mov.b32 r311, {rs30, rs30};
{
mul.f16x2 r309, r79, r311;
}
{
add.f16x2 r312, r288, r309;
}
mov.f64 fd227, 0dBFEF7484007FAEF3;
{
cvt.rn.f16.f64 rs31, fd227;
}
mov.b32 r317, {rs31, rs31};
{
mul.f16x2 r315, r85, r317;
}
{
add.f16x2 r318, r294, r315;
}
mov.f64 fd228, 0d3FC7851AACD6C6B4;
{
cvt.rn.f16.f64 rs32, fd228;
}
mov.b32 r323, {rs32, rs32};
{
mul.f16x2 r321, r94, r323;
}
{
add.f16x2 r324, r300, r321;
}
{
cvt.rn.f16.f64 rs33, fd227;
}
mov.b32 r329, {rs33, rs33};
{
mul.f16x2 r327, r88, r329;
}
{
add.f16x2 r330, r306, r327;
}
{
cvt.rn.f16.f64 rs34, fd228;
}
mov.b32 r335, {rs34, rs34};
{
mul.f16x2 r333, r91, r335;
}
{
add.f16x2 r336, r312, r333;
}
{
sub.f16x2 %2, r318, r324;
}
{
add.f16x2 %3, r330, r336;
}
{
add.f16x2 %32, r318, r324;
}
{
sub.f16x2 %33, r330, r336;
}
cvt.rn.f16.s32 rs35, r1588;
mov.b32 r363, {rs35, rs35};
cvt.rn.f16.s32 rs36, r1588;
mov.b32 r375, {rs36, rs36};
{
cvt.rn.f16.f64 rs37, fd239;
}
mov.b32 r355, {rs37, rs37};
{
mul.f16x2 r353, r1, r355;
}
{
add.f16x2 r356, %34, r353;
}
{
cvt.rn.f16.f64 rs38, fd184;
}
mov.b32 r361, {rs38, rs38};
{
mul.f16x2 r359, r10, r361;
}
{
add.f16x2 r362, r363, r359;
}
{
cvt.rn.f16.f64 rs39, fd239;
}
mov.b32 r367, {rs39, rs39};
{
mul.f16x2 r365, r4, r367;
}
{
add.f16x2 r368, %35, r365;
}
{
cvt.rn.f16.f64 rs40, fd184;
}
mov.b32 r373, {rs40, rs40};
{
mul.f16x2 r371, r7, r373;
}
{
add.f16x2 r374, r375, r371;
}
{
cvt.rn.f16.f64 rs41, fd255;
}
mov.b32 r379, {rs41, rs41};
{
mul.f16x2 r377, r13, r379;
}
{
add.f16x2 r380, r356, r377;
}
{
cvt.rn.f16.f64 rs42, fd204;
}
mov.b32 r385, {rs42, rs42};
{
mul.f16x2 r383, r22, r385;
}
{
add.f16x2 r386, r362, r383;
}
{
cvt.rn.f16.f64 rs43, fd255;
}
mov.b32 r391, {rs43, rs43};
{
mul.f16x2 r389, r16, r391;
}
{
add.f16x2 r392, r368, r389;
}
{
cvt.rn.f16.f64 rs44, fd204;
}
mov.b32 r397, {rs44, rs44};
{
mul.f16x2 r395, r19, r397;
}
{
add.f16x2 r398, r374, r395;
}
{
cvt.rn.f16.f64 rs45, fd243;
}
mov.b32 r403, {rs45, rs45};
{
mul.f16x2 r401, r25, r403;
}
{
add.f16x2 r404, r380, r401;
}
{
cvt.rn.f16.f64 rs46, fd244;
}
mov.b32 r409, {rs46, rs46};
{
mul.f16x2 r407, r34, r409;
}
{
add.f16x2 r410, r386, r407;
}
{
cvt.rn.f16.f64 rs47, fd243;
}
mov.b32 r415, {rs47, rs47};
{
mul.f16x2 r413, r28, r415;
}
{
add.f16x2 r416, r392, r413;
}
{
cvt.rn.f16.f64 rs48, fd244;
}
mov.b32 r421, {rs48, rs48};
{
mul.f16x2 r419, r31, r421;
}
{
add.f16x2 r422, r398, r419;
}
{
cvt.rn.f16.f64 rs49, fd227;
}
mov.b32 r427, {rs49, rs49};
{
mul.f16x2 r425, r37, r427;
}
{
add.f16x2 r428, r404, r425;
}
{
cvt.rn.f16.f64 rs50, fd228;
}
mov.b32 r433, {rs50, rs50};
{
mul.f16x2 r431, r46, r433;
}
{
add.f16x2 r434, r410, r431;
}
{
cvt.rn.f16.f64 rs51, fd227;
}
mov.b32 r439, {rs51, rs51};
{
mul.f16x2 r437, r40, r439;
}
{
add.f16x2 r440, r416, r437;
}
{
cvt.rn.f16.f64 rs52, fd228;
}
mov.b32 r445, {rs52, rs52};
{
mul.f16x2 r443, r43, r445;
}
{
add.f16x2 r446, r422, r443;
}
{
cvt.rn.f16.f64 rs53, fd235;
}
mov.b32 r451, {rs53, rs53};
{
mul.f16x2 r449, r49, r451;
}
{
add.f16x2 r452, r428, r449;
}
mov.f64 fd136, 0dBFE0D8884363DD80;
{
cvt.rn.f16.f64 rs54, fd136;
}
mov.b32 r457, {rs54, rs54};
{
mul.f16x2 r455, r58, r457;
}
{
add.f16x2 r458, r434, r455;
}
{
cvt.rn.f16.f64 rs55, fd235;
}
mov.b32 r463, {rs55, rs55};
{
mul.f16x2 r461, r52, r463;
}
{
add.f16x2 r464, r440, r461;
}
{
cvt.rn.f16.f64 rs56, fd136;
}
mov.b32 r469, {rs56, rs56};
{
mul.f16x2 r467, r55, r469;
}
{
add.f16x2 r470, r446, r467;
}
{
cvt.rn.f16.f64 rs57, fd251;
}
mov.b32 r475, {rs57, rs57};
{
mul.f16x2 r473, r61, r475;
}
{
add.f16x2 r476, r452, r473;
}
mov.f64 fd168, 0dBFEEC746923C349F;
{
cvt.rn.f16.f64 rs58, fd168;
}
mov.b32 r481, {rs58, rs58};
{
mul.f16x2 r479, r70, r481;
}
{
add.f16x2 r482, r458, r479;
}
{
cvt.rn.f16.f64 rs59, fd251;
}
mov.b32 r487, {rs59, rs59};
{
mul.f16x2 r485, r64, r487;
}
{
add.f16x2 r488, r464, r485;
}
{
cvt.rn.f16.f64 rs60, fd168;
}
mov.b32 r493, {rs60, rs60};
{
mul.f16x2 r491, r67, r493;
}
{
add.f16x2 r494, r470, r491;
}
{
cvt.rn.f16.f64 rs61, fd247;
}
mov.b32 r499, {rs61, rs61};
{
mul.f16x2 r497, r73, r499;
}
{
add.f16x2 r500, r476, r497;
}
mov.f64 fd248, 0dBFECA52D7C9E640B;
{
cvt.rn.f16.f64 rs62, fd248;
}
mov.b32 r505, {rs62, rs62};
{
mul.f16x2 r503, r82, r505;
}
{
add.f16x2 r506, r482, r503;
}
{
cvt.rn.f16.f64 rs63, fd247;
}
mov.b32 r511, {rs63, rs63};
{
mul.f16x2 r509, r76, r511;
}
{
add.f16x2 r512, r488, r509;
}
{
cvt.rn.f16.f64 rs64, fd248;
}
mov.b32 r517, {rs64, rs64};
{
mul.f16x2 r515, r79, r517;
}
{
add.f16x2 r518, r494, r515;
}
{
cvt.rn.f16.f64 rs65, fd231;
}
mov.b32 r523, {rs65, rs65};
{
mul.f16x2 r521, r85, r523;
}
{
add.f16x2 r524, r500, r521;
}
mov.f64 fd232, 0dBFD71E955D8E7CDC;
{
cvt.rn.f16.f64 rs66, fd232;
}
mov.b32 r529, {rs66, rs66};
{
mul.f16x2 r527, r94, r529;
}
{
add.f16x2 r530, r506, r527;
}
{
cvt.rn.f16.f64 rs67, fd231;
}
mov.b32 r535, {rs67, rs67};
{
mul.f16x2 r533, r88, r535;
}
{
add.f16x2 r536, r512, r533;
}
{
cvt.rn.f16.f64 rs68, fd232;
}
mov.b32 r541, {rs68, rs68};
{
mul.f16x2 r539, r91, r541;
}
{
add.f16x2 r542, r518, r539;
}
{
sub.f16x2 %4, r524, r530;
}
{
add.f16x2 %5, r536, r542;
}
{
add.f16x2 %30, r524, r530;
}
{
sub.f16x2 %31, r536, r542;
}
cvt.rn.f16.s32 rs69, r1588;
mov.b32 r569, {rs69, rs69};
cvt.rn.f16.s32 rs70, r1588;
mov.b32 r581, {rs70, rs70};
{
cvt.rn.f16.f64 rs71, fd247;
}
mov.b32 r561, {rs71, rs71};
{
mul.f16x2 r559, r1, r561;
}
{
add.f16x2 r562, %34, r559;
}
{
cvt.rn.f16.f64 rs72, fd144;
}
mov.b32 r567, {rs72, rs72};
{
mul.f16x2 r565, r10, r567;
}
{
add.f16x2 r568, r569, r565;
}
{
cvt.rn.f16.f64 rs73, fd247;
}
mov.b32 r573, {rs73, rs73};
{
mul.f16x2 r571, r4, r573;
}
{
add.f16x2 r574, %35, r571;
}
{
cvt.rn.f16.f64 rs74, fd144;
}
mov.b32 r579, {rs74, rs74};
{
mul.f16x2 r577, r7, r579;
}
{
add.f16x2 r580, r581, r577;
}
{
cvt.rn.f16.f64 rs75, fd243;
}
mov.b32 r585, {rs75, rs75};
{
mul.f16x2 r583, r13, r585;
}
{
add.f16x2 r586, r562, r583;
}
{
cvt.rn.f16.f64 rs76, fd244;
}
mov.b32 r591, {rs76, rs76};
{
mul.f16x2 r589, r22, r591;
}
{
add.f16x2 r592, r568, r589;
}
{
cvt.rn.f16.f64 rs77, fd243;
}
mov.b32 r597, {rs77, rs77};
{
mul.f16x2 r595, r16, r597;
}
{
add.f16x2 r598, r574, r595;
}
{
cvt.rn.f16.f64 rs78, fd244;
}
mov.b32 r603, {rs78, rs78};
{
mul.f16x2 r601, r19, r603;
}
{
add.f16x2 r604, r580, r601;
}
{
cvt.rn.f16.f64 rs79, fd227;
}
mov.b32 r609, {rs79, rs79};
{
mul.f16x2 r607, r25, r609;
}
{
add.f16x2 r610, r586, r607;
}
mov.f64 fd76, 0dBFC7851AACD6C6B4;
{
cvt.rn.f16.f64 rs80, fd76;
}
mov.b32 r615, {rs80, rs80};
{
mul.f16x2 r613, r34, r615;
}
{
add.f16x2 r616, r592, r613;
}
{
cvt.rn.f16.f64 rs81, fd227;
}
mov.b32 r621, {rs81, rs81};
{
mul.f16x2 r619, r28, r621;
}
{
add.f16x2 r622, r598, r619;
}
{
cvt.rn.f16.f64 rs82, fd76;
}
mov.b32 r627, {rs82, rs82};
{
mul.f16x2 r625, r31, r627;
}
{
add.f16x2 r628, r604, r625;
}
{
cvt.rn.f16.f64 rs83, fd251;
}
mov.b32 r633, {rs83, rs83};
{
mul.f16x2 r631, r37, r633;
}
{
add.f16x2 r634, r610, r631;
}
{
cvt.rn.f16.f64 rs84, fd168;
}
mov.b32 r639, {rs84, rs84};
{
mul.f16x2 r637, r46, r639;
}
{
add.f16x2 r640, r616, r637;
}
{
cvt.rn.f16.f64 rs85, fd251;
}
mov.b32 r645, {rs85, rs85};
{
mul.f16x2 r643, r40, r645;
}
{
add.f16x2 r646, r622, r643;
}
{
cvt.rn.f16.f64 rs86, fd168;
}
mov.b32 r651, {rs86, rs86};
{
mul.f16x2 r649, r43, r651;
}
{
add.f16x2 r652, r628, r649;
}
{
cvt.rn.f16.f64 rs87, fd239;
}
mov.b32 r657, {rs87, rs87};
{
mul.f16x2 r655, r49, r657;
}
{
add.f16x2 r658, r634, r655;
}
mov.f64 fd240, 0dBFE58EEA2A9D6DA3;
{
cvt.rn.f16.f64 rs88, fd240;
}
mov.b32 r663, {rs88, rs88};
{
mul.f16x2 r661, r58, r663;
}
{
add.f16x2 r664, r640, r661;
}
{
cvt.rn.f16.f64 rs89, fd239;
}
mov.b32 r669, {rs89, rs89};
{
mul.f16x2 r667, r52, r669;
}
{
add.f16x2 r670, r646, r667;
}
{
cvt.rn.f16.f64 rs90, fd240;
}
mov.b32 r675, {rs90, rs90};
{
mul.f16x2 r673, r55, r675;
}
{
add.f16x2 r676, r652, r673;
}
{
cvt.rn.f16.f64 rs91, fd231;
}
mov.b32 r681, {rs91, rs91};
{
mul.f16x2 r679, r61, r681;
}
{
add.f16x2 r682, r658, r679;
}
{
cvt.rn.f16.f64 rs92, fd212;
}
mov.b32 r687, {rs92, rs92};
{
mul.f16x2 r685, r70, r687;
}
{
add.f16x2 r688, r664, r685;
}
{
cvt.rn.f16.f64 rs93, fd231;
}
mov.b32 r693, {rs93, rs93};
{
mul.f16x2 r691, r64, r693;
}
{
add.f16x2 r694, r670, r691;
}
{
cvt.rn.f16.f64 rs94, fd212;
}
mov.b32 r699, {rs94, rs94};
{
mul.f16x2 r697, r67, r699;
}
{
add.f16x2 r700, r676, r697;
}
{
cvt.rn.f16.f64 rs95, fd255;
}
mov.b32 r705, {rs95, rs95};
{
mul.f16x2 r703, r73, r705;
}
{
add.f16x2 r706, r682, r703;
}
{
cvt.rn.f16.f64 rs96, fd204;
}
mov.b32 r711, {rs96, rs96};
{
mul.f16x2 r709, r82, r711;
}
{
add.f16x2 r712, r688, r709;
}
{
cvt.rn.f16.f64 rs97, fd255;
}
mov.b32 r717, {rs97, rs97};
{
mul.f16x2 r715, r76, r717;
}
{
add.f16x2 r718, r694, r715;
}
{
cvt.rn.f16.f64 rs98, fd204;
}
mov.b32 r723, {rs98, rs98};
{
mul.f16x2 r721, r79, r723;
}
{
add.f16x2 r724, r700, r721;
}
{
cvt.rn.f16.f64 rs99, fd235;
}
mov.b32 r729, {rs99, rs99};
{
mul.f16x2 r727, r85, r729;
}
{
add.f16x2 r730, r706, r727;
}
{
cvt.rn.f16.f64 rs100, fd236;
}
mov.b32 r735, {rs100, rs100};
{
mul.f16x2 r733, r94, r735;
}
{
add.f16x2 r736, r712, r733;
}
{
cvt.rn.f16.f64 rs101, fd235;
}
mov.b32 r741, {rs101, rs101};
{
mul.f16x2 r739, r88, r741;
}
{
add.f16x2 r742, r718, r739;
}
{
cvt.rn.f16.f64 rs102, fd236;
}
mov.b32 r747, {rs102, rs102};
{
mul.f16x2 r745, r91, r747;
}
{
add.f16x2 r748, r724, r745;
}
{
sub.f16x2 %6, r730, r736;
}
{
add.f16x2 %7, r742, r748;
}
{
add.f16x2 %28, r730, r736;
}
{
sub.f16x2 %29, r742, r748;
}
cvt.rn.f16.s32 rs103, r1588;
mov.b32 r775, {rs103, rs103};
cvt.rn.f16.s32 rs104, r1588;
mov.b32 r787, {rs104, rs104};
{
cvt.rn.f16.f64 rs105, fd255;
}
mov.b32 r767, {rs105, rs105};
{
mul.f16x2 r765, r1, r767;
}
{
add.f16x2 r768, %34, r765;
}
{
cvt.rn.f16.f64 rs106, fd204;
}
mov.b32 r773, {rs106, rs106};
{
mul.f16x2 r771, r10, r773;
}
{
add.f16x2 r774, r775, r771;
}
{
cvt.rn.f16.f64 rs107, fd255;
}
mov.b32 r779, {rs107, rs107};
{
mul.f16x2 r777, r4, r779;
}
{
add.f16x2 r780, %35, r777;
}
{
cvt.rn.f16.f64 rs108, fd204;
}
mov.b32 r785, {rs108, rs108};
{
mul.f16x2 r783, r7, r785;
}
{
add.f16x2 r786, r787, r783;
}
{
cvt.rn.f16.f64 rs109, fd227;
}
mov.b32 r791, {rs109, rs109};
{
mul.f16x2 r789, r13, r791;
}
{
add.f16x2 r792, r768, r789;
}
{
cvt.rn.f16.f64 rs110, fd228;
}
mov.b32 r797, {rs110, rs110};
{
mul.f16x2 r795, r22, r797;
}
{
add.f16x2 r798, r774, r795;
}
{
cvt.rn.f16.f64 rs111, fd227;
}
mov.b32 r803, {rs111, rs111};
{
mul.f16x2 r801, r16, r803;
}
{
add.f16x2 r804, r780, r801;
}
{
cvt.rn.f16.f64 rs112, fd228;
}
mov.b32 r809, {rs112, rs112};
{
mul.f16x2 r807, r19, r809;
}
{
add.f16x2 r810, r786, r807;
}
{
cvt.rn.f16.f64 rs113, fd251;
}
mov.b32 r815, {rs113, rs113};
{
mul.f16x2 r813, r25, r815;
}
{
add.f16x2 r816, r792, r813;
}
{
cvt.rn.f16.f64 rs114, fd168;
}
mov.b32 r821, {rs114, rs114};
{
mul.f16x2 r819, r34, r821;
}
{
add.f16x2 r822, r798, r819;
}
{
cvt.rn.f16.f64 rs115, fd251;
}
mov.b32 r827, {rs115, rs115};
{
mul.f16x2 r825, r28, r827;
}
{
add.f16x2 r828, r804, r825;
}
{
cvt.rn.f16.f64 rs116, fd168;
}
mov.b32 r833, {rs116, rs116};
{
mul.f16x2 r831, r31, r833;
}
{
add.f16x2 r834, r810, r831;
}
{
cvt.rn.f16.f64 rs117, fd231;
}
mov.b32 r839, {rs117, rs117};
{
mul.f16x2 r837, r37, r839;
}
{
add.f16x2 r840, r816, r837;
}
{
cvt.rn.f16.f64 rs118, fd232;
}
mov.b32 r845, {rs118, rs118};
{
mul.f16x2 r843, r46, r845;
}
{
add.f16x2 r846, r822, r843;
}
{
cvt.rn.f16.f64 rs119, fd231;
}
mov.b32 r851, {rs119, rs119};
{
mul.f16x2 r849, r40, r851;
}
{
add.f16x2 r852, r828, r849;
}
{
cvt.rn.f16.f64 rs120, fd232;
}
mov.b32 r857, {rs120, rs120};
{
mul.f16x2 r855, r43, r857;
}
{
add.f16x2 r858, r834, r855;
}
{
cvt.rn.f16.f64 rs121, fd247;
}
mov.b32 r863, {rs121, rs121};
{
mul.f16x2 r861, r49, r863;
}
{
add.f16x2 r864, r840, r861;
}
{
cvt.rn.f16.f64 rs122, fd144;
}
mov.b32 r869, {rs122, rs122};
{
mul.f16x2 r867, r58, r869;
}
{
add.f16x2 r870, r846, r867;
}
{
cvt.rn.f16.f64 rs123, fd247;
}
mov.b32 r875, {rs123, rs123};
{
mul.f16x2 r873, r52, r875;
}
{
add.f16x2 r876, r852, r873;
}
{
cvt.rn.f16.f64 rs124, fd144;
}
mov.b32 r881, {rs124, rs124};
{
mul.f16x2 r879, r55, r881;
}
{
add.f16x2 r882, r858, r879;
}
{
cvt.rn.f16.f64 rs125, fd235;
}
mov.b32 r887, {rs125, rs125};
{
mul.f16x2 r885, r61, r887;
}
{
add.f16x2 r888, r864, r885;
}
{
cvt.rn.f16.f64 rs126, fd236;
}
mov.b32 r893, {rs126, rs126};
{
mul.f16x2 r891, r70, r893;
}
{
add.f16x2 r894, r870, r891;
}
{
cvt.rn.f16.f64 rs127, fd235;
}
mov.b32 r899, {rs127, rs127};
{
mul.f16x2 r897, r64, r899;
}
{
add.f16x2 r900, r876, r897;
}
{
cvt.rn.f16.f64 rs128, fd236;
}
mov.b32 r905, {rs128, rs128};
{
mul.f16x2 r903, r67, r905;
}
{
add.f16x2 r906, r882, r903;
}
{
cvt.rn.f16.f64 rs129, fd243;
}
mov.b32 r911, {rs129, rs129};
{
mul.f16x2 r909, r73, r911;
}
{
add.f16x2 r912, r888, r909;
}
mov.f64 fd208, 0dBFE9895B6C9A05F6;
{
cvt.rn.f16.f64 rs130, fd208;
}
mov.b32 r917, {rs130, rs130};
{
mul.f16x2 r915, r82, r917;
}
{
add.f16x2 r918, r894, r915;
}
{
cvt.rn.f16.f64 rs131, fd243;
}
mov.b32 r923, {rs131, rs131};
{
mul.f16x2 r921, r76, r923;
}
{
add.f16x2 r924, r900, r921;
}
{
cvt.rn.f16.f64 rs132, fd208;
}
mov.b32 r929, {rs132, rs132};
{
mul.f16x2 r927, r79, r929;
}
{
add.f16x2 r930, r906, r927;
}
{
cvt.rn.f16.f64 rs133, fd239;
}
mov.b32 r935, {rs133, rs133};
{
mul.f16x2 r933, r85, r935;
}
{
add.f16x2 r936, r912, r933;
}
{
cvt.rn.f16.f64 rs134, fd240;
}
mov.b32 r941, {rs134, rs134};
{
mul.f16x2 r939, r94, r941;
}
{
add.f16x2 r942, r918, r939;
}
{
cvt.rn.f16.f64 rs135, fd239;
}
mov.b32 r947, {rs135, rs135};
{
mul.f16x2 r945, r88, r947;
}
{
add.f16x2 r948, r924, r945;
}
{
cvt.rn.f16.f64 rs136, fd240;
}
mov.b32 r953, {rs136, rs136};
{
mul.f16x2 r951, r91, r953;
}
{
add.f16x2 r954, r930, r951;
}
{
sub.f16x2 %8, r936, r942;
}
{
add.f16x2 %9, r948, r954;
}
{
add.f16x2 %26, r936, r942;
}
{
sub.f16x2 %27, r948, r954;
}
cvt.rn.f16.s32 rs137, r1588;
mov.b32 r981, {rs137, rs137};
cvt.rn.f16.s32 rs138, r1588;
mov.b32 r993, {rs138, rs138};
{
cvt.rn.f16.f64 rs139, fd251;
}
mov.b32 r973, {rs139, rs139};
{
mul.f16x2 r971, r1, r973;
}
{
add.f16x2 r974, %34, r971;
}
{
cvt.rn.f16.f64 rs140, fd252;
}
mov.b32 r979, {rs140, rs140};
{
mul.f16x2 r977, r10, r979;
}
{
add.f16x2 r980, r981, r977;
}
{
cvt.rn.f16.f64 rs141, fd251;
}
mov.b32 r985, {rs141, rs141};
{
mul.f16x2 r983, r4, r985;
}
{
add.f16x2 r986, %35, r983;
}
{
cvt.rn.f16.f64 rs142, fd252;
}
mov.b32 r991, {rs142, rs142};
{
mul.f16x2 r989, r7, r991;
}
{
add.f16x2 r992, r993, r989;
}
{
cvt.rn.f16.f64 rs143, fd235;
}
mov.b32 r997, {rs143, rs143};
{
mul.f16x2 r995, r13, r997;
}
{
add.f16x2 r998, r974, r995;
}
{
cvt.rn.f16.f64 rs144, fd136;
}
mov.b32 r1003, {rs144, rs144};
{
mul.f16x2 r1001, r22, r1003;
}
{
add.f16x2 r1004, r980, r1001;
}
{
cvt.rn.f16.f64 rs145, fd235;
}
mov.b32 r1009, {rs145, rs145};
{
mul.f16x2 r1007, r16, r1009;
}
{
add.f16x2 r1010, r986, r1007;
}
{
cvt.rn.f16.f64 rs146, fd136;
}
mov.b32 r1015, {rs146, rs146};
{
mul.f16x2 r1013, r19, r1015;
}
{
add.f16x2 r1016, r992, r1013;
}
{
cvt.rn.f16.f64 rs147, fd239;
}
mov.b32 r1021, {rs147, rs147};
{
mul.f16x2 r1019, r25, r1021;
}
{
add.f16x2 r1022, r998, r1019;
}
{
cvt.rn.f16.f64 rs148, fd240;
}
mov.b32 r1027, {rs148, rs148};
{
mul.f16x2 r1025, r34, r1027;
}
{
add.f16x2 r1028, r1004, r1025;
}
{
cvt.rn.f16.f64 rs149, fd239;
}
mov.b32 r1033, {rs149, rs149};
{
mul.f16x2 r1031, r28, r1033;
}
{
add.f16x2 r1034, r1010, r1031;
}
{
cvt.rn.f16.f64 rs150, fd240;
}
mov.b32 r1039, {rs150, rs150};
{
mul.f16x2 r1037, r31, r1039;
}
{
add.f16x2 r1040, r1016, r1037;
}
{
cvt.rn.f16.f64 rs151, fd247;
}
mov.b32 r1045, {rs151, rs151};
{
mul.f16x2 r1043, r37, r1045;
}
{
add.f16x2 r1046, r1022, r1043;
}
{
cvt.rn.f16.f64 rs152, fd144;
}
mov.b32 r1051, {rs152, rs152};
{
mul.f16x2 r1049, r46, r1051;
}
{
add.f16x2 r1052, r1028, r1049;
}
{
cvt.rn.f16.f64 rs153, fd247;
}
mov.b32 r1057, {rs153, rs153};
{
mul.f16x2 r1055, r40, r1057;
}
{
add.f16x2 r1058, r1034, r1055;
}
{
cvt.rn.f16.f64 rs154, fd144;
}
mov.b32 r1063, {rs154, rs154};
{
mul.f16x2 r1061, r43, r1063;
}
{
add.f16x2 r1064, r1040, r1061;
}
{
cvt.rn.f16.f64 rs155, fd227;
}
mov.b32 r1069, {rs155, rs155};
{
mul.f16x2 r1067, r49, r1069;
}
{
add.f16x2 r1070, r1046, r1067;
}
{
cvt.rn.f16.f64 rs156, fd228;
}
mov.b32 r1075, {rs156, rs156};
{
mul.f16x2 r1073, r58, r1075;
}
{
add.f16x2 r1076, r1052, r1073;
}
{
cvt.rn.f16.f64 rs157, fd227;
}
mov.b32 r1081, {rs157, rs157};
{
mul.f16x2 r1079, r52, r1081;
}
{
add.f16x2 r1082, r1058, r1079;
}
{
cvt.rn.f16.f64 rs158, fd228;
}
mov.b32 r1087, {rs158, rs158};
{
mul.f16x2 r1085, r55, r1087;
}
{
add.f16x2 r1088, r1064, r1085;
}
{
cvt.rn.f16.f64 rs159, fd255;
}
mov.b32 r1093, {rs159, rs159};
{
mul.f16x2 r1091, r61, r1093;
}
{
add.f16x2 r1094, r1070, r1091;
}
mov.f64 fd256, 0dBFEFDD0DEB564B22;
{
cvt.rn.f16.f64 rs160, fd256;
}
mov.b32 r1099, {rs160, rs160};
{
mul.f16x2 r1097, r70, r1099;
}
{
add.f16x2 r1100, r1076, r1097;
}
{
cvt.rn.f16.f64 rs161, fd255;
}
mov.b32 r1105, {rs161, rs161};
{
mul.f16x2 r1103, r64, r1105;
}
{
add.f16x2 r1106, r1082, r1103;
}
{
cvt.rn.f16.f64 rs162, fd256;
}
mov.b32 r1111, {rs162, rs162};
{
mul.f16x2 r1109, r67, r1111;
}
{
add.f16x2 r1112, r1088, r1109;
}
{
cvt.rn.f16.f64 rs163, fd231;
}
mov.b32 r1117, {rs163, rs163};
{
mul.f16x2 r1115, r73, r1117;
}
{
add.f16x2 r1118, r1094, r1115;
}
{
cvt.rn.f16.f64 rs164, fd212;
}
mov.b32 r1123, {rs164, rs164};
{
mul.f16x2 r1121, r82, r1123;
}
{
add.f16x2 r1124, r1100, r1121;
}
{
cvt.rn.f16.f64 rs165, fd231;
}
mov.b32 r1129, {rs165, rs165};
{
mul.f16x2 r1127, r76, r1129;
}
{
add.f16x2 r1130, r1106, r1127;
}
{
cvt.rn.f16.f64 rs166, fd212;
}
mov.b32 r1135, {rs166, rs166};
{
mul.f16x2 r1133, r79, r1135;
}
{
add.f16x2 r1136, r1112, r1133;
}
{
cvt.rn.f16.f64 rs167, fd243;
}
mov.b32 r1141, {rs167, rs167};
{
mul.f16x2 r1139, r85, r1141;
}
{
add.f16x2 r1142, r1118, r1139;
}
{
cvt.rn.f16.f64 rs168, fd244;
}
mov.b32 r1147, {rs168, rs168};
{
mul.f16x2 r1145, r94, r1147;
}
{
add.f16x2 r1148, r1124, r1145;
}
{
cvt.rn.f16.f64 rs169, fd243;
}
mov.b32 r1153, {rs169, rs169};
{
mul.f16x2 r1151, r88, r1153;
}
{
add.f16x2 r1154, r1130, r1151;
}
{
cvt.rn.f16.f64 rs170, fd244;
}
mov.b32 r1159, {rs170, rs170};
{
mul.f16x2 r1157, r91, r1159;
}
{
add.f16x2 r1160, r1136, r1157;
}
{
sub.f16x2 %10, r1142, r1148;
}
{
add.f16x2 %11, r1154, r1160;
}
{
add.f16x2 %24, r1142, r1148;
}
{
sub.f16x2 %25, r1154, r1160;
}
cvt.rn.f16.s32 rs171, r1588;
mov.b32 r1187, {rs171, rs171};
cvt.rn.f16.s32 rs172, r1588;
mov.b32 r1199, {rs172, rs172};
{
cvt.rn.f16.f64 rs173, fd243;
}
mov.b32 r1179, {rs173, rs173};
{
mul.f16x2 r1177, r1, r1179;
}
{
add.f16x2 r1180, %34, r1177;
}
{
cvt.rn.f16.f64 rs174, fd244;
}
mov.b32 r1185, {rs174, rs174};
{
mul.f16x2 r1183, r10, r1185;
}
{
add.f16x2 r1186, r1187, r1183;
}
{
cvt.rn.f16.f64 rs175, fd243;
}
mov.b32 r1191, {rs175, rs175};
{
mul.f16x2 r1189, r4, r1191;
}
{
add.f16x2 r1192, %35, r1189;
}
{
cvt.rn.f16.f64 rs176, fd244;
}
mov.b32 r1197, {rs176, rs176};
{
mul.f16x2 r1195, r7, r1197;
}
{
add.f16x2 r1198, r1199, r1195;
}
{
cvt.rn.f16.f64 rs177, fd251;
}
mov.b32 r1203, {rs177, rs177};
{
mul.f16x2 r1201, r13, r1203;
}
{
add.f16x2 r1204, r1180, r1201;
}
{
cvt.rn.f16.f64 rs178, fd168;
}
mov.b32 r1209, {rs178, rs178};
{
mul.f16x2 r1207, r22, r1209;
}
{
add.f16x2 r1210, r1186, r1207;
}
{
cvt.rn.f16.f64 rs179, fd251;
}
mov.b32 r1215, {rs179, rs179};
{
mul.f16x2 r1213, r16, r1215;
}
{
add.f16x2 r1216, r1192, r1213;
}
{
cvt.rn.f16.f64 rs180, fd168;
}
mov.b32 r1221, {rs180, rs180};
{
mul.f16x2 r1219, r19, r1221;
}
{
add.f16x2 r1222, r1198, r1219;
}
{
cvt.rn.f16.f64 rs181, fd231;
}
mov.b32 r1227, {rs181, rs181};
{
mul.f16x2 r1225, r25, r1227;
}
{
add.f16x2 r1228, r1204, r1225;
}
{
cvt.rn.f16.f64 rs182, fd212;
}
mov.b32 r1233, {rs182, rs182};
{
mul.f16x2 r1231, r34, r1233;
}
{
add.f16x2 r1234, r1210, r1231;
}
{
cvt.rn.f16.f64 rs183, fd231;
}
mov.b32 r1239, {rs183, rs183};
{
mul.f16x2 r1237, r28, r1239;
}
{
add.f16x2 r1240, r1216, r1237;
}
{
cvt.rn.f16.f64 rs184, fd212;
}
mov.b32 r1245, {rs184, rs184};
{
mul.f16x2 r1243, r31, r1245;
}
{
add.f16x2 r1246, r1222, r1243;
}
{
cvt.rn.f16.f64 rs185, fd235;
}
mov.b32 r1251, {rs185, rs185};
{
mul.f16x2 r1249, r37, r1251;
}
{
add.f16x2 r1252, r1228, r1249;
}
{
cvt.rn.f16.f64 rs186, fd236;
}
mov.b32 r1257, {rs186, rs186};
{
mul.f16x2 r1255, r46, r1257;
}
{
add.f16x2 r1258, r1234, r1255;
}
{
cvt.rn.f16.f64 rs187, fd235;
}
mov.b32 r1263, {rs187, rs187};
{
mul.f16x2 r1261, r40, r1263;
}
{
add.f16x2 r1264, r1240, r1261;
}
{
cvt.rn.f16.f64 rs188, fd236;
}
mov.b32 r1269, {rs188, rs188};
{
mul.f16x2 r1267, r43, r1269;
}
{
add.f16x2 r1270, r1246, r1267;
}
{
cvt.rn.f16.f64 rs189, fd255;
}
mov.b32 r1275, {rs189, rs189};
{
mul.f16x2 r1273, r49, r1275;
}
{
add.f16x2 r1276, r1252, r1273;
}
{
cvt.rn.f16.f64 rs190, fd256;
}
mov.b32 r1281, {rs190, rs190};
{
mul.f16x2 r1279, r58, r1281;
}
{
add.f16x2 r1282, r1258, r1279;
}
{
cvt.rn.f16.f64 rs191, fd255;
}
mov.b32 r1287, {rs191, rs191};
{
mul.f16x2 r1285, r52, r1287;
}
{
add.f16x2 r1288, r1264, r1285;
}
{
cvt.rn.f16.f64 rs192, fd256;
}
mov.b32 r1293, {rs192, rs192};
{
mul.f16x2 r1291, r55, r1293;
}
{
add.f16x2 r1294, r1270, r1291;
}
{
cvt.rn.f16.f64 rs193, fd239;
}
mov.b32 r1299, {rs193, rs193};
{
mul.f16x2 r1297, r61, r1299;
}
{
add.f16x2 r1300, r1276, r1297;
}
{
cvt.rn.f16.f64 rs194, fd184;
}
mov.b32 r1305, {rs194, rs194};
{
mul.f16x2 r1303, r70, r1305;
}
{
add.f16x2 r1306, r1282, r1303;
}
{
cvt.rn.f16.f64 rs195, fd239;
}
mov.b32 r1311, {rs195, rs195};
{
mul.f16x2 r1309, r64, r1311;
}
{
add.f16x2 r1312, r1288, r1309;
}
{
cvt.rn.f16.f64 rs196, fd184;
}
mov.b32 r1317, {rs196, rs196};
{
mul.f16x2 r1315, r67, r1317;
}
{
add.f16x2 r1318, r1294, r1315;
}
{
cvt.rn.f16.f64 rs197, fd227;
}
mov.b32 r1323, {rs197, rs197};
{
mul.f16x2 r1321, r73, r1323;
}
{
add.f16x2 r1324, r1300, r1321;
}
{
cvt.rn.f16.f64 rs198, fd228;
}
mov.b32 r1329, {rs198, rs198};
{
mul.f16x2 r1327, r82, r1329;
}
{
add.f16x2 r1330, r1306, r1327;
}
{
cvt.rn.f16.f64 rs199, fd227;
}
mov.b32 r1335, {rs199, rs199};
{
mul.f16x2 r1333, r76, r1335;
}
{
add.f16x2 r1336, r1312, r1333;
}
{
cvt.rn.f16.f64 rs200, fd228;
}
mov.b32 r1341, {rs200, rs200};
{
mul.f16x2 r1339, r79, r1341;
}
{
add.f16x2 r1342, r1318, r1339;
}
{
cvt.rn.f16.f64 rs201, fd247;
}
mov.b32 r1347, {rs201, rs201};
{
mul.f16x2 r1345, r85, r1347;
}
{
add.f16x2 r1348, r1324, r1345;
}
{
cvt.rn.f16.f64 rs202, fd248;
}
mov.b32 r1353, {rs202, rs202};
{
mul.f16x2 r1351, r94, r1353;
}
{
add.f16x2 r1354, r1330, r1351;
}
{
cvt.rn.f16.f64 rs203, fd247;
}
mov.b32 r1359, {rs203, rs203};
{
mul.f16x2 r1357, r88, r1359;
}
{
add.f16x2 r1360, r1336, r1357;
}
{
cvt.rn.f16.f64 rs204, fd248;
}
mov.b32 r1365, {rs204, rs204};
{
mul.f16x2 r1363, r91, r1365;
}
{
add.f16x2 r1366, r1342, r1363;
}
{
sub.f16x2 %12, r1348, r1354;
}
{
add.f16x2 %13, r1360, r1366;
}
{
add.f16x2 %22, r1348, r1354;
}
{
sub.f16x2 %23, r1360, r1366;
}
cvt.rn.f16.s32 rs205, r1588;
mov.b32 r1393, {rs205, rs205};
cvt.rn.f16.s32 rs206, r1588;
mov.b32 r1405, {rs206, rs206};
{
cvt.rn.f16.f64 rs207, fd235;
}
mov.b32 r1385, {rs207, rs207};
{
mul.f16x2 r1383, r1, r1385;
}
{
add.f16x2 r1386, %34, r1383;
}
{
cvt.rn.f16.f64 rs208, fd236;
}
mov.b32 r1391, {rs208, rs208};
{
mul.f16x2 r1389, r10, r1391;
}
{
add.f16x2 r1392, r1393, r1389;
}
{
cvt.rn.f16.f64 rs209, fd235;
}
mov.b32 r1397, {rs209, rs209};
{
mul.f16x2 r1395, r4, r1397;
}
{
add.f16x2 r1398, %35, r1395;
}
{
cvt.rn.f16.f64 rs210, fd236;
}
mov.b32 r1403, {rs210, rs210};
{
mul.f16x2 r1401, r7, r1403;
}
{
add.f16x2 r1404, r1405, r1401;
}
{
cvt.rn.f16.f64 rs211, fd247;
}
mov.b32 r1409, {rs211, rs211};
{
mul.f16x2 r1407, r13, r1409;
}
{
add.f16x2 r1410, r1386, r1407;
}
{
cvt.rn.f16.f64 rs212, fd248;
}
mov.b32 r1415, {rs212, rs212};
{
mul.f16x2 r1413, r22, r1415;
}
{
add.f16x2 r1416, r1392, r1413;
}
{
cvt.rn.f16.f64 rs213, fd247;
}
mov.b32 r1421, {rs213, rs213};
{
mul.f16x2 r1419, r16, r1421;
}
{
add.f16x2 r1422, r1398, r1419;
}
{
cvt.rn.f16.f64 rs214, fd248;
}
mov.b32 r1427, {rs214, rs214};
{
mul.f16x2 r1425, r19, r1427;
}
{
add.f16x2 r1428, r1404, r1425;
}
{
cvt.rn.f16.f64 rs215, fd255;
}
mov.b32 r1433, {rs215, rs215};
{
mul.f16x2 r1431, r25, r1433;
}
{
add.f16x2 r1434, r1410, r1431;
}
{
cvt.rn.f16.f64 rs216, fd204;
}
mov.b32 r1439, {rs216, rs216};
{
mul.f16x2 r1437, r34, r1439;
}
{
add.f16x2 r1440, r1416, r1437;
}
{
cvt.rn.f16.f64 rs217, fd255;
}
mov.b32 r1445, {rs217, rs217};
{
mul.f16x2 r1443, r28, r1445;
}
{
add.f16x2 r1446, r1422, r1443;
}
{
cvt.rn.f16.f64 rs218, fd204;
}
mov.b32 r1451, {rs218, rs218};
{
mul.f16x2 r1449, r31, r1451;
}
{
add.f16x2 r1452, r1428, r1449;
}
{
cvt.rn.f16.f64 rs219, fd243;
}
mov.b32 r1457, {rs219, rs219};
{
mul.f16x2 r1455, r37, r1457;
}
{
add.f16x2 r1458, r1434, r1455;
}
{
cvt.rn.f16.f64 rs220, fd208;
}
mov.b32 r1463, {rs220, rs220};
{
mul.f16x2 r1461, r46, r1463;
}
{
add.f16x2 r1464, r1440, r1461;
}
{
cvt.rn.f16.f64 rs221, fd243;
}
mov.b32 r1469, {rs221, rs221};
{
mul.f16x2 r1467, r40, r1469;
}
{
add.f16x2 r1470, r1446, r1467;
}
{
cvt.rn.f16.f64 rs222, fd208;
}
mov.b32 r1475, {rs222, rs222};
{
mul.f16x2 r1473, r43, r1475;
}
{
add.f16x2 r1476, r1452, r1473;
}
{
cvt.rn.f16.f64 rs223, fd231;
}
mov.b32 r1481, {rs223, rs223};
{
mul.f16x2 r1479, r49, r1481;
}
{
add.f16x2 r1482, r1458, r1479;
}
{
cvt.rn.f16.f64 rs224, fd212;
}
mov.b32 r1487, {rs224, rs224};
{
mul.f16x2 r1485, r58, r1487;
}
{
add.f16x2 r1488, r1464, r1485;
}
{
cvt.rn.f16.f64 rs225, fd231;
}
mov.b32 r1493, {rs225, rs225};
{
mul.f16x2 r1491, r52, r1493;
}
{
add.f16x2 r1494, r1470, r1491;
}
{
cvt.rn.f16.f64 rs226, fd212;
}
mov.b32 r1499, {rs226, rs226};
{
mul.f16x2 r1497, r55, r1499;
}
{
add.f16x2 r1500, r1476, r1497;
}
{
cvt.rn.f16.f64 rs227, fd227;
}
mov.b32 r1505, {rs227, rs227};
{
mul.f16x2 r1503, r61, r1505;
}
{
add.f16x2 r1506, r1482, r1503;
}
{
cvt.rn.f16.f64 rs228, fd228;
}
mov.b32 r1511, {rs228, rs228};
{
mul.f16x2 r1509, r70, r1511;
}
{
add.f16x2 r1512, r1488, r1509;
}
{
cvt.rn.f16.f64 rs229, fd227;
}
mov.b32 r1517, {rs229, rs229};
{
mul.f16x2 r1515, r64, r1517;
}
{
add.f16x2 r1518, r1494, r1515;
}
{
cvt.rn.f16.f64 rs230, fd228;
}
mov.b32 r1523, {rs230, rs230};
{
mul.f16x2 r1521, r67, r1523;
}
{
add.f16x2 r1524, r1500, r1521;
}
{
cvt.rn.f16.f64 rs231, fd239;
}
mov.b32 r1529, {rs231, rs231};
{
mul.f16x2 r1527, r73, r1529;
}
{
add.f16x2 r1530, r1506, r1527;
}
{
cvt.rn.f16.f64 rs232, fd240;
}
mov.b32 r1535, {rs232, rs232};
{
mul.f16x2 r1533, r82, r1535;
}
{
add.f16x2 r1536, r1512, r1533;
}
{
cvt.rn.f16.f64 rs233, fd239;
}
mov.b32 r1541, {rs233, rs233};
{
mul.f16x2 r1539, r76, r1541;
}
{
add.f16x2 r1542, r1518, r1539;
}
{
cvt.rn.f16.f64 rs234, fd240;
}
mov.b32 r1547, {rs234, rs234};
{
mul.f16x2 r1545, r79, r1547;
}
{
add.f16x2 r1548, r1524, r1545;
}
{
cvt.rn.f16.f64 rs235, fd251;
}
mov.b32 r1553, {rs235, rs235};
{
mul.f16x2 r1551, r85, r1553;
}
{
add.f16x2 r1554, r1530, r1551;
}
{
cvt.rn.f16.f64 rs236, fd252;
}
mov.b32 r1559, {rs236, rs236};
{
mul.f16x2 r1557, r94, r1559;
}
{
add.f16x2 r1560, r1536, r1557;
}
{
cvt.rn.f16.f64 rs237, fd251;
}
mov.b32 r1565, {rs237, rs237};
{
mul.f16x2 r1563, r88, r1565;
}
{
add.f16x2 r1566, r1542, r1563;
}
{
cvt.rn.f16.f64 rs238, fd252;
}
mov.b32 r1571, {rs238, rs238};
{
mul.f16x2 r1569, r91, r1571;
}
{
add.f16x2 r1572, r1548, r1569;
}
{
sub.f16x2 %14, r1554, r1560;
}
{
add.f16x2 %15, r1566, r1572;
}
{
add.f16x2 %20, r1554, r1560;
}
{
sub.f16x2 %21, r1566, r1572;
}
cvt.rn.f16.s32 rs239, r1588;
mov.b32 r1599, {rs239, rs239};
cvt.rn.f16.s32 rs240, r1588;
mov.b32 r1611, {rs240, rs240};
{
cvt.rn.f16.f64 rs241, fd227;
}
mov.b32 r1591, {rs241, rs241};
{
mul.f16x2 r1589, r1, r1591;
}
{
add.f16x2 r1592, %34, r1589;
}
{
cvt.rn.f16.f64 rs242, fd228;
}
mov.b32 r1597, {rs242, rs242};
{
mul.f16x2 r1595, r10, r1597;
}
{
add.f16x2 r1598, r1599, r1595;
}
{
cvt.rn.f16.f64 rs243, fd227;
}
mov.b32 r1603, {rs243, rs243};
{
mul.f16x2 r1601, r4, r1603;
}
{
add.f16x2 r1604, %35, r1601;
}
{
cvt.rn.f16.f64 rs244, fd228;
}
mov.b32 r1609, {rs244, rs244};
{
mul.f16x2 r1607, r7, r1609;
}
{
add.f16x2 r1610, r1611, r1607;
}
{
cvt.rn.f16.f64 rs245, fd231;
}
mov.b32 r1615, {rs245, rs245};
{
mul.f16x2 r1613, r13, r1615;
}
{
add.f16x2 r1616, r1592, r1613;
}
{
cvt.rn.f16.f64 rs246, fd232;
}
mov.b32 r1621, {rs246, rs246};
{
mul.f16x2 r1619, r22, r1621;
}
{
add.f16x2 r1622, r1598, r1619;
}
{
cvt.rn.f16.f64 rs247, fd231;
}
mov.b32 r1627, {rs247, rs247};
{
mul.f16x2 r1625, r16, r1627;
}
{
add.f16x2 r1628, r1604, r1625;
}
{
cvt.rn.f16.f64 rs248, fd232;
}
mov.b32 r1633, {rs248, rs248};
{
mul.f16x2 r1631, r19, r1633;
}
{
add.f16x2 r1634, r1610, r1631;
}
{
cvt.rn.f16.f64 rs249, fd235;
}
mov.b32 r1639, {rs249, rs249};
{
mul.f16x2 r1637, r25, r1639;
}
{
add.f16x2 r1640, r1616, r1637;
}
{
cvt.rn.f16.f64 rs250, fd236;
}
mov.b32 r1645, {rs250, rs250};
{
mul.f16x2 r1643, r34, r1645;
}
{
add.f16x2 r1646, r1622, r1643;
}
{
cvt.rn.f16.f64 rs251, fd235;
}
mov.b32 r1651, {rs251, rs251};
{
mul.f16x2 r1649, r28, r1651;
}
{
add.f16x2 r1652, r1628, r1649;
}
{
cvt.rn.f16.f64 rs252, fd236;
}
mov.b32 r1657, {rs252, rs252};
{
mul.f16x2 r1655, r31, r1657;
}
{
add.f16x2 r1658, r1634, r1655;
}
{
cvt.rn.f16.f64 rs253, fd239;
}
mov.b32 r1663, {rs253, rs253};
{
mul.f16x2 r1661, r37, r1663;
}
{
add.f16x2 r1664, r1640, r1661;
}
{
cvt.rn.f16.f64 rs254, fd240;
}
mov.b32 r1669, {rs254, rs254};
{
mul.f16x2 r1667, r46, r1669;
}
{
add.f16x2 r1670, r1646, r1667;
}
{
cvt.rn.f16.f64 rs255, fd239;
}
mov.b32 r1675, {rs255, rs255};
{
mul.f16x2 r1673, r40, r1675;
}
{
add.f16x2 r1676, r1652, r1673;
}
{
cvt.rn.f16.f64 rs256, fd240;
}
mov.b32 r1681, {rs256, rs256};
{
mul.f16x2 r1679, r43, r1681;
}
{
add.f16x2 r1682, r1658, r1679;
}
{
cvt.rn.f16.f64 rs257, fd243;
}
mov.b32 r1687, {rs257, rs257};
{
mul.f16x2 r1685, r49, r1687;
}
{
add.f16x2 r1688, r1664, r1685;
}
{
cvt.rn.f16.f64 rs258, fd244;
}
mov.b32 r1693, {rs258, rs258};
{
mul.f16x2 r1691, r58, r1693;
}
{
add.f16x2 r1694, r1670, r1691;
}
{
cvt.rn.f16.f64 rs259, fd243;
}
mov.b32 r1699, {rs259, rs259};
{
mul.f16x2 r1697, r52, r1699;
}
{
add.f16x2 r1700, r1676, r1697;
}
{
cvt.rn.f16.f64 rs260, fd244;
}
mov.b32 r1705, {rs260, rs260};
{
mul.f16x2 r1703, r55, r1705;
}
{
add.f16x2 r1706, r1682, r1703;
}
{
cvt.rn.f16.f64 rs261, fd247;
}
mov.b32 r1711, {rs261, rs261};
{
mul.f16x2 r1709, r61, r1711;
}
{
add.f16x2 r1712, r1688, r1709;
}
{
cvt.rn.f16.f64 rs262, fd248;
}
mov.b32 r1717, {rs262, rs262};
{
mul.f16x2 r1715, r70, r1717;
}
{
add.f16x2 r1718, r1694, r1715;
}
{
cvt.rn.f16.f64 rs263, fd247;
}
mov.b32 r1723, {rs263, rs263};
{
mul.f16x2 r1721, r64, r1723;
}
{
add.f16x2 r1724, r1700, r1721;
}
{
cvt.rn.f16.f64 rs264, fd248;
}
mov.b32 r1729, {rs264, rs264};
{
mul.f16x2 r1727, r67, r1729;
}
{
add.f16x2 r1730, r1706, r1727;
}
{
cvt.rn.f16.f64 rs265, fd251;
}
mov.b32 r1735, {rs265, rs265};
{
mul.f16x2 r1733, r73, r1735;
}
{
add.f16x2 r1736, r1712, r1733;
}
{
cvt.rn.f16.f64 rs266, fd252;
}
mov.b32 r1741, {rs266, rs266};
{
mul.f16x2 r1739, r82, r1741;
}
{
add.f16x2 r1742, r1718, r1739;
}
{
cvt.rn.f16.f64 rs267, fd251;
}
mov.b32 r1747, {rs267, rs267};
{
mul.f16x2 r1745, r76, r1747;
}
{
add.f16x2 r1748, r1724, r1745;
}
{
cvt.rn.f16.f64 rs268, fd252;
}
mov.b32 r1753, {rs268, rs268};
{
mul.f16x2 r1751, r79, r1753;
}
{
add.f16x2 r1754, r1730, r1751;
}
{
cvt.rn.f16.f64 rs269, fd255;
}
mov.b32 r1759, {rs269, rs269};
{
mul.f16x2 r1757, r85, r1759;
}
{
add.f16x2 r1760, r1736, r1757;
}
{
cvt.rn.f16.f64 rs270, fd256;
}
mov.b32 r1765, {rs270, rs270};
{
mul.f16x2 r1763, r94, r1765;
}
{
add.f16x2 r1766, r1742, r1763;
}
{
cvt.rn.f16.f64 rs271, fd255;
}
mov.b32 r1771, {rs271, rs271};
{
mul.f16x2 r1769, r88, r1771;
}
{
add.f16x2 r1772, r1748, r1769;
}
{
cvt.rn.f16.f64 rs272, fd256;
}
mov.b32 r1777, {rs272, rs272};
{
mul.f16x2 r1775, r91, r1777;
}
{
add.f16x2 r1778, r1754, r1775;
}
{
sub.f16x2 %16, r1760, r1766;
}
{
add.f16x2 %17, r1772, r1778;
}
{
add.f16x2 %18, r1760, r1766;
}
{
sub.f16x2 %19, r1772, r1778;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[16].y)));
};


#endif
