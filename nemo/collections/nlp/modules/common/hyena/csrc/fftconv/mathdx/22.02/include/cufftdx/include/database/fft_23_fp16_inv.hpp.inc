#ifndef CUFFTDX_FFT_23_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_23_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<957, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<507>;
.reg .b32 r<3257>;
.reg .f64 fd<485>;
.reg .b64 rd<2>;
{
add.f16x2 r1, %48, %90;
}
{
add.f16x2 r4, %49, %91;
}
{
sub.f16x2 r7, %48, %90;
}
{
sub.f16x2 r10, %49, %91;
}
{
add.f16x2 r13, %50, %88;
}
{
add.f16x2 r16, %51, %89;
}
{
sub.f16x2 r19, %50, %88;
}
{
sub.f16x2 r22, %51, %89;
}
{
add.f16x2 r25, %52, %86;
}
{
add.f16x2 r28, %53, %87;
}
{
sub.f16x2 r31, %52, %86;
}
{
sub.f16x2 r34, %53, %87;
}
{
add.f16x2 r37, %54, %84;
}
{
add.f16x2 r40, %55, %85;
}
{
sub.f16x2 r43, %54, %84;
}
{
sub.f16x2 r46, %55, %85;
}
{
add.f16x2 r49, %56, %82;
}
{
add.f16x2 r52, %57, %83;
}
{
sub.f16x2 r55, %56, %82;
}
{
sub.f16x2 r58, %57, %83;
}
{
add.f16x2 r61, %58, %80;
}
{
add.f16x2 r64, %59, %81;
}
{
sub.f16x2 r67, %58, %80;
}
{
sub.f16x2 r70, %59, %81;
}
{
add.f16x2 r73, %60, %78;
}
{
add.f16x2 r76, %61, %79;
}
{
sub.f16x2 r79, %60, %78;
}
{
sub.f16x2 r82, %61, %79;
}
{
add.f16x2 r85, %62, %76;
}
{
add.f16x2 r88, %63, %77;
}
{
sub.f16x2 r91, %62, %76;
}
{
sub.f16x2 r94, %63, %77;
}
{
add.f16x2 r97, %64, %74;
}
{
add.f16x2 r100, %65, %75;
}
{
sub.f16x2 r103, %64, %74;
}
{
sub.f16x2 r106, %65, %75;
}
{
add.f16x2 r109, %66, %72;
}
{
add.f16x2 r112, %67, %73;
}
{
sub.f16x2 r115, %66, %72;
}
{
sub.f16x2 r118, %67, %73;
}
{
add.f16x2 r121, %68, %70;
}
{
add.f16x2 r124, %69, %71;
}
{
sub.f16x2 r127, %68, %70;
}
{
sub.f16x2 r130, %69, %71;
}
{
add.f16x2 r133, %46, r1;
}
{
add.f16x2 r136, %47, r4;
}
{
add.f16x2 r139, r133, r13;
}
{
add.f16x2 r142, r136, r16;
}
{
add.f16x2 r145, r139, r25;
}
{
add.f16x2 r148, r142, r28;
}
{
add.f16x2 r151, r145, r37;
}
{
add.f16x2 r154, r148, r40;
}
{
add.f16x2 r157, r151, r49;
}
{
add.f16x2 r160, r154, r52;
}
{
add.f16x2 r163, r157, r61;
}
{
add.f16x2 r166, r160, r64;
}
{
add.f16x2 r169, r163, r73;
}
{
add.f16x2 r172, r166, r76;
}
{
add.f16x2 r175, r169, r85;
}
{
add.f16x2 r178, r172, r88;
}
{
add.f16x2 r181, r175, r97;
}
{
add.f16x2 r184, r178, r100;
}
{
add.f16x2 r187, r181, r109;
}
{
add.f16x2 r190, r184, r112;
}
{
add.f16x2 %0, r187, r121;
}
{
add.f16x2 %1, r190, r124;
}
mov.u32 r2980, 0;
cvt.rn.f16.s32 rs1, r2980;
mov.b32 r211, {rs1, rs1};
cvt.rn.f16.s32 rs2, r2980;
mov.b32 r223, {rs2, rs2};
mov.f64 fd447, 0d3FEED037EA3D2DBB;
{
cvt.rn.f16.f64 rs3, fd447;
}
mov.b32 r203, {rs3, rs3};
{
mul.f16x2 r201, r1, r203;
}
{
add.f16x2 r204, %46, r201;
}
mov.f64 fd424, 0d3FD14459AD2BE466;
{
cvt.rn.f16.f64 rs4, fd424;
}
mov.b32 r209, {rs4, rs4};
{
mul.f16x2 r207, r10, r209;
}
{
add.f16x2 r210, r211, r207;
}
{
cvt.rn.f16.f64 rs5, fd447;
}
mov.b32 r215, {rs5, rs5};
{
mul.f16x2 r213, r4, r215;
}
{
add.f16x2 r216, %47, r213;
}
{
cvt.rn.f16.f64 rs6, fd424;
}
mov.b32 r221, {rs6, rs6};
{
mul.f16x2 r219, r7, r221;
}
{
add.f16x2 r222, r223, r219;
}
mov.f64 fd455, 0d3FEB57675CF309EE;
{
cvt.rn.f16.f64 rs7, fd455;
}
mov.b32 r227, {rs7, rs7};
{
mul.f16x2 r225, r13, r227;
}
{
add.f16x2 r228, r204, r225;
}
mov.f64 fd332, 0d3FE0A06E851DB7CA;
{
cvt.rn.f16.f64 rs8, fd332;
}
mov.b32 r233, {rs8, rs8};
{
mul.f16x2 r231, r22, r233;
}
{
add.f16x2 r234, r210, r231;
}
{
cvt.rn.f16.f64 rs9, fd455;
}
mov.b32 r239, {rs9, rs9};
{
mul.f16x2 r237, r16, r239;
}
{
add.f16x2 r240, r216, r237;
}
{
cvt.rn.f16.f64 rs10, fd332;
}
mov.b32 r245, {rs10, rs10};
{
mul.f16x2 r243, r19, r245;
}
{
add.f16x2 r246, r222, r243;
}
mov.f64 fd463, 0d3FE5D779B07CFEF7;
{
cvt.rn.f16.f64 rs11, fd463;
}
mov.b32 r251, {rs11, rs11};
{
mul.f16x2 r249, r25, r251;
}
{
add.f16x2 r252, r228, r249;
}
mov.f64 fd384, 0d3FE763021AAA15DA;
{
cvt.rn.f16.f64 rs12, fd384;
}
mov.b32 r257, {rs12, rs12};
{
mul.f16x2 r255, r34, r257;
}
{
add.f16x2 r258, r234, r255;
}
{
cvt.rn.f16.f64 rs13, fd463;
}
mov.b32 r263, {rs13, rs13};
{
mul.f16x2 r261, r28, r263;
}
{
add.f16x2 r264, r240, r261;
}
{
cvt.rn.f16.f64 rs14, fd384;
}
mov.b32 r269, {rs14, rs14};
{
mul.f16x2 r267, r31, r269;
}
{
add.f16x2 r270, r246, r267;
}
mov.f64 fd471, 0d3FDD71B4A0C5A6C8;
{
cvt.rn.f16.f64 rs15, fd471;
}
mov.b32 r275, {rs15, rs15};
{
mul.f16x2 r273, r37, r275;
}
{
add.f16x2 r276, r252, r273;
}
mov.f64 fd416, 0d3FEC698E42F47B09;
{
cvt.rn.f16.f64 rs16, fd416;
}
mov.b32 r281, {rs16, rs16};
{
mul.f16x2 r279, r46, r281;
}
{
add.f16x2 r282, r258, r279;
}
{
cvt.rn.f16.f64 rs17, fd471;
}
mov.b32 r287, {rs17, rs17};
{
mul.f16x2 r285, r40, r287;
}
{
add.f16x2 r288, r264, r285;
}
{
cvt.rn.f16.f64 rs18, fd416;
}
mov.b32 r293, {rs18, rs18};
{
mul.f16x2 r291, r43, r293;
}
{
add.f16x2 r294, r270, r291;
}
mov.f64 fd479, 0d3FCA0AD8BD1E2882;
{
cvt.rn.f16.f64 rs19, fd479;
}
mov.b32 r299, {rs19, rs19};
{
mul.f16x2 r297, r49, r299;
}
{
add.f16x2 r300, r276, r297;
}
mov.f64 fd280, 0d3FEF54A827142577;
{
cvt.rn.f16.f64 rs20, fd280;
}
mov.b32 r305, {rs20, rs20};
{
mul.f16x2 r303, r58, r305;
}
{
add.f16x2 r306, r282, r303;
}
{
cvt.rn.f16.f64 rs21, fd479;
}
mov.b32 r311, {rs21, rs21};
{
mul.f16x2 r309, r52, r311;
}
{
add.f16x2 r312, r288, r309;
}
{
cvt.rn.f16.f64 rs22, fd280;
}
mov.b32 r317, {rs22, rs22};
{
mul.f16x2 r315, r55, r317;
}
{
add.f16x2 r318, r294, r315;
}
mov.f64 fd483, 0dBFB17855B599F3B9;
{
cvt.rn.f16.f64 rs23, fd483;
}
mov.b32 r323, {rs23, rs23};
{
mul.f16x2 r321, r61, r323;
}
{
add.f16x2 r324, r300, r321;
}
mov.f64 fd484, 0d3FEFECE70DFD3EFB;
{
cvt.rn.f16.f64 rs24, fd484;
}
mov.b32 r329, {rs24, rs24};
{
mul.f16x2 r327, r70, r329;
}
{
add.f16x2 r330, r306, r327;
}
{
cvt.rn.f16.f64 rs25, fd483;
}
mov.b32 r335, {rs25, rs25};
{
mul.f16x2 r333, r64, r335;
}
{
add.f16x2 r336, r312, r333;
}
{
cvt.rn.f16.f64 rs26, fd484;
}
mov.b32 r341, {rs26, rs26};
{
mul.f16x2 r339, r67, r341;
}
{
add.f16x2 r342, r318, r339;
}
mov.f64 fd475, 0dBFD56EAAE597C776;
{
cvt.rn.f16.f64 rs27, fd475;
}
mov.b32 r347, {rs27, rs27};
{
mul.f16x2 r345, r73, r347;
}
{
add.f16x2 r348, r324, r345;
}
mov.f64 fd476, 0d3FEE270060999288;
{
cvt.rn.f16.f64 rs28, fd476;
}
mov.b32 r353, {rs28, rs28};
{
mul.f16x2 r351, r82, r353;
}
{
add.f16x2 r354, r330, r351;
}
{
cvt.rn.f16.f64 rs29, fd475;
}
mov.b32 r359, {rs29, rs29};
{
mul.f16x2 r357, r76, r359;
}
{
add.f16x2 r360, r336, r357;
}
{
cvt.rn.f16.f64 rs30, fd476;
}
mov.b32 r365, {rs30, rs30};
{
mul.f16x2 r363, r79, r365;
}
{
add.f16x2 r366, r342, r363;
}
mov.f64 fd467, 0dBFE2742A4A775CFB;
{
cvt.rn.f16.f64 rs31, fd467;
}
mov.b32 r371, {rs31, rs31};
{
mul.f16x2 r369, r85, r371;
}
{
add.f16x2 r372, r348, r369;
}
mov.f64 fd468, 0d3FEA249E0B897CA9;
{
cvt.rn.f16.f64 rs32, fd468;
}
mov.b32 r377, {rs32, rs32};
{
mul.f16x2 r375, r94, r377;
}
{
add.f16x2 r378, r354, r375;
}
{
cvt.rn.f16.f64 rs33, fd467;
}
mov.b32 r383, {rs33, rs33};
{
mul.f16x2 r381, r88, r383;
}
{
add.f16x2 r384, r360, r381;
}
{
cvt.rn.f16.f64 rs34, fd468;
}
mov.b32 r389, {rs34, rs34};
{
mul.f16x2 r387, r91, r389;
}
{
add.f16x2 r390, r366, r387;
}
mov.f64 fd459, 0dBFE8D2A07C16D46F;
{
cvt.rn.f16.f64 rs35, fd459;
}
mov.b32 r395, {rs35, rs35};
{
mul.f16x2 r393, r97, r395;
}
{
add.f16x2 r396, r372, r393;
}
mov.f64 fd460, 0d3FE431DF5838F7EF;
{
cvt.rn.f16.f64 rs36, fd460;
}
mov.b32 r401, {rs36, rs36};
{
mul.f16x2 r399, r106, r401;
}
{
add.f16x2 r402, r378, r399;
}
{
cvt.rn.f16.f64 rs37, fd459;
}
mov.b32 r407, {rs37, rs37};
{
mul.f16x2 r405, r100, r407;
}
{
add.f16x2 r408, r384, r405;
}
{
cvt.rn.f16.f64 rs38, fd460;
}
mov.b32 r413, {rs38, rs38};
{
mul.f16x2 r411, r103, r413;
}
{
add.f16x2 r414, r390, r411;
}
mov.f64 fd451, 0dBFED59CB83EF99BC;
{
cvt.rn.f16.f64 rs39, fd451;
}
mov.b32 r419, {rs39, rs39};
{
mul.f16x2 r417, r109, r419;
}
{
add.f16x2 r420, r396, r417;
}
mov.f64 fd452, 0d3FD97F6748E524B2;
{
cvt.rn.f16.f64 rs40, fd452;
}
mov.b32 r425, {rs40, rs40};
{
mul.f16x2 r423, r118, r425;
}
{
add.f16x2 r426, r402, r423;
}
{
cvt.rn.f16.f64 rs41, fd451;
}
mov.b32 r431, {rs41, rs41};
{
mul.f16x2 r429, r112, r431;
}
{
add.f16x2 r432, r408, r429;
}
{
cvt.rn.f16.f64 rs42, fd452;
}
mov.b32 r437, {rs42, rs42};
{
mul.f16x2 r435, r115, r437;
}
{
add.f16x2 r438, r414, r435;
}
mov.f64 fd443, 0dBFEFB3B3035AA6CD;
{
cvt.rn.f16.f64 rs43, fd443;
}
mov.b32 r443, {rs43, rs43};
{
mul.f16x2 r441, r121, r443;
}
{
add.f16x2 r444, r420, r441;
}
mov.f64 fd444, 0d3FC16DE8A4564F0A;
{
cvt.rn.f16.f64 rs44, fd444;
}
mov.b32 r449, {rs44, rs44};
{
mul.f16x2 r447, r130, r449;
}
{
add.f16x2 r450, r426, r447;
}
{
cvt.rn.f16.f64 rs45, fd443;
}
mov.b32 r455, {rs45, rs45};
{
mul.f16x2 r453, r124, r455;
}
{
add.f16x2 r456, r432, r453;
}
{
cvt.rn.f16.f64 rs46, fd444;
}
mov.b32 r461, {rs46, rs46};
{
mul.f16x2 r459, r127, r461;
}
{
add.f16x2 r462, r438, r459;
}
{
sub.f16x2 %2, r444, r450;
}
{
add.f16x2 %3, r456, r462;
}
{
add.f16x2 %44, r444, r450;
}
{
sub.f16x2 %45, r456, r462;
}
cvt.rn.f16.s32 rs47, r2980;
mov.b32 r489, {rs47, rs47};
cvt.rn.f16.s32 rs48, r2980;
mov.b32 r501, {rs48, rs48};
{
cvt.rn.f16.f64 rs49, fd455;
}
mov.b32 r481, {rs49, rs49};
{
mul.f16x2 r479, r1, r481;
}
{
add.f16x2 r482, %46, r479;
}
{
cvt.rn.f16.f64 rs50, fd332;
}
mov.b32 r487, {rs50, rs50};
{
mul.f16x2 r485, r10, r487;
}
{
add.f16x2 r488, r489, r485;
}
{
cvt.rn.f16.f64 rs51, fd455;
}
mov.b32 r493, {rs51, rs51};
{
mul.f16x2 r491, r4, r493;
}
{
add.f16x2 r494, %47, r491;
}
{
cvt.rn.f16.f64 rs52, fd332;
}
mov.b32 r499, {rs52, rs52};
{
mul.f16x2 r497, r7, r499;
}
{
add.f16x2 r500, r501, r497;
}
{
cvt.rn.f16.f64 rs53, fd471;
}
mov.b32 r505, {rs53, rs53};
{
mul.f16x2 r503, r13, r505;
}
{
add.f16x2 r506, r482, r503;
}
{
cvt.rn.f16.f64 rs54, fd416;
}
mov.b32 r511, {rs54, rs54};
{
mul.f16x2 r509, r22, r511;
}
{
add.f16x2 r512, r488, r509;
}
{
cvt.rn.f16.f64 rs55, fd471;
}
mov.b32 r517, {rs55, rs55};
{
mul.f16x2 r515, r16, r517;
}
{
add.f16x2 r518, r494, r515;
}
{
cvt.rn.f16.f64 rs56, fd416;
}
mov.b32 r523, {rs56, rs56};
{
mul.f16x2 r521, r19, r523;
}
{
add.f16x2 r524, r500, r521;
}
{
cvt.rn.f16.f64 rs57, fd483;
}
mov.b32 r529, {rs57, rs57};
{
mul.f16x2 r527, r25, r529;
}
{
add.f16x2 r530, r506, r527;
}
{
cvt.rn.f16.f64 rs58, fd484;
}
mov.b32 r535, {rs58, rs58};
{
mul.f16x2 r533, r34, r535;
}
{
add.f16x2 r536, r512, r533;
}
{
cvt.rn.f16.f64 rs59, fd483;
}
mov.b32 r541, {rs59, rs59};
{
mul.f16x2 r539, r28, r541;
}
{
add.f16x2 r542, r518, r539;
}
{
cvt.rn.f16.f64 rs60, fd484;
}
mov.b32 r547, {rs60, rs60};
{
mul.f16x2 r545, r31, r547;
}
{
add.f16x2 r548, r524, r545;
}
{
cvt.rn.f16.f64 rs61, fd467;
}
mov.b32 r553, {rs61, rs61};
{
mul.f16x2 r551, r37, r553;
}
{
add.f16x2 r554, r530, r551;
}
{
cvt.rn.f16.f64 rs62, fd468;
}
mov.b32 r559, {rs62, rs62};
{
mul.f16x2 r557, r46, r559;
}
{
add.f16x2 r560, r536, r557;
}
{
cvt.rn.f16.f64 rs63, fd467;
}
mov.b32 r565, {rs63, rs63};
{
mul.f16x2 r563, r40, r565;
}
{
add.f16x2 r566, r542, r563;
}
{
cvt.rn.f16.f64 rs64, fd468;
}
mov.b32 r571, {rs64, rs64};
{
mul.f16x2 r569, r43, r571;
}
{
add.f16x2 r572, r548, r569;
}
{
cvt.rn.f16.f64 rs65, fd451;
}
mov.b32 r577, {rs65, rs65};
{
mul.f16x2 r575, r49, r577;
}
{
add.f16x2 r578, r554, r575;
}
{
cvt.rn.f16.f64 rs66, fd452;
}
mov.b32 r583, {rs66, rs66};
{
mul.f16x2 r581, r58, r583;
}
{
add.f16x2 r584, r560, r581;
}
{
cvt.rn.f16.f64 rs67, fd451;
}
mov.b32 r589, {rs67, rs67};
{
mul.f16x2 r587, r52, r589;
}
{
add.f16x2 r590, r566, r587;
}
{
cvt.rn.f16.f64 rs68, fd452;
}
mov.b32 r595, {rs68, rs68};
{
mul.f16x2 r593, r55, r595;
}
{
add.f16x2 r596, r572, r593;
}
{
cvt.rn.f16.f64 rs69, fd443;
}
mov.b32 r601, {rs69, rs69};
{
mul.f16x2 r599, r61, r601;
}
{
add.f16x2 r602, r578, r599;
}
mov.f64 fd388, 0dBFC16DE8A4564F0A;
{
cvt.rn.f16.f64 rs70, fd388;
}
mov.b32 r607, {rs70, rs70};
{
mul.f16x2 r605, r70, r607;
}
{
add.f16x2 r608, r584, r605;
}
{
cvt.rn.f16.f64 rs71, fd443;
}
mov.b32 r613, {rs71, rs71};
{
mul.f16x2 r611, r64, r613;
}
{
add.f16x2 r614, r590, r611;
}
{
cvt.rn.f16.f64 rs72, fd388;
}
mov.b32 r619, {rs72, rs72};
{
mul.f16x2 r617, r67, r619;
}
{
add.f16x2 r620, r596, r617;
}
{
cvt.rn.f16.f64 rs73, fd459;
}
mov.b32 r625, {rs73, rs73};
{
mul.f16x2 r623, r73, r625;
}
{
add.f16x2 r626, r602, r623;
}
mov.f64 fd420, 0dBFE431DF5838F7EF;
{
cvt.rn.f16.f64 rs74, fd420;
}
mov.b32 r631, {rs74, rs74};
{
mul.f16x2 r629, r82, r631;
}
{
add.f16x2 r632, r608, r629;
}
{
cvt.rn.f16.f64 rs75, fd459;
}
mov.b32 r637, {rs75, rs75};
{
mul.f16x2 r635, r76, r637;
}
{
add.f16x2 r638, r614, r635;
}
{
cvt.rn.f16.f64 rs76, fd420;
}
mov.b32 r643, {rs76, rs76};
{
mul.f16x2 r641, r79, r643;
}
{
add.f16x2 r644, r620, r641;
}
{
cvt.rn.f16.f64 rs77, fd475;
}
mov.b32 r649, {rs77, rs77};
{
mul.f16x2 r647, r85, r649;
}
{
add.f16x2 r650, r626, r647;
}
mov.f64 fd316, 0dBFEE270060999288;
{
cvt.rn.f16.f64 rs78, fd316;
}
mov.b32 r655, {rs78, rs78};
{
mul.f16x2 r653, r94, r655;
}
{
add.f16x2 r656, r632, r653;
}
{
cvt.rn.f16.f64 rs79, fd475;
}
mov.b32 r661, {rs79, rs79};
{
mul.f16x2 r659, r88, r661;
}
{
add.f16x2 r662, r638, r659;
}
{
cvt.rn.f16.f64 rs80, fd316;
}
mov.b32 r667, {rs80, rs80};
{
mul.f16x2 r665, r91, r667;
}
{
add.f16x2 r668, r644, r665;
}
{
cvt.rn.f16.f64 rs81, fd479;
}
mov.b32 r673, {rs81, rs81};
{
mul.f16x2 r671, r97, r673;
}
{
add.f16x2 r674, r650, r671;
}
mov.f64 fd480, 0dBFEF54A827142577;
{
cvt.rn.f16.f64 rs82, fd480;
}
mov.b32 r679, {rs82, rs82};
{
mul.f16x2 r677, r106, r679;
}
{
add.f16x2 r680, r656, r677;
}
{
cvt.rn.f16.f64 rs83, fd479;
}
mov.b32 r685, {rs83, rs83};
{
mul.f16x2 r683, r100, r685;
}
{
add.f16x2 r686, r662, r683;
}
{
cvt.rn.f16.f64 rs84, fd480;
}
mov.b32 r691, {rs84, rs84};
{
mul.f16x2 r689, r103, r691;
}
{
add.f16x2 r692, r668, r689;
}
{
cvt.rn.f16.f64 rs85, fd463;
}
mov.b32 r697, {rs85, rs85};
{
mul.f16x2 r695, r109, r697;
}
{
add.f16x2 r698, r674, r695;
}
mov.f64 fd464, 0dBFE763021AAA15DA;
{
cvt.rn.f16.f64 rs86, fd464;
}
mov.b32 r703, {rs86, rs86};
{
mul.f16x2 r701, r118, r703;
}
{
add.f16x2 r704, r680, r701;
}
{
cvt.rn.f16.f64 rs87, fd463;
}
mov.b32 r709, {rs87, rs87};
{
mul.f16x2 r707, r112, r709;
}
{
add.f16x2 r710, r686, r707;
}
{
cvt.rn.f16.f64 rs88, fd464;
}
mov.b32 r715, {rs88, rs88};
{
mul.f16x2 r713, r115, r715;
}
{
add.f16x2 r716, r692, r713;
}
{
cvt.rn.f16.f64 rs89, fd447;
}
mov.b32 r721, {rs89, rs89};
{
mul.f16x2 r719, r121, r721;
}
{
add.f16x2 r722, r698, r719;
}
mov.f64 fd448, 0dBFD14459AD2BE466;
{
cvt.rn.f16.f64 rs90, fd448;
}
mov.b32 r727, {rs90, rs90};
{
mul.f16x2 r725, r130, r727;
}
{
add.f16x2 r728, r704, r725;
}
{
cvt.rn.f16.f64 rs91, fd447;
}
mov.b32 r733, {rs91, rs91};
{
mul.f16x2 r731, r124, r733;
}
{
add.f16x2 r734, r710, r731;
}
{
cvt.rn.f16.f64 rs92, fd448;
}
mov.b32 r739, {rs92, rs92};
{
mul.f16x2 r737, r127, r739;
}
{
add.f16x2 r740, r716, r737;
}
{
sub.f16x2 %4, r722, r728;
}
{
add.f16x2 %5, r734, r740;
}
{
add.f16x2 %42, r722, r728;
}
{
sub.f16x2 %43, r734, r740;
}
cvt.rn.f16.s32 rs93, r2980;
mov.b32 r767, {rs93, rs93};
cvt.rn.f16.s32 rs94, r2980;
mov.b32 r779, {rs94, rs94};
{
cvt.rn.f16.f64 rs95, fd463;
}
mov.b32 r759, {rs95, rs95};
{
mul.f16x2 r757, r1, r759;
}
{
add.f16x2 r760, %46, r757;
}
{
cvt.rn.f16.f64 rs96, fd384;
}
mov.b32 r765, {rs96, rs96};
{
mul.f16x2 r763, r10, r765;
}
{
add.f16x2 r766, r767, r763;
}
{
cvt.rn.f16.f64 rs97, fd463;
}
mov.b32 r771, {rs97, rs97};
{
mul.f16x2 r769, r4, r771;
}
{
add.f16x2 r772, %47, r769;
}
{
cvt.rn.f16.f64 rs98, fd384;
}
mov.b32 r777, {rs98, rs98};
{
mul.f16x2 r775, r7, r777;
}
{
add.f16x2 r778, r779, r775;
}
{
cvt.rn.f16.f64 rs99, fd483;
}
mov.b32 r783, {rs99, rs99};
{
mul.f16x2 r781, r13, r783;
}
{
add.f16x2 r784, r760, r781;
}
{
cvt.rn.f16.f64 rs100, fd484;
}
mov.b32 r789, {rs100, rs100};
{
mul.f16x2 r787, r22, r789;
}
{
add.f16x2 r790, r766, r787;
}
{
cvt.rn.f16.f64 rs101, fd483;
}
mov.b32 r795, {rs101, rs101};
{
mul.f16x2 r793, r16, r795;
}
{
add.f16x2 r796, r772, r793;
}
{
cvt.rn.f16.f64 rs102, fd484;
}
mov.b32 r801, {rs102, rs102};
{
mul.f16x2 r799, r19, r801;
}
{
add.f16x2 r802, r778, r799;
}
{
cvt.rn.f16.f64 rs103, fd459;
}
mov.b32 r807, {rs103, rs103};
{
mul.f16x2 r805, r25, r807;
}
{
add.f16x2 r808, r784, r805;
}
{
cvt.rn.f16.f64 rs104, fd460;
}
mov.b32 r813, {rs104, rs104};
{
mul.f16x2 r811, r34, r813;
}
{
add.f16x2 r814, r790, r811;
}
{
cvt.rn.f16.f64 rs105, fd459;
}
mov.b32 r819, {rs105, rs105};
{
mul.f16x2 r817, r28, r819;
}
{
add.f16x2 r820, r796, r817;
}
{
cvt.rn.f16.f64 rs106, fd460;
}
mov.b32 r825, {rs106, rs106};
{
mul.f16x2 r823, r31, r825;
}
{
add.f16x2 r826, r802, r823;
}
{
cvt.rn.f16.f64 rs107, fd443;
}
mov.b32 r831, {rs107, rs107};
{
mul.f16x2 r829, r37, r831;
}
{
add.f16x2 r832, r808, r829;
}
{
cvt.rn.f16.f64 rs108, fd388;
}
mov.b32 r837, {rs108, rs108};
{
mul.f16x2 r835, r46, r837;
}
{
add.f16x2 r838, r814, r835;
}
{
cvt.rn.f16.f64 rs109, fd443;
}
mov.b32 r843, {rs109, rs109};
{
mul.f16x2 r841, r40, r843;
}
{
add.f16x2 r844, r820, r841;
}
{
cvt.rn.f16.f64 rs110, fd388;
}
mov.b32 r849, {rs110, rs110};
{
mul.f16x2 r847, r43, r849;
}
{
add.f16x2 r850, r826, r847;
}
{
cvt.rn.f16.f64 rs111, fd467;
}
mov.b32 r855, {rs111, rs111};
{
mul.f16x2 r853, r49, r855;
}
{
add.f16x2 r856, r832, r853;
}
mov.f64 fd188, 0dBFEA249E0B897CA9;
{
cvt.rn.f16.f64 rs112, fd188;
}
mov.b32 r861, {rs112, rs112};
{
mul.f16x2 r859, r58, r861;
}
{
add.f16x2 r862, r838, r859;
}
{
cvt.rn.f16.f64 rs113, fd467;
}
mov.b32 r867, {rs113, rs113};
{
mul.f16x2 r865, r52, r867;
}
{
add.f16x2 r868, r844, r865;
}
{
cvt.rn.f16.f64 rs114, fd188;
}
mov.b32 r873, {rs114, rs114};
{
mul.f16x2 r871, r55, r873;
}
{
add.f16x2 r874, r850, r871;
}
{
cvt.rn.f16.f64 rs115, fd479;
}
mov.b32 r879, {rs115, rs115};
{
mul.f16x2 r877, r61, r879;
}
{
add.f16x2 r880, r856, r877;
}
{
cvt.rn.f16.f64 rs116, fd480;
}
mov.b32 r885, {rs116, rs116};
{
mul.f16x2 r883, r70, r885;
}
{
add.f16x2 r886, r862, r883;
}
{
cvt.rn.f16.f64 rs117, fd479;
}
mov.b32 r891, {rs117, rs117};
{
mul.f16x2 r889, r64, r891;
}
{
add.f16x2 r892, r868, r889;
}
{
cvt.rn.f16.f64 rs118, fd480;
}
mov.b32 r897, {rs118, rs118};
{
mul.f16x2 r895, r67, r897;
}
{
add.f16x2 r898, r874, r895;
}
{
cvt.rn.f16.f64 rs119, fd455;
}
mov.b32 r903, {rs119, rs119};
{
mul.f16x2 r901, r73, r903;
}
{
add.f16x2 r904, r880, r901;
}
mov.f64 fd456, 0dBFE0A06E851DB7CA;
{
cvt.rn.f16.f64 rs120, fd456;
}
mov.b32 r909, {rs120, rs120};
{
mul.f16x2 r907, r82, r909;
}
{
add.f16x2 r910, r886, r907;
}
{
cvt.rn.f16.f64 rs121, fd455;
}
mov.b32 r915, {rs121, rs121};
{
mul.f16x2 r913, r76, r915;
}
{
add.f16x2 r916, r892, r913;
}
{
cvt.rn.f16.f64 rs122, fd456;
}
mov.b32 r921, {rs122, rs122};
{
mul.f16x2 r919, r79, r921;
}
{
add.f16x2 r922, r898, r919;
}
{
cvt.rn.f16.f64 rs123, fd447;
}
mov.b32 r927, {rs123, rs123};
{
mul.f16x2 r925, r85, r927;
}
{
add.f16x2 r928, r904, r925;
}
{
cvt.rn.f16.f64 rs124, fd424;
}
mov.b32 r933, {rs124, rs124};
{
mul.f16x2 r931, r94, r933;
}
{
add.f16x2 r934, r910, r931;
}
{
cvt.rn.f16.f64 rs125, fd447;
}
mov.b32 r939, {rs125, rs125};
{
mul.f16x2 r937, r88, r939;
}
{
add.f16x2 r940, r916, r937;
}
{
cvt.rn.f16.f64 rs126, fd424;
}
mov.b32 r945, {rs126, rs126};
{
mul.f16x2 r943, r91, r945;
}
{
add.f16x2 r946, r922, r943;
}
{
cvt.rn.f16.f64 rs127, fd471;
}
mov.b32 r951, {rs127, rs127};
{
mul.f16x2 r949, r97, r951;
}
{
add.f16x2 r952, r928, r949;
}
{
cvt.rn.f16.f64 rs128, fd416;
}
mov.b32 r957, {rs128, rs128};
{
mul.f16x2 r955, r106, r957;
}
{
add.f16x2 r958, r934, r955;
}
{
cvt.rn.f16.f64 rs129, fd471;
}
mov.b32 r963, {rs129, rs129};
{
mul.f16x2 r961, r100, r963;
}
{
add.f16x2 r964, r940, r961;
}
{
cvt.rn.f16.f64 rs130, fd416;
}
mov.b32 r969, {rs130, rs130};
{
mul.f16x2 r967, r103, r969;
}
{
add.f16x2 r970, r946, r967;
}
{
cvt.rn.f16.f64 rs131, fd475;
}
mov.b32 r975, {rs131, rs131};
{
mul.f16x2 r973, r109, r975;
}
{
add.f16x2 r976, r952, r973;
}
{
cvt.rn.f16.f64 rs132, fd476;
}
mov.b32 r981, {rs132, rs132};
{
mul.f16x2 r979, r118, r981;
}
{
add.f16x2 r982, r958, r979;
}
{
cvt.rn.f16.f64 rs133, fd475;
}
mov.b32 r987, {rs133, rs133};
{
mul.f16x2 r985, r112, r987;
}
{
add.f16x2 r988, r964, r985;
}
{
cvt.rn.f16.f64 rs134, fd476;
}
mov.b32 r993, {rs134, rs134};
{
mul.f16x2 r991, r115, r993;
}
{
add.f16x2 r994, r970, r991;
}
{
cvt.rn.f16.f64 rs135, fd451;
}
mov.b32 r999, {rs135, rs135};
{
mul.f16x2 r997, r121, r999;
}
{
add.f16x2 r1000, r976, r997;
}
{
cvt.rn.f16.f64 rs136, fd452;
}
mov.b32 r1005, {rs136, rs136};
{
mul.f16x2 r1003, r130, r1005;
}
{
add.f16x2 r1006, r982, r1003;
}
{
cvt.rn.f16.f64 rs137, fd451;
}
mov.b32 r1011, {rs137, rs137};
{
mul.f16x2 r1009, r124, r1011;
}
{
add.f16x2 r1012, r988, r1009;
}
{
cvt.rn.f16.f64 rs138, fd452;
}
mov.b32 r1017, {rs138, rs138};
{
mul.f16x2 r1015, r127, r1017;
}
{
add.f16x2 r1018, r994, r1015;
}
{
sub.f16x2 %6, r1000, r1006;
}
{
add.f16x2 %7, r1012, r1018;
}
{
add.f16x2 %40, r1000, r1006;
}
{
sub.f16x2 %41, r1012, r1018;
}
cvt.rn.f16.s32 rs139, r2980;
mov.b32 r1045, {rs139, rs139};
cvt.rn.f16.s32 rs140, r2980;
mov.b32 r1057, {rs140, rs140};
{
cvt.rn.f16.f64 rs141, fd471;
}
mov.b32 r1037, {rs141, rs141};
{
mul.f16x2 r1035, r1, r1037;
}
{
add.f16x2 r1038, %46, r1035;
}
{
cvt.rn.f16.f64 rs142, fd416;
}
mov.b32 r1043, {rs142, rs142};
{
mul.f16x2 r1041, r10, r1043;
}
{
add.f16x2 r1044, r1045, r1041;
}
{
cvt.rn.f16.f64 rs143, fd471;
}
mov.b32 r1049, {rs143, rs143};
{
mul.f16x2 r1047, r4, r1049;
}
{
add.f16x2 r1050, %47, r1047;
}
{
cvt.rn.f16.f64 rs144, fd416;
}
mov.b32 r1055, {rs144, rs144};
{
mul.f16x2 r1053, r7, r1055;
}
{
add.f16x2 r1056, r1057, r1053;
}
{
cvt.rn.f16.f64 rs145, fd467;
}
mov.b32 r1061, {rs145, rs145};
{
mul.f16x2 r1059, r13, r1061;
}
{
add.f16x2 r1062, r1038, r1059;
}
{
cvt.rn.f16.f64 rs146, fd468;
}
mov.b32 r1067, {rs146, rs146};
{
mul.f16x2 r1065, r22, r1067;
}
{
add.f16x2 r1068, r1044, r1065;
}
{
cvt.rn.f16.f64 rs147, fd467;
}
mov.b32 r1073, {rs147, rs147};
{
mul.f16x2 r1071, r16, r1073;
}
{
add.f16x2 r1074, r1050, r1071;
}
{
cvt.rn.f16.f64 rs148, fd468;
}
mov.b32 r1079, {rs148, rs148};
{
mul.f16x2 r1077, r19, r1079;
}
{
add.f16x2 r1080, r1056, r1077;
}
{
cvt.rn.f16.f64 rs149, fd443;
}
mov.b32 r1085, {rs149, rs149};
{
mul.f16x2 r1083, r25, r1085;
}
{
add.f16x2 r1086, r1062, r1083;
}
{
cvt.rn.f16.f64 rs150, fd388;
}
mov.b32 r1091, {rs150, rs150};
{
mul.f16x2 r1089, r34, r1091;
}
{
add.f16x2 r1092, r1068, r1089;
}
{
cvt.rn.f16.f64 rs151, fd443;
}
mov.b32 r1097, {rs151, rs151};
{
mul.f16x2 r1095, r28, r1097;
}
{
add.f16x2 r1098, r1074, r1095;
}
{
cvt.rn.f16.f64 rs152, fd388;
}
mov.b32 r1103, {rs152, rs152};
{
mul.f16x2 r1101, r31, r1103;
}
{
add.f16x2 r1104, r1080, r1101;
}
{
cvt.rn.f16.f64 rs153, fd475;
}
mov.b32 r1109, {rs153, rs153};
{
mul.f16x2 r1107, r37, r1109;
}
{
add.f16x2 r1110, r1086, r1107;
}
{
cvt.rn.f16.f64 rs154, fd316;
}
mov.b32 r1115, {rs154, rs154};
{
mul.f16x2 r1113, r46, r1115;
}
{
add.f16x2 r1116, r1092, r1113;
}
{
cvt.rn.f16.f64 rs155, fd475;
}
mov.b32 r1121, {rs155, rs155};
{
mul.f16x2 r1119, r40, r1121;
}
{
add.f16x2 r1122, r1098, r1119;
}
{
cvt.rn.f16.f64 rs156, fd316;
}
mov.b32 r1127, {rs156, rs156};
{
mul.f16x2 r1125, r43, r1127;
}
{
add.f16x2 r1128, r1104, r1125;
}
{
cvt.rn.f16.f64 rs157, fd463;
}
mov.b32 r1133, {rs157, rs157};
{
mul.f16x2 r1131, r49, r1133;
}
{
add.f16x2 r1134, r1110, r1131;
}
{
cvt.rn.f16.f64 rs158, fd464;
}
mov.b32 r1139, {rs158, rs158};
{
mul.f16x2 r1137, r58, r1139;
}
{
add.f16x2 r1140, r1116, r1137;
}
{
cvt.rn.f16.f64 rs159, fd463;
}
mov.b32 r1145, {rs159, rs159};
{
mul.f16x2 r1143, r52, r1145;
}
{
add.f16x2 r1146, r1122, r1143;
}
{
cvt.rn.f16.f64 rs160, fd464;
}
mov.b32 r1151, {rs160, rs160};
{
mul.f16x2 r1149, r55, r1151;
}
{
add.f16x2 r1152, r1128, r1149;
}
{
cvt.rn.f16.f64 rs161, fd447;
}
mov.b32 r1157, {rs161, rs161};
{
mul.f16x2 r1155, r61, r1157;
}
{
add.f16x2 r1158, r1134, r1155;
}
{
cvt.rn.f16.f64 rs162, fd424;
}
mov.b32 r1163, {rs162, rs162};
{
mul.f16x2 r1161, r70, r1163;
}
{
add.f16x2 r1164, r1140, r1161;
}
{
cvt.rn.f16.f64 rs163, fd447;
}
mov.b32 r1169, {rs163, rs163};
{
mul.f16x2 r1167, r64, r1169;
}
{
add.f16x2 r1170, r1146, r1167;
}
{
cvt.rn.f16.f64 rs164, fd424;
}
mov.b32 r1175, {rs164, rs164};
{
mul.f16x2 r1173, r67, r1175;
}
{
add.f16x2 r1176, r1152, r1173;
}
{
cvt.rn.f16.f64 rs165, fd479;
}
mov.b32 r1181, {rs165, rs165};
{
mul.f16x2 r1179, r73, r1181;
}
{
add.f16x2 r1182, r1158, r1179;
}
{
cvt.rn.f16.f64 rs166, fd280;
}
mov.b32 r1187, {rs166, rs166};
{
mul.f16x2 r1185, r82, r1187;
}
{
add.f16x2 r1188, r1164, r1185;
}
{
cvt.rn.f16.f64 rs167, fd479;
}
mov.b32 r1193, {rs167, rs167};
{
mul.f16x2 r1191, r76, r1193;
}
{
add.f16x2 r1194, r1170, r1191;
}
{
cvt.rn.f16.f64 rs168, fd280;
}
mov.b32 r1199, {rs168, rs168};
{
mul.f16x2 r1197, r79, r1199;
}
{
add.f16x2 r1200, r1176, r1197;
}
{
cvt.rn.f16.f64 rs169, fd459;
}
mov.b32 r1205, {rs169, rs169};
{
mul.f16x2 r1203, r85, r1205;
}
{
add.f16x2 r1206, r1182, r1203;
}
{
cvt.rn.f16.f64 rs170, fd460;
}
mov.b32 r1211, {rs170, rs170};
{
mul.f16x2 r1209, r94, r1211;
}
{
add.f16x2 r1212, r1188, r1209;
}
{
cvt.rn.f16.f64 rs171, fd459;
}
mov.b32 r1217, {rs171, rs171};
{
mul.f16x2 r1215, r88, r1217;
}
{
add.f16x2 r1218, r1194, r1215;
}
{
cvt.rn.f16.f64 rs172, fd460;
}
mov.b32 r1223, {rs172, rs172};
{
mul.f16x2 r1221, r91, r1223;
}
{
add.f16x2 r1224, r1200, r1221;
}
{
cvt.rn.f16.f64 rs173, fd451;
}
mov.b32 r1229, {rs173, rs173};
{
mul.f16x2 r1227, r97, r1229;
}
{
add.f16x2 r1230, r1206, r1227;
}
mov.f64 fd368, 0dBFD97F6748E524B2;
{
cvt.rn.f16.f64 rs174, fd368;
}
mov.b32 r1235, {rs174, rs174};
{
mul.f16x2 r1233, r106, r1235;
}
{
add.f16x2 r1236, r1212, r1233;
}
{
cvt.rn.f16.f64 rs175, fd451;
}
mov.b32 r1241, {rs175, rs175};
{
mul.f16x2 r1239, r100, r1241;
}
{
add.f16x2 r1242, r1218, r1239;
}
{
cvt.rn.f16.f64 rs176, fd368;
}
mov.b32 r1247, {rs176, rs176};
{
mul.f16x2 r1245, r103, r1247;
}
{
add.f16x2 r1248, r1224, r1245;
}
{
cvt.rn.f16.f64 rs177, fd483;
}
mov.b32 r1253, {rs177, rs177};
{
mul.f16x2 r1251, r109, r1253;
}
{
add.f16x2 r1254, r1230, r1251;
}
mov.f64 fd412, 0dBFEFECE70DFD3EFB;
{
cvt.rn.f16.f64 rs178, fd412;
}
mov.b32 r1259, {rs178, rs178};
{
mul.f16x2 r1257, r118, r1259;
}
{
add.f16x2 r1260, r1236, r1257;
}
{
cvt.rn.f16.f64 rs179, fd483;
}
mov.b32 r1265, {rs179, rs179};
{
mul.f16x2 r1263, r112, r1265;
}
{
add.f16x2 r1266, r1242, r1263;
}
{
cvt.rn.f16.f64 rs180, fd412;
}
mov.b32 r1271, {rs180, rs180};
{
mul.f16x2 r1269, r115, r1271;
}
{
add.f16x2 r1272, r1248, r1269;
}
{
cvt.rn.f16.f64 rs181, fd455;
}
mov.b32 r1277, {rs181, rs181};
{
mul.f16x2 r1275, r121, r1277;
}
{
add.f16x2 r1278, r1254, r1275;
}
{
cvt.rn.f16.f64 rs182, fd456;
}
mov.b32 r1283, {rs182, rs182};
{
mul.f16x2 r1281, r130, r1283;
}
{
add.f16x2 r1284, r1260, r1281;
}
{
cvt.rn.f16.f64 rs183, fd455;
}
mov.b32 r1289, {rs183, rs183};
{
mul.f16x2 r1287, r124, r1289;
}
{
add.f16x2 r1290, r1266, r1287;
}
{
cvt.rn.f16.f64 rs184, fd456;
}
mov.b32 r1295, {rs184, rs184};
{
mul.f16x2 r1293, r127, r1295;
}
{
add.f16x2 r1296, r1272, r1293;
}
{
sub.f16x2 %8, r1278, r1284;
}
{
add.f16x2 %9, r1290, r1296;
}
{
add.f16x2 %38, r1278, r1284;
}
{
sub.f16x2 %39, r1290, r1296;
}
cvt.rn.f16.s32 rs185, r2980;
mov.b32 r1323, {rs185, rs185};
cvt.rn.f16.s32 rs186, r2980;
mov.b32 r1335, {rs186, rs186};
{
cvt.rn.f16.f64 rs187, fd479;
}
mov.b32 r1315, {rs187, rs187};
{
mul.f16x2 r1313, r1, r1315;
}
{
add.f16x2 r1316, %46, r1313;
}
{
cvt.rn.f16.f64 rs188, fd280;
}
mov.b32 r1321, {rs188, rs188};
{
mul.f16x2 r1319, r10, r1321;
}
{
add.f16x2 r1322, r1323, r1319;
}
{
cvt.rn.f16.f64 rs189, fd479;
}
mov.b32 r1327, {rs189, rs189};
{
mul.f16x2 r1325, r4, r1327;
}
{
add.f16x2 r1328, %47, r1325;
}
{
cvt.rn.f16.f64 rs190, fd280;
}
mov.b32 r1333, {rs190, rs190};
{
mul.f16x2 r1331, r7, r1333;
}
{
add.f16x2 r1334, r1335, r1331;
}
{
cvt.rn.f16.f64 rs191, fd451;
}
mov.b32 r1339, {rs191, rs191};
{
mul.f16x2 r1337, r13, r1339;
}
{
add.f16x2 r1340, r1316, r1337;
}
{
cvt.rn.f16.f64 rs192, fd452;
}
mov.b32 r1345, {rs192, rs192};
{
mul.f16x2 r1343, r22, r1345;
}
{
add.f16x2 r1346, r1322, r1343;
}
{
cvt.rn.f16.f64 rs193, fd451;
}
mov.b32 r1351, {rs193, rs193};
{
mul.f16x2 r1349, r16, r1351;
}
{
add.f16x2 r1352, r1328, r1349;
}
{
cvt.rn.f16.f64 rs194, fd452;
}
mov.b32 r1357, {rs194, rs194};
{
mul.f16x2 r1355, r19, r1357;
}
{
add.f16x2 r1358, r1334, r1355;
}
{
cvt.rn.f16.f64 rs195, fd467;
}
mov.b32 r1363, {rs195, rs195};
{
mul.f16x2 r1361, r25, r1363;
}
{
add.f16x2 r1364, r1340, r1361;
}
{
cvt.rn.f16.f64 rs196, fd188;
}
mov.b32 r1369, {rs196, rs196};
{
mul.f16x2 r1367, r34, r1369;
}
{
add.f16x2 r1370, r1346, r1367;
}
{
cvt.rn.f16.f64 rs197, fd467;
}
mov.b32 r1375, {rs197, rs197};
{
mul.f16x2 r1373, r28, r1375;
}
{
add.f16x2 r1376, r1352, r1373;
}
{
cvt.rn.f16.f64 rs198, fd188;
}
mov.b32 r1381, {rs198, rs198};
{
mul.f16x2 r1379, r31, r1381;
}
{
add.f16x2 r1382, r1358, r1379;
}
{
cvt.rn.f16.f64 rs199, fd463;
}
mov.b32 r1387, {rs199, rs199};
{
mul.f16x2 r1385, r37, r1387;
}
{
add.f16x2 r1388, r1364, r1385;
}
{
cvt.rn.f16.f64 rs200, fd464;
}
mov.b32 r1393, {rs200, rs200};
{
mul.f16x2 r1391, r46, r1393;
}
{
add.f16x2 r1394, r1370, r1391;
}
{
cvt.rn.f16.f64 rs201, fd463;
}
mov.b32 r1399, {rs201, rs201};
{
mul.f16x2 r1397, r40, r1399;
}
{
add.f16x2 r1400, r1376, r1397;
}
{
cvt.rn.f16.f64 rs202, fd464;
}
mov.b32 r1405, {rs202, rs202};
{
mul.f16x2 r1403, r43, r1405;
}
{
add.f16x2 r1406, r1382, r1403;
}
{
cvt.rn.f16.f64 rs203, fd455;
}
mov.b32 r1411, {rs203, rs203};
{
mul.f16x2 r1409, r49, r1411;
}
{
add.f16x2 r1412, r1388, r1409;
}
{
cvt.rn.f16.f64 rs204, fd332;
}
mov.b32 r1417, {rs204, rs204};
{
mul.f16x2 r1415, r58, r1417;
}
{
add.f16x2 r1418, r1394, r1415;
}
{
cvt.rn.f16.f64 rs205, fd455;
}
mov.b32 r1423, {rs205, rs205};
{
mul.f16x2 r1421, r52, r1423;
}
{
add.f16x2 r1424, r1400, r1421;
}
{
cvt.rn.f16.f64 rs206, fd332;
}
mov.b32 r1429, {rs206, rs206};
{
mul.f16x2 r1427, r55, r1429;
}
{
add.f16x2 r1430, r1406, r1427;
}
{
cvt.rn.f16.f64 rs207, fd475;
}
mov.b32 r1435, {rs207, rs207};
{
mul.f16x2 r1433, r61, r1435;
}
{
add.f16x2 r1436, r1412, r1433;
}
{
cvt.rn.f16.f64 rs208, fd476;
}
mov.b32 r1441, {rs208, rs208};
{
mul.f16x2 r1439, r70, r1441;
}
{
add.f16x2 r1442, r1418, r1439;
}
{
cvt.rn.f16.f64 rs209, fd475;
}
mov.b32 r1447, {rs209, rs209};
{
mul.f16x2 r1445, r64, r1447;
}
{
add.f16x2 r1448, r1424, r1445;
}
{
cvt.rn.f16.f64 rs210, fd476;
}
mov.b32 r1453, {rs210, rs210};
{
mul.f16x2 r1451, r67, r1453;
}
{
add.f16x2 r1454, r1430, r1451;
}
{
cvt.rn.f16.f64 rs211, fd443;
}
mov.b32 r1459, {rs211, rs211};
{
mul.f16x2 r1457, r73, r1459;
}
{
add.f16x2 r1460, r1436, r1457;
}
{
cvt.rn.f16.f64 rs212, fd388;
}
mov.b32 r1465, {rs212, rs212};
{
mul.f16x2 r1463, r82, r1465;
}
{
add.f16x2 r1466, r1442, r1463;
}
{
cvt.rn.f16.f64 rs213, fd443;
}
mov.b32 r1471, {rs213, rs213};
{
mul.f16x2 r1469, r76, r1471;
}
{
add.f16x2 r1472, r1448, r1469;
}
{
cvt.rn.f16.f64 rs214, fd388;
}
mov.b32 r1477, {rs214, rs214};
{
mul.f16x2 r1475, r79, r1477;
}
{
add.f16x2 r1478, r1454, r1475;
}
{
cvt.rn.f16.f64 rs215, fd483;
}
mov.b32 r1483, {rs215, rs215};
{
mul.f16x2 r1481, r85, r1483;
}
{
add.f16x2 r1484, r1460, r1481;
}
{
cvt.rn.f16.f64 rs216, fd412;
}
mov.b32 r1489, {rs216, rs216};
{
mul.f16x2 r1487, r94, r1489;
}
{
add.f16x2 r1490, r1466, r1487;
}
{
cvt.rn.f16.f64 rs217, fd483;
}
mov.b32 r1495, {rs217, rs217};
{
mul.f16x2 r1493, r88, r1495;
}
{
add.f16x2 r1496, r1472, r1493;
}
{
cvt.rn.f16.f64 rs218, fd412;
}
mov.b32 r1501, {rs218, rs218};
{
mul.f16x2 r1499, r91, r1501;
}
{
add.f16x2 r1502, r1478, r1499;
}
{
cvt.rn.f16.f64 rs219, fd447;
}
mov.b32 r1507, {rs219, rs219};
{
mul.f16x2 r1505, r97, r1507;
}
{
add.f16x2 r1508, r1484, r1505;
}
{
cvt.rn.f16.f64 rs220, fd448;
}
mov.b32 r1513, {rs220, rs220};
{
mul.f16x2 r1511, r106, r1513;
}
{
add.f16x2 r1514, r1490, r1511;
}
{
cvt.rn.f16.f64 rs221, fd447;
}
mov.b32 r1519, {rs221, rs221};
{
mul.f16x2 r1517, r100, r1519;
}
{
add.f16x2 r1520, r1496, r1517;
}
{
cvt.rn.f16.f64 rs222, fd448;
}
mov.b32 r1525, {rs222, rs222};
{
mul.f16x2 r1523, r103, r1525;
}
{
add.f16x2 r1526, r1502, r1523;
}
{
cvt.rn.f16.f64 rs223, fd471;
}
mov.b32 r1531, {rs223, rs223};
{
mul.f16x2 r1529, r109, r1531;
}
{
add.f16x2 r1532, r1508, r1529;
}
{
cvt.rn.f16.f64 rs224, fd416;
}
mov.b32 r1537, {rs224, rs224};
{
mul.f16x2 r1535, r118, r1537;
}
{
add.f16x2 r1538, r1514, r1535;
}
{
cvt.rn.f16.f64 rs225, fd471;
}
mov.b32 r1543, {rs225, rs225};
{
mul.f16x2 r1541, r112, r1543;
}
{
add.f16x2 r1544, r1520, r1541;
}
{
cvt.rn.f16.f64 rs226, fd416;
}
mov.b32 r1549, {rs226, rs226};
{
mul.f16x2 r1547, r115, r1549;
}
{
add.f16x2 r1550, r1526, r1547;
}
{
cvt.rn.f16.f64 rs227, fd459;
}
mov.b32 r1555, {rs227, rs227};
{
mul.f16x2 r1553, r121, r1555;
}
{
add.f16x2 r1556, r1532, r1553;
}
{
cvt.rn.f16.f64 rs228, fd460;
}
mov.b32 r1561, {rs228, rs228};
{
mul.f16x2 r1559, r130, r1561;
}
{
add.f16x2 r1562, r1538, r1559;
}
{
cvt.rn.f16.f64 rs229, fd459;
}
mov.b32 r1567, {rs229, rs229};
{
mul.f16x2 r1565, r124, r1567;
}
{
add.f16x2 r1568, r1544, r1565;
}
{
cvt.rn.f16.f64 rs230, fd460;
}
mov.b32 r1573, {rs230, rs230};
{
mul.f16x2 r1571, r127, r1573;
}
{
add.f16x2 r1574, r1550, r1571;
}
{
sub.f16x2 %10, r1556, r1562;
}
{
add.f16x2 %11, r1568, r1574;
}
{
add.f16x2 %36, r1556, r1562;
}
{
sub.f16x2 %37, r1568, r1574;
}
cvt.rn.f16.s32 rs231, r2980;
mov.b32 r1601, {rs231, rs231};
cvt.rn.f16.s32 rs232, r2980;
mov.b32 r1613, {rs232, rs232};
{
cvt.rn.f16.f64 rs233, fd483;
}
mov.b32 r1593, {rs233, rs233};
{
mul.f16x2 r1591, r1, r1593;
}
{
add.f16x2 r1594, %46, r1591;
}
{
cvt.rn.f16.f64 rs234, fd484;
}
mov.b32 r1599, {rs234, rs234};
{
mul.f16x2 r1597, r10, r1599;
}
{
add.f16x2 r1600, r1601, r1597;
}
{
cvt.rn.f16.f64 rs235, fd483;
}
mov.b32 r1605, {rs235, rs235};
{
mul.f16x2 r1603, r4, r1605;
}
{
add.f16x2 r1606, %47, r1603;
}
{
cvt.rn.f16.f64 rs236, fd484;
}
mov.b32 r1611, {rs236, rs236};
{
mul.f16x2 r1609, r7, r1611;
}
{
add.f16x2 r1612, r1613, r1609;
}
{
cvt.rn.f16.f64 rs237, fd443;
}
mov.b32 r1617, {rs237, rs237};
{
mul.f16x2 r1615, r13, r1617;
}
{
add.f16x2 r1618, r1594, r1615;
}
{
cvt.rn.f16.f64 rs238, fd388;
}
mov.b32 r1623, {rs238, rs238};
{
mul.f16x2 r1621, r22, r1623;
}
{
add.f16x2 r1624, r1600, r1621;
}
{
cvt.rn.f16.f64 rs239, fd443;
}
mov.b32 r1629, {rs239, rs239};
{
mul.f16x2 r1627, r16, r1629;
}
{
add.f16x2 r1630, r1606, r1627;
}
{
cvt.rn.f16.f64 rs240, fd388;
}
mov.b32 r1635, {rs240, rs240};
{
mul.f16x2 r1633, r19, r1635;
}
{
add.f16x2 r1636, r1612, r1633;
}
{
cvt.rn.f16.f64 rs241, fd479;
}
mov.b32 r1641, {rs241, rs241};
{
mul.f16x2 r1639, r25, r1641;
}
{
add.f16x2 r1642, r1618, r1639;
}
{
cvt.rn.f16.f64 rs242, fd480;
}
mov.b32 r1647, {rs242, rs242};
{
mul.f16x2 r1645, r34, r1647;
}
{
add.f16x2 r1648, r1624, r1645;
}
{
cvt.rn.f16.f64 rs243, fd479;
}
mov.b32 r1653, {rs243, rs243};
{
mul.f16x2 r1651, r28, r1653;
}
{
add.f16x2 r1654, r1630, r1651;
}
{
cvt.rn.f16.f64 rs244, fd480;
}
mov.b32 r1659, {rs244, rs244};
{
mul.f16x2 r1657, r31, r1659;
}
{
add.f16x2 r1660, r1636, r1657;
}
{
cvt.rn.f16.f64 rs245, fd447;
}
mov.b32 r1665, {rs245, rs245};
{
mul.f16x2 r1663, r37, r1665;
}
{
add.f16x2 r1666, r1642, r1663;
}
{
cvt.rn.f16.f64 rs246, fd424;
}
mov.b32 r1671, {rs246, rs246};
{
mul.f16x2 r1669, r46, r1671;
}
{
add.f16x2 r1672, r1648, r1669;
}
{
cvt.rn.f16.f64 rs247, fd447;
}
mov.b32 r1677, {rs247, rs247};
{
mul.f16x2 r1675, r40, r1677;
}
{
add.f16x2 r1678, r1654, r1675;
}
{
cvt.rn.f16.f64 rs248, fd424;
}
mov.b32 r1683, {rs248, rs248};
{
mul.f16x2 r1681, r43, r1683;
}
{
add.f16x2 r1684, r1660, r1681;
}
{
cvt.rn.f16.f64 rs249, fd475;
}
mov.b32 r1689, {rs249, rs249};
{
mul.f16x2 r1687, r49, r1689;
}
{
add.f16x2 r1690, r1666, r1687;
}
{
cvt.rn.f16.f64 rs250, fd476;
}
mov.b32 r1695, {rs250, rs250};
{
mul.f16x2 r1693, r58, r1695;
}
{
add.f16x2 r1696, r1672, r1693;
}
{
cvt.rn.f16.f64 rs251, fd475;
}
mov.b32 r1701, {rs251, rs251};
{
mul.f16x2 r1699, r52, r1701;
}
{
add.f16x2 r1702, r1678, r1699;
}
{
cvt.rn.f16.f64 rs252, fd476;
}
mov.b32 r1707, {rs252, rs252};
{
mul.f16x2 r1705, r55, r1707;
}
{
add.f16x2 r1708, r1684, r1705;
}
{
cvt.rn.f16.f64 rs253, fd451;
}
mov.b32 r1713, {rs253, rs253};
{
mul.f16x2 r1711, r61, r1713;
}
{
add.f16x2 r1714, r1690, r1711;
}
{
cvt.rn.f16.f64 rs254, fd368;
}
mov.b32 r1719, {rs254, rs254};
{
mul.f16x2 r1717, r70, r1719;
}
{
add.f16x2 r1720, r1696, r1717;
}
{
cvt.rn.f16.f64 rs255, fd451;
}
mov.b32 r1725, {rs255, rs255};
{
mul.f16x2 r1723, r64, r1725;
}
{
add.f16x2 r1726, r1702, r1723;
}
{
cvt.rn.f16.f64 rs256, fd368;
}
mov.b32 r1731, {rs256, rs256};
{
mul.f16x2 r1729, r67, r1731;
}
{
add.f16x2 r1732, r1708, r1729;
}
{
cvt.rn.f16.f64 rs257, fd471;
}
mov.b32 r1737, {rs257, rs257};
{
mul.f16x2 r1735, r73, r1737;
}
{
add.f16x2 r1738, r1714, r1735;
}
mov.f64 fd472, 0dBFEC698E42F47B09;
{
cvt.rn.f16.f64 rs258, fd472;
}
mov.b32 r1743, {rs258, rs258};
{
mul.f16x2 r1741, r82, r1743;
}
{
add.f16x2 r1744, r1720, r1741;
}
{
cvt.rn.f16.f64 rs259, fd471;
}
mov.b32 r1749, {rs259, rs259};
{
mul.f16x2 r1747, r76, r1749;
}
{
add.f16x2 r1750, r1726, r1747;
}
{
cvt.rn.f16.f64 rs260, fd472;
}
mov.b32 r1755, {rs260, rs260};
{
mul.f16x2 r1753, r79, r1755;
}
{
add.f16x2 r1756, r1732, r1753;
}
{
cvt.rn.f16.f64 rs261, fd455;
}
mov.b32 r1761, {rs261, rs261};
{
mul.f16x2 r1759, r85, r1761;
}
{
add.f16x2 r1762, r1738, r1759;
}
{
cvt.rn.f16.f64 rs262, fd332;
}
mov.b32 r1767, {rs262, rs262};
{
mul.f16x2 r1765, r94, r1767;
}
{
add.f16x2 r1768, r1744, r1765;
}
{
cvt.rn.f16.f64 rs263, fd455;
}
mov.b32 r1773, {rs263, rs263};
{
mul.f16x2 r1771, r88, r1773;
}
{
add.f16x2 r1774, r1750, r1771;
}
{
cvt.rn.f16.f64 rs264, fd332;
}
mov.b32 r1779, {rs264, rs264};
{
mul.f16x2 r1777, r91, r1779;
}
{
add.f16x2 r1780, r1756, r1777;
}
{
cvt.rn.f16.f64 rs265, fd467;
}
mov.b32 r1785, {rs265, rs265};
{
mul.f16x2 r1783, r97, r1785;
}
{
add.f16x2 r1786, r1762, r1783;
}
{
cvt.rn.f16.f64 rs266, fd468;
}
mov.b32 r1791, {rs266, rs266};
{
mul.f16x2 r1789, r106, r1791;
}
{
add.f16x2 r1792, r1768, r1789;
}
{
cvt.rn.f16.f64 rs267, fd467;
}
mov.b32 r1797, {rs267, rs267};
{
mul.f16x2 r1795, r100, r1797;
}
{
add.f16x2 r1798, r1774, r1795;
}
{
cvt.rn.f16.f64 rs268, fd468;
}
mov.b32 r1803, {rs268, rs268};
{
mul.f16x2 r1801, r103, r1803;
}
{
add.f16x2 r1804, r1780, r1801;
}
{
cvt.rn.f16.f64 rs269, fd459;
}
mov.b32 r1809, {rs269, rs269};
{
mul.f16x2 r1807, r109, r1809;
}
{
add.f16x2 r1810, r1786, r1807;
}
{
cvt.rn.f16.f64 rs270, fd420;
}
mov.b32 r1815, {rs270, rs270};
{
mul.f16x2 r1813, r118, r1815;
}
{
add.f16x2 r1816, r1792, r1813;
}
{
cvt.rn.f16.f64 rs271, fd459;
}
mov.b32 r1821, {rs271, rs271};
{
mul.f16x2 r1819, r112, r1821;
}
{
add.f16x2 r1822, r1798, r1819;
}
{
cvt.rn.f16.f64 rs272, fd420;
}
mov.b32 r1827, {rs272, rs272};
{
mul.f16x2 r1825, r115, r1827;
}
{
add.f16x2 r1828, r1804, r1825;
}
{
cvt.rn.f16.f64 rs273, fd463;
}
mov.b32 r1833, {rs273, rs273};
{
mul.f16x2 r1831, r121, r1833;
}
{
add.f16x2 r1834, r1810, r1831;
}
{
cvt.rn.f16.f64 rs274, fd464;
}
mov.b32 r1839, {rs274, rs274};
{
mul.f16x2 r1837, r130, r1839;
}
{
add.f16x2 r1840, r1816, r1837;
}
{
cvt.rn.f16.f64 rs275, fd463;
}
mov.b32 r1845, {rs275, rs275};
{
mul.f16x2 r1843, r124, r1845;
}
{
add.f16x2 r1846, r1822, r1843;
}
{
cvt.rn.f16.f64 rs276, fd464;
}
mov.b32 r1851, {rs276, rs276};
{
mul.f16x2 r1849, r127, r1851;
}
{
add.f16x2 r1852, r1828, r1849;
}
{
sub.f16x2 %12, r1834, r1840;
}
{
add.f16x2 %13, r1846, r1852;
}
{
add.f16x2 %34, r1834, r1840;
}
{
sub.f16x2 %35, r1846, r1852;
}
cvt.rn.f16.s32 rs277, r2980;
mov.b32 r1879, {rs277, rs277};
cvt.rn.f16.s32 rs278, r2980;
mov.b32 r1891, {rs278, rs278};
{
cvt.rn.f16.f64 rs279, fd475;
}
mov.b32 r1871, {rs279, rs279};
{
mul.f16x2 r1869, r1, r1871;
}
{
add.f16x2 r1872, %46, r1869;
}
{
cvt.rn.f16.f64 rs280, fd476;
}
mov.b32 r1877, {rs280, rs280};
{
mul.f16x2 r1875, r10, r1877;
}
{
add.f16x2 r1878, r1879, r1875;
}
{
cvt.rn.f16.f64 rs281, fd475;
}
mov.b32 r1883, {rs281, rs281};
{
mul.f16x2 r1881, r4, r1883;
}
{
add.f16x2 r1884, %47, r1881;
}
{
cvt.rn.f16.f64 rs282, fd476;
}
mov.b32 r1889, {rs282, rs282};
{
mul.f16x2 r1887, r7, r1889;
}
{
add.f16x2 r1890, r1891, r1887;
}
{
cvt.rn.f16.f64 rs283, fd459;
}
mov.b32 r1895, {rs283, rs283};
{
mul.f16x2 r1893, r13, r1895;
}
{
add.f16x2 r1896, r1872, r1893;
}
{
cvt.rn.f16.f64 rs284, fd420;
}
mov.b32 r1901, {rs284, rs284};
{
mul.f16x2 r1899, r22, r1901;
}
{
add.f16x2 r1902, r1878, r1899;
}
{
cvt.rn.f16.f64 rs285, fd459;
}
mov.b32 r1907, {rs285, rs285};
{
mul.f16x2 r1905, r16, r1907;
}
{
add.f16x2 r1908, r1884, r1905;
}
{
cvt.rn.f16.f64 rs286, fd420;
}
mov.b32 r1913, {rs286, rs286};
{
mul.f16x2 r1911, r19, r1913;
}
{
add.f16x2 r1914, r1890, r1911;
}
{
cvt.rn.f16.f64 rs287, fd455;
}
mov.b32 r1919, {rs287, rs287};
{
mul.f16x2 r1917, r25, r1919;
}
{
add.f16x2 r1920, r1896, r1917;
}
{
cvt.rn.f16.f64 rs288, fd456;
}
mov.b32 r1925, {rs288, rs288};
{
mul.f16x2 r1923, r34, r1925;
}
{
add.f16x2 r1926, r1902, r1923;
}
{
cvt.rn.f16.f64 rs289, fd455;
}
mov.b32 r1931, {rs289, rs289};
{
mul.f16x2 r1929, r28, r1931;
}
{
add.f16x2 r1932, r1908, r1929;
}
{
cvt.rn.f16.f64 rs290, fd456;
}
mov.b32 r1937, {rs290, rs290};
{
mul.f16x2 r1935, r31, r1937;
}
{
add.f16x2 r1938, r1914, r1935;
}
{
cvt.rn.f16.f64 rs291, fd479;
}
mov.b32 r1943, {rs291, rs291};
{
mul.f16x2 r1941, r37, r1943;
}
{
add.f16x2 r1944, r1920, r1941;
}
{
cvt.rn.f16.f64 rs292, fd280;
}
mov.b32 r1949, {rs292, rs292};
{
mul.f16x2 r1947, r46, r1949;
}
{
add.f16x2 r1950, r1926, r1947;
}
{
cvt.rn.f16.f64 rs293, fd479;
}
mov.b32 r1955, {rs293, rs293};
{
mul.f16x2 r1953, r40, r1955;
}
{
add.f16x2 r1956, r1932, r1953;
}
{
cvt.rn.f16.f64 rs294, fd280;
}
mov.b32 r1961, {rs294, rs294};
{
mul.f16x2 r1959, r43, r1961;
}
{
add.f16x2 r1962, r1938, r1959;
}
{
cvt.rn.f16.f64 rs295, fd443;
}
mov.b32 r1967, {rs295, rs295};
{
mul.f16x2 r1965, r49, r1967;
}
{
add.f16x2 r1968, r1944, r1965;
}
{
cvt.rn.f16.f64 rs296, fd388;
}
mov.b32 r1973, {rs296, rs296};
{
mul.f16x2 r1971, r58, r1973;
}
{
add.f16x2 r1974, r1950, r1971;
}
{
cvt.rn.f16.f64 rs297, fd443;
}
mov.b32 r1979, {rs297, rs297};
{
mul.f16x2 r1977, r52, r1979;
}
{
add.f16x2 r1980, r1956, r1977;
}
{
cvt.rn.f16.f64 rs298, fd388;
}
mov.b32 r1985, {rs298, rs298};
{
mul.f16x2 r1983, r55, r1985;
}
{
add.f16x2 r1986, r1962, r1983;
}
{
cvt.rn.f16.f64 rs299, fd471;
}
mov.b32 r1991, {rs299, rs299};
{
mul.f16x2 r1989, r61, r1991;
}
{
add.f16x2 r1992, r1968, r1989;
}
{
cvt.rn.f16.f64 rs300, fd472;
}
mov.b32 r1997, {rs300, rs300};
{
mul.f16x2 r1995, r70, r1997;
}
{
add.f16x2 r1998, r1974, r1995;
}
{
cvt.rn.f16.f64 rs301, fd471;
}
mov.b32 r2003, {rs301, rs301};
{
mul.f16x2 r2001, r64, r2003;
}
{
add.f16x2 r2004, r1980, r2001;
}
{
cvt.rn.f16.f64 rs302, fd472;
}
mov.b32 r2009, {rs302, rs302};
{
mul.f16x2 r2007, r67, r2009;
}
{
add.f16x2 r2010, r1986, r2007;
}
{
cvt.rn.f16.f64 rs303, fd463;
}
mov.b32 r2015, {rs303, rs303};
{
mul.f16x2 r2013, r73, r2015;
}
{
add.f16x2 r2016, r1992, r2013;
}
{
cvt.rn.f16.f64 rs304, fd384;
}
mov.b32 r2021, {rs304, rs304};
{
mul.f16x2 r2019, r82, r2021;
}
{
add.f16x2 r2022, r1998, r2019;
}
{
cvt.rn.f16.f64 rs305, fd463;
}
mov.b32 r2027, {rs305, rs305};
{
mul.f16x2 r2025, r76, r2027;
}
{
add.f16x2 r2028, r2004, r2025;
}
{
cvt.rn.f16.f64 rs306, fd384;
}
mov.b32 r2033, {rs306, rs306};
{
mul.f16x2 r2031, r79, r2033;
}
{
add.f16x2 r2034, r2010, r2031;
}
{
cvt.rn.f16.f64 rs307, fd451;
}
mov.b32 r2039, {rs307, rs307};
{
mul.f16x2 r2037, r85, r2039;
}
{
add.f16x2 r2040, r2016, r2037;
}
{
cvt.rn.f16.f64 rs308, fd452;
}
mov.b32 r2045, {rs308, rs308};
{
mul.f16x2 r2043, r94, r2045;
}
{
add.f16x2 r2046, r2022, r2043;
}
{
cvt.rn.f16.f64 rs309, fd451;
}
mov.b32 r2051, {rs309, rs309};
{
mul.f16x2 r2049, r88, r2051;
}
{
add.f16x2 r2052, r2028, r2049;
}
{
cvt.rn.f16.f64 rs310, fd452;
}
mov.b32 r2057, {rs310, rs310};
{
mul.f16x2 r2055, r91, r2057;
}
{
add.f16x2 r2058, r2034, r2055;
}
{
cvt.rn.f16.f64 rs311, fd483;
}
mov.b32 r2063, {rs311, rs311};
{
mul.f16x2 r2061, r97, r2063;
}
{
add.f16x2 r2064, r2040, r2061;
}
{
cvt.rn.f16.f64 rs312, fd412;
}
mov.b32 r2069, {rs312, rs312};
{
mul.f16x2 r2067, r106, r2069;
}
{
add.f16x2 r2070, r2046, r2067;
}
{
cvt.rn.f16.f64 rs313, fd483;
}
mov.b32 r2075, {rs313, rs313};
{
mul.f16x2 r2073, r100, r2075;
}
{
add.f16x2 r2076, r2052, r2073;
}
{
cvt.rn.f16.f64 rs314, fd412;
}
mov.b32 r2081, {rs314, rs314};
{
mul.f16x2 r2079, r103, r2081;
}
{
add.f16x2 r2082, r2058, r2079;
}
{
cvt.rn.f16.f64 rs315, fd447;
}
mov.b32 r2087, {rs315, rs315};
{
mul.f16x2 r2085, r109, r2087;
}
{
add.f16x2 r2088, r2064, r2085;
}
{
cvt.rn.f16.f64 rs316, fd424;
}
mov.b32 r2093, {rs316, rs316};
{
mul.f16x2 r2091, r118, r2093;
}
{
add.f16x2 r2094, r2070, r2091;
}
{
cvt.rn.f16.f64 rs317, fd447;
}
mov.b32 r2099, {rs317, rs317};
{
mul.f16x2 r2097, r112, r2099;
}
{
add.f16x2 r2100, r2076, r2097;
}
{
cvt.rn.f16.f64 rs318, fd424;
}
mov.b32 r2105, {rs318, rs318};
{
mul.f16x2 r2103, r115, r2105;
}
{
add.f16x2 r2106, r2082, r2103;
}
{
cvt.rn.f16.f64 rs319, fd467;
}
mov.b32 r2111, {rs319, rs319};
{
mul.f16x2 r2109, r121, r2111;
}
{
add.f16x2 r2112, r2088, r2109;
}
{
cvt.rn.f16.f64 rs320, fd468;
}
mov.b32 r2117, {rs320, rs320};
{
mul.f16x2 r2115, r130, r2117;
}
{
add.f16x2 r2118, r2094, r2115;
}
{
cvt.rn.f16.f64 rs321, fd467;
}
mov.b32 r2123, {rs321, rs321};
{
mul.f16x2 r2121, r124, r2123;
}
{
add.f16x2 r2124, r2100, r2121;
}
{
cvt.rn.f16.f64 rs322, fd468;
}
mov.b32 r2129, {rs322, rs322};
{
mul.f16x2 r2127, r127, r2129;
}
{
add.f16x2 r2130, r2106, r2127;
}
{
sub.f16x2 %14, r2112, r2118;
}
{
add.f16x2 %15, r2124, r2130;
}
{
add.f16x2 %32, r2112, r2118;
}
{
sub.f16x2 %33, r2124, r2130;
}
cvt.rn.f16.s32 rs323, r2980;
mov.b32 r2157, {rs323, rs323};
cvt.rn.f16.s32 rs324, r2980;
mov.b32 r2169, {rs324, rs324};
{
cvt.rn.f16.f64 rs325, fd467;
}
mov.b32 r2149, {rs325, rs325};
{
mul.f16x2 r2147, r1, r2149;
}
{
add.f16x2 r2150, %46, r2147;
}
{
cvt.rn.f16.f64 rs326, fd468;
}
mov.b32 r2155, {rs326, rs326};
{
mul.f16x2 r2153, r10, r2155;
}
{
add.f16x2 r2156, r2157, r2153;
}
{
cvt.rn.f16.f64 rs327, fd467;
}
mov.b32 r2161, {rs327, rs327};
{
mul.f16x2 r2159, r4, r2161;
}
{
add.f16x2 r2162, %47, r2159;
}
{
cvt.rn.f16.f64 rs328, fd468;
}
mov.b32 r2167, {rs328, rs328};
{
mul.f16x2 r2165, r7, r2167;
}
{
add.f16x2 r2168, r2169, r2165;
}
{
cvt.rn.f16.f64 rs329, fd475;
}
mov.b32 r2173, {rs329, rs329};
{
mul.f16x2 r2171, r13, r2173;
}
{
add.f16x2 r2174, r2150, r2171;
}
{
cvt.rn.f16.f64 rs330, fd316;
}
mov.b32 r2179, {rs330, rs330};
{
mul.f16x2 r2177, r22, r2179;
}
{
add.f16x2 r2180, r2156, r2177;
}
{
cvt.rn.f16.f64 rs331, fd475;
}
mov.b32 r2185, {rs331, rs331};
{
mul.f16x2 r2183, r16, r2185;
}
{
add.f16x2 r2186, r2162, r2183;
}
{
cvt.rn.f16.f64 rs332, fd316;
}
mov.b32 r2191, {rs332, rs332};
{
mul.f16x2 r2189, r19, r2191;
}
{
add.f16x2 r2192, r2168, r2189;
}
{
cvt.rn.f16.f64 rs333, fd447;
}
mov.b32 r2197, {rs333, rs333};
{
mul.f16x2 r2195, r25, r2197;
}
{
add.f16x2 r2198, r2174, r2195;
}
{
cvt.rn.f16.f64 rs334, fd424;
}
mov.b32 r2203, {rs334, rs334};
{
mul.f16x2 r2201, r34, r2203;
}
{
add.f16x2 r2204, r2180, r2201;
}
{
cvt.rn.f16.f64 rs335, fd447;
}
mov.b32 r2209, {rs335, rs335};
{
mul.f16x2 r2207, r28, r2209;
}
{
add.f16x2 r2210, r2186, r2207;
}
{
cvt.rn.f16.f64 rs336, fd424;
}
mov.b32 r2215, {rs336, rs336};
{
mul.f16x2 r2213, r31, r2215;
}
{
add.f16x2 r2216, r2192, r2213;
}
{
cvt.rn.f16.f64 rs337, fd459;
}
mov.b32 r2221, {rs337, rs337};
{
mul.f16x2 r2219, r37, r2221;
}
{
add.f16x2 r2222, r2198, r2219;
}
{
cvt.rn.f16.f64 rs338, fd460;
}
mov.b32 r2227, {rs338, rs338};
{
mul.f16x2 r2225, r46, r2227;
}
{
add.f16x2 r2228, r2204, r2225;
}
{
cvt.rn.f16.f64 rs339, fd459;
}
mov.b32 r2233, {rs339, rs339};
{
mul.f16x2 r2231, r40, r2233;
}
{
add.f16x2 r2234, r2210, r2231;
}
{
cvt.rn.f16.f64 rs340, fd460;
}
mov.b32 r2239, {rs340, rs340};
{
mul.f16x2 r2237, r43, r2239;
}
{
add.f16x2 r2240, r2216, r2237;
}
{
cvt.rn.f16.f64 rs341, fd483;
}
mov.b32 r2245, {rs341, rs341};
{
mul.f16x2 r2243, r49, r2245;
}
{
add.f16x2 r2246, r2222, r2243;
}
{
cvt.rn.f16.f64 rs342, fd412;
}
mov.b32 r2251, {rs342, rs342};
{
mul.f16x2 r2249, r58, r2251;
}
{
add.f16x2 r2252, r2228, r2249;
}
{
cvt.rn.f16.f64 rs343, fd483;
}
mov.b32 r2257, {rs343, rs343};
{
mul.f16x2 r2255, r52, r2257;
}
{
add.f16x2 r2258, r2234, r2255;
}
{
cvt.rn.f16.f64 rs344, fd412;
}
mov.b32 r2263, {rs344, rs344};
{
mul.f16x2 r2261, r55, r2263;
}
{
add.f16x2 r2264, r2240, r2261;
}
{
cvt.rn.f16.f64 rs345, fd455;
}
mov.b32 r2269, {rs345, rs345};
{
mul.f16x2 r2267, r61, r2269;
}
{
add.f16x2 r2270, r2246, r2267;
}
{
cvt.rn.f16.f64 rs346, fd332;
}
mov.b32 r2275, {rs346, rs346};
{
mul.f16x2 r2273, r70, r2275;
}
{
add.f16x2 r2276, r2252, r2273;
}
{
cvt.rn.f16.f64 rs347, fd455;
}
mov.b32 r2281, {rs347, rs347};
{
mul.f16x2 r2279, r64, r2281;
}
{
add.f16x2 r2282, r2258, r2279;
}
{
cvt.rn.f16.f64 rs348, fd332;
}
mov.b32 r2287, {rs348, rs348};
{
mul.f16x2 r2285, r67, r2287;
}
{
add.f16x2 r2288, r2264, r2285;
}
{
cvt.rn.f16.f64 rs349, fd451;
}
mov.b32 r2293, {rs349, rs349};
{
mul.f16x2 r2291, r73, r2293;
}
{
add.f16x2 r2294, r2270, r2291;
}
{
cvt.rn.f16.f64 rs350, fd452;
}
mov.b32 r2299, {rs350, rs350};
{
mul.f16x2 r2297, r82, r2299;
}
{
add.f16x2 r2300, r2276, r2297;
}
{
cvt.rn.f16.f64 rs351, fd451;
}
mov.b32 r2305, {rs351, rs351};
{
mul.f16x2 r2303, r76, r2305;
}
{
add.f16x2 r2306, r2282, r2303;
}
{
cvt.rn.f16.f64 rs352, fd452;
}
mov.b32 r2311, {rs352, rs352};
{
mul.f16x2 r2309, r79, r2311;
}
{
add.f16x2 r2312, r2288, r2309;
}
{
cvt.rn.f16.f64 rs353, fd479;
}
mov.b32 r2317, {rs353, rs353};
{
mul.f16x2 r2315, r85, r2317;
}
{
add.f16x2 r2318, r2294, r2315;
}
{
cvt.rn.f16.f64 rs354, fd480;
}
mov.b32 r2323, {rs354, rs354};
{
mul.f16x2 r2321, r94, r2323;
}
{
add.f16x2 r2324, r2300, r2321;
}
{
cvt.rn.f16.f64 rs355, fd479;
}
mov.b32 r2329, {rs355, rs355};
{
mul.f16x2 r2327, r88, r2329;
}
{
add.f16x2 r2330, r2306, r2327;
}
{
cvt.rn.f16.f64 rs356, fd480;
}
mov.b32 r2335, {rs356, rs356};
{
mul.f16x2 r2333, r91, r2335;
}
{
add.f16x2 r2336, r2312, r2333;
}
{
cvt.rn.f16.f64 rs357, fd463;
}
mov.b32 r2341, {rs357, rs357};
{
mul.f16x2 r2339, r97, r2341;
}
{
add.f16x2 r2342, r2318, r2339;
}
{
cvt.rn.f16.f64 rs358, fd384;
}
mov.b32 r2347, {rs358, rs358};
{
mul.f16x2 r2345, r106, r2347;
}
{
add.f16x2 r2348, r2324, r2345;
}
{
cvt.rn.f16.f64 rs359, fd463;
}
mov.b32 r2353, {rs359, rs359};
{
mul.f16x2 r2351, r100, r2353;
}
{
add.f16x2 r2354, r2330, r2351;
}
{
cvt.rn.f16.f64 rs360, fd384;
}
mov.b32 r2359, {rs360, rs360};
{
mul.f16x2 r2357, r103, r2359;
}
{
add.f16x2 r2360, r2336, r2357;
}
{
cvt.rn.f16.f64 rs361, fd443;
}
mov.b32 r2365, {rs361, rs361};
{
mul.f16x2 r2363, r109, r2365;
}
{
add.f16x2 r2366, r2342, r2363;
}
{
cvt.rn.f16.f64 rs362, fd444;
}
mov.b32 r2371, {rs362, rs362};
{
mul.f16x2 r2369, r118, r2371;
}
{
add.f16x2 r2372, r2348, r2369;
}
{
cvt.rn.f16.f64 rs363, fd443;
}
mov.b32 r2377, {rs363, rs363};
{
mul.f16x2 r2375, r112, r2377;
}
{
add.f16x2 r2378, r2354, r2375;
}
{
cvt.rn.f16.f64 rs364, fd444;
}
mov.b32 r2383, {rs364, rs364};
{
mul.f16x2 r2381, r115, r2383;
}
{
add.f16x2 r2384, r2360, r2381;
}
{
cvt.rn.f16.f64 rs365, fd471;
}
mov.b32 r2389, {rs365, rs365};
{
mul.f16x2 r2387, r121, r2389;
}
{
add.f16x2 r2390, r2366, r2387;
}
{
cvt.rn.f16.f64 rs366, fd472;
}
mov.b32 r2395, {rs366, rs366};
{
mul.f16x2 r2393, r130, r2395;
}
{
add.f16x2 r2396, r2372, r2393;
}
{
cvt.rn.f16.f64 rs367, fd471;
}
mov.b32 r2401, {rs367, rs367};
{
mul.f16x2 r2399, r124, r2401;
}
{
add.f16x2 r2402, r2378, r2399;
}
{
cvt.rn.f16.f64 rs368, fd472;
}
mov.b32 r2407, {rs368, rs368};
{
mul.f16x2 r2405, r127, r2407;
}
{
add.f16x2 r2408, r2384, r2405;
}
{
sub.f16x2 %16, r2390, r2396;
}
{
add.f16x2 %17, r2402, r2408;
}
{
add.f16x2 %30, r2390, r2396;
}
{
sub.f16x2 %31, r2402, r2408;
}
cvt.rn.f16.s32 rs369, r2980;
mov.b32 r2435, {rs369, rs369};
cvt.rn.f16.s32 rs370, r2980;
mov.b32 r2447, {rs370, rs370};
{
cvt.rn.f16.f64 rs371, fd459;
}
mov.b32 r2427, {rs371, rs371};
{
mul.f16x2 r2425, r1, r2427;
}
{
add.f16x2 r2428, %46, r2425;
}
{
cvt.rn.f16.f64 rs372, fd460;
}
mov.b32 r2433, {rs372, rs372};
{
mul.f16x2 r2431, r10, r2433;
}
{
add.f16x2 r2434, r2435, r2431;
}
{
cvt.rn.f16.f64 rs373, fd459;
}
mov.b32 r2439, {rs373, rs373};
{
mul.f16x2 r2437, r4, r2439;
}
{
add.f16x2 r2440, %47, r2437;
}
{
cvt.rn.f16.f64 rs374, fd460;
}
mov.b32 r2445, {rs374, rs374};
{
mul.f16x2 r2443, r7, r2445;
}
{
add.f16x2 r2446, r2447, r2443;
}
{
cvt.rn.f16.f64 rs375, fd479;
}
mov.b32 r2451, {rs375, rs375};
{
mul.f16x2 r2449, r13, r2451;
}
{
add.f16x2 r2452, r2428, r2449;
}
{
cvt.rn.f16.f64 rs376, fd480;
}
mov.b32 r2457, {rs376, rs376};
{
mul.f16x2 r2455, r22, r2457;
}
{
add.f16x2 r2458, r2434, r2455;
}
{
cvt.rn.f16.f64 rs377, fd479;
}
mov.b32 r2463, {rs377, rs377};
{
mul.f16x2 r2461, r16, r2463;
}
{
add.f16x2 r2464, r2440, r2461;
}
{
cvt.rn.f16.f64 rs378, fd480;
}
mov.b32 r2469, {rs378, rs378};
{
mul.f16x2 r2467, r19, r2469;
}
{
add.f16x2 r2470, r2446, r2467;
}
{
cvt.rn.f16.f64 rs379, fd471;
}
mov.b32 r2475, {rs379, rs379};
{
mul.f16x2 r2473, r25, r2475;
}
{
add.f16x2 r2476, r2452, r2473;
}
{
cvt.rn.f16.f64 rs380, fd416;
}
mov.b32 r2481, {rs380, rs380};
{
mul.f16x2 r2479, r34, r2481;
}
{
add.f16x2 r2482, r2458, r2479;
}
{
cvt.rn.f16.f64 rs381, fd471;
}
mov.b32 r2487, {rs381, rs381};
{
mul.f16x2 r2485, r28, r2487;
}
{
add.f16x2 r2488, r2464, r2485;
}
{
cvt.rn.f16.f64 rs382, fd416;
}
mov.b32 r2493, {rs382, rs382};
{
mul.f16x2 r2491, r31, r2493;
}
{
add.f16x2 r2494, r2470, r2491;
}
{
cvt.rn.f16.f64 rs383, fd451;
}
mov.b32 r2499, {rs383, rs383};
{
mul.f16x2 r2497, r37, r2499;
}
{
add.f16x2 r2500, r2476, r2497;
}
{
cvt.rn.f16.f64 rs384, fd368;
}
mov.b32 r2505, {rs384, rs384};
{
mul.f16x2 r2503, r46, r2505;
}
{
add.f16x2 r2506, r2482, r2503;
}
{
cvt.rn.f16.f64 rs385, fd451;
}
mov.b32 r2511, {rs385, rs385};
{
mul.f16x2 r2509, r40, r2511;
}
{
add.f16x2 r2512, r2488, r2509;
}
{
cvt.rn.f16.f64 rs386, fd368;
}
mov.b32 r2517, {rs386, rs386};
{
mul.f16x2 r2515, r43, r2517;
}
{
add.f16x2 r2518, r2494, r2515;
}
{
cvt.rn.f16.f64 rs387, fd447;
}
mov.b32 r2523, {rs387, rs387};
{
mul.f16x2 r2521, r49, r2523;
}
{
add.f16x2 r2524, r2500, r2521;
}
{
cvt.rn.f16.f64 rs388, fd448;
}
mov.b32 r2529, {rs388, rs388};
{
mul.f16x2 r2527, r58, r2529;
}
{
add.f16x2 r2530, r2506, r2527;
}
{
cvt.rn.f16.f64 rs389, fd447;
}
mov.b32 r2535, {rs389, rs389};
{
mul.f16x2 r2533, r52, r2535;
}
{
add.f16x2 r2536, r2512, r2533;
}
{
cvt.rn.f16.f64 rs390, fd448;
}
mov.b32 r2541, {rs390, rs390};
{
mul.f16x2 r2539, r55, r2541;
}
{
add.f16x2 r2542, r2518, r2539;
}
{
cvt.rn.f16.f64 rs391, fd467;
}
mov.b32 r2547, {rs391, rs391};
{
mul.f16x2 r2545, r61, r2547;
}
{
add.f16x2 r2548, r2524, r2545;
}
{
cvt.rn.f16.f64 rs392, fd468;
}
mov.b32 r2553, {rs392, rs392};
{
mul.f16x2 r2551, r70, r2553;
}
{
add.f16x2 r2554, r2530, r2551;
}
{
cvt.rn.f16.f64 rs393, fd467;
}
mov.b32 r2559, {rs393, rs393};
{
mul.f16x2 r2557, r64, r2559;
}
{
add.f16x2 r2560, r2536, r2557;
}
{
cvt.rn.f16.f64 rs394, fd468;
}
mov.b32 r2565, {rs394, rs394};
{
mul.f16x2 r2563, r67, r2565;
}
{
add.f16x2 r2566, r2542, r2563;
}
{
cvt.rn.f16.f64 rs395, fd483;
}
mov.b32 r2571, {rs395, rs395};
{
mul.f16x2 r2569, r73, r2571;
}
{
add.f16x2 r2572, r2548, r2569;
}
{
cvt.rn.f16.f64 rs396, fd412;
}
mov.b32 r2577, {rs396, rs396};
{
mul.f16x2 r2575, r82, r2577;
}
{
add.f16x2 r2578, r2554, r2575;
}
{
cvt.rn.f16.f64 rs397, fd483;
}
mov.b32 r2583, {rs397, rs397};
{
mul.f16x2 r2581, r76, r2583;
}
{
add.f16x2 r2584, r2560, r2581;
}
{
cvt.rn.f16.f64 rs398, fd412;
}
mov.b32 r2589, {rs398, rs398};
{
mul.f16x2 r2587, r79, r2589;
}
{
add.f16x2 r2590, r2566, r2587;
}
{
cvt.rn.f16.f64 rs399, fd463;
}
mov.b32 r2595, {rs399, rs399};
{
mul.f16x2 r2593, r85, r2595;
}
{
add.f16x2 r2596, r2572, r2593;
}
{
cvt.rn.f16.f64 rs400, fd384;
}
mov.b32 r2601, {rs400, rs400};
{
mul.f16x2 r2599, r94, r2601;
}
{
add.f16x2 r2602, r2578, r2599;
}
{
cvt.rn.f16.f64 rs401, fd463;
}
mov.b32 r2607, {rs401, rs401};
{
mul.f16x2 r2605, r88, r2607;
}
{
add.f16x2 r2608, r2584, r2605;
}
{
cvt.rn.f16.f64 rs402, fd384;
}
mov.b32 r2613, {rs402, rs402};
{
mul.f16x2 r2611, r91, r2613;
}
{
add.f16x2 r2614, r2590, r2611;
}
{
cvt.rn.f16.f64 rs403, fd443;
}
mov.b32 r2619, {rs403, rs403};
{
mul.f16x2 r2617, r97, r2619;
}
{
add.f16x2 r2620, r2596, r2617;
}
{
cvt.rn.f16.f64 rs404, fd388;
}
mov.b32 r2625, {rs404, rs404};
{
mul.f16x2 r2623, r106, r2625;
}
{
add.f16x2 r2626, r2602, r2623;
}
{
cvt.rn.f16.f64 rs405, fd443;
}
mov.b32 r2631, {rs405, rs405};
{
mul.f16x2 r2629, r100, r2631;
}
{
add.f16x2 r2632, r2608, r2629;
}
{
cvt.rn.f16.f64 rs406, fd388;
}
mov.b32 r2637, {rs406, rs406};
{
mul.f16x2 r2635, r103, r2637;
}
{
add.f16x2 r2638, r2614, r2635;
}
{
cvt.rn.f16.f64 rs407, fd455;
}
mov.b32 r2643, {rs407, rs407};
{
mul.f16x2 r2641, r109, r2643;
}
{
add.f16x2 r2644, r2620, r2641;
}
{
cvt.rn.f16.f64 rs408, fd456;
}
mov.b32 r2649, {rs408, rs408};
{
mul.f16x2 r2647, r118, r2649;
}
{
add.f16x2 r2650, r2626, r2647;
}
{
cvt.rn.f16.f64 rs409, fd455;
}
mov.b32 r2655, {rs409, rs409};
{
mul.f16x2 r2653, r112, r2655;
}
{
add.f16x2 r2656, r2632, r2653;
}
{
cvt.rn.f16.f64 rs410, fd456;
}
mov.b32 r2661, {rs410, rs410};
{
mul.f16x2 r2659, r115, r2661;
}
{
add.f16x2 r2662, r2638, r2659;
}
{
cvt.rn.f16.f64 rs411, fd475;
}
mov.b32 r2667, {rs411, rs411};
{
mul.f16x2 r2665, r121, r2667;
}
{
add.f16x2 r2668, r2644, r2665;
}
{
cvt.rn.f16.f64 rs412, fd476;
}
mov.b32 r2673, {rs412, rs412};
{
mul.f16x2 r2671, r130, r2673;
}
{
add.f16x2 r2674, r2650, r2671;
}
{
cvt.rn.f16.f64 rs413, fd475;
}
mov.b32 r2679, {rs413, rs413};
{
mul.f16x2 r2677, r124, r2679;
}
{
add.f16x2 r2680, r2656, r2677;
}
{
cvt.rn.f16.f64 rs414, fd476;
}
mov.b32 r2685, {rs414, rs414};
{
mul.f16x2 r2683, r127, r2685;
}
{
add.f16x2 r2686, r2662, r2683;
}
{
sub.f16x2 %18, r2668, r2674;
}
{
add.f16x2 %19, r2680, r2686;
}
{
add.f16x2 %28, r2668, r2674;
}
{
sub.f16x2 %29, r2680, r2686;
}
cvt.rn.f16.s32 rs415, r2980;
mov.b32 r2713, {rs415, rs415};
cvt.rn.f16.s32 rs416, r2980;
mov.b32 r2725, {rs416, rs416};
{
cvt.rn.f16.f64 rs417, fd451;
}
mov.b32 r2705, {rs417, rs417};
{
mul.f16x2 r2703, r1, r2705;
}
{
add.f16x2 r2706, %46, r2703;
}
{
cvt.rn.f16.f64 rs418, fd452;
}
mov.b32 r2711, {rs418, rs418};
{
mul.f16x2 r2709, r10, r2711;
}
{
add.f16x2 r2712, r2713, r2709;
}
{
cvt.rn.f16.f64 rs419, fd451;
}
mov.b32 r2717, {rs419, rs419};
{
mul.f16x2 r2715, r4, r2717;
}
{
add.f16x2 r2718, %47, r2715;
}
{
cvt.rn.f16.f64 rs420, fd452;
}
mov.b32 r2723, {rs420, rs420};
{
mul.f16x2 r2721, r7, r2723;
}
{
add.f16x2 r2724, r2725, r2721;
}
{
cvt.rn.f16.f64 rs421, fd463;
}
mov.b32 r2729, {rs421, rs421};
{
mul.f16x2 r2727, r13, r2729;
}
{
add.f16x2 r2730, r2706, r2727;
}
{
cvt.rn.f16.f64 rs422, fd464;
}
mov.b32 r2735, {rs422, rs422};
{
mul.f16x2 r2733, r22, r2735;
}
{
add.f16x2 r2736, r2712, r2733;
}
{
cvt.rn.f16.f64 rs423, fd463;
}
mov.b32 r2741, {rs423, rs423};
{
mul.f16x2 r2739, r16, r2741;
}
{
add.f16x2 r2742, r2718, r2739;
}
{
cvt.rn.f16.f64 rs424, fd464;
}
mov.b32 r2747, {rs424, rs424};
{
mul.f16x2 r2745, r19, r2747;
}
{
add.f16x2 r2748, r2724, r2745;
}
{
cvt.rn.f16.f64 rs425, fd475;
}
mov.b32 r2753, {rs425, rs425};
{
mul.f16x2 r2751, r25, r2753;
}
{
add.f16x2 r2754, r2730, r2751;
}
{
cvt.rn.f16.f64 rs426, fd476;
}
mov.b32 r2759, {rs426, rs426};
{
mul.f16x2 r2757, r34, r2759;
}
{
add.f16x2 r2760, r2736, r2757;
}
{
cvt.rn.f16.f64 rs427, fd475;
}
mov.b32 r2765, {rs427, rs427};
{
mul.f16x2 r2763, r28, r2765;
}
{
add.f16x2 r2766, r2742, r2763;
}
{
cvt.rn.f16.f64 rs428, fd476;
}
mov.b32 r2771, {rs428, rs428};
{
mul.f16x2 r2769, r31, r2771;
}
{
add.f16x2 r2772, r2748, r2769;
}
{
cvt.rn.f16.f64 rs429, fd483;
}
mov.b32 r2777, {rs429, rs429};
{
mul.f16x2 r2775, r37, r2777;
}
{
add.f16x2 r2778, r2754, r2775;
}
{
cvt.rn.f16.f64 rs430, fd412;
}
mov.b32 r2783, {rs430, rs430};
{
mul.f16x2 r2781, r46, r2783;
}
{
add.f16x2 r2784, r2760, r2781;
}
{
cvt.rn.f16.f64 rs431, fd483;
}
mov.b32 r2789, {rs431, rs431};
{
mul.f16x2 r2787, r40, r2789;
}
{
add.f16x2 r2790, r2766, r2787;
}
{
cvt.rn.f16.f64 rs432, fd412;
}
mov.b32 r2795, {rs432, rs432};
{
mul.f16x2 r2793, r43, r2795;
}
{
add.f16x2 r2796, r2772, r2793;
}
{
cvt.rn.f16.f64 rs433, fd471;
}
mov.b32 r2801, {rs433, rs433};
{
mul.f16x2 r2799, r49, r2801;
}
{
add.f16x2 r2802, r2778, r2799;
}
{
cvt.rn.f16.f64 rs434, fd416;
}
mov.b32 r2807, {rs434, rs434};
{
mul.f16x2 r2805, r58, r2807;
}
{
add.f16x2 r2808, r2784, r2805;
}
{
cvt.rn.f16.f64 rs435, fd471;
}
mov.b32 r2813, {rs435, rs435};
{
mul.f16x2 r2811, r52, r2813;
}
{
add.f16x2 r2814, r2790, r2811;
}
{
cvt.rn.f16.f64 rs436, fd416;
}
mov.b32 r2819, {rs436, rs436};
{
mul.f16x2 r2817, r55, r2819;
}
{
add.f16x2 r2820, r2796, r2817;
}
{
cvt.rn.f16.f64 rs437, fd459;
}
mov.b32 r2825, {rs437, rs437};
{
mul.f16x2 r2823, r61, r2825;
}
{
add.f16x2 r2826, r2802, r2823;
}
{
cvt.rn.f16.f64 rs438, fd420;
}
mov.b32 r2831, {rs438, rs438};
{
mul.f16x2 r2829, r70, r2831;
}
{
add.f16x2 r2832, r2808, r2829;
}
{
cvt.rn.f16.f64 rs439, fd459;
}
mov.b32 r2837, {rs439, rs439};
{
mul.f16x2 r2835, r64, r2837;
}
{
add.f16x2 r2838, r2814, r2835;
}
{
cvt.rn.f16.f64 rs440, fd420;
}
mov.b32 r2843, {rs440, rs440};
{
mul.f16x2 r2841, r67, r2843;
}
{
add.f16x2 r2844, r2820, r2841;
}
{
cvt.rn.f16.f64 rs441, fd447;
}
mov.b32 r2849, {rs441, rs441};
{
mul.f16x2 r2847, r73, r2849;
}
{
add.f16x2 r2850, r2826, r2847;
}
{
cvt.rn.f16.f64 rs442, fd424;
}
mov.b32 r2855, {rs442, rs442};
{
mul.f16x2 r2853, r82, r2855;
}
{
add.f16x2 r2856, r2832, r2853;
}
{
cvt.rn.f16.f64 rs443, fd447;
}
mov.b32 r2861, {rs443, rs443};
{
mul.f16x2 r2859, r76, r2861;
}
{
add.f16x2 r2862, r2838, r2859;
}
{
cvt.rn.f16.f64 rs444, fd424;
}
mov.b32 r2867, {rs444, rs444};
{
mul.f16x2 r2865, r79, r2867;
}
{
add.f16x2 r2868, r2844, r2865;
}
{
cvt.rn.f16.f64 rs445, fd443;
}
mov.b32 r2873, {rs445, rs445};
{
mul.f16x2 r2871, r85, r2873;
}
{
add.f16x2 r2874, r2850, r2871;
}
{
cvt.rn.f16.f64 rs446, fd444;
}
mov.b32 r2879, {rs446, rs446};
{
mul.f16x2 r2877, r94, r2879;
}
{
add.f16x2 r2880, r2856, r2877;
}
{
cvt.rn.f16.f64 rs447, fd443;
}
mov.b32 r2885, {rs447, rs447};
{
mul.f16x2 r2883, r88, r2885;
}
{
add.f16x2 r2886, r2862, r2883;
}
{
cvt.rn.f16.f64 rs448, fd444;
}
mov.b32 r2891, {rs448, rs448};
{
mul.f16x2 r2889, r91, r2891;
}
{
add.f16x2 r2892, r2868, r2889;
}
{
cvt.rn.f16.f64 rs449, fd455;
}
mov.b32 r2897, {rs449, rs449};
{
mul.f16x2 r2895, r97, r2897;
}
{
add.f16x2 r2898, r2874, r2895;
}
{
cvt.rn.f16.f64 rs450, fd456;
}
mov.b32 r2903, {rs450, rs450};
{
mul.f16x2 r2901, r106, r2903;
}
{
add.f16x2 r2904, r2880, r2901;
}
{
cvt.rn.f16.f64 rs451, fd455;
}
mov.b32 r2909, {rs451, rs451};
{
mul.f16x2 r2907, r100, r2909;
}
{
add.f16x2 r2910, r2886, r2907;
}
{
cvt.rn.f16.f64 rs452, fd456;
}
mov.b32 r2915, {rs452, rs452};
{
mul.f16x2 r2913, r103, r2915;
}
{
add.f16x2 r2916, r2892, r2913;
}
{
cvt.rn.f16.f64 rs453, fd467;
}
mov.b32 r2921, {rs453, rs453};
{
mul.f16x2 r2919, r109, r2921;
}
{
add.f16x2 r2922, r2898, r2919;
}
{
cvt.rn.f16.f64 rs454, fd468;
}
mov.b32 r2927, {rs454, rs454};
{
mul.f16x2 r2925, r118, r2927;
}
{
add.f16x2 r2928, r2904, r2925;
}
{
cvt.rn.f16.f64 rs455, fd467;
}
mov.b32 r2933, {rs455, rs455};
{
mul.f16x2 r2931, r112, r2933;
}
{
add.f16x2 r2934, r2910, r2931;
}
{
cvt.rn.f16.f64 rs456, fd468;
}
mov.b32 r2939, {rs456, rs456};
{
mul.f16x2 r2937, r115, r2939;
}
{
add.f16x2 r2940, r2916, r2937;
}
{
cvt.rn.f16.f64 rs457, fd479;
}
mov.b32 r2945, {rs457, rs457};
{
mul.f16x2 r2943, r121, r2945;
}
{
add.f16x2 r2946, r2922, r2943;
}
{
cvt.rn.f16.f64 rs458, fd480;
}
mov.b32 r2951, {rs458, rs458};
{
mul.f16x2 r2949, r130, r2951;
}
{
add.f16x2 r2952, r2928, r2949;
}
{
cvt.rn.f16.f64 rs459, fd479;
}
mov.b32 r2957, {rs459, rs459};
{
mul.f16x2 r2955, r124, r2957;
}
{
add.f16x2 r2958, r2934, r2955;
}
{
cvt.rn.f16.f64 rs460, fd480;
}
mov.b32 r2963, {rs460, rs460};
{
mul.f16x2 r2961, r127, r2963;
}
{
add.f16x2 r2964, r2940, r2961;
}
{
sub.f16x2 %20, r2946, r2952;
}
{
add.f16x2 %21, r2958, r2964;
}
{
add.f16x2 %26, r2946, r2952;
}
{
sub.f16x2 %27, r2958, r2964;
}
cvt.rn.f16.s32 rs461, r2980;
mov.b32 r2991, {rs461, rs461};
cvt.rn.f16.s32 rs462, r2980;
mov.b32 r3003, {rs462, rs462};
{
cvt.rn.f16.f64 rs463, fd443;
}
mov.b32 r2983, {rs463, rs463};
{
mul.f16x2 r2981, r1, r2983;
}
{
add.f16x2 r2984, %46, r2981;
}
{
cvt.rn.f16.f64 rs464, fd444;
}
mov.b32 r2989, {rs464, rs464};
{
mul.f16x2 r2987, r10, r2989;
}
{
add.f16x2 r2990, r2991, r2987;
}
{
cvt.rn.f16.f64 rs465, fd443;
}
mov.b32 r2995, {rs465, rs465};
{
mul.f16x2 r2993, r4, r2995;
}
{
add.f16x2 r2996, %47, r2993;
}
{
cvt.rn.f16.f64 rs466, fd444;
}
mov.b32 r3001, {rs466, rs466};
{
mul.f16x2 r2999, r7, r3001;
}
{
add.f16x2 r3002, r3003, r2999;
}
{
cvt.rn.f16.f64 rs467, fd447;
}
mov.b32 r3007, {rs467, rs467};
{
mul.f16x2 r3005, r13, r3007;
}
{
add.f16x2 r3008, r2984, r3005;
}
{
cvt.rn.f16.f64 rs468, fd448;
}
mov.b32 r3013, {rs468, rs468};
{
mul.f16x2 r3011, r22, r3013;
}
{
add.f16x2 r3014, r2990, r3011;
}
{
cvt.rn.f16.f64 rs469, fd447;
}
mov.b32 r3019, {rs469, rs469};
{
mul.f16x2 r3017, r16, r3019;
}
{
add.f16x2 r3020, r2996, r3017;
}
{
cvt.rn.f16.f64 rs470, fd448;
}
mov.b32 r3025, {rs470, rs470};
{
mul.f16x2 r3023, r19, r3025;
}
{
add.f16x2 r3026, r3002, r3023;
}
{
cvt.rn.f16.f64 rs471, fd451;
}
mov.b32 r3031, {rs471, rs471};
{
mul.f16x2 r3029, r25, r3031;
}
{
add.f16x2 r3032, r3008, r3029;
}
{
cvt.rn.f16.f64 rs472, fd452;
}
mov.b32 r3037, {rs472, rs472};
{
mul.f16x2 r3035, r34, r3037;
}
{
add.f16x2 r3038, r3014, r3035;
}
{
cvt.rn.f16.f64 rs473, fd451;
}
mov.b32 r3043, {rs473, rs473};
{
mul.f16x2 r3041, r28, r3043;
}
{
add.f16x2 r3044, r3020, r3041;
}
{
cvt.rn.f16.f64 rs474, fd452;
}
mov.b32 r3049, {rs474, rs474};
{
mul.f16x2 r3047, r31, r3049;
}
{
add.f16x2 r3050, r3026, r3047;
}
{
cvt.rn.f16.f64 rs475, fd455;
}
mov.b32 r3055, {rs475, rs475};
{
mul.f16x2 r3053, r37, r3055;
}
{
add.f16x2 r3056, r3032, r3053;
}
{
cvt.rn.f16.f64 rs476, fd456;
}
mov.b32 r3061, {rs476, rs476};
{
mul.f16x2 r3059, r46, r3061;
}
{
add.f16x2 r3062, r3038, r3059;
}
{
cvt.rn.f16.f64 rs477, fd455;
}
mov.b32 r3067, {rs477, rs477};
{
mul.f16x2 r3065, r40, r3067;
}
{
add.f16x2 r3068, r3044, r3065;
}
{
cvt.rn.f16.f64 rs478, fd456;
}
mov.b32 r3073, {rs478, rs478};
{
mul.f16x2 r3071, r43, r3073;
}
{
add.f16x2 r3074, r3050, r3071;
}
{
cvt.rn.f16.f64 rs479, fd459;
}
mov.b32 r3079, {rs479, rs479};
{
mul.f16x2 r3077, r49, r3079;
}
{
add.f16x2 r3080, r3056, r3077;
}
{
cvt.rn.f16.f64 rs480, fd460;
}
mov.b32 r3085, {rs480, rs480};
{
mul.f16x2 r3083, r58, r3085;
}
{
add.f16x2 r3086, r3062, r3083;
}
{
cvt.rn.f16.f64 rs481, fd459;
}
mov.b32 r3091, {rs481, rs481};
{
mul.f16x2 r3089, r52, r3091;
}
{
add.f16x2 r3092, r3068, r3089;
}
{
cvt.rn.f16.f64 rs482, fd460;
}
mov.b32 r3097, {rs482, rs482};
{
mul.f16x2 r3095, r55, r3097;
}
{
add.f16x2 r3098, r3074, r3095;
}
{
cvt.rn.f16.f64 rs483, fd463;
}
mov.b32 r3103, {rs483, rs483};
{
mul.f16x2 r3101, r61, r3103;
}
{
add.f16x2 r3104, r3080, r3101;
}
{
cvt.rn.f16.f64 rs484, fd464;
}
mov.b32 r3109, {rs484, rs484};
{
mul.f16x2 r3107, r70, r3109;
}
{
add.f16x2 r3110, r3086, r3107;
}
{
cvt.rn.f16.f64 rs485, fd463;
}
mov.b32 r3115, {rs485, rs485};
{
mul.f16x2 r3113, r64, r3115;
}
{
add.f16x2 r3116, r3092, r3113;
}
{
cvt.rn.f16.f64 rs486, fd464;
}
mov.b32 r3121, {rs486, rs486};
{
mul.f16x2 r3119, r67, r3121;
}
{
add.f16x2 r3122, r3098, r3119;
}
{
cvt.rn.f16.f64 rs487, fd467;
}
mov.b32 r3127, {rs487, rs487};
{
mul.f16x2 r3125, r73, r3127;
}
{
add.f16x2 r3128, r3104, r3125;
}
{
cvt.rn.f16.f64 rs488, fd468;
}
mov.b32 r3133, {rs488, rs488};
{
mul.f16x2 r3131, r82, r3133;
}
{
add.f16x2 r3134, r3110, r3131;
}
{
cvt.rn.f16.f64 rs489, fd467;
}
mov.b32 r3139, {rs489, rs489};
{
mul.f16x2 r3137, r76, r3139;
}
{
add.f16x2 r3140, r3116, r3137;
}
{
cvt.rn.f16.f64 rs490, fd468;
}
mov.b32 r3145, {rs490, rs490};
{
mul.f16x2 r3143, r79, r3145;
}
{
add.f16x2 r3146, r3122, r3143;
}
{
cvt.rn.f16.f64 rs491, fd471;
}
mov.b32 r3151, {rs491, rs491};
{
mul.f16x2 r3149, r85, r3151;
}
{
add.f16x2 r3152, r3128, r3149;
}
{
cvt.rn.f16.f64 rs492, fd472;
}
mov.b32 r3157, {rs492, rs492};
{
mul.f16x2 r3155, r94, r3157;
}
{
add.f16x2 r3158, r3134, r3155;
}
{
cvt.rn.f16.f64 rs493, fd471;
}
mov.b32 r3163, {rs493, rs493};
{
mul.f16x2 r3161, r88, r3163;
}
{
add.f16x2 r3164, r3140, r3161;
}
{
cvt.rn.f16.f64 rs494, fd472;
}
mov.b32 r3169, {rs494, rs494};
{
mul.f16x2 r3167, r91, r3169;
}
{
add.f16x2 r3170, r3146, r3167;
}
{
cvt.rn.f16.f64 rs495, fd475;
}
mov.b32 r3175, {rs495, rs495};
{
mul.f16x2 r3173, r97, r3175;
}
{
add.f16x2 r3176, r3152, r3173;
}
{
cvt.rn.f16.f64 rs496, fd476;
}
mov.b32 r3181, {rs496, rs496};
{
mul.f16x2 r3179, r106, r3181;
}
{
add.f16x2 r3182, r3158, r3179;
}
{
cvt.rn.f16.f64 rs497, fd475;
}
mov.b32 r3187, {rs497, rs497};
{
mul.f16x2 r3185, r100, r3187;
}
{
add.f16x2 r3188, r3164, r3185;
}
{
cvt.rn.f16.f64 rs498, fd476;
}
mov.b32 r3193, {rs498, rs498};
{
mul.f16x2 r3191, r103, r3193;
}
{
add.f16x2 r3194, r3170, r3191;
}
{
cvt.rn.f16.f64 rs499, fd479;
}
mov.b32 r3199, {rs499, rs499};
{
mul.f16x2 r3197, r109, r3199;
}
{
add.f16x2 r3200, r3176, r3197;
}
{
cvt.rn.f16.f64 rs500, fd480;
}
mov.b32 r3205, {rs500, rs500};
{
mul.f16x2 r3203, r118, r3205;
}
{
add.f16x2 r3206, r3182, r3203;
}
{
cvt.rn.f16.f64 rs501, fd479;
}
mov.b32 r3211, {rs501, rs501};
{
mul.f16x2 r3209, r112, r3211;
}
{
add.f16x2 r3212, r3188, r3209;
}
{
cvt.rn.f16.f64 rs502, fd480;
}
mov.b32 r3217, {rs502, rs502};
{
mul.f16x2 r3215, r115, r3217;
}
{
add.f16x2 r3218, r3194, r3215;
}
{
cvt.rn.f16.f64 rs503, fd483;
}
mov.b32 r3223, {rs503, rs503};
{
mul.f16x2 r3221, r121, r3223;
}
{
add.f16x2 r3224, r3200, r3221;
}
{
cvt.rn.f16.f64 rs504, fd484;
}
mov.b32 r3229, {rs504, rs504};
{
mul.f16x2 r3227, r130, r3229;
}
{
add.f16x2 r3230, r3206, r3227;
}
{
cvt.rn.f16.f64 rs505, fd483;
}
mov.b32 r3235, {rs505, rs505};
{
mul.f16x2 r3233, r124, r3235;
}
{
add.f16x2 r3236, r3212, r3233;
}
{
cvt.rn.f16.f64 rs506, fd484;
}
mov.b32 r3241, {rs506, rs506};
{
mul.f16x2 r3239, r127, r3241;
}
{
add.f16x2 r3242, r3218, r3239;
}
{
sub.f16x2 %22, r3224, r3230;
}
{
add.f16x2 %23, r3236, r3242;
}
{
add.f16x2 %24, r3224, r3230;
}
{
sub.f16x2 %25, r3236, r3242;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[22].y)));
};


#endif
