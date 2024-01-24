#ifndef CUFFTDX_FFT_19_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_19_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<953, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<343>;
.reg .b32 r<2233>;
.reg .f64 fd<325>;
.reg .b64 rd<2>;
{
add.f16x2 r1, %40, %74;
}
{
add.f16x2 r4, %41, %75;
}
{
sub.f16x2 r7, %40, %74;
}
{
sub.f16x2 r10, %41, %75;
}
{
add.f16x2 r13, %42, %72;
}
{
add.f16x2 r16, %43, %73;
}
{
sub.f16x2 r19, %42, %72;
}
{
sub.f16x2 r22, %43, %73;
}
{
add.f16x2 r25, %44, %70;
}
{
add.f16x2 r28, %45, %71;
}
{
sub.f16x2 r31, %44, %70;
}
{
sub.f16x2 r34, %45, %71;
}
{
add.f16x2 r37, %46, %68;
}
{
add.f16x2 r40, %47, %69;
}
{
sub.f16x2 r43, %46, %68;
}
{
sub.f16x2 r46, %47, %69;
}
{
add.f16x2 r49, %48, %66;
}
{
add.f16x2 r52, %49, %67;
}
{
sub.f16x2 r55, %48, %66;
}
{
sub.f16x2 r58, %49, %67;
}
{
add.f16x2 r61, %50, %64;
}
{
add.f16x2 r64, %51, %65;
}
{
sub.f16x2 r67, %50, %64;
}
{
sub.f16x2 r70, %51, %65;
}
{
add.f16x2 r73, %52, %62;
}
{
add.f16x2 r76, %53, %63;
}
{
sub.f16x2 r79, %52, %62;
}
{
sub.f16x2 r82, %53, %63;
}
{
add.f16x2 r85, %54, %60;
}
{
add.f16x2 r88, %55, %61;
}
{
sub.f16x2 r91, %54, %60;
}
{
sub.f16x2 r94, %55, %61;
}
{
add.f16x2 r97, %56, %58;
}
{
add.f16x2 r100, %57, %59;
}
{
sub.f16x2 r103, %56, %58;
}
{
sub.f16x2 r106, %57, %59;
}
{
add.f16x2 r109, %38, r1;
}
{
add.f16x2 r112, %39, r4;
}
{
add.f16x2 r115, r109, r13;
}
{
add.f16x2 r118, r112, r16;
}
{
add.f16x2 r121, r115, r25;
}
{
add.f16x2 r124, r118, r28;
}
{
add.f16x2 r127, r121, r37;
}
{
add.f16x2 r130, r124, r40;
}
{
add.f16x2 r133, r127, r49;
}
{
add.f16x2 r136, r130, r52;
}
{
add.f16x2 r139, r133, r61;
}
{
add.f16x2 r142, r136, r64;
}
{
add.f16x2 r145, r139, r73;
}
{
add.f16x2 r148, r142, r76;
}
{
add.f16x2 r151, r145, r85;
}
{
add.f16x2 r154, r148, r88;
}
{
add.f16x2 %0, r151, r97;
}
{
add.f16x2 %1, r154, r100;
}
mov.u32 r2004, 0;
cvt.rn.f16.s32 rs1, r2004;
mov.b32 r175, {rs1, rs1};
cvt.rn.f16.s32 rs2, r2004;
mov.b32 r187, {rs2, rs2};
mov.f64 fd295, 0d3FEE442285231BE1;
{
cvt.rn.f16.f64 rs3, fd295;
}
mov.b32 r167, {rs3, rs3};
{
mul.f16x2 r165, r1, r167;
}
{
add.f16x2 r168, %38, r165;
}
mov.f64 fd160, 0d3FD4C7E04850CFAA;
{
cvt.rn.f16.f64 rs4, fd160;
}
mov.b32 r173, {rs4, rs4};
{
mul.f16x2 r171, r10, r173;
}
{
add.f16x2 r174, r175, r171;
}
{
cvt.rn.f16.f64 rs5, fd295;
}
mov.b32 r179, {rs5, rs5};
{
mul.f16x2 r177, r4, r179;
}
{
add.f16x2 r180, %39, r177;
}
{
cvt.rn.f16.f64 rs6, fd160;
}
mov.b32 r185, {rs6, rs6};
{
mul.f16x2 r183, r7, r185;
}
{
add.f16x2 r186, r187, r183;
}
mov.f64 fd303, 0d3FE940A398F9CD23;
{
cvt.rn.f16.f64 rs7, fd303;
}
mov.b32 r191, {rs7, rs7};
{
mul.f16x2 r189, r13, r191;
}
{
add.f16x2 r192, r168, r189;
}
mov.f64 fd272, 0d3FE3A7A16B394423;
{
cvt.rn.f16.f64 rs8, fd272;
}
mov.b32 r197, {rs8, rs8};
{
mul.f16x2 r195, r22, r197;
}
{
add.f16x2 r198, r174, r195;
}
{
cvt.rn.f16.f64 rs9, fd303;
}
mov.b32 r203, {rs9, rs9};
{
mul.f16x2 r201, r16, r203;
}
{
add.f16x2 r204, r180, r201;
}
{
cvt.rn.f16.f64 rs10, fd272;
}
mov.b32 r209, {rs10, rs10};
{
mul.f16x2 r207, r19, r209;
}
{
add.f16x2 r210, r186, r207;
}
mov.f64 fd311, 0d3FE180996C77C8CA;
{
cvt.rn.f16.f64 rs11, fd311;
}
mov.b32 r215, {rs11, rs11};
{
mul.f16x2 r213, r25, r215;
}
{
add.f16x2 r216, r192, r213;
}
mov.f64 fd76, 0d3FEACA115AAE3DE4;
{
cvt.rn.f16.f64 rs12, fd76;
}
mov.b32 r221, {rs12, rs12};
{
mul.f16x2 r219, r34, r221;
}
{
add.f16x2 r222, r198, r219;
}
{
cvt.rn.f16.f64 rs13, fd311;
}
mov.b32 r227, {rs13, rs13};
{
mul.f16x2 r225, r28, r227;
}
{
add.f16x2 r228, r204, r225;
}
{
cvt.rn.f16.f64 rs14, fd76;
}
mov.b32 r233, {rs14, rs14};
{
mul.f16x2 r231, r31, r233;
}
{
add.f16x2 r234, r210, r231;
}
mov.f64 fd319, 0d3FCF6C118574C83E;
{
cvt.rn.f16.f64 rs15, fd319;
}
mov.b32 r239, {rs15, rs15};
{
mul.f16x2 r237, r37, r239;
}
{
add.f16x2 r240, r216, r237;
}
mov.f64 fd240, 0d3FEF0553B4DE2E18;
{
cvt.rn.f16.f64 rs16, fd240;
}
mov.b32 r245, {rs16, rs16};
{
mul.f16x2 r243, r46, r245;
}
{
add.f16x2 r246, r222, r243;
}
{
cvt.rn.f16.f64 rs17, fd319;
}
mov.b32 r251, {rs17, rs17};
{
mul.f16x2 r249, r40, r251;
}
{
add.f16x2 r252, r228, r249;
}
{
cvt.rn.f16.f64 rs18, fd240;
}
mov.b32 r257, {rs18, rs18};
{
mul.f16x2 r255, r43, r257;
}
{
add.f16x2 r258, r234, r255;
}
mov.f64 fd323, 0dBFB523EB8420F5F5;
{
cvt.rn.f16.f64 rs19, fd323;
}
mov.b32 r263, {rs19, rs19};
{
mul.f16x2 r261, r49, r263;
}
{
add.f16x2 r264, r240, r261;
}
mov.f64 fd324, 0d3FEFE40529A542AA;
{
cvt.rn.f16.f64 rs20, fd324;
}
mov.b32 r269, {rs20, rs20};
{
mul.f16x2 r267, r58, r269;
}
{
add.f16x2 r270, r246, r267;
}
{
cvt.rn.f16.f64 rs21, fd323;
}
mov.b32 r275, {rs21, rs21};
{
mul.f16x2 r273, r52, r275;
}
{
add.f16x2 r276, r252, r273;
}
{
cvt.rn.f16.f64 rs22, fd324;
}
mov.b32 r281, {rs22, rs22};
{
mul.f16x2 r279, r55, r281;
}
{
add.f16x2 r282, r258, r279;
}
mov.f64 fd315, 0dBFD9B560B9F596EA;
{
cvt.rn.f16.f64 rs23, fd315;
}
mov.b32 r287, {rs23, rs23};
{
mul.f16x2 r285, r61, r287;
}
{
add.f16x2 r288, r264, r285;
}
mov.f64 fd316, 0d3FED4E03DD110B08;
{
cvt.rn.f16.f64 rs24, fd316;
}
mov.b32 r293, {rs24, rs24};
{
mul.f16x2 r291, r70, r293;
}
{
add.f16x2 r294, r270, r291;
}
{
cvt.rn.f16.f64 rs25, fd315;
}
mov.b32 r299, {rs25, rs25};
{
mul.f16x2 r297, r64, r299;
}
{
add.f16x2 r300, r276, r297;
}
{
cvt.rn.f16.f64 rs26, fd316;
}
mov.b32 r305, {rs26, rs26};
{
mul.f16x2 r303, r67, r305;
}
{
add.f16x2 r306, r282, r303;
}
mov.f64 fd307, 0dBFE5AC4A670A1CFF;
{
cvt.rn.f16.f64 rs27, fd307;
}
mov.b32 r311, {rs27, rs27};
{
mul.f16x2 r309, r73, r311;
}
{
add.f16x2 r312, r288, r309;
}
mov.f64 fd308, 0d3FE78B0CDEE73E0F;
{
cvt.rn.f16.f64 rs28, fd308;
}
mov.b32 r317, {rs28, rs28};
{
mul.f16x2 r315, r82, r317;
}
{
add.f16x2 r318, r294, r315;
}
{
cvt.rn.f16.f64 rs29, fd307;
}
mov.b32 r323, {rs29, rs29};
{
mul.f16x2 r321, r76, r323;
}
{
add.f16x2 r324, r300, r321;
}
{
cvt.rn.f16.f64 rs30, fd308;
}
mov.b32 r329, {rs30, rs30};
{
mul.f16x2 r327, r79, r329;
}
{
add.f16x2 r330, r306, r327;
}
mov.f64 fd299, 0dBFEC24A622E3E9F9;
{
cvt.rn.f16.f64 rs31, fd299;
}
mov.b32 r335, {rs31, rs31};
{
mul.f16x2 r333, r85, r335;
}
{
add.f16x2 r336, r312, r333;
}
mov.f64 fd300, 0d3FDE75EC0DED7BEE;
{
cvt.rn.f16.f64 rs32, fd300;
}
mov.b32 r341, {rs32, rs32};
{
mul.f16x2 r339, r94, r341;
}
{
add.f16x2 r342, r318, r339;
}
{
cvt.rn.f16.f64 rs33, fd299;
}
mov.b32 r347, {rs33, rs33};
{
mul.f16x2 r345, r88, r347;
}
{
add.f16x2 r348, r324, r345;
}
{
cvt.rn.f16.f64 rs34, fd300;
}
mov.b32 r353, {rs34, rs34};
{
mul.f16x2 r351, r91, r353;
}
{
add.f16x2 r354, r330, r351;
}
mov.f64 fd291, 0dBFEF90459484F2B2;
{
cvt.rn.f16.f64 rs35, fd291;
}
mov.b32 r359, {rs35, rs35};
{
mul.f16x2 r357, r97, r359;
}
{
add.f16x2 r360, r336, r357;
}
mov.f64 fd292, 0d3FC5116F7F2D58C5;
{
cvt.rn.f16.f64 rs36, fd292;
}
mov.b32 r365, {rs36, rs36};
{
mul.f16x2 r363, r106, r365;
}
{
add.f16x2 r366, r342, r363;
}
{
cvt.rn.f16.f64 rs37, fd291;
}
mov.b32 r371, {rs37, rs37};
{
mul.f16x2 r369, r100, r371;
}
{
add.f16x2 r372, r348, r369;
}
{
cvt.rn.f16.f64 rs38, fd292;
}
mov.b32 r377, {rs38, rs38};
{
mul.f16x2 r375, r103, r377;
}
{
add.f16x2 r378, r354, r375;
}
{
sub.f16x2 %2, r360, r366;
}
{
add.f16x2 %3, r372, r378;
}
{
add.f16x2 %36, r360, r366;
}
{
sub.f16x2 %37, r372, r378;
}
cvt.rn.f16.s32 rs39, r2004;
mov.b32 r405, {rs39, rs39};
cvt.rn.f16.s32 rs40, r2004;
mov.b32 r417, {rs40, rs40};
{
cvt.rn.f16.f64 rs41, fd303;
}
mov.b32 r397, {rs41, rs41};
{
mul.f16x2 r395, r1, r397;
}
{
add.f16x2 r398, %38, r395;
}
{
cvt.rn.f16.f64 rs42, fd272;
}
mov.b32 r403, {rs42, rs42};
{
mul.f16x2 r401, r10, r403;
}
{
add.f16x2 r404, r405, r401;
}
{
cvt.rn.f16.f64 rs43, fd303;
}
mov.b32 r409, {rs43, rs43};
{
mul.f16x2 r407, r4, r409;
}
{
add.f16x2 r410, %39, r407;
}
{
cvt.rn.f16.f64 rs44, fd272;
}
mov.b32 r415, {rs44, rs44};
{
mul.f16x2 r413, r7, r415;
}
{
add.f16x2 r416, r417, r413;
}
{
cvt.rn.f16.f64 rs45, fd319;
}
mov.b32 r421, {rs45, rs45};
{
mul.f16x2 r419, r13, r421;
}
{
add.f16x2 r422, r398, r419;
}
{
cvt.rn.f16.f64 rs46, fd240;
}
mov.b32 r427, {rs46, rs46};
{
mul.f16x2 r425, r22, r427;
}
{
add.f16x2 r428, r404, r425;
}
{
cvt.rn.f16.f64 rs47, fd319;
}
mov.b32 r433, {rs47, rs47};
{
mul.f16x2 r431, r16, r433;
}
{
add.f16x2 r434, r410, r431;
}
{
cvt.rn.f16.f64 rs48, fd240;
}
mov.b32 r439, {rs48, rs48};
{
mul.f16x2 r437, r19, r439;
}
{
add.f16x2 r440, r416, r437;
}
{
cvt.rn.f16.f64 rs49, fd315;
}
mov.b32 r445, {rs49, rs49};
{
mul.f16x2 r443, r25, r445;
}
{
add.f16x2 r446, r422, r443;
}
{
cvt.rn.f16.f64 rs50, fd316;
}
mov.b32 r451, {rs50, rs50};
{
mul.f16x2 r449, r34, r451;
}
{
add.f16x2 r452, r428, r449;
}
{
cvt.rn.f16.f64 rs51, fd315;
}
mov.b32 r457, {rs51, rs51};
{
mul.f16x2 r455, r28, r457;
}
{
add.f16x2 r458, r434, r455;
}
{
cvt.rn.f16.f64 rs52, fd316;
}
mov.b32 r463, {rs52, rs52};
{
mul.f16x2 r461, r31, r463;
}
{
add.f16x2 r464, r440, r461;
}
{
cvt.rn.f16.f64 rs53, fd299;
}
mov.b32 r469, {rs53, rs53};
{
mul.f16x2 r467, r37, r469;
}
{
add.f16x2 r470, r446, r467;
}
{
cvt.rn.f16.f64 rs54, fd300;
}
mov.b32 r475, {rs54, rs54};
{
mul.f16x2 r473, r46, r475;
}
{
add.f16x2 r476, r452, r473;
}
{
cvt.rn.f16.f64 rs55, fd299;
}
mov.b32 r481, {rs55, rs55};
{
mul.f16x2 r479, r40, r481;
}
{
add.f16x2 r482, r458, r479;
}
{
cvt.rn.f16.f64 rs56, fd300;
}
mov.b32 r487, {rs56, rs56};
{
mul.f16x2 r485, r43, r487;
}
{
add.f16x2 r488, r464, r485;
}
{
cvt.rn.f16.f64 rs57, fd291;
}
mov.b32 r493, {rs57, rs57};
{
mul.f16x2 r491, r49, r493;
}
{
add.f16x2 r494, r470, r491;
}
mov.f64 fd276, 0dBFC5116F7F2D58C5;
{
cvt.rn.f16.f64 rs58, fd276;
}
mov.b32 r499, {rs58, rs58};
{
mul.f16x2 r497, r58, r499;
}
{
add.f16x2 r500, r476, r497;
}
{
cvt.rn.f16.f64 rs59, fd291;
}
mov.b32 r505, {rs59, rs59};
{
mul.f16x2 r503, r52, r505;
}
{
add.f16x2 r506, r482, r503;
}
{
cvt.rn.f16.f64 rs60, fd276;
}
mov.b32 r511, {rs60, rs60};
{
mul.f16x2 r509, r55, r511;
}
{
add.f16x2 r512, r488, r509;
}
{
cvt.rn.f16.f64 rs61, fd307;
}
mov.b32 r517, {rs61, rs61};
{
mul.f16x2 r515, r61, r517;
}
{
add.f16x2 r518, r494, r515;
}
mov.f64 fd188, 0dBFE78B0CDEE73E0F;
{
cvt.rn.f16.f64 rs62, fd188;
}
mov.b32 r523, {rs62, rs62};
{
mul.f16x2 r521, r70, r523;
}
{
add.f16x2 r524, r500, r521;
}
{
cvt.rn.f16.f64 rs63, fd307;
}
mov.b32 r529, {rs63, rs63};
{
mul.f16x2 r527, r64, r529;
}
{
add.f16x2 r530, r506, r527;
}
{
cvt.rn.f16.f64 rs64, fd188;
}
mov.b32 r535, {rs64, rs64};
{
mul.f16x2 r533, r67, r535;
}
{
add.f16x2 r536, r512, r533;
}
{
cvt.rn.f16.f64 rs65, fd323;
}
mov.b32 r541, {rs65, rs65};
{
mul.f16x2 r539, r73, r541;
}
{
add.f16x2 r542, r518, r539;
}
mov.f64 fd224, 0dBFEFE40529A542AA;
{
cvt.rn.f16.f64 rs66, fd224;
}
mov.b32 r547, {rs66, rs66};
{
mul.f16x2 r545, r82, r547;
}
{
add.f16x2 r548, r524, r545;
}
{
cvt.rn.f16.f64 rs67, fd323;
}
mov.b32 r553, {rs67, rs67};
{
mul.f16x2 r551, r76, r553;
}
{
add.f16x2 r554, r530, r551;
}
{
cvt.rn.f16.f64 rs68, fd224;
}
mov.b32 r559, {rs68, rs68};
{
mul.f16x2 r557, r79, r559;
}
{
add.f16x2 r560, r536, r557;
}
{
cvt.rn.f16.f64 rs69, fd311;
}
mov.b32 r565, {rs69, rs69};
{
mul.f16x2 r563, r85, r565;
}
{
add.f16x2 r566, r542, r563;
}
mov.f64 fd312, 0dBFEACA115AAE3DE4;
{
cvt.rn.f16.f64 rs70, fd312;
}
mov.b32 r571, {rs70, rs70};
{
mul.f16x2 r569, r94, r571;
}
{
add.f16x2 r572, r548, r569;
}
{
cvt.rn.f16.f64 rs71, fd311;
}
mov.b32 r577, {rs71, rs71};
{
mul.f16x2 r575, r88, r577;
}
{
add.f16x2 r578, r554, r575;
}
{
cvt.rn.f16.f64 rs72, fd312;
}
mov.b32 r583, {rs72, rs72};
{
mul.f16x2 r581, r91, r583;
}
{
add.f16x2 r584, r560, r581;
}
{
cvt.rn.f16.f64 rs73, fd295;
}
mov.b32 r589, {rs73, rs73};
{
mul.f16x2 r587, r97, r589;
}
{
add.f16x2 r590, r566, r587;
}
mov.f64 fd296, 0dBFD4C7E04850CFAA;
{
cvt.rn.f16.f64 rs74, fd296;
}
mov.b32 r595, {rs74, rs74};
{
mul.f16x2 r593, r106, r595;
}
{
add.f16x2 r596, r572, r593;
}
{
cvt.rn.f16.f64 rs75, fd295;
}
mov.b32 r601, {rs75, rs75};
{
mul.f16x2 r599, r100, r601;
}
{
add.f16x2 r602, r578, r599;
}
{
cvt.rn.f16.f64 rs76, fd296;
}
mov.b32 r607, {rs76, rs76};
{
mul.f16x2 r605, r103, r607;
}
{
add.f16x2 r608, r584, r605;
}
{
sub.f16x2 %4, r590, r596;
}
{
add.f16x2 %5, r602, r608;
}
{
add.f16x2 %34, r590, r596;
}
{
sub.f16x2 %35, r602, r608;
}
cvt.rn.f16.s32 rs77, r2004;
mov.b32 r635, {rs77, rs77};
cvt.rn.f16.s32 rs78, r2004;
mov.b32 r647, {rs78, rs78};
{
cvt.rn.f16.f64 rs79, fd311;
}
mov.b32 r627, {rs79, rs79};
{
mul.f16x2 r625, r1, r627;
}
{
add.f16x2 r628, %38, r625;
}
{
cvt.rn.f16.f64 rs80, fd76;
}
mov.b32 r633, {rs80, rs80};
{
mul.f16x2 r631, r10, r633;
}
{
add.f16x2 r634, r635, r631;
}
{
cvt.rn.f16.f64 rs81, fd311;
}
mov.b32 r639, {rs81, rs81};
{
mul.f16x2 r637, r4, r639;
}
{
add.f16x2 r640, %39, r637;
}
{
cvt.rn.f16.f64 rs82, fd76;
}
mov.b32 r645, {rs82, rs82};
{
mul.f16x2 r643, r7, r645;
}
{
add.f16x2 r646, r647, r643;
}
{
cvt.rn.f16.f64 rs83, fd315;
}
mov.b32 r651, {rs83, rs83};
{
mul.f16x2 r649, r13, r651;
}
{
add.f16x2 r652, r628, r649;
}
{
cvt.rn.f16.f64 rs84, fd316;
}
mov.b32 r657, {rs84, rs84};
{
mul.f16x2 r655, r22, r657;
}
{
add.f16x2 r658, r634, r655;
}
{
cvt.rn.f16.f64 rs85, fd315;
}
mov.b32 r663, {rs85, rs85};
{
mul.f16x2 r661, r16, r663;
}
{
add.f16x2 r664, r640, r661;
}
{
cvt.rn.f16.f64 rs86, fd316;
}
mov.b32 r669, {rs86, rs86};
{
mul.f16x2 r667, r19, r669;
}
{
add.f16x2 r670, r646, r667;
}
{
cvt.rn.f16.f64 rs87, fd291;
}
mov.b32 r675, {rs87, rs87};
{
mul.f16x2 r673, r25, r675;
}
{
add.f16x2 r676, r652, r673;
}
{
cvt.rn.f16.f64 rs88, fd292;
}
mov.b32 r681, {rs88, rs88};
{
mul.f16x2 r679, r34, r681;
}
{
add.f16x2 r682, r658, r679;
}
{
cvt.rn.f16.f64 rs89, fd291;
}
mov.b32 r687, {rs89, rs89};
{
mul.f16x2 r685, r28, r687;
}
{
add.f16x2 r688, r664, r685;
}
{
cvt.rn.f16.f64 rs90, fd292;
}
mov.b32 r693, {rs90, rs90};
{
mul.f16x2 r691, r31, r693;
}
{
add.f16x2 r694, r670, r691;
}
{
cvt.rn.f16.f64 rs91, fd307;
}
mov.b32 r699, {rs91, rs91};
{
mul.f16x2 r697, r37, r699;
}
{
add.f16x2 r700, r676, r697;
}
{
cvt.rn.f16.f64 rs92, fd188;
}
mov.b32 r705, {rs92, rs92};
{
mul.f16x2 r703, r46, r705;
}
{
add.f16x2 r706, r682, r703;
}
{
cvt.rn.f16.f64 rs93, fd307;
}
mov.b32 r711, {rs93, rs93};
{
mul.f16x2 r709, r40, r711;
}
{
add.f16x2 r712, r688, r709;
}
{
cvt.rn.f16.f64 rs94, fd188;
}
mov.b32 r717, {rs94, rs94};
{
mul.f16x2 r715, r43, r717;
}
{
add.f16x2 r718, r694, r715;
}
{
cvt.rn.f16.f64 rs95, fd319;
}
mov.b32 r723, {rs95, rs95};
{
mul.f16x2 r721, r49, r723;
}
{
add.f16x2 r724, r700, r721;
}
mov.f64 fd320, 0dBFEF0553B4DE2E18;
{
cvt.rn.f16.f64 rs96, fd320;
}
mov.b32 r729, {rs96, rs96};
{
mul.f16x2 r727, r58, r729;
}
{
add.f16x2 r730, r706, r727;
}
{
cvt.rn.f16.f64 rs97, fd319;
}
mov.b32 r735, {rs97, rs97};
{
mul.f16x2 r733, r52, r735;
}
{
add.f16x2 r736, r712, r733;
}
{
cvt.rn.f16.f64 rs98, fd320;
}
mov.b32 r741, {rs98, rs98};
{
mul.f16x2 r739, r55, r741;
}
{
add.f16x2 r742, r718, r739;
}
{
cvt.rn.f16.f64 rs99, fd295;
}
mov.b32 r747, {rs99, rs99};
{
mul.f16x2 r745, r61, r747;
}
{
add.f16x2 r748, r724, r745;
}
{
cvt.rn.f16.f64 rs100, fd296;
}
mov.b32 r753, {rs100, rs100};
{
mul.f16x2 r751, r70, r753;
}
{
add.f16x2 r754, r730, r751;
}
{
cvt.rn.f16.f64 rs101, fd295;
}
mov.b32 r759, {rs101, rs101};
{
mul.f16x2 r757, r64, r759;
}
{
add.f16x2 r760, r736, r757;
}
{
cvt.rn.f16.f64 rs102, fd296;
}
mov.b32 r765, {rs102, rs102};
{
mul.f16x2 r763, r67, r765;
}
{
add.f16x2 r766, r742, r763;
}
{
cvt.rn.f16.f64 rs103, fd303;
}
mov.b32 r771, {rs103, rs103};
{
mul.f16x2 r769, r73, r771;
}
{
add.f16x2 r772, r748, r769;
}
{
cvt.rn.f16.f64 rs104, fd272;
}
mov.b32 r777, {rs104, rs104};
{
mul.f16x2 r775, r82, r777;
}
{
add.f16x2 r778, r754, r775;
}
{
cvt.rn.f16.f64 rs105, fd303;
}
mov.b32 r783, {rs105, rs105};
{
mul.f16x2 r781, r76, r783;
}
{
add.f16x2 r784, r760, r781;
}
{
cvt.rn.f16.f64 rs106, fd272;
}
mov.b32 r789, {rs106, rs106};
{
mul.f16x2 r787, r79, r789;
}
{
add.f16x2 r790, r766, r787;
}
{
cvt.rn.f16.f64 rs107, fd323;
}
mov.b32 r795, {rs107, rs107};
{
mul.f16x2 r793, r85, r795;
}
{
add.f16x2 r796, r772, r793;
}
{
cvt.rn.f16.f64 rs108, fd324;
}
mov.b32 r801, {rs108, rs108};
{
mul.f16x2 r799, r94, r801;
}
{
add.f16x2 r802, r778, r799;
}
{
cvt.rn.f16.f64 rs109, fd323;
}
mov.b32 r807, {rs109, rs109};
{
mul.f16x2 r805, r88, r807;
}
{
add.f16x2 r808, r784, r805;
}
{
cvt.rn.f16.f64 rs110, fd324;
}
mov.b32 r813, {rs110, rs110};
{
mul.f16x2 r811, r91, r813;
}
{
add.f16x2 r814, r790, r811;
}
{
cvt.rn.f16.f64 rs111, fd299;
}
mov.b32 r819, {rs111, rs111};
{
mul.f16x2 r817, r97, r819;
}
{
add.f16x2 r820, r796, r817;
}
{
cvt.rn.f16.f64 rs112, fd300;
}
mov.b32 r825, {rs112, rs112};
{
mul.f16x2 r823, r106, r825;
}
{
add.f16x2 r826, r802, r823;
}
{
cvt.rn.f16.f64 rs113, fd299;
}
mov.b32 r831, {rs113, rs113};
{
mul.f16x2 r829, r100, r831;
}
{
add.f16x2 r832, r808, r829;
}
{
cvt.rn.f16.f64 rs114, fd300;
}
mov.b32 r837, {rs114, rs114};
{
mul.f16x2 r835, r103, r837;
}
{
add.f16x2 r838, r814, r835;
}
{
sub.f16x2 %6, r820, r826;
}
{
add.f16x2 %7, r832, r838;
}
{
add.f16x2 %32, r820, r826;
}
{
sub.f16x2 %33, r832, r838;
}
cvt.rn.f16.s32 rs115, r2004;
mov.b32 r865, {rs115, rs115};
cvt.rn.f16.s32 rs116, r2004;
mov.b32 r877, {rs116, rs116};
{
cvt.rn.f16.f64 rs117, fd319;
}
mov.b32 r857, {rs117, rs117};
{
mul.f16x2 r855, r1, r857;
}
{
add.f16x2 r858, %38, r855;
}
{
cvt.rn.f16.f64 rs118, fd240;
}
mov.b32 r863, {rs118, rs118};
{
mul.f16x2 r861, r10, r863;
}
{
add.f16x2 r864, r865, r861;
}
{
cvt.rn.f16.f64 rs119, fd319;
}
mov.b32 r869, {rs119, rs119};
{
mul.f16x2 r867, r4, r869;
}
{
add.f16x2 r870, %39, r867;
}
{
cvt.rn.f16.f64 rs120, fd240;
}
mov.b32 r875, {rs120, rs120};
{
mul.f16x2 r873, r7, r875;
}
{
add.f16x2 r876, r877, r873;
}
{
cvt.rn.f16.f64 rs121, fd299;
}
mov.b32 r881, {rs121, rs121};
{
mul.f16x2 r879, r13, r881;
}
{
add.f16x2 r882, r858, r879;
}
{
cvt.rn.f16.f64 rs122, fd300;
}
mov.b32 r887, {rs122, rs122};
{
mul.f16x2 r885, r22, r887;
}
{
add.f16x2 r888, r864, r885;
}
{
cvt.rn.f16.f64 rs123, fd299;
}
mov.b32 r893, {rs123, rs123};
{
mul.f16x2 r891, r16, r893;
}
{
add.f16x2 r894, r870, r891;
}
{
cvt.rn.f16.f64 rs124, fd300;
}
mov.b32 r899, {rs124, rs124};
{
mul.f16x2 r897, r19, r899;
}
{
add.f16x2 r900, r876, r897;
}
{
cvt.rn.f16.f64 rs125, fd307;
}
mov.b32 r905, {rs125, rs125};
{
mul.f16x2 r903, r25, r905;
}
{
add.f16x2 r906, r882, r903;
}
{
cvt.rn.f16.f64 rs126, fd188;
}
mov.b32 r911, {rs126, rs126};
{
mul.f16x2 r909, r34, r911;
}
{
add.f16x2 r912, r888, r909;
}
{
cvt.rn.f16.f64 rs127, fd307;
}
mov.b32 r917, {rs127, rs127};
{
mul.f16x2 r915, r28, r917;
}
{
add.f16x2 r918, r894, r915;
}
{
cvt.rn.f16.f64 rs128, fd188;
}
mov.b32 r923, {rs128, rs128};
{
mul.f16x2 r921, r31, r923;
}
{
add.f16x2 r924, r900, r921;
}
{
cvt.rn.f16.f64 rs129, fd311;
}
mov.b32 r929, {rs129, rs129};
{
mul.f16x2 r927, r37, r929;
}
{
add.f16x2 r930, r906, r927;
}
{
cvt.rn.f16.f64 rs130, fd312;
}
mov.b32 r935, {rs130, rs130};
{
mul.f16x2 r933, r46, r935;
}
{
add.f16x2 r936, r912, r933;
}
{
cvt.rn.f16.f64 rs131, fd311;
}
mov.b32 r941, {rs131, rs131};
{
mul.f16x2 r939, r40, r941;
}
{
add.f16x2 r942, r918, r939;
}
{
cvt.rn.f16.f64 rs132, fd312;
}
mov.b32 r947, {rs132, rs132};
{
mul.f16x2 r945, r43, r947;
}
{
add.f16x2 r948, r924, r945;
}
{
cvt.rn.f16.f64 rs133, fd295;
}
mov.b32 r953, {rs133, rs133};
{
mul.f16x2 r951, r49, r953;
}
{
add.f16x2 r954, r930, r951;
}
{
cvt.rn.f16.f64 rs134, fd160;
}
mov.b32 r959, {rs134, rs134};
{
mul.f16x2 r957, r58, r959;
}
{
add.f16x2 r960, r936, r957;
}
{
cvt.rn.f16.f64 rs135, fd295;
}
mov.b32 r965, {rs135, rs135};
{
mul.f16x2 r963, r52, r965;
}
{
add.f16x2 r966, r942, r963;
}
{
cvt.rn.f16.f64 rs136, fd160;
}
mov.b32 r971, {rs136, rs136};
{
mul.f16x2 r969, r55, r971;
}
{
add.f16x2 r972, r948, r969;
}
{
cvt.rn.f16.f64 rs137, fd323;
}
mov.b32 r977, {rs137, rs137};
{
mul.f16x2 r975, r61, r977;
}
{
add.f16x2 r978, r954, r975;
}
{
cvt.rn.f16.f64 rs138, fd324;
}
mov.b32 r983, {rs138, rs138};
{
mul.f16x2 r981, r70, r983;
}
{
add.f16x2 r984, r960, r981;
}
{
cvt.rn.f16.f64 rs139, fd323;
}
mov.b32 r989, {rs139, rs139};
{
mul.f16x2 r987, r64, r989;
}
{
add.f16x2 r990, r966, r987;
}
{
cvt.rn.f16.f64 rs140, fd324;
}
mov.b32 r995, {rs140, rs140};
{
mul.f16x2 r993, r67, r995;
}
{
add.f16x2 r996, r972, r993;
}
{
cvt.rn.f16.f64 rs141, fd291;
}
mov.b32 r1001, {rs141, rs141};
{
mul.f16x2 r999, r73, r1001;
}
{
add.f16x2 r1002, r978, r999;
}
{
cvt.rn.f16.f64 rs142, fd292;
}
mov.b32 r1007, {rs142, rs142};
{
mul.f16x2 r1005, r82, r1007;
}
{
add.f16x2 r1008, r984, r1005;
}
{
cvt.rn.f16.f64 rs143, fd291;
}
mov.b32 r1013, {rs143, rs143};
{
mul.f16x2 r1011, r76, r1013;
}
{
add.f16x2 r1014, r990, r1011;
}
{
cvt.rn.f16.f64 rs144, fd292;
}
mov.b32 r1019, {rs144, rs144};
{
mul.f16x2 r1017, r79, r1019;
}
{
add.f16x2 r1020, r996, r1017;
}
{
cvt.rn.f16.f64 rs145, fd315;
}
mov.b32 r1025, {rs145, rs145};
{
mul.f16x2 r1023, r85, r1025;
}
{
add.f16x2 r1026, r1002, r1023;
}
mov.f64 fd268, 0dBFED4E03DD110B08;
{
cvt.rn.f16.f64 rs146, fd268;
}
mov.b32 r1031, {rs146, rs146};
{
mul.f16x2 r1029, r94, r1031;
}
{
add.f16x2 r1032, r1008, r1029;
}
{
cvt.rn.f16.f64 rs147, fd315;
}
mov.b32 r1037, {rs147, rs147};
{
mul.f16x2 r1035, r88, r1037;
}
{
add.f16x2 r1038, r1014, r1035;
}
{
cvt.rn.f16.f64 rs148, fd268;
}
mov.b32 r1043, {rs148, rs148};
{
mul.f16x2 r1041, r91, r1043;
}
{
add.f16x2 r1044, r1020, r1041;
}
{
cvt.rn.f16.f64 rs149, fd303;
}
mov.b32 r1049, {rs149, rs149};
{
mul.f16x2 r1047, r97, r1049;
}
{
add.f16x2 r1050, r1026, r1047;
}
mov.f64 fd304, 0dBFE3A7A16B394423;
{
cvt.rn.f16.f64 rs150, fd304;
}
mov.b32 r1055, {rs150, rs150};
{
mul.f16x2 r1053, r106, r1055;
}
{
add.f16x2 r1056, r1032, r1053;
}
{
cvt.rn.f16.f64 rs151, fd303;
}
mov.b32 r1061, {rs151, rs151};
{
mul.f16x2 r1059, r100, r1061;
}
{
add.f16x2 r1062, r1038, r1059;
}
{
cvt.rn.f16.f64 rs152, fd304;
}
mov.b32 r1067, {rs152, rs152};
{
mul.f16x2 r1065, r103, r1067;
}
{
add.f16x2 r1068, r1044, r1065;
}
{
sub.f16x2 %8, r1050, r1056;
}
{
add.f16x2 %9, r1062, r1068;
}
{
add.f16x2 %30, r1050, r1056;
}
{
sub.f16x2 %31, r1062, r1068;
}
cvt.rn.f16.s32 rs153, r2004;
mov.b32 r1095, {rs153, rs153};
cvt.rn.f16.s32 rs154, r2004;
mov.b32 r1107, {rs154, rs154};
{
cvt.rn.f16.f64 rs155, fd323;
}
mov.b32 r1087, {rs155, rs155};
{
mul.f16x2 r1085, r1, r1087;
}
{
add.f16x2 r1088, %38, r1085;
}
{
cvt.rn.f16.f64 rs156, fd324;
}
mov.b32 r1093, {rs156, rs156};
{
mul.f16x2 r1091, r10, r1093;
}
{
add.f16x2 r1094, r1095, r1091;
}
{
cvt.rn.f16.f64 rs157, fd323;
}
mov.b32 r1099, {rs157, rs157};
{
mul.f16x2 r1097, r4, r1099;
}
{
add.f16x2 r1100, %39, r1097;
}
{
cvt.rn.f16.f64 rs158, fd324;
}
mov.b32 r1105, {rs158, rs158};
{
mul.f16x2 r1103, r7, r1105;
}
{
add.f16x2 r1106, r1107, r1103;
}
{
cvt.rn.f16.f64 rs159, fd291;
}
mov.b32 r1111, {rs159, rs159};
{
mul.f16x2 r1109, r13, r1111;
}
{
add.f16x2 r1112, r1088, r1109;
}
{
cvt.rn.f16.f64 rs160, fd276;
}
mov.b32 r1117, {rs160, rs160};
{
mul.f16x2 r1115, r22, r1117;
}
{
add.f16x2 r1118, r1094, r1115;
}
{
cvt.rn.f16.f64 rs161, fd291;
}
mov.b32 r1123, {rs161, rs161};
{
mul.f16x2 r1121, r16, r1123;
}
{
add.f16x2 r1124, r1100, r1121;
}
{
cvt.rn.f16.f64 rs162, fd276;
}
mov.b32 r1129, {rs162, rs162};
{
mul.f16x2 r1127, r19, r1129;
}
{
add.f16x2 r1130, r1106, r1127;
}
{
cvt.rn.f16.f64 rs163, fd319;
}
mov.b32 r1135, {rs163, rs163};
{
mul.f16x2 r1133, r25, r1135;
}
{
add.f16x2 r1136, r1112, r1133;
}
{
cvt.rn.f16.f64 rs164, fd320;
}
mov.b32 r1141, {rs164, rs164};
{
mul.f16x2 r1139, r34, r1141;
}
{
add.f16x2 r1142, r1118, r1139;
}
{
cvt.rn.f16.f64 rs165, fd319;
}
mov.b32 r1147, {rs165, rs165};
{
mul.f16x2 r1145, r28, r1147;
}
{
add.f16x2 r1148, r1124, r1145;
}
{
cvt.rn.f16.f64 rs166, fd320;
}
mov.b32 r1153, {rs166, rs166};
{
mul.f16x2 r1151, r31, r1153;
}
{
add.f16x2 r1154, r1130, r1151;
}
{
cvt.rn.f16.f64 rs167, fd295;
}
mov.b32 r1159, {rs167, rs167};
{
mul.f16x2 r1157, r37, r1159;
}
{
add.f16x2 r1160, r1136, r1157;
}
{
cvt.rn.f16.f64 rs168, fd160;
}
mov.b32 r1165, {rs168, rs168};
{
mul.f16x2 r1163, r46, r1165;
}
{
add.f16x2 r1166, r1142, r1163;
}
{
cvt.rn.f16.f64 rs169, fd295;
}
mov.b32 r1171, {rs169, rs169};
{
mul.f16x2 r1169, r40, r1171;
}
{
add.f16x2 r1172, r1148, r1169;
}
{
cvt.rn.f16.f64 rs170, fd160;
}
mov.b32 r1177, {rs170, rs170};
{
mul.f16x2 r1175, r43, r1177;
}
{
add.f16x2 r1178, r1154, r1175;
}
{
cvt.rn.f16.f64 rs171, fd315;
}
mov.b32 r1183, {rs171, rs171};
{
mul.f16x2 r1181, r49, r1183;
}
{
add.f16x2 r1184, r1160, r1181;
}
{
cvt.rn.f16.f64 rs172, fd316;
}
mov.b32 r1189, {rs172, rs172};
{
mul.f16x2 r1187, r58, r1189;
}
{
add.f16x2 r1190, r1166, r1187;
}
{
cvt.rn.f16.f64 rs173, fd315;
}
mov.b32 r1195, {rs173, rs173};
{
mul.f16x2 r1193, r52, r1195;
}
{
add.f16x2 r1196, r1172, r1193;
}
{
cvt.rn.f16.f64 rs174, fd316;
}
mov.b32 r1201, {rs174, rs174};
{
mul.f16x2 r1199, r55, r1201;
}
{
add.f16x2 r1202, r1178, r1199;
}
{
cvt.rn.f16.f64 rs175, fd299;
}
mov.b32 r1207, {rs175, rs175};
{
mul.f16x2 r1205, r61, r1207;
}
{
add.f16x2 r1208, r1184, r1205;
}
mov.f64 fd244, 0dBFDE75EC0DED7BEE;
{
cvt.rn.f16.f64 rs176, fd244;
}
mov.b32 r1213, {rs176, rs176};
{
mul.f16x2 r1211, r70, r1213;
}
{
add.f16x2 r1214, r1190, r1211;
}
{
cvt.rn.f16.f64 rs177, fd299;
}
mov.b32 r1219, {rs177, rs177};
{
mul.f16x2 r1217, r64, r1219;
}
{
add.f16x2 r1220, r1196, r1217;
}
{
cvt.rn.f16.f64 rs178, fd244;
}
mov.b32 r1225, {rs178, rs178};
{
mul.f16x2 r1223, r67, r1225;
}
{
add.f16x2 r1226, r1202, r1223;
}
{
cvt.rn.f16.f64 rs179, fd311;
}
mov.b32 r1231, {rs179, rs179};
{
mul.f16x2 r1229, r73, r1231;
}
{
add.f16x2 r1232, r1208, r1229;
}
{
cvt.rn.f16.f64 rs180, fd312;
}
mov.b32 r1237, {rs180, rs180};
{
mul.f16x2 r1235, r82, r1237;
}
{
add.f16x2 r1238, r1214, r1235;
}
{
cvt.rn.f16.f64 rs181, fd311;
}
mov.b32 r1243, {rs181, rs181};
{
mul.f16x2 r1241, r76, r1243;
}
{
add.f16x2 r1244, r1220, r1241;
}
{
cvt.rn.f16.f64 rs182, fd312;
}
mov.b32 r1249, {rs182, rs182};
{
mul.f16x2 r1247, r79, r1249;
}
{
add.f16x2 r1250, r1226, r1247;
}
{
cvt.rn.f16.f64 rs183, fd303;
}
mov.b32 r1255, {rs183, rs183};
{
mul.f16x2 r1253, r85, r1255;
}
{
add.f16x2 r1256, r1232, r1253;
}
{
cvt.rn.f16.f64 rs184, fd272;
}
mov.b32 r1261, {rs184, rs184};
{
mul.f16x2 r1259, r94, r1261;
}
{
add.f16x2 r1262, r1238, r1259;
}
{
cvt.rn.f16.f64 rs185, fd303;
}
mov.b32 r1267, {rs185, rs185};
{
mul.f16x2 r1265, r88, r1267;
}
{
add.f16x2 r1268, r1244, r1265;
}
{
cvt.rn.f16.f64 rs186, fd272;
}
mov.b32 r1273, {rs186, rs186};
{
mul.f16x2 r1271, r91, r1273;
}
{
add.f16x2 r1274, r1250, r1271;
}
{
cvt.rn.f16.f64 rs187, fd307;
}
mov.b32 r1279, {rs187, rs187};
{
mul.f16x2 r1277, r97, r1279;
}
{
add.f16x2 r1280, r1256, r1277;
}
{
cvt.rn.f16.f64 rs188, fd308;
}
mov.b32 r1285, {rs188, rs188};
{
mul.f16x2 r1283, r106, r1285;
}
{
add.f16x2 r1286, r1262, r1283;
}
{
cvt.rn.f16.f64 rs189, fd307;
}
mov.b32 r1291, {rs189, rs189};
{
mul.f16x2 r1289, r100, r1291;
}
{
add.f16x2 r1292, r1268, r1289;
}
{
cvt.rn.f16.f64 rs190, fd308;
}
mov.b32 r1297, {rs190, rs190};
{
mul.f16x2 r1295, r103, r1297;
}
{
add.f16x2 r1298, r1274, r1295;
}
{
sub.f16x2 %10, r1280, r1286;
}
{
add.f16x2 %11, r1292, r1298;
}
{
add.f16x2 %28, r1280, r1286;
}
{
sub.f16x2 %29, r1292, r1298;
}
cvt.rn.f16.s32 rs191, r2004;
mov.b32 r1325, {rs191, rs191};
cvt.rn.f16.s32 rs192, r2004;
mov.b32 r1337, {rs192, rs192};
{
cvt.rn.f16.f64 rs193, fd315;
}
mov.b32 r1317, {rs193, rs193};
{
mul.f16x2 r1315, r1, r1317;
}
{
add.f16x2 r1318, %38, r1315;
}
{
cvt.rn.f16.f64 rs194, fd316;
}
mov.b32 r1323, {rs194, rs194};
{
mul.f16x2 r1321, r10, r1323;
}
{
add.f16x2 r1324, r1325, r1321;
}
{
cvt.rn.f16.f64 rs195, fd315;
}
mov.b32 r1329, {rs195, rs195};
{
mul.f16x2 r1327, r4, r1329;
}
{
add.f16x2 r1330, %39, r1327;
}
{
cvt.rn.f16.f64 rs196, fd316;
}
mov.b32 r1335, {rs196, rs196};
{
mul.f16x2 r1333, r7, r1335;
}
{
add.f16x2 r1336, r1337, r1333;
}
{
cvt.rn.f16.f64 rs197, fd307;
}
mov.b32 r1341, {rs197, rs197};
{
mul.f16x2 r1339, r13, r1341;
}
{
add.f16x2 r1342, r1318, r1339;
}
{
cvt.rn.f16.f64 rs198, fd188;
}
mov.b32 r1347, {rs198, rs198};
{
mul.f16x2 r1345, r22, r1347;
}
{
add.f16x2 r1348, r1324, r1345;
}
{
cvt.rn.f16.f64 rs199, fd307;
}
mov.b32 r1353, {rs199, rs199};
{
mul.f16x2 r1351, r16, r1353;
}
{
add.f16x2 r1354, r1330, r1351;
}
{
cvt.rn.f16.f64 rs200, fd188;
}
mov.b32 r1359, {rs200, rs200};
{
mul.f16x2 r1357, r19, r1359;
}
{
add.f16x2 r1360, r1336, r1357;
}
{
cvt.rn.f16.f64 rs201, fd295;
}
mov.b32 r1365, {rs201, rs201};
{
mul.f16x2 r1363, r25, r1365;
}
{
add.f16x2 r1366, r1342, r1363;
}
{
cvt.rn.f16.f64 rs202, fd296;
}
mov.b32 r1371, {rs202, rs202};
{
mul.f16x2 r1369, r34, r1371;
}
{
add.f16x2 r1372, r1348, r1369;
}
{
cvt.rn.f16.f64 rs203, fd295;
}
mov.b32 r1377, {rs203, rs203};
{
mul.f16x2 r1375, r28, r1377;
}
{
add.f16x2 r1378, r1354, r1375;
}
{
cvt.rn.f16.f64 rs204, fd296;
}
mov.b32 r1383, {rs204, rs204};
{
mul.f16x2 r1381, r31, r1383;
}
{
add.f16x2 r1384, r1360, r1381;
}
{
cvt.rn.f16.f64 rs205, fd323;
}
mov.b32 r1389, {rs205, rs205};
{
mul.f16x2 r1387, r37, r1389;
}
{
add.f16x2 r1390, r1366, r1387;
}
{
cvt.rn.f16.f64 rs206, fd324;
}
mov.b32 r1395, {rs206, rs206};
{
mul.f16x2 r1393, r46, r1395;
}
{
add.f16x2 r1396, r1372, r1393;
}
{
cvt.rn.f16.f64 rs207, fd323;
}
mov.b32 r1401, {rs207, rs207};
{
mul.f16x2 r1399, r40, r1401;
}
{
add.f16x2 r1402, r1378, r1399;
}
{
cvt.rn.f16.f64 rs208, fd324;
}
mov.b32 r1407, {rs208, rs208};
{
mul.f16x2 r1405, r43, r1407;
}
{
add.f16x2 r1408, r1384, r1405;
}
{
cvt.rn.f16.f64 rs209, fd299;
}
mov.b32 r1413, {rs209, rs209};
{
mul.f16x2 r1411, r49, r1413;
}
{
add.f16x2 r1414, r1390, r1411;
}
{
cvt.rn.f16.f64 rs210, fd244;
}
mov.b32 r1419, {rs210, rs210};
{
mul.f16x2 r1417, r58, r1419;
}
{
add.f16x2 r1420, r1396, r1417;
}
{
cvt.rn.f16.f64 rs211, fd299;
}
mov.b32 r1425, {rs211, rs211};
{
mul.f16x2 r1423, r52, r1425;
}
{
add.f16x2 r1426, r1402, r1423;
}
{
cvt.rn.f16.f64 rs212, fd244;
}
mov.b32 r1431, {rs212, rs212};
{
mul.f16x2 r1429, r55, r1431;
}
{
add.f16x2 r1432, r1408, r1429;
}
{
cvt.rn.f16.f64 rs213, fd303;
}
mov.b32 r1437, {rs213, rs213};
{
mul.f16x2 r1435, r61, r1437;
}
{
add.f16x2 r1438, r1414, r1435;
}
{
cvt.rn.f16.f64 rs214, fd304;
}
mov.b32 r1443, {rs214, rs214};
{
mul.f16x2 r1441, r70, r1443;
}
{
add.f16x2 r1444, r1420, r1441;
}
{
cvt.rn.f16.f64 rs215, fd303;
}
mov.b32 r1449, {rs215, rs215};
{
mul.f16x2 r1447, r64, r1449;
}
{
add.f16x2 r1450, r1426, r1447;
}
{
cvt.rn.f16.f64 rs216, fd304;
}
mov.b32 r1455, {rs216, rs216};
{
mul.f16x2 r1453, r67, r1455;
}
{
add.f16x2 r1456, r1432, r1453;
}
{
cvt.rn.f16.f64 rs217, fd319;
}
mov.b32 r1461, {rs217, rs217};
{
mul.f16x2 r1459, r73, r1461;
}
{
add.f16x2 r1462, r1438, r1459;
}
{
cvt.rn.f16.f64 rs218, fd240;
}
mov.b32 r1467, {rs218, rs218};
{
mul.f16x2 r1465, r82, r1467;
}
{
add.f16x2 r1468, r1444, r1465;
}
{
cvt.rn.f16.f64 rs219, fd319;
}
mov.b32 r1473, {rs219, rs219};
{
mul.f16x2 r1471, r76, r1473;
}
{
add.f16x2 r1474, r1450, r1471;
}
{
cvt.rn.f16.f64 rs220, fd240;
}
mov.b32 r1479, {rs220, rs220};
{
mul.f16x2 r1477, r79, r1479;
}
{
add.f16x2 r1480, r1456, r1477;
}
{
cvt.rn.f16.f64 rs221, fd291;
}
mov.b32 r1485, {rs221, rs221};
{
mul.f16x2 r1483, r85, r1485;
}
{
add.f16x2 r1486, r1462, r1483;
}
{
cvt.rn.f16.f64 rs222, fd276;
}
mov.b32 r1491, {rs222, rs222};
{
mul.f16x2 r1489, r94, r1491;
}
{
add.f16x2 r1492, r1468, r1489;
}
{
cvt.rn.f16.f64 rs223, fd291;
}
mov.b32 r1497, {rs223, rs223};
{
mul.f16x2 r1495, r88, r1497;
}
{
add.f16x2 r1498, r1474, r1495;
}
{
cvt.rn.f16.f64 rs224, fd276;
}
mov.b32 r1503, {rs224, rs224};
{
mul.f16x2 r1501, r91, r1503;
}
{
add.f16x2 r1504, r1480, r1501;
}
{
cvt.rn.f16.f64 rs225, fd311;
}
mov.b32 r1509, {rs225, rs225};
{
mul.f16x2 r1507, r97, r1509;
}
{
add.f16x2 r1510, r1486, r1507;
}
{
cvt.rn.f16.f64 rs226, fd312;
}
mov.b32 r1515, {rs226, rs226};
{
mul.f16x2 r1513, r106, r1515;
}
{
add.f16x2 r1516, r1492, r1513;
}
{
cvt.rn.f16.f64 rs227, fd311;
}
mov.b32 r1521, {rs227, rs227};
{
mul.f16x2 r1519, r100, r1521;
}
{
add.f16x2 r1522, r1498, r1519;
}
{
cvt.rn.f16.f64 rs228, fd312;
}
mov.b32 r1527, {rs228, rs228};
{
mul.f16x2 r1525, r103, r1527;
}
{
add.f16x2 r1528, r1504, r1525;
}
{
sub.f16x2 %12, r1510, r1516;
}
{
add.f16x2 %13, r1522, r1528;
}
{
add.f16x2 %26, r1510, r1516;
}
{
sub.f16x2 %27, r1522, r1528;
}
cvt.rn.f16.s32 rs229, r2004;
mov.b32 r1555, {rs229, rs229};
cvt.rn.f16.s32 rs230, r2004;
mov.b32 r1567, {rs230, rs230};
{
cvt.rn.f16.f64 rs231, fd307;
}
mov.b32 r1547, {rs231, rs231};
{
mul.f16x2 r1545, r1, r1547;
}
{
add.f16x2 r1548, %38, r1545;
}
{
cvt.rn.f16.f64 rs232, fd308;
}
mov.b32 r1553, {rs232, rs232};
{
mul.f16x2 r1551, r10, r1553;
}
{
add.f16x2 r1554, r1555, r1551;
}
{
cvt.rn.f16.f64 rs233, fd307;
}
mov.b32 r1559, {rs233, rs233};
{
mul.f16x2 r1557, r4, r1559;
}
{
add.f16x2 r1560, %39, r1557;
}
{
cvt.rn.f16.f64 rs234, fd308;
}
mov.b32 r1565, {rs234, rs234};
{
mul.f16x2 r1563, r7, r1565;
}
{
add.f16x2 r1566, r1567, r1563;
}
{
cvt.rn.f16.f64 rs235, fd323;
}
mov.b32 r1571, {rs235, rs235};
{
mul.f16x2 r1569, r13, r1571;
}
{
add.f16x2 r1572, r1548, r1569;
}
{
cvt.rn.f16.f64 rs236, fd224;
}
mov.b32 r1577, {rs236, rs236};
{
mul.f16x2 r1575, r22, r1577;
}
{
add.f16x2 r1578, r1554, r1575;
}
{
cvt.rn.f16.f64 rs237, fd323;
}
mov.b32 r1583, {rs237, rs237};
{
mul.f16x2 r1581, r16, r1583;
}
{
add.f16x2 r1584, r1560, r1581;
}
{
cvt.rn.f16.f64 rs238, fd224;
}
mov.b32 r1589, {rs238, rs238};
{
mul.f16x2 r1587, r19, r1589;
}
{
add.f16x2 r1590, r1566, r1587;
}
{
cvt.rn.f16.f64 rs239, fd303;
}
mov.b32 r1595, {rs239, rs239};
{
mul.f16x2 r1593, r25, r1595;
}
{
add.f16x2 r1596, r1572, r1593;
}
{
cvt.rn.f16.f64 rs240, fd272;
}
mov.b32 r1601, {rs240, rs240};
{
mul.f16x2 r1599, r34, r1601;
}
{
add.f16x2 r1602, r1578, r1599;
}
{
cvt.rn.f16.f64 rs241, fd303;
}
mov.b32 r1607, {rs241, rs241};
{
mul.f16x2 r1605, r28, r1607;
}
{
add.f16x2 r1608, r1584, r1605;
}
{
cvt.rn.f16.f64 rs242, fd272;
}
mov.b32 r1613, {rs242, rs242};
{
mul.f16x2 r1611, r31, r1613;
}
{
add.f16x2 r1614, r1590, r1611;
}
{
cvt.rn.f16.f64 rs243, fd291;
}
mov.b32 r1619, {rs243, rs243};
{
mul.f16x2 r1617, r37, r1619;
}
{
add.f16x2 r1620, r1596, r1617;
}
{
cvt.rn.f16.f64 rs244, fd292;
}
mov.b32 r1625, {rs244, rs244};
{
mul.f16x2 r1623, r46, r1625;
}
{
add.f16x2 r1626, r1602, r1623;
}
{
cvt.rn.f16.f64 rs245, fd291;
}
mov.b32 r1631, {rs245, rs245};
{
mul.f16x2 r1629, r40, r1631;
}
{
add.f16x2 r1632, r1608, r1629;
}
{
cvt.rn.f16.f64 rs246, fd292;
}
mov.b32 r1637, {rs246, rs246};
{
mul.f16x2 r1635, r43, r1637;
}
{
add.f16x2 r1638, r1614, r1635;
}
{
cvt.rn.f16.f64 rs247, fd311;
}
mov.b32 r1643, {rs247, rs247};
{
mul.f16x2 r1641, r49, r1643;
}
{
add.f16x2 r1644, r1620, r1641;
}
{
cvt.rn.f16.f64 rs248, fd312;
}
mov.b32 r1649, {rs248, rs248};
{
mul.f16x2 r1647, r58, r1649;
}
{
add.f16x2 r1650, r1626, r1647;
}
{
cvt.rn.f16.f64 rs249, fd311;
}
mov.b32 r1655, {rs249, rs249};
{
mul.f16x2 r1653, r52, r1655;
}
{
add.f16x2 r1656, r1632, r1653;
}
{
cvt.rn.f16.f64 rs250, fd312;
}
mov.b32 r1661, {rs250, rs250};
{
mul.f16x2 r1659, r55, r1661;
}
{
add.f16x2 r1662, r1638, r1659;
}
{
cvt.rn.f16.f64 rs251, fd319;
}
mov.b32 r1667, {rs251, rs251};
{
mul.f16x2 r1665, r61, r1667;
}
{
add.f16x2 r1668, r1644, r1665;
}
{
cvt.rn.f16.f64 rs252, fd240;
}
mov.b32 r1673, {rs252, rs252};
{
mul.f16x2 r1671, r70, r1673;
}
{
add.f16x2 r1674, r1650, r1671;
}
{
cvt.rn.f16.f64 rs253, fd319;
}
mov.b32 r1679, {rs253, rs253};
{
mul.f16x2 r1677, r64, r1679;
}
{
add.f16x2 r1680, r1656, r1677;
}
{
cvt.rn.f16.f64 rs254, fd240;
}
mov.b32 r1685, {rs254, rs254};
{
mul.f16x2 r1683, r67, r1685;
}
{
add.f16x2 r1686, r1662, r1683;
}
{
cvt.rn.f16.f64 rs255, fd299;
}
mov.b32 r1691, {rs255, rs255};
{
mul.f16x2 r1689, r73, r1691;
}
{
add.f16x2 r1692, r1668, r1689;
}
{
cvt.rn.f16.f64 rs256, fd244;
}
mov.b32 r1697, {rs256, rs256};
{
mul.f16x2 r1695, r82, r1697;
}
{
add.f16x2 r1698, r1674, r1695;
}
{
cvt.rn.f16.f64 rs257, fd299;
}
mov.b32 r1703, {rs257, rs257};
{
mul.f16x2 r1701, r76, r1703;
}
{
add.f16x2 r1704, r1680, r1701;
}
{
cvt.rn.f16.f64 rs258, fd244;
}
mov.b32 r1709, {rs258, rs258};
{
mul.f16x2 r1707, r79, r1709;
}
{
add.f16x2 r1710, r1686, r1707;
}
{
cvt.rn.f16.f64 rs259, fd295;
}
mov.b32 r1715, {rs259, rs259};
{
mul.f16x2 r1713, r85, r1715;
}
{
add.f16x2 r1716, r1692, r1713;
}
{
cvt.rn.f16.f64 rs260, fd296;
}
mov.b32 r1721, {rs260, rs260};
{
mul.f16x2 r1719, r94, r1721;
}
{
add.f16x2 r1722, r1698, r1719;
}
{
cvt.rn.f16.f64 rs261, fd295;
}
mov.b32 r1727, {rs261, rs261};
{
mul.f16x2 r1725, r88, r1727;
}
{
add.f16x2 r1728, r1704, r1725;
}
{
cvt.rn.f16.f64 rs262, fd296;
}
mov.b32 r1733, {rs262, rs262};
{
mul.f16x2 r1731, r91, r1733;
}
{
add.f16x2 r1734, r1710, r1731;
}
{
cvt.rn.f16.f64 rs263, fd315;
}
mov.b32 r1739, {rs263, rs263};
{
mul.f16x2 r1737, r97, r1739;
}
{
add.f16x2 r1740, r1716, r1737;
}
{
cvt.rn.f16.f64 rs264, fd316;
}
mov.b32 r1745, {rs264, rs264};
{
mul.f16x2 r1743, r106, r1745;
}
{
add.f16x2 r1746, r1722, r1743;
}
{
cvt.rn.f16.f64 rs265, fd315;
}
mov.b32 r1751, {rs265, rs265};
{
mul.f16x2 r1749, r100, r1751;
}
{
add.f16x2 r1752, r1728, r1749;
}
{
cvt.rn.f16.f64 rs266, fd316;
}
mov.b32 r1757, {rs266, rs266};
{
mul.f16x2 r1755, r103, r1757;
}
{
add.f16x2 r1758, r1734, r1755;
}
{
sub.f16x2 %14, r1740, r1746;
}
{
add.f16x2 %15, r1752, r1758;
}
{
add.f16x2 %24, r1740, r1746;
}
{
sub.f16x2 %25, r1752, r1758;
}
cvt.rn.f16.s32 rs267, r2004;
mov.b32 r1785, {rs267, rs267};
cvt.rn.f16.s32 rs268, r2004;
mov.b32 r1797, {rs268, rs268};
{
cvt.rn.f16.f64 rs269, fd299;
}
mov.b32 r1777, {rs269, rs269};
{
mul.f16x2 r1775, r1, r1777;
}
{
add.f16x2 r1778, %38, r1775;
}
{
cvt.rn.f16.f64 rs270, fd300;
}
mov.b32 r1783, {rs270, rs270};
{
mul.f16x2 r1781, r10, r1783;
}
{
add.f16x2 r1784, r1785, r1781;
}
{
cvt.rn.f16.f64 rs271, fd299;
}
mov.b32 r1789, {rs271, rs271};
{
mul.f16x2 r1787, r4, r1789;
}
{
add.f16x2 r1790, %39, r1787;
}
{
cvt.rn.f16.f64 rs272, fd300;
}
mov.b32 r1795, {rs272, rs272};
{
mul.f16x2 r1793, r7, r1795;
}
{
add.f16x2 r1796, r1797, r1793;
}
{
cvt.rn.f16.f64 rs273, fd311;
}
mov.b32 r1801, {rs273, rs273};
{
mul.f16x2 r1799, r13, r1801;
}
{
add.f16x2 r1802, r1778, r1799;
}
{
cvt.rn.f16.f64 rs274, fd312;
}
mov.b32 r1807, {rs274, rs274};
{
mul.f16x2 r1805, r22, r1807;
}
{
add.f16x2 r1808, r1784, r1805;
}
{
cvt.rn.f16.f64 rs275, fd311;
}
mov.b32 r1813, {rs275, rs275};
{
mul.f16x2 r1811, r16, r1813;
}
{
add.f16x2 r1814, r1790, r1811;
}
{
cvt.rn.f16.f64 rs276, fd312;
}
mov.b32 r1819, {rs276, rs276};
{
mul.f16x2 r1817, r19, r1819;
}
{
add.f16x2 r1820, r1796, r1817;
}
{
cvt.rn.f16.f64 rs277, fd323;
}
mov.b32 r1825, {rs277, rs277};
{
mul.f16x2 r1823, r25, r1825;
}
{
add.f16x2 r1826, r1802, r1823;
}
{
cvt.rn.f16.f64 rs278, fd324;
}
mov.b32 r1831, {rs278, rs278};
{
mul.f16x2 r1829, r34, r1831;
}
{
add.f16x2 r1832, r1808, r1829;
}
{
cvt.rn.f16.f64 rs279, fd323;
}
mov.b32 r1837, {rs279, rs279};
{
mul.f16x2 r1835, r28, r1837;
}
{
add.f16x2 r1838, r1814, r1835;
}
{
cvt.rn.f16.f64 rs280, fd324;
}
mov.b32 r1843, {rs280, rs280};
{
mul.f16x2 r1841, r31, r1843;
}
{
add.f16x2 r1844, r1820, r1841;
}
{
cvt.rn.f16.f64 rs281, fd315;
}
mov.b32 r1849, {rs281, rs281};
{
mul.f16x2 r1847, r37, r1849;
}
{
add.f16x2 r1850, r1826, r1847;
}
{
cvt.rn.f16.f64 rs282, fd268;
}
mov.b32 r1855, {rs282, rs282};
{
mul.f16x2 r1853, r46, r1855;
}
{
add.f16x2 r1856, r1832, r1853;
}
{
cvt.rn.f16.f64 rs283, fd315;
}
mov.b32 r1861, {rs283, rs283};
{
mul.f16x2 r1859, r40, r1861;
}
{
add.f16x2 r1862, r1838, r1859;
}
{
cvt.rn.f16.f64 rs284, fd268;
}
mov.b32 r1867, {rs284, rs284};
{
mul.f16x2 r1865, r43, r1867;
}
{
add.f16x2 r1868, r1844, r1865;
}
{
cvt.rn.f16.f64 rs285, fd303;
}
mov.b32 r1873, {rs285, rs285};
{
mul.f16x2 r1871, r49, r1873;
}
{
add.f16x2 r1874, r1850, r1871;
}
{
cvt.rn.f16.f64 rs286, fd272;
}
mov.b32 r1879, {rs286, rs286};
{
mul.f16x2 r1877, r58, r1879;
}
{
add.f16x2 r1880, r1856, r1877;
}
{
cvt.rn.f16.f64 rs287, fd303;
}
mov.b32 r1885, {rs287, rs287};
{
mul.f16x2 r1883, r52, r1885;
}
{
add.f16x2 r1886, r1862, r1883;
}
{
cvt.rn.f16.f64 rs288, fd272;
}
mov.b32 r1891, {rs288, rs288};
{
mul.f16x2 r1889, r55, r1891;
}
{
add.f16x2 r1892, r1868, r1889;
}
{
cvt.rn.f16.f64 rs289, fd291;
}
mov.b32 r1897, {rs289, rs289};
{
mul.f16x2 r1895, r61, r1897;
}
{
add.f16x2 r1898, r1874, r1895;
}
{
cvt.rn.f16.f64 rs290, fd276;
}
mov.b32 r1903, {rs290, rs290};
{
mul.f16x2 r1901, r70, r1903;
}
{
add.f16x2 r1904, r1880, r1901;
}
{
cvt.rn.f16.f64 rs291, fd291;
}
mov.b32 r1909, {rs291, rs291};
{
mul.f16x2 r1907, r64, r1909;
}
{
add.f16x2 r1910, r1886, r1907;
}
{
cvt.rn.f16.f64 rs292, fd276;
}
mov.b32 r1915, {rs292, rs292};
{
mul.f16x2 r1913, r67, r1915;
}
{
add.f16x2 r1916, r1892, r1913;
}
{
cvt.rn.f16.f64 rs293, fd295;
}
mov.b32 r1921, {rs293, rs293};
{
mul.f16x2 r1919, r73, r1921;
}
{
add.f16x2 r1922, r1898, r1919;
}
{
cvt.rn.f16.f64 rs294, fd296;
}
mov.b32 r1927, {rs294, rs294};
{
mul.f16x2 r1925, r82, r1927;
}
{
add.f16x2 r1928, r1904, r1925;
}
{
cvt.rn.f16.f64 rs295, fd295;
}
mov.b32 r1933, {rs295, rs295};
{
mul.f16x2 r1931, r76, r1933;
}
{
add.f16x2 r1934, r1910, r1931;
}
{
cvt.rn.f16.f64 rs296, fd296;
}
mov.b32 r1939, {rs296, rs296};
{
mul.f16x2 r1937, r79, r1939;
}
{
add.f16x2 r1940, r1916, r1937;
}
{
cvt.rn.f16.f64 rs297, fd307;
}
mov.b32 r1945, {rs297, rs297};
{
mul.f16x2 r1943, r85, r1945;
}
{
add.f16x2 r1946, r1922, r1943;
}
{
cvt.rn.f16.f64 rs298, fd308;
}
mov.b32 r1951, {rs298, rs298};
{
mul.f16x2 r1949, r94, r1951;
}
{
add.f16x2 r1952, r1928, r1949;
}
{
cvt.rn.f16.f64 rs299, fd307;
}
mov.b32 r1957, {rs299, rs299};
{
mul.f16x2 r1955, r88, r1957;
}
{
add.f16x2 r1958, r1934, r1955;
}
{
cvt.rn.f16.f64 rs300, fd308;
}
mov.b32 r1963, {rs300, rs300};
{
mul.f16x2 r1961, r91, r1963;
}
{
add.f16x2 r1964, r1940, r1961;
}
{
cvt.rn.f16.f64 rs301, fd319;
}
mov.b32 r1969, {rs301, rs301};
{
mul.f16x2 r1967, r97, r1969;
}
{
add.f16x2 r1970, r1946, r1967;
}
{
cvt.rn.f16.f64 rs302, fd320;
}
mov.b32 r1975, {rs302, rs302};
{
mul.f16x2 r1973, r106, r1975;
}
{
add.f16x2 r1976, r1952, r1973;
}
{
cvt.rn.f16.f64 rs303, fd319;
}
mov.b32 r1981, {rs303, rs303};
{
mul.f16x2 r1979, r100, r1981;
}
{
add.f16x2 r1982, r1958, r1979;
}
{
cvt.rn.f16.f64 rs304, fd320;
}
mov.b32 r1987, {rs304, rs304};
{
mul.f16x2 r1985, r103, r1987;
}
{
add.f16x2 r1988, r1964, r1985;
}
{
sub.f16x2 %16, r1970, r1976;
}
{
add.f16x2 %17, r1982, r1988;
}
{
add.f16x2 %22, r1970, r1976;
}
{
sub.f16x2 %23, r1982, r1988;
}
cvt.rn.f16.s32 rs305, r2004;
mov.b32 r2015, {rs305, rs305};
cvt.rn.f16.s32 rs306, r2004;
mov.b32 r2027, {rs306, rs306};
{
cvt.rn.f16.f64 rs307, fd291;
}
mov.b32 r2007, {rs307, rs307};
{
mul.f16x2 r2005, r1, r2007;
}
{
add.f16x2 r2008, %38, r2005;
}
{
cvt.rn.f16.f64 rs308, fd292;
}
mov.b32 r2013, {rs308, rs308};
{
mul.f16x2 r2011, r10, r2013;
}
{
add.f16x2 r2014, r2015, r2011;
}
{
cvt.rn.f16.f64 rs309, fd291;
}
mov.b32 r2019, {rs309, rs309};
{
mul.f16x2 r2017, r4, r2019;
}
{
add.f16x2 r2020, %39, r2017;
}
{
cvt.rn.f16.f64 rs310, fd292;
}
mov.b32 r2025, {rs310, rs310};
{
mul.f16x2 r2023, r7, r2025;
}
{
add.f16x2 r2026, r2027, r2023;
}
{
cvt.rn.f16.f64 rs311, fd295;
}
mov.b32 r2031, {rs311, rs311};
{
mul.f16x2 r2029, r13, r2031;
}
{
add.f16x2 r2032, r2008, r2029;
}
{
cvt.rn.f16.f64 rs312, fd296;
}
mov.b32 r2037, {rs312, rs312};
{
mul.f16x2 r2035, r22, r2037;
}
{
add.f16x2 r2038, r2014, r2035;
}
{
cvt.rn.f16.f64 rs313, fd295;
}
mov.b32 r2043, {rs313, rs313};
{
mul.f16x2 r2041, r16, r2043;
}
{
add.f16x2 r2044, r2020, r2041;
}
{
cvt.rn.f16.f64 rs314, fd296;
}
mov.b32 r2049, {rs314, rs314};
{
mul.f16x2 r2047, r19, r2049;
}
{
add.f16x2 r2050, r2026, r2047;
}
{
cvt.rn.f16.f64 rs315, fd299;
}
mov.b32 r2055, {rs315, rs315};
{
mul.f16x2 r2053, r25, r2055;
}
{
add.f16x2 r2056, r2032, r2053;
}
{
cvt.rn.f16.f64 rs316, fd300;
}
mov.b32 r2061, {rs316, rs316};
{
mul.f16x2 r2059, r34, r2061;
}
{
add.f16x2 r2062, r2038, r2059;
}
{
cvt.rn.f16.f64 rs317, fd299;
}
mov.b32 r2067, {rs317, rs317};
{
mul.f16x2 r2065, r28, r2067;
}
{
add.f16x2 r2068, r2044, r2065;
}
{
cvt.rn.f16.f64 rs318, fd300;
}
mov.b32 r2073, {rs318, rs318};
{
mul.f16x2 r2071, r31, r2073;
}
{
add.f16x2 r2074, r2050, r2071;
}
{
cvt.rn.f16.f64 rs319, fd303;
}
mov.b32 r2079, {rs319, rs319};
{
mul.f16x2 r2077, r37, r2079;
}
{
add.f16x2 r2080, r2056, r2077;
}
{
cvt.rn.f16.f64 rs320, fd304;
}
mov.b32 r2085, {rs320, rs320};
{
mul.f16x2 r2083, r46, r2085;
}
{
add.f16x2 r2086, r2062, r2083;
}
{
cvt.rn.f16.f64 rs321, fd303;
}
mov.b32 r2091, {rs321, rs321};
{
mul.f16x2 r2089, r40, r2091;
}
{
add.f16x2 r2092, r2068, r2089;
}
{
cvt.rn.f16.f64 rs322, fd304;
}
mov.b32 r2097, {rs322, rs322};
{
mul.f16x2 r2095, r43, r2097;
}
{
add.f16x2 r2098, r2074, r2095;
}
{
cvt.rn.f16.f64 rs323, fd307;
}
mov.b32 r2103, {rs323, rs323};
{
mul.f16x2 r2101, r49, r2103;
}
{
add.f16x2 r2104, r2080, r2101;
}
{
cvt.rn.f16.f64 rs324, fd308;
}
mov.b32 r2109, {rs324, rs324};
{
mul.f16x2 r2107, r58, r2109;
}
{
add.f16x2 r2110, r2086, r2107;
}
{
cvt.rn.f16.f64 rs325, fd307;
}
mov.b32 r2115, {rs325, rs325};
{
mul.f16x2 r2113, r52, r2115;
}
{
add.f16x2 r2116, r2092, r2113;
}
{
cvt.rn.f16.f64 rs326, fd308;
}
mov.b32 r2121, {rs326, rs326};
{
mul.f16x2 r2119, r55, r2121;
}
{
add.f16x2 r2122, r2098, r2119;
}
{
cvt.rn.f16.f64 rs327, fd311;
}
mov.b32 r2127, {rs327, rs327};
{
mul.f16x2 r2125, r61, r2127;
}
{
add.f16x2 r2128, r2104, r2125;
}
{
cvt.rn.f16.f64 rs328, fd312;
}
mov.b32 r2133, {rs328, rs328};
{
mul.f16x2 r2131, r70, r2133;
}
{
add.f16x2 r2134, r2110, r2131;
}
{
cvt.rn.f16.f64 rs329, fd311;
}
mov.b32 r2139, {rs329, rs329};
{
mul.f16x2 r2137, r64, r2139;
}
{
add.f16x2 r2140, r2116, r2137;
}
{
cvt.rn.f16.f64 rs330, fd312;
}
mov.b32 r2145, {rs330, rs330};
{
mul.f16x2 r2143, r67, r2145;
}
{
add.f16x2 r2146, r2122, r2143;
}
{
cvt.rn.f16.f64 rs331, fd315;
}
mov.b32 r2151, {rs331, rs331};
{
mul.f16x2 r2149, r73, r2151;
}
{
add.f16x2 r2152, r2128, r2149;
}
{
cvt.rn.f16.f64 rs332, fd316;
}
mov.b32 r2157, {rs332, rs332};
{
mul.f16x2 r2155, r82, r2157;
}
{
add.f16x2 r2158, r2134, r2155;
}
{
cvt.rn.f16.f64 rs333, fd315;
}
mov.b32 r2163, {rs333, rs333};
{
mul.f16x2 r2161, r76, r2163;
}
{
add.f16x2 r2164, r2140, r2161;
}
{
cvt.rn.f16.f64 rs334, fd316;
}
mov.b32 r2169, {rs334, rs334};
{
mul.f16x2 r2167, r79, r2169;
}
{
add.f16x2 r2170, r2146, r2167;
}
{
cvt.rn.f16.f64 rs335, fd319;
}
mov.b32 r2175, {rs335, rs335};
{
mul.f16x2 r2173, r85, r2175;
}
{
add.f16x2 r2176, r2152, r2173;
}
{
cvt.rn.f16.f64 rs336, fd320;
}
mov.b32 r2181, {rs336, rs336};
{
mul.f16x2 r2179, r94, r2181;
}
{
add.f16x2 r2182, r2158, r2179;
}
{
cvt.rn.f16.f64 rs337, fd319;
}
mov.b32 r2187, {rs337, rs337};
{
mul.f16x2 r2185, r88, r2187;
}
{
add.f16x2 r2188, r2164, r2185;
}
{
cvt.rn.f16.f64 rs338, fd320;
}
mov.b32 r2193, {rs338, rs338};
{
mul.f16x2 r2191, r91, r2193;
}
{
add.f16x2 r2194, r2170, r2191;
}
{
cvt.rn.f16.f64 rs339, fd323;
}
mov.b32 r2199, {rs339, rs339};
{
mul.f16x2 r2197, r97, r2199;
}
{
add.f16x2 r2200, r2176, r2197;
}
{
cvt.rn.f16.f64 rs340, fd324;
}
mov.b32 r2205, {rs340, rs340};
{
mul.f16x2 r2203, r106, r2205;
}
{
add.f16x2 r2206, r2182, r2203;
}
{
cvt.rn.f16.f64 rs341, fd323;
}
mov.b32 r2211, {rs341, rs341};
{
mul.f16x2 r2209, r100, r2211;
}
{
add.f16x2 r2212, r2188, r2209;
}
{
cvt.rn.f16.f64 rs342, fd324;
}
mov.b32 r2217, {rs342, rs342};
{
mul.f16x2 r2215, r103, r2217;
}
{
add.f16x2 r2218, r2194, r2215;
}
{
sub.f16x2 %18, r2200, r2206;
}
{
add.f16x2 %19, r2212, r2218;
}
{
add.f16x2 %20, r2200, r2206;
}
{
sub.f16x2 %21, r2212, r2218;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[18].y)));
};


#endif
