#ifndef CUFFTDX_FFT_29_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_29_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<759, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<813>;
.reg .b32 r<5153>;
.reg .f64 fd<785>;
.reg .b64 rd<2>;
{
add.f16x2 r1, %60, %114;
}
{
add.f16x2 r4, %61, %115;
}
{
sub.f16x2 r7, %60, %114;
}
{
sub.f16x2 r10, %61, %115;
}
{
add.f16x2 r13, %62, %112;
}
{
add.f16x2 r16, %63, %113;
}
{
sub.f16x2 r19, %62, %112;
}
{
sub.f16x2 r22, %63, %113;
}
{
add.f16x2 r25, %64, %110;
}
{
add.f16x2 r28, %65, %111;
}
{
sub.f16x2 r31, %64, %110;
}
{
sub.f16x2 r34, %65, %111;
}
{
add.f16x2 r37, %66, %108;
}
{
add.f16x2 r40, %67, %109;
}
{
sub.f16x2 r43, %66, %108;
}
{
sub.f16x2 r46, %67, %109;
}
{
add.f16x2 r49, %68, %106;
}
{
add.f16x2 r52, %69, %107;
}
{
sub.f16x2 r55, %68, %106;
}
{
sub.f16x2 r58, %69, %107;
}
{
add.f16x2 r61, %70, %104;
}
{
add.f16x2 r64, %71, %105;
}
{
sub.f16x2 r67, %70, %104;
}
{
sub.f16x2 r70, %71, %105;
}
{
add.f16x2 r73, %72, %102;
}
{
add.f16x2 r76, %73, %103;
}
{
sub.f16x2 r79, %72, %102;
}
{
sub.f16x2 r82, %73, %103;
}
{
add.f16x2 r85, %74, %100;
}
{
add.f16x2 r88, %75, %101;
}
{
sub.f16x2 r91, %74, %100;
}
{
sub.f16x2 r94, %75, %101;
}
{
add.f16x2 r97, %76, %98;
}
{
add.f16x2 r100, %77, %99;
}
{
sub.f16x2 r103, %76, %98;
}
{
sub.f16x2 r106, %77, %99;
}
{
add.f16x2 r109, %78, %96;
}
{
add.f16x2 r112, %79, %97;
}
{
sub.f16x2 r115, %78, %96;
}
{
sub.f16x2 r118, %79, %97;
}
{
add.f16x2 r121, %80, %94;
}
{
add.f16x2 r124, %81, %95;
}
{
sub.f16x2 r127, %80, %94;
}
{
sub.f16x2 r130, %81, %95;
}
{
add.f16x2 r133, %82, %92;
}
{
add.f16x2 r136, %83, %93;
}
{
sub.f16x2 r139, %82, %92;
}
{
sub.f16x2 r142, %83, %93;
}
{
add.f16x2 r145, %84, %90;
}
{
add.f16x2 r148, %85, %91;
}
{
sub.f16x2 r151, %84, %90;
}
{
sub.f16x2 r154, %85, %91;
}
{
add.f16x2 r157, %86, %88;
}
{
add.f16x2 r160, %87, %89;
}
{
sub.f16x2 r163, %86, %88;
}
{
sub.f16x2 r166, %87, %89;
}
{
add.f16x2 r169, %58, r1;
}
{
add.f16x2 r172, %59, r4;
}
{
add.f16x2 r175, r169, r13;
}
{
add.f16x2 r178, r172, r16;
}
{
add.f16x2 r181, r175, r25;
}
{
add.f16x2 r184, r178, r28;
}
{
add.f16x2 r187, r181, r37;
}
{
add.f16x2 r190, r184, r40;
}
{
add.f16x2 r193, r187, r49;
}
{
add.f16x2 r196, r190, r52;
}
{
add.f16x2 r199, r193, r61;
}
{
add.f16x2 r202, r196, r64;
}
{
add.f16x2 r205, r199, r73;
}
{
add.f16x2 r208, r202, r76;
}
{
add.f16x2 r211, r205, r85;
}
{
add.f16x2 r214, r208, r88;
}
{
add.f16x2 r217, r211, r97;
}
{
add.f16x2 r220, r214, r100;
}
{
add.f16x2 r223, r217, r109;
}
{
add.f16x2 r226, r220, r112;
}
{
add.f16x2 r229, r223, r121;
}
{
add.f16x2 r232, r226, r124;
}
{
add.f16x2 r235, r229, r133;
}
{
add.f16x2 r238, r232, r136;
}
{
add.f16x2 r241, r235, r145;
}
{
add.f16x2 r244, r238, r148;
}
{
add.f16x2 %0, r241, r157;
}
{
add.f16x2 %1, r244, r160;
}
mov.u32 r4804, 0;
cvt.rn.f16.s32 rs1, r4804;
mov.b32 r265, {rs1, rs1};
cvt.rn.f16.s32 rs2, r4804;
mov.b32 r277, {rs2, rs2};
mov.f64 fd735, 0d3FEF4079C06C0992;
{
cvt.rn.f16.f64 rs3, fd735;
}
mov.b32 r257, {rs3, rs3};
{
mul.f16x2 r255, r1, r257;
}
{
add.f16x2 r258, %58, r255;
}
mov.f64 fd708, 0dBFCB8426C12812BC;
{
cvt.rn.f16.f64 rs4, fd708;
}
mov.b32 r263, {rs4, rs4};
{
mul.f16x2 r261, r10, r263;
}
{
add.f16x2 r264, r265, r261;
}
{
cvt.rn.f16.f64 rs5, fd735;
}
mov.b32 r269, {rs5, rs5};
{
mul.f16x2 r267, r4, r269;
}
{
add.f16x2 r270, %59, r267;
}
{
cvt.rn.f16.f64 rs6, fd708;
}
mov.b32 r275, {rs6, rs6};
{
mul.f16x2 r273, r7, r275;
}
{
add.f16x2 r276, r277, r273;
}
mov.f64 fd743, 0d3FED0ADB9B447CCF;
{
cvt.rn.f16.f64 rs7, fd743;
}
mov.b32 r281, {rs7, rs7};
{
mul.f16x2 r279, r13, r281;
}
{
add.f16x2 r282, r258, r279;
}
mov.f64 fd636, 0dBFDADF7689C97B70;
{
cvt.rn.f16.f64 rs8, fd636;
}
mov.b32 r287, {rs8, rs8};
{
mul.f16x2 r285, r22, r287;
}
{
add.f16x2 r288, r264, r285;
}
{
cvt.rn.f16.f64 rs9, fd743;
}
mov.b32 r293, {rs9, rs9};
{
mul.f16x2 r291, r16, r293;
}
{
add.f16x2 r294, r270, r291;
}
{
cvt.rn.f16.f64 rs10, fd636;
}
mov.b32 r299, {rs10, rs10};
{
mul.f16x2 r297, r19, r299;
}
{
add.f16x2 r300, r276, r297;
}
mov.f64 fd751, 0d3FE979982A38E65A;
{
cvt.rn.f16.f64 rs11, fd751;
}
mov.b32 r305, {rs11, rs11};
{
mul.f16x2 r303, r25, r305;
}
{
add.f16x2 r306, r282, r303;
}
mov.f64 fd540, 0dBFE35D9650D47852;
{
cvt.rn.f16.f64 rs12, fd540;
}
mov.b32 r311, {rs12, rs12};
{
mul.f16x2 r309, r34, r311;
}
{
add.f16x2 r312, r288, r309;
}
{
cvt.rn.f16.f64 rs13, fd751;
}
mov.b32 r317, {rs13, rs13};
{
mul.f16x2 r315, r28, r317;
}
{
add.f16x2 r318, r294, r315;
}
{
cvt.rn.f16.f64 rs14, fd540;
}
mov.b32 r323, {rs14, rs14};
{
mul.f16x2 r321, r31, r323;
}
{
add.f16x2 r324, r300, r321;
}
mov.f64 fd759, 0d3FE4B76371208A62;
{
cvt.rn.f16.f64 rs15, fd759;
}
mov.b32 r329, {rs15, rs15};
{
mul.f16x2 r327, r37, r329;
}
{
add.f16x2 r330, r306, r327;
}
mov.f64 fd700, 0dBFE863A1ADA0CFA6;
{
cvt.rn.f16.f64 rs16, fd700;
}
mov.b32 r335, {rs16, rs16};
{
mul.f16x2 r333, r46, r335;
}
{
add.f16x2 r336, r312, r333;
}
{
cvt.rn.f16.f64 rs17, fd759;
}
mov.b32 r341, {rs17, rs17};
{
mul.f16x2 r339, r40, r341;
}
{
add.f16x2 r342, r318, r339;
}
{
cvt.rn.f16.f64 rs18, fd700;
}
mov.b32 r347, {rs18, rs18};
{
mul.f16x2 r345, r43, r347;
}
{
add.f16x2 r348, r324, r345;
}
mov.f64 fd767, 0d3FDDFA67657E7608;
{
cvt.rn.f16.f64 rs19, fd767;
}
mov.b32 r353, {rs19, rs19};
{
mul.f16x2 r351, r49, r353;
}
{
add.f16x2 r354, r330, r351;
}
mov.f64 fd604, 0dBFEC45BB0D10918C;
{
cvt.rn.f16.f64 rs20, fd604;
}
mov.b32 r359, {rs20, rs20};
{
mul.f16x2 r357, r58, r359;
}
{
add.f16x2 r360, r336, r357;
}
{
cvt.rn.f16.f64 rs21, fd767;
}
mov.b32 r365, {rs21, rs21};
{
mul.f16x2 r363, r52, r365;
}
{
add.f16x2 r366, r342, r363;
}
{
cvt.rn.f16.f64 rs22, fd604;
}
mov.b32 r371, {rs22, rs22};
{
mul.f16x2 r369, r55, r371;
}
{
add.f16x2 r372, r348, r369;
}
mov.f64 fd775, 0d3FD11F2F2E2F1E3B;
{
cvt.rn.f16.f64 rs23, fd775;
}
mov.b32 r377, {rs23, rs23};
{
mul.f16x2 r375, r61, r377;
}
{
add.f16x2 r378, r354, r375;
}
mov.f64 fd424, 0dBFEED566CB3DCBA1;
{
cvt.rn.f16.f64 rs24, fd424;
}
mov.b32 r383, {rs24, rs24};
{
mul.f16x2 r381, r70, r383;
}
{
add.f16x2 r384, r360, r381;
}
{
cvt.rn.f16.f64 rs25, fd775;
}
mov.b32 r389, {rs25, rs25};
{
mul.f16x2 r387, r64, r389;
}
{
add.f16x2 r390, r366, r387;
}
{
cvt.rn.f16.f64 rs26, fd424;
}
mov.b32 r395, {rs26, rs26};
{
mul.f16x2 r393, r67, r395;
}
{
add.f16x2 r396, r372, r393;
}
mov.f64 fd783, 0d3FABB81853A18977;
{
cvt.rn.f16.f64 rs27, fd783;
}
mov.b32 r401, {rs27, rs27};
{
mul.f16x2 r399, r73, r401;
}
{
add.f16x2 r402, r378, r399;
}
mov.f64 fd692, 0dBFEFF3FC588E859D;
{
cvt.rn.f16.f64 rs28, fd692;
}
mov.b32 r407, {rs28, rs28};
{
mul.f16x2 r405, r82, r407;
}
{
add.f16x2 r408, r384, r405;
}
{
cvt.rn.f16.f64 rs29, fd783;
}
mov.b32 r413, {rs29, rs29};
{
mul.f16x2 r411, r76, r413;
}
{
add.f16x2 r414, r390, r411;
}
{
cvt.rn.f16.f64 rs30, fd692;
}
mov.b32 r419, {rs30, rs30};
{
mul.f16x2 r417, r79, r419;
}
{
add.f16x2 r420, r396, r417;
}
mov.f64 fd779, 0dBFC4B545C0234A71;
{
cvt.rn.f16.f64 rs31, fd779;
}
mov.b32 r425, {rs31, rs31};
{
mul.f16x2 r423, r85, r425;
}
{
add.f16x2 r426, r402, r423;
}
mov.f64 fd780, 0dBFEF941537248537;
{
cvt.rn.f16.f64 rs32, fd780;
}
mov.b32 r431, {rs32, rs32};
{
mul.f16x2 r429, r94, r431;
}
{
add.f16x2 r432, r408, r429;
}
{
cvt.rn.f16.f64 rs33, fd779;
}
mov.b32 r437, {rs33, rs33};
{
mul.f16x2 r435, r88, r437;
}
{
add.f16x2 r438, r414, r435;
}
{
cvt.rn.f16.f64 rs34, fd780;
}
mov.b32 r443, {rs34, rs34};
{
mul.f16x2 r441, r91, r443;
}
{
add.f16x2 r444, r420, r441;
}
mov.f64 fd771, 0dBFD7B057F20BF2E4;
{
cvt.rn.f16.f64 rs35, fd771;
}
mov.b32 r449, {rs35, rs35};
{
mul.f16x2 r447, r97, r449;
}
{
add.f16x2 r450, r426, r447;
}
mov.f64 fd772, 0dBFEDBA2D62CB789F;
{
cvt.rn.f16.f64 rs36, fd772;
}
mov.b32 r455, {rs36, rs36};
{
mul.f16x2 r453, r106, r455;
}
{
add.f16x2 r456, r432, r453;
}
{
cvt.rn.f16.f64 rs37, fd771;
}
mov.b32 r461, {rs37, rs37};
{
mul.f16x2 r459, r100, r461;
}
{
add.f16x2 r462, r438, r459;
}
{
cvt.rn.f16.f64 rs38, fd772;
}
mov.b32 r467, {rs38, rs38};
{
mul.f16x2 r465, r103, r467;
}
{
add.f16x2 r468, r444, r465;
}
mov.f64 fd763, 0dBFE1F53E93956DBF;
{
cvt.rn.f16.f64 rs39, fd763;
}
mov.b32 r473, {rs39, rs39};
{
mul.f16x2 r471, r109, r473;
}
{
add.f16x2 r474, r450, r471;
}
mov.f64 fd764, 0dBFEA7C6DA34AF89F;
{
cvt.rn.f16.f64 rs40, fd764;
}
mov.b32 r479, {rs40, rs40};
{
mul.f16x2 r477, r118, r479;
}
{
add.f16x2 r480, r456, r477;
}
{
cvt.rn.f16.f64 rs41, fd763;
}
mov.b32 r485, {rs41, rs41};
{
mul.f16x2 r483, r112, r485;
}
{
add.f16x2 r486, r462, r483;
}
{
cvt.rn.f16.f64 rs42, fd764;
}
mov.b32 r491, {rs42, rs42};
{
mul.f16x2 r489, r115, r491;
}
{
add.f16x2 r492, r468, r489;
}
mov.f64 fd755, 0dBFE73B5AE5DB4E10;
{
cvt.rn.f16.f64 rs43, fd755;
}
mov.b32 r497, {rs43, rs43};
{
mul.f16x2 r495, r121, r497;
}
{
add.f16x2 r498, r474, r495;
}
mov.f64 fd756, 0dBFE601A24BA81342;
{
cvt.rn.f16.f64 rs44, fd756;
}
mov.b32 r503, {rs44, rs44};
{
mul.f16x2 r501, r130, r503;
}
{
add.f16x2 r504, r480, r501;
}
{
cvt.rn.f16.f64 rs45, fd755;
}
mov.b32 r509, {rs45, rs45};
{
mul.f16x2 r507, r124, r509;
}
{
add.f16x2 r510, r486, r507;
}
{
cvt.rn.f16.f64 rs46, fd756;
}
mov.b32 r515, {rs46, rs46};
{
mul.f16x2 r513, r127, r515;
}
{
add.f16x2 r516, r492, r513;
}
mov.f64 fd747, 0dBFEB6B5FBD9F7255;
{
cvt.rn.f16.f64 rs47, fd747;
}
mov.b32 r521, {rs47, rs47};
{
mul.f16x2 r519, r133, r521;
}
{
add.f16x2 r522, r498, r519;
}
mov.f64 fd748, 0dBFE07F6ACD7CDCE2;
{
cvt.rn.f16.f64 rs48, fd748;
}
mov.b32 r527, {rs48, rs48};
{
mul.f16x2 r525, r142, r527;
}
{
add.f16x2 r528, r504, r525;
}
{
cvt.rn.f16.f64 rs49, fd747;
}
mov.b32 r533, {rs49, rs49};
{
mul.f16x2 r531, r136, r533;
}
{
add.f16x2 r534, r510, r531;
}
{
cvt.rn.f16.f64 rs50, fd748;
}
mov.b32 r539, {rs50, rs50};
{
mul.f16x2 r537, r139, r539;
}
{
add.f16x2 r540, r516, r537;
}
mov.f64 fd739, 0dBFEE532CBE45C954;
{
cvt.rn.f16.f64 rs51, fd739;
}
mov.b32 r545, {rs51, rs51};
{
mul.f16x2 r543, r145, r545;
}
{
add.f16x2 r546, r522, r543;
}
mov.f64 fd740, 0dBFD46F6FAF5FCB72;
{
cvt.rn.f16.f64 rs52, fd740;
}
mov.b32 r551, {rs52, rs52};
{
mul.f16x2 r549, r154, r551;
}
{
add.f16x2 r552, r528, r549;
}
{
cvt.rn.f16.f64 rs53, fd739;
}
mov.b32 r557, {rs53, rs53};
{
mul.f16x2 r555, r148, r557;
}
{
add.f16x2 r558, r534, r555;
}
{
cvt.rn.f16.f64 rs54, fd740;
}
mov.b32 r563, {rs54, rs54};
{
mul.f16x2 r561, r151, r563;
}
{
add.f16x2 r564, r540, r561;
}
mov.f64 fd731, 0dBFEFCFFA67B61650;
{
cvt.rn.f16.f64 rs55, fd731;
}
mov.b32 r569, {rs55, rs55};
{
mul.f16x2 r567, r157, r569;
}
{
add.f16x2 r570, r546, r567;
}
mov.f64 fd732, 0dBFBBADB02034D9FF;
{
cvt.rn.f16.f64 rs56, fd732;
}
mov.b32 r575, {rs56, rs56};
{
mul.f16x2 r573, r166, r575;
}
{
add.f16x2 r576, r552, r573;
}
{
cvt.rn.f16.f64 rs57, fd731;
}
mov.b32 r581, {rs57, rs57};
{
mul.f16x2 r579, r160, r581;
}
{
add.f16x2 r582, r558, r579;
}
{
cvt.rn.f16.f64 rs58, fd732;
}
mov.b32 r587, {rs58, rs58};
{
mul.f16x2 r585, r163, r587;
}
{
add.f16x2 r588, r564, r585;
}
{
sub.f16x2 %2, r570, r576;
}
{
add.f16x2 %3, r582, r588;
}
{
add.f16x2 %56, r570, r576;
}
{
sub.f16x2 %57, r582, r588;
}
cvt.rn.f16.s32 rs59, r4804;
mov.b32 r615, {rs59, rs59};
cvt.rn.f16.s32 rs60, r4804;
mov.b32 r627, {rs60, rs60};
{
cvt.rn.f16.f64 rs61, fd743;
}
mov.b32 r607, {rs61, rs61};
{
mul.f16x2 r605, r1, r607;
}
{
add.f16x2 r608, %58, r605;
}
{
cvt.rn.f16.f64 rs62, fd636;
}
mov.b32 r613, {rs62, rs62};
{
mul.f16x2 r611, r10, r613;
}
{
add.f16x2 r614, r615, r611;
}
{
cvt.rn.f16.f64 rs63, fd743;
}
mov.b32 r619, {rs63, rs63};
{
mul.f16x2 r617, r4, r619;
}
{
add.f16x2 r620, %59, r617;
}
{
cvt.rn.f16.f64 rs64, fd636;
}
mov.b32 r625, {rs64, rs64};
{
mul.f16x2 r623, r7, r625;
}
{
add.f16x2 r626, r627, r623;
}
{
cvt.rn.f16.f64 rs65, fd759;
}
mov.b32 r631, {rs65, rs65};
{
mul.f16x2 r629, r13, r631;
}
{
add.f16x2 r632, r608, r629;
}
{
cvt.rn.f16.f64 rs66, fd700;
}
mov.b32 r637, {rs66, rs66};
{
mul.f16x2 r635, r22, r637;
}
{
add.f16x2 r638, r614, r635;
}
{
cvt.rn.f16.f64 rs67, fd759;
}
mov.b32 r643, {rs67, rs67};
{
mul.f16x2 r641, r16, r643;
}
{
add.f16x2 r644, r620, r641;
}
{
cvt.rn.f16.f64 rs68, fd700;
}
mov.b32 r649, {rs68, rs68};
{
mul.f16x2 r647, r19, r649;
}
{
add.f16x2 r650, r626, r647;
}
{
cvt.rn.f16.f64 rs69, fd775;
}
mov.b32 r655, {rs69, rs69};
{
mul.f16x2 r653, r25, r655;
}
{
add.f16x2 r656, r632, r653;
}
{
cvt.rn.f16.f64 rs70, fd424;
}
mov.b32 r661, {rs70, rs70};
{
mul.f16x2 r659, r34, r661;
}
{
add.f16x2 r662, r638, r659;
}
{
cvt.rn.f16.f64 rs71, fd775;
}
mov.b32 r667, {rs71, rs71};
{
mul.f16x2 r665, r28, r667;
}
{
add.f16x2 r668, r644, r665;
}
{
cvt.rn.f16.f64 rs72, fd424;
}
mov.b32 r673, {rs72, rs72};
{
mul.f16x2 r671, r31, r673;
}
{
add.f16x2 r674, r650, r671;
}
{
cvt.rn.f16.f64 rs73, fd779;
}
mov.b32 r679, {rs73, rs73};
{
mul.f16x2 r677, r37, r679;
}
{
add.f16x2 r680, r656, r677;
}
{
cvt.rn.f16.f64 rs74, fd780;
}
mov.b32 r685, {rs74, rs74};
{
mul.f16x2 r683, r46, r685;
}
{
add.f16x2 r686, r662, r683;
}
{
cvt.rn.f16.f64 rs75, fd779;
}
mov.b32 r691, {rs75, rs75};
{
mul.f16x2 r689, r40, r691;
}
{
add.f16x2 r692, r668, r689;
}
{
cvt.rn.f16.f64 rs76, fd780;
}
mov.b32 r697, {rs76, rs76};
{
mul.f16x2 r695, r43, r697;
}
{
add.f16x2 r698, r674, r695;
}
{
cvt.rn.f16.f64 rs77, fd763;
}
mov.b32 r703, {rs77, rs77};
{
mul.f16x2 r701, r49, r703;
}
{
add.f16x2 r704, r680, r701;
}
{
cvt.rn.f16.f64 rs78, fd764;
}
mov.b32 r709, {rs78, rs78};
{
mul.f16x2 r707, r58, r709;
}
{
add.f16x2 r710, r686, r707;
}
{
cvt.rn.f16.f64 rs79, fd763;
}
mov.b32 r715, {rs79, rs79};
{
mul.f16x2 r713, r52, r715;
}
{
add.f16x2 r716, r692, r713;
}
{
cvt.rn.f16.f64 rs80, fd764;
}
mov.b32 r721, {rs80, rs80};
{
mul.f16x2 r719, r55, r721;
}
{
add.f16x2 r722, r698, r719;
}
{
cvt.rn.f16.f64 rs81, fd747;
}
mov.b32 r727, {rs81, rs81};
{
mul.f16x2 r725, r61, r727;
}
{
add.f16x2 r728, r704, r725;
}
{
cvt.rn.f16.f64 rs82, fd748;
}
mov.b32 r733, {rs82, rs82};
{
mul.f16x2 r731, r70, r733;
}
{
add.f16x2 r734, r710, r731;
}
{
cvt.rn.f16.f64 rs83, fd747;
}
mov.b32 r739, {rs83, rs83};
{
mul.f16x2 r737, r64, r739;
}
{
add.f16x2 r740, r716, r737;
}
{
cvt.rn.f16.f64 rs84, fd748;
}
mov.b32 r745, {rs84, rs84};
{
mul.f16x2 r743, r67, r745;
}
{
add.f16x2 r746, r722, r743;
}
{
cvt.rn.f16.f64 rs85, fd731;
}
mov.b32 r751, {rs85, rs85};
{
mul.f16x2 r749, r73, r751;
}
{
add.f16x2 r752, r728, r749;
}
{
cvt.rn.f16.f64 rs86, fd732;
}
mov.b32 r757, {rs86, rs86};
{
mul.f16x2 r755, r82, r757;
}
{
add.f16x2 r758, r734, r755;
}
{
cvt.rn.f16.f64 rs87, fd731;
}
mov.b32 r763, {rs87, rs87};
{
mul.f16x2 r761, r76, r763;
}
{
add.f16x2 r764, r740, r761;
}
{
cvt.rn.f16.f64 rs88, fd732;
}
mov.b32 r769, {rs88, rs88};
{
mul.f16x2 r767, r79, r769;
}
{
add.f16x2 r770, r746, r767;
}
{
cvt.rn.f16.f64 rs89, fd739;
}
mov.b32 r775, {rs89, rs89};
{
mul.f16x2 r773, r85, r775;
}
{
add.f16x2 r776, r752, r773;
}
mov.f64 fd660, 0d3FD46F6FAF5FCB72;
{
cvt.rn.f16.f64 rs90, fd660;
}
mov.b32 r781, {rs90, rs90};
{
mul.f16x2 r779, r94, r781;
}
{
add.f16x2 r782, r758, r779;
}
{
cvt.rn.f16.f64 rs91, fd739;
}
mov.b32 r787, {rs91, rs91};
{
mul.f16x2 r785, r88, r787;
}
{
add.f16x2 r788, r764, r785;
}
{
cvt.rn.f16.f64 rs92, fd660;
}
mov.b32 r793, {rs92, rs92};
{
mul.f16x2 r791, r91, r793;
}
{
add.f16x2 r794, r770, r791;
}
{
cvt.rn.f16.f64 rs93, fd755;
}
mov.b32 r799, {rs93, rs93};
{
mul.f16x2 r797, r97, r799;
}
{
add.f16x2 r800, r776, r797;
}
mov.f64 fd456, 0d3FE601A24BA81342;
{
cvt.rn.f16.f64 rs94, fd456;
}
mov.b32 r805, {rs94, rs94};
{
mul.f16x2 r803, r106, r805;
}
{
add.f16x2 r806, r782, r803;
}
{
cvt.rn.f16.f64 rs95, fd755;
}
mov.b32 r811, {rs95, rs95};
{
mul.f16x2 r809, r100, r811;
}
{
add.f16x2 r812, r788, r809;
}
{
cvt.rn.f16.f64 rs96, fd456;
}
mov.b32 r817, {rs96, rs96};
{
mul.f16x2 r815, r103, r817;
}
{
add.f16x2 r818, r794, r815;
}
{
cvt.rn.f16.f64 rs97, fd771;
}
mov.b32 r823, {rs97, rs97};
{
mul.f16x2 r821, r109, r823;
}
{
add.f16x2 r824, r800, r821;
}
mov.f64 fd696, 0d3FEDBA2D62CB789F;
{
cvt.rn.f16.f64 rs98, fd696;
}
mov.b32 r829, {rs98, rs98};
{
mul.f16x2 r827, r118, r829;
}
{
add.f16x2 r830, r806, r827;
}
{
cvt.rn.f16.f64 rs99, fd771;
}
mov.b32 r835, {rs99, rs99};
{
mul.f16x2 r833, r112, r835;
}
{
add.f16x2 r836, r812, r833;
}
{
cvt.rn.f16.f64 rs100, fd696;
}
mov.b32 r841, {rs100, rs100};
{
mul.f16x2 r839, r115, r841;
}
{
add.f16x2 r842, r818, r839;
}
{
cvt.rn.f16.f64 rs101, fd783;
}
mov.b32 r847, {rs101, rs101};
{
mul.f16x2 r845, r121, r847;
}
{
add.f16x2 r848, r824, r845;
}
mov.f64 fd784, 0d3FEFF3FC588E859D;
{
cvt.rn.f16.f64 rs102, fd784;
}
mov.b32 r853, {rs102, rs102};
{
mul.f16x2 r851, r130, r853;
}
{
add.f16x2 r854, r830, r851;
}
{
cvt.rn.f16.f64 rs103, fd783;
}
mov.b32 r859, {rs103, rs103};
{
mul.f16x2 r857, r124, r859;
}
{
add.f16x2 r860, r836, r857;
}
{
cvt.rn.f16.f64 rs104, fd784;
}
mov.b32 r865, {rs104, rs104};
{
mul.f16x2 r863, r127, r865;
}
{
add.f16x2 r866, r842, r863;
}
{
cvt.rn.f16.f64 rs105, fd767;
}
mov.b32 r871, {rs105, rs105};
{
mul.f16x2 r869, r133, r871;
}
{
add.f16x2 r872, r848, r869;
}
mov.f64 fd768, 0d3FEC45BB0D10918C;
{
cvt.rn.f16.f64 rs106, fd768;
}
mov.b32 r877, {rs106, rs106};
{
mul.f16x2 r875, r142, r877;
}
{
add.f16x2 r878, r854, r875;
}
{
cvt.rn.f16.f64 rs107, fd767;
}
mov.b32 r883, {rs107, rs107};
{
mul.f16x2 r881, r136, r883;
}
{
add.f16x2 r884, r860, r881;
}
{
cvt.rn.f16.f64 rs108, fd768;
}
mov.b32 r889, {rs108, rs108};
{
mul.f16x2 r887, r139, r889;
}
{
add.f16x2 r890, r866, r887;
}
{
cvt.rn.f16.f64 rs109, fd751;
}
mov.b32 r895, {rs109, rs109};
{
mul.f16x2 r893, r145, r895;
}
{
add.f16x2 r896, r872, r893;
}
mov.f64 fd752, 0d3FE35D9650D47852;
{
cvt.rn.f16.f64 rs110, fd752;
}
mov.b32 r901, {rs110, rs110};
{
mul.f16x2 r899, r154, r901;
}
{
add.f16x2 r902, r878, r899;
}
{
cvt.rn.f16.f64 rs111, fd751;
}
mov.b32 r907, {rs111, rs111};
{
mul.f16x2 r905, r148, r907;
}
{
add.f16x2 r908, r884, r905;
}
{
cvt.rn.f16.f64 rs112, fd752;
}
mov.b32 r913, {rs112, rs112};
{
mul.f16x2 r911, r151, r913;
}
{
add.f16x2 r914, r890, r911;
}
{
cvt.rn.f16.f64 rs113, fd735;
}
mov.b32 r919, {rs113, rs113};
{
mul.f16x2 r917, r157, r919;
}
{
add.f16x2 r920, r896, r917;
}
mov.f64 fd736, 0d3FCB8426C12812BC;
{
cvt.rn.f16.f64 rs114, fd736;
}
mov.b32 r925, {rs114, rs114};
{
mul.f16x2 r923, r166, r925;
}
{
add.f16x2 r926, r902, r923;
}
{
cvt.rn.f16.f64 rs115, fd735;
}
mov.b32 r931, {rs115, rs115};
{
mul.f16x2 r929, r160, r931;
}
{
add.f16x2 r932, r908, r929;
}
{
cvt.rn.f16.f64 rs116, fd736;
}
mov.b32 r937, {rs116, rs116};
{
mul.f16x2 r935, r163, r937;
}
{
add.f16x2 r938, r914, r935;
}
{
sub.f16x2 %4, r920, r926;
}
{
add.f16x2 %5, r932, r938;
}
{
add.f16x2 %54, r920, r926;
}
{
sub.f16x2 %55, r932, r938;
}
cvt.rn.f16.s32 rs117, r4804;
mov.b32 r965, {rs117, rs117};
cvt.rn.f16.s32 rs118, r4804;
mov.b32 r977, {rs118, rs118};
{
cvt.rn.f16.f64 rs119, fd751;
}
mov.b32 r957, {rs119, rs119};
{
mul.f16x2 r955, r1, r957;
}
{
add.f16x2 r958, %58, r955;
}
{
cvt.rn.f16.f64 rs120, fd540;
}
mov.b32 r963, {rs120, rs120};
{
mul.f16x2 r961, r10, r963;
}
{
add.f16x2 r964, r965, r961;
}
{
cvt.rn.f16.f64 rs121, fd751;
}
mov.b32 r969, {rs121, rs121};
{
mul.f16x2 r967, r4, r969;
}
{
add.f16x2 r970, %59, r967;
}
{
cvt.rn.f16.f64 rs122, fd540;
}
mov.b32 r975, {rs122, rs122};
{
mul.f16x2 r973, r7, r975;
}
{
add.f16x2 r976, r977, r973;
}
{
cvt.rn.f16.f64 rs123, fd775;
}
mov.b32 r981, {rs123, rs123};
{
mul.f16x2 r979, r13, r981;
}
{
add.f16x2 r982, r958, r979;
}
{
cvt.rn.f16.f64 rs124, fd424;
}
mov.b32 r987, {rs124, rs124};
{
mul.f16x2 r985, r22, r987;
}
{
add.f16x2 r988, r964, r985;
}
{
cvt.rn.f16.f64 rs125, fd775;
}
mov.b32 r993, {rs125, rs125};
{
mul.f16x2 r991, r16, r993;
}
{
add.f16x2 r994, r970, r991;
}
{
cvt.rn.f16.f64 rs126, fd424;
}
mov.b32 r999, {rs126, rs126};
{
mul.f16x2 r997, r19, r999;
}
{
add.f16x2 r1000, r976, r997;
}
{
cvt.rn.f16.f64 rs127, fd771;
}
mov.b32 r1005, {rs127, rs127};
{
mul.f16x2 r1003, r25, r1005;
}
{
add.f16x2 r1006, r982, r1003;
}
{
cvt.rn.f16.f64 rs128, fd772;
}
mov.b32 r1011, {rs128, rs128};
{
mul.f16x2 r1009, r34, r1011;
}
{
add.f16x2 r1012, r988, r1009;
}
{
cvt.rn.f16.f64 rs129, fd771;
}
mov.b32 r1017, {rs129, rs129};
{
mul.f16x2 r1015, r28, r1017;
}
{
add.f16x2 r1018, r994, r1015;
}
{
cvt.rn.f16.f64 rs130, fd772;
}
mov.b32 r1023, {rs130, rs130};
{
mul.f16x2 r1021, r31, r1023;
}
{
add.f16x2 r1024, r1000, r1021;
}
{
cvt.rn.f16.f64 rs131, fd747;
}
mov.b32 r1029, {rs131, rs131};
{
mul.f16x2 r1027, r37, r1029;
}
{
add.f16x2 r1030, r1006, r1027;
}
{
cvt.rn.f16.f64 rs132, fd748;
}
mov.b32 r1035, {rs132, rs132};
{
mul.f16x2 r1033, r46, r1035;
}
{
add.f16x2 r1036, r1012, r1033;
}
{
cvt.rn.f16.f64 rs133, fd747;
}
mov.b32 r1041, {rs133, rs133};
{
mul.f16x2 r1039, r40, r1041;
}
{
add.f16x2 r1042, r1018, r1039;
}
{
cvt.rn.f16.f64 rs134, fd748;
}
mov.b32 r1047, {rs134, rs134};
{
mul.f16x2 r1045, r43, r1047;
}
{
add.f16x2 r1048, r1024, r1045;
}
{
cvt.rn.f16.f64 rs135, fd731;
}
mov.b32 r1053, {rs135, rs135};
{
mul.f16x2 r1051, r49, r1053;
}
{
add.f16x2 r1054, r1030, r1051;
}
mov.f64 fd576, 0d3FBBADB02034D9FF;
{
cvt.rn.f16.f64 rs136, fd576;
}
mov.b32 r1059, {rs136, rs136};
{
mul.f16x2 r1057, r58, r1059;
}
{
add.f16x2 r1060, r1036, r1057;
}
{
cvt.rn.f16.f64 rs137, fd731;
}
mov.b32 r1065, {rs137, rs137};
{
mul.f16x2 r1063, r52, r1065;
}
{
add.f16x2 r1066, r1042, r1063;
}
{
cvt.rn.f16.f64 rs138, fd576;
}
mov.b32 r1071, {rs138, rs138};
{
mul.f16x2 r1069, r55, r1071;
}
{
add.f16x2 r1072, r1048, r1069;
}
{
cvt.rn.f16.f64 rs139, fd755;
}
mov.b32 r1077, {rs139, rs139};
{
mul.f16x2 r1075, r61, r1077;
}
{
add.f16x2 r1078, r1054, r1075;
}
{
cvt.rn.f16.f64 rs140, fd456;
}
mov.b32 r1083, {rs140, rs140};
{
mul.f16x2 r1081, r70, r1083;
}
{
add.f16x2 r1084, r1060, r1081;
}
{
cvt.rn.f16.f64 rs141, fd755;
}
mov.b32 r1089, {rs141, rs141};
{
mul.f16x2 r1087, r64, r1089;
}
{
add.f16x2 r1090, r1066, r1087;
}
{
cvt.rn.f16.f64 rs142, fd456;
}
mov.b32 r1095, {rs142, rs142};
{
mul.f16x2 r1093, r67, r1095;
}
{
add.f16x2 r1096, r1072, r1093;
}
{
cvt.rn.f16.f64 rs143, fd779;
}
mov.b32 r1101, {rs143, rs143};
{
mul.f16x2 r1099, r73, r1101;
}
{
add.f16x2 r1102, r1078, r1099;
}
mov.f64 fd652, 0d3FEF941537248537;
{
cvt.rn.f16.f64 rs144, fd652;
}
mov.b32 r1107, {rs144, rs144};
{
mul.f16x2 r1105, r82, r1107;
}
{
add.f16x2 r1108, r1084, r1105;
}
{
cvt.rn.f16.f64 rs145, fd779;
}
mov.b32 r1113, {rs145, rs145};
{
mul.f16x2 r1111, r76, r1113;
}
{
add.f16x2 r1114, r1090, r1111;
}
{
cvt.rn.f16.f64 rs146, fd652;
}
mov.b32 r1119, {rs146, rs146};
{
mul.f16x2 r1117, r79, r1119;
}
{
add.f16x2 r1120, r1096, r1117;
}
{
cvt.rn.f16.f64 rs147, fd767;
}
mov.b32 r1125, {rs147, rs147};
{
mul.f16x2 r1123, r85, r1125;
}
{
add.f16x2 r1126, r1102, r1123;
}
{
cvt.rn.f16.f64 rs148, fd768;
}
mov.b32 r1131, {rs148, rs148};
{
mul.f16x2 r1129, r94, r1131;
}
{
add.f16x2 r1132, r1108, r1129;
}
{
cvt.rn.f16.f64 rs149, fd767;
}
mov.b32 r1137, {rs149, rs149};
{
mul.f16x2 r1135, r88, r1137;
}
{
add.f16x2 r1138, r1114, r1135;
}
{
cvt.rn.f16.f64 rs150, fd768;
}
mov.b32 r1143, {rs150, rs150};
{
mul.f16x2 r1141, r91, r1143;
}
{
add.f16x2 r1144, r1120, r1141;
}
{
cvt.rn.f16.f64 rs151, fd743;
}
mov.b32 r1149, {rs151, rs151};
{
mul.f16x2 r1147, r97, r1149;
}
{
add.f16x2 r1150, r1126, r1147;
}
mov.f64 fd744, 0d3FDADF7689C97B70;
{
cvt.rn.f16.f64 rs152, fd744;
}
mov.b32 r1155, {rs152, rs152};
{
mul.f16x2 r1153, r106, r1155;
}
{
add.f16x2 r1156, r1132, r1153;
}
{
cvt.rn.f16.f64 rs153, fd743;
}
mov.b32 r1161, {rs153, rs153};
{
mul.f16x2 r1159, r100, r1161;
}
{
add.f16x2 r1162, r1138, r1159;
}
{
cvt.rn.f16.f64 rs154, fd744;
}
mov.b32 r1167, {rs154, rs154};
{
mul.f16x2 r1165, r103, r1167;
}
{
add.f16x2 r1168, r1144, r1165;
}
{
cvt.rn.f16.f64 rs155, fd735;
}
mov.b32 r1173, {rs155, rs155};
{
mul.f16x2 r1171, r109, r1173;
}
{
add.f16x2 r1174, r1150, r1171;
}
{
cvt.rn.f16.f64 rs156, fd708;
}
mov.b32 r1179, {rs156, rs156};
{
mul.f16x2 r1177, r118, r1179;
}
{
add.f16x2 r1180, r1156, r1177;
}
{
cvt.rn.f16.f64 rs157, fd735;
}
mov.b32 r1185, {rs157, rs157};
{
mul.f16x2 r1183, r112, r1185;
}
{
add.f16x2 r1186, r1162, r1183;
}
{
cvt.rn.f16.f64 rs158, fd708;
}
mov.b32 r1191, {rs158, rs158};
{
mul.f16x2 r1189, r115, r1191;
}
{
add.f16x2 r1192, r1168, r1189;
}
{
cvt.rn.f16.f64 rs159, fd759;
}
mov.b32 r1197, {rs159, rs159};
{
mul.f16x2 r1195, r121, r1197;
}
{
add.f16x2 r1198, r1174, r1195;
}
{
cvt.rn.f16.f64 rs160, fd700;
}
mov.b32 r1203, {rs160, rs160};
{
mul.f16x2 r1201, r130, r1203;
}
{
add.f16x2 r1204, r1180, r1201;
}
{
cvt.rn.f16.f64 rs161, fd759;
}
mov.b32 r1209, {rs161, rs161};
{
mul.f16x2 r1207, r124, r1209;
}
{
add.f16x2 r1210, r1186, r1207;
}
{
cvt.rn.f16.f64 rs162, fd700;
}
mov.b32 r1215, {rs162, rs162};
{
mul.f16x2 r1213, r127, r1215;
}
{
add.f16x2 r1216, r1192, r1213;
}
{
cvt.rn.f16.f64 rs163, fd783;
}
mov.b32 r1221, {rs163, rs163};
{
mul.f16x2 r1219, r133, r1221;
}
{
add.f16x2 r1222, r1198, r1219;
}
{
cvt.rn.f16.f64 rs164, fd692;
}
mov.b32 r1227, {rs164, rs164};
{
mul.f16x2 r1225, r142, r1227;
}
{
add.f16x2 r1228, r1204, r1225;
}
{
cvt.rn.f16.f64 rs165, fd783;
}
mov.b32 r1233, {rs165, rs165};
{
mul.f16x2 r1231, r136, r1233;
}
{
add.f16x2 r1234, r1210, r1231;
}
{
cvt.rn.f16.f64 rs166, fd692;
}
mov.b32 r1239, {rs166, rs166};
{
mul.f16x2 r1237, r139, r1239;
}
{
add.f16x2 r1240, r1216, r1237;
}
{
cvt.rn.f16.f64 rs167, fd763;
}
mov.b32 r1245, {rs167, rs167};
{
mul.f16x2 r1243, r145, r1245;
}
{
add.f16x2 r1246, r1222, r1243;
}
{
cvt.rn.f16.f64 rs168, fd764;
}
mov.b32 r1251, {rs168, rs168};
{
mul.f16x2 r1249, r154, r1251;
}
{
add.f16x2 r1252, r1228, r1249;
}
{
cvt.rn.f16.f64 rs169, fd763;
}
mov.b32 r1257, {rs169, rs169};
{
mul.f16x2 r1255, r148, r1257;
}
{
add.f16x2 r1258, r1234, r1255;
}
{
cvt.rn.f16.f64 rs170, fd764;
}
mov.b32 r1263, {rs170, rs170};
{
mul.f16x2 r1261, r151, r1263;
}
{
add.f16x2 r1264, r1240, r1261;
}
{
cvt.rn.f16.f64 rs171, fd739;
}
mov.b32 r1269, {rs171, rs171};
{
mul.f16x2 r1267, r157, r1269;
}
{
add.f16x2 r1270, r1246, r1267;
}
{
cvt.rn.f16.f64 rs172, fd740;
}
mov.b32 r1275, {rs172, rs172};
{
mul.f16x2 r1273, r166, r1275;
}
{
add.f16x2 r1276, r1252, r1273;
}
{
cvt.rn.f16.f64 rs173, fd739;
}
mov.b32 r1281, {rs173, rs173};
{
mul.f16x2 r1279, r160, r1281;
}
{
add.f16x2 r1282, r1258, r1279;
}
{
cvt.rn.f16.f64 rs174, fd740;
}
mov.b32 r1287, {rs174, rs174};
{
mul.f16x2 r1285, r163, r1287;
}
{
add.f16x2 r1288, r1264, r1285;
}
{
sub.f16x2 %6, r1270, r1276;
}
{
add.f16x2 %7, r1282, r1288;
}
{
add.f16x2 %52, r1270, r1276;
}
{
sub.f16x2 %53, r1282, r1288;
}
cvt.rn.f16.s32 rs175, r4804;
mov.b32 r1315, {rs175, rs175};
cvt.rn.f16.s32 rs176, r4804;
mov.b32 r1327, {rs176, rs176};
{
cvt.rn.f16.f64 rs177, fd759;
}
mov.b32 r1307, {rs177, rs177};
{
mul.f16x2 r1305, r1, r1307;
}
{
add.f16x2 r1308, %58, r1305;
}
{
cvt.rn.f16.f64 rs178, fd700;
}
mov.b32 r1313, {rs178, rs178};
{
mul.f16x2 r1311, r10, r1313;
}
{
add.f16x2 r1314, r1315, r1311;
}
{
cvt.rn.f16.f64 rs179, fd759;
}
mov.b32 r1319, {rs179, rs179};
{
mul.f16x2 r1317, r4, r1319;
}
{
add.f16x2 r1320, %59, r1317;
}
{
cvt.rn.f16.f64 rs180, fd700;
}
mov.b32 r1325, {rs180, rs180};
{
mul.f16x2 r1323, r7, r1325;
}
{
add.f16x2 r1326, r1327, r1323;
}
{
cvt.rn.f16.f64 rs181, fd779;
}
mov.b32 r1331, {rs181, rs181};
{
mul.f16x2 r1329, r13, r1331;
}
{
add.f16x2 r1332, r1308, r1329;
}
{
cvt.rn.f16.f64 rs182, fd780;
}
mov.b32 r1337, {rs182, rs182};
{
mul.f16x2 r1335, r22, r1337;
}
{
add.f16x2 r1338, r1314, r1335;
}
{
cvt.rn.f16.f64 rs183, fd779;
}
mov.b32 r1343, {rs183, rs183};
{
mul.f16x2 r1341, r16, r1343;
}
{
add.f16x2 r1344, r1320, r1341;
}
{
cvt.rn.f16.f64 rs184, fd780;
}
mov.b32 r1349, {rs184, rs184};
{
mul.f16x2 r1347, r19, r1349;
}
{
add.f16x2 r1350, r1326, r1347;
}
{
cvt.rn.f16.f64 rs185, fd747;
}
mov.b32 r1355, {rs185, rs185};
{
mul.f16x2 r1353, r25, r1355;
}
{
add.f16x2 r1356, r1332, r1353;
}
{
cvt.rn.f16.f64 rs186, fd748;
}
mov.b32 r1361, {rs186, rs186};
{
mul.f16x2 r1359, r34, r1361;
}
{
add.f16x2 r1362, r1338, r1359;
}
{
cvt.rn.f16.f64 rs187, fd747;
}
mov.b32 r1367, {rs187, rs187};
{
mul.f16x2 r1365, r28, r1367;
}
{
add.f16x2 r1368, r1344, r1365;
}
{
cvt.rn.f16.f64 rs188, fd748;
}
mov.b32 r1373, {rs188, rs188};
{
mul.f16x2 r1371, r31, r1373;
}
{
add.f16x2 r1374, r1350, r1371;
}
{
cvt.rn.f16.f64 rs189, fd739;
}
mov.b32 r1379, {rs189, rs189};
{
mul.f16x2 r1377, r37, r1379;
}
{
add.f16x2 r1380, r1356, r1377;
}
{
cvt.rn.f16.f64 rs190, fd660;
}
mov.b32 r1385, {rs190, rs190};
{
mul.f16x2 r1383, r46, r1385;
}
{
add.f16x2 r1386, r1362, r1383;
}
{
cvt.rn.f16.f64 rs191, fd739;
}
mov.b32 r1391, {rs191, rs191};
{
mul.f16x2 r1389, r40, r1391;
}
{
add.f16x2 r1392, r1368, r1389;
}
{
cvt.rn.f16.f64 rs192, fd660;
}
mov.b32 r1397, {rs192, rs192};
{
mul.f16x2 r1395, r43, r1397;
}
{
add.f16x2 r1398, r1374, r1395;
}
{
cvt.rn.f16.f64 rs193, fd771;
}
mov.b32 r1403, {rs193, rs193};
{
mul.f16x2 r1401, r49, r1403;
}
{
add.f16x2 r1404, r1380, r1401;
}
{
cvt.rn.f16.f64 rs194, fd696;
}
mov.b32 r1409, {rs194, rs194};
{
mul.f16x2 r1407, r58, r1409;
}
{
add.f16x2 r1410, r1386, r1407;
}
{
cvt.rn.f16.f64 rs195, fd771;
}
mov.b32 r1415, {rs195, rs195};
{
mul.f16x2 r1413, r52, r1415;
}
{
add.f16x2 r1416, r1392, r1413;
}
{
cvt.rn.f16.f64 rs196, fd696;
}
mov.b32 r1421, {rs196, rs196};
{
mul.f16x2 r1419, r55, r1421;
}
{
add.f16x2 r1422, r1398, r1419;
}
{
cvt.rn.f16.f64 rs197, fd767;
}
mov.b32 r1427, {rs197, rs197};
{
mul.f16x2 r1425, r61, r1427;
}
{
add.f16x2 r1428, r1404, r1425;
}
{
cvt.rn.f16.f64 rs198, fd768;
}
mov.b32 r1433, {rs198, rs198};
{
mul.f16x2 r1431, r70, r1433;
}
{
add.f16x2 r1434, r1410, r1431;
}
{
cvt.rn.f16.f64 rs199, fd767;
}
mov.b32 r1439, {rs199, rs199};
{
mul.f16x2 r1437, r64, r1439;
}
{
add.f16x2 r1440, r1416, r1437;
}
{
cvt.rn.f16.f64 rs200, fd768;
}
mov.b32 r1445, {rs200, rs200};
{
mul.f16x2 r1443, r67, r1445;
}
{
add.f16x2 r1446, r1422, r1443;
}
{
cvt.rn.f16.f64 rs201, fd735;
}
mov.b32 r1451, {rs201, rs201};
{
mul.f16x2 r1449, r73, r1451;
}
{
add.f16x2 r1452, r1428, r1449;
}
{
cvt.rn.f16.f64 rs202, fd736;
}
mov.b32 r1457, {rs202, rs202};
{
mul.f16x2 r1455, r82, r1457;
}
{
add.f16x2 r1458, r1434, r1455;
}
{
cvt.rn.f16.f64 rs203, fd735;
}
mov.b32 r1463, {rs203, rs203};
{
mul.f16x2 r1461, r76, r1463;
}
{
add.f16x2 r1464, r1440, r1461;
}
{
cvt.rn.f16.f64 rs204, fd736;
}
mov.b32 r1469, {rs204, rs204};
{
mul.f16x2 r1467, r79, r1469;
}
{
add.f16x2 r1470, r1446, r1467;
}
{
cvt.rn.f16.f64 rs205, fd751;
}
mov.b32 r1475, {rs205, rs205};
{
mul.f16x2 r1473, r85, r1475;
}
{
add.f16x2 r1476, r1452, r1473;
}
{
cvt.rn.f16.f64 rs206, fd540;
}
mov.b32 r1481, {rs206, rs206};
{
mul.f16x2 r1479, r94, r1481;
}
{
add.f16x2 r1482, r1458, r1479;
}
{
cvt.rn.f16.f64 rs207, fd751;
}
mov.b32 r1487, {rs207, rs207};
{
mul.f16x2 r1485, r88, r1487;
}
{
add.f16x2 r1488, r1464, r1485;
}
{
cvt.rn.f16.f64 rs208, fd540;
}
mov.b32 r1493, {rs208, rs208};
{
mul.f16x2 r1491, r91, r1493;
}
{
add.f16x2 r1494, r1470, r1491;
}
{
cvt.rn.f16.f64 rs209, fd783;
}
mov.b32 r1499, {rs209, rs209};
{
mul.f16x2 r1497, r97, r1499;
}
{
add.f16x2 r1500, r1476, r1497;
}
{
cvt.rn.f16.f64 rs210, fd692;
}
mov.b32 r1505, {rs210, rs210};
{
mul.f16x2 r1503, r106, r1505;
}
{
add.f16x2 r1506, r1482, r1503;
}
{
cvt.rn.f16.f64 rs211, fd783;
}
mov.b32 r1511, {rs211, rs211};
{
mul.f16x2 r1509, r100, r1511;
}
{
add.f16x2 r1512, r1488, r1509;
}
{
cvt.rn.f16.f64 rs212, fd692;
}
mov.b32 r1517, {rs212, rs212};
{
mul.f16x2 r1515, r103, r1517;
}
{
add.f16x2 r1518, r1494, r1515;
}
{
cvt.rn.f16.f64 rs213, fd755;
}
mov.b32 r1523, {rs213, rs213};
{
mul.f16x2 r1521, r109, r1523;
}
{
add.f16x2 r1524, r1500, r1521;
}
{
cvt.rn.f16.f64 rs214, fd756;
}
mov.b32 r1529, {rs214, rs214};
{
mul.f16x2 r1527, r118, r1529;
}
{
add.f16x2 r1530, r1506, r1527;
}
{
cvt.rn.f16.f64 rs215, fd755;
}
mov.b32 r1535, {rs215, rs215};
{
mul.f16x2 r1533, r112, r1535;
}
{
add.f16x2 r1536, r1512, r1533;
}
{
cvt.rn.f16.f64 rs216, fd756;
}
mov.b32 r1541, {rs216, rs216};
{
mul.f16x2 r1539, r115, r1541;
}
{
add.f16x2 r1542, r1518, r1539;
}
{
cvt.rn.f16.f64 rs217, fd731;
}
mov.b32 r1547, {rs217, rs217};
{
mul.f16x2 r1545, r121, r1547;
}
{
add.f16x2 r1548, r1524, r1545;
}
{
cvt.rn.f16.f64 rs218, fd576;
}
mov.b32 r1553, {rs218, rs218};
{
mul.f16x2 r1551, r130, r1553;
}
{
add.f16x2 r1554, r1530, r1551;
}
{
cvt.rn.f16.f64 rs219, fd731;
}
mov.b32 r1559, {rs219, rs219};
{
mul.f16x2 r1557, r124, r1559;
}
{
add.f16x2 r1560, r1536, r1557;
}
{
cvt.rn.f16.f64 rs220, fd576;
}
mov.b32 r1565, {rs220, rs220};
{
mul.f16x2 r1563, r127, r1565;
}
{
add.f16x2 r1566, r1542, r1563;
}
{
cvt.rn.f16.f64 rs221, fd763;
}
mov.b32 r1571, {rs221, rs221};
{
mul.f16x2 r1569, r133, r1571;
}
{
add.f16x2 r1572, r1548, r1569;
}
mov.f64 fd632, 0d3FEA7C6DA34AF89F;
{
cvt.rn.f16.f64 rs222, fd632;
}
mov.b32 r1577, {rs222, rs222};
{
mul.f16x2 r1575, r142, r1577;
}
{
add.f16x2 r1578, r1554, r1575;
}
{
cvt.rn.f16.f64 rs223, fd763;
}
mov.b32 r1583, {rs223, rs223};
{
mul.f16x2 r1581, r136, r1583;
}
{
add.f16x2 r1584, r1560, r1581;
}
{
cvt.rn.f16.f64 rs224, fd632;
}
mov.b32 r1589, {rs224, rs224};
{
mul.f16x2 r1587, r139, r1589;
}
{
add.f16x2 r1590, r1566, r1587;
}
{
cvt.rn.f16.f64 rs225, fd775;
}
mov.b32 r1595, {rs225, rs225};
{
mul.f16x2 r1593, r145, r1595;
}
{
add.f16x2 r1596, r1572, r1593;
}
mov.f64 fd776, 0d3FEED566CB3DCBA1;
{
cvt.rn.f16.f64 rs226, fd776;
}
mov.b32 r1601, {rs226, rs226};
{
mul.f16x2 r1599, r154, r1601;
}
{
add.f16x2 r1602, r1578, r1599;
}
{
cvt.rn.f16.f64 rs227, fd775;
}
mov.b32 r1607, {rs227, rs227};
{
mul.f16x2 r1605, r148, r1607;
}
{
add.f16x2 r1608, r1584, r1605;
}
{
cvt.rn.f16.f64 rs228, fd776;
}
mov.b32 r1613, {rs228, rs228};
{
mul.f16x2 r1611, r151, r1613;
}
{
add.f16x2 r1614, r1590, r1611;
}
{
cvt.rn.f16.f64 rs229, fd743;
}
mov.b32 r1619, {rs229, rs229};
{
mul.f16x2 r1617, r157, r1619;
}
{
add.f16x2 r1620, r1596, r1617;
}
{
cvt.rn.f16.f64 rs230, fd744;
}
mov.b32 r1625, {rs230, rs230};
{
mul.f16x2 r1623, r166, r1625;
}
{
add.f16x2 r1626, r1602, r1623;
}
{
cvt.rn.f16.f64 rs231, fd743;
}
mov.b32 r1631, {rs231, rs231};
{
mul.f16x2 r1629, r160, r1631;
}
{
add.f16x2 r1632, r1608, r1629;
}
{
cvt.rn.f16.f64 rs232, fd744;
}
mov.b32 r1637, {rs232, rs232};
{
mul.f16x2 r1635, r163, r1637;
}
{
add.f16x2 r1638, r1614, r1635;
}
{
sub.f16x2 %8, r1620, r1626;
}
{
add.f16x2 %9, r1632, r1638;
}
{
add.f16x2 %50, r1620, r1626;
}
{
sub.f16x2 %51, r1632, r1638;
}
cvt.rn.f16.s32 rs233, r4804;
mov.b32 r1665, {rs233, rs233};
cvt.rn.f16.s32 rs234, r4804;
mov.b32 r1677, {rs234, rs234};
{
cvt.rn.f16.f64 rs235, fd767;
}
mov.b32 r1657, {rs235, rs235};
{
mul.f16x2 r1655, r1, r1657;
}
{
add.f16x2 r1658, %58, r1655;
}
{
cvt.rn.f16.f64 rs236, fd604;
}
mov.b32 r1663, {rs236, rs236};
{
mul.f16x2 r1661, r10, r1663;
}
{
add.f16x2 r1664, r1665, r1661;
}
{
cvt.rn.f16.f64 rs237, fd767;
}
mov.b32 r1669, {rs237, rs237};
{
mul.f16x2 r1667, r4, r1669;
}
{
add.f16x2 r1670, %59, r1667;
}
{
cvt.rn.f16.f64 rs238, fd604;
}
mov.b32 r1675, {rs238, rs238};
{
mul.f16x2 r1673, r7, r1675;
}
{
add.f16x2 r1676, r1677, r1673;
}
{
cvt.rn.f16.f64 rs239, fd763;
}
mov.b32 r1681, {rs239, rs239};
{
mul.f16x2 r1679, r13, r1681;
}
{
add.f16x2 r1682, r1658, r1679;
}
{
cvt.rn.f16.f64 rs240, fd764;
}
mov.b32 r1687, {rs240, rs240};
{
mul.f16x2 r1685, r22, r1687;
}
{
add.f16x2 r1688, r1664, r1685;
}
{
cvt.rn.f16.f64 rs241, fd763;
}
mov.b32 r1693, {rs241, rs241};
{
mul.f16x2 r1691, r16, r1693;
}
{
add.f16x2 r1694, r1670, r1691;
}
{
cvt.rn.f16.f64 rs242, fd764;
}
mov.b32 r1699, {rs242, rs242};
{
mul.f16x2 r1697, r19, r1699;
}
{
add.f16x2 r1700, r1676, r1697;
}
{
cvt.rn.f16.f64 rs243, fd731;
}
mov.b32 r1705, {rs243, rs243};
{
mul.f16x2 r1703, r25, r1705;
}
{
add.f16x2 r1706, r1682, r1703;
}
{
cvt.rn.f16.f64 rs244, fd576;
}
mov.b32 r1711, {rs244, rs244};
{
mul.f16x2 r1709, r34, r1711;
}
{
add.f16x2 r1712, r1688, r1709;
}
{
cvt.rn.f16.f64 rs245, fd731;
}
mov.b32 r1717, {rs245, rs245};
{
mul.f16x2 r1715, r28, r1717;
}
{
add.f16x2 r1718, r1694, r1715;
}
{
cvt.rn.f16.f64 rs246, fd576;
}
mov.b32 r1723, {rs246, rs246};
{
mul.f16x2 r1721, r31, r1723;
}
{
add.f16x2 r1724, r1700, r1721;
}
{
cvt.rn.f16.f64 rs247, fd771;
}
mov.b32 r1729, {rs247, rs247};
{
mul.f16x2 r1727, r37, r1729;
}
{
add.f16x2 r1730, r1706, r1727;
}
{
cvt.rn.f16.f64 rs248, fd696;
}
mov.b32 r1735, {rs248, rs248};
{
mul.f16x2 r1733, r46, r1735;
}
{
add.f16x2 r1736, r1712, r1733;
}
{
cvt.rn.f16.f64 rs249, fd771;
}
mov.b32 r1741, {rs249, rs249};
{
mul.f16x2 r1739, r40, r1741;
}
{
add.f16x2 r1742, r1718, r1739;
}
{
cvt.rn.f16.f64 rs250, fd696;
}
mov.b32 r1747, {rs250, rs250};
{
mul.f16x2 r1745, r43, r1747;
}
{
add.f16x2 r1748, r1724, r1745;
}
{
cvt.rn.f16.f64 rs251, fd759;
}
mov.b32 r1753, {rs251, rs251};
{
mul.f16x2 r1751, r49, r1753;
}
{
add.f16x2 r1754, r1730, r1751;
}
mov.f64 fd760, 0d3FE863A1ADA0CFA6;
{
cvt.rn.f16.f64 rs252, fd760;
}
mov.b32 r1759, {rs252, rs252};
{
mul.f16x2 r1757, r58, r1759;
}
{
add.f16x2 r1760, r1736, r1757;
}
{
cvt.rn.f16.f64 rs253, fd759;
}
mov.b32 r1765, {rs253, rs253};
{
mul.f16x2 r1763, r52, r1765;
}
{
add.f16x2 r1766, r1742, r1763;
}
{
cvt.rn.f16.f64 rs254, fd760;
}
mov.b32 r1771, {rs254, rs254};
{
mul.f16x2 r1769, r55, r1771;
}
{
add.f16x2 r1772, r1748, r1769;
}
{
cvt.rn.f16.f64 rs255, fd735;
}
mov.b32 r1777, {rs255, rs255};
{
mul.f16x2 r1775, r61, r1777;
}
{
add.f16x2 r1778, r1754, r1775;
}
{
cvt.rn.f16.f64 rs256, fd708;
}
mov.b32 r1783, {rs256, rs256};
{
mul.f16x2 r1781, r70, r1783;
}
{
add.f16x2 r1784, r1760, r1781;
}
{
cvt.rn.f16.f64 rs257, fd735;
}
mov.b32 r1789, {rs257, rs257};
{
mul.f16x2 r1787, r64, r1789;
}
{
add.f16x2 r1790, r1766, r1787;
}
{
cvt.rn.f16.f64 rs258, fd708;
}
mov.b32 r1795, {rs258, rs258};
{
mul.f16x2 r1793, r67, r1795;
}
{
add.f16x2 r1796, r1772, r1793;
}
{
cvt.rn.f16.f64 rs259, fd775;
}
mov.b32 r1801, {rs259, rs259};
{
mul.f16x2 r1799, r73, r1801;
}
{
add.f16x2 r1802, r1778, r1799;
}
{
cvt.rn.f16.f64 rs260, fd424;
}
mov.b32 r1807, {rs260, rs260};
{
mul.f16x2 r1805, r82, r1807;
}
{
add.f16x2 r1808, r1784, r1805;
}
{
cvt.rn.f16.f64 rs261, fd775;
}
mov.b32 r1813, {rs261, rs261};
{
mul.f16x2 r1811, r76, r1813;
}
{
add.f16x2 r1814, r1790, r1811;
}
{
cvt.rn.f16.f64 rs262, fd424;
}
mov.b32 r1819, {rs262, rs262};
{
mul.f16x2 r1817, r79, r1819;
}
{
add.f16x2 r1820, r1796, r1817;
}
{
cvt.rn.f16.f64 rs263, fd755;
}
mov.b32 r1825, {rs263, rs263};
{
mul.f16x2 r1823, r85, r1825;
}
{
add.f16x2 r1826, r1802, r1823;
}
{
cvt.rn.f16.f64 rs264, fd756;
}
mov.b32 r1831, {rs264, rs264};
{
mul.f16x2 r1829, r94, r1831;
}
{
add.f16x2 r1832, r1808, r1829;
}
{
cvt.rn.f16.f64 rs265, fd755;
}
mov.b32 r1837, {rs265, rs265};
{
mul.f16x2 r1835, r88, r1837;
}
{
add.f16x2 r1838, r1814, r1835;
}
{
cvt.rn.f16.f64 rs266, fd756;
}
mov.b32 r1843, {rs266, rs266};
{
mul.f16x2 r1841, r91, r1843;
}
{
add.f16x2 r1844, r1820, r1841;
}
{
cvt.rn.f16.f64 rs267, fd739;
}
mov.b32 r1849, {rs267, rs267};
{
mul.f16x2 r1847, r97, r1849;
}
{
add.f16x2 r1850, r1826, r1847;
}
{
cvt.rn.f16.f64 rs268, fd660;
}
mov.b32 r1855, {rs268, rs268};
{
mul.f16x2 r1853, r106, r1855;
}
{
add.f16x2 r1856, r1832, r1853;
}
{
cvt.rn.f16.f64 rs269, fd739;
}
mov.b32 r1861, {rs269, rs269};
{
mul.f16x2 r1859, r100, r1861;
}
{
add.f16x2 r1862, r1838, r1859;
}
{
cvt.rn.f16.f64 rs270, fd660;
}
mov.b32 r1867, {rs270, rs270};
{
mul.f16x2 r1865, r103, r1867;
}
{
add.f16x2 r1868, r1844, r1865;
}
{
cvt.rn.f16.f64 rs271, fd779;
}
mov.b32 r1873, {rs271, rs271};
{
mul.f16x2 r1871, r109, r1873;
}
{
add.f16x2 r1874, r1850, r1871;
}
{
cvt.rn.f16.f64 rs272, fd652;
}
mov.b32 r1879, {rs272, rs272};
{
mul.f16x2 r1877, r118, r1879;
}
{
add.f16x2 r1880, r1856, r1877;
}
{
cvt.rn.f16.f64 rs273, fd779;
}
mov.b32 r1885, {rs273, rs273};
{
mul.f16x2 r1883, r112, r1885;
}
{
add.f16x2 r1886, r1862, r1883;
}
{
cvt.rn.f16.f64 rs274, fd652;
}
mov.b32 r1891, {rs274, rs274};
{
mul.f16x2 r1889, r115, r1891;
}
{
add.f16x2 r1892, r1868, r1889;
}
{
cvt.rn.f16.f64 rs275, fd751;
}
mov.b32 r1897, {rs275, rs275};
{
mul.f16x2 r1895, r121, r1897;
}
{
add.f16x2 r1898, r1874, r1895;
}
{
cvt.rn.f16.f64 rs276, fd752;
}
mov.b32 r1903, {rs276, rs276};
{
mul.f16x2 r1901, r130, r1903;
}
{
add.f16x2 r1904, r1880, r1901;
}
{
cvt.rn.f16.f64 rs277, fd751;
}
mov.b32 r1909, {rs277, rs277};
{
mul.f16x2 r1907, r124, r1909;
}
{
add.f16x2 r1910, r1886, r1907;
}
{
cvt.rn.f16.f64 rs278, fd752;
}
mov.b32 r1915, {rs278, rs278};
{
mul.f16x2 r1913, r127, r1915;
}
{
add.f16x2 r1916, r1892, r1913;
}
{
cvt.rn.f16.f64 rs279, fd743;
}
mov.b32 r1921, {rs279, rs279};
{
mul.f16x2 r1919, r133, r1921;
}
{
add.f16x2 r1922, r1898, r1919;
}
{
cvt.rn.f16.f64 rs280, fd636;
}
mov.b32 r1927, {rs280, rs280};
{
mul.f16x2 r1925, r142, r1927;
}
{
add.f16x2 r1928, r1904, r1925;
}
{
cvt.rn.f16.f64 rs281, fd743;
}
mov.b32 r1933, {rs281, rs281};
{
mul.f16x2 r1931, r136, r1933;
}
{
add.f16x2 r1934, r1910, r1931;
}
{
cvt.rn.f16.f64 rs282, fd636;
}
mov.b32 r1939, {rs282, rs282};
{
mul.f16x2 r1937, r139, r1939;
}
{
add.f16x2 r1940, r1916, r1937;
}
{
cvt.rn.f16.f64 rs283, fd783;
}
mov.b32 r1945, {rs283, rs283};
{
mul.f16x2 r1943, r145, r1945;
}
{
add.f16x2 r1946, r1922, r1943;
}
{
cvt.rn.f16.f64 rs284, fd692;
}
mov.b32 r1951, {rs284, rs284};
{
mul.f16x2 r1949, r154, r1951;
}
{
add.f16x2 r1952, r1928, r1949;
}
{
cvt.rn.f16.f64 rs285, fd783;
}
mov.b32 r1957, {rs285, rs285};
{
mul.f16x2 r1955, r148, r1957;
}
{
add.f16x2 r1958, r1934, r1955;
}
{
cvt.rn.f16.f64 rs286, fd692;
}
mov.b32 r1963, {rs286, rs286};
{
mul.f16x2 r1961, r151, r1963;
}
{
add.f16x2 r1964, r1940, r1961;
}
{
cvt.rn.f16.f64 rs287, fd747;
}
mov.b32 r1969, {rs287, rs287};
{
mul.f16x2 r1967, r157, r1969;
}
{
add.f16x2 r1970, r1946, r1967;
}
{
cvt.rn.f16.f64 rs288, fd748;
}
mov.b32 r1975, {rs288, rs288};
{
mul.f16x2 r1973, r166, r1975;
}
{
add.f16x2 r1976, r1952, r1973;
}
{
cvt.rn.f16.f64 rs289, fd747;
}
mov.b32 r1981, {rs289, rs289};
{
mul.f16x2 r1979, r160, r1981;
}
{
add.f16x2 r1982, r1958, r1979;
}
{
cvt.rn.f16.f64 rs290, fd748;
}
mov.b32 r1987, {rs290, rs290};
{
mul.f16x2 r1985, r163, r1987;
}
{
add.f16x2 r1988, r1964, r1985;
}
{
sub.f16x2 %10, r1970, r1976;
}
{
add.f16x2 %11, r1982, r1988;
}
{
add.f16x2 %48, r1970, r1976;
}
{
sub.f16x2 %49, r1982, r1988;
}
cvt.rn.f16.s32 rs291, r4804;
mov.b32 r2015, {rs291, rs291};
cvt.rn.f16.s32 rs292, r4804;
mov.b32 r2027, {rs292, rs292};
{
cvt.rn.f16.f64 rs293, fd775;
}
mov.b32 r2007, {rs293, rs293};
{
mul.f16x2 r2005, r1, r2007;
}
{
add.f16x2 r2008, %58, r2005;
}
{
cvt.rn.f16.f64 rs294, fd424;
}
mov.b32 r2013, {rs294, rs294};
{
mul.f16x2 r2011, r10, r2013;
}
{
add.f16x2 r2014, r2015, r2011;
}
{
cvt.rn.f16.f64 rs295, fd775;
}
mov.b32 r2019, {rs295, rs295};
{
mul.f16x2 r2017, r4, r2019;
}
{
add.f16x2 r2020, %59, r2017;
}
{
cvt.rn.f16.f64 rs296, fd424;
}
mov.b32 r2025, {rs296, rs296};
{
mul.f16x2 r2023, r7, r2025;
}
{
add.f16x2 r2026, r2027, r2023;
}
{
cvt.rn.f16.f64 rs297, fd747;
}
mov.b32 r2031, {rs297, rs297};
{
mul.f16x2 r2029, r13, r2031;
}
{
add.f16x2 r2032, r2008, r2029;
}
{
cvt.rn.f16.f64 rs298, fd748;
}
mov.b32 r2037, {rs298, rs298};
{
mul.f16x2 r2035, r22, r2037;
}
{
add.f16x2 r2038, r2014, r2035;
}
{
cvt.rn.f16.f64 rs299, fd747;
}
mov.b32 r2043, {rs299, rs299};
{
mul.f16x2 r2041, r16, r2043;
}
{
add.f16x2 r2044, r2020, r2041;
}
{
cvt.rn.f16.f64 rs300, fd748;
}
mov.b32 r2049, {rs300, rs300};
{
mul.f16x2 r2047, r19, r2049;
}
{
add.f16x2 r2050, r2026, r2047;
}
{
cvt.rn.f16.f64 rs301, fd755;
}
mov.b32 r2055, {rs301, rs301};
{
mul.f16x2 r2053, r25, r2055;
}
{
add.f16x2 r2056, r2032, r2053;
}
{
cvt.rn.f16.f64 rs302, fd456;
}
mov.b32 r2061, {rs302, rs302};
{
mul.f16x2 r2059, r34, r2061;
}
{
add.f16x2 r2062, r2038, r2059;
}
{
cvt.rn.f16.f64 rs303, fd755;
}
mov.b32 r2067, {rs303, rs303};
{
mul.f16x2 r2065, r28, r2067;
}
{
add.f16x2 r2068, r2044, r2065;
}
{
cvt.rn.f16.f64 rs304, fd456;
}
mov.b32 r2073, {rs304, rs304};
{
mul.f16x2 r2071, r31, r2073;
}
{
add.f16x2 r2074, r2050, r2071;
}
{
cvt.rn.f16.f64 rs305, fd767;
}
mov.b32 r2079, {rs305, rs305};
{
mul.f16x2 r2077, r37, r2079;
}
{
add.f16x2 r2080, r2056, r2077;
}
{
cvt.rn.f16.f64 rs306, fd768;
}
mov.b32 r2085, {rs306, rs306};
{
mul.f16x2 r2083, r46, r2085;
}
{
add.f16x2 r2086, r2062, r2083;
}
{
cvt.rn.f16.f64 rs307, fd767;
}
mov.b32 r2091, {rs307, rs307};
{
mul.f16x2 r2089, r40, r2091;
}
{
add.f16x2 r2092, r2068, r2089;
}
{
cvt.rn.f16.f64 rs308, fd768;
}
mov.b32 r2097, {rs308, rs308};
{
mul.f16x2 r2095, r43, r2097;
}
{
add.f16x2 r2098, r2074, r2095;
}
{
cvt.rn.f16.f64 rs309, fd735;
}
mov.b32 r2103, {rs309, rs309};
{
mul.f16x2 r2101, r49, r2103;
}
{
add.f16x2 r2104, r2080, r2101;
}
{
cvt.rn.f16.f64 rs310, fd708;
}
mov.b32 r2109, {rs310, rs310};
{
mul.f16x2 r2107, r58, r2109;
}
{
add.f16x2 r2110, r2086, r2107;
}
{
cvt.rn.f16.f64 rs311, fd735;
}
mov.b32 r2115, {rs311, rs311};
{
mul.f16x2 r2113, r52, r2115;
}
{
add.f16x2 r2116, r2092, r2113;
}
{
cvt.rn.f16.f64 rs312, fd708;
}
mov.b32 r2121, {rs312, rs312};
{
mul.f16x2 r2119, r55, r2121;
}
{
add.f16x2 r2122, r2098, r2119;
}
{
cvt.rn.f16.f64 rs313, fd783;
}
mov.b32 r2127, {rs313, rs313};
{
mul.f16x2 r2125, r61, r2127;
}
{
add.f16x2 r2128, r2104, r2125;
}
{
cvt.rn.f16.f64 rs314, fd692;
}
mov.b32 r2133, {rs314, rs314};
{
mul.f16x2 r2131, r70, r2133;
}
{
add.f16x2 r2134, r2110, r2131;
}
{
cvt.rn.f16.f64 rs315, fd783;
}
mov.b32 r2139, {rs315, rs315};
{
mul.f16x2 r2137, r64, r2139;
}
{
add.f16x2 r2140, r2116, r2137;
}
{
cvt.rn.f16.f64 rs316, fd692;
}
mov.b32 r2145, {rs316, rs316};
{
mul.f16x2 r2143, r67, r2145;
}
{
add.f16x2 r2146, r2122, r2143;
}
{
cvt.rn.f16.f64 rs317, fd739;
}
mov.b32 r2151, {rs317, rs317};
{
mul.f16x2 r2149, r73, r2151;
}
{
add.f16x2 r2152, r2128, r2149;
}
{
cvt.rn.f16.f64 rs318, fd740;
}
mov.b32 r2157, {rs318, rs318};
{
mul.f16x2 r2155, r82, r2157;
}
{
add.f16x2 r2158, r2134, r2155;
}
{
cvt.rn.f16.f64 rs319, fd739;
}
mov.b32 r2163, {rs319, rs319};
{
mul.f16x2 r2161, r76, r2163;
}
{
add.f16x2 r2164, r2140, r2161;
}
{
cvt.rn.f16.f64 rs320, fd740;
}
mov.b32 r2169, {rs320, rs320};
{
mul.f16x2 r2167, r79, r2169;
}
{
add.f16x2 r2170, r2146, r2167;
}
{
cvt.rn.f16.f64 rs321, fd763;
}
mov.b32 r2175, {rs321, rs321};
{
mul.f16x2 r2173, r85, r2175;
}
{
add.f16x2 r2176, r2152, r2173;
}
{
cvt.rn.f16.f64 rs322, fd632;
}
mov.b32 r2181, {rs322, rs322};
{
mul.f16x2 r2179, r94, r2181;
}
{
add.f16x2 r2182, r2158, r2179;
}
{
cvt.rn.f16.f64 rs323, fd763;
}
mov.b32 r2187, {rs323, rs323};
{
mul.f16x2 r2185, r88, r2187;
}
{
add.f16x2 r2188, r2164, r2185;
}
{
cvt.rn.f16.f64 rs324, fd632;
}
mov.b32 r2193, {rs324, rs324};
{
mul.f16x2 r2191, r91, r2193;
}
{
add.f16x2 r2194, r2170, r2191;
}
{
cvt.rn.f16.f64 rs325, fd759;
}
mov.b32 r2199, {rs325, rs325};
{
mul.f16x2 r2197, r97, r2199;
}
{
add.f16x2 r2200, r2176, r2197;
}
{
cvt.rn.f16.f64 rs326, fd760;
}
mov.b32 r2205, {rs326, rs326};
{
mul.f16x2 r2203, r106, r2205;
}
{
add.f16x2 r2206, r2182, r2203;
}
{
cvt.rn.f16.f64 rs327, fd759;
}
mov.b32 r2211, {rs327, rs327};
{
mul.f16x2 r2209, r100, r2211;
}
{
add.f16x2 r2212, r2188, r2209;
}
{
cvt.rn.f16.f64 rs328, fd760;
}
mov.b32 r2217, {rs328, rs328};
{
mul.f16x2 r2215, r103, r2217;
}
{
add.f16x2 r2218, r2194, r2215;
}
{
cvt.rn.f16.f64 rs329, fd743;
}
mov.b32 r2223, {rs329, rs329};
{
mul.f16x2 r2221, r109, r2223;
}
{
add.f16x2 r2224, r2200, r2221;
}
{
cvt.rn.f16.f64 rs330, fd636;
}
mov.b32 r2229, {rs330, rs330};
{
mul.f16x2 r2227, r118, r2229;
}
{
add.f16x2 r2230, r2206, r2227;
}
{
cvt.rn.f16.f64 rs331, fd743;
}
mov.b32 r2235, {rs331, rs331};
{
mul.f16x2 r2233, r112, r2235;
}
{
add.f16x2 r2236, r2212, r2233;
}
{
cvt.rn.f16.f64 rs332, fd636;
}
mov.b32 r2241, {rs332, rs332};
{
mul.f16x2 r2239, r115, r2241;
}
{
add.f16x2 r2242, r2218, r2239;
}
{
cvt.rn.f16.f64 rs333, fd779;
}
mov.b32 r2247, {rs333, rs333};
{
mul.f16x2 r2245, r121, r2247;
}
{
add.f16x2 r2248, r2224, r2245;
}
{
cvt.rn.f16.f64 rs334, fd780;
}
mov.b32 r2253, {rs334, rs334};
{
mul.f16x2 r2251, r130, r2253;
}
{
add.f16x2 r2254, r2230, r2251;
}
{
cvt.rn.f16.f64 rs335, fd779;
}
mov.b32 r2259, {rs335, rs335};
{
mul.f16x2 r2257, r124, r2259;
}
{
add.f16x2 r2260, r2236, r2257;
}
{
cvt.rn.f16.f64 rs336, fd780;
}
mov.b32 r2265, {rs336, rs336};
{
mul.f16x2 r2263, r127, r2265;
}
{
add.f16x2 r2266, r2242, r2263;
}
{
cvt.rn.f16.f64 rs337, fd731;
}
mov.b32 r2271, {rs337, rs337};
{
mul.f16x2 r2269, r133, r2271;
}
{
add.f16x2 r2272, r2248, r2269;
}
{
cvt.rn.f16.f64 rs338, fd732;
}
mov.b32 r2277, {rs338, rs338};
{
mul.f16x2 r2275, r142, r2277;
}
{
add.f16x2 r2278, r2254, r2275;
}
{
cvt.rn.f16.f64 rs339, fd731;
}
mov.b32 r2283, {rs339, rs339};
{
mul.f16x2 r2281, r136, r2283;
}
{
add.f16x2 r2284, r2260, r2281;
}
{
cvt.rn.f16.f64 rs340, fd732;
}
mov.b32 r2289, {rs340, rs340};
{
mul.f16x2 r2287, r139, r2289;
}
{
add.f16x2 r2290, r2266, r2287;
}
{
cvt.rn.f16.f64 rs341, fd771;
}
mov.b32 r2295, {rs341, rs341};
{
mul.f16x2 r2293, r145, r2295;
}
{
add.f16x2 r2296, r2272, r2293;
}
{
cvt.rn.f16.f64 rs342, fd696;
}
mov.b32 r2301, {rs342, rs342};
{
mul.f16x2 r2299, r154, r2301;
}
{
add.f16x2 r2302, r2278, r2299;
}
{
cvt.rn.f16.f64 rs343, fd771;
}
mov.b32 r2307, {rs343, rs343};
{
mul.f16x2 r2305, r148, r2307;
}
{
add.f16x2 r2308, r2284, r2305;
}
{
cvt.rn.f16.f64 rs344, fd696;
}
mov.b32 r2313, {rs344, rs344};
{
mul.f16x2 r2311, r151, r2313;
}
{
add.f16x2 r2314, r2290, r2311;
}
{
cvt.rn.f16.f64 rs345, fd751;
}
mov.b32 r2319, {rs345, rs345};
{
mul.f16x2 r2317, r157, r2319;
}
{
add.f16x2 r2320, r2296, r2317;
}
{
cvt.rn.f16.f64 rs346, fd752;
}
mov.b32 r2325, {rs346, rs346};
{
mul.f16x2 r2323, r166, r2325;
}
{
add.f16x2 r2326, r2302, r2323;
}
{
cvt.rn.f16.f64 rs347, fd751;
}
mov.b32 r2331, {rs347, rs347};
{
mul.f16x2 r2329, r160, r2331;
}
{
add.f16x2 r2332, r2308, r2329;
}
{
cvt.rn.f16.f64 rs348, fd752;
}
mov.b32 r2337, {rs348, rs348};
{
mul.f16x2 r2335, r163, r2337;
}
{
add.f16x2 r2338, r2314, r2335;
}
{
sub.f16x2 %12, r2320, r2326;
}
{
add.f16x2 %13, r2332, r2338;
}
{
add.f16x2 %46, r2320, r2326;
}
{
sub.f16x2 %47, r2332, r2338;
}
cvt.rn.f16.s32 rs349, r4804;
mov.b32 r2365, {rs349, rs349};
cvt.rn.f16.s32 rs350, r4804;
mov.b32 r2377, {rs350, rs350};
{
cvt.rn.f16.f64 rs351, fd783;
}
mov.b32 r2357, {rs351, rs351};
{
mul.f16x2 r2355, r1, r2357;
}
{
add.f16x2 r2358, %58, r2355;
}
{
cvt.rn.f16.f64 rs352, fd692;
}
mov.b32 r2363, {rs352, rs352};
{
mul.f16x2 r2361, r10, r2363;
}
{
add.f16x2 r2364, r2365, r2361;
}
{
cvt.rn.f16.f64 rs353, fd783;
}
mov.b32 r2369, {rs353, rs353};
{
mul.f16x2 r2367, r4, r2369;
}
{
add.f16x2 r2370, %59, r2367;
}
{
cvt.rn.f16.f64 rs354, fd692;
}
mov.b32 r2375, {rs354, rs354};
{
mul.f16x2 r2373, r7, r2375;
}
{
add.f16x2 r2376, r2377, r2373;
}
{
cvt.rn.f16.f64 rs355, fd731;
}
mov.b32 r2381, {rs355, rs355};
{
mul.f16x2 r2379, r13, r2381;
}
{
add.f16x2 r2382, r2358, r2379;
}
{
cvt.rn.f16.f64 rs356, fd732;
}
mov.b32 r2387, {rs356, rs356};
{
mul.f16x2 r2385, r22, r2387;
}
{
add.f16x2 r2388, r2364, r2385;
}
{
cvt.rn.f16.f64 rs357, fd731;
}
mov.b32 r2393, {rs357, rs357};
{
mul.f16x2 r2391, r16, r2393;
}
{
add.f16x2 r2394, r2370, r2391;
}
{
cvt.rn.f16.f64 rs358, fd732;
}
mov.b32 r2399, {rs358, rs358};
{
mul.f16x2 r2397, r19, r2399;
}
{
add.f16x2 r2400, r2376, r2397;
}
{
cvt.rn.f16.f64 rs359, fd779;
}
mov.b32 r2405, {rs359, rs359};
{
mul.f16x2 r2403, r25, r2405;
}
{
add.f16x2 r2406, r2382, r2403;
}
{
cvt.rn.f16.f64 rs360, fd652;
}
mov.b32 r2411, {rs360, rs360};
{
mul.f16x2 r2409, r34, r2411;
}
{
add.f16x2 r2412, r2388, r2409;
}
{
cvt.rn.f16.f64 rs361, fd779;
}
mov.b32 r2417, {rs361, rs361};
{
mul.f16x2 r2415, r28, r2417;
}
{
add.f16x2 r2418, r2394, r2415;
}
{
cvt.rn.f16.f64 rs362, fd652;
}
mov.b32 r2423, {rs362, rs362};
{
mul.f16x2 r2421, r31, r2423;
}
{
add.f16x2 r2424, r2400, r2421;
}
{
cvt.rn.f16.f64 rs363, fd735;
}
mov.b32 r2429, {rs363, rs363};
{
mul.f16x2 r2427, r37, r2429;
}
{
add.f16x2 r2430, r2406, r2427;
}
{
cvt.rn.f16.f64 rs364, fd736;
}
mov.b32 r2435, {rs364, rs364};
{
mul.f16x2 r2433, r46, r2435;
}
{
add.f16x2 r2436, r2412, r2433;
}
{
cvt.rn.f16.f64 rs365, fd735;
}
mov.b32 r2441, {rs365, rs365};
{
mul.f16x2 r2439, r40, r2441;
}
{
add.f16x2 r2442, r2418, r2439;
}
{
cvt.rn.f16.f64 rs366, fd736;
}
mov.b32 r2447, {rs366, rs366};
{
mul.f16x2 r2445, r43, r2447;
}
{
add.f16x2 r2448, r2424, r2445;
}
{
cvt.rn.f16.f64 rs367, fd775;
}
mov.b32 r2453, {rs367, rs367};
{
mul.f16x2 r2451, r49, r2453;
}
{
add.f16x2 r2454, r2430, r2451;
}
{
cvt.rn.f16.f64 rs368, fd424;
}
mov.b32 r2459, {rs368, rs368};
{
mul.f16x2 r2457, r58, r2459;
}
{
add.f16x2 r2460, r2436, r2457;
}
{
cvt.rn.f16.f64 rs369, fd775;
}
mov.b32 r2465, {rs369, rs369};
{
mul.f16x2 r2463, r52, r2465;
}
{
add.f16x2 r2466, r2442, r2463;
}
{
cvt.rn.f16.f64 rs370, fd424;
}
mov.b32 r2471, {rs370, rs370};
{
mul.f16x2 r2469, r55, r2471;
}
{
add.f16x2 r2472, r2448, r2469;
}
{
cvt.rn.f16.f64 rs371, fd739;
}
mov.b32 r2477, {rs371, rs371};
{
mul.f16x2 r2475, r61, r2477;
}
{
add.f16x2 r2478, r2454, r2475;
}
{
cvt.rn.f16.f64 rs372, fd740;
}
mov.b32 r2483, {rs372, rs372};
{
mul.f16x2 r2481, r70, r2483;
}
{
add.f16x2 r2484, r2460, r2481;
}
{
cvt.rn.f16.f64 rs373, fd739;
}
mov.b32 r2489, {rs373, rs373};
{
mul.f16x2 r2487, r64, r2489;
}
{
add.f16x2 r2490, r2466, r2487;
}
{
cvt.rn.f16.f64 rs374, fd740;
}
mov.b32 r2495, {rs374, rs374};
{
mul.f16x2 r2493, r67, r2495;
}
{
add.f16x2 r2496, r2472, r2493;
}
{
cvt.rn.f16.f64 rs375, fd771;
}
mov.b32 r2501, {rs375, rs375};
{
mul.f16x2 r2499, r73, r2501;
}
{
add.f16x2 r2502, r2478, r2499;
}
{
cvt.rn.f16.f64 rs376, fd696;
}
mov.b32 r2507, {rs376, rs376};
{
mul.f16x2 r2505, r82, r2507;
}
{
add.f16x2 r2508, r2484, r2505;
}
{
cvt.rn.f16.f64 rs377, fd771;
}
mov.b32 r2513, {rs377, rs377};
{
mul.f16x2 r2511, r76, r2513;
}
{
add.f16x2 r2514, r2490, r2511;
}
{
cvt.rn.f16.f64 rs378, fd696;
}
mov.b32 r2519, {rs378, rs378};
{
mul.f16x2 r2517, r79, r2519;
}
{
add.f16x2 r2520, r2496, r2517;
}
{
cvt.rn.f16.f64 rs379, fd743;
}
mov.b32 r2525, {rs379, rs379};
{
mul.f16x2 r2523, r85, r2525;
}
{
add.f16x2 r2526, r2502, r2523;
}
{
cvt.rn.f16.f64 rs380, fd744;
}
mov.b32 r2531, {rs380, rs380};
{
mul.f16x2 r2529, r94, r2531;
}
{
add.f16x2 r2532, r2508, r2529;
}
{
cvt.rn.f16.f64 rs381, fd743;
}
mov.b32 r2537, {rs381, rs381};
{
mul.f16x2 r2535, r88, r2537;
}
{
add.f16x2 r2538, r2514, r2535;
}
{
cvt.rn.f16.f64 rs382, fd744;
}
mov.b32 r2543, {rs382, rs382};
{
mul.f16x2 r2541, r91, r2543;
}
{
add.f16x2 r2544, r2520, r2541;
}
{
cvt.rn.f16.f64 rs383, fd767;
}
mov.b32 r2549, {rs383, rs383};
{
mul.f16x2 r2547, r97, r2549;
}
{
add.f16x2 r2550, r2526, r2547;
}
{
cvt.rn.f16.f64 rs384, fd604;
}
mov.b32 r2555, {rs384, rs384};
{
mul.f16x2 r2553, r106, r2555;
}
{
add.f16x2 r2556, r2532, r2553;
}
{
cvt.rn.f16.f64 rs385, fd767;
}
mov.b32 r2561, {rs385, rs385};
{
mul.f16x2 r2559, r100, r2561;
}
{
add.f16x2 r2562, r2538, r2559;
}
{
cvt.rn.f16.f64 rs386, fd604;
}
mov.b32 r2567, {rs386, rs386};
{
mul.f16x2 r2565, r103, r2567;
}
{
add.f16x2 r2568, r2544, r2565;
}
{
cvt.rn.f16.f64 rs387, fd747;
}
mov.b32 r2573, {rs387, rs387};
{
mul.f16x2 r2571, r109, r2573;
}
{
add.f16x2 r2574, r2550, r2571;
}
{
cvt.rn.f16.f64 rs388, fd748;
}
mov.b32 r2579, {rs388, rs388};
{
mul.f16x2 r2577, r118, r2579;
}
{
add.f16x2 r2580, r2556, r2577;
}
{
cvt.rn.f16.f64 rs389, fd747;
}
mov.b32 r2585, {rs389, rs389};
{
mul.f16x2 r2583, r112, r2585;
}
{
add.f16x2 r2586, r2562, r2583;
}
{
cvt.rn.f16.f64 rs390, fd748;
}
mov.b32 r2591, {rs390, rs390};
{
mul.f16x2 r2589, r115, r2591;
}
{
add.f16x2 r2592, r2568, r2589;
}
{
cvt.rn.f16.f64 rs391, fd763;
}
mov.b32 r2597, {rs391, rs391};
{
mul.f16x2 r2595, r121, r2597;
}
{
add.f16x2 r2598, r2574, r2595;
}
{
cvt.rn.f16.f64 rs392, fd632;
}
mov.b32 r2603, {rs392, rs392};
{
mul.f16x2 r2601, r130, r2603;
}
{
add.f16x2 r2604, r2580, r2601;
}
{
cvt.rn.f16.f64 rs393, fd763;
}
mov.b32 r2609, {rs393, rs393};
{
mul.f16x2 r2607, r124, r2609;
}
{
add.f16x2 r2610, r2586, r2607;
}
{
cvt.rn.f16.f64 rs394, fd632;
}
mov.b32 r2615, {rs394, rs394};
{
mul.f16x2 r2613, r127, r2615;
}
{
add.f16x2 r2616, r2592, r2613;
}
{
cvt.rn.f16.f64 rs395, fd751;
}
mov.b32 r2621, {rs395, rs395};
{
mul.f16x2 r2619, r133, r2621;
}
{
add.f16x2 r2622, r2598, r2619;
}
{
cvt.rn.f16.f64 rs396, fd752;
}
mov.b32 r2627, {rs396, rs396};
{
mul.f16x2 r2625, r142, r2627;
}
{
add.f16x2 r2628, r2604, r2625;
}
{
cvt.rn.f16.f64 rs397, fd751;
}
mov.b32 r2633, {rs397, rs397};
{
mul.f16x2 r2631, r136, r2633;
}
{
add.f16x2 r2634, r2610, r2631;
}
{
cvt.rn.f16.f64 rs398, fd752;
}
mov.b32 r2639, {rs398, rs398};
{
mul.f16x2 r2637, r139, r2639;
}
{
add.f16x2 r2640, r2616, r2637;
}
{
cvt.rn.f16.f64 rs399, fd759;
}
mov.b32 r2645, {rs399, rs399};
{
mul.f16x2 r2643, r145, r2645;
}
{
add.f16x2 r2646, r2622, r2643;
}
{
cvt.rn.f16.f64 rs400, fd700;
}
mov.b32 r2651, {rs400, rs400};
{
mul.f16x2 r2649, r154, r2651;
}
{
add.f16x2 r2652, r2628, r2649;
}
{
cvt.rn.f16.f64 rs401, fd759;
}
mov.b32 r2657, {rs401, rs401};
{
mul.f16x2 r2655, r148, r2657;
}
{
add.f16x2 r2658, r2634, r2655;
}
{
cvt.rn.f16.f64 rs402, fd700;
}
mov.b32 r2663, {rs402, rs402};
{
mul.f16x2 r2661, r151, r2663;
}
{
add.f16x2 r2664, r2640, r2661;
}
{
cvt.rn.f16.f64 rs403, fd755;
}
mov.b32 r2669, {rs403, rs403};
{
mul.f16x2 r2667, r157, r2669;
}
{
add.f16x2 r2670, r2646, r2667;
}
{
cvt.rn.f16.f64 rs404, fd756;
}
mov.b32 r2675, {rs404, rs404};
{
mul.f16x2 r2673, r166, r2675;
}
{
add.f16x2 r2676, r2652, r2673;
}
{
cvt.rn.f16.f64 rs405, fd755;
}
mov.b32 r2681, {rs405, rs405};
{
mul.f16x2 r2679, r160, r2681;
}
{
add.f16x2 r2682, r2658, r2679;
}
{
cvt.rn.f16.f64 rs406, fd756;
}
mov.b32 r2687, {rs406, rs406};
{
mul.f16x2 r2685, r163, r2687;
}
{
add.f16x2 r2688, r2664, r2685;
}
{
sub.f16x2 %14, r2670, r2676;
}
{
add.f16x2 %15, r2682, r2688;
}
{
add.f16x2 %44, r2670, r2676;
}
{
sub.f16x2 %45, r2682, r2688;
}
cvt.rn.f16.s32 rs407, r4804;
mov.b32 r2715, {rs407, rs407};
cvt.rn.f16.s32 rs408, r4804;
mov.b32 r2727, {rs408, rs408};
{
cvt.rn.f16.f64 rs409, fd779;
}
mov.b32 r2707, {rs409, rs409};
{
mul.f16x2 r2705, r1, r2707;
}
{
add.f16x2 r2708, %58, r2705;
}
{
cvt.rn.f16.f64 rs410, fd780;
}
mov.b32 r2713, {rs410, rs410};
{
mul.f16x2 r2711, r10, r2713;
}
{
add.f16x2 r2714, r2715, r2711;
}
{
cvt.rn.f16.f64 rs411, fd779;
}
mov.b32 r2719, {rs411, rs411};
{
mul.f16x2 r2717, r4, r2719;
}
{
add.f16x2 r2720, %59, r2717;
}
{
cvt.rn.f16.f64 rs412, fd780;
}
mov.b32 r2725, {rs412, rs412};
{
mul.f16x2 r2723, r7, r2725;
}
{
add.f16x2 r2726, r2727, r2723;
}
{
cvt.rn.f16.f64 rs413, fd739;
}
mov.b32 r2731, {rs413, rs413};
{
mul.f16x2 r2729, r13, r2731;
}
{
add.f16x2 r2732, r2708, r2729;
}
{
cvt.rn.f16.f64 rs414, fd660;
}
mov.b32 r2737, {rs414, rs414};
{
mul.f16x2 r2735, r22, r2737;
}
{
add.f16x2 r2738, r2714, r2735;
}
{
cvt.rn.f16.f64 rs415, fd739;
}
mov.b32 r2743, {rs415, rs415};
{
mul.f16x2 r2741, r16, r2743;
}
{
add.f16x2 r2744, r2720, r2741;
}
{
cvt.rn.f16.f64 rs416, fd660;
}
mov.b32 r2749, {rs416, rs416};
{
mul.f16x2 r2747, r19, r2749;
}
{
add.f16x2 r2750, r2726, r2747;
}
{
cvt.rn.f16.f64 rs417, fd767;
}
mov.b32 r2755, {rs417, rs417};
{
mul.f16x2 r2753, r25, r2755;
}
{
add.f16x2 r2756, r2732, r2753;
}
{
cvt.rn.f16.f64 rs418, fd768;
}
mov.b32 r2761, {rs418, rs418};
{
mul.f16x2 r2759, r34, r2761;
}
{
add.f16x2 r2762, r2738, r2759;
}
{
cvt.rn.f16.f64 rs419, fd767;
}
mov.b32 r2767, {rs419, rs419};
{
mul.f16x2 r2765, r28, r2767;
}
{
add.f16x2 r2768, r2744, r2765;
}
{
cvt.rn.f16.f64 rs420, fd768;
}
mov.b32 r2773, {rs420, rs420};
{
mul.f16x2 r2771, r31, r2773;
}
{
add.f16x2 r2774, r2750, r2771;
}
{
cvt.rn.f16.f64 rs421, fd751;
}
mov.b32 r2779, {rs421, rs421};
{
mul.f16x2 r2777, r37, r2779;
}
{
add.f16x2 r2780, r2756, r2777;
}
{
cvt.rn.f16.f64 rs422, fd540;
}
mov.b32 r2785, {rs422, rs422};
{
mul.f16x2 r2783, r46, r2785;
}
{
add.f16x2 r2786, r2762, r2783;
}
{
cvt.rn.f16.f64 rs423, fd751;
}
mov.b32 r2791, {rs423, rs423};
{
mul.f16x2 r2789, r40, r2791;
}
{
add.f16x2 r2792, r2768, r2789;
}
{
cvt.rn.f16.f64 rs424, fd540;
}
mov.b32 r2797, {rs424, rs424};
{
mul.f16x2 r2795, r43, r2797;
}
{
add.f16x2 r2798, r2774, r2795;
}
{
cvt.rn.f16.f64 rs425, fd755;
}
mov.b32 r2803, {rs425, rs425};
{
mul.f16x2 r2801, r49, r2803;
}
{
add.f16x2 r2804, r2780, r2801;
}
{
cvt.rn.f16.f64 rs426, fd756;
}
mov.b32 r2809, {rs426, rs426};
{
mul.f16x2 r2807, r58, r2809;
}
{
add.f16x2 r2810, r2786, r2807;
}
{
cvt.rn.f16.f64 rs427, fd755;
}
mov.b32 r2815, {rs427, rs427};
{
mul.f16x2 r2813, r52, r2815;
}
{
add.f16x2 r2816, r2792, r2813;
}
{
cvt.rn.f16.f64 rs428, fd756;
}
mov.b32 r2821, {rs428, rs428};
{
mul.f16x2 r2819, r55, r2821;
}
{
add.f16x2 r2822, r2798, r2819;
}
{
cvt.rn.f16.f64 rs429, fd763;
}
mov.b32 r2827, {rs429, rs429};
{
mul.f16x2 r2825, r61, r2827;
}
{
add.f16x2 r2828, r2804, r2825;
}
{
cvt.rn.f16.f64 rs430, fd632;
}
mov.b32 r2833, {rs430, rs430};
{
mul.f16x2 r2831, r70, r2833;
}
{
add.f16x2 r2834, r2810, r2831;
}
{
cvt.rn.f16.f64 rs431, fd763;
}
mov.b32 r2839, {rs431, rs431};
{
mul.f16x2 r2837, r64, r2839;
}
{
add.f16x2 r2840, r2816, r2837;
}
{
cvt.rn.f16.f64 rs432, fd632;
}
mov.b32 r2845, {rs432, rs432};
{
mul.f16x2 r2843, r67, r2845;
}
{
add.f16x2 r2846, r2822, r2843;
}
{
cvt.rn.f16.f64 rs433, fd743;
}
mov.b32 r2851, {rs433, rs433};
{
mul.f16x2 r2849, r73, r2851;
}
{
add.f16x2 r2852, r2828, r2849;
}
{
cvt.rn.f16.f64 rs434, fd744;
}
mov.b32 r2857, {rs434, rs434};
{
mul.f16x2 r2855, r82, r2857;
}
{
add.f16x2 r2858, r2834, r2855;
}
{
cvt.rn.f16.f64 rs435, fd743;
}
mov.b32 r2863, {rs435, rs435};
{
mul.f16x2 r2861, r76, r2863;
}
{
add.f16x2 r2864, r2840, r2861;
}
{
cvt.rn.f16.f64 rs436, fd744;
}
mov.b32 r2869, {rs436, rs436};
{
mul.f16x2 r2867, r79, r2869;
}
{
add.f16x2 r2870, r2846, r2867;
}
{
cvt.rn.f16.f64 rs437, fd775;
}
mov.b32 r2875, {rs437, rs437};
{
mul.f16x2 r2873, r85, r2875;
}
{
add.f16x2 r2876, r2852, r2873;
}
{
cvt.rn.f16.f64 rs438, fd424;
}
mov.b32 r2881, {rs438, rs438};
{
mul.f16x2 r2879, r94, r2881;
}
{
add.f16x2 r2882, r2858, r2879;
}
{
cvt.rn.f16.f64 rs439, fd775;
}
mov.b32 r2887, {rs439, rs439};
{
mul.f16x2 r2885, r88, r2887;
}
{
add.f16x2 r2888, r2864, r2885;
}
{
cvt.rn.f16.f64 rs440, fd424;
}
mov.b32 r2893, {rs440, rs440};
{
mul.f16x2 r2891, r91, r2893;
}
{
add.f16x2 r2894, r2870, r2891;
}
{
cvt.rn.f16.f64 rs441, fd731;
}
mov.b32 r2899, {rs441, rs441};
{
mul.f16x2 r2897, r97, r2899;
}
{
add.f16x2 r2900, r2876, r2897;
}
{
cvt.rn.f16.f64 rs442, fd732;
}
mov.b32 r2905, {rs442, rs442};
{
mul.f16x2 r2903, r106, r2905;
}
{
add.f16x2 r2906, r2882, r2903;
}
{
cvt.rn.f16.f64 rs443, fd731;
}
mov.b32 r2911, {rs443, rs443};
{
mul.f16x2 r2909, r100, r2911;
}
{
add.f16x2 r2912, r2888, r2909;
}
{
cvt.rn.f16.f64 rs444, fd732;
}
mov.b32 r2917, {rs444, rs444};
{
mul.f16x2 r2915, r103, r2917;
}
{
add.f16x2 r2918, r2894, r2915;
}
{
cvt.rn.f16.f64 rs445, fd783;
}
mov.b32 r2923, {rs445, rs445};
{
mul.f16x2 r2921, r109, r2923;
}
{
add.f16x2 r2924, r2900, r2921;
}
{
cvt.rn.f16.f64 rs446, fd784;
}
mov.b32 r2929, {rs446, rs446};
{
mul.f16x2 r2927, r118, r2929;
}
{
add.f16x2 r2930, r2906, r2927;
}
{
cvt.rn.f16.f64 rs447, fd783;
}
mov.b32 r2935, {rs447, rs447};
{
mul.f16x2 r2933, r112, r2935;
}
{
add.f16x2 r2936, r2912, r2933;
}
{
cvt.rn.f16.f64 rs448, fd784;
}
mov.b32 r2941, {rs448, rs448};
{
mul.f16x2 r2939, r115, r2941;
}
{
add.f16x2 r2942, r2918, r2939;
}
{
cvt.rn.f16.f64 rs449, fd735;
}
mov.b32 r2947, {rs449, rs449};
{
mul.f16x2 r2945, r121, r2947;
}
{
add.f16x2 r2948, r2924, r2945;
}
{
cvt.rn.f16.f64 rs450, fd708;
}
mov.b32 r2953, {rs450, rs450};
{
mul.f16x2 r2951, r130, r2953;
}
{
add.f16x2 r2954, r2930, r2951;
}
{
cvt.rn.f16.f64 rs451, fd735;
}
mov.b32 r2959, {rs451, rs451};
{
mul.f16x2 r2957, r124, r2959;
}
{
add.f16x2 r2960, r2936, r2957;
}
{
cvt.rn.f16.f64 rs452, fd708;
}
mov.b32 r2965, {rs452, rs452};
{
mul.f16x2 r2963, r127, r2965;
}
{
add.f16x2 r2966, r2942, r2963;
}
{
cvt.rn.f16.f64 rs453, fd771;
}
mov.b32 r2971, {rs453, rs453};
{
mul.f16x2 r2969, r133, r2971;
}
{
add.f16x2 r2972, r2948, r2969;
}
{
cvt.rn.f16.f64 rs454, fd772;
}
mov.b32 r2977, {rs454, rs454};
{
mul.f16x2 r2975, r142, r2977;
}
{
add.f16x2 r2978, r2954, r2975;
}
{
cvt.rn.f16.f64 rs455, fd771;
}
mov.b32 r2983, {rs455, rs455};
{
mul.f16x2 r2981, r136, r2983;
}
{
add.f16x2 r2984, r2960, r2981;
}
{
cvt.rn.f16.f64 rs456, fd772;
}
mov.b32 r2989, {rs456, rs456};
{
mul.f16x2 r2987, r139, r2989;
}
{
add.f16x2 r2990, r2966, r2987;
}
{
cvt.rn.f16.f64 rs457, fd747;
}
mov.b32 r2995, {rs457, rs457};
{
mul.f16x2 r2993, r145, r2995;
}
{
add.f16x2 r2996, r2972, r2993;
}
mov.f64 fd704, 0d3FE07F6ACD7CDCE2;
{
cvt.rn.f16.f64 rs458, fd704;
}
mov.b32 r3001, {rs458, rs458};
{
mul.f16x2 r2999, r154, r3001;
}
{
add.f16x2 r3002, r2978, r2999;
}
{
cvt.rn.f16.f64 rs459, fd747;
}
mov.b32 r3007, {rs459, rs459};
{
mul.f16x2 r3005, r148, r3007;
}
{
add.f16x2 r3008, r2984, r3005;
}
{
cvt.rn.f16.f64 rs460, fd704;
}
mov.b32 r3013, {rs460, rs460};
{
mul.f16x2 r3011, r151, r3013;
}
{
add.f16x2 r3014, r2990, r3011;
}
{
cvt.rn.f16.f64 rs461, fd759;
}
mov.b32 r3019, {rs461, rs461};
{
mul.f16x2 r3017, r157, r3019;
}
{
add.f16x2 r3020, r2996, r3017;
}
{
cvt.rn.f16.f64 rs462, fd760;
}
mov.b32 r3025, {rs462, rs462};
{
mul.f16x2 r3023, r166, r3025;
}
{
add.f16x2 r3026, r3002, r3023;
}
{
cvt.rn.f16.f64 rs463, fd759;
}
mov.b32 r3031, {rs463, rs463};
{
mul.f16x2 r3029, r160, r3031;
}
{
add.f16x2 r3032, r3008, r3029;
}
{
cvt.rn.f16.f64 rs464, fd760;
}
mov.b32 r3037, {rs464, rs464};
{
mul.f16x2 r3035, r163, r3037;
}
{
add.f16x2 r3038, r3014, r3035;
}
{
sub.f16x2 %16, r3020, r3026;
}
{
add.f16x2 %17, r3032, r3038;
}
{
add.f16x2 %42, r3020, r3026;
}
{
sub.f16x2 %43, r3032, r3038;
}
cvt.rn.f16.s32 rs465, r4804;
mov.b32 r3065, {rs465, rs465};
cvt.rn.f16.s32 rs466, r4804;
mov.b32 r3077, {rs466, rs466};
{
cvt.rn.f16.f64 rs467, fd771;
}
mov.b32 r3057, {rs467, rs467};
{
mul.f16x2 r3055, r1, r3057;
}
{
add.f16x2 r3058, %58, r3055;
}
{
cvt.rn.f16.f64 rs468, fd772;
}
mov.b32 r3063, {rs468, rs468};
{
mul.f16x2 r3061, r10, r3063;
}
{
add.f16x2 r3064, r3065, r3061;
}
{
cvt.rn.f16.f64 rs469, fd771;
}
mov.b32 r3069, {rs469, rs469};
{
mul.f16x2 r3067, r4, r3069;
}
{
add.f16x2 r3070, %59, r3067;
}
{
cvt.rn.f16.f64 rs470, fd772;
}
mov.b32 r3075, {rs470, rs470};
{
mul.f16x2 r3073, r7, r3075;
}
{
add.f16x2 r3076, r3077, r3073;
}
{
cvt.rn.f16.f64 rs471, fd755;
}
mov.b32 r3081, {rs471, rs471};
{
mul.f16x2 r3079, r13, r3081;
}
{
add.f16x2 r3082, r3058, r3079;
}
{
cvt.rn.f16.f64 rs472, fd456;
}
mov.b32 r3087, {rs472, rs472};
{
mul.f16x2 r3085, r22, r3087;
}
{
add.f16x2 r3088, r3064, r3085;
}
{
cvt.rn.f16.f64 rs473, fd755;
}
mov.b32 r3093, {rs473, rs473};
{
mul.f16x2 r3091, r16, r3093;
}
{
add.f16x2 r3094, r3070, r3091;
}
{
cvt.rn.f16.f64 rs474, fd456;
}
mov.b32 r3099, {rs474, rs474};
{
mul.f16x2 r3097, r19, r3099;
}
{
add.f16x2 r3100, r3076, r3097;
}
{
cvt.rn.f16.f64 rs475, fd743;
}
mov.b32 r3105, {rs475, rs475};
{
mul.f16x2 r3103, r25, r3105;
}
{
add.f16x2 r3106, r3082, r3103;
}
{
cvt.rn.f16.f64 rs476, fd744;
}
mov.b32 r3111, {rs476, rs476};
{
mul.f16x2 r3109, r34, r3111;
}
{
add.f16x2 r3112, r3088, r3109;
}
{
cvt.rn.f16.f64 rs477, fd743;
}
mov.b32 r3117, {rs477, rs477};
{
mul.f16x2 r3115, r28, r3117;
}
{
add.f16x2 r3118, r3094, r3115;
}
{
cvt.rn.f16.f64 rs478, fd744;
}
mov.b32 r3123, {rs478, rs478};
{
mul.f16x2 r3121, r31, r3123;
}
{
add.f16x2 r3124, r3100, r3121;
}
{
cvt.rn.f16.f64 rs479, fd783;
}
mov.b32 r3129, {rs479, rs479};
{
mul.f16x2 r3127, r37, r3129;
}
{
add.f16x2 r3130, r3106, r3127;
}
{
cvt.rn.f16.f64 rs480, fd692;
}
mov.b32 r3135, {rs480, rs480};
{
mul.f16x2 r3133, r46, r3135;
}
{
add.f16x2 r3136, r3112, r3133;
}
{
cvt.rn.f16.f64 rs481, fd783;
}
mov.b32 r3141, {rs481, rs481};
{
mul.f16x2 r3139, r40, r3141;
}
{
add.f16x2 r3142, r3118, r3139;
}
{
cvt.rn.f16.f64 rs482, fd692;
}
mov.b32 r3147, {rs482, rs482};
{
mul.f16x2 r3145, r43, r3147;
}
{
add.f16x2 r3148, r3124, r3145;
}
{
cvt.rn.f16.f64 rs483, fd739;
}
mov.b32 r3153, {rs483, rs483};
{
mul.f16x2 r3151, r49, r3153;
}
{
add.f16x2 r3154, r3130, r3151;
}
{
cvt.rn.f16.f64 rs484, fd660;
}
mov.b32 r3159, {rs484, rs484};
{
mul.f16x2 r3157, r58, r3159;
}
{
add.f16x2 r3160, r3136, r3157;
}
{
cvt.rn.f16.f64 rs485, fd739;
}
mov.b32 r3165, {rs485, rs485};
{
mul.f16x2 r3163, r52, r3165;
}
{
add.f16x2 r3166, r3142, r3163;
}
{
cvt.rn.f16.f64 rs486, fd660;
}
mov.b32 r3171, {rs486, rs486};
{
mul.f16x2 r3169, r55, r3171;
}
{
add.f16x2 r3172, r3148, r3169;
}
{
cvt.rn.f16.f64 rs487, fd759;
}
mov.b32 r3177, {rs487, rs487};
{
mul.f16x2 r3175, r61, r3177;
}
{
add.f16x2 r3178, r3154, r3175;
}
{
cvt.rn.f16.f64 rs488, fd760;
}
mov.b32 r3183, {rs488, rs488};
{
mul.f16x2 r3181, r70, r3183;
}
{
add.f16x2 r3184, r3160, r3181;
}
{
cvt.rn.f16.f64 rs489, fd759;
}
mov.b32 r3189, {rs489, rs489};
{
mul.f16x2 r3187, r64, r3189;
}
{
add.f16x2 r3190, r3166, r3187;
}
{
cvt.rn.f16.f64 rs490, fd760;
}
mov.b32 r3195, {rs490, rs490};
{
mul.f16x2 r3193, r67, r3195;
}
{
add.f16x2 r3196, r3172, r3193;
}
{
cvt.rn.f16.f64 rs491, fd767;
}
mov.b32 r3201, {rs491, rs491};
{
mul.f16x2 r3199, r73, r3201;
}
{
add.f16x2 r3202, r3178, r3199;
}
{
cvt.rn.f16.f64 rs492, fd604;
}
mov.b32 r3207, {rs492, rs492};
{
mul.f16x2 r3205, r82, r3207;
}
{
add.f16x2 r3208, r3184, r3205;
}
{
cvt.rn.f16.f64 rs493, fd767;
}
mov.b32 r3213, {rs493, rs493};
{
mul.f16x2 r3211, r76, r3213;
}
{
add.f16x2 r3214, r3190, r3211;
}
{
cvt.rn.f16.f64 rs494, fd604;
}
mov.b32 r3219, {rs494, rs494};
{
mul.f16x2 r3217, r79, r3219;
}
{
add.f16x2 r3220, r3196, r3217;
}
{
cvt.rn.f16.f64 rs495, fd731;
}
mov.b32 r3225, {rs495, rs495};
{
mul.f16x2 r3223, r85, r3225;
}
{
add.f16x2 r3226, r3202, r3223;
}
{
cvt.rn.f16.f64 rs496, fd732;
}
mov.b32 r3231, {rs496, rs496};
{
mul.f16x2 r3229, r94, r3231;
}
{
add.f16x2 r3232, r3208, r3229;
}
{
cvt.rn.f16.f64 rs497, fd731;
}
mov.b32 r3237, {rs497, rs497};
{
mul.f16x2 r3235, r88, r3237;
}
{
add.f16x2 r3238, r3214, r3235;
}
{
cvt.rn.f16.f64 rs498, fd732;
}
mov.b32 r3243, {rs498, rs498};
{
mul.f16x2 r3241, r91, r3243;
}
{
add.f16x2 r3244, r3220, r3241;
}
{
cvt.rn.f16.f64 rs499, fd775;
}
mov.b32 r3249, {rs499, rs499};
{
mul.f16x2 r3247, r97, r3249;
}
{
add.f16x2 r3250, r3226, r3247;
}
{
cvt.rn.f16.f64 rs500, fd776;
}
mov.b32 r3255, {rs500, rs500};
{
mul.f16x2 r3253, r106, r3255;
}
{
add.f16x2 r3256, r3232, r3253;
}
{
cvt.rn.f16.f64 rs501, fd775;
}
mov.b32 r3261, {rs501, rs501};
{
mul.f16x2 r3259, r100, r3261;
}
{
add.f16x2 r3262, r3238, r3259;
}
{
cvt.rn.f16.f64 rs502, fd776;
}
mov.b32 r3267, {rs502, rs502};
{
mul.f16x2 r3265, r103, r3267;
}
{
add.f16x2 r3268, r3244, r3265;
}
{
cvt.rn.f16.f64 rs503, fd751;
}
mov.b32 r3273, {rs503, rs503};
{
mul.f16x2 r3271, r109, r3273;
}
{
add.f16x2 r3274, r3250, r3271;
}
{
cvt.rn.f16.f64 rs504, fd540;
}
mov.b32 r3279, {rs504, rs504};
{
mul.f16x2 r3277, r118, r3279;
}
{
add.f16x2 r3280, r3256, r3277;
}
{
cvt.rn.f16.f64 rs505, fd751;
}
mov.b32 r3285, {rs505, rs505};
{
mul.f16x2 r3283, r112, r3285;
}
{
add.f16x2 r3286, r3262, r3283;
}
{
cvt.rn.f16.f64 rs506, fd540;
}
mov.b32 r3291, {rs506, rs506};
{
mul.f16x2 r3289, r115, r3291;
}
{
add.f16x2 r3292, r3268, r3289;
}
{
cvt.rn.f16.f64 rs507, fd747;
}
mov.b32 r3297, {rs507, rs507};
{
mul.f16x2 r3295, r121, r3297;
}
{
add.f16x2 r3298, r3274, r3295;
}
{
cvt.rn.f16.f64 rs508, fd748;
}
mov.b32 r3303, {rs508, rs508};
{
mul.f16x2 r3301, r130, r3303;
}
{
add.f16x2 r3304, r3280, r3301;
}
{
cvt.rn.f16.f64 rs509, fd747;
}
mov.b32 r3309, {rs509, rs509};
{
mul.f16x2 r3307, r124, r3309;
}
{
add.f16x2 r3310, r3286, r3307;
}
{
cvt.rn.f16.f64 rs510, fd748;
}
mov.b32 r3315, {rs510, rs510};
{
mul.f16x2 r3313, r127, r3315;
}
{
add.f16x2 r3316, r3292, r3313;
}
{
cvt.rn.f16.f64 rs511, fd779;
}
mov.b32 r3321, {rs511, rs511};
{
mul.f16x2 r3319, r133, r3321;
}
{
add.f16x2 r3322, r3298, r3319;
}
{
cvt.rn.f16.f64 rs512, fd652;
}
mov.b32 r3327, {rs512, rs512};
{
mul.f16x2 r3325, r142, r3327;
}
{
add.f16x2 r3328, r3304, r3325;
}
{
cvt.rn.f16.f64 rs513, fd779;
}
mov.b32 r3333, {rs513, rs513};
{
mul.f16x2 r3331, r136, r3333;
}
{
add.f16x2 r3334, r3310, r3331;
}
{
cvt.rn.f16.f64 rs514, fd652;
}
mov.b32 r3339, {rs514, rs514};
{
mul.f16x2 r3337, r139, r3339;
}
{
add.f16x2 r3340, r3316, r3337;
}
{
cvt.rn.f16.f64 rs515, fd735;
}
mov.b32 r3345, {rs515, rs515};
{
mul.f16x2 r3343, r145, r3345;
}
{
add.f16x2 r3346, r3322, r3343;
}
{
cvt.rn.f16.f64 rs516, fd708;
}
mov.b32 r3351, {rs516, rs516};
{
mul.f16x2 r3349, r154, r3351;
}
{
add.f16x2 r3352, r3328, r3349;
}
{
cvt.rn.f16.f64 rs517, fd735;
}
mov.b32 r3357, {rs517, rs517};
{
mul.f16x2 r3355, r148, r3357;
}
{
add.f16x2 r3358, r3334, r3355;
}
{
cvt.rn.f16.f64 rs518, fd708;
}
mov.b32 r3363, {rs518, rs518};
{
mul.f16x2 r3361, r151, r3363;
}
{
add.f16x2 r3364, r3340, r3361;
}
{
cvt.rn.f16.f64 rs519, fd763;
}
mov.b32 r3369, {rs519, rs519};
{
mul.f16x2 r3367, r157, r3369;
}
{
add.f16x2 r3370, r3346, r3367;
}
{
cvt.rn.f16.f64 rs520, fd764;
}
mov.b32 r3375, {rs520, rs520};
{
mul.f16x2 r3373, r166, r3375;
}
{
add.f16x2 r3376, r3352, r3373;
}
{
cvt.rn.f16.f64 rs521, fd763;
}
mov.b32 r3381, {rs521, rs521};
{
mul.f16x2 r3379, r160, r3381;
}
{
add.f16x2 r3382, r3358, r3379;
}
{
cvt.rn.f16.f64 rs522, fd764;
}
mov.b32 r3387, {rs522, rs522};
{
mul.f16x2 r3385, r163, r3387;
}
{
add.f16x2 r3388, r3364, r3385;
}
{
sub.f16x2 %18, r3370, r3376;
}
{
add.f16x2 %19, r3382, r3388;
}
{
add.f16x2 %40, r3370, r3376;
}
{
sub.f16x2 %41, r3382, r3388;
}
cvt.rn.f16.s32 rs523, r4804;
mov.b32 r3415, {rs523, rs523};
cvt.rn.f16.s32 rs524, r4804;
mov.b32 r3427, {rs524, rs524};
{
cvt.rn.f16.f64 rs525, fd763;
}
mov.b32 r3407, {rs525, rs525};
{
mul.f16x2 r3405, r1, r3407;
}
{
add.f16x2 r3408, %58, r3405;
}
{
cvt.rn.f16.f64 rs526, fd764;
}
mov.b32 r3413, {rs526, rs526};
{
mul.f16x2 r3411, r10, r3413;
}
{
add.f16x2 r3414, r3415, r3411;
}
{
cvt.rn.f16.f64 rs527, fd763;
}
mov.b32 r3419, {rs527, rs527};
{
mul.f16x2 r3417, r4, r3419;
}
{
add.f16x2 r3420, %59, r3417;
}
{
cvt.rn.f16.f64 rs528, fd764;
}
mov.b32 r3425, {rs528, rs528};
{
mul.f16x2 r3423, r7, r3425;
}
{
add.f16x2 r3426, r3427, r3423;
}
{
cvt.rn.f16.f64 rs529, fd771;
}
mov.b32 r3431, {rs529, rs529};
{
mul.f16x2 r3429, r13, r3431;
}
{
add.f16x2 r3432, r3408, r3429;
}
{
cvt.rn.f16.f64 rs530, fd696;
}
mov.b32 r3437, {rs530, rs530};
{
mul.f16x2 r3435, r22, r3437;
}
{
add.f16x2 r3438, r3414, r3435;
}
{
cvt.rn.f16.f64 rs531, fd771;
}
mov.b32 r3443, {rs531, rs531};
{
mul.f16x2 r3441, r16, r3443;
}
{
add.f16x2 r3444, r3420, r3441;
}
{
cvt.rn.f16.f64 rs532, fd696;
}
mov.b32 r3449, {rs532, rs532};
{
mul.f16x2 r3447, r19, r3449;
}
{
add.f16x2 r3450, r3426, r3447;
}
{
cvt.rn.f16.f64 rs533, fd735;
}
mov.b32 r3455, {rs533, rs533};
{
mul.f16x2 r3453, r25, r3455;
}
{
add.f16x2 r3456, r3432, r3453;
}
{
cvt.rn.f16.f64 rs534, fd708;
}
mov.b32 r3461, {rs534, rs534};
{
mul.f16x2 r3459, r34, r3461;
}
{
add.f16x2 r3462, r3438, r3459;
}
{
cvt.rn.f16.f64 rs535, fd735;
}
mov.b32 r3467, {rs535, rs535};
{
mul.f16x2 r3465, r28, r3467;
}
{
add.f16x2 r3468, r3444, r3465;
}
{
cvt.rn.f16.f64 rs536, fd708;
}
mov.b32 r3473, {rs536, rs536};
{
mul.f16x2 r3471, r31, r3473;
}
{
add.f16x2 r3474, r3450, r3471;
}
{
cvt.rn.f16.f64 rs537, fd755;
}
mov.b32 r3479, {rs537, rs537};
{
mul.f16x2 r3477, r37, r3479;
}
{
add.f16x2 r3480, r3456, r3477;
}
{
cvt.rn.f16.f64 rs538, fd756;
}
mov.b32 r3485, {rs538, rs538};
{
mul.f16x2 r3483, r46, r3485;
}
{
add.f16x2 r3486, r3462, r3483;
}
{
cvt.rn.f16.f64 rs539, fd755;
}
mov.b32 r3491, {rs539, rs539};
{
mul.f16x2 r3489, r40, r3491;
}
{
add.f16x2 r3492, r3468, r3489;
}
{
cvt.rn.f16.f64 rs540, fd756;
}
mov.b32 r3497, {rs540, rs540};
{
mul.f16x2 r3495, r43, r3497;
}
{
add.f16x2 r3498, r3474, r3495;
}
{
cvt.rn.f16.f64 rs541, fd779;
}
mov.b32 r3503, {rs541, rs541};
{
mul.f16x2 r3501, r49, r3503;
}
{
add.f16x2 r3504, r3480, r3501;
}
{
cvt.rn.f16.f64 rs542, fd652;
}
mov.b32 r3509, {rs542, rs542};
{
mul.f16x2 r3507, r58, r3509;
}
{
add.f16x2 r3510, r3486, r3507;
}
{
cvt.rn.f16.f64 rs543, fd779;
}
mov.b32 r3515, {rs543, rs543};
{
mul.f16x2 r3513, r52, r3515;
}
{
add.f16x2 r3516, r3492, r3513;
}
{
cvt.rn.f16.f64 rs544, fd652;
}
mov.b32 r3521, {rs544, rs544};
{
mul.f16x2 r3519, r55, r3521;
}
{
add.f16x2 r3522, r3498, r3519;
}
{
cvt.rn.f16.f64 rs545, fd743;
}
mov.b32 r3527, {rs545, rs545};
{
mul.f16x2 r3525, r61, r3527;
}
{
add.f16x2 r3528, r3504, r3525;
}
{
cvt.rn.f16.f64 rs546, fd636;
}
mov.b32 r3533, {rs546, rs546};
{
mul.f16x2 r3531, r70, r3533;
}
{
add.f16x2 r3534, r3510, r3531;
}
{
cvt.rn.f16.f64 rs547, fd743;
}
mov.b32 r3539, {rs547, rs547};
{
mul.f16x2 r3537, r64, r3539;
}
{
add.f16x2 r3540, r3516, r3537;
}
{
cvt.rn.f16.f64 rs548, fd636;
}
mov.b32 r3545, {rs548, rs548};
{
mul.f16x2 r3543, r67, r3545;
}
{
add.f16x2 r3546, r3522, r3543;
}
{
cvt.rn.f16.f64 rs549, fd747;
}
mov.b32 r3551, {rs549, rs549};
{
mul.f16x2 r3549, r73, r3551;
}
{
add.f16x2 r3552, r3528, r3549;
}
{
cvt.rn.f16.f64 rs550, fd748;
}
mov.b32 r3557, {rs550, rs550};
{
mul.f16x2 r3555, r82, r3557;
}
{
add.f16x2 r3558, r3534, r3555;
}
{
cvt.rn.f16.f64 rs551, fd747;
}
mov.b32 r3563, {rs551, rs551};
{
mul.f16x2 r3561, r76, r3563;
}
{
add.f16x2 r3564, r3540, r3561;
}
{
cvt.rn.f16.f64 rs552, fd748;
}
mov.b32 r3569, {rs552, rs552};
{
mul.f16x2 r3567, r79, r3569;
}
{
add.f16x2 r3570, r3546, r3567;
}
{
cvt.rn.f16.f64 rs553, fd783;
}
mov.b32 r3575, {rs553, rs553};
{
mul.f16x2 r3573, r85, r3575;
}
{
add.f16x2 r3576, r3552, r3573;
}
{
cvt.rn.f16.f64 rs554, fd784;
}
mov.b32 r3581, {rs554, rs554};
{
mul.f16x2 r3579, r94, r3581;
}
{
add.f16x2 r3582, r3558, r3579;
}
{
cvt.rn.f16.f64 rs555, fd783;
}
mov.b32 r3587, {rs555, rs555};
{
mul.f16x2 r3585, r88, r3587;
}
{
add.f16x2 r3588, r3564, r3585;
}
{
cvt.rn.f16.f64 rs556, fd784;
}
mov.b32 r3593, {rs556, rs556};
{
mul.f16x2 r3591, r91, r3593;
}
{
add.f16x2 r3594, r3570, r3591;
}
{
cvt.rn.f16.f64 rs557, fd751;
}
mov.b32 r3599, {rs557, rs557};
{
mul.f16x2 r3597, r97, r3599;
}
{
add.f16x2 r3600, r3576, r3597;
}
{
cvt.rn.f16.f64 rs558, fd540;
}
mov.b32 r3605, {rs558, rs558};
{
mul.f16x2 r3603, r106, r3605;
}
{
add.f16x2 r3606, r3582, r3603;
}
{
cvt.rn.f16.f64 rs559, fd751;
}
mov.b32 r3611, {rs559, rs559};
{
mul.f16x2 r3609, r100, r3611;
}
{
add.f16x2 r3612, r3588, r3609;
}
{
cvt.rn.f16.f64 rs560, fd540;
}
mov.b32 r3617, {rs560, rs560};
{
mul.f16x2 r3615, r103, r3617;
}
{
add.f16x2 r3618, r3594, r3615;
}
{
cvt.rn.f16.f64 rs561, fd739;
}
mov.b32 r3623, {rs561, rs561};
{
mul.f16x2 r3621, r109, r3623;
}
{
add.f16x2 r3624, r3600, r3621;
}
{
cvt.rn.f16.f64 rs562, fd740;
}
mov.b32 r3629, {rs562, rs562};
{
mul.f16x2 r3627, r118, r3629;
}
{
add.f16x2 r3630, r3606, r3627;
}
{
cvt.rn.f16.f64 rs563, fd739;
}
mov.b32 r3635, {rs563, rs563};
{
mul.f16x2 r3633, r112, r3635;
}
{
add.f16x2 r3636, r3612, r3633;
}
{
cvt.rn.f16.f64 rs564, fd740;
}
mov.b32 r3641, {rs564, rs564};
{
mul.f16x2 r3639, r115, r3641;
}
{
add.f16x2 r3642, r3618, r3639;
}
{
cvt.rn.f16.f64 rs565, fd775;
}
mov.b32 r3647, {rs565, rs565};
{
mul.f16x2 r3645, r121, r3647;
}
{
add.f16x2 r3648, r3624, r3645;
}
{
cvt.rn.f16.f64 rs566, fd776;
}
mov.b32 r3653, {rs566, rs566};
{
mul.f16x2 r3651, r130, r3653;
}
{
add.f16x2 r3654, r3630, r3651;
}
{
cvt.rn.f16.f64 rs567, fd775;
}
mov.b32 r3659, {rs567, rs567};
{
mul.f16x2 r3657, r124, r3659;
}
{
add.f16x2 r3660, r3636, r3657;
}
{
cvt.rn.f16.f64 rs568, fd776;
}
mov.b32 r3665, {rs568, rs568};
{
mul.f16x2 r3663, r127, r3665;
}
{
add.f16x2 r3666, r3642, r3663;
}
{
cvt.rn.f16.f64 rs569, fd759;
}
mov.b32 r3671, {rs569, rs569};
{
mul.f16x2 r3669, r133, r3671;
}
{
add.f16x2 r3672, r3648, r3669;
}
{
cvt.rn.f16.f64 rs570, fd700;
}
mov.b32 r3677, {rs570, rs570};
{
mul.f16x2 r3675, r142, r3677;
}
{
add.f16x2 r3678, r3654, r3675;
}
{
cvt.rn.f16.f64 rs571, fd759;
}
mov.b32 r3683, {rs571, rs571};
{
mul.f16x2 r3681, r136, r3683;
}
{
add.f16x2 r3684, r3660, r3681;
}
{
cvt.rn.f16.f64 rs572, fd700;
}
mov.b32 r3689, {rs572, rs572};
{
mul.f16x2 r3687, r139, r3689;
}
{
add.f16x2 r3690, r3666, r3687;
}
{
cvt.rn.f16.f64 rs573, fd731;
}
mov.b32 r3695, {rs573, rs573};
{
mul.f16x2 r3693, r145, r3695;
}
{
add.f16x2 r3696, r3672, r3693;
}
{
cvt.rn.f16.f64 rs574, fd732;
}
mov.b32 r3701, {rs574, rs574};
{
mul.f16x2 r3699, r154, r3701;
}
{
add.f16x2 r3702, r3678, r3699;
}
{
cvt.rn.f16.f64 rs575, fd731;
}
mov.b32 r3707, {rs575, rs575};
{
mul.f16x2 r3705, r148, r3707;
}
{
add.f16x2 r3708, r3684, r3705;
}
{
cvt.rn.f16.f64 rs576, fd732;
}
mov.b32 r3713, {rs576, rs576};
{
mul.f16x2 r3711, r151, r3713;
}
{
add.f16x2 r3714, r3690, r3711;
}
{
cvt.rn.f16.f64 rs577, fd767;
}
mov.b32 r3719, {rs577, rs577};
{
mul.f16x2 r3717, r157, r3719;
}
{
add.f16x2 r3720, r3696, r3717;
}
{
cvt.rn.f16.f64 rs578, fd768;
}
mov.b32 r3725, {rs578, rs578};
{
mul.f16x2 r3723, r166, r3725;
}
{
add.f16x2 r3726, r3702, r3723;
}
{
cvt.rn.f16.f64 rs579, fd767;
}
mov.b32 r3731, {rs579, rs579};
{
mul.f16x2 r3729, r160, r3731;
}
{
add.f16x2 r3732, r3708, r3729;
}
{
cvt.rn.f16.f64 rs580, fd768;
}
mov.b32 r3737, {rs580, rs580};
{
mul.f16x2 r3735, r163, r3737;
}
{
add.f16x2 r3738, r3714, r3735;
}
{
sub.f16x2 %20, r3720, r3726;
}
{
add.f16x2 %21, r3732, r3738;
}
{
add.f16x2 %38, r3720, r3726;
}
{
sub.f16x2 %39, r3732, r3738;
}
cvt.rn.f16.s32 rs581, r4804;
mov.b32 r3765, {rs581, rs581};
cvt.rn.f16.s32 rs582, r4804;
mov.b32 r3777, {rs582, rs582};
{
cvt.rn.f16.f64 rs583, fd755;
}
mov.b32 r3757, {rs583, rs583};
{
mul.f16x2 r3755, r1, r3757;
}
{
add.f16x2 r3758, %58, r3755;
}
{
cvt.rn.f16.f64 rs584, fd756;
}
mov.b32 r3763, {rs584, rs584};
{
mul.f16x2 r3761, r10, r3763;
}
{
add.f16x2 r3764, r3765, r3761;
}
{
cvt.rn.f16.f64 rs585, fd755;
}
mov.b32 r3769, {rs585, rs585};
{
mul.f16x2 r3767, r4, r3769;
}
{
add.f16x2 r3770, %59, r3767;
}
{
cvt.rn.f16.f64 rs586, fd756;
}
mov.b32 r3775, {rs586, rs586};
{
mul.f16x2 r3773, r7, r3775;
}
{
add.f16x2 r3776, r3777, r3773;
}
{
cvt.rn.f16.f64 rs587, fd783;
}
mov.b32 r3781, {rs587, rs587};
{
mul.f16x2 r3779, r13, r3781;
}
{
add.f16x2 r3782, r3758, r3779;
}
{
cvt.rn.f16.f64 rs588, fd784;
}
mov.b32 r3787, {rs588, rs588};
{
mul.f16x2 r3785, r22, r3787;
}
{
add.f16x2 r3788, r3764, r3785;
}
{
cvt.rn.f16.f64 rs589, fd783;
}
mov.b32 r3793, {rs589, rs589};
{
mul.f16x2 r3791, r16, r3793;
}
{
add.f16x2 r3794, r3770, r3791;
}
{
cvt.rn.f16.f64 rs590, fd784;
}
mov.b32 r3799, {rs590, rs590};
{
mul.f16x2 r3797, r19, r3799;
}
{
add.f16x2 r3800, r3776, r3797;
}
{
cvt.rn.f16.f64 rs591, fd759;
}
mov.b32 r3805, {rs591, rs591};
{
mul.f16x2 r3803, r25, r3805;
}
{
add.f16x2 r3806, r3782, r3803;
}
{
cvt.rn.f16.f64 rs592, fd700;
}
mov.b32 r3811, {rs592, rs592};
{
mul.f16x2 r3809, r34, r3811;
}
{
add.f16x2 r3812, r3788, r3809;
}
{
cvt.rn.f16.f64 rs593, fd759;
}
mov.b32 r3817, {rs593, rs593};
{
mul.f16x2 r3815, r28, r3817;
}
{
add.f16x2 r3818, r3794, r3815;
}
{
cvt.rn.f16.f64 rs594, fd700;
}
mov.b32 r3823, {rs594, rs594};
{
mul.f16x2 r3821, r31, r3823;
}
{
add.f16x2 r3824, r3800, r3821;
}
{
cvt.rn.f16.f64 rs595, fd731;
}
mov.b32 r3829, {rs595, rs595};
{
mul.f16x2 r3827, r37, r3829;
}
{
add.f16x2 r3830, r3806, r3827;
}
{
cvt.rn.f16.f64 rs596, fd576;
}
mov.b32 r3835, {rs596, rs596};
{
mul.f16x2 r3833, r46, r3835;
}
{
add.f16x2 r3836, r3812, r3833;
}
{
cvt.rn.f16.f64 rs597, fd731;
}
mov.b32 r3841, {rs597, rs597};
{
mul.f16x2 r3839, r40, r3841;
}
{
add.f16x2 r3842, r3818, r3839;
}
{
cvt.rn.f16.f64 rs598, fd576;
}
mov.b32 r3847, {rs598, rs598};
{
mul.f16x2 r3845, r43, r3847;
}
{
add.f16x2 r3848, r3824, r3845;
}
{
cvt.rn.f16.f64 rs599, fd751;
}
mov.b32 r3853, {rs599, rs599};
{
mul.f16x2 r3851, r49, r3853;
}
{
add.f16x2 r3854, r3830, r3851;
}
{
cvt.rn.f16.f64 rs600, fd752;
}
mov.b32 r3859, {rs600, rs600};
{
mul.f16x2 r3857, r58, r3859;
}
{
add.f16x2 r3860, r3836, r3857;
}
{
cvt.rn.f16.f64 rs601, fd751;
}
mov.b32 r3865, {rs601, rs601};
{
mul.f16x2 r3863, r52, r3865;
}
{
add.f16x2 r3866, r3842, r3863;
}
{
cvt.rn.f16.f64 rs602, fd752;
}
mov.b32 r3871, {rs602, rs602};
{
mul.f16x2 r3869, r55, r3871;
}
{
add.f16x2 r3872, r3848, r3869;
}
{
cvt.rn.f16.f64 rs603, fd779;
}
mov.b32 r3877, {rs603, rs603};
{
mul.f16x2 r3875, r61, r3877;
}
{
add.f16x2 r3878, r3854, r3875;
}
{
cvt.rn.f16.f64 rs604, fd780;
}
mov.b32 r3883, {rs604, rs604};
{
mul.f16x2 r3881, r70, r3883;
}
{
add.f16x2 r3884, r3860, r3881;
}
{
cvt.rn.f16.f64 rs605, fd779;
}
mov.b32 r3889, {rs605, rs605};
{
mul.f16x2 r3887, r64, r3889;
}
{
add.f16x2 r3890, r3866, r3887;
}
{
cvt.rn.f16.f64 rs606, fd780;
}
mov.b32 r3895, {rs606, rs606};
{
mul.f16x2 r3893, r67, r3895;
}
{
add.f16x2 r3896, r3872, r3893;
}
{
cvt.rn.f16.f64 rs607, fd763;
}
mov.b32 r3901, {rs607, rs607};
{
mul.f16x2 r3899, r73, r3901;
}
{
add.f16x2 r3902, r3878, r3899;
}
{
cvt.rn.f16.f64 rs608, fd632;
}
mov.b32 r3907, {rs608, rs608};
{
mul.f16x2 r3905, r82, r3907;
}
{
add.f16x2 r3908, r3884, r3905;
}
{
cvt.rn.f16.f64 rs609, fd763;
}
mov.b32 r3913, {rs609, rs609};
{
mul.f16x2 r3911, r76, r3913;
}
{
add.f16x2 r3914, r3890, r3911;
}
{
cvt.rn.f16.f64 rs610, fd632;
}
mov.b32 r3919, {rs610, rs610};
{
mul.f16x2 r3917, r79, r3919;
}
{
add.f16x2 r3920, r3896, r3917;
}
{
cvt.rn.f16.f64 rs611, fd735;
}
mov.b32 r3925, {rs611, rs611};
{
mul.f16x2 r3923, r85, r3925;
}
{
add.f16x2 r3926, r3902, r3923;
}
{
cvt.rn.f16.f64 rs612, fd708;
}
mov.b32 r3931, {rs612, rs612};
{
mul.f16x2 r3929, r94, r3931;
}
{
add.f16x2 r3932, r3908, r3929;
}
{
cvt.rn.f16.f64 rs613, fd735;
}
mov.b32 r3937, {rs613, rs613};
{
mul.f16x2 r3935, r88, r3937;
}
{
add.f16x2 r3938, r3914, r3935;
}
{
cvt.rn.f16.f64 rs614, fd708;
}
mov.b32 r3943, {rs614, rs614};
{
mul.f16x2 r3941, r91, r3943;
}
{
add.f16x2 r3944, r3920, r3941;
}
{
cvt.rn.f16.f64 rs615, fd747;
}
mov.b32 r3949, {rs615, rs615};
{
mul.f16x2 r3947, r97, r3949;
}
{
add.f16x2 r3950, r3926, r3947;
}
{
cvt.rn.f16.f64 rs616, fd748;
}
mov.b32 r3955, {rs616, rs616};
{
mul.f16x2 r3953, r106, r3955;
}
{
add.f16x2 r3956, r3932, r3953;
}
{
cvt.rn.f16.f64 rs617, fd747;
}
mov.b32 r3961, {rs617, rs617};
{
mul.f16x2 r3959, r100, r3961;
}
{
add.f16x2 r3962, r3938, r3959;
}
{
cvt.rn.f16.f64 rs618, fd748;
}
mov.b32 r3967, {rs618, rs618};
{
mul.f16x2 r3965, r103, r3967;
}
{
add.f16x2 r3968, r3944, r3965;
}
{
cvt.rn.f16.f64 rs619, fd775;
}
mov.b32 r3973, {rs619, rs619};
{
mul.f16x2 r3971, r109, r3973;
}
{
add.f16x2 r3974, r3950, r3971;
}
{
cvt.rn.f16.f64 rs620, fd776;
}
mov.b32 r3979, {rs620, rs620};
{
mul.f16x2 r3977, r118, r3979;
}
{
add.f16x2 r3980, r3956, r3977;
}
{
cvt.rn.f16.f64 rs621, fd775;
}
mov.b32 r3985, {rs621, rs621};
{
mul.f16x2 r3983, r112, r3985;
}
{
add.f16x2 r3986, r3962, r3983;
}
{
cvt.rn.f16.f64 rs622, fd776;
}
mov.b32 r3991, {rs622, rs622};
{
mul.f16x2 r3989, r115, r3991;
}
{
add.f16x2 r3992, r3968, r3989;
}
{
cvt.rn.f16.f64 rs623, fd767;
}
mov.b32 r3997, {rs623, rs623};
{
mul.f16x2 r3995, r121, r3997;
}
{
add.f16x2 r3998, r3974, r3995;
}
{
cvt.rn.f16.f64 rs624, fd604;
}
mov.b32 r4003, {rs624, rs624};
{
mul.f16x2 r4001, r130, r4003;
}
{
add.f16x2 r4004, r3980, r4001;
}
{
cvt.rn.f16.f64 rs625, fd767;
}
mov.b32 r4009, {rs625, rs625};
{
mul.f16x2 r4007, r124, r4009;
}
{
add.f16x2 r4010, r3986, r4007;
}
{
cvt.rn.f16.f64 rs626, fd604;
}
mov.b32 r4015, {rs626, rs626};
{
mul.f16x2 r4013, r127, r4015;
}
{
add.f16x2 r4016, r3992, r4013;
}
{
cvt.rn.f16.f64 rs627, fd739;
}
mov.b32 r4021, {rs627, rs627};
{
mul.f16x2 r4019, r133, r4021;
}
{
add.f16x2 r4022, r3998, r4019;
}
{
cvt.rn.f16.f64 rs628, fd660;
}
mov.b32 r4027, {rs628, rs628};
{
mul.f16x2 r4025, r142, r4027;
}
{
add.f16x2 r4028, r4004, r4025;
}
{
cvt.rn.f16.f64 rs629, fd739;
}
mov.b32 r4033, {rs629, rs629};
{
mul.f16x2 r4031, r136, r4033;
}
{
add.f16x2 r4034, r4010, r4031;
}
{
cvt.rn.f16.f64 rs630, fd660;
}
mov.b32 r4039, {rs630, rs630};
{
mul.f16x2 r4037, r139, r4039;
}
{
add.f16x2 r4040, r4016, r4037;
}
{
cvt.rn.f16.f64 rs631, fd743;
}
mov.b32 r4045, {rs631, rs631};
{
mul.f16x2 r4043, r145, r4045;
}
{
add.f16x2 r4046, r4022, r4043;
}
{
cvt.rn.f16.f64 rs632, fd744;
}
mov.b32 r4051, {rs632, rs632};
{
mul.f16x2 r4049, r154, r4051;
}
{
add.f16x2 r4052, r4028, r4049;
}
{
cvt.rn.f16.f64 rs633, fd743;
}
mov.b32 r4057, {rs633, rs633};
{
mul.f16x2 r4055, r148, r4057;
}
{
add.f16x2 r4058, r4034, r4055;
}
{
cvt.rn.f16.f64 rs634, fd744;
}
mov.b32 r4063, {rs634, rs634};
{
mul.f16x2 r4061, r151, r4063;
}
{
add.f16x2 r4064, r4040, r4061;
}
{
cvt.rn.f16.f64 rs635, fd771;
}
mov.b32 r4069, {rs635, rs635};
{
mul.f16x2 r4067, r157, r4069;
}
{
add.f16x2 r4070, r4046, r4067;
}
{
cvt.rn.f16.f64 rs636, fd772;
}
mov.b32 r4075, {rs636, rs636};
{
mul.f16x2 r4073, r166, r4075;
}
{
add.f16x2 r4076, r4052, r4073;
}
{
cvt.rn.f16.f64 rs637, fd771;
}
mov.b32 r4081, {rs637, rs637};
{
mul.f16x2 r4079, r160, r4081;
}
{
add.f16x2 r4082, r4058, r4079;
}
{
cvt.rn.f16.f64 rs638, fd772;
}
mov.b32 r4087, {rs638, rs638};
{
mul.f16x2 r4085, r163, r4087;
}
{
add.f16x2 r4088, r4064, r4085;
}
{
sub.f16x2 %22, r4070, r4076;
}
{
add.f16x2 %23, r4082, r4088;
}
{
add.f16x2 %36, r4070, r4076;
}
{
sub.f16x2 %37, r4082, r4088;
}
cvt.rn.f16.s32 rs639, r4804;
mov.b32 r4115, {rs639, rs639};
cvt.rn.f16.s32 rs640, r4804;
mov.b32 r4127, {rs640, rs640};
{
cvt.rn.f16.f64 rs641, fd747;
}
mov.b32 r4107, {rs641, rs641};
{
mul.f16x2 r4105, r1, r4107;
}
{
add.f16x2 r4108, %58, r4105;
}
{
cvt.rn.f16.f64 rs642, fd748;
}
mov.b32 r4113, {rs642, rs642};
{
mul.f16x2 r4111, r10, r4113;
}
{
add.f16x2 r4114, r4115, r4111;
}
{
cvt.rn.f16.f64 rs643, fd747;
}
mov.b32 r4119, {rs643, rs643};
{
mul.f16x2 r4117, r4, r4119;
}
{
add.f16x2 r4120, %59, r4117;
}
{
cvt.rn.f16.f64 rs644, fd748;
}
mov.b32 r4125, {rs644, rs644};
{
mul.f16x2 r4123, r7, r4125;
}
{
add.f16x2 r4126, r4127, r4123;
}
{
cvt.rn.f16.f64 rs645, fd767;
}
mov.b32 r4131, {rs645, rs645};
{
mul.f16x2 r4129, r13, r4131;
}
{
add.f16x2 r4132, r4108, r4129;
}
{
cvt.rn.f16.f64 rs646, fd768;
}
mov.b32 r4137, {rs646, rs646};
{
mul.f16x2 r4135, r22, r4137;
}
{
add.f16x2 r4138, r4114, r4135;
}
{
cvt.rn.f16.f64 rs647, fd767;
}
mov.b32 r4143, {rs647, rs647};
{
mul.f16x2 r4141, r16, r4143;
}
{
add.f16x2 r4144, r4120, r4141;
}
{
cvt.rn.f16.f64 rs648, fd768;
}
mov.b32 r4149, {rs648, rs648};
{
mul.f16x2 r4147, r19, r4149;
}
{
add.f16x2 r4150, r4126, r4147;
}
{
cvt.rn.f16.f64 rs649, fd783;
}
mov.b32 r4155, {rs649, rs649};
{
mul.f16x2 r4153, r25, r4155;
}
{
add.f16x2 r4156, r4132, r4153;
}
{
cvt.rn.f16.f64 rs650, fd692;
}
mov.b32 r4161, {rs650, rs650};
{
mul.f16x2 r4159, r34, r4161;
}
{
add.f16x2 r4162, r4138, r4159;
}
{
cvt.rn.f16.f64 rs651, fd783;
}
mov.b32 r4167, {rs651, rs651};
{
mul.f16x2 r4165, r28, r4167;
}
{
add.f16x2 r4168, r4144, r4165;
}
{
cvt.rn.f16.f64 rs652, fd692;
}
mov.b32 r4173, {rs652, rs652};
{
mul.f16x2 r4171, r31, r4173;
}
{
add.f16x2 r4174, r4150, r4171;
}
{
cvt.rn.f16.f64 rs653, fd763;
}
mov.b32 r4179, {rs653, rs653};
{
mul.f16x2 r4177, r37, r4179;
}
{
add.f16x2 r4180, r4156, r4177;
}
{
cvt.rn.f16.f64 rs654, fd632;
}
mov.b32 r4185, {rs654, rs654};
{
mul.f16x2 r4183, r46, r4185;
}
{
add.f16x2 r4186, r4162, r4183;
}
{
cvt.rn.f16.f64 rs655, fd763;
}
mov.b32 r4191, {rs655, rs655};
{
mul.f16x2 r4189, r40, r4191;
}
{
add.f16x2 r4192, r4168, r4189;
}
{
cvt.rn.f16.f64 rs656, fd632;
}
mov.b32 r4197, {rs656, rs656};
{
mul.f16x2 r4195, r43, r4197;
}
{
add.f16x2 r4198, r4174, r4195;
}
{
cvt.rn.f16.f64 rs657, fd743;
}
mov.b32 r4203, {rs657, rs657};
{
mul.f16x2 r4201, r49, r4203;
}
{
add.f16x2 r4204, r4180, r4201;
}
{
cvt.rn.f16.f64 rs658, fd636;
}
mov.b32 r4209, {rs658, rs658};
{
mul.f16x2 r4207, r58, r4209;
}
{
add.f16x2 r4210, r4186, r4207;
}
{
cvt.rn.f16.f64 rs659, fd743;
}
mov.b32 r4215, {rs659, rs659};
{
mul.f16x2 r4213, r52, r4215;
}
{
add.f16x2 r4216, r4192, r4213;
}
{
cvt.rn.f16.f64 rs660, fd636;
}
mov.b32 r4221, {rs660, rs660};
{
mul.f16x2 r4219, r55, r4221;
}
{
add.f16x2 r4222, r4198, r4219;
}
{
cvt.rn.f16.f64 rs661, fd731;
}
mov.b32 r4227, {rs661, rs661};
{
mul.f16x2 r4225, r61, r4227;
}
{
add.f16x2 r4228, r4204, r4225;
}
{
cvt.rn.f16.f64 rs662, fd732;
}
mov.b32 r4233, {rs662, rs662};
{
mul.f16x2 r4231, r70, r4233;
}
{
add.f16x2 r4234, r4210, r4231;
}
{
cvt.rn.f16.f64 rs663, fd731;
}
mov.b32 r4239, {rs663, rs663};
{
mul.f16x2 r4237, r64, r4239;
}
{
add.f16x2 r4240, r4216, r4237;
}
{
cvt.rn.f16.f64 rs664, fd732;
}
mov.b32 r4245, {rs664, rs664};
{
mul.f16x2 r4243, r67, r4245;
}
{
add.f16x2 r4246, r4222, r4243;
}
{
cvt.rn.f16.f64 rs665, fd751;
}
mov.b32 r4251, {rs665, rs665};
{
mul.f16x2 r4249, r73, r4251;
}
{
add.f16x2 r4252, r4228, r4249;
}
{
cvt.rn.f16.f64 rs666, fd752;
}
mov.b32 r4257, {rs666, rs666};
{
mul.f16x2 r4255, r82, r4257;
}
{
add.f16x2 r4258, r4234, r4255;
}
{
cvt.rn.f16.f64 rs667, fd751;
}
mov.b32 r4263, {rs667, rs667};
{
mul.f16x2 r4261, r76, r4263;
}
{
add.f16x2 r4264, r4240, r4261;
}
{
cvt.rn.f16.f64 rs668, fd752;
}
mov.b32 r4269, {rs668, rs668};
{
mul.f16x2 r4267, r79, r4269;
}
{
add.f16x2 r4270, r4246, r4267;
}
{
cvt.rn.f16.f64 rs669, fd771;
}
mov.b32 r4275, {rs669, rs669};
{
mul.f16x2 r4273, r85, r4275;
}
{
add.f16x2 r4276, r4252, r4273;
}
{
cvt.rn.f16.f64 rs670, fd772;
}
mov.b32 r4281, {rs670, rs670};
{
mul.f16x2 r4279, r94, r4281;
}
{
add.f16x2 r4282, r4258, r4279;
}
{
cvt.rn.f16.f64 rs671, fd771;
}
mov.b32 r4287, {rs671, rs671};
{
mul.f16x2 r4285, r88, r4287;
}
{
add.f16x2 r4288, r4264, r4285;
}
{
cvt.rn.f16.f64 rs672, fd772;
}
mov.b32 r4293, {rs672, rs672};
{
mul.f16x2 r4291, r91, r4293;
}
{
add.f16x2 r4294, r4270, r4291;
}
{
cvt.rn.f16.f64 rs673, fd779;
}
mov.b32 r4299, {rs673, rs673};
{
mul.f16x2 r4297, r97, r4299;
}
{
add.f16x2 r4300, r4276, r4297;
}
{
cvt.rn.f16.f64 rs674, fd652;
}
mov.b32 r4305, {rs674, rs674};
{
mul.f16x2 r4303, r106, r4305;
}
{
add.f16x2 r4306, r4282, r4303;
}
{
cvt.rn.f16.f64 rs675, fd779;
}
mov.b32 r4311, {rs675, rs675};
{
mul.f16x2 r4309, r100, r4311;
}
{
add.f16x2 r4312, r4288, r4309;
}
{
cvt.rn.f16.f64 rs676, fd652;
}
mov.b32 r4317, {rs676, rs676};
{
mul.f16x2 r4315, r103, r4317;
}
{
add.f16x2 r4318, r4294, r4315;
}
{
cvt.rn.f16.f64 rs677, fd759;
}
mov.b32 r4323, {rs677, rs677};
{
mul.f16x2 r4321, r109, r4323;
}
{
add.f16x2 r4324, r4300, r4321;
}
{
cvt.rn.f16.f64 rs678, fd700;
}
mov.b32 r4329, {rs678, rs678};
{
mul.f16x2 r4327, r118, r4329;
}
{
add.f16x2 r4330, r4306, r4327;
}
{
cvt.rn.f16.f64 rs679, fd759;
}
mov.b32 r4335, {rs679, rs679};
{
mul.f16x2 r4333, r112, r4335;
}
{
add.f16x2 r4336, r4312, r4333;
}
{
cvt.rn.f16.f64 rs680, fd700;
}
mov.b32 r4341, {rs680, rs680};
{
mul.f16x2 r4339, r115, r4341;
}
{
add.f16x2 r4342, r4318, r4339;
}
{
cvt.rn.f16.f64 rs681, fd739;
}
mov.b32 r4347, {rs681, rs681};
{
mul.f16x2 r4345, r121, r4347;
}
{
add.f16x2 r4348, r4324, r4345;
}
{
cvt.rn.f16.f64 rs682, fd660;
}
mov.b32 r4353, {rs682, rs682};
{
mul.f16x2 r4351, r130, r4353;
}
{
add.f16x2 r4354, r4330, r4351;
}
{
cvt.rn.f16.f64 rs683, fd739;
}
mov.b32 r4359, {rs683, rs683};
{
mul.f16x2 r4357, r124, r4359;
}
{
add.f16x2 r4360, r4336, r4357;
}
{
cvt.rn.f16.f64 rs684, fd660;
}
mov.b32 r4365, {rs684, rs684};
{
mul.f16x2 r4363, r127, r4365;
}
{
add.f16x2 r4366, r4342, r4363;
}
{
cvt.rn.f16.f64 rs685, fd735;
}
mov.b32 r4371, {rs685, rs685};
{
mul.f16x2 r4369, r133, r4371;
}
{
add.f16x2 r4372, r4348, r4369;
}
{
cvt.rn.f16.f64 rs686, fd736;
}
mov.b32 r4377, {rs686, rs686};
{
mul.f16x2 r4375, r142, r4377;
}
{
add.f16x2 r4378, r4354, r4375;
}
{
cvt.rn.f16.f64 rs687, fd735;
}
mov.b32 r4383, {rs687, rs687};
{
mul.f16x2 r4381, r136, r4383;
}
{
add.f16x2 r4384, r4360, r4381;
}
{
cvt.rn.f16.f64 rs688, fd736;
}
mov.b32 r4389, {rs688, rs688};
{
mul.f16x2 r4387, r139, r4389;
}
{
add.f16x2 r4390, r4366, r4387;
}
{
cvt.rn.f16.f64 rs689, fd755;
}
mov.b32 r4395, {rs689, rs689};
{
mul.f16x2 r4393, r145, r4395;
}
{
add.f16x2 r4396, r4372, r4393;
}
{
cvt.rn.f16.f64 rs690, fd756;
}
mov.b32 r4401, {rs690, rs690};
{
mul.f16x2 r4399, r154, r4401;
}
{
add.f16x2 r4402, r4378, r4399;
}
{
cvt.rn.f16.f64 rs691, fd755;
}
mov.b32 r4407, {rs691, rs691};
{
mul.f16x2 r4405, r148, r4407;
}
{
add.f16x2 r4408, r4384, r4405;
}
{
cvt.rn.f16.f64 rs692, fd756;
}
mov.b32 r4413, {rs692, rs692};
{
mul.f16x2 r4411, r151, r4413;
}
{
add.f16x2 r4414, r4390, r4411;
}
{
cvt.rn.f16.f64 rs693, fd775;
}
mov.b32 r4419, {rs693, rs693};
{
mul.f16x2 r4417, r157, r4419;
}
{
add.f16x2 r4420, r4396, r4417;
}
{
cvt.rn.f16.f64 rs694, fd776;
}
mov.b32 r4425, {rs694, rs694};
{
mul.f16x2 r4423, r166, r4425;
}
{
add.f16x2 r4426, r4402, r4423;
}
{
cvt.rn.f16.f64 rs695, fd775;
}
mov.b32 r4431, {rs695, rs695};
{
mul.f16x2 r4429, r160, r4431;
}
{
add.f16x2 r4432, r4408, r4429;
}
{
cvt.rn.f16.f64 rs696, fd776;
}
mov.b32 r4437, {rs696, rs696};
{
mul.f16x2 r4435, r163, r4437;
}
{
add.f16x2 r4438, r4414, r4435;
}
{
sub.f16x2 %24, r4420, r4426;
}
{
add.f16x2 %25, r4432, r4438;
}
{
add.f16x2 %34, r4420, r4426;
}
{
sub.f16x2 %35, r4432, r4438;
}
cvt.rn.f16.s32 rs697, r4804;
mov.b32 r4465, {rs697, rs697};
cvt.rn.f16.s32 rs698, r4804;
mov.b32 r4477, {rs698, rs698};
{
cvt.rn.f16.f64 rs699, fd739;
}
mov.b32 r4457, {rs699, rs699};
{
mul.f16x2 r4455, r1, r4457;
}
{
add.f16x2 r4458, %58, r4455;
}
{
cvt.rn.f16.f64 rs700, fd740;
}
mov.b32 r4463, {rs700, rs700};
{
mul.f16x2 r4461, r10, r4463;
}
{
add.f16x2 r4464, r4465, r4461;
}
{
cvt.rn.f16.f64 rs701, fd739;
}
mov.b32 r4469, {rs701, rs701};
{
mul.f16x2 r4467, r4, r4469;
}
{
add.f16x2 r4470, %59, r4467;
}
{
cvt.rn.f16.f64 rs702, fd740;
}
mov.b32 r4475, {rs702, rs702};
{
mul.f16x2 r4473, r7, r4475;
}
{
add.f16x2 r4476, r4477, r4473;
}
{
cvt.rn.f16.f64 rs703, fd751;
}
mov.b32 r4481, {rs703, rs703};
{
mul.f16x2 r4479, r13, r4481;
}
{
add.f16x2 r4482, r4458, r4479;
}
{
cvt.rn.f16.f64 rs704, fd752;
}
mov.b32 r4487, {rs704, rs704};
{
mul.f16x2 r4485, r22, r4487;
}
{
add.f16x2 r4488, r4464, r4485;
}
{
cvt.rn.f16.f64 rs705, fd751;
}
mov.b32 r4493, {rs705, rs705};
{
mul.f16x2 r4491, r16, r4493;
}
{
add.f16x2 r4494, r4470, r4491;
}
{
cvt.rn.f16.f64 rs706, fd752;
}
mov.b32 r4499, {rs706, rs706};
{
mul.f16x2 r4497, r19, r4499;
}
{
add.f16x2 r4500, r4476, r4497;
}
{
cvt.rn.f16.f64 rs707, fd763;
}
mov.b32 r4505, {rs707, rs707};
{
mul.f16x2 r4503, r25, r4505;
}
{
add.f16x2 r4506, r4482, r4503;
}
{
cvt.rn.f16.f64 rs708, fd764;
}
mov.b32 r4511, {rs708, rs708};
{
mul.f16x2 r4509, r34, r4511;
}
{
add.f16x2 r4512, r4488, r4509;
}
{
cvt.rn.f16.f64 rs709, fd763;
}
mov.b32 r4517, {rs709, rs709};
{
mul.f16x2 r4515, r28, r4517;
}
{
add.f16x2 r4518, r4494, r4515;
}
{
cvt.rn.f16.f64 rs710, fd764;
}
mov.b32 r4523, {rs710, rs710};
{
mul.f16x2 r4521, r31, r4523;
}
{
add.f16x2 r4524, r4500, r4521;
}
{
cvt.rn.f16.f64 rs711, fd775;
}
mov.b32 r4529, {rs711, rs711};
{
mul.f16x2 r4527, r37, r4529;
}
{
add.f16x2 r4530, r4506, r4527;
}
{
cvt.rn.f16.f64 rs712, fd776;
}
mov.b32 r4535, {rs712, rs712};
{
mul.f16x2 r4533, r46, r4535;
}
{
add.f16x2 r4536, r4512, r4533;
}
{
cvt.rn.f16.f64 rs713, fd775;
}
mov.b32 r4541, {rs713, rs713};
{
mul.f16x2 r4539, r40, r4541;
}
{
add.f16x2 r4542, r4518, r4539;
}
{
cvt.rn.f16.f64 rs714, fd776;
}
mov.b32 r4547, {rs714, rs714};
{
mul.f16x2 r4545, r43, r4547;
}
{
add.f16x2 r4548, r4524, r4545;
}
{
cvt.rn.f16.f64 rs715, fd783;
}
mov.b32 r4553, {rs715, rs715};
{
mul.f16x2 r4551, r49, r4553;
}
{
add.f16x2 r4554, r4530, r4551;
}
{
cvt.rn.f16.f64 rs716, fd692;
}
mov.b32 r4559, {rs716, rs716};
{
mul.f16x2 r4557, r58, r4559;
}
{
add.f16x2 r4560, r4536, r4557;
}
{
cvt.rn.f16.f64 rs717, fd783;
}
mov.b32 r4565, {rs717, rs717};
{
mul.f16x2 r4563, r52, r4565;
}
{
add.f16x2 r4566, r4542, r4563;
}
{
cvt.rn.f16.f64 rs718, fd692;
}
mov.b32 r4571, {rs718, rs718};
{
mul.f16x2 r4569, r55, r4571;
}
{
add.f16x2 r4572, r4548, r4569;
}
{
cvt.rn.f16.f64 rs719, fd771;
}
mov.b32 r4577, {rs719, rs719};
{
mul.f16x2 r4575, r61, r4577;
}
{
add.f16x2 r4578, r4554, r4575;
}
{
cvt.rn.f16.f64 rs720, fd696;
}
mov.b32 r4583, {rs720, rs720};
{
mul.f16x2 r4581, r70, r4583;
}
{
add.f16x2 r4584, r4560, r4581;
}
{
cvt.rn.f16.f64 rs721, fd771;
}
mov.b32 r4589, {rs721, rs721};
{
mul.f16x2 r4587, r64, r4589;
}
{
add.f16x2 r4590, r4566, r4587;
}
{
cvt.rn.f16.f64 rs722, fd696;
}
mov.b32 r4595, {rs722, rs722};
{
mul.f16x2 r4593, r67, r4595;
}
{
add.f16x2 r4596, r4572, r4593;
}
{
cvt.rn.f16.f64 rs723, fd759;
}
mov.b32 r4601, {rs723, rs723};
{
mul.f16x2 r4599, r73, r4601;
}
{
add.f16x2 r4602, r4578, r4599;
}
{
cvt.rn.f16.f64 rs724, fd700;
}
mov.b32 r4607, {rs724, rs724};
{
mul.f16x2 r4605, r82, r4607;
}
{
add.f16x2 r4608, r4584, r4605;
}
{
cvt.rn.f16.f64 rs725, fd759;
}
mov.b32 r4613, {rs725, rs725};
{
mul.f16x2 r4611, r76, r4613;
}
{
add.f16x2 r4614, r4590, r4611;
}
{
cvt.rn.f16.f64 rs726, fd700;
}
mov.b32 r4619, {rs726, rs726};
{
mul.f16x2 r4617, r79, r4619;
}
{
add.f16x2 r4620, r4596, r4617;
}
{
cvt.rn.f16.f64 rs727, fd747;
}
mov.b32 r4625, {rs727, rs727};
{
mul.f16x2 r4623, r85, r4625;
}
{
add.f16x2 r4626, r4602, r4623;
}
{
cvt.rn.f16.f64 rs728, fd704;
}
mov.b32 r4631, {rs728, rs728};
{
mul.f16x2 r4629, r94, r4631;
}
{
add.f16x2 r4632, r4608, r4629;
}
{
cvt.rn.f16.f64 rs729, fd747;
}
mov.b32 r4637, {rs729, rs729};
{
mul.f16x2 r4635, r88, r4637;
}
{
add.f16x2 r4638, r4614, r4635;
}
{
cvt.rn.f16.f64 rs730, fd704;
}
mov.b32 r4643, {rs730, rs730};
{
mul.f16x2 r4641, r91, r4643;
}
{
add.f16x2 r4644, r4620, r4641;
}
{
cvt.rn.f16.f64 rs731, fd735;
}
mov.b32 r4649, {rs731, rs731};
{
mul.f16x2 r4647, r97, r4649;
}
{
add.f16x2 r4650, r4626, r4647;
}
{
cvt.rn.f16.f64 rs732, fd708;
}
mov.b32 r4655, {rs732, rs732};
{
mul.f16x2 r4653, r106, r4655;
}
{
add.f16x2 r4656, r4632, r4653;
}
{
cvt.rn.f16.f64 rs733, fd735;
}
mov.b32 r4661, {rs733, rs733};
{
mul.f16x2 r4659, r100, r4661;
}
{
add.f16x2 r4662, r4638, r4659;
}
{
cvt.rn.f16.f64 rs734, fd708;
}
mov.b32 r4667, {rs734, rs734};
{
mul.f16x2 r4665, r103, r4667;
}
{
add.f16x2 r4668, r4644, r4665;
}
{
cvt.rn.f16.f64 rs735, fd731;
}
mov.b32 r4673, {rs735, rs735};
{
mul.f16x2 r4671, r109, r4673;
}
{
add.f16x2 r4674, r4650, r4671;
}
{
cvt.rn.f16.f64 rs736, fd732;
}
mov.b32 r4679, {rs736, rs736};
{
mul.f16x2 r4677, r118, r4679;
}
{
add.f16x2 r4680, r4656, r4677;
}
{
cvt.rn.f16.f64 rs737, fd731;
}
mov.b32 r4685, {rs737, rs737};
{
mul.f16x2 r4683, r112, r4685;
}
{
add.f16x2 r4686, r4662, r4683;
}
{
cvt.rn.f16.f64 rs738, fd732;
}
mov.b32 r4691, {rs738, rs738};
{
mul.f16x2 r4689, r115, r4691;
}
{
add.f16x2 r4692, r4668, r4689;
}
{
cvt.rn.f16.f64 rs739, fd743;
}
mov.b32 r4697, {rs739, rs739};
{
mul.f16x2 r4695, r121, r4697;
}
{
add.f16x2 r4698, r4674, r4695;
}
{
cvt.rn.f16.f64 rs740, fd744;
}
mov.b32 r4703, {rs740, rs740};
{
mul.f16x2 r4701, r130, r4703;
}
{
add.f16x2 r4704, r4680, r4701;
}
{
cvt.rn.f16.f64 rs741, fd743;
}
mov.b32 r4709, {rs741, rs741};
{
mul.f16x2 r4707, r124, r4709;
}
{
add.f16x2 r4710, r4686, r4707;
}
{
cvt.rn.f16.f64 rs742, fd744;
}
mov.b32 r4715, {rs742, rs742};
{
mul.f16x2 r4713, r127, r4715;
}
{
add.f16x2 r4716, r4692, r4713;
}
{
cvt.rn.f16.f64 rs743, fd755;
}
mov.b32 r4721, {rs743, rs743};
{
mul.f16x2 r4719, r133, r4721;
}
{
add.f16x2 r4722, r4698, r4719;
}
{
cvt.rn.f16.f64 rs744, fd756;
}
mov.b32 r4727, {rs744, rs744};
{
mul.f16x2 r4725, r142, r4727;
}
{
add.f16x2 r4728, r4704, r4725;
}
{
cvt.rn.f16.f64 rs745, fd755;
}
mov.b32 r4733, {rs745, rs745};
{
mul.f16x2 r4731, r136, r4733;
}
{
add.f16x2 r4734, r4710, r4731;
}
{
cvt.rn.f16.f64 rs746, fd756;
}
mov.b32 r4739, {rs746, rs746};
{
mul.f16x2 r4737, r139, r4739;
}
{
add.f16x2 r4740, r4716, r4737;
}
{
cvt.rn.f16.f64 rs747, fd767;
}
mov.b32 r4745, {rs747, rs747};
{
mul.f16x2 r4743, r145, r4745;
}
{
add.f16x2 r4746, r4722, r4743;
}
{
cvt.rn.f16.f64 rs748, fd768;
}
mov.b32 r4751, {rs748, rs748};
{
mul.f16x2 r4749, r154, r4751;
}
{
add.f16x2 r4752, r4728, r4749;
}
{
cvt.rn.f16.f64 rs749, fd767;
}
mov.b32 r4757, {rs749, rs749};
{
mul.f16x2 r4755, r148, r4757;
}
{
add.f16x2 r4758, r4734, r4755;
}
{
cvt.rn.f16.f64 rs750, fd768;
}
mov.b32 r4763, {rs750, rs750};
{
mul.f16x2 r4761, r151, r4763;
}
{
add.f16x2 r4764, r4740, r4761;
}
{
cvt.rn.f16.f64 rs751, fd779;
}
mov.b32 r4769, {rs751, rs751};
{
mul.f16x2 r4767, r157, r4769;
}
{
add.f16x2 r4770, r4746, r4767;
}
{
cvt.rn.f16.f64 rs752, fd780;
}
mov.b32 r4775, {rs752, rs752};
{
mul.f16x2 r4773, r166, r4775;
}
{
add.f16x2 r4776, r4752, r4773;
}
{
cvt.rn.f16.f64 rs753, fd779;
}
mov.b32 r4781, {rs753, rs753};
{
mul.f16x2 r4779, r160, r4781;
}
{
add.f16x2 r4782, r4758, r4779;
}
{
cvt.rn.f16.f64 rs754, fd780;
}
mov.b32 r4787, {rs754, rs754};
{
mul.f16x2 r4785, r163, r4787;
}
{
add.f16x2 r4788, r4764, r4785;
}
{
sub.f16x2 %26, r4770, r4776;
}
{
add.f16x2 %27, r4782, r4788;
}
{
add.f16x2 %32, r4770, r4776;
}
{
sub.f16x2 %33, r4782, r4788;
}
cvt.rn.f16.s32 rs755, r4804;
mov.b32 r4815, {rs755, rs755};
cvt.rn.f16.s32 rs756, r4804;
mov.b32 r4827, {rs756, rs756};
{
cvt.rn.f16.f64 rs757, fd731;
}
mov.b32 r4807, {rs757, rs757};
{
mul.f16x2 r4805, r1, r4807;
}
{
add.f16x2 r4808, %58, r4805;
}
{
cvt.rn.f16.f64 rs758, fd732;
}
mov.b32 r4813, {rs758, rs758};
{
mul.f16x2 r4811, r10, r4813;
}
{
add.f16x2 r4814, r4815, r4811;
}
{
cvt.rn.f16.f64 rs759, fd731;
}
mov.b32 r4819, {rs759, rs759};
{
mul.f16x2 r4817, r4, r4819;
}
{
add.f16x2 r4820, %59, r4817;
}
{
cvt.rn.f16.f64 rs760, fd732;
}
mov.b32 r4825, {rs760, rs760};
{
mul.f16x2 r4823, r7, r4825;
}
{
add.f16x2 r4826, r4827, r4823;
}
{
cvt.rn.f16.f64 rs761, fd735;
}
mov.b32 r4831, {rs761, rs761};
{
mul.f16x2 r4829, r13, r4831;
}
{
add.f16x2 r4832, r4808, r4829;
}
{
cvt.rn.f16.f64 rs762, fd736;
}
mov.b32 r4837, {rs762, rs762};
{
mul.f16x2 r4835, r22, r4837;
}
{
add.f16x2 r4838, r4814, r4835;
}
{
cvt.rn.f16.f64 rs763, fd735;
}
mov.b32 r4843, {rs763, rs763};
{
mul.f16x2 r4841, r16, r4843;
}
{
add.f16x2 r4844, r4820, r4841;
}
{
cvt.rn.f16.f64 rs764, fd736;
}
mov.b32 r4849, {rs764, rs764};
{
mul.f16x2 r4847, r19, r4849;
}
{
add.f16x2 r4850, r4826, r4847;
}
{
cvt.rn.f16.f64 rs765, fd739;
}
mov.b32 r4855, {rs765, rs765};
{
mul.f16x2 r4853, r25, r4855;
}
{
add.f16x2 r4856, r4832, r4853;
}
{
cvt.rn.f16.f64 rs766, fd740;
}
mov.b32 r4861, {rs766, rs766};
{
mul.f16x2 r4859, r34, r4861;
}
{
add.f16x2 r4862, r4838, r4859;
}
{
cvt.rn.f16.f64 rs767, fd739;
}
mov.b32 r4867, {rs767, rs767};
{
mul.f16x2 r4865, r28, r4867;
}
{
add.f16x2 r4868, r4844, r4865;
}
{
cvt.rn.f16.f64 rs768, fd740;
}
mov.b32 r4873, {rs768, rs768};
{
mul.f16x2 r4871, r31, r4873;
}
{
add.f16x2 r4874, r4850, r4871;
}
{
cvt.rn.f16.f64 rs769, fd743;
}
mov.b32 r4879, {rs769, rs769};
{
mul.f16x2 r4877, r37, r4879;
}
{
add.f16x2 r4880, r4856, r4877;
}
{
cvt.rn.f16.f64 rs770, fd744;
}
mov.b32 r4885, {rs770, rs770};
{
mul.f16x2 r4883, r46, r4885;
}
{
add.f16x2 r4886, r4862, r4883;
}
{
cvt.rn.f16.f64 rs771, fd743;
}
mov.b32 r4891, {rs771, rs771};
{
mul.f16x2 r4889, r40, r4891;
}
{
add.f16x2 r4892, r4868, r4889;
}
{
cvt.rn.f16.f64 rs772, fd744;
}
mov.b32 r4897, {rs772, rs772};
{
mul.f16x2 r4895, r43, r4897;
}
{
add.f16x2 r4898, r4874, r4895;
}
{
cvt.rn.f16.f64 rs773, fd747;
}
mov.b32 r4903, {rs773, rs773};
{
mul.f16x2 r4901, r49, r4903;
}
{
add.f16x2 r4904, r4880, r4901;
}
{
cvt.rn.f16.f64 rs774, fd748;
}
mov.b32 r4909, {rs774, rs774};
{
mul.f16x2 r4907, r58, r4909;
}
{
add.f16x2 r4910, r4886, r4907;
}
{
cvt.rn.f16.f64 rs775, fd747;
}
mov.b32 r4915, {rs775, rs775};
{
mul.f16x2 r4913, r52, r4915;
}
{
add.f16x2 r4916, r4892, r4913;
}
{
cvt.rn.f16.f64 rs776, fd748;
}
mov.b32 r4921, {rs776, rs776};
{
mul.f16x2 r4919, r55, r4921;
}
{
add.f16x2 r4922, r4898, r4919;
}
{
cvt.rn.f16.f64 rs777, fd751;
}
mov.b32 r4927, {rs777, rs777};
{
mul.f16x2 r4925, r61, r4927;
}
{
add.f16x2 r4928, r4904, r4925;
}
{
cvt.rn.f16.f64 rs778, fd752;
}
mov.b32 r4933, {rs778, rs778};
{
mul.f16x2 r4931, r70, r4933;
}
{
add.f16x2 r4934, r4910, r4931;
}
{
cvt.rn.f16.f64 rs779, fd751;
}
mov.b32 r4939, {rs779, rs779};
{
mul.f16x2 r4937, r64, r4939;
}
{
add.f16x2 r4940, r4916, r4937;
}
{
cvt.rn.f16.f64 rs780, fd752;
}
mov.b32 r4945, {rs780, rs780};
{
mul.f16x2 r4943, r67, r4945;
}
{
add.f16x2 r4946, r4922, r4943;
}
{
cvt.rn.f16.f64 rs781, fd755;
}
mov.b32 r4951, {rs781, rs781};
{
mul.f16x2 r4949, r73, r4951;
}
{
add.f16x2 r4952, r4928, r4949;
}
{
cvt.rn.f16.f64 rs782, fd756;
}
mov.b32 r4957, {rs782, rs782};
{
mul.f16x2 r4955, r82, r4957;
}
{
add.f16x2 r4958, r4934, r4955;
}
{
cvt.rn.f16.f64 rs783, fd755;
}
mov.b32 r4963, {rs783, rs783};
{
mul.f16x2 r4961, r76, r4963;
}
{
add.f16x2 r4964, r4940, r4961;
}
{
cvt.rn.f16.f64 rs784, fd756;
}
mov.b32 r4969, {rs784, rs784};
{
mul.f16x2 r4967, r79, r4969;
}
{
add.f16x2 r4970, r4946, r4967;
}
{
cvt.rn.f16.f64 rs785, fd759;
}
mov.b32 r4975, {rs785, rs785};
{
mul.f16x2 r4973, r85, r4975;
}
{
add.f16x2 r4976, r4952, r4973;
}
{
cvt.rn.f16.f64 rs786, fd760;
}
mov.b32 r4981, {rs786, rs786};
{
mul.f16x2 r4979, r94, r4981;
}
{
add.f16x2 r4982, r4958, r4979;
}
{
cvt.rn.f16.f64 rs787, fd759;
}
mov.b32 r4987, {rs787, rs787};
{
mul.f16x2 r4985, r88, r4987;
}
{
add.f16x2 r4988, r4964, r4985;
}
{
cvt.rn.f16.f64 rs788, fd760;
}
mov.b32 r4993, {rs788, rs788};
{
mul.f16x2 r4991, r91, r4993;
}
{
add.f16x2 r4994, r4970, r4991;
}
{
cvt.rn.f16.f64 rs789, fd763;
}
mov.b32 r4999, {rs789, rs789};
{
mul.f16x2 r4997, r97, r4999;
}
{
add.f16x2 r5000, r4976, r4997;
}
{
cvt.rn.f16.f64 rs790, fd764;
}
mov.b32 r5005, {rs790, rs790};
{
mul.f16x2 r5003, r106, r5005;
}
{
add.f16x2 r5006, r4982, r5003;
}
{
cvt.rn.f16.f64 rs791, fd763;
}
mov.b32 r5011, {rs791, rs791};
{
mul.f16x2 r5009, r100, r5011;
}
{
add.f16x2 r5012, r4988, r5009;
}
{
cvt.rn.f16.f64 rs792, fd764;
}
mov.b32 r5017, {rs792, rs792};
{
mul.f16x2 r5015, r103, r5017;
}
{
add.f16x2 r5018, r4994, r5015;
}
{
cvt.rn.f16.f64 rs793, fd767;
}
mov.b32 r5023, {rs793, rs793};
{
mul.f16x2 r5021, r109, r5023;
}
{
add.f16x2 r5024, r5000, r5021;
}
{
cvt.rn.f16.f64 rs794, fd768;
}
mov.b32 r5029, {rs794, rs794};
{
mul.f16x2 r5027, r118, r5029;
}
{
add.f16x2 r5030, r5006, r5027;
}
{
cvt.rn.f16.f64 rs795, fd767;
}
mov.b32 r5035, {rs795, rs795};
{
mul.f16x2 r5033, r112, r5035;
}
{
add.f16x2 r5036, r5012, r5033;
}
{
cvt.rn.f16.f64 rs796, fd768;
}
mov.b32 r5041, {rs796, rs796};
{
mul.f16x2 r5039, r115, r5041;
}
{
add.f16x2 r5042, r5018, r5039;
}
{
cvt.rn.f16.f64 rs797, fd771;
}
mov.b32 r5047, {rs797, rs797};
{
mul.f16x2 r5045, r121, r5047;
}
{
add.f16x2 r5048, r5024, r5045;
}
{
cvt.rn.f16.f64 rs798, fd772;
}
mov.b32 r5053, {rs798, rs798};
{
mul.f16x2 r5051, r130, r5053;
}
{
add.f16x2 r5054, r5030, r5051;
}
{
cvt.rn.f16.f64 rs799, fd771;
}
mov.b32 r5059, {rs799, rs799};
{
mul.f16x2 r5057, r124, r5059;
}
{
add.f16x2 r5060, r5036, r5057;
}
{
cvt.rn.f16.f64 rs800, fd772;
}
mov.b32 r5065, {rs800, rs800};
{
mul.f16x2 r5063, r127, r5065;
}
{
add.f16x2 r5066, r5042, r5063;
}
{
cvt.rn.f16.f64 rs801, fd775;
}
mov.b32 r5071, {rs801, rs801};
{
mul.f16x2 r5069, r133, r5071;
}
{
add.f16x2 r5072, r5048, r5069;
}
{
cvt.rn.f16.f64 rs802, fd776;
}
mov.b32 r5077, {rs802, rs802};
{
mul.f16x2 r5075, r142, r5077;
}
{
add.f16x2 r5078, r5054, r5075;
}
{
cvt.rn.f16.f64 rs803, fd775;
}
mov.b32 r5083, {rs803, rs803};
{
mul.f16x2 r5081, r136, r5083;
}
{
add.f16x2 r5084, r5060, r5081;
}
{
cvt.rn.f16.f64 rs804, fd776;
}
mov.b32 r5089, {rs804, rs804};
{
mul.f16x2 r5087, r139, r5089;
}
{
add.f16x2 r5090, r5066, r5087;
}
{
cvt.rn.f16.f64 rs805, fd779;
}
mov.b32 r5095, {rs805, rs805};
{
mul.f16x2 r5093, r145, r5095;
}
{
add.f16x2 r5096, r5072, r5093;
}
{
cvt.rn.f16.f64 rs806, fd780;
}
mov.b32 r5101, {rs806, rs806};
{
mul.f16x2 r5099, r154, r5101;
}
{
add.f16x2 r5102, r5078, r5099;
}
{
cvt.rn.f16.f64 rs807, fd779;
}
mov.b32 r5107, {rs807, rs807};
{
mul.f16x2 r5105, r148, r5107;
}
{
add.f16x2 r5108, r5084, r5105;
}
{
cvt.rn.f16.f64 rs808, fd780;
}
mov.b32 r5113, {rs808, rs808};
{
mul.f16x2 r5111, r151, r5113;
}
{
add.f16x2 r5114, r5090, r5111;
}
{
cvt.rn.f16.f64 rs809, fd783;
}
mov.b32 r5119, {rs809, rs809};
{
mul.f16x2 r5117, r157, r5119;
}
{
add.f16x2 r5120, r5096, r5117;
}
{
cvt.rn.f16.f64 rs810, fd784;
}
mov.b32 r5125, {rs810, rs810};
{
mul.f16x2 r5123, r166, r5125;
}
{
add.f16x2 r5126, r5102, r5123;
}
{
cvt.rn.f16.f64 rs811, fd783;
}
mov.b32 r5131, {rs811, rs811};
{
mul.f16x2 r5129, r160, r5131;
}
{
add.f16x2 r5132, r5108, r5129;
}
{
cvt.rn.f16.f64 rs812, fd784;
}
mov.b32 r5137, {rs812, rs812};
{
mul.f16x2 r5135, r163, r5137;
}
{
add.f16x2 r5138, r5114, r5135;
}
{
sub.f16x2 %28, r5120, r5126;
}
{
add.f16x2 %29, r5132, r5138;
}
{
add.f16x2 %30, r5120, r5126;
}
{
sub.f16x2 %31, r5132, r5138;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)), "=r"(__HALF2_TO_UI(rmem[27].x)), "=r"(__HALF2_TO_UI(rmem[27].y)), "=r"(__HALF2_TO_UI(rmem[28].x)), "=r"(__HALF2_TO_UI(rmem[28].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[27].x)), "r"(__HALF2_TO_UI(rmem[27].y)), "r"(__HALF2_TO_UI(rmem[28].x)), "r"(__HALF2_TO_UI(rmem[28].y)));
};


#endif
