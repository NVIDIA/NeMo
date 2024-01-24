#ifndef CUFFTDX_FFT_31_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_31_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<963, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<931>;
.reg .b32 r<5881>;
.reg .f64 fd<901>;
.reg .b64 rd<2>;
{
add.f16x2 r1, %64, %122;
}
{
add.f16x2 r4, %65, %123;
}
{
sub.f16x2 r7, %64, %122;
}
{
sub.f16x2 r10, %65, %123;
}
{
add.f16x2 r13, %66, %120;
}
{
add.f16x2 r16, %67, %121;
}
{
sub.f16x2 r19, %66, %120;
}
{
sub.f16x2 r22, %67, %121;
}
{
add.f16x2 r25, %68, %118;
}
{
add.f16x2 r28, %69, %119;
}
{
sub.f16x2 r31, %68, %118;
}
{
sub.f16x2 r34, %69, %119;
}
{
add.f16x2 r37, %70, %116;
}
{
add.f16x2 r40, %71, %117;
}
{
sub.f16x2 r43, %70, %116;
}
{
sub.f16x2 r46, %71, %117;
}
{
add.f16x2 r49, %72, %114;
}
{
add.f16x2 r52, %73, %115;
}
{
sub.f16x2 r55, %72, %114;
}
{
sub.f16x2 r58, %73, %115;
}
{
add.f16x2 r61, %74, %112;
}
{
add.f16x2 r64, %75, %113;
}
{
sub.f16x2 r67, %74, %112;
}
{
sub.f16x2 r70, %75, %113;
}
{
add.f16x2 r73, %76, %110;
}
{
add.f16x2 r76, %77, %111;
}
{
sub.f16x2 r79, %76, %110;
}
{
sub.f16x2 r82, %77, %111;
}
{
add.f16x2 r85, %78, %108;
}
{
add.f16x2 r88, %79, %109;
}
{
sub.f16x2 r91, %78, %108;
}
{
sub.f16x2 r94, %79, %109;
}
{
add.f16x2 r97, %80, %106;
}
{
add.f16x2 r100, %81, %107;
}
{
sub.f16x2 r103, %80, %106;
}
{
sub.f16x2 r106, %81, %107;
}
{
add.f16x2 r109, %82, %104;
}
{
add.f16x2 r112, %83, %105;
}
{
sub.f16x2 r115, %82, %104;
}
{
sub.f16x2 r118, %83, %105;
}
{
add.f16x2 r121, %84, %102;
}
{
add.f16x2 r124, %85, %103;
}
{
sub.f16x2 r127, %84, %102;
}
{
sub.f16x2 r130, %85, %103;
}
{
add.f16x2 r133, %86, %100;
}
{
add.f16x2 r136, %87, %101;
}
{
sub.f16x2 r139, %86, %100;
}
{
sub.f16x2 r142, %87, %101;
}
{
add.f16x2 r145, %88, %98;
}
{
add.f16x2 r148, %89, %99;
}
{
sub.f16x2 r151, %88, %98;
}
{
sub.f16x2 r154, %89, %99;
}
{
add.f16x2 r157, %90, %96;
}
{
add.f16x2 r160, %91, %97;
}
{
sub.f16x2 r163, %90, %96;
}
{
sub.f16x2 r166, %91, %97;
}
{
add.f16x2 r169, %92, %94;
}
{
add.f16x2 r172, %93, %95;
}
{
sub.f16x2 r175, %92, %94;
}
{
sub.f16x2 r178, %93, %95;
}
{
add.f16x2 r181, %62, r1;
}
{
add.f16x2 r184, %63, r4;
}
{
add.f16x2 r187, r181, r13;
}
{
add.f16x2 r190, r184, r16;
}
{
add.f16x2 r193, r187, r25;
}
{
add.f16x2 r196, r190, r28;
}
{
add.f16x2 r199, r193, r37;
}
{
add.f16x2 r202, r196, r40;
}
{
add.f16x2 r205, r199, r49;
}
{
add.f16x2 r208, r202, r52;
}
{
add.f16x2 r211, r205, r61;
}
{
add.f16x2 r214, r208, r64;
}
{
add.f16x2 r217, r211, r73;
}
{
add.f16x2 r220, r214, r76;
}
{
add.f16x2 r223, r217, r85;
}
{
add.f16x2 r226, r220, r88;
}
{
add.f16x2 r229, r223, r97;
}
{
add.f16x2 r232, r226, r100;
}
{
add.f16x2 r235, r229, r109;
}
{
add.f16x2 r238, r232, r112;
}
{
add.f16x2 r241, r235, r121;
}
{
add.f16x2 r244, r238, r124;
}
{
add.f16x2 r247, r241, r133;
}
{
add.f16x2 r250, r244, r136;
}
{
add.f16x2 r253, r247, r145;
}
{
add.f16x2 r256, r250, r148;
}
{
add.f16x2 r259, r253, r157;
}
{
add.f16x2 r262, r256, r160;
}
{
add.f16x2 %0, r259, r169;
}
{
add.f16x2 %1, r262, r172;
}
mov.u32 r5508, 0;
cvt.rn.f16.s32 rs1, r5508;
mov.b32 r283, {rs1, rs1};
cvt.rn.f16.s32 rs2, r5508;
mov.b32 r295, {rs2, rs2};
mov.f64 fd847, 0d3FEF584F2CE43B84;
{
cvt.rn.f16.f64 rs3, fd847;
}
mov.b32 r275, {rs3, rs3};
{
mul.f16x2 r273, r1, r275;
}
{
add.f16x2 r276, %62, r273;
}
mov.f64 fd768, 0d3FC9C4266041CA8F;
{
cvt.rn.f16.f64 rs4, fd768;
}
mov.b32 r281, {rs4, rs4};
{
mul.f16x2 r279, r10, r281;
}
{
add.f16x2 r282, r283, r279;
}
{
cvt.rn.f16.f64 rs5, fd847;
}
mov.b32 r287, {rs5, rs5};
{
mul.f16x2 r285, r4, r287;
}
{
add.f16x2 r288, %63, r285;
}
{
cvt.rn.f16.f64 rs6, fd768;
}
mov.b32 r293, {rs6, rs6};
{
mul.f16x2 r291, r7, r293;
}
{
add.f16x2 r294, r295, r291;
}
mov.f64 fd855, 0d3FED681A366A00FA;
{
cvt.rn.f16.f64 rs7, fd855;
}
mov.b32 r299, {rs7, rs7};
{
mul.f16x2 r297, r13, r299;
}
{
add.f16x2 r300, r276, r297;
}
mov.f64 fd816, 0d3FD93D20572CA90B;
{
cvt.rn.f16.f64 rs8, fd816;
}
mov.b32 r305, {rs8, rs8};
{
mul.f16x2 r303, r22, r305;
}
{
add.f16x2 r306, r282, r303;
}
{
cvt.rn.f16.f64 rs9, fd855;
}
mov.b32 r311, {rs9, rs9};
{
mul.f16x2 r309, r16, r311;
}
{
add.f16x2 r312, r288, r309;
}
{
cvt.rn.f16.f64 rs10, fd816;
}
mov.b32 r317, {rs10, rs10};
{
mul.f16x2 r315, r19, r317;
}
{
add.f16x2 r318, r294, r315;
}
mov.f64 fd863, 0d3FEA43B1B1379AFF;
{
cvt.rn.f16.f64 rs11, fd863;
}
mov.b32 r323, {rs11, rs11};
{
mul.f16x2 r321, r25, r323;
}
{
add.f16x2 r324, r300, r321;
}
mov.f64 fd740, 0d3FE247D447A27216;
{
cvt.rn.f16.f64 rs12, fd740;
}
mov.b32 r329, {rs12, rs12};
{
mul.f16x2 r327, r34, r329;
}
{
add.f16x2 r330, r306, r327;
}
{
cvt.rn.f16.f64 rs13, fd863;
}
mov.b32 r335, {rs13, rs13};
{
mul.f16x2 r333, r28, r335;
}
{
add.f16x2 r336, r312, r333;
}
{
cvt.rn.f16.f64 rs14, fd740;
}
mov.b32 r341, {rs14, rs14};
{
mul.f16x2 r339, r31, r341;
}
{
add.f16x2 r342, r318, r339;
}
mov.f64 fd871, 0d3FE60C045A2E9729;
{
cvt.rn.f16.f64 rs15, fd871;
}
mov.b32 r347, {rs15, rs15};
{
mul.f16x2 r345, r37, r347;
}
{
add.f16x2 r348, r324, r345;
}
mov.f64 fd624, 0d3FE73180A4B0D300;
{
cvt.rn.f16.f64 rs16, fd624;
}
mov.b32 r353, {rs16, rs16};
{
mul.f16x2 r351, r46, r353;
}
{
add.f16x2 r354, r330, r351;
}
{
cvt.rn.f16.f64 rs17, fd871;
}
mov.b32 r359, {rs17, rs17};
{
mul.f16x2 r357, r40, r359;
}
{
add.f16x2 r360, r336, r357;
}
{
cvt.rn.f16.f64 rs18, fd624;
}
mov.b32 r365, {rs18, rs18};
{
mul.f16x2 r363, r43, r365;
}
{
add.f16x2 r366, r342, r363;
}
mov.f64 fd879, 0d3FE0ED45EEA3B09F;
{
cvt.rn.f16.f64 rs19, fd879;
}
mov.b32 r371, {rs19, rs19};
{
mul.f16x2 r369, r49, r371;
}
{
add.f16x2 r372, r348, r369;
}
mov.f64 fd808, 0d3FEB2818007C19DF;
{
cvt.rn.f16.f64 rs20, fd808;
}
mov.b32 r377, {rs20, rs20};
{
mul.f16x2 r375, r58, r377;
}
{
add.f16x2 r378, r354, r375;
}
{
cvt.rn.f16.f64 rs21, fd879;
}
mov.b32 r383, {rs21, rs21};
{
mul.f16x2 r381, r52, r383;
}
{
add.f16x2 r384, r360, r381;
}
{
cvt.rn.f16.f64 rs22, fd808;
}
mov.b32 r389, {rs22, rs22};
{
mul.f16x2 r387, r55, r389;
}
{
add.f16x2 r390, r366, r387;
}
mov.f64 fd887, 0d3FD63A3FCFACA412;
{
cvt.rn.f16.f64 rs23, fd887;
}
mov.b32 r395, {rs23, rs23};
{
mul.f16x2 r393, r61, r395;
}
{
add.f16x2 r396, r372, r393;
}
mov.f64 fd760, 0d3FEE0210C26A6E6F;
{
cvt.rn.f16.f64 rs24, fd760;
}
mov.b32 r401, {rs24, rs24};
{
mul.f16x2 r399, r70, r401;
}
{
add.f16x2 r402, r378, r399;
}
{
cvt.rn.f16.f64 rs25, fd887;
}
mov.b32 r407, {rs25, rs25};
{
mul.f16x2 r405, r64, r407;
}
{
add.f16x2 r408, r384, r405;
}
{
cvt.rn.f16.f64 rs26, fd760;
}
mov.b32 r413, {rs26, rs26};
{
mul.f16x2 r411, r67, r413;
}
{
add.f16x2 r414, r390, r411;
}
mov.f64 fd895, 0d3FC361FC440B478F;
{
cvt.rn.f16.f64 rs27, fd895;
}
mov.b32 r419, {rs27, rs27};
{
mul.f16x2 r417, r73, r419;
}
{
add.f16x2 r420, r396, r417;
}
mov.f64 fd580, 0d3FEFA18852C3E08A;
{
cvt.rn.f16.f64 rs28, fd580;
}
mov.b32 r425, {rs28, rs28};
{
mul.f16x2 r423, r82, r425;
}
{
add.f16x2 r426, r402, r423;
}
{
cvt.rn.f16.f64 rs29, fd895;
}
mov.b32 r431, {rs29, rs29};
{
mul.f16x2 r429, r76, r431;
}
{
add.f16x2 r432, r408, r429;
}
{
cvt.rn.f16.f64 rs30, fd580;
}
mov.b32 r437, {rs30, rs30};
{
mul.f16x2 r435, r79, r437;
}
{
add.f16x2 r438, r414, r435;
}
mov.f64 fd899, 0dBFA9EEB01776B57D;
{
cvt.rn.f16.f64 rs31, fd899;
}
mov.b32 r443, {rs31, rs31};
{
mul.f16x2 r441, r85, r443;
}
{
add.f16x2 r444, r420, r441;
}
mov.f64 fd900, 0d3FEFF57C5208CCF9;
{
cvt.rn.f16.f64 rs32, fd900;
}
mov.b32 r449, {rs32, rs32};
{
mul.f16x2 r447, r94, r449;
}
{
add.f16x2 r450, r426, r447;
}
{
cvt.rn.f16.f64 rs33, fd899;
}
mov.b32 r455, {rs33, rs33};
{
mul.f16x2 r453, r88, r455;
}
{
add.f16x2 r456, r432, r453;
}
{
cvt.rn.f16.f64 rs34, fd900;
}
mov.b32 r461, {rs34, rs34};
{
mul.f16x2 r459, r91, r461;
}
{
add.f16x2 r462, r438, r459;
}
mov.f64 fd891, 0dBFD00AB0EB2D7D94;
{
cvt.rn.f16.f64 rs35, fd891;
}
mov.b32 r467, {rs35, rs35};
{
mul.f16x2 r465, r97, r467;
}
{
add.f16x2 r468, r444, r465;
}
mov.f64 fd892, 0d3FEEFA7CDDB128FA;
{
cvt.rn.f16.f64 rs36, fd892;
}
mov.b32 r473, {rs36, rs36};
{
mul.f16x2 r471, r106, r473;
}
{
add.f16x2 r474, r450, r471;
}
{
cvt.rn.f16.f64 rs37, fd891;
}
mov.b32 r479, {rs37, rs37};
{
mul.f16x2 r477, r100, r479;
}
{
add.f16x2 r480, r456, r477;
}
{
cvt.rn.f16.f64 rs38, fd892;
}
mov.b32 r485, {rs38, rs38};
{
mul.f16x2 r483, r103, r485;
}
{
add.f16x2 r486, r462, r483;
}
mov.f64 fd883, 0dBFDC2F6AF3928A8E;
{
cvt.rn.f16.f64 rs39, fd883;
}
mov.b32 r491, {rs39, rs39};
{
mul.f16x2 r489, r109, r491;
}
{
add.f16x2 r492, r468, r489;
}
mov.f64 fd884, 0d3FECBAD095F50378;
{
cvt.rn.f16.f64 rs40, fd884;
}
mov.b32 r497, {rs40, rs40};
{
mul.f16x2 r495, r118, r497;
}
{
add.f16x2 r498, r474, r495;
}
{
cvt.rn.f16.f64 rs41, fd883;
}
mov.b32 r503, {rs41, rs41};
{
mul.f16x2 r501, r112, r503;
}
{
add.f16x2 r504, r480, r501;
}
{
cvt.rn.f16.f64 rs42, fd884;
}
mov.b32 r509, {rs42, rs42};
{
mul.f16x2 r507, r115, r509;
}
{
add.f16x2 r510, r486, r507;
}
mov.f64 fd875, 0dBFE3965F49174D13;
{
cvt.rn.f16.f64 rs43, fd875;
}
mov.b32 r515, {rs43, rs43};
{
mul.f16x2 r513, r121, r515;
}
{
add.f16x2 r516, r492, r513;
}
mov.f64 fd876, 0d3FE94E08EB13C451;
{
cvt.rn.f16.f64 rs44, fd876;
}
mov.b32 r521, {rs44, rs44};
{
mul.f16x2 r519, r130, r521;
}
{
add.f16x2 r522, r498, r519;
}
{
cvt.rn.f16.f64 rs45, fd875;
}
mov.b32 r527, {rs45, rs45};
{
mul.f16x2 r525, r124, r527;
}
{
add.f16x2 r528, r504, r525;
}
{
cvt.rn.f16.f64 rs46, fd876;
}
mov.b32 r533, {rs46, rs46};
{
mul.f16x2 r531, r127, r533;
}
{
add.f16x2 r534, r510, r531;
}
mov.f64 fd867, 0dBFE847BF1D5146CC;
{
cvt.rn.f16.f64 rs47, fd867;
}
mov.b32 r539, {rs47, rs47};
{
mul.f16x2 r537, r133, r539;
}
{
add.f16x2 r540, r516, r537;
}
mov.f64 fd868, 0d3FE4D80B1AD9CCF6;
{
cvt.rn.f16.f64 rs48, fd868;
}
mov.b32 r545, {rs48, rs48};
{
mul.f16x2 r543, r142, r545;
}
{
add.f16x2 r546, r522, r543;
}
{
cvt.rn.f16.f64 rs49, fd867;
}
mov.b32 r551, {rs49, rs49};
{
mul.f16x2 r549, r136, r551;
}
{
add.f16x2 r552, r528, r549;
}
{
cvt.rn.f16.f64 rs50, fd868;
}
mov.b32 r557, {rs50, rs50};
{
mul.f16x2 r555, r139, r557;
}
{
add.f16x2 r558, r534, r555;
}
mov.f64 fd859, 0dBFEBFAA5C136B224;
{
cvt.rn.f16.f64 rs51, fd859;
}
mov.b32 r563, {rs51, rs51};
{
mul.f16x2 r561, r145, r563;
}
{
add.f16x2 r564, r540, r561;
}
mov.f64 fd860, 0d3FDF0F2FF6705BEC;
{
cvt.rn.f16.f64 rs52, fd860;
}
mov.b32 r569, {rs52, rs52};
{
mul.f16x2 r567, r154, r569;
}
{
add.f16x2 r570, r546, r567;
}
{
cvt.rn.f16.f64 rs53, fd859;
}
mov.b32 r575, {rs53, rs53};
{
mul.f16x2 r573, r148, r575;
}
{
add.f16x2 r576, r552, r573;
}
{
cvt.rn.f16.f64 rs54, fd860;
}
mov.b32 r581, {rs54, rs54};
{
mul.f16x2 r579, r151, r581;
}
{
add.f16x2 r582, r558, r579;
}
mov.f64 fd851, 0dBFEE884F0CC22CCC;
{
cvt.rn.f16.f64 rs55, fd851;
}
mov.b32 r587, {rs55, rs55};
{
mul.f16x2 r585, r157, r587;
}
{
add.f16x2 r588, r564, r585;
}
mov.f64 fd852, 0d3FD328C3F1B322CB;
{
cvt.rn.f16.f64 rs56, fd852;
}
mov.b32 r593, {rs56, rs56};
{
mul.f16x2 r591, r166, r593;
}
{
add.f16x2 r594, r570, r591;
}
{
cvt.rn.f16.f64 rs57, fd851;
}
mov.b32 r599, {rs57, rs57};
{
mul.f16x2 r597, r160, r599;
}
{
add.f16x2 r600, r576, r597;
}
{
cvt.rn.f16.f64 rs58, fd852;
}
mov.b32 r605, {rs58, rs58};
{
mul.f16x2 r603, r163, r605;
}
{
add.f16x2 r606, r582, r603;
}
mov.f64 fd843, 0dBFEFD5F830F860F9;
{
cvt.rn.f16.f64 rs59, fd843;
}
mov.b32 r611, {rs59, rs59};
{
mul.f16x2 r609, r169, r611;
}
{
add.f16x2 r612, r588, r609;
}
mov.f64 fd844, 0d3FB9E62ACA53C49F;
{
cvt.rn.f16.f64 rs60, fd844;
}
mov.b32 r617, {rs60, rs60};
{
mul.f16x2 r615, r178, r617;
}
{
add.f16x2 r618, r594, r615;
}
{
cvt.rn.f16.f64 rs61, fd843;
}
mov.b32 r623, {rs61, rs61};
{
mul.f16x2 r621, r172, r623;
}
{
add.f16x2 r624, r600, r621;
}
{
cvt.rn.f16.f64 rs62, fd844;
}
mov.b32 r629, {rs62, rs62};
{
mul.f16x2 r627, r175, r629;
}
{
add.f16x2 r630, r606, r627;
}
{
sub.f16x2 %2, r612, r618;
}
{
add.f16x2 %3, r624, r630;
}
{
add.f16x2 %60, r612, r618;
}
{
sub.f16x2 %61, r624, r630;
}
cvt.rn.f16.s32 rs63, r5508;
mov.b32 r657, {rs63, rs63};
cvt.rn.f16.s32 rs64, r5508;
mov.b32 r669, {rs64, rs64};
{
cvt.rn.f16.f64 rs65, fd855;
}
mov.b32 r649, {rs65, rs65};
{
mul.f16x2 r647, r1, r649;
}
{
add.f16x2 r650, %62, r647;
}
{
cvt.rn.f16.f64 rs66, fd816;
}
mov.b32 r655, {rs66, rs66};
{
mul.f16x2 r653, r10, r655;
}
{
add.f16x2 r656, r657, r653;
}
{
cvt.rn.f16.f64 rs67, fd855;
}
mov.b32 r661, {rs67, rs67};
{
mul.f16x2 r659, r4, r661;
}
{
add.f16x2 r662, %63, r659;
}
{
cvt.rn.f16.f64 rs68, fd816;
}
mov.b32 r667, {rs68, rs68};
{
mul.f16x2 r665, r7, r667;
}
{
add.f16x2 r668, r669, r665;
}
{
cvt.rn.f16.f64 rs69, fd871;
}
mov.b32 r673, {rs69, rs69};
{
mul.f16x2 r671, r13, r673;
}
{
add.f16x2 r674, r650, r671;
}
{
cvt.rn.f16.f64 rs70, fd624;
}
mov.b32 r679, {rs70, rs70};
{
mul.f16x2 r677, r22, r679;
}
{
add.f16x2 r680, r656, r677;
}
{
cvt.rn.f16.f64 rs71, fd871;
}
mov.b32 r685, {rs71, rs71};
{
mul.f16x2 r683, r16, r685;
}
{
add.f16x2 r686, r662, r683;
}
{
cvt.rn.f16.f64 rs72, fd624;
}
mov.b32 r691, {rs72, rs72};
{
mul.f16x2 r689, r19, r691;
}
{
add.f16x2 r692, r668, r689;
}
{
cvt.rn.f16.f64 rs73, fd887;
}
mov.b32 r697, {rs73, rs73};
{
mul.f16x2 r695, r25, r697;
}
{
add.f16x2 r698, r674, r695;
}
{
cvt.rn.f16.f64 rs74, fd760;
}
mov.b32 r703, {rs74, rs74};
{
mul.f16x2 r701, r34, r703;
}
{
add.f16x2 r704, r680, r701;
}
{
cvt.rn.f16.f64 rs75, fd887;
}
mov.b32 r709, {rs75, rs75};
{
mul.f16x2 r707, r28, r709;
}
{
add.f16x2 r710, r686, r707;
}
{
cvt.rn.f16.f64 rs76, fd760;
}
mov.b32 r715, {rs76, rs76};
{
mul.f16x2 r713, r31, r715;
}
{
add.f16x2 r716, r692, r713;
}
{
cvt.rn.f16.f64 rs77, fd899;
}
mov.b32 r721, {rs77, rs77};
{
mul.f16x2 r719, r37, r721;
}
{
add.f16x2 r722, r698, r719;
}
{
cvt.rn.f16.f64 rs78, fd900;
}
mov.b32 r727, {rs78, rs78};
{
mul.f16x2 r725, r46, r727;
}
{
add.f16x2 r728, r704, r725;
}
{
cvt.rn.f16.f64 rs79, fd899;
}
mov.b32 r733, {rs79, rs79};
{
mul.f16x2 r731, r40, r733;
}
{
add.f16x2 r734, r710, r731;
}
{
cvt.rn.f16.f64 rs80, fd900;
}
mov.b32 r739, {rs80, rs80};
{
mul.f16x2 r737, r43, r739;
}
{
add.f16x2 r740, r716, r737;
}
{
cvt.rn.f16.f64 rs81, fd883;
}
mov.b32 r745, {rs81, rs81};
{
mul.f16x2 r743, r49, r745;
}
{
add.f16x2 r746, r722, r743;
}
{
cvt.rn.f16.f64 rs82, fd884;
}
mov.b32 r751, {rs82, rs82};
{
mul.f16x2 r749, r58, r751;
}
{
add.f16x2 r752, r728, r749;
}
{
cvt.rn.f16.f64 rs83, fd883;
}
mov.b32 r757, {rs83, rs83};
{
mul.f16x2 r755, r52, r757;
}
{
add.f16x2 r758, r734, r755;
}
{
cvt.rn.f16.f64 rs84, fd884;
}
mov.b32 r763, {rs84, rs84};
{
mul.f16x2 r761, r55, r763;
}
{
add.f16x2 r764, r740, r761;
}
{
cvt.rn.f16.f64 rs85, fd867;
}
mov.b32 r769, {rs85, rs85};
{
mul.f16x2 r767, r61, r769;
}
{
add.f16x2 r770, r746, r767;
}
{
cvt.rn.f16.f64 rs86, fd868;
}
mov.b32 r775, {rs86, rs86};
{
mul.f16x2 r773, r70, r775;
}
{
add.f16x2 r776, r752, r773;
}
{
cvt.rn.f16.f64 rs87, fd867;
}
mov.b32 r781, {rs87, rs87};
{
mul.f16x2 r779, r64, r781;
}
{
add.f16x2 r782, r758, r779;
}
{
cvt.rn.f16.f64 rs88, fd868;
}
mov.b32 r787, {rs88, rs88};
{
mul.f16x2 r785, r67, r787;
}
{
add.f16x2 r788, r764, r785;
}
{
cvt.rn.f16.f64 rs89, fd851;
}
mov.b32 r793, {rs89, rs89};
{
mul.f16x2 r791, r73, r793;
}
{
add.f16x2 r794, r770, r791;
}
{
cvt.rn.f16.f64 rs90, fd852;
}
mov.b32 r799, {rs90, rs90};
{
mul.f16x2 r797, r82, r799;
}
{
add.f16x2 r800, r776, r797;
}
{
cvt.rn.f16.f64 rs91, fd851;
}
mov.b32 r805, {rs91, rs91};
{
mul.f16x2 r803, r76, r805;
}
{
add.f16x2 r806, r782, r803;
}
{
cvt.rn.f16.f64 rs92, fd852;
}
mov.b32 r811, {rs92, rs92};
{
mul.f16x2 r809, r79, r811;
}
{
add.f16x2 r812, r788, r809;
}
{
cvt.rn.f16.f64 rs93, fd843;
}
mov.b32 r817, {rs93, rs93};
{
mul.f16x2 r815, r85, r817;
}
{
add.f16x2 r818, r794, r815;
}
mov.f64 fd820, 0dBFB9E62ACA53C49F;
{
cvt.rn.f16.f64 rs94, fd820;
}
mov.b32 r823, {rs94, rs94};
{
mul.f16x2 r821, r94, r823;
}
{
add.f16x2 r824, r800, r821;
}
{
cvt.rn.f16.f64 rs95, fd843;
}
mov.b32 r829, {rs95, rs95};
{
mul.f16x2 r827, r88, r829;
}
{
add.f16x2 r830, r806, r827;
}
{
cvt.rn.f16.f64 rs96, fd820;
}
mov.b32 r835, {rs96, rs96};
{
mul.f16x2 r833, r91, r835;
}
{
add.f16x2 r836, r812, r833;
}
{
cvt.rn.f16.f64 rs97, fd859;
}
mov.b32 r841, {rs97, rs97};
{
mul.f16x2 r839, r97, r841;
}
{
add.f16x2 r842, r818, r839;
}
mov.f64 fd572, 0dBFDF0F2FF6705BEC;
{
cvt.rn.f16.f64 rs98, fd572;
}
mov.b32 r847, {rs98, rs98};
{
mul.f16x2 r845, r106, r847;
}
{
add.f16x2 r848, r824, r845;
}
{
cvt.rn.f16.f64 rs99, fd859;
}
mov.b32 r853, {rs99, rs99};
{
mul.f16x2 r851, r100, r853;
}
{
add.f16x2 r854, r830, r851;
}
{
cvt.rn.f16.f64 rs100, fd572;
}
mov.b32 r859, {rs100, rs100};
{
mul.f16x2 r857, r103, r859;
}
{
add.f16x2 r860, r836, r857;
}
{
cvt.rn.f16.f64 rs101, fd875;
}
mov.b32 r865, {rs101, rs101};
{
mul.f16x2 r863, r109, r865;
}
{
add.f16x2 r866, r842, r863;
}
mov.f64 fd708, 0dBFE94E08EB13C451;
{
cvt.rn.f16.f64 rs102, fd708;
}
mov.b32 r871, {rs102, rs102};
{
mul.f16x2 r869, r118, r871;
}
{
add.f16x2 r872, r848, r869;
}
{
cvt.rn.f16.f64 rs103, fd875;
}
mov.b32 r877, {rs103, rs103};
{
mul.f16x2 r875, r112, r877;
}
{
add.f16x2 r878, r854, r875;
}
{
cvt.rn.f16.f64 rs104, fd708;
}
mov.b32 r883, {rs104, rs104};
{
mul.f16x2 r881, r115, r883;
}
{
add.f16x2 r884, r860, r881;
}
{
cvt.rn.f16.f64 rs105, fd891;
}
mov.b32 r889, {rs105, rs105};
{
mul.f16x2 r887, r121, r889;
}
{
add.f16x2 r890, r866, r887;
}
mov.f64 fd804, 0dBFEEFA7CDDB128FA;
{
cvt.rn.f16.f64 rs106, fd804;
}
mov.b32 r895, {rs106, rs106};
{
mul.f16x2 r893, r130, r895;
}
{
add.f16x2 r896, r872, r893;
}
{
cvt.rn.f16.f64 rs107, fd891;
}
mov.b32 r901, {rs107, rs107};
{
mul.f16x2 r899, r124, r901;
}
{
add.f16x2 r902, r878, r899;
}
{
cvt.rn.f16.f64 rs108, fd804;
}
mov.b32 r907, {rs108, rs108};
{
mul.f16x2 r905, r127, r907;
}
{
add.f16x2 r908, r884, r905;
}
{
cvt.rn.f16.f64 rs109, fd895;
}
mov.b32 r913, {rs109, rs109};
{
mul.f16x2 r911, r133, r913;
}
{
add.f16x2 r914, r890, r911;
}
mov.f64 fd896, 0dBFEFA18852C3E08A;
{
cvt.rn.f16.f64 rs110, fd896;
}
mov.b32 r919, {rs110, rs110};
{
mul.f16x2 r917, r142, r919;
}
{
add.f16x2 r920, r896, r917;
}
{
cvt.rn.f16.f64 rs111, fd895;
}
mov.b32 r925, {rs111, rs111};
{
mul.f16x2 r923, r136, r925;
}
{
add.f16x2 r926, r902, r923;
}
{
cvt.rn.f16.f64 rs112, fd896;
}
mov.b32 r931, {rs112, rs112};
{
mul.f16x2 r929, r139, r931;
}
{
add.f16x2 r932, r908, r929;
}
{
cvt.rn.f16.f64 rs113, fd879;
}
mov.b32 r937, {rs113, rs113};
{
mul.f16x2 r935, r145, r937;
}
{
add.f16x2 r938, r914, r935;
}
mov.f64 fd880, 0dBFEB2818007C19DF;
{
cvt.rn.f16.f64 rs114, fd880;
}
mov.b32 r943, {rs114, rs114};
{
mul.f16x2 r941, r154, r943;
}
{
add.f16x2 r944, r920, r941;
}
{
cvt.rn.f16.f64 rs115, fd879;
}
mov.b32 r949, {rs115, rs115};
{
mul.f16x2 r947, r148, r949;
}
{
add.f16x2 r950, r926, r947;
}
{
cvt.rn.f16.f64 rs116, fd880;
}
mov.b32 r955, {rs116, rs116};
{
mul.f16x2 r953, r151, r955;
}
{
add.f16x2 r956, r932, r953;
}
{
cvt.rn.f16.f64 rs117, fd863;
}
mov.b32 r961, {rs117, rs117};
{
mul.f16x2 r959, r157, r961;
}
{
add.f16x2 r962, r938, r959;
}
mov.f64 fd864, 0dBFE247D447A27216;
{
cvt.rn.f16.f64 rs118, fd864;
}
mov.b32 r967, {rs118, rs118};
{
mul.f16x2 r965, r166, r967;
}
{
add.f16x2 r968, r944, r965;
}
{
cvt.rn.f16.f64 rs119, fd863;
}
mov.b32 r973, {rs119, rs119};
{
mul.f16x2 r971, r160, r973;
}
{
add.f16x2 r974, r950, r971;
}
{
cvt.rn.f16.f64 rs120, fd864;
}
mov.b32 r979, {rs120, rs120};
{
mul.f16x2 r977, r163, r979;
}
{
add.f16x2 r980, r956, r977;
}
{
cvt.rn.f16.f64 rs121, fd847;
}
mov.b32 r985, {rs121, rs121};
{
mul.f16x2 r983, r169, r985;
}
{
add.f16x2 r986, r962, r983;
}
mov.f64 fd848, 0dBFC9C4266041CA8F;
{
cvt.rn.f16.f64 rs122, fd848;
}
mov.b32 r991, {rs122, rs122};
{
mul.f16x2 r989, r178, r991;
}
{
add.f16x2 r992, r968, r989;
}
{
cvt.rn.f16.f64 rs123, fd847;
}
mov.b32 r997, {rs123, rs123};
{
mul.f16x2 r995, r172, r997;
}
{
add.f16x2 r998, r974, r995;
}
{
cvt.rn.f16.f64 rs124, fd848;
}
mov.b32 r1003, {rs124, rs124};
{
mul.f16x2 r1001, r175, r1003;
}
{
add.f16x2 r1004, r980, r1001;
}
{
sub.f16x2 %4, r986, r992;
}
{
add.f16x2 %5, r998, r1004;
}
{
add.f16x2 %58, r986, r992;
}
{
sub.f16x2 %59, r998, r1004;
}
cvt.rn.f16.s32 rs125, r5508;
mov.b32 r1031, {rs125, rs125};
cvt.rn.f16.s32 rs126, r5508;
mov.b32 r1043, {rs126, rs126};
{
cvt.rn.f16.f64 rs127, fd863;
}
mov.b32 r1023, {rs127, rs127};
{
mul.f16x2 r1021, r1, r1023;
}
{
add.f16x2 r1024, %62, r1021;
}
{
cvt.rn.f16.f64 rs128, fd740;
}
mov.b32 r1029, {rs128, rs128};
{
mul.f16x2 r1027, r10, r1029;
}
{
add.f16x2 r1030, r1031, r1027;
}
{
cvt.rn.f16.f64 rs129, fd863;
}
mov.b32 r1035, {rs129, rs129};
{
mul.f16x2 r1033, r4, r1035;
}
{
add.f16x2 r1036, %63, r1033;
}
{
cvt.rn.f16.f64 rs130, fd740;
}
mov.b32 r1041, {rs130, rs130};
{
mul.f16x2 r1039, r7, r1041;
}
{
add.f16x2 r1042, r1043, r1039;
}
{
cvt.rn.f16.f64 rs131, fd887;
}
mov.b32 r1047, {rs131, rs131};
{
mul.f16x2 r1045, r13, r1047;
}
{
add.f16x2 r1048, r1024, r1045;
}
{
cvt.rn.f16.f64 rs132, fd760;
}
mov.b32 r1053, {rs132, rs132};
{
mul.f16x2 r1051, r22, r1053;
}
{
add.f16x2 r1054, r1030, r1051;
}
{
cvt.rn.f16.f64 rs133, fd887;
}
mov.b32 r1059, {rs133, rs133};
{
mul.f16x2 r1057, r16, r1059;
}
{
add.f16x2 r1060, r1036, r1057;
}
{
cvt.rn.f16.f64 rs134, fd760;
}
mov.b32 r1065, {rs134, rs134};
{
mul.f16x2 r1063, r19, r1065;
}
{
add.f16x2 r1066, r1042, r1063;
}
{
cvt.rn.f16.f64 rs135, fd891;
}
mov.b32 r1071, {rs135, rs135};
{
mul.f16x2 r1069, r25, r1071;
}
{
add.f16x2 r1072, r1048, r1069;
}
{
cvt.rn.f16.f64 rs136, fd892;
}
mov.b32 r1077, {rs136, rs136};
{
mul.f16x2 r1075, r34, r1077;
}
{
add.f16x2 r1078, r1054, r1075;
}
{
cvt.rn.f16.f64 rs137, fd891;
}
mov.b32 r1083, {rs137, rs137};
{
mul.f16x2 r1081, r28, r1083;
}
{
add.f16x2 r1084, r1060, r1081;
}
{
cvt.rn.f16.f64 rs138, fd892;
}
mov.b32 r1089, {rs138, rs138};
{
mul.f16x2 r1087, r31, r1089;
}
{
add.f16x2 r1090, r1066, r1087;
}
{
cvt.rn.f16.f64 rs139, fd867;
}
mov.b32 r1095, {rs139, rs139};
{
mul.f16x2 r1093, r37, r1095;
}
{
add.f16x2 r1096, r1072, r1093;
}
{
cvt.rn.f16.f64 rs140, fd868;
}
mov.b32 r1101, {rs140, rs140};
{
mul.f16x2 r1099, r46, r1101;
}
{
add.f16x2 r1102, r1078, r1099;
}
{
cvt.rn.f16.f64 rs141, fd867;
}
mov.b32 r1107, {rs141, rs141};
{
mul.f16x2 r1105, r40, r1107;
}
{
add.f16x2 r1108, r1084, r1105;
}
{
cvt.rn.f16.f64 rs142, fd868;
}
mov.b32 r1113, {rs142, rs142};
{
mul.f16x2 r1111, r43, r1113;
}
{
add.f16x2 r1114, r1090, r1111;
}
{
cvt.rn.f16.f64 rs143, fd843;
}
mov.b32 r1119, {rs143, rs143};
{
mul.f16x2 r1117, r49, r1119;
}
{
add.f16x2 r1120, r1096, r1117;
}
{
cvt.rn.f16.f64 rs144, fd844;
}
mov.b32 r1125, {rs144, rs144};
{
mul.f16x2 r1123, r58, r1125;
}
{
add.f16x2 r1126, r1102, r1123;
}
{
cvt.rn.f16.f64 rs145, fd843;
}
mov.b32 r1131, {rs145, rs145};
{
mul.f16x2 r1129, r52, r1131;
}
{
add.f16x2 r1132, r1108, r1129;
}
{
cvt.rn.f16.f64 rs146, fd844;
}
mov.b32 r1137, {rs146, rs146};
{
mul.f16x2 r1135, r55, r1137;
}
{
add.f16x2 r1138, r1114, r1135;
}
{
cvt.rn.f16.f64 rs147, fd859;
}
mov.b32 r1143, {rs147, rs147};
{
mul.f16x2 r1141, r61, r1143;
}
{
add.f16x2 r1144, r1120, r1141;
}
{
cvt.rn.f16.f64 rs148, fd572;
}
mov.b32 r1149, {rs148, rs148};
{
mul.f16x2 r1147, r70, r1149;
}
{
add.f16x2 r1150, r1126, r1147;
}
{
cvt.rn.f16.f64 rs149, fd859;
}
mov.b32 r1155, {rs149, rs149};
{
mul.f16x2 r1153, r64, r1155;
}
{
add.f16x2 r1156, r1132, r1153;
}
{
cvt.rn.f16.f64 rs150, fd572;
}
mov.b32 r1161, {rs150, rs150};
{
mul.f16x2 r1159, r67, r1161;
}
{
add.f16x2 r1162, r1138, r1159;
}
{
cvt.rn.f16.f64 rs151, fd883;
}
mov.b32 r1167, {rs151, rs151};
{
mul.f16x2 r1165, r73, r1167;
}
{
add.f16x2 r1168, r1144, r1165;
}
mov.f64 fd736, 0dBFECBAD095F50378;
{
cvt.rn.f16.f64 rs152, fd736;
}
mov.b32 r1173, {rs152, rs152};
{
mul.f16x2 r1171, r82, r1173;
}
{
add.f16x2 r1174, r1150, r1171;
}
{
cvt.rn.f16.f64 rs153, fd883;
}
mov.b32 r1179, {rs153, rs153};
{
mul.f16x2 r1177, r76, r1179;
}
{
add.f16x2 r1180, r1156, r1177;
}
{
cvt.rn.f16.f64 rs154, fd736;
}
mov.b32 r1185, {rs154, rs154};
{
mul.f16x2 r1183, r79, r1185;
}
{
add.f16x2 r1186, r1162, r1183;
}
{
cvt.rn.f16.f64 rs155, fd895;
}
mov.b32 r1191, {rs155, rs155};
{
mul.f16x2 r1189, r85, r1191;
}
{
add.f16x2 r1192, r1168, r1189;
}
{
cvt.rn.f16.f64 rs156, fd896;
}
mov.b32 r1197, {rs156, rs156};
{
mul.f16x2 r1195, r94, r1197;
}
{
add.f16x2 r1198, r1174, r1195;
}
{
cvt.rn.f16.f64 rs157, fd895;
}
mov.b32 r1203, {rs157, rs157};
{
mul.f16x2 r1201, r88, r1203;
}
{
add.f16x2 r1204, r1180, r1201;
}
{
cvt.rn.f16.f64 rs158, fd896;
}
mov.b32 r1209, {rs158, rs158};
{
mul.f16x2 r1207, r91, r1209;
}
{
add.f16x2 r1210, r1186, r1207;
}
{
cvt.rn.f16.f64 rs159, fd871;
}
mov.b32 r1215, {rs159, rs159};
{
mul.f16x2 r1213, r97, r1215;
}
{
add.f16x2 r1216, r1192, r1213;
}
mov.f64 fd872, 0dBFE73180A4B0D300;
{
cvt.rn.f16.f64 rs160, fd872;
}
mov.b32 r1221, {rs160, rs160};
{
mul.f16x2 r1219, r106, r1221;
}
{
add.f16x2 r1222, r1198, r1219;
}
{
cvt.rn.f16.f64 rs161, fd871;
}
mov.b32 r1227, {rs161, rs161};
{
mul.f16x2 r1225, r100, r1227;
}
{
add.f16x2 r1228, r1204, r1225;
}
{
cvt.rn.f16.f64 rs162, fd872;
}
mov.b32 r1233, {rs162, rs162};
{
mul.f16x2 r1231, r103, r1233;
}
{
add.f16x2 r1234, r1210, r1231;
}
{
cvt.rn.f16.f64 rs163, fd847;
}
mov.b32 r1239, {rs163, rs163};
{
mul.f16x2 r1237, r109, r1239;
}
{
add.f16x2 r1240, r1216, r1237;
}
{
cvt.rn.f16.f64 rs164, fd848;
}
mov.b32 r1245, {rs164, rs164};
{
mul.f16x2 r1243, r118, r1245;
}
{
add.f16x2 r1246, r1222, r1243;
}
{
cvt.rn.f16.f64 rs165, fd847;
}
mov.b32 r1251, {rs165, rs165};
{
mul.f16x2 r1249, r112, r1251;
}
{
add.f16x2 r1252, r1228, r1249;
}
{
cvt.rn.f16.f64 rs166, fd848;
}
mov.b32 r1257, {rs166, rs166};
{
mul.f16x2 r1255, r115, r1257;
}
{
add.f16x2 r1258, r1234, r1255;
}
{
cvt.rn.f16.f64 rs167, fd855;
}
mov.b32 r1263, {rs167, rs167};
{
mul.f16x2 r1261, r121, r1263;
}
{
add.f16x2 r1264, r1240, r1261;
}
{
cvt.rn.f16.f64 rs168, fd816;
}
mov.b32 r1269, {rs168, rs168};
{
mul.f16x2 r1267, r130, r1269;
}
{
add.f16x2 r1270, r1246, r1267;
}
{
cvt.rn.f16.f64 rs169, fd855;
}
mov.b32 r1275, {rs169, rs169};
{
mul.f16x2 r1273, r124, r1275;
}
{
add.f16x2 r1276, r1252, r1273;
}
{
cvt.rn.f16.f64 rs170, fd816;
}
mov.b32 r1281, {rs170, rs170};
{
mul.f16x2 r1279, r127, r1281;
}
{
add.f16x2 r1282, r1258, r1279;
}
{
cvt.rn.f16.f64 rs171, fd879;
}
mov.b32 r1287, {rs171, rs171};
{
mul.f16x2 r1285, r133, r1287;
}
{
add.f16x2 r1288, r1264, r1285;
}
{
cvt.rn.f16.f64 rs172, fd808;
}
mov.b32 r1293, {rs172, rs172};
{
mul.f16x2 r1291, r142, r1293;
}
{
add.f16x2 r1294, r1270, r1291;
}
{
cvt.rn.f16.f64 rs173, fd879;
}
mov.b32 r1299, {rs173, rs173};
{
mul.f16x2 r1297, r136, r1299;
}
{
add.f16x2 r1300, r1276, r1297;
}
{
cvt.rn.f16.f64 rs174, fd808;
}
mov.b32 r1305, {rs174, rs174};
{
mul.f16x2 r1303, r139, r1305;
}
{
add.f16x2 r1306, r1282, r1303;
}
{
cvt.rn.f16.f64 rs175, fd899;
}
mov.b32 r1311, {rs175, rs175};
{
mul.f16x2 r1309, r145, r1311;
}
{
add.f16x2 r1312, r1288, r1309;
}
{
cvt.rn.f16.f64 rs176, fd900;
}
mov.b32 r1317, {rs176, rs176};
{
mul.f16x2 r1315, r154, r1317;
}
{
add.f16x2 r1318, r1294, r1315;
}
{
cvt.rn.f16.f64 rs177, fd899;
}
mov.b32 r1323, {rs177, rs177};
{
mul.f16x2 r1321, r148, r1323;
}
{
add.f16x2 r1324, r1300, r1321;
}
{
cvt.rn.f16.f64 rs178, fd900;
}
mov.b32 r1329, {rs178, rs178};
{
mul.f16x2 r1327, r151, r1329;
}
{
add.f16x2 r1330, r1306, r1327;
}
{
cvt.rn.f16.f64 rs179, fd875;
}
mov.b32 r1335, {rs179, rs179};
{
mul.f16x2 r1333, r157, r1335;
}
{
add.f16x2 r1336, r1312, r1333;
}
{
cvt.rn.f16.f64 rs180, fd876;
}
mov.b32 r1341, {rs180, rs180};
{
mul.f16x2 r1339, r166, r1341;
}
{
add.f16x2 r1342, r1318, r1339;
}
{
cvt.rn.f16.f64 rs181, fd875;
}
mov.b32 r1347, {rs181, rs181};
{
mul.f16x2 r1345, r160, r1347;
}
{
add.f16x2 r1348, r1324, r1345;
}
{
cvt.rn.f16.f64 rs182, fd876;
}
mov.b32 r1353, {rs182, rs182};
{
mul.f16x2 r1351, r163, r1353;
}
{
add.f16x2 r1354, r1330, r1351;
}
{
cvt.rn.f16.f64 rs183, fd851;
}
mov.b32 r1359, {rs183, rs183};
{
mul.f16x2 r1357, r169, r1359;
}
{
add.f16x2 r1360, r1336, r1357;
}
{
cvt.rn.f16.f64 rs184, fd852;
}
mov.b32 r1365, {rs184, rs184};
{
mul.f16x2 r1363, r178, r1365;
}
{
add.f16x2 r1366, r1342, r1363;
}
{
cvt.rn.f16.f64 rs185, fd851;
}
mov.b32 r1371, {rs185, rs185};
{
mul.f16x2 r1369, r172, r1371;
}
{
add.f16x2 r1372, r1348, r1369;
}
{
cvt.rn.f16.f64 rs186, fd852;
}
mov.b32 r1377, {rs186, rs186};
{
mul.f16x2 r1375, r175, r1377;
}
{
add.f16x2 r1378, r1354, r1375;
}
{
sub.f16x2 %6, r1360, r1366;
}
{
add.f16x2 %7, r1372, r1378;
}
{
add.f16x2 %56, r1360, r1366;
}
{
sub.f16x2 %57, r1372, r1378;
}
cvt.rn.f16.s32 rs187, r5508;
mov.b32 r1405, {rs187, rs187};
cvt.rn.f16.s32 rs188, r5508;
mov.b32 r1417, {rs188, rs188};
{
cvt.rn.f16.f64 rs189, fd871;
}
mov.b32 r1397, {rs189, rs189};
{
mul.f16x2 r1395, r1, r1397;
}
{
add.f16x2 r1398, %62, r1395;
}
{
cvt.rn.f16.f64 rs190, fd624;
}
mov.b32 r1403, {rs190, rs190};
{
mul.f16x2 r1401, r10, r1403;
}
{
add.f16x2 r1404, r1405, r1401;
}
{
cvt.rn.f16.f64 rs191, fd871;
}
mov.b32 r1409, {rs191, rs191};
{
mul.f16x2 r1407, r4, r1409;
}
{
add.f16x2 r1410, %63, r1407;
}
{
cvt.rn.f16.f64 rs192, fd624;
}
mov.b32 r1415, {rs192, rs192};
{
mul.f16x2 r1413, r7, r1415;
}
{
add.f16x2 r1416, r1417, r1413;
}
{
cvt.rn.f16.f64 rs193, fd899;
}
mov.b32 r1421, {rs193, rs193};
{
mul.f16x2 r1419, r13, r1421;
}
{
add.f16x2 r1422, r1398, r1419;
}
{
cvt.rn.f16.f64 rs194, fd900;
}
mov.b32 r1427, {rs194, rs194};
{
mul.f16x2 r1425, r22, r1427;
}
{
add.f16x2 r1428, r1404, r1425;
}
{
cvt.rn.f16.f64 rs195, fd899;
}
mov.b32 r1433, {rs195, rs195};
{
mul.f16x2 r1431, r16, r1433;
}
{
add.f16x2 r1434, r1410, r1431;
}
{
cvt.rn.f16.f64 rs196, fd900;
}
mov.b32 r1439, {rs196, rs196};
{
mul.f16x2 r1437, r19, r1439;
}
{
add.f16x2 r1440, r1416, r1437;
}
{
cvt.rn.f16.f64 rs197, fd867;
}
mov.b32 r1445, {rs197, rs197};
{
mul.f16x2 r1443, r25, r1445;
}
{
add.f16x2 r1446, r1422, r1443;
}
{
cvt.rn.f16.f64 rs198, fd868;
}
mov.b32 r1451, {rs198, rs198};
{
mul.f16x2 r1449, r34, r1451;
}
{
add.f16x2 r1452, r1428, r1449;
}
{
cvt.rn.f16.f64 rs199, fd867;
}
mov.b32 r1457, {rs199, rs199};
{
mul.f16x2 r1455, r28, r1457;
}
{
add.f16x2 r1458, r1434, r1455;
}
{
cvt.rn.f16.f64 rs200, fd868;
}
mov.b32 r1463, {rs200, rs200};
{
mul.f16x2 r1461, r31, r1463;
}
{
add.f16x2 r1464, r1440, r1461;
}
{
cvt.rn.f16.f64 rs201, fd843;
}
mov.b32 r1469, {rs201, rs201};
{
mul.f16x2 r1467, r37, r1469;
}
{
add.f16x2 r1470, r1446, r1467;
}
{
cvt.rn.f16.f64 rs202, fd820;
}
mov.b32 r1475, {rs202, rs202};
{
mul.f16x2 r1473, r46, r1475;
}
{
add.f16x2 r1476, r1452, r1473;
}
{
cvt.rn.f16.f64 rs203, fd843;
}
mov.b32 r1481, {rs203, rs203};
{
mul.f16x2 r1479, r40, r1481;
}
{
add.f16x2 r1482, r1458, r1479;
}
{
cvt.rn.f16.f64 rs204, fd820;
}
mov.b32 r1487, {rs204, rs204};
{
mul.f16x2 r1485, r43, r1487;
}
{
add.f16x2 r1488, r1464, r1485;
}
{
cvt.rn.f16.f64 rs205, fd875;
}
mov.b32 r1493, {rs205, rs205};
{
mul.f16x2 r1491, r49, r1493;
}
{
add.f16x2 r1494, r1470, r1491;
}
{
cvt.rn.f16.f64 rs206, fd708;
}
mov.b32 r1499, {rs206, rs206};
{
mul.f16x2 r1497, r58, r1499;
}
{
add.f16x2 r1500, r1476, r1497;
}
{
cvt.rn.f16.f64 rs207, fd875;
}
mov.b32 r1505, {rs207, rs207};
{
mul.f16x2 r1503, r52, r1505;
}
{
add.f16x2 r1506, r1482, r1503;
}
{
cvt.rn.f16.f64 rs208, fd708;
}
mov.b32 r1511, {rs208, rs208};
{
mul.f16x2 r1509, r55, r1511;
}
{
add.f16x2 r1512, r1488, r1509;
}
{
cvt.rn.f16.f64 rs209, fd895;
}
mov.b32 r1517, {rs209, rs209};
{
mul.f16x2 r1515, r61, r1517;
}
{
add.f16x2 r1518, r1494, r1515;
}
{
cvt.rn.f16.f64 rs210, fd896;
}
mov.b32 r1523, {rs210, rs210};
{
mul.f16x2 r1521, r70, r1523;
}
{
add.f16x2 r1524, r1500, r1521;
}
{
cvt.rn.f16.f64 rs211, fd895;
}
mov.b32 r1529, {rs211, rs211};
{
mul.f16x2 r1527, r64, r1529;
}
{
add.f16x2 r1530, r1506, r1527;
}
{
cvt.rn.f16.f64 rs212, fd896;
}
mov.b32 r1535, {rs212, rs212};
{
mul.f16x2 r1533, r67, r1535;
}
{
add.f16x2 r1536, r1512, r1533;
}
{
cvt.rn.f16.f64 rs213, fd863;
}
mov.b32 r1541, {rs213, rs213};
{
mul.f16x2 r1539, r73, r1541;
}
{
add.f16x2 r1542, r1518, r1539;
}
{
cvt.rn.f16.f64 rs214, fd864;
}
mov.b32 r1547, {rs214, rs214};
{
mul.f16x2 r1545, r82, r1547;
}
{
add.f16x2 r1548, r1524, r1545;
}
{
cvt.rn.f16.f64 rs215, fd863;
}
mov.b32 r1553, {rs215, rs215};
{
mul.f16x2 r1551, r76, r1553;
}
{
add.f16x2 r1554, r1530, r1551;
}
{
cvt.rn.f16.f64 rs216, fd864;
}
mov.b32 r1559, {rs216, rs216};
{
mul.f16x2 r1557, r79, r1559;
}
{
add.f16x2 r1560, r1536, r1557;
}
{
cvt.rn.f16.f64 rs217, fd847;
}
mov.b32 r1565, {rs217, rs217};
{
mul.f16x2 r1563, r85, r1565;
}
{
add.f16x2 r1566, r1542, r1563;
}
{
cvt.rn.f16.f64 rs218, fd768;
}
mov.b32 r1571, {rs218, rs218};
{
mul.f16x2 r1569, r94, r1571;
}
{
add.f16x2 r1572, r1548, r1569;
}
{
cvt.rn.f16.f64 rs219, fd847;
}
mov.b32 r1577, {rs219, rs219};
{
mul.f16x2 r1575, r88, r1577;
}
{
add.f16x2 r1578, r1554, r1575;
}
{
cvt.rn.f16.f64 rs220, fd768;
}
mov.b32 r1583, {rs220, rs220};
{
mul.f16x2 r1581, r91, r1583;
}
{
add.f16x2 r1584, r1560, r1581;
}
{
cvt.rn.f16.f64 rs221, fd879;
}
mov.b32 r1589, {rs221, rs221};
{
mul.f16x2 r1587, r97, r1589;
}
{
add.f16x2 r1590, r1566, r1587;
}
{
cvt.rn.f16.f64 rs222, fd808;
}
mov.b32 r1595, {rs222, rs222};
{
mul.f16x2 r1593, r106, r1595;
}
{
add.f16x2 r1596, r1572, r1593;
}
{
cvt.rn.f16.f64 rs223, fd879;
}
mov.b32 r1601, {rs223, rs223};
{
mul.f16x2 r1599, r100, r1601;
}
{
add.f16x2 r1602, r1578, r1599;
}
{
cvt.rn.f16.f64 rs224, fd808;
}
mov.b32 r1607, {rs224, rs224};
{
mul.f16x2 r1605, r103, r1607;
}
{
add.f16x2 r1608, r1584, r1605;
}
{
cvt.rn.f16.f64 rs225, fd891;
}
mov.b32 r1613, {rs225, rs225};
{
mul.f16x2 r1611, r109, r1613;
}
{
add.f16x2 r1614, r1590, r1611;
}
{
cvt.rn.f16.f64 rs226, fd892;
}
mov.b32 r1619, {rs226, rs226};
{
mul.f16x2 r1617, r118, r1619;
}
{
add.f16x2 r1620, r1596, r1617;
}
{
cvt.rn.f16.f64 rs227, fd891;
}
mov.b32 r1625, {rs227, rs227};
{
mul.f16x2 r1623, r112, r1625;
}
{
add.f16x2 r1626, r1602, r1623;
}
{
cvt.rn.f16.f64 rs228, fd892;
}
mov.b32 r1631, {rs228, rs228};
{
mul.f16x2 r1629, r115, r1631;
}
{
add.f16x2 r1632, r1608, r1629;
}
{
cvt.rn.f16.f64 rs229, fd859;
}
mov.b32 r1637, {rs229, rs229};
{
mul.f16x2 r1635, r121, r1637;
}
{
add.f16x2 r1638, r1614, r1635;
}
{
cvt.rn.f16.f64 rs230, fd860;
}
mov.b32 r1643, {rs230, rs230};
{
mul.f16x2 r1641, r130, r1643;
}
{
add.f16x2 r1644, r1620, r1641;
}
{
cvt.rn.f16.f64 rs231, fd859;
}
mov.b32 r1649, {rs231, rs231};
{
mul.f16x2 r1647, r124, r1649;
}
{
add.f16x2 r1650, r1626, r1647;
}
{
cvt.rn.f16.f64 rs232, fd860;
}
mov.b32 r1655, {rs232, rs232};
{
mul.f16x2 r1653, r127, r1655;
}
{
add.f16x2 r1656, r1632, r1653;
}
{
cvt.rn.f16.f64 rs233, fd851;
}
mov.b32 r1661, {rs233, rs233};
{
mul.f16x2 r1659, r133, r1661;
}
{
add.f16x2 r1662, r1638, r1659;
}
mov.f64 fd676, 0dBFD328C3F1B322CB;
{
cvt.rn.f16.f64 rs234, fd676;
}
mov.b32 r1667, {rs234, rs234};
{
mul.f16x2 r1665, r142, r1667;
}
{
add.f16x2 r1668, r1644, r1665;
}
{
cvt.rn.f16.f64 rs235, fd851;
}
mov.b32 r1673, {rs235, rs235};
{
mul.f16x2 r1671, r136, r1673;
}
{
add.f16x2 r1674, r1650, r1671;
}
{
cvt.rn.f16.f64 rs236, fd676;
}
mov.b32 r1679, {rs236, rs236};
{
mul.f16x2 r1677, r139, r1679;
}
{
add.f16x2 r1680, r1656, r1677;
}
{
cvt.rn.f16.f64 rs237, fd883;
}
mov.b32 r1685, {rs237, rs237};
{
mul.f16x2 r1683, r145, r1685;
}
{
add.f16x2 r1686, r1662, r1683;
}
{
cvt.rn.f16.f64 rs238, fd736;
}
mov.b32 r1691, {rs238, rs238};
{
mul.f16x2 r1689, r154, r1691;
}
{
add.f16x2 r1692, r1668, r1689;
}
{
cvt.rn.f16.f64 rs239, fd883;
}
mov.b32 r1697, {rs239, rs239};
{
mul.f16x2 r1695, r148, r1697;
}
{
add.f16x2 r1698, r1674, r1695;
}
{
cvt.rn.f16.f64 rs240, fd736;
}
mov.b32 r1703, {rs240, rs240};
{
mul.f16x2 r1701, r151, r1703;
}
{
add.f16x2 r1704, r1680, r1701;
}
{
cvt.rn.f16.f64 rs241, fd887;
}
mov.b32 r1709, {rs241, rs241};
{
mul.f16x2 r1707, r157, r1709;
}
{
add.f16x2 r1710, r1686, r1707;
}
mov.f64 fd888, 0dBFEE0210C26A6E6F;
{
cvt.rn.f16.f64 rs242, fd888;
}
mov.b32 r1715, {rs242, rs242};
{
mul.f16x2 r1713, r166, r1715;
}
{
add.f16x2 r1716, r1692, r1713;
}
{
cvt.rn.f16.f64 rs243, fd887;
}
mov.b32 r1721, {rs243, rs243};
{
mul.f16x2 r1719, r160, r1721;
}
{
add.f16x2 r1722, r1698, r1719;
}
{
cvt.rn.f16.f64 rs244, fd888;
}
mov.b32 r1727, {rs244, rs244};
{
mul.f16x2 r1725, r163, r1727;
}
{
add.f16x2 r1728, r1704, r1725;
}
{
cvt.rn.f16.f64 rs245, fd855;
}
mov.b32 r1733, {rs245, rs245};
{
mul.f16x2 r1731, r169, r1733;
}
{
add.f16x2 r1734, r1710, r1731;
}
mov.f64 fd856, 0dBFD93D20572CA90B;
{
cvt.rn.f16.f64 rs246, fd856;
}
mov.b32 r1739, {rs246, rs246};
{
mul.f16x2 r1737, r178, r1739;
}
{
add.f16x2 r1740, r1716, r1737;
}
{
cvt.rn.f16.f64 rs247, fd855;
}
mov.b32 r1745, {rs247, rs247};
{
mul.f16x2 r1743, r172, r1745;
}
{
add.f16x2 r1746, r1722, r1743;
}
{
cvt.rn.f16.f64 rs248, fd856;
}
mov.b32 r1751, {rs248, rs248};
{
mul.f16x2 r1749, r175, r1751;
}
{
add.f16x2 r1752, r1728, r1749;
}
{
sub.f16x2 %8, r1734, r1740;
}
{
add.f16x2 %9, r1746, r1752;
}
{
add.f16x2 %54, r1734, r1740;
}
{
sub.f16x2 %55, r1746, r1752;
}
cvt.rn.f16.s32 rs249, r5508;
mov.b32 r1779, {rs249, rs249};
cvt.rn.f16.s32 rs250, r5508;
mov.b32 r1791, {rs250, rs250};
{
cvt.rn.f16.f64 rs251, fd879;
}
mov.b32 r1771, {rs251, rs251};
{
mul.f16x2 r1769, r1, r1771;
}
{
add.f16x2 r1772, %62, r1769;
}
{
cvt.rn.f16.f64 rs252, fd808;
}
mov.b32 r1777, {rs252, rs252};
{
mul.f16x2 r1775, r10, r1777;
}
{
add.f16x2 r1778, r1779, r1775;
}
{
cvt.rn.f16.f64 rs253, fd879;
}
mov.b32 r1783, {rs253, rs253};
{
mul.f16x2 r1781, r4, r1783;
}
{
add.f16x2 r1784, %63, r1781;
}
{
cvt.rn.f16.f64 rs254, fd808;
}
mov.b32 r1789, {rs254, rs254};
{
mul.f16x2 r1787, r7, r1789;
}
{
add.f16x2 r1790, r1791, r1787;
}
{
cvt.rn.f16.f64 rs255, fd883;
}
mov.b32 r1795, {rs255, rs255};
{
mul.f16x2 r1793, r13, r1795;
}
{
add.f16x2 r1796, r1772, r1793;
}
{
cvt.rn.f16.f64 rs256, fd884;
}
mov.b32 r1801, {rs256, rs256};
{
mul.f16x2 r1799, r22, r1801;
}
{
add.f16x2 r1802, r1778, r1799;
}
{
cvt.rn.f16.f64 rs257, fd883;
}
mov.b32 r1807, {rs257, rs257};
{
mul.f16x2 r1805, r16, r1807;
}
{
add.f16x2 r1808, r1784, r1805;
}
{
cvt.rn.f16.f64 rs258, fd884;
}
mov.b32 r1813, {rs258, rs258};
{
mul.f16x2 r1811, r19, r1813;
}
{
add.f16x2 r1814, r1790, r1811;
}
{
cvt.rn.f16.f64 rs259, fd843;
}
mov.b32 r1819, {rs259, rs259};
{
mul.f16x2 r1817, r25, r1819;
}
{
add.f16x2 r1820, r1796, r1817;
}
{
cvt.rn.f16.f64 rs260, fd844;
}
mov.b32 r1825, {rs260, rs260};
{
mul.f16x2 r1823, r34, r1825;
}
{
add.f16x2 r1826, r1802, r1823;
}
{
cvt.rn.f16.f64 rs261, fd843;
}
mov.b32 r1831, {rs261, rs261};
{
mul.f16x2 r1829, r28, r1831;
}
{
add.f16x2 r1832, r1808, r1829;
}
{
cvt.rn.f16.f64 rs262, fd844;
}
mov.b32 r1837, {rs262, rs262};
{
mul.f16x2 r1835, r31, r1837;
}
{
add.f16x2 r1838, r1814, r1835;
}
{
cvt.rn.f16.f64 rs263, fd875;
}
mov.b32 r1843, {rs263, rs263};
{
mul.f16x2 r1841, r37, r1843;
}
{
add.f16x2 r1844, r1820, r1841;
}
{
cvt.rn.f16.f64 rs264, fd708;
}
mov.b32 r1849, {rs264, rs264};
{
mul.f16x2 r1847, r46, r1849;
}
{
add.f16x2 r1850, r1826, r1847;
}
{
cvt.rn.f16.f64 rs265, fd875;
}
mov.b32 r1855, {rs265, rs265};
{
mul.f16x2 r1853, r40, r1855;
}
{
add.f16x2 r1856, r1832, r1853;
}
{
cvt.rn.f16.f64 rs266, fd708;
}
mov.b32 r1861, {rs266, rs266};
{
mul.f16x2 r1859, r43, r1861;
}
{
add.f16x2 r1862, r1838, r1859;
}
{
cvt.rn.f16.f64 rs267, fd887;
}
mov.b32 r1867, {rs267, rs267};
{
mul.f16x2 r1865, r49, r1867;
}
{
add.f16x2 r1868, r1844, r1865;
}
{
cvt.rn.f16.f64 rs268, fd888;
}
mov.b32 r1873, {rs268, rs268};
{
mul.f16x2 r1871, r58, r1873;
}
{
add.f16x2 r1874, r1850, r1871;
}
{
cvt.rn.f16.f64 rs269, fd887;
}
mov.b32 r1879, {rs269, rs269};
{
mul.f16x2 r1877, r52, r1879;
}
{
add.f16x2 r1880, r1856, r1877;
}
{
cvt.rn.f16.f64 rs270, fd888;
}
mov.b32 r1885, {rs270, rs270};
{
mul.f16x2 r1883, r55, r1885;
}
{
add.f16x2 r1886, r1862, r1883;
}
{
cvt.rn.f16.f64 rs271, fd847;
}
mov.b32 r1891, {rs271, rs271};
{
mul.f16x2 r1889, r61, r1891;
}
{
add.f16x2 r1892, r1868, r1889;
}
{
cvt.rn.f16.f64 rs272, fd848;
}
mov.b32 r1897, {rs272, rs272};
{
mul.f16x2 r1895, r70, r1897;
}
{
add.f16x2 r1898, r1874, r1895;
}
{
cvt.rn.f16.f64 rs273, fd847;
}
mov.b32 r1903, {rs273, rs273};
{
mul.f16x2 r1901, r64, r1903;
}
{
add.f16x2 r1904, r1880, r1901;
}
{
cvt.rn.f16.f64 rs274, fd848;
}
mov.b32 r1909, {rs274, rs274};
{
mul.f16x2 r1907, r67, r1909;
}
{
add.f16x2 r1910, r1886, r1907;
}
{
cvt.rn.f16.f64 rs275, fd871;
}
mov.b32 r1915, {rs275, rs275};
{
mul.f16x2 r1913, r73, r1915;
}
{
add.f16x2 r1916, r1892, r1913;
}
{
cvt.rn.f16.f64 rs276, fd624;
}
mov.b32 r1921, {rs276, rs276};
{
mul.f16x2 r1919, r82, r1921;
}
{
add.f16x2 r1922, r1898, r1919;
}
{
cvt.rn.f16.f64 rs277, fd871;
}
mov.b32 r1927, {rs277, rs277};
{
mul.f16x2 r1925, r76, r1927;
}
{
add.f16x2 r1928, r1904, r1925;
}
{
cvt.rn.f16.f64 rs278, fd624;
}
mov.b32 r1933, {rs278, rs278};
{
mul.f16x2 r1931, r79, r1933;
}
{
add.f16x2 r1934, r1910, r1931;
}
{
cvt.rn.f16.f64 rs279, fd891;
}
mov.b32 r1939, {rs279, rs279};
{
mul.f16x2 r1937, r85, r1939;
}
{
add.f16x2 r1940, r1916, r1937;
}
{
cvt.rn.f16.f64 rs280, fd892;
}
mov.b32 r1945, {rs280, rs280};
{
mul.f16x2 r1943, r94, r1945;
}
{
add.f16x2 r1946, r1922, r1943;
}
{
cvt.rn.f16.f64 rs281, fd891;
}
mov.b32 r1951, {rs281, rs281};
{
mul.f16x2 r1949, r88, r1951;
}
{
add.f16x2 r1952, r1928, r1949;
}
{
cvt.rn.f16.f64 rs282, fd892;
}
mov.b32 r1957, {rs282, rs282};
{
mul.f16x2 r1955, r91, r1957;
}
{
add.f16x2 r1958, r1934, r1955;
}
{
cvt.rn.f16.f64 rs283, fd851;
}
mov.b32 r1963, {rs283, rs283};
{
mul.f16x2 r1961, r97, r1963;
}
{
add.f16x2 r1964, r1940, r1961;
}
{
cvt.rn.f16.f64 rs284, fd852;
}
mov.b32 r1969, {rs284, rs284};
{
mul.f16x2 r1967, r106, r1969;
}
{
add.f16x2 r1970, r1946, r1967;
}
{
cvt.rn.f16.f64 rs285, fd851;
}
mov.b32 r1975, {rs285, rs285};
{
mul.f16x2 r1973, r100, r1975;
}
{
add.f16x2 r1976, r1952, r1973;
}
{
cvt.rn.f16.f64 rs286, fd852;
}
mov.b32 r1981, {rs286, rs286};
{
mul.f16x2 r1979, r103, r1981;
}
{
add.f16x2 r1982, r1958, r1979;
}
{
cvt.rn.f16.f64 rs287, fd867;
}
mov.b32 r1987, {rs287, rs287};
{
mul.f16x2 r1985, r109, r1987;
}
{
add.f16x2 r1988, r1964, r1985;
}
mov.f64 fd812, 0dBFE4D80B1AD9CCF6;
{
cvt.rn.f16.f64 rs288, fd812;
}
mov.b32 r1993, {rs288, rs288};
{
mul.f16x2 r1991, r118, r1993;
}
{
add.f16x2 r1994, r1970, r1991;
}
{
cvt.rn.f16.f64 rs289, fd867;
}
mov.b32 r1999, {rs289, rs289};
{
mul.f16x2 r1997, r112, r1999;
}
{
add.f16x2 r2000, r1976, r1997;
}
{
cvt.rn.f16.f64 rs290, fd812;
}
mov.b32 r2005, {rs290, rs290};
{
mul.f16x2 r2003, r115, r2005;
}
{
add.f16x2 r2006, r1982, r2003;
}
{
cvt.rn.f16.f64 rs291, fd895;
}
mov.b32 r2011, {rs291, rs291};
{
mul.f16x2 r2009, r121, r2011;
}
{
add.f16x2 r2012, r1988, r2009;
}
{
cvt.rn.f16.f64 rs292, fd896;
}
mov.b32 r2017, {rs292, rs292};
{
mul.f16x2 r2015, r130, r2017;
}
{
add.f16x2 r2018, r1994, r2015;
}
{
cvt.rn.f16.f64 rs293, fd895;
}
mov.b32 r2023, {rs293, rs293};
{
mul.f16x2 r2021, r124, r2023;
}
{
add.f16x2 r2024, r2000, r2021;
}
{
cvt.rn.f16.f64 rs294, fd896;
}
mov.b32 r2029, {rs294, rs294};
{
mul.f16x2 r2027, r127, r2029;
}
{
add.f16x2 r2030, r2006, r2027;
}
{
cvt.rn.f16.f64 rs295, fd855;
}
mov.b32 r2035, {rs295, rs295};
{
mul.f16x2 r2033, r133, r2035;
}
{
add.f16x2 r2036, r2012, r2033;
}
{
cvt.rn.f16.f64 rs296, fd856;
}
mov.b32 r2041, {rs296, rs296};
{
mul.f16x2 r2039, r142, r2041;
}
{
add.f16x2 r2042, r2018, r2039;
}
{
cvt.rn.f16.f64 rs297, fd855;
}
mov.b32 r2047, {rs297, rs297};
{
mul.f16x2 r2045, r136, r2047;
}
{
add.f16x2 r2048, r2024, r2045;
}
{
cvt.rn.f16.f64 rs298, fd856;
}
mov.b32 r2053, {rs298, rs298};
{
mul.f16x2 r2051, r139, r2053;
}
{
add.f16x2 r2054, r2030, r2051;
}
{
cvt.rn.f16.f64 rs299, fd863;
}
mov.b32 r2059, {rs299, rs299};
{
mul.f16x2 r2057, r145, r2059;
}
{
add.f16x2 r2060, r2036, r2057;
}
{
cvt.rn.f16.f64 rs300, fd740;
}
mov.b32 r2065, {rs300, rs300};
{
mul.f16x2 r2063, r154, r2065;
}
{
add.f16x2 r2066, r2042, r2063;
}
{
cvt.rn.f16.f64 rs301, fd863;
}
mov.b32 r2071, {rs301, rs301};
{
mul.f16x2 r2069, r148, r2071;
}
{
add.f16x2 r2072, r2048, r2069;
}
{
cvt.rn.f16.f64 rs302, fd740;
}
mov.b32 r2077, {rs302, rs302};
{
mul.f16x2 r2075, r151, r2077;
}
{
add.f16x2 r2078, r2054, r2075;
}
{
cvt.rn.f16.f64 rs303, fd899;
}
mov.b32 r2083, {rs303, rs303};
{
mul.f16x2 r2081, r157, r2083;
}
{
add.f16x2 r2084, r2060, r2081;
}
{
cvt.rn.f16.f64 rs304, fd900;
}
mov.b32 r2089, {rs304, rs304};
{
mul.f16x2 r2087, r166, r2089;
}
{
add.f16x2 r2090, r2066, r2087;
}
{
cvt.rn.f16.f64 rs305, fd899;
}
mov.b32 r2095, {rs305, rs305};
{
mul.f16x2 r2093, r160, r2095;
}
{
add.f16x2 r2096, r2072, r2093;
}
{
cvt.rn.f16.f64 rs306, fd900;
}
mov.b32 r2101, {rs306, rs306};
{
mul.f16x2 r2099, r163, r2101;
}
{
add.f16x2 r2102, r2078, r2099;
}
{
cvt.rn.f16.f64 rs307, fd859;
}
mov.b32 r2107, {rs307, rs307};
{
mul.f16x2 r2105, r169, r2107;
}
{
add.f16x2 r2108, r2084, r2105;
}
{
cvt.rn.f16.f64 rs308, fd860;
}
mov.b32 r2113, {rs308, rs308};
{
mul.f16x2 r2111, r178, r2113;
}
{
add.f16x2 r2114, r2090, r2111;
}
{
cvt.rn.f16.f64 rs309, fd859;
}
mov.b32 r2119, {rs309, rs309};
{
mul.f16x2 r2117, r172, r2119;
}
{
add.f16x2 r2120, r2096, r2117;
}
{
cvt.rn.f16.f64 rs310, fd860;
}
mov.b32 r2125, {rs310, rs310};
{
mul.f16x2 r2123, r175, r2125;
}
{
add.f16x2 r2126, r2102, r2123;
}
{
sub.f16x2 %10, r2108, r2114;
}
{
add.f16x2 %11, r2120, r2126;
}
{
add.f16x2 %52, r2108, r2114;
}
{
sub.f16x2 %53, r2120, r2126;
}
cvt.rn.f16.s32 rs311, r5508;
mov.b32 r2153, {rs311, rs311};
cvt.rn.f16.s32 rs312, r5508;
mov.b32 r2165, {rs312, rs312};
{
cvt.rn.f16.f64 rs313, fd887;
}
mov.b32 r2145, {rs313, rs313};
{
mul.f16x2 r2143, r1, r2145;
}
{
add.f16x2 r2146, %62, r2143;
}
{
cvt.rn.f16.f64 rs314, fd760;
}
mov.b32 r2151, {rs314, rs314};
{
mul.f16x2 r2149, r10, r2151;
}
{
add.f16x2 r2152, r2153, r2149;
}
{
cvt.rn.f16.f64 rs315, fd887;
}
mov.b32 r2157, {rs315, rs315};
{
mul.f16x2 r2155, r4, r2157;
}
{
add.f16x2 r2158, %63, r2155;
}
{
cvt.rn.f16.f64 rs316, fd760;
}
mov.b32 r2163, {rs316, rs316};
{
mul.f16x2 r2161, r7, r2163;
}
{
add.f16x2 r2164, r2165, r2161;
}
{
cvt.rn.f16.f64 rs317, fd867;
}
mov.b32 r2169, {rs317, rs317};
{
mul.f16x2 r2167, r13, r2169;
}
{
add.f16x2 r2170, r2146, r2167;
}
{
cvt.rn.f16.f64 rs318, fd868;
}
mov.b32 r2175, {rs318, rs318};
{
mul.f16x2 r2173, r22, r2175;
}
{
add.f16x2 r2176, r2152, r2173;
}
{
cvt.rn.f16.f64 rs319, fd867;
}
mov.b32 r2181, {rs319, rs319};
{
mul.f16x2 r2179, r16, r2181;
}
{
add.f16x2 r2182, r2158, r2179;
}
{
cvt.rn.f16.f64 rs320, fd868;
}
mov.b32 r2187, {rs320, rs320};
{
mul.f16x2 r2185, r19, r2187;
}
{
add.f16x2 r2188, r2164, r2185;
}
{
cvt.rn.f16.f64 rs321, fd859;
}
mov.b32 r2193, {rs321, rs321};
{
mul.f16x2 r2191, r25, r2193;
}
{
add.f16x2 r2194, r2170, r2191;
}
{
cvt.rn.f16.f64 rs322, fd572;
}
mov.b32 r2199, {rs322, rs322};
{
mul.f16x2 r2197, r34, r2199;
}
{
add.f16x2 r2200, r2176, r2197;
}
{
cvt.rn.f16.f64 rs323, fd859;
}
mov.b32 r2205, {rs323, rs323};
{
mul.f16x2 r2203, r28, r2205;
}
{
add.f16x2 r2206, r2182, r2203;
}
{
cvt.rn.f16.f64 rs324, fd572;
}
mov.b32 r2211, {rs324, rs324};
{
mul.f16x2 r2209, r31, r2211;
}
{
add.f16x2 r2212, r2188, r2209;
}
{
cvt.rn.f16.f64 rs325, fd895;
}
mov.b32 r2217, {rs325, rs325};
{
mul.f16x2 r2215, r37, r2217;
}
{
add.f16x2 r2218, r2194, r2215;
}
{
cvt.rn.f16.f64 rs326, fd896;
}
mov.b32 r2223, {rs326, rs326};
{
mul.f16x2 r2221, r46, r2223;
}
{
add.f16x2 r2224, r2200, r2221;
}
{
cvt.rn.f16.f64 rs327, fd895;
}
mov.b32 r2229, {rs327, rs327};
{
mul.f16x2 r2227, r40, r2229;
}
{
add.f16x2 r2230, r2206, r2227;
}
{
cvt.rn.f16.f64 rs328, fd896;
}
mov.b32 r2235, {rs328, rs328};
{
mul.f16x2 r2233, r43, r2235;
}
{
add.f16x2 r2236, r2212, r2233;
}
{
cvt.rn.f16.f64 rs329, fd847;
}
mov.b32 r2241, {rs329, rs329};
{
mul.f16x2 r2239, r49, r2241;
}
{
add.f16x2 r2242, r2218, r2239;
}
{
cvt.rn.f16.f64 rs330, fd848;
}
mov.b32 r2247, {rs330, rs330};
{
mul.f16x2 r2245, r58, r2247;
}
{
add.f16x2 r2248, r2224, r2245;
}
{
cvt.rn.f16.f64 rs331, fd847;
}
mov.b32 r2253, {rs331, rs331};
{
mul.f16x2 r2251, r52, r2253;
}
{
add.f16x2 r2254, r2230, r2251;
}
{
cvt.rn.f16.f64 rs332, fd848;
}
mov.b32 r2259, {rs332, rs332};
{
mul.f16x2 r2257, r55, r2259;
}
{
add.f16x2 r2260, r2236, r2257;
}
{
cvt.rn.f16.f64 rs333, fd879;
}
mov.b32 r2265, {rs333, rs333};
{
mul.f16x2 r2263, r61, r2265;
}
{
add.f16x2 r2266, r2242, r2263;
}
{
cvt.rn.f16.f64 rs334, fd808;
}
mov.b32 r2271, {rs334, rs334};
{
mul.f16x2 r2269, r70, r2271;
}
{
add.f16x2 r2272, r2248, r2269;
}
{
cvt.rn.f16.f64 rs335, fd879;
}
mov.b32 r2277, {rs335, rs335};
{
mul.f16x2 r2275, r64, r2277;
}
{
add.f16x2 r2278, r2254, r2275;
}
{
cvt.rn.f16.f64 rs336, fd808;
}
mov.b32 r2283, {rs336, rs336};
{
mul.f16x2 r2281, r67, r2283;
}
{
add.f16x2 r2284, r2260, r2281;
}
{
cvt.rn.f16.f64 rs337, fd875;
}
mov.b32 r2289, {rs337, rs337};
{
mul.f16x2 r2287, r73, r2289;
}
{
add.f16x2 r2290, r2266, r2287;
}
{
cvt.rn.f16.f64 rs338, fd876;
}
mov.b32 r2295, {rs338, rs338};
{
mul.f16x2 r2293, r82, r2295;
}
{
add.f16x2 r2296, r2272, r2293;
}
{
cvt.rn.f16.f64 rs339, fd875;
}
mov.b32 r2301, {rs339, rs339};
{
mul.f16x2 r2299, r76, r2301;
}
{
add.f16x2 r2302, r2278, r2299;
}
{
cvt.rn.f16.f64 rs340, fd876;
}
mov.b32 r2307, {rs340, rs340};
{
mul.f16x2 r2305, r79, r2307;
}
{
add.f16x2 r2308, r2284, r2305;
}
{
cvt.rn.f16.f64 rs341, fd851;
}
mov.b32 r2313, {rs341, rs341};
{
mul.f16x2 r2311, r85, r2313;
}
{
add.f16x2 r2314, r2290, r2311;
}
{
cvt.rn.f16.f64 rs342, fd676;
}
mov.b32 r2319, {rs342, rs342};
{
mul.f16x2 r2317, r94, r2319;
}
{
add.f16x2 r2320, r2296, r2317;
}
{
cvt.rn.f16.f64 rs343, fd851;
}
mov.b32 r2325, {rs343, rs343};
{
mul.f16x2 r2323, r88, r2325;
}
{
add.f16x2 r2326, r2302, r2323;
}
{
cvt.rn.f16.f64 rs344, fd676;
}
mov.b32 r2331, {rs344, rs344};
{
mul.f16x2 r2329, r91, r2331;
}
{
add.f16x2 r2332, r2308, r2329;
}
{
cvt.rn.f16.f64 rs345, fd899;
}
mov.b32 r2337, {rs345, rs345};
{
mul.f16x2 r2335, r97, r2337;
}
{
add.f16x2 r2338, r2314, r2335;
}
mov.f64 fd504, 0dBFEFF57C5208CCF9;
{
cvt.rn.f16.f64 rs346, fd504;
}
mov.b32 r2343, {rs346, rs346};
{
mul.f16x2 r2341, r106, r2343;
}
{
add.f16x2 r2344, r2320, r2341;
}
{
cvt.rn.f16.f64 rs347, fd899;
}
mov.b32 r2349, {rs347, rs347};
{
mul.f16x2 r2347, r100, r2349;
}
{
add.f16x2 r2350, r2326, r2347;
}
{
cvt.rn.f16.f64 rs348, fd504;
}
mov.b32 r2355, {rs348, rs348};
{
mul.f16x2 r2353, r103, r2355;
}
{
add.f16x2 r2356, r2332, r2353;
}
{
cvt.rn.f16.f64 rs349, fd855;
}
mov.b32 r2361, {rs349, rs349};
{
mul.f16x2 r2359, r109, r2361;
}
{
add.f16x2 r2362, r2338, r2359;
}
{
cvt.rn.f16.f64 rs350, fd856;
}
mov.b32 r2367, {rs350, rs350};
{
mul.f16x2 r2365, r118, r2367;
}
{
add.f16x2 r2368, r2344, r2365;
}
{
cvt.rn.f16.f64 rs351, fd855;
}
mov.b32 r2373, {rs351, rs351};
{
mul.f16x2 r2371, r112, r2373;
}
{
add.f16x2 r2374, r2350, r2371;
}
{
cvt.rn.f16.f64 rs352, fd856;
}
mov.b32 r2379, {rs352, rs352};
{
mul.f16x2 r2377, r115, r2379;
}
{
add.f16x2 r2380, r2356, r2377;
}
{
cvt.rn.f16.f64 rs353, fd871;
}
mov.b32 r2385, {rs353, rs353};
{
mul.f16x2 r2383, r121, r2385;
}
{
add.f16x2 r2386, r2362, r2383;
}
{
cvt.rn.f16.f64 rs354, fd624;
}
mov.b32 r2391, {rs354, rs354};
{
mul.f16x2 r2389, r130, r2391;
}
{
add.f16x2 r2392, r2368, r2389;
}
{
cvt.rn.f16.f64 rs355, fd871;
}
mov.b32 r2397, {rs355, rs355};
{
mul.f16x2 r2395, r124, r2397;
}
{
add.f16x2 r2398, r2374, r2395;
}
{
cvt.rn.f16.f64 rs356, fd624;
}
mov.b32 r2403, {rs356, rs356};
{
mul.f16x2 r2401, r127, r2403;
}
{
add.f16x2 r2404, r2380, r2401;
}
{
cvt.rn.f16.f64 rs357, fd883;
}
mov.b32 r2409, {rs357, rs357};
{
mul.f16x2 r2407, r133, r2409;
}
{
add.f16x2 r2410, r2386, r2407;
}
{
cvt.rn.f16.f64 rs358, fd884;
}
mov.b32 r2415, {rs358, rs358};
{
mul.f16x2 r2413, r142, r2415;
}
{
add.f16x2 r2416, r2392, r2413;
}
{
cvt.rn.f16.f64 rs359, fd883;
}
mov.b32 r2421, {rs359, rs359};
{
mul.f16x2 r2419, r136, r2421;
}
{
add.f16x2 r2422, r2398, r2419;
}
{
cvt.rn.f16.f64 rs360, fd884;
}
mov.b32 r2427, {rs360, rs360};
{
mul.f16x2 r2425, r139, r2427;
}
{
add.f16x2 r2428, r2404, r2425;
}
{
cvt.rn.f16.f64 rs361, fd843;
}
mov.b32 r2433, {rs361, rs361};
{
mul.f16x2 r2431, r145, r2433;
}
{
add.f16x2 r2434, r2410, r2431;
}
{
cvt.rn.f16.f64 rs362, fd820;
}
mov.b32 r2439, {rs362, rs362};
{
mul.f16x2 r2437, r154, r2439;
}
{
add.f16x2 r2440, r2416, r2437;
}
{
cvt.rn.f16.f64 rs363, fd843;
}
mov.b32 r2445, {rs363, rs363};
{
mul.f16x2 r2443, r148, r2445;
}
{
add.f16x2 r2446, r2422, r2443;
}
{
cvt.rn.f16.f64 rs364, fd820;
}
mov.b32 r2451, {rs364, rs364};
{
mul.f16x2 r2449, r151, r2451;
}
{
add.f16x2 r2452, r2428, r2449;
}
{
cvt.rn.f16.f64 rs365, fd891;
}
mov.b32 r2457, {rs365, rs365};
{
mul.f16x2 r2455, r157, r2457;
}
{
add.f16x2 r2458, r2434, r2455;
}
{
cvt.rn.f16.f64 rs366, fd804;
}
mov.b32 r2463, {rs366, rs366};
{
mul.f16x2 r2461, r166, r2463;
}
{
add.f16x2 r2464, r2440, r2461;
}
{
cvt.rn.f16.f64 rs367, fd891;
}
mov.b32 r2469, {rs367, rs367};
{
mul.f16x2 r2467, r160, r2469;
}
{
add.f16x2 r2470, r2446, r2467;
}
{
cvt.rn.f16.f64 rs368, fd804;
}
mov.b32 r2475, {rs368, rs368};
{
mul.f16x2 r2473, r163, r2475;
}
{
add.f16x2 r2476, r2452, r2473;
}
{
cvt.rn.f16.f64 rs369, fd863;
}
mov.b32 r2481, {rs369, rs369};
{
mul.f16x2 r2479, r169, r2481;
}
{
add.f16x2 r2482, r2458, r2479;
}
{
cvt.rn.f16.f64 rs370, fd864;
}
mov.b32 r2487, {rs370, rs370};
{
mul.f16x2 r2485, r178, r2487;
}
{
add.f16x2 r2488, r2464, r2485;
}
{
cvt.rn.f16.f64 rs371, fd863;
}
mov.b32 r2493, {rs371, rs371};
{
mul.f16x2 r2491, r172, r2493;
}
{
add.f16x2 r2494, r2470, r2491;
}
{
cvt.rn.f16.f64 rs372, fd864;
}
mov.b32 r2499, {rs372, rs372};
{
mul.f16x2 r2497, r175, r2499;
}
{
add.f16x2 r2500, r2476, r2497;
}
{
sub.f16x2 %12, r2482, r2488;
}
{
add.f16x2 %13, r2494, r2500;
}
{
add.f16x2 %50, r2482, r2488;
}
{
sub.f16x2 %51, r2494, r2500;
}
cvt.rn.f16.s32 rs373, r5508;
mov.b32 r2527, {rs373, rs373};
cvt.rn.f16.s32 rs374, r5508;
mov.b32 r2539, {rs374, rs374};
{
cvt.rn.f16.f64 rs375, fd895;
}
mov.b32 r2519, {rs375, rs375};
{
mul.f16x2 r2517, r1, r2519;
}
{
add.f16x2 r2520, %62, r2517;
}
{
cvt.rn.f16.f64 rs376, fd580;
}
mov.b32 r2525, {rs376, rs376};
{
mul.f16x2 r2523, r10, r2525;
}
{
add.f16x2 r2526, r2527, r2523;
}
{
cvt.rn.f16.f64 rs377, fd895;
}
mov.b32 r2531, {rs377, rs377};
{
mul.f16x2 r2529, r4, r2531;
}
{
add.f16x2 r2532, %63, r2529;
}
{
cvt.rn.f16.f64 rs378, fd580;
}
mov.b32 r2537, {rs378, rs378};
{
mul.f16x2 r2535, r7, r2537;
}
{
add.f16x2 r2538, r2539, r2535;
}
{
cvt.rn.f16.f64 rs379, fd851;
}
mov.b32 r2543, {rs379, rs379};
{
mul.f16x2 r2541, r13, r2543;
}
{
add.f16x2 r2544, r2520, r2541;
}
{
cvt.rn.f16.f64 rs380, fd852;
}
mov.b32 r2549, {rs380, rs380};
{
mul.f16x2 r2547, r22, r2549;
}
{
add.f16x2 r2550, r2526, r2547;
}
{
cvt.rn.f16.f64 rs381, fd851;
}
mov.b32 r2555, {rs381, rs381};
{
mul.f16x2 r2553, r16, r2555;
}
{
add.f16x2 r2556, r2532, r2553;
}
{
cvt.rn.f16.f64 rs382, fd852;
}
mov.b32 r2561, {rs382, rs382};
{
mul.f16x2 r2559, r19, r2561;
}
{
add.f16x2 r2562, r2538, r2559;
}
{
cvt.rn.f16.f64 rs383, fd883;
}
mov.b32 r2567, {rs383, rs383};
{
mul.f16x2 r2565, r25, r2567;
}
{
add.f16x2 r2568, r2544, r2565;
}
{
cvt.rn.f16.f64 rs384, fd736;
}
mov.b32 r2573, {rs384, rs384};
{
mul.f16x2 r2571, r34, r2573;
}
{
add.f16x2 r2574, r2550, r2571;
}
{
cvt.rn.f16.f64 rs385, fd883;
}
mov.b32 r2579, {rs385, rs385};
{
mul.f16x2 r2577, r28, r2579;
}
{
add.f16x2 r2580, r2556, r2577;
}
{
cvt.rn.f16.f64 rs386, fd736;
}
mov.b32 r2585, {rs386, rs386};
{
mul.f16x2 r2583, r31, r2585;
}
{
add.f16x2 r2586, r2562, r2583;
}
{
cvt.rn.f16.f64 rs387, fd863;
}
mov.b32 r2591, {rs387, rs387};
{
mul.f16x2 r2589, r37, r2591;
}
{
add.f16x2 r2592, r2568, r2589;
}
{
cvt.rn.f16.f64 rs388, fd864;
}
mov.b32 r2597, {rs388, rs388};
{
mul.f16x2 r2595, r46, r2597;
}
{
add.f16x2 r2598, r2574, r2595;
}
{
cvt.rn.f16.f64 rs389, fd863;
}
mov.b32 r2603, {rs389, rs389};
{
mul.f16x2 r2601, r40, r2603;
}
{
add.f16x2 r2604, r2580, r2601;
}
{
cvt.rn.f16.f64 rs390, fd864;
}
mov.b32 r2609, {rs390, rs390};
{
mul.f16x2 r2607, r43, r2609;
}
{
add.f16x2 r2610, r2586, r2607;
}
{
cvt.rn.f16.f64 rs391, fd871;
}
mov.b32 r2615, {rs391, rs391};
{
mul.f16x2 r2613, r49, r2615;
}
{
add.f16x2 r2616, r2592, r2613;
}
{
cvt.rn.f16.f64 rs392, fd624;
}
mov.b32 r2621, {rs392, rs392};
{
mul.f16x2 r2619, r58, r2621;
}
{
add.f16x2 r2622, r2598, r2619;
}
{
cvt.rn.f16.f64 rs393, fd871;
}
mov.b32 r2627, {rs393, rs393};
{
mul.f16x2 r2625, r52, r2627;
}
{
add.f16x2 r2628, r2604, r2625;
}
{
cvt.rn.f16.f64 rs394, fd624;
}
mov.b32 r2633, {rs394, rs394};
{
mul.f16x2 r2631, r55, r2633;
}
{
add.f16x2 r2634, r2610, r2631;
}
{
cvt.rn.f16.f64 rs395, fd875;
}
mov.b32 r2639, {rs395, rs395};
{
mul.f16x2 r2637, r61, r2639;
}
{
add.f16x2 r2640, r2616, r2637;
}
{
cvt.rn.f16.f64 rs396, fd876;
}
mov.b32 r2645, {rs396, rs396};
{
mul.f16x2 r2643, r70, r2645;
}
{
add.f16x2 r2646, r2622, r2643;
}
{
cvt.rn.f16.f64 rs397, fd875;
}
mov.b32 r2651, {rs397, rs397};
{
mul.f16x2 r2649, r64, r2651;
}
{
add.f16x2 r2652, r2628, r2649;
}
{
cvt.rn.f16.f64 rs398, fd876;
}
mov.b32 r2657, {rs398, rs398};
{
mul.f16x2 r2655, r67, r2657;
}
{
add.f16x2 r2658, r2634, r2655;
}
{
cvt.rn.f16.f64 rs399, fd859;
}
mov.b32 r2663, {rs399, rs399};
{
mul.f16x2 r2661, r73, r2663;
}
{
add.f16x2 r2664, r2640, r2661;
}
{
cvt.rn.f16.f64 rs400, fd572;
}
mov.b32 r2669, {rs400, rs400};
{
mul.f16x2 r2667, r82, r2669;
}
{
add.f16x2 r2670, r2646, r2667;
}
{
cvt.rn.f16.f64 rs401, fd859;
}
mov.b32 r2675, {rs401, rs401};
{
mul.f16x2 r2673, r76, r2675;
}
{
add.f16x2 r2676, r2652, r2673;
}
{
cvt.rn.f16.f64 rs402, fd572;
}
mov.b32 r2681, {rs402, rs402};
{
mul.f16x2 r2679, r79, r2681;
}
{
add.f16x2 r2682, r2658, r2679;
}
{
cvt.rn.f16.f64 rs403, fd887;
}
mov.b32 r2687, {rs403, rs403};
{
mul.f16x2 r2685, r85, r2687;
}
{
add.f16x2 r2688, r2664, r2685;
}
{
cvt.rn.f16.f64 rs404, fd888;
}
mov.b32 r2693, {rs404, rs404};
{
mul.f16x2 r2691, r94, r2693;
}
{
add.f16x2 r2694, r2670, r2691;
}
{
cvt.rn.f16.f64 rs405, fd887;
}
mov.b32 r2699, {rs405, rs405};
{
mul.f16x2 r2697, r88, r2699;
}
{
add.f16x2 r2700, r2676, r2697;
}
{
cvt.rn.f16.f64 rs406, fd888;
}
mov.b32 r2705, {rs406, rs406};
{
mul.f16x2 r2703, r91, r2705;
}
{
add.f16x2 r2706, r2682, r2703;
}
{
cvt.rn.f16.f64 rs407, fd847;
}
mov.b32 r2711, {rs407, rs407};
{
mul.f16x2 r2709, r97, r2711;
}
{
add.f16x2 r2712, r2688, r2709;
}
{
cvt.rn.f16.f64 rs408, fd768;
}
mov.b32 r2717, {rs408, rs408};
{
mul.f16x2 r2715, r106, r2717;
}
{
add.f16x2 r2718, r2694, r2715;
}
{
cvt.rn.f16.f64 rs409, fd847;
}
mov.b32 r2723, {rs409, rs409};
{
mul.f16x2 r2721, r100, r2723;
}
{
add.f16x2 r2724, r2700, r2721;
}
{
cvt.rn.f16.f64 rs410, fd768;
}
mov.b32 r2729, {rs410, rs410};
{
mul.f16x2 r2727, r103, r2729;
}
{
add.f16x2 r2730, r2706, r2727;
}
{
cvt.rn.f16.f64 rs411, fd899;
}
mov.b32 r2735, {rs411, rs411};
{
mul.f16x2 r2733, r109, r2735;
}
{
add.f16x2 r2736, r2712, r2733;
}
{
cvt.rn.f16.f64 rs412, fd900;
}
mov.b32 r2741, {rs412, rs412};
{
mul.f16x2 r2739, r118, r2741;
}
{
add.f16x2 r2742, r2718, r2739;
}
{
cvt.rn.f16.f64 rs413, fd899;
}
mov.b32 r2747, {rs413, rs413};
{
mul.f16x2 r2745, r112, r2747;
}
{
add.f16x2 r2748, r2724, r2745;
}
{
cvt.rn.f16.f64 rs414, fd900;
}
mov.b32 r2753, {rs414, rs414};
{
mul.f16x2 r2751, r115, r2753;
}
{
add.f16x2 r2754, r2730, r2751;
}
{
cvt.rn.f16.f64 rs415, fd843;
}
mov.b32 r2759, {rs415, rs415};
{
mul.f16x2 r2757, r121, r2759;
}
{
add.f16x2 r2760, r2736, r2757;
}
{
cvt.rn.f16.f64 rs416, fd844;
}
mov.b32 r2765, {rs416, rs416};
{
mul.f16x2 r2763, r130, r2765;
}
{
add.f16x2 r2766, r2742, r2763;
}
{
cvt.rn.f16.f64 rs417, fd843;
}
mov.b32 r2771, {rs417, rs417};
{
mul.f16x2 r2769, r124, r2771;
}
{
add.f16x2 r2772, r2748, r2769;
}
{
cvt.rn.f16.f64 rs418, fd844;
}
mov.b32 r2777, {rs418, rs418};
{
mul.f16x2 r2775, r127, r2777;
}
{
add.f16x2 r2778, r2754, r2775;
}
{
cvt.rn.f16.f64 rs419, fd891;
}
mov.b32 r2783, {rs419, rs419};
{
mul.f16x2 r2781, r133, r2783;
}
{
add.f16x2 r2784, r2760, r2781;
}
{
cvt.rn.f16.f64 rs420, fd804;
}
mov.b32 r2789, {rs420, rs420};
{
mul.f16x2 r2787, r142, r2789;
}
{
add.f16x2 r2790, r2766, r2787;
}
{
cvt.rn.f16.f64 rs421, fd891;
}
mov.b32 r2795, {rs421, rs421};
{
mul.f16x2 r2793, r136, r2795;
}
{
add.f16x2 r2796, r2772, r2793;
}
{
cvt.rn.f16.f64 rs422, fd804;
}
mov.b32 r2801, {rs422, rs422};
{
mul.f16x2 r2799, r139, r2801;
}
{
add.f16x2 r2802, r2778, r2799;
}
{
cvt.rn.f16.f64 rs423, fd855;
}
mov.b32 r2807, {rs423, rs423};
{
mul.f16x2 r2805, r145, r2807;
}
{
add.f16x2 r2808, r2784, r2805;
}
{
cvt.rn.f16.f64 rs424, fd856;
}
mov.b32 r2813, {rs424, rs424};
{
mul.f16x2 r2811, r154, r2813;
}
{
add.f16x2 r2814, r2790, r2811;
}
{
cvt.rn.f16.f64 rs425, fd855;
}
mov.b32 r2819, {rs425, rs425};
{
mul.f16x2 r2817, r148, r2819;
}
{
add.f16x2 r2820, r2796, r2817;
}
{
cvt.rn.f16.f64 rs426, fd856;
}
mov.b32 r2825, {rs426, rs426};
{
mul.f16x2 r2823, r151, r2825;
}
{
add.f16x2 r2826, r2802, r2823;
}
{
cvt.rn.f16.f64 rs427, fd879;
}
mov.b32 r2831, {rs427, rs427};
{
mul.f16x2 r2829, r157, r2831;
}
{
add.f16x2 r2832, r2808, r2829;
}
{
cvt.rn.f16.f64 rs428, fd808;
}
mov.b32 r2837, {rs428, rs428};
{
mul.f16x2 r2835, r166, r2837;
}
{
add.f16x2 r2838, r2814, r2835;
}
{
cvt.rn.f16.f64 rs429, fd879;
}
mov.b32 r2843, {rs429, rs429};
{
mul.f16x2 r2841, r160, r2843;
}
{
add.f16x2 r2844, r2820, r2841;
}
{
cvt.rn.f16.f64 rs430, fd808;
}
mov.b32 r2849, {rs430, rs430};
{
mul.f16x2 r2847, r163, r2849;
}
{
add.f16x2 r2850, r2826, r2847;
}
{
cvt.rn.f16.f64 rs431, fd867;
}
mov.b32 r2855, {rs431, rs431};
{
mul.f16x2 r2853, r169, r2855;
}
{
add.f16x2 r2856, r2832, r2853;
}
{
cvt.rn.f16.f64 rs432, fd868;
}
mov.b32 r2861, {rs432, rs432};
{
mul.f16x2 r2859, r178, r2861;
}
{
add.f16x2 r2862, r2838, r2859;
}
{
cvt.rn.f16.f64 rs433, fd867;
}
mov.b32 r2867, {rs433, rs433};
{
mul.f16x2 r2865, r172, r2867;
}
{
add.f16x2 r2868, r2844, r2865;
}
{
cvt.rn.f16.f64 rs434, fd868;
}
mov.b32 r2873, {rs434, rs434};
{
mul.f16x2 r2871, r175, r2873;
}
{
add.f16x2 r2874, r2850, r2871;
}
{
sub.f16x2 %14, r2856, r2862;
}
{
add.f16x2 %15, r2868, r2874;
}
{
add.f16x2 %48, r2856, r2862;
}
{
sub.f16x2 %49, r2868, r2874;
}
cvt.rn.f16.s32 rs435, r5508;
mov.b32 r2901, {rs435, rs435};
cvt.rn.f16.s32 rs436, r5508;
mov.b32 r2913, {rs436, rs436};
{
cvt.rn.f16.f64 rs437, fd899;
}
mov.b32 r2893, {rs437, rs437};
{
mul.f16x2 r2891, r1, r2893;
}
{
add.f16x2 r2894, %62, r2891;
}
{
cvt.rn.f16.f64 rs438, fd900;
}
mov.b32 r2899, {rs438, rs438};
{
mul.f16x2 r2897, r10, r2899;
}
{
add.f16x2 r2900, r2901, r2897;
}
{
cvt.rn.f16.f64 rs439, fd899;
}
mov.b32 r2905, {rs439, rs439};
{
mul.f16x2 r2903, r4, r2905;
}
{
add.f16x2 r2906, %63, r2903;
}
{
cvt.rn.f16.f64 rs440, fd900;
}
mov.b32 r2911, {rs440, rs440};
{
mul.f16x2 r2909, r7, r2911;
}
{
add.f16x2 r2912, r2913, r2909;
}
{
cvt.rn.f16.f64 rs441, fd843;
}
mov.b32 r2917, {rs441, rs441};
{
mul.f16x2 r2915, r13, r2917;
}
{
add.f16x2 r2918, r2894, r2915;
}
{
cvt.rn.f16.f64 rs442, fd820;
}
mov.b32 r2923, {rs442, rs442};
{
mul.f16x2 r2921, r22, r2923;
}
{
add.f16x2 r2924, r2900, r2921;
}
{
cvt.rn.f16.f64 rs443, fd843;
}
mov.b32 r2929, {rs443, rs443};
{
mul.f16x2 r2927, r16, r2929;
}
{
add.f16x2 r2930, r2906, r2927;
}
{
cvt.rn.f16.f64 rs444, fd820;
}
mov.b32 r2935, {rs444, rs444};
{
mul.f16x2 r2933, r19, r2935;
}
{
add.f16x2 r2936, r2912, r2933;
}
{
cvt.rn.f16.f64 rs445, fd895;
}
mov.b32 r2941, {rs445, rs445};
{
mul.f16x2 r2939, r25, r2941;
}
{
add.f16x2 r2942, r2918, r2939;
}
{
cvt.rn.f16.f64 rs446, fd896;
}
mov.b32 r2947, {rs446, rs446};
{
mul.f16x2 r2945, r34, r2947;
}
{
add.f16x2 r2948, r2924, r2945;
}
{
cvt.rn.f16.f64 rs447, fd895;
}
mov.b32 r2953, {rs447, rs447};
{
mul.f16x2 r2951, r28, r2953;
}
{
add.f16x2 r2954, r2930, r2951;
}
{
cvt.rn.f16.f64 rs448, fd896;
}
mov.b32 r2959, {rs448, rs448};
{
mul.f16x2 r2957, r31, r2959;
}
{
add.f16x2 r2960, r2936, r2957;
}
{
cvt.rn.f16.f64 rs449, fd847;
}
mov.b32 r2965, {rs449, rs449};
{
mul.f16x2 r2963, r37, r2965;
}
{
add.f16x2 r2966, r2942, r2963;
}
{
cvt.rn.f16.f64 rs450, fd768;
}
mov.b32 r2971, {rs450, rs450};
{
mul.f16x2 r2969, r46, r2971;
}
{
add.f16x2 r2972, r2948, r2969;
}
{
cvt.rn.f16.f64 rs451, fd847;
}
mov.b32 r2977, {rs451, rs451};
{
mul.f16x2 r2975, r40, r2977;
}
{
add.f16x2 r2978, r2954, r2975;
}
{
cvt.rn.f16.f64 rs452, fd768;
}
mov.b32 r2983, {rs452, rs452};
{
mul.f16x2 r2981, r43, r2983;
}
{
add.f16x2 r2984, r2960, r2981;
}
{
cvt.rn.f16.f64 rs453, fd891;
}
mov.b32 r2989, {rs453, rs453};
{
mul.f16x2 r2987, r49, r2989;
}
{
add.f16x2 r2990, r2966, r2987;
}
{
cvt.rn.f16.f64 rs454, fd892;
}
mov.b32 r2995, {rs454, rs454};
{
mul.f16x2 r2993, r58, r2995;
}
{
add.f16x2 r2996, r2972, r2993;
}
{
cvt.rn.f16.f64 rs455, fd891;
}
mov.b32 r3001, {rs455, rs455};
{
mul.f16x2 r2999, r52, r3001;
}
{
add.f16x2 r3002, r2978, r2999;
}
{
cvt.rn.f16.f64 rs456, fd892;
}
mov.b32 r3007, {rs456, rs456};
{
mul.f16x2 r3005, r55, r3007;
}
{
add.f16x2 r3008, r2984, r3005;
}
{
cvt.rn.f16.f64 rs457, fd851;
}
mov.b32 r3013, {rs457, rs457};
{
mul.f16x2 r3011, r61, r3013;
}
{
add.f16x2 r3014, r2990, r3011;
}
{
cvt.rn.f16.f64 rs458, fd676;
}
mov.b32 r3019, {rs458, rs458};
{
mul.f16x2 r3017, r70, r3019;
}
{
add.f16x2 r3020, r2996, r3017;
}
{
cvt.rn.f16.f64 rs459, fd851;
}
mov.b32 r3025, {rs459, rs459};
{
mul.f16x2 r3023, r64, r3025;
}
{
add.f16x2 r3026, r3002, r3023;
}
{
cvt.rn.f16.f64 rs460, fd676;
}
mov.b32 r3031, {rs460, rs460};
{
mul.f16x2 r3029, r67, r3031;
}
{
add.f16x2 r3032, r3008, r3029;
}
{
cvt.rn.f16.f64 rs461, fd887;
}
mov.b32 r3037, {rs461, rs461};
{
mul.f16x2 r3035, r73, r3037;
}
{
add.f16x2 r3038, r3014, r3035;
}
{
cvt.rn.f16.f64 rs462, fd888;
}
mov.b32 r3043, {rs462, rs462};
{
mul.f16x2 r3041, r82, r3043;
}
{
add.f16x2 r3044, r3020, r3041;
}
{
cvt.rn.f16.f64 rs463, fd887;
}
mov.b32 r3049, {rs463, rs463};
{
mul.f16x2 r3047, r76, r3049;
}
{
add.f16x2 r3050, r3026, r3047;
}
{
cvt.rn.f16.f64 rs464, fd888;
}
mov.b32 r3055, {rs464, rs464};
{
mul.f16x2 r3053, r79, r3055;
}
{
add.f16x2 r3056, r3032, r3053;
}
{
cvt.rn.f16.f64 rs465, fd855;
}
mov.b32 r3061, {rs465, rs465};
{
mul.f16x2 r3059, r85, r3061;
}
{
add.f16x2 r3062, r3038, r3059;
}
{
cvt.rn.f16.f64 rs466, fd816;
}
mov.b32 r3067, {rs466, rs466};
{
mul.f16x2 r3065, r94, r3067;
}
{
add.f16x2 r3068, r3044, r3065;
}
{
cvt.rn.f16.f64 rs467, fd855;
}
mov.b32 r3073, {rs467, rs467};
{
mul.f16x2 r3071, r88, r3073;
}
{
add.f16x2 r3074, r3050, r3071;
}
{
cvt.rn.f16.f64 rs468, fd816;
}
mov.b32 r3079, {rs468, rs468};
{
mul.f16x2 r3077, r91, r3079;
}
{
add.f16x2 r3080, r3056, r3077;
}
{
cvt.rn.f16.f64 rs469, fd883;
}
mov.b32 r3085, {rs469, rs469};
{
mul.f16x2 r3083, r97, r3085;
}
{
add.f16x2 r3086, r3062, r3083;
}
{
cvt.rn.f16.f64 rs470, fd884;
}
mov.b32 r3091, {rs470, rs470};
{
mul.f16x2 r3089, r106, r3091;
}
{
add.f16x2 r3092, r3068, r3089;
}
{
cvt.rn.f16.f64 rs471, fd883;
}
mov.b32 r3097, {rs471, rs471};
{
mul.f16x2 r3095, r100, r3097;
}
{
add.f16x2 r3098, r3074, r3095;
}
{
cvt.rn.f16.f64 rs472, fd884;
}
mov.b32 r3103, {rs472, rs472};
{
mul.f16x2 r3101, r103, r3103;
}
{
add.f16x2 r3104, r3080, r3101;
}
{
cvt.rn.f16.f64 rs473, fd859;
}
mov.b32 r3109, {rs473, rs473};
{
mul.f16x2 r3107, r109, r3109;
}
{
add.f16x2 r3110, r3086, r3107;
}
{
cvt.rn.f16.f64 rs474, fd572;
}
mov.b32 r3115, {rs474, rs474};
{
mul.f16x2 r3113, r118, r3115;
}
{
add.f16x2 r3116, r3092, r3113;
}
{
cvt.rn.f16.f64 rs475, fd859;
}
mov.b32 r3121, {rs475, rs475};
{
mul.f16x2 r3119, r112, r3121;
}
{
add.f16x2 r3122, r3098, r3119;
}
{
cvt.rn.f16.f64 rs476, fd572;
}
mov.b32 r3127, {rs476, rs476};
{
mul.f16x2 r3125, r115, r3127;
}
{
add.f16x2 r3128, r3104, r3125;
}
{
cvt.rn.f16.f64 rs477, fd879;
}
mov.b32 r3133, {rs477, rs477};
{
mul.f16x2 r3131, r121, r3133;
}
{
add.f16x2 r3134, r3110, r3131;
}
{
cvt.rn.f16.f64 rs478, fd880;
}
mov.b32 r3139, {rs478, rs478};
{
mul.f16x2 r3137, r130, r3139;
}
{
add.f16x2 r3140, r3116, r3137;
}
{
cvt.rn.f16.f64 rs479, fd879;
}
mov.b32 r3145, {rs479, rs479};
{
mul.f16x2 r3143, r124, r3145;
}
{
add.f16x2 r3146, r3122, r3143;
}
{
cvt.rn.f16.f64 rs480, fd880;
}
mov.b32 r3151, {rs480, rs480};
{
mul.f16x2 r3149, r127, r3151;
}
{
add.f16x2 r3152, r3128, r3149;
}
{
cvt.rn.f16.f64 rs481, fd863;
}
mov.b32 r3157, {rs481, rs481};
{
mul.f16x2 r3155, r133, r3157;
}
{
add.f16x2 r3158, r3134, r3155;
}
{
cvt.rn.f16.f64 rs482, fd740;
}
mov.b32 r3163, {rs482, rs482};
{
mul.f16x2 r3161, r142, r3163;
}
{
add.f16x2 r3164, r3140, r3161;
}
{
cvt.rn.f16.f64 rs483, fd863;
}
mov.b32 r3169, {rs483, rs483};
{
mul.f16x2 r3167, r136, r3169;
}
{
add.f16x2 r3170, r3146, r3167;
}
{
cvt.rn.f16.f64 rs484, fd740;
}
mov.b32 r3175, {rs484, rs484};
{
mul.f16x2 r3173, r139, r3175;
}
{
add.f16x2 r3176, r3152, r3173;
}
{
cvt.rn.f16.f64 rs485, fd875;
}
mov.b32 r3181, {rs485, rs485};
{
mul.f16x2 r3179, r145, r3181;
}
{
add.f16x2 r3182, r3158, r3179;
}
{
cvt.rn.f16.f64 rs486, fd876;
}
mov.b32 r3187, {rs486, rs486};
{
mul.f16x2 r3185, r154, r3187;
}
{
add.f16x2 r3188, r3164, r3185;
}
{
cvt.rn.f16.f64 rs487, fd875;
}
mov.b32 r3193, {rs487, rs487};
{
mul.f16x2 r3191, r148, r3193;
}
{
add.f16x2 r3194, r3170, r3191;
}
{
cvt.rn.f16.f64 rs488, fd876;
}
mov.b32 r3199, {rs488, rs488};
{
mul.f16x2 r3197, r151, r3199;
}
{
add.f16x2 r3200, r3176, r3197;
}
{
cvt.rn.f16.f64 rs489, fd867;
}
mov.b32 r3205, {rs489, rs489};
{
mul.f16x2 r3203, r157, r3205;
}
{
add.f16x2 r3206, r3182, r3203;
}
{
cvt.rn.f16.f64 rs490, fd812;
}
mov.b32 r3211, {rs490, rs490};
{
mul.f16x2 r3209, r166, r3211;
}
{
add.f16x2 r3212, r3188, r3209;
}
{
cvt.rn.f16.f64 rs491, fd867;
}
mov.b32 r3217, {rs491, rs491};
{
mul.f16x2 r3215, r160, r3217;
}
{
add.f16x2 r3218, r3194, r3215;
}
{
cvt.rn.f16.f64 rs492, fd812;
}
mov.b32 r3223, {rs492, rs492};
{
mul.f16x2 r3221, r163, r3223;
}
{
add.f16x2 r3224, r3200, r3221;
}
{
cvt.rn.f16.f64 rs493, fd871;
}
mov.b32 r3229, {rs493, rs493};
{
mul.f16x2 r3227, r169, r3229;
}
{
add.f16x2 r3230, r3206, r3227;
}
{
cvt.rn.f16.f64 rs494, fd872;
}
mov.b32 r3235, {rs494, rs494};
{
mul.f16x2 r3233, r178, r3235;
}
{
add.f16x2 r3236, r3212, r3233;
}
{
cvt.rn.f16.f64 rs495, fd871;
}
mov.b32 r3241, {rs495, rs495};
{
mul.f16x2 r3239, r172, r3241;
}
{
add.f16x2 r3242, r3218, r3239;
}
{
cvt.rn.f16.f64 rs496, fd872;
}
mov.b32 r3247, {rs496, rs496};
{
mul.f16x2 r3245, r175, r3247;
}
{
add.f16x2 r3248, r3224, r3245;
}
{
sub.f16x2 %16, r3230, r3236;
}
{
add.f16x2 %17, r3242, r3248;
}
{
add.f16x2 %46, r3230, r3236;
}
{
sub.f16x2 %47, r3242, r3248;
}
cvt.rn.f16.s32 rs497, r5508;
mov.b32 r3275, {rs497, rs497};
cvt.rn.f16.s32 rs498, r5508;
mov.b32 r3287, {rs498, rs498};
{
cvt.rn.f16.f64 rs499, fd891;
}
mov.b32 r3267, {rs499, rs499};
{
mul.f16x2 r3265, r1, r3267;
}
{
add.f16x2 r3268, %62, r3265;
}
{
cvt.rn.f16.f64 rs500, fd892;
}
mov.b32 r3273, {rs500, rs500};
{
mul.f16x2 r3271, r10, r3273;
}
{
add.f16x2 r3274, r3275, r3271;
}
{
cvt.rn.f16.f64 rs501, fd891;
}
mov.b32 r3279, {rs501, rs501};
{
mul.f16x2 r3277, r4, r3279;
}
{
add.f16x2 r3280, %63, r3277;
}
{
cvt.rn.f16.f64 rs502, fd892;
}
mov.b32 r3285, {rs502, rs502};
{
mul.f16x2 r3283, r7, r3285;
}
{
add.f16x2 r3286, r3287, r3283;
}
{
cvt.rn.f16.f64 rs503, fd859;
}
mov.b32 r3291, {rs503, rs503};
{
mul.f16x2 r3289, r13, r3291;
}
{
add.f16x2 r3292, r3268, r3289;
}
{
cvt.rn.f16.f64 rs504, fd572;
}
mov.b32 r3297, {rs504, rs504};
{
mul.f16x2 r3295, r22, r3297;
}
{
add.f16x2 r3298, r3274, r3295;
}
{
cvt.rn.f16.f64 rs505, fd859;
}
mov.b32 r3303, {rs505, rs505};
{
mul.f16x2 r3301, r16, r3303;
}
{
add.f16x2 r3304, r3280, r3301;
}
{
cvt.rn.f16.f64 rs506, fd572;
}
mov.b32 r3309, {rs506, rs506};
{
mul.f16x2 r3307, r19, r3309;
}
{
add.f16x2 r3310, r3286, r3307;
}
{
cvt.rn.f16.f64 rs507, fd871;
}
mov.b32 r3315, {rs507, rs507};
{
mul.f16x2 r3313, r25, r3315;
}
{
add.f16x2 r3316, r3292, r3313;
}
{
cvt.rn.f16.f64 rs508, fd872;
}
mov.b32 r3321, {rs508, rs508};
{
mul.f16x2 r3319, r34, r3321;
}
{
add.f16x2 r3322, r3298, r3319;
}
{
cvt.rn.f16.f64 rs509, fd871;
}
mov.b32 r3327, {rs509, rs509};
{
mul.f16x2 r3325, r28, r3327;
}
{
add.f16x2 r3328, r3304, r3325;
}
{
cvt.rn.f16.f64 rs510, fd872;
}
mov.b32 r3333, {rs510, rs510};
{
mul.f16x2 r3331, r31, r3333;
}
{
add.f16x2 r3334, r3310, r3331;
}
{
cvt.rn.f16.f64 rs511, fd879;
}
mov.b32 r3339, {rs511, rs511};
{
mul.f16x2 r3337, r37, r3339;
}
{
add.f16x2 r3340, r3316, r3337;
}
{
cvt.rn.f16.f64 rs512, fd808;
}
mov.b32 r3345, {rs512, rs512};
{
mul.f16x2 r3343, r46, r3345;
}
{
add.f16x2 r3346, r3322, r3343;
}
{
cvt.rn.f16.f64 rs513, fd879;
}
mov.b32 r3351, {rs513, rs513};
{
mul.f16x2 r3349, r40, r3351;
}
{
add.f16x2 r3352, r3328, r3349;
}
{
cvt.rn.f16.f64 rs514, fd808;
}
mov.b32 r3357, {rs514, rs514};
{
mul.f16x2 r3355, r43, r3357;
}
{
add.f16x2 r3358, r3334, r3355;
}
{
cvt.rn.f16.f64 rs515, fd851;
}
mov.b32 r3363, {rs515, rs515};
{
mul.f16x2 r3361, r49, r3363;
}
{
add.f16x2 r3364, r3340, r3361;
}
{
cvt.rn.f16.f64 rs516, fd852;
}
mov.b32 r3369, {rs516, rs516};
{
mul.f16x2 r3367, r58, r3369;
}
{
add.f16x2 r3370, r3346, r3367;
}
{
cvt.rn.f16.f64 rs517, fd851;
}
mov.b32 r3375, {rs517, rs517};
{
mul.f16x2 r3373, r52, r3375;
}
{
add.f16x2 r3376, r3352, r3373;
}
{
cvt.rn.f16.f64 rs518, fd852;
}
mov.b32 r3381, {rs518, rs518};
{
mul.f16x2 r3379, r55, r3381;
}
{
add.f16x2 r3382, r3358, r3379;
}
{
cvt.rn.f16.f64 rs519, fd899;
}
mov.b32 r3387, {rs519, rs519};
{
mul.f16x2 r3385, r61, r3387;
}
{
add.f16x2 r3388, r3364, r3385;
}
{
cvt.rn.f16.f64 rs520, fd504;
}
mov.b32 r3393, {rs520, rs520};
{
mul.f16x2 r3391, r70, r3393;
}
{
add.f16x2 r3394, r3370, r3391;
}
{
cvt.rn.f16.f64 rs521, fd899;
}
mov.b32 r3399, {rs521, rs521};
{
mul.f16x2 r3397, r64, r3399;
}
{
add.f16x2 r3400, r3376, r3397;
}
{
cvt.rn.f16.f64 rs522, fd504;
}
mov.b32 r3405, {rs522, rs522};
{
mul.f16x2 r3403, r67, r3405;
}
{
add.f16x2 r3406, r3382, r3403;
}
{
cvt.rn.f16.f64 rs523, fd847;
}
mov.b32 r3411, {rs523, rs523};
{
mul.f16x2 r3409, r73, r3411;
}
{
add.f16x2 r3412, r3388, r3409;
}
{
cvt.rn.f16.f64 rs524, fd768;
}
mov.b32 r3417, {rs524, rs524};
{
mul.f16x2 r3415, r82, r3417;
}
{
add.f16x2 r3418, r3394, r3415;
}
{
cvt.rn.f16.f64 rs525, fd847;
}
mov.b32 r3423, {rs525, rs525};
{
mul.f16x2 r3421, r76, r3423;
}
{
add.f16x2 r3424, r3400, r3421;
}
{
cvt.rn.f16.f64 rs526, fd768;
}
mov.b32 r3429, {rs526, rs526};
{
mul.f16x2 r3427, r79, r3429;
}
{
add.f16x2 r3430, r3406, r3427;
}
{
cvt.rn.f16.f64 rs527, fd883;
}
mov.b32 r3435, {rs527, rs527};
{
mul.f16x2 r3433, r85, r3435;
}
{
add.f16x2 r3436, r3412, r3433;
}
{
cvt.rn.f16.f64 rs528, fd884;
}
mov.b32 r3441, {rs528, rs528};
{
mul.f16x2 r3439, r94, r3441;
}
{
add.f16x2 r3442, r3418, r3439;
}
{
cvt.rn.f16.f64 rs529, fd883;
}
mov.b32 r3447, {rs529, rs529};
{
mul.f16x2 r3445, r88, r3447;
}
{
add.f16x2 r3448, r3424, r3445;
}
{
cvt.rn.f16.f64 rs530, fd884;
}
mov.b32 r3453, {rs530, rs530};
{
mul.f16x2 r3451, r91, r3453;
}
{
add.f16x2 r3454, r3430, r3451;
}
{
cvt.rn.f16.f64 rs531, fd867;
}
mov.b32 r3459, {rs531, rs531};
{
mul.f16x2 r3457, r97, r3459;
}
{
add.f16x2 r3460, r3436, r3457;
}
{
cvt.rn.f16.f64 rs532, fd812;
}
mov.b32 r3465, {rs532, rs532};
{
mul.f16x2 r3463, r106, r3465;
}
{
add.f16x2 r3466, r3442, r3463;
}
{
cvt.rn.f16.f64 rs533, fd867;
}
mov.b32 r3471, {rs533, rs533};
{
mul.f16x2 r3469, r100, r3471;
}
{
add.f16x2 r3472, r3448, r3469;
}
{
cvt.rn.f16.f64 rs534, fd812;
}
mov.b32 r3477, {rs534, rs534};
{
mul.f16x2 r3475, r103, r3477;
}
{
add.f16x2 r3478, r3454, r3475;
}
{
cvt.rn.f16.f64 rs535, fd863;
}
mov.b32 r3483, {rs535, rs535};
{
mul.f16x2 r3481, r109, r3483;
}
{
add.f16x2 r3484, r3460, r3481;
}
{
cvt.rn.f16.f64 rs536, fd864;
}
mov.b32 r3489, {rs536, rs536};
{
mul.f16x2 r3487, r118, r3489;
}
{
add.f16x2 r3490, r3466, r3487;
}
{
cvt.rn.f16.f64 rs537, fd863;
}
mov.b32 r3495, {rs537, rs537};
{
mul.f16x2 r3493, r112, r3495;
}
{
add.f16x2 r3496, r3472, r3493;
}
{
cvt.rn.f16.f64 rs538, fd864;
}
mov.b32 r3501, {rs538, rs538};
{
mul.f16x2 r3499, r115, r3501;
}
{
add.f16x2 r3502, r3478, r3499;
}
{
cvt.rn.f16.f64 rs539, fd887;
}
mov.b32 r3507, {rs539, rs539};
{
mul.f16x2 r3505, r121, r3507;
}
{
add.f16x2 r3508, r3484, r3505;
}
{
cvt.rn.f16.f64 rs540, fd760;
}
mov.b32 r3513, {rs540, rs540};
{
mul.f16x2 r3511, r130, r3513;
}
{
add.f16x2 r3514, r3490, r3511;
}
{
cvt.rn.f16.f64 rs541, fd887;
}
mov.b32 r3519, {rs541, rs541};
{
mul.f16x2 r3517, r124, r3519;
}
{
add.f16x2 r3520, r3496, r3517;
}
{
cvt.rn.f16.f64 rs542, fd760;
}
mov.b32 r3525, {rs542, rs542};
{
mul.f16x2 r3523, r127, r3525;
}
{
add.f16x2 r3526, r3502, r3523;
}
{
cvt.rn.f16.f64 rs543, fd843;
}
mov.b32 r3531, {rs543, rs543};
{
mul.f16x2 r3529, r133, r3531;
}
{
add.f16x2 r3532, r3508, r3529;
}
{
cvt.rn.f16.f64 rs544, fd844;
}
mov.b32 r3537, {rs544, rs544};
{
mul.f16x2 r3535, r142, r3537;
}
{
add.f16x2 r3538, r3514, r3535;
}
{
cvt.rn.f16.f64 rs545, fd843;
}
mov.b32 r3543, {rs545, rs545};
{
mul.f16x2 r3541, r136, r3543;
}
{
add.f16x2 r3544, r3520, r3541;
}
{
cvt.rn.f16.f64 rs546, fd844;
}
mov.b32 r3549, {rs546, rs546};
{
mul.f16x2 r3547, r139, r3549;
}
{
add.f16x2 r3550, r3526, r3547;
}
{
cvt.rn.f16.f64 rs547, fd895;
}
mov.b32 r3555, {rs547, rs547};
{
mul.f16x2 r3553, r145, r3555;
}
{
add.f16x2 r3556, r3532, r3553;
}
{
cvt.rn.f16.f64 rs548, fd896;
}
mov.b32 r3561, {rs548, rs548};
{
mul.f16x2 r3559, r154, r3561;
}
{
add.f16x2 r3562, r3538, r3559;
}
{
cvt.rn.f16.f64 rs549, fd895;
}
mov.b32 r3567, {rs549, rs549};
{
mul.f16x2 r3565, r148, r3567;
}
{
add.f16x2 r3568, r3544, r3565;
}
{
cvt.rn.f16.f64 rs550, fd896;
}
mov.b32 r3573, {rs550, rs550};
{
mul.f16x2 r3571, r151, r3573;
}
{
add.f16x2 r3574, r3550, r3571;
}
{
cvt.rn.f16.f64 rs551, fd855;
}
mov.b32 r3579, {rs551, rs551};
{
mul.f16x2 r3577, r157, r3579;
}
{
add.f16x2 r3580, r3556, r3577;
}
{
cvt.rn.f16.f64 rs552, fd816;
}
mov.b32 r3585, {rs552, rs552};
{
mul.f16x2 r3583, r166, r3585;
}
{
add.f16x2 r3586, r3562, r3583;
}
{
cvt.rn.f16.f64 rs553, fd855;
}
mov.b32 r3591, {rs553, rs553};
{
mul.f16x2 r3589, r160, r3591;
}
{
add.f16x2 r3592, r3568, r3589;
}
{
cvt.rn.f16.f64 rs554, fd816;
}
mov.b32 r3597, {rs554, rs554};
{
mul.f16x2 r3595, r163, r3597;
}
{
add.f16x2 r3598, r3574, r3595;
}
{
cvt.rn.f16.f64 rs555, fd875;
}
mov.b32 r3603, {rs555, rs555};
{
mul.f16x2 r3601, r169, r3603;
}
{
add.f16x2 r3604, r3580, r3601;
}
{
cvt.rn.f16.f64 rs556, fd876;
}
mov.b32 r3609, {rs556, rs556};
{
mul.f16x2 r3607, r178, r3609;
}
{
add.f16x2 r3610, r3586, r3607;
}
{
cvt.rn.f16.f64 rs557, fd875;
}
mov.b32 r3615, {rs557, rs557};
{
mul.f16x2 r3613, r172, r3615;
}
{
add.f16x2 r3616, r3592, r3613;
}
{
cvt.rn.f16.f64 rs558, fd876;
}
mov.b32 r3621, {rs558, rs558};
{
mul.f16x2 r3619, r175, r3621;
}
{
add.f16x2 r3622, r3598, r3619;
}
{
sub.f16x2 %18, r3604, r3610;
}
{
add.f16x2 %19, r3616, r3622;
}
{
add.f16x2 %44, r3604, r3610;
}
{
sub.f16x2 %45, r3616, r3622;
}
cvt.rn.f16.s32 rs559, r5508;
mov.b32 r3649, {rs559, rs559};
cvt.rn.f16.s32 rs560, r5508;
mov.b32 r3661, {rs560, rs560};
{
cvt.rn.f16.f64 rs561, fd883;
}
mov.b32 r3641, {rs561, rs561};
{
mul.f16x2 r3639, r1, r3641;
}
{
add.f16x2 r3642, %62, r3639;
}
{
cvt.rn.f16.f64 rs562, fd884;
}
mov.b32 r3647, {rs562, rs562};
{
mul.f16x2 r3645, r10, r3647;
}
{
add.f16x2 r3648, r3649, r3645;
}
{
cvt.rn.f16.f64 rs563, fd883;
}
mov.b32 r3653, {rs563, rs563};
{
mul.f16x2 r3651, r4, r3653;
}
{
add.f16x2 r3654, %63, r3651;
}
{
cvt.rn.f16.f64 rs564, fd884;
}
mov.b32 r3659, {rs564, rs564};
{
mul.f16x2 r3657, r7, r3659;
}
{
add.f16x2 r3660, r3661, r3657;
}
{
cvt.rn.f16.f64 rs565, fd875;
}
mov.b32 r3665, {rs565, rs565};
{
mul.f16x2 r3663, r13, r3665;
}
{
add.f16x2 r3666, r3642, r3663;
}
{
cvt.rn.f16.f64 rs566, fd708;
}
mov.b32 r3671, {rs566, rs566};
{
mul.f16x2 r3669, r22, r3671;
}
{
add.f16x2 r3672, r3648, r3669;
}
{
cvt.rn.f16.f64 rs567, fd875;
}
mov.b32 r3677, {rs567, rs567};
{
mul.f16x2 r3675, r16, r3677;
}
{
add.f16x2 r3678, r3654, r3675;
}
{
cvt.rn.f16.f64 rs568, fd708;
}
mov.b32 r3683, {rs568, rs568};
{
mul.f16x2 r3681, r19, r3683;
}
{
add.f16x2 r3684, r3660, r3681;
}
{
cvt.rn.f16.f64 rs569, fd847;
}
mov.b32 r3689, {rs569, rs569};
{
mul.f16x2 r3687, r25, r3689;
}
{
add.f16x2 r3690, r3666, r3687;
}
{
cvt.rn.f16.f64 rs570, fd848;
}
mov.b32 r3695, {rs570, rs570};
{
mul.f16x2 r3693, r34, r3695;
}
{
add.f16x2 r3696, r3672, r3693;
}
{
cvt.rn.f16.f64 rs571, fd847;
}
mov.b32 r3701, {rs571, rs571};
{
mul.f16x2 r3699, r28, r3701;
}
{
add.f16x2 r3702, r3678, r3699;
}
{
cvt.rn.f16.f64 rs572, fd848;
}
mov.b32 r3707, {rs572, rs572};
{
mul.f16x2 r3705, r31, r3707;
}
{
add.f16x2 r3708, r3684, r3705;
}
{
cvt.rn.f16.f64 rs573, fd891;
}
mov.b32 r3713, {rs573, rs573};
{
mul.f16x2 r3711, r37, r3713;
}
{
add.f16x2 r3714, r3690, r3711;
}
{
cvt.rn.f16.f64 rs574, fd892;
}
mov.b32 r3719, {rs574, rs574};
{
mul.f16x2 r3717, r46, r3719;
}
{
add.f16x2 r3720, r3696, r3717;
}
{
cvt.rn.f16.f64 rs575, fd891;
}
mov.b32 r3725, {rs575, rs575};
{
mul.f16x2 r3723, r40, r3725;
}
{
add.f16x2 r3726, r3702, r3723;
}
{
cvt.rn.f16.f64 rs576, fd892;
}
mov.b32 r3731, {rs576, rs576};
{
mul.f16x2 r3729, r43, r3731;
}
{
add.f16x2 r3732, r3708, r3729;
}
{
cvt.rn.f16.f64 rs577, fd867;
}
mov.b32 r3737, {rs577, rs577};
{
mul.f16x2 r3735, r49, r3737;
}
{
add.f16x2 r3738, r3714, r3735;
}
{
cvt.rn.f16.f64 rs578, fd812;
}
mov.b32 r3743, {rs578, rs578};
{
mul.f16x2 r3741, r58, r3743;
}
{
add.f16x2 r3744, r3720, r3741;
}
{
cvt.rn.f16.f64 rs579, fd867;
}
mov.b32 r3749, {rs579, rs579};
{
mul.f16x2 r3747, r52, r3749;
}
{
add.f16x2 r3750, r3726, r3747;
}
{
cvt.rn.f16.f64 rs580, fd812;
}
mov.b32 r3755, {rs580, rs580};
{
mul.f16x2 r3753, r55, r3755;
}
{
add.f16x2 r3756, r3732, r3753;
}
{
cvt.rn.f16.f64 rs581, fd855;
}
mov.b32 r3761, {rs581, rs581};
{
mul.f16x2 r3759, r61, r3761;
}
{
add.f16x2 r3762, r3738, r3759;
}
{
cvt.rn.f16.f64 rs582, fd856;
}
mov.b32 r3767, {rs582, rs582};
{
mul.f16x2 r3765, r70, r3767;
}
{
add.f16x2 r3768, r3744, r3765;
}
{
cvt.rn.f16.f64 rs583, fd855;
}
mov.b32 r3773, {rs583, rs583};
{
mul.f16x2 r3771, r64, r3773;
}
{
add.f16x2 r3774, r3750, r3771;
}
{
cvt.rn.f16.f64 rs584, fd856;
}
mov.b32 r3779, {rs584, rs584};
{
mul.f16x2 r3777, r67, r3779;
}
{
add.f16x2 r3780, r3756, r3777;
}
{
cvt.rn.f16.f64 rs585, fd899;
}
mov.b32 r3785, {rs585, rs585};
{
mul.f16x2 r3783, r73, r3785;
}
{
add.f16x2 r3786, r3762, r3783;
}
{
cvt.rn.f16.f64 rs586, fd900;
}
mov.b32 r3791, {rs586, rs586};
{
mul.f16x2 r3789, r82, r3791;
}
{
add.f16x2 r3792, r3768, r3789;
}
{
cvt.rn.f16.f64 rs587, fd899;
}
mov.b32 r3797, {rs587, rs587};
{
mul.f16x2 r3795, r76, r3797;
}
{
add.f16x2 r3798, r3774, r3795;
}
{
cvt.rn.f16.f64 rs588, fd900;
}
mov.b32 r3803, {rs588, rs588};
{
mul.f16x2 r3801, r79, r3803;
}
{
add.f16x2 r3804, r3780, r3801;
}
{
cvt.rn.f16.f64 rs589, fd859;
}
mov.b32 r3809, {rs589, rs589};
{
mul.f16x2 r3807, r85, r3809;
}
{
add.f16x2 r3810, r3786, r3807;
}
{
cvt.rn.f16.f64 rs590, fd572;
}
mov.b32 r3815, {rs590, rs590};
{
mul.f16x2 r3813, r94, r3815;
}
{
add.f16x2 r3816, r3792, r3813;
}
{
cvt.rn.f16.f64 rs591, fd859;
}
mov.b32 r3821, {rs591, rs591};
{
mul.f16x2 r3819, r88, r3821;
}
{
add.f16x2 r3822, r3798, r3819;
}
{
cvt.rn.f16.f64 rs592, fd572;
}
mov.b32 r3827, {rs592, rs592};
{
mul.f16x2 r3825, r91, r3827;
}
{
add.f16x2 r3828, r3804, r3825;
}
{
cvt.rn.f16.f64 rs593, fd863;
}
mov.b32 r3833, {rs593, rs593};
{
mul.f16x2 r3831, r97, r3833;
}
{
add.f16x2 r3834, r3810, r3831;
}
{
cvt.rn.f16.f64 rs594, fd864;
}
mov.b32 r3839, {rs594, rs594};
{
mul.f16x2 r3837, r106, r3839;
}
{
add.f16x2 r3840, r3816, r3837;
}
{
cvt.rn.f16.f64 rs595, fd863;
}
mov.b32 r3845, {rs595, rs595};
{
mul.f16x2 r3843, r100, r3845;
}
{
add.f16x2 r3846, r3822, r3843;
}
{
cvt.rn.f16.f64 rs596, fd864;
}
mov.b32 r3851, {rs596, rs596};
{
mul.f16x2 r3849, r103, r3851;
}
{
add.f16x2 r3852, r3828, r3849;
}
{
cvt.rn.f16.f64 rs597, fd895;
}
mov.b32 r3857, {rs597, rs597};
{
mul.f16x2 r3855, r109, r3857;
}
{
add.f16x2 r3858, r3834, r3855;
}
{
cvt.rn.f16.f64 rs598, fd580;
}
mov.b32 r3863, {rs598, rs598};
{
mul.f16x2 r3861, r118, r3863;
}
{
add.f16x2 r3864, r3840, r3861;
}
{
cvt.rn.f16.f64 rs599, fd895;
}
mov.b32 r3869, {rs599, rs599};
{
mul.f16x2 r3867, r112, r3869;
}
{
add.f16x2 r3870, r3846, r3867;
}
{
cvt.rn.f16.f64 rs600, fd580;
}
mov.b32 r3875, {rs600, rs600};
{
mul.f16x2 r3873, r115, r3875;
}
{
add.f16x2 r3876, r3852, r3873;
}
{
cvt.rn.f16.f64 rs601, fd851;
}
mov.b32 r3881, {rs601, rs601};
{
mul.f16x2 r3879, r121, r3881;
}
{
add.f16x2 r3882, r3858, r3879;
}
{
cvt.rn.f16.f64 rs602, fd676;
}
mov.b32 r3887, {rs602, rs602};
{
mul.f16x2 r3885, r130, r3887;
}
{
add.f16x2 r3888, r3864, r3885;
}
{
cvt.rn.f16.f64 rs603, fd851;
}
mov.b32 r3893, {rs603, rs603};
{
mul.f16x2 r3891, r124, r3893;
}
{
add.f16x2 r3894, r3870, r3891;
}
{
cvt.rn.f16.f64 rs604, fd676;
}
mov.b32 r3899, {rs604, rs604};
{
mul.f16x2 r3897, r127, r3899;
}
{
add.f16x2 r3900, r3876, r3897;
}
{
cvt.rn.f16.f64 rs605, fd871;
}
mov.b32 r3905, {rs605, rs605};
{
mul.f16x2 r3903, r133, r3905;
}
{
add.f16x2 r3906, r3882, r3903;
}
{
cvt.rn.f16.f64 rs606, fd872;
}
mov.b32 r3911, {rs606, rs606};
{
mul.f16x2 r3909, r142, r3911;
}
{
add.f16x2 r3912, r3888, r3909;
}
{
cvt.rn.f16.f64 rs607, fd871;
}
mov.b32 r3917, {rs607, rs607};
{
mul.f16x2 r3915, r136, r3917;
}
{
add.f16x2 r3918, r3894, r3915;
}
{
cvt.rn.f16.f64 rs608, fd872;
}
mov.b32 r3923, {rs608, rs608};
{
mul.f16x2 r3921, r139, r3923;
}
{
add.f16x2 r3924, r3900, r3921;
}
{
cvt.rn.f16.f64 rs609, fd887;
}
mov.b32 r3929, {rs609, rs609};
{
mul.f16x2 r3927, r145, r3929;
}
{
add.f16x2 r3930, r3906, r3927;
}
{
cvt.rn.f16.f64 rs610, fd760;
}
mov.b32 r3935, {rs610, rs610};
{
mul.f16x2 r3933, r154, r3935;
}
{
add.f16x2 r3936, r3912, r3933;
}
{
cvt.rn.f16.f64 rs611, fd887;
}
mov.b32 r3941, {rs611, rs611};
{
mul.f16x2 r3939, r148, r3941;
}
{
add.f16x2 r3942, r3918, r3939;
}
{
cvt.rn.f16.f64 rs612, fd760;
}
mov.b32 r3947, {rs612, rs612};
{
mul.f16x2 r3945, r151, r3947;
}
{
add.f16x2 r3948, r3924, r3945;
}
{
cvt.rn.f16.f64 rs613, fd843;
}
mov.b32 r3953, {rs613, rs613};
{
mul.f16x2 r3951, r157, r3953;
}
{
add.f16x2 r3954, r3930, r3951;
}
{
cvt.rn.f16.f64 rs614, fd820;
}
mov.b32 r3959, {rs614, rs614};
{
mul.f16x2 r3957, r166, r3959;
}
{
add.f16x2 r3960, r3936, r3957;
}
{
cvt.rn.f16.f64 rs615, fd843;
}
mov.b32 r3965, {rs615, rs615};
{
mul.f16x2 r3963, r160, r3965;
}
{
add.f16x2 r3966, r3942, r3963;
}
{
cvt.rn.f16.f64 rs616, fd820;
}
mov.b32 r3971, {rs616, rs616};
{
mul.f16x2 r3969, r163, r3971;
}
{
add.f16x2 r3972, r3948, r3969;
}
{
cvt.rn.f16.f64 rs617, fd879;
}
mov.b32 r3977, {rs617, rs617};
{
mul.f16x2 r3975, r169, r3977;
}
{
add.f16x2 r3978, r3954, r3975;
}
{
cvt.rn.f16.f64 rs618, fd880;
}
mov.b32 r3983, {rs618, rs618};
{
mul.f16x2 r3981, r178, r3983;
}
{
add.f16x2 r3984, r3960, r3981;
}
{
cvt.rn.f16.f64 rs619, fd879;
}
mov.b32 r3989, {rs619, rs619};
{
mul.f16x2 r3987, r172, r3989;
}
{
add.f16x2 r3990, r3966, r3987;
}
{
cvt.rn.f16.f64 rs620, fd880;
}
mov.b32 r3995, {rs620, rs620};
{
mul.f16x2 r3993, r175, r3995;
}
{
add.f16x2 r3996, r3972, r3993;
}
{
sub.f16x2 %20, r3978, r3984;
}
{
add.f16x2 %21, r3990, r3996;
}
{
add.f16x2 %42, r3978, r3984;
}
{
sub.f16x2 %43, r3990, r3996;
}
cvt.rn.f16.s32 rs621, r5508;
mov.b32 r4023, {rs621, rs621};
cvt.rn.f16.s32 rs622, r5508;
mov.b32 r4035, {rs622, rs622};
{
cvt.rn.f16.f64 rs623, fd875;
}
mov.b32 r4015, {rs623, rs623};
{
mul.f16x2 r4013, r1, r4015;
}
{
add.f16x2 r4016, %62, r4013;
}
{
cvt.rn.f16.f64 rs624, fd876;
}
mov.b32 r4021, {rs624, rs624};
{
mul.f16x2 r4019, r10, r4021;
}
{
add.f16x2 r4022, r4023, r4019;
}
{
cvt.rn.f16.f64 rs625, fd875;
}
mov.b32 r4027, {rs625, rs625};
{
mul.f16x2 r4025, r4, r4027;
}
{
add.f16x2 r4028, %63, r4025;
}
{
cvt.rn.f16.f64 rs626, fd876;
}
mov.b32 r4033, {rs626, rs626};
{
mul.f16x2 r4031, r7, r4033;
}
{
add.f16x2 r4034, r4035, r4031;
}
{
cvt.rn.f16.f64 rs627, fd891;
}
mov.b32 r4039, {rs627, rs627};
{
mul.f16x2 r4037, r13, r4039;
}
{
add.f16x2 r4040, r4016, r4037;
}
{
cvt.rn.f16.f64 rs628, fd804;
}
mov.b32 r4045, {rs628, rs628};
{
mul.f16x2 r4043, r22, r4045;
}
{
add.f16x2 r4046, r4022, r4043;
}
{
cvt.rn.f16.f64 rs629, fd891;
}
mov.b32 r4051, {rs629, rs629};
{
mul.f16x2 r4049, r16, r4051;
}
{
add.f16x2 r4052, r4028, r4049;
}
{
cvt.rn.f16.f64 rs630, fd804;
}
mov.b32 r4057, {rs630, rs630};
{
mul.f16x2 r4055, r19, r4057;
}
{
add.f16x2 r4058, r4034, r4055;
}
{
cvt.rn.f16.f64 rs631, fd855;
}
mov.b32 r4063, {rs631, rs631};
{
mul.f16x2 r4061, r25, r4063;
}
{
add.f16x2 r4064, r4040, r4061;
}
{
cvt.rn.f16.f64 rs632, fd816;
}
mov.b32 r4069, {rs632, rs632};
{
mul.f16x2 r4067, r34, r4069;
}
{
add.f16x2 r4070, r4046, r4067;
}
{
cvt.rn.f16.f64 rs633, fd855;
}
mov.b32 r4075, {rs633, rs633};
{
mul.f16x2 r4073, r28, r4075;
}
{
add.f16x2 r4076, r4052, r4073;
}
{
cvt.rn.f16.f64 rs634, fd816;
}
mov.b32 r4081, {rs634, rs634};
{
mul.f16x2 r4079, r31, r4081;
}
{
add.f16x2 r4082, r4058, r4079;
}
{
cvt.rn.f16.f64 rs635, fd859;
}
mov.b32 r4087, {rs635, rs635};
{
mul.f16x2 r4085, r37, r4087;
}
{
add.f16x2 r4088, r4064, r4085;
}
{
cvt.rn.f16.f64 rs636, fd860;
}
mov.b32 r4093, {rs636, rs636};
{
mul.f16x2 r4091, r46, r4093;
}
{
add.f16x2 r4094, r4070, r4091;
}
{
cvt.rn.f16.f64 rs637, fd859;
}
mov.b32 r4099, {rs637, rs637};
{
mul.f16x2 r4097, r40, r4099;
}
{
add.f16x2 r4100, r4076, r4097;
}
{
cvt.rn.f16.f64 rs638, fd860;
}
mov.b32 r4105, {rs638, rs638};
{
mul.f16x2 r4103, r43, r4105;
}
{
add.f16x2 r4106, r4082, r4103;
}
{
cvt.rn.f16.f64 rs639, fd895;
}
mov.b32 r4111, {rs639, rs639};
{
mul.f16x2 r4109, r49, r4111;
}
{
add.f16x2 r4112, r4088, r4109;
}
{
cvt.rn.f16.f64 rs640, fd896;
}
mov.b32 r4117, {rs640, rs640};
{
mul.f16x2 r4115, r58, r4117;
}
{
add.f16x2 r4118, r4094, r4115;
}
{
cvt.rn.f16.f64 rs641, fd895;
}
mov.b32 r4123, {rs641, rs641};
{
mul.f16x2 r4121, r52, r4123;
}
{
add.f16x2 r4124, r4100, r4121;
}
{
cvt.rn.f16.f64 rs642, fd896;
}
mov.b32 r4129, {rs642, rs642};
{
mul.f16x2 r4127, r55, r4129;
}
{
add.f16x2 r4130, r4106, r4127;
}
{
cvt.rn.f16.f64 rs643, fd871;
}
mov.b32 r4135, {rs643, rs643};
{
mul.f16x2 r4133, r61, r4135;
}
{
add.f16x2 r4136, r4112, r4133;
}
{
cvt.rn.f16.f64 rs644, fd624;
}
mov.b32 r4141, {rs644, rs644};
{
mul.f16x2 r4139, r70, r4141;
}
{
add.f16x2 r4142, r4118, r4139;
}
{
cvt.rn.f16.f64 rs645, fd871;
}
mov.b32 r4147, {rs645, rs645};
{
mul.f16x2 r4145, r64, r4147;
}
{
add.f16x2 r4148, r4124, r4145;
}
{
cvt.rn.f16.f64 rs646, fd624;
}
mov.b32 r4153, {rs646, rs646};
{
mul.f16x2 r4151, r67, r4153;
}
{
add.f16x2 r4154, r4130, r4151;
}
{
cvt.rn.f16.f64 rs647, fd843;
}
mov.b32 r4159, {rs647, rs647};
{
mul.f16x2 r4157, r73, r4159;
}
{
add.f16x2 r4160, r4136, r4157;
}
{
cvt.rn.f16.f64 rs648, fd844;
}
mov.b32 r4165, {rs648, rs648};
{
mul.f16x2 r4163, r82, r4165;
}
{
add.f16x2 r4166, r4142, r4163;
}
{
cvt.rn.f16.f64 rs649, fd843;
}
mov.b32 r4171, {rs649, rs649};
{
mul.f16x2 r4169, r76, r4171;
}
{
add.f16x2 r4172, r4148, r4169;
}
{
cvt.rn.f16.f64 rs650, fd844;
}
mov.b32 r4177, {rs650, rs650};
{
mul.f16x2 r4175, r79, r4177;
}
{
add.f16x2 r4178, r4154, r4175;
}
{
cvt.rn.f16.f64 rs651, fd879;
}
mov.b32 r4183, {rs651, rs651};
{
mul.f16x2 r4181, r85, r4183;
}
{
add.f16x2 r4184, r4160, r4181;
}
{
cvt.rn.f16.f64 rs652, fd880;
}
mov.b32 r4189, {rs652, rs652};
{
mul.f16x2 r4187, r94, r4189;
}
{
add.f16x2 r4190, r4166, r4187;
}
{
cvt.rn.f16.f64 rs653, fd879;
}
mov.b32 r4195, {rs653, rs653};
{
mul.f16x2 r4193, r88, r4195;
}
{
add.f16x2 r4196, r4172, r4193;
}
{
cvt.rn.f16.f64 rs654, fd880;
}
mov.b32 r4201, {rs654, rs654};
{
mul.f16x2 r4199, r91, r4201;
}
{
add.f16x2 r4202, r4178, r4199;
}
{
cvt.rn.f16.f64 rs655, fd887;
}
mov.b32 r4207, {rs655, rs655};
{
mul.f16x2 r4205, r97, r4207;
}
{
add.f16x2 r4208, r4184, r4205;
}
{
cvt.rn.f16.f64 rs656, fd760;
}
mov.b32 r4213, {rs656, rs656};
{
mul.f16x2 r4211, r106, r4213;
}
{
add.f16x2 r4214, r4190, r4211;
}
{
cvt.rn.f16.f64 rs657, fd887;
}
mov.b32 r4219, {rs657, rs657};
{
mul.f16x2 r4217, r100, r4219;
}
{
add.f16x2 r4220, r4196, r4217;
}
{
cvt.rn.f16.f64 rs658, fd760;
}
mov.b32 r4225, {rs658, rs658};
{
mul.f16x2 r4223, r103, r4225;
}
{
add.f16x2 r4226, r4202, r4223;
}
{
cvt.rn.f16.f64 rs659, fd851;
}
mov.b32 r4231, {rs659, rs659};
{
mul.f16x2 r4229, r109, r4231;
}
{
add.f16x2 r4232, r4208, r4229;
}
{
cvt.rn.f16.f64 rs660, fd676;
}
mov.b32 r4237, {rs660, rs660};
{
mul.f16x2 r4235, r118, r4237;
}
{
add.f16x2 r4238, r4214, r4235;
}
{
cvt.rn.f16.f64 rs661, fd851;
}
mov.b32 r4243, {rs661, rs661};
{
mul.f16x2 r4241, r112, r4243;
}
{
add.f16x2 r4244, r4220, r4241;
}
{
cvt.rn.f16.f64 rs662, fd676;
}
mov.b32 r4249, {rs662, rs662};
{
mul.f16x2 r4247, r115, r4249;
}
{
add.f16x2 r4250, r4226, r4247;
}
{
cvt.rn.f16.f64 rs663, fd863;
}
mov.b32 r4255, {rs663, rs663};
{
mul.f16x2 r4253, r121, r4255;
}
{
add.f16x2 r4256, r4232, r4253;
}
{
cvt.rn.f16.f64 rs664, fd864;
}
mov.b32 r4261, {rs664, rs664};
{
mul.f16x2 r4259, r130, r4261;
}
{
add.f16x2 r4262, r4238, r4259;
}
{
cvt.rn.f16.f64 rs665, fd863;
}
mov.b32 r4267, {rs665, rs665};
{
mul.f16x2 r4265, r124, r4267;
}
{
add.f16x2 r4268, r4244, r4265;
}
{
cvt.rn.f16.f64 rs666, fd864;
}
mov.b32 r4273, {rs666, rs666};
{
mul.f16x2 r4271, r127, r4273;
}
{
add.f16x2 r4274, r4250, r4271;
}
{
cvt.rn.f16.f64 rs667, fd899;
}
mov.b32 r4279, {rs667, rs667};
{
mul.f16x2 r4277, r133, r4279;
}
{
add.f16x2 r4280, r4256, r4277;
}
{
cvt.rn.f16.f64 rs668, fd900;
}
mov.b32 r4285, {rs668, rs668};
{
mul.f16x2 r4283, r142, r4285;
}
{
add.f16x2 r4286, r4262, r4283;
}
{
cvt.rn.f16.f64 rs669, fd899;
}
mov.b32 r4291, {rs669, rs669};
{
mul.f16x2 r4289, r136, r4291;
}
{
add.f16x2 r4292, r4268, r4289;
}
{
cvt.rn.f16.f64 rs670, fd900;
}
mov.b32 r4297, {rs670, rs670};
{
mul.f16x2 r4295, r139, r4297;
}
{
add.f16x2 r4298, r4274, r4295;
}
{
cvt.rn.f16.f64 rs671, fd867;
}
mov.b32 r4303, {rs671, rs671};
{
mul.f16x2 r4301, r145, r4303;
}
{
add.f16x2 r4304, r4280, r4301;
}
{
cvt.rn.f16.f64 rs672, fd812;
}
mov.b32 r4309, {rs672, rs672};
{
mul.f16x2 r4307, r154, r4309;
}
{
add.f16x2 r4310, r4286, r4307;
}
{
cvt.rn.f16.f64 rs673, fd867;
}
mov.b32 r4315, {rs673, rs673};
{
mul.f16x2 r4313, r148, r4315;
}
{
add.f16x2 r4316, r4292, r4313;
}
{
cvt.rn.f16.f64 rs674, fd812;
}
mov.b32 r4321, {rs674, rs674};
{
mul.f16x2 r4319, r151, r4321;
}
{
add.f16x2 r4322, r4298, r4319;
}
{
cvt.rn.f16.f64 rs675, fd847;
}
mov.b32 r4327, {rs675, rs675};
{
mul.f16x2 r4325, r157, r4327;
}
{
add.f16x2 r4328, r4304, r4325;
}
{
cvt.rn.f16.f64 rs676, fd848;
}
mov.b32 r4333, {rs676, rs676};
{
mul.f16x2 r4331, r166, r4333;
}
{
add.f16x2 r4334, r4310, r4331;
}
{
cvt.rn.f16.f64 rs677, fd847;
}
mov.b32 r4339, {rs677, rs677};
{
mul.f16x2 r4337, r160, r4339;
}
{
add.f16x2 r4340, r4316, r4337;
}
{
cvt.rn.f16.f64 rs678, fd848;
}
mov.b32 r4345, {rs678, rs678};
{
mul.f16x2 r4343, r163, r4345;
}
{
add.f16x2 r4346, r4322, r4343;
}
{
cvt.rn.f16.f64 rs679, fd883;
}
mov.b32 r4351, {rs679, rs679};
{
mul.f16x2 r4349, r169, r4351;
}
{
add.f16x2 r4352, r4328, r4349;
}
{
cvt.rn.f16.f64 rs680, fd884;
}
mov.b32 r4357, {rs680, rs680};
{
mul.f16x2 r4355, r178, r4357;
}
{
add.f16x2 r4358, r4334, r4355;
}
{
cvt.rn.f16.f64 rs681, fd883;
}
mov.b32 r4363, {rs681, rs681};
{
mul.f16x2 r4361, r172, r4363;
}
{
add.f16x2 r4364, r4340, r4361;
}
{
cvt.rn.f16.f64 rs682, fd884;
}
mov.b32 r4369, {rs682, rs682};
{
mul.f16x2 r4367, r175, r4369;
}
{
add.f16x2 r4370, r4346, r4367;
}
{
sub.f16x2 %22, r4352, r4358;
}
{
add.f16x2 %23, r4364, r4370;
}
{
add.f16x2 %40, r4352, r4358;
}
{
sub.f16x2 %41, r4364, r4370;
}
cvt.rn.f16.s32 rs683, r5508;
mov.b32 r4397, {rs683, rs683};
cvt.rn.f16.s32 rs684, r5508;
mov.b32 r4409, {rs684, rs684};
{
cvt.rn.f16.f64 rs685, fd867;
}
mov.b32 r4389, {rs685, rs685};
{
mul.f16x2 r4387, r1, r4389;
}
{
add.f16x2 r4390, %62, r4387;
}
{
cvt.rn.f16.f64 rs686, fd868;
}
mov.b32 r4395, {rs686, rs686};
{
mul.f16x2 r4393, r10, r4395;
}
{
add.f16x2 r4396, r4397, r4393;
}
{
cvt.rn.f16.f64 rs687, fd867;
}
mov.b32 r4401, {rs687, rs687};
{
mul.f16x2 r4399, r4, r4401;
}
{
add.f16x2 r4402, %63, r4399;
}
{
cvt.rn.f16.f64 rs688, fd868;
}
mov.b32 r4407, {rs688, rs688};
{
mul.f16x2 r4405, r7, r4407;
}
{
add.f16x2 r4408, r4409, r4405;
}
{
cvt.rn.f16.f64 rs689, fd895;
}
mov.b32 r4413, {rs689, rs689};
{
mul.f16x2 r4411, r13, r4413;
}
{
add.f16x2 r4414, r4390, r4411;
}
{
cvt.rn.f16.f64 rs690, fd896;
}
mov.b32 r4419, {rs690, rs690};
{
mul.f16x2 r4417, r22, r4419;
}
{
add.f16x2 r4420, r4396, r4417;
}
{
cvt.rn.f16.f64 rs691, fd895;
}
mov.b32 r4425, {rs691, rs691};
{
mul.f16x2 r4423, r16, r4425;
}
{
add.f16x2 r4426, r4402, r4423;
}
{
cvt.rn.f16.f64 rs692, fd896;
}
mov.b32 r4431, {rs692, rs692};
{
mul.f16x2 r4429, r19, r4431;
}
{
add.f16x2 r4432, r4408, r4429;
}
{
cvt.rn.f16.f64 rs693, fd879;
}
mov.b32 r4437, {rs693, rs693};
{
mul.f16x2 r4435, r25, r4437;
}
{
add.f16x2 r4438, r4414, r4435;
}
{
cvt.rn.f16.f64 rs694, fd808;
}
mov.b32 r4443, {rs694, rs694};
{
mul.f16x2 r4441, r34, r4443;
}
{
add.f16x2 r4444, r4420, r4441;
}
{
cvt.rn.f16.f64 rs695, fd879;
}
mov.b32 r4449, {rs695, rs695};
{
mul.f16x2 r4447, r28, r4449;
}
{
add.f16x2 r4450, r4426, r4447;
}
{
cvt.rn.f16.f64 rs696, fd808;
}
mov.b32 r4455, {rs696, rs696};
{
mul.f16x2 r4453, r31, r4455;
}
{
add.f16x2 r4456, r4432, r4453;
}
{
cvt.rn.f16.f64 rs697, fd851;
}
mov.b32 r4461, {rs697, rs697};
{
mul.f16x2 r4459, r37, r4461;
}
{
add.f16x2 r4462, r4438, r4459;
}
{
cvt.rn.f16.f64 rs698, fd676;
}
mov.b32 r4467, {rs698, rs698};
{
mul.f16x2 r4465, r46, r4467;
}
{
add.f16x2 r4468, r4444, r4465;
}
{
cvt.rn.f16.f64 rs699, fd851;
}
mov.b32 r4473, {rs699, rs699};
{
mul.f16x2 r4471, r40, r4473;
}
{
add.f16x2 r4474, r4450, r4471;
}
{
cvt.rn.f16.f64 rs700, fd676;
}
mov.b32 r4479, {rs700, rs700};
{
mul.f16x2 r4477, r43, r4479;
}
{
add.f16x2 r4480, r4456, r4477;
}
{
cvt.rn.f16.f64 rs701, fd855;
}
mov.b32 r4485, {rs701, rs701};
{
mul.f16x2 r4483, r49, r4485;
}
{
add.f16x2 r4486, r4462, r4483;
}
{
cvt.rn.f16.f64 rs702, fd856;
}
mov.b32 r4491, {rs702, rs702};
{
mul.f16x2 r4489, r58, r4491;
}
{
add.f16x2 r4492, r4468, r4489;
}
{
cvt.rn.f16.f64 rs703, fd855;
}
mov.b32 r4497, {rs703, rs703};
{
mul.f16x2 r4495, r52, r4497;
}
{
add.f16x2 r4498, r4474, r4495;
}
{
cvt.rn.f16.f64 rs704, fd856;
}
mov.b32 r4503, {rs704, rs704};
{
mul.f16x2 r4501, r55, r4503;
}
{
add.f16x2 r4504, r4480, r4501;
}
{
cvt.rn.f16.f64 rs705, fd883;
}
mov.b32 r4509, {rs705, rs705};
{
mul.f16x2 r4507, r61, r4509;
}
{
add.f16x2 r4510, r4486, r4507;
}
{
cvt.rn.f16.f64 rs706, fd884;
}
mov.b32 r4515, {rs706, rs706};
{
mul.f16x2 r4513, r70, r4515;
}
{
add.f16x2 r4516, r4492, r4513;
}
{
cvt.rn.f16.f64 rs707, fd883;
}
mov.b32 r4521, {rs707, rs707};
{
mul.f16x2 r4519, r64, r4521;
}
{
add.f16x2 r4522, r4498, r4519;
}
{
cvt.rn.f16.f64 rs708, fd884;
}
mov.b32 r4527, {rs708, rs708};
{
mul.f16x2 r4525, r67, r4527;
}
{
add.f16x2 r4528, r4504, r4525;
}
{
cvt.rn.f16.f64 rs709, fd891;
}
mov.b32 r4533, {rs709, rs709};
{
mul.f16x2 r4531, r73, r4533;
}
{
add.f16x2 r4534, r4510, r4531;
}
{
cvt.rn.f16.f64 rs710, fd804;
}
mov.b32 r4539, {rs710, rs710};
{
mul.f16x2 r4537, r82, r4539;
}
{
add.f16x2 r4540, r4516, r4537;
}
{
cvt.rn.f16.f64 rs711, fd891;
}
mov.b32 r4545, {rs711, rs711};
{
mul.f16x2 r4543, r76, r4545;
}
{
add.f16x2 r4546, r4522, r4543;
}
{
cvt.rn.f16.f64 rs712, fd804;
}
mov.b32 r4551, {rs712, rs712};
{
mul.f16x2 r4549, r79, r4551;
}
{
add.f16x2 r4552, r4528, r4549;
}
{
cvt.rn.f16.f64 rs713, fd863;
}
mov.b32 r4557, {rs713, rs713};
{
mul.f16x2 r4555, r85, r4557;
}
{
add.f16x2 r4558, r4534, r4555;
}
{
cvt.rn.f16.f64 rs714, fd740;
}
mov.b32 r4563, {rs714, rs714};
{
mul.f16x2 r4561, r94, r4563;
}
{
add.f16x2 r4564, r4540, r4561;
}
{
cvt.rn.f16.f64 rs715, fd863;
}
mov.b32 r4569, {rs715, rs715};
{
mul.f16x2 r4567, r88, r4569;
}
{
add.f16x2 r4570, r4546, r4567;
}
{
cvt.rn.f16.f64 rs716, fd740;
}
mov.b32 r4575, {rs716, rs716};
{
mul.f16x2 r4573, r91, r4575;
}
{
add.f16x2 r4576, r4552, r4573;
}
{
cvt.rn.f16.f64 rs717, fd843;
}
mov.b32 r4581, {rs717, rs717};
{
mul.f16x2 r4579, r97, r4581;
}
{
add.f16x2 r4582, r4558, r4579;
}
{
cvt.rn.f16.f64 rs718, fd844;
}
mov.b32 r4587, {rs718, rs718};
{
mul.f16x2 r4585, r106, r4587;
}
{
add.f16x2 r4588, r4564, r4585;
}
{
cvt.rn.f16.f64 rs719, fd843;
}
mov.b32 r4593, {rs719, rs719};
{
mul.f16x2 r4591, r100, r4593;
}
{
add.f16x2 r4594, r4570, r4591;
}
{
cvt.rn.f16.f64 rs720, fd844;
}
mov.b32 r4599, {rs720, rs720};
{
mul.f16x2 r4597, r103, r4599;
}
{
add.f16x2 r4600, r4576, r4597;
}
{
cvt.rn.f16.f64 rs721, fd871;
}
mov.b32 r4605, {rs721, rs721};
{
mul.f16x2 r4603, r109, r4605;
}
{
add.f16x2 r4606, r4582, r4603;
}
{
cvt.rn.f16.f64 rs722, fd872;
}
mov.b32 r4611, {rs722, rs722};
{
mul.f16x2 r4609, r118, r4611;
}
{
add.f16x2 r4612, r4588, r4609;
}
{
cvt.rn.f16.f64 rs723, fd871;
}
mov.b32 r4617, {rs723, rs723};
{
mul.f16x2 r4615, r112, r4617;
}
{
add.f16x2 r4618, r4594, r4615;
}
{
cvt.rn.f16.f64 rs724, fd872;
}
mov.b32 r4623, {rs724, rs724};
{
mul.f16x2 r4621, r115, r4623;
}
{
add.f16x2 r4624, r4600, r4621;
}
{
cvt.rn.f16.f64 rs725, fd899;
}
mov.b32 r4629, {rs725, rs725};
{
mul.f16x2 r4627, r121, r4629;
}
{
add.f16x2 r4630, r4606, r4627;
}
{
cvt.rn.f16.f64 rs726, fd900;
}
mov.b32 r4635, {rs726, rs726};
{
mul.f16x2 r4633, r130, r4635;
}
{
add.f16x2 r4636, r4612, r4633;
}
{
cvt.rn.f16.f64 rs727, fd899;
}
mov.b32 r4641, {rs727, rs727};
{
mul.f16x2 r4639, r124, r4641;
}
{
add.f16x2 r4642, r4618, r4639;
}
{
cvt.rn.f16.f64 rs728, fd900;
}
mov.b32 r4647, {rs728, rs728};
{
mul.f16x2 r4645, r127, r4647;
}
{
add.f16x2 r4648, r4624, r4645;
}
{
cvt.rn.f16.f64 rs729, fd875;
}
mov.b32 r4653, {rs729, rs729};
{
mul.f16x2 r4651, r133, r4653;
}
{
add.f16x2 r4654, r4630, r4651;
}
{
cvt.rn.f16.f64 rs730, fd708;
}
mov.b32 r4659, {rs730, rs730};
{
mul.f16x2 r4657, r142, r4659;
}
{
add.f16x2 r4660, r4636, r4657;
}
{
cvt.rn.f16.f64 rs731, fd875;
}
mov.b32 r4665, {rs731, rs731};
{
mul.f16x2 r4663, r136, r4665;
}
{
add.f16x2 r4666, r4642, r4663;
}
{
cvt.rn.f16.f64 rs732, fd708;
}
mov.b32 r4671, {rs732, rs732};
{
mul.f16x2 r4669, r139, r4671;
}
{
add.f16x2 r4672, r4648, r4669;
}
{
cvt.rn.f16.f64 rs733, fd847;
}
mov.b32 r4677, {rs733, rs733};
{
mul.f16x2 r4675, r145, r4677;
}
{
add.f16x2 r4678, r4654, r4675;
}
{
cvt.rn.f16.f64 rs734, fd768;
}
mov.b32 r4683, {rs734, rs734};
{
mul.f16x2 r4681, r154, r4683;
}
{
add.f16x2 r4684, r4660, r4681;
}
{
cvt.rn.f16.f64 rs735, fd847;
}
mov.b32 r4689, {rs735, rs735};
{
mul.f16x2 r4687, r148, r4689;
}
{
add.f16x2 r4690, r4666, r4687;
}
{
cvt.rn.f16.f64 rs736, fd768;
}
mov.b32 r4695, {rs736, rs736};
{
mul.f16x2 r4693, r151, r4695;
}
{
add.f16x2 r4696, r4672, r4693;
}
{
cvt.rn.f16.f64 rs737, fd859;
}
mov.b32 r4701, {rs737, rs737};
{
mul.f16x2 r4699, r157, r4701;
}
{
add.f16x2 r4702, r4678, r4699;
}
{
cvt.rn.f16.f64 rs738, fd860;
}
mov.b32 r4707, {rs738, rs738};
{
mul.f16x2 r4705, r166, r4707;
}
{
add.f16x2 r4708, r4684, r4705;
}
{
cvt.rn.f16.f64 rs739, fd859;
}
mov.b32 r4713, {rs739, rs739};
{
mul.f16x2 r4711, r160, r4713;
}
{
add.f16x2 r4714, r4690, r4711;
}
{
cvt.rn.f16.f64 rs740, fd860;
}
mov.b32 r4719, {rs740, rs740};
{
mul.f16x2 r4717, r163, r4719;
}
{
add.f16x2 r4720, r4696, r4717;
}
{
cvt.rn.f16.f64 rs741, fd887;
}
mov.b32 r4725, {rs741, rs741};
{
mul.f16x2 r4723, r169, r4725;
}
{
add.f16x2 r4726, r4702, r4723;
}
{
cvt.rn.f16.f64 rs742, fd888;
}
mov.b32 r4731, {rs742, rs742};
{
mul.f16x2 r4729, r178, r4731;
}
{
add.f16x2 r4732, r4708, r4729;
}
{
cvt.rn.f16.f64 rs743, fd887;
}
mov.b32 r4737, {rs743, rs743};
{
mul.f16x2 r4735, r172, r4737;
}
{
add.f16x2 r4738, r4714, r4735;
}
{
cvt.rn.f16.f64 rs744, fd888;
}
mov.b32 r4743, {rs744, rs744};
{
mul.f16x2 r4741, r175, r4743;
}
{
add.f16x2 r4744, r4720, r4741;
}
{
sub.f16x2 %24, r4726, r4732;
}
{
add.f16x2 %25, r4738, r4744;
}
{
add.f16x2 %38, r4726, r4732;
}
{
sub.f16x2 %39, r4738, r4744;
}
cvt.rn.f16.s32 rs745, r5508;
mov.b32 r4771, {rs745, rs745};
cvt.rn.f16.s32 rs746, r5508;
mov.b32 r4783, {rs746, rs746};
{
cvt.rn.f16.f64 rs747, fd859;
}
mov.b32 r4763, {rs747, rs747};
{
mul.f16x2 r4761, r1, r4763;
}
{
add.f16x2 r4764, %62, r4761;
}
{
cvt.rn.f16.f64 rs748, fd860;
}
mov.b32 r4769, {rs748, rs748};
{
mul.f16x2 r4767, r10, r4769;
}
{
add.f16x2 r4770, r4771, r4767;
}
{
cvt.rn.f16.f64 rs749, fd859;
}
mov.b32 r4775, {rs749, rs749};
{
mul.f16x2 r4773, r4, r4775;
}
{
add.f16x2 r4776, %63, r4773;
}
{
cvt.rn.f16.f64 rs750, fd860;
}
mov.b32 r4781, {rs750, rs750};
{
mul.f16x2 r4779, r7, r4781;
}
{
add.f16x2 r4782, r4783, r4779;
}
{
cvt.rn.f16.f64 rs751, fd879;
}
mov.b32 r4787, {rs751, rs751};
{
mul.f16x2 r4785, r13, r4787;
}
{
add.f16x2 r4788, r4764, r4785;
}
{
cvt.rn.f16.f64 rs752, fd880;
}
mov.b32 r4793, {rs752, rs752};
{
mul.f16x2 r4791, r22, r4793;
}
{
add.f16x2 r4794, r4770, r4791;
}
{
cvt.rn.f16.f64 rs753, fd879;
}
mov.b32 r4799, {rs753, rs753};
{
mul.f16x2 r4797, r16, r4799;
}
{
add.f16x2 r4800, r4776, r4797;
}
{
cvt.rn.f16.f64 rs754, fd880;
}
mov.b32 r4805, {rs754, rs754};
{
mul.f16x2 r4803, r19, r4805;
}
{
add.f16x2 r4806, r4782, r4803;
}
{
cvt.rn.f16.f64 rs755, fd899;
}
mov.b32 r4811, {rs755, rs755};
{
mul.f16x2 r4809, r25, r4811;
}
{
add.f16x2 r4812, r4788, r4809;
}
{
cvt.rn.f16.f64 rs756, fd900;
}
mov.b32 r4817, {rs756, rs756};
{
mul.f16x2 r4815, r34, r4817;
}
{
add.f16x2 r4818, r4794, r4815;
}
{
cvt.rn.f16.f64 rs757, fd899;
}
mov.b32 r4823, {rs757, rs757};
{
mul.f16x2 r4821, r28, r4823;
}
{
add.f16x2 r4824, r4800, r4821;
}
{
cvt.rn.f16.f64 rs758, fd900;
}
mov.b32 r4829, {rs758, rs758};
{
mul.f16x2 r4827, r31, r4829;
}
{
add.f16x2 r4830, r4806, r4827;
}
{
cvt.rn.f16.f64 rs759, fd883;
}
mov.b32 r4835, {rs759, rs759};
{
mul.f16x2 r4833, r37, r4835;
}
{
add.f16x2 r4836, r4812, r4833;
}
{
cvt.rn.f16.f64 rs760, fd736;
}
mov.b32 r4841, {rs760, rs760};
{
mul.f16x2 r4839, r46, r4841;
}
{
add.f16x2 r4842, r4818, r4839;
}
{
cvt.rn.f16.f64 rs761, fd883;
}
mov.b32 r4847, {rs761, rs761};
{
mul.f16x2 r4845, r40, r4847;
}
{
add.f16x2 r4848, r4824, r4845;
}
{
cvt.rn.f16.f64 rs762, fd736;
}
mov.b32 r4853, {rs762, rs762};
{
mul.f16x2 r4851, r43, r4853;
}
{
add.f16x2 r4854, r4830, r4851;
}
{
cvt.rn.f16.f64 rs763, fd863;
}
mov.b32 r4859, {rs763, rs763};
{
mul.f16x2 r4857, r49, r4859;
}
{
add.f16x2 r4860, r4836, r4857;
}
{
cvt.rn.f16.f64 rs764, fd740;
}
mov.b32 r4865, {rs764, rs764};
{
mul.f16x2 r4863, r58, r4865;
}
{
add.f16x2 r4866, r4842, r4863;
}
{
cvt.rn.f16.f64 rs765, fd863;
}
mov.b32 r4871, {rs765, rs765};
{
mul.f16x2 r4869, r52, r4871;
}
{
add.f16x2 r4872, r4848, r4869;
}
{
cvt.rn.f16.f64 rs766, fd740;
}
mov.b32 r4877, {rs766, rs766};
{
mul.f16x2 r4875, r55, r4877;
}
{
add.f16x2 r4878, r4854, r4875;
}
{
cvt.rn.f16.f64 rs767, fd843;
}
mov.b32 r4883, {rs767, rs767};
{
mul.f16x2 r4881, r61, r4883;
}
{
add.f16x2 r4884, r4860, r4881;
}
{
cvt.rn.f16.f64 rs768, fd820;
}
mov.b32 r4889, {rs768, rs768};
{
mul.f16x2 r4887, r70, r4889;
}
{
add.f16x2 r4890, r4866, r4887;
}
{
cvt.rn.f16.f64 rs769, fd843;
}
mov.b32 r4895, {rs769, rs769};
{
mul.f16x2 r4893, r64, r4895;
}
{
add.f16x2 r4896, r4872, r4893;
}
{
cvt.rn.f16.f64 rs770, fd820;
}
mov.b32 r4901, {rs770, rs770};
{
mul.f16x2 r4899, r67, r4901;
}
{
add.f16x2 r4902, r4878, r4899;
}
{
cvt.rn.f16.f64 rs771, fd855;
}
mov.b32 r4907, {rs771, rs771};
{
mul.f16x2 r4905, r73, r4907;
}
{
add.f16x2 r4908, r4884, r4905;
}
{
cvt.rn.f16.f64 rs772, fd856;
}
mov.b32 r4913, {rs772, rs772};
{
mul.f16x2 r4911, r82, r4913;
}
{
add.f16x2 r4914, r4890, r4911;
}
{
cvt.rn.f16.f64 rs773, fd855;
}
mov.b32 r4919, {rs773, rs773};
{
mul.f16x2 r4917, r76, r4919;
}
{
add.f16x2 r4920, r4896, r4917;
}
{
cvt.rn.f16.f64 rs774, fd856;
}
mov.b32 r4925, {rs774, rs774};
{
mul.f16x2 r4923, r79, r4925;
}
{
add.f16x2 r4926, r4902, r4923;
}
{
cvt.rn.f16.f64 rs775, fd875;
}
mov.b32 r4931, {rs775, rs775};
{
mul.f16x2 r4929, r85, r4931;
}
{
add.f16x2 r4932, r4908, r4929;
}
{
cvt.rn.f16.f64 rs776, fd876;
}
mov.b32 r4937, {rs776, rs776};
{
mul.f16x2 r4935, r94, r4937;
}
{
add.f16x2 r4938, r4914, r4935;
}
{
cvt.rn.f16.f64 rs777, fd875;
}
mov.b32 r4943, {rs777, rs777};
{
mul.f16x2 r4941, r88, r4943;
}
{
add.f16x2 r4944, r4920, r4941;
}
{
cvt.rn.f16.f64 rs778, fd876;
}
mov.b32 r4949, {rs778, rs778};
{
mul.f16x2 r4947, r91, r4949;
}
{
add.f16x2 r4950, r4926, r4947;
}
{
cvt.rn.f16.f64 rs779, fd895;
}
mov.b32 r4955, {rs779, rs779};
{
mul.f16x2 r4953, r97, r4955;
}
{
add.f16x2 r4956, r4932, r4953;
}
{
cvt.rn.f16.f64 rs780, fd896;
}
mov.b32 r4961, {rs780, rs780};
{
mul.f16x2 r4959, r106, r4961;
}
{
add.f16x2 r4962, r4938, r4959;
}
{
cvt.rn.f16.f64 rs781, fd895;
}
mov.b32 r4967, {rs781, rs781};
{
mul.f16x2 r4965, r100, r4967;
}
{
add.f16x2 r4968, r4944, r4965;
}
{
cvt.rn.f16.f64 rs782, fd896;
}
mov.b32 r4973, {rs782, rs782};
{
mul.f16x2 r4971, r103, r4973;
}
{
add.f16x2 r4974, r4950, r4971;
}
{
cvt.rn.f16.f64 rs783, fd887;
}
mov.b32 r4979, {rs783, rs783};
{
mul.f16x2 r4977, r109, r4979;
}
{
add.f16x2 r4980, r4956, r4977;
}
{
cvt.rn.f16.f64 rs784, fd760;
}
mov.b32 r4985, {rs784, rs784};
{
mul.f16x2 r4983, r118, r4985;
}
{
add.f16x2 r4986, r4962, r4983;
}
{
cvt.rn.f16.f64 rs785, fd887;
}
mov.b32 r4991, {rs785, rs785};
{
mul.f16x2 r4989, r112, r4991;
}
{
add.f16x2 r4992, r4968, r4989;
}
{
cvt.rn.f16.f64 rs786, fd760;
}
mov.b32 r4997, {rs786, rs786};
{
mul.f16x2 r4995, r115, r4997;
}
{
add.f16x2 r4998, r4974, r4995;
}
{
cvt.rn.f16.f64 rs787, fd867;
}
mov.b32 r5003, {rs787, rs787};
{
mul.f16x2 r5001, r121, r5003;
}
{
add.f16x2 r5004, r4980, r5001;
}
{
cvt.rn.f16.f64 rs788, fd812;
}
mov.b32 r5009, {rs788, rs788};
{
mul.f16x2 r5007, r130, r5009;
}
{
add.f16x2 r5010, r4986, r5007;
}
{
cvt.rn.f16.f64 rs789, fd867;
}
mov.b32 r5015, {rs789, rs789};
{
mul.f16x2 r5013, r124, r5015;
}
{
add.f16x2 r5016, r4992, r5013;
}
{
cvt.rn.f16.f64 rs790, fd812;
}
mov.b32 r5021, {rs790, rs790};
{
mul.f16x2 r5019, r127, r5021;
}
{
add.f16x2 r5022, r4998, r5019;
}
{
cvt.rn.f16.f64 rs791, fd847;
}
mov.b32 r5027, {rs791, rs791};
{
mul.f16x2 r5025, r133, r5027;
}
{
add.f16x2 r5028, r5004, r5025;
}
{
cvt.rn.f16.f64 rs792, fd768;
}
mov.b32 r5033, {rs792, rs792};
{
mul.f16x2 r5031, r142, r5033;
}
{
add.f16x2 r5034, r5010, r5031;
}
{
cvt.rn.f16.f64 rs793, fd847;
}
mov.b32 r5039, {rs793, rs793};
{
mul.f16x2 r5037, r136, r5039;
}
{
add.f16x2 r5040, r5016, r5037;
}
{
cvt.rn.f16.f64 rs794, fd768;
}
mov.b32 r5045, {rs794, rs794};
{
mul.f16x2 r5043, r139, r5045;
}
{
add.f16x2 r5046, r5022, r5043;
}
{
cvt.rn.f16.f64 rs795, fd851;
}
mov.b32 r5051, {rs795, rs795};
{
mul.f16x2 r5049, r145, r5051;
}
{
add.f16x2 r5052, r5028, r5049;
}
{
cvt.rn.f16.f64 rs796, fd852;
}
mov.b32 r5057, {rs796, rs796};
{
mul.f16x2 r5055, r154, r5057;
}
{
add.f16x2 r5058, r5034, r5055;
}
{
cvt.rn.f16.f64 rs797, fd851;
}
mov.b32 r5063, {rs797, rs797};
{
mul.f16x2 r5061, r148, r5063;
}
{
add.f16x2 r5064, r5040, r5061;
}
{
cvt.rn.f16.f64 rs798, fd852;
}
mov.b32 r5069, {rs798, rs798};
{
mul.f16x2 r5067, r151, r5069;
}
{
add.f16x2 r5070, r5046, r5067;
}
{
cvt.rn.f16.f64 rs799, fd871;
}
mov.b32 r5075, {rs799, rs799};
{
mul.f16x2 r5073, r157, r5075;
}
{
add.f16x2 r5076, r5052, r5073;
}
{
cvt.rn.f16.f64 rs800, fd872;
}
mov.b32 r5081, {rs800, rs800};
{
mul.f16x2 r5079, r166, r5081;
}
{
add.f16x2 r5082, r5058, r5079;
}
{
cvt.rn.f16.f64 rs801, fd871;
}
mov.b32 r5087, {rs801, rs801};
{
mul.f16x2 r5085, r160, r5087;
}
{
add.f16x2 r5088, r5064, r5085;
}
{
cvt.rn.f16.f64 rs802, fd872;
}
mov.b32 r5093, {rs802, rs802};
{
mul.f16x2 r5091, r163, r5093;
}
{
add.f16x2 r5094, r5070, r5091;
}
{
cvt.rn.f16.f64 rs803, fd891;
}
mov.b32 r5099, {rs803, rs803};
{
mul.f16x2 r5097, r169, r5099;
}
{
add.f16x2 r5100, r5076, r5097;
}
{
cvt.rn.f16.f64 rs804, fd892;
}
mov.b32 r5105, {rs804, rs804};
{
mul.f16x2 r5103, r178, r5105;
}
{
add.f16x2 r5106, r5082, r5103;
}
{
cvt.rn.f16.f64 rs805, fd891;
}
mov.b32 r5111, {rs805, rs805};
{
mul.f16x2 r5109, r172, r5111;
}
{
add.f16x2 r5112, r5088, r5109;
}
{
cvt.rn.f16.f64 rs806, fd892;
}
mov.b32 r5117, {rs806, rs806};
{
mul.f16x2 r5115, r175, r5117;
}
{
add.f16x2 r5118, r5094, r5115;
}
{
sub.f16x2 %26, r5100, r5106;
}
{
add.f16x2 %27, r5112, r5118;
}
{
add.f16x2 %36, r5100, r5106;
}
{
sub.f16x2 %37, r5112, r5118;
}
cvt.rn.f16.s32 rs807, r5508;
mov.b32 r5145, {rs807, rs807};
cvt.rn.f16.s32 rs808, r5508;
mov.b32 r5157, {rs808, rs808};
{
cvt.rn.f16.f64 rs809, fd851;
}
mov.b32 r5137, {rs809, rs809};
{
mul.f16x2 r5135, r1, r5137;
}
{
add.f16x2 r5138, %62, r5135;
}
{
cvt.rn.f16.f64 rs810, fd852;
}
mov.b32 r5143, {rs810, rs810};
{
mul.f16x2 r5141, r10, r5143;
}
{
add.f16x2 r5144, r5145, r5141;
}
{
cvt.rn.f16.f64 rs811, fd851;
}
mov.b32 r5149, {rs811, rs811};
{
mul.f16x2 r5147, r4, r5149;
}
{
add.f16x2 r5150, %63, r5147;
}
{
cvt.rn.f16.f64 rs812, fd852;
}
mov.b32 r5155, {rs812, rs812};
{
mul.f16x2 r5153, r7, r5155;
}
{
add.f16x2 r5156, r5157, r5153;
}
{
cvt.rn.f16.f64 rs813, fd863;
}
mov.b32 r5161, {rs813, rs813};
{
mul.f16x2 r5159, r13, r5161;
}
{
add.f16x2 r5162, r5138, r5159;
}
{
cvt.rn.f16.f64 rs814, fd864;
}
mov.b32 r5167, {rs814, rs814};
{
mul.f16x2 r5165, r22, r5167;
}
{
add.f16x2 r5168, r5144, r5165;
}
{
cvt.rn.f16.f64 rs815, fd863;
}
mov.b32 r5173, {rs815, rs815};
{
mul.f16x2 r5171, r16, r5173;
}
{
add.f16x2 r5174, r5150, r5171;
}
{
cvt.rn.f16.f64 rs816, fd864;
}
mov.b32 r5179, {rs816, rs816};
{
mul.f16x2 r5177, r19, r5179;
}
{
add.f16x2 r5180, r5156, r5177;
}
{
cvt.rn.f16.f64 rs817, fd875;
}
mov.b32 r5185, {rs817, rs817};
{
mul.f16x2 r5183, r25, r5185;
}
{
add.f16x2 r5186, r5162, r5183;
}
{
cvt.rn.f16.f64 rs818, fd876;
}
mov.b32 r5191, {rs818, rs818};
{
mul.f16x2 r5189, r34, r5191;
}
{
add.f16x2 r5192, r5168, r5189;
}
{
cvt.rn.f16.f64 rs819, fd875;
}
mov.b32 r5197, {rs819, rs819};
{
mul.f16x2 r5195, r28, r5197;
}
{
add.f16x2 r5198, r5174, r5195;
}
{
cvt.rn.f16.f64 rs820, fd876;
}
mov.b32 r5203, {rs820, rs820};
{
mul.f16x2 r5201, r31, r5203;
}
{
add.f16x2 r5204, r5180, r5201;
}
{
cvt.rn.f16.f64 rs821, fd887;
}
mov.b32 r5209, {rs821, rs821};
{
mul.f16x2 r5207, r37, r5209;
}
{
add.f16x2 r5210, r5186, r5207;
}
{
cvt.rn.f16.f64 rs822, fd888;
}
mov.b32 r5215, {rs822, rs822};
{
mul.f16x2 r5213, r46, r5215;
}
{
add.f16x2 r5216, r5192, r5213;
}
{
cvt.rn.f16.f64 rs823, fd887;
}
mov.b32 r5221, {rs823, rs823};
{
mul.f16x2 r5219, r40, r5221;
}
{
add.f16x2 r5222, r5198, r5219;
}
{
cvt.rn.f16.f64 rs824, fd888;
}
mov.b32 r5227, {rs824, rs824};
{
mul.f16x2 r5225, r43, r5227;
}
{
add.f16x2 r5228, r5204, r5225;
}
{
cvt.rn.f16.f64 rs825, fd899;
}
mov.b32 r5233, {rs825, rs825};
{
mul.f16x2 r5231, r49, r5233;
}
{
add.f16x2 r5234, r5210, r5231;
}
{
cvt.rn.f16.f64 rs826, fd900;
}
mov.b32 r5239, {rs826, rs826};
{
mul.f16x2 r5237, r58, r5239;
}
{
add.f16x2 r5240, r5216, r5237;
}
{
cvt.rn.f16.f64 rs827, fd899;
}
mov.b32 r5245, {rs827, rs827};
{
mul.f16x2 r5243, r52, r5245;
}
{
add.f16x2 r5246, r5222, r5243;
}
{
cvt.rn.f16.f64 rs828, fd900;
}
mov.b32 r5251, {rs828, rs828};
{
mul.f16x2 r5249, r55, r5251;
}
{
add.f16x2 r5252, r5228, r5249;
}
{
cvt.rn.f16.f64 rs829, fd891;
}
mov.b32 r5257, {rs829, rs829};
{
mul.f16x2 r5255, r61, r5257;
}
{
add.f16x2 r5258, r5234, r5255;
}
{
cvt.rn.f16.f64 rs830, fd804;
}
mov.b32 r5263, {rs830, rs830};
{
mul.f16x2 r5261, r70, r5263;
}
{
add.f16x2 r5264, r5240, r5261;
}
{
cvt.rn.f16.f64 rs831, fd891;
}
mov.b32 r5269, {rs831, rs831};
{
mul.f16x2 r5267, r64, r5269;
}
{
add.f16x2 r5270, r5246, r5267;
}
{
cvt.rn.f16.f64 rs832, fd804;
}
mov.b32 r5275, {rs832, rs832};
{
mul.f16x2 r5273, r67, r5275;
}
{
add.f16x2 r5276, r5252, r5273;
}
{
cvt.rn.f16.f64 rs833, fd879;
}
mov.b32 r5281, {rs833, rs833};
{
mul.f16x2 r5279, r73, r5281;
}
{
add.f16x2 r5282, r5258, r5279;
}
{
cvt.rn.f16.f64 rs834, fd808;
}
mov.b32 r5287, {rs834, rs834};
{
mul.f16x2 r5285, r82, r5287;
}
{
add.f16x2 r5288, r5264, r5285;
}
{
cvt.rn.f16.f64 rs835, fd879;
}
mov.b32 r5293, {rs835, rs835};
{
mul.f16x2 r5291, r76, r5293;
}
{
add.f16x2 r5294, r5270, r5291;
}
{
cvt.rn.f16.f64 rs836, fd808;
}
mov.b32 r5299, {rs836, rs836};
{
mul.f16x2 r5297, r79, r5299;
}
{
add.f16x2 r5300, r5276, r5297;
}
{
cvt.rn.f16.f64 rs837, fd867;
}
mov.b32 r5305, {rs837, rs837};
{
mul.f16x2 r5303, r85, r5305;
}
{
add.f16x2 r5306, r5282, r5303;
}
{
cvt.rn.f16.f64 rs838, fd812;
}
mov.b32 r5311, {rs838, rs838};
{
mul.f16x2 r5309, r94, r5311;
}
{
add.f16x2 r5312, r5288, r5309;
}
{
cvt.rn.f16.f64 rs839, fd867;
}
mov.b32 r5317, {rs839, rs839};
{
mul.f16x2 r5315, r88, r5317;
}
{
add.f16x2 r5318, r5294, r5315;
}
{
cvt.rn.f16.f64 rs840, fd812;
}
mov.b32 r5323, {rs840, rs840};
{
mul.f16x2 r5321, r91, r5323;
}
{
add.f16x2 r5324, r5300, r5321;
}
{
cvt.rn.f16.f64 rs841, fd855;
}
mov.b32 r5329, {rs841, rs841};
{
mul.f16x2 r5327, r97, r5329;
}
{
add.f16x2 r5330, r5306, r5327;
}
{
cvt.rn.f16.f64 rs842, fd816;
}
mov.b32 r5335, {rs842, rs842};
{
mul.f16x2 r5333, r106, r5335;
}
{
add.f16x2 r5336, r5312, r5333;
}
{
cvt.rn.f16.f64 rs843, fd855;
}
mov.b32 r5341, {rs843, rs843};
{
mul.f16x2 r5339, r100, r5341;
}
{
add.f16x2 r5342, r5318, r5339;
}
{
cvt.rn.f16.f64 rs844, fd816;
}
mov.b32 r5347, {rs844, rs844};
{
mul.f16x2 r5345, r103, r5347;
}
{
add.f16x2 r5348, r5324, r5345;
}
{
cvt.rn.f16.f64 rs845, fd843;
}
mov.b32 r5353, {rs845, rs845};
{
mul.f16x2 r5351, r109, r5353;
}
{
add.f16x2 r5354, r5330, r5351;
}
{
cvt.rn.f16.f64 rs846, fd820;
}
mov.b32 r5359, {rs846, rs846};
{
mul.f16x2 r5357, r118, r5359;
}
{
add.f16x2 r5360, r5336, r5357;
}
{
cvt.rn.f16.f64 rs847, fd843;
}
mov.b32 r5365, {rs847, rs847};
{
mul.f16x2 r5363, r112, r5365;
}
{
add.f16x2 r5366, r5342, r5363;
}
{
cvt.rn.f16.f64 rs848, fd820;
}
mov.b32 r5371, {rs848, rs848};
{
mul.f16x2 r5369, r115, r5371;
}
{
add.f16x2 r5372, r5348, r5369;
}
{
cvt.rn.f16.f64 rs849, fd847;
}
mov.b32 r5377, {rs849, rs849};
{
mul.f16x2 r5375, r121, r5377;
}
{
add.f16x2 r5378, r5354, r5375;
}
{
cvt.rn.f16.f64 rs850, fd848;
}
mov.b32 r5383, {rs850, rs850};
{
mul.f16x2 r5381, r130, r5383;
}
{
add.f16x2 r5384, r5360, r5381;
}
{
cvt.rn.f16.f64 rs851, fd847;
}
mov.b32 r5389, {rs851, rs851};
{
mul.f16x2 r5387, r124, r5389;
}
{
add.f16x2 r5390, r5366, r5387;
}
{
cvt.rn.f16.f64 rs852, fd848;
}
mov.b32 r5395, {rs852, rs852};
{
mul.f16x2 r5393, r127, r5395;
}
{
add.f16x2 r5396, r5372, r5393;
}
{
cvt.rn.f16.f64 rs853, fd859;
}
mov.b32 r5401, {rs853, rs853};
{
mul.f16x2 r5399, r133, r5401;
}
{
add.f16x2 r5402, r5378, r5399;
}
{
cvt.rn.f16.f64 rs854, fd860;
}
mov.b32 r5407, {rs854, rs854};
{
mul.f16x2 r5405, r142, r5407;
}
{
add.f16x2 r5408, r5384, r5405;
}
{
cvt.rn.f16.f64 rs855, fd859;
}
mov.b32 r5413, {rs855, rs855};
{
mul.f16x2 r5411, r136, r5413;
}
{
add.f16x2 r5414, r5390, r5411;
}
{
cvt.rn.f16.f64 rs856, fd860;
}
mov.b32 r5419, {rs856, rs856};
{
mul.f16x2 r5417, r139, r5419;
}
{
add.f16x2 r5420, r5396, r5417;
}
{
cvt.rn.f16.f64 rs857, fd871;
}
mov.b32 r5425, {rs857, rs857};
{
mul.f16x2 r5423, r145, r5425;
}
{
add.f16x2 r5426, r5402, r5423;
}
{
cvt.rn.f16.f64 rs858, fd872;
}
mov.b32 r5431, {rs858, rs858};
{
mul.f16x2 r5429, r154, r5431;
}
{
add.f16x2 r5432, r5408, r5429;
}
{
cvt.rn.f16.f64 rs859, fd871;
}
mov.b32 r5437, {rs859, rs859};
{
mul.f16x2 r5435, r148, r5437;
}
{
add.f16x2 r5438, r5414, r5435;
}
{
cvt.rn.f16.f64 rs860, fd872;
}
mov.b32 r5443, {rs860, rs860};
{
mul.f16x2 r5441, r151, r5443;
}
{
add.f16x2 r5444, r5420, r5441;
}
{
cvt.rn.f16.f64 rs861, fd883;
}
mov.b32 r5449, {rs861, rs861};
{
mul.f16x2 r5447, r157, r5449;
}
{
add.f16x2 r5450, r5426, r5447;
}
{
cvt.rn.f16.f64 rs862, fd884;
}
mov.b32 r5455, {rs862, rs862};
{
mul.f16x2 r5453, r166, r5455;
}
{
add.f16x2 r5456, r5432, r5453;
}
{
cvt.rn.f16.f64 rs863, fd883;
}
mov.b32 r5461, {rs863, rs863};
{
mul.f16x2 r5459, r160, r5461;
}
{
add.f16x2 r5462, r5438, r5459;
}
{
cvt.rn.f16.f64 rs864, fd884;
}
mov.b32 r5467, {rs864, rs864};
{
mul.f16x2 r5465, r163, r5467;
}
{
add.f16x2 r5468, r5444, r5465;
}
{
cvt.rn.f16.f64 rs865, fd895;
}
mov.b32 r5473, {rs865, rs865};
{
mul.f16x2 r5471, r169, r5473;
}
{
add.f16x2 r5474, r5450, r5471;
}
{
cvt.rn.f16.f64 rs866, fd896;
}
mov.b32 r5479, {rs866, rs866};
{
mul.f16x2 r5477, r178, r5479;
}
{
add.f16x2 r5480, r5456, r5477;
}
{
cvt.rn.f16.f64 rs867, fd895;
}
mov.b32 r5485, {rs867, rs867};
{
mul.f16x2 r5483, r172, r5485;
}
{
add.f16x2 r5486, r5462, r5483;
}
{
cvt.rn.f16.f64 rs868, fd896;
}
mov.b32 r5491, {rs868, rs868};
{
mul.f16x2 r5489, r175, r5491;
}
{
add.f16x2 r5492, r5468, r5489;
}
{
sub.f16x2 %28, r5474, r5480;
}
{
add.f16x2 %29, r5486, r5492;
}
{
add.f16x2 %34, r5474, r5480;
}
{
sub.f16x2 %35, r5486, r5492;
}
cvt.rn.f16.s32 rs869, r5508;
mov.b32 r5519, {rs869, rs869};
cvt.rn.f16.s32 rs870, r5508;
mov.b32 r5531, {rs870, rs870};
{
cvt.rn.f16.f64 rs871, fd843;
}
mov.b32 r5511, {rs871, rs871};
{
mul.f16x2 r5509, r1, r5511;
}
{
add.f16x2 r5512, %62, r5509;
}
{
cvt.rn.f16.f64 rs872, fd844;
}
mov.b32 r5517, {rs872, rs872};
{
mul.f16x2 r5515, r10, r5517;
}
{
add.f16x2 r5518, r5519, r5515;
}
{
cvt.rn.f16.f64 rs873, fd843;
}
mov.b32 r5523, {rs873, rs873};
{
mul.f16x2 r5521, r4, r5523;
}
{
add.f16x2 r5524, %63, r5521;
}
{
cvt.rn.f16.f64 rs874, fd844;
}
mov.b32 r5529, {rs874, rs874};
{
mul.f16x2 r5527, r7, r5529;
}
{
add.f16x2 r5530, r5531, r5527;
}
{
cvt.rn.f16.f64 rs875, fd847;
}
mov.b32 r5535, {rs875, rs875};
{
mul.f16x2 r5533, r13, r5535;
}
{
add.f16x2 r5536, r5512, r5533;
}
{
cvt.rn.f16.f64 rs876, fd848;
}
mov.b32 r5541, {rs876, rs876};
{
mul.f16x2 r5539, r22, r5541;
}
{
add.f16x2 r5542, r5518, r5539;
}
{
cvt.rn.f16.f64 rs877, fd847;
}
mov.b32 r5547, {rs877, rs877};
{
mul.f16x2 r5545, r16, r5547;
}
{
add.f16x2 r5548, r5524, r5545;
}
{
cvt.rn.f16.f64 rs878, fd848;
}
mov.b32 r5553, {rs878, rs878};
{
mul.f16x2 r5551, r19, r5553;
}
{
add.f16x2 r5554, r5530, r5551;
}
{
cvt.rn.f16.f64 rs879, fd851;
}
mov.b32 r5559, {rs879, rs879};
{
mul.f16x2 r5557, r25, r5559;
}
{
add.f16x2 r5560, r5536, r5557;
}
{
cvt.rn.f16.f64 rs880, fd852;
}
mov.b32 r5565, {rs880, rs880};
{
mul.f16x2 r5563, r34, r5565;
}
{
add.f16x2 r5566, r5542, r5563;
}
{
cvt.rn.f16.f64 rs881, fd851;
}
mov.b32 r5571, {rs881, rs881};
{
mul.f16x2 r5569, r28, r5571;
}
{
add.f16x2 r5572, r5548, r5569;
}
{
cvt.rn.f16.f64 rs882, fd852;
}
mov.b32 r5577, {rs882, rs882};
{
mul.f16x2 r5575, r31, r5577;
}
{
add.f16x2 r5578, r5554, r5575;
}
{
cvt.rn.f16.f64 rs883, fd855;
}
mov.b32 r5583, {rs883, rs883};
{
mul.f16x2 r5581, r37, r5583;
}
{
add.f16x2 r5584, r5560, r5581;
}
{
cvt.rn.f16.f64 rs884, fd856;
}
mov.b32 r5589, {rs884, rs884};
{
mul.f16x2 r5587, r46, r5589;
}
{
add.f16x2 r5590, r5566, r5587;
}
{
cvt.rn.f16.f64 rs885, fd855;
}
mov.b32 r5595, {rs885, rs885};
{
mul.f16x2 r5593, r40, r5595;
}
{
add.f16x2 r5596, r5572, r5593;
}
{
cvt.rn.f16.f64 rs886, fd856;
}
mov.b32 r5601, {rs886, rs886};
{
mul.f16x2 r5599, r43, r5601;
}
{
add.f16x2 r5602, r5578, r5599;
}
{
cvt.rn.f16.f64 rs887, fd859;
}
mov.b32 r5607, {rs887, rs887};
{
mul.f16x2 r5605, r49, r5607;
}
{
add.f16x2 r5608, r5584, r5605;
}
{
cvt.rn.f16.f64 rs888, fd860;
}
mov.b32 r5613, {rs888, rs888};
{
mul.f16x2 r5611, r58, r5613;
}
{
add.f16x2 r5614, r5590, r5611;
}
{
cvt.rn.f16.f64 rs889, fd859;
}
mov.b32 r5619, {rs889, rs889};
{
mul.f16x2 r5617, r52, r5619;
}
{
add.f16x2 r5620, r5596, r5617;
}
{
cvt.rn.f16.f64 rs890, fd860;
}
mov.b32 r5625, {rs890, rs890};
{
mul.f16x2 r5623, r55, r5625;
}
{
add.f16x2 r5626, r5602, r5623;
}
{
cvt.rn.f16.f64 rs891, fd863;
}
mov.b32 r5631, {rs891, rs891};
{
mul.f16x2 r5629, r61, r5631;
}
{
add.f16x2 r5632, r5608, r5629;
}
{
cvt.rn.f16.f64 rs892, fd864;
}
mov.b32 r5637, {rs892, rs892};
{
mul.f16x2 r5635, r70, r5637;
}
{
add.f16x2 r5638, r5614, r5635;
}
{
cvt.rn.f16.f64 rs893, fd863;
}
mov.b32 r5643, {rs893, rs893};
{
mul.f16x2 r5641, r64, r5643;
}
{
add.f16x2 r5644, r5620, r5641;
}
{
cvt.rn.f16.f64 rs894, fd864;
}
mov.b32 r5649, {rs894, rs894};
{
mul.f16x2 r5647, r67, r5649;
}
{
add.f16x2 r5650, r5626, r5647;
}
{
cvt.rn.f16.f64 rs895, fd867;
}
mov.b32 r5655, {rs895, rs895};
{
mul.f16x2 r5653, r73, r5655;
}
{
add.f16x2 r5656, r5632, r5653;
}
{
cvt.rn.f16.f64 rs896, fd868;
}
mov.b32 r5661, {rs896, rs896};
{
mul.f16x2 r5659, r82, r5661;
}
{
add.f16x2 r5662, r5638, r5659;
}
{
cvt.rn.f16.f64 rs897, fd867;
}
mov.b32 r5667, {rs897, rs897};
{
mul.f16x2 r5665, r76, r5667;
}
{
add.f16x2 r5668, r5644, r5665;
}
{
cvt.rn.f16.f64 rs898, fd868;
}
mov.b32 r5673, {rs898, rs898};
{
mul.f16x2 r5671, r79, r5673;
}
{
add.f16x2 r5674, r5650, r5671;
}
{
cvt.rn.f16.f64 rs899, fd871;
}
mov.b32 r5679, {rs899, rs899};
{
mul.f16x2 r5677, r85, r5679;
}
{
add.f16x2 r5680, r5656, r5677;
}
{
cvt.rn.f16.f64 rs900, fd872;
}
mov.b32 r5685, {rs900, rs900};
{
mul.f16x2 r5683, r94, r5685;
}
{
add.f16x2 r5686, r5662, r5683;
}
{
cvt.rn.f16.f64 rs901, fd871;
}
mov.b32 r5691, {rs901, rs901};
{
mul.f16x2 r5689, r88, r5691;
}
{
add.f16x2 r5692, r5668, r5689;
}
{
cvt.rn.f16.f64 rs902, fd872;
}
mov.b32 r5697, {rs902, rs902};
{
mul.f16x2 r5695, r91, r5697;
}
{
add.f16x2 r5698, r5674, r5695;
}
{
cvt.rn.f16.f64 rs903, fd875;
}
mov.b32 r5703, {rs903, rs903};
{
mul.f16x2 r5701, r97, r5703;
}
{
add.f16x2 r5704, r5680, r5701;
}
{
cvt.rn.f16.f64 rs904, fd876;
}
mov.b32 r5709, {rs904, rs904};
{
mul.f16x2 r5707, r106, r5709;
}
{
add.f16x2 r5710, r5686, r5707;
}
{
cvt.rn.f16.f64 rs905, fd875;
}
mov.b32 r5715, {rs905, rs905};
{
mul.f16x2 r5713, r100, r5715;
}
{
add.f16x2 r5716, r5692, r5713;
}
{
cvt.rn.f16.f64 rs906, fd876;
}
mov.b32 r5721, {rs906, rs906};
{
mul.f16x2 r5719, r103, r5721;
}
{
add.f16x2 r5722, r5698, r5719;
}
{
cvt.rn.f16.f64 rs907, fd879;
}
mov.b32 r5727, {rs907, rs907};
{
mul.f16x2 r5725, r109, r5727;
}
{
add.f16x2 r5728, r5704, r5725;
}
{
cvt.rn.f16.f64 rs908, fd880;
}
mov.b32 r5733, {rs908, rs908};
{
mul.f16x2 r5731, r118, r5733;
}
{
add.f16x2 r5734, r5710, r5731;
}
{
cvt.rn.f16.f64 rs909, fd879;
}
mov.b32 r5739, {rs909, rs909};
{
mul.f16x2 r5737, r112, r5739;
}
{
add.f16x2 r5740, r5716, r5737;
}
{
cvt.rn.f16.f64 rs910, fd880;
}
mov.b32 r5745, {rs910, rs910};
{
mul.f16x2 r5743, r115, r5745;
}
{
add.f16x2 r5746, r5722, r5743;
}
{
cvt.rn.f16.f64 rs911, fd883;
}
mov.b32 r5751, {rs911, rs911};
{
mul.f16x2 r5749, r121, r5751;
}
{
add.f16x2 r5752, r5728, r5749;
}
{
cvt.rn.f16.f64 rs912, fd884;
}
mov.b32 r5757, {rs912, rs912};
{
mul.f16x2 r5755, r130, r5757;
}
{
add.f16x2 r5758, r5734, r5755;
}
{
cvt.rn.f16.f64 rs913, fd883;
}
mov.b32 r5763, {rs913, rs913};
{
mul.f16x2 r5761, r124, r5763;
}
{
add.f16x2 r5764, r5740, r5761;
}
{
cvt.rn.f16.f64 rs914, fd884;
}
mov.b32 r5769, {rs914, rs914};
{
mul.f16x2 r5767, r127, r5769;
}
{
add.f16x2 r5770, r5746, r5767;
}
{
cvt.rn.f16.f64 rs915, fd887;
}
mov.b32 r5775, {rs915, rs915};
{
mul.f16x2 r5773, r133, r5775;
}
{
add.f16x2 r5776, r5752, r5773;
}
{
cvt.rn.f16.f64 rs916, fd888;
}
mov.b32 r5781, {rs916, rs916};
{
mul.f16x2 r5779, r142, r5781;
}
{
add.f16x2 r5782, r5758, r5779;
}
{
cvt.rn.f16.f64 rs917, fd887;
}
mov.b32 r5787, {rs917, rs917};
{
mul.f16x2 r5785, r136, r5787;
}
{
add.f16x2 r5788, r5764, r5785;
}
{
cvt.rn.f16.f64 rs918, fd888;
}
mov.b32 r5793, {rs918, rs918};
{
mul.f16x2 r5791, r139, r5793;
}
{
add.f16x2 r5794, r5770, r5791;
}
{
cvt.rn.f16.f64 rs919, fd891;
}
mov.b32 r5799, {rs919, rs919};
{
mul.f16x2 r5797, r145, r5799;
}
{
add.f16x2 r5800, r5776, r5797;
}
{
cvt.rn.f16.f64 rs920, fd892;
}
mov.b32 r5805, {rs920, rs920};
{
mul.f16x2 r5803, r154, r5805;
}
{
add.f16x2 r5806, r5782, r5803;
}
{
cvt.rn.f16.f64 rs921, fd891;
}
mov.b32 r5811, {rs921, rs921};
{
mul.f16x2 r5809, r148, r5811;
}
{
add.f16x2 r5812, r5788, r5809;
}
{
cvt.rn.f16.f64 rs922, fd892;
}
mov.b32 r5817, {rs922, rs922};
{
mul.f16x2 r5815, r151, r5817;
}
{
add.f16x2 r5818, r5794, r5815;
}
{
cvt.rn.f16.f64 rs923, fd895;
}
mov.b32 r5823, {rs923, rs923};
{
mul.f16x2 r5821, r157, r5823;
}
{
add.f16x2 r5824, r5800, r5821;
}
{
cvt.rn.f16.f64 rs924, fd896;
}
mov.b32 r5829, {rs924, rs924};
{
mul.f16x2 r5827, r166, r5829;
}
{
add.f16x2 r5830, r5806, r5827;
}
{
cvt.rn.f16.f64 rs925, fd895;
}
mov.b32 r5835, {rs925, rs925};
{
mul.f16x2 r5833, r160, r5835;
}
{
add.f16x2 r5836, r5812, r5833;
}
{
cvt.rn.f16.f64 rs926, fd896;
}
mov.b32 r5841, {rs926, rs926};
{
mul.f16x2 r5839, r163, r5841;
}
{
add.f16x2 r5842, r5818, r5839;
}
{
cvt.rn.f16.f64 rs927, fd899;
}
mov.b32 r5847, {rs927, rs927};
{
mul.f16x2 r5845, r169, r5847;
}
{
add.f16x2 r5848, r5824, r5845;
}
{
cvt.rn.f16.f64 rs928, fd900;
}
mov.b32 r5853, {rs928, rs928};
{
mul.f16x2 r5851, r178, r5853;
}
{
add.f16x2 r5854, r5830, r5851;
}
{
cvt.rn.f16.f64 rs929, fd899;
}
mov.b32 r5859, {rs929, rs929};
{
mul.f16x2 r5857, r172, r5859;
}
{
add.f16x2 r5860, r5836, r5857;
}
{
cvt.rn.f16.f64 rs930, fd900;
}
mov.b32 r5865, {rs930, rs930};
{
mul.f16x2 r5863, r175, r5865;
}
{
add.f16x2 r5866, r5842, r5863;
}
{
sub.f16x2 %30, r5848, r5854;
}
{
add.f16x2 %31, r5860, r5866;
}
{
add.f16x2 %32, r5848, r5854;
}
{
sub.f16x2 %33, r5860, r5866;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)), "=r"(__HALF2_TO_UI(rmem[27].x)), "=r"(__HALF2_TO_UI(rmem[27].y)), "=r"(__HALF2_TO_UI(rmem[28].x)), "=r"(__HALF2_TO_UI(rmem[28].y)), "=r"(__HALF2_TO_UI(rmem[29].x)), "=r"(__HALF2_TO_UI(rmem[29].y)), "=r"(__HALF2_TO_UI(rmem[30].x)), "=r"(__HALF2_TO_UI(rmem[30].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[27].x)), "r"(__HALF2_TO_UI(rmem[27].y)), "r"(__HALF2_TO_UI(rmem[28].x)), "r"(__HALF2_TO_UI(rmem[28].y)), "r"(__HALF2_TO_UI(rmem[29].x)), "r"(__HALF2_TO_UI(rmem[29].y)), "r"(__HALF2_TO_UI(rmem[30].x)), "r"(__HALF2_TO_UI(rmem[30].y)));
};


#endif
