#ifndef CUFFTDX_FFT_24_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_24_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<958, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<147>;
.reg .b32 r<1575>;
.reg .f64 fd<147>;
.reg .b64 rd<3>;
mov.f64 fd115, 0dBFE0000000000000;
{
cvt.rn.f16.f64 rs1, fd115;
}
mov.b32 r72, {rs1, rs1};
mov.f64 fd119, 0dBFEBB67AE8584CAA;
{
cvt.rn.f16.f64 rs2, fd119;
}
mov.b32 r81, {rs2, rs2};
{
add.f16x2 r1, %50, %48;
}
{
add.f16x2 r4, %51, r1;
}
{
add.f16x2 r7, %52, %49;
}
{
add.f16x2 r10, %53, r7;
}
{
add.f16x2 r13, %50, %48;
}
{
mul.f16x2 r16, r13, r72;
}
{
add.f16x2 r19, %51, r16;
}
{
sub.f16x2 r22, %52, %49;
}
{
mul.f16x2 r25, r22, r81;
}
{
add.f16x2 r28, r19, r25;
}
{
add.f16x2 r31, %50, %48;
}
{
mul.f16x2 r34, r31, r72;
}
{
add.f16x2 r37, %51, r34;
}
{
sub.f16x2 r40, %52, %49;
}
{
mul.f16x2 r43, r40, r81;
}
{
sub.f16x2 r46, r37, r43;
}
{
add.f16x2 r49, %52, %49;
}
{
mul.f16x2 r52, r49, r72;
}
{
add.f16x2 r55, %53, r52;
}
{
sub.f16x2 r58, %50, %48;
}
{
mul.f16x2 r61, r58, r81;
}
{
sub.f16x2 r64, r55, r61;
}
{
add.f16x2 r67, %52, %49;
}
{
mul.f16x2 r70, r67, r72;
}
{
add.f16x2 r73, %53, r70;
}
{
sub.f16x2 r76, %50, %48;
}
{
mul.f16x2 r79, r76, r81;
}
{
add.f16x2 r82, r73, r79;
}
{
cvt.rn.f16.f64 rs3, fd115;
}
mov.b32 r156, {rs3, rs3};
{
cvt.rn.f16.f64 rs4, fd119;
}
mov.b32 r165, {rs4, rs4};
{
add.f16x2 r85, %55, %59;
}
{
add.f16x2 r88, %56, r85;
}
{
add.f16x2 r91, %57, %54;
}
{
add.f16x2 r94, %58, r91;
}
{
add.f16x2 r97, %55, %59;
}
{
mul.f16x2 r100, r97, r156;
}
{
add.f16x2 r103, %56, r100;
}
{
sub.f16x2 r106, %57, %54;
}
{
mul.f16x2 r109, r106, r165;
}
{
add.f16x2 r112, r103, r109;
}
{
add.f16x2 r115, %55, %59;
}
{
mul.f16x2 r118, r115, r156;
}
{
add.f16x2 r121, %56, r118;
}
{
sub.f16x2 r124, %57, %54;
}
{
mul.f16x2 r127, r124, r165;
}
{
sub.f16x2 r130, r121, r127;
}
{
add.f16x2 r133, %57, %54;
}
{
mul.f16x2 r136, r133, r156;
}
{
add.f16x2 r139, %58, r136;
}
{
sub.f16x2 r142, %55, %59;
}
{
mul.f16x2 r145, r142, r165;
}
{
sub.f16x2 r148, r139, r145;
}
{
add.f16x2 r151, %57, %54;
}
{
mul.f16x2 r154, r151, r156;
}
{
add.f16x2 r157, %58, r154;
}
{
sub.f16x2 r160, %55, %59;
}
{
mul.f16x2 r163, r160, r165;
}
{
add.f16x2 r166, r157, r163;
}
mov.f64 fd120, 0d3FE0000000000000;
{
cvt.rn.f16.f64 rs5, fd120;
}
mov.f64 fd116, 0d3FEBB67AE8584CAA;
{
cvt.rn.f16.f64 rs6, fd116;
}
{
cvt.rn.f16.f64 rs7, fd115;
}
{
cvt.rn.f16.f64 rs8, fd116;
}
mov.b32 r183, {rs5, rs5};
{
mul.f16x2 r169, r112, r183;
}
mov.b32 r180, {rs6, rs6};
{
mul.f16x2 r172, r148, r180;
}
{
sub.f16x2 r175, r169, r172;
}
{
mul.f16x2 r178, r112, r180;
}
{
fma.rn.f16x2 r181, r148, r183, r178;
}
mov.b32 r199, {rs7, rs7};
{
mul.f16x2 r185, r130, r199;
}
mov.b32 r196, {rs8, rs8};
{
mul.f16x2 r188, r166, r196;
}
{
sub.f16x2 r191, r185, r188;
}
{
mul.f16x2 r194, r130, r196;
}
{
fma.rn.f16x2 r197, r166, r199, r194;
}
{
add.f16x2 r201, r4, r88;
}
{
add.f16x2 r204, r10, r94;
}
{
sub.f16x2 r207, r4, r88;
}
{
sub.f16x2 r210, r10, r94;
}
{
add.f16x2 r213, r28, r175;
}
{
add.f16x2 r216, r64, r181;
}
{
sub.f16x2 r219, r28, r175;
}
{
sub.f16x2 r222, r64, r181;
}
{
add.f16x2 r225, r46, r191;
}
{
add.f16x2 r228, r82, r197;
}
{
sub.f16x2 r231, r46, r191;
}
{
sub.f16x2 r234, r82, r197;
}
{
cvt.rn.f16.f64 rs15, fd115;
}
mov.b32 r308, {rs15, rs15};
{
cvt.rn.f16.f64 rs16, fd119;
}
mov.b32 r317, {rs16, rs16};
{
add.f16x2 r237, %61, %65;
}
{
add.f16x2 r240, %62, r237;
}
{
add.f16x2 r243, %63, %60;
}
{
add.f16x2 r246, %64, r243;
}
{
add.f16x2 r249, %61, %65;
}
{
mul.f16x2 r252, r249, r308;
}
{
add.f16x2 r255, %62, r252;
}
{
sub.f16x2 r258, %63, %60;
}
{
mul.f16x2 r261, r258, r317;
}
{
add.f16x2 r264, r255, r261;
}
{
add.f16x2 r267, %61, %65;
}
{
mul.f16x2 r270, r267, r308;
}
{
add.f16x2 r273, %62, r270;
}
{
sub.f16x2 r276, %63, %60;
}
{
mul.f16x2 r279, r276, r317;
}
{
sub.f16x2 r282, r273, r279;
}
{
add.f16x2 r285, %63, %60;
}
{
mul.f16x2 r288, r285, r308;
}
{
add.f16x2 r291, %64, r288;
}
{
sub.f16x2 r294, %61, %65;
}
{
mul.f16x2 r297, r294, r317;
}
{
sub.f16x2 r300, r291, r297;
}
{
add.f16x2 r303, %63, %60;
}
{
mul.f16x2 r306, r303, r308;
}
{
add.f16x2 r309, %64, r306;
}
{
sub.f16x2 r312, %61, %65;
}
{
mul.f16x2 r315, r312, r317;
}
{
add.f16x2 r318, r309, r315;
}
{
cvt.rn.f16.f64 rs17, fd115;
}
mov.b32 r392, {rs17, rs17};
{
cvt.rn.f16.f64 rs18, fd119;
}
mov.b32 r401, {rs18, rs18};
{
add.f16x2 r321, %70, %68;
}
{
add.f16x2 r324, %71, r321;
}
{
add.f16x2 r327, %66, %69;
}
{
add.f16x2 r330, %67, r327;
}
{
add.f16x2 r333, %70, %68;
}
{
mul.f16x2 r336, r333, r392;
}
{
add.f16x2 r339, %71, r336;
}
{
sub.f16x2 r342, %66, %69;
}
{
mul.f16x2 r345, r342, r401;
}
{
add.f16x2 r348, r339, r345;
}
{
add.f16x2 r351, %70, %68;
}
{
mul.f16x2 r354, r351, r392;
}
{
add.f16x2 r357, %71, r354;
}
{
sub.f16x2 r360, %66, %69;
}
{
mul.f16x2 r363, r360, r401;
}
{
sub.f16x2 r366, r357, r363;
}
{
add.f16x2 r369, %66, %69;
}
{
mul.f16x2 r372, r369, r392;
}
{
add.f16x2 r375, %67, r372;
}
{
sub.f16x2 r378, %70, %68;
}
{
mul.f16x2 r381, r378, r401;
}
{
sub.f16x2 r384, r375, r381;
}
{
add.f16x2 r387, %66, %69;
}
{
mul.f16x2 r390, r387, r392;
}
{
add.f16x2 r393, %67, r390;
}
{
sub.f16x2 r396, %70, %68;
}
{
mul.f16x2 r399, r396, r401;
}
{
add.f16x2 r402, r393, r399;
}
{
cvt.rn.f16.f64 rs19, fd120;
}
{
cvt.rn.f16.f64 rs20, fd116;
}
{
cvt.rn.f16.f64 rs21, fd115;
}
{
cvt.rn.f16.f64 rs22, fd116;
}
mov.b32 r419, {rs19, rs19};
{
mul.f16x2 r405, r348, r419;
}
mov.b32 r416, {rs20, rs20};
{
mul.f16x2 r408, r384, r416;
}
{
sub.f16x2 r411, r405, r408;
}
{
mul.f16x2 r414, r348, r416;
}
{
fma.rn.f16x2 r417, r384, r419, r414;
}
mov.b32 r435, {rs21, rs21};
{
mul.f16x2 r421, r366, r435;
}
mov.b32 r432, {rs22, rs22};
{
mul.f16x2 r424, r402, r432;
}
{
sub.f16x2 r427, r421, r424;
}
{
mul.f16x2 r430, r366, r432;
}
{
fma.rn.f16x2 r433, r402, r435, r430;
}
{
add.f16x2 r437, r240, r324;
}
{
add.f16x2 r440, r246, r330;
}
{
sub.f16x2 r443, r240, r324;
}
{
sub.f16x2 r446, r246, r330;
}
{
add.f16x2 r449, r264, r411;
}
{
add.f16x2 r452, r300, r417;
}
{
sub.f16x2 r455, r264, r411;
}
{
sub.f16x2 r458, r300, r417;
}
{
add.f16x2 r461, r282, r427;
}
{
add.f16x2 r464, r318, r433;
}
{
sub.f16x2 r467, r282, r427;
}
{
sub.f16x2 r470, r318, r433;
}
{
cvt.rn.f16.f64 rs29, fd116;
}
{
cvt.rn.f16.f64 rs30, fd120;
}
{
cvt.rn.f16.f64 rs31, fd120;
}
{
cvt.rn.f16.f64 rs32, fd116;
}
{
cvt.rn.f16.f64 rs35, fd115;
}
{
cvt.rn.f16.f64 rs36, fd116;
}
{
cvt.rn.f16.f64 rs37, fd119;
}
{
cvt.rn.f16.f64 rs38, fd120;
}
mov.b32 r487, {rs29, rs29};
{
mul.f16x2 r473, r449, r487;
}
mov.b32 r484, {rs30, rs30};
{
mul.f16x2 r476, r452, r484;
}
{
sub.f16x2 r479, r473, r476;
}
{
mul.f16x2 r482, r449, r484;
}
{
fma.rn.f16x2 r485, r452, r487, r482;
}
mov.b32 r503, {rs31, rs31};
{
mul.f16x2 r489, r461, r503;
}
mov.b32 r500, {rs32, rs32};
{
mul.f16x2 r492, r464, r500;
}
{
sub.f16x2 r495, r489, r492;
}
{
mul.f16x2 r498, r461, r500;
}
{
fma.rn.f16x2 r501, r464, r503, r498;
}
{
neg.f16x2 r505, r446;
}
mov.b32 r521, {rs35, rs35};
{
mul.f16x2 r507, r455, r521;
}
mov.b32 r518, {rs36, rs36};
{
mul.f16x2 r510, r458, r518;
}
{
sub.f16x2 r513, r507, r510;
}
{
mul.f16x2 r516, r455, r518;
}
{
fma.rn.f16x2 r519, r458, r521, r516;
}
mov.b32 r537, {rs37, rs37};
{
mul.f16x2 r523, r467, r537;
}
mov.b32 r534, {rs38, rs38};
{
mul.f16x2 r526, r470, r534;
}
{
sub.f16x2 r529, r523, r526;
}
{
mul.f16x2 r532, r467, r534;
}
{
fma.rn.f16x2 r535, r470, r537, r532;
}
{
add.f16x2 r539, r201, r437;
}
{
add.f16x2 r542, r204, r440;
}
{
sub.f16x2 r545, r201, r437;
}
{
sub.f16x2 r548, r204, r440;
}
{
add.f16x2 r551, r213, r479;
}
{
add.f16x2 r554, r216, r485;
}
{
sub.f16x2 r557, r213, r479;
}
{
sub.f16x2 r560, r216, r485;
}
{
add.f16x2 r563, r225, r495;
}
{
add.f16x2 r566, r228, r501;
}
{
sub.f16x2 r569, r225, r495;
}
{
sub.f16x2 r572, r228, r501;
}
{
add.f16x2 r575, r207, r505;
}
{
add.f16x2 r578, r210, r443;
}
{
sub.f16x2 r581, r207, r505;
}
{
sub.f16x2 r584, r210, r443;
}
{
add.f16x2 r587, r219, r513;
}
{
add.f16x2 r590, r222, r519;
}
{
sub.f16x2 r593, r219, r513;
}
{
sub.f16x2 r596, r222, r519;
}
{
add.f16x2 r599, r231, r529;
}
{
add.f16x2 r602, r234, r535;
}
{
sub.f16x2 r605, r231, r529;
}
{
sub.f16x2 r608, r234, r535;
}
{
cvt.rn.f16.f64 rs51, fd115;
}
mov.b32 r682, {rs51, rs51};
{
cvt.rn.f16.f64 rs52, fd119;
}
mov.b32 r691, {rs52, rs52};
{
add.f16x2 r611, %74, %72;
}
{
add.f16x2 r614, %75, r611;
}
{
add.f16x2 r617, %76, %73;
}
{
add.f16x2 r620, %77, r617;
}
{
add.f16x2 r623, %74, %72;
}
{
mul.f16x2 r626, r623, r682;
}
{
add.f16x2 r629, %75, r626;
}
{
sub.f16x2 r632, %76, %73;
}
{
mul.f16x2 r635, r632, r691;
}
{
add.f16x2 r638, r629, r635;
}
{
add.f16x2 r641, %74, %72;
}
{
mul.f16x2 r644, r641, r682;
}
{
add.f16x2 r647, %75, r644;
}
{
sub.f16x2 r650, %76, %73;
}
{
mul.f16x2 r653, r650, r691;
}
{
sub.f16x2 r656, r647, r653;
}
{
add.f16x2 r659, %76, %73;
}
{
mul.f16x2 r662, r659, r682;
}
{
add.f16x2 r665, %77, r662;
}
{
sub.f16x2 r668, %74, %72;
}
{
mul.f16x2 r671, r668, r691;
}
{
sub.f16x2 r674, r665, r671;
}
{
add.f16x2 r677, %76, %73;
}
{
mul.f16x2 r680, r677, r682;
}
{
add.f16x2 r683, %77, r680;
}
{
sub.f16x2 r686, %74, %72;
}
{
mul.f16x2 r689, r686, r691;
}
{
add.f16x2 r692, r683, r689;
}
{
cvt.rn.f16.f64 rs53, fd115;
}
mov.b32 r766, {rs53, rs53};
{
cvt.rn.f16.f64 rs54, fd119;
}
mov.b32 r775, {rs54, rs54};
{
add.f16x2 r695, %83, %81;
}
{
add.f16x2 r698, %78, r695;
}
{
add.f16x2 r701, %79, %82;
}
{
add.f16x2 r704, %80, r701;
}
{
add.f16x2 r707, %83, %81;
}
{
mul.f16x2 r710, r707, r766;
}
{
add.f16x2 r713, %78, r710;
}
{
sub.f16x2 r716, %79, %82;
}
{
mul.f16x2 r719, r716, r775;
}
{
add.f16x2 r722, r713, r719;
}
{
add.f16x2 r725, %83, %81;
}
{
mul.f16x2 r728, r725, r766;
}
{
add.f16x2 r731, %78, r728;
}
{
sub.f16x2 r734, %79, %82;
}
{
mul.f16x2 r737, r734, r775;
}
{
sub.f16x2 r740, r731, r737;
}
{
add.f16x2 r743, %79, %82;
}
{
mul.f16x2 r746, r743, r766;
}
{
add.f16x2 r749, %80, r746;
}
{
sub.f16x2 r752, %83, %81;
}
{
mul.f16x2 r755, r752, r775;
}
{
sub.f16x2 r758, r749, r755;
}
{
add.f16x2 r761, %79, %82;
}
{
mul.f16x2 r764, r761, r766;
}
{
add.f16x2 r767, %80, r764;
}
{
sub.f16x2 r770, %83, %81;
}
{
mul.f16x2 r773, r770, r775;
}
{
add.f16x2 r776, r767, r773;
}
{
cvt.rn.f16.f64 rs55, fd120;
}
{
cvt.rn.f16.f64 rs56, fd116;
}
{
cvt.rn.f16.f64 rs57, fd115;
}
{
cvt.rn.f16.f64 rs58, fd116;
}
mov.b32 r793, {rs55, rs55};
{
mul.f16x2 r779, r722, r793;
}
mov.b32 r790, {rs56, rs56};
{
mul.f16x2 r782, r758, r790;
}
{
sub.f16x2 r785, r779, r782;
}
{
mul.f16x2 r788, r722, r790;
}
{
fma.rn.f16x2 r791, r758, r793, r788;
}
mov.b32 r809, {rs57, rs57};
{
mul.f16x2 r795, r740, r809;
}
mov.b32 r806, {rs58, rs58};
{
mul.f16x2 r798, r776, r806;
}
{
sub.f16x2 r801, r795, r798;
}
{
mul.f16x2 r804, r740, r806;
}
{
fma.rn.f16x2 r807, r776, r809, r804;
}
{
add.f16x2 r811, r614, r698;
}
{
add.f16x2 r814, r620, r704;
}
{
sub.f16x2 r817, r614, r698;
}
{
sub.f16x2 r820, r620, r704;
}
{
add.f16x2 r823, r638, r785;
}
{
add.f16x2 r826, r674, r791;
}
{
sub.f16x2 r829, r638, r785;
}
{
sub.f16x2 r832, r674, r791;
}
{
add.f16x2 r835, r656, r801;
}
{
add.f16x2 r838, r692, r807;
}
{
sub.f16x2 r841, r656, r801;
}
{
sub.f16x2 r844, r692, r807;
}
{
cvt.rn.f16.f64 rs65, fd115;
}
mov.b32 r918, {rs65, rs65};
{
cvt.rn.f16.f64 rs66, fd119;
}
mov.b32 r927, {rs66, rs66};
{
add.f16x2 r847, %84, %88;
}
{
add.f16x2 r850, %85, r847;
}
{
add.f16x2 r853, %86, %89;
}
{
add.f16x2 r856, %87, r853;
}
{
add.f16x2 r859, %84, %88;
}
{
mul.f16x2 r862, r859, r918;
}
{
add.f16x2 r865, %85, r862;
}
{
sub.f16x2 r868, %86, %89;
}
{
mul.f16x2 r871, r868, r927;
}
{
add.f16x2 r874, r865, r871;
}
{
add.f16x2 r877, %84, %88;
}
{
mul.f16x2 r880, r877, r918;
}
{
add.f16x2 r883, %85, r880;
}
{
sub.f16x2 r886, %86, %89;
}
{
mul.f16x2 r889, r886, r927;
}
{
sub.f16x2 r892, r883, r889;
}
{
add.f16x2 r895, %86, %89;
}
{
mul.f16x2 r898, r895, r918;
}
{
add.f16x2 r901, %87, r898;
}
{
sub.f16x2 r904, %84, %88;
}
{
mul.f16x2 r907, r904, r927;
}
{
sub.f16x2 r910, r901, r907;
}
{
add.f16x2 r913, %86, %89;
}
{
mul.f16x2 r916, r913, r918;
}
{
add.f16x2 r919, %87, r916;
}
{
sub.f16x2 r922, %84, %88;
}
{
mul.f16x2 r925, r922, r927;
}
{
add.f16x2 r928, r919, r925;
}
{
cvt.rn.f16.f64 rs67, fd115;
}
mov.b32 r1002, {rs67, rs67};
{
cvt.rn.f16.f64 rs68, fd119;
}
mov.b32 r1011, {rs68, rs68};
{
add.f16x2 r931, %92, %90;
}
{
add.f16x2 r934, %93, r931;
}
{
add.f16x2 r937, %94, %91;
}
{
add.f16x2 r940, %95, r937;
}
{
add.f16x2 r943, %92, %90;
}
{
mul.f16x2 r946, r943, r1002;
}
{
add.f16x2 r949, %93, r946;
}
{
sub.f16x2 r952, %94, %91;
}
{
mul.f16x2 r955, r952, r1011;
}
{
add.f16x2 r958, r949, r955;
}
{
add.f16x2 r961, %92, %90;
}
{
mul.f16x2 r964, r961, r1002;
}
{
add.f16x2 r967, %93, r964;
}
{
sub.f16x2 r970, %94, %91;
}
{
mul.f16x2 r973, r970, r1011;
}
{
sub.f16x2 r976, r967, r973;
}
{
add.f16x2 r979, %94, %91;
}
{
mul.f16x2 r982, r979, r1002;
}
{
add.f16x2 r985, %95, r982;
}
{
sub.f16x2 r988, %92, %90;
}
{
mul.f16x2 r991, r988, r1011;
}
{
sub.f16x2 r994, r985, r991;
}
{
add.f16x2 r997, %94, %91;
}
{
mul.f16x2 r1000, r997, r1002;
}
{
add.f16x2 r1003, %95, r1000;
}
{
sub.f16x2 r1006, %92, %90;
}
{
mul.f16x2 r1009, r1006, r1011;
}
{
add.f16x2 r1012, r1003, r1009;
}
{
cvt.rn.f16.f64 rs69, fd120;
}
{
cvt.rn.f16.f64 rs70, fd116;
}
{
cvt.rn.f16.f64 rs71, fd115;
}
{
cvt.rn.f16.f64 rs72, fd116;
}
mov.b32 r1029, {rs69, rs69};
{
mul.f16x2 r1015, r958, r1029;
}
mov.b32 r1026, {rs70, rs70};
{
mul.f16x2 r1018, r994, r1026;
}
{
sub.f16x2 r1021, r1015, r1018;
}
{
mul.f16x2 r1024, r958, r1026;
}
{
fma.rn.f16x2 r1027, r994, r1029, r1024;
}
mov.b32 r1045, {rs71, rs71};
{
mul.f16x2 r1031, r976, r1045;
}
mov.b32 r1042, {rs72, rs72};
{
mul.f16x2 r1034, r1012, r1042;
}
{
sub.f16x2 r1037, r1031, r1034;
}
{
mul.f16x2 r1040, r976, r1042;
}
{
fma.rn.f16x2 r1043, r1012, r1045, r1040;
}
{
add.f16x2 r1047, r850, r934;
}
{
add.f16x2 r1050, r856, r940;
}
{
sub.f16x2 r1053, r850, r934;
}
{
sub.f16x2 r1056, r856, r940;
}
{
add.f16x2 r1059, r874, r1021;
}
{
add.f16x2 r1062, r910, r1027;
}
{
sub.f16x2 r1065, r874, r1021;
}
{
sub.f16x2 r1068, r910, r1027;
}
{
add.f16x2 r1071, r892, r1037;
}
{
add.f16x2 r1074, r928, r1043;
}
{
sub.f16x2 r1077, r892, r1037;
}
{
sub.f16x2 r1080, r928, r1043;
}
{
cvt.rn.f16.f64 rs79, fd116;
}
{
cvt.rn.f16.f64 rs80, fd120;
}
{
cvt.rn.f16.f64 rs81, fd120;
}
{
cvt.rn.f16.f64 rs82, fd116;
}
{
cvt.rn.f16.f64 rs85, fd115;
}
{
cvt.rn.f16.f64 rs86, fd116;
}
{
cvt.rn.f16.f64 rs87, fd119;
}
{
cvt.rn.f16.f64 rs88, fd120;
}
mov.b32 r1097, {rs79, rs79};
{
mul.f16x2 r1083, r1059, r1097;
}
mov.b32 r1094, {rs80, rs80};
{
mul.f16x2 r1086, r1062, r1094;
}
{
sub.f16x2 r1089, r1083, r1086;
}
{
mul.f16x2 r1092, r1059, r1094;
}
{
fma.rn.f16x2 r1095, r1062, r1097, r1092;
}
mov.b32 r1113, {rs81, rs81};
{
mul.f16x2 r1099, r1071, r1113;
}
mov.b32 r1110, {rs82, rs82};
{
mul.f16x2 r1102, r1074, r1110;
}
{
sub.f16x2 r1105, r1099, r1102;
}
{
mul.f16x2 r1108, r1071, r1110;
}
{
fma.rn.f16x2 r1111, r1074, r1113, r1108;
}
{
neg.f16x2 r1115, r1056;
}
mov.b32 r1131, {rs85, rs85};
{
mul.f16x2 r1117, r1065, r1131;
}
mov.b32 r1128, {rs86, rs86};
{
mul.f16x2 r1120, r1068, r1128;
}
{
sub.f16x2 r1123, r1117, r1120;
}
{
mul.f16x2 r1126, r1065, r1128;
}
{
fma.rn.f16x2 r1129, r1068, r1131, r1126;
}
mov.b32 r1147, {rs87, rs87};
{
mul.f16x2 r1133, r1077, r1147;
}
mov.b32 r1144, {rs88, rs88};
{
mul.f16x2 r1136, r1080, r1144;
}
{
sub.f16x2 r1139, r1133, r1136;
}
{
mul.f16x2 r1142, r1077, r1144;
}
{
fma.rn.f16x2 r1145, r1080, r1147, r1142;
}
{
add.f16x2 r1149, r811, r1047;
}
{
add.f16x2 r1152, r814, r1050;
}
{
sub.f16x2 r1155, r811, r1047;
}
{
sub.f16x2 r1158, r814, r1050;
}
{
add.f16x2 r1161, r823, r1089;
}
{
add.f16x2 r1164, r826, r1095;
}
{
sub.f16x2 r1167, r823, r1089;
}
{
sub.f16x2 r1170, r826, r1095;
}
{
add.f16x2 r1173, r835, r1105;
}
{
add.f16x2 r1176, r838, r1111;
}
{
sub.f16x2 r1179, r835, r1105;
}
{
sub.f16x2 r1182, r838, r1111;
}
{
add.f16x2 r1185, r817, r1115;
}
{
add.f16x2 r1188, r820, r1053;
}
{
sub.f16x2 r1191, r817, r1115;
}
{
sub.f16x2 r1194, r820, r1053;
}
{
add.f16x2 r1197, r829, r1123;
}
{
add.f16x2 r1200, r832, r1129;
}
{
sub.f16x2 r1203, r829, r1123;
}
{
sub.f16x2 r1206, r832, r1129;
}
{
add.f16x2 r1209, r841, r1139;
}
{
add.f16x2 r1212, r844, r1145;
}
{
sub.f16x2 r1215, r841, r1139;
}
{
sub.f16x2 r1218, r844, r1145;
}
mov.f64 fd114, 0d3FEEE8DD4748BF15;
{
cvt.rn.f16.f64 rs101, fd114;
}
mov.f64 fd122, 0d3FD0907DC1930690;
{
cvt.rn.f16.f64 rs102, fd122;
}
{
cvt.rn.f16.f64 rs103, fd116;
}
{
cvt.rn.f16.f64 rs104, fd120;
}
mov.f64 fd118, 0d3FE6A09E667F3BCD;
{
cvt.rn.f16.f64 rs105, fd118;
}
{
cvt.rn.f16.f64 rs106, fd118;
}
{
cvt.rn.f16.f64 rs107, fd120;
}
{
cvt.rn.f16.f64 rs108, fd116;
}
{
cvt.rn.f16.f64 rs109, fd122;
}
{
cvt.rn.f16.f64 rs110, fd114;
}
mov.f64 fd113, 0dBFD0907DC1930690;
{
cvt.rn.f16.f64 rs113, fd113;
}
{
cvt.rn.f16.f64 rs114, fd114;
}
{
cvt.rn.f16.f64 rs115, fd115;
}
{
cvt.rn.f16.f64 rs116, fd116;
}
mov.f64 fd117, 0dBFE6A09E667F3BCD;
{
cvt.rn.f16.f64 rs117, fd117;
}
{
cvt.rn.f16.f64 rs118, fd118;
}
{
cvt.rn.f16.f64 rs119, fd119;
}
{
cvt.rn.f16.f64 rs120, fd120;
}
mov.f64 fd121, 0dBFEEE8DD4748BF15;
{
cvt.rn.f16.f64 rs121, fd121;
}
{
cvt.rn.f16.f64 rs122, fd122;
}
mov.b32 r1235, {rs101, rs101};
{
mul.f16x2 r1221, r1161, r1235;
}
mov.b32 r1232, {rs102, rs102};
{
mul.f16x2 r1224, r1164, r1232;
}
{
sub.f16x2 r1227, r1221, r1224;
}
{
mul.f16x2 r1230, r1161, r1232;
}
{
fma.rn.f16x2 r1233, r1164, r1235, r1230;
}
mov.b32 r1251, {rs103, rs103};
{
mul.f16x2 r1237, r1173, r1251;
}
mov.b32 r1248, {rs104, rs104};
{
mul.f16x2 r1240, r1176, r1248;
}
{
sub.f16x2 r1243, r1237, r1240;
}
{
mul.f16x2 r1246, r1173, r1248;
}
{
fma.rn.f16x2 r1249, r1176, r1251, r1246;
}
mov.b32 r1267, {rs105, rs105};
{
mul.f16x2 r1253, r1185, r1267;
}
mov.b32 r1264, {rs106, rs106};
{
mul.f16x2 r1256, r1188, r1264;
}
{
sub.f16x2 r1259, r1253, r1256;
}
{
mul.f16x2 r1262, r1185, r1264;
}
{
fma.rn.f16x2 r1265, r1188, r1267, r1262;
}
mov.b32 r1283, {rs107, rs107};
{
mul.f16x2 r1269, r1197, r1283;
}
mov.b32 r1280, {rs108, rs108};
{
mul.f16x2 r1272, r1200, r1280;
}
{
sub.f16x2 r1275, r1269, r1272;
}
{
mul.f16x2 r1278, r1197, r1280;
}
{
fma.rn.f16x2 r1281, r1200, r1283, r1278;
}
mov.b32 r1299, {rs109, rs109};
{
mul.f16x2 r1285, r1209, r1299;
}
mov.b32 r1296, {rs110, rs110};
{
mul.f16x2 r1288, r1212, r1296;
}
{
sub.f16x2 r1291, r1285, r1288;
}
{
mul.f16x2 r1294, r1209, r1296;
}
{
fma.rn.f16x2 r1297, r1212, r1299, r1294;
}
{
neg.f16x2 r1301, r1158;
}
mov.b32 r1317, {rs113, rs113};
{
mul.f16x2 r1303, r1167, r1317;
}
mov.b32 r1314, {rs114, rs114};
{
mul.f16x2 r1306, r1170, r1314;
}
{
sub.f16x2 r1309, r1303, r1306;
}
{
mul.f16x2 r1312, r1167, r1314;
}
{
fma.rn.f16x2 r1315, r1170, r1317, r1312;
}
mov.b32 r1333, {rs115, rs115};
{
mul.f16x2 r1319, r1179, r1333;
}
mov.b32 r1330, {rs116, rs116};
{
mul.f16x2 r1322, r1182, r1330;
}
{
sub.f16x2 r1325, r1319, r1322;
}
{
mul.f16x2 r1328, r1179, r1330;
}
{
fma.rn.f16x2 r1331, r1182, r1333, r1328;
}
mov.b32 r1349, {rs117, rs117};
{
mul.f16x2 r1335, r1191, r1349;
}
mov.b32 r1346, {rs118, rs118};
{
mul.f16x2 r1338, r1194, r1346;
}
{
sub.f16x2 r1341, r1335, r1338;
}
{
mul.f16x2 r1344, r1191, r1346;
}
{
fma.rn.f16x2 r1347, r1194, r1349, r1344;
}
mov.b32 r1365, {rs119, rs119};
{
mul.f16x2 r1351, r1203, r1365;
}
mov.b32 r1362, {rs120, rs120};
{
mul.f16x2 r1354, r1206, r1362;
}
{
sub.f16x2 r1357, r1351, r1354;
}
{
mul.f16x2 r1360, r1203, r1362;
}
{
fma.rn.f16x2 r1363, r1206, r1365, r1360;
}
mov.b32 r1381, {rs121, rs121};
{
mul.f16x2 r1367, r1215, r1381;
}
mov.b32 r1378, {rs122, rs122};
{
mul.f16x2 r1370, r1218, r1378;
}
{
sub.f16x2 r1373, r1367, r1370;
}
{
mul.f16x2 r1376, r1215, r1378;
}
{
fma.rn.f16x2 r1379, r1218, r1381, r1376;
}
{
add.f16x2 %0, r539, r1149;
}
{
add.f16x2 %1, r542, r1152;
}
{
sub.f16x2 %24, r539, r1149;
}
{
sub.f16x2 %25, r542, r1152;
}
{
add.f16x2 %2, r551, r1227;
}
{
add.f16x2 %3, r554, r1233;
}
{
sub.f16x2 %26, r551, r1227;
}
{
sub.f16x2 %27, r554, r1233;
}
{
add.f16x2 %4, r563, r1243;
}
{
add.f16x2 %5, r566, r1249;
}
{
sub.f16x2 %28, r563, r1243;
}
{
sub.f16x2 %29, r566, r1249;
}
{
add.f16x2 %6, r575, r1259;
}
{
add.f16x2 %7, r578, r1265;
}
{
sub.f16x2 %30, r575, r1259;
}
{
sub.f16x2 %31, r578, r1265;
}
{
add.f16x2 %8, r587, r1275;
}
{
add.f16x2 %9, r590, r1281;
}
{
sub.f16x2 %32, r587, r1275;
}
{
sub.f16x2 %33, r590, r1281;
}
{
add.f16x2 %10, r599, r1291;
}
{
add.f16x2 %11, r602, r1297;
}
{
sub.f16x2 %34, r599, r1291;
}
{
sub.f16x2 %35, r602, r1297;
}
{
add.f16x2 %12, r545, r1301;
}
{
add.f16x2 %13, r548, r1155;
}
{
sub.f16x2 %36, r545, r1301;
}
{
sub.f16x2 %37, r548, r1155;
}
{
add.f16x2 %14, r557, r1309;
}
{
add.f16x2 %15, r560, r1315;
}
{
sub.f16x2 %38, r557, r1309;
}
{
sub.f16x2 %39, r560, r1315;
}
{
add.f16x2 %16, r569, r1325;
}
{
add.f16x2 %17, r572, r1331;
}
{
sub.f16x2 %40, r569, r1325;
}
{
sub.f16x2 %41, r572, r1331;
}
{
add.f16x2 %18, r581, r1341;
}
{
add.f16x2 %19, r584, r1347;
}
{
sub.f16x2 %42, r581, r1341;
}
{
sub.f16x2 %43, r584, r1347;
}
{
add.f16x2 %20, r593, r1357;
}
{
add.f16x2 %21, r596, r1363;
}
{
sub.f16x2 %44, r593, r1357;
}
{
sub.f16x2 %45, r596, r1363;
}
{
add.f16x2 %22, r605, r1373;
}
{
add.f16x2 %23, r608, r1379;
}
{
sub.f16x2 %46, r605, r1373;
}
{
sub.f16x2 %47, r608, r1379;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)): "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[7].y)));
};


#endif
