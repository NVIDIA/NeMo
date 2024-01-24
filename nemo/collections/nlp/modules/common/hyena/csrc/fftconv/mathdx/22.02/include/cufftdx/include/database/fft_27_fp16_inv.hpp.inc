#ifndef CUFFTDX_FFT_27_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_27_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1069, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<155>;
.reg .b32 r<2771>;
.reg .f64 fd<155>;
.reg .b64 rd<3>;
mov.f64 fd153, 0dBFE0000000000000;
{
cvt.rn.f16.f64 rs1, fd153;
}
mov.b32 r72, {rs1, rs1};
mov.f64 fd154, 0dBFEBB67AE8584CAA;
{
cvt.rn.f16.f64 rs2, fd154;
}
mov.b32 r81, {rs2, rs2};
{
add.f16x2 r1, %56, %54;
}
{
add.f16x2 r4, %57, r1;
}
{
add.f16x2 r7, %58, %55;
}
{
add.f16x2 r10, %59, r7;
}
{
add.f16x2 r13, %56, %54;
}
{
mul.f16x2 r16, r13, r72;
}
{
add.f16x2 r19, %57, r16;
}
{
sub.f16x2 r22, %58, %55;
}
{
mul.f16x2 r25, r22, r81;
}
{
add.f16x2 r28, r19, r25;
}
{
add.f16x2 r31, %56, %54;
}
{
mul.f16x2 r34, r31, r72;
}
{
add.f16x2 r37, %57, r34;
}
{
sub.f16x2 r40, %58, %55;
}
{
mul.f16x2 r43, r40, r81;
}
{
sub.f16x2 r46, r37, r43;
}
{
add.f16x2 r49, %58, %55;
}
{
mul.f16x2 r52, r49, r72;
}
{
add.f16x2 r55, %59, r52;
}
{
sub.f16x2 r58, %56, %54;
}
{
mul.f16x2 r61, r58, r81;
}
{
sub.f16x2 r64, r55, r61;
}
{
add.f16x2 r67, %58, %55;
}
{
mul.f16x2 r70, r67, r72;
}
{
add.f16x2 r73, %59, r70;
}
{
sub.f16x2 r76, %56, %54;
}
{
mul.f16x2 r79, r76, r81;
}
{
add.f16x2 r82, r73, r79;
}
{
cvt.rn.f16.f64 rs3, fd153;
}
mov.b32 r156, {rs3, rs3};
{
cvt.rn.f16.f64 rs4, fd154;
}
mov.b32 r165, {rs4, rs4};
{
add.f16x2 r85, %61, %65;
}
{
add.f16x2 r88, %62, r85;
}
{
add.f16x2 r91, %63, %60;
}
{
add.f16x2 r94, %64, r91;
}
{
add.f16x2 r97, %61, %65;
}
{
mul.f16x2 r100, r97, r156;
}
{
add.f16x2 r103, %62, r100;
}
{
sub.f16x2 r106, %63, %60;
}
{
mul.f16x2 r109, r106, r165;
}
{
add.f16x2 r112, r103, r109;
}
{
add.f16x2 r115, %61, %65;
}
{
mul.f16x2 r118, r115, r156;
}
{
add.f16x2 r121, %62, r118;
}
{
sub.f16x2 r124, %63, %60;
}
{
mul.f16x2 r127, r124, r165;
}
{
sub.f16x2 r130, r121, r127;
}
{
add.f16x2 r133, %63, %60;
}
{
mul.f16x2 r136, r133, r156;
}
{
add.f16x2 r139, %64, r136;
}
{
sub.f16x2 r142, %61, %65;
}
{
mul.f16x2 r145, r142, r165;
}
{
sub.f16x2 r148, r139, r145;
}
{
add.f16x2 r151, %63, %60;
}
{
mul.f16x2 r154, r151, r156;
}
{
add.f16x2 r157, %64, r154;
}
{
sub.f16x2 r160, %61, %65;
}
{
mul.f16x2 r163, r160, r165;
}
{
add.f16x2 r166, r157, r163;
}
{
cvt.rn.f16.f64 rs5, fd153;
}
mov.b32 r240, {rs5, rs5};
{
cvt.rn.f16.f64 rs6, fd154;
}
mov.b32 r249, {rs6, rs6};
{
add.f16x2 r169, %69, %67;
}
{
add.f16x2 r172, %70, r169;
}
{
add.f16x2 r175, %71, %68;
}
{
add.f16x2 r178, %66, r175;
}
{
add.f16x2 r181, %69, %67;
}
{
mul.f16x2 r184, r181, r240;
}
{
add.f16x2 r187, %70, r184;
}
{
sub.f16x2 r190, %71, %68;
}
{
mul.f16x2 r193, r190, r249;
}
{
add.f16x2 r196, r187, r193;
}
{
add.f16x2 r199, %69, %67;
}
{
mul.f16x2 r202, r199, r240;
}
{
add.f16x2 r205, %70, r202;
}
{
sub.f16x2 r208, %71, %68;
}
{
mul.f16x2 r211, r208, r249;
}
{
sub.f16x2 r214, r205, r211;
}
{
add.f16x2 r217, %71, %68;
}
{
mul.f16x2 r220, r217, r240;
}
{
add.f16x2 r223, %66, r220;
}
{
sub.f16x2 r226, %69, %67;
}
{
mul.f16x2 r229, r226, r249;
}
{
sub.f16x2 r232, r223, r229;
}
{
add.f16x2 r235, %71, %68;
}
{
mul.f16x2 r238, r235, r240;
}
{
add.f16x2 r241, %66, r238;
}
{
sub.f16x2 r244, %69, %67;
}
{
mul.f16x2 r247, r244, r249;
}
{
add.f16x2 r250, r241, r247;
}
mov.f64 fd89, 0d3FE8836FA2CF5039;
{
cvt.rn.f16.f64 rs7, fd89;
}
mov.f64 fd90, 0d3FE491B7523C161D;
{
cvt.rn.f16.f64 rs8, fd90;
}
mov.f64 fd95, 0d3FC63A1A7E0B738A;
{
cvt.rn.f16.f64 rs9, fd95;
}
mov.f64 fd96, 0d3FEF838B8C811C17;
{
cvt.rn.f16.f64 rs10, fd96;
}
mov.f64 fd107, 0dBFEE11F642522D1C;
{
cvt.rn.f16.f64 rs13, fd107;
}
mov.f64 fd108, 0d3FD5E3A8748A0BF5;
{
cvt.rn.f16.f64 rs14, fd108;
}
mov.b32 r267, {rs7, rs7};
{
mul.f16x2 r253, r112, r267;
}
mov.b32 r264, {rs8, rs8};
{
mul.f16x2 r256, r148, r264;
}
{
sub.f16x2 r259, r253, r256;
}
{
mul.f16x2 r262, r112, r264;
}
{
fma.rn.f16x2 r265, r148, r267, r262;
}
mov.b32 r299, {rs9, rs9};
{
mul.f16x2 r269, r196, r299;
}
mov.b32 r296, {rs10, rs10};
{
mul.f16x2 r272, r232, r296;
}
{
sub.f16x2 r275, r269, r272;
}
{
mul.f16x2 r278, r196, r296;
}
{
fma.rn.f16x2 r281, r232, r299, r278;
}
{
mul.f16x2 r285, r130, r299;
}
{
mul.f16x2 r288, r166, r296;
}
{
sub.f16x2 r291, r285, r288;
}
{
mul.f16x2 r294, r130, r296;
}
{
fma.rn.f16x2 r297, r166, r299, r294;
}
mov.b32 r315, {rs13, rs13};
{
mul.f16x2 r301, r214, r315;
}
mov.b32 r312, {rs14, rs14};
{
mul.f16x2 r304, r250, r312;
}
{
sub.f16x2 r307, r301, r304;
}
{
mul.f16x2 r310, r214, r312;
}
{
fma.rn.f16x2 r313, r250, r315, r310;
}
{
cvt.rn.f16.f64 rs23, fd153;
}
mov.b32 r388, {rs23, rs23};
{
cvt.rn.f16.f64 rs24, fd154;
}
mov.b32 r397, {rs24, rs24};
{
add.f16x2 r317, r88, r172;
}
{
add.f16x2 r320, r4, r317;
}
{
add.f16x2 r323, r94, r178;
}
{
add.f16x2 r326, r10, r323;
}
{
add.f16x2 r329, r88, r172;
}
{
mul.f16x2 r332, r329, r388;
}
{
add.f16x2 r335, r4, r332;
}
{
sub.f16x2 r338, r94, r178;
}
{
mul.f16x2 r341, r338, r397;
}
{
add.f16x2 r344, r335, r341;
}
{
add.f16x2 r347, r88, r172;
}
{
mul.f16x2 r350, r347, r388;
}
{
add.f16x2 r353, r4, r350;
}
{
sub.f16x2 r356, r94, r178;
}
{
mul.f16x2 r359, r356, r397;
}
{
sub.f16x2 r362, r353, r359;
}
{
add.f16x2 r365, r94, r178;
}
{
mul.f16x2 r368, r365, r388;
}
{
add.f16x2 r371, r10, r368;
}
{
sub.f16x2 r374, r88, r172;
}
{
mul.f16x2 r377, r374, r397;
}
{
sub.f16x2 r380, r371, r377;
}
{
add.f16x2 r383, r94, r178;
}
{
mul.f16x2 r386, r383, r388;
}
{
add.f16x2 r389, r10, r386;
}
{
sub.f16x2 r392, r88, r172;
}
{
mul.f16x2 r395, r392, r397;
}
{
add.f16x2 r398, r389, r395;
}
{
cvt.rn.f16.f64 rs25, fd153;
}
mov.b32 r472, {rs25, rs25};
{
cvt.rn.f16.f64 rs26, fd154;
}
mov.b32 r481, {rs26, rs26};
{
add.f16x2 r401, r259, r275;
}
{
add.f16x2 r404, r28, r401;
}
{
add.f16x2 r407, r265, r281;
}
{
add.f16x2 r410, r64, r407;
}
{
add.f16x2 r413, r259, r275;
}
{
mul.f16x2 r416, r413, r472;
}
{
add.f16x2 r419, r28, r416;
}
{
sub.f16x2 r422, r265, r281;
}
{
mul.f16x2 r425, r422, r481;
}
{
add.f16x2 r428, r419, r425;
}
{
add.f16x2 r431, r259, r275;
}
{
mul.f16x2 r434, r431, r472;
}
{
add.f16x2 r437, r28, r434;
}
{
sub.f16x2 r440, r265, r281;
}
{
mul.f16x2 r443, r440, r481;
}
{
sub.f16x2 r446, r437, r443;
}
{
add.f16x2 r449, r265, r281;
}
{
mul.f16x2 r452, r449, r472;
}
{
add.f16x2 r455, r64, r452;
}
{
sub.f16x2 r458, r259, r275;
}
{
mul.f16x2 r461, r458, r481;
}
{
sub.f16x2 r464, r455, r461;
}
{
add.f16x2 r467, r265, r281;
}
{
mul.f16x2 r470, r467, r472;
}
{
add.f16x2 r473, r64, r470;
}
{
sub.f16x2 r476, r259, r275;
}
{
mul.f16x2 r479, r476, r481;
}
{
add.f16x2 r482, r473, r479;
}
{
cvt.rn.f16.f64 rs27, fd153;
}
mov.b32 r556, {rs27, rs27};
{
cvt.rn.f16.f64 rs28, fd154;
}
mov.b32 r565, {rs28, rs28};
{
add.f16x2 r485, r291, r307;
}
{
add.f16x2 r488, r46, r485;
}
{
add.f16x2 r491, r297, r313;
}
{
add.f16x2 r494, r82, r491;
}
{
add.f16x2 r497, r291, r307;
}
{
mul.f16x2 r500, r497, r556;
}
{
add.f16x2 r503, r46, r500;
}
{
sub.f16x2 r506, r297, r313;
}
{
mul.f16x2 r509, r506, r565;
}
{
add.f16x2 r512, r503, r509;
}
{
add.f16x2 r515, r291, r307;
}
{
mul.f16x2 r518, r515, r556;
}
{
add.f16x2 r521, r46, r518;
}
{
sub.f16x2 r524, r297, r313;
}
{
mul.f16x2 r527, r524, r565;
}
{
sub.f16x2 r530, r521, r527;
}
{
add.f16x2 r533, r297, r313;
}
{
mul.f16x2 r536, r533, r556;
}
{
add.f16x2 r539, r82, r536;
}
{
sub.f16x2 r542, r291, r307;
}
{
mul.f16x2 r545, r542, r565;
}
{
sub.f16x2 r548, r539, r545;
}
{
add.f16x2 r551, r297, r313;
}
{
mul.f16x2 r554, r551, r556;
}
{
add.f16x2 r557, r82, r554;
}
{
sub.f16x2 r560, r291, r307;
}
{
mul.f16x2 r563, r560, r565;
}
{
add.f16x2 r566, r557, r563;
}
{
cvt.rn.f16.f64 rs29, fd153;
}
mov.b32 r640, {rs29, rs29};
{
cvt.rn.f16.f64 rs30, fd154;
}
mov.b32 r649, {rs30, rs30};
{
add.f16x2 r569, %73, %77;
}
{
add.f16x2 r572, %74, r569;
}
{
add.f16x2 r575, %75, %72;
}
{
add.f16x2 r578, %76, r575;
}
{
add.f16x2 r581, %73, %77;
}
{
mul.f16x2 r584, r581, r640;
}
{
add.f16x2 r587, %74, r584;
}
{
sub.f16x2 r590, %75, %72;
}
{
mul.f16x2 r593, r590, r649;
}
{
add.f16x2 r596, r587, r593;
}
{
add.f16x2 r599, %73, %77;
}
{
mul.f16x2 r602, r599, r640;
}
{
add.f16x2 r605, %74, r602;
}
{
sub.f16x2 r608, %75, %72;
}
{
mul.f16x2 r611, r608, r649;
}
{
sub.f16x2 r614, r605, r611;
}
{
add.f16x2 r617, %75, %72;
}
{
mul.f16x2 r620, r617, r640;
}
{
add.f16x2 r623, %76, r620;
}
{
sub.f16x2 r626, %73, %77;
}
{
mul.f16x2 r629, r626, r649;
}
{
sub.f16x2 r632, r623, r629;
}
{
add.f16x2 r635, %75, %72;
}
{
mul.f16x2 r638, r635, r640;
}
{
add.f16x2 r641, %76, r638;
}
{
sub.f16x2 r644, %73, %77;
}
{
mul.f16x2 r647, r644, r649;
}
{
add.f16x2 r650, r641, r647;
}
{
cvt.rn.f16.f64 rs31, fd153;
}
mov.b32 r724, {rs31, rs31};
{
cvt.rn.f16.f64 rs32, fd154;
}
mov.b32 r733, {rs32, rs32};
{
add.f16x2 r653, %82, %80;
}
{
add.f16x2 r656, %83, r653;
}
{
add.f16x2 r659, %78, %81;
}
{
add.f16x2 r662, %79, r659;
}
{
add.f16x2 r665, %82, %80;
}
{
mul.f16x2 r668, r665, r724;
}
{
add.f16x2 r671, %83, r668;
}
{
sub.f16x2 r674, %78, %81;
}
{
mul.f16x2 r677, r674, r733;
}
{
add.f16x2 r680, r671, r677;
}
{
add.f16x2 r683, %82, %80;
}
{
mul.f16x2 r686, r683, r724;
}
{
add.f16x2 r689, %83, r686;
}
{
sub.f16x2 r692, %78, %81;
}
{
mul.f16x2 r695, r692, r733;
}
{
sub.f16x2 r698, r689, r695;
}
{
add.f16x2 r701, %78, %81;
}
{
mul.f16x2 r704, r701, r724;
}
{
add.f16x2 r707, %79, r704;
}
{
sub.f16x2 r710, %82, %80;
}
{
mul.f16x2 r713, r710, r733;
}
{
sub.f16x2 r716, r707, r713;
}
{
add.f16x2 r719, %78, %81;
}
{
mul.f16x2 r722, r719, r724;
}
{
add.f16x2 r725, %79, r722;
}
{
sub.f16x2 r728, %82, %80;
}
{
mul.f16x2 r731, r728, r733;
}
{
add.f16x2 r734, r725, r731;
}
{
cvt.rn.f16.f64 rs33, fd153;
}
mov.b32 r808, {rs33, rs33};
{
cvt.rn.f16.f64 rs34, fd154;
}
mov.b32 r817, {rs34, rs34};
{
add.f16x2 r737, %86, %84;
}
{
add.f16x2 r740, %87, r737;
}
{
add.f16x2 r743, %88, %85;
}
{
add.f16x2 r746, %89, r743;
}
{
add.f16x2 r749, %86, %84;
}
{
mul.f16x2 r752, r749, r808;
}
{
add.f16x2 r755, %87, r752;
}
{
sub.f16x2 r758, %88, %85;
}
{
mul.f16x2 r761, r758, r817;
}
{
add.f16x2 r764, r755, r761;
}
{
add.f16x2 r767, %86, %84;
}
{
mul.f16x2 r770, r767, r808;
}
{
add.f16x2 r773, %87, r770;
}
{
sub.f16x2 r776, %88, %85;
}
{
mul.f16x2 r779, r776, r817;
}
{
sub.f16x2 r782, r773, r779;
}
{
add.f16x2 r785, %88, %85;
}
{
mul.f16x2 r788, r785, r808;
}
{
add.f16x2 r791, %89, r788;
}
{
sub.f16x2 r794, %86, %84;
}
{
mul.f16x2 r797, r794, r817;
}
{
sub.f16x2 r800, r791, r797;
}
{
add.f16x2 r803, %88, %85;
}
{
mul.f16x2 r806, r803, r808;
}
{
add.f16x2 r809, %89, r806;
}
{
sub.f16x2 r812, %86, %84;
}
{
mul.f16x2 r815, r812, r817;
}
{
add.f16x2 r818, r809, r815;
}
{
cvt.rn.f16.f64 rs35, fd89;
}
{
cvt.rn.f16.f64 rs36, fd90;
}
{
cvt.rn.f16.f64 rs37, fd95;
}
{
cvt.rn.f16.f64 rs38, fd96;
}
{
cvt.rn.f16.f64 rs41, fd107;
}
{
cvt.rn.f16.f64 rs42, fd108;
}
mov.b32 r835, {rs35, rs35};
{
mul.f16x2 r821, r680, r835;
}
mov.b32 r832, {rs36, rs36};
{
mul.f16x2 r824, r716, r832;
}
{
sub.f16x2 r827, r821, r824;
}
{
mul.f16x2 r830, r680, r832;
}
{
fma.rn.f16x2 r833, r716, r835, r830;
}
mov.b32 r867, {rs37, rs37};
{
mul.f16x2 r837, r764, r867;
}
mov.b32 r864, {rs38, rs38};
{
mul.f16x2 r840, r800, r864;
}
{
sub.f16x2 r843, r837, r840;
}
{
mul.f16x2 r846, r764, r864;
}
{
fma.rn.f16x2 r849, r800, r867, r846;
}
{
mul.f16x2 r853, r698, r867;
}
{
mul.f16x2 r856, r734, r864;
}
{
sub.f16x2 r859, r853, r856;
}
{
mul.f16x2 r862, r698, r864;
}
{
fma.rn.f16x2 r865, r734, r867, r862;
}
mov.b32 r883, {rs41, rs41};
{
mul.f16x2 r869, r782, r883;
}
mov.b32 r880, {rs42, rs42};
{
mul.f16x2 r872, r818, r880;
}
{
sub.f16x2 r875, r869, r872;
}
{
mul.f16x2 r878, r782, r880;
}
{
fma.rn.f16x2 r881, r818, r883, r878;
}
{
cvt.rn.f16.f64 rs51, fd153;
}
mov.b32 r956, {rs51, rs51};
{
cvt.rn.f16.f64 rs52, fd154;
}
mov.b32 r965, {rs52, rs52};
{
add.f16x2 r885, r656, r740;
}
{
add.f16x2 r888, r572, r885;
}
{
add.f16x2 r891, r662, r746;
}
{
add.f16x2 r894, r578, r891;
}
{
add.f16x2 r897, r656, r740;
}
{
mul.f16x2 r900, r897, r956;
}
{
add.f16x2 r903, r572, r900;
}
{
sub.f16x2 r906, r662, r746;
}
{
mul.f16x2 r909, r906, r965;
}
{
add.f16x2 r912, r903, r909;
}
{
add.f16x2 r915, r656, r740;
}
{
mul.f16x2 r918, r915, r956;
}
{
add.f16x2 r921, r572, r918;
}
{
sub.f16x2 r924, r662, r746;
}
{
mul.f16x2 r927, r924, r965;
}
{
sub.f16x2 r930, r921, r927;
}
{
add.f16x2 r933, r662, r746;
}
{
mul.f16x2 r936, r933, r956;
}
{
add.f16x2 r939, r578, r936;
}
{
sub.f16x2 r942, r656, r740;
}
{
mul.f16x2 r945, r942, r965;
}
{
sub.f16x2 r948, r939, r945;
}
{
add.f16x2 r951, r662, r746;
}
{
mul.f16x2 r954, r951, r956;
}
{
add.f16x2 r957, r578, r954;
}
{
sub.f16x2 r960, r656, r740;
}
{
mul.f16x2 r963, r960, r965;
}
{
add.f16x2 r966, r957, r963;
}
{
cvt.rn.f16.f64 rs53, fd153;
}
mov.b32 r1040, {rs53, rs53};
{
cvt.rn.f16.f64 rs54, fd154;
}
mov.b32 r1049, {rs54, rs54};
{
add.f16x2 r969, r827, r843;
}
{
add.f16x2 r972, r596, r969;
}
{
add.f16x2 r975, r833, r849;
}
{
add.f16x2 r978, r632, r975;
}
{
add.f16x2 r981, r827, r843;
}
{
mul.f16x2 r984, r981, r1040;
}
{
add.f16x2 r987, r596, r984;
}
{
sub.f16x2 r990, r833, r849;
}
{
mul.f16x2 r993, r990, r1049;
}
{
add.f16x2 r996, r987, r993;
}
{
add.f16x2 r999, r827, r843;
}
{
mul.f16x2 r1002, r999, r1040;
}
{
add.f16x2 r1005, r596, r1002;
}
{
sub.f16x2 r1008, r833, r849;
}
{
mul.f16x2 r1011, r1008, r1049;
}
{
sub.f16x2 r1014, r1005, r1011;
}
{
add.f16x2 r1017, r833, r849;
}
{
mul.f16x2 r1020, r1017, r1040;
}
{
add.f16x2 r1023, r632, r1020;
}
{
sub.f16x2 r1026, r827, r843;
}
{
mul.f16x2 r1029, r1026, r1049;
}
{
sub.f16x2 r1032, r1023, r1029;
}
{
add.f16x2 r1035, r833, r849;
}
{
mul.f16x2 r1038, r1035, r1040;
}
{
add.f16x2 r1041, r632, r1038;
}
{
sub.f16x2 r1044, r827, r843;
}
{
mul.f16x2 r1047, r1044, r1049;
}
{
add.f16x2 r1050, r1041, r1047;
}
{
cvt.rn.f16.f64 rs55, fd153;
}
mov.b32 r1124, {rs55, rs55};
{
cvt.rn.f16.f64 rs56, fd154;
}
mov.b32 r1133, {rs56, rs56};
{
add.f16x2 r1053, r859, r875;
}
{
add.f16x2 r1056, r614, r1053;
}
{
add.f16x2 r1059, r865, r881;
}
{
add.f16x2 r1062, r650, r1059;
}
{
add.f16x2 r1065, r859, r875;
}
{
mul.f16x2 r1068, r1065, r1124;
}
{
add.f16x2 r1071, r614, r1068;
}
{
sub.f16x2 r1074, r865, r881;
}
{
mul.f16x2 r1077, r1074, r1133;
}
{
add.f16x2 r1080, r1071, r1077;
}
{
add.f16x2 r1083, r859, r875;
}
{
mul.f16x2 r1086, r1083, r1124;
}
{
add.f16x2 r1089, r614, r1086;
}
{
sub.f16x2 r1092, r865, r881;
}
{
mul.f16x2 r1095, r1092, r1133;
}
{
sub.f16x2 r1098, r1089, r1095;
}
{
add.f16x2 r1101, r865, r881;
}
{
mul.f16x2 r1104, r1101, r1124;
}
{
add.f16x2 r1107, r650, r1104;
}
{
sub.f16x2 r1110, r859, r875;
}
{
mul.f16x2 r1113, r1110, r1133;
}
{
sub.f16x2 r1116, r1107, r1113;
}
{
add.f16x2 r1119, r865, r881;
}
{
mul.f16x2 r1122, r1119, r1124;
}
{
add.f16x2 r1125, r650, r1122;
}
{
sub.f16x2 r1128, r859, r875;
}
{
mul.f16x2 r1131, r1128, r1133;
}
{
add.f16x2 r1134, r1125, r1131;
}
{
cvt.rn.f16.f64 rs57, fd153;
}
mov.b32 r1208, {rs57, rs57};
{
cvt.rn.f16.f64 rs58, fd154;
}
mov.b32 r1217, {rs58, rs58};
{
add.f16x2 r1137, %94, %92;
}
{
add.f16x2 r1140, %95, r1137;
}
{
add.f16x2 r1143, %90, %93;
}
{
add.f16x2 r1146, %91, r1143;
}
{
add.f16x2 r1149, %94, %92;
}
{
mul.f16x2 r1152, r1149, r1208;
}
{
add.f16x2 r1155, %95, r1152;
}
{
sub.f16x2 r1158, %90, %93;
}
{
mul.f16x2 r1161, r1158, r1217;
}
{
add.f16x2 r1164, r1155, r1161;
}
{
add.f16x2 r1167, %94, %92;
}
{
mul.f16x2 r1170, r1167, r1208;
}
{
add.f16x2 r1173, %95, r1170;
}
{
sub.f16x2 r1176, %90, %93;
}
{
mul.f16x2 r1179, r1176, r1217;
}
{
sub.f16x2 r1182, r1173, r1179;
}
{
add.f16x2 r1185, %90, %93;
}
{
mul.f16x2 r1188, r1185, r1208;
}
{
add.f16x2 r1191, %91, r1188;
}
{
sub.f16x2 r1194, %94, %92;
}
{
mul.f16x2 r1197, r1194, r1217;
}
{
sub.f16x2 r1200, r1191, r1197;
}
{
add.f16x2 r1203, %90, %93;
}
{
mul.f16x2 r1206, r1203, r1208;
}
{
add.f16x2 r1209, %91, r1206;
}
{
sub.f16x2 r1212, %94, %92;
}
{
mul.f16x2 r1215, r1212, r1217;
}
{
add.f16x2 r1218, r1209, r1215;
}
{
cvt.rn.f16.f64 rs59, fd153;
}
mov.b32 r1292, {rs59, rs59};
{
cvt.rn.f16.f64 rs60, fd154;
}
mov.b32 r1301, {rs60, rs60};
{
add.f16x2 r1221, %98, %96;
}
{
add.f16x2 r1224, %99, r1221;
}
{
add.f16x2 r1227, %100, %97;
}
{
add.f16x2 r1230, %101, r1227;
}
{
add.f16x2 r1233, %98, %96;
}
{
mul.f16x2 r1236, r1233, r1292;
}
{
add.f16x2 r1239, %99, r1236;
}
{
sub.f16x2 r1242, %100, %97;
}
{
mul.f16x2 r1245, r1242, r1301;
}
{
add.f16x2 r1248, r1239, r1245;
}
{
add.f16x2 r1251, %98, %96;
}
{
mul.f16x2 r1254, r1251, r1292;
}
{
add.f16x2 r1257, %99, r1254;
}
{
sub.f16x2 r1260, %100, %97;
}
{
mul.f16x2 r1263, r1260, r1301;
}
{
sub.f16x2 r1266, r1257, r1263;
}
{
add.f16x2 r1269, %100, %97;
}
{
mul.f16x2 r1272, r1269, r1292;
}
{
add.f16x2 r1275, %101, r1272;
}
{
sub.f16x2 r1278, %98, %96;
}
{
mul.f16x2 r1281, r1278, r1301;
}
{
sub.f16x2 r1284, r1275, r1281;
}
{
add.f16x2 r1287, %100, %97;
}
{
mul.f16x2 r1290, r1287, r1292;
}
{
add.f16x2 r1293, %101, r1290;
}
{
sub.f16x2 r1296, %98, %96;
}
{
mul.f16x2 r1299, r1296, r1301;
}
{
add.f16x2 r1302, r1293, r1299;
}
{
cvt.rn.f16.f64 rs61, fd153;
}
mov.b32 r1376, {rs61, rs61};
{
cvt.rn.f16.f64 rs62, fd154;
}
mov.b32 r1385, {rs62, rs62};
{
add.f16x2 r1305, %107, %105;
}
{
add.f16x2 r1308, %102, r1305;
}
{
add.f16x2 r1311, %103, %106;
}
{
add.f16x2 r1314, %104, r1311;
}
{
add.f16x2 r1317, %107, %105;
}
{
mul.f16x2 r1320, r1317, r1376;
}
{
add.f16x2 r1323, %102, r1320;
}
{
sub.f16x2 r1326, %103, %106;
}
{
mul.f16x2 r1329, r1326, r1385;
}
{
add.f16x2 r1332, r1323, r1329;
}
{
add.f16x2 r1335, %107, %105;
}
{
mul.f16x2 r1338, r1335, r1376;
}
{
add.f16x2 r1341, %102, r1338;
}
{
sub.f16x2 r1344, %103, %106;
}
{
mul.f16x2 r1347, r1344, r1385;
}
{
sub.f16x2 r1350, r1341, r1347;
}
{
add.f16x2 r1353, %103, %106;
}
{
mul.f16x2 r1356, r1353, r1376;
}
{
add.f16x2 r1359, %104, r1356;
}
{
sub.f16x2 r1362, %107, %105;
}
{
mul.f16x2 r1365, r1362, r1385;
}
{
sub.f16x2 r1368, r1359, r1365;
}
{
add.f16x2 r1371, %103, %106;
}
{
mul.f16x2 r1374, r1371, r1376;
}
{
add.f16x2 r1377, %104, r1374;
}
{
sub.f16x2 r1380, %107, %105;
}
{
mul.f16x2 r1383, r1380, r1385;
}
{
add.f16x2 r1386, r1377, r1383;
}
{
cvt.rn.f16.f64 rs63, fd89;
}
{
cvt.rn.f16.f64 rs64, fd90;
}
{
cvt.rn.f16.f64 rs65, fd95;
}
{
cvt.rn.f16.f64 rs66, fd96;
}
{
cvt.rn.f16.f64 rs69, fd107;
}
{
cvt.rn.f16.f64 rs70, fd108;
}
mov.b32 r1403, {rs63, rs63};
{
mul.f16x2 r1389, r1248, r1403;
}
mov.b32 r1400, {rs64, rs64};
{
mul.f16x2 r1392, r1284, r1400;
}
{
sub.f16x2 r1395, r1389, r1392;
}
{
mul.f16x2 r1398, r1248, r1400;
}
{
fma.rn.f16x2 r1401, r1284, r1403, r1398;
}
mov.b32 r1435, {rs65, rs65};
{
mul.f16x2 r1405, r1332, r1435;
}
mov.b32 r1432, {rs66, rs66};
{
mul.f16x2 r1408, r1368, r1432;
}
{
sub.f16x2 r1411, r1405, r1408;
}
{
mul.f16x2 r1414, r1332, r1432;
}
{
fma.rn.f16x2 r1417, r1368, r1435, r1414;
}
{
mul.f16x2 r1421, r1266, r1435;
}
{
mul.f16x2 r1424, r1302, r1432;
}
{
sub.f16x2 r1427, r1421, r1424;
}
{
mul.f16x2 r1430, r1266, r1432;
}
{
fma.rn.f16x2 r1433, r1302, r1435, r1430;
}
mov.b32 r1451, {rs69, rs69};
{
mul.f16x2 r1437, r1350, r1451;
}
mov.b32 r1448, {rs70, rs70};
{
mul.f16x2 r1440, r1386, r1448;
}
{
sub.f16x2 r1443, r1437, r1440;
}
{
mul.f16x2 r1446, r1350, r1448;
}
{
fma.rn.f16x2 r1449, r1386, r1451, r1446;
}
{
cvt.rn.f16.f64 rs79, fd153;
}
mov.b32 r1524, {rs79, rs79};
{
cvt.rn.f16.f64 rs80, fd154;
}
mov.b32 r1533, {rs80, rs80};
{
add.f16x2 r1453, r1224, r1308;
}
{
add.f16x2 r1456, r1140, r1453;
}
{
add.f16x2 r1459, r1230, r1314;
}
{
add.f16x2 r1462, r1146, r1459;
}
{
add.f16x2 r1465, r1224, r1308;
}
{
mul.f16x2 r1468, r1465, r1524;
}
{
add.f16x2 r1471, r1140, r1468;
}
{
sub.f16x2 r1474, r1230, r1314;
}
{
mul.f16x2 r1477, r1474, r1533;
}
{
add.f16x2 r1480, r1471, r1477;
}
{
add.f16x2 r1483, r1224, r1308;
}
{
mul.f16x2 r1486, r1483, r1524;
}
{
add.f16x2 r1489, r1140, r1486;
}
{
sub.f16x2 r1492, r1230, r1314;
}
{
mul.f16x2 r1495, r1492, r1533;
}
{
sub.f16x2 r1498, r1489, r1495;
}
{
add.f16x2 r1501, r1230, r1314;
}
{
mul.f16x2 r1504, r1501, r1524;
}
{
add.f16x2 r1507, r1146, r1504;
}
{
sub.f16x2 r1510, r1224, r1308;
}
{
mul.f16x2 r1513, r1510, r1533;
}
{
sub.f16x2 r1516, r1507, r1513;
}
{
add.f16x2 r1519, r1230, r1314;
}
{
mul.f16x2 r1522, r1519, r1524;
}
{
add.f16x2 r1525, r1146, r1522;
}
{
sub.f16x2 r1528, r1224, r1308;
}
{
mul.f16x2 r1531, r1528, r1533;
}
{
add.f16x2 r1534, r1525, r1531;
}
{
cvt.rn.f16.f64 rs81, fd153;
}
mov.b32 r1608, {rs81, rs81};
{
cvt.rn.f16.f64 rs82, fd154;
}
mov.b32 r1617, {rs82, rs82};
{
add.f16x2 r1537, r1395, r1411;
}
{
add.f16x2 r1540, r1164, r1537;
}
{
add.f16x2 r1543, r1401, r1417;
}
{
add.f16x2 r1546, r1200, r1543;
}
{
add.f16x2 r1549, r1395, r1411;
}
{
mul.f16x2 r1552, r1549, r1608;
}
{
add.f16x2 r1555, r1164, r1552;
}
{
sub.f16x2 r1558, r1401, r1417;
}
{
mul.f16x2 r1561, r1558, r1617;
}
{
add.f16x2 r1564, r1555, r1561;
}
{
add.f16x2 r1567, r1395, r1411;
}
{
mul.f16x2 r1570, r1567, r1608;
}
{
add.f16x2 r1573, r1164, r1570;
}
{
sub.f16x2 r1576, r1401, r1417;
}
{
mul.f16x2 r1579, r1576, r1617;
}
{
sub.f16x2 r1582, r1573, r1579;
}
{
add.f16x2 r1585, r1401, r1417;
}
{
mul.f16x2 r1588, r1585, r1608;
}
{
add.f16x2 r1591, r1200, r1588;
}
{
sub.f16x2 r1594, r1395, r1411;
}
{
mul.f16x2 r1597, r1594, r1617;
}
{
sub.f16x2 r1600, r1591, r1597;
}
{
add.f16x2 r1603, r1401, r1417;
}
{
mul.f16x2 r1606, r1603, r1608;
}
{
add.f16x2 r1609, r1200, r1606;
}
{
sub.f16x2 r1612, r1395, r1411;
}
{
mul.f16x2 r1615, r1612, r1617;
}
{
add.f16x2 r1618, r1609, r1615;
}
{
cvt.rn.f16.f64 rs83, fd153;
}
mov.b32 r1692, {rs83, rs83};
{
cvt.rn.f16.f64 rs84, fd154;
}
mov.b32 r1701, {rs84, rs84};
{
add.f16x2 r1621, r1427, r1443;
}
{
add.f16x2 r1624, r1182, r1621;
}
{
add.f16x2 r1627, r1433, r1449;
}
{
add.f16x2 r1630, r1218, r1627;
}
{
add.f16x2 r1633, r1427, r1443;
}
{
mul.f16x2 r1636, r1633, r1692;
}
{
add.f16x2 r1639, r1182, r1636;
}
{
sub.f16x2 r1642, r1433, r1449;
}
{
mul.f16x2 r1645, r1642, r1701;
}
{
add.f16x2 r1648, r1639, r1645;
}
{
add.f16x2 r1651, r1427, r1443;
}
{
mul.f16x2 r1654, r1651, r1692;
}
{
add.f16x2 r1657, r1182, r1654;
}
{
sub.f16x2 r1660, r1433, r1449;
}
{
mul.f16x2 r1663, r1660, r1701;
}
{
sub.f16x2 r1666, r1657, r1663;
}
{
add.f16x2 r1669, r1433, r1449;
}
{
mul.f16x2 r1672, r1669, r1692;
}
{
add.f16x2 r1675, r1218, r1672;
}
{
sub.f16x2 r1678, r1427, r1443;
}
{
mul.f16x2 r1681, r1678, r1701;
}
{
sub.f16x2 r1684, r1675, r1681;
}
{
add.f16x2 r1687, r1433, r1449;
}
{
mul.f16x2 r1690, r1687, r1692;
}
{
add.f16x2 r1693, r1218, r1690;
}
{
sub.f16x2 r1696, r1427, r1443;
}
{
mul.f16x2 r1699, r1696, r1701;
}
{
add.f16x2 r1702, r1693, r1699;
}
mov.f64 fd85, 0d3FEF232EFF15C9E6;
{
cvt.rn.f16.f64 rs85, fd85;
}
mov.f64 fd86, 0d3FCD84D223638000;
{
cvt.rn.f16.f64 rs86, fd86;
}
mov.f64 fd87, 0d3FEC98A37A9A7850;
{
cvt.rn.f16.f64 rs87, fd87;
}
mov.f64 fd88, 0d3FDCB920325BAFA6;
{
cvt.rn.f16.f64 rs88, fd88;
}
{
cvt.rn.f16.f64 rs89, fd89;
}
{
cvt.rn.f16.f64 rs90, fd90;
}
mov.f64 fd91, 0d3FE31BEC55BC71BC;
{
cvt.rn.f16.f64 rs91, fd91;
}
mov.f64 fd92, 0d3FE9AAFE4207DF5F;
{
cvt.rn.f16.f64 rs92, fd92;
}
mov.f64 fd93, 0d3FD9595EF26FB670;
{
cvt.rn.f16.f64 rs93, fd93;
}
mov.f64 fd94, 0d3FED6206BEB6C24B;
{
cvt.rn.f16.f64 rs94, fd94;
}
{
cvt.rn.f16.f64 rs95, fd95;
}
{
cvt.rn.f16.f64 rs96, fd96;
}
mov.f64 fd97, 0dBFADC528B5343A86;
{
cvt.rn.f16.f64 rs97, fd97;
}
mov.f64 fd98, 0d3FEFF223F3635CE3;
{
cvt.rn.f16.f64 rs98, fd98;
}
mov.f64 fd99, 0dBFD25AFBF23865BF;
{
cvt.rn.f16.f64 rs99, fd99;
}
mov.f64 fd100, 0d3FEEA7D99F29CADE;
{
cvt.rn.f16.f64 rs100, fd100;
}
mov.f64 fd103, 0dBFE5F5B105F99707;
{
cvt.rn.f16.f64 rs103, fd103;
}
mov.f64 fd104, 0d3FE746A51650EADE;
{
cvt.rn.f16.f64 rs104, fd104;
}
mov.f64 fd115, 0dBFEABC50EF4734A7;
{
cvt.rn.f16.f64 rs107, fd107;
}
{
cvt.rn.f16.f64 rs108, fd108;
}
mov.f64 fd111, 0dBFEFC89BCEF44CF4;
{
cvt.rn.f16.f64 rs111, fd111;
}
mov.f64 fd112, 0dBFBDB843E577175E;
{
cvt.rn.f16.f64 rs112, fd112;
}
{
cvt.rn.f16.f64 rs115, fd115;
}
mov.f64 fd116, 0dBFE19593DA358510;
{
cvt.rn.f16.f64 rs116, fd116;
}
mov.b32 r1719, {rs85, rs85};
{
mul.f16x2 r1705, r972, r1719;
}
mov.b32 r1716, {rs86, rs86};
{
mul.f16x2 r1708, r978, r1716;
}
{
sub.f16x2 r1711, r1705, r1708;
}
{
mul.f16x2 r1714, r972, r1716;
}
{
fma.rn.f16x2 r1717, r978, r1719, r1714;
}
mov.b32 r1751, {rs87, rs87};
{
mul.f16x2 r1721, r1540, r1751;
}
mov.b32 r1748, {rs88, rs88};
{
mul.f16x2 r1724, r1546, r1748;
}
{
sub.f16x2 r1727, r1721, r1724;
}
{
mul.f16x2 r1730, r1540, r1748;
}
{
fma.rn.f16x2 r1733, r1546, r1751, r1730;
}
{
mul.f16x2 r1737, r1056, r1751;
}
{
mul.f16x2 r1740, r1062, r1748;
}
{
sub.f16x2 r1743, r1737, r1740;
}
{
mul.f16x2 r1746, r1056, r1748;
}
{
fma.rn.f16x2 r1749, r1062, r1751, r1746;
}
mov.b32 r1815, {rs91, rs91};
{
mul.f16x2 r1753, r1624, r1815;
}
mov.b32 r1812, {rs92, rs92};
{
mul.f16x2 r1756, r1630, r1812;
}
{
sub.f16x2 r1759, r1753, r1756;
}
{
mul.f16x2 r1762, r1624, r1812;
}
{
fma.rn.f16x2 r1765, r1630, r1815, r1762;
}
mov.b32 r1783, {rs89, rs89};
{
mul.f16x2 r1769, r912, r1783;
}
mov.b32 r1780, {rs90, rs90};
{
mul.f16x2 r1772, r948, r1780;
}
{
sub.f16x2 r1775, r1769, r1772;
}
{
mul.f16x2 r1778, r912, r1780;
}
{
fma.rn.f16x2 r1781, r948, r1783, r1778;
}
mov.b32 r1879, {rs95, rs95};
{
mul.f16x2 r1785, r1480, r1879;
}
mov.b32 r1876, {rs96, rs96};
{
mul.f16x2 r1788, r1516, r1876;
}
{
sub.f16x2 r1791, r1785, r1788;
}
{
mul.f16x2 r1794, r1480, r1876;
}
{
fma.rn.f16x2 r1797, r1516, r1879, r1794;
}
{
mul.f16x2 r1801, r996, r1815;
}
{
mul.f16x2 r1804, r1032, r1812;
}
{
sub.f16x2 r1807, r1801, r1804;
}
{
mul.f16x2 r1810, r996, r1812;
}
{
fma.rn.f16x2 r1813, r1032, r1815, r1810;
}
mov.b32 r1943, {rs99, rs99};
{
mul.f16x2 r1817, r1564, r1943;
}
mov.b32 r1940, {rs100, rs100};
{
mul.f16x2 r1820, r1600, r1940;
}
{
sub.f16x2 r1823, r1817, r1820;
}
{
mul.f16x2 r1826, r1564, r1940;
}
{
fma.rn.f16x2 r1829, r1600, r1943, r1826;
}
mov.b32 r1847, {rs93, rs93};
{
mul.f16x2 r1833, r1080, r1847;
}
mov.b32 r1844, {rs94, rs94};
{
mul.f16x2 r1836, r1116, r1844;
}
{
sub.f16x2 r1839, r1833, r1836;
}
{
mul.f16x2 r1842, r1080, r1844;
}
{
fma.rn.f16x2 r1845, r1116, r1847, r1842;
}
mov.b32 r1863, {rs103, rs103};
{
mul.f16x2 r1849, r1648, r1863;
}
mov.b32 r1860, {rs104, rs104};
{
mul.f16x2 r1852, r1684, r1860;
}
{
sub.f16x2 r1855, r1849, r1852;
}
{
mul.f16x2 r1858, r1648, r1860;
}
{
fma.rn.f16x2 r1861, r1684, r1863, r1858;
}
{
mul.f16x2 r1865, r930, r1879;
}
{
mul.f16x2 r1868, r966, r1876;
}
{
sub.f16x2 r1871, r1865, r1868;
}
{
mul.f16x2 r1874, r930, r1876;
}
{
fma.rn.f16x2 r1877, r966, r1879, r1874;
}
mov.b32 r1895, {rs107, rs107};
{
mul.f16x2 r1881, r1498, r1895;
}
mov.b32 r1892, {rs108, rs108};
{
mul.f16x2 r1884, r1534, r1892;
}
{
sub.f16x2 r1887, r1881, r1884;
}
{
mul.f16x2 r1890, r1498, r1892;
}
{
fma.rn.f16x2 r1893, r1534, r1895, r1890;
}
mov.b32 r1911, {rs97, rs97};
{
mul.f16x2 r1897, r1014, r1911;
}
mov.b32 r1908, {rs98, rs98};
{
mul.f16x2 r1900, r1050, r1908;
}
{
sub.f16x2 r1903, r1897, r1900;
}
{
mul.f16x2 r1906, r1014, r1908;
}
{
fma.rn.f16x2 r1909, r1050, r1911, r1906;
}
mov.b32 r1927, {rs111, rs111};
{
mul.f16x2 r1913, r1582, r1927;
}
mov.b32 r1924, {rs112, rs112};
{
mul.f16x2 r1916, r1618, r1924;
}
{
sub.f16x2 r1919, r1913, r1916;
}
{
mul.f16x2 r1922, r1582, r1924;
}
{
fma.rn.f16x2 r1925, r1618, r1927, r1922;
}
{
mul.f16x2 r1929, r1098, r1943;
}
{
mul.f16x2 r1932, r1134, r1940;
}
{
sub.f16x2 r1935, r1929, r1932;
}
{
mul.f16x2 r1938, r1098, r1940;
}
{
fma.rn.f16x2 r1941, r1134, r1943, r1938;
}
mov.b32 r1959, {rs115, rs115};
{
mul.f16x2 r1945, r1666, r1959;
}
mov.b32 r1956, {rs116, rs116};
{
mul.f16x2 r1948, r1702, r1956;
}
{
sub.f16x2 r1951, r1945, r1948;
}
{
mul.f16x2 r1954, r1666, r1956;
}
{
fma.rn.f16x2 r1957, r1702, r1959, r1954;
}
{
cvt.rn.f16.f64 rs137, fd153;
}
mov.b32 r2032, {rs137, rs137};
{
cvt.rn.f16.f64 rs138, fd154;
}
mov.b32 r2041, {rs138, rs138};
{
add.f16x2 r1961, r888, r1456;
}
{
add.f16x2 %0, r320, r1961;
}
{
add.f16x2 r1967, r894, r1462;
}
{
add.f16x2 %1, r326, r1967;
}
{
add.f16x2 r1973, r888, r1456;
}
{
mul.f16x2 r1976, r1973, r2032;
}
{
add.f16x2 r1979, r320, r1976;
}
{
sub.f16x2 r1982, r894, r1462;
}
{
mul.f16x2 r1985, r1982, r2041;
}
{
add.f16x2 %18, r1979, r1985;
}
{
add.f16x2 r1991, r888, r1456;
}
{
mul.f16x2 r1994, r1991, r2032;
}
{
add.f16x2 r1997, r320, r1994;
}
{
sub.f16x2 r2000, r894, r1462;
}
{
mul.f16x2 r2003, r2000, r2041;
}
{
sub.f16x2 %36, r1997, r2003;
}
{
add.f16x2 r2009, r894, r1462;
}
{
mul.f16x2 r2012, r2009, r2032;
}
{
add.f16x2 r2015, r326, r2012;
}
{
sub.f16x2 r2018, r888, r1456;
}
{
mul.f16x2 r2021, r2018, r2041;
}
{
sub.f16x2 %19, r2015, r2021;
}
{
add.f16x2 r2027, r894, r1462;
}
{
mul.f16x2 r2030, r2027, r2032;
}
{
add.f16x2 r2033, r326, r2030;
}
{
sub.f16x2 r2036, r888, r1456;
}
{
mul.f16x2 r2039, r2036, r2041;
}
{
add.f16x2 %37, r2033, r2039;
}
{
cvt.rn.f16.f64 rs139, fd153;
}
mov.b32 r2116, {rs139, rs139};
{
cvt.rn.f16.f64 rs140, fd154;
}
mov.b32 r2125, {rs140, rs140};
{
add.f16x2 r2045, r1711, r1727;
}
{
add.f16x2 %2, r404, r2045;
}
{
add.f16x2 r2051, r1717, r1733;
}
{
add.f16x2 %3, r410, r2051;
}
{
add.f16x2 r2057, r1711, r1727;
}
{
mul.f16x2 r2060, r2057, r2116;
}
{
add.f16x2 r2063, r404, r2060;
}
{
sub.f16x2 r2066, r1717, r1733;
}
{
mul.f16x2 r2069, r2066, r2125;
}
{
add.f16x2 %20, r2063, r2069;
}
{
add.f16x2 r2075, r1711, r1727;
}
{
mul.f16x2 r2078, r2075, r2116;
}
{
add.f16x2 r2081, r404, r2078;
}
{
sub.f16x2 r2084, r1717, r1733;
}
{
mul.f16x2 r2087, r2084, r2125;
}
{
sub.f16x2 %38, r2081, r2087;
}
{
add.f16x2 r2093, r1717, r1733;
}
{
mul.f16x2 r2096, r2093, r2116;
}
{
add.f16x2 r2099, r410, r2096;
}
{
sub.f16x2 r2102, r1711, r1727;
}
{
mul.f16x2 r2105, r2102, r2125;
}
{
sub.f16x2 %21, r2099, r2105;
}
{
add.f16x2 r2111, r1717, r1733;
}
{
mul.f16x2 r2114, r2111, r2116;
}
{
add.f16x2 r2117, r410, r2114;
}
{
sub.f16x2 r2120, r1711, r1727;
}
{
mul.f16x2 r2123, r2120, r2125;
}
{
add.f16x2 %39, r2117, r2123;
}
{
cvt.rn.f16.f64 rs141, fd153;
}
mov.b32 r2200, {rs141, rs141};
{
cvt.rn.f16.f64 rs142, fd154;
}
mov.b32 r2209, {rs142, rs142};
{
add.f16x2 r2129, r1743, r1759;
}
{
add.f16x2 %4, r488, r2129;
}
{
add.f16x2 r2135, r1749, r1765;
}
{
add.f16x2 %5, r494, r2135;
}
{
add.f16x2 r2141, r1743, r1759;
}
{
mul.f16x2 r2144, r2141, r2200;
}
{
add.f16x2 r2147, r488, r2144;
}
{
sub.f16x2 r2150, r1749, r1765;
}
{
mul.f16x2 r2153, r2150, r2209;
}
{
add.f16x2 %22, r2147, r2153;
}
{
add.f16x2 r2159, r1743, r1759;
}
{
mul.f16x2 r2162, r2159, r2200;
}
{
add.f16x2 r2165, r488, r2162;
}
{
sub.f16x2 r2168, r1749, r1765;
}
{
mul.f16x2 r2171, r2168, r2209;
}
{
sub.f16x2 %40, r2165, r2171;
}
{
add.f16x2 r2177, r1749, r1765;
}
{
mul.f16x2 r2180, r2177, r2200;
}
{
add.f16x2 r2183, r494, r2180;
}
{
sub.f16x2 r2186, r1743, r1759;
}
{
mul.f16x2 r2189, r2186, r2209;
}
{
sub.f16x2 %23, r2183, r2189;
}
{
add.f16x2 r2195, r1749, r1765;
}
{
mul.f16x2 r2198, r2195, r2200;
}
{
add.f16x2 r2201, r494, r2198;
}
{
sub.f16x2 r2204, r1743, r1759;
}
{
mul.f16x2 r2207, r2204, r2209;
}
{
add.f16x2 %41, r2201, r2207;
}
{
cvt.rn.f16.f64 rs143, fd153;
}
mov.b32 r2284, {rs143, rs143};
{
cvt.rn.f16.f64 rs144, fd154;
}
mov.b32 r2293, {rs144, rs144};
{
add.f16x2 r2213, r1775, r1791;
}
{
add.f16x2 %6, r344, r2213;
}
{
add.f16x2 r2219, r1781, r1797;
}
{
add.f16x2 %7, r380, r2219;
}
{
add.f16x2 r2225, r1775, r1791;
}
{
mul.f16x2 r2228, r2225, r2284;
}
{
add.f16x2 r2231, r344, r2228;
}
{
sub.f16x2 r2234, r1781, r1797;
}
{
mul.f16x2 r2237, r2234, r2293;
}
{
add.f16x2 %24, r2231, r2237;
}
{
add.f16x2 r2243, r1775, r1791;
}
{
mul.f16x2 r2246, r2243, r2284;
}
{
add.f16x2 r2249, r344, r2246;
}
{
sub.f16x2 r2252, r1781, r1797;
}
{
mul.f16x2 r2255, r2252, r2293;
}
{
sub.f16x2 %42, r2249, r2255;
}
{
add.f16x2 r2261, r1781, r1797;
}
{
mul.f16x2 r2264, r2261, r2284;
}
{
add.f16x2 r2267, r380, r2264;
}
{
sub.f16x2 r2270, r1775, r1791;
}
{
mul.f16x2 r2273, r2270, r2293;
}
{
sub.f16x2 %25, r2267, r2273;
}
{
add.f16x2 r2279, r1781, r1797;
}
{
mul.f16x2 r2282, r2279, r2284;
}
{
add.f16x2 r2285, r380, r2282;
}
{
sub.f16x2 r2288, r1775, r1791;
}
{
mul.f16x2 r2291, r2288, r2293;
}
{
add.f16x2 %43, r2285, r2291;
}
{
cvt.rn.f16.f64 rs145, fd153;
}
mov.b32 r2368, {rs145, rs145};
{
cvt.rn.f16.f64 rs146, fd154;
}
mov.b32 r2377, {rs146, rs146};
{
add.f16x2 r2297, r1807, r1823;
}
{
add.f16x2 %8, r428, r2297;
}
{
add.f16x2 r2303, r1813, r1829;
}
{
add.f16x2 %9, r464, r2303;
}
{
add.f16x2 r2309, r1807, r1823;
}
{
mul.f16x2 r2312, r2309, r2368;
}
{
add.f16x2 r2315, r428, r2312;
}
{
sub.f16x2 r2318, r1813, r1829;
}
{
mul.f16x2 r2321, r2318, r2377;
}
{
add.f16x2 %26, r2315, r2321;
}
{
add.f16x2 r2327, r1807, r1823;
}
{
mul.f16x2 r2330, r2327, r2368;
}
{
add.f16x2 r2333, r428, r2330;
}
{
sub.f16x2 r2336, r1813, r1829;
}
{
mul.f16x2 r2339, r2336, r2377;
}
{
sub.f16x2 %44, r2333, r2339;
}
{
add.f16x2 r2345, r1813, r1829;
}
{
mul.f16x2 r2348, r2345, r2368;
}
{
add.f16x2 r2351, r464, r2348;
}
{
sub.f16x2 r2354, r1807, r1823;
}
{
mul.f16x2 r2357, r2354, r2377;
}
{
sub.f16x2 %27, r2351, r2357;
}
{
add.f16x2 r2363, r1813, r1829;
}
{
mul.f16x2 r2366, r2363, r2368;
}
{
add.f16x2 r2369, r464, r2366;
}
{
sub.f16x2 r2372, r1807, r1823;
}
{
mul.f16x2 r2375, r2372, r2377;
}
{
add.f16x2 %45, r2369, r2375;
}
{
cvt.rn.f16.f64 rs147, fd153;
}
mov.b32 r2452, {rs147, rs147};
{
cvt.rn.f16.f64 rs148, fd154;
}
mov.b32 r2461, {rs148, rs148};
{
add.f16x2 r2381, r1839, r1855;
}
{
add.f16x2 %10, r512, r2381;
}
{
add.f16x2 r2387, r1845, r1861;
}
{
add.f16x2 %11, r548, r2387;
}
{
add.f16x2 r2393, r1839, r1855;
}
{
mul.f16x2 r2396, r2393, r2452;
}
{
add.f16x2 r2399, r512, r2396;
}
{
sub.f16x2 r2402, r1845, r1861;
}
{
mul.f16x2 r2405, r2402, r2461;
}
{
add.f16x2 %28, r2399, r2405;
}
{
add.f16x2 r2411, r1839, r1855;
}
{
mul.f16x2 r2414, r2411, r2452;
}
{
add.f16x2 r2417, r512, r2414;
}
{
sub.f16x2 r2420, r1845, r1861;
}
{
mul.f16x2 r2423, r2420, r2461;
}
{
sub.f16x2 %46, r2417, r2423;
}
{
add.f16x2 r2429, r1845, r1861;
}
{
mul.f16x2 r2432, r2429, r2452;
}
{
add.f16x2 r2435, r548, r2432;
}
{
sub.f16x2 r2438, r1839, r1855;
}
{
mul.f16x2 r2441, r2438, r2461;
}
{
sub.f16x2 %29, r2435, r2441;
}
{
add.f16x2 r2447, r1845, r1861;
}
{
mul.f16x2 r2450, r2447, r2452;
}
{
add.f16x2 r2453, r548, r2450;
}
{
sub.f16x2 r2456, r1839, r1855;
}
{
mul.f16x2 r2459, r2456, r2461;
}
{
add.f16x2 %47, r2453, r2459;
}
{
cvt.rn.f16.f64 rs149, fd153;
}
mov.b32 r2536, {rs149, rs149};
{
cvt.rn.f16.f64 rs150, fd154;
}
mov.b32 r2545, {rs150, rs150};
{
add.f16x2 r2465, r1871, r1887;
}
{
add.f16x2 %12, r362, r2465;
}
{
add.f16x2 r2471, r1877, r1893;
}
{
add.f16x2 %13, r398, r2471;
}
{
add.f16x2 r2477, r1871, r1887;
}
{
mul.f16x2 r2480, r2477, r2536;
}
{
add.f16x2 r2483, r362, r2480;
}
{
sub.f16x2 r2486, r1877, r1893;
}
{
mul.f16x2 r2489, r2486, r2545;
}
{
add.f16x2 %30, r2483, r2489;
}
{
add.f16x2 r2495, r1871, r1887;
}
{
mul.f16x2 r2498, r2495, r2536;
}
{
add.f16x2 r2501, r362, r2498;
}
{
sub.f16x2 r2504, r1877, r1893;
}
{
mul.f16x2 r2507, r2504, r2545;
}
{
sub.f16x2 %48, r2501, r2507;
}
{
add.f16x2 r2513, r1877, r1893;
}
{
mul.f16x2 r2516, r2513, r2536;
}
{
add.f16x2 r2519, r398, r2516;
}
{
sub.f16x2 r2522, r1871, r1887;
}
{
mul.f16x2 r2525, r2522, r2545;
}
{
sub.f16x2 %31, r2519, r2525;
}
{
add.f16x2 r2531, r1877, r1893;
}
{
mul.f16x2 r2534, r2531, r2536;
}
{
add.f16x2 r2537, r398, r2534;
}
{
sub.f16x2 r2540, r1871, r1887;
}
{
mul.f16x2 r2543, r2540, r2545;
}
{
add.f16x2 %49, r2537, r2543;
}
{
cvt.rn.f16.f64 rs151, fd153;
}
mov.b32 r2620, {rs151, rs151};
{
cvt.rn.f16.f64 rs152, fd154;
}
mov.b32 r2629, {rs152, rs152};
{
add.f16x2 r2549, r1903, r1919;
}
{
add.f16x2 %14, r446, r2549;
}
{
add.f16x2 r2555, r1909, r1925;
}
{
add.f16x2 %15, r482, r2555;
}
{
add.f16x2 r2561, r1903, r1919;
}
{
mul.f16x2 r2564, r2561, r2620;
}
{
add.f16x2 r2567, r446, r2564;
}
{
sub.f16x2 r2570, r1909, r1925;
}
{
mul.f16x2 r2573, r2570, r2629;
}
{
add.f16x2 %32, r2567, r2573;
}
{
add.f16x2 r2579, r1903, r1919;
}
{
mul.f16x2 r2582, r2579, r2620;
}
{
add.f16x2 r2585, r446, r2582;
}
{
sub.f16x2 r2588, r1909, r1925;
}
{
mul.f16x2 r2591, r2588, r2629;
}
{
sub.f16x2 %50, r2585, r2591;
}
{
add.f16x2 r2597, r1909, r1925;
}
{
mul.f16x2 r2600, r2597, r2620;
}
{
add.f16x2 r2603, r482, r2600;
}
{
sub.f16x2 r2606, r1903, r1919;
}
{
mul.f16x2 r2609, r2606, r2629;
}
{
sub.f16x2 %33, r2603, r2609;
}
{
add.f16x2 r2615, r1909, r1925;
}
{
mul.f16x2 r2618, r2615, r2620;
}
{
add.f16x2 r2621, r482, r2618;
}
{
sub.f16x2 r2624, r1903, r1919;
}
{
mul.f16x2 r2627, r2624, r2629;
}
{
add.f16x2 %51, r2621, r2627;
}
{
cvt.rn.f16.f64 rs153, fd153;
}
mov.b32 r2704, {rs153, rs153};
{
cvt.rn.f16.f64 rs154, fd154;
}
mov.b32 r2713, {rs154, rs154};
{
add.f16x2 r2633, r1935, r1951;
}
{
add.f16x2 %16, r530, r2633;
}
{
add.f16x2 r2639, r1941, r1957;
}
{
add.f16x2 %17, r566, r2639;
}
{
add.f16x2 r2645, r1935, r1951;
}
{
mul.f16x2 r2648, r2645, r2704;
}
{
add.f16x2 r2651, r530, r2648;
}
{
sub.f16x2 r2654, r1941, r1957;
}
{
mul.f16x2 r2657, r2654, r2713;
}
{
add.f16x2 %34, r2651, r2657;
}
{
add.f16x2 r2663, r1935, r1951;
}
{
mul.f16x2 r2666, r2663, r2704;
}
{
add.f16x2 r2669, r530, r2666;
}
{
sub.f16x2 r2672, r1941, r1957;
}
{
mul.f16x2 r2675, r2672, r2713;
}
{
sub.f16x2 %52, r2669, r2675;
}
{
add.f16x2 r2681, r1941, r1957;
}
{
mul.f16x2 r2684, r2681, r2704;
}
{
add.f16x2 r2687, r566, r2684;
}
{
sub.f16x2 r2690, r1935, r1951;
}
{
mul.f16x2 r2693, r2690, r2713;
}
{
sub.f16x2 %35, r2687, r2693;
}
{
add.f16x2 r2699, r1941, r1957;
}
{
mul.f16x2 r2702, r2699, r2704;
}
{
add.f16x2 r2705, r566, r2702;
}
{
sub.f16x2 r2708, r1935, r1951;
}
{
mul.f16x2 r2711, r2708, r2713;
}
{
add.f16x2 %53, r2705, r2711;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)): "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[17].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1071, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<92>;
.reg .b32 r<1165>;
.reg .b64 rd<4>;
mov.u32 r1154, %tid.y;
mov.u32 r1155, %18;
mad.lo.s32 r1156, r1154, 216, r1155;
mov.u32 r1157, %tid.x;
mov.f32 f86, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r1, {low, high};
}
mov.f32 f88, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r2, {low, high};
}
{
add.f16x2 r3, %25, %31;
}
{
add.f16x2 r6, %19, r3;
}
{
add.f16x2 r9, %26, %32;
}
{
add.f16x2 r12, %20, r9;
}
{
add.f16x2 r15, %25, %31;
}
{
mul.f16x2 r18, r15, r1;
}
{
add.f16x2 r21, %19, r18;
}
{
sub.f16x2 r24, %26, %32;
}
{
mul.f16x2 r27, r24, r2;
}
{
add.f16x2 r30, r21, r27;
}
{
add.f16x2 r33, %25, %31;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %19, r36;
}
{
sub.f16x2 r42, %26, %32;
}
{
mul.f16x2 r45, r42, r2;
}
{
sub.f16x2 r48, r39, r45;
}
{
add.f16x2 r51, %26, %32;
}
{
mul.f16x2 r54, r51, r1;
}
{
add.f16x2 r57, %20, r54;
}
{
sub.f16x2 r60, %25, %31;
}
{
mul.f16x2 r63, r60, r2;
}
{
sub.f16x2 r66, r57, r63;
}
{
add.f16x2 r69, %26, %32;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %20, r72;
}
{
sub.f16x2 r78, %25, %31;
}
{
mul.f16x2 r81, r78, r2;
}
{
add.f16x2 r84, r75, r81;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r88, {low, high};
}
{
add.f16x2 r89, %27, %33;
}
{
add.f16x2 r92, %21, r89;
}
{
add.f16x2 r95, %28, %34;
}
{
add.f16x2 r98, %22, r95;
}
{
add.f16x2 r101, %27, %33;
}
{
mul.f16x2 r104, r101, r87;
}
{
add.f16x2 r107, %21, r104;
}
{
sub.f16x2 r110, %28, %34;
}
{
mul.f16x2 r113, r110, r88;
}
{
add.f16x2 r116, r107, r113;
}
{
add.f16x2 r119, %27, %33;
}
{
mul.f16x2 r122, r119, r87;
}
{
add.f16x2 r125, %21, r122;
}
{
sub.f16x2 r128, %28, %34;
}
{
mul.f16x2 r131, r128, r88;
}
{
sub.f16x2 r134, r125, r131;
}
{
add.f16x2 r137, %28, %34;
}
{
mul.f16x2 r140, r137, r87;
}
{
add.f16x2 r143, %22, r140;
}
{
sub.f16x2 r146, %27, %33;
}
{
mul.f16x2 r149, r146, r88;
}
{
sub.f16x2 r152, r143, r149;
}
{
add.f16x2 r155, %28, %34;
}
{
mul.f16x2 r158, r155, r87;
}
{
add.f16x2 r161, %22, r158;
}
{
sub.f16x2 r164, %27, %33;
}
{
mul.f16x2 r167, r164, r88;
}
{
add.f16x2 r170, r161, r167;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r173, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r174, {low, high};
}
{
add.f16x2 r175, %29, %35;
}
{
add.f16x2 r178, %23, r175;
}
{
add.f16x2 r181, %30, %36;
}
{
add.f16x2 r184, %24, r181;
}
{
add.f16x2 r187, %29, %35;
}
{
mul.f16x2 r190, r187, r173;
}
{
add.f16x2 r193, %23, r190;
}
{
sub.f16x2 r196, %30, %36;
}
{
mul.f16x2 r199, r196, r174;
}
{
add.f16x2 r202, r193, r199;
}
{
add.f16x2 r205, %29, %35;
}
{
mul.f16x2 r208, r205, r173;
}
{
add.f16x2 r211, %23, r208;
}
{
sub.f16x2 r214, %30, %36;
}
{
mul.f16x2 r217, r214, r174;
}
{
sub.f16x2 r220, r211, r217;
}
{
add.f16x2 r223, %30, %36;
}
{
mul.f16x2 r226, r223, r173;
}
{
add.f16x2 r229, %24, r226;
}
{
sub.f16x2 r232, %29, %35;
}
{
mul.f16x2 r235, r232, r174;
}
{
sub.f16x2 r238, r229, r235;
}
{
add.f16x2 r241, %30, %36;
}
{
mul.f16x2 r244, r241, r173;
}
{
add.f16x2 r247, %24, r244;
}
{
sub.f16x2 r250, %29, %35;
}
{
mul.f16x2 r253, r250, r174;
}
{
add.f16x2 r256, r247, r253;
}
mov.f32 f14, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f14;
cvt.rn.f16.f32 high, f14;
mov.b32 r259, {low, high};
}
mov.f32 f16, 0f3F248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f16;
cvt.rn.f16.f32 high, f16;
mov.b32 r260, {low, high};
}
mov.f32 f18, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f18;
cvt.rn.f16.f32 high, f18;
mov.b32 r261, {low, high};
}
mov.f32 f20, 0f3F7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f20;
cvt.rn.f16.f32 high, f20;
mov.b32 r262, {low, high};
}
mov.f32 f26, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f26;
cvt.rn.f16.f32 high, f26;
mov.b32 r265, {low, high};
}
mov.f32 f28, 0f3EAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f28;
cvt.rn.f16.f32 high, f28;
mov.b32 r266, {low, high};
}
{
mul.f16x2 r275, r116, r259;
}
{
mul.f16x2 r278, r152, r260;
}
{
sub.f16x2 r281, r275, r278;
}
{
mul.f16x2 r284, r116, r260;
}
{
fma.rn.f16x2 r287, r152, r259, r284;
}
{
mul.f16x2 r291, r202, r261;
}
{
mul.f16x2 r294, r238, r262;
}
{
sub.f16x2 r297, r291, r294;
}
{
mul.f16x2 r300, r202, r262;
}
{
fma.rn.f16x2 r303, r238, r261, r300;
}
{
mul.f16x2 r307, r134, r261;
}
{
mul.f16x2 r310, r170, r262;
}
{
sub.f16x2 r313, r307, r310;
}
{
mul.f16x2 r316, r134, r262;
}
{
fma.rn.f16x2 r319, r170, r261, r316;
}
{
mul.f16x2 r323, r220, r265;
}
{
mul.f16x2 r326, r256, r266;
}
{
sub.f16x2 r329, r323, r326;
}
{
mul.f16x2 r332, r220, r266;
}
{
fma.rn.f16x2 r335, r256, r265, r332;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r339, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r340, {low, high};
}
{
add.f16x2 r341, r92, r178;
}
{
add.f16x2 r344, r6, r341;
}
{
add.f16x2 r347, r98, r184;
}
{
add.f16x2 r350, r12, r347;
}
{
add.f16x2 r353, r92, r178;
}
{
mul.f16x2 r356, r353, r339;
}
{
add.f16x2 r359, r6, r356;
}
{
sub.f16x2 r362, r98, r184;
}
{
mul.f16x2 r365, r362, r340;
}
{
add.f16x2 r368, r359, r365;
}
{
add.f16x2 r371, r92, r178;
}
{
mul.f16x2 r374, r371, r339;
}
{
add.f16x2 r377, r6, r374;
}
{
sub.f16x2 r380, r98, r184;
}
{
mul.f16x2 r383, r380, r340;
}
{
sub.f16x2 r386, r377, r383;
}
{
add.f16x2 r389, r98, r184;
}
{
mul.f16x2 r392, r389, r339;
}
{
add.f16x2 r395, r12, r392;
}
{
sub.f16x2 r398, r92, r178;
}
{
mul.f16x2 r401, r398, r340;
}
{
sub.f16x2 r404, r395, r401;
}
{
add.f16x2 r407, r98, r184;
}
{
mul.f16x2 r410, r407, r339;
}
{
add.f16x2 r413, r12, r410;
}
{
sub.f16x2 r416, r92, r178;
}
{
mul.f16x2 r419, r416, r340;
}
{
add.f16x2 r422, r413, r419;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r425, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r426, {low, high};
}
{
add.f16x2 r427, r281, r297;
}
{
add.f16x2 r430, r30, r427;
}
{
add.f16x2 r433, r287, r303;
}
{
add.f16x2 r436, r66, r433;
}
{
add.f16x2 r439, r281, r297;
}
{
mul.f16x2 r442, r439, r425;
}
{
add.f16x2 r445, r30, r442;
}
{
sub.f16x2 r448, r287, r303;
}
{
mul.f16x2 r451, r448, r426;
}
{
add.f16x2 r454, r445, r451;
}
{
add.f16x2 r457, r281, r297;
}
{
mul.f16x2 r460, r457, r425;
}
{
add.f16x2 r463, r30, r460;
}
{
sub.f16x2 r466, r287, r303;
}
{
mul.f16x2 r469, r466, r426;
}
{
sub.f16x2 r472, r463, r469;
}
{
add.f16x2 r475, r287, r303;
}
{
mul.f16x2 r478, r475, r425;
}
{
add.f16x2 r481, r66, r478;
}
{
sub.f16x2 r484, r281, r297;
}
{
mul.f16x2 r487, r484, r426;
}
{
sub.f16x2 r490, r481, r487;
}
{
add.f16x2 r493, r287, r303;
}
{
mul.f16x2 r496, r493, r425;
}
{
add.f16x2 r499, r66, r496;
}
{
sub.f16x2 r502, r281, r297;
}
{
mul.f16x2 r505, r502, r426;
}
{
add.f16x2 r508, r499, r505;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r512, {low, high};
}
{
add.f16x2 r513, r313, r329;
}
{
add.f16x2 r516, r48, r513;
}
{
add.f16x2 r519, r319, r335;
}
{
add.f16x2 r522, r84, r519;
}
{
add.f16x2 r525, r313, r329;
}
{
mul.f16x2 r528, r525, r511;
}
{
add.f16x2 r531, r48, r528;
}
{
sub.f16x2 r534, r319, r335;
}
{
mul.f16x2 r537, r534, r512;
}
{
add.f16x2 r540, r531, r537;
}
{
add.f16x2 r543, r313, r329;
}
{
mul.f16x2 r546, r543, r511;
}
{
add.f16x2 r549, r48, r546;
}
{
sub.f16x2 r552, r319, r335;
}
{
mul.f16x2 r555, r552, r512;
}
{
sub.f16x2 r558, r549, r555;
}
{
add.f16x2 r561, r319, r335;
}
{
mul.f16x2 r564, r561, r511;
}
{
add.f16x2 r567, r84, r564;
}
{
sub.f16x2 r570, r313, r329;
}
{
mul.f16x2 r573, r570, r512;
}
{
sub.f16x2 r576, r567, r573;
}
{
add.f16x2 r579, r319, r335;
}
{
mul.f16x2 r582, r579, r511;
}
{
add.f16x2 r585, r84, r582;
}
{
sub.f16x2 r588, r313, r329;
}
{
mul.f16x2 r591, r588, r512;
}
{
add.f16x2 r594, r585, r591;
}
mul.wide.u32 rd2, r1157, -1431655765;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r1158, rd3;
mul.lo.s32 r1159, r1158, 3;
sub.s32 r1160, r1157, r1159;
cvt.rn.f32.u32 f89, r1160;
mul.f32 f90, f89, 0f3E6E4BAE;
cos.approx.f32 f57, f90;
sin.approx.f32 f91, f90;
neg.f32 f58, f91;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f57;
cvt.rn.f16.f32 high, f58;
mov.b32 r597, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r600, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r602, {high, high};
}
{
mul.f16x2 r604, r436, r602;
}
{
fma.rn.f16x2 r607, r430, r600, r604;
}
{
mul.f16x2 r611, r430, r602;
}
{
neg.f16x2 r614, r611;
}
{
fma.rn.f16x2 r616, r436, r600, r614;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r620, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r622, {high, high};
}
mov.f32 f73, 0fBF800000;
mov.f32 f74, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r624, {low, high};
}
{
mul.f16x2 r625, r622, r624;
}
{
mul.f16x2 r628, r597, r620;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r631, {high, low};
}
{
fma.rn.f16x2 r633, r625, r631, r628;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r633;
mov.b32 r637, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r633;
mov.b32 r639, {high, high};
}
{
mul.f16x2 r641, r522, r639;
}
{
fma.rn.f16x2 r644, r516, r637, r641;
}
{
mul.f16x2 r648, r516, r639;
}
{
neg.f16x2 r651, r648;
}
{
fma.rn.f16x2 r653, r522, r637, r651;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r657, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r659, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r661, {low, high};
}
{
mul.f16x2 r662, r659, r661;
}
{
mul.f16x2 r665, r633, r657;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r633;
mov.b32 r668, {high, low};
}
{
fma.rn.f16x2 r670, r662, r668, r665;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r670;
mov.b32 r674, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r670;
mov.b32 r676, {high, high};
}
{
mul.f16x2 r678, r404, r676;
}
{
fma.rn.f16x2 r681, r368, r674, r678;
}
{
mul.f16x2 r685, r368, r676;
}
{
neg.f16x2 r688, r685;
}
{
fma.rn.f16x2 r690, r404, r674, r688;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r694, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r696, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r698, {low, high};
}
{
mul.f16x2 r699, r696, r698;
}
{
mul.f16x2 r702, r670, r694;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r670;
mov.b32 r705, {high, low};
}
{
fma.rn.f16x2 r707, r699, r705, r702;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r707;
mov.b32 r711, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r707;
mov.b32 r713, {high, high};
}
{
mul.f16x2 r715, r490, r713;
}
{
fma.rn.f16x2 r718, r454, r711, r715;
}
{
mul.f16x2 r722, r454, r713;
}
{
neg.f16x2 r725, r722;
}
{
fma.rn.f16x2 r727, r490, r711, r725;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r731, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r733, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r735, {low, high};
}
{
mul.f16x2 r736, r733, r735;
}
{
mul.f16x2 r739, r707, r731;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r707;
mov.b32 r742, {high, low};
}
{
fma.rn.f16x2 r744, r736, r742, r739;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r744;
mov.b32 r748, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r744;
mov.b32 r750, {high, high};
}
{
mul.f16x2 r752, r576, r750;
}
{
fma.rn.f16x2 r755, r540, r748, r752;
}
{
mul.f16x2 r759, r540, r750;
}
{
neg.f16x2 r762, r759;
}
{
fma.rn.f16x2 r764, r576, r748, r762;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r768, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r770, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r772, {low, high};
}
{
mul.f16x2 r773, r770, r772;
}
{
mul.f16x2 r776, r744, r768;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r744;
mov.b32 r779, {high, low};
}
{
fma.rn.f16x2 r781, r773, r779, r776;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r781;
mov.b32 r785, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r781;
mov.b32 r787, {high, high};
}
{
mul.f16x2 r789, r422, r787;
}
{
fma.rn.f16x2 r792, r386, r785, r789;
}
{
mul.f16x2 r796, r386, r787;
}
{
neg.f16x2 r799, r796;
}
{
fma.rn.f16x2 r801, r422, r785, r799;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r805, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r807, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r809, {low, high};
}
{
mul.f16x2 r810, r807, r809;
}
{
mul.f16x2 r813, r781, r805;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r781;
mov.b32 r816, {high, low};
}
{
fma.rn.f16x2 r818, r810, r816, r813;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r818;
mov.b32 r822, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r818;
mov.b32 r824, {high, high};
}
{
mul.f16x2 r826, r508, r824;
}
{
fma.rn.f16x2 r829, r472, r822, r826;
}
{
mul.f16x2 r833, r472, r824;
}
{
neg.f16x2 r836, r833;
}
{
fma.rn.f16x2 r838, r508, r822, r836;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r842, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r844, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r846, {low, high};
}
{
mul.f16x2 r847, r844, r846;
}
{
mul.f16x2 r850, r818, r842;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r818;
mov.b32 r853, {high, low};
}
{
fma.rn.f16x2 r855, r847, r853, r850;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r855;
mov.b32 r859, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r855;
mov.b32 r861, {high, high};
}
{
mul.f16x2 r863, r594, r861;
}
{
fma.rn.f16x2 r866, r558, r859, r863;
}
{
mul.f16x2 r870, r558, r861;
}
{
neg.f16x2 r873, r870;
}
{
fma.rn.f16x2 r875, r594, r859, r873;
}
mad.lo.s32 r1161, r1158, 216, r1156;
barrier.sync 0;
mad.lo.s32 r1162, r1160, 72, r1161;
st.shared.v2.f32 [r1162], {r344, r350};
st.shared.v2.f32 [r1162+8], {r607, r616};
st.shared.v2.f32 [r1162+16], {r644, r653};
st.shared.v2.f32 [r1162+24], {r681, r690};
st.shared.v2.f32 [r1162+32], {r718, r727};
st.shared.v2.f32 [r1162+40], {r755, r764};
st.shared.v2.f32 [r1162+48], {r792, r801};
st.shared.v2.f32 [r1162+56], {r829, r838};
st.shared.v2.f32 [r1162+64], {r866, r875};
barrier.sync 0;
shl.b32 r1163, r1160, 6;
sub.s32 r1164, r1162, r1163;
ld.shared.u32 r902, [r1164];
ld.shared.u32 r908, [r1164+4];
ld.shared.u32 r988, [r1164+24];
ld.shared.u32 r994, [r1164+28];
ld.shared.u32 r1074, [r1164+48];
ld.shared.u32 r1080, [r1164+52];
ld.shared.u32 r899, [r1164+72];
ld.shared.u32 r905, [r1164+76];
ld.shared.u32 r985, [r1164+96];
ld.shared.u32 r991, [r1164+100];
ld.shared.u32 r1071, [r1164+120];
ld.shared.u32 r1077, [r1164+124];
ld.shared.u32 r900, [r1164+144];
ld.shared.u32 r906, [r1164+148];
ld.shared.u32 r986, [r1164+168];
ld.shared.u32 r992, [r1164+172];
ld.shared.u32 r1072, [r1164+192];
ld.shared.u32 r1078, [r1164+196];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r896, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r897, {low, high};
}
{
add.f16x2 r898, r899, r900;
}
{
add.f16x2 %0, r902, r898;
}
{
add.f16x2 r904, r905, r906;
}
{
add.f16x2 %1, r908, r904;
}
{
add.f16x2 r910, r899, r900;
}
{
mul.f16x2 r913, r910, r896;
}
{
add.f16x2 r916, r902, r913;
}
{
sub.f16x2 r919, r905, r906;
}
{
mul.f16x2 r922, r919, r897;
}
{
add.f16x2 %6, r916, r922;
}
{
add.f16x2 r928, r899, r900;
}
{
mul.f16x2 r931, r928, r896;
}
{
add.f16x2 r934, r902, r931;
}
{
sub.f16x2 r937, r905, r906;
}
{
mul.f16x2 r940, r937, r897;
}
{
sub.f16x2 %12, r934, r940;
}
{
add.f16x2 r946, r905, r906;
}
{
mul.f16x2 r949, r946, r896;
}
{
add.f16x2 r952, r908, r949;
}
{
sub.f16x2 r955, r899, r900;
}
{
mul.f16x2 r958, r955, r897;
}
{
sub.f16x2 %7, r952, r958;
}
{
add.f16x2 r964, r905, r906;
}
{
mul.f16x2 r967, r964, r896;
}
{
add.f16x2 r970, r908, r967;
}
{
sub.f16x2 r973, r899, r900;
}
{
mul.f16x2 r976, r973, r897;
}
{
add.f16x2 %13, r970, r976;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r982, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r983, {low, high};
}
{
add.f16x2 r984, r985, r986;
}
{
add.f16x2 %2, r988, r984;
}
{
add.f16x2 r990, r991, r992;
}
{
add.f16x2 %3, r994, r990;
}
{
add.f16x2 r996, r985, r986;
}
{
mul.f16x2 r999, r996, r982;
}
{
add.f16x2 r1002, r988, r999;
}
{
sub.f16x2 r1005, r991, r992;
}
{
mul.f16x2 r1008, r1005, r983;
}
{
add.f16x2 %8, r1002, r1008;
}
{
add.f16x2 r1014, r985, r986;
}
{
mul.f16x2 r1017, r1014, r982;
}
{
add.f16x2 r1020, r988, r1017;
}
{
sub.f16x2 r1023, r991, r992;
}
{
mul.f16x2 r1026, r1023, r983;
}
{
sub.f16x2 %14, r1020, r1026;
}
{
add.f16x2 r1032, r991, r992;
}
{
mul.f16x2 r1035, r1032, r982;
}
{
add.f16x2 r1038, r994, r1035;
}
{
sub.f16x2 r1041, r985, r986;
}
{
mul.f16x2 r1044, r1041, r983;
}
{
sub.f16x2 %9, r1038, r1044;
}
{
add.f16x2 r1050, r991, r992;
}
{
mul.f16x2 r1053, r1050, r982;
}
{
add.f16x2 r1056, r994, r1053;
}
{
sub.f16x2 r1059, r985, r986;
}
{
mul.f16x2 r1062, r1059, r983;
}
{
add.f16x2 %15, r1056, r1062;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r1068, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r1069, {low, high};
}
{
add.f16x2 r1070, r1071, r1072;
}
{
add.f16x2 %4, r1074, r1070;
}
{
add.f16x2 r1076, r1077, r1078;
}
{
add.f16x2 %5, r1080, r1076;
}
{
add.f16x2 r1082, r1071, r1072;
}
{
mul.f16x2 r1085, r1082, r1068;
}
{
add.f16x2 r1088, r1074, r1085;
}
{
sub.f16x2 r1091, r1077, r1078;
}
{
mul.f16x2 r1094, r1091, r1069;
}
{
add.f16x2 %10, r1088, r1094;
}
{
add.f16x2 r1100, r1071, r1072;
}
{
mul.f16x2 r1103, r1100, r1068;
}
{
add.f16x2 r1106, r1074, r1103;
}
{
sub.f16x2 r1109, r1077, r1078;
}
{
mul.f16x2 r1112, r1109, r1069;
}
{
sub.f16x2 %16, r1106, r1112;
}
{
add.f16x2 r1118, r1077, r1078;
}
{
mul.f16x2 r1121, r1118, r1068;
}
{
add.f16x2 r1124, r1080, r1121;
}
{
sub.f16x2 r1127, r1071, r1072;
}
{
mul.f16x2 r1130, r1127, r1069;
}
{
sub.f16x2 %11, r1124, r1130;
}
{
add.f16x2 r1136, r1077, r1078;
}
{
mul.f16x2 r1139, r1136, r1068;
}
{
add.f16x2 r1142, r1080, r1139;
}
{
sub.f16x2 r1145, r1071, r1072;
}
{
mul.f16x2 r1148, r1145, r1069;
}
{
add.f16x2 %17, r1142, r1148;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1070, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<92>;
.reg .b32 r<1165>;
.reg .b64 rd<4>;
mov.u32 r1154, %tid.y;
mov.u32 r1155, %18;
mad.lo.s32 r1156, r1154, 108, r1155;
mov.u32 r1157, %tid.x;
mov.f32 f86, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r1, {low, high};
}
mov.f32 f88, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r2, {low, high};
}
{
add.f16x2 r3, %25, %31;
}
{
add.f16x2 r6, %19, r3;
}
{
add.f16x2 r9, %26, %32;
}
{
add.f16x2 r12, %20, r9;
}
{
add.f16x2 r15, %25, %31;
}
{
mul.f16x2 r18, r15, r1;
}
{
add.f16x2 r21, %19, r18;
}
{
sub.f16x2 r24, %26, %32;
}
{
mul.f16x2 r27, r24, r2;
}
{
add.f16x2 r30, r21, r27;
}
{
add.f16x2 r33, %25, %31;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %19, r36;
}
{
sub.f16x2 r42, %26, %32;
}
{
mul.f16x2 r45, r42, r2;
}
{
sub.f16x2 r48, r39, r45;
}
{
add.f16x2 r51, %26, %32;
}
{
mul.f16x2 r54, r51, r1;
}
{
add.f16x2 r57, %20, r54;
}
{
sub.f16x2 r60, %25, %31;
}
{
mul.f16x2 r63, r60, r2;
}
{
sub.f16x2 r66, r57, r63;
}
{
add.f16x2 r69, %26, %32;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %20, r72;
}
{
sub.f16x2 r78, %25, %31;
}
{
mul.f16x2 r81, r78, r2;
}
{
add.f16x2 r84, r75, r81;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r88, {low, high};
}
{
add.f16x2 r89, %27, %33;
}
{
add.f16x2 r92, %21, r89;
}
{
add.f16x2 r95, %28, %34;
}
{
add.f16x2 r98, %22, r95;
}
{
add.f16x2 r101, %27, %33;
}
{
mul.f16x2 r104, r101, r87;
}
{
add.f16x2 r107, %21, r104;
}
{
sub.f16x2 r110, %28, %34;
}
{
mul.f16x2 r113, r110, r88;
}
{
add.f16x2 r116, r107, r113;
}
{
add.f16x2 r119, %27, %33;
}
{
mul.f16x2 r122, r119, r87;
}
{
add.f16x2 r125, %21, r122;
}
{
sub.f16x2 r128, %28, %34;
}
{
mul.f16x2 r131, r128, r88;
}
{
sub.f16x2 r134, r125, r131;
}
{
add.f16x2 r137, %28, %34;
}
{
mul.f16x2 r140, r137, r87;
}
{
add.f16x2 r143, %22, r140;
}
{
sub.f16x2 r146, %27, %33;
}
{
mul.f16x2 r149, r146, r88;
}
{
sub.f16x2 r152, r143, r149;
}
{
add.f16x2 r155, %28, %34;
}
{
mul.f16x2 r158, r155, r87;
}
{
add.f16x2 r161, %22, r158;
}
{
sub.f16x2 r164, %27, %33;
}
{
mul.f16x2 r167, r164, r88;
}
{
add.f16x2 r170, r161, r167;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r173, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r174, {low, high};
}
{
add.f16x2 r175, %29, %35;
}
{
add.f16x2 r178, %23, r175;
}
{
add.f16x2 r181, %30, %36;
}
{
add.f16x2 r184, %24, r181;
}
{
add.f16x2 r187, %29, %35;
}
{
mul.f16x2 r190, r187, r173;
}
{
add.f16x2 r193, %23, r190;
}
{
sub.f16x2 r196, %30, %36;
}
{
mul.f16x2 r199, r196, r174;
}
{
add.f16x2 r202, r193, r199;
}
{
add.f16x2 r205, %29, %35;
}
{
mul.f16x2 r208, r205, r173;
}
{
add.f16x2 r211, %23, r208;
}
{
sub.f16x2 r214, %30, %36;
}
{
mul.f16x2 r217, r214, r174;
}
{
sub.f16x2 r220, r211, r217;
}
{
add.f16x2 r223, %30, %36;
}
{
mul.f16x2 r226, r223, r173;
}
{
add.f16x2 r229, %24, r226;
}
{
sub.f16x2 r232, %29, %35;
}
{
mul.f16x2 r235, r232, r174;
}
{
sub.f16x2 r238, r229, r235;
}
{
add.f16x2 r241, %30, %36;
}
{
mul.f16x2 r244, r241, r173;
}
{
add.f16x2 r247, %24, r244;
}
{
sub.f16x2 r250, %29, %35;
}
{
mul.f16x2 r253, r250, r174;
}
{
add.f16x2 r256, r247, r253;
}
mov.f32 f14, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f14;
cvt.rn.f16.f32 high, f14;
mov.b32 r259, {low, high};
}
mov.f32 f16, 0f3F248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f16;
cvt.rn.f16.f32 high, f16;
mov.b32 r260, {low, high};
}
mov.f32 f18, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f18;
cvt.rn.f16.f32 high, f18;
mov.b32 r261, {low, high};
}
mov.f32 f20, 0f3F7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f20;
cvt.rn.f16.f32 high, f20;
mov.b32 r262, {low, high};
}
mov.f32 f26, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f26;
cvt.rn.f16.f32 high, f26;
mov.b32 r265, {low, high};
}
mov.f32 f28, 0f3EAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f28;
cvt.rn.f16.f32 high, f28;
mov.b32 r266, {low, high};
}
{
mul.f16x2 r275, r116, r259;
}
{
mul.f16x2 r278, r152, r260;
}
{
sub.f16x2 r281, r275, r278;
}
{
mul.f16x2 r284, r116, r260;
}
{
fma.rn.f16x2 r287, r152, r259, r284;
}
{
mul.f16x2 r291, r202, r261;
}
{
mul.f16x2 r294, r238, r262;
}
{
sub.f16x2 r297, r291, r294;
}
{
mul.f16x2 r300, r202, r262;
}
{
fma.rn.f16x2 r303, r238, r261, r300;
}
{
mul.f16x2 r307, r134, r261;
}
{
mul.f16x2 r310, r170, r262;
}
{
sub.f16x2 r313, r307, r310;
}
{
mul.f16x2 r316, r134, r262;
}
{
fma.rn.f16x2 r319, r170, r261, r316;
}
{
mul.f16x2 r323, r220, r265;
}
{
mul.f16x2 r326, r256, r266;
}
{
sub.f16x2 r329, r323, r326;
}
{
mul.f16x2 r332, r220, r266;
}
{
fma.rn.f16x2 r335, r256, r265, r332;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r339, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r340, {low, high};
}
{
add.f16x2 r341, r92, r178;
}
{
add.f16x2 r344, r6, r341;
}
{
add.f16x2 r347, r98, r184;
}
{
add.f16x2 r350, r12, r347;
}
{
add.f16x2 r353, r92, r178;
}
{
mul.f16x2 r356, r353, r339;
}
{
add.f16x2 r359, r6, r356;
}
{
sub.f16x2 r362, r98, r184;
}
{
mul.f16x2 r365, r362, r340;
}
{
add.f16x2 r368, r359, r365;
}
{
add.f16x2 r371, r92, r178;
}
{
mul.f16x2 r374, r371, r339;
}
{
add.f16x2 r377, r6, r374;
}
{
sub.f16x2 r380, r98, r184;
}
{
mul.f16x2 r383, r380, r340;
}
{
sub.f16x2 r386, r377, r383;
}
{
add.f16x2 r389, r98, r184;
}
{
mul.f16x2 r392, r389, r339;
}
{
add.f16x2 r395, r12, r392;
}
{
sub.f16x2 r398, r92, r178;
}
{
mul.f16x2 r401, r398, r340;
}
{
sub.f16x2 r404, r395, r401;
}
{
add.f16x2 r407, r98, r184;
}
{
mul.f16x2 r410, r407, r339;
}
{
add.f16x2 r413, r12, r410;
}
{
sub.f16x2 r416, r92, r178;
}
{
mul.f16x2 r419, r416, r340;
}
{
add.f16x2 r422, r413, r419;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r425, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r426, {low, high};
}
{
add.f16x2 r427, r281, r297;
}
{
add.f16x2 r430, r30, r427;
}
{
add.f16x2 r433, r287, r303;
}
{
add.f16x2 r436, r66, r433;
}
{
add.f16x2 r439, r281, r297;
}
{
mul.f16x2 r442, r439, r425;
}
{
add.f16x2 r445, r30, r442;
}
{
sub.f16x2 r448, r287, r303;
}
{
mul.f16x2 r451, r448, r426;
}
{
add.f16x2 r454, r445, r451;
}
{
add.f16x2 r457, r281, r297;
}
{
mul.f16x2 r460, r457, r425;
}
{
add.f16x2 r463, r30, r460;
}
{
sub.f16x2 r466, r287, r303;
}
{
mul.f16x2 r469, r466, r426;
}
{
sub.f16x2 r472, r463, r469;
}
{
add.f16x2 r475, r287, r303;
}
{
mul.f16x2 r478, r475, r425;
}
{
add.f16x2 r481, r66, r478;
}
{
sub.f16x2 r484, r281, r297;
}
{
mul.f16x2 r487, r484, r426;
}
{
sub.f16x2 r490, r481, r487;
}
{
add.f16x2 r493, r287, r303;
}
{
mul.f16x2 r496, r493, r425;
}
{
add.f16x2 r499, r66, r496;
}
{
sub.f16x2 r502, r281, r297;
}
{
mul.f16x2 r505, r502, r426;
}
{
add.f16x2 r508, r499, r505;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r511, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r512, {low, high};
}
{
add.f16x2 r513, r313, r329;
}
{
add.f16x2 r516, r48, r513;
}
{
add.f16x2 r519, r319, r335;
}
{
add.f16x2 r522, r84, r519;
}
{
add.f16x2 r525, r313, r329;
}
{
mul.f16x2 r528, r525, r511;
}
{
add.f16x2 r531, r48, r528;
}
{
sub.f16x2 r534, r319, r335;
}
{
mul.f16x2 r537, r534, r512;
}
{
add.f16x2 r540, r531, r537;
}
{
add.f16x2 r543, r313, r329;
}
{
mul.f16x2 r546, r543, r511;
}
{
add.f16x2 r549, r48, r546;
}
{
sub.f16x2 r552, r319, r335;
}
{
mul.f16x2 r555, r552, r512;
}
{
sub.f16x2 r558, r549, r555;
}
{
add.f16x2 r561, r319, r335;
}
{
mul.f16x2 r564, r561, r511;
}
{
add.f16x2 r567, r84, r564;
}
{
sub.f16x2 r570, r313, r329;
}
{
mul.f16x2 r573, r570, r512;
}
{
sub.f16x2 r576, r567, r573;
}
{
add.f16x2 r579, r319, r335;
}
{
mul.f16x2 r582, r579, r511;
}
{
add.f16x2 r585, r84, r582;
}
{
sub.f16x2 r588, r313, r329;
}
{
mul.f16x2 r591, r588, r512;
}
{
add.f16x2 r594, r585, r591;
}
mul.wide.u32 rd2, r1157, -1431655765;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r1158, rd3;
mul.lo.s32 r1159, r1158, 3;
sub.s32 r1160, r1157, r1159;
mad.lo.s32 r1161, r1158, 108, r1156;
cvt.rn.f32.u32 f89, r1160;
mul.f32 f90, f89, 0f3E6E4BAE;
cos.approx.f32 f57, f90;
sin.approx.f32 f91, f90;
neg.f32 f58, f91;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f57;
cvt.rn.f16.f32 high, f58;
mov.b32 r597, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r600, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r602, {high, high};
}
{
mul.f16x2 r604, r436, r602;
}
{
fma.rn.f16x2 r607, r430, r600, r604;
}
{
mul.f16x2 r611, r430, r602;
}
{
neg.f16x2 r614, r611;
}
{
fma.rn.f16x2 r616, r436, r600, r614;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r620, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r622, {high, high};
}
mov.f32 f73, 0fBF800000;
mov.f32 f74, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r624, {low, high};
}
{
mul.f16x2 r625, r622, r624;
}
{
mul.f16x2 r628, r597, r620;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r631, {high, low};
}
{
fma.rn.f16x2 r633, r625, r631, r628;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r633;
mov.b32 r637, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r633;
mov.b32 r639, {high, high};
}
{
mul.f16x2 r641, r522, r639;
}
{
fma.rn.f16x2 r644, r516, r637, r641;
}
{
mul.f16x2 r648, r516, r639;
}
{
neg.f16x2 r651, r648;
}
{
fma.rn.f16x2 r653, r522, r637, r651;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r657, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r659, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r661, {low, high};
}
{
mul.f16x2 r662, r659, r661;
}
{
mul.f16x2 r665, r633, r657;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r633;
mov.b32 r668, {high, low};
}
{
fma.rn.f16x2 r670, r662, r668, r665;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r670;
mov.b32 r674, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r670;
mov.b32 r676, {high, high};
}
{
mul.f16x2 r678, r404, r676;
}
{
fma.rn.f16x2 r681, r368, r674, r678;
}
{
mul.f16x2 r685, r368, r676;
}
{
neg.f16x2 r688, r685;
}
{
fma.rn.f16x2 r690, r404, r674, r688;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r694, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r696, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r698, {low, high};
}
{
mul.f16x2 r699, r696, r698;
}
{
mul.f16x2 r702, r670, r694;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r670;
mov.b32 r705, {high, low};
}
{
fma.rn.f16x2 r707, r699, r705, r702;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r707;
mov.b32 r711, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r707;
mov.b32 r713, {high, high};
}
{
mul.f16x2 r715, r490, r713;
}
{
fma.rn.f16x2 r718, r454, r711, r715;
}
{
mul.f16x2 r722, r454, r713;
}
{
neg.f16x2 r725, r722;
}
{
fma.rn.f16x2 r727, r490, r711, r725;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r731, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r733, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r735, {low, high};
}
{
mul.f16x2 r736, r733, r735;
}
{
mul.f16x2 r739, r707, r731;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r707;
mov.b32 r742, {high, low};
}
{
fma.rn.f16x2 r744, r736, r742, r739;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r744;
mov.b32 r748, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r744;
mov.b32 r750, {high, high};
}
{
mul.f16x2 r752, r576, r750;
}
{
fma.rn.f16x2 r755, r540, r748, r752;
}
{
mul.f16x2 r759, r540, r750;
}
{
neg.f16x2 r762, r759;
}
{
fma.rn.f16x2 r764, r576, r748, r762;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r768, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r770, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r772, {low, high};
}
{
mul.f16x2 r773, r770, r772;
}
{
mul.f16x2 r776, r744, r768;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r744;
mov.b32 r779, {high, low};
}
{
fma.rn.f16x2 r781, r773, r779, r776;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r781;
mov.b32 r785, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r781;
mov.b32 r787, {high, high};
}
{
mul.f16x2 r789, r422, r787;
}
{
fma.rn.f16x2 r792, r386, r785, r789;
}
{
mul.f16x2 r796, r386, r787;
}
{
neg.f16x2 r799, r796;
}
{
fma.rn.f16x2 r801, r422, r785, r799;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r805, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r807, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r809, {low, high};
}
{
mul.f16x2 r810, r807, r809;
}
{
mul.f16x2 r813, r781, r805;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r781;
mov.b32 r816, {high, low};
}
{
fma.rn.f16x2 r818, r810, r816, r813;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r818;
mov.b32 r822, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r818;
mov.b32 r824, {high, high};
}
{
mul.f16x2 r826, r508, r824;
}
{
fma.rn.f16x2 r829, r472, r822, r826;
}
{
mul.f16x2 r833, r472, r824;
}
{
neg.f16x2 r836, r833;
}
{
fma.rn.f16x2 r838, r508, r822, r836;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r842, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r597;
mov.b32 r844, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r846, {low, high};
}
{
mul.f16x2 r847, r844, r846;
}
{
mul.f16x2 r850, r818, r842;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r818;
mov.b32 r853, {high, low};
}
{
fma.rn.f16x2 r855, r847, r853, r850;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r855;
mov.b32 r859, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r855;
mov.b32 r861, {high, high};
}
{
mul.f16x2 r863, r594, r861;
}
{
fma.rn.f16x2 r866, r558, r859, r863;
}
{
mul.f16x2 r870, r558, r861;
}
{
neg.f16x2 r873, r870;
}
{
fma.rn.f16x2 r875, r594, r859, r873;
}
barrier.sync 0;
mad.lo.s32 r1162, r1160, 36, r1161;
st.shared.u32 [r1162], r344;
st.shared.u32 [r1162+4], r607;
st.shared.u32 [r1162+8], r644;
st.shared.u32 [r1162+12], r681;
st.shared.u32 [r1162+16], r718;
st.shared.u32 [r1162+20], r755;
st.shared.u32 [r1162+24], r792;
st.shared.u32 [r1162+28], r829;
st.shared.u32 [r1162+32], r866;
barrier.sync 0;
shl.b32 r1163, r1160, 5;
sub.s32 r1164, r1162, r1163;
ld.shared.u32 r902, [r1164];
ld.shared.u32 r988, [r1164+12];
ld.shared.u32 r1074, [r1164+24];
ld.shared.u32 r899, [r1164+36];
ld.shared.u32 r985, [r1164+48];
ld.shared.u32 r1071, [r1164+60];
ld.shared.u32 r900, [r1164+72];
ld.shared.u32 r986, [r1164+84];
ld.shared.u32 r1072, [r1164+96];
barrier.sync 0;
st.shared.u32 [r1162], r350;
st.shared.u32 [r1162+4], r616;
st.shared.u32 [r1162+8], r653;
st.shared.u32 [r1162+12], r690;
st.shared.u32 [r1162+16], r727;
st.shared.u32 [r1162+20], r764;
st.shared.u32 [r1162+24], r801;
st.shared.u32 [r1162+28], r838;
st.shared.u32 [r1162+32], r875;
barrier.sync 0;
ld.shared.u32 r908, [r1164];
ld.shared.u32 r994, [r1164+12];
ld.shared.u32 r1080, [r1164+24];
ld.shared.u32 r905, [r1164+36];
ld.shared.u32 r991, [r1164+48];
ld.shared.u32 r1077, [r1164+60];
ld.shared.u32 r906, [r1164+72];
ld.shared.u32 r992, [r1164+84];
ld.shared.u32 r1078, [r1164+96];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r896, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r897, {low, high};
}
{
add.f16x2 r898, r899, r900;
}
{
add.f16x2 %0, r902, r898;
}
{
add.f16x2 r904, r905, r906;
}
{
add.f16x2 %1, r908, r904;
}
{
add.f16x2 r910, r899, r900;
}
{
mul.f16x2 r913, r910, r896;
}
{
add.f16x2 r916, r902, r913;
}
{
sub.f16x2 r919, r905, r906;
}
{
mul.f16x2 r922, r919, r897;
}
{
add.f16x2 %6, r916, r922;
}
{
add.f16x2 r928, r899, r900;
}
{
mul.f16x2 r931, r928, r896;
}
{
add.f16x2 r934, r902, r931;
}
{
sub.f16x2 r937, r905, r906;
}
{
mul.f16x2 r940, r937, r897;
}
{
sub.f16x2 %12, r934, r940;
}
{
add.f16x2 r946, r905, r906;
}
{
mul.f16x2 r949, r946, r896;
}
{
add.f16x2 r952, r908, r949;
}
{
sub.f16x2 r955, r899, r900;
}
{
mul.f16x2 r958, r955, r897;
}
{
sub.f16x2 %7, r952, r958;
}
{
add.f16x2 r964, r905, r906;
}
{
mul.f16x2 r967, r964, r896;
}
{
add.f16x2 r970, r908, r967;
}
{
sub.f16x2 r973, r899, r900;
}
{
mul.f16x2 r976, r973, r897;
}
{
add.f16x2 %13, r970, r976;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r982, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r983, {low, high};
}
{
add.f16x2 r984, r985, r986;
}
{
add.f16x2 %2, r988, r984;
}
{
add.f16x2 r990, r991, r992;
}
{
add.f16x2 %3, r994, r990;
}
{
add.f16x2 r996, r985, r986;
}
{
mul.f16x2 r999, r996, r982;
}
{
add.f16x2 r1002, r988, r999;
}
{
sub.f16x2 r1005, r991, r992;
}
{
mul.f16x2 r1008, r1005, r983;
}
{
add.f16x2 %8, r1002, r1008;
}
{
add.f16x2 r1014, r985, r986;
}
{
mul.f16x2 r1017, r1014, r982;
}
{
add.f16x2 r1020, r988, r1017;
}
{
sub.f16x2 r1023, r991, r992;
}
{
mul.f16x2 r1026, r1023, r983;
}
{
sub.f16x2 %14, r1020, r1026;
}
{
add.f16x2 r1032, r991, r992;
}
{
mul.f16x2 r1035, r1032, r982;
}
{
add.f16x2 r1038, r994, r1035;
}
{
sub.f16x2 r1041, r985, r986;
}
{
mul.f16x2 r1044, r1041, r983;
}
{
sub.f16x2 %9, r1038, r1044;
}
{
add.f16x2 r1050, r991, r992;
}
{
mul.f16x2 r1053, r1050, r982;
}
{
add.f16x2 r1056, r994, r1053;
}
{
sub.f16x2 r1059, r985, r986;
}
{
mul.f16x2 r1062, r1059, r983;
}
{
add.f16x2 %15, r1056, r1062;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r1068, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r1069, {low, high};
}
{
add.f16x2 r1070, r1071, r1072;
}
{
add.f16x2 %4, r1074, r1070;
}
{
add.f16x2 r1076, r1077, r1078;
}
{
add.f16x2 %5, r1080, r1076;
}
{
add.f16x2 r1082, r1071, r1072;
}
{
mul.f16x2 r1085, r1082, r1068;
}
{
add.f16x2 r1088, r1074, r1085;
}
{
sub.f16x2 r1091, r1077, r1078;
}
{
mul.f16x2 r1094, r1091, r1069;
}
{
add.f16x2 %10, r1088, r1094;
}
{
add.f16x2 r1100, r1071, r1072;
}
{
mul.f16x2 r1103, r1100, r1068;
}
{
add.f16x2 r1106, r1074, r1103;
}
{
sub.f16x2 r1109, r1077, r1078;
}
{
mul.f16x2 r1112, r1109, r1069;
}
{
sub.f16x2 %16, r1106, r1112;
}
{
add.f16x2 r1118, r1077, r1078;
}
{
mul.f16x2 r1121, r1118, r1068;
}
{
add.f16x2 r1124, r1080, r1121;
}
{
sub.f16x2 r1127, r1071, r1072;
}
{
mul.f16x2 r1130, r1127, r1069;
}
{
sub.f16x2 %11, r1124, r1130;
}
{
add.f16x2 r1136, r1077, r1078;
}
{
mul.f16x2 r1139, r1136, r1068;
}
{
add.f16x2 r1142, r1080, r1139;
}
{
sub.f16x2 r1145, r1071, r1072;
}
{
mul.f16x2 r1148, r1145, r1069;
}
{
add.f16x2 %17, r1142, r1148;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1072, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<35>;
.reg .b32 r<430>;
.reg .b64 rd<6>;
mov.u32 r413, %tid.y;
mov.u32 r414, %6;
mad.lo.s32 r415, r413, 216, r414;
mov.u32 r416, %tid.x;
mov.f32 f26, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f26;
cvt.rn.f16.f32 high, f26;
mov.b32 r1, {low, high};
}
mov.f32 f28, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f28;
cvt.rn.f16.f32 high, f28;
mov.b32 r2, {low, high};
}
{
add.f16x2 r3, %9, %11;
}
{
add.f16x2 r6, %7, r3;
}
{
add.f16x2 r9, %10, %12;
}
{
add.f16x2 r12, %8, r9;
}
{
add.f16x2 r15, %9, %11;
}
{
mul.f16x2 r18, r15, r1;
}
{
add.f16x2 r21, %7, r18;
}
{
sub.f16x2 r24, %10, %12;
}
{
mul.f16x2 r27, r24, r2;
}
{
add.f16x2 r30, r21, r27;
}
{
add.f16x2 r33, %9, %11;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %7, r36;
}
{
sub.f16x2 r42, %10, %12;
}
{
mul.f16x2 r45, r42, r2;
}
{
sub.f16x2 r48, r39, r45;
}
{
add.f16x2 r51, %10, %12;
}
{
mul.f16x2 r54, r51, r1;
}
{
add.f16x2 r57, %8, r54;
}
{
sub.f16x2 r60, %9, %11;
}
{
mul.f16x2 r63, r60, r2;
}
{
sub.f16x2 r66, r57, r63;
}
{
add.f16x2 r69, %10, %12;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %8, r72;
}
{
sub.f16x2 r78, %9, %11;
}
{
mul.f16x2 r81, r78, r2;
}
{
add.f16x2 r84, r75, r81;
}
mul.wide.u32 rd2, r416, 954437177;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r417, rd3;
mul.lo.s32 r418, r417, 9;
sub.s32 r419, r416, r418;
mad.lo.s32 r420, r417, 216, r415;
cvt.rn.f32.u32 f29, r419;
mul.f32 f30, f29, 0f3E6E4BAE;
cos.approx.f32 f5, f30;
sin.approx.f32 f31, f30;
neg.f32 f6, f31;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f5;
cvt.rn.f16.f32 high, f6;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r90, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r92, {high, high};
}
{
mul.f16x2 r94, r66, r92;
}
{
fma.rn.f16x2 r97, r30, r90, r94;
}
{
mul.f16x2 r101, r30, r92;
}
{
neg.f16x2 r104, r101;
}
{
fma.rn.f16x2 r106, r66, r90, r104;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r110, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r112, {high, high};
}
mov.f32 f21, 0fBF800000;
mov.f32 f22, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f21;
cvt.rn.f16.f32 high, f22;
mov.b32 r114, {low, high};
}
{
mul.f16x2 r115, r112, r114;
}
{
mul.f16x2 r118, r87, r110;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r121, {high, low};
}
{
fma.rn.f16x2 r123, r115, r121, r118;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r123;
mov.b32 r127, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r123;
mov.b32 r129, {high, high};
}
{
mul.f16x2 r131, r84, r129;
}
{
fma.rn.f16x2 r134, r48, r127, r131;
}
{
mul.f16x2 r138, r48, r129;
}
{
neg.f16x2 r141, r138;
}
{
fma.rn.f16x2 r143, r84, r127, r141;
}
barrier.sync 0;
mad.lo.s32 r421, r419, 24, r420;
st.shared.v2.f32 [r421], {r6, r12};
st.shared.v2.f32 [r421+8], {r97, r106};
st.shared.v2.f32 [r421+16], {r134, r143};
barrier.sync 0;
shl.b32 r422, r419, 4;
sub.s32 r423, r421, r422;
ld.shared.u32 r170, [r423];
ld.shared.u32 r176, [r423+4];
ld.shared.u32 r167, [r423+72];
ld.shared.u32 r173, [r423+76];
ld.shared.u32 r168, [r423+144];
ld.shared.u32 r174, [r423+148];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f26;
cvt.rn.f16.f32 high, f26;
mov.b32 r164, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f28;
cvt.rn.f16.f32 high, f28;
mov.b32 r165, {low, high};
}
{
add.f16x2 r166, r167, r168;
}
{
add.f16x2 r169, r170, r166;
}
{
add.f16x2 r172, r173, r174;
}
{
add.f16x2 r175, r176, r172;
}
{
add.f16x2 r178, r167, r168;
}
{
mul.f16x2 r181, r178, r164;
}
{
add.f16x2 r184, r170, r181;
}
{
sub.f16x2 r187, r173, r174;
}
{
mul.f16x2 r190, r187, r165;
}
{
add.f16x2 r193, r184, r190;
}
{
add.f16x2 r196, r167, r168;
}
{
mul.f16x2 r199, r196, r164;
}
{
add.f16x2 r202, r170, r199;
}
{
sub.f16x2 r205, r173, r174;
}
{
mul.f16x2 r208, r205, r165;
}
{
sub.f16x2 r211, r202, r208;
}
{
add.f16x2 r214, r173, r174;
}
{
mul.f16x2 r217, r214, r164;
}
{
add.f16x2 r220, r176, r217;
}
{
sub.f16x2 r223, r167, r168;
}
{
mul.f16x2 r226, r223, r165;
}
{
sub.f16x2 r229, r220, r226;
}
{
add.f16x2 r232, r173, r174;
}
{
mul.f16x2 r235, r232, r164;
}
{
add.f16x2 r238, r176, r235;
}
{
sub.f16x2 r241, r167, r168;
}
{
mul.f16x2 r244, r241, r165;
}
{
add.f16x2 r247, r238, r244;
}
mul.wide.u32 rd4, r419, -1431655765;
shr.u64 rd5, rd4, 33;
cvt.u32.u64 r424, rd5;
mul.lo.s32 r425, r424, 3;
sub.s32 r426, r419, r425;
shl.b32 r427, r426, 3;
add.s32 r428, r420, r427;
cvt.rn.f32.u32 f32, r424;
mul.f32 f33, f32, 0f3F32B8C2;
cos.approx.f32 f17, f33;
sin.approx.f32 f34, f33;
neg.f32 f18, f34;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f17;
cvt.rn.f16.f32 high, f18;
mov.b32 r250, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r250;
mov.b32 r253, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r250;
mov.b32 r255, {high, high};
}
{
mul.f16x2 r257, r229, r255;
}
{
fma.rn.f16x2 r260, r193, r253, r257;
}
{
mul.f16x2 r264, r193, r255;
}
{
neg.f16x2 r267, r264;
}
{
fma.rn.f16x2 r269, r229, r253, r267;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r250;
mov.b32 r273, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r250;
mov.b32 r275, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f21;
cvt.rn.f16.f32 high, f22;
mov.b32 r277, {low, high};
}
{
mul.f16x2 r278, r275, r277;
}
{
mul.f16x2 r281, r250, r273;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r250;
mov.b32 r284, {high, low};
}
{
fma.rn.f16x2 r286, r278, r284, r281;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r286;
mov.b32 r290, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r286;
mov.b32 r292, {high, high};
}
{
mul.f16x2 r294, r247, r292;
}
{
fma.rn.f16x2 r297, r211, r290, r294;
}
{
mul.f16x2 r301, r211, r292;
}
{
neg.f16x2 r304, r301;
}
{
fma.rn.f16x2 r306, r247, r290, r304;
}
barrier.sync 0;
mad.lo.s32 r429, r424, 72, r428;
st.shared.u32 [r429], r169;
st.shared.u32 [r429+4], r175;
st.shared.u32 [r429+24], r260;
st.shared.u32 [r429+28], r269;
st.shared.u32 [r429+48], r297;
st.shared.u32 [r429+52], r306;
barrier.sync 0;
ld.shared.u32 r333, [r423];
ld.shared.u32 r339, [r423+4];
ld.shared.u32 r330, [r423+72];
ld.shared.u32 r336, [r423+76];
ld.shared.u32 r331, [r423+144];
ld.shared.u32 r337, [r423+148];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f26;
cvt.rn.f16.f32 high, f26;
mov.b32 r327, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f28;
cvt.rn.f16.f32 high, f28;
mov.b32 r328, {low, high};
}
{
add.f16x2 r329, r330, r331;
}
{
add.f16x2 %0, r333, r329;
}
{
add.f16x2 r335, r336, r337;
}
{
add.f16x2 %1, r339, r335;
}
{
add.f16x2 r341, r330, r331;
}
{
mul.f16x2 r344, r341, r327;
}
{
add.f16x2 r347, r333, r344;
}
{
sub.f16x2 r350, r336, r337;
}
{
mul.f16x2 r353, r350, r328;
}
{
add.f16x2 %2, r347, r353;
}
{
add.f16x2 r359, r330, r331;
}
{
mul.f16x2 r362, r359, r327;
}
{
add.f16x2 r365, r333, r362;
}
{
sub.f16x2 r368, r336, r337;
}
{
mul.f16x2 r371, r368, r328;
}
{
sub.f16x2 %4, r365, r371;
}
{
add.f16x2 r377, r336, r337;
}
{
mul.f16x2 r380, r377, r327;
}
{
add.f16x2 r383, r339, r380;
}
{
sub.f16x2 r386, r330, r331;
}
{
mul.f16x2 r389, r386, r328;
}
{
sub.f16x2 %3, r383, r389;
}
{
add.f16x2 r395, r336, r337;
}
{
mul.f16x2 r398, r395, r327;
}
{
add.f16x2 r401, r339, r398;
}
{
sub.f16x2 r404, r330, r331;
}
{
mul.f16x2 r407, r404, r328;
}
{
add.f16x2 %5, r401, r407;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1073, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<35>;
.reg .b32 r<430>;
.reg .b64 rd<6>;
mov.u32 r413, %tid.y;
mov.u32 r414, %6;
mad.lo.s32 r415, r413, 108, r414;
mov.u32 r416, %tid.x;
mov.f32 f26, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f26;
cvt.rn.f16.f32 high, f26;
mov.b32 r1, {low, high};
}
mov.f32 f28, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f28;
cvt.rn.f16.f32 high, f28;
mov.b32 r2, {low, high};
}
{
add.f16x2 r3, %9, %11;
}
{
add.f16x2 r6, %7, r3;
}
{
add.f16x2 r9, %10, %12;
}
{
add.f16x2 r12, %8, r9;
}
{
add.f16x2 r15, %9, %11;
}
{
mul.f16x2 r18, r15, r1;
}
{
add.f16x2 r21, %7, r18;
}
{
sub.f16x2 r24, %10, %12;
}
{
mul.f16x2 r27, r24, r2;
}
{
add.f16x2 r30, r21, r27;
}
{
add.f16x2 r33, %9, %11;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %7, r36;
}
{
sub.f16x2 r42, %10, %12;
}
{
mul.f16x2 r45, r42, r2;
}
{
sub.f16x2 r48, r39, r45;
}
{
add.f16x2 r51, %10, %12;
}
{
mul.f16x2 r54, r51, r1;
}
{
add.f16x2 r57, %8, r54;
}
{
sub.f16x2 r60, %9, %11;
}
{
mul.f16x2 r63, r60, r2;
}
{
sub.f16x2 r66, r57, r63;
}
{
add.f16x2 r69, %10, %12;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %8, r72;
}
{
sub.f16x2 r78, %9, %11;
}
{
mul.f16x2 r81, r78, r2;
}
{
add.f16x2 r84, r75, r81;
}
mul.wide.u32 rd2, r416, 954437177;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r417, rd3;
mul.lo.s32 r418, r417, 9;
sub.s32 r419, r416, r418;
mad.lo.s32 r420, r417, 108, r415;
cvt.rn.f32.u32 f29, r419;
mul.f32 f30, f29, 0f3E6E4BAE;
cos.approx.f32 f5, f30;
sin.approx.f32 f31, f30;
neg.f32 f6, f31;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f5;
cvt.rn.f16.f32 high, f6;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r90, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r92, {high, high};
}
{
mul.f16x2 r94, r66, r92;
}
{
fma.rn.f16x2 r97, r30, r90, r94;
}
{
mul.f16x2 r101, r30, r92;
}
{
neg.f16x2 r104, r101;
}
{
fma.rn.f16x2 r106, r66, r90, r104;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r110, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r112, {high, high};
}
mov.f32 f21, 0fBF800000;
mov.f32 f22, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f21;
cvt.rn.f16.f32 high, f22;
mov.b32 r114, {low, high};
}
{
mul.f16x2 r115, r112, r114;
}
{
mul.f16x2 r118, r87, r110;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r121, {high, low};
}
{
fma.rn.f16x2 r123, r115, r121, r118;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r123;
mov.b32 r127, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r123;
mov.b32 r129, {high, high};
}
{
mul.f16x2 r131, r84, r129;
}
{
fma.rn.f16x2 r134, r48, r127, r131;
}
{
mul.f16x2 r138, r48, r129;
}
{
neg.f16x2 r141, r138;
}
{
fma.rn.f16x2 r143, r84, r127, r141;
}
barrier.sync 0;
mad.lo.s32 r421, r419, 12, r420;
st.shared.u32 [r421], r6;
st.shared.u32 [r421+4], r97;
st.shared.u32 [r421+8], r134;
barrier.sync 0;
shl.b32 r422, r419, 3;
sub.s32 r423, r421, r422;
ld.shared.u32 r170, [r423];
ld.shared.u32 r167, [r423+36];
ld.shared.u32 r168, [r423+72];
barrier.sync 0;
st.shared.u32 [r421], r12;
st.shared.u32 [r421+4], r106;
st.shared.u32 [r421+8], r143;
barrier.sync 0;
ld.shared.u32 r176, [r423];
ld.shared.u32 r173, [r423+36];
ld.shared.u32 r174, [r423+72];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f26;
cvt.rn.f16.f32 high, f26;
mov.b32 r164, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f28;
cvt.rn.f16.f32 high, f28;
mov.b32 r165, {low, high};
}
{
add.f16x2 r166, r167, r168;
}
{
add.f16x2 r169, r170, r166;
}
{
add.f16x2 r172, r173, r174;
}
{
add.f16x2 r175, r176, r172;
}
{
add.f16x2 r178, r167, r168;
}
{
mul.f16x2 r181, r178, r164;
}
{
add.f16x2 r184, r170, r181;
}
{
sub.f16x2 r187, r173, r174;
}
{
mul.f16x2 r190, r187, r165;
}
{
add.f16x2 r193, r184, r190;
}
{
add.f16x2 r196, r167, r168;
}
{
mul.f16x2 r199, r196, r164;
}
{
add.f16x2 r202, r170, r199;
}
{
sub.f16x2 r205, r173, r174;
}
{
mul.f16x2 r208, r205, r165;
}
{
sub.f16x2 r211, r202, r208;
}
{
add.f16x2 r214, r173, r174;
}
{
mul.f16x2 r217, r214, r164;
}
{
add.f16x2 r220, r176, r217;
}
{
sub.f16x2 r223, r167, r168;
}
{
mul.f16x2 r226, r223, r165;
}
{
sub.f16x2 r229, r220, r226;
}
{
add.f16x2 r232, r173, r174;
}
{
mul.f16x2 r235, r232, r164;
}
{
add.f16x2 r238, r176, r235;
}
{
sub.f16x2 r241, r167, r168;
}
{
mul.f16x2 r244, r241, r165;
}
{
add.f16x2 r247, r238, r244;
}
mul.wide.u32 rd4, r419, -1431655765;
shr.u64 rd5, rd4, 33;
cvt.u32.u64 r424, rd5;
mul.lo.s32 r425, r424, 3;
sub.s32 r426, r419, r425;
shl.b32 r427, r426, 2;
add.s32 r428, r420, r427;
cvt.rn.f32.u32 f32, r424;
mul.f32 f33, f32, 0f3F32B8C2;
cos.approx.f32 f17, f33;
sin.approx.f32 f34, f33;
neg.f32 f18, f34;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f17;
cvt.rn.f16.f32 high, f18;
mov.b32 r250, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r250;
mov.b32 r253, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r250;
mov.b32 r255, {high, high};
}
{
mul.f16x2 r257, r229, r255;
}
{
fma.rn.f16x2 r260, r193, r253, r257;
}
{
mul.f16x2 r264, r193, r255;
}
{
neg.f16x2 r267, r264;
}
{
fma.rn.f16x2 r269, r229, r253, r267;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r250;
mov.b32 r273, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r250;
mov.b32 r275, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f21;
cvt.rn.f16.f32 high, f22;
mov.b32 r277, {low, high};
}
{
mul.f16x2 r278, r275, r277;
}
{
mul.f16x2 r281, r250, r273;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r250;
mov.b32 r284, {high, low};
}
{
fma.rn.f16x2 r286, r278, r284, r281;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r286;
mov.b32 r290, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r286;
mov.b32 r292, {high, high};
}
{
mul.f16x2 r294, r247, r292;
}
{
fma.rn.f16x2 r297, r211, r290, r294;
}
{
mul.f16x2 r301, r211, r292;
}
{
neg.f16x2 r304, r301;
}
{
fma.rn.f16x2 r306, r247, r290, r304;
}
barrier.sync 0;
mad.lo.s32 r429, r424, 36, r428;
st.shared.u32 [r429], r169;
st.shared.u32 [r429+12], r260;
st.shared.u32 [r429+24], r297;
barrier.sync 0;
ld.shared.u32 r333, [r423];
ld.shared.u32 r330, [r423+36];
ld.shared.u32 r331, [r423+72];
barrier.sync 0;
st.shared.u32 [r429], r175;
st.shared.u32 [r429+12], r269;
st.shared.u32 [r429+24], r306;
barrier.sync 0;
ld.shared.u32 r339, [r423];
ld.shared.u32 r336, [r423+36];
ld.shared.u32 r337, [r423+72];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f26;
cvt.rn.f16.f32 high, f26;
mov.b32 r327, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f28;
cvt.rn.f16.f32 high, f28;
mov.b32 r328, {low, high};
}
{
add.f16x2 r329, r330, r331;
}
{
add.f16x2 %0, r333, r329;
}
{
add.f16x2 r335, r336, r337;
}
{
add.f16x2 %1, r339, r335;
}
{
add.f16x2 r341, r330, r331;
}
{
mul.f16x2 r344, r341, r327;
}
{
add.f16x2 r347, r333, r344;
}
{
sub.f16x2 r350, r336, r337;
}
{
mul.f16x2 r353, r350, r328;
}
{
add.f16x2 %2, r347, r353;
}
{
add.f16x2 r359, r330, r331;
}
{
mul.f16x2 r362, r359, r327;
}
{
add.f16x2 r365, r333, r362;
}
{
sub.f16x2 r368, r336, r337;
}
{
mul.f16x2 r371, r368, r328;
}
{
sub.f16x2 %4, r365, r371;
}
{
add.f16x2 r377, r336, r337;
}
{
mul.f16x2 r380, r377, r327;
}
{
add.f16x2 r383, r339, r380;
}
{
sub.f16x2 r386, r330, r331;
}
{
mul.f16x2 r389, r386, r328;
}
{
sub.f16x2 %3, r383, r389;
}
{
add.f16x2 r395, r336, r337;
}
{
mul.f16x2 r398, r395, r327;
}
{
add.f16x2 r401, r339, r398;
}
{
sub.f16x2 r404, r330, r331;
}
{
mul.f16x2 r407, r404, r328;
}
{
add.f16x2 %5, r401, r407;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)));
};


#endif
