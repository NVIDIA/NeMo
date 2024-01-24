#ifndef CUFFTDX_FFT_27_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_27_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<867, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<209>;
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
{
neg.f16 rs3, rs2;
}
mov.b32 r81, {rs3, rs3};
{
add.f16x2 r1, %59, %57;
}
{
add.f16x2 r4, %54, r1;
}
{
add.f16x2 r7, %55, %58;
}
{
add.f16x2 r10, %56, r7;
}
{
add.f16x2 r13, %59, %57;
}
{
mul.f16x2 r16, r13, r72;
}
{
add.f16x2 r19, %54, r16;
}
{
sub.f16x2 r22, %55, %58;
}
{
mul.f16x2 r25, r22, r81;
}
{
add.f16x2 r28, r19, r25;
}
{
add.f16x2 r31, %59, %57;
}
{
mul.f16x2 r34, r31, r72;
}
{
add.f16x2 r37, %54, r34;
}
{
sub.f16x2 r40, %55, %58;
}
{
mul.f16x2 r43, r40, r81;
}
{
sub.f16x2 r46, r37, r43;
}
{
add.f16x2 r49, %55, %58;
}
{
mul.f16x2 r52, r49, r72;
}
{
add.f16x2 r55, %56, r52;
}
{
sub.f16x2 r58, %59, %57;
}
{
mul.f16x2 r61, r58, r81;
}
{
sub.f16x2 r64, r55, r61;
}
{
add.f16x2 r67, %55, %58;
}
{
mul.f16x2 r70, r67, r72;
}
{
add.f16x2 r73, %56, r70;
}
{
sub.f16x2 r76, %59, %57;
}
{
mul.f16x2 r79, r76, r81;
}
{
add.f16x2 r82, r73, r79;
}
{
cvt.rn.f16.f64 rs5, fd153;
}
mov.b32 r156, {rs5, rs5};
{
cvt.rn.f16.f64 rs6, fd154;
}
{
neg.f16 rs7, rs6;
}
mov.b32 r165, {rs7, rs7};
{
add.f16x2 r85, %65, %63;
}
{
add.f16x2 r88, %60, r85;
}
{
add.f16x2 r91, %61, %64;
}
{
add.f16x2 r94, %62, r91;
}
{
add.f16x2 r97, %65, %63;
}
{
mul.f16x2 r100, r97, r156;
}
{
add.f16x2 r103, %60, r100;
}
{
sub.f16x2 r106, %61, %64;
}
{
mul.f16x2 r109, r106, r165;
}
{
add.f16x2 r112, r103, r109;
}
{
add.f16x2 r115, %65, %63;
}
{
mul.f16x2 r118, r115, r156;
}
{
add.f16x2 r121, %60, r118;
}
{
sub.f16x2 r124, %61, %64;
}
{
mul.f16x2 r127, r124, r165;
}
{
sub.f16x2 r130, r121, r127;
}
{
add.f16x2 r133, %61, %64;
}
{
mul.f16x2 r136, r133, r156;
}
{
add.f16x2 r139, %62, r136;
}
{
sub.f16x2 r142, %65, %63;
}
{
mul.f16x2 r145, r142, r165;
}
{
sub.f16x2 r148, r139, r145;
}
{
add.f16x2 r151, %61, %64;
}
{
mul.f16x2 r154, r151, r156;
}
{
add.f16x2 r157, %62, r154;
}
{
sub.f16x2 r160, %65, %63;
}
{
mul.f16x2 r163, r160, r165;
}
{
add.f16x2 r166, r157, r163;
}
{
cvt.rn.f16.f64 rs9, fd153;
}
mov.b32 r240, {rs9, rs9};
{
cvt.rn.f16.f64 rs10, fd154;
}
{
neg.f16 rs11, rs10;
}
mov.b32 r249, {rs11, rs11};
{
add.f16x2 r169, %71, %69;
}
{
add.f16x2 r172, %66, r169;
}
{
add.f16x2 r175, %67, %70;
}
{
add.f16x2 r178, %68, r175;
}
{
add.f16x2 r181, %71, %69;
}
{
mul.f16x2 r184, r181, r240;
}
{
add.f16x2 r187, %66, r184;
}
{
sub.f16x2 r190, %67, %70;
}
{
mul.f16x2 r193, r190, r249;
}
{
add.f16x2 r196, r187, r193;
}
{
add.f16x2 r199, %71, %69;
}
{
mul.f16x2 r202, r199, r240;
}
{
add.f16x2 r205, %66, r202;
}
{
sub.f16x2 r208, %67, %70;
}
{
mul.f16x2 r211, r208, r249;
}
{
sub.f16x2 r214, r205, r211;
}
{
add.f16x2 r217, %67, %70;
}
{
mul.f16x2 r220, r217, r240;
}
{
add.f16x2 r223, %68, r220;
}
{
sub.f16x2 r226, %71, %69;
}
{
mul.f16x2 r229, r226, r249;
}
{
sub.f16x2 r232, r223, r229;
}
{
add.f16x2 r235, %67, %70;
}
{
mul.f16x2 r238, r235, r240;
}
{
add.f16x2 r241, %68, r238;
}
{
sub.f16x2 r244, %71, %69;
}
{
mul.f16x2 r247, r244, r249;
}
{
add.f16x2 r250, r241, r247;
}
mov.f64 fd89, 0d3FE8836FA2CF5039;
{
cvt.rn.f16.f64 rs13, fd89;
}
mov.f64 fd90, 0dBFE491B7523C161D;
{
cvt.rn.f16.f64 rs14, fd90;
}
mov.f64 fd95, 0d3FC63A1A7E0B738A;
{
cvt.rn.f16.f64 rs15, fd95;
}
mov.f64 fd96, 0dBFEF838B8C811C17;
{
cvt.rn.f16.f64 rs16, fd96;
}
mov.f64 fd107, 0dBFEE11F642522D1C;
{
cvt.rn.f16.f64 rs19, fd107;
}
mov.f64 fd108, 0dBFD5E3A8748A0BF5;
{
cvt.rn.f16.f64 rs20, fd108;
}
mov.b32 r267, {rs13, rs13};
{
mul.f16x2 r253, r112, r267;
}
mov.b32 r264, {rs14, rs14};
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
mov.b32 r299, {rs15, rs15};
{
mul.f16x2 r269, r196, r299;
}
mov.b32 r296, {rs16, rs16};
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
mov.b32 r315, {rs19, rs19};
{
mul.f16x2 r301, r214, r315;
}
mov.b32 r312, {rs20, rs20};
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
cvt.rn.f16.f64 rs29, fd153;
}
mov.b32 r388, {rs29, rs29};
{
cvt.rn.f16.f64 rs30, fd154;
}
{
neg.f16 rs31, rs30;
}
mov.b32 r397, {rs31, rs31};
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
cvt.rn.f16.f64 rs33, fd153;
}
mov.b32 r472, {rs33, rs33};
{
cvt.rn.f16.f64 rs34, fd154;
}
{
neg.f16 rs35, rs34;
}
mov.b32 r481, {rs35, rs35};
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
cvt.rn.f16.f64 rs37, fd153;
}
mov.b32 r556, {rs37, rs37};
{
cvt.rn.f16.f64 rs38, fd154;
}
{
neg.f16 rs39, rs38;
}
mov.b32 r565, {rs39, rs39};
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
cvt.rn.f16.f64 rs41, fd153;
}
mov.b32 r640, {rs41, rs41};
{
cvt.rn.f16.f64 rs42, fd154;
}
{
neg.f16 rs43, rs42;
}
mov.b32 r649, {rs43, rs43};
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
cvt.rn.f16.f64 rs45, fd153;
}
mov.b32 r724, {rs45, rs45};
{
cvt.rn.f16.f64 rs46, fd154;
}
{
neg.f16 rs47, rs46;
}
mov.b32 r733, {rs47, rs47};
{
add.f16x2 r653, %79, %83;
}
{
add.f16x2 r656, %80, r653;
}
{
add.f16x2 r659, %81, %78;
}
{
add.f16x2 r662, %82, r659;
}
{
add.f16x2 r665, %79, %83;
}
{
mul.f16x2 r668, r665, r724;
}
{
add.f16x2 r671, %80, r668;
}
{
sub.f16x2 r674, %81, %78;
}
{
mul.f16x2 r677, r674, r733;
}
{
add.f16x2 r680, r671, r677;
}
{
add.f16x2 r683, %79, %83;
}
{
mul.f16x2 r686, r683, r724;
}
{
add.f16x2 r689, %80, r686;
}
{
sub.f16x2 r692, %81, %78;
}
{
mul.f16x2 r695, r692, r733;
}
{
sub.f16x2 r698, r689, r695;
}
{
add.f16x2 r701, %81, %78;
}
{
mul.f16x2 r704, r701, r724;
}
{
add.f16x2 r707, %82, r704;
}
{
sub.f16x2 r710, %79, %83;
}
{
mul.f16x2 r713, r710, r733;
}
{
sub.f16x2 r716, r707, r713;
}
{
add.f16x2 r719, %81, %78;
}
{
mul.f16x2 r722, r719, r724;
}
{
add.f16x2 r725, %82, r722;
}
{
sub.f16x2 r728, %79, %83;
}
{
mul.f16x2 r731, r728, r733;
}
{
add.f16x2 r734, r725, r731;
}
{
cvt.rn.f16.f64 rs49, fd153;
}
mov.b32 r808, {rs49, rs49};
{
cvt.rn.f16.f64 rs50, fd154;
}
{
neg.f16 rs51, rs50;
}
mov.b32 r817, {rs51, rs51};
{
add.f16x2 r737, %85, %89;
}
{
add.f16x2 r740, %86, r737;
}
{
add.f16x2 r743, %87, %84;
}
{
add.f16x2 r746, %88, r743;
}
{
add.f16x2 r749, %85, %89;
}
{
mul.f16x2 r752, r749, r808;
}
{
add.f16x2 r755, %86, r752;
}
{
sub.f16x2 r758, %87, %84;
}
{
mul.f16x2 r761, r758, r817;
}
{
add.f16x2 r764, r755, r761;
}
{
add.f16x2 r767, %85, %89;
}
{
mul.f16x2 r770, r767, r808;
}
{
add.f16x2 r773, %86, r770;
}
{
sub.f16x2 r776, %87, %84;
}
{
mul.f16x2 r779, r776, r817;
}
{
sub.f16x2 r782, r773, r779;
}
{
add.f16x2 r785, %87, %84;
}
{
mul.f16x2 r788, r785, r808;
}
{
add.f16x2 r791, %88, r788;
}
{
sub.f16x2 r794, %85, %89;
}
{
mul.f16x2 r797, r794, r817;
}
{
sub.f16x2 r800, r791, r797;
}
{
add.f16x2 r803, %87, %84;
}
{
mul.f16x2 r806, r803, r808;
}
{
add.f16x2 r809, %88, r806;
}
{
sub.f16x2 r812, %85, %89;
}
{
mul.f16x2 r815, r812, r817;
}
{
add.f16x2 r818, r809, r815;
}
{
cvt.rn.f16.f64 rs53, fd89;
}
{
cvt.rn.f16.f64 rs54, fd90;
}
{
cvt.rn.f16.f64 rs55, fd95;
}
{
cvt.rn.f16.f64 rs56, fd96;
}
{
cvt.rn.f16.f64 rs59, fd107;
}
{
cvt.rn.f16.f64 rs60, fd108;
}
mov.b32 r835, {rs53, rs53};
{
mul.f16x2 r821, r680, r835;
}
mov.b32 r832, {rs54, rs54};
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
mov.b32 r867, {rs55, rs55};
{
mul.f16x2 r837, r764, r867;
}
mov.b32 r864, {rs56, rs56};
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
mov.b32 r883, {rs59, rs59};
{
mul.f16x2 r869, r782, r883;
}
mov.b32 r880, {rs60, rs60};
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
cvt.rn.f16.f64 rs69, fd153;
}
mov.b32 r956, {rs69, rs69};
{
cvt.rn.f16.f64 rs70, fd154;
}
{
neg.f16 rs71, rs70;
}
mov.b32 r965, {rs71, rs71};
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
cvt.rn.f16.f64 rs73, fd153;
}
mov.b32 r1040, {rs73, rs73};
{
cvt.rn.f16.f64 rs74, fd154;
}
{
neg.f16 rs75, rs74;
}
mov.b32 r1049, {rs75, rs75};
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
cvt.rn.f16.f64 rs77, fd153;
}
mov.b32 r1124, {rs77, rs77};
{
cvt.rn.f16.f64 rs78, fd154;
}
{
neg.f16 rs79, rs78;
}
mov.b32 r1133, {rs79, rs79};
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
cvt.rn.f16.f64 rs81, fd153;
}
mov.b32 r1208, {rs81, rs81};
{
cvt.rn.f16.f64 rs82, fd154;
}
{
neg.f16 rs83, rs82;
}
mov.b32 r1217, {rs83, rs83};
{
add.f16x2 r1137, %91, %95;
}
{
add.f16x2 r1140, %92, r1137;
}
{
add.f16x2 r1143, %93, %90;
}
{
add.f16x2 r1146, %94, r1143;
}
{
add.f16x2 r1149, %91, %95;
}
{
mul.f16x2 r1152, r1149, r1208;
}
{
add.f16x2 r1155, %92, r1152;
}
{
sub.f16x2 r1158, %93, %90;
}
{
mul.f16x2 r1161, r1158, r1217;
}
{
add.f16x2 r1164, r1155, r1161;
}
{
add.f16x2 r1167, %91, %95;
}
{
mul.f16x2 r1170, r1167, r1208;
}
{
add.f16x2 r1173, %92, r1170;
}
{
sub.f16x2 r1176, %93, %90;
}
{
mul.f16x2 r1179, r1176, r1217;
}
{
sub.f16x2 r1182, r1173, r1179;
}
{
add.f16x2 r1185, %93, %90;
}
{
mul.f16x2 r1188, r1185, r1208;
}
{
add.f16x2 r1191, %94, r1188;
}
{
sub.f16x2 r1194, %91, %95;
}
{
mul.f16x2 r1197, r1194, r1217;
}
{
sub.f16x2 r1200, r1191, r1197;
}
{
add.f16x2 r1203, %93, %90;
}
{
mul.f16x2 r1206, r1203, r1208;
}
{
add.f16x2 r1209, %94, r1206;
}
{
sub.f16x2 r1212, %91, %95;
}
{
mul.f16x2 r1215, r1212, r1217;
}
{
add.f16x2 r1218, r1209, r1215;
}
{
cvt.rn.f16.f64 rs85, fd153;
}
mov.b32 r1292, {rs85, rs85};
{
cvt.rn.f16.f64 rs86, fd154;
}
{
neg.f16 rs87, rs86;
}
mov.b32 r1301, {rs87, rs87};
{
add.f16x2 r1221, %97, %101;
}
{
add.f16x2 r1224, %98, r1221;
}
{
add.f16x2 r1227, %99, %96;
}
{
add.f16x2 r1230, %100, r1227;
}
{
add.f16x2 r1233, %97, %101;
}
{
mul.f16x2 r1236, r1233, r1292;
}
{
add.f16x2 r1239, %98, r1236;
}
{
sub.f16x2 r1242, %99, %96;
}
{
mul.f16x2 r1245, r1242, r1301;
}
{
add.f16x2 r1248, r1239, r1245;
}
{
add.f16x2 r1251, %97, %101;
}
{
mul.f16x2 r1254, r1251, r1292;
}
{
add.f16x2 r1257, %98, r1254;
}
{
sub.f16x2 r1260, %99, %96;
}
{
mul.f16x2 r1263, r1260, r1301;
}
{
sub.f16x2 r1266, r1257, r1263;
}
{
add.f16x2 r1269, %99, %96;
}
{
mul.f16x2 r1272, r1269, r1292;
}
{
add.f16x2 r1275, %100, r1272;
}
{
sub.f16x2 r1278, %97, %101;
}
{
mul.f16x2 r1281, r1278, r1301;
}
{
sub.f16x2 r1284, r1275, r1281;
}
{
add.f16x2 r1287, %99, %96;
}
{
mul.f16x2 r1290, r1287, r1292;
}
{
add.f16x2 r1293, %100, r1290;
}
{
sub.f16x2 r1296, %97, %101;
}
{
mul.f16x2 r1299, r1296, r1301;
}
{
add.f16x2 r1302, r1293, r1299;
}
{
cvt.rn.f16.f64 rs89, fd153;
}
mov.b32 r1376, {rs89, rs89};
{
cvt.rn.f16.f64 rs90, fd154;
}
{
neg.f16 rs91, rs90;
}
mov.b32 r1385, {rs91, rs91};
{
add.f16x2 r1305, %103, %107;
}
{
add.f16x2 r1308, %104, r1305;
}
{
add.f16x2 r1311, %105, %102;
}
{
add.f16x2 r1314, %106, r1311;
}
{
add.f16x2 r1317, %103, %107;
}
{
mul.f16x2 r1320, r1317, r1376;
}
{
add.f16x2 r1323, %104, r1320;
}
{
sub.f16x2 r1326, %105, %102;
}
{
mul.f16x2 r1329, r1326, r1385;
}
{
add.f16x2 r1332, r1323, r1329;
}
{
add.f16x2 r1335, %103, %107;
}
{
mul.f16x2 r1338, r1335, r1376;
}
{
add.f16x2 r1341, %104, r1338;
}
{
sub.f16x2 r1344, %105, %102;
}
{
mul.f16x2 r1347, r1344, r1385;
}
{
sub.f16x2 r1350, r1341, r1347;
}
{
add.f16x2 r1353, %105, %102;
}
{
mul.f16x2 r1356, r1353, r1376;
}
{
add.f16x2 r1359, %106, r1356;
}
{
sub.f16x2 r1362, %103, %107;
}
{
mul.f16x2 r1365, r1362, r1385;
}
{
sub.f16x2 r1368, r1359, r1365;
}
{
add.f16x2 r1371, %105, %102;
}
{
mul.f16x2 r1374, r1371, r1376;
}
{
add.f16x2 r1377, %106, r1374;
}
{
sub.f16x2 r1380, %103, %107;
}
{
mul.f16x2 r1383, r1380, r1385;
}
{
add.f16x2 r1386, r1377, r1383;
}
{
cvt.rn.f16.f64 rs93, fd89;
}
{
cvt.rn.f16.f64 rs94, fd90;
}
{
cvt.rn.f16.f64 rs95, fd95;
}
{
cvt.rn.f16.f64 rs96, fd96;
}
{
cvt.rn.f16.f64 rs99, fd107;
}
{
cvt.rn.f16.f64 rs100, fd108;
}
mov.b32 r1403, {rs93, rs93};
{
mul.f16x2 r1389, r1248, r1403;
}
mov.b32 r1400, {rs94, rs94};
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
mov.b32 r1435, {rs95, rs95};
{
mul.f16x2 r1405, r1332, r1435;
}
mov.b32 r1432, {rs96, rs96};
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
mov.b32 r1451, {rs99, rs99};
{
mul.f16x2 r1437, r1350, r1451;
}
mov.b32 r1448, {rs100, rs100};
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
cvt.rn.f16.f64 rs109, fd153;
}
mov.b32 r1524, {rs109, rs109};
{
cvt.rn.f16.f64 rs110, fd154;
}
{
neg.f16 rs111, rs110;
}
mov.b32 r1533, {rs111, rs111};
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
cvt.rn.f16.f64 rs113, fd153;
}
mov.b32 r1608, {rs113, rs113};
{
cvt.rn.f16.f64 rs114, fd154;
}
{
neg.f16 rs115, rs114;
}
mov.b32 r1617, {rs115, rs115};
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
cvt.rn.f16.f64 rs117, fd153;
}
mov.b32 r1692, {rs117, rs117};
{
cvt.rn.f16.f64 rs118, fd154;
}
{
neg.f16 rs119, rs118;
}
mov.b32 r1701, {rs119, rs119};
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
cvt.rn.f16.f64 rs121, fd85;
}
mov.f64 fd86, 0dBFCD84D223638000;
{
cvt.rn.f16.f64 rs122, fd86;
}
mov.f64 fd87, 0d3FEC98A37A9A7850;
{
cvt.rn.f16.f64 rs123, fd87;
}
mov.f64 fd88, 0dBFDCB920325BAFA6;
{
cvt.rn.f16.f64 rs124, fd88;
}
{
cvt.rn.f16.f64 rs125, fd89;
}
{
cvt.rn.f16.f64 rs126, fd90;
}
mov.f64 fd91, 0d3FE31BEC55BC71BC;
{
cvt.rn.f16.f64 rs127, fd91;
}
mov.f64 fd92, 0dBFE9AAFE4207DF5F;
{
cvt.rn.f16.f64 rs128, fd92;
}
mov.f64 fd93, 0d3FD9595EF26FB670;
{
cvt.rn.f16.f64 rs129, fd93;
}
mov.f64 fd94, 0dBFED6206BEB6C24B;
{
cvt.rn.f16.f64 rs130, fd94;
}
{
cvt.rn.f16.f64 rs131, fd95;
}
{
cvt.rn.f16.f64 rs132, fd96;
}
mov.f64 fd97, 0dBFADC528B5343A86;
{
cvt.rn.f16.f64 rs133, fd97;
}
mov.f64 fd98, 0dBFEFF223F3635CE3;
{
cvt.rn.f16.f64 rs134, fd98;
}
mov.f64 fd99, 0dBFD25AFBF23865BF;
{
cvt.rn.f16.f64 rs135, fd99;
}
mov.f64 fd100, 0dBFEEA7D99F29CADE;
{
cvt.rn.f16.f64 rs136, fd100;
}
mov.f64 fd103, 0dBFE5F5B105F99707;
{
cvt.rn.f16.f64 rs139, fd103;
}
mov.f64 fd104, 0dBFE746A51650EADE;
{
cvt.rn.f16.f64 rs140, fd104;
}
mov.f64 fd115, 0dBFEABC50EF4734A7;
{
cvt.rn.f16.f64 rs143, fd107;
}
{
cvt.rn.f16.f64 rs144, fd108;
}
mov.f64 fd111, 0dBFEFC89BCEF44CF4;
{
cvt.rn.f16.f64 rs147, fd111;
}
mov.f64 fd112, 0d3FBDB843E577175E;
{
cvt.rn.f16.f64 rs148, fd112;
}
{
cvt.rn.f16.f64 rs151, fd115;
}
mov.f64 fd116, 0d3FE19593DA358510;
{
cvt.rn.f16.f64 rs152, fd116;
}
mov.b32 r1719, {rs121, rs121};
{
mul.f16x2 r1705, r972, r1719;
}
mov.b32 r1716, {rs122, rs122};
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
mov.b32 r1751, {rs123, rs123};
{
mul.f16x2 r1721, r1540, r1751;
}
mov.b32 r1748, {rs124, rs124};
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
mov.b32 r1815, {rs127, rs127};
{
mul.f16x2 r1753, r1624, r1815;
}
mov.b32 r1812, {rs128, rs128};
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
mov.b32 r1783, {rs125, rs125};
{
mul.f16x2 r1769, r912, r1783;
}
mov.b32 r1780, {rs126, rs126};
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
mov.b32 r1879, {rs131, rs131};
{
mul.f16x2 r1785, r1480, r1879;
}
mov.b32 r1876, {rs132, rs132};
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
mov.b32 r1943, {rs135, rs135};
{
mul.f16x2 r1817, r1564, r1943;
}
mov.b32 r1940, {rs136, rs136};
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
mov.b32 r1847, {rs129, rs129};
{
mul.f16x2 r1833, r1080, r1847;
}
mov.b32 r1844, {rs130, rs130};
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
mov.b32 r1863, {rs139, rs139};
{
mul.f16x2 r1849, r1648, r1863;
}
mov.b32 r1860, {rs140, rs140};
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
mov.b32 r1895, {rs143, rs143};
{
mul.f16x2 r1881, r1498, r1895;
}
mov.b32 r1892, {rs144, rs144};
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
mov.b32 r1911, {rs133, rs133};
{
mul.f16x2 r1897, r1014, r1911;
}
mov.b32 r1908, {rs134, rs134};
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
mov.b32 r1927, {rs147, rs147};
{
mul.f16x2 r1913, r1582, r1927;
}
mov.b32 r1924, {rs148, rs148};
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
mov.b32 r1959, {rs151, rs151};
{
mul.f16x2 r1945, r1666, r1959;
}
mov.b32 r1956, {rs152, rs152};
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
cvt.rn.f16.f64 rs173, fd153;
}
mov.b32 r2032, {rs173, rs173};
{
cvt.rn.f16.f64 rs174, fd154;
}
{
neg.f16 rs175, rs174;
}
mov.b32 r2041, {rs175, rs175};
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
cvt.rn.f16.f64 rs177, fd153;
}
mov.b32 r2116, {rs177, rs177};
{
cvt.rn.f16.f64 rs178, fd154;
}
{
neg.f16 rs179, rs178;
}
mov.b32 r2125, {rs179, rs179};
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
cvt.rn.f16.f64 rs181, fd153;
}
mov.b32 r2200, {rs181, rs181};
{
cvt.rn.f16.f64 rs182, fd154;
}
{
neg.f16 rs183, rs182;
}
mov.b32 r2209, {rs183, rs183};
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
cvt.rn.f16.f64 rs185, fd153;
}
mov.b32 r2284, {rs185, rs185};
{
cvt.rn.f16.f64 rs186, fd154;
}
{
neg.f16 rs187, rs186;
}
mov.b32 r2293, {rs187, rs187};
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
cvt.rn.f16.f64 rs189, fd153;
}
mov.b32 r2368, {rs189, rs189};
{
cvt.rn.f16.f64 rs190, fd154;
}
{
neg.f16 rs191, rs190;
}
mov.b32 r2377, {rs191, rs191};
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
cvt.rn.f16.f64 rs193, fd153;
}
mov.b32 r2452, {rs193, rs193};
{
cvt.rn.f16.f64 rs194, fd154;
}
{
neg.f16 rs195, rs194;
}
mov.b32 r2461, {rs195, rs195};
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
cvt.rn.f16.f64 rs197, fd153;
}
mov.b32 r2536, {rs197, rs197};
{
cvt.rn.f16.f64 rs198, fd154;
}
{
neg.f16 rs199, rs198;
}
mov.b32 r2545, {rs199, rs199};
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
cvt.rn.f16.f64 rs201, fd153;
}
mov.b32 r2620, {rs201, rs201};
{
cvt.rn.f16.f64 rs202, fd154;
}
{
neg.f16 rs203, rs202;
}
mov.b32 r2629, {rs203, rs203};
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
cvt.rn.f16.f64 rs205, fd153;
}
mov.b32 r2704, {rs205, rs205};
{
cvt.rn.f16.f64 rs206, fd154;
}
{
neg.f16 rs207, rs206;
}
mov.b32 r2713, {rs207, rs207};
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
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[26].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<869, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<92>;
.reg .b32 r<1183>;
.reg .b64 rd<4>;
mov.u32 r1172, %tid.y;
mov.u32 r1173, %18;
mad.lo.s32 r1174, r1172, 216, r1173;
mov.u32 r1175, %tid.x;
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
neg.f16x2 r3, r2;
}
{
add.f16x2 r5, %25, %31;
}
{
add.f16x2 r8, %19, r5;
}
{
add.f16x2 r11, %26, %32;
}
{
add.f16x2 r14, %20, r11;
}
{
add.f16x2 r17, %25, %31;
}
{
mul.f16x2 r20, r17, r1;
}
{
add.f16x2 r23, %19, r20;
}
{
sub.f16x2 r26, %26, %32;
}
{
mul.f16x2 r29, r26, r3;
}
{
add.f16x2 r32, r23, r29;
}
{
add.f16x2 r35, %25, %31;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %19, r38;
}
{
sub.f16x2 r44, %26, %32;
}
{
mul.f16x2 r47, r44, r3;
}
{
sub.f16x2 r50, r41, r47;
}
{
add.f16x2 r53, %26, %32;
}
{
mul.f16x2 r56, r53, r1;
}
{
add.f16x2 r59, %20, r56;
}
{
sub.f16x2 r62, %25, %31;
}
{
mul.f16x2 r65, r62, r3;
}
{
sub.f16x2 r68, r59, r65;
}
{
add.f16x2 r71, %26, %32;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %20, r74;
}
{
sub.f16x2 r80, %25, %31;
}
{
mul.f16x2 r83, r80, r3;
}
{
add.f16x2 r86, r77, r83;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r90, {low, high};
}
{
neg.f16x2 r91, r90;
}
{
add.f16x2 r93, %27, %33;
}
{
add.f16x2 r96, %21, r93;
}
{
add.f16x2 r99, %28, %34;
}
{
add.f16x2 r102, %22, r99;
}
{
add.f16x2 r105, %27, %33;
}
{
mul.f16x2 r108, r105, r89;
}
{
add.f16x2 r111, %21, r108;
}
{
sub.f16x2 r114, %28, %34;
}
{
mul.f16x2 r117, r114, r91;
}
{
add.f16x2 r120, r111, r117;
}
{
add.f16x2 r123, %27, %33;
}
{
mul.f16x2 r126, r123, r89;
}
{
add.f16x2 r129, %21, r126;
}
{
sub.f16x2 r132, %28, %34;
}
{
mul.f16x2 r135, r132, r91;
}
{
sub.f16x2 r138, r129, r135;
}
{
add.f16x2 r141, %28, %34;
}
{
mul.f16x2 r144, r141, r89;
}
{
add.f16x2 r147, %22, r144;
}
{
sub.f16x2 r150, %27, %33;
}
{
mul.f16x2 r153, r150, r91;
}
{
sub.f16x2 r156, r147, r153;
}
{
add.f16x2 r159, %28, %34;
}
{
mul.f16x2 r162, r159, r89;
}
{
add.f16x2 r165, %22, r162;
}
{
sub.f16x2 r168, %27, %33;
}
{
mul.f16x2 r171, r168, r91;
}
{
add.f16x2 r174, r165, r171;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r177, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r178, {low, high};
}
{
neg.f16x2 r179, r178;
}
{
add.f16x2 r181, %29, %35;
}
{
add.f16x2 r184, %23, r181;
}
{
add.f16x2 r187, %30, %36;
}
{
add.f16x2 r190, %24, r187;
}
{
add.f16x2 r193, %29, %35;
}
{
mul.f16x2 r196, r193, r177;
}
{
add.f16x2 r199, %23, r196;
}
{
sub.f16x2 r202, %30, %36;
}
{
mul.f16x2 r205, r202, r179;
}
{
add.f16x2 r208, r199, r205;
}
{
add.f16x2 r211, %29, %35;
}
{
mul.f16x2 r214, r211, r177;
}
{
add.f16x2 r217, %23, r214;
}
{
sub.f16x2 r220, %30, %36;
}
{
mul.f16x2 r223, r220, r179;
}
{
sub.f16x2 r226, r217, r223;
}
{
add.f16x2 r229, %30, %36;
}
{
mul.f16x2 r232, r229, r177;
}
{
add.f16x2 r235, %24, r232;
}
{
sub.f16x2 r238, %29, %35;
}
{
mul.f16x2 r241, r238, r179;
}
{
sub.f16x2 r244, r235, r241;
}
{
add.f16x2 r247, %30, %36;
}
{
mul.f16x2 r250, r247, r177;
}
{
add.f16x2 r253, %24, r250;
}
{
sub.f16x2 r256, %29, %35;
}
{
mul.f16x2 r259, r256, r179;
}
{
add.f16x2 r262, r253, r259;
}
mov.f32 f14, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f14;
cvt.rn.f16.f32 high, f14;
mov.b32 r265, {low, high};
}
mov.f32 f16, 0fBF248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f16;
cvt.rn.f16.f32 high, f16;
mov.b32 r266, {low, high};
}
mov.f32 f18, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f18;
cvt.rn.f16.f32 high, f18;
mov.b32 r267, {low, high};
}
mov.f32 f20, 0fBF7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f20;
cvt.rn.f16.f32 high, f20;
mov.b32 r268, {low, high};
}
mov.f32 f26, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f26;
cvt.rn.f16.f32 high, f26;
mov.b32 r271, {low, high};
}
mov.f32 f28, 0fBEAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f28;
cvt.rn.f16.f32 high, f28;
mov.b32 r272, {low, high};
}
{
mul.f16x2 r281, r120, r265;
}
{
mul.f16x2 r284, r156, r266;
}
{
sub.f16x2 r287, r281, r284;
}
{
mul.f16x2 r290, r120, r266;
}
{
fma.rn.f16x2 r293, r156, r265, r290;
}
{
mul.f16x2 r297, r208, r267;
}
{
mul.f16x2 r300, r244, r268;
}
{
sub.f16x2 r303, r297, r300;
}
{
mul.f16x2 r306, r208, r268;
}
{
fma.rn.f16x2 r309, r244, r267, r306;
}
{
mul.f16x2 r313, r138, r267;
}
{
mul.f16x2 r316, r174, r268;
}
{
sub.f16x2 r319, r313, r316;
}
{
mul.f16x2 r322, r138, r268;
}
{
fma.rn.f16x2 r325, r174, r267, r322;
}
{
mul.f16x2 r329, r226, r271;
}
{
mul.f16x2 r332, r262, r272;
}
{
sub.f16x2 r335, r329, r332;
}
{
mul.f16x2 r338, r226, r272;
}
{
fma.rn.f16x2 r341, r262, r271, r338;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r345, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r346, {low, high};
}
{
neg.f16x2 r347, r346;
}
{
add.f16x2 r349, r96, r184;
}
{
add.f16x2 r352, r8, r349;
}
{
add.f16x2 r355, r102, r190;
}
{
add.f16x2 r358, r14, r355;
}
{
add.f16x2 r361, r96, r184;
}
{
mul.f16x2 r364, r361, r345;
}
{
add.f16x2 r367, r8, r364;
}
{
sub.f16x2 r370, r102, r190;
}
{
mul.f16x2 r373, r370, r347;
}
{
add.f16x2 r376, r367, r373;
}
{
add.f16x2 r379, r96, r184;
}
{
mul.f16x2 r382, r379, r345;
}
{
add.f16x2 r385, r8, r382;
}
{
sub.f16x2 r388, r102, r190;
}
{
mul.f16x2 r391, r388, r347;
}
{
sub.f16x2 r394, r385, r391;
}
{
add.f16x2 r397, r102, r190;
}
{
mul.f16x2 r400, r397, r345;
}
{
add.f16x2 r403, r14, r400;
}
{
sub.f16x2 r406, r96, r184;
}
{
mul.f16x2 r409, r406, r347;
}
{
sub.f16x2 r412, r403, r409;
}
{
add.f16x2 r415, r102, r190;
}
{
mul.f16x2 r418, r415, r345;
}
{
add.f16x2 r421, r14, r418;
}
{
sub.f16x2 r424, r96, r184;
}
{
mul.f16x2 r427, r424, r347;
}
{
add.f16x2 r430, r421, r427;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r433, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r434, {low, high};
}
{
neg.f16x2 r435, r434;
}
{
add.f16x2 r437, r287, r303;
}
{
add.f16x2 r440, r32, r437;
}
{
add.f16x2 r443, r293, r309;
}
{
add.f16x2 r446, r68, r443;
}
{
add.f16x2 r449, r287, r303;
}
{
mul.f16x2 r452, r449, r433;
}
{
add.f16x2 r455, r32, r452;
}
{
sub.f16x2 r458, r293, r309;
}
{
mul.f16x2 r461, r458, r435;
}
{
add.f16x2 r464, r455, r461;
}
{
add.f16x2 r467, r287, r303;
}
{
mul.f16x2 r470, r467, r433;
}
{
add.f16x2 r473, r32, r470;
}
{
sub.f16x2 r476, r293, r309;
}
{
mul.f16x2 r479, r476, r435;
}
{
sub.f16x2 r482, r473, r479;
}
{
add.f16x2 r485, r293, r309;
}
{
mul.f16x2 r488, r485, r433;
}
{
add.f16x2 r491, r68, r488;
}
{
sub.f16x2 r494, r287, r303;
}
{
mul.f16x2 r497, r494, r435;
}
{
sub.f16x2 r500, r491, r497;
}
{
add.f16x2 r503, r293, r309;
}
{
mul.f16x2 r506, r503, r433;
}
{
add.f16x2 r509, r68, r506;
}
{
sub.f16x2 r512, r287, r303;
}
{
mul.f16x2 r515, r512, r435;
}
{
add.f16x2 r518, r509, r515;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r521, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r522, {low, high};
}
{
neg.f16x2 r523, r522;
}
{
add.f16x2 r525, r319, r335;
}
{
add.f16x2 r528, r50, r525;
}
{
add.f16x2 r531, r325, r341;
}
{
add.f16x2 r534, r86, r531;
}
{
add.f16x2 r537, r319, r335;
}
{
mul.f16x2 r540, r537, r521;
}
{
add.f16x2 r543, r50, r540;
}
{
sub.f16x2 r546, r325, r341;
}
{
mul.f16x2 r549, r546, r523;
}
{
add.f16x2 r552, r543, r549;
}
{
add.f16x2 r555, r319, r335;
}
{
mul.f16x2 r558, r555, r521;
}
{
add.f16x2 r561, r50, r558;
}
{
sub.f16x2 r564, r325, r341;
}
{
mul.f16x2 r567, r564, r523;
}
{
sub.f16x2 r570, r561, r567;
}
{
add.f16x2 r573, r325, r341;
}
{
mul.f16x2 r576, r573, r521;
}
{
add.f16x2 r579, r86, r576;
}
{
sub.f16x2 r582, r319, r335;
}
{
mul.f16x2 r585, r582, r523;
}
{
sub.f16x2 r588, r579, r585;
}
{
add.f16x2 r591, r325, r341;
}
{
mul.f16x2 r594, r591, r521;
}
{
add.f16x2 r597, r86, r594;
}
{
sub.f16x2 r600, r319, r335;
}
{
mul.f16x2 r603, r600, r523;
}
{
add.f16x2 r606, r597, r603;
}
mul.wide.u32 rd2, r1175, -1431655765;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r1176, rd3;
mul.lo.s32 r1177, r1176, 3;
sub.s32 r1178, r1175, r1177;
cvt.rn.f32.u32 f89, r1178;
mul.f32 f90, f89, 0f3E6E4BAE;
cos.approx.f32 f57, f90;
sin.approx.f32 f91, f90;
neg.f32 f58, f91;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f57;
cvt.rn.f16.f32 high, f58;
mov.b32 r609, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r612, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r614, {high, high};
}
{
mul.f16x2 r616, r446, r614;
}
{
neg.f16x2 r619, r616;
}
{
fma.rn.f16x2 r621, r440, r612, r619;
}
{
mul.f16x2 r625, r440, r614;
}
{
fma.rn.f16x2 r628, r446, r612, r625;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r632, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r634, {high, high};
}
mov.f32 f73, 0fBF800000;
mov.f32 f74, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r636, {low, high};
}
{
mul.f16x2 r637, r634, r636;
}
{
mul.f16x2 r640, r609, r632;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r643, {high, low};
}
{
fma.rn.f16x2 r645, r637, r643, r640;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r645;
mov.b32 r649, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r645;
mov.b32 r651, {high, high};
}
{
mul.f16x2 r653, r534, r651;
}
{
neg.f16x2 r656, r653;
}
{
fma.rn.f16x2 r658, r528, r649, r656;
}
{
mul.f16x2 r662, r528, r651;
}
{
fma.rn.f16x2 r665, r534, r649, r662;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r669, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r671, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r673, {low, high};
}
{
mul.f16x2 r674, r671, r673;
}
{
mul.f16x2 r677, r645, r669;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r645;
mov.b32 r680, {high, low};
}
{
fma.rn.f16x2 r682, r674, r680, r677;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r682;
mov.b32 r686, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r682;
mov.b32 r688, {high, high};
}
{
mul.f16x2 r690, r412, r688;
}
{
neg.f16x2 r693, r690;
}
{
fma.rn.f16x2 r695, r376, r686, r693;
}
{
mul.f16x2 r699, r376, r688;
}
{
fma.rn.f16x2 r702, r412, r686, r699;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r706, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r708, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r710, {low, high};
}
{
mul.f16x2 r711, r708, r710;
}
{
mul.f16x2 r714, r682, r706;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r682;
mov.b32 r717, {high, low};
}
{
fma.rn.f16x2 r719, r711, r717, r714;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r719;
mov.b32 r723, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r719;
mov.b32 r725, {high, high};
}
{
mul.f16x2 r727, r500, r725;
}
{
neg.f16x2 r730, r727;
}
{
fma.rn.f16x2 r732, r464, r723, r730;
}
{
mul.f16x2 r736, r464, r725;
}
{
fma.rn.f16x2 r739, r500, r723, r736;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r743, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r745, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r747, {low, high};
}
{
mul.f16x2 r748, r745, r747;
}
{
mul.f16x2 r751, r719, r743;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r719;
mov.b32 r754, {high, low};
}
{
fma.rn.f16x2 r756, r748, r754, r751;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r756;
mov.b32 r760, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r756;
mov.b32 r762, {high, high};
}
{
mul.f16x2 r764, r588, r762;
}
{
neg.f16x2 r767, r764;
}
{
fma.rn.f16x2 r769, r552, r760, r767;
}
{
mul.f16x2 r773, r552, r762;
}
{
fma.rn.f16x2 r776, r588, r760, r773;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r780, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r782, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r784, {low, high};
}
{
mul.f16x2 r785, r782, r784;
}
{
mul.f16x2 r788, r756, r780;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r756;
mov.b32 r791, {high, low};
}
{
fma.rn.f16x2 r793, r785, r791, r788;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r793;
mov.b32 r797, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r793;
mov.b32 r799, {high, high};
}
{
mul.f16x2 r801, r430, r799;
}
{
neg.f16x2 r804, r801;
}
{
fma.rn.f16x2 r806, r394, r797, r804;
}
{
mul.f16x2 r810, r394, r799;
}
{
fma.rn.f16x2 r813, r430, r797, r810;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r817, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r819, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r821, {low, high};
}
{
mul.f16x2 r822, r819, r821;
}
{
mul.f16x2 r825, r793, r817;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r793;
mov.b32 r828, {high, low};
}
{
fma.rn.f16x2 r830, r822, r828, r825;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r830;
mov.b32 r834, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r830;
mov.b32 r836, {high, high};
}
{
mul.f16x2 r838, r518, r836;
}
{
neg.f16x2 r841, r838;
}
{
fma.rn.f16x2 r843, r482, r834, r841;
}
{
mul.f16x2 r847, r482, r836;
}
{
fma.rn.f16x2 r850, r518, r834, r847;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r854, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r856, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r858, {low, high};
}
{
mul.f16x2 r859, r856, r858;
}
{
mul.f16x2 r862, r830, r854;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r830;
mov.b32 r865, {high, low};
}
{
fma.rn.f16x2 r867, r859, r865, r862;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r867;
mov.b32 r871, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r867;
mov.b32 r873, {high, high};
}
{
mul.f16x2 r875, r606, r873;
}
{
neg.f16x2 r878, r875;
}
{
fma.rn.f16x2 r880, r570, r871, r878;
}
{
mul.f16x2 r884, r570, r873;
}
{
fma.rn.f16x2 r887, r606, r871, r884;
}
mad.lo.s32 r1179, r1176, 216, r1174;
barrier.sync 0;
mad.lo.s32 r1180, r1178, 72, r1179;
st.shared.v2.f32 [r1180], {r352, r358};
st.shared.v2.f32 [r1180+8], {r621, r628};
st.shared.v2.f32 [r1180+16], {r658, r665};
st.shared.v2.f32 [r1180+24], {r695, r702};
st.shared.v2.f32 [r1180+32], {r732, r739};
st.shared.v2.f32 [r1180+40], {r769, r776};
st.shared.v2.f32 [r1180+48], {r806, r813};
st.shared.v2.f32 [r1180+56], {r843, r850};
st.shared.v2.f32 [r1180+64], {r880, r887};
barrier.sync 0;
shl.b32 r1181, r1178, 6;
sub.s32 r1182, r1180, r1181;
ld.shared.u32 r916, [r1182];
ld.shared.u32 r922, [r1182+4];
ld.shared.u32 r1004, [r1182+24];
ld.shared.u32 r1010, [r1182+28];
ld.shared.u32 r1092, [r1182+48];
ld.shared.u32 r1098, [r1182+52];
ld.shared.u32 r913, [r1182+72];
ld.shared.u32 r919, [r1182+76];
ld.shared.u32 r1001, [r1182+96];
ld.shared.u32 r1007, [r1182+100];
ld.shared.u32 r1089, [r1182+120];
ld.shared.u32 r1095, [r1182+124];
ld.shared.u32 r914, [r1182+144];
ld.shared.u32 r920, [r1182+148];
ld.shared.u32 r1002, [r1182+168];
ld.shared.u32 r1008, [r1182+172];
ld.shared.u32 r1090, [r1182+192];
ld.shared.u32 r1096, [r1182+196];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r908, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r909, {low, high};
}
{
neg.f16x2 r910, r909;
}
{
add.f16x2 r912, r913, r914;
}
{
add.f16x2 %0, r916, r912;
}
{
add.f16x2 r918, r919, r920;
}
{
add.f16x2 %1, r922, r918;
}
{
add.f16x2 r924, r913, r914;
}
{
mul.f16x2 r927, r924, r908;
}
{
add.f16x2 r930, r916, r927;
}
{
sub.f16x2 r933, r919, r920;
}
{
mul.f16x2 r936, r933, r910;
}
{
add.f16x2 %6, r930, r936;
}
{
add.f16x2 r942, r913, r914;
}
{
mul.f16x2 r945, r942, r908;
}
{
add.f16x2 r948, r916, r945;
}
{
sub.f16x2 r951, r919, r920;
}
{
mul.f16x2 r954, r951, r910;
}
{
sub.f16x2 %12, r948, r954;
}
{
add.f16x2 r960, r919, r920;
}
{
mul.f16x2 r963, r960, r908;
}
{
add.f16x2 r966, r922, r963;
}
{
sub.f16x2 r969, r913, r914;
}
{
mul.f16x2 r972, r969, r910;
}
{
sub.f16x2 %7, r966, r972;
}
{
add.f16x2 r978, r919, r920;
}
{
mul.f16x2 r981, r978, r908;
}
{
add.f16x2 r984, r922, r981;
}
{
sub.f16x2 r987, r913, r914;
}
{
mul.f16x2 r990, r987, r910;
}
{
add.f16x2 %13, r984, r990;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r996, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r997, {low, high};
}
{
neg.f16x2 r998, r997;
}
{
add.f16x2 r1000, r1001, r1002;
}
{
add.f16x2 %2, r1004, r1000;
}
{
add.f16x2 r1006, r1007, r1008;
}
{
add.f16x2 %3, r1010, r1006;
}
{
add.f16x2 r1012, r1001, r1002;
}
{
mul.f16x2 r1015, r1012, r996;
}
{
add.f16x2 r1018, r1004, r1015;
}
{
sub.f16x2 r1021, r1007, r1008;
}
{
mul.f16x2 r1024, r1021, r998;
}
{
add.f16x2 %8, r1018, r1024;
}
{
add.f16x2 r1030, r1001, r1002;
}
{
mul.f16x2 r1033, r1030, r996;
}
{
add.f16x2 r1036, r1004, r1033;
}
{
sub.f16x2 r1039, r1007, r1008;
}
{
mul.f16x2 r1042, r1039, r998;
}
{
sub.f16x2 %14, r1036, r1042;
}
{
add.f16x2 r1048, r1007, r1008;
}
{
mul.f16x2 r1051, r1048, r996;
}
{
add.f16x2 r1054, r1010, r1051;
}
{
sub.f16x2 r1057, r1001, r1002;
}
{
mul.f16x2 r1060, r1057, r998;
}
{
sub.f16x2 %9, r1054, r1060;
}
{
add.f16x2 r1066, r1007, r1008;
}
{
mul.f16x2 r1069, r1066, r996;
}
{
add.f16x2 r1072, r1010, r1069;
}
{
sub.f16x2 r1075, r1001, r1002;
}
{
mul.f16x2 r1078, r1075, r998;
}
{
add.f16x2 %15, r1072, r1078;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r1084, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r1085, {low, high};
}
{
neg.f16x2 r1086, r1085;
}
{
add.f16x2 r1088, r1089, r1090;
}
{
add.f16x2 %4, r1092, r1088;
}
{
add.f16x2 r1094, r1095, r1096;
}
{
add.f16x2 %5, r1098, r1094;
}
{
add.f16x2 r1100, r1089, r1090;
}
{
mul.f16x2 r1103, r1100, r1084;
}
{
add.f16x2 r1106, r1092, r1103;
}
{
sub.f16x2 r1109, r1095, r1096;
}
{
mul.f16x2 r1112, r1109, r1086;
}
{
add.f16x2 %10, r1106, r1112;
}
{
add.f16x2 r1118, r1089, r1090;
}
{
mul.f16x2 r1121, r1118, r1084;
}
{
add.f16x2 r1124, r1092, r1121;
}
{
sub.f16x2 r1127, r1095, r1096;
}
{
mul.f16x2 r1130, r1127, r1086;
}
{
sub.f16x2 %16, r1124, r1130;
}
{
add.f16x2 r1136, r1095, r1096;
}
{
mul.f16x2 r1139, r1136, r1084;
}
{
add.f16x2 r1142, r1098, r1139;
}
{
sub.f16x2 r1145, r1089, r1090;
}
{
mul.f16x2 r1148, r1145, r1086;
}
{
sub.f16x2 %11, r1142, r1148;
}
{
add.f16x2 r1154, r1095, r1096;
}
{
mul.f16x2 r1157, r1154, r1084;
}
{
add.f16x2 r1160, r1098, r1157;
}
{
sub.f16x2 r1163, r1089, r1090;
}
{
mul.f16x2 r1166, r1163, r1086;
}
{
add.f16x2 %17, r1160, r1166;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<868, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<92>;
.reg .b32 r<1183>;
.reg .b64 rd<4>;
mov.u32 r1172, %tid.y;
mov.u32 r1173, %18;
mad.lo.s32 r1174, r1172, 108, r1173;
mov.u32 r1175, %tid.x;
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
neg.f16x2 r3, r2;
}
{
add.f16x2 r5, %25, %31;
}
{
add.f16x2 r8, %19, r5;
}
{
add.f16x2 r11, %26, %32;
}
{
add.f16x2 r14, %20, r11;
}
{
add.f16x2 r17, %25, %31;
}
{
mul.f16x2 r20, r17, r1;
}
{
add.f16x2 r23, %19, r20;
}
{
sub.f16x2 r26, %26, %32;
}
{
mul.f16x2 r29, r26, r3;
}
{
add.f16x2 r32, r23, r29;
}
{
add.f16x2 r35, %25, %31;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %19, r38;
}
{
sub.f16x2 r44, %26, %32;
}
{
mul.f16x2 r47, r44, r3;
}
{
sub.f16x2 r50, r41, r47;
}
{
add.f16x2 r53, %26, %32;
}
{
mul.f16x2 r56, r53, r1;
}
{
add.f16x2 r59, %20, r56;
}
{
sub.f16x2 r62, %25, %31;
}
{
mul.f16x2 r65, r62, r3;
}
{
sub.f16x2 r68, r59, r65;
}
{
add.f16x2 r71, %26, %32;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %20, r74;
}
{
sub.f16x2 r80, %25, %31;
}
{
mul.f16x2 r83, r80, r3;
}
{
add.f16x2 r86, r77, r83;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r90, {low, high};
}
{
neg.f16x2 r91, r90;
}
{
add.f16x2 r93, %27, %33;
}
{
add.f16x2 r96, %21, r93;
}
{
add.f16x2 r99, %28, %34;
}
{
add.f16x2 r102, %22, r99;
}
{
add.f16x2 r105, %27, %33;
}
{
mul.f16x2 r108, r105, r89;
}
{
add.f16x2 r111, %21, r108;
}
{
sub.f16x2 r114, %28, %34;
}
{
mul.f16x2 r117, r114, r91;
}
{
add.f16x2 r120, r111, r117;
}
{
add.f16x2 r123, %27, %33;
}
{
mul.f16x2 r126, r123, r89;
}
{
add.f16x2 r129, %21, r126;
}
{
sub.f16x2 r132, %28, %34;
}
{
mul.f16x2 r135, r132, r91;
}
{
sub.f16x2 r138, r129, r135;
}
{
add.f16x2 r141, %28, %34;
}
{
mul.f16x2 r144, r141, r89;
}
{
add.f16x2 r147, %22, r144;
}
{
sub.f16x2 r150, %27, %33;
}
{
mul.f16x2 r153, r150, r91;
}
{
sub.f16x2 r156, r147, r153;
}
{
add.f16x2 r159, %28, %34;
}
{
mul.f16x2 r162, r159, r89;
}
{
add.f16x2 r165, %22, r162;
}
{
sub.f16x2 r168, %27, %33;
}
{
mul.f16x2 r171, r168, r91;
}
{
add.f16x2 r174, r165, r171;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r177, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r178, {low, high};
}
{
neg.f16x2 r179, r178;
}
{
add.f16x2 r181, %29, %35;
}
{
add.f16x2 r184, %23, r181;
}
{
add.f16x2 r187, %30, %36;
}
{
add.f16x2 r190, %24, r187;
}
{
add.f16x2 r193, %29, %35;
}
{
mul.f16x2 r196, r193, r177;
}
{
add.f16x2 r199, %23, r196;
}
{
sub.f16x2 r202, %30, %36;
}
{
mul.f16x2 r205, r202, r179;
}
{
add.f16x2 r208, r199, r205;
}
{
add.f16x2 r211, %29, %35;
}
{
mul.f16x2 r214, r211, r177;
}
{
add.f16x2 r217, %23, r214;
}
{
sub.f16x2 r220, %30, %36;
}
{
mul.f16x2 r223, r220, r179;
}
{
sub.f16x2 r226, r217, r223;
}
{
add.f16x2 r229, %30, %36;
}
{
mul.f16x2 r232, r229, r177;
}
{
add.f16x2 r235, %24, r232;
}
{
sub.f16x2 r238, %29, %35;
}
{
mul.f16x2 r241, r238, r179;
}
{
sub.f16x2 r244, r235, r241;
}
{
add.f16x2 r247, %30, %36;
}
{
mul.f16x2 r250, r247, r177;
}
{
add.f16x2 r253, %24, r250;
}
{
sub.f16x2 r256, %29, %35;
}
{
mul.f16x2 r259, r256, r179;
}
{
add.f16x2 r262, r253, r259;
}
mov.f32 f14, 0f3F441B7D;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f14;
cvt.rn.f16.f32 high, f14;
mov.b32 r265, {low, high};
}
mov.f32 f16, 0fBF248DBB;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f16;
cvt.rn.f16.f32 high, f16;
mov.b32 r266, {low, high};
}
mov.f32 f18, 0f3E31D0D4;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f18;
cvt.rn.f16.f32 high, f18;
mov.b32 r267, {low, high};
}
mov.f32 f20, 0fBF7C1C5C;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f20;
cvt.rn.f16.f32 high, f20;
mov.b32 r268, {low, high};
}
mov.f32 f26, 0fBF708FB2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f26;
cvt.rn.f16.f32 high, f26;
mov.b32 r271, {low, high};
}
mov.f32 f28, 0fBEAF1D44;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f28;
cvt.rn.f16.f32 high, f28;
mov.b32 r272, {low, high};
}
{
mul.f16x2 r281, r120, r265;
}
{
mul.f16x2 r284, r156, r266;
}
{
sub.f16x2 r287, r281, r284;
}
{
mul.f16x2 r290, r120, r266;
}
{
fma.rn.f16x2 r293, r156, r265, r290;
}
{
mul.f16x2 r297, r208, r267;
}
{
mul.f16x2 r300, r244, r268;
}
{
sub.f16x2 r303, r297, r300;
}
{
mul.f16x2 r306, r208, r268;
}
{
fma.rn.f16x2 r309, r244, r267, r306;
}
{
mul.f16x2 r313, r138, r267;
}
{
mul.f16x2 r316, r174, r268;
}
{
sub.f16x2 r319, r313, r316;
}
{
mul.f16x2 r322, r138, r268;
}
{
fma.rn.f16x2 r325, r174, r267, r322;
}
{
mul.f16x2 r329, r226, r271;
}
{
mul.f16x2 r332, r262, r272;
}
{
sub.f16x2 r335, r329, r332;
}
{
mul.f16x2 r338, r226, r272;
}
{
fma.rn.f16x2 r341, r262, r271, r338;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r345, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r346, {low, high};
}
{
neg.f16x2 r347, r346;
}
{
add.f16x2 r349, r96, r184;
}
{
add.f16x2 r352, r8, r349;
}
{
add.f16x2 r355, r102, r190;
}
{
add.f16x2 r358, r14, r355;
}
{
add.f16x2 r361, r96, r184;
}
{
mul.f16x2 r364, r361, r345;
}
{
add.f16x2 r367, r8, r364;
}
{
sub.f16x2 r370, r102, r190;
}
{
mul.f16x2 r373, r370, r347;
}
{
add.f16x2 r376, r367, r373;
}
{
add.f16x2 r379, r96, r184;
}
{
mul.f16x2 r382, r379, r345;
}
{
add.f16x2 r385, r8, r382;
}
{
sub.f16x2 r388, r102, r190;
}
{
mul.f16x2 r391, r388, r347;
}
{
sub.f16x2 r394, r385, r391;
}
{
add.f16x2 r397, r102, r190;
}
{
mul.f16x2 r400, r397, r345;
}
{
add.f16x2 r403, r14, r400;
}
{
sub.f16x2 r406, r96, r184;
}
{
mul.f16x2 r409, r406, r347;
}
{
sub.f16x2 r412, r403, r409;
}
{
add.f16x2 r415, r102, r190;
}
{
mul.f16x2 r418, r415, r345;
}
{
add.f16x2 r421, r14, r418;
}
{
sub.f16x2 r424, r96, r184;
}
{
mul.f16x2 r427, r424, r347;
}
{
add.f16x2 r430, r421, r427;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r433, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r434, {low, high};
}
{
neg.f16x2 r435, r434;
}
{
add.f16x2 r437, r287, r303;
}
{
add.f16x2 r440, r32, r437;
}
{
add.f16x2 r443, r293, r309;
}
{
add.f16x2 r446, r68, r443;
}
{
add.f16x2 r449, r287, r303;
}
{
mul.f16x2 r452, r449, r433;
}
{
add.f16x2 r455, r32, r452;
}
{
sub.f16x2 r458, r293, r309;
}
{
mul.f16x2 r461, r458, r435;
}
{
add.f16x2 r464, r455, r461;
}
{
add.f16x2 r467, r287, r303;
}
{
mul.f16x2 r470, r467, r433;
}
{
add.f16x2 r473, r32, r470;
}
{
sub.f16x2 r476, r293, r309;
}
{
mul.f16x2 r479, r476, r435;
}
{
sub.f16x2 r482, r473, r479;
}
{
add.f16x2 r485, r293, r309;
}
{
mul.f16x2 r488, r485, r433;
}
{
add.f16x2 r491, r68, r488;
}
{
sub.f16x2 r494, r287, r303;
}
{
mul.f16x2 r497, r494, r435;
}
{
sub.f16x2 r500, r491, r497;
}
{
add.f16x2 r503, r293, r309;
}
{
mul.f16x2 r506, r503, r433;
}
{
add.f16x2 r509, r68, r506;
}
{
sub.f16x2 r512, r287, r303;
}
{
mul.f16x2 r515, r512, r435;
}
{
add.f16x2 r518, r509, r515;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r521, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r522, {low, high};
}
{
neg.f16x2 r523, r522;
}
{
add.f16x2 r525, r319, r335;
}
{
add.f16x2 r528, r50, r525;
}
{
add.f16x2 r531, r325, r341;
}
{
add.f16x2 r534, r86, r531;
}
{
add.f16x2 r537, r319, r335;
}
{
mul.f16x2 r540, r537, r521;
}
{
add.f16x2 r543, r50, r540;
}
{
sub.f16x2 r546, r325, r341;
}
{
mul.f16x2 r549, r546, r523;
}
{
add.f16x2 r552, r543, r549;
}
{
add.f16x2 r555, r319, r335;
}
{
mul.f16x2 r558, r555, r521;
}
{
add.f16x2 r561, r50, r558;
}
{
sub.f16x2 r564, r325, r341;
}
{
mul.f16x2 r567, r564, r523;
}
{
sub.f16x2 r570, r561, r567;
}
{
add.f16x2 r573, r325, r341;
}
{
mul.f16x2 r576, r573, r521;
}
{
add.f16x2 r579, r86, r576;
}
{
sub.f16x2 r582, r319, r335;
}
{
mul.f16x2 r585, r582, r523;
}
{
sub.f16x2 r588, r579, r585;
}
{
add.f16x2 r591, r325, r341;
}
{
mul.f16x2 r594, r591, r521;
}
{
add.f16x2 r597, r86, r594;
}
{
sub.f16x2 r600, r319, r335;
}
{
mul.f16x2 r603, r600, r523;
}
{
add.f16x2 r606, r597, r603;
}
mul.wide.u32 rd2, r1175, -1431655765;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r1176, rd3;
mul.lo.s32 r1177, r1176, 3;
sub.s32 r1178, r1175, r1177;
mad.lo.s32 r1179, r1176, 108, r1174;
cvt.rn.f32.u32 f89, r1178;
mul.f32 f90, f89, 0f3E6E4BAE;
cos.approx.f32 f57, f90;
sin.approx.f32 f91, f90;
neg.f32 f58, f91;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f57;
cvt.rn.f16.f32 high, f58;
mov.b32 r609, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r612, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r614, {high, high};
}
{
mul.f16x2 r616, r446, r614;
}
{
neg.f16x2 r619, r616;
}
{
fma.rn.f16x2 r621, r440, r612, r619;
}
{
mul.f16x2 r625, r440, r614;
}
{
fma.rn.f16x2 r628, r446, r612, r625;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r632, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r634, {high, high};
}
mov.f32 f73, 0fBF800000;
mov.f32 f74, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r636, {low, high};
}
{
mul.f16x2 r637, r634, r636;
}
{
mul.f16x2 r640, r609, r632;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r643, {high, low};
}
{
fma.rn.f16x2 r645, r637, r643, r640;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r645;
mov.b32 r649, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r645;
mov.b32 r651, {high, high};
}
{
mul.f16x2 r653, r534, r651;
}
{
neg.f16x2 r656, r653;
}
{
fma.rn.f16x2 r658, r528, r649, r656;
}
{
mul.f16x2 r662, r528, r651;
}
{
fma.rn.f16x2 r665, r534, r649, r662;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r669, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r671, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r673, {low, high};
}
{
mul.f16x2 r674, r671, r673;
}
{
mul.f16x2 r677, r645, r669;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r645;
mov.b32 r680, {high, low};
}
{
fma.rn.f16x2 r682, r674, r680, r677;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r682;
mov.b32 r686, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r682;
mov.b32 r688, {high, high};
}
{
mul.f16x2 r690, r412, r688;
}
{
neg.f16x2 r693, r690;
}
{
fma.rn.f16x2 r695, r376, r686, r693;
}
{
mul.f16x2 r699, r376, r688;
}
{
fma.rn.f16x2 r702, r412, r686, r699;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r706, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r708, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r710, {low, high};
}
{
mul.f16x2 r711, r708, r710;
}
{
mul.f16x2 r714, r682, r706;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r682;
mov.b32 r717, {high, low};
}
{
fma.rn.f16x2 r719, r711, r717, r714;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r719;
mov.b32 r723, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r719;
mov.b32 r725, {high, high};
}
{
mul.f16x2 r727, r500, r725;
}
{
neg.f16x2 r730, r727;
}
{
fma.rn.f16x2 r732, r464, r723, r730;
}
{
mul.f16x2 r736, r464, r725;
}
{
fma.rn.f16x2 r739, r500, r723, r736;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r743, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r745, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r747, {low, high};
}
{
mul.f16x2 r748, r745, r747;
}
{
mul.f16x2 r751, r719, r743;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r719;
mov.b32 r754, {high, low};
}
{
fma.rn.f16x2 r756, r748, r754, r751;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r756;
mov.b32 r760, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r756;
mov.b32 r762, {high, high};
}
{
mul.f16x2 r764, r588, r762;
}
{
neg.f16x2 r767, r764;
}
{
fma.rn.f16x2 r769, r552, r760, r767;
}
{
mul.f16x2 r773, r552, r762;
}
{
fma.rn.f16x2 r776, r588, r760, r773;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r780, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r782, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r784, {low, high};
}
{
mul.f16x2 r785, r782, r784;
}
{
mul.f16x2 r788, r756, r780;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r756;
mov.b32 r791, {high, low};
}
{
fma.rn.f16x2 r793, r785, r791, r788;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r793;
mov.b32 r797, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r793;
mov.b32 r799, {high, high};
}
{
mul.f16x2 r801, r430, r799;
}
{
neg.f16x2 r804, r801;
}
{
fma.rn.f16x2 r806, r394, r797, r804;
}
{
mul.f16x2 r810, r394, r799;
}
{
fma.rn.f16x2 r813, r430, r797, r810;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r817, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r819, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r821, {low, high};
}
{
mul.f16x2 r822, r819, r821;
}
{
mul.f16x2 r825, r793, r817;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r793;
mov.b32 r828, {high, low};
}
{
fma.rn.f16x2 r830, r822, r828, r825;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r830;
mov.b32 r834, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r830;
mov.b32 r836, {high, high};
}
{
mul.f16x2 r838, r518, r836;
}
{
neg.f16x2 r841, r838;
}
{
fma.rn.f16x2 r843, r482, r834, r841;
}
{
mul.f16x2 r847, r482, r836;
}
{
fma.rn.f16x2 r850, r518, r834, r847;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r854, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r609;
mov.b32 r856, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f73;
cvt.rn.f16.f32 high, f74;
mov.b32 r858, {low, high};
}
{
mul.f16x2 r859, r856, r858;
}
{
mul.f16x2 r862, r830, r854;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r830;
mov.b32 r865, {high, low};
}
{
fma.rn.f16x2 r867, r859, r865, r862;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r867;
mov.b32 r871, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r867;
mov.b32 r873, {high, high};
}
{
mul.f16x2 r875, r606, r873;
}
{
neg.f16x2 r878, r875;
}
{
fma.rn.f16x2 r880, r570, r871, r878;
}
{
mul.f16x2 r884, r570, r873;
}
{
fma.rn.f16x2 r887, r606, r871, r884;
}
barrier.sync 0;
mad.lo.s32 r1180, r1178, 36, r1179;
st.shared.u32 [r1180], r352;
st.shared.u32 [r1180+4], r621;
st.shared.u32 [r1180+8], r658;
st.shared.u32 [r1180+12], r695;
st.shared.u32 [r1180+16], r732;
st.shared.u32 [r1180+20], r769;
st.shared.u32 [r1180+24], r806;
st.shared.u32 [r1180+28], r843;
st.shared.u32 [r1180+32], r880;
barrier.sync 0;
shl.b32 r1181, r1178, 5;
sub.s32 r1182, r1180, r1181;
ld.shared.u32 r916, [r1182];
ld.shared.u32 r1004, [r1182+12];
ld.shared.u32 r1092, [r1182+24];
ld.shared.u32 r913, [r1182+36];
ld.shared.u32 r1001, [r1182+48];
ld.shared.u32 r1089, [r1182+60];
ld.shared.u32 r914, [r1182+72];
ld.shared.u32 r1002, [r1182+84];
ld.shared.u32 r1090, [r1182+96];
barrier.sync 0;
st.shared.u32 [r1180], r358;
st.shared.u32 [r1180+4], r628;
st.shared.u32 [r1180+8], r665;
st.shared.u32 [r1180+12], r702;
st.shared.u32 [r1180+16], r739;
st.shared.u32 [r1180+20], r776;
st.shared.u32 [r1180+24], r813;
st.shared.u32 [r1180+28], r850;
st.shared.u32 [r1180+32], r887;
barrier.sync 0;
ld.shared.u32 r922, [r1182];
ld.shared.u32 r1010, [r1182+12];
ld.shared.u32 r1098, [r1182+24];
ld.shared.u32 r919, [r1182+36];
ld.shared.u32 r1007, [r1182+48];
ld.shared.u32 r1095, [r1182+60];
ld.shared.u32 r920, [r1182+72];
ld.shared.u32 r1008, [r1182+84];
ld.shared.u32 r1096, [r1182+96];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r908, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r909, {low, high};
}
{
neg.f16x2 r910, r909;
}
{
add.f16x2 r912, r913, r914;
}
{
add.f16x2 %0, r916, r912;
}
{
add.f16x2 r918, r919, r920;
}
{
add.f16x2 %1, r922, r918;
}
{
add.f16x2 r924, r913, r914;
}
{
mul.f16x2 r927, r924, r908;
}
{
add.f16x2 r930, r916, r927;
}
{
sub.f16x2 r933, r919, r920;
}
{
mul.f16x2 r936, r933, r910;
}
{
add.f16x2 %6, r930, r936;
}
{
add.f16x2 r942, r913, r914;
}
{
mul.f16x2 r945, r942, r908;
}
{
add.f16x2 r948, r916, r945;
}
{
sub.f16x2 r951, r919, r920;
}
{
mul.f16x2 r954, r951, r910;
}
{
sub.f16x2 %12, r948, r954;
}
{
add.f16x2 r960, r919, r920;
}
{
mul.f16x2 r963, r960, r908;
}
{
add.f16x2 r966, r922, r963;
}
{
sub.f16x2 r969, r913, r914;
}
{
mul.f16x2 r972, r969, r910;
}
{
sub.f16x2 %7, r966, r972;
}
{
add.f16x2 r978, r919, r920;
}
{
mul.f16x2 r981, r978, r908;
}
{
add.f16x2 r984, r922, r981;
}
{
sub.f16x2 r987, r913, r914;
}
{
mul.f16x2 r990, r987, r910;
}
{
add.f16x2 %13, r984, r990;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r996, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r997, {low, high};
}
{
neg.f16x2 r998, r997;
}
{
add.f16x2 r1000, r1001, r1002;
}
{
add.f16x2 %2, r1004, r1000;
}
{
add.f16x2 r1006, r1007, r1008;
}
{
add.f16x2 %3, r1010, r1006;
}
{
add.f16x2 r1012, r1001, r1002;
}
{
mul.f16x2 r1015, r1012, r996;
}
{
add.f16x2 r1018, r1004, r1015;
}
{
sub.f16x2 r1021, r1007, r1008;
}
{
mul.f16x2 r1024, r1021, r998;
}
{
add.f16x2 %8, r1018, r1024;
}
{
add.f16x2 r1030, r1001, r1002;
}
{
mul.f16x2 r1033, r1030, r996;
}
{
add.f16x2 r1036, r1004, r1033;
}
{
sub.f16x2 r1039, r1007, r1008;
}
{
mul.f16x2 r1042, r1039, r998;
}
{
sub.f16x2 %14, r1036, r1042;
}
{
add.f16x2 r1048, r1007, r1008;
}
{
mul.f16x2 r1051, r1048, r996;
}
{
add.f16x2 r1054, r1010, r1051;
}
{
sub.f16x2 r1057, r1001, r1002;
}
{
mul.f16x2 r1060, r1057, r998;
}
{
sub.f16x2 %9, r1054, r1060;
}
{
add.f16x2 r1066, r1007, r1008;
}
{
mul.f16x2 r1069, r1066, r996;
}
{
add.f16x2 r1072, r1010, r1069;
}
{
sub.f16x2 r1075, r1001, r1002;
}
{
mul.f16x2 r1078, r1075, r998;
}
{
add.f16x2 %15, r1072, r1078;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f86;
cvt.rn.f16.f32 high, f86;
mov.b32 r1084, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f88;
cvt.rn.f16.f32 high, f88;
mov.b32 r1085, {low, high};
}
{
neg.f16x2 r1086, r1085;
}
{
add.f16x2 r1088, r1089, r1090;
}
{
add.f16x2 %4, r1092, r1088;
}
{
add.f16x2 r1094, r1095, r1096;
}
{
add.f16x2 %5, r1098, r1094;
}
{
add.f16x2 r1100, r1089, r1090;
}
{
mul.f16x2 r1103, r1100, r1084;
}
{
add.f16x2 r1106, r1092, r1103;
}
{
sub.f16x2 r1109, r1095, r1096;
}
{
mul.f16x2 r1112, r1109, r1086;
}
{
add.f16x2 %10, r1106, r1112;
}
{
add.f16x2 r1118, r1089, r1090;
}
{
mul.f16x2 r1121, r1118, r1084;
}
{
add.f16x2 r1124, r1092, r1121;
}
{
sub.f16x2 r1127, r1095, r1096;
}
{
mul.f16x2 r1130, r1127, r1086;
}
{
sub.f16x2 %16, r1124, r1130;
}
{
add.f16x2 r1136, r1095, r1096;
}
{
mul.f16x2 r1139, r1136, r1084;
}
{
add.f16x2 r1142, r1098, r1139;
}
{
sub.f16x2 r1145, r1089, r1090;
}
{
mul.f16x2 r1148, r1145, r1086;
}
{
sub.f16x2 %11, r1142, r1148;
}
{
add.f16x2 r1154, r1095, r1096;
}
{
mul.f16x2 r1157, r1154, r1084;
}
{
add.f16x2 r1160, r1098, r1157;
}
{
sub.f16x2 r1163, r1089, r1090;
}
{
mul.f16x2 r1166, r1163, r1086;
}
{
add.f16x2 %17, r1160, r1166;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<870, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<35>;
.reg .b32 r<436>;
.reg .b64 rd<6>;
mov.u32 r419, %tid.y;
mov.u32 r420, %6;
mad.lo.s32 r421, r419, 216, r420;
mov.u32 r422, %tid.x;
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
neg.f16x2 r3, r2;
}
{
add.f16x2 r5, %9, %11;
}
{
add.f16x2 r8, %7, r5;
}
{
add.f16x2 r11, %10, %12;
}
{
add.f16x2 r14, %8, r11;
}
{
add.f16x2 r17, %9, %11;
}
{
mul.f16x2 r20, r17, r1;
}
{
add.f16x2 r23, %7, r20;
}
{
sub.f16x2 r26, %10, %12;
}
{
mul.f16x2 r29, r26, r3;
}
{
add.f16x2 r32, r23, r29;
}
{
add.f16x2 r35, %9, %11;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %7, r38;
}
{
sub.f16x2 r44, %10, %12;
}
{
mul.f16x2 r47, r44, r3;
}
{
sub.f16x2 r50, r41, r47;
}
{
add.f16x2 r53, %10, %12;
}
{
mul.f16x2 r56, r53, r1;
}
{
add.f16x2 r59, %8, r56;
}
{
sub.f16x2 r62, %9, %11;
}
{
mul.f16x2 r65, r62, r3;
}
{
sub.f16x2 r68, r59, r65;
}
{
add.f16x2 r71, %10, %12;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %8, r74;
}
{
sub.f16x2 r80, %9, %11;
}
{
mul.f16x2 r83, r80, r3;
}
{
add.f16x2 r86, r77, r83;
}
mul.wide.u32 rd2, r422, 954437177;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r423, rd3;
mul.lo.s32 r424, r423, 9;
sub.s32 r425, r422, r424;
mad.lo.s32 r426, r423, 216, r421;
cvt.rn.f32.u32 f29, r425;
mul.f32 f30, f29, 0f3E6E4BAE;
cos.approx.f32 f5, f30;
sin.approx.f32 f31, f30;
neg.f32 f6, f31;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f5;
cvt.rn.f16.f32 high, f6;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r92, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r94, {high, high};
}
{
mul.f16x2 r96, r68, r94;
}
{
neg.f16x2 r99, r96;
}
{
fma.rn.f16x2 r101, r32, r92, r99;
}
{
mul.f16x2 r105, r32, r94;
}
{
fma.rn.f16x2 r108, r68, r92, r105;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r112, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r114, {high, high};
}
mov.f32 f21, 0fBF800000;
mov.f32 f22, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f21;
cvt.rn.f16.f32 high, f22;
mov.b32 r116, {low, high};
}
{
mul.f16x2 r117, r114, r116;
}
{
mul.f16x2 r120, r89, r112;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r123, {high, low};
}
{
fma.rn.f16x2 r125, r117, r123, r120;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r125;
mov.b32 r129, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r125;
mov.b32 r131, {high, high};
}
{
mul.f16x2 r133, r86, r131;
}
{
neg.f16x2 r136, r133;
}
{
fma.rn.f16x2 r138, r50, r129, r136;
}
{
mul.f16x2 r142, r50, r131;
}
{
fma.rn.f16x2 r145, r86, r129, r142;
}
barrier.sync 0;
mad.lo.s32 r427, r425, 24, r426;
st.shared.v2.f32 [r427], {r8, r14};
st.shared.v2.f32 [r427+8], {r101, r108};
st.shared.v2.f32 [r427+16], {r138, r145};
barrier.sync 0;
shl.b32 r428, r425, 4;
sub.s32 r429, r427, r428;
ld.shared.u32 r174, [r429];
ld.shared.u32 r180, [r429+4];
ld.shared.u32 r171, [r429+72];
ld.shared.u32 r177, [r429+76];
ld.shared.u32 r172, [r429+144];
ld.shared.u32 r178, [r429+148];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f26;
cvt.rn.f16.f32 high, f26;
mov.b32 r166, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f28;
cvt.rn.f16.f32 high, f28;
mov.b32 r167, {low, high};
}
{
neg.f16x2 r168, r167;
}
{
add.f16x2 r170, r171, r172;
}
{
add.f16x2 r173, r174, r170;
}
{
add.f16x2 r176, r177, r178;
}
{
add.f16x2 r179, r180, r176;
}
{
add.f16x2 r182, r171, r172;
}
{
mul.f16x2 r185, r182, r166;
}
{
add.f16x2 r188, r174, r185;
}
{
sub.f16x2 r191, r177, r178;
}
{
mul.f16x2 r194, r191, r168;
}
{
add.f16x2 r197, r188, r194;
}
{
add.f16x2 r200, r171, r172;
}
{
mul.f16x2 r203, r200, r166;
}
{
add.f16x2 r206, r174, r203;
}
{
sub.f16x2 r209, r177, r178;
}
{
mul.f16x2 r212, r209, r168;
}
{
sub.f16x2 r215, r206, r212;
}
{
add.f16x2 r218, r177, r178;
}
{
mul.f16x2 r221, r218, r166;
}
{
add.f16x2 r224, r180, r221;
}
{
sub.f16x2 r227, r171, r172;
}
{
mul.f16x2 r230, r227, r168;
}
{
sub.f16x2 r233, r224, r230;
}
{
add.f16x2 r236, r177, r178;
}
{
mul.f16x2 r239, r236, r166;
}
{
add.f16x2 r242, r180, r239;
}
{
sub.f16x2 r245, r171, r172;
}
{
mul.f16x2 r248, r245, r168;
}
{
add.f16x2 r251, r242, r248;
}
mul.wide.u32 rd4, r425, -1431655765;
shr.u64 rd5, rd4, 33;
cvt.u32.u64 r430, rd5;
mul.lo.s32 r431, r430, 3;
sub.s32 r432, r425, r431;
shl.b32 r433, r432, 3;
add.s32 r434, r426, r433;
cvt.rn.f32.u32 f32, r430;
mul.f32 f33, f32, 0f3F32B8C2;
cos.approx.f32 f17, f33;
sin.approx.f32 f34, f33;
neg.f32 f18, f34;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f17;
cvt.rn.f16.f32 high, f18;
mov.b32 r254, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r254;
mov.b32 r257, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r254;
mov.b32 r259, {high, high};
}
{
mul.f16x2 r261, r233, r259;
}
{
neg.f16x2 r264, r261;
}
{
fma.rn.f16x2 r266, r197, r257, r264;
}
{
mul.f16x2 r270, r197, r259;
}
{
fma.rn.f16x2 r273, r233, r257, r270;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r254;
mov.b32 r277, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r254;
mov.b32 r279, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f21;
cvt.rn.f16.f32 high, f22;
mov.b32 r281, {low, high};
}
{
mul.f16x2 r282, r279, r281;
}
{
mul.f16x2 r285, r254, r277;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r254;
mov.b32 r288, {high, low};
}
{
fma.rn.f16x2 r290, r282, r288, r285;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r290;
mov.b32 r294, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r290;
mov.b32 r296, {high, high};
}
{
mul.f16x2 r298, r251, r296;
}
{
neg.f16x2 r301, r298;
}
{
fma.rn.f16x2 r303, r215, r294, r301;
}
{
mul.f16x2 r307, r215, r296;
}
{
fma.rn.f16x2 r310, r251, r294, r307;
}
barrier.sync 0;
mad.lo.s32 r435, r430, 72, r434;
st.shared.u32 [r435], r173;
st.shared.u32 [r435+4], r179;
st.shared.u32 [r435+24], r266;
st.shared.u32 [r435+28], r273;
st.shared.u32 [r435+48], r303;
st.shared.u32 [r435+52], r310;
barrier.sync 0;
ld.shared.u32 r339, [r429];
ld.shared.u32 r345, [r429+4];
ld.shared.u32 r336, [r429+72];
ld.shared.u32 r342, [r429+76];
ld.shared.u32 r337, [r429+144];
ld.shared.u32 r343, [r429+148];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f26;
cvt.rn.f16.f32 high, f26;
mov.b32 r331, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f28;
cvt.rn.f16.f32 high, f28;
mov.b32 r332, {low, high};
}
{
neg.f16x2 r333, r332;
}
{
add.f16x2 r335, r336, r337;
}
{
add.f16x2 %0, r339, r335;
}
{
add.f16x2 r341, r342, r343;
}
{
add.f16x2 %1, r345, r341;
}
{
add.f16x2 r347, r336, r337;
}
{
mul.f16x2 r350, r347, r331;
}
{
add.f16x2 r353, r339, r350;
}
{
sub.f16x2 r356, r342, r343;
}
{
mul.f16x2 r359, r356, r333;
}
{
add.f16x2 %2, r353, r359;
}
{
add.f16x2 r365, r336, r337;
}
{
mul.f16x2 r368, r365, r331;
}
{
add.f16x2 r371, r339, r368;
}
{
sub.f16x2 r374, r342, r343;
}
{
mul.f16x2 r377, r374, r333;
}
{
sub.f16x2 %4, r371, r377;
}
{
add.f16x2 r383, r342, r343;
}
{
mul.f16x2 r386, r383, r331;
}
{
add.f16x2 r389, r345, r386;
}
{
sub.f16x2 r392, r336, r337;
}
{
mul.f16x2 r395, r392, r333;
}
{
sub.f16x2 %3, r389, r395;
}
{
add.f16x2 r401, r342, r343;
}
{
mul.f16x2 r404, r401, r331;
}
{
add.f16x2 r407, r345, r404;
}
{
sub.f16x2 r410, r336, r337;
}
{
mul.f16x2 r413, r410, r333;
}
{
add.f16x2 %5, r407, r413;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<871, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<35>;
.reg .b32 r<436>;
.reg .b64 rd<6>;
mov.u32 r419, %tid.y;
mov.u32 r420, %6;
mad.lo.s32 r421, r419, 108, r420;
mov.u32 r422, %tid.x;
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
neg.f16x2 r3, r2;
}
{
add.f16x2 r5, %9, %11;
}
{
add.f16x2 r8, %7, r5;
}
{
add.f16x2 r11, %10, %12;
}
{
add.f16x2 r14, %8, r11;
}
{
add.f16x2 r17, %9, %11;
}
{
mul.f16x2 r20, r17, r1;
}
{
add.f16x2 r23, %7, r20;
}
{
sub.f16x2 r26, %10, %12;
}
{
mul.f16x2 r29, r26, r3;
}
{
add.f16x2 r32, r23, r29;
}
{
add.f16x2 r35, %9, %11;
}
{
mul.f16x2 r38, r35, r1;
}
{
add.f16x2 r41, %7, r38;
}
{
sub.f16x2 r44, %10, %12;
}
{
mul.f16x2 r47, r44, r3;
}
{
sub.f16x2 r50, r41, r47;
}
{
add.f16x2 r53, %10, %12;
}
{
mul.f16x2 r56, r53, r1;
}
{
add.f16x2 r59, %8, r56;
}
{
sub.f16x2 r62, %9, %11;
}
{
mul.f16x2 r65, r62, r3;
}
{
sub.f16x2 r68, r59, r65;
}
{
add.f16x2 r71, %10, %12;
}
{
mul.f16x2 r74, r71, r1;
}
{
add.f16x2 r77, %8, r74;
}
{
sub.f16x2 r80, %9, %11;
}
{
mul.f16x2 r83, r80, r3;
}
{
add.f16x2 r86, r77, r83;
}
mul.wide.u32 rd2, r422, 954437177;
shr.u64 rd3, rd2, 33;
cvt.u32.u64 r423, rd3;
mul.lo.s32 r424, r423, 9;
sub.s32 r425, r422, r424;
mad.lo.s32 r426, r423, 108, r421;
cvt.rn.f32.u32 f29, r425;
mul.f32 f30, f29, 0f3E6E4BAE;
cos.approx.f32 f5, f30;
sin.approx.f32 f31, f30;
neg.f32 f6, f31;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f5;
cvt.rn.f16.f32 high, f6;
mov.b32 r89, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r92, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r94, {high, high};
}
{
mul.f16x2 r96, r68, r94;
}
{
neg.f16x2 r99, r96;
}
{
fma.rn.f16x2 r101, r32, r92, r99;
}
{
mul.f16x2 r105, r32, r94;
}
{
fma.rn.f16x2 r108, r68, r92, r105;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r112, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r114, {high, high};
}
mov.f32 f21, 0fBF800000;
mov.f32 f22, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f21;
cvt.rn.f16.f32 high, f22;
mov.b32 r116, {low, high};
}
{
mul.f16x2 r117, r114, r116;
}
{
mul.f16x2 r120, r89, r112;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r89;
mov.b32 r123, {high, low};
}
{
fma.rn.f16x2 r125, r117, r123, r120;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r125;
mov.b32 r129, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r125;
mov.b32 r131, {high, high};
}
{
mul.f16x2 r133, r86, r131;
}
{
neg.f16x2 r136, r133;
}
{
fma.rn.f16x2 r138, r50, r129, r136;
}
{
mul.f16x2 r142, r50, r131;
}
{
fma.rn.f16x2 r145, r86, r129, r142;
}
barrier.sync 0;
mad.lo.s32 r427, r425, 12, r426;
st.shared.u32 [r427], r8;
st.shared.u32 [r427+4], r101;
st.shared.u32 [r427+8], r138;
barrier.sync 0;
shl.b32 r428, r425, 3;
sub.s32 r429, r427, r428;
ld.shared.u32 r174, [r429];
ld.shared.u32 r171, [r429+36];
ld.shared.u32 r172, [r429+72];
barrier.sync 0;
st.shared.u32 [r427], r14;
st.shared.u32 [r427+4], r108;
st.shared.u32 [r427+8], r145;
barrier.sync 0;
ld.shared.u32 r180, [r429];
ld.shared.u32 r177, [r429+36];
ld.shared.u32 r178, [r429+72];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f26;
cvt.rn.f16.f32 high, f26;
mov.b32 r166, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f28;
cvt.rn.f16.f32 high, f28;
mov.b32 r167, {low, high};
}
{
neg.f16x2 r168, r167;
}
{
add.f16x2 r170, r171, r172;
}
{
add.f16x2 r173, r174, r170;
}
{
add.f16x2 r176, r177, r178;
}
{
add.f16x2 r179, r180, r176;
}
{
add.f16x2 r182, r171, r172;
}
{
mul.f16x2 r185, r182, r166;
}
{
add.f16x2 r188, r174, r185;
}
{
sub.f16x2 r191, r177, r178;
}
{
mul.f16x2 r194, r191, r168;
}
{
add.f16x2 r197, r188, r194;
}
{
add.f16x2 r200, r171, r172;
}
{
mul.f16x2 r203, r200, r166;
}
{
add.f16x2 r206, r174, r203;
}
{
sub.f16x2 r209, r177, r178;
}
{
mul.f16x2 r212, r209, r168;
}
{
sub.f16x2 r215, r206, r212;
}
{
add.f16x2 r218, r177, r178;
}
{
mul.f16x2 r221, r218, r166;
}
{
add.f16x2 r224, r180, r221;
}
{
sub.f16x2 r227, r171, r172;
}
{
mul.f16x2 r230, r227, r168;
}
{
sub.f16x2 r233, r224, r230;
}
{
add.f16x2 r236, r177, r178;
}
{
mul.f16x2 r239, r236, r166;
}
{
add.f16x2 r242, r180, r239;
}
{
sub.f16x2 r245, r171, r172;
}
{
mul.f16x2 r248, r245, r168;
}
{
add.f16x2 r251, r242, r248;
}
mul.wide.u32 rd4, r425, -1431655765;
shr.u64 rd5, rd4, 33;
cvt.u32.u64 r430, rd5;
mul.lo.s32 r431, r430, 3;
sub.s32 r432, r425, r431;
shl.b32 r433, r432, 2;
add.s32 r434, r426, r433;
cvt.rn.f32.u32 f32, r430;
mul.f32 f33, f32, 0f3F32B8C2;
cos.approx.f32 f17, f33;
sin.approx.f32 f34, f33;
neg.f32 f18, f34;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f17;
cvt.rn.f16.f32 high, f18;
mov.b32 r254, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r254;
mov.b32 r257, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r254;
mov.b32 r259, {high, high};
}
{
mul.f16x2 r261, r233, r259;
}
{
neg.f16x2 r264, r261;
}
{
fma.rn.f16x2 r266, r197, r257, r264;
}
{
mul.f16x2 r270, r197, r259;
}
{
fma.rn.f16x2 r273, r233, r257, r270;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r254;
mov.b32 r277, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r254;
mov.b32 r279, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f21;
cvt.rn.f16.f32 high, f22;
mov.b32 r281, {low, high};
}
{
mul.f16x2 r282, r279, r281;
}
{
mul.f16x2 r285, r254, r277;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r254;
mov.b32 r288, {high, low};
}
{
fma.rn.f16x2 r290, r282, r288, r285;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r290;
mov.b32 r294, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r290;
mov.b32 r296, {high, high};
}
{
mul.f16x2 r298, r251, r296;
}
{
neg.f16x2 r301, r298;
}
{
fma.rn.f16x2 r303, r215, r294, r301;
}
{
mul.f16x2 r307, r215, r296;
}
{
fma.rn.f16x2 r310, r251, r294, r307;
}
barrier.sync 0;
mad.lo.s32 r435, r430, 36, r434;
st.shared.u32 [r435], r173;
st.shared.u32 [r435+12], r266;
st.shared.u32 [r435+24], r303;
barrier.sync 0;
ld.shared.u32 r339, [r429];
ld.shared.u32 r336, [r429+36];
ld.shared.u32 r337, [r429+72];
barrier.sync 0;
st.shared.u32 [r435], r179;
st.shared.u32 [r435+12], r273;
st.shared.u32 [r435+24], r310;
barrier.sync 0;
ld.shared.u32 r345, [r429];
ld.shared.u32 r342, [r429+36];
ld.shared.u32 r343, [r429+72];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f26;
cvt.rn.f16.f32 high, f26;
mov.b32 r331, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f28;
cvt.rn.f16.f32 high, f28;
mov.b32 r332, {low, high};
}
{
neg.f16x2 r333, r332;
}
{
add.f16x2 r335, r336, r337;
}
{
add.f16x2 %0, r339, r335;
}
{
add.f16x2 r341, r342, r343;
}
{
add.f16x2 %1, r345, r341;
}
{
add.f16x2 r347, r336, r337;
}
{
mul.f16x2 r350, r347, r331;
}
{
add.f16x2 r353, r339, r350;
}
{
sub.f16x2 r356, r342, r343;
}
{
mul.f16x2 r359, r356, r333;
}
{
add.f16x2 %2, r353, r359;
}
{
add.f16x2 r365, r336, r337;
}
{
mul.f16x2 r368, r365, r331;
}
{
add.f16x2 r371, r339, r368;
}
{
sub.f16x2 r374, r342, r343;
}
{
mul.f16x2 r377, r374, r333;
}
{
sub.f16x2 %4, r371, r377;
}
{
add.f16x2 r383, r342, r343;
}
{
mul.f16x2 r386, r383, r331;
}
{
add.f16x2 r389, r345, r386;
}
{
sub.f16x2 r392, r336, r337;
}
{
mul.f16x2 r395, r392, r333;
}
{
sub.f16x2 %3, r389, r395;
}
{
add.f16x2 r401, r342, r343;
}
{
mul.f16x2 r404, r401, r331;
}
{
add.f16x2 r407, r345, r404;
}
{
sub.f16x2 r410, r336, r337;
}
{
mul.f16x2 r413, r410, r333;
}
{
add.f16x2 %5, r407, r413;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)));
};


#endif
