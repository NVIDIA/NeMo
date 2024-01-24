#ifndef CUFFTDX_FFT_18_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_18_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<750, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<115>;
.reg .b32 r<1373>;
.reg .f64 fd<91>;
.reg .b64 rd<2>;
mov.f64 fd67, 0dBFE0000000000000;
{
cvt.rn.f16.f64 rs1, fd67;
}
mov.b32 r72, {rs1, rs1};
mov.f64 fd68, 0dBFEBB67AE8584CAA;
{
cvt.rn.f16.f64 rs2, fd68;
}
{
neg.f16 rs3, rs2;
}
mov.b32 r81, {rs3, rs3};
{
add.f16x2 r1, %48, %60;
}
{
add.f16x2 r4, %36, r1;
}
{
add.f16x2 r7, %49, %61;
}
{
add.f16x2 r10, %37, r7;
}
{
add.f16x2 r13, %48, %60;
}
{
mul.f16x2 r16, r13, r72;
}
{
add.f16x2 r19, %36, r16;
}
{
sub.f16x2 r22, %49, %61;
}
{
mul.f16x2 r25, r22, r81;
}
{
add.f16x2 r28, r19, r25;
}
{
add.f16x2 r31, %48, %60;
}
{
mul.f16x2 r34, r31, r72;
}
{
add.f16x2 r37, %36, r34;
}
{
sub.f16x2 r40, %49, %61;
}
{
mul.f16x2 r43, r40, r81;
}
{
sub.f16x2 r46, r37, r43;
}
{
add.f16x2 r49, %49, %61;
}
{
mul.f16x2 r52, r49, r72;
}
{
add.f16x2 r55, %37, r52;
}
{
sub.f16x2 r58, %48, %60;
}
{
mul.f16x2 r61, r58, r81;
}
{
sub.f16x2 r64, r55, r61;
}
{
add.f16x2 r67, %49, %61;
}
{
mul.f16x2 r70, r67, r72;
}
{
add.f16x2 r73, %37, r70;
}
{
sub.f16x2 r76, %48, %60;
}
{
mul.f16x2 r79, r76, r81;
}
{
add.f16x2 r82, r73, r79;
}
{
cvt.rn.f16.f64 rs5, fd67;
}
mov.b32 r156, {rs5, rs5};
{
cvt.rn.f16.f64 rs6, fd68;
}
{
neg.f16 rs7, rs6;
}
mov.b32 r165, {rs7, rs7};
{
add.f16x2 r85, %52, %64;
}
{
add.f16x2 r88, %40, r85;
}
{
add.f16x2 r91, %53, %65;
}
{
add.f16x2 r94, %41, r91;
}
{
add.f16x2 r97, %52, %64;
}
{
mul.f16x2 r100, r97, r156;
}
{
add.f16x2 r103, %40, r100;
}
{
sub.f16x2 r106, %53, %65;
}
{
mul.f16x2 r109, r106, r165;
}
{
add.f16x2 r112, r103, r109;
}
{
add.f16x2 r115, %52, %64;
}
{
mul.f16x2 r118, r115, r156;
}
{
add.f16x2 r121, %40, r118;
}
{
sub.f16x2 r124, %53, %65;
}
{
mul.f16x2 r127, r124, r165;
}
{
sub.f16x2 r130, r121, r127;
}
{
add.f16x2 r133, %53, %65;
}
{
mul.f16x2 r136, r133, r156;
}
{
add.f16x2 r139, %41, r136;
}
{
sub.f16x2 r142, %52, %64;
}
{
mul.f16x2 r145, r142, r165;
}
{
sub.f16x2 r148, r139, r145;
}
{
add.f16x2 r151, %53, %65;
}
{
mul.f16x2 r154, r151, r156;
}
{
add.f16x2 r157, %41, r154;
}
{
sub.f16x2 r160, %52, %64;
}
{
mul.f16x2 r163, r160, r165;
}
{
add.f16x2 r166, r157, r163;
}
{
cvt.rn.f16.f64 rs9, fd67;
}
mov.b32 r240, {rs9, rs9};
{
cvt.rn.f16.f64 rs10, fd68;
}
{
neg.f16 rs11, rs10;
}
mov.b32 r249, {rs11, rs11};
{
add.f16x2 r169, %56, %68;
}
{
add.f16x2 r172, %44, r169;
}
{
add.f16x2 r175, %57, %69;
}
{
add.f16x2 r178, %45, r175;
}
{
add.f16x2 r181, %56, %68;
}
{
mul.f16x2 r184, r181, r240;
}
{
add.f16x2 r187, %44, r184;
}
{
sub.f16x2 r190, %57, %69;
}
{
mul.f16x2 r193, r190, r249;
}
{
add.f16x2 r196, r187, r193;
}
{
add.f16x2 r199, %56, %68;
}
{
mul.f16x2 r202, r199, r240;
}
{
add.f16x2 r205, %44, r202;
}
{
sub.f16x2 r208, %57, %69;
}
{
mul.f16x2 r211, r208, r249;
}
{
sub.f16x2 r214, r205, r211;
}
{
add.f16x2 r217, %57, %69;
}
{
mul.f16x2 r220, r217, r240;
}
{
add.f16x2 r223, %45, r220;
}
{
sub.f16x2 r226, %56, %68;
}
{
mul.f16x2 r229, r226, r249;
}
{
sub.f16x2 r232, r223, r229;
}
{
add.f16x2 r235, %57, %69;
}
{
mul.f16x2 r238, r235, r240;
}
{
add.f16x2 r241, %45, r238;
}
{
sub.f16x2 r244, %56, %68;
}
{
mul.f16x2 r247, r244, r249;
}
{
add.f16x2 r250, r241, r247;
}
mov.f64 fd59, 0d3FE8836FA2CF5039;
{
cvt.rn.f16.f64 rs13, fd59;
}
mov.f64 fd70, 0dBFE491B7523C161D;
{
cvt.rn.f16.f64 rs14, fd70;
}
mov.f64 fd63, 0d3FC63A1A7E0B738A;
{
cvt.rn.f16.f64 rs15, fd63;
}
mov.f64 fd66, 0dBFEF838B8C811C17;
{
cvt.rn.f16.f64 rs16, fd66;
}
mov.f64 fd71, 0dBFEE11F642522D1C;
{
cvt.rn.f16.f64 rs19, fd71;
}
mov.f64 fd72, 0dBFD5E3A8748A0BF5;
{
cvt.rn.f16.f64 rs20, fd72;
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
cvt.rn.f16.f64 rs29, fd67;
}
mov.b32 r388, {rs29, rs29};
{
cvt.rn.f16.f64 rs30, fd68;
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
cvt.rn.f16.f64 rs33, fd67;
}
mov.b32 r472, {rs33, rs33};
{
cvt.rn.f16.f64 rs34, fd68;
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
cvt.rn.f16.f64 rs37, fd67;
}
mov.b32 r556, {rs37, rs37};
{
cvt.rn.f16.f64 rs38, fd68;
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
cvt.rn.f16.f64 rs41, fd67;
}
mov.b32 r640, {rs41, rs41};
{
cvt.rn.f16.f64 rs42, fd68;
}
{
neg.f16 rs43, rs42;
}
mov.b32 r649, {rs43, rs43};
{
add.f16x2 r569, %50, %62;
}
{
add.f16x2 r572, %38, r569;
}
{
add.f16x2 r575, %51, %63;
}
{
add.f16x2 r578, %39, r575;
}
{
add.f16x2 r581, %50, %62;
}
{
mul.f16x2 r584, r581, r640;
}
{
add.f16x2 r587, %38, r584;
}
{
sub.f16x2 r590, %51, %63;
}
{
mul.f16x2 r593, r590, r649;
}
{
add.f16x2 r596, r587, r593;
}
{
add.f16x2 r599, %50, %62;
}
{
mul.f16x2 r602, r599, r640;
}
{
add.f16x2 r605, %38, r602;
}
{
sub.f16x2 r608, %51, %63;
}
{
mul.f16x2 r611, r608, r649;
}
{
sub.f16x2 r614, r605, r611;
}
{
add.f16x2 r617, %51, %63;
}
{
mul.f16x2 r620, r617, r640;
}
{
add.f16x2 r623, %39, r620;
}
{
sub.f16x2 r626, %50, %62;
}
{
mul.f16x2 r629, r626, r649;
}
{
sub.f16x2 r632, r623, r629;
}
{
add.f16x2 r635, %51, %63;
}
{
mul.f16x2 r638, r635, r640;
}
{
add.f16x2 r641, %39, r638;
}
{
sub.f16x2 r644, %50, %62;
}
{
mul.f16x2 r647, r644, r649;
}
{
add.f16x2 r650, r641, r647;
}
{
cvt.rn.f16.f64 rs45, fd67;
}
mov.b32 r724, {rs45, rs45};
{
cvt.rn.f16.f64 rs46, fd68;
}
{
neg.f16 rs47, rs46;
}
mov.b32 r733, {rs47, rs47};
{
add.f16x2 r653, %54, %66;
}
{
add.f16x2 r656, %42, r653;
}
{
add.f16x2 r659, %55, %67;
}
{
add.f16x2 r662, %43, r659;
}
{
add.f16x2 r665, %54, %66;
}
{
mul.f16x2 r668, r665, r724;
}
{
add.f16x2 r671, %42, r668;
}
{
sub.f16x2 r674, %55, %67;
}
{
mul.f16x2 r677, r674, r733;
}
{
add.f16x2 r680, r671, r677;
}
{
add.f16x2 r683, %54, %66;
}
{
mul.f16x2 r686, r683, r724;
}
{
add.f16x2 r689, %42, r686;
}
{
sub.f16x2 r692, %55, %67;
}
{
mul.f16x2 r695, r692, r733;
}
{
sub.f16x2 r698, r689, r695;
}
{
add.f16x2 r701, %55, %67;
}
{
mul.f16x2 r704, r701, r724;
}
{
add.f16x2 r707, %43, r704;
}
{
sub.f16x2 r710, %54, %66;
}
{
mul.f16x2 r713, r710, r733;
}
{
sub.f16x2 r716, r707, r713;
}
{
add.f16x2 r719, %55, %67;
}
{
mul.f16x2 r722, r719, r724;
}
{
add.f16x2 r725, %43, r722;
}
{
sub.f16x2 r728, %54, %66;
}
{
mul.f16x2 r731, r728, r733;
}
{
add.f16x2 r734, r725, r731;
}
{
cvt.rn.f16.f64 rs49, fd67;
}
mov.b32 r808, {rs49, rs49};
{
cvt.rn.f16.f64 rs50, fd68;
}
{
neg.f16 rs51, rs50;
}
mov.b32 r817, {rs51, rs51};
{
add.f16x2 r737, %58, %70;
}
{
add.f16x2 r740, %46, r737;
}
{
add.f16x2 r743, %59, %71;
}
{
add.f16x2 r746, %47, r743;
}
{
add.f16x2 r749, %58, %70;
}
{
mul.f16x2 r752, r749, r808;
}
{
add.f16x2 r755, %46, r752;
}
{
sub.f16x2 r758, %59, %71;
}
{
mul.f16x2 r761, r758, r817;
}
{
add.f16x2 r764, r755, r761;
}
{
add.f16x2 r767, %58, %70;
}
{
mul.f16x2 r770, r767, r808;
}
{
add.f16x2 r773, %46, r770;
}
{
sub.f16x2 r776, %59, %71;
}
{
mul.f16x2 r779, r776, r817;
}
{
sub.f16x2 r782, r773, r779;
}
{
add.f16x2 r785, %59, %71;
}
{
mul.f16x2 r788, r785, r808;
}
{
add.f16x2 r791, %47, r788;
}
{
sub.f16x2 r794, %58, %70;
}
{
mul.f16x2 r797, r794, r817;
}
{
sub.f16x2 r800, r791, r797;
}
{
add.f16x2 r803, %59, %71;
}
{
mul.f16x2 r806, r803, r808;
}
{
add.f16x2 r809, %47, r806;
}
{
sub.f16x2 r812, %58, %70;
}
{
mul.f16x2 r815, r812, r817;
}
{
add.f16x2 r818, r809, r815;
}
{
cvt.rn.f16.f64 rs53, fd59;
}
{
cvt.rn.f16.f64 rs54, fd70;
}
{
cvt.rn.f16.f64 rs55, fd63;
}
{
cvt.rn.f16.f64 rs56, fd66;
}
{
cvt.rn.f16.f64 rs59, fd71;
}
{
cvt.rn.f16.f64 rs60, fd72;
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
cvt.rn.f16.f64 rs69, fd67;
}
mov.b32 r956, {rs69, rs69};
{
cvt.rn.f16.f64 rs70, fd68;
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
cvt.rn.f16.f64 rs73, fd67;
}
mov.b32 r1040, {rs73, rs73};
{
cvt.rn.f16.f64 rs74, fd68;
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
cvt.rn.f16.f64 rs77, fd67;
}
mov.b32 r1124, {rs77, rs77};
{
cvt.rn.f16.f64 rs78, fd68;
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
mov.f64 fd57, 0d3FEE11F642522D1C;
{
cvt.rn.f16.f64 rs81, fd57;
}
{
cvt.rn.f16.f64 rs82, fd72;
}
{
cvt.rn.f16.f64 rs83, fd59;
}
{
cvt.rn.f16.f64 rs84, fd70;
}
mov.f64 fd61, 0d3FE0000000000000;
{
cvt.rn.f16.f64 rs85, fd61;
}
{
cvt.rn.f16.f64 rs86, fd68;
}
{
cvt.rn.f16.f64 rs87, fd63;
}
{
cvt.rn.f16.f64 rs88, fd66;
}
mov.f64 fd65, 0dBFC63A1A7E0B738A;
{
cvt.rn.f16.f64 rs89, fd65;
}
{
cvt.rn.f16.f64 rs90, fd66;
}
{
cvt.rn.f16.f64 rs91, fd67;
}
{
cvt.rn.f16.f64 rs92, fd68;
}
mov.f64 fd69, 0dBFE8836FA2CF5039;
{
cvt.rn.f16.f64 rs93, fd69;
}
{
cvt.rn.f16.f64 rs94, fd70;
}
{
cvt.rn.f16.f64 rs95, fd71;
}
{
cvt.rn.f16.f64 rs96, fd72;
}
mov.b32 r1151, {rs81, rs81};
{
mul.f16x2 r1137, r972, r1151;
}
mov.b32 r1148, {rs82, rs82};
{
mul.f16x2 r1140, r978, r1148;
}
{
sub.f16x2 r1143, r1137, r1140;
}
{
mul.f16x2 r1146, r972, r1148;
}
{
fma.rn.f16x2 r1149, r978, r1151, r1146;
}
mov.b32 r1167, {rs83, rs83};
{
mul.f16x2 r1153, r1056, r1167;
}
mov.b32 r1164, {rs84, rs84};
{
mul.f16x2 r1156, r1062, r1164;
}
{
sub.f16x2 r1159, r1153, r1156;
}
{
mul.f16x2 r1162, r1056, r1164;
}
{
fma.rn.f16x2 r1165, r1062, r1167, r1162;
}
mov.b32 r1183, {rs85, rs85};
{
mul.f16x2 r1169, r912, r1183;
}
mov.b32 r1180, {rs86, rs86};
{
mul.f16x2 r1172, r948, r1180;
}
{
sub.f16x2 r1175, r1169, r1172;
}
{
mul.f16x2 r1178, r912, r1180;
}
{
fma.rn.f16x2 r1181, r948, r1183, r1178;
}
mov.b32 r1199, {rs87, rs87};
{
mul.f16x2 r1185, r996, r1199;
}
mov.b32 r1196, {rs88, rs88};
{
mul.f16x2 r1188, r1032, r1196;
}
{
sub.f16x2 r1191, r1185, r1188;
}
{
mul.f16x2 r1194, r996, r1196;
}
{
fma.rn.f16x2 r1197, r1032, r1199, r1194;
}
mov.b32 r1215, {rs89, rs89};
{
mul.f16x2 r1201, r1080, r1215;
}
mov.b32 r1212, {rs90, rs90};
{
mul.f16x2 r1204, r1116, r1212;
}
{
sub.f16x2 r1207, r1201, r1204;
}
{
mul.f16x2 r1210, r1080, r1212;
}
{
fma.rn.f16x2 r1213, r1116, r1215, r1210;
}
mov.b32 r1231, {rs91, rs91};
{
mul.f16x2 r1217, r930, r1231;
}
mov.b32 r1228, {rs92, rs92};
{
mul.f16x2 r1220, r966, r1228;
}
{
sub.f16x2 r1223, r1217, r1220;
}
{
mul.f16x2 r1226, r930, r1228;
}
{
fma.rn.f16x2 r1229, r966, r1231, r1226;
}
mov.b32 r1247, {rs93, rs93};
{
mul.f16x2 r1233, r1014, r1247;
}
mov.b32 r1244, {rs94, rs94};
{
mul.f16x2 r1236, r1050, r1244;
}
{
sub.f16x2 r1239, r1233, r1236;
}
{
mul.f16x2 r1242, r1014, r1244;
}
{
fma.rn.f16x2 r1245, r1050, r1247, r1242;
}
mov.b32 r1263, {rs95, rs95};
{
mul.f16x2 r1249, r1098, r1263;
}
mov.b32 r1260, {rs96, rs96};
{
mul.f16x2 r1252, r1134, r1260;
}
{
sub.f16x2 r1255, r1249, r1252;
}
{
mul.f16x2 r1258, r1098, r1260;
}
{
fma.rn.f16x2 r1261, r1134, r1263, r1258;
}
{
add.f16x2 %0, r320, r888;
}
{
add.f16x2 %1, r326, r894;
}
{
sub.f16x2 %18, r320, r888;
}
{
sub.f16x2 %19, r326, r894;
}
{
add.f16x2 %2, r404, r1143;
}
{
add.f16x2 %3, r410, r1149;
}
{
sub.f16x2 %20, r404, r1143;
}
{
sub.f16x2 %21, r410, r1149;
}
{
add.f16x2 %4, r488, r1159;
}
{
add.f16x2 %5, r494, r1165;
}
{
sub.f16x2 %22, r488, r1159;
}
{
sub.f16x2 %23, r494, r1165;
}
{
add.f16x2 %6, r344, r1175;
}
{
add.f16x2 %7, r380, r1181;
}
{
sub.f16x2 %24, r344, r1175;
}
{
sub.f16x2 %25, r380, r1181;
}
{
add.f16x2 %8, r428, r1191;
}
{
add.f16x2 %9, r464, r1197;
}
{
sub.f16x2 %26, r428, r1191;
}
{
sub.f16x2 %27, r464, r1197;
}
{
add.f16x2 %10, r512, r1207;
}
{
add.f16x2 %11, r548, r1213;
}
{
sub.f16x2 %28, r512, r1207;
}
{
sub.f16x2 %29, r548, r1213;
}
{
add.f16x2 %12, r362, r1223;
}
{
add.f16x2 %13, r398, r1229;
}
{
sub.f16x2 %30, r362, r1223;
}
{
sub.f16x2 %31, r398, r1229;
}
{
add.f16x2 %14, r446, r1239;
}
{
add.f16x2 %15, r482, r1245;
}
{
sub.f16x2 %32, r446, r1239;
}
{
sub.f16x2 %33, r482, r1245;
}
{
add.f16x2 %16, r530, r1255;
}
{
add.f16x2 %17, r566, r1261;
}
{
sub.f16x2 %34, r530, r1255;
}
{
sub.f16x2 %35, r566, r1261;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[17].y)));
};


#endif
