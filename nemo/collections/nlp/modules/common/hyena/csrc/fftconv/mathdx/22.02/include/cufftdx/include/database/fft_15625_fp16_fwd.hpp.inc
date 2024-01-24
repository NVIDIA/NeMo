#ifndef CUFFTDX_FFT_15625_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_15625_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1179, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.local .align 8 .b8 __local_depot0[200];
.reg .b64 SP;
.reg .b64 SPL;
.reg .pred p<3>;
.reg .f32 f<695>;
.reg .b32 r<11354>;
.reg .b64 rd<21>;
mov.u64 SPL, __local_depot0;
add.u64 rd3, SPL, 0;
mov.u32 r3531, %tid.y;
mul.lo.s32 r1, r3531, 15625;
add.s64 rd4, rd3, 4;
mov.f32 f214, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r23, {low, high};
}
mov.f32 f216, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r24, {low, high};
}
mov.f32 f210, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r25, {low, high};
}
mov.f32 f212, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r26, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r27, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r28, {low, high};
}
{
neg.f16x2 r29, r28;
}
{
add.f16x2 r31, %61, %91;
}
{
add.f16x2 r34, %51, r31;
}
{
add.f16x2 r37, %71, %81;
}
{
add.f16x2 r40, r34, r37;
}
{
add.f16x2 r43, %62, %92;
}
{
add.f16x2 r46, %52, r43;
}
{
add.f16x2 r49, %72, %82;
}
{
add.f16x2 r52, r46, r49;
}
{
add.f16x2 r55, %61, %91;
}
{
mul.f16x2 r58, r55, r23;
}
{
add.f16x2 r61, %51, r58;
}
{
add.f16x2 r64, %71, %81;
}
{
mul.f16x2 r67, r64, r25;
}
{
add.f16x2 r70, r61, r67;
}
{
sub.f16x2 r73, %62, %92;
}
{
mul.f16x2 r76, r73, r24;
}
{
sub.f16x2 r79, %72, %82;
}
{
mul.f16x2 r82, r79, r26;
}
{
add.f16x2 r85, r76, r82;
}
{
sub.f16x2 r88, r70, r85;
}
{
add.f16x2 r91, %61, %91;
}
{
mul.f16x2 r94, r91, r23;
}
{
add.f16x2 r97, %51, r94;
}
{
add.f16x2 r100, %71, %81;
}
{
mul.f16x2 r103, r100, r25;
}
{
add.f16x2 r106, r97, r103;
}
{
sub.f16x2 r109, %62, %92;
}
{
mul.f16x2 r112, r109, r24;
}
{
sub.f16x2 r115, %72, %82;
}
{
mul.f16x2 r118, r115, r26;
}
{
add.f16x2 r121, r112, r118;
}
{
add.f16x2 r124, r106, r121;
}
{
add.f16x2 r127, %61, %91;
}
{
mul.f16x2 r130, r127, r25;
}
{
add.f16x2 r133, %51, r130;
}
{
add.f16x2 r136, %71, %81;
}
{
mul.f16x2 r139, r136, r27;
}
{
add.f16x2 r142, r133, r139;
}
{
sub.f16x2 r145, %62, %92;
}
{
mul.f16x2 r148, r145, r26;
}
{
sub.f16x2 r151, %72, %82;
}
{
mul.f16x2 r154, r151, r29;
}
{
add.f16x2 r157, r148, r154;
}
{
sub.f16x2 r160, r142, r157;
}
{
add.f16x2 r163, %61, %91;
}
{
mul.f16x2 r166, r163, r25;
}
{
add.f16x2 r169, %51, r166;
}
{
add.f16x2 r172, %71, %81;
}
{
mul.f16x2 r175, r172, r27;
}
{
add.f16x2 r178, r169, r175;
}
{
sub.f16x2 r181, %62, %92;
}
{
mul.f16x2 r184, r181, r26;
}
{
sub.f16x2 r187, %72, %82;
}
{
mul.f16x2 r190, r187, r29;
}
{
add.f16x2 r193, r184, r190;
}
{
add.f16x2 r196, r178, r193;
}
{
add.f16x2 r199, %62, %92;
}
{
mul.f16x2 r202, r199, r23;
}
{
add.f16x2 r205, %52, r202;
}
{
add.f16x2 r208, %72, %82;
}
{
mul.f16x2 r211, r208, r25;
}
{
add.f16x2 r214, r205, r211;
}
{
sub.f16x2 r217, %61, %91;
}
{
mul.f16x2 r220, r217, r24;
}
{
sub.f16x2 r223, %71, %81;
}
{
mul.f16x2 r226, r223, r26;
}
{
add.f16x2 r229, r220, r226;
}
{
add.f16x2 r232, r214, r229;
}
{
add.f16x2 r235, %62, %92;
}
{
mul.f16x2 r238, r235, r23;
}
{
add.f16x2 r241, %52, r238;
}
{
add.f16x2 r244, %72, %82;
}
{
mul.f16x2 r247, r244, r25;
}
{
add.f16x2 r250, r241, r247;
}
{
sub.f16x2 r253, %61, %91;
}
{
mul.f16x2 r256, r253, r24;
}
{
sub.f16x2 r259, %71, %81;
}
{
mul.f16x2 r262, r259, r26;
}
{
add.f16x2 r265, r256, r262;
}
{
sub.f16x2 r268, r250, r265;
}
{
add.f16x2 r271, %62, %92;
}
{
mul.f16x2 r274, r271, r25;
}
{
add.f16x2 r277, %52, r274;
}
{
add.f16x2 r280, %72, %82;
}
{
mul.f16x2 r283, r280, r27;
}
{
add.f16x2 r286, r277, r283;
}
{
sub.f16x2 r289, %61, %91;
}
{
mul.f16x2 r292, r289, r26;
}
{
sub.f16x2 r295, %71, %81;
}
{
mul.f16x2 r298, r295, r29;
}
{
add.f16x2 r301, r292, r298;
}
{
add.f16x2 r304, r286, r301;
}
{
add.f16x2 r307, %62, %92;
}
{
mul.f16x2 r310, r307, r25;
}
{
add.f16x2 r313, %52, r310;
}
{
add.f16x2 r316, %72, %82;
}
{
mul.f16x2 r319, r316, r27;
}
{
add.f16x2 r322, r313, r319;
}
{
sub.f16x2 r325, %61, %91;
}
{
mul.f16x2 r328, r325, r26;
}
{
sub.f16x2 r331, %71, %81;
}
{
mul.f16x2 r334, r331, r29;
}
{
add.f16x2 r337, r328, r334;
}
{
sub.f16x2 r340, r322, r337;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r343, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r344, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r345, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r346, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r347, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r348, {low, high};
}
{
neg.f16x2 r349, r348;
}
{
add.f16x2 r351, %63, %93;
}
{
add.f16x2 r354, %53, r351;
}
{
add.f16x2 r357, %73, %83;
}
{
add.f16x2 r360, r354, r357;
}
{
add.f16x2 r363, %64, %94;
}
{
add.f16x2 r366, %54, r363;
}
{
add.f16x2 r369, %74, %84;
}
{
add.f16x2 r372, r366, r369;
}
{
add.f16x2 r375, %63, %93;
}
{
mul.f16x2 r378, r375, r343;
}
{
add.f16x2 r381, %53, r378;
}
{
add.f16x2 r384, %73, %83;
}
{
mul.f16x2 r387, r384, r345;
}
{
add.f16x2 r390, r381, r387;
}
{
sub.f16x2 r393, %64, %94;
}
{
mul.f16x2 r396, r393, r344;
}
{
sub.f16x2 r399, %74, %84;
}
{
mul.f16x2 r402, r399, r346;
}
{
add.f16x2 r405, r396, r402;
}
{
sub.f16x2 r408, r390, r405;
}
{
add.f16x2 r411, %63, %93;
}
{
mul.f16x2 r414, r411, r343;
}
{
add.f16x2 r417, %53, r414;
}
{
add.f16x2 r420, %73, %83;
}
{
mul.f16x2 r423, r420, r345;
}
{
add.f16x2 r426, r417, r423;
}
{
sub.f16x2 r429, %64, %94;
}
{
mul.f16x2 r432, r429, r344;
}
{
sub.f16x2 r435, %74, %84;
}
{
mul.f16x2 r438, r435, r346;
}
{
add.f16x2 r441, r432, r438;
}
{
add.f16x2 r444, r426, r441;
}
{
add.f16x2 r447, %63, %93;
}
{
mul.f16x2 r450, r447, r345;
}
{
add.f16x2 r453, %53, r450;
}
{
add.f16x2 r456, %73, %83;
}
{
mul.f16x2 r459, r456, r347;
}
{
add.f16x2 r462, r453, r459;
}
{
sub.f16x2 r465, %64, %94;
}
{
mul.f16x2 r468, r465, r346;
}
{
sub.f16x2 r471, %74, %84;
}
{
mul.f16x2 r474, r471, r349;
}
{
add.f16x2 r477, r468, r474;
}
{
sub.f16x2 r480, r462, r477;
}
{
add.f16x2 r483, %63, %93;
}
{
mul.f16x2 r486, r483, r345;
}
{
add.f16x2 r489, %53, r486;
}
{
add.f16x2 r492, %73, %83;
}
{
mul.f16x2 r495, r492, r347;
}
{
add.f16x2 r498, r489, r495;
}
{
sub.f16x2 r501, %64, %94;
}
{
mul.f16x2 r504, r501, r346;
}
{
sub.f16x2 r507, %74, %84;
}
{
mul.f16x2 r510, r507, r349;
}
{
add.f16x2 r513, r504, r510;
}
{
add.f16x2 r516, r498, r513;
}
{
add.f16x2 r519, %64, %94;
}
{
mul.f16x2 r522, r519, r343;
}
{
add.f16x2 r525, %54, r522;
}
{
add.f16x2 r528, %74, %84;
}
{
mul.f16x2 r531, r528, r345;
}
{
add.f16x2 r534, r525, r531;
}
{
sub.f16x2 r537, %63, %93;
}
{
mul.f16x2 r540, r537, r344;
}
{
sub.f16x2 r543, %73, %83;
}
{
mul.f16x2 r546, r543, r346;
}
{
add.f16x2 r549, r540, r546;
}
{
add.f16x2 r552, r534, r549;
}
{
add.f16x2 r555, %64, %94;
}
{
mul.f16x2 r558, r555, r343;
}
{
add.f16x2 r561, %54, r558;
}
{
add.f16x2 r564, %74, %84;
}
{
mul.f16x2 r567, r564, r345;
}
{
add.f16x2 r570, r561, r567;
}
{
sub.f16x2 r573, %63, %93;
}
{
mul.f16x2 r576, r573, r344;
}
{
sub.f16x2 r579, %73, %83;
}
{
mul.f16x2 r582, r579, r346;
}
{
add.f16x2 r585, r576, r582;
}
{
sub.f16x2 r588, r570, r585;
}
{
add.f16x2 r591, %64, %94;
}
{
mul.f16x2 r594, r591, r345;
}
{
add.f16x2 r597, %54, r594;
}
{
add.f16x2 r600, %74, %84;
}
{
mul.f16x2 r603, r600, r347;
}
{
add.f16x2 r606, r597, r603;
}
{
sub.f16x2 r609, %63, %93;
}
{
mul.f16x2 r612, r609, r346;
}
{
sub.f16x2 r615, %73, %83;
}
{
mul.f16x2 r618, r615, r349;
}
{
add.f16x2 r621, r612, r618;
}
{
add.f16x2 r624, r606, r621;
}
{
add.f16x2 r627, %64, %94;
}
{
mul.f16x2 r630, r627, r345;
}
{
add.f16x2 r633, %54, r630;
}
{
add.f16x2 r636, %74, %84;
}
{
mul.f16x2 r639, r636, r347;
}
{
add.f16x2 r642, r633, r639;
}
{
sub.f16x2 r645, %63, %93;
}
{
mul.f16x2 r648, r645, r346;
}
{
sub.f16x2 r651, %73, %83;
}
{
mul.f16x2 r654, r651, r349;
}
{
add.f16x2 r657, r648, r654;
}
{
sub.f16x2 r660, r642, r657;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r663, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r664, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r665, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r666, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r667, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r668, {low, high};
}
{
neg.f16x2 r669, r668;
}
{
add.f16x2 r671, %65, %95;
}
{
add.f16x2 r674, %55, r671;
}
{
add.f16x2 r677, %75, %85;
}
{
add.f16x2 r680, r674, r677;
}
{
add.f16x2 r683, %66, %96;
}
{
add.f16x2 r686, %56, r683;
}
{
add.f16x2 r689, %76, %86;
}
{
add.f16x2 r692, r686, r689;
}
{
add.f16x2 r695, %65, %95;
}
{
mul.f16x2 r698, r695, r663;
}
{
add.f16x2 r701, %55, r698;
}
{
add.f16x2 r704, %75, %85;
}
{
mul.f16x2 r707, r704, r665;
}
{
add.f16x2 r710, r701, r707;
}
{
sub.f16x2 r713, %66, %96;
}
{
mul.f16x2 r716, r713, r664;
}
{
sub.f16x2 r719, %76, %86;
}
{
mul.f16x2 r722, r719, r666;
}
{
add.f16x2 r725, r716, r722;
}
{
sub.f16x2 r728, r710, r725;
}
{
add.f16x2 r731, %65, %95;
}
{
mul.f16x2 r734, r731, r663;
}
{
add.f16x2 r737, %55, r734;
}
{
add.f16x2 r740, %75, %85;
}
{
mul.f16x2 r743, r740, r665;
}
{
add.f16x2 r746, r737, r743;
}
{
sub.f16x2 r749, %66, %96;
}
{
mul.f16x2 r752, r749, r664;
}
{
sub.f16x2 r755, %76, %86;
}
{
mul.f16x2 r758, r755, r666;
}
{
add.f16x2 r761, r752, r758;
}
{
add.f16x2 r764, r746, r761;
}
{
add.f16x2 r767, %65, %95;
}
{
mul.f16x2 r770, r767, r665;
}
{
add.f16x2 r773, %55, r770;
}
{
add.f16x2 r776, %75, %85;
}
{
mul.f16x2 r779, r776, r667;
}
{
add.f16x2 r782, r773, r779;
}
{
sub.f16x2 r785, %66, %96;
}
{
mul.f16x2 r788, r785, r666;
}
{
sub.f16x2 r791, %76, %86;
}
{
mul.f16x2 r794, r791, r669;
}
{
add.f16x2 r797, r788, r794;
}
{
sub.f16x2 r800, r782, r797;
}
{
add.f16x2 r803, %65, %95;
}
{
mul.f16x2 r806, r803, r665;
}
{
add.f16x2 r809, %55, r806;
}
{
add.f16x2 r812, %75, %85;
}
{
mul.f16x2 r815, r812, r667;
}
{
add.f16x2 r818, r809, r815;
}
{
sub.f16x2 r821, %66, %96;
}
{
mul.f16x2 r824, r821, r666;
}
{
sub.f16x2 r827, %76, %86;
}
{
mul.f16x2 r830, r827, r669;
}
{
add.f16x2 r833, r824, r830;
}
{
add.f16x2 r836, r818, r833;
}
{
add.f16x2 r839, %66, %96;
}
{
mul.f16x2 r842, r839, r663;
}
{
add.f16x2 r845, %56, r842;
}
{
add.f16x2 r848, %76, %86;
}
{
mul.f16x2 r851, r848, r665;
}
{
add.f16x2 r854, r845, r851;
}
{
sub.f16x2 r857, %65, %95;
}
{
mul.f16x2 r860, r857, r664;
}
{
sub.f16x2 r863, %75, %85;
}
{
mul.f16x2 r866, r863, r666;
}
{
add.f16x2 r869, r860, r866;
}
{
add.f16x2 r872, r854, r869;
}
{
add.f16x2 r875, %66, %96;
}
{
mul.f16x2 r878, r875, r663;
}
{
add.f16x2 r881, %56, r878;
}
{
add.f16x2 r884, %76, %86;
}
{
mul.f16x2 r887, r884, r665;
}
{
add.f16x2 r890, r881, r887;
}
{
sub.f16x2 r893, %65, %95;
}
{
mul.f16x2 r896, r893, r664;
}
{
sub.f16x2 r899, %75, %85;
}
{
mul.f16x2 r902, r899, r666;
}
{
add.f16x2 r905, r896, r902;
}
{
sub.f16x2 r908, r890, r905;
}
{
add.f16x2 r911, %66, %96;
}
{
mul.f16x2 r914, r911, r665;
}
{
add.f16x2 r917, %56, r914;
}
{
add.f16x2 r920, %76, %86;
}
{
mul.f16x2 r923, r920, r667;
}
{
add.f16x2 r926, r917, r923;
}
{
sub.f16x2 r929, %65, %95;
}
{
mul.f16x2 r932, r929, r666;
}
{
sub.f16x2 r935, %75, %85;
}
{
mul.f16x2 r938, r935, r669;
}
{
add.f16x2 r941, r932, r938;
}
{
add.f16x2 r944, r926, r941;
}
{
add.f16x2 r947, %66, %96;
}
{
mul.f16x2 r950, r947, r665;
}
{
add.f16x2 r953, %56, r950;
}
{
add.f16x2 r956, %76, %86;
}
{
mul.f16x2 r959, r956, r667;
}
{
add.f16x2 r962, r953, r959;
}
{
sub.f16x2 r965, %65, %95;
}
{
mul.f16x2 r968, r965, r666;
}
{
sub.f16x2 r971, %75, %85;
}
{
mul.f16x2 r974, r971, r669;
}
{
add.f16x2 r977, r968, r974;
}
{
sub.f16x2 r980, r962, r977;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r983, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r984, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r985, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r986, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r987, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r988, {low, high};
}
{
neg.f16x2 r989, r988;
}
{
add.f16x2 r991, %67, %97;
}
{
add.f16x2 r994, %57, r991;
}
{
add.f16x2 r997, %77, %87;
}
{
add.f16x2 r1000, r994, r997;
}
{
add.f16x2 r1003, %68, %98;
}
{
add.f16x2 r1006, %58, r1003;
}
{
add.f16x2 r1009, %78, %88;
}
{
add.f16x2 r1012, r1006, r1009;
}
{
add.f16x2 r1015, %67, %97;
}
{
mul.f16x2 r1018, r1015, r983;
}
{
add.f16x2 r1021, %57, r1018;
}
{
add.f16x2 r1024, %77, %87;
}
{
mul.f16x2 r1027, r1024, r985;
}
{
add.f16x2 r1030, r1021, r1027;
}
{
sub.f16x2 r1033, %68, %98;
}
{
mul.f16x2 r1036, r1033, r984;
}
{
sub.f16x2 r1039, %78, %88;
}
{
mul.f16x2 r1042, r1039, r986;
}
{
add.f16x2 r1045, r1036, r1042;
}
{
sub.f16x2 r1048, r1030, r1045;
}
{
add.f16x2 r1051, %67, %97;
}
{
mul.f16x2 r1054, r1051, r983;
}
{
add.f16x2 r1057, %57, r1054;
}
{
add.f16x2 r1060, %77, %87;
}
{
mul.f16x2 r1063, r1060, r985;
}
{
add.f16x2 r1066, r1057, r1063;
}
{
sub.f16x2 r1069, %68, %98;
}
{
mul.f16x2 r1072, r1069, r984;
}
{
sub.f16x2 r1075, %78, %88;
}
{
mul.f16x2 r1078, r1075, r986;
}
{
add.f16x2 r1081, r1072, r1078;
}
{
add.f16x2 r1084, r1066, r1081;
}
{
add.f16x2 r1087, %67, %97;
}
{
mul.f16x2 r1090, r1087, r985;
}
{
add.f16x2 r1093, %57, r1090;
}
{
add.f16x2 r1096, %77, %87;
}
{
mul.f16x2 r1099, r1096, r987;
}
{
add.f16x2 r1102, r1093, r1099;
}
{
sub.f16x2 r1105, %68, %98;
}
{
mul.f16x2 r1108, r1105, r986;
}
{
sub.f16x2 r1111, %78, %88;
}
{
mul.f16x2 r1114, r1111, r989;
}
{
add.f16x2 r1117, r1108, r1114;
}
{
sub.f16x2 r1120, r1102, r1117;
}
{
add.f16x2 r1123, %67, %97;
}
{
mul.f16x2 r1126, r1123, r985;
}
{
add.f16x2 r1129, %57, r1126;
}
{
add.f16x2 r1132, %77, %87;
}
{
mul.f16x2 r1135, r1132, r987;
}
{
add.f16x2 r1138, r1129, r1135;
}
{
sub.f16x2 r1141, %68, %98;
}
{
mul.f16x2 r1144, r1141, r986;
}
{
sub.f16x2 r1147, %78, %88;
}
{
mul.f16x2 r1150, r1147, r989;
}
{
add.f16x2 r1153, r1144, r1150;
}
{
add.f16x2 r1156, r1138, r1153;
}
{
add.f16x2 r1159, %68, %98;
}
{
mul.f16x2 r1162, r1159, r983;
}
{
add.f16x2 r1165, %58, r1162;
}
{
add.f16x2 r1168, %78, %88;
}
{
mul.f16x2 r1171, r1168, r985;
}
{
add.f16x2 r1174, r1165, r1171;
}
{
sub.f16x2 r1177, %67, %97;
}
{
mul.f16x2 r1180, r1177, r984;
}
{
sub.f16x2 r1183, %77, %87;
}
{
mul.f16x2 r1186, r1183, r986;
}
{
add.f16x2 r1189, r1180, r1186;
}
{
add.f16x2 r1192, r1174, r1189;
}
{
add.f16x2 r1195, %68, %98;
}
{
mul.f16x2 r1198, r1195, r983;
}
{
add.f16x2 r1201, %58, r1198;
}
{
add.f16x2 r1204, %78, %88;
}
{
mul.f16x2 r1207, r1204, r985;
}
{
add.f16x2 r1210, r1201, r1207;
}
{
sub.f16x2 r1213, %67, %97;
}
{
mul.f16x2 r1216, r1213, r984;
}
{
sub.f16x2 r1219, %77, %87;
}
{
mul.f16x2 r1222, r1219, r986;
}
{
add.f16x2 r1225, r1216, r1222;
}
{
sub.f16x2 r1228, r1210, r1225;
}
{
add.f16x2 r1231, %68, %98;
}
{
mul.f16x2 r1234, r1231, r985;
}
{
add.f16x2 r1237, %58, r1234;
}
{
add.f16x2 r1240, %78, %88;
}
{
mul.f16x2 r1243, r1240, r987;
}
{
add.f16x2 r1246, r1237, r1243;
}
{
sub.f16x2 r1249, %67, %97;
}
{
mul.f16x2 r1252, r1249, r986;
}
{
sub.f16x2 r1255, %77, %87;
}
{
mul.f16x2 r1258, r1255, r989;
}
{
add.f16x2 r1261, r1252, r1258;
}
{
add.f16x2 r1264, r1246, r1261;
}
{
add.f16x2 r1267, %68, %98;
}
{
mul.f16x2 r1270, r1267, r985;
}
{
add.f16x2 r1273, %58, r1270;
}
{
add.f16x2 r1276, %78, %88;
}
{
mul.f16x2 r1279, r1276, r987;
}
{
add.f16x2 r1282, r1273, r1279;
}
{
sub.f16x2 r1285, %67, %97;
}
{
mul.f16x2 r1288, r1285, r986;
}
{
sub.f16x2 r1291, %77, %87;
}
{
mul.f16x2 r1294, r1291, r989;
}
{
add.f16x2 r1297, r1288, r1294;
}
{
sub.f16x2 r1300, r1282, r1297;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1303, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1304, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r1305, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r1306, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1307, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1308, {low, high};
}
{
neg.f16x2 r1309, r1308;
}
{
add.f16x2 r1311, %69, %99;
}
{
add.f16x2 r1314, %59, r1311;
}
{
add.f16x2 r1317, %79, %89;
}
{
add.f16x2 r1320, r1314, r1317;
}
{
add.f16x2 r1323, %70, %100;
}
{
add.f16x2 r1326, %60, r1323;
}
{
add.f16x2 r1329, %80, %90;
}
{
add.f16x2 r1332, r1326, r1329;
}
{
add.f16x2 r1335, %69, %99;
}
{
mul.f16x2 r1338, r1335, r1303;
}
{
add.f16x2 r1341, %59, r1338;
}
{
add.f16x2 r1344, %79, %89;
}
{
mul.f16x2 r1347, r1344, r1305;
}
{
add.f16x2 r1350, r1341, r1347;
}
{
sub.f16x2 r1353, %70, %100;
}
{
mul.f16x2 r1356, r1353, r1304;
}
{
sub.f16x2 r1359, %80, %90;
}
{
mul.f16x2 r1362, r1359, r1306;
}
{
add.f16x2 r1365, r1356, r1362;
}
{
sub.f16x2 r1368, r1350, r1365;
}
{
add.f16x2 r1371, %69, %99;
}
{
mul.f16x2 r1374, r1371, r1303;
}
{
add.f16x2 r1377, %59, r1374;
}
{
add.f16x2 r1380, %79, %89;
}
{
mul.f16x2 r1383, r1380, r1305;
}
{
add.f16x2 r1386, r1377, r1383;
}
{
sub.f16x2 r1389, %70, %100;
}
{
mul.f16x2 r1392, r1389, r1304;
}
{
sub.f16x2 r1395, %80, %90;
}
{
mul.f16x2 r1398, r1395, r1306;
}
{
add.f16x2 r1401, r1392, r1398;
}
{
add.f16x2 r1404, r1386, r1401;
}
{
add.f16x2 r1407, %69, %99;
}
{
mul.f16x2 r1410, r1407, r1305;
}
{
add.f16x2 r1413, %59, r1410;
}
{
add.f16x2 r1416, %79, %89;
}
{
mul.f16x2 r1419, r1416, r1307;
}
{
add.f16x2 r1422, r1413, r1419;
}
{
sub.f16x2 r1425, %70, %100;
}
{
mul.f16x2 r1428, r1425, r1306;
}
{
sub.f16x2 r1431, %80, %90;
}
{
mul.f16x2 r1434, r1431, r1309;
}
{
add.f16x2 r1437, r1428, r1434;
}
{
sub.f16x2 r1440, r1422, r1437;
}
{
add.f16x2 r1443, %69, %99;
}
{
mul.f16x2 r1446, r1443, r1305;
}
{
add.f16x2 r1449, %59, r1446;
}
{
add.f16x2 r1452, %79, %89;
}
{
mul.f16x2 r1455, r1452, r1307;
}
{
add.f16x2 r1458, r1449, r1455;
}
{
sub.f16x2 r1461, %70, %100;
}
{
mul.f16x2 r1464, r1461, r1306;
}
{
sub.f16x2 r1467, %80, %90;
}
{
mul.f16x2 r1470, r1467, r1309;
}
{
add.f16x2 r1473, r1464, r1470;
}
{
add.f16x2 r1476, r1458, r1473;
}
{
add.f16x2 r1479, %70, %100;
}
{
mul.f16x2 r1482, r1479, r1303;
}
{
add.f16x2 r1485, %60, r1482;
}
{
add.f16x2 r1488, %80, %90;
}
{
mul.f16x2 r1491, r1488, r1305;
}
{
add.f16x2 r1494, r1485, r1491;
}
{
sub.f16x2 r1497, %69, %99;
}
{
mul.f16x2 r1500, r1497, r1304;
}
{
sub.f16x2 r1503, %79, %89;
}
{
mul.f16x2 r1506, r1503, r1306;
}
{
add.f16x2 r1509, r1500, r1506;
}
{
add.f16x2 r1512, r1494, r1509;
}
{
add.f16x2 r1515, %70, %100;
}
{
mul.f16x2 r1518, r1515, r1303;
}
{
add.f16x2 r1521, %60, r1518;
}
{
add.f16x2 r1524, %80, %90;
}
{
mul.f16x2 r1527, r1524, r1305;
}
{
add.f16x2 r1530, r1521, r1527;
}
{
sub.f16x2 r1533, %69, %99;
}
{
mul.f16x2 r1536, r1533, r1304;
}
{
sub.f16x2 r1539, %79, %89;
}
{
mul.f16x2 r1542, r1539, r1306;
}
{
add.f16x2 r1545, r1536, r1542;
}
{
sub.f16x2 r1548, r1530, r1545;
}
{
add.f16x2 r1551, %70, %100;
}
{
mul.f16x2 r1554, r1551, r1305;
}
{
add.f16x2 r1557, %60, r1554;
}
{
add.f16x2 r1560, %80, %90;
}
{
mul.f16x2 r1563, r1560, r1307;
}
{
add.f16x2 r1566, r1557, r1563;
}
{
sub.f16x2 r1569, %69, %99;
}
{
mul.f16x2 r1572, r1569, r1306;
}
{
sub.f16x2 r1575, %79, %89;
}
{
mul.f16x2 r1578, r1575, r1309;
}
{
add.f16x2 r1581, r1572, r1578;
}
{
add.f16x2 r1584, r1566, r1581;
}
{
add.f16x2 r1587, %70, %100;
}
{
mul.f16x2 r1590, r1587, r1305;
}
{
add.f16x2 r1593, %60, r1590;
}
{
add.f16x2 r1596, %80, %90;
}
{
mul.f16x2 r1599, r1596, r1307;
}
{
add.f16x2 r1602, r1593, r1599;
}
{
sub.f16x2 r1605, %69, %99;
}
{
mul.f16x2 r1608, r1605, r1306;
}
{
sub.f16x2 r1611, %79, %89;
}
{
mul.f16x2 r1614, r1611, r1309;
}
{
add.f16x2 r1617, r1608, r1614;
}
{
sub.f16x2 r1620, r1602, r1617;
}
mov.f32 f62, 0f3F77F511;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f62;
cvt.rn.f16.f32 high, f62;
mov.b32 r1623, {low, high};
}
mov.f32 f64, 0fBE7EA890;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f64;
cvt.rn.f16.f32 high, f64;
mov.b32 r1624, {low, high};
}
mov.f32 f66, 0f3F6055A2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f66;
cvt.rn.f16.f32 high, f66;
mov.b32 r1625, {low, high};
}
mov.f32 f68, 0fBEF6A86B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f68;
cvt.rn.f16.f32 high, f68;
mov.b32 r1626, {low, high};
}
mov.f32 f70, 0f3F3A9DB0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f70;
cvt.rn.f16.f32 high, f70;
mov.b32 r1627, {low, high};
}
mov.f32 f72, 0fBF2F3E7B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f72;
cvt.rn.f16.f32 high, f72;
mov.b32 r1628, {low, high};
}
mov.f32 f74, 0f3F092BF2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r1629, {low, high};
}
mov.f32 f76, 0fBF5825E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r1630, {low, high};
}
mov.f32 f82, 0f3D809851;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r1633, {low, high};
}
mov.f32 f84, 0fBF7F7EAE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r1634, {low, high};
}
mov.f32 f90, 0fBED9FFBE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r1637, {low, high};
}
mov.f32 f92, 0fBF67A2BF;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r1638, {low, high};
}
mov.f32 f122, 0fBF232E38;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r1639, {low, high};
}
mov.f32 f96, 0fBF45405B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r1640, {low, high};
}
mov.f32 f106, 0fBF7DFB3B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1645, {low, high};
}
mov.f32 f108, 0fBE00575B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1646, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r1653, {low, high};
}
mov.f32 f124, 0f3F45405B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r1654, {low, high};
}
{
mul.f16x2 r1671, r408, r1623;
}
{
mul.f16x2 r1674, r552, r1624;
}
{
sub.f16x2 r1677, r1671, r1674;
}
{
mul.f16x2 r1680, r408, r1624;
}
{
fma.rn.f16x2 r1683, r552, r1623, r1680;
}
{
mul.f16x2 r1687, r728, r1625;
}
{
mul.f16x2 r1690, r872, r1626;
}
{
sub.f16x2 r1693, r1687, r1690;
}
{
mul.f16x2 r1696, r728, r1626;
}
{
fma.rn.f16x2 r1699, r872, r1625, r1696;
}
{
mul.f16x2 r1703, r1048, r1627;
}
{
mul.f16x2 r1706, r1192, r1628;
}
{
sub.f16x2 r1709, r1703, r1706;
}
{
mul.f16x2 r1712, r1048, r1628;
}
{
fma.rn.f16x2 r1715, r1192, r1627, r1712;
}
{
mul.f16x2 r1719, r1368, r1629;
}
{
mul.f16x2 r1722, r1512, r1630;
}
{
sub.f16x2 r1725, r1719, r1722;
}
{
mul.f16x2 r1728, r1368, r1630;
}
{
fma.rn.f16x2 r1731, r1512, r1629, r1728;
}
{
mul.f16x2 r1735, r480, r1625;
}
{
mul.f16x2 r1738, r624, r1626;
}
{
sub.f16x2 r1741, r1735, r1738;
}
{
mul.f16x2 r1744, r480, r1626;
}
{
fma.rn.f16x2 r1747, r624, r1625, r1744;
}
{
mul.f16x2 r1751, r800, r1629;
}
{
mul.f16x2 r1754, r944, r1630;
}
{
sub.f16x2 r1757, r1751, r1754;
}
{
mul.f16x2 r1760, r800, r1630;
}
{
fma.rn.f16x2 r1763, r944, r1629, r1760;
}
{
mul.f16x2 r1767, r1120, r1633;
}
{
mul.f16x2 r1770, r1264, r1634;
}
{
sub.f16x2 r1773, r1767, r1770;
}
{
mul.f16x2 r1776, r1120, r1634;
}
{
fma.rn.f16x2 r1779, r1264, r1633, r1776;
}
{
mul.f16x2 r1783, r1440, r1637;
}
{
mul.f16x2 r1786, r1584, r1638;
}
{
sub.f16x2 r1789, r1783, r1786;
}
{
mul.f16x2 r1792, r1440, r1638;
}
{
fma.rn.f16x2 r1795, r1584, r1637, r1792;
}
{
mul.f16x2 r1799, r516, r1627;
}
{
mul.f16x2 r1802, r660, r1628;
}
{
sub.f16x2 r1805, r1799, r1802;
}
{
mul.f16x2 r1808, r516, r1628;
}
{
fma.rn.f16x2 r1811, r660, r1627, r1808;
}
{
mul.f16x2 r1815, r836, r1633;
}
{
mul.f16x2 r1818, r980, r1634;
}
{
sub.f16x2 r1821, r1815, r1818;
}
{
mul.f16x2 r1824, r836, r1634;
}
{
fma.rn.f16x2 r1827, r980, r1633, r1824;
}
{
mul.f16x2 r1831, r1156, r1639;
}
{
mul.f16x2 r1834, r1300, r1640;
}
{
sub.f16x2 r1837, r1831, r1834;
}
{
mul.f16x2 r1840, r1156, r1640;
}
{
fma.rn.f16x2 r1843, r1300, r1639, r1840;
}
{
mul.f16x2 r1847, r1476, r1645;
}
{
mul.f16x2 r1850, r1620, r1646;
}
{
sub.f16x2 r1853, r1847, r1850;
}
{
mul.f16x2 r1856, r1476, r1646;
}
{
fma.rn.f16x2 r1859, r1620, r1645, r1856;
}
{
mul.f16x2 r1863, r444, r1629;
}
{
mul.f16x2 r1866, r588, r1630;
}
{
sub.f16x2 r1869, r1863, r1866;
}
{
mul.f16x2 r1872, r444, r1630;
}
{
fma.rn.f16x2 r1875, r588, r1629, r1872;
}
{
mul.f16x2 r1879, r764, r1637;
}
{
mul.f16x2 r1882, r908, r1638;
}
{
sub.f16x2 r1885, r1879, r1882;
}
{
mul.f16x2 r1888, r764, r1638;
}
{
fma.rn.f16x2 r1891, r908, r1637, r1888;
}
{
mul.f16x2 r1895, r1084, r1645;
}
{
mul.f16x2 r1898, r1228, r1646;
}
{
sub.f16x2 r1901, r1895, r1898;
}
{
mul.f16x2 r1904, r1084, r1646;
}
{
fma.rn.f16x2 r1907, r1228, r1645, r1904;
}
{
mul.f16x2 r1911, r1404, r1653;
}
{
mul.f16x2 r1914, r1548, r1654;
}
{
sub.f16x2 r1917, r1911, r1914;
}
{
mul.f16x2 r1920, r1404, r1654;
}
{
fma.rn.f16x2 r1923, r1548, r1653, r1920;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1927, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1928, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r1929, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r1930, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1931, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1932, {low, high};
}
{
neg.f16x2 r1933, r1932;
}
{
add.f16x2 r1935, r360, r1320;
}
{
add.f16x2 r1938, r40, r1935;
}
{
add.f16x2 r1941, r680, r1000;
}
{
add.f16x2 r1944, r1938, r1941;
}
st.local.u32 [rd3], r1944;
{
add.f16x2 r1947, r372, r1332;
}
{
add.f16x2 r1950, r52, r1947;
}
{
add.f16x2 r1953, r692, r1012;
}
{
add.f16x2 r1956, r1950, r1953;
}
st.local.u32 [rd3+4], r1956;
{
add.f16x2 r1959, r360, r1320;
}
{
mul.f16x2 r1962, r1959, r1927;
}
{
add.f16x2 r1965, r40, r1962;
}
{
add.f16x2 r1968, r680, r1000;
}
{
mul.f16x2 r1971, r1968, r1929;
}
{
add.f16x2 r1974, r1965, r1971;
}
{
sub.f16x2 r1977, r372, r1332;
}
{
mul.f16x2 r1980, r1977, r1928;
}
{
sub.f16x2 r1983, r692, r1012;
}
{
mul.f16x2 r1986, r1983, r1930;
}
{
add.f16x2 r1989, r1980, r1986;
}
{
sub.f16x2 r1992, r1974, r1989;
}
st.local.u32 [rd3+40], r1992;
{
add.f16x2 r1995, r360, r1320;
}
{
mul.f16x2 r1998, r1995, r1927;
}
{
add.f16x2 r2001, r40, r1998;
}
{
add.f16x2 r2004, r680, r1000;
}
{
mul.f16x2 r2007, r2004, r1929;
}
{
add.f16x2 r2010, r2001, r2007;
}
{
sub.f16x2 r2013, r372, r1332;
}
{
mul.f16x2 r2016, r2013, r1928;
}
{
sub.f16x2 r2019, r692, r1012;
}
{
mul.f16x2 r2022, r2019, r1930;
}
{
add.f16x2 r2025, r2016, r2022;
}
{
add.f16x2 r2028, r2010, r2025;
}
st.local.u32 [rd3+160], r2028;
{
add.f16x2 r2031, r360, r1320;
}
{
mul.f16x2 r2034, r2031, r1929;
}
{
add.f16x2 r2037, r40, r2034;
}
{
add.f16x2 r2040, r680, r1000;
}
{
mul.f16x2 r2043, r2040, r1931;
}
{
add.f16x2 r2046, r2037, r2043;
}
{
sub.f16x2 r2049, r372, r1332;
}
{
mul.f16x2 r2052, r2049, r1930;
}
{
sub.f16x2 r2055, r692, r1012;
}
{
mul.f16x2 r2058, r2055, r1933;
}
{
add.f16x2 r2061, r2052, r2058;
}
{
sub.f16x2 r2064, r2046, r2061;
}
st.local.u32 [rd3+80], r2064;
{
add.f16x2 r2067, r360, r1320;
}
{
mul.f16x2 r2070, r2067, r1929;
}
{
add.f16x2 r2073, r40, r2070;
}
{
add.f16x2 r2076, r680, r1000;
}
{
mul.f16x2 r2079, r2076, r1931;
}
{
add.f16x2 r2082, r2073, r2079;
}
{
sub.f16x2 r2085, r372, r1332;
}
{
mul.f16x2 r2088, r2085, r1930;
}
{
sub.f16x2 r2091, r692, r1012;
}
{
mul.f16x2 r2094, r2091, r1933;
}
{
add.f16x2 r2097, r2088, r2094;
}
{
add.f16x2 r2100, r2082, r2097;
}
st.local.u32 [rd3+120], r2100;
{
add.f16x2 r2103, r372, r1332;
}
{
mul.f16x2 r2106, r2103, r1927;
}
{
add.f16x2 r2109, r52, r2106;
}
{
add.f16x2 r2112, r692, r1012;
}
{
mul.f16x2 r2115, r2112, r1929;
}
{
add.f16x2 r2118, r2109, r2115;
}
{
sub.f16x2 r2121, r360, r1320;
}
{
mul.f16x2 r2124, r2121, r1928;
}
{
sub.f16x2 r2127, r680, r1000;
}
{
mul.f16x2 r2130, r2127, r1930;
}
{
add.f16x2 r2133, r2124, r2130;
}
{
add.f16x2 r2136, r2118, r2133;
}
st.local.u32 [rd3+44], r2136;
{
add.f16x2 r2139, r372, r1332;
}
{
mul.f16x2 r2142, r2139, r1927;
}
{
add.f16x2 r2145, r52, r2142;
}
{
add.f16x2 r2148, r692, r1012;
}
{
mul.f16x2 r2151, r2148, r1929;
}
{
add.f16x2 r2154, r2145, r2151;
}
{
sub.f16x2 r2157, r360, r1320;
}
{
mul.f16x2 r2160, r2157, r1928;
}
{
sub.f16x2 r2163, r680, r1000;
}
{
mul.f16x2 r2166, r2163, r1930;
}
{
add.f16x2 r2169, r2160, r2166;
}
{
sub.f16x2 r2172, r2154, r2169;
}
st.local.u32 [rd3+164], r2172;
{
add.f16x2 r2175, r372, r1332;
}
{
mul.f16x2 r2178, r2175, r1929;
}
{
add.f16x2 r2181, r52, r2178;
}
{
add.f16x2 r2184, r692, r1012;
}
{
mul.f16x2 r2187, r2184, r1931;
}
{
add.f16x2 r2190, r2181, r2187;
}
{
sub.f16x2 r2193, r360, r1320;
}
{
mul.f16x2 r2196, r2193, r1930;
}
{
sub.f16x2 r2199, r680, r1000;
}
{
mul.f16x2 r2202, r2199, r1933;
}
{
add.f16x2 r2205, r2196, r2202;
}
{
add.f16x2 r2208, r2190, r2205;
}
st.local.u32 [rd3+84], r2208;
{
add.f16x2 r2211, r372, r1332;
}
{
mul.f16x2 r2214, r2211, r1929;
}
{
add.f16x2 r2217, r52, r2214;
}
{
add.f16x2 r2220, r692, r1012;
}
{
mul.f16x2 r2223, r2220, r1931;
}
{
add.f16x2 r2226, r2217, r2223;
}
{
sub.f16x2 r2229, r360, r1320;
}
{
mul.f16x2 r2232, r2229, r1930;
}
{
sub.f16x2 r2235, r680, r1000;
}
{
mul.f16x2 r2238, r2235, r1933;
}
{
add.f16x2 r2241, r2232, r2238;
}
{
sub.f16x2 r2244, r2226, r2241;
}
st.local.u32 [rd3+124], r2244;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2247, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2248, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r2249, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r2250, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2251, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2252, {low, high};
}
{
neg.f16x2 r2253, r2252;
}
{
add.f16x2 r2255, r1677, r1725;
}
{
add.f16x2 r2258, r88, r2255;
}
{
add.f16x2 r2261, r1693, r1709;
}
{
add.f16x2 r11348, r2258, r2261;
}
st.local.u32 [rd3+8], r11348;
{
add.f16x2 r2267, r1683, r1731;
}
{
add.f16x2 r2270, r232, r2267;
}
{
add.f16x2 r2273, r1699, r1715;
}
{
add.f16x2 r2276, r2270, r2273;
}
st.local.u32 [rd3+12], r2276;
{
add.f16x2 r2279, r1677, r1725;
}
{
mul.f16x2 r2282, r2279, r2247;
}
{
add.f16x2 r2285, r88, r2282;
}
{
add.f16x2 r2288, r1693, r1709;
}
{
mul.f16x2 r2291, r2288, r2249;
}
{
add.f16x2 r2294, r2285, r2291;
}
{
sub.f16x2 r2297, r1683, r1731;
}
{
mul.f16x2 r2300, r2297, r2248;
}
{
sub.f16x2 r2303, r1699, r1715;
}
{
mul.f16x2 r2306, r2303, r2250;
}
{
add.f16x2 r2309, r2300, r2306;
}
{
sub.f16x2 r2312, r2294, r2309;
}
st.local.u32 [rd3+48], r2312;
{
add.f16x2 r2315, r1677, r1725;
}
{
mul.f16x2 r2318, r2315, r2247;
}
{
add.f16x2 r2321, r88, r2318;
}
{
add.f16x2 r2324, r1693, r1709;
}
{
mul.f16x2 r2327, r2324, r2249;
}
{
add.f16x2 r2330, r2321, r2327;
}
{
sub.f16x2 r2333, r1683, r1731;
}
{
mul.f16x2 r2336, r2333, r2248;
}
{
sub.f16x2 r2339, r1699, r1715;
}
{
mul.f16x2 r2342, r2339, r2250;
}
{
add.f16x2 r2345, r2336, r2342;
}
{
add.f16x2 r2348, r2330, r2345;
}
st.local.u32 [rd3+168], r2348;
{
add.f16x2 r2351, r1677, r1725;
}
{
mul.f16x2 r2354, r2351, r2249;
}
{
add.f16x2 r2357, r88, r2354;
}
{
add.f16x2 r2360, r1693, r1709;
}
{
mul.f16x2 r2363, r2360, r2251;
}
{
add.f16x2 r2366, r2357, r2363;
}
{
sub.f16x2 r2369, r1683, r1731;
}
{
mul.f16x2 r2372, r2369, r2250;
}
{
sub.f16x2 r2375, r1699, r1715;
}
{
mul.f16x2 r2378, r2375, r2253;
}
{
add.f16x2 r2381, r2372, r2378;
}
{
sub.f16x2 r2384, r2366, r2381;
}
st.local.u32 [rd3+88], r2384;
{
add.f16x2 r2387, r1677, r1725;
}
{
mul.f16x2 r2390, r2387, r2249;
}
{
add.f16x2 r2393, r88, r2390;
}
{
add.f16x2 r2396, r1693, r1709;
}
{
mul.f16x2 r2399, r2396, r2251;
}
{
add.f16x2 r2402, r2393, r2399;
}
{
sub.f16x2 r2405, r1683, r1731;
}
{
mul.f16x2 r2408, r2405, r2250;
}
{
sub.f16x2 r2411, r1699, r1715;
}
{
mul.f16x2 r2414, r2411, r2253;
}
{
add.f16x2 r2417, r2408, r2414;
}
{
add.f16x2 r2420, r2402, r2417;
}
st.local.u32 [rd3+128], r2420;
{
add.f16x2 r2423, r1683, r1731;
}
{
mul.f16x2 r2426, r2423, r2247;
}
{
add.f16x2 r2429, r232, r2426;
}
{
add.f16x2 r2432, r1699, r1715;
}
{
mul.f16x2 r2435, r2432, r2249;
}
{
add.f16x2 r2438, r2429, r2435;
}
{
sub.f16x2 r2441, r1677, r1725;
}
{
mul.f16x2 r2444, r2441, r2248;
}
{
sub.f16x2 r2447, r1693, r1709;
}
{
mul.f16x2 r2450, r2447, r2250;
}
{
add.f16x2 r2453, r2444, r2450;
}
{
add.f16x2 r2456, r2438, r2453;
}
st.local.u32 [rd3+52], r2456;
{
add.f16x2 r2459, r1683, r1731;
}
{
mul.f16x2 r2462, r2459, r2247;
}
{
add.f16x2 r2465, r232, r2462;
}
{
add.f16x2 r2468, r1699, r1715;
}
{
mul.f16x2 r2471, r2468, r2249;
}
{
add.f16x2 r2474, r2465, r2471;
}
{
sub.f16x2 r2477, r1677, r1725;
}
{
mul.f16x2 r2480, r2477, r2248;
}
{
sub.f16x2 r2483, r1693, r1709;
}
{
mul.f16x2 r2486, r2483, r2250;
}
{
add.f16x2 r2489, r2480, r2486;
}
{
sub.f16x2 r2492, r2474, r2489;
}
st.local.u32 [rd3+172], r2492;
{
add.f16x2 r2495, r1683, r1731;
}
{
mul.f16x2 r2498, r2495, r2249;
}
{
add.f16x2 r2501, r232, r2498;
}
{
add.f16x2 r2504, r1699, r1715;
}
{
mul.f16x2 r2507, r2504, r2251;
}
{
add.f16x2 r2510, r2501, r2507;
}
{
sub.f16x2 r2513, r1677, r1725;
}
{
mul.f16x2 r2516, r2513, r2250;
}
{
sub.f16x2 r2519, r1693, r1709;
}
{
mul.f16x2 r2522, r2519, r2253;
}
{
add.f16x2 r2525, r2516, r2522;
}
{
add.f16x2 r2528, r2510, r2525;
}
st.local.u32 [rd3+92], r2528;
{
add.f16x2 r2531, r1683, r1731;
}
{
mul.f16x2 r2534, r2531, r2249;
}
{
add.f16x2 r2537, r232, r2534;
}
{
add.f16x2 r2540, r1699, r1715;
}
{
mul.f16x2 r2543, r2540, r2251;
}
{
add.f16x2 r2546, r2537, r2543;
}
{
sub.f16x2 r2549, r1677, r1725;
}
{
mul.f16x2 r2552, r2549, r2250;
}
{
sub.f16x2 r2555, r1693, r1709;
}
{
mul.f16x2 r2558, r2555, r2253;
}
{
add.f16x2 r2561, r2552, r2558;
}
{
sub.f16x2 r2564, r2546, r2561;
}
st.local.u32 [rd3+132], r2564;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2567, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2568, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r2569, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r2570, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2571, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2572, {low, high};
}
{
neg.f16x2 r2573, r2572;
}
{
add.f16x2 r2575, r1741, r1789;
}
{
add.f16x2 r2578, r160, r2575;
}
{
add.f16x2 r2581, r1757, r1773;
}
{
add.f16x2 r2584, r2578, r2581;
}
st.local.u32 [rd3+16], r2584;
{
add.f16x2 r2587, r1747, r1795;
}
{
add.f16x2 r2590, r304, r2587;
}
{
add.f16x2 r2593, r1763, r1779;
}
{
add.f16x2 r2596, r2590, r2593;
}
st.local.u32 [rd3+20], r2596;
{
add.f16x2 r2599, r1741, r1789;
}
{
mul.f16x2 r2602, r2599, r2567;
}
{
add.f16x2 r2605, r160, r2602;
}
{
add.f16x2 r2608, r1757, r1773;
}
{
mul.f16x2 r2611, r2608, r2569;
}
{
add.f16x2 r2614, r2605, r2611;
}
{
sub.f16x2 r2617, r1747, r1795;
}
{
mul.f16x2 r2620, r2617, r2568;
}
{
sub.f16x2 r2623, r1763, r1779;
}
{
mul.f16x2 r2626, r2623, r2570;
}
{
add.f16x2 r2629, r2620, r2626;
}
{
sub.f16x2 r2632, r2614, r2629;
}
st.local.u32 [rd3+56], r2632;
{
add.f16x2 r2635, r1741, r1789;
}
{
mul.f16x2 r2638, r2635, r2567;
}
{
add.f16x2 r2641, r160, r2638;
}
{
add.f16x2 r2644, r1757, r1773;
}
{
mul.f16x2 r2647, r2644, r2569;
}
{
add.f16x2 r2650, r2641, r2647;
}
{
sub.f16x2 r2653, r1747, r1795;
}
{
mul.f16x2 r2656, r2653, r2568;
}
{
sub.f16x2 r2659, r1763, r1779;
}
{
mul.f16x2 r2662, r2659, r2570;
}
{
add.f16x2 r2665, r2656, r2662;
}
{
add.f16x2 r2668, r2650, r2665;
}
st.local.u32 [rd3+176], r2668;
{
add.f16x2 r2671, r1741, r1789;
}
{
mul.f16x2 r2674, r2671, r2569;
}
{
add.f16x2 r2677, r160, r2674;
}
{
add.f16x2 r2680, r1757, r1773;
}
{
mul.f16x2 r2683, r2680, r2571;
}
{
add.f16x2 r2686, r2677, r2683;
}
{
sub.f16x2 r2689, r1747, r1795;
}
{
mul.f16x2 r2692, r2689, r2570;
}
{
sub.f16x2 r2695, r1763, r1779;
}
{
mul.f16x2 r2698, r2695, r2573;
}
{
add.f16x2 r2701, r2692, r2698;
}
{
sub.f16x2 r2704, r2686, r2701;
}
st.local.u32 [rd3+96], r2704;
{
add.f16x2 r2707, r1741, r1789;
}
{
mul.f16x2 r2710, r2707, r2569;
}
{
add.f16x2 r2713, r160, r2710;
}
{
add.f16x2 r2716, r1757, r1773;
}
{
mul.f16x2 r2719, r2716, r2571;
}
{
add.f16x2 r2722, r2713, r2719;
}
{
sub.f16x2 r2725, r1747, r1795;
}
{
mul.f16x2 r2728, r2725, r2570;
}
{
sub.f16x2 r2731, r1763, r1779;
}
{
mul.f16x2 r2734, r2731, r2573;
}
{
add.f16x2 r2737, r2728, r2734;
}
{
add.f16x2 r2740, r2722, r2737;
}
st.local.u32 [rd3+136], r2740;
{
add.f16x2 r2743, r1747, r1795;
}
{
mul.f16x2 r2746, r2743, r2567;
}
{
add.f16x2 r2749, r304, r2746;
}
{
add.f16x2 r2752, r1763, r1779;
}
{
mul.f16x2 r2755, r2752, r2569;
}
{
add.f16x2 r2758, r2749, r2755;
}
{
sub.f16x2 r2761, r1741, r1789;
}
{
mul.f16x2 r2764, r2761, r2568;
}
{
sub.f16x2 r2767, r1757, r1773;
}
{
mul.f16x2 r2770, r2767, r2570;
}
{
add.f16x2 r2773, r2764, r2770;
}
{
add.f16x2 r2776, r2758, r2773;
}
st.local.u32 [rd3+60], r2776;
{
add.f16x2 r2779, r1747, r1795;
}
{
mul.f16x2 r2782, r2779, r2567;
}
{
add.f16x2 r2785, r304, r2782;
}
{
add.f16x2 r2788, r1763, r1779;
}
{
mul.f16x2 r2791, r2788, r2569;
}
{
add.f16x2 r2794, r2785, r2791;
}
{
sub.f16x2 r2797, r1741, r1789;
}
{
mul.f16x2 r2800, r2797, r2568;
}
{
sub.f16x2 r2803, r1757, r1773;
}
{
mul.f16x2 r2806, r2803, r2570;
}
{
add.f16x2 r2809, r2800, r2806;
}
{
sub.f16x2 r2812, r2794, r2809;
}
st.local.u32 [rd3+180], r2812;
{
add.f16x2 r2815, r1747, r1795;
}
{
mul.f16x2 r2818, r2815, r2569;
}
{
add.f16x2 r2821, r304, r2818;
}
{
add.f16x2 r2824, r1763, r1779;
}
{
mul.f16x2 r2827, r2824, r2571;
}
{
add.f16x2 r2830, r2821, r2827;
}
{
sub.f16x2 r2833, r1741, r1789;
}
{
mul.f16x2 r2836, r2833, r2570;
}
{
sub.f16x2 r2839, r1757, r1773;
}
{
mul.f16x2 r2842, r2839, r2573;
}
{
add.f16x2 r2845, r2836, r2842;
}
{
add.f16x2 r2848, r2830, r2845;
}
st.local.u32 [rd3+100], r2848;
{
add.f16x2 r2851, r1747, r1795;
}
{
mul.f16x2 r2854, r2851, r2569;
}
{
add.f16x2 r2857, r304, r2854;
}
{
add.f16x2 r2860, r1763, r1779;
}
{
mul.f16x2 r2863, r2860, r2571;
}
{
add.f16x2 r2866, r2857, r2863;
}
{
sub.f16x2 r2869, r1741, r1789;
}
{
mul.f16x2 r2872, r2869, r2570;
}
{
sub.f16x2 r2875, r1757, r1773;
}
{
mul.f16x2 r2878, r2875, r2573;
}
{
add.f16x2 r2881, r2872, r2878;
}
{
sub.f16x2 r2884, r2866, r2881;
}
st.local.u32 [rd3+140], r2884;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2887, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2888, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r2889, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r2890, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2891, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2892, {low, high};
}
{
neg.f16x2 r2893, r2892;
}
{
add.f16x2 r2895, r1805, r1853;
}
{
add.f16x2 r2898, r196, r2895;
}
{
add.f16x2 r2901, r1821, r1837;
}
{
add.f16x2 r2904, r2898, r2901;
}
st.local.u32 [rd3+24], r2904;
{
add.f16x2 r2907, r1811, r1859;
}
{
add.f16x2 r2910, r340, r2907;
}
{
add.f16x2 r2913, r1827, r1843;
}
{
add.f16x2 r2916, r2910, r2913;
}
st.local.u32 [rd3+28], r2916;
{
add.f16x2 r2919, r1805, r1853;
}
{
mul.f16x2 r2922, r2919, r2887;
}
{
add.f16x2 r2925, r196, r2922;
}
{
add.f16x2 r2928, r1821, r1837;
}
{
mul.f16x2 r2931, r2928, r2889;
}
{
add.f16x2 r2934, r2925, r2931;
}
{
sub.f16x2 r2937, r1811, r1859;
}
{
mul.f16x2 r2940, r2937, r2888;
}
{
sub.f16x2 r2943, r1827, r1843;
}
{
mul.f16x2 r2946, r2943, r2890;
}
{
add.f16x2 r2949, r2940, r2946;
}
{
sub.f16x2 r2952, r2934, r2949;
}
st.local.u32 [rd3+64], r2952;
{
add.f16x2 r2955, r1805, r1853;
}
{
mul.f16x2 r2958, r2955, r2887;
}
{
add.f16x2 r2961, r196, r2958;
}
{
add.f16x2 r2964, r1821, r1837;
}
{
mul.f16x2 r2967, r2964, r2889;
}
{
add.f16x2 r2970, r2961, r2967;
}
{
sub.f16x2 r2973, r1811, r1859;
}
{
mul.f16x2 r2976, r2973, r2888;
}
{
sub.f16x2 r2979, r1827, r1843;
}
{
mul.f16x2 r2982, r2979, r2890;
}
{
add.f16x2 r2985, r2976, r2982;
}
{
add.f16x2 r2988, r2970, r2985;
}
st.local.u32 [rd3+184], r2988;
{
add.f16x2 r2991, r1805, r1853;
}
{
mul.f16x2 r2994, r2991, r2889;
}
{
add.f16x2 r2997, r196, r2994;
}
{
add.f16x2 r3000, r1821, r1837;
}
{
mul.f16x2 r3003, r3000, r2891;
}
{
add.f16x2 r3006, r2997, r3003;
}
{
sub.f16x2 r3009, r1811, r1859;
}
{
mul.f16x2 r3012, r3009, r2890;
}
{
sub.f16x2 r3015, r1827, r1843;
}
{
mul.f16x2 r3018, r3015, r2893;
}
{
add.f16x2 r3021, r3012, r3018;
}
{
sub.f16x2 r3024, r3006, r3021;
}
st.local.u32 [rd3+104], r3024;
{
add.f16x2 r3027, r1805, r1853;
}
{
mul.f16x2 r3030, r3027, r2889;
}
{
add.f16x2 r3033, r196, r3030;
}
{
add.f16x2 r3036, r1821, r1837;
}
{
mul.f16x2 r3039, r3036, r2891;
}
{
add.f16x2 r3042, r3033, r3039;
}
{
sub.f16x2 r3045, r1811, r1859;
}
{
mul.f16x2 r3048, r3045, r2890;
}
{
sub.f16x2 r3051, r1827, r1843;
}
{
mul.f16x2 r3054, r3051, r2893;
}
{
add.f16x2 r3057, r3048, r3054;
}
{
add.f16x2 r3060, r3042, r3057;
}
st.local.u32 [rd3+144], r3060;
{
add.f16x2 r3063, r1811, r1859;
}
{
mul.f16x2 r3066, r3063, r2887;
}
{
add.f16x2 r3069, r340, r3066;
}
{
add.f16x2 r3072, r1827, r1843;
}
{
mul.f16x2 r3075, r3072, r2889;
}
{
add.f16x2 r3078, r3069, r3075;
}
{
sub.f16x2 r3081, r1805, r1853;
}
{
mul.f16x2 r3084, r3081, r2888;
}
{
sub.f16x2 r3087, r1821, r1837;
}
{
mul.f16x2 r3090, r3087, r2890;
}
{
add.f16x2 r3093, r3084, r3090;
}
{
add.f16x2 r3096, r3078, r3093;
}
st.local.u32 [rd3+68], r3096;
{
add.f16x2 r3099, r1811, r1859;
}
{
mul.f16x2 r3102, r3099, r2887;
}
{
add.f16x2 r3105, r340, r3102;
}
{
add.f16x2 r3108, r1827, r1843;
}
{
mul.f16x2 r3111, r3108, r2889;
}
{
add.f16x2 r3114, r3105, r3111;
}
{
sub.f16x2 r3117, r1805, r1853;
}
{
mul.f16x2 r3120, r3117, r2888;
}
{
sub.f16x2 r3123, r1821, r1837;
}
{
mul.f16x2 r3126, r3123, r2890;
}
{
add.f16x2 r3129, r3120, r3126;
}
{
sub.f16x2 r3132, r3114, r3129;
}
st.local.u32 [rd3+188], r3132;
{
add.f16x2 r3135, r1811, r1859;
}
{
mul.f16x2 r3138, r3135, r2889;
}
{
add.f16x2 r3141, r340, r3138;
}
{
add.f16x2 r3144, r1827, r1843;
}
{
mul.f16x2 r3147, r3144, r2891;
}
{
add.f16x2 r3150, r3141, r3147;
}
{
sub.f16x2 r3153, r1805, r1853;
}
{
mul.f16x2 r3156, r3153, r2890;
}
{
sub.f16x2 r3159, r1821, r1837;
}
{
mul.f16x2 r3162, r3159, r2893;
}
{
add.f16x2 r3165, r3156, r3162;
}
{
add.f16x2 r3168, r3150, r3165;
}
st.local.u32 [rd3+108], r3168;
{
add.f16x2 r3171, r1811, r1859;
}
{
mul.f16x2 r3174, r3171, r2889;
}
{
add.f16x2 r3177, r340, r3174;
}
{
add.f16x2 r3180, r1827, r1843;
}
{
mul.f16x2 r3183, r3180, r2891;
}
{
add.f16x2 r3186, r3177, r3183;
}
{
sub.f16x2 r3189, r1805, r1853;
}
{
mul.f16x2 r3192, r3189, r2890;
}
{
sub.f16x2 r3195, r1821, r1837;
}
{
mul.f16x2 r3198, r3195, r2893;
}
{
add.f16x2 r3201, r3192, r3198;
}
{
sub.f16x2 r3204, r3186, r3201;
}
st.local.u32 [rd3+148], r3204;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r3207, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r3208, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r3209, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r3210, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r3211, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r3212, {low, high};
}
{
neg.f16x2 r3213, r3212;
}
{
add.f16x2 r3215, r1869, r1917;
}
{
add.f16x2 r3218, r124, r3215;
}
{
add.f16x2 r3221, r1885, r1901;
}
{
add.f16x2 r3224, r3218, r3221;
}
st.local.u32 [rd3+32], r3224;
{
add.f16x2 r3227, r1875, r1923;
}
{
add.f16x2 r3230, r268, r3227;
}
{
add.f16x2 r3233, r1891, r1907;
}
{
add.f16x2 r3236, r3230, r3233;
}
st.local.u32 [rd3+36], r3236;
{
add.f16x2 r3239, r1869, r1917;
}
{
mul.f16x2 r3242, r3239, r3207;
}
{
add.f16x2 r3245, r124, r3242;
}
{
add.f16x2 r3248, r1885, r1901;
}
{
mul.f16x2 r3251, r3248, r3209;
}
{
add.f16x2 r3254, r3245, r3251;
}
{
sub.f16x2 r3257, r1875, r1923;
}
{
mul.f16x2 r3260, r3257, r3208;
}
{
sub.f16x2 r3263, r1891, r1907;
}
{
mul.f16x2 r3266, r3263, r3210;
}
{
add.f16x2 r3269, r3260, r3266;
}
{
sub.f16x2 r3272, r3254, r3269;
}
st.local.u32 [rd3+72], r3272;
{
add.f16x2 r3275, r1869, r1917;
}
{
mul.f16x2 r3278, r3275, r3207;
}
{
add.f16x2 r3281, r124, r3278;
}
{
add.f16x2 r3284, r1885, r1901;
}
{
mul.f16x2 r3287, r3284, r3209;
}
{
add.f16x2 r3290, r3281, r3287;
}
{
sub.f16x2 r3293, r1875, r1923;
}
{
mul.f16x2 r3296, r3293, r3208;
}
{
sub.f16x2 r3299, r1891, r1907;
}
{
mul.f16x2 r3302, r3299, r3210;
}
{
add.f16x2 r3305, r3296, r3302;
}
{
add.f16x2 r3308, r3290, r3305;
}
st.local.u32 [rd3+192], r3308;
{
add.f16x2 r3311, r1869, r1917;
}
{
mul.f16x2 r3314, r3311, r3209;
}
{
add.f16x2 r3317, r124, r3314;
}
{
add.f16x2 r3320, r1885, r1901;
}
{
mul.f16x2 r3323, r3320, r3211;
}
{
add.f16x2 r3326, r3317, r3323;
}
{
sub.f16x2 r3329, r1875, r1923;
}
{
mul.f16x2 r3332, r3329, r3210;
}
{
sub.f16x2 r3335, r1891, r1907;
}
{
mul.f16x2 r3338, r3335, r3213;
}
{
add.f16x2 r3341, r3332, r3338;
}
{
sub.f16x2 r3344, r3326, r3341;
}
st.local.u32 [rd3+112], r3344;
{
add.f16x2 r3347, r1869, r1917;
}
{
mul.f16x2 r3350, r3347, r3209;
}
{
add.f16x2 r3353, r124, r3350;
}
{
add.f16x2 r3356, r1885, r1901;
}
{
mul.f16x2 r3359, r3356, r3211;
}
{
add.f16x2 r3362, r3353, r3359;
}
{
sub.f16x2 r3365, r1875, r1923;
}
{
mul.f16x2 r3368, r3365, r3210;
}
{
sub.f16x2 r3371, r1891, r1907;
}
{
mul.f16x2 r3374, r3371, r3213;
}
{
add.f16x2 r3377, r3368, r3374;
}
{
add.f16x2 r3380, r3362, r3377;
}
st.local.u32 [rd3+152], r3380;
{
add.f16x2 r3383, r1875, r1923;
}
{
mul.f16x2 r3386, r3383, r3207;
}
{
add.f16x2 r3389, r268, r3386;
}
{
add.f16x2 r3392, r1891, r1907;
}
{
mul.f16x2 r3395, r3392, r3209;
}
{
add.f16x2 r3398, r3389, r3395;
}
{
sub.f16x2 r3401, r1869, r1917;
}
{
mul.f16x2 r3404, r3401, r3208;
}
{
sub.f16x2 r3407, r1885, r1901;
}
{
mul.f16x2 r3410, r3407, r3210;
}
{
add.f16x2 r3413, r3404, r3410;
}
{
add.f16x2 r3416, r3398, r3413;
}
st.local.u32 [rd3+76], r3416;
{
add.f16x2 r3419, r1875, r1923;
}
{
mul.f16x2 r3422, r3419, r3207;
}
{
add.f16x2 r3425, r268, r3422;
}
{
add.f16x2 r3428, r1891, r1907;
}
{
mul.f16x2 r3431, r3428, r3209;
}
{
add.f16x2 r3434, r3425, r3431;
}
{
sub.f16x2 r3437, r1869, r1917;
}
{
mul.f16x2 r3440, r3437, r3208;
}
{
sub.f16x2 r3443, r1885, r1901;
}
{
mul.f16x2 r3446, r3443, r3210;
}
{
add.f16x2 r3449, r3440, r3446;
}
{
sub.f16x2 r3452, r3434, r3449;
}
st.local.u32 [rd3+196], r3452;
{
add.f16x2 r3455, r1875, r1923;
}
{
mul.f16x2 r3458, r3455, r3209;
}
{
add.f16x2 r3461, r268, r3458;
}
{
add.f16x2 r3464, r1891, r1907;
}
{
mul.f16x2 r3467, r3464, r3211;
}
{
add.f16x2 r3470, r3461, r3467;
}
{
sub.f16x2 r3473, r1869, r1917;
}
{
mul.f16x2 r3476, r3473, r3210;
}
{
sub.f16x2 r3479, r1885, r1901;
}
{
mul.f16x2 r3482, r3479, r3213;
}
{
add.f16x2 r3485, r3476, r3482;
}
{
add.f16x2 r3488, r3470, r3485;
}
st.local.u32 [rd3+116], r3488;
{
add.f16x2 r3491, r1875, r1923;
}
{
mul.f16x2 r3494, r3491, r3209;
}
{
add.f16x2 r3497, r268, r3494;
}
{
add.f16x2 r3500, r1891, r1907;
}
{
mul.f16x2 r3503, r3500, r3211;
}
{
add.f16x2 r3506, r3497, r3503;
}
{
sub.f16x2 r3509, r1869, r1917;
}
{
mul.f16x2 r3512, r3509, r3210;
}
{
sub.f16x2 r3515, r1885, r1901;
}
{
mul.f16x2 r3518, r3515, r3213;
}
{
add.f16x2 r3521, r3512, r3518;
}
{
sub.f16x2 r3524, r3506, r3521;
}
st.local.u32 [rd3+156], r3524;
mov.u32 r3532, %tid.x;
mul.wide.u32 rd11, r3532, -776530087;
shr.u64 rd12, rd11, 41;
cvt.u32.u64 r4, rd12;
mul.lo.s32 r3533, r4, 625;
sub.s32 r3, r3532, r3533;
cvt.rn.f32.u32 f221, r3;
mul.f32 f222, f221, 0f39D2D427;
cos.approx.f32 f217, f222;
sin.approx.f32 f223, f222;
neg.f32 f218, f223;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f217;
cvt.rn.f16.f32 high, f218;
mov.b32 r11350, {low, high};
}
mov.u32 r11349, 1;
{
.reg .f16 low, high;
mov.b32 {low, high}, r11350;
mov.b32 r3554, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11350;
mov.b32 r3556, {high, high};
}
bra.uni LBB0_1;
LBB0_2:
ld.local.u32 r11348, [rd5+60];
LBB0_1:
{
.reg .f16 low, high;
mov.b32 {low, high}, r11350;
mov.b32 r3534, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11350;
mov.b32 r3536, {high, high};
}
mul.wide.u32 rd13, r11349, 8;
add.s64 rd14, rd3, rd13;
add.s64 rd5, rd14, 4;
ld.local.u32 r3539, [rd14+4];
{
mul.f16x2 r3538, r3539, r3536;
}
{
neg.f16x2 r3541, r3538;
}
{
fma.rn.f16x2 r3543, r11348, r3534, r3541;
}
st.local.u32 [rd14], r3543;
{
mul.f16x2 r3547, r11348, r3536;
}
{
fma.rn.f16x2 r3550, r3539, r3534, r3547;
}
st.local.u32 [rd14+4], r3550;
mov.f32 f238, 0fBF800000;
mov.f32 f239, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3558, {low, high};
}
{
mul.f16x2 r3559, r3556, r3558;
}
{
mul.f16x2 r3562, r11350, r3554;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11350;
mov.b32 r3565, {high, low};
}
{
fma.rn.f16x2 r3567, r3559, r3565, r3562;
}
ld.local.u32 r3585, [rd14+8];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3567;
mov.b32 r3571, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3567;
mov.b32 r3573, {high, high};
}
ld.local.u32 r3588, [rd14+12];
{
mul.f16x2 r3575, r3588, r3573;
}
{
neg.f16x2 r3578, r3575;
}
{
fma.rn.f16x2 r3580, r3585, r3571, r3578;
}
st.local.u32 [rd14+8], r3580;
{
mul.f16x2 r3584, r3585, r3573;
}
{
fma.rn.f16x2 r3587, r3588, r3571, r3584;
}
st.local.u32 [rd14+12], r3587;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3595, {low, high};
}
{
mul.f16x2 r3596, r3556, r3595;
}
{
mul.f16x2 r3599, r3567, r3554;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3567;
mov.b32 r3602, {high, low};
}
{
fma.rn.f16x2 r3604, r3596, r3602, r3599;
}
ld.local.u32 r3622, [rd14+16];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3604;
mov.b32 r3608, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3604;
mov.b32 r3610, {high, high};
}
ld.local.u32 r3625, [rd14+20];
{
mul.f16x2 r3612, r3625, r3610;
}
{
neg.f16x2 r3615, r3612;
}
{
fma.rn.f16x2 r3617, r3622, r3608, r3615;
}
st.local.u32 [rd14+16], r3617;
{
mul.f16x2 r3621, r3622, r3610;
}
{
fma.rn.f16x2 r3624, r3625, r3608, r3621;
}
st.local.u32 [rd14+20], r3624;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3632, {low, high};
}
{
mul.f16x2 r3633, r3556, r3632;
}
{
mul.f16x2 r3636, r3604, r3554;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3604;
mov.b32 r3639, {high, low};
}
{
fma.rn.f16x2 r3641, r3633, r3639, r3636;
}
ld.local.u32 r3659, [rd14+24];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3641;
mov.b32 r3645, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3641;
mov.b32 r3647, {high, high};
}
ld.local.u32 r3662, [rd14+28];
{
mul.f16x2 r3649, r3662, r3647;
}
{
neg.f16x2 r3652, r3649;
}
{
fma.rn.f16x2 r3654, r3659, r3645, r3652;
}
st.local.u32 [rd14+24], r3654;
{
mul.f16x2 r3658, r3659, r3647;
}
{
fma.rn.f16x2 r3661, r3662, r3645, r3658;
}
st.local.u32 [rd14+28], r3661;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3669, {low, high};
}
{
mul.f16x2 r3670, r3556, r3669;
}
{
mul.f16x2 r3673, r3641, r3554;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3641;
mov.b32 r3676, {high, low};
}
{
fma.rn.f16x2 r3678, r3670, r3676, r3673;
}
ld.local.u32 r3692, [rd14+32];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3678;
mov.b32 r3682, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3678;
mov.b32 r3684, {high, high};
}
ld.local.u32 r3699, [rd14+36];
{
mul.f16x2 r3686, r3699, r3684;
}
{
neg.f16x2 r3689, r3686;
}
{
fma.rn.f16x2 r3691, r3692, r3682, r3689;
}
st.local.u32 [rd14+32], r3691;
{
mul.f16x2 r3695, r3692, r3684;
}
{
fma.rn.f16x2 r3698, r3699, r3682, r3695;
}
st.local.u32 [rd14+36], r3698;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3706, {low, high};
}
{
mul.f16x2 r3707, r3556, r3706;
}
{
mul.f16x2 r3710, r3678, r3554;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3678;
mov.b32 r3713, {high, low};
}
{
fma.rn.f16x2 r3715, r3707, r3713, r3710;
}
ld.local.u32 r3729, [rd14+40];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3715;
mov.b32 r3719, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3715;
mov.b32 r3721, {high, high};
}
ld.local.u32 r3724, [rd14+44];
{
mul.f16x2 r3723, r3724, r3721;
}
{
neg.f16x2 r3726, r3723;
}
{
fma.rn.f16x2 r3728, r3729, r3719, r3726;
}
st.local.u32 [rd14+40], r3728;
{
mul.f16x2 r3732, r3729, r3721;
}
{
fma.rn.f16x2 r3735, r3724, r3719, r3732;
}
st.local.u32 [rd14+44], r3735;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3743, {low, high};
}
{
mul.f16x2 r3744, r3556, r3743;
}
{
mul.f16x2 r3747, r3715, r3554;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3715;
mov.b32 r3750, {high, low};
}
{
fma.rn.f16x2 r3752, r3744, r3750, r3747;
}
ld.local.u32 r3766, [rd14+48];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3752;
mov.b32 r3756, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3752;
mov.b32 r3758, {high, high};
}
ld.local.u32 r3761, [rd14+52];
{
mul.f16x2 r3760, r3761, r3758;
}
{
neg.f16x2 r3763, r3760;
}
{
fma.rn.f16x2 r3765, r3766, r3756, r3763;
}
st.local.u32 [rd14+48], r3765;
{
mul.f16x2 r3769, r3766, r3758;
}
{
fma.rn.f16x2 r3772, r3761, r3756, r3769;
}
st.local.u32 [rd14+52], r3772;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3780, {low, high};
}
{
mul.f16x2 r3781, r3556, r3780;
}
{
mul.f16x2 r3784, r3752, r3554;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3752;
mov.b32 r3787, {high, low};
}
{
fma.rn.f16x2 r3789, r3781, r3787, r3784;
}
ld.local.u32 r3803, [rd14+56];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3789;
mov.b32 r3793, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3789;
mov.b32 r3795, {high, high};
}
ld.local.u32 r3798, [rd14+60];
{
mul.f16x2 r3797, r3798, r3795;
}
{
neg.f16x2 r3800, r3797;
}
{
fma.rn.f16x2 r3802, r3803, r3793, r3800;
}
st.local.u32 [rd14+56], r3802;
{
mul.f16x2 r3806, r3803, r3795;
}
{
fma.rn.f16x2 r3809, r3798, r3793, r3806;
}
st.local.u32 [rd14+60], r3809;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3817, {low, high};
}
{
mul.f16x2 r3818, r3556, r3817;
}
{
mul.f16x2 r3821, r3789, r3554;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3789;
mov.b32 r3824, {high, low};
}
{
fma.rn.f16x2 r11350, r3818, r3824, r3821;
}
add.s32 r11349, r11349, 8;
setp.eq.s32 p1, r11349, 25;
@p1 bra LBB0_3;
bra.uni LBB0_2;
LBB0_3:
shl.b32 r7338, r1, 3;
mov.u32 r7339, %50;
add.s32 r7340, r7339, r7338;
mad.lo.s32 r12, r4, 125000, r7340;
barrier.sync 0;
mad.lo.s32 r7341, r3, 200, r12;
ld.local.v2.u32 {r7342, r7343}, [rd3];
st.shared.v2.f32 [r7341], {r7342, r7343};
ld.local.v2.u32 {r7346, r7347}, [rd4+4];
st.shared.v2.f32 [r7341+8], {r7346, r7347};
ld.local.v2.u32 {r7350, r7351}, [rd4+12];
st.shared.v2.f32 [r7341+16], {r7350, r7351};
ld.local.v2.u32 {r7354, r7355}, [rd4+20];
st.shared.v2.f32 [r7341+24], {r7354, r7355};
ld.local.v2.u32 {r7358, r7359}, [rd4+28];
st.shared.v2.f32 [r7341+32], {r7358, r7359};
ld.local.v2.u32 {r7362, r7363}, [rd4+36];
st.shared.v2.f32 [r7341+40], {r7362, r7363};
ld.local.v2.u32 {r7366, r7367}, [rd4+44];
st.shared.v2.f32 [r7341+48], {r7366, r7367};
ld.local.v2.u32 {r7370, r7371}, [rd4+52];
st.shared.v2.f32 [r7341+56], {r7370, r7371};
ld.local.v2.u32 {r7374, r7375}, [rd4+60];
st.shared.v2.f32 [r7341+64], {r7374, r7375};
ld.local.v2.u32 {r7378, r7379}, [rd4+68];
st.shared.v2.f32 [r7341+72], {r7378, r7379};
ld.local.v2.u32 {r7382, r7383}, [rd4+76];
st.shared.v2.f32 [r7341+80], {r7382, r7383};
ld.local.v2.u32 {r7386, r7387}, [rd4+84];
st.shared.v2.f32 [r7341+88], {r7386, r7387};
ld.local.v2.u32 {r7390, r7391}, [rd4+92];
st.shared.v2.f32 [r7341+96], {r7390, r7391};
ld.local.v2.u32 {r7394, r7395}, [rd4+100];
st.shared.v2.f32 [r7341+104], {r7394, r7395};
ld.local.v2.u32 {r7398, r7399}, [rd4+108];
st.shared.v2.f32 [r7341+112], {r7398, r7399};
ld.local.v2.u32 {r7402, r7403}, [rd4+116];
st.shared.v2.f32 [r7341+120], {r7402, r7403};
ld.local.v2.u32 {r7406, r7407}, [rd4+124];
st.shared.v2.f32 [r7341+128], {r7406, r7407};
ld.local.v2.u32 {r7410, r7411}, [rd4+132];
st.shared.v2.f32 [r7341+136], {r7410, r7411};
ld.local.v2.u32 {r7414, r7415}, [rd4+140];
st.shared.v2.f32 [r7341+144], {r7414, r7415};
ld.local.v2.u32 {r7418, r7419}, [rd4+148];
st.shared.v2.f32 [r7341+152], {r7418, r7419};
ld.local.v2.u32 {r7422, r7423}, [rd4+156];
st.shared.v2.f32 [r7341+160], {r7422, r7423};
ld.local.v2.u32 {r7426, r7427}, [rd4+164];
st.shared.v2.f32 [r7341+168], {r7426, r7427};
ld.local.v2.u32 {r7430, r7431}, [rd4+172];
st.shared.v2.f32 [r7341+176], {r7430, r7431};
ld.local.v2.u32 {r7434, r7435}, [rd4+180];
st.shared.v2.f32 [r7341+184], {r7434, r7435};
ld.local.v2.u32 {r7438, r7439}, [rd4+188];
st.shared.v2.f32 [r7341+192], {r7438, r7439};
barrier.sync 0;
mad.lo.s32 r13, r3, -192, r7341;
ld.shared.u32 r3842, [r13];
ld.shared.u32 r3854, [r13+4];
ld.shared.u32 r4162, [r13+5000];
ld.shared.u32 r4174, [r13+5004];
ld.shared.u32 r4482, [r13+10000];
ld.shared.u32 r4494, [r13+10004];
ld.shared.u32 r4802, [r13+15000];
ld.shared.u32 r4814, [r13+15004];
ld.shared.u32 r5122, [r13+20000];
ld.shared.u32 r5134, [r13+20004];
ld.shared.u32 r3839, [r13+25000];
ld.shared.u32 r3851, [r13+25004];
ld.shared.u32 r4159, [r13+30000];
ld.shared.u32 r4171, [r13+30004];
ld.shared.u32 r4479, [r13+35000];
ld.shared.u32 r4491, [r13+35004];
ld.shared.u32 r4799, [r13+40000];
ld.shared.u32 r4811, [r13+40004];
ld.shared.u32 r5119, [r13+45000];
ld.shared.u32 r5131, [r13+45004];
ld.shared.u32 r3845, [r13+50000];
ld.shared.u32 r3857, [r13+50004];
ld.shared.u32 r4165, [r13+55000];
ld.shared.u32 r4177, [r13+55004];
ld.shared.u32 r4485, [r13+60000];
ld.shared.u32 r4497, [r13+60004];
ld.shared.u32 r4805, [r13+65000];
ld.shared.u32 r4817, [r13+65004];
ld.shared.u32 r5125, [r13+70000];
ld.shared.u32 r5137, [r13+70004];
ld.shared.u32 r3846, [r13+75000];
ld.shared.u32 r3858, [r13+75004];
ld.shared.u32 r4166, [r13+80000];
ld.shared.u32 r4178, [r13+80004];
ld.shared.u32 r4486, [r13+85000];
ld.shared.u32 r4498, [r13+85004];
ld.shared.u32 r4806, [r13+90000];
ld.shared.u32 r4818, [r13+90004];
ld.shared.u32 r5126, [r13+95000];
ld.shared.u32 r5138, [r13+95004];
ld.shared.u32 r3840, [r13+100000];
ld.shared.u32 r3852, [r13+100004];
ld.shared.u32 r4160, [r13+105000];
ld.shared.u32 r4172, [r13+105004];
ld.shared.u32 r4480, [r13+110000];
ld.shared.u32 r4492, [r13+110004];
ld.shared.u32 r4800, [r13+115000];
ld.shared.u32 r4812, [r13+115004];
ld.shared.u32 r5120, [r13+120000];
ld.shared.u32 r5132, [r13+120004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r3830, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r3831, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r3832, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r3833, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r3834, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r3835, {low, high};
}
{
neg.f16x2 r3836, r3835;
}
{
add.f16x2 r3838, r3839, r3840;
}
{
add.f16x2 r3841, r3842, r3838;
}
{
add.f16x2 r3844, r3845, r3846;
}
{
add.f16x2 r3847, r3841, r3844;
}
{
add.f16x2 r3850, r3851, r3852;
}
{
add.f16x2 r3853, r3854, r3850;
}
{
add.f16x2 r3856, r3857, r3858;
}
{
add.f16x2 r3859, r3853, r3856;
}
{
add.f16x2 r3862, r3839, r3840;
}
{
mul.f16x2 r3865, r3862, r3830;
}
{
add.f16x2 r3868, r3842, r3865;
}
{
add.f16x2 r3871, r3845, r3846;
}
{
mul.f16x2 r3874, r3871, r3832;
}
{
add.f16x2 r3877, r3868, r3874;
}
{
sub.f16x2 r3880, r3851, r3852;
}
{
mul.f16x2 r3883, r3880, r3831;
}
{
sub.f16x2 r3886, r3857, r3858;
}
{
mul.f16x2 r3889, r3886, r3833;
}
{
add.f16x2 r3892, r3883, r3889;
}
{
sub.f16x2 r3895, r3877, r3892;
}
{
add.f16x2 r3898, r3839, r3840;
}
{
mul.f16x2 r3901, r3898, r3830;
}
{
add.f16x2 r3904, r3842, r3901;
}
{
add.f16x2 r3907, r3845, r3846;
}
{
mul.f16x2 r3910, r3907, r3832;
}
{
add.f16x2 r3913, r3904, r3910;
}
{
sub.f16x2 r3916, r3851, r3852;
}
{
mul.f16x2 r3919, r3916, r3831;
}
{
sub.f16x2 r3922, r3857, r3858;
}
{
mul.f16x2 r3925, r3922, r3833;
}
{
add.f16x2 r3928, r3919, r3925;
}
{
add.f16x2 r3931, r3913, r3928;
}
{
add.f16x2 r3934, r3839, r3840;
}
{
mul.f16x2 r3937, r3934, r3832;
}
{
add.f16x2 r3940, r3842, r3937;
}
{
add.f16x2 r3943, r3845, r3846;
}
{
mul.f16x2 r3946, r3943, r3834;
}
{
add.f16x2 r3949, r3940, r3946;
}
{
sub.f16x2 r3952, r3851, r3852;
}
{
mul.f16x2 r3955, r3952, r3833;
}
{
sub.f16x2 r3958, r3857, r3858;
}
{
mul.f16x2 r3961, r3958, r3836;
}
{
add.f16x2 r3964, r3955, r3961;
}
{
sub.f16x2 r3967, r3949, r3964;
}
{
add.f16x2 r3970, r3839, r3840;
}
{
mul.f16x2 r3973, r3970, r3832;
}
{
add.f16x2 r3976, r3842, r3973;
}
{
add.f16x2 r3979, r3845, r3846;
}
{
mul.f16x2 r3982, r3979, r3834;
}
{
add.f16x2 r3985, r3976, r3982;
}
{
sub.f16x2 r3988, r3851, r3852;
}
{
mul.f16x2 r3991, r3988, r3833;
}
{
sub.f16x2 r3994, r3857, r3858;
}
{
mul.f16x2 r3997, r3994, r3836;
}
{
add.f16x2 r4000, r3991, r3997;
}
{
add.f16x2 r4003, r3985, r4000;
}
{
add.f16x2 r4006, r3851, r3852;
}
{
mul.f16x2 r4009, r4006, r3830;
}
{
add.f16x2 r4012, r3854, r4009;
}
{
add.f16x2 r4015, r3857, r3858;
}
{
mul.f16x2 r4018, r4015, r3832;
}
{
add.f16x2 r4021, r4012, r4018;
}
{
sub.f16x2 r4024, r3839, r3840;
}
{
mul.f16x2 r4027, r4024, r3831;
}
{
sub.f16x2 r4030, r3845, r3846;
}
{
mul.f16x2 r4033, r4030, r3833;
}
{
add.f16x2 r4036, r4027, r4033;
}
{
add.f16x2 r4039, r4021, r4036;
}
{
add.f16x2 r4042, r3851, r3852;
}
{
mul.f16x2 r4045, r4042, r3830;
}
{
add.f16x2 r4048, r3854, r4045;
}
{
add.f16x2 r4051, r3857, r3858;
}
{
mul.f16x2 r4054, r4051, r3832;
}
{
add.f16x2 r4057, r4048, r4054;
}
{
sub.f16x2 r4060, r3839, r3840;
}
{
mul.f16x2 r4063, r4060, r3831;
}
{
sub.f16x2 r4066, r3845, r3846;
}
{
mul.f16x2 r4069, r4066, r3833;
}
{
add.f16x2 r4072, r4063, r4069;
}
{
sub.f16x2 r4075, r4057, r4072;
}
{
add.f16x2 r4078, r3851, r3852;
}
{
mul.f16x2 r4081, r4078, r3832;
}
{
add.f16x2 r4084, r3854, r4081;
}
{
add.f16x2 r4087, r3857, r3858;
}
{
mul.f16x2 r4090, r4087, r3834;
}
{
add.f16x2 r4093, r4084, r4090;
}
{
sub.f16x2 r4096, r3839, r3840;
}
{
mul.f16x2 r4099, r4096, r3833;
}
{
sub.f16x2 r4102, r3845, r3846;
}
{
mul.f16x2 r4105, r4102, r3836;
}
{
add.f16x2 r4108, r4099, r4105;
}
{
add.f16x2 r4111, r4093, r4108;
}
{
add.f16x2 r4114, r3851, r3852;
}
{
mul.f16x2 r4117, r4114, r3832;
}
{
add.f16x2 r4120, r3854, r4117;
}
{
add.f16x2 r4123, r3857, r3858;
}
{
mul.f16x2 r4126, r4123, r3834;
}
{
add.f16x2 r4129, r4120, r4126;
}
{
sub.f16x2 r4132, r3839, r3840;
}
{
mul.f16x2 r4135, r4132, r3833;
}
{
sub.f16x2 r4138, r3845, r3846;
}
{
mul.f16x2 r4141, r4138, r3836;
}
{
add.f16x2 r4144, r4135, r4141;
}
{
sub.f16x2 r4147, r4129, r4144;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4150, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4151, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r4152, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r4153, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4154, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4155, {low, high};
}
{
neg.f16x2 r4156, r4155;
}
{
add.f16x2 r4158, r4159, r4160;
}
{
add.f16x2 r4161, r4162, r4158;
}
{
add.f16x2 r4164, r4165, r4166;
}
{
add.f16x2 r4167, r4161, r4164;
}
{
add.f16x2 r4170, r4171, r4172;
}
{
add.f16x2 r4173, r4174, r4170;
}
{
add.f16x2 r4176, r4177, r4178;
}
{
add.f16x2 r4179, r4173, r4176;
}
{
add.f16x2 r4182, r4159, r4160;
}
{
mul.f16x2 r4185, r4182, r4150;
}
{
add.f16x2 r4188, r4162, r4185;
}
{
add.f16x2 r4191, r4165, r4166;
}
{
mul.f16x2 r4194, r4191, r4152;
}
{
add.f16x2 r4197, r4188, r4194;
}
{
sub.f16x2 r4200, r4171, r4172;
}
{
mul.f16x2 r4203, r4200, r4151;
}
{
sub.f16x2 r4206, r4177, r4178;
}
{
mul.f16x2 r4209, r4206, r4153;
}
{
add.f16x2 r4212, r4203, r4209;
}
{
sub.f16x2 r4215, r4197, r4212;
}
{
add.f16x2 r4218, r4159, r4160;
}
{
mul.f16x2 r4221, r4218, r4150;
}
{
add.f16x2 r4224, r4162, r4221;
}
{
add.f16x2 r4227, r4165, r4166;
}
{
mul.f16x2 r4230, r4227, r4152;
}
{
add.f16x2 r4233, r4224, r4230;
}
{
sub.f16x2 r4236, r4171, r4172;
}
{
mul.f16x2 r4239, r4236, r4151;
}
{
sub.f16x2 r4242, r4177, r4178;
}
{
mul.f16x2 r4245, r4242, r4153;
}
{
add.f16x2 r4248, r4239, r4245;
}
{
add.f16x2 r4251, r4233, r4248;
}
{
add.f16x2 r4254, r4159, r4160;
}
{
mul.f16x2 r4257, r4254, r4152;
}
{
add.f16x2 r4260, r4162, r4257;
}
{
add.f16x2 r4263, r4165, r4166;
}
{
mul.f16x2 r4266, r4263, r4154;
}
{
add.f16x2 r4269, r4260, r4266;
}
{
sub.f16x2 r4272, r4171, r4172;
}
{
mul.f16x2 r4275, r4272, r4153;
}
{
sub.f16x2 r4278, r4177, r4178;
}
{
mul.f16x2 r4281, r4278, r4156;
}
{
add.f16x2 r4284, r4275, r4281;
}
{
sub.f16x2 r4287, r4269, r4284;
}
{
add.f16x2 r4290, r4159, r4160;
}
{
mul.f16x2 r4293, r4290, r4152;
}
{
add.f16x2 r4296, r4162, r4293;
}
{
add.f16x2 r4299, r4165, r4166;
}
{
mul.f16x2 r4302, r4299, r4154;
}
{
add.f16x2 r4305, r4296, r4302;
}
{
sub.f16x2 r4308, r4171, r4172;
}
{
mul.f16x2 r4311, r4308, r4153;
}
{
sub.f16x2 r4314, r4177, r4178;
}
{
mul.f16x2 r4317, r4314, r4156;
}
{
add.f16x2 r4320, r4311, r4317;
}
{
add.f16x2 r4323, r4305, r4320;
}
{
add.f16x2 r4326, r4171, r4172;
}
{
mul.f16x2 r4329, r4326, r4150;
}
{
add.f16x2 r4332, r4174, r4329;
}
{
add.f16x2 r4335, r4177, r4178;
}
{
mul.f16x2 r4338, r4335, r4152;
}
{
add.f16x2 r4341, r4332, r4338;
}
{
sub.f16x2 r4344, r4159, r4160;
}
{
mul.f16x2 r4347, r4344, r4151;
}
{
sub.f16x2 r4350, r4165, r4166;
}
{
mul.f16x2 r4353, r4350, r4153;
}
{
add.f16x2 r4356, r4347, r4353;
}
{
add.f16x2 r4359, r4341, r4356;
}
{
add.f16x2 r4362, r4171, r4172;
}
{
mul.f16x2 r4365, r4362, r4150;
}
{
add.f16x2 r4368, r4174, r4365;
}
{
add.f16x2 r4371, r4177, r4178;
}
{
mul.f16x2 r4374, r4371, r4152;
}
{
add.f16x2 r4377, r4368, r4374;
}
{
sub.f16x2 r4380, r4159, r4160;
}
{
mul.f16x2 r4383, r4380, r4151;
}
{
sub.f16x2 r4386, r4165, r4166;
}
{
mul.f16x2 r4389, r4386, r4153;
}
{
add.f16x2 r4392, r4383, r4389;
}
{
sub.f16x2 r4395, r4377, r4392;
}
{
add.f16x2 r4398, r4171, r4172;
}
{
mul.f16x2 r4401, r4398, r4152;
}
{
add.f16x2 r4404, r4174, r4401;
}
{
add.f16x2 r4407, r4177, r4178;
}
{
mul.f16x2 r4410, r4407, r4154;
}
{
add.f16x2 r4413, r4404, r4410;
}
{
sub.f16x2 r4416, r4159, r4160;
}
{
mul.f16x2 r4419, r4416, r4153;
}
{
sub.f16x2 r4422, r4165, r4166;
}
{
mul.f16x2 r4425, r4422, r4156;
}
{
add.f16x2 r4428, r4419, r4425;
}
{
add.f16x2 r4431, r4413, r4428;
}
{
add.f16x2 r4434, r4171, r4172;
}
{
mul.f16x2 r4437, r4434, r4152;
}
{
add.f16x2 r4440, r4174, r4437;
}
{
add.f16x2 r4443, r4177, r4178;
}
{
mul.f16x2 r4446, r4443, r4154;
}
{
add.f16x2 r4449, r4440, r4446;
}
{
sub.f16x2 r4452, r4159, r4160;
}
{
mul.f16x2 r4455, r4452, r4153;
}
{
sub.f16x2 r4458, r4165, r4166;
}
{
mul.f16x2 r4461, r4458, r4156;
}
{
add.f16x2 r4464, r4455, r4461;
}
{
sub.f16x2 r4467, r4449, r4464;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4470, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4471, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r4472, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r4473, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4474, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4475, {low, high};
}
{
neg.f16x2 r4476, r4475;
}
{
add.f16x2 r4478, r4479, r4480;
}
{
add.f16x2 r4481, r4482, r4478;
}
{
add.f16x2 r4484, r4485, r4486;
}
{
add.f16x2 r4487, r4481, r4484;
}
{
add.f16x2 r4490, r4491, r4492;
}
{
add.f16x2 r4493, r4494, r4490;
}
{
add.f16x2 r4496, r4497, r4498;
}
{
add.f16x2 r4499, r4493, r4496;
}
{
add.f16x2 r4502, r4479, r4480;
}
{
mul.f16x2 r4505, r4502, r4470;
}
{
add.f16x2 r4508, r4482, r4505;
}
{
add.f16x2 r4511, r4485, r4486;
}
{
mul.f16x2 r4514, r4511, r4472;
}
{
add.f16x2 r4517, r4508, r4514;
}
{
sub.f16x2 r4520, r4491, r4492;
}
{
mul.f16x2 r4523, r4520, r4471;
}
{
sub.f16x2 r4526, r4497, r4498;
}
{
mul.f16x2 r4529, r4526, r4473;
}
{
add.f16x2 r4532, r4523, r4529;
}
{
sub.f16x2 r4535, r4517, r4532;
}
{
add.f16x2 r4538, r4479, r4480;
}
{
mul.f16x2 r4541, r4538, r4470;
}
{
add.f16x2 r4544, r4482, r4541;
}
{
add.f16x2 r4547, r4485, r4486;
}
{
mul.f16x2 r4550, r4547, r4472;
}
{
add.f16x2 r4553, r4544, r4550;
}
{
sub.f16x2 r4556, r4491, r4492;
}
{
mul.f16x2 r4559, r4556, r4471;
}
{
sub.f16x2 r4562, r4497, r4498;
}
{
mul.f16x2 r4565, r4562, r4473;
}
{
add.f16x2 r4568, r4559, r4565;
}
{
add.f16x2 r4571, r4553, r4568;
}
{
add.f16x2 r4574, r4479, r4480;
}
{
mul.f16x2 r4577, r4574, r4472;
}
{
add.f16x2 r4580, r4482, r4577;
}
{
add.f16x2 r4583, r4485, r4486;
}
{
mul.f16x2 r4586, r4583, r4474;
}
{
add.f16x2 r4589, r4580, r4586;
}
{
sub.f16x2 r4592, r4491, r4492;
}
{
mul.f16x2 r4595, r4592, r4473;
}
{
sub.f16x2 r4598, r4497, r4498;
}
{
mul.f16x2 r4601, r4598, r4476;
}
{
add.f16x2 r4604, r4595, r4601;
}
{
sub.f16x2 r4607, r4589, r4604;
}
{
add.f16x2 r4610, r4479, r4480;
}
{
mul.f16x2 r4613, r4610, r4472;
}
{
add.f16x2 r4616, r4482, r4613;
}
{
add.f16x2 r4619, r4485, r4486;
}
{
mul.f16x2 r4622, r4619, r4474;
}
{
add.f16x2 r4625, r4616, r4622;
}
{
sub.f16x2 r4628, r4491, r4492;
}
{
mul.f16x2 r4631, r4628, r4473;
}
{
sub.f16x2 r4634, r4497, r4498;
}
{
mul.f16x2 r4637, r4634, r4476;
}
{
add.f16x2 r4640, r4631, r4637;
}
{
add.f16x2 r4643, r4625, r4640;
}
{
add.f16x2 r4646, r4491, r4492;
}
{
mul.f16x2 r4649, r4646, r4470;
}
{
add.f16x2 r4652, r4494, r4649;
}
{
add.f16x2 r4655, r4497, r4498;
}
{
mul.f16x2 r4658, r4655, r4472;
}
{
add.f16x2 r4661, r4652, r4658;
}
{
sub.f16x2 r4664, r4479, r4480;
}
{
mul.f16x2 r4667, r4664, r4471;
}
{
sub.f16x2 r4670, r4485, r4486;
}
{
mul.f16x2 r4673, r4670, r4473;
}
{
add.f16x2 r4676, r4667, r4673;
}
{
add.f16x2 r4679, r4661, r4676;
}
{
add.f16x2 r4682, r4491, r4492;
}
{
mul.f16x2 r4685, r4682, r4470;
}
{
add.f16x2 r4688, r4494, r4685;
}
{
add.f16x2 r4691, r4497, r4498;
}
{
mul.f16x2 r4694, r4691, r4472;
}
{
add.f16x2 r4697, r4688, r4694;
}
{
sub.f16x2 r4700, r4479, r4480;
}
{
mul.f16x2 r4703, r4700, r4471;
}
{
sub.f16x2 r4706, r4485, r4486;
}
{
mul.f16x2 r4709, r4706, r4473;
}
{
add.f16x2 r4712, r4703, r4709;
}
{
sub.f16x2 r4715, r4697, r4712;
}
{
add.f16x2 r4718, r4491, r4492;
}
{
mul.f16x2 r4721, r4718, r4472;
}
{
add.f16x2 r4724, r4494, r4721;
}
{
add.f16x2 r4727, r4497, r4498;
}
{
mul.f16x2 r4730, r4727, r4474;
}
{
add.f16x2 r4733, r4724, r4730;
}
{
sub.f16x2 r4736, r4479, r4480;
}
{
mul.f16x2 r4739, r4736, r4473;
}
{
sub.f16x2 r4742, r4485, r4486;
}
{
mul.f16x2 r4745, r4742, r4476;
}
{
add.f16x2 r4748, r4739, r4745;
}
{
add.f16x2 r4751, r4733, r4748;
}
{
add.f16x2 r4754, r4491, r4492;
}
{
mul.f16x2 r4757, r4754, r4472;
}
{
add.f16x2 r4760, r4494, r4757;
}
{
add.f16x2 r4763, r4497, r4498;
}
{
mul.f16x2 r4766, r4763, r4474;
}
{
add.f16x2 r4769, r4760, r4766;
}
{
sub.f16x2 r4772, r4479, r4480;
}
{
mul.f16x2 r4775, r4772, r4473;
}
{
sub.f16x2 r4778, r4485, r4486;
}
{
mul.f16x2 r4781, r4778, r4476;
}
{
add.f16x2 r4784, r4775, r4781;
}
{
sub.f16x2 r4787, r4769, r4784;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4790, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4791, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r4792, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r4793, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4794, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4795, {low, high};
}
{
neg.f16x2 r4796, r4795;
}
{
add.f16x2 r4798, r4799, r4800;
}
{
add.f16x2 r4801, r4802, r4798;
}
{
add.f16x2 r4804, r4805, r4806;
}
{
add.f16x2 r4807, r4801, r4804;
}
{
add.f16x2 r4810, r4811, r4812;
}
{
add.f16x2 r4813, r4814, r4810;
}
{
add.f16x2 r4816, r4817, r4818;
}
{
add.f16x2 r4819, r4813, r4816;
}
{
add.f16x2 r4822, r4799, r4800;
}
{
mul.f16x2 r4825, r4822, r4790;
}
{
add.f16x2 r4828, r4802, r4825;
}
{
add.f16x2 r4831, r4805, r4806;
}
{
mul.f16x2 r4834, r4831, r4792;
}
{
add.f16x2 r4837, r4828, r4834;
}
{
sub.f16x2 r4840, r4811, r4812;
}
{
mul.f16x2 r4843, r4840, r4791;
}
{
sub.f16x2 r4846, r4817, r4818;
}
{
mul.f16x2 r4849, r4846, r4793;
}
{
add.f16x2 r4852, r4843, r4849;
}
{
sub.f16x2 r4855, r4837, r4852;
}
{
add.f16x2 r4858, r4799, r4800;
}
{
mul.f16x2 r4861, r4858, r4790;
}
{
add.f16x2 r4864, r4802, r4861;
}
{
add.f16x2 r4867, r4805, r4806;
}
{
mul.f16x2 r4870, r4867, r4792;
}
{
add.f16x2 r4873, r4864, r4870;
}
{
sub.f16x2 r4876, r4811, r4812;
}
{
mul.f16x2 r4879, r4876, r4791;
}
{
sub.f16x2 r4882, r4817, r4818;
}
{
mul.f16x2 r4885, r4882, r4793;
}
{
add.f16x2 r4888, r4879, r4885;
}
{
add.f16x2 r4891, r4873, r4888;
}
{
add.f16x2 r4894, r4799, r4800;
}
{
mul.f16x2 r4897, r4894, r4792;
}
{
add.f16x2 r4900, r4802, r4897;
}
{
add.f16x2 r4903, r4805, r4806;
}
{
mul.f16x2 r4906, r4903, r4794;
}
{
add.f16x2 r4909, r4900, r4906;
}
{
sub.f16x2 r4912, r4811, r4812;
}
{
mul.f16x2 r4915, r4912, r4793;
}
{
sub.f16x2 r4918, r4817, r4818;
}
{
mul.f16x2 r4921, r4918, r4796;
}
{
add.f16x2 r4924, r4915, r4921;
}
{
sub.f16x2 r4927, r4909, r4924;
}
{
add.f16x2 r4930, r4799, r4800;
}
{
mul.f16x2 r4933, r4930, r4792;
}
{
add.f16x2 r4936, r4802, r4933;
}
{
add.f16x2 r4939, r4805, r4806;
}
{
mul.f16x2 r4942, r4939, r4794;
}
{
add.f16x2 r4945, r4936, r4942;
}
{
sub.f16x2 r4948, r4811, r4812;
}
{
mul.f16x2 r4951, r4948, r4793;
}
{
sub.f16x2 r4954, r4817, r4818;
}
{
mul.f16x2 r4957, r4954, r4796;
}
{
add.f16x2 r4960, r4951, r4957;
}
{
add.f16x2 r4963, r4945, r4960;
}
{
add.f16x2 r4966, r4811, r4812;
}
{
mul.f16x2 r4969, r4966, r4790;
}
{
add.f16x2 r4972, r4814, r4969;
}
{
add.f16x2 r4975, r4817, r4818;
}
{
mul.f16x2 r4978, r4975, r4792;
}
{
add.f16x2 r4981, r4972, r4978;
}
{
sub.f16x2 r4984, r4799, r4800;
}
{
mul.f16x2 r4987, r4984, r4791;
}
{
sub.f16x2 r4990, r4805, r4806;
}
{
mul.f16x2 r4993, r4990, r4793;
}
{
add.f16x2 r4996, r4987, r4993;
}
{
add.f16x2 r4999, r4981, r4996;
}
{
add.f16x2 r5002, r4811, r4812;
}
{
mul.f16x2 r5005, r5002, r4790;
}
{
add.f16x2 r5008, r4814, r5005;
}
{
add.f16x2 r5011, r4817, r4818;
}
{
mul.f16x2 r5014, r5011, r4792;
}
{
add.f16x2 r5017, r5008, r5014;
}
{
sub.f16x2 r5020, r4799, r4800;
}
{
mul.f16x2 r5023, r5020, r4791;
}
{
sub.f16x2 r5026, r4805, r4806;
}
{
mul.f16x2 r5029, r5026, r4793;
}
{
add.f16x2 r5032, r5023, r5029;
}
{
sub.f16x2 r5035, r5017, r5032;
}
{
add.f16x2 r5038, r4811, r4812;
}
{
mul.f16x2 r5041, r5038, r4792;
}
{
add.f16x2 r5044, r4814, r5041;
}
{
add.f16x2 r5047, r4817, r4818;
}
{
mul.f16x2 r5050, r5047, r4794;
}
{
add.f16x2 r5053, r5044, r5050;
}
{
sub.f16x2 r5056, r4799, r4800;
}
{
mul.f16x2 r5059, r5056, r4793;
}
{
sub.f16x2 r5062, r4805, r4806;
}
{
mul.f16x2 r5065, r5062, r4796;
}
{
add.f16x2 r5068, r5059, r5065;
}
{
add.f16x2 r5071, r5053, r5068;
}
{
add.f16x2 r5074, r4811, r4812;
}
{
mul.f16x2 r5077, r5074, r4792;
}
{
add.f16x2 r5080, r4814, r5077;
}
{
add.f16x2 r5083, r4817, r4818;
}
{
mul.f16x2 r5086, r5083, r4794;
}
{
add.f16x2 r5089, r5080, r5086;
}
{
sub.f16x2 r5092, r4799, r4800;
}
{
mul.f16x2 r5095, r5092, r4793;
}
{
sub.f16x2 r5098, r4805, r4806;
}
{
mul.f16x2 r5101, r5098, r4796;
}
{
add.f16x2 r5104, r5095, r5101;
}
{
sub.f16x2 r5107, r5089, r5104;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r5110, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r5111, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r5112, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r5113, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r5114, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r5115, {low, high};
}
{
neg.f16x2 r5116, r5115;
}
{
add.f16x2 r5118, r5119, r5120;
}
{
add.f16x2 r5121, r5122, r5118;
}
{
add.f16x2 r5124, r5125, r5126;
}
{
add.f16x2 r5127, r5121, r5124;
}
{
add.f16x2 r5130, r5131, r5132;
}
{
add.f16x2 r5133, r5134, r5130;
}
{
add.f16x2 r5136, r5137, r5138;
}
{
add.f16x2 r5139, r5133, r5136;
}
{
add.f16x2 r5142, r5119, r5120;
}
{
mul.f16x2 r5145, r5142, r5110;
}
{
add.f16x2 r5148, r5122, r5145;
}
{
add.f16x2 r5151, r5125, r5126;
}
{
mul.f16x2 r5154, r5151, r5112;
}
{
add.f16x2 r5157, r5148, r5154;
}
{
sub.f16x2 r5160, r5131, r5132;
}
{
mul.f16x2 r5163, r5160, r5111;
}
{
sub.f16x2 r5166, r5137, r5138;
}
{
mul.f16x2 r5169, r5166, r5113;
}
{
add.f16x2 r5172, r5163, r5169;
}
{
sub.f16x2 r5175, r5157, r5172;
}
{
add.f16x2 r5178, r5119, r5120;
}
{
mul.f16x2 r5181, r5178, r5110;
}
{
add.f16x2 r5184, r5122, r5181;
}
{
add.f16x2 r5187, r5125, r5126;
}
{
mul.f16x2 r5190, r5187, r5112;
}
{
add.f16x2 r5193, r5184, r5190;
}
{
sub.f16x2 r5196, r5131, r5132;
}
{
mul.f16x2 r5199, r5196, r5111;
}
{
sub.f16x2 r5202, r5137, r5138;
}
{
mul.f16x2 r5205, r5202, r5113;
}
{
add.f16x2 r5208, r5199, r5205;
}
{
add.f16x2 r5211, r5193, r5208;
}
{
add.f16x2 r5214, r5119, r5120;
}
{
mul.f16x2 r5217, r5214, r5112;
}
{
add.f16x2 r5220, r5122, r5217;
}
{
add.f16x2 r5223, r5125, r5126;
}
{
mul.f16x2 r5226, r5223, r5114;
}
{
add.f16x2 r5229, r5220, r5226;
}
{
sub.f16x2 r5232, r5131, r5132;
}
{
mul.f16x2 r5235, r5232, r5113;
}
{
sub.f16x2 r5238, r5137, r5138;
}
{
mul.f16x2 r5241, r5238, r5116;
}
{
add.f16x2 r5244, r5235, r5241;
}
{
sub.f16x2 r5247, r5229, r5244;
}
{
add.f16x2 r5250, r5119, r5120;
}
{
mul.f16x2 r5253, r5250, r5112;
}
{
add.f16x2 r5256, r5122, r5253;
}
{
add.f16x2 r5259, r5125, r5126;
}
{
mul.f16x2 r5262, r5259, r5114;
}
{
add.f16x2 r5265, r5256, r5262;
}
{
sub.f16x2 r5268, r5131, r5132;
}
{
mul.f16x2 r5271, r5268, r5113;
}
{
sub.f16x2 r5274, r5137, r5138;
}
{
mul.f16x2 r5277, r5274, r5116;
}
{
add.f16x2 r5280, r5271, r5277;
}
{
add.f16x2 r5283, r5265, r5280;
}
{
add.f16x2 r5286, r5131, r5132;
}
{
mul.f16x2 r5289, r5286, r5110;
}
{
add.f16x2 r5292, r5134, r5289;
}
{
add.f16x2 r5295, r5137, r5138;
}
{
mul.f16x2 r5298, r5295, r5112;
}
{
add.f16x2 r5301, r5292, r5298;
}
{
sub.f16x2 r5304, r5119, r5120;
}
{
mul.f16x2 r5307, r5304, r5111;
}
{
sub.f16x2 r5310, r5125, r5126;
}
{
mul.f16x2 r5313, r5310, r5113;
}
{
add.f16x2 r5316, r5307, r5313;
}
{
add.f16x2 r5319, r5301, r5316;
}
{
add.f16x2 r5322, r5131, r5132;
}
{
mul.f16x2 r5325, r5322, r5110;
}
{
add.f16x2 r5328, r5134, r5325;
}
{
add.f16x2 r5331, r5137, r5138;
}
{
mul.f16x2 r5334, r5331, r5112;
}
{
add.f16x2 r5337, r5328, r5334;
}
{
sub.f16x2 r5340, r5119, r5120;
}
{
mul.f16x2 r5343, r5340, r5111;
}
{
sub.f16x2 r5346, r5125, r5126;
}
{
mul.f16x2 r5349, r5346, r5113;
}
{
add.f16x2 r5352, r5343, r5349;
}
{
sub.f16x2 r5355, r5337, r5352;
}
{
add.f16x2 r5358, r5131, r5132;
}
{
mul.f16x2 r5361, r5358, r5112;
}
{
add.f16x2 r5364, r5134, r5361;
}
{
add.f16x2 r5367, r5137, r5138;
}
{
mul.f16x2 r5370, r5367, r5114;
}
{
add.f16x2 r5373, r5364, r5370;
}
{
sub.f16x2 r5376, r5119, r5120;
}
{
mul.f16x2 r5379, r5376, r5113;
}
{
sub.f16x2 r5382, r5125, r5126;
}
{
mul.f16x2 r5385, r5382, r5116;
}
{
add.f16x2 r5388, r5379, r5385;
}
{
add.f16x2 r5391, r5373, r5388;
}
{
add.f16x2 r5394, r5131, r5132;
}
{
mul.f16x2 r5397, r5394, r5112;
}
{
add.f16x2 r5400, r5134, r5397;
}
{
add.f16x2 r5403, r5137, r5138;
}
{
mul.f16x2 r5406, r5403, r5114;
}
{
add.f16x2 r5409, r5400, r5406;
}
{
sub.f16x2 r5412, r5119, r5120;
}
{
mul.f16x2 r5415, r5412, r5113;
}
{
sub.f16x2 r5418, r5125, r5126;
}
{
mul.f16x2 r5421, r5418, r5116;
}
{
add.f16x2 r5424, r5415, r5421;
}
{
sub.f16x2 r5427, r5409, r5424;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f62;
cvt.rn.f16.f32 high, f62;
mov.b32 r5430, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f64;
cvt.rn.f16.f32 high, f64;
mov.b32 r5431, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f66;
cvt.rn.f16.f32 high, f66;
mov.b32 r5432, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f68;
cvt.rn.f16.f32 high, f68;
mov.b32 r5433, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f70;
cvt.rn.f16.f32 high, f70;
mov.b32 r5434, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f72;
cvt.rn.f16.f32 high, f72;
mov.b32 r5435, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r5436, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r5437, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r5440, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r5441, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r5444, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r5445, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r5446, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r5447, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r5452, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r5453, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r5460, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r5461, {low, high};
}
{
mul.f16x2 r5478, r4215, r5430;
}
{
mul.f16x2 r5481, r4359, r5431;
}
{
sub.f16x2 r5484, r5478, r5481;
}
{
mul.f16x2 r5487, r4215, r5431;
}
{
fma.rn.f16x2 r5490, r4359, r5430, r5487;
}
{
mul.f16x2 r5494, r4535, r5432;
}
{
mul.f16x2 r5497, r4679, r5433;
}
{
sub.f16x2 r5500, r5494, r5497;
}
{
mul.f16x2 r5503, r4535, r5433;
}
{
fma.rn.f16x2 r5506, r4679, r5432, r5503;
}
{
mul.f16x2 r5510, r4855, r5434;
}
{
mul.f16x2 r5513, r4999, r5435;
}
{
sub.f16x2 r5516, r5510, r5513;
}
{
mul.f16x2 r5519, r4855, r5435;
}
{
fma.rn.f16x2 r5522, r4999, r5434, r5519;
}
{
mul.f16x2 r5526, r5175, r5436;
}
{
mul.f16x2 r5529, r5319, r5437;
}
{
sub.f16x2 r5532, r5526, r5529;
}
{
mul.f16x2 r5535, r5175, r5437;
}
{
fma.rn.f16x2 r5538, r5319, r5436, r5535;
}
{
mul.f16x2 r5542, r4287, r5432;
}
{
mul.f16x2 r5545, r4431, r5433;
}
{
sub.f16x2 r5548, r5542, r5545;
}
{
mul.f16x2 r5551, r4287, r5433;
}
{
fma.rn.f16x2 r5554, r4431, r5432, r5551;
}
{
mul.f16x2 r5558, r4607, r5436;
}
{
mul.f16x2 r5561, r4751, r5437;
}
{
sub.f16x2 r5564, r5558, r5561;
}
{
mul.f16x2 r5567, r4607, r5437;
}
{
fma.rn.f16x2 r5570, r4751, r5436, r5567;
}
{
mul.f16x2 r5574, r4927, r5440;
}
{
mul.f16x2 r5577, r5071, r5441;
}
{
sub.f16x2 r5580, r5574, r5577;
}
{
mul.f16x2 r5583, r4927, r5441;
}
{
fma.rn.f16x2 r5586, r5071, r5440, r5583;
}
{
mul.f16x2 r5590, r5247, r5444;
}
{
mul.f16x2 r5593, r5391, r5445;
}
{
sub.f16x2 r5596, r5590, r5593;
}
{
mul.f16x2 r5599, r5247, r5445;
}
{
fma.rn.f16x2 r5602, r5391, r5444, r5599;
}
{
mul.f16x2 r5606, r4323, r5434;
}
{
mul.f16x2 r5609, r4467, r5435;
}
{
sub.f16x2 r5612, r5606, r5609;
}
{
mul.f16x2 r5615, r4323, r5435;
}
{
fma.rn.f16x2 r5618, r4467, r5434, r5615;
}
{
mul.f16x2 r5622, r4643, r5440;
}
{
mul.f16x2 r5625, r4787, r5441;
}
{
sub.f16x2 r5628, r5622, r5625;
}
{
mul.f16x2 r5631, r4643, r5441;
}
{
fma.rn.f16x2 r5634, r4787, r5440, r5631;
}
{
mul.f16x2 r5638, r4963, r5446;
}
{
mul.f16x2 r5641, r5107, r5447;
}
{
sub.f16x2 r5644, r5638, r5641;
}
{
mul.f16x2 r5647, r4963, r5447;
}
{
fma.rn.f16x2 r5650, r5107, r5446, r5647;
}
{
mul.f16x2 r5654, r5283, r5452;
}
{
mul.f16x2 r5657, r5427, r5453;
}
{
sub.f16x2 r5660, r5654, r5657;
}
{
mul.f16x2 r5663, r5283, r5453;
}
{
fma.rn.f16x2 r5666, r5427, r5452, r5663;
}
{
mul.f16x2 r5670, r4251, r5436;
}
{
mul.f16x2 r5673, r4395, r5437;
}
{
sub.f16x2 r5676, r5670, r5673;
}
{
mul.f16x2 r5679, r4251, r5437;
}
{
fma.rn.f16x2 r5682, r4395, r5436, r5679;
}
{
mul.f16x2 r5686, r4571, r5444;
}
{
mul.f16x2 r5689, r4715, r5445;
}
{
sub.f16x2 r5692, r5686, r5689;
}
{
mul.f16x2 r5695, r4571, r5445;
}
{
fma.rn.f16x2 r5698, r4715, r5444, r5695;
}
{
mul.f16x2 r5702, r4891, r5452;
}
{
mul.f16x2 r5705, r5035, r5453;
}
{
sub.f16x2 r5708, r5702, r5705;
}
{
mul.f16x2 r5711, r4891, r5453;
}
{
fma.rn.f16x2 r5714, r5035, r5452, r5711;
}
{
mul.f16x2 r5718, r5211, r5460;
}
{
mul.f16x2 r5721, r5355, r5461;
}
{
sub.f16x2 r5724, r5718, r5721;
}
{
mul.f16x2 r5727, r5211, r5461;
}
{
fma.rn.f16x2 r5730, r5355, r5460, r5727;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r5734, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r5735, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r5736, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r5737, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r5738, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r5739, {low, high};
}
{
neg.f16x2 r5740, r5739;
}
{
add.f16x2 r5742, r4167, r5127;
}
{
add.f16x2 r5745, r3847, r5742;
}
{
add.f16x2 r5748, r4487, r4807;
}
{
add.f16x2 r5751, r5745, r5748;
}
st.local.u32 [rd3], r5751;
{
add.f16x2 r5754, r4179, r5139;
}
{
add.f16x2 r5757, r3859, r5754;
}
{
add.f16x2 r5760, r4499, r4819;
}
{
add.f16x2 r5763, r5757, r5760;
}
st.local.u32 [rd4], r5763;
{
add.f16x2 r5766, r4167, r5127;
}
{
mul.f16x2 r5769, r5766, r5734;
}
{
add.f16x2 r5772, r3847, r5769;
}
{
add.f16x2 r5775, r4487, r4807;
}
{
mul.f16x2 r5778, r5775, r5736;
}
{
add.f16x2 r5781, r5772, r5778;
}
{
sub.f16x2 r5784, r4179, r5139;
}
{
mul.f16x2 r5787, r5784, r5735;
}
{
sub.f16x2 r5790, r4499, r4819;
}
{
mul.f16x2 r5793, r5790, r5737;
}
{
add.f16x2 r5796, r5787, r5793;
}
{
sub.f16x2 r5799, r5781, r5796;
}
st.local.u32 [rd4+36], r5799;
{
add.f16x2 r5802, r4167, r5127;
}
{
mul.f16x2 r5805, r5802, r5734;
}
{
add.f16x2 r5808, r3847, r5805;
}
{
add.f16x2 r5811, r4487, r4807;
}
{
mul.f16x2 r5814, r5811, r5736;
}
{
add.f16x2 r5817, r5808, r5814;
}
{
sub.f16x2 r5820, r4179, r5139;
}
{
mul.f16x2 r5823, r5820, r5735;
}
{
sub.f16x2 r5826, r4499, r4819;
}
{
mul.f16x2 r5829, r5826, r5737;
}
{
add.f16x2 r5832, r5823, r5829;
}
{
add.f16x2 r5835, r5817, r5832;
}
st.local.u32 [rd4+156], r5835;
{
add.f16x2 r5838, r4167, r5127;
}
{
mul.f16x2 r5841, r5838, r5736;
}
{
add.f16x2 r5844, r3847, r5841;
}
{
add.f16x2 r5847, r4487, r4807;
}
{
mul.f16x2 r5850, r5847, r5738;
}
{
add.f16x2 r5853, r5844, r5850;
}
{
sub.f16x2 r5856, r4179, r5139;
}
{
mul.f16x2 r5859, r5856, r5737;
}
{
sub.f16x2 r5862, r4499, r4819;
}
{
mul.f16x2 r5865, r5862, r5740;
}
{
add.f16x2 r5868, r5859, r5865;
}
{
sub.f16x2 r5871, r5853, r5868;
}
st.local.u32 [rd4+76], r5871;
{
add.f16x2 r5874, r4167, r5127;
}
{
mul.f16x2 r5877, r5874, r5736;
}
{
add.f16x2 r5880, r3847, r5877;
}
{
add.f16x2 r5883, r4487, r4807;
}
{
mul.f16x2 r5886, r5883, r5738;
}
{
add.f16x2 r5889, r5880, r5886;
}
{
sub.f16x2 r5892, r4179, r5139;
}
{
mul.f16x2 r5895, r5892, r5737;
}
{
sub.f16x2 r5898, r4499, r4819;
}
{
mul.f16x2 r5901, r5898, r5740;
}
{
add.f16x2 r5904, r5895, r5901;
}
{
add.f16x2 r5907, r5889, r5904;
}
st.local.u32 [rd4+116], r5907;
{
add.f16x2 r5910, r4179, r5139;
}
{
mul.f16x2 r5913, r5910, r5734;
}
{
add.f16x2 r5916, r3859, r5913;
}
{
add.f16x2 r5919, r4499, r4819;
}
{
mul.f16x2 r5922, r5919, r5736;
}
{
add.f16x2 r5925, r5916, r5922;
}
{
sub.f16x2 r5928, r4167, r5127;
}
{
mul.f16x2 r5931, r5928, r5735;
}
{
sub.f16x2 r5934, r4487, r4807;
}
{
mul.f16x2 r5937, r5934, r5737;
}
{
add.f16x2 r5940, r5931, r5937;
}
{
add.f16x2 r5943, r5925, r5940;
}
st.local.u32 [rd4+40], r5943;
{
add.f16x2 r5946, r4179, r5139;
}
{
mul.f16x2 r5949, r5946, r5734;
}
{
add.f16x2 r5952, r3859, r5949;
}
{
add.f16x2 r5955, r4499, r4819;
}
{
mul.f16x2 r5958, r5955, r5736;
}
{
add.f16x2 r5961, r5952, r5958;
}
{
sub.f16x2 r5964, r4167, r5127;
}
{
mul.f16x2 r5967, r5964, r5735;
}
{
sub.f16x2 r5970, r4487, r4807;
}
{
mul.f16x2 r5973, r5970, r5737;
}
{
add.f16x2 r5976, r5967, r5973;
}
{
sub.f16x2 r5979, r5961, r5976;
}
st.local.u32 [rd4+160], r5979;
{
add.f16x2 r5982, r4179, r5139;
}
{
mul.f16x2 r5985, r5982, r5736;
}
{
add.f16x2 r5988, r3859, r5985;
}
{
add.f16x2 r5991, r4499, r4819;
}
{
mul.f16x2 r5994, r5991, r5738;
}
{
add.f16x2 r5997, r5988, r5994;
}
{
sub.f16x2 r6000, r4167, r5127;
}
{
mul.f16x2 r6003, r6000, r5737;
}
{
sub.f16x2 r6006, r4487, r4807;
}
{
mul.f16x2 r6009, r6006, r5740;
}
{
add.f16x2 r6012, r6003, r6009;
}
{
add.f16x2 r6015, r5997, r6012;
}
st.local.u32 [rd4+80], r6015;
{
add.f16x2 r6018, r4179, r5139;
}
{
mul.f16x2 r6021, r6018, r5736;
}
{
add.f16x2 r6024, r3859, r6021;
}
{
add.f16x2 r6027, r4499, r4819;
}
{
mul.f16x2 r6030, r6027, r5738;
}
{
add.f16x2 r6033, r6024, r6030;
}
{
sub.f16x2 r6036, r4167, r5127;
}
{
mul.f16x2 r6039, r6036, r5737;
}
{
sub.f16x2 r6042, r4487, r4807;
}
{
mul.f16x2 r6045, r6042, r5740;
}
{
add.f16x2 r6048, r6039, r6045;
}
{
sub.f16x2 r6051, r6033, r6048;
}
st.local.u32 [rd4+120], r6051;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6054, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6055, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r6056, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r6057, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6058, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6059, {low, high};
}
{
neg.f16x2 r6060, r6059;
}
{
add.f16x2 r6062, r5484, r5532;
}
{
add.f16x2 r6065, r3895, r6062;
}
{
add.f16x2 r6068, r5500, r5516;
}
{
add.f16x2 r11352, r6065, r6068;
}
st.local.u32 [rd4+4], r11352;
{
add.f16x2 r6074, r5490, r5538;
}
{
add.f16x2 r6077, r4039, r6074;
}
{
add.f16x2 r6080, r5506, r5522;
}
{
add.f16x2 r6083, r6077, r6080;
}
st.local.u32 [rd4+8], r6083;
{
add.f16x2 r6086, r5484, r5532;
}
{
mul.f16x2 r6089, r6086, r6054;
}
{
add.f16x2 r6092, r3895, r6089;
}
{
add.f16x2 r6095, r5500, r5516;
}
{
mul.f16x2 r6098, r6095, r6056;
}
{
add.f16x2 r6101, r6092, r6098;
}
{
sub.f16x2 r6104, r5490, r5538;
}
{
mul.f16x2 r6107, r6104, r6055;
}
{
sub.f16x2 r6110, r5506, r5522;
}
{
mul.f16x2 r6113, r6110, r6057;
}
{
add.f16x2 r6116, r6107, r6113;
}
{
sub.f16x2 r6119, r6101, r6116;
}
st.local.u32 [rd4+44], r6119;
{
add.f16x2 r6122, r5484, r5532;
}
{
mul.f16x2 r6125, r6122, r6054;
}
{
add.f16x2 r6128, r3895, r6125;
}
{
add.f16x2 r6131, r5500, r5516;
}
{
mul.f16x2 r6134, r6131, r6056;
}
{
add.f16x2 r6137, r6128, r6134;
}
{
sub.f16x2 r6140, r5490, r5538;
}
{
mul.f16x2 r6143, r6140, r6055;
}
{
sub.f16x2 r6146, r5506, r5522;
}
{
mul.f16x2 r6149, r6146, r6057;
}
{
add.f16x2 r6152, r6143, r6149;
}
{
add.f16x2 r6155, r6137, r6152;
}
st.local.u32 [rd4+164], r6155;
{
add.f16x2 r6158, r5484, r5532;
}
{
mul.f16x2 r6161, r6158, r6056;
}
{
add.f16x2 r6164, r3895, r6161;
}
{
add.f16x2 r6167, r5500, r5516;
}
{
mul.f16x2 r6170, r6167, r6058;
}
{
add.f16x2 r6173, r6164, r6170;
}
{
sub.f16x2 r6176, r5490, r5538;
}
{
mul.f16x2 r6179, r6176, r6057;
}
{
sub.f16x2 r6182, r5506, r5522;
}
{
mul.f16x2 r6185, r6182, r6060;
}
{
add.f16x2 r6188, r6179, r6185;
}
{
sub.f16x2 r6191, r6173, r6188;
}
st.local.u32 [rd4+84], r6191;
{
add.f16x2 r6194, r5484, r5532;
}
{
mul.f16x2 r6197, r6194, r6056;
}
{
add.f16x2 r6200, r3895, r6197;
}
{
add.f16x2 r6203, r5500, r5516;
}
{
mul.f16x2 r6206, r6203, r6058;
}
{
add.f16x2 r6209, r6200, r6206;
}
{
sub.f16x2 r6212, r5490, r5538;
}
{
mul.f16x2 r6215, r6212, r6057;
}
{
sub.f16x2 r6218, r5506, r5522;
}
{
mul.f16x2 r6221, r6218, r6060;
}
{
add.f16x2 r6224, r6215, r6221;
}
{
add.f16x2 r6227, r6209, r6224;
}
st.local.u32 [rd4+124], r6227;
{
add.f16x2 r6230, r5490, r5538;
}
{
mul.f16x2 r6233, r6230, r6054;
}
{
add.f16x2 r6236, r4039, r6233;
}
{
add.f16x2 r6239, r5506, r5522;
}
{
mul.f16x2 r6242, r6239, r6056;
}
{
add.f16x2 r6245, r6236, r6242;
}
{
sub.f16x2 r6248, r5484, r5532;
}
{
mul.f16x2 r6251, r6248, r6055;
}
{
sub.f16x2 r6254, r5500, r5516;
}
{
mul.f16x2 r6257, r6254, r6057;
}
{
add.f16x2 r6260, r6251, r6257;
}
{
add.f16x2 r6263, r6245, r6260;
}
st.local.u32 [rd4+48], r6263;
{
add.f16x2 r6266, r5490, r5538;
}
{
mul.f16x2 r6269, r6266, r6054;
}
{
add.f16x2 r6272, r4039, r6269;
}
{
add.f16x2 r6275, r5506, r5522;
}
{
mul.f16x2 r6278, r6275, r6056;
}
{
add.f16x2 r6281, r6272, r6278;
}
{
sub.f16x2 r6284, r5484, r5532;
}
{
mul.f16x2 r6287, r6284, r6055;
}
{
sub.f16x2 r6290, r5500, r5516;
}
{
mul.f16x2 r6293, r6290, r6057;
}
{
add.f16x2 r6296, r6287, r6293;
}
{
sub.f16x2 r6299, r6281, r6296;
}
st.local.u32 [rd4+168], r6299;
{
add.f16x2 r6302, r5490, r5538;
}
{
mul.f16x2 r6305, r6302, r6056;
}
{
add.f16x2 r6308, r4039, r6305;
}
{
add.f16x2 r6311, r5506, r5522;
}
{
mul.f16x2 r6314, r6311, r6058;
}
{
add.f16x2 r6317, r6308, r6314;
}
{
sub.f16x2 r6320, r5484, r5532;
}
{
mul.f16x2 r6323, r6320, r6057;
}
{
sub.f16x2 r6326, r5500, r5516;
}
{
mul.f16x2 r6329, r6326, r6060;
}
{
add.f16x2 r6332, r6323, r6329;
}
{
add.f16x2 r6335, r6317, r6332;
}
st.local.u32 [rd4+88], r6335;
{
add.f16x2 r6338, r5490, r5538;
}
{
mul.f16x2 r6341, r6338, r6056;
}
{
add.f16x2 r6344, r4039, r6341;
}
{
add.f16x2 r6347, r5506, r5522;
}
{
mul.f16x2 r6350, r6347, r6058;
}
{
add.f16x2 r6353, r6344, r6350;
}
{
sub.f16x2 r6356, r5484, r5532;
}
{
mul.f16x2 r6359, r6356, r6057;
}
{
sub.f16x2 r6362, r5500, r5516;
}
{
mul.f16x2 r6365, r6362, r6060;
}
{
add.f16x2 r6368, r6359, r6365;
}
{
sub.f16x2 r6371, r6353, r6368;
}
st.local.u32 [rd4+128], r6371;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6374, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6375, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r6376, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r6377, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6378, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6379, {low, high};
}
{
neg.f16x2 r6380, r6379;
}
{
add.f16x2 r6382, r5548, r5596;
}
{
add.f16x2 r6385, r3967, r6382;
}
{
add.f16x2 r6388, r5564, r5580;
}
{
add.f16x2 r6391, r6385, r6388;
}
st.local.u32 [rd4+12], r6391;
{
add.f16x2 r6394, r5554, r5602;
}
{
add.f16x2 r6397, r4111, r6394;
}
{
add.f16x2 r6400, r5570, r5586;
}
{
add.f16x2 r6403, r6397, r6400;
}
st.local.u32 [rd4+16], r6403;
{
add.f16x2 r6406, r5548, r5596;
}
{
mul.f16x2 r6409, r6406, r6374;
}
{
add.f16x2 r6412, r3967, r6409;
}
{
add.f16x2 r6415, r5564, r5580;
}
{
mul.f16x2 r6418, r6415, r6376;
}
{
add.f16x2 r6421, r6412, r6418;
}
{
sub.f16x2 r6424, r5554, r5602;
}
{
mul.f16x2 r6427, r6424, r6375;
}
{
sub.f16x2 r6430, r5570, r5586;
}
{
mul.f16x2 r6433, r6430, r6377;
}
{
add.f16x2 r6436, r6427, r6433;
}
{
sub.f16x2 r6439, r6421, r6436;
}
st.local.u32 [rd4+52], r6439;
{
add.f16x2 r6442, r5548, r5596;
}
{
mul.f16x2 r6445, r6442, r6374;
}
{
add.f16x2 r6448, r3967, r6445;
}
{
add.f16x2 r6451, r5564, r5580;
}
{
mul.f16x2 r6454, r6451, r6376;
}
{
add.f16x2 r6457, r6448, r6454;
}
{
sub.f16x2 r6460, r5554, r5602;
}
{
mul.f16x2 r6463, r6460, r6375;
}
{
sub.f16x2 r6466, r5570, r5586;
}
{
mul.f16x2 r6469, r6466, r6377;
}
{
add.f16x2 r6472, r6463, r6469;
}
{
add.f16x2 r6475, r6457, r6472;
}
st.local.u32 [rd4+172], r6475;
{
add.f16x2 r6478, r5548, r5596;
}
{
mul.f16x2 r6481, r6478, r6376;
}
{
add.f16x2 r6484, r3967, r6481;
}
{
add.f16x2 r6487, r5564, r5580;
}
{
mul.f16x2 r6490, r6487, r6378;
}
{
add.f16x2 r6493, r6484, r6490;
}
{
sub.f16x2 r6496, r5554, r5602;
}
{
mul.f16x2 r6499, r6496, r6377;
}
{
sub.f16x2 r6502, r5570, r5586;
}
{
mul.f16x2 r6505, r6502, r6380;
}
{
add.f16x2 r6508, r6499, r6505;
}
{
sub.f16x2 r6511, r6493, r6508;
}
st.local.u32 [rd4+92], r6511;
{
add.f16x2 r6514, r5548, r5596;
}
{
mul.f16x2 r6517, r6514, r6376;
}
{
add.f16x2 r6520, r3967, r6517;
}
{
add.f16x2 r6523, r5564, r5580;
}
{
mul.f16x2 r6526, r6523, r6378;
}
{
add.f16x2 r6529, r6520, r6526;
}
{
sub.f16x2 r6532, r5554, r5602;
}
{
mul.f16x2 r6535, r6532, r6377;
}
{
sub.f16x2 r6538, r5570, r5586;
}
{
mul.f16x2 r6541, r6538, r6380;
}
{
add.f16x2 r6544, r6535, r6541;
}
{
add.f16x2 r6547, r6529, r6544;
}
st.local.u32 [rd4+132], r6547;
{
add.f16x2 r6550, r5554, r5602;
}
{
mul.f16x2 r6553, r6550, r6374;
}
{
add.f16x2 r6556, r4111, r6553;
}
{
add.f16x2 r6559, r5570, r5586;
}
{
mul.f16x2 r6562, r6559, r6376;
}
{
add.f16x2 r6565, r6556, r6562;
}
{
sub.f16x2 r6568, r5548, r5596;
}
{
mul.f16x2 r6571, r6568, r6375;
}
{
sub.f16x2 r6574, r5564, r5580;
}
{
mul.f16x2 r6577, r6574, r6377;
}
{
add.f16x2 r6580, r6571, r6577;
}
{
add.f16x2 r6583, r6565, r6580;
}
st.local.u32 [rd4+56], r6583;
{
add.f16x2 r6586, r5554, r5602;
}
{
mul.f16x2 r6589, r6586, r6374;
}
{
add.f16x2 r6592, r4111, r6589;
}
{
add.f16x2 r6595, r5570, r5586;
}
{
mul.f16x2 r6598, r6595, r6376;
}
{
add.f16x2 r6601, r6592, r6598;
}
{
sub.f16x2 r6604, r5548, r5596;
}
{
mul.f16x2 r6607, r6604, r6375;
}
{
sub.f16x2 r6610, r5564, r5580;
}
{
mul.f16x2 r6613, r6610, r6377;
}
{
add.f16x2 r6616, r6607, r6613;
}
{
sub.f16x2 r6619, r6601, r6616;
}
st.local.u32 [rd4+176], r6619;
{
add.f16x2 r6622, r5554, r5602;
}
{
mul.f16x2 r6625, r6622, r6376;
}
{
add.f16x2 r6628, r4111, r6625;
}
{
add.f16x2 r6631, r5570, r5586;
}
{
mul.f16x2 r6634, r6631, r6378;
}
{
add.f16x2 r6637, r6628, r6634;
}
{
sub.f16x2 r6640, r5548, r5596;
}
{
mul.f16x2 r6643, r6640, r6377;
}
{
sub.f16x2 r6646, r5564, r5580;
}
{
mul.f16x2 r6649, r6646, r6380;
}
{
add.f16x2 r6652, r6643, r6649;
}
{
add.f16x2 r6655, r6637, r6652;
}
st.local.u32 [rd4+96], r6655;
{
add.f16x2 r6658, r5554, r5602;
}
{
mul.f16x2 r6661, r6658, r6376;
}
{
add.f16x2 r6664, r4111, r6661;
}
{
add.f16x2 r6667, r5570, r5586;
}
{
mul.f16x2 r6670, r6667, r6378;
}
{
add.f16x2 r6673, r6664, r6670;
}
{
sub.f16x2 r6676, r5548, r5596;
}
{
mul.f16x2 r6679, r6676, r6377;
}
{
sub.f16x2 r6682, r5564, r5580;
}
{
mul.f16x2 r6685, r6682, r6380;
}
{
add.f16x2 r6688, r6679, r6685;
}
{
sub.f16x2 r6691, r6673, r6688;
}
st.local.u32 [rd4+136], r6691;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6694, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6695, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r6696, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r6697, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6698, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6699, {low, high};
}
{
neg.f16x2 r6700, r6699;
}
{
add.f16x2 r6702, r5612, r5660;
}
{
add.f16x2 r6705, r4003, r6702;
}
{
add.f16x2 r6708, r5628, r5644;
}
{
add.f16x2 r6711, r6705, r6708;
}
st.local.u32 [rd4+20], r6711;
{
add.f16x2 r6714, r5618, r5666;
}
{
add.f16x2 r6717, r4147, r6714;
}
{
add.f16x2 r6720, r5634, r5650;
}
{
add.f16x2 r6723, r6717, r6720;
}
st.local.u32 [rd4+24], r6723;
{
add.f16x2 r6726, r5612, r5660;
}
{
mul.f16x2 r6729, r6726, r6694;
}
{
add.f16x2 r6732, r4003, r6729;
}
{
add.f16x2 r6735, r5628, r5644;
}
{
mul.f16x2 r6738, r6735, r6696;
}
{
add.f16x2 r6741, r6732, r6738;
}
{
sub.f16x2 r6744, r5618, r5666;
}
{
mul.f16x2 r6747, r6744, r6695;
}
{
sub.f16x2 r6750, r5634, r5650;
}
{
mul.f16x2 r6753, r6750, r6697;
}
{
add.f16x2 r6756, r6747, r6753;
}
{
sub.f16x2 r6759, r6741, r6756;
}
st.local.u32 [rd4+60], r6759;
{
add.f16x2 r6762, r5612, r5660;
}
{
mul.f16x2 r6765, r6762, r6694;
}
{
add.f16x2 r6768, r4003, r6765;
}
{
add.f16x2 r6771, r5628, r5644;
}
{
mul.f16x2 r6774, r6771, r6696;
}
{
add.f16x2 r6777, r6768, r6774;
}
{
sub.f16x2 r6780, r5618, r5666;
}
{
mul.f16x2 r6783, r6780, r6695;
}
{
sub.f16x2 r6786, r5634, r5650;
}
{
mul.f16x2 r6789, r6786, r6697;
}
{
add.f16x2 r6792, r6783, r6789;
}
{
add.f16x2 r6795, r6777, r6792;
}
st.local.u32 [rd4+180], r6795;
{
add.f16x2 r6798, r5612, r5660;
}
{
mul.f16x2 r6801, r6798, r6696;
}
{
add.f16x2 r6804, r4003, r6801;
}
{
add.f16x2 r6807, r5628, r5644;
}
{
mul.f16x2 r6810, r6807, r6698;
}
{
add.f16x2 r6813, r6804, r6810;
}
{
sub.f16x2 r6816, r5618, r5666;
}
{
mul.f16x2 r6819, r6816, r6697;
}
{
sub.f16x2 r6822, r5634, r5650;
}
{
mul.f16x2 r6825, r6822, r6700;
}
{
add.f16x2 r6828, r6819, r6825;
}
{
sub.f16x2 r6831, r6813, r6828;
}
st.local.u32 [rd4+100], r6831;
{
add.f16x2 r6834, r5612, r5660;
}
{
mul.f16x2 r6837, r6834, r6696;
}
{
add.f16x2 r6840, r4003, r6837;
}
{
add.f16x2 r6843, r5628, r5644;
}
{
mul.f16x2 r6846, r6843, r6698;
}
{
add.f16x2 r6849, r6840, r6846;
}
{
sub.f16x2 r6852, r5618, r5666;
}
{
mul.f16x2 r6855, r6852, r6697;
}
{
sub.f16x2 r6858, r5634, r5650;
}
{
mul.f16x2 r6861, r6858, r6700;
}
{
add.f16x2 r6864, r6855, r6861;
}
{
add.f16x2 r6867, r6849, r6864;
}
st.local.u32 [rd4+140], r6867;
{
add.f16x2 r6870, r5618, r5666;
}
{
mul.f16x2 r6873, r6870, r6694;
}
{
add.f16x2 r6876, r4147, r6873;
}
{
add.f16x2 r6879, r5634, r5650;
}
{
mul.f16x2 r6882, r6879, r6696;
}
{
add.f16x2 r6885, r6876, r6882;
}
{
sub.f16x2 r6888, r5612, r5660;
}
{
mul.f16x2 r6891, r6888, r6695;
}
{
sub.f16x2 r6894, r5628, r5644;
}
{
mul.f16x2 r6897, r6894, r6697;
}
{
add.f16x2 r6900, r6891, r6897;
}
{
add.f16x2 r6903, r6885, r6900;
}
st.local.u32 [rd4+64], r6903;
{
add.f16x2 r6906, r5618, r5666;
}
{
mul.f16x2 r6909, r6906, r6694;
}
{
add.f16x2 r6912, r4147, r6909;
}
{
add.f16x2 r6915, r5634, r5650;
}
{
mul.f16x2 r6918, r6915, r6696;
}
{
add.f16x2 r6921, r6912, r6918;
}
{
sub.f16x2 r6924, r5612, r5660;
}
{
mul.f16x2 r6927, r6924, r6695;
}
{
sub.f16x2 r6930, r5628, r5644;
}
{
mul.f16x2 r6933, r6930, r6697;
}
{
add.f16x2 r6936, r6927, r6933;
}
{
sub.f16x2 r6939, r6921, r6936;
}
st.local.u32 [rd4+184], r6939;
{
add.f16x2 r6942, r5618, r5666;
}
{
mul.f16x2 r6945, r6942, r6696;
}
{
add.f16x2 r6948, r4147, r6945;
}
{
add.f16x2 r6951, r5634, r5650;
}
{
mul.f16x2 r6954, r6951, r6698;
}
{
add.f16x2 r6957, r6948, r6954;
}
{
sub.f16x2 r6960, r5612, r5660;
}
{
mul.f16x2 r6963, r6960, r6697;
}
{
sub.f16x2 r6966, r5628, r5644;
}
{
mul.f16x2 r6969, r6966, r6700;
}
{
add.f16x2 r6972, r6963, r6969;
}
{
add.f16x2 r6975, r6957, r6972;
}
st.local.u32 [rd4+104], r6975;
{
add.f16x2 r6978, r5618, r5666;
}
{
mul.f16x2 r6981, r6978, r6696;
}
{
add.f16x2 r6984, r4147, r6981;
}
{
add.f16x2 r6987, r5634, r5650;
}
{
mul.f16x2 r6990, r6987, r6698;
}
{
add.f16x2 r6993, r6984, r6990;
}
{
sub.f16x2 r6996, r5612, r5660;
}
{
mul.f16x2 r6999, r6996, r6697;
}
{
sub.f16x2 r7002, r5628, r5644;
}
{
mul.f16x2 r7005, r7002, r6700;
}
{
add.f16x2 r7008, r6999, r7005;
}
{
sub.f16x2 r7011, r6993, r7008;
}
st.local.u32 [rd4+144], r7011;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r7014, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r7015, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r7016, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r7017, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r7018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r7019, {low, high};
}
{
neg.f16x2 r7020, r7019;
}
{
add.f16x2 r7022, r5676, r5724;
}
{
add.f16x2 r7025, r3931, r7022;
}
{
add.f16x2 r7028, r5692, r5708;
}
{
add.f16x2 r7031, r7025, r7028;
}
st.local.u32 [rd4+28], r7031;
{
add.f16x2 r7034, r5682, r5730;
}
{
add.f16x2 r7037, r4075, r7034;
}
{
add.f16x2 r7040, r5698, r5714;
}
{
add.f16x2 r7043, r7037, r7040;
}
st.local.u32 [rd4+32], r7043;
{
add.f16x2 r7046, r5676, r5724;
}
{
mul.f16x2 r7049, r7046, r7014;
}
{
add.f16x2 r7052, r3931, r7049;
}
{
add.f16x2 r7055, r5692, r5708;
}
{
mul.f16x2 r7058, r7055, r7016;
}
{
add.f16x2 r7061, r7052, r7058;
}
{
sub.f16x2 r7064, r5682, r5730;
}
{
mul.f16x2 r7067, r7064, r7015;
}
{
sub.f16x2 r7070, r5698, r5714;
}
{
mul.f16x2 r7073, r7070, r7017;
}
{
add.f16x2 r7076, r7067, r7073;
}
{
sub.f16x2 r7079, r7061, r7076;
}
st.local.u32 [rd4+68], r7079;
{
add.f16x2 r7082, r5676, r5724;
}
{
mul.f16x2 r7085, r7082, r7014;
}
{
add.f16x2 r7088, r3931, r7085;
}
{
add.f16x2 r7091, r5692, r5708;
}
{
mul.f16x2 r7094, r7091, r7016;
}
{
add.f16x2 r7097, r7088, r7094;
}
{
sub.f16x2 r7100, r5682, r5730;
}
{
mul.f16x2 r7103, r7100, r7015;
}
{
sub.f16x2 r7106, r5698, r5714;
}
{
mul.f16x2 r7109, r7106, r7017;
}
{
add.f16x2 r7112, r7103, r7109;
}
{
add.f16x2 r7115, r7097, r7112;
}
st.local.u32 [rd4+188], r7115;
{
add.f16x2 r7118, r5676, r5724;
}
{
mul.f16x2 r7121, r7118, r7016;
}
{
add.f16x2 r7124, r3931, r7121;
}
{
add.f16x2 r7127, r5692, r5708;
}
{
mul.f16x2 r7130, r7127, r7018;
}
{
add.f16x2 r7133, r7124, r7130;
}
{
sub.f16x2 r7136, r5682, r5730;
}
{
mul.f16x2 r7139, r7136, r7017;
}
{
sub.f16x2 r7142, r5698, r5714;
}
{
mul.f16x2 r7145, r7142, r7020;
}
{
add.f16x2 r7148, r7139, r7145;
}
{
sub.f16x2 r7151, r7133, r7148;
}
st.local.u32 [rd4+108], r7151;
{
add.f16x2 r7154, r5676, r5724;
}
{
mul.f16x2 r7157, r7154, r7016;
}
{
add.f16x2 r7160, r3931, r7157;
}
{
add.f16x2 r7163, r5692, r5708;
}
{
mul.f16x2 r7166, r7163, r7018;
}
{
add.f16x2 r7169, r7160, r7166;
}
{
sub.f16x2 r7172, r5682, r5730;
}
{
mul.f16x2 r7175, r7172, r7017;
}
{
sub.f16x2 r7178, r5698, r5714;
}
{
mul.f16x2 r7181, r7178, r7020;
}
{
add.f16x2 r7184, r7175, r7181;
}
{
add.f16x2 r7187, r7169, r7184;
}
st.local.u32 [rd4+148], r7187;
{
add.f16x2 r7190, r5682, r5730;
}
{
mul.f16x2 r7193, r7190, r7014;
}
{
add.f16x2 r7196, r4075, r7193;
}
{
add.f16x2 r7199, r5698, r5714;
}
{
mul.f16x2 r7202, r7199, r7016;
}
{
add.f16x2 r7205, r7196, r7202;
}
{
sub.f16x2 r7208, r5676, r5724;
}
{
mul.f16x2 r7211, r7208, r7015;
}
{
sub.f16x2 r7214, r5692, r5708;
}
{
mul.f16x2 r7217, r7214, r7017;
}
{
add.f16x2 r7220, r7211, r7217;
}
{
add.f16x2 r7223, r7205, r7220;
}
st.local.u32 [rd4+72], r7223;
{
add.f16x2 r7226, r5682, r5730;
}
{
mul.f16x2 r7229, r7226, r7014;
}
{
add.f16x2 r7232, r4075, r7229;
}
{
add.f16x2 r7235, r5698, r5714;
}
{
mul.f16x2 r7238, r7235, r7016;
}
{
add.f16x2 r7241, r7232, r7238;
}
{
sub.f16x2 r7244, r5676, r5724;
}
{
mul.f16x2 r7247, r7244, r7015;
}
{
sub.f16x2 r7250, r5692, r5708;
}
{
mul.f16x2 r7253, r7250, r7017;
}
{
add.f16x2 r7256, r7247, r7253;
}
{
sub.f16x2 r7259, r7241, r7256;
}
st.local.u32 [rd4+192], r7259;
{
add.f16x2 r7262, r5682, r5730;
}
{
mul.f16x2 r7265, r7262, r7016;
}
{
add.f16x2 r7268, r4075, r7265;
}
{
add.f16x2 r7271, r5698, r5714;
}
{
mul.f16x2 r7274, r7271, r7018;
}
{
add.f16x2 r7277, r7268, r7274;
}
{
sub.f16x2 r7280, r5676, r5724;
}
{
mul.f16x2 r7283, r7280, r7017;
}
{
sub.f16x2 r7286, r5692, r5708;
}
{
mul.f16x2 r7289, r7286, r7020;
}
{
add.f16x2 r7292, r7283, r7289;
}
{
add.f16x2 r7295, r7277, r7292;
}
st.local.u32 [rd4+112], r7295;
{
add.f16x2 r7298, r5682, r5730;
}
{
mul.f16x2 r7301, r7298, r7016;
}
{
add.f16x2 r7304, r4075, r7301;
}
{
add.f16x2 r7307, r5698, r5714;
}
{
mul.f16x2 r7310, r7307, r7018;
}
{
add.f16x2 r7313, r7304, r7310;
}
{
sub.f16x2 r7316, r5676, r5724;
}
{
mul.f16x2 r7319, r7316, r7017;
}
{
sub.f16x2 r7322, r5692, r5708;
}
{
mul.f16x2 r7325, r7322, r7020;
}
{
add.f16x2 r7328, r7319, r7325;
}
{
sub.f16x2 r7331, r7313, r7328;
}
st.local.u32 [rd4+152], r7331;
mul.wide.u32 rd16, r3, 1374389535;
shr.u64 rd17, rd16, 35;
cvt.u32.u64 r15, rd17;
cvt.rn.f32.u32 f460, r15;
mul.f32 f461, f460, 0f3C24B5BE;
cos.approx.f32 f456, f461;
sin.approx.f32 f462, f461;
neg.f32 f457, f462;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f457;
mov.b32 r11353, {low, high};
}
mov.u32 r11351, -8;
{
.reg .f16 low, high;
mov.b32 {low, high}, r11353;
mov.b32 r7462, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11353;
mov.b32 r7464, {high, high};
}
mov.u64 rd20, rd3;
bra.uni LBB0_4;
LBB0_5:
ld.local.u32 r11352, [rd20+72];
add.s64 rd20, rd20, 64;
LBB0_4:
{
.reg .f16 low, high;
mov.b32 {low, high}, r11353;
mov.b32 r7442, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11353;
mov.b32 r7444, {high, high};
}
ld.local.u32 r7447, [rd20+12];
{
mul.f16x2 r7446, r7447, r7444;
}
{
neg.f16x2 r7449, r7446;
}
{
fma.rn.f16x2 r7451, r11352, r7442, r7449;
}
st.local.u32 [rd20+8], r7451;
{
mul.f16x2 r7455, r11352, r7444;
}
{
fma.rn.f16x2 r7458, r7447, r7442, r7455;
}
st.local.u32 [rd20+12], r7458;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7466, {low, high};
}
{
mul.f16x2 r7467, r7464, r7466;
}
{
mul.f16x2 r7470, r11353, r7462;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11353;
mov.b32 r7473, {high, low};
}
{
fma.rn.f16x2 r7475, r7467, r7473, r7470;
}
ld.local.u32 r7493, [rd20+16];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7475;
mov.b32 r7479, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7475;
mov.b32 r7481, {high, high};
}
ld.local.u32 r7496, [rd20+20];
{
mul.f16x2 r7483, r7496, r7481;
}
{
neg.f16x2 r7486, r7483;
}
{
fma.rn.f16x2 r7488, r7493, r7479, r7486;
}
st.local.u32 [rd20+16], r7488;
{
mul.f16x2 r7492, r7493, r7481;
}
{
fma.rn.f16x2 r7495, r7496, r7479, r7492;
}
st.local.u32 [rd20+20], r7495;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7503, {low, high};
}
{
mul.f16x2 r7504, r7464, r7503;
}
{
mul.f16x2 r7507, r7475, r7462;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7475;
mov.b32 r7510, {high, low};
}
{
fma.rn.f16x2 r7512, r7504, r7510, r7507;
}
ld.local.u32 r7530, [rd20+24];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7512;
mov.b32 r7516, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7512;
mov.b32 r7518, {high, high};
}
ld.local.u32 r7533, [rd20+28];
{
mul.f16x2 r7520, r7533, r7518;
}
{
neg.f16x2 r7523, r7520;
}
{
fma.rn.f16x2 r7525, r7530, r7516, r7523;
}
st.local.u32 [rd20+24], r7525;
{
mul.f16x2 r7529, r7530, r7518;
}
{
fma.rn.f16x2 r7532, r7533, r7516, r7529;
}
st.local.u32 [rd20+28], r7532;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7540, {low, high};
}
{
mul.f16x2 r7541, r7464, r7540;
}
{
mul.f16x2 r7544, r7512, r7462;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7512;
mov.b32 r7547, {high, low};
}
{
fma.rn.f16x2 r7549, r7541, r7547, r7544;
}
ld.local.u32 r7567, [rd20+32];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7549;
mov.b32 r7553, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7549;
mov.b32 r7555, {high, high};
}
ld.local.u32 r7570, [rd20+36];
{
mul.f16x2 r7557, r7570, r7555;
}
{
neg.f16x2 r7560, r7557;
}
{
fma.rn.f16x2 r7562, r7567, r7553, r7560;
}
st.local.u32 [rd20+32], r7562;
{
mul.f16x2 r7566, r7567, r7555;
}
{
fma.rn.f16x2 r7569, r7570, r7553, r7566;
}
st.local.u32 [rd20+36], r7569;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7577, {low, high};
}
{
mul.f16x2 r7578, r7464, r7577;
}
{
mul.f16x2 r7581, r7549, r7462;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7549;
mov.b32 r7584, {high, low};
}
{
fma.rn.f16x2 r7586, r7578, r7584, r7581;
}
ld.local.u32 r7600, [rd20+40];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7586;
mov.b32 r7590, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7586;
mov.b32 r7592, {high, high};
}
ld.local.u32 r7607, [rd20+44];
{
mul.f16x2 r7594, r7607, r7592;
}
{
neg.f16x2 r7597, r7594;
}
{
fma.rn.f16x2 r7599, r7600, r7590, r7597;
}
st.local.u32 [rd20+40], r7599;
{
mul.f16x2 r7603, r7600, r7592;
}
{
fma.rn.f16x2 r7606, r7607, r7590, r7603;
}
st.local.u32 [rd20+44], r7606;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7614, {low, high};
}
{
mul.f16x2 r7615, r7464, r7614;
}
{
mul.f16x2 r7618, r7586, r7462;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7586;
mov.b32 r7621, {high, low};
}
{
fma.rn.f16x2 r7623, r7615, r7621, r7618;
}
ld.local.u32 r7637, [rd20+48];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7623;
mov.b32 r7627, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7623;
mov.b32 r7629, {high, high};
}
ld.local.u32 r7632, [rd20+52];
{
mul.f16x2 r7631, r7632, r7629;
}
{
neg.f16x2 r7634, r7631;
}
{
fma.rn.f16x2 r7636, r7637, r7627, r7634;
}
st.local.u32 [rd20+48], r7636;
{
mul.f16x2 r7640, r7637, r7629;
}
{
fma.rn.f16x2 r7643, r7632, r7627, r7640;
}
st.local.u32 [rd20+52], r7643;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7651, {low, high};
}
{
mul.f16x2 r7652, r7464, r7651;
}
{
mul.f16x2 r7655, r7623, r7462;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7623;
mov.b32 r7658, {high, low};
}
{
fma.rn.f16x2 r7660, r7652, r7658, r7655;
}
ld.local.u32 r7674, [rd20+56];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7660;
mov.b32 r7664, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7660;
mov.b32 r7666, {high, high};
}
ld.local.u32 r7669, [rd20+60];
{
mul.f16x2 r7668, r7669, r7666;
}
{
neg.f16x2 r7671, r7668;
}
{
fma.rn.f16x2 r7673, r7674, r7664, r7671;
}
st.local.u32 [rd20+56], r7673;
{
mul.f16x2 r7677, r7674, r7666;
}
{
fma.rn.f16x2 r7680, r7669, r7664, r7677;
}
st.local.u32 [rd20+60], r7680;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7688, {low, high};
}
{
mul.f16x2 r7689, r7464, r7688;
}
{
mul.f16x2 r7692, r7660, r7462;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7660;
mov.b32 r7695, {high, low};
}
{
fma.rn.f16x2 r7697, r7689, r7695, r7692;
}
ld.local.u32 r7711, [rd20+64];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7697;
mov.b32 r7701, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7697;
mov.b32 r7703, {high, high};
}
ld.local.u32 r7706, [rd20+68];
{
mul.f16x2 r7705, r7706, r7703;
}
{
neg.f16x2 r7708, r7705;
}
{
fma.rn.f16x2 r7710, r7711, r7701, r7708;
}
st.local.u32 [rd20+64], r7710;
{
mul.f16x2 r7714, r7711, r7703;
}
{
fma.rn.f16x2 r7717, r7706, r7701, r7714;
}
st.local.u32 [rd20+68], r7717;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7725, {low, high};
}
{
mul.f16x2 r7726, r7464, r7725;
}
{
mul.f16x2 r7729, r7697, r7462;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7697;
mov.b32 r7732, {high, low};
}
{
fma.rn.f16x2 r11353, r7726, r7732, r7729;
}
add.s32 r11351, r11351, 8;
setp.eq.s32 p2, r11351, 16;
@p2 bra LBB0_6;
bra.uni LBB0_5;
LBB0_6:
mul.lo.s32 r11243, r15, 25;
sub.s32 r11244, r3, r11243;
shl.b32 r11245, r11244, 3;
add.s32 r11246, r12, r11245;
barrier.sync 0;
mad.lo.s32 r11247, r15, 5000, r11246;
ld.local.v2.u32 {r11248, r11249}, [rd3];
st.shared.u32 [r11247], r11248;
st.shared.u32 [r11247+4], r11249;
ld.local.v2.u32 {r11252, r11253}, [rd4+4];
st.shared.u32 [r11247+200], r11252;
st.shared.u32 [r11247+204], r11253;
ld.local.v2.u32 {r11256, r11257}, [rd4+12];
st.shared.u32 [r11247+400], r11256;
st.shared.u32 [r11247+404], r11257;
ld.local.v2.u32 {r11260, r11261}, [rd4+20];
st.shared.u32 [r11247+600], r11260;
st.shared.u32 [r11247+604], r11261;
ld.local.v2.u32 {r11264, r11265}, [rd4+28];
st.shared.u32 [r11247+800], r11264;
st.shared.u32 [r11247+804], r11265;
ld.local.v2.u32 {r11268, r11269}, [rd4+36];
st.shared.u32 [r11247+1000], r11268;
st.shared.u32 [r11247+1004], r11269;
ld.local.v2.u32 {r11272, r11273}, [rd4+44];
st.shared.u32 [r11247+1200], r11272;
st.shared.u32 [r11247+1204], r11273;
ld.local.v2.u32 {r11276, r11277}, [rd4+52];
st.shared.u32 [r11247+1400], r11276;
st.shared.u32 [r11247+1404], r11277;
ld.local.v2.u32 {r11280, r11281}, [rd4+60];
st.shared.u32 [r11247+1600], r11280;
st.shared.u32 [r11247+1604], r11281;
ld.local.v2.u32 {r11284, r11285}, [rd4+68];
st.shared.u32 [r11247+1800], r11284;
st.shared.u32 [r11247+1804], r11285;
ld.local.v2.u32 {r11288, r11289}, [rd4+76];
st.shared.u32 [r11247+2000], r11288;
st.shared.u32 [r11247+2004], r11289;
ld.local.v2.u32 {r11292, r11293}, [rd4+84];
st.shared.u32 [r11247+2200], r11292;
st.shared.u32 [r11247+2204], r11293;
ld.local.v2.u32 {r11296, r11297}, [rd4+92];
st.shared.u32 [r11247+2400], r11296;
st.shared.u32 [r11247+2404], r11297;
ld.local.v2.u32 {r11300, r11301}, [rd4+100];
st.shared.u32 [r11247+2600], r11300;
st.shared.u32 [r11247+2604], r11301;
ld.local.v2.u32 {r11304, r11305}, [rd4+108];
st.shared.u32 [r11247+2800], r11304;
st.shared.u32 [r11247+2804], r11305;
ld.local.v2.u32 {r11308, r11309}, [rd4+116];
st.shared.u32 [r11247+3000], r11308;
st.shared.u32 [r11247+3004], r11309;
ld.local.v2.u32 {r11312, r11313}, [rd4+124];
st.shared.u32 [r11247+3200], r11312;
st.shared.u32 [r11247+3204], r11313;
ld.local.v2.u32 {r11316, r11317}, [rd4+132];
st.shared.u32 [r11247+3400], r11316;
st.shared.u32 [r11247+3404], r11317;
ld.local.v2.u32 {r11320, r11321}, [rd4+140];
st.shared.u32 [r11247+3600], r11320;
st.shared.u32 [r11247+3604], r11321;
ld.local.v2.u32 {r11324, r11325}, [rd4+148];
st.shared.u32 [r11247+3800], r11324;
st.shared.u32 [r11247+3804], r11325;
ld.local.v2.u32 {r11328, r11329}, [rd4+156];
st.shared.u32 [r11247+4000], r11328;
st.shared.u32 [r11247+4004], r11329;
ld.local.v2.u32 {r11332, r11333}, [rd4+164];
st.shared.u32 [r11247+4200], r11332;
st.shared.u32 [r11247+4204], r11333;
ld.local.v2.u32 {r11336, r11337}, [rd4+172];
st.shared.u32 [r11247+4400], r11336;
st.shared.u32 [r11247+4404], r11337;
ld.local.v2.u32 {r11340, r11341}, [rd4+180];
st.shared.u32 [r11247+4600], r11340;
st.shared.u32 [r11247+4604], r11341;
ld.local.v2.u32 {r11344, r11345}, [rd4+188];
st.shared.u32 [r11247+4800], r11344;
st.shared.u32 [r11247+4804], r11345;
barrier.sync 0;
ld.shared.u32 r7750, [r13];
ld.shared.u32 r7762, [r13+4];
ld.shared.u32 r8070, [r13+5000];
ld.shared.u32 r8082, [r13+5004];
ld.shared.u32 r8390, [r13+10000];
ld.shared.u32 r8402, [r13+10004];
ld.shared.u32 r8710, [r13+15000];
ld.shared.u32 r8722, [r13+15004];
ld.shared.u32 r9030, [r13+20000];
ld.shared.u32 r9042, [r13+20004];
ld.shared.u32 r7747, [r13+25000];
ld.shared.u32 r7759, [r13+25004];
ld.shared.u32 r8067, [r13+30000];
ld.shared.u32 r8079, [r13+30004];
ld.shared.u32 r8387, [r13+35000];
ld.shared.u32 r8399, [r13+35004];
ld.shared.u32 r8707, [r13+40000];
ld.shared.u32 r8719, [r13+40004];
ld.shared.u32 r9027, [r13+45000];
ld.shared.u32 r9039, [r13+45004];
ld.shared.u32 r7753, [r13+50000];
ld.shared.u32 r7765, [r13+50004];
ld.shared.u32 r8073, [r13+55000];
ld.shared.u32 r8085, [r13+55004];
ld.shared.u32 r8393, [r13+60000];
ld.shared.u32 r8405, [r13+60004];
ld.shared.u32 r8713, [r13+65000];
ld.shared.u32 r8725, [r13+65004];
ld.shared.u32 r9033, [r13+70000];
ld.shared.u32 r9045, [r13+70004];
ld.shared.u32 r7754, [r13+75000];
ld.shared.u32 r7766, [r13+75004];
ld.shared.u32 r8074, [r13+80000];
ld.shared.u32 r8086, [r13+80004];
ld.shared.u32 r8394, [r13+85000];
ld.shared.u32 r8406, [r13+85004];
ld.shared.u32 r8714, [r13+90000];
ld.shared.u32 r8726, [r13+90004];
ld.shared.u32 r9034, [r13+95000];
ld.shared.u32 r9046, [r13+95004];
ld.shared.u32 r7748, [r13+100000];
ld.shared.u32 r7760, [r13+100004];
ld.shared.u32 r8068, [r13+105000];
ld.shared.u32 r8080, [r13+105004];
ld.shared.u32 r8388, [r13+110000];
ld.shared.u32 r8400, [r13+110004];
ld.shared.u32 r8708, [r13+115000];
ld.shared.u32 r8720, [r13+115004];
ld.shared.u32 r9028, [r13+120000];
ld.shared.u32 r9040, [r13+120004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r7738, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r7739, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r7740, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r7741, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r7742, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r7743, {low, high};
}
{
neg.f16x2 r7744, r7743;
}
{
add.f16x2 r7746, r7747, r7748;
}
{
add.f16x2 r7749, r7750, r7746;
}
{
add.f16x2 r7752, r7753, r7754;
}
{
add.f16x2 r7755, r7749, r7752;
}
{
add.f16x2 r7758, r7759, r7760;
}
{
add.f16x2 r7761, r7762, r7758;
}
{
add.f16x2 r7764, r7765, r7766;
}
{
add.f16x2 r7767, r7761, r7764;
}
{
add.f16x2 r7770, r7747, r7748;
}
{
mul.f16x2 r7773, r7770, r7738;
}
{
add.f16x2 r7776, r7750, r7773;
}
{
add.f16x2 r7779, r7753, r7754;
}
{
mul.f16x2 r7782, r7779, r7740;
}
{
add.f16x2 r7785, r7776, r7782;
}
{
sub.f16x2 r7788, r7759, r7760;
}
{
mul.f16x2 r7791, r7788, r7739;
}
{
sub.f16x2 r7794, r7765, r7766;
}
{
mul.f16x2 r7797, r7794, r7741;
}
{
add.f16x2 r7800, r7791, r7797;
}
{
sub.f16x2 r7803, r7785, r7800;
}
{
add.f16x2 r7806, r7747, r7748;
}
{
mul.f16x2 r7809, r7806, r7738;
}
{
add.f16x2 r7812, r7750, r7809;
}
{
add.f16x2 r7815, r7753, r7754;
}
{
mul.f16x2 r7818, r7815, r7740;
}
{
add.f16x2 r7821, r7812, r7818;
}
{
sub.f16x2 r7824, r7759, r7760;
}
{
mul.f16x2 r7827, r7824, r7739;
}
{
sub.f16x2 r7830, r7765, r7766;
}
{
mul.f16x2 r7833, r7830, r7741;
}
{
add.f16x2 r7836, r7827, r7833;
}
{
add.f16x2 r7839, r7821, r7836;
}
{
add.f16x2 r7842, r7747, r7748;
}
{
mul.f16x2 r7845, r7842, r7740;
}
{
add.f16x2 r7848, r7750, r7845;
}
{
add.f16x2 r7851, r7753, r7754;
}
{
mul.f16x2 r7854, r7851, r7742;
}
{
add.f16x2 r7857, r7848, r7854;
}
{
sub.f16x2 r7860, r7759, r7760;
}
{
mul.f16x2 r7863, r7860, r7741;
}
{
sub.f16x2 r7866, r7765, r7766;
}
{
mul.f16x2 r7869, r7866, r7744;
}
{
add.f16x2 r7872, r7863, r7869;
}
{
sub.f16x2 r7875, r7857, r7872;
}
{
add.f16x2 r7878, r7747, r7748;
}
{
mul.f16x2 r7881, r7878, r7740;
}
{
add.f16x2 r7884, r7750, r7881;
}
{
add.f16x2 r7887, r7753, r7754;
}
{
mul.f16x2 r7890, r7887, r7742;
}
{
add.f16x2 r7893, r7884, r7890;
}
{
sub.f16x2 r7896, r7759, r7760;
}
{
mul.f16x2 r7899, r7896, r7741;
}
{
sub.f16x2 r7902, r7765, r7766;
}
{
mul.f16x2 r7905, r7902, r7744;
}
{
add.f16x2 r7908, r7899, r7905;
}
{
add.f16x2 r7911, r7893, r7908;
}
{
add.f16x2 r7914, r7759, r7760;
}
{
mul.f16x2 r7917, r7914, r7738;
}
{
add.f16x2 r7920, r7762, r7917;
}
{
add.f16x2 r7923, r7765, r7766;
}
{
mul.f16x2 r7926, r7923, r7740;
}
{
add.f16x2 r7929, r7920, r7926;
}
{
sub.f16x2 r7932, r7747, r7748;
}
{
mul.f16x2 r7935, r7932, r7739;
}
{
sub.f16x2 r7938, r7753, r7754;
}
{
mul.f16x2 r7941, r7938, r7741;
}
{
add.f16x2 r7944, r7935, r7941;
}
{
add.f16x2 r7947, r7929, r7944;
}
{
add.f16x2 r7950, r7759, r7760;
}
{
mul.f16x2 r7953, r7950, r7738;
}
{
add.f16x2 r7956, r7762, r7953;
}
{
add.f16x2 r7959, r7765, r7766;
}
{
mul.f16x2 r7962, r7959, r7740;
}
{
add.f16x2 r7965, r7956, r7962;
}
{
sub.f16x2 r7968, r7747, r7748;
}
{
mul.f16x2 r7971, r7968, r7739;
}
{
sub.f16x2 r7974, r7753, r7754;
}
{
mul.f16x2 r7977, r7974, r7741;
}
{
add.f16x2 r7980, r7971, r7977;
}
{
sub.f16x2 r7983, r7965, r7980;
}
{
add.f16x2 r7986, r7759, r7760;
}
{
mul.f16x2 r7989, r7986, r7740;
}
{
add.f16x2 r7992, r7762, r7989;
}
{
add.f16x2 r7995, r7765, r7766;
}
{
mul.f16x2 r7998, r7995, r7742;
}
{
add.f16x2 r8001, r7992, r7998;
}
{
sub.f16x2 r8004, r7747, r7748;
}
{
mul.f16x2 r8007, r8004, r7741;
}
{
sub.f16x2 r8010, r7753, r7754;
}
{
mul.f16x2 r8013, r8010, r7744;
}
{
add.f16x2 r8016, r8007, r8013;
}
{
add.f16x2 r8019, r8001, r8016;
}
{
add.f16x2 r8022, r7759, r7760;
}
{
mul.f16x2 r8025, r8022, r7740;
}
{
add.f16x2 r8028, r7762, r8025;
}
{
add.f16x2 r8031, r7765, r7766;
}
{
mul.f16x2 r8034, r8031, r7742;
}
{
add.f16x2 r8037, r8028, r8034;
}
{
sub.f16x2 r8040, r7747, r7748;
}
{
mul.f16x2 r8043, r8040, r7741;
}
{
sub.f16x2 r8046, r7753, r7754;
}
{
mul.f16x2 r8049, r8046, r7744;
}
{
add.f16x2 r8052, r8043, r8049;
}
{
sub.f16x2 r8055, r8037, r8052;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8058, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8059, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r8060, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r8061, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8062, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8063, {low, high};
}
{
neg.f16x2 r8064, r8063;
}
{
add.f16x2 r8066, r8067, r8068;
}
{
add.f16x2 r8069, r8070, r8066;
}
{
add.f16x2 r8072, r8073, r8074;
}
{
add.f16x2 r8075, r8069, r8072;
}
{
add.f16x2 r8078, r8079, r8080;
}
{
add.f16x2 r8081, r8082, r8078;
}
{
add.f16x2 r8084, r8085, r8086;
}
{
add.f16x2 r8087, r8081, r8084;
}
{
add.f16x2 r8090, r8067, r8068;
}
{
mul.f16x2 r8093, r8090, r8058;
}
{
add.f16x2 r8096, r8070, r8093;
}
{
add.f16x2 r8099, r8073, r8074;
}
{
mul.f16x2 r8102, r8099, r8060;
}
{
add.f16x2 r8105, r8096, r8102;
}
{
sub.f16x2 r8108, r8079, r8080;
}
{
mul.f16x2 r8111, r8108, r8059;
}
{
sub.f16x2 r8114, r8085, r8086;
}
{
mul.f16x2 r8117, r8114, r8061;
}
{
add.f16x2 r8120, r8111, r8117;
}
{
sub.f16x2 r8123, r8105, r8120;
}
{
add.f16x2 r8126, r8067, r8068;
}
{
mul.f16x2 r8129, r8126, r8058;
}
{
add.f16x2 r8132, r8070, r8129;
}
{
add.f16x2 r8135, r8073, r8074;
}
{
mul.f16x2 r8138, r8135, r8060;
}
{
add.f16x2 r8141, r8132, r8138;
}
{
sub.f16x2 r8144, r8079, r8080;
}
{
mul.f16x2 r8147, r8144, r8059;
}
{
sub.f16x2 r8150, r8085, r8086;
}
{
mul.f16x2 r8153, r8150, r8061;
}
{
add.f16x2 r8156, r8147, r8153;
}
{
add.f16x2 r8159, r8141, r8156;
}
{
add.f16x2 r8162, r8067, r8068;
}
{
mul.f16x2 r8165, r8162, r8060;
}
{
add.f16x2 r8168, r8070, r8165;
}
{
add.f16x2 r8171, r8073, r8074;
}
{
mul.f16x2 r8174, r8171, r8062;
}
{
add.f16x2 r8177, r8168, r8174;
}
{
sub.f16x2 r8180, r8079, r8080;
}
{
mul.f16x2 r8183, r8180, r8061;
}
{
sub.f16x2 r8186, r8085, r8086;
}
{
mul.f16x2 r8189, r8186, r8064;
}
{
add.f16x2 r8192, r8183, r8189;
}
{
sub.f16x2 r8195, r8177, r8192;
}
{
add.f16x2 r8198, r8067, r8068;
}
{
mul.f16x2 r8201, r8198, r8060;
}
{
add.f16x2 r8204, r8070, r8201;
}
{
add.f16x2 r8207, r8073, r8074;
}
{
mul.f16x2 r8210, r8207, r8062;
}
{
add.f16x2 r8213, r8204, r8210;
}
{
sub.f16x2 r8216, r8079, r8080;
}
{
mul.f16x2 r8219, r8216, r8061;
}
{
sub.f16x2 r8222, r8085, r8086;
}
{
mul.f16x2 r8225, r8222, r8064;
}
{
add.f16x2 r8228, r8219, r8225;
}
{
add.f16x2 r8231, r8213, r8228;
}
{
add.f16x2 r8234, r8079, r8080;
}
{
mul.f16x2 r8237, r8234, r8058;
}
{
add.f16x2 r8240, r8082, r8237;
}
{
add.f16x2 r8243, r8085, r8086;
}
{
mul.f16x2 r8246, r8243, r8060;
}
{
add.f16x2 r8249, r8240, r8246;
}
{
sub.f16x2 r8252, r8067, r8068;
}
{
mul.f16x2 r8255, r8252, r8059;
}
{
sub.f16x2 r8258, r8073, r8074;
}
{
mul.f16x2 r8261, r8258, r8061;
}
{
add.f16x2 r8264, r8255, r8261;
}
{
add.f16x2 r8267, r8249, r8264;
}
{
add.f16x2 r8270, r8079, r8080;
}
{
mul.f16x2 r8273, r8270, r8058;
}
{
add.f16x2 r8276, r8082, r8273;
}
{
add.f16x2 r8279, r8085, r8086;
}
{
mul.f16x2 r8282, r8279, r8060;
}
{
add.f16x2 r8285, r8276, r8282;
}
{
sub.f16x2 r8288, r8067, r8068;
}
{
mul.f16x2 r8291, r8288, r8059;
}
{
sub.f16x2 r8294, r8073, r8074;
}
{
mul.f16x2 r8297, r8294, r8061;
}
{
add.f16x2 r8300, r8291, r8297;
}
{
sub.f16x2 r8303, r8285, r8300;
}
{
add.f16x2 r8306, r8079, r8080;
}
{
mul.f16x2 r8309, r8306, r8060;
}
{
add.f16x2 r8312, r8082, r8309;
}
{
add.f16x2 r8315, r8085, r8086;
}
{
mul.f16x2 r8318, r8315, r8062;
}
{
add.f16x2 r8321, r8312, r8318;
}
{
sub.f16x2 r8324, r8067, r8068;
}
{
mul.f16x2 r8327, r8324, r8061;
}
{
sub.f16x2 r8330, r8073, r8074;
}
{
mul.f16x2 r8333, r8330, r8064;
}
{
add.f16x2 r8336, r8327, r8333;
}
{
add.f16x2 r8339, r8321, r8336;
}
{
add.f16x2 r8342, r8079, r8080;
}
{
mul.f16x2 r8345, r8342, r8060;
}
{
add.f16x2 r8348, r8082, r8345;
}
{
add.f16x2 r8351, r8085, r8086;
}
{
mul.f16x2 r8354, r8351, r8062;
}
{
add.f16x2 r8357, r8348, r8354;
}
{
sub.f16x2 r8360, r8067, r8068;
}
{
mul.f16x2 r8363, r8360, r8061;
}
{
sub.f16x2 r8366, r8073, r8074;
}
{
mul.f16x2 r8369, r8366, r8064;
}
{
add.f16x2 r8372, r8363, r8369;
}
{
sub.f16x2 r8375, r8357, r8372;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8378, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8379, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r8380, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r8381, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8382, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8383, {low, high};
}
{
neg.f16x2 r8384, r8383;
}
{
add.f16x2 r8386, r8387, r8388;
}
{
add.f16x2 r8389, r8390, r8386;
}
{
add.f16x2 r8392, r8393, r8394;
}
{
add.f16x2 r8395, r8389, r8392;
}
{
add.f16x2 r8398, r8399, r8400;
}
{
add.f16x2 r8401, r8402, r8398;
}
{
add.f16x2 r8404, r8405, r8406;
}
{
add.f16x2 r8407, r8401, r8404;
}
{
add.f16x2 r8410, r8387, r8388;
}
{
mul.f16x2 r8413, r8410, r8378;
}
{
add.f16x2 r8416, r8390, r8413;
}
{
add.f16x2 r8419, r8393, r8394;
}
{
mul.f16x2 r8422, r8419, r8380;
}
{
add.f16x2 r8425, r8416, r8422;
}
{
sub.f16x2 r8428, r8399, r8400;
}
{
mul.f16x2 r8431, r8428, r8379;
}
{
sub.f16x2 r8434, r8405, r8406;
}
{
mul.f16x2 r8437, r8434, r8381;
}
{
add.f16x2 r8440, r8431, r8437;
}
{
sub.f16x2 r8443, r8425, r8440;
}
{
add.f16x2 r8446, r8387, r8388;
}
{
mul.f16x2 r8449, r8446, r8378;
}
{
add.f16x2 r8452, r8390, r8449;
}
{
add.f16x2 r8455, r8393, r8394;
}
{
mul.f16x2 r8458, r8455, r8380;
}
{
add.f16x2 r8461, r8452, r8458;
}
{
sub.f16x2 r8464, r8399, r8400;
}
{
mul.f16x2 r8467, r8464, r8379;
}
{
sub.f16x2 r8470, r8405, r8406;
}
{
mul.f16x2 r8473, r8470, r8381;
}
{
add.f16x2 r8476, r8467, r8473;
}
{
add.f16x2 r8479, r8461, r8476;
}
{
add.f16x2 r8482, r8387, r8388;
}
{
mul.f16x2 r8485, r8482, r8380;
}
{
add.f16x2 r8488, r8390, r8485;
}
{
add.f16x2 r8491, r8393, r8394;
}
{
mul.f16x2 r8494, r8491, r8382;
}
{
add.f16x2 r8497, r8488, r8494;
}
{
sub.f16x2 r8500, r8399, r8400;
}
{
mul.f16x2 r8503, r8500, r8381;
}
{
sub.f16x2 r8506, r8405, r8406;
}
{
mul.f16x2 r8509, r8506, r8384;
}
{
add.f16x2 r8512, r8503, r8509;
}
{
sub.f16x2 r8515, r8497, r8512;
}
{
add.f16x2 r8518, r8387, r8388;
}
{
mul.f16x2 r8521, r8518, r8380;
}
{
add.f16x2 r8524, r8390, r8521;
}
{
add.f16x2 r8527, r8393, r8394;
}
{
mul.f16x2 r8530, r8527, r8382;
}
{
add.f16x2 r8533, r8524, r8530;
}
{
sub.f16x2 r8536, r8399, r8400;
}
{
mul.f16x2 r8539, r8536, r8381;
}
{
sub.f16x2 r8542, r8405, r8406;
}
{
mul.f16x2 r8545, r8542, r8384;
}
{
add.f16x2 r8548, r8539, r8545;
}
{
add.f16x2 r8551, r8533, r8548;
}
{
add.f16x2 r8554, r8399, r8400;
}
{
mul.f16x2 r8557, r8554, r8378;
}
{
add.f16x2 r8560, r8402, r8557;
}
{
add.f16x2 r8563, r8405, r8406;
}
{
mul.f16x2 r8566, r8563, r8380;
}
{
add.f16x2 r8569, r8560, r8566;
}
{
sub.f16x2 r8572, r8387, r8388;
}
{
mul.f16x2 r8575, r8572, r8379;
}
{
sub.f16x2 r8578, r8393, r8394;
}
{
mul.f16x2 r8581, r8578, r8381;
}
{
add.f16x2 r8584, r8575, r8581;
}
{
add.f16x2 r8587, r8569, r8584;
}
{
add.f16x2 r8590, r8399, r8400;
}
{
mul.f16x2 r8593, r8590, r8378;
}
{
add.f16x2 r8596, r8402, r8593;
}
{
add.f16x2 r8599, r8405, r8406;
}
{
mul.f16x2 r8602, r8599, r8380;
}
{
add.f16x2 r8605, r8596, r8602;
}
{
sub.f16x2 r8608, r8387, r8388;
}
{
mul.f16x2 r8611, r8608, r8379;
}
{
sub.f16x2 r8614, r8393, r8394;
}
{
mul.f16x2 r8617, r8614, r8381;
}
{
add.f16x2 r8620, r8611, r8617;
}
{
sub.f16x2 r8623, r8605, r8620;
}
{
add.f16x2 r8626, r8399, r8400;
}
{
mul.f16x2 r8629, r8626, r8380;
}
{
add.f16x2 r8632, r8402, r8629;
}
{
add.f16x2 r8635, r8405, r8406;
}
{
mul.f16x2 r8638, r8635, r8382;
}
{
add.f16x2 r8641, r8632, r8638;
}
{
sub.f16x2 r8644, r8387, r8388;
}
{
mul.f16x2 r8647, r8644, r8381;
}
{
sub.f16x2 r8650, r8393, r8394;
}
{
mul.f16x2 r8653, r8650, r8384;
}
{
add.f16x2 r8656, r8647, r8653;
}
{
add.f16x2 r8659, r8641, r8656;
}
{
add.f16x2 r8662, r8399, r8400;
}
{
mul.f16x2 r8665, r8662, r8380;
}
{
add.f16x2 r8668, r8402, r8665;
}
{
add.f16x2 r8671, r8405, r8406;
}
{
mul.f16x2 r8674, r8671, r8382;
}
{
add.f16x2 r8677, r8668, r8674;
}
{
sub.f16x2 r8680, r8387, r8388;
}
{
mul.f16x2 r8683, r8680, r8381;
}
{
sub.f16x2 r8686, r8393, r8394;
}
{
mul.f16x2 r8689, r8686, r8384;
}
{
add.f16x2 r8692, r8683, r8689;
}
{
sub.f16x2 r8695, r8677, r8692;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8698, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8699, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r8700, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r8701, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8702, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8703, {low, high};
}
{
neg.f16x2 r8704, r8703;
}
{
add.f16x2 r8706, r8707, r8708;
}
{
add.f16x2 r8709, r8710, r8706;
}
{
add.f16x2 r8712, r8713, r8714;
}
{
add.f16x2 r8715, r8709, r8712;
}
{
add.f16x2 r8718, r8719, r8720;
}
{
add.f16x2 r8721, r8722, r8718;
}
{
add.f16x2 r8724, r8725, r8726;
}
{
add.f16x2 r8727, r8721, r8724;
}
{
add.f16x2 r8730, r8707, r8708;
}
{
mul.f16x2 r8733, r8730, r8698;
}
{
add.f16x2 r8736, r8710, r8733;
}
{
add.f16x2 r8739, r8713, r8714;
}
{
mul.f16x2 r8742, r8739, r8700;
}
{
add.f16x2 r8745, r8736, r8742;
}
{
sub.f16x2 r8748, r8719, r8720;
}
{
mul.f16x2 r8751, r8748, r8699;
}
{
sub.f16x2 r8754, r8725, r8726;
}
{
mul.f16x2 r8757, r8754, r8701;
}
{
add.f16x2 r8760, r8751, r8757;
}
{
sub.f16x2 r8763, r8745, r8760;
}
{
add.f16x2 r8766, r8707, r8708;
}
{
mul.f16x2 r8769, r8766, r8698;
}
{
add.f16x2 r8772, r8710, r8769;
}
{
add.f16x2 r8775, r8713, r8714;
}
{
mul.f16x2 r8778, r8775, r8700;
}
{
add.f16x2 r8781, r8772, r8778;
}
{
sub.f16x2 r8784, r8719, r8720;
}
{
mul.f16x2 r8787, r8784, r8699;
}
{
sub.f16x2 r8790, r8725, r8726;
}
{
mul.f16x2 r8793, r8790, r8701;
}
{
add.f16x2 r8796, r8787, r8793;
}
{
add.f16x2 r8799, r8781, r8796;
}
{
add.f16x2 r8802, r8707, r8708;
}
{
mul.f16x2 r8805, r8802, r8700;
}
{
add.f16x2 r8808, r8710, r8805;
}
{
add.f16x2 r8811, r8713, r8714;
}
{
mul.f16x2 r8814, r8811, r8702;
}
{
add.f16x2 r8817, r8808, r8814;
}
{
sub.f16x2 r8820, r8719, r8720;
}
{
mul.f16x2 r8823, r8820, r8701;
}
{
sub.f16x2 r8826, r8725, r8726;
}
{
mul.f16x2 r8829, r8826, r8704;
}
{
add.f16x2 r8832, r8823, r8829;
}
{
sub.f16x2 r8835, r8817, r8832;
}
{
add.f16x2 r8838, r8707, r8708;
}
{
mul.f16x2 r8841, r8838, r8700;
}
{
add.f16x2 r8844, r8710, r8841;
}
{
add.f16x2 r8847, r8713, r8714;
}
{
mul.f16x2 r8850, r8847, r8702;
}
{
add.f16x2 r8853, r8844, r8850;
}
{
sub.f16x2 r8856, r8719, r8720;
}
{
mul.f16x2 r8859, r8856, r8701;
}
{
sub.f16x2 r8862, r8725, r8726;
}
{
mul.f16x2 r8865, r8862, r8704;
}
{
add.f16x2 r8868, r8859, r8865;
}
{
add.f16x2 r8871, r8853, r8868;
}
{
add.f16x2 r8874, r8719, r8720;
}
{
mul.f16x2 r8877, r8874, r8698;
}
{
add.f16x2 r8880, r8722, r8877;
}
{
add.f16x2 r8883, r8725, r8726;
}
{
mul.f16x2 r8886, r8883, r8700;
}
{
add.f16x2 r8889, r8880, r8886;
}
{
sub.f16x2 r8892, r8707, r8708;
}
{
mul.f16x2 r8895, r8892, r8699;
}
{
sub.f16x2 r8898, r8713, r8714;
}
{
mul.f16x2 r8901, r8898, r8701;
}
{
add.f16x2 r8904, r8895, r8901;
}
{
add.f16x2 r8907, r8889, r8904;
}
{
add.f16x2 r8910, r8719, r8720;
}
{
mul.f16x2 r8913, r8910, r8698;
}
{
add.f16x2 r8916, r8722, r8913;
}
{
add.f16x2 r8919, r8725, r8726;
}
{
mul.f16x2 r8922, r8919, r8700;
}
{
add.f16x2 r8925, r8916, r8922;
}
{
sub.f16x2 r8928, r8707, r8708;
}
{
mul.f16x2 r8931, r8928, r8699;
}
{
sub.f16x2 r8934, r8713, r8714;
}
{
mul.f16x2 r8937, r8934, r8701;
}
{
add.f16x2 r8940, r8931, r8937;
}
{
sub.f16x2 r8943, r8925, r8940;
}
{
add.f16x2 r8946, r8719, r8720;
}
{
mul.f16x2 r8949, r8946, r8700;
}
{
add.f16x2 r8952, r8722, r8949;
}
{
add.f16x2 r8955, r8725, r8726;
}
{
mul.f16x2 r8958, r8955, r8702;
}
{
add.f16x2 r8961, r8952, r8958;
}
{
sub.f16x2 r8964, r8707, r8708;
}
{
mul.f16x2 r8967, r8964, r8701;
}
{
sub.f16x2 r8970, r8713, r8714;
}
{
mul.f16x2 r8973, r8970, r8704;
}
{
add.f16x2 r8976, r8967, r8973;
}
{
add.f16x2 r8979, r8961, r8976;
}
{
add.f16x2 r8982, r8719, r8720;
}
{
mul.f16x2 r8985, r8982, r8700;
}
{
add.f16x2 r8988, r8722, r8985;
}
{
add.f16x2 r8991, r8725, r8726;
}
{
mul.f16x2 r8994, r8991, r8702;
}
{
add.f16x2 r8997, r8988, r8994;
}
{
sub.f16x2 r9000, r8707, r8708;
}
{
mul.f16x2 r9003, r9000, r8701;
}
{
sub.f16x2 r9006, r8713, r8714;
}
{
mul.f16x2 r9009, r9006, r8704;
}
{
add.f16x2 r9012, r9003, r9009;
}
{
sub.f16x2 r9015, r8997, r9012;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r9018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r9019, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r9020, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r9021, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r9022, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r9023, {low, high};
}
{
neg.f16x2 r9024, r9023;
}
{
add.f16x2 r9026, r9027, r9028;
}
{
add.f16x2 r9029, r9030, r9026;
}
{
add.f16x2 r9032, r9033, r9034;
}
{
add.f16x2 r9035, r9029, r9032;
}
{
add.f16x2 r9038, r9039, r9040;
}
{
add.f16x2 r9041, r9042, r9038;
}
{
add.f16x2 r9044, r9045, r9046;
}
{
add.f16x2 r9047, r9041, r9044;
}
{
add.f16x2 r9050, r9027, r9028;
}
{
mul.f16x2 r9053, r9050, r9018;
}
{
add.f16x2 r9056, r9030, r9053;
}
{
add.f16x2 r9059, r9033, r9034;
}
{
mul.f16x2 r9062, r9059, r9020;
}
{
add.f16x2 r9065, r9056, r9062;
}
{
sub.f16x2 r9068, r9039, r9040;
}
{
mul.f16x2 r9071, r9068, r9019;
}
{
sub.f16x2 r9074, r9045, r9046;
}
{
mul.f16x2 r9077, r9074, r9021;
}
{
add.f16x2 r9080, r9071, r9077;
}
{
sub.f16x2 r9083, r9065, r9080;
}
{
add.f16x2 r9086, r9027, r9028;
}
{
mul.f16x2 r9089, r9086, r9018;
}
{
add.f16x2 r9092, r9030, r9089;
}
{
add.f16x2 r9095, r9033, r9034;
}
{
mul.f16x2 r9098, r9095, r9020;
}
{
add.f16x2 r9101, r9092, r9098;
}
{
sub.f16x2 r9104, r9039, r9040;
}
{
mul.f16x2 r9107, r9104, r9019;
}
{
sub.f16x2 r9110, r9045, r9046;
}
{
mul.f16x2 r9113, r9110, r9021;
}
{
add.f16x2 r9116, r9107, r9113;
}
{
add.f16x2 r9119, r9101, r9116;
}
{
add.f16x2 r9122, r9027, r9028;
}
{
mul.f16x2 r9125, r9122, r9020;
}
{
add.f16x2 r9128, r9030, r9125;
}
{
add.f16x2 r9131, r9033, r9034;
}
{
mul.f16x2 r9134, r9131, r9022;
}
{
add.f16x2 r9137, r9128, r9134;
}
{
sub.f16x2 r9140, r9039, r9040;
}
{
mul.f16x2 r9143, r9140, r9021;
}
{
sub.f16x2 r9146, r9045, r9046;
}
{
mul.f16x2 r9149, r9146, r9024;
}
{
add.f16x2 r9152, r9143, r9149;
}
{
sub.f16x2 r9155, r9137, r9152;
}
{
add.f16x2 r9158, r9027, r9028;
}
{
mul.f16x2 r9161, r9158, r9020;
}
{
add.f16x2 r9164, r9030, r9161;
}
{
add.f16x2 r9167, r9033, r9034;
}
{
mul.f16x2 r9170, r9167, r9022;
}
{
add.f16x2 r9173, r9164, r9170;
}
{
sub.f16x2 r9176, r9039, r9040;
}
{
mul.f16x2 r9179, r9176, r9021;
}
{
sub.f16x2 r9182, r9045, r9046;
}
{
mul.f16x2 r9185, r9182, r9024;
}
{
add.f16x2 r9188, r9179, r9185;
}
{
add.f16x2 r9191, r9173, r9188;
}
{
add.f16x2 r9194, r9039, r9040;
}
{
mul.f16x2 r9197, r9194, r9018;
}
{
add.f16x2 r9200, r9042, r9197;
}
{
add.f16x2 r9203, r9045, r9046;
}
{
mul.f16x2 r9206, r9203, r9020;
}
{
add.f16x2 r9209, r9200, r9206;
}
{
sub.f16x2 r9212, r9027, r9028;
}
{
mul.f16x2 r9215, r9212, r9019;
}
{
sub.f16x2 r9218, r9033, r9034;
}
{
mul.f16x2 r9221, r9218, r9021;
}
{
add.f16x2 r9224, r9215, r9221;
}
{
add.f16x2 r9227, r9209, r9224;
}
{
add.f16x2 r9230, r9039, r9040;
}
{
mul.f16x2 r9233, r9230, r9018;
}
{
add.f16x2 r9236, r9042, r9233;
}
{
add.f16x2 r9239, r9045, r9046;
}
{
mul.f16x2 r9242, r9239, r9020;
}
{
add.f16x2 r9245, r9236, r9242;
}
{
sub.f16x2 r9248, r9027, r9028;
}
{
mul.f16x2 r9251, r9248, r9019;
}
{
sub.f16x2 r9254, r9033, r9034;
}
{
mul.f16x2 r9257, r9254, r9021;
}
{
add.f16x2 r9260, r9251, r9257;
}
{
sub.f16x2 r9263, r9245, r9260;
}
{
add.f16x2 r9266, r9039, r9040;
}
{
mul.f16x2 r9269, r9266, r9020;
}
{
add.f16x2 r9272, r9042, r9269;
}
{
add.f16x2 r9275, r9045, r9046;
}
{
mul.f16x2 r9278, r9275, r9022;
}
{
add.f16x2 r9281, r9272, r9278;
}
{
sub.f16x2 r9284, r9027, r9028;
}
{
mul.f16x2 r9287, r9284, r9021;
}
{
sub.f16x2 r9290, r9033, r9034;
}
{
mul.f16x2 r9293, r9290, r9024;
}
{
add.f16x2 r9296, r9287, r9293;
}
{
add.f16x2 r9299, r9281, r9296;
}
{
add.f16x2 r9302, r9039, r9040;
}
{
mul.f16x2 r9305, r9302, r9020;
}
{
add.f16x2 r9308, r9042, r9305;
}
{
add.f16x2 r9311, r9045, r9046;
}
{
mul.f16x2 r9314, r9311, r9022;
}
{
add.f16x2 r9317, r9308, r9314;
}
{
sub.f16x2 r9320, r9027, r9028;
}
{
mul.f16x2 r9323, r9320, r9021;
}
{
sub.f16x2 r9326, r9033, r9034;
}
{
mul.f16x2 r9329, r9326, r9024;
}
{
add.f16x2 r9332, r9323, r9329;
}
{
sub.f16x2 r9335, r9317, r9332;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f62;
cvt.rn.f16.f32 high, f62;
mov.b32 r9338, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f64;
cvt.rn.f16.f32 high, f64;
mov.b32 r9339, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f66;
cvt.rn.f16.f32 high, f66;
mov.b32 r9340, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f68;
cvt.rn.f16.f32 high, f68;
mov.b32 r9341, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f70;
cvt.rn.f16.f32 high, f70;
mov.b32 r9342, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f72;
cvt.rn.f16.f32 high, f72;
mov.b32 r9343, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r9344, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r9345, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r9348, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r9349, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r9352, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r9353, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r9354, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r9355, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r9360, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r9361, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r9368, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r9369, {low, high};
}
{
mul.f16x2 r9386, r8123, r9338;
}
{
mul.f16x2 r9389, r8267, r9339;
}
{
sub.f16x2 r9392, r9386, r9389;
}
{
mul.f16x2 r9395, r8123, r9339;
}
{
fma.rn.f16x2 r9398, r8267, r9338, r9395;
}
{
mul.f16x2 r9402, r8443, r9340;
}
{
mul.f16x2 r9405, r8587, r9341;
}
{
sub.f16x2 r9408, r9402, r9405;
}
{
mul.f16x2 r9411, r8443, r9341;
}
{
fma.rn.f16x2 r9414, r8587, r9340, r9411;
}
{
mul.f16x2 r9418, r8763, r9342;
}
{
mul.f16x2 r9421, r8907, r9343;
}
{
sub.f16x2 r9424, r9418, r9421;
}
{
mul.f16x2 r9427, r8763, r9343;
}
{
fma.rn.f16x2 r9430, r8907, r9342, r9427;
}
{
mul.f16x2 r9434, r9083, r9344;
}
{
mul.f16x2 r9437, r9227, r9345;
}
{
sub.f16x2 r9440, r9434, r9437;
}
{
mul.f16x2 r9443, r9083, r9345;
}
{
fma.rn.f16x2 r9446, r9227, r9344, r9443;
}
{
mul.f16x2 r9450, r8195, r9340;
}
{
mul.f16x2 r9453, r8339, r9341;
}
{
sub.f16x2 r9456, r9450, r9453;
}
{
mul.f16x2 r9459, r8195, r9341;
}
{
fma.rn.f16x2 r9462, r8339, r9340, r9459;
}
{
mul.f16x2 r9466, r8515, r9344;
}
{
mul.f16x2 r9469, r8659, r9345;
}
{
sub.f16x2 r9472, r9466, r9469;
}
{
mul.f16x2 r9475, r8515, r9345;
}
{
fma.rn.f16x2 r9478, r8659, r9344, r9475;
}
{
mul.f16x2 r9482, r8835, r9348;
}
{
mul.f16x2 r9485, r8979, r9349;
}
{
sub.f16x2 r9488, r9482, r9485;
}
{
mul.f16x2 r9491, r8835, r9349;
}
{
fma.rn.f16x2 r9494, r8979, r9348, r9491;
}
{
mul.f16x2 r9498, r9155, r9352;
}
{
mul.f16x2 r9501, r9299, r9353;
}
{
sub.f16x2 r9504, r9498, r9501;
}
{
mul.f16x2 r9507, r9155, r9353;
}
{
fma.rn.f16x2 r9510, r9299, r9352, r9507;
}
{
mul.f16x2 r9514, r8231, r9342;
}
{
mul.f16x2 r9517, r8375, r9343;
}
{
sub.f16x2 r9520, r9514, r9517;
}
{
mul.f16x2 r9523, r8231, r9343;
}
{
fma.rn.f16x2 r9526, r8375, r9342, r9523;
}
{
mul.f16x2 r9530, r8551, r9348;
}
{
mul.f16x2 r9533, r8695, r9349;
}
{
sub.f16x2 r9536, r9530, r9533;
}
{
mul.f16x2 r9539, r8551, r9349;
}
{
fma.rn.f16x2 r9542, r8695, r9348, r9539;
}
{
mul.f16x2 r9546, r8871, r9354;
}
{
mul.f16x2 r9549, r9015, r9355;
}
{
sub.f16x2 r9552, r9546, r9549;
}
{
mul.f16x2 r9555, r8871, r9355;
}
{
fma.rn.f16x2 r9558, r9015, r9354, r9555;
}
{
mul.f16x2 r9562, r9191, r9360;
}
{
mul.f16x2 r9565, r9335, r9361;
}
{
sub.f16x2 r9568, r9562, r9565;
}
{
mul.f16x2 r9571, r9191, r9361;
}
{
fma.rn.f16x2 r9574, r9335, r9360, r9571;
}
{
mul.f16x2 r9578, r8159, r9344;
}
{
mul.f16x2 r9581, r8303, r9345;
}
{
sub.f16x2 r9584, r9578, r9581;
}
{
mul.f16x2 r9587, r8159, r9345;
}
{
fma.rn.f16x2 r9590, r8303, r9344, r9587;
}
{
mul.f16x2 r9594, r8479, r9352;
}
{
mul.f16x2 r9597, r8623, r9353;
}
{
sub.f16x2 r9600, r9594, r9597;
}
{
mul.f16x2 r9603, r8479, r9353;
}
{
fma.rn.f16x2 r9606, r8623, r9352, r9603;
}
{
mul.f16x2 r9610, r8799, r9360;
}
{
mul.f16x2 r9613, r8943, r9361;
}
{
sub.f16x2 r9616, r9610, r9613;
}
{
mul.f16x2 r9619, r8799, r9361;
}
{
fma.rn.f16x2 r9622, r8943, r9360, r9619;
}
{
mul.f16x2 r9626, r9119, r9368;
}
{
mul.f16x2 r9629, r9263, r9369;
}
{
sub.f16x2 r9632, r9626, r9629;
}
{
mul.f16x2 r9635, r9119, r9369;
}
{
fma.rn.f16x2 r9638, r9263, r9368, r9635;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r9642, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r9643, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r9644, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r9645, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r9646, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r9647, {low, high};
}
{
neg.f16x2 r9648, r9647;
}
{
add.f16x2 r9650, r8075, r9035;
}
{
add.f16x2 r9653, r7755, r9650;
}
{
add.f16x2 r9656, r8395, r8715;
}
{
add.f16x2 %0, r9653, r9656;
}
{
add.f16x2 r9662, r8087, r9047;
}
{
add.f16x2 r9665, r7767, r9662;
}
{
add.f16x2 r9668, r8407, r8727;
}
{
add.f16x2 %1, r9665, r9668;
}
{
add.f16x2 r9674, r8075, r9035;
}
{
mul.f16x2 r9677, r9674, r9642;
}
{
add.f16x2 r9680, r7755, r9677;
}
{
add.f16x2 r9683, r8395, r8715;
}
{
mul.f16x2 r9686, r9683, r9644;
}
{
add.f16x2 r9689, r9680, r9686;
}
{
sub.f16x2 r9692, r8087, r9047;
}
{
mul.f16x2 r9695, r9692, r9643;
}
{
sub.f16x2 r9698, r8407, r8727;
}
{
mul.f16x2 r9701, r9698, r9645;
}
{
add.f16x2 r9704, r9695, r9701;
}
{
sub.f16x2 %10, r9689, r9704;
}
{
add.f16x2 r9710, r8075, r9035;
}
{
mul.f16x2 r9713, r9710, r9642;
}
{
add.f16x2 r9716, r7755, r9713;
}
{
add.f16x2 r9719, r8395, r8715;
}
{
mul.f16x2 r9722, r9719, r9644;
}
{
add.f16x2 r9725, r9716, r9722;
}
{
sub.f16x2 r9728, r8087, r9047;
}
{
mul.f16x2 r9731, r9728, r9643;
}
{
sub.f16x2 r9734, r8407, r8727;
}
{
mul.f16x2 r9737, r9734, r9645;
}
{
add.f16x2 r9740, r9731, r9737;
}
{
add.f16x2 %40, r9725, r9740;
}
{
add.f16x2 r9746, r8075, r9035;
}
{
mul.f16x2 r9749, r9746, r9644;
}
{
add.f16x2 r9752, r7755, r9749;
}
{
add.f16x2 r9755, r8395, r8715;
}
{
mul.f16x2 r9758, r9755, r9646;
}
{
add.f16x2 r9761, r9752, r9758;
}
{
sub.f16x2 r9764, r8087, r9047;
}
{
mul.f16x2 r9767, r9764, r9645;
}
{
sub.f16x2 r9770, r8407, r8727;
}
{
mul.f16x2 r9773, r9770, r9648;
}
{
add.f16x2 r9776, r9767, r9773;
}
{
sub.f16x2 %20, r9761, r9776;
}
{
add.f16x2 r9782, r8075, r9035;
}
{
mul.f16x2 r9785, r9782, r9644;
}
{
add.f16x2 r9788, r7755, r9785;
}
{
add.f16x2 r9791, r8395, r8715;
}
{
mul.f16x2 r9794, r9791, r9646;
}
{
add.f16x2 r9797, r9788, r9794;
}
{
sub.f16x2 r9800, r8087, r9047;
}
{
mul.f16x2 r9803, r9800, r9645;
}
{
sub.f16x2 r9806, r8407, r8727;
}
{
mul.f16x2 r9809, r9806, r9648;
}
{
add.f16x2 r9812, r9803, r9809;
}
{
add.f16x2 %30, r9797, r9812;
}
{
add.f16x2 r9818, r8087, r9047;
}
{
mul.f16x2 r9821, r9818, r9642;
}
{
add.f16x2 r9824, r7767, r9821;
}
{
add.f16x2 r9827, r8407, r8727;
}
{
mul.f16x2 r9830, r9827, r9644;
}
{
add.f16x2 r9833, r9824, r9830;
}
{
sub.f16x2 r9836, r8075, r9035;
}
{
mul.f16x2 r9839, r9836, r9643;
}
{
sub.f16x2 r9842, r8395, r8715;
}
{
mul.f16x2 r9845, r9842, r9645;
}
{
add.f16x2 r9848, r9839, r9845;
}
{
add.f16x2 %11, r9833, r9848;
}
{
add.f16x2 r9854, r8087, r9047;
}
{
mul.f16x2 r9857, r9854, r9642;
}
{
add.f16x2 r9860, r7767, r9857;
}
{
add.f16x2 r9863, r8407, r8727;
}
{
mul.f16x2 r9866, r9863, r9644;
}
{
add.f16x2 r9869, r9860, r9866;
}
{
sub.f16x2 r9872, r8075, r9035;
}
{
mul.f16x2 r9875, r9872, r9643;
}
{
sub.f16x2 r9878, r8395, r8715;
}
{
mul.f16x2 r9881, r9878, r9645;
}
{
add.f16x2 r9884, r9875, r9881;
}
{
sub.f16x2 %41, r9869, r9884;
}
{
add.f16x2 r9890, r8087, r9047;
}
{
mul.f16x2 r9893, r9890, r9644;
}
{
add.f16x2 r9896, r7767, r9893;
}
{
add.f16x2 r9899, r8407, r8727;
}
{
mul.f16x2 r9902, r9899, r9646;
}
{
add.f16x2 r9905, r9896, r9902;
}
{
sub.f16x2 r9908, r8075, r9035;
}
{
mul.f16x2 r9911, r9908, r9645;
}
{
sub.f16x2 r9914, r8395, r8715;
}
{
mul.f16x2 r9917, r9914, r9648;
}
{
add.f16x2 r9920, r9911, r9917;
}
{
add.f16x2 %21, r9905, r9920;
}
{
add.f16x2 r9926, r8087, r9047;
}
{
mul.f16x2 r9929, r9926, r9644;
}
{
add.f16x2 r9932, r7767, r9929;
}
{
add.f16x2 r9935, r8407, r8727;
}
{
mul.f16x2 r9938, r9935, r9646;
}
{
add.f16x2 r9941, r9932, r9938;
}
{
sub.f16x2 r9944, r8075, r9035;
}
{
mul.f16x2 r9947, r9944, r9645;
}
{
sub.f16x2 r9950, r8395, r8715;
}
{
mul.f16x2 r9953, r9950, r9648;
}
{
add.f16x2 r9956, r9947, r9953;
}
{
sub.f16x2 %31, r9941, r9956;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r9962, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r9963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r9964, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r9965, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r9966, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r9967, {low, high};
}
{
neg.f16x2 r9968, r9967;
}
{
add.f16x2 r9970, r9392, r9440;
}
{
add.f16x2 r9973, r7803, r9970;
}
{
add.f16x2 r9976, r9408, r9424;
}
{
add.f16x2 %2, r9973, r9976;
}
{
add.f16x2 r9982, r9398, r9446;
}
{
add.f16x2 r9985, r7947, r9982;
}
{
add.f16x2 r9988, r9414, r9430;
}
{
add.f16x2 %3, r9985, r9988;
}
{
add.f16x2 r9994, r9392, r9440;
}
{
mul.f16x2 r9997, r9994, r9962;
}
{
add.f16x2 r10000, r7803, r9997;
}
{
add.f16x2 r10003, r9408, r9424;
}
{
mul.f16x2 r10006, r10003, r9964;
}
{
add.f16x2 r10009, r10000, r10006;
}
{
sub.f16x2 r10012, r9398, r9446;
}
{
mul.f16x2 r10015, r10012, r9963;
}
{
sub.f16x2 r10018, r9414, r9430;
}
{
mul.f16x2 r10021, r10018, r9965;
}
{
add.f16x2 r10024, r10015, r10021;
}
{
sub.f16x2 %12, r10009, r10024;
}
{
add.f16x2 r10030, r9392, r9440;
}
{
mul.f16x2 r10033, r10030, r9962;
}
{
add.f16x2 r10036, r7803, r10033;
}
{
add.f16x2 r10039, r9408, r9424;
}
{
mul.f16x2 r10042, r10039, r9964;
}
{
add.f16x2 r10045, r10036, r10042;
}
{
sub.f16x2 r10048, r9398, r9446;
}
{
mul.f16x2 r10051, r10048, r9963;
}
{
sub.f16x2 r10054, r9414, r9430;
}
{
mul.f16x2 r10057, r10054, r9965;
}
{
add.f16x2 r10060, r10051, r10057;
}
{
add.f16x2 %42, r10045, r10060;
}
{
add.f16x2 r10066, r9392, r9440;
}
{
mul.f16x2 r10069, r10066, r9964;
}
{
add.f16x2 r10072, r7803, r10069;
}
{
add.f16x2 r10075, r9408, r9424;
}
{
mul.f16x2 r10078, r10075, r9966;
}
{
add.f16x2 r10081, r10072, r10078;
}
{
sub.f16x2 r10084, r9398, r9446;
}
{
mul.f16x2 r10087, r10084, r9965;
}
{
sub.f16x2 r10090, r9414, r9430;
}
{
mul.f16x2 r10093, r10090, r9968;
}
{
add.f16x2 r10096, r10087, r10093;
}
{
sub.f16x2 %22, r10081, r10096;
}
{
add.f16x2 r10102, r9392, r9440;
}
{
mul.f16x2 r10105, r10102, r9964;
}
{
add.f16x2 r10108, r7803, r10105;
}
{
add.f16x2 r10111, r9408, r9424;
}
{
mul.f16x2 r10114, r10111, r9966;
}
{
add.f16x2 r10117, r10108, r10114;
}
{
sub.f16x2 r10120, r9398, r9446;
}
{
mul.f16x2 r10123, r10120, r9965;
}
{
sub.f16x2 r10126, r9414, r9430;
}
{
mul.f16x2 r10129, r10126, r9968;
}
{
add.f16x2 r10132, r10123, r10129;
}
{
add.f16x2 %32, r10117, r10132;
}
{
add.f16x2 r10138, r9398, r9446;
}
{
mul.f16x2 r10141, r10138, r9962;
}
{
add.f16x2 r10144, r7947, r10141;
}
{
add.f16x2 r10147, r9414, r9430;
}
{
mul.f16x2 r10150, r10147, r9964;
}
{
add.f16x2 r10153, r10144, r10150;
}
{
sub.f16x2 r10156, r9392, r9440;
}
{
mul.f16x2 r10159, r10156, r9963;
}
{
sub.f16x2 r10162, r9408, r9424;
}
{
mul.f16x2 r10165, r10162, r9965;
}
{
add.f16x2 r10168, r10159, r10165;
}
{
add.f16x2 %13, r10153, r10168;
}
{
add.f16x2 r10174, r9398, r9446;
}
{
mul.f16x2 r10177, r10174, r9962;
}
{
add.f16x2 r10180, r7947, r10177;
}
{
add.f16x2 r10183, r9414, r9430;
}
{
mul.f16x2 r10186, r10183, r9964;
}
{
add.f16x2 r10189, r10180, r10186;
}
{
sub.f16x2 r10192, r9392, r9440;
}
{
mul.f16x2 r10195, r10192, r9963;
}
{
sub.f16x2 r10198, r9408, r9424;
}
{
mul.f16x2 r10201, r10198, r9965;
}
{
add.f16x2 r10204, r10195, r10201;
}
{
sub.f16x2 %43, r10189, r10204;
}
{
add.f16x2 r10210, r9398, r9446;
}
{
mul.f16x2 r10213, r10210, r9964;
}
{
add.f16x2 r10216, r7947, r10213;
}
{
add.f16x2 r10219, r9414, r9430;
}
{
mul.f16x2 r10222, r10219, r9966;
}
{
add.f16x2 r10225, r10216, r10222;
}
{
sub.f16x2 r10228, r9392, r9440;
}
{
mul.f16x2 r10231, r10228, r9965;
}
{
sub.f16x2 r10234, r9408, r9424;
}
{
mul.f16x2 r10237, r10234, r9968;
}
{
add.f16x2 r10240, r10231, r10237;
}
{
add.f16x2 %23, r10225, r10240;
}
{
add.f16x2 r10246, r9398, r9446;
}
{
mul.f16x2 r10249, r10246, r9964;
}
{
add.f16x2 r10252, r7947, r10249;
}
{
add.f16x2 r10255, r9414, r9430;
}
{
mul.f16x2 r10258, r10255, r9966;
}
{
add.f16x2 r10261, r10252, r10258;
}
{
sub.f16x2 r10264, r9392, r9440;
}
{
mul.f16x2 r10267, r10264, r9965;
}
{
sub.f16x2 r10270, r9408, r9424;
}
{
mul.f16x2 r10273, r10270, r9968;
}
{
add.f16x2 r10276, r10267, r10273;
}
{
sub.f16x2 %33, r10261, r10276;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10282, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10283, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r10284, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r10285, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10286, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10287, {low, high};
}
{
neg.f16x2 r10288, r10287;
}
{
add.f16x2 r10290, r9456, r9504;
}
{
add.f16x2 r10293, r7875, r10290;
}
{
add.f16x2 r10296, r9472, r9488;
}
{
add.f16x2 %4, r10293, r10296;
}
{
add.f16x2 r10302, r9462, r9510;
}
{
add.f16x2 r10305, r8019, r10302;
}
{
add.f16x2 r10308, r9478, r9494;
}
{
add.f16x2 %5, r10305, r10308;
}
{
add.f16x2 r10314, r9456, r9504;
}
{
mul.f16x2 r10317, r10314, r10282;
}
{
add.f16x2 r10320, r7875, r10317;
}
{
add.f16x2 r10323, r9472, r9488;
}
{
mul.f16x2 r10326, r10323, r10284;
}
{
add.f16x2 r10329, r10320, r10326;
}
{
sub.f16x2 r10332, r9462, r9510;
}
{
mul.f16x2 r10335, r10332, r10283;
}
{
sub.f16x2 r10338, r9478, r9494;
}
{
mul.f16x2 r10341, r10338, r10285;
}
{
add.f16x2 r10344, r10335, r10341;
}
{
sub.f16x2 %14, r10329, r10344;
}
{
add.f16x2 r10350, r9456, r9504;
}
{
mul.f16x2 r10353, r10350, r10282;
}
{
add.f16x2 r10356, r7875, r10353;
}
{
add.f16x2 r10359, r9472, r9488;
}
{
mul.f16x2 r10362, r10359, r10284;
}
{
add.f16x2 r10365, r10356, r10362;
}
{
sub.f16x2 r10368, r9462, r9510;
}
{
mul.f16x2 r10371, r10368, r10283;
}
{
sub.f16x2 r10374, r9478, r9494;
}
{
mul.f16x2 r10377, r10374, r10285;
}
{
add.f16x2 r10380, r10371, r10377;
}
{
add.f16x2 %44, r10365, r10380;
}
{
add.f16x2 r10386, r9456, r9504;
}
{
mul.f16x2 r10389, r10386, r10284;
}
{
add.f16x2 r10392, r7875, r10389;
}
{
add.f16x2 r10395, r9472, r9488;
}
{
mul.f16x2 r10398, r10395, r10286;
}
{
add.f16x2 r10401, r10392, r10398;
}
{
sub.f16x2 r10404, r9462, r9510;
}
{
mul.f16x2 r10407, r10404, r10285;
}
{
sub.f16x2 r10410, r9478, r9494;
}
{
mul.f16x2 r10413, r10410, r10288;
}
{
add.f16x2 r10416, r10407, r10413;
}
{
sub.f16x2 %24, r10401, r10416;
}
{
add.f16x2 r10422, r9456, r9504;
}
{
mul.f16x2 r10425, r10422, r10284;
}
{
add.f16x2 r10428, r7875, r10425;
}
{
add.f16x2 r10431, r9472, r9488;
}
{
mul.f16x2 r10434, r10431, r10286;
}
{
add.f16x2 r10437, r10428, r10434;
}
{
sub.f16x2 r10440, r9462, r9510;
}
{
mul.f16x2 r10443, r10440, r10285;
}
{
sub.f16x2 r10446, r9478, r9494;
}
{
mul.f16x2 r10449, r10446, r10288;
}
{
add.f16x2 r10452, r10443, r10449;
}
{
add.f16x2 %34, r10437, r10452;
}
{
add.f16x2 r10458, r9462, r9510;
}
{
mul.f16x2 r10461, r10458, r10282;
}
{
add.f16x2 r10464, r8019, r10461;
}
{
add.f16x2 r10467, r9478, r9494;
}
{
mul.f16x2 r10470, r10467, r10284;
}
{
add.f16x2 r10473, r10464, r10470;
}
{
sub.f16x2 r10476, r9456, r9504;
}
{
mul.f16x2 r10479, r10476, r10283;
}
{
sub.f16x2 r10482, r9472, r9488;
}
{
mul.f16x2 r10485, r10482, r10285;
}
{
add.f16x2 r10488, r10479, r10485;
}
{
add.f16x2 %15, r10473, r10488;
}
{
add.f16x2 r10494, r9462, r9510;
}
{
mul.f16x2 r10497, r10494, r10282;
}
{
add.f16x2 r10500, r8019, r10497;
}
{
add.f16x2 r10503, r9478, r9494;
}
{
mul.f16x2 r10506, r10503, r10284;
}
{
add.f16x2 r10509, r10500, r10506;
}
{
sub.f16x2 r10512, r9456, r9504;
}
{
mul.f16x2 r10515, r10512, r10283;
}
{
sub.f16x2 r10518, r9472, r9488;
}
{
mul.f16x2 r10521, r10518, r10285;
}
{
add.f16x2 r10524, r10515, r10521;
}
{
sub.f16x2 %45, r10509, r10524;
}
{
add.f16x2 r10530, r9462, r9510;
}
{
mul.f16x2 r10533, r10530, r10284;
}
{
add.f16x2 r10536, r8019, r10533;
}
{
add.f16x2 r10539, r9478, r9494;
}
{
mul.f16x2 r10542, r10539, r10286;
}
{
add.f16x2 r10545, r10536, r10542;
}
{
sub.f16x2 r10548, r9456, r9504;
}
{
mul.f16x2 r10551, r10548, r10285;
}
{
sub.f16x2 r10554, r9472, r9488;
}
{
mul.f16x2 r10557, r10554, r10288;
}
{
add.f16x2 r10560, r10551, r10557;
}
{
add.f16x2 %25, r10545, r10560;
}
{
add.f16x2 r10566, r9462, r9510;
}
{
mul.f16x2 r10569, r10566, r10284;
}
{
add.f16x2 r10572, r8019, r10569;
}
{
add.f16x2 r10575, r9478, r9494;
}
{
mul.f16x2 r10578, r10575, r10286;
}
{
add.f16x2 r10581, r10572, r10578;
}
{
sub.f16x2 r10584, r9456, r9504;
}
{
mul.f16x2 r10587, r10584, r10285;
}
{
sub.f16x2 r10590, r9472, r9488;
}
{
mul.f16x2 r10593, r10590, r10288;
}
{
add.f16x2 r10596, r10587, r10593;
}
{
sub.f16x2 %35, r10581, r10596;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10602, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10603, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r10604, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r10605, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10606, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10607, {low, high};
}
{
neg.f16x2 r10608, r10607;
}
{
add.f16x2 r10610, r9520, r9568;
}
{
add.f16x2 r10613, r7911, r10610;
}
{
add.f16x2 r10616, r9536, r9552;
}
{
add.f16x2 %6, r10613, r10616;
}
{
add.f16x2 r10622, r9526, r9574;
}
{
add.f16x2 r10625, r8055, r10622;
}
{
add.f16x2 r10628, r9542, r9558;
}
{
add.f16x2 %7, r10625, r10628;
}
{
add.f16x2 r10634, r9520, r9568;
}
{
mul.f16x2 r10637, r10634, r10602;
}
{
add.f16x2 r10640, r7911, r10637;
}
{
add.f16x2 r10643, r9536, r9552;
}
{
mul.f16x2 r10646, r10643, r10604;
}
{
add.f16x2 r10649, r10640, r10646;
}
{
sub.f16x2 r10652, r9526, r9574;
}
{
mul.f16x2 r10655, r10652, r10603;
}
{
sub.f16x2 r10658, r9542, r9558;
}
{
mul.f16x2 r10661, r10658, r10605;
}
{
add.f16x2 r10664, r10655, r10661;
}
{
sub.f16x2 %16, r10649, r10664;
}
{
add.f16x2 r10670, r9520, r9568;
}
{
mul.f16x2 r10673, r10670, r10602;
}
{
add.f16x2 r10676, r7911, r10673;
}
{
add.f16x2 r10679, r9536, r9552;
}
{
mul.f16x2 r10682, r10679, r10604;
}
{
add.f16x2 r10685, r10676, r10682;
}
{
sub.f16x2 r10688, r9526, r9574;
}
{
mul.f16x2 r10691, r10688, r10603;
}
{
sub.f16x2 r10694, r9542, r9558;
}
{
mul.f16x2 r10697, r10694, r10605;
}
{
add.f16x2 r10700, r10691, r10697;
}
{
add.f16x2 %46, r10685, r10700;
}
{
add.f16x2 r10706, r9520, r9568;
}
{
mul.f16x2 r10709, r10706, r10604;
}
{
add.f16x2 r10712, r7911, r10709;
}
{
add.f16x2 r10715, r9536, r9552;
}
{
mul.f16x2 r10718, r10715, r10606;
}
{
add.f16x2 r10721, r10712, r10718;
}
{
sub.f16x2 r10724, r9526, r9574;
}
{
mul.f16x2 r10727, r10724, r10605;
}
{
sub.f16x2 r10730, r9542, r9558;
}
{
mul.f16x2 r10733, r10730, r10608;
}
{
add.f16x2 r10736, r10727, r10733;
}
{
sub.f16x2 %26, r10721, r10736;
}
{
add.f16x2 r10742, r9520, r9568;
}
{
mul.f16x2 r10745, r10742, r10604;
}
{
add.f16x2 r10748, r7911, r10745;
}
{
add.f16x2 r10751, r9536, r9552;
}
{
mul.f16x2 r10754, r10751, r10606;
}
{
add.f16x2 r10757, r10748, r10754;
}
{
sub.f16x2 r10760, r9526, r9574;
}
{
mul.f16x2 r10763, r10760, r10605;
}
{
sub.f16x2 r10766, r9542, r9558;
}
{
mul.f16x2 r10769, r10766, r10608;
}
{
add.f16x2 r10772, r10763, r10769;
}
{
add.f16x2 %36, r10757, r10772;
}
{
add.f16x2 r10778, r9526, r9574;
}
{
mul.f16x2 r10781, r10778, r10602;
}
{
add.f16x2 r10784, r8055, r10781;
}
{
add.f16x2 r10787, r9542, r9558;
}
{
mul.f16x2 r10790, r10787, r10604;
}
{
add.f16x2 r10793, r10784, r10790;
}
{
sub.f16x2 r10796, r9520, r9568;
}
{
mul.f16x2 r10799, r10796, r10603;
}
{
sub.f16x2 r10802, r9536, r9552;
}
{
mul.f16x2 r10805, r10802, r10605;
}
{
add.f16x2 r10808, r10799, r10805;
}
{
add.f16x2 %17, r10793, r10808;
}
{
add.f16x2 r10814, r9526, r9574;
}
{
mul.f16x2 r10817, r10814, r10602;
}
{
add.f16x2 r10820, r8055, r10817;
}
{
add.f16x2 r10823, r9542, r9558;
}
{
mul.f16x2 r10826, r10823, r10604;
}
{
add.f16x2 r10829, r10820, r10826;
}
{
sub.f16x2 r10832, r9520, r9568;
}
{
mul.f16x2 r10835, r10832, r10603;
}
{
sub.f16x2 r10838, r9536, r9552;
}
{
mul.f16x2 r10841, r10838, r10605;
}
{
add.f16x2 r10844, r10835, r10841;
}
{
sub.f16x2 %47, r10829, r10844;
}
{
add.f16x2 r10850, r9526, r9574;
}
{
mul.f16x2 r10853, r10850, r10604;
}
{
add.f16x2 r10856, r8055, r10853;
}
{
add.f16x2 r10859, r9542, r9558;
}
{
mul.f16x2 r10862, r10859, r10606;
}
{
add.f16x2 r10865, r10856, r10862;
}
{
sub.f16x2 r10868, r9520, r9568;
}
{
mul.f16x2 r10871, r10868, r10605;
}
{
sub.f16x2 r10874, r9536, r9552;
}
{
mul.f16x2 r10877, r10874, r10608;
}
{
add.f16x2 r10880, r10871, r10877;
}
{
add.f16x2 %27, r10865, r10880;
}
{
add.f16x2 r10886, r9526, r9574;
}
{
mul.f16x2 r10889, r10886, r10604;
}
{
add.f16x2 r10892, r8055, r10889;
}
{
add.f16x2 r10895, r9542, r9558;
}
{
mul.f16x2 r10898, r10895, r10606;
}
{
add.f16x2 r10901, r10892, r10898;
}
{
sub.f16x2 r10904, r9520, r9568;
}
{
mul.f16x2 r10907, r10904, r10605;
}
{
sub.f16x2 r10910, r9536, r9552;
}
{
mul.f16x2 r10913, r10910, r10608;
}
{
add.f16x2 r10916, r10907, r10913;
}
{
sub.f16x2 %37, r10901, r10916;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10922, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10923, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r10924, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r10925, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10926, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10927, {low, high};
}
{
neg.f16x2 r10928, r10927;
}
{
add.f16x2 r10930, r9584, r9632;
}
{
add.f16x2 r10933, r7839, r10930;
}
{
add.f16x2 r10936, r9600, r9616;
}
{
add.f16x2 %8, r10933, r10936;
}
{
add.f16x2 r10942, r9590, r9638;
}
{
add.f16x2 r10945, r7983, r10942;
}
{
add.f16x2 r10948, r9606, r9622;
}
{
add.f16x2 %9, r10945, r10948;
}
{
add.f16x2 r10954, r9584, r9632;
}
{
mul.f16x2 r10957, r10954, r10922;
}
{
add.f16x2 r10960, r7839, r10957;
}
{
add.f16x2 r10963, r9600, r9616;
}
{
mul.f16x2 r10966, r10963, r10924;
}
{
add.f16x2 r10969, r10960, r10966;
}
{
sub.f16x2 r10972, r9590, r9638;
}
{
mul.f16x2 r10975, r10972, r10923;
}
{
sub.f16x2 r10978, r9606, r9622;
}
{
mul.f16x2 r10981, r10978, r10925;
}
{
add.f16x2 r10984, r10975, r10981;
}
{
sub.f16x2 %18, r10969, r10984;
}
{
add.f16x2 r10990, r9584, r9632;
}
{
mul.f16x2 r10993, r10990, r10922;
}
{
add.f16x2 r10996, r7839, r10993;
}
{
add.f16x2 r10999, r9600, r9616;
}
{
mul.f16x2 r11002, r10999, r10924;
}
{
add.f16x2 r11005, r10996, r11002;
}
{
sub.f16x2 r11008, r9590, r9638;
}
{
mul.f16x2 r11011, r11008, r10923;
}
{
sub.f16x2 r11014, r9606, r9622;
}
{
mul.f16x2 r11017, r11014, r10925;
}
{
add.f16x2 r11020, r11011, r11017;
}
{
add.f16x2 %48, r11005, r11020;
}
{
add.f16x2 r11026, r9584, r9632;
}
{
mul.f16x2 r11029, r11026, r10924;
}
{
add.f16x2 r11032, r7839, r11029;
}
{
add.f16x2 r11035, r9600, r9616;
}
{
mul.f16x2 r11038, r11035, r10926;
}
{
add.f16x2 r11041, r11032, r11038;
}
{
sub.f16x2 r11044, r9590, r9638;
}
{
mul.f16x2 r11047, r11044, r10925;
}
{
sub.f16x2 r11050, r9606, r9622;
}
{
mul.f16x2 r11053, r11050, r10928;
}
{
add.f16x2 r11056, r11047, r11053;
}
{
sub.f16x2 %28, r11041, r11056;
}
{
add.f16x2 r11062, r9584, r9632;
}
{
mul.f16x2 r11065, r11062, r10924;
}
{
add.f16x2 r11068, r7839, r11065;
}
{
add.f16x2 r11071, r9600, r9616;
}
{
mul.f16x2 r11074, r11071, r10926;
}
{
add.f16x2 r11077, r11068, r11074;
}
{
sub.f16x2 r11080, r9590, r9638;
}
{
mul.f16x2 r11083, r11080, r10925;
}
{
sub.f16x2 r11086, r9606, r9622;
}
{
mul.f16x2 r11089, r11086, r10928;
}
{
add.f16x2 r11092, r11083, r11089;
}
{
add.f16x2 %38, r11077, r11092;
}
{
add.f16x2 r11098, r9590, r9638;
}
{
mul.f16x2 r11101, r11098, r10922;
}
{
add.f16x2 r11104, r7983, r11101;
}
{
add.f16x2 r11107, r9606, r9622;
}
{
mul.f16x2 r11110, r11107, r10924;
}
{
add.f16x2 r11113, r11104, r11110;
}
{
sub.f16x2 r11116, r9584, r9632;
}
{
mul.f16x2 r11119, r11116, r10923;
}
{
sub.f16x2 r11122, r9600, r9616;
}
{
mul.f16x2 r11125, r11122, r10925;
}
{
add.f16x2 r11128, r11119, r11125;
}
{
add.f16x2 %19, r11113, r11128;
}
{
add.f16x2 r11134, r9590, r9638;
}
{
mul.f16x2 r11137, r11134, r10922;
}
{
add.f16x2 r11140, r7983, r11137;
}
{
add.f16x2 r11143, r9606, r9622;
}
{
mul.f16x2 r11146, r11143, r10924;
}
{
add.f16x2 r11149, r11140, r11146;
}
{
sub.f16x2 r11152, r9584, r9632;
}
{
mul.f16x2 r11155, r11152, r10923;
}
{
sub.f16x2 r11158, r9600, r9616;
}
{
mul.f16x2 r11161, r11158, r10925;
}
{
add.f16x2 r11164, r11155, r11161;
}
{
sub.f16x2 %49, r11149, r11164;
}
{
add.f16x2 r11170, r9590, r9638;
}
{
mul.f16x2 r11173, r11170, r10924;
}
{
add.f16x2 r11176, r7983, r11173;
}
{
add.f16x2 r11179, r9606, r9622;
}
{
mul.f16x2 r11182, r11179, r10926;
}
{
add.f16x2 r11185, r11176, r11182;
}
{
sub.f16x2 r11188, r9584, r9632;
}
{
mul.f16x2 r11191, r11188, r10925;
}
{
sub.f16x2 r11194, r9600, r9616;
}
{
mul.f16x2 r11197, r11194, r10928;
}
{
add.f16x2 r11200, r11191, r11197;
}
{
add.f16x2 %29, r11185, r11200;
}
{
add.f16x2 r11206, r9590, r9638;
}
{
mul.f16x2 r11209, r11206, r10924;
}
{
add.f16x2 r11212, r7983, r11209;
}
{
add.f16x2 r11215, r9606, r9622;
}
{
mul.f16x2 r11218, r11215, r10926;
}
{
add.f16x2 r11221, r11212, r11218;
}
{
sub.f16x2 r11224, r9584, r9632;
}
{
mul.f16x2 r11227, r11224, r10925;
}
{
sub.f16x2 r11230, r9600, r9616;
}
{
mul.f16x2 r11233, r11230, r10928;
}
{
add.f16x2 r11236, r11227, r11233;
}
{
sub.f16x2 %39, r11221, r11236;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[24].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<916, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.local .align 8 .b8 __local_depot1[200];
.reg .b64 SP;
.reg .b64 SPL;
.reg .pred p<3>;
.reg .f32 f<695>;
.reg .b32 r<11254>;
.reg .b64 rd<19>;
mov.u64 SPL, __local_depot1;
add.u64 rd3, SPL, 0;
mov.u32 r3531, %tid.y;
mul.lo.s32 r1, r3531, 15625;
add.s64 rd4, rd3, 4;
mov.f32 f214, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r23, {low, high};
}
mov.f32 f216, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r24, {low, high};
}
mov.f32 f210, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r25, {low, high};
}
mov.f32 f212, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r26, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r27, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r28, {low, high};
}
{
neg.f16x2 r29, r28;
}
{
add.f16x2 r31, %61, %91;
}
{
add.f16x2 r34, %51, r31;
}
{
add.f16x2 r37, %71, %81;
}
{
add.f16x2 r40, r34, r37;
}
{
add.f16x2 r43, %62, %92;
}
{
add.f16x2 r46, %52, r43;
}
{
add.f16x2 r49, %72, %82;
}
{
add.f16x2 r52, r46, r49;
}
{
add.f16x2 r55, %61, %91;
}
{
mul.f16x2 r58, r55, r23;
}
{
add.f16x2 r61, %51, r58;
}
{
add.f16x2 r64, %71, %81;
}
{
mul.f16x2 r67, r64, r25;
}
{
add.f16x2 r70, r61, r67;
}
{
sub.f16x2 r73, %62, %92;
}
{
mul.f16x2 r76, r73, r24;
}
{
sub.f16x2 r79, %72, %82;
}
{
mul.f16x2 r82, r79, r26;
}
{
add.f16x2 r85, r76, r82;
}
{
sub.f16x2 r88, r70, r85;
}
{
add.f16x2 r91, %61, %91;
}
{
mul.f16x2 r94, r91, r23;
}
{
add.f16x2 r97, %51, r94;
}
{
add.f16x2 r100, %71, %81;
}
{
mul.f16x2 r103, r100, r25;
}
{
add.f16x2 r106, r97, r103;
}
{
sub.f16x2 r109, %62, %92;
}
{
mul.f16x2 r112, r109, r24;
}
{
sub.f16x2 r115, %72, %82;
}
{
mul.f16x2 r118, r115, r26;
}
{
add.f16x2 r121, r112, r118;
}
{
add.f16x2 r124, r106, r121;
}
{
add.f16x2 r127, %61, %91;
}
{
mul.f16x2 r130, r127, r25;
}
{
add.f16x2 r133, %51, r130;
}
{
add.f16x2 r136, %71, %81;
}
{
mul.f16x2 r139, r136, r27;
}
{
add.f16x2 r142, r133, r139;
}
{
sub.f16x2 r145, %62, %92;
}
{
mul.f16x2 r148, r145, r26;
}
{
sub.f16x2 r151, %72, %82;
}
{
mul.f16x2 r154, r151, r29;
}
{
add.f16x2 r157, r148, r154;
}
{
sub.f16x2 r160, r142, r157;
}
{
add.f16x2 r163, %61, %91;
}
{
mul.f16x2 r166, r163, r25;
}
{
add.f16x2 r169, %51, r166;
}
{
add.f16x2 r172, %71, %81;
}
{
mul.f16x2 r175, r172, r27;
}
{
add.f16x2 r178, r169, r175;
}
{
sub.f16x2 r181, %62, %92;
}
{
mul.f16x2 r184, r181, r26;
}
{
sub.f16x2 r187, %72, %82;
}
{
mul.f16x2 r190, r187, r29;
}
{
add.f16x2 r193, r184, r190;
}
{
add.f16x2 r196, r178, r193;
}
{
add.f16x2 r199, %62, %92;
}
{
mul.f16x2 r202, r199, r23;
}
{
add.f16x2 r205, %52, r202;
}
{
add.f16x2 r208, %72, %82;
}
{
mul.f16x2 r211, r208, r25;
}
{
add.f16x2 r214, r205, r211;
}
{
sub.f16x2 r217, %61, %91;
}
{
mul.f16x2 r220, r217, r24;
}
{
sub.f16x2 r223, %71, %81;
}
{
mul.f16x2 r226, r223, r26;
}
{
add.f16x2 r229, r220, r226;
}
{
add.f16x2 r232, r214, r229;
}
{
add.f16x2 r235, %62, %92;
}
{
mul.f16x2 r238, r235, r23;
}
{
add.f16x2 r241, %52, r238;
}
{
add.f16x2 r244, %72, %82;
}
{
mul.f16x2 r247, r244, r25;
}
{
add.f16x2 r250, r241, r247;
}
{
sub.f16x2 r253, %61, %91;
}
{
mul.f16x2 r256, r253, r24;
}
{
sub.f16x2 r259, %71, %81;
}
{
mul.f16x2 r262, r259, r26;
}
{
add.f16x2 r265, r256, r262;
}
{
sub.f16x2 r268, r250, r265;
}
{
add.f16x2 r271, %62, %92;
}
{
mul.f16x2 r274, r271, r25;
}
{
add.f16x2 r277, %52, r274;
}
{
add.f16x2 r280, %72, %82;
}
{
mul.f16x2 r283, r280, r27;
}
{
add.f16x2 r286, r277, r283;
}
{
sub.f16x2 r289, %61, %91;
}
{
mul.f16x2 r292, r289, r26;
}
{
sub.f16x2 r295, %71, %81;
}
{
mul.f16x2 r298, r295, r29;
}
{
add.f16x2 r301, r292, r298;
}
{
add.f16x2 r304, r286, r301;
}
{
add.f16x2 r307, %62, %92;
}
{
mul.f16x2 r310, r307, r25;
}
{
add.f16x2 r313, %52, r310;
}
{
add.f16x2 r316, %72, %82;
}
{
mul.f16x2 r319, r316, r27;
}
{
add.f16x2 r322, r313, r319;
}
{
sub.f16x2 r325, %61, %91;
}
{
mul.f16x2 r328, r325, r26;
}
{
sub.f16x2 r331, %71, %81;
}
{
mul.f16x2 r334, r331, r29;
}
{
add.f16x2 r337, r328, r334;
}
{
sub.f16x2 r340, r322, r337;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r343, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r344, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r345, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r346, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r347, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r348, {low, high};
}
{
neg.f16x2 r349, r348;
}
{
add.f16x2 r351, %63, %93;
}
{
add.f16x2 r354, %53, r351;
}
{
add.f16x2 r357, %73, %83;
}
{
add.f16x2 r360, r354, r357;
}
{
add.f16x2 r363, %64, %94;
}
{
add.f16x2 r366, %54, r363;
}
{
add.f16x2 r369, %74, %84;
}
{
add.f16x2 r372, r366, r369;
}
{
add.f16x2 r375, %63, %93;
}
{
mul.f16x2 r378, r375, r343;
}
{
add.f16x2 r381, %53, r378;
}
{
add.f16x2 r384, %73, %83;
}
{
mul.f16x2 r387, r384, r345;
}
{
add.f16x2 r390, r381, r387;
}
{
sub.f16x2 r393, %64, %94;
}
{
mul.f16x2 r396, r393, r344;
}
{
sub.f16x2 r399, %74, %84;
}
{
mul.f16x2 r402, r399, r346;
}
{
add.f16x2 r405, r396, r402;
}
{
sub.f16x2 r408, r390, r405;
}
{
add.f16x2 r411, %63, %93;
}
{
mul.f16x2 r414, r411, r343;
}
{
add.f16x2 r417, %53, r414;
}
{
add.f16x2 r420, %73, %83;
}
{
mul.f16x2 r423, r420, r345;
}
{
add.f16x2 r426, r417, r423;
}
{
sub.f16x2 r429, %64, %94;
}
{
mul.f16x2 r432, r429, r344;
}
{
sub.f16x2 r435, %74, %84;
}
{
mul.f16x2 r438, r435, r346;
}
{
add.f16x2 r441, r432, r438;
}
{
add.f16x2 r444, r426, r441;
}
{
add.f16x2 r447, %63, %93;
}
{
mul.f16x2 r450, r447, r345;
}
{
add.f16x2 r453, %53, r450;
}
{
add.f16x2 r456, %73, %83;
}
{
mul.f16x2 r459, r456, r347;
}
{
add.f16x2 r462, r453, r459;
}
{
sub.f16x2 r465, %64, %94;
}
{
mul.f16x2 r468, r465, r346;
}
{
sub.f16x2 r471, %74, %84;
}
{
mul.f16x2 r474, r471, r349;
}
{
add.f16x2 r477, r468, r474;
}
{
sub.f16x2 r480, r462, r477;
}
{
add.f16x2 r483, %63, %93;
}
{
mul.f16x2 r486, r483, r345;
}
{
add.f16x2 r489, %53, r486;
}
{
add.f16x2 r492, %73, %83;
}
{
mul.f16x2 r495, r492, r347;
}
{
add.f16x2 r498, r489, r495;
}
{
sub.f16x2 r501, %64, %94;
}
{
mul.f16x2 r504, r501, r346;
}
{
sub.f16x2 r507, %74, %84;
}
{
mul.f16x2 r510, r507, r349;
}
{
add.f16x2 r513, r504, r510;
}
{
add.f16x2 r516, r498, r513;
}
{
add.f16x2 r519, %64, %94;
}
{
mul.f16x2 r522, r519, r343;
}
{
add.f16x2 r525, %54, r522;
}
{
add.f16x2 r528, %74, %84;
}
{
mul.f16x2 r531, r528, r345;
}
{
add.f16x2 r534, r525, r531;
}
{
sub.f16x2 r537, %63, %93;
}
{
mul.f16x2 r540, r537, r344;
}
{
sub.f16x2 r543, %73, %83;
}
{
mul.f16x2 r546, r543, r346;
}
{
add.f16x2 r549, r540, r546;
}
{
add.f16x2 r552, r534, r549;
}
{
add.f16x2 r555, %64, %94;
}
{
mul.f16x2 r558, r555, r343;
}
{
add.f16x2 r561, %54, r558;
}
{
add.f16x2 r564, %74, %84;
}
{
mul.f16x2 r567, r564, r345;
}
{
add.f16x2 r570, r561, r567;
}
{
sub.f16x2 r573, %63, %93;
}
{
mul.f16x2 r576, r573, r344;
}
{
sub.f16x2 r579, %73, %83;
}
{
mul.f16x2 r582, r579, r346;
}
{
add.f16x2 r585, r576, r582;
}
{
sub.f16x2 r588, r570, r585;
}
{
add.f16x2 r591, %64, %94;
}
{
mul.f16x2 r594, r591, r345;
}
{
add.f16x2 r597, %54, r594;
}
{
add.f16x2 r600, %74, %84;
}
{
mul.f16x2 r603, r600, r347;
}
{
add.f16x2 r606, r597, r603;
}
{
sub.f16x2 r609, %63, %93;
}
{
mul.f16x2 r612, r609, r346;
}
{
sub.f16x2 r615, %73, %83;
}
{
mul.f16x2 r618, r615, r349;
}
{
add.f16x2 r621, r612, r618;
}
{
add.f16x2 r624, r606, r621;
}
{
add.f16x2 r627, %64, %94;
}
{
mul.f16x2 r630, r627, r345;
}
{
add.f16x2 r633, %54, r630;
}
{
add.f16x2 r636, %74, %84;
}
{
mul.f16x2 r639, r636, r347;
}
{
add.f16x2 r642, r633, r639;
}
{
sub.f16x2 r645, %63, %93;
}
{
mul.f16x2 r648, r645, r346;
}
{
sub.f16x2 r651, %73, %83;
}
{
mul.f16x2 r654, r651, r349;
}
{
add.f16x2 r657, r648, r654;
}
{
sub.f16x2 r660, r642, r657;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r663, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r664, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r665, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r666, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r667, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r668, {low, high};
}
{
neg.f16x2 r669, r668;
}
{
add.f16x2 r671, %65, %95;
}
{
add.f16x2 r674, %55, r671;
}
{
add.f16x2 r677, %75, %85;
}
{
add.f16x2 r680, r674, r677;
}
{
add.f16x2 r683, %66, %96;
}
{
add.f16x2 r686, %56, r683;
}
{
add.f16x2 r689, %76, %86;
}
{
add.f16x2 r692, r686, r689;
}
{
add.f16x2 r695, %65, %95;
}
{
mul.f16x2 r698, r695, r663;
}
{
add.f16x2 r701, %55, r698;
}
{
add.f16x2 r704, %75, %85;
}
{
mul.f16x2 r707, r704, r665;
}
{
add.f16x2 r710, r701, r707;
}
{
sub.f16x2 r713, %66, %96;
}
{
mul.f16x2 r716, r713, r664;
}
{
sub.f16x2 r719, %76, %86;
}
{
mul.f16x2 r722, r719, r666;
}
{
add.f16x2 r725, r716, r722;
}
{
sub.f16x2 r728, r710, r725;
}
{
add.f16x2 r731, %65, %95;
}
{
mul.f16x2 r734, r731, r663;
}
{
add.f16x2 r737, %55, r734;
}
{
add.f16x2 r740, %75, %85;
}
{
mul.f16x2 r743, r740, r665;
}
{
add.f16x2 r746, r737, r743;
}
{
sub.f16x2 r749, %66, %96;
}
{
mul.f16x2 r752, r749, r664;
}
{
sub.f16x2 r755, %76, %86;
}
{
mul.f16x2 r758, r755, r666;
}
{
add.f16x2 r761, r752, r758;
}
{
add.f16x2 r764, r746, r761;
}
{
add.f16x2 r767, %65, %95;
}
{
mul.f16x2 r770, r767, r665;
}
{
add.f16x2 r773, %55, r770;
}
{
add.f16x2 r776, %75, %85;
}
{
mul.f16x2 r779, r776, r667;
}
{
add.f16x2 r782, r773, r779;
}
{
sub.f16x2 r785, %66, %96;
}
{
mul.f16x2 r788, r785, r666;
}
{
sub.f16x2 r791, %76, %86;
}
{
mul.f16x2 r794, r791, r669;
}
{
add.f16x2 r797, r788, r794;
}
{
sub.f16x2 r800, r782, r797;
}
{
add.f16x2 r803, %65, %95;
}
{
mul.f16x2 r806, r803, r665;
}
{
add.f16x2 r809, %55, r806;
}
{
add.f16x2 r812, %75, %85;
}
{
mul.f16x2 r815, r812, r667;
}
{
add.f16x2 r818, r809, r815;
}
{
sub.f16x2 r821, %66, %96;
}
{
mul.f16x2 r824, r821, r666;
}
{
sub.f16x2 r827, %76, %86;
}
{
mul.f16x2 r830, r827, r669;
}
{
add.f16x2 r833, r824, r830;
}
{
add.f16x2 r836, r818, r833;
}
{
add.f16x2 r839, %66, %96;
}
{
mul.f16x2 r842, r839, r663;
}
{
add.f16x2 r845, %56, r842;
}
{
add.f16x2 r848, %76, %86;
}
{
mul.f16x2 r851, r848, r665;
}
{
add.f16x2 r854, r845, r851;
}
{
sub.f16x2 r857, %65, %95;
}
{
mul.f16x2 r860, r857, r664;
}
{
sub.f16x2 r863, %75, %85;
}
{
mul.f16x2 r866, r863, r666;
}
{
add.f16x2 r869, r860, r866;
}
{
add.f16x2 r872, r854, r869;
}
{
add.f16x2 r875, %66, %96;
}
{
mul.f16x2 r878, r875, r663;
}
{
add.f16x2 r881, %56, r878;
}
{
add.f16x2 r884, %76, %86;
}
{
mul.f16x2 r887, r884, r665;
}
{
add.f16x2 r890, r881, r887;
}
{
sub.f16x2 r893, %65, %95;
}
{
mul.f16x2 r896, r893, r664;
}
{
sub.f16x2 r899, %75, %85;
}
{
mul.f16x2 r902, r899, r666;
}
{
add.f16x2 r905, r896, r902;
}
{
sub.f16x2 r908, r890, r905;
}
{
add.f16x2 r911, %66, %96;
}
{
mul.f16x2 r914, r911, r665;
}
{
add.f16x2 r917, %56, r914;
}
{
add.f16x2 r920, %76, %86;
}
{
mul.f16x2 r923, r920, r667;
}
{
add.f16x2 r926, r917, r923;
}
{
sub.f16x2 r929, %65, %95;
}
{
mul.f16x2 r932, r929, r666;
}
{
sub.f16x2 r935, %75, %85;
}
{
mul.f16x2 r938, r935, r669;
}
{
add.f16x2 r941, r932, r938;
}
{
add.f16x2 r944, r926, r941;
}
{
add.f16x2 r947, %66, %96;
}
{
mul.f16x2 r950, r947, r665;
}
{
add.f16x2 r953, %56, r950;
}
{
add.f16x2 r956, %76, %86;
}
{
mul.f16x2 r959, r956, r667;
}
{
add.f16x2 r962, r953, r959;
}
{
sub.f16x2 r965, %65, %95;
}
{
mul.f16x2 r968, r965, r666;
}
{
sub.f16x2 r971, %75, %85;
}
{
mul.f16x2 r974, r971, r669;
}
{
add.f16x2 r977, r968, r974;
}
{
sub.f16x2 r980, r962, r977;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r983, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r984, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r985, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r986, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r987, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r988, {low, high};
}
{
neg.f16x2 r989, r988;
}
{
add.f16x2 r991, %67, %97;
}
{
add.f16x2 r994, %57, r991;
}
{
add.f16x2 r997, %77, %87;
}
{
add.f16x2 r1000, r994, r997;
}
{
add.f16x2 r1003, %68, %98;
}
{
add.f16x2 r1006, %58, r1003;
}
{
add.f16x2 r1009, %78, %88;
}
{
add.f16x2 r1012, r1006, r1009;
}
{
add.f16x2 r1015, %67, %97;
}
{
mul.f16x2 r1018, r1015, r983;
}
{
add.f16x2 r1021, %57, r1018;
}
{
add.f16x2 r1024, %77, %87;
}
{
mul.f16x2 r1027, r1024, r985;
}
{
add.f16x2 r1030, r1021, r1027;
}
{
sub.f16x2 r1033, %68, %98;
}
{
mul.f16x2 r1036, r1033, r984;
}
{
sub.f16x2 r1039, %78, %88;
}
{
mul.f16x2 r1042, r1039, r986;
}
{
add.f16x2 r1045, r1036, r1042;
}
{
sub.f16x2 r1048, r1030, r1045;
}
{
add.f16x2 r1051, %67, %97;
}
{
mul.f16x2 r1054, r1051, r983;
}
{
add.f16x2 r1057, %57, r1054;
}
{
add.f16x2 r1060, %77, %87;
}
{
mul.f16x2 r1063, r1060, r985;
}
{
add.f16x2 r1066, r1057, r1063;
}
{
sub.f16x2 r1069, %68, %98;
}
{
mul.f16x2 r1072, r1069, r984;
}
{
sub.f16x2 r1075, %78, %88;
}
{
mul.f16x2 r1078, r1075, r986;
}
{
add.f16x2 r1081, r1072, r1078;
}
{
add.f16x2 r1084, r1066, r1081;
}
{
add.f16x2 r1087, %67, %97;
}
{
mul.f16x2 r1090, r1087, r985;
}
{
add.f16x2 r1093, %57, r1090;
}
{
add.f16x2 r1096, %77, %87;
}
{
mul.f16x2 r1099, r1096, r987;
}
{
add.f16x2 r1102, r1093, r1099;
}
{
sub.f16x2 r1105, %68, %98;
}
{
mul.f16x2 r1108, r1105, r986;
}
{
sub.f16x2 r1111, %78, %88;
}
{
mul.f16x2 r1114, r1111, r989;
}
{
add.f16x2 r1117, r1108, r1114;
}
{
sub.f16x2 r1120, r1102, r1117;
}
{
add.f16x2 r1123, %67, %97;
}
{
mul.f16x2 r1126, r1123, r985;
}
{
add.f16x2 r1129, %57, r1126;
}
{
add.f16x2 r1132, %77, %87;
}
{
mul.f16x2 r1135, r1132, r987;
}
{
add.f16x2 r1138, r1129, r1135;
}
{
sub.f16x2 r1141, %68, %98;
}
{
mul.f16x2 r1144, r1141, r986;
}
{
sub.f16x2 r1147, %78, %88;
}
{
mul.f16x2 r1150, r1147, r989;
}
{
add.f16x2 r1153, r1144, r1150;
}
{
add.f16x2 r1156, r1138, r1153;
}
{
add.f16x2 r1159, %68, %98;
}
{
mul.f16x2 r1162, r1159, r983;
}
{
add.f16x2 r1165, %58, r1162;
}
{
add.f16x2 r1168, %78, %88;
}
{
mul.f16x2 r1171, r1168, r985;
}
{
add.f16x2 r1174, r1165, r1171;
}
{
sub.f16x2 r1177, %67, %97;
}
{
mul.f16x2 r1180, r1177, r984;
}
{
sub.f16x2 r1183, %77, %87;
}
{
mul.f16x2 r1186, r1183, r986;
}
{
add.f16x2 r1189, r1180, r1186;
}
{
add.f16x2 r1192, r1174, r1189;
}
{
add.f16x2 r1195, %68, %98;
}
{
mul.f16x2 r1198, r1195, r983;
}
{
add.f16x2 r1201, %58, r1198;
}
{
add.f16x2 r1204, %78, %88;
}
{
mul.f16x2 r1207, r1204, r985;
}
{
add.f16x2 r1210, r1201, r1207;
}
{
sub.f16x2 r1213, %67, %97;
}
{
mul.f16x2 r1216, r1213, r984;
}
{
sub.f16x2 r1219, %77, %87;
}
{
mul.f16x2 r1222, r1219, r986;
}
{
add.f16x2 r1225, r1216, r1222;
}
{
sub.f16x2 r1228, r1210, r1225;
}
{
add.f16x2 r1231, %68, %98;
}
{
mul.f16x2 r1234, r1231, r985;
}
{
add.f16x2 r1237, %58, r1234;
}
{
add.f16x2 r1240, %78, %88;
}
{
mul.f16x2 r1243, r1240, r987;
}
{
add.f16x2 r1246, r1237, r1243;
}
{
sub.f16x2 r1249, %67, %97;
}
{
mul.f16x2 r1252, r1249, r986;
}
{
sub.f16x2 r1255, %77, %87;
}
{
mul.f16x2 r1258, r1255, r989;
}
{
add.f16x2 r1261, r1252, r1258;
}
{
add.f16x2 r1264, r1246, r1261;
}
{
add.f16x2 r1267, %68, %98;
}
{
mul.f16x2 r1270, r1267, r985;
}
{
add.f16x2 r1273, %58, r1270;
}
{
add.f16x2 r1276, %78, %88;
}
{
mul.f16x2 r1279, r1276, r987;
}
{
add.f16x2 r1282, r1273, r1279;
}
{
sub.f16x2 r1285, %67, %97;
}
{
mul.f16x2 r1288, r1285, r986;
}
{
sub.f16x2 r1291, %77, %87;
}
{
mul.f16x2 r1294, r1291, r989;
}
{
add.f16x2 r1297, r1288, r1294;
}
{
sub.f16x2 r1300, r1282, r1297;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1303, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1304, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r1305, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r1306, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1307, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1308, {low, high};
}
{
neg.f16x2 r1309, r1308;
}
{
add.f16x2 r1311, %69, %99;
}
{
add.f16x2 r1314, %59, r1311;
}
{
add.f16x2 r1317, %79, %89;
}
{
add.f16x2 r1320, r1314, r1317;
}
{
add.f16x2 r1323, %70, %100;
}
{
add.f16x2 r1326, %60, r1323;
}
{
add.f16x2 r1329, %80, %90;
}
{
add.f16x2 r1332, r1326, r1329;
}
{
add.f16x2 r1335, %69, %99;
}
{
mul.f16x2 r1338, r1335, r1303;
}
{
add.f16x2 r1341, %59, r1338;
}
{
add.f16x2 r1344, %79, %89;
}
{
mul.f16x2 r1347, r1344, r1305;
}
{
add.f16x2 r1350, r1341, r1347;
}
{
sub.f16x2 r1353, %70, %100;
}
{
mul.f16x2 r1356, r1353, r1304;
}
{
sub.f16x2 r1359, %80, %90;
}
{
mul.f16x2 r1362, r1359, r1306;
}
{
add.f16x2 r1365, r1356, r1362;
}
{
sub.f16x2 r1368, r1350, r1365;
}
{
add.f16x2 r1371, %69, %99;
}
{
mul.f16x2 r1374, r1371, r1303;
}
{
add.f16x2 r1377, %59, r1374;
}
{
add.f16x2 r1380, %79, %89;
}
{
mul.f16x2 r1383, r1380, r1305;
}
{
add.f16x2 r1386, r1377, r1383;
}
{
sub.f16x2 r1389, %70, %100;
}
{
mul.f16x2 r1392, r1389, r1304;
}
{
sub.f16x2 r1395, %80, %90;
}
{
mul.f16x2 r1398, r1395, r1306;
}
{
add.f16x2 r1401, r1392, r1398;
}
{
add.f16x2 r1404, r1386, r1401;
}
{
add.f16x2 r1407, %69, %99;
}
{
mul.f16x2 r1410, r1407, r1305;
}
{
add.f16x2 r1413, %59, r1410;
}
{
add.f16x2 r1416, %79, %89;
}
{
mul.f16x2 r1419, r1416, r1307;
}
{
add.f16x2 r1422, r1413, r1419;
}
{
sub.f16x2 r1425, %70, %100;
}
{
mul.f16x2 r1428, r1425, r1306;
}
{
sub.f16x2 r1431, %80, %90;
}
{
mul.f16x2 r1434, r1431, r1309;
}
{
add.f16x2 r1437, r1428, r1434;
}
{
sub.f16x2 r1440, r1422, r1437;
}
{
add.f16x2 r1443, %69, %99;
}
{
mul.f16x2 r1446, r1443, r1305;
}
{
add.f16x2 r1449, %59, r1446;
}
{
add.f16x2 r1452, %79, %89;
}
{
mul.f16x2 r1455, r1452, r1307;
}
{
add.f16x2 r1458, r1449, r1455;
}
{
sub.f16x2 r1461, %70, %100;
}
{
mul.f16x2 r1464, r1461, r1306;
}
{
sub.f16x2 r1467, %80, %90;
}
{
mul.f16x2 r1470, r1467, r1309;
}
{
add.f16x2 r1473, r1464, r1470;
}
{
add.f16x2 r1476, r1458, r1473;
}
{
add.f16x2 r1479, %70, %100;
}
{
mul.f16x2 r1482, r1479, r1303;
}
{
add.f16x2 r1485, %60, r1482;
}
{
add.f16x2 r1488, %80, %90;
}
{
mul.f16x2 r1491, r1488, r1305;
}
{
add.f16x2 r1494, r1485, r1491;
}
{
sub.f16x2 r1497, %69, %99;
}
{
mul.f16x2 r1500, r1497, r1304;
}
{
sub.f16x2 r1503, %79, %89;
}
{
mul.f16x2 r1506, r1503, r1306;
}
{
add.f16x2 r1509, r1500, r1506;
}
{
add.f16x2 r1512, r1494, r1509;
}
{
add.f16x2 r1515, %70, %100;
}
{
mul.f16x2 r1518, r1515, r1303;
}
{
add.f16x2 r1521, %60, r1518;
}
{
add.f16x2 r1524, %80, %90;
}
{
mul.f16x2 r1527, r1524, r1305;
}
{
add.f16x2 r1530, r1521, r1527;
}
{
sub.f16x2 r1533, %69, %99;
}
{
mul.f16x2 r1536, r1533, r1304;
}
{
sub.f16x2 r1539, %79, %89;
}
{
mul.f16x2 r1542, r1539, r1306;
}
{
add.f16x2 r1545, r1536, r1542;
}
{
sub.f16x2 r1548, r1530, r1545;
}
{
add.f16x2 r1551, %70, %100;
}
{
mul.f16x2 r1554, r1551, r1305;
}
{
add.f16x2 r1557, %60, r1554;
}
{
add.f16x2 r1560, %80, %90;
}
{
mul.f16x2 r1563, r1560, r1307;
}
{
add.f16x2 r1566, r1557, r1563;
}
{
sub.f16x2 r1569, %69, %99;
}
{
mul.f16x2 r1572, r1569, r1306;
}
{
sub.f16x2 r1575, %79, %89;
}
{
mul.f16x2 r1578, r1575, r1309;
}
{
add.f16x2 r1581, r1572, r1578;
}
{
add.f16x2 r1584, r1566, r1581;
}
{
add.f16x2 r1587, %70, %100;
}
{
mul.f16x2 r1590, r1587, r1305;
}
{
add.f16x2 r1593, %60, r1590;
}
{
add.f16x2 r1596, %80, %90;
}
{
mul.f16x2 r1599, r1596, r1307;
}
{
add.f16x2 r1602, r1593, r1599;
}
{
sub.f16x2 r1605, %69, %99;
}
{
mul.f16x2 r1608, r1605, r1306;
}
{
sub.f16x2 r1611, %79, %89;
}
{
mul.f16x2 r1614, r1611, r1309;
}
{
add.f16x2 r1617, r1608, r1614;
}
{
sub.f16x2 r1620, r1602, r1617;
}
mov.f32 f62, 0f3F77F511;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f62;
cvt.rn.f16.f32 high, f62;
mov.b32 r1623, {low, high};
}
mov.f32 f64, 0fBE7EA890;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f64;
cvt.rn.f16.f32 high, f64;
mov.b32 r1624, {low, high};
}
mov.f32 f66, 0f3F6055A2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f66;
cvt.rn.f16.f32 high, f66;
mov.b32 r1625, {low, high};
}
mov.f32 f68, 0fBEF6A86B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f68;
cvt.rn.f16.f32 high, f68;
mov.b32 r1626, {low, high};
}
mov.f32 f70, 0f3F3A9DB0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f70;
cvt.rn.f16.f32 high, f70;
mov.b32 r1627, {low, high};
}
mov.f32 f72, 0fBF2F3E7B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f72;
cvt.rn.f16.f32 high, f72;
mov.b32 r1628, {low, high};
}
mov.f32 f74, 0f3F092BF2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r1629, {low, high};
}
mov.f32 f76, 0fBF5825E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r1630, {low, high};
}
mov.f32 f82, 0f3D809851;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r1633, {low, high};
}
mov.f32 f84, 0fBF7F7EAE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r1634, {low, high};
}
mov.f32 f90, 0fBED9FFBE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r1637, {low, high};
}
mov.f32 f92, 0fBF67A2BF;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r1638, {low, high};
}
mov.f32 f122, 0fBF232E38;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r1639, {low, high};
}
mov.f32 f96, 0fBF45405B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r1640, {low, high};
}
mov.f32 f106, 0fBF7DFB3B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1645, {low, high};
}
mov.f32 f108, 0fBE00575B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1646, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r1653, {low, high};
}
mov.f32 f124, 0f3F45405B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r1654, {low, high};
}
{
mul.f16x2 r1671, r408, r1623;
}
{
mul.f16x2 r1674, r552, r1624;
}
{
sub.f16x2 r1677, r1671, r1674;
}
{
mul.f16x2 r1680, r408, r1624;
}
{
fma.rn.f16x2 r1683, r552, r1623, r1680;
}
{
mul.f16x2 r1687, r728, r1625;
}
{
mul.f16x2 r1690, r872, r1626;
}
{
sub.f16x2 r1693, r1687, r1690;
}
{
mul.f16x2 r1696, r728, r1626;
}
{
fma.rn.f16x2 r1699, r872, r1625, r1696;
}
{
mul.f16x2 r1703, r1048, r1627;
}
{
mul.f16x2 r1706, r1192, r1628;
}
{
sub.f16x2 r1709, r1703, r1706;
}
{
mul.f16x2 r1712, r1048, r1628;
}
{
fma.rn.f16x2 r1715, r1192, r1627, r1712;
}
{
mul.f16x2 r1719, r1368, r1629;
}
{
mul.f16x2 r1722, r1512, r1630;
}
{
sub.f16x2 r1725, r1719, r1722;
}
{
mul.f16x2 r1728, r1368, r1630;
}
{
fma.rn.f16x2 r1731, r1512, r1629, r1728;
}
{
mul.f16x2 r1735, r480, r1625;
}
{
mul.f16x2 r1738, r624, r1626;
}
{
sub.f16x2 r1741, r1735, r1738;
}
{
mul.f16x2 r1744, r480, r1626;
}
{
fma.rn.f16x2 r1747, r624, r1625, r1744;
}
{
mul.f16x2 r1751, r800, r1629;
}
{
mul.f16x2 r1754, r944, r1630;
}
{
sub.f16x2 r1757, r1751, r1754;
}
{
mul.f16x2 r1760, r800, r1630;
}
{
fma.rn.f16x2 r1763, r944, r1629, r1760;
}
{
mul.f16x2 r1767, r1120, r1633;
}
{
mul.f16x2 r1770, r1264, r1634;
}
{
sub.f16x2 r1773, r1767, r1770;
}
{
mul.f16x2 r1776, r1120, r1634;
}
{
fma.rn.f16x2 r1779, r1264, r1633, r1776;
}
{
mul.f16x2 r1783, r1440, r1637;
}
{
mul.f16x2 r1786, r1584, r1638;
}
{
sub.f16x2 r1789, r1783, r1786;
}
{
mul.f16x2 r1792, r1440, r1638;
}
{
fma.rn.f16x2 r1795, r1584, r1637, r1792;
}
{
mul.f16x2 r1799, r516, r1627;
}
{
mul.f16x2 r1802, r660, r1628;
}
{
sub.f16x2 r1805, r1799, r1802;
}
{
mul.f16x2 r1808, r516, r1628;
}
{
fma.rn.f16x2 r1811, r660, r1627, r1808;
}
{
mul.f16x2 r1815, r836, r1633;
}
{
mul.f16x2 r1818, r980, r1634;
}
{
sub.f16x2 r1821, r1815, r1818;
}
{
mul.f16x2 r1824, r836, r1634;
}
{
fma.rn.f16x2 r1827, r980, r1633, r1824;
}
{
mul.f16x2 r1831, r1156, r1639;
}
{
mul.f16x2 r1834, r1300, r1640;
}
{
sub.f16x2 r1837, r1831, r1834;
}
{
mul.f16x2 r1840, r1156, r1640;
}
{
fma.rn.f16x2 r1843, r1300, r1639, r1840;
}
{
mul.f16x2 r1847, r1476, r1645;
}
{
mul.f16x2 r1850, r1620, r1646;
}
{
sub.f16x2 r1853, r1847, r1850;
}
{
mul.f16x2 r1856, r1476, r1646;
}
{
fma.rn.f16x2 r1859, r1620, r1645, r1856;
}
{
mul.f16x2 r1863, r444, r1629;
}
{
mul.f16x2 r1866, r588, r1630;
}
{
sub.f16x2 r1869, r1863, r1866;
}
{
mul.f16x2 r1872, r444, r1630;
}
{
fma.rn.f16x2 r1875, r588, r1629, r1872;
}
{
mul.f16x2 r1879, r764, r1637;
}
{
mul.f16x2 r1882, r908, r1638;
}
{
sub.f16x2 r1885, r1879, r1882;
}
{
mul.f16x2 r1888, r764, r1638;
}
{
fma.rn.f16x2 r1891, r908, r1637, r1888;
}
{
mul.f16x2 r1895, r1084, r1645;
}
{
mul.f16x2 r1898, r1228, r1646;
}
{
sub.f16x2 r1901, r1895, r1898;
}
{
mul.f16x2 r1904, r1084, r1646;
}
{
fma.rn.f16x2 r1907, r1228, r1645, r1904;
}
{
mul.f16x2 r1911, r1404, r1653;
}
{
mul.f16x2 r1914, r1548, r1654;
}
{
sub.f16x2 r1917, r1911, r1914;
}
{
mul.f16x2 r1920, r1404, r1654;
}
{
fma.rn.f16x2 r1923, r1548, r1653, r1920;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1927, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1928, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r1929, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r1930, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1931, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1932, {low, high};
}
{
neg.f16x2 r1933, r1932;
}
{
add.f16x2 r1935, r360, r1320;
}
{
add.f16x2 r1938, r40, r1935;
}
{
add.f16x2 r1941, r680, r1000;
}
{
add.f16x2 r1944, r1938, r1941;
}
st.local.u32 [rd3], r1944;
{
add.f16x2 r1947, r372, r1332;
}
{
add.f16x2 r1950, r52, r1947;
}
{
add.f16x2 r1953, r692, r1012;
}
{
add.f16x2 r1956, r1950, r1953;
}
st.local.u32 [rd3+4], r1956;
{
add.f16x2 r1959, r360, r1320;
}
{
mul.f16x2 r1962, r1959, r1927;
}
{
add.f16x2 r1965, r40, r1962;
}
{
add.f16x2 r1968, r680, r1000;
}
{
mul.f16x2 r1971, r1968, r1929;
}
{
add.f16x2 r1974, r1965, r1971;
}
{
sub.f16x2 r1977, r372, r1332;
}
{
mul.f16x2 r1980, r1977, r1928;
}
{
sub.f16x2 r1983, r692, r1012;
}
{
mul.f16x2 r1986, r1983, r1930;
}
{
add.f16x2 r1989, r1980, r1986;
}
{
sub.f16x2 r1992, r1974, r1989;
}
st.local.u32 [rd3+40], r1992;
{
add.f16x2 r1995, r360, r1320;
}
{
mul.f16x2 r1998, r1995, r1927;
}
{
add.f16x2 r2001, r40, r1998;
}
{
add.f16x2 r2004, r680, r1000;
}
{
mul.f16x2 r2007, r2004, r1929;
}
{
add.f16x2 r2010, r2001, r2007;
}
{
sub.f16x2 r2013, r372, r1332;
}
{
mul.f16x2 r2016, r2013, r1928;
}
{
sub.f16x2 r2019, r692, r1012;
}
{
mul.f16x2 r2022, r2019, r1930;
}
{
add.f16x2 r2025, r2016, r2022;
}
{
add.f16x2 r2028, r2010, r2025;
}
st.local.u32 [rd3+160], r2028;
{
add.f16x2 r2031, r360, r1320;
}
{
mul.f16x2 r2034, r2031, r1929;
}
{
add.f16x2 r2037, r40, r2034;
}
{
add.f16x2 r2040, r680, r1000;
}
{
mul.f16x2 r2043, r2040, r1931;
}
{
add.f16x2 r2046, r2037, r2043;
}
{
sub.f16x2 r2049, r372, r1332;
}
{
mul.f16x2 r2052, r2049, r1930;
}
{
sub.f16x2 r2055, r692, r1012;
}
{
mul.f16x2 r2058, r2055, r1933;
}
{
add.f16x2 r2061, r2052, r2058;
}
{
sub.f16x2 r2064, r2046, r2061;
}
st.local.u32 [rd3+80], r2064;
{
add.f16x2 r2067, r360, r1320;
}
{
mul.f16x2 r2070, r2067, r1929;
}
{
add.f16x2 r2073, r40, r2070;
}
{
add.f16x2 r2076, r680, r1000;
}
{
mul.f16x2 r2079, r2076, r1931;
}
{
add.f16x2 r2082, r2073, r2079;
}
{
sub.f16x2 r2085, r372, r1332;
}
{
mul.f16x2 r2088, r2085, r1930;
}
{
sub.f16x2 r2091, r692, r1012;
}
{
mul.f16x2 r2094, r2091, r1933;
}
{
add.f16x2 r2097, r2088, r2094;
}
{
add.f16x2 r2100, r2082, r2097;
}
st.local.u32 [rd3+120], r2100;
{
add.f16x2 r2103, r372, r1332;
}
{
mul.f16x2 r2106, r2103, r1927;
}
{
add.f16x2 r2109, r52, r2106;
}
{
add.f16x2 r2112, r692, r1012;
}
{
mul.f16x2 r2115, r2112, r1929;
}
{
add.f16x2 r2118, r2109, r2115;
}
{
sub.f16x2 r2121, r360, r1320;
}
{
mul.f16x2 r2124, r2121, r1928;
}
{
sub.f16x2 r2127, r680, r1000;
}
{
mul.f16x2 r2130, r2127, r1930;
}
{
add.f16x2 r2133, r2124, r2130;
}
{
add.f16x2 r2136, r2118, r2133;
}
st.local.u32 [rd3+44], r2136;
{
add.f16x2 r2139, r372, r1332;
}
{
mul.f16x2 r2142, r2139, r1927;
}
{
add.f16x2 r2145, r52, r2142;
}
{
add.f16x2 r2148, r692, r1012;
}
{
mul.f16x2 r2151, r2148, r1929;
}
{
add.f16x2 r2154, r2145, r2151;
}
{
sub.f16x2 r2157, r360, r1320;
}
{
mul.f16x2 r2160, r2157, r1928;
}
{
sub.f16x2 r2163, r680, r1000;
}
{
mul.f16x2 r2166, r2163, r1930;
}
{
add.f16x2 r2169, r2160, r2166;
}
{
sub.f16x2 r2172, r2154, r2169;
}
st.local.u32 [rd3+164], r2172;
{
add.f16x2 r2175, r372, r1332;
}
{
mul.f16x2 r2178, r2175, r1929;
}
{
add.f16x2 r2181, r52, r2178;
}
{
add.f16x2 r2184, r692, r1012;
}
{
mul.f16x2 r2187, r2184, r1931;
}
{
add.f16x2 r2190, r2181, r2187;
}
{
sub.f16x2 r2193, r360, r1320;
}
{
mul.f16x2 r2196, r2193, r1930;
}
{
sub.f16x2 r2199, r680, r1000;
}
{
mul.f16x2 r2202, r2199, r1933;
}
{
add.f16x2 r2205, r2196, r2202;
}
{
add.f16x2 r2208, r2190, r2205;
}
st.local.u32 [rd3+84], r2208;
{
add.f16x2 r2211, r372, r1332;
}
{
mul.f16x2 r2214, r2211, r1929;
}
{
add.f16x2 r2217, r52, r2214;
}
{
add.f16x2 r2220, r692, r1012;
}
{
mul.f16x2 r2223, r2220, r1931;
}
{
add.f16x2 r2226, r2217, r2223;
}
{
sub.f16x2 r2229, r360, r1320;
}
{
mul.f16x2 r2232, r2229, r1930;
}
{
sub.f16x2 r2235, r680, r1000;
}
{
mul.f16x2 r2238, r2235, r1933;
}
{
add.f16x2 r2241, r2232, r2238;
}
{
sub.f16x2 r2244, r2226, r2241;
}
st.local.u32 [rd3+124], r2244;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2247, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2248, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r2249, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r2250, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2251, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2252, {low, high};
}
{
neg.f16x2 r2253, r2252;
}
{
add.f16x2 r2255, r1677, r1725;
}
{
add.f16x2 r2258, r88, r2255;
}
{
add.f16x2 r2261, r1693, r1709;
}
{
add.f16x2 r11248, r2258, r2261;
}
st.local.u32 [rd3+8], r11248;
{
add.f16x2 r2267, r1683, r1731;
}
{
add.f16x2 r2270, r232, r2267;
}
{
add.f16x2 r2273, r1699, r1715;
}
{
add.f16x2 r2276, r2270, r2273;
}
st.local.u32 [rd3+12], r2276;
{
add.f16x2 r2279, r1677, r1725;
}
{
mul.f16x2 r2282, r2279, r2247;
}
{
add.f16x2 r2285, r88, r2282;
}
{
add.f16x2 r2288, r1693, r1709;
}
{
mul.f16x2 r2291, r2288, r2249;
}
{
add.f16x2 r2294, r2285, r2291;
}
{
sub.f16x2 r2297, r1683, r1731;
}
{
mul.f16x2 r2300, r2297, r2248;
}
{
sub.f16x2 r2303, r1699, r1715;
}
{
mul.f16x2 r2306, r2303, r2250;
}
{
add.f16x2 r2309, r2300, r2306;
}
{
sub.f16x2 r2312, r2294, r2309;
}
st.local.u32 [rd3+48], r2312;
{
add.f16x2 r2315, r1677, r1725;
}
{
mul.f16x2 r2318, r2315, r2247;
}
{
add.f16x2 r2321, r88, r2318;
}
{
add.f16x2 r2324, r1693, r1709;
}
{
mul.f16x2 r2327, r2324, r2249;
}
{
add.f16x2 r2330, r2321, r2327;
}
{
sub.f16x2 r2333, r1683, r1731;
}
{
mul.f16x2 r2336, r2333, r2248;
}
{
sub.f16x2 r2339, r1699, r1715;
}
{
mul.f16x2 r2342, r2339, r2250;
}
{
add.f16x2 r2345, r2336, r2342;
}
{
add.f16x2 r2348, r2330, r2345;
}
st.local.u32 [rd3+168], r2348;
{
add.f16x2 r2351, r1677, r1725;
}
{
mul.f16x2 r2354, r2351, r2249;
}
{
add.f16x2 r2357, r88, r2354;
}
{
add.f16x2 r2360, r1693, r1709;
}
{
mul.f16x2 r2363, r2360, r2251;
}
{
add.f16x2 r2366, r2357, r2363;
}
{
sub.f16x2 r2369, r1683, r1731;
}
{
mul.f16x2 r2372, r2369, r2250;
}
{
sub.f16x2 r2375, r1699, r1715;
}
{
mul.f16x2 r2378, r2375, r2253;
}
{
add.f16x2 r2381, r2372, r2378;
}
{
sub.f16x2 r2384, r2366, r2381;
}
st.local.u32 [rd3+88], r2384;
{
add.f16x2 r2387, r1677, r1725;
}
{
mul.f16x2 r2390, r2387, r2249;
}
{
add.f16x2 r2393, r88, r2390;
}
{
add.f16x2 r2396, r1693, r1709;
}
{
mul.f16x2 r2399, r2396, r2251;
}
{
add.f16x2 r2402, r2393, r2399;
}
{
sub.f16x2 r2405, r1683, r1731;
}
{
mul.f16x2 r2408, r2405, r2250;
}
{
sub.f16x2 r2411, r1699, r1715;
}
{
mul.f16x2 r2414, r2411, r2253;
}
{
add.f16x2 r2417, r2408, r2414;
}
{
add.f16x2 r2420, r2402, r2417;
}
st.local.u32 [rd3+128], r2420;
{
add.f16x2 r2423, r1683, r1731;
}
{
mul.f16x2 r2426, r2423, r2247;
}
{
add.f16x2 r2429, r232, r2426;
}
{
add.f16x2 r2432, r1699, r1715;
}
{
mul.f16x2 r2435, r2432, r2249;
}
{
add.f16x2 r2438, r2429, r2435;
}
{
sub.f16x2 r2441, r1677, r1725;
}
{
mul.f16x2 r2444, r2441, r2248;
}
{
sub.f16x2 r2447, r1693, r1709;
}
{
mul.f16x2 r2450, r2447, r2250;
}
{
add.f16x2 r2453, r2444, r2450;
}
{
add.f16x2 r2456, r2438, r2453;
}
st.local.u32 [rd3+52], r2456;
{
add.f16x2 r2459, r1683, r1731;
}
{
mul.f16x2 r2462, r2459, r2247;
}
{
add.f16x2 r2465, r232, r2462;
}
{
add.f16x2 r2468, r1699, r1715;
}
{
mul.f16x2 r2471, r2468, r2249;
}
{
add.f16x2 r2474, r2465, r2471;
}
{
sub.f16x2 r2477, r1677, r1725;
}
{
mul.f16x2 r2480, r2477, r2248;
}
{
sub.f16x2 r2483, r1693, r1709;
}
{
mul.f16x2 r2486, r2483, r2250;
}
{
add.f16x2 r2489, r2480, r2486;
}
{
sub.f16x2 r2492, r2474, r2489;
}
st.local.u32 [rd3+172], r2492;
{
add.f16x2 r2495, r1683, r1731;
}
{
mul.f16x2 r2498, r2495, r2249;
}
{
add.f16x2 r2501, r232, r2498;
}
{
add.f16x2 r2504, r1699, r1715;
}
{
mul.f16x2 r2507, r2504, r2251;
}
{
add.f16x2 r2510, r2501, r2507;
}
{
sub.f16x2 r2513, r1677, r1725;
}
{
mul.f16x2 r2516, r2513, r2250;
}
{
sub.f16x2 r2519, r1693, r1709;
}
{
mul.f16x2 r2522, r2519, r2253;
}
{
add.f16x2 r2525, r2516, r2522;
}
{
add.f16x2 r2528, r2510, r2525;
}
st.local.u32 [rd3+92], r2528;
{
add.f16x2 r2531, r1683, r1731;
}
{
mul.f16x2 r2534, r2531, r2249;
}
{
add.f16x2 r2537, r232, r2534;
}
{
add.f16x2 r2540, r1699, r1715;
}
{
mul.f16x2 r2543, r2540, r2251;
}
{
add.f16x2 r2546, r2537, r2543;
}
{
sub.f16x2 r2549, r1677, r1725;
}
{
mul.f16x2 r2552, r2549, r2250;
}
{
sub.f16x2 r2555, r1693, r1709;
}
{
mul.f16x2 r2558, r2555, r2253;
}
{
add.f16x2 r2561, r2552, r2558;
}
{
sub.f16x2 r2564, r2546, r2561;
}
st.local.u32 [rd3+132], r2564;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2567, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2568, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r2569, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r2570, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2571, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2572, {low, high};
}
{
neg.f16x2 r2573, r2572;
}
{
add.f16x2 r2575, r1741, r1789;
}
{
add.f16x2 r2578, r160, r2575;
}
{
add.f16x2 r2581, r1757, r1773;
}
{
add.f16x2 r2584, r2578, r2581;
}
st.local.u32 [rd3+16], r2584;
{
add.f16x2 r2587, r1747, r1795;
}
{
add.f16x2 r2590, r304, r2587;
}
{
add.f16x2 r2593, r1763, r1779;
}
{
add.f16x2 r2596, r2590, r2593;
}
st.local.u32 [rd3+20], r2596;
{
add.f16x2 r2599, r1741, r1789;
}
{
mul.f16x2 r2602, r2599, r2567;
}
{
add.f16x2 r2605, r160, r2602;
}
{
add.f16x2 r2608, r1757, r1773;
}
{
mul.f16x2 r2611, r2608, r2569;
}
{
add.f16x2 r2614, r2605, r2611;
}
{
sub.f16x2 r2617, r1747, r1795;
}
{
mul.f16x2 r2620, r2617, r2568;
}
{
sub.f16x2 r2623, r1763, r1779;
}
{
mul.f16x2 r2626, r2623, r2570;
}
{
add.f16x2 r2629, r2620, r2626;
}
{
sub.f16x2 r2632, r2614, r2629;
}
st.local.u32 [rd3+56], r2632;
{
add.f16x2 r2635, r1741, r1789;
}
{
mul.f16x2 r2638, r2635, r2567;
}
{
add.f16x2 r2641, r160, r2638;
}
{
add.f16x2 r2644, r1757, r1773;
}
{
mul.f16x2 r2647, r2644, r2569;
}
{
add.f16x2 r2650, r2641, r2647;
}
{
sub.f16x2 r2653, r1747, r1795;
}
{
mul.f16x2 r2656, r2653, r2568;
}
{
sub.f16x2 r2659, r1763, r1779;
}
{
mul.f16x2 r2662, r2659, r2570;
}
{
add.f16x2 r2665, r2656, r2662;
}
{
add.f16x2 r2668, r2650, r2665;
}
st.local.u32 [rd3+176], r2668;
{
add.f16x2 r2671, r1741, r1789;
}
{
mul.f16x2 r2674, r2671, r2569;
}
{
add.f16x2 r2677, r160, r2674;
}
{
add.f16x2 r2680, r1757, r1773;
}
{
mul.f16x2 r2683, r2680, r2571;
}
{
add.f16x2 r2686, r2677, r2683;
}
{
sub.f16x2 r2689, r1747, r1795;
}
{
mul.f16x2 r2692, r2689, r2570;
}
{
sub.f16x2 r2695, r1763, r1779;
}
{
mul.f16x2 r2698, r2695, r2573;
}
{
add.f16x2 r2701, r2692, r2698;
}
{
sub.f16x2 r2704, r2686, r2701;
}
st.local.u32 [rd3+96], r2704;
{
add.f16x2 r2707, r1741, r1789;
}
{
mul.f16x2 r2710, r2707, r2569;
}
{
add.f16x2 r2713, r160, r2710;
}
{
add.f16x2 r2716, r1757, r1773;
}
{
mul.f16x2 r2719, r2716, r2571;
}
{
add.f16x2 r2722, r2713, r2719;
}
{
sub.f16x2 r2725, r1747, r1795;
}
{
mul.f16x2 r2728, r2725, r2570;
}
{
sub.f16x2 r2731, r1763, r1779;
}
{
mul.f16x2 r2734, r2731, r2573;
}
{
add.f16x2 r2737, r2728, r2734;
}
{
add.f16x2 r2740, r2722, r2737;
}
st.local.u32 [rd3+136], r2740;
{
add.f16x2 r2743, r1747, r1795;
}
{
mul.f16x2 r2746, r2743, r2567;
}
{
add.f16x2 r2749, r304, r2746;
}
{
add.f16x2 r2752, r1763, r1779;
}
{
mul.f16x2 r2755, r2752, r2569;
}
{
add.f16x2 r2758, r2749, r2755;
}
{
sub.f16x2 r2761, r1741, r1789;
}
{
mul.f16x2 r2764, r2761, r2568;
}
{
sub.f16x2 r2767, r1757, r1773;
}
{
mul.f16x2 r2770, r2767, r2570;
}
{
add.f16x2 r2773, r2764, r2770;
}
{
add.f16x2 r2776, r2758, r2773;
}
st.local.u32 [rd3+60], r2776;
{
add.f16x2 r2779, r1747, r1795;
}
{
mul.f16x2 r2782, r2779, r2567;
}
{
add.f16x2 r2785, r304, r2782;
}
{
add.f16x2 r2788, r1763, r1779;
}
{
mul.f16x2 r2791, r2788, r2569;
}
{
add.f16x2 r2794, r2785, r2791;
}
{
sub.f16x2 r2797, r1741, r1789;
}
{
mul.f16x2 r2800, r2797, r2568;
}
{
sub.f16x2 r2803, r1757, r1773;
}
{
mul.f16x2 r2806, r2803, r2570;
}
{
add.f16x2 r2809, r2800, r2806;
}
{
sub.f16x2 r2812, r2794, r2809;
}
st.local.u32 [rd3+180], r2812;
{
add.f16x2 r2815, r1747, r1795;
}
{
mul.f16x2 r2818, r2815, r2569;
}
{
add.f16x2 r2821, r304, r2818;
}
{
add.f16x2 r2824, r1763, r1779;
}
{
mul.f16x2 r2827, r2824, r2571;
}
{
add.f16x2 r2830, r2821, r2827;
}
{
sub.f16x2 r2833, r1741, r1789;
}
{
mul.f16x2 r2836, r2833, r2570;
}
{
sub.f16x2 r2839, r1757, r1773;
}
{
mul.f16x2 r2842, r2839, r2573;
}
{
add.f16x2 r2845, r2836, r2842;
}
{
add.f16x2 r2848, r2830, r2845;
}
st.local.u32 [rd3+100], r2848;
{
add.f16x2 r2851, r1747, r1795;
}
{
mul.f16x2 r2854, r2851, r2569;
}
{
add.f16x2 r2857, r304, r2854;
}
{
add.f16x2 r2860, r1763, r1779;
}
{
mul.f16x2 r2863, r2860, r2571;
}
{
add.f16x2 r2866, r2857, r2863;
}
{
sub.f16x2 r2869, r1741, r1789;
}
{
mul.f16x2 r2872, r2869, r2570;
}
{
sub.f16x2 r2875, r1757, r1773;
}
{
mul.f16x2 r2878, r2875, r2573;
}
{
add.f16x2 r2881, r2872, r2878;
}
{
sub.f16x2 r2884, r2866, r2881;
}
st.local.u32 [rd3+140], r2884;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2887, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2888, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r2889, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r2890, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2891, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2892, {low, high};
}
{
neg.f16x2 r2893, r2892;
}
{
add.f16x2 r2895, r1805, r1853;
}
{
add.f16x2 r2898, r196, r2895;
}
{
add.f16x2 r2901, r1821, r1837;
}
{
add.f16x2 r2904, r2898, r2901;
}
st.local.u32 [rd3+24], r2904;
{
add.f16x2 r2907, r1811, r1859;
}
{
add.f16x2 r2910, r340, r2907;
}
{
add.f16x2 r2913, r1827, r1843;
}
{
add.f16x2 r2916, r2910, r2913;
}
st.local.u32 [rd3+28], r2916;
{
add.f16x2 r2919, r1805, r1853;
}
{
mul.f16x2 r2922, r2919, r2887;
}
{
add.f16x2 r2925, r196, r2922;
}
{
add.f16x2 r2928, r1821, r1837;
}
{
mul.f16x2 r2931, r2928, r2889;
}
{
add.f16x2 r2934, r2925, r2931;
}
{
sub.f16x2 r2937, r1811, r1859;
}
{
mul.f16x2 r2940, r2937, r2888;
}
{
sub.f16x2 r2943, r1827, r1843;
}
{
mul.f16x2 r2946, r2943, r2890;
}
{
add.f16x2 r2949, r2940, r2946;
}
{
sub.f16x2 r2952, r2934, r2949;
}
st.local.u32 [rd3+64], r2952;
{
add.f16x2 r2955, r1805, r1853;
}
{
mul.f16x2 r2958, r2955, r2887;
}
{
add.f16x2 r2961, r196, r2958;
}
{
add.f16x2 r2964, r1821, r1837;
}
{
mul.f16x2 r2967, r2964, r2889;
}
{
add.f16x2 r2970, r2961, r2967;
}
{
sub.f16x2 r2973, r1811, r1859;
}
{
mul.f16x2 r2976, r2973, r2888;
}
{
sub.f16x2 r2979, r1827, r1843;
}
{
mul.f16x2 r2982, r2979, r2890;
}
{
add.f16x2 r2985, r2976, r2982;
}
{
add.f16x2 r2988, r2970, r2985;
}
st.local.u32 [rd3+184], r2988;
{
add.f16x2 r2991, r1805, r1853;
}
{
mul.f16x2 r2994, r2991, r2889;
}
{
add.f16x2 r2997, r196, r2994;
}
{
add.f16x2 r3000, r1821, r1837;
}
{
mul.f16x2 r3003, r3000, r2891;
}
{
add.f16x2 r3006, r2997, r3003;
}
{
sub.f16x2 r3009, r1811, r1859;
}
{
mul.f16x2 r3012, r3009, r2890;
}
{
sub.f16x2 r3015, r1827, r1843;
}
{
mul.f16x2 r3018, r3015, r2893;
}
{
add.f16x2 r3021, r3012, r3018;
}
{
sub.f16x2 r3024, r3006, r3021;
}
st.local.u32 [rd3+104], r3024;
{
add.f16x2 r3027, r1805, r1853;
}
{
mul.f16x2 r3030, r3027, r2889;
}
{
add.f16x2 r3033, r196, r3030;
}
{
add.f16x2 r3036, r1821, r1837;
}
{
mul.f16x2 r3039, r3036, r2891;
}
{
add.f16x2 r3042, r3033, r3039;
}
{
sub.f16x2 r3045, r1811, r1859;
}
{
mul.f16x2 r3048, r3045, r2890;
}
{
sub.f16x2 r3051, r1827, r1843;
}
{
mul.f16x2 r3054, r3051, r2893;
}
{
add.f16x2 r3057, r3048, r3054;
}
{
add.f16x2 r3060, r3042, r3057;
}
st.local.u32 [rd3+144], r3060;
{
add.f16x2 r3063, r1811, r1859;
}
{
mul.f16x2 r3066, r3063, r2887;
}
{
add.f16x2 r3069, r340, r3066;
}
{
add.f16x2 r3072, r1827, r1843;
}
{
mul.f16x2 r3075, r3072, r2889;
}
{
add.f16x2 r3078, r3069, r3075;
}
{
sub.f16x2 r3081, r1805, r1853;
}
{
mul.f16x2 r3084, r3081, r2888;
}
{
sub.f16x2 r3087, r1821, r1837;
}
{
mul.f16x2 r3090, r3087, r2890;
}
{
add.f16x2 r3093, r3084, r3090;
}
{
add.f16x2 r3096, r3078, r3093;
}
st.local.u32 [rd3+68], r3096;
{
add.f16x2 r3099, r1811, r1859;
}
{
mul.f16x2 r3102, r3099, r2887;
}
{
add.f16x2 r3105, r340, r3102;
}
{
add.f16x2 r3108, r1827, r1843;
}
{
mul.f16x2 r3111, r3108, r2889;
}
{
add.f16x2 r3114, r3105, r3111;
}
{
sub.f16x2 r3117, r1805, r1853;
}
{
mul.f16x2 r3120, r3117, r2888;
}
{
sub.f16x2 r3123, r1821, r1837;
}
{
mul.f16x2 r3126, r3123, r2890;
}
{
add.f16x2 r3129, r3120, r3126;
}
{
sub.f16x2 r3132, r3114, r3129;
}
st.local.u32 [rd3+188], r3132;
{
add.f16x2 r3135, r1811, r1859;
}
{
mul.f16x2 r3138, r3135, r2889;
}
{
add.f16x2 r3141, r340, r3138;
}
{
add.f16x2 r3144, r1827, r1843;
}
{
mul.f16x2 r3147, r3144, r2891;
}
{
add.f16x2 r3150, r3141, r3147;
}
{
sub.f16x2 r3153, r1805, r1853;
}
{
mul.f16x2 r3156, r3153, r2890;
}
{
sub.f16x2 r3159, r1821, r1837;
}
{
mul.f16x2 r3162, r3159, r2893;
}
{
add.f16x2 r3165, r3156, r3162;
}
{
add.f16x2 r3168, r3150, r3165;
}
st.local.u32 [rd3+108], r3168;
{
add.f16x2 r3171, r1811, r1859;
}
{
mul.f16x2 r3174, r3171, r2889;
}
{
add.f16x2 r3177, r340, r3174;
}
{
add.f16x2 r3180, r1827, r1843;
}
{
mul.f16x2 r3183, r3180, r2891;
}
{
add.f16x2 r3186, r3177, r3183;
}
{
sub.f16x2 r3189, r1805, r1853;
}
{
mul.f16x2 r3192, r3189, r2890;
}
{
sub.f16x2 r3195, r1821, r1837;
}
{
mul.f16x2 r3198, r3195, r2893;
}
{
add.f16x2 r3201, r3192, r3198;
}
{
sub.f16x2 r3204, r3186, r3201;
}
st.local.u32 [rd3+148], r3204;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r3207, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r3208, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r3209, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r3210, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r3211, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r3212, {low, high};
}
{
neg.f16x2 r3213, r3212;
}
{
add.f16x2 r3215, r1869, r1917;
}
{
add.f16x2 r3218, r124, r3215;
}
{
add.f16x2 r3221, r1885, r1901;
}
{
add.f16x2 r3224, r3218, r3221;
}
st.local.u32 [rd3+32], r3224;
{
add.f16x2 r3227, r1875, r1923;
}
{
add.f16x2 r3230, r268, r3227;
}
{
add.f16x2 r3233, r1891, r1907;
}
{
add.f16x2 r3236, r3230, r3233;
}
st.local.u32 [rd3+36], r3236;
{
add.f16x2 r3239, r1869, r1917;
}
{
mul.f16x2 r3242, r3239, r3207;
}
{
add.f16x2 r3245, r124, r3242;
}
{
add.f16x2 r3248, r1885, r1901;
}
{
mul.f16x2 r3251, r3248, r3209;
}
{
add.f16x2 r3254, r3245, r3251;
}
{
sub.f16x2 r3257, r1875, r1923;
}
{
mul.f16x2 r3260, r3257, r3208;
}
{
sub.f16x2 r3263, r1891, r1907;
}
{
mul.f16x2 r3266, r3263, r3210;
}
{
add.f16x2 r3269, r3260, r3266;
}
{
sub.f16x2 r3272, r3254, r3269;
}
st.local.u32 [rd3+72], r3272;
{
add.f16x2 r3275, r1869, r1917;
}
{
mul.f16x2 r3278, r3275, r3207;
}
{
add.f16x2 r3281, r124, r3278;
}
{
add.f16x2 r3284, r1885, r1901;
}
{
mul.f16x2 r3287, r3284, r3209;
}
{
add.f16x2 r3290, r3281, r3287;
}
{
sub.f16x2 r3293, r1875, r1923;
}
{
mul.f16x2 r3296, r3293, r3208;
}
{
sub.f16x2 r3299, r1891, r1907;
}
{
mul.f16x2 r3302, r3299, r3210;
}
{
add.f16x2 r3305, r3296, r3302;
}
{
add.f16x2 r3308, r3290, r3305;
}
st.local.u32 [rd3+192], r3308;
{
add.f16x2 r3311, r1869, r1917;
}
{
mul.f16x2 r3314, r3311, r3209;
}
{
add.f16x2 r3317, r124, r3314;
}
{
add.f16x2 r3320, r1885, r1901;
}
{
mul.f16x2 r3323, r3320, r3211;
}
{
add.f16x2 r3326, r3317, r3323;
}
{
sub.f16x2 r3329, r1875, r1923;
}
{
mul.f16x2 r3332, r3329, r3210;
}
{
sub.f16x2 r3335, r1891, r1907;
}
{
mul.f16x2 r3338, r3335, r3213;
}
{
add.f16x2 r3341, r3332, r3338;
}
{
sub.f16x2 r3344, r3326, r3341;
}
st.local.u32 [rd3+112], r3344;
{
add.f16x2 r3347, r1869, r1917;
}
{
mul.f16x2 r3350, r3347, r3209;
}
{
add.f16x2 r3353, r124, r3350;
}
{
add.f16x2 r3356, r1885, r1901;
}
{
mul.f16x2 r3359, r3356, r3211;
}
{
add.f16x2 r3362, r3353, r3359;
}
{
sub.f16x2 r3365, r1875, r1923;
}
{
mul.f16x2 r3368, r3365, r3210;
}
{
sub.f16x2 r3371, r1891, r1907;
}
{
mul.f16x2 r3374, r3371, r3213;
}
{
add.f16x2 r3377, r3368, r3374;
}
{
add.f16x2 r3380, r3362, r3377;
}
st.local.u32 [rd3+152], r3380;
{
add.f16x2 r3383, r1875, r1923;
}
{
mul.f16x2 r3386, r3383, r3207;
}
{
add.f16x2 r3389, r268, r3386;
}
{
add.f16x2 r3392, r1891, r1907;
}
{
mul.f16x2 r3395, r3392, r3209;
}
{
add.f16x2 r3398, r3389, r3395;
}
{
sub.f16x2 r3401, r1869, r1917;
}
{
mul.f16x2 r3404, r3401, r3208;
}
{
sub.f16x2 r3407, r1885, r1901;
}
{
mul.f16x2 r3410, r3407, r3210;
}
{
add.f16x2 r3413, r3404, r3410;
}
{
add.f16x2 r3416, r3398, r3413;
}
st.local.u32 [rd3+76], r3416;
{
add.f16x2 r3419, r1875, r1923;
}
{
mul.f16x2 r3422, r3419, r3207;
}
{
add.f16x2 r3425, r268, r3422;
}
{
add.f16x2 r3428, r1891, r1907;
}
{
mul.f16x2 r3431, r3428, r3209;
}
{
add.f16x2 r3434, r3425, r3431;
}
{
sub.f16x2 r3437, r1869, r1917;
}
{
mul.f16x2 r3440, r3437, r3208;
}
{
sub.f16x2 r3443, r1885, r1901;
}
{
mul.f16x2 r3446, r3443, r3210;
}
{
add.f16x2 r3449, r3440, r3446;
}
{
sub.f16x2 r3452, r3434, r3449;
}
st.local.u32 [rd3+196], r3452;
{
add.f16x2 r3455, r1875, r1923;
}
{
mul.f16x2 r3458, r3455, r3209;
}
{
add.f16x2 r3461, r268, r3458;
}
{
add.f16x2 r3464, r1891, r1907;
}
{
mul.f16x2 r3467, r3464, r3211;
}
{
add.f16x2 r3470, r3461, r3467;
}
{
sub.f16x2 r3473, r1869, r1917;
}
{
mul.f16x2 r3476, r3473, r3210;
}
{
sub.f16x2 r3479, r1885, r1901;
}
{
mul.f16x2 r3482, r3479, r3213;
}
{
add.f16x2 r3485, r3476, r3482;
}
{
add.f16x2 r3488, r3470, r3485;
}
st.local.u32 [rd3+116], r3488;
{
add.f16x2 r3491, r1875, r1923;
}
{
mul.f16x2 r3494, r3491, r3209;
}
{
add.f16x2 r3497, r268, r3494;
}
{
add.f16x2 r3500, r1891, r1907;
}
{
mul.f16x2 r3503, r3500, r3211;
}
{
add.f16x2 r3506, r3497, r3503;
}
{
sub.f16x2 r3509, r1869, r1917;
}
{
mul.f16x2 r3512, r3509, r3210;
}
{
sub.f16x2 r3515, r1885, r1901;
}
{
mul.f16x2 r3518, r3515, r3213;
}
{
add.f16x2 r3521, r3512, r3518;
}
{
sub.f16x2 r3524, r3506, r3521;
}
st.local.u32 [rd3+156], r3524;
mov.u32 r3532, %tid.x;
mul.wide.u32 rd9, r3532, -776530087;
shr.u64 rd10, rd9, 41;
cvt.u32.u64 r4, rd10;
mul.lo.s32 r3533, r4, 625;
sub.s32 r3, r3532, r3533;
cvt.rn.f32.u32 f221, r3;
mul.f32 f222, f221, 0f39D2D427;
cos.approx.f32 f217, f222;
sin.approx.f32 f223, f222;
neg.f32 f218, f223;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f217;
cvt.rn.f16.f32 high, f218;
mov.b32 r11250, {low, high};
}
mov.u32 r11249, 1;
{
.reg .f16 low, high;
mov.b32 {low, high}, r11250;
mov.b32 r3554, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11250;
mov.b32 r3556, {high, high};
}
bra.uni LBB1_1;
LBB1_2:
ld.local.u32 r11248, [rd5+60];
LBB1_1:
{
.reg .f16 low, high;
mov.b32 {low, high}, r11250;
mov.b32 r3534, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11250;
mov.b32 r3536, {high, high};
}
mul.wide.u32 rd11, r11249, 8;
add.s64 rd12, rd3, rd11;
add.s64 rd5, rd12, 4;
ld.local.u32 r3539, [rd12+4];
{
mul.f16x2 r3538, r3539, r3536;
}
{
neg.f16x2 r3541, r3538;
}
{
fma.rn.f16x2 r3543, r11248, r3534, r3541;
}
st.local.u32 [rd12], r3543;
{
mul.f16x2 r3547, r11248, r3536;
}
{
fma.rn.f16x2 r3550, r3539, r3534, r3547;
}
st.local.u32 [rd12+4], r3550;
mov.f32 f238, 0fBF800000;
mov.f32 f239, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3558, {low, high};
}
{
mul.f16x2 r3559, r3556, r3558;
}
{
mul.f16x2 r3562, r11250, r3554;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11250;
mov.b32 r3565, {high, low};
}
{
fma.rn.f16x2 r3567, r3559, r3565, r3562;
}
ld.local.u32 r3585, [rd12+8];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3567;
mov.b32 r3571, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3567;
mov.b32 r3573, {high, high};
}
ld.local.u32 r3588, [rd12+12];
{
mul.f16x2 r3575, r3588, r3573;
}
{
neg.f16x2 r3578, r3575;
}
{
fma.rn.f16x2 r3580, r3585, r3571, r3578;
}
st.local.u32 [rd12+8], r3580;
{
mul.f16x2 r3584, r3585, r3573;
}
{
fma.rn.f16x2 r3587, r3588, r3571, r3584;
}
st.local.u32 [rd12+12], r3587;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3595, {low, high};
}
{
mul.f16x2 r3596, r3556, r3595;
}
{
mul.f16x2 r3599, r3567, r3554;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3567;
mov.b32 r3602, {high, low};
}
{
fma.rn.f16x2 r3604, r3596, r3602, r3599;
}
ld.local.u32 r3622, [rd12+16];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3604;
mov.b32 r3608, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3604;
mov.b32 r3610, {high, high};
}
ld.local.u32 r3625, [rd12+20];
{
mul.f16x2 r3612, r3625, r3610;
}
{
neg.f16x2 r3615, r3612;
}
{
fma.rn.f16x2 r3617, r3622, r3608, r3615;
}
st.local.u32 [rd12+16], r3617;
{
mul.f16x2 r3621, r3622, r3610;
}
{
fma.rn.f16x2 r3624, r3625, r3608, r3621;
}
st.local.u32 [rd12+20], r3624;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3632, {low, high};
}
{
mul.f16x2 r3633, r3556, r3632;
}
{
mul.f16x2 r3636, r3604, r3554;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3604;
mov.b32 r3639, {high, low};
}
{
fma.rn.f16x2 r3641, r3633, r3639, r3636;
}
ld.local.u32 r3659, [rd12+24];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3641;
mov.b32 r3645, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3641;
mov.b32 r3647, {high, high};
}
ld.local.u32 r3662, [rd12+28];
{
mul.f16x2 r3649, r3662, r3647;
}
{
neg.f16x2 r3652, r3649;
}
{
fma.rn.f16x2 r3654, r3659, r3645, r3652;
}
st.local.u32 [rd12+24], r3654;
{
mul.f16x2 r3658, r3659, r3647;
}
{
fma.rn.f16x2 r3661, r3662, r3645, r3658;
}
st.local.u32 [rd12+28], r3661;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3669, {low, high};
}
{
mul.f16x2 r3670, r3556, r3669;
}
{
mul.f16x2 r3673, r3641, r3554;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3641;
mov.b32 r3676, {high, low};
}
{
fma.rn.f16x2 r3678, r3670, r3676, r3673;
}
ld.local.u32 r3692, [rd12+32];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3678;
mov.b32 r3682, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3678;
mov.b32 r3684, {high, high};
}
ld.local.u32 r3699, [rd12+36];
{
mul.f16x2 r3686, r3699, r3684;
}
{
neg.f16x2 r3689, r3686;
}
{
fma.rn.f16x2 r3691, r3692, r3682, r3689;
}
st.local.u32 [rd12+32], r3691;
{
mul.f16x2 r3695, r3692, r3684;
}
{
fma.rn.f16x2 r3698, r3699, r3682, r3695;
}
st.local.u32 [rd12+36], r3698;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3706, {low, high};
}
{
mul.f16x2 r3707, r3556, r3706;
}
{
mul.f16x2 r3710, r3678, r3554;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3678;
mov.b32 r3713, {high, low};
}
{
fma.rn.f16x2 r3715, r3707, r3713, r3710;
}
ld.local.u32 r3729, [rd12+40];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3715;
mov.b32 r3719, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3715;
mov.b32 r3721, {high, high};
}
ld.local.u32 r3724, [rd12+44];
{
mul.f16x2 r3723, r3724, r3721;
}
{
neg.f16x2 r3726, r3723;
}
{
fma.rn.f16x2 r3728, r3729, r3719, r3726;
}
st.local.u32 [rd12+40], r3728;
{
mul.f16x2 r3732, r3729, r3721;
}
{
fma.rn.f16x2 r3735, r3724, r3719, r3732;
}
st.local.u32 [rd12+44], r3735;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3743, {low, high};
}
{
mul.f16x2 r3744, r3556, r3743;
}
{
mul.f16x2 r3747, r3715, r3554;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3715;
mov.b32 r3750, {high, low};
}
{
fma.rn.f16x2 r3752, r3744, r3750, r3747;
}
ld.local.u32 r3766, [rd12+48];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3752;
mov.b32 r3756, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3752;
mov.b32 r3758, {high, high};
}
ld.local.u32 r3761, [rd12+52];
{
mul.f16x2 r3760, r3761, r3758;
}
{
neg.f16x2 r3763, r3760;
}
{
fma.rn.f16x2 r3765, r3766, r3756, r3763;
}
st.local.u32 [rd12+48], r3765;
{
mul.f16x2 r3769, r3766, r3758;
}
{
fma.rn.f16x2 r3772, r3761, r3756, r3769;
}
st.local.u32 [rd12+52], r3772;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3780, {low, high};
}
{
mul.f16x2 r3781, r3556, r3780;
}
{
mul.f16x2 r3784, r3752, r3554;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3752;
mov.b32 r3787, {high, low};
}
{
fma.rn.f16x2 r3789, r3781, r3787, r3784;
}
ld.local.u32 r3803, [rd12+56];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3789;
mov.b32 r3793, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3789;
mov.b32 r3795, {high, high};
}
ld.local.u32 r3798, [rd12+60];
{
mul.f16x2 r3797, r3798, r3795;
}
{
neg.f16x2 r3800, r3797;
}
{
fma.rn.f16x2 r3802, r3803, r3793, r3800;
}
st.local.u32 [rd12+56], r3802;
{
mul.f16x2 r3806, r3803, r3795;
}
{
fma.rn.f16x2 r3809, r3798, r3793, r3806;
}
st.local.u32 [rd12+60], r3809;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3817, {low, high};
}
{
mul.f16x2 r3818, r3556, r3817;
}
{
mul.f16x2 r3821, r3789, r3554;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3789;
mov.b32 r3824, {high, low};
}
{
fma.rn.f16x2 r11250, r3818, r3824, r3821;
}
add.s32 r11249, r11249, 8;
setp.eq.s32 p1, r11249, 25;
@p1 bra LBB1_3;
bra.uni LBB1_2;
LBB1_3:
shl.b32 r7338, r1, 2;
mov.u32 r7339, %50;
add.s32 r7340, r7339, r7338;
mad.lo.s32 r12, r4, 62500, r7340;
barrier.sync 0;
mad.lo.s32 r7341, r3, 100, r12;
ld.local.u32 r7342, [rd3];
st.shared.u32 [r7341], r7342;
ld.local.u32 r7343, [rd4+4];
st.shared.u32 [r7341+4], r7343;
ld.local.u32 r7344, [rd4+12];
st.shared.u32 [r7341+8], r7344;
ld.local.u32 r7345, [rd4+20];
st.shared.u32 [r7341+12], r7345;
ld.local.u32 r7346, [rd4+28];
st.shared.u32 [r7341+16], r7346;
ld.local.u32 r7347, [rd4+36];
st.shared.u32 [r7341+20], r7347;
ld.local.u32 r7348, [rd4+44];
st.shared.u32 [r7341+24], r7348;
ld.local.u32 r7349, [rd4+52];
st.shared.u32 [r7341+28], r7349;
ld.local.u32 r7350, [rd4+60];
st.shared.u32 [r7341+32], r7350;
ld.local.u32 r7351, [rd4+68];
st.shared.u32 [r7341+36], r7351;
ld.local.u32 r7352, [rd4+76];
st.shared.u32 [r7341+40], r7352;
ld.local.u32 r7353, [rd4+84];
st.shared.u32 [r7341+44], r7353;
ld.local.u32 r7354, [rd4+92];
st.shared.u32 [r7341+48], r7354;
ld.local.u32 r7355, [rd4+100];
st.shared.u32 [r7341+52], r7355;
ld.local.u32 r7356, [rd4+108];
st.shared.u32 [r7341+56], r7356;
ld.local.u32 r7357, [rd4+116];
st.shared.u32 [r7341+60], r7357;
ld.local.u32 r7358, [rd4+124];
st.shared.u32 [r7341+64], r7358;
ld.local.u32 r7359, [rd4+132];
st.shared.u32 [r7341+68], r7359;
ld.local.u32 r7360, [rd4+140];
st.shared.u32 [r7341+72], r7360;
ld.local.u32 r7361, [rd4+148];
st.shared.u32 [r7341+76], r7361;
ld.local.u32 r7362, [rd4+156];
st.shared.u32 [r7341+80], r7362;
ld.local.u32 r7363, [rd4+164];
st.shared.u32 [r7341+84], r7363;
ld.local.u32 r7364, [rd4+172];
st.shared.u32 [r7341+88], r7364;
ld.local.u32 r7365, [rd4+180];
st.shared.u32 [r7341+92], r7365;
ld.local.u32 r7366, [rd4+188];
st.shared.u32 [r7341+96], r7366;
barrier.sync 0;
mad.lo.s32 r13, r3, -96, r7341;
ld.shared.u32 r3842, [r13];
ld.shared.u32 r4162, [r13+2500];
ld.shared.u32 r4482, [r13+5000];
ld.shared.u32 r4802, [r13+7500];
ld.shared.u32 r5122, [r13+10000];
ld.shared.u32 r3839, [r13+12500];
ld.shared.u32 r4159, [r13+15000];
ld.shared.u32 r4479, [r13+17500];
ld.shared.u32 r4799, [r13+20000];
ld.shared.u32 r5119, [r13+22500];
ld.shared.u32 r3845, [r13+25000];
ld.shared.u32 r4165, [r13+27500];
ld.shared.u32 r4485, [r13+30000];
ld.shared.u32 r4805, [r13+32500];
ld.shared.u32 r5125, [r13+35000];
ld.shared.u32 r3846, [r13+37500];
ld.shared.u32 r4166, [r13+40000];
ld.shared.u32 r4486, [r13+42500];
ld.shared.u32 r4806, [r13+45000];
ld.shared.u32 r5126, [r13+47500];
ld.shared.u32 r3840, [r13+50000];
ld.shared.u32 r4160, [r13+52500];
ld.shared.u32 r4480, [r13+55000];
ld.shared.u32 r4800, [r13+57500];
ld.shared.u32 r5120, [r13+60000];
barrier.sync 0;
ld.local.u32 r7367, [rd4];
st.shared.u32 [r7341], r7367;
ld.local.u32 r7368, [rd4+8];
st.shared.u32 [r7341+4], r7368;
ld.local.u32 r7369, [rd4+16];
st.shared.u32 [r7341+8], r7369;
ld.local.u32 r7370, [rd4+24];
st.shared.u32 [r7341+12], r7370;
ld.local.u32 r7371, [rd4+32];
st.shared.u32 [r7341+16], r7371;
ld.local.u32 r7372, [rd4+40];
st.shared.u32 [r7341+20], r7372;
ld.local.u32 r7373, [rd4+48];
st.shared.u32 [r7341+24], r7373;
ld.local.u32 r7374, [rd4+56];
st.shared.u32 [r7341+28], r7374;
ld.local.u32 r7375, [rd4+64];
st.shared.u32 [r7341+32], r7375;
ld.local.u32 r7376, [rd4+72];
st.shared.u32 [r7341+36], r7376;
ld.local.u32 r7377, [rd4+80];
st.shared.u32 [r7341+40], r7377;
ld.local.u32 r7378, [rd4+88];
st.shared.u32 [r7341+44], r7378;
ld.local.u32 r7379, [rd4+96];
st.shared.u32 [r7341+48], r7379;
ld.local.u32 r7380, [rd4+104];
st.shared.u32 [r7341+52], r7380;
ld.local.u32 r7381, [rd4+112];
st.shared.u32 [r7341+56], r7381;
ld.local.u32 r7382, [rd4+120];
st.shared.u32 [r7341+60], r7382;
ld.local.u32 r7383, [rd4+128];
st.shared.u32 [r7341+64], r7383;
ld.local.u32 r7384, [rd4+136];
st.shared.u32 [r7341+68], r7384;
ld.local.u32 r7385, [rd4+144];
st.shared.u32 [r7341+72], r7385;
ld.local.u32 r7386, [rd4+152];
st.shared.u32 [r7341+76], r7386;
ld.local.u32 r7387, [rd4+160];
st.shared.u32 [r7341+80], r7387;
ld.local.u32 r7388, [rd4+168];
st.shared.u32 [r7341+84], r7388;
ld.local.u32 r7389, [rd4+176];
st.shared.u32 [r7341+88], r7389;
ld.local.u32 r7390, [rd4+184];
st.shared.u32 [r7341+92], r7390;
ld.local.u32 r7391, [rd4+192];
st.shared.u32 [r7341+96], r7391;
barrier.sync 0;
ld.shared.u32 r3854, [r13];
ld.shared.u32 r4174, [r13+2500];
ld.shared.u32 r4494, [r13+5000];
ld.shared.u32 r4814, [r13+7500];
ld.shared.u32 r5134, [r13+10000];
ld.shared.u32 r3851, [r13+12500];
ld.shared.u32 r4171, [r13+15000];
ld.shared.u32 r4491, [r13+17500];
ld.shared.u32 r4811, [r13+20000];
ld.shared.u32 r5131, [r13+22500];
ld.shared.u32 r3857, [r13+25000];
ld.shared.u32 r4177, [r13+27500];
ld.shared.u32 r4497, [r13+30000];
ld.shared.u32 r4817, [r13+32500];
ld.shared.u32 r5137, [r13+35000];
ld.shared.u32 r3858, [r13+37500];
ld.shared.u32 r4178, [r13+40000];
ld.shared.u32 r4498, [r13+42500];
ld.shared.u32 r4818, [r13+45000];
ld.shared.u32 r5138, [r13+47500];
ld.shared.u32 r3852, [r13+50000];
ld.shared.u32 r4172, [r13+52500];
ld.shared.u32 r4492, [r13+55000];
ld.shared.u32 r4812, [r13+57500];
ld.shared.u32 r5132, [r13+60000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r3830, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r3831, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r3832, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r3833, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r3834, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r3835, {low, high};
}
{
neg.f16x2 r3836, r3835;
}
{
add.f16x2 r3838, r3839, r3840;
}
{
add.f16x2 r3841, r3842, r3838;
}
{
add.f16x2 r3844, r3845, r3846;
}
{
add.f16x2 r3847, r3841, r3844;
}
{
add.f16x2 r3850, r3851, r3852;
}
{
add.f16x2 r3853, r3854, r3850;
}
{
add.f16x2 r3856, r3857, r3858;
}
{
add.f16x2 r3859, r3853, r3856;
}
{
add.f16x2 r3862, r3839, r3840;
}
{
mul.f16x2 r3865, r3862, r3830;
}
{
add.f16x2 r3868, r3842, r3865;
}
{
add.f16x2 r3871, r3845, r3846;
}
{
mul.f16x2 r3874, r3871, r3832;
}
{
add.f16x2 r3877, r3868, r3874;
}
{
sub.f16x2 r3880, r3851, r3852;
}
{
mul.f16x2 r3883, r3880, r3831;
}
{
sub.f16x2 r3886, r3857, r3858;
}
{
mul.f16x2 r3889, r3886, r3833;
}
{
add.f16x2 r3892, r3883, r3889;
}
{
sub.f16x2 r3895, r3877, r3892;
}
{
add.f16x2 r3898, r3839, r3840;
}
{
mul.f16x2 r3901, r3898, r3830;
}
{
add.f16x2 r3904, r3842, r3901;
}
{
add.f16x2 r3907, r3845, r3846;
}
{
mul.f16x2 r3910, r3907, r3832;
}
{
add.f16x2 r3913, r3904, r3910;
}
{
sub.f16x2 r3916, r3851, r3852;
}
{
mul.f16x2 r3919, r3916, r3831;
}
{
sub.f16x2 r3922, r3857, r3858;
}
{
mul.f16x2 r3925, r3922, r3833;
}
{
add.f16x2 r3928, r3919, r3925;
}
{
add.f16x2 r3931, r3913, r3928;
}
{
add.f16x2 r3934, r3839, r3840;
}
{
mul.f16x2 r3937, r3934, r3832;
}
{
add.f16x2 r3940, r3842, r3937;
}
{
add.f16x2 r3943, r3845, r3846;
}
{
mul.f16x2 r3946, r3943, r3834;
}
{
add.f16x2 r3949, r3940, r3946;
}
{
sub.f16x2 r3952, r3851, r3852;
}
{
mul.f16x2 r3955, r3952, r3833;
}
{
sub.f16x2 r3958, r3857, r3858;
}
{
mul.f16x2 r3961, r3958, r3836;
}
{
add.f16x2 r3964, r3955, r3961;
}
{
sub.f16x2 r3967, r3949, r3964;
}
{
add.f16x2 r3970, r3839, r3840;
}
{
mul.f16x2 r3973, r3970, r3832;
}
{
add.f16x2 r3976, r3842, r3973;
}
{
add.f16x2 r3979, r3845, r3846;
}
{
mul.f16x2 r3982, r3979, r3834;
}
{
add.f16x2 r3985, r3976, r3982;
}
{
sub.f16x2 r3988, r3851, r3852;
}
{
mul.f16x2 r3991, r3988, r3833;
}
{
sub.f16x2 r3994, r3857, r3858;
}
{
mul.f16x2 r3997, r3994, r3836;
}
{
add.f16x2 r4000, r3991, r3997;
}
{
add.f16x2 r4003, r3985, r4000;
}
{
add.f16x2 r4006, r3851, r3852;
}
{
mul.f16x2 r4009, r4006, r3830;
}
{
add.f16x2 r4012, r3854, r4009;
}
{
add.f16x2 r4015, r3857, r3858;
}
{
mul.f16x2 r4018, r4015, r3832;
}
{
add.f16x2 r4021, r4012, r4018;
}
{
sub.f16x2 r4024, r3839, r3840;
}
{
mul.f16x2 r4027, r4024, r3831;
}
{
sub.f16x2 r4030, r3845, r3846;
}
{
mul.f16x2 r4033, r4030, r3833;
}
{
add.f16x2 r4036, r4027, r4033;
}
{
add.f16x2 r4039, r4021, r4036;
}
{
add.f16x2 r4042, r3851, r3852;
}
{
mul.f16x2 r4045, r4042, r3830;
}
{
add.f16x2 r4048, r3854, r4045;
}
{
add.f16x2 r4051, r3857, r3858;
}
{
mul.f16x2 r4054, r4051, r3832;
}
{
add.f16x2 r4057, r4048, r4054;
}
{
sub.f16x2 r4060, r3839, r3840;
}
{
mul.f16x2 r4063, r4060, r3831;
}
{
sub.f16x2 r4066, r3845, r3846;
}
{
mul.f16x2 r4069, r4066, r3833;
}
{
add.f16x2 r4072, r4063, r4069;
}
{
sub.f16x2 r4075, r4057, r4072;
}
{
add.f16x2 r4078, r3851, r3852;
}
{
mul.f16x2 r4081, r4078, r3832;
}
{
add.f16x2 r4084, r3854, r4081;
}
{
add.f16x2 r4087, r3857, r3858;
}
{
mul.f16x2 r4090, r4087, r3834;
}
{
add.f16x2 r4093, r4084, r4090;
}
{
sub.f16x2 r4096, r3839, r3840;
}
{
mul.f16x2 r4099, r4096, r3833;
}
{
sub.f16x2 r4102, r3845, r3846;
}
{
mul.f16x2 r4105, r4102, r3836;
}
{
add.f16x2 r4108, r4099, r4105;
}
{
add.f16x2 r4111, r4093, r4108;
}
{
add.f16x2 r4114, r3851, r3852;
}
{
mul.f16x2 r4117, r4114, r3832;
}
{
add.f16x2 r4120, r3854, r4117;
}
{
add.f16x2 r4123, r3857, r3858;
}
{
mul.f16x2 r4126, r4123, r3834;
}
{
add.f16x2 r4129, r4120, r4126;
}
{
sub.f16x2 r4132, r3839, r3840;
}
{
mul.f16x2 r4135, r4132, r3833;
}
{
sub.f16x2 r4138, r3845, r3846;
}
{
mul.f16x2 r4141, r4138, r3836;
}
{
add.f16x2 r4144, r4135, r4141;
}
{
sub.f16x2 r4147, r4129, r4144;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4150, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4151, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r4152, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r4153, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4154, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4155, {low, high};
}
{
neg.f16x2 r4156, r4155;
}
{
add.f16x2 r4158, r4159, r4160;
}
{
add.f16x2 r4161, r4162, r4158;
}
{
add.f16x2 r4164, r4165, r4166;
}
{
add.f16x2 r4167, r4161, r4164;
}
{
add.f16x2 r4170, r4171, r4172;
}
{
add.f16x2 r4173, r4174, r4170;
}
{
add.f16x2 r4176, r4177, r4178;
}
{
add.f16x2 r4179, r4173, r4176;
}
{
add.f16x2 r4182, r4159, r4160;
}
{
mul.f16x2 r4185, r4182, r4150;
}
{
add.f16x2 r4188, r4162, r4185;
}
{
add.f16x2 r4191, r4165, r4166;
}
{
mul.f16x2 r4194, r4191, r4152;
}
{
add.f16x2 r4197, r4188, r4194;
}
{
sub.f16x2 r4200, r4171, r4172;
}
{
mul.f16x2 r4203, r4200, r4151;
}
{
sub.f16x2 r4206, r4177, r4178;
}
{
mul.f16x2 r4209, r4206, r4153;
}
{
add.f16x2 r4212, r4203, r4209;
}
{
sub.f16x2 r4215, r4197, r4212;
}
{
add.f16x2 r4218, r4159, r4160;
}
{
mul.f16x2 r4221, r4218, r4150;
}
{
add.f16x2 r4224, r4162, r4221;
}
{
add.f16x2 r4227, r4165, r4166;
}
{
mul.f16x2 r4230, r4227, r4152;
}
{
add.f16x2 r4233, r4224, r4230;
}
{
sub.f16x2 r4236, r4171, r4172;
}
{
mul.f16x2 r4239, r4236, r4151;
}
{
sub.f16x2 r4242, r4177, r4178;
}
{
mul.f16x2 r4245, r4242, r4153;
}
{
add.f16x2 r4248, r4239, r4245;
}
{
add.f16x2 r4251, r4233, r4248;
}
{
add.f16x2 r4254, r4159, r4160;
}
{
mul.f16x2 r4257, r4254, r4152;
}
{
add.f16x2 r4260, r4162, r4257;
}
{
add.f16x2 r4263, r4165, r4166;
}
{
mul.f16x2 r4266, r4263, r4154;
}
{
add.f16x2 r4269, r4260, r4266;
}
{
sub.f16x2 r4272, r4171, r4172;
}
{
mul.f16x2 r4275, r4272, r4153;
}
{
sub.f16x2 r4278, r4177, r4178;
}
{
mul.f16x2 r4281, r4278, r4156;
}
{
add.f16x2 r4284, r4275, r4281;
}
{
sub.f16x2 r4287, r4269, r4284;
}
{
add.f16x2 r4290, r4159, r4160;
}
{
mul.f16x2 r4293, r4290, r4152;
}
{
add.f16x2 r4296, r4162, r4293;
}
{
add.f16x2 r4299, r4165, r4166;
}
{
mul.f16x2 r4302, r4299, r4154;
}
{
add.f16x2 r4305, r4296, r4302;
}
{
sub.f16x2 r4308, r4171, r4172;
}
{
mul.f16x2 r4311, r4308, r4153;
}
{
sub.f16x2 r4314, r4177, r4178;
}
{
mul.f16x2 r4317, r4314, r4156;
}
{
add.f16x2 r4320, r4311, r4317;
}
{
add.f16x2 r4323, r4305, r4320;
}
{
add.f16x2 r4326, r4171, r4172;
}
{
mul.f16x2 r4329, r4326, r4150;
}
{
add.f16x2 r4332, r4174, r4329;
}
{
add.f16x2 r4335, r4177, r4178;
}
{
mul.f16x2 r4338, r4335, r4152;
}
{
add.f16x2 r4341, r4332, r4338;
}
{
sub.f16x2 r4344, r4159, r4160;
}
{
mul.f16x2 r4347, r4344, r4151;
}
{
sub.f16x2 r4350, r4165, r4166;
}
{
mul.f16x2 r4353, r4350, r4153;
}
{
add.f16x2 r4356, r4347, r4353;
}
{
add.f16x2 r4359, r4341, r4356;
}
{
add.f16x2 r4362, r4171, r4172;
}
{
mul.f16x2 r4365, r4362, r4150;
}
{
add.f16x2 r4368, r4174, r4365;
}
{
add.f16x2 r4371, r4177, r4178;
}
{
mul.f16x2 r4374, r4371, r4152;
}
{
add.f16x2 r4377, r4368, r4374;
}
{
sub.f16x2 r4380, r4159, r4160;
}
{
mul.f16x2 r4383, r4380, r4151;
}
{
sub.f16x2 r4386, r4165, r4166;
}
{
mul.f16x2 r4389, r4386, r4153;
}
{
add.f16x2 r4392, r4383, r4389;
}
{
sub.f16x2 r4395, r4377, r4392;
}
{
add.f16x2 r4398, r4171, r4172;
}
{
mul.f16x2 r4401, r4398, r4152;
}
{
add.f16x2 r4404, r4174, r4401;
}
{
add.f16x2 r4407, r4177, r4178;
}
{
mul.f16x2 r4410, r4407, r4154;
}
{
add.f16x2 r4413, r4404, r4410;
}
{
sub.f16x2 r4416, r4159, r4160;
}
{
mul.f16x2 r4419, r4416, r4153;
}
{
sub.f16x2 r4422, r4165, r4166;
}
{
mul.f16x2 r4425, r4422, r4156;
}
{
add.f16x2 r4428, r4419, r4425;
}
{
add.f16x2 r4431, r4413, r4428;
}
{
add.f16x2 r4434, r4171, r4172;
}
{
mul.f16x2 r4437, r4434, r4152;
}
{
add.f16x2 r4440, r4174, r4437;
}
{
add.f16x2 r4443, r4177, r4178;
}
{
mul.f16x2 r4446, r4443, r4154;
}
{
add.f16x2 r4449, r4440, r4446;
}
{
sub.f16x2 r4452, r4159, r4160;
}
{
mul.f16x2 r4455, r4452, r4153;
}
{
sub.f16x2 r4458, r4165, r4166;
}
{
mul.f16x2 r4461, r4458, r4156;
}
{
add.f16x2 r4464, r4455, r4461;
}
{
sub.f16x2 r4467, r4449, r4464;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4470, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4471, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r4472, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r4473, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4474, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4475, {low, high};
}
{
neg.f16x2 r4476, r4475;
}
{
add.f16x2 r4478, r4479, r4480;
}
{
add.f16x2 r4481, r4482, r4478;
}
{
add.f16x2 r4484, r4485, r4486;
}
{
add.f16x2 r4487, r4481, r4484;
}
{
add.f16x2 r4490, r4491, r4492;
}
{
add.f16x2 r4493, r4494, r4490;
}
{
add.f16x2 r4496, r4497, r4498;
}
{
add.f16x2 r4499, r4493, r4496;
}
{
add.f16x2 r4502, r4479, r4480;
}
{
mul.f16x2 r4505, r4502, r4470;
}
{
add.f16x2 r4508, r4482, r4505;
}
{
add.f16x2 r4511, r4485, r4486;
}
{
mul.f16x2 r4514, r4511, r4472;
}
{
add.f16x2 r4517, r4508, r4514;
}
{
sub.f16x2 r4520, r4491, r4492;
}
{
mul.f16x2 r4523, r4520, r4471;
}
{
sub.f16x2 r4526, r4497, r4498;
}
{
mul.f16x2 r4529, r4526, r4473;
}
{
add.f16x2 r4532, r4523, r4529;
}
{
sub.f16x2 r4535, r4517, r4532;
}
{
add.f16x2 r4538, r4479, r4480;
}
{
mul.f16x2 r4541, r4538, r4470;
}
{
add.f16x2 r4544, r4482, r4541;
}
{
add.f16x2 r4547, r4485, r4486;
}
{
mul.f16x2 r4550, r4547, r4472;
}
{
add.f16x2 r4553, r4544, r4550;
}
{
sub.f16x2 r4556, r4491, r4492;
}
{
mul.f16x2 r4559, r4556, r4471;
}
{
sub.f16x2 r4562, r4497, r4498;
}
{
mul.f16x2 r4565, r4562, r4473;
}
{
add.f16x2 r4568, r4559, r4565;
}
{
add.f16x2 r4571, r4553, r4568;
}
{
add.f16x2 r4574, r4479, r4480;
}
{
mul.f16x2 r4577, r4574, r4472;
}
{
add.f16x2 r4580, r4482, r4577;
}
{
add.f16x2 r4583, r4485, r4486;
}
{
mul.f16x2 r4586, r4583, r4474;
}
{
add.f16x2 r4589, r4580, r4586;
}
{
sub.f16x2 r4592, r4491, r4492;
}
{
mul.f16x2 r4595, r4592, r4473;
}
{
sub.f16x2 r4598, r4497, r4498;
}
{
mul.f16x2 r4601, r4598, r4476;
}
{
add.f16x2 r4604, r4595, r4601;
}
{
sub.f16x2 r4607, r4589, r4604;
}
{
add.f16x2 r4610, r4479, r4480;
}
{
mul.f16x2 r4613, r4610, r4472;
}
{
add.f16x2 r4616, r4482, r4613;
}
{
add.f16x2 r4619, r4485, r4486;
}
{
mul.f16x2 r4622, r4619, r4474;
}
{
add.f16x2 r4625, r4616, r4622;
}
{
sub.f16x2 r4628, r4491, r4492;
}
{
mul.f16x2 r4631, r4628, r4473;
}
{
sub.f16x2 r4634, r4497, r4498;
}
{
mul.f16x2 r4637, r4634, r4476;
}
{
add.f16x2 r4640, r4631, r4637;
}
{
add.f16x2 r4643, r4625, r4640;
}
{
add.f16x2 r4646, r4491, r4492;
}
{
mul.f16x2 r4649, r4646, r4470;
}
{
add.f16x2 r4652, r4494, r4649;
}
{
add.f16x2 r4655, r4497, r4498;
}
{
mul.f16x2 r4658, r4655, r4472;
}
{
add.f16x2 r4661, r4652, r4658;
}
{
sub.f16x2 r4664, r4479, r4480;
}
{
mul.f16x2 r4667, r4664, r4471;
}
{
sub.f16x2 r4670, r4485, r4486;
}
{
mul.f16x2 r4673, r4670, r4473;
}
{
add.f16x2 r4676, r4667, r4673;
}
{
add.f16x2 r4679, r4661, r4676;
}
{
add.f16x2 r4682, r4491, r4492;
}
{
mul.f16x2 r4685, r4682, r4470;
}
{
add.f16x2 r4688, r4494, r4685;
}
{
add.f16x2 r4691, r4497, r4498;
}
{
mul.f16x2 r4694, r4691, r4472;
}
{
add.f16x2 r4697, r4688, r4694;
}
{
sub.f16x2 r4700, r4479, r4480;
}
{
mul.f16x2 r4703, r4700, r4471;
}
{
sub.f16x2 r4706, r4485, r4486;
}
{
mul.f16x2 r4709, r4706, r4473;
}
{
add.f16x2 r4712, r4703, r4709;
}
{
sub.f16x2 r4715, r4697, r4712;
}
{
add.f16x2 r4718, r4491, r4492;
}
{
mul.f16x2 r4721, r4718, r4472;
}
{
add.f16x2 r4724, r4494, r4721;
}
{
add.f16x2 r4727, r4497, r4498;
}
{
mul.f16x2 r4730, r4727, r4474;
}
{
add.f16x2 r4733, r4724, r4730;
}
{
sub.f16x2 r4736, r4479, r4480;
}
{
mul.f16x2 r4739, r4736, r4473;
}
{
sub.f16x2 r4742, r4485, r4486;
}
{
mul.f16x2 r4745, r4742, r4476;
}
{
add.f16x2 r4748, r4739, r4745;
}
{
add.f16x2 r4751, r4733, r4748;
}
{
add.f16x2 r4754, r4491, r4492;
}
{
mul.f16x2 r4757, r4754, r4472;
}
{
add.f16x2 r4760, r4494, r4757;
}
{
add.f16x2 r4763, r4497, r4498;
}
{
mul.f16x2 r4766, r4763, r4474;
}
{
add.f16x2 r4769, r4760, r4766;
}
{
sub.f16x2 r4772, r4479, r4480;
}
{
mul.f16x2 r4775, r4772, r4473;
}
{
sub.f16x2 r4778, r4485, r4486;
}
{
mul.f16x2 r4781, r4778, r4476;
}
{
add.f16x2 r4784, r4775, r4781;
}
{
sub.f16x2 r4787, r4769, r4784;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4790, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4791, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r4792, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r4793, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4794, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4795, {low, high};
}
{
neg.f16x2 r4796, r4795;
}
{
add.f16x2 r4798, r4799, r4800;
}
{
add.f16x2 r4801, r4802, r4798;
}
{
add.f16x2 r4804, r4805, r4806;
}
{
add.f16x2 r4807, r4801, r4804;
}
{
add.f16x2 r4810, r4811, r4812;
}
{
add.f16x2 r4813, r4814, r4810;
}
{
add.f16x2 r4816, r4817, r4818;
}
{
add.f16x2 r4819, r4813, r4816;
}
{
add.f16x2 r4822, r4799, r4800;
}
{
mul.f16x2 r4825, r4822, r4790;
}
{
add.f16x2 r4828, r4802, r4825;
}
{
add.f16x2 r4831, r4805, r4806;
}
{
mul.f16x2 r4834, r4831, r4792;
}
{
add.f16x2 r4837, r4828, r4834;
}
{
sub.f16x2 r4840, r4811, r4812;
}
{
mul.f16x2 r4843, r4840, r4791;
}
{
sub.f16x2 r4846, r4817, r4818;
}
{
mul.f16x2 r4849, r4846, r4793;
}
{
add.f16x2 r4852, r4843, r4849;
}
{
sub.f16x2 r4855, r4837, r4852;
}
{
add.f16x2 r4858, r4799, r4800;
}
{
mul.f16x2 r4861, r4858, r4790;
}
{
add.f16x2 r4864, r4802, r4861;
}
{
add.f16x2 r4867, r4805, r4806;
}
{
mul.f16x2 r4870, r4867, r4792;
}
{
add.f16x2 r4873, r4864, r4870;
}
{
sub.f16x2 r4876, r4811, r4812;
}
{
mul.f16x2 r4879, r4876, r4791;
}
{
sub.f16x2 r4882, r4817, r4818;
}
{
mul.f16x2 r4885, r4882, r4793;
}
{
add.f16x2 r4888, r4879, r4885;
}
{
add.f16x2 r4891, r4873, r4888;
}
{
add.f16x2 r4894, r4799, r4800;
}
{
mul.f16x2 r4897, r4894, r4792;
}
{
add.f16x2 r4900, r4802, r4897;
}
{
add.f16x2 r4903, r4805, r4806;
}
{
mul.f16x2 r4906, r4903, r4794;
}
{
add.f16x2 r4909, r4900, r4906;
}
{
sub.f16x2 r4912, r4811, r4812;
}
{
mul.f16x2 r4915, r4912, r4793;
}
{
sub.f16x2 r4918, r4817, r4818;
}
{
mul.f16x2 r4921, r4918, r4796;
}
{
add.f16x2 r4924, r4915, r4921;
}
{
sub.f16x2 r4927, r4909, r4924;
}
{
add.f16x2 r4930, r4799, r4800;
}
{
mul.f16x2 r4933, r4930, r4792;
}
{
add.f16x2 r4936, r4802, r4933;
}
{
add.f16x2 r4939, r4805, r4806;
}
{
mul.f16x2 r4942, r4939, r4794;
}
{
add.f16x2 r4945, r4936, r4942;
}
{
sub.f16x2 r4948, r4811, r4812;
}
{
mul.f16x2 r4951, r4948, r4793;
}
{
sub.f16x2 r4954, r4817, r4818;
}
{
mul.f16x2 r4957, r4954, r4796;
}
{
add.f16x2 r4960, r4951, r4957;
}
{
add.f16x2 r4963, r4945, r4960;
}
{
add.f16x2 r4966, r4811, r4812;
}
{
mul.f16x2 r4969, r4966, r4790;
}
{
add.f16x2 r4972, r4814, r4969;
}
{
add.f16x2 r4975, r4817, r4818;
}
{
mul.f16x2 r4978, r4975, r4792;
}
{
add.f16x2 r4981, r4972, r4978;
}
{
sub.f16x2 r4984, r4799, r4800;
}
{
mul.f16x2 r4987, r4984, r4791;
}
{
sub.f16x2 r4990, r4805, r4806;
}
{
mul.f16x2 r4993, r4990, r4793;
}
{
add.f16x2 r4996, r4987, r4993;
}
{
add.f16x2 r4999, r4981, r4996;
}
{
add.f16x2 r5002, r4811, r4812;
}
{
mul.f16x2 r5005, r5002, r4790;
}
{
add.f16x2 r5008, r4814, r5005;
}
{
add.f16x2 r5011, r4817, r4818;
}
{
mul.f16x2 r5014, r5011, r4792;
}
{
add.f16x2 r5017, r5008, r5014;
}
{
sub.f16x2 r5020, r4799, r4800;
}
{
mul.f16x2 r5023, r5020, r4791;
}
{
sub.f16x2 r5026, r4805, r4806;
}
{
mul.f16x2 r5029, r5026, r4793;
}
{
add.f16x2 r5032, r5023, r5029;
}
{
sub.f16x2 r5035, r5017, r5032;
}
{
add.f16x2 r5038, r4811, r4812;
}
{
mul.f16x2 r5041, r5038, r4792;
}
{
add.f16x2 r5044, r4814, r5041;
}
{
add.f16x2 r5047, r4817, r4818;
}
{
mul.f16x2 r5050, r5047, r4794;
}
{
add.f16x2 r5053, r5044, r5050;
}
{
sub.f16x2 r5056, r4799, r4800;
}
{
mul.f16x2 r5059, r5056, r4793;
}
{
sub.f16x2 r5062, r4805, r4806;
}
{
mul.f16x2 r5065, r5062, r4796;
}
{
add.f16x2 r5068, r5059, r5065;
}
{
add.f16x2 r5071, r5053, r5068;
}
{
add.f16x2 r5074, r4811, r4812;
}
{
mul.f16x2 r5077, r5074, r4792;
}
{
add.f16x2 r5080, r4814, r5077;
}
{
add.f16x2 r5083, r4817, r4818;
}
{
mul.f16x2 r5086, r5083, r4794;
}
{
add.f16x2 r5089, r5080, r5086;
}
{
sub.f16x2 r5092, r4799, r4800;
}
{
mul.f16x2 r5095, r5092, r4793;
}
{
sub.f16x2 r5098, r4805, r4806;
}
{
mul.f16x2 r5101, r5098, r4796;
}
{
add.f16x2 r5104, r5095, r5101;
}
{
sub.f16x2 r5107, r5089, r5104;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r5110, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r5111, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r5112, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r5113, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r5114, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r5115, {low, high};
}
{
neg.f16x2 r5116, r5115;
}
{
add.f16x2 r5118, r5119, r5120;
}
{
add.f16x2 r5121, r5122, r5118;
}
{
add.f16x2 r5124, r5125, r5126;
}
{
add.f16x2 r5127, r5121, r5124;
}
{
add.f16x2 r5130, r5131, r5132;
}
{
add.f16x2 r5133, r5134, r5130;
}
{
add.f16x2 r5136, r5137, r5138;
}
{
add.f16x2 r5139, r5133, r5136;
}
{
add.f16x2 r5142, r5119, r5120;
}
{
mul.f16x2 r5145, r5142, r5110;
}
{
add.f16x2 r5148, r5122, r5145;
}
{
add.f16x2 r5151, r5125, r5126;
}
{
mul.f16x2 r5154, r5151, r5112;
}
{
add.f16x2 r5157, r5148, r5154;
}
{
sub.f16x2 r5160, r5131, r5132;
}
{
mul.f16x2 r5163, r5160, r5111;
}
{
sub.f16x2 r5166, r5137, r5138;
}
{
mul.f16x2 r5169, r5166, r5113;
}
{
add.f16x2 r5172, r5163, r5169;
}
{
sub.f16x2 r5175, r5157, r5172;
}
{
add.f16x2 r5178, r5119, r5120;
}
{
mul.f16x2 r5181, r5178, r5110;
}
{
add.f16x2 r5184, r5122, r5181;
}
{
add.f16x2 r5187, r5125, r5126;
}
{
mul.f16x2 r5190, r5187, r5112;
}
{
add.f16x2 r5193, r5184, r5190;
}
{
sub.f16x2 r5196, r5131, r5132;
}
{
mul.f16x2 r5199, r5196, r5111;
}
{
sub.f16x2 r5202, r5137, r5138;
}
{
mul.f16x2 r5205, r5202, r5113;
}
{
add.f16x2 r5208, r5199, r5205;
}
{
add.f16x2 r5211, r5193, r5208;
}
{
add.f16x2 r5214, r5119, r5120;
}
{
mul.f16x2 r5217, r5214, r5112;
}
{
add.f16x2 r5220, r5122, r5217;
}
{
add.f16x2 r5223, r5125, r5126;
}
{
mul.f16x2 r5226, r5223, r5114;
}
{
add.f16x2 r5229, r5220, r5226;
}
{
sub.f16x2 r5232, r5131, r5132;
}
{
mul.f16x2 r5235, r5232, r5113;
}
{
sub.f16x2 r5238, r5137, r5138;
}
{
mul.f16x2 r5241, r5238, r5116;
}
{
add.f16x2 r5244, r5235, r5241;
}
{
sub.f16x2 r5247, r5229, r5244;
}
{
add.f16x2 r5250, r5119, r5120;
}
{
mul.f16x2 r5253, r5250, r5112;
}
{
add.f16x2 r5256, r5122, r5253;
}
{
add.f16x2 r5259, r5125, r5126;
}
{
mul.f16x2 r5262, r5259, r5114;
}
{
add.f16x2 r5265, r5256, r5262;
}
{
sub.f16x2 r5268, r5131, r5132;
}
{
mul.f16x2 r5271, r5268, r5113;
}
{
sub.f16x2 r5274, r5137, r5138;
}
{
mul.f16x2 r5277, r5274, r5116;
}
{
add.f16x2 r5280, r5271, r5277;
}
{
add.f16x2 r5283, r5265, r5280;
}
{
add.f16x2 r5286, r5131, r5132;
}
{
mul.f16x2 r5289, r5286, r5110;
}
{
add.f16x2 r5292, r5134, r5289;
}
{
add.f16x2 r5295, r5137, r5138;
}
{
mul.f16x2 r5298, r5295, r5112;
}
{
add.f16x2 r5301, r5292, r5298;
}
{
sub.f16x2 r5304, r5119, r5120;
}
{
mul.f16x2 r5307, r5304, r5111;
}
{
sub.f16x2 r5310, r5125, r5126;
}
{
mul.f16x2 r5313, r5310, r5113;
}
{
add.f16x2 r5316, r5307, r5313;
}
{
add.f16x2 r5319, r5301, r5316;
}
{
add.f16x2 r5322, r5131, r5132;
}
{
mul.f16x2 r5325, r5322, r5110;
}
{
add.f16x2 r5328, r5134, r5325;
}
{
add.f16x2 r5331, r5137, r5138;
}
{
mul.f16x2 r5334, r5331, r5112;
}
{
add.f16x2 r5337, r5328, r5334;
}
{
sub.f16x2 r5340, r5119, r5120;
}
{
mul.f16x2 r5343, r5340, r5111;
}
{
sub.f16x2 r5346, r5125, r5126;
}
{
mul.f16x2 r5349, r5346, r5113;
}
{
add.f16x2 r5352, r5343, r5349;
}
{
sub.f16x2 r5355, r5337, r5352;
}
{
add.f16x2 r5358, r5131, r5132;
}
{
mul.f16x2 r5361, r5358, r5112;
}
{
add.f16x2 r5364, r5134, r5361;
}
{
add.f16x2 r5367, r5137, r5138;
}
{
mul.f16x2 r5370, r5367, r5114;
}
{
add.f16x2 r5373, r5364, r5370;
}
{
sub.f16x2 r5376, r5119, r5120;
}
{
mul.f16x2 r5379, r5376, r5113;
}
{
sub.f16x2 r5382, r5125, r5126;
}
{
mul.f16x2 r5385, r5382, r5116;
}
{
add.f16x2 r5388, r5379, r5385;
}
{
add.f16x2 r5391, r5373, r5388;
}
{
add.f16x2 r5394, r5131, r5132;
}
{
mul.f16x2 r5397, r5394, r5112;
}
{
add.f16x2 r5400, r5134, r5397;
}
{
add.f16x2 r5403, r5137, r5138;
}
{
mul.f16x2 r5406, r5403, r5114;
}
{
add.f16x2 r5409, r5400, r5406;
}
{
sub.f16x2 r5412, r5119, r5120;
}
{
mul.f16x2 r5415, r5412, r5113;
}
{
sub.f16x2 r5418, r5125, r5126;
}
{
mul.f16x2 r5421, r5418, r5116;
}
{
add.f16x2 r5424, r5415, r5421;
}
{
sub.f16x2 r5427, r5409, r5424;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f62;
cvt.rn.f16.f32 high, f62;
mov.b32 r5430, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f64;
cvt.rn.f16.f32 high, f64;
mov.b32 r5431, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f66;
cvt.rn.f16.f32 high, f66;
mov.b32 r5432, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f68;
cvt.rn.f16.f32 high, f68;
mov.b32 r5433, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f70;
cvt.rn.f16.f32 high, f70;
mov.b32 r5434, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f72;
cvt.rn.f16.f32 high, f72;
mov.b32 r5435, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r5436, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r5437, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r5440, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r5441, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r5444, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r5445, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r5446, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r5447, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r5452, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r5453, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r5460, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r5461, {low, high};
}
{
mul.f16x2 r5478, r4215, r5430;
}
{
mul.f16x2 r5481, r4359, r5431;
}
{
sub.f16x2 r5484, r5478, r5481;
}
{
mul.f16x2 r5487, r4215, r5431;
}
{
fma.rn.f16x2 r5490, r4359, r5430, r5487;
}
{
mul.f16x2 r5494, r4535, r5432;
}
{
mul.f16x2 r5497, r4679, r5433;
}
{
sub.f16x2 r5500, r5494, r5497;
}
{
mul.f16x2 r5503, r4535, r5433;
}
{
fma.rn.f16x2 r5506, r4679, r5432, r5503;
}
{
mul.f16x2 r5510, r4855, r5434;
}
{
mul.f16x2 r5513, r4999, r5435;
}
{
sub.f16x2 r5516, r5510, r5513;
}
{
mul.f16x2 r5519, r4855, r5435;
}
{
fma.rn.f16x2 r5522, r4999, r5434, r5519;
}
{
mul.f16x2 r5526, r5175, r5436;
}
{
mul.f16x2 r5529, r5319, r5437;
}
{
sub.f16x2 r5532, r5526, r5529;
}
{
mul.f16x2 r5535, r5175, r5437;
}
{
fma.rn.f16x2 r5538, r5319, r5436, r5535;
}
{
mul.f16x2 r5542, r4287, r5432;
}
{
mul.f16x2 r5545, r4431, r5433;
}
{
sub.f16x2 r5548, r5542, r5545;
}
{
mul.f16x2 r5551, r4287, r5433;
}
{
fma.rn.f16x2 r5554, r4431, r5432, r5551;
}
{
mul.f16x2 r5558, r4607, r5436;
}
{
mul.f16x2 r5561, r4751, r5437;
}
{
sub.f16x2 r5564, r5558, r5561;
}
{
mul.f16x2 r5567, r4607, r5437;
}
{
fma.rn.f16x2 r5570, r4751, r5436, r5567;
}
{
mul.f16x2 r5574, r4927, r5440;
}
{
mul.f16x2 r5577, r5071, r5441;
}
{
sub.f16x2 r5580, r5574, r5577;
}
{
mul.f16x2 r5583, r4927, r5441;
}
{
fma.rn.f16x2 r5586, r5071, r5440, r5583;
}
{
mul.f16x2 r5590, r5247, r5444;
}
{
mul.f16x2 r5593, r5391, r5445;
}
{
sub.f16x2 r5596, r5590, r5593;
}
{
mul.f16x2 r5599, r5247, r5445;
}
{
fma.rn.f16x2 r5602, r5391, r5444, r5599;
}
{
mul.f16x2 r5606, r4323, r5434;
}
{
mul.f16x2 r5609, r4467, r5435;
}
{
sub.f16x2 r5612, r5606, r5609;
}
{
mul.f16x2 r5615, r4323, r5435;
}
{
fma.rn.f16x2 r5618, r4467, r5434, r5615;
}
{
mul.f16x2 r5622, r4643, r5440;
}
{
mul.f16x2 r5625, r4787, r5441;
}
{
sub.f16x2 r5628, r5622, r5625;
}
{
mul.f16x2 r5631, r4643, r5441;
}
{
fma.rn.f16x2 r5634, r4787, r5440, r5631;
}
{
mul.f16x2 r5638, r4963, r5446;
}
{
mul.f16x2 r5641, r5107, r5447;
}
{
sub.f16x2 r5644, r5638, r5641;
}
{
mul.f16x2 r5647, r4963, r5447;
}
{
fma.rn.f16x2 r5650, r5107, r5446, r5647;
}
{
mul.f16x2 r5654, r5283, r5452;
}
{
mul.f16x2 r5657, r5427, r5453;
}
{
sub.f16x2 r5660, r5654, r5657;
}
{
mul.f16x2 r5663, r5283, r5453;
}
{
fma.rn.f16x2 r5666, r5427, r5452, r5663;
}
{
mul.f16x2 r5670, r4251, r5436;
}
{
mul.f16x2 r5673, r4395, r5437;
}
{
sub.f16x2 r5676, r5670, r5673;
}
{
mul.f16x2 r5679, r4251, r5437;
}
{
fma.rn.f16x2 r5682, r4395, r5436, r5679;
}
{
mul.f16x2 r5686, r4571, r5444;
}
{
mul.f16x2 r5689, r4715, r5445;
}
{
sub.f16x2 r5692, r5686, r5689;
}
{
mul.f16x2 r5695, r4571, r5445;
}
{
fma.rn.f16x2 r5698, r4715, r5444, r5695;
}
{
mul.f16x2 r5702, r4891, r5452;
}
{
mul.f16x2 r5705, r5035, r5453;
}
{
sub.f16x2 r5708, r5702, r5705;
}
{
mul.f16x2 r5711, r4891, r5453;
}
{
fma.rn.f16x2 r5714, r5035, r5452, r5711;
}
{
mul.f16x2 r5718, r5211, r5460;
}
{
mul.f16x2 r5721, r5355, r5461;
}
{
sub.f16x2 r5724, r5718, r5721;
}
{
mul.f16x2 r5727, r5211, r5461;
}
{
fma.rn.f16x2 r5730, r5355, r5460, r5727;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r5734, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r5735, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r5736, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r5737, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r5738, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r5739, {low, high};
}
{
neg.f16x2 r5740, r5739;
}
{
add.f16x2 r5742, r4167, r5127;
}
{
add.f16x2 r5745, r3847, r5742;
}
{
add.f16x2 r5748, r4487, r4807;
}
{
add.f16x2 r5751, r5745, r5748;
}
st.local.u32 [rd3], r5751;
{
add.f16x2 r5754, r4179, r5139;
}
{
add.f16x2 r5757, r3859, r5754;
}
{
add.f16x2 r5760, r4499, r4819;
}
{
add.f16x2 r5763, r5757, r5760;
}
st.local.u32 [rd4], r5763;
{
add.f16x2 r5766, r4167, r5127;
}
{
mul.f16x2 r5769, r5766, r5734;
}
{
add.f16x2 r5772, r3847, r5769;
}
{
add.f16x2 r5775, r4487, r4807;
}
{
mul.f16x2 r5778, r5775, r5736;
}
{
add.f16x2 r5781, r5772, r5778;
}
{
sub.f16x2 r5784, r4179, r5139;
}
{
mul.f16x2 r5787, r5784, r5735;
}
{
sub.f16x2 r5790, r4499, r4819;
}
{
mul.f16x2 r5793, r5790, r5737;
}
{
add.f16x2 r5796, r5787, r5793;
}
{
sub.f16x2 r5799, r5781, r5796;
}
st.local.u32 [rd4+36], r5799;
{
add.f16x2 r5802, r4167, r5127;
}
{
mul.f16x2 r5805, r5802, r5734;
}
{
add.f16x2 r5808, r3847, r5805;
}
{
add.f16x2 r5811, r4487, r4807;
}
{
mul.f16x2 r5814, r5811, r5736;
}
{
add.f16x2 r5817, r5808, r5814;
}
{
sub.f16x2 r5820, r4179, r5139;
}
{
mul.f16x2 r5823, r5820, r5735;
}
{
sub.f16x2 r5826, r4499, r4819;
}
{
mul.f16x2 r5829, r5826, r5737;
}
{
add.f16x2 r5832, r5823, r5829;
}
{
add.f16x2 r5835, r5817, r5832;
}
st.local.u32 [rd4+156], r5835;
{
add.f16x2 r5838, r4167, r5127;
}
{
mul.f16x2 r5841, r5838, r5736;
}
{
add.f16x2 r5844, r3847, r5841;
}
{
add.f16x2 r5847, r4487, r4807;
}
{
mul.f16x2 r5850, r5847, r5738;
}
{
add.f16x2 r5853, r5844, r5850;
}
{
sub.f16x2 r5856, r4179, r5139;
}
{
mul.f16x2 r5859, r5856, r5737;
}
{
sub.f16x2 r5862, r4499, r4819;
}
{
mul.f16x2 r5865, r5862, r5740;
}
{
add.f16x2 r5868, r5859, r5865;
}
{
sub.f16x2 r5871, r5853, r5868;
}
st.local.u32 [rd4+76], r5871;
{
add.f16x2 r5874, r4167, r5127;
}
{
mul.f16x2 r5877, r5874, r5736;
}
{
add.f16x2 r5880, r3847, r5877;
}
{
add.f16x2 r5883, r4487, r4807;
}
{
mul.f16x2 r5886, r5883, r5738;
}
{
add.f16x2 r5889, r5880, r5886;
}
{
sub.f16x2 r5892, r4179, r5139;
}
{
mul.f16x2 r5895, r5892, r5737;
}
{
sub.f16x2 r5898, r4499, r4819;
}
{
mul.f16x2 r5901, r5898, r5740;
}
{
add.f16x2 r5904, r5895, r5901;
}
{
add.f16x2 r5907, r5889, r5904;
}
st.local.u32 [rd4+116], r5907;
{
add.f16x2 r5910, r4179, r5139;
}
{
mul.f16x2 r5913, r5910, r5734;
}
{
add.f16x2 r5916, r3859, r5913;
}
{
add.f16x2 r5919, r4499, r4819;
}
{
mul.f16x2 r5922, r5919, r5736;
}
{
add.f16x2 r5925, r5916, r5922;
}
{
sub.f16x2 r5928, r4167, r5127;
}
{
mul.f16x2 r5931, r5928, r5735;
}
{
sub.f16x2 r5934, r4487, r4807;
}
{
mul.f16x2 r5937, r5934, r5737;
}
{
add.f16x2 r5940, r5931, r5937;
}
{
add.f16x2 r5943, r5925, r5940;
}
st.local.u32 [rd4+40], r5943;
{
add.f16x2 r5946, r4179, r5139;
}
{
mul.f16x2 r5949, r5946, r5734;
}
{
add.f16x2 r5952, r3859, r5949;
}
{
add.f16x2 r5955, r4499, r4819;
}
{
mul.f16x2 r5958, r5955, r5736;
}
{
add.f16x2 r5961, r5952, r5958;
}
{
sub.f16x2 r5964, r4167, r5127;
}
{
mul.f16x2 r5967, r5964, r5735;
}
{
sub.f16x2 r5970, r4487, r4807;
}
{
mul.f16x2 r5973, r5970, r5737;
}
{
add.f16x2 r5976, r5967, r5973;
}
{
sub.f16x2 r5979, r5961, r5976;
}
st.local.u32 [rd4+160], r5979;
{
add.f16x2 r5982, r4179, r5139;
}
{
mul.f16x2 r5985, r5982, r5736;
}
{
add.f16x2 r5988, r3859, r5985;
}
{
add.f16x2 r5991, r4499, r4819;
}
{
mul.f16x2 r5994, r5991, r5738;
}
{
add.f16x2 r5997, r5988, r5994;
}
{
sub.f16x2 r6000, r4167, r5127;
}
{
mul.f16x2 r6003, r6000, r5737;
}
{
sub.f16x2 r6006, r4487, r4807;
}
{
mul.f16x2 r6009, r6006, r5740;
}
{
add.f16x2 r6012, r6003, r6009;
}
{
add.f16x2 r6015, r5997, r6012;
}
st.local.u32 [rd4+80], r6015;
{
add.f16x2 r6018, r4179, r5139;
}
{
mul.f16x2 r6021, r6018, r5736;
}
{
add.f16x2 r6024, r3859, r6021;
}
{
add.f16x2 r6027, r4499, r4819;
}
{
mul.f16x2 r6030, r6027, r5738;
}
{
add.f16x2 r6033, r6024, r6030;
}
{
sub.f16x2 r6036, r4167, r5127;
}
{
mul.f16x2 r6039, r6036, r5737;
}
{
sub.f16x2 r6042, r4487, r4807;
}
{
mul.f16x2 r6045, r6042, r5740;
}
{
add.f16x2 r6048, r6039, r6045;
}
{
sub.f16x2 r6051, r6033, r6048;
}
st.local.u32 [rd4+120], r6051;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6054, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6055, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r6056, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r6057, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6058, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6059, {low, high};
}
{
neg.f16x2 r6060, r6059;
}
{
add.f16x2 r6062, r5484, r5532;
}
{
add.f16x2 r6065, r3895, r6062;
}
{
add.f16x2 r6068, r5500, r5516;
}
{
add.f16x2 r11251, r6065, r6068;
}
st.local.u32 [rd4+4], r11251;
{
add.f16x2 r6074, r5490, r5538;
}
{
add.f16x2 r6077, r4039, r6074;
}
{
add.f16x2 r6080, r5506, r5522;
}
{
add.f16x2 r6083, r6077, r6080;
}
st.local.u32 [rd4+8], r6083;
{
add.f16x2 r6086, r5484, r5532;
}
{
mul.f16x2 r6089, r6086, r6054;
}
{
add.f16x2 r6092, r3895, r6089;
}
{
add.f16x2 r6095, r5500, r5516;
}
{
mul.f16x2 r6098, r6095, r6056;
}
{
add.f16x2 r6101, r6092, r6098;
}
{
sub.f16x2 r6104, r5490, r5538;
}
{
mul.f16x2 r6107, r6104, r6055;
}
{
sub.f16x2 r6110, r5506, r5522;
}
{
mul.f16x2 r6113, r6110, r6057;
}
{
add.f16x2 r6116, r6107, r6113;
}
{
sub.f16x2 r6119, r6101, r6116;
}
st.local.u32 [rd4+44], r6119;
{
add.f16x2 r6122, r5484, r5532;
}
{
mul.f16x2 r6125, r6122, r6054;
}
{
add.f16x2 r6128, r3895, r6125;
}
{
add.f16x2 r6131, r5500, r5516;
}
{
mul.f16x2 r6134, r6131, r6056;
}
{
add.f16x2 r6137, r6128, r6134;
}
{
sub.f16x2 r6140, r5490, r5538;
}
{
mul.f16x2 r6143, r6140, r6055;
}
{
sub.f16x2 r6146, r5506, r5522;
}
{
mul.f16x2 r6149, r6146, r6057;
}
{
add.f16x2 r6152, r6143, r6149;
}
{
add.f16x2 r6155, r6137, r6152;
}
st.local.u32 [rd4+164], r6155;
{
add.f16x2 r6158, r5484, r5532;
}
{
mul.f16x2 r6161, r6158, r6056;
}
{
add.f16x2 r6164, r3895, r6161;
}
{
add.f16x2 r6167, r5500, r5516;
}
{
mul.f16x2 r6170, r6167, r6058;
}
{
add.f16x2 r6173, r6164, r6170;
}
{
sub.f16x2 r6176, r5490, r5538;
}
{
mul.f16x2 r6179, r6176, r6057;
}
{
sub.f16x2 r6182, r5506, r5522;
}
{
mul.f16x2 r6185, r6182, r6060;
}
{
add.f16x2 r6188, r6179, r6185;
}
{
sub.f16x2 r6191, r6173, r6188;
}
st.local.u32 [rd4+84], r6191;
{
add.f16x2 r6194, r5484, r5532;
}
{
mul.f16x2 r6197, r6194, r6056;
}
{
add.f16x2 r6200, r3895, r6197;
}
{
add.f16x2 r6203, r5500, r5516;
}
{
mul.f16x2 r6206, r6203, r6058;
}
{
add.f16x2 r6209, r6200, r6206;
}
{
sub.f16x2 r6212, r5490, r5538;
}
{
mul.f16x2 r6215, r6212, r6057;
}
{
sub.f16x2 r6218, r5506, r5522;
}
{
mul.f16x2 r6221, r6218, r6060;
}
{
add.f16x2 r6224, r6215, r6221;
}
{
add.f16x2 r6227, r6209, r6224;
}
st.local.u32 [rd4+124], r6227;
{
add.f16x2 r6230, r5490, r5538;
}
{
mul.f16x2 r6233, r6230, r6054;
}
{
add.f16x2 r6236, r4039, r6233;
}
{
add.f16x2 r6239, r5506, r5522;
}
{
mul.f16x2 r6242, r6239, r6056;
}
{
add.f16x2 r6245, r6236, r6242;
}
{
sub.f16x2 r6248, r5484, r5532;
}
{
mul.f16x2 r6251, r6248, r6055;
}
{
sub.f16x2 r6254, r5500, r5516;
}
{
mul.f16x2 r6257, r6254, r6057;
}
{
add.f16x2 r6260, r6251, r6257;
}
{
add.f16x2 r6263, r6245, r6260;
}
st.local.u32 [rd4+48], r6263;
{
add.f16x2 r6266, r5490, r5538;
}
{
mul.f16x2 r6269, r6266, r6054;
}
{
add.f16x2 r6272, r4039, r6269;
}
{
add.f16x2 r6275, r5506, r5522;
}
{
mul.f16x2 r6278, r6275, r6056;
}
{
add.f16x2 r6281, r6272, r6278;
}
{
sub.f16x2 r6284, r5484, r5532;
}
{
mul.f16x2 r6287, r6284, r6055;
}
{
sub.f16x2 r6290, r5500, r5516;
}
{
mul.f16x2 r6293, r6290, r6057;
}
{
add.f16x2 r6296, r6287, r6293;
}
{
sub.f16x2 r6299, r6281, r6296;
}
st.local.u32 [rd4+168], r6299;
{
add.f16x2 r6302, r5490, r5538;
}
{
mul.f16x2 r6305, r6302, r6056;
}
{
add.f16x2 r6308, r4039, r6305;
}
{
add.f16x2 r6311, r5506, r5522;
}
{
mul.f16x2 r6314, r6311, r6058;
}
{
add.f16x2 r6317, r6308, r6314;
}
{
sub.f16x2 r6320, r5484, r5532;
}
{
mul.f16x2 r6323, r6320, r6057;
}
{
sub.f16x2 r6326, r5500, r5516;
}
{
mul.f16x2 r6329, r6326, r6060;
}
{
add.f16x2 r6332, r6323, r6329;
}
{
add.f16x2 r6335, r6317, r6332;
}
st.local.u32 [rd4+88], r6335;
{
add.f16x2 r6338, r5490, r5538;
}
{
mul.f16x2 r6341, r6338, r6056;
}
{
add.f16x2 r6344, r4039, r6341;
}
{
add.f16x2 r6347, r5506, r5522;
}
{
mul.f16x2 r6350, r6347, r6058;
}
{
add.f16x2 r6353, r6344, r6350;
}
{
sub.f16x2 r6356, r5484, r5532;
}
{
mul.f16x2 r6359, r6356, r6057;
}
{
sub.f16x2 r6362, r5500, r5516;
}
{
mul.f16x2 r6365, r6362, r6060;
}
{
add.f16x2 r6368, r6359, r6365;
}
{
sub.f16x2 r6371, r6353, r6368;
}
st.local.u32 [rd4+128], r6371;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6374, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6375, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r6376, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r6377, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6378, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6379, {low, high};
}
{
neg.f16x2 r6380, r6379;
}
{
add.f16x2 r6382, r5548, r5596;
}
{
add.f16x2 r6385, r3967, r6382;
}
{
add.f16x2 r6388, r5564, r5580;
}
{
add.f16x2 r6391, r6385, r6388;
}
st.local.u32 [rd4+12], r6391;
{
add.f16x2 r6394, r5554, r5602;
}
{
add.f16x2 r6397, r4111, r6394;
}
{
add.f16x2 r6400, r5570, r5586;
}
{
add.f16x2 r6403, r6397, r6400;
}
st.local.u32 [rd4+16], r6403;
{
add.f16x2 r6406, r5548, r5596;
}
{
mul.f16x2 r6409, r6406, r6374;
}
{
add.f16x2 r6412, r3967, r6409;
}
{
add.f16x2 r6415, r5564, r5580;
}
{
mul.f16x2 r6418, r6415, r6376;
}
{
add.f16x2 r6421, r6412, r6418;
}
{
sub.f16x2 r6424, r5554, r5602;
}
{
mul.f16x2 r6427, r6424, r6375;
}
{
sub.f16x2 r6430, r5570, r5586;
}
{
mul.f16x2 r6433, r6430, r6377;
}
{
add.f16x2 r6436, r6427, r6433;
}
{
sub.f16x2 r6439, r6421, r6436;
}
st.local.u32 [rd4+52], r6439;
{
add.f16x2 r6442, r5548, r5596;
}
{
mul.f16x2 r6445, r6442, r6374;
}
{
add.f16x2 r6448, r3967, r6445;
}
{
add.f16x2 r6451, r5564, r5580;
}
{
mul.f16x2 r6454, r6451, r6376;
}
{
add.f16x2 r6457, r6448, r6454;
}
{
sub.f16x2 r6460, r5554, r5602;
}
{
mul.f16x2 r6463, r6460, r6375;
}
{
sub.f16x2 r6466, r5570, r5586;
}
{
mul.f16x2 r6469, r6466, r6377;
}
{
add.f16x2 r6472, r6463, r6469;
}
{
add.f16x2 r6475, r6457, r6472;
}
st.local.u32 [rd4+172], r6475;
{
add.f16x2 r6478, r5548, r5596;
}
{
mul.f16x2 r6481, r6478, r6376;
}
{
add.f16x2 r6484, r3967, r6481;
}
{
add.f16x2 r6487, r5564, r5580;
}
{
mul.f16x2 r6490, r6487, r6378;
}
{
add.f16x2 r6493, r6484, r6490;
}
{
sub.f16x2 r6496, r5554, r5602;
}
{
mul.f16x2 r6499, r6496, r6377;
}
{
sub.f16x2 r6502, r5570, r5586;
}
{
mul.f16x2 r6505, r6502, r6380;
}
{
add.f16x2 r6508, r6499, r6505;
}
{
sub.f16x2 r6511, r6493, r6508;
}
st.local.u32 [rd4+92], r6511;
{
add.f16x2 r6514, r5548, r5596;
}
{
mul.f16x2 r6517, r6514, r6376;
}
{
add.f16x2 r6520, r3967, r6517;
}
{
add.f16x2 r6523, r5564, r5580;
}
{
mul.f16x2 r6526, r6523, r6378;
}
{
add.f16x2 r6529, r6520, r6526;
}
{
sub.f16x2 r6532, r5554, r5602;
}
{
mul.f16x2 r6535, r6532, r6377;
}
{
sub.f16x2 r6538, r5570, r5586;
}
{
mul.f16x2 r6541, r6538, r6380;
}
{
add.f16x2 r6544, r6535, r6541;
}
{
add.f16x2 r6547, r6529, r6544;
}
st.local.u32 [rd4+132], r6547;
{
add.f16x2 r6550, r5554, r5602;
}
{
mul.f16x2 r6553, r6550, r6374;
}
{
add.f16x2 r6556, r4111, r6553;
}
{
add.f16x2 r6559, r5570, r5586;
}
{
mul.f16x2 r6562, r6559, r6376;
}
{
add.f16x2 r6565, r6556, r6562;
}
{
sub.f16x2 r6568, r5548, r5596;
}
{
mul.f16x2 r6571, r6568, r6375;
}
{
sub.f16x2 r6574, r5564, r5580;
}
{
mul.f16x2 r6577, r6574, r6377;
}
{
add.f16x2 r6580, r6571, r6577;
}
{
add.f16x2 r6583, r6565, r6580;
}
st.local.u32 [rd4+56], r6583;
{
add.f16x2 r6586, r5554, r5602;
}
{
mul.f16x2 r6589, r6586, r6374;
}
{
add.f16x2 r6592, r4111, r6589;
}
{
add.f16x2 r6595, r5570, r5586;
}
{
mul.f16x2 r6598, r6595, r6376;
}
{
add.f16x2 r6601, r6592, r6598;
}
{
sub.f16x2 r6604, r5548, r5596;
}
{
mul.f16x2 r6607, r6604, r6375;
}
{
sub.f16x2 r6610, r5564, r5580;
}
{
mul.f16x2 r6613, r6610, r6377;
}
{
add.f16x2 r6616, r6607, r6613;
}
{
sub.f16x2 r6619, r6601, r6616;
}
st.local.u32 [rd4+176], r6619;
{
add.f16x2 r6622, r5554, r5602;
}
{
mul.f16x2 r6625, r6622, r6376;
}
{
add.f16x2 r6628, r4111, r6625;
}
{
add.f16x2 r6631, r5570, r5586;
}
{
mul.f16x2 r6634, r6631, r6378;
}
{
add.f16x2 r6637, r6628, r6634;
}
{
sub.f16x2 r6640, r5548, r5596;
}
{
mul.f16x2 r6643, r6640, r6377;
}
{
sub.f16x2 r6646, r5564, r5580;
}
{
mul.f16x2 r6649, r6646, r6380;
}
{
add.f16x2 r6652, r6643, r6649;
}
{
add.f16x2 r6655, r6637, r6652;
}
st.local.u32 [rd4+96], r6655;
{
add.f16x2 r6658, r5554, r5602;
}
{
mul.f16x2 r6661, r6658, r6376;
}
{
add.f16x2 r6664, r4111, r6661;
}
{
add.f16x2 r6667, r5570, r5586;
}
{
mul.f16x2 r6670, r6667, r6378;
}
{
add.f16x2 r6673, r6664, r6670;
}
{
sub.f16x2 r6676, r5548, r5596;
}
{
mul.f16x2 r6679, r6676, r6377;
}
{
sub.f16x2 r6682, r5564, r5580;
}
{
mul.f16x2 r6685, r6682, r6380;
}
{
add.f16x2 r6688, r6679, r6685;
}
{
sub.f16x2 r6691, r6673, r6688;
}
st.local.u32 [rd4+136], r6691;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6694, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6695, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r6696, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r6697, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6698, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6699, {low, high};
}
{
neg.f16x2 r6700, r6699;
}
{
add.f16x2 r6702, r5612, r5660;
}
{
add.f16x2 r6705, r4003, r6702;
}
{
add.f16x2 r6708, r5628, r5644;
}
{
add.f16x2 r6711, r6705, r6708;
}
st.local.u32 [rd4+20], r6711;
{
add.f16x2 r6714, r5618, r5666;
}
{
add.f16x2 r6717, r4147, r6714;
}
{
add.f16x2 r6720, r5634, r5650;
}
{
add.f16x2 r6723, r6717, r6720;
}
st.local.u32 [rd4+24], r6723;
{
add.f16x2 r6726, r5612, r5660;
}
{
mul.f16x2 r6729, r6726, r6694;
}
{
add.f16x2 r6732, r4003, r6729;
}
{
add.f16x2 r6735, r5628, r5644;
}
{
mul.f16x2 r6738, r6735, r6696;
}
{
add.f16x2 r6741, r6732, r6738;
}
{
sub.f16x2 r6744, r5618, r5666;
}
{
mul.f16x2 r6747, r6744, r6695;
}
{
sub.f16x2 r6750, r5634, r5650;
}
{
mul.f16x2 r6753, r6750, r6697;
}
{
add.f16x2 r6756, r6747, r6753;
}
{
sub.f16x2 r6759, r6741, r6756;
}
st.local.u32 [rd4+60], r6759;
{
add.f16x2 r6762, r5612, r5660;
}
{
mul.f16x2 r6765, r6762, r6694;
}
{
add.f16x2 r6768, r4003, r6765;
}
{
add.f16x2 r6771, r5628, r5644;
}
{
mul.f16x2 r6774, r6771, r6696;
}
{
add.f16x2 r6777, r6768, r6774;
}
{
sub.f16x2 r6780, r5618, r5666;
}
{
mul.f16x2 r6783, r6780, r6695;
}
{
sub.f16x2 r6786, r5634, r5650;
}
{
mul.f16x2 r6789, r6786, r6697;
}
{
add.f16x2 r6792, r6783, r6789;
}
{
add.f16x2 r6795, r6777, r6792;
}
st.local.u32 [rd4+180], r6795;
{
add.f16x2 r6798, r5612, r5660;
}
{
mul.f16x2 r6801, r6798, r6696;
}
{
add.f16x2 r6804, r4003, r6801;
}
{
add.f16x2 r6807, r5628, r5644;
}
{
mul.f16x2 r6810, r6807, r6698;
}
{
add.f16x2 r6813, r6804, r6810;
}
{
sub.f16x2 r6816, r5618, r5666;
}
{
mul.f16x2 r6819, r6816, r6697;
}
{
sub.f16x2 r6822, r5634, r5650;
}
{
mul.f16x2 r6825, r6822, r6700;
}
{
add.f16x2 r6828, r6819, r6825;
}
{
sub.f16x2 r6831, r6813, r6828;
}
st.local.u32 [rd4+100], r6831;
{
add.f16x2 r6834, r5612, r5660;
}
{
mul.f16x2 r6837, r6834, r6696;
}
{
add.f16x2 r6840, r4003, r6837;
}
{
add.f16x2 r6843, r5628, r5644;
}
{
mul.f16x2 r6846, r6843, r6698;
}
{
add.f16x2 r6849, r6840, r6846;
}
{
sub.f16x2 r6852, r5618, r5666;
}
{
mul.f16x2 r6855, r6852, r6697;
}
{
sub.f16x2 r6858, r5634, r5650;
}
{
mul.f16x2 r6861, r6858, r6700;
}
{
add.f16x2 r6864, r6855, r6861;
}
{
add.f16x2 r6867, r6849, r6864;
}
st.local.u32 [rd4+140], r6867;
{
add.f16x2 r6870, r5618, r5666;
}
{
mul.f16x2 r6873, r6870, r6694;
}
{
add.f16x2 r6876, r4147, r6873;
}
{
add.f16x2 r6879, r5634, r5650;
}
{
mul.f16x2 r6882, r6879, r6696;
}
{
add.f16x2 r6885, r6876, r6882;
}
{
sub.f16x2 r6888, r5612, r5660;
}
{
mul.f16x2 r6891, r6888, r6695;
}
{
sub.f16x2 r6894, r5628, r5644;
}
{
mul.f16x2 r6897, r6894, r6697;
}
{
add.f16x2 r6900, r6891, r6897;
}
{
add.f16x2 r6903, r6885, r6900;
}
st.local.u32 [rd4+64], r6903;
{
add.f16x2 r6906, r5618, r5666;
}
{
mul.f16x2 r6909, r6906, r6694;
}
{
add.f16x2 r6912, r4147, r6909;
}
{
add.f16x2 r6915, r5634, r5650;
}
{
mul.f16x2 r6918, r6915, r6696;
}
{
add.f16x2 r6921, r6912, r6918;
}
{
sub.f16x2 r6924, r5612, r5660;
}
{
mul.f16x2 r6927, r6924, r6695;
}
{
sub.f16x2 r6930, r5628, r5644;
}
{
mul.f16x2 r6933, r6930, r6697;
}
{
add.f16x2 r6936, r6927, r6933;
}
{
sub.f16x2 r6939, r6921, r6936;
}
st.local.u32 [rd4+184], r6939;
{
add.f16x2 r6942, r5618, r5666;
}
{
mul.f16x2 r6945, r6942, r6696;
}
{
add.f16x2 r6948, r4147, r6945;
}
{
add.f16x2 r6951, r5634, r5650;
}
{
mul.f16x2 r6954, r6951, r6698;
}
{
add.f16x2 r6957, r6948, r6954;
}
{
sub.f16x2 r6960, r5612, r5660;
}
{
mul.f16x2 r6963, r6960, r6697;
}
{
sub.f16x2 r6966, r5628, r5644;
}
{
mul.f16x2 r6969, r6966, r6700;
}
{
add.f16x2 r6972, r6963, r6969;
}
{
add.f16x2 r6975, r6957, r6972;
}
st.local.u32 [rd4+104], r6975;
{
add.f16x2 r6978, r5618, r5666;
}
{
mul.f16x2 r6981, r6978, r6696;
}
{
add.f16x2 r6984, r4147, r6981;
}
{
add.f16x2 r6987, r5634, r5650;
}
{
mul.f16x2 r6990, r6987, r6698;
}
{
add.f16x2 r6993, r6984, r6990;
}
{
sub.f16x2 r6996, r5612, r5660;
}
{
mul.f16x2 r6999, r6996, r6697;
}
{
sub.f16x2 r7002, r5628, r5644;
}
{
mul.f16x2 r7005, r7002, r6700;
}
{
add.f16x2 r7008, r6999, r7005;
}
{
sub.f16x2 r7011, r6993, r7008;
}
st.local.u32 [rd4+144], r7011;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r7014, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r7015, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r7016, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r7017, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r7018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r7019, {low, high};
}
{
neg.f16x2 r7020, r7019;
}
{
add.f16x2 r7022, r5676, r5724;
}
{
add.f16x2 r7025, r3931, r7022;
}
{
add.f16x2 r7028, r5692, r5708;
}
{
add.f16x2 r7031, r7025, r7028;
}
st.local.u32 [rd4+28], r7031;
{
add.f16x2 r7034, r5682, r5730;
}
{
add.f16x2 r7037, r4075, r7034;
}
{
add.f16x2 r7040, r5698, r5714;
}
{
add.f16x2 r7043, r7037, r7040;
}
st.local.u32 [rd4+32], r7043;
{
add.f16x2 r7046, r5676, r5724;
}
{
mul.f16x2 r7049, r7046, r7014;
}
{
add.f16x2 r7052, r3931, r7049;
}
{
add.f16x2 r7055, r5692, r5708;
}
{
mul.f16x2 r7058, r7055, r7016;
}
{
add.f16x2 r7061, r7052, r7058;
}
{
sub.f16x2 r7064, r5682, r5730;
}
{
mul.f16x2 r7067, r7064, r7015;
}
{
sub.f16x2 r7070, r5698, r5714;
}
{
mul.f16x2 r7073, r7070, r7017;
}
{
add.f16x2 r7076, r7067, r7073;
}
{
sub.f16x2 r7079, r7061, r7076;
}
st.local.u32 [rd4+68], r7079;
{
add.f16x2 r7082, r5676, r5724;
}
{
mul.f16x2 r7085, r7082, r7014;
}
{
add.f16x2 r7088, r3931, r7085;
}
{
add.f16x2 r7091, r5692, r5708;
}
{
mul.f16x2 r7094, r7091, r7016;
}
{
add.f16x2 r7097, r7088, r7094;
}
{
sub.f16x2 r7100, r5682, r5730;
}
{
mul.f16x2 r7103, r7100, r7015;
}
{
sub.f16x2 r7106, r5698, r5714;
}
{
mul.f16x2 r7109, r7106, r7017;
}
{
add.f16x2 r7112, r7103, r7109;
}
{
add.f16x2 r7115, r7097, r7112;
}
st.local.u32 [rd4+188], r7115;
{
add.f16x2 r7118, r5676, r5724;
}
{
mul.f16x2 r7121, r7118, r7016;
}
{
add.f16x2 r7124, r3931, r7121;
}
{
add.f16x2 r7127, r5692, r5708;
}
{
mul.f16x2 r7130, r7127, r7018;
}
{
add.f16x2 r7133, r7124, r7130;
}
{
sub.f16x2 r7136, r5682, r5730;
}
{
mul.f16x2 r7139, r7136, r7017;
}
{
sub.f16x2 r7142, r5698, r5714;
}
{
mul.f16x2 r7145, r7142, r7020;
}
{
add.f16x2 r7148, r7139, r7145;
}
{
sub.f16x2 r7151, r7133, r7148;
}
st.local.u32 [rd4+108], r7151;
{
add.f16x2 r7154, r5676, r5724;
}
{
mul.f16x2 r7157, r7154, r7016;
}
{
add.f16x2 r7160, r3931, r7157;
}
{
add.f16x2 r7163, r5692, r5708;
}
{
mul.f16x2 r7166, r7163, r7018;
}
{
add.f16x2 r7169, r7160, r7166;
}
{
sub.f16x2 r7172, r5682, r5730;
}
{
mul.f16x2 r7175, r7172, r7017;
}
{
sub.f16x2 r7178, r5698, r5714;
}
{
mul.f16x2 r7181, r7178, r7020;
}
{
add.f16x2 r7184, r7175, r7181;
}
{
add.f16x2 r7187, r7169, r7184;
}
st.local.u32 [rd4+148], r7187;
{
add.f16x2 r7190, r5682, r5730;
}
{
mul.f16x2 r7193, r7190, r7014;
}
{
add.f16x2 r7196, r4075, r7193;
}
{
add.f16x2 r7199, r5698, r5714;
}
{
mul.f16x2 r7202, r7199, r7016;
}
{
add.f16x2 r7205, r7196, r7202;
}
{
sub.f16x2 r7208, r5676, r5724;
}
{
mul.f16x2 r7211, r7208, r7015;
}
{
sub.f16x2 r7214, r5692, r5708;
}
{
mul.f16x2 r7217, r7214, r7017;
}
{
add.f16x2 r7220, r7211, r7217;
}
{
add.f16x2 r7223, r7205, r7220;
}
st.local.u32 [rd4+72], r7223;
{
add.f16x2 r7226, r5682, r5730;
}
{
mul.f16x2 r7229, r7226, r7014;
}
{
add.f16x2 r7232, r4075, r7229;
}
{
add.f16x2 r7235, r5698, r5714;
}
{
mul.f16x2 r7238, r7235, r7016;
}
{
add.f16x2 r7241, r7232, r7238;
}
{
sub.f16x2 r7244, r5676, r5724;
}
{
mul.f16x2 r7247, r7244, r7015;
}
{
sub.f16x2 r7250, r5692, r5708;
}
{
mul.f16x2 r7253, r7250, r7017;
}
{
add.f16x2 r7256, r7247, r7253;
}
{
sub.f16x2 r7259, r7241, r7256;
}
st.local.u32 [rd4+192], r7259;
{
add.f16x2 r7262, r5682, r5730;
}
{
mul.f16x2 r7265, r7262, r7016;
}
{
add.f16x2 r7268, r4075, r7265;
}
{
add.f16x2 r7271, r5698, r5714;
}
{
mul.f16x2 r7274, r7271, r7018;
}
{
add.f16x2 r7277, r7268, r7274;
}
{
sub.f16x2 r7280, r5676, r5724;
}
{
mul.f16x2 r7283, r7280, r7017;
}
{
sub.f16x2 r7286, r5692, r5708;
}
{
mul.f16x2 r7289, r7286, r7020;
}
{
add.f16x2 r7292, r7283, r7289;
}
{
add.f16x2 r7295, r7277, r7292;
}
st.local.u32 [rd4+112], r7295;
{
add.f16x2 r7298, r5682, r5730;
}
{
mul.f16x2 r7301, r7298, r7016;
}
{
add.f16x2 r7304, r4075, r7301;
}
{
add.f16x2 r7307, r5698, r5714;
}
{
mul.f16x2 r7310, r7307, r7018;
}
{
add.f16x2 r7313, r7304, r7310;
}
{
sub.f16x2 r7316, r5676, r5724;
}
{
mul.f16x2 r7319, r7316, r7017;
}
{
sub.f16x2 r7322, r5692, r5708;
}
{
mul.f16x2 r7325, r7322, r7020;
}
{
add.f16x2 r7328, r7319, r7325;
}
{
sub.f16x2 r7331, r7313, r7328;
}
st.local.u32 [rd4+152], r7331;
mul.wide.u32 rd13, r3, 1374389535;
shr.u64 rd14, rd13, 35;
cvt.u32.u64 r15, rd14;
cvt.rn.f32.u32 f460, r15;
mul.f32 f461, f460, 0f3C24B5BE;
cos.approx.f32 f456, f461;
sin.approx.f32 f462, f461;
neg.f32 f457, f462;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f457;
mov.b32 r11253, {low, high};
}
mov.u32 r11252, 1;
{
.reg .f16 low, high;
mov.b32 {low, high}, r11253;
mov.b32 r7412, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11253;
mov.b32 r7414, {high, high};
}
bra.uni LBB1_4;
LBB1_5:
ld.local.u32 r11251, [rd7+60];
LBB1_4:
{
.reg .f16 low, high;
mov.b32 {low, high}, r11253;
mov.b32 r7392, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11253;
mov.b32 r7394, {high, high};
}
mul.wide.u32 rd15, r11252, 8;
add.s64 rd16, rd3, rd15;
add.s64 rd7, rd16, 4;
ld.local.u32 r7397, [rd16+4];
{
mul.f16x2 r7396, r7397, r7394;
}
{
neg.f16x2 r7399, r7396;
}
{
fma.rn.f16x2 r7401, r11251, r7392, r7399;
}
st.local.u32 [rd16], r7401;
{
mul.f16x2 r7405, r11251, r7394;
}
{
fma.rn.f16x2 r7408, r7397, r7392, r7405;
}
st.local.u32 [rd16+4], r7408;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7416, {low, high};
}
{
mul.f16x2 r7417, r7414, r7416;
}
{
mul.f16x2 r7420, r11253, r7412;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11253;
mov.b32 r7423, {high, low};
}
{
fma.rn.f16x2 r7425, r7417, r7423, r7420;
}
ld.local.u32 r7443, [rd16+8];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7425;
mov.b32 r7429, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7425;
mov.b32 r7431, {high, high};
}
ld.local.u32 r7446, [rd16+12];
{
mul.f16x2 r7433, r7446, r7431;
}
{
neg.f16x2 r7436, r7433;
}
{
fma.rn.f16x2 r7438, r7443, r7429, r7436;
}
st.local.u32 [rd16+8], r7438;
{
mul.f16x2 r7442, r7443, r7431;
}
{
fma.rn.f16x2 r7445, r7446, r7429, r7442;
}
st.local.u32 [rd16+12], r7445;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7453, {low, high};
}
{
mul.f16x2 r7454, r7414, r7453;
}
{
mul.f16x2 r7457, r7425, r7412;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7425;
mov.b32 r7460, {high, low};
}
{
fma.rn.f16x2 r7462, r7454, r7460, r7457;
}
ld.local.u32 r7480, [rd16+16];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7462;
mov.b32 r7466, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7462;
mov.b32 r7468, {high, high};
}
ld.local.u32 r7483, [rd16+20];
{
mul.f16x2 r7470, r7483, r7468;
}
{
neg.f16x2 r7473, r7470;
}
{
fma.rn.f16x2 r7475, r7480, r7466, r7473;
}
st.local.u32 [rd16+16], r7475;
{
mul.f16x2 r7479, r7480, r7468;
}
{
fma.rn.f16x2 r7482, r7483, r7466, r7479;
}
st.local.u32 [rd16+20], r7482;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7490, {low, high};
}
{
mul.f16x2 r7491, r7414, r7490;
}
{
mul.f16x2 r7494, r7462, r7412;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7462;
mov.b32 r7497, {high, low};
}
{
fma.rn.f16x2 r7499, r7491, r7497, r7494;
}
ld.local.u32 r7517, [rd16+24];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7499;
mov.b32 r7503, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7499;
mov.b32 r7505, {high, high};
}
ld.local.u32 r7520, [rd16+28];
{
mul.f16x2 r7507, r7520, r7505;
}
{
neg.f16x2 r7510, r7507;
}
{
fma.rn.f16x2 r7512, r7517, r7503, r7510;
}
st.local.u32 [rd16+24], r7512;
{
mul.f16x2 r7516, r7517, r7505;
}
{
fma.rn.f16x2 r7519, r7520, r7503, r7516;
}
st.local.u32 [rd16+28], r7519;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7527, {low, high};
}
{
mul.f16x2 r7528, r7414, r7527;
}
{
mul.f16x2 r7531, r7499, r7412;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7499;
mov.b32 r7534, {high, low};
}
{
fma.rn.f16x2 r7536, r7528, r7534, r7531;
}
ld.local.u32 r7550, [rd16+32];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7536;
mov.b32 r7540, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7536;
mov.b32 r7542, {high, high};
}
ld.local.u32 r7557, [rd16+36];
{
mul.f16x2 r7544, r7557, r7542;
}
{
neg.f16x2 r7547, r7544;
}
{
fma.rn.f16x2 r7549, r7550, r7540, r7547;
}
st.local.u32 [rd16+32], r7549;
{
mul.f16x2 r7553, r7550, r7542;
}
{
fma.rn.f16x2 r7556, r7557, r7540, r7553;
}
st.local.u32 [rd16+36], r7556;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7564, {low, high};
}
{
mul.f16x2 r7565, r7414, r7564;
}
{
mul.f16x2 r7568, r7536, r7412;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7536;
mov.b32 r7571, {high, low};
}
{
fma.rn.f16x2 r7573, r7565, r7571, r7568;
}
ld.local.u32 r7587, [rd16+40];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7573;
mov.b32 r7577, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7573;
mov.b32 r7579, {high, high};
}
ld.local.u32 r7582, [rd16+44];
{
mul.f16x2 r7581, r7582, r7579;
}
{
neg.f16x2 r7584, r7581;
}
{
fma.rn.f16x2 r7586, r7587, r7577, r7584;
}
st.local.u32 [rd16+40], r7586;
{
mul.f16x2 r7590, r7587, r7579;
}
{
fma.rn.f16x2 r7593, r7582, r7577, r7590;
}
st.local.u32 [rd16+44], r7593;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7601, {low, high};
}
{
mul.f16x2 r7602, r7414, r7601;
}
{
mul.f16x2 r7605, r7573, r7412;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7573;
mov.b32 r7608, {high, low};
}
{
fma.rn.f16x2 r7610, r7602, r7608, r7605;
}
ld.local.u32 r7624, [rd16+48];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7610;
mov.b32 r7614, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7610;
mov.b32 r7616, {high, high};
}
ld.local.u32 r7619, [rd16+52];
{
mul.f16x2 r7618, r7619, r7616;
}
{
neg.f16x2 r7621, r7618;
}
{
fma.rn.f16x2 r7623, r7624, r7614, r7621;
}
st.local.u32 [rd16+48], r7623;
{
mul.f16x2 r7627, r7624, r7616;
}
{
fma.rn.f16x2 r7630, r7619, r7614, r7627;
}
st.local.u32 [rd16+52], r7630;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7638, {low, high};
}
{
mul.f16x2 r7639, r7414, r7638;
}
{
mul.f16x2 r7642, r7610, r7412;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7610;
mov.b32 r7645, {high, low};
}
{
fma.rn.f16x2 r7647, r7639, r7645, r7642;
}
ld.local.u32 r7661, [rd16+56];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7647;
mov.b32 r7651, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7647;
mov.b32 r7653, {high, high};
}
ld.local.u32 r7656, [rd16+60];
{
mul.f16x2 r7655, r7656, r7653;
}
{
neg.f16x2 r7658, r7655;
}
{
fma.rn.f16x2 r7660, r7661, r7651, r7658;
}
st.local.u32 [rd16+56], r7660;
{
mul.f16x2 r7664, r7661, r7653;
}
{
fma.rn.f16x2 r7667, r7656, r7651, r7664;
}
st.local.u32 [rd16+60], r7667;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7675, {low, high};
}
{
mul.f16x2 r7676, r7414, r7675;
}
{
mul.f16x2 r7679, r7647, r7412;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7647;
mov.b32 r7682, {high, low};
}
{
fma.rn.f16x2 r11253, r7676, r7682, r7679;
}
add.s32 r11252, r11252, 8;
setp.eq.s32 p2, r11252, 25;
@p2 bra LBB1_6;
bra.uni LBB1_5;
LBB1_6:
mul.lo.s32 r11193, r15, 25;
sub.s32 r11194, r3, r11193;
shl.b32 r11195, r11194, 2;
add.s32 r11196, r12, r11195;
barrier.sync 0;
mad.lo.s32 r11197, r15, 2500, r11196;
ld.local.u32 r11198, [rd3];
st.shared.u32 [r11197], r11198;
ld.local.u32 r11199, [rd4+4];
st.shared.u32 [r11197+100], r11199;
ld.local.u32 r11200, [rd4+12];
st.shared.u32 [r11197+200], r11200;
ld.local.u32 r11201, [rd4+20];
st.shared.u32 [r11197+300], r11201;
ld.local.u32 r11202, [rd4+28];
st.shared.u32 [r11197+400], r11202;
ld.local.u32 r11203, [rd4+36];
st.shared.u32 [r11197+500], r11203;
ld.local.u32 r11204, [rd4+44];
st.shared.u32 [r11197+600], r11204;
ld.local.u32 r11205, [rd4+52];
st.shared.u32 [r11197+700], r11205;
ld.local.u32 r11206, [rd4+60];
st.shared.u32 [r11197+800], r11206;
ld.local.u32 r11207, [rd4+68];
st.shared.u32 [r11197+900], r11207;
ld.local.u32 r11208, [rd4+76];
st.shared.u32 [r11197+1000], r11208;
ld.local.u32 r11209, [rd4+84];
st.shared.u32 [r11197+1100], r11209;
ld.local.u32 r11210, [rd4+92];
st.shared.u32 [r11197+1200], r11210;
ld.local.u32 r11211, [rd4+100];
st.shared.u32 [r11197+1300], r11211;
ld.local.u32 r11212, [rd4+108];
st.shared.u32 [r11197+1400], r11212;
ld.local.u32 r11213, [rd4+116];
st.shared.u32 [r11197+1500], r11213;
ld.local.u32 r11214, [rd4+124];
st.shared.u32 [r11197+1600], r11214;
ld.local.u32 r11215, [rd4+132];
st.shared.u32 [r11197+1700], r11215;
ld.local.u32 r11216, [rd4+140];
st.shared.u32 [r11197+1800], r11216;
ld.local.u32 r11217, [rd4+148];
st.shared.u32 [r11197+1900], r11217;
ld.local.u32 r11218, [rd4+156];
st.shared.u32 [r11197+2000], r11218;
ld.local.u32 r11219, [rd4+164];
st.shared.u32 [r11197+2100], r11219;
ld.local.u32 r11220, [rd4+172];
st.shared.u32 [r11197+2200], r11220;
ld.local.u32 r11221, [rd4+180];
st.shared.u32 [r11197+2300], r11221;
ld.local.u32 r11222, [rd4+188];
st.shared.u32 [r11197+2400], r11222;
barrier.sync 0;
ld.shared.u32 r7700, [r13];
ld.shared.u32 r8020, [r13+2500];
ld.shared.u32 r8340, [r13+5000];
ld.shared.u32 r8660, [r13+7500];
ld.shared.u32 r8980, [r13+10000];
ld.shared.u32 r7697, [r13+12500];
ld.shared.u32 r8017, [r13+15000];
ld.shared.u32 r8337, [r13+17500];
ld.shared.u32 r8657, [r13+20000];
ld.shared.u32 r8977, [r13+22500];
ld.shared.u32 r7703, [r13+25000];
ld.shared.u32 r8023, [r13+27500];
ld.shared.u32 r8343, [r13+30000];
ld.shared.u32 r8663, [r13+32500];
ld.shared.u32 r8983, [r13+35000];
ld.shared.u32 r7704, [r13+37500];
ld.shared.u32 r8024, [r13+40000];
ld.shared.u32 r8344, [r13+42500];
ld.shared.u32 r8664, [r13+45000];
ld.shared.u32 r8984, [r13+47500];
ld.shared.u32 r7698, [r13+50000];
ld.shared.u32 r8018, [r13+52500];
ld.shared.u32 r8338, [r13+55000];
ld.shared.u32 r8658, [r13+57500];
ld.shared.u32 r8978, [r13+60000];
barrier.sync 0;
ld.local.u32 r11223, [rd4];
st.shared.u32 [r11197], r11223;
ld.local.u32 r11224, [rd4+8];
st.shared.u32 [r11197+100], r11224;
ld.local.u32 r11225, [rd4+16];
st.shared.u32 [r11197+200], r11225;
ld.local.u32 r11226, [rd4+24];
st.shared.u32 [r11197+300], r11226;
ld.local.u32 r11227, [rd4+32];
st.shared.u32 [r11197+400], r11227;
ld.local.u32 r11228, [rd4+40];
st.shared.u32 [r11197+500], r11228;
ld.local.u32 r11229, [rd4+48];
st.shared.u32 [r11197+600], r11229;
ld.local.u32 r11230, [rd4+56];
st.shared.u32 [r11197+700], r11230;
ld.local.u32 r11231, [rd4+64];
st.shared.u32 [r11197+800], r11231;
ld.local.u32 r11232, [rd4+72];
st.shared.u32 [r11197+900], r11232;
ld.local.u32 r11233, [rd4+80];
st.shared.u32 [r11197+1000], r11233;
ld.local.u32 r11234, [rd4+88];
st.shared.u32 [r11197+1100], r11234;
ld.local.u32 r11235, [rd4+96];
st.shared.u32 [r11197+1200], r11235;
ld.local.u32 r11236, [rd4+104];
st.shared.u32 [r11197+1300], r11236;
ld.local.u32 r11237, [rd4+112];
st.shared.u32 [r11197+1400], r11237;
ld.local.u32 r11238, [rd4+120];
st.shared.u32 [r11197+1500], r11238;
ld.local.u32 r11239, [rd4+128];
st.shared.u32 [r11197+1600], r11239;
ld.local.u32 r11240, [rd4+136];
st.shared.u32 [r11197+1700], r11240;
ld.local.u32 r11241, [rd4+144];
st.shared.u32 [r11197+1800], r11241;
ld.local.u32 r11242, [rd4+152];
st.shared.u32 [r11197+1900], r11242;
ld.local.u32 r11243, [rd4+160];
st.shared.u32 [r11197+2000], r11243;
ld.local.u32 r11244, [rd4+168];
st.shared.u32 [r11197+2100], r11244;
ld.local.u32 r11245, [rd4+176];
st.shared.u32 [r11197+2200], r11245;
ld.local.u32 r11246, [rd4+184];
st.shared.u32 [r11197+2300], r11246;
ld.local.u32 r11247, [rd4+192];
st.shared.u32 [r11197+2400], r11247;
barrier.sync 0;
ld.shared.u32 r7712, [r13];
ld.shared.u32 r8032, [r13+2500];
ld.shared.u32 r8352, [r13+5000];
ld.shared.u32 r8672, [r13+7500];
ld.shared.u32 r8992, [r13+10000];
ld.shared.u32 r7709, [r13+12500];
ld.shared.u32 r8029, [r13+15000];
ld.shared.u32 r8349, [r13+17500];
ld.shared.u32 r8669, [r13+20000];
ld.shared.u32 r8989, [r13+22500];
ld.shared.u32 r7715, [r13+25000];
ld.shared.u32 r8035, [r13+27500];
ld.shared.u32 r8355, [r13+30000];
ld.shared.u32 r8675, [r13+32500];
ld.shared.u32 r8995, [r13+35000];
ld.shared.u32 r7716, [r13+37500];
ld.shared.u32 r8036, [r13+40000];
ld.shared.u32 r8356, [r13+42500];
ld.shared.u32 r8676, [r13+45000];
ld.shared.u32 r8996, [r13+47500];
ld.shared.u32 r7710, [r13+50000];
ld.shared.u32 r8030, [r13+52500];
ld.shared.u32 r8350, [r13+55000];
ld.shared.u32 r8670, [r13+57500];
ld.shared.u32 r8990, [r13+60000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r7688, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r7689, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r7690, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r7691, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r7692, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r7693, {low, high};
}
{
neg.f16x2 r7694, r7693;
}
{
add.f16x2 r7696, r7697, r7698;
}
{
add.f16x2 r7699, r7700, r7696;
}
{
add.f16x2 r7702, r7703, r7704;
}
{
add.f16x2 r7705, r7699, r7702;
}
{
add.f16x2 r7708, r7709, r7710;
}
{
add.f16x2 r7711, r7712, r7708;
}
{
add.f16x2 r7714, r7715, r7716;
}
{
add.f16x2 r7717, r7711, r7714;
}
{
add.f16x2 r7720, r7697, r7698;
}
{
mul.f16x2 r7723, r7720, r7688;
}
{
add.f16x2 r7726, r7700, r7723;
}
{
add.f16x2 r7729, r7703, r7704;
}
{
mul.f16x2 r7732, r7729, r7690;
}
{
add.f16x2 r7735, r7726, r7732;
}
{
sub.f16x2 r7738, r7709, r7710;
}
{
mul.f16x2 r7741, r7738, r7689;
}
{
sub.f16x2 r7744, r7715, r7716;
}
{
mul.f16x2 r7747, r7744, r7691;
}
{
add.f16x2 r7750, r7741, r7747;
}
{
sub.f16x2 r7753, r7735, r7750;
}
{
add.f16x2 r7756, r7697, r7698;
}
{
mul.f16x2 r7759, r7756, r7688;
}
{
add.f16x2 r7762, r7700, r7759;
}
{
add.f16x2 r7765, r7703, r7704;
}
{
mul.f16x2 r7768, r7765, r7690;
}
{
add.f16x2 r7771, r7762, r7768;
}
{
sub.f16x2 r7774, r7709, r7710;
}
{
mul.f16x2 r7777, r7774, r7689;
}
{
sub.f16x2 r7780, r7715, r7716;
}
{
mul.f16x2 r7783, r7780, r7691;
}
{
add.f16x2 r7786, r7777, r7783;
}
{
add.f16x2 r7789, r7771, r7786;
}
{
add.f16x2 r7792, r7697, r7698;
}
{
mul.f16x2 r7795, r7792, r7690;
}
{
add.f16x2 r7798, r7700, r7795;
}
{
add.f16x2 r7801, r7703, r7704;
}
{
mul.f16x2 r7804, r7801, r7692;
}
{
add.f16x2 r7807, r7798, r7804;
}
{
sub.f16x2 r7810, r7709, r7710;
}
{
mul.f16x2 r7813, r7810, r7691;
}
{
sub.f16x2 r7816, r7715, r7716;
}
{
mul.f16x2 r7819, r7816, r7694;
}
{
add.f16x2 r7822, r7813, r7819;
}
{
sub.f16x2 r7825, r7807, r7822;
}
{
add.f16x2 r7828, r7697, r7698;
}
{
mul.f16x2 r7831, r7828, r7690;
}
{
add.f16x2 r7834, r7700, r7831;
}
{
add.f16x2 r7837, r7703, r7704;
}
{
mul.f16x2 r7840, r7837, r7692;
}
{
add.f16x2 r7843, r7834, r7840;
}
{
sub.f16x2 r7846, r7709, r7710;
}
{
mul.f16x2 r7849, r7846, r7691;
}
{
sub.f16x2 r7852, r7715, r7716;
}
{
mul.f16x2 r7855, r7852, r7694;
}
{
add.f16x2 r7858, r7849, r7855;
}
{
add.f16x2 r7861, r7843, r7858;
}
{
add.f16x2 r7864, r7709, r7710;
}
{
mul.f16x2 r7867, r7864, r7688;
}
{
add.f16x2 r7870, r7712, r7867;
}
{
add.f16x2 r7873, r7715, r7716;
}
{
mul.f16x2 r7876, r7873, r7690;
}
{
add.f16x2 r7879, r7870, r7876;
}
{
sub.f16x2 r7882, r7697, r7698;
}
{
mul.f16x2 r7885, r7882, r7689;
}
{
sub.f16x2 r7888, r7703, r7704;
}
{
mul.f16x2 r7891, r7888, r7691;
}
{
add.f16x2 r7894, r7885, r7891;
}
{
add.f16x2 r7897, r7879, r7894;
}
{
add.f16x2 r7900, r7709, r7710;
}
{
mul.f16x2 r7903, r7900, r7688;
}
{
add.f16x2 r7906, r7712, r7903;
}
{
add.f16x2 r7909, r7715, r7716;
}
{
mul.f16x2 r7912, r7909, r7690;
}
{
add.f16x2 r7915, r7906, r7912;
}
{
sub.f16x2 r7918, r7697, r7698;
}
{
mul.f16x2 r7921, r7918, r7689;
}
{
sub.f16x2 r7924, r7703, r7704;
}
{
mul.f16x2 r7927, r7924, r7691;
}
{
add.f16x2 r7930, r7921, r7927;
}
{
sub.f16x2 r7933, r7915, r7930;
}
{
add.f16x2 r7936, r7709, r7710;
}
{
mul.f16x2 r7939, r7936, r7690;
}
{
add.f16x2 r7942, r7712, r7939;
}
{
add.f16x2 r7945, r7715, r7716;
}
{
mul.f16x2 r7948, r7945, r7692;
}
{
add.f16x2 r7951, r7942, r7948;
}
{
sub.f16x2 r7954, r7697, r7698;
}
{
mul.f16x2 r7957, r7954, r7691;
}
{
sub.f16x2 r7960, r7703, r7704;
}
{
mul.f16x2 r7963, r7960, r7694;
}
{
add.f16x2 r7966, r7957, r7963;
}
{
add.f16x2 r7969, r7951, r7966;
}
{
add.f16x2 r7972, r7709, r7710;
}
{
mul.f16x2 r7975, r7972, r7690;
}
{
add.f16x2 r7978, r7712, r7975;
}
{
add.f16x2 r7981, r7715, r7716;
}
{
mul.f16x2 r7984, r7981, r7692;
}
{
add.f16x2 r7987, r7978, r7984;
}
{
sub.f16x2 r7990, r7697, r7698;
}
{
mul.f16x2 r7993, r7990, r7691;
}
{
sub.f16x2 r7996, r7703, r7704;
}
{
mul.f16x2 r7999, r7996, r7694;
}
{
add.f16x2 r8002, r7993, r7999;
}
{
sub.f16x2 r8005, r7987, r8002;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8008, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8009, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r8010, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r8011, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8012, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8013, {low, high};
}
{
neg.f16x2 r8014, r8013;
}
{
add.f16x2 r8016, r8017, r8018;
}
{
add.f16x2 r8019, r8020, r8016;
}
{
add.f16x2 r8022, r8023, r8024;
}
{
add.f16x2 r8025, r8019, r8022;
}
{
add.f16x2 r8028, r8029, r8030;
}
{
add.f16x2 r8031, r8032, r8028;
}
{
add.f16x2 r8034, r8035, r8036;
}
{
add.f16x2 r8037, r8031, r8034;
}
{
add.f16x2 r8040, r8017, r8018;
}
{
mul.f16x2 r8043, r8040, r8008;
}
{
add.f16x2 r8046, r8020, r8043;
}
{
add.f16x2 r8049, r8023, r8024;
}
{
mul.f16x2 r8052, r8049, r8010;
}
{
add.f16x2 r8055, r8046, r8052;
}
{
sub.f16x2 r8058, r8029, r8030;
}
{
mul.f16x2 r8061, r8058, r8009;
}
{
sub.f16x2 r8064, r8035, r8036;
}
{
mul.f16x2 r8067, r8064, r8011;
}
{
add.f16x2 r8070, r8061, r8067;
}
{
sub.f16x2 r8073, r8055, r8070;
}
{
add.f16x2 r8076, r8017, r8018;
}
{
mul.f16x2 r8079, r8076, r8008;
}
{
add.f16x2 r8082, r8020, r8079;
}
{
add.f16x2 r8085, r8023, r8024;
}
{
mul.f16x2 r8088, r8085, r8010;
}
{
add.f16x2 r8091, r8082, r8088;
}
{
sub.f16x2 r8094, r8029, r8030;
}
{
mul.f16x2 r8097, r8094, r8009;
}
{
sub.f16x2 r8100, r8035, r8036;
}
{
mul.f16x2 r8103, r8100, r8011;
}
{
add.f16x2 r8106, r8097, r8103;
}
{
add.f16x2 r8109, r8091, r8106;
}
{
add.f16x2 r8112, r8017, r8018;
}
{
mul.f16x2 r8115, r8112, r8010;
}
{
add.f16x2 r8118, r8020, r8115;
}
{
add.f16x2 r8121, r8023, r8024;
}
{
mul.f16x2 r8124, r8121, r8012;
}
{
add.f16x2 r8127, r8118, r8124;
}
{
sub.f16x2 r8130, r8029, r8030;
}
{
mul.f16x2 r8133, r8130, r8011;
}
{
sub.f16x2 r8136, r8035, r8036;
}
{
mul.f16x2 r8139, r8136, r8014;
}
{
add.f16x2 r8142, r8133, r8139;
}
{
sub.f16x2 r8145, r8127, r8142;
}
{
add.f16x2 r8148, r8017, r8018;
}
{
mul.f16x2 r8151, r8148, r8010;
}
{
add.f16x2 r8154, r8020, r8151;
}
{
add.f16x2 r8157, r8023, r8024;
}
{
mul.f16x2 r8160, r8157, r8012;
}
{
add.f16x2 r8163, r8154, r8160;
}
{
sub.f16x2 r8166, r8029, r8030;
}
{
mul.f16x2 r8169, r8166, r8011;
}
{
sub.f16x2 r8172, r8035, r8036;
}
{
mul.f16x2 r8175, r8172, r8014;
}
{
add.f16x2 r8178, r8169, r8175;
}
{
add.f16x2 r8181, r8163, r8178;
}
{
add.f16x2 r8184, r8029, r8030;
}
{
mul.f16x2 r8187, r8184, r8008;
}
{
add.f16x2 r8190, r8032, r8187;
}
{
add.f16x2 r8193, r8035, r8036;
}
{
mul.f16x2 r8196, r8193, r8010;
}
{
add.f16x2 r8199, r8190, r8196;
}
{
sub.f16x2 r8202, r8017, r8018;
}
{
mul.f16x2 r8205, r8202, r8009;
}
{
sub.f16x2 r8208, r8023, r8024;
}
{
mul.f16x2 r8211, r8208, r8011;
}
{
add.f16x2 r8214, r8205, r8211;
}
{
add.f16x2 r8217, r8199, r8214;
}
{
add.f16x2 r8220, r8029, r8030;
}
{
mul.f16x2 r8223, r8220, r8008;
}
{
add.f16x2 r8226, r8032, r8223;
}
{
add.f16x2 r8229, r8035, r8036;
}
{
mul.f16x2 r8232, r8229, r8010;
}
{
add.f16x2 r8235, r8226, r8232;
}
{
sub.f16x2 r8238, r8017, r8018;
}
{
mul.f16x2 r8241, r8238, r8009;
}
{
sub.f16x2 r8244, r8023, r8024;
}
{
mul.f16x2 r8247, r8244, r8011;
}
{
add.f16x2 r8250, r8241, r8247;
}
{
sub.f16x2 r8253, r8235, r8250;
}
{
add.f16x2 r8256, r8029, r8030;
}
{
mul.f16x2 r8259, r8256, r8010;
}
{
add.f16x2 r8262, r8032, r8259;
}
{
add.f16x2 r8265, r8035, r8036;
}
{
mul.f16x2 r8268, r8265, r8012;
}
{
add.f16x2 r8271, r8262, r8268;
}
{
sub.f16x2 r8274, r8017, r8018;
}
{
mul.f16x2 r8277, r8274, r8011;
}
{
sub.f16x2 r8280, r8023, r8024;
}
{
mul.f16x2 r8283, r8280, r8014;
}
{
add.f16x2 r8286, r8277, r8283;
}
{
add.f16x2 r8289, r8271, r8286;
}
{
add.f16x2 r8292, r8029, r8030;
}
{
mul.f16x2 r8295, r8292, r8010;
}
{
add.f16x2 r8298, r8032, r8295;
}
{
add.f16x2 r8301, r8035, r8036;
}
{
mul.f16x2 r8304, r8301, r8012;
}
{
add.f16x2 r8307, r8298, r8304;
}
{
sub.f16x2 r8310, r8017, r8018;
}
{
mul.f16x2 r8313, r8310, r8011;
}
{
sub.f16x2 r8316, r8023, r8024;
}
{
mul.f16x2 r8319, r8316, r8014;
}
{
add.f16x2 r8322, r8313, r8319;
}
{
sub.f16x2 r8325, r8307, r8322;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8328, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8329, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r8330, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r8331, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8332, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8333, {low, high};
}
{
neg.f16x2 r8334, r8333;
}
{
add.f16x2 r8336, r8337, r8338;
}
{
add.f16x2 r8339, r8340, r8336;
}
{
add.f16x2 r8342, r8343, r8344;
}
{
add.f16x2 r8345, r8339, r8342;
}
{
add.f16x2 r8348, r8349, r8350;
}
{
add.f16x2 r8351, r8352, r8348;
}
{
add.f16x2 r8354, r8355, r8356;
}
{
add.f16x2 r8357, r8351, r8354;
}
{
add.f16x2 r8360, r8337, r8338;
}
{
mul.f16x2 r8363, r8360, r8328;
}
{
add.f16x2 r8366, r8340, r8363;
}
{
add.f16x2 r8369, r8343, r8344;
}
{
mul.f16x2 r8372, r8369, r8330;
}
{
add.f16x2 r8375, r8366, r8372;
}
{
sub.f16x2 r8378, r8349, r8350;
}
{
mul.f16x2 r8381, r8378, r8329;
}
{
sub.f16x2 r8384, r8355, r8356;
}
{
mul.f16x2 r8387, r8384, r8331;
}
{
add.f16x2 r8390, r8381, r8387;
}
{
sub.f16x2 r8393, r8375, r8390;
}
{
add.f16x2 r8396, r8337, r8338;
}
{
mul.f16x2 r8399, r8396, r8328;
}
{
add.f16x2 r8402, r8340, r8399;
}
{
add.f16x2 r8405, r8343, r8344;
}
{
mul.f16x2 r8408, r8405, r8330;
}
{
add.f16x2 r8411, r8402, r8408;
}
{
sub.f16x2 r8414, r8349, r8350;
}
{
mul.f16x2 r8417, r8414, r8329;
}
{
sub.f16x2 r8420, r8355, r8356;
}
{
mul.f16x2 r8423, r8420, r8331;
}
{
add.f16x2 r8426, r8417, r8423;
}
{
add.f16x2 r8429, r8411, r8426;
}
{
add.f16x2 r8432, r8337, r8338;
}
{
mul.f16x2 r8435, r8432, r8330;
}
{
add.f16x2 r8438, r8340, r8435;
}
{
add.f16x2 r8441, r8343, r8344;
}
{
mul.f16x2 r8444, r8441, r8332;
}
{
add.f16x2 r8447, r8438, r8444;
}
{
sub.f16x2 r8450, r8349, r8350;
}
{
mul.f16x2 r8453, r8450, r8331;
}
{
sub.f16x2 r8456, r8355, r8356;
}
{
mul.f16x2 r8459, r8456, r8334;
}
{
add.f16x2 r8462, r8453, r8459;
}
{
sub.f16x2 r8465, r8447, r8462;
}
{
add.f16x2 r8468, r8337, r8338;
}
{
mul.f16x2 r8471, r8468, r8330;
}
{
add.f16x2 r8474, r8340, r8471;
}
{
add.f16x2 r8477, r8343, r8344;
}
{
mul.f16x2 r8480, r8477, r8332;
}
{
add.f16x2 r8483, r8474, r8480;
}
{
sub.f16x2 r8486, r8349, r8350;
}
{
mul.f16x2 r8489, r8486, r8331;
}
{
sub.f16x2 r8492, r8355, r8356;
}
{
mul.f16x2 r8495, r8492, r8334;
}
{
add.f16x2 r8498, r8489, r8495;
}
{
add.f16x2 r8501, r8483, r8498;
}
{
add.f16x2 r8504, r8349, r8350;
}
{
mul.f16x2 r8507, r8504, r8328;
}
{
add.f16x2 r8510, r8352, r8507;
}
{
add.f16x2 r8513, r8355, r8356;
}
{
mul.f16x2 r8516, r8513, r8330;
}
{
add.f16x2 r8519, r8510, r8516;
}
{
sub.f16x2 r8522, r8337, r8338;
}
{
mul.f16x2 r8525, r8522, r8329;
}
{
sub.f16x2 r8528, r8343, r8344;
}
{
mul.f16x2 r8531, r8528, r8331;
}
{
add.f16x2 r8534, r8525, r8531;
}
{
add.f16x2 r8537, r8519, r8534;
}
{
add.f16x2 r8540, r8349, r8350;
}
{
mul.f16x2 r8543, r8540, r8328;
}
{
add.f16x2 r8546, r8352, r8543;
}
{
add.f16x2 r8549, r8355, r8356;
}
{
mul.f16x2 r8552, r8549, r8330;
}
{
add.f16x2 r8555, r8546, r8552;
}
{
sub.f16x2 r8558, r8337, r8338;
}
{
mul.f16x2 r8561, r8558, r8329;
}
{
sub.f16x2 r8564, r8343, r8344;
}
{
mul.f16x2 r8567, r8564, r8331;
}
{
add.f16x2 r8570, r8561, r8567;
}
{
sub.f16x2 r8573, r8555, r8570;
}
{
add.f16x2 r8576, r8349, r8350;
}
{
mul.f16x2 r8579, r8576, r8330;
}
{
add.f16x2 r8582, r8352, r8579;
}
{
add.f16x2 r8585, r8355, r8356;
}
{
mul.f16x2 r8588, r8585, r8332;
}
{
add.f16x2 r8591, r8582, r8588;
}
{
sub.f16x2 r8594, r8337, r8338;
}
{
mul.f16x2 r8597, r8594, r8331;
}
{
sub.f16x2 r8600, r8343, r8344;
}
{
mul.f16x2 r8603, r8600, r8334;
}
{
add.f16x2 r8606, r8597, r8603;
}
{
add.f16x2 r8609, r8591, r8606;
}
{
add.f16x2 r8612, r8349, r8350;
}
{
mul.f16x2 r8615, r8612, r8330;
}
{
add.f16x2 r8618, r8352, r8615;
}
{
add.f16x2 r8621, r8355, r8356;
}
{
mul.f16x2 r8624, r8621, r8332;
}
{
add.f16x2 r8627, r8618, r8624;
}
{
sub.f16x2 r8630, r8337, r8338;
}
{
mul.f16x2 r8633, r8630, r8331;
}
{
sub.f16x2 r8636, r8343, r8344;
}
{
mul.f16x2 r8639, r8636, r8334;
}
{
add.f16x2 r8642, r8633, r8639;
}
{
sub.f16x2 r8645, r8627, r8642;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8648, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8649, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r8650, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r8651, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8652, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8653, {low, high};
}
{
neg.f16x2 r8654, r8653;
}
{
add.f16x2 r8656, r8657, r8658;
}
{
add.f16x2 r8659, r8660, r8656;
}
{
add.f16x2 r8662, r8663, r8664;
}
{
add.f16x2 r8665, r8659, r8662;
}
{
add.f16x2 r8668, r8669, r8670;
}
{
add.f16x2 r8671, r8672, r8668;
}
{
add.f16x2 r8674, r8675, r8676;
}
{
add.f16x2 r8677, r8671, r8674;
}
{
add.f16x2 r8680, r8657, r8658;
}
{
mul.f16x2 r8683, r8680, r8648;
}
{
add.f16x2 r8686, r8660, r8683;
}
{
add.f16x2 r8689, r8663, r8664;
}
{
mul.f16x2 r8692, r8689, r8650;
}
{
add.f16x2 r8695, r8686, r8692;
}
{
sub.f16x2 r8698, r8669, r8670;
}
{
mul.f16x2 r8701, r8698, r8649;
}
{
sub.f16x2 r8704, r8675, r8676;
}
{
mul.f16x2 r8707, r8704, r8651;
}
{
add.f16x2 r8710, r8701, r8707;
}
{
sub.f16x2 r8713, r8695, r8710;
}
{
add.f16x2 r8716, r8657, r8658;
}
{
mul.f16x2 r8719, r8716, r8648;
}
{
add.f16x2 r8722, r8660, r8719;
}
{
add.f16x2 r8725, r8663, r8664;
}
{
mul.f16x2 r8728, r8725, r8650;
}
{
add.f16x2 r8731, r8722, r8728;
}
{
sub.f16x2 r8734, r8669, r8670;
}
{
mul.f16x2 r8737, r8734, r8649;
}
{
sub.f16x2 r8740, r8675, r8676;
}
{
mul.f16x2 r8743, r8740, r8651;
}
{
add.f16x2 r8746, r8737, r8743;
}
{
add.f16x2 r8749, r8731, r8746;
}
{
add.f16x2 r8752, r8657, r8658;
}
{
mul.f16x2 r8755, r8752, r8650;
}
{
add.f16x2 r8758, r8660, r8755;
}
{
add.f16x2 r8761, r8663, r8664;
}
{
mul.f16x2 r8764, r8761, r8652;
}
{
add.f16x2 r8767, r8758, r8764;
}
{
sub.f16x2 r8770, r8669, r8670;
}
{
mul.f16x2 r8773, r8770, r8651;
}
{
sub.f16x2 r8776, r8675, r8676;
}
{
mul.f16x2 r8779, r8776, r8654;
}
{
add.f16x2 r8782, r8773, r8779;
}
{
sub.f16x2 r8785, r8767, r8782;
}
{
add.f16x2 r8788, r8657, r8658;
}
{
mul.f16x2 r8791, r8788, r8650;
}
{
add.f16x2 r8794, r8660, r8791;
}
{
add.f16x2 r8797, r8663, r8664;
}
{
mul.f16x2 r8800, r8797, r8652;
}
{
add.f16x2 r8803, r8794, r8800;
}
{
sub.f16x2 r8806, r8669, r8670;
}
{
mul.f16x2 r8809, r8806, r8651;
}
{
sub.f16x2 r8812, r8675, r8676;
}
{
mul.f16x2 r8815, r8812, r8654;
}
{
add.f16x2 r8818, r8809, r8815;
}
{
add.f16x2 r8821, r8803, r8818;
}
{
add.f16x2 r8824, r8669, r8670;
}
{
mul.f16x2 r8827, r8824, r8648;
}
{
add.f16x2 r8830, r8672, r8827;
}
{
add.f16x2 r8833, r8675, r8676;
}
{
mul.f16x2 r8836, r8833, r8650;
}
{
add.f16x2 r8839, r8830, r8836;
}
{
sub.f16x2 r8842, r8657, r8658;
}
{
mul.f16x2 r8845, r8842, r8649;
}
{
sub.f16x2 r8848, r8663, r8664;
}
{
mul.f16x2 r8851, r8848, r8651;
}
{
add.f16x2 r8854, r8845, r8851;
}
{
add.f16x2 r8857, r8839, r8854;
}
{
add.f16x2 r8860, r8669, r8670;
}
{
mul.f16x2 r8863, r8860, r8648;
}
{
add.f16x2 r8866, r8672, r8863;
}
{
add.f16x2 r8869, r8675, r8676;
}
{
mul.f16x2 r8872, r8869, r8650;
}
{
add.f16x2 r8875, r8866, r8872;
}
{
sub.f16x2 r8878, r8657, r8658;
}
{
mul.f16x2 r8881, r8878, r8649;
}
{
sub.f16x2 r8884, r8663, r8664;
}
{
mul.f16x2 r8887, r8884, r8651;
}
{
add.f16x2 r8890, r8881, r8887;
}
{
sub.f16x2 r8893, r8875, r8890;
}
{
add.f16x2 r8896, r8669, r8670;
}
{
mul.f16x2 r8899, r8896, r8650;
}
{
add.f16x2 r8902, r8672, r8899;
}
{
add.f16x2 r8905, r8675, r8676;
}
{
mul.f16x2 r8908, r8905, r8652;
}
{
add.f16x2 r8911, r8902, r8908;
}
{
sub.f16x2 r8914, r8657, r8658;
}
{
mul.f16x2 r8917, r8914, r8651;
}
{
sub.f16x2 r8920, r8663, r8664;
}
{
mul.f16x2 r8923, r8920, r8654;
}
{
add.f16x2 r8926, r8917, r8923;
}
{
add.f16x2 r8929, r8911, r8926;
}
{
add.f16x2 r8932, r8669, r8670;
}
{
mul.f16x2 r8935, r8932, r8650;
}
{
add.f16x2 r8938, r8672, r8935;
}
{
add.f16x2 r8941, r8675, r8676;
}
{
mul.f16x2 r8944, r8941, r8652;
}
{
add.f16x2 r8947, r8938, r8944;
}
{
sub.f16x2 r8950, r8657, r8658;
}
{
mul.f16x2 r8953, r8950, r8651;
}
{
sub.f16x2 r8956, r8663, r8664;
}
{
mul.f16x2 r8959, r8956, r8654;
}
{
add.f16x2 r8962, r8953, r8959;
}
{
sub.f16x2 r8965, r8947, r8962;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8968, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8969, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r8970, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r8971, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8972, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8973, {low, high};
}
{
neg.f16x2 r8974, r8973;
}
{
add.f16x2 r8976, r8977, r8978;
}
{
add.f16x2 r8979, r8980, r8976;
}
{
add.f16x2 r8982, r8983, r8984;
}
{
add.f16x2 r8985, r8979, r8982;
}
{
add.f16x2 r8988, r8989, r8990;
}
{
add.f16x2 r8991, r8992, r8988;
}
{
add.f16x2 r8994, r8995, r8996;
}
{
add.f16x2 r8997, r8991, r8994;
}
{
add.f16x2 r9000, r8977, r8978;
}
{
mul.f16x2 r9003, r9000, r8968;
}
{
add.f16x2 r9006, r8980, r9003;
}
{
add.f16x2 r9009, r8983, r8984;
}
{
mul.f16x2 r9012, r9009, r8970;
}
{
add.f16x2 r9015, r9006, r9012;
}
{
sub.f16x2 r9018, r8989, r8990;
}
{
mul.f16x2 r9021, r9018, r8969;
}
{
sub.f16x2 r9024, r8995, r8996;
}
{
mul.f16x2 r9027, r9024, r8971;
}
{
add.f16x2 r9030, r9021, r9027;
}
{
sub.f16x2 r9033, r9015, r9030;
}
{
add.f16x2 r9036, r8977, r8978;
}
{
mul.f16x2 r9039, r9036, r8968;
}
{
add.f16x2 r9042, r8980, r9039;
}
{
add.f16x2 r9045, r8983, r8984;
}
{
mul.f16x2 r9048, r9045, r8970;
}
{
add.f16x2 r9051, r9042, r9048;
}
{
sub.f16x2 r9054, r8989, r8990;
}
{
mul.f16x2 r9057, r9054, r8969;
}
{
sub.f16x2 r9060, r8995, r8996;
}
{
mul.f16x2 r9063, r9060, r8971;
}
{
add.f16x2 r9066, r9057, r9063;
}
{
add.f16x2 r9069, r9051, r9066;
}
{
add.f16x2 r9072, r8977, r8978;
}
{
mul.f16x2 r9075, r9072, r8970;
}
{
add.f16x2 r9078, r8980, r9075;
}
{
add.f16x2 r9081, r8983, r8984;
}
{
mul.f16x2 r9084, r9081, r8972;
}
{
add.f16x2 r9087, r9078, r9084;
}
{
sub.f16x2 r9090, r8989, r8990;
}
{
mul.f16x2 r9093, r9090, r8971;
}
{
sub.f16x2 r9096, r8995, r8996;
}
{
mul.f16x2 r9099, r9096, r8974;
}
{
add.f16x2 r9102, r9093, r9099;
}
{
sub.f16x2 r9105, r9087, r9102;
}
{
add.f16x2 r9108, r8977, r8978;
}
{
mul.f16x2 r9111, r9108, r8970;
}
{
add.f16x2 r9114, r8980, r9111;
}
{
add.f16x2 r9117, r8983, r8984;
}
{
mul.f16x2 r9120, r9117, r8972;
}
{
add.f16x2 r9123, r9114, r9120;
}
{
sub.f16x2 r9126, r8989, r8990;
}
{
mul.f16x2 r9129, r9126, r8971;
}
{
sub.f16x2 r9132, r8995, r8996;
}
{
mul.f16x2 r9135, r9132, r8974;
}
{
add.f16x2 r9138, r9129, r9135;
}
{
add.f16x2 r9141, r9123, r9138;
}
{
add.f16x2 r9144, r8989, r8990;
}
{
mul.f16x2 r9147, r9144, r8968;
}
{
add.f16x2 r9150, r8992, r9147;
}
{
add.f16x2 r9153, r8995, r8996;
}
{
mul.f16x2 r9156, r9153, r8970;
}
{
add.f16x2 r9159, r9150, r9156;
}
{
sub.f16x2 r9162, r8977, r8978;
}
{
mul.f16x2 r9165, r9162, r8969;
}
{
sub.f16x2 r9168, r8983, r8984;
}
{
mul.f16x2 r9171, r9168, r8971;
}
{
add.f16x2 r9174, r9165, r9171;
}
{
add.f16x2 r9177, r9159, r9174;
}
{
add.f16x2 r9180, r8989, r8990;
}
{
mul.f16x2 r9183, r9180, r8968;
}
{
add.f16x2 r9186, r8992, r9183;
}
{
add.f16x2 r9189, r8995, r8996;
}
{
mul.f16x2 r9192, r9189, r8970;
}
{
add.f16x2 r9195, r9186, r9192;
}
{
sub.f16x2 r9198, r8977, r8978;
}
{
mul.f16x2 r9201, r9198, r8969;
}
{
sub.f16x2 r9204, r8983, r8984;
}
{
mul.f16x2 r9207, r9204, r8971;
}
{
add.f16x2 r9210, r9201, r9207;
}
{
sub.f16x2 r9213, r9195, r9210;
}
{
add.f16x2 r9216, r8989, r8990;
}
{
mul.f16x2 r9219, r9216, r8970;
}
{
add.f16x2 r9222, r8992, r9219;
}
{
add.f16x2 r9225, r8995, r8996;
}
{
mul.f16x2 r9228, r9225, r8972;
}
{
add.f16x2 r9231, r9222, r9228;
}
{
sub.f16x2 r9234, r8977, r8978;
}
{
mul.f16x2 r9237, r9234, r8971;
}
{
sub.f16x2 r9240, r8983, r8984;
}
{
mul.f16x2 r9243, r9240, r8974;
}
{
add.f16x2 r9246, r9237, r9243;
}
{
add.f16x2 r9249, r9231, r9246;
}
{
add.f16x2 r9252, r8989, r8990;
}
{
mul.f16x2 r9255, r9252, r8970;
}
{
add.f16x2 r9258, r8992, r9255;
}
{
add.f16x2 r9261, r8995, r8996;
}
{
mul.f16x2 r9264, r9261, r8972;
}
{
add.f16x2 r9267, r9258, r9264;
}
{
sub.f16x2 r9270, r8977, r8978;
}
{
mul.f16x2 r9273, r9270, r8971;
}
{
sub.f16x2 r9276, r8983, r8984;
}
{
mul.f16x2 r9279, r9276, r8974;
}
{
add.f16x2 r9282, r9273, r9279;
}
{
sub.f16x2 r9285, r9267, r9282;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f62;
cvt.rn.f16.f32 high, f62;
mov.b32 r9288, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f64;
cvt.rn.f16.f32 high, f64;
mov.b32 r9289, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f66;
cvt.rn.f16.f32 high, f66;
mov.b32 r9290, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f68;
cvt.rn.f16.f32 high, f68;
mov.b32 r9291, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f70;
cvt.rn.f16.f32 high, f70;
mov.b32 r9292, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f72;
cvt.rn.f16.f32 high, f72;
mov.b32 r9293, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r9294, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r9295, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r9298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r9299, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r9302, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r9303, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r9304, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r9305, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r9310, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r9311, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r9318, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r9319, {low, high};
}
{
mul.f16x2 r9336, r8073, r9288;
}
{
mul.f16x2 r9339, r8217, r9289;
}
{
sub.f16x2 r9342, r9336, r9339;
}
{
mul.f16x2 r9345, r8073, r9289;
}
{
fma.rn.f16x2 r9348, r8217, r9288, r9345;
}
{
mul.f16x2 r9352, r8393, r9290;
}
{
mul.f16x2 r9355, r8537, r9291;
}
{
sub.f16x2 r9358, r9352, r9355;
}
{
mul.f16x2 r9361, r8393, r9291;
}
{
fma.rn.f16x2 r9364, r8537, r9290, r9361;
}
{
mul.f16x2 r9368, r8713, r9292;
}
{
mul.f16x2 r9371, r8857, r9293;
}
{
sub.f16x2 r9374, r9368, r9371;
}
{
mul.f16x2 r9377, r8713, r9293;
}
{
fma.rn.f16x2 r9380, r8857, r9292, r9377;
}
{
mul.f16x2 r9384, r9033, r9294;
}
{
mul.f16x2 r9387, r9177, r9295;
}
{
sub.f16x2 r9390, r9384, r9387;
}
{
mul.f16x2 r9393, r9033, r9295;
}
{
fma.rn.f16x2 r9396, r9177, r9294, r9393;
}
{
mul.f16x2 r9400, r8145, r9290;
}
{
mul.f16x2 r9403, r8289, r9291;
}
{
sub.f16x2 r9406, r9400, r9403;
}
{
mul.f16x2 r9409, r8145, r9291;
}
{
fma.rn.f16x2 r9412, r8289, r9290, r9409;
}
{
mul.f16x2 r9416, r8465, r9294;
}
{
mul.f16x2 r9419, r8609, r9295;
}
{
sub.f16x2 r9422, r9416, r9419;
}
{
mul.f16x2 r9425, r8465, r9295;
}
{
fma.rn.f16x2 r9428, r8609, r9294, r9425;
}
{
mul.f16x2 r9432, r8785, r9298;
}
{
mul.f16x2 r9435, r8929, r9299;
}
{
sub.f16x2 r9438, r9432, r9435;
}
{
mul.f16x2 r9441, r8785, r9299;
}
{
fma.rn.f16x2 r9444, r8929, r9298, r9441;
}
{
mul.f16x2 r9448, r9105, r9302;
}
{
mul.f16x2 r9451, r9249, r9303;
}
{
sub.f16x2 r9454, r9448, r9451;
}
{
mul.f16x2 r9457, r9105, r9303;
}
{
fma.rn.f16x2 r9460, r9249, r9302, r9457;
}
{
mul.f16x2 r9464, r8181, r9292;
}
{
mul.f16x2 r9467, r8325, r9293;
}
{
sub.f16x2 r9470, r9464, r9467;
}
{
mul.f16x2 r9473, r8181, r9293;
}
{
fma.rn.f16x2 r9476, r8325, r9292, r9473;
}
{
mul.f16x2 r9480, r8501, r9298;
}
{
mul.f16x2 r9483, r8645, r9299;
}
{
sub.f16x2 r9486, r9480, r9483;
}
{
mul.f16x2 r9489, r8501, r9299;
}
{
fma.rn.f16x2 r9492, r8645, r9298, r9489;
}
{
mul.f16x2 r9496, r8821, r9304;
}
{
mul.f16x2 r9499, r8965, r9305;
}
{
sub.f16x2 r9502, r9496, r9499;
}
{
mul.f16x2 r9505, r8821, r9305;
}
{
fma.rn.f16x2 r9508, r8965, r9304, r9505;
}
{
mul.f16x2 r9512, r9141, r9310;
}
{
mul.f16x2 r9515, r9285, r9311;
}
{
sub.f16x2 r9518, r9512, r9515;
}
{
mul.f16x2 r9521, r9141, r9311;
}
{
fma.rn.f16x2 r9524, r9285, r9310, r9521;
}
{
mul.f16x2 r9528, r8109, r9294;
}
{
mul.f16x2 r9531, r8253, r9295;
}
{
sub.f16x2 r9534, r9528, r9531;
}
{
mul.f16x2 r9537, r8109, r9295;
}
{
fma.rn.f16x2 r9540, r8253, r9294, r9537;
}
{
mul.f16x2 r9544, r8429, r9302;
}
{
mul.f16x2 r9547, r8573, r9303;
}
{
sub.f16x2 r9550, r9544, r9547;
}
{
mul.f16x2 r9553, r8429, r9303;
}
{
fma.rn.f16x2 r9556, r8573, r9302, r9553;
}
{
mul.f16x2 r9560, r8749, r9310;
}
{
mul.f16x2 r9563, r8893, r9311;
}
{
sub.f16x2 r9566, r9560, r9563;
}
{
mul.f16x2 r9569, r8749, r9311;
}
{
fma.rn.f16x2 r9572, r8893, r9310, r9569;
}
{
mul.f16x2 r9576, r9069, r9318;
}
{
mul.f16x2 r9579, r9213, r9319;
}
{
sub.f16x2 r9582, r9576, r9579;
}
{
mul.f16x2 r9585, r9069, r9319;
}
{
fma.rn.f16x2 r9588, r9213, r9318, r9585;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r9592, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r9593, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r9594, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r9595, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r9596, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r9597, {low, high};
}
{
neg.f16x2 r9598, r9597;
}
{
add.f16x2 r9600, r8025, r8985;
}
{
add.f16x2 r9603, r7705, r9600;
}
{
add.f16x2 r9606, r8345, r8665;
}
{
add.f16x2 %0, r9603, r9606;
}
{
add.f16x2 r9612, r8037, r8997;
}
{
add.f16x2 r9615, r7717, r9612;
}
{
add.f16x2 r9618, r8357, r8677;
}
{
add.f16x2 %1, r9615, r9618;
}
{
add.f16x2 r9624, r8025, r8985;
}
{
mul.f16x2 r9627, r9624, r9592;
}
{
add.f16x2 r9630, r7705, r9627;
}
{
add.f16x2 r9633, r8345, r8665;
}
{
mul.f16x2 r9636, r9633, r9594;
}
{
add.f16x2 r9639, r9630, r9636;
}
{
sub.f16x2 r9642, r8037, r8997;
}
{
mul.f16x2 r9645, r9642, r9593;
}
{
sub.f16x2 r9648, r8357, r8677;
}
{
mul.f16x2 r9651, r9648, r9595;
}
{
add.f16x2 r9654, r9645, r9651;
}
{
sub.f16x2 %10, r9639, r9654;
}
{
add.f16x2 r9660, r8025, r8985;
}
{
mul.f16x2 r9663, r9660, r9592;
}
{
add.f16x2 r9666, r7705, r9663;
}
{
add.f16x2 r9669, r8345, r8665;
}
{
mul.f16x2 r9672, r9669, r9594;
}
{
add.f16x2 r9675, r9666, r9672;
}
{
sub.f16x2 r9678, r8037, r8997;
}
{
mul.f16x2 r9681, r9678, r9593;
}
{
sub.f16x2 r9684, r8357, r8677;
}
{
mul.f16x2 r9687, r9684, r9595;
}
{
add.f16x2 r9690, r9681, r9687;
}
{
add.f16x2 %40, r9675, r9690;
}
{
add.f16x2 r9696, r8025, r8985;
}
{
mul.f16x2 r9699, r9696, r9594;
}
{
add.f16x2 r9702, r7705, r9699;
}
{
add.f16x2 r9705, r8345, r8665;
}
{
mul.f16x2 r9708, r9705, r9596;
}
{
add.f16x2 r9711, r9702, r9708;
}
{
sub.f16x2 r9714, r8037, r8997;
}
{
mul.f16x2 r9717, r9714, r9595;
}
{
sub.f16x2 r9720, r8357, r8677;
}
{
mul.f16x2 r9723, r9720, r9598;
}
{
add.f16x2 r9726, r9717, r9723;
}
{
sub.f16x2 %20, r9711, r9726;
}
{
add.f16x2 r9732, r8025, r8985;
}
{
mul.f16x2 r9735, r9732, r9594;
}
{
add.f16x2 r9738, r7705, r9735;
}
{
add.f16x2 r9741, r8345, r8665;
}
{
mul.f16x2 r9744, r9741, r9596;
}
{
add.f16x2 r9747, r9738, r9744;
}
{
sub.f16x2 r9750, r8037, r8997;
}
{
mul.f16x2 r9753, r9750, r9595;
}
{
sub.f16x2 r9756, r8357, r8677;
}
{
mul.f16x2 r9759, r9756, r9598;
}
{
add.f16x2 r9762, r9753, r9759;
}
{
add.f16x2 %30, r9747, r9762;
}
{
add.f16x2 r9768, r8037, r8997;
}
{
mul.f16x2 r9771, r9768, r9592;
}
{
add.f16x2 r9774, r7717, r9771;
}
{
add.f16x2 r9777, r8357, r8677;
}
{
mul.f16x2 r9780, r9777, r9594;
}
{
add.f16x2 r9783, r9774, r9780;
}
{
sub.f16x2 r9786, r8025, r8985;
}
{
mul.f16x2 r9789, r9786, r9593;
}
{
sub.f16x2 r9792, r8345, r8665;
}
{
mul.f16x2 r9795, r9792, r9595;
}
{
add.f16x2 r9798, r9789, r9795;
}
{
add.f16x2 %11, r9783, r9798;
}
{
add.f16x2 r9804, r8037, r8997;
}
{
mul.f16x2 r9807, r9804, r9592;
}
{
add.f16x2 r9810, r7717, r9807;
}
{
add.f16x2 r9813, r8357, r8677;
}
{
mul.f16x2 r9816, r9813, r9594;
}
{
add.f16x2 r9819, r9810, r9816;
}
{
sub.f16x2 r9822, r8025, r8985;
}
{
mul.f16x2 r9825, r9822, r9593;
}
{
sub.f16x2 r9828, r8345, r8665;
}
{
mul.f16x2 r9831, r9828, r9595;
}
{
add.f16x2 r9834, r9825, r9831;
}
{
sub.f16x2 %41, r9819, r9834;
}
{
add.f16x2 r9840, r8037, r8997;
}
{
mul.f16x2 r9843, r9840, r9594;
}
{
add.f16x2 r9846, r7717, r9843;
}
{
add.f16x2 r9849, r8357, r8677;
}
{
mul.f16x2 r9852, r9849, r9596;
}
{
add.f16x2 r9855, r9846, r9852;
}
{
sub.f16x2 r9858, r8025, r8985;
}
{
mul.f16x2 r9861, r9858, r9595;
}
{
sub.f16x2 r9864, r8345, r8665;
}
{
mul.f16x2 r9867, r9864, r9598;
}
{
add.f16x2 r9870, r9861, r9867;
}
{
add.f16x2 %21, r9855, r9870;
}
{
add.f16x2 r9876, r8037, r8997;
}
{
mul.f16x2 r9879, r9876, r9594;
}
{
add.f16x2 r9882, r7717, r9879;
}
{
add.f16x2 r9885, r8357, r8677;
}
{
mul.f16x2 r9888, r9885, r9596;
}
{
add.f16x2 r9891, r9882, r9888;
}
{
sub.f16x2 r9894, r8025, r8985;
}
{
mul.f16x2 r9897, r9894, r9595;
}
{
sub.f16x2 r9900, r8345, r8665;
}
{
mul.f16x2 r9903, r9900, r9598;
}
{
add.f16x2 r9906, r9897, r9903;
}
{
sub.f16x2 %31, r9891, r9906;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r9912, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r9913, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r9914, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r9915, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r9916, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r9917, {low, high};
}
{
neg.f16x2 r9918, r9917;
}
{
add.f16x2 r9920, r9342, r9390;
}
{
add.f16x2 r9923, r7753, r9920;
}
{
add.f16x2 r9926, r9358, r9374;
}
{
add.f16x2 %2, r9923, r9926;
}
{
add.f16x2 r9932, r9348, r9396;
}
{
add.f16x2 r9935, r7897, r9932;
}
{
add.f16x2 r9938, r9364, r9380;
}
{
add.f16x2 %3, r9935, r9938;
}
{
add.f16x2 r9944, r9342, r9390;
}
{
mul.f16x2 r9947, r9944, r9912;
}
{
add.f16x2 r9950, r7753, r9947;
}
{
add.f16x2 r9953, r9358, r9374;
}
{
mul.f16x2 r9956, r9953, r9914;
}
{
add.f16x2 r9959, r9950, r9956;
}
{
sub.f16x2 r9962, r9348, r9396;
}
{
mul.f16x2 r9965, r9962, r9913;
}
{
sub.f16x2 r9968, r9364, r9380;
}
{
mul.f16x2 r9971, r9968, r9915;
}
{
add.f16x2 r9974, r9965, r9971;
}
{
sub.f16x2 %12, r9959, r9974;
}
{
add.f16x2 r9980, r9342, r9390;
}
{
mul.f16x2 r9983, r9980, r9912;
}
{
add.f16x2 r9986, r7753, r9983;
}
{
add.f16x2 r9989, r9358, r9374;
}
{
mul.f16x2 r9992, r9989, r9914;
}
{
add.f16x2 r9995, r9986, r9992;
}
{
sub.f16x2 r9998, r9348, r9396;
}
{
mul.f16x2 r10001, r9998, r9913;
}
{
sub.f16x2 r10004, r9364, r9380;
}
{
mul.f16x2 r10007, r10004, r9915;
}
{
add.f16x2 r10010, r10001, r10007;
}
{
add.f16x2 %42, r9995, r10010;
}
{
add.f16x2 r10016, r9342, r9390;
}
{
mul.f16x2 r10019, r10016, r9914;
}
{
add.f16x2 r10022, r7753, r10019;
}
{
add.f16x2 r10025, r9358, r9374;
}
{
mul.f16x2 r10028, r10025, r9916;
}
{
add.f16x2 r10031, r10022, r10028;
}
{
sub.f16x2 r10034, r9348, r9396;
}
{
mul.f16x2 r10037, r10034, r9915;
}
{
sub.f16x2 r10040, r9364, r9380;
}
{
mul.f16x2 r10043, r10040, r9918;
}
{
add.f16x2 r10046, r10037, r10043;
}
{
sub.f16x2 %22, r10031, r10046;
}
{
add.f16x2 r10052, r9342, r9390;
}
{
mul.f16x2 r10055, r10052, r9914;
}
{
add.f16x2 r10058, r7753, r10055;
}
{
add.f16x2 r10061, r9358, r9374;
}
{
mul.f16x2 r10064, r10061, r9916;
}
{
add.f16x2 r10067, r10058, r10064;
}
{
sub.f16x2 r10070, r9348, r9396;
}
{
mul.f16x2 r10073, r10070, r9915;
}
{
sub.f16x2 r10076, r9364, r9380;
}
{
mul.f16x2 r10079, r10076, r9918;
}
{
add.f16x2 r10082, r10073, r10079;
}
{
add.f16x2 %32, r10067, r10082;
}
{
add.f16x2 r10088, r9348, r9396;
}
{
mul.f16x2 r10091, r10088, r9912;
}
{
add.f16x2 r10094, r7897, r10091;
}
{
add.f16x2 r10097, r9364, r9380;
}
{
mul.f16x2 r10100, r10097, r9914;
}
{
add.f16x2 r10103, r10094, r10100;
}
{
sub.f16x2 r10106, r9342, r9390;
}
{
mul.f16x2 r10109, r10106, r9913;
}
{
sub.f16x2 r10112, r9358, r9374;
}
{
mul.f16x2 r10115, r10112, r9915;
}
{
add.f16x2 r10118, r10109, r10115;
}
{
add.f16x2 %13, r10103, r10118;
}
{
add.f16x2 r10124, r9348, r9396;
}
{
mul.f16x2 r10127, r10124, r9912;
}
{
add.f16x2 r10130, r7897, r10127;
}
{
add.f16x2 r10133, r9364, r9380;
}
{
mul.f16x2 r10136, r10133, r9914;
}
{
add.f16x2 r10139, r10130, r10136;
}
{
sub.f16x2 r10142, r9342, r9390;
}
{
mul.f16x2 r10145, r10142, r9913;
}
{
sub.f16x2 r10148, r9358, r9374;
}
{
mul.f16x2 r10151, r10148, r9915;
}
{
add.f16x2 r10154, r10145, r10151;
}
{
sub.f16x2 %43, r10139, r10154;
}
{
add.f16x2 r10160, r9348, r9396;
}
{
mul.f16x2 r10163, r10160, r9914;
}
{
add.f16x2 r10166, r7897, r10163;
}
{
add.f16x2 r10169, r9364, r9380;
}
{
mul.f16x2 r10172, r10169, r9916;
}
{
add.f16x2 r10175, r10166, r10172;
}
{
sub.f16x2 r10178, r9342, r9390;
}
{
mul.f16x2 r10181, r10178, r9915;
}
{
sub.f16x2 r10184, r9358, r9374;
}
{
mul.f16x2 r10187, r10184, r9918;
}
{
add.f16x2 r10190, r10181, r10187;
}
{
add.f16x2 %23, r10175, r10190;
}
{
add.f16x2 r10196, r9348, r9396;
}
{
mul.f16x2 r10199, r10196, r9914;
}
{
add.f16x2 r10202, r7897, r10199;
}
{
add.f16x2 r10205, r9364, r9380;
}
{
mul.f16x2 r10208, r10205, r9916;
}
{
add.f16x2 r10211, r10202, r10208;
}
{
sub.f16x2 r10214, r9342, r9390;
}
{
mul.f16x2 r10217, r10214, r9915;
}
{
sub.f16x2 r10220, r9358, r9374;
}
{
mul.f16x2 r10223, r10220, r9918;
}
{
add.f16x2 r10226, r10217, r10223;
}
{
sub.f16x2 %33, r10211, r10226;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10232, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10233, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r10234, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r10235, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10236, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10237, {low, high};
}
{
neg.f16x2 r10238, r10237;
}
{
add.f16x2 r10240, r9406, r9454;
}
{
add.f16x2 r10243, r7825, r10240;
}
{
add.f16x2 r10246, r9422, r9438;
}
{
add.f16x2 %4, r10243, r10246;
}
{
add.f16x2 r10252, r9412, r9460;
}
{
add.f16x2 r10255, r7969, r10252;
}
{
add.f16x2 r10258, r9428, r9444;
}
{
add.f16x2 %5, r10255, r10258;
}
{
add.f16x2 r10264, r9406, r9454;
}
{
mul.f16x2 r10267, r10264, r10232;
}
{
add.f16x2 r10270, r7825, r10267;
}
{
add.f16x2 r10273, r9422, r9438;
}
{
mul.f16x2 r10276, r10273, r10234;
}
{
add.f16x2 r10279, r10270, r10276;
}
{
sub.f16x2 r10282, r9412, r9460;
}
{
mul.f16x2 r10285, r10282, r10233;
}
{
sub.f16x2 r10288, r9428, r9444;
}
{
mul.f16x2 r10291, r10288, r10235;
}
{
add.f16x2 r10294, r10285, r10291;
}
{
sub.f16x2 %14, r10279, r10294;
}
{
add.f16x2 r10300, r9406, r9454;
}
{
mul.f16x2 r10303, r10300, r10232;
}
{
add.f16x2 r10306, r7825, r10303;
}
{
add.f16x2 r10309, r9422, r9438;
}
{
mul.f16x2 r10312, r10309, r10234;
}
{
add.f16x2 r10315, r10306, r10312;
}
{
sub.f16x2 r10318, r9412, r9460;
}
{
mul.f16x2 r10321, r10318, r10233;
}
{
sub.f16x2 r10324, r9428, r9444;
}
{
mul.f16x2 r10327, r10324, r10235;
}
{
add.f16x2 r10330, r10321, r10327;
}
{
add.f16x2 %44, r10315, r10330;
}
{
add.f16x2 r10336, r9406, r9454;
}
{
mul.f16x2 r10339, r10336, r10234;
}
{
add.f16x2 r10342, r7825, r10339;
}
{
add.f16x2 r10345, r9422, r9438;
}
{
mul.f16x2 r10348, r10345, r10236;
}
{
add.f16x2 r10351, r10342, r10348;
}
{
sub.f16x2 r10354, r9412, r9460;
}
{
mul.f16x2 r10357, r10354, r10235;
}
{
sub.f16x2 r10360, r9428, r9444;
}
{
mul.f16x2 r10363, r10360, r10238;
}
{
add.f16x2 r10366, r10357, r10363;
}
{
sub.f16x2 %24, r10351, r10366;
}
{
add.f16x2 r10372, r9406, r9454;
}
{
mul.f16x2 r10375, r10372, r10234;
}
{
add.f16x2 r10378, r7825, r10375;
}
{
add.f16x2 r10381, r9422, r9438;
}
{
mul.f16x2 r10384, r10381, r10236;
}
{
add.f16x2 r10387, r10378, r10384;
}
{
sub.f16x2 r10390, r9412, r9460;
}
{
mul.f16x2 r10393, r10390, r10235;
}
{
sub.f16x2 r10396, r9428, r9444;
}
{
mul.f16x2 r10399, r10396, r10238;
}
{
add.f16x2 r10402, r10393, r10399;
}
{
add.f16x2 %34, r10387, r10402;
}
{
add.f16x2 r10408, r9412, r9460;
}
{
mul.f16x2 r10411, r10408, r10232;
}
{
add.f16x2 r10414, r7969, r10411;
}
{
add.f16x2 r10417, r9428, r9444;
}
{
mul.f16x2 r10420, r10417, r10234;
}
{
add.f16x2 r10423, r10414, r10420;
}
{
sub.f16x2 r10426, r9406, r9454;
}
{
mul.f16x2 r10429, r10426, r10233;
}
{
sub.f16x2 r10432, r9422, r9438;
}
{
mul.f16x2 r10435, r10432, r10235;
}
{
add.f16x2 r10438, r10429, r10435;
}
{
add.f16x2 %15, r10423, r10438;
}
{
add.f16x2 r10444, r9412, r9460;
}
{
mul.f16x2 r10447, r10444, r10232;
}
{
add.f16x2 r10450, r7969, r10447;
}
{
add.f16x2 r10453, r9428, r9444;
}
{
mul.f16x2 r10456, r10453, r10234;
}
{
add.f16x2 r10459, r10450, r10456;
}
{
sub.f16x2 r10462, r9406, r9454;
}
{
mul.f16x2 r10465, r10462, r10233;
}
{
sub.f16x2 r10468, r9422, r9438;
}
{
mul.f16x2 r10471, r10468, r10235;
}
{
add.f16x2 r10474, r10465, r10471;
}
{
sub.f16x2 %45, r10459, r10474;
}
{
add.f16x2 r10480, r9412, r9460;
}
{
mul.f16x2 r10483, r10480, r10234;
}
{
add.f16x2 r10486, r7969, r10483;
}
{
add.f16x2 r10489, r9428, r9444;
}
{
mul.f16x2 r10492, r10489, r10236;
}
{
add.f16x2 r10495, r10486, r10492;
}
{
sub.f16x2 r10498, r9406, r9454;
}
{
mul.f16x2 r10501, r10498, r10235;
}
{
sub.f16x2 r10504, r9422, r9438;
}
{
mul.f16x2 r10507, r10504, r10238;
}
{
add.f16x2 r10510, r10501, r10507;
}
{
add.f16x2 %25, r10495, r10510;
}
{
add.f16x2 r10516, r9412, r9460;
}
{
mul.f16x2 r10519, r10516, r10234;
}
{
add.f16x2 r10522, r7969, r10519;
}
{
add.f16x2 r10525, r9428, r9444;
}
{
mul.f16x2 r10528, r10525, r10236;
}
{
add.f16x2 r10531, r10522, r10528;
}
{
sub.f16x2 r10534, r9406, r9454;
}
{
mul.f16x2 r10537, r10534, r10235;
}
{
sub.f16x2 r10540, r9422, r9438;
}
{
mul.f16x2 r10543, r10540, r10238;
}
{
add.f16x2 r10546, r10537, r10543;
}
{
sub.f16x2 %35, r10531, r10546;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10552, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10553, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r10554, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r10555, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10556, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10557, {low, high};
}
{
neg.f16x2 r10558, r10557;
}
{
add.f16x2 r10560, r9470, r9518;
}
{
add.f16x2 r10563, r7861, r10560;
}
{
add.f16x2 r10566, r9486, r9502;
}
{
add.f16x2 %6, r10563, r10566;
}
{
add.f16x2 r10572, r9476, r9524;
}
{
add.f16x2 r10575, r8005, r10572;
}
{
add.f16x2 r10578, r9492, r9508;
}
{
add.f16x2 %7, r10575, r10578;
}
{
add.f16x2 r10584, r9470, r9518;
}
{
mul.f16x2 r10587, r10584, r10552;
}
{
add.f16x2 r10590, r7861, r10587;
}
{
add.f16x2 r10593, r9486, r9502;
}
{
mul.f16x2 r10596, r10593, r10554;
}
{
add.f16x2 r10599, r10590, r10596;
}
{
sub.f16x2 r10602, r9476, r9524;
}
{
mul.f16x2 r10605, r10602, r10553;
}
{
sub.f16x2 r10608, r9492, r9508;
}
{
mul.f16x2 r10611, r10608, r10555;
}
{
add.f16x2 r10614, r10605, r10611;
}
{
sub.f16x2 %16, r10599, r10614;
}
{
add.f16x2 r10620, r9470, r9518;
}
{
mul.f16x2 r10623, r10620, r10552;
}
{
add.f16x2 r10626, r7861, r10623;
}
{
add.f16x2 r10629, r9486, r9502;
}
{
mul.f16x2 r10632, r10629, r10554;
}
{
add.f16x2 r10635, r10626, r10632;
}
{
sub.f16x2 r10638, r9476, r9524;
}
{
mul.f16x2 r10641, r10638, r10553;
}
{
sub.f16x2 r10644, r9492, r9508;
}
{
mul.f16x2 r10647, r10644, r10555;
}
{
add.f16x2 r10650, r10641, r10647;
}
{
add.f16x2 %46, r10635, r10650;
}
{
add.f16x2 r10656, r9470, r9518;
}
{
mul.f16x2 r10659, r10656, r10554;
}
{
add.f16x2 r10662, r7861, r10659;
}
{
add.f16x2 r10665, r9486, r9502;
}
{
mul.f16x2 r10668, r10665, r10556;
}
{
add.f16x2 r10671, r10662, r10668;
}
{
sub.f16x2 r10674, r9476, r9524;
}
{
mul.f16x2 r10677, r10674, r10555;
}
{
sub.f16x2 r10680, r9492, r9508;
}
{
mul.f16x2 r10683, r10680, r10558;
}
{
add.f16x2 r10686, r10677, r10683;
}
{
sub.f16x2 %26, r10671, r10686;
}
{
add.f16x2 r10692, r9470, r9518;
}
{
mul.f16x2 r10695, r10692, r10554;
}
{
add.f16x2 r10698, r7861, r10695;
}
{
add.f16x2 r10701, r9486, r9502;
}
{
mul.f16x2 r10704, r10701, r10556;
}
{
add.f16x2 r10707, r10698, r10704;
}
{
sub.f16x2 r10710, r9476, r9524;
}
{
mul.f16x2 r10713, r10710, r10555;
}
{
sub.f16x2 r10716, r9492, r9508;
}
{
mul.f16x2 r10719, r10716, r10558;
}
{
add.f16x2 r10722, r10713, r10719;
}
{
add.f16x2 %36, r10707, r10722;
}
{
add.f16x2 r10728, r9476, r9524;
}
{
mul.f16x2 r10731, r10728, r10552;
}
{
add.f16x2 r10734, r8005, r10731;
}
{
add.f16x2 r10737, r9492, r9508;
}
{
mul.f16x2 r10740, r10737, r10554;
}
{
add.f16x2 r10743, r10734, r10740;
}
{
sub.f16x2 r10746, r9470, r9518;
}
{
mul.f16x2 r10749, r10746, r10553;
}
{
sub.f16x2 r10752, r9486, r9502;
}
{
mul.f16x2 r10755, r10752, r10555;
}
{
add.f16x2 r10758, r10749, r10755;
}
{
add.f16x2 %17, r10743, r10758;
}
{
add.f16x2 r10764, r9476, r9524;
}
{
mul.f16x2 r10767, r10764, r10552;
}
{
add.f16x2 r10770, r8005, r10767;
}
{
add.f16x2 r10773, r9492, r9508;
}
{
mul.f16x2 r10776, r10773, r10554;
}
{
add.f16x2 r10779, r10770, r10776;
}
{
sub.f16x2 r10782, r9470, r9518;
}
{
mul.f16x2 r10785, r10782, r10553;
}
{
sub.f16x2 r10788, r9486, r9502;
}
{
mul.f16x2 r10791, r10788, r10555;
}
{
add.f16x2 r10794, r10785, r10791;
}
{
sub.f16x2 %47, r10779, r10794;
}
{
add.f16x2 r10800, r9476, r9524;
}
{
mul.f16x2 r10803, r10800, r10554;
}
{
add.f16x2 r10806, r8005, r10803;
}
{
add.f16x2 r10809, r9492, r9508;
}
{
mul.f16x2 r10812, r10809, r10556;
}
{
add.f16x2 r10815, r10806, r10812;
}
{
sub.f16x2 r10818, r9470, r9518;
}
{
mul.f16x2 r10821, r10818, r10555;
}
{
sub.f16x2 r10824, r9486, r9502;
}
{
mul.f16x2 r10827, r10824, r10558;
}
{
add.f16x2 r10830, r10821, r10827;
}
{
add.f16x2 %27, r10815, r10830;
}
{
add.f16x2 r10836, r9476, r9524;
}
{
mul.f16x2 r10839, r10836, r10554;
}
{
add.f16x2 r10842, r8005, r10839;
}
{
add.f16x2 r10845, r9492, r9508;
}
{
mul.f16x2 r10848, r10845, r10556;
}
{
add.f16x2 r10851, r10842, r10848;
}
{
sub.f16x2 r10854, r9470, r9518;
}
{
mul.f16x2 r10857, r10854, r10555;
}
{
sub.f16x2 r10860, r9486, r9502;
}
{
mul.f16x2 r10863, r10860, r10558;
}
{
add.f16x2 r10866, r10857, r10863;
}
{
sub.f16x2 %37, r10851, r10866;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10872, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10873, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r10874, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r10875, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10876, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10877, {low, high};
}
{
neg.f16x2 r10878, r10877;
}
{
add.f16x2 r10880, r9534, r9582;
}
{
add.f16x2 r10883, r7789, r10880;
}
{
add.f16x2 r10886, r9550, r9566;
}
{
add.f16x2 %8, r10883, r10886;
}
{
add.f16x2 r10892, r9540, r9588;
}
{
add.f16x2 r10895, r7933, r10892;
}
{
add.f16x2 r10898, r9556, r9572;
}
{
add.f16x2 %9, r10895, r10898;
}
{
add.f16x2 r10904, r9534, r9582;
}
{
mul.f16x2 r10907, r10904, r10872;
}
{
add.f16x2 r10910, r7789, r10907;
}
{
add.f16x2 r10913, r9550, r9566;
}
{
mul.f16x2 r10916, r10913, r10874;
}
{
add.f16x2 r10919, r10910, r10916;
}
{
sub.f16x2 r10922, r9540, r9588;
}
{
mul.f16x2 r10925, r10922, r10873;
}
{
sub.f16x2 r10928, r9556, r9572;
}
{
mul.f16x2 r10931, r10928, r10875;
}
{
add.f16x2 r10934, r10925, r10931;
}
{
sub.f16x2 %18, r10919, r10934;
}
{
add.f16x2 r10940, r9534, r9582;
}
{
mul.f16x2 r10943, r10940, r10872;
}
{
add.f16x2 r10946, r7789, r10943;
}
{
add.f16x2 r10949, r9550, r9566;
}
{
mul.f16x2 r10952, r10949, r10874;
}
{
add.f16x2 r10955, r10946, r10952;
}
{
sub.f16x2 r10958, r9540, r9588;
}
{
mul.f16x2 r10961, r10958, r10873;
}
{
sub.f16x2 r10964, r9556, r9572;
}
{
mul.f16x2 r10967, r10964, r10875;
}
{
add.f16x2 r10970, r10961, r10967;
}
{
add.f16x2 %48, r10955, r10970;
}
{
add.f16x2 r10976, r9534, r9582;
}
{
mul.f16x2 r10979, r10976, r10874;
}
{
add.f16x2 r10982, r7789, r10979;
}
{
add.f16x2 r10985, r9550, r9566;
}
{
mul.f16x2 r10988, r10985, r10876;
}
{
add.f16x2 r10991, r10982, r10988;
}
{
sub.f16x2 r10994, r9540, r9588;
}
{
mul.f16x2 r10997, r10994, r10875;
}
{
sub.f16x2 r11000, r9556, r9572;
}
{
mul.f16x2 r11003, r11000, r10878;
}
{
add.f16x2 r11006, r10997, r11003;
}
{
sub.f16x2 %28, r10991, r11006;
}
{
add.f16x2 r11012, r9534, r9582;
}
{
mul.f16x2 r11015, r11012, r10874;
}
{
add.f16x2 r11018, r7789, r11015;
}
{
add.f16x2 r11021, r9550, r9566;
}
{
mul.f16x2 r11024, r11021, r10876;
}
{
add.f16x2 r11027, r11018, r11024;
}
{
sub.f16x2 r11030, r9540, r9588;
}
{
mul.f16x2 r11033, r11030, r10875;
}
{
sub.f16x2 r11036, r9556, r9572;
}
{
mul.f16x2 r11039, r11036, r10878;
}
{
add.f16x2 r11042, r11033, r11039;
}
{
add.f16x2 %38, r11027, r11042;
}
{
add.f16x2 r11048, r9540, r9588;
}
{
mul.f16x2 r11051, r11048, r10872;
}
{
add.f16x2 r11054, r7933, r11051;
}
{
add.f16x2 r11057, r9556, r9572;
}
{
mul.f16x2 r11060, r11057, r10874;
}
{
add.f16x2 r11063, r11054, r11060;
}
{
sub.f16x2 r11066, r9534, r9582;
}
{
mul.f16x2 r11069, r11066, r10873;
}
{
sub.f16x2 r11072, r9550, r9566;
}
{
mul.f16x2 r11075, r11072, r10875;
}
{
add.f16x2 r11078, r11069, r11075;
}
{
add.f16x2 %19, r11063, r11078;
}
{
add.f16x2 r11084, r9540, r9588;
}
{
mul.f16x2 r11087, r11084, r10872;
}
{
add.f16x2 r11090, r7933, r11087;
}
{
add.f16x2 r11093, r9556, r9572;
}
{
mul.f16x2 r11096, r11093, r10874;
}
{
add.f16x2 r11099, r11090, r11096;
}
{
sub.f16x2 r11102, r9534, r9582;
}
{
mul.f16x2 r11105, r11102, r10873;
}
{
sub.f16x2 r11108, r9550, r9566;
}
{
mul.f16x2 r11111, r11108, r10875;
}
{
add.f16x2 r11114, r11105, r11111;
}
{
sub.f16x2 %49, r11099, r11114;
}
{
add.f16x2 r11120, r9540, r9588;
}
{
mul.f16x2 r11123, r11120, r10874;
}
{
add.f16x2 r11126, r7933, r11123;
}
{
add.f16x2 r11129, r9556, r9572;
}
{
mul.f16x2 r11132, r11129, r10876;
}
{
add.f16x2 r11135, r11126, r11132;
}
{
sub.f16x2 r11138, r9534, r9582;
}
{
mul.f16x2 r11141, r11138, r10875;
}
{
sub.f16x2 r11144, r9550, r9566;
}
{
mul.f16x2 r11147, r11144, r10878;
}
{
add.f16x2 r11150, r11141, r11147;
}
{
add.f16x2 %29, r11135, r11150;
}
{
add.f16x2 r11156, r9540, r9588;
}
{
mul.f16x2 r11159, r11156, r10874;
}
{
add.f16x2 r11162, r7933, r11159;
}
{
add.f16x2 r11165, r9556, r9572;
}
{
mul.f16x2 r11168, r11165, r10876;
}
{
add.f16x2 r11171, r11162, r11168;
}
{
sub.f16x2 r11174, r9534, r9582;
}
{
mul.f16x2 r11177, r11174, r10875;
}
{
sub.f16x2 r11180, r9550, r9566;
}
{
mul.f16x2 r11183, r11180, r10878;
}
{
add.f16x2 r11186, r11177, r11183;
}
{
sub.f16x2 %39, r11171, r11186;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[24].y)));
};


#endif
