#ifndef CUFFTDX_FFT_20_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_20_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<752, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<107>;
.reg .b32 r<1747>;
.reg .f64 fd<99>;
.reg .b64 rd<2>;
mov.f64 fd67, 0d3FD3C6EF372FE950;
{
cvt.rn.f16.f64 rs1, fd67;
}
mov.b32 r210, {rs1, rs1};
mov.f64 fd77, 0dBFEE6F0E134454FF;
{
cvt.rn.f16.f64 rs2, fd77;
}
mov.b32 r228, {rs2, rs2};
mov.f64 fd75, 0dBFE9E3779B97F4A8;
{
cvt.rn.f16.f64 rs3, fd75;
}
mov.b32 r282, {rs3, rs3};
mov.f64 fd76, 0dBFE2CF2304755A5E;
{
cvt.rn.f16.f64 rs4, fd76;
}
mov.b32 r300, {rs4, rs4};
{
cvt.rn.f16.f64 rs5, fd67;
}
mov.b32 r291, {rs5, rs5};
{
cvt.rn.f16.f64 rs6, fd77;
}
{
neg.f16 rs7, rs6;
}
mov.b32 r306, {rs7, rs7};
{
add.f16x2 r1, %48, %72;
}
{
add.f16x2 r4, %40, r1;
}
{
add.f16x2 r7, %56, %64;
}
{
add.f16x2 r10, r4, r7;
}
{
add.f16x2 r13, %49, %73;
}
{
add.f16x2 r16, %41, r13;
}
{
add.f16x2 r19, %57, %65;
}
{
add.f16x2 r22, r16, r19;
}
{
add.f16x2 r25, %48, %72;
}
{
mul.f16x2 r28, r25, r210;
}
{
add.f16x2 r31, %40, r28;
}
{
add.f16x2 r34, %56, %64;
}
{
mul.f16x2 r37, r34, r282;
}
{
add.f16x2 r40, r31, r37;
}
{
sub.f16x2 r43, %49, %73;
}
{
mul.f16x2 r46, r43, r228;
}
{
sub.f16x2 r49, %57, %65;
}
{
mul.f16x2 r52, r49, r300;
}
{
add.f16x2 r55, r46, r52;
}
{
sub.f16x2 r58, r40, r55;
}
{
add.f16x2 r61, %48, %72;
}
{
mul.f16x2 r64, r61, r210;
}
{
add.f16x2 r67, %40, r64;
}
{
add.f16x2 r70, %56, %64;
}
{
mul.f16x2 r73, r70, r282;
}
{
add.f16x2 r76, r67, r73;
}
{
sub.f16x2 r79, %49, %73;
}
{
mul.f16x2 r82, r79, r228;
}
{
sub.f16x2 r85, %57, %65;
}
{
mul.f16x2 r88, r85, r300;
}
{
add.f16x2 r91, r82, r88;
}
{
add.f16x2 r94, r76, r91;
}
{
add.f16x2 r97, %48, %72;
}
{
mul.f16x2 r100, r97, r282;
}
{
add.f16x2 r103, %40, r100;
}
{
add.f16x2 r106, %56, %64;
}
{
mul.f16x2 r109, r106, r291;
}
{
add.f16x2 r112, r103, r109;
}
{
sub.f16x2 r115, %49, %73;
}
{
mul.f16x2 r118, r115, r300;
}
{
sub.f16x2 r121, %57, %65;
}
{
mul.f16x2 r124, r121, r306;
}
{
add.f16x2 r127, r118, r124;
}
{
sub.f16x2 r130, r112, r127;
}
{
add.f16x2 r133, %48, %72;
}
{
mul.f16x2 r136, r133, r282;
}
{
add.f16x2 r139, %40, r136;
}
{
add.f16x2 r142, %56, %64;
}
{
mul.f16x2 r145, r142, r291;
}
{
add.f16x2 r148, r139, r145;
}
{
sub.f16x2 r151, %49, %73;
}
{
mul.f16x2 r154, r151, r300;
}
{
sub.f16x2 r157, %57, %65;
}
{
mul.f16x2 r160, r157, r306;
}
{
add.f16x2 r163, r154, r160;
}
{
add.f16x2 r166, r148, r163;
}
{
add.f16x2 r169, %49, %73;
}
{
mul.f16x2 r172, r169, r210;
}
{
add.f16x2 r175, %41, r172;
}
{
add.f16x2 r178, %57, %65;
}
{
mul.f16x2 r181, r178, r282;
}
{
add.f16x2 r184, r175, r181;
}
{
sub.f16x2 r187, %48, %72;
}
{
mul.f16x2 r190, r187, r228;
}
{
sub.f16x2 r193, %56, %64;
}
{
mul.f16x2 r196, r193, r300;
}
{
add.f16x2 r199, r190, r196;
}
{
add.f16x2 r202, r184, r199;
}
{
add.f16x2 r205, %49, %73;
}
{
mul.f16x2 r208, r205, r210;
}
{
add.f16x2 r211, %41, r208;
}
{
add.f16x2 r214, %57, %65;
}
{
mul.f16x2 r217, r214, r282;
}
{
add.f16x2 r220, r211, r217;
}
{
sub.f16x2 r223, %48, %72;
}
{
mul.f16x2 r226, r223, r228;
}
{
sub.f16x2 r229, %56, %64;
}
{
mul.f16x2 r232, r229, r300;
}
{
add.f16x2 r235, r226, r232;
}
{
sub.f16x2 r238, r220, r235;
}
{
add.f16x2 r241, %49, %73;
}
{
mul.f16x2 r244, r241, r282;
}
{
add.f16x2 r247, %41, r244;
}
{
add.f16x2 r250, %57, %65;
}
{
mul.f16x2 r253, r250, r291;
}
{
add.f16x2 r256, r247, r253;
}
{
sub.f16x2 r259, %48, %72;
}
{
mul.f16x2 r262, r259, r300;
}
{
sub.f16x2 r265, %56, %64;
}
{
mul.f16x2 r268, r265, r306;
}
{
add.f16x2 r271, r262, r268;
}
{
add.f16x2 r274, r256, r271;
}
{
add.f16x2 r277, %49, %73;
}
{
mul.f16x2 r280, r277, r282;
}
{
add.f16x2 r283, %41, r280;
}
{
add.f16x2 r286, %57, %65;
}
{
mul.f16x2 r289, r286, r291;
}
{
add.f16x2 r292, r283, r289;
}
{
sub.f16x2 r295, %48, %72;
}
{
mul.f16x2 r298, r295, r300;
}
{
sub.f16x2 r301, %56, %64;
}
{
mul.f16x2 r304, r301, r306;
}
{
add.f16x2 r307, r298, r304;
}
{
sub.f16x2 r310, r292, r307;
}
{
cvt.rn.f16.f64 rs9, fd67;
}
mov.b32 r522, {rs9, rs9};
{
cvt.rn.f16.f64 rs10, fd77;
}
mov.b32 r540, {rs10, rs10};
{
cvt.rn.f16.f64 rs11, fd75;
}
mov.b32 r594, {rs11, rs11};
{
cvt.rn.f16.f64 rs12, fd76;
}
mov.b32 r612, {rs12, rs12};
{
cvt.rn.f16.f64 rs13, fd67;
}
mov.b32 r603, {rs13, rs13};
{
cvt.rn.f16.f64 rs14, fd77;
}
{
neg.f16 rs15, rs14;
}
mov.b32 r618, {rs15, rs15};
{
add.f16x2 r313, %52, %76;
}
{
add.f16x2 r316, %44, r313;
}
{
add.f16x2 r319, %60, %68;
}
{
add.f16x2 r322, r316, r319;
}
{
add.f16x2 r325, %53, %77;
}
{
add.f16x2 r328, %45, r325;
}
{
add.f16x2 r331, %61, %69;
}
{
add.f16x2 r334, r328, r331;
}
{
add.f16x2 r337, %52, %76;
}
{
mul.f16x2 r340, r337, r522;
}
{
add.f16x2 r343, %44, r340;
}
{
add.f16x2 r346, %60, %68;
}
{
mul.f16x2 r349, r346, r594;
}
{
add.f16x2 r352, r343, r349;
}
{
sub.f16x2 r355, %53, %77;
}
{
mul.f16x2 r358, r355, r540;
}
{
sub.f16x2 r361, %61, %69;
}
{
mul.f16x2 r364, r361, r612;
}
{
add.f16x2 r367, r358, r364;
}
{
sub.f16x2 r370, r352, r367;
}
{
add.f16x2 r373, %52, %76;
}
{
mul.f16x2 r376, r373, r522;
}
{
add.f16x2 r379, %44, r376;
}
{
add.f16x2 r382, %60, %68;
}
{
mul.f16x2 r385, r382, r594;
}
{
add.f16x2 r388, r379, r385;
}
{
sub.f16x2 r391, %53, %77;
}
{
mul.f16x2 r394, r391, r540;
}
{
sub.f16x2 r397, %61, %69;
}
{
mul.f16x2 r400, r397, r612;
}
{
add.f16x2 r403, r394, r400;
}
{
add.f16x2 r406, r388, r403;
}
{
add.f16x2 r409, %52, %76;
}
{
mul.f16x2 r412, r409, r594;
}
{
add.f16x2 r415, %44, r412;
}
{
add.f16x2 r418, %60, %68;
}
{
mul.f16x2 r421, r418, r603;
}
{
add.f16x2 r424, r415, r421;
}
{
sub.f16x2 r427, %53, %77;
}
{
mul.f16x2 r430, r427, r612;
}
{
sub.f16x2 r433, %61, %69;
}
{
mul.f16x2 r436, r433, r618;
}
{
add.f16x2 r439, r430, r436;
}
{
sub.f16x2 r442, r424, r439;
}
{
add.f16x2 r445, %52, %76;
}
{
mul.f16x2 r448, r445, r594;
}
{
add.f16x2 r451, %44, r448;
}
{
add.f16x2 r454, %60, %68;
}
{
mul.f16x2 r457, r454, r603;
}
{
add.f16x2 r460, r451, r457;
}
{
sub.f16x2 r463, %53, %77;
}
{
mul.f16x2 r466, r463, r612;
}
{
sub.f16x2 r469, %61, %69;
}
{
mul.f16x2 r472, r469, r618;
}
{
add.f16x2 r475, r466, r472;
}
{
add.f16x2 r478, r460, r475;
}
{
add.f16x2 r481, %53, %77;
}
{
mul.f16x2 r484, r481, r522;
}
{
add.f16x2 r487, %45, r484;
}
{
add.f16x2 r490, %61, %69;
}
{
mul.f16x2 r493, r490, r594;
}
{
add.f16x2 r496, r487, r493;
}
{
sub.f16x2 r499, %52, %76;
}
{
mul.f16x2 r502, r499, r540;
}
{
sub.f16x2 r505, %60, %68;
}
{
mul.f16x2 r508, r505, r612;
}
{
add.f16x2 r511, r502, r508;
}
{
add.f16x2 r514, r496, r511;
}
{
add.f16x2 r517, %53, %77;
}
{
mul.f16x2 r520, r517, r522;
}
{
add.f16x2 r523, %45, r520;
}
{
add.f16x2 r526, %61, %69;
}
{
mul.f16x2 r529, r526, r594;
}
{
add.f16x2 r532, r523, r529;
}
{
sub.f16x2 r535, %52, %76;
}
{
mul.f16x2 r538, r535, r540;
}
{
sub.f16x2 r541, %60, %68;
}
{
mul.f16x2 r544, r541, r612;
}
{
add.f16x2 r547, r538, r544;
}
{
sub.f16x2 r550, r532, r547;
}
{
add.f16x2 r553, %53, %77;
}
{
mul.f16x2 r556, r553, r594;
}
{
add.f16x2 r559, %45, r556;
}
{
add.f16x2 r562, %61, %69;
}
{
mul.f16x2 r565, r562, r603;
}
{
add.f16x2 r568, r559, r565;
}
{
sub.f16x2 r571, %52, %76;
}
{
mul.f16x2 r574, r571, r612;
}
{
sub.f16x2 r577, %60, %68;
}
{
mul.f16x2 r580, r577, r618;
}
{
add.f16x2 r583, r574, r580;
}
{
add.f16x2 r586, r568, r583;
}
{
add.f16x2 r589, %53, %77;
}
{
mul.f16x2 r592, r589, r594;
}
{
add.f16x2 r595, %45, r592;
}
{
add.f16x2 r598, %61, %69;
}
{
mul.f16x2 r601, r598, r603;
}
{
add.f16x2 r604, r595, r601;
}
{
sub.f16x2 r607, %52, %76;
}
{
mul.f16x2 r610, r607, r612;
}
{
sub.f16x2 r613, %60, %68;
}
{
mul.f16x2 r616, r613, r618;
}
{
add.f16x2 r619, r610, r616;
}
{
sub.f16x2 r622, r604, r619;
}
mov.f64 fd63, 0d3FE9E3779B97F4A8;
{
cvt.rn.f16.f64 rs17, fd63;
}
{
cvt.rn.f16.f64 rs18, fd76;
}
{
cvt.rn.f16.f64 rs19, fd67;
}
{
cvt.rn.f16.f64 rs20, fd77;
}
mov.f64 fd78, 0dBFD3C6EF372FE950;
{
cvt.rn.f16.f64 rs21, fd78;
}
{
cvt.rn.f16.f64 rs22, fd77;
}
{
cvt.rn.f16.f64 rs23, fd75;
}
{
cvt.rn.f16.f64 rs24, fd76;
}
mov.f64 fd65, 0d3FE2CF2304755A5E;
mov.f64 fd61, 0d3FEE6F0E134454FF;
mov.b32 r639, {rs17, rs17};
{
mul.f16x2 r625, r370, r639;
}
mov.b32 r636, {rs18, rs18};
{
mul.f16x2 r628, r514, r636;
}
{
sub.f16x2 r631, r625, r628;
}
{
mul.f16x2 r634, r370, r636;
}
{
fma.rn.f16x2 r637, r514, r639, r634;
}
mov.b32 r655, {rs19, rs19};
{
mul.f16x2 r641, r442, r655;
}
mov.b32 r652, {rs20, rs20};
{
mul.f16x2 r644, r586, r652;
}
{
sub.f16x2 r647, r641, r644;
}
{
mul.f16x2 r650, r442, r652;
}
{
fma.rn.f16x2 r653, r586, r655, r650;
}
mov.b32 r671, {rs21, rs21};
{
mul.f16x2 r657, r478, r671;
}
mov.b32 r668, {rs22, rs22};
{
mul.f16x2 r660, r622, r668;
}
{
sub.f16x2 r663, r657, r660;
}
{
mul.f16x2 r666, r478, r668;
}
{
fma.rn.f16x2 r669, r622, r671, r666;
}
mov.b32 r687, {rs23, rs23};
{
mul.f16x2 r673, r406, r687;
}
mov.b32 r684, {rs24, rs24};
{
mul.f16x2 r676, r550, r684;
}
{
sub.f16x2 r679, r673, r676;
}
{
mul.f16x2 r682, r406, r684;
}
{
fma.rn.f16x2 r685, r550, r687, r682;
}
{
add.f16x2 r689, r10, r322;
}
{
add.f16x2 r692, r22, r334;
}
{
sub.f16x2 r695, r10, r322;
}
{
sub.f16x2 r698, r22, r334;
}
{
add.f16x2 r701, r58, r631;
}
{
add.f16x2 r704, r202, r637;
}
{
sub.f16x2 r707, r58, r631;
}
{
sub.f16x2 r710, r202, r637;
}
{
add.f16x2 r713, r130, r647;
}
{
add.f16x2 r716, r274, r653;
}
{
sub.f16x2 r719, r130, r647;
}
{
sub.f16x2 r722, r274, r653;
}
{
add.f16x2 r725, r166, r663;
}
{
add.f16x2 r728, r310, r669;
}
{
sub.f16x2 r731, r166, r663;
}
{
sub.f16x2 r734, r310, r669;
}
{
add.f16x2 r737, r94, r679;
}
{
add.f16x2 r740, r238, r685;
}
{
sub.f16x2 r743, r94, r679;
}
{
sub.f16x2 r746, r238, r685;
}
{
cvt.rn.f16.f64 rs35, fd67;
}
mov.b32 r958, {rs35, rs35};
{
cvt.rn.f16.f64 rs36, fd77;
}
mov.b32 r976, {rs36, rs36};
{
cvt.rn.f16.f64 rs37, fd75;
}
mov.b32 r1030, {rs37, rs37};
{
cvt.rn.f16.f64 rs38, fd76;
}
mov.b32 r1048, {rs38, rs38};
{
cvt.rn.f16.f64 rs39, fd67;
}
mov.b32 r1039, {rs39, rs39};
{
cvt.rn.f16.f64 rs40, fd77;
}
{
neg.f16 rs41, rs40;
}
mov.b32 r1054, {rs41, rs41};
{
add.f16x2 r749, %50, %74;
}
{
add.f16x2 r752, %42, r749;
}
{
add.f16x2 r755, %58, %66;
}
{
add.f16x2 r758, r752, r755;
}
{
add.f16x2 r761, %51, %75;
}
{
add.f16x2 r764, %43, r761;
}
{
add.f16x2 r767, %59, %67;
}
{
add.f16x2 r770, r764, r767;
}
{
add.f16x2 r773, %50, %74;
}
{
mul.f16x2 r776, r773, r958;
}
{
add.f16x2 r779, %42, r776;
}
{
add.f16x2 r782, %58, %66;
}
{
mul.f16x2 r785, r782, r1030;
}
{
add.f16x2 r788, r779, r785;
}
{
sub.f16x2 r791, %51, %75;
}
{
mul.f16x2 r794, r791, r976;
}
{
sub.f16x2 r797, %59, %67;
}
{
mul.f16x2 r800, r797, r1048;
}
{
add.f16x2 r803, r794, r800;
}
{
sub.f16x2 r806, r788, r803;
}
{
add.f16x2 r809, %50, %74;
}
{
mul.f16x2 r812, r809, r958;
}
{
add.f16x2 r815, %42, r812;
}
{
add.f16x2 r818, %58, %66;
}
{
mul.f16x2 r821, r818, r1030;
}
{
add.f16x2 r824, r815, r821;
}
{
sub.f16x2 r827, %51, %75;
}
{
mul.f16x2 r830, r827, r976;
}
{
sub.f16x2 r833, %59, %67;
}
{
mul.f16x2 r836, r833, r1048;
}
{
add.f16x2 r839, r830, r836;
}
{
add.f16x2 r842, r824, r839;
}
{
add.f16x2 r845, %50, %74;
}
{
mul.f16x2 r848, r845, r1030;
}
{
add.f16x2 r851, %42, r848;
}
{
add.f16x2 r854, %58, %66;
}
{
mul.f16x2 r857, r854, r1039;
}
{
add.f16x2 r860, r851, r857;
}
{
sub.f16x2 r863, %51, %75;
}
{
mul.f16x2 r866, r863, r1048;
}
{
sub.f16x2 r869, %59, %67;
}
{
mul.f16x2 r872, r869, r1054;
}
{
add.f16x2 r875, r866, r872;
}
{
sub.f16x2 r878, r860, r875;
}
{
add.f16x2 r881, %50, %74;
}
{
mul.f16x2 r884, r881, r1030;
}
{
add.f16x2 r887, %42, r884;
}
{
add.f16x2 r890, %58, %66;
}
{
mul.f16x2 r893, r890, r1039;
}
{
add.f16x2 r896, r887, r893;
}
{
sub.f16x2 r899, %51, %75;
}
{
mul.f16x2 r902, r899, r1048;
}
{
sub.f16x2 r905, %59, %67;
}
{
mul.f16x2 r908, r905, r1054;
}
{
add.f16x2 r911, r902, r908;
}
{
add.f16x2 r914, r896, r911;
}
{
add.f16x2 r917, %51, %75;
}
{
mul.f16x2 r920, r917, r958;
}
{
add.f16x2 r923, %43, r920;
}
{
add.f16x2 r926, %59, %67;
}
{
mul.f16x2 r929, r926, r1030;
}
{
add.f16x2 r932, r923, r929;
}
{
sub.f16x2 r935, %50, %74;
}
{
mul.f16x2 r938, r935, r976;
}
{
sub.f16x2 r941, %58, %66;
}
{
mul.f16x2 r944, r941, r1048;
}
{
add.f16x2 r947, r938, r944;
}
{
add.f16x2 r950, r932, r947;
}
{
add.f16x2 r953, %51, %75;
}
{
mul.f16x2 r956, r953, r958;
}
{
add.f16x2 r959, %43, r956;
}
{
add.f16x2 r962, %59, %67;
}
{
mul.f16x2 r965, r962, r1030;
}
{
add.f16x2 r968, r959, r965;
}
{
sub.f16x2 r971, %50, %74;
}
{
mul.f16x2 r974, r971, r976;
}
{
sub.f16x2 r977, %58, %66;
}
{
mul.f16x2 r980, r977, r1048;
}
{
add.f16x2 r983, r974, r980;
}
{
sub.f16x2 r986, r968, r983;
}
{
add.f16x2 r989, %51, %75;
}
{
mul.f16x2 r992, r989, r1030;
}
{
add.f16x2 r995, %43, r992;
}
{
add.f16x2 r998, %59, %67;
}
{
mul.f16x2 r1001, r998, r1039;
}
{
add.f16x2 r1004, r995, r1001;
}
{
sub.f16x2 r1007, %50, %74;
}
{
mul.f16x2 r1010, r1007, r1048;
}
{
sub.f16x2 r1013, %58, %66;
}
{
mul.f16x2 r1016, r1013, r1054;
}
{
add.f16x2 r1019, r1010, r1016;
}
{
add.f16x2 r1022, r1004, r1019;
}
{
add.f16x2 r1025, %51, %75;
}
{
mul.f16x2 r1028, r1025, r1030;
}
{
add.f16x2 r1031, %43, r1028;
}
{
add.f16x2 r1034, %59, %67;
}
{
mul.f16x2 r1037, r1034, r1039;
}
{
add.f16x2 r1040, r1031, r1037;
}
{
sub.f16x2 r1043, %50, %74;
}
{
mul.f16x2 r1046, r1043, r1048;
}
{
sub.f16x2 r1049, %58, %66;
}
{
mul.f16x2 r1052, r1049, r1054;
}
{
add.f16x2 r1055, r1046, r1052;
}
{
sub.f16x2 r1058, r1040, r1055;
}
{
cvt.rn.f16.f64 rs43, fd67;
}
mov.b32 r1270, {rs43, rs43};
{
cvt.rn.f16.f64 rs44, fd77;
}
mov.b32 r1288, {rs44, rs44};
{
cvt.rn.f16.f64 rs45, fd75;
}
mov.b32 r1342, {rs45, rs45};
{
cvt.rn.f16.f64 rs46, fd76;
}
mov.b32 r1360, {rs46, rs46};
{
cvt.rn.f16.f64 rs47, fd67;
}
mov.b32 r1351, {rs47, rs47};
{
cvt.rn.f16.f64 rs48, fd77;
}
{
neg.f16 rs49, rs48;
}
mov.b32 r1366, {rs49, rs49};
{
add.f16x2 r1061, %54, %78;
}
{
add.f16x2 r1064, %46, r1061;
}
{
add.f16x2 r1067, %62, %70;
}
{
add.f16x2 r1070, r1064, r1067;
}
{
add.f16x2 r1073, %55, %79;
}
{
add.f16x2 r1076, %47, r1073;
}
{
add.f16x2 r1079, %63, %71;
}
{
add.f16x2 r1082, r1076, r1079;
}
{
add.f16x2 r1085, %54, %78;
}
{
mul.f16x2 r1088, r1085, r1270;
}
{
add.f16x2 r1091, %46, r1088;
}
{
add.f16x2 r1094, %62, %70;
}
{
mul.f16x2 r1097, r1094, r1342;
}
{
add.f16x2 r1100, r1091, r1097;
}
{
sub.f16x2 r1103, %55, %79;
}
{
mul.f16x2 r1106, r1103, r1288;
}
{
sub.f16x2 r1109, %63, %71;
}
{
mul.f16x2 r1112, r1109, r1360;
}
{
add.f16x2 r1115, r1106, r1112;
}
{
sub.f16x2 r1118, r1100, r1115;
}
{
add.f16x2 r1121, %54, %78;
}
{
mul.f16x2 r1124, r1121, r1270;
}
{
add.f16x2 r1127, %46, r1124;
}
{
add.f16x2 r1130, %62, %70;
}
{
mul.f16x2 r1133, r1130, r1342;
}
{
add.f16x2 r1136, r1127, r1133;
}
{
sub.f16x2 r1139, %55, %79;
}
{
mul.f16x2 r1142, r1139, r1288;
}
{
sub.f16x2 r1145, %63, %71;
}
{
mul.f16x2 r1148, r1145, r1360;
}
{
add.f16x2 r1151, r1142, r1148;
}
{
add.f16x2 r1154, r1136, r1151;
}
{
add.f16x2 r1157, %54, %78;
}
{
mul.f16x2 r1160, r1157, r1342;
}
{
add.f16x2 r1163, %46, r1160;
}
{
add.f16x2 r1166, %62, %70;
}
{
mul.f16x2 r1169, r1166, r1351;
}
{
add.f16x2 r1172, r1163, r1169;
}
{
sub.f16x2 r1175, %55, %79;
}
{
mul.f16x2 r1178, r1175, r1360;
}
{
sub.f16x2 r1181, %63, %71;
}
{
mul.f16x2 r1184, r1181, r1366;
}
{
add.f16x2 r1187, r1178, r1184;
}
{
sub.f16x2 r1190, r1172, r1187;
}
{
add.f16x2 r1193, %54, %78;
}
{
mul.f16x2 r1196, r1193, r1342;
}
{
add.f16x2 r1199, %46, r1196;
}
{
add.f16x2 r1202, %62, %70;
}
{
mul.f16x2 r1205, r1202, r1351;
}
{
add.f16x2 r1208, r1199, r1205;
}
{
sub.f16x2 r1211, %55, %79;
}
{
mul.f16x2 r1214, r1211, r1360;
}
{
sub.f16x2 r1217, %63, %71;
}
{
mul.f16x2 r1220, r1217, r1366;
}
{
add.f16x2 r1223, r1214, r1220;
}
{
add.f16x2 r1226, r1208, r1223;
}
{
add.f16x2 r1229, %55, %79;
}
{
mul.f16x2 r1232, r1229, r1270;
}
{
add.f16x2 r1235, %47, r1232;
}
{
add.f16x2 r1238, %63, %71;
}
{
mul.f16x2 r1241, r1238, r1342;
}
{
add.f16x2 r1244, r1235, r1241;
}
{
sub.f16x2 r1247, %54, %78;
}
{
mul.f16x2 r1250, r1247, r1288;
}
{
sub.f16x2 r1253, %62, %70;
}
{
mul.f16x2 r1256, r1253, r1360;
}
{
add.f16x2 r1259, r1250, r1256;
}
{
add.f16x2 r1262, r1244, r1259;
}
{
add.f16x2 r1265, %55, %79;
}
{
mul.f16x2 r1268, r1265, r1270;
}
{
add.f16x2 r1271, %47, r1268;
}
{
add.f16x2 r1274, %63, %71;
}
{
mul.f16x2 r1277, r1274, r1342;
}
{
add.f16x2 r1280, r1271, r1277;
}
{
sub.f16x2 r1283, %54, %78;
}
{
mul.f16x2 r1286, r1283, r1288;
}
{
sub.f16x2 r1289, %62, %70;
}
{
mul.f16x2 r1292, r1289, r1360;
}
{
add.f16x2 r1295, r1286, r1292;
}
{
sub.f16x2 r1298, r1280, r1295;
}
{
add.f16x2 r1301, %55, %79;
}
{
mul.f16x2 r1304, r1301, r1342;
}
{
add.f16x2 r1307, %47, r1304;
}
{
add.f16x2 r1310, %63, %71;
}
{
mul.f16x2 r1313, r1310, r1351;
}
{
add.f16x2 r1316, r1307, r1313;
}
{
sub.f16x2 r1319, %54, %78;
}
{
mul.f16x2 r1322, r1319, r1360;
}
{
sub.f16x2 r1325, %62, %70;
}
{
mul.f16x2 r1328, r1325, r1366;
}
{
add.f16x2 r1331, r1322, r1328;
}
{
add.f16x2 r1334, r1316, r1331;
}
{
add.f16x2 r1337, %55, %79;
}
{
mul.f16x2 r1340, r1337, r1342;
}
{
add.f16x2 r1343, %47, r1340;
}
{
add.f16x2 r1346, %63, %71;
}
{
mul.f16x2 r1349, r1346, r1351;
}
{
add.f16x2 r1352, r1343, r1349;
}
{
sub.f16x2 r1355, %54, %78;
}
{
mul.f16x2 r1358, r1355, r1360;
}
{
sub.f16x2 r1361, %62, %70;
}
{
mul.f16x2 r1364, r1361, r1366;
}
{
add.f16x2 r1367, r1358, r1364;
}
{
sub.f16x2 r1370, r1352, r1367;
}
{
cvt.rn.f16.f64 rs51, fd63;
}
{
cvt.rn.f16.f64 rs52, fd76;
}
{
cvt.rn.f16.f64 rs53, fd67;
}
{
cvt.rn.f16.f64 rs54, fd77;
}
{
cvt.rn.f16.f64 rs55, fd78;
}
{
cvt.rn.f16.f64 rs56, fd77;
}
{
cvt.rn.f16.f64 rs57, fd75;
}
{
cvt.rn.f16.f64 rs58, fd76;
}
mov.b32 r1387, {rs51, rs51};
{
mul.f16x2 r1373, r1118, r1387;
}
mov.b32 r1384, {rs52, rs52};
{
mul.f16x2 r1376, r1262, r1384;
}
{
sub.f16x2 r1379, r1373, r1376;
}
{
mul.f16x2 r1382, r1118, r1384;
}
{
fma.rn.f16x2 r1385, r1262, r1387, r1382;
}
mov.b32 r1403, {rs53, rs53};
{
mul.f16x2 r1389, r1190, r1403;
}
mov.b32 r1400, {rs54, rs54};
{
mul.f16x2 r1392, r1334, r1400;
}
{
sub.f16x2 r1395, r1389, r1392;
}
{
mul.f16x2 r1398, r1190, r1400;
}
{
fma.rn.f16x2 r1401, r1334, r1403, r1398;
}
mov.b32 r1419, {rs55, rs55};
{
mul.f16x2 r1405, r1226, r1419;
}
mov.b32 r1416, {rs56, rs56};
{
mul.f16x2 r1408, r1370, r1416;
}
{
sub.f16x2 r1411, r1405, r1408;
}
{
mul.f16x2 r1414, r1226, r1416;
}
{
fma.rn.f16x2 r1417, r1370, r1419, r1414;
}
mov.b32 r1435, {rs57, rs57};
{
mul.f16x2 r1421, r1154, r1435;
}
mov.b32 r1432, {rs58, rs58};
{
mul.f16x2 r1424, r1298, r1432;
}
{
sub.f16x2 r1427, r1421, r1424;
}
{
mul.f16x2 r1430, r1154, r1432;
}
{
fma.rn.f16x2 r1433, r1298, r1435, r1430;
}
{
add.f16x2 r1437, r758, r1070;
}
{
add.f16x2 r1440, r770, r1082;
}
{
sub.f16x2 r1443, r758, r1070;
}
{
sub.f16x2 r1446, r770, r1082;
}
{
add.f16x2 r1449, r806, r1379;
}
{
add.f16x2 r1452, r950, r1385;
}
{
sub.f16x2 r1455, r806, r1379;
}
{
sub.f16x2 r1458, r950, r1385;
}
{
add.f16x2 r1461, r878, r1395;
}
{
add.f16x2 r1464, r1022, r1401;
}
{
sub.f16x2 r1467, r878, r1395;
}
{
sub.f16x2 r1470, r1022, r1401;
}
{
add.f16x2 r1473, r914, r1411;
}
{
add.f16x2 r1476, r1058, r1417;
}
{
sub.f16x2 r1479, r914, r1411;
}
{
sub.f16x2 r1482, r1058, r1417;
}
{
add.f16x2 r1485, r842, r1427;
}
{
add.f16x2 r1488, r986, r1433;
}
{
sub.f16x2 r1491, r842, r1427;
}
{
sub.f16x2 r1494, r986, r1433;
}
{
cvt.rn.f16.f64 rs69, fd61;
}
{
cvt.rn.f16.f64 rs70, fd78;
}
{
cvt.rn.f16.f64 rs71, fd63;
}
{
cvt.rn.f16.f64 rs72, fd76;
}
{
cvt.rn.f16.f64 rs73, fd65;
}
{
cvt.rn.f16.f64 rs74, fd75;
}
{
cvt.rn.f16.f64 rs75, fd67;
}
{
cvt.rn.f16.f64 rs76, fd77;
}
{
cvt.rn.f16.f64 rs79, fd78;
}
{
cvt.rn.f16.f64 rs80, fd77;
}
{
cvt.rn.f16.f64 rs81, fd76;
}
{
cvt.rn.f16.f64 rs82, fd75;
}
{
cvt.rn.f16.f64 rs83, fd75;
}
{
cvt.rn.f16.f64 rs84, fd76;
}
{
cvt.rn.f16.f64 rs85, fd77;
}
{
cvt.rn.f16.f64 rs86, fd78;
}
mov.b32 r1511, {rs69, rs69};
{
mul.f16x2 r1497, r1449, r1511;
}
mov.b32 r1508, {rs70, rs70};
{
mul.f16x2 r1500, r1452, r1508;
}
{
sub.f16x2 r1503, r1497, r1500;
}
{
mul.f16x2 r1506, r1449, r1508;
}
{
fma.rn.f16x2 r1509, r1452, r1511, r1506;
}
mov.b32 r1527, {rs71, rs71};
{
mul.f16x2 r1513, r1461, r1527;
}
mov.b32 r1524, {rs72, rs72};
{
mul.f16x2 r1516, r1464, r1524;
}
{
sub.f16x2 r1519, r1513, r1516;
}
{
mul.f16x2 r1522, r1461, r1524;
}
{
fma.rn.f16x2 r1525, r1464, r1527, r1522;
}
mov.b32 r1543, {rs73, rs73};
{
mul.f16x2 r1529, r1473, r1543;
}
mov.b32 r1540, {rs74, rs74};
{
mul.f16x2 r1532, r1476, r1540;
}
{
sub.f16x2 r1535, r1529, r1532;
}
{
mul.f16x2 r1538, r1473, r1540;
}
{
fma.rn.f16x2 r1541, r1476, r1543, r1538;
}
mov.b32 r1559, {rs75, rs75};
{
mul.f16x2 r1545, r1485, r1559;
}
mov.b32 r1556, {rs76, rs76};
{
mul.f16x2 r1548, r1488, r1556;
}
{
sub.f16x2 r1551, r1545, r1548;
}
{
mul.f16x2 r1554, r1485, r1556;
}
{
fma.rn.f16x2 r1557, r1488, r1559, r1554;
}
{
neg.f16x2 r1561, r1443;
}
mov.b32 r1577, {rs79, rs79};
{
mul.f16x2 r1563, r1455, r1577;
}
mov.b32 r1574, {rs80, rs80};
{
mul.f16x2 r1566, r1458, r1574;
}
{
sub.f16x2 r1569, r1563, r1566;
}
{
mul.f16x2 r1572, r1455, r1574;
}
{
fma.rn.f16x2 r1575, r1458, r1577, r1572;
}
mov.b32 r1593, {rs81, rs81};
{
mul.f16x2 r1579, r1467, r1593;
}
mov.b32 r1590, {rs82, rs82};
{
mul.f16x2 r1582, r1470, r1590;
}
{
sub.f16x2 r1585, r1579, r1582;
}
{
mul.f16x2 r1588, r1467, r1590;
}
{
fma.rn.f16x2 r1591, r1470, r1593, r1588;
}
mov.b32 r1609, {rs83, rs83};
{
mul.f16x2 r1595, r1479, r1609;
}
mov.b32 r1606, {rs84, rs84};
{
mul.f16x2 r1598, r1482, r1606;
}
{
sub.f16x2 r1601, r1595, r1598;
}
{
mul.f16x2 r1604, r1479, r1606;
}
{
fma.rn.f16x2 r1607, r1482, r1609, r1604;
}
mov.b32 r1625, {rs85, rs85};
{
mul.f16x2 r1611, r1491, r1625;
}
mov.b32 r1622, {rs86, rs86};
{
mul.f16x2 r1614, r1494, r1622;
}
{
sub.f16x2 r1617, r1611, r1614;
}
{
mul.f16x2 r1620, r1491, r1622;
}
{
fma.rn.f16x2 r1623, r1494, r1625, r1620;
}
{
add.f16x2 %0, r689, r1437;
}
{
add.f16x2 %1, r692, r1440;
}
{
sub.f16x2 %20, r689, r1437;
}
{
sub.f16x2 %21, r692, r1440;
}
{
add.f16x2 %2, r701, r1503;
}
{
add.f16x2 %3, r704, r1509;
}
{
sub.f16x2 %22, r701, r1503;
}
{
sub.f16x2 %23, r704, r1509;
}
{
add.f16x2 %4, r713, r1519;
}
{
add.f16x2 %5, r716, r1525;
}
{
sub.f16x2 %24, r713, r1519;
}
{
sub.f16x2 %25, r716, r1525;
}
{
add.f16x2 %6, r725, r1535;
}
{
add.f16x2 %7, r728, r1541;
}
{
sub.f16x2 %26, r725, r1535;
}
{
sub.f16x2 %27, r728, r1541;
}
{
add.f16x2 %8, r737, r1551;
}
{
add.f16x2 %9, r740, r1557;
}
{
sub.f16x2 %28, r737, r1551;
}
{
sub.f16x2 %29, r740, r1557;
}
{
add.f16x2 %10, r695, r1446;
}
{
add.f16x2 %11, r698, r1561;
}
{
sub.f16x2 %30, r695, r1446;
}
{
sub.f16x2 %31, r698, r1561;
}
{
add.f16x2 %12, r707, r1569;
}
{
add.f16x2 %13, r710, r1575;
}
{
sub.f16x2 %32, r707, r1569;
}
{
sub.f16x2 %33, r710, r1575;
}
{
add.f16x2 %14, r719, r1585;
}
{
add.f16x2 %15, r722, r1591;
}
{
sub.f16x2 %34, r719, r1585;
}
{
sub.f16x2 %35, r722, r1591;
}
{
add.f16x2 %16, r731, r1601;
}
{
add.f16x2 %17, r734, r1607;
}
{
sub.f16x2 %36, r731, r1601;
}
{
sub.f16x2 %37, r734, r1607;
}
{
add.f16x2 %18, r743, r1617;
}
{
add.f16x2 %19, r746, r1623;
}
{
sub.f16x2 %38, r743, r1617;
}
{
sub.f16x2 %39, r746, r1623;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[19].y)));
};


#endif
