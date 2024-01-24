#ifndef CUFFTDX_FFT_15_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_15_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<950, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<69>;
.reg .b32 r<1485>;
.reg .f64 fd<57>;
.reg .b64 rd<2>;
mov.f64 fd23, 0d3FD3C6EF372FE950;
{
cvt.rn.f16.f64 rs1, fd23;
}
mov.b32 r210, {rs1, rs1};
mov.f64 fd18, 0dBFEE6F0E134454FF;
{
cvt.rn.f16.f64 rs2, fd18;
}
{
neg.f16 rs3, rs2;
}
mov.b32 r228, {rs3, rs3};
mov.f64 fd29, 0dBFE9E3779B97F4A8;
{
cvt.rn.f16.f64 rs5, fd29;
}
mov.b32 r282, {rs5, rs5};
mov.f64 fd16, 0dBFE2CF2304755A5E;
{
cvt.rn.f16.f64 rs6, fd16;
}
{
neg.f16 rs7, rs6;
}
mov.b32 r300, {rs7, rs7};
{
cvt.rn.f16.f64 rs9, fd23;
}
mov.b32 r291, {rs9, rs9};
{
cvt.rn.f16.f64 rs10, fd18;
}
mov.b32 r306, {rs10, rs10};
{
add.f16x2 r1, %36, %54;
}
{
add.f16x2 r4, %30, r1;
}
{
add.f16x2 r7, %42, %48;
}
{
add.f16x2 r10, r4, r7;
}
{
add.f16x2 r13, %37, %55;
}
{
add.f16x2 r16, %31, r13;
}
{
add.f16x2 r19, %43, %49;
}
{
add.f16x2 r22, r16, r19;
}
{
add.f16x2 r25, %36, %54;
}
{
mul.f16x2 r28, r25, r210;
}
{
add.f16x2 r31, %30, r28;
}
{
add.f16x2 r34, %42, %48;
}
{
mul.f16x2 r37, r34, r282;
}
{
add.f16x2 r40, r31, r37;
}
{
sub.f16x2 r43, %37, %55;
}
{
mul.f16x2 r46, r43, r228;
}
{
sub.f16x2 r49, %43, %49;
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
add.f16x2 r61, %36, %54;
}
{
mul.f16x2 r64, r61, r210;
}
{
add.f16x2 r67, %30, r64;
}
{
add.f16x2 r70, %42, %48;
}
{
mul.f16x2 r73, r70, r282;
}
{
add.f16x2 r76, r67, r73;
}
{
sub.f16x2 r79, %37, %55;
}
{
mul.f16x2 r82, r79, r228;
}
{
sub.f16x2 r85, %43, %49;
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
add.f16x2 r97, %36, %54;
}
{
mul.f16x2 r100, r97, r282;
}
{
add.f16x2 r103, %30, r100;
}
{
add.f16x2 r106, %42, %48;
}
{
mul.f16x2 r109, r106, r291;
}
{
add.f16x2 r112, r103, r109;
}
{
sub.f16x2 r115, %37, %55;
}
{
mul.f16x2 r118, r115, r300;
}
{
sub.f16x2 r121, %43, %49;
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
add.f16x2 r133, %36, %54;
}
{
mul.f16x2 r136, r133, r282;
}
{
add.f16x2 r139, %30, r136;
}
{
add.f16x2 r142, %42, %48;
}
{
mul.f16x2 r145, r142, r291;
}
{
add.f16x2 r148, r139, r145;
}
{
sub.f16x2 r151, %37, %55;
}
{
mul.f16x2 r154, r151, r300;
}
{
sub.f16x2 r157, %43, %49;
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
add.f16x2 r169, %37, %55;
}
{
mul.f16x2 r172, r169, r210;
}
{
add.f16x2 r175, %31, r172;
}
{
add.f16x2 r178, %43, %49;
}
{
mul.f16x2 r181, r178, r282;
}
{
add.f16x2 r184, r175, r181;
}
{
sub.f16x2 r187, %36, %54;
}
{
mul.f16x2 r190, r187, r228;
}
{
sub.f16x2 r193, %42, %48;
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
add.f16x2 r205, %37, %55;
}
{
mul.f16x2 r208, r205, r210;
}
{
add.f16x2 r211, %31, r208;
}
{
add.f16x2 r214, %43, %49;
}
{
mul.f16x2 r217, r214, r282;
}
{
add.f16x2 r220, r211, r217;
}
{
sub.f16x2 r223, %36, %54;
}
{
mul.f16x2 r226, r223, r228;
}
{
sub.f16x2 r229, %42, %48;
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
add.f16x2 r241, %37, %55;
}
{
mul.f16x2 r244, r241, r282;
}
{
add.f16x2 r247, %31, r244;
}
{
add.f16x2 r250, %43, %49;
}
{
mul.f16x2 r253, r250, r291;
}
{
add.f16x2 r256, r247, r253;
}
{
sub.f16x2 r259, %36, %54;
}
{
mul.f16x2 r262, r259, r300;
}
{
sub.f16x2 r265, %42, %48;
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
add.f16x2 r277, %37, %55;
}
{
mul.f16x2 r280, r277, r282;
}
{
add.f16x2 r283, %31, r280;
}
{
add.f16x2 r286, %43, %49;
}
{
mul.f16x2 r289, r286, r291;
}
{
add.f16x2 r292, r283, r289;
}
{
sub.f16x2 r295, %36, %54;
}
{
mul.f16x2 r298, r295, r300;
}
{
sub.f16x2 r301, %42, %48;
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
cvt.rn.f16.f64 rs11, fd23;
}
mov.b32 r522, {rs11, rs11};
{
cvt.rn.f16.f64 rs12, fd18;
}
{
neg.f16 rs13, rs12;
}
mov.b32 r540, {rs13, rs13};
{
cvt.rn.f16.f64 rs15, fd29;
}
mov.b32 r594, {rs15, rs15};
{
cvt.rn.f16.f64 rs16, fd16;
}
{
neg.f16 rs17, rs16;
}
mov.b32 r612, {rs17, rs17};
{
cvt.rn.f16.f64 rs19, fd23;
}
mov.b32 r603, {rs19, rs19};
{
cvt.rn.f16.f64 rs20, fd18;
}
mov.b32 r618, {rs20, rs20};
{
add.f16x2 r313, %38, %56;
}
{
add.f16x2 r316, %32, r313;
}
{
add.f16x2 r319, %44, %50;
}
{
add.f16x2 r322, r316, r319;
}
{
add.f16x2 r325, %39, %57;
}
{
add.f16x2 r328, %33, r325;
}
{
add.f16x2 r331, %45, %51;
}
{
add.f16x2 r334, r328, r331;
}
{
add.f16x2 r337, %38, %56;
}
{
mul.f16x2 r340, r337, r522;
}
{
add.f16x2 r343, %32, r340;
}
{
add.f16x2 r346, %44, %50;
}
{
mul.f16x2 r349, r346, r594;
}
{
add.f16x2 r352, r343, r349;
}
{
sub.f16x2 r355, %39, %57;
}
{
mul.f16x2 r358, r355, r540;
}
{
sub.f16x2 r361, %45, %51;
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
add.f16x2 r373, %38, %56;
}
{
mul.f16x2 r376, r373, r522;
}
{
add.f16x2 r379, %32, r376;
}
{
add.f16x2 r382, %44, %50;
}
{
mul.f16x2 r385, r382, r594;
}
{
add.f16x2 r388, r379, r385;
}
{
sub.f16x2 r391, %39, %57;
}
{
mul.f16x2 r394, r391, r540;
}
{
sub.f16x2 r397, %45, %51;
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
add.f16x2 r409, %38, %56;
}
{
mul.f16x2 r412, r409, r594;
}
{
add.f16x2 r415, %32, r412;
}
{
add.f16x2 r418, %44, %50;
}
{
mul.f16x2 r421, r418, r603;
}
{
add.f16x2 r424, r415, r421;
}
{
sub.f16x2 r427, %39, %57;
}
{
mul.f16x2 r430, r427, r612;
}
{
sub.f16x2 r433, %45, %51;
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
add.f16x2 r445, %38, %56;
}
{
mul.f16x2 r448, r445, r594;
}
{
add.f16x2 r451, %32, r448;
}
{
add.f16x2 r454, %44, %50;
}
{
mul.f16x2 r457, r454, r603;
}
{
add.f16x2 r460, r451, r457;
}
{
sub.f16x2 r463, %39, %57;
}
{
mul.f16x2 r466, r463, r612;
}
{
sub.f16x2 r469, %45, %51;
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
add.f16x2 r481, %39, %57;
}
{
mul.f16x2 r484, r481, r522;
}
{
add.f16x2 r487, %33, r484;
}
{
add.f16x2 r490, %45, %51;
}
{
mul.f16x2 r493, r490, r594;
}
{
add.f16x2 r496, r487, r493;
}
{
sub.f16x2 r499, %38, %56;
}
{
mul.f16x2 r502, r499, r540;
}
{
sub.f16x2 r505, %44, %50;
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
add.f16x2 r517, %39, %57;
}
{
mul.f16x2 r520, r517, r522;
}
{
add.f16x2 r523, %33, r520;
}
{
add.f16x2 r526, %45, %51;
}
{
mul.f16x2 r529, r526, r594;
}
{
add.f16x2 r532, r523, r529;
}
{
sub.f16x2 r535, %38, %56;
}
{
mul.f16x2 r538, r535, r540;
}
{
sub.f16x2 r541, %44, %50;
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
add.f16x2 r553, %39, %57;
}
{
mul.f16x2 r556, r553, r594;
}
{
add.f16x2 r559, %33, r556;
}
{
add.f16x2 r562, %45, %51;
}
{
mul.f16x2 r565, r562, r603;
}
{
add.f16x2 r568, r559, r565;
}
{
sub.f16x2 r571, %38, %56;
}
{
mul.f16x2 r574, r571, r612;
}
{
sub.f16x2 r577, %44, %50;
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
add.f16x2 r589, %39, %57;
}
{
mul.f16x2 r592, r589, r594;
}
{
add.f16x2 r595, %33, r592;
}
{
add.f16x2 r598, %45, %51;
}
{
mul.f16x2 r601, r598, r603;
}
{
add.f16x2 r604, r595, r601;
}
{
sub.f16x2 r607, %38, %56;
}
{
mul.f16x2 r610, r607, r612;
}
{
sub.f16x2 r613, %44, %50;
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
{
cvt.rn.f16.f64 rs21, fd23;
}
mov.b32 r834, {rs21, rs21};
{
cvt.rn.f16.f64 rs22, fd18;
}
{
neg.f16 rs23, rs22;
}
mov.b32 r852, {rs23, rs23};
{
cvt.rn.f16.f64 rs25, fd29;
}
mov.b32 r906, {rs25, rs25};
{
cvt.rn.f16.f64 rs26, fd16;
}
{
neg.f16 rs27, rs26;
}
mov.b32 r924, {rs27, rs27};
{
cvt.rn.f16.f64 rs29, fd23;
}
mov.b32 r915, {rs29, rs29};
{
cvt.rn.f16.f64 rs30, fd18;
}
mov.b32 r930, {rs30, rs30};
{
add.f16x2 r625, %40, %58;
}
{
add.f16x2 r628, %34, r625;
}
{
add.f16x2 r631, %46, %52;
}
{
add.f16x2 r634, r628, r631;
}
{
add.f16x2 r637, %41, %59;
}
{
add.f16x2 r640, %35, r637;
}
{
add.f16x2 r643, %47, %53;
}
{
add.f16x2 r646, r640, r643;
}
{
add.f16x2 r649, %40, %58;
}
{
mul.f16x2 r652, r649, r834;
}
{
add.f16x2 r655, %34, r652;
}
{
add.f16x2 r658, %46, %52;
}
{
mul.f16x2 r661, r658, r906;
}
{
add.f16x2 r664, r655, r661;
}
{
sub.f16x2 r667, %41, %59;
}
{
mul.f16x2 r670, r667, r852;
}
{
sub.f16x2 r673, %47, %53;
}
{
mul.f16x2 r676, r673, r924;
}
{
add.f16x2 r679, r670, r676;
}
{
sub.f16x2 r682, r664, r679;
}
{
add.f16x2 r685, %40, %58;
}
{
mul.f16x2 r688, r685, r834;
}
{
add.f16x2 r691, %34, r688;
}
{
add.f16x2 r694, %46, %52;
}
{
mul.f16x2 r697, r694, r906;
}
{
add.f16x2 r700, r691, r697;
}
{
sub.f16x2 r703, %41, %59;
}
{
mul.f16x2 r706, r703, r852;
}
{
sub.f16x2 r709, %47, %53;
}
{
mul.f16x2 r712, r709, r924;
}
{
add.f16x2 r715, r706, r712;
}
{
add.f16x2 r718, r700, r715;
}
{
add.f16x2 r721, %40, %58;
}
{
mul.f16x2 r724, r721, r906;
}
{
add.f16x2 r727, %34, r724;
}
{
add.f16x2 r730, %46, %52;
}
{
mul.f16x2 r733, r730, r915;
}
{
add.f16x2 r736, r727, r733;
}
{
sub.f16x2 r739, %41, %59;
}
{
mul.f16x2 r742, r739, r924;
}
{
sub.f16x2 r745, %47, %53;
}
{
mul.f16x2 r748, r745, r930;
}
{
add.f16x2 r751, r742, r748;
}
{
sub.f16x2 r754, r736, r751;
}
{
add.f16x2 r757, %40, %58;
}
{
mul.f16x2 r760, r757, r906;
}
{
add.f16x2 r763, %34, r760;
}
{
add.f16x2 r766, %46, %52;
}
{
mul.f16x2 r769, r766, r915;
}
{
add.f16x2 r772, r763, r769;
}
{
sub.f16x2 r775, %41, %59;
}
{
mul.f16x2 r778, r775, r924;
}
{
sub.f16x2 r781, %47, %53;
}
{
mul.f16x2 r784, r781, r930;
}
{
add.f16x2 r787, r778, r784;
}
{
add.f16x2 r790, r772, r787;
}
{
add.f16x2 r793, %41, %59;
}
{
mul.f16x2 r796, r793, r834;
}
{
add.f16x2 r799, %35, r796;
}
{
add.f16x2 r802, %47, %53;
}
{
mul.f16x2 r805, r802, r906;
}
{
add.f16x2 r808, r799, r805;
}
{
sub.f16x2 r811, %40, %58;
}
{
mul.f16x2 r814, r811, r852;
}
{
sub.f16x2 r817, %46, %52;
}
{
mul.f16x2 r820, r817, r924;
}
{
add.f16x2 r823, r814, r820;
}
{
add.f16x2 r826, r808, r823;
}
{
add.f16x2 r829, %41, %59;
}
{
mul.f16x2 r832, r829, r834;
}
{
add.f16x2 r835, %35, r832;
}
{
add.f16x2 r838, %47, %53;
}
{
mul.f16x2 r841, r838, r906;
}
{
add.f16x2 r844, r835, r841;
}
{
sub.f16x2 r847, %40, %58;
}
{
mul.f16x2 r850, r847, r852;
}
{
sub.f16x2 r853, %46, %52;
}
{
mul.f16x2 r856, r853, r924;
}
{
add.f16x2 r859, r850, r856;
}
{
sub.f16x2 r862, r844, r859;
}
{
add.f16x2 r865, %41, %59;
}
{
mul.f16x2 r868, r865, r906;
}
{
add.f16x2 r871, %35, r868;
}
{
add.f16x2 r874, %47, %53;
}
{
mul.f16x2 r877, r874, r915;
}
{
add.f16x2 r880, r871, r877;
}
{
sub.f16x2 r883, %40, %58;
}
{
mul.f16x2 r886, r883, r924;
}
{
sub.f16x2 r889, %46, %52;
}
{
mul.f16x2 r892, r889, r930;
}
{
add.f16x2 r895, r886, r892;
}
{
add.f16x2 r898, r880, r895;
}
{
add.f16x2 r901, %41, %59;
}
{
mul.f16x2 r904, r901, r906;
}
{
add.f16x2 r907, %35, r904;
}
{
add.f16x2 r910, %47, %53;
}
{
mul.f16x2 r913, r910, r915;
}
{
add.f16x2 r916, r907, r913;
}
{
sub.f16x2 r919, %40, %58;
}
{
mul.f16x2 r922, r919, r924;
}
{
sub.f16x2 r925, %46, %52;
}
{
mul.f16x2 r928, r925, r930;
}
{
add.f16x2 r931, r922, r928;
}
{
sub.f16x2 r934, r916, r931;
}
mov.f64 fd19, 0d3FED3BC3AEFF7F95;
{
cvt.rn.f16.f64 rs31, fd19;
}
mov.f64 fd20, 0d3FDA07F921061AD1;
{
cvt.rn.f16.f64 rs32, fd20;
}
mov.f64 fd21, 0d3FE5698496E20BD8;
{
cvt.rn.f16.f64 rs33, fd21;
}
mov.f64 fd22, 0d3FE7C7D7A833BEC2;
{
cvt.rn.f16.f64 rs34, fd22;
}
{
cvt.rn.f16.f64 rs35, fd23;
}
mov.f64 fd24, 0d3FEE6F0E134454FF;
{
cvt.rn.f16.f64 rs36, fd24;
}
mov.f64 fd25, 0dBFBAC2609B3C576C;
{
cvt.rn.f16.f64 rs37, fd25;
}
mov.f64 fd26, 0d3FEFD31F94F867C6;
{
cvt.rn.f16.f64 rs38, fd26;
}
mov.f64 fd55, 0dBFE0000000000000;
{
cvt.rn.f16.f64 rs41, fd29;
}
mov.f64 fd30, 0d3FE2CF2304755A5E;
{
cvt.rn.f16.f64 rs42, fd30;
}
mov.f64 fd33, 0dBFEF4CFC327A0080;
{
cvt.rn.f16.f64 rs45, fd33;
}
mov.f64 fd34, 0dBFCA9CD9AC4258F6;
{
cvt.rn.f16.f64 rs46, fd34;
}
mov.f64 fd56, 0dBFEBB67AE8584CAA;
mov.b32 r951, {rs31, rs31};
{
mul.f16x2 r937, r370, r951;
}
mov.b32 r948, {rs32, rs32};
{
mul.f16x2 r940, r514, r948;
}
{
sub.f16x2 r943, r937, r940;
}
{
mul.f16x2 r946, r370, r948;
}
{
fma.rn.f16x2 r949, r514, r951, r946;
}
mov.b32 r983, {rs33, rs33};
{
mul.f16x2 r953, r682, r983;
}
mov.b32 r980, {rs34, rs34};
{
mul.f16x2 r956, r826, r980;
}
{
sub.f16x2 r959, r953, r956;
}
{
mul.f16x2 r962, r682, r980;
}
{
fma.rn.f16x2 r965, r826, r983, r962;
}
{
mul.f16x2 r969, r442, r983;
}
{
mul.f16x2 r972, r586, r980;
}
{
sub.f16x2 r975, r969, r972;
}
{
mul.f16x2 r978, r442, r980;
}
{
fma.rn.f16x2 r981, r586, r983, r978;
}
mov.b32 r1047, {rs37, rs37};
{
mul.f16x2 r985, r754, r1047;
}
mov.b32 r1044, {rs38, rs38};
{
mul.f16x2 r988, r898, r1044;
}
{
sub.f16x2 r991, r985, r988;
}
{
mul.f16x2 r994, r754, r1044;
}
{
fma.rn.f16x2 r997, r898, r1047, r994;
}
mov.b32 r1015, {rs35, rs35};
{
mul.f16x2 r1001, r478, r1015;
}
mov.b32 r1012, {rs36, rs36};
{
mul.f16x2 r1004, r622, r1012;
}
{
sub.f16x2 r1007, r1001, r1004;
}
{
mul.f16x2 r1010, r478, r1012;
}
{
fma.rn.f16x2 r1013, r622, r1015, r1010;
}
mov.b32 r1031, {rs41, rs41};
{
mul.f16x2 r1017, r790, r1031;
}
mov.b32 r1028, {rs42, rs42};
{
mul.f16x2 r1020, r934, r1028;
}
{
sub.f16x2 r1023, r1017, r1020;
}
{
mul.f16x2 r1026, r790, r1028;
}
{
fma.rn.f16x2 r1029, r934, r1031, r1026;
}
{
mul.f16x2 r1033, r406, r1047;
}
{
mul.f16x2 r1036, r550, r1044;
}
{
sub.f16x2 r1039, r1033, r1036;
}
{
mul.f16x2 r1042, r406, r1044;
}
{
fma.rn.f16x2 r1045, r550, r1047, r1042;
}
mov.b32 r1063, {rs45, rs45};
{
mul.f16x2 r1049, r718, r1063;
}
mov.b32 r1060, {rs46, rs46};
{
mul.f16x2 r1052, r862, r1060;
}
{
sub.f16x2 r1055, r1049, r1052;
}
{
mul.f16x2 r1058, r718, r1060;
}
{
fma.rn.f16x2 r1061, r862, r1063, r1058;
}
{
cvt.rn.f16.f64 rs59, fd55;
}
mov.b32 r1136, {rs59, rs59};
{
cvt.rn.f16.f64 rs60, fd56;
}
mov.b32 r1145, {rs60, rs60};
{
add.f16x2 r1065, r322, r634;
}
{
add.f16x2 %0, r10, r1065;
}
{
add.f16x2 r1071, r334, r646;
}
{
add.f16x2 %1, r22, r1071;
}
{
add.f16x2 r1077, r322, r634;
}
{
mul.f16x2 r1080, r1077, r1136;
}
{
add.f16x2 r1083, r10, r1080;
}
{
sub.f16x2 r1086, r334, r646;
}
{
mul.f16x2 r1089, r1086, r1145;
}
{
add.f16x2 %10, r1083, r1089;
}
{
add.f16x2 r1095, r322, r634;
}
{
mul.f16x2 r1098, r1095, r1136;
}
{
add.f16x2 r1101, r10, r1098;
}
{
sub.f16x2 r1104, r334, r646;
}
{
mul.f16x2 r1107, r1104, r1145;
}
{
sub.f16x2 %20, r1101, r1107;
}
{
add.f16x2 r1113, r334, r646;
}
{
mul.f16x2 r1116, r1113, r1136;
}
{
add.f16x2 r1119, r22, r1116;
}
{
sub.f16x2 r1122, r322, r634;
}
{
mul.f16x2 r1125, r1122, r1145;
}
{
sub.f16x2 %11, r1119, r1125;
}
{
add.f16x2 r1131, r334, r646;
}
{
mul.f16x2 r1134, r1131, r1136;
}
{
add.f16x2 r1137, r22, r1134;
}
{
sub.f16x2 r1140, r322, r634;
}
{
mul.f16x2 r1143, r1140, r1145;
}
{
add.f16x2 %21, r1137, r1143;
}
{
cvt.rn.f16.f64 rs61, fd55;
}
mov.b32 r1220, {rs61, rs61};
{
cvt.rn.f16.f64 rs62, fd56;
}
mov.b32 r1229, {rs62, rs62};
{
add.f16x2 r1149, r943, r959;
}
{
add.f16x2 %2, r58, r1149;
}
{
add.f16x2 r1155, r949, r965;
}
{
add.f16x2 %3, r202, r1155;
}
{
add.f16x2 r1161, r943, r959;
}
{
mul.f16x2 r1164, r1161, r1220;
}
{
add.f16x2 r1167, r58, r1164;
}
{
sub.f16x2 r1170, r949, r965;
}
{
mul.f16x2 r1173, r1170, r1229;
}
{
add.f16x2 %12, r1167, r1173;
}
{
add.f16x2 r1179, r943, r959;
}
{
mul.f16x2 r1182, r1179, r1220;
}
{
add.f16x2 r1185, r58, r1182;
}
{
sub.f16x2 r1188, r949, r965;
}
{
mul.f16x2 r1191, r1188, r1229;
}
{
sub.f16x2 %22, r1185, r1191;
}
{
add.f16x2 r1197, r949, r965;
}
{
mul.f16x2 r1200, r1197, r1220;
}
{
add.f16x2 r1203, r202, r1200;
}
{
sub.f16x2 r1206, r943, r959;
}
{
mul.f16x2 r1209, r1206, r1229;
}
{
sub.f16x2 %13, r1203, r1209;
}
{
add.f16x2 r1215, r949, r965;
}
{
mul.f16x2 r1218, r1215, r1220;
}
{
add.f16x2 r1221, r202, r1218;
}
{
sub.f16x2 r1224, r943, r959;
}
{
mul.f16x2 r1227, r1224, r1229;
}
{
add.f16x2 %23, r1221, r1227;
}
{
cvt.rn.f16.f64 rs63, fd55;
}
mov.b32 r1304, {rs63, rs63};
{
cvt.rn.f16.f64 rs64, fd56;
}
mov.b32 r1313, {rs64, rs64};
{
add.f16x2 r1233, r975, r991;
}
{
add.f16x2 %4, r130, r1233;
}
{
add.f16x2 r1239, r981, r997;
}
{
add.f16x2 %5, r274, r1239;
}
{
add.f16x2 r1245, r975, r991;
}
{
mul.f16x2 r1248, r1245, r1304;
}
{
add.f16x2 r1251, r130, r1248;
}
{
sub.f16x2 r1254, r981, r997;
}
{
mul.f16x2 r1257, r1254, r1313;
}
{
add.f16x2 %14, r1251, r1257;
}
{
add.f16x2 r1263, r975, r991;
}
{
mul.f16x2 r1266, r1263, r1304;
}
{
add.f16x2 r1269, r130, r1266;
}
{
sub.f16x2 r1272, r981, r997;
}
{
mul.f16x2 r1275, r1272, r1313;
}
{
sub.f16x2 %24, r1269, r1275;
}
{
add.f16x2 r1281, r981, r997;
}
{
mul.f16x2 r1284, r1281, r1304;
}
{
add.f16x2 r1287, r274, r1284;
}
{
sub.f16x2 r1290, r975, r991;
}
{
mul.f16x2 r1293, r1290, r1313;
}
{
sub.f16x2 %15, r1287, r1293;
}
{
add.f16x2 r1299, r981, r997;
}
{
mul.f16x2 r1302, r1299, r1304;
}
{
add.f16x2 r1305, r274, r1302;
}
{
sub.f16x2 r1308, r975, r991;
}
{
mul.f16x2 r1311, r1308, r1313;
}
{
add.f16x2 %25, r1305, r1311;
}
{
cvt.rn.f16.f64 rs65, fd55;
}
mov.b32 r1388, {rs65, rs65};
{
cvt.rn.f16.f64 rs66, fd56;
}
mov.b32 r1397, {rs66, rs66};
{
add.f16x2 r1317, r1007, r1023;
}
{
add.f16x2 %6, r166, r1317;
}
{
add.f16x2 r1323, r1013, r1029;
}
{
add.f16x2 %7, r310, r1323;
}
{
add.f16x2 r1329, r1007, r1023;
}
{
mul.f16x2 r1332, r1329, r1388;
}
{
add.f16x2 r1335, r166, r1332;
}
{
sub.f16x2 r1338, r1013, r1029;
}
{
mul.f16x2 r1341, r1338, r1397;
}
{
add.f16x2 %16, r1335, r1341;
}
{
add.f16x2 r1347, r1007, r1023;
}
{
mul.f16x2 r1350, r1347, r1388;
}
{
add.f16x2 r1353, r166, r1350;
}
{
sub.f16x2 r1356, r1013, r1029;
}
{
mul.f16x2 r1359, r1356, r1397;
}
{
sub.f16x2 %26, r1353, r1359;
}
{
add.f16x2 r1365, r1013, r1029;
}
{
mul.f16x2 r1368, r1365, r1388;
}
{
add.f16x2 r1371, r310, r1368;
}
{
sub.f16x2 r1374, r1007, r1023;
}
{
mul.f16x2 r1377, r1374, r1397;
}
{
sub.f16x2 %17, r1371, r1377;
}
{
add.f16x2 r1383, r1013, r1029;
}
{
mul.f16x2 r1386, r1383, r1388;
}
{
add.f16x2 r1389, r310, r1386;
}
{
sub.f16x2 r1392, r1007, r1023;
}
{
mul.f16x2 r1395, r1392, r1397;
}
{
add.f16x2 %27, r1389, r1395;
}
{
cvt.rn.f16.f64 rs67, fd55;
}
mov.b32 r1472, {rs67, rs67};
{
cvt.rn.f16.f64 rs68, fd56;
}
mov.b32 r1481, {rs68, rs68};
{
add.f16x2 r1401, r1039, r1055;
}
{
add.f16x2 %8, r94, r1401;
}
{
add.f16x2 r1407, r1045, r1061;
}
{
add.f16x2 %9, r238, r1407;
}
{
add.f16x2 r1413, r1039, r1055;
}
{
mul.f16x2 r1416, r1413, r1472;
}
{
add.f16x2 r1419, r94, r1416;
}
{
sub.f16x2 r1422, r1045, r1061;
}
{
mul.f16x2 r1425, r1422, r1481;
}
{
add.f16x2 %18, r1419, r1425;
}
{
add.f16x2 r1431, r1039, r1055;
}
{
mul.f16x2 r1434, r1431, r1472;
}
{
add.f16x2 r1437, r94, r1434;
}
{
sub.f16x2 r1440, r1045, r1061;
}
{
mul.f16x2 r1443, r1440, r1481;
}
{
sub.f16x2 %28, r1437, r1443;
}
{
add.f16x2 r1449, r1045, r1061;
}
{
mul.f16x2 r1452, r1449, r1472;
}
{
add.f16x2 r1455, r238, r1452;
}
{
sub.f16x2 r1458, r1039, r1055;
}
{
mul.f16x2 r1461, r1458, r1481;
}
{
sub.f16x2 %19, r1455, r1461;
}
{
add.f16x2 r1467, r1045, r1061;
}
{
mul.f16x2 r1470, r1467, r1472;
}
{
add.f16x2 r1473, r238, r1470;
}
{
sub.f16x2 r1476, r1039, r1055;
}
{
mul.f16x2 r1479, r1476, r1481;
}
{
add.f16x2 %29, r1473, r1479;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)));
};


#endif
