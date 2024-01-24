#ifndef CUFFTDX_FFT_21_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_21_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<955, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<103>;
.reg .b32 r<2833>;
.reg .f64 fd<85>;
.reg .b64 rd<2>;
mov.f64 fd35, 0d3FE3F3A0E28BEDD1;
{
cvt.rn.f16.f64 rs1, fd35;
}
mov.b32 r420, {rs1, rs1};
mov.f64 fd30, 0dBFE904C37505DE4B;
{
cvt.rn.f16.f64 rs2, fd30;
}
{
neg.f16 rs3, rs2;
}
mov.b32 r447, {rs3, rs3};
mov.f64 fd41, 0dBFCC7B90E3024582;
{
cvt.rn.f16.f64 rs5, fd41;
}
mov.b32 r654, {rs5, rs5};
mov.f64 fd24, 0dBFEF329C0558E969;
{
cvt.rn.f16.f64 rs6, fd24;
}
{
neg.f16 rs7, rs6;
}
mov.b32 r678, {rs7, rs7};
mov.f64 fd53, 0dBFECD4BCA9CB5C71;
{
cvt.rn.f16.f64 rs9, fd53;
}
mov.b32 r636, {rs9, rs9};
mov.f64 fd54, 0dBFDBC4C04D71ABC1;
{
cvt.rn.f16.f64 rs10, fd54;
}
{
neg.f16 rs11, rs10;
}
mov.b32 r663, {rs11, rs11};
{
cvt.rn.f16.f64 rs13, fd53;
}
mov.b32 r537, {rs13, rs13};
{
cvt.rn.f16.f64 rs14, fd54;
}
mov.b32 r561, {rs14, rs14};
{
cvt.rn.f16.f64 rs15, fd35;
}
mov.b32 r645, {rs15, rs15};
{
cvt.rn.f16.f64 rs16, fd30;
}
mov.b32 r669, {rs16, rs16};
{
add.f16x2 r1, %48, %78;
}
{
add.f16x2 r4, %42, r1;
}
{
add.f16x2 r7, %54, %72;
}
{
add.f16x2 r10, r4, r7;
}
{
add.f16x2 r13, %60, %66;
}
{
add.f16x2 r16, r10, r13;
}
{
add.f16x2 r19, %49, %79;
}
{
add.f16x2 r22, %43, r19;
}
{
add.f16x2 r25, %55, %73;
}
{
add.f16x2 r28, r22, r25;
}
{
add.f16x2 r31, %61, %67;
}
{
add.f16x2 r34, r28, r31;
}
{
add.f16x2 r37, %48, %78;
}
{
mul.f16x2 r40, r37, r420;
}
{
add.f16x2 r43, %42, r40;
}
{
add.f16x2 r46, %54, %72;
}
{
mul.f16x2 r49, r46, r654;
}
{
add.f16x2 r52, r43, r49;
}
{
add.f16x2 r55, %60, %66;
}
{
mul.f16x2 r58, r55, r636;
}
{
add.f16x2 r61, r52, r58;
}
{
sub.f16x2 r64, %49, %79;
}
{
mul.f16x2 r67, r64, r447;
}
{
sub.f16x2 r70, %55, %73;
}
{
mul.f16x2 r73, r70, r678;
}
{
add.f16x2 r76, r67, r73;
}
{
sub.f16x2 r79, %61, %67;
}
{
mul.f16x2 r82, r79, r663;
}
{
add.f16x2 r85, r76, r82;
}
{
sub.f16x2 r88, r61, r85;
}
{
add.f16x2 r91, %48, %78;
}
{
mul.f16x2 r94, r91, r420;
}
{
add.f16x2 r97, %42, r94;
}
{
add.f16x2 r100, %54, %72;
}
{
mul.f16x2 r103, r100, r654;
}
{
add.f16x2 r106, r97, r103;
}
{
add.f16x2 r109, %60, %66;
}
{
mul.f16x2 r112, r109, r636;
}
{
add.f16x2 r115, r106, r112;
}
{
sub.f16x2 r118, %49, %79;
}
{
mul.f16x2 r121, r118, r447;
}
{
sub.f16x2 r124, %55, %73;
}
{
mul.f16x2 r127, r124, r678;
}
{
add.f16x2 r130, r121, r127;
}
{
sub.f16x2 r133, %61, %67;
}
{
mul.f16x2 r136, r133, r663;
}
{
add.f16x2 r139, r130, r136;
}
{
add.f16x2 r142, r115, r139;
}
{
add.f16x2 r145, %48, %78;
}
{
mul.f16x2 r148, r145, r654;
}
{
add.f16x2 r151, %42, r148;
}
{
add.f16x2 r154, %54, %72;
}
{
mul.f16x2 r157, r154, r537;
}
{
add.f16x2 r160, r151, r157;
}
{
add.f16x2 r163, %60, %66;
}
{
mul.f16x2 r166, r163, r645;
}
{
add.f16x2 r169, r160, r166;
}
{
sub.f16x2 r172, %49, %79;
}
{
mul.f16x2 r175, r172, r678;
}
{
sub.f16x2 r178, %55, %73;
}
{
mul.f16x2 r181, r178, r561;
}
{
add.f16x2 r184, r175, r181;
}
{
sub.f16x2 r187, %61, %67;
}
{
mul.f16x2 r190, r187, r669;
}
{
add.f16x2 r193, r184, r190;
}
{
sub.f16x2 r196, r169, r193;
}
{
add.f16x2 r199, %48, %78;
}
{
mul.f16x2 r202, r199, r654;
}
{
add.f16x2 r205, %42, r202;
}
{
add.f16x2 r208, %54, %72;
}
{
mul.f16x2 r211, r208, r537;
}
{
add.f16x2 r214, r205, r211;
}
{
add.f16x2 r217, %60, %66;
}
{
mul.f16x2 r220, r217, r645;
}
{
add.f16x2 r223, r214, r220;
}
{
sub.f16x2 r226, %49, %79;
}
{
mul.f16x2 r229, r226, r678;
}
{
sub.f16x2 r232, %55, %73;
}
{
mul.f16x2 r235, r232, r561;
}
{
add.f16x2 r238, r229, r235;
}
{
sub.f16x2 r241, %61, %67;
}
{
mul.f16x2 r244, r241, r669;
}
{
add.f16x2 r247, r238, r244;
}
{
add.f16x2 r250, r223, r247;
}
{
add.f16x2 r253, %48, %78;
}
{
mul.f16x2 r256, r253, r636;
}
{
add.f16x2 r259, %42, r256;
}
{
add.f16x2 r262, %54, %72;
}
{
mul.f16x2 r265, r262, r645;
}
{
add.f16x2 r268, r259, r265;
}
{
add.f16x2 r271, %60, %66;
}
{
mul.f16x2 r274, r271, r654;
}
{
add.f16x2 r277, r268, r274;
}
{
sub.f16x2 r280, %49, %79;
}
{
mul.f16x2 r283, r280, r663;
}
{
sub.f16x2 r286, %55, %73;
}
{
mul.f16x2 r289, r286, r669;
}
{
add.f16x2 r292, r283, r289;
}
{
sub.f16x2 r295, %61, %67;
}
{
mul.f16x2 r298, r295, r678;
}
{
add.f16x2 r301, r292, r298;
}
{
sub.f16x2 r304, r277, r301;
}
{
add.f16x2 r307, %48, %78;
}
{
mul.f16x2 r310, r307, r636;
}
{
add.f16x2 r313, %42, r310;
}
{
add.f16x2 r316, %54, %72;
}
{
mul.f16x2 r319, r316, r645;
}
{
add.f16x2 r322, r313, r319;
}
{
add.f16x2 r325, %60, %66;
}
{
mul.f16x2 r328, r325, r654;
}
{
add.f16x2 r331, r322, r328;
}
{
sub.f16x2 r334, %49, %79;
}
{
mul.f16x2 r337, r334, r663;
}
{
sub.f16x2 r340, %55, %73;
}
{
mul.f16x2 r343, r340, r669;
}
{
add.f16x2 r346, r337, r343;
}
{
sub.f16x2 r349, %61, %67;
}
{
mul.f16x2 r352, r349, r678;
}
{
add.f16x2 r355, r346, r352;
}
{
add.f16x2 r358, r331, r355;
}
{
add.f16x2 r361, %49, %79;
}
{
mul.f16x2 r364, r361, r420;
}
{
add.f16x2 r367, %43, r364;
}
{
add.f16x2 r370, %55, %73;
}
{
mul.f16x2 r373, r370, r654;
}
{
add.f16x2 r376, r367, r373;
}
{
add.f16x2 r379, %61, %67;
}
{
mul.f16x2 r382, r379, r636;
}
{
add.f16x2 r385, r376, r382;
}
{
sub.f16x2 r388, %48, %78;
}
{
mul.f16x2 r391, r388, r447;
}
{
sub.f16x2 r394, %54, %72;
}
{
mul.f16x2 r397, r394, r678;
}
{
add.f16x2 r400, r391, r397;
}
{
sub.f16x2 r403, %60, %66;
}
{
mul.f16x2 r406, r403, r663;
}
{
add.f16x2 r409, r400, r406;
}
{
add.f16x2 r412, r385, r409;
}
{
add.f16x2 r415, %49, %79;
}
{
mul.f16x2 r418, r415, r420;
}
{
add.f16x2 r421, %43, r418;
}
{
add.f16x2 r424, %55, %73;
}
{
mul.f16x2 r427, r424, r654;
}
{
add.f16x2 r430, r421, r427;
}
{
add.f16x2 r433, %61, %67;
}
{
mul.f16x2 r436, r433, r636;
}
{
add.f16x2 r439, r430, r436;
}
{
sub.f16x2 r442, %48, %78;
}
{
mul.f16x2 r445, r442, r447;
}
{
sub.f16x2 r448, %54, %72;
}
{
mul.f16x2 r451, r448, r678;
}
{
add.f16x2 r454, r445, r451;
}
{
sub.f16x2 r457, %60, %66;
}
{
mul.f16x2 r460, r457, r663;
}
{
add.f16x2 r463, r454, r460;
}
{
sub.f16x2 r466, r439, r463;
}
{
add.f16x2 r469, %49, %79;
}
{
mul.f16x2 r472, r469, r654;
}
{
add.f16x2 r475, %43, r472;
}
{
add.f16x2 r478, %55, %73;
}
{
mul.f16x2 r481, r478, r537;
}
{
add.f16x2 r484, r475, r481;
}
{
add.f16x2 r487, %61, %67;
}
{
mul.f16x2 r490, r487, r645;
}
{
add.f16x2 r493, r484, r490;
}
{
sub.f16x2 r496, %48, %78;
}
{
mul.f16x2 r499, r496, r678;
}
{
sub.f16x2 r502, %54, %72;
}
{
mul.f16x2 r505, r502, r561;
}
{
add.f16x2 r508, r499, r505;
}
{
sub.f16x2 r511, %60, %66;
}
{
mul.f16x2 r514, r511, r669;
}
{
add.f16x2 r517, r508, r514;
}
{
add.f16x2 r520, r493, r517;
}
{
add.f16x2 r523, %49, %79;
}
{
mul.f16x2 r526, r523, r654;
}
{
add.f16x2 r529, %43, r526;
}
{
add.f16x2 r532, %55, %73;
}
{
mul.f16x2 r535, r532, r537;
}
{
add.f16x2 r538, r529, r535;
}
{
add.f16x2 r541, %61, %67;
}
{
mul.f16x2 r544, r541, r645;
}
{
add.f16x2 r547, r538, r544;
}
{
sub.f16x2 r550, %48, %78;
}
{
mul.f16x2 r553, r550, r678;
}
{
sub.f16x2 r556, %54, %72;
}
{
mul.f16x2 r559, r556, r561;
}
{
add.f16x2 r562, r553, r559;
}
{
sub.f16x2 r565, %60, %66;
}
{
mul.f16x2 r568, r565, r669;
}
{
add.f16x2 r571, r562, r568;
}
{
sub.f16x2 r574, r547, r571;
}
{
add.f16x2 r577, %49, %79;
}
{
mul.f16x2 r580, r577, r636;
}
{
add.f16x2 r583, %43, r580;
}
{
add.f16x2 r586, %55, %73;
}
{
mul.f16x2 r589, r586, r645;
}
{
add.f16x2 r592, r583, r589;
}
{
add.f16x2 r595, %61, %67;
}
{
mul.f16x2 r598, r595, r654;
}
{
add.f16x2 r601, r592, r598;
}
{
sub.f16x2 r604, %48, %78;
}
{
mul.f16x2 r607, r604, r663;
}
{
sub.f16x2 r610, %54, %72;
}
{
mul.f16x2 r613, r610, r669;
}
{
add.f16x2 r616, r607, r613;
}
{
sub.f16x2 r619, %60, %66;
}
{
mul.f16x2 r622, r619, r678;
}
{
add.f16x2 r625, r616, r622;
}
{
add.f16x2 r628, r601, r625;
}
{
add.f16x2 r631, %49, %79;
}
{
mul.f16x2 r634, r631, r636;
}
{
add.f16x2 r637, %43, r634;
}
{
add.f16x2 r640, %55, %73;
}
{
mul.f16x2 r643, r640, r645;
}
{
add.f16x2 r646, r637, r643;
}
{
add.f16x2 r649, %61, %67;
}
{
mul.f16x2 r652, r649, r654;
}
{
add.f16x2 r655, r646, r652;
}
{
sub.f16x2 r658, %48, %78;
}
{
mul.f16x2 r661, r658, r663;
}
{
sub.f16x2 r664, %54, %72;
}
{
mul.f16x2 r667, r664, r669;
}
{
add.f16x2 r670, r661, r667;
}
{
sub.f16x2 r673, %60, %66;
}
{
mul.f16x2 r676, r673, r678;
}
{
add.f16x2 r679, r670, r676;
}
{
sub.f16x2 r682, r655, r679;
}
{
cvt.rn.f16.f64 rs17, fd35;
}
mov.b32 r1104, {rs17, rs17};
{
cvt.rn.f16.f64 rs18, fd30;
}
{
neg.f16 rs19, rs18;
}
mov.b32 r1131, {rs19, rs19};
{
cvt.rn.f16.f64 rs21, fd41;
}
mov.b32 r1338, {rs21, rs21};
{
cvt.rn.f16.f64 rs22, fd24;
}
{
neg.f16 rs23, rs22;
}
mov.b32 r1362, {rs23, rs23};
{
cvt.rn.f16.f64 rs25, fd53;
}
mov.b32 r1320, {rs25, rs25};
{
cvt.rn.f16.f64 rs26, fd54;
}
{
neg.f16 rs27, rs26;
}
mov.b32 r1347, {rs27, rs27};
{
cvt.rn.f16.f64 rs29, fd53;
}
mov.b32 r1221, {rs29, rs29};
{
cvt.rn.f16.f64 rs30, fd54;
}
mov.b32 r1245, {rs30, rs30};
{
cvt.rn.f16.f64 rs31, fd35;
}
mov.b32 r1329, {rs31, rs31};
{
cvt.rn.f16.f64 rs32, fd30;
}
mov.b32 r1353, {rs32, rs32};
{
add.f16x2 r685, %50, %80;
}
{
add.f16x2 r688, %44, r685;
}
{
add.f16x2 r691, %56, %74;
}
{
add.f16x2 r694, r688, r691;
}
{
add.f16x2 r697, %62, %68;
}
{
add.f16x2 r700, r694, r697;
}
{
add.f16x2 r703, %51, %81;
}
{
add.f16x2 r706, %45, r703;
}
{
add.f16x2 r709, %57, %75;
}
{
add.f16x2 r712, r706, r709;
}
{
add.f16x2 r715, %63, %69;
}
{
add.f16x2 r718, r712, r715;
}
{
add.f16x2 r721, %50, %80;
}
{
mul.f16x2 r724, r721, r1104;
}
{
add.f16x2 r727, %44, r724;
}
{
add.f16x2 r730, %56, %74;
}
{
mul.f16x2 r733, r730, r1338;
}
{
add.f16x2 r736, r727, r733;
}
{
add.f16x2 r739, %62, %68;
}
{
mul.f16x2 r742, r739, r1320;
}
{
add.f16x2 r745, r736, r742;
}
{
sub.f16x2 r748, %51, %81;
}
{
mul.f16x2 r751, r748, r1131;
}
{
sub.f16x2 r754, %57, %75;
}
{
mul.f16x2 r757, r754, r1362;
}
{
add.f16x2 r760, r751, r757;
}
{
sub.f16x2 r763, %63, %69;
}
{
mul.f16x2 r766, r763, r1347;
}
{
add.f16x2 r769, r760, r766;
}
{
sub.f16x2 r772, r745, r769;
}
{
add.f16x2 r775, %50, %80;
}
{
mul.f16x2 r778, r775, r1104;
}
{
add.f16x2 r781, %44, r778;
}
{
add.f16x2 r784, %56, %74;
}
{
mul.f16x2 r787, r784, r1338;
}
{
add.f16x2 r790, r781, r787;
}
{
add.f16x2 r793, %62, %68;
}
{
mul.f16x2 r796, r793, r1320;
}
{
add.f16x2 r799, r790, r796;
}
{
sub.f16x2 r802, %51, %81;
}
{
mul.f16x2 r805, r802, r1131;
}
{
sub.f16x2 r808, %57, %75;
}
{
mul.f16x2 r811, r808, r1362;
}
{
add.f16x2 r814, r805, r811;
}
{
sub.f16x2 r817, %63, %69;
}
{
mul.f16x2 r820, r817, r1347;
}
{
add.f16x2 r823, r814, r820;
}
{
add.f16x2 r826, r799, r823;
}
{
add.f16x2 r829, %50, %80;
}
{
mul.f16x2 r832, r829, r1338;
}
{
add.f16x2 r835, %44, r832;
}
{
add.f16x2 r838, %56, %74;
}
{
mul.f16x2 r841, r838, r1221;
}
{
add.f16x2 r844, r835, r841;
}
{
add.f16x2 r847, %62, %68;
}
{
mul.f16x2 r850, r847, r1329;
}
{
add.f16x2 r853, r844, r850;
}
{
sub.f16x2 r856, %51, %81;
}
{
mul.f16x2 r859, r856, r1362;
}
{
sub.f16x2 r862, %57, %75;
}
{
mul.f16x2 r865, r862, r1245;
}
{
add.f16x2 r868, r859, r865;
}
{
sub.f16x2 r871, %63, %69;
}
{
mul.f16x2 r874, r871, r1353;
}
{
add.f16x2 r877, r868, r874;
}
{
sub.f16x2 r880, r853, r877;
}
{
add.f16x2 r883, %50, %80;
}
{
mul.f16x2 r886, r883, r1338;
}
{
add.f16x2 r889, %44, r886;
}
{
add.f16x2 r892, %56, %74;
}
{
mul.f16x2 r895, r892, r1221;
}
{
add.f16x2 r898, r889, r895;
}
{
add.f16x2 r901, %62, %68;
}
{
mul.f16x2 r904, r901, r1329;
}
{
add.f16x2 r907, r898, r904;
}
{
sub.f16x2 r910, %51, %81;
}
{
mul.f16x2 r913, r910, r1362;
}
{
sub.f16x2 r916, %57, %75;
}
{
mul.f16x2 r919, r916, r1245;
}
{
add.f16x2 r922, r913, r919;
}
{
sub.f16x2 r925, %63, %69;
}
{
mul.f16x2 r928, r925, r1353;
}
{
add.f16x2 r931, r922, r928;
}
{
add.f16x2 r934, r907, r931;
}
{
add.f16x2 r937, %50, %80;
}
{
mul.f16x2 r940, r937, r1320;
}
{
add.f16x2 r943, %44, r940;
}
{
add.f16x2 r946, %56, %74;
}
{
mul.f16x2 r949, r946, r1329;
}
{
add.f16x2 r952, r943, r949;
}
{
add.f16x2 r955, %62, %68;
}
{
mul.f16x2 r958, r955, r1338;
}
{
add.f16x2 r961, r952, r958;
}
{
sub.f16x2 r964, %51, %81;
}
{
mul.f16x2 r967, r964, r1347;
}
{
sub.f16x2 r970, %57, %75;
}
{
mul.f16x2 r973, r970, r1353;
}
{
add.f16x2 r976, r967, r973;
}
{
sub.f16x2 r979, %63, %69;
}
{
mul.f16x2 r982, r979, r1362;
}
{
add.f16x2 r985, r976, r982;
}
{
sub.f16x2 r988, r961, r985;
}
{
add.f16x2 r991, %50, %80;
}
{
mul.f16x2 r994, r991, r1320;
}
{
add.f16x2 r997, %44, r994;
}
{
add.f16x2 r1000, %56, %74;
}
{
mul.f16x2 r1003, r1000, r1329;
}
{
add.f16x2 r1006, r997, r1003;
}
{
add.f16x2 r1009, %62, %68;
}
{
mul.f16x2 r1012, r1009, r1338;
}
{
add.f16x2 r1015, r1006, r1012;
}
{
sub.f16x2 r1018, %51, %81;
}
{
mul.f16x2 r1021, r1018, r1347;
}
{
sub.f16x2 r1024, %57, %75;
}
{
mul.f16x2 r1027, r1024, r1353;
}
{
add.f16x2 r1030, r1021, r1027;
}
{
sub.f16x2 r1033, %63, %69;
}
{
mul.f16x2 r1036, r1033, r1362;
}
{
add.f16x2 r1039, r1030, r1036;
}
{
add.f16x2 r1042, r1015, r1039;
}
{
add.f16x2 r1045, %51, %81;
}
{
mul.f16x2 r1048, r1045, r1104;
}
{
add.f16x2 r1051, %45, r1048;
}
{
add.f16x2 r1054, %57, %75;
}
{
mul.f16x2 r1057, r1054, r1338;
}
{
add.f16x2 r1060, r1051, r1057;
}
{
add.f16x2 r1063, %63, %69;
}
{
mul.f16x2 r1066, r1063, r1320;
}
{
add.f16x2 r1069, r1060, r1066;
}
{
sub.f16x2 r1072, %50, %80;
}
{
mul.f16x2 r1075, r1072, r1131;
}
{
sub.f16x2 r1078, %56, %74;
}
{
mul.f16x2 r1081, r1078, r1362;
}
{
add.f16x2 r1084, r1075, r1081;
}
{
sub.f16x2 r1087, %62, %68;
}
{
mul.f16x2 r1090, r1087, r1347;
}
{
add.f16x2 r1093, r1084, r1090;
}
{
add.f16x2 r1096, r1069, r1093;
}
{
add.f16x2 r1099, %51, %81;
}
{
mul.f16x2 r1102, r1099, r1104;
}
{
add.f16x2 r1105, %45, r1102;
}
{
add.f16x2 r1108, %57, %75;
}
{
mul.f16x2 r1111, r1108, r1338;
}
{
add.f16x2 r1114, r1105, r1111;
}
{
add.f16x2 r1117, %63, %69;
}
{
mul.f16x2 r1120, r1117, r1320;
}
{
add.f16x2 r1123, r1114, r1120;
}
{
sub.f16x2 r1126, %50, %80;
}
{
mul.f16x2 r1129, r1126, r1131;
}
{
sub.f16x2 r1132, %56, %74;
}
{
mul.f16x2 r1135, r1132, r1362;
}
{
add.f16x2 r1138, r1129, r1135;
}
{
sub.f16x2 r1141, %62, %68;
}
{
mul.f16x2 r1144, r1141, r1347;
}
{
add.f16x2 r1147, r1138, r1144;
}
{
sub.f16x2 r1150, r1123, r1147;
}
{
add.f16x2 r1153, %51, %81;
}
{
mul.f16x2 r1156, r1153, r1338;
}
{
add.f16x2 r1159, %45, r1156;
}
{
add.f16x2 r1162, %57, %75;
}
{
mul.f16x2 r1165, r1162, r1221;
}
{
add.f16x2 r1168, r1159, r1165;
}
{
add.f16x2 r1171, %63, %69;
}
{
mul.f16x2 r1174, r1171, r1329;
}
{
add.f16x2 r1177, r1168, r1174;
}
{
sub.f16x2 r1180, %50, %80;
}
{
mul.f16x2 r1183, r1180, r1362;
}
{
sub.f16x2 r1186, %56, %74;
}
{
mul.f16x2 r1189, r1186, r1245;
}
{
add.f16x2 r1192, r1183, r1189;
}
{
sub.f16x2 r1195, %62, %68;
}
{
mul.f16x2 r1198, r1195, r1353;
}
{
add.f16x2 r1201, r1192, r1198;
}
{
add.f16x2 r1204, r1177, r1201;
}
{
add.f16x2 r1207, %51, %81;
}
{
mul.f16x2 r1210, r1207, r1338;
}
{
add.f16x2 r1213, %45, r1210;
}
{
add.f16x2 r1216, %57, %75;
}
{
mul.f16x2 r1219, r1216, r1221;
}
{
add.f16x2 r1222, r1213, r1219;
}
{
add.f16x2 r1225, %63, %69;
}
{
mul.f16x2 r1228, r1225, r1329;
}
{
add.f16x2 r1231, r1222, r1228;
}
{
sub.f16x2 r1234, %50, %80;
}
{
mul.f16x2 r1237, r1234, r1362;
}
{
sub.f16x2 r1240, %56, %74;
}
{
mul.f16x2 r1243, r1240, r1245;
}
{
add.f16x2 r1246, r1237, r1243;
}
{
sub.f16x2 r1249, %62, %68;
}
{
mul.f16x2 r1252, r1249, r1353;
}
{
add.f16x2 r1255, r1246, r1252;
}
{
sub.f16x2 r1258, r1231, r1255;
}
{
add.f16x2 r1261, %51, %81;
}
{
mul.f16x2 r1264, r1261, r1320;
}
{
add.f16x2 r1267, %45, r1264;
}
{
add.f16x2 r1270, %57, %75;
}
{
mul.f16x2 r1273, r1270, r1329;
}
{
add.f16x2 r1276, r1267, r1273;
}
{
add.f16x2 r1279, %63, %69;
}
{
mul.f16x2 r1282, r1279, r1338;
}
{
add.f16x2 r1285, r1276, r1282;
}
{
sub.f16x2 r1288, %50, %80;
}
{
mul.f16x2 r1291, r1288, r1347;
}
{
sub.f16x2 r1294, %56, %74;
}
{
mul.f16x2 r1297, r1294, r1353;
}
{
add.f16x2 r1300, r1291, r1297;
}
{
sub.f16x2 r1303, %62, %68;
}
{
mul.f16x2 r1306, r1303, r1362;
}
{
add.f16x2 r1309, r1300, r1306;
}
{
add.f16x2 r1312, r1285, r1309;
}
{
add.f16x2 r1315, %51, %81;
}
{
mul.f16x2 r1318, r1315, r1320;
}
{
add.f16x2 r1321, %45, r1318;
}
{
add.f16x2 r1324, %57, %75;
}
{
mul.f16x2 r1327, r1324, r1329;
}
{
add.f16x2 r1330, r1321, r1327;
}
{
add.f16x2 r1333, %63, %69;
}
{
mul.f16x2 r1336, r1333, r1338;
}
{
add.f16x2 r1339, r1330, r1336;
}
{
sub.f16x2 r1342, %50, %80;
}
{
mul.f16x2 r1345, r1342, r1347;
}
{
sub.f16x2 r1348, %56, %74;
}
{
mul.f16x2 r1351, r1348, r1353;
}
{
add.f16x2 r1354, r1345, r1351;
}
{
sub.f16x2 r1357, %62, %68;
}
{
mul.f16x2 r1360, r1357, r1362;
}
{
add.f16x2 r1363, r1354, r1360;
}
{
sub.f16x2 r1366, r1339, r1363;
}
{
cvt.rn.f16.f64 rs33, fd35;
}
mov.b32 r1788, {rs33, rs33};
{
cvt.rn.f16.f64 rs34, fd30;
}
{
neg.f16 rs35, rs34;
}
mov.b32 r1815, {rs35, rs35};
{
cvt.rn.f16.f64 rs37, fd41;
}
mov.b32 r2022, {rs37, rs37};
{
cvt.rn.f16.f64 rs38, fd24;
}
{
neg.f16 rs39, rs38;
}
mov.b32 r2046, {rs39, rs39};
{
cvt.rn.f16.f64 rs41, fd53;
}
mov.b32 r2004, {rs41, rs41};
{
cvt.rn.f16.f64 rs42, fd54;
}
{
neg.f16 rs43, rs42;
}
mov.b32 r2031, {rs43, rs43};
{
cvt.rn.f16.f64 rs45, fd53;
}
mov.b32 r1905, {rs45, rs45};
{
cvt.rn.f16.f64 rs46, fd54;
}
mov.b32 r1929, {rs46, rs46};
{
cvt.rn.f16.f64 rs47, fd35;
}
mov.b32 r2013, {rs47, rs47};
{
cvt.rn.f16.f64 rs48, fd30;
}
mov.b32 r2037, {rs48, rs48};
{
add.f16x2 r1369, %52, %82;
}
{
add.f16x2 r1372, %46, r1369;
}
{
add.f16x2 r1375, %58, %76;
}
{
add.f16x2 r1378, r1372, r1375;
}
{
add.f16x2 r1381, %64, %70;
}
{
add.f16x2 r1384, r1378, r1381;
}
{
add.f16x2 r1387, %53, %83;
}
{
add.f16x2 r1390, %47, r1387;
}
{
add.f16x2 r1393, %59, %77;
}
{
add.f16x2 r1396, r1390, r1393;
}
{
add.f16x2 r1399, %65, %71;
}
{
add.f16x2 r1402, r1396, r1399;
}
{
add.f16x2 r1405, %52, %82;
}
{
mul.f16x2 r1408, r1405, r1788;
}
{
add.f16x2 r1411, %46, r1408;
}
{
add.f16x2 r1414, %58, %76;
}
{
mul.f16x2 r1417, r1414, r2022;
}
{
add.f16x2 r1420, r1411, r1417;
}
{
add.f16x2 r1423, %64, %70;
}
{
mul.f16x2 r1426, r1423, r2004;
}
{
add.f16x2 r1429, r1420, r1426;
}
{
sub.f16x2 r1432, %53, %83;
}
{
mul.f16x2 r1435, r1432, r1815;
}
{
sub.f16x2 r1438, %59, %77;
}
{
mul.f16x2 r1441, r1438, r2046;
}
{
add.f16x2 r1444, r1435, r1441;
}
{
sub.f16x2 r1447, %65, %71;
}
{
mul.f16x2 r1450, r1447, r2031;
}
{
add.f16x2 r1453, r1444, r1450;
}
{
sub.f16x2 r1456, r1429, r1453;
}
{
add.f16x2 r1459, %52, %82;
}
{
mul.f16x2 r1462, r1459, r1788;
}
{
add.f16x2 r1465, %46, r1462;
}
{
add.f16x2 r1468, %58, %76;
}
{
mul.f16x2 r1471, r1468, r2022;
}
{
add.f16x2 r1474, r1465, r1471;
}
{
add.f16x2 r1477, %64, %70;
}
{
mul.f16x2 r1480, r1477, r2004;
}
{
add.f16x2 r1483, r1474, r1480;
}
{
sub.f16x2 r1486, %53, %83;
}
{
mul.f16x2 r1489, r1486, r1815;
}
{
sub.f16x2 r1492, %59, %77;
}
{
mul.f16x2 r1495, r1492, r2046;
}
{
add.f16x2 r1498, r1489, r1495;
}
{
sub.f16x2 r1501, %65, %71;
}
{
mul.f16x2 r1504, r1501, r2031;
}
{
add.f16x2 r1507, r1498, r1504;
}
{
add.f16x2 r1510, r1483, r1507;
}
{
add.f16x2 r1513, %52, %82;
}
{
mul.f16x2 r1516, r1513, r2022;
}
{
add.f16x2 r1519, %46, r1516;
}
{
add.f16x2 r1522, %58, %76;
}
{
mul.f16x2 r1525, r1522, r1905;
}
{
add.f16x2 r1528, r1519, r1525;
}
{
add.f16x2 r1531, %64, %70;
}
{
mul.f16x2 r1534, r1531, r2013;
}
{
add.f16x2 r1537, r1528, r1534;
}
{
sub.f16x2 r1540, %53, %83;
}
{
mul.f16x2 r1543, r1540, r2046;
}
{
sub.f16x2 r1546, %59, %77;
}
{
mul.f16x2 r1549, r1546, r1929;
}
{
add.f16x2 r1552, r1543, r1549;
}
{
sub.f16x2 r1555, %65, %71;
}
{
mul.f16x2 r1558, r1555, r2037;
}
{
add.f16x2 r1561, r1552, r1558;
}
{
sub.f16x2 r1564, r1537, r1561;
}
{
add.f16x2 r1567, %52, %82;
}
{
mul.f16x2 r1570, r1567, r2022;
}
{
add.f16x2 r1573, %46, r1570;
}
{
add.f16x2 r1576, %58, %76;
}
{
mul.f16x2 r1579, r1576, r1905;
}
{
add.f16x2 r1582, r1573, r1579;
}
{
add.f16x2 r1585, %64, %70;
}
{
mul.f16x2 r1588, r1585, r2013;
}
{
add.f16x2 r1591, r1582, r1588;
}
{
sub.f16x2 r1594, %53, %83;
}
{
mul.f16x2 r1597, r1594, r2046;
}
{
sub.f16x2 r1600, %59, %77;
}
{
mul.f16x2 r1603, r1600, r1929;
}
{
add.f16x2 r1606, r1597, r1603;
}
{
sub.f16x2 r1609, %65, %71;
}
{
mul.f16x2 r1612, r1609, r2037;
}
{
add.f16x2 r1615, r1606, r1612;
}
{
add.f16x2 r1618, r1591, r1615;
}
{
add.f16x2 r1621, %52, %82;
}
{
mul.f16x2 r1624, r1621, r2004;
}
{
add.f16x2 r1627, %46, r1624;
}
{
add.f16x2 r1630, %58, %76;
}
{
mul.f16x2 r1633, r1630, r2013;
}
{
add.f16x2 r1636, r1627, r1633;
}
{
add.f16x2 r1639, %64, %70;
}
{
mul.f16x2 r1642, r1639, r2022;
}
{
add.f16x2 r1645, r1636, r1642;
}
{
sub.f16x2 r1648, %53, %83;
}
{
mul.f16x2 r1651, r1648, r2031;
}
{
sub.f16x2 r1654, %59, %77;
}
{
mul.f16x2 r1657, r1654, r2037;
}
{
add.f16x2 r1660, r1651, r1657;
}
{
sub.f16x2 r1663, %65, %71;
}
{
mul.f16x2 r1666, r1663, r2046;
}
{
add.f16x2 r1669, r1660, r1666;
}
{
sub.f16x2 r1672, r1645, r1669;
}
{
add.f16x2 r1675, %52, %82;
}
{
mul.f16x2 r1678, r1675, r2004;
}
{
add.f16x2 r1681, %46, r1678;
}
{
add.f16x2 r1684, %58, %76;
}
{
mul.f16x2 r1687, r1684, r2013;
}
{
add.f16x2 r1690, r1681, r1687;
}
{
add.f16x2 r1693, %64, %70;
}
{
mul.f16x2 r1696, r1693, r2022;
}
{
add.f16x2 r1699, r1690, r1696;
}
{
sub.f16x2 r1702, %53, %83;
}
{
mul.f16x2 r1705, r1702, r2031;
}
{
sub.f16x2 r1708, %59, %77;
}
{
mul.f16x2 r1711, r1708, r2037;
}
{
add.f16x2 r1714, r1705, r1711;
}
{
sub.f16x2 r1717, %65, %71;
}
{
mul.f16x2 r1720, r1717, r2046;
}
{
add.f16x2 r1723, r1714, r1720;
}
{
add.f16x2 r1726, r1699, r1723;
}
{
add.f16x2 r1729, %53, %83;
}
{
mul.f16x2 r1732, r1729, r1788;
}
{
add.f16x2 r1735, %47, r1732;
}
{
add.f16x2 r1738, %59, %77;
}
{
mul.f16x2 r1741, r1738, r2022;
}
{
add.f16x2 r1744, r1735, r1741;
}
{
add.f16x2 r1747, %65, %71;
}
{
mul.f16x2 r1750, r1747, r2004;
}
{
add.f16x2 r1753, r1744, r1750;
}
{
sub.f16x2 r1756, %52, %82;
}
{
mul.f16x2 r1759, r1756, r1815;
}
{
sub.f16x2 r1762, %58, %76;
}
{
mul.f16x2 r1765, r1762, r2046;
}
{
add.f16x2 r1768, r1759, r1765;
}
{
sub.f16x2 r1771, %64, %70;
}
{
mul.f16x2 r1774, r1771, r2031;
}
{
add.f16x2 r1777, r1768, r1774;
}
{
add.f16x2 r1780, r1753, r1777;
}
{
add.f16x2 r1783, %53, %83;
}
{
mul.f16x2 r1786, r1783, r1788;
}
{
add.f16x2 r1789, %47, r1786;
}
{
add.f16x2 r1792, %59, %77;
}
{
mul.f16x2 r1795, r1792, r2022;
}
{
add.f16x2 r1798, r1789, r1795;
}
{
add.f16x2 r1801, %65, %71;
}
{
mul.f16x2 r1804, r1801, r2004;
}
{
add.f16x2 r1807, r1798, r1804;
}
{
sub.f16x2 r1810, %52, %82;
}
{
mul.f16x2 r1813, r1810, r1815;
}
{
sub.f16x2 r1816, %58, %76;
}
{
mul.f16x2 r1819, r1816, r2046;
}
{
add.f16x2 r1822, r1813, r1819;
}
{
sub.f16x2 r1825, %64, %70;
}
{
mul.f16x2 r1828, r1825, r2031;
}
{
add.f16x2 r1831, r1822, r1828;
}
{
sub.f16x2 r1834, r1807, r1831;
}
{
add.f16x2 r1837, %53, %83;
}
{
mul.f16x2 r1840, r1837, r2022;
}
{
add.f16x2 r1843, %47, r1840;
}
{
add.f16x2 r1846, %59, %77;
}
{
mul.f16x2 r1849, r1846, r1905;
}
{
add.f16x2 r1852, r1843, r1849;
}
{
add.f16x2 r1855, %65, %71;
}
{
mul.f16x2 r1858, r1855, r2013;
}
{
add.f16x2 r1861, r1852, r1858;
}
{
sub.f16x2 r1864, %52, %82;
}
{
mul.f16x2 r1867, r1864, r2046;
}
{
sub.f16x2 r1870, %58, %76;
}
{
mul.f16x2 r1873, r1870, r1929;
}
{
add.f16x2 r1876, r1867, r1873;
}
{
sub.f16x2 r1879, %64, %70;
}
{
mul.f16x2 r1882, r1879, r2037;
}
{
add.f16x2 r1885, r1876, r1882;
}
{
add.f16x2 r1888, r1861, r1885;
}
{
add.f16x2 r1891, %53, %83;
}
{
mul.f16x2 r1894, r1891, r2022;
}
{
add.f16x2 r1897, %47, r1894;
}
{
add.f16x2 r1900, %59, %77;
}
{
mul.f16x2 r1903, r1900, r1905;
}
{
add.f16x2 r1906, r1897, r1903;
}
{
add.f16x2 r1909, %65, %71;
}
{
mul.f16x2 r1912, r1909, r2013;
}
{
add.f16x2 r1915, r1906, r1912;
}
{
sub.f16x2 r1918, %52, %82;
}
{
mul.f16x2 r1921, r1918, r2046;
}
{
sub.f16x2 r1924, %58, %76;
}
{
mul.f16x2 r1927, r1924, r1929;
}
{
add.f16x2 r1930, r1921, r1927;
}
{
sub.f16x2 r1933, %64, %70;
}
{
mul.f16x2 r1936, r1933, r2037;
}
{
add.f16x2 r1939, r1930, r1936;
}
{
sub.f16x2 r1942, r1915, r1939;
}
{
add.f16x2 r1945, %53, %83;
}
{
mul.f16x2 r1948, r1945, r2004;
}
{
add.f16x2 r1951, %47, r1948;
}
{
add.f16x2 r1954, %59, %77;
}
{
mul.f16x2 r1957, r1954, r2013;
}
{
add.f16x2 r1960, r1951, r1957;
}
{
add.f16x2 r1963, %65, %71;
}
{
mul.f16x2 r1966, r1963, r2022;
}
{
add.f16x2 r1969, r1960, r1966;
}
{
sub.f16x2 r1972, %52, %82;
}
{
mul.f16x2 r1975, r1972, r2031;
}
{
sub.f16x2 r1978, %58, %76;
}
{
mul.f16x2 r1981, r1978, r2037;
}
{
add.f16x2 r1984, r1975, r1981;
}
{
sub.f16x2 r1987, %64, %70;
}
{
mul.f16x2 r1990, r1987, r2046;
}
{
add.f16x2 r1993, r1984, r1990;
}
{
add.f16x2 r1996, r1969, r1993;
}
{
add.f16x2 r1999, %53, %83;
}
{
mul.f16x2 r2002, r1999, r2004;
}
{
add.f16x2 r2005, %47, r2002;
}
{
add.f16x2 r2008, %59, %77;
}
{
mul.f16x2 r2011, r2008, r2013;
}
{
add.f16x2 r2014, r2005, r2011;
}
{
add.f16x2 r2017, %65, %71;
}
{
mul.f16x2 r2020, r2017, r2022;
}
{
add.f16x2 r2023, r2014, r2020;
}
{
sub.f16x2 r2026, %52, %82;
}
{
mul.f16x2 r2029, r2026, r2031;
}
{
sub.f16x2 r2032, %58, %76;
}
{
mul.f16x2 r2035, r2032, r2037;
}
{
add.f16x2 r2038, r2029, r2035;
}
{
sub.f16x2 r2041, %64, %70;
}
{
mul.f16x2 r2044, r2041, r2046;
}
{
add.f16x2 r2047, r2038, r2044;
}
{
sub.f16x2 r2050, r2023, r2047;
}
mov.f64 fd31, 0d3FEE940D6BB98CC5;
{
cvt.rn.f16.f64 rs49, fd31;
}
mov.f64 fd32, 0d3FD2DD44CE9AFBA7;
{
cvt.rn.f16.f64 rs50, fd32;
}
mov.f64 fd33, 0d3FEA708C4C4BFA74;
{
cvt.rn.f16.f64 rs51, fd33;
}
mov.f64 fd34, 0d3FE206B7C9520CED;
{
cvt.rn.f16.f64 rs52, fd34;
}
{
cvt.rn.f16.f64 rs53, fd35;
}
mov.f64 fd36, 0d3FE904C37505DE4B;
{
cvt.rn.f16.f64 rs54, fd36;
}
mov.f64 fd37, 0d3FD761BF51E29C90;
{
cvt.rn.f16.f64 rs55, fd37;
}
mov.f64 fd38, 0d3FEDC9B7BE64378E;
{
cvt.rn.f16.f64 rs56, fd38;
}
mov.f64 fd39, 0d3FB32182EBFB0FE9;
{
cvt.rn.f16.f64 rs57, fd39;
}
mov.f64 fd40, 0d3FEFE917F00AE2CD;
{
cvt.rn.f16.f64 rs58, fd40;
}
{
cvt.rn.f16.f64 rs59, fd41;
}
mov.f64 fd42, 0d3FEF329C0558E969;
{
cvt.rn.f16.f64 rs60, fd42;
}
mov.f64 fd83, 0dBFE0000000000000;
mov.f64 fd45, 0dBFE7752932F8FB65;
{
cvt.rn.f16.f64 rs63, fd45;
}
mov.f64 fd46, 0d3FE5C3F99E0B6B95;
{
cvt.rn.f16.f64 rs64, fd46;
}
mov.f64 fd49, 0dBFEFA4808B7D3C19;
{
cvt.rn.f16.f64 rs67, fd49;
}
mov.f64 fd50, 0d3FC313D12579650C;
{
cvt.rn.f16.f64 rs68, fd50;
}
{
cvt.rn.f16.f64 rs71, fd53;
}
{
cvt.rn.f16.f64 rs72, fd54;
}
mov.f64 fd84, 0dBFEBB67AE8584CAA;
mov.b32 r2067, {rs49, rs49};
{
mul.f16x2 r2053, r772, r2067;
}
mov.b32 r2064, {rs50, rs50};
{
mul.f16x2 r2056, r1096, r2064;
}
{
sub.f16x2 r2059, r2053, r2056;
}
{
mul.f16x2 r2062, r772, r2064;
}
{
fma.rn.f16x2 r2065, r1096, r2067, r2062;
}
mov.b32 r2099, {rs51, rs51};
{
mul.f16x2 r2069, r1456, r2099;
}
mov.b32 r2096, {rs52, rs52};
{
mul.f16x2 r2072, r1780, r2096;
}
{
sub.f16x2 r2075, r2069, r2072;
}
{
mul.f16x2 r2078, r1456, r2096;
}
{
fma.rn.f16x2 r2081, r1780, r2099, r2078;
}
{
mul.f16x2 r2085, r880, r2099;
}
{
mul.f16x2 r2088, r1204, r2096;
}
{
sub.f16x2 r2091, r2085, r2088;
}
{
mul.f16x2 r2094, r880, r2096;
}
{
fma.rn.f16x2 r2097, r1204, r2099, r2094;
}
mov.b32 r2163, {rs55, rs55};
{
mul.f16x2 r2101, r1564, r2163;
}
mov.b32 r2160, {rs56, rs56};
{
mul.f16x2 r2104, r1888, r2160;
}
{
sub.f16x2 r2107, r2101, r2104;
}
{
mul.f16x2 r2110, r1564, r2160;
}
{
fma.rn.f16x2 r2113, r1888, r2163, r2110;
}
mov.b32 r2131, {rs53, rs53};
{
mul.f16x2 r2117, r988, r2131;
}
mov.b32 r2128, {rs54, rs54};
{
mul.f16x2 r2120, r1312, r2128;
}
{
sub.f16x2 r2123, r2117, r2120;
}
{
mul.f16x2 r2126, r988, r2128;
}
{
fma.rn.f16x2 r2129, r1312, r2131, r2126;
}
mov.b32 r2227, {rs59, rs59};
{
mul.f16x2 r2133, r1672, r2227;
}
mov.b32 r2224, {rs60, rs60};
{
mul.f16x2 r2136, r1996, r2224;
}
{
sub.f16x2 r2139, r2133, r2136;
}
{
mul.f16x2 r2142, r1672, r2224;
}
{
fma.rn.f16x2 r2145, r1996, r2227, r2142;
}
{
mul.f16x2 r2149, r1042, r2163;
}
{
mul.f16x2 r2152, r1366, r2160;
}
{
sub.f16x2 r2155, r2149, r2152;
}
{
mul.f16x2 r2158, r1042, r2160;
}
{
fma.rn.f16x2 r2161, r1366, r2163, r2158;
}
mov.b32 r2179, {rs63, rs63};
{
mul.f16x2 r2165, r1726, r2179;
}
mov.b32 r2176, {rs64, rs64};
{
mul.f16x2 r2168, r2050, r2176;
}
{
sub.f16x2 r2171, r2165, r2168;
}
{
mul.f16x2 r2174, r1726, r2176;
}
{
fma.rn.f16x2 r2177, r2050, r2179, r2174;
}
mov.b32 r2195, {rs57, rs57};
{
mul.f16x2 r2181, r934, r2195;
}
mov.b32 r2192, {rs58, rs58};
{
mul.f16x2 r2184, r1258, r2192;
}
{
sub.f16x2 r2187, r2181, r2184;
}
{
mul.f16x2 r2190, r934, r2192;
}
{
fma.rn.f16x2 r2193, r1258, r2195, r2190;
}
mov.b32 r2211, {rs67, rs67};
{
mul.f16x2 r2197, r1618, r2211;
}
mov.b32 r2208, {rs68, rs68};
{
mul.f16x2 r2200, r1942, r2208;
}
{
sub.f16x2 r2203, r2197, r2200;
}
{
mul.f16x2 r2206, r1618, r2208;
}
{
fma.rn.f16x2 r2209, r1942, r2211, r2206;
}
{
mul.f16x2 r2213, r826, r2227;
}
{
mul.f16x2 r2216, r1150, r2224;
}
{
sub.f16x2 r2219, r2213, r2216;
}
{
mul.f16x2 r2222, r826, r2224;
}
{
fma.rn.f16x2 r2225, r1150, r2227, r2222;
}
mov.b32 r2243, {rs71, rs71};
{
mul.f16x2 r2229, r1510, r2243;
}
mov.b32 r2240, {rs72, rs72};
{
mul.f16x2 r2232, r1834, r2240;
}
{
sub.f16x2 r2235, r2229, r2232;
}
{
mul.f16x2 r2238, r1510, r2240;
}
{
fma.rn.f16x2 r2241, r1834, r2243, r2238;
}
{
cvt.rn.f16.f64 rs89, fd83;
}
mov.b32 r2316, {rs89, rs89};
{
cvt.rn.f16.f64 rs90, fd84;
}
mov.b32 r2325, {rs90, rs90};
{
add.f16x2 r2245, r700, r1384;
}
{
add.f16x2 %0, r16, r2245;
}
{
add.f16x2 r2251, r718, r1402;
}
{
add.f16x2 %1, r34, r2251;
}
{
add.f16x2 r2257, r700, r1384;
}
{
mul.f16x2 r2260, r2257, r2316;
}
{
add.f16x2 r2263, r16, r2260;
}
{
sub.f16x2 r2266, r718, r1402;
}
{
mul.f16x2 r2269, r2266, r2325;
}
{
add.f16x2 %14, r2263, r2269;
}
{
add.f16x2 r2275, r700, r1384;
}
{
mul.f16x2 r2278, r2275, r2316;
}
{
add.f16x2 r2281, r16, r2278;
}
{
sub.f16x2 r2284, r718, r1402;
}
{
mul.f16x2 r2287, r2284, r2325;
}
{
sub.f16x2 %28, r2281, r2287;
}
{
add.f16x2 r2293, r718, r1402;
}
{
mul.f16x2 r2296, r2293, r2316;
}
{
add.f16x2 r2299, r34, r2296;
}
{
sub.f16x2 r2302, r700, r1384;
}
{
mul.f16x2 r2305, r2302, r2325;
}
{
sub.f16x2 %15, r2299, r2305;
}
{
add.f16x2 r2311, r718, r1402;
}
{
mul.f16x2 r2314, r2311, r2316;
}
{
add.f16x2 r2317, r34, r2314;
}
{
sub.f16x2 r2320, r700, r1384;
}
{
mul.f16x2 r2323, r2320, r2325;
}
{
add.f16x2 %29, r2317, r2323;
}
{
cvt.rn.f16.f64 rs91, fd83;
}
mov.b32 r2400, {rs91, rs91};
{
cvt.rn.f16.f64 rs92, fd84;
}
mov.b32 r2409, {rs92, rs92};
{
add.f16x2 r2329, r2059, r2075;
}
{
add.f16x2 %2, r88, r2329;
}
{
add.f16x2 r2335, r2065, r2081;
}
{
add.f16x2 %3, r412, r2335;
}
{
add.f16x2 r2341, r2059, r2075;
}
{
mul.f16x2 r2344, r2341, r2400;
}
{
add.f16x2 r2347, r88, r2344;
}
{
sub.f16x2 r2350, r2065, r2081;
}
{
mul.f16x2 r2353, r2350, r2409;
}
{
add.f16x2 %16, r2347, r2353;
}
{
add.f16x2 r2359, r2059, r2075;
}
{
mul.f16x2 r2362, r2359, r2400;
}
{
add.f16x2 r2365, r88, r2362;
}
{
sub.f16x2 r2368, r2065, r2081;
}
{
mul.f16x2 r2371, r2368, r2409;
}
{
sub.f16x2 %30, r2365, r2371;
}
{
add.f16x2 r2377, r2065, r2081;
}
{
mul.f16x2 r2380, r2377, r2400;
}
{
add.f16x2 r2383, r412, r2380;
}
{
sub.f16x2 r2386, r2059, r2075;
}
{
mul.f16x2 r2389, r2386, r2409;
}
{
sub.f16x2 %17, r2383, r2389;
}
{
add.f16x2 r2395, r2065, r2081;
}
{
mul.f16x2 r2398, r2395, r2400;
}
{
add.f16x2 r2401, r412, r2398;
}
{
sub.f16x2 r2404, r2059, r2075;
}
{
mul.f16x2 r2407, r2404, r2409;
}
{
add.f16x2 %31, r2401, r2407;
}
{
cvt.rn.f16.f64 rs93, fd83;
}
mov.b32 r2484, {rs93, rs93};
{
cvt.rn.f16.f64 rs94, fd84;
}
mov.b32 r2493, {rs94, rs94};
{
add.f16x2 r2413, r2091, r2107;
}
{
add.f16x2 %4, r196, r2413;
}
{
add.f16x2 r2419, r2097, r2113;
}
{
add.f16x2 %5, r520, r2419;
}
{
add.f16x2 r2425, r2091, r2107;
}
{
mul.f16x2 r2428, r2425, r2484;
}
{
add.f16x2 r2431, r196, r2428;
}
{
sub.f16x2 r2434, r2097, r2113;
}
{
mul.f16x2 r2437, r2434, r2493;
}
{
add.f16x2 %18, r2431, r2437;
}
{
add.f16x2 r2443, r2091, r2107;
}
{
mul.f16x2 r2446, r2443, r2484;
}
{
add.f16x2 r2449, r196, r2446;
}
{
sub.f16x2 r2452, r2097, r2113;
}
{
mul.f16x2 r2455, r2452, r2493;
}
{
sub.f16x2 %32, r2449, r2455;
}
{
add.f16x2 r2461, r2097, r2113;
}
{
mul.f16x2 r2464, r2461, r2484;
}
{
add.f16x2 r2467, r520, r2464;
}
{
sub.f16x2 r2470, r2091, r2107;
}
{
mul.f16x2 r2473, r2470, r2493;
}
{
sub.f16x2 %19, r2467, r2473;
}
{
add.f16x2 r2479, r2097, r2113;
}
{
mul.f16x2 r2482, r2479, r2484;
}
{
add.f16x2 r2485, r520, r2482;
}
{
sub.f16x2 r2488, r2091, r2107;
}
{
mul.f16x2 r2491, r2488, r2493;
}
{
add.f16x2 %33, r2485, r2491;
}
{
cvt.rn.f16.f64 rs95, fd83;
}
mov.b32 r2568, {rs95, rs95};
{
cvt.rn.f16.f64 rs96, fd84;
}
mov.b32 r2577, {rs96, rs96};
{
add.f16x2 r2497, r2123, r2139;
}
{
add.f16x2 %6, r304, r2497;
}
{
add.f16x2 r2503, r2129, r2145;
}
{
add.f16x2 %7, r628, r2503;
}
{
add.f16x2 r2509, r2123, r2139;
}
{
mul.f16x2 r2512, r2509, r2568;
}
{
add.f16x2 r2515, r304, r2512;
}
{
sub.f16x2 r2518, r2129, r2145;
}
{
mul.f16x2 r2521, r2518, r2577;
}
{
add.f16x2 %20, r2515, r2521;
}
{
add.f16x2 r2527, r2123, r2139;
}
{
mul.f16x2 r2530, r2527, r2568;
}
{
add.f16x2 r2533, r304, r2530;
}
{
sub.f16x2 r2536, r2129, r2145;
}
{
mul.f16x2 r2539, r2536, r2577;
}
{
sub.f16x2 %34, r2533, r2539;
}
{
add.f16x2 r2545, r2129, r2145;
}
{
mul.f16x2 r2548, r2545, r2568;
}
{
add.f16x2 r2551, r628, r2548;
}
{
sub.f16x2 r2554, r2123, r2139;
}
{
mul.f16x2 r2557, r2554, r2577;
}
{
sub.f16x2 %21, r2551, r2557;
}
{
add.f16x2 r2563, r2129, r2145;
}
{
mul.f16x2 r2566, r2563, r2568;
}
{
add.f16x2 r2569, r628, r2566;
}
{
sub.f16x2 r2572, r2123, r2139;
}
{
mul.f16x2 r2575, r2572, r2577;
}
{
add.f16x2 %35, r2569, r2575;
}
{
cvt.rn.f16.f64 rs97, fd83;
}
mov.b32 r2652, {rs97, rs97};
{
cvt.rn.f16.f64 rs98, fd84;
}
mov.b32 r2661, {rs98, rs98};
{
add.f16x2 r2581, r2155, r2171;
}
{
add.f16x2 %8, r358, r2581;
}
{
add.f16x2 r2587, r2161, r2177;
}
{
add.f16x2 %9, r682, r2587;
}
{
add.f16x2 r2593, r2155, r2171;
}
{
mul.f16x2 r2596, r2593, r2652;
}
{
add.f16x2 r2599, r358, r2596;
}
{
sub.f16x2 r2602, r2161, r2177;
}
{
mul.f16x2 r2605, r2602, r2661;
}
{
add.f16x2 %22, r2599, r2605;
}
{
add.f16x2 r2611, r2155, r2171;
}
{
mul.f16x2 r2614, r2611, r2652;
}
{
add.f16x2 r2617, r358, r2614;
}
{
sub.f16x2 r2620, r2161, r2177;
}
{
mul.f16x2 r2623, r2620, r2661;
}
{
sub.f16x2 %36, r2617, r2623;
}
{
add.f16x2 r2629, r2161, r2177;
}
{
mul.f16x2 r2632, r2629, r2652;
}
{
add.f16x2 r2635, r682, r2632;
}
{
sub.f16x2 r2638, r2155, r2171;
}
{
mul.f16x2 r2641, r2638, r2661;
}
{
sub.f16x2 %23, r2635, r2641;
}
{
add.f16x2 r2647, r2161, r2177;
}
{
mul.f16x2 r2650, r2647, r2652;
}
{
add.f16x2 r2653, r682, r2650;
}
{
sub.f16x2 r2656, r2155, r2171;
}
{
mul.f16x2 r2659, r2656, r2661;
}
{
add.f16x2 %37, r2653, r2659;
}
{
cvt.rn.f16.f64 rs99, fd83;
}
mov.b32 r2736, {rs99, rs99};
{
cvt.rn.f16.f64 rs100, fd84;
}
mov.b32 r2745, {rs100, rs100};
{
add.f16x2 r2665, r2187, r2203;
}
{
add.f16x2 %10, r250, r2665;
}
{
add.f16x2 r2671, r2193, r2209;
}
{
add.f16x2 %11, r574, r2671;
}
{
add.f16x2 r2677, r2187, r2203;
}
{
mul.f16x2 r2680, r2677, r2736;
}
{
add.f16x2 r2683, r250, r2680;
}
{
sub.f16x2 r2686, r2193, r2209;
}
{
mul.f16x2 r2689, r2686, r2745;
}
{
add.f16x2 %24, r2683, r2689;
}
{
add.f16x2 r2695, r2187, r2203;
}
{
mul.f16x2 r2698, r2695, r2736;
}
{
add.f16x2 r2701, r250, r2698;
}
{
sub.f16x2 r2704, r2193, r2209;
}
{
mul.f16x2 r2707, r2704, r2745;
}
{
sub.f16x2 %38, r2701, r2707;
}
{
add.f16x2 r2713, r2193, r2209;
}
{
mul.f16x2 r2716, r2713, r2736;
}
{
add.f16x2 r2719, r574, r2716;
}
{
sub.f16x2 r2722, r2187, r2203;
}
{
mul.f16x2 r2725, r2722, r2745;
}
{
sub.f16x2 %25, r2719, r2725;
}
{
add.f16x2 r2731, r2193, r2209;
}
{
mul.f16x2 r2734, r2731, r2736;
}
{
add.f16x2 r2737, r574, r2734;
}
{
sub.f16x2 r2740, r2187, r2203;
}
{
mul.f16x2 r2743, r2740, r2745;
}
{
add.f16x2 %39, r2737, r2743;
}
{
cvt.rn.f16.f64 rs101, fd83;
}
mov.b32 r2820, {rs101, rs101};
{
cvt.rn.f16.f64 rs102, fd84;
}
mov.b32 r2829, {rs102, rs102};
{
add.f16x2 r2749, r2219, r2235;
}
{
add.f16x2 %12, r142, r2749;
}
{
add.f16x2 r2755, r2225, r2241;
}
{
add.f16x2 %13, r466, r2755;
}
{
add.f16x2 r2761, r2219, r2235;
}
{
mul.f16x2 r2764, r2761, r2820;
}
{
add.f16x2 r2767, r142, r2764;
}
{
sub.f16x2 r2770, r2225, r2241;
}
{
mul.f16x2 r2773, r2770, r2829;
}
{
add.f16x2 %26, r2767, r2773;
}
{
add.f16x2 r2779, r2219, r2235;
}
{
mul.f16x2 r2782, r2779, r2820;
}
{
add.f16x2 r2785, r142, r2782;
}
{
sub.f16x2 r2788, r2225, r2241;
}
{
mul.f16x2 r2791, r2788, r2829;
}
{
sub.f16x2 %40, r2785, r2791;
}
{
add.f16x2 r2797, r2225, r2241;
}
{
mul.f16x2 r2800, r2797, r2820;
}
{
add.f16x2 r2803, r466, r2800;
}
{
sub.f16x2 r2806, r2219, r2235;
}
{
mul.f16x2 r2809, r2806, r2829;
}
{
sub.f16x2 %27, r2803, r2809;
}
{
add.f16x2 r2815, r2225, r2241;
}
{
mul.f16x2 r2818, r2815, r2820;
}
{
add.f16x2 r2821, r466, r2818;
}
{
sub.f16x2 r2824, r2219, r2235;
}
{
mul.f16x2 r2827, r2824, r2829;
}
{
add.f16x2 %41, r2821, r2827;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[20].y)));
};


#endif
