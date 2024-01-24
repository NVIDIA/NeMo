#ifndef CUFFTDX_FFT_14_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_14_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<949, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<59>;
.reg .b32 r<1549>;
.reg .f64 fd<47>;
.reg .b64 rd<2>;
mov.f64 fd23, 0d3FE3F3A0E28BEDD1;
{
cvt.rn.f16.f64 rs1, fd23;
}
mov.b32 r420, {rs1, rs1};
mov.f64 fd20, 0dBFE904C37505DE4B;
{
cvt.rn.f16.f64 rs2, fd20;
}
{
neg.f16 rs3, rs2;
}
mov.b32 r447, {rs3, rs3};
mov.f64 fd27, 0dBFCC7B90E3024582;
{
cvt.rn.f16.f64 rs5, fd27;
}
mov.b32 r654, {rs5, rs5};
mov.f64 fd14, 0dBFEF329C0558E969;
{
cvt.rn.f16.f64 rs6, fd14;
}
{
neg.f16 rs7, rs6;
}
mov.b32 r678, {rs7, rs7};
mov.f64 fd31, 0dBFECD4BCA9CB5C71;
{
cvt.rn.f16.f64 rs9, fd31;
}
mov.b32 r636, {rs9, rs9};
mov.f64 fd18, 0dBFDBC4C04D71ABC1;
{
cvt.rn.f16.f64 rs10, fd18;
}
{
neg.f16 rs11, rs10;
}
mov.b32 r663, {rs11, rs11};
{
cvt.rn.f16.f64 rs13, fd31;
}
mov.b32 r537, {rs13, rs13};
{
cvt.rn.f16.f64 rs14, fd18;
}
mov.b32 r561, {rs14, rs14};
{
cvt.rn.f16.f64 rs15, fd23;
}
mov.b32 r645, {rs15, rs15};
{
cvt.rn.f16.f64 rs16, fd20;
}
mov.b32 r669, {rs16, rs16};
{
add.f16x2 r1, %32, %52;
}
{
add.f16x2 r4, %28, r1;
}
{
add.f16x2 r7, %36, %48;
}
{
add.f16x2 r10, r4, r7;
}
{
add.f16x2 r13, %40, %44;
}
{
add.f16x2 r16, r10, r13;
}
{
add.f16x2 r19, %33, %53;
}
{
add.f16x2 r22, %29, r19;
}
{
add.f16x2 r25, %37, %49;
}
{
add.f16x2 r28, r22, r25;
}
{
add.f16x2 r31, %41, %45;
}
{
add.f16x2 r34, r28, r31;
}
{
add.f16x2 r37, %32, %52;
}
{
mul.f16x2 r40, r37, r420;
}
{
add.f16x2 r43, %28, r40;
}
{
add.f16x2 r46, %36, %48;
}
{
mul.f16x2 r49, r46, r654;
}
{
add.f16x2 r52, r43, r49;
}
{
add.f16x2 r55, %40, %44;
}
{
mul.f16x2 r58, r55, r636;
}
{
add.f16x2 r61, r52, r58;
}
{
sub.f16x2 r64, %33, %53;
}
{
mul.f16x2 r67, r64, r447;
}
{
sub.f16x2 r70, %37, %49;
}
{
mul.f16x2 r73, r70, r678;
}
{
add.f16x2 r76, r67, r73;
}
{
sub.f16x2 r79, %41, %45;
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
add.f16x2 r91, %32, %52;
}
{
mul.f16x2 r94, r91, r420;
}
{
add.f16x2 r97, %28, r94;
}
{
add.f16x2 r100, %36, %48;
}
{
mul.f16x2 r103, r100, r654;
}
{
add.f16x2 r106, r97, r103;
}
{
add.f16x2 r109, %40, %44;
}
{
mul.f16x2 r112, r109, r636;
}
{
add.f16x2 r115, r106, r112;
}
{
sub.f16x2 r118, %33, %53;
}
{
mul.f16x2 r121, r118, r447;
}
{
sub.f16x2 r124, %37, %49;
}
{
mul.f16x2 r127, r124, r678;
}
{
add.f16x2 r130, r121, r127;
}
{
sub.f16x2 r133, %41, %45;
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
add.f16x2 r145, %32, %52;
}
{
mul.f16x2 r148, r145, r654;
}
{
add.f16x2 r151, %28, r148;
}
{
add.f16x2 r154, %36, %48;
}
{
mul.f16x2 r157, r154, r537;
}
{
add.f16x2 r160, r151, r157;
}
{
add.f16x2 r163, %40, %44;
}
{
mul.f16x2 r166, r163, r645;
}
{
add.f16x2 r169, r160, r166;
}
{
sub.f16x2 r172, %33, %53;
}
{
mul.f16x2 r175, r172, r678;
}
{
sub.f16x2 r178, %37, %49;
}
{
mul.f16x2 r181, r178, r561;
}
{
add.f16x2 r184, r175, r181;
}
{
sub.f16x2 r187, %41, %45;
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
add.f16x2 r199, %32, %52;
}
{
mul.f16x2 r202, r199, r654;
}
{
add.f16x2 r205, %28, r202;
}
{
add.f16x2 r208, %36, %48;
}
{
mul.f16x2 r211, r208, r537;
}
{
add.f16x2 r214, r205, r211;
}
{
add.f16x2 r217, %40, %44;
}
{
mul.f16x2 r220, r217, r645;
}
{
add.f16x2 r223, r214, r220;
}
{
sub.f16x2 r226, %33, %53;
}
{
mul.f16x2 r229, r226, r678;
}
{
sub.f16x2 r232, %37, %49;
}
{
mul.f16x2 r235, r232, r561;
}
{
add.f16x2 r238, r229, r235;
}
{
sub.f16x2 r241, %41, %45;
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
add.f16x2 r253, %32, %52;
}
{
mul.f16x2 r256, r253, r636;
}
{
add.f16x2 r259, %28, r256;
}
{
add.f16x2 r262, %36, %48;
}
{
mul.f16x2 r265, r262, r645;
}
{
add.f16x2 r268, r259, r265;
}
{
add.f16x2 r271, %40, %44;
}
{
mul.f16x2 r274, r271, r654;
}
{
add.f16x2 r277, r268, r274;
}
{
sub.f16x2 r280, %33, %53;
}
{
mul.f16x2 r283, r280, r663;
}
{
sub.f16x2 r286, %37, %49;
}
{
mul.f16x2 r289, r286, r669;
}
{
add.f16x2 r292, r283, r289;
}
{
sub.f16x2 r295, %41, %45;
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
add.f16x2 r307, %32, %52;
}
{
mul.f16x2 r310, r307, r636;
}
{
add.f16x2 r313, %28, r310;
}
{
add.f16x2 r316, %36, %48;
}
{
mul.f16x2 r319, r316, r645;
}
{
add.f16x2 r322, r313, r319;
}
{
add.f16x2 r325, %40, %44;
}
{
mul.f16x2 r328, r325, r654;
}
{
add.f16x2 r331, r322, r328;
}
{
sub.f16x2 r334, %33, %53;
}
{
mul.f16x2 r337, r334, r663;
}
{
sub.f16x2 r340, %37, %49;
}
{
mul.f16x2 r343, r340, r669;
}
{
add.f16x2 r346, r337, r343;
}
{
sub.f16x2 r349, %41, %45;
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
add.f16x2 r361, %33, %53;
}
{
mul.f16x2 r364, r361, r420;
}
{
add.f16x2 r367, %29, r364;
}
{
add.f16x2 r370, %37, %49;
}
{
mul.f16x2 r373, r370, r654;
}
{
add.f16x2 r376, r367, r373;
}
{
add.f16x2 r379, %41, %45;
}
{
mul.f16x2 r382, r379, r636;
}
{
add.f16x2 r385, r376, r382;
}
{
sub.f16x2 r388, %32, %52;
}
{
mul.f16x2 r391, r388, r447;
}
{
sub.f16x2 r394, %36, %48;
}
{
mul.f16x2 r397, r394, r678;
}
{
add.f16x2 r400, r391, r397;
}
{
sub.f16x2 r403, %40, %44;
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
add.f16x2 r415, %33, %53;
}
{
mul.f16x2 r418, r415, r420;
}
{
add.f16x2 r421, %29, r418;
}
{
add.f16x2 r424, %37, %49;
}
{
mul.f16x2 r427, r424, r654;
}
{
add.f16x2 r430, r421, r427;
}
{
add.f16x2 r433, %41, %45;
}
{
mul.f16x2 r436, r433, r636;
}
{
add.f16x2 r439, r430, r436;
}
{
sub.f16x2 r442, %32, %52;
}
{
mul.f16x2 r445, r442, r447;
}
{
sub.f16x2 r448, %36, %48;
}
{
mul.f16x2 r451, r448, r678;
}
{
add.f16x2 r454, r445, r451;
}
{
sub.f16x2 r457, %40, %44;
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
add.f16x2 r469, %33, %53;
}
{
mul.f16x2 r472, r469, r654;
}
{
add.f16x2 r475, %29, r472;
}
{
add.f16x2 r478, %37, %49;
}
{
mul.f16x2 r481, r478, r537;
}
{
add.f16x2 r484, r475, r481;
}
{
add.f16x2 r487, %41, %45;
}
{
mul.f16x2 r490, r487, r645;
}
{
add.f16x2 r493, r484, r490;
}
{
sub.f16x2 r496, %32, %52;
}
{
mul.f16x2 r499, r496, r678;
}
{
sub.f16x2 r502, %36, %48;
}
{
mul.f16x2 r505, r502, r561;
}
{
add.f16x2 r508, r499, r505;
}
{
sub.f16x2 r511, %40, %44;
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
add.f16x2 r523, %33, %53;
}
{
mul.f16x2 r526, r523, r654;
}
{
add.f16x2 r529, %29, r526;
}
{
add.f16x2 r532, %37, %49;
}
{
mul.f16x2 r535, r532, r537;
}
{
add.f16x2 r538, r529, r535;
}
{
add.f16x2 r541, %41, %45;
}
{
mul.f16x2 r544, r541, r645;
}
{
add.f16x2 r547, r538, r544;
}
{
sub.f16x2 r550, %32, %52;
}
{
mul.f16x2 r553, r550, r678;
}
{
sub.f16x2 r556, %36, %48;
}
{
mul.f16x2 r559, r556, r561;
}
{
add.f16x2 r562, r553, r559;
}
{
sub.f16x2 r565, %40, %44;
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
add.f16x2 r577, %33, %53;
}
{
mul.f16x2 r580, r577, r636;
}
{
add.f16x2 r583, %29, r580;
}
{
add.f16x2 r586, %37, %49;
}
{
mul.f16x2 r589, r586, r645;
}
{
add.f16x2 r592, r583, r589;
}
{
add.f16x2 r595, %41, %45;
}
{
mul.f16x2 r598, r595, r654;
}
{
add.f16x2 r601, r592, r598;
}
{
sub.f16x2 r604, %32, %52;
}
{
mul.f16x2 r607, r604, r663;
}
{
sub.f16x2 r610, %36, %48;
}
{
mul.f16x2 r613, r610, r669;
}
{
add.f16x2 r616, r607, r613;
}
{
sub.f16x2 r619, %40, %44;
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
add.f16x2 r631, %33, %53;
}
{
mul.f16x2 r634, r631, r636;
}
{
add.f16x2 r637, %29, r634;
}
{
add.f16x2 r640, %37, %49;
}
{
mul.f16x2 r643, r640, r645;
}
{
add.f16x2 r646, r637, r643;
}
{
add.f16x2 r649, %41, %45;
}
{
mul.f16x2 r652, r649, r654;
}
{
add.f16x2 r655, r646, r652;
}
{
sub.f16x2 r658, %32, %52;
}
{
mul.f16x2 r661, r658, r663;
}
{
sub.f16x2 r664, %36, %48;
}
{
mul.f16x2 r667, r664, r669;
}
{
add.f16x2 r670, r661, r667;
}
{
sub.f16x2 r673, %40, %44;
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
cvt.rn.f16.f64 rs17, fd23;
}
mov.b32 r1104, {rs17, rs17};
{
cvt.rn.f16.f64 rs18, fd20;
}
{
neg.f16 rs19, rs18;
}
mov.b32 r1131, {rs19, rs19};
{
cvt.rn.f16.f64 rs21, fd27;
}
mov.b32 r1338, {rs21, rs21};
{
cvt.rn.f16.f64 rs22, fd14;
}
{
neg.f16 rs23, rs22;
}
mov.b32 r1362, {rs23, rs23};
{
cvt.rn.f16.f64 rs25, fd31;
}
mov.b32 r1320, {rs25, rs25};
{
cvt.rn.f16.f64 rs26, fd18;
}
{
neg.f16 rs27, rs26;
}
mov.b32 r1347, {rs27, rs27};
{
cvt.rn.f16.f64 rs29, fd31;
}
mov.b32 r1221, {rs29, rs29};
{
cvt.rn.f16.f64 rs30, fd18;
}
mov.b32 r1245, {rs30, rs30};
{
cvt.rn.f16.f64 rs31, fd23;
}
mov.b32 r1329, {rs31, rs31};
{
cvt.rn.f16.f64 rs32, fd20;
}
mov.b32 r1353, {rs32, rs32};
{
add.f16x2 r685, %34, %54;
}
{
add.f16x2 r688, %30, r685;
}
{
add.f16x2 r691, %38, %50;
}
{
add.f16x2 r694, r688, r691;
}
{
add.f16x2 r697, %42, %46;
}
{
add.f16x2 r700, r694, r697;
}
{
add.f16x2 r703, %35, %55;
}
{
add.f16x2 r706, %31, r703;
}
{
add.f16x2 r709, %39, %51;
}
{
add.f16x2 r712, r706, r709;
}
{
add.f16x2 r715, %43, %47;
}
{
add.f16x2 r718, r712, r715;
}
{
add.f16x2 r721, %34, %54;
}
{
mul.f16x2 r724, r721, r1104;
}
{
add.f16x2 r727, %30, r724;
}
{
add.f16x2 r730, %38, %50;
}
{
mul.f16x2 r733, r730, r1338;
}
{
add.f16x2 r736, r727, r733;
}
{
add.f16x2 r739, %42, %46;
}
{
mul.f16x2 r742, r739, r1320;
}
{
add.f16x2 r745, r736, r742;
}
{
sub.f16x2 r748, %35, %55;
}
{
mul.f16x2 r751, r748, r1131;
}
{
sub.f16x2 r754, %39, %51;
}
{
mul.f16x2 r757, r754, r1362;
}
{
add.f16x2 r760, r751, r757;
}
{
sub.f16x2 r763, %43, %47;
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
add.f16x2 r775, %34, %54;
}
{
mul.f16x2 r778, r775, r1104;
}
{
add.f16x2 r781, %30, r778;
}
{
add.f16x2 r784, %38, %50;
}
{
mul.f16x2 r787, r784, r1338;
}
{
add.f16x2 r790, r781, r787;
}
{
add.f16x2 r793, %42, %46;
}
{
mul.f16x2 r796, r793, r1320;
}
{
add.f16x2 r799, r790, r796;
}
{
sub.f16x2 r802, %35, %55;
}
{
mul.f16x2 r805, r802, r1131;
}
{
sub.f16x2 r808, %39, %51;
}
{
mul.f16x2 r811, r808, r1362;
}
{
add.f16x2 r814, r805, r811;
}
{
sub.f16x2 r817, %43, %47;
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
add.f16x2 r829, %34, %54;
}
{
mul.f16x2 r832, r829, r1338;
}
{
add.f16x2 r835, %30, r832;
}
{
add.f16x2 r838, %38, %50;
}
{
mul.f16x2 r841, r838, r1221;
}
{
add.f16x2 r844, r835, r841;
}
{
add.f16x2 r847, %42, %46;
}
{
mul.f16x2 r850, r847, r1329;
}
{
add.f16x2 r853, r844, r850;
}
{
sub.f16x2 r856, %35, %55;
}
{
mul.f16x2 r859, r856, r1362;
}
{
sub.f16x2 r862, %39, %51;
}
{
mul.f16x2 r865, r862, r1245;
}
{
add.f16x2 r868, r859, r865;
}
{
sub.f16x2 r871, %43, %47;
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
add.f16x2 r883, %34, %54;
}
{
mul.f16x2 r886, r883, r1338;
}
{
add.f16x2 r889, %30, r886;
}
{
add.f16x2 r892, %38, %50;
}
{
mul.f16x2 r895, r892, r1221;
}
{
add.f16x2 r898, r889, r895;
}
{
add.f16x2 r901, %42, %46;
}
{
mul.f16x2 r904, r901, r1329;
}
{
add.f16x2 r907, r898, r904;
}
{
sub.f16x2 r910, %35, %55;
}
{
mul.f16x2 r913, r910, r1362;
}
{
sub.f16x2 r916, %39, %51;
}
{
mul.f16x2 r919, r916, r1245;
}
{
add.f16x2 r922, r913, r919;
}
{
sub.f16x2 r925, %43, %47;
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
add.f16x2 r937, %34, %54;
}
{
mul.f16x2 r940, r937, r1320;
}
{
add.f16x2 r943, %30, r940;
}
{
add.f16x2 r946, %38, %50;
}
{
mul.f16x2 r949, r946, r1329;
}
{
add.f16x2 r952, r943, r949;
}
{
add.f16x2 r955, %42, %46;
}
{
mul.f16x2 r958, r955, r1338;
}
{
add.f16x2 r961, r952, r958;
}
{
sub.f16x2 r964, %35, %55;
}
{
mul.f16x2 r967, r964, r1347;
}
{
sub.f16x2 r970, %39, %51;
}
{
mul.f16x2 r973, r970, r1353;
}
{
add.f16x2 r976, r967, r973;
}
{
sub.f16x2 r979, %43, %47;
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
add.f16x2 r991, %34, %54;
}
{
mul.f16x2 r994, r991, r1320;
}
{
add.f16x2 r997, %30, r994;
}
{
add.f16x2 r1000, %38, %50;
}
{
mul.f16x2 r1003, r1000, r1329;
}
{
add.f16x2 r1006, r997, r1003;
}
{
add.f16x2 r1009, %42, %46;
}
{
mul.f16x2 r1012, r1009, r1338;
}
{
add.f16x2 r1015, r1006, r1012;
}
{
sub.f16x2 r1018, %35, %55;
}
{
mul.f16x2 r1021, r1018, r1347;
}
{
sub.f16x2 r1024, %39, %51;
}
{
mul.f16x2 r1027, r1024, r1353;
}
{
add.f16x2 r1030, r1021, r1027;
}
{
sub.f16x2 r1033, %43, %47;
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
add.f16x2 r1045, %35, %55;
}
{
mul.f16x2 r1048, r1045, r1104;
}
{
add.f16x2 r1051, %31, r1048;
}
{
add.f16x2 r1054, %39, %51;
}
{
mul.f16x2 r1057, r1054, r1338;
}
{
add.f16x2 r1060, r1051, r1057;
}
{
add.f16x2 r1063, %43, %47;
}
{
mul.f16x2 r1066, r1063, r1320;
}
{
add.f16x2 r1069, r1060, r1066;
}
{
sub.f16x2 r1072, %34, %54;
}
{
mul.f16x2 r1075, r1072, r1131;
}
{
sub.f16x2 r1078, %38, %50;
}
{
mul.f16x2 r1081, r1078, r1362;
}
{
add.f16x2 r1084, r1075, r1081;
}
{
sub.f16x2 r1087, %42, %46;
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
add.f16x2 r1099, %35, %55;
}
{
mul.f16x2 r1102, r1099, r1104;
}
{
add.f16x2 r1105, %31, r1102;
}
{
add.f16x2 r1108, %39, %51;
}
{
mul.f16x2 r1111, r1108, r1338;
}
{
add.f16x2 r1114, r1105, r1111;
}
{
add.f16x2 r1117, %43, %47;
}
{
mul.f16x2 r1120, r1117, r1320;
}
{
add.f16x2 r1123, r1114, r1120;
}
{
sub.f16x2 r1126, %34, %54;
}
{
mul.f16x2 r1129, r1126, r1131;
}
{
sub.f16x2 r1132, %38, %50;
}
{
mul.f16x2 r1135, r1132, r1362;
}
{
add.f16x2 r1138, r1129, r1135;
}
{
sub.f16x2 r1141, %42, %46;
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
add.f16x2 r1153, %35, %55;
}
{
mul.f16x2 r1156, r1153, r1338;
}
{
add.f16x2 r1159, %31, r1156;
}
{
add.f16x2 r1162, %39, %51;
}
{
mul.f16x2 r1165, r1162, r1221;
}
{
add.f16x2 r1168, r1159, r1165;
}
{
add.f16x2 r1171, %43, %47;
}
{
mul.f16x2 r1174, r1171, r1329;
}
{
add.f16x2 r1177, r1168, r1174;
}
{
sub.f16x2 r1180, %34, %54;
}
{
mul.f16x2 r1183, r1180, r1362;
}
{
sub.f16x2 r1186, %38, %50;
}
{
mul.f16x2 r1189, r1186, r1245;
}
{
add.f16x2 r1192, r1183, r1189;
}
{
sub.f16x2 r1195, %42, %46;
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
add.f16x2 r1207, %35, %55;
}
{
mul.f16x2 r1210, r1207, r1338;
}
{
add.f16x2 r1213, %31, r1210;
}
{
add.f16x2 r1216, %39, %51;
}
{
mul.f16x2 r1219, r1216, r1221;
}
{
add.f16x2 r1222, r1213, r1219;
}
{
add.f16x2 r1225, %43, %47;
}
{
mul.f16x2 r1228, r1225, r1329;
}
{
add.f16x2 r1231, r1222, r1228;
}
{
sub.f16x2 r1234, %34, %54;
}
{
mul.f16x2 r1237, r1234, r1362;
}
{
sub.f16x2 r1240, %38, %50;
}
{
mul.f16x2 r1243, r1240, r1245;
}
{
add.f16x2 r1246, r1237, r1243;
}
{
sub.f16x2 r1249, %42, %46;
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
add.f16x2 r1261, %35, %55;
}
{
mul.f16x2 r1264, r1261, r1320;
}
{
add.f16x2 r1267, %31, r1264;
}
{
add.f16x2 r1270, %39, %51;
}
{
mul.f16x2 r1273, r1270, r1329;
}
{
add.f16x2 r1276, r1267, r1273;
}
{
add.f16x2 r1279, %43, %47;
}
{
mul.f16x2 r1282, r1279, r1338;
}
{
add.f16x2 r1285, r1276, r1282;
}
{
sub.f16x2 r1288, %34, %54;
}
{
mul.f16x2 r1291, r1288, r1347;
}
{
sub.f16x2 r1294, %38, %50;
}
{
mul.f16x2 r1297, r1294, r1353;
}
{
add.f16x2 r1300, r1291, r1297;
}
{
sub.f16x2 r1303, %42, %46;
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
add.f16x2 r1315, %35, %55;
}
{
mul.f16x2 r1318, r1315, r1320;
}
{
add.f16x2 r1321, %31, r1318;
}
{
add.f16x2 r1324, %39, %51;
}
{
mul.f16x2 r1327, r1324, r1329;
}
{
add.f16x2 r1330, r1321, r1327;
}
{
add.f16x2 r1333, %43, %47;
}
{
mul.f16x2 r1336, r1333, r1338;
}
{
add.f16x2 r1339, r1330, r1336;
}
{
sub.f16x2 r1342, %34, %54;
}
{
mul.f16x2 r1345, r1342, r1347;
}
{
sub.f16x2 r1348, %38, %50;
}
{
mul.f16x2 r1351, r1348, r1353;
}
{
add.f16x2 r1354, r1345, r1351;
}
{
sub.f16x2 r1357, %42, %46;
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
mov.f64 fd21, 0d3FECD4BCA9CB5C71;
{
cvt.rn.f16.f64 rs33, fd21;
}
mov.f64 fd32, 0d3FDBC4C04D71ABC1;
{
cvt.rn.f16.f64 rs34, fd32;
}
{
cvt.rn.f16.f64 rs35, fd23;
}
mov.f64 fd30, 0d3FE904C37505DE4B;
{
cvt.rn.f16.f64 rs36, fd30;
}
mov.f64 fd25, 0d3FCC7B90E3024582;
{
cvt.rn.f16.f64 rs37, fd25;
}
mov.f64 fd28, 0d3FEF329C0558E969;
{
cvt.rn.f16.f64 rs38, fd28;
}
{
cvt.rn.f16.f64 rs39, fd27;
}
{
cvt.rn.f16.f64 rs40, fd28;
}
mov.f64 fd29, 0dBFE3F3A0E28BEDD1;
{
cvt.rn.f16.f64 rs41, fd29;
}
{
cvt.rn.f16.f64 rs42, fd30;
}
{
cvt.rn.f16.f64 rs43, fd31;
}
{
cvt.rn.f16.f64 rs44, fd32;
}
mov.b32 r1383, {rs33, rs33};
{
mul.f16x2 r1369, r772, r1383;
}
mov.b32 r1380, {rs34, rs34};
{
mul.f16x2 r1372, r1096, r1380;
}
{
sub.f16x2 r1375, r1369, r1372;
}
{
mul.f16x2 r1378, r772, r1380;
}
{
fma.rn.f16x2 r1381, r1096, r1383, r1378;
}
mov.b32 r1399, {rs35, rs35};
{
mul.f16x2 r1385, r880, r1399;
}
mov.b32 r1396, {rs36, rs36};
{
mul.f16x2 r1388, r1204, r1396;
}
{
sub.f16x2 r1391, r1385, r1388;
}
{
mul.f16x2 r1394, r880, r1396;
}
{
fma.rn.f16x2 r1397, r1204, r1399, r1394;
}
mov.b32 r1415, {rs37, rs37};
{
mul.f16x2 r1401, r988, r1415;
}
mov.b32 r1412, {rs38, rs38};
{
mul.f16x2 r1404, r1312, r1412;
}
{
sub.f16x2 r1407, r1401, r1404;
}
{
mul.f16x2 r1410, r988, r1412;
}
{
fma.rn.f16x2 r1413, r1312, r1415, r1410;
}
mov.b32 r1431, {rs39, rs39};
{
mul.f16x2 r1417, r1042, r1431;
}
mov.b32 r1428, {rs40, rs40};
{
mul.f16x2 r1420, r1366, r1428;
}
{
sub.f16x2 r1423, r1417, r1420;
}
{
mul.f16x2 r1426, r1042, r1428;
}
{
fma.rn.f16x2 r1429, r1366, r1431, r1426;
}
mov.b32 r1447, {rs41, rs41};
{
mul.f16x2 r1433, r934, r1447;
}
mov.b32 r1444, {rs42, rs42};
{
mul.f16x2 r1436, r1258, r1444;
}
{
sub.f16x2 r1439, r1433, r1436;
}
{
mul.f16x2 r1442, r934, r1444;
}
{
fma.rn.f16x2 r1445, r1258, r1447, r1442;
}
mov.b32 r1463, {rs43, rs43};
{
mul.f16x2 r1449, r826, r1463;
}
mov.b32 r1460, {rs44, rs44};
{
mul.f16x2 r1452, r1150, r1460;
}
{
sub.f16x2 r1455, r1449, r1452;
}
{
mul.f16x2 r1458, r826, r1460;
}
{
fma.rn.f16x2 r1461, r1150, r1463, r1458;
}
{
add.f16x2 %0, r16, r700;
}
{
add.f16x2 %1, r34, r718;
}
{
sub.f16x2 %14, r16, r700;
}
{
sub.f16x2 %15, r34, r718;
}
{
add.f16x2 %2, r88, r1375;
}
{
add.f16x2 %3, r412, r1381;
}
{
sub.f16x2 %16, r88, r1375;
}
{
sub.f16x2 %17, r412, r1381;
}
{
add.f16x2 %4, r196, r1391;
}
{
add.f16x2 %5, r520, r1397;
}
{
sub.f16x2 %18, r196, r1391;
}
{
sub.f16x2 %19, r520, r1397;
}
{
add.f16x2 %6, r304, r1407;
}
{
add.f16x2 %7, r628, r1413;
}
{
sub.f16x2 %20, r304, r1407;
}
{
sub.f16x2 %21, r628, r1413;
}
{
add.f16x2 %8, r358, r1423;
}
{
add.f16x2 %9, r682, r1429;
}
{
sub.f16x2 %22, r358, r1423;
}
{
sub.f16x2 %23, r682, r1429;
}
{
add.f16x2 %10, r250, r1439;
}
{
add.f16x2 %11, r574, r1445;
}
{
sub.f16x2 %24, r250, r1439;
}
{
sub.f16x2 %25, r574, r1445;
}
{
add.f16x2 %12, r142, r1455;
}
{
add.f16x2 %13, r466, r1461;
}
{
sub.f16x2 %26, r142, r1455;
}
{
sub.f16x2 %27, r466, r1461;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)));
};


#endif
