#ifndef CUFFTDX_FFT_7_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_7_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1119, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<17>;
.reg .b32 r<685>;
.reg .f64 fd<11>;
.reg .b64 rd<2>;
mov.f64 fd9, 0d3FE3F3A0E28BEDD1;
{
cvt.rn.f16.f64 rs1, fd9;
}
mov.b32 r420, {rs1, rs1};
mov.f64 fd10, 0dBFE904C37505DE4B;
{
cvt.rn.f16.f64 rs2, fd10;
}
{
neg.f16 rs3, rs2;
}
mov.b32 r447, {rs3, rs3};
mov.f64 fd3, 0dBFCC7B90E3024582;
{
cvt.rn.f16.f64 rs5, fd3;
}
mov.b32 r654, {rs5, rs5};
mov.f64 fd4, 0dBFEF329C0558E969;
{
cvt.rn.f16.f64 rs6, fd4;
}
{
neg.f16 rs7, rs6;
}
mov.b32 r678, {rs7, rs7};
mov.f64 fd7, 0dBFECD4BCA9CB5C71;
{
cvt.rn.f16.f64 rs9, fd7;
}
mov.b32 r636, {rs9, rs9};
mov.f64 fd8, 0dBFDBC4C04D71ABC1;
{
cvt.rn.f16.f64 rs10, fd8;
}
{
neg.f16 rs11, rs10;
}
mov.b32 r663, {rs11, rs11};
{
cvt.rn.f16.f64 rs13, fd7;
}
mov.b32 r537, {rs13, rs13};
{
cvt.rn.f16.f64 rs14, fd8;
}
mov.b32 r561, {rs14, rs14};
{
cvt.rn.f16.f64 rs15, fd9;
}
mov.b32 r645, {rs15, rs15};
{
cvt.rn.f16.f64 rs16, fd10;
}
mov.b32 r669, {rs16, rs16};
{
add.f16x2 r1, %23, %21;
}
{
add.f16x2 r4, %24, r1;
}
{
add.f16x2 r7, %25, %19;
}
{
add.f16x2 r10, r4, r7;
}
{
add.f16x2 r13, %15, %17;
}
{
add.f16x2 %0, r10, r13;
}
{
add.f16x2 r19, %26, %22;
}
{
add.f16x2 r22, %27, r19;
}
{
add.f16x2 r25, %14, %20;
}
{
add.f16x2 r28, r22, r25;
}
{
add.f16x2 r31, %16, %18;
}
{
add.f16x2 %1, r28, r31;
}
{
add.f16x2 r37, %23, %21;
}
{
mul.f16x2 r40, r37, r420;
}
{
add.f16x2 r43, %24, r40;
}
{
add.f16x2 r46, %25, %19;
}
{
mul.f16x2 r49, r46, r654;
}
{
add.f16x2 r52, r43, r49;
}
{
add.f16x2 r55, %15, %17;
}
{
mul.f16x2 r58, r55, r636;
}
{
add.f16x2 r61, r52, r58;
}
{
sub.f16x2 r64, %26, %22;
}
{
mul.f16x2 r67, r64, r447;
}
{
sub.f16x2 r70, %14, %20;
}
{
mul.f16x2 r73, r70, r678;
}
{
add.f16x2 r76, r67, r73;
}
{
sub.f16x2 r79, %16, %18;
}
{
mul.f16x2 r82, r79, r663;
}
{
add.f16x2 r85, r76, r82;
}
{
sub.f16x2 %2, r61, r85;
}
{
add.f16x2 r91, %23, %21;
}
{
mul.f16x2 r94, r91, r420;
}
{
add.f16x2 r97, %24, r94;
}
{
add.f16x2 r100, %25, %19;
}
{
mul.f16x2 r103, r100, r654;
}
{
add.f16x2 r106, r97, r103;
}
{
add.f16x2 r109, %15, %17;
}
{
mul.f16x2 r112, r109, r636;
}
{
add.f16x2 r115, r106, r112;
}
{
sub.f16x2 r118, %26, %22;
}
{
mul.f16x2 r121, r118, r447;
}
{
sub.f16x2 r124, %14, %20;
}
{
mul.f16x2 r127, r124, r678;
}
{
add.f16x2 r130, r121, r127;
}
{
sub.f16x2 r133, %16, %18;
}
{
mul.f16x2 r136, r133, r663;
}
{
add.f16x2 r139, r130, r136;
}
{
add.f16x2 %12, r115, r139;
}
{
add.f16x2 r145, %23, %21;
}
{
mul.f16x2 r148, r145, r654;
}
{
add.f16x2 r151, %24, r148;
}
{
add.f16x2 r154, %25, %19;
}
{
mul.f16x2 r157, r154, r537;
}
{
add.f16x2 r160, r151, r157;
}
{
add.f16x2 r163, %15, %17;
}
{
mul.f16x2 r166, r163, r645;
}
{
add.f16x2 r169, r160, r166;
}
{
sub.f16x2 r172, %26, %22;
}
{
mul.f16x2 r175, r172, r678;
}
{
sub.f16x2 r178, %14, %20;
}
{
mul.f16x2 r181, r178, r561;
}
{
add.f16x2 r184, r175, r181;
}
{
sub.f16x2 r187, %16, %18;
}
{
mul.f16x2 r190, r187, r669;
}
{
add.f16x2 r193, r184, r190;
}
{
sub.f16x2 %4, r169, r193;
}
{
add.f16x2 r199, %23, %21;
}
{
mul.f16x2 r202, r199, r654;
}
{
add.f16x2 r205, %24, r202;
}
{
add.f16x2 r208, %25, %19;
}
{
mul.f16x2 r211, r208, r537;
}
{
add.f16x2 r214, r205, r211;
}
{
add.f16x2 r217, %15, %17;
}
{
mul.f16x2 r220, r217, r645;
}
{
add.f16x2 r223, r214, r220;
}
{
sub.f16x2 r226, %26, %22;
}
{
mul.f16x2 r229, r226, r678;
}
{
sub.f16x2 r232, %14, %20;
}
{
mul.f16x2 r235, r232, r561;
}
{
add.f16x2 r238, r229, r235;
}
{
sub.f16x2 r241, %16, %18;
}
{
mul.f16x2 r244, r241, r669;
}
{
add.f16x2 r247, r238, r244;
}
{
add.f16x2 %10, r223, r247;
}
{
add.f16x2 r253, %23, %21;
}
{
mul.f16x2 r256, r253, r636;
}
{
add.f16x2 r259, %24, r256;
}
{
add.f16x2 r262, %25, %19;
}
{
mul.f16x2 r265, r262, r645;
}
{
add.f16x2 r268, r259, r265;
}
{
add.f16x2 r271, %15, %17;
}
{
mul.f16x2 r274, r271, r654;
}
{
add.f16x2 r277, r268, r274;
}
{
sub.f16x2 r280, %26, %22;
}
{
mul.f16x2 r283, r280, r663;
}
{
sub.f16x2 r286, %14, %20;
}
{
mul.f16x2 r289, r286, r669;
}
{
add.f16x2 r292, r283, r289;
}
{
sub.f16x2 r295, %16, %18;
}
{
mul.f16x2 r298, r295, r678;
}
{
add.f16x2 r301, r292, r298;
}
{
sub.f16x2 %6, r277, r301;
}
{
add.f16x2 r307, %23, %21;
}
{
mul.f16x2 r310, r307, r636;
}
{
add.f16x2 r313, %24, r310;
}
{
add.f16x2 r316, %25, %19;
}
{
mul.f16x2 r319, r316, r645;
}
{
add.f16x2 r322, r313, r319;
}
{
add.f16x2 r325, %15, %17;
}
{
mul.f16x2 r328, r325, r654;
}
{
add.f16x2 r331, r322, r328;
}
{
sub.f16x2 r334, %26, %22;
}
{
mul.f16x2 r337, r334, r663;
}
{
sub.f16x2 r340, %14, %20;
}
{
mul.f16x2 r343, r340, r669;
}
{
add.f16x2 r346, r337, r343;
}
{
sub.f16x2 r349, %16, %18;
}
{
mul.f16x2 r352, r349, r678;
}
{
add.f16x2 r355, r346, r352;
}
{
add.f16x2 %8, r331, r355;
}
{
add.f16x2 r361, %26, %22;
}
{
mul.f16x2 r364, r361, r420;
}
{
add.f16x2 r367, %27, r364;
}
{
add.f16x2 r370, %14, %20;
}
{
mul.f16x2 r373, r370, r654;
}
{
add.f16x2 r376, r367, r373;
}
{
add.f16x2 r379, %16, %18;
}
{
mul.f16x2 r382, r379, r636;
}
{
add.f16x2 r385, r376, r382;
}
{
sub.f16x2 r388, %23, %21;
}
{
mul.f16x2 r391, r388, r447;
}
{
sub.f16x2 r394, %25, %19;
}
{
mul.f16x2 r397, r394, r678;
}
{
add.f16x2 r400, r391, r397;
}
{
sub.f16x2 r403, %15, %17;
}
{
mul.f16x2 r406, r403, r663;
}
{
add.f16x2 r409, r400, r406;
}
{
add.f16x2 %3, r385, r409;
}
{
add.f16x2 r415, %26, %22;
}
{
mul.f16x2 r418, r415, r420;
}
{
add.f16x2 r421, %27, r418;
}
{
add.f16x2 r424, %14, %20;
}
{
mul.f16x2 r427, r424, r654;
}
{
add.f16x2 r430, r421, r427;
}
{
add.f16x2 r433, %16, %18;
}
{
mul.f16x2 r436, r433, r636;
}
{
add.f16x2 r439, r430, r436;
}
{
sub.f16x2 r442, %23, %21;
}
{
mul.f16x2 r445, r442, r447;
}
{
sub.f16x2 r448, %25, %19;
}
{
mul.f16x2 r451, r448, r678;
}
{
add.f16x2 r454, r445, r451;
}
{
sub.f16x2 r457, %15, %17;
}
{
mul.f16x2 r460, r457, r663;
}
{
add.f16x2 r463, r454, r460;
}
{
sub.f16x2 %13, r439, r463;
}
{
add.f16x2 r469, %26, %22;
}
{
mul.f16x2 r472, r469, r654;
}
{
add.f16x2 r475, %27, r472;
}
{
add.f16x2 r478, %14, %20;
}
{
mul.f16x2 r481, r478, r537;
}
{
add.f16x2 r484, r475, r481;
}
{
add.f16x2 r487, %16, %18;
}
{
mul.f16x2 r490, r487, r645;
}
{
add.f16x2 r493, r484, r490;
}
{
sub.f16x2 r496, %23, %21;
}
{
mul.f16x2 r499, r496, r678;
}
{
sub.f16x2 r502, %25, %19;
}
{
mul.f16x2 r505, r502, r561;
}
{
add.f16x2 r508, r499, r505;
}
{
sub.f16x2 r511, %15, %17;
}
{
mul.f16x2 r514, r511, r669;
}
{
add.f16x2 r517, r508, r514;
}
{
add.f16x2 %5, r493, r517;
}
{
add.f16x2 r523, %26, %22;
}
{
mul.f16x2 r526, r523, r654;
}
{
add.f16x2 r529, %27, r526;
}
{
add.f16x2 r532, %14, %20;
}
{
mul.f16x2 r535, r532, r537;
}
{
add.f16x2 r538, r529, r535;
}
{
add.f16x2 r541, %16, %18;
}
{
mul.f16x2 r544, r541, r645;
}
{
add.f16x2 r547, r538, r544;
}
{
sub.f16x2 r550, %23, %21;
}
{
mul.f16x2 r553, r550, r678;
}
{
sub.f16x2 r556, %25, %19;
}
{
mul.f16x2 r559, r556, r561;
}
{
add.f16x2 r562, r553, r559;
}
{
sub.f16x2 r565, %15, %17;
}
{
mul.f16x2 r568, r565, r669;
}
{
add.f16x2 r571, r562, r568;
}
{
sub.f16x2 %11, r547, r571;
}
{
add.f16x2 r577, %26, %22;
}
{
mul.f16x2 r580, r577, r636;
}
{
add.f16x2 r583, %27, r580;
}
{
add.f16x2 r586, %14, %20;
}
{
mul.f16x2 r589, r586, r645;
}
{
add.f16x2 r592, r583, r589;
}
{
add.f16x2 r595, %16, %18;
}
{
mul.f16x2 r598, r595, r654;
}
{
add.f16x2 r601, r592, r598;
}
{
sub.f16x2 r604, %23, %21;
}
{
mul.f16x2 r607, r604, r663;
}
{
sub.f16x2 r610, %25, %19;
}
{
mul.f16x2 r613, r610, r669;
}
{
add.f16x2 r616, r607, r613;
}
{
sub.f16x2 r619, %15, %17;
}
{
mul.f16x2 r622, r619, r678;
}
{
add.f16x2 r625, r616, r622;
}
{
add.f16x2 %7, r601, r625;
}
{
add.f16x2 r631, %26, %22;
}
{
mul.f16x2 r634, r631, r636;
}
{
add.f16x2 r637, %27, r634;
}
{
add.f16x2 r640, %14, %20;
}
{
mul.f16x2 r643, r640, r645;
}
{
add.f16x2 r646, r637, r643;
}
{
add.f16x2 r649, %16, %18;
}
{
mul.f16x2 r652, r649, r654;
}
{
add.f16x2 r655, r646, r652;
}
{
sub.f16x2 r658, %23, %21;
}
{
mul.f16x2 r661, r658, r663;
}
{
sub.f16x2 r664, %25, %19;
}
{
mul.f16x2 r667, r664, r669;
}
{
add.f16x2 r670, r661, r667;
}
{
sub.f16x2 r673, %15, %17;
}
{
mul.f16x2 r676, r673, r678;
}
{
add.f16x2 r679, r670, r676;
}
{
sub.f16x2 %9, r655, r679;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)): "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[0].y)));
};


#endif
