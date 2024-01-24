#ifndef CUFFTDX_FFT_10_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_10_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<936, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<35>;
.reg .b32 r<749>;
.reg .f64 fd<31>;
.reg .b64 rd<2>;
mov.f64 fd15, 0d3FD3C6EF372FE950;
{
cvt.rn.f16.f64 rs1, fd15;
}
mov.b32 r210, {rs1, rs1};
mov.f64 fd18, 0dBFEE6F0E134454FF;
{
cvt.rn.f16.f64 rs2, fd18;
}
mov.b32 r228, {rs2, rs2};
mov.f64 fd19, 0dBFE9E3779B97F4A8;
{
cvt.rn.f16.f64 rs3, fd19;
}
mov.b32 r282, {rs3, rs3};
mov.f64 fd20, 0dBFE2CF2304755A5E;
{
cvt.rn.f16.f64 rs4, fd20;
}
mov.b32 r300, {rs4, rs4};
{
cvt.rn.f16.f64 rs5, fd15;
}
mov.b32 r291, {rs5, rs5};
{
cvt.rn.f16.f64 rs6, fd18;
}
{
neg.f16 rs7, rs6;
}
mov.b32 r306, {rs7, rs7};
{
add.f16x2 r1, %24, %36;
}
{
add.f16x2 r4, %20, r1;
}
{
add.f16x2 r7, %28, %32;
}
{
add.f16x2 r10, r4, r7;
}
{
add.f16x2 r13, %25, %37;
}
{
add.f16x2 r16, %21, r13;
}
{
add.f16x2 r19, %29, %33;
}
{
add.f16x2 r22, r16, r19;
}
{
add.f16x2 r25, %24, %36;
}
{
mul.f16x2 r28, r25, r210;
}
{
add.f16x2 r31, %20, r28;
}
{
add.f16x2 r34, %28, %32;
}
{
mul.f16x2 r37, r34, r282;
}
{
add.f16x2 r40, r31, r37;
}
{
sub.f16x2 r43, %25, %37;
}
{
mul.f16x2 r46, r43, r228;
}
{
sub.f16x2 r49, %29, %33;
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
add.f16x2 r61, %24, %36;
}
{
mul.f16x2 r64, r61, r210;
}
{
add.f16x2 r67, %20, r64;
}
{
add.f16x2 r70, %28, %32;
}
{
mul.f16x2 r73, r70, r282;
}
{
add.f16x2 r76, r67, r73;
}
{
sub.f16x2 r79, %25, %37;
}
{
mul.f16x2 r82, r79, r228;
}
{
sub.f16x2 r85, %29, %33;
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
add.f16x2 r97, %24, %36;
}
{
mul.f16x2 r100, r97, r282;
}
{
add.f16x2 r103, %20, r100;
}
{
add.f16x2 r106, %28, %32;
}
{
mul.f16x2 r109, r106, r291;
}
{
add.f16x2 r112, r103, r109;
}
{
sub.f16x2 r115, %25, %37;
}
{
mul.f16x2 r118, r115, r300;
}
{
sub.f16x2 r121, %29, %33;
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
add.f16x2 r133, %24, %36;
}
{
mul.f16x2 r136, r133, r282;
}
{
add.f16x2 r139, %20, r136;
}
{
add.f16x2 r142, %28, %32;
}
{
mul.f16x2 r145, r142, r291;
}
{
add.f16x2 r148, r139, r145;
}
{
sub.f16x2 r151, %25, %37;
}
{
mul.f16x2 r154, r151, r300;
}
{
sub.f16x2 r157, %29, %33;
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
add.f16x2 r169, %25, %37;
}
{
mul.f16x2 r172, r169, r210;
}
{
add.f16x2 r175, %21, r172;
}
{
add.f16x2 r178, %29, %33;
}
{
mul.f16x2 r181, r178, r282;
}
{
add.f16x2 r184, r175, r181;
}
{
sub.f16x2 r187, %24, %36;
}
{
mul.f16x2 r190, r187, r228;
}
{
sub.f16x2 r193, %28, %32;
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
add.f16x2 r205, %25, %37;
}
{
mul.f16x2 r208, r205, r210;
}
{
add.f16x2 r211, %21, r208;
}
{
add.f16x2 r214, %29, %33;
}
{
mul.f16x2 r217, r214, r282;
}
{
add.f16x2 r220, r211, r217;
}
{
sub.f16x2 r223, %24, %36;
}
{
mul.f16x2 r226, r223, r228;
}
{
sub.f16x2 r229, %28, %32;
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
add.f16x2 r241, %25, %37;
}
{
mul.f16x2 r244, r241, r282;
}
{
add.f16x2 r247, %21, r244;
}
{
add.f16x2 r250, %29, %33;
}
{
mul.f16x2 r253, r250, r291;
}
{
add.f16x2 r256, r247, r253;
}
{
sub.f16x2 r259, %24, %36;
}
{
mul.f16x2 r262, r259, r300;
}
{
sub.f16x2 r265, %28, %32;
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
add.f16x2 r277, %25, %37;
}
{
mul.f16x2 r280, r277, r282;
}
{
add.f16x2 r283, %21, r280;
}
{
add.f16x2 r286, %29, %33;
}
{
mul.f16x2 r289, r286, r291;
}
{
add.f16x2 r292, r283, r289;
}
{
sub.f16x2 r295, %24, %36;
}
{
mul.f16x2 r298, r295, r300;
}
{
sub.f16x2 r301, %28, %32;
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
cvt.rn.f16.f64 rs9, fd15;
}
mov.b32 r522, {rs9, rs9};
{
cvt.rn.f16.f64 rs10, fd18;
}
mov.b32 r540, {rs10, rs10};
{
cvt.rn.f16.f64 rs11, fd19;
}
mov.b32 r594, {rs11, rs11};
{
cvt.rn.f16.f64 rs12, fd20;
}
mov.b32 r612, {rs12, rs12};
{
cvt.rn.f16.f64 rs13, fd15;
}
mov.b32 r603, {rs13, rs13};
{
cvt.rn.f16.f64 rs14, fd18;
}
{
neg.f16 rs15, rs14;
}
mov.b32 r618, {rs15, rs15};
{
add.f16x2 r313, %26, %38;
}
{
add.f16x2 r316, %22, r313;
}
{
add.f16x2 r319, %30, %34;
}
{
add.f16x2 r322, r316, r319;
}
{
add.f16x2 r325, %27, %39;
}
{
add.f16x2 r328, %23, r325;
}
{
add.f16x2 r331, %31, %35;
}
{
add.f16x2 r334, r328, r331;
}
{
add.f16x2 r337, %26, %38;
}
{
mul.f16x2 r340, r337, r522;
}
{
add.f16x2 r343, %22, r340;
}
{
add.f16x2 r346, %30, %34;
}
{
mul.f16x2 r349, r346, r594;
}
{
add.f16x2 r352, r343, r349;
}
{
sub.f16x2 r355, %27, %39;
}
{
mul.f16x2 r358, r355, r540;
}
{
sub.f16x2 r361, %31, %35;
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
add.f16x2 r373, %26, %38;
}
{
mul.f16x2 r376, r373, r522;
}
{
add.f16x2 r379, %22, r376;
}
{
add.f16x2 r382, %30, %34;
}
{
mul.f16x2 r385, r382, r594;
}
{
add.f16x2 r388, r379, r385;
}
{
sub.f16x2 r391, %27, %39;
}
{
mul.f16x2 r394, r391, r540;
}
{
sub.f16x2 r397, %31, %35;
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
add.f16x2 r409, %26, %38;
}
{
mul.f16x2 r412, r409, r594;
}
{
add.f16x2 r415, %22, r412;
}
{
add.f16x2 r418, %30, %34;
}
{
mul.f16x2 r421, r418, r603;
}
{
add.f16x2 r424, r415, r421;
}
{
sub.f16x2 r427, %27, %39;
}
{
mul.f16x2 r430, r427, r612;
}
{
sub.f16x2 r433, %31, %35;
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
add.f16x2 r445, %26, %38;
}
{
mul.f16x2 r448, r445, r594;
}
{
add.f16x2 r451, %22, r448;
}
{
add.f16x2 r454, %30, %34;
}
{
mul.f16x2 r457, r454, r603;
}
{
add.f16x2 r460, r451, r457;
}
{
sub.f16x2 r463, %27, %39;
}
{
mul.f16x2 r466, r463, r612;
}
{
sub.f16x2 r469, %31, %35;
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
add.f16x2 r481, %27, %39;
}
{
mul.f16x2 r484, r481, r522;
}
{
add.f16x2 r487, %23, r484;
}
{
add.f16x2 r490, %31, %35;
}
{
mul.f16x2 r493, r490, r594;
}
{
add.f16x2 r496, r487, r493;
}
{
sub.f16x2 r499, %26, %38;
}
{
mul.f16x2 r502, r499, r540;
}
{
sub.f16x2 r505, %30, %34;
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
add.f16x2 r517, %27, %39;
}
{
mul.f16x2 r520, r517, r522;
}
{
add.f16x2 r523, %23, r520;
}
{
add.f16x2 r526, %31, %35;
}
{
mul.f16x2 r529, r526, r594;
}
{
add.f16x2 r532, r523, r529;
}
{
sub.f16x2 r535, %26, %38;
}
{
mul.f16x2 r538, r535, r540;
}
{
sub.f16x2 r541, %30, %34;
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
add.f16x2 r553, %27, %39;
}
{
mul.f16x2 r556, r553, r594;
}
{
add.f16x2 r559, %23, r556;
}
{
add.f16x2 r562, %31, %35;
}
{
mul.f16x2 r565, r562, r603;
}
{
add.f16x2 r568, r559, r565;
}
{
sub.f16x2 r571, %26, %38;
}
{
mul.f16x2 r574, r571, r612;
}
{
sub.f16x2 r577, %30, %34;
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
add.f16x2 r589, %27, %39;
}
{
mul.f16x2 r592, r589, r594;
}
{
add.f16x2 r595, %23, r592;
}
{
add.f16x2 r598, %31, %35;
}
{
mul.f16x2 r601, r598, r603;
}
{
add.f16x2 r604, r595, r601;
}
{
sub.f16x2 r607, %26, %38;
}
{
mul.f16x2 r610, r607, r612;
}
{
sub.f16x2 r613, %30, %34;
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
mov.f64 fd13, 0d3FE9E3779B97F4A8;
{
cvt.rn.f16.f64 rs17, fd13;
}
{
cvt.rn.f16.f64 rs18, fd20;
}
{
cvt.rn.f16.f64 rs19, fd15;
}
{
cvt.rn.f16.f64 rs20, fd18;
}
mov.f64 fd17, 0dBFD3C6EF372FE950;
{
cvt.rn.f16.f64 rs21, fd17;
}
{
cvt.rn.f16.f64 rs22, fd18;
}
{
cvt.rn.f16.f64 rs23, fd19;
}
{
cvt.rn.f16.f64 rs24, fd20;
}
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
add.f16x2 %0, r10, r322;
}
{
add.f16x2 %1, r22, r334;
}
{
sub.f16x2 %10, r10, r322;
}
{
sub.f16x2 %11, r22, r334;
}
{
add.f16x2 %2, r58, r631;
}
{
add.f16x2 %3, r202, r637;
}
{
sub.f16x2 %12, r58, r631;
}
{
sub.f16x2 %13, r202, r637;
}
{
add.f16x2 %4, r130, r647;
}
{
add.f16x2 %5, r274, r653;
}
{
sub.f16x2 %14, r130, r647;
}
{
sub.f16x2 %15, r274, r653;
}
{
add.f16x2 %6, r166, r663;
}
{
add.f16x2 %7, r310, r669;
}
{
sub.f16x2 %16, r166, r663;
}
{
sub.f16x2 %17, r310, r669;
}
{
add.f16x2 %8, r94, r679;
}
{
add.f16x2 %9, r238, r685;
}
{
sub.f16x2 %18, r94, r679;
}
{
sub.f16x2 %19, r238, r685;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)));
};


#endif
