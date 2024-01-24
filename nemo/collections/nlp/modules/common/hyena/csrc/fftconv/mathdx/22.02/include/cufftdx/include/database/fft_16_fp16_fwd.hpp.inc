#ifndef CUFFTDX_FFT_16_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_16_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<771, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<59>;
.reg .b32 r<559>;
.reg .f64 fd<59>;
.reg .b64 rd<2>;
{
add.f16x2 r1, %32, %48;
}
{
add.f16x2 r4, %33, %49;
}
{
sub.f16x2 r7, %32, %48;
}
{
sub.f16x2 r10, %33, %49;
}
{
add.f16x2 r13, %40, %56;
}
{
add.f16x2 r16, %41, %57;
}
{
sub.f16x2 r19, %40, %56;
}
{
sub.f16x2 r22, %41, %57;
}
{
neg.f16x2 r25, r19;
}
{
add.f16x2 r27, r1, r13;
}
{
add.f16x2 r30, r4, r16;
}
{
sub.f16x2 r33, r1, r13;
}
{
sub.f16x2 r36, r4, r16;
}
{
add.f16x2 r39, r7, r22;
}
{
add.f16x2 r42, r10, r25;
}
{
sub.f16x2 r45, r7, r22;
}
{
sub.f16x2 r48, r10, r25;
}
{
add.f16x2 r51, %36, %52;
}
{
add.f16x2 r54, %37, %53;
}
{
sub.f16x2 r57, %36, %52;
}
{
sub.f16x2 r60, %37, %53;
}
{
add.f16x2 r63, %44, %60;
}
{
add.f16x2 r66, %45, %61;
}
{
sub.f16x2 r69, %44, %60;
}
{
sub.f16x2 r72, %45, %61;
}
{
neg.f16x2 r75, r69;
}
{
add.f16x2 r77, r51, r63;
}
{
add.f16x2 r80, r54, r66;
}
{
sub.f16x2 r83, r51, r63;
}
{
sub.f16x2 r86, r54, r66;
}
{
add.f16x2 r89, r57, r72;
}
{
add.f16x2 r92, r60, r75;
}
{
sub.f16x2 r95, r57, r72;
}
{
sub.f16x2 r98, r60, r75;
}
mov.f64 fd31, 0d3FE6A09E667F3BCD;
{
cvt.rn.f16.f64 rs1, fd31;
}
mov.f64 fd40, 0dBFE6A09E667F3BCD;
{
cvt.rn.f16.f64 rs2, fd40;
}
{
cvt.rn.f16.f64 rs5, fd40;
}
{
cvt.rn.f16.f64 rs6, fd40;
}
mov.b32 r115, {rs1, rs1};
{
mul.f16x2 r101, r89, r115;
}
mov.b32 r112, {rs2, rs2};
{
mul.f16x2 r104, r92, r112;
}
{
sub.f16x2 r107, r101, r104;
}
{
mul.f16x2 r110, r89, r112;
}
{
fma.rn.f16x2 r113, r92, r115, r110;
}
{
neg.f16x2 r117, r83;
}
mov.b32 r133, {rs5, rs5};
{
mul.f16x2 r119, r95, r133;
}
mov.b32 r130, {rs6, rs6};
{
mul.f16x2 r122, r98, r130;
}
{
sub.f16x2 r125, r119, r122;
}
{
mul.f16x2 r128, r95, r130;
}
{
fma.rn.f16x2 r131, r98, r133, r128;
}
{
add.f16x2 r135, r27, r77;
}
{
add.f16x2 r138, r30, r80;
}
{
sub.f16x2 r141, r27, r77;
}
{
sub.f16x2 r144, r30, r80;
}
{
add.f16x2 r147, r39, r107;
}
{
add.f16x2 r150, r42, r113;
}
{
sub.f16x2 r153, r39, r107;
}
{
sub.f16x2 r156, r42, r113;
}
{
add.f16x2 r159, r33, r86;
}
{
add.f16x2 r162, r36, r117;
}
{
sub.f16x2 r165, r33, r86;
}
{
sub.f16x2 r168, r36, r117;
}
{
add.f16x2 r171, r45, r125;
}
{
add.f16x2 r174, r48, r131;
}
{
sub.f16x2 r177, r45, r125;
}
{
sub.f16x2 r180, r48, r131;
}
{
add.f16x2 r183, %34, %50;
}
{
add.f16x2 r186, %35, %51;
}
{
sub.f16x2 r189, %34, %50;
}
{
sub.f16x2 r192, %35, %51;
}
{
add.f16x2 r195, %42, %58;
}
{
add.f16x2 r198, %43, %59;
}
{
sub.f16x2 r201, %42, %58;
}
{
sub.f16x2 r204, %43, %59;
}
{
neg.f16x2 r207, r201;
}
{
add.f16x2 r209, r183, r195;
}
{
add.f16x2 r212, r186, r198;
}
{
sub.f16x2 r215, r183, r195;
}
{
sub.f16x2 r218, r186, r198;
}
{
add.f16x2 r221, r189, r204;
}
{
add.f16x2 r224, r192, r207;
}
{
sub.f16x2 r227, r189, r204;
}
{
sub.f16x2 r230, r192, r207;
}
{
add.f16x2 r233, %38, %54;
}
{
add.f16x2 r236, %39, %55;
}
{
sub.f16x2 r239, %38, %54;
}
{
sub.f16x2 r242, %39, %55;
}
{
add.f16x2 r245, %46, %62;
}
{
add.f16x2 r248, %47, %63;
}
{
sub.f16x2 r251, %46, %62;
}
{
sub.f16x2 r254, %47, %63;
}
{
neg.f16x2 r257, r251;
}
{
add.f16x2 r259, r233, r245;
}
{
add.f16x2 r262, r236, r248;
}
{
sub.f16x2 r265, r233, r245;
}
{
sub.f16x2 r268, r236, r248;
}
{
add.f16x2 r271, r239, r254;
}
{
add.f16x2 r274, r242, r257;
}
{
sub.f16x2 r277, r239, r254;
}
{
sub.f16x2 r280, r242, r257;
}
{
cvt.rn.f16.f64 rs15, fd31;
}
{
cvt.rn.f16.f64 rs16, fd40;
}
{
cvt.rn.f16.f64 rs19, fd40;
}
{
cvt.rn.f16.f64 rs20, fd40;
}
mov.b32 r297, {rs15, rs15};
{
mul.f16x2 r283, r271, r297;
}
mov.b32 r294, {rs16, rs16};
{
mul.f16x2 r286, r274, r294;
}
{
sub.f16x2 r289, r283, r286;
}
{
mul.f16x2 r292, r271, r294;
}
{
fma.rn.f16x2 r295, r274, r297, r292;
}
{
neg.f16x2 r299, r265;
}
mov.b32 r315, {rs19, rs19};
{
mul.f16x2 r301, r277, r315;
}
mov.b32 r312, {rs20, rs20};
{
mul.f16x2 r304, r280, r312;
}
{
sub.f16x2 r307, r301, r304;
}
{
mul.f16x2 r310, r277, r312;
}
{
fma.rn.f16x2 r313, r280, r315, r310;
}
{
add.f16x2 r317, r209, r259;
}
{
add.f16x2 r320, r212, r262;
}
{
sub.f16x2 r323, r209, r259;
}
{
sub.f16x2 r326, r212, r262;
}
{
add.f16x2 r329, r221, r289;
}
{
add.f16x2 r332, r224, r295;
}
{
sub.f16x2 r335, r221, r289;
}
{
sub.f16x2 r338, r224, r295;
}
{
add.f16x2 r341, r215, r268;
}
{
add.f16x2 r344, r218, r299;
}
{
sub.f16x2 r347, r215, r268;
}
{
sub.f16x2 r350, r218, r299;
}
{
add.f16x2 r353, r227, r307;
}
{
add.f16x2 r356, r230, r313;
}
{
sub.f16x2 r359, r227, r307;
}
{
sub.f16x2 r362, r230, r313;
}
mov.f64 fd29, 0d3FED906BCF328D46;
{
cvt.rn.f16.f64 rs29, fd29;
}
mov.f64 fd42, 0dBFD87DE2A6AEA963;
{
cvt.rn.f16.f64 rs30, fd42;
}
{
cvt.rn.f16.f64 rs31, fd31;
}
{
cvt.rn.f16.f64 rs32, fd40;
}
mov.f64 fd33, 0d3FD87DE2A6AEA963;
{
cvt.rn.f16.f64 rs33, fd33;
}
mov.f64 fd41, 0dBFED906BCF328D46;
{
cvt.rn.f16.f64 rs34, fd41;
}
{
cvt.rn.f16.f64 rs37, fd42;
}
{
cvt.rn.f16.f64 rs38, fd41;
}
{
cvt.rn.f16.f64 rs39, fd40;
}
{
cvt.rn.f16.f64 rs40, fd40;
}
{
cvt.rn.f16.f64 rs41, fd41;
}
{
cvt.rn.f16.f64 rs42, fd42;
}
mov.b32 r379, {rs29, rs29};
{
mul.f16x2 r365, r329, r379;
}
mov.b32 r376, {rs30, rs30};
{
mul.f16x2 r368, r332, r376;
}
{
sub.f16x2 r371, r365, r368;
}
{
mul.f16x2 r374, r329, r376;
}
{
fma.rn.f16x2 r377, r332, r379, r374;
}
mov.b32 r395, {rs31, rs31};
{
mul.f16x2 r381, r341, r395;
}
mov.b32 r392, {rs32, rs32};
{
mul.f16x2 r384, r344, r392;
}
{
sub.f16x2 r387, r381, r384;
}
{
mul.f16x2 r390, r341, r392;
}
{
fma.rn.f16x2 r393, r344, r395, r390;
}
mov.b32 r411, {rs33, rs33};
{
mul.f16x2 r397, r353, r411;
}
mov.b32 r408, {rs34, rs34};
{
mul.f16x2 r400, r356, r408;
}
{
sub.f16x2 r403, r397, r400;
}
{
mul.f16x2 r406, r353, r408;
}
{
fma.rn.f16x2 r409, r356, r411, r406;
}
{
neg.f16x2 r413, r323;
}
mov.b32 r429, {rs37, rs37};
{
mul.f16x2 r415, r335, r429;
}
mov.b32 r426, {rs38, rs38};
{
mul.f16x2 r418, r338, r426;
}
{
sub.f16x2 r421, r415, r418;
}
{
mul.f16x2 r424, r335, r426;
}
{
fma.rn.f16x2 r427, r338, r429, r424;
}
mov.b32 r445, {rs39, rs39};
{
mul.f16x2 r431, r347, r445;
}
mov.b32 r442, {rs40, rs40};
{
mul.f16x2 r434, r350, r442;
}
{
sub.f16x2 r437, r431, r434;
}
{
mul.f16x2 r440, r347, r442;
}
{
fma.rn.f16x2 r443, r350, r445, r440;
}
mov.b32 r461, {rs41, rs41};
{
mul.f16x2 r447, r359, r461;
}
mov.b32 r458, {rs42, rs42};
{
mul.f16x2 r450, r362, r458;
}
{
sub.f16x2 r453, r447, r450;
}
{
mul.f16x2 r456, r359, r458;
}
{
fma.rn.f16x2 r459, r362, r461, r456;
}
{
add.f16x2 %0, r135, r317;
}
{
add.f16x2 %1, r138, r320;
}
{
sub.f16x2 %16, r135, r317;
}
{
sub.f16x2 %17, r138, r320;
}
{
add.f16x2 %2, r147, r371;
}
{
add.f16x2 %3, r150, r377;
}
{
sub.f16x2 %18, r147, r371;
}
{
sub.f16x2 %19, r150, r377;
}
{
add.f16x2 %4, r159, r387;
}
{
add.f16x2 %5, r162, r393;
}
{
sub.f16x2 %20, r159, r387;
}
{
sub.f16x2 %21, r162, r393;
}
{
add.f16x2 %6, r171, r403;
}
{
add.f16x2 %7, r174, r409;
}
{
sub.f16x2 %22, r171, r403;
}
{
sub.f16x2 %23, r174, r409;
}
{
add.f16x2 %8, r141, r326;
}
{
add.f16x2 %9, r144, r413;
}
{
sub.f16x2 %24, r141, r326;
}
{
sub.f16x2 %25, r144, r413;
}
{
add.f16x2 %10, r153, r421;
}
{
add.f16x2 %11, r156, r427;
}
{
sub.f16x2 %26, r153, r421;
}
{
sub.f16x2 %27, r156, r427;
}
{
add.f16x2 %12, r165, r437;
}
{
add.f16x2 %13, r168, r443;
}
{
sub.f16x2 %28, r165, r437;
}
{
sub.f16x2 %29, r168, r443;
}
{
add.f16x2 %14, r177, r453;
}
{
add.f16x2 %15, r180, r459;
}
{
sub.f16x2 %30, r177, r453;
}
{
sub.f16x2 %31, r180, r459;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<772, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<14>;
.reg .b32 r<227>;
.reg .b64 rd<2>;
mov.u32 r215, %tid.y;
shl.b32 r216, r215, 6;
mov.u32 r217, %8;
add.s32 r218, r217, r216;
mov.u32 r219, %tid.x;
{
add.f16x2 r1, %9, %13;
}
{
add.f16x2 r4, %10, %14;
}
{
sub.f16x2 r7, %9, %13;
}
{
sub.f16x2 r10, %10, %14;
}
{
add.f16x2 r13, %11, %15;
}
{
add.f16x2 r16, %12, %16;
}
{
sub.f16x2 r19, %11, %15;
}
{
sub.f16x2 r22, %12, %16;
}
{
neg.f16x2 r25, r19;
}
{
add.f16x2 r27, r1, r13;
}
{
add.f16x2 r30, r4, r16;
}
{
sub.f16x2 r33, r1, r13;
}
{
sub.f16x2 r36, r4, r16;
}
{
add.f16x2 r39, r7, r22;
}
{
add.f16x2 r42, r10, r25;
}
{
sub.f16x2 r45, r7, r22;
}
{
sub.f16x2 r48, r10, r25;
}
and.b32 r220, r219, 3;
shl.b32 r221, r219, 4;
and.b32 r222, r221, -64;
add.s32 r223, r218, r222;
cvt.rn.f32.u32 f11, r220;
mul.f32 f12, f11, 0f3EC90FDB;
cos.approx.f32 f1, f12;
sin.approx.f32 f13, f12;
neg.f32 f2, f13;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1;
cvt.rn.f16.f32 high, f2;
mov.b32 r51, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r54, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r56, {high, high};
}
{
mul.f16x2 r58, r42, r56;
}
{
neg.f16x2 r61, r58;
}
{
fma.rn.f16x2 r63, r39, r54, r61;
}
{
mul.f16x2 r67, r39, r56;
}
{
fma.rn.f16x2 r70, r42, r54, r67;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r74, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r76, {high, high};
}
mov.f32 f7, 0fBF800000;
mov.f32 f8, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f7;
cvt.rn.f16.f32 high, f8;
mov.b32 r78, {low, high};
}
{
mul.f16x2 r79, r76, r78;
}
{
mul.f16x2 r82, r51, r74;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r85, {high, low};
}
{
fma.rn.f16x2 r87, r79, r85, r82;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r91, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r93, {high, high};
}
{
mul.f16x2 r95, r36, r93;
}
{
neg.f16x2 r98, r95;
}
{
fma.rn.f16x2 r100, r33, r91, r98;
}
{
mul.f16x2 r104, r33, r93;
}
{
fma.rn.f16x2 r107, r36, r91, r104;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r111, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r113, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f7;
cvt.rn.f16.f32 high, f8;
mov.b32 r115, {low, high};
}
{
mul.f16x2 r116, r113, r115;
}
{
mul.f16x2 r119, r87, r111;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r122, {high, low};
}
{
fma.rn.f16x2 r124, r116, r122, r119;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r124;
mov.b32 r128, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r124;
mov.b32 r130, {high, high};
}
{
mul.f16x2 r132, r48, r130;
}
{
neg.f16x2 r135, r132;
}
{
fma.rn.f16x2 r137, r45, r128, r135;
}
{
mul.f16x2 r141, r45, r130;
}
{
fma.rn.f16x2 r144, r48, r128, r141;
}
barrier.sync 0;
and.b32 r224, r221, 48;
add.s32 r225, r223, r224;
st.shared.v4.f32 [r225], {r27, r63, r100, r137};
barrier.sync 0;
mad.lo.s32 r226, r220, -12, r225;
ld.shared.u32 r166, [r226];
ld.shared.u32 r178, [r226+16];
ld.shared.u32 r167, [r226+32];
ld.shared.u32 r179, [r226+48];
barrier.sync 0;
st.shared.v4.f32 [r225], {r30, r70, r107, r144};
barrier.sync 0;
ld.shared.u32 r169, [r226];
ld.shared.u32 r181, [r226+16];
ld.shared.u32 r170, [r226+32];
ld.shared.u32 r182, [r226+48];
{
add.f16x2 r165, r166, r167;
}
{
add.f16x2 r168, r169, r170;
}
{
sub.f16x2 r171, r166, r167;
}
{
sub.f16x2 r174, r169, r170;
}
{
add.f16x2 r177, r178, r179;
}
{
add.f16x2 r180, r181, r182;
}
{
sub.f16x2 r183, r178, r179;
}
{
sub.f16x2 r186, r181, r182;
}
{
neg.f16x2 r189, r183;
}
{
add.f16x2 %0, r165, r177;
}
{
add.f16x2 %1, r168, r180;
}
{
sub.f16x2 %4, r165, r177;
}
{
sub.f16x2 %5, r168, r180;
}
{
add.f16x2 %2, r171, r186;
}
{
add.f16x2 %3, r174, r189;
}
{
sub.f16x2 %6, r171, r186;
}
{
sub.f16x2 %7, r174, r189;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<773, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<14>;
.reg .b32 r<227>;
.reg .b64 rd<2>;
mov.u32 r215, %tid.y;
shl.b32 r216, r215, 7;
mov.u32 r217, %8;
add.s32 r218, r217, r216;
mov.u32 r219, %tid.x;
{
add.f16x2 r1, %9, %13;
}
{
add.f16x2 r4, %10, %14;
}
{
sub.f16x2 r7, %9, %13;
}
{
sub.f16x2 r10, %10, %14;
}
{
add.f16x2 r13, %11, %15;
}
{
add.f16x2 r16, %12, %16;
}
{
sub.f16x2 r19, %11, %15;
}
{
sub.f16x2 r22, %12, %16;
}
{
neg.f16x2 r25, r19;
}
{
add.f16x2 r27, r1, r13;
}
{
add.f16x2 r30, r4, r16;
}
{
sub.f16x2 r33, r1, r13;
}
{
sub.f16x2 r36, r4, r16;
}
{
add.f16x2 r39, r7, r22;
}
{
add.f16x2 r42, r10, r25;
}
{
sub.f16x2 r45, r7, r22;
}
{
sub.f16x2 r48, r10, r25;
}
and.b32 r220, r219, 3;
shl.b32 r221, r219, 5;
and.b32 r222, r221, -128;
add.s32 r223, r218, r222;
cvt.rn.f32.u32 f11, r220;
mul.f32 f12, f11, 0f3EC90FDB;
cos.approx.f32 f1, f12;
sin.approx.f32 f13, f12;
neg.f32 f2, f13;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1;
cvt.rn.f16.f32 high, f2;
mov.b32 r51, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r54, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r56, {high, high};
}
{
mul.f16x2 r58, r42, r56;
}
{
neg.f16x2 r61, r58;
}
{
fma.rn.f16x2 r63, r39, r54, r61;
}
{
mul.f16x2 r67, r39, r56;
}
{
fma.rn.f16x2 r70, r42, r54, r67;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r74, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r76, {high, high};
}
mov.f32 f7, 0fBF800000;
mov.f32 f8, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f7;
cvt.rn.f16.f32 high, f8;
mov.b32 r78, {low, high};
}
{
mul.f16x2 r79, r76, r78;
}
{
mul.f16x2 r82, r51, r74;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r85, {high, low};
}
{
fma.rn.f16x2 r87, r79, r85, r82;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r91, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r93, {high, high};
}
{
mul.f16x2 r95, r36, r93;
}
{
neg.f16x2 r98, r95;
}
{
fma.rn.f16x2 r100, r33, r91, r98;
}
{
mul.f16x2 r104, r33, r93;
}
{
fma.rn.f16x2 r107, r36, r91, r104;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r111, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r113, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f7;
cvt.rn.f16.f32 high, f8;
mov.b32 r115, {low, high};
}
{
mul.f16x2 r116, r113, r115;
}
{
mul.f16x2 r119, r87, r111;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r122, {high, low};
}
{
fma.rn.f16x2 r124, r116, r122, r119;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r124;
mov.b32 r128, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r124;
mov.b32 r130, {high, high};
}
{
mul.f16x2 r132, r48, r130;
}
{
neg.f16x2 r135, r132;
}
{
fma.rn.f16x2 r137, r45, r128, r135;
}
{
mul.f16x2 r141, r45, r130;
}
{
fma.rn.f16x2 r144, r48, r128, r141;
}
barrier.sync 0;
and.b32 r224, r221, 96;
add.s32 r225, r223, r224;
st.shared.v4.f32 [r225], {r27, r30, r63, r70};
st.shared.v4.f32 [r225+16], {r100, r107, r137, r144};
barrier.sync 0;
mad.lo.s32 r226, r220, -24, r225;
ld.shared.u32 r166, [r226];
ld.shared.u32 r169, [r226+4];
ld.shared.u32 r178, [r226+32];
ld.shared.u32 r181, [r226+36];
ld.shared.u32 r167, [r226+64];
ld.shared.u32 r170, [r226+68];
ld.shared.u32 r179, [r226+96];
ld.shared.u32 r182, [r226+100];
{
add.f16x2 r165, r166, r167;
}
{
add.f16x2 r168, r169, r170;
}
{
sub.f16x2 r171, r166, r167;
}
{
sub.f16x2 r174, r169, r170;
}
{
add.f16x2 r177, r178, r179;
}
{
add.f16x2 r180, r181, r182;
}
{
sub.f16x2 r183, r178, r179;
}
{
sub.f16x2 r186, r181, r182;
}
{
neg.f16x2 r189, r183;
}
{
add.f16x2 %0, r165, r177;
}
{
add.f16x2 %1, r168, r180;
}
{
sub.f16x2 %4, r165, r177;
}
{
sub.f16x2 %5, r168, r180;
}
{
add.f16x2 %2, r171, r186;
}
{
add.f16x2 %3, r174, r189;
}
{
sub.f16x2 %6, r171, r186;
}
{
sub.f16x2 %7, r174, r189;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<774, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<50>;
.reg .b32 r<519>;
.reg .b64 rd<2>;
mov.u32 r507, %tid.y;
shl.b32 r508, r507, 6;
mov.u32 r509, %16;
add.s32 r510, r509, r508;
mov.u32 r511, %tid.x;
{
add.f16x2 r1, %17, %25;
}
{
add.f16x2 r4, %18, %26;
}
{
sub.f16x2 r7, %17, %25;
}
{
sub.f16x2 r10, %18, %26;
}
{
add.f16x2 r13, %21, %29;
}
{
add.f16x2 r16, %22, %30;
}
{
sub.f16x2 r19, %21, %29;
}
{
sub.f16x2 r22, %22, %30;
}
{
neg.f16x2 r25, r19;
}
{
add.f16x2 r27, r1, r13;
}
{
add.f16x2 r30, r4, r16;
}
{
sub.f16x2 r33, r1, r13;
}
{
sub.f16x2 r36, r4, r16;
}
{
add.f16x2 r39, r7, r22;
}
{
add.f16x2 r42, r10, r25;
}
{
sub.f16x2 r45, r7, r22;
}
{
sub.f16x2 r48, r10, r25;
}
{
add.f16x2 r51, %19, %27;
}
{
add.f16x2 r54, %20, %28;
}
{
sub.f16x2 r57, %19, %27;
}
{
sub.f16x2 r60, %20, %28;
}
{
add.f16x2 r63, %23, %31;
}
{
add.f16x2 r66, %24, %32;
}
{
sub.f16x2 r69, %23, %31;
}
{
sub.f16x2 r72, %24, %32;
}
{
neg.f16x2 r75, r69;
}
{
add.f16x2 r77, r51, r63;
}
{
add.f16x2 r80, r54, r66;
}
{
sub.f16x2 r83, r51, r63;
}
{
sub.f16x2 r86, r54, r66;
}
{
add.f16x2 r89, r57, r72;
}
{
add.f16x2 r92, r60, r75;
}
{
sub.f16x2 r95, r57, r72;
}
{
sub.f16x2 r98, r60, r75;
}
mov.f32 f2, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f2;
cvt.rn.f16.f32 high, f2;
mov.b32 r101, {low, high};
}
mov.f32 f12, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f12;
cvt.rn.f16.f32 high, f12;
mov.b32 r102, {low, high};
}
mov.f32 f43, 0fBF800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f12;
cvt.rn.f16.f32 high, f12;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f12;
cvt.rn.f16.f32 high, f12;
mov.b32 r106, {low, high};
}
mov.f32 f44, 0f3F800000;
{
mul.f16x2 r115, r89, r101;
}
{
mul.f16x2 r118, r92, r102;
}
{
sub.f16x2 r121, r115, r118;
}
{
mul.f16x2 r124, r89, r102;
}
{
fma.rn.f16x2 r127, r92, r101, r124;
}
{
neg.f16x2 r131, r83;
}
{
mul.f16x2 r133, r95, r105;
}
{
mul.f16x2 r136, r98, r106;
}
{
sub.f16x2 r139, r133, r136;
}
{
mul.f16x2 r142, r95, r106;
}
{
fma.rn.f16x2 r145, r98, r105, r142;
}
{
add.f16x2 r149, r27, r77;
}
{
add.f16x2 r152, r30, r80;
}
{
sub.f16x2 r155, r27, r77;
}
{
sub.f16x2 r158, r30, r80;
}
{
add.f16x2 r161, r39, r121;
}
{
add.f16x2 r164, r42, r127;
}
{
sub.f16x2 r167, r39, r121;
}
{
sub.f16x2 r170, r42, r127;
}
{
add.f16x2 r173, r33, r86;
}
{
add.f16x2 r176, r36, r131;
}
{
sub.f16x2 r179, r33, r86;
}
{
sub.f16x2 r182, r36, r131;
}
{
add.f16x2 r185, r45, r139;
}
{
add.f16x2 r188, r48, r145;
}
{
sub.f16x2 r191, r45, r139;
}
{
sub.f16x2 r194, r48, r145;
}
and.b32 r512, r511, 1;
shl.b32 r513, r511, 5;
and.b32 r514, r513, -64;
add.s32 r515, r510, r514;
cvt.rn.f32.u32 f47, r512;
mul.f32 f48, f47, 0f3EC90FDB;
cos.approx.f32 f29, f48;
sin.approx.f32 f49, f48;
neg.f32 f30, f49;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f29;
cvt.rn.f16.f32 high, f30;
mov.b32 r197, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r200, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r202, {high, high};
}
{
mul.f16x2 r204, r164, r202;
}
{
neg.f16x2 r207, r204;
}
{
fma.rn.f16x2 r209, r161, r200, r207;
}
{
mul.f16x2 r213, r161, r202;
}
{
fma.rn.f16x2 r216, r164, r200, r213;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r220, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r222, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r224, {low, high};
}
{
mul.f16x2 r225, r222, r224;
}
{
mul.f16x2 r228, r197, r220;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r231, {high, low};
}
{
fma.rn.f16x2 r233, r225, r231, r228;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r233;
mov.b32 r237, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r233;
mov.b32 r239, {high, high};
}
{
mul.f16x2 r241, r176, r239;
}
{
neg.f16x2 r244, r241;
}
{
fma.rn.f16x2 r246, r173, r237, r244;
}
{
mul.f16x2 r250, r173, r239;
}
{
fma.rn.f16x2 r253, r176, r237, r250;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r257, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r259, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r261, {low, high};
}
{
mul.f16x2 r262, r259, r261;
}
{
mul.f16x2 r265, r233, r257;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r233;
mov.b32 r268, {high, low};
}
{
fma.rn.f16x2 r270, r262, r268, r265;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r270;
mov.b32 r274, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r270;
mov.b32 r276, {high, high};
}
{
mul.f16x2 r278, r188, r276;
}
{
neg.f16x2 r281, r278;
}
{
fma.rn.f16x2 r283, r185, r274, r281;
}
{
mul.f16x2 r287, r185, r276;
}
{
fma.rn.f16x2 r290, r188, r274, r287;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r294, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r296, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r298, {low, high};
}
{
mul.f16x2 r299, r296, r298;
}
{
mul.f16x2 r302, r270, r294;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r270;
mov.b32 r305, {high, low};
}
{
fma.rn.f16x2 r307, r299, r305, r302;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r307;
mov.b32 r311, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r307;
mov.b32 r313, {high, high};
}
{
mul.f16x2 r315, r158, r313;
}
{
neg.f16x2 r318, r315;
}
{
fma.rn.f16x2 r320, r155, r311, r318;
}
{
mul.f16x2 r324, r155, r313;
}
{
fma.rn.f16x2 r327, r158, r311, r324;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r331, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r333, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r335, {low, high};
}
{
mul.f16x2 r336, r333, r335;
}
{
mul.f16x2 r339, r307, r331;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r307;
mov.b32 r342, {high, low};
}
{
fma.rn.f16x2 r344, r336, r342, r339;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r344;
mov.b32 r348, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r344;
mov.b32 r350, {high, high};
}
{
mul.f16x2 r352, r170, r350;
}
{
neg.f16x2 r355, r352;
}
{
fma.rn.f16x2 r357, r167, r348, r355;
}
{
mul.f16x2 r361, r167, r350;
}
{
fma.rn.f16x2 r364, r170, r348, r361;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r368, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r370, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r372, {low, high};
}
{
mul.f16x2 r373, r370, r372;
}
{
mul.f16x2 r376, r344, r368;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r344;
mov.b32 r379, {high, low};
}
{
fma.rn.f16x2 r381, r373, r379, r376;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r381;
mov.b32 r385, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r381;
mov.b32 r387, {high, high};
}
{
mul.f16x2 r389, r182, r387;
}
{
neg.f16x2 r392, r389;
}
{
fma.rn.f16x2 r394, r179, r385, r392;
}
{
mul.f16x2 r398, r179, r387;
}
{
fma.rn.f16x2 r401, r182, r385, r398;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r405, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r407, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r409, {low, high};
}
{
mul.f16x2 r410, r407, r409;
}
{
mul.f16x2 r413, r381, r405;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r381;
mov.b32 r416, {high, low};
}
{
fma.rn.f16x2 r418, r410, r416, r413;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r418;
mov.b32 r422, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r418;
mov.b32 r424, {high, high};
}
{
mul.f16x2 r426, r194, r424;
}
{
neg.f16x2 r429, r426;
}
{
fma.rn.f16x2 r431, r191, r422, r429;
}
{
mul.f16x2 r435, r191, r424;
}
{
fma.rn.f16x2 r438, r194, r422, r435;
}
barrier.sync 0;
and.b32 r516, r513, 32;
add.s32 r517, r515, r516;
st.shared.v4.f32 [r517], {r149, r209, r246, r283};
st.shared.v4.f32 [r517+16], {r320, r357, r394, r431};
barrier.sync 0;
mad.lo.s32 r518, r512, -28, r517;
ld.shared.u32 r460, [r518];
ld.shared.u32 r472, [r518+8];
ld.shared.u32 r484, [r518+16];
ld.shared.u32 r496, [r518+24];
ld.shared.u32 r461, [r518+32];
ld.shared.u32 r473, [r518+40];
ld.shared.u32 r485, [r518+48];
ld.shared.u32 r497, [r518+56];
barrier.sync 0;
st.shared.v4.f32 [r517], {r152, r216, r253, r290};
st.shared.v4.f32 [r517+16], {r327, r364, r401, r438};
barrier.sync 0;
ld.shared.u32 r463, [r518];
ld.shared.u32 r475, [r518+8];
ld.shared.u32 r487, [r518+16];
ld.shared.u32 r499, [r518+24];
ld.shared.u32 r464, [r518+32];
ld.shared.u32 r476, [r518+40];
ld.shared.u32 r488, [r518+48];
ld.shared.u32 r500, [r518+56];
{
add.f16x2 %0, r460, r461;
}
{
add.f16x2 %1, r463, r464;
}
{
sub.f16x2 %8, r460, r461;
}
{
sub.f16x2 %9, r463, r464;
}
{
add.f16x2 %2, r472, r473;
}
{
add.f16x2 %3, r475, r476;
}
{
sub.f16x2 %10, r472, r473;
}
{
sub.f16x2 %11, r475, r476;
}
{
add.f16x2 %4, r484, r485;
}
{
add.f16x2 %5, r487, r488;
}
{
sub.f16x2 %12, r484, r485;
}
{
sub.f16x2 %13, r487, r488;
}
{
add.f16x2 %6, r496, r497;
}
{
add.f16x2 %7, r499, r500;
}
{
sub.f16x2 %14, r496, r497;
}
{
sub.f16x2 %15, r499, r500;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<775, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<50>;
.reg .b32 r<519>;
.reg .b64 rd<2>;
mov.u32 r507, %tid.y;
shl.b32 r508, r507, 7;
mov.u32 r509, %16;
add.s32 r510, r509, r508;
mov.u32 r511, %tid.x;
{
add.f16x2 r1, %17, %25;
}
{
add.f16x2 r4, %18, %26;
}
{
sub.f16x2 r7, %17, %25;
}
{
sub.f16x2 r10, %18, %26;
}
{
add.f16x2 r13, %21, %29;
}
{
add.f16x2 r16, %22, %30;
}
{
sub.f16x2 r19, %21, %29;
}
{
sub.f16x2 r22, %22, %30;
}
{
neg.f16x2 r25, r19;
}
{
add.f16x2 r27, r1, r13;
}
{
add.f16x2 r30, r4, r16;
}
{
sub.f16x2 r33, r1, r13;
}
{
sub.f16x2 r36, r4, r16;
}
{
add.f16x2 r39, r7, r22;
}
{
add.f16x2 r42, r10, r25;
}
{
sub.f16x2 r45, r7, r22;
}
{
sub.f16x2 r48, r10, r25;
}
{
add.f16x2 r51, %19, %27;
}
{
add.f16x2 r54, %20, %28;
}
{
sub.f16x2 r57, %19, %27;
}
{
sub.f16x2 r60, %20, %28;
}
{
add.f16x2 r63, %23, %31;
}
{
add.f16x2 r66, %24, %32;
}
{
sub.f16x2 r69, %23, %31;
}
{
sub.f16x2 r72, %24, %32;
}
{
neg.f16x2 r75, r69;
}
{
add.f16x2 r77, r51, r63;
}
{
add.f16x2 r80, r54, r66;
}
{
sub.f16x2 r83, r51, r63;
}
{
sub.f16x2 r86, r54, r66;
}
{
add.f16x2 r89, r57, r72;
}
{
add.f16x2 r92, r60, r75;
}
{
sub.f16x2 r95, r57, r72;
}
{
sub.f16x2 r98, r60, r75;
}
mov.f32 f2, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f2;
cvt.rn.f16.f32 high, f2;
mov.b32 r101, {low, high};
}
mov.f32 f12, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f12;
cvt.rn.f16.f32 high, f12;
mov.b32 r102, {low, high};
}
mov.f32 f43, 0fBF800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f12;
cvt.rn.f16.f32 high, f12;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f12;
cvt.rn.f16.f32 high, f12;
mov.b32 r106, {low, high};
}
mov.f32 f44, 0f3F800000;
{
mul.f16x2 r115, r89, r101;
}
{
mul.f16x2 r118, r92, r102;
}
{
sub.f16x2 r121, r115, r118;
}
{
mul.f16x2 r124, r89, r102;
}
{
fma.rn.f16x2 r127, r92, r101, r124;
}
{
neg.f16x2 r131, r83;
}
{
mul.f16x2 r133, r95, r105;
}
{
mul.f16x2 r136, r98, r106;
}
{
sub.f16x2 r139, r133, r136;
}
{
mul.f16x2 r142, r95, r106;
}
{
fma.rn.f16x2 r145, r98, r105, r142;
}
{
add.f16x2 r149, r27, r77;
}
{
add.f16x2 r152, r30, r80;
}
{
sub.f16x2 r155, r27, r77;
}
{
sub.f16x2 r158, r30, r80;
}
{
add.f16x2 r161, r39, r121;
}
{
add.f16x2 r164, r42, r127;
}
{
sub.f16x2 r167, r39, r121;
}
{
sub.f16x2 r170, r42, r127;
}
{
add.f16x2 r173, r33, r86;
}
{
add.f16x2 r176, r36, r131;
}
{
sub.f16x2 r179, r33, r86;
}
{
sub.f16x2 r182, r36, r131;
}
{
add.f16x2 r185, r45, r139;
}
{
add.f16x2 r188, r48, r145;
}
{
sub.f16x2 r191, r45, r139;
}
{
sub.f16x2 r194, r48, r145;
}
and.b32 r512, r511, 1;
shl.b32 r513, r511, 6;
and.b32 r514, r513, -128;
add.s32 r515, r510, r514;
cvt.rn.f32.u32 f47, r512;
mul.f32 f48, f47, 0f3EC90FDB;
cos.approx.f32 f29, f48;
sin.approx.f32 f49, f48;
neg.f32 f30, f49;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f29;
cvt.rn.f16.f32 high, f30;
mov.b32 r197, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r200, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r202, {high, high};
}
{
mul.f16x2 r204, r164, r202;
}
{
neg.f16x2 r207, r204;
}
{
fma.rn.f16x2 r209, r161, r200, r207;
}
{
mul.f16x2 r213, r161, r202;
}
{
fma.rn.f16x2 r216, r164, r200, r213;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r220, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r222, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r224, {low, high};
}
{
mul.f16x2 r225, r222, r224;
}
{
mul.f16x2 r228, r197, r220;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r231, {high, low};
}
{
fma.rn.f16x2 r233, r225, r231, r228;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r233;
mov.b32 r237, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r233;
mov.b32 r239, {high, high};
}
{
mul.f16x2 r241, r176, r239;
}
{
neg.f16x2 r244, r241;
}
{
fma.rn.f16x2 r246, r173, r237, r244;
}
{
mul.f16x2 r250, r173, r239;
}
{
fma.rn.f16x2 r253, r176, r237, r250;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r257, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r259, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r261, {low, high};
}
{
mul.f16x2 r262, r259, r261;
}
{
mul.f16x2 r265, r233, r257;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r233;
mov.b32 r268, {high, low};
}
{
fma.rn.f16x2 r270, r262, r268, r265;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r270;
mov.b32 r274, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r270;
mov.b32 r276, {high, high};
}
{
mul.f16x2 r278, r188, r276;
}
{
neg.f16x2 r281, r278;
}
{
fma.rn.f16x2 r283, r185, r274, r281;
}
{
mul.f16x2 r287, r185, r276;
}
{
fma.rn.f16x2 r290, r188, r274, r287;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r294, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r296, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r298, {low, high};
}
{
mul.f16x2 r299, r296, r298;
}
{
mul.f16x2 r302, r270, r294;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r270;
mov.b32 r305, {high, low};
}
{
fma.rn.f16x2 r307, r299, r305, r302;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r307;
mov.b32 r311, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r307;
mov.b32 r313, {high, high};
}
{
mul.f16x2 r315, r158, r313;
}
{
neg.f16x2 r318, r315;
}
{
fma.rn.f16x2 r320, r155, r311, r318;
}
{
mul.f16x2 r324, r155, r313;
}
{
fma.rn.f16x2 r327, r158, r311, r324;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r331, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r333, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r335, {low, high};
}
{
mul.f16x2 r336, r333, r335;
}
{
mul.f16x2 r339, r307, r331;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r307;
mov.b32 r342, {high, low};
}
{
fma.rn.f16x2 r344, r336, r342, r339;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r344;
mov.b32 r348, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r344;
mov.b32 r350, {high, high};
}
{
mul.f16x2 r352, r170, r350;
}
{
neg.f16x2 r355, r352;
}
{
fma.rn.f16x2 r357, r167, r348, r355;
}
{
mul.f16x2 r361, r167, r350;
}
{
fma.rn.f16x2 r364, r170, r348, r361;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r368, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r370, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r372, {low, high};
}
{
mul.f16x2 r373, r370, r372;
}
{
mul.f16x2 r376, r344, r368;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r344;
mov.b32 r379, {high, low};
}
{
fma.rn.f16x2 r381, r373, r379, r376;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r381;
mov.b32 r385, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r381;
mov.b32 r387, {high, high};
}
{
mul.f16x2 r389, r182, r387;
}
{
neg.f16x2 r392, r389;
}
{
fma.rn.f16x2 r394, r179, r385, r392;
}
{
mul.f16x2 r398, r179, r387;
}
{
fma.rn.f16x2 r401, r182, r385, r398;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r405, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r407, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r409, {low, high};
}
{
mul.f16x2 r410, r407, r409;
}
{
mul.f16x2 r413, r381, r405;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r381;
mov.b32 r416, {high, low};
}
{
fma.rn.f16x2 r418, r410, r416, r413;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r418;
mov.b32 r422, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r418;
mov.b32 r424, {high, high};
}
{
mul.f16x2 r426, r194, r424;
}
{
neg.f16x2 r429, r426;
}
{
fma.rn.f16x2 r431, r191, r422, r429;
}
{
mul.f16x2 r435, r191, r424;
}
{
fma.rn.f16x2 r438, r194, r422, r435;
}
barrier.sync 0;
and.b32 r516, r513, 64;
add.s32 r517, r515, r516;
st.shared.v4.f32 [r517], {r149, r152, r209, r216};
st.shared.v4.f32 [r517+16], {r246, r253, r283, r290};
st.shared.v4.f32 [r517+32], {r320, r327, r357, r364};
st.shared.v4.f32 [r517+48], {r394, r401, r431, r438};
barrier.sync 0;
mad.lo.s32 r518, r512, -56, r517;
ld.shared.u32 r460, [r518];
ld.shared.u32 r463, [r518+4];
ld.shared.u32 r472, [r518+16];
ld.shared.u32 r475, [r518+20];
ld.shared.u32 r484, [r518+32];
ld.shared.u32 r487, [r518+36];
ld.shared.u32 r496, [r518+48];
ld.shared.u32 r499, [r518+52];
ld.shared.u32 r461, [r518+64];
ld.shared.u32 r464, [r518+68];
ld.shared.u32 r473, [r518+80];
ld.shared.u32 r476, [r518+84];
ld.shared.u32 r485, [r518+96];
ld.shared.u32 r488, [r518+100];
ld.shared.u32 r497, [r518+112];
ld.shared.u32 r500, [r518+116];
{
add.f16x2 %0, r460, r461;
}
{
add.f16x2 %1, r463, r464;
}
{
sub.f16x2 %8, r460, r461;
}
{
sub.f16x2 %9, r463, r464;
}
{
add.f16x2 %2, r472, r473;
}
{
add.f16x2 %3, r475, r476;
}
{
sub.f16x2 %10, r472, r473;
}
{
sub.f16x2 %11, r475, r476;
}
{
add.f16x2 %4, r484, r485;
}
{
add.f16x2 %5, r487, r488;
}
{
sub.f16x2 %12, r484, r485;
}
{
sub.f16x2 %13, r487, r488;
}
{
add.f16x2 %6, r496, r497;
}
{
add.f16x2 %7, r499, r500;
}
{
sub.f16x2 %14, r496, r497;
}
{
sub.f16x2 %15, r499, r500;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<776, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<28>;
.reg .b32 r<197>;
.reg .b64 rd<2>;
mov.u32 r169, %tid.y;
shl.b32 r170, r169, 6;
mov.u32 r171, %4;
add.s32 r172, r171, r170;
mov.u32 r173, %tid.x;
{
add.f16x2 r1, %5, %7;
}
{
add.f16x2 r4, %6, %8;
}
{
sub.f16x2 r7, %5, %7;
}
{
sub.f16x2 r10, %6, %8;
}
and.b32 r174, r173, 7;
shl.b32 r175, r173, 3;
and.b32 r176, r175, -64;
add.s32 r177, r172, r176;
cvt.rn.f32.u32 f19, r174;
mul.f32 f20, f19, 0f3EC90FDB;
cos.approx.f32 f1, f20;
sin.approx.f32 f21, f20;
neg.f32 f2, f21;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1;
cvt.rn.f16.f32 high, f2;
mov.b32 r13, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r13;
mov.b32 r16, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r13;
mov.b32 r18, {high, high};
}
{
mul.f16x2 r20, r10, r18;
}
{
neg.f16x2 r23, r20;
}
{
fma.rn.f16x2 r25, r7, r16, r23;
}
{
mul.f16x2 r29, r7, r18;
}
{
fma.rn.f16x2 r32, r10, r16, r29;
}
barrier.sync 0;
and.b32 r178, r175, 56;
add.s32 r179, r177, r178;
st.shared.v2.f32 [r179], {r1, r25};
barrier.sync 0;
shl.b32 r180, r173, 2;
and.b32 r181, r180, 28;
sub.s32 r182, r179, r181;
ld.shared.u32 r54, [r182];
ld.shared.u32 r55, [r182+32];
barrier.sync 0;
st.shared.v2.f32 [r179], {r4, r32};
barrier.sync 0;
ld.shared.u32 r57, [r182];
ld.shared.u32 r58, [r182+32];
{
add.f16x2 r53, r54, r55;
}
{
add.f16x2 r56, r57, r58;
}
{
sub.f16x2 r59, r54, r55;
}
{
sub.f16x2 r62, r57, r58;
}
bfe.u32 r183, r173, 1, 2;
and.b32 r184, r180, 4;
add.s32 r185, r177, r184;
cvt.rn.f32.u32 f22, r183;
mul.f32 f23, f22, 0f3F490FDB;
cos.approx.f32 f7, f23;
sin.approx.f32 f24, f23;
neg.f32 f8, f24;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f7;
cvt.rn.f16.f32 high, f8;
mov.b32 r65, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r65;
mov.b32 r68, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r65;
mov.b32 r70, {high, high};
}
{
mul.f16x2 r72, r62, r70;
}
{
neg.f16x2 r75, r72;
}
{
fma.rn.f16x2 r77, r59, r68, r75;
}
{
mul.f16x2 r81, r59, r70;
}
{
fma.rn.f16x2 r84, r62, r68, r81;
}
barrier.sync 0;
and.b32 r186, r175, 48;
add.s32 r187, r185, r186;
st.shared.u32 [r187], r53;
st.shared.u32 [r187+8], r77;
barrier.sync 0;
and.b32 r188, r180, 24;
sub.s32 r189, r187, r188;
ld.shared.u32 r106, [r189];
ld.shared.u32 r107, [r189+32];
barrier.sync 0;
st.shared.u32 [r187], r56;
st.shared.u32 [r187+8], r84;
barrier.sync 0;
ld.shared.u32 r109, [r189];
ld.shared.u32 r110, [r189+32];
{
add.f16x2 r105, r106, r107;
}
{
add.f16x2 r108, r109, r110;
}
{
sub.f16x2 r111, r106, r107;
}
{
sub.f16x2 r114, r109, r110;
}
bfe.u32 r190, r173, 2, 1;
and.b32 r191, r180, 12;
add.s32 r192, r177, r191;
cvt.rn.f32.u32 f25, r190;
mul.f32 f26, f25, 0f3FC90FDB;
cos.approx.f32 f13, f26;
sin.approx.f32 f27, f26;
neg.f32 f14, f27;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f13;
cvt.rn.f16.f32 high, f14;
mov.b32 r117, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r117;
mov.b32 r120, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r117;
mov.b32 r122, {high, high};
}
{
mul.f16x2 r124, r114, r122;
}
{
neg.f16x2 r127, r124;
}
{
fma.rn.f16x2 r129, r111, r120, r127;
}
{
mul.f16x2 r133, r111, r122;
}
{
fma.rn.f16x2 r136, r114, r120, r133;
}
barrier.sync 0;
and.b32 r193, r175, 32;
add.s32 r194, r192, r193;
st.shared.u32 [r194], r105;
st.shared.u32 [r194+16], r129;
barrier.sync 0;
and.b32 r195, r180, 16;
sub.s32 r196, r194, r195;
ld.shared.u32 r158, [r196];
ld.shared.u32 r159, [r196+32];
barrier.sync 0;
st.shared.u32 [r194], r108;
st.shared.u32 [r194+16], r136;
barrier.sync 0;
ld.shared.u32 r161, [r196];
ld.shared.u32 r162, [r196+32];
{
add.f16x2 %0, r158, r159;
}
{
add.f16x2 %1, r161, r162;
}
{
sub.f16x2 %2, r158, r159;
}
{
sub.f16x2 %3, r161, r162;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<777, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<28>;
.reg .b32 r<197>;
.reg .b64 rd<2>;
mov.u32 r169, %tid.y;
shl.b32 r170, r169, 7;
mov.u32 r171, %4;
add.s32 r172, r171, r170;
mov.u32 r173, %tid.x;
{
add.f16x2 r1, %5, %7;
}
{
add.f16x2 r4, %6, %8;
}
{
sub.f16x2 r7, %5, %7;
}
{
sub.f16x2 r10, %6, %8;
}
and.b32 r174, r173, 7;
shl.b32 r175, r173, 4;
and.b32 r176, r175, -128;
add.s32 r177, r172, r176;
cvt.rn.f32.u32 f19, r174;
mul.f32 f20, f19, 0f3EC90FDB;
cos.approx.f32 f1, f20;
sin.approx.f32 f21, f20;
neg.f32 f2, f21;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1;
cvt.rn.f16.f32 high, f2;
mov.b32 r13, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r13;
mov.b32 r16, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r13;
mov.b32 r18, {high, high};
}
{
mul.f16x2 r20, r10, r18;
}
{
neg.f16x2 r23, r20;
}
{
fma.rn.f16x2 r25, r7, r16, r23;
}
{
mul.f16x2 r29, r7, r18;
}
{
fma.rn.f16x2 r32, r10, r16, r29;
}
barrier.sync 0;
and.b32 r178, r175, 112;
add.s32 r179, r177, r178;
st.shared.v2.f32 [r179], {r1, r4};
st.shared.v2.f32 [r179+8], {r25, r32};
barrier.sync 0;
shl.b32 r180, r173, 3;
and.b32 r181, r180, 56;
sub.s32 r182, r179, r181;
ld.shared.u32 r54, [r182];
ld.shared.u32 r57, [r182+4];
ld.shared.u32 r55, [r182+64];
ld.shared.u32 r58, [r182+68];
{
add.f16x2 r53, r54, r55;
}
{
add.f16x2 r56, r57, r58;
}
{
sub.f16x2 r59, r54, r55;
}
{
sub.f16x2 r62, r57, r58;
}
bfe.u32 r183, r173, 1, 2;
cvt.rn.f32.u32 f22, r183;
mul.f32 f23, f22, 0f3F490FDB;
cos.approx.f32 f7, f23;
sin.approx.f32 f24, f23;
neg.f32 f8, f24;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f7;
cvt.rn.f16.f32 high, f8;
mov.b32 r65, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r65;
mov.b32 r68, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r65;
mov.b32 r70, {high, high};
}
{
mul.f16x2 r72, r62, r70;
}
{
neg.f16x2 r75, r72;
}
{
fma.rn.f16x2 r77, r59, r68, r75;
}
{
mul.f16x2 r81, r59, r70;
}
{
fma.rn.f16x2 r84, r62, r68, r81;
}
and.b32 r184, r180, 8;
add.s32 r185, r177, r184;
barrier.sync 0;
and.b32 r186, r175, 96;
add.s32 r187, r185, r186;
st.shared.u32 [r187], r53;
st.shared.u32 [r187+4], r56;
st.shared.u32 [r187+16], r77;
st.shared.u32 [r187+20], r84;
barrier.sync 0;
and.b32 r188, r180, 48;
sub.s32 r189, r187, r188;
ld.shared.u32 r106, [r189];
ld.shared.u32 r109, [r189+4];
ld.shared.u32 r107, [r189+64];
ld.shared.u32 r110, [r189+68];
{
add.f16x2 r105, r106, r107;
}
{
add.f16x2 r108, r109, r110;
}
{
sub.f16x2 r111, r106, r107;
}
{
sub.f16x2 r114, r109, r110;
}
bfe.u32 r190, r173, 2, 1;
cvt.rn.f32.u32 f25, r190;
mul.f32 f26, f25, 0f3FC90FDB;
cos.approx.f32 f13, f26;
sin.approx.f32 f27, f26;
neg.f32 f14, f27;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f13;
cvt.rn.f16.f32 high, f14;
mov.b32 r117, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r117;
mov.b32 r120, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r117;
mov.b32 r122, {high, high};
}
{
mul.f16x2 r124, r114, r122;
}
{
neg.f16x2 r127, r124;
}
{
fma.rn.f16x2 r129, r111, r120, r127;
}
{
mul.f16x2 r133, r111, r122;
}
{
fma.rn.f16x2 r136, r114, r120, r133;
}
and.b32 r191, r180, 24;
add.s32 r192, r177, r191;
barrier.sync 0;
and.b32 r193, r175, 64;
add.s32 r194, r192, r193;
st.shared.u32 [r194], r105;
st.shared.u32 [r194+4], r108;
st.shared.u32 [r194+32], r129;
st.shared.u32 [r194+36], r136;
barrier.sync 0;
and.b32 r195, r180, 32;
sub.s32 r196, r194, r195;
ld.shared.u32 r158, [r196];
ld.shared.u32 r161, [r196+4];
ld.shared.u32 r159, [r196+64];
ld.shared.u32 r162, [r196+68];
{
add.f16x2 %0, r158, r159;
}
{
add.f16x2 %1, r161, r162;
}
{
sub.f16x2 %2, r158, r159;
}
{
sub.f16x2 %3, r161, r162;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)));
};


#endif
