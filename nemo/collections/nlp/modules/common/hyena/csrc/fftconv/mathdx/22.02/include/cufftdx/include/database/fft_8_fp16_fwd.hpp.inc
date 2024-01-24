#ifndef CUFFTDX_FFT_8_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_8_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<766, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<15>;
.reg .b32 r<183>;
.reg .f64 fd<15>;
.reg .b64 rd<2>;
{
add.f16x2 r1, %29, %21;
}
{
add.f16x2 r4, %30, %22;
}
{
sub.f16x2 r7, %29, %21;
}
{
sub.f16x2 r10, %30, %22;
}
{
add.f16x2 r13, %17, %25;
}
{
add.f16x2 r16, %18, %26;
}
{
sub.f16x2 r19, %17, %25;
}
{
sub.f16x2 r22, %18, %26;
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
add.f16x2 r51, %31, %23;
}
{
add.f16x2 r54, %16, %24;
}
{
sub.f16x2 r57, %31, %23;
}
{
sub.f16x2 r60, %16, %24;
}
{
add.f16x2 r63, %19, %27;
}
{
add.f16x2 r66, %20, %28;
}
{
sub.f16x2 r69, %19, %27;
}
{
sub.f16x2 r72, %20, %28;
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
mov.f64 fd1, 0d3FE6A09E667F3BCD;
{
cvt.rn.f16.f64 rs1, fd1;
}
mov.f64 fd6, 0dBFE6A09E667F3BCD;
{
cvt.rn.f16.f64 rs2, fd6;
}
{
cvt.rn.f16.f64 rs5, fd6;
}
{
cvt.rn.f16.f64 rs6, fd6;
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
add.f16x2 %0, r27, r77;
}
{
add.f16x2 %1, r30, r80;
}
{
sub.f16x2 %8, r27, r77;
}
{
sub.f16x2 %9, r30, r80;
}
{
add.f16x2 %2, r39, r107;
}
{
add.f16x2 %3, r42, r113;
}
{
sub.f16x2 %10, r39, r107;
}
{
sub.f16x2 %11, r42, r113;
}
{
add.f16x2 %4, r33, r86;
}
{
add.f16x2 %5, r36, r117;
}
{
sub.f16x2 %12, r33, r86;
}
{
sub.f16x2 %13, r36, r117;
}
{
add.f16x2 %6, r45, r125;
}
{
add.f16x2 %7, r48, r131;
}
{
sub.f16x2 %14, r45, r125;
}
{
sub.f16x2 %15, r48, r131;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)): "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<767, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<14>;
.reg .b32 r<201>;
.reg .b64 rd<2>;
mov.u32 r189, %tid.y;
shl.b32 r190, r189, 5;
mov.u32 r191, %8;
add.s32 r192, r191, r190;
mov.u32 r193, %tid.x;
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
and.b32 r194, r193, 1;
shl.b32 r195, r193, 4;
and.b32 r196, r195, -32;
add.s32 r197, r192, r196;
cvt.rn.f32.u32 f11, r194;
mul.f32 f12, f11, 0f3F490FDB;
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
and.b32 r198, r195, 16;
add.s32 r199, r197, r198;
st.shared.v4.f32 [r199], {r27, r63, r100, r137};
barrier.sync 0;
mad.lo.s32 r200, r194, -12, r199;
ld.shared.u32 r166, [r200];
ld.shared.u32 r178, [r200+8];
ld.shared.u32 r167, [r200+16];
ld.shared.u32 r179, [r200+24];
barrier.sync 0;
st.shared.v4.f32 [r199], {r30, r70, r107, r144};
barrier.sync 0;
ld.shared.u32 r169, [r200];
ld.shared.u32 r181, [r200+8];
ld.shared.u32 r170, [r200+16];
ld.shared.u32 r182, [r200+24];
{
add.f16x2 %0, r166, r167;
}
{
add.f16x2 %1, r169, r170;
}
{
sub.f16x2 %4, r166, r167;
}
{
sub.f16x2 %5, r169, r170;
}
{
add.f16x2 %2, r178, r179;
}
{
add.f16x2 %3, r181, r182;
}
{
sub.f16x2 %6, r178, r179;
}
{
sub.f16x2 %7, r181, r182;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<768, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<14>;
.reg .b32 r<201>;
.reg .b64 rd<2>;
mov.u32 r189, %tid.y;
shl.b32 r190, r189, 6;
mov.u32 r191, %8;
add.s32 r192, r191, r190;
mov.u32 r193, %tid.x;
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
and.b32 r194, r193, 1;
shl.b32 r195, r193, 5;
and.b32 r196, r195, -64;
add.s32 r197, r192, r196;
cvt.rn.f32.u32 f11, r194;
mul.f32 f12, f11, 0f3F490FDB;
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
and.b32 r198, r195, 32;
add.s32 r199, r197, r198;
st.shared.v4.f32 [r199], {r27, r30, r63, r70};
st.shared.v4.f32 [r199+16], {r100, r107, r137, r144};
barrier.sync 0;
mad.lo.s32 r200, r194, -24, r199;
ld.shared.u32 r166, [r200];
ld.shared.u32 r169, [r200+4];
ld.shared.u32 r178, [r200+16];
ld.shared.u32 r181, [r200+20];
ld.shared.u32 r167, [r200+32];
ld.shared.u32 r170, [r200+36];
ld.shared.u32 r179, [r200+48];
ld.shared.u32 r182, [r200+52];
{
add.f16x2 %0, r166, r167;
}
{
add.f16x2 %1, r169, r170;
}
{
sub.f16x2 %4, r166, r167;
}
{
sub.f16x2 %5, r169, r170;
}
{
add.f16x2 %2, r178, r179;
}
{
add.f16x2 %3, r181, r182;
}
{
sub.f16x2 %6, r178, r179;
}
{
sub.f16x2 %7, r181, r182;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<769, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<19>;
.reg .b32 r<138>;
.reg .b64 rd<2>;
mov.u32 r117, %tid.y;
shl.b32 r118, r117, 5;
mov.u32 r119, %4;
add.s32 r120, r119, r118;
mov.u32 r121, %tid.x;
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
and.b32 r122, r121, 3;
shl.b32 r123, r121, 3;
and.b32 r124, r123, -32;
add.s32 r125, r120, r124;
cvt.rn.f32.u32 f13, r122;
mul.f32 f14, f13, 0f3F490FDB;
cos.approx.f32 f1, f14;
sin.approx.f32 f15, f14;
neg.f32 f2, f15;
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
and.b32 r126, r123, 24;
add.s32 r127, r125, r126;
st.shared.v2.f32 [r127], {r1, r25};
barrier.sync 0;
shl.b32 r128, r121, 2;
and.b32 r129, r128, 12;
sub.s32 r130, r127, r129;
ld.shared.u32 r54, [r130];
ld.shared.u32 r55, [r130+16];
barrier.sync 0;
st.shared.v2.f32 [r127], {r4, r32};
barrier.sync 0;
ld.shared.u32 r57, [r130];
ld.shared.u32 r58, [r130+16];
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
bfe.u32 r131, r121, 1, 1;
and.b32 r132, r128, 4;
add.s32 r133, r125, r132;
cvt.rn.f32.u32 f16, r131;
mul.f32 f17, f16, 0f3FC90FDB;
cos.approx.f32 f7, f17;
sin.approx.f32 f18, f17;
neg.f32 f8, f18;
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
and.b32 r134, r123, 16;
add.s32 r135, r133, r134;
st.shared.u32 [r135], r53;
st.shared.u32 [r135+8], r77;
barrier.sync 0;
and.b32 r136, r128, 8;
sub.s32 r137, r135, r136;
ld.shared.u32 r106, [r137];
ld.shared.u32 r107, [r137+16];
barrier.sync 0;
st.shared.u32 [r135], r56;
st.shared.u32 [r135+8], r84;
barrier.sync 0;
ld.shared.u32 r109, [r137];
ld.shared.u32 r110, [r137+16];
{
add.f16x2 %0, r106, r107;
}
{
add.f16x2 %1, r109, r110;
}
{
sub.f16x2 %2, r106, r107;
}
{
sub.f16x2 %3, r109, r110;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<770, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<19>;
.reg .b32 r<138>;
.reg .b64 rd<2>;
mov.u32 r117, %tid.y;
shl.b32 r118, r117, 6;
mov.u32 r119, %4;
add.s32 r120, r119, r118;
mov.u32 r121, %tid.x;
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
and.b32 r122, r121, 3;
shl.b32 r123, r121, 4;
and.b32 r124, r123, -64;
add.s32 r125, r120, r124;
cvt.rn.f32.u32 f13, r122;
mul.f32 f14, f13, 0f3F490FDB;
cos.approx.f32 f1, f14;
sin.approx.f32 f15, f14;
neg.f32 f2, f15;
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
and.b32 r126, r123, 48;
add.s32 r127, r125, r126;
st.shared.v2.f32 [r127], {r1, r4};
st.shared.v2.f32 [r127+8], {r25, r32};
barrier.sync 0;
shl.b32 r128, r121, 3;
and.b32 r129, r128, 24;
sub.s32 r130, r127, r129;
ld.shared.u32 r54, [r130];
ld.shared.u32 r57, [r130+4];
ld.shared.u32 r55, [r130+32];
ld.shared.u32 r58, [r130+36];
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
bfe.u32 r131, r121, 1, 1;
cvt.rn.f32.u32 f16, r131;
mul.f32 f17, f16, 0f3FC90FDB;
cos.approx.f32 f7, f17;
sin.approx.f32 f18, f17;
neg.f32 f8, f18;
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
and.b32 r132, r128, 8;
add.s32 r133, r125, r132;
barrier.sync 0;
and.b32 r134, r123, 32;
add.s32 r135, r133, r134;
st.shared.u32 [r135], r53;
st.shared.u32 [r135+4], r56;
st.shared.u32 [r135+16], r77;
st.shared.u32 [r135+20], r84;
barrier.sync 0;
and.b32 r136, r128, 16;
sub.s32 r137, r135, r136;
ld.shared.u32 r106, [r137];
ld.shared.u32 r109, [r137+4];
ld.shared.u32 r107, [r137+32];
ld.shared.u32 r110, [r137+36];
{
add.f16x2 %0, r106, r107;
}
{
add.f16x2 %1, r109, r110;
}
{
sub.f16x2 %2, r106, r107;
}
{
sub.f16x2 %3, r109, r110;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)));
};


#endif
